//! Spatial mapping: per-pixel fitting with rayon parallelization.
//!
//! Applies the single-spectrum fitting pipeline across all pixels in
//! a hyperspectral neutron imaging dataset to produce 2D composition maps.

use ndarray::{Array2, ArrayView3, s};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nereids_physics::resolution::build_resolution_plan;
use nereids_physics::transmission::{
    InstrumentParams, broadened_cross_sections, unbroadened_cross_sections,
};

use crate::error::PipelineError;
use crate::pipeline::SpectrumFitResult;

/// Result of spatial mapping over a 2D image.
///
/// **NaN-on-failure contract (issue #458 B1/B2):**
/// every per-pixel parameter map
/// (`density_maps`, `uncertainty_maps`, `chi_squared_map`,
/// `deviance_per_dof_map`, `temperature_map`,
/// `temperature_uncertainty_map`, `anorm_map`, `background_maps`,
/// `t0_us_map`, `l_scale_map`) contains `NaN` at every pixel where
/// `converged_map` is `false`.  The only map written unconditionally
/// is `converged_map` itself — it is how callers discover that a
/// pixel failed.  Callers rendering numeric values should gate on
/// `converged_map` (or check `value.is_finite()`) to avoid displaying
/// the placeholder `NaN`.
#[derive(Debug)]
pub struct SpatialResult {
    /// Fitted areal density maps, one per isotope.
    /// Each Array2 has shape (height, width).
    /// NaN at pixels where `converged_map` is `false`.
    pub density_maps: Vec<Array2<f64>>,
    /// Uncertainty maps, one per isotope.
    /// NaN at pixels where `converged_map` is `false`.
    pub uncertainty_maps: Vec<Array2<f64>>,
    /// Reduced chi-squared map.  For the counts-KL dispatch (joint-Poisson
    /// deviance per memo 35 §P1.2) this is back-compat-mirrored to
    /// `D/(n−k)`; the semantically-correct per-pixel value is also
    /// exposed as [`Self::deviance_per_dof_map`].
    /// NaN at pixels where `converged_map` is `false`.
    pub chi_squared_map: Array2<f64>,
    /// Per-pixel conditional binomial deviance `D/(n−k)` map.  `Some` when
    /// the effective per-pixel solver is the counts-KL dispatch
    /// (joint-Poisson); `None` for LM-only runs and transmission+PoissonKL
    /// where Pearson χ²/dof is the GOF.
    /// NaN at pixels where `converged_map` is `false`.
    pub deviance_per_dof_map: Option<Array2<f64>>,
    /// Convergence map (true = converged).
    pub converged_map: Array2<bool>,
    /// Fitted temperature map (K). `Some` when `config.fit_temperature()` is true.
    /// NaN at pixels where `converged_map` is `false`.
    pub temperature_map: Option<Array2<f64>>,
    /// Per-pixel temperature uncertainty map (K, 1-sigma).
    /// `Some` when `config.fit_temperature()` is true.
    /// Entries are NaN where uncertainty was unavailable for that pixel.
    pub temperature_uncertainty_map: Option<Array2<f64>>,
    /// Isotope labels captured at compute time, one per density map.
    /// Ensures display labels stay in sync with density data even if the
    /// user modifies the isotope list after fitting.
    pub isotope_labels: Vec<String>,
    /// Per-pixel normalization / signal-scale map (when background fitting is enabled).
    /// NaN at pixels where `converged_map` is `false`.
    pub anorm_map: Option<Array2<f64>>,
    /// Per-pixel background parameter maps.
    /// Transmission LM uses `[BackA, BackB, BackC]`.
    /// Counts KL background uses `[b0, b1, alpha_2]`.
    /// NaN at pixels where `converged_map` is `false`.
    pub background_maps: Option<[Array2<f64>; 3]>,
    /// Per-pixel fitted SAMMY TZERO offset (µs) map.
    /// `Some` when `config.fit_energy_scale` is true; `None` otherwise.
    /// NaN at pixels where `converged_map` is `false`.
    pub t0_us_map: Option<Array2<f64>>,
    /// Per-pixel fitted SAMMY TZERO flight-path scale factor.
    /// `Some` when `config.fit_energy_scale` is true; `None` otherwise.
    /// NaN at pixels where `converged_map` is `false`.
    pub l_scale_map: Option<Array2<f64>>,
    /// Number of pixels that converged.
    pub n_converged: usize,
    /// Total number of pixels fitted.
    pub n_total: usize,
    /// Number of pixels where the fitter returned an error (not just
    /// non-convergence — a hard failure like invalid parameters or NaN
    /// model output). These pixels have NaN density and false convergence.
    pub n_failed: usize,
}

// ── Phase 3: InputData3D + spatial_map_typed ─────────────────────────────

use crate::pipeline::{InputData, SolverConfig, UnifiedFitConfig, fit_spectrum_typed};

/// 3D input data for spatial mapping.
///
/// The outer dimension is energy (axis 0), inner dimensions are spatial (y, x).
/// The two variants correspond to [`InputData`] but carry 3D arrays.
#[derive(Debug)]
pub enum InputData3D<'a> {
    /// Pre-normalized transmission + uncertainty.
    Transmission {
        transmission: ArrayView3<'a, f64>,
        uncertainty: ArrayView3<'a, f64>,
    },
    /// Raw detector counts + open beam reference.
    Counts {
        sample_counts: ArrayView3<'a, f64>,
        open_beam_counts: ArrayView3<'a, f64>,
    },
    /// Raw detector counts with explicit nuisance spectra.
    CountsWithNuisance {
        sample_counts: ArrayView3<'a, f64>,
        flux: ArrayView3<'a, f64>,
        background: ArrayView3<'a, f64>,
    },
}

impl InputData3D<'_> {
    /// Shape of the data: (n_energies, height, width).
    pub(crate) fn shape(&self) -> (usize, usize, usize) {
        let s = match self {
            Self::Transmission { transmission, .. } => transmission.shape(),
            Self::Counts { sample_counts, .. } => sample_counts.shape(),
            Self::CountsWithNuisance { sample_counts, .. } => sample_counts.shape(),
        };
        (s[0], s[1], s[2])
    }

    /// `true` when the input is a counts variant (Counts or CountsWithNuisance)
    /// — i.e. the per-pixel dispatch goes through the counts-KL path
    /// (joint-Poisson deviance) rather than transmission.
    pub fn is_counts(&self) -> bool {
        matches!(self, Self::Counts { .. } | Self::CountsWithNuisance { .. })
    }
}

/// Spatial mapping using the typed input data API.
///
/// Dispatches per-pixel fitting based on the `InputData3D` variant:
/// - **Transmission**: per-pixel LM (or KL, opt-in) on transmission values.
/// - **Counts**: per-pixel counts-KL dispatch (joint-Poisson conditional
///   binomial deviance per memo 35 §P1) on the sample cube, paired
///   against the **spatially-averaged open-beam flux**.  See the inline
///   comment on `averaged_flux` for the rationale: this is a deliberate
///   bias-variance trade that reduces per-pixel OB shot-noise at the
///   cost of the exact per-pixel paired joint-Poisson observation model.
///   Callers needing the exact paired form should supply per-pixel
///   nuisance spectra via [`InputData3D::CountsWithNuisance`] instead.
/// - **CountsWithNuisance**: per-pixel counts-KL dispatch with the
///   caller-supplied per-pixel flux and background cubes.  No averaging.
///
/// Always returns [`SpatialResult`].
/// Apply the multi-pixel polish auto-disable rule (memo 38 §6).
///
/// For `n_pixels > 1`, return a config with `counts_enable_polish`
/// forced to `Some(false)` UNLESS the caller already set an explicit
/// override — in which case the caller's choice wins.  For `n_pixels
/// <= 1` or when the caller overrode, the config is returned as-is.
///
/// Extracted as a pure helper so the decision logic is directly
/// unit-testable without timing-based assertions in spatial tests.
fn apply_spatial_polish_default(config: UnifiedFitConfig, n_pixels: usize) -> UnifiedFitConfig {
    if n_pixels > 1 && config.counts_enable_polish().is_none() {
        config.with_counts_enable_polish(Some(false))
    } else {
        config
    }
}

pub fn spatial_map_typed(
    input: &InputData3D<'_>,
    config: &UnifiedFitConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
    progress: Option<&AtomicUsize>,
) -> Result<SpatialResult, PipelineError> {
    let (n_energies, height, width) = input.shape();
    // n_maps = number of density maps to return (one per group or per isotope).
    let n_maps = config.n_density_params();

    // Validate shapes
    if n_energies != config.energies().len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "input spectral axis ({n_energies}) != config.energies length ({})",
            config.energies().len(),
        )));
    }
    match input {
        InputData3D::Transmission {
            transmission,
            uncertainty,
        } => {
            if uncertainty.shape() != transmission.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "uncertainty shape {:?} != transmission shape {:?}",
                    uncertainty.shape(),
                    transmission.shape(),
                )));
            }
        }
        InputData3D::Counts {
            sample_counts,
            open_beam_counts,
        } => {
            if open_beam_counts.shape() != sample_counts.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "open_beam shape {:?} != sample shape {:?}",
                    open_beam_counts.shape(),
                    sample_counts.shape(),
                )));
            }
        }
        InputData3D::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        } => {
            if flux.shape() != sample_counts.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "flux shape {:?} != sample shape {:?}",
                    flux.shape(),
                    sample_counts.shape(),
                )));
            }
            if background.shape() != sample_counts.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "background shape {:?} != sample shape {:?}",
                    background.shape(),
                    sample_counts.shape(),
                )));
            }
        }
    }
    if let Some(dp) = dead_pixels
        && dp.shape() != [height, width]
    {
        return Err(PipelineError::ShapeMismatch(format!(
            "dead_pixels shape {:?} != spatial dimensions ({height}, {width})",
            dp.shape(),
        )));
    }

    // Reject known-broken configurations at entry.
    //
    // Issue #458 B3: per-pixel LM with `fit_energy_scale=True` on
    // counts data is numerically ill-conditioned.  On real VENUS Hf
    // 120 min, only ~8 % of pixels converged; `t0` drifts to the
    // ±10 µs bounds while `density` absorbs the compensating shift
    // (4-order-of-magnitude errors).  Reject upfront with a pointer
    // to the global-calibration workaround.
    //
    // Note: the LM-on-transmission path with `fit_energy_scale=True`
    // has the same structural issue, but is left unblocked here —
    // per-pixel transmission has higher SNR per bin (pre-normalised
    // by open-beam) and this combination is sometimes useful for
    // calibration crosschecks.  The config still produces NaN maps
    // for failed pixels thanks to B1 gating.
    if input.is_counts()
        && matches!(config.solver(), SolverConfig::LevenbergMarquardt(_))
        && config.fit_energy_scale()
    {
        return Err(PipelineError::InvalidParameter(
            "spatial_map_typed: solver='lm' + fit_energy_scale=true on counts input is \
             numerically unstable per-pixel (issue #458 B3). Recommended workaround: fit \
             TZERO once on the aggregated spectrum via fit_counts_spectrum_typed, then \
             build the corrected energy grid and pass it to spatial_map_typed with \
             fit_energy_scale=false. For counts data, solver='kl' (or 'auto') is robust \
             with per-pixel TZERO fitting."
                .into(),
        ));
    }

    // Issue #458 (Codex review): `fit_energy_scale` + `fit_temperature`
    // is not a supported combination — `EnergyScaleTransmissionModel`
    // and the temperature-fitting path are mutually exclusive at the
    // single-spectrum fitter (`pipeline.rs:830, 976, 1183`).  Without
    // this spatial-layer guard, every per-pixel call would error and
    // `spatial_map_typed` would report `n_failed == n_total` with an
    // all-NaN map — a silently-failed map is worse than a clear error.
    if config.fit_energy_scale() && config.fit_temperature() {
        return Err(PipelineError::InvalidParameter(
            "spatial_map_typed: fit_energy_scale=true and fit_temperature=true cannot \
             both be set — EnergyScaleTransmissionModel does not support temperature \
             fitting. Choose one: either calibrate TZERO with a fixed temperature, or \
             fit temperature on the nominal energy grid."
                .into(),
        ));
    }

    // Collect live pixel coordinates
    let mut pixel_coords: Vec<(usize, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let is_dead = dead_pixels.is_some_and(|m| m[[y, x]]);
            if !is_dead {
                pixel_coords.push((y, x));
            }
        }
    }

    let isotope_labels = config.isotope_names().to_vec();
    let has_background_outputs =
        config.transmission_background().is_some() || config.counts_background().is_some();

    // Whether the per-pixel dispatch routes through the counts-KL
    // (joint-Poisson) solver.  True iff the input is counts AND the
    // effective solver is either explicit `PoissonKL` or `Auto`
    // (Auto resolves to PoissonKL on counts input).  When false (LM
    // dispatch on counts, or any transmission input), per-pixel
    // SpectrumFitResult.deviance_per_dof is `None`, so the spatial
    // deviance_per_dof_map should also be `None` — otherwise GUI /
    // Python consumers using `is_some()` to label GOF as "D/dof"
    // would mislabel an all-NaN map.
    let dispatches_to_counts_kl =
        input.is_counts() && !matches!(config.solver(), SolverConfig::LevenbergMarquardt(_));

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(PipelineError::Cancelled);
    }
    if pixel_coords.is_empty() {
        // All pixels filtered out (typically by `dead_pixels` mask).  Per
        // the NaN-on-failure contract (issue #458 B1 + Copilot review),
        // every parameter map must be NaN at every pixel — including
        // density, which was previously initialised with zeros here.
        // `converged_map` is all `false`, which is the caller's signal
        // that no fits ran.
        return Ok(SpatialResult {
            density_maps: (0..n_maps)
                .map(|_| Array2::from_elem((height, width), f64::NAN))
                .collect(),
            uncertainty_maps: (0..n_maps)
                .map(|_| Array2::from_elem((height, width), f64::NAN))
                .collect(),
            chi_squared_map: Array2::from_elem((height, width), f64::NAN),
            deviance_per_dof_map: if dispatches_to_counts_kl {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            converged_map: Array2::from_elem((height, width), false),
            temperature_map: if config.fit_temperature() {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            temperature_uncertainty_map: if config.fit_temperature() {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            isotope_labels,
            anorm_map: if has_background_outputs {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            background_maps: if has_background_outputs {
                Some([
                    Array2::from_elem((height, width), f64::NAN),
                    Array2::from_elem((height, width), f64::NAN),
                    Array2::from_elem((height, width), f64::NAN),
                ])
            } else {
                None
            },
            t0_us_map: if config.fit_energy_scale() {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            l_scale_map: if config.fit_energy_scale() {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            n_converged: 0,
            n_total: 0,
            n_failed: 0,
        });
    }

    // Transpose data to (height, width, n_energies) for cache locality.
    let (data_a, data_b, data_c) = match input {
        InputData3D::Transmission {
            transmission,
            uncertainty,
        } => {
            let a = transmission
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let b = uncertainty
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            (a, b, None)
        }
        InputData3D::Counts {
            sample_counts,
            open_beam_counts,
        } => {
            let a = sample_counts
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let b = open_beam_counts
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            (a, b, None)
        }
        InputData3D::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        } => {
            let a = sample_counts
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let b = flux
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let c = background
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            (a, b, Some(c))
        }
    };

    // Precompute cross-sections once (shared across all pixels)
    let xs: Arc<Vec<Vec<f64>>> = match config.precomputed_cross_sections().cloned() {
        Some(cached) => cached,
        None => {
            let instrument = config.resolution().map(|r| InstrumentParams {
                resolution: r.clone(),
            });
            let xs_raw = broadened_cross_sections(
                config.energies(),
                config.resonance_data(),
                config.temperature_k(),
                instrument.as_ref(),
                cancel,
            )?;
            Arc::new(xs_raw)
        }
    };

    // When groups are active and temperature is NOT being fitted, collapse
    // per-member broadened XS into per-group σ_eff once here.  This avoids
    // redundant O(n_members × n_energies) collapsing inside
    // build_transmission_model on every per-pixel call.
    let xs = if !config.fit_temperature()
        && let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios)
        && xs.len() == di.len()
        && di.len() == dr.len()
    {
        let n_e = xs[0].len();
        let mut eff = vec![vec![0.0f64; n_e]; n_maps];
        for ((&idx, &ratio), member_xs) in di.iter().zip(dr.iter()).zip(xs.iter()) {
            for (j, &sigma) in member_xs.iter().enumerate() {
                eff[idx][j] += ratio * sigma;
            }
        }
        Arc::new(eff)
    } else {
        xs
    };

    // Build the resolution broadening plan once for the shared grid.
    //
    // The plan is valid for any per-pixel fit that applies resolution
    // on the (fixed) data energy grid — i.e. every spatial dispatch
    // EXCEPT the energy-scale (TZERO) path, where the grid changes
    // per (t0, l_scale) trial.  In that case the plan would always
    // miss so we skip the build; `EnergyScaleTransmissionModel` runs
    // the non-plan broadening path (see its `evaluate_at` comment).
    //
    // `build_resolution_plan` returns `None` for Gaussian resolution
    // (no worthwhile cache at this level) and `Some(plan)` for
    // tabulated kernels.  The error branch fires only on an unsorted
    // grid; when `precomputed_cross_sections` is already cached
    // (`config.precomputed_cross_sections().is_some()`), the
    // `broadened_cross_sections` call above is skipped, so the plan
    // build here is the *first* sort-check in that path.  Wrapping
    // the `ResolutionError` via `TransmissionError::from` keeps the
    // outward-facing error variant (`PipelineError::Transmission`)
    // consistent regardless of cache state.
    let resolution_plan: Option<Arc<nereids_physics::resolution::ResolutionPlan>> =
        if !config.fit_energy_scale() {
            match config.resolution() {
                // Route the unsorted-grid failure through
                // `TransmissionError::Resolution` so callers observe
                // the same error variant whether or not
                // `precomputed_cross_sections` is cached (the non-
                // cached path already surfaces this via
                // `broadened_cross_sections`).  Copilot #7.
                Some(res) => build_resolution_plan(config.energies(), res)
                    .map_err(|e| {
                        PipelineError::Transmission(
                            nereids_physics::transmission::TransmissionError::from(e),
                        )
                    })?
                    .map(Arc::new),
                None => None,
            }
        } else {
            None
        };

    // Build the sparse empirical cubature plan (epic #472) when the
    // fit is on the k ≥ 2 multi-isotope fixed-calibration path.  The
    // plan compiles the exact ResolutionMatrix from the resolution
    // plan above, then runs a per-row feasibility LP to collapse each
    // row to ≤ `S + k + 1` atoms.  One-shot cost per spatial_map
    // call, amortized across every pixel.  Falls back to `None` when:
    //   * no resolution plan (Gaussian or missing);
    //   * temperature or energy-scale fitting is active (σ / grid
    //     can change at runtime, invalidating atoms);
    //   * k == 1 (scalar fast-path is PR #475's scope);
    //   * xs is not pre-collapsed to per-group σ (cubature needs the
    //     final σ stack, not per-isotope σ × ratios).
    // Capture any caller-supplied cubature plan BEFORE the local
    // rebuild pathway — the `with_precomputed_cross_sections` setter
    // clears `precomputed_sparse_cubature_plan` as a defence against
    // stale-XS dispatch (Codex round-3 P3 on PR #480), so without
    // this snapshot a plan the caller attached via
    // `UnifiedFitConfig::with_precomputed_sparse_cubature_plan` would
    // be dropped and lost on every call.  Codex round-5 P3 on PR #480.
    let caller_cubature = config.precomputed_sparse_cubature_plan().cloned();
    let sparse_cubature_plan: Option<Arc<nereids_physics::surrogate::SparseEmpiricalCubaturePlan>> =
        if !config.fit_temperature()
            && !config.fit_energy_scale()
            && resolution_plan.is_some()
            && xs.len() >= 2
        {
            let plan = resolution_plan.as_deref().expect("guarded above");
            let matrix = plan.compile_to_matrix();
            let k = xs.len();
            let n_rows = matrix.len();
            // Flatten xs (Vec<Vec<f64>> of shape [k][n_rows]) into the
            // row-major `sigmas[j * n_rows + ℓ]` layout the cubature
            // builder expects.
            let mut sigmas_flat = Vec::with_capacity(k * n_rows);
            for row in xs.iter() {
                if row.len() != n_rows {
                    // Shape mismatch — surrender cubature, fall back.
                    sigmas_flat.clear();
                    break;
                }
                sigmas_flat.extend_from_slice(row);
            }
            if sigmas_flat.len() == k * n_rows {
                // Invariant pinning: the caller (this function's xs
                // assembly above) must have pre-aggregated σ by
                // isotope-group ratios so `xs[j]` already stores the
                // per-density-param effective σ that the cubature
                // builder needs.  If a future refactor inserts a
                // different σ mutation after this point, or the
                // collapse stops running first, the builder will
                // receive wrong σ and this assertion catches it in
                // debug builds.  Codex/Claude round-1 P2 on PR #480.
                debug_assert_eq!(
                    sigmas_flat.len(),
                    k * n_rows,
                    "cubature σ dimensions: expected {k} × {n_rows} = {}, got {}",
                    k * n_rows,
                    sigmas_flat.len(),
                );
                // Training box: 2 × the initial density — same convention
                // the codex04 reference uses.  Anchor at the midpoint
                // (0.5 × train_max).
                //
                // **Known limitation — deferred to PR #476**: this
                // policy doesn't cross-check the solver's actual
                // fit-bound constraints.  If `initial_densities`
                // is near zero (or far from where the solver
                // actually explores), the cubature is built for a
                // box that may not contain the fit trajectory, and
                // held-out forward accuracy degrades silently.
                // PR #476 (trust-region wrapper) owns box
                // invalidation / rebuild-on-escape and naturally
                // replaces this static policy; see Claude round-1
                // P2-b on PR #480.
                let train_max: Vec<f64> = config
                    .initial_densities()
                    .iter()
                    .map(|&n0| 2.0 * n0.max(1e-6))
                    .collect();
                let training =
                nereids_physics::surrogate::SparseEmpiricalCubaturePlan::default_training_points(
                    &train_max,
                );
                let anchor =
                nereids_physics::surrogate::SparseEmpiricalCubaturePlan::default_jacobian_anchor(
                    &train_max,
                );
                match nereids_physics::surrogate::SparseEmpiricalCubaturePlan::build(
                    &matrix,
                    &sigmas_flat,
                    k,
                    &training,
                    &anchor,
                ) {
                    Ok(plan) => {
                        // Record the training box on the plan so
                        // the per-pixel dispatch can safely refuse
                        // to fire when a fit iterate escapes the
                        // trained region — rather than silently
                        // running the surrogate out-of-domain.
                        // Codex round-4 P1 on PR #480.
                        Some(Arc::new(plan.with_density_box(train_max.clone())))
                    }
                    Err(e) => {
                        // Surface the build failure to stderr rather
                        // than silently swallow it — downstream fits
                        // continue via the exact path, but a missing
                        // cubature on a supposedly-eligible call is
                        // a debugging signal that deserves
                        // visibility.  Codex/Claude round-1 P2 on
                        // PR #480.
                        eprintln!(
                            "spatial_map_typed: sparse cubature build failed ({e}); \
                             falling back to exact ResolutionPlan path for this call",
                        );
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

    // Caller-fallback: if we didn't build a local plan (build
    // failed, or conditions weren't met), but the caller supplied
    // one that matches the current grid + k, reuse it.  This
    // saves the LP build cost on repeat spatial_map calls that
    // share the same `(grid, isotope_set, density_box)` and
    // preserves explicit `with_precomputed_sparse_cubature_plan`
    // attachments across the setter chain below.
    let sparse_cubature_plan = sparse_cubature_plan.or_else(|| {
        caller_cubature.filter(|p| {
            p.len() == xs.first().map(|r| r.len()).unwrap_or(0)
                && p.k() == xs.len()
                && p.target_energies() == config.energies()
        })
    });

    // Scalar (k = 1) surrogate plan — parallels the cubature build
    // but dispatches on `xs.len() == 1` (grouped fits / single-
    // isotope).  Reuses the compiled ResolutionMatrix from the
    // resolution plan.  Falls back silently on build failure; no
    // local plan means the exact `apply_resolution_with_plan` path
    // runs as today.  PR #475 benched both Lanczos σ-pushforward
    // Gauss quadrature and Chebyshev-in-density on real VENUS
    // (3471-bin production grid); Chebyshev won on both the
    // accuracy (≤ 2e-15 vs ≤ 4e-15) and wall-time axes.  Lanczos
    // code was deleted per the issue's "drop the loser" contract;
    // this build site now always returns the Chebyshev variant
    // via the public `ScalarSurrogatePlan` type alias
    // (= `ScalarChebyshevPlan`).
    let caller_scalar = config.precomputed_sparse_scalar_plan().cloned();
    let sparse_scalar_plan: Option<Arc<nereids_physics::surrogate::ScalarSurrogatePlan>> =
        if let Some(plan) = resolution_plan.as_ref()
            && !config.fit_temperature()
            && !config.fit_energy_scale()
            && xs.len() == 1
        {
            let sigma_row = &xs[0];
            // Chebyshev-in-density at M = 16 (PR #475 bench-off
            // winner).  Training box: 2 × the initial density;
            // Chebyshev's interpolant is exact at its nodes and
            // tight (≤ 1e-15 rel err) across a well-chosen box.
            //
            // If `n_max` is too wide for 16 nodes to resolve
            // `exp(-n · σ)` accurately (e.g. caller passes a
            // giant `initial_density` on a strong-peak σ), the
            // build's midpoint self-check fires and returns
            // `InsufficientAccuracyOnBox`; we log and fall back
            // to the exact path rather than install a plan that
            // could corrupt the fit.  Codex PR #475 round-2 P2.
            //
            // **Known limitation — deferred to PR #476** (mirrors
            // the cubature policy, see lines 578-588): if
            // `initial_densities[0]` is near zero the floor clamps
            // `n_max` to 2e-6, but the solver may explore well
            // past that.  The `scalar_density_within_box` guard in
            // `transmission_model.rs` catches this (strict
            // `n ≤ n_max` post-round-2 P1) and falls back to the
            // exact `ResolutionPlan` path, so the worst outcome
            // is lost speedup — never silent accuracy loss.
            // PR #476 (trust-region wrapper) owns
            // box-rebuild-on-escape and replaces this static
            // policy.  Claude round-1 P2-#5 on PR #475.
            const CHEBYSHEV_NODES: usize = 16;
            let n_max: f64 = 2.0 * config.initial_densities()[0].max(1e-6);
            match nereids_physics::surrogate::ScalarChebyshevPlan::build(
                Arc::clone(plan),
                sigma_row,
                n_max,
                CHEBYSHEV_NODES,
            ) {
                Ok(plan) => Some(Arc::new(plan)),
                Err(e) => {
                    eprintln!(
                        "spatial_map_typed: scalar Chebyshev build failed ({e}); \
                         falling back to exact ResolutionPlan path",
                    );
                    None
                }
            }
        } else {
            None
        };
    // Preserve caller-supplied scalar plan if local build didn't run.
    // Grid-identity check uses `to_bits()` per element (matches
    // `scalar_eligible` / `cubature_eligible`), not `==`, so `-0.0`
    // vs `+0.0` and NaN-bit mismatches can't silently slip through
    // the caller-fallback pre-filter.  Claude round-1 P2 on PR #475.
    let sparse_scalar_plan = sparse_scalar_plan.or_else(|| {
        caller_scalar.filter(|p| {
            let expected_len = xs.first().map(|r| r.len()).unwrap_or(0);
            if p.len() != expected_len {
                return false;
            }
            let plan_grid = p.target_energies();
            let cfg_grid = config.energies();
            if plan_grid.len() != cfg_grid.len() {
                return false;
            }
            plan_grid
                .iter()
                .zip(cfg_grid)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        })
    });

    // Precompute unbroadened (base) cross-sections for temperature fitting.
    // This avoids 74× overhead from redundant Reich-Moore evaluation per
    // KL iteration (112ms Reich-Moore vs 1.5ms Doppler rebroadening).
    let fast_config = if config.fit_temperature() {
        let base_xs: Vec<Vec<f64>> =
            unbroadened_cross_sections(config.energies(), config.resonance_data(), cancel)
                .map_err(PipelineError::Transmission)?;
        let mut cfg = config
            .clone()
            .with_precomputed_cross_sections(xs)
            .with_precomputed_base_xs(Arc::new(base_xs))
            .with_compute_covariance(true);
        if let Some(plan) = resolution_plan.clone() {
            cfg = cfg.with_precomputed_resolution_plan(plan);
        }
        // Cubature / scalar plans stay None on the temperature path
        // (builder guards above).  No-op here but explicit for
        // future readers.
        cfg
    } else {
        // For non-temperature path: xs is already collapsed to σ_eff when
        // groups are active, so clear group mapping to prevent double-collapse
        // inside build_transmission_model.
        let mut cfg = config.clone();
        if cfg.density_indices.is_some() {
            cfg.density_indices = None;
            cfg.density_ratios = None;
        }
        let mut cfg = cfg
            .with_precomputed_cross_sections(xs)
            .with_compute_covariance(true);
        if let Some(plan) = resolution_plan.clone() {
            cfg = cfg.with_precomputed_resolution_plan(plan);
        }
        if let Some(plan) = sparse_cubature_plan.clone() {
            cfg = cfg.with_precomputed_sparse_cubature_plan(plan);
        }
        if let Some(plan) = sparse_scalar_plan.clone() {
            cfg = cfg.with_precomputed_sparse_scalar_plan(plan);
        }
        cfg
    };

    // Auto-disable Nelder-Mead polish for multi-pixel counts-KL spatial
    // maps (memo 38 §6 recommendation).  Polish is a single-spectrum
    // research knob — on the VENUS Hf 120min aggregated fit it took
    // ~1 000 s; at 512 × 512 pixels that is untenable even with rayon.
    // Per-pixel fits also rarely hit the over-parameterized stall regime
    // polish targets.  The caller can force polish back on via
    // [`UnifiedFitConfig::with_counts_enable_polish(Some(true))`].
    let fast_config = apply_spatial_polish_default(fast_config, pixel_coords.len());

    // ── Modeling choice: spatially-averaged open-beam flux ──
    //
    // For `InputData3D::Counts`, every pixel's sample spectrum is paired
    // with the **same** open-beam spectrum: the spatial average across
    // all live pixels (`pixel_coords`).  This is INTENTIONAL, not a
    // per-pixel paired observation.  The rationale:
    //
    // 1. The open-beam counts `O(E)` are a *reference flux* that is
    //    approximately spatially uniform (the sample casts a shadow
    //    on an otherwise flat beam profile).  Averaging reduces the
    //    shot-noise contamination of the flux estimate by √n_pixels.
    // 2. In the joint-Poisson profile-deviance form
    //    (`λ̂_i = c·(O_i + S_i) / (1 + c·T_i)`), a noisy per-pixel
    //    `O_i` propagates directly into `λ̂_i`, which in turn inflates
    //    the deviance without improving density recovery.
    //
    // This is a bias-variance trade: we lose the exact per-pixel paired
    // likelihood structure in exchange for a tighter density-fidelity
    // variance across pixels.  Empirically (evidence/37-…json), this is
    // the right call for the VENUS-style "flat beam with a masking
    // sample" geometry.
    //
    // **If this isn't the right assumption for your data** — e.g. you
    // have a genuinely spatially-varying beam profile and pre-estimated
    // per-pixel flux + detector-background spectra — use
    // [`InputData3D::CountsWithNuisance`] instead.  That variant
    // bypasses the averaging and pairs each pixel's sample with the
    // caller-supplied per-pixel flux and bg spectra.
    //
    // TODO(future): expose a config flag to switch the counts dispatch
    // between "averaged OB" (current, stability-oriented) and "raw
    // per-pixel OB" (exact paired joint-Poisson) if a use case arises
    // where both options are needed at call sites.
    let averaged_flux: Option<Vec<f64>> = if matches!(input, InputData3D::Counts { .. }) {
        let n_e = data_b.shape()[2]; // data_b is transposed: (h, w, n_e)
        let mut flux = vec![0.0f64; n_e];
        let n_live = pixel_coords.len() as f64;
        if n_live > 0.0 {
            for &(y, x) in &pixel_coords {
                let ob_spectrum = data_b.slice(s![y, x, ..]);
                for (e, &v) in ob_spectrum.iter().enumerate() {
                    flux[e] += v;
                }
            }
            for v in &mut flux {
                *v /= n_live;
            }
        }
        Some(flux)
    } else {
        None
    };
    let background_zeros: Vec<f64> = if matches!(input, InputData3D::Counts { .. }) {
        vec![0.0f64; data_b.shape()[2]]
    } else {
        Vec::new()
    };

    // Fit all pixels in parallel
    let failed_count = AtomicUsize::new(0);
    let results: Vec<((usize, usize), SpectrumFitResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            let spectrum_a: Vec<f64> = data_a.slice(s![y, x, ..]).to_vec();

            // Build per-pixel 1D InputData
            let pixel_input = match input {
                InputData3D::Counts { .. } => {
                    let sample_clamped: Vec<f64> = spectrum_a.iter().map(|&v| v.max(0.0)).collect();
                    let ob_spectrum: Vec<f64> = data_b.slice(s![y, x, ..]).to_vec();

                    // Check effective solver: KL uses CountsWithNuisance
                    // (averaged flux), LM uses raw Counts (auto-converts to
                    // transmission inside fit_spectrum_typed).
                    let effective = fast_config.effective_solver(&InputData::Counts {
                        sample_counts: sample_clamped.clone(),
                        open_beam_counts: ob_spectrum.clone(),
                    });
                    match effective {
                        SolverConfig::PoissonKL(_) => InputData::CountsWithNuisance {
                            sample_counts: sample_clamped,
                            flux: averaged_flux.as_ref().unwrap().clone(),
                            // Raw-count spatial path currently assumes zero
                            // detector background unless the caller provides
                            // explicit nuisance spectra.
                            background: background_zeros.clone(),
                        },
                        _ => InputData::Counts {
                            sample_counts: sample_clamped,
                            open_beam_counts: ob_spectrum,
                        },
                    }
                }
                InputData3D::CountsWithNuisance { .. } => InputData::CountsWithNuisance {
                    sample_counts: spectrum_a.iter().map(|&v| v.max(0.0)).collect(),
                    flux: data_b.slice(s![y, x, ..]).to_vec(),
                    background: data_c
                        .as_ref()
                        .expect("CountsWithNuisance requires background cube")
                        .slice(s![y, x, ..])
                        .to_vec(),
                },
                InputData3D::Transmission { .. } => {
                    let spectrum_b: Vec<f64> = data_b
                        .slice(s![y, x, ..])
                        .iter()
                        .map(|&v| v.max(1e-10))
                        .collect();
                    InputData::Transmission {
                        transmission: spectrum_a,
                        uncertainty: spectrum_b,
                    }
                }
            };

            let out = match fit_spectrum_typed(&pixel_input, &fast_config) {
                Ok(result) => Some(((y, x), result)),
                Err(_) => {
                    failed_count.fetch_add(1, Ordering::Relaxed);
                    None
                }
            };
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            out
        })
        .collect();

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) && results.is_empty() {
        return Err(PipelineError::Cancelled);
    }

    // Assemble output maps
    let mut density_maps: Vec<Array2<f64>> = (0..n_maps)
        .map(|_| Array2::from_elem((height, width), f64::NAN))
        .collect();
    let mut uncertainty_maps: Vec<Array2<f64>> = (0..n_maps)
        .map(|_| Array2::from_elem((height, width), f64::NAN))
        .collect();
    let mut chi_squared_map = Array2::from_elem((height, width), f64::NAN);
    let mut deviance_per_dof_map: Option<Array2<f64>> = if dispatches_to_counts_kl {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut converged_map = Array2::from_elem((height, width), false);
    let mut anorm_map: Option<Array2<f64>> = if has_background_outputs {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut background_maps: Option<[Array2<f64>; 3]> = if has_background_outputs {
        Some([
            Array2::from_elem((height, width), f64::NAN),
            Array2::from_elem((height, width), f64::NAN),
            Array2::from_elem((height, width), f64::NAN),
        ])
    } else {
        None
    };
    let mut t0_us_map: Option<Array2<f64>> = if config.fit_energy_scale() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut l_scale_map: Option<Array2<f64>> = if config.fit_energy_scale() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut n_converged = 0;
    let mut temperature_map: Option<Array2<f64>> = if config.fit_temperature() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut temperature_uncertainty_map: Option<Array2<f64>> = if config.fit_temperature() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };

    // Aggregate per-pixel fit results into 2-D maps.
    //
    // **Only the `converged_map` entry is written unconditionally.**
    // All other per-pixel parameter writes are gated on
    // `result.converged`, so un-converged pixels keep their initial
    // `NaN` value from the allocation above.
    //
    // Rationale (issue #458 B1/B2): the LM solver's
    // `LAMBDA_BREAKOUT` and stagnation paths restore `params` to the
    // last-accepted trial step and return `converged = false`.  That
    // "last accepted" state can be arbitrarily far from optimal if
    // LM walked astray before getting stuck — e.g., on real VENUS
    // per-pixel counts with TZERO enabled, LM pins `t0` at the
    // ±10 µs bound and lets `density` absorb the drift, producing
    // densities 4 orders of magnitude off.  Writing those garbage
    // values into the density/t0/L/background maps masked an 8 %
    // convergence rate as "map of mostly-sensible numbers with a
    // few outliers" rather than "map of NaN holes with a few fits".
    //
    // NaN-on-failure is also the convention asserted by
    // `test_spatial_map_failed_pixels_remain_nan`; this block makes
    // it hold for *every* non-converged pixel, not only the hard
    // failure path.
    for ((y, x), result) in &results {
        // Always record the convergence flag — this is how callers
        // discover that a pixel failed.
        converged_map[[*y, *x]] = result.converged;
        if !result.converged {
            continue;
        }

        n_converged += 1;

        for i in 0..n_maps {
            density_maps[i][[*y, *x]] = result.densities[i];
            if let Some(ref unc) = result.uncertainties {
                uncertainty_maps[i][[*y, *x]] = unc[i];
            }
        }
        chi_squared_map[[*y, *x]] = result.reduced_chi_squared;
        if let (Some(dpd), Some(v)) = (&mut deviance_per_dof_map, result.deviance_per_dof) {
            dpd[[*y, *x]] = v;
        }
        if let (Some(t_map), Some(t)) = (&mut temperature_map, result.temperature_k) {
            t_map[[*y, *x]] = t;
        }
        if let (Some(tu_map), Some(tu)) =
            (&mut temperature_uncertainty_map, result.temperature_k_unc)
        {
            tu_map[[*y, *x]] = tu;
        }
        if let Some(ref mut a_map) = anorm_map {
            a_map[[*y, *x]] = result.anorm;
        }
        if let Some(ref mut bg_maps) = background_maps {
            bg_maps[0][[*y, *x]] = result.background[0];
            bg_maps[1][[*y, *x]] = result.background[1];
            bg_maps[2][[*y, *x]] = result.background[2];
        }
        if let (Some(map), Some(v)) = (&mut t0_us_map, result.t0_us) {
            map[[*y, *x]] = v;
        }
        if let (Some(map), Some(v)) = (&mut l_scale_map, result.l_scale) {
            map[[*y, *x]] = v;
        }
    }

    Ok(SpatialResult {
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        deviance_per_dof_map,
        converged_map,
        temperature_map,
        temperature_uncertainty_map,
        isotope_labels,
        anorm_map,
        background_maps,
        t0_us_map,
        l_scale_map,
        n_converged,
        n_total: pixel_coords.len(),
        n_failed: failed_count.load(Ordering::Relaxed),
    })
}

// ── End Phase 3 ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use nereids_fitting::lm::{FitModel, LmConfig};
    use nereids_fitting::poisson::PoissonConfig;
    use nereids_fitting::transmission_model::PrecomputedTransmissionModel;

    use crate::pipeline::{SolverConfig, UnifiedFitConfig};
    use crate::test_helpers::{synthetic_single_resonance, u238_single_resonance};

    /// Build a 4x4 synthetic transmission stack from known density.
    fn synthetic_4x4_transmission(
        res_data: &nereids_endf::resonance::ResonanceData,
        true_density: f64,
        energies: &[f64],
    ) -> (Array3<f64>, Array3<f64>) {
        let n_e = energies.len();
        let xs = nereids_physics::transmission::broadened_cross_sections(
            energies,
            std::slice::from_ref(res_data),
            0.0,
            None,
            None,
        )
        .unwrap();
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs),
            density_indices: Arc::new(vec![0]),
            energies: None,
            instrument: None,
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
        };
        let t_1d = model.evaluate(&[true_density]).unwrap();
        let sigma_1d: Vec<f64> = t_1d.iter().map(|&v| 0.01 * v.max(0.01)).collect();

        // Fill a 4x4 grid with the same spectrum
        let mut t_3d = Array3::zeros((n_e, 4, 4));
        let mut u_3d = Array3::zeros((n_e, 4, 4));
        for y in 0..4 {
            for x in 0..4 {
                for (i, (&t, &s)) in t_1d.iter().zip(sigma_1d.iter()).enumerate() {
                    t_3d[[i, y, x]] = t;
                    u_3d[[i, y, x]] = s;
                }
            }
        }
        (t_3d, u_3d)
    }

    /// Build a 4x4 synthetic counts stack from known density.
    fn synthetic_4x4_counts(
        res_data: &nereids_endf::resonance::ResonanceData,
        true_density: f64,
        energies: &[f64],
        i0: f64,
    ) -> (Array3<f64>, Array3<f64>) {
        let (t_3d, _) = synthetic_4x4_transmission(res_data, true_density, energies);
        let n_e = energies.len();
        let mut sample = Array3::zeros((n_e, 4, 4));
        let mut ob = Array3::zeros((n_e, 4, 4));
        for y in 0..4 {
            for x in 0..4 {
                for i in 0..n_e {
                    ob[[i, y, x]] = i0;
                    sample[[i, y, x]] = (t_3d[[i, y, x]] * i0).round().max(0.0);
                }
            }
        }
        (sample, ob)
    }

    #[test]
    fn test_spatial_map_typed_transmission_lm() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&data, true_density, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_total, 16);
        assert!(result.n_converged >= 14, "Most pixels should converge");

        // Check mean density of converged pixels
        let d = &result.density_maps[0];
        let conv = &result.converged_map;
        let mean: f64 = d
            .iter()
            .zip(conv.iter())
            .filter(|(_, c)| **c)
            .map(|(d, _)| *d)
            .sum::<f64>()
            / result.n_converged as f64;
        assert!(
            (mean - true_density).abs() / true_density < 0.05,
            "mean density: {mean}, true: {true_density}"
        );
    }

    #[test]
    fn test_spatial_map_typed_counts_kl() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (sample, ob) = synthetic_4x4_counts(&data, true_density, &energies, 1000.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_total, 16);
        assert!(
            result.n_converged >= 14,
            "Most pixels should converge with KL"
        );

        let d = &result.density_maps[0];
        let conv = &result.converged_map;
        let mean: f64 = d
            .iter()
            .zip(conv.iter())
            .filter(|(_, c)| **c)
            .map(|(d, _)| *d)
            .sum::<f64>()
            / result.n_converged.max(1) as f64;
        assert!(
            (mean - true_density).abs() / true_density < 0.10,
            "KL mean density: {mean}, true: {true_density}"
        );
    }

    /// Build a minimal synthetic tabulated resolution kernel.  Two
    /// reference energies × a 5-point triangular offset-weight block
    /// is enough to exercise the plan build + apply hot path without
    /// pulling in the external VENUS resolution file.
    ///
    /// The kernel width is deliberately small (sub-microsecond) so
    /// broadening perturbs a non-broadened synthetic spectrum only
    /// slightly — keeps the spatial fit in its convergence basin
    /// without building a full R⊗T forward pass into the test
    /// fixture.
    fn synthetic_tabulated_text() -> String {
        // File format (parsed by TabulatedResolution::from_text):
        //   header line
        //   separator line
        //   for each block: energy marker line, then N offset/weight
        //   pairs, then a blank line between blocks.
        "header\n---\n\
         5.0 0.0\n\
         -0.01 0.0\n\
         -0.005 0.5\n\
         0.0 1.0\n\
         0.005 0.5\n\
         0.01 0.0\n\
         \n\
         200.0 0.0\n\
         -0.02 0.0\n\
         -0.01 0.5\n\
         0.0 1.0\n\
         0.01 0.5\n\
         0.02 0.0\n"
            .to_string()
    }

    /// Gate: end-to-end smoke + determinism test for the per-pixel
    /// spatial path with an attached resolution plan (tabulated
    /// kernel).  Asserts that `spatial_map_typed` runs to
    /// completion, most pixels converge, the recovered mean density
    /// is sensible on the synthetic fixture, and every converged
    /// pixel in the 4×4 crop produces a bit-identical density (no
    /// plan-cache state leaks across the rayon fanout).
    ///
    /// Exact `apply_resolution` / `apply_resolution_with_plan`
    /// equivalence is covered bit-for-bit by the unit tests in
    /// `resolution.rs`; this spatial test only confirms that plan
    /// attachment does not disturb the higher-level dispatch.
    #[test]
    fn test_spatial_map_typed_with_resolution_plan_converges_and_is_deterministic() {
        use nereids_physics::resolution::{ResolutionFunction, TabulatedResolution};

        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&data, true_density, &energies);

        let tab = TabulatedResolution::from_text(&synthetic_tabulated_text(), 25.0).unwrap();
        let resolution = ResolutionFunction::Tabulated(Arc::new(tab));

        let config = UnifiedFitConfig::new(
            energies.clone(),
            vec![data.clone()],
            vec!["U-238".into()],
            0.0,
            Some(resolution),
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let result_with_plan = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result_with_plan.n_total, 16);
        assert!(
            result_with_plan.n_converged >= 14,
            "plan path: {} / 16 pixels converged",
            result_with_plan.n_converged,
        );

        let d = &result_with_plan.density_maps[0];
        let conv = &result_with_plan.converged_map;
        let mean: f64 = d
            .iter()
            .zip(conv.iter())
            .filter(|(_, c)| **c)
            .map(|(d, _)| *d)
            .sum::<f64>()
            / result_with_plan.n_converged.max(1) as f64;
        assert!(
            (mean - true_density).abs() / true_density < 0.10,
            "mean density with plan: {mean}, true: {true_density}"
        );

        // Every converged pixel in the 4x4 crop shares the identical
        // input spectrum, so every density-map entry must be bit-
        // equal to every other converged entry.  This catches any
        // plan-cache corruption that would leak pixel-specific state
        // across the rayon fanout.
        let reference = d
            .iter()
            .zip(conv.iter())
            .find(|(_, c)| **c)
            .map(|(d, _)| *d)
            .expect("at least one pixel converged");
        for (&cell, &c) in d.iter().zip(conv.iter()) {
            if c {
                assert_eq!(
                    cell.to_bits(),
                    reference.to_bits(),
                    "plan cache leaked pixel-specific state: density cell {cell} != reference {reference}"
                );
            }
        }
    }

    #[test]
    fn test_spatial_map_typed_counts_kl_low_counts() {
        // I0=10: the regime where KL excels
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (sample, ob) = synthetic_4x4_counts(&data, true_density, &energies, 10.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap(); // Auto solver → KL for counts

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_total, 16);
        // At I0=10, KL should still converge for most pixels
        assert!(
            result.n_converged >= 10,
            "KL at I0=10: only {}/{} converged",
            result.n_converged,
            result.n_total
        );
    }

    // Removed as part of the counts-KL collapse (Phase 0):
    //   test_spatial_map_typed_counts_with_nuisance_surfaces_background_maps
    // Tested fit_alpha_1 / fit_alpha_2 nuisance fitting on a 4×4 spatial
    // grid, which is no longer supported on the counts-KL dispatch (the
    // joint-Poisson profile λ̂ absorbs alpha_1 and alpha_2 / B_det is
    // P3.2-deferred; memo 35 §P3).  The SAMMY-style A_n + B_A/B/C wiring
    // on counts input is covered at pipeline scale by
    // `test_joint_poisson_with_transmission_background` (pipeline.rs);
    // a dedicated 3D-grid counterpart is not currently a test invariant
    // (spatial_map_typed is a thin per-pixel dispatcher over
    // fit_spectrum_typed, which is already covered).

    #[test]
    fn test_spatial_map_typed_dead_pixels() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap();

        // Mask half the pixels as dead
        let mut dead = Array2::from_elem((4, 4), false);
        for y in 0..2 {
            for x in 0..4 {
                dead[[y, x]] = true;
            }
        }

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let result = spatial_map_typed(&input, &config, Some(&dead), None, None).unwrap();
        assert_eq!(result.n_total, 8, "Only 8 live pixels");
    }

    #[test]
    fn test_spatial_map_failed_pixels_remain_nan() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&data, true_density, &energies, 1000.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_counts_background(crate::pipeline::CountsBackgroundConfig {
            alpha_1_init: 1.0,
            alpha_2_init: 1.0,
            fit_alpha_1: false,
            fit_alpha_2: true,
            c: 1.0,
        });

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_converged, 0);
        assert!(
            result.density_maps[0].iter().all(|v| v.is_nan()),
            "failed pixels must remain NaN rather than looking like zero-density fits"
        );
        assert!(
            result.chi_squared_map.iter().all(|v| v.is_nan()),
            "failed pixels must retain NaN chi-squared"
        );
    }

    /// Spatial map with isotope groups: 2 isotopes in 1 group on a 2×2 grid.
    /// Verifies group-level density recovery and that only 1 density map is returned.
    #[test]
    fn test_spatial_map_grouped() {
        let rd1 = synthetic_single_resonance(92, 235, 233.025, 5.0);
        let rd2 = synthetic_single_resonance(92, 238, 236.006, 7.0);

        let iso1 = nereids_core::types::Isotope::new(92, 235).unwrap();
        let iso2 = nereids_core::types::Isotope::new(92, 238).unwrap();
        let group = nereids_core::types::IsotopeGroup::custom(
            "U (60/40)".into(),
            vec![(iso1, 0.6), (iso2, 0.4)],
        )
        .unwrap();

        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let n_e = energies.len();
        let true_density = 0.0005;

        // Generate synthetic transmission for the group
        let sample = nereids_physics::transmission::SampleParams::new(
            0.0,
            vec![
                (rd1.clone(), true_density * 0.6),
                (rd2.clone(), true_density * 0.4),
            ],
        )
        .unwrap();
        let t_1d = nereids_physics::transmission::forward_model(&energies, &sample, None).unwrap();
        let s_1d: Vec<f64> = t_1d.iter().map(|&v| 0.01 * v.max(0.01)).collect();

        // Fill 2×2 grid
        let mut t_3d = Array3::zeros((n_e, 2, 2));
        let mut u_3d = Array3::zeros((n_e, 2, 2));
        for y in 0..2 {
            for x in 0..2 {
                for (i, (&t, &s)) in t_1d.iter().zip(s_1d.iter()).enumerate() {
                    t_3d[[i, y, x]] = t;
                    u_3d[[i, y, x]] = s;
                }
            }
        }

        let config = UnifiedFitConfig::new(
            energies,
            vec![rd1.clone()],
            vec!["placeholder".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_groups(&[(&group, &[rd1, rd2])], vec![0.001])
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();

        // Should have 1 density map (1 group), not 2
        assert_eq!(
            result.density_maps.len(),
            1,
            "should have 1 group density map"
        );
        assert_eq!(result.isotope_labels, vec!["U (60/40)"]);
        assert_eq!(result.n_total, 4);

        // All pixels should recover true density within 5%
        for y in 0..2 {
            for x in 0..2 {
                let fitted = result.density_maps[0][[y, x]];
                let rel_error = (fitted - true_density).abs() / true_density;
                assert!(
                    rel_error < 0.05,
                    "pixel ({y},{x}): fitted={fitted}, true={true_density}, rel_error={rel_error}"
                );
            }
        }
    }

    // ── Phase 3: Spatial uncertainty propagation tests ──────────────────────

    /// Spatial LM transmission fit populates density uncertainty maps.
    #[test]
    fn test_spatial_lm_populates_density_uncertainty() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        // Add deterministic pseudo-noise so reduced chi-squared > 0
        // (a perfect fit gives chi2r=0, zeroing covariance).
        for y in 0..4 {
            for x in 0..4 {
                for e in 0..energies.len() {
                    let noise = 0.002 * ((e * 7 + y * 13 + x * 29) % 17) as f64 / 17.0 - 0.001;
                    t_3d[[e, y, x]] = (t_3d[[e, y, x]] + noise).max(0.001);
                }
            }
        }
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(result.n_converged > 0, "some pixels should converge");
        // Uncertainty maps should have finite positive values for converged pixels.
        let unc_map = &result.uncertainty_maps[0];
        let conv_map = &result.converged_map;
        let mut n_finite = 0;
        for y in 0..4 {
            for x in 0..4 {
                if conv_map[[y, x]] {
                    let u = unc_map[[y, x]];
                    assert!(
                        u.is_finite() && u > 0.0,
                        "LM density unc at ({y},{x}) should be finite+positive, got {u}"
                    );
                    n_finite += 1;
                }
            }
        }
        assert!(
            n_finite > 0,
            "at least one converged pixel should have finite unc"
        );
    }

    /// Spatial KL counts fit populates density uncertainty maps.
    #[test]
    fn test_spatial_kl_populates_density_uncertainty() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, _) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        // Convert to counts: OB=1000, sample = OB * T
        let ob_3d = Array3::from_elem(t_3d.raw_dim(), 1000.0);
        let sample_3d = &t_3d * &ob_3d;
        let data = InputData3D::Counts {
            sample_counts: sample_3d.view(),
            open_beam_counts: ob_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(result.n_converged > 0);
        let unc_map = &result.uncertainty_maps[0];
        let conv_map = &result.converged_map;
        let mut n_finite = 0;
        for y in 0..4 {
            for x in 0..4 {
                if conv_map[[y, x]] {
                    let u = unc_map[[y, x]];
                    assert!(
                        u.is_finite() && u > 0.0,
                        "KL density unc at ({y},{x}) should be finite+positive, got {u}"
                    );
                    n_finite += 1;
                }
            }
        }
        assert!(n_finite > 0);
    }

    /// Spatial temperature-fitting populates temperature_uncertainty_map.
    #[test]
    fn test_spatial_temperature_uncertainty_map() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.05).collect();
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        // Add pseudo-noise for nonzero chi2r.
        for y in 0..4 {
            for x in 0..4 {
                for e in 0..energies.len() {
                    let noise = 0.002 * ((e * 7 + y * 13 + x * 29) % 17) as f64 / 17.0 - 0.001;
                    t_3d[[e, y, x]] = (t_3d[[e, y, x]] + noise).max(0.001);
                }
            }
        }
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_temperature(true);

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(result.temperature_map.is_some());
        let tu_map = result
            .temperature_uncertainty_map
            .as_ref()
            .expect("temperature_uncertainty_map should be Some when fit_temperature=true");
        assert_eq!(tu_map.shape(), [4, 4]);
        // At least some converged pixels should have finite temperature uncertainty.
        let mut n_finite = 0;
        for y in 0..4 {
            for x in 0..4 {
                if result.converged_map[[y, x]] {
                    let tu = tu_map[[y, x]];
                    if tu.is_finite() && tu > 0.0 {
                        n_finite += 1;
                    }
                }
            }
        }
        assert!(
            n_finite > 0,
            "at least one converged pixel should have finite temperature uncertainty"
        );
    }

    /// Unconverged pixels remain NaN across **every** output map
    /// (density, uncertainty, chi², t0, l_scale, temperature, anorm,
    /// background) — not just uncertainty.  Issue #458 B1/B2:
    /// previously, failed LM fits that restored to their last-accepted
    /// trial step wrote those drifted parameter values into the maps
    /// with `converged=false`, producing a "4096 pixels with sensible
    /// densities, 92 % of which are converged=false" result that
    /// masked catastrophic fit failure.
    #[test]
    fn test_spatial_unconverged_pixels_are_nan() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        // Pick a deliberately wrong initial density (100× true) and cap
        // LM at one iteration so the fit MUST return with
        // `converged=false` and `params = last_walked_step` ≠ initial.
        // This mimics the real-world pattern the bug produced: a fit
        // that walked partway toward the optimum, then ran out of
        // iterations.
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.1], // 100× true — LM can't reach optimum in 1 iter.
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig {
            max_iter: 1,
            ..Default::default()
        }))
        .with_transmission_background(crate::pipeline::BackgroundConfig::default());

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();

        // At least one pixel must fail to converge under this setup —
        // the point of the test is to verify NaN-on-failure for the
        // aggregation path, so we locate an unconverged pixel and
        // check every map at that pixel.
        let unconverged_pixel = (0..4)
            .flat_map(|y| (0..4).map(move |x| (y, x)))
            .find(|(y, x)| !result.converged_map[[*y, *x]]);
        let (uy, ux) = match unconverged_pixel {
            Some(p) => p,
            None => panic!(
                "every pixel converged in max_iter=1 + 100×-off initial density setup — \
                 test is no longer exercising the un-converged aggregation path; \
                 tighten the setup (larger offset or fewer iterations)"
            ),
        };

        // Every output map must be NaN at that pixel.
        for (i, m) in result.density_maps.iter().enumerate() {
            let v = m[[uy, ux]];
            assert!(
                v.is_nan(),
                "density_maps[{i}] at unconverged pixel ({uy},{ux}) must be NaN, got {v}"
            );
        }
        for (i, m) in result.uncertainty_maps.iter().enumerate() {
            let v = m[[uy, ux]];
            assert!(
                v.is_nan(),
                "uncertainty_maps[{i}] at unconverged pixel ({uy},{ux}) must be NaN, got {v}"
            );
        }
        let chi2 = result.chi_squared_map[[uy, ux]];
        assert!(
            chi2.is_nan(),
            "chi_squared_map at unconverged pixel ({uy},{ux}) must be NaN, got {chi2}"
        );
        if let Some(ref a_map) = result.anorm_map {
            let v = a_map[[uy, ux]];
            assert!(
                v.is_nan(),
                "anorm_map at unconverged pixel ({uy},{ux}) must be NaN, got {v}"
            );
        }
        if let Some(ref bg) = result.background_maps {
            for (i, m) in bg.iter().enumerate() {
                let v = m[[uy, ux]];
                assert!(
                    v.is_nan(),
                    "background_maps[{i}] at unconverged pixel ({uy},{ux}) must be NaN, got {v}"
                );
            }
        }
    }

    // ── Counts-KL spatial path (post-collapse) ────────────────────────

    /// Spatial counts-KL dispatch routes through `fit_counts_joint_poisson`
    /// and populates `deviance_per_dof_map`.  Polish auto-disable makes
    /// the per-pixel fits fast enough to run in a unit test; the result
    /// still recovers density on noise-free synthetic.
    #[test]
    fn test_spatial_map_typed_counts_kl_populates_deviance_per_dof_map() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, _) = synthetic_4x4_transmission(&data, true_density, &energies);
        let n_e = energies.len();

        // Synthesize counts: c=2.0, lam_ob=500.  E[O]=lam_ob, E[S]=c·lam_ob·T.
        let c_val = 2.0_f64;
        let lam_ob = 500.0_f64;
        let mut sample = Array3::zeros((n_e, 4, 4));
        let mut open_beam = Array3::from_elem((n_e, 4, 4), lam_ob);
        for y in 0..4 {
            for x in 0..4 {
                for (i, _) in energies.iter().enumerate() {
                    open_beam[[i, y, x]] = lam_ob;
                    sample[[i, y, x]] = c_val * lam_ob * t_3d[[i, y, x]];
                }
            }
        }

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_counts_background(crate::pipeline::CountsBackgroundConfig {
            c: c_val,
            ..Default::default()
        });

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: open_beam.view(),
        };
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        // Deviance map populated (counts-KL path).
        let dpd = r
            .deviance_per_dof_map
            .as_ref()
            .expect("counts-KL spatial should populate deviance_per_dof_map");
        assert_eq!(dpd.shape(), &[4, 4]);
        let sample_val = dpd[[0, 0]];
        assert!(
            sample_val.is_finite(),
            "deviance_per_dof_map[0,0] = {sample_val} (should be finite)"
        );
        // Density recovery (noise-free).
        let density_mean: f64 = r.density_maps[0].iter().copied().sum::<f64>() / 16.0;
        assert!(
            (density_mean - true_density).abs() / true_density < 0.05,
            "mean density {density_mean} vs truth {true_density}",
        );
    }

    /// Polish auto-disable: the `apply_spatial_polish_default` helper
    /// sets `counts_enable_polish = Some(false)` for multi-pixel fits
    /// when the caller has not overridden it.  This asserts the decision
    /// directly (no timing-based heuristics — tested by checking the
    /// resolved config).
    #[test]
    fn test_apply_spatial_polish_default_multi_pixel_auto_disables() {
        // Minimal UnifiedFitConfig — the helper only reads
        // `counts_enable_polish`, so the rest can be stub data.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..10).map(|i| 1.0 + i as f64).collect();
        let cfg = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap();

        // Multi-pixel (n > 1), no caller override → auto-disabled.
        assert_eq!(cfg.counts_enable_polish(), None);
        let resolved = apply_spatial_polish_default(cfg.clone(), 16);
        assert_eq!(
            resolved.counts_enable_polish(),
            Some(false),
            "multi-pixel with no override should auto-disable polish"
        );

        // Single-pixel (n = 1) → no change (let the library default decide).
        let resolved = apply_spatial_polish_default(cfg.clone(), 1);
        assert_eq!(
            resolved.counts_enable_polish(),
            None,
            "single-pixel should preserve the caller's unset state"
        );

        // Caller explicitly turned polish on → multi-pixel must respect it.
        let cfg_forced_on = cfg.clone().with_counts_enable_polish(Some(true));
        let resolved = apply_spatial_polish_default(cfg_forced_on, 16);
        assert_eq!(
            resolved.counts_enable_polish(),
            Some(true),
            "caller override Some(true) must be preserved for multi-pixel"
        );

        // Caller explicitly turned polish off → still off.
        let cfg_forced_off = cfg.with_counts_enable_polish(Some(false));
        let resolved = apply_spatial_polish_default(cfg_forced_off, 16);
        assert_eq!(resolved.counts_enable_polish(), Some(false));
    }

    /// End-to-end: counts-KL spatial map populates `deviance_per_dof_map`
    /// and completes without hitting the polish maxiter cap.  No
    /// wall-clock assertion — relies on the helper test above for the
    /// auto-disable decision.
    #[test]
    fn test_spatial_map_typed_counts_kl_populates_map_without_polish_regression() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, _) = synthetic_4x4_transmission(&data, 0.0005, &energies);
        let n_e = energies.len();

        let mut sample = Array3::zeros((n_e, 4, 4));
        let open_beam = Array3::from_elem((n_e, 4, 4), 500.0);
        for y in 0..4 {
            for x in 0..4 {
                for i in 0..n_e {
                    sample[[i, y, x]] = 500.0 * t_3d[[i, y, x]];
                }
            }
        }

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: open_beam.view(),
        };
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert!(r.deviance_per_dof_map.is_some());
        // All 16 live pixels should have a finite D/dof value.
        let dpd = r.deviance_per_dof_map.as_ref().unwrap();
        assert!(dpd.iter().all(|v| v.is_finite()));
    }

    /// `(Counts, LM)` spatial dispatch must NOT allocate a
    /// `deviance_per_dof_map` — the per-pixel LM path doesn't populate
    /// `deviance_per_dof`, so an `Some(all-NaN)` map would mislead GUI /
    /// Python consumers that switch the GOF label on `is_some()`.
    #[test]
    fn test_spatial_map_typed_counts_lm_no_deviance_map() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, _) = synthetic_4x4_transmission(&data, 0.0005, &energies);
        let n_e = energies.len();
        let mut sample = Array3::zeros((n_e, 4, 4));
        let open_beam = Array3::from_elem((n_e, 4, 4), 500.0);
        for y in 0..4 {
            for x in 0..4 {
                for i in 0..n_e {
                    sample[[i, y, x]] = 500.0 * t_3d[[i, y, x]];
                }
            }
        }

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        // Force LM (counts → transmission conversion under the hood); no
        // deviance is computed by that dispatch.
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: open_beam.view(),
        };
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert!(
            r.deviance_per_dof_map.is_none(),
            "(Counts, LM) must not allocate deviance_per_dof_map (would mislabel GOF in GUI)"
        );
        // chi_squared_map (Pearson) is the GOF on the LM path.
        assert!(r.chi_squared_map.iter().any(|v| v.is_finite()));
    }

    /// Transmission input must never produce a `deviance_per_dof_map`
    /// (regardless of solver — the counts-KL dispatch isn't reached).
    #[test]
    fn test_spatial_map_typed_transmission_no_deviance_map() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap();
        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert!(r.deviance_per_dof_map.is_none());
    }

    /// `fit_energy_scale=True` on the spatial path routes per-pixel TZERO
    /// calibration through the same config used by single-spectrum fits,
    /// populates `t0_us_map` and `l_scale_map`, and leaves them `None`
    /// when the flag is off.  Regression against the prior gap where
    /// the Python binding accepted `fit_energy_scale` for single
    /// spectra but not for spatial, forcing callers to pre-calibrate.
    #[test]
    fn test_spatial_map_typed_fit_energy_scale_populates_maps() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_energy_scale(0.0, 1.0, 25.0);

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        let t0_map = result
            .t0_us_map
            .as_ref()
            .expect("t0_us_map must be Some when fit_energy_scale=true");
        let l_map = result
            .l_scale_map
            .as_ref()
            .expect("l_scale_map must be Some when fit_energy_scale=true");
        assert_eq!(t0_map.shape(), [4, 4]);
        assert_eq!(l_map.shape(), [4, 4]);
        // Post-#458 B1 semantics:
        //   * Converged pixel  → finite t0 / L_scale in the maps
        //   * Un-converged pixel → NaN in the maps (the LM last-walked
        //     value is NOT leaked)
        // Parameter-value correctness (t0 ≈ 0, L ≈ 1 on noise-free
        // nominal-grid data) is tested at the fitting layer, not here;
        // this test only exercises wiring + aggregation gating.
        for y in 0..4 {
            for x in 0..4 {
                let converged = result.converged_map[[y, x]];
                let t0 = t0_map[[y, x]];
                let ls = l_map[[y, x]];
                if converged {
                    assert!(
                        t0.is_finite() && ls.is_finite(),
                        "converged pixel ({y},{x}) must have finite t0/L, got t0={t0}, L={ls}"
                    );
                } else {
                    assert!(
                        t0.is_nan() && ls.is_nan(),
                        "un-converged pixel ({y},{x}) must have NaN t0/L (B1 gating), got t0={t0}, L={ls}"
                    );
                }
            }
        }
    }

    /// Without `fit_energy_scale`, the TZERO maps are `None` — gate check.
    #[test]
    fn test_spatial_map_typed_no_energy_scale_no_maps() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(result.t0_us_map.is_none());
        assert!(result.l_scale_map.is_none());
    }

    /// `(Counts + LM + fit_energy_scale=true)` must be rejected at
    /// `spatial_map_typed` entry (issue #458 B3).  The combination
    /// passed silently before and produced 92 % non-convergence with
    /// garbage parameter values on real VENUS data.
    #[test]
    fn test_spatial_map_typed_rejects_counts_lm_with_energy_scale() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.001, &energies, 1000.0);
        let data = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_energy_scale(0.0, 1.0, 25.0);

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("LM + counts + fit_energy_scale must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("fit_energy_scale") && msg.contains("lm"),
            "error message should name both culprits, got: {msg}"
        );
        assert!(
            msg.contains("#458"),
            "error message should reference the tracking issue, got: {msg}"
        );
    }

    /// `(Counts + KL + fit_energy_scale=true)` is allowed — KL is
    /// robust per-pixel even with energy-scale on real data.
    #[test]
    fn test_spatial_map_typed_allows_counts_kl_with_energy_scale() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.001, &energies, 1000.0);
        let data = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_energy_scale(0.0, 1.0, 25.0);

        let result = spatial_map_typed(&data, &config, None, None, None)
            .expect("KL + counts + fit_energy_scale must be allowed");
        assert!(result.t0_us_map.is_some());
    }

    /// `fit_energy_scale + fit_temperature` must be rejected at
    /// spatial entry (Codex review follow-up to #458).  The
    /// single-spectrum fitter errors on this combination, but without
    /// a spatial-layer guard every pixel would error and
    /// `spatial_map_typed` would silently return `n_failed == n_total`
    /// with an all-NaN map instead of a clear error.
    #[test]
    fn test_spatial_map_typed_rejects_energy_scale_with_temperature() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_temperature(true)
        .with_energy_scale(0.0, 1.0, 25.0);

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("fit_energy_scale + fit_temperature must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("fit_energy_scale") && msg.contains("fit_temperature"),
            "error message should name both culprits, got: {msg}"
        );
    }

    /// `(Transmission + LM + fit_energy_scale=true)` is allowed —
    /// per-pixel transmission has higher SNR per bin than raw counts
    /// and this combination is sometimes useful for calibration
    /// crosschecks.  NaN-on-failure gating (B1) still protects
    /// downstream consumers.
    #[test]
    fn test_spatial_map_typed_allows_transmission_lm_with_energy_scale() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_energy_scale(0.0, 1.0, 25.0);

        let result = spatial_map_typed(&data, &config, None, None, None)
            .expect("LM + transmission + fit_energy_scale must be allowed");
        assert!(result.t0_us_map.is_some());
    }
}
