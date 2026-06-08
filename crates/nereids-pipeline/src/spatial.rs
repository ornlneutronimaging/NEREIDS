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
    InstrumentParams, broadened_cross_sections_on_working_grid, unbroadened_cross_sections,
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
/// `back_d_map`, `back_f_map`, `t0_us_map`, `l_scale_map`)
/// contains `NaN` at every pixel where
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
    /// Per-pixel SAMMY `Anorm` map (when background fitting is enabled).
    /// NaN at pixels where `converged_map` is `false`.
    pub anorm_map: Option<Array2<f64>>,
    /// Per-pixel SAMMY background polynomial coefficient maps —
    /// **only the first three coefficients** `[BackA, BackB, BackC]` of
    /// the SAMMY 6-term form
    /// `bg(E) = BackA + BackB/√E + BackC·√E + BackD·exp(-BackF/√E)`.
    /// Both LM-transmission and counts-KL paths use these semantics
    /// (legacy alpha-fitting `[b0, b1, alpha_2]` layout was retired with
    /// `fit_counts_poisson` in PR #450).
    ///
    /// The exponential `BackD`/`BackF` terms are surfaced separately
    /// in [`Self::back_d_map`] / [`Self::back_f_map`] — both `None`
    /// for counts-KL runs (the joint-Poisson dispatch never fits the
    /// exponential tail) and for LM transmission runs that left
    /// `fit_back_d` / `fit_back_f` at their default `false`.
    ///
    /// NaN at pixels where `converged_map` is `false`.
    pub background_maps: Option<[Array2<f64>; 3]>,
    /// Per-pixel fitted SAMMY exponential background amplitude `BackD`.
    /// `Some` only when the LM transmission background path was active
    /// AND `fit_back_d=true`; `None` otherwise (counts-KL runs, LM
    /// runs without a background model, and LM runs that fit the
    /// polynomial terms but left the exponential tail at its initial
    /// value).
    /// NaN at pixels where `converged_map` is `false`.
    pub back_d_map: Option<Array2<f64>>,
    /// Per-pixel fitted SAMMY exponential background decay constant
    /// `BackF`.  `Some` only when the LM transmission background path
    /// was active AND `fit_back_f=true`; `None` otherwise.  Mirrors
    /// [`Self::back_d_map`]'s gating because `BackD` and `BackF` are
    /// required to fit together (see `validate_transmission_background`
    /// in `crate::pipeline`).
    /// NaN at pixels where `converged_map` is `false`.
    pub back_f_map: Option<Array2<f64>>,
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

use crate::pipeline::{
    InputData, SolverConfig, UnifiedFitConfig, count_free_params, fit_spectrum_typed,
    required_active_bins, validate_transmission_background,
};

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

/// Hoist whole-config `InvalidParameter` rejections out of the per-pixel
/// rayon closure so they surface as a single boundary error instead of
/// silently degrading to an all-NaN `SpatialResult` via the
/// `Err(_) => failed_count += 1` swallow at the bottom of the loop.
///
/// Every gate here mirrors a per-pixel `Err(PipelineError::InvalidParameter)`
/// raised inside `fit_spectrum_typed` / `fit_transmission_poisson` /
/// `fit_counts_joint_poisson` whose decision depends only on
/// `(input variant, config)` — i.e. fires identically for every pixel.
/// Per-pixel error variants (numerical fit failure, per-pixel detector
/// background contamination on `CountsWithNuisance`) intentionally stay
/// inside the closure where they correctly produce a NaN-only single
/// pixel rather than a whole-map error.
///
/// The error messages here are kept byte-identical to the originating
/// per-pixel sites so the user-facing diagnostic does not bifurcate
/// based on whether the call came through the single-spectrum or
/// spatial entry point.
fn validate_spatial_fit_preflight(
    input: &InputData3D<'_>,
    config: &UnifiedFitConfig,
) -> Result<(), PipelineError> {
    // Gate: `fit_temperature && temperature_k < 1.0` (mirrors
    // `pipeline.rs::fit_spectrum_typed` temperature-init guard).
    // Without hoisting, a user who forgets units and writes `0.025`
    // for 25 meV would see `Ok(SpatialResult { n_converged: 0,
    // density_maps: all-NaN })` instead of the actionable message.
    if config.fit_temperature() && config.temperature_k() < 1.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "temperature must be >= 1.0 K when fit_temperature is true, got {}",
            config.temperature_k(),
        )));
    }

    // Resolve `SolverConfig::Auto` against the input variant — counts
    // → PoissonKL, transmission → LM.  `effective_solver` lives on
    // `UnifiedFitConfig` but takes the 1D `InputData`; inline the
    // resolution here so we do not have to materialise a 1D stub.
    let is_counts = input.is_counts();
    let is_kl = matches!(config.solver(), SolverConfig::PoissonKL(_))
        || (matches!(config.solver(), SolverConfig::Auto) && is_counts);

    // Gate: transmission + Poisson-KL solver path does not honour
    // `fit_energy_range` — `fit_transmission_poisson` rejects this
    // combination per-pixel (`pipeline.rs::fit_transmission_poisson`).
    // Without hoisting, every pixel errors and the spatial layer
    // hides the dispatch-level incompatibility.  Counts-KL (joint-
    // Poisson) and LM transmission both honour the mask correctly,
    // so this gate is scoped to the transmission + KL combination.
    if !is_counts && is_kl && config.fit_energy_range().is_some() {
        return Err(PipelineError::InvalidParameter(
            "fit_energy_range is not supported for the transmission + \
             Poisson-KL solver path. Use joint-Poisson (provide sample + \
             open-beam counts) or switch to the LM transmission solver."
                .into(),
        ));
    }

    // Gate: `fit_energy_range` selects fewer active bins than the
    // dispatch can solve.  The active-mask + grid are shared by every
    // pixel, so the per-pixel `n_active < required` rejection in the
    // LM transmission path (`pipeline.rs::fit_transmission_lm`) and
    // the joint-Poisson path (`pipeline.rs::fit_counts_joint_poisson`)
    // both fire identically across the map.  We compute `required`
    // from the config's free-parameter count (densities + temperature
    // + energy-scale + transmission_background flags), clamped to a
    // floor of 2 — that combined `max(2, n_free)` covers both the
    // numerical-stability minimum and the underdetermined-system
    // rejection.  Without the `n_free` factor, a config with
    // multiple densities + background terms + temperature + energy-
    // scale (n_free can reach ~10) would silently pass the preflight
    // with a 3-bin window and every pixel would return non-converged
    // / NaN — the all-NaN spatial-result class this preflight exists
    // to prevent.  See [`required_active_bins`] in `pipeline.rs`.
    if let Some((e_min, e_max)) = config.fit_energy_range() {
        let active_mask = nereids_fitting::active_mask::build_active_mask(
            config.energies(),
            config.fit_energy_range(),
        );
        let n_active = nereids_fitting::active_mask::active_count(
            active_mask.as_deref(),
            config.energies().len(),
        );
        let required = required_active_bins(config);
        if n_active < required {
            // Mirror the per-pixel string from whichever path the
            // dispatcher would actually take.  LM and joint-Poisson
            // both reach this branch; transmission + Poisson-KL is
            // already rejected by the previous gate above.
            let path_msg = if is_counts && is_kl {
                "joint-Poisson"
            } else {
                "LM transmission"
            };
            return Err(PipelineError::InvalidParameter(format!(
                "fit_energy_range [{e_min}, {e_max}] eV selects {n_active} active bin(s) \
                 on the configured energy grid; at least {required} active bin(s) are \
                 required for {path_msg} fitting with {n_free} free parameter(s) \
                 (underdetermined when n_active < n_free)",
                n_free = count_free_params(config),
            )));
        }
    }

    // ── Counts-KL (joint-Poisson) whole-config gates ────────────────
    // Every gate below mirrors a per-pixel rejection in
    // `pipeline.rs::fit_counts_joint_poisson`.  All fire identically
    // across the map because they depend only on shared config flags
    // (alpha fitting, B_A/B/C interlock, `c` value); per-pixel
    // detector-background contamination is *not* hoisted because
    // `CountsWithNuisance` carries per-pixel `background` slices and
    // contamination is a legitimately per-pixel signal.
    if is_counts && is_kl {
        if let Some(bg) = config.counts_background() {
            if bg.fit_alpha_1 || bg.fit_alpha_2 {
                return Err(PipelineError::InvalidParameter(
                    "joint-Poisson solver does not support fit_alpha_1/fit_alpha_2: \
                     the profile lambda-hat absorbs the global flux scale (alpha_1 redundant); \
                     alpha_2 / B_det wiring is deferred to memo 35 §P3."
                        .into(),
                ));
            }
            // `c` defaults to `1.0` when absent, matching the
            // `.unwrap_or(1.0)` in `fit_counts_joint_poisson`; only an
            // explicit non-finite or non-positive `c` is rejected.
            // Python pre-validates this at the binding boundary so
            // Python users hit a `ValueError` before this gate, but
            // Rust core callers can still reach this path.
            if !(bg.c.is_finite() && bg.c > 0.0) {
                return Err(PipelineError::InvalidParameter(format!(
                    "joint-Poisson solver requires finite c > 0 in CountsBackgroundConfig, got {}",
                    bg.c,
                )));
            }
        }
        if let Some(bg) = config.transmission_background()
            && (bg.fit_back_b || bg.fit_back_c)
            && !bg.fit_back_a
        {
            return Err(PipelineError::InvalidParameter(
                "joint-Poisson transmission_background: B_A (fit_back_a) must be \
                 enabled whenever any of B_B / B_C is enabled (memo 35 §P2.2 — \
                 A_n alone cannot absorb a constant offset; EG2 S2 C_An → −23% \
                 density bias)."
                    .into(),
            ));
        }
    }

    Ok(())
}

/// Validity domain for an up-front detector-cube value check.
///
/// Each variant encodes the physically-meaningful constraint for one class of
/// cube (see [`validate_spatial_data_values`]).
#[derive(Clone, Copy)]
enum CubeDomain {
    /// Finite (NaN / ±∞ rejected); sign unconstrained.  Used for the
    /// transmission **value**: SAMMY does not reject negative transmission —
    /// measurement noise / open-beam over-subtraction can push a measured
    /// point below 0 — so only finiteness is required.
    Finite,
    /// Finite **and strictly > 0**.  Used for the 1-σ uncertainty: a zero or
    /// negative error bar is a singular weight (SAMMY: zero uncertainties are
    /// never allowed).  Without this guard the old `σ.max(1e-10)` floor turned
    /// a bad σ into a `1/(1e-10)² = 1e20` maximum-confidence bin — the
    /// opposite of the LM core's `s <= 0.0 => 1/1e30` negligible-weight rule.
    FinitePositive,
    /// Finite **and ≥ 0**.  Used for raw detector counts / open-beam / flux:
    /// non-negative by construction (zero is legitimate — "no counts in this
    /// bin"), so a negative or non-finite value signals an upstream loader /
    /// TOF-normalisation bug, exactly as the `validate_counts` docstring in
    /// `nereids_fitting::joint_poisson` describes.
    FiniteNonNegative,
}

impl CubeDomain {
    #[inline]
    fn accepts(self, v: f64) -> bool {
        match self {
            CubeDomain::Finite => v.is_finite(),
            CubeDomain::FinitePositive => v.is_finite() && v > 0.0,
            CubeDomain::FiniteNonNegative => v.is_finite() && v >= 0.0,
        }
    }

    fn describe(self) -> &'static str {
        match self {
            CubeDomain::Finite => "finite",
            CubeDomain::FinitePositive => "finite and > 0",
            CubeDomain::FiniteNonNegative => "finite and >= 0",
        }
    }
}

/// Check every relevant element of one detector cube, returning the first
/// violation as a typed `InvalidParameter` naming the cube and the offending
/// `(y, x, e)`.
///
/// Iterates in memory order — energy plane `e` outer (a contiguous `h × w`
/// block in the `(n_energies, height, width)` input layout), live pixels
/// inner — and short-circuits on the first bad value.  When `active_mask` is
/// `Some` (the transmission / uncertainty cubes), bins outside the user's
/// `fit_energy_range` are skipped: the LM core excludes them from the fit, so
/// a non-finite value there is irrelevant.  The raw-count cubes pass `None` to
/// check every bin (see [`validate_spatial_data_values`] for the rationale).
fn check_cube(
    cube: &ArrayView3<'_, f64>,
    field: &'static str,
    domain: CubeDomain,
    live_pixels: &[(usize, usize)],
    active_mask: Option<&[bool]>,
) -> Result<(), PipelineError> {
    let n_energies = cube.shape()[0];
    for e in 0..n_energies {
        if active_mask.is_some_and(|m| !m[e]) {
            continue;
        }
        for &(y, x) in live_pixels {
            let v = cube[[e, y, x]];
            if !domain.accepts(v) {
                return Err(PipelineError::InvalidParameter(format!(
                    "{field} at (y={y}, x={x}, e={e}) must be {}, got {v}",
                    domain.describe(),
                )));
            }
        }
    }
    Ok(())
}

/// Reject non-finite / out-of-domain detector-cube **values** up front, so bad
/// input fails with a typed `InvalidParameter` (mapped to `PyValueError` at
/// the Python boundary) instead of being silently transformed by the
/// per-pixel sanitation that used to run inside the rayon closure
/// (`v.max(0.0)` on counts, `σ.max(1e-10)` on uncertainty).  That sanitation
/// defeated the downstream joint-Poisson `validate_counts` guard
/// (`NaN.max(0.0) == 0.0` passes silently) and turned a bad σ into a
/// maximum-confidence bin — concealing precisely the upstream TOF-norm /
/// loader bugs the guards exist to surface.
///
/// Only **live** pixels are checked: a `dead_pixels`-masked pixel is excluded
/// from the fit and from the averaged open-beam flux, so its data is never
/// read and may legitimately hold detector garbage.
///
/// Bin scope differs by quantity, matching each path's existing downstream
/// contract so that no currently-passing fit changes behaviour:
/// - **transmission / uncertainty** are checked on **active bins only**.
///   Transmission is derived (`sample / open_beam`) and is legitimately
///   undefined where open-beam → 0; the LM core deliberately *skips* inactive
///   bins (`nereids_fitting::lm` — "y_obs is NaN outside the user's
///   fit-energy range"), so a NaN in an out-of-`fit_energy_range` bin is
///   harmless and must not be rejected.
/// - **counts / open-beam / flux** are checked on **all bins** — raw detector
///   quantities, where a bad value anywhere is an upstream bug (matching the
///   all-bins `validate_counts`).
/// - **background** (CountsWithNuisance) is checked **finite, all bins**,
///   closing the `NaN.abs() > 1e-12 == false` finiteness leak in the
///   per-pixel detector-background gate.
fn validate_spatial_data_values(
    input: &InputData3D<'_>,
    live_pixels: &[(usize, usize)],
    active_mask: Option<&[bool]>,
) -> Result<(), PipelineError> {
    match input {
        InputData3D::Transmission {
            transmission,
            uncertainty,
        } => {
            check_cube(
                transmission,
                "transmission",
                CubeDomain::Finite,
                live_pixels,
                active_mask,
            )?;
            check_cube(
                uncertainty,
                "uncertainty",
                CubeDomain::FinitePositive,
                live_pixels,
                active_mask,
            )?;
        }
        InputData3D::Counts {
            sample_counts,
            open_beam_counts,
        } => {
            check_cube(
                sample_counts,
                "sample_counts",
                CubeDomain::FiniteNonNegative,
                live_pixels,
                None,
            )?;
            check_cube(
                open_beam_counts,
                "open_beam_counts",
                CubeDomain::FiniteNonNegative,
                live_pixels,
                None,
            )?;
        }
        InputData3D::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        } => {
            check_cube(
                sample_counts,
                "sample_counts",
                CubeDomain::FiniteNonNegative,
                live_pixels,
                None,
            )?;
            check_cube(
                flux,
                "flux",
                CubeDomain::FiniteNonNegative,
                live_pixels,
                None,
            )?;
            check_cube(
                background,
                "background",
                CubeDomain::Finite,
                live_pixels,
                None,
            )?;
        }
    }
    Ok(())
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

    // `fit_spectrum_typed` rejects `CountsWithNuisance + LM` per-pixel
    // (see `validate_input_solver` in `pipeline.rs` — "CountsWithNuisance
    // requires a counts-domain solver"), but per-pixel errors here are
    // swallowed as `n_failed` and `spatial_map_typed` returns
    // `Ok(SpatialResult)` with all-NaN maps.  Hoist the rejection so
    // callers get a clear diagnostic instead of a silently-failed
    // spatial result.
    if matches!(input, InputData3D::CountsWithNuisance { .. })
        && matches!(config.solver(), SolverConfig::LevenbergMarquardt(_))
    {
        return Err(PipelineError::InvalidParameter(
            "spatial_map_typed: InputData3D::CountsWithNuisance requires a counts-domain \
             solver (joint-Poisson via SolverConfig::PoissonKL or SolverConfig::Auto); \
             SolverConfig::LevenbergMarquardt cannot use the user-supplied nuisance \
             parameters (alpha_1, alpha_2).  Choose a counts-domain solver, or drop the \
             nuisance arm by passing `InputData3D::Counts` instead."
                .into(),
        ));
    }

    // Validate `transmission_background` BackD/BackF here rather than
    // per-pixel.  Invalid configs (unpaired flags, non-finite or non-
    // positive init values, counts-KL plus exponential tail) would
    // otherwise be swallowed as `n_failed` per pixel and produce an
    // all-NaN map with no diagnostic.
    if let Some(bg) = config.transmission_background() {
        // SAMMY pairs BackD/BackF — enabling only one leaves the other
        // registered but unused.  Already enforced per-pixel in the LM
        // solver; surface up-front for the spatial dispatch.
        validate_transmission_background(bg)?;
        // BackF's Jacobian column zeros out at BackD ≈ 0 (and BackD
        // becomes a constant duplicate of BackA at BackF ≈ 0).  Reject
        // non-positive initial values so the LM solver does not silently
        // produce all-NaN maps via a degenerate Jacobian.  Also reject
        // `NaN` / `+inf` — both pass `<= 0.0` (NaN comparisons are
        // always false; +inf is > 0) but propagate into the fit
        // parameters and silently corrupt the result.
        if bg.fit_back_d && (!bg.back_d_init.is_finite() || bg.back_d_init <= 0.0) {
            return Err(PipelineError::InvalidParameter(format!(
                "transmission_background.back_d_init must be finite and strictly \
                 positive when fit_back_d=true (got {}). BackF's Jacobian column \
                 zeros out at BackD ≈ 0; non-finite or non-positive initial values \
                 produce a degenerate fit that LM cannot recover.",
                bg.back_d_init,
            )));
        }
        if bg.fit_back_f && (!bg.back_f_init.is_finite() || bg.back_f_init <= 0.0) {
            return Err(PipelineError::InvalidParameter(format!(
                "transmission_background.back_f_init must be finite and strictly \
                 positive when fit_back_f=true (got {}). BackD becomes a constant \
                 duplicate of BackA at BackF ≈ 0; non-finite or non-positive initial \
                 values produce a degenerate fit that LM cannot recover.",
                bg.back_f_init,
            )));
        }
        // The joint-Poisson (counts-KL) dispatch never fits the SAMMY
        // exponential tail — `fit_counts_joint_poisson` rejects
        // `fit_back_d || fit_back_f` per pixel.  Surface up-front so the
        // user gets a clear diagnostic instead of an all-NaN map.
        if (bg.fit_back_d || bg.fit_back_f)
            && input.is_counts()
            && !matches!(config.solver(), SolverConfig::LevenbergMarquardt(_))
        {
            return Err(PipelineError::InvalidParameter(
                "spatial_map_typed: transmission_background with fit_back_d=true / \
                 fit_back_f=true cannot be combined with the counts-KL (joint-Poisson) \
                 dispatch. The joint-Poisson solver does not fit the SAMMY exponential \
                 tail. Either switch to SolverConfig::LevenbergMarquardt or disable the \
                 exponential tail (fit_back_d=false, fit_back_f=false)."
                    .into(),
            ));
        }
    }

    // Hoist whole-config `InvalidParameter` rejections so they surface as
    // a single boundary error instead of being swallowed pixel-by-pixel
    // into an all-NaN `SpatialResult`.  See
    // `validate_spatial_fit_preflight` for the full gate list and
    // per-gate rationale.  Must run before any rayon work.
    //
    // Ordering note: the preflight runs *after* the dispatch /
    // solver-compatibility guards above (CountsWithNuisance + LM,
    // fit_energy_scale + fit_temperature, transmission_background
    // BackD/BackF interlocks, …).  The fit-range, temperature and
    // alpha gates inside the preflight only meaningfully apply once
    // the input → solver dispatch is known to be valid; otherwise
    // a downstream "LM transmission active-bin" message would
    // shadow the more fundamental "CountsWithNuisance requires a
    // counts-domain solver" diagnostic.
    validate_spatial_fit_preflight(input, config)?;

    // Reject a malformed caller-supplied precomputed cross-section stack once,
    // before the per-pixel rayon loop (and before the σ_eff group-collapse
    // below, which indexes `xs[0]`).  A freshly-computed stack carries no
    // `precomputed_cross_sections`, so this is a no-op on the common path.
    crate::pipeline::validate_precomputed_cross_sections(config)?;

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
    // The exponential `BackD` / `BackF` terms are LM-transmission-only:
    // `fit_counts_joint_poisson` rejects `fit_back_d || fit_back_f` for
    // the counts-KL path.  Gate both maps on the transmission-background
    // config carrying the per-term `fit_back_d` / `fit_back_f` flags so
    // callers can distinguish "map full of NaN because no pixel
    // converged" (`Some([NaN, ...])`) from "the exponential tail was
    // never engaged" (`None`).
    let has_back_d_map = config
        .transmission_background()
        .is_some_and(|bg| bg.fit_back_d);
    let has_back_f_map = config
        .transmission_background()
        .is_some_and(|bg| bg.fit_back_f);

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
            back_d_map: if has_back_d_map {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            back_f_map: if has_back_f_map {
                Some(Array2::from_elem((height, width), f64::NAN))
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

    // Reject non-finite / out-of-domain detector-cube VALUES up front —
    // before the (potentially multi-GB) transpose below and the shared
    // cross-section precompute — so bad input fails with a clear
    // `InvalidParameter` instead of being silently sanitised per-pixel.
    // `pixel_coords` is non-empty here (the all-dead case returned above), so
    // only live pixels are checked.  The mask scopes the transmission /
    // uncertainty check to the user's fit-energy range; see
    // `validate_spatial_data_values` for the per-cube domains and rationale.
    let value_active_mask = nereids_fitting::active_mask::build_active_mask(
        config.energies(),
        config.fit_energy_range(),
    );
    validate_spatial_data_values(input, &pixel_coords, value_active_mask.as_deref())?;

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

    // Precompute cross-sections once (shared across all pixels).
    //
    // Issue #608: broaden σ on the WORKING grid (auxiliary extended grid when a
    // Gaussian resolution function is active, else the data grid) so each
    // per-pixel `PrecomputedTransmissionModel` applies Beer-Lambert +
    // resolution on the working grid and extracts the data points last —
    // matching `forward_model`.  `xs` (data-grid σ) is still needed for the
    // cubature / scalar surrogate builders and shape validation; `work_xs`
    // carries the working-grid σ.  For tabulated / no resolution the working
    // grid IS the data grid, the layout is the identity, and `work_xs` is left
    // unset (the model falls back to the data-grid σ, preserving the surrogate
    // fast paths byte-for-byte).
    let instrument = config.resolution().map(|r| InstrumentParams {
        resolution: r.clone(),
    });

    // Determine the working-grid layout FIRST, cheaply (no Doppler
    // broadening): `resolution_working_grid` only builds the auxiliary grid
    // geometry (boundary extension + resonance fine-structure) — it does NOT
    // evaluate or broaden σ.  When the layout is the identity (tabulated / no
    // resolution) the working grid IS the data grid, so no working-grid σ is
    // needed and we must NOT pay for the full per-isotope
    // `broadened_cross_sections_on_working_grid` just to discover the layout
    // is trivial.  Only the genuine Gaussian aux-grid case below runs the
    // expensive broadening.
    let rd_refs: Vec<&_> = config.resonance_data().iter().collect();
    let layout = nereids_physics::transmission::resolution_working_grid(
        config.energies(),
        instrument.as_ref(),
        &rd_refs,
    )
    .map_err(PipelineError::Transmission)?;
    let aux_grid_active = !layout.is_identity();

    // (xs = data-grid σ, work_xs = working-grid σ when an aux grid exists).
    let (xs, work_xs) = match config.precomputed_cross_sections().cloned() {
        // Caller supplied data-grid σ.  When a Gaussian aux grid exists we
        // still need working-grid σ for the #608-correct path, so recompute it
        // from resonance data (the data-grid σ alone cannot be de-extracted
        // back onto the aux grid).  When no aux grid exists the supplied σ is
        // already the working-grid σ and we skip Doppler broadening entirely.
        Some(cached) if !aux_grid_active => (cached, None),
        Some(cached) => {
            let working = broadened_cross_sections_on_working_grid(
                config.energies(),
                config.resonance_data(),
                config.temperature_k(),
                instrument.as_ref(),
                cancel,
            )?;
            (cached, Some(Arc::new(working.sigma)))
        }
        None => {
            let working = broadened_cross_sections_on_working_grid(
                config.energies(),
                config.resonance_data(),
                config.temperature_k(),
                instrument.as_ref(),
                cancel,
            )?;
            if aux_grid_active {
                // Aux grid: extract the data-grid σ; keep the working σ.
                let data_xs: Vec<Vec<f64>> = working
                    .sigma
                    .iter()
                    .map(|s| working.layout.extract(s))
                    .collect();
                (Arc::new(data_xs), Some(Arc::new(working.sigma)))
            } else {
                // Working grid == data grid: σ is the data-grid σ directly.
                (Arc::new(working.sigma), None)
            }
        }
    };

    // Working-grid layout (energies + data-index map) shared across pixels,
    // reusing the layout computed above.  Only attached when a Gaussian aux
    // grid is active so the per-pixel precomputed model extracts data points
    // after resolution; `None` for tabulated / no resolution.
    let work_layout: Option<Arc<nereids_physics::transmission::WorkingGridLayout>> =
        if work_xs.is_some() {
            Some(Arc::new(layout))
        } else {
            None
        };

    // When groups are active and temperature is NOT being fitted, collapse
    // per-member broadened XS into per-group σ_eff once here.  This avoids
    // redundant O(n_members × n_energies) collapsing inside
    // build_transmission_model on every per-pixel call.  Applied to BOTH the
    // data-grid σ and the working-grid σ so they stay aligned (issue #608).
    let collapse = |xs: &Arc<Vec<Vec<f64>>>| -> Arc<Vec<Vec<f64>>> {
        if !config.fit_temperature()
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
            Arc::clone(xs)
        }
    };
    let xs = collapse(&xs);
    let work_xs = work_xs.as_ref().map(collapse);

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
        // Issue #608: attach the working-grid σ + layout for the Gaussian
        // aux-grid path so each per-pixel `PrecomputedTransmissionModel` applies
        // resolution on the working grid and extracts the data points last.
        // `with_precomputed_cross_sections` (above) clears any stale work σ, so
        // this must come AFTER it.  `None` for tabulated / no resolution (the
        // model uses the data-grid σ directly).
        if let (Some(work_xs), Some(layout)) = (work_xs.clone(), work_layout.clone()) {
            cfg = cfg.with_precomputed_work_cross_sections(work_xs, layout);
        }
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
    //
    // **If this isn't the right assumption for your data** — e.g. you
    // have a genuinely spatially-varying beam profile and pre-estimated
    // per-pixel flux + detector-background spectra — use
    // [`InputData3D::CountsWithNuisance`] instead.  That variant
    // bypasses the averaging and pairs each pixel's sample with the
    // caller-supplied per-pixel flux and bg spectra.
    //
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
            // Each open-beam bin is individually finite and non-negative
            // (validated above), but summing many large finite values can
            // still overflow to +inf.  Surface that as the same up-front
            // `InvalidParameter` rather than letting a non-finite averaged
            // flux degrade silently into all-NaN pixels downstream.
            if let Some(e) = flux.iter().position(|v| !v.is_finite()) {
                return Err(PipelineError::InvalidParameter(format!(
                    "spatially-averaged open-beam flux is non-finite at energy \
                     bin e={e} (got {}); summed open-beam counts overflowed. \
                     Check the open-beam cube magnitude.",
                    flux[e],
                )));
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
                    let ob_spectrum: Vec<f64> = data_b.slice(s![y, x, ..]).to_vec();

                    // Sample counts flow through unsanitised: NaN / negative
                    // values are rejected up-front by
                    // `validate_spatial_data_values`, so the per-pixel
                    // `v.max(0.0)` clamp that used to conceal them (and pass a
                    // bogus 0 through the joint-Poisson `validate_counts`
                    // guard) is gone.
                    //
                    // Check effective solver: KL uses CountsWithNuisance
                    // (averaged flux), LM uses raw Counts (auto-converts to
                    // transmission inside fit_spectrum_typed).
                    let effective = fast_config.effective_solver(&InputData::Counts {
                        sample_counts: spectrum_a.clone(),
                        open_beam_counts: ob_spectrum.clone(),
                    });
                    match effective {
                        SolverConfig::PoissonKL(_) => InputData::CountsWithNuisance {
                            sample_counts: spectrum_a,
                            flux: averaged_flux.as_ref().unwrap().clone(),
                            // Raw-count spatial path currently assumes zero
                            // detector background unless the caller provides
                            // explicit nuisance spectra.
                            background: background_zeros.clone(),
                        },
                        _ => InputData::Counts {
                            sample_counts: spectrum_a,
                            open_beam_counts: ob_spectrum,
                        },
                    }
                }
                InputData3D::CountsWithNuisance { .. } => InputData::CountsWithNuisance {
                    // Sample flows through unsanitised — bad values are
                    // rejected up-front by `validate_spatial_data_values`.
                    sample_counts: spectrum_a,
                    flux: data_b.slice(s![y, x, ..]).to_vec(),
                    background: data_c
                        .as_ref()
                        .expect("CountsWithNuisance requires background cube")
                        .slice(s![y, x, ..])
                        .to_vec(),
                },
                InputData3D::Transmission { .. } => {
                    // Uncertainty flows through unsanitised: a zero / negative
                    // / non-finite σ in an active bin is rejected up-front by
                    // `validate_spatial_data_values`, so the per-pixel
                    // `σ.max(1e-10)` floor (which turned a bad σ into a 1e20
                    // maximum-confidence weight) is gone.  This matches the
                    // single-spectrum path, which passes σ straight to the LM
                    // core (`pipeline::fit_transmission_lm`).
                    let spectrum_b: Vec<f64> = data_b.slice(s![y, x, ..]).to_vec();
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

    // If cancellation was requested at any point, return `Err(Cancelled)` —
    // NOT a partial `Ok(SpatialResult)`. The rayon closure stops launching new
    // pixel fits once `cancel` is set, so by the time we get here `results`
    // holds only the pixels that finished before cancellation; every other
    // pixel would be left as a NaN hole, indistinguishable from a genuinely
    // failed fit. A non-GUI caller (e.g. the Python binding) has no other
    // signal that the map is incomplete, so a partial map is silently wrong.
    // The previous `&& results.is_empty()` guard only caught the rare case
    // where cancellation beat *every* pixel; mid-run cancellation slipped
    // through and produced a partial map.
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
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
    let mut back_d_map: Option<Array2<f64>> = if has_back_d_map {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut back_f_map: Option<Array2<f64>> = if has_back_f_map {
        Some(Array2::from_elem((height, width), f64::NAN))
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
    // `test_spatial_unconverged_pixels_are_nan`; this block makes
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
        // `SpectrumFitResult` carries `back_d` / `back_f` as
        // `Option<f64>` — `None` when the bg model never fit the
        // exponential tail.  Maps here are only materialised when LM
        // actually fit them (gated via `has_back_d_map` /
        // `has_back_f_map`), so a converged pixel should always carry
        // `Some(value)`.  Fall back to NaN for the rare case of `None`
        // at a converged pixel — that surfaces an upstream bug via the
        // NaN-on-failure contract rather than a misleading sentinel
        // `0.0`.
        if let Some(ref mut map) = back_d_map {
            map[[*y, *x]] = result.back_d.unwrap_or(f64::NAN);
        }
        if let Some(ref mut map) = back_f_map {
            map[[*y, *x]] = result.back_f.unwrap_or(f64::NAN);
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
        back_d_map,
        back_f_map,
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
    use nereids_endf::resonance::test_support::{
        synthetic_single_resonance, u238_single_resonance,
    };

    /// Build a synthetic transmission stack of shape `(n_e, height, width)`
    /// where every pixel holds the same spectrum for a known density.
    fn synthetic_grid_transmission(
        res_data: &nereids_endf::resonance::ResonanceData,
        true_density: f64,
        energies: &[f64],
        height: usize,
        width: usize,
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
            work_layout: None,
        };
        let t_1d = model.evaluate(&[true_density]).unwrap();
        let sigma_1d: Vec<f64> = t_1d.iter().map(|&v| 0.01 * v.max(0.01)).collect();

        let mut t_3d = Array3::zeros((n_e, height, width));
        let mut u_3d = Array3::zeros((n_e, height, width));
        for y in 0..height {
            for x in 0..width {
                for (i, (&t, &s)) in t_1d.iter().zip(sigma_1d.iter()).enumerate() {
                    t_3d[[i, y, x]] = t;
                    u_3d[[i, y, x]] = s;
                }
            }
        }
        (t_3d, u_3d)
    }

    /// Build a 4x4 synthetic transmission stack from known density.
    fn synthetic_4x4_transmission(
        res_data: &nereids_endf::resonance::ResonanceData,
        true_density: f64,
        energies: &[f64],
    ) -> (Array3<f64>, Array3<f64>) {
        synthetic_grid_transmission(res_data, true_density, energies, 4, 4)
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

    /// A caller-supplied precomputed cross-section stack with the wrong shape
    /// must be rejected up front (before the rayon loop), not panic on
    /// `xs[0]` in the σ_eff collapse / forward-model builder or be swallowed
    /// per-pixel as `n_failed`.
    #[test]
    fn test_spatial_map_rejects_wrong_shape_precomputed_cross_sections() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..21).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);

        // 1 isotope → 1 σ row expected; inject 2 rows of the right length.
        let n_e = energies.len();
        let bad_xs = Arc::new(vec![vec![1.0; n_e], vec![1.0; n_e]]);
        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_precomputed_cross_sections(bad_xs);

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let err = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("wrong-shape precomputed XS must be rejected up front");
        assert!(
            matches!(err, PipelineError::ShapeMismatch(_)),
            "expected ShapeMismatch, got {err:?}"
        );
    }

    /// Mid-run cancellation must return `Err(Cancelled)`, not a partial
    /// `Ok(SpatialResult)` whose cancelled pixels are left as NaN holes
    /// (indistinguishable from genuine fit failures, with no signal to a
    /// non-GUI caller that the map is incomplete).
    ///
    /// The previous post-loop guard only fired when cancellation beat *every*
    /// pixel (`results.is_empty()`); a cancellation that lands after the first
    /// pixel completes slipped through and produced a partial map.  This test
    /// reproduces exactly that: a watcher thread flips `cancel` as soon as the
    /// `progress` counter shows the first pixel finished, while the remaining
    /// pixels are still fitting.  The pre-loop guard sees `cancel == false`
    /// (so it does not short-circuit), pixels complete into `results`, and the
    /// post-loop guard then observes `cancel == true` with `results`
    /// non-empty.
    #[test]
    fn test_spatial_map_mid_run_cancellation_returns_err() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};

        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        // A wide grid: many real LM fits, so the watcher reliably flips
        // `cancel` mid-run (after pixel 1, with dozens of pixels left to skip).
        let (t_3d, u_3d) = synthetic_grid_transmission(&data, 0.0005, &energies, 1, 64);

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

        let cancel = AtomicBool::new(false);
        let progress = AtomicUsize::new(0);

        let result = std::thread::scope(|s| {
            // Watcher: once at least one pixel has finished, request
            // cancellation while the rest are still being fit.
            s.spawn(|| {
                while progress.load(Ordering::Relaxed) < 1 {
                    std::hint::spin_loop();
                }
                cancel.store(true, Ordering::Relaxed);
            });
            spatial_map_typed(&input, &config, None, Some(&cancel), Some(&progress))
        });

        assert!(
            matches!(result, Err(PipelineError::Cancelled)),
            "mid-run cancellation must return Err(Cancelled), got {result:?}"
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

    /// Issue #608 (R4): the GAUSSIAN-resolution spatial path — `spatial_map_typed`'s
    /// `aux_grid_active` branch (work σ via `broadened_cross_sections_on_working_grid`,
    /// per-pixel injection through `with_precomputed_work_cross_sections`) plus
    /// `build_transmission_model`'s working-grid selection — is the bulk of the
    /// #608 wiring but had no integration test (only the Tabulated/plan path,
    /// above, was covered).  Mirror that test with `ResolutionFunction::Gaussian`,
    /// data generated by `forward_model` WITH the same Gaussian (so the fit can
    /// recover density), a ‖kernel − none‖ non-vacuity pre-check, and per-pixel
    /// density recovery + determinism assertions.
    #[test]
    fn test_spatial_map_typed_gaussian_aux_grid_recovers_density() {
        use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};
        use nereids_physics::transmission::{SampleParams, forward_model};

        let data = u238_single_resonance(); // resonance @ ~6.674 eV
        let true_density = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        // Synthetic data CONSISTENT with a Gaussian-broadened forward model, so
        // the fit (which also broadens on the aux grid) can recover the density.
        let sample = SampleParams::new(temperature, vec![(data.clone(), true_density)]).unwrap();
        let t_1d = forward_model(&energies, &sample, Some(&inst)).unwrap();

        // ‖kernel − none‖ non-vacuity: the Gaussian must broaden the spectrum,
        // else the aux-grid path is a no-op and the test is vacuous.
        let t_none = forward_model(&energies, &sample, None).unwrap();
        let broaden = t_1d
            .iter()
            .zip(t_none.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            broaden > 1e-4,
            "Gaussian kernel must broaden the spectrum non-trivially (got {broaden:.3e})"
        );

        // Replicate to a 4x4 cube — identical pixels double as a determinism check.
        let n_e = energies.len();
        let sigma_1d: Vec<f64> = t_1d.iter().map(|&v| 0.01 * v.max(0.01)).collect();
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

        let config = UnifiedFitConfig::new(
            energies.clone(),
            vec![data],
            vec!["U-238".into()],
            temperature,
            Some(ResolutionFunction::Gaussian(
                ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            )),
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
        assert!(
            result.n_converged >= 14,
            "Gaussian aux-grid path: {} / 16 pixels converged",
            result.n_converged,
        );

        // Per-pixel density recovery against the forward_model-generated synthetic.
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
            "Gaussian aux-grid mean density: {mean}, true: {true_density}"
        );

        // Determinism: identical pixels ⇒ bit-equal density across the rayon
        // fanout (catches aux-grid work-σ / layout state leaking across pixels).
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
                    "aux-grid path leaked pixel-specific state: density cell {cell} != reference {reference}"
                );
            }
        }
    }

    /// Issue #608 (PR #609 coverage): `spatial_map_typed`'s `Some(cached)` +
    /// aux-grid arm — when a caller PRE-SUPPLIES data-grid σ AND a Gaussian aux
    /// grid is active, the working-grid σ is recomputed from resonance data (the
    /// cached data σ cannot be de-extracted back onto the aux grid).  The
    /// sibling Gaussian test exercises the `None` arm; this supplies precomputed
    /// σ to hit the `Some(cached)` arm.
    #[test]
    fn test_spatial_map_typed_gaussian_aux_grid_with_precomputed_sigma() {
        use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};
        use nereids_physics::transmission::{
            SampleParams, broadened_cross_sections, forward_model,
        };

        let data = u238_single_resonance();
        let true_density = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });
        let sample = SampleParams::new(temperature, vec![(data.clone(), true_density)]).unwrap();
        let t_1d = forward_model(&energies, &sample, Some(&inst)).unwrap();
        let n_e = energies.len();
        let sigma_1d: Vec<f64> = t_1d.iter().map(|&v| 0.01 * v.max(0.01)).collect();
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
        // Pre-supply the Doppler-broadened, data-grid σ ⇒ the Some(cached) arm.
        let data_sigma = broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            None,
            None,
        )
        .unwrap();
        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            temperature,
            Some(ResolutionFunction::Gaussian(
                ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            )),
            vec![0.001],
        )
        .unwrap()
        .with_precomputed_cross_sections(Arc::new(data_sigma))
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));
        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_total, 16);
        assert!(
            result.n_converged >= 14,
            "Some(cached)+aux path: {} / 16 pixels converged",
            result.n_converged,
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
            "Some(cached)+aux mean density: {mean}, true: {true_density}"
        );
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

    /// Counts-KL + `fit_alpha_2=true` (and the symmetric `fit_alpha_1`
    /// case) is a whole-config rejection that fires identically on
    /// every pixel.  Previously this test codified the silent swallow:
    /// the spatial layer returned `Ok(SpatialResult)` with `n_failed =
    /// n_total` and an all-NaN density map, hiding the actionable
    /// `joint-Poisson does not support fit_alpha_*` diagnostic from
    /// the caller.  After the preflight hoist, the spatial call
    /// surfaces the same `Err(InvalidParameter)` the single-spectrum
    /// fitter would have raised — Python maps it to `PyValueError`.
    #[test]
    fn test_spatial_map_rejects_counts_kl_alpha_up_front() {
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

        let err = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("counts-KL with fit_alpha_2 must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("fit_alpha_1") || msg.contains("fit_alpha_2"),
            "error must name the offending flag, got: {msg}"
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
        if let Some(ref m) = result.back_d_map {
            let v = m[[uy, ux]];
            assert!(
                v.is_nan(),
                "back_d_map at unconverged pixel ({uy},{ux}) must be NaN, got {v}"
            );
        }
        if let Some(ref m) = result.back_f_map {
            let v = m[[uy, ux]];
            assert!(
                v.is_nan(),
                "back_f_map at unconverged pixel ({uy},{ux}) must be NaN, got {v}"
            );
        }
    }

    /// `back_d_map` / `back_f_map` stay `None` whenever `fit_back_d` /
    /// `fit_back_f` are left at their defaults, even when a
    /// transmission background config is attached.  This is the
    /// "exponential tail never engaged" arm of the gating contract.
    #[test]
    fn test_spatial_map_back_d_f_maps_none_when_fit_disabled() {
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
            vec![0.001],
        )
        .unwrap()
        // background=true but fit_back_d/fit_back_f are left at their
        // default `false` — back_*_map must remain None.
        .with_transmission_background(crate::pipeline::BackgroundConfig::default());

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(
            result.background_maps.is_some(),
            "background_maps should be Some when transmission_background is attached"
        );
        assert!(
            result.back_d_map.is_none(),
            "back_d_map must be None when fit_back_d=false"
        );
        assert!(
            result.back_f_map.is_none(),
            "back_f_map must be None when fit_back_f=false"
        );
    }

    /// `back_d_map` / `back_f_map` are `Some` (and carry finite values
    /// at converged pixels) when the LM transmission background is fit
    /// with both exponential-tail flags set.  Synthesises a 4×4 cube
    /// with a known exponential tail on top of U-238 absorption so the
    /// BackD/BackF Jacobian columns are not degenerate (a smooth
    /// resonance-only model is unidentifiable in BackD/BackF — `anorm`
    /// absorbs them — so the fitter stagnates and converges = false on
    /// every pixel).  Mirrors the single-spectrum coverage in
    /// `fitting::transmission_model::tests::exponential_fit_recovers_all_params`
    /// while exercising the spatial aggregation path.
    #[test]
    fn test_spatial_map_back_d_f_maps_some_when_fit_enabled() {
        let rd = u238_single_resonance();
        // 101-bin grid (matches `test_spatial_map_typed_transmission_lm`).
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let true_density = 0.0005;
        let true_back_d = 0.03;
        let true_back_f = 2.0;
        // Build the resonance-only transmission first, then add the
        // exponential tail in-place so the fitter sees a model whose
        // BackD/BackF columns carry non-degenerate signal.  The 1/√E
        // factor (NormalizedTransmissionModel exponential wrapper)
        // makes BackD/BackF identifiable across the [1, 11] eV range.
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&rd, true_density, &energies);
        for (i, &e) in energies.iter().enumerate() {
            let inv_sqrt_e = 1.0 / e.sqrt();
            let tail = true_back_d * (-true_back_f * inv_sqrt_e).exp();
            for y in 0..4 {
                for x in 0..4 {
                    t_3d[[i, y, x]] += tail;
                }
            }
        }
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        // SAMMY pairs BackD/BackF — `validate_transmission_background`
        // rejects fitting only one.  Both initial values must stay
        // strictly positive (the BackF Jacobian column zeros out when
        // BackD ≈ 0; see BackgroundConfig docstring).
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_d: true,
            fit_back_f: true,
            back_d_init: 0.01,
            back_f_init: 1.0,
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![true_density],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig {
            max_iter: 500,
            ..LmConfig::default()
        }))
        .with_transmission_background(bg);

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        let bd = result
            .back_d_map
            .as_ref()
            .expect("back_d_map should be Some when fit_back_d=true");
        let bf = result
            .back_f_map
            .as_ref()
            .expect("back_f_map should be Some when fit_back_f=true");
        assert_eq!(bd.shape(), [4, 4]);
        assert_eq!(bf.shape(), [4, 4]);
        assert!(
            result.n_converged > 0,
            "no pixels converged with LM + 7-param transmission background \
             on synthetic data carrying an exponential tail — test fixture \
             is no longer exercising the gating contract"
        );
        // At converged pixels both must be finite; at unconverged pixels
        // the NaN-on-failure contract leaves them NaN.
        let mut n_finite_d = 0;
        let mut n_finite_f = 0;
        for y in 0..4 {
            for x in 0..4 {
                if result.converged_map[[y, x]] {
                    if bd[[y, x]].is_finite() {
                        n_finite_d += 1;
                    }
                    if bf[[y, x]].is_finite() {
                        n_finite_f += 1;
                    }
                } else {
                    assert!(
                        bd[[y, x]].is_nan(),
                        "back_d_map at unconverged ({y},{x}) must be NaN"
                    );
                    assert!(
                        bf[[y, x]].is_nan(),
                        "back_f_map at unconverged ({y},{x}) must be NaN"
                    );
                }
            }
        }
        // At least one converged pixel must populate finite back_d/back_f
        // — otherwise the gating is vacuous.
        assert!(
            n_finite_d > 0 && n_finite_f > 0,
            "at least one converged pixel must produce finite back_d/back_f \
             (n_converged={}, n_finite_d={n_finite_d}, n_finite_f={n_finite_f})",
            result.n_converged
        );
    }

    /// Counts-KL never fits the exponential tail, so `back_d_map` /
    /// `back_f_map` must remain `None` even when the counts-KL
    /// background is attached.  Keeps the joint-Poisson dispatch from
    /// accidentally surfacing a map of sentinel zeros.
    #[test]
    fn test_spatial_map_counts_kl_back_d_f_maps_are_none() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
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
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_counts_background(crate::pipeline::CountsBackgroundConfig::default());
        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(
            result.back_d_map.is_none(),
            "back_d_map must be None on the counts-KL path"
        );
        assert!(
            result.back_f_map.is_none(),
            "back_f_map must be None on the counts-KL path"
        );
    }

    /// Unpaired `fit_back_d` / `fit_back_f` must be rejected up-front
    /// by `spatial_map_typed`, not just per-pixel.  Without this guard
    /// the per-pixel solver errors are swallowed as `n_failed` and the
    /// caller sees an all-NaN map with no diagnostic.
    #[test]
    fn test_spatial_map_back_d_f_unpaired_rejected_up_front() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_d: true,
            fit_back_f: false, // unpaired — must be rejected
            back_d_init: 0.01,
            back_f_init: 1.0,
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_transmission_background(bg);
        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("unpaired fit_back_d/fit_back_f must be rejected up-front");
        let msg = err.to_string();
        assert!(
            msg.contains("fit_back_d") && msg.contains("fit_back_f"),
            "error message must reference both fit flags, got: {msg}"
        );
    }

    /// Non-positive `back_d_init` is rejected up-front so the LM
    /// solver does not silently produce a degenerate Jacobian (BackF's
    /// column zeros out at BackD ≈ 0).
    #[test]
    fn test_spatial_map_back_d_init_non_positive_rejected() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_d: true,
            fit_back_f: true,
            back_d_init: 0.0, // non-positive — must be rejected
            back_f_init: 1.0,
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_transmission_background(bg);
        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("back_d_init=0.0 with fit_back_d=true must be rejected up-front");
        assert!(
            err.to_string().contains("back_d_init"),
            "error must reference back_d_init, got: {err}"
        );
    }

    /// Non-positive `back_f_init` is rejected up-front for the same
    /// reason as `back_d_init` (BackD becomes a duplicate of BackA at
    /// BackF ≈ 0).
    #[test]
    fn test_spatial_map_back_f_init_non_positive_rejected() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_d: true,
            fit_back_f: true,
            back_d_init: 0.01,
            back_f_init: -1.0, // negative — must be rejected
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_transmission_background(bg);
        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("back_f_init=-1.0 with fit_back_f=true must be rejected up-front");
        assert!(
            err.to_string().contains("back_f_init"),
            "error must reference back_f_init, got: {err}"
        );
    }

    /// NaN `back_d_init` is rejected up-front.  Without the
    /// `is_finite()` guard, NaN passes the `<= 0.0` check (NaN
    /// comparisons are always false) and propagates into the fit
    /// parameters.
    #[test]
    fn test_spatial_map_back_d_init_nan_rejected() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_d: true,
            fit_back_f: true,
            back_d_init: f64::NAN, // NaN — must be rejected
            back_f_init: 1.0,
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_transmission_background(bg);
        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("NaN back_d_init must be rejected up-front");
        let msg = err.to_string();
        assert!(
            msg.contains("back_d_init") && (msg.contains("finite") || msg.contains("NaN")),
            "error must mention finite/NaN for back_d_init, got: {msg}"
        );
    }

    /// +inf `back_f_init` is rejected up-front.  Without the
    /// `is_finite()` guard, +inf passes the `<= 0.0` check (positive
    /// infinity is > 0) and propagates into the fit parameters.
    #[test]
    fn test_spatial_map_back_f_init_inf_rejected() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_d: true,
            fit_back_f: true,
            back_d_init: 0.01,
            back_f_init: f64::INFINITY, // +inf — must be rejected
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_transmission_background(bg);
        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("+inf back_f_init must be rejected up-front");
        let msg = err.to_string();
        assert!(
            msg.contains("back_f_init") && (msg.contains("finite") || msg.contains("inf")),
            "error must mention finite/inf for back_f_init, got: {msg}"
        );
    }

    /// The joint-Poisson (counts-KL) dispatch combined with a
    /// `transmission_background` carrying `fit_back_d=true` /
    /// `fit_back_f=true` is rejected up-front so the user gets a clear
    /// diagnostic instead of an all-NaN map from per-pixel `n_failed`
    /// swallowing.
    #[test]
    fn test_spatial_map_counts_kl_plus_back_d_rejected_up_front() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
        let data = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_d: true,
            fit_back_f: true,
            back_d_init: 0.01,
            back_f_init: 1.0,
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_transmission_background(bg);
        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("counts-KL + fit_back_d/fit_back_f must be rejected up-front");
        let msg = err.to_string();
        assert!(
            msg.contains("counts-KL") || msg.contains("joint-Poisson"),
            "error must reference the counts-KL incompatibility, got: {msg}"
        );
    }

    /// `CountsWithNuisance + LM` is rejected up-front so the caller
    /// does not get an all-NaN spatial result from per-pixel `n_failed`
    /// swallowing.  `fit_spectrum_typed` rejects this combo per-pixel;
    /// the hoisted spatial-level rejection surfaces the same diagnostic
    /// at the boundary instead of pretending the fit ran.
    #[test]
    fn test_spatial_map_counts_with_nuisance_plus_lm_rejected_up_front() {
        use ndarray::Array3;
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, _ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
        // `CountsWithNuisance` carries (sample, flux, background) per
        // pixel.  The validation under test fires before any field is
        // consumed, so synthetic flat 4x4 arrays suffice.
        let flux: Array3<f64> = Array3::from_elem((energies.len(), 4, 4), 1000.0);
        let background: Array3<f64> = Array3::from_elem((energies.len(), 4, 4), 0.0);
        let data = InputData3D::CountsWithNuisance {
            sample_counts: sample.view(),
            flux: flux.view(),
            background: background.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));
        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("CountsWithNuisance + LM must be rejected up-front");
        let msg = err.to_string();
        assert!(
            msg.contains("CountsWithNuisance") && msg.contains("counts-domain"),
            "error must mention CountsWithNuisance + counts-domain requirement, got: {msg}"
        );
    }

    /// Diagnostic-priority regression: when a config violates both a
    /// dispatch-level guard (e.g. `CountsWithNuisance + LM` is
    /// rejected because LM cannot consume the nuisance arm) AND a
    /// downstream preflight gate (e.g. `fit_energy_range` selects too
    /// few active bins), the user must see the *dispatch* mismatch
    /// first — the fit-range / temperature gates only meaningfully
    /// apply once the dispatch is known to be valid.  Otherwise an
    /// "LM transmission active-bin" message shadows the more
    /// fundamental "requires a counts-domain solver" diagnostic.
    #[test]
    fn test_spatial_map_reports_solver_mismatch_before_fit_range_gate() {
        use ndarray::Array3;
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, _ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
        let flux: Array3<f64> = Array3::from_elem((energies.len(), 4, 4), 1000.0);
        let background: Array3<f64> = Array3::from_elem((energies.len(), 4, 4), 0.0);
        let data = InputData3D::CountsWithNuisance {
            sample_counts: sample.view(),
            flux: flux.view(),
            background: background.view(),
        };
        // Combine the dispatch-level violation (LM + CountsWithNuisance)
        // with a downstream preflight violation (too-narrow
        // `fit_energy_range` selecting < 2 active bins on the configured
        // 0.2 eV grid).  Either guard could fire, but the dispatch
        // mismatch is the actionable cause; the fit-range gate would
        // never matter because the dispatch never reaches LM with this
        // input.
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_energy_range(Some((5.0, 5.05)))
        .unwrap();

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("CountsWithNuisance + LM + narrow fit_energy_range must be rejected");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("CountsWithNuisance") && msg.contains("counts-domain"),
            "error must surface the solver mismatch (not the fit-range gate), got: {msg}"
        );
        assert!(
            !msg.contains("active bin"),
            "error must not be the downstream fit-range diagnostic, got: {msg}"
        );
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

    // ── NV-6 preflight hoist regression tests ────────────────────────
    //
    // Each of these constructs a whole-config rejection that the
    // single-spectrum fitter would raise per-pixel.  Before the
    // hoist, `spatial_map_typed` swallowed those errors at the rayon
    // closure and returned `Ok(SpatialResult)` with `n_failed =
    // n_total`, an all-NaN density map, and no diagnostic.  The fix
    // wires `validate_spatial_fit_preflight` immediately after shape
    // validation so every gate below surfaces as a single
    // `Err(PipelineError::InvalidParameter)`.

    #[test]
    fn test_spatial_map_rejects_fit_temperature_below_one_up_front() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        // Sub-1 K initial temperature with `fit_temperature=true` is
        // rejected by `fit_spectrum_typed` per-pixel.  Pick 0.5 K
        // (the canonical "user wrote 25 meV instead of 25 K" case).
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.5,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_temperature(true);

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("fit_temperature with temperature_k < 1.0 must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("temperature") && msg.contains("1.0"),
            "error must mention the 1.0 K floor, got: {msg}"
        );
    }

    #[test]
    fn test_spatial_map_transmission_poisson_rejects_fit_energy_range_up_front() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        // Transmission + Poisson-KL + any `fit_energy_range` is
        // unsupported because the transmission-domain `poisson_fit`
        // does not honour the active mask.  The per-pixel rejection
        // would otherwise silently produce an all-NaN map.
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_fit_energy_range(Some((2.0, 8.0)))
        .unwrap();

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("transmission + Poisson-KL + fit_energy_range must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("fit_energy_range") && msg.contains("Poisson-KL"),
            "error must name the incompatibility, got: {msg}"
        );
    }

    #[test]
    fn test_spatial_map_lm_rejects_too_narrow_fit_energy_range_up_front() {
        let rd = u238_single_resonance();
        // Energies 1, 1.2, 1.4, ..., 11.0 → 51 bins on a 0.2 eV grid.
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        // Window narrower than one bin → at most one active bin on the
        // grid; LM transmission needs at least 2.
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_energy_range(Some((5.0, 5.05)))
        .unwrap();

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("LM with too-narrow fit_energy_range must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("active bin") && msg.contains("LM transmission"),
            "error must mention narrow active-bin count for the LM path, got: {msg}"
        );
    }

    #[test]
    fn test_spatial_map_counts_kl_rejects_too_narrow_fit_energy_range_up_front() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
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
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_fit_energy_range(Some((5.0, 5.05)))
        .unwrap();

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("counts-KL with too-narrow fit_energy_range must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("active bin") && msg.contains("joint-Poisson"),
            "error must mention narrow active-bin count for the joint-Poisson path, got: {msg}"
        );
    }

    #[test]
    fn test_spatial_map_counts_kl_rejects_invalid_c_up_front() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
        let data = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        // Non-positive `c` (`Q_s/Q_ob`) is invalid for the counts-KL
        // dispatch.  Python pre-validates this at the binding
        // boundary, but Rust core callers reach the per-pixel
        // rejection — which the spatial layer used to swallow.
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_counts_background(crate::pipeline::CountsBackgroundConfig {
            c: -1.0,
            ..Default::default()
        });

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("counts-KL with non-positive c must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("finite c > 0"),
            "error must mention the c > 0 requirement, got: {msg}"
        );
    }

    #[test]
    fn test_spatial_map_counts_kl_requires_back_a_for_back_b_c_up_front() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
        let data = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        // B_B fitted but B_A not fitted is rejected by the
        // joint-Poisson dispatch (memo 35 §P2.2): A_n alone cannot
        // absorb a constant offset.  Test the B_B branch; the B_C
        // branch shares the same code path.
        let bg = crate::pipeline::BackgroundConfig {
            fit_back_a: false,
            fit_back_b: true,
            fit_back_c: false,
            fit_back_d: false,
            fit_back_f: false,
            ..crate::pipeline::BackgroundConfig::default()
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_transmission_background(bg);

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("counts-KL with B_B but no B_A must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(
            msg.contains("B_A") && msg.contains("fit_back_a"),
            "error must name the B_A requirement, got: {msg}"
        );
    }

    /// Underdetermined-system rejection: a `fit_energy_range` window
    /// that selects fewer active bins than the dispatch has free
    /// parameters must be rejected up-front with a diagnostic that
    /// names the underdetermined condition.  Before this guard, a
    /// config with many free params (densities + temperature +
    /// background) and a too-narrow window passed the old
    /// `n_active < 2` floor but every per-pixel fit returned
    /// non-converged, producing the silent all-NaN spatial result
    /// that the rest of the preflight exists to eliminate.
    #[test]
    fn test_spatial_map_rejects_underdetermined_fit_range() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        // Free-parameter count for this config:
        //   1 density + fit_temperature (=1) + fit_anorm + fit_back_a
        //   + fit_back_b + fit_back_c (=4 background flags from the
        //   BackgroundConfig::default())  →  n_free = 6.
        // The fit_energy_range window [5.0, 5.5] picks up the grid
        // points 5.0, 5.2, 5.4 → 3 active bins, comfortably above
        // the legacy `n_active < 2` floor but below `n_free`, so the
        // problem is structurally underdetermined and the LM core
        // would return `converged=false` for every pixel.
        let bg = crate::pipeline::BackgroundConfig::default();
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            // fit_temperature requires temperature_k >= 1.0; pick a
            // physically reasonable value so the temperature gate
            // does not pre-empt the underdetermined-system gate.
            293.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_temperature(true)
        .with_transmission_background(bg)
        .with_fit_energy_range(Some((5.0, 5.5)))
        .unwrap();

        let err = spatial_map_typed(&data, &config, None, None, None)
            .expect_err("underdetermined fit_energy_range must be rejected up-front");
        let msg = err.to_string();
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        // Diagnostic must name both the active-bin count and the
        // free-parameter requirement so the user can see *why* their
        // window is too narrow.
        assert!(
            msg.contains("active bin")
                && msg.contains("free parameter")
                && msg.contains("underdetermined"),
            "error must explain the underdetermined condition, got: {msg}"
        );
    }

    // ── Up-front detector-cube VALUE validation ─────────────────────────
    //
    // These tests exercise bad *values* (NaN / +inf / negative / zero σ) in
    // each detector cube — the path `validate_spatial_data_values` guards.
    // Before that guard existed the per-pixel `v.max(0.0)` / `σ.max(1e-10)`
    // clamps silently transformed bad input into a plausible-but-wrong or
    // all-NaN map; the asserts below lock in a hard `InvalidParameter`
    // instead.  The earlier spatial tests cover only bad *config*.

    fn lm_transmission_config(
        energies: Vec<f64>,
        data: nereids_endf::resonance::ResonanceData,
    ) -> UnifiedFitConfig {
        UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
    }

    fn kl_counts_config(
        energies: Vec<f64>,
        data: nereids_endf::resonance::ResonanceData,
    ) -> UnifiedFitConfig {
        UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
    }

    #[test]
    fn test_spatial_rejects_bad_transmission_value() {
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let data = u238_single_resonance();
            let (mut t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);
            t_3d[[10, 1, 2]] = bad;
            let config = lm_transmission_config(energies.clone(), data);
            let input = InputData3D::Transmission {
                transmission: t_3d.view(),
                uncertainty: u_3d.view(),
            };
            let err = spatial_map_typed(&input, &config, None, None, None)
                .expect_err("non-finite transmission value must be rejected up-front");
            assert!(
                matches!(err, PipelineError::InvalidParameter(_)),
                "got {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains("transmission") && msg.contains("(y="),
                "error must name the cube and (y, x, e): {msg}"
            );
        }
    }

    #[test]
    fn test_spatial_rejects_bad_uncertainty() {
        // NaN / +inf / zero / negative σ are all rejected (finite and > 0):
        // a zero σ is a singular weight, and the old floor turned it into a
        // 1e20 maximum-confidence bin.
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let data = u238_single_resonance();
            let (t_3d, mut u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);
            u_3d[[9, 1, 0]] = bad;
            let config = lm_transmission_config(energies.clone(), data);
            let input = InputData3D::Transmission {
                transmission: t_3d.view(),
                uncertainty: u_3d.view(),
            };
            let err = spatial_map_typed(&input, &config, None, None, None)
                .expect_err("bad uncertainty must be rejected up-front");
            assert!(
                matches!(err, PipelineError::InvalidParameter(_)),
                "got {err:?}"
            );
            assert!(
                err.to_string().contains("uncertainty"),
                "error must name the uncertainty cube, got: {err}"
            );
        }
    }

    #[test]
    fn test_spatial_accepts_negative_transmission_value() {
        // SAMMY does not reject negative transmission (noise / open-beam
        // over-subtraction); only finiteness is required.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);
        t_3d[[12, 2, 2]] = -0.05;
        let config = lm_transmission_config(energies, data);
        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let result = spatial_map_typed(&input, &config, None, None, None)
            .expect("a finite negative transmission value must not be rejected");
        assert_eq!(result.n_total, 16);
    }

    #[test]
    fn test_spatial_rejects_bad_sample_counts() {
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let data = u238_single_resonance();
            let (mut sample, ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
            sample[[8, 0, 3]] = bad;
            let config = kl_counts_config(energies.clone(), data);
            let input = InputData3D::Counts {
                sample_counts: sample.view(),
                open_beam_counts: ob.view(),
            };
            let err = spatial_map_typed(&input, &config, None, None, None)
                .expect_err("bad sample count must be rejected up-front");
            assert!(
                matches!(err, PipelineError::InvalidParameter(_)),
                "got {err:?}"
            );
            assert!(
                err.to_string().contains("sample_counts"),
                "error must name the sample_counts cube, got: {err}"
            );
        }
    }

    #[test]
    fn test_spatial_rejects_bad_open_beam() {
        // A single bad open-beam bin would otherwise poison the spatially-
        // averaged flux for ALL pixels (KL path); it must surface as a hard
        // error rather than a silently all-NaN map.
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let data = u238_single_resonance();
            let (sample, mut ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
            ob[[6, 3, 1]] = bad;
            let config = kl_counts_config(energies.clone(), data);
            let input = InputData3D::Counts {
                sample_counts: sample.view(),
                open_beam_counts: ob.view(),
            };
            let err = spatial_map_typed(&input, &config, None, None, None)
                .expect_err("bad open-beam must be rejected up-front");
            assert!(
                matches!(err, PipelineError::InvalidParameter(_)),
                "got {err:?}"
            );
            assert!(
                err.to_string().contains("open_beam_counts"),
                "error must name the open_beam_counts cube, got: {err}"
            );
        }
    }

    #[test]
    fn test_spatial_counts_with_nuisance_rejects_bad_flux() {
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        for bad in [f64::NAN, -1.0] {
            let data = u238_single_resonance();
            let (sample, _ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
            let mut flux = Array3::from_elem((energies.len(), 4, 4), 1000.0);
            let background = Array3::from_elem((energies.len(), 4, 4), 0.0);
            flux[[4, 2, 1]] = bad;
            let config = kl_counts_config(energies.clone(), data);
            let input = InputData3D::CountsWithNuisance {
                sample_counts: sample.view(),
                flux: flux.view(),
                background: background.view(),
            };
            let err = spatial_map_typed(&input, &config, None, None, None)
                .expect_err("bad flux must be rejected up-front");
            assert!(
                matches!(err, PipelineError::InvalidParameter(_)),
                "got {err:?}"
            );
            assert!(
                err.to_string().contains("flux"),
                "error must name the flux cube, got: {err}"
            );
        }
    }

    #[test]
    fn test_spatial_counts_with_nuisance_rejects_nonfinite_background() {
        // Background is validated finite (sign deferred to the per-pixel
        // detector-background gate); this closes the `NaN.abs() > 1e-12 ==
        // false` finiteness leak in that gate at the boundary.
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        for bad in [f64::NAN, f64::INFINITY] {
            let data = u238_single_resonance();
            let (sample, _ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
            let flux = Array3::from_elem((energies.len(), 4, 4), 1000.0);
            let mut background = Array3::from_elem((energies.len(), 4, 4), 0.0);
            background[[2, 3, 3]] = bad;
            let config = kl_counts_config(energies.clone(), data);
            let input = InputData3D::CountsWithNuisance {
                sample_counts: sample.view(),
                flux: flux.view(),
                background: background.view(),
            };
            let err = spatial_map_typed(&input, &config, None, None, None)
                .expect_err("non-finite background must be rejected up-front");
            assert!(
                matches!(err, PipelineError::InvalidParameter(_)),
                "got {err:?}"
            );
            assert!(
                err.to_string().contains("background"),
                "error must name the background cube, got: {err}"
            );
        }
    }

    #[test]
    fn test_spatial_transmission_tolerates_nan_in_inactive_bin() {
        // A NaN in an out-of-`fit_energy_range` (inactive) bin is legitimate
        // (transmission is undefined where open-beam → 0) and is skipped by
        // the LM core, so it must NOT be rejected — the canonical "set
        // fit_energy_range to exclude a bad region" workflow.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);
        // energies[0] = 1.0 eV is below E_min = 3.0 → inactive.
        t_3d[[0, 1, 1]] = f64::NAN;
        let config = lm_transmission_config(energies, data)
            .with_fit_energy_range(Some((3.0, 9.0)))
            .unwrap();
        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let result = spatial_map_typed(&input, &config, None, None, None)
            .expect("NaN in an inactive (out-of-range) bin must be tolerated");
        assert!(
            result.n_converged > 0,
            "the active-bin fit should still converge"
        );
    }

    #[test]
    fn test_spatial_rejects_nan_transmission_in_active_bin_with_range() {
        // The mirror of the previous test: a NaN inside the active window
        // must still be rejected.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);
        // energies[20] = 5.0 eV is inside [3.0, 9.0] → active.
        t_3d[[20, 0, 0]] = f64::NAN;
        let config = lm_transmission_config(energies, data)
            .with_fit_energy_range(Some((3.0, 9.0)))
            .unwrap();
        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let err = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("NaN in an active bin must be rejected up-front");
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("transmission"),
            "error must name the transmission cube, got: {err}"
        );
    }

    #[test]
    fn test_spatial_accepts_bad_value_in_dead_pixel() {
        // A `dead_pixels`-masked pixel is never read, so detector garbage in
        // it must not reject the whole map (live-pixels-only validation).
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (mut sample, ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
        sample[[5, 0, 0]] = f64::NAN;
        let config = kl_counts_config(energies, data);
        let mut dead = Array2::from_elem((4, 4), false);
        dead[[0, 0]] = true;
        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let result = spatial_map_typed(&input, &config, Some(&dead), None, None)
            .expect("a bad value in a dead-masked pixel must be tolerated");
        assert!(
            result.n_converged > 0,
            "the remaining live pixels should still fit"
        );
    }

    #[test]
    fn test_spatial_accepts_zero_counts_and_open_beam() {
        // Zero is legitimate ("no counts in this bin"); the joint-Poisson
        // xlogy_ratio zero-branch handles it.  Must not be rejected.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (mut sample, mut ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
        sample[[3, 2, 2]] = 0.0;
        ob[[7, 1, 1]] = 0.0;
        let config = kl_counts_config(energies, data);
        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let result = spatial_map_typed(&input, &config, None, None, None)
            .expect("zero counts / zero open-beam are legitimate and must not be rejected");
        assert_eq!(result.n_total, 16);
    }

    #[test]
    fn test_spatial_rejects_open_beam_flux_overflow() {
        // Each open-beam bin is individually finite (passes the up-front
        // FiniteNonNegative check), but summing `f64::MAX` across live pixels
        // overflows the spatially-averaged flux to +inf.  That must surface as
        // an up-front `InvalidParameter` rather than a silently all-NaN map
        // (the averaged flux would otherwise fail inside each per-pixel fit and
        // be swallowed as `n_failed`).
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, mut ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
        for y in 0..4 {
            for x in 0..4 {
                ob[[5, y, x]] = f64::MAX;
            }
        }
        let config = kl_counts_config(energies, data);
        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let err = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("an overflowing averaged open-beam flux must be rejected up-front");
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("averaged open-beam flux"),
            "error must name the averaged-flux overflow, got: {err}"
        );
    }
}
