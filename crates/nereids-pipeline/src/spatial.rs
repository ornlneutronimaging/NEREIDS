//! Spatial mapping: per-pixel fitting with rayon parallelization.
//!
//! Applies the single-spectrum fitting pipeline across all pixels in
//! a hyperspectral neutron imaging dataset to produce 2D composition maps.

use ndarray::{Array2, Array3, ArrayView3, s};
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
    /// deviance) this is back-compat-mirrored to
    /// `D/(n−k)`; the semantically-correct per-pixel value is also
    /// exposed as [`Self::deviance_per_dof_map`].
    /// NaN at pixels where `converged_map` is `false`.
    pub chi_squared_map: Array2<f64>,
    /// Per-pixel conditional binomial deviance `D/(n−k)` map.  `Some` when
    /// the effective per-pixel solver is the counts-KL dispatch
    /// (joint-Poisson); `None` for LM transmission runs.
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
    ///
    /// **Covariance-only lower bound.** For the raw-covariance solver paths
    /// (joint-Poisson) each σ_T is the square root of the
    /// temperature entry of the inverse curvature (Fisher) matrix at the
    /// converged point. That is a *lower bound* on the true uncertainty: it
    /// captures only the statistical curvature and omits baseline/model
    /// mis-specification noise, so on real data it can **underestimate the
    /// observed per-superpixel scatter by ~3–4×**. Enable
    /// `UnifiedFitConfig::scale_by_chi2` to inflate σ_T by `sqrt` of the
    /// goodness-of-fit this result reports (Gaussian `reduced_chi_squared` on the
    /// transmission LM path, `deviance_per_dof` on the counts joint-Poisson path)
    /// for a goodness-of-fit-scaled estimate. The LM transmission path is already
    /// χ²-scaled (Numerical Recipes §15.6), so the flag is a no-op there.
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
    /// (legacy alpha-fitting `[b0, b1, alpha_2]` layout was retired
    /// together with `fit_counts_poisson`).
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
    /// Nominal flight path (m) the energy-scale fit was configured with —
    /// recorded AT FIT TIME so downstream consumers (e.g. the GUI overlay's
    /// per-pixel `SpectrumFitResult::corrected_energies`) reproduce the
    /// transform with the fit's own flight path even if the live beamline
    /// setting is edited afterwards (issue #634 review).  `Some` when
    /// `config.fit_energy_scale` is true; `None` otherwise.
    pub energy_scale_flight_path_m: Option<f64>,
    /// Global multiplicative-baseline coefficients `[b0, b1, b2]` (issue
    /// #635).  `Some` when a baseline was configured with
    /// `spatial_global = true`: stage 1 fits the baseline ONCE on the
    /// aggregated mean spectrum, then freezes it for every pixel (per-pixel
    /// baselines at low counts biased fitted temperatures by up to +150 K;
    /// the global mode removed ~80 % of that).  `None` when no baseline was
    /// configured or in per-pixel mode (see [`Self::baseline_maps`]).
    pub baseline_global: Option<[f64; 3]>,
    /// Reference energy `E_ref` (eV) of the baseline's centered
    /// `ln(E/E_ref)` basis — the geometric midpoint `√(E_min·E_max)` of the
    /// fit grid, stored so consumers reconstruct `B(E)` with the exact
    /// reference the fit used.  `Some` whenever a baseline was configured
    /// (global or per-pixel mode).
    pub baseline_e_ref_ev: Option<f64>,
    /// Per-pixel multiplicative-baseline coefficient maps `[b0, b1, b2]`.
    /// `Some` when a baseline was configured with `spatial_global = false`
    /// (each pixel fits its own baseline); `None` in global mode.
    /// NaN at pixels where `converged_map` is `false`.
    pub baseline_maps: Option<[Array2<f64>; 3]>,
    /// Structured fit-configuration warnings (issue #635) — currently the
    /// degenerate normalization trio (free `Anorm` + free temperature +
    /// ≥1 free density).  Mirrors `SpectrumFitResult::warnings`; also
    /// printed once to stderr since spatial runs are long.
    pub warnings: Vec<String>,
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
    InputData, MultiplicativeBaselineConfig, SolverConfig, UnifiedFitConfig, count_free_params,
    degenerate_normalization_warning, fit_spectrum_typed, required_active_bins,
    validate_counts_resolution_route, validate_multiplicative_baseline,
    validate_transmission_background,
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
///   binomial deviance) on the sample cube, paired
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
/// Apply the multi-pixel polish auto-disable rule.
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
/// Per-pixel numerical fit failures intentionally stay inside the closure
/// where they correctly produce a NaN-only single pixel rather than a
/// whole-map error. Unsupported detector background is validated separately
/// on every live pixel before the closure starts.
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

    // Gate: a fully-constrained fit (issue #633) — every density frozen and
    // no other free parameter — would leave each pixel a converged no-op via
    // the all-fixed solver fast path, i.e. an all-frozen "success" map.
    // Reject the whole map up front with a clear message (mirrors the
    // `fit_spectrum_typed` guard).
    if count_free_params(config) == 0 {
        return Err(PipelineError::InvalidParameter(
            "no free parameters to fit: all densities are frozen and no other \
             parameter is free — free at least one density (with_density_free) \
             or enable fit_temperature / energy-scale / background"
                .into(),
        ));
    }

    // Gate (issue #635): in GLOBAL baseline mode, stage 2 freezes the
    // baseline coefficients before the per-pixel fits — so the free-param
    // count that matters per-pixel EXCLUDES the baseline flags.  Without
    // this check, a config whose only free parameters are the baseline
    // coefficients passes the guard above, stage 1 fits the global
    // baseline, and then EVERY pixel hits fit_spectrum_typed's
    // "no free parameters" rejection — which the rayon loop records as a
    // per-pixel failure, returning Ok(SpatialResult) with all-NaN maps and
    // n_failed == n_total.  That masks a whole-config error as per-pixel
    // failures (the exact class the validate-up-front rule forbids).
    if let Some(bl) = config.multiplicative_baseline()
        && bl.spatial_global
    {
        let n_baseline_free =
            usize::from(bl.fit_b0) + usize::from(bl.fit_b1) + usize::from(bl.fit_b2);
        if count_free_params(config) == n_baseline_free {
            return Err(PipelineError::InvalidParameter(
                "global multiplicative baseline (spatial_global = true) is the \
                 only free parameter block: after stage 1 freezes the fitted \
                 baseline, the per-pixel fits would have nothing left to fit. \
                 Free at least one per-pixel parameter (density / temperature / \
                 energy scale / background), fit the aggregated spectrum with a \
                 single-spectrum fitter instead, or set spatial_global = false \
                 to fit per-pixel baselines."
                    .into(),
            ));
        }
    }

    // Resolve `SolverConfig::Auto` against the input variant — counts
    // → PoissonKL, transmission → LM.  `effective_solver` lives on
    // `UnifiedFitConfig` but takes the 1D `InputData`; inline the
    // resolution here so we do not have to materialise a 1D stub.
    let is_counts = input.is_counts();
    if config.exact_count_response().is_some() {
        return Err(PipelineError::InvalidParameter(
            "exact resolved counts are currently supported by the single-spectrum \
             count fitter only; spatial mapping would rebuild the detector matrix \
             for every pixel and is disabled until that fixed matrix is cached once"
                .into(),
        ));
    }
    // Hoist the scientifically unsupported counts + resolution combination so
    // it becomes one actionable boundary error, not an all-NaN map after every
    // per-pixel error is swallowed by the rayon loop.
    validate_counts_resolution_route(is_counts, input.shape().0, config)?;
    let is_kl = matches!(config.solver(), SolverConfig::PoissonKL(_))
        || (matches!(config.solver(), SolverConfig::Auto) && is_counts);

    // Fractional transmission is not Poisson count data. Hoist the
    // single-spectrum rejection so the pixel loop cannot turn it into an
    // all-NaN success-shaped result.
    if !is_counts && is_kl {
        return Err(PipelineError::InvalidParameter(
            "spatial_map_typed: normalized transmission cannot use the Poisson/KL \
             count objective because fractional transmission is not Poisson count \
             data and the supplied uncertainty would be ignored; use the LM \
             least-squares transmission engine, or supply separate open/sample counts"
                .into(),
        ));
    }
    if !is_counts && config.counts_background().is_some() {
        return Err(PipelineError::InvalidParameter(
            "spatial_map_typed: counts background configuration cannot be used with \
             transmission data; use SAMMY transmission_background or \
             multiplicative_baseline, or supply separate open/sample counts"
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
    // + energy-scale + transmission_background flags +
    // multiplicative-baseline flags, #635), clamped to a
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
            // both supported routes reach this branch.
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

    // Gate: multiplicative-baseline config errors (issue #635) fire
    // identically for every pixel (inits/bounds/positivity are grid+config
    // properties, and the free-Anorm degeneracy is a config property) —
    // hoist them so the caller gets one clear error instead of an all-NaN
    // map with n_failed == n_total.
    validate_multiplicative_baseline(config)?;

    // ── Counts-KL (joint-Poisson) whole-config gates ────────────────
    // Every gate below mirrors a per-pixel rejection in
    // `pipeline.rs::fit_counts_joint_poisson`.  All fire identically
    // across the map because they depend only on shared config flags
    // (alpha fitting, B_A/B/C interlock, `c` value). Unsupported nonzero
    // detector background is rejected across all live pixels by
    // `validate_spatial_data_values` before any fit starts.
    if is_counts && is_kl {
        if let Some(bg) = config.counts_background() {
            if bg.fit_alpha_1 || bg.fit_alpha_2 {
                return Err(PipelineError::InvalidParameter(
                    "joint-Poisson solver does not support fit_alpha_1/fit_alpha_2: \
                     the profile lambda-hat absorbs the global flux scale (alpha_1 redundant); \
                     alpha_2 / B_det wiring is not yet implemented."
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
                 enabled whenever any of B_B / B_C is enabled (A_n alone cannot \
                 absorb a constant offset — benchmarked at −23% density bias)."
                    .into(),
            ));
        }
    }

    Ok(())
}

/// Stage 1 of the two-stage global multiplicative baseline (issue #635):
/// fit the FULL configured model (density / temperature / background /
/// baseline) once on the **aggregated mean spectrum** over all live pixels,
/// and return the fitted `[b0, b1, b2]` for stage 2 to freeze per-pixel.
///
/// Aggregation conventions (must mirror the per-pixel dispatch):
/// - **Transmission**: per-bin mean transmission over live pixels, with the
///   standard error of the mean `√(Σσ²)/n` as the aggregated 1-σ.
/// - **Counts**: per-bin mean sample counts, paired against the SAME
///   spatially-averaged open-beam flux the per-pixel KL dispatch uses
///   (`averaged_flux`); routed as `CountsWithNuisance` + zero background
///   for the KL solver exactly like the rayon closure.  Mean counts are
///   non-integer, which the binomial deviance handles exactly.
/// - **CountsWithNuisance**: per-bin means of all three caller cubes.
///
/// Non-convergence is a HARD error by design: silently falling back to
/// per-pixel baselines would reintroduce the +150 K low-count temperature
/// bias the global mode exists to remove.
#[allow(clippy::too_many_arguments)]
fn fit_global_baseline_stage1(
    input: &InputData3D<'_>,
    fast_config: &UnifiedFitConfig,
    data_a: &Array3<f64>,
    data_b: &Array3<f64>,
    data_c: Option<&Array3<f64>>,
    pixel_coords: &[(usize, usize)],
    averaged_flux: Option<&[f64]>,
) -> Result<[f64; 3], PipelineError> {
    let n_e = data_a.shape()[2];
    let n_live = pixel_coords.len() as f64;
    let mean_over = |cube: &Array3<f64>| -> Vec<f64> {
        let mut m = vec![0.0f64; n_e];
        for &(y, x) in pixel_coords {
            for (e, &v) in cube.slice(s![y, x, ..]).iter().enumerate() {
                m[e] += v;
            }
        }
        for v in &mut m {
            *v /= n_live;
        }
        m
    };

    let aggregate = match input {
        InputData3D::Transmission { .. } => {
            let mean_t = mean_over(data_a);
            // Standard error of the mean under independent per-pixel σ.
            let mut se = vec![0.0f64; n_e];
            for &(y, x) in pixel_coords {
                for (e, &sig) in data_b.slice(s![y, x, ..]).iter().enumerate() {
                    se[e] += sig * sig;
                }
            }
            for v in &mut se {
                *v = v.sqrt() / n_live;
            }
            InputData::Transmission {
                transmission: mean_t,
                uncertainty: se,
            }
        }
        InputData3D::Counts { .. } => {
            let mean_s = mean_over(data_a);
            let flux = averaged_flux
                .expect("averaged_flux is Some for InputData3D::Counts")
                .to_vec();
            // Mirror the per-pixel dispatch: KL → CountsWithNuisance with
            // the averaged flux + zero background; LM → raw Counts.
            let effective = fast_config.effective_solver(&InputData::Counts {
                sample_counts: mean_s.clone(),
                open_beam_counts: flux.clone(),
            });
            match effective {
                SolverConfig::PoissonKL(_) => InputData::CountsWithNuisance {
                    sample_counts: mean_s,
                    flux,
                    background: vec![0.0f64; n_e],
                },
                _ => InputData::Counts {
                    sample_counts: mean_s,
                    open_beam_counts: flux,
                },
            }
        }
        InputData3D::CountsWithNuisance { .. } => InputData::CountsWithNuisance {
            sample_counts: mean_over(data_a),
            flux: mean_over(data_b),
            background: mean_over(data_c.expect("CountsWithNuisance carries a background cube")),
        },
    };

    let agg = fit_spectrum_typed(&aggregate, fast_config).map_err(|e| {
        PipelineError::InvalidParameter(format!(
            "multiplicative-baseline stage 1 (global fit on the aggregated \
             mean spectrum) failed: {e}"
        ))
    })?;
    if !agg.converged {
        return Err(PipelineError::InvalidParameter(
            "multiplicative-baseline stage 1 did not converge on the \
             aggregated mean spectrum; refusing to fall back to per-pixel \
             baselines (at low counts they biased fitted temperatures by up \
             to +150 K). Check the baseline bounds/inits, or set \
             spatial_global = false to fit per-pixel baselines explicitly."
                .into(),
        ));
    }
    Ok(agg
        .baseline
        .expect("stage 1 ran with a configured baseline, so the result carries it"))
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
            if live_pixels.iter().any(|&(y, x)| {
                (0..background.shape()[0]).any(|energy| background[[energy, y, x]].abs() > 1.0e-12)
            }) {
                return Err(PipelineError::InvalidParameter(
                    "joint-Poisson solver with non-zero detector_background is not yet \
                     supported (B_det wiring is deferred)."
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

/// Fit every pixel of a 3-D data cube and return per-pixel maps — the
/// spatial-mapping entry point of the pipeline.
///
/// Runs the single-spectrum fitter once per `(y, x)` pixel of `input`
/// (shape `(n_energies, height, width)`), in parallel over pixels with
/// rayon, and assembles the results into [`SpatialResult`]: one areal
/// density map and uncertainty map per fitted isotope/group, the χ²
/// (or deviance-per-dof) map, the convergence mask, and any optional
/// maps the configuration enables (temperature, normalization,
/// background terms, t0 / flight-path scale).
///
/// # Input modes
///
/// `input` selects the per-pixel objective: pre-normalized
/// [`InputData3D::Transmission`] (+ per-bin uncertainty),
/// [`InputData3D::Counts`] (sample + open-beam), or
/// [`InputData3D::CountsWithNuisance`] (legacy compatibility input; a nonzero
/// background is rejected because it is not connected to the physical
/// two-arm likelihood).
///
/// # Validation (all up-front, before any pixel is fitted)
///
/// * The cube's spectral axis must match `config.energies()`, the
///   mode's companion cubes must match the primary cube's shape, and
///   `dead_pixels` (when given) must be `(height, width)`.
/// * Cube *values* are validated on live pixels, each against its
///   domain — transmission finite, uncertainty finite and strictly
///   positive, counts/flux finite and non-negative, background finite —
///   so a corrupt cube fails loudly instead of producing a quietly-NaN
///   map.  For transmission inputs
///   with a `fit_energy_range`, the value checks are scoped to the
///   active bins — out-of-range bins may contain NaN by design.
/// * Invalid domain/engine configurations are rejected with a diagnostic
///   rather than letting every pixel fail into an all-NaN map. Raw counts use
///   the joint-Poisson engine; LM is reserved for normalized transmission.
///   `transmission_background` settings are validated here for the
///   same reason.  (`fit_energy_scale` together with `fit_temperature`
///   is SUPPORTED since issue #634 — the energy-scale model carries a
///   fitted temperature column.)
///
/// Per-pixel fit *failures* after validation are not errors: the pixel
/// is recorded as NaN in the maps, `converged_map` is `false` there,
/// and `n_failed` counts it.
///
/// # Cancellation and progress
///
/// `cancel` is polled before the sweep and at every pixel; once set,
/// remaining pixels are skipped and the call returns
/// [`PipelineError::Cancelled`] (partial results are discarded).
/// `progress` is incremented once per completed live pixel, so a UI
/// thread can poll it against the number of live pixels
/// (`height × width` minus the `dead_pixels`-masked count).
///
/// # Errors
///
/// [`PipelineError::ShapeMismatch`] for axis/shape disagreements,
/// [`PipelineError::InvalidParameter`] for rejected configurations and
/// invalid cube values, [`PipelineError::Transmission`] when the shared
/// cross-section / resolution-plan precompute fails (e.g. a
/// resolution-kernel or working-grid build error), and
/// [`PipelineError::Cancelled`] when `cancel` was set.
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

    // Raw counts are not silently divided into transmission for LM. Hoist the
    // single-spectrum rejection so a spatial call cannot degrade into an
    // all-NaN success-shaped result after every pixel fails.
    if input.is_counts() && matches!(config.solver(), SolverConfig::LevenbergMarquardt(_)) {
        return Err(PipelineError::InvalidParameter(
            "spatial_map_typed: separate open/sample counts cannot use the LM \
             least-squares transmission engine because silent ratio conversion loses \
             open-beam uncertainty and count statistics; use the Poisson/KL count \
             engine or SolverConfig::Auto"
                .into(),
        ));
    }

    // Issue #634: `fit_energy_scale` + `fit_temperature` is now supported —
    // `EnergyScaleTransmissionModel` wires a fitted temperature column, so
    // per-pixel `fit_spectrum_typed` handles the combination and no spatial
    // guard is needed. (The #458 B3 guard above — LM + fit_energy_scale on
    // counts — is a separate, still-active numerical-stability restriction.)

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
    // transmission_background BackD/BackF interlocks, …).  The
    // fit-range, temperature and
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
    // (Auto resolves to PoissonKL on counts input). When false (a transmission
    // LM input), per-pixel
    // SpectrumFitResult.deviance_per_dof is `None`, so the spatial
    // deviance_per_dof_map should also be `None` — otherwise GUI /
    // Python consumers using `is_some()` to label GOF as "D/dof"
    // would mislabel an all-NaN map.
    let dispatches_to_counts_kl =
        input.is_counts() && !matches!(config.solver(), SolverConfig::LevenbergMarquardt(_));

    // Issue #635: baseline output shape.  Global mode → scalar
    // `baseline_global`; per-pixel mode → `baseline_maps`.
    let baseline_global_mode = config
        .multiplicative_baseline()
        .is_some_and(|bl| bl.spatial_global);
    let has_baseline_maps = config.multiplicative_baseline().is_some() && !baseline_global_mode;
    let baseline_e_ref_ev = config
        .multiplicative_baseline()
        .map(|_| config.baseline_reference_energy());

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(PipelineError::Cancelled);
    }
    if pixel_coords.is_empty() {
        // All pixels filtered out (typically by `dead_pixels` mask).  Per
        // the NaN-on-failure contract (issue #458 B1),
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
            energy_scale_flight_path_m: config.fit_energy_scale().then(|| config.flight_path_m()),
            // No live pixels → stage 1 never ran, so a FITTED global
            // baseline is absent.  A fully FROZEN global baseline involves
            // no fitting, though — the caller's inits ARE the baseline —
            // so mirror the main path and echo them (review R4: the same
            // config must not report Some(inits) on a live map but None on
            // an all-dead one).
            baseline_global: config
                .multiplicative_baseline()
                .filter(|bl| bl.spatial_global && !bl.fit_b0 && !bl.fit_b1 && !bl.fit_b2)
                .map(|bl| [bl.b0_init, bl.b1_init, bl.b2_init]),
            baseline_e_ref_ev,
            baseline_maps: if has_baseline_maps {
                Some([
                    Array2::from_elem((height, width), f64::NAN),
                    Array2::from_elem((height, width), f64::NAN),
                    Array2::from_elem((height, width), f64::NAN),
                ])
            } else {
                None
            },
            warnings: degenerate_normalization_warning(config)
                .into_iter()
                .collect(),
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
                // `broadened_cross_sections`).
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
    //   * k == 1 (handled by the separate scalar surrogate plan below);
    //   * xs is not pre-collapsed to per-group σ (cubature needs the
    //     final σ stack, not per-isotope σ × ratios).
    // Capture any caller-supplied cubature plan BEFORE the local
    // rebuild pathway — the `with_precomputed_cross_sections` setter
    // clears `precomputed_sparse_cubature_plan` as a defence against
    // stale-XS dispatch, so without this snapshot a plan the caller
    // attached via `UnifiedFitConfig::with_precomputed_sparse_cubature_plan`
    // would be dropped and lost on every call.
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
                // debug builds.
                debug_assert_eq!(
                    sigmas_flat.len(),
                    k * n_rows,
                    "cubature σ dimensions: expected {k} × {n_rows} = {}, got {}",
                    k * n_rows,
                    sigmas_flat.len(),
                );
                // Training box: 2 × the initial density — same convention
                // the design study's reference implementation uses.
                // Anchor at the midpoint (0.5 × train_max).
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
                        Some(Arc::new(plan.with_density_box(train_max.clone())))
                    }
                    Err(e) => {
                        // Surface the build failure to stderr rather
                        // than silently swallow it — downstream fits
                        // continue via the exact path, but a missing
                        // cubature on a supposedly-eligible call is
                        // a debugging signal that deserves
                        // visibility.
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
    // runs as today.  A bench-off compared Lanczos σ-pushforward
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
            // Chebyshev-in-density at M = 16 (bench-off winner).
            // Training box: 2 × the initial density;
            // Chebyshev's interpolant is exact at its nodes and
            // tight (≤ 1e-15 rel err) across a well-chosen box.
            //
            // If `n_max` is too wide for 16 nodes to resolve
            // `exp(-n · σ)` accurately (e.g. caller passes a
            // giant `initial_density` on a strong-peak σ), the
            // build's midpoint self-check fires and returns
            // `InsufficientAccuracyOnBox`; we log and fall back
            // to the exact path rather than install a plan that
            // could corrupt the fit.
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
    // the caller-fallback pre-filter.
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

    // A temperature fit retains the resonance source for supported SLBW/MLBW
    // data. Cache the zero-temperature rows once only as the all-isotope
    // fallback for unsupported formalisms; the fit model never selects these
    // rows for an isotope covered by the continuous source evaluator.
    let fallback_base_xs = if config.fit_temperature() && !config.has_precomputed_base_xs() {
        Some(Arc::new(unbroadened_cross_sections(
            config.energies(),
            config.resonance_data(),
            cancel,
        )?))
    } else {
        None
    };
    let fast_config = if config.fit_temperature() {
        let mut cfg = config
            .clone()
            .with_precomputed_cross_sections(xs)
            .with_compute_covariance(true);
        if let Some(fallback_base_xs) = fallback_base_xs {
            cfg = cfg.with_precomputed_fallback_base_xs(fallback_base_xs);
        }
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
    // maps.  Polish is a single-spectrum
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

    // ── Issue #635: two-stage global multiplicative baseline ──
    //
    // Surface the degenerate-normalization warning once, up front (spatial
    // runs are long; a warning buried after the rayon loop is useless), and
    // carry it on the result for GUI / Python consumers.
    let warnings: Vec<String> = degenerate_normalization_warning(config)
        .into_iter()
        .inspect(|w| eprintln!("spatial_map_typed: warning: {w}"))
        .collect();

    // Stage 1 (global mode): fit the baseline ONCE on the aggregated mean
    // spectrum, then FREEZE it into the per-pixel config (the same
    // fixed-parameter substrate as frozen densities).  Non-convergence is a
    // HARD error: silently falling back to per-pixel baselines would
    // reintroduce the +150 K low-count temperature bias the global mode
    // exists to remove.
    let (fast_config, baseline_global) = match fast_config.multiplicative_baseline().cloned() {
        Some(bl) if bl.spatial_global => {
            let b_global = if bl.fit_b0 || bl.fit_b1 || bl.fit_b2 {
                fit_global_baseline_stage1(
                    input,
                    &fast_config,
                    &data_a,
                    &data_b,
                    data_c.as_ref(),
                    &pixel_coords,
                    averaged_flux.as_deref(),
                )?
            } else {
                // Caller froze every coefficient — stage 1 has nothing to
                // fit; the frozen inits ARE the global baseline.
                [bl.b0_init, bl.b1_init, bl.b2_init]
            };
            let frozen = MultiplicativeBaselineConfig {
                b0_init: b_global[0],
                b1_init: b_global[1],
                b2_init: b_global[2],
                fit_b0: false,
                fit_b1: false,
                fit_b2: false,
                ..bl
            };
            (
                fast_config.with_multiplicative_baseline(frozen),
                Some(b_global),
            )
        }
        // Per-pixel mode (or no baseline): pass the config through.
        _ => (fast_config, None),
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
    let mut baseline_maps: Option<[Array2<f64>; 3]> = if has_baseline_maps {
        Some([
            Array2::from_elem((height, width), f64::NAN),
            Array2::from_elem((height, width), f64::NAN),
            Array2::from_elem((height, width), f64::NAN),
        ])
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
        // Per-pixel baseline mode (issue #635): each converged pixel
        // carries its own fitted coefficients.
        if let (Some(maps), Some(b)) = (&mut baseline_maps, result.baseline) {
            maps[0][[*y, *x]] = b[0];
            maps[1][[*y, *x]] = b[1];
            maps[2][[*y, *x]] = b[2];
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
        energy_scale_flight_path_m: config.fit_energy_scale().then(|| config.flight_path_m()),
        baseline_global,
        baseline_e_ref_ev,
        baseline_maps,
        warnings,
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

        // The watcher race is inherently lossy: on a fast or oversubscribed
        // runner the whole 64-pixel sweep (and the post-loop cancel check)
        // can finish before the watcher thread's store becomes visible, in
        // which case the run observed no cancellation at all and a COMPLETE
        // Ok map is the correct output.  That outcome carries no information
        // about the regression under test, so it retries; the regression —
        // a PARTIAL Ok map (cancellation observed mid-loop but swallowed) —
        // fails immediately on any attempt.
        let mut saw_cancelled = false;
        for _attempt in 0..5 {
            let cancel = AtomicBool::new(false);
            let progress = AtomicUsize::new(0);

            let result = std::thread::scope(|s| {
                // Watcher: once at least one pixel has finished, request
                // cancellation while the rest are still being fit.
                s.spawn(|| {
                    while progress.load(Ordering::Relaxed) < 1 {
                        // yield instead of spinning: on a fully subscribed
                        // CI box a busy-spin can be starved for the whole
                        // sweep, losing the race every time.
                        std::thread::yield_now();
                    }
                    cancel.store(true, Ordering::Relaxed);
                });
                spatial_map_typed(&input, &config, None, Some(&cancel), Some(&progress))
            });

            match result {
                Err(PipelineError::Cancelled) => {
                    saw_cancelled = true;
                    break;
                }
                Ok(r) if r.n_converged == r.n_total && r.n_failed == 0 => {
                    // Sweep finished before the flip became visible —
                    // inconclusive; try again.
                    continue;
                }
                other => panic!(
                    "mid-run cancellation must return Err(Cancelled) (or lose \
                     the race with a COMPLETE map), got {other:?}"
                ),
            }
        }
        assert!(
            saw_cancelled,
            "all 5 attempts completed the whole sweep before the cancellation \
             flip became visible — enlarge the pixel grid for this runner"
        );
    }

    /// Cancellation during the `fit_temperature` precompute must surface
    /// as `Err(Cancelled)`, not `Err(Transmission(Cancelled))`.
    ///
    /// The expensive Reich-Moore base-XS precompute
    /// (`unbroadened_cross_sections`) polls `cancel` internally and
    /// returns `TransmissionError::Cancelled`; the documented contract is
    /// that every cancellation path yields `PipelineError::Cancelled`
    /// (the `From<TransmissionError>` impl performs that mapping — a
    /// `.map_err(PipelineError::Transmission)` on the call site would
    /// bypass it and turn a clean user cancel into an error toast).
    ///
    /// Window engineering, so the flip deterministically lands inside
    /// the `unbroadened_cross_sections` call rather than some other
    /// (already correctly mapped) cancellation poll: the caller supplies
    /// precomputed broadened cross-sections, which removes the earlier
    /// expensive broadened-XS window entirely, and the energy grid is
    /// dense enough that the base-XS precompute takes tens of
    /// milliseconds while the watcher flips `cancel` a few ms in.
    /// Wherever the flip lands the correct result is `Err(Cancelled)`,
    /// so the assertion can never flake — only the discrimination
    /// margin varies.  (Mutation-checked: restoring the `map_err` makes
    /// this test fail.)
    #[test]
    fn test_fit_temperature_precompute_cancellation_maps_to_cancelled() {
        use std::sync::atomic::AtomicBool;

        let data = u238_single_resonance();
        // Dense grid: make the base-XS (Reich-Moore) precompute long
        // enough that a few-ms cancel lands inside it on any realistic
        // machine.
        let n_e = 100_001usize;
        let energies: Vec<f64> = (0..n_e).map(|i| 1.0 + (i as f64) * 2e-4).collect();
        let (t_3d, u_3d) = synthetic_grid_transmission(&data, 0.0005, &energies, 2, 2);

        // Caller-supplied broadened XS (values irrelevant — the fit is
        // cancelled before any pixel is evaluated) skip the broadened
        // precompute, so the only long-running pre-sweep stage left is
        // the fit_temperature base-XS precompute under test.
        let precomputed_xs = vec![vec![0.0f64; n_e]];

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            293.6,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_temperature(true)
        .with_precomputed_cross_sections(precomputed_xs.into());

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let cancel = AtomicBool::new(false);
        let result = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(5));
                cancel.store(true, Ordering::Relaxed);
            });
            spatial_map_typed(&input, &config, None, Some(&cancel), None)
        });

        assert!(
            matches!(result, Err(PipelineError::Cancelled)),
            "cancellation during the fit_temperature precompute must map to \
             Err(Cancelled), got {result:?}"
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

    /// Issue #608: the GAUSSIAN-resolution spatial path — `spatial_map_typed`'s
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

    /// Issue #608: `spatial_map_typed`'s `Some(cached)` +
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

    /// Gate 1: spatial count fitting must surface the unsupported response
    /// model once at the public boundary.  Returning an all-NaN map would hide
    /// the fact that every pixel tried to use R[T] instead of the physical
    /// separate-arm response R[Phi*T] / R[Phi].
    #[test]
    fn spatial_counts_resolution_requires_exact_count_response() {
        use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};

        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, ob) = synthetic_4x4_counts(&rd, 0.0005, &energies, 1000.0);
        let flux = Array3::from_elem((energies.len(), 4, 4), 1000.0);
        let background = Array3::zeros((energies.len(), 4, 4));
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            293.6,
            Some(ResolutionFunction::Gaussian(
                ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            )),
            vec![0.001],
        )
        .unwrap();

        let inputs = [
            InputData3D::Counts {
                sample_counts: sample.view(),
                open_beam_counts: ob.view(),
            },
            InputData3D::CountsWithNuisance {
                sample_counts: sample.view(),
                flux: flux.view(),
                background: background.view(),
            },
        ];
        for input in inputs {
            let err = spatial_map_typed(&input, &config, None, None, None)
                .expect_err("spatial counts + resolution must fail at preflight");
            let msg = err.to_string();
            assert!(
                msg.contains("instrument resolution") && msg.contains("separate-arm model"),
                "expected physical counts-response rejection, got: {msg}"
            );
        }
    }

    /// Every count variant plus LM is rejected up-front so the caller
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
            msg.contains("counts") && msg.contains("least-squares") && msg.contains("Poisson"),
            "error must explain the count-domain engine requirement, got: {msg}"
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
            msg.contains("counts") && msg.contains("least-squares") && msg.contains("Poisson"),
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

    /// `(Counts, LM)` is rejected at the spatial boundary rather than
    /// returning a success-shaped map from a lossy ratio conversion.
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
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: open_beam.view(),
        };
        let error = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("counts plus LM must be rejected before the pixel loop");
        let message = error.to_string();
        assert!(
            message.contains("counts")
                && message.contains("least-squares")
                && message.contains("Poisson"),
            "rejection must explain the valid counts engine, got: {message}"
        );
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

    /// Counts plus LM is rejected before optional energy-scale settings are
    /// considered; the invalid engine/domain pair is the primary cause.
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
            msg.contains("counts") && msg.contains("least-squares") && msg.contains("Poisson"),
            "error message should explain the valid counts engine, got: {msg}"
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

    /// Issue #634: `fit_energy_scale + fit_temperature` is now SUPPORTED at
    /// spatial entry (the per-pixel fitter wires a temperature column into the
    /// energy-scale model). `spatial_map_typed` must run without the old guard
    /// error, actually CONVERGE per pixel, and write finite values into both
    /// the temperature and t0/L_scale maps.  Some-ness alone is vacuous — the
    /// maps are pre-allocated as `Some(NaN-filled)` from the config flags, so
    /// an all-pixels-failed run (the exact hazard the replaced guard's doc
    /// comment warned about) would still pass a Some-only assertion.
    #[test]
    fn test_spatial_map_typed_allows_energy_scale_with_temperature() {
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

        let result = spatial_map_typed(&data, &config, None, None, None)
            .expect("fit_energy_scale + fit_temperature is now supported (#634)");
        assert_eq!(result.n_total, 16, "4×4 map");
        // Real acceptance: the joint per-pixel fits must actually converge
        // (neighbouring-spatial-test convention), not merely be dispatched.
        assert!(
            result.n_converged >= 14,
            "joint fit should converge on (nearly) all pixels, got {}/16",
            result.n_converged
        );
        // Converged pixels write FINITE values into all three maps — this is
        // what distinguishes success from the pre-allocated NaN fill.
        let finite_count = |m: &Option<ndarray::Array2<f64>>| {
            m.as_ref()
                .expect("map allocated when its flag is set")
                .iter()
                .filter(|v| v.is_finite())
                .count()
        };
        for (name, map) in [
            ("temperature_map", &result.temperature_map),
            ("t0_us_map", &result.t0_us_map),
            ("l_scale_map", &result.l_scale_map),
        ] {
            let n_finite = finite_count(map);
            assert!(
                n_finite >= result.n_converged,
                "{name}: {n_finite} finite entries < {} converged pixels — \
                 converged pixels must write finite values",
                result.n_converged
            );
        }
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
        // Transmission + Poisson-KL is rejected regardless of optional
        // `fit_energy_range`; the domain mismatch is the primary cause.
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
            msg.contains("normalized transmission") && msg.contains("Poisson"),
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
        // joint-Poisson dispatch: A_n alone cannot
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
        // Background is validated finite before the nonzero-background gate;
        // this closes the `NaN.abs() > 1e-12 == false` comparison leak.
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
    fn test_spatial_counts_with_nuisance_rejects_nonzero_live_background() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (sample, _ob) = synthetic_4x4_counts(&data, 0.0005, &energies, 1000.0);
        let flux = Array3::from_elem((energies.len(), 4, 4), 1000.0);
        let mut background = Array3::from_elem((energies.len(), 4, 4), 0.0);
        background[[2, 3, 3]] = 1.0;
        let config = kl_counts_config(energies.clone(), data);
        let input = InputData3D::CountsWithNuisance {
            sample_counts: sample.view(),
            flux: flux.view(),
            background: background.view(),
        };

        let err = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("unsupported detector background must fail before pixel fitting");
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("non-zero detector_background"),
            "error must name the unsupported detector background, got: {err}"
        );
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

    // ── Issue #635: spatial multiplicative-baseline tests ────────────────

    /// Truth baseline for the spatial closed loops (shared with the
    /// pipeline-level tests): a few % off unity, curved, strictly positive
    /// on the test grids, inside the DEFAULT bounds.
    const SPATIAL_BL_TRUE: [f64; 3] = [1.02, -0.03, 0.01];

    fn spatial_baseline_at(e: f64, e_ref: f64) -> f64 {
        let z = (e / e_ref).ln();
        SPATIAL_BL_TRUE[0] + SPATIAL_BL_TRUE[1] * z + SPATIAL_BL_TRUE[2] * z * z
    }

    /// Low-count 3x3 thermometry cube: counts follow
    /// `lambda(e) = i0 * B(e) * T_600K(e)` with deterministic ~1-sigma
    /// pseudo-Poisson noise (no rand dep; `round(lambda + sqrt(lambda)*g)`
    /// with a sin-hash g).  This is the regime where PER-PIXEL baselines
    /// biased fitted temperatures on real data and the global mode fixed it.
    fn baseline_thermometry_cube(
        energies: &[f64],
        true_density: f64,
        true_temp: f64,
        i0: f64,
    ) -> (Array3<f64>, Array3<f64>) {
        let data = u238_single_resonance();
        let xs = nereids_physics::transmission::broadened_cross_sections(
            energies,
            std::slice::from_ref(&data),
            true_temp,
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
        let e_ref = nereids_fitting::transmission_model::baseline_reference_energy(energies);
        let n_e = energies.len();
        let mut sample = Array3::zeros((n_e, 3, 3));
        let mut ob = Array3::zeros((n_e, 3, 3));
        for y in 0..3 {
            for x in 0..3 {
                for (i, (&t, &e)) in t_1d.iter().zip(energies.iter()).enumerate() {
                    let lam = i0 * spatial_baseline_at(e, e_ref) * t;
                    // Deterministic ~1-sigma pseudo-noise.
                    let g = (1.7 * (i as f64) + 7.9 * (y as f64) + 13.3 * (x as f64)).sin();
                    sample[[i, y, x]] = (lam + lam.sqrt() * g).round().max(0.0);
                    ob[[i, y, x]] = i0;
                }
            }
        }
        (sample, ob)
    }

    #[test]
    fn spatial_global_baseline_recovers_truth_and_beats_unmodeled_control() {
        let true_density = 0.002;
        let true_temp = 600.0;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, ob) = baseline_thermometry_cube(&energies, true_density, true_temp, 400.0);

        // The production thermometry pattern: counts-KL, density frozen at
        // the known areal density, temperature free (seeded 100 K low).
        let base_config = UnifiedFitConfig::new(
            energies.clone(),
            vec![u238_single_resonance()],
            vec!["U-238".into()],
            500.0,
            None,
            vec![true_density],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_fit_temperature(true)
        .with_fix_densities(true);

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };

        // ── Global-baseline run ──
        let with_bl = base_config
            .clone()
            .with_multiplicative_baseline(crate::pipeline::MultiplicativeBaselineConfig::default());
        let r = spatial_map_typed(&input, &with_bl, None, None, None).unwrap();
        assert_eq!(r.n_converged, 9, "all 9 pixels converge in global mode");
        assert!(
            r.warnings.is_empty(),
            "no degenerate trio here: {:?}",
            r.warnings
        );
        assert!(
            r.baseline_maps.is_none(),
            "global mode reports a scalar baseline, not maps"
        );

        // Stage-1 recovery of the injected baseline (probe run measured
        // |error| <= 2e-4 per coefficient at this noise level; 0.01 leaves
        // a 50x margin without admitting a shape-blind fit).
        let bg = r.baseline_global.expect("global baseline populated");
        for (i, (&fitted, &truth)) in bg.iter().zip(SPATIAL_BL_TRUE.iter()).enumerate() {
            assert!(
                (fitted - truth).abs() < 0.01,
                "baseline_global[{i}] = {fitted} vs truth {truth}"
            );
        }
        let e_ref_expected =
            nereids_fitting::transmission_model::baseline_reference_energy(&energies);
        let e_ref = r.baseline_e_ref_ev.expect("E_ref reported");
        assert!(
            (e_ref - e_ref_expected).abs() < 1e-12,
            "E_ref {e_ref} != geometric midpoint {e_ref_expected}"
        );

        // Temperature recovery through the frozen per-pixel baseline
        // (probe: median 603.9 at ~1-sigma pseudo-noise, i0 = 400).
        let t_map = r.temperature_map.as_ref().unwrap();
        let mut temps: Vec<f64> = t_map.iter().copied().filter(|v| v.is_finite()).collect();
        temps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_t = temps[temps.len() / 2];
        assert!(
            (median_t - true_temp).abs() < 15.0,
            "median fitted T = {median_t} vs truth {true_temp}"
        );

        // ── Non-vacuity: the baseline is genuinely in the data ──
        // A control fit WITHOUT the baseline on the SAME cube must show the
        // model mismatch as a strictly worse per-pixel deviance (the 2 %
        // multiplicative distortion contributes ~0.16 per bin at 400 counts,
        // well above the D/dof ~ 1 noise floor).  Without this check the
        // recovery assertions above could pass on data where the baseline
        // injection silently no-opped.
        let control = spatial_map_typed(&input, &base_config, None, None, None).unwrap();
        let mean_dpd = |res: &SpatialResult| -> f64 {
            let m = res.deviance_per_dof_map.as_ref().unwrap();
            let v: Vec<f64> = m.iter().copied().filter(|v| v.is_finite()).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let dpd_baseline = mean_dpd(&r);
        let dpd_control = mean_dpd(&control);
        assert!(
            dpd_baseline < dpd_control,
            "modeling the baseline must improve the fit: D/dof {dpd_baseline} \
             (baseline) vs {dpd_control} (unmodeled control)"
        );
    }

    #[test]
    fn spatial_per_pixel_baseline_mode_populates_maps() {
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, ob) = baseline_thermometry_cube(&energies, 0.002, 600.0, 400.0);
        let config = UnifiedFitConfig::new(
            energies,
            vec![u238_single_resonance()],
            vec!["U-238".into()],
            500.0,
            None,
            vec![0.002],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_fit_temperature(true)
        .with_fix_densities(true)
        .with_multiplicative_baseline(crate::pipeline::MultiplicativeBaselineConfig {
            spatial_global: false,
            ..Default::default()
        });
        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert!(
            r.baseline_global.is_none(),
            "per-pixel mode has no global baseline"
        );
        assert!(
            r.baseline_e_ref_ev.is_some(),
            "E_ref reported in both modes"
        );
        let maps = r.baseline_maps.as_ref().expect("per-pixel baseline maps");
        for y in 0..3 {
            for x in 0..3 {
                if !r.converged_map[[y, x]] {
                    continue;
                }
                let b0 = maps[0][[y, x]];
                assert!(
                    (b0 - SPATIAL_BL_TRUE[0]).abs() < 0.05,
                    "per-pixel b0[{y},{x}] = {b0} vs truth {}",
                    SPATIAL_BL_TRUE[0]
                );
                assert!(maps[1][[y, x]].is_finite() && maps[2][[y, x]].is_finite());
            }
        }
        assert!(r.n_converged > 0, "at least some pixels converge");
    }

    /// Review R1 P0: a config whose ONLY free parameters are the global
    /// baseline coefficients must be rejected up front.  Pre-fix, preflight
    /// counted the (still-free) baseline flags, stage 1 fitted the global
    /// baseline, and then the stage-2 freeze left every per-pixel fit with
    /// zero free parameters — each pixel's "no free parameters" error was
    /// swallowed as a per-pixel failure and the call returned
    /// Ok(SpatialResult) with all-NaN maps and n_failed == n_total.
    #[test]
    fn spatial_global_baseline_as_only_free_block_rejected_up_front() {
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, ob) = baseline_thermometry_cube(&energies, 0.002, 600.0, 400.0);
        // Densities frozen, NO temperature / energy-scale / background —
        // the baseline is the only free block, and global mode will freeze
        // it before the per-pixel stage.
        let config = UnifiedFitConfig::new(
            energies,
            vec![u238_single_resonance()],
            vec!["U-238".into()],
            600.0,
            None,
            vec![0.002],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_fix_densities(true)
        .with_multiplicative_baseline(crate::pipeline::MultiplicativeBaselineConfig::default());
        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };
        let err = spatial_map_typed(&input, &config, None, None, None).expect_err(
            "global-baseline-only config must be a whole-map rejection, not \
             an Ok(all-NaN) result",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("only free parameter block"),
            "error must explain the stage-2 freeze consequence, got: {msg}"
        );

        // Per-pixel mode with the SAME parameter set stays legal: the
        // baseline coefficients remain free in every pixel fit.
        let per_pixel =
            config.with_multiplicative_baseline(crate::pipeline::MultiplicativeBaselineConfig {
                spatial_global: false,
                ..Default::default()
            });
        let r = spatial_map_typed(&input, &per_pixel, None, None, None)
            .expect("per-pixel baseline-only fits are well-posed");
        assert!(r.n_converged > 0, "per-pixel baseline-only fits converge");
    }

    #[test]
    fn spatial_stage1_nonconvergence_is_hard_error() {
        // LM with max_iter = 1 cannot converge from the identity baseline
        // seed on baseline-distorted data — stage 1 must surface a HARD
        // error rather than silently falling back to per-pixel baselines.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, sigma_3d) = synthetic_grid_transmission(&data, 0.002, &energies, 2, 2);
        let e_ref = nereids_fitting::transmission_model::baseline_reference_energy(&energies);
        let mut t_bl = t_3d.clone();
        for y in 0..2 {
            for x in 0..2 {
                for (i, &e) in energies.iter().enumerate() {
                    t_bl[[i, y, x]] = t_3d[[i, y, x]] * spatial_baseline_at(e, e_ref);
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
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig {
            max_iter: 1,
            ..LmConfig::default()
        }))
        .with_multiplicative_baseline(crate::pipeline::MultiplicativeBaselineConfig::default());
        let input = InputData3D::Transmission {
            transmission: t_bl.view(),
            uncertainty: sigma_3d.view(),
        };
        let err = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("non-converged stage 1 must be a hard error");
        assert!(
            err.to_string().contains("stage 1 did not converge"),
            "error must name stage 1, got: {err}"
        );
    }

    #[test]
    fn spatial_rejects_free_anorm_with_baseline_up_front() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..11).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, sigma_3d) = synthetic_grid_transmission(&data, 0.002, &energies, 2, 2);
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
        // fit_anorm defaults to true — the rejected degenerate combination.
        .with_transmission_background(crate::pipeline::BackgroundConfig::default())
        .with_multiplicative_baseline(crate::pipeline::MultiplicativeBaselineConfig::default());
        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: sigma_3d.view(),
        };
        let err = spatial_map_typed(&input, &config, None, None, None)
            .expect_err("free Anorm + baseline must be hoisted to a whole-map rejection");
        assert!(
            err.to_string().contains("Anorm"),
            "rejection must name the degeneracy, got: {err}"
        );
    }

    #[test]
    fn spatial_result_carries_degenerate_trio_warning() {
        // Free Anorm + free temperature + free density (NO baseline — that
        // combination is rejected outright) must surface the structured
        // warning on the SpatialResult even when pixels fail to converge.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, sigma_3d) = synthetic_grid_transmission(&data, 0.002, &energies, 2, 2);
        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig {
            max_iter: 2,
            ..LmConfig::default()
        }))
        .with_fit_temperature(true)
        .with_transmission_background(crate::pipeline::BackgroundConfig::default());
        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: sigma_3d.view(),
        };
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert!(
            r.warnings.iter().any(|w| w.contains("degenerate")),
            "spatial result must carry the degenerate-trio warning, got {:?}",
            r.warnings
        );
    }
}
