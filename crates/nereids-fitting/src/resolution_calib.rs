//! Instrument-resolution calibration.
//!
//! Fits the **instrument-resolution parameters** of a chosen model family to a
//! known-(ρ, T) calibrant, holding the sample density and temperature FIXED.
//! This is the calibrate step of the standard calibrate→pin→fit procedure:
//! characterize the beamline resolution once on a known standard, pin it, then
//! fit unknown samples ([`crate::transmission_model`] / the typed fitters).
//!
//! Mechanism: an outer [`crate::nelder_mead`] optimizer over the few resolution
//! parameters; each evaluation builds a [`ResolutionFunction`] from the
//! parameter vector, runs the existing [`forward_model`] at the fixed
//! [`SampleParams`], and returns χ²/dof after analytically fitting a
//! normalization (`anorm`) and optional low-order baseline (so a baseline offset
//! does not leak into the resolution). Calibration is once-per-experiment, so a
//! derivative-free outer loop — not LM resolution-Jacobians — is the right tool;
//! this mirrors the established outer-loop pattern in
//! [`crate::joint_poisson`]'s polish stage.
//!
//! Families ([`ResolutionFamily`]):
//! - **Gaussian** — fit `(Δt, ΔL)`.
//! - **UddCorr** — fit a shape-preserving width correction `s(E)=s0·(E/Eref)^p`
//!   on a base tabulated UDD ([`TabulatedResolution::width_corrected`]); trusts
//!   the Monte-Carlo shape, calibrates its width/energy-dependence.
//! - **IkedaCarpenter** — fit `α(E)=a0√E+a1` (free analytic prompt-width shape);
//!   `β` is held fixed because `R≈0` in the eV regime makes the storage term, and
//!   hence `β`, unidentifiable.

use std::sync::Arc;

use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, ResolutionParams, TabulatedResolution};
use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};

use crate::error::FittingError;
use crate::nelder_mead::{NelderMeadConfig, NelderMeadResult, nelder_mead_minimize};

/// Reference energy (eV) for the UDD width-correction power law `s(E)`.
const UDD_E_REF: f64 = 10.0;
/// Width-scale clamp for the UDD correction (`s0 = clamp(exp(log_s0), …)`).
/// `pub` so the Python binding decodes the reported `s0` against the *same*
/// bounds the optimizer used, rather than duplicating the literals.
pub const UDD_S0_MIN: f64 = 0.2;
/// Upper width-scale clamp; see [`UDD_S0_MIN`].
pub const UDD_S0_MAX: f64 = 5.0;
/// Fixed storage rate for the IC calibration family. `R(E)≈0` across the eV
/// resonance regime makes the slow/storage term — and hence `β` — unidentifiable,
/// so it is held fixed rather than reported as a meaningless fit result.
const IC_FIXED_BETA: f64 = 0.1;

/// The resolution-model family to calibrate.
#[derive(Debug, Clone)]
pub enum ResolutionFamily {
    /// Gaussian `(Δt_µs, ΔL_m)`.
    Gaussian,
    /// Width-corrected tabulated UDD: fit `(log s0, p)` against `base`.
    UddCorr {
        /// Base Monte-Carlo kernel to correct.
        base: Arc<TabulatedResolution>,
    },
    /// Ikeda–Carpenter: fit `(a0, a1)` with `α(E)=a0√E+a1`; `β` held fixed
    /// (`IC_FIXED_BETA`) because it is unidentifiable in the eV regime.
    IkedaCarpenter,
}

impl ResolutionFamily {
    /// Number of free parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        match self {
            ResolutionFamily::Gaussian
            | ResolutionFamily::UddCorr { .. }
            | ResolutionFamily::IkedaCarpenter => 2,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            ResolutionFamily::Gaussian => "gaussian",
            ResolutionFamily::UddCorr { .. } => "udd_corr",
            ResolutionFamily::IkedaCarpenter => "ic",
        }
    }

    /// `(start vector, box bounds)` for the optimizer (mirrors the validated
    /// Python reference: `udd_corr` uses log-`s0`; bounds keep widths positive).
    fn x0_bounds(&self) -> (Vec<f64>, Vec<(f64, f64)>) {
        match self {
            ResolutionFamily::Gaussian => (vec![2.0, 1e-3], vec![(1e-3, 50.0), (0.0, 0.5)]),
            ResolutionFamily::UddCorr { .. } => {
                // (log s0, p): s0 = exp(log_s0) clamped to [0.2, 5].
                (
                    vec![0.0, 0.0],
                    vec![(UDD_S0_MIN.ln(), UDD_S0_MAX.ln()), (-4.0, 4.0)],
                )
            }
            ResolutionFamily::IkedaCarpenter => (vec![0.30, 0.0], vec![(0.01, 5.0), (-2.0, 2.0)]),
        }
    }
}

/// Configuration for [`calibrate_resolution`].
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Flight-path length (m).
    pub flight_path_m: f64,
    /// Fit a low-order baseline (anorm + constant + linear) instead of anorm only.
    pub fit_background: bool,
    /// Number of optimizer restarts (perturbed starts; keep the best).
    pub restarts: usize,
    /// Nelder–Mead simplex-spread tolerance.
    pub xatol: f64,
    /// Nelder–Mead objective-range tolerance.
    pub fatol: f64,
    /// Nelder–Mead maximum iterations.
    pub max_iter: usize,
    /// IC synthesis grid resolution (energies × τ-samples per kernel).
    pub ic_n_energies: usize,
    pub ic_n_tau: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        // Matches the validated Python calibrator (fatol=1e-3, not the
        // NelderMeadConfig default 1e-4). The IC synthesis grid is DELIBERATELY
        // lighter than the standalone IkedaCarpenter default (64×500 here vs the
        // DEFAULT_N_ENERGIES×DEFAULT_N_TAU = 64×600 synthesis default): the outer
        // loop re-synthesizes the kernel on every evaluation, and 500 τ-samples is
        // ample for χ²/dof comparison.
        Self {
            flight_path_m: 25.0,
            fit_background: false,
            restarts: 1,
            xatol: 1e-4,
            fatol: 1e-3,
            max_iter: 800,
            ic_n_energies: 64,
            ic_n_tau: 500,
        }
    }
}

/// Result of a resolution calibration.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Family label (`"gaussian"` | `"udd_corr"` | `"ic"`).
    pub family: String,
    /// Fitted parameter vector (raw optimizer space; see [`ResolutionFamily`]).
    pub theta: Vec<f64>,
    /// χ²/dof of the best fit (after anorm/baseline).
    pub chi2_dof: f64,
    /// The calibrated resolution, ready to pin into a sample fit.
    pub resolution: ResolutionFunction,
    /// Optimizer iterations of the winning restart.
    pub iterations: usize,
    /// Whether the winning restart self-converged.
    pub converged: bool,
}

fn build_resolution(
    family: &ResolutionFamily,
    theta: &[f64],
    e_min: f64,
    e_max: f64,
    cfg: &CalibrationConfig,
) -> Result<ResolutionFunction, FittingError> {
    match family {
        ResolutionFamily::Gaussian => {
            let params =
                ResolutionParams::new(cfg.flight_path_m, theta[0].abs(), theta[1].abs(), 0.0)
                    .map_err(|e| FittingError::EvaluationFailed(format!("gaussian res: {e:?}")))?;
            Ok(ResolutionFunction::Gaussian(params))
        }
        ResolutionFamily::UddCorr { base } => {
            let s0 = theta[0].exp().clamp(UDD_S0_MIN, UDD_S0_MAX);
            let corrected = base
                .width_corrected(s0, theta[1], UDD_E_REF)
                .map_err(|e| FittingError::EvaluationFailed(format!("udd_corr width: {e}")))?;
            Ok(ResolutionFunction::Tabulated(Arc::new(corrected)))
        }
        ResolutionFamily::IkedaCarpenter => {
            // R(E)=exp(-E_meV/25) -> ~0 across the eV resonance regime, so the
            // slow/storage term vanishes and β is unidentifiable; hold β fixed
            // and fit only the prompt-width law α(E)=a0·√E + a1.
            let params = IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE {
                    a0: theta[0].abs(),
                    a1: theta[1],
                },
                beta: IC_FIXED_BETA,
                r: EnergyLaw::ExpMilliEv { kappa: 25.0 },
                burst_sigma_us: None,
                channel_fwhm_us: None,
            };
            let grid = SynthesisGrid {
                e_min_ev: (e_min * 0.5).max(1e-3),
                e_max_ev: e_max * 2.0,
                n_energies: cfg.ic_n_energies,
                n_tau: cfg.ic_n_tau,
            };
            let ic = IkedaCarpenter::new(params, cfg.flight_path_m, &grid)
                .map_err(|e| FittingError::EvaluationFailed(format!("ic res: {e:?}")))?;
            Ok(ResolutionFunction::IkedaCarpenter(Arc::new(ic)))
        }
    }
}

/// χ²/dof after analytically fitting `anorm` (+ optional constant+linear
/// baseline): `data ≈ a·model (+ b0 + b1·x)`, weighted by `1/unc`. `n_res_params`
/// is the number of resolution parameters fit by the outer loop; it is subtracted
/// from the dof alongside the linear `anorm`/baseline columns so the reported
/// χ²/dof counts *all* free parameters (the outer-loop resolution params are not
/// in the linear system but still consume degrees of freedom).
fn inner_chi2(data: &[f64], unc: &[f64], model: &[f64], fit_bg: bool, n_res_params: usize) -> f64 {
    let n = data.len();
    let k = if fit_bg { 3 } else { 1 };
    let mut ata = vec![0.0f64; k * k];
    let mut atb = vec![0.0f64; k];
    for i in 0..n {
        let w2 = 1.0 / unc[i].max(1e-9).powi(2);
        let x = if n > 1 {
            -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)
        } else {
            0.0
        };
        let col = [model[i], 1.0, x];
        for a in 0..k {
            atb[a] += w2 * col[a] * data[i];
            for b in 0..k {
                ata[a * k + b] += w2 * col[a] * col[b];
            }
        }
    }
    // A singular normal-equations system (e.g. a degenerate/constant model
    // column) means anorm/baseline are unfit-able for this θ — report the point
    // as infeasible so the optimizer steps away, rather than a spuriously
    // inflated finite χ² from a zeroed solution.
    let Some(coef) = solve_small(&ata, &atb, k) else {
        return f64::INFINITY;
    };
    let mut ssr = 0.0;
    for i in 0..n {
        let w2 = 1.0 / unc[i].max(1e-9).powi(2);
        let x = if n > 1 {
            -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)
        } else {
            0.0
        };
        let pred = if fit_bg {
            coef[0] * model[i] + coef[1] + coef[2] * x
        } else {
            coef[0] * model[i]
        };
        ssr += (data[i] - pred).powi(2) * w2;
    }
    let dof = n.saturating_sub(k + n_res_params).max(1) as f64;
    ssr / dof
}

/// Solve a small `k×k` linear system `A x = b` (k ≤ 3) by Gaussian elimination
/// with partial pivoting. Returns `None` on a singular system.
fn solve_small(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    // Relative pivot threshold scaled by the matrix norm, so ill-conditioned
    // systems (not just exactly-singular ones) are reported infeasible.
    let scale = a
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let mut m = a.to_vec();
    let mut y = b.to_vec();
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if m[r * k + col].abs() > m[piv * k + col].abs() {
                piv = r;
            }
        }
        if m[piv * k + col].abs() < 1e-12 * scale {
            return None;
        }
        if piv != col {
            for c in 0..k {
                m.swap(piv * k + c, col * k + c);
            }
            y.swap(piv, col);
        }
        for r in (col + 1)..k {
            let f = m[r * k + col] / m[col * k + col];
            for c in col..k {
                m[r * k + c] -= f * m[col * k + c];
            }
            y[r] -= f * y[col];
        }
    }
    let mut x = vec![0.0; k];
    for col in (0..k).rev() {
        let mut s = y[col];
        for c in (col + 1)..k {
            s -= m[col * k + c] * x[c];
        }
        x[col] = s / m[col * k + col];
    }
    Some(x)
}

/// Calibrate the resolution parameters of `family` against a known-(ρ,T)
/// calibrant.
///
/// `sample` carries the FIXED density and temperature (and isotopes/groups);
/// only the resolution parameters are optimized. Returns the fitted parameters,
/// χ²/dof, and the calibrated [`ResolutionFunction`] (ready to pin).
///
/// # Errors
/// [`FittingError::EmptyData`] / [`FittingError::LengthMismatch`] for bad
/// inputs; propagates optimizer errors.
pub fn calibrate_resolution(
    family: ResolutionFamily,
    energies: &[f64],
    data: &[f64],
    unc: &[f64],
    sample: &SampleParams,
    config: &CalibrationConfig,
) -> Result<CalibrationResult, FittingError> {
    if data.is_empty() {
        return Err(FittingError::EmptyData);
    }
    if energies.len() != data.len() || unc.len() != data.len() {
        return Err(FittingError::LengthMismatch {
            expected: data.len(),
            actual: energies.len().min(unc.len()),
            field: "energies/unc vs data",
        });
    }
    // Reject non-finite inputs up front: a NaN datum would otherwise propagate
    // to a NaN χ², and since `NaN < x` is false the optimizer could retain it as
    // "best" and return a NaN-objective fit silently.
    if !energies.iter().all(|v| v.is_finite())
        || !data.iter().all(|v| v.is_finite())
        || !unc.iter().all(|v| v.is_finite() && *v > 0.0)
    {
        return Err(FittingError::InvalidConfig(
            "energies, data must be finite and uncertainty finite and > 0".into(),
        ));
    }
    // Energy grid must be strictly positive and strictly ascending — mirror the
    // Python entry point's `validate_energy_grid` so both public APIs reject the
    // same inputs up front. Without this, a zero/negative energy panics deep in
    // the Reich–Moore cross-section assert, a descending grid errors late as a
    // generic "forward model failed", and duplicate energies are silently
    // accepted (the recurring NEREIDS sibling-path validation gap).
    if energies[0] <= 0.0 {
        return Err(FittingError::InvalidConfig(
            "energies must be strictly positive".into(),
        ));
    }
    if !energies.windows(2).all(|w| w[1] > w[0]) {
        return Err(FittingError::InvalidConfig(
            "energies must be strictly ascending (no duplicates)".into(),
        ));
    }
    // The calibrant must have at least one isotope with a finite, positive areal
    // density. Otherwise `forward_model` skips every isotope (thickness ≤ 0) and
    // returns a flat T≡1 that is independent of the resolution parameters, so the
    // optimizer would converge to a finite but physically meaningless result —
    // silently masking a whole-config error. Mirrors the Python wrapper's guard.
    if !sample
        .isotopes()
        .iter()
        .any(|(_, density)| density.is_finite() && *density > 0.0)
    {
        return Err(FittingError::InvalidConfig(
            "calibrant must have at least one isotope with a finite, positive density".into(),
        ));
    }
    // Reject under-determined calibrants: need more data points than the total
    // free parameters (resolution params + the anorm/baseline columns), else the
    // reported χ²/dof is meaningless.
    let baseline_cols = if config.fit_background { 3 } else { 1 };
    if data.len() <= family.n_params() + baseline_cols {
        return Err(FittingError::InvalidConfig(format!(
            "calibrant has {} points but the model has {} resolution + {} baseline parameters; \
             need strictly more data points than parameters",
            data.len(),
            family.n_params(),
            baseline_cols,
        )));
    }
    let e_min = energies.first().copied().unwrap_or(1.0);
    let e_max = energies.last().copied().unwrap_or(1.0);
    let (x0, bounds) = family.x0_bounds();
    let nm = NelderMeadConfig {
        xatol: config.xatol,
        fatol: config.fatol,
        max_iter: config.max_iter,
        ..Default::default()
    };

    let mut best: Option<NelderMeadResult> = None;
    for r in 0..config.restarts.max(1) {
        // Additive perturbation (a fraction of each parameter's bound range) so
        // restarts move even for zero-valued start components — a multiplicative
        // `x0·(1+0.1r)` left `udd_corr`'s `[0, 0]` start identical every restart.
        let start: Vec<f64> = x0
            .iter()
            .zip(&bounds)
            .map(|(&v, &(lo, hi))| (v + 0.1 * r as f64 * (hi - lo)).clamp(lo, hi))
            .collect();
        let obj = |theta: &[f64]| -> Result<f64, FittingError> {
            let res = build_resolution(&family, theta, e_min, e_max, config)?;
            let inst = InstrumentParams { resolution: res };
            let model = forward_model(energies, sample, Some(&inst))
                .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))?;
            if !model.iter().all(|v| v.is_finite()) {
                return Err(FittingError::EvaluationFailed("non-finite model".into()));
            }
            Ok(inner_chi2(
                data,
                unc,
                &model,
                config.fit_background,
                family.n_params(),
            ))
        };
        let res = nelder_mead_minimize(obj, &start, Some(&bounds), &nm)?;
        if best.as_ref().is_none_or(|b| res.fun < b.fun) {
            best = Some(res);
        }
    }
    let best = best.expect("at least one restart runs");
    if !best.fun.is_finite() {
        return Err(FittingError::EvaluationFailed(
            "calibration found no finite-χ² resolution (the forward model failed for every \
             parameter vector tried)"
                .into(),
        ));
    }
    let resolution = build_resolution(&family, &best.x, e_min, e_max, config)?;
    Ok(CalibrationResult {
        family: family.label().to_string(),
        theta: best.x,
        chi2_dof: best.fun,
        resolution,
        iterations: best.iterations,
        converged: best.self_converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::test_support::synthetic_isotope;

    fn synthetic_base_udd() -> TabulatedResolution {
        // Asymmetric kernel (sharp rise, +TOF tail), at two reference energies.
        let offs = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let wts = vec![0.05, 0.3, 1.0, 0.8, 0.5, 0.3, 0.15, 0.05];
        TabulatedResolution::from_kernels(
            vec![5.0, 50.0],
            vec![(offs.clone(), wts.clone()), (offs, wts)],
            25.0,
        )
        .unwrap()
    }

    #[test]
    fn inner_chi2_zero_on_exact_anorm() {
        let model = vec![0.9, 0.7, 0.5, 0.8];
        let data: Vec<f64> = model.iter().map(|m| 1.0 * m).collect();
        let unc = vec![0.01; 4];
        assert!(inner_chi2(&data, &unc, &model, false, 0) < 1e-18);
    }

    #[test]
    fn udd_corr_recovers_known_width_scale() {
        // Loop-closure / OPTIMIZER test: truth and fit both use width_corrected, so
        // this checks that the calibrator finds the s0=1.5 minimum — NOT that
        // width_corrected itself is physically correct. The width-scale physics
        // (centroid invariance + std scaling) is independently verified by
        // `width_corrected_preserves_centroid_scales_width_and_energy_dependence`
        // in nereids-physics.
        // Synthetic Hf-178-like resonance at 20 eV; calibrant generated with a
        // UDD truth scaled by s0=1.5; udd_corr must recover s0≈1.5 at χ²≈0.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..400).map(|i| 12.0 + i as f64 * 0.04).collect();
        let base = synthetic_base_udd();
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(1.5, 0.0, UDD_E_REF).unwrap(),
        ));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];

        let cfg = CalibrationConfig {
            restarts: 2,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::UddCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let s0 = r.theta[0].exp().clamp(UDD_S0_MIN, UDD_S0_MAX);
        assert!((s0 - 1.5).abs() < 0.05, "recovered s0={s0}, expected 1.5");
        assert!(r.chi2_dof < 1e-2, "matched χ²/dof={} too high", r.chi2_dof);
        assert!(matches!(r.resolution, ResolutionFunction::Tabulated(_)));
    }

    #[test]
    fn gaussian_and_ic_families_run_and_converge() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..300).map(|i| 12.0 + i as f64 * 0.05).collect();
        let base = synthetic_base_udd();
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(1.2, 0.0, UDD_E_REF).unwrap(),
        ));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig::default();
        for fam in [ResolutionFamily::Gaussian, ResolutionFamily::IkedaCarpenter] {
            let label = fam.label().to_string();
            let r = calibrate_resolution(fam, &energies, &data, &unc, &sample, &cfg).unwrap();
            assert!(r.chi2_dof.is_finite(), "{label} χ² not finite");
            assert_eq!(r.theta.len(), 2, "{label} should fit 2 params");
            // The objective is smooth and noise-free, so Nelder–Mead reaches its
            // tolerance well within max_iter — guard the "_and_converge" promise.
            assert!(r.converged, "{label} did not self-converge");
        }
    }

    #[test]
    fn n_params_matches_family() {
        assert_eq!(ResolutionFamily::Gaussian.n_params(), 2);
        assert_eq!(
            ResolutionFamily::UddCorr {
                base: Arc::new(synthetic_base_udd())
            }
            .n_params(),
            2
        );
        // IC fits only (a0, a1); β is held fixed (unidentifiable in the eV regime).
        assert_eq!(ResolutionFamily::IkedaCarpenter.n_params(), 2);
    }

    #[test]
    fn rejects_empty_mismatched_and_non_finite_inputs() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let cfg = CalibrationConfig::default();
        assert!(matches!(
            calibrate_resolution(ResolutionFamily::Gaussian, &[], &[], &[], &sample, &cfg),
            Err(FittingError::EmptyData)
        ));
        let e = vec![1.0, 2.0, 3.0];
        let d = vec![0.5, 0.5];
        let u = vec![0.1, 0.1];
        assert!(matches!(
            calibrate_resolution(ResolutionFamily::Gaussian, &e, &d, &u, &sample, &cfg),
            Err(FittingError::LengthMismatch { .. })
        ));
        // Non-finite data and non-positive uncertainty are rejected up front.
        let e = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            calibrate_resolution(
                ResolutionFamily::Gaussian,
                &e,
                &[0.5, f64::NAN, 0.7],
                &[0.1; 3],
                &sample,
                &cfg
            ),
            Err(FittingError::InvalidConfig(_))
        ));
        assert!(matches!(
            calibrate_resolution(
                ResolutionFamily::Gaussian,
                &e,
                &[0.5; 3],
                &[0.1, 0.0, 0.1],
                &sample,
                &cfg
            ),
            Err(FittingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_nonascending_and_nonpositive_energy_grid() {
        // Sibling-path parity with the Python `validate_energy_grid`: descending,
        // duplicate, zero, and negative energy grids must be rejected up front
        // rather than panicking deep in the cross-section assert or erroring late.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let cfg = CalibrationConfig::default();
        for grid in [
            vec![4.0, 3.0, 2.0, 1.0],  // descending
            vec![1.0, 2.0, 2.0, 3.0],  // duplicate
            vec![0.0, 1.0, 2.0, 3.0],  // zero
            vec![-1.0, 1.0, 2.0, 3.0], // negative
        ] {
            let n = grid.len();
            assert!(
                matches!(
                    calibrate_resolution(
                        ResolutionFamily::Gaussian,
                        &grid,
                        &vec![0.5; n],
                        &vec![0.1; n],
                        &sample,
                        &cfg
                    ),
                    Err(FittingError::InvalidConfig(_))
                ),
                "expected InvalidConfig for grid {grid:?}"
            );
        }
    }

    #[test]
    fn rejects_degenerate_calibrant_composition() {
        // A calibrant with no isotopes, or only zero/negative densities, yields a
        // flat (resolution-independent) forward model; the optimizer would return
        // a finite but meaningless result. Reject up front (Python-sibling parity).
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..64).map(|i| 5.0 + i as f64 * 0.4).collect();
        let data = vec![0.8; energies.len()];
        let unc = vec![0.01; energies.len()];
        let cfg = CalibrationConfig::default();
        for bad_sample in [
            SampleParams::new(300.0, vec![]).unwrap(),
            SampleParams::new(300.0, vec![(iso.clone(), 0.0)]).unwrap(),
            SampleParams::new(300.0, vec![(iso, -1.0e-3)]).unwrap(),
        ] {
            assert!(
                matches!(
                    calibrate_resolution(
                        ResolutionFamily::Gaussian,
                        &energies,
                        &data,
                        &unc,
                        &bad_sample,
                        &cfg
                    ),
                    Err(FittingError::InvalidConfig(_))
                ),
                "degenerate calibrant composition should be rejected"
            );
        }
    }

    #[test]
    fn invalid_flight_path_propagates_build_error() {
        // flight_path <= 0 makes ResolutionParams::new fail on every eval, so the
        // calibration cannot build a resolution and returns an error.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let e: Vec<f64> = (0..60).map(|i| 15.0 + i as f64 * 0.2).collect();
        let d = vec![0.9; 60];
        let u = vec![0.01; 60];
        let cfg = CalibrationConfig {
            flight_path_m: -1.0,
            ..Default::default()
        };
        assert!(
            calibrate_resolution(ResolutionFamily::Gaussian, &e, &d, &u, &sample, &cfg).is_err()
        );
    }

    #[test]
    fn inner_chi2_background_path_and_degenerate_model() {
        // 3-column baseline fit (anorm + const + linear) recovers an offset exactly.
        let model = vec![0.9, 0.7, 0.5, 0.8, 0.6];
        let data: Vec<f64> = model.iter().map(|m| 0.5 * m + 0.1).collect();
        let unc = vec![0.01; 5];
        assert!(inner_chi2(&data, &unc, &model, true, 0) < 1e-12);
        // all-zero model -> singular normal equations -> infeasible (χ²=∞), so the
        // optimizer steps away rather than seeing a spuriously inflated finite χ².
        let v = inner_chi2(&data, &unc, &[0.0; 5], false, 0);
        assert_eq!(v, f64::INFINITY);
    }

    #[test]
    fn calibrate_with_background_runs() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..200).map(|i| 14.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udd();
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(1.3, 0.0, UDD_E_REF).unwrap(),
        ));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig {
            fit_background: true,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::UddCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        assert!(r.chi2_dof.is_finite());
    }

    #[test]
    fn ic_recovers_known_alpha() {
        // Loop-closure / optimizer test (same caveat as udd_corr): truth and fit
        // both use the IC synthesis, so this checks the optimizer recovers a0 —
        // the IC pulse physics is independently covered by the ic_pulse tests in
        // nereids-physics. Truth a0 = 0.35; the calibration must recover it.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..400).map(|i| 12.0 + i as f64 * 0.04).collect();
        let ic_truth = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.0 },
                beta: IC_FIXED_BETA,
                r: EnergyLaw::ExpMilliEv { kappa: 25.0 },
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            25.0,
            &SynthesisGrid {
                e_min_ev: 6.0,
                e_max_ev: 60.0,
                n_energies: 64,
                n_tau: 500,
            },
        )
        .unwrap();
        let truth = ResolutionFunction::IkedaCarpenter(Arc::new(ic_truth));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig {
            restarts: 2,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter,
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let a0 = r.theta[0].abs();
        assert!((a0 - 0.35).abs() < 0.04, "recovered a0={a0}, expected 0.35");
        assert!(r.chi2_dof < 1.0, "matched χ²/dof={} too high", r.chi2_dof);
    }
}
