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
//! - **IkedaCarpenter** — fit `α(E)=a0√E+a1` and `β` (free analytic shape).

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
const UDD_S0_MIN: f64 = 0.2;
const UDD_S0_MAX: f64 = 5.0;

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
    /// Ikeda–Carpenter: fit `(a0, a1, β)` with `α(E)=a0√E+a1`.
    IkedaCarpenter,
}

impl ResolutionFamily {
    /// Number of free parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        match self {
            ResolutionFamily::Gaussian | ResolutionFamily::UddCorr { .. } => 2,
            ResolutionFamily::IkedaCarpenter => 3,
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
            ResolutionFamily::IkedaCarpenter => (
                vec![0.30, 0.0, 0.10],
                vec![(0.01, 5.0), (-2.0, 2.0), (1e-3, 2.0)],
            ),
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
        // NelderMeadConfig default 1e-4; IC grid 64×500).
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
            let corrected = base.width_corrected(s0, theta[1], UDD_E_REF);
            Ok(ResolutionFunction::Tabulated(Arc::new(corrected)))
        }
        ResolutionFamily::IkedaCarpenter => {
            let params = IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE {
                    a0: theta[0].abs(),
                    a1: theta[1],
                },
                beta: theta[2].abs().max(1e-3),
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
/// baseline): `data ≈ a·model (+ b0 + b1·x)`, weighted by `1/unc`.
fn inner_chi2(data: &[f64], unc: &[f64], model: &[f64], fit_bg: bool) -> f64 {
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
    let coef = solve_small(&ata, &atb, k);
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
    let dof = n.saturating_sub(k).max(1) as f64;
    ssr / dof
}

/// Solve a small `k×k` linear system `A x = b` (k ≤ 3) by Gaussian elimination
/// with partial pivoting. Returns zeros on a singular system (the objective
/// then reports a large residual and the optimizer steps away).
fn solve_small(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut y = b.to_vec();
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if m[r * k + col].abs() > m[piv * k + col].abs() {
                piv = r;
            }
        }
        if m[piv * k + col].abs() < 1e-300 {
            return vec![0.0; k];
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
    x
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
        let start: Vec<f64> = x0.iter().map(|&v| v * (1.0 + 0.1 * r as f64)).collect();
        let obj = |theta: &[f64]| -> Result<f64, FittingError> {
            let res = build_resolution(&family, theta, e_min, e_max, config)?;
            let inst = InstrumentParams { resolution: res };
            let model = forward_model(energies, sample, Some(&inst))
                .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))?;
            if !model.iter().all(|v| v.is_finite()) {
                return Err(FittingError::EvaluationFailed("non-finite model".into()));
            }
            Ok(inner_chi2(data, unc, &model, config.fit_background))
        };
        let res = nelder_mead_minimize(obj, &start, Some(&bounds), &nm)?;
        if best.as_ref().is_none_or(|b| res.fun < b.fun) {
            best = Some(res);
        }
    }
    let best = best.expect("at least one restart runs");
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
        assert!(inner_chi2(&data, &unc, &model, false) < 1e-18);
    }

    #[test]
    fn udd_corr_recovers_known_width_scale() {
        // Synthetic Hf-178-like resonance at 20 eV; calibrant generated with a
        // UDD truth scaled by s0=1.5; udd_corr must recover s0≈1.5 at χ²≈0.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..400).map(|i| 12.0 + i as f64 * 0.04).collect();
        let base = synthetic_base_udd();
        let truth =
            ResolutionFunction::Tabulated(Arc::new(base.width_corrected(1.5, 0.0, UDD_E_REF)));
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
        let truth =
            ResolutionFunction::Tabulated(Arc::new(base.width_corrected(1.2, 0.0, UDD_E_REF)));
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
            assert_eq!(r.theta.len(), if label == "ic" { 3 } else { 2 });
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
        assert_eq!(ResolutionFamily::IkedaCarpenter.n_params(), 3);
    }

    #[test]
    fn rejects_empty_and_mismatched_inputs() {
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
        assert!(inner_chi2(&data, &unc, &model, true) < 1e-12);
        // all-zero model -> singular normal equations -> solve_small returns zeros, finite chi2.
        let v = inner_chi2(&data, &unc, &[0.0; 5], false);
        assert!(v.is_finite());
    }

    #[test]
    fn calibrate_with_background_runs() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..200).map(|i| 14.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udd();
        let truth =
            ResolutionFunction::Tabulated(Arc::new(base.width_corrected(1.3, 0.0, UDD_E_REF)));
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
}
