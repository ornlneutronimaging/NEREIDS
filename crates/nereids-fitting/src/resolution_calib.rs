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
//! - **UdrCorr** — fit a shape-preserving width correction `s(E)=s0·(E/Eref)^p`
//!   on a base tabulated UDR ([`TabulatedResolution::width_corrected`]); trusts
//!   the Monte-Carlo shape, calibrates its width/energy-dependence. **UDR** =
//!   *User-Defined Resolution*, SAMMY's term for a numerical (table-supplied)
//!   resolution function.
//! - **IkedaCarpenter** — fit `α(E)=a0√E+a1` (free analytic prompt-width shape);
//!   `β` is held fixed because `R≈0` in the eV regime makes the storage term, and
//!   hence `β`, unidentifiable.

use std::sync::Arc;

use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    ResolutionFunction, ResolutionParams, TOF_FACTOR, TabulatedResolution,
};
use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};

use crate::error::FittingError;
use crate::nelder_mead::{NelderMeadConfig, NelderMeadResult, nelder_mead_minimize};

/// Reference energy (eV) for the UDR width-correction power law `s(E)`.
const UDR_E_REF: f64 = 10.0;
/// Width-scale clamp for the UDR correction (`s0 = clamp(exp(log_s0), …)`).
/// `pub` so the Python binding decodes the reported `s0` against the *same*
/// bounds the optimizer used, rather than duplicating the literals.
pub const UDR_S0_MIN: f64 = 0.2;
/// Upper width-scale clamp; see [`UDR_S0_MIN`].
pub const UDR_S0_MAX: f64 = 5.0;
/// Fixed storage rate for the IC calibration family. `R(E)≈0` across the eV
/// resonance regime makes the slow/storage term — and hence `β` — unidentifiable,
/// so it is held fixed rather than reported as a meaningless fit result.
const IC_FIXED_BETA: f64 = 0.1;
/// Guard-rail bound (µs) on the optional fitted TOF zero `t0`. `t0` and `L_scale`
/// are the SAMMY *energy-scale* parameters; resolution calibration pins them by
/// default and fits them only as an explicit, prior-constrained opt-in (see
/// [`CalibrationConfig::with_position_prior`]). ±5 µs is a guard rail far inside
/// the feasible `|t0| < min(TOF)`; the *real* constraint on `t0` is the metrology
/// prior, not this bound.
///
/// History: a previous design fit a *free per-family* constant `t0` and discarded
/// it ("position nuisance") to make the cross-family χ² compare shape/width. That
/// was wrong — the asymmetric-kernel mode→centroid lag is `≈1/√E` (exact for the
/// `a1=0` prompt law `α=a0√E`, leading-order otherwise), the SAME basis as an
/// `L_scale` error, so a free per-family `t0`/`L_scale` lets a wrong (symmetric)
/// family imitate the lag and buy back the strongest evidence against it (χ² 6.0 →
/// 1.3 in the Hf-177 study). Position is now a SHARED energy-scale parameter with a
/// metrology prior, never a free per-family knob.
const POSITION_T0_US_MAX: f64 = 5.0;
/// Guard-rail bounds (±2%) on the optional fitted flight-path scale `L_scale`.
/// The IC mode→centroid lag needs only `ΔL/L ≈ 0.22%` to be mimicked, so a *free*
/// `L_scale` absorbs the lag and corrupts the calibrated width — fit it only under
/// a prior, for an explicit energy-scale / identifiability study.
const POSITION_L_SCALE_MIN: f64 = 0.98;
/// Upper guard-rail bound on `L_scale`; see [`POSITION_L_SCALE_MIN`].
const POSITION_L_SCALE_MAX: f64 = 1.02;

/// Map an energy grid through the SAMMY energy-scale `(t0, L_scale)`, using the
/// SAME convention as `EnergyScaleTransmissionModel::corrected_energies`: with
/// nominal `tof(E) = TOF_FACTOR·L/√E`, the corrected energy is
/// `E' = (TOF_FACTOR·L·L_scale / (tof − t0))²`. Identity at `(t0, L_scale) = (0, 1)`.
///
/// Note the `−t0` sign (a positive `t0` is *subtracted* from the measured TOF, so
/// it raises the corrected energy) — this is the shipped energy-scale convention,
/// opposite to the `+t0` form used by the retired position nuisance. Errors if any
/// corrected TOF `tof − t0 ≤ 0` (a `t0` past the shortest flight time).
///
/// `pub(crate)` so the equivalence to the runtime
/// [`EnergyScaleTransmissionModel::corrected_energies`] is *pinned by a test*
/// (`corrected_energy_grid_matches_energy_scale_model`) rather than only asserted
/// in prose — a future edit to either convention then fails fast.
pub(crate) fn corrected_energy_grid(
    energies: &[f64],
    t0_us: f64,
    l_scale: f64,
    flight_path_m: f64,
) -> Result<Vec<f64>, FittingError> {
    if t0_us == 0.0 && l_scale == 1.0 {
        return Ok(energies.to_vec());
    }
    let kl = TOF_FACTOR * flight_path_m;
    energies
        .iter()
        .map(|&e| {
            let tof = kl / e.sqrt();
            let denom = tof - t0_us;
            if denom <= 0.0 || !denom.is_finite() {
                return Err(FittingError::EvaluationFailed(
                    "corrected TOF ≤ 0: t0 exceeds the shortest flight time".into(),
                ));
            }
            Ok((kl * l_scale / denom).powi(2))
        })
        .collect()
}

/// The resolution-model family to calibrate.
#[derive(Debug, Clone)]
pub enum ResolutionFamily {
    /// Gaussian `(Δt_µs, ΔL_m)`.
    Gaussian,
    /// Width-corrected tabulated UDR: fit `(log s0, p)` against `base`.
    UdrCorr {
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
            | ResolutionFamily::UdrCorr { .. }
            | ResolutionFamily::IkedaCarpenter => 2,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            ResolutionFamily::Gaussian => "gaussian",
            ResolutionFamily::UdrCorr { .. } => "udr_corr",
            ResolutionFamily::IkedaCarpenter => "ic",
        }
    }

    /// `(start vector, box bounds)` for the optimizer (mirrors the validated
    /// Python reference: `udr_corr` uses log-`s0`; bounds keep widths positive).
    fn x0_bounds(&self) -> (Vec<f64>, Vec<(f64, f64)>) {
        match self {
            ResolutionFamily::Gaussian => (vec![2.0, 1e-3], vec![(1e-3, 50.0), (0.0, 0.5)]),
            ResolutionFamily::UdrCorr { .. } => {
                // (log s0, p): s0 = exp(log_s0) clamped to [0.2, 5].
                (
                    vec![0.0, 0.0],
                    vec![(UDR_S0_MIN.ln(), UDR_S0_MAX.ln()), (-4.0, 4.0)],
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
    /// Fit the SAMMY TOF-zero `t0` (µs) as a SHARED energy-scale parameter.
    /// **Default `false`** — position is pinned at [`position_t0_center_us`] so
    /// calibration is a pure shape/width fit (matching SAMMY, where `t0`/`L` are a
    /// separate energy-scale calibration). Opt in only *with* a metrology prior;
    /// see [`with_position_prior`](CalibrationConfig::with_position_prior).
    pub fit_t0: bool,
    /// Fit the flight-path scale `L_scale` as a shared energy-scale parameter.
    /// **Default `false`.** A free `L_scale` shares the asymmetric-kernel lag's
    /// `1/√E` basis and corrupts the calibrated width — fit it only under a prior.
    pub fit_l_scale: bool,
    /// Prior mean (and pinned value when [`fit_t0`](Self::fit_t0) is false) of the
    /// TOF zero `t0` (µs). Default `0.0`. Lets a caller inject a pre-calibrated `t0`.
    pub position_t0_center_us: f64,
    /// Prior mean (and pinned value when [`fit_l_scale`](Self::fit_l_scale) is
    /// false) of `L_scale`. Default `1.0`.
    pub position_l_scale_center: f64,
    /// Gaussian prior σ on `t0` (µs); `None` = flat (bounded only). When set, adds
    /// `((t0 − center)/σ)²` to the data χ² (a metrology penalty, *not* part of the
    /// reported `chi2_dof`).
    pub position_t0_prior_us: Option<f64>,
    /// Gaussian prior σ on `L_scale`; `None` = flat. See [`position_t0_prior_us`](Self::position_t0_prior_us).
    pub position_l_scale_prior: Option<f64>,
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
            // Position is PINNED by default: pure shape/width calibration on the
            // (already energy-calibrated) grid. Energy-scale fitting is an explicit
            // opt-in via `with_position_prior`.
            fit_t0: false,
            fit_l_scale: false,
            position_t0_center_us: 0.0,
            position_l_scale_center: 1.0,
            position_t0_prior_us: None,
            position_l_scale_prior: None,
        }
    }
}

impl CalibrationConfig {
    /// Enable a SHARED, metrology-priored energy-scale `(t0, L_scale)` fit: sets
    /// [`fit_t0`](Self::fit_t0)/[`fit_l_scale`](Self::fit_l_scale), the prior means
    /// (`*_center`), and the Gaussian prior σ. Use this for joint energy-scale or
    /// cross-family identifiability work; the default config pins position (pure
    /// shape/width calibration). Pass the prior σ from the instrument's independent
    /// flight-path / timing metrology — a loose σ marginalizes position (weak,
    /// honest shape-only discrimination), a tight σ pins it.
    #[must_use]
    pub fn with_position_prior(
        mut self,
        t0_center_us: f64,
        l_scale_center: f64,
        sigma_t0_us: f64,
        sigma_l_scale: f64,
    ) -> Self {
        self.fit_t0 = true;
        self.fit_l_scale = true;
        self.position_t0_center_us = t0_center_us;
        self.position_l_scale_center = l_scale_center;
        self.position_t0_prior_us = Some(sigma_t0_us);
        self.position_l_scale_prior = Some(sigma_l_scale);
        self
    }
}

/// Result of a resolution calibration.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Family label (`"gaussian"` | `"udr_corr"` | `"ic"`).
    pub family: String,
    /// Fitted parameter vector (raw optimizer space; see [`ResolutionFamily`]).
    pub theta: Vec<f64>,
    /// Reduced **data** χ²/dof of the best fit (after anorm/baseline). The
    /// energy-scale prior penalty is *excluded* — it is reported separately as
    /// [`prior_penalty`](Self::prior_penalty).
    pub chi2_dof: f64,
    /// The calibrated resolution, ready to pin into a sample fit.
    pub resolution: ResolutionFunction,
    /// Optimizer iterations of the winning restart.
    pub iterations: usize,
    /// Whether the winning restart self-converged.
    pub converged: bool,
    /// Fitted (or pinned) SAMMY energy-scale TOF zero `t0` (µs). Equals
    /// `config.position_t0_center_us` when `fit_t0` is false (pinned). When fit, it
    /// is a SHARED energy-scale parameter (not a per-family nuisance): the resonance
    /// dip position is confounded with flight-path geometry (the asymmetric-kernel
    /// lag is the same `1/√E` basis as `L_scale`), so `t0`/`L_scale` are constrained
    /// by the metrology prior, not free.
    pub position_t0_us: f64,
    /// Fitted (or pinned) flight-path scale `L_scale`. Equals
    /// `config.position_l_scale_center` when `fit_l_scale` is false.
    pub position_l_scale: f64,
    /// Gaussian-prior penalty `Σ((θ−center)/σ)²` on the fitted `(t0, L_scale)` at the
    /// solution (0 when no position prior is active). `objective = χ²_data +
    /// prior_penalty`; report it alongside `chi2_dof` so a large position move
    /// (e.g. a wrong family needing ΔL/L ≫ the metrology σ) is visible, not hidden.
    pub prior_penalty: f64,
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
        ResolutionFamily::UdrCorr { base } => {
            let s0 = theta[0].exp().clamp(UDR_S0_MIN, UDR_S0_MAX);
            let corrected = base
                .width_corrected(s0, theta[1], UDR_E_REF)
                .map_err(|e| FittingError::EvaluationFailed(format!("udr_corr width: {e}")))?;
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

/// Weighted residual sum of squares after analytically profiling out `anorm`
/// (+ optional constant+linear baseline): `data ≈ a·model (+ b0 + b1·x)`, weighted
/// by `1/unc²`. Returns `(ssr, k)` where `k` is the number of linear nuisance
/// columns (1 = anorm only, 3 = anorm+const+linear). This is the **raw** χ² (not
/// divided by dof) so an energy-scale **prior penalty** can be added to it in the
/// same units before the optimizer minimizes — adding a penalty to a *reduced* χ²
/// would silently rescale the prior by the dof. Returns `None` on a singular
/// normal-equations system (a degenerate/constant model column), so the caller can
/// treat the point as infeasible rather than as a spuriously zeroed fit.
fn inner_ssr(data: &[f64], unc: &[f64], model: &[f64], fit_bg: bool) -> Option<(f64, usize)> {
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
    let coef = solve_small(&ata, &atb, k)?;
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
    Some((ssr, k))
}

/// Reduced χ²/dof = [`inner_ssr`] `/ (n − k − n_res_params)`, ∞ on a singular
/// system. `n_res_params` counts the outer-loop parameters (resolution + any free
/// position) which are not in the linear system but still consume dof. Test-only:
/// the calibrator minimizes raw `inner_ssr` (+ prior) and reduces at the solution.
#[cfg(test)]
fn inner_chi2(data: &[f64], unc: &[f64], model: &[f64], fit_bg: bool, n_res_params: usize) -> f64 {
    match inner_ssr(data, unc, model, fit_bg) {
        Some((ssr, k)) => {
            let dof = data.len().saturating_sub(k + n_res_params).max(1) as f64;
            ssr / dof
        }
        None => f64::INFINITY,
    }
}

/// Gaussian-prior penalty `Σ((θ − center)/σ)²` on the fitted energy-scale
/// `(t0, L_scale)`. Only active coordinates (fit + prior σ set) contribute; a flat
/// (σ = `None`) or pinned coordinate contributes 0.
fn position_prior_penalty(t0_us: f64, l_scale: f64, cfg: &CalibrationConfig) -> f64 {
    let mut penalty = 0.0;
    if cfg.fit_t0
        && let Some(sigma) = cfg.position_t0_prior_us
    {
        penalty += ((t0_us - cfg.position_t0_center_us) / sigma).powi(2);
    }
    if cfg.fit_l_scale
        && let Some(sigma) = cfg.position_l_scale_prior
    {
        penalty += ((l_scale - cfg.position_l_scale_center) / sigma).powi(2);
    }
    penalty
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
/// `sample` carries the FIXED density and temperature (and isotopes/groups). By
/// default **only the resolution shape/width is optimized**, at the pinned energy
/// scale `(t0, L_scale) = (center, center)` — a pure broadening calibration on an
/// already energy-calibrated grid (this is the SAMMY split: resolution is a
/// broadening kernel; `t0`/`L` are a *separate* energy-scale calibration).
///
/// Set [`CalibrationConfig::fit_t0`]/[`fit_l_scale`](CalibrationConfig::fit_l_scale)
/// (e.g. via [`CalibrationConfig::with_position_prior`]) to *also* fit the SHARED
/// energy-scale `(t0, L_scale)` under a Gaussian metrology prior — for joint
/// energy-scale work or a cross-family identifiability study. Do **not** fit
/// position with a flat prior in production: the asymmetric-kernel mode→centroid
/// lag is the same `1/√E` basis as `L_scale`, so a free `L_scale` absorbs the lag
/// and corrupts the calibrated width.
///
/// Returns the fitted shape parameters, the reduced **data** χ²/dof, the fitted (or
/// pinned) `(t0, L_scale)`, the prior penalty, and the calibrated
/// [`ResolutionFunction`] (ready to pin).
///
/// # Errors
/// [`FittingError::EmptyData`] / [`FittingError::LengthMismatch`] for bad
/// inputs; [`FittingError::InvalidConfig`] for a bad grid or position config;
/// propagates optimizer errors.
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
    // Reject under-determined calibrants: need strictly more data points than the
    // total free parameters (resolution + any *fitted* position + anorm/baseline).
    let n_res = family.n_params();
    let n_pos = usize::from(config.fit_t0) + usize::from(config.fit_l_scale);
    let baseline_cols = if config.fit_background { 3 } else { 1 };
    if data.len() <= n_res + n_pos + baseline_cols {
        return Err(FittingError::InvalidConfig(format!(
            "calibrant has {} points but the model has {} resolution + {} position + {} baseline \
             parameters; need strictly more data points than parameters",
            data.len(),
            n_res,
            n_pos,
            baseline_cols,
        )));
    }
    // Flight path is a physical positive length. A non-positive / non-finite value
    // would invert the t0 feasibility bound below (min_tof < 0 ⇒ t0_hi < t0_lo ⇒
    // `clamp(lo, hi)` panic) as soon as `fit_t0` appends a bounded coordinate, and
    // otherwise only surfaces as a generic "no finite-objective" error. Reject it
    // precisely up front (covers every family and the fit/pin paths alike).
    if !(config.flight_path_m.is_finite() && config.flight_path_m > 0.0) {
        return Err(FittingError::InvalidConfig(
            "flight_path_m must be finite and > 0".into(),
        ));
    }
    // Validate the energy-scale (t0, L_scale) prior/center configuration up front.
    if !config.position_t0_center_us.is_finite()
        || !config.position_l_scale_center.is_finite()
        || config.position_l_scale_center <= 0.0
    {
        return Err(FittingError::InvalidConfig(
            "position centers must be finite and the L_scale center > 0".into(),
        ));
    }
    if config.position_t0_center_us.abs() >= POSITION_T0_US_MAX {
        return Err(FittingError::InvalidConfig(format!(
            "position_t0_center_us must lie within ±{POSITION_T0_US_MAX} µs"
        )));
    }
    if config.position_l_scale_center < POSITION_L_SCALE_MIN
        || config.position_l_scale_center > POSITION_L_SCALE_MAX
    {
        return Err(FittingError::InvalidConfig(format!(
            "position_l_scale_center must lie within [{POSITION_L_SCALE_MIN}, {POSITION_L_SCALE_MAX}]"
        )));
    }
    for (sigma, name) in [
        (config.position_t0_prior_us, "position_t0_prior_us"),
        (config.position_l_scale_prior, "position_l_scale_prior"),
    ] {
        if let Some(s) = sigma
            && !(s.is_finite() && s > 0.0)
        {
            return Err(FittingError::InvalidConfig(format!(
                "{name} must be finite and > 0 when set"
            )));
        }
    }

    let e_min = energies.first().copied().unwrap_or(1.0);
    let e_max = energies.last().copied().unwrap_or(1.0);
    // Feasible t0 upper bound: the corrected TOF `tof − t0` must stay positive for
    // every energy, i.e. `t0 < min(tof) = TOF_FACTOR·L/√E_max`. Far outside ±5 µs in
    // the eV regime, but clamp defensively so a wide window can never make it bite.
    let min_tof = TOF_FACTOR * config.flight_path_m / e_max.max(1e-12).sqrt();
    // The (pinned or prior-mean) t0 center must itself be feasible: corrected_energy_grid
    // needs `t0 < min(tof)` for every energy. In the eV regime min_tof ≫ 5 µs, but a
    // short flight path or very high E_max can shrink it — reject up front with a precise
    // message instead of a late, generic "corrected TOF ≤ 0" from the final recompute.
    if config.position_t0_center_us >= min_tof {
        return Err(FittingError::InvalidConfig(format!(
            "position_t0_center_us ({:.3} µs) must be below the shortest flight time \
             min_tof = TOF_FACTOR·L/√E_max = {min_tof:.3} µs",
            config.position_t0_center_us
        )));
    }
    let t0_lo = -POSITION_T0_US_MAX;
    let t0_hi = POSITION_T0_US_MAX.min(min_tof - 1e-6);

    // Optimizer coordinates: [resolution params (n_res)..., t0?, L_scale?]. A
    // position coordinate is appended only when fit; otherwise it is pinned at its
    // center. (Position is a SHARED energy-scale parameter, not a per-family
    // nuisance — fitting it is an explicit, prior-constrained opt-in.)
    let (mut x0, mut bounds) = family.x0_bounds();
    if config.fit_t0 {
        x0.push(config.position_t0_center_us.clamp(t0_lo, t0_hi));
        bounds.push((t0_lo, t0_hi));
    }
    if config.fit_l_scale {
        x0.push(
            config
                .position_l_scale_center
                .clamp(POSITION_L_SCALE_MIN, POSITION_L_SCALE_MAX),
        );
        bounds.push((POSITION_L_SCALE_MIN, POSITION_L_SCALE_MAX));
    }
    let nm = NelderMeadConfig {
        xatol: config.xatol,
        fatol: config.fatol,
        max_iter: config.max_iter,
        ..Default::default()
    };

    // Read the (possibly pinned) position coordinates out of an optimizer vector.
    let unpack_position = |theta: &[f64]| -> (f64, f64) {
        let mut idx = n_res;
        let t0 = if config.fit_t0 {
            let v = theta[idx];
            idx += 1;
            v
        } else {
            config.position_t0_center_us
        };
        let l_scale = if config.fit_l_scale {
            theta[idx]
        } else {
            config.position_l_scale_center
        };
        (t0, l_scale)
    };

    let mut best: Option<NelderMeadResult> = None;
    for r in 0..config.restarts.max(1) {
        // Additive perturbation (a fraction of each parameter's bound range) so
        // restarts move even for zero-valued start components — a multiplicative
        // `x0·(1+0.1r)` left `udr_corr`'s `[0, 0]` start identical every restart.
        let start: Vec<f64> = x0
            .iter()
            .zip(&bounds)
            .map(|(&v, &(lo, hi))| (v + 0.1 * r as f64 * (hi - lo)).clamp(lo, hi))
            .collect();
        let obj = |theta: &[f64]| -> Result<f64, FittingError> {
            // theta = [resolution params (n_res)..., t0?, L_scale?]. The resolution
            // kernel uses only the first n_res; (t0, L_scale) set the energy scale.
            let res = build_resolution(&family, theta, e_min, e_max, config)?;
            let inst = InstrumentParams { resolution: res };
            let (t0, l_scale) = unpack_position(theta);
            // Infeasible energy scale (corrected TOF ≤ 0) → step away.
            let Ok(grid) = corrected_energy_grid(energies, t0, l_scale, config.flight_path_m)
            else {
                return Ok(f64::INFINITY);
            };
            let model = forward_model(&grid, sample, Some(&inst))
                .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))?;
            if !model.iter().all(|v| v.is_finite()) {
                return Err(FittingError::EvaluationFailed("non-finite model".into()));
            }
            // Minimize RAW χ²_data + metrology prior penalty (same units — adding
            // the penalty to a reduced χ² would rescale the prior by the dof).
            let Some((ssr, _k)) = inner_ssr(data, unc, &model, config.fit_background) else {
                return Ok(f64::INFINITY);
            };
            Ok(ssr + position_prior_penalty(t0, l_scale, config))
        };
        let res = nelder_mead_minimize(obj, &start, Some(&bounds), &nm)?;
        if best.as_ref().is_none_or(|b| res.fun < b.fun) {
            best = Some(res);
        }
    }
    let best = best.expect("at least one restart runs");
    if !best.fun.is_finite() {
        return Err(FittingError::EvaluationFailed(
            "calibration found no finite-objective resolution (the forward model failed for \
             every parameter vector tried)"
                .into(),
        ));
    }
    let (position_t0_us, position_l_scale) = unpack_position(&best.x);
    let prior_penalty = position_prior_penalty(position_t0_us, position_l_scale, config);
    // Recompute the reduced DATA χ²/dof at the solution: the objective carries the
    // prior penalty, so `best.fun` is the penalized objective, not the data χ². dof
    // subtracts the linear anorm/baseline columns AND the outer-loop params
    // (resolution + any fitted position).
    let resolution = build_resolution(&family, &best.x, e_min, e_max, config)?;
    let grid = corrected_energy_grid(
        energies,
        position_t0_us,
        position_l_scale,
        config.flight_path_m,
    )?;
    let inst = InstrumentParams { resolution };
    let model = forward_model(&grid, sample, Some(&inst))
        .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))?;
    let (ssr, k) = inner_ssr(data, unc, &model, config.fit_background).ok_or_else(|| {
        FittingError::EvaluationFailed("singular anorm/baseline at the solution".into())
    })?;
    let dof = data.len().saturating_sub(k + n_res + n_pos).max(1) as f64;
    let chi2_dof = ssr / dof;
    let theta = best.x[..n_res].to_vec();
    Ok(CalibrationResult {
        family: family.label().to_string(),
        theta,
        chi2_dof,
        resolution: inst.resolution,
        iterations: best.iterations,
        converged: best.self_converged,
        position_t0_us,
        position_l_scale,
        prior_penalty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::test_support::synthetic_isotope;

    fn synthetic_base_udr() -> TabulatedResolution {
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
    fn udr_corr_recovers_known_width_scale() {
        // Loop-closure / OPTIMIZER test: truth and fit both use width_corrected, so
        // this checks that the calibrator finds the s0=1.5 minimum — NOT that
        // width_corrected itself is physically correct. The width-scale physics
        // (centroid invariance + std scaling) is independently verified by
        // `width_corrected_preserves_centroid_scales_width_and_energy_dependence`
        // in nereids-physics.
        // Two well-separated resonances (15 + 45 eV) so the width is identifiable
        // (a single resonance leaves a width↔position ridge). Position is pinned by
        // default. Calibrant generated with a UDR truth scaled by s0=1.5; udr_corr
        // must recover s0≈1.5 at χ²≈0.
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udr();
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(1.5, 0.0, UDR_E_REF).unwrap(),
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
            ResolutionFamily::UdrCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let s0 = r.theta[0].exp().clamp(UDR_S0_MIN, UDR_S0_MAX);
        assert!((s0 - 1.5).abs() < 0.05, "recovered s0={s0}, expected 1.5");
        assert!(r.chi2_dof < 1e-2, "matched χ²/dof={} too high", r.chi2_dof);
        assert!(matches!(r.resolution, ResolutionFunction::Tabulated(_)));
    }

    #[test]
    fn udr_corr_recovers_known_width_scale_and_exponent() {
        // Two resonances at well-separated energies make the width EXPONENT p
        // identifiable — a single resonance constrains only s(E) at one energy (a
        // ridge in (s0, p)). Truth: s0=1.3, p=-0.5; the calibrator must recover
        // both (the s0-only test never exercised the p knob).
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udr();
        let (s0_true, p_true) = (1.3, -0.5);
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(s0_true, p_true, UDR_E_REF).unwrap(),
        ));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig {
            restarts: 3,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::UdrCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let s0 = r.theta[0].exp().clamp(UDR_S0_MIN, UDR_S0_MAX);
        let p = r.theta[1];
        assert!(
            (s0 - s0_true).abs() < 0.1,
            "recovered s0={s0}, expected {s0_true}"
        );
        assert!(
            (p - p_true).abs() < 0.2,
            "recovered p={p}, expected {p_true}"
        );
        assert!(r.chi2_dof < 1e-2, "χ²/dof={} too high", r.chi2_dof);
    }

    #[test]
    fn udr_corr_recovers_independent_raw_kernel() {
        // External-oracle coverage: the truth resolution is the RAW hand-built UDR
        // kernel broadened directly — it does NOT pass through `width_corrected`,
        // so truth-generation no longer shares the width-correction code with the
        // fit. Fitting udr_corr against that base must recover the identity width
        // (s0≈1) at χ²≈0. (The broadening OPERATOR itself is independently
        // validated by this crate's bit-exact `broaden_presorted_reference` tests.)
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udr();
        // Truth = the RAW base kernel (no width_corrected call).
        let truth = ResolutionFunction::Tabulated(Arc::new(base.clone()));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig {
            restarts: 3,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::UdrCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let s0 = r.theta[0].exp().clamp(UDR_S0_MIN, UDR_S0_MAX);
        assert!((s0 - 1.0).abs() < 0.1, "recovered s0={s0}, expected ~1.0");
        assert!(r.chi2_dof < 1e-2, "χ²/dof={} too high", r.chi2_dof);
    }

    #[test]
    fn gaussian_recovers_known_width() {
        // Gaussian loop-closure: a Gaussian truth must be recovered by the gaussian
        // family (the smoke test only checked finiteness+convergence). Two
        // resonances break the Δt/ΔL degeneracy (Δt is flat in TOF; ΔL scales with
        // TOF ∝ 1/√E).
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let (dt_true, dl_true) = (1.5, 1.0e-3);
        let truth = ResolutionFunction::Gaussian(
            ResolutionParams::new(25.0, dt_true, dl_true, 0.0).unwrap(),
        );
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig {
            restarts: 3,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::Gaussian,
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let (dt, dl) = (r.theta[0].abs(), r.theta[1].abs());
        assert!(r.chi2_dof < 1e-2, "χ²/dof={} too high", r.chi2_dof);
        assert!(
            (dt - dt_true).abs() < 0.2,
            "recovered Δt={dt}, expected {dt_true}"
        );
        assert!(
            (dl - dl_true).abs() < 1.0e-3,
            "recovered ΔL={dl}, expected {dl_true}"
        );
    }

    #[test]
    fn fit_t0_recovers_injected_energy_scale_shift() {
        // With fit_t0 enabled, an injected TOF-zero offset in the calibrant is
        // recovered as the SHARED energy-scale t0 while the width is still
        // recovered — position is a fitted energy-scale parameter (−t0 convention),
        // not folded into the resolution. (Default config pins position; this test
        // opts in.) L_scale stays pinned at 1.
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udr();
        let (s0_true, t0_inject) = (1.4, 1.5_f64); // µs (energy-scale −t0 convention)
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(s0_true, 0.0, UDR_E_REF).unwrap(),
        ));
        // Calibrant generated on a grid displaced by the energy-scale t0.
        let shifted = corrected_energy_grid(&energies, t0_inject, 1.0, 25.0).unwrap();
        let data = forward_model(
            &shifted,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        // Opt into fitting t0 (flat prior); L_scale stays pinned at 1.
        let cfg = CalibrationConfig {
            restarts: 3,
            fit_t0: true,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::UdrCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let s0 = r.theta[0].exp().clamp(UDR_S0_MIN, UDR_S0_MAX);
        assert!(
            (s0 - s0_true).abs() < 0.1,
            "recovered s0={s0}, expected {s0_true}"
        );
        assert!(
            (r.position_t0_us - t0_inject).abs() < 0.3,
            "recovered t0={}, expected {t0_inject}",
            r.position_t0_us
        );
        assert!(
            (r.position_l_scale - 1.0).abs() < 1e-9,
            "L_scale should stay pinned at 1, got {}",
            r.position_l_scale
        );
        assert!(r.chi2_dof < 1e-2, "χ²/dof={} too high", r.chi2_dof);
    }

    #[test]
    fn pinned_position_is_the_default_and_works_for_udr() {
        // The default config pins position (fit_t0/fit_l_scale = false) — a pure
        // shape/width fit. This is the no-position reference that the retired design
        // could NOT construct for the UDR family in Python (the width-correction was
        // Rust-internal). Self-fit must recover s0≈1 with position reported at its
        // pinned center and zero prior penalty.
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udr();
        let truth = ResolutionFunction::Tabulated(Arc::new(base.clone()));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig::default();
        assert!(!cfg.fit_t0 && !cfg.fit_l_scale, "default must pin position");
        let r = calibrate_resolution(
            ResolutionFamily::UdrCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let s0 = r.theta[0].exp().clamp(UDR_S0_MIN, UDR_S0_MAX);
        assert!((s0 - 1.0).abs() < 0.1, "recovered s0={s0}, expected ~1.0");
        assert_eq!(r.position_t0_us, 0.0, "t0 pinned at center 0");
        assert_eq!(r.position_l_scale, 1.0, "L_scale pinned at center 1");
        assert_eq!(r.prior_penalty, 0.0, "no prior active when pinned");
        assert!(r.chi2_dof < 1e-2, "χ²/dof={} too high", r.chi2_dof);
    }

    #[test]
    fn free_l_scale_absorbs_asymmetric_lag_and_erodes_discrimination() {
        // The asymmetric IC mode→centroid lag is pure 1/√E — the SAME basis as an
        // L_scale error. So a Gaussian fitting an IC-broadened calibrant fits much
        // BETTER when L_scale is free than when position is pinned: a free physical
        // position lets the wrong (symmetric) family buy back the position evidence.
        // This is exactly why fitting position with a flat prior is unsafe for
        // family discrimination (and why the default pins it).
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let ic = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.30, a1: 0.0 },
                beta: IC_FIXED_BETA,
                r: EnergyLaw::ExpMilliEv { kappa: 25.0 },
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            25.0,
            &SynthesisGrid {
                e_min_ev: 4.0,
                e_max_ev: 100.0,
                n_energies: 64,
                n_tau: 500,
            },
        )
        .unwrap();
        let truth = ResolutionFunction::IkedaCarpenter(Arc::new(ic));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let pinned = CalibrationConfig {
            restarts: 3,
            ..Default::default()
        };
        // Free physical position (flat priors): fit_t0 + fit_l_scale, sigmas None.
        let free_pos = CalibrationConfig {
            restarts: 3,
            fit_t0: true,
            fit_l_scale: true,
            ..Default::default()
        };
        let gau = |cfg: &CalibrationConfig| {
            calibrate_resolution(
                ResolutionFamily::Gaussian,
                &energies,
                &data,
                &unc,
                &sample,
                cfg,
            )
            .unwrap()
            .chi2_dof
        };
        let gau_pinned = gau(&pinned);
        let gau_free = gau(&free_pos);
        assert!(
            gau_free < 0.5 * gau_pinned,
            "free (t0,L_scale) should sharply erode the wrong-family penalty: \
             pinned χ²={gau_pinned}, free χ²={gau_free}"
        );
    }

    #[test]
    fn position_prior_penalizes_displacement() {
        // A tight prior on t0 (center 0) penalizes a calibrant whose true t0 is
        // displaced: the fit cannot freely move to the displacement, so it pays a
        // prior penalty and leaves residual data χ². A loose prior recovers the
        // displacement with ~zero penalty. (Demonstrates the prior is the real
        // constraint on position, per the metrology-prior design.)
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udr();
        let t0_inject = 1.5_f64;
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(1.0, 0.0, UDR_E_REF).unwrap(),
        ));
        let shifted = corrected_energy_grid(&energies, t0_inject, 1.0, 25.0).unwrap();
        let data = forward_model(
            &shifted,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let mk = |sigma_t0: f64| CalibrationConfig {
            restarts: 3,
            fit_t0: true,
            position_t0_prior_us: Some(sigma_t0),
            ..Default::default()
        };
        let mkbase = || ResolutionFamily::UdrCorr {
            base: Arc::new(base.clone()),
        };
        // Tight prior (σ=0.2 µs ≪ 1.5 µs displacement): can't reach t0, pays penalty.
        let tight =
            calibrate_resolution(mkbase(), &energies, &data, &unc, &sample, &mk(0.2)).unwrap();
        // Loose prior (σ=100 µs): recovers the displacement, ~no penalty.
        let loose =
            calibrate_resolution(mkbase(), &energies, &data, &unc, &sample, &mk(100.0)).unwrap();
        assert!(
            tight.prior_penalty > 1.0,
            "tight prior should incur a real penalty, got {}",
            tight.prior_penalty
        );
        assert!(
            tight.position_t0_us.abs() < t0_inject,
            "tight prior should pull t0 toward the center, got {}",
            tight.position_t0_us
        );
        assert!(
            (loose.position_t0_us - t0_inject).abs() < 0.3,
            "loose prior should recover the displacement, got {}",
            loose.position_t0_us
        );
        assert!(
            loose.prior_penalty < tight.prior_penalty,
            "loose penalty {} should be below tight penalty {}",
            loose.prior_penalty,
            tight.prior_penalty
        );
        assert!(
            loose.chi2_dof < tight.chi2_dof,
            "loose data χ² {} should beat tight data χ² {} (tight can't reach t0)",
            loose.chi2_dof,
            tight.chi2_dof
        );
    }

    #[test]
    fn with_position_prior_builder_sets_fields() {
        let cfg = CalibrationConfig::default().with_position_prior(0.5, 1.001, 0.3, 0.002);
        assert!(cfg.fit_t0 && cfg.fit_l_scale);
        assert_eq!(cfg.position_t0_center_us, 0.5);
        assert_eq!(cfg.position_l_scale_center, 1.001);
        assert_eq!(cfg.position_t0_prior_us, Some(0.3));
        assert_eq!(cfg.position_l_scale_prior, Some(0.002));
    }

    #[test]
    fn fit_l_scale_only_pins_t0() {
        // Per-coordinate control: fitting ONLY L_scale (fit_t0=false) must fit
        // position_l_scale while pinning t0 at its center — locks the
        // `unpack_position` indexing when only the SECOND position coordinate is
        // active (the single-coordinate path the round-2 review flagged).
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let base = synthetic_base_udr();
        let truth = ResolutionFunction::Tabulated(Arc::new(base.clone()));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig {
            restarts: 2,
            fit_l_scale: true,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::UdrCorr {
                base: Arc::new(base),
            },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        assert_eq!(
            r.position_t0_us, 0.0,
            "t0 must stay pinned when fit_t0=false"
        );
        assert!(
            (r.position_l_scale - 1.0).abs() < 0.02,
            "L_scale fit within bound (~1 for a self-fit), got {}",
            r.position_l_scale
        );
        assert!(r.chi2_dof < 1e-1, "self-fit χ²/dof={} too high", r.chi2_dof);
    }

    #[test]
    fn cross_family_chi2_selects_the_true_shape() {
        // Model-family discrimination at a KNOWN (pinned) energy scale: an
        // asymmetric IC-broadened calibrant generated at the nominal position
        // (t0=0, L_scale=1) must be best-fit by the IC family and clearly worse by
        // the symmetric Gaussian. With position pinned (the default), the Gaussian
        // is penalized for both shape AND the asymmetry-induced dip shift it cannot
        // reproduce — legitimate here because the truth's position is known exactly.
        // (When position is uncertain, that shift is confounded with flight-path L —
        // see `free_l_scale_absorbs_asymmetric_lag_and_erodes_discrimination`.)
        // Truth has NO width-correction/Gaussian generator, so the Gaussian arm is a
        // genuinely different shape (not loop-closure).
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..700).map(|i| 8.0 + i as f64 * 0.06).collect();
        let ic = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.30, a1: 0.0 },
                beta: IC_FIXED_BETA,
                r: EnergyLaw::ExpMilliEv { kappa: 25.0 },
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            25.0,
            &SynthesisGrid {
                e_min_ev: 4.0,
                e_max_ev: 100.0,
                n_energies: 64,
                n_tau: 500,
            },
        )
        .unwrap();
        let truth = ResolutionFunction::IkedaCarpenter(Arc::new(ic));
        let data = forward_model(
            &energies,
            &sample,
            Some(&InstrumentParams { resolution: truth }),
        )
        .unwrap();
        let unc = vec![0.004; energies.len()];
        let cfg = CalibrationConfig {
            restarts: 3,
            ..Default::default()
        };
        let chi2 = |fam| {
            calibrate_resolution(fam, &energies, &data, &unc, &sample, &cfg)
                .unwrap()
                .chi2_dof
        };
        let ic_chi2 = chi2(ResolutionFamily::IkedaCarpenter);
        let gau_chi2 = chi2(ResolutionFamily::Gaussian);
        assert!(
            ic_chi2 < gau_chi2,
            "true (IC) shape χ²={ic_chi2} should beat the Gaussian χ²={gau_chi2}"
        );
        assert!(
            ic_chi2 < 1.0,
            "IC (true shape) should fit well: χ²={ic_chi2}"
        );
    }

    #[test]
    fn gaussian_and_ic_families_run_and_converge() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..300).map(|i| 12.0 + i as f64 * 0.05).collect();
        let base = synthetic_base_udr();
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(1.2, 0.0, UDR_E_REF).unwrap(),
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
            ResolutionFamily::UdrCorr {
                base: Arc::new(synthetic_base_udr())
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
        let base = synthetic_base_udr();
        let truth = ResolutionFunction::Tabulated(Arc::new(
            base.width_corrected(1.3, 0.0, UDR_E_REF).unwrap(),
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
            ResolutionFamily::UdrCorr {
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
        // Loop-closure / optimizer test (same caveat as udr_corr): truth and fit
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
