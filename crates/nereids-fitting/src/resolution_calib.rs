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
//! - **IkedaCarpenter** — physics-complete bounded moderator fit (#642):
//!   `α(E) = e^{θ0}·√E + e^{θ1}` (positive at every energy by construction),
//!   `β = e^{θ2}` (bounded), scalar storage fraction `R = θ3 ∈ [0, 1]`, all
//!   folded with the SNS PSR channel triangle
//!   ([`CalibrationConfig::psr_fwhm_ns`], default 350 ns; optionally fitted
//!   via `fit_psr`). Beware the β↔R ridge: as `R → 0` the storage term
//!   vanishes and β is unconstrained — such a fit reports `"r:lower"` in
//!   [`CalibrationResult::bounds_hit`] and its β carries no information.

use std::sync::Arc;

use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    ResolutionFunction, ResolutionParams, TOF_FACTOR, TabulatedResolution, apply_resolution,
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
// --- Ikeda–Carpenter calibration family: θ encoding and physics bounds (#642) ---
//
// θ = [ln a0, ln a1, ln β, R, (PSR FWHM µs iff fit_psr)]. The rates are
// exp-encoded so that `α(E) = e^{θ0}·√E + e^{θ1} > 0` at EVERY energy — on the
// calibration grid and on any production grid the pinned kernel is later
// applied to — by construction (a real calibration under the old plain-(a0,a1)
// box once returned a1 = −0.396, which drives α(E) < 0 at low energy and makes
// the pulse unphysical). β and R are FREE: the 2-parameter predecessor (β
// pinned, R ≡ exp(−E_meV/25) ≈ 0 in the eV regime) lacked the storage-shape
// freedom, which re-expressed as a ~90 K temperature degeneracy on real data.

/// Lower box bound on the prompt-rate coefficient `a0` (µs⁻¹ per √eV) of
/// `α(E) = a0·√E + a1`; same span as the previous plain box `(0.01, 5.0)`.
/// α ≈ 1–3 µs⁻¹ in the eV regime (Ikeda & Carpenter, NIM A239 (1985) 536)
/// gives a0 ≈ 0.2–0.5; the decade of head-room on each side is deliberate.
const IC_A0_MIN: f64 = 0.01;
/// Upper box bound on `a0`; see [`IC_A0_MIN`].
const IC_A0_MAX: f64 = 5.0;
/// Optimizer start for `a0` — the VENUS-scale prompt slope used since the
/// family was introduced (matches the previous start).
const IC_A0_X0: f64 = 0.30;
/// Lower box bound on the energy-independent prompt offset `a1` (µs⁻¹).
/// Strictly positive (exp-encoded) so `α(E) → a1 > 0` as `E → 0`: the offset
/// can no longer flip α negative below the calibration window.
const IC_A1_MIN: f64 = 1e-3;
/// Upper box bound on `a1`: 2 µs⁻¹ already exceeds the whole prompt rate at
/// 1 eV for any plausible a0, so the bound is not physically restrictive.
const IC_A1_MAX: f64 = 2.0;
/// Optimizer start for `a1` — small positive (a mostly-√E law).
const IC_A1_X0: f64 = 0.05;
/// Lower box bound on the storage (slow) rate `β` (µs⁻¹). Covers the
/// canonical Ikeda–Carpenter ambient-moderator value β ≈ 0.031 µs⁻¹
/// (NIM A239 (1985) 536; also Mantid `IkedaCarpenterPV`'s β default) with
/// margin below it. The τ-grid is prompt-anchored and capped (nereids-physics
/// `MAX_TAU_SAMPLES`), so the 16/β ≈ 800 µs tail at this bound stays sampled
/// at a ≈ 0.098 µs capped step — fine enough for the default 0.35 µs (and
/// any ≥ ~0.3 µs) PSR triangle and for prompt rates up to α ≈ 26 µs⁻¹. A
/// fitted PSR near its 0.05 µs floor combined with β near this bound is
/// unresolvable within the cap; such θ are treated as infeasible points
/// (∞ objective) during the search, never as a calibration abort — see
/// `ic_box_worst_corner_synthesizes_within_tau_cap` /
/// `ic_unresolvable_theta_errs_in_build_resolution` /
/// `ic_infeasible_pocket_inside_box_completes_calibration`.
const IC_BETA_MIN: f64 = 0.02;
/// Upper box bound on `β`: at 5 µs⁻¹ the storage tail is as fast as the
/// prompt core itself (α range), beyond which β↔α are indistinguishable.
const IC_BETA_MAX: f64 = 5.0;
/// Optimizer start for `β` — the value the retired fixed-β family pinned.
const IC_BETA_X0: f64 = 0.10;
/// Lower box bound on the storage mixing fraction `R` (physical: a fraction).
const IC_R_MIN: f64 = 0.0;
/// Upper box bound on `R`; see [`IC_R_MIN`].
const IC_R_MAX: f64 = 1.0;
/// Optimizer start for `R`. A scalar `R` replaces the retired
/// `ExpMilliEv{κ=25}` law: a free κ is unidentifiable in the eV regime
/// (R ≡ 0 across 1–200 eV for ANY plausible κ), whereas a scalar R lets the
/// data decide whether a storage tail is present at all.
const IC_R_X0: f64 = 0.1;
/// Default SNS PSR (proton-storage-ring / accumulator) channel-triangle FWHM
/// in **ns**, folded into the IC family's kernel. The SNS proton pulse is
/// shaped by the accumulator ring into an ~triangular ~700 ns base (FWHM ≈
/// 350 ns) — the VENUS tabulated FTS kernel header records exactly this
/// ("folded triang FWHM 350 ns PSR"). SAMMY's analog is the Gaussian-burst
/// FWHM `DELTAG` (Manual Sec. III.C.1.a, eq. III C1 a.12) or square `BURST`
/// width (Sec. III.C.2.a). `pub` so the Python binding's default and the Rust
/// default cannot drift apart.
pub const DEFAULT_PSR_FWHM_NS: f64 = 350.0;
/// Lower box bound (µs) on a FITTED PSR triangle FWHM (`fit_psr = true`):
/// below 50 ns the triangle is far under the IC prompt width in the eV
/// regime and unidentifiable.
const PSR_FWHM_US_MIN: f64 = 0.05;
/// Upper box bound (µs) on a fitted PSR FWHM: 1 µs is ~3× the physical SNS
/// pulse base — anything larger is the moderator's job (α, β), not the burst.
const PSR_FWHM_US_MAX: f64 = 1.0;
/// Sanity ceiling (µs) on the configured PSR triangle FWHM: one decade above
/// the `PSR_FWHM_US_MAX` fit bound. [`CalibrationConfig::psr_fwhm_ns`] is in
/// NANOSECONDS (the VENUS FTS header convention: "folded triang FWHM 350 ns
/// PSR"), and kernel-synthesis cost grows QUADRATICALLY with a wide fold's
/// width. The mechanism is NOT τ-step refinement — that applies only to
/// folds FINER than the prompt design step, and a 50–350 µs fold's FWHM/3
/// resolution floor is far coarser, leaving the step unchanged — it is the
/// convolution itself: the ±FWHM fold-reach margin adds O(FWHM/step)
/// τ-samples, each folded in `convolve_same` against a sampled triangle
/// itself O(FWHM/step) long. Measured ~12 ms at 0.35 µs but ~1.3 s at 50 µs
/// and ~28 s at 350 µs per single kernel-table synthesis at the default
/// grid. A µs-as-ns unit slip
/// (passing `350` meaning µs → interpreted as a 350 µs pin) would therefore
/// turn a calibration into a multi-hour silent hang behind a physically
/// fictitious fold. Any genuine width sits inside the fitted box; one decade
/// of headroom keeps deliberate sensitivity studies possible while still
/// catching the 1000× ns↔µs slip. `0.0` (fold disabled) is always accepted.
/// `pub` for parity with the Python binding's mirrored validation.
pub const PSR_FWHM_PIN_CEILING_US: f64 = 10.0 * PSR_FWHM_US_MAX;
/// Nanoseconds → microseconds ([`CalibrationConfig::psr_fwhm_ns`] is in ns to
/// match the FTS header convention; the kernel synthesis takes µs). `pub` so
/// the Python binding's mirrored [`PSR_FWHM_PIN_CEILING_US`] check converts
/// with the identical factor.
pub const NS_TO_US: f64 = 1e-3;
/// A coordinate within this fraction of its box range of a bound is reported
/// in [`CalibrationResult::bounds_hit`] as pinned.
const BOUND_HIT_REL_TOL: f64 = 1e-3;
/// Cap on per-restart Nelder–Mead simplex re-inflations (fresh simplex
/// restarted at the incumbent while it keeps improving by more than `fatol`).
/// Guards against premature simplex collapse — see the re-inflation comment
/// in [`calibrate_resolution`] — while guaranteeing termination.
const MAX_SIMPLEX_REINFLATIONS: usize = 5;
/// Initial-step fraction for the RE-INFLATED simplex (vs the default 0.05 of
/// the first descent). A collapsed simplex rebuilt at the same 5 % scale
/// deterministically re-collapses to the same trap (observed on the IC
/// family's curved α↔β↔R valley); a 25 % edge straddles the valley and lets
/// the restarted simplex see the descent direction.
const REINFLATE_STEP_FRAC: f64 = 0.25;
/// Absolute re-inflation step for near-zero coordinates (`|x| < 1e-8`),
/// matching the box scale of the bounded coordinates (R ∈ [0, 1]).
const REINFLATE_STEP_ABS: f64 = 0.1;
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
/// SAMMY reference for the `−t0` sign convention: `dat/mdat0.f90:189`
/// (`etzero = ee*(Elzero/(1−Tzero·√ee/Tttzzz))²`) — the measured TOF has TZERO
/// subtracted before the energy conversion, so a positive `t0` raises the
/// corrected energy. This is the canonical NEREIDS implementation of the
/// formula; the runtime
/// `EnergyScaleTransmissionModel::corrected_energies` is pinned bit-for-bit
/// to it by `corrected_energy_grid_matches_energy_scale_model`, and
/// `SpectrumFitResult::corrected_energies` (issue #634) reuses it so callers
/// never re-derive the transform (a `+t0` slip caused a silent +400 K bias).
pub fn corrected_energy_grid(
    energies: &[f64],
    t0_us: f64,
    l_scale: f64,
    flight_path_m: f64,
) -> Result<Vec<f64>, FittingError> {
    // Validate scale inputs up front (issue #634 review): the transform is
    // EVEN in `l_scale` (`(kl·l_scale/denom)²`), so a negative `l_scale`
    // would silently return the identical plausible grid as its positive
    // counterpart, a NaN `l_scale` would pass the denominator-only guard and
    // return `Ok(vec![NaN])` ("NaN bypasses guards"), and
    // `flight_path_m = 0` with a negative fitted `t0` would return an
    // all-zeros grid as `Ok`.  Matches the sibling fit entry points'
    // `validate_energy_scale_params` rejection (issue #458) — this is the
    // canonical public transform, so it must not hand back plausible
    // garbage for invalid inputs.
    if !t0_us.is_finite() {
        return Err(FittingError::EvaluationFailed(format!(
            "corrected_energy_grid: t0_us must be finite, got {t0_us}"
        )));
    }
    if !l_scale.is_finite() || l_scale <= 0.0 {
        return Err(FittingError::EvaluationFailed(format!(
            "corrected_energy_grid: l_scale must be finite and positive, got {l_scale}"
        )));
    }
    if !flight_path_m.is_finite() || flight_path_m <= 0.0 {
        return Err(FittingError::EvaluationFailed(format!(
            "corrected_energy_grid: flight_path_m must be finite and positive, \
             got {flight_path_m}"
        )));
    }
    // Per-bin grid validation runs BEFORE the identity shortcut so the
    // Ok/Err contract is uniform: previously (t0=0, l_scale=1) returned a
    // NaN/non-positive grid verbatim while any other transform rejected the
    // same grid via the denominator guard (#634 review).  Empty grids are
    // rejected for the same uniformity (the Python binding already errors
    // on them; a per-bin loop is vacuous on an empty slice).
    if energies.is_empty() {
        return Err(FittingError::EvaluationFailed(
            "corrected_energy_grid: energies must not be empty".into(),
        ));
    }
    for (i, &e) in energies.iter().enumerate() {
        if !e.is_finite() || e <= 0.0 {
            return Err(FittingError::EvaluationFailed(format!(
                "corrected_energy_grid: energies[{i}] must be finite and positive, got {e}"
            )));
        }
        // Strict ascending order, matching the Python binding's standard
        // energy-grid validation (issue #634 review): a non-monotone input
        // would otherwise map to a plausible but non-monotone corrected axis.
        if i > 0 && e <= energies[i - 1] {
            return Err(FittingError::EvaluationFailed(format!(
                "corrected_energy_grid: energies must be strictly ascending; \
                 energies[{i}] = {e} <= energies[{}] = {}",
                i - 1,
                energies[i - 1],
            )));
        }
    }
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
    /// Ikeda–Carpenter, physics-complete and bounded (#642):
    /// `θ = [ln a0, ln a1, ln β, R]` with `α(E) = e^{θ0}√E + e^{θ1}` (positive
    /// by construction), `β = e^{θ2}` bounded, scalar `R = θ3 ∈ [0, 1]`, all
    /// folded with the SNS PSR channel triangle
    /// ([`CalibrationConfig::psr_fwhm_ns`], default [`DEFAULT_PSR_FWHM_NS`]).
    IkedaCarpenter {
        /// Also fit the PSR triangle FWHM: appends `θ4` (µs, box-bounded
        /// 0.05–1.0 µs, started at [`CalibrationConfig::psr_fwhm_ns`]
        /// clamped into that box). A positive starting width outside the box
        /// — legal as a pin up to [`PSR_FWHM_PIN_CEILING_US`] — starts at
        /// the nearer box edge with a stderr warning; a fit that stays there
        /// reports `psr_fwhm_us:lower` / `:upper` in
        /// [`CalibrationResult::bounds_hit`]. Off by default — the 350 ns
        /// SNS PSR width is machine metrology, not a per-experiment unknown.
        fit_psr: bool,
    },
}

impl ResolutionFamily {
    /// Number of free parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        match self {
            ResolutionFamily::Gaussian | ResolutionFamily::UdrCorr { .. } => 2,
            ResolutionFamily::IkedaCarpenter { fit_psr } => 4 + usize::from(*fit_psr),
        }
    }

    /// Names of the raw optimizer coordinates, in [`CalibrationResult::theta`]
    /// order. Used to label [`CalibrationResult::bounds_hit`]; the `ln_*`
    /// prefixes flag exp-encoded coordinates (decode via
    /// [`CalibrationResult::resolution`] rather than by hand).
    #[must_use]
    pub fn param_names(&self) -> Vec<&'static str> {
        match self {
            ResolutionFamily::Gaussian => vec!["delta_t_us", "delta_l_m"],
            ResolutionFamily::UdrCorr { .. } => vec!["log_s0", "p"],
            ResolutionFamily::IkedaCarpenter { fit_psr } => {
                let mut names = vec!["ln_a0", "ln_a1", "ln_beta", "r"];
                if *fit_psr {
                    names.push("psr_fwhm_us");
                }
                names
            }
        }
    }

    fn label(&self) -> &'static str {
        match self {
            ResolutionFamily::Gaussian => "gaussian",
            ResolutionFamily::UdrCorr { .. } => "udr_corr",
            ResolutionFamily::IkedaCarpenter { .. } => "ic",
        }
    }

    /// `(start vector, box bounds)` for the optimizer (mirrors the validated
    /// Python reference: `udr_corr` uses log-`s0`; bounds keep widths positive).
    /// `cfg` supplies the starting PSR FWHM when the IC family fits it.
    fn x0_bounds(&self, cfg: &CalibrationConfig) -> (Vec<f64>, Vec<(f64, f64)>) {
        match self {
            ResolutionFamily::Gaussian => (vec![2.0, 1e-3], vec![(1e-3, 50.0), (0.0, 0.5)]),
            ResolutionFamily::UdrCorr { .. } => {
                // (log s0, p): s0 = exp(log_s0) clamped to [0.2, 5].
                (
                    vec![0.0, 0.0],
                    vec![(UDR_S0_MIN.ln(), UDR_S0_MAX.ln()), (-4.0, 4.0)],
                )
            }
            ResolutionFamily::IkedaCarpenter { fit_psr } => {
                let mut x0 = vec![IC_A0_X0.ln(), IC_A1_X0.ln(), IC_BETA_X0.ln(), IC_R_X0];
                let mut bounds = vec![
                    (IC_A0_MIN.ln(), IC_A0_MAX.ln()),
                    (IC_A1_MIN.ln(), IC_A1_MAX.ln()),
                    (IC_BETA_MIN.ln(), IC_BETA_MAX.ln()),
                    (IC_R_MIN, IC_R_MAX),
                ];
                if *fit_psr {
                    // cfg.psr_fwhm_ns > 0 is guaranteed here (fit_psr with a
                    // zero width is rejected up front — "0 disables" cannot
                    // silently become a fitted 0.05 µs). A positive start
                    // outside the fit box — legal as a PIN up to
                    // PSR_FWHM_PIN_CEILING_US — is CLAMPED to the nearer box
                    // edge, not rejected (#645 round 4, F3), and the clamp is
                    // announced on stderr: a clamped start that never leaves
                    // its edge additionally surfaces as "psr_fwhm_us:lower" /
                    // ":upper" in `CalibrationResult::bounds_hit`.
                    let start_us = cfg.psr_fwhm_ns * NS_TO_US;
                    let clamped_us = start_us.clamp(PSR_FWHM_US_MIN, PSR_FWHM_US_MAX);
                    if clamped_us != start_us {
                        eprintln!(
                            "warning: fit_psr starting width psr_fwhm_ns = {} ns lies \
                             outside the PSR fit box [{PSR_FWHM_US_MIN}, {PSR_FWHM_US_MAX}] µs; \
                             starting the fit at the nearer box edge ({clamped_us} µs). A fit \
                             that stays there reports \"psr_fwhm_us:lower\" / \":upper\" in \
                             bounds_hit.",
                            cfg.psr_fwhm_ns
                        );
                    }
                    x0.push(clamped_us);
                    bounds.push((PSR_FWHM_US_MIN, PSR_FWHM_US_MAX));
                }
                (x0, bounds)
            }
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
    /// SNS PSR (accumulator-ring) channel-triangle FWHM in **ns**, folded into
    /// the **IC family only** (default [`DEFAULT_PSR_FWHM_NS`]; `0.0`
    /// disables the fold). Tabulated/UDR (FTS) kernels already carry the fold
    /// in the file itself (header: "folded triang FWHM 350 ns PSR") and are
    /// structurally never re-folded here — applying it twice would
    /// double-count the burst. When the family is
    /// `IkedaCarpenter { fit_psr: true }` this value is the fit's starting
    /// point instead of a pin, clamped into the 0.05–1 µs fit box: a width
    /// in (1, 10] µs is a legal pin but an out-of-box start — the fit then
    /// starts at the box top (announced by a stderr warning), and if it
    /// stays there it reports `psr_fwhm_us:upper` in
    /// [`CalibrationResult::bounds_hit`]. Nonzero widths above
    /// [`PSR_FWHM_PIN_CEILING_US`] (10 µs = 10 000 ns) are rejected as a
    /// ns↔µs unit slip — see that constant for the quadratic-cost rationale.
    pub psr_fwhm_ns: f64,
    /// Fit the SAMMY TOF-zero `t0` (µs) as a SHARED energy-scale parameter.
    /// **Default `false`** — position is pinned at
    /// [`position_t0_center_us`](Self::position_t0_center_us) so calibration is a
    /// pure shape/width fit (matching SAMMY, where `t0`/`L` are a separate
    /// energy-scale calibration). Opt in only *with* a metrology prior; see
    /// [`with_position_prior`](CalibrationConfig::with_position_prior).
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
            psr_fwhm_ns: DEFAULT_PSR_FWHM_NS,
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
    /// Fitted parameter vector (raw optimizer space; see [`ResolutionFamily`]
    /// and [`ResolutionFamily::param_names`]). For the IC family these are
    /// ln/box-encoded — read decoded physical values off
    /// [`resolution`](Self::resolution) instead of exponentiating by hand.
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
    /// Total outer-loop free parameters: resolution θ plus any FITTED position
    /// coordinates (`t0`, `L_scale`). Makes cross-family χ² comparisons and
    /// dof bookkeeping explicit now that families differ in size (IC is 4–5
    /// parameters, Gaussian/UdrCorr are 2).
    pub n_free_params: usize,
    /// Coordinates that finished within `BOUND_HIT_REL_TOL·(hi−lo)` of a box
    /// bound, as `"name:lower"` / `"name:upper"` (names from
    /// [`ResolutionFamily::param_names`], plus `"t0_us"` / `"l_scale"` when
    /// position is fitted). Empty = interior solution. A pinned bound makes a
    /// degenerate calibration visible instead of silent: e.g. an eV-regime
    /// calibrant with no storage tail drives `R → 0` (`"r:lower"`) — on that
    /// β↔R ridge the storage term vanishes and β is unconstrained, so the
    /// reported β must not be physically interpreted.
    pub bounds_hit: Vec<String>,
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
        ResolutionFamily::IkedaCarpenter { fit_psr } => {
            // θ = [ln a0, ln a1, ln β, R, (PSR FWHM µs iff fit_psr)] — see the
            // IC_* constants for the bounds and their physics. Decoding the
            // exp-encoded coordinates here (not in a new EnergyLaw variant)
            // keeps the kernel physics in nereids-physics untouched: the
            // optimizer space guarantees α(E) > 0 and β > 0 by construction.
            let psr_us = if *fit_psr {
                theta[4]
            } else {
                cfg.psr_fwhm_ns * NS_TO_US
            };
            let params = IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE {
                    a0: theta[0].exp(),
                    a1: theta[1].exp(),
                },
                beta: EnergyLaw::Const(theta[2].exp()),
                // Scalar R: a free κ in ExpMilliEv is unidentifiable in the eV
                // regime (R ≡ 0 across 1–200 eV for ANY plausible κ); a scalar
                // lets the calibrant decide whether a storage tail is present.
                r: EnergyLaw::Const(theta[3]),
                burst_sigma_us: None,
                // SNS PSR channel-triangle fold (0 disables). IC family only —
                // tabulated/UDR kernels already carry the fold in the file.
                channel_fwhm_us: (psr_us > 0.0).then_some(psr_us),
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
/// The IC family fits the full bounded moderator shape (#642): `α(E) =
/// e^{θ0}√E + e^{θ1}` — positive at every energy by construction — plus free
/// bounded `β = e^{θ2}` and scalar storage fraction `R = θ3 ∈ [0, 1]`, folded
/// with the SNS PSR channel triangle
/// ([`CalibrationConfig::psr_fwhm_ns`], default 350 ns; `0` disables;
/// optionally fitted via `IkedaCarpenter { fit_psr: true }` — a zero width
/// combined with `fit_psr` contradicts "0 disables" and is rejected).
///
/// Returns the fitted shape parameters, the reduced **data** χ²/dof, the fitted (or
/// pinned) `(t0, L_scale)`, the prior penalty, the calibrated
/// [`ResolutionFunction`] (ready to pin), the free-parameter count
/// ([`CalibrationResult::n_free_params`]), and the pinned-bound report
/// ([`CalibrationResult::bounds_hit`]).
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
    // PSR triangle width: finite and >= 0 (0.0 disables the fold). A NaN or
    // negative width would otherwise flow into IkedaCarpenter::new on every
    // evaluation and only surface as the generic "no finite-objective
    // resolution" error. Validated for every family (it is inert outside IC)
    // so a mis-set config is caught regardless of the family under test.
    if !config.psr_fwhm_ns.is_finite() || config.psr_fwhm_ns < 0.0 {
        return Err(FittingError::InvalidConfig(format!(
            "psr_fwhm_ns must be finite and >= 0 (0 disables the PSR fold), got {}",
            config.psr_fwhm_ns
        )));
    }
    // Sanity ceiling on the width itself (see PSR_FWHM_PIN_CEILING_US):
    // psr_fwhm_ns is NANOSECONDS and synthesis cost is quadratic in the fold
    // width, so a µs-as-ns unit slip pins a fictitious multi-hundred-µs fold
    // that hangs the calibration for hours. Reject loudly, up front, for
    // every family (inert outside IC, same rationale as the checks above).
    if config.psr_fwhm_ns * NS_TO_US > PSR_FWHM_PIN_CEILING_US {
        return Err(FittingError::InvalidConfig(format!(
            "psr_fwhm_ns = {} ns (= {} µs) exceeds the {PSR_FWHM_PIN_CEILING_US} µs sanity \
             ceiling (10x the {PSR_FWHM_US_MAX} µs fit bound). psr_fwhm_ns is in NANOSECONDS \
             — the SNS/VENUS FTS convention is 350 ns — and kernel-synthesis cost grows \
             quadratically with the fold width, so a µs-as-ns unit slip would hang the \
             calibration behind a fictitious fold. Pass the width in ns, or 0 to disable \
             the PSR fold",
            config.psr_fwhm_ns,
            config.psr_fwhm_ns * NS_TO_US
        )));
    }
    // fit_psr fits the PSR FWHM from the psr_fwhm_ns starting value, but 0 is
    // documented as "no fold": a zero start would be silently clamped into the
    // [PSR_FWHM_US_MIN, PSR_FWHM_US_MAX] fit box, contradicting the "0
    // disables" contract. Reject the contradiction loudly.
    if matches!(family, ResolutionFamily::IkedaCarpenter { fit_psr: true })
        && config.psr_fwhm_ns == 0.0
    {
        return Err(FittingError::InvalidConfig(
            "fit_psr requires a positive psr_fwhm_ns starting value (psr_fwhm_ns = 0 disables \
             the PSR fold; use fit_psr = false to calibrate without one)"
                .into(),
        ));
    }
    // IC synthesis-grid resolution: validate up front for the IC family (inert
    // for the others) so an out-of-range value gives this precise error instead
    // of every IkedaCarpenter::new evaluation failing into the generic late
    // "no finite-objective resolution" error. Thresholds mirror both
    // IkedaCarpenter::new (n_energies >= 2, n_tau >= 8) and the Python
    // binding's sibling validation, so the two public entry points reject the
    // same inputs.
    if matches!(family, ResolutionFamily::IkedaCarpenter { .. }) {
        if config.ic_n_energies < 2 {
            return Err(FittingError::InvalidConfig(format!(
                "ic_n_energies must be >= 2 for the IC family, got {}",
                config.ic_n_energies
            )));
        }
        if config.ic_n_tau < 8 {
            return Err(FittingError::InvalidConfig(format!(
                "ic_n_tau must be >= 8 for the IC family, got {}",
                config.ic_n_tau
            )));
        }
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
    let (mut x0, mut bounds) = family.x0_bounds(config);
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
    // PRE-FLIGHT the start (#645 round 3, F1): synthesize the resolution once
    // at x0 before any optimization. A start whose kernel cannot be
    // synthesized — e.g. any PSR triangle under ~58.6 ns: the default β/R
    // start (β = 0.1, R = 0.1) spans a 16/β = 160 µs storage tail, capping
    // the τ-step at 160/8191 ≈ 19.53 ns, above such a triangle's FWHM/3
    // resolution floor (note the PSR fit-box floor 0.05 µs = 50 ns is ITSELF
    // in this class, so a `>= PSR_FWHM_US_MIN` value check could not cover
    // it) — passes every value-level config check above yet makes EVERY
    // initial-simplex vertex infeasible (∞ objective): the Nelder–Mead
    // objective range is then ∞ − ∞ = NaN, so it can never self-converge,
    // burns max_iter, and used to die late with the generic "no
    // finite-objective" error blaming the forward model. Reject the START
    // precisely instead, surfacing the τ-geometry/synthesis diagnosis. A θ
    // that becomes infeasible only DURING the search remains an ∞ point the
    // simplex steps away from (see the objective below) — this pre-flight
    // rejects only an infeasible start.
    if let Err(synth_err) = build_resolution(&family, &x0, e_min, e_max, config) {
        let psr_note = if matches!(family, ResolutionFamily::IkedaCarpenter { .. }) {
            format!(
                " The starting PSR width comes from psr_fwhm_ns = {} ns — widen the \
                 triangle (the SNS/VENUS FTS convention is 350 ns), or pass 0 to \
                 disable the fold when not fitting it.",
                config.psr_fwhm_ns
            )
        } else {
            String::new()
        };
        return Err(FittingError::InvalidConfig(format!(
            "resolution kernel synthesis is infeasible at the starting parameter \
             vector, so every optimizer restart would begin from an all-infeasible \
             simplex: {synth_err}.{psr_note}"
        )));
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

    // For tabulated UDR and IC kernels, the working grid is the data grid.
    // When the energy scale is pinned, the Doppler-broadened Beer-Lambert
    // transmission is therefore identical at every optimizer evaluation;
    // only the resolution kernel changes.  Compute that fixed nuclear result
    // once and reuse it.  Gaussian resolution is deliberately excluded because
    // its auxiliary working grid depends on the trial width.  A fitted t0 or
    // L_scale is also excluded because it changes the nuclear evaluation grid.
    let fixed_unresolved = if !config.fit_t0
        && !config.fit_l_scale
        && !matches!(&family, ResolutionFamily::Gaussian)
    {
        let grid = corrected_energy_grid(
            energies,
            config.position_t0_center_us,
            config.position_l_scale_center,
            config.flight_path_m,
        )?;
        let transmission = forward_model(&grid, sample, None)
            .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))?;
        Some((grid, transmission))
    } else {
        None
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
        let mut obj = |theta: &[f64]| -> Result<f64, FittingError> {
            // theta = [resolution params (n_res)..., t0?, L_scale?]. The resolution
            // kernel uses only the first n_res; (t0, L_scale) set the energy scale.
            // An UNRESOLVABLE θ — nereids-physics rejects a kernel whose τ-grid
            // cannot resolve the requested fold / prompt core within the
            // MAX_TAU_SAMPLES cap (e.g. a fitted PSR at its 0.05 µs floor
            // against β at its own floor) — is an infeasible POINT of the
            // search, not a broken calibration: step away (mirrors the
            // corrected-TOF ≤ 0 guard below). Config-level failures cannot
            // reach here: they are rejected up front by calibrate_resolution.
            let Ok(res) = build_resolution(&family, theta, e_min, e_max, config) else {
                return Ok(f64::INFINITY);
            };
            let inst = InstrumentParams { resolution: res };
            let (t0, l_scale) = unpack_position(theta);
            let model = if let Some((grid, unresolved)) = &fixed_unresolved {
                apply_resolution(grid, unresolved, &inst.resolution)
                    .map_err(|e| FittingError::EvaluationFailed(format!("resolution: {e:?}")))?
            } else {
                // Infeasible energy scale (corrected TOF ≤ 0) → step away.
                let Ok(grid) = corrected_energy_grid(energies, t0, l_scale, config.flight_path_m)
                else {
                    return Ok(f64::INFINITY);
                };
                forward_model(&grid, sample, Some(&inst))
                    .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))?
            };
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
        let mut res = nelder_mead_minimize(&mut obj, &start, Some(&bounds), &nm)?;
        // Simplex RE-INFLATION: Nelder–Mead's known failure mode is premature
        // simplex collapse — the spread criteria are met (`self_converged`)
        // at a point that is NOT the basin minimum. Observed on the
        // 4-parameter IC family: a 300 K synthetic calibrant stalled at
        // Δχ² ≈ +130 above the noise floor in the curved α↔β↔R valley, and
        // the ~1.5 % kernel-width error re-expressed as a ~23 K temperature
        // bias in the downstream pinned fit. Standard cure: restart a FRESH,
        // *larger* simplex at the incumbent (same 5 % edge re-collapses to
        // the same trap deterministically) and keep the improvement, until
        // it stops helping (bounded by MAX_SIMPLEX_REINFLATIONS). A
        // re-inflation from a true minimum re-contracts quickly, so the
        // extra cost there is small.
        let reinflate_nm = NelderMeadConfig {
            initial_step_frac: REINFLATE_STEP_FRAC,
            initial_step_abs: REINFLATE_STEP_ABS,
            ..nm.clone()
        };
        for _ in 0..MAX_SIMPLEX_REINFLATIONS {
            let again = nelder_mead_minimize(&mut obj, &res.x, Some(&bounds), &reinflate_nm)?;
            let improved = again.fun + nm.fatol < res.fun;
            res.iterations += again.iterations;
            res.n_evals += again.n_evals;
            if improved {
                res.x = again.x;
                res.fun = again.fun;
                res.self_converged = again.self_converged;
            } else {
                break;
            }
        }
        if best.as_ref().is_none_or(|b| res.fun < b.fun) {
            best = Some(res);
        }
    }
    let best = best.expect("at least one restart runs");
    if !best.fun.is_finite() {
        // Every ∞ source of the objective (#645 round 3 F1, round 4 F1):
        // `nelder_mead_minimize` maps every objective `Err` — forward-model
        // failures included — to an infeasible +∞ point rather than aborting
        // (see the `eval` closure in `nelder_mead.rs`), so no failure class
        // raises its own error during the search. Reaching here means every
        // vector tried hit one of them — kernel synthesis rejected (τ-grid
        // cap vs fold/prompt geometry), an invalid energy scale (corrected
        // TOF ≤ 0), a singular anorm/baseline system, or a forward-model
        // (transmission) failure. The start itself synthesized (pre-flighted
        // above), so the infeasibility arose during the search.
        return Err(FittingError::EvaluationFailed(
            "calibration found no finite-objective resolution: every parameter vector \
             tried was infeasible — kernel synthesis rejected it (τ-grid cap vs \
             fold/prompt geometry), the energy scale was invalid (corrected TOF ≤ 0), \
             the anorm/baseline system was singular, or the forward model failed"
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
    let model = if let Some((fixed_grid, unresolved)) = &fixed_unresolved {
        apply_resolution(fixed_grid, unresolved, &inst.resolution)
            .map_err(|e| FittingError::EvaluationFailed(format!("resolution: {e:?}")))?
    } else {
        forward_model(&grid, sample, Some(&inst))
            .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))?
    };
    let (ssr, k) = inner_ssr(data, unc, &model, config.fit_background).ok_or_else(|| {
        FittingError::EvaluationFailed("singular anorm/baseline at the solution".into())
    })?;
    let dof = data.len().saturating_sub(k + n_res + n_pos).max(1) as f64;
    let chi2_dof = ssr / dof;
    let theta = best.x[..n_res].to_vec();
    // Bound-pinning report: label every optimizer coordinate that finished
    // within BOUND_HIT_REL_TOL·(hi−lo) of its box bound. This is the
    // degeneracy flag for the wider IC family (e.g. "r:lower" ⇒ the β↔R ridge:
    // no storage tail in the data, β unconstrained).
    let mut coord_names = family.param_names();
    if config.fit_t0 {
        coord_names.push("t0_us");
    }
    if config.fit_l_scale {
        coord_names.push("l_scale");
    }
    let bounds_hit: Vec<String> = best
        .x
        .iter()
        .zip(&bounds)
        .zip(&coord_names)
        .flat_map(|((&v, &(lo, hi)), name)| {
            let tol = BOUND_HIT_REL_TOL * (hi - lo);
            let mut hits = Vec::new();
            if v - lo <= tol {
                hits.push(format!("{name}:lower"));
            }
            if hi - v <= tol {
                hits.push(format!("{name}:upper"));
            }
            hits
        })
        .collect();
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
        n_free_params: n_res + n_pos,
        bounds_hit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::test_support::synthetic_isotope;

    /// Decode the calibrated IC parameters `(a0, a1, β, R, psr_fwhm_us)` off
    /// the result's resolution — the single source of truth (raw `theta` is
    /// ln/box-encoded).
    fn decoded_ic(r: &CalibrationResult) -> (f64, f64, f64, f64, f64) {
        let ResolutionFunction::IkedaCarpenter(ic) = &r.resolution else {
            panic!("expected an IC resolution for family {}", r.family);
        };
        let p = ic.params();
        let EnergyLaw::SqrtE { a0, a1 } = p.alpha else {
            panic!("expected a SqrtE alpha law");
        };
        let EnergyLaw::Const(rr) = p.r else {
            panic!("expected a Const R law");
        };
        let EnergyLaw::Const(beta) = p.beta else {
            panic!("expected a Const beta law");
        };
        (a0, a1, beta, rr, p.channel_fwhm_us.unwrap_or(0.0))
    }

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

    /// #645 round 4, F3: a `fit_psr` starting width outside the 0.05–1 µs fit
    /// box (legal as a PIN up to [`PSR_FWHM_PIN_CEILING_US`]) is clamped to
    /// the nearer box edge — documented behavior, not an error.
    #[test]
    fn fit_psr_out_of_box_start_clamps_to_box_edge() {
        let family = ResolutionFamily::IkedaCarpenter { fit_psr: true };
        // 5 000 ns = 5 µs: a valid pin width, above the 1 µs fit-box top.
        let above = CalibrationConfig {
            psr_fwhm_ns: 5_000.0,
            ..CalibrationConfig::default()
        };
        let (x0, bounds) = family.x0_bounds(&above);
        assert_eq!(x0.len(), 5);
        assert_eq!(x0[4], PSR_FWHM_US_MAX);
        assert_eq!(bounds[4], (PSR_FWHM_US_MIN, PSR_FWHM_US_MAX));
        // 10 ns: below the 50 ns identifiability floor — clamped UP.
        let below = CalibrationConfig {
            psr_fwhm_ns: 10.0,
            ..CalibrationConfig::default()
        };
        let (x0, _) = family.x0_bounds(&below);
        assert_eq!(x0[4], PSR_FWHM_US_MIN);
        // The in-box default (350 ns) passes through unclamped.
        let inside = CalibrationConfig::default();
        let (x0, _) = family.x0_bounds(&inside);
        assert_eq!(x0[4], DEFAULT_PSR_FWHM_NS * NS_TO_US);
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
                beta: EnergyLaw::Const(0.1),
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
        // Tight prior: σ must be small enough that the quadratic prior
        // curvature rivals the data-χ² curvature in t0, or the optimum
        // sits at the displacement and the pull is invisible. The
        // width-correct kernel interpolation sharpened the data term
        // (narrower between-reference kernels carry more positional
        // information than the over-wide chord blend used to), so the
        // binding regime needs a tighter σ than it once did.
        let tight =
            calibrate_resolution(mkbase(), &energies, &data, &unc, &sample, &mk(0.005)).unwrap();
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
                beta: EnergyLaw::Const(0.1),
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
        // The truth kernel carries NO PSR fold, so disable the calibrator's
        // default 350 ns fold — otherwise the IC family could not close.
        let cfg = CalibrationConfig {
            restarts: 3,
            psr_fwhm_ns: 0.0,
            ..Default::default()
        };
        let chi2 = |fam| {
            calibrate_resolution(fam, &energies, &data, &unc, &sample, &cfg)
                .unwrap()
                .chi2_dof
        };
        let ic_chi2 = chi2(ResolutionFamily::IkedaCarpenter { fit_psr: false });
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
        for (fam, n_expected) in [
            (ResolutionFamily::Gaussian, 2),
            (ResolutionFamily::IkedaCarpenter { fit_psr: false }, 4),
        ] {
            let label = fam.label().to_string();
            let r = calibrate_resolution(fam, &energies, &data, &unc, &sample, &cfg).unwrap();
            assert!(r.chi2_dof.is_finite(), "{label} χ² not finite");
            assert_eq!(
                r.theta.len(),
                n_expected,
                "{label} should fit {n_expected} params"
            );
            assert_eq!(r.n_free_params, n_expected, "{label} n_free_params");
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
        // IC fits θ = [ln a0, ln a1, ln β, R] (+ PSR FWHM iff fit_psr).
        assert_eq!(
            ResolutionFamily::IkedaCarpenter { fit_psr: false }.n_params(),
            4
        );
        assert_eq!(
            ResolutionFamily::IkedaCarpenter { fit_psr: true }.n_params(),
            5
        );
        // param_names track n_params, coordinate for coordinate.
        for fam in [
            ResolutionFamily::Gaussian,
            ResolutionFamily::IkedaCarpenter { fit_psr: false },
            ResolutionFamily::IkedaCarpenter { fit_psr: true },
        ] {
            assert_eq!(fam.param_names().len(), fam.n_params());
        }
        assert_eq!(
            ResolutionFamily::IkedaCarpenter { fit_psr: true }
                .param_names()
                .last()
                .copied(),
            Some("psr_fwhm_us")
        );
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
        // both use the IC synthesis, so this checks the optimizer recovers the
        // full bounded 4-parameter family (#642) — the IC pulse physics is
        // independently covered by the ic_pulse tests in nereids-physics.
        // Truth, all interior to the new boxes: a0=0.35, a1=0.05, β=0.1, R=0.1,
        // PSR triangle 0.35 µs (= the calibrator's default 350 ns pin).
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..400).map(|i| 12.0 + i as f64 * 0.04).collect();
        let cfg = CalibrationConfig {
            restarts: 2,
            ..Default::default()
        };
        // Truth kernel on the SAME derived grid the calibrator synthesizes on,
        // so the loop closes exactly (a grid mismatch would leak into recovery).
        let ic_truth = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                beta: EnergyLaw::Const(0.1),
                r: EnergyLaw::Const(0.1),
                burst_sigma_us: None,
                channel_fwhm_us: Some(0.35),
            },
            cfg.flight_path_m,
            &SynthesisGrid {
                e_min_ev: (energies[0] * 0.5).max(1e-3),
                e_max_ev: energies.last().unwrap() * 2.0,
                n_energies: cfg.ic_n_energies,
                n_tau: cfg.ic_n_tau,
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
        let r = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter { fit_psr: false },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let (a0, a1, beta, rr, psr) = decoded_ic(&r);
        assert!((a0 - 0.35).abs() < 0.05, "recovered a0={a0}, expected 0.35");
        assert!(a1 > 0.0, "a1 positive by construction, got {a1}");
        // β and R shape the kernel only jointly through the storage tail
        // (weight R, decay 1/β), so their windows are deliberately loose.
        assert!(
            (beta - 0.1).abs() < 0.08,
            "recovered β={beta}, expected 0.1"
        );
        assert!((rr - 0.1).abs() < 0.08, "recovered R={rr}, expected 0.1");
        assert!(
            (psr - 0.35).abs() < 1e-12,
            "PSR pin {psr} µs != 0.35 µs (config default)"
        );
        assert_eq!(r.n_free_params, 4);
        assert!(
            r.bounds_hit.is_empty(),
            "interior truth must not pin bounds, got {:?}",
            r.bounds_hit
        );
        assert!(r.chi2_dof < 1.0, "matched χ²/dof={} too high", r.chi2_dof);
    }

    #[test]
    fn ic_recovers_known_psr_when_fit() {
        // Loop-closure / optimizer test for fit_psr (#645 F2, same caveat as
        // ic_recovers_known_alpha: truth and fit share the IC synthesis, so
        // this checks the 5-parameter optimizer, not the pulse physics).
        // Truth PSR FWHM = 0.6 µs — interior to the [0.05, 1] µs box and far
        // from the 0.35 µs default start — with the rest of the truth kernel
        // identical to ic_recovers_known_alpha. Two resonances (15 + 45 eV)
        // give the E-leverage that separates the E-independent triangle
        // width from the α(E) = a0·√E + a1 prompt law (a single resonance
        // probes the kernel at essentially one energy).
        // Full-density run (420 pts, 64×500 grid, restarts 2) recovers
        // psr = 0.5984 µs at χ²/dof ≈ 1e-6; this slimmed grid keeps the same
        // loop-closure semantics at a debug-friendly runtime.
        let iso_lo = synthetic_isotope(72, 178, 15.0, 0.05, 0.06);
        let iso_hi = synthetic_isotope(72, 179, 45.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso_lo, 2.0e-3), (iso_hi, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..280).map(|i| 8.0 + i as f64 * 0.15).collect();
        let cfg = CalibrationConfig {
            restarts: 1,
            ic_n_energies: 32,
            ic_n_tau: 320,
            ..Default::default()
        };
        let psr_true = 0.6;
        let ic_truth = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                beta: EnergyLaw::Const(0.1),
                r: EnergyLaw::Const(0.1),
                burst_sigma_us: None,
                channel_fwhm_us: Some(psr_true),
            },
            cfg.flight_path_m,
            &SynthesisGrid {
                e_min_ev: (energies[0] * 0.5).max(1e-3),
                e_max_ev: energies.last().unwrap() * 2.0,
                n_energies: cfg.ic_n_energies,
                n_tau: cfg.ic_n_tau,
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
        let r = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter { fit_psr: true },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let (a0, _a1, _beta, _rr, psr) = decoded_ic(&r);
        assert_eq!(r.n_free_params, 5);
        assert!(
            (psr - psr_true).abs() < 0.1,
            "recovered PSR FWHM {psr} µs, expected {psr_true} µs"
        );
        assert!((a0 - 0.35).abs() < 0.05, "recovered a0={a0}, expected 0.35");
        assert!(r.chi2_dof < 1.0, "matched χ²/dof={} too high", r.chi2_dof);
    }

    #[test]
    fn psr_disabled_at_zero_width() {
        // psr_fwhm_ns = 0.0 disables the triangle fold entirely: an UNFOLDED
        // truth is reproduced and the calibrated kernel carries no channel.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..250).map(|i| 14.0 + i as f64 * 0.05).collect();
        let cfg = CalibrationConfig {
            psr_fwhm_ns: 0.0,
            ic_n_energies: 32,
            ic_n_tau: 300,
            ..Default::default()
        };
        let ic_truth = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                beta: EnergyLaw::Const(0.1),
                r: EnergyLaw::Const(0.1),
                burst_sigma_us: None,
                channel_fwhm_us: None, // unfolded truth
            },
            cfg.flight_path_m,
            &SynthesisGrid {
                e_min_ev: (energies[0] * 0.5).max(1e-3),
                e_max_ev: energies.last().unwrap() * 2.0,
                n_energies: cfg.ic_n_energies,
                n_tau: cfg.ic_n_tau,
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
        let r = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter { fit_psr: false },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        let ResolutionFunction::IkedaCarpenter(ic) = &r.resolution else {
            panic!("expected an IC resolution");
        };
        assert!(
            ic.params().channel_fwhm_us.is_none(),
            "psr_fwhm_ns = 0 must leave channel_fwhm_us = None, got {:?}",
            ic.params().channel_fwhm_us
        );
        assert!(
            r.chi2_dof < 1.0,
            "unfolded self-fit χ²/dof={} too high",
            r.chi2_dof
        );
    }

    #[test]
    fn bounds_hit_reports_pinned_parameter() {
        // A truth WITHOUT a storage tail (R = 0) drives the fitted R onto its
        // lower box bound; the result must say so ("r:lower") — the β↔R-ridge
        // degeneracy flag (with no tail, β is unconstrained).
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let energies: Vec<f64> = (0..250).map(|i| 14.0 + i as f64 * 0.05).collect();
        let cfg = CalibrationConfig {
            ic_n_energies: 32,
            ic_n_tau: 300,
            restarts: 2,
            ..Default::default()
        };
        let ic_truth = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                beta: EnergyLaw::Const(0.1), // irrelevant at R = 0 (no storage term)
                r: EnergyLaw::Const(0.0),
                burst_sigma_us: None,
                channel_fwhm_us: Some(0.35),
            },
            cfg.flight_path_m,
            &SynthesisGrid {
                e_min_ev: (energies[0] * 0.5).max(1e-3),
                e_max_ev: energies.last().unwrap() * 2.0,
                n_energies: cfg.ic_n_energies,
                n_tau: cfg.ic_n_tau,
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
        let r = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter { fit_psr: false },
            &energies,
            &data,
            &unc,
            &sample,
            &cfg,
        )
        .unwrap();
        assert!(
            r.bounds_hit.iter().any(|s| s == "r:lower"),
            "R = 0 truth must pin the storage fraction: bounds_hit = {:?}, decoded = {:?}",
            r.bounds_hit,
            decoded_ic(&r)
        );
    }

    #[test]
    fn rejects_invalid_psr_fwhm_ns() {
        // NaN / negative / infinite PSR widths are config errors caught up
        // front (NaN would silently disable the `> 0.0` fold gate; a negative
        // width would fail deep in IkedaCarpenter::new on every evaluation).
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let e: Vec<f64> = (0..60).map(|i| 15.0 + i as f64 * 0.2).collect();
        let d = vec![0.9; 60];
        let u = vec![0.01; 60];
        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            let cfg = CalibrationConfig {
                psr_fwhm_ns: bad,
                ..Default::default()
            };
            assert!(
                matches!(
                    calibrate_resolution(
                        ResolutionFamily::IkedaCarpenter { fit_psr: false },
                        &e,
                        &d,
                        &u,
                        &sample,
                        &cfg
                    ),
                    Err(FittingError::InvalidConfig(_))
                ),
                "psr_fwhm_ns={bad} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_absurd_pinned_psr_width() {
        // Review #645 round 2, F1: psr_fwhm_ns is NANOSECONDS (FTS convention
        // 350 ns) and synthesis cost is quadratic in the fold width — a
        // µs-as-ns unit slip (350 meaning µs → 350_000 ns) previously passed
        // the finite/sign check and pinned a fictitious 350 µs fold: a
        // multi-hour silent hang. Widths above PSR_FWHM_PIN_CEILING_US
        // (10 µs = 10_000 ns) must be a loud up-front config error.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let e: Vec<f64> = (0..60).map(|i| 15.0 + i as f64 * 0.2).collect();
        let d = vec![0.9; 60];
        let u = vec![0.01; 60];
        let cfg = CalibrationConfig {
            psr_fwhm_ns: 350_000.0, // "350 µs" unit slip
            ..Default::default()
        };
        let err = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter { fit_psr: false },
            &e,
            &d,
            &u,
            &sample,
            &cfg,
        )
        .expect_err("a 350_000 ns (350 µs) pinned PSR width must be rejected");
        assert!(
            matches!(
                &err,
                FittingError::InvalidConfig(msg)
                    if msg.contains("NANOSECONDS") && msg.contains("350 ns")
            ),
            "ceiling error must name the ns unit and the 350-ns convention, got {err:?}"
        );

        // Boundary + normal pins stay valid: exactly 10_000 ns sits ON the
        // ceiling (rejection is strict `>`; 10_000·1e-3 rounds to exactly
        // 10.0) and 350 ns is the FTS default. psr_fwhm_ns = 0 (disable) is
        // pinned valid by rejects_fit_psr_with_zero_psr_width. Tiny
        // grid/iteration budget: these arms assert config validity, not fit
        // quality.
        for ok_ns in [350.0, 10_000.0] {
            let cheap = CalibrationConfig {
                psr_fwhm_ns: ok_ns,
                ic_n_energies: 8,
                ic_n_tau: 32,
                max_iter: 10,
                ..Default::default()
            };
            assert!(
                calibrate_resolution(
                    ResolutionFamily::IkedaCarpenter { fit_psr: false },
                    &e,
                    &d,
                    &u,
                    &sample,
                    &cheap,
                )
                .is_ok(),
                "psr_fwhm_ns = {ok_ns} ns must remain a valid pinned width"
            );
        }
    }

    #[test]
    fn rejects_infeasible_psr_start_width() {
        // Review #645 round 3, F1: a nonzero PSR width in (0, ~58.6 ns)
        // passes every value-level check (finite / sign / ceiling) yet cannot
        // be SYNTHESIZED at the optimizer start: the default β/R start
        // (β = 0.1, R = 0.1 > R_NEGLIGIBLE) spans a 16/β = 160 µs storage
        // tail, capping the τ-step at 160/8191 ≈ 19.53 ns, and tau_geometry
        // rejects any triangle whose FWHM/3 floor is below that (fwhm <
        // ~58.6 ns). Every initial-simplex vertex was then ∞ (objective range
        // ∞ − ∞ = NaN — no self-convergence), so the calibration burned
        // max_iter and died with the generic "no finite-objective" error
        // blaming the forward model. The pre-flight must reject the START
        // precisely, surfacing the τ-geometry diagnosis and naming
        // psr_fwhm_ns. The fit_psr arm starts AT the fit-box floor
        // PSR_FWHM_US_MIN = 0.05 µs (50 ns), which is itself infeasible at
        // the default start — proof that a `>= PSR_FWHM_US_MIN` value check
        // would not be sufficient.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let e: Vec<f64> = (0..60).map(|i| 15.0 + i as f64 * 0.2).collect();
        let d = vec![0.9; 60];
        let u = vec![0.01; 60];
        for (fit_psr, psr_ns) in [(false, 55.0), (true, 50.0)] {
            let cfg = CalibrationConfig {
                psr_fwhm_ns: psr_ns,
                ..Default::default()
            };
            let err = calibrate_resolution(
                ResolutionFamily::IkedaCarpenter { fit_psr },
                &e,
                &d,
                &u,
                &sample,
                &cfg,
            )
            .expect_err("a sub-59-ns PSR start must be rejected up front");
            assert!(
                matches!(
                    &err,
                    FittingError::InvalidConfig(msg)
                        if msg.contains("starting parameter vector")
                            && msg.contains("psr_fwhm_ns")
                            && msg.contains("cannot resolve")
                ),
                "pre-flight error must name the start, psr_fwhm_ns and the τ-cap cause \
                 (fit_psr = {fit_psr}, psr_ns = {psr_ns}), got {err:?}"
            );
        }
    }

    #[test]
    fn rejects_fit_psr_with_zero_psr_width() {
        // psr_fwhm_ns = 0 means "no PSR fold"; fit_psr = true would silently
        // clamp that 0 start into the [0.05, 1] µs fit box, contradicting the
        // documented "0 disables". The contradiction is a config error.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let e: Vec<f64> = (0..60).map(|i| 15.0 + i as f64 * 0.2).collect();
        let d = vec![0.9; 60];
        let u = vec![0.01; 60];
        let cfg = CalibrationConfig {
            psr_fwhm_ns: 0.0,
            ..Default::default()
        };
        let err = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter { fit_psr: true },
            &e,
            &d,
            &u,
            &sample,
            &cfg,
        )
        .expect_err("fit_psr with psr_fwhm_ns = 0 must be rejected");
        assert!(
            matches!(&err, FittingError::InvalidConfig(msg) if msg.contains("fit_psr")),
            "expected an InvalidConfig naming fit_psr, got {err:?}"
        );
        // The same zero width WITHOUT fit_psr stays valid ("0 disables").
        // Tiny grid/iteration budget: this arm only asserts the config
        // passes validation, not fit quality.
        let cheap = CalibrationConfig {
            psr_fwhm_ns: 0.0,
            ic_n_energies: 8,
            ic_n_tau: 32,
            max_iter: 10,
            ..Default::default()
        };
        assert!(
            calibrate_resolution(
                ResolutionFamily::IkedaCarpenter { fit_psr: false },
                &e,
                &d,
                &u,
                &sample,
                &cheap,
            )
            .is_ok(),
            "psr_fwhm_ns = 0 with fit_psr = false must remain a valid config"
        );
    }

    #[test]
    fn rejects_undersized_ic_synthesis_grid() {
        // ic_n_energies < 2 / ic_n_tau < 8 previously surfaced only as the
        // late, generic "no finite-objective resolution" error (every
        // IkedaCarpenter::new evaluation failed). They must be precise
        // up-front InvalidConfig errors for the IC family — sibling parity
        // with the Python binding's validation.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let e: Vec<f64> = (0..60).map(|i| 15.0 + i as f64 * 0.2).collect();
        let d = vec![0.9; 60];
        let u = vec![0.01; 60];
        for (ne, nt, what) in [(1, 500, "ic_n_energies"), (64, 7, "ic_n_tau")] {
            // The loose iteration/tolerance budget only cheapens the Gaussian
            // is_ok arm below (validation-only assertion, not fit quality);
            // the InvalidConfig arm rejects before any optimization runs.
            let cfg = CalibrationConfig {
                ic_n_energies: ne,
                ic_n_tau: nt,
                max_iter: 5,
                xatol: 1.0,
                fatol: 1.0,
                ..Default::default()
            };
            let err = calibrate_resolution(
                ResolutionFamily::IkedaCarpenter { fit_psr: false },
                &e,
                &d,
                &u,
                &sample,
                &cfg,
            )
            .expect_err("undersized IC synthesis grid must be rejected");
            assert!(
                matches!(&err, FittingError::InvalidConfig(msg) if msg.contains(what)),
                "expected an InvalidConfig naming {what}, got {err:?}"
            );
            // The same values are inert for a non-IC family (mirrors the
            // Python binding: the knobs only size the IC synthesis grid).
            assert!(
                calibrate_resolution(ResolutionFamily::Gaussian, &e, &d, &u, &sample, &cfg).is_ok(),
                "ic grid knobs must stay inert for the Gaussian family"
            );
        }
    }

    #[test]
    fn ic_box_worst_corner_synthesizes_within_tau_cap() {
        // #645 F1: the calibrator's box must not abort a calibration on an
        // unresolvable τ-grid. Worst corner of the box in the τ-cap sense:
        // β at its floor (slow reach 16/β = 800 µs — the longest admitted
        // storage tail, so the capped step is at its widest ≈ 0.098 µs),
        // R = 1 (storage fully active), a1 at its ceiling and a0 at the
        // documented physical ceiling (α ≈ 1–3 µs⁻¹ in the eV regime ⇒
        // a0 ≈ 0.2–0.5, see IC_A0_MIN), at the calibrator's default
        // n_tau = 500 on a representative eV-regime synthesis window. The
        // capped step resolves both the PSR triangle box (0.35 µs default
        // pin up to the 1 µs fitted ceiling: ≥ 3 samples per side) and any
        // prompt core with α ≤ 18/(7 · 0.098) ≈ 26 µs⁻¹ — far above eV-regime
        // moderator physics. (The remaining unresolvable pockets — a fitted
        // PSR near its 0.05 µs floor together with β near its floor, or
        // a0 driven ~50× past the physical ceiling — are handled as
        // infeasible points, see the companion test below.)
        let cfg = CalibrationConfig::default();
        for fwhm_us in [DEFAULT_PSR_FWHM_NS * NS_TO_US, PSR_FWHM_US_MAX] {
            let corner = IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE {
                    a0: 0.5,
                    a1: IC_A1_MAX,
                },
                beta: EnergyLaw::Const(IC_BETA_MIN),
                r: EnergyLaw::Const(IC_R_MAX),
                burst_sigma_us: None,
                channel_fwhm_us: Some(fwhm_us),
            };
            let grid = SynthesisGrid {
                e_min_ev: 6.0,
                e_max_ev: 112.0,
                n_energies: cfg.ic_n_energies,
                n_tau: cfg.ic_n_tau,
            };
            assert!(
                IkedaCarpenter::new(corner, cfg.flight_path_m, &grid).is_ok(),
                "calibration-box worst corner must synthesize (fwhm = {fwhm_us} µs)"
            );
        }
    }

    #[test]
    fn ic_unresolvable_theta_errs_in_build_resolution() {
        // A θ inside the box can still be unresolvable: a fitted PSR at its
        // 0.05 µs floor against β at its own floor needs a τ-step ≤ FWHM/3 ≈
        // 0.017 µs across an 800 µs storage tail — past the 8192-sample cap.
        // This test asserts the build_resolution half only: such θ must Err.
        // The calibration-level half — the objective maps that Err to an ∞
        // point the simplex steps away from, never aborting the calibration —
        // is asserted by ic_infeasible_pocket_inside_box_completes_calibration
        // below (#645 round 3, F4).
        let cfg = CalibrationConfig::default();
        let theta = [
            IC_A0_X0.ln(),
            IC_A1_X0.ln(),
            IC_BETA_MIN.ln(),
            0.5,
            PSR_FWHM_US_MIN,
        ];
        let fam = ResolutionFamily::IkedaCarpenter { fit_psr: true };
        assert!(
            build_resolution(&fam, &theta, 6.0, 112.0, &cfg).is_err(),
            "β at its floor + PSR at its floor must be unresolvable"
        );
    }

    #[test]
    fn ic_infeasible_pocket_inside_box_completes_calibration() {
        // Review #645 round 3, F4 — the calibration-level half of the claim
        // above: with fit_psr the box CONTAINS the unresolvable pocket (PSR
        // near its 0.05 µs floor against β near its own floor), and the
        // simplex demonstrably brushes it — the 60 ns start sits just above
        // the ~58.6 ns feasibility edge at the default β/R start (the
        // pre-flight passes: 60/3 = 20 ns floor > 19.53 ns capped step), so
        // the FIRST simplex already carries an ∞ vertex: the β-decreased
        // vertex (ln β step is negative, β 0.1 → ~0.089) widens the storage
        // reach to ~180 µs and the capped step to ~21.9 ns, past the 20 ns
        // floor. The optimizer must treat such vertices as infeasible points
        // and finish: Ok, finite χ², decoded resolution inside the box. Tiny
        // grid/iteration budget — this asserts non-abortion, not fit quality.
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).unwrap();
        let e: Vec<f64> = (0..60).map(|i| 15.0 + i as f64 * 0.2).collect();
        let d = vec![0.9; 60];
        let u = vec![0.01; 60];
        let cfg = CalibrationConfig {
            psr_fwhm_ns: 60.0,
            ic_n_energies: 8,
            ic_n_tau: 32,
            max_iter: 60,
            ..Default::default()
        };
        let r = calibrate_resolution(
            ResolutionFamily::IkedaCarpenter { fit_psr: true },
            &e,
            &d,
            &u,
            &sample,
            &cfg,
        )
        .expect("an infeasible pocket inside the box must not abort the calibration");
        assert!(
            r.chi2_dof.is_finite(),
            "calibration through the infeasible pocket must return a finite χ²/dof, got {}",
            r.chi2_dof
        );
        let (_a0, _a1, beta, _r, psr_us) = decoded_ic(&r);
        assert!(
            (PSR_FWHM_US_MIN..=PSR_FWHM_US_MAX).contains(&psr_us)
                && (IC_BETA_MIN..=IC_BETA_MAX).contains(&beta),
            "decoded solution must be feasible and inside the box: β = {beta}, \
             psr = {psr_us} µs"
        );
    }
}
