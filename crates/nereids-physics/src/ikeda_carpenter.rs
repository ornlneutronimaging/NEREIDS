//! Ikeda–Carpenter analytical moderator resolution model.
//!
//! A third instrument-resolution model alongside the analytical Gaussian
//! ([`crate::resolution::ResolutionParams`]) and the Monte-Carlo tabulated
//! kernel ([`crate::resolution::TabulatedResolution`]). It exists to settle a
//! methodological dispute about the VENUS instrument resolution: one camp
//! trusts the MC-simulated tabulated kernel (UDR/FTS file); the instrument
//! scientist distrusts the unproven MC and prefers an analytical
//! Ikeda–Carpenter moderator model. NEREIDS implements IC as a first-class
//! model so all three can be cross-validated against synthetic loop-closure
//! and real VENUS data.
//!
//! # Physics — the Ikeda–Carpenter pulse
//!
//! Reference: S. Ikeda & J. M. Carpenter, *Nucl. Instrum. Methods* **A239**
//! (1985) 536–544. The neutron emission-time distribution from a pulsed
//! spallation moderator is
//!
//! ```text
//!   I(τ) = (1−R)·g₃(τ;α)  +  R·[g₃(·;α) ⊛ β e^{−β·}](τ),    τ ≥ 0
//! ```
//!
//! where the prompt (slowing-down) term is a Gamma/Erlang density of shape 3:
//!
//! ```text
//!   g₃(τ;α) = α³ τ² e^{−ατ} / 2          (mode 2/α, mean 3/α, ∫ = 1)
//! ```
//!
//! and the delayed (storage) term is `g₃` convolved with an exponential of
//! rate β. The convolution has the closed form
//!
//! ```text
//!   g₃(·;α) ⊛ β e^{−β·} (τ) = β (α/γ)³ [ e^{−βτ} − e^{−ατ}(1 + γτ + ½γ²τ²) ],  γ = α−β
//! ```
//!
//! (re-derived and confirmed against the Codex independent derivation and the
//! SAMMY-RPI χ²+double-exp moderator form). Both terms are individually
//! unit-area, so `I` is unit-area for any α,β>0 and 0≤R≤1, with first moment
//! `⟨τ⟩ = 3/α + R/β`. The asymmetry — sharp rise, long tail toward larger τ
//! (later TOF, lower apparent energy) — is the physical origin of the
//! asymmetric MC kernel.
//!
//! ## Parameters and their energy dependence
//!
//! - `α(E)` [1/µs]: fast moderation/leakage rate; sets the prompt width.
//!   Leading epithermal scaling `α ∝ √E` (Mantid `α = 1/(α₀+α₁λ)`, λ ∝ 1/√E).
//! - `β(E)` [1/µs]: slow storage rate; sets the delayed tail. Constant by
//!   default; optionally energy-dependent through an [`EnergyLaw`].
//! - `R(E)`, 0 ≤ R ≤ 1: storage mixing fraction; `R ≈ exp(−E_meV/κ)` → **R→0 in
//!   the 1–200 eV resonance regime**, so IC there is dominated by the
//!   one-parameter prompt Gamma(3, α(E)) term.
//!
//! Parameters are **fixed** when the instrument scientist provides them, or
//! **fit** from a known calibration foil otherwise (general case).
//!
//! ## Time → energy kernel and centering
//!
//! For a flight path `L`, a resonance at `E_r` has nominal TOF
//! `t_r = TOF_FACTOR·L/√E_r`. A neutron with emission delay τ arrives at TOF
//! `t_r+τ`, apparent energy `E' = (TOF_FACTOR·L/(t_r+τ))²` — that is the kernel's
//! *definition* (its positive-τ tail is delayed emission). Sampling `I(τ)` on
//! a τ-grid and mapping to TOF-offsets yields exactly the `(offset, weight)`
//! kernel representation that [`crate::resolution::TabulatedResolution`]
//! consumes — so IC rides the *same* verified broadening machinery, whose
//! *application* is the convolution gather: the broadened value at measured TOF
//! `t` reads theory at `t−τ` (a neutron measured at `t` with delay τ really flew
//! `t−τ`; see `resolution::broaden` and SAMMY `udr/mudr4.f90` `Ud_Convolute`).
//! At apply time
//! `interpolated_kernel` blends the two bracketing reference kernels as a
//! width-normalized shape blend (offsets scaled to the geometrically
//! interpolated width, shapes merged on the union grid) — unequal point counts
//! from IC's per-kernel tail trimming no longer trigger a nearest-reference
//! fallback, so the between-reference width follows the physical power law
//! smoothly. IC also synthesizes a dense reference
//! grid (default 64 energies), so the between-reference error is negligible. The
//! kernel is anchored with its **mode at offset 0** (peak-centering), matching the
//! UDR file convention (peak at offset 0); `interpolated_kernel` does not
//! re-center it. Because the IC pulse is right-skewed, its *mean* lags its mode by
//! ~1/α(E) in TOF, so the centroid — and even the minimum — of a broadened
//! resonance shifts toward **lower apparent energy** by an α(E)-dependent amount
//! (order 1e-2 eV, ~1e-3 relative, for α≈1.5 in the eV regime; larger toward
//! lower energy). So it *does* move the broadened dip off the nominal energy,
//! by a small amount.
//!
//! For the prompt-only law `α(E) = a0·√E + a1` with `R≈0`, the lag `~1/α(E)` is
//! **exactly** the `1/√E` basis of a flight-path (`L_scale`) error *iff* `a1 = 0`:
//! then the centroid offset scales as `c/√E`, and an `L → L(1+δ)` change shifts
//! every TOF by `δ·K·L/√E`, also `∝ 1/√E`, so the two are degenerate. With `a1 ≠ 0`
//! the lag `1/(a0√E + a1)` only *approximately* follows that basis (and the
//! storage term `R/β`, negligible in the eV regime, adds a further small
//! departure). To leading order, then, the lag is **confounded with the energy
//! scale**, and is handled by the SHARED `(t0, L_scale)` energy scale, not by any
//! per-family knob:
//! - **Run-time fitting** fits the `t0`/`L_scale` energy-scale, which absorbs the
//!   constant-`L` part of the lag exactly (same basis); only the *shape*
//!   (skew/tail) of the asymmetry is not absorbable by position.
//! - **Resolution calibration** (the `nereids-fitting` calibrator) **pins** the
//!   energy scale by default — a pure shape/width fit. It can optionally fit a
//!   SHARED `(t0, L_scale)` under a metrology prior (`with_position_prior`) for
//!   joint energy-scale / identifiability work. A *free, per-family* position knob
//!   is deliberately NOT used: because the lag is the same basis as `L_scale`, a
//!   free position lets a wrong (symmetric) family imitate the asymmetric shift
//!   and erodes the model-selection χ² (the discriminator is then only the
//!   position-independent skew/tail). See `nereids-fitting`'s
//!   `free_l_scale_absorbs_asymmetric_lag_and_erodes_discrimination`.
//!
//! `ic_centering_shifts_broadened_symmetric_dip_with_alpha` quantifies the bare
//! mode→centroid shift; the calibrator's `fit_t0_recovers_injected_energy_scale_shift`
//! checks the energy-scale fit recovers an injected offset. Re-centering the kernel
//! on its centroid (a future convention change spanning the UDR path too) would
//! remove the shift at the source.
//!
//! ## Optional instrument convolutions
//!
//! The full instrument function folds the moderator with a proton-burst
//! (Gaussian σ) and a chopper/channel (triangle, FWHM) term. Both are optional
//! here (`None` ⇒ omitted).
//!
//! **Provenance of the triangle (`channel_fwhm_us`).** SAMMY broadens for the
//! accelerator burst either as a Gaussian of FWHM `DELTAG` (SAMMY Manual R8
//! Sec. III.C.1.a, eq. III C1 a.12) or as a square pulse of width `BURST`
//! (Sec. III.C.2.a). At SNS the proton pulse delivered to the target is shaped
//! by the accumulator ring (Proton Storage Ring, PSR) into an approximately
//! triangular ~700 ns base — FWHM ≈ 350 ns — which is what the VENUS tabulated
//! FTS kernel header records as "folded triang FWHM 350 ns PSR". NEREIDS folds
//! that PSR triangle via `channel_fwhm_us` (symmetric triangle, half-base =
//! FWHM, `triangle_kernel`). Note: a tabulated file whose header says the
//! triangle is already "folded" in must NOT be double-counted against an IC
//! model that also applies it (the `nereids-fitting` calibrator therefore
//! applies its `psr_fwhm_ns` fold to the IC family only, never to
//! tabulated/UDR kernels).

use crate::resolution::{
    ResolutionParseError, TOF_FACTOR, TabulatedResolution, piecewise_linear_bin_masses,
};

/// de Broglie wavelength factor: λ (Å) = `LAMBDA_ANGSTROM_FACTOR` / √(E in eV).
///
/// `λ = h/√(2·m_n·E)`; with CODATA 2018 `h`, `m_n` and the 2019-SI eV this is
/// `0.285993 Å·√eV`. Used only by [`EnergyLaw::InverseLambda`]; its precise
/// value folds into the fitted `α₀,α₁`, so high precision is not load-bearing.
const LAMBDA_ANGSTROM_FACTOR: f64 = 0.285_993;

/// Rates below this (1/µs) are clamped to keep the pulse well-defined.
const MIN_RATE: f64 = 1e-9;

/// Relative-to-peak weight below which kernel tails are trimmed.
const TRIM_REL: f64 = 1e-7;

/// Default number of log-spaced reference energies in a synthesized table.
pub const DEFAULT_N_ENERGIES: usize = 64;

/// Default number of τ-samples spanning the prompt core of each kernel.
pub const DEFAULT_N_TAU: usize = 600;

/// Minimum accepted `n_tau` (τ-samples across the prompt core). Doubles as
/// the module's prompt-core **resolution floor**: when [`MAX_TAU_SAMPLES`]
/// widens the τ-step (see [`tau_geometry`]), the step may never exceed
/// `fast_reach / (MIN_N_TAU − 1)` — the coarsest prompt sampling
/// [`IkedaCarpenter::new`] has ever accepted as valid via its `n_tau ≥ 8`
/// check.
const MIN_N_TAU: usize = 8;

/// Minimum samples per side SPANNING a sampled channel triangle
/// (`dtau ≤ FWHM / 3`, half-base = FWHM). At the exactly-admitted boundary
/// `dtau = FWHM/3` the per-side samples sit at `{FWHM/3, 2FWHM/3, FWHM}` —
/// triangle weights `{2/3, 1/3, 0}` — so each side carries ≥ 2 strictly
/// interior (nonzero) samples while the endpoint lands ON the triangle
/// zero; the sampled fold is distinctly non-delta (discrete variance
/// 4·FWHM²/27 ≈ 89 % of the analytic FWHM²/6). Coarser steps degenerate
/// the discrete triangle toward the exact delta `[0, 1, 0]`, silently
/// erasing a requested fold — [`tau_geometry`] rejects that instead
/// (strictly: `capped_step > FWHM/3`).
///
/// This floor guarantees MOMENT-level accuracy (the calibration consumer's
/// contract); per-bin detector probabilities need the far stricter
/// [`TRI_BIN_SAMPLES_PER_SIDE`], enforced at the detector-bin gate rather
/// than here so realistic long-storage-tail calibrations stay buildable
/// under the [`MAX_TAU_SAMPLES`] cap.
const TRI_MIN_SAMPLES_PER_SIDE: f64 = 3.0;

/// Per-bin accuracy floor for the SAMPLED detector-bin path
/// (`dtau ≤ FWHM / TRI_BIN_SAMPLES_PER_SIDE` required by
/// `detector_bin_probabilities` when a channel fold is active). The
/// point-sampled discrete convolution mis-assigns individual detector-bin
/// probability as O(dtau²) even while the total mass is conserved (the
/// triangle kernel's kink drives the error; measured at α = 1 µs⁻¹,
/// FWHM = 10 µs: 7.5e-3 max per-bin error at 3 samples per side, ~1.2e-4 at
/// 24, ~3e-7 at the 600-sample default). The per-bin gate takes the
/// STRICTEST of the applicable floors — this one, the burst
/// [`GAUSS_BIN_SAMPLES_PER_SIGMA`], and the prompt-core
/// [`PROMPT_BIN_SAMPLES`] — so every accepted per-bin call is bounded at
/// ~1e-4 whichever feature binds; a coarser sampled grid is rejected loudly
/// rather than silently redistributing leading-edge mass. SAMMY's UDR
/// convolution integrates piecewise-linear segment products analytically
/// (`udr/mudr4.f90` `Ud_Convolute`/`Udr_Add`) and needs no such floor; the
/// sampled route keeps one and enforces it at the consumer whose contract
/// is per-bin.
const TRI_BIN_SAMPLES_PER_SIDE: f64 = 24.0;

/// Per-bin accuracy floor for a sampled Gaussian burst
/// (`dtau ≤ σ / GAUSS_BIN_SAMPLES_PER_SIGMA` at the detector-bin gate).
/// Measured at α = 1 µs⁻¹, σ = 2 µs, admitted `dtau = σ`: 1.46e-2 max
/// per-bin error while the total conserved to 5e-10; the O(dtau²) scaling
/// puts twelve samples per σ at ~1e-4. SAMMY integrates the Gaussian burst
/// analytically over piecewise-linear segments (`Ud_Burst`) and needs no
/// floor; the sampled sibling of the triangle gate keeps one.
const GAUSS_BIN_SAMPLES_PER_SIGMA: f64 = 12.0;

/// Per-bin accuracy floor for the PROMPT core on the sampled fold path
/// (`dtau ≤ fast_reach / PROMPT_BIN_SAMPLES`). A wide fold can set a bin
/// floor far coarser than the Γ₃ pulse's own structure (scale `1/α`), so
/// the fold floors alone would admit prompt-undersampled grids; the α = 1,
/// FWHM = 10 µs measurement (1.2e-4 at dtau = fast_reach/43) anchors
/// forty-eight samples across the prompt reach at ~1e-4. Only the sampled
/// fold path needs this — the fold-free branch is analytic.
const PROMPT_BIN_SAMPLES: f64 = 48.0;

/// Reach of the sampled/folded Gaussian burst in standard deviations. At
/// ±`GAUSS_REACH_SIGMAS`·σ = ±8σ the truncated two-sided Gaussian mass is
/// erfc(8/√2) ≈ 1.2e-15, so the retained-mass bookkeeping in
/// [`gaussian_kernel`] stays exact at f64 scale and no physically meaningful
/// mass is discarded for any detector window. Also fixes the burst
/// resolution floor `dtau ≤ σ` (≥ `GAUSS_REACH_SIGMAS` samples per side).
const GAUSS_REACH_SIGMAS: f64 = 8.0;

/// Prompt-tail reach in e-folds: the τ-grid spans `FAST_REACH_E_FOLDS / α`
/// so the prompt Gamma(3) tail `τ²e^{−ατ}` is < ~1e-8 of peak at the edge.
const FAST_REACH_E_FOLDS: f64 = 18.0;

/// Slow/storage-tail reach in e-folds (`SLOW_REACH_E_FOLDS / β` when storage
/// is active). Sits AT the trim horizon: `e^{−16} ≈ 1.1e-7 ≈` [`TRIM_REL`]
/// (`ln(1/TRIM_REL) ≈ 16.1`), so the reach cannot be truncated shorter
/// without discarding tail weight the [`TRIM_REL`] trim would keep — no
/// hidden approximation lives in this constant.
const SLOW_REACH_E_FOLDS: f64 = 16.0;

/// Storage fractions below this are treated as "no storage tail" when sizing
/// the τ-grid. DELIBERATELY two decades below [`TRIM_REL`], not derived from
/// it: whether a slow tail actually survives the [`TRIM_REL`] trim depends on
/// the α/β contrast (the trim threshold is relative to the prompt-dominated
/// peak, and the slow term's peak weight scales with both R and β/α), so no
/// single constant derived from [`TRIM_REL`] is exact for every admitted
/// (α, β) — treating a larger R as absent would be a hidden approximation.
/// The margin's only consequence is conservatism: for R ∈ (1e-9, ~1e-7) the
/// τ-grid is sized — and the [`MAX_TAU_SAMPLES`] cap gate applied — for a
/// slow tail whose weight the trim then discards anyway, spending samples
/// (or rejecting a configuration) for physics that cannot appear in the
/// kernel. It can never drop tail weight the trim would have kept.
const R_NEGLIGIBLE: f64 = 1e-9;

/// Cap on the τ-sample count spanning the PULSE BODY (`[0, τ_max]`) of one
/// synthesized kernel. The τ-step is anchored to the prompt core and refined
/// to resolve active folds (see [`tau_geometry`]), so a long storage tail
/// (β ≪ α with R > 0) grows the sample COUNT rather than the step; this cap
/// bounds that growth (CPU/memory) by widening the step to
/// `tau_max / (MAX_TAU_SAMPLES − 1)` — but never past the resolution floor
/// (prompt core: `fast_reach / (MIN_N_TAU − 1)`; triangle: FWHM /
/// [`TRI_MIN_SAMPLES_PER_SIDE`]; burst: σ). A parameter/grid combination
/// whose floor cannot be met within the cap is REJECTED loudly by
/// [`IkedaCarpenter::new`] instead of silently under-sampled. The actual
/// guarantee is on the pulse body only: the symmetric burst/channel fold
/// margin (± `GAUSS_REACH_SIGMAS·σ + FWHM` at the resolved step) adds its
/// samples ON TOP of the cap, so the final grid can exceed
/// `MAX_TAU_SAMPLES` by the margin sample count.
const MAX_TAU_SAMPLES: usize = 8192;

/// Taylor expansion of `h(u)/u³` where `h(u) = 1 − e^{−u}(1 + u + ½u²)`, for
/// `|u|` small (the `α ≈ β` limit), where direct evaluation cancels
/// catastrophically. As `u → 0`, `h(u)/u³ → 1/6`.
#[inline]
fn h_over_cube_taylor(u: f64) -> f64 {
    // h(u)/u³ = 1/6 − u/8 + u²/20 − u³/72 + u⁴/336 + O(u⁵). Carrying the u⁴ term
    // makes the bounded↔Taylor branch boundary (|u|=0.05) continuous to ~1e-11,
    // below any tolerance that consumes the synthesized kernel.
    let u2 = u * u;
    1.0 / 6.0 - u / 8.0 + u2 / 20.0 - u2 * u / 72.0 + u2 * u2 / 336.0
}

/// Ikeda–Carpenter moderator emission density `I(τ)`.
///
/// `τ` in µs, rates `α,β` in 1/µs, mixing `r ∈ [0,1]`. Returns 0 for `τ < 0`.
/// Unit-area over `τ ∈ [0,∞)`; first moment `3/α + r/β`. **NaN-free** for all
/// `α,β > 0` (including `α ≈ β` and `β ≫ α`): the slow/storage term is evaluated
/// from the bounded bracket `e^{−βτ} − e^{−ατ}(1+u+½u²)` (both exponentials ≤ 1
/// for `τ,α,β > 0`, so no `e^{|u|}` overflow), falling back to a Taylor limit
/// near `u = α−β·τ → 0` where that bracket cancels.
#[must_use]
pub fn ic_pulse(alpha: f64, beta: f64, r: f64, tau: f64) -> f64 {
    // Same domain contract as ic_cdf: non-finite parameters return NaN so
    // garbage in stays visible instead of flooring into a plausible pulse.
    if !alpha.is_finite() || !beta.is_finite() || !r.is_finite() {
        return f64::NAN;
    }
    if !tau.is_finite() || tau < 0.0 {
        return 0.0;
    }
    let alpha = alpha.max(MIN_RATE);
    let beta = beta.max(MIN_RATE);
    let at = alpha * tau;
    // prompt Gamma(3,α): α³τ²/2 · e^{−ατ} = (α/2)(ατ)² e^{−ατ}
    let fast = 0.5 * alpha * at * at * (-at).exp();
    if r <= 0.0 {
        return fast;
    }
    // slow/storage term = β(α/γ)³[e^{−βτ} − e^{−ατ}(1+u+½u²)], u = γτ = (α−β)τ.
    let u = (alpha - beta) * tau;
    let coeff = beta * alpha.powi(3) * tau.powi(3);
    let slow = if u.abs() < 0.05 {
        // α ≈ β: bracket/u³ → e^{−βτ}·h(u)/u³ (Taylor); avoids 0/0 cancellation.
        coeff * (-beta * tau).exp() * h_over_cube_taylor(u)
    } else {
        // Bounded form: both exponentials are ≤ 1, so β ≫ α (u ≪ 0) cannot
        // overflow (the old `e^{−βτ}·h(u)` factored an `e^{|u|}` → 0·∞ = NaN).
        let bracket = (-beta * tau).exp() - (-alpha * tau).exp() * (1.0 + u + 0.5 * u * u);
        coeff * bracket / (u * u * u)
    };
    (1.0 - r) * fast + r * slow
}

/// Cumulative probability of the Ikeda–Carpenter moderator pulse.
///
/// Returns `P(U <= tau)` for moderator delay `U`. `tau` is in µs and rates
/// are in 1/µs. The prompt term is the Gamma(3, α) cumulative distribution;
/// the storage term is the cumulative distribution of Gamma(3, α) plus an
/// independent exponential delay with rate β.
///
/// The expression is evaluated without subtracting nearly equal exponentials
/// when `α ≈ β`. This is the bin-integral companion to [`ic_pulse`].
///
/// Domain contract: non-finite `alpha`, `beta`, or `r` returns NaN so garbage
/// in stays visible; finite non-positive rates are floored to `MIN_RATE`
/// (1e-9 µs⁻¹) and `r <= 0` disables the storage term, matching
/// [`ic_pulse`]. A NaN `tau`
/// returns 0 — `tau` is the integration coordinate, not a parameter, and a
/// non-arriving coordinate contributes no mass.
#[must_use]
pub fn ic_cdf(alpha: f64, beta: f64, r: f64, tau: f64) -> f64 {
    if !alpha.is_finite() || !beta.is_finite() || !r.is_finite() {
        return f64::NAN;
    }
    if tau.is_nan() || tau <= 0.0 {
        return 0.0;
    }
    if tau == f64::INFINITY {
        return 1.0;
    }

    let alpha = alpha.max(MIN_RATE);
    let beta = beta.max(MIN_RATE);
    let at = alpha * tau;
    let fast = gamma3_cdf(at);
    if r <= 0.0 {
        return fast;
    }

    // Far in the tails the general expressions overflow — `at³` or `u³`
    // reach f64::INFINITY and produce inf·0 / inf/inf = NaN — while the
    // limits are exact to full f64 precision, so return them directly.
    // For at ≥ 1e100 the prompt CDF is 1 and the storage correction is
    // exp(−βτ) with relative error ≤ 3βτ/at; for βτ ≥ 1e100 the storage
    // delay is instantaneous and the correction is 0.
    const CDF_TAIL_LIMIT: f64 = 1.0e100;
    if at >= CDF_TAIL_LIMIT {
        return (1.0 - r * (-beta * tau).exp()).clamp(0.0, 1.0);
    }
    if beta * tau >= CDF_TAIL_LIMIT {
        return fast;
    }

    // The storage CDF is the prompt CDF minus the positive survival
    // correction below. Written this way, the α=β limit is Gamma(4, α).
    let u = (alpha - beta) * tau;
    let correction = if u.abs() < 0.05 {
        at.powi(3) * (-beta * tau).exp() * h_over_cube_taylor(u)
    } else {
        // Bounded form: neither exponential can overflow even when β >> α.
        let bracket = (-beta * tau).exp() - (-alpha * tau).exp() * (1.0 + u + 0.5 * u * u);
        at.powi(3) * bracket / u.powi(3)
    };
    (fast - r * correction).clamp(0.0, 1.0)
}

/// Gamma(3, rate=1) cumulative distribution at dimensionless time `x`.
fn gamma3_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Beyond x ≈ 750 the survival term exp(−x)·(1+x+x²/2) underflows to an
    // exact zero long before the polynomial can overflow (x² reaches
    // f64::INFINITY only at x ~ 1.3e154, where 0·inf would be NaN).
    if x >= 750.0 {
        return 1.0;
    }
    if x < 0.5 {
        // Integral of x² exp(-x) / 2 as an alternating series. The direct
        // `1 - exp(-x)(1+x+x²/2)` loses most digits for small x.
        let mut n = 0_u32;
        let mut term = x.powi(3) / 6.0;
        let mut sum = term;
        loop {
            let n_f = f64::from(n);
            term *= -x * (n_f + 3.0) / ((n_f + 1.0) * (n_f + 4.0));
            let next = sum + term;
            n += 1;
            if next == sum || n >= 128 {
                return next.clamp(0.0, 1.0);
            }
            sum = next;
        }
    }
    (1.0 - (-x).exp() * (1.0 + x + 0.5 * x * x)).clamp(0.0, 1.0)
}

/// Energy-dependence law for an Ikeda–Carpenter parameter.
///
/// A small closed set so the *fixed-or-fit* cases share one representation:
/// fixed ⇒ [`Const`](EnergyLaw::Const); fit ⇒ a parametric law whose
/// coefficients are the fit variables.
#[derive(Debug, Clone, PartialEq)]
pub enum EnergyLaw {
    /// Energy-independent constant.
    Const(f64),
    /// `a0·√(E[eV]) + a1` — leading epithermal scaling of the fast rate `α(E)`.
    SqrtE { a0: f64, a1: f64 },
    /// Mantid IC form `1/(a0 + a1·λ)`, λ (Å) = `LAMBDA_ANGSTROM_FACTOR`/√E.
    /// Behaves as `α ∝ √E` at low E and saturates to `1/a0` at high E.
    InverseLambda { a0: f64, a1: f64 },
    /// `exp(−E[meV]/kappa)` — storage fraction `R(E)`, → 0 in the eV regime.
    ExpMilliEv { kappa: f64 },
}

impl EnergyLaw {
    /// Evaluate the law at `energy_ev` (eV). Non-positive energy yields the
    /// `E→0` limit where well-defined (and a clamped value otherwise).
    #[must_use]
    pub fn eval(&self, energy_ev: f64) -> f64 {
        let e = energy_ev.max(0.0);
        match *self {
            EnergyLaw::Const(c) => c,
            EnergyLaw::SqrtE { a0, a1 } => a0 * e.sqrt() + a1,
            EnergyLaw::InverseLambda { a0, a1 } => {
                let denom = inverse_lambda_denom(a0, a1, e);
                if denom.abs() < MIN_RATE {
                    1.0 / MIN_RATE
                } else {
                    1.0 / denom
                }
            }
            EnergyLaw::ExpMilliEv { kappa } => {
                if kappa.abs() < MIN_RATE {
                    0.0
                } else {
                    (-(e * 1000.0) / kappa).exp()
                }
            }
        }
    }

    /// True when the law is numerically singular at `energy_ev`: the raw
    /// `InverseLambda` denominator lies inside the ±[`MIN_RATE`] window that
    /// [`EnergyLaw::eval`] floors away. The floor keeps a fit trial step from
    /// dividing by zero mid-optimization, but a *configured* law inside that
    /// window is a mathematically undefined rate, not a large one — without
    /// this check the floor converts an undefined (or tiny-negative)
    /// denominator into a plausible huge positive rate before the range
    /// validation below can see it.
    #[must_use]
    pub(crate) fn is_singular_at(&self, energy_ev: f64) -> bool {
        match *self {
            EnergyLaw::InverseLambda { a0, a1 } => {
                inverse_lambda_denom(a0, a1, energy_ev.max(0.0)).abs() < MIN_RATE
            }
            // κ = 0 is undefined and a tiny NEGATIVE κ is a divergent law
            // (exp(+E/|κ|)); eval's floor maps both to 0.0, which is the
            // legitimate κ → 0⁺ limit only for positive κ.
            EnergyLaw::ExpMilliEv { kappa } => (-MIN_RATE..=0.0).contains(&kappa),
            _ => false,
        }
    }
}

/// Raw `InverseLambda` denominator `a0 + a1·λ(E)` — shared by [`EnergyLaw::eval`]
/// and the singularity check so the two can never disagree on the window.
fn inverse_lambda_denom(a0: f64, a1: f64, e: f64) -> f64 {
    let lambda = if e > 0.0 {
        LAMBDA_ANGSTROM_FACTOR / e.sqrt()
    } else {
        f64::INFINITY
    };
    a0 + a1 * lambda
}

/// Parameters of the Ikeda–Carpenter resolution model.
#[derive(Debug, Clone)]
pub struct IkedaCarpenterParams {
    /// Fast (slowing-down) rate `α(E)`, 1/µs. Must evaluate to > 0.
    pub alpha: EnergyLaw,
    /// Slow (storage) rate `β(E)`, 1/µs. Must evaluate to > 0.
    pub beta: EnergyLaw,
    /// Storage mixing fraction `R(E)`, 0 ≤ R ≤ 1.
    pub r: EnergyLaw,
    /// Optional proton-burst Gaussian standard deviation (µs).
    pub burst_sigma_us: Option<f64>,
    /// Optional chopper/channel triangle FWHM (µs).
    pub channel_fwhm_us: Option<f64>,
}

impl IkedaCarpenterParams {
    /// A pure-moderator parameter set (no burst, no channel) with constant
    /// rates — the simplest case, useful for fixed-parameter or unit-test use.
    #[must_use]
    pub fn constant(alpha: f64, beta: f64, r: f64) -> Self {
        Self {
            alpha: EnergyLaw::Const(alpha),
            beta: EnergyLaw::Const(beta),
            r: EnergyLaw::Const(r),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        }
    }
}

/// Configuration of the energy / time grids used to synthesize the IC kernel
/// table. The energy grid spans the data range densely enough that interref
/// interpolation error is negligible.
#[derive(Debug, Clone)]
pub struct SynthesisGrid {
    /// Lowest reference energy (eV), > 0.
    pub e_min_ev: f64,
    /// Highest reference energy (eV), > `e_min_ev`.
    pub e_max_ev: f64,
    /// Number of log-spaced reference energies (≥ 2).
    pub n_energies: usize,
    /// Number of τ-samples spanning the prompt core of each kernel
    /// (≥ [`MIN_N_TAU`](crate::ikeda_carpenter) = 8). A long storage tail
    /// (β ≪ α with R > 0) grows the per-kernel sample count beyond `n_tau`;
    /// that count is capped at 8192 (`MAX_TAU_SAMPLES`), past which the
    /// τ-step widens — never below the resolution floor (prompt core at the
    /// `n_tau = 8` density, folds at ≥ 3 triangle samples per side / ≥ 1
    /// sample per burst σ): a combination that cannot be resolved within the
    /// cap is rejected by [`IkedaCarpenter::new`]. Active burst/channel folds
    /// add their ±(`GAUSS_REACH_SIGMAS`·σ + FWHM) margin samples on top of
    /// the cap.
    pub n_tau: usize,
}

impl SynthesisGrid {
    /// A sensible default grid: `[e_min, e_max]` log-spaced over
    /// [`DEFAULT_N_ENERGIES`] points with [`DEFAULT_N_TAU`] τ-samples.
    #[must_use]
    pub fn new(e_min_ev: f64, e_max_ev: f64) -> Self {
        Self {
            e_min_ev,
            e_max_ev,
            n_energies: DEFAULT_N_ENERGIES,
            n_tau: DEFAULT_N_TAU,
        }
    }
}

/// Analytical Ikeda–Carpenter resolution model.
///
/// Synthesizes a [`TabulatedResolution`] at construction and applies it through
/// the same broadening path as a Monte-Carlo file. Cloning is cheap (the
/// synthesized table is the only large field). Re-synthesize (construct anew)
/// when fitting changes the parameters.
#[derive(Debug, Clone)]
pub struct IkedaCarpenter {
    params: IkedaCarpenterParams,
    flight_path_m: f64,
    ref_energies: Vec<f64>,
    n_tau: usize,
    tabulated: TabulatedResolution,
}

impl IkedaCarpenter {
    /// Build the model, synthesizing its kernel table over `grid`.
    ///
    /// # Errors
    /// Returns [`ResolutionParseError::InvalidFormat`] for a non-positive
    /// flight path, a degenerate grid (`n_energies < 2`, `n_tau < 8`,
    /// `e_min ≤ 0`, `e_max ≤ e_min`), a non-positive `β(E)`, a parameter/grid
    /// combination whose τ-grid cannot resolve the prompt core and requested
    /// folds within the `MAX_TAU_SAMPLES` cap at some reference energy (see
    /// `tau_geometry` — remedy: larger `β`, `R = 0`, or a wider/disabled
    /// fold), or if the synthesized kernels fail
    /// [`TabulatedResolution::from_kernels`] validation.
    pub fn new(
        params: IkedaCarpenterParams,
        flight_path_m: f64,
        grid: &SynthesisGrid,
    ) -> Result<Self, ResolutionParseError> {
        if !flight_path_m.is_finite() || flight_path_m <= 0.0 {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Flight path must be a positive finite number, got {flight_path_m}"
            )));
        }
        if grid.n_energies < 2 {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "n_energies must be >= 2, got {}",
                grid.n_energies
            )));
        }
        if grid.n_tau < MIN_N_TAU {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "n_tau must be >= {MIN_N_TAU}, got {}",
                grid.n_tau
            )));
        }
        if !grid.e_min_ev.is_finite()
            || !grid.e_max_ev.is_finite()
            || grid.e_min_ev <= 0.0
            || grid.e_max_ev <= grid.e_min_ev
        {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Require finite 0 < e_min < e_max, got [{}, {}]",
                grid.e_min_ev, grid.e_max_ev
            )));
        }
        let ln_lo = grid.e_min_ev.ln();
        let ln_hi = grid.e_max_ev.ln();
        let denom = (grid.n_energies - 1) as f64;
        let ref_energies: Vec<f64> = (0..grid.n_energies)
            .map(|i| (ln_lo + (i as f64 / denom) * (ln_hi - ln_lo)).exp())
            .collect();

        // Reject singular rate laws before the range checks: a near-zero
        // InverseLambda denominator is an undefined configuration, and eval's
        // numerical floor would otherwise convert it into a plausible huge
        // positive rate that the α > 0 / β > 0 checks below cannot distinguish
        // from genuine physics.
        for (name, law) in [("α", &params.alpha), ("β", &params.beta), ("R", &params.r)] {
            if let Some(&bad) = ref_energies.iter().find(|&&e| law.is_singular_at(e)) {
                return Err(ResolutionParseError::InvalidFormat(format!(
                    "Ikeda–Carpenter {name}(E) law is singular at E = {bad} eV \
                     (an InverseLambda denominator within \
                     ±{MIN_RATE} of zero, or an ExpMilliEv κ in [−{MIN_RATE}, 0])"
                )));
            }
        }
        // Reject parameter laws that yield a non-positive fast rate α(E): the
        // pulse would otherwise degenerate (synthesis clamps α to a tiny floor,
        // producing a meaningless near-flat kernel rather than failing loudly).
        if let Some(&bad) = ref_energies.iter().find(|&&e| {
            let a = params.alpha.eval(e);
            !a.is_finite() || a <= 0.0
        }) {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Ikeda–Carpenter α(E) must be > 0, but α({bad}) = {} is not",
                params.alpha.eval(bad)
            )));
        }
        // β is a rate, so every synthesized reference energy must give a
        // positive finite value. Reject invalid laws instead of clamping them
        // into a different pulse.
        if let Some(&bad) = ref_energies.iter().find(|&&e| {
            let beta = params.beta.eval(e);
            !beta.is_finite() || beta <= 0.0
        }) {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Ikeda–Carpenter β(E) must be > 0, but β({bad}) = {} is not",
                params.beta.eval(bad)
            )));
        }
        // Reject storage-fraction laws that fall outside [0, 1] (synthesis clamps
        // R, masking a mis-specified law); a physical mixing fraction is in [0,1].
        if let Some(&bad) = ref_energies.iter().find(|&&e| {
            let r = params.r.eval(e);
            !r.is_finite() || !(0.0..=1.0).contains(&r)
        }) {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Ikeda–Carpenter R(E) must be in [0, 1], but R({bad}) = {} is not",
                params.r.eval(bad)
            )));
        }
        // Reject invalid optional instrument-convolution widths up front; synthesis
        // otherwise masks a negative width via `.abs()` and silently swallows a NaN.
        for (name, v) in [
            ("burst_sigma_us", params.burst_sigma_us),
            ("channel_fwhm_us", params.channel_fwhm_us),
        ] {
            if let Some(x) = v
                && (!x.is_finite() || x < 0.0)
            {
                return Err(ResolutionParseError::InvalidFormat(format!(
                    "Ikeda–Carpenter {name} must be finite and >= 0, got {x}"
                )));
            }
        }

        // Synthesis is fallible: a kernel whose τ-grid cannot resolve the
        // requested physics within MAX_TAU_SAMPLES (long slow tail vs a fine
        // fold / fast prompt core) errs loudly here instead of silently
        // degrading — see `tau_geometry`.
        let kernels: Vec<(Vec<f64>, Vec<f64>)> = ref_energies
            .iter()
            .map(|&e| synth_kernel(&params, grid.n_tau, e))
            .collect::<Result<_, _>>()?;

        let tabulated =
            TabulatedResolution::from_kernels(ref_energies.clone(), kernels, flight_path_m)?;

        Ok(Self {
            params,
            flight_path_m,
            ref_energies,
            n_tau: grid.n_tau,
            tabulated,
        })
    }

    /// The synthesized tabulated kernel set (the broadening engine).
    #[must_use]
    pub fn tabulated(&self) -> &TabulatedResolution {
        &self.tabulated
    }

    /// The IC parameters.
    #[must_use]
    pub fn params(&self) -> &IkedaCarpenterParams {
        &self.params
    }

    /// Flight-path length (m).
    #[must_use]
    pub fn flight_path_m(&self) -> f64 {
        self.flight_path_m
    }

    /// Reference energies (eV, ascending) the table was synthesized on.
    #[must_use]
    pub fn ref_energies(&self) -> &[f64] {
        &self.ref_energies
    }

    /// Evaluate the (burst/channel-folded) IC kernel at one energy.
    ///
    /// Returns ascending TOF-offsets (µs, mode at 0) and peak-normalized
    /// weights (max = 1), matching the [`TabulatedResolution`] storage
    /// convention.
    ///
    /// # Errors
    /// Returns [`ResolutionParseError::InvalidFormat`] when the requested
    /// energy or an energy-dependent rate/fraction is non-physical, or when
    /// the τ-grid cannot resolve the prompt core and requested folds within
    /// `MAX_TAU_SAMPLES` at this energy. Construction validates every
    /// *reference* energy, but a probe outside `[e_min, e_max]` can still leave
    /// the physical or resolvable region.
    pub fn kernel_at(&self, energy_ev: f64) -> Result<(Vec<f64>, Vec<f64>), ResolutionParseError> {
        self.validate_probe_energy(energy_ev)?;
        synth_kernel(&self.params, self.n_tau, energy_ev)
    }

    /// Evaluate the physical source pulse at one true neutron energy.
    ///
    /// Returns sampled moderator-delay coordinates in µs and peak-normalized
    /// densities. Unlike [`Self::kernel_at`], this method does not move the
    /// pulse mode to zero. With no symmetric proton/channel fold, the delay is
    /// causal and starts at zero. A symmetric fold may extend the sampled
    /// support below zero relative to its stated time origin.
    ///
    /// # Errors
    /// Returns [`ResolutionParseError::InvalidFormat`] if `energy_ev` is not a
    /// positive finite true energy, if an energy law is unphysical at that
    /// energy, or if the requested pulse cannot be resolved by the configured
    /// sampling limits.
    pub fn source_pulse_at(
        &self,
        energy_ev: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), ResolutionParseError> {
        self.validate_probe_energy(energy_ev)?;
        synth_source_pulse(&self.params, self.n_tau, energy_ev)
    }

    /// Probability that a neutron of known true energy is recorded in each
    /// supplied detector-time bin.
    ///
    /// `detector_time_edges_us` are the actual measured bin edges. The nominal
    /// arrival time is
    /// `timing_offset_us + TOF_FACTOR * flight_path_m / sqrt(true_energy_ev)`.
    /// `timing_offset_us` represents the shared clock/detector offset; it does
    /// not absorb or remove the moderator pulse's physical mode.
    ///
    /// The returned vector has one entry per adjacent edge pair and is not
    /// renormalized to the supplied window: bins that do not cover the full
    /// pulse correctly sum to less than one.
    ///
    /// With no burst or channel fold, probabilities come directly from the
    /// analytical IC CDF. With either optional fold, the continuous pulse is
    /// represented on the configured synthesis grid and integrated as a
    /// piecewise-linear density. The finite numerical support is not silently
    /// renormalized; omitted physical tail probability remains omitted.
    ///
    /// # Errors
    /// Returns [`ResolutionParseError::InvalidFormat`] unless the true energy
    /// is physical, the timing offset is finite, and at least two finite bin
    /// edges are supplied in strictly increasing order.
    pub fn detector_bin_probabilities(
        &self,
        true_energy_ev: f64,
        detector_time_edges_us: &[f64],
        timing_offset_us: f64,
    ) -> Result<Vec<f64>, ResolutionParseError> {
        self.validate_probe_energy(true_energy_ev)?;
        if !timing_offset_us.is_finite() {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "timing_offset_us must be finite, got {timing_offset_us}"
            )));
        }
        if detector_time_edges_us.len() < 2
            || detector_time_edges_us.iter().any(|x| !x.is_finite())
            || detector_time_edges_us.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(ResolutionParseError::InvalidFormat(
                "detector time edges must contain at least two finite, strictly increasing values"
                    .to_string(),
            ));
        }

        let nominal_arrival =
            timing_offset_us + TOF_FACTOR * self.flight_path_m / true_energy_ev.sqrt();
        let relative_edges: Vec<f64> = detector_time_edges_us
            .iter()
            .map(|edge| edge - nominal_arrival)
            .collect();

        if self.params.burst_sigma_us.unwrap_or(0.0) == 0.0
            && self.params.channel_fwhm_us.unwrap_or(0.0) == 0.0
        {
            let alpha = self.params.alpha.eval(true_energy_ev);
            let beta = self.params.beta.eval(true_energy_ev);
            let r = self.params.r.eval(true_energy_ev);
            return Ok(relative_edges
                .windows(2)
                .map(|edge| {
                    (ic_cdf(alpha, beta, r, edge[1]) - ic_cdf(alpha, beta, r, edge[0])).max(0.0)
                })
                .collect());
        }

        let (times, densities) =
            synth_source_pulse_density(&self.params, self.n_tau, true_energy_ev)?;
        // Per-bin accuracy gate: the point-sampled fold convolution
        // mis-assigns individual bins as O(dtau²) even while conserving the
        // total (leading-edge mass below the sampled support silently
        // redistributes into the window). Synthesis accepts the moment-level
        // steps for the calibration consumer; this per-bin consumer requires
        // the STRICTEST applicable bin floor — prompt core, channel
        // triangle, Gaussian burst — and rejects a coarser grid loudly.
        if times.len() >= 2 {
            let dtau = times[1] - times[0];
            let alpha_probe = self.params.alpha.eval(true_energy_ev);
            let mut bin_floor = FAST_REACH_E_FOLDS / alpha_probe.max(MIN_RATE) / PROMPT_BIN_SAMPLES;
            let mut binding = "prompt core".to_string();
            if let Some(fwhm) = self.params.channel_fwhm_us.filter(|&f| f > 0.0) {
                let tri = fwhm / TRI_BIN_SAMPLES_PER_SIDE;
                if tri < bin_floor {
                    bin_floor = tri;
                    binding = format!("{fwhm} µs channel triangle");
                }
            }
            if let Some(sigma) = self.params.burst_sigma_us.filter(|&s| s > 0.0) {
                let gauss = sigma / GAUSS_BIN_SAMPLES_PER_SIGMA;
                if gauss < bin_floor {
                    bin_floor = gauss;
                    binding = format!("{sigma} µs Gaussian burst");
                }
            }
            if dtau > bin_floor * (1.0 + 1e-12) {
                return Err(ResolutionParseError::InvalidFormat(format!(
                    "Ikeda–Carpenter detector-bin probabilities at E = \
                     {true_energy_ev} eV: the sampled τ-step {dtau:.4} µs \
                     exceeds the per-bin accuracy floor {bin_floor:.4} µs \
                     set by the {binding}; increase n_tau (or shorten the \
                     storage tail) so the sampled fold meets the per-bin bound"
                )));
            }
        }
        piecewise_linear_bin_masses(&times, &densities, &relative_edges).ok_or_else(|| {
            ResolutionParseError::InvalidFormat(format!(
                "Ikeda–Carpenter pulse at E = {true_energy_ev} eV has zero sampled area"
            ))
        })
    }

    fn validate_probe_energy(&self, energy_ev: f64) -> Result<(), ResolutionParseError> {
        if !energy_ev.is_finite() || energy_ev <= 0.0 {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "true energy must be positive and finite, got {energy_ev}"
            )));
        }
        for (name, law) in [
            ("alpha", &self.params.alpha),
            ("beta", &self.params.beta),
            ("R", &self.params.r),
        ] {
            if law.is_singular_at(energy_ev) {
                return Err(ResolutionParseError::InvalidFormat(format!(
                    "Ikeda–Carpenter {name}({energy_ev}) law is singular (an \
                     InverseLambda denominator within ±{MIN_RATE} of zero, \
                     or an ExpMilliEv κ in [−{MIN_RATE}, 0])"
                )));
            }
        }
        let alpha = self.params.alpha.eval(energy_ev);
        let beta = self.params.beta.eval(energy_ev);
        let r = self.params.r.eval(energy_ev);
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Ikeda–Carpenter alpha({energy_ev}) must be positive and finite, got {alpha}"
            )));
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Ikeda–Carpenter beta({energy_ev}) must be positive and finite, got {beta}"
            )));
        }
        if !r.is_finite() || !(0.0..=1.0).contains(&r) {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "Ikeda–Carpenter R({energy_ev}) must be in [0, 1], got {r}"
            )));
        }
        Ok(())
    }
}

/// τ-grid geometry for one kernel: `(dtau, tau_max, margin)`, or a
/// descriptive error when no exact sampled representation fits the cap.
///
/// The step is anchored to the PROMPT core — `n_tau` samples across the fast
/// Gamma(3) pulse (`fast_reach / (n_tau − 1)`) — and REFINED to resolve any
/// requested instrument fold (triangle: ≥ [`TRI_MIN_SAMPLES_PER_SIDE`]
/// samples per side, i.e. `dtau ≤ FWHM/TRI_MIN_SAMPLES_PER_SIDE`; Gaussian burst: `dtau ≤ σ`, i.e.
/// ≥ [`GAUSS_REACH_SIGMAS`] samples per ±`GAUSS_REACH_SIGMAS`·σ side). A longer storage tail
/// (β ≪ α, R > 0) extends the SAMPLE COUNT (`j_hi ∝ tau_max/dtau`) instead
/// of the step; [`MAX_TAU_SAMPLES`] bounds that count by widening the step —
/// but never past the resolution FLOOR (`fast_reach / (MIN_N_TAU − 1)` for
/// the prompt core, the fold minima above for folds). A combination whose
/// floor cannot be met within the cap has no faithful sampled representation
/// here, so it is rejected loudly: a capped step above the fold width would
/// degenerate the sampled triangle to an exact delta `[0,1,0]` (the fold
/// silently vanishes), and a capped step above the prompt scale steps OVER
/// the prompt pulse entirely (probe: α = 250, β = 0.02, R = 0.1 loses the
/// prompt's 0.9 weight share).
///
/// Bit-identical to the pre-#642-review `max(fast_reach/(n_tau−1),
/// tau_max/(MAX_TAU_SAMPLES−1))` step whenever no fold is finer than the
/// prompt design step and the capped step stays at or below the floor.
fn tau_geometry(
    params: &IkedaCarpenterParams,
    n_tau: usize,
    alpha: f64,
    beta: f64,
    r: f64,
) -> Result<(f64, f64, f64), String> {
    // τ_max: reach far enough that the prompt tail (e^{−ατ}) and, when
    // storage is active, the slow tail (e^{−βτ}) are below the trim level.
    let fast_reach = FAST_REACH_E_FOLDS / alpha;
    let slow_reach = if r > R_NEGLIGIBLE {
        SLOW_REACH_E_FOLDS / beta
    } else {
        0.0
    };
    let tau_max = fast_reach.max(slow_reach);

    // Requested step and resolution floor. `floor ≥ dtau_req` always: the
    // prompt terms satisfy MIN_N_TAU ≤ n_tau (validated by `new`) and the
    // fold terms are common to both.
    let mut dtau_req = fast_reach / (n_tau as f64 - 1.0);
    let mut floor = fast_reach / (MIN_N_TAU as f64 - 1.0);
    let mut fold_desc = String::new();
    // With any fold active, the REQUESTED step targets the per-bin accuracy
    // floors (prompt core, triangle, burst — see the *_BIN_* constants) so
    // the detector-bin path is accurate whenever the sample cap affords it;
    // the HARD floors stay at the moment level (MIN_N_TAU prompt density,
    // FWHM/TRI_MIN_SAMPLES_PER_SIDE, σ) so cap-limited long-tail
    // configurations still synthesize for the moment-level consumers
    // (calibration), and only the per-bin gate in
    // `detector_bin_probabilities` rejects them.
    let any_fold = params.channel_fwhm_us.filter(|&f| f > 0.0).is_some()
        || params.burst_sigma_us.filter(|&s| s > 0.0).is_some();
    if any_fold {
        dtau_req = dtau_req.min(fast_reach / PROMPT_BIN_SAMPLES);
    }
    if let Some(fwhm) = params.channel_fwhm_us
        && fwhm > 0.0
    {
        dtau_req = dtau_req.min(fwhm / TRI_BIN_SAMPLES_PER_SIDE);
        floor = floor.min(fwhm / TRI_MIN_SAMPLES_PER_SIDE);
        fold_desc.push_str(&format!(", channel triangle FWHM = {fwhm} µs"));
    }
    if let Some(sigma) = params.burst_sigma_us
        && sigma > 0.0
    {
        dtau_req = dtau_req.min(sigma / GAUSS_BIN_SAMPLES_PER_SIGMA);
        floor = floor.min(sigma);
        fold_desc.push_str(&format!(", burst σ = {sigma} µs"));
    }

    let capped_step = tau_max / (MAX_TAU_SAMPLES as f64 - 1.0);
    if capped_step > floor {
        return Err(format!(
            "the {MAX_TAU_SAMPLES}-sample τ-grid cap cannot resolve the requested physics: \
             the pulse spans τ_max = {tau_max:.3} µs (α = {alpha:.4} µs⁻¹, β = {beta:.4} µs⁻¹, \
             R = {r:.3}{fold_desc}), forcing a τ-step of {capped_step:.4} µs — above the \
             finest-feature resolution floor of {floor:.4} µs. Increase β (shorter storage \
             tail), set R = 0 (drop the storage term), or widen/disable the burst/channel fold"
        ));
    }
    Ok((dtau_req.max(capped_step), tau_max, margin_of(params)))
}

/// Symmetric τ-grid margin for the burst/channel folds:
/// ±([`GAUSS_REACH_SIGMAS`]·σ + FWHM), the folds' full reach. These samples
/// come ON TOP of [`MAX_TAU_SAMPLES`] (the cap governs the pulse body only).
fn margin_of(params: &IkedaCarpenterParams) -> f64 {
    params
        .burst_sigma_us
        .map_or(0.0, |s| GAUSS_REACH_SIGMAS * s)
        + params.channel_fwhm_us.unwrap_or(0.0)
}

/// Synthesize one `(offsets, weights)` kernel for `energy_ev` from the IC
/// parameters: sample `I(τ)`, fold in burst + channel, anchor the mode at
/// offset 0, trim negligible tails, peak-normalize.
///
/// # Errors
/// [`ResolutionParseError::InvalidFormat`] when [`tau_geometry`] cannot
/// resolve the prompt core and requested folds within [`MAX_TAU_SAMPLES`].
fn synth_kernel(
    params: &IkedaCarpenterParams,
    n_tau: usize,
    energy_ev: f64,
) -> Result<(Vec<f64>, Vec<f64>), ResolutionParseError> {
    let (mut offsets, weights) = synth_source_pulse(params, n_tau, energy_ev)?;
    let peak_time = offsets[argmax(&weights)];
    for offset in &mut offsets {
        *offset -= peak_time;
    }
    Ok((offsets, weights))
}

/// Synthesize one physical-time source pulse without moving its mode.
fn synth_source_pulse_density(
    params: &IkedaCarpenterParams,
    n_tau: usize,
    energy_ev: f64,
) -> Result<(Vec<f64>, Vec<f64>), ResolutionParseError> {
    let alpha = params.alpha.eval(energy_ev).max(MIN_RATE);
    let beta = params.beta.eval(energy_ev).max(MIN_RATE);
    let r = params.r.eval(energy_ev).clamp(0.0, 1.0);

    let (dtau, tau_max, margin) = tau_geometry(params, n_tau, alpha, beta, r).map_err(|msg| {
        ResolutionParseError::InvalidFormat(format!(
            "Ikeda–Carpenter kernel at E = {energy_ev} eV: {msg}"
        ))
    })?;

    // Extend the grid to slightly negative τ so a symmetric burst/channel can
    // spread the leading edge correctly (the moderator pulse itself is 0 there).
    // Widths are validated finite and >= 0 by `IkedaCarpenter::new`, so they are
    // used directly (no `.abs()` masking of a sign error).
    let j_lo: isize = -((margin / dtau).ceil() as isize);
    let j_hi: isize = ((tau_max + margin) / dtau).ceil() as isize;

    let taus: Vec<f64> = (j_lo..=j_hi).map(|j| j as f64 * dtau).collect();
    let mut weights: Vec<f64> = taus.iter().map(|&t| ic_pulse(alpha, beta, r, t)).collect();

    // Correct only the sampled quadrature error of the analytical moderator
    // density. The target is its exact CDF at the finite grid endpoint, not
    // one, so physical moderator probability beyond the grid is not moved
    // back into the sampled support. This matters for the coarsest admitted
    // n_tau values, where peak-normalization used to hide a large area error.
    let sampled_area = dtau
        * (0.5 * weights[0]
            + weights[1..weights.len() - 1].iter().sum::<f64>()
            + 0.5 * weights[weights.len() - 1]);
    let moderator_mass = ic_cdf(alpha, beta, r, *taus.last().expect("non-empty tau grid"));
    if !sampled_area.is_finite() || sampled_area <= 0.0 {
        return Err(ResolutionParseError::InvalidFormat(format!(
            "Ikeda–Carpenter pulse at E = {energy_ev} eV has zero sampled area"
        )));
    }
    let area_correction = moderator_mass / sampled_area;
    for weight in &mut weights {
        *weight *= area_correction;
    }

    if let Some(sigma) = params.burst_sigma_us
        && sigma > 0.0
    {
        let (kernel, retained_mass) = gaussian_kernel(dtau, sigma);
        weights = convolve_same(&weights, &kernel);
        for weight in &mut weights {
            *weight *= retained_mass;
        }
    }
    if let Some(fwhm) = params.channel_fwhm_us
        && fwhm > 0.0
    {
        weights = convolve_same(&weights, &triangle_kernel(dtau, fwhm));
    }

    let peak_idx = argmax(&weights);
    let peak_val = weights[peak_idx].max(f64::MIN_POSITIVE);

    // Trim tails below TRIM_REL of peak, keeping one guard sample each side so
    // the convolution's neighbor-difference trapezoid widths stay defined.
    let thresh = TRIM_REL * peak_val;
    let lo = weights
        .iter()
        .position(|&w| w > thresh)
        .map_or(0, |i| i.saturating_sub(1));
    let hi = weights
        .iter()
        .rposition(|&w| w > thresh)
        .map_or(weights.len() - 1, |i| (i + 1).min(weights.len() - 1));

    let offsets: Vec<f64> = (lo..=hi).map(|j| taus[j]).collect();
    let densities: Vec<f64> = (lo..=hi).map(|j| weights[j]).collect();
    Ok((offsets, densities))
}

/// Synthesize the public peak-normalized source-pulse representation.
fn synth_source_pulse(
    params: &IkedaCarpenterParams,
    n_tau: usize,
    energy_ev: f64,
) -> Result<(Vec<f64>, Vec<f64>), ResolutionParseError> {
    let (offsets, densities) = synth_source_pulse_density(params, n_tau, energy_ev)?;
    let peak = densities
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let weights = densities.into_iter().map(|value| value / peak).collect();
    Ok((offsets, weights))
}

/// Index of the maximum element (first on ties). Slice is non-empty by
/// construction in [`synth_kernel`].
fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    let mut best_v = xs[0];
    for (i, &x) in xs.iter().enumerate().skip(1) {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Symmetric, unit-sum Gaussian kernel sampled on a `dtau`-spaced grid out to
/// ±[`GAUSS_REACH_SIGMAS`]·σ.
fn gaussian_kernel(dtau: f64, sigma: f64) -> (Vec<f64>, f64) {
    let half = ((GAUSS_REACH_SIGMAS * sigma / dtau).ceil() as isize).max(1);
    let mut k: Vec<f64> = (-half..=half)
        .map(|j| {
            let t = j as f64 * dtau / sigma;
            (-0.5 * t * t).exp()
        })
        .collect();
    let raw_sum: f64 = k.iter().sum();
    let retained_mass = (raw_sum * dtau / (sigma * std::f64::consts::TAU.sqrt())).clamp(0.0, 1.0);
    normalize_sum(&mut k);
    (k, retained_mass)
}

/// Symmetric, unit-sum triangle kernel of FWHM `fwhm` (half-base = FWHM;
/// variance FWHM²/6) sampled on a `dtau`-spaced grid.
fn triangle_kernel(dtau: f64, fwhm: f64) -> Vec<f64> {
    let a = fwhm; // half-base equals FWHM for a symmetric triangle
    let half = ((a / dtau).ceil() as isize).max(1);
    let mut k: Vec<f64> = (-half..=half)
        .map(|j| (1.0 - (j as f64 * dtau).abs() / a).max(0.0))
        .collect();
    normalize_sum(&mut k);
    k
}

fn normalize_sum(k: &mut [f64]) {
    let s: f64 = k.iter().sum();
    if s > 0.0 {
        for v in k.iter_mut() {
            *v /= s;
        }
    }
}

/// Discrete convolution with a centered symmetric `kernel` (odd length),
/// returning an output the same length as `input` (zero-padded edges).
fn convolve_same(input: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = input.len();
    let kh = (kernel.len() / 2) as isize;
    let mut out = vec![0.0f64; n];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (kk, &kv) in kernel.iter().enumerate() {
            let src = i as isize + (kk as isize - kh);
            if src >= 0 && (src as usize) < n {
                acc += input[src as usize] * kv;
            }
        }
        *o = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::{
        ResolutionFunction, apply_resolution, apply_resolution_with_plan, build_resolution_plan,
    };
    use std::sync::Arc;

    /// Trapezoidal integral of `I(τ)` over a fine grid out to many decay times.
    fn pulse_area(alpha: f64, beta: f64, r: f64) -> f64 {
        let tau_max = (18.0 / alpha).max(if r > 0.0 { 18.0 / beta } else { 0.0 });
        let n = 200_000;
        let dt = tau_max / n as f64;
        let mut area = 0.0;
        for i in 0..n {
            let t0 = i as f64 * dt;
            let t1 = (i + 1) as f64 * dt;
            area += 0.5 * (ic_pulse(alpha, beta, r, t0) + ic_pulse(alpha, beta, r, t1)) * dt;
        }
        area
    }

    fn pulse_mean(alpha: f64, beta: f64, r: f64) -> f64 {
        let tau_max = (24.0 / alpha).max(if r > 0.0 { 24.0 / beta } else { 0.0 });
        let n = 400_000;
        let dt = tau_max / n as f64;
        let mut m = 0.0;
        for i in 0..n {
            let t = (i as f64 + 0.5) * dt;
            m += t * ic_pulse(alpha, beta, r, t) * dt;
        }
        m
    }

    #[test]
    fn pulse_is_unit_area() {
        for &(a, b, r) in &[
            (0.5, 0.05, 0.0),
            (1.0, 0.1, 0.3),
            (2.0, 0.2, 0.6),
            (0.8, 0.5, 0.9),
        ] {
            let area = pulse_area(a, b, r);
            assert!(
                (area - 1.0).abs() < 1e-3,
                "area for (α={a},β={b},R={r}) = {area}, expected 1"
            );
        }
    }

    #[test]
    fn pulse_mean_matches_formula() {
        // ⟨τ⟩ = 3/α + R/β
        for &(a, b, r) in &[(1.0, 0.1, 0.0), (1.0, 0.1, 0.4), (2.0, 0.25, 0.7)] {
            let want = 3.0 / a + r / b;
            let got = pulse_mean(a, b, r);
            assert!(
                (got - want).abs() / want < 2e-3,
                "mean (α={a},β={b},R={r}) = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn pulse_mode_is_two_over_alpha_for_pure_fast() {
        // r=0 ⇒ Gamma(3,α): mode at τ=2/α.
        let alpha = 1.3;
        let mode = 2.0 / alpha;
        let here = ic_pulse(alpha, 0.1, 0.0, mode);
        for d in [-0.3, -0.1, 0.1, 0.3] {
            assert!(ic_pulse(alpha, 0.1, 0.0, mode + d) <= here + 1e-12);
        }
    }

    #[test]
    fn pulse_alpha_equals_beta_is_finite_gamma4() {
        // α=β limit: slow term → α⁴τ³/6·e^{−ατ} (Gamma(4)). r=1 isolates it.
        let a = 1.0;
        let tau = 2.5;
        let got = ic_pulse(a, a, 1.0, tau);
        let want = a.powi(4) * tau.powi(3) / 6.0 * (-a * tau).exp();
        assert!(got.is_finite());
        assert!(
            (got - want).abs() < 1e-9,
            "α=β pulse {got} != Gamma(4) {want}"
        );
        // Near-degenerate (β just below α) must also be finite and close.
        let near = ic_pulse(a, a - 1e-9, 1.0, tau);
        assert!(near.is_finite() && (near - want).abs() < 1e-6);
    }

    #[test]
    fn pulse_is_finite_for_beta_much_greater_than_alpha() {
        // Regression: β ≫ α with storage active previously produced NaN (the
        // old e^{−βτ}·h(u) form factored an e^{|u|} that overflowed → 0·∞ = NaN).
        for &tau in &[0.0, 1.0, 50.0, 400.0, 1000.0] {
            let v = ic_pulse(0.05, 4.0, 0.5, tau);
            assert!(
                v.is_finite() && v >= 0.0,
                "ic_pulse(0.05,4,0.5,{tau}) = {v}"
            );
        }
        // The synthesized kernel must contain no non-finite entries.
        let p = IkedaCarpenterParams {
            alpha: EnergyLaw::Const(0.05),
            beta: EnergyLaw::Const(4.0),
            r: EnergyLaw::Const(0.5),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        };
        let (offs, wts) = synth_kernel(&p, 600, 1.0).unwrap();
        assert!(offs.iter().chain(wts.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn pulse_is_nonnegative_and_zero_before_t0() {
        assert_eq!(ic_pulse(1.0, 0.1, 0.5, -0.5), 0.0);
        for i in 0..200 {
            let t = i as f64 * 0.1;
            assert!(ic_pulse(1.0, 0.1, 0.5, t) >= 0.0);
        }
    }

    #[test]
    fn energy_law_eval() {
        assert_eq!(EnergyLaw::Const(3.0).eval(50.0), 3.0);
        let s = EnergyLaw::SqrtE { a0: 0.2, a1: 0.1 };
        assert!((s.eval(100.0) - (0.2 * 10.0 + 0.1)).abs() < 1e-12);
        // InverseLambda: α grows with E (λ shrinks).
        let il = EnergyLaw::InverseLambda { a0: 0.1, a1: 0.5 };
        assert!(il.eval(200.0) > il.eval(5.0));
        // R = exp(−E_meV/κ) → ~0 at eV-scale energies, ~1 at sub-meV.
        let rr = EnergyLaw::ExpMilliEv { kappa: 25.0 };
        assert!(rr.eval(10.0) < 1e-6); // 10 eV
        assert!(rr.eval(0.001) > 0.9); // 1 meV
    }

    #[test]
    fn kernel_mode_anchored_at_zero() {
        let p = IkedaCarpenterParams::constant(1.0, 0.1, 0.3);
        let (offsets, weights) = synth_kernel(&p, 600, 10.0).unwrap();
        let peak = argmax(&weights);
        // The peak offset is the closest to zero of all offsets.
        let peak_abs = offsets[peak].abs();
        for &o in &offsets {
            assert!(peak_abs <= o.abs() + 1e-9);
        }
        assert!(peak_abs < (offsets[1] - offsets[0]).abs() + 1e-9);
    }

    #[test]
    fn kernel_tail_points_to_positive_offset() {
        // Asymmetry: longer/heavier tail toward +offset (later TOF, lower E).
        let p = IkedaCarpenterParams::constant(1.0, 0.1, 0.2);
        let (offsets, weights) = synth_kernel(&p, 600, 10.0).unwrap();
        let max_pos = offsets.iter().cloned().fold(f64::MIN, f64::max);
        let min_neg = offsets.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            max_pos > min_neg.abs(),
            "expected longer +offset tail: +{max_pos} vs −{}",
            min_neg.abs()
        );
        let pos_w: f64 = offsets
            .iter()
            .zip(&weights)
            .filter(|(o, _)| **o > 0.0)
            .map(|(_, w)| *w)
            .sum();
        let neg_w: f64 = offsets
            .iter()
            .zip(&weights)
            .filter(|(o, _)| **o < 0.0)
            .map(|(_, w)| *w)
            .sum();
        assert!(pos_w > neg_w, "expected more weight at +offset");
    }

    #[test]
    fn higher_energy_gives_narrower_kernel() {
        // α(E) ∝ √E ⇒ prompt width 1/α shrinks with E ⇒ smaller TOF support.
        let p = IkedaCarpenterParams {
            alpha: EnergyLaw::SqrtE { a0: 0.3, a1: 0.0 },
            beta: EnergyLaw::Const(0.1),
            r: EnergyLaw::Const(0.0),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        };
        let support = |e: f64| {
            let (o, _) = synth_kernel(&p, 600, e).unwrap();
            o.iter().cloned().fold(f64::MIN, f64::max) - o.iter().cloned().fold(f64::MAX, f64::min)
        };
        assert!(support(100.0) < support(5.0));
    }

    #[test]
    fn synthesize_builds_valid_ascending_table() {
        let p = IkedaCarpenterParams::constant(1.0, 0.1, 0.2);
        let grid = SynthesisGrid {
            e_min_ev: 0.5e-3,
            e_max_ev: 1000.0,
            n_energies: 32,
            n_tau: 400,
        };
        let ic = IkedaCarpenter::new(p, 25.0, &grid).expect("synthesis");
        assert_eq!(ic.ref_energies().len(), 32);
        assert_eq!(ic.tabulated().ref_energies().len(), 32);
        for w in ic.ref_energies().windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn rejects_bad_config() {
        let p = IkedaCarpenterParams::constant(1.0, 0.1, 0.2);
        let bad = SynthesisGrid {
            e_min_ev: 1.0,
            e_max_ev: 0.5,
            n_energies: 16,
            n_tau: 100,
        };
        assert!(IkedaCarpenter::new(p.clone(), 25.0, &bad).is_err());
        assert!(IkedaCarpenter::new(p, -1.0, &SynthesisGrid::new(1.0, 10.0)).is_err());
        // A parameter law that yields α(E) ≤ 0 is rejected, not silently clamped.
        let neg_alpha = IkedaCarpenterParams {
            alpha: EnergyLaw::Const(-1.0),
            ..IkedaCarpenterParams::constant(1.0, 0.1, 0.0)
        };
        assert!(IkedaCarpenter::new(neg_alpha, 25.0, &SynthesisGrid::new(1.0, 100.0)).is_err());
        // Negative / non-finite burst or channel widths are rejected up front
        // (not `.abs()`-masked or NaN-swallowed during synthesis).
        for bad_width in [-1.0, f64::NAN, f64::INFINITY] {
            let neg_burst = IkedaCarpenterParams {
                burst_sigma_us: Some(bad_width),
                ..IkedaCarpenterParams::constant(1.0, 0.1, 0.0)
            };
            assert!(
                IkedaCarpenter::new(neg_burst, 25.0, &SynthesisGrid::new(1.0, 100.0)).is_err(),
                "burst_sigma_us={bad_width} should be rejected"
            );
            let neg_chan = IkedaCarpenterParams {
                channel_fwhm_us: Some(bad_width),
                ..IkedaCarpenterParams::constant(1.0, 0.1, 0.0)
            };
            assert!(
                IkedaCarpenter::new(neg_chan, 25.0, &SynthesisGrid::new(1.0, 100.0)).is_err(),
                "channel_fwhm_us={bad_width} should be rejected"
            );
        }
    }

    #[test]
    fn broadens_constant_to_constant() {
        // An area-normalized kernel preserves a flat spectrum.
        let p = IkedaCarpenterParams::constant(1.0, 0.1, 0.3);
        let ic = IkedaCarpenter::new(p, 25.0, &SynthesisGrid::new(1.0, 200.0)).unwrap();
        let energies: Vec<f64> = (0..400).map(|i| 1.0 + i as f64 * 0.5).collect();
        let spectrum = vec![0.7f64; energies.len()];
        let res = ResolutionFunction::IkedaCarpenter(Arc::new(ic));
        let out = apply_resolution(&energies, &spectrum, &res).unwrap();
        // Interior points (away from grid edges where the kernel is clipped).
        for v in &out[40..energies.len() - 40] {
            assert!((v - 0.7).abs() < 1e-3, "flat broadening drifted: {v}");
        }
    }

    #[test]
    fn ic_centering_shifts_broadened_symmetric_dip_with_alpha() {
        // The IC kernel anchors its MODE at offset 0, but the right-skewed pulse's
        // intensity centroid lags the mode by ~1/α in TOF (module docstring).
        // Broadening a symmetric-in-energy resonance therefore moves the dip
        // minimum off the nominal energy by an α-dependent amount. This guards the
        // documented bias (the loop-closure calibration tests cannot see it) and
        // pins it against a silent "fix" (e.g. re-centering on the centroid).
        const E0: f64 = 20.0;
        // Absorption-weighted centroid of the broadened dip — a robust position
        // estimator (the bare minimum is fragile under the wide low-α kernel).
        fn dip_centroid_energy(alpha: f64) -> f64 {
            // Constant-α kernel so the shape (hence the mode→mean lag) is uniform.
            let p = IkedaCarpenterParams::constant(alpha, 0.1, 0.0);
            let ic = IkedaCarpenter::new(p, 25.0, &SynthesisGrid::new(1.0, 200.0)).unwrap();
            let res = ResolutionFunction::IkedaCarpenter(Arc::new(ic));
            // Fine uniform grid; symmetric (in energy) Gaussian dip centered at E0.
            let energies: Vec<f64> = (0..4000).map(|i| 10.0 + i as f64 * 0.005).collect();
            let spectrum: Vec<f64> = energies
                .iter()
                .map(|&e| 1.0 - 0.8 * (-((e - E0) / 0.1).powi(2)).exp())
                .collect();
            let out = apply_resolution(&energies, &spectrum, &res).unwrap();
            let (mut num, mut den) = (0.0, 0.0);
            for (&e, &t) in energies.iter().zip(&out) {
                if (e - E0).abs() <= 2.0 {
                    let a = (1.0 - t).max(0.0); // absorption weight
                    num += e * a;
                    den += a;
                }
            }
            num / den
        }
        let signed_small_alpha = dip_centroid_energy(0.8) - E0;
        let signed_large_alpha = dip_centroid_energy(2.0) - E0;
        // The shift is toward LOWER apparent energy: the delayed-emission
        // tail gathers theory from earlier TOF (higher E), so the theory
        // dip at E0 surfaces at measured energies below E0. A positive
        // shift here means the kernel was applied time-mirrored.
        assert!(
            signed_small_alpha < 0.0 && signed_large_alpha < 0.0,
            "centering shift must be toward lower energy: \
             α=0.8 {signed_small_alpha:+e}, α=2.0 {signed_large_alpha:+e}"
        );
        let shift_small_alpha = signed_small_alpha.abs();
        let shift_large_alpha = signed_large_alpha.abs();
        // The bias is real (resolvable at the 1e-3 eV level)…
        assert!(
            shift_small_alpha > 1e-3,
            "centering shift vanished: {shift_small_alpha}"
        );
        // …and shrinks with increasing α (the ~1/α scaling of the mode→mean lag).
        assert!(
            shift_small_alpha > 1.3 * shift_large_alpha,
            "shift should scale ~1/α: α=0.8 {shift_small_alpha} vs α=2.0 {shift_large_alpha}"
        );
    }

    #[test]
    fn plan_path_matches_direct_path() {
        let p = IkedaCarpenterParams::constant(1.2, 0.15, 0.25);
        let ic = IkedaCarpenter::new(p, 25.0, &SynthesisGrid::new(1.0, 200.0)).unwrap();
        let res = ResolutionFunction::IkedaCarpenter(Arc::new(ic));
        let energies: Vec<f64> = (0..300).map(|i| 1.0 + i as f64 * 0.6).collect();
        // A localized dip to exercise the convolution non-trivially.
        let spectrum: Vec<f64> = energies
            .iter()
            .map(|&e| if (e - 90.0).abs() < 5.0 { 0.2 } else { 1.0 })
            .collect();
        let direct = apply_resolution(&energies, &spectrum, &res).unwrap();
        let plan = build_resolution_plan(&energies, &res).unwrap();
        let planned =
            apply_resolution_with_plan(plan.as_ref(), &energies, &spectrum, &res).unwrap();
        assert_eq!(direct.len(), planned.len());
        for (a, b) in direct.iter().zip(&planned) {
            assert!((a - b).abs() < 1e-12, "plan vs direct mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn tau_step_anchors_to_prompt_core_not_storage_tail() {
        // Regression for the τ-grid fix: with a slow storage tail active
        // (β ≪ α, R > 0) the τ-step must stay anchored to the prompt core —
        // the old `tau_max/(n_tau−1)` step let the tail dilate dtau and
        // degenerate the 0.35 µs channel triangle toward a delta.
        let n_tau = 600;
        let alpha = 2.0; // fast_reach = 18/α = 9 µs
        let prompt_step = (18.0 / alpha) / (n_tau as f64 - 1.0);

        // (a) Moderate tail (β = 0.25 ⇒ slow_reach = 64 µs): the cap does not
        // bite and the 0.5 µs triangle's per-bin refinement target
        // (FWHM/24 ≈ 0.021 µs) sits above the prompt-core step, so the step
        // equals the prompt-core step exactly and the grid still spans the
        // full tail.
        let p = IkedaCarpenterParams {
            channel_fwhm_us: Some(0.5),
            ..IkedaCarpenterParams::constant(alpha, 0.25, 0.5)
        };
        let (offs, wts) = synth_kernel(&p, n_tau, 10.0).unwrap();
        let dtau = offs[1] - offs[0];
        assert!(
            (dtau - prompt_step).abs() < 1e-12,
            "uncapped step {dtau} != prompt-core step {prompt_step}"
        );
        assert!(
            dtau <= prompt_step + 1e-12,
            "prompt-core spacing {dtau} > fast_reach/(n_tau−1) = {prompt_step}"
        );
        // ≥ 3 nonzero triangle samples per side at this step.
        let tri = triangle_kernel(dtau, 0.5);
        let nonzero_per_side = tri.iter().take(tri.len() / 2).filter(|&&v| v > 0.0).count();
        assert!(
            nonzero_per_side >= 3,
            "triangle degenerated: {nonzero_per_side} nonzero samples per side"
        );
        // The tail is still reached: for β = 0.25 the slow tail crosses the
        // TRIM_REL = 1e-7 trim level near τ ≈ 63 µs (e^{−βτ}·slow-amplitude ÷
        // peak = 1e-7 at τ ≈ 62.6 µs), so the trimmed kernel must still span
        // well past 40 µs — a step anchored to τ_max instead of the prompt
        // core would pass a narrow-span check, but a grid that stops short of
        // the storage tail would not survive this one.
        let span = offs.last().unwrap() - offs.first().unwrap();
        let peak = wts.iter().cloned().fold(f64::MIN, f64::max);
        assert!((peak - 1.0).abs() < 1e-12, "weights not peak-normalized");
        assert!(span > 3.0 / alpha, "kernel span {span} lost the pulse body");
        assert!(
            span > 40.0,
            "kernel span {span} µs stops short of the β = 0.25 storage tail \
             (trim horizon ≈ 63 µs)"
        );

        // (b) Extreme admitted tail (β = 0.02 ⇒ slow_reach = 800 µs): the
        // MAX_TAU_SAMPLES cap bites; the step widens to tau_max/(cap−1) but
        // must still resolve the 0.35 µs triangle with ≥ 3 samples per side
        // (the old tail-anchored step was 800/599 ≈ 1.34 µs — a delta).
        let p_ext = IkedaCarpenterParams {
            channel_fwhm_us: Some(0.35),
            ..IkedaCarpenterParams::constant(alpha, 0.02, 0.5)
        };
        let (offs_ext, _) = synth_kernel(&p_ext, n_tau, 10.0).unwrap();
        let dtau_ext = offs_ext[1] - offs_ext[0];
        let capped_step = 800.0 / (MAX_TAU_SAMPLES as f64 - 1.0);
        assert!(
            (dtau_ext - capped_step).abs() < 1e-9,
            "capped step {dtau_ext} != tau_max/(MAX_TAU_SAMPLES−1) = {capped_step}"
        );
        assert!(
            dtau_ext <= 0.35 / 3.0,
            "capped step {dtau_ext} cannot resolve the 0.35 µs triangle"
        );
        let tri_ext = triangle_kernel(dtau_ext, 0.35);
        let nonzero_ext = tri_ext
            .iter()
            .take(tri_ext.len() / 2)
            .filter(|&&v| v > 0.0)
            .count();
        assert!(
            nonzero_ext >= 3,
            "triangle degenerated under cap: {nonzero_ext} nonzero samples per side"
        );
    }

    #[test]
    fn burst_and_channel_broaden_further() {
        // Folding in burst + channel widens the kernel TOF support.
        let base = IkedaCarpenterParams::constant(1.0, 0.1, 0.0);
        let (o0, _) = synth_kernel(&base, 600, 10.0).unwrap();
        let support0 = o0.iter().cloned().fold(f64::MIN, f64::max)
            - o0.iter().cloned().fold(f64::MAX, f64::min);
        let folded = IkedaCarpenterParams {
            burst_sigma_us: Some(0.3),
            channel_fwhm_us: Some(0.35),
            ..IkedaCarpenterParams::constant(1.0, 0.1, 0.0)
        };
        let (o1, _) = synth_kernel(&folded, 600, 10.0).unwrap();
        let support1 = o1.iter().cloned().fold(f64::MIN, f64::max)
            - o1.iter().cloned().fold(f64::MAX, f64::min);
        assert!(support1 > support0, "folded {support1} !> bare {support0}");
    }

    /// Sum-weighted variance of a sampled `(offsets, weights)` kernel on a
    /// uniform τ-grid. Normalization-independent (divides by Σw).
    fn kernel_variance(offsets: &[f64], weights: &[f64]) -> f64 {
        let w_sum: f64 = weights.iter().sum();
        let mean: f64 = offsets.iter().zip(weights).map(|(o, w)| o * w).sum::<f64>() / w_sum;
        offsets
            .iter()
            .zip(weights)
            .map(|(o, w)| (o - mean).powi(2) * w)
            .sum::<f64>()
            / w_sum
    }

    #[test]
    fn unresolvable_tau_grid_is_rejected_loudly() {
        // Review #645 F1, probe 1: β = 0.005 ⇒ slow reach 16/β = 3200 µs ⇒
        // capped step 3200/8191 ≈ 0.39 µs > the 0.35 µs triangle FWHM — the
        // sampled triangle would be the exact delta [0, 1, 0] and the
        // requested fold would silently vanish from the kernel. Must be a
        // loud construction error, not silent physics degradation.
        let p1 = IkedaCarpenterParams {
            channel_fwhm_us: Some(0.35),
            ..IkedaCarpenterParams::constant(2.0, 0.005, 0.5)
        };
        let err = IkedaCarpenter::new(p1.clone(), 25.0, &SynthesisGrid::new(1.0, 100.0))
            .expect_err("a delta-degenerate fold must be rejected");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("cannot resolve") && msg.contains("Increase"),
            "error must diagnose the cap and name remedies: {msg}"
        );
        // The per-energy synthesis (kernel_at path) refuses identically.
        assert!(synth_kernel(&p1, 600, 10.0).is_err());

        // Probe 2: α = 250 ⇒ the whole prompt pulse spans 18/α ≈ 0.07 µs,
        // less than ONE capped step (800/8191 ≈ 0.098 µs): sampling would
        // step over the prompt term entirely (0.9 of the pulse weight at
        // R = 0.1). Must also be rejected.
        let p2 = IkedaCarpenterParams::constant(250.0, 0.02, 0.1);
        assert!(
            IkedaCarpenter::new(p2, 25.0, &SynthesisGrid::new(1.0, 100.0)).is_err(),
            "a capped step wider than the prompt core must be rejected"
        );
    }

    #[test]
    fn requested_fold_is_never_a_silent_no_op() {
        // Adjacent to probe 1 but resolvable (β = 0.05 ⇒ capped step
        // 320/8191 ≈ 0.039 µs ≤ FWHM/3): when synthesis is Ok and a fold is
        // requested, the folded kernel must genuinely differ from the
        // unfolded one — by approximately the analytic fold variance
        // FWHM²/6 — never by ~0 (the silent delta no-op this guards against).
        let base = IkedaCarpenterParams::constant(2.0, 0.05, 0.5);
        let folded = IkedaCarpenterParams {
            channel_fwhm_us: Some(0.35),
            ..base.clone()
        };
        let (o0, w0) = synth_kernel(&base, 600, 10.0).unwrap();
        let (o1, w1) = synth_kernel(&folded, 600, 10.0).unwrap();
        let dv = kernel_variance(&o1, &w1) - kernel_variance(&o0, &w0);
        let expected = 0.35f64.powi(2) / 6.0;
        assert!(
            dv > 0.5 * expected,
            "fold variance increment {dv} µs² vs analytic {expected} µs²: \
             the requested fold (nearly) vanished"
        );
    }

    #[test]
    fn boundary_step_at_triangle_floor_is_admitted_and_fold_survives() {
        // Pin the exactly-admitted uncapped step `dtau = FWHM /
        // TRI_BIN_SAMPLES_PER_SIDE` (the bin-eager refinement target),
        // written in terms of the constant so the pin survives value
        // changes. For N samples per side the discrete triangle's variance
        // is (1 − 1/N²)·FWHM²/6 exactly (N = 3 gives the historical
        // 4·FWHM²/27 of the moment-level floor), and each side keeps N − 1
        // strictly interior nonzero samples with the endpoint ON the
        // triangle zero.
        //
        // Route to the boundary: n_tau = MIN_N_TAU = 8 with α = 1 gives a
        // prompt design step 18/7 ≈ 2.57 µs, far coarser than the target for
        // a 1 µs triangle, so the fold refinement pins dtau to exactly the
        // target. R = 0 keeps the cap inert (capped step 18/8191 ≪ target).
        let n = TRI_BIN_SAMPLES_PER_SIDE;
        let fwhm = 1.0;
        let p = IkedaCarpenterParams {
            channel_fwhm_us: Some(fwhm),
            ..IkedaCarpenterParams::constant(1.0, 0.1, 0.0)
        };
        let (offs, _) = synth_kernel(&p, MIN_N_TAU, 10.0)
            .expect("the dtau = floor boundary must be admitted, not rejected");
        let dtau = offs[1] - offs[0];
        let boundary = fwhm / n;
        assert!(
            (dtau - boundary).abs() < 1e-12,
            "step {dtau} µs is not the FWHM/{n} boundary {boundary} µs"
        );

        // The sampled triangle at the boundary: N − 1 strictly interior
        // nonzero samples per side, center far below the delta's 1.
        let tri = triangle_kernel(dtau, fwhm);
        let per_side = tri
            .iter()
            .take(tri.len() / 2)
            .filter(|&&v| v > 1e-9)
            .count();
        assert_eq!(
            per_side,
            n as usize - 1,
            "boundary triangle must keep N − 1 strictly interior nonzero \
             samples per side: {tri:?}"
        );
        let center = tri[tri.len() / 2];
        assert!(
            center < 0.5,
            "center weight {center} — boundary triangle degenerated toward a delta"
        );

        // Fold effectiveness: discrete variance (1 − 1/N²)·FWHM²/6.
        let tri_offs: Vec<f64> = (0..tri.len())
            .map(|i| (i as f64 - (tri.len() / 2) as f64) * dtau)
            .collect();
        let v = kernel_variance(&tri_offs, &tri);
        let want = (1.0 - 1.0 / (n * n)) * fwhm * fwhm / 6.0;
        assert!(
            ((v - want) / want).abs() < 0.02,
            "boundary triangle variance {v} µs² vs discrete analytic {want} µs²"
        );
        assert!(
            v > 0.5 * (fwhm * fwhm / 6.0),
            "boundary fold variance {v} µs² collapsed below half the \
             continuous FWHM²/6 — fold (nearly) vanished"
        );
    }

    #[test]
    fn fold_variance_matches_analytic_oracle() {
        // Independent analytic oracle for the fold convention (#645 F8): a
        // convolution adds second central moments, so the sampled kernel's
        // variance must grow by EXACTLY the fold kernel's analytic variance —
        // FWHM²/6 for the symmetric channel triangle (half-base = FWHM),
        // σ² for the Gaussian burst. Unlike the closed-loop calibration
        // tests, the expectation here comes from analysis, not from the
        // shared synthesis code. Pure-prompt pulse (R = 0) keeps trim /
        // edge-truncation error far below the increments.
        let base = IkedaCarpenterParams::constant(2.0, 0.1, 0.0);
        let (o0, w0) = synth_kernel(&base, 600, 10.0).unwrap();
        let v0 = kernel_variance(&o0, &w0);
        // Sanity: the bare Gamma(3, α) variance is 3/α².
        let v_gamma = 3.0 / (2.0f64 * 2.0);
        assert!(
            (v0 - v_gamma).abs() < 0.01 * v_gamma,
            "bare pulse variance {v0}, Gamma(3) analytic {v_gamma}"
        );

        let fwhm = 0.35;
        let tri = IkedaCarpenterParams {
            channel_fwhm_us: Some(fwhm),
            ..base.clone()
        };
        let (o1, w1) = synth_kernel(&tri, 600, 10.0).unwrap();
        let dv_tri = kernel_variance(&o1, &w1) - v0;
        let want_tri = fwhm * fwhm / 6.0;
        assert!(
            ((dv_tri - want_tri) / want_tri).abs() < 0.02,
            "triangle fold added {dv_tri} µs², analytic FWHM²/6 = {want_tri} µs²"
        );

        let sigma = 0.3;
        let gau = IkedaCarpenterParams {
            burst_sigma_us: Some(sigma),
            ..base.clone()
        };
        let (o2, w2) = synth_kernel(&gau, 600, 10.0).unwrap();
        let dv_gau = kernel_variance(&o2, &w2) - v0;
        let want_gau = sigma * sigma;
        assert!(
            ((dv_gau - want_gau) / want_gau).abs() < 0.02,
            "Gaussian burst added {dv_gau} µs², analytic σ² = {want_gau} µs²"
        );
    }
}
