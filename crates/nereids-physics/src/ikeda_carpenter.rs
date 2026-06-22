//! Ikeda–Carpenter analytical moderator resolution model.
//!
//! A third instrument-resolution model alongside the analytical Gaussian
//! ([`crate::resolution::ResolutionParams`]) and the Monte-Carlo tabulated
//! kernel ([`crate::resolution::TabulatedResolution`]). It exists to settle a
//! methodological dispute about the VENUS instrument resolution: one camp
//! trusts the MC-simulated tabulated kernel (UDD/FTS file); the instrument
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
//! - `β` [1/µs]: slow storage rate; sets the delayed tail. Energy-independent.
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
//! `t_r+τ`, apparent energy `E' = (TOF_FACTOR·L/(t_r+τ))²`. Sampling `I(τ)` on
//! a τ-grid and mapping to TOF-offsets yields exactly the `(offset, weight)`
//! kernel representation that [`crate::resolution::TabulatedResolution`]
//! consumes — so IC rides the *same* verified broadening machinery. At apply time
//! `interpolated_kernel` blends the two bracketing reference kernels when they
//! have equal point counts, else it falls back to the nearer reference; because
//! IC trims each kernel's tail independently the point counts often differ, so the
//! nearest-reference path is common. Either way IC synthesizes a dense reference
//! grid (default 64 energies), so the between-reference error is negligible. The kernel is anchored with its
//! **mode at offset 0** (peak-centering), matching the UDD file convention
//! (peak at offset 0); `interpolated_kernel` does not re-center it. Because the
//! IC pulse is skewed, its *mean* lags its mode, so a broadened resonance's
//! centroid carries a small TOF lag whose constant part is absorbed by the fitted
//! `t0/L` energy-scale. Its residual energy-dependent part (the mode→mean offset
//! varies with α(E)) is a known, bounded limitation shared with the peak-centered
//! UDD kernel — it does not move the broadened *peak* off the nominal energy.
//!
//! ## Optional instrument convolutions
//!
//! The full instrument function folds the moderator with a proton-burst
//! (Gaussian σ) and a chopper/channel (triangle, FWHM) term. Both are optional
//! here (`None` ⇒ omitted). Note: a tabulated file whose header says the
//! channel triangle is already "folded" in must NOT be double-counted against
//! an IC model that also applies the channel.

use crate::resolution::{ResolutionParseError, TabulatedResolution};

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

/// Taylor expansion of `h(u)/u³` where `h(u) = 1 − e^{−u}(1 + u + ½u²)`, for
/// `|u|` small (the `α ≈ β` limit), where direct evaluation cancels
/// catastrophically. As `u → 0`, `h(u)/u³ → 1/6`.
#[inline]
fn h_over_cube_taylor(u: f64) -> f64 {
    // h(u)/u³ = 1/6 − u/8 + u²/20 − u³/72 + O(u⁴).
    let u2 = u * u;
    1.0 / 6.0 - u / 8.0 + u2 / 20.0 - u2 * u / 72.0
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
                let lambda = if e > 0.0 {
                    LAMBDA_ANGSTROM_FACTOR / e.sqrt()
                } else {
                    f64::INFINITY
                };
                let denom = a0 + a1 * lambda;
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
}

/// Parameters of the Ikeda–Carpenter resolution model.
#[derive(Debug, Clone)]
pub struct IkedaCarpenterParams {
    /// Fast (slowing-down) rate `α(E)`, 1/µs. Must evaluate to > 0.
    pub alpha: EnergyLaw,
    /// Slow (storage) rate `β`, 1/µs. Energy-independent, must be > 0.
    pub beta: f64,
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
            beta,
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
    /// Number of τ-samples spanning the prompt core of each kernel (≥ 8).
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
    /// `e_min ≤ 0`, `e_max ≤ e_min`), a non-positive `β`, or if the synthesized
    /// kernels fail [`TabulatedResolution::from_kernels`] validation.
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
        if grid.n_tau < 8 {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "n_tau must be >= 8, got {}",
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
        if !params.beta.is_finite() || params.beta <= 0.0 {
            return Err(ResolutionParseError::InvalidFormat(format!(
                "beta must be a positive finite number, got {}",
                params.beta
            )));
        }

        let ln_lo = grid.e_min_ev.ln();
        let ln_hi = grid.e_max_ev.ln();
        let denom = (grid.n_energies - 1) as f64;
        let ref_energies: Vec<f64> = (0..grid.n_energies)
            .map(|i| (ln_lo + (i as f64 / denom) * (ln_hi - ln_lo)).exp())
            .collect();

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

        let kernels: Vec<(Vec<f64>, Vec<f64>)> = ref_energies
            .iter()
            .map(|&e| synth_kernel(&params, grid.n_tau, e))
            .collect();

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
    #[must_use]
    pub fn kernel_at(&self, energy_ev: f64) -> (Vec<f64>, Vec<f64>) {
        synth_kernel(&self.params, self.n_tau, energy_ev)
    }
}

/// Synthesize one `(offsets, weights)` kernel for `energy_ev` from the IC
/// parameters: sample `I(τ)`, fold in burst + channel, anchor the mode at
/// offset 0, trim negligible tails, peak-normalize.
fn synth_kernel(
    params: &IkedaCarpenterParams,
    n_tau: usize,
    energy_ev: f64,
) -> (Vec<f64>, Vec<f64>) {
    let alpha = params.alpha.eval(energy_ev).max(MIN_RATE);
    let beta = params.beta.max(MIN_RATE);
    let r = params.r.eval(energy_ev).clamp(0.0, 1.0);

    // τ_max: reach far enough that the prompt Gamma(3) tail (e^{−ατ}) and, when
    // storage is present, the slow tail (e^{−βτ}) are < ~1e-8 of peak.
    let fast_reach = 18.0 / alpha;
    let slow_reach = if r > 1e-9 { 16.0 / beta } else { 0.0 };
    let tau_max = fast_reach.max(slow_reach);
    let dtau = tau_max / (n_tau as f64 - 1.0);

    // Extend the grid to slightly negative τ so a symmetric burst/channel can
    // spread the leading edge correctly (the moderator pulse itself is 0 there).
    let margin = params.burst_sigma_us.map_or(0.0, |s| 4.0 * s.abs())
        + params.channel_fwhm_us.map_or(0.0, |f| f.abs());
    let j_lo: isize = -((margin / dtau).ceil() as isize);
    let j_hi: isize = ((tau_max + margin) / dtau).ceil() as isize;

    let taus: Vec<f64> = (j_lo..=j_hi).map(|j| j as f64 * dtau).collect();
    let mut weights: Vec<f64> = taus.iter().map(|&t| ic_pulse(alpha, beta, r, t)).collect();

    if let Some(sigma) = params.burst_sigma_us
        && sigma.abs() > 0.0
    {
        weights = convolve_same(&weights, &gaussian_kernel(dtau, sigma.abs()));
    }
    if let Some(fwhm) = params.channel_fwhm_us
        && fwhm.abs() > 0.0
    {
        weights = convolve_same(&weights, &triangle_kernel(dtau, fwhm.abs()));
    }

    // Anchor the mode at offset 0.
    let peak_idx = argmax(&weights);
    let peak_val = weights[peak_idx].max(f64::MIN_POSITIVE);
    let tau_peak = taus[peak_idx];

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

    let offsets: Vec<f64> = (lo..=hi).map(|j| taus[j] - tau_peak).collect();
    let w: Vec<f64> = (lo..=hi).map(|j| weights[j] / peak_val).collect();
    (offsets, w)
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
/// ±4σ.
fn gaussian_kernel(dtau: f64, sigma: f64) -> Vec<f64> {
    let half = ((4.0 * sigma / dtau).ceil() as isize).max(1);
    let mut k: Vec<f64> = (-half..=half)
        .map(|j| {
            let t = j as f64 * dtau / sigma;
            (-0.5 * t * t).exp()
        })
        .collect();
    normalize_sum(&mut k);
    k
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
            beta: 4.0,
            r: EnergyLaw::Const(0.5),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        };
        let (offs, wts) = synth_kernel(&p, 600, 1.0);
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
        let (offsets, weights) = synth_kernel(&p, 600, 10.0);
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
        let (offsets, weights) = synth_kernel(&p, 600, 10.0);
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
            beta: 0.1,
            r: EnergyLaw::Const(0.0),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        };
        let support = |e: f64| {
            let (o, _) = synth_kernel(&p, 600, e);
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
    fn burst_and_channel_broaden_further() {
        // Folding in burst + channel widens the kernel TOF support.
        let base = IkedaCarpenterParams::constant(1.0, 0.1, 0.0);
        let (o0, _) = synth_kernel(&base, 600, 10.0);
        let support0 = o0.iter().cloned().fold(f64::MIN, f64::max)
            - o0.iter().cloned().fold(f64::MAX, f64::min);
        let folded = IkedaCarpenterParams {
            burst_sigma_us: Some(0.3),
            channel_fwhm_us: Some(0.35),
            ..IkedaCarpenterParams::constant(1.0, 0.1, 0.0)
        };
        let (o1, _) = synth_kernel(&folded, 600, 10.0);
        let support1 = o1.iter().cloned().fold(f64::MIN, f64::max)
            - o1.iter().cloned().fold(f64::MAX, f64::min);
        assert!(support1 > support0, "folded {support1} !> bare {support0}");
    }
}
