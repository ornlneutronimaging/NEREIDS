//! Resolution broadening via convolution with instrument resolution function.
//!
//! Convolves theoretical cross-sections (or transmission) with the instrument
//! resolution function to account for finite energy resolution. The resolution
//! function is modeled as a Gaussian with energy-dependent width, optionally
//! combined with an exponential tail, derived from time-of-flight instrument
//! parameters.
//!
//! ## SAMMY Reference
//! - `rsl/mrsl1.f90` — Main RSL resolution broadening routines (Resbrd)
//! - `rsl/mrsl4.f90` — Resolution width calculation (Wdsint, Rolowg)
//! - `rsl/mrsl5.f90` — Exponential tail peak shift (Shftge)
//! - `fnc/exerfc.f90` — Scaled complementary error function
//! - `convolution/DopplerAndResolutionBroadener.cpp` — Xcoef quadrature weights
//! - Manual Section 3.2 (Resolution Broadening), Eq. IV B 3.8
//!
//! ## Physics
//!
//! For a time-of-flight instrument, the energy resolution is:
//!
//!   (ΔE/E)² = (2·Δt/t)² + (2·ΔL/L)²
//!
//! where t = L/v is the neutron time-of-flight, Δt is the total timing
//! uncertainty, and ΔL is the flight path uncertainty. Since t ∝ 1/√E,
//! the timing contribution gives ΔE ∝ E^(3/2) while the path contribution
//! gives ΔE ∝ E.
//!
//! The broadened cross-section is:
//!
//!   σ_res(E) = ∫ R(E, E') · σ(E') dE'
//!
//! When Deltae = 0, R is a pure Gaussian (Iesopr=1):
//!   R(E, E') = exp(-(E-E')²/Wg²) / (Wg·√π)
//!
//! When Deltae > 0, R is the convolution of a Gaussian with an exponential
//! tail (Iesopr=3):
//!   R(E, E') ∝ exp(2·C·A + C²) · erfc(C + A)
//!
//! where C = Wg/(2·We), A = (E - E')/Wg, Wg = Gaussian width, We = exponential
//! width. This is the analytical result for convolving exp(-x²/Wg²) with
//! exp(-x/We)·H(x).

use nereids_core::constants::{DIVISION_FLOOR, NEAR_ZERO_FLOOR};
use std::fmt;
use std::sync::Arc;

/// TOF conversion factor: t[μs] = TOF_FACTOR × L[m] / √(E[eV]).
///
/// Derived from t = L / √(2E/m_n), converting to microseconds:
///   TOF_FACTOR = 1e6 / √(2 × EV_TO_JOULES / NEUTRON_MASS_KG)
///
/// Uses CODATA 2018 values (both exact in the 2019 SI).
const TOF_FACTOR: f64 = 72.298_254_398_292_8;

/// Errors from resolution broadening operations.
#[derive(Debug, PartialEq)]
pub enum ResolutionError {
    /// The energy grid is not sorted in ascending order.
    UnsortedEnergies,
    /// The energy grid and data arrays have mismatched lengths.
    LengthMismatch { energies: usize, data: usize },
    /// A [`ResolutionPlan`] was passed together with an `energies`
    /// slice that does not match the grid the plan was built for.
    ///
    /// Cheapest-available check hierarchy: length mismatch is caught
    /// first via [`Self::LengthMismatch`] (`plan.len() ==
    /// energies.len()` is necessary but not sufficient); a content
    /// mismatch fires `PlanGridMismatch` with the index of the first
    /// differing element so callers can diagnose silent-staleness
    /// bugs at the cache layer.
    PlanGridMismatch { first_diff_index: usize },
    /// A [`ResolutionMatrix`] was passed together with an `energies`
    /// slice that does not match the grid the matrix was compiled for.
    /// Same semantics as [`Self::PlanGridMismatch`] but for the CSR
    /// path (see [`apply_r`]).
    MatrixGridMismatch { first_diff_index: usize },
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsortedEnergies => write!(
                f,
                "energy grid must be sorted in non-descending order for binary search"
            ),
            Self::LengthMismatch { energies, data } => write!(
                f,
                "energy grid length ({}) must match data length ({})",
                energies, data
            ),
            Self::PlanGridMismatch { first_diff_index } => write!(
                f,
                "resolution plan was built for a different energy grid than was \
                 passed to apply_resolution_with_plan (first differing index: {})",
                first_diff_index,
            ),
            Self::MatrixGridMismatch { first_diff_index } => write!(
                f,
                "resolution matrix was compiled for a different energy grid than was \
                 passed to apply_resolution_with_matrix (first differing index: {})",
                first_diff_index,
            ),
        }
    }
}

impl std::error::Error for ResolutionError {}

/// Errors from `ResolutionParams` construction.
#[derive(Debug, PartialEq)]
pub enum ResolutionParamsError {
    /// Flight path must be positive and finite.
    InvalidFlightPath(f64),
    /// Timing uncertainty must be non-negative and finite.
    InvalidDeltaT(f64),
    /// Path length uncertainty must be non-negative and finite.
    InvalidDeltaL(f64),
    /// Exponential tail parameter must be non-negative and finite.
    InvalidDeltaE(f64),
}

impl fmt::Display for ResolutionParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFlightPath(v) => {
                write!(f, "flight_path_m must be positive and finite, got {v}")
            }
            Self::InvalidDeltaT(v) => {
                write!(f, "delta_t_us must be non-negative and finite, got {v}")
            }
            Self::InvalidDeltaL(v) => {
                write!(f, "delta_l_m must be non-negative and finite, got {v}")
            }
            Self::InvalidDeltaE(v) => {
                write!(f, "delta_e_us must be non-negative and finite, got {v}")
            }
        }
    }
}

impl std::error::Error for ResolutionParamsError {}

/// Resolution function parameters for time-of-flight instruments.
#[derive(Debug, Clone, Copy)]
pub struct ResolutionParams {
    /// Flight path length in meters (source to detector).
    flight_path_m: f64,
    /// Total timing uncertainty (1σ Gaussian) in microseconds.
    /// Combines moderator pulse width, detector timing, and electronics.
    delta_t_us: f64,
    /// Flight path uncertainty (1σ Gaussian) in meters.
    delta_l_m: f64,
    /// Exponential tail parameter (SAMMY Deltae, raw SAMMY units).
    ///
    /// When zero, pure Gaussian broadening is used (SAMMY Iesopr=1).
    /// When positive, the kernel is the convolution of a Gaussian with an
    /// exponential tail (SAMMY Iesopr=3).
    ///
    /// SAMMY Ref: `RslResolutionFunction_M.f90` getCo2, `rsl/mrsl4.f90` Wdsint.
    delta_e_us: f64,
}

impl ResolutionParams {
    /// Create validated resolution parameters.
    ///
    /// # Arguments
    /// * `flight_path_m` — Flight path length in meters (must be > 0).
    /// * `delta_t_us` — Timing uncertainty in microseconds (must be >= 0).
    /// * `delta_l_m` — Flight path uncertainty in meters (must be >= 0).
    /// * `delta_e_us` — Exponential tail parameter in SAMMY Deltae units
    ///   (must be >= 0). When 0, pure Gaussian broadening is used.
    ///
    /// # Errors
    /// Returns `ResolutionParamsError::InvalidFlightPath` if `flight_path_m <= 0.0`
    /// or is not finite.
    /// Returns `ResolutionParamsError::InvalidDeltaT` if `delta_t_us < 0.0` or is
    /// not finite.
    /// Returns `ResolutionParamsError::InvalidDeltaL` if `delta_l_m < 0.0` or is
    /// not finite.
    /// Returns `ResolutionParamsError::InvalidDeltaE` if `delta_e_us < 0.0` or is
    /// not finite.
    pub fn new(
        flight_path_m: f64,
        delta_t_us: f64,
        delta_l_m: f64,
        delta_e_us: f64,
    ) -> Result<Self, ResolutionParamsError> {
        if !flight_path_m.is_finite() || flight_path_m <= 0.0 {
            return Err(ResolutionParamsError::InvalidFlightPath(flight_path_m));
        }
        if !delta_t_us.is_finite() || delta_t_us < 0.0 {
            return Err(ResolutionParamsError::InvalidDeltaT(delta_t_us));
        }
        if !delta_l_m.is_finite() || delta_l_m < 0.0 {
            return Err(ResolutionParamsError::InvalidDeltaL(delta_l_m));
        }
        if !delta_e_us.is_finite() || delta_e_us < 0.0 {
            return Err(ResolutionParamsError::InvalidDeltaE(delta_e_us));
        }
        Ok(Self {
            flight_path_m,
            delta_t_us,
            delta_l_m,
            delta_e_us,
        })
    }

    /// Returns the flight path length in meters.
    #[must_use]
    pub fn flight_path_m(&self) -> f64 {
        self.flight_path_m
    }

    /// Total timing uncertainty (1σ Gaussian) in microseconds.
    ///
    /// The factor of 2 in [`gaussian_width()`](Self::gaussian_width) comes from
    /// the energy-TOF derivative dE/E = 2·dt/t, not from a σ-to-FWHM conversion.
    #[must_use]
    pub fn delta_t_us(&self) -> f64 {
        self.delta_t_us
    }

    /// Returns the flight path uncertainty (1σ Gaussian) in meters.
    #[must_use]
    pub fn delta_l_m(&self) -> f64 {
        self.delta_l_m
    }

    /// Returns the exponential tail parameter (SAMMY Deltae units).
    #[must_use]
    pub fn delta_e_us(&self) -> f64 {
        self.delta_e_us
    }

    /// Whether the exponential tail is active (Deltae > 0, SAMMY Iesopr=3).
    #[must_use]
    pub fn has_exponential_tail(&self) -> bool {
        self.delta_e_us > NEAR_ZERO_FLOOR
    }

    /// Exponential tail width Widexp(E) in eV.
    ///
    /// SAMMY Ref: `rsl/mrsl4.f90` Wdsint lines 55-56 (Kedxfw=false path):
    ///   `Widexp = E * Co2 * sqrt(E)` where `Co2 = 2·Deltae / (Sm2·Dist)`.
    ///
    /// Combined: `Widexp = 2·Deltae·E^(3/2) / (TOF_FACTOR·L)`.
    #[must_use]
    pub fn exp_width(&self, energy_ev: f64) -> f64 {
        if energy_ev <= 0.0 || self.delta_e_us <= 0.0 {
            return 0.0;
        }
        2.0 * self.delta_e_us * energy_ev.powf(1.5) / (TOF_FACTOR * self.flight_path_m)
    }

    /// Gaussian resolution width σ_E(E) in eV.
    ///
    /// Combines timing and flight-path contributions in quadrature:
    ///   σ_E² = (2·Δt/t × E)² + (2·ΔL/L × E)²
    ///
    /// where t = TOF_FACTOR × L / √E is the time-of-flight in μs.
    #[must_use]
    pub fn gaussian_width(&self, energy_ev: f64) -> f64 {
        if energy_ev <= 0.0 || self.flight_path_m <= 0.0 {
            return 0.0;
        }

        // Timing contribution: σ_t = 2 × Δt × E^(3/2) / (TOF_FACTOR × L)
        let timing =
            2.0 * self.delta_t_us * energy_ev.powf(1.5) / (TOF_FACTOR * self.flight_path_m);

        // Path length contribution: σ_L = 2 × ΔL × E / L
        let path = 2.0 * self.delta_l_m * energy_ev / self.flight_path_m;

        (timing * timing + path * path).sqrt()
    }

    /// FWHM of the resolution function at energy E, in eV.
    #[must_use]
    pub fn fwhm(&self, energy_ev: f64) -> f64 {
        2.0 * (2.0_f64.ln()).sqrt() * self.gaussian_width(energy_ev)
    }
}

/// Apply Gaussian resolution broadening to cross-section data.
///
/// Convolves the input cross-sections with a Gaussian kernel whose width
/// varies with energy according to the instrument resolution function.
///
/// # Arguments
/// * `energies` — Energy grid in eV (must be sorted ascending).
/// * `cross_sections` — Cross-sections in barns at each energy point.
/// * `params` — Resolution function parameters.
///
/// # Returns
/// Resolution-broadened cross-sections on the same energy grid.
///
/// # Errors
/// Returns [`ResolutionError::LengthMismatch`] if the arrays differ in length,
/// or [`ResolutionError::UnsortedEnergies`] if the energy grid is not sorted
/// in non-descending order.
pub fn resolution_broaden(
    energies: &[f64],
    cross_sections: &[f64],
    params: &ResolutionParams,
) -> Result<Vec<f64>, ResolutionError> {
    validate_inputs(energies, cross_sections)?;
    Ok(resolution_broaden_presorted(
        energies,
        cross_sections,
        params,
    ))
}

/// Check that the energy grid is sorted and that its length matches the data.
fn validate_inputs(energies: &[f64], data: &[f64]) -> Result<(), ResolutionError> {
    if energies.len() != data.len() {
        return Err(ResolutionError::LengthMismatch {
            energies: energies.len(),
            data: data.len(),
        });
    }
    if !energies.windows(2).all(|w| w[0] <= w[1]) {
        return Err(ResolutionError::UnsortedEnergies);
    }
    Ok(())
}

// ─── Xcoef quadrature weights ──────────────────────────────────────────────────

/// Compute SAMMY's 4-point quadrature weights for a non-uniform energy grid.
///
/// Replaces the simple trapezoidal rule `de = (E[j+1] - E[j-1]) / 2` with
/// SAMMY's higher-order scheme from Eq. IV B 3.8 (page 80 of SAMMY manual R3).
///
/// SAMMY Ref: `convolution/DopplerAndResolutionBroadener.cpp`, `setXcoefWeights()`.
///
/// The weights include a correction term x2(k) that accounts for non-uniform
/// grid spacing, providing 4th-order accuracy on smooth grids.
///
/// Note: the returned weights are 12x the quantity in Eq. IV B 3.8. This
/// constant factor cancels during normalization (sum/norm), so the broadened
/// result is independent of the scaling.
fn compute_xcoef_weights(energies: &[f64]) -> Vec<f64> {
    let n = energies.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    // SAMMY's 4-point quadrature weights (Eq. IV B 3.8, SAMMY Manual R3 p80).
    //
    // Uses a sliding window of 5 consecutive energies E[0..4] to compute
    // coefficients A[0..5] at each grid point k:
    //
    //   A[0] = v1  (k >= 2)
    //   A[1] = 5·v2  (k >= 1)
    //   A[2] = 5·v3  (k < n-1)
    //   A[3] = v4  (k < n-2)
    //   A[4] = (v3² - v1²)/v2   curvature correction  (k >= 2)
    //   A[5] = -(v4² - v2²)/v3  curvature correction  (k >= 1)
    //
    // where v1..v4 are consecutive grid spacings around point k.
    //
    // The result is 12× Eq. IV B 3.8; this constant factor cancels during
    // normalization (sum/norm) in the broadening loop.
    //
    // SAMMY Ref: `convolution/DopplerAndResolutionBroadener.cpp` lines 365-457
    let mut weights = vec![0.0f64; n];

    // Sliding window: e[j] holds energies relative to current k.
    // At loop start for k: e[0]=E[k-2], e[1]=E[k-1], e[2]=E[k],
    //                       e[3]=E[k+1], e[4]=E[k+2]
    // Out-of-bounds positions are 0.0 (matching SAMMY's convention).
    let mut e = [0.0f64; 5];
    e[3] = energies[0];
    if n > 1 {
        e[4] = energies[1];
    }

    for k in 0..n {
        // Shift window left.
        e[0] = e[1];
        e[1] = e[2];
        e[2] = e[3];
        e[3] = e[4];
        e[4] = if k + 2 < n { energies[k + 2] } else { 0.0 };

        let v1 = e[1] - e[0];
        let v2 = e[2] - e[1];
        let v3 = e[3] - e[2];
        let v4 = e[4] - e[3];

        let mut a = [0.0f64; 6];

        if k >= 2 {
            a[0] = v1;
            // Curvature correction: x2(k-2) = (v3² - v1²) / v2
            if v2.abs() > NEAR_ZERO_FLOOR {
                a[4] = (v3 * v3 - v1 * v1) / v2;
            }
        }
        if k >= 1 {
            a[1] = 5.0 * v2;
            // Curvature correction: -x2(k-1) = -(v4² - v2²) / v3
            if v3.abs() > NEAR_ZERO_FLOOR {
                a[5] = -(v4 * v4 - v2 * v2) / v3;
            }
        }
        if k != n - 1 {
            a[2] = 5.0 * v3;
        }
        if k < n.saturating_sub(2) {
            a[3] = v4;
        }

        // Boundary overrides (SAMMY source lines 446-450).
        if k == n.saturating_sub(2) {
            a[5] = 0.0;
        }
        if k == n - 1 {
            a[4] = 0.0;
            a[5] = 0.0;
        }

        weights[k] = a.iter().sum::<f64>();
    }

    weights
}

/// Compute erfc(x) using the existing `exerfc` function.
///
/// erfc(x) = exp(-x²) · exerfc(x) / √π
///
/// For x < 0: erfc(-|x|) = 2 - erfc(|x|)
fn erfc_from_exerfc(x: f64) -> f64 {
    const SQRT_PI: f64 = 1.772_453_850_905_516;
    if x >= 0.0 {
        (-x * x).exp() * exerfc(x) / SQRT_PI
    } else {
        let xp = -x;
        2.0 - (-xp * xp).exp() * exerfc(xp) / SQRT_PI
    }
}

// ─── Scaled complementary error function ───────────────────────────────────────

/// Compute exp(x²)·erfc(x)·√π, numerically stable for all x.
///
/// SAMMY Ref: `fnc/exerfc.f90`.
///
/// Uses rational approximation for |x| < 5.01 and asymptotic expansion
/// (Abramowitz & Stegun 7.1.23) for |x| >= 5.01.
pub(crate) fn exerfc(x: f64) -> f64 {
    const SQRT_PI: f64 = 1.772_453_850_905_516;
    const TWO_SQRT_PI: f64 = 3.544_907_701_811_032;
    const XMAX: f64 = 5.01;
    // Rational approximation coefficients (from SAMMY's exerfc.f90)
    const A1: f64 = 8.584_076_57e-1;
    const A2: f64 = 3.078_181_93e-1;
    const A3: f64 = 6.383_238_91e-2;
    const A4: f64 = 1.824_050_75e-4;
    const A5: f64 = 6.509_742_65e-1;
    const A6: f64 = 2.294_848_19e-1;
    const A7: f64 = 3.403_018_23e-2;

    if x < 0.0 {
        let xp = -x;
        if xp > XMAX {
            TWO_SQRT_PI - asympt(xp)
        } else {
            let a =
                (A1 + xp * (A2 + xp * (A3 - xp * A4))) / (1.0 + xp * (A5 + xp * (A6 + xp * A7)));
            let b = SQRT_PI + xp * (2.0 - a);
            let a_rat = b / (xp * b + 1.0);
            TWO_SQRT_PI * (x * x).exp() - a_rat
        }
    } else if x > XMAX {
        asympt(x)
    } else if x > 0.0 {
        let a = (A1 + x * (A2 + x * (A3 - x * A4))) / (1.0 + x * (A5 + x * (A6 + x * A7)));
        let b = SQRT_PI + x * (2.0 - a);
        b / (x * b + 1.0)
    } else {
        SQRT_PI
    }
}

/// Asymptotic expansion of exp(x²)·erfc(x)·√π for large positive x.
///
/// SAMMY Ref: `fnc/exerfc.f90`, Asympt function.
/// Uses Abramowitz & Stegun 7.1.23.
fn asympt(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let e = 1.0 / x;
    if e == 0.0 {
        return 0.0;
    }
    let b = 1.0 / (x * x);
    let mut a = 1.0;
    let mut c = b * 0.5;
    for n in 1..=40 {
        a -= c;
        c *= -(n as f64 + 0.5) * b;
        if (a - c) == a || (c / a).abs() < 1e-8 {
            break;
        }
    }
    a * e
}

/// Compute the Gaussian+exponential combined kernel weight Z(A, B).
///
/// Returns √π · exp(-A² + B²) · erfc(B), computed via exerfc for stability.
///
/// SAMMY Ref: `rsl/mrsl1.f90` lines 467-484 (Resbrd, Iesopr=3 path).
///
/// When B >= 0: `Z = exp(-A²) · Exerfc(B)`
/// When B < 0:  `Z = Xxerfc(B, A)` which is the same mathematical function
///   computed with different numerical strategy for stability.
fn gauss_exp_kernel(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        let exp_neg_a2 = (-a * a).exp();
        if exp_neg_a2 == 0.0 {
            return 0.0;
        }
        exp_neg_a2 * exerfc(b)
    } else {
        // Xxerfc(B, A): compute exp(-A² + B²) · erfc(-B) · √π
        // Using the same rational approximation as exerfc but for negative B.
        //
        // SAMMY Ref: `fnc/xxerfc.f90`.
        xxerfc(b, a)
    }
}

/// Compute exp(-xxx² + xx²) · erfc(-xx) · √π for xx assumed negative (B < 0).
///
/// SAMMY Ref: `fnc/xxerfc.f90`. Note: SAMMY says "Xx is assumed positive"
/// but the caller passes B < 0 as Xx. The code handles this by immediately
/// computing X = -Xx (which is positive).
///
/// When x = -xx exceeds XMAX, the rational approximation loses accuracy.
/// We switch to `exp(-xxx²) · asympt(x)`, mirroring exerfc's large-argument
/// path.
fn xxerfc(xx: f64, xxx: f64) -> f64 {
    const SQRT_PI: f64 = 1.772_453_850_905_516;
    const XMAX: f64 = 5.01;
    const A1: f64 = 8.584_076_57e-1;
    const A2: f64 = 3.078_181_93e-1;
    const A3: f64 = 6.383_238_91e-2;
    const A4: f64 = 1.824_050_75e-4;
    const A5: f64 = 6.509_742_65e-1;
    const A6: f64 = 2.294_848_19e-1;
    const A7: f64 = 3.403_018_23e-2;

    let x = -xx; // x is positive (xx is B < 0)

    // For large x, the rational approximation loses accuracy.
    // exp(-xxx² + x²)·erfc(x)·√π = exp(-xxx²)·[exp(x²)·erfc(x)·√π]
    //                              = exp(-xxx²)·asympt(x)
    if x > XMAX {
        return (-xxx * xxx).exp() * asympt(x);
    }

    let a_rat = (A1 + x * (A2 + x * (A3 - x * A4))) / (1.0 + x * (A5 + x * (A6 + x * A7)));
    let b_int = SQRT_PI + x * (2.0 - a_rat);
    let a_final = b_int / (x * b_int + 1.0);
    // exp(-xxx² + x²) = exp(-A² + B²) since x = -B, xx = B
    let exp_term = (-xxx * xxx + x * x).exp();
    SQRT_PI * 2.0 * exp_term - a_final * (-xxx * xxx).exp()
}

/// Compute the energy shift for the Gaussian+exponential kernel peak.
///
/// Finds the peak of the combined kernel relative to E=0 via Newton-Raphson
/// iteration. This centers the convolution window on the kernel maximum.
///
/// SAMMY Ref: `rsl/mrsl5.f90`, Shftge function.
///
/// # Arguments
/// * `c` — Mixing parameter: Widgau / (2·Widexp)
/// * `widgau` — Gaussian resolution width (eV)
///
/// # Returns
/// The energy shift Est (eV) to apply to the measurement energy.
fn shftge(c: f64, widgau: f64) -> f64 {
    const ONE_OVER_SQRT_PI: f64 = 0.564_189_583_547_756_3;
    const SMALL: f64 = 0.01;

    let ax = c;
    let bx = widgau;

    // Initial guess
    let mut x0 = if ax > ONE_OVER_SQRT_PI { ax } else { 0.0 };

    let f0_initial = ax * exerfc(x0) - 1.0;
    let mut f0 = f0_initial;
    let fff = f0;

    for _iter in 0..100 {
        let f = ax * exerfc(x0) - 1.0;
        let xma = x0 - ax;
        let q = 1.0 - 2.0 * x0 * xma;
        let delx = if q.abs() < NEAR_ZERO_FLOOR {
            // q ≈ 0: division would overflow; accept current estimate.
            break;
        } else if xma * xma - q * f > 0.0 {
            let disc = (xma * xma - q * f).sqrt();
            if xma > 0.0 {
                (-xma + disc) / q
            } else {
                (-xma - disc) / q
            }
        } else {
            if xma.abs() < NEAR_ZERO_FLOOR {
                break;
            }
            -f * 0.5 / xma
        };
        let x1 = x0 + delx;
        let shftg = (ax - x1) * bx;
        if (x1 - x0).abs() / x1.abs().max(1.0) < SMALL
            && fff.abs() > NEAR_ZERO_FLOOR
            && (f - f0).abs() / fff.abs() < SMALL
        {
            return shftg;
        }
        f0 = f;
        x0 = x1;
    }

    (ax - x0) * bx
}

/// Threshold for the ratio C = W_g / (2·W_e) above which the exponential
/// tail is negligible and the pure Gaussian PW-linear path is used instead.
///
/// At C = 2.5, erfc(2.5) ≈ 0.0005, so the exp tail contributes <0.05% of the
/// kernel integral.  Using the pure Gaussian path at this threshold introduces
/// negligible systematic error while enabling the more accurate PW-linear
/// integration and adaptive intermediate point insertion.
const EXP_TAIL_NEGLIGIBLE_C: f64 = 2.5;

/// Resolution broadening assuming the energy grid is already validated
/// (sorted ascending, same length as cross_sections).
///
/// For each broadening energy, selects the optimal integration method:
/// - **PW-linear Gaussian** (exact, second-order): when `delta_e == 0` or
///   the ratio C = W_g/(2·W_e) > [`EXP_TAIL_NEGLIGIBLE_C`] (exp tail negligible).
/// - **Combined Gaussian+exp kernel** with SAMMY Xcoef quadrature: when the
///   exponential tail is significant (C ≤ threshold).
///
/// SAMMY Ref: `rsl/mrsl1.f90` Resbrd, `convolution/DopplerAndResolutionBroadener.cpp`
pub(crate) fn resolution_broaden_presorted(
    energies: &[f64],
    cross_sections: &[f64],
    params: &ResolutionParams,
) -> Vec<f64> {
    let n = energies.len();
    if n == 0 {
        return vec![];
    }

    // Precompute Xcoef weights (used only by the combined kernel path).
    // Even if some energies take the PW-linear path, we compute weights for
    // the full grid — cheaper than branching per-energy.
    let xcoef = if params.has_exponential_tail() {
        compute_xcoef_weights(energies)
    } else {
        vec![]
    };
    let n_sigma = 5.0; // Integrate out to 5σ for Gaussian
    let mut broadened = vec![0.0f64; n];

    for i in 0..n {
        let e = energies[i];
        let widgau = params.gaussian_width(e);

        if widgau < NEAR_ZERO_FLOOR {
            broadened[i] = cross_sections[i];
            continue;
        }

        // Per-energy decision: use combined kernel only when the exp tail
        // is significant at THIS energy.
        let widexp = params.exp_width(e);
        let use_combined =
            widexp > NEAR_ZERO_FLOOR && widgau / (2.0 * widexp) <= EXP_TAIL_NEGLIGIBLE_C;

        // Compute integration limits.
        let (e_low, e_high) = if use_combined {
            // SAMMY Ref: mrsl4.f90 lines 57-65
            let wlow = n_sigma * widgau;
            let rwid = widgau / widexp;
            let wup = if rwid <= 1.0 {
                6.25 * widexp
            } else if rwid <= 2.0 {
                n_sigma * (3.0 - rwid) * widgau
            } else {
                n_sigma * widgau
            };
            (e - wlow, e + wup)
        } else {
            (e - n_sigma * widgau, e + n_sigma * widgau)
        };

        let j_lo = energies.partition_point(|&ej| ej < e_low);
        let j_hi = energies.partition_point(|&ej| ej <= e_high);

        if j_hi.saturating_sub(j_lo) <= 1 {
            broadened[i] = cross_sections[i];
            continue;
        }

        let mut sum = 0.0;
        let mut norm = 0.0;

        if use_combined {
            // Combined Gaussian + exponential kernel (SAMMY Iesopr=3)
            // with 4-point Xcoef quadrature weights.
            // SAMMY Ref: mrsl1.f90 lines 455-484
            let c = widgau * 0.5 / widexp;
            let est = shftge(c, widgau);
            let y = c * widgau + e - est;

            for j in j_lo..j_hi {
                let ee = energies[j];
                let a = (e - est - ee) / widgau;
                let b = (y - ee) / widgau;
                let z = gauss_exp_kernel(a, b);
                let wt = xcoef[j] * z;
                sum += wt * cross_sections[j];
                norm += wt;
            }
        } else {
            // Pure Gaussian kernel with piecewise-linear exact integration.
            //
            // For each interval [E_j, E_{j+1}], integrate G(E_i - E') × σ_linear(E')
            // exactly, where G(x) = exp(-x²/W²) / (W√π).
            //
            // Substituting u = (E' - E_i)/W, dE' = W du:
            //   ∫ G × [σ_j + slope×(E'-E_j)] dE'
            //   = (1/√π) ∫ exp(-u²) [σ_j + slope×W×(u - a_j)] du
            //
            // With I₀ = erf(a_{j+1}) - erf(a_j) and
            //      I₁ = (exp(-a_j²) - exp(-a_{j+1}²)) / 2:
            //
            // The normalization integral is I₀/2, so after sum/norm (2 cancels):
            //   sum += σ_j × I₀ + slope × W × (2/√π × I₁ - a_j × I₀)
            //   norm += I₀
            //
            // The factor 2/√π on I₁ comes from the u·exp(-u²) integral
            // needing to match the normalization convention erf(x) = 2/√π ∫ exp(-t²) dt.
            const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;
            let inv_w = 1.0 / widgau;
            for j in j_lo..j_hi.saturating_sub(1) {
                let e_j = energies[j];
                let e_j1 = energies[j + 1];
                let h = e_j1 - e_j;
                if h < NEAR_ZERO_FLOOR {
                    continue;
                }

                let a_j = (e_j - e) * inv_w;
                let a_j1 = (e_j1 - e) * inv_w;

                // I₀ = erf(a_{j+1}) - erf(a_j) = erfc(a_j) - erfc(a_{j+1})
                let erfc_aj = erfc_from_exerfc(a_j);
                let erfc_aj1 = erfc_from_exerfc(a_j1);
                let i0 = erfc_aj - erfc_aj1;

                if i0 < NEAR_ZERO_FLOOR {
                    continue;
                }

                // I₁ = (exp(-a_j²) - exp(-a_{j+1}²)) / 2
                let i1 = ((-a_j * a_j).exp() - (-a_j1 * a_j1).exp()) * 0.5;

                let slope = (cross_sections[j + 1] - cross_sections[j]) / h;

                // σ_j × I₀ + slope × W × (2/√π × I₁ - a_j × I₀)
                sum += cross_sections[j] * i0 + slope * widgau * (TWO_OVER_SQRT_PI * i1 - a_j * i0);
                norm += i0;
            }
        }

        if norm > DIVISION_FLOOR {
            broadened[i] = sum / norm;
        } else {
            broadened[i] = cross_sections[i];
        }
    }

    broadened
}

/// Apply resolution broadening to transmission data.
///
/// This is the same Gaussian convolution but applied to transmission
/// spectra rather than cross-sections. The distinction matters because
/// resolution broadening of transmission is physically different from
/// broadening cross-sections (Beer-Lambert law is nonlinear).
///
/// # Arguments
/// * `energies` — Energy grid in eV (sorted ascending).
/// * `transmission` — Transmission values (0 to 1) at each energy point.
/// * `params` — Resolution function parameters.
///
/// # Returns
/// Resolution-broadened transmission on the same energy grid.
///
/// # Errors
/// Returns [`ResolutionError`] if the energy grid is unsorted or array
/// lengths do not match.
pub fn resolution_broaden_transmission(
    energies: &[f64],
    transmission: &[f64],
    params: &ResolutionParams,
) -> Result<Vec<f64>, ResolutionError> {
    // The convolution kernel is the same; only the interpretation differs.
    // Validation is handled by resolution_broaden.
    resolution_broaden(energies, transmission, params)
}

/// A tabulated resolution function from Monte Carlo instrument simulation.
///
/// Contains reference kernels R(Δt; E_ref) at discrete energies, stored in
/// TOF-offset space (μs). Kernels are interpolated between reference energies
/// and converted from TOF to energy space when applied.
///
/// ## File Format (VENUS/FTS)
///
/// ```text
/// FTS BL10 case i00dd folded triang FWHM 350 ns PSR   ← header
/// -----                                                 ← separator
///    5.00000e-004   0.00000e+000                        ← energy block start
/// -53.458917835671329 2.051764258257523e-04             ← (tof_offset_μs, weight)
/// ...
///                                                       ← blank line separates blocks
///    1.00000e-003   0.00000e+000                        ← next energy block
/// ...
/// ```
#[derive(Debug, Clone)]
pub struct TabulatedResolution {
    /// Reference energies (eV), sorted ascending.
    ref_energies: Vec<f64>,
    /// For each reference energy: (tof_offsets_μs, weights) pairs.
    /// Weights are peak-normalized (max=1.0).
    kernels: Vec<(Vec<f64>, Vec<f64>)>,
    /// Flight path length in meters (needed for TOF↔energy conversion).
    flight_path_m: f64,
}

impl TabulatedResolution {
    /// Reference energies (eV), sorted ascending.
    pub fn ref_energies(&self) -> &[f64] {
        &self.ref_energies
    }

    /// For each reference energy: (tof_offsets_μs, weights) pairs.
    /// Weights are peak-normalized (max=1.0).
    pub fn kernels(&self) -> &[(Vec<f64>, Vec<f64>)] {
        &self.kernels
    }

    /// Flight path length in meters (needed for TOF↔energy conversion).
    pub fn flight_path_m(&self) -> f64 {
        self.flight_path_m
    }
}

/// Resolution function: either analytical Gaussian or tabulated from Monte Carlo.
///
/// The `Tabulated` variant wraps an `Arc` so that cloning (e.g., per-pixel in
/// spatial mapping) is a cheap reference-count bump rather than a deep copy.
#[derive(Debug, Clone)]
pub enum ResolutionFunction {
    /// Analytical Gaussian resolution from instrument parameters.
    Gaussian(ResolutionParams),
    /// Tabulated resolution from Monte Carlo instrument simulation.
    Tabulated(Arc<TabulatedResolution>),
}

/// Pre-built resolution-broadening plan for a specific target energy grid.
///
/// Encodes every quantity that depends only on the target grid, the
/// reference kernel, and the flight path — so applying the plan to a
/// spectrum reduces to a gather + multiply-add loop with no
/// transcendentals, no allocations, and no binary / pointer search.
///
/// Build via [`TabulatedResolution::plan`] — returns a `Result` and
/// validates the sorted-grid precondition that `broaden` enforces.
/// Apply via [`ResolutionPlan::apply`].  One plan is tied to one
/// `(target_energies, ref_energies, flight_path_m)` triple; the plan
/// owns a copy of the target-energy grid so callers cannot apply it to
/// a spectrum that was measured on a *different* grid even when the
/// grid length matches — use [`Self::target_energies`] to verify the
/// grid identity before applying.
///
/// The layout is a flat Struct-of-Arrays (SoA): per-target `(lo_idx,
/// frac, weight)` tuples packed into three parallel `Vec`s, with
/// `starts[i]..starts[i+1]` naming the range for target `i`.  SoA keeps
/// the inner loop memory-access pattern sequential and cache-friendly.
#[derive(Debug, Clone)]
pub struct ResolutionPlan {
    /// Target energy grid the plan was built for (owned copy).
    ///
    /// Stored so `apply()` can verify `spectrum.len() == self.len()`
    /// and expose a cheap grid identity for caller-side caching.
    /// ~28 KB for the VENUS 3471-point grid — negligible compared to
    /// the ~8 MB `lo_idx`/`frac`/`weight` footprint of a full plan.
    target_energies: Vec<f64>,
    /// `starts[i]..starts[i+1]` indexes into `lo_idx`/`frac`/`weight`
    /// for target `i`.  `starts` has length `target_energies.len() + 1`.
    starts: Vec<u32>,
    /// For each valid (target, kernel-point) entry: the lower bracket
    /// index into the target grid (spectrum[lo] + frac * (spectrum[lo+1]
    /// - spectrum[lo])).
    lo_idx: Vec<u32>,
    /// Spectrum-interp fraction in [0, 1].  Set to 0 for degenerate
    /// brackets; the apply-time loop short-circuits `frac == 0.0` so
    /// degenerate entries never touch `spectrum[lo+1]`.  This matches
    /// `broaden_presorted` even when `spectrum[lo+1]` is NaN/±∞.
    frac: Vec<f64>,
    /// Pre-computed per-entry weight (`w * dt_width.abs()`).  Summing
    /// these yields the per-target normalisation.
    weight: Vec<f64>,
    /// Pre-summed `Σ weight` per target (in the same accumulation order
    /// as `broaden_presorted` visits the valid entries).  When `norm <=
    /// DIVISION_FLOOR` the apply path returns `spectrum[i]` directly
    /// — the exact `broaden_presorted` passthrough behaviour.
    norm: Vec<f64>,
}

impl ResolutionPlan {
    /// Number of target energies this plan covers.
    pub fn len(&self) -> usize {
        self.target_energies.len()
    }

    /// True when the plan covers no target energies.
    pub fn is_empty(&self) -> bool {
        self.target_energies.is_empty()
    }

    /// Total number of (target, kernel-point) entries retained across
    /// all target energies.  Exposed for diagnostics — skipped entries
    /// (w ≤ 0, tof_prime ≤ 0, `e_prime` out of range) are not counted.
    pub fn n_entries(&self) -> usize {
        self.weight.len()
    }

    /// Target energy grid the plan was built for.
    ///
    /// Callers implementing plan caches can compare this against their
    /// current grid to decide whether the plan is still valid.  Using
    /// pointer identity of the returned slice gives an O(1) check when
    /// the grid hasn't moved; slice equality is `O(n)` but catches
    /// cases where the underlying buffer was reallocated.
    pub fn target_energies(&self) -> &[f64] {
        &self.target_energies
    }

    /// Apply the plan to a spectrum on the same target grid the plan
    /// was built for.
    ///
    /// The spectrum length must equal [`Self::len`].  Passing a
    /// spectrum on a different grid that happens to have the same
    /// length is caller error — verify via [`Self::target_energies`]
    /// when in doubt.
    ///
    /// Bit-exact with `broaden_presorted(target_energies, spectrum)`
    /// for finite spectrum values; degenerate-bracket entries
    /// short-circuit the interpolation so the equivalence also holds
    /// when `spectrum[lo+1]` is NaN or ±∞ (the reference path returns
    /// `spectrum[lo]` directly in that case without touching the upper
    /// bracket).
    pub fn apply(&self, spectrum: &[f64]) -> Vec<f64> {
        let n = self.target_energies.len();
        assert_eq!(
            spectrum.len(),
            n,
            "spectrum length ({}) must match plan target-grid length ({})",
            spectrum.len(),
            n,
        );
        if n == 0 {
            return Vec::new();
        }

        let mut result = vec![0.0f64; n];

        // Pre-bind plan slices once per call and pre-slice each
        // target's entry range before the hot loop.  This is a
        // bounds-check-elimination (BCE) refactor — every per-entry
        // index is proven in-bounds by the invariants established in
        // `plan_presorted`, so the inner loop uses `get_unchecked`
        // with SAFETY comments citing those invariants.  The compiler
        // then auto-vectorizes the inner compute where profitable.
        //
        // We deliberately do NOT use explicit 2-wide SIMD here — an
        // experiment via the `wide` crate (commit abandoned;
        // `perf-lessons.md`) showed that 2-wide f64x2 with gather
        // emulation is net-negative on AArch64 Neon vs the compiler's
        // scalar auto-vectorization of the BCE'd inner loop.  On
        // wider targets (x86 AVX2 / AVX-512) a SIMD rewrite could
        // still pay off but is out of scope here.
        //
        // Control flow, accumulation order, and the `frac == 0.0`
        // NaN-safety short-circuit are all preserved exactly so the
        // bit-exact contract with `broaden_presorted` holds for
        // finite AND pathological (NaN, ±∞) spectra.
        let lo_idx = self.lo_idx.as_slice();
        let frac_all = self.frac.as_slice();
        let weight_all = self.weight.as_slice();
        let starts = self.starts.as_slice();
        let norm = self.norm.as_slice();
        let spec = spectrum;

        // Defence-in-depth: debug-only invariant checks right after
        // slice binding, so a future change to `plan_presorted` that
        // silently violates the `unsafe { get_unchecked }` SAFETY
        // claims below fails loudly in debug builds.  Zero release-
        // build cost.  Copilot review finding on PR #470.
        debug_assert_eq!(starts.len(), n + 1);
        debug_assert_eq!(
            starts.last().copied(),
            Some(lo_idx.len() as u32),
            "plan_presorted invariant: starts.last() must equal lo_idx.len()",
        );
        debug_assert_eq!(lo_idx.len(), frac_all.len());
        debug_assert_eq!(lo_idx.len(), weight_all.len());
        debug_assert_eq!(norm.len(), n);
        debug_assert_eq!(spec.len(), n);

        for i in 0..n {
            let norm_i = norm[i];
            if norm_i <= DIVISION_FLOOR {
                // Passthrough — matches `broaden_presorted`'s
                // `spectrum[i]` fallback for e ≤ 0, empty kernel, or
                // degenerate norm accumulation.
                result[i] = spec[i];
                continue;
            }
            let start = starts[i] as usize;
            let end = starts[i + 1] as usize;
            // Zip-compatible pre-bound slices of exactly `end - start`
            // elements each — the per-j bounds check is elided by the
            // compiler because the slice length bounds the loop.
            let los = &lo_idx[start..end];
            let fracs = &frac_all[start..end];
            let ws = &weight_all[start..end];

            let mut sum = 0.0f64;
            for k in 0..los.len() {
                // SAFETY: `k < los.len()` is guaranteed by the range;
                // `los`, `fracs`, and `ws` all have length `end - start`
                // (same subslice bounds), so each `get_unchecked(k)`
                // read is in-bounds.
                let lo = unsafe { *los.get_unchecked(k) } as usize;
                let frac = unsafe { *fracs.get_unchecked(k) };
                let w = unsafe { *ws.get_unchecked(k) };

                // Degenerate-bracket short-circuit: when the plan
                // built `frac = -0.0` (span < NEAR_ZERO_FLOOR) we skip
                // `spectrum[lo+1]` entirely.  Without this branch,
                // `0.0 * NaN = NaN` would propagate and diverge from
                // the reference `broaden_presorted`, which returns
                // `spectrum[lo]` directly for that case.  Branch is
                // well-predicted (degenerate brackets are rare on
                // real grids) and preserves bit-exactness under
                // pathological spectra.
                //
                // The check MUST use `to_bits()` because the non-
                // degenerate path can legitimately produce
                // `frac == +0.0` when `e_prime == energies[lo]`
                // exactly.  In that case `broaden_presorted` still
                // reads `spectrum[lo+1]` (and propagates NaN if
                // present there), so the short-circuit MUST NOT
                // trigger.  `+0.0 == -0.0` returns `true` but
                // `(+0.0).to_bits() != (-0.0).to_bits()`, so the
                // bit-pattern check disambiguates exactly which
                // semantic `plan_presorted` meant.  Copilot review
                // finding on PR #470.
                let s = if frac.to_bits() == (-0.0_f64).to_bits() {
                    // SAFETY: `lo < n` by plan invariant.
                    // `plan_presorted` only pushes `lo = bracket_hi - 1`
                    // with `bracket_hi ∈ [1, n - 1]`, so `lo ∈
                    // [0, n - 2]`.  `spec.len() == n` by the
                    // precondition assert at the top of `apply`.
                    unsafe { *spec.get_unchecked(lo) }
                } else {
                    // SAFETY: same `lo ∈ [0, n - 2]` invariant, so
                    // `lo + 1 ∈ [1, n - 1]` is also in-bounds.
                    let s_lo = unsafe { *spec.get_unchecked(lo) };
                    let s_hi = unsafe { *spec.get_unchecked(lo + 1) };
                    s_lo + frac * (s_hi - s_lo)
                };
                // Serial accumulation preserved — no multi-accumulator
                // reassociation, no SIMD lane-wise tree reduce.
                // IEEE-754 addition is not associative; changing the
                // order would break bit-exactness with
                // `broaden_presorted_reference` (and all
                // `*_bit_exact_*` unit tests + real-VENUS
                // `baseline_dump.py --verify`).
                sum += w * s;
            }
            result[i] = sum / norm_i;
        }

        result
    }

    /// Compile this plan into a row-stochastic CSR
    /// [`ResolutionMatrix`].
    ///
    /// The compiled matrix is an explicit sparse representation of
    /// the resolution operator `R` on the plan's target grid.  Each
    /// row sums to 1.0 to machine precision (passthrough rows store
    /// a single `(i, i, 1.0)` entry to match [`ResolutionPlan::apply`]
    /// 's `norm ≤ DIVISION_FLOOR` fallback).
    ///
    /// Degenerate-bracket handling uses the `-0.0` sentinel
    /// convention introduced in PR #470: if `plan.frac[e]` has the
    /// bit pattern of `-0.0`, the entry contributes `weight / norm`
    /// at column `lo` only (no `lo+1` bracket).  A regular `+0.0`
    /// frac contributes `weight * 1.0 / norm` at `lo` and
    /// `weight * 0.0 / norm = 0.0` at `lo+1` — those zero columns
    /// are retained in CSR with `value = 0.0` to preserve
    /// downstream NaN-safety if the consumer re-multiplies by a
    /// spectrum containing NaN at `lo+1`.
    ///
    /// # Equivalence contract (finite spectra only)
    ///
    /// For a spectrum with **all finite values**, [`apply_r`] on the
    /// compiled matrix produces per-element output within `1e-12`
    /// relative tolerance of [`Self::apply`] on the same spectrum —
    /// not bit-exact, because the CSR matvec sums contributions in
    /// column order while `apply` sums in entry order and IEEE-754
    /// addition is non-associative.  The `1e-12` bound accounts for
    /// accumulation error across the ~82 entries per row on the
    /// 3471-bin VENUS production grid (500 × 2.22e-16 ≈ 1.1e-13 per
    /// row; `1e-12` leaves comfortable headroom).
    ///
    /// # Non-finite and near-overflow spectra
    ///
    /// The equivalence bound does **NOT** extend to spectra with
    /// `NaN` / `±∞` values, **nor to near-f64::MAX overflow inputs**
    /// (Codex round-2 P3).  Both divergences trace back to the same
    /// algebraic rewrite:
    ///
    /// * [`Self::apply`] computes each entry as `spec[lo] + frac *
    ///   (spec[lo+1] - spec[lo])`, which can overflow the
    ///   subtraction even for finite inputs (opposite-sign
    ///   f64::MAX → `-∞`).
    /// * The compiled CSR form splits the interp into `(1 - frac) *
    ///   spec[lo] + frac * spec[lo + 1]`, which scales before
    ///   summing and stays finite in the same case.
    ///
    /// For bounded finite Beer-Lambert transmissions (`T ∈ [0, 1]`)
    /// neither divergence can arise; callers who deliberately pass
    /// non-finite or near-overflow spectra (e.g., as debug sentinels
    /// or out-of-range diagnostics) must not rely on cross-API
    /// equivalence.  See `resolution_matrix_nonfinite_contract` and
    /// `resolution_matrix_large_finite_contract` for executable
    /// demonstrations.
    pub fn compile_to_matrix(&self) -> ResolutionMatrix {
        let n = self.target_energies.len();
        let mut row_starts: Vec<u32> = Vec::with_capacity(n + 1);
        row_starts.push(0);
        let mut col_indices: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();

        // Reusable per-row accumulator.  Columns accumulate into a
        // BTreeMap keyed by spectrum index so the final CSR row is
        // emitted in ascending column order — the required CSR
        // invariant and the condition the `apply_r` equivalence
        // bound depends on.
        let mut acc: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();

        for i in 0..n {
            acc.clear();
            let norm_i = self.norm[i];
            if norm_i <= DIVISION_FLOOR {
                // Passthrough row — matches `apply`'s early return.
                col_indices.push(i as u32);
                values.push(1.0);
                // See u32-overflow `debug_assert!` below — the same
                // bound applies after every `push`.
                debug_assert!(
                    col_indices.len() <= u32::MAX as usize,
                    "CSR row_starts/col_indices u32 overflow: nnz = {}",
                    col_indices.len(),
                );
                row_starts.push(col_indices.len() as u32);
                continue;
            }
            let start = self.starts[i] as usize;
            let end = self.starts[i + 1] as usize;
            for e in start..end {
                let lo = self.lo_idx[e];
                let frac = self.frac[e];
                let w = self.weight[e];
                if frac.to_bits() == (-0.0_f64).to_bits() {
                    // Degenerate bracket — `apply` reads `spec[lo]`
                    // only, so the CSR row contributes only at `lo`.
                    *acc.entry(lo).or_insert(0.0) += w / norm_i;
                } else {
                    // Regular linear-interp entry: `w * ((1 - frac)
                    // * spec[lo] + frac * spec[lo + 1]) / norm_i`.
                    *acc.entry(lo).or_insert(0.0) += w * (1.0 - frac) / norm_i;
                    *acc.entry(lo + 1).or_insert(0.0) += w * frac / norm_i;
                }
            }
            for (&col, &val) in acc.iter() {
                col_indices.push(col);
                values.push(val);
            }
            // Defence-in-depth: a future large-grid caller that
            // accumulates more than u32::MAX entries would silently
            // truncate the `as u32` cast below.  The `plan_presorted`
            // helper already has matching `debug_assert!` guards on
            // its u32 offsets (resolution.rs, `plan_presorted`).
            debug_assert!(
                col_indices.len() <= u32::MAX as usize,
                "CSR row_starts/col_indices u32 overflow: nnz = {}",
                col_indices.len(),
            );
            row_starts.push(col_indices.len() as u32);
        }

        ResolutionMatrix {
            target_energies: self.target_energies.clone(),
            row_starts,
            col_indices,
            values,
        }
    }
}

/// Row-stochastic CSR representation of the resolution operator `R`
/// on a fixed target energy grid.
///
/// Built from a [`ResolutionPlan`] via
/// [`ResolutionPlan::compile_to_matrix`].  Exposed so downstream
/// surrogates (see epic #472) can access the row-local entries
/// `R_{i, j}` directly for LP / quadrature construction.
///
/// Owns a copy of the target energy grid for the same reason
/// [`ResolutionPlan`] does: caller-side grid-identity checks and
/// explicit grid-mismatch errors via
/// [`ResolutionError::MatrixGridMismatch`].
#[derive(Debug, Clone)]
pub struct ResolutionMatrix {
    /// Target energy grid the matrix was compiled for (owned copy).
    target_energies: Vec<f64>,
    /// `row_starts[i]..row_starts[i+1]` indexes into
    /// `col_indices`/`values` for row `i`.  Length `n + 1`.
    row_starts: Vec<u32>,
    /// Column indices in ascending order within each row.
    col_indices: Vec<u32>,
    /// CSR values.  Row `i` sums to 1.0 within machine precision
    /// (passthrough rows store exactly `1.0` at column `i`).
    values: Vec<f64>,
}

impl ResolutionMatrix {
    /// Number of rows (target-grid size) covered by this matrix.
    pub fn len(&self) -> usize {
        self.target_energies.len()
    }

    /// True when the matrix covers no target energies.
    pub fn is_empty(&self) -> bool {
        self.target_energies.is_empty()
    }

    /// Total number of stored entries (structural nnz).
    ///
    /// Regular-bracket entries with `frac == +0.0` retain a
    /// zero-valued contribution at the `lo + 1` column to preserve
    /// NaN-safety under re-application to spectra with NaN at that
    /// column; those stored zeros are counted in this total.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Target energy grid the matrix was compiled for.
    pub fn target_energies(&self) -> &[f64] {
        &self.target_energies
    }

    /// CSR row-start offsets.  `row_starts()[i]..row_starts()[i+1]`
    /// names the entry range for row `i`.  Length `len() + 1`.
    pub fn row_starts(&self) -> &[u32] {
        &self.row_starts
    }

    /// CSR column indices.  Sorted ascending within each row.
    pub fn col_indices(&self) -> &[u32] {
        &self.col_indices
    }

    /// CSR values.  Each row sums to 1.0 to machine precision.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Apply a compiled [`ResolutionMatrix`] to a spectrum on the same
/// target grid the matrix was compiled for.
///
/// For finite spectra, the output is numerically equivalent to
/// [`ResolutionPlan::apply`] on the same spectrum within `1e-12`
/// relative tolerance per element; not bit-exact, because CSR matvec
/// sums in column order while `ResolutionPlan::apply` sums in entry
/// order.
///
/// # Non-finite and near-overflow inputs
///
/// See [`ResolutionPlan::compile_to_matrix`] for the full contract
/// on `NaN` / `±∞` spectra **and on near-f64::MAX finite spectra** —
/// the equivalence bound does not extend to either.  Production
/// forward models feed Beer-Lambert transmissions (`T ∈ [0, 1]`) so
/// the distinction never arises in practice.
///
/// # Panics
///
/// Panics if `spectrum.len() != matrix.len()`.  Use
/// [`apply_resolution_with_matrix`] for a checked entrypoint that
/// returns [`ResolutionError::LengthMismatch`] instead.
pub fn apply_r(matrix: &ResolutionMatrix, spectrum: &[f64]) -> Vec<f64> {
    let n = matrix.len();
    assert_eq!(
        spectrum.len(),
        n,
        "spectrum length ({}) must match matrix grid length ({})",
        spectrum.len(),
        n,
    );
    let mut out = vec![0.0f64; n];
    for (i, out_i) in out.iter_mut().enumerate() {
        let start = matrix.row_starts[i] as usize;
        let end = matrix.row_starts[i + 1] as usize;
        let mut sum = 0.0f64;
        for e in start..end {
            let col = matrix.col_indices[e] as usize;
            sum += matrix.values[e] * spectrum[col];
        }
        *out_i = sum;
    }
    out
}

/// Checked variant of [`apply_r`] that validates the matrix was
/// compiled for `energies` before applying.
///
/// Returns [`ResolutionError::LengthMismatch`] when either
/// `energies` or `spectrum` has a length that disagrees with the
/// matrix grid size.  For the `spectrum` check, the `energies` field
/// of the returned error holds the matrix grid length (the required
/// length) so callers can read it as "expected vs got".  Returns
/// [`ResolutionError::MatrixGridMismatch`] when the lengths match
/// but the grid contents differ (per-element `to_bits()` compare).
///
/// Unlike [`apply_resolution_with_plan`], this entrypoint does not
/// enforce an ascending `energies` grid through the crate's internal
/// `validate_inputs` helper.  That check is redundant here: the plan
/// that produced the matrix was itself built on a sorted grid (via
/// [`TabulatedResolution::plan`], which validates sortedness), and the
/// stored `target_energies` copy is used in the `to_bits()`
/// grid-identity check above.  Any `energies` slice that is not
/// bit-identical to the matrix's stored copy — including an unsorted
/// permutation of the same values — fails with
/// [`ResolutionError::MatrixGridMismatch`].
pub fn apply_resolution_with_matrix(
    energies: &[f64],
    matrix: &ResolutionMatrix,
    spectrum: &[f64],
) -> Result<Vec<f64>, ResolutionError> {
    if energies.len() != matrix.len() {
        return Err(ResolutionError::LengthMismatch {
            energies: energies.len(),
            data: matrix.len(),
        });
    }
    if spectrum.len() != matrix.len() {
        // Reuse the `LengthMismatch` variant for the spectrum branch:
        // `energies` = expected length (matrix grid size), `data` =
        // actual spectrum length.  See docstring above.
        return Err(ResolutionError::LengthMismatch {
            energies: matrix.len(),
            data: spectrum.len(),
        });
    }
    for (i, (e_cur, e_ref)) in energies.iter().zip(matrix.target_energies()).enumerate() {
        // `to_bits()` equality catches `-0.0 vs +0.0` and NaN-bit
        // differences that float `==` silently accepts or rejects.
        if e_cur.to_bits() != e_ref.to_bits() {
            return Err(ResolutionError::MatrixGridMismatch {
                first_diff_index: i,
            });
        }
    }
    Ok(apply_r(matrix, spectrum))
}

impl TabulatedResolution {
    /// Parse a VENUS/FTS resolution file.
    ///
    /// # Arguments
    /// * `text` — File contents as a string.
    /// * `flight_path_m` — Flight path length in meters.
    pub fn from_text(text: &str, flight_path_m: f64) -> Result<Self, ResolutionParseError> {
        let mut lines = text.lines();

        // Skip header and separator
        let _header = lines
            .next()
            .ok_or(ResolutionParseError::InvalidFormat("Empty file".into()))?;
        let _sep = lines.next().ok_or(ResolutionParseError::InvalidFormat(
            "Missing separator".into(),
        ))?;

        let mut ref_energies = Vec::new();
        let mut kernels = Vec::new();
        let mut current_energy: Option<f64> = None;
        let mut current_offsets: Vec<f64> = Vec::new();
        let mut current_weights: Vec<f64> = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // End of current block
                if let Some(e) = current_energy.take() {
                    ref_energies.push(e);
                    kernels.push((
                        std::mem::take(&mut current_offsets),
                        std::mem::take(&mut current_weights),
                    ));
                }
                continue;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() != 2 {
                if current_energy.is_some() {
                    return Err(ResolutionParseError::InvalidFormat(format!(
                        "Expected 2 columns inside energy block, got {}: '{}'",
                        parts.len(),
                        trimmed
                    )));
                }
                // Outside a data block (e.g. extra header lines) — skip
                continue;
            }

            let x: f64 = parts[0].parse().map_err(|_| {
                ResolutionParseError::InvalidFormat(format!("Cannot parse float: '{}'", parts[0]))
            })?;
            let y: f64 = parts[1].parse().map_err(|_| {
                ResolutionParseError::InvalidFormat(format!("Cannot parse float: '{}'", parts[1]))
            })?;

            if current_energy.is_none() {
                // First line of block: energy + 0.0 marker
                current_energy = Some(x);
            } else {
                current_offsets.push(x);
                current_weights.push(y);
            }
        }

        // Flush last block
        if let Some(e) = current_energy.take() {
            ref_energies.push(e);
            kernels.push((current_offsets, current_weights));
        }

        if ref_energies.is_empty() {
            return Err(ResolutionParseError::InvalidFormat(
                "No energy blocks found".into(),
            ));
        }

        // Validate strictly ascending reference energies
        for i in 1..ref_energies.len() {
            if ref_energies[i] <= ref_energies[i - 1] {
                return Err(ResolutionParseError::InvalidFormat(format!(
                    "Reference energies must be strictly ascending, but E[{}]={} <= E[{}]={}",
                    i,
                    ref_energies[i],
                    i - 1,
                    ref_energies[i - 1],
                )));
            }
        }

        Ok(TabulatedResolution {
            ref_energies,
            kernels,
            flight_path_m,
        })
    }

    /// Parse a VENUS/FTS resolution file from disk.
    pub fn from_file(path: &str, flight_path_m: f64) -> Result<Self, ResolutionParseError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ResolutionParseError::IoError(format!("Cannot read '{}': {}", path, e)))?;
        Self::from_text(&text, flight_path_m)
    }

    /// Apply tabulated resolution broadening to a spectrum.
    ///
    /// For each energy point:
    /// 1. Find bracketing reference energies and interpolate kernel (log-space)
    /// 2. Convert TOF offsets to energy offsets using exact TOF↔energy relation
    /// 3. Convolve spectrum with interpolated kernel (trapezoidal integration)
    ///
    /// # Errors
    /// Returns [`ResolutionError::LengthMismatch`] if the arrays differ in
    /// length, or [`ResolutionError::UnsortedEnergies`] if the energy grid is
    /// not sorted in non-descending order.
    pub fn broaden(&self, energies: &[f64], spectrum: &[f64]) -> Result<Vec<f64>, ResolutionError> {
        validate_inputs(energies, spectrum)?;
        Ok(self.broaden_presorted(energies, spectrum))
    }

    /// Tabulated resolution broadening assuming the energy grid is already
    /// validated (sorted ascending, same length as spectrum).
    ///
    /// ## Inner-loop optimization
    ///
    /// The per-kernel-point spectrum interpolation uses a **two-pointer
    /// walk** instead of a binary search: `e_prime` is monotonically
    /// decreasing in `k` (since `dt = offsets[k]` is non-decreasing,
    /// `TOF' = tof_center + dt` is non-decreasing, and `E' = (L/TOF')²`
    /// is non-increasing).  We maintain `bracket_hi` as the smallest
    /// index into `energies[]` whose value is `>= e_prime`, and walk it
    /// downward as `k` advances.  Amortized O(1) per kernel point.
    ///
    /// Math is identical to the reference implementation pinned by
    /// `broaden_presorted_reference` in the test module.
    ///
    /// For callers that broaden many spectra on the same target grid —
    /// LM iterations with fixed TZERO, spatial maps with a pre-calibrated
    /// energy axis — [`TabulatedResolution::plan`] +
    /// [`ResolutionPlan::apply`] produce bit-exact output while
    /// hoisting the per-target invariants (TOF conversion, kernel
    /// interpolation, bracket lookup, trapezoidal widths) out of the
    /// broadening hot loop.  This `broaden_presorted` entry is the
    /// single-broadening path and keeps the original inline
    /// implementation to avoid plan-construction overhead on one-shot
    /// callers.
    pub(crate) fn broaden_presorted(&self, energies: &[f64], spectrum: &[f64]) -> Vec<f64> {
        let n = energies.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return spectrum.to_vec();
        }

        let e_min = energies[0];
        let e_max = energies[n - 1];

        let mut result = vec![0.0f64; n];

        for i in 0..n {
            let e = energies[i];
            if e <= 0.0 {
                result[i] = spectrum[i];
                continue;
            }

            let tof_center = TOF_FACTOR * self.flight_path_m / e.sqrt();
            let (offsets, weights) = self.interpolated_kernel(e);
            let n_k = offsets.len();
            let mut bracket_hi: usize = n - 1;

            let mut sum = 0.0;
            let mut norm = 0.0;

            for k in 0..n_k {
                let dt = offsets[k];
                let w = weights[k];
                if w <= 0.0 {
                    continue;
                }

                let tof_prime = tof_center + dt;
                if tof_prime <= 0.0 {
                    continue;
                }

                let e_prime = (TOF_FACTOR * self.flight_path_m / tof_prime).powi(2);

                if e_prime < e_min || e_prime > e_max {
                    continue;
                }

                while bracket_hi > 1 && energies[bracket_hi - 1] > e_prime {
                    bracket_hi -= 1;
                }
                while bracket_hi < n - 1 && energies[bracket_hi] <= e_prime {
                    bracket_hi += 1;
                }

                let lo = bracket_hi - 1;
                let hi = bracket_hi;
                let span = energies[hi] - energies[lo];
                let s = if span.abs() < NEAR_ZERO_FLOOR {
                    spectrum[lo]
                } else {
                    let frac = (e_prime - energies[lo]) / span;
                    spectrum[lo] + frac * (spectrum[hi] - spectrum[lo])
                };

                let dt_width = if k > 0 && k < n_k - 1 {
                    (offsets[k + 1] - offsets[k - 1]) * 0.5
                } else if k == 0 && n_k > 1 {
                    offsets[1] - offsets[0]
                } else if k == n_k - 1 && n_k > 1 {
                    offsets[k] - offsets[k - 1]
                } else {
                    1.0
                };

                let weight = w * dt_width.abs();
                sum += weight * s;
                norm += weight;
            }

            result[i] = if norm > DIVISION_FLOOR {
                sum / norm
            } else {
                spectrum[i]
            };
        }

        result
    }

    /// Build a reusable broadening plan for a specific target energy grid.
    ///
    /// Validates that `energies` is non-descending — the same sorted-grid
    /// precondition enforced by [`TabulatedResolution::broaden`] via
    /// `validate_inputs`.  An
    /// unsorted grid would produce a silently-wrong plan (misbracketed
    /// `e_prime` lookups against `e_min` / `e_max`), so it must be
    /// caught at build time rather than returning garbage from
    /// [`ResolutionPlan::apply`].
    ///
    /// The plan hoists every quantity that depends only on
    /// `(target_energies, self.ref_energies, self.flight_path_m)` —
    /// namely the TOF conversion, the log-space kernel interpolation,
    /// the per-kernel-point `e_prime` and spectrum-bracket lookup, and
    /// the trapezoidal integration widths.  Applying the plan to a
    /// spectrum becomes a pure gather + multiply-add loop.
    ///
    /// Build cost: same as one call to the private `broaden_presorted`
    /// helper (O(N_target × N_kernel) TOF / bracket / interp work, plus
    /// ~2 × N_kernel log-interp ops per target energy for
    /// `interpolated_kernel`).  Apply cost per target: 1 branch +
    /// ~3 loads + 3 flops per retained entry, plus the final divide —
    /// typically < 10 % of the build cost.  The payoff comes from
    /// reusing one plan across many spectra.
    ///
    /// Bit-exact with `broaden_presorted`: pre-computes the same
    /// floating-point sequences (TOF, `e_prime`, `dt_width`, `frac`,
    /// `weight`, `norm`) in the same order.
    ///
    /// # Errors
    /// Returns [`ResolutionError::UnsortedEnergies`] if `energies` is
    /// not non-descending.
    pub fn plan(&self, energies: &[f64]) -> Result<ResolutionPlan, ResolutionError> {
        if !energies.windows(2).all(|w| w[0] <= w[1]) {
            return Err(ResolutionError::UnsortedEnergies);
        }
        Ok(self.plan_presorted(energies))
    }

    /// Build a plan assuming `energies` is already validated as
    /// non-descending.  Used internally by `broaden_presorted` (whose
    /// caller already validated the grid) and by `plan()` after its
    /// validation succeeded.
    fn plan_presorted(&self, energies: &[f64]) -> ResolutionPlan {
        let n = energies.len();
        if n == 0 {
            return ResolutionPlan {
                target_energies: Vec::new(),
                starts: vec![0],
                lo_idx: Vec::new(),
                frac: Vec::new(),
                weight: Vec::new(),
                norm: Vec::new(),
            };
        }
        if n == 1 {
            // No bracket available; passthrough. Represent as n=1 with
            // zero entries and norm=0, which triggers the passthrough
            // branch in `ResolutionPlan::apply`.
            return ResolutionPlan {
                target_energies: energies.to_vec(),
                starts: vec![0, 0],
                lo_idx: Vec::new(),
                frac: Vec::new(),
                weight: Vec::new(),
                norm: vec![0.0],
            };
        }

        let e_min = energies[0];
        let e_max = energies[n - 1];

        // Preallocate the entry Vecs to ~n × kernel_len so the inner
        // pushes avoid repeated reallocations.  Real VENUS grids push
        // ~n × 499 entries total; over-allocating by up to 2× (if some
        // kernel points are skipped) is cheap vs. repeated grow-and-
        // memcpy during plan build.
        let estimated_kernel_len = self.kernels.first().map_or(0, |(off, _)| off.len());
        let estimated_entries = n.saturating_mul(estimated_kernel_len);

        let mut starts: Vec<u32> = Vec::with_capacity(n + 1);
        let mut lo_idx: Vec<u32> = Vec::with_capacity(estimated_entries);
        let mut frac: Vec<f64> = Vec::with_capacity(estimated_entries);
        let mut weight: Vec<f64> = Vec::with_capacity(estimated_entries);
        let mut norm: Vec<f64> = Vec::with_capacity(n);

        starts.push(0);

        for i in 0..n {
            let e = energies[i];
            if e <= 0.0 {
                // Passthrough: no entries contribute, norm=0.
                norm.push(0.0);
                // Guard the u32 invariant for diagnostic callers; the
                // headroom is enormous for any realistic grid (VENUS
                // 3471 × 499 ≈ 1.7M entries, u32::MAX ≈ 4.29B), but
                // the debug-only assert documents the contract.
                debug_assert!(
                    lo_idx.len() <= u32::MAX as usize,
                    "plan entry count overflows u32"
                );
                starts.push(lo_idx.len() as u32);
                continue;
            }

            // TOF at this energy: t = TOF_FACTOR * L / sqrt(E).
            // Computed here in the plan build and NOT at apply time — this
            // is the main invariant we hoist.
            let tof_center = TOF_FACTOR * self.flight_path_m / e.sqrt();

            // Interpolated kernel at this target energy.  Allocates two
            // ~N_kernel Vecs; those allocations happen once per plan
            // build instead of once per broadening call.
            let (offsets, weights) = self.interpolated_kernel(e);
            let n_k = offsets.len();

            // Two-pointer walk state (same invariant as broaden_presorted).
            let mut bracket_hi: usize = n - 1;

            let mut target_norm = 0.0;

            for k in 0..n_k {
                let dt = offsets[k];
                let w = weights[k];
                if w <= 0.0 {
                    continue;
                }

                let tof_prime = tof_center + dt;
                if tof_prime <= 0.0 {
                    continue;
                }

                let e_prime = (TOF_FACTOR * self.flight_path_m / tof_prime).powi(2);

                if e_prime < e_min || e_prime > e_max {
                    continue;
                }

                // Two-pointer walk — same logic + invariants as
                // broaden_presorted, in the same order, so bracket_hi
                // reaches the identical position for each kept (i, k).
                while bracket_hi > 1 && energies[bracket_hi - 1] > e_prime {
                    bracket_hi -= 1;
                }
                while bracket_hi < n - 1 && energies[bracket_hi] <= e_prime {
                    bracket_hi += 1;
                }

                let lo = bracket_hi - 1;
                let hi = bracket_hi;
                let span = energies[hi] - energies[lo];
                // Degenerate-bracket guard: if span < NEAR_ZERO_FLOOR,
                // broaden_presorted returns `spectrum[lo]` directly
                // without the interp arithmetic.  Store `frac = -0.0`
                // — the apply path short-circuits on the exact bit
                // pattern of `-0.0` and returns `spectrum[lo]` without
                // touching `spectrum[lo+1]`, so bit-exactness holds
                // even if `spectrum[lo+1]` is NaN or ±∞.
                //
                // `-0.0` (negative-signed zero) is used as the sentinel
                // because the non-degenerate path can legitimately
                // produce `frac == +0.0` when `e_prime == energies[lo]`
                // exactly — in that case `broaden_presorted` still
                // reads `spectrum[lo+1]` (and propagates NaN if present
                // there), so the apply path MUST do the same.  `+0.0`
                // and `-0.0` compare equal under `==` but differ in
                // `to_bits()`, which is what apply uses to disambiguate.
                //  Copilot review finding on PR #470.
                let entry_frac = if span.abs() < NEAR_ZERO_FLOOR {
                    -0.0_f64
                } else {
                    (e_prime - energies[lo]) / span
                };

                let dt_width = if k > 0 && k < n_k - 1 {
                    (offsets[k + 1] - offsets[k - 1]) * 0.5
                } else if k == 0 && n_k > 1 {
                    offsets[1] - offsets[0]
                } else if k == n_k - 1 && n_k > 1 {
                    offsets[k] - offsets[k - 1]
                } else {
                    1.0
                };

                let entry_weight = w * dt_width.abs();

                debug_assert!(
                    lo_idx.len() < u32::MAX as usize,
                    "plan entry count overflows u32"
                );
                lo_idx.push(lo as u32);
                frac.push(entry_frac);
                weight.push(entry_weight);
                target_norm += entry_weight;
            }

            norm.push(target_norm);
            starts.push(lo_idx.len() as u32);
        }

        ResolutionPlan {
            target_energies: energies.to_vec(),
            starts,
            lo_idx,
            frac,
            weight,
            norm,
        }
    }

    /// Interpolate kernel at an arbitrary energy using log-space linear interpolation
    /// between the two nearest reference energies.
    ///
    /// `ref_energies` is validated as strictly ascending by `from_text()` /
    /// `from_file()` at construction time, so no per-call sort check is needed.
    fn interpolated_kernel(&self, energy: f64) -> (Vec<f64>, Vec<f64>) {
        debug_assert!(
            self.ref_energies.windows(2).all(|w| w[0] < w[1]),
            "ref_energies must be strictly ascending (invariant broken)"
        );
        let n_ref = self.ref_energies.len();

        // Clamp to nearest reference if outside range
        if energy <= self.ref_energies[0] || n_ref == 1 {
            return self.kernels[0].clone();
        }
        if energy >= self.ref_energies[n_ref - 1] {
            return self.kernels[n_ref - 1].clone();
        }

        // Find bracketing indices
        let pos = self.ref_energies.partition_point(|&e| e < energy);
        let idx = if pos == 0 {
            0
        } else {
            (pos - 1).min(n_ref - 2)
        };

        let e_lo = self.ref_energies[idx];
        let e_hi = self.ref_energies[idx + 1];

        // Log-space interpolation fraction
        let frac = (energy.ln() - e_lo.ln()) / (e_hi.ln() - e_lo.ln());

        let (off_lo, w_lo) = &self.kernels[idx];
        let (off_hi, w_hi) = &self.kernels[idx + 1];

        // If both kernels have the same number of points, interpolate element-wise
        if off_lo.len() == off_hi.len() {
            let offsets: Vec<f64> = off_lo
                .iter()
                .zip(off_hi.iter())
                .map(|(&a, &b)| a + frac * (b - a))
                .collect();
            let weights: Vec<f64> = w_lo
                .iter()
                .zip(w_hi.iter())
                .map(|(&a, &b)| a + frac * (b - a))
                .collect();
            (offsets, weights)
        } else {
            // Different sizes: use nearest
            if frac < 0.5 {
                self.kernels[idx].clone()
            } else {
                self.kernels[idx + 1].clone()
            }
        }
    }
}

/// Apply resolution broadening using either Gaussian or tabulated kernel.
///
/// # Errors
/// Returns [`ResolutionError`] if the energy grid is unsorted or array
/// lengths do not match.
pub fn apply_resolution(
    energies: &[f64],
    spectrum: &[f64],
    resolution: &ResolutionFunction,
) -> Result<Vec<f64>, ResolutionError> {
    match resolution {
        ResolutionFunction::Gaussian(params) => resolution_broaden(energies, spectrum, params),
        ResolutionFunction::Tabulated(tab) => tab.broaden(energies, spectrum),
    }
}

/// Apply resolution broadening assuming the energy grid is already validated
/// (sorted ascending, same length as spectrum).
///
/// Used by `transmission.rs` to avoid redundant O(N) sort checks when
/// broadening multiple isotopes on the same pre-validated energy grid.
pub(crate) fn apply_resolution_presorted(
    energies: &[f64],
    spectrum: &[f64],
    resolution: &ResolutionFunction,
) -> Vec<f64> {
    match resolution {
        ResolutionFunction::Gaussian(params) => {
            resolution_broaden_presorted(energies, spectrum, params)
        }
        ResolutionFunction::Tabulated(tab) => tab.broaden_presorted(energies, spectrum),
    }
}

/// Build a broadening plan for `(energies, resolution)`.
///
/// Returns `Some(plan)` for [`ResolutionFunction::Tabulated`] — the
/// plan hoists the per-target TOF / kernel-interpolation / bracket
/// / trap-weight work that would otherwise run on every call to
/// [`apply_resolution`].  Returns `None` for
/// [`ResolutionFunction::Gaussian`] — the Gaussian path has no
/// meaningful pixel-invariant kernel structure to cache at this
/// level, so callers fall back to the per-call broadening path with
/// no loss.
///
/// Callers that want a single-branch API can unconditionally call
/// [`apply_resolution_with_plan`] passing `plan.as_ref()`; when the
/// plan is `None` it transparently forwards to the non-plan path and
/// returns byte-identical output.
///
/// # Errors
/// Returns [`ResolutionError::UnsortedEnergies`] if `energies` is not
/// non-descending — the same precondition that [`apply_resolution`]
/// enforces per-call.
pub fn build_resolution_plan(
    energies: &[f64],
    resolution: &ResolutionFunction,
) -> Result<Option<ResolutionPlan>, ResolutionError> {
    match resolution {
        ResolutionFunction::Gaussian(_) => {
            if !energies.windows(2).all(|w| w[0] <= w[1]) {
                return Err(ResolutionError::UnsortedEnergies);
            }
            Ok(None)
        }
        ResolutionFunction::Tabulated(tab) => tab.plan(energies).map(Some),
    }
}

/// Apply resolution broadening, optionally via a pre-built
/// [`ResolutionPlan`].
///
/// When `plan` is `Some(p)` and `resolution` is a tabulated kernel,
/// `p.apply(spectrum)` runs the cached per-target broadening inner
/// loop — the expensive TOF / kernel-interpolation / bracket work
/// was already captured at plan build time.
///
/// When `plan` is `None`, or when `resolution` is Gaussian, the call
/// forwards to [`apply_resolution`] and is byte-identical to the
/// un-planned path.
///
/// # Errors
/// * Returns the same errors as [`apply_resolution`] on the non-plan
///   path.
/// * Returns [`ResolutionError::LengthMismatch`] if the plan was built
///   for a different-length grid than `energies`, or if
///   `energies.len() != spectrum.len()`.
/// * Returns [`ResolutionError::PlanGridMismatch`] if the plan was
///   built for a different grid of the same length — the cached
///   `(lo_idx, frac, weight)` entries encode brackets into the old
///   grid and would silently produce a wrong broadened spectrum if
///   applied.
pub fn apply_resolution_with_plan(
    plan: Option<&ResolutionPlan>,
    energies: &[f64],
    spectrum: &[f64],
    resolution: &ResolutionFunction,
) -> Result<Vec<f64>, ResolutionError> {
    if let Some(p) = plan
        && matches!(resolution, ResolutionFunction::Tabulated(_))
    {
        validate_inputs(energies, spectrum)?;
        if p.len() != energies.len() {
            return Err(ResolutionError::LengthMismatch {
                energies: energies.len(),
                data: p.len(),
            });
        }
        // Grid-identity check.  A plan built for a different grid of
        // the same length would still pass the length check and then
        // gather spectrum values at brackets that belong to the old
        // grid — silently corrupt output.  Pointer identity is not
        // enough here because callers legitimately hold the plan and
        // the target grid in separate `Arc`s whose storage may or
        // may not alias; bit-exact content equality is the only
        // robust invariant.  The cost is one full grid scan per
        // broadening call (O(n), ~27 KB of f64 values for the VENUS
        // 3471-point grid) — orders of magnitude cheaper than the
        // broadening itself and cheap vs the silent-staleness
        // failure mode.
        let plan_grid = p.target_energies();
        for i in 0..plan_grid.len() {
            if plan_grid[i].to_bits() != energies[i].to_bits() {
                return Err(ResolutionError::PlanGridMismatch {
                    first_diff_index: i,
                });
            }
        }
        return Ok(p.apply(spectrum));
    }
    apply_resolution(energies, spectrum, resolution)
}

/// Errors from resolution file parsing.
#[derive(Debug)]
pub enum ResolutionParseError {
    InvalidFormat(String),
    IoError(String),
}

impl fmt::Display for ResolutionParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "Invalid resolution file format: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for ResolutionParseError {}

/// Test-only helpers exposed to other `nereids-physics` test modules
/// that need to synthesize a [`ResolutionPlan`] without going through
/// the full `TabulatedResolution::plan` path.  Gated `#[cfg(test)]` so
/// the raw constructor never ships in a release build.
#[cfg(test)]
pub(crate) mod test_support {
    use super::ResolutionPlan;

    /// Build a [`ResolutionPlan`] directly from its SoA fields.
    ///
    /// The caller is responsible for maintaining the invariants that
    /// `plan_presorted` normally enforces (`starts.last() ==
    /// lo_idx.len()`, lo_idx in [0, n-2] for regular entries, etc.).
    /// Used by surrogate-module tests to construct hand-designed
    /// plans that exercise specific CSR patterns.
    pub(crate) fn plan_from_raw_parts(
        target_energies: Vec<f64>,
        starts: Vec<u32>,
        lo_idx: Vec<u32>,
        frac: Vec<f64>,
        weight: Vec<f64>,
        norm: Vec<f64>,
    ) -> ResolutionPlan {
        ResolutionPlan {
            target_energies,
            starts,
            lo_idx,
            frac,
            weight,
            norm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::constants;

    #[test]
    fn test_tof_factor_consistency() {
        // Verify our TOF_FACTOR matches the constants module.
        let e = 10.0; // eV
        let l = 25.0; // meters
        let tof_constants = constants::energy_to_tof(e, l);
        let tof_ours = TOF_FACTOR * l / e.sqrt();
        let rel_diff = (tof_constants - tof_ours).abs() / tof_constants;
        assert!(
            rel_diff < 1e-10,
            "TOF mismatch: constants={}, ours={}, diff={:.4}%",
            tof_constants,
            tof_ours,
            rel_diff * 100.0
        );
    }

    #[test]
    fn test_resolution_width_scaling() {
        let params = ResolutionParams::new(25.0, 1.0, 0.01, 0.0).unwrap();

        // Resolution width should increase with energy.
        let w1 = params.gaussian_width(1.0);
        let w10 = params.gaussian_width(10.0);
        let w100 = params.gaussian_width(100.0);

        assert!(w10 > w1, "Width should increase with energy");
        assert!(w100 > w10, "Width should increase with energy");

        // At low energies, timing dominates: ΔE ∝ E^(3/2)
        // At high energies, path dominates: ΔE ∝ E
        // The ratio ΔE(10)/ΔE(1) should be between 10 and 31.6 (= 10^1.5)
        let ratio = w10 / w1;
        assert!(
            ratio > 5.0 && ratio < 40.0,
            "Width ratio = {}, expected between 10 and 31.6",
            ratio
        );
    }

    #[test]
    fn test_zero_width_passthrough() {
        // If resolution parameters are zero, output should equal input.
        let energies = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let xs = vec![10.0, 20.0, 30.0, 20.0, 10.0];
        let params = ResolutionParams::new(25.0, 0.0, 0.0, 0.0).unwrap();
        let broadened = resolution_broaden(&energies, &xs, &params).unwrap();
        assert_eq!(broadened, xs);
    }

    #[test]
    fn test_broadening_reduces_peak() {
        // Resolution broadening should reduce peak heights and fill valleys.
        let n = 1001;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 0.01).collect();
        let center = 10.0;
        let gamma: f64 = 0.1; // Resonance width
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let de = e - center;
                1000.0 * (gamma / 2.0).powi(2) / (de * de + (gamma / 2.0).powi(2))
            })
            .collect();

        let params = ResolutionParams::new(25.0, 5.0, 0.01, 0.0).unwrap();
        let broadened = resolution_broaden(&energies, &xs, &params).unwrap();

        let orig_peak = xs.iter().cloned().fold(0.0_f64, f64::max);
        let broad_peak = broadened.iter().cloned().fold(0.0_f64, f64::max);

        assert!(
            broad_peak < orig_peak,
            "Broadened peak ({}) should be < original ({})",
            broad_peak,
            orig_peak
        );
        assert!(
            broad_peak > 1.0,
            "Broadened peak ({}) should still be substantial",
            broad_peak
        );
    }

    #[test]
    fn test_broadening_conserves_area() {
        // Resolution broadening should approximately conserve the area
        // under the cross-section curve.
        let n = 2001;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 0.005).collect();
        let center = 10.0;
        let gamma: f64 = 0.5;
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let de = e - center;
                1000.0 * (gamma / 2.0).powi(2) / (de * de + (gamma / 2.0).powi(2))
            })
            .collect();

        let params = ResolutionParams::new(25.0, 1.0, 0.01, 0.0).unwrap();
        let broadened = resolution_broaden(&energies, &xs, &params).unwrap();

        // Trapezoidal area
        let area_orig: f64 = (0..n - 1)
            .map(|i| 0.5 * (xs[i] + xs[i + 1]) * (energies[i + 1] - energies[i]))
            .sum();
        let area_broad: f64 = (0..n - 1)
            .map(|i| 0.5 * (broadened[i] + broadened[i + 1]) * (energies[i + 1] - energies[i]))
            .sum();

        let rel_diff = (area_orig - area_broad).abs() / area_orig;
        assert!(
            rel_diff < 0.02,
            "Area not conserved: orig={:.2}, broad={:.2}, rel_diff={:.4}",
            area_orig,
            area_broad,
            rel_diff
        );
    }

    #[test]
    fn test_gaussian_broadening_analytical() {
        // Broadening a Gaussian with a Gaussian should give a wider Gaussian.
        //
        // Input:  exp(-x²/(2σ₁²)) with σ₁ = 0.5 eV (standard Gaussian form)
        // Kernel: exp(-x²/W²) with W = 0.3 eV → std dev σ₂ = W/√2 = 0.2121 eV
        // Output: Gaussian with σ_out = √(σ₁² + σ₂²) = √(0.25 + 0.045) = 0.543 eV
        //
        // Note: kernel width varies slightly with energy (σ_E ∝ E for the
        // path-length contribution), so we allow ~5% tolerance.
        let n = 2001;
        let center = 10.0;
        let sigma_input = 0.5; // eV (standard deviation)
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 0.005).collect();
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let de = e - center;
                1000.0 * (-de * de / (2.0 * sigma_input * sigma_input)).exp()
            })
            .collect();

        // Set delta_l such that W = gaussian_width(E=10) ≈ 0.3 eV.
        // W = 2·ΔL·E/L, so ΔL = W·L/(2E) = 0.3×25/(20) = 0.375 m
        let w_kernel = 0.3; // Kernel parameter W (exp(-x²/W²))
        let params =
            ResolutionParams::new(25.0, 0.0, w_kernel * 25.0 / (2.0 * center), 0.0).unwrap();

        // Verify kernel W at center energy
        let w_at_center = params.gaussian_width(center);
        assert!(
            (w_at_center - w_kernel).abs() / w_kernel < 0.01,
            "Kernel W at center: {}, expected {}",
            w_at_center,
            w_kernel
        );

        let broadened = resolution_broaden(&energies, &xs, &params).unwrap();

        // Kernel std dev = W/√2
        let sigma_kernel = w_kernel / 2.0_f64.sqrt();
        let sigma_expected = (sigma_input * sigma_input + sigma_kernel * sigma_kernel).sqrt();
        let fwhm_expected = 2.0 * (2.0_f64.ln() * 2.0).sqrt() * sigma_expected;

        // Measure FWHM from the broadened output
        let peak_idx = broadened
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let peak_val = broadened[peak_idx];
        let half_max = peak_val / 2.0;

        let mut left_hm = energies[0];
        for i in (0..peak_idx).rev() {
            if broadened[i] < half_max {
                let t = (half_max - broadened[i]) / (broadened[i + 1] - broadened[i]);
                left_hm = energies[i] + t * (energies[i + 1] - energies[i]);
                break;
            }
        }
        let mut right_hm = energies[n - 1];
        for i in peak_idx..n - 1 {
            if broadened[i + 1] < half_max {
                let t = (half_max - broadened[i]) / (broadened[i + 1] - broadened[i]);
                right_hm = energies[i] + t * (energies[i + 1] - energies[i]);
                break;
            }
        }

        let fwhm_measured = right_hm - left_hm;
        let rel_err = (fwhm_measured - fwhm_expected).abs() / fwhm_expected;

        assert!(
            rel_err < 0.05,
            "FWHM: measured={:.4}, expected={:.4}, rel_err={:.2}%",
            fwhm_measured,
            fwhm_expected,
            rel_err * 100.0
        );
    }

    #[test]
    fn test_venus_typical_resolution() {
        // Verify resolution width for typical VENUS parameters.
        // VENUS: L ≈ 25 m, Δt ≈ 10 μs (pulsed source), ΔL ≈ 0.01 m
        let params = ResolutionParams::new(25.0, 10.0, 0.01, 0.0).unwrap();

        // At 1 eV: ΔE/E should be small (good resolution)
        let de_1 = params.gaussian_width(1.0);
        let de_over_e_1 = de_1 / 1.0;
        assert!(
            de_over_e_1 < 0.05,
            "ΔE/E at 1 eV = {:.4}, should be < 5%",
            de_over_e_1
        );

        // At 100 eV: resolution degrades
        let de_100 = params.gaussian_width(100.0);
        let de_over_e_100 = de_100 / 100.0;
        assert!(
            de_over_e_100 > de_over_e_1,
            "Resolution should degrade at higher energies"
        );
    }

    #[test]
    fn test_unsorted_energies_returns_error() {
        let energies = vec![1.0, 3.0, 2.0, 4.0]; // not sorted
        let xs = vec![10.0, 30.0, 20.0, 40.0];
        let params = ResolutionParams::new(25.0, 1.0, 0.01, 0.0).unwrap();
        let result = resolution_broaden(&energies, &xs, &params);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResolutionError::UnsortedEnergies
        ));
    }

    #[test]
    fn test_length_mismatch_returns_error() {
        let energies = vec![1.0, 2.0, 3.0];
        let xs = vec![10.0, 20.0]; // wrong length
        let params = ResolutionParams::new(25.0, 1.0, 0.01, 0.0).unwrap();
        let result = resolution_broaden(&energies, &xs, &params);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResolutionError::LengthMismatch {
                energies: 3,
                data: 2
            }
        ));
    }

    // --- ResolutionParams validation tests ---

    #[test]
    fn test_resolution_params_valid() {
        let p = ResolutionParams::new(25.0, 1.0, 0.01, 0.0).unwrap();
        assert!((p.flight_path_m() - 25.0).abs() < 1e-15);
        assert!((p.delta_t_us() - 1.0).abs() < 1e-15);
        assert!((p.delta_l_m() - 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_resolution_params_rejects_zero_flight_path() {
        let err = ResolutionParams::new(0.0, 1.0, 0.01, 0.0).unwrap_err();
        assert_eq!(err, ResolutionParamsError::InvalidFlightPath(0.0));
    }

    #[test]
    fn test_resolution_params_rejects_negative_flight_path() {
        let err = ResolutionParams::new(-1.0, 1.0, 0.01, 0.0).unwrap_err();
        assert_eq!(err, ResolutionParamsError::InvalidFlightPath(-1.0));
    }

    #[test]
    fn test_resolution_params_rejects_nan_flight_path() {
        let err = ResolutionParams::new(f64::NAN, 1.0, 0.01, 0.0).unwrap_err();
        assert!(matches!(err, ResolutionParamsError::InvalidFlightPath(_)));
    }

    #[test]
    fn test_resolution_params_rejects_negative_delta_t() {
        let err = ResolutionParams::new(25.0, -1.0, 0.01, 0.0).unwrap_err();
        assert_eq!(err, ResolutionParamsError::InvalidDeltaT(-1.0));
    }

    #[test]
    fn test_resolution_params_rejects_nan_delta_t() {
        let err = ResolutionParams::new(25.0, f64::NAN, 0.01, 0.0).unwrap_err();
        assert!(matches!(err, ResolutionParamsError::InvalidDeltaT(_)));
    }

    #[test]
    fn test_resolution_params_rejects_negative_delta_l() {
        let err = ResolutionParams::new(25.0, 1.0, -0.01, 0.0).unwrap_err();
        assert_eq!(err, ResolutionParamsError::InvalidDeltaL(-0.01));
    }

    #[test]
    fn test_resolution_params_rejects_inf_delta_l() {
        let err = ResolutionParams::new(25.0, 1.0, f64::INFINITY, 0.0).unwrap_err();
        assert!(matches!(err, ResolutionParamsError::InvalidDeltaL(_)));
    }

    #[test]
    fn test_resolution_params_rejects_negative_delta_e() {
        let err = ResolutionParams::new(25.0, 1.0, 0.01, -0.05).unwrap_err();
        assert_eq!(err, ResolutionParamsError::InvalidDeltaE(-0.05));
    }

    #[test]
    fn test_resolution_params_rejects_nan_delta_e() {
        let err = ResolutionParams::new(25.0, 1.0, 0.01, f64::NAN).unwrap_err();
        assert!(matches!(err, ResolutionParamsError::InvalidDeltaE(_)));
    }

    #[test]
    fn test_resolution_params_accepts_zero_delta_e() {
        let p = ResolutionParams::new(25.0, 1.0, 0.01, 0.0).unwrap();
        assert!((p.delta_e_us() - 0.0).abs() < 1e-15);
        assert!(!p.has_exponential_tail());
    }

    // ─── broaden_presorted bit-exact equivalence harness ─────────────────────
    //
    // The optimized broaden_presorted uses a two-pointer walk instead of
    // binary search inside the inner convolution loop.  These tests pin
    // the math: the same formula, in the same order, must yield bit-exact
    // output against a canonical reference implementation that preserves
    // the pre-optimization code path.

    /// Binary-search spectrum interpolation.  Preserved here because the
    /// reference broaden_presorted uses it; the production inner loop now
    /// uses a two-pointer walk and no longer needs this helper.
    fn interp_spectrum(energies: &[f64], spectrum: &[f64], e: f64) -> Option<f64> {
        let n = energies.len();
        if n == 0 {
            return None;
        }
        if e < energies[0] || e > energies[n - 1] {
            return None;
        }
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if energies[mid] <= e {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = energies[hi] - energies[lo];
        if span.abs() < NEAR_ZERO_FLOOR {
            return Some(spectrum[lo]);
        }
        let frac = (e - energies[lo]) / span;
        Some(spectrum[lo] + frac * (spectrum[hi] - spectrum[lo]))
    }

    /// Reference implementation — the pre-optimization broaden_presorted.
    /// Kept in the test module only; used solely as the equivalence oracle.
    fn broaden_presorted_reference(
        tab: &TabulatedResolution,
        energies: &[f64],
        spectrum: &[f64],
    ) -> Vec<f64> {
        let n = energies.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0f64; n];

        for i in 0..n {
            let e = energies[i];
            if e <= 0.0 {
                result[i] = spectrum[i];
                continue;
            }

            let tof_center = TOF_FACTOR * tab.flight_path_m / e.sqrt();
            let (offsets, weights) = tab.interpolated_kernel(e);

            let mut sum = 0.0;
            let mut norm = 0.0;

            for k in 0..offsets.len() {
                let dt = offsets[k];
                let w = weights[k];
                if w <= 0.0 {
                    continue;
                }

                let tof_prime = tof_center + dt;
                if tof_prime <= 0.0 {
                    continue;
                }

                let e_prime = (TOF_FACTOR * tab.flight_path_m / tof_prime).powi(2);

                let s = match interp_spectrum(energies, spectrum, e_prime) {
                    Some(v) => v,
                    None => continue,
                };

                let dt_width = if k > 0 && k < offsets.len() - 1 {
                    (offsets[k + 1] - offsets[k - 1]) * 0.5
                } else if k == 0 && offsets.len() > 1 {
                    offsets[1] - offsets[0]
                } else if k == offsets.len() - 1 && offsets.len() > 1 {
                    offsets[k] - offsets[k - 1]
                } else {
                    1.0
                };

                let weight = w * dt_width.abs();
                sum += weight * s;
                norm += weight;
            }

            result[i] = if norm > DIVISION_FLOOR {
                sum / norm
            } else {
                spectrum[i]
            };
        }

        result
    }

    /// Synthetic TabulatedResolution with 3 reference energies and a
    /// triangular kernel of varying widths.  Deterministic, no I/O.
    fn synthetic_tab_resolution() -> TabulatedResolution {
        fn triangle(width_us: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
            let half = width_us;
            let dt_step = 2.0 * half / (n - 1) as f64;
            let offsets: Vec<f64> = (0..n).map(|i| -half + i as f64 * dt_step).collect();
            let weights: Vec<f64> = offsets
                .iter()
                .map(|&dt| (1.0 - dt.abs() / half).max(0.0))
                .collect();
            (offsets, weights)
        }
        TabulatedResolution {
            ref_energies: vec![5.0, 50.0, 500.0],
            kernels: vec![triangle(0.5, 31), triangle(1.0, 41), triangle(2.0, 51)],
            flight_path_m: 25.0,
        }
    }

    fn assert_bit_exact(reference: &[f64], actual: &[f64], label: &str) {
        assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
        for (i, (&a, &b)) in reference.iter().zip(actual.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: element {i} mismatch: reference={a:.17e} actual={b:.17e}"
            );
        }
    }

    #[test]
    fn test_broaden_presorted_bit_exact_synthetic_uniform() {
        let tab = synthetic_tab_resolution();
        // Uniform log-spaced grid typical of VENUS analysis.
        let energies: Vec<f64> = (0..401).map(|i| 7.0 + i as f64 * 0.4825).collect();
        // Triangular dip + smooth background (resonance-like spectrum).
        let spectrum: Vec<f64> = energies
            .iter()
            .enumerate()
            .map(|(i, &e)| 0.9 - 0.7 * (-((e - 50.0).powi(2) / 4.0)).exp() + 0.001 * (i as f64))
            .collect();

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "synthetic_uniform");
    }

    #[test]
    fn test_broaden_presorted_bit_exact_synthetic_nonuniform() {
        let tab = synthetic_tab_resolution();
        // Non-uniform: denser near 6.674 eV (resonance-like), sparser far away.
        let energies: Vec<f64> = {
            let mut e = Vec::new();
            for i in 0..200 {
                e.push(5.0 + (i as f64) * 0.05);
            }
            for i in 0..100 {
                e.push(15.0 + (i as f64) * 0.5);
            }
            for i in 0..50 {
                e.push(65.0 + (i as f64) * 2.0);
            }
            e
        };
        let spectrum: Vec<f64> = energies
            .iter()
            .map(|&e| 1.0 - 0.5 * (-((e - 6.674).powi(2) / 0.1)).exp())
            .collect();

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "synthetic_nonuniform");
    }

    #[test]
    fn test_broaden_presorted_bit_exact_constant_spectrum() {
        // Constant spectrum must pass through unchanged (within trapezoid
        // normalization) — preserves integral exactly.
        let tab = synthetic_tab_resolution();
        let energies: Vec<f64> = (0..501).map(|i| 1.0 + i as f64 * 0.5).collect();
        let spectrum = vec![0.42f64; energies.len()];

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "constant_spectrum");
    }

    #[test]
    fn test_broaden_presorted_bit_exact_short_grid() {
        // Edge case: 2-point grid.  Exercises the smallest grid that has
        // a valid (lo, hi) bracket — tests bracket_hi bounds handling.
        let tab = synthetic_tab_resolution();
        let energies = vec![10.0, 12.0];
        let spectrum = vec![0.5, 0.8];

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "short_grid");
    }

    #[test]
    fn test_broaden_presorted_bit_exact_single_point_grid() {
        // Edge case: 1-point grid.  Exercises the n == 1 early-return
        // pass-through guard that the optimized path adds (no bracket
        // available for interpolation).
        let tab = synthetic_tab_resolution();
        let energies = vec![10.0];
        let spectrum = vec![0.5];

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "single_point_grid");
    }

    #[test]
    fn test_broaden_presorted_bit_exact_exact_equality_target() {
        // Regression: exercise the tie-break case where `e_prime` lands
        // exactly on a grid point.  The kernel has a point at dt=0, so
        // `e_prime == energies[i]` exactly at the center kernel offset
        // for every target `i`.  The optimized path must match the
        // reference's upper-bound binary-search semantics bit-exactly.
        let tab = synthetic_tab_resolution();
        // Irregular-spacing grid so the spectrum interp at the equality
        // point isn't trivially reducible to the input value.
        let mut energies: Vec<f64> = Vec::new();
        let mut e = 3.0f64;
        for k in 0..800 {
            energies.push(e);
            e += 0.05 + 0.01 * (k as f64).sin();
        }
        // Spectrum with large local gradient so `a + (b - a)` vs `b`
        // would diverge at 1 ULP if the tie-break were wrong.
        let spectrum: Vec<f64> = energies
            .iter()
            .map(|&e| 1.0e10 * (-(((e - 6.0) / 0.2).powi(2))).exp() + 1.0e-10 * e)
            .collect();

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "exact_equality_target");
    }

    #[test]
    fn test_broaden_presorted_bit_exact_random_spectrum() {
        // Random spectrum with varied magnitudes exercises the interpolation
        // arithmetic across sign changes and scales.
        let tab = synthetic_tab_resolution();
        let energies: Vec<f64> = (0..1001).map(|i| 2.0 + i as f64 * 0.2).collect();
        // Deterministic pseudo-random via a simple LCG (no external dep).
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let spectrum: Vec<f64> = energies
            .iter()
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let f = ((state >> 33) as f64) / (u32::MAX as f64);
                f * 2.0 - 1.0
            })
            .collect();

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "random_spectrum");
    }

    /// Real PLEIADES bl10 resolution file + real Hf-like resonance
    /// spectrum on the full VENUS analysis grid.  This is the closest
    /// regression of the production A.1 / B.2 workload.
    ///
    /// Marked `#[ignore]` because `_fts_bl10_0p5meV_1keV_25pts.txt` is
    /// gitignored at the repo root per the "not approved for public
    /// release" policy (.gitignore line 48).  Run locally with:
    ///
    /// ```text
    /// cargo test -p nereids-physics \
    ///   test_broaden_presorted_bit_exact_on_pleiades_resolution \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn test_broaden_presorted_bit_exact_on_pleiades_resolution() {
        let res_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("_fts_bl10_0p5meV_1keV_25pts.txt");
        let text = std::fs::read_to_string(&res_path).expect(
            "missing PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at the repo root \
             (the file is gitignored per policy; place it locally before running this test)",
        );
        let tab = TabulatedResolution::from_text(&text, 25.0).unwrap();

        // Production-like grid: uniform 7..200 eV with ~3500 bins
        let n = 3471;
        let energies: Vec<f64> = (0..n)
            .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
            .collect();
        // Resonance-dip spectrum (toy model, exercises the math regardless
        // of actual Hf σ, which is what we want for an interp test).
        let spectrum: Vec<f64> = energies
            .iter()
            .map(|&e| {
                1.0 - 0.8 * (-((e - 7.8).powi(2) / 0.01)).exp()
                    - 0.5 * (-((e - 13.9).powi(2) / 0.04)).exp()
                    - 0.6 * (-((e - 22.4).powi(2) / 0.1)).exp()
            })
            .collect();

        let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        let actual = tab.broaden_presorted(&energies, &spectrum);
        assert_bit_exact(&reference, &actual, "pleiades_real_resolution");
    }

    /// Microbenchmark: two-pointer broaden_presorted vs binary-search
    /// reference.  Run with:
    ///
    /// ```text
    /// cargo test --release -p nereids-physics \
    ///   test_broaden_presorted_bench -- --ignored --nocapture
    /// ```
    #[test]
    fn test_plan_reuse_bit_exact_across_multiple_spectra() {
        // Core promise of ResolutionPlan: building the plan once and
        // applying it to K different spectra must yield the same output
        // as K independent `broaden_presorted` calls.
        let tab = synthetic_tab_resolution();
        let energies: Vec<f64> = (0..401).map(|i| 7.0 + i as f64 * 0.4825).collect();

        // Build plan ONCE.
        let plan = tab.plan(&energies).expect("sorted grid must validate");
        assert_eq!(plan.len(), energies.len());
        assert_eq!(plan.target_energies(), &energies[..]);

        // Apply across 5 varied spectra.
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        for spec_idx in 0..5 {
            let spectrum: Vec<f64> = energies
                .iter()
                .enumerate()
                .map(|(i, &e)| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let noise = ((state >> 33) as f64) / (u32::MAX as f64);
                    // Varied magnitudes and shapes per spectrum to catch
                    // spectrum-dependent arithmetic drift.
                    (10.0f64).powi(spec_idx - 2) * (1.0 - 0.5 * noise)
                        + 0.3 * (-((e - 50.0).powi(2) / 4.0)).exp()
                        + (spec_idx as f64) * 1e-8 * (i as f64)
                })
                .collect();

            let via_plan = plan.apply(&spectrum);
            let via_reference = broaden_presorted_reference(&tab, &energies, &spectrum);
            assert_bit_exact(
                &via_reference,
                &via_plan,
                &format!("plan_reuse[spec_idx={spec_idx}]"),
            );
        }
    }

    #[test]
    fn test_plan_passthrough_cases() {
        // n == 0, n == 1, and e <= 0.0 must all produce the same
        // passthrough behaviour via plan as via broaden_presorted.
        let tab = synthetic_tab_resolution();

        // n == 0: empty plan, empty result.
        let plan = tab.plan(&[]).unwrap();
        assert_eq!(plan.len(), 0);
        assert!(plan.is_empty());
        let out: Vec<f64> = plan.apply(&[]);
        assert!(out.is_empty());

        // n == 1: passthrough for any spectrum value.
        let plan1 = tab.plan(&[5.0]).unwrap();
        assert_eq!(plan1.len(), 1);
        let out1 = plan1.apply(&[0.42]);
        assert_eq!(out1, vec![0.42]);

        // e <= 0.0 in the middle of a grid: passthrough at that index.
        // Mixed positive / non-positive energies are pathological but
        // the current implementation handles them, and the plan must
        // match.  Grid is still non-descending (0.0 ≤ 10.0 etc.) so
        // plan() accepts it.
        let energies = vec![1.0, 1.0, 10.0, 100.0];
        let spectrum = vec![0.1, 0.5, 0.9, 0.3];
        let via_plan = {
            let plan = tab.plan(&energies).unwrap();
            plan.apply(&spectrum)
        };
        let via_reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        assert_bit_exact(
            &via_reference,
            &via_plan,
            "mixed_positive_and_zero_energies",
        );
    }

    #[test]
    fn test_plan_rejects_unsorted_energies() {
        // `broaden()` rejects unsorted grids via validate_inputs; `plan()`
        // must do the same so a caller doesn't silently build a plan with
        // misbracketed e_prime lookups and then produce wrong σ output
        // from `ResolutionPlan::apply`.
        let tab = synthetic_tab_resolution();
        let result = tab.plan(&[10.0, 1.0, 100.0]);
        assert!(matches!(result, Err(ResolutionError::UnsortedEnergies)));
    }

    #[test]
    fn test_plan_apply_is_nan_safe_at_degenerate_bracket() {
        // When two adjacent target energies are equal (span = 0), the
        // plan encodes `frac = 0.0` and the apply path must short-
        // circuit to `spectrum[lo]` without reading `spectrum[lo+1]`.
        // A NaN at the upper bracket would propagate through
        // `0.0 * NaN = NaN` and corrupt the result otherwise.
        let tab = synthetic_tab_resolution();
        // Grid has a degenerate duplicate at indices 1 and 2.
        let energies = vec![8.0, 10.0, 10.0, 12.0, 50.0, 100.0];
        // Spectrum with NaN exactly at the upper-bracket index (2) that
        // the degenerate pair maps to; any retained (target, kernel-
        // point) entry whose `e_prime` lands inside that duplicate
        // bracket MUST NOT pull the NaN into the output.
        let spectrum = vec![0.1, 0.5, f64::NAN, 0.9, 0.2, 0.05];
        let plan = tab.plan(&energies).unwrap();
        let via_plan = plan.apply(&spectrum);
        let via_reference = broaden_presorted_reference(&tab, &energies, &spectrum);
        // Both paths must agree on the non-pathological targets.  The
        // reference path returns `spectrum[lo]` directly in the
        // degenerate case (no touch of `spectrum[lo+1]`) and the plan
        // path's `frac == 0.0` short-circuit matches bit-exactly.
        // Targets whose kernel legitimately interpolates across index 2
        // will pull the NaN in BOTH paths equally — that's physics, not
        // a bug — so we compare bit-pattern with a NaN-aware helper.
        assert_eq!(via_plan.len(), via_reference.len());
        for (i, (&a, &b)) in via_reference.iter().zip(via_plan.iter()).enumerate() {
            // Both NaN or both finite and bit-exact.
            if a.is_nan() {
                assert!(b.is_nan(), "plan[{i}]={b} but reference is NaN");
            } else {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "nan_safe mismatch at {i}: reference={a} plan={b}"
                );
            }
        }
    }

    #[test]
    fn test_plan_apply_exact_match_frac_plus_zero_propagates_nan() {
        // Regression gate for the subtle P1 Copilot caught on PR #470.
        //
        // When `e_prime` aligns EXACTLY with a grid point `energies[lo]`,
        // `plan_presorted`'s interp fraction computes to `+0.0`, yet the
        // bracket is NOT degenerate (span is a normal positive float).
        // In that case `broaden_presorted` still evaluates
        //   s = spectrum[lo] + (+0.0) * (spectrum[lo+1] - spectrum[lo])
        // which, for `spectrum[lo+1] = NaN`, reads `0.0 * NaN = NaN` and
        // propagates `NaN` into `s`.  The earlier `frac == 0.0` branch
        // in `ResolutionPlan::apply` incorrectly treated this case as
        // degenerate (since `+0.0 == -0.0` under `==`) and short-circuited
        // to `spectrum[lo]`, producing a finite output where the scalar
        // reference produced NaN.
        //
        // Fix: `plan_presorted` now stores `-0.0` (negative-signed zero)
        // for the degenerate sentinel, and `apply` disambiguates via
        // `to_bits()`, so the non-degenerate `+0.0` path correctly reads
        // `spectrum[lo+1]` and propagates NaN.
        let tab = synthetic_tab_resolution();

        // Grid has a point at energy = 10.0.  We engineer a target grid
        // where the broadened kernel at one of the targets produces an
        // `e_prime` that aligns exactly with `energies[lo]` of one of its
        // retained entries.  Achieved by building a coarse target grid
        // and letting the two-pointer walk land on an exact match on at
        // least one (target, kernel-point) pair.
        let energies: Vec<f64> = (0..32).map(|i| 1.0 + i as f64).collect();
        // Spectrum with NaN scattered at multiple lo+1 indices.  At
        // least one retained plan entry in this synthetic configuration
        // will have `frac == +0.0` from an exact-match case, which must
        // propagate NaN in apply.
        let mut spectrum = vec![0.5_f64; energies.len()];
        for v in &mut spectrum[3..] {
            *v = f64::NAN;
        }
        let plan = tab.plan(&energies).unwrap();
        let via_plan = plan.apply(&spectrum);
        let via_reference = broaden_presorted_reference(&tab, &energies, &spectrum);

        // Bit-exact equivalence on all targets, including NaN-propagated
        // ones.  This test would FAIL pre-fix (plan returns finite where
        // reference returns NaN for any exact-match plan entry with a
        // NaN at `lo+1`).
        assert_eq!(via_plan.len(), via_reference.len());
        for (i, (&a, &b)) in via_reference.iter().zip(via_plan.iter()).enumerate() {
            if a.is_nan() {
                assert!(
                    b.is_nan(),
                    "target {i}: reference produced NaN (NaN propagated through \
                     exact-match `frac = +0.0` path) but plan returned finite {b}",
                );
            } else {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "target {i}: reference={a} plan={b}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "must match plan target-grid length")]
    fn test_plan_apply_spectrum_length_mismatch_panics() {
        let tab = synthetic_tab_resolution();
        let plan = tab.plan(&[1.0, 2.0, 3.0]).unwrap();
        // Wrong spectrum length — caller error should panic with a
        // clear message rather than silently producing garbage.
        let _ = plan.apply(&[0.1, 0.2]);
    }

    // ─── apply_resolution_with_plan / _presorted_with_plan dispatch harness ───
    //
    // These gates cover the public/crate-visible wrappers added for
    // production plan-caching.  Every production caller (fit-model
    // layer, spatial dispatch) goes through one of these two entries;
    // their dispatch choices must be byte-identical to the non-plan
    // paths they replace.

    #[test]
    fn test_apply_resolution_with_plan_tabulated_matches_non_plan_path() {
        let tab = synthetic_tab_resolution();
        let resolution = ResolutionFunction::Tabulated(Arc::new(tab.clone()));
        let energies: Vec<f64> = (0..128).map(|i| 1.0 + i as f64 * (200.0 / 128.0)).collect();
        let spectrum: Vec<f64> = energies
            .iter()
            .map(|&e| 1.0 - 0.3 * (-((e - 20.0).powi(2) / 4.0)).exp())
            .collect();

        let baseline = apply_resolution(&energies, &spectrum, &resolution).unwrap();

        let plan = build_resolution_plan(&energies, &resolution).unwrap();
        assert!(
            plan.is_some(),
            "tabulated resolution must produce Some(plan)"
        );
        let planned =
            apply_resolution_with_plan(plan.as_ref(), &energies, &spectrum, &resolution).unwrap();
        assert_eq!(planned.len(), baseline.len());
        for (i, (&a, &b)) in baseline.iter().zip(planned.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "apply_resolution_with_plan mismatch at {i}: baseline={a} planned={b}"
            );
        }
    }

    #[test]
    fn test_apply_resolution_with_plan_gaussian_returns_none_plan_and_matches() {
        let resolution =
            ResolutionFunction::Gaussian(ResolutionParams::new(25.0, 1.0e-3, 0.02, 0.01).unwrap());
        let energies: Vec<f64> = (0..64).map(|i| 1.0 + i as f64 * 3.0).collect();
        let spectrum: Vec<f64> = energies.iter().map(|&e| 1.0 / e).collect();

        let plan = build_resolution_plan(&energies, &resolution).unwrap();
        assert!(
            plan.is_none(),
            "Gaussian resolution must not produce a plan"
        );

        let baseline = apply_resolution(&energies, &spectrum, &resolution).unwrap();
        let planned =
            apply_resolution_with_plan(plan.as_ref(), &energies, &spectrum, &resolution).unwrap();
        assert_eq!(planned.len(), baseline.len());
        for (i, (&a, &b)) in baseline.iter().zip(planned.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "gaussian fallback mismatch at {i}: baseline={a} planned={b}"
            );
        }
    }

    #[test]
    fn test_apply_resolution_with_plan_rejects_same_length_different_grid() {
        // Codex finding: `p.len() == energies.len()` is necessary
        // but not sufficient.  A plan built for one grid and applied
        // to a different same-length grid would silently gather
        // spectrum values at brackets belonging to the original grid
        // — wrong σ output without any error surfaced.  The grid-
        // identity check in `apply_resolution_with_plan` guards this
        // failure mode and reports the first differing index.
        let tab = synthetic_tab_resolution();
        let resolution = ResolutionFunction::Tabulated(Arc::new(tab.clone()));
        let energies_plan: Vec<f64> = (0..32).map(|i| 1.0 + i as f64).collect();
        let mut energies_apply = energies_plan.clone();
        // Perturb a single interior point so lengths still match.
        energies_apply[5] += 0.25;
        let spectrum = vec![0.7; energies_apply.len()];

        let plan = tab.plan(&energies_plan).unwrap();
        let result =
            apply_resolution_with_plan(Some(&plan), &energies_apply, &spectrum, &resolution);
        match result {
            Err(ResolutionError::PlanGridMismatch { first_diff_index }) => {
                assert_eq!(first_diff_index, 5);
            }
            other => panic!("expected PlanGridMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_apply_resolution_with_plan_rejects_length_mismatch() {
        let tab = synthetic_tab_resolution();
        let resolution = ResolutionFunction::Tabulated(Arc::new(tab.clone()));
        let energies_plan: Vec<f64> = (0..32).map(|i| 1.0 + i as f64).collect();
        let energies_apply: Vec<f64> = (0..48).map(|i| 1.0 + i as f64).collect();
        let spectrum = vec![0.5; energies_apply.len()];

        let plan = tab.plan(&energies_plan).unwrap();
        let result =
            apply_resolution_with_plan(Some(&plan), &energies_apply, &spectrum, &resolution);
        match result {
            Err(ResolutionError::LengthMismatch { energies, data }) => {
                assert_eq!(energies, 48);
                assert_eq!(data, 32);
            }
            other => panic!("expected LengthMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_build_resolution_plan_rejects_unsorted_energies_for_gaussian() {
        // Gaussian returns None on success, but must still reject an
        // unsorted grid — callers use `build_resolution_plan` to
        // centralise the sort-check so the downstream `apply` path can
        // skip it.
        let resolution =
            ResolutionFunction::Gaussian(ResolutionParams::new(25.0, 1.0e-3, 0.02, 0.01).unwrap());
        let result = build_resolution_plan(&[3.0, 1.0, 2.0], &resolution);
        assert!(matches!(result, Err(ResolutionError::UnsortedEnergies)));
    }

    #[test]
    #[ignore = "microbenchmark; requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root"]
    fn test_broaden_presorted_bench() {
        let res_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("_fts_bl10_0p5meV_1keV_25pts.txt");
        let text = std::fs::read_to_string(&res_path).expect(
            "missing PLEIADES resolution file at repo root (see `#[ignore]` message for details)",
        );
        let tab = TabulatedResolution::from_text(&text, 25.0).unwrap();

        let n = 3471;
        let energies: Vec<f64> = (0..n)
            .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
            .collect();
        let spectrum: Vec<f64> = energies
            .iter()
            .map(|&e| {
                1.0 - 0.8 * (-((e - 7.8).powi(2) / 0.01)).exp()
                    - 0.6 * (-((e - 22.4).powi(2) / 0.1)).exp()
            })
            .collect();

        let repeats = 30;

        let start = std::time::Instant::now();
        let mut sink_ref = 0.0f64;
        for _ in 0..repeats {
            let r = broaden_presorted_reference(&tab, &energies, &spectrum);
            sink_ref += r.iter().sum::<f64>();
        }
        let t_ref = start.elapsed();

        let start = std::time::Instant::now();
        let mut sink_new = 0.0f64;
        for _ in 0..repeats {
            let r = tab.broaden_presorted(&energies, &spectrum);
            sink_new += r.iter().sum::<f64>();
        }
        let t_new = start.elapsed();

        let speedup = t_ref.as_secs_f64() / t_new.as_secs_f64();
        println!(
            "broaden_presorted microbench (n_grid={n}, repeats={repeats}, 499-pt kernel):\n\
             reference (binary search): {t_ref:?}  (sink={sink_ref:.3})\n\
             two-pointer walk         : {t_new:?}  (sink={sink_new:.3})\n\
             speedup                  : {speedup:.2}x"
        );
        assert_eq!(sink_ref.to_bits(), sink_new.to_bits());
    }

    /// Microbenchmark: plan-reuse path vs per-call `broaden_presorted`.
    ///
    /// This is the payoff the `plan()` + `ResolutionPlan::apply()` API
    /// is designed to deliver: when broadening many spectra on the same
    /// target grid (e.g., LM iterations with fixed TZERO, spatial maps
    /// with pre-calibrated energies), building the plan once and
    /// applying it N times beats rebuilding the plan internally on
    /// every call.
    ///
    /// Run manually with:
    ///
    /// ```text
    /// cargo test --release -p nereids-physics \
    ///   test_plan_reuse_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "microbenchmark; requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root"]
    fn test_plan_reuse_bench() {
        let res_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("_fts_bl10_0p5meV_1keV_25pts.txt");
        let text = std::fs::read_to_string(&res_path).expect(
            "missing PLEIADES resolution file at repo root (see `#[ignore]` message for details)",
        );
        let tab = TabulatedResolution::from_text(&text, 25.0).unwrap();

        let n = 3471;
        let energies: Vec<f64> = (0..n)
            .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
            .collect();

        // Many spectra simulating an LM fit's sequence of evaluations.
        let repeats = 100;
        let mut state: u64 = 0xA5A5_A5A5_DEAD_BEEF;
        let spectra: Vec<Vec<f64>> = (0..repeats)
            .map(|_| {
                energies
                    .iter()
                    .map(|&e| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let noise = ((state >> 33) as f64) / (u32::MAX as f64);
                        1.0 - 0.8 * (-((e - 7.8).powi(2) / 0.01)).exp() + 1e-3 * noise
                    })
                    .collect()
            })
            .collect();

        // Per-call path: same pipeline as today.  Build cost paid every call.
        let start = std::time::Instant::now();
        let mut sink_percall = 0.0f64;
        for spec in &spectra {
            let r = tab.broaden_presorted(&energies, spec);
            sink_percall += r.iter().sum::<f64>();
        }
        let t_percall = start.elapsed();

        // Plan-reuse path: one build, many applies.
        let start = std::time::Instant::now();
        let plan = tab.plan(&energies).expect("sorted grid must validate");
        let t_build = start.elapsed();
        let mut sink_plan = 0.0f64;
        for spec in &spectra {
            let r = plan.apply(spec);
            sink_plan += r.iter().sum::<f64>();
        }
        let t_apply_total = start.elapsed() - t_build;

        let speedup = t_percall.as_secs_f64() / (t_build + t_apply_total).as_secs_f64();
        println!(
            "plan-reuse microbench (n_grid={n}, {repeats} spectra, 499-pt kernel):\n\
             per-call broaden_presorted : {t_percall:?}  (sink={sink_percall:.3})\n\
             plan build (once)          : {t_build:?}\n\
             apply × {repeats}          : {t_apply_total:?}\n\
             total plan path            : {:?}  (sink={sink_plan:.3})\n\
             speedup vs per-call        : {speedup:.2}x",
            t_build + t_apply_total,
        );
        assert_eq!(sink_percall.to_bits(), sink_plan.to_bits());
    }

    // ---------- ResolutionMatrix (CSR compile) tests ----------
    //
    // Two tiers of tests:
    //
    // 1. **CI-hermetic synthetic tests** — use hand-constructed
    //    `ResolutionPlan`s via `make_synthetic_plan`; no fixture
    //    dependency, run on every `cargo test` invocation.  Cover
    //    passthrough rows, `-0.0` sentinel rows, regular
    //    linear-interp rows, CSR invariants, and the non-finite
    //    contract exclusion.
    //
    // 2. **Fixture-dependent tests** (`#[ignore]`) — require
    //    `_fts_bl10_0p5meV_1keV_25pts.txt` at the repo root (a
    //    gitignored PLEIADES file).  Cover end-to-end equivalence
    //    against the production VENUS operator at realistic grid
    //    sizes (512, 3471).  Run locally with `-- --ignored`.  Same
    //    pattern as `test_broaden_presorted_bit_exact_on_pleiades_resolution`
    //    already in this module.

    /// Helper: build a TabulatedResolution + plan + matrix on a
    /// uniform energy grid using the VENUS fixture kernel.  Only used
    /// by `#[ignore]`d tests because the fixture file is gitignored
    /// at the repo root per the "not approved for public release"
    /// policy (.gitignore line 49).
    fn build_fixture_plan_and_matrix(
        n_grid: usize,
    ) -> (Vec<f64>, ResolutionPlan, ResolutionMatrix) {
        let res_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("_fts_bl10_0p5meV_1keV_25pts.txt");
        let text = std::fs::read_to_string(&res_path).expect(
            "missing PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at the repo root \
             (the file is gitignored per policy; place it locally before running this test)",
        );
        let res =
            TabulatedResolution::from_text(&text, 25.0).expect("parse VENUS resolution fixture");
        let energies: Vec<f64> = (0..n_grid)
            .map(|i| 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64))
            .collect();
        let plan = res.plan(&energies).expect("build plan on sorted grid");
        let matrix = plan.compile_to_matrix();
        (energies, plan, matrix)
    }

    /// Hybrid abs+rel tolerance used across equivalence tests.  Guards
    /// against the `a ≈ 0` trap where `a.abs().max(1e-300)` produces
    /// meaningless relative errors for genuinely-zero reference values.
    fn max_hybrid_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let denom = x.abs().max(y.abs()).max(1e-12);
                (x - y).abs() / denom
            })
            .fold(0.0_f64, f64::max)
    }

    /// Build a synthetic multi-row plan with realistic overlap
    /// patterns — used as a CI-hermetic stand-in for the VENUS
    /// kernel.  Each target row `i` draws weights from a triangular
    /// kernel around column `i`, normalized so the row is
    /// row-stochastic.  `half_kernel` controls the spread.
    fn make_synthetic_overlap_plan(n_grid: usize, half_kernel: usize) -> ResolutionPlan {
        assert!(n_grid > 2 * half_kernel, "grid too small for kernel");
        let energies: Vec<f64> = (0..n_grid).map(|i| 10.0 + i as f64).collect();
        let mut rows: Vec<SyntheticRow> = Vec::with_capacity(n_grid);
        for i in 0..n_grid {
            let lo_min = i.saturating_sub(half_kernel);
            // Clamp so `lo ∈ [0, n_grid - 2]` — the linear-interp
            // branch reads `spec[lo + 1]`, and the `-0.0` sentinel is
            // the only way to safely go up to `lo = n_grid - 1`.  We
            // keep all synthetic entries on the regular branch here.
            let lo_max = (i + half_kernel).min(n_grid - 2);
            let entries: Vec<SyntheticEntry> = (lo_min..=lo_max)
                .map(|lo| {
                    let d = (lo as i64 - i as i64).abs() as f64;
                    let w = 1.0 - d / (half_kernel as f64 + 1.0);
                    // A uniform `frac = 0.5` distributes each entry's
                    // weight evenly across `lo` and `lo + 1`, which
                    // exercises the regular linear-interp branch of
                    // `compile_to_matrix`.
                    SyntheticEntry {
                        lo: lo as u32,
                        frac: 0.5,
                        weight: w,
                    }
                })
                .collect();
            let norm: f64 = entries.iter().map(|e| e.weight).sum();
            rows.push(SyntheticRow { entries, norm });
        }
        make_synthetic_plan(energies, rows)
    }

    /// CI-hermetic: row-stochasticity on a synthetic multi-row plan.
    #[test]
    fn resolution_matrix_is_row_stochastic_synthetic() {
        let plan = make_synthetic_overlap_plan(40, 5);
        let matrix = plan.compile_to_matrix();
        for i in 0..matrix.len() {
            let start = matrix.row_starts()[i] as usize;
            let end = matrix.row_starts()[i + 1] as usize;
            let row_sum: f64 = matrix.values()[start..end].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-13,
                "row {} sum = {} (expected 1.0 within 1e-13)",
                i,
                row_sum,
            );
        }
    }

    /// CI-hermetic: equivalence of `apply_r` and `plan.apply` on a
    /// synthetic multi-row plan, 40-point grid, half-kernel 5.
    #[test]
    fn resolution_matrix_apply_equivalent_to_plan_apply_synthetic() {
        let plan = make_synthetic_overlap_plan(40, 5);
        let matrix = plan.compile_to_matrix();
        // Beer-Lambert-shaped synthetic spectrum, bounded [0, 1].
        let spec: Vec<f64> = (0..matrix.len())
            .map(|i| {
                let x = i as f64 / 39.0;
                1.0 - 0.7 * (-((x - 0.5).powi(2)) / 0.01).exp()
            })
            .collect();
        let plan_out = plan.apply(&spec);
        let matrix_out = apply_r(&matrix, &spec);
        let max_err = max_hybrid_err(&plan_out, &matrix_out);
        assert!(
            max_err < 1e-12,
            "synthetic apply_r vs plan.apply max hybrid err = {:.3e} (expected < 1e-12)",
            max_err,
        );
    }

    /// CI-hermetic: CSR column indices strictly ascending per row on
    /// a synthetic multi-row plan.
    #[test]
    fn resolution_matrix_csr_column_indices_sorted_per_row_synthetic() {
        let plan = make_synthetic_overlap_plan(30, 4);
        let matrix = plan.compile_to_matrix();
        for i in 0..matrix.len() {
            let start = matrix.row_starts()[i] as usize;
            let end = matrix.row_starts()[i + 1] as usize;
            let row_cols = &matrix.col_indices()[start..end];
            for w in row_cols.windows(2) {
                assert!(
                    w[0] < w[1],
                    "row {} col_indices not strictly ascending: {:?}",
                    i,
                    row_cols,
                );
            }
        }
    }

    /// CI-hermetic: grid-mismatch / length-mismatch detection via
    /// `apply_resolution_with_matrix` on a synthetic plan.
    #[test]
    fn resolution_matrix_grid_and_length_mismatch_synthetic() {
        let plan = make_synthetic_overlap_plan(16, 3);
        let matrix = plan.compile_to_matrix();
        let n = matrix.len();
        let energies: Vec<f64> = (0..n).map(|i| 10.0 + i as f64).collect();
        let spec = vec![1.0_f64; n];

        // Same grid + length → passes.
        assert!(apply_resolution_with_matrix(&energies, &matrix, &spec).is_ok());

        // Perturb one energy → MatrixGridMismatch with offending
        // index.
        let mut mutated = energies.clone();
        mutated[7] += 1e-12;
        let err = apply_resolution_with_matrix(&mutated, &matrix, &spec)
            .expect_err("grid mismatch must error");
        assert_eq!(
            err,
            ResolutionError::MatrixGridMismatch {
                first_diff_index: 7,
            }
        );

        // Short spectrum → LengthMismatch.
        let short = vec![1.0_f64; n - 1];
        let err = apply_resolution_with_matrix(&energies, &matrix, &short)
            .expect_err("length mismatch must error");
        assert!(matches!(err, ResolutionError::LengthMismatch { .. }));
    }

    /// End-to-end bit-level equivalence on the real VENUS kernel,
    /// 512-point grid.  Gated on the PLEIADES fixture per the
    /// established `#[ignore]` pattern in this module.
    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn resolution_matrix_is_row_stochastic_on_venus_kernel() {
        let (_energies, _plan, matrix) = build_fixture_plan_and_matrix(512);
        for i in 0..matrix.len() {
            let start = matrix.row_starts()[i] as usize;
            let end = matrix.row_starts()[i + 1] as usize;
            let row_sum: f64 = matrix.values()[start..end].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-13,
                "row {} sum = {} (expected 1.0 within 1e-13)",
                i,
                row_sum,
            );
        }
    }

    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn resolution_matrix_apply_equivalent_to_plan_apply_on_venus_kernel() {
        let (_energies, plan, matrix) = build_fixture_plan_and_matrix(512);
        let n_grid = matrix.len();
        let spec: Vec<f64> = (0..n_grid)
            .map(|i| {
                let e = 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64);
                let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                    + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
                (-1.6e-4 * sigma).exp()
            })
            .collect();
        let plan_out = plan.apply(&spec);
        let matrix_out = apply_r(&matrix, &spec);
        let max_err = max_hybrid_err(&plan_out, &matrix_out);
        assert!(
            max_err < 1e-12,
            "apply_r vs plan.apply max hybrid err = {:.3e} (expected < 1e-12)",
            max_err,
        );
    }

    /// Production-grid guardrail for the `1e-12` tolerance documented
    /// on [`ResolutionPlan::compile_to_matrix`].  The 3471-bin VENUS
    /// grid has ~82 entries per row, so accumulation error is an
    /// order of magnitude larger than on the synthetic multi-row
    /// tests above; this test pins the equivalence bound at
    /// production scale so a future regression in either `apply` or
    /// `apply_r` summation order fails loudly.  Logs the observed
    /// `max_hybrid_err` via `eprintln!` so `-- --ignored --nocapture`
    /// runs surface the actual headroom against the 1e-12 ceiling.
    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn resolution_matrix_apply_equivalent_at_production_grid() {
        let (_energies, plan, matrix) = build_fixture_plan_and_matrix(3471);
        let n_grid = matrix.len();
        // Same Beer-Lambert test spectrum as the 512-point test.
        let spec: Vec<f64> = (0..n_grid)
            .map(|i| {
                let e = 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64);
                let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                    + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
                (-1.6e-4 * sigma).exp()
            })
            .collect();
        let plan_out = plan.apply(&spec);
        let matrix_out = apply_r(&matrix, &spec);
        let max_err = max_hybrid_err(&plan_out, &matrix_out);
        eprintln!(
            "3471-grid apply_r vs plan.apply observed max_hybrid_err = {:.3e} \
             (ceiling 1e-12; theoretical bound ~1e-13 per row × 82 rows/entry)",
            max_err,
        );
        assert!(
            max_err < 1e-12,
            "3471-grid apply_r vs plan.apply max hybrid err = {:.3e} (expected < 1e-12)",
            max_err,
        );
    }

    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn resolution_matrix_apply_equivalent_across_densities() {
        let (_energies, plan, matrix) = build_fixture_plan_and_matrix(512);
        let n_grid = matrix.len();
        for &n_density in &[1e-5_f64, 1e-4, 1.6e-4, 1e-3] {
            let spec: Vec<f64> = (0..n_grid)
                .map(|i| {
                    let e = 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64);
                    let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                        + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
                    (-n_density * sigma).exp()
                })
                .collect();
            let plan_out = plan.apply(&spec);
            let matrix_out = apply_r(&matrix, &spec);
            let max_err = max_hybrid_err(&plan_out, &matrix_out);
            assert!(
                max_err < 1e-12,
                "density n={:.1e}: max hybrid err {:.3e} (expected < 1e-12)",
                n_density,
                max_err,
            );
        }
    }

    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn resolution_matrix_csr_column_indices_sorted_per_row() {
        let (_energies, _plan, matrix) = build_fixture_plan_and_matrix(256);
        for i in 0..matrix.len() {
            let start = matrix.row_starts()[i] as usize;
            let end = matrix.row_starts()[i + 1] as usize;
            let row_cols = &matrix.col_indices()[start..end];
            for w in row_cols.windows(2) {
                assert!(
                    w[0] < w[1],
                    "row {} col_indices not strictly ascending: {:?}",
                    i,
                    row_cols,
                );
            }
        }
    }

    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn resolution_matrix_grid_mismatch_detected() {
        let (energies, _plan, matrix) = build_fixture_plan_and_matrix(128);
        let spec = vec![1.0_f64; matrix.len()];

        // Same grid → passes.
        let ok = apply_resolution_with_matrix(&energies, &matrix, &spec);
        assert!(ok.is_ok());

        // Perturb one energy → MatrixGridMismatch with the
        // offending index.
        let mut mutated = energies.clone();
        mutated[37] += 1e-12;
        let err = apply_resolution_with_matrix(&mutated, &matrix, &spec)
            .expect_err("grid mismatch must error");
        assert_eq!(
            err,
            ResolutionError::MatrixGridMismatch {
                first_diff_index: 37,
            }
        );
    }

    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn resolution_matrix_length_mismatch_detected() {
        let (energies, _plan, matrix) = build_fixture_plan_and_matrix(64);
        let short_spec = vec![1.0_f64; matrix.len() - 1];
        let err = apply_resolution_with_matrix(&energies, &matrix, &short_spec)
            .expect_err("length mismatch must error");
        assert!(matches!(err, ResolutionError::LengthMismatch { .. }));
    }

    #[test]
    fn resolution_matrix_empty_plan() {
        // Compile must not panic and must produce a valid empty
        // matrix when the plan itself is empty.  Build the empty
        // plan synthetically (no fixture needed) — an empty
        // `target_energies` plus empty `norm` / `starts = [0]`
        // yields the same zero-row plan that
        // `TabulatedResolution::plan(&[])` would produce.
        let plan = make_synthetic_plan(Vec::new(), Vec::new());
        let matrix = plan.compile_to_matrix();
        assert_eq!(matrix.len(), 0);
        assert!(matrix.is_empty());
        assert_eq!(matrix.nnz(), 0);
    }

    /// Microbenchmark: `apply_r` (ResolutionMatrix CSR) vs
    /// `ResolutionPlan::apply`, 3471-bin VENUS production grid × 100
    /// spectra. Exercised manually to decide whether the CSR compile +
    /// CSR matvec beats the plan's two-pointer walk at the
    /// no-SIMD-no-unsafe baseline promised in #473.
    ///
    /// Run manually with:
    ///
    /// ```text
    /// cargo test --release -p nereids-physics \
    ///   resolution_matrix_apply_microbench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "microbenchmark; requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root"]
    fn resolution_matrix_apply_microbench() {
        let res_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("_fts_bl10_0p5meV_1keV_25pts.txt");
        let text = std::fs::read_to_string(&res_path).expect(
            "missing PLEIADES resolution file at repo root (see `#[ignore]` message for details)",
        );
        let tab = TabulatedResolution::from_text(&text, 25.0).unwrap();

        let n = 3471_usize;
        let energies: Vec<f64> = (0..n)
            .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
            .collect();
        let plan = tab.plan(&energies).expect("sorted grid must validate");

        let t_compile = std::time::Instant::now();
        let matrix = plan.compile_to_matrix();
        let t_compile = t_compile.elapsed();

        let spec: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                    + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
                (-1.6e-4 * sigma).exp()
            })
            .collect();

        let repeats = 100_usize;

        // Warm both paths so the first call's cache-miss latency
        // does not skew the micro-times.
        for _ in 0..5 {
            let _ = plan.apply(&spec);
            let _ = apply_r(&matrix, &spec);
        }

        let start = std::time::Instant::now();
        let mut sink_plan = 0.0f64;
        for _ in 0..repeats {
            sink_plan += plan.apply(&spec).iter().sum::<f64>();
        }
        let t_plan = start.elapsed();

        let start = std::time::Instant::now();
        let mut sink_matrix = 0.0f64;
        for _ in 0..repeats {
            sink_matrix += apply_r(&matrix, &spec).iter().sum::<f64>();
        }
        let t_matrix = start.elapsed();

        let speedup = t_plan.as_secs_f64() / t_matrix.as_secs_f64();
        println!(
            "ResolutionMatrix microbench (n_grid={n}, {repeats} spectra):\n\
             compile (once)       : {:?}  ({} nnz)\n\
             plan.apply × {repeats} : {:?}\n\
             apply_r   × {repeats} : {:?}\n\
             speedup vs plan      : {:.2}x\n\
             sinks (plan/matrix)  : {:.6e} / {:.6e}",
            t_compile,
            matrix.nnz(),
            t_plan,
            t_matrix,
            speedup,
            sink_plan,
            sink_matrix,
        );
    }

    /// Hand-construct a `ResolutionPlan` that deliberately exercises
    /// both the passthrough branch (`norm ≤ DIVISION_FLOOR`) and the
    /// `-0.0` degenerate-bracket sentinel — neither of which is
    /// reached on the VENUS fixture at the tested grid sizes.  The
    /// Round-1 audit flagged the earlier fixture-based passthrough
    /// test as vacuous, so this replacement verifies the two unreached
    /// branches with direct assertions on the resulting CSR.
    fn make_synthetic_plan(target_energies: Vec<f64>, rows: Vec<SyntheticRow>) -> ResolutionPlan {
        let n = target_energies.len();
        assert_eq!(rows.len(), n);
        let mut starts: Vec<u32> = Vec::with_capacity(n + 1);
        starts.push(0);
        let mut lo_idx: Vec<u32> = Vec::new();
        let mut frac: Vec<f64> = Vec::new();
        let mut weight: Vec<f64> = Vec::new();
        let mut norm: Vec<f64> = Vec::with_capacity(n);
        for row in &rows {
            norm.push(row.norm);
            for entry in &row.entries {
                lo_idx.push(entry.lo);
                frac.push(entry.frac);
                weight.push(entry.weight);
            }
            starts.push(lo_idx.len() as u32);
        }
        ResolutionPlan {
            target_energies,
            starts,
            lo_idx,
            frac,
            weight,
            norm,
        }
    }

    struct SyntheticRow {
        entries: Vec<SyntheticEntry>,
        norm: f64,
    }

    struct SyntheticEntry {
        lo: u32,
        frac: f64,
        weight: f64,
    }

    #[test]
    fn resolution_matrix_passthrough_row_compiles_to_identity_entry() {
        // Row 0: passthrough via norm ≤ DIVISION_FLOOR.
        // Row 1: regular linear-interp entry (lo=1 → reads cols 1, 2).
        // Row 2: degenerate `-0.0` sentinel entry (lo=2 → reads col 2 only).
        //
        // Grid has 4 cells so `lo ∈ [0, n-2] = [0, 2]` holds for all
        // entries — this preserves the `ResolutionPlan::apply` SAFETY
        // invariant that `lo + 1 < n` even if a future refactor
        // weakens the `-0.0` sentinel short-circuit (round-2 self-
        // audit NEW-P2 #1).
        let plan = make_synthetic_plan(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![
                SyntheticRow {
                    entries: vec![],
                    // 0.0 is <= DIVISION_FLOOR, so row 0 goes through
                    // the passthrough branch.
                    norm: 0.0,
                },
                SyntheticRow {
                    entries: vec![SyntheticEntry {
                        lo: 1,
                        frac: 0.25,
                        weight: 1.0,
                    }],
                    norm: 1.0,
                },
                SyntheticRow {
                    entries: vec![SyntheticEntry {
                        lo: 2,
                        frac: -0.0,
                        weight: 1.0,
                    }],
                    norm: 1.0,
                },
                // Row 3: passthrough too, to round out the 4-cell grid.
                SyntheticRow {
                    entries: vec![],
                    norm: 0.0,
                },
            ],
        );
        let matrix = plan.compile_to_matrix();

        // Row 0 — single (0, 0, 1.0).
        let r0_start = matrix.row_starts()[0] as usize;
        let r0_end = matrix.row_starts()[1] as usize;
        assert_eq!(r0_end - r0_start, 1, "passthrough row must have 1 entry");
        assert_eq!(matrix.col_indices()[r0_start], 0);
        assert_eq!(matrix.values()[r0_start].to_bits(), 1.0_f64.to_bits());

        // Row 1 — linear-interp: contributes at col 1 and col 2.
        let r1_start = matrix.row_starts()[1] as usize;
        let r1_end = matrix.row_starts()[2] as usize;
        assert_eq!(
            r1_end - r1_start,
            2,
            "linear-interp row must have 2 entries"
        );
        assert_eq!(matrix.col_indices()[r1_start], 1);
        assert_eq!(matrix.col_indices()[r1_start + 1], 2);
        assert!((matrix.values()[r1_start] - 0.75).abs() < 1e-14);
        assert!((matrix.values()[r1_start + 1] - 0.25).abs() < 1e-14);

        // Row 2 — `-0.0` sentinel: single entry at col 2 (no col 3).
        let r2_start = matrix.row_starts()[2] as usize;
        let r2_end = matrix.row_starts()[3] as usize;
        assert_eq!(
            r2_end - r2_start,
            1,
            "-0.0 sentinel row must have exactly 1 entry (not 2)",
        );
        assert_eq!(matrix.col_indices()[r2_start], 2);
        assert_eq!(matrix.values()[r2_start].to_bits(), 1.0_f64.to_bits());

        // Cross-check with apply semantics: spec[3] is chosen so the
        // sentinel row, if buggy, would contaminate the output.
        // Both `plan.apply` and `apply_r` must ignore spec[3] at
        // row 2.
        let spec = vec![7.0, 11.0, 13.0, 999.0];
        let plan_out = plan.apply(&spec);
        let matrix_out = apply_r(&matrix, &spec);
        // Row 0 passthrough: out[0] = spec[0] = 7.
        assert!((matrix_out[0] - 7.0).abs() < 1e-14);
        assert!((plan_out[0] - 7.0).abs() < 1e-14);
        // Row 1: 0.75 * spec[1] + 0.25 * spec[2] = 0.75*11 + 0.25*13 = 11.5.
        assert!((matrix_out[1] - 11.5).abs() < 1e-14);
        assert!((plan_out[1] - 11.5).abs() < 1e-14);
        // Row 2 sentinel: 1.0 * spec[2] = 13 — NOT 999 (would indicate
        // spec[lo+1] was read).
        assert!((matrix_out[2] - 13.0).abs() < 1e-14);
        assert!((plan_out[2] - 13.0).abs() < 1e-14);
        // Row 3 passthrough: out[3] = spec[3] = 999.
        assert!((matrix_out[3] - 999.0).abs() < 1e-14);
        assert!((plan_out[3] - 999.0).abs() < 1e-14);
    }

    /// Documents (and guards) the explicit contract exclusion on
    /// non-finite spectra between `ResolutionPlan::apply` and
    /// `apply_r`.  See [`ResolutionPlan::compile_to_matrix`] docstring
    /// for the full reasoning; this test simply pins the divergence
    /// so a future unification attempt fails loudly.
    #[test]
    fn resolution_matrix_nonfinite_contract() {
        // 3-cell grid so `lo = 0` for the regular row reads cols 0, 1
        // and the sentinel row at `lo = 1` reads col 1 only — `lo ∈
        // [0, n-2] = [0, 1]` satisfied.
        let plan = make_synthetic_plan(
            vec![10.0, 20.0, 30.0],
            vec![
                SyntheticRow {
                    entries: vec![SyntheticEntry {
                        lo: 0,
                        frac: 0.5,
                        weight: 1.0,
                    }],
                    norm: 1.0,
                },
                SyntheticRow {
                    entries: vec![SyntheticEntry {
                        lo: 1,
                        frac: -0.0, // sentinel: short-circuit to spec[lo]
                        weight: 1.0,
                    }],
                    norm: 1.0,
                },
                SyntheticRow {
                    entries: vec![],
                    norm: 0.0, // passthrough
                },
            ],
        );
        let matrix = plan.compile_to_matrix();

        // Spectrum with same-sign infinities in both bins of row 0's
        // non-degenerate bracket.
        let inf_spec = vec![f64::INFINITY, f64::INFINITY, 0.0];
        let plan_out = plan.apply(&inf_spec);
        let matrix_out = apply_r(&matrix, &inf_spec);

        // Row 0: plan.apply evaluates `s_lo + frac * (s_hi - s_lo)`
        // = `+∞ + 0.5 * (+∞ - +∞)` = `+∞ + 0.5 * NaN` = NaN.
        // apply_r evaluates `0.5 * +∞ + 0.5 * +∞` = `+∞`.
        assert!(plan_out[0].is_nan(), "plan.apply must produce NaN on ∞+∞");
        assert!(matrix_out[0].is_infinite(), "apply_r collapses ∞+∞ to ∞");

        // Row 1 (sentinel): both paths short-circuit to spec[lo] = ∞,
        // so there is no divergence here.
        assert!(plan_out[1].is_infinite());
        assert!(matrix_out[1].is_infinite());
    }

    /// Round-2 Codex P3: documents (and guards) the analogous
    /// divergence on **finite spectra near f64 overflow**.  With
    /// opposite-sign neighboring bins at f64::MAX, `plan.apply`'s
    /// `s_lo + frac * (s_hi - s_lo)` overflows in the subtraction
    /// and returns `±∞`, while `apply_r`'s `(1 - frac) * s_lo +
    /// frac * s_hi` stays finite because the overflow is avoided by
    /// scaling before summation.  This is why the equivalence
    /// contract on [`ResolutionPlan::compile_to_matrix`] is scoped
    /// to bounded finite spectra (Beer-Lambert `T ∈ [0, 1]`) — no
    /// production forward model can hit this case.
    #[test]
    fn resolution_matrix_large_finite_contract() {
        let plan = make_synthetic_plan(
            vec![10.0, 20.0, 30.0],
            vec![
                SyntheticRow {
                    entries: vec![SyntheticEntry {
                        lo: 0,
                        frac: 0.5,
                        weight: 1.0,
                    }],
                    norm: 1.0,
                },
                SyntheticRow {
                    entries: vec![],
                    norm: 0.0, // passthrough
                },
                SyntheticRow {
                    entries: vec![],
                    norm: 0.0,
                },
            ],
        );
        let matrix = plan.compile_to_matrix();

        // Opposite-sign large finite bins at row 0's non-degenerate
        // bracket.  `s_hi - s_lo = -f64::MAX - f64::MAX = -∞`.
        let big_spec = vec![f64::MAX, -f64::MAX, 0.0];
        let plan_out = plan.apply(&big_spec);
        let matrix_out = apply_r(&matrix, &big_spec);

        // plan.apply: s_lo + frac * (s_hi - s_lo) = MAX + 0.5 * (-∞)
        // = MAX + -∞ = -∞.
        assert!(
            plan_out[0].is_infinite() && plan_out[0] < 0.0,
            "plan.apply must overflow to -∞ on opposite-sign MAX bins; got {}",
            plan_out[0],
        );
        // apply_r: 0.5 * MAX + 0.5 * -MAX = 0.
        assert!(
            matrix_out[0].is_finite(),
            "apply_r must stay finite (scaled before summation); got {}",
            matrix_out[0],
        );
        assert!(matrix_out[0].abs() < 1e-280);
    }
}
