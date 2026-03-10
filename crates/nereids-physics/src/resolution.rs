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
    pub(crate) fn broaden_presorted(&self, energies: &[f64], spectrum: &[f64]) -> Vec<f64> {
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

            // TOF at this energy: t = TOF_FACTOR * L / sqrt(E)
            let tof_center = TOF_FACTOR * self.flight_path_m / e.sqrt();

            // Compute interpolated kernel on the fly to avoid O(N * kernel_len) memory
            let (offsets, weights) = self.interpolated_kernel(e);

            // Convolve: for each kernel point, find the energy corresponding to
            // tof_center + dt_offset, then interpolate spectrum at that energy.
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

                // Convert TOF to energy: E' = (TOF_FACTOR * L / t')^2
                let e_prime = (TOF_FACTOR * self.flight_path_m / tof_prime).powi(2);

                // Interpolate spectrum at e_prime; skip if outside the grid
                let s = match interp_spectrum(energies, spectrum, e_prime) {
                    Some(v) => v,
                    None => continue,
                };

                // Trapezoidal weight for the TOF integral
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

/// Linear interpolation of spectrum at an arbitrary energy.
///
/// Returns `None` if `e` is outside the grid range, so callers can
/// exclude off-grid kernel samples instead of clamping to boundary values.
fn interp_spectrum(energies: &[f64], spectrum: &[f64], e: f64) -> Option<f64> {
    let n = energies.len();
    if n == 0 {
        return None;
    }
    if e < energies[0] || e > energies[n - 1] {
        return None;
    }

    // Binary search for bracketing index
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

    // Guard: if energies[hi] == energies[lo] (duplicate grid points or
    // single-point grid where lo==hi), the denominator is zero.  Return the
    // value at the lower bracket to avoid NaN.
    let span = energies[hi] - energies[lo];
    if span.abs() < NEAR_ZERO_FLOOR {
        return Some(spectrum[lo]);
    }
    let frac = (e - energies[lo]) / span;
    Some(spectrum[lo] + frac * (spectrum[hi] - spectrum[lo]))
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
}
