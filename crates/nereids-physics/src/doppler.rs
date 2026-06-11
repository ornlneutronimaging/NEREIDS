//! Doppler broadening via the Free Gas Model (FGM).
//!
//! The FGM treats target atoms as a free ideal gas at temperature T.
//! The Doppler-broadened cross-section is obtained by averaging the
//! unbroadened cross-section over the Maxwell-Boltzmann velocity
//! distribution of the target atoms.
//!
//! ## SAMMY Reference
//! - Manual Section III.B.1 (Free-Gas Model of Doppler Broadening)
//! - `fgm/mfgm1.f90` subroutine `Dopfgm` (quadrature in `mfgm2.f90`
//!   Modsmp/Modfpl)
//!
//! ## Method
//!
//! We implement the exact FGM integral in velocity space (SAMMY
//! Eq. III B1.7), including its w/v integrand weight:
//!
//!   v²·σ_D(v²) = (1/(u√π)) ∫ exp(-(v-w)²/u²) · w² · s(w) dw
//!
//! where v = √E, u = √(k_B·T / AWR), and:
//!   s(w) =  σ(w²)  for w > 0
//!   s(w) = -σ(w²)  for w < 0
//!
//! This matches SAMMY's `Dopfgm`, which multiplies the normalized Gaussian
//! quadrature weights by w² and divides the integral by E = v²
//! (`fgm/mfgm2.f90` Modsmp/Modfpl `Wts·Velcty**2`, `mfgm4.f90` `val/Em`).
//! Two analytic consequences (both pinned by
//! `kernel_error_scales_pinned_vs_full_fgm_reference`): a constant σ is
//! broadened to σ·(1 + u²/2v²) — the physical low-energy upturn — and a
//! 1/v cross-section is preserved exactly.  (An earlier revision omitted
//! the w/v weight, which skewed Doppler-broadened resonance flanks by a
//! first-order ~u/v; the pinning test fails loudly on any regression to
//! that kernel.)
//!
//! The key advantage of the velocity-space formulation is that u is
//! independent of energy, making it a true convolution.
//!
//! ## Doppler Width
//!
//! The SAMMY Doppler width at energy E is:
//!   Δ_D(E) = √(4·k_B·T·E / AWR)

use std::fmt;

use nereids_core::constants::{self, DIVISION_FLOOR, NEAR_ZERO_FLOOR};

use crate::resolution::exerfc;

/// Number of standard deviations beyond the velocity range for the FGM
/// integration window.  The Gaussian kernel exp(-arg²) contributes less
/// than exp(-36) ≈ 2.3e-16 outside this window, which is below f64
/// machine epsilon.
const DOPPLER_N_SIGMA: f64 = 6.0;

/// Floor for distinguishing negative-velocity grid points from zero.
///
/// When building the extended velocity grid for the FGM integral, we
/// generate points from `v_neg_limit` up to (but not including) zero.
/// This threshold prevents the last negative-velocity point from being
/// so close to zero that it is numerically indistinguishable, which would
/// create a near-duplicate of the explicit v = 0 anchor point.
const NEGATIVE_VELOCITY_FLOOR: f64 = 1e-15;

/// Errors from `DopplerParams` construction.
#[derive(Debug, PartialEq)]
pub enum DopplerParamsError {
    /// AWR must be strictly positive.
    InvalidAwr(f64),
    /// Temperature must be finite (may be zero for "no broadening").
    NonFiniteTemperature(f64),
    /// Temperature must be non-negative (negative Kelvin is physically meaningless).
    NegativeTemperature(f64),
}

impl fmt::Display for DopplerParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAwr(v) => write!(f, "AWR must be positive, got {v}"),
            Self::NonFiniteTemperature(v) => write!(f, "temperature must be finite, got {v}"),
            Self::NegativeTemperature(v) => {
                write!(f, "temperature must be non-negative, got {v}")
            }
        }
    }
}

impl std::error::Error for DopplerParamsError {}

/// Errors from Doppler broadening computation (not parameter construction).
///
/// Marked `#[non_exhaustive]` because this enum is publicly exported from
/// `nereids-physics` and may grow new validation variants over time (e.g. if
/// future contracts add bounds on AWR/energy combinations). Without the
/// attribute, adding a variant would be a SemVer-breaking change for any
/// downstream crate that exhaustively matches on `DopplerError`.
#[derive(Debug)]
#[non_exhaustive]
pub enum DopplerError {
    /// Energy and cross-section arrays have different lengths.
    LengthMismatch {
        /// Number of energy points.
        energies: usize,
        /// Number of cross-section values.
        cross_sections: usize,
    },
    /// An energy value is non-finite (NaN/±∞) or non-positive (≤ 0).
    ///
    /// The FGM velocity transform computes `v = √E`, so non-positive or
    /// non-finite energies produce NaN velocities that silently propagate
    /// through the convolution. Per-point guards in the convolution loop
    /// rely on `v < FLOOR` comparisons which evaluate to `false` for NaN
    /// (see "NaN bypasses guards" project convention), so the function
    /// would return wrong outputs rather than erroring. The contract is
    /// "every energy is finite and strictly positive."
    InvalidEnergy {
        /// Position in the energy array where the bad value was found.
        index: usize,
        /// The offending energy value.
        value: f64,
    },
    /// The energy grid is not strictly increasing at `index`.
    ///
    /// `doppler_broaden` uses `partition_point` over the extended velocity
    /// grid (built from `energies` via `v = √E`), which has an unspecified
    /// return value on an unsorted slice and therefore would silently
    /// produce garbage indices in release builds. The contract is
    /// "energies are strictly ascending"; duplicate points are also rejected.
    UnsortedEnergies {
        /// Position where the strict-ascending invariant was first violated.
        index: usize,
        /// The previous (smaller-index) energy value.
        previous: f64,
        /// The current (larger-index) energy value that broke the invariant.
        current: f64,
    },
}

impl fmt::Display for DopplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                energies,
                cross_sections,
            } => write!(
                f,
                "energies length ({energies}) must match cross_sections length ({cross_sections})"
            ),
            Self::InvalidEnergy { index, value } => write!(
                f,
                "energies[{index}] = {value} is not finite or not strictly positive (Doppler broadening requires every energy to satisfy is_finite() && > 0)"
            ),
            Self::UnsortedEnergies {
                index,
                previous,
                current,
            } => write!(
                f,
                "energies[{index}] = {current} is not strictly greater than energies[{}] = {previous} (Doppler broadening requires the energy grid to be strictly ascending)",
                index.saturating_sub(1)
            ),
        }
    }
}

impl std::error::Error for DopplerError {}

/// Validate that `energies` satisfies the Doppler-broadening grid contract:
/// every entry is finite, strictly positive, and strictly greater than the
/// previous entry. An empty slice is permitted (the caller has its own
/// length-handling fast path).
///
/// The check is O(n) and is run unconditionally on every entry to
/// `doppler_broaden` and `doppler_broaden_with_derivative` so that
/// malformed grids surface as a typed `Err` rather than silent NaN
/// propagation or unspecified `partition_point` behaviour.
fn validate_doppler_grid(energies: &[f64]) -> Result<(), DopplerError> {
    for (i, &e) in energies.iter().enumerate() {
        if !e.is_finite() || e <= 0.0 {
            return Err(DopplerError::InvalidEnergy { index: i, value: e });
        }
        if i > 0 {
            // Safe to use `e <= prev` here: the `is_finite()` check above
            // already rejected any NaN entries, so the partial-ord comparison
            // is total. (NaN comparisons returning false would otherwise
            // silently let NaN through this branch.)
            let prev = energies[i - 1];
            if e <= prev {
                return Err(DopplerError::UnsortedEnergies {
                    index: i,
                    previous: prev,
                    current: e,
                });
            }
        }
    }
    Ok(())
}

/// Doppler broadening parameters.
#[derive(Debug, Clone, Copy)]
pub struct DopplerParams {
    /// Effective sample temperature in Kelvin.
    temperature_k: f64,
    /// Atomic weight ratio (target mass / neutron mass) from ENDF.
    awr: f64,
}

impl DopplerParams {
    /// Create validated Doppler parameters.
    ///
    /// # Errors
    /// Returns `DopplerParamsError::InvalidAwr` if `awr <= 0.0` or is NaN.
    /// Returns `DopplerParamsError::NonFiniteTemperature` if `temperature_k`
    /// is NaN or infinity.
    /// Returns `DopplerParamsError::NegativeTemperature` if `temperature_k < 0.0`.
    /// Zero temperature is allowed — it means "no broadening".
    pub fn new(temperature_k: f64, awr: f64) -> Result<Self, DopplerParamsError> {
        if !awr.is_finite() || awr <= 0.0 {
            return Err(DopplerParamsError::InvalidAwr(awr));
        }
        if !temperature_k.is_finite() {
            return Err(DopplerParamsError::NonFiniteTemperature(temperature_k));
        }
        if temperature_k < 0.0 {
            return Err(DopplerParamsError::NegativeTemperature(temperature_k));
        }
        Ok(Self { temperature_k, awr })
    }

    /// Returns the effective sample temperature in Kelvin.
    #[must_use]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Returns the atomic weight ratio (target mass / neutron mass).
    #[must_use]
    pub fn awr(&self) -> f64 {
        self.awr
    }

    /// Velocity-space Doppler width u = √(k_B·T / AWR).
    ///
    /// This is the standard deviation of the Gaussian kernel in √eV units.
    #[must_use]
    pub fn u(&self) -> f64 {
        (constants::BOLTZMANN_EV_PER_K * self.temperature_k / self.awr).sqrt()
    }

    /// Energy-dependent Doppler width Δ_D(E) = √(4·k_B·T·E / AWR).
    ///
    /// This is the width that SAMMY reports in the .lpt file.
    #[must_use]
    pub fn doppler_width(&self, energy_ev: f64) -> f64 {
        (4.0 * constants::BOLTZMANN_EV_PER_K * self.temperature_k * energy_ev / self.awr).sqrt()
    }
}

/// √π constant for erfc computation.
const SQRT_PI: f64 = 1.772_453_850_905_516;

/// Complementary error function erfc(x) = 1 - erf(x).
///
/// For x ≥ 0: uses the scaled complementary error function `exerfc`
/// (SAMMY `fnc/exerfc.f90`):
///   erfc(x) = exp(-x²) · exerfc(x) / √π
///
/// For x < 0: uses the identity erfc(-|x|) = 2 - erfc(|x|) to avoid
/// the `exerfc` negative-argument branch, which has a numerical issue
/// for |x| > 5.01 (missing exp(x²) factor in the large-|x| path).
fn erfc_val(x: f64) -> f64 {
    if x >= 0.0 {
        (-x * x).exp() * exerfc(x) / SQRT_PI
    } else {
        let xp = -x;
        2.0 - (-xp * xp).exp() * exerfc(xp) / SQRT_PI
    }
}

/// Apply FGM Doppler broadening to cross-section data.
///
/// The cross-sections are broadened in velocity space using the exact
/// Free Gas Model integral from SAMMY manual Eq. III B1.7 (w²-weighted
/// integrand; see the module docs).
///
/// # Arguments
/// * `energies` — Energy grid in eV. Every entry must satisfy
///   `is_finite() && > 0.0`, and the grid must be **strictly ascending**
///   (duplicates are rejected). The contract is enforced at the public
///   boundary by `validate_doppler_grid`.
/// * `cross_sections` — Unbroadened cross-sections in barns at each energy point.
/// * `params` — Doppler broadening parameters (temperature and AWR).
///
/// # Returns
/// Doppler-broadened cross-sections in barns on the same energy grid.
///
/// # Errors
/// * `DopplerError::LengthMismatch` if `energies.len() != cross_sections.len()`.
/// * `DopplerError::InvalidEnergy` if any energy is non-finite or ≤ 0.
/// * `DopplerError::UnsortedEnergies` if the grid is not strictly ascending.
///
/// # Algorithm
/// 1. Convert energy grid to velocity space (v = √E).
/// 2. Build extended grid including negative velocities for the FGM integral.
/// 3. Compute the integrand Y(w) = w² · s(w) on the extended grid.
/// 4. For each output velocity, evaluate the Gaussian convolution integral.
/// 5. Transform back: σ_D(E) = result / E.
pub fn doppler_broaden(
    energies: &[f64],
    cross_sections: &[f64],
    params: &DopplerParams,
) -> Result<Vec<f64>, DopplerError> {
    if energies.len() != cross_sections.len() {
        return Err(DopplerError::LengthMismatch {
            energies: energies.len(),
            cross_sections: cross_sections.len(),
        });
    }

    // Validate the energy-grid contract before any sqrt / partition_point /
    // interpolation work. Without this guard, NaN energies would silently
    // produce NaN velocities (and the per-point `v < FLOOR` check evaluates
    // to false for NaN, allowing NaN to enter the convolution kernel), and
    // unsorted grids would give unspecified `partition_point` indices.
    validate_doppler_grid(energies)?;

    if params.temperature_k() <= 0.0 || energies.is_empty() {
        return Ok(cross_sections.to_vec());
    }

    let u = params.u();
    if u < NEAR_ZERO_FLOOR {
        return Ok(cross_sections.to_vec());
    }

    let n = energies.len();

    // Convert to velocity grid: v_i = sqrt(E_i)
    let velocities: Vec<f64> = energies.iter().map(|&e| e.sqrt()).collect();

    // Build the integrand Y(w) = w² * s(w) on the velocity grid
    // (Eq. III B1.7 with the w/v weight folded in; the 1/v is applied at
    // the end as the 1/E division).
    // From Eq. III B1.6: s(w) = σ(w²) for w>0, s(w) = -σ(w²) for w<0, so
    //   Y(w) =  w² * σ(w²)  for w > 0
    //   Y(w) = -w² * σ(w²)  for w < 0
    // i.e. Y is an ODD function passing smoothly through Y(0) = 0.

    // Determine how many negative velocity points we need.
    // We need points down to v_min - N_sigma * u, which may go negative.
    let v_min = velocities[0];
    let v_neg_limit = v_min - DOPPLER_N_SIGMA * u;

    // Build extended velocity grid: negative points (if needed) + positive points.
    // Pre-compute total capacity: negative points + zero + positive + upper extension.
    let dv_lo = if n > 1 {
        (velocities[1] - velocities[0]).max(u * 0.1)
    } else {
        u * 0.5
    };
    let dv_hi = if n > 1 {
        (velocities[n - 1] - velocities[n - 2]).max(u * 0.1)
    } else {
        u * 0.5
    };
    let n_neg = if v_neg_limit < 0.0 {
        // Points from v_neg_limit to just below zero, plus the v=0 anchor.
        (((-v_neg_limit - NEGATIVE_VELOCITY_FLOOR) / dv_lo).ceil() as usize).saturating_add(1)
    } else {
        0
    };
    let v_max = velocities[n - 1];
    let v_max_limit = v_max + DOPPLER_N_SIGMA * u;
    let n_hi = if v_max < v_max_limit {
        ((v_max_limit - v_max) / dv_hi).ceil() as usize
    } else {
        0
    };
    let capacity = n_neg + n + n_hi;
    let mut ext_v: Vec<f64> = Vec::with_capacity(capacity);
    let mut ext_y: Vec<f64> = Vec::with_capacity(capacity);

    if v_neg_limit < 0.0 {
        // We need negative velocity points.
        // Use the same spacing as the low-energy end of the positive grid,
        // but in velocity space (uniform dv).
        let mut v = v_neg_limit;
        while v < -NEGATIVE_VELOCITY_FLOOR {
            ext_v.push(v);
            // Y(w) = -w² * σ(w²) for negative w (odd integrand)
            // σ at E = w² — interpolate from the positive grid
            let e = v * v;
            let sigma = interpolate_cross_section(energies, cross_sections, e);
            ext_y.push(-(v * v) * sigma);
            v += dv_lo;
        }

        // Add v = 0 point
        ext_v.push(0.0);
        ext_y.push(0.0);
    }

    // Add the positive velocity points
    for i in 0..n {
        ext_v.push(velocities[i]);
        ext_y.push(velocities[i] * velocities[i] * cross_sections[i]);
    }

    // Add points beyond the highest velocity if needed
    if v_max < v_max_limit {
        let mut v = v_max + dv_hi;
        while v <= v_max_limit {
            ext_v.push(v);
            let e = v * v;
            let sigma = interpolate_cross_section(energies, cross_sections, e);
            ext_y.push(v * v * sigma);
            v += dv_hi;
        }
    }

    let n_ext = ext_v.len();

    // The extended velocity grid must be sorted ascending (negative → 0 → positive)
    // for the partition_point binary searches below to work correctly.
    debug_assert!(
        ext_v.windows(2).all(|w| w[0] <= w[1]),
        "ext_v must be sorted ascending for partition_point"
    );

    // For each output energy point, compute the broadened cross-section
    // using piecewise-linear interpolation of Y(w) = w²·s(w) combined
    // with exact Gaussian integration over each segment.
    //
    // SAMMY Ref: `fgm/mfgm2.f90` Modsmp (linear), Modfpl (4-point Lagrange).
    // Our PW-linear approach matches Modsmp's 2-point interpolation with
    // analytical Gaussian integration via Abcerf/Abcexp.
    //
    // For each segment [w_j, w_{j+1}], the integrand Y is approximated as:
    //   Y(w) ≈ Y_j + slope × (w − w_j)
    //
    // The exact integral of G(v,w) × Y_linear(w) dw over the segment is:
    //   u × [C_j × J₀ − u × slope × J₁]
    //
    // where C_j = Y_j + slope × (v − w_j), and:
    //   J₀ = ∫ exp(−t²) dt = (√π/2)(erfc(b_{j+1}) − erfc(b_j))
    //   J₁ = ∫ t·exp(−t²) dt = [exp(−b_{j+1}²) − exp(−b_j²)] / 2
    //   b_j = (v − w_j) / u
    //
    // This provides second-order accuracy (error ∝ h²) compared to the
    // zeroth-order Voronoi cell approach (error ∝ h).

    let mut broadened = vec![0.0f64; n];

    for i in 0..n {
        let v = velocities[i];
        let e = energies[i];
        if v < NEAR_ZERO_FLOOR || e < NEAR_ZERO_FLOOR {
            broadened[i] = cross_sections[i];
            continue;
        }

        // O(N×W) optimisation: binary search restricts the inner loop to the
        // Gaussian window [v − n_sigma·u, v + n_sigma·u].
        let v_lo = v - DOPPLER_N_SIGMA * u;
        let v_hi = v + DOPPLER_N_SIGMA * u;
        let j_lo = ext_v.partition_point(|&w| w < v_lo);
        let j_hi = ext_v.partition_point(|&w| w <= v_hi);

        if j_lo >= j_hi {
            broadened[i] = cross_sections[i];
            continue;
        }

        // PW-linear FGM integral: segment-by-segment exact integration.
        //
        // v² × σ_D(v²) = Σ [C_j × J₀_j − u × slope_j × J₁_j] / Σ J₀_j
        // σ_D(E) = Σ[…] / (Σ J₀ × E)        (E = v²)
        //
        // SAMMY Ref: `fgm/mfgm2.f90` Modsmp lines 80-87 (linear weights
        // with Abcerf B-coefficient = first moment correction; final
        // weights carry the w² factor, lines 101/203) and `mfgm4.f90`
        // (division by Em).
        let mut sum_y = 0.0f64; // Numerator: Σ [C × J₀ − u × slope × J₁]
        let mut sum_g = 0.0f64; // Denominator: Σ J₀

        // Process segments [j, j+1] that overlap the Gaussian window.
        let seg_lo = if j_lo > 0 { j_lo - 1 } else { j_lo };
        let seg_hi = j_hi.min(n_ext - 1);

        for j in seg_lo..seg_hi {
            let w_j = ext_v[j];
            let w_j1 = ext_v[j + 1];
            let h_w = w_j1 - w_j;
            if h_w < NEAR_ZERO_FLOOR {
                continue;
            }

            // Scaled distances from target velocity.
            let b_j = (v - w_j) / u;
            let b_j1 = (v - w_j1) / u;

            // J₀ = ∫_{b_{j+1}}^{b_j} exp(−t²) dt
            //     = (√π/2)(erfc(b_{j+1}) − erfc(b_j))
            let erfc_bj = erfc_val(b_j);
            let erfc_bj1 = erfc_val(b_j1);
            let j0 = SQRT_PI * 0.5 * (erfc_bj1 - erfc_bj);

            if j0 < NEAR_ZERO_FLOOR {
                continue;
            }

            // J₁ = ∫_{b_{j+1}}^{b_j} t·exp(−t²) dt
            //     = [exp(−b_{j+1}²) − exp(−b_j²)] / 2
            let j1 = ((-b_j1 * b_j1).exp() - (-b_j * b_j).exp()) * 0.5;

            let y_j = ext_y[j];
            let y_j1 = ext_y[j + 1];
            let slope = (y_j1 - y_j) / h_w;

            // C_j = Y_j + slope × (v − w_j) = Y_j + slope × u × b_j
            let c_j = y_j + slope * u * b_j;

            // Contribution: C × J₀ − u × slope × J₁
            sum_y += c_j * j0 - u * slope * j1;
            sum_g += j0;
        }

        if sum_g < DIVISION_FLOOR {
            broadened[i] = cross_sections[i];
            continue;
        }

        // σ_D(E) = Σ(C × J₀ − u × slope × J₁) / (Σ J₀ × E)
        broadened[i] = sum_y / (sum_g * e);

        // Ensure non-negative
        if broadened[i] < 0.0 {
            broadened[i] = 0.0;
        }
    }

    Ok(broadened)
}

/// Doppler-broaden cross-sections AND compute the analytical temperature
/// derivative ∂σ_D/∂T in a single pass.
///
/// This computes the exact derivative by differentiating the FGM integral
/// with respect to the Doppler width parameter u = √(k_B·T / AWR), then
/// applying the chain rule: ∂σ_D/∂T = (∂σ_D/∂u) · u/(2T).
///
/// The derivative uses intermediate quantities already computed in the
/// forward pass (b_k, exp(-b_k²), J₀, J₁, C_j, slope), adding only
/// ~10 FLOPs per segment with NO extra broadening evaluations.
///
/// ## Mathematical Derivation
///
/// Per segment [w_j, w_{j+1}]:
///   M₀_j = b_{j+1}·exp(-b_{j+1}²) - b_j·exp(-b_j²)
///   M₁_j = b_{j+1}²·exp(-b_{j+1}²) - b_j²·exp(-b_j²)
///   ∂I_j/∂u = (C_j/u)·M₀_j - slope_j·J₁_j - slope_j·M₁_j
///
/// Full result (quotient rule on sum_y / (sum_g · E), E = v² being
/// temperature-independent):
///   ∂σ_D/∂T = u/(2T·E) · (dsum_y·sum_g - sum_y·dsum_g) / sum_g²
///
/// SAMMY uses finite differences for this (mfgm4.f90 Xdofgm, Del=0.02).
/// Our analytical approach is exact and avoids the 3× broadening cost.
///
/// # Arguments
/// * `energies` — Energy grid in eV. Same contract as [`doppler_broaden`]:
///   every entry must be finite and strictly positive, and the grid must
///   be strictly ascending. The first `doppler_broaden` call below
///   propagates the validation error through the `?` operator.
/// * `cross_sections` — Unbroadened cross-sections in barns at each energy point.
/// * `params` — Doppler broadening parameters (temperature and AWR).
///
/// # Errors
/// Returns the same `DopplerError` variants as [`doppler_broaden`].
pub fn doppler_broaden_with_derivative(
    energies: &[f64],
    cross_sections: &[f64],
    params: &DopplerParams,
) -> Result<(Vec<f64>, Vec<f64>), DopplerError> {
    // First, compute the broadened values using the SAME code path as
    // doppler_broaden to guarantee identical forward-pass results.
    let broadened = doppler_broaden(energies, cross_sections, params)?;

    let n = energies.len();
    if n == 0 {
        return Ok((broadened, vec![]));
    }
    if params.temperature_k < NEAR_ZERO_FLOOR {
        return Ok((broadened, vec![0.0; n]));
    }

    let u = params.u();
    let temperature_k = params.temperature_k;

    // Rebuild the same extended grid as doppler_broaden.
    // This duplicates the grid construction, but guarantees consistency
    // with the forward pass. The cost is O(n) — negligible compared to
    // the O(n × n_segments) integration.
    let velocities: Vec<f64> = energies.iter().map(|&e| e.sqrt()).collect();
    let v_min = velocities[0];
    let v_neg_limit = v_min - DOPPLER_N_SIGMA * u;
    let dv_lo = if n > 1 {
        (velocities[1] - velocities[0]).max(u * 0.1)
    } else {
        u * 0.5
    };
    let dv_hi = if n > 1 {
        (velocities[n - 1] - velocities[n - 2]).max(u * 0.1)
    } else {
        u * 0.5
    };
    let v_max = velocities[n - 1];
    let v_max_limit = v_max + DOPPLER_N_SIGMA * u;

    let mut ext_v: Vec<f64> = Vec::new();
    let mut ext_y: Vec<f64> = Vec::new();

    if v_neg_limit < 0.0 {
        let mut v = v_neg_limit;
        while v < -NEGATIVE_VELOCITY_FLOOR {
            ext_v.push(v);
            let e = v * v;
            let sigma = interpolate_cross_section(energies, cross_sections, e);
            ext_y.push(-(v * v) * sigma); // odd integrand: Y(w) = w²·s(w)
            v += dv_lo;
        }
        ext_v.push(0.0);
        ext_y.push(0.0);
    }

    for i in 0..n {
        ext_v.push(velocities[i]);
        ext_y.push(velocities[i] * velocities[i] * cross_sections[i]);
    }

    if v_max < v_max_limit {
        let mut v = v_max + dv_hi;
        while v <= v_max_limit {
            ext_v.push(v);
            let e = v * v;
            let sigma = interpolate_cross_section(energies, cross_sections, e);
            ext_y.push(v * v * sigma);
            v += dv_hi;
        }
    }

    let n_ext = ext_v.len();

    // Compute the derivative in a second pass over the same grid.
    let mut derivative = vec![0.0f64; n];

    for i in 0..n {
        let v = velocities[i];
        let e = energies[i];
        if v < NEAR_ZERO_FLOOR || e < NEAR_ZERO_FLOOR {
            derivative[i] = 0.0;
            continue;
        }

        let v_lo = v - DOPPLER_N_SIGMA * u;
        let v_hi = v + DOPPLER_N_SIGMA * u;
        let j_lo = ext_v.partition_point(|&w| w < v_lo);
        let j_hi = ext_v.partition_point(|&w| w <= v_hi);

        if j_lo >= j_hi {
            derivative[i] = 0.0;
            continue;
        }

        // Re-integrate to get sum_y and sum_g (needed for quotient rule).
        // Also accumulate derivative terms in the same loop.
        let mut sum_y = 0.0f64;
        let mut sum_g = 0.0f64;
        let mut dsum_y = 0.0f64;
        let mut sum_m0 = 0.0f64;

        let seg_lo = if j_lo > 0 { j_lo - 1 } else { j_lo };
        let seg_hi = j_hi.min(n_ext - 1);

        for j in seg_lo..seg_hi {
            let w_j = ext_v[j];
            let w_j1 = ext_v[j + 1];
            let h_w = w_j1 - w_j;
            if h_w < NEAR_ZERO_FLOOR {
                continue;
            }

            let b_j = (v - w_j) / u;
            let b_j1 = (v - w_j1) / u;

            let erfc_bj = erfc_val(b_j);
            let erfc_bj1 = erfc_val(b_j1);
            let j0 = SQRT_PI * 0.5 * (erfc_bj1 - erfc_bj);

            if j0 < NEAR_ZERO_FLOOR {
                continue;
            }

            let exp_bj = (-b_j * b_j).exp();
            let exp_bj1 = (-b_j1 * b_j1).exp();
            let j1 = (exp_bj1 - exp_bj) * 0.5;

            let y_j = ext_y[j];
            let y_j1 = ext_y[j + 1];
            let slope = (y_j1 - y_j) / h_w;
            let c_j = y_j + slope * (v - w_j);

            // Forward accumulators (for quotient rule denominator).
            sum_y += c_j * j0 - u * slope * j1;
            sum_g += j0;

            // Derivative terms.
            let m0 = b_j1 * exp_bj1 - b_j * exp_bj;
            let m1 = b_j1 * b_j1 * exp_bj1 - b_j * b_j * exp_bj;
            dsum_y += (c_j / u) * m0 - slope * j1 - slope * m1;
            sum_m0 += m0;
        }

        if sum_g < DIVISION_FLOOR {
            derivative[i] = 0.0;
            continue;
        }

        // ∂σ_D/∂T = (u · dsum_y · sum_g - sum_y · sum_m0) / (2T · E · sum_g²)
        let numerator = u * dsum_y * sum_g - sum_y * sum_m0;
        let denominator = 2.0 * temperature_k * e * sum_g * sum_g;
        if denominator.abs() > NEAR_ZERO_FLOOR {
            derivative[i] = numerator / denominator;
        } else {
            derivative[i] = 0.0;
        }
    }

    Ok((broadened, derivative))
}

/// Linear interpolation of cross-section at an arbitrary energy.
///
/// Unlike `resolution::interp_spectrum` (which returns `None` for off-grid
/// queries), this function extrapolates using the 1/v law.  A future
/// consolidation could unify both behind a shared trait or closure-based
/// extrapolation strategy; for now they remain separate to avoid coupling
/// the two broadening modules.
fn interpolate_cross_section(energies: &[f64], cross_sections: &[f64], energy: f64) -> f64 {
    if energies.is_empty() {
        return 0.0;
    }

    // Guard against NaN energy: NaN comparisons are always false, so the
    // boundary checks below would both be skipped.  The binary search would
    // then return Err(0), and `idx = 0 - 1` would underflow on usize.
    if energy.is_nan() {
        return 0.0;
    }

    if energy <= energies[0] {
        // Extrapolate using 1/v law: σ ∝ 1/√E.
        // Guard: if energy <= 0, the ratio energies[0]/energy would be negative
        // or infinite, producing NaN from sqrt.  Return the boundary value directly.
        if energy <= 0.0 {
            return cross_sections[0];
        }
        if energies[0] > NEAR_ZERO_FLOOR {
            return cross_sections[0] * (energies[0] / energy).sqrt();
        }
        return cross_sections[0];
    }

    if energy >= energies[energies.len() - 1] {
        // Extrapolate using 1/v law
        let last = energies.len() - 1;
        if energy > NEAR_ZERO_FLOOR {
            return cross_sections[last] * (energies[last] / energy).sqrt();
        }
        return cross_sections[last];
    }

    // Binary search for the interval.
    // Use total_cmp-style fallback to avoid panic on NaN comparisons.
    // With the current comparator (NaNs treated as Ordering::Less), NaN
    // values in the energy grid are pushed to the right, so Err(0) should
    // not occur in normal operation. The Err(0) arm is kept as a
    // defense-in-depth guard: if the NaN guard on `energy` is ever removed
    // or the comparator behavior changes and Err(0) becomes possible, we
    // avoid `0 - 1` underflow on usize by returning the first cross-section.
    let idx = match energies
        .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(i) => return cross_sections[i],
        Err(0) => return cross_sections[0],
        Err(i) => i - 1,
    };

    // Linear interpolation.
    // Guard against duplicate energy grid points: if e0 == e1 (or nearly so),
    // no interpolation is needed — use the value at that point directly.
    // Use a combined relative+absolute threshold that works across the full
    // energy range (meV to MeV): |de| < |e0|·ε_mach + NEAR_ZERO_FLOOR.
    // The relative part handles large energies where f64::EPSILON alone would
    // miss near-duplicates; the absolute part handles energies near zero.
    // This is consistent with resolution.rs interp_spectrum.
    let e0 = energies[idx];
    let e1 = energies[idx + 1];
    let s0 = cross_sections[idx];
    let s1 = cross_sections[idx + 1];
    let de = e1 - e0;
    if de.abs() < e0.abs() * f64::EPSILON + NEAR_ZERO_FLOOR {
        return s0;
    }
    let t = (energy - e0) / de;
    s0 + t * (s1 - s0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DopplerError Display rendering tests ---
    //
    // The Display impls use single-line format-string literals to avoid
    // embedding indentation into the rendered error messages. These tests
    // pin that contract: a stray `\<newline>    ` continuation in the
    // literal would silently inject a run of spaces into the user-facing
    // string and would only be caught by eyeballing log output.

    #[test]
    fn test_doppler_error_display_no_embedded_indentation() {
        let e = DopplerError::InvalidEnergy {
            index: 1,
            value: f64::NAN,
        };
        let rendered = format!("{e}");
        assert!(
            !rendered.contains("  "),
            "InvalidEnergy Display contains double-space (embedded indentation?): {rendered:?}"
        );

        let e = DopplerError::UnsortedEnergies {
            index: 3,
            previous: 4.0,
            current: 2.5,
        };
        let rendered = format!("{e}");
        assert!(
            !rendered.contains("  "),
            "UnsortedEnergies Display contains double-space (embedded indentation?): {rendered:?}"
        );

        let e = DopplerError::LengthMismatch {
            energies: 5,
            cross_sections: 4,
        };
        let rendered = format!("{e}");
        assert!(
            !rendered.contains("  "),
            "LengthMismatch Display contains double-space (embedded indentation?): {rendered:?}"
        );
    }

    // --- DopplerParams::new() validation tests ---

    #[test]
    fn test_new_negative_temperature_rejected() {
        assert_eq!(
            DopplerParams::new(-1.0, 238.0).unwrap_err(),
            DopplerParamsError::NegativeTemperature(-1.0)
        );
    }

    #[test]
    fn test_new_nan_temperature_rejected() {
        let err = DopplerParams::new(f64::NAN, 238.0).unwrap_err();
        assert!(
            matches!(err, DopplerParamsError::NonFiniteTemperature(v) if v.is_nan()),
            "NaN temperature should return NonFiniteTemperature"
        );
    }

    #[test]
    fn test_new_infinity_temperature_rejected() {
        assert_eq!(
            DopplerParams::new(f64::INFINITY, 238.0).unwrap_err(),
            DopplerParamsError::NonFiniteTemperature(f64::INFINITY)
        );
    }

    #[test]
    fn test_new_negative_awr_rejected() {
        assert_eq!(
            DopplerParams::new(300.0, -1.0).unwrap_err(),
            DopplerParamsError::InvalidAwr(-1.0)
        );
    }

    #[test]
    fn test_new_zero_awr_rejected() {
        assert_eq!(
            DopplerParams::new(300.0, 0.0).unwrap_err(),
            DopplerParamsError::InvalidAwr(0.0)
        );
    }

    #[test]
    fn test_new_nan_awr_rejected() {
        let err = DopplerParams::new(300.0, f64::NAN).unwrap_err();
        assert!(
            matches!(err, DopplerParamsError::InvalidAwr(v) if v.is_nan()),
            "NaN AWR should return InvalidAwr"
        );
    }

    #[test]
    fn test_new_zero_temperature_allowed() {
        let params = DopplerParams::new(0.0, 238.0);
        assert!(params.is_ok(), "zero temperature should be allowed");
        let p = params.unwrap();
        assert_eq!(p.temperature_k(), 0.0);
        assert_eq!(p.awr(), 238.0);
    }

    #[test]
    fn test_new_valid_params() {
        let params = DopplerParams::new(300.0, 238.0);
        assert!(params.is_ok(), "valid params should succeed");
        let p = params.unwrap();
        assert_eq!(p.temperature_k(), 300.0);
        assert_eq!(p.awr(), 238.0);
    }

    // --- End validation tests ---

    #[test]
    fn test_doppler_width_u238() {
        // SAMMY reports: Doppler width at 6.075 eV = 0.05159437 eV for U-238 at 300K
        // AWR = 238.050972, T = 300 K
        let params = DopplerParams::new(300.0, 238.050972).unwrap();
        let dw = params.doppler_width(6.075);
        // SAMMY uses kB = 0.000086173420 eV/K (slightly different from CODATA 2018)
        // Our kB = 8.617333262e-5. The difference is ~0.003%.
        // So we expect close but not exact match.
        assert!(
            (dw - 0.05159437).abs() < 5e-4,
            "Doppler width = {}, expected ~0.05159",
            dw
        );
    }

    #[test]
    fn test_doppler_width_fictitious() {
        // ex001: A=10, T=300K. Δ_D at 10 eV = √(4kBTE/AWR).
        // SAMMY reports Δ_D = 0.3216 eV, FWHM = 2√(ln2) × Δ_D = 0.5355 eV.
        // (SAMMY lpt uses slightly different kB, giving FWHM = 0.5378 eV.)
        let params = DopplerParams::new(300.0, 10.0).unwrap();
        let dw = params.doppler_width(10.0);
        // Δ_D = √(4 × 8.617e-5 × 300 × 10 / 10) = √(0.10341) ≈ 0.3216 eV
        assert!(
            (dw - 0.3216).abs() < 0.01,
            "Doppler width = {}, expected ~0.32",
            dw
        );
    }

    #[test]
    fn test_zero_temperature() {
        // At T=0, broadening should return the original cross-sections.
        let energies = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let xs = vec![10.0, 20.0, 30.0, 20.0, 10.0];
        let params = DopplerParams::new(0.0, 238.0).unwrap();
        let broadened = doppler_broaden(&energies, &xs, &params).unwrap();
        assert_eq!(broadened, xs);
    }

    #[test]
    fn test_broadening_reduces_peak() {
        // Doppler broadening should reduce the peak height and spread it out.
        // Create a sharp resonance peak.
        let n = 201;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 0.05).collect();
        let center = 10.0;
        let gamma: f64 = 0.02; // narrow resonance
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let de = e - center;
                100.0 * (gamma / 2.0).powi(2) / (de * de + (gamma / 2.0).powi(2))
            })
            .collect();

        let params = DopplerParams::new(300.0, 238.0).unwrap();
        let broadened = doppler_broaden(&energies, &xs, &params).unwrap();

        // Find peaks
        let orig_peak = xs.iter().cloned().fold(0.0_f64, f64::max);
        let broad_peak = broadened.iter().cloned().fold(0.0_f64, f64::max);

        assert!(
            broad_peak < orig_peak,
            "Broadened peak ({}) should be less than original ({})",
            broad_peak,
            orig_peak
        );

        // The broadened peak should still be substantial (not wiped out)
        assert!(
            broad_peak > 0.1,
            "Broadened peak ({}) should still be positive",
            broad_peak
        );
    }

    /// SAMMY ex001 validation: single resonance, A=10, T=300K, FGM Doppler.
    ///
    /// Reference: ex001a.lst (column 4 = theoretical Doppler-broadened capture σ)
    /// Par file: E₀ = 10 eV, Γγ = 1.0 meV, Γn = 0.5 meV
    /// SAMMY par file widths are in meV; we convert to eV (×0.001) for our code.
    /// AWR = 10.0, radius = 2.908 fm, T = 300 K
    #[test]
    fn test_sammy_ex001_fgm_doppler() {
        // Build the ex001 resonance data: single SLBW resonance at 10 eV,
        // ZA=1010, AWR=10.0, AP=2.908 fm (SAMMY par-file widths in meV are
        // pre-converted to eV inside `ex001_hydrogen_single_resonance`).
        let data = nereids_endf::resonance::test_support::ex001_hydrogen_single_resonance();

        // Generate unbroadened cross-sections on a non-uniform grid.
        // The resonance is very narrow (Γ ≈ 1.5 meV) — we need fine spacing
        // near E₀ = 10 eV and coarser spacing in the wings.
        let mut energies: Vec<f64> = Vec::new();
        // Wings: 6.0 to 9.95 and 10.05 to 14.0 with 0.005 eV spacing
        let mut e = 6.0;
        while e < 9.95 {
            energies.push(e);
            e += 0.005;
        }
        // Core: 9.95 to 10.05 with 0.00005 eV spacing (resolves 1.5 meV resonance)
        while e < 10.05 {
            energies.push(e);
            e += 0.00005;
        }
        // Upper wing: 10.05 to 14.0
        while e <= 14.0 {
            energies.push(e);
            e += 0.005;
        }
        energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        energies.dedup();
        let unbroadened: Vec<f64> = energies
            .iter()
            .map(|&e| crate::slbw::slbw_cross_sections(&data, e).capture)
            .collect();

        // Apply FGM Doppler broadening.
        let params = DopplerParams::new(300.0, 10.0).unwrap();
        let broadened = doppler_broaden(&energies, &unbroadened, &params).unwrap();

        // SAMMY ex001a.lst reference points: (energy, broadened capture σ in barns).
        // Focus on the core region where our grid has good coverage.
        let sammy_ref = [
            (9.3594, 5.4125807788),    // lower shoulder
            (9.8572, 238.1729827317),  // near peak
            (9.9869, 285.6111456228),  // peak
            (10.0092, 285.2175881633), // just past peak
            (10.1282, 241.3304410052), // upper shoulder
            (10.3430, 91.4783098707),  // falling slope
            (10.5382, 18.3744223751),  // upper wing
        ];

        // Interpolate our broadened result onto SAMMY energy points and compare.
        let mut max_rel_err = 0.0f64;
        for &(e_ref, sigma_ref) in &sammy_ref {
            let sigma_us = interpolate_cross_section(&energies, &broadened, e_ref);
            let rel_err = (sigma_us - sigma_ref).abs() / sigma_ref;
            max_rel_err = max_rel_err.max(rel_err);
        }
        eprintln!("ex001 FGM: max_rel_err={max_rel_err:.6}");
        // PW-linear segment integration differs from SAMMY's quadrature at
        // grid-spacing transitions (wing region).  Measured with the exact
        // w²-weighted kernel: 2.37%; the legacy w¹ kernel measured 5.48%
        // (the A=10 target makes u/v large, so the kernel's first-order
        // term was a visible part of the old error).
        assert!(
            max_rel_err < 0.03,
            "Max relative error = {:.2}% (exceeds 3%)",
            max_rel_err * 100.0
        );

        // Check peak height specifically (should be close to 285.6 barns).
        let peak_idx = broadened
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let peak_energy = energies[peak_idx];
        let peak_sigma = broadened[peak_idx];

        // Peak should be near 10 eV (slight shift to lower E due to 1/v weighting).
        assert!(
            (peak_energy - 9.99).abs() < 0.1,
            "Peak energy = {:.4}, expected near 9.99",
            peak_energy
        );
        assert!(
            (peak_sigma - 285.6).abs() < 30.0,
            "Peak σ = {:.2}, expected ~285.6",
            peak_sigma
        );
    }

    #[test]
    fn test_broadening_conserves_area() {
        // Doppler broadening should approximately conserve the area under
        // the cross-section curve (energy × cross-section is conserved).
        let n = 401;
        let energies: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let center = 10.0;
        let gamma: f64 = 0.5;
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let de = e - center;
                1000.0 * (gamma / 2.0).powi(2) / (de * de + (gamma / 2.0).powi(2))
            })
            .collect();

        let params = DopplerParams::new(300.0, 100.0).unwrap();
        let broadened = doppler_broaden(&energies, &xs, &params).unwrap();

        // Compute area (trapezoidal) for both
        let area_orig: f64 = (0..n - 1)
            .map(|i| 0.5 * (xs[i] + xs[i + 1]) * (energies[i + 1] - energies[i]))
            .sum();
        let area_broad: f64 = (0..n - 1)
            .map(|i| 0.5 * (broadened[i] + broadened[i + 1]) * (energies[i + 1] - energies[i]))
            .sum();

        let rel_diff = (area_orig - area_broad).abs() / area_orig;
        assert!(
            rel_diff < 0.05,
            "Area not conserved: orig={}, broad={}, rel_diff={:.4}",
            area_orig,
            area_broad,
            rel_diff
        );
    }

    /// NaN query energy: interpolate_cross_section must return 0.0 without
    /// panicking (the NaN guard at line 282 catches this).
    #[test]
    fn test_interpolate_nan_energy() {
        let energies = vec![1.0, 2.0, 3.0];
        let xs = vec![10.0, 20.0, 30.0];
        let result = interpolate_cross_section(&energies, &xs, f64::NAN);
        assert_eq!(result, 0.0, "NaN energy should return 0.0");
    }

    /// Err(0) guard in binary search: if the binary search were to return
    /// Err(0) (insertion point = 0), `i - 1` would underflow on usize.
    /// The guard returns cross_sections[0] instead.
    ///
    /// This path is hard to trigger with well-formed grids (the boundary
    /// check `energy <= energies[0]` catches it first), but can occur if
    /// the grid or the comparison function behaves unexpectedly (e.g.
    /// NaN contamination with a different comparison strategy).  The guard
    /// is cheap defense-in-depth against arithmetic underflow.
    ///
    /// NOTE: This test exercises the `energy <= energies[0]` boundary path
    /// (1/v extrapolation), *not* the `Err(0)` binary-search guard itself.
    ///
    /// We test the NaN query guard separately (`test_interpolate_nan_energy`),
    /// the NaN grid guard separately (`test_interpolate_nan_grid_no_panic`),
    /// and the duplicate-point guard separately (`test_interpolate_duplicate_grid_points`).
    ///
    /// The `Err(0)` binary-search guard is primarily a defense-in-depth
    /// safety net against unexpected grid or comparison behavior.
    #[test]
    fn test_interpolate_below_grid_minimum() {
        let energies = vec![5.0, 10.0, 15.0];
        let xs = vec![50.0, 100.0, 150.0];
        // Energy below the grid minimum: hits the `energy <= energies[0]` guard
        // and returns via 1/v extrapolation, not the binary search.
        let result = interpolate_cross_section(&energies, &xs, 2.0);
        assert!(
            result.is_finite() && result > 0.0,
            "Below-grid query should return a finite positive value via 1/v extrapolation, got {result}"
        );
        // Check 1/v scaling: σ(2) ≈ σ(5) × √(5/2)
        let expected = 50.0 * (5.0 / 2.0_f64).sqrt();
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected 1/v extrapolation: {expected}, got {result}"
        );
    }

    /// Duplicate grid points: two adjacent energies are identical.
    /// The combined relative+absolute threshold must detect this and
    /// return the value at the duplicate point without division by zero.
    #[test]
    fn test_interpolate_duplicate_grid_points() {
        let energies = vec![1.0, 2.0, 2.0, 3.0];
        let xs = vec![10.0, 20.0, 25.0, 30.0];
        // Query at exactly 2.0 should hit the Ok(i) branch.
        let result = interpolate_cross_section(&energies, &xs, 2.0);
        assert!(
            (result - 20.0).abs() < 1e-10 || (result - 25.0).abs() < 1e-10,
            "At duplicate point 2.0, should return one of the boundary values, got {result}"
        );
        // Query at 2.0 + tiny epsilon should trigger the duplicate guard.
        let result2 = interpolate_cross_section(&energies, &xs, 2.0 + 1e-16);
        assert!(
            result2.is_finite(),
            "Near-duplicate query should return finite result, got {result2}"
        );

        // Exercise the `de.abs() < |e0|*EPS + NEAR_ZERO_FLOOR` threshold
        // with near-zero adjacent energies where de is essentially zero.
        // With e0 = 1e-50, the relative term |e0|*EPS ≈ 2e-66 is smaller
        // than NEAR_ZERO_FLOOR (1e-60), so the absolute floor dominates.
        let tiny_energies = vec![1e-50, 1e-50 + 1e-105, 1.0];
        let tiny_xs = vec![100.0, 200.0, 300.0];
        // Query between the two near-zero points: de ≈ 1e-105 which is
        // far below the absolute threshold NEAR_ZERO_FLOOR (1e-60),
        // and the relative term (|1e-50| * EPS ≈ 2e-66) is even smaller,
        // so the absolute floor is the binding constraint.
        let result3 = interpolate_cross_section(&tiny_energies, &tiny_xs, 1e-50 + 5e-106);
        assert!(
            result3.is_finite(),
            "Near-zero de should be caught by the absolute threshold, got {result3}"
        );
        // Should return s0 (100.0) since the guard short-circuits.
        assert!(
            (result3 - 100.0).abs() < 1e-10,
            "Expected s0=100.0 from the de threshold guard, got {result3}"
        );
    }

    /// NaN-contaminated energy grid: verify no panic occurs and the NaN
    /// query guard (line 282) protects against the `Err(0)` binary search
    /// underflow path (line 317).
    ///
    /// With the current comparator (`unwrap_or(Ordering::Less)`), NaN grid
    /// entries are treated as "less than" any query, pushing the binary
    /// search rightward.  This means NaN *in the grid* alone cannot produce
    /// `Err(0)` — it always produces `Err(k)` with k > 0.  However, a NaN
    /// *query* bypasses comparisons entirely and could reach `Err(0)` if the
    /// earlier NaN guard (line 282) were removed.  That guard returns 0.0
    /// before the binary search, making `Err(0)` unreachable in practice.
    ///
    /// The `Err(0)` match arm is therefore pure defense-in-depth against
    /// future comparator changes.  This test verifies:
    ///   1. NaN query → returns 0.0 (guard fires, `Err(0)` never reached).
    ///   2. NaN in grid → no panic (does not underflow).
    #[test]
    fn test_interpolate_nan_grid_no_panic() {
        let xs = vec![10.0, 20.0, 30.0];

        // Case 1: NaN query on a clean grid — the NaN guard at line 282
        // returns 0.0 before reaching the binary search.  This is the only
        // code path that *would* hit Err(0) if the guard were absent.
        let clean_grid = vec![1.0, 2.0, 3.0];
        let result = interpolate_cross_section(&clean_grid, &xs, f64::NAN);
        assert_eq!(result, 0.0, "NaN query should return 0.0 via the guard");

        // Case 2: NaN in the grid at position 0 — the boundary check
        // `energy <= energies[0]` is false (NaN comparison), so we fall
        // through to the binary search.  The search treats NaN as Less,
        // returning Err(k>0), so the Err(0) arm is NOT reached.  The
        // function should not panic.
        let nan_grid = vec![f64::NAN, 2.0, 3.0];
        let result2 = interpolate_cross_section(&nan_grid, &xs, 1.5);
        // Result may be NaN (interpolating with a NaN grid point), but
        // the important thing is no panic from usize underflow.
        let _ = result2; // just verify no panic
    }

    // ── Milestone A: Analytical derivative validation ──

    /// Helper: generate a simple resonance-like cross-section for testing.
    fn test_resonance_xs(energies: &[f64], e_res: f64, gamma: f64, peak: f64) -> Vec<f64> {
        energies
            .iter()
            .map(|&e| {
                let x = (e - e_res) / gamma;
                peak / (1.0 + x * x) + 10.0 // Breit-Wigner + constant
            })
            .collect()
    }

    /// A1: Analytical derivative vs central FD for U-238 at 293.6K.
    #[test]
    fn test_analytical_derivative_vs_fd_u238_293k() {
        let energies: Vec<f64> = (0..200).map(|i| 1.0 + i as f64 * 0.05).collect();
        let xs = test_resonance_xs(&energies, 6.67, 0.025, 5000.0);
        let params = DopplerParams::new(293.6, 238.051).unwrap();

        // Analytical derivative
        let (broadened, dxs_dt) = doppler_broaden_with_derivative(&energies, &xs, &params).unwrap();

        // Central FD derivative
        let dt = 1e-4 * (1.0 + params.temperature_k);
        let params_up = DopplerParams::new(params.temperature_k + dt, params.awr).unwrap();
        let params_down =
            DopplerParams::new((params.temperature_k - dt).max(0.1), params.awr).unwrap();
        let actual_2dt = (params.temperature_k + dt) - (params.temperature_k - dt).max(0.1);

        let xs_up = doppler_broaden(&energies, &xs, &params_up).unwrap();
        let xs_down = doppler_broaden(&energies, &xs, &params_down).unwrap();

        // Use combined error metric: relative where derivative is significant,
        // absolute where derivative is small (avoiding catastrophic cancellation
        // in flat regions far from resonances — a known limitation of the
        // quotient-rule formulation when sum_y and dsum_g nearly cancel).
        let max_deriv: f64 = (0..energies.len())
            .map(|i| ((xs_up[i] - xs_down[i]) / actual_2dt).abs())
            .fold(0.0f64, f64::max);
        let abs_tol = max_deriv * 1e-4;

        let mut max_rel_err = 0.0f64;
        let mut n_significant = 0;
        for i in 0..energies.len() {
            let fd = (xs_up[i] - xs_down[i]) / actual_2dt;
            if fd.abs() < 1e-15 {
                continue;
            }
            // For significant derivatives (> 1% of peak), check relative error.
            if fd.abs() > max_deriv * 0.01 {
                let rel_err = ((dxs_dt[i] - fd) / fd).abs();
                max_rel_err = max_rel_err.max(rel_err);
                n_significant += 1;
            } else {
                // For small derivatives, check absolute error.
                let abs_err = (dxs_dt[i] - fd).abs();
                assert!(
                    abs_err < abs_tol,
                    "E={:.3}: abs error {:.2e} exceeds tol {:.2e} (analytical={:.2e}, FD={:.2e})",
                    energies[i],
                    abs_err,
                    abs_tol,
                    dxs_dt[i],
                    fd
                );
            }
        }
        assert!(
            n_significant > 5,
            "need at least 5 significant derivative points, got {n_significant}"
        );
        assert!(
            max_rel_err < 1e-6,
            "analytical vs FD relative error (significant bins) = {max_rel_err:.2e}, expected < 1e-6"
        );

        // Verify forward pass matches standalone doppler_broaden
        let broadened_ref = doppler_broaden(&energies, &xs, &params).unwrap();
        for i in 0..energies.len() {
            assert!(
                (broadened[i] - broadened_ref[i]).abs() < 1e-14,
                "forward pass mismatch at bin {i}: {} vs {}",
                broadened[i],
                broadened_ref[i]
            );
        }
    }

    /// A2: Stability across temperature range (100K, 500K, 1000K).
    #[test]
    fn test_analytical_derivative_temperature_range() {
        let energies: Vec<f64> = (0..200).map(|i| 1.0 + i as f64 * 0.05).collect();
        let xs = test_resonance_xs(&energies, 6.67, 0.025, 5000.0);

        for &temp in &[100.0, 500.0, 1000.0] {
            let params = DopplerParams::new(temp, 238.051).unwrap();
            let (_broadened, dxs_dt) =
                doppler_broaden_with_derivative(&energies, &xs, &params).unwrap();

            // FD reference
            let dt = 1e-4 * (1.0 + temp);
            let p_up = DopplerParams::new(temp + dt, 238.051).unwrap();
            let p_down = DopplerParams::new((temp - dt).max(0.1), 238.051).unwrap();
            let actual_2dt = (temp + dt) - (temp - dt).max(0.1);
            let xs_up = doppler_broaden(&energies, &xs, &p_up).unwrap();
            let xs_down = doppler_broaden(&energies, &xs, &p_down).unwrap();

            // Same combined metric as A1: relative for significant, absolute for small.
            let max_deriv: f64 = (0..energies.len())
                .map(|i| ((xs_up[i] - xs_down[i]) / actual_2dt).abs())
                .fold(0.0f64, f64::max);
            let mut max_rel_err = 0.0f64;
            for i in 0..energies.len() {
                let fd = (xs_up[i] - xs_down[i]) / actual_2dt;
                if fd.abs() < max_deriv * 0.01 {
                    continue; // skip small derivatives
                }
                max_rel_err = max_rel_err.max(((dxs_dt[i] - fd) / fd).abs());
            }
            assert!(
                max_rel_err < 1e-6,
                "T={temp}K: analytical vs FD max rel error = {max_rel_err:.2e}"
            );
        }
    }

    /// A3: Different AWR (Hf-178, heavier nucleus).
    #[test]
    fn test_analytical_derivative_hf178() {
        let energies: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.1).collect();
        let xs = test_resonance_xs(&energies, 7.8, 0.05, 3000.0);
        let params = DopplerParams::new(293.6, 177.95).unwrap();

        let (_broadened, dxs_dt) =
            doppler_broaden_with_derivative(&energies, &xs, &params).unwrap();

        let dt = 1e-4 * (1.0 + 293.6);
        let p_up = DopplerParams::new(293.6 + dt, 177.95).unwrap();
        let p_down = DopplerParams::new(293.6 - dt, 177.95).unwrap();
        let xs_up = doppler_broaden(&energies, &xs, &p_up).unwrap();
        let xs_down = doppler_broaden(&energies, &xs, &p_down).unwrap();

        let max_deriv: f64 = (0..energies.len())
            .map(|i| ((xs_up[i] - xs_down[i]) / (2.0 * dt)).abs())
            .fold(0.0f64, f64::max);
        let mut max_rel_err = 0.0f64;
        for i in 0..energies.len() {
            let fd = (xs_up[i] - xs_down[i]) / (2.0 * dt);
            if fd.abs() < max_deriv * 0.01 {
                continue;
            }
            max_rel_err = max_rel_err.max(((dxs_dt[i] - fd) / fd).abs());
        }
        assert!(
            max_rel_err < 1e-6,
            "Hf-178: analytical vs FD max rel error = {max_rel_err:.2e}"
        );
    }

    /// A4: Compare against SAMMY-style FD (±2% Doppler width perturbation).
    #[test]
    fn test_analytical_derivative_vs_sammy_style_fd() {
        let energies: Vec<f64> = (0..200).map(|i| 1.0 + i as f64 * 0.05).collect();
        let xs = test_resonance_xs(&energies, 6.67, 0.025, 5000.0);
        let params = DopplerParams::new(293.6, 238.051).unwrap();

        let (_broadened, dxs_dt) =
            doppler_broaden_with_derivative(&energies, &xs, &params).unwrap();

        // SAMMY-style: perturb Doppler width by ±2%
        let del = 0.02;
        let _u = params.u(); // retained for documentation; T_up/T_down use (1±del)²
        // D_up = u * (1 + del), corresponds to T_up such that √(kT_up/AWR) = u*(1+del)
        // T_up = T * (1+del)²
        let t_up = params.temperature_k * (1.0 + del) * (1.0 + del);
        let t_down = params.temperature_k * (1.0 - del) * (1.0 - del);
        let p_up = DopplerParams::new(t_up, params.awr).unwrap();
        let p_down = DopplerParams::new(t_down, params.awr).unwrap();
        let xs_up = doppler_broaden(&energies, &xs, &p_up).unwrap();
        let xs_down = doppler_broaden(&energies, &xs, &p_down).unwrap();

        // SAMMY: ∂σ/∂D = (σ(1.02·D) - σ(0.98·D)) / (0.04·D)
        // ∂σ/∂T = ∂σ/∂D · D/(2T)
        // Combined: ∂σ/∂T ≈ (σ(T_up) - σ(T_down)) / (T_up - T_down)
        let actual_dt = t_up - t_down;

        // SAMMY FD has O(del²) = O(4e-4) truncation error, so we allow
        // slightly looser tolerance. Use same combined metric.
        let max_deriv: f64 = (0..energies.len())
            .map(|i| ((xs_up[i] - xs_down[i]) / actual_dt).abs())
            .fold(0.0f64, f64::max);
        let mut max_rel_err = 0.0f64;
        for i in 0..energies.len() {
            let sammy_fd = (xs_up[i] - xs_down[i]) / actual_dt;
            if sammy_fd.abs() < max_deriv * 0.01 {
                continue; // skip small derivatives
            }
            let rel_err = ((dxs_dt[i] - sammy_fd) / sammy_fd).abs();
            max_rel_err = max_rel_err.max(rel_err);
        }
        assert!(
            max_rel_err < 1e-3,
            "analytical vs SAMMY-style FD max rel error = {max_rel_err:.2e}, expected < 1e-3"
        );
    }

    /// Kernel-discrimination pin: the production kernel must be the FULL
    /// FGM kernel (Eq. III B1.7, w²-weighted), verified against in-test
    /// Simpson references for BOTH kernels.  The SAMMY ex001 oracle alone
    /// is too loose (grid artifacts dominate) to detect a kernel-form
    /// regression; this test fails loudly on one.
    ///
    /// (a) Smooth limit: the w¹ (legacy) kernel preserves a constant σ
    ///     (quadrature-noise level), while the full kernel yields
    ///     σ·(1 + u²/2v²) — the kT/(2·AWR·E) physical low-energy upturn.
    /// (b) Resonance line shape (U-238-like Lorentzian: E_r = 6.674 eV,
    ///     Γ = 0.027 eV, AWR = 236.0058, 300 K): the w¹-vs-full deviation
    ///     at the ±Δ_D flanks is FIRST order — antisymmetric, within
    ///     [0.1%, 1%] — and second-order small at the peak.  These two
    ///     reference-vs-reference pins are kernel-independent analytics.
    /// (c) The production `doppler_broaden` agrees with the FULL-kernel
    ///     reference at those points (< 5e-4) AND differs from the legacy
    ///     w¹ reference by the first-order flank skew with the correct
    ///     signs — so a silent regression to the legacy kernel fails this
    ///     test in the discrimination direction.
    #[test]
    fn kernel_error_scales_pinned_vs_full_fgm_reference() {
        use std::f64::consts::PI;

        let awr = 236.0058;
        let t_k = 300.0;
        let e_r = 6.674; // eV
        let gamma = 0.027; // eV (total width scale; Lorentzian discriminator)
        let params = DopplerParams::new(t_k, awr).unwrap();
        let u = params.u();

        // Reference quadrature of the analytic integrand on [v−12u, v+12u]
        // (Simpson).  The negative-velocity image branch is omitted: it is
        // suppressed by exp(−(v/u)²) with v/u ≈ 247 here.  `full` selects
        // the full FGM kernel (w², divide by v² — the production kernel)
        // vs the legacy w¹ kernel (divide by v).
        let broadened_ref = |sigma: &dyn Fn(f64) -> f64, e: f64, full: bool| -> f64 {
            let v = e.sqrt();
            let (lo, hi) = (v - 12.0 * u, v + 12.0 * u);
            // Enforce the image-branch-omission precondition: the window must
            // stay in positive-w territory, which also bounds the omitted
            // image term at ≤ exp(−(v/u)²) ≤ exp(−144) — far below quadrature
            // noise.  If the test parameters (E, T, AWR) ever change such that
            // this fails, implement the negative-w branch instead.
            assert!(
                lo > 0.0,
                "reference quadrature window crosses w = 0 (v/u = {:.1} < 12); \
                 the omitted image branch is no longer negligible",
                v / u
            );
            let n = 4800usize; // even (Simpson); h = 0.005·u
            let h = (hi - lo) / n as f64;
            let f = |w: f64| -> f64 {
                let g = (-((v - w) / u).powi(2)).exp();
                let wp = if full { w * w } else { w };
                g * wp * sigma(w * w)
            };
            let mut s = f(lo) + f(hi);
            for i in 1..n {
                let w = lo + i as f64 * h;
                s += f(w) * if i % 2 == 1 { 4.0 } else { 2.0 };
            }
            let integral = s * h / 3.0;
            let norm = u * PI.sqrt() * if full { v * v } else { v };
            integral / norm
        };

        // (a) Constant cross-section.
        let const_sigma = |_e: f64| 1.0_f64;
        let apx_const = broadened_ref(&const_sigma, e_r, false);
        let full_const = broadened_ref(&const_sigma, e_r, true);
        let u2_over_2v2 = u * u / (2.0 * e_r); // v² = E
        assert!(
            (apx_const - 1.0).abs() < 1e-8,
            "legacy w¹ kernel reference must preserve constant σ (got dev {:.3e})",
            apx_const - 1.0
        );
        assert!(
            ((full_const - 1.0) - u2_over_2v2).abs() < 0.05 * u2_over_2v2,
            "full kernel on constant σ must give 1 + u²/2v² = 1 + {:.3e} (got 1 + {:.3e})",
            u2_over_2v2,
            full_const - 1.0
        );

        // (b) Lorentzian line shape: first-order antisymmetric flank skew.
        let lorentzian = |e: f64| {
            let x = (e - e_r) / (gamma / 2.0);
            1.0 / (1.0 + x * x)
        };
        let delta_d = params.doppler_width(e_r);
        let dev_at = |e: f64| -> f64 {
            let apx = broadened_ref(&lorentzian, e, false);
            let full = broadened_ref(&lorentzian, e, true);
            (full - apx) / full
        };
        let dev_lo = dev_at(e_r - delta_d);
        let dev_hi = dev_at(e_r + delta_d);
        let dev_peak = dev_at(e_r);
        assert!(
            dev_lo > 1.0e-3 && dev_lo < 1.0e-2,
            "low-flank deviation must be first-order positive (~0.3%), got {dev_lo:.3e}"
        );
        assert!(
            dev_hi < -1.0e-3 && dev_hi > -1.0e-2,
            "high-flank deviation must be first-order negative (~−0.3%), got {dev_hi:.3e}"
        );
        assert!(
            dev_peak.abs() < 5.0e-5,
            "peak deviation must be second-order small, got {dev_peak:.3e}"
        );

        // (c) The shipping doppler_broaden matches the FULL-kernel reference
        // at the same energies (grid fine enough that production quadrature
        // error ≪ the 0.3% flank signal), and DIFFERS from the legacy w¹
        // reference by the first-order flank skew with the correct signs —
        // a silent regression to the legacy kernel trips the second check.
        let n_grid = 3001usize;
        let (e_lo, e_hi) = (e_r - 1.2, e_r + 1.2);
        let energies: Vec<f64> = (0..n_grid)
            .map(|i| e_lo + (e_hi - e_lo) * i as f64 / (n_grid - 1) as f64)
            .collect();
        let xs: Vec<f64> = energies.iter().map(|&e| lorentzian(e)).collect();
        let broadened = doppler_broaden(&energies, &xs, &params).unwrap();
        // expect_skew: Some(true) = low flank (production above the legacy
        // kernel), Some(false) = high flank (below), None = peak (no
        // first-order term).
        for (target, expect_skew) in [
            (e_r - delta_d, Some(true)),
            (e_r, None),
            (e_r + delta_d, Some(false)),
        ] {
            let idx = energies
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - target).abs().total_cmp(&(*b - target).abs()))
                .map(|(i, _)| i)
                .unwrap();
            let e_eval = energies[idx];
            let ref_full = broadened_ref(&lorentzian, e_eval, true);
            let rel_full = (broadened[idx] - ref_full).abs() / ref_full;
            assert!(
                rel_full < 5.0e-4,
                "production doppler_broaden vs FULL-kernel reference at \
                 E = {e_eval:.4} eV: rel dev {rel_full:.3e} (must be ≪ the 3e-3 flank signal)"
            );
            let ref_legacy = broadened_ref(&lorentzian, e_eval, false);
            let dev_legacy = (broadened[idx] - ref_legacy) / ref_legacy;
            match expect_skew {
                Some(true) => assert!(
                    dev_legacy > 1.0e-3 && dev_legacy < 1.0e-2,
                    "low flank: production must sit first-order ABOVE the \
                     legacy w¹ kernel (got {dev_legacy:.3e})"
                ),
                Some(false) => assert!(
                    dev_legacy < -1.0e-3 && dev_legacy > -1.0e-2,
                    "high flank: production must sit first-order BELOW the \
                     legacy w¹ kernel (got {dev_legacy:.3e})"
                ),
                None => assert!(
                    dev_legacy.abs() < 1.0e-3,
                    "peak: production-vs-legacy must have no first-order term \
                     (got {dev_legacy:.3e})"
                ),
            }
        }

        // (d) Production-level pins on the two analytic full-kernel
        // signatures stated in the module docs.
        //
        // 1/v: Y₂(w) = w²·(c/w) = c·w is linear in w, so the PW-linear
        // quadrature integrates it exactly and the self-normalization
        // returns c/v unchanged (the grid extension also extrapolates by
        // 1/v, so even the window edges are exact).
        let inv_v_xs: Vec<f64> = energies.iter().map(|&e| 3.0 / e.sqrt()).collect();
        let inv_v_broad = doppler_broaden(&energies, &inv_v_xs, &params).unwrap();
        for i in (n_grid / 10)..(9 * n_grid / 10) {
            let rel = (inv_v_broad[i] - inv_v_xs[i]).abs() / inv_v_xs[i];
            assert!(
                rel < 1.0e-9,
                "1/v cross-section must be preserved exactly at E = {:.4} eV \
                 (got rel dev {rel:.3e})",
                energies[i]
            );
        }
        // Constant σ: the full kernel produces the physical low-energy
        // upturn σ·(1 + u²/2v²); at these parameters u²/2E ≈ 8.2e-6.
        let const_xs = vec![2.0f64; n_grid];
        let const_broad = doppler_broaden(&energies, &const_xs, &params).unwrap();
        for i in (n_grid / 10)..(9 * n_grid / 10) {
            let e_i = energies[i];
            let expected = 2.0 * (1.0 + u * u / (2.0 * e_i));
            let rel = (const_broad[i] - expected).abs() / expected;
            assert!(
                rel < 1.0e-7,
                "constant σ must broaden to σ·(1 + u²/2v²) at E = {:.4} eV \
                 (got rel dev {rel:.3e} from the expected upturn)",
                e_i
            );
        }
    }

    /// Low-energy / light-target derivative check that EXERCISES the
    /// negative-velocity image branch in `doppler_broaden_with_derivative`
    /// (its rebuilt extended grid duplicates `doppler_broaden`'s; every
    /// other derivative test runs at E ≥ 1 eV with AWR ≥ 177, where the
    /// branch is unreachable).  The FD side of the comparison goes through
    /// `doppler_broaden`'s copy of the branch — which the tr165 SAMMY
    /// baseline anchors at low energies — so a sign or weight defect in
    /// the derivative twin's copy breaks the FD agreement here.
    #[test]
    fn test_analytical_derivative_vs_fd_low_energy_image_branch() {
        // AWR = 1, 300 K: u ≈ 0.161 √eV, so 6u ≈ 0.965 √eV and grids
        // starting below E = (6u)² ≈ 0.93 eV enter the image branch.
        let energies: Vec<f64> = (0..400).map(|i| 0.05 + i as f64 * 0.005).collect();
        let xs = test_resonance_xs(&energies, 1.0, 0.05, 100.0);
        let params = DopplerParams::new(300.0, 1.0).unwrap();

        // Precondition: the extended grid must actually reach w < 0.
        assert!(
            energies[0].sqrt() < DOPPLER_N_SIGMA * params.u(),
            "grid must enter the negative-velocity image branch \
             (v_min = {:.4}, 6u = {:.4})",
            energies[0].sqrt(),
            DOPPLER_N_SIGMA * params.u()
        );

        let (_broadened, dxs_dt) =
            doppler_broaden_with_derivative(&energies, &xs, &params).unwrap();

        let dt = 1e-4 * (1.0 + params.temperature_k());
        let params_up = DopplerParams::new(params.temperature_k() + dt, params.awr()).unwrap();
        let params_down =
            DopplerParams::new((params.temperature_k() - dt).max(0.1), params.awr()).unwrap();
        let actual_2dt = (params.temperature_k() + dt) - (params.temperature_k() - dt).max(0.1);

        let xs_up = doppler_broaden(&energies, &xs, &params_up).unwrap();
        let xs_down = doppler_broaden(&energies, &xs, &params_down).unwrap();

        let max_deriv: f64 = (0..energies.len())
            .map(|i| ((xs_up[i] - xs_down[i]) / actual_2dt).abs())
            .fold(0.0f64, f64::max);
        let abs_tol = max_deriv * 1e-4;

        let mut max_rel_err = 0.0f64;
        let mut n_significant = 0;
        for i in 0..energies.len() {
            let fd = (xs_up[i] - xs_down[i]) / actual_2dt;
            if fd.abs() < 1e-15 {
                continue;
            }
            if fd.abs() > max_deriv * 0.01 {
                let rel_err = ((dxs_dt[i] - fd) / fd).abs();
                max_rel_err = max_rel_err.max(rel_err);
                n_significant += 1;
            } else {
                let abs_err = (dxs_dt[i] - fd).abs();
                assert!(
                    abs_err < abs_tol,
                    "E={:.3}: abs error {:.2e} exceeds tol {:.2e}",
                    energies[i],
                    abs_err,
                    abs_tol
                );
            }
        }
        assert!(n_significant > 50, "too few significant-derivative points");
        // Tolerance is the FD noise floor on this grid, not 1e-6 as in the
        // high-energy tests: the extended velocity grid is itself
        // u-dependent (v_min − 6u start, u-scaled spacing, ceil()'d node
        // count), so the two FD evaluations at T ± dt integrate over
        // slightly different node sets — measured noise 4.0e-5 here, where
        // u/v reaches ~0.7.  A sign or weight defect in the image branch
        // would appear at ≥ 1e-3 (the negative-w contribution is
        // ~1e-3–1e-2 of σ_D on this grid), so the 1e-4 gate still
        // discriminates by ≥ 10×.
        assert!(
            max_rel_err < 1e-4,
            "analytical vs FD max rel error = {max_rel_err:.2e} on the \
             image-branch grid, expected < 1e-4"
        );
    }
}
