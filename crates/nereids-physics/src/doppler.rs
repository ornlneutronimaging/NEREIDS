//! Doppler broadening via the Free Gas Model (FGM).
//!
//! The FGM treats target atoms as a free ideal gas at temperature T.
//! The Doppler-broadened cross-section is obtained by averaging the
//! unbroadened cross-section over the Maxwell-Boltzmann velocity
//! distribution of the target atoms.
//!
//! ## SAMMY Reference
//! - Manual Section III.B.1 (Free-Gas Model of Doppler Broadening)
//! - `dop/` module (Leal-Hwang implementation)
//!
//! ## Method
//!
//! We implement the exact FGM integral in velocity space (SAMMY Eq. III B1.7):
//!
//!   v·σ_D(v²) = (1/(u√π)) ∫ exp(-(v-w)²/u²) · w · s(w) dw
//!
//! where v = √E, u = √(k_B·T / AWR), and:
//!   s(w) =  σ(w²)  for w > 0
//!   s(w) = -σ(w²)  for w < 0
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
#[derive(Debug)]
pub enum DopplerParamsError {
    /// AWR must be strictly positive.
    InvalidAwr(f64),
    /// Temperature must be finite (may be zero for "no broadening").
    NonFiniteTemperature(f64),
}

impl fmt::Display for DopplerParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAwr(v) => write!(f, "AWR must be positive, got {v}"),
            Self::NonFiniteTemperature(v) => write!(f, "temperature must be finite, got {v}"),
        }
    }
}

impl std::error::Error for DopplerParamsError {}

/// Doppler broadening parameters.
#[derive(Debug, Clone, Copy)]
pub struct DopplerParams {
    /// Effective sample temperature in Kelvin.
    pub temperature_k: f64,
    /// Atomic weight ratio (target mass / neutron mass) from ENDF.
    pub awr: f64,
}

impl DopplerParams {
    /// Create validated Doppler parameters.
    ///
    /// # Errors
    /// Returns `DopplerParamsError::InvalidAwr` if `awr <= 0.0` or is NaN.
    /// Returns `DopplerParamsError::NonFiniteTemperature` if `temperature_k`
    /// is NaN or infinity (zero is allowed — it means "no broadening").
    pub fn new(temperature_k: f64, awr: f64) -> Result<Self, DopplerParamsError> {
        if !awr.is_finite() || awr <= 0.0 {
            return Err(DopplerParamsError::InvalidAwr(awr));
        }
        if !temperature_k.is_finite() {
            return Err(DopplerParamsError::NonFiniteTemperature(temperature_k));
        }
        Ok(Self { temperature_k, awr })
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

/// Apply FGM Doppler broadening to cross-section data.
///
/// The cross-sections are broadened in velocity space using the exact
/// Free Gas Model integral from SAMMY manual Eq. III B1.7.
///
/// # Arguments
/// * `energies` — Energy grid in eV (must be positive and sorted ascending).
/// * `cross_sections` — Unbroadened cross-sections in barns at each energy point.
/// * `params` — Doppler broadening parameters (temperature and AWR).
///
/// # Returns
/// Doppler-broadened cross-sections in barns on the same energy grid.
///
/// # Algorithm
/// 1. Convert energy grid to velocity space (v = √E).
/// 2. Build extended grid including negative velocities for the FGM integral.
/// 3. Compute the integrand Y(w) = w · s(w) on the extended grid.
/// 4. For each output velocity, evaluate the Gaussian convolution integral.
/// 5. Transform back: σ_D(E) = result / √E.
pub fn doppler_broaden(
    energies: &[f64],
    cross_sections: &[f64],
    params: &DopplerParams,
) -> Vec<f64> {
    assert_eq!(energies.len(), cross_sections.len());

    if params.temperature_k <= 0.0 || energies.is_empty() {
        return cross_sections.to_vec();
    }

    let u = params.u();
    if u < NEAR_ZERO_FLOOR {
        return cross_sections.to_vec();
    }

    let n = energies.len();

    // Convert to velocity grid: v_i = sqrt(E_i)
    let velocities: Vec<f64> = energies.iter().map(|&e| e.sqrt()).collect();

    // Build the integrand Y(w) = w * s(w) on the velocity grid.
    // For positive v: Y(v) = v * σ(v²)
    // We also need negative velocity points where Y(-v) = -v * s(-v) = -v * (-σ(v²)) = v * σ(v²)
    // So Y(w) = |w| * σ(w²) for both positive and negative w.
    // Actually from Eq. III B1.6: s(w) = σ(w²) for w>0, s(w) = -σ(w²) for w<0
    // So Y(w) = w * s(w) = w * σ(w²) for w>0, Y(w) = w * (-σ(w²)) = -w * σ(w²) for w<0
    // But since w<0, -w>0, so Y(w) = |w| * σ(w²) = |w| * σ(|w|²)
    // This means Y(w) = |w| * σ(|w|²) for all w, i.e., Y is an even function.

    // Determine how many negative velocity points we need.
    // We need points down to v_min - N_sigma * u, which may go negative.
    let v_min = velocities[0];
    let v_neg_limit = v_min - DOPPLER_N_SIGMA * u;

    // Build extended velocity grid: negative points (if needed) + positive points.
    let mut ext_v: Vec<f64> = Vec::new();
    let mut ext_y: Vec<f64> = Vec::new();

    if v_neg_limit < 0.0 {
        // We need negative velocity points.
        // Use the same spacing as the low-energy end of the positive grid,
        // but in velocity space (uniform dv).
        let dv = if n > 1 {
            (velocities[1] - velocities[0]).max(u * 0.1)
        } else {
            u * 0.5
        };

        // Add negative velocity points from v_neg_limit to -dv
        let mut v = v_neg_limit;
        while v < -NEGATIVE_VELOCITY_FLOOR {
            ext_v.push(v);
            // Y(w) = |w| * σ(|w|²) for negative w
            // σ at E = w² — interpolate from the positive grid
            let e = v * v;
            let sigma = interpolate_cross_section(energies, cross_sections, e);
            ext_y.push(v.abs() * sigma); // Y is even
            v += dv;
        }

        // Add v = 0 point
        ext_v.push(0.0);
        ext_y.push(0.0);
    }

    // Add the positive velocity points
    for i in 0..n {
        ext_v.push(velocities[i]);
        ext_y.push(velocities[i] * cross_sections[i]);
    }

    // Add points beyond the highest velocity if needed
    let v_max = velocities[n - 1];
    let v_max_limit = v_max + DOPPLER_N_SIGMA * u;
    if v_max < v_max_limit {
        let dv = if n > 1 {
            (velocities[n - 1] - velocities[n - 2]).max(u * 0.1)
        } else {
            u * 0.5
        };
        let mut v = v_max + dv;
        while v <= v_max_limit {
            ext_v.push(v);
            let e = v * v;
            let sigma = interpolate_cross_section(energies, cross_sections, e);
            ext_y.push(v * sigma);
            v += dv;
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
    // using the SAMMY FGM formula (manual Sec III.B.1):
    //
    //   σ_D(E) = (1/E) × [Σ w_norm_i × v_i² × σ(E_i)]
    //
    // where w_norm_i are Gaussian weights normalized to sum to 1,
    // v_i = ext_v[i], and σ(E_i) is the cross-section at E_i = v_i².
    //
    // We also compute equivalent raw Gaussian weights for the extended
    // velocity grid, then normalize, apply v² factor, multiply by σ,
    // and divide by E.
    //
    // For negative velocities: E_i = v_i², σ(E_i) is the cross-section
    // at energy v_i² (same as for positive v_i).

    // Build the cross-section array on the extended grid.
    // ext_y stores |w|×σ(w²), so σ(w²) = ext_y[j] / |ext_v[j]|
    // for non-zero velocities.
    let ext_sigma: Vec<f64> = (0..n_ext)
        .map(|j| {
            let w = ext_v[j];
            if w.abs() < NEAR_ZERO_FLOOR {
                // At v=0, cross-section is the extrapolated value
                if !energies.is_empty() {
                    interpolate_cross_section(energies, cross_sections, 0.0)
                } else {
                    0.0
                }
            } else {
                ext_y[j] / w.abs()
            }
        })
        .collect();

    let mut broadened = vec![0.0f64; n];

    for i in 0..n {
        let v = velocities[i];
        let e = energies[i];
        if v < NEAR_ZERO_FLOOR || e < NEAR_ZERO_FLOOR {
            broadened[i] = cross_sections[i];
            continue;
        }

        // O(N×W) optimisation: use binary search to restrict the inner loop
        // to the Gaussian window [v − n_sigma·u, v + n_sigma·u].  The
        // velocity-space Doppler width u is energy-independent, so the window
        // width W is constant across all output energies.
        let v_lo = v - DOPPLER_N_SIGMA * u;
        let v_hi = v + DOPPLER_N_SIGMA * u;
        let j_lo = ext_v.partition_point(|&w| w < v_lo);
        let j_hi = ext_v.partition_point(|&w| w <= v_hi);

        // Single-pass accumulation: compute Gaussian-weighted sum and
        // normalisation simultaneously, avoiding a per-point Vec allocation.
        // Weight_j = exp(-(v - w_j)²/u²) × (dw_j)
        // where dw_j is the trapezoidal width at point j.
        let mut sum_weights = 0.0f64;
        let mut result = 0.0f64;

        for j in j_lo..j_hi {
            let arg = (v - ext_v[j]) / u;
            if arg * arg > 100.0 {
                continue;
            }
            let g = (-arg * arg).exp();

            // Trapezoidal half-widths
            let dw_left = if j > 0 {
                (ext_v[j] - ext_v[j - 1]) * 0.5
            } else {
                0.0
            };
            let dw_right = if j < n_ext - 1 {
                (ext_v[j + 1] - ext_v[j]) * 0.5
            } else {
                0.0
            };
            let dw = dw_left + dw_right;

            let w = g * dw;
            sum_weights += w;
            // v_j² × σ(E_j) — same as the original two-pass formula:
            //   result += w_norm × v_j² × ext_sigma[j]
            // but deferred normalisation (divide by sum_weights after the loop).
            let vj2 = ext_v[j] * ext_v[j]; // v_j² = E_j
            result += w * vj2 * ext_sigma[j];
        }

        if sum_weights < DIVISION_FLOOR {
            broadened[i] = cross_sections[i];
            continue;
        }

        // σ_D(E) = (1/E) × Σ [w_norm × v_j² × σ(E_j)]
        //        = (1/E) × (Σ w × v_j² × σ(E_j)) / (Σ w)
        broadened[i] = (result / sum_weights) / e;

        // Ensure non-negative
        if broadened[i] < 0.0 {
            broadened[i] = 0.0;
        }
    }

    broadened
}

/// Linear interpolation of cross-section at an arbitrary energy.
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

    #[test]
    fn test_doppler_width_u238() {
        // SAMMY reports: Doppler width at 6.075 eV = 0.05159437 eV for U-238 at 300K
        // AWR = 238.050972, T = 300 K
        let params = DopplerParams {
            temperature_k: 300.0,
            awr: 238.050972,
        };
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
        let params = DopplerParams {
            temperature_k: 300.0,
            awr: 10.0,
        };
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
        let params = DopplerParams {
            temperature_k: 0.0,
            awr: 238.0,
        };
        let broadened = doppler_broaden(&energies, &xs, &params);
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

        let params = DopplerParams {
            temperature_k: 300.0,
            awr: 238.0,
        };
        let broadened = doppler_broaden(&energies, &xs, &params);

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
        use nereids_core::types::Isotope;
        use nereids_endf::resonance::{
            LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
        };

        // Build the ex001 resonance data: single resonance at 10 eV.
        // SAMMY par file widths are in meV — convert to eV (×0.001).
        let data = ResonanceData {
            isotope: Isotope::new(1, 10).unwrap(),
            za: 1010,
            awr: 10.0,
            ranges: vec![ResonanceRange {
                energy_low: 0.0,
                energy_high: 100.0,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: 2.908,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 10.0,
                    apl: 2.908,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 10.0,
                        j: 0.5,
                        gn: 0.5e-3, // 0.5 meV → eV
                        gg: 1.0e-3, // 1.0 meV → eV
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
            }],
        };

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
        let params = DopplerParams {
            temperature_k: 300.0,
            awr: 10.0,
        };
        let broadened = doppler_broaden(&energies, &unbroadened, &params);

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
            eprintln!(
                "  E={:.4} eV: ours={:.4}, SAMMY={:.4}, ratio={:.4}",
                e_ref,
                sigma_us,
                sigma_ref,
                sigma_us / sigma_ref
            );
            max_rel_err = max_rel_err.max(rel_err);
        }
        // Allow up to 5% relative error (trapezoidal integration + constant differences).
        assert!(
            max_rel_err < 0.05,
            "Max relative error = {:.2}% (exceeds 5%)",
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

        let params = DopplerParams {
            temperature_k: 300.0,
            awr: 100.0,
        };
        let broadened = doppler_broaden(&energies, &xs, &params);

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
}
