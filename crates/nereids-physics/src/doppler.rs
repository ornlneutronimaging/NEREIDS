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

use nereids_core::constants;

/// Doppler broadening parameters.
#[derive(Debug, Clone, Copy)]
pub struct DopplerParams {
    /// Effective sample temperature in Kelvin.
    pub temperature_k: f64,
    /// Atomic weight ratio (target mass / neutron mass) from ENDF.
    pub awr: f64,
}

impl DopplerParams {
    /// Velocity-space Doppler width u = √(k_B·T / AWR).
    ///
    /// This is the standard deviation of the Gaussian kernel in √eV units.
    pub fn u(&self) -> f64 {
        (constants::BOLTZMANN_EV_PER_K * self.temperature_k / self.awr).sqrt()
    }

    /// Energy-dependent Doppler width Δ_D(E) = √(4·k_B·T·E / AWR).
    ///
    /// This is the width that SAMMY reports in the .lpt file.
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
    if u < 1e-30 {
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
    let n_sigma = 6.0; // Integration extends 6σ beyond the range
    let v_min = velocities[0];
    let v_neg_limit = v_min - n_sigma * u;

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
        while v < -1e-15 {
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
    let v_max_limit = v_max + n_sigma * u;
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
            if w.abs() < 1e-30 {
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
        if v < 1e-30 || e < 1e-30 {
            broadened[i] = cross_sections[i];
            continue;
        }

        // Compute trapezoidal Gaussian weights on the extended grid.
        // Weight_j = exp(-(v - w_j)²/u²) × (dw_j)
        // where dw_j is the trapezoidal width at point j.
        let mut weights = vec![0.0f64; n_ext];
        let mut sum_weights = 0.0;

        // Compute raw Gaussian weights using trapezoidal integration.
        for j in 0..n_ext {
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

            weights[j] = g * dw;
            sum_weights += weights[j];
        }

        if sum_weights < 1e-50 {
            broadened[i] = cross_sections[i];
            continue;
        }

        // Normalize weights, apply v² factor, sum with σ, divide by E.
        // σ_D(E) = (1/E) × Σ [w_norm × v_j² × σ(E_j)]
        let inv_sum = 1.0 / sum_weights;
        let mut result = 0.0;
        for j in 0..n_ext {
            if weights[j] == 0.0 {
                continue;
            }
            let w_norm = weights[j] * inv_sum;
            let vj2 = ext_v[j] * ext_v[j]; // v_j² = E_j
            result += w_norm * vj2 * ext_sigma[j];
        }
        broadened[i] = result / e;

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

    if energy <= energies[0] {
        // Extrapolate using 1/v law: σ ∝ 1/√E
        if energies[0] > 1e-30 {
            return cross_sections[0] * (energies[0] / energy).sqrt();
        }
        return cross_sections[0];
    }

    if energy >= energies[energies.len() - 1] {
        // Extrapolate using 1/v law
        let last = energies.len() - 1;
        if energy > 1e-30 {
            return cross_sections[last] * (energies[last] / energy).sqrt();
        }
        return cross_sections[last];
    }

    // Binary search for the interval
    let idx = match energies.binary_search_by(|e| e.partial_cmp(&energy).unwrap()) {
        Ok(i) => return cross_sections[i],
        Err(i) => i - 1,
    };

    // Linear interpolation
    let e0 = energies[idx];
    let e1 = energies[idx + 1];
    let s0 = cross_sections[idx];
    let s1 = cross_sections[idx + 1];
    let t = (energy - e0) / (e1 - e0);
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
            isotope: Isotope { z: 1, a: 10 },
            za: 1010,
            awr: 10.0,
            ranges: vec![ResonanceRange {
                energy_low: 0.0,
                energy_high: 100.0,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: 2.908,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 10.0,
                    apl: 2.908,
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
}
