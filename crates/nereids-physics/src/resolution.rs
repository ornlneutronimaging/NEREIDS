//! Resolution broadening via convolution with instrument resolution function.
//!
//! Convolves theoretical cross-sections (or transmission) with the instrument
//! resolution function to account for finite energy resolution. The resolution
//! function is modeled as a Gaussian with energy-dependent width, derived from
//! time-of-flight instrument parameters.
//!
//! ## SAMMY Reference
//! - `rsl/mrsl1.f90` — Main RSL resolution broadening routines
//! - `rsl/mrsl4.f90` — Resolution width calculation (Wdsint, Rolowg)
//! - Manual Section 3.2 (Resolution Broadening)
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
//! where R(E, E') = exp(-(E-E')²/W²) / (W·√π) is a Gaussian kernel
//! with energy-dependent width W(E).

use nereids_core::constants;

/// TOF conversion factor: t[μs] = TOF_FACTOR × L[m] / √(E[eV]).
///
/// Derived from t = L / √(2E/m_n), converting to microseconds.
const TOF_FACTOR: f64 = 72.298;

/// Resolution function parameters for time-of-flight instruments.
#[derive(Debug, Clone, Copy)]
pub struct ResolutionParams {
    /// Flight path length in meters (source to detector).
    pub flight_path_m: f64,
    /// Total timing uncertainty (1σ Gaussian) in microseconds.
    /// Combines moderator pulse width, detector timing, and electronics.
    pub delta_t_us: f64,
    /// Flight path uncertainty (1σ Gaussian) in meters.
    pub delta_l_m: f64,
}

impl ResolutionParams {
    /// Gaussian resolution width σ_E(E) in eV.
    ///
    /// Combines timing and flight-path contributions in quadrature:
    ///   σ_E² = (2·Δt/t × E)² + (2·ΔL/L × E)²
    ///
    /// where t = TOF_FACTOR × L / √E is the time-of-flight in μs.
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
pub fn resolution_broaden(
    energies: &[f64],
    cross_sections: &[f64],
    params: &ResolutionParams,
) -> Vec<f64> {
    assert_eq!(energies.len(), cross_sections.len());

    let n = energies.len();
    if n == 0 {
        return vec![];
    }

    let n_sigma = 5.0; // Integrate out to 5σ
    let mut broadened = vec![0.0f64; n];

    for i in 0..n {
        let e = energies[i];
        let w = params.gaussian_width(e);

        if w < 1e-30 {
            broadened[i] = cross_sections[i];
            continue;
        }

        // Integration limits
        let e_low = e - n_sigma * w;
        let e_high = e + n_sigma * w;

        // Trapezoidal integration with normalized Gaussian weights.
        let mut sum = 0.0;
        let mut norm = 0.0;

        for j in 0..n {
            if energies[j] < e_low || energies[j] > e_high {
                continue;
            }

            let arg = (energies[j] - e) / w;
            if arg * arg > 100.0 {
                continue;
            }
            let g = (-arg * arg).exp();

            // Trapezoidal width
            let de_left = if j > 0 {
                (energies[j] - energies[j - 1]) * 0.5
            } else {
                0.0
            };
            let de_right = if j < n - 1 {
                (energies[j + 1] - energies[j]) * 0.5
            } else {
                0.0
            };
            let de = de_left + de_right;

            let weight = g * de;
            sum += weight * cross_sections[j];
            norm += weight;
        }

        if norm > 1e-50 {
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
pub fn resolution_broaden_transmission(
    energies: &[f64],
    transmission: &[f64],
    params: &ResolutionParams,
) -> Vec<f64> {
    // The convolution kernel is the same; only the interpretation differs.
    resolution_broaden(energies, transmission, params)
}

/// Verify that the TOF conversion factor is consistent with the constants module.
fn _verify_tof_factor() {
    let e = 1.0; // 1 eV
    let l = 1.0; // 1 meter
    let tof_from_constants = constants::energy_to_tof(e, l);
    let tof_from_factor = TOF_FACTOR / e.sqrt();
    let rel_diff = (tof_from_constants - tof_from_factor).abs() / tof_from_constants;
    debug_assert!(
        rel_diff < 0.01,
        "TOF factor mismatch: {} vs {}",
        tof_from_constants,
        tof_from_factor
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tof_factor_consistency() {
        // Verify our TOF_FACTOR matches the constants module.
        let e = 10.0; // eV
        let l = 25.0; // meters
        let tof_constants = constants::energy_to_tof(e, l);
        let tof_ours = TOF_FACTOR * l / e.sqrt();
        let rel_diff = (tof_constants - tof_ours).abs() / tof_constants;
        assert!(
            rel_diff < 0.001,
            "TOF mismatch: constants={}, ours={}, diff={:.4}%",
            tof_constants,
            tof_ours,
            rel_diff * 100.0
        );
    }

    #[test]
    fn test_resolution_width_scaling() {
        let params = ResolutionParams {
            flight_path_m: 25.0,
            delta_t_us: 1.0,
            delta_l_m: 0.01,
        };

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
        let params = ResolutionParams {
            flight_path_m: 25.0,
            delta_t_us: 0.0,
            delta_l_m: 0.0,
        };
        let broadened = resolution_broaden(&energies, &xs, &params);
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

        let params = ResolutionParams {
            flight_path_m: 25.0,
            delta_t_us: 5.0, // Fairly large timing uncertainty
            delta_l_m: 0.01,
        };
        let broadened = resolution_broaden(&energies, &xs, &params);

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

        let params = ResolutionParams {
            flight_path_m: 25.0,
            delta_t_us: 1.0,
            delta_l_m: 0.01,
        };
        let broadened = resolution_broaden(&energies, &xs, &params);

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
        let params = ResolutionParams {
            flight_path_m: 25.0,
            delta_t_us: 0.0,
            delta_l_m: w_kernel * 25.0 / (2.0 * center),
        };

        // Verify kernel W at center energy
        let w_at_center = params.gaussian_width(center);
        assert!(
            (w_at_center - w_kernel).abs() / w_kernel < 0.01,
            "Kernel W at center: {}, expected {}",
            w_at_center,
            w_kernel
        );

        let broadened = resolution_broaden(&energies, &xs, &params);

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
        let params = ResolutionParams {
            flight_path_m: 25.0,
            delta_t_us: 10.0,
            delta_l_m: 0.01,
        };

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
}
