//! Transmission forward model via the Beer-Lambert law.
//!
//! Computes theoretical neutron transmission spectra from resonance parameters,
//! applying cross-section calculation, Doppler broadening, resolution broadening,
//! and the Beer-Lambert attenuation law.
//!
//! ## Beer-Lambert Law
//!
//! For a single isotope:
//!   T(E) = exp(-n·d·σ(E))
//!
//! For multiple isotopes:
//!   T(E) = exp(-Σᵢ nᵢ·dᵢ·σᵢ(E))
//!
//! where n is number density (atoms/cm³), d is thickness (cm),
//! and σ(E) is the total cross-section in barns (1 barn = 10⁻²⁴ cm²).
//!
//! In practice, the product n·d is expressed as "areal density" in
//! atoms/barn, so T(E) = exp(-thickness × σ(E)) with thickness in atoms/barn.
//!
//! ## SAMMY Reference
//! - `cro/` and `xxx/` modules — cross-section to transmission conversion
//! - Manual Section 2 (transmission definition), Section 5 (experimental corrections)

use std::sync::atomic::{AtomicBool, Ordering};

use nereids_endf::resonance::ResonanceData;

use crate::doppler::{self, DopplerParams};
use crate::reich_moore;
use crate::resolution::{self, ResolutionFunction};

/// Compute transmission from cross-sections via Beer-Lambert law.
///
/// T(E) = exp(-thickness × σ(E))
///
/// # Arguments
/// * `cross_sections` — Total cross-sections in barns at each energy point.
/// * `thickness` — Areal density in atoms/barn (= number_density × path_length).
///
/// # Returns
/// Transmission values (0 to 1) at each energy point.
pub fn beer_lambert(cross_sections: &[f64], thickness: f64) -> Vec<f64> {
    cross_sections
        .iter()
        .map(|&sigma| (-thickness * sigma).exp())
        .collect()
}

/// Compute transmission for multiple isotopes.
///
/// T(E) = exp(-Σᵢ thicknessᵢ × σᵢ(E))
///
/// # Arguments
/// * `cross_sections_per_isotope` — Vec of cross-section arrays, one per isotope.
///   Each inner slice has the same length as the energy grid.
/// * `thicknesses` — Areal density (atoms/barn) for each isotope.
///
/// # Returns
/// Combined transmission values at each energy point.
pub fn beer_lambert_multi(cross_sections_per_isotope: &[&[f64]], thicknesses: &[f64]) -> Vec<f64> {
    assert_eq!(cross_sections_per_isotope.len(), thicknesses.len());
    assert!(!cross_sections_per_isotope.is_empty());

    let n_energies = cross_sections_per_isotope[0].len();

    (0..n_energies)
        .map(|i| {
            let total_attenuation: f64 = cross_sections_per_isotope
                .iter()
                .zip(thicknesses.iter())
                .map(|(sigma, &thick)| thick * sigma[i])
                .sum();
            (-total_attenuation).exp()
        })
        .collect()
}

/// Sample description for the forward model.
#[derive(Debug, Clone)]
pub struct SampleParams {
    /// Temperature in Kelvin (for Doppler broadening).
    pub temperature_k: f64,
    /// Isotope compositions: (resonance data, areal density in atoms/barn).
    pub isotopes: Vec<(ResonanceData, f64)>,
}

/// Optional instrument resolution parameters.
#[derive(Debug, Clone)]
pub struct InstrumentParams {
    /// Resolution broadening function (Gaussian or tabulated).
    pub resolution: ResolutionFunction,
}

/// Compute a complete theoretical transmission spectrum.
///
/// This is the main forward model that chains:
///   ENDF parameters → cross-sections → Doppler broadening → resolution → transmission
///
/// # Arguments
/// * `energies` — Energy grid in eV (sorted ascending).
/// * `sample` — Sample parameters (isotopes with areal densities, temperature).
/// * `instrument` — Optional instrument parameters (resolution broadening).
///
/// # Returns
/// Theoretical transmission spectrum on the energy grid.
pub fn forward_model(
    energies: &[f64],
    sample: &SampleParams,
    instrument: Option<&InstrumentParams>,
) -> Vec<f64> {
    let n = energies.len();
    if n == 0 {
        return vec![];
    }

    // Accumulate total attenuation: Σᵢ thicknessᵢ × σᵢ(E)
    let mut total_attenuation = vec![0.0f64; n];

    for (res_data, thickness) in &sample.isotopes {
        if *thickness <= 0.0 {
            continue;
        }

        // 1. Compute unbroadened total cross-sections
        let unbroadened: Vec<f64> = energies
            .iter()
            .map(|&e| reich_moore::cross_sections_at_energy(res_data, e).total)
            .collect();

        // 2. Apply Doppler broadening
        let after_doppler = if sample.temperature_k > 0.0 {
            let doppler_params = DopplerParams {
                temperature_k: sample.temperature_k,
                awr: res_data.awr,
            };
            doppler::doppler_broaden(energies, &unbroadened, &doppler_params)
        } else {
            unbroadened
        };

        // 3. Apply resolution broadening
        let after_resolution = if let Some(inst) = instrument {
            resolution::apply_resolution(energies, &after_doppler, &inst.resolution)
        } else {
            after_doppler
        };

        // 4. Accumulate attenuation
        for i in 0..n {
            total_attenuation[i] += thickness * after_resolution[i];
        }
    }

    // 5. Beer-Lambert: T = exp(-attenuation)
    total_attenuation.iter().map(|&att| (-att).exp()).collect()
}

/// Compute Doppler- and resolution-broadened cross-sections for each isotope.
///
/// This is the expensive physics step that should be done **once** before
/// fitting many pixels with the same isotopes and energy grid.  The result
/// feeds directly into `nereids_fitting::transmission_model::PrecomputedTransmissionModel`,
/// making per-pixel Beer-Lambert evaluation trivial.
///
/// # Arguments
/// * `energies`        — Energy grid in eV (sorted ascending).
/// * `resonance_data`  — Resonance parameters for each isotope.
/// * `temperature_k`   — Sample temperature for Doppler broadening.
/// * `instrument`      — Optional instrument resolution parameters.
/// * `cancel`          — Optional cancellation token.  When set, the function
///   returns `None` after completing the current isotope.
///
/// # Returns
/// `Some(xs)` — one cross-section vector per isotope — on success.
/// `None` if the `cancel` flag was observed between isotopes.
pub fn broadened_cross_sections(
    energies: &[f64],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
    cancel: Option<&AtomicBool>,
) -> Option<Vec<Vec<f64>>> {
    let mut result = Vec::with_capacity(resonance_data.len());

    for rd in resonance_data {
        // Check cancellation between isotopes — each per-isotope broadening
        // step can be expensive (Doppler FGM × N_energy), so we bail here
        // rather than inside the inner loop.
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
        }

        // 1. Unbroadened total cross-sections
        let unbroadened: Vec<f64> = energies
            .iter()
            .map(|&e| reich_moore::cross_sections_at_energy(rd, e).total)
            .collect();

        // 2. Doppler broadening
        let after_doppler = if temperature_k > 0.0 {
            let params = DopplerParams {
                temperature_k,
                awr: rd.awr,
            };
            doppler::doppler_broaden(energies, &unbroadened, &params)
        } else {
            unbroadened
        };

        // 3. Resolution broadening
        let xs = if let Some(inst) = instrument {
            resolution::apply_resolution(energies, &after_doppler, &inst.resolution)
        } else {
            after_doppler
        };

        result.push(xs);
    }

    Some(result)
}

/// Compute broadened cross-sections and their temperature derivative.
///
/// Returns `(sigma_k, dsigma_k_dT)` where `sigma_k[k][e]` is the
/// Doppler+resolution-broadened cross-section for isotope `k` at energy
/// index `e`, and `dsigma_k_dT[k][e]` is its central finite-difference
/// derivative with respect to temperature.
///
/// The derivative uses step size `dT = 1e-4 * (1 + T)`, which balances
/// truncation error and roundoff for the T ~ 1..2000 K regime relevant
/// to neutron resonance experiments.
///
/// # Cost
/// Three calls to `broadened_cross_sections` (at T, T+dT, T-dT).
pub fn broadened_cross_sections_with_derivative(
    energies: &[f64],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let dt = 1e-4 * (1.0 + temperature_k);
    let t_up = temperature_k + dt;
    let t_down = (temperature_k - dt).max(0.1); // stay physical
    let actual_2dt = t_up - t_down;

    let xs_center =
        broadened_cross_sections(energies, resonance_data, temperature_k, instrument, None)
            .expect("broadened_cross_sections should not be cancelled (no cancel token)");
    let xs_up = broadened_cross_sections(energies, resonance_data, t_up, instrument, None)
        .expect("broadened_cross_sections should not be cancelled (no cancel token)");
    let xs_down = broadened_cross_sections(energies, resonance_data, t_down, instrument, None)
        .expect("broadened_cross_sections should not be cancelled (no cancel token)");

    let dxs_dt: Vec<Vec<f64>> = xs_up
        .iter()
        .zip(xs_down.iter())
        .map(|(up, down)| {
            up.iter()
                .zip(down.iter())
                .map(|(&u, &d)| (u - d) / actual_2dt)
                .collect()
        })
        .collect();

    (xs_center, dxs_dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::types::Isotope;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};

    fn u238_single_resonance() -> ResonanceData {
        ResonanceData {
            isotope: Isotope::new(92, 238),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
                    resonances: vec![Resonance {
                        energy: 6.674,
                        j: 0.5,
                        gn: 1.493e-3,
                        gg: 23.0e-3,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
            }],
        }
    }

    #[test]
    fn test_beer_lambert_zero_thickness() {
        let xs = vec![100.0, 200.0, 300.0];
        let t = beer_lambert(&xs, 0.0);
        assert_eq!(t, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_beer_lambert_basic() {
        // σ = 100 barns, thickness = 0.01 atoms/barn
        // T = exp(-1.0) ≈ 0.3679
        let xs = vec![100.0];
        let t = beer_lambert(&xs, 0.01);
        assert!(
            (t[0] - (-1.0_f64).exp()).abs() < 1e-10,
            "T = {}, expected {}",
            t[0],
            (-1.0_f64).exp()
        );
    }

    #[test]
    fn test_beer_lambert_opaque() {
        // Very thick sample: T should be 0 (exp(-1000) underflows)
        let xs = vec![1000.0];
        let t = beer_lambert(&xs, 1.0);
        assert_eq!(t[0], 0.0, "T = {}, expected 0.0", t[0]);
    }

    #[test]
    fn test_beer_lambert_multi_additive() {
        // Two isotopes should combine additively in the exponent.
        // σ₁ = 100 barns, t₁ = 0.01 → att₁ = 1.0
        // σ₂ = 200 barns, t₂ = 0.005 → att₂ = 1.0
        // T = exp(-(1.0 + 1.0)) = exp(-2.0)
        let xs1 = vec![100.0];
        let xs2 = vec![200.0];
        let t = beer_lambert_multi(&[&xs1, &xs2], &[0.01, 0.005]);
        assert!(
            (t[0] - (-2.0_f64).exp()).abs() < 1e-10,
            "T = {}, expected {}",
            t[0],
            (-2.0_f64).exp()
        );
    }

    #[test]
    fn test_transmission_dip_at_resonance() {
        // U-238 has a huge capture resonance at 6.674 eV.
        // A thin sample should show a transmission dip there.
        let data = u238_single_resonance();
        let thickness = 0.001; // atoms/barn (thin)

        // Evaluate at a few energies
        let energies = [1.0, 3.0, 6.674, 10.0, 20.0];
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| reich_moore::cross_sections_at_energy(&data, e).total)
            .collect();
        let trans = beer_lambert(&xs, thickness);

        // At 6.674 eV (on resonance), transmission should be much lower
        let t_on_res = trans[2];
        let t_off_res = trans[0]; // 1 eV, off resonance

        assert!(
            t_on_res < t_off_res,
            "On-resonance T ({}) should be < off-resonance T ({})",
            t_on_res,
            t_off_res
        );

        // On-resonance with huge σ (~25000 barns), T ≈ exp(-25) ≈ 0
        assert!(
            t_on_res < 0.01,
            "On-resonance T ({}) should be very small",
            t_on_res
        );
    }

    #[test]
    fn test_forward_model_no_broadening() {
        // Forward model at T=0 with no resolution should give
        // the same result as direct Beer-Lambert on unbroadened σ.
        let data = u238_single_resonance();
        let thickness = 0.001;

        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();

        // Direct calculation
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| reich_moore::cross_sections_at_energy(&data, e).total)
            .collect();
        let t_direct = beer_lambert(&xs, thickness);

        // Forward model
        let sample = SampleParams {
            temperature_k: 0.0,
            isotopes: vec![(data, thickness)],
        };
        let t_forward = forward_model(&energies, &sample, None);

        for i in 0..energies.len() {
            assert!(
                (t_direct[i] - t_forward[i]).abs() < 1e-10,
                "Mismatch at E={}: direct={}, forward={}",
                energies[i],
                t_direct[i],
                t_forward[i]
            );
        }
    }

    #[test]
    fn test_forward_model_with_broadening() {
        // Forward model with Doppler broadening should smooth out the
        // transmission dip, making it wider and shallower.
        let data = u238_single_resonance();
        let thickness = 0.0001; // Very thin (to avoid total absorption)

        let energies: Vec<f64> = (0..401).map(|i| 5.0 + (i as f64) * 0.01).collect();

        // Cold (no broadening)
        let sample_cold = SampleParams {
            temperature_k: 0.0,
            isotopes: vec![(data.clone(), thickness)],
        };
        let t_cold = forward_model(&energies, &sample_cold, None);

        // Hot (300 K Doppler)
        let sample_hot = SampleParams {
            temperature_k: 300.0,
            isotopes: vec![(data, thickness)],
        };
        let t_hot = forward_model(&energies, &sample_hot, None);

        // Find minima
        let min_cold = t_cold.iter().cloned().fold(f64::MAX, f64::min);
        let min_hot = t_hot.iter().cloned().fold(f64::MAX, f64::min);

        // Broadened dip should be shallower (higher minimum transmission)
        assert!(
            min_hot > min_cold,
            "Broadened min T ({}) should be > unbroadened min T ({})",
            min_hot,
            min_cold
        );
    }

    #[test]
    fn test_forward_model_multi_isotope() {
        // Two isotopes with different resonances should create two dips.
        let u238 = u238_single_resonance();

        // Create a fictitious second isotope with a resonance at 20 eV
        let other = ResonanceData {
            isotope: Isotope::new(1, 10),
            za: 1010,
            awr: 10.0,
            ranges: vec![ResonanceRange {
                energy_low: 0.0,
                energy_high: 100.0,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 5.0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 10.0,
                    apl: 5.0,
                    resonances: vec![Resonance {
                        energy: 20.0,
                        j: 0.5,
                        gn: 0.1,
                        gg: 0.05,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
            }],
        };

        let energies: Vec<f64> = (0..301).map(|i| 1.0 + (i as f64) * 0.1).collect();

        let sample = SampleParams {
            temperature_k: 0.0,
            isotopes: vec![(u238, 0.0001), (other, 0.0001)],
        };
        let t = forward_model(&energies, &sample, None);

        // Find the transmission near 6.674 eV (U-238 resonance)
        let idx_u238 = energies
            .iter()
            .position(|&e| (e - 6.7).abs() < 0.05)
            .unwrap();
        // Find the transmission near 20 eV (other resonance)
        let idx_other = energies
            .iter()
            .position(|&e| (e - 20.0).abs() < 0.05)
            .unwrap();
        // Off-resonance
        let idx_off = energies
            .iter()
            .position(|&e| (e - 15.0).abs() < 0.05)
            .unwrap();

        // Both dips should be visible
        assert!(
            t[idx_u238] < t[idx_off],
            "U-238 dip at 6.7 eV: T={}, off-res: T={}",
            t[idx_u238],
            t[idx_off]
        );
        assert!(
            t[idx_other] < t[idx_off],
            "Other dip at 20 eV: T={}, off-res: T={}",
            t[idx_other],
            t[idx_off]
        );
    }

    #[test]
    fn test_broadened_xs_derivative() {
        // Verify ∂σ/∂T via Richardson-like consistency: compute the derivative
        // at two different step sizes and check they agree to reasonable
        // tolerance (the internal step dT = 1e-4*(1+T) is O(h²)-accurate).
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();
        let temperature = 300.0;

        let (xs, dxs_dt) =
            broadened_cross_sections_with_derivative(&energies, &[data.clone()], temperature, None);

        // Basic shape checks
        assert_eq!(xs.len(), 1, "one isotope");
        assert_eq!(dxs_dt.len(), 1, "one isotope derivative");
        assert_eq!(xs[0].len(), energies.len());
        assert_eq!(dxs_dt[0].len(), energies.len());

        // The derivative should be non-zero near the resonance at 6.674 eV
        // where Doppler broadening has a strong effect.
        let idx_res = energies
            .iter()
            .position(|&e| (e - 6.674).abs() < 0.05)
            .unwrap();
        assert!(
            dxs_dt[0][idx_res].abs() > 0.0,
            "dσ/dT should be non-zero near resonance, got {}",
            dxs_dt[0][idx_res]
        );

        // Cross-check: compute a manual FD at a larger step (10×) and verify
        // the two derivatives are consistent (within ~1% relative error on
        // the derivative near the resonance peak).
        let big_dt = 1.0; // 1 K step — much larger than internal 0.03 K
        let xs_up =
            broadened_cross_sections(&energies, &[data.clone()], temperature + big_dt, None, None)
                .unwrap();
        let xs_down =
            broadened_cross_sections(&energies, &[data], temperature - big_dt, None, None).unwrap();

        let manual_deriv: Vec<f64> = xs_up[0]
            .iter()
            .zip(xs_down[0].iter())
            .map(|(&u, &d)| (u - d) / (2.0 * big_dt))
            .collect();

        // Compare near the resonance where the derivative is large.
        // Allow up to 5% relative difference due to O(h²) truncation.
        let deriv_fine = dxs_dt[0][idx_res];
        let deriv_coarse = manual_deriv[idx_res];
        let rel_err =
            (deriv_fine - deriv_coarse).abs() / deriv_fine.abs().max(deriv_coarse.abs()).max(1e-30);
        assert!(
            rel_err < 0.05,
            "FD derivatives at two step sizes disagree: fine={}, coarse={}, rel_err={}",
            deriv_fine,
            deriv_coarse,
            rel_err,
        );
    }
}
