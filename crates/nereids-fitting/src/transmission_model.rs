//! Transmission forward model adapter for fitting.
//!
//! Wraps the physics `forward_model` function into a `FitModel` trait object
//! that the LM optimizer can call. The fit parameters are the areal densities
//! (thicknesses) of each isotope in the sample.

use nereids_physics::transmission::{self, InstrumentParams, SampleParams};
use nereids_endf::resonance::ResonanceData;

use crate::lm::FitModel;

/// Forward model for fitting isotopic areal densities from transmission data.
///
/// The model computes T(E) for a set of isotopes with variable areal densities.
/// Each isotope's resonance data and the energy grid are fixed; only the
/// areal densities are adjusted during fitting.
pub struct TransmissionFitModel {
    /// Energy grid (eV), ascending.
    pub energies: Vec<f64>,
    /// Resonance data for each isotope.
    pub resonance_data: Vec<ResonanceData>,
    /// Sample temperature in Kelvin.
    pub temperature_k: f64,
    /// Optional instrument resolution parameters.
    pub instrument: Option<InstrumentParams>,
    /// Index mapping: which `params` indices correspond to areal densities.
    /// params[density_indices[i]] = areal density of isotope i.
    pub density_indices: Vec<usize>,
}

impl FitModel for TransmissionFitModel {
    fn evaluate(&self, params: &[f64]) -> Vec<f64> {
        let isotopes: Vec<(ResonanceData, f64)> = self
            .resonance_data
            .iter()
            .zip(self.density_indices.iter())
            .map(|(rd, &idx): (&ResonanceData, &usize)| (rd.clone(), params[idx]))
            .collect();

        let sample = SampleParams {
            temperature_k: self.temperature_k,
            isotopes,
        };

        transmission::forward_model(
            &self.energies,
            &sample,
            self.instrument.as_ref(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{self, LmConfig};
    use crate::parameters::{FitParameter, ParameterSet};
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
            }],
        }
    }

    #[test]
    fn test_recover_single_isotope_thickness() {
        let data = u238_single_resonance();
        let true_thickness = 0.0005;

        // Generate synthetic data
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
        };

        let y_obs = model.evaluate(&[true_thickness]);
        let sigma = vec![0.01; y_obs.len()]; // 1% uncertainty

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("thickness", 0.001), // initial guess 2× off
        ]);

        let result = lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

        assert!(result.converged, "Fit did not converge");
        let fitted = result.params[0];
        assert!(
            (fitted - true_thickness).abs() / true_thickness < 0.01,
            "Fitted thickness = {}, true = {}, error = {:.1}%",
            fitted,
            true_thickness,
            (fitted - true_thickness).abs() / true_thickness * 100.0,
        );
    }

    #[test]
    fn test_recover_two_isotope_thicknesses() {
        let u238 = u238_single_resonance();

        // Second isotope with resonance at 20 eV
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
            }],
        };

        let true_t1 = 0.0003;
        let true_t2 = 0.0001;

        let energies: Vec<f64> = (0..301).map(|i| 1.0 + (i as f64) * 0.1).collect();

        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![u238, other],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0, 1],
        };

        let y_obs = model.evaluate(&[true_t1, true_t2]);
        let sigma = vec![0.01; y_obs.len()];

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("U-238 thickness", 0.001),
            FitParameter::non_negative("Other thickness", 0.001),
        ]);

        let result = lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

        assert!(result.converged, "Fit did not converge after {} iterations", result.iterations);

        let (fit_t1, fit_t2) = (result.params[0], result.params[1]);
        assert!(
            (fit_t1 - true_t1).abs() / true_t1 < 0.05,
            "U-238: fitted={}, true={}, error={:.1}%",
            fit_t1, true_t1, (fit_t1 - true_t1).abs() / true_t1 * 100.0,
        );
        assert!(
            (fit_t2 - true_t2).abs() / true_t2 < 0.05,
            "Other: fitted={}, true={}, error={:.1}%",
            fit_t2, true_t2, (fit_t2 - true_t2).abs() / true_t2 * 100.0,
        );
    }
}
