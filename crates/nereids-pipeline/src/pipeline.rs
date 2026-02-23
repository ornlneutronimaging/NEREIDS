//! Single-spectrum analysis pipeline.
//!
//! Orchestrates the full analysis chain for a single transmission spectrum:
//! ENDF loading → cross-section calculation → broadening → fitting.
//!
//! This is the building block for the spatial mapping pipeline.

use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::lm::{self, LmConfig};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::transmission_model::{PrecomputedTransmissionModel, TransmissionFitModel};
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::InstrumentParams;

/// Configuration for a single-spectrum fit.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Energy grid in eV (ascending).
    pub energies: Vec<f64>,
    /// Resonance data for each isotope to fit.
    pub resonance_data: Vec<ResonanceData>,
    /// Isotope names (for reporting).
    pub isotope_names: Vec<String>,
    /// Sample temperature in Kelvin.
    pub temperature_k: f64,
    /// Optional instrument resolution function (Gaussian or tabulated).
    pub resolution: Option<ResolutionFunction>,
    /// Initial guess for areal densities (atoms/barn), one per isotope.
    pub initial_densities: Vec<f64>,
    /// LM optimizer configuration.
    pub lm_config: LmConfig,
    /// Precomputed Doppler+resolution-broadened cross-sections, one `Vec<f64>`
    /// per isotope.  When `Some`, `fit_spectrum` skips the expensive resonance
    /// and broadening computation and uses `PrecomputedTransmissionModel` instead.
    ///
    /// Compute with `nereids_physics::transmission::broadened_cross_sections`
    /// and wrap in `Arc` before the `spatial_map` / `fit_roi` loop so the
    /// result is shared read-only across all rayon threads.
    pub precomputed_cross_sections: Option<Arc<Vec<Vec<f64>>>>,
}

/// Result of fitting a single spectrum.
#[derive(Debug, Clone)]
pub struct SpectrumFitResult {
    /// Fitted areal densities (atoms/barn), one per isotope.
    pub densities: Vec<f64>,
    /// Uncertainty on each density.
    pub uncertainties: Vec<f64>,
    /// Reduced chi-squared of the fit.
    pub reduced_chi_squared: f64,
    /// Whether the fit converged.
    pub converged: bool,
    /// Number of iterations.
    pub iterations: usize,
}

/// Fit a single measured transmission spectrum.
///
/// # Arguments
/// * `measured_t` — Measured transmission values at each energy point.
/// * `sigma` — Uncertainties on measured transmission.
/// * `config` — Fit configuration (isotopes, energy grid, etc.).
///
/// # Returns
/// Fit result with densities, uncertainties, and fit quality metrics.
pub fn fit_spectrum(measured_t: &[f64], sigma: &[f64], config: &FitConfig) -> SpectrumFitResult {
    let n_isotopes = config.resonance_data.len();

    assert_eq!(
        config.initial_densities.len(),
        n_isotopes,
        "initial_densities length ({}) must match resonance_data length ({})",
        config.initial_densities.len(),
        n_isotopes,
    );
    assert_eq!(
        measured_t.len(),
        config.energies.len(),
        "measured_t length ({}) must match energies length ({})",
        measured_t.len(),
        config.energies.len(),
    );

    let mut params = ParameterSet::new(
        config
            .initial_densities
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                FitParameter::non_negative(
                    config
                        .isotope_names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("isotope_{}", i)),
                    d,
                )
            })
            .collect(),
    );

    // Use precomputed cross-sections when available (fast path for spatial_map).
    // Fall back to the full forward-model path for single-spectrum calls.
    let result = if let Some(xs) = &config.precomputed_cross_sections {
        assert!(n_isotopes > 0, "resonance_data is empty — nothing to fit",);
        assert_eq!(
            xs.len(),
            n_isotopes,
            "precomputed_cross_sections has {} isotope(s) but resonance_data has {}",
            xs.len(),
            n_isotopes,
        );
        let n_e = config.energies.len();
        for (i, row) in xs.iter().enumerate() {
            assert_eq!(
                row.len(),
                n_e,
                "precomputed_cross_sections[{}] has {} energy points but energies has {}",
                i,
                row.len(),
                n_e,
            );
        }
        let model = PrecomputedTransmissionModel {
            cross_sections: xs.clone(),
            density_indices: (0..n_isotopes).collect(),
        };
        lm::levenberg_marquardt(&model, measured_t, sigma, &mut params, &config.lm_config)
    } else {
        let instrument = config
            .resolution
            .clone()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        let model = TransmissionFitModel {
            energies: config.energies.clone(),
            resonance_data: config.resonance_data.clone(),
            temperature_k: config.temperature_k,
            instrument,
            density_indices: (0..n_isotopes).collect(),
        };
        lm::levenberg_marquardt(&model, measured_t, sigma, &mut params, &config.lm_config)
    };

    let densities: Vec<f64> = (0..n_isotopes).map(|i| result.params[i]).collect();
    let uncertainties = result
        .uncertainties
        .unwrap_or_else(|| vec![f64::NAN; n_isotopes]);

    SpectrumFitResult {
        densities,
        uncertainties,
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::types::Isotope;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};
    use nereids_fitting::lm::FitModel;

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
                ap_table: None,
            }],
        }
    }

    #[test]
    fn test_fit_spectrum_single_isotope() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        // Generate synthetic data using the forward model
        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data.clone()],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
        };
        let y_obs = model.evaluate(&[true_density]);
        let sigma = vec![0.01; y_obs.len()];

        let config = FitConfig {
            energies,
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
            precomputed_cross_sections: None,
        };

        let result = fit_spectrum(&y_obs, &sigma, &config);

        assert!(result.converged);
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.01,
            "Fitted density = {}, true = {}",
            result.densities[0],
            true_density,
        );
        assert!(result.uncertainties[0] > 0.0);
    }
}
