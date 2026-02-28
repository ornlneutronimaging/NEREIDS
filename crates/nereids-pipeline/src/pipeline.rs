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

use crate::error::PipelineError;

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
    /// When `true`, `temperature_k` is treated as an initial guess and fitted
    /// jointly with the areal densities.  The precomputed fast path is
    /// disabled (cross-sections must be recomputed per evaluation at the
    /// current trial temperature).
    ///
    /// Default: `false` — temperature is held fixed.
    pub fit_temperature: bool,
    /// Whether to compute the covariance matrix (and parameter uncertainties)
    /// after convergence.  When `false`, the post-convergence Jacobian
    /// recomputation and matrix inversion are skipped, and
    /// `SpectrumFitResult::uncertainties` will be `None`.
    ///
    /// Default: `true` (backwards-compatible for single-spectrum use).
    /// Set to `false` in `spatial_map` for speed when only densities are needed.
    pub compute_covariance: bool,
}

/// Result of fitting a single spectrum.
#[derive(Debug, Clone)]
pub struct SpectrumFitResult {
    /// Fitted areal densities (atoms/barn), one per isotope.
    pub densities: Vec<f64>,
    /// Uncertainty on each density.
    ///
    /// `None` when covariance computation was skipped
    /// (`FitConfig::compute_covariance == false`).
    pub uncertainties: Option<Vec<f64>>,
    /// Reduced chi-squared of the fit.
    pub reduced_chi_squared: f64,
    /// Whether the fit converged.
    pub converged: bool,
    /// Number of iterations.
    pub iterations: usize,
    /// Fitted temperature in Kelvin (only when `FitConfig::fit_temperature` is true).
    pub temperature_k: Option<f64>,
    /// 1-sigma uncertainty on the fitted temperature (from covariance matrix).
    pub temperature_k_unc: Option<f64>,
}

/// Fit a single measured transmission spectrum.
///
/// # Arguments
/// * `measured_t` — Measured transmission values at each energy point.
/// * `sigma` — Uncertainties on measured transmission.
/// * `config` — Fit configuration (isotopes, energy grid, etc.).
///
/// # Errors
/// Returns `PipelineError::ShapeMismatch` if array lengths are inconsistent,
/// or `PipelineError::InvalidParameter` if configuration values are invalid.
///
/// # Returns
/// Fit result with densities, uncertainties, and fit quality metrics.
pub fn fit_spectrum(
    measured_t: &[f64],
    sigma: &[f64],
    config: &FitConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let n_isotopes = config.resonance_data.len();

    if config.energies.is_empty() {
        return Err(PipelineError::InvalidParameter(
            "config.energies must be non-empty".into(),
        ));
    }
    if config.initial_densities.len() != n_isotopes {
        return Err(PipelineError::ShapeMismatch(format!(
            "initial_densities length ({}) must match resonance_data length ({})",
            config.initial_densities.len(),
            n_isotopes,
        )));
    }
    if measured_t.len() != config.energies.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "measured_t length ({}) must match energies length ({})",
            measured_t.len(),
            config.energies.len(),
        )));
    }
    if sigma.len() != measured_t.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "sigma length ({}) must match measured_t length ({})",
            sigma.len(),
            measured_t.len(),
        )));
    }
    if !config.temperature_k.is_finite() {
        return Err(PipelineError::InvalidParameter(format!(
            "temperature_k must be finite, got {}",
            config.temperature_k,
        )));
    }
    if config.fit_temperature && config.temperature_k < 1.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "temperature_k ({}) must be >= 1.0 K when fit_temperature is true",
            config.temperature_k,
        )));
    }

    let mut param_vec: Vec<FitParameter> = config
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
        .collect();

    // When fitting temperature, append it as an additional free parameter
    // after the density parameters.  Temperature uses bounded constraints
    // (physical range 1–5000 K).
    let temperature_index = if config.fit_temperature {
        param_vec.push(FitParameter {
            name: "temperature_k".into(),
            value: config.temperature_k,
            lower: 1.0,
            upper: 5000.0,
            fixed: false,
        });
        Some(n_isotopes)
    } else {
        None
    };

    let mut params = ParameterSet::new(param_vec);

    // Propagate the pipeline-level compute_covariance flag into the LM config.
    // This lets spatial_map disable covariance without
    // modifying the caller's LmConfig.
    let mut lm_config = config.lm_config.clone();
    lm_config.compute_covariance = config.compute_covariance;

    // Use precomputed cross-sections when available (fast path for spatial_map).
    // Fall back to the full forward-model path for single-spectrum calls.
    // When fitting temperature, always use the full forward model (can't
    // precompute when T is free).
    let result = if !config.fit_temperature {
        if let Some(xs) = &config.precomputed_cross_sections {
            if n_isotopes == 0 {
                return Err(PipelineError::InvalidParameter(
                    "resonance_data is empty — nothing to fit".into(),
                ));
            }
            if xs.len() != n_isotopes {
                return Err(PipelineError::ShapeMismatch(format!(
                    "precomputed_cross_sections has {} isotope(s) but resonance_data has {}",
                    xs.len(),
                    n_isotopes,
                )));
            }
            let n_e = config.energies.len();
            for (i, row) in xs.iter().enumerate() {
                if row.len() != n_e {
                    return Err(PipelineError::ShapeMismatch(format!(
                        "precomputed_cross_sections[{}] has {} energy points but energies has {}",
                        i,
                        row.len(),
                        n_e,
                    )));
                }
            }
            let model = PrecomputedTransmissionModel {
                cross_sections: xs.clone(),
                density_indices: Arc::new((0..n_isotopes).collect()),
            };
            lm::levenberg_marquardt(&model, measured_t, sigma, &mut params, &lm_config)
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
                temperature_index: None,
            };
            lm::levenberg_marquardt(&model, measured_t, sigma, &mut params, &lm_config)
        }
    } else {
        // Temperature fitting: always use the full TransmissionFitModel
        // with temperature_index pointing to the appended temperature param.
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
            temperature_index,
        };
        lm::levenberg_marquardt(&model, measured_t, sigma, &mut params, &lm_config)
    };

    let densities: Vec<f64> = (0..n_isotopes).map(|i| result.params[i]).collect();

    // When covariance was computed, extract per-isotope and temperature
    // uncertainties from the LM result.  When skipped (None), propagate None.
    let (uncertainties, temperature_k, temperature_k_unc) = match result.uncertainties {
        Some(unc_all) => {
            let (temp_k, temp_unc) = if config.fit_temperature {
                (
                    Some(result.params[n_isotopes]),
                    Some(*unc_all.get(n_isotopes).unwrap_or(&f64::NAN)),
                )
            } else {
                (None, None)
            };

            // Guard: the LM result should always produce at least n_isotopes
            // uncertainties, but use a safe fallback to NaN if the invariant
            // is ever violated rather than panicking on the slice.
            let unc = unc_all
                .get(..n_isotopes)
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![f64::NAN; n_isotopes]);

            (Some(unc), temp_k, temp_unc)
        }
        None => {
            // Covariance was skipped — no uncertainties available.
            let (temp_k, temp_unc) = if config.fit_temperature {
                (Some(result.params[n_isotopes]), None)
            } else {
                (None, None)
            };
            (None, temp_k, temp_unc)
        }
    };

    Ok(SpectrumFitResult {
        densities,
        uncertainties,
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
        temperature_k,
        temperature_k_unc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_fitting::lm::FitModel;

    use crate::test_helpers::u238_single_resonance;

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
            temperature_index: None,
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
            fit_temperature: false,
            compute_covariance: true,
        };

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged);
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.01,
            "Fitted density = {}, true = {}",
            result.densities[0],
            true_density,
        );
        let unc = result
            .uncertainties
            .expect("uncertainties should be Some when compute_covariance=true");
        assert!(unc[0] > 0.0);
    }

    #[test]
    fn test_fit_spectrum_no_covariance() {
        // When compute_covariance is false, fit_spectrum should still produce
        // correct densities but uncertainties should be None.
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data.clone()],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
            temperature_index: None,
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
            fit_temperature: false,
            compute_covariance: false,
        };

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged);
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.01,
            "Fitted density = {}, true = {}",
            result.densities[0],
            true_density,
        );
        assert!(
            result.uncertainties.is_none(),
            "uncertainties should be None when compute_covariance=false"
        );
    }

    #[test]
    fn test_fit_spectrum_rejects_length_mismatch() {
        let data = u238_single_resonance();
        let config = FitConfig {
            energies: vec![1.0, 2.0, 3.0],
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
            precomputed_cross_sections: None,
            fit_temperature: false,
            compute_covariance: true,
        };

        // measured_t length (2) != energies length (3)
        let result = fit_spectrum(&[0.9, 0.8], &[0.01, 0.01], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_spectrum_rejects_sigma_length_mismatch() {
        let data = u238_single_resonance();
        let config = FitConfig {
            energies: vec![1.0, 2.0, 3.0],
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
            precomputed_cross_sections: None,
            fit_temperature: false,
            compute_covariance: true,
        };

        // sigma length (2) != measured_t length (3) — should return ShapeMismatch
        let result = fit_spectrum(&[0.9, 0.8, 0.7], &[0.01, 0.01], &config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("sigma length"),
            "Expected sigma mismatch error, got: {}",
            err_msg,
        );
    }
}
