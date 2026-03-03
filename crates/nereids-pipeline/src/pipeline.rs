//! Single-spectrum analysis pipeline.
//!
//! Orchestrates the full analysis chain for a single transmission spectrum:
//! ENDF loading → cross-section calculation → broadening → fitting.
//!
//! This is the building block for the spatial mapping pipeline.

use std::fmt;
use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::lm::{self, FitModel, LmConfig, LmResult};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{self, PoissonConfig};
use nereids_fitting::transmission_model::{PrecomputedTransmissionModel, TransmissionFitModel};
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::InstrumentParams;

use crate::error::PipelineError;

/// Which optimizer to use for spectrum fitting.
#[derive(Debug, Clone, Default)]
pub enum SolverChoice {
    /// Levenberg-Marquardt chi-squared minimizer (default).
    #[default]
    LevenbergMarquardt,
    /// Poisson negative log-likelihood via L-BFGS with projected gradients
    /// and bound constraints. Appropriate for low-count data (< ~30 counts/bin).
    PoissonKL(PoissonConfig),
}

/// Errors from `FitConfig` construction.
#[derive(Debug, PartialEq)]
pub enum FitConfigError {
    /// Energy grid must be non-empty.
    EmptyEnergies,
    /// Resonance data must be non-empty.
    EmptyResonanceData,
    /// initial_densities length must match resonance_data length.
    DensityCountMismatch { densities: usize, isotopes: usize },
    /// isotope_names length must match resonance_data length.
    NameCountMismatch { names: usize, isotopes: usize },
    /// Temperature must be finite.
    NonFiniteTemperature(f64),
    /// Temperature must be non-negative.
    NegativeTemperature(f64),
    /// When fit_temperature is true, temperature must be >= 1.0 K.
    FitTemperatureTooLow(f64),
}

impl fmt::Display for FitConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEnergies => write!(f, "energy grid must be non-empty"),
            Self::EmptyResonanceData => write!(f, "resonance_data must be non-empty"),
            Self::DensityCountMismatch {
                densities,
                isotopes,
            } => write!(
                f,
                "initial_densities length ({densities}) must match resonance_data length ({isotopes})"
            ),
            Self::NameCountMismatch { names, isotopes } => write!(
                f,
                "isotope_names length ({names}) must match resonance_data length ({isotopes})"
            ),
            Self::NonFiniteTemperature(v) => {
                write!(f, "temperature must be finite, got {v}")
            }
            Self::NegativeTemperature(v) => {
                write!(f, "temperature must be non-negative, got {v}")
            }
            Self::FitTemperatureTooLow(v) => {
                write!(
                    f,
                    "temperature must be >= 1.0 K when fit_temperature is true, got {v}"
                )
            }
        }
    }
}

impl std::error::Error for FitConfigError {}

/// Configuration for a single-spectrum fit.
///
/// Fields are private to enforce validation invariants.
/// Use [`FitConfig::new`] to construct, then builder methods for optional fields.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Energy grid in eV (ascending).
    energies: Vec<f64>,
    /// Resonance data for each isotope to fit.
    resonance_data: Vec<ResonanceData>,
    /// Isotope names (for reporting).
    isotope_names: Vec<String>,
    /// Sample temperature in Kelvin.
    temperature_k: f64,
    /// Optional instrument resolution function (Gaussian or tabulated).
    resolution: Option<ResolutionFunction>,
    /// Initial guess for areal densities (atoms/barn), one per isotope.
    initial_densities: Vec<f64>,
    /// LM optimizer configuration.
    lm_config: LmConfig,
    /// Precomputed Doppler+resolution-broadened cross-sections, one `Vec<f64>`
    /// per isotope.  When `Some`, `fit_spectrum` skips the expensive resonance
    /// and broadening computation and uses `PrecomputedTransmissionModel` instead.
    precomputed_cross_sections: Option<Arc<Vec<Vec<f64>>>>,
    /// When `true`, `temperature_k` is treated as an initial guess and fitted
    /// jointly with the areal densities.
    fit_temperature: bool,
    /// Whether to compute the covariance matrix (and parameter uncertainties)
    /// after convergence.
    compute_covariance: bool,
    /// Which optimizer to use. Default: LevenbergMarquardt.
    solver: SolverChoice,
}

impl FitConfig {
    /// Create a validated fit configuration.
    ///
    /// # Arguments
    /// * `energies` — Energy grid in eV (must be non-empty).
    /// * `resonance_data` — Resonance data per isotope (must be non-empty).
    /// * `isotope_names` — Display names for each isotope.
    /// * `temperature_k` — Sample temperature in Kelvin (must be finite and non-negative).
    /// * `resolution` — Optional instrument resolution function.
    /// * `initial_densities` — Initial density guesses (length must match `resonance_data`).
    /// * `lm_config` — LM optimizer configuration.
    ///
    /// # Errors
    /// Returns `FitConfigError` if any invariant is violated.
    pub fn new(
        energies: Vec<f64>,
        resonance_data: Vec<ResonanceData>,
        isotope_names: Vec<String>,
        temperature_k: f64,
        resolution: Option<ResolutionFunction>,
        initial_densities: Vec<f64>,
        lm_config: LmConfig,
    ) -> Result<Self, FitConfigError> {
        if energies.is_empty() {
            return Err(FitConfigError::EmptyEnergies);
        }
        if resonance_data.is_empty() {
            return Err(FitConfigError::EmptyResonanceData);
        }
        if initial_densities.len() != resonance_data.len() {
            return Err(FitConfigError::DensityCountMismatch {
                densities: initial_densities.len(),
                isotopes: resonance_data.len(),
            });
        }
        if isotope_names.len() != resonance_data.len() {
            return Err(FitConfigError::NameCountMismatch {
                names: isotope_names.len(),
                isotopes: resonance_data.len(),
            });
        }
        if !temperature_k.is_finite() {
            return Err(FitConfigError::NonFiniteTemperature(temperature_k));
        }
        if temperature_k < 0.0 {
            return Err(FitConfigError::NegativeTemperature(temperature_k));
        }
        Ok(Self {
            energies,
            resonance_data,
            isotope_names,
            temperature_k,
            resolution,
            initial_densities,
            lm_config,
            precomputed_cross_sections: None,
            fit_temperature: false,
            compute_covariance: true,
            solver: SolverChoice::default(),
        })
    }

    /// Set precomputed cross-sections (builder pattern).
    #[must_use]
    pub fn with_precomputed_cross_sections(mut self, xs: Arc<Vec<Vec<f64>>>) -> Self {
        self.precomputed_cross_sections = Some(xs);
        self
    }

    /// Set whether to compute the covariance matrix (builder pattern).
    #[must_use]
    pub fn with_compute_covariance(mut self, compute: bool) -> Self {
        self.compute_covariance = compute;
        self
    }

    /// Set whether to fit temperature jointly (builder pattern).
    ///
    /// # Errors
    /// Returns `FitConfigError::FitTemperatureTooLow` if `fit` is true and
    /// `temperature_k < 1.0`.
    pub fn with_fit_temperature(mut self, fit: bool) -> Result<Self, FitConfigError> {
        if fit && self.temperature_k < 1.0 {
            return Err(FitConfigError::FitTemperatureTooLow(self.temperature_k));
        }
        self.fit_temperature = fit;
        Ok(self)
    }

    /// Returns the energy grid in eV.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Returns the resonance data for each isotope.
    #[must_use]
    pub fn resonance_data(&self) -> &[ResonanceData] {
        &self.resonance_data
    }

    /// Returns the isotope names.
    #[must_use]
    pub fn isotope_names(&self) -> &[String] {
        &self.isotope_names
    }

    /// Returns the sample temperature in Kelvin.
    #[must_use]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Returns the optional resolution function.
    #[must_use]
    pub fn resolution(&self) -> Option<&ResolutionFunction> {
        self.resolution.as_ref()
    }

    /// Returns the initial density guesses.
    #[must_use]
    pub fn initial_densities(&self) -> &[f64] {
        &self.initial_densities
    }

    /// Returns the LM optimizer configuration.
    #[must_use]
    pub fn lm_config(&self) -> &LmConfig {
        &self.lm_config
    }

    /// Returns the precomputed cross-sections, if any.
    #[must_use]
    pub fn precomputed_cross_sections(&self) -> Option<&Arc<Vec<Vec<f64>>>> {
        self.precomputed_cross_sections.as_ref()
    }

    /// Returns whether temperature fitting is enabled.
    #[must_use]
    pub fn fit_temperature(&self) -> bool {
        self.fit_temperature
    }

    /// Returns whether covariance computation is enabled.
    #[must_use]
    pub fn compute_covariance(&self) -> bool {
        self.compute_covariance
    }

    /// Set which optimizer to use (builder pattern).
    #[must_use]
    pub fn with_solver(mut self, solver: SolverChoice) -> Self {
        self.solver = solver;
        self
    }

    /// Returns the solver choice.
    #[must_use]
    pub fn solver(&self) -> &SolverChoice {
        &self.solver
    }
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

/// Dispatch between LM and Poisson solvers.
///
/// Both solvers use the same `FitModel` interface. The Poisson result is
/// mapped to `LmResult` so the caller can treat all solvers uniformly.
fn dispatch_solver(
    model: &dyn FitModel,
    measured_t: &[f64],
    sigma: &[f64],
    params: &mut ParameterSet,
    solver: &SolverChoice,
    lm_config: &LmConfig,
) -> Result<LmResult, PipelineError> {
    match solver {
        SolverChoice::LevenbergMarquardt => Ok(lm::levenberg_marquardt(
            model, measured_t, sigma, params, lm_config,
        )?),
        SolverChoice::PoissonKL(poisson_config) => {
            let pr = poisson::poisson_fit(model, measured_t, params, poisson_config);
            let n_free = params.n_free();
            let dof = measured_t.len().saturating_sub(n_free).max(1);
            // Compute Pearson chi-squared for display consistency with LM solver.
            let y_model = model.evaluate(&pr.params);
            let chi_sq: f64 = measured_t
                .iter()
                .zip(y_model.iter())
                .zip(sigma.iter())
                .map(|((obs, mdl), s)| {
                    let residual = obs - mdl;
                    (residual * residual) / (s * s).max(1e-30)
                })
                .sum();
            Ok(LmResult {
                chi_squared: chi_sq,
                reduced_chi_squared: chi_sq / dof as f64,
                iterations: pr.iterations,
                converged: pr.converged,
                params: pr.params,
                covariance: None,
                uncertainties: None,
            })
        }
    }
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
    let n_isotopes = config.resonance_data().len();

    // Config-level invariants (non-empty energies, resonance_data, density
    // count, temperature) are enforced by FitConfig::new().  Only per-call
    // shape checks remain here.
    if measured_t.len() != config.energies().len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "measured_t length ({}) must match energies length ({})",
            measured_t.len(),
            config.energies().len(),
        )));
    }
    if sigma.len() != measured_t.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "sigma length ({}) must match measured_t length ({})",
            sigma.len(),
            measured_t.len(),
        )));
    }

    let mut param_vec: Vec<FitParameter> = config
        .initial_densities()
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            FitParameter::non_negative(
                config
                    .isotope_names()
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
    let temperature_index = if config.fit_temperature() {
        param_vec.push(FitParameter {
            name: "temperature_k".into(),
            value: config.temperature_k(),
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
    let mut lm_config = config.lm_config().clone();
    lm_config.compute_covariance = config.compute_covariance();

    // Use precomputed cross-sections when available (fast path for spatial_map).
    // Fall back to the full forward-model path for single-spectrum calls.
    // When fitting temperature, always use the full forward model (can't
    // precompute when T is free).
    let result = if !config.fit_temperature() {
        if let Some(xs) = config.precomputed_cross_sections() {
            if xs.len() != n_isotopes {
                return Err(PipelineError::ShapeMismatch(format!(
                    "precomputed_cross_sections has {} isotope(s) but resonance_data has {}",
                    xs.len(),
                    n_isotopes,
                )));
            }
            let n_e = config.energies().len();
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
            dispatch_solver(
                &model,
                measured_t,
                sigma,
                &mut params,
                config.solver(),
                &lm_config,
            )?
        } else {
            let instrument = config
                .resolution()
                .cloned()
                .map(|r| Arc::new(InstrumentParams { resolution: r }));
            let model = TransmissionFitModel::new(
                config.energies().to_vec(),
                config.resonance_data().to_vec(),
                config.temperature_k(),
                instrument,
                (0..n_isotopes).collect(),
                None,
            )?;
            dispatch_solver(
                &model,
                measured_t,
                sigma,
                &mut params,
                config.solver(),
                &lm_config,
            )?
        }
    } else {
        // Temperature fitting: always use the full TransmissionFitModel
        // with temperature_index pointing to the appended temperature param.
        let instrument = config
            .resolution()
            .cloned()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        let model = TransmissionFitModel::new(
            config.energies().to_vec(),
            config.resonance_data().to_vec(),
            config.temperature_k(),
            instrument,
            (0..n_isotopes).collect(),
            temperature_index,
        )?;
        dispatch_solver(
            &model,
            measured_t,
            sigma,
            &mut params,
            config.solver(),
            &lm_config,
        )?
    };

    let densities: Vec<f64> = (0..n_isotopes).map(|i| result.params[i]).collect();

    // When covariance was computed, extract per-isotope and temperature
    // uncertainties from the LM result.  When skipped (None), propagate None.
    let (uncertainties, temperature_k, temperature_k_unc) = match result.uncertainties {
        Some(unc_all) => {
            let (temp_k, temp_unc) = if config.fit_temperature() {
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
            let (temp_k, temp_unc) = if config.fit_temperature() {
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
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            vec![0],
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density]);
        let sigma = vec![0.01; y_obs.len()];

        let config = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

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

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            vec![0],
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density]);
        let sigma = vec![0.01; y_obs.len()];

        let config = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_compute_covariance(false);

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
        let config = FitConfig::new(
            vec![1.0, 2.0, 3.0],
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        // measured_t length (2) != energies length (3)
        let result = fit_spectrum(&[0.9, 0.8], &[0.01, 0.01], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_spectrum_rejects_sigma_length_mismatch() {
        let data = u238_single_resonance();
        let config = FitConfig::new(
            vec![1.0, 2.0, 3.0],
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

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

    // --- FitConfig validation tests ---

    #[test]
    fn test_fit_config_valid() {
        let data = u238_single_resonance();
        let config = FitConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        );
        assert!(config.is_ok());
    }

    #[test]
    fn test_fit_config_rejects_empty_energies() {
        let data = u238_single_resonance();
        let err = FitConfig::new(
            vec![],
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, FitConfigError::EmptyEnergies);
    }

    #[test]
    fn test_fit_config_rejects_empty_resonance_data() {
        let err = FitConfig::new(
            vec![1.0, 2.0],
            vec![],
            vec![],
            300.0,
            None,
            vec![],
            LmConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, FitConfigError::EmptyResonanceData);
    }

    #[test]
    fn test_fit_config_rejects_density_count_mismatch() {
        let data = u238_single_resonance();
        let err = FitConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001, 0.002], // 2 densities but only 1 isotope
            LmConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, FitConfigError::DensityCountMismatch { .. }));
    }

    #[test]
    fn test_fit_config_rejects_name_count_mismatch() {
        let data = u238_single_resonance();
        let err = FitConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into(), "extra".into()], // 2 names but only 1 isotope
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, FitConfigError::NameCountMismatch { .. }));
    }

    #[test]
    fn test_fit_config_rejects_nan_temperature() {
        let data = u238_single_resonance();
        let err = FitConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            f64::NAN,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, FitConfigError::NonFiniteTemperature(_)));
    }

    #[test]
    fn test_fit_config_rejects_negative_temperature() {
        let data = u238_single_resonance();
        let err = FitConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            -1.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, FitConfigError::NegativeTemperature(-1.0));
    }

    #[test]
    fn test_fit_config_fit_temperature_rejects_low_temp() {
        let data = u238_single_resonance();
        let config = FitConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            0.5,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();
        let err = config.with_fit_temperature(true).unwrap_err();
        assert!(matches!(err, FitConfigError::FitTemperatureTooLow(_)));
    }

    #[test]
    fn test_fit_config_builder_methods() {
        let data = u238_single_resonance();
        let config = FitConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_compute_covariance(false);
        assert!(!config.compute_covariance());

        let config = config.with_fit_temperature(true).unwrap();
        assert!(config.fit_temperature());
    }

    #[test]
    fn test_fit_spectrum_poisson_kl() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        // Generate synthetic data using the forward model
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            vec![0],
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density]);
        let sigma = vec![0.01; y_obs.len()];

        let config = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_solver(SolverChoice::PoissonKL(PoissonConfig::default()));

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged);
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.05,
            "Poisson KL fitted density = {}, true = {}",
            result.densities[0],
            true_density,
        );
        // Poisson FD does not compute covariance
        assert!(result.uncertainties.is_none());
    }
}
