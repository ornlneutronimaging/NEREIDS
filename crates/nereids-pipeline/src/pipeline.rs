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
use nereids_fitting::poisson::{self, PoissonConfig, TemperatureContext};
use nereids_fitting::transmission_model::{PrecomputedTransmissionModel, TransmissionFitModel};
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::InstrumentParams;
use nereids_physics::transmission::{self as phys_transmission};

use crate::error::PipelineError;

/// How a fit parameter participates in the optimization.
///
/// Used in [`ParameterConstraints`] to control which isotope densities and
/// temperature are free (optimized) vs fixed (held constant).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ParameterRole {
    /// Optimized during fitting (default).
    /// Uses `initial_densities[i]` as the starting point.
    #[default]
    Free,
    /// Held at the specified value, excluded from optimization.
    /// The value is used directly in the forward model.
    Fixed(f64),
}

/// Per-parameter fitting constraints for a spectrum fit.
///
/// Controls which isotope densities and the sample temperature are
/// free (optimized) vs fixed (held constant) during fitting.
///
/// # Example workflows
///
/// - **Fix known matrix, fit traces**: set bulk isotope densities to
///   `Fixed(known_value)`, leave trace isotopes as `Free`.
/// - **Fix temperature, fit densities**: set `temperature` to
///   `Fixed(293.6)`, all densities `Free`.
/// - **Two-stage**: first pass with `temperature: Fixed(T)` and all
///   densities `Free`, then second pass with densities `Fixed` at
///   fitted values and `temperature: Free`.
#[derive(Debug, Clone)]
pub struct ParameterConstraints {
    /// Per-isotope constraint. Length must match `FitConfig::resonance_data`.
    /// `Free` isotopes use `initial_densities[i]` as the starting guess.
    /// `Fixed(val)` isotopes are held at `val` regardless of `initial_densities`.
    pub densities: Vec<ParameterRole>,
    /// Temperature constraint. Only meaningful when `fit_temperature` is true.
    /// When `Fixed(T)`, the temperature is held constant at `T` even if
    /// `fit_temperature` was enabled (the parameter is present but fixed).
    /// When `Free`, temperature is optimized (current default behavior).
    pub temperature: ParameterRole,
}

impl ParameterConstraints {
    /// Create constraints where all densities are free and temperature is free.
    pub fn all_free(n_isotopes: usize) -> Self {
        Self {
            densities: vec![ParameterRole::Free; n_isotopes],
            temperature: ParameterRole::Free,
        }
    }

    /// Returns true if all density constraints are `Free` and temperature is `Free`.
    pub fn is_all_free(&self) -> bool {
        self.densities
            .iter()
            .all(|r| matches!(r, ParameterRole::Free))
            && matches!(self.temperature, ParameterRole::Free)
    }
}

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
    /// constraints.densities length must match resonance_data length.
    ConstraintCountMismatch { constraints: usize, isotopes: usize },
    /// Fixed density value must be finite.
    NonFiniteFixedDensity { index: usize, value: f64 },
    /// Fixed temperature must be finite and non-negative.
    InvalidFixedTemperature(f64),
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
            Self::ConstraintCountMismatch {
                constraints,
                isotopes,
            } => write!(
                f,
                "constraints.densities length ({constraints}) must match resonance_data length ({isotopes})"
            ),
            Self::NonFiniteFixedDensity { index, value } => write!(
                f,
                "fixed density at index {index} must be finite, got {value}"
            ),
            Self::InvalidFixedTemperature(v) => write!(
                f,
                "fixed temperature must be finite and non-negative, got {v}"
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
    /// Precomputed unbroadened (Reich-Moore) cross-sections, one `Vec<f64>` per
    /// isotope.  When `Some` and temperature fitting is enabled, `fit_spectrum`
    /// skips the per-pixel Reich-Moore evaluation and uses `_from_base` variants
    /// for both the `TransmissionFitModel` and `TemperatureContext`.
    precomputed_base_xs: Option<Arc<Vec<Vec<f64>>>>,
    /// When `true`, `temperature_k` is treated as an initial guess and fitted
    /// jointly with the areal densities.
    fit_temperature: bool,
    /// Whether to compute the covariance matrix (and parameter uncertainties)
    /// after convergence.
    compute_covariance: bool,
    /// Which optimizer to use. Default: LevenbergMarquardt.
    solver: SolverChoice,
    /// Per-parameter constraints (fixed/free). When `None`, all parameters
    /// are free (current default behavior).
    constraints: Option<ParameterConstraints>,
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
            precomputed_base_xs: None,
            fit_temperature: false,
            compute_covariance: true,
            solver: SolverChoice::default(),
            constraints: None,
        })
    }

    /// Set precomputed cross-sections (builder pattern).
    #[must_use]
    pub fn with_precomputed_cross_sections(mut self, xs: Arc<Vec<Vec<f64>>>) -> Self {
        self.precomputed_cross_sections = Some(xs);
        self
    }

    /// Set precomputed unbroadened (base) cross-sections (builder pattern).
    ///
    /// When set, temperature fitting skips the per-pixel Reich-Moore evaluation.
    #[must_use]
    pub fn with_precomputed_base_xs(mut self, xs: Arc<Vec<Vec<f64>>>) -> Self {
        self.precomputed_base_xs = Some(xs);
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

    /// Returns the precomputed unbroadened (base) cross-sections, if any.
    #[must_use]
    pub fn precomputed_base_xs(&self) -> Option<&Arc<Vec<Vec<f64>>>> {
        self.precomputed_base_xs.as_ref()
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

    /// Set per-parameter constraints (builder pattern).
    ///
    /// # Errors
    /// Returns `FitConfigError::ConstraintCountMismatch` if `constraints.densities`
    /// length does not match `resonance_data` length, or validation errors for
    /// non-finite fixed values.
    pub fn with_constraints(
        mut self,
        constraints: ParameterConstraints,
    ) -> Result<Self, FitConfigError> {
        if constraints.densities.len() != self.resonance_data.len() {
            return Err(FitConfigError::ConstraintCountMismatch {
                constraints: constraints.densities.len(),
                isotopes: self.resonance_data.len(),
            });
        }
        for (i, role) in constraints.densities.iter().enumerate() {
            if let ParameterRole::Fixed(v) = role
                && !v.is_finite()
            {
                return Err(FitConfigError::NonFiniteFixedDensity {
                    index: i,
                    value: *v,
                });
            }
        }
        if let ParameterRole::Fixed(v) = constraints.temperature
            && (!v.is_finite() || v < 0.0)
        {
            return Err(FitConfigError::InvalidFixedTemperature(v));
        }
        self.constraints = Some(constraints);
        Ok(self)
    }

    /// Returns the parameter constraints, if any.
    #[must_use]
    pub fn constraints(&self) -> Option<&ParameterConstraints> {
        self.constraints.as_ref()
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
            let pr = poisson::poisson_fit(model, measured_t, params, poisson_config)?;
            let n_free = params.n_free();
            let dof = measured_t.len().saturating_sub(n_free).max(1);
            // Compute Pearson chi-squared for display consistency with LM solver.
            let y_model = model.evaluate(&pr.params)?;
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

    let constraints = config.constraints();
    let mut param_vec: Vec<FitParameter> = config
        .initial_densities()
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let name = config
                .isotope_names()
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("isotope_{}", i));
            match constraints.and_then(|c| c.densities.get(i)) {
                Some(ParameterRole::Fixed(val)) => FitParameter::fixed(name, *val),
                _ => FitParameter::non_negative(name, d),
            }
        })
        .collect();

    // When fitting temperature, append it as an additional parameter
    // after the density parameters.  Temperature uses bounded constraints
    // (physical range 1–5000 K).  When constraints specify Fixed(T),
    // the temperature parameter is present but marked fixed.
    let temperature_index = if config.fit_temperature() {
        let is_temp_fixed =
            constraints.is_some_and(|c| matches!(c.temperature, ParameterRole::Fixed(_)));
        let temp_value = match constraints {
            Some(ParameterConstraints {
                temperature: ParameterRole::Fixed(v),
                ..
            }) => *v,
            _ => config.temperature_k(),
        };
        if is_temp_fixed {
            param_vec.push(FitParameter {
                name: "temperature_k".into(),
                value: temp_value,
                lower: 1.0,
                upper: 5000.0,
                fixed: true,
            });
        } else {
            param_vec.push(FitParameter {
                name: "temperature_k".into(),
                value: temp_value,
                lower: 1.0,
                upper: 5000.0,
                fixed: false,
            });
        }
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

        // When spatial_map precomputes base_xs once for all pixels, inject
        // them into the model to skip per-pixel Reich-Moore evaluation.
        let base_xs_for_model = config.precomputed_base_xs().cloned();
        let model = TransmissionFitModel::new(
            config.energies().to_vec(),
            config.resonance_data().to_vec(),
            config.temperature_k(),
            instrument.clone(),
            (0..n_isotopes).collect(),
            temperature_index,
            base_xs_for_model,
        )?;

        match config.solver() {
            SolverChoice::LevenbergMarquardt => dispatch_solver(
                &model,
                measured_t,
                sigma,
                &mut params,
                config.solver(),
                &lm_config,
            )?,
            SolverChoice::PoissonKL(poisson_config) => {
                // The basic poisson_fit uses finite-difference gradient descent,
                // which cannot effectively optimize temperature: the gradient is
                // dominated by density parameters (scale ~1e-4) so the temperature
                // (scale ~200) never moves.
                //
                // Use poisson_fit_analytic with Fisher preconditioning, which
                // handles the scale mismatch via diagonal Hessian normalization.
                //
                // In transmission space, flux = 1.0 everywhere (since Y = T,
                // not Y = Φ·T + B).  The analytical gradient formulas reduce
                // correctly with Φ=1.
                let density_indices: Vec<usize> = (0..n_isotopes).collect();
                let instrument_plain = instrument.as_deref().cloned();

                // Use precomputed base_xs when available (spatial_map path);
                // fall back to computing them here (single-pixel path).
                let base_xs: Arc<Vec<Vec<f64>>> = match config.precomputed_base_xs() {
                    Some(cached) => Arc::clone(cached),
                    None => Arc::new(phys_transmission::unbroadened_cross_sections(
                        config.energies(),
                        config.resonance_data(),
                        None,
                    )?),
                };

                // Compute initial broadened XS from base (Doppler + resolution
                // only, no redundant Reich-Moore).
                let xs = phys_transmission::broadened_cross_sections_from_base(
                    config.energies(),
                    &base_xs,
                    config.resonance_data(),
                    config.temperature_k(),
                    instrument.as_deref(),
                )?;

                let temp_ctx = TemperatureContext {
                    temperature_index: n_isotopes,
                    resonance_data: config.resonance_data().to_vec(),
                    energies: config.energies().to_vec(),
                    instrument: instrument_plain,
                    base_xs: Some(base_xs),
                };

                let flux = vec![1.0f64; config.energies().len()];

                let pr = poisson::poisson_fit_analytic(
                    &model,
                    measured_t,
                    &flux,
                    &xs,
                    &density_indices,
                    &mut params,
                    poisson_config,
                    Some(&temp_ctx),
                )?;

                let n_free = params.n_free();
                let dof = measured_t.len().saturating_sub(n_free).max(1);
                let y_model = model.evaluate(&pr.params)?;
                let chi_sq: f64 = measured_t
                    .iter()
                    .zip(y_model.iter())
                    .zip(sigma.iter())
                    .map(|((obs, mdl), s)| {
                        let residual = obs - mdl;
                        (residual * residual) / (s * s).max(1e-30)
                    })
                    .sum();
                LmResult {
                    chi_squared: chi_sq,
                    reduced_chi_squared: chi_sq / dof as f64,
                    iterations: pr.iterations,
                    converged: pr.converged,
                    params: pr.params,
                    covariance: None,
                    uncertainties: None,
                }
            }
        }
    };

    let densities: Vec<f64> = (0..n_isotopes).map(|i| result.params[i]).collect();

    // When covariance was computed, extract per-isotope and temperature
    // uncertainties from the LM result.  When skipped (None), propagate None.
    //
    // The LM result stores uncertainties only for free parameters (n_free
    // elements).  When parameter constraints fix some densities, we expand
    // the compact free-param uncertainties back to all-param positions,
    // inserting 0.0 for fixed parameters.
    let n_all_params = params.params.len();
    let (uncertainties, temperature_k, temperature_k_unc) = match result.uncertainties {
        Some(unc_free) => {
            // Expand free-param uncertainties to all-param positions.
            let free_indices = params.free_indices();
            let mut unc_all = vec![0.0f64; n_all_params];
            for (j, &all_idx) in free_indices.iter().enumerate() {
                if let Some(&u) = unc_free.get(j) {
                    unc_all[all_idx] = u;
                }
            }

            let (temp_k, temp_unc) = if config.fit_temperature() {
                (
                    Some(result.params[n_isotopes]),
                    Some(*unc_all.get(n_isotopes).unwrap_or(&f64::NAN)),
                )
            } else {
                (None, None)
            };

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
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density]).unwrap();
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
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density]).unwrap();
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
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density]).unwrap();
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

    // ── Temperature fitting round-trip tests ────────────────────────────────

    /// Generate synthetic transmission for the given isotopes at a known
    /// temperature and density, then fit through `fit_spectrum` and verify
    /// recovery.
    fn temperature_round_trip(
        resonance_data: Vec<ResonanceData>,
        names: Vec<String>,
        true_densities: &[f64],
        true_temp: f64,
        solver: SolverChoice,
        tolerance: f64,
        label: &str,
    ) {
        use nereids_physics::transmission::{SampleParams, forward_model};

        // Energy grid spanning both the W-182 (4.15 eV) and U-238 (6.67 eV)
        // resonances with enough points for the optimizer.
        let energies: Vec<f64> = (0..501).map(|i| 2.0 + (i as f64) * 0.02).collect();

        // Build isotope list for the forward model at truth.
        let isotopes: Vec<(ResonanceData, f64)> = resonance_data
            .iter()
            .zip(true_densities.iter())
            .map(|(rd, &d)| (rd.clone(), d))
            .collect();
        let sample = SampleParams::new(true_temp, isotopes).unwrap();
        let y_obs = forward_model(&energies, &sample, None).unwrap();

        // Use uniform small sigma for the LM path.
        let sigma = vec![0.005; y_obs.len()];

        // Initial guesses: densities at 2× truth, temperature 100 K off.
        let initial_densities: Vec<f64> = true_densities.iter().map(|d| d * 2.0).collect();
        let config = FitConfig::new(
            energies,
            resonance_data,
            names,
            true_temp - 100.0, // initial T guess offset from truth
            None,
            initial_densities,
            LmConfig {
                max_iter: 300,
                ..LmConfig::default()
            },
        )
        .unwrap()
        .with_fit_temperature(true)
        .unwrap()
        .with_solver(solver);

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(
            result.converged,
            "[{label}] did not converge after {} iterations",
            result.iterations,
        );

        // Check each density.
        for (i, (&fitted, &truth)) in result
            .densities
            .iter()
            .zip(true_densities.iter())
            .enumerate()
        {
            let rel_err = (fitted - truth).abs() / truth;
            assert!(
                rel_err < tolerance,
                "[{label}] density[{i}]: fitted={fitted:.6e}, true={truth:.6e}, error={:.2}%",
                rel_err * 100.0,
            );
        }

        // Check temperature.
        let fitted_temp = result
            .temperature_k
            .expect("temperature_k should be Some when fit_temperature=true");
        let temp_rel_err = (fitted_temp - true_temp).abs() / true_temp;
        assert!(
            temp_rel_err < tolerance,
            "[{label}] temperature: fitted={fitted_temp:.1} K, true={true_temp:.1} K, error={:.2}%",
            temp_rel_err * 100.0,
        );
    }

    #[test]
    fn test_fit_spectrum_single_isotope_temperature_lm() {
        let u238 = u238_single_resonance();
        temperature_round_trip(
            vec![u238],
            vec!["U-238".into()],
            &[0.0005],
            300.0,
            SolverChoice::LevenbergMarquardt,
            0.01, // 1% tolerance (noise-free data)
            "1-isotope LM",
        );
    }

    #[test]
    fn test_fit_spectrum_multi_isotope_temperature_lm() {
        use crate::test_helpers::w182_single_resonance;
        let u238 = u238_single_resonance();
        let w182 = w182_single_resonance();
        temperature_round_trip(
            vec![u238, w182],
            vec!["U-238".into(), "W-182".into()],
            &[0.0005, 0.0003],
            300.0,
            SolverChoice::LevenbergMarquardt,
            0.01, // 1%
            "2-isotope LM",
        );
    }

    #[test]
    fn test_fit_spectrum_single_isotope_temperature_poisson() {
        let u238 = u238_single_resonance();
        temperature_round_trip(
            vec![u238],
            vec!["U-238".into()],
            &[0.0005],
            300.0,
            SolverChoice::PoissonKL(PoissonConfig {
                max_iter: 500,
                ..PoissonConfig::default()
            }),
            0.02, // 2% tolerance (Poisson solver slightly less precise)
            "1-isotope Poisson",
        );
    }

    #[test]
    fn test_fit_spectrum_multi_isotope_temperature_poisson() {
        use crate::test_helpers::w182_single_resonance;
        let u238 = u238_single_resonance();
        let w182 = w182_single_resonance();
        temperature_round_trip(
            vec![u238, w182],
            vec!["U-238".into(), "W-182".into()],
            &[0.0005, 0.0003],
            300.0,
            SolverChoice::PoissonKL(PoissonConfig {
                max_iter: 500,
                ..PoissonConfig::default()
            }),
            0.02, // 2%
            "2-isotope Poisson",
        );
    }

    // ── Parameter constraint tests ───────────────────────────────────────────

    #[test]
    fn test_constraints_all_free_unchanged() {
        // All-free constraints should produce identical results to no constraints.
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
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density]).unwrap();
        let sigma = vec![0.01; y_obs.len()];

        // Without constraints
        let config_no_constraints = FitConfig::new(
            energies.clone(),
            vec![data.clone()],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();
        let result_no = fit_spectrum(&y_obs, &sigma, &config_no_constraints).unwrap();

        // With all-free constraints
        let config_all_free = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_constraints(ParameterConstraints::all_free(1))
        .unwrap();
        let result_free = fit_spectrum(&y_obs, &sigma, &config_all_free).unwrap();

        assert!(
            (result_no.densities[0] - result_free.densities[0]).abs() < 1e-12,
            "All-free constraints should give identical results: {} vs {}",
            result_no.densities[0],
            result_free.densities[0],
        );
    }

    #[test]
    fn test_constraints_fix_one_density() {
        // Fix U-238 at known value, fit the "other" isotope.
        use crate::test_helpers::w182_single_resonance;

        let u238 = u238_single_resonance();
        let w182 = w182_single_resonance();
        let true_u = 0.0005;
        let true_w = 0.0003;

        let energies: Vec<f64> = (0..401).map(|i| 2.0 + (i as f64) * 0.025).collect();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![u238.clone(), w182.clone()],
            0.0,
            None,
            vec![0, 1],
            None,
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_u, true_w]).unwrap();
        let sigma = vec![0.01; y_obs.len()];

        // Fix U-238, fit W-182
        let constraints = ParameterConstraints {
            densities: vec![ParameterRole::Fixed(true_u), ParameterRole::Free],
            temperature: ParameterRole::Free,
        };
        let config = FitConfig::new(
            energies,
            vec![u238, w182],
            vec!["U-238".into(), "W-182".into()],
            0.0,
            None,
            vec![0.001, 0.001], // initial guesses (U-238 guess ignored)
            LmConfig::default(),
        )
        .unwrap()
        .with_constraints(constraints)
        .unwrap();

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged, "Fit should converge");

        // U-238 density must be exactly the fixed value
        assert!(
            (result.densities[0] - true_u).abs() < 1e-15,
            "Fixed density should be exact: got {}, expected {}",
            result.densities[0],
            true_u,
        );

        // W-182 density should be recovered within tolerance
        assert!(
            (result.densities[1] - true_w).abs() / true_w < 0.02,
            "Free density recovery: got {}, expected {}, error = {:.1}%",
            result.densities[1],
            true_w,
            (result.densities[1] - true_w).abs() / true_w * 100.0,
        );

        // U-238 uncertainty should be 0 (fixed param)
        if let Some(ref unc) = result.uncertainties {
            assert!(
                unc[0].abs() < 1e-15,
                "Fixed param uncertainty should be 0, got {}",
                unc[0],
            );
            // W-182 uncertainty should be positive
            assert!(
                unc[1] > 0.0,
                "Free param uncertainty should be positive, got {}",
                unc[1],
            );
        }
    }

    #[test]
    fn test_constraints_fix_temperature() {
        // Fix temperature, fit densities only.
        let u238 = u238_single_resonance();
        let true_density = 0.0005;
        let true_temp = 300.0;

        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.025).collect();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![u238.clone()],
            0.0,
            None,
            vec![0],
            Some(1),
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_density, true_temp]).unwrap();
        let sigma = vec![0.005; y_obs.len()];

        let constraints = ParameterConstraints {
            densities: vec![ParameterRole::Free],
            temperature: ParameterRole::Fixed(true_temp),
        };
        let config = FitConfig::new(
            energies,
            vec![u238],
            vec!["U-238".into()],
            true_temp,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_fit_temperature(true)
        .unwrap()
        .with_constraints(constraints)
        .unwrap();

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged, "Fit should converge");

        // Density should be recovered
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.01,
            "Density recovery: got {}, expected {}",
            result.densities[0],
            true_density,
        );

        // Temperature should be exactly the fixed value
        let fitted_temp = result.temperature_k.unwrap();
        assert!(
            (fitted_temp - true_temp).abs() < 1e-10,
            "Fixed temperature should be exact: got {}, expected {}",
            fitted_temp,
            true_temp,
        );
    }

    #[test]
    fn test_constraints_fix_all_but_one() {
        // Fix all isotopes except one — the single free param should converge.
        use crate::test_helpers::w182_single_resonance;

        let u238 = u238_single_resonance();
        let w182 = w182_single_resonance();
        let true_u = 0.0005;
        let true_w = 0.0003;

        let energies: Vec<f64> = (0..401).map(|i| 2.0 + (i as f64) * 0.025).collect();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![u238.clone(), w182.clone()],
            0.0,
            None,
            vec![0, 1],
            None,
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_u, true_w]).unwrap();
        let sigma = vec![0.01; y_obs.len()];

        // Fix W-182, fit only U-238
        let constraints = ParameterConstraints {
            densities: vec![ParameterRole::Free, ParameterRole::Fixed(true_w)],
            temperature: ParameterRole::Free,
        };
        let config = FitConfig::new(
            energies,
            vec![u238, w182],
            vec!["U-238".into(), "W-182".into()],
            0.0,
            None,
            vec![0.001, 0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_constraints(constraints)
        .unwrap();

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged, "Fit should converge");

        // U-238 recovered
        assert!(
            (result.densities[0] - true_u).abs() / true_u < 0.02,
            "U-238: got {}, expected {}, error = {:.1}%",
            result.densities[0],
            true_u,
            (result.densities[0] - true_u).abs() / true_u * 100.0,
        );

        // W-182 exact
        assert!(
            (result.densities[1] - true_w).abs() < 1e-15,
            "Fixed W-182: got {}, expected {}",
            result.densities[1],
            true_w,
        );
    }

    #[test]
    fn test_constraints_fixed_values_in_output() {
        // Verify that fixed density values appear at the correct indices in the
        // result, even when both fixed and free params are present.
        use crate::test_helpers::w182_single_resonance;

        let u238 = u238_single_resonance();
        let w182 = w182_single_resonance();
        let fixed_u = 0.00042;
        let fixed_w = 0.00031;

        let energies: Vec<f64> = (0..201).map(|i| 2.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![u238.clone(), w182.clone()],
            0.0,
            None,
            vec![0, 1],
            None,
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[fixed_u, fixed_w]).unwrap();
        let sigma = vec![0.01; y_obs.len()];

        // Fix BOTH densities
        let constraints = ParameterConstraints {
            densities: vec![ParameterRole::Fixed(fixed_u), ParameterRole::Fixed(fixed_w)],
            temperature: ParameterRole::Free,
        };
        let config = FitConfig::new(
            energies,
            vec![u238, w182],
            vec!["U-238".into(), "W-182".into()],
            0.0,
            None,
            vec![0.001, 0.001], // ignored for fixed
            LmConfig::default(),
        )
        .unwrap()
        .with_constraints(constraints)
        .unwrap();

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged, "All-fixed should 'converge'");
        assert_eq!(result.densities.len(), 2);
        assert!(
            (result.densities[0] - fixed_u).abs() < 1e-15,
            "index 0: got {}, expected {}",
            result.densities[0],
            fixed_u,
        );
        assert!(
            (result.densities[1] - fixed_w).abs() < 1e-15,
            "index 1: got {}, expected {}",
            result.densities[1],
            fixed_w,
        );
    }

    #[test]
    fn test_constraints_poisson_fix_one_density() {
        // Same as test_constraints_fix_one_density but with Poisson solver.
        use crate::test_helpers::w182_single_resonance;

        let u238 = u238_single_resonance();
        let w182 = w182_single_resonance();
        let true_u = 0.0005;
        let true_w = 0.0003;

        let energies: Vec<f64> = (0..401).map(|i| 2.0 + (i as f64) * 0.025).collect();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![u238.clone(), w182.clone()],
            0.0,
            None,
            vec![0, 1],
            None,
            None,
        )
        .unwrap();
        let y_obs = model.evaluate(&[true_u, true_w]).unwrap();
        let sigma = vec![0.01; y_obs.len()];

        // Fix U-238, fit W-182 with Poisson solver
        let constraints = ParameterConstraints {
            densities: vec![ParameterRole::Fixed(true_u), ParameterRole::Free],
            temperature: ParameterRole::Free,
        };
        let config = FitConfig::new(
            energies,
            vec![u238, w182],
            vec!["U-238".into(), "W-182".into()],
            0.0,
            None,
            vec![0.001, 0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_constraints(constraints)
        .unwrap()
        .with_solver(SolverChoice::PoissonKL(PoissonConfig::default()));

        let result = fit_spectrum(&y_obs, &sigma, &config).unwrap();

        assert!(result.converged, "Poisson fit should converge");

        // U-238 fixed
        assert!(
            (result.densities[0] - true_u).abs() < 1e-15,
            "Fixed U-238: got {}, expected {}",
            result.densities[0],
            true_u,
        );

        // W-182 recovered
        assert!(
            (result.densities[1] - true_w).abs() / true_w < 0.05,
            "Poisson W-182: got {}, expected {}, error = {:.1}%",
            result.densities[1],
            true_w,
            (result.densities[1] - true_w).abs() / true_w * 100.0,
        );
    }

    // ── Constraint validation tests ──────────────────────────────────────────

    #[test]
    fn test_constraints_count_mismatch() {
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
        .unwrap();

        // 2 constraints but only 1 isotope
        let err = config
            .with_constraints(ParameterConstraints {
                densities: vec![ParameterRole::Free, ParameterRole::Free],
                temperature: ParameterRole::Free,
            })
            .unwrap_err();
        assert!(matches!(
            err,
            FitConfigError::ConstraintCountMismatch { .. }
        ));
    }

    #[test]
    fn test_constraints_non_finite_fixed_density() {
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
        .unwrap();

        let err = config
            .with_constraints(ParameterConstraints {
                densities: vec![ParameterRole::Fixed(f64::NAN)],
                temperature: ParameterRole::Free,
            })
            .unwrap_err();
        assert!(matches!(err, FitConfigError::NonFiniteFixedDensity { .. }));
    }

    #[test]
    fn test_constraints_invalid_fixed_temperature() {
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
        .unwrap();

        let err = config
            .with_constraints(ParameterConstraints {
                densities: vec![ParameterRole::Free],
                temperature: ParameterRole::Fixed(-10.0),
            })
            .unwrap_err();
        assert!(matches!(err, FitConfigError::InvalidFixedTemperature(_)));
    }
}
