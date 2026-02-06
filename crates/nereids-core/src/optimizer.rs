//! Optimizer trait and fit result types.

use crate::error::FitError;
use crate::nuclear::RMatrixParameters;
use crate::transmission::TransmissionSpectrum;

/// Configuration for a fitting run.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on chi-squared change.
    pub tolerance: f64,
    /// Prior parameter values for Bayesian fitting (length `n_params`).
    pub prior_values: Option<Vec<f64>>,
    /// Prior covariance matrix for Bayesian fitting
    /// (flattened row-major, `n_params` x `n_params`).
    ///
    /// This is the M matrix in SAMMY's (M+W) inversion. For independent
    /// priors, only the diagonal is non-zero, but the full matrix is stored
    /// to support correlated priors.
    pub prior_covariance: Option<Vec<f64>>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            tolerance: 1e-4,
            prior_values: None,
            prior_covariance: None,
        }
    }
}

/// Result of a fitting operation.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Fitted parameter values.
    pub parameters: Vec<f64>,
    /// Posterior covariance matrix (flattened, row-major, `n_params` x `n_params`).
    pub covariance: Vec<f64>,
    /// Final chi-squared value.
    pub chi_squared: f64,
    /// Number of degrees of freedom.
    pub degrees_of_freedom: usize,
    /// Whether the fit converged.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Trait for fitting algorithms.
///
/// The default implementation is Bayes/GLS (generalized least squares
/// with Bayesian priors), matching SAMMY's (M+W) inversion scheme.
/// The trait is designed to be pluggable for alternative optimizers
/// (BFGS, proximal gradient, etc.).
pub trait Optimizer: Send + Sync {
    /// Fit model parameters to observed data.
    fn fit(
        &self,
        observed: &TransmissionSpectrum,
        params: &RMatrixParameters,
        config: &FitConfig,
    ) -> Result<FitResult, FitError>;
}
