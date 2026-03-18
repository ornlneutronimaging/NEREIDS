//! Solver-agnostic forward model trait.
//!
//! Unlike [`FitModel`](crate::lm::FitModel), which is LM-specific (evaluates
//! residuals + Jacobian in chi-squared space), `ForwardModel` exposes the
//! raw model prediction and parameter Jacobian.  Each solver wraps this to
//! compute its own objective and gradient:
//!
//! - **LM**: residuals = `(data - predict) / σ`, J_lm = `-jacobian / σ`
//! - **KL**: gradient = `Σ (1 - data/predict) · jacobian`
//!
//! This makes new model extensions (background, temperature, etc.) work with
//! both solvers automatically — implement `ForwardModel` once, both solvers
//! benefit.

use crate::error::FittingError;

/// Solver-agnostic forward model.
///
/// Implementations provide the model prediction and (optionally) its
/// analytical Jacobian.  Solvers wrap this trait to compute solver-specific
/// objectives (chi-squared, Poisson NLL, etc.).
pub trait ForwardModel {
    /// Predict model output for the given parameter vector.
    ///
    /// Returns a vector of predicted values (transmission, counts, etc.)
    /// with length equal to the number of data points.
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError>;

    /// Analytical Jacobian: `J[i][j] = ∂predict[i] / ∂params[j]`.
    ///
    /// Only the columns corresponding to `free_param_indices` are needed.
    /// Returns `None` to signal "use finite differences" (the default).
    ///
    /// `y_current` is the output of `predict(params)`, provided so
    /// implementations can avoid redundant computation.
    fn jacobian(
        &self,
        _params: &[f64],
        _free_param_indices: &[usize],
        _y_current: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        None
    }

    /// Number of data points in the model output.
    fn n_data(&self) -> usize;

    /// Number of parameters (total, including fixed).
    fn n_params(&self) -> usize;
}
