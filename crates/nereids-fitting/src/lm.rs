//! Levenberg-Marquardt least-squares optimizer.
//!
//! Minimizes χ² = Σᵢ [(y_obs - y_model)² / σᵢ²] by iteratively solving:
//!
//!   (JᵀWJ + λ·diag(JᵀWJ)) · δ = JᵀW·r
//!
//! where J is the Jacobian, W = diag(1/σ²), r = y_obs - y_model,
//! and λ is the damping parameter.
//!
//! ## SAMMY Reference
//! - `fit/` module, manual Sec 4 (Bayes equations / generalized least-squares)

use crate::parameters::ParameterSet;

/// Configuration for the LM optimizer.
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Initial damping parameter λ.
    pub lambda_init: f64,
    /// Factor to increase λ on rejected step.
    pub lambda_up: f64,
    /// Factor to decrease λ on accepted step.
    pub lambda_down: f64,
    /// Convergence tolerance on relative χ² change.
    pub tol_chi2: f64,
    /// Convergence tolerance on relative parameter change.
    pub tol_param: f64,
    /// Step size for finite-difference Jacobian.
    pub fd_step: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            lambda_init: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            tol_chi2: 1e-8,
            tol_param: 1e-8,
            fd_step: 1e-6,
        }
    }
}

/// Result of a Levenberg-Marquardt fit.
#[derive(Debug, Clone)]
pub struct LmResult {
    /// Final chi-squared value.
    pub chi_squared: f64,
    /// Reduced chi-squared (χ²/ν where ν = n_data - n_params).
    pub reduced_chi_squared: f64,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Whether the fit converged.
    pub converged: bool,
    /// Final parameter values (all parameters, including fixed).
    pub params: Vec<f64>,
    /// Covariance matrix of free parameters (n_free × n_free), if available.
    pub covariance: Option<Vec<Vec<f64>>>,
    /// Standard errors of free parameters (diagonal of covariance).
    pub uncertainties: Option<Vec<f64>>,
}

/// A model function that can be fitted.
///
/// Given parameter values (all params including fixed), computes
/// the model prediction at each data point.
pub trait FitModel {
    /// Evaluate the model for the given parameters.
    ///
    /// Returns a vector of model predictions, same length as the data.
    fn evaluate(&self, params: &[f64]) -> Vec<f64>;
}

/// Compute weighted chi-squared: Σ [(y_obs - y_model)² / σ²].
fn chi_squared(residuals: &[f64], weights: &[f64]) -> f64 {
    residuals
        .iter()
        .zip(weights.iter())
        .map(|(&r, &w)| r * r * w)
        .sum()
}

/// Compute the Jacobian by finite differences.
///
/// J[i][j] = ∂model[i] / ∂param[j]
fn compute_jacobian(
    model: &dyn FitModel,
    params: &mut ParameterSet,
    fd_step: f64,
) -> Vec<Vec<f64>> {
    let base = model.evaluate(&params.all_values());
    let n_data = base.len();
    let free_indices = params.free_indices();
    let n_free = free_indices.len();

    let mut jacobian = vec![vec![0.0; n_free]; n_data];

    for (j, &idx) in free_indices.iter().enumerate() {
        let original = params.params[idx].value;
        let step = fd_step * (1.0 + original.abs());

        params.params[idx].value = original + step;
        params.params[idx].clamp();
        let actual_step = params.params[idx].value - original;

        if actual_step.abs() < 1e-30 {
            // Parameter is at a bound, cannot step forward
            params.params[idx].value = original;
            continue;
        }

        let perturbed = model.evaluate(&params.all_values());
        params.params[idx].value = original;

        for i in 0..n_data {
            jacobian[i][j] = (perturbed[i] - base[i]) / actual_step;
        }
    }

    jacobian
}

/// Solve (A + λ·diag(A)) · x = b using Gaussian elimination.
///
/// A is n×n symmetric positive definite (approximately).
/// Returns the solution vector x.
fn solve_damped_system(a: &[Vec<f64>], b: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(vec![]);
    }

    // Build the augmented matrix [A + λ·diag(A) | b]
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][i] += lambda * a[i][i].max(1e-10); // Ensure non-zero diagonal
        aug[i][n] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-30 {
            return None; // Singular
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Invert a symmetric positive definite matrix (for covariance).
fn invert_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Some(vec![]);
    }

    // Build [A | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-30 {
            return None;
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let inv: Vec<Vec<f64>> = aug
        .into_iter()
        .map(|row| row[n..].to_vec())
        .collect();

    Some(inv)
}

/// Run the Levenberg-Marquardt optimizer.
///
/// # Arguments
/// * `model` — Forward model implementing `FitModel`.
/// * `y_obs` — Observed data values.
/// * `sigma` — Uncertainties on observed data (standard deviations).
/// * `params` — Initial parameter set (modified in place on convergence).
/// * `config` — LM configuration.
///
/// # Returns
/// Fit result including final parameters, chi-squared, and uncertainties.
pub fn levenberg_marquardt(
    model: &dyn FitModel,
    y_obs: &[f64],
    sigma: &[f64],
    params: &mut ParameterSet,
    config: &LmConfig,
) -> LmResult {
    let n_data = y_obs.len();
    let n_free = params.n_free();
    let dof = if n_data > n_free { n_data - n_free } else { 1 };

    // Weights = 1/σ²
    let weights: Vec<f64> = sigma.iter().map(|&s| 1.0 / (s * s)).collect();

    // Initial residuals and chi²
    let y_model = model.evaluate(&params.all_values());
    let mut residuals: Vec<f64> = y_obs
        .iter()
        .zip(y_model.iter())
        .map(|(&obs, &mdl)| obs - mdl)
        .collect();
    let mut chi2 = chi_squared(&residuals, &weights);

    let mut lambda = config.lambda_init;
    let mut converged = false;
    let mut iter = 0;

    for _ in 0..config.max_iter {
        iter += 1;

        // Compute Jacobian
        let jacobian = compute_jacobian(model, params, config.fd_step);

        // Build normal equations: JᵀWJ and JᵀWr
        let mut jtw_j = vec![vec![0.0; n_free]; n_free];
        let mut jtw_r = vec![0.0; n_free];

        for i in 0..n_data {
            for j in 0..n_free {
                jtw_r[j] += jacobian[i][j] * weights[i] * residuals[i];
                for k in 0..n_free {
                    jtw_j[j][k] += jacobian[i][j] * weights[i] * jacobian[i][k];
                }
            }
        }

        // Solve (JᵀWJ + λ·diag(JᵀWJ)) · δ = JᵀWr
        let delta = match solve_damped_system(&jtw_j, &jtw_r, lambda) {
            Some(d) => d,
            None => break, // Singular system
        };

        // Trial step
        let old_free = params.free_values();
        let trial_free: Vec<f64> = old_free
            .iter()
            .zip(delta.iter())
            .map(|(&v, &d)| v + d)
            .collect();
        params.set_free_values(&trial_free);

        let y_trial = model.evaluate(&params.all_values());
        let trial_residuals: Vec<f64> = y_obs
            .iter()
            .zip(y_trial.iter())
            .map(|(&obs, &mdl)| obs - mdl)
            .collect();
        let trial_chi2 = chi_squared(&trial_residuals, &weights);

        if trial_chi2 < chi2 {
            // Accept step
            let rel_change = (chi2 - trial_chi2) / (chi2 + 1e-30);
            chi2 = trial_chi2;
            residuals = trial_residuals;
            lambda *= config.lambda_down;

            // Check convergence: either relative chi2 change is tiny,
            // or chi2 itself is essentially zero, or parameters stopped moving.
            let param_change: f64 = delta
                .iter()
                .zip(old_free.iter())
                .map(|(&d, &v)| (d / (v.abs() + 1e-30)).powi(2))
                .sum::<f64>()
                .sqrt();

            if rel_change < config.tol_chi2
                || param_change < config.tol_param
                || chi2 < config.tol_chi2 * n_data as f64
            {
                converged = true;
                break;
            }
        } else {
            // Reject step, restore parameters
            params.set_free_values(&old_free);
            lambda *= config.lambda_up;
        }
    }

    // Compute covariance matrix: (JᵀWJ)⁻¹
    let jacobian = compute_jacobian(model, params, config.fd_step);
    let mut jtw_j = vec![vec![0.0; n_free]; n_free];
    for i in 0..n_data {
        for j in 0..n_free {
            for k in 0..n_free {
                jtw_j[j][k] += jacobian[i][j] * weights[i] * jacobian[i][k];
            }
        }
    }

    let (covariance, uncertainties) = if let Some(cov) = invert_matrix(&jtw_j) {
        let unc: Vec<f64> = (0..n_free).map(|i| cov[i][i].max(0.0).sqrt()).collect();
        (Some(cov), Some(unc))
    } else {
        (None, None)
    };

    LmResult {
        chi_squared: chi2,
        reduced_chi_squared: chi2 / dof as f64,
        iterations: iter,
        converged,
        params: params.all_values(),
        covariance,
        uncertainties,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::{FitParameter, ParameterSet};

    /// Simple linear model: y = a*x + b
    struct LinearModel {
        x: Vec<f64>,
    }

    impl FitModel for LinearModel {
        fn evaluate(&self, params: &[f64]) -> Vec<f64> {
            let a = params[0];
            let b = params[1];
            self.x.iter().map(|&x| a * x + b).collect()
        }
    }

    #[test]
    fn test_fit_linear_exact() {
        // Fit y = 2x + 3 with exact data (no noise)
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let sigma = vec![1.0; 10];

        let model = LinearModel { x };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0), // initial guess
            FitParameter::unbounded("b", 1.0),
        ]);

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

        assert!(result.converged, "Fit did not converge");
        assert!(
            (result.params[0] - 2.0).abs() < 1e-4,
            "a = {}, expected 2.0",
            result.params[0]
        );
        assert!(
            (result.params[1] - 3.0).abs() < 1e-4,
            "b = {}, expected 3.0",
            result.params[1]
        );
        assert!(result.chi_squared < 1e-6);
    }

    #[test]
    fn test_fit_linear_with_fixed_param() {
        // Fit y = a*x + 3 with b fixed at 3.0
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let sigma = vec![1.0; 10];

        let model = LinearModel { x };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0),
            FitParameter::fixed("b", 3.0),
        ]);

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

        assert!(result.converged);
        assert!(
            (result.params[0] - 2.0).abs() < 1e-6,
            "a = {}",
            result.params[0]
        );
        assert_eq!(result.params[1], 3.0); // fixed
    }

    #[test]
    fn test_fit_quadratic() {
        // Fit y = a*x² + b*x + c to quadratic data
        struct QuadModel {
            x: Vec<f64>,
        }
        impl FitModel for QuadModel {
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                let (a, b, c) = (params[0], params[1], params[2]);
                self.x.iter().map(|&x| a * x * x + b * x + c).collect()
            }
        }

        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 0.5 * xi * xi - 2.0 * xi + 1.0).collect();
        let sigma = vec![1.0; 20];

        let model = QuadModel { x };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0),
            FitParameter::unbounded("b", 0.0),
            FitParameter::unbounded("c", 0.0),
        ]);

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

        assert!(result.converged);
        assert!(
            (result.params[0] - 0.5).abs() < 1e-5,
            "a = {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - (-2.0)).abs() < 1e-5,
            "b = {}",
            result.params[1]
        );
        assert!(
            (result.params[2] - 1.0).abs() < 1e-5,
            "c = {}",
            result.params[2]
        );
    }

    #[test]
    fn test_non_negative_constraint() {
        // Fit y = a*x with data that has negative slope,
        // but parameter a is constrained to be non-negative.
        // Should converge to a ≈ 0.
        struct SlopeModel {
            x: Vec<f64>,
        }
        impl FitModel for SlopeModel {
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                let a = params[0];
                self.x.iter().map(|&x| a * x).collect()
            }
        }

        let x: Vec<f64> = (1..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| -2.0 * xi).collect();
        let sigma = vec![1.0; 9];

        let model = SlopeModel { x };
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("a", 1.0),
        ]);

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

        // Should be clamped at 0
        assert!(
            result.params[0] >= 0.0 && result.params[0] < 0.1,
            "a = {}, expected ~0",
            result.params[0]
        );
    }

    #[test]
    fn test_uncertainty_estimation() {
        // Fit linear model; uncertainties should be reasonable
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let sigma = vec![0.1; 100]; // Small uncertainty

        let model = LinearModel { x };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0),
            FitParameter::unbounded("b", 1.0),
        ]);

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

        assert!(result.converged);
        assert!(result.uncertainties.is_some());
        let unc = result.uncertainties.unwrap();
        // Uncertainties should be positive and small
        assert!(unc[0] > 0.0 && unc[0] < 0.01, "σ_a = {}", unc[0]);
        assert!(unc[1] > 0.0 && unc[1] < 0.1, "σ_b = {}", unc[1]);
    }

    #[test]
    fn test_solve_damped_system_identity() {
        // (I + λ·I)x = b → x = b/(1+λ)
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![2.0, 4.0];
        let lambda = 1.0;
        let x = solve_damped_system(&a, &b, lambda).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_matrix_2x2() {
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert_matrix(&a).unwrap();
        // A⁻¹ = 1/10 × [6 -7; -2 4]
        assert!((inv[0][0] - 0.6).abs() < 1e-10);
        assert!((inv[0][1] - (-0.7)).abs() < 1e-10);
        assert!((inv[1][0] - (-0.2)).abs() < 1e-10);
        assert!((inv[1][1] - 0.4).abs() < 1e-10);
    }
}
