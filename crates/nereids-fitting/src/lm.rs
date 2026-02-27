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

    /// Optionally provide an analytical Jacobian.
    ///
    /// `free_param_indices`: indices (into `params`) of the free parameters,
    /// in the same order as the Jacobian columns.
    ///
    /// `y_current`: current model output, i.e. `self.evaluate(params)`.
    /// Provided so implementations can compute J analytically from T without
    /// an extra `evaluate` call.
    ///
    /// Returns `Some(J)` where `J[i][j] = ∂model[i]/∂params[free_param_indices[j]]`.
    /// Return `None` to fall back to finite-difference Jacobian (the default).
    fn analytical_jacobian(
        &self,
        _params: &[f64],
        _free_param_indices: &[usize],
        _y_current: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        None
    }
}

/// Compute weighted chi-squared: Σ [(y_obs - y_model)² / σ²].
fn chi_squared(residuals: &[f64], weights: &[f64]) -> f64 {
    residuals
        .iter()
        .zip(weights.iter())
        .map(|(&r, &w)| r * r * w)
        .sum()
}

/// Compute the Jacobian, preferring an analytical formula over finite differences.
///
/// `y_current` must equal `model.evaluate(&params.all_values())` at the
/// current parameter values — it is passed in to avoid a redundant evaluate
/// call (the LM loop already has this vector from the previous accepted step).
///
/// J[i][j] = ∂model[i] / ∂free_param[j]
fn compute_jacobian(
    model: &dyn FitModel,
    params: &mut ParameterSet,
    y_current: &[f64],
    fd_step: f64,
) -> Vec<Vec<f64>> {
    let free_indices = params.free_indices();
    let n_free = free_indices.len();
    let n_data = y_current.len();

    // Try analytical Jacobian first (no extra evaluate calls).
    if let Some(j) = model.analytical_jacobian(&params.all_values(), &free_indices, y_current) {
        return j;
    }

    // Fallback: forward finite differences, reusing y_current as the base.
    let mut jacobian = vec![vec![0.0; n_free]; n_data];

    for (j, &idx) in free_indices.iter().enumerate() {
        let original = params.params[idx].value;
        let step = fd_step * (1.0 + original.abs());

        params.params[idx].value = original + step;
        params.params[idx].clamp();
        let mut actual_step = params.params[idx].value - original;

        // #112: If the forward step is blocked by an upper bound, try the
        // backward step so the Jacobian column is not frozen at zero.
        if actual_step.abs() < 1e-30 {
            params.params[idx].value = original - step;
            params.params[idx].clamp();
            actual_step = params.params[idx].value - original;
            if actual_step.abs() < 1e-30 {
                // Truly stuck at a point constraint — skip this parameter.
                params.params[idx].value = original;
                continue;
            }
        }

        let perturbed = model.evaluate(&params.all_values());
        params.params[idx].value = original;

        for i in 0..n_data {
            jacobian[i][j] = (perturbed[i] - y_current[i]) / actual_step;
        }
    }

    jacobian
}

/// Solve (A + λ·diag(A)) · x = b using Gaussian elimination.
///
/// A is n×n symmetric positive definite (approximately).
/// Returns the solution vector x.
#[allow(clippy::needless_range_loop)]
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
#[allow(clippy::needless_range_loop)]
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

    let inv: Vec<Vec<f64>> = aug.into_iter().map(|row| row[n..].to_vec()).collect();

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
    assert!(n_data > 0, "y_obs must not be empty",);
    assert_eq!(
        sigma.len(),
        n_data,
        "sigma length ({}) must match y_obs length ({})",
        sigma.len(),
        n_data,
    );

    let n_free = params.n_free();

    // Early return when all parameters are fixed: evaluate once and report the
    // model's chi-squared.  There is nothing to optimize, so iterating would
    // waste cycles.
    if n_free == 0 {
        let weights: Vec<f64> = sigma
            .iter()
            .map(|&s| {
                if !s.is_finite() || s <= 0.0 {
                    1.0 / 1e30
                } else {
                    1.0 / (s * s)
                }
            })
            .collect();
        let y_model = model.evaluate(&params.all_values());
        let residuals: Vec<f64> = y_obs
            .iter()
            .zip(y_model.iter())
            .map(|(&obs, &mdl)| obs - mdl)
            .collect();
        let chi2 = chi_squared(&residuals, &weights);
        return LmResult {
            chi_squared: chi2,
            reduced_chi_squared: chi2 / n_data as f64,
            iterations: 0,
            converged: true,
            params: params.all_values(),
            covariance: Some(vec![]),
            uncertainties: Some(vec![]),
        };
    }

    // #108.3: Underdetermined systems — when n_data <= n_free, the problem is
    // underdetermined and reduced chi-squared is meaningless.  Return early
    // with converged=false so callers can detect the problem.
    if n_data <= n_free {
        return LmResult {
            chi_squared: f64::NAN,
            reduced_chi_squared: f64::NAN,
            iterations: 0,
            converged: false,
            params: params.all_values(),
            covariance: None,
            uncertainties: None,
        };
    }
    let dof = n_data - n_free;

    // #104: Validate sigma — division by zero or non-finite sigma would produce
    // NaN/Inf weights and silently corrupt the entire fit.  Clamp to a small
    // floor instead of rejecting outright, so callers with a few zero-sigma
    // bins still get a usable fit.
    let weights: Vec<f64> = sigma
        .iter()
        .map(|&s| {
            if !s.is_finite() || s <= 0.0 {
                // Treat as negligible weight (huge sigma) rather than panicking.
                1.0 / 1e30
            } else {
                1.0 / (s * s)
            }
        })
        .collect();

    // Initial model output, residuals, and chi².
    // y_current is kept up-to-date after accepted steps so that the next
    // Jacobian call can reuse it without an extra evaluate() call.
    let mut y_current = model.evaluate(&params.all_values());
    let mut residuals: Vec<f64> = y_obs
        .iter()
        .zip(y_current.iter())
        .map(|(&obs, &mdl)| obs - mdl)
        .collect();
    let mut chi2 = chi_squared(&residuals, &weights);

    let mut lambda = config.lambda_init;
    let mut converged = false;
    let mut iter = 0;

    for _ in 0..config.max_iter {
        iter += 1;

        // Compute Jacobian — uses y_current to avoid a redundant evaluate().
        // Analytical Jacobian (if provided by the model) costs 0 extra evaluates;
        // finite-difference fallback costs N_free extra evaluates.
        let jacobian = compute_jacobian(model, params, &y_current, config.fd_step);

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

        // #113: If the model produced NaN/Inf, treat as a bad step (same as
        // chi2 increase) — increase lambda and try again.
        if y_trial.iter().any(|v| !v.is_finite()) {
            params.set_free_values(&old_free);
            lambda *= config.lambda_up;
            if lambda > 1e16 {
                converged = false;
                break;
            }
            continue;
        }

        let trial_residuals: Vec<f64> = y_obs
            .iter()
            .zip(y_trial.iter())
            .map(|(&obs, &mdl)| obs - mdl)
            .collect();
        let trial_chi2 = chi_squared(&trial_residuals, &weights);

        if trial_chi2 < chi2 {
            // Accept step — cache y_trial so the next iteration can skip
            // the base evaluate() inside compute_jacobian.
            let rel_change = (chi2 - trial_chi2) / (chi2 + 1e-30);
            chi2 = trial_chi2;
            residuals = trial_residuals;
            y_current = y_trial;
            lambda *= config.lambda_down;

            // Check convergence: relative chi2 change is tiny or parameters
            // stopped moving.  The old third condition
            // `chi2 < tol_chi2 * n_data` was scale-dependent and could cause
            // premature convergence on data with small residuals.  (#108.2)
            let param_change: f64 = delta
                .iter()
                .zip(old_free.iter())
                .map(|(&d, &v)| (d / (v.abs() + 1e-30)).powi(2))
                .sum::<f64>()
                .sqrt();

            if rel_change < config.tol_chi2 || param_change < config.tol_param {
                converged = true;
                break;
            }
        } else {
            // Reject step, restore parameters.
            // y_current stays valid (parameters reverted to old_free).
            params.set_free_values(&old_free);
            lambda *= config.lambda_up;

            // #108.4: If lambda is astronomically large, the optimizer is stuck
            // in a region where no step improves chi2.  Break out rather than
            // wasting iterations.
            if lambda > 1e16 {
                converged = false;
                break;
            }
        }
    }

    // Compute covariance matrix: (JᵀWJ)⁻¹ at the final parameters.
    let jacobian = compute_jacobian(model, params, &y_current, config.fd_step);
    let mut jtw_j = vec![vec![0.0; n_free]; n_free];
    for i in 0..n_data {
        for j in 0..n_free {
            for k in 0..n_free {
                jtw_j[j][k] += jacobian[i][j] * weights[i] * jacobian[i][k];
            }
        }
    }

    // #108.1: Scale covariance by reduced chi-squared.
    //
    // The raw (JᵀWJ)⁻¹ gives the covariance only when the model is a perfect
    // description and the weights are exact.  Multiplying by χ²/ν accounts for
    // misfit (model inadequacy or underestimated errors).  This is the standard
    // statistical prescription (see e.g. Numerical Recipes §15.6).
    let reduced_chi2 = chi2 / dof as f64;
    let (covariance, uncertainties) = if let Some(mut cov) = invert_matrix(&jtw_j) {
        for row in cov.iter_mut() {
            for elem in row.iter_mut() {
                *elem *= reduced_chi2;
            }
        }
        let unc: Vec<f64> = (0..n_free)
            .map(|i| {
                if cov[i][i].is_finite() && cov[i][i] > 0.0 {
                    cov[i][i].sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect();
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
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("a", 1.0)]);

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
