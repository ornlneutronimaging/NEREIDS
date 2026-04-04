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

use nereids_core::constants::{LM_DIAGONAL_FLOOR, PIVOT_FLOOR};

use crate::error::FittingError;
use crate::parameters::ParameterSet;

/// Row-major flat matrix for cache-friendly storage.
///
/// Replaces `Vec<Vec<f64>>` to collapse ~N separate heap allocations into 1
/// and improve cache locality for JtWJ assembly.  Access: `data[i * ncols + j]`.
#[derive(Debug, Clone)]
pub struct FlatMatrix {
    /// Flat row-major storage: `data[i * ncols + j]` = element at row i, col j.
    pub data: Vec<f64>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl FlatMatrix {
    /// Create a new zero-filled matrix with the given dimensions.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let len = nrows
            .checked_mul(ncols)
            .expect("FlatMatrix dimensions overflow usize");
        Self {
            data: vec![0.0; len],
            nrows,
            ncols,
        }
    }

    /// Access element at (row, col) immutably.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.nrows && col < self.ncols);
        self.data[row * self.ncols + col]
    }

    /// Access element at (row, col) mutably.
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        debug_assert!(row < self.nrows && col < self.ncols);
        &mut self.data[row * self.ncols + col]
    }
}

/// #125.4: Maximum damping parameter before the optimizer gives up.
///
/// When λ exceeds this threshold, the optimizer is stuck in a region where no
/// step improves chi-squared.  Breaking out avoids wasting iterations.
const LAMBDA_BREAKOUT: f64 = 1e16;

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
    /// Whether to compute the covariance matrix (and uncertainties) after
    /// convergence.  This requires an extra Jacobian evaluation + matrix
    /// inversion at the final parameters.  Set to `false` for per-pixel
    /// spatial mapping where only densities are needed.
    ///
    /// Default: `true`.
    pub compute_covariance: bool,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            lambda_init: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            tol_chi2: 1e-8,
            tol_param: 1e-8,
            fd_step: 1e-6,
            compute_covariance: true,
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
    pub covariance: Option<FlatMatrix>,
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
    /// On success, returns a vector of model predictions with the same
    /// length as the data being fitted. On failure, returns a
    /// [`FittingError`] indicating that the model could not be evaluated
    /// (e.g. a broadening or physics error).
    ///
    /// **Optimizer semantics:** during Levenberg-Marquardt trial steps,
    /// an `Err` is treated as a failed step (increase λ / backtrack).
    /// At the initial point or post-convergence, `Err` is propagated.
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError>;

    /// Optionally provide an analytical Jacobian.
    ///
    /// `free_param_indices`: indices (into `params`) of the free parameters,
    /// in the same order as the Jacobian columns.
    ///
    /// `y_current`: current model output, i.e. `self.evaluate(params)`.
    /// Provided so implementations can compute J analytically from T without
    /// an extra `evaluate` call.
    ///
    /// Returns `Some(J)` where `J.get(i, j) = ∂model[i]/∂params[free_param_indices[j]]`.
    /// The matrix has `y_current.len()` rows and `free_param_indices.len()` columns.
    /// Return `None` to fall back to finite-difference Jacobian (the default).
    fn analytical_jacobian(
        &self,
        _params: &[f64],
        _free_param_indices: &[usize],
        _y_current: &[f64],
    ) -> Option<FlatMatrix> {
        None
    }
}

/// Blanket implementation: shared references to any `FitModel` also implement
/// `FitModel`, forwarding all calls to the underlying implementation.
///
/// This enables `NormalizedTransmissionModel<&dyn FitModel>` to work when the
/// inner model is borrowed (e.g. in `fit_spectrum` where the inner model is a
/// local variable wrapped conditionally).
impl<M: FitModel + ?Sized> FitModel for &M {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        (**self).evaluate(params)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        (**self).analytical_jacobian(params, free_param_indices, y_current)
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

/// Infinity norm of the gradient, scaled by local curvature and residual size.
///
/// This is a dimensionless first-order optimality measure for least squares:
/// small values indicate that the current point is stationary even when χ² is
/// nonzero because the data are noisy or the model is imperfect.
fn scaled_gradient_inf_norm(jtw_j: &FlatMatrix, jtw_r: &[f64], chi2: f64) -> f64 {
    let residual_scale = chi2.sqrt().max(1.0);
    let mut max_scaled: f64 = 0.0;
    for (j, &grad_j) in jtw_r.iter().enumerate() {
        let curvature = jtw_j.get(j, j).abs().sqrt();
        let scale = curvature * residual_scale + PIVOT_FLOOR;
        max_scaled = max_scaled.max(grad_j.abs() / scale);
    }
    max_scaled
}

/// Whether the local model has meaningful curvature in at least one free direction.
///
/// Purely flat models (J = 0 everywhere) should still report failure rather than
/// "converged", even though their gradient is numerically zero.
fn has_informative_curvature(jtw_j: &FlatMatrix) -> bool {
    (0..jtw_j.nrows).any(|j| jtw_j.get(j, j) > LM_DIAGONAL_FLOOR)
}

/// Compute the Jacobian, preferring an analytical formula over finite differences.
///
/// `y_current` must equal `model.evaluate(&params.all_values())` at the
/// current parameter values — it is passed in to avoid a redundant evaluate
/// call (the LM loop already has this vector from the previous accepted step).
///
/// `all_vals_buf` is a scratch buffer reused across the per-parameter FD loop
/// to avoid allocating a fresh `Vec<f64>` on every `model.evaluate()` call.
///
/// `free_idx_buf` is a scratch buffer for `params.free_indices_into()`, reused
/// across iterations to avoid per-Jacobian allocation.
///
/// J.get(i, j) = ∂model[i] / ∂free_param[j]
fn compute_jacobian(
    model: &dyn FitModel,
    params: &mut ParameterSet,
    y_current: &[f64],
    fd_step: f64,
    all_vals_buf: &mut Vec<f64>,
    free_idx_buf: &mut Vec<usize>,
) -> Result<FlatMatrix, FittingError> {
    params.free_indices_into(free_idx_buf);
    let n_free = free_idx_buf.len();
    let n_data = y_current.len();

    // Try analytical Jacobian first (no extra evaluate calls).
    params.all_values_into(all_vals_buf);
    if let Some(j) = model.analytical_jacobian(all_vals_buf, free_idx_buf, y_current) {
        debug_assert!(
            j.nrows == n_data && j.ncols == n_free && j.data.len() == n_data * n_free,
            "analytical_jacobian shape mismatch: got ({}x{}, len={}), expected ({}x{}, len={})",
            j.nrows,
            j.ncols,
            j.data.len(),
            n_data,
            n_free,
            n_data * n_free,
        );
        return Ok(j);
    }

    // Fallback: forward finite differences, reusing y_current as the base.
    let mut jacobian = FlatMatrix::zeros(n_data, n_free);

    for (j, &idx) in free_idx_buf.iter().enumerate() {
        let original = params.params[idx].value;
        let step = fd_step * (1.0 + original.abs());

        params.params[idx].value = original + step;
        params.params[idx].clamp();
        let mut actual_step = params.params[idx].value - original;

        // #112: If the forward step is blocked by an upper bound, try the
        // backward step so the Jacobian column is not frozen at zero.
        if actual_step.abs() < PIVOT_FLOOR {
            params.params[idx].value = original - step;
            params.params[idx].clamp();
            actual_step = params.params[idx].value - original;
            if actual_step.abs() < PIVOT_FLOOR {
                // Truly stuck at a point constraint — skip this parameter.
                params.params[idx].value = original;
                continue;
            }
        }

        params.all_values_into(all_vals_buf);
        let perturbed = match model.evaluate(all_vals_buf) {
            Ok(v) => v,
            Err(_) => {
                // Restore original before skipping — leaving the column as
                // zero makes this parameter unresponsive for one LM step,
                // which is safe (matches Poisson compute_gradient pattern).
                params.params[idx].value = original;
                continue;
            }
        };
        params.params[idx].value = original;

        for i in 0..n_data {
            *jacobian.get_mut(i, j) = (perturbed[i] - y_current[i]) / actual_step;
        }
    }

    Ok(jacobian)
}

/// Solve (A + λ·diag(A)) · x = b using Gaussian elimination.
///
/// A is a flat n×n symmetric positive definite matrix (approximately).
/// Returns the solution vector x.
pub(crate) fn solve_damped_system(a: &FlatMatrix, b: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(vec![]);
    }

    // Build the augmented matrix [A + λ·diag(A) | b] as flat (n × (n+1)).
    let ncols = n + 1;
    let mut aug = FlatMatrix::zeros(n, ncols);
    for (i, &bi) in b.iter().enumerate() {
        for j in 0..n {
            *aug.get_mut(i, j) = a.get(i, j);
        }
        *aug.get_mut(i, i) += lambda * a.get(i, i).max(LM_DIAGONAL_FLOOR); // Ensure non-zero diagonal
        *aug.get_mut(i, n) = bi;
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug.get(col, col).abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug.get(row, col).abs() > max_val {
                max_val = aug.get(row, col).abs();
                max_row = row;
            }
        }

        if max_val < PIVOT_FLOOR {
            return None; // Singular
        }

        // Swap rows col and max_row in the flat buffer.
        if col != max_row {
            let (row_a, row_b) = (col * ncols, max_row * ncols);
            let (first, second) = aug.data.split_at_mut(row_b);
            first[row_a..row_a + ncols].swap_with_slice(&mut second[..ncols]);
        }

        let pivot = aug.get(col, col);
        for row in (col + 1)..n {
            let factor = aug.get(row, col) / pivot;
            for j in col..=n {
                let val = aug.get(col, j);
                *aug.get_mut(row, j) -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug.get(i, n);
        for (j, &xj) in x.iter().enumerate().skip(i + 1) {
            sum -= aug.get(i, j) * xj;
        }
        x[i] = sum / aug.get(i, i);
    }

    Some(x)
}

/// Invert a symmetric positive definite matrix (for covariance).
///
/// Input: flat n×n matrix. Output: flat n×n inverse, or None if singular.
pub(crate) fn invert_matrix(a: &FlatMatrix) -> Option<FlatMatrix> {
    let n = a.nrows;
    if n == 0 {
        return Some(FlatMatrix::zeros(0, 0));
    }

    // Build [A | I] as flat (n × 2n).
    let ncols = 2 * n;
    let mut aug = FlatMatrix::zeros(n, ncols);
    for i in 0..n {
        for j in 0..n {
            *aug.get_mut(i, j) = a.get(i, j);
        }
        *aug.get_mut(i, n + i) = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_val = aug.get(col, col).abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug.get(row, col).abs() > max_val {
                max_val = aug.get(row, col).abs();
                max_row = row;
            }
        }

        if max_val < PIVOT_FLOOR {
            return None;
        }

        // Swap rows col and max_row.
        if col != max_row {
            let (row_a, row_b) = (col * ncols, max_row * ncols);
            let (first, second) = aug.data.split_at_mut(row_b);
            first[row_a..row_a + ncols].swap_with_slice(&mut second[..ncols]);
        }

        let pivot = aug.get(col, col);
        for j in 0..ncols {
            *aug.get_mut(col, j) /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug.get(row, col);
                for j in 0..ncols {
                    let val = aug.get(col, j);
                    *aug.get_mut(row, j) -= factor * val;
                }
            }
        }
    }

    // Extract the right half [I|A⁻¹] → A⁻¹
    let mut inv = FlatMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *inv.get_mut(i, j) = aug.get(i, n + j);
        }
    }

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
) -> Result<LmResult, FittingError> {
    let n_data = y_obs.len();
    if n_data == 0 {
        return Err(FittingError::EmptyData);
    }
    if sigma.len() != n_data {
        return Err(FittingError::LengthMismatch {
            expected: n_data,
            actual: sigma.len(),
            field: "sigma",
        });
    }

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
        let y_model = model.evaluate(&params.all_values())?;

        // #P1: If the model produces NaN/Inf with all-fixed parameters,
        // return converged=false rather than silently propagating NaN chi².
        // Covariance/uncertainties are None because the fit did not converge —
        // an unconverged result has no meaningful covariance to report.
        if !y_model.iter().all(|v| v.is_finite()) {
            return Ok(LmResult {
                chi_squared: f64::NAN,
                reduced_chi_squared: f64::NAN,
                iterations: 0,
                converged: false,
                params: params.all_values(),
                covariance: None,
                uncertainties: None,
            });
        }

        let residuals: Vec<f64> = y_obs
            .iter()
            .zip(y_model.iter())
            .map(|(&obs, &mdl)| obs - mdl)
            .collect();
        let chi2 = chi_squared(&residuals, &weights);
        // #125.5: Compute dof via `n_data - n_free` (even though n_free == 0 here)
        // to mirror the main path and keep a single visible formula.
        let dof = n_data - n_free;
        // Note: Given n_free == 0 and the earlier assert that n_data > 0,
        // dof is always > 0 here; the guard below is a defensive check
        // kept for consistency with the main path.
        let reduced = if dof > 0 { chi2 / dof as f64 } else { f64::NAN };
        return Ok(LmResult {
            chi_squared: chi2,
            reduced_chi_squared: reduced,
            iterations: 0,
            converged: true,
            params: params.all_values(),
            covariance: Some(FlatMatrix::zeros(0, 0)),
            uncertainties: Some(vec![]),
        });
    }

    // #108.3: Underdetermined systems — when n_data < n_free, the problem is
    // underdetermined and the Jacobian cannot be full rank.  Return early
    // with converged=false so callers can detect the problem.
    // n_data == n_free is exactly determined (dof=0) and still solvable;
    // we allow it and report reduced_chi_squared = NaN (0/0).
    if n_data < n_free {
        return Ok(LmResult {
            chi_squared: f64::NAN,
            reduced_chi_squared: f64::NAN,
            iterations: 0,
            converged: false,
            params: params.all_values(),
            covariance: None,
            uncertainties: None,
        });
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

    // Scratch buffers reused across the optimization loop for
    // params.all_values_into() calls in compute_jacobian (1 + N_free calls
    // per Jacobian computation) and the trial-step evaluation,
    // params.free_values_into() calls for snapshotting free parameters
    // before trial steps, and params.free_indices_into() calls inside
    // compute_jacobian.
    let mut all_vals_buf = Vec::with_capacity(params.params.len());
    let mut free_vals_buf = Vec::with_capacity(n_free);
    let mut free_idx_buf = Vec::with_capacity(n_free);

    // Initial model output, residuals, and chi².
    // y_current is kept up-to-date after accepted steps so that the next
    // Jacobian call can reuse it without an extra evaluate() call.
    params.all_values_into(&mut all_vals_buf);
    let mut y_current = model.evaluate(&all_vals_buf)?;
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
        let jacobian = compute_jacobian(
            model,
            params,
            &y_current,
            config.fd_step,
            &mut all_vals_buf,
            &mut free_idx_buf,
        )?;

        // Build normal equations: JᵀWJ and JᵀWr
        let mut jtw_j = FlatMatrix::zeros(n_free, n_free);
        let mut jtw_r = vec![0.0; n_free];

        for (i, (&wi, &ri)) in weights.iter().zip(residuals.iter()).enumerate() {
            for (j, jtw_r_j) in jtw_r.iter_mut().enumerate() {
                let jij = jacobian.get(i, j);
                *jtw_r_j += jij * wi * ri;
                for k in 0..n_free {
                    *jtw_j.get_mut(j, k) += jij * wi * jacobian.get(i, k);
                }
            }
        }
        let scaled_grad_inf = scaled_gradient_inf_norm(&jtw_j, &jtw_r, chi2);
        let informative_curvature = has_informative_curvature(&jtw_j);

        // Solve (JᵀWJ + λ·diag(JᵀWJ)) · δ = JᵀWr
        let delta = match solve_damped_system(&jtw_j, &jtw_r, lambda) {
            Some(d) => d,
            None => break, // Singular system
        };

        // Trial step — snapshot free values into a reusable buffer to avoid
        // per-iteration allocation.
        params.free_values_into(&mut free_vals_buf);
        let trial_free: Vec<f64> = free_vals_buf
            .iter()
            .zip(delta.iter())
            .map(|(&v, &d)| v + d)
            .collect();
        let param_change: f64 = delta
            .iter()
            .zip(free_vals_buf.iter())
            .map(|(&d, &v)| (d / (v.abs() + PIVOT_FLOOR)).powi(2))
            .sum::<f64>()
            .sqrt();
        params.set_free_values(&trial_free);

        params.all_values_into(&mut all_vals_buf);
        let y_trial = match model.evaluate(&all_vals_buf) {
            Ok(y) => y,
            Err(_) => {
                // Treat evaluation error as a bad step (same as NaN).
                params.set_free_values(&free_vals_buf);
                lambda *= config.lambda_up;
                if lambda > LAMBDA_BREAKOUT {
                    converged = false;
                    break;
                }
                continue;
            }
        };

        // #113: If the model produced NaN/Inf, treat as a bad step (same as
        // chi2 increase) — increase lambda and try again.
        if y_trial.iter().any(|v| !v.is_finite()) {
            params.set_free_values(&free_vals_buf);
            lambda *= config.lambda_up;
            if lambda > LAMBDA_BREAKOUT {
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
        let chi2_delta = (trial_chi2 - chi2).abs();
        let chi2_scale = chi2.abs().max(trial_chi2.abs()).max(1.0);
        let chi2_stagnated = chi2_delta <= config.tol_chi2 * chi2_scale;

        if trial_chi2 < chi2 {
            // Accept step — cache y_trial so the next iteration can skip
            // the base evaluate() inside compute_jacobian.
            let rel_change = (chi2 - trial_chi2) / (chi2 + PIVOT_FLOOR);
            chi2 = trial_chi2;
            residuals = trial_residuals;
            y_current = y_trial;
            lambda *= config.lambda_down;

            // Check convergence: relative chi2 change is tiny or parameters
            // stopped moving.  The old third condition
            // `chi2 < tol_chi2 * n_data` was scale-dependent and could cause
            // premature convergence on data with small residuals.  (#108.2)
            if rel_change < config.tol_chi2 || param_change < config.tol_param {
                converged = true;
                break;
            }
        } else {
            // Numerical stagnation: the strict LM acceptance test keeps
            // `trial_chi2 == chi2` in the reject path, but when both the
            // objective change and parameter step are tiny, the optimizer may
            // already be at a nonzero-χ² stationary point (noisy data,
            // correlated parameters, imperfect model). Report convergence if
            // the gradient is also tiny and the local model has real
            // curvature, rather than inflating lambda until breakout.
            let grad_tol = config.tol_chi2.sqrt().max(config.tol_param.sqrt());
            if chi2_stagnated
                && param_change < config.tol_param
                && scaled_grad_inf < grad_tol
                && informative_curvature
            {
                params.set_free_values(&free_vals_buf);
                converged = true;
                break;
            }

            // Reject step, restore parameters.
            // y_current stays valid (parameters reverted to free_vals_buf snapshot).
            params.set_free_values(&free_vals_buf);
            lambda *= config.lambda_up;

            // #108.4: If lambda is astronomically large, the optimizer is stuck
            // in a region where no step improves chi2.  Break out rather than
            // wasting iterations.
            if lambda > LAMBDA_BREAKOUT {
                converged = false;
                break;
            }
        }
    }

    let reduced_chi2 = if dof > 0 { chi2 / dof as f64 } else { f64::NAN };

    // Compute covariance matrix: (JᵀWJ)⁻¹ at the final parameters.
    //
    // This block requires an extra Jacobian evaluation + O(n_free³) matrix
    // inversion.  When `compute_covariance` is false (e.g. per-pixel spatial
    // mapping), we skip it entirely — the caller only needs densities and
    // chi-squared, not uncertainties.
    let (covariance, uncertainties) = if converged && config.compute_covariance {
        let jacobian = compute_jacobian(
            model,
            params,
            &y_current,
            config.fd_step,
            &mut all_vals_buf,
            &mut free_idx_buf,
        )?;
        let mut jtw_j = FlatMatrix::zeros(n_free, n_free);
        for (i, &wi) in weights.iter().enumerate() {
            for j in 0..n_free {
                let jij = jacobian.get(i, j);
                for k in 0..n_free {
                    *jtw_j.get_mut(j, k) += jij * wi * jacobian.get(i, k);
                }
            }
        }

        // #108.1: Scale covariance by reduced chi-squared.
        //
        // The raw (JᵀWJ)⁻¹ gives the covariance only when the model is a perfect
        // description and the weights are exact.  Multiplying by χ²/ν accounts for
        // misfit (model inadequacy or underestimated errors).  This is the standard
        // statistical prescription (see e.g. Numerical Recipes §15.6).
        //
        // When dof == 0 (exactly determined system), reduced chi-squared is
        // undefined (0/0).  We report NaN and skip covariance scaling entirely,
        // returning None for covariance and uncertainties.
        if dof > 0 {
            if let Some(mut cov) = invert_matrix(&jtw_j) {
                for elem in cov.data.iter_mut() {
                    *elem *= reduced_chi2;
                }
                let unc: Vec<f64> = (0..n_free)
                    .map(|i| {
                        let diag = cov.get(i, i);
                        if diag.is_finite() && diag > 0.0 {
                            diag.sqrt()
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                (Some(cov), Some(unc))
            } else {
                (None, None)
            }
        } else {
            // dof == 0: covariance scaling is undefined; report None.
            (None, None)
        }
    } else {
        // Covariance computation skipped (compute_covariance == false).
        (None, None)
    };

    Ok(LmResult {
        chi_squared: chi2,
        reduced_chi_squared: reduced_chi2,
        iterations: iter,
        converged,
        params: params.all_values(),
        covariance,
        uncertainties,
    })
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
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            let a = params[0];
            let b = params[1];
            Ok(self.x.iter().map(|&x| a * x + b).collect())
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

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

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
    fn test_converges_on_exact_flat_bottom_without_lambda_breakout() {
        // Exact data with zero initial damping reaches the optimum in one
        // Newton step. The next iteration sits on a flat χ² floor where the
        // strict `trial_chi2 < chi2` check must not force a false non-
        // convergence.
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let sigma = vec![1.0; 10];

        let model = LinearModel { x };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 0.0),
            FitParameter::unbounded("b", 0.0),
        ]);
        let config = LmConfig {
            lambda_init: 0.0,
            ..LmConfig::default()
        };

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &config).unwrap();

        assert!(
            result.converged,
            "Fit should converge on an exact flat bottom"
        );
        assert!(result.chi_squared < 1e-20, "chi2 = {}", result.chi_squared);
        assert!(
            result.iterations < config.max_iter,
            "LM should stop by convergence, not by iteration exhaustion"
        );
    }

    #[test]
    fn test_converges_on_nonzero_chi2_stationary_point() {
        // Noisy overdetermined data have a nonzero-chi2 optimum. With very
        // strict tolerances, the accepted step to the optimum does not trip
        // the accept-branch convergence checks, so the next iteration reaches
        // a reject-path stationary point with trial_chi2 == chi2.
        struct AffineModel {
            x: Vec<f64>,
        }
        impl FitModel for AffineModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let a = params[0];
                let b = params[1];
                Ok(self.x.iter().map(|&x| a * x + b).collect())
            }
        }

        let model = AffineModel {
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        };
        let y_obs = vec![0.1, 0.9, 2.2, 2.8, 4.1];
        let sigma = vec![1.0; y_obs.len()];
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 0.0),
            FitParameter::unbounded("b", 0.0),
        ]);
        let config = LmConfig {
            max_iter: 200,
            tol_chi2: 1e-16,
            tol_param: 1e-16,
            ..LmConfig::default()
        };

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &config).unwrap();

        assert!(
            result.converged,
            "stationary nonzero-chi2 optimum should converge instead of lambda breakout"
        );
        assert!(
            result.reduced_chi_squared.is_finite() && result.reduced_chi_squared > 0.0,
            "expected nonzero reduced chi2 at noisy optimum, got {}",
            result.reduced_chi_squared
        );
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

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

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
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let (a, b, c) = (params[0], params[1], params[2]);
                Ok(self.x.iter().map(|&x| a * x * x + b * x + c).collect())
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

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

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
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let a = params[0];
                Ok(self.x.iter().map(|&x| a * x).collect())
            }
        }

        let x: Vec<f64> = (1..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| -2.0 * xi).collect();
        let sigma = vec![1.0; 9];

        let model = SlopeModel { x };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("a", 1.0)]);

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

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

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

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
        let a = FlatMatrix {
            data: vec![1.0, 0.0, 0.0, 1.0],
            nrows: 2,
            ncols: 2,
        };
        let b = vec![2.0, 4.0];
        let lambda = 1.0;
        let x = solve_damped_system(&a, &b, lambda).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_matrix_2x2() {
        let a = FlatMatrix {
            data: vec![4.0, 7.0, 2.0, 6.0],
            nrows: 2,
            ncols: 2,
        };
        let inv = invert_matrix(&a).unwrap();
        // A⁻¹ = 1/10 × [6 -7; -2 4]
        assert!((inv.get(0, 0) - 0.6).abs() < 1e-10);
        assert!((inv.get(0, 1) - (-0.7)).abs() < 1e-10);
        assert!((inv.get(1, 0) - (-0.2)).abs() < 1e-10);
        assert!((inv.get(1, 1) - 0.4).abs() < 1e-10);
    }

    // ---- Edge-case tests for issue #125 ----

    #[test]
    fn test_all_fixed_params_nan_model() {
        // #125.1: When all parameters are fixed and the model produces NaN,
        // the result must report converged=false (not converged=true with NaN chi2).
        struct NanModel;
        impl FitModel for NanModel {
            fn evaluate(&self, _params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![f64::NAN; 5])
            }
        }

        let y_obs = vec![1.0; 5];
        let sigma = vec![1.0; 5];
        let mut params = ParameterSet::new(vec![FitParameter::fixed("a", 1.0)]);

        let result =
            levenberg_marquardt(&NanModel, &y_obs, &sigma, &mut params, &LmConfig::default())
                .unwrap();

        assert!(!result.converged, "All-fixed NaN model should not converge");
        assert!(result.chi_squared.is_nan(), "chi2 should be NaN");
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_underdetermined_system() {
        // #125.6: More free parameters than data points → underdetermined.
        // Should return converged=false immediately.
        let y_obs = vec![1.0, 2.0]; // 2 data points
        let sigma = vec![1.0, 1.0];

        // 2 free params for 2 data points is exactly determined (ok),
        // but 3 free params for 2 data points is underdetermined.
        struct ThreeParamModel;
        impl FitModel for ThreeParamModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![params[0] + params[1] + params[2]; 2])
            }
        }

        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0),
            FitParameter::unbounded("b", 1.0),
            FitParameter::unbounded("c", 1.0),
        ]);

        let result = levenberg_marquardt(
            &ThreeParamModel,
            &y_obs,
            &sigma,
            &mut params,
            &LmConfig::default(),
        )
        .unwrap();

        assert!(
            !result.converged,
            "Underdetermined system should not converge"
        );
        assert!(result.chi_squared.is_nan());
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_exactly_determined_dof_zero() {
        // #125.6: n_data == n_free → dof=0, exactly determined.
        // Should still converge but reduced_chi_squared is NaN (0/0).
        let y_obs = vec![5.0, 11.0]; // y = 2x + 3 at x=1,4
        let sigma = vec![1.0, 1.0];

        let model = LinearModel { x: vec![1.0, 4.0] };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0),
            FitParameter::unbounded("b", 1.0),
        ]);

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

        assert!(
            result.converged,
            "Exactly-determined system should converge"
        );
        assert!(
            result.chi_squared < 1e-6,
            "chi2 should be ~0, got {}",
            result.chi_squared
        );
        // dof=0 → reduced chi2 is NaN
        assert!(
            result.reduced_chi_squared.is_nan(),
            "reduced_chi2 should be NaN for dof=0, got {}",
            result.reduced_chi_squared
        );
        // No covariance when dof=0
        assert!(result.covariance.is_none());
        assert!(result.uncertainties.is_none());
    }

    #[test]
    fn test_lambda_breakout() {
        // #125.6: A model that never improves should trigger lambda breakout.
        struct ConstantModel;
        impl FitModel for ConstantModel {
            fn evaluate(&self, _params: &[f64]) -> Result<Vec<f64>, FittingError> {
                // Returns constant output regardless of parameters,
                // so the Jacobian is zero and no step can improve chi2.
                Ok(vec![42.0; 5])
            }
        }

        let y_obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigma = vec![1.0; 5];
        let mut params = ParameterSet::new(vec![FitParameter::unbounded("a", 1.0)]);

        let config = LmConfig {
            max_iter: 1000,
            ..LmConfig::default()
        };

        let result =
            levenberg_marquardt(&ConstantModel, &y_obs, &sigma, &mut params, &config).unwrap();

        assert!(
            !result.converged,
            "Flat model should not converge (lambda breakout)"
        );
        assert!(
            result.covariance.is_none(),
            "unconverged fit should not report covariance"
        );
        assert!(
            result.uncertainties.is_none(),
            "unconverged fit should not report uncertainties"
        );
    }

    #[test]
    fn test_nan_model_during_iteration() {
        // #125.6: Model that produces NaN for certain parameter values.
        // The optimizer should treat NaN steps as bad and try smaller steps.
        struct NanAtLargeModel {
            x: Vec<f64>,
        }
        impl FitModel for NanAtLargeModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let a = params[0];
                Ok(self
                    .x
                    .iter()
                    .map(|&x| if a > 5.0 { f64::NAN } else { a * x + 1.0 })
                    .collect())
            }
        }

        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let sigma = vec![1.0; 10];

        let model = NanAtLargeModel { x };
        let mut params = ParameterSet::new(vec![FitParameter::unbounded("a", 3.0)]);

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

        // Should converge to a≈2 while avoiding the NaN region a>5.
        assert!(result.converged, "Should converge avoiding NaN region");
        assert!(
            (result.params[0] - 2.0).abs() < 0.1,
            "a = {}, expected ~2.0",
            result.params[0]
        );
    }

    #[test]
    fn test_err_model_during_trial_step() {
        // Model that returns Err for large parameter values.
        // The optimizer should treat Err trial steps as bad steps (increase λ)
        // and converge without panicking.
        struct ErrAtLargeModel {
            x: Vec<f64>,
        }
        impl FitModel for ErrAtLargeModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let a = params[0];
                if a > 5.0 {
                    return Err(FittingError::EvaluationFailed(
                        "parameter out of valid range".into(),
                    ));
                }
                Ok(self.x.iter().map(|&x| a * x + 1.0).collect())
            }
        }

        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let sigma = vec![1.0; 10];

        let model = ErrAtLargeModel { x };
        let mut params = ParameterSet::new(vec![FitParameter::unbounded("a", 3.0)]);

        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

        // Should converge to a≈2 while avoiding the Err region a>5.
        assert!(result.converged, "Should converge avoiding Err region");
        assert!(
            (result.params[0] - 2.0).abs() < 0.1,
            "a = {}, expected ~2.0",
            result.params[0]
        );
    }

    #[test]
    fn test_fit_linear_no_covariance() {
        // When compute_covariance is false, the fit should still converge and
        // produce correct parameters, but covariance and uncertainties are None.
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_obs: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let sigma = vec![1.0; 10];

        let model = LinearModel { x };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0),
            FitParameter::unbounded("b", 1.0),
        ]);

        let config = LmConfig {
            compute_covariance: false,
            ..LmConfig::default()
        };

        let result = levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &config).unwrap();

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
        assert!(
            result.covariance.is_none(),
            "covariance should be None when compute_covariance=false"
        );
        assert!(
            result.uncertainties.is_none(),
            "uncertainties should be None when compute_covariance=false"
        );
    }

    #[test]
    fn test_zero_negative_sigma_clamping() {
        // #125.6: Zero and negative sigma should be clamped to huge sigma (tiny weight),
        // not cause NaN/panic.
        let y_obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigma = vec![0.0, -1.0, f64::NAN, f64::INFINITY, 1.0];

        let model = LinearModel {
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        };
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("a", 1.0),
            FitParameter::unbounded("b", 0.0),
        ]);

        // Should not panic and should produce a finite result.
        let result =
            levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default()).unwrap();

        assert!(
            result.chi_squared.is_finite(),
            "chi2 should be finite despite bad sigma, got {}",
            result.chi_squared
        );
        assert!(
            result.converged,
            "Fit should converge despite bad sigma values"
        );
        // The only valid data point with sigma=1.0 is (x=4, y=5).
        // The fitted line y = a*x + b should pass near that point.
        let y_at_4 = result.params[0] * 4.0 + result.params[1];
        assert!(
            (y_at_4 - 5.0).abs() < 1.0,
            "Fitted line should pass near (4, 5): a={}, b={}, y(4)={}",
            result.params[0],
            result.params[1],
            y_at_4,
        );
    }
}
