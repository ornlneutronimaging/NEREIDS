//! Poisson-likelihood optimizer for low-count neutron data.
//!
//! When neutron counts are low (< ~30 per bin), the Gaussian/chi-squared
//! assumption breaks down and Poisson statistics must be used directly.
//!
//! ## Negative log-likelihood
//!
//! L(θ) = Σᵢ [y_model(θ)ᵢ - y_obs,ᵢ · ln(y_model(θ)ᵢ)]
//!
//! This is minimized using a projected optimizer:
//! - damped Gauss-Newton / Fisher steps when an analytical Jacobian exists
//! - finite-difference projected gradient fallback otherwise
//!
//! Both paths use backtracking line search and bound constraints
//! (non-negativity on densities).
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` — Poisson NLL and APGM optimizer

use nereids_core::constants::{PIVOT_FLOOR, POISSON_EPSILON};

use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};
use crate::parameters::{FitParameter, ParameterSet};

/// Configuration for the Poisson optimizer.
#[derive(Debug, Clone)]
pub struct PoissonConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Step size for finite-difference gradient.
    pub fd_step: f64,
    /// Initial step size for line search.
    pub step_size: f64,
    /// Convergence tolerance used for both parameter displacement (L2 norm of step)
    /// and gradient-norm convergence checks in `poisson_fit`.
    pub tol_param: f64,
    /// Armijo line search parameter (sufficient decrease).
    pub armijo_c: f64,
    /// Line search backtracking factor.
    pub backtrack: f64,
    /// Relative diagonal damping for analytical Gauss-Newton / Fisher steps.
    pub gauss_newton_lambda: f64,
    /// History size for the finite-difference L-BFGS fallback.
    pub lbfgs_history: usize,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            fd_step: 1e-7,
            step_size: 1.0,
            tol_param: 1e-8,
            armijo_c: 1e-4,
            backtrack: 0.5,
            gauss_newton_lambda: 1e-3,
            lbfgs_history: 8,
        }
    }
}

/// Result of Poisson-likelihood optimization.
#[derive(Debug, Clone)]
pub struct PoissonResult {
    /// Final negative log-likelihood.
    pub nll: f64,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Final parameter values (all parameters, including fixed).
    pub params: Vec<f64>,
}

/// Compute Poisson negative log-likelihood.
///
/// NLL = Σᵢ [y_model - y_obs · ln(y_model)]
///
/// #109.2: For y_model ≤ epsilon, use a smooth C¹ quadratic extrapolation
/// instead of a hard 1e30 penalty.  This keeps the NLL and its gradient
/// continuous, so gradient-based optimizers (projected gradient, L-BFGS)
/// can smoothly steer back into the feasible region rather than hitting
/// a discontinuous cliff that stalls the line search.
fn poisson_nll(y_obs: &[f64], y_model: &[f64]) -> f64 {
    y_obs
        .iter()
        .zip(y_model.iter())
        .map(|(&obs, &mdl)| poisson_nll_term(obs, mdl))
        .sum()
}

/// Single-bin Poisson NLL with smooth extrapolation for mdl <= epsilon.
///
/// For mdl > 0: NLL = mdl - obs * ln(mdl)
/// For mdl <= epsilon: quadratic Taylor expansion about epsilon,
///   NLL(ε) + NLL'(ε)·(mdl−ε) + ½·NLL''(ε)·(mdl−ε)²
/// where NLL'(x) = 1 − obs/x and NLL''(x) = obs/x².
///
/// Since delta = ε − mdl ≥ 0, this becomes:
///   NLL(ε) − NLL'(ε)·delta + ½·NLL''(ε)·delta²
///
/// When obs == 0, the exact Hessian obs/ε² vanishes, leaving only a linear
/// term that decreases without bound as mdl → −∞.  This can cause the
/// optimizer to diverge.  We impose a minimum curvature of 1/ε so the
/// quadratic penalty still curves upward for negative predictions.
#[inline]
fn poisson_nll_term(obs: f64, mdl: f64) -> f64 {
    // #125.3: Negative observed counts would produce wrong-signed NLL terms.
    // Release builds skip this check; callers must ensure non-negative counts. See #125 item 3.
    debug_assert!(
        obs.is_finite() && obs >= 0.0,
        "poisson_nll_term: obs must be finite and >= 0, got {obs}"
    );
    if mdl > POISSON_EPSILON {
        mdl - obs * mdl.ln()
    } else {
        let eps = POISSON_EPSILON;
        let nll_eps = eps - obs * eps.ln();
        let grad_eps = 1.0 - obs / eps;
        // Minimum curvature 1/eps ensures the penalty grows quadratically
        // even when obs == 0 (where the exact Hessian obs/eps^2 vanishes).
        let hess_eps = if obs > 0.0 {
            obs / (eps * eps)
        } else {
            1.0 / eps
        };
        let delta = eps - mdl;
        // Taylor expansion: f(eps) + f'(eps)*(mdl - eps) + 0.5*f''(eps)*(mdl - eps)^2
        // Since (mdl - eps) = -delta, the linear term flips sign.
        nll_eps - grad_eps * delta + 0.5 * hess_eps * delta * delta
    }
}

/// Per-bin Poisson NLL weight: ∂f(obs, mdl)/∂mdl.
///
/// For mdl > ε: w = 1 - obs/mdl
/// For mdl ≤ ε: derivative of the smooth quadratic extrapolation,
///   w = grad_eps - hess_eps · (ε - mdl), continuous at boundary.
#[inline]
fn poisson_nll_weight(obs: f64, mdl: f64) -> f64 {
    if mdl > POISSON_EPSILON {
        1.0 - obs / mdl
    } else {
        let eps = POISSON_EPSILON;
        let grad_eps = 1.0 - obs / eps;
        let hess_eps = if obs > 0.0 {
            obs / (eps * eps)
        } else {
            1.0 / eps
        };
        grad_eps - hess_eps * (eps - mdl)
    }
}

/// Per-bin Poisson NLL curvature: ∂²f(obs, mdl)/∂mdl².
///
/// For mdl > ε: h = obs / mdl²
/// For mdl ≤ ε: curvature of the smooth quadratic extrapolation.
#[inline]
fn poisson_nll_curvature(obs: f64, mdl: f64) -> f64 {
    if mdl > POISSON_EPSILON {
        obs / (mdl * mdl)
    } else {
        let eps = POISSON_EPSILON;
        if obs > 0.0 {
            obs / (eps * eps)
        } else {
            1.0 / eps
        }
    }
}

/// Analytical first/second-order information for the Poisson objective.
#[derive(Debug)]
struct AnalyticalStepData {
    /// Gradient of the Poisson NLL: grad = J^T · w.
    grad: Vec<f64>,
    /// Full Gauss-Newton / Fisher curvature approximation: J^T H J.
    fisher: FlatMatrix,
}

/// Compute gradient and Gauss-Newton / Fisher curvature of the Poisson NLL
/// using the analytical Jacobian.
///
/// `grad_j = Σᵢ wᵢ · J_{i,j}` where `wᵢ = ∂NLL/∂y_model_i`
/// and `J_{i,j} = ∂y_model_i/∂θⱼ` from `model.analytical_jacobian()`.
///
/// The curvature uses the Poisson Hessian with respect to the model output:
/// `fisher_{j,k} = Σᵢ hᵢ · J_{i,j} · J_{i,k}` where
/// `hᵢ = ∂²NLL/∂y_model_i²`.
///
/// Returns `Some(step_data)` if the model provides an analytical Jacobian,
/// `None` otherwise (caller should fall back to finite differences).
fn compute_analytical_step_data(
    model: &dyn FitModel,
    params: &ParameterSet,
    y_obs: &[f64],
    y_model: &[f64],
    all_vals_buf: &mut Vec<f64>,
    free_idx_buf: &mut Vec<usize>,
) -> Option<AnalyticalStepData> {
    params.all_values_into(all_vals_buf);
    params.free_indices_into(free_idx_buf);
    let jac = model.analytical_jacobian(all_vals_buf, free_idx_buf, y_model)?;
    let n_e = y_obs.len();
    let n_free = free_idx_buf.len();
    let mut grad = vec![0.0f64; n_free];
    let mut fisher = FlatMatrix::zeros(n_free, n_free);
    for i in 0..n_e {
        let w = poisson_nll_weight(y_obs[i], y_model[i]);
        let h = poisson_nll_curvature(y_obs[i], y_model[i]);
        for (g, j) in grad.iter_mut().zip(0..n_free) {
            let jij = jac.get(i, j);
            *g += w * jij;
            for k in 0..n_free {
                *fisher.get_mut(j, k) += h * jij * jac.get(i, k);
            }
        }
    }
    Some(AnalyticalStepData { grad, fisher })
}

/// Compute gradient of Poisson NLL by finite differences.
///
/// `all_vals_buf` is a reusable scratch buffer for `params.all_values_into()`,
/// avoiding a fresh allocation on every `model.evaluate()` call inside the
/// per-parameter FD loop (N_free+1 allocations saved per gradient call).
///
/// `free_idx_buf` is a scratch buffer for `params.free_indices_into()`, reused
/// across iterations to avoid per-gradient allocation.
fn compute_gradient(
    model: &dyn FitModel,
    params: &mut ParameterSet,
    y_obs: &[f64],
    fd_step: f64,
    all_vals_buf: &mut Vec<f64>,
    free_idx_buf: &mut Vec<usize>,
) -> Result<Vec<f64>, FittingError> {
    params.all_values_into(all_vals_buf);
    let base_model = model.evaluate(all_vals_buf)?;
    let base_nll = poisson_nll(y_obs, &base_model);

    params.free_indices_into(free_idx_buf);
    let mut grad = vec![0.0; free_idx_buf.len()];

    for (j, &idx) in free_idx_buf.iter().enumerate() {
        let original = params.params[idx].value;
        let step = fd_step * (1.0 + original.abs());

        params.params[idx].value = original + step;
        params.params[idx].clamp();
        let mut actual_step = params.params[idx].value - original;

        // #112: If the forward step is blocked by an upper bound, try the
        // backward step so the gradient component is not frozen at zero.
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
        let perturbed_model = match model.evaluate(all_vals_buf) {
            Ok(v) => v,
            Err(_) => {
                params.params[idx].value = original;
                continue;
            }
        };
        let perturbed_nll = poisson_nll(y_obs, &perturbed_model);
        params.params[idx].value = original;

        grad[j] = (perturbed_nll - base_nll) / actual_step;
    }

    Ok(grad)
}

fn normalized_step_norm(
    old_free: &[f64],
    new_free: &[f64],
    params: &ParameterSet,
    free_param_indices: &[usize],
) -> f64 {
    old_free
        .iter()
        .zip(new_free.iter())
        .zip(free_param_indices.iter())
        .map(|((&old, &new), &idx)| {
            let range = params.params[idx].upper - params.params[idx].lower;
            let scale = if range.is_finite() && range > 1e-10 {
                range
            } else {
                old.abs().max(1e-3)
            };
            ((old - new) / scale).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

fn is_bound_active(param: &FitParameter, grad: f64) -> bool {
    let at_lower = param.lower.is_finite() && (param.value - param.lower).abs() <= PIVOT_FLOOR;
    let at_upper = param.upper.is_finite() && (param.value - param.upper).abs() <= PIVOT_FLOOR;
    (at_lower && grad > 0.0) || (at_upper && grad < 0.0)
}

fn inactive_free_positions(
    params: &ParameterSet,
    free_param_indices: &[usize],
    grad: &[f64],
) -> Vec<usize> {
    free_param_indices
        .iter()
        .zip(grad.iter())
        .enumerate()
        .filter_map(|(pos, (&idx, &g))| (!is_bound_active(&params.params[idx], g)).then_some(pos))
        .collect()
}

fn inactive_free_mask(
    params: &ParameterSet,
    free_param_indices: &[usize],
    grad: &[f64],
) -> Vec<bool> {
    free_param_indices
        .iter()
        .zip(grad.iter())
        .map(|(&idx, &g)| !is_bound_active(&params.params[idx], g))
        .collect()
}

fn projected_gradient_norm(
    params: &ParameterSet,
    free_param_indices: &[usize],
    grad: &[f64],
) -> f64 {
    free_param_indices
        .iter()
        .zip(grad.iter())
        .map(|(&idx, &g)| {
            if is_bound_active(&params.params[idx], g) {
                0.0
            } else {
                g * g
            }
        })
        .sum::<f64>()
        .sqrt()
}

fn extract_submatrix(matrix: &FlatMatrix, positions: &[usize]) -> FlatMatrix {
    let n = positions.len();
    let mut sub = FlatMatrix::zeros(n, n);
    for (row_out, &row_in) in positions.iter().enumerate() {
        for (col_out, &col_in) in positions.iter().enumerate() {
            *sub.get_mut(row_out, col_out) = matrix.get(row_in, col_in);
        }
    }
    sub
}

#[derive(Debug, Clone)]
struct LbfgsHistory {
    s_list: Vec<Vec<f64>>,
    y_list: Vec<Vec<f64>>,
    max_pairs: usize,
}

impl LbfgsHistory {
    fn new(max_pairs: usize) -> Self {
        Self {
            s_list: Vec::with_capacity(max_pairs),
            y_list: Vec::with_capacity(max_pairs),
            max_pairs,
        }
    }

    fn clear(&mut self) {
        self.s_list.clear();
        self.y_list.clear();
    }

    fn update(&mut self, old_free: &[f64], new_free: &[f64], old_grad: &[f64], new_grad: &[f64]) {
        if self.max_pairs == 0 {
            return;
        }
        let s: Vec<f64> = new_free
            .iter()
            .zip(old_free.iter())
            .map(|(&new, &old)| new - old)
            .collect();
        let y: Vec<f64> = new_grad
            .iter()
            .zip(old_grad.iter())
            .map(|(&new, &old)| new - old)
            .collect();
        let sy = dot(&s, &y);
        let s_norm = dot(&s, &s).sqrt();
        let y_norm = dot(&y, &y).sqrt();
        if sy <= 1e-12 * s_norm * y_norm.max(1.0) {
            return;
        }
        if self.s_list.len() == self.max_pairs {
            self.s_list.remove(0);
            self.y_list.remove(0);
        }
        self.s_list.push(s);
        self.y_list.push(y);
    }

    fn apply_on_positions(&self, grad: &[f64], positions: &[usize]) -> Option<Vec<f64>> {
        if self.s_list.is_empty() || positions.is_empty() {
            return None;
        }

        let mut q: Vec<f64> = positions.iter().map(|&pos| grad[pos]).collect();
        let mut alpha = vec![0.0; self.s_list.len()];
        let mut rho = vec![0.0; self.s_list.len()];
        let mut used = vec![false; self.s_list.len()];

        for i in (0..self.s_list.len()).rev() {
            let s_sub = extract_positions(&self.s_list[i], positions);
            let y_sub = extract_positions(&self.y_list[i], positions);
            let sy = dot(&s_sub, &y_sub);
            if sy <= 1e-12 {
                continue;
            }
            rho[i] = 1.0 / sy;
            used[i] = true;
            alpha[i] = rho[i] * dot(&s_sub, &q);
            axpy(&mut q, -alpha[i], &y_sub);
        }

        let gamma = used
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, &is_used)| {
                if !is_used {
                    return None;
                }
                let last_s = extract_positions(&self.s_list[i], positions);
                let last_y = extract_positions(&self.y_list[i], positions);
                let yy = dot(&last_y, &last_y);
                (yy > 0.0).then_some(dot(&last_s, &last_y) / yy)
            })
            .unwrap_or(1.0);

        let mut r: Vec<f64> = q.into_iter().map(|v| gamma * v).collect();
        for i in 0..self.s_list.len() {
            if !used[i] {
                continue;
            }
            let s_sub = extract_positions(&self.s_list[i], positions);
            let y_sub = extract_positions(&self.y_list[i], positions);
            let beta = rho[i] * dot(&y_sub, &r);
            axpy(&mut r, alpha[i] - beta, &s_sub);
        }

        let mut full = vec![0.0; grad.len()];
        for (&pos, &value) in positions.iter().zip(r.iter()) {
            full[pos] = value;
        }
        Some(full)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn axpy(dst: &mut [f64], alpha: f64, src: &[f64]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += alpha * s;
    }
}

fn extract_positions(values: &[f64], positions: &[usize]) -> Vec<f64> {
    positions.iter().map(|&pos| values[pos]).collect()
}

fn parameter_scaled_gradient_direction(
    params: &ParameterSet,
    free_param_indices: &[usize],
    grad: &[f64],
) -> Vec<f64> {
    grad.iter()
        .zip(free_param_indices.iter())
        .map(|(&g, &idx)| {
            if is_bound_active(&params.params[idx], g) {
                return 0.0;
            }
            let p = &params.params[idx];
            let range = p.upper - p.lower;
            if range.is_finite() && range > 1e-10 {
                g * range * range
            } else {
                let scale = p.value.abs().max(1e-3);
                g * scale * scale
            }
        })
        .collect()
}

fn max_feasible_step(
    params: &ParameterSet,
    free_param_indices: &[usize],
    old_free: &[f64],
    search_dir: &[f64],
) -> f64 {
    let mut alpha_max = f64::INFINITY;
    for ((&idx, &x), &d) in free_param_indices
        .iter()
        .zip(old_free.iter())
        .zip(search_dir.iter())
    {
        if d.abs() <= PIVOT_FLOOR {
            continue;
        }
        let p = &params.params[idx];
        let candidate = if d > 0.0 && p.lower.is_finite() {
            (x - p.lower) / d
        } else if d < 0.0 && p.upper.is_finite() {
            (p.upper - x) / (-d)
        } else {
            f64::INFINITY
        };
        alpha_max = alpha_max.min(candidate);
    }
    alpha_max.max(0.0)
}

enum LineSearchResult {
    Accepted {
        nll: f64,
        y_model: Vec<f64>,
        hit_boundary: bool,
    },
    Stagnated,
    Failed,
}

const MAX_FACE_STEPS_PER_ITER: usize = 4;

/// Backtracking line search with Armijo condition.
///
/// Backtracking line search with Armijo sufficient-decrease condition:
/// try a step, reject NaN/Inf model outputs, check the Armijo sufficient-decrease
/// condition, and backtrack if needed.
///
/// # Arguments
/// * `model`        — Forward model (maps parameters -> predicted counts).
/// * `params`       — Parameter set (modified in place on success).
/// * `y_obs`        — Observed counts.
/// * `old_free`     — Free parameter values before the step.
/// * `search_dir`   — Search direction (gradient or preconditioned gradient).
/// * `initial_alpha`— Initial step size.
/// * `config`       — Optimizer configuration (backtrack factor, Armijo c).
/// * `grad`         — Raw gradient (used for Armijo descent computation).
/// * `nll`          — Current negative log-likelihood.
/// * `all_vals_buf` — Scratch buffer for `params.all_values_into()`, reused
///   across the up-to-50 backtracking iterations to avoid per-trial allocation.
/// * `free_vals_buf`— Scratch buffer for `params.free_values_into()`.
/// * `trial_free_buf` — Scratch buffer for the trial free-parameter vector,
///   reused across backtracking iterations to avoid up to 50 allocations.
///
/// # Returns
/// `Some((new_nll, y_model))` if a step was accepted, `None` if the line search
/// exhausted all backtracking attempts. Returns the model output alongside NLL
/// so the caller can cache it for the next analytical gradient computation.
///
/// # Failure contract
///
/// On `None` return (line search exhausted), `params` is restored to
/// `old_free` before returning. Callers need not restore manually.
// All 12 arguments are genuinely needed: 9 original + 3 scratch buffers that
// avoid per-backtracking-iteration allocations inside the 50-trial loop.
#[allow(clippy::too_many_arguments)]
fn backtracking_line_search(
    model: &dyn FitModel,
    params: &mut ParameterSet,
    y_obs: &[f64],
    old_free: &[f64],
    free_param_indices: &[usize],
    search_dir: &[f64],
    initial_alpha: f64,
    config: &PoissonConfig,
    grad: &[f64],
    nll: f64,
    all_vals_buf: &mut Vec<f64>,
    free_vals_buf: &mut Vec<f64>,
    trial_free_buf: &mut Vec<f64>,
) -> LineSearchResult {
    let alpha_max = max_feasible_step(params, free_param_indices, old_free, search_dir);
    if alpha_max <= PIVOT_FLOOR {
        params.set_free_values(old_free);
        return LineSearchResult::Stagnated;
    }
    let mut alpha = initial_alpha.min(alpha_max);
    for _ in 0..50 {
        // Trial step along the feasible path: x_new = x - alpha * d, with
        // alpha capped so inactive-subspace directions hit bounds exactly
        // instead of relying on projection to distort the step.
        trial_free_buf.clear();
        for ((&idx, &v), &d) in free_param_indices
            .iter()
            .zip(old_free.iter())
            .zip(search_dir.iter())
        {
            let p = &params.params[idx];
            trial_free_buf.push((v - alpha * d).clamp(p.lower, p.upper));
        }
        params.set_free_values(trial_free_buf);

        params.all_values_into(all_vals_buf);
        let trial_model = match model.evaluate(all_vals_buf) {
            Ok(v) => v,
            Err(_) => {
                alpha *= config.backtrack;
                continue;
            }
        };

        // #113: If the model produced NaN/Inf, reduce step size rather
        // than accepting a garbage NLL.
        if trial_model.iter().any(|v| !v.is_finite()) {
            alpha *= config.backtrack;
            continue;
        }

        let trial_nll = poisson_nll(y_obs, &trial_model);

        // Armijo condition: f(x_new) <= f(x) - c * descent
        params.free_values_into(free_vals_buf);
        let step_norm = normalized_step_norm(old_free, free_vals_buf, params, free_param_indices);
        let descent = grad
            .iter()
            .zip(old_free.iter())
            .zip(free_vals_buf.iter())
            .map(|((&g, &old), &new)| g * (old - new))
            .sum::<f64>();

        if trial_nll.is_finite() && trial_nll <= nll - config.armijo_c * descent {
            return LineSearchResult::Accepted {
                nll: trial_nll,
                y_model: trial_model,
                hit_boundary: alpha_max.is_finite()
                    && (alpha_max - alpha).abs() <= 1e-12 * alpha_max.max(1.0),
            };
        }

        let nll_delta = (trial_nll - nll).abs();
        let nll_scale = trial_nll.abs().max(nll.abs()).max(1.0);
        if trial_nll.is_finite()
            && step_norm < config.tol_param
            && nll_delta <= config.tol_param * nll_scale
        {
            params.set_free_values(old_free);
            return LineSearchResult::Stagnated;
        }

        // Backtrack
        alpha *= config.backtrack;
        if alpha <= PIVOT_FLOOR {
            break;
        }
    }
    params.set_free_values(old_free);
    LineSearchResult::Failed
}

/// Early return for all-fixed parameters: evaluate once and report.
///
/// Returns `Ok(Some(PoissonResult))` if all parameters are fixed (either a
/// valid result with converged=true, or a non-finite NLL with converged=false).
/// Returns `Ok(None)` if there are free parameters and optimization should
/// proceed. Returns `Err(FittingError)` if model evaluation fails.
fn try_early_return_fixed(
    model: &dyn FitModel,
    y_obs: &[f64],
    params: &ParameterSet,
) -> Result<Option<PoissonResult>, FittingError> {
    if params.n_free() != 0 {
        return Ok(None);
    }
    let y_model = model.evaluate(&params.all_values())?;
    let nll = poisson_nll(y_obs, &y_model);
    if !nll.is_finite() {
        return Ok(Some(PoissonResult {
            nll,
            iterations: 0,
            converged: false,
            params: params.all_values(),
        }));
    }
    Ok(Some(PoissonResult {
        nll,
        iterations: 0,
        converged: true,
        params: params.all_values(),
    }))
}

/// Run Poisson-likelihood optimization using a projected KL optimizer.
///
/// Uses damped Gauss-Newton / Fisher steps when an analytical Jacobian is
/// available, falling back to projected gradient descent otherwise. Both paths
/// use backtracking line search with Armijo condition and projection onto
/// parameter bounds after each step.
///
/// # Arguments
/// * `model` — Forward model (maps parameters → predicted counts).
/// * `y_obs` — Observed counts at each data point.
/// * `params` — Parameter set (modified in place).
/// * `config` — Optimizer configuration.
///
/// # Returns
/// `Ok(PoissonResult)` with final NLL, parameters, and convergence status.
/// `Err(FittingError)` if model evaluation fails at the initial point.
/// Evaluation errors during line-search trials are treated as bad steps
/// (backtrack), not fatal errors.
pub fn poisson_fit(
    model: &dyn FitModel,
    y_obs: &[f64],
    params: &mut ParameterSet,
    config: &PoissonConfig,
) -> Result<PoissonResult, FittingError> {
    if let Some(result) = try_early_return_fixed(model, y_obs, params)? {
        return Ok(result);
    }

    // Scratch buffers reused across the entire optimization loop to avoid
    // per-iteration allocations inside compute_gradient (N_free+1 calls)
    // and backtracking_line_search (up to 50 calls).
    let mut all_vals_buf = Vec::with_capacity(params.params.len());
    let mut free_vals_buf = Vec::with_capacity(params.n_free());
    let mut old_free_buf: Vec<f64> = Vec::with_capacity(params.n_free());
    let mut trial_free_buf: Vec<f64> = Vec::with_capacity(params.n_free());
    let mut free_idx_buf: Vec<usize> = Vec::with_capacity(params.n_free());
    let mut fd_history = LbfgsHistory::new(config.lbfgs_history);
    let mut pending_fd_state: Option<(Vec<f64>, Vec<f64>, Vec<bool>)> = None;

    params.all_values_into(&mut all_vals_buf);
    let mut y_model = model.evaluate(&all_vals_buf)?;
    let mut nll = poisson_nll(y_obs, &y_model);

    // Guard: if the initial NLL is non-finite, bail out immediately rather
    // than entering the optimization loop with garbage values.
    if !nll.is_finite() {
        return Ok(PoissonResult {
            nll,
            iterations: 0,
            converged: false,
            params: params.all_values(),
        });
    }

    let mut converged = false;
    let mut iter = 0;

    'outer: for _ in 0..config.max_iter {
        iter += 1;
        let mut face_steps = 0usize;

        loop {
            // Compute gradient: try analytical (grad = J^T · w) first,
            // fall back to finite differences if the model doesn't provide
            // an analytical Jacobian.
            let analytical_step = compute_analytical_step_data(
                model,
                params,
                y_obs,
                &y_model,
                &mut all_vals_buf,
                &mut free_idx_buf,
            );
            let grad = if let Some(ref analytical) = analytical_step {
                analytical.grad.clone()
            } else {
                compute_gradient(
                    model,
                    params,
                    y_obs,
                    config.fd_step,
                    &mut all_vals_buf,
                    &mut free_idx_buf,
                )?
            };

            params.free_indices_into(&mut free_idx_buf);

            let using_fd = analytical_step.is_none();
            if using_fd {
                params.free_values_into(&mut free_vals_buf);
                let current_mask = inactive_free_mask(params, &free_idx_buf, &grad);
                if let Some((prev_free, prev_grad, prev_mask)) = pending_fd_state.take() {
                    if prev_mask == current_mask {
                        fd_history.update(&prev_free, &free_vals_buf, &prev_grad, &grad);
                    } else {
                        fd_history.clear();
                    }
                }
                pending_fd_state = Some((free_vals_buf.clone(), grad.clone(), current_mask));
            } else {
                pending_fd_state.take();
                fd_history.clear();
            }

            // Use projected-gradient optimality for bound-constrained problems.
            let projected_grad_norm = projected_gradient_norm(params, &free_idx_buf, &grad);
            if projected_grad_norm < config.tol_param {
                converged = true;
                break 'outer;
            }

            let (search_dir, initial_alpha): (Vec<f64>, f64) =
                if let Some(ref analytical) = analytical_step {
                    let inactive_positions = inactive_free_positions(params, &free_idx_buf, &grad);
                    if inactive_positions.is_empty() {
                        converged = true;
                        break 'outer;
                    }
                    let reduced_fisher = extract_submatrix(&analytical.fisher, &inactive_positions);
                    let reduced_grad: Vec<f64> =
                        inactive_positions.iter().map(|&pos| grad[pos]).collect();
                    let reduced_dir = crate::lm::solve_damped_system(
                        &reduced_fisher,
                        &reduced_grad,
                        config.gauss_newton_lambda,
                    )
                    .unwrap_or_else(|| {
                        reduced_grad
                            .iter()
                            .enumerate()
                            .map(|(j, &g)| g / reduced_fisher.get(j, j).max(1e-12))
                            .collect()
                    });
                    let mut dir = vec![0.0; grad.len()];
                    for (&pos, &value) in inactive_positions.iter().zip(reduced_dir.iter()) {
                        dir[pos] = value;
                    }
                    (dir, config.step_size)
                } else {
                    let inactive_positions = inactive_free_positions(params, &free_idx_buf, &grad);
                    if inactive_positions.is_empty() {
                        converged = true;
                        break 'outer;
                    }
                    let used_history = !fd_history.s_list.is_empty();
                    let mut dir = fd_history
                        .apply_on_positions(&grad, &inactive_positions)
                        .unwrap_or_else(|| {
                            parameter_scaled_gradient_direction(params, &free_idx_buf, &grad)
                        });
                    let descent = dot(&grad, &dir);
                    if !descent.is_finite() || descent <= 0.0 {
                        dir = parameter_scaled_gradient_direction(params, &free_idx_buf, &grad);
                    }
                    if used_history && descent.is_finite() && descent > 0.0 {
                        (dir, config.step_size)
                    } else {
                        let search_norm: f64 = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
                        (dir, config.step_size / search_norm.max(1.0))
                    }
                };

            params.free_values_into(&mut free_vals_buf);
            old_free_buf.clear();
            old_free_buf.extend_from_slice(&free_vals_buf);

            match backtracking_line_search(
                model,
                params,
                y_obs,
                &old_free_buf,
                &free_idx_buf,
                &search_dir,
                initial_alpha,
                config,
                &grad,
                nll,
                &mut all_vals_buf,
                &mut free_vals_buf,
                &mut trial_free_buf,
            ) {
                LineSearchResult::Accepted {
                    nll: new_nll,
                    y_model: new_y_model,
                    hit_boundary,
                } => {
                    if !using_fd {
                        pending_fd_state = None;
                    }
                    nll = new_nll;
                    y_model = new_y_model;

                    if hit_boundary && face_steps < MAX_FACE_STEPS_PER_ITER {
                        face_steps += 1;
                        continue;
                    }
                }
                LineSearchResult::Stagnated => {
                    converged = true;
                    break 'outer;
                }
                LineSearchResult::Failed => {
                    break 'outer;
                }
            }

            params.free_values_into(&mut free_vals_buf);
            let step_norm =
                normalized_step_norm(&old_free_buf, &free_vals_buf, params, &free_idx_buf);
            if step_norm < config.tol_param {
                converged = true;
                break 'outer;
            }

            break;
        }
    }

    Ok(PoissonResult {
        nll,
        iterations: iter,
        converged,
        params: params.all_values(),
    })
}

/// Convert transmission model + counts to a counts-based forward model.
///
/// Given:
///   Y_model = flux × T_model(θ) + background
///
/// This wraps a transmission model to predict observed counts.
///
/// The `flux` and `background` slices must have the same length as the
/// transmission vector returned by the inner model. In debug builds,
/// `evaluate()` asserts this invariant.
pub struct CountsModel<'a> {
    /// Underlying transmission model.
    pub transmission_model: &'a dyn FitModel,
    /// Incident flux (counts per bin in open beam, after normalization).
    pub flux: &'a [f64],
    /// Background counts per bin.
    pub background: &'a [f64],
    /// Total parameter count in the wrapped model.
    pub n_params: usize,
}

impl<'a> FitModel for CountsModel<'a> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let transmission = self.transmission_model.evaluate(params)?;
        debug_assert_eq!(
            transmission.len(),
            self.flux.len(),
            "CountsModel: transmission length ({}) != flux length ({})",
            transmission.len(),
            self.flux.len(),
        );
        debug_assert_eq!(
            self.flux.len(),
            self.background.len(),
            "CountsModel: flux length ({}) != background length ({})",
            self.flux.len(),
            self.background.len(),
        );
        Ok(transmission
            .iter()
            .zip(self.flux.iter())
            .zip(self.background.iter())
            .map(|((&t, &f), &b)| f * t + b)
            .collect())
    }

    /// Analytical Jacobian: ∂Y/∂θ = flux · ∂T_inner/∂θ.
    ///
    /// Background is constant w.r.t. θ and drops out.
    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        // Recover inner transmission: T = (Y - background) / flux
        let t_inner: Vec<f64> = y_current
            .iter()
            .zip(self.flux.iter())
            .zip(self.background.iter())
            .map(|((&y, &f), &b)| if f.abs() > 1e-30 { (y - b) / f } else { 0.0 })
            .collect();
        let inner_jac =
            self.transmission_model
                .analytical_jacobian(params, free_param_indices, &t_inner)?;
        let n_free = free_param_indices.len();
        let mut jac = FlatMatrix::zeros(n_e, n_free);
        for i in 0..n_e {
            for j in 0..n_free {
                *jac.get_mut(i, j) = self.flux[i] * inner_jac.get(i, j);
            }
        }
        Some(jac)
    }
}

// ── ForwardModel implementation for CountsModel (Phase 1) ────────────────

impl<'a> crate::forward_model::ForwardModel for CountsModel<'a> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    // No analytical jacobian — uses finite differences (same as FitModel).

    fn n_data(&self) -> usize {
        self.flux.len()
    }

    fn n_params(&self) -> usize {
        self.n_params
    }
}

/// Counts model with optional nuisance scaling of signal and detector background.
///
/// Given a transmission model `T(θ)`, predicts:
///
///   Y(E) = α₁ · [Φ(E) · T(θ)] + α₂ · B(E)
///
/// where `α₁` and `α₂` are parameter-vector entries.
pub struct CountsBackgroundScaleModel<'a> {
    /// Underlying transmission model.
    pub transmission_model: &'a dyn FitModel,
    /// Incident flux spectrum.
    pub flux: &'a [f64],
    /// Detector background spectrum.
    pub background: &'a [f64],
    /// Index of α₁ in the parameter vector.
    pub alpha1_index: usize,
    /// Index of α₂ in the parameter vector.
    pub alpha2_index: usize,
    /// Total parameter count in the wrapped model.
    pub n_params: usize,
}

impl<'a> FitModel for CountsBackgroundScaleModel<'a> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let transmission = self.transmission_model.evaluate(params)?;
        let alpha1 = params[self.alpha1_index];
        let alpha2 = params[self.alpha2_index];
        debug_assert_eq!(transmission.len(), self.flux.len());
        debug_assert_eq!(self.flux.len(), self.background.len());
        Ok(transmission
            .iter()
            .zip(self.flux.iter())
            .zip(self.background.iter())
            .map(|((&t, &f), &b)| alpha1 * f * t + alpha2 * b)
            .collect())
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        let n_free = free_param_indices.len();
        let alpha1 = params[self.alpha1_index];
        let alpha2 = params[self.alpha2_index];
        let alpha1_col = free_param_indices.iter().position(|&i| i == self.alpha1_index);
        let alpha2_col = free_param_indices.iter().position(|&i| i == self.alpha2_index);
        let inner_free: Vec<usize> = free_param_indices
            .iter()
            .copied()
            .filter(|&i| i != self.alpha1_index && i != self.alpha2_index)
            .collect();

        let t_inner: Vec<f64> = y_current
            .iter()
            .zip(self.background.iter())
            .zip(self.flux.iter())
            .map(|((&y, &b), &f)| {
                if f.abs() > 1e-30 && alpha1.abs() > 1e-30 {
                    (y - alpha2 * b) / (alpha1 * f)
                } else {
                    0.0
                }
            })
            .collect();

        let inner_jac = if !inner_free.is_empty() {
            self.transmission_model
                .analytical_jacobian(params, &inner_free, &t_inner)
        } else {
            None
        };

        let mut jacobian = FlatMatrix::zeros(n_e, n_free);
        if let Some(ref ij) = inner_jac {
            let mut inner_col = 0;
            for (col, &fp) in free_param_indices.iter().enumerate() {
                if fp == self.alpha1_index || fp == self.alpha2_index {
                    continue;
                }
                for row in 0..n_e {
                    *jacobian.get_mut(row, col) = alpha1 * self.flux[row] * ij.get(row, inner_col);
                }
                inner_col += 1;
            }
        } else if !inner_free.is_empty() {
            return None;
        }

        if let Some(col) = alpha1_col {
            for row in 0..n_e {
                *jacobian.get_mut(row, col) = self.flux[row] * t_inner[row];
            }
        }
        if let Some(col) = alpha2_col {
            for row in 0..n_e {
                *jacobian.get_mut(row, col) = self.background[row];
            }
        }

        Some(jacobian)
    }
}

impl<'a> crate::forward_model::ForwardModel for CountsBackgroundScaleModel<'a> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn n_data(&self) -> usize {
        self.flux.len()
    }

    fn n_params(&self) -> usize {
        self.n_params
    }
}

/// KL-compatible background model for transmission data.
///
/// Given a transmission model T_inner(θ), predicts:
///
///   T_out(E) = T_inner(E) + b₀ + b₁/√E
///
/// where b₀ and b₁ are the additive background parameters at indices
/// `b0_index` and `b1_index` in the parameter vector.
///
/// Unlike `NormalizedTransmissionModel` (which uses `Anorm * T + BackA +
/// BackB/√E + BackC√E` with 4 free parameters), this model:
/// - Has only 2 background parameters (b₀, b₁), reducing overfitting risk
/// - Constrains b₀, b₁ ≥ 0 via parameter bounds (physical: background
///   adds counts, never subtracts), ensuring T_out > 0 for valid Poisson NLL
/// - Does NOT multiply T_inner by a normalization factor — normalization
///   is handled separately (nuisance estimation for counts, or pre-processing
///   for transmission data)
///
/// ## Gradient
///
/// - ∂T_out/∂nₖ = ∂T_inner/∂nₖ = -σₖ(E)·T_inner(E)  (same as bare model)
/// - ∂T_out/∂b₀ = 1
/// - ∂T_out/∂b₁ = 1/√E
pub struct TransmissionKLBackgroundModel<'a> {
    /// Underlying transmission model (density parameters only).
    pub inner: &'a dyn FitModel,
    /// Precomputed 1/√E for each energy bin.
    pub inv_sqrt_energies: Vec<f64>,
    /// Index of b₀ (constant background) in the parameter vector.
    pub b0_index: usize,
    /// Index of b₁ (1/√E background) in the parameter vector.
    pub b1_index: usize,
    /// Total parameter count in the wrapped model.
    pub n_params: usize,
}

impl<'a> FitModel for TransmissionKLBackgroundModel<'a> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let t_inner = self.inner.evaluate(params)?;
        let b0 = params[self.b0_index];
        let b1 = params[self.b1_index];
        Ok(t_inner
            .iter()
            .zip(self.inv_sqrt_energies.iter())
            .map(|(&t, &inv_sqrt_e)| t + b0 + b1 * inv_sqrt_e)
            .collect())
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        let n_free = free_param_indices.len();

        // Identify which free params are background vs inner model.
        let b0_col = free_param_indices.iter().position(|&i| i == self.b0_index);
        let b1_col = free_param_indices.iter().position(|&i| i == self.b1_index);

        // Inner model free params (those not b0 or b1).
        let inner_free: Vec<usize> = free_param_indices
            .iter()
            .copied()
            .filter(|&i| i != self.b0_index && i != self.b1_index)
            .collect();

        // Get inner model Jacobian for density columns.
        let inner_jac = if !inner_free.is_empty() {
            // Evaluate inner model at current params to get T_inner for y_current.
            let t_inner = self.inner.evaluate(params).ok()?;
            self.inner
                .analytical_jacobian(params, &inner_free, &t_inner)
        } else {
            None
        };

        let mut jacobian = FlatMatrix::zeros(n_e, n_free);

        // Fill inner model columns (density, temperature).
        // Inner Jacobian is the same as bare model — background doesn't
        // affect ∂T_inner/∂nₖ.
        if let Some(ref ij) = inner_jac {
            let mut inner_col = 0;
            for (col, &fp) in free_param_indices.iter().enumerate() {
                if fp == self.b0_index || fp == self.b1_index {
                    continue;
                }
                for row in 0..n_e {
                    *jacobian.get_mut(row, col) = ij.get(row, inner_col);
                }
                inner_col += 1;
            }
        } else {
            // No analytical inner Jacobian — fall back to FD for entire model.
            return None;
        }

        // Background columns.
        if let Some(col) = b0_col {
            for row in 0..n_e {
                *jacobian.get_mut(row, col) = 1.0; // ∂T_out/∂b₀ = 1
            }
        }
        if let Some(col) = b1_col {
            for row in 0..n_e {
                *jacobian.get_mut(row, col) = self.inv_sqrt_energies[row]; // ∂T_out/∂b₁ = 1/√E
            }
        }

        Some(jacobian)
    }
}

impl<'a> crate::forward_model::ForwardModel for TransmissionKLBackgroundModel<'a> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn n_data(&self) -> usize {
        self.inv_sqrt_energies.len()
    }

    fn n_params(&self) -> usize {
        self.n_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::FitModel;
    use crate::parameters::FitParameter;

    /// Simple model: y = a * exp(-b * x)
    /// This mimics transmission: counts = flux * exp(-density * sigma)
    struct ExponentialModel {
        x: Vec<f64>,
        flux: Vec<f64>,
    }

    impl FitModel for ExponentialModel {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            let b = params[0]; // "density"
            Ok(self
                .x
                .iter()
                .zip(self.flux.iter())
                .map(|(&xi, &fi)| fi * (-b * xi).exp())
                .collect())
        }
    }

    #[test]
    fn test_poisson_nll_perfect_match() {
        let y_obs = vec![10.0, 20.0, 30.0];
        let y_model = vec![10.0, 20.0, 30.0];
        let nll = poisson_nll(&y_obs, &y_model);
        // NLL = Σ(y_model - y_obs*ln(y_model))
        let expected: f64 = y_obs
            .iter()
            .zip(y_model.iter())
            .map(|(&o, &m)| m - o * m.ln())
            .sum();
        assert!((nll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_fit_exponential() {
        // Generate synthetic Poisson data from y = 1000 * exp(-0.5 * x)
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let true_b = 0.5;
        let flux: Vec<f64> = vec![1000.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };

        // Use exact expected counts (no noise) for reproducibility
        let y_obs = model.evaluate(&[true_b]).unwrap();

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("b", 1.0), // Initial guess 2× off
        ]);

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default()).unwrap();

        assert!(
            result.converged,
            "Poisson fit did not converge after {} iterations",
            result.iterations,
        );
        assert!(
            (result.params[0] - true_b).abs() / true_b < 0.05,
            "Fitted b = {}, true = {}, error = {:.1}%",
            result.params[0],
            true_b,
            (result.params[0] - true_b).abs() / true_b * 100.0,
        );
    }

    #[test]
    fn test_poisson_fit_low_counts() {
        // Low-count regime: flux = 10 counts per bin
        let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
        let true_b = 0.3;
        let flux: Vec<f64> = vec![10.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };

        let y_obs = model.evaluate(&[true_b]).unwrap();

        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", 0.1)]);

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default()).unwrap();

        assert!(result.converged);
        assert!(
            (result.params[0] - true_b).abs() / true_b < 0.1,
            "Low-count: fitted b = {}, true = {}",
            result.params[0],
            true_b,
        );
    }

    #[test]
    fn test_poisson_non_negativity() {
        // Data that would drive parameter negative without constraint
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let flux: Vec<f64> = vec![100.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };

        // Generate data with b=0 (constant), but start with b=1
        let y_obs: Vec<f64> = vec![100.0; x.len()];

        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", 1.0)]);

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default()).unwrap();

        assert!(
            result.params[0] >= 0.0,
            "b should be non-negative, got {}",
            result.params[0],
        );
        assert!(
            result.params[0] < 0.1,
            "b should be ~0, got {}",
            result.params[0],
        );
    }

    #[test]
    fn test_counts_model() {
        struct ConstTransmission;
        impl FitModel for ConstTransmission {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![params[0]; 3])
            }
        }

        let t_model = ConstTransmission;
        let flux = [100.0, 200.0, 300.0];
        let background = [5.0, 10.0, 15.0];
        let counts_model = CountsModel {
            transmission_model: &t_model,
            flux: &flux,
            background: &background,
            n_params: 1,
        };

        // T = 0.5 → counts = flux*0.5 + background
        let result = counts_model.evaluate(&[0.5]).unwrap();
        assert!((result[0] - 55.0).abs() < 1e-10);
        assert!((result[1] - 110.0).abs() < 1e-10);
        assert!((result[2] - 165.0).abs() < 1e-10);
        assert_eq!(
            crate::forward_model::ForwardModel::n_params(&counts_model),
            1
        );
    }

    #[test]
    fn test_counts_background_scale_model() {
        struct ConstTransmission;
        impl FitModel for ConstTransmission {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![params[0]; 3])
            }

            fn analytical_jacobian(
                &self,
                _params: &[f64],
                free_param_indices: &[usize],
                _y_current: &[f64],
            ) -> Option<FlatMatrix> {
                let mut jac = FlatMatrix::zeros(3, free_param_indices.len());
                for (col, &fp) in free_param_indices.iter().enumerate() {
                    if fp == 0 {
                        for row in 0..3 {
                            *jac.get_mut(row, col) = 1.0;
                        }
                    }
                }
                Some(jac)
            }
        }

        let t_model = ConstTransmission;
        let flux = [100.0, 200.0, 300.0];
        let background = [5.0, 10.0, 15.0];
        let counts_model = CountsBackgroundScaleModel {
            transmission_model: &t_model,
            flux: &flux,
            background: &background,
            alpha1_index: 1,
            alpha2_index: 2,
            n_params: 3,
        };

        let params = [0.5, 0.8, 1.5];
        let result = counts_model.evaluate(&params).unwrap();
        assert!((result[0] - 47.5).abs() < 1e-10);
        assert!((result[1] - 95.0).abs() < 1e-10);
        assert!((result[2] - 142.5).abs() < 1e-10);
        assert_eq!(
            crate::forward_model::ForwardModel::n_params(&counts_model),
            3
        );
    }

    #[test]
    fn test_poisson_fit_multi_density_temperature_converges() {
        struct MultiDensityCountsModel {
            energies: Vec<f64>,
            flux: Vec<f64>,
            density_count: usize,
            temp_index: usize,
        }

        impl MultiDensityCountsModel {
            fn sigma(&self, iso: usize, energy: f64, temp_k: f64) -> f64 {
                let center = 6.0 + iso as f64 * 4.5;
                let amp = 150.0 + 25.0 * iso as f64;
                let base_width = 0.8 + 0.12 * iso as f64;
                let width_coeff = 0.05 + 0.01 * iso as f64;
                let width = (base_width * (1.0 + width_coeff * (temp_k - 300.0) / 300.0)).max(0.1);
                let delta = energy - center;
                let gauss = (-(delta * delta) / (2.0 * width * width)).exp();
                amp * gauss
            }

            fn dsigma_dt(&self, iso: usize, energy: f64, temp_k: f64) -> f64 {
                let center = 6.0 + iso as f64 * 4.5;
                let amp = 150.0 + 25.0 * iso as f64;
                let base_width = 0.8 + 0.12 * iso as f64;
                let width_coeff = 0.05 + 0.01 * iso as f64;
                let width = (base_width * (1.0 + width_coeff * (temp_k - 300.0) / 300.0)).max(0.1);
                let delta = energy - center;
                let gauss = (-(delta * delta) / (2.0 * width * width)).exp();
                let dwidth_dt = base_width * width_coeff / 300.0;
                amp * gauss * (delta * delta) * dwidth_dt / width.powi(3)
            }
        }

        impl FitModel for MultiDensityCountsModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let temp_k = params[self.temp_index];
                let mut out = Vec::with_capacity(self.energies.len());
                for (i, &energy) in self.energies.iter().enumerate() {
                    let optical_depth = (0..self.density_count)
                        .map(|iso| params[iso] * self.sigma(iso, energy, temp_k))
                        .sum::<f64>();
                    out.push(self.flux[i] * (-optical_depth).exp());
                }
                Ok(out)
            }

            fn analytical_jacobian(
                &self,
                params: &[f64],
                free_param_indices: &[usize],
                y_current: &[f64],
            ) -> Option<crate::lm::FlatMatrix> {
                let temp_k = params[self.temp_index];
                let mut jac =
                    crate::lm::FlatMatrix::zeros(self.energies.len(), free_param_indices.len());
                for (row, &energy) in self.energies.iter().enumerate() {
                    let y = y_current[row];
                    let mut sum_n_dsigma_dt = 0.0;
                    for (iso, &density) in params[..self.density_count].iter().enumerate() {
                        sum_n_dsigma_dt += density * self.dsigma_dt(iso, energy, temp_k);
                    }
                    for (col, &fp) in free_param_indices.iter().enumerate() {
                        let val = if fp == self.temp_index {
                            -y * sum_n_dsigma_dt
                        } else {
                            -y * self.sigma(fp, energy, temp_k)
                        };
                        *jac.get_mut(row, col) = val;
                    }
                }
                Some(jac)
            }
        }

        let energies: Vec<f64> = (0..220).map(|i| 1.0 + 0.18 * i as f64).collect();
        let flux: Vec<f64> = energies
            .iter()
            .map(|&e| 1500.0 * (1.0 + 0.15 * (e / 8.0).sin()).max(0.2))
            .collect();
        let density_count = 6usize;
        let temp_index = density_count;
        let model = MultiDensityCountsModel {
            energies,
            flux,
            density_count,
            temp_index,
        };

        let true_params = vec![3.2e-4, 2.4e-4, 1.7e-4, 1.1e-4, 7.5e-5, 4.2e-5, 360.0];
        let y_obs = model.evaluate(&true_params).unwrap();

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("n0", 6.0e-4),
            FitParameter::non_negative("n1", 4.0e-4),
            FitParameter::non_negative("n2", 2.5e-4),
            FitParameter::non_negative("n3", 1.8e-4),
            FitParameter::non_negative("n4", 1.0e-4),
            FitParameter::non_negative("n5", 8.0e-5),
            FitParameter {
                name: "temperature_k".into(),
                value: 300.0,
                lower: 1.0,
                upper: 5000.0,
                fixed: false,
            },
        ]);

        let config = PoissonConfig {
            max_iter: 200,
            gauss_newton_lambda: 1e-4,
            ..PoissonConfig::default()
        };
        let result = poisson_fit(&model, &y_obs, &mut params, &config).unwrap();

        assert!(
            result.converged,
            "multi-density+temperature Poisson fit did not converge after {} iterations",
            result.iterations,
        );
        assert!(
            result.iterations < config.max_iter,
            "fit hit max_iter={}, params={:?}",
            config.max_iter,
            result.params,
        );

        for (i, (&fit, &truth)) in result.params[..density_count]
            .iter()
            .zip(true_params[..density_count].iter())
            .enumerate()
        {
            let rel_err = (fit - truth).abs() / truth;
            assert!(
                rel_err < 0.10,
                "density[{i}] fit={fit} truth={truth} rel_err={rel_err:.3}",
            );
        }

        let fitted_temp = result.params[temp_index];
        assert!(
            (fitted_temp - true_params[temp_index]).abs() < 10.0,
            "temperature fit={fitted_temp} truth={}",
            true_params[temp_index],
        );
        assert!(
            result.iterations <= 80,
            "expected analytical KL path to converge well before max_iter; got {}",
            result.iterations,
        );
    }

    #[test]
    fn test_poisson_fit_exact_optimum_without_analytical_jacobian_converges() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let true_b = 0.5;
        let flux: Vec<f64> = vec![1000.0; x.len()];

        let model = ExponentialModel { x, flux };
        let y_obs = model.evaluate(&[true_b]).unwrap();
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", true_b)]);
        let config = PoissonConfig {
            fd_step: 1e-4,
            tol_param: 1e-12,
            max_iter: 50,
            ..PoissonConfig::default()
        };

        let result = poisson_fit(&model, &y_obs, &mut params, &config).unwrap();

        assert!(
            result.converged,
            "exact-optimum FD fit should converge instead of exhausting line search"
        );
        assert!(
            result.iterations < config.max_iter,
            "fit should stop by convergence, not hit max_iter"
        );
        assert!(
            (result.params[0] - true_b).abs() < 1e-6,
            "parameter drifted away from optimum: {}",
            result.params[0]
        );
    }

    #[test]
    fn test_projected_gradient_ignores_lower_bound_blocked_direction() {
        let params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.0),
            FitParameter::unbounded("temp", 300.0),
        ]);
        let free_idx = vec![0, 1];
        let grad = vec![0.25, -0.5];

        let inactive = inactive_free_positions(&params, &free_idx, &grad);
        assert_eq!(
            inactive,
            vec![1],
            "lower-bound blocked density should be active"
        );

        let pg_norm = projected_gradient_norm(&params, &free_idx, &grad);
        assert!(
            (pg_norm - 0.5).abs() < 1e-12,
            "projected gradient should ignore blocked lower-bound component"
        );
    }

    #[test]
    fn test_max_feasible_step_hits_lower_bound() {
        let params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.2),
            FitParameter::unbounded("temp", 300.0),
        ]);
        let alpha = max_feasible_step(&params, &[0, 1], &[0.2, 300.0], &[0.5, 0.0]);
        assert!(
            (alpha - 0.4).abs() < 1e-12,
            "feasible step should stop exactly at lower bound"
        );
    }

    #[test]
    fn test_max_feasible_step_hits_upper_bound() {
        let params = ParameterSet::new(vec![FitParameter {
            name: "temp".into(),
            value: 300.0,
            lower: 1.0,
            upper: 500.0,
            fixed: false,
        }]);
        let alpha = max_feasible_step(&params, &[0], &[300.0], &[-50.0]);
        assert!(
            (alpha - 4.0).abs() < 1e-12,
            "feasible step should stop exactly at upper bound"
        );
    }

    #[test]
    fn test_inactive_mask_changes_when_bound_activity_changes() {
        let free_idx = vec![0, 1];
        let params_at_bound = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.0),
            FitParameter::non_negative("temp", 1.0),
        ]);
        let params_free = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.2),
            FitParameter::non_negative("temp", 1.0),
        ]);

        let mask_at_bound = inactive_free_mask(&params_at_bound, &free_idx, &[0.3, -0.2]);
        let mask_free = inactive_free_mask(&params_free, &free_idx, &[0.3, -0.2]);

        assert_eq!(mask_at_bound, vec![false, true]);
        assert_eq!(mask_free, vec![true, true]);
        assert_ne!(
            mask_at_bound, mask_free,
            "active-set changes should invalidate FD quasi-Newton history"
        );
    }

    #[test]
    fn test_lbfgs_history_two_loop_matches_secant_direction() {
        let mut history = LbfgsHistory::new(4);
        history.update(&[0.0], &[1.0], &[0.0], &[2.0]);
        let dir = history
            .apply_on_positions(&[4.0], &[0])
            .expect("history should produce a direction");
        assert!(
            (dir[0] - 2.0).abs() < 1e-12,
            "1D secant pair should scale gradient by inverse curvature"
        );
    }

    #[test]
    fn test_lbfgs_subspace_ignores_active_components() {
        let mut history = LbfgsHistory::new(4);
        history.update(&[0.0, 0.0], &[1.0, 100.0], &[0.0, 0.0], &[2.0, 100.0]);
        let dir = history
            .apply_on_positions(&[4.0, 50.0], &[0])
            .expect("subspace history should produce a direction");
        assert!(
            (dir[0] - 2.0).abs() < 1e-12,
            "inactive-subspace L-BFGS should match 1D secant scaling on the free variable"
        );
        assert!(
            dir[1].abs() < 1e-12,
            "inactive-subspace L-BFGS should not leak blocked-variable history into the direction"
        );
    }

    #[test]
    fn test_poisson_fit_converges_at_bound_active_optimum() {
        struct OffsetModel {
            base: Vec<f64>,
        }

        impl FitModel for OffsetModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(self.base.iter().map(|&b| b + params[0]).collect())
            }

            fn analytical_jacobian(
                &self,
                _params: &[f64],
                free_param_indices: &[usize],
                y_current: &[f64],
            ) -> Option<FlatMatrix> {
                let mut jac = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
                for (col, &fp) in free_param_indices.iter().enumerate() {
                    assert_eq!(fp, 0);
                    for row in 0..y_current.len() {
                        *jac.get_mut(row, col) = 1.0;
                    }
                }
                Some(jac)
            }
        }

        let model = OffsetModel {
            base: vec![10.0; 12],
        };
        let y_obs = vec![8.0; 12];
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("offset", 0.0)]);

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default()).unwrap();

        assert!(
            result.converged,
            "bound-active optimum should satisfy projected optimality"
        );
        assert_eq!(
            result.iterations, 1,
            "should stop on projected-gradient check"
        );
        assert!(
            result.params[0].abs() < 1e-12,
            "offset should stay pinned at lower bound, got {}",
            result.params[0]
        );
    }

    #[test]
    fn test_poisson_fit_fd_lbfgs_handles_coupled_two_parameter_model() {
        struct CoupledExponentialModel {
            x: Vec<f64>,
        }

        impl FitModel for CoupledExponentialModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let amp = params[0];
                let decay = params[1];
                Ok(self
                    .x
                    .iter()
                    .map(|&x| amp * (-decay * x).exp() + 1.0)
                    .collect())
            }
        }

        let model = CoupledExponentialModel {
            x: (0..60).map(|i| i as f64 * 0.08).collect(),
        };
        let true_params = [120.0, 0.45];
        let y_obs = model.evaluate(&true_params).unwrap();
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("amp", 30.0),
            FitParameter::non_negative("decay", 1.2),
        ]);
        let config = PoissonConfig {
            max_iter: 120,
            lbfgs_history: 8,
            ..PoissonConfig::default()
        };
        let result = poisson_fit(&model, &y_obs, &mut params, &config).unwrap();
        let mut baseline_params = ParameterSet::new(vec![
            FitParameter::non_negative("amp", 30.0),
            FitParameter::non_negative("decay", 1.2),
        ]);
        let baseline = poisson_fit(
            &model,
            &y_obs,
            &mut baseline_params,
            &PoissonConfig {
                lbfgs_history: 0,
                ..config.clone()
            },
        )
        .unwrap();

        assert!(
            result.converged,
            "FD L-BFGS fit did not converge: {result:?}"
        );
        assert!(
            baseline.converged,
            "baseline FD fit should still converge: {baseline:?}"
        );
        assert!(
            result.iterations <= 60,
            "expected FD quasi-Newton path to converge well before max_iter; got {}",
            result.iterations,
        );
        assert!(
            result.iterations < baseline.iterations,
            "L-BFGS fallback should beat no-history gradient scaling: lbfgs={} baseline={}",
            result.iterations,
            baseline.iterations,
        );
        assert!(
            (result.params[0] - true_params[0]).abs() / true_params[0] < 0.02,
            "amplitude fit={}, true={}",
            result.params[0],
            true_params[0],
        );
        assert!(
            (result.params[1] - true_params[1]).abs() / true_params[1] < 0.02,
            "decay fit={}, true={}",
            result.params[1],
            true_params[1],
        );
    }

    #[test]
    fn test_poisson_fit_fd_lbfgs_with_bound_active_offset_uses_subspace() {
        struct OffsetDecayModel {
            x: Vec<f64>,
        }

        impl FitModel for OffsetDecayModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let offset = params[0];
                let decay = params[1];
                Ok(self
                    .x
                    .iter()
                    .map(|&x| offset + (-decay * x).exp())
                    .collect())
            }
        }

        let model = OffsetDecayModel {
            x: (0..60).map(|i| i as f64 * 0.08).collect(),
        };
        let true_params = [0.0, 0.35];
        let y_obs = model.evaluate(&true_params).unwrap();

        let config = PoissonConfig {
            max_iter: 120,
            lbfgs_history: 8,
            ..PoissonConfig::default()
        };
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("offset", 0.0),
            FitParameter::non_negative("decay", 1.1),
        ]);
        let result = poisson_fit(&model, &y_obs, &mut params, &config).unwrap();
        assert!(
            result.converged,
            "subspace FD L-BFGS fit did not converge: {result:?}"
        );
        assert!(
            result.iterations <= 20,
            "bound-active subspace FD fit should converge comfortably before max_iter; got {}",
            result.iterations,
        );
        assert!(
            result.params[0].abs() < 1e-8,
            "offset should remain at the lower bound, got {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - true_params[1]).abs() / true_params[1] < 0.02,
            "decay fit={}, true={}",
            result.params[1],
            true_params[1],
        );
    }

    #[test]
    fn test_poisson_fit_temperature_and_background_converges() {
        struct TempTransmissionModel {
            energies: Vec<f64>,
        }

        impl TempTransmissionModel {
            fn sigma(&self, energy: f64, temp_k: f64) -> f64 {
                let center = 6.0;
                let amp = 110.0;
                let base_width = 0.55;
                let width = (base_width * (temp_k / 300.0).sqrt()).max(0.08);
                let delta = energy - center;
                amp * (-(delta * delta) / (2.0 * width * width)).exp()
            }

            fn dsigma_dt(&self, energy: f64, temp_k: f64) -> f64 {
                let center = 6.0;
                let amp = 110.0;
                let base_width = 0.55;
                let width = (base_width * (temp_k / 300.0).sqrt()).max(0.08);
                let dwidth_dt = if temp_k > 0.0 {
                    base_width / (2.0 * (300.0 * temp_k).sqrt())
                } else {
                    0.0
                };
                let delta = energy - center;
                let gauss = (-(delta * delta) / (2.0 * width * width)).exp();
                amp * gauss * (delta * delta) * dwidth_dt / width.powi(3)
            }
        }

        impl FitModel for TempTransmissionModel {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let density = params[0];
                let temp_k = params[1];
                Ok(self
                    .energies
                    .iter()
                    .map(|&energy| (-density * self.sigma(energy, temp_k)).exp())
                    .collect())
            }

            fn analytical_jacobian(
                &self,
                params: &[f64],
                free_param_indices: &[usize],
                y_current: &[f64],
            ) -> Option<FlatMatrix> {
                let density = params[0];
                let temp_k = params[1];
                let mut jac = FlatMatrix::zeros(self.energies.len(), free_param_indices.len());
                for (row, &energy) in self.energies.iter().enumerate() {
                    let y = y_current[row];
                    let sigma = self.sigma(energy, temp_k);
                    let dsigma_dt = self.dsigma_dt(energy, temp_k);
                    for (col, &fp) in free_param_indices.iter().enumerate() {
                        let deriv = match fp {
                            0 => -sigma * y,
                            1 => -density * dsigma_dt * y,
                            _ => unreachable!("unexpected parameter index {fp}"),
                        };
                        *jac.get_mut(row, col) = deriv;
                    }
                }
                Some(jac)
            }
        }

        let energies: Vec<f64> = (0..180).map(|i| 1.0 + 0.06 * i as f64).collect();
        let inner = TempTransmissionModel {
            energies: energies.clone(),
        };
        let inv_sqrt_energies: Vec<f64> = energies.iter().map(|&e| 1.0 / e.sqrt()).collect();
        let wrapped = TransmissionKLBackgroundModel {
            inner: &inner,
            inv_sqrt_energies,
            b0_index: 2,
            b1_index: 3,
            n_params: 4,
        };

        let true_params = vec![4.5e-4, 345.0, 0.012, 0.008];
        let y_obs = wrapped.evaluate(&true_params).unwrap();

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 8.0e-4),
            FitParameter {
                name: "temperature_k".into(),
                value: 290.0,
                lower: 1.0,
                upper: 5000.0,
                fixed: false,
            },
            FitParameter {
                name: "kl_b0".into(),
                value: 0.0,
                lower: 0.0,
                upper: 0.5,
                fixed: false,
            },
            FitParameter {
                name: "kl_b1".into(),
                value: 0.0,
                lower: 0.0,
                upper: 0.5,
                fixed: false,
            },
        ]);

        let config = PoissonConfig {
            max_iter: 120,
            gauss_newton_lambda: 1e-4,
            ..PoissonConfig::default()
        };
        let result = poisson_fit(&wrapped, &y_obs, &mut params, &config).unwrap();

        assert!(result.converged, "fit did not converge: {result:?}");
        assert!(
            result.iterations <= 80,
            "expected convergence well before max_iter; got {}",
            result.iterations,
        );
        assert!(
            (result.params[0] - true_params[0]).abs() / true_params[0] < 0.05,
            "density fit={}, true={}",
            result.params[0],
            true_params[0],
        );
        assert!(
            (result.params[1] - true_params[1]).abs() < 8.0,
            "temperature fit={}, true={}",
            result.params[1],
            true_params[1],
        );
        assert!(
            (result.params[2] - true_params[2]).abs() < 5e-3,
            "b0 fit={}, true={}",
            result.params[2],
            true_params[2],
        );
        assert!(
            (result.params[3] - true_params[3]).abs() < 5e-3,
            "b1 fit={}, true={}",
            result.params[3],
            true_params[3],
        );
    }
}
