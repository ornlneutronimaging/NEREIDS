//! Poisson-likelihood optimizer for low-count neutron data.
//!
//! When neutron counts are low (< ~30 per bin), the Gaussian/chi-squared
//! assumption breaks down and Poisson statistics must be used directly.
//!
//! ## Negative log-likelihood
//!
//! L(θ) = Σᵢ [y_model(θ)ᵢ - y_obs,ᵢ · ln(y_model(θ)ᵢ)]
//!
//! This is minimized using projected gradient descent with backtracking
//! line search and bound constraints (non-negativity on densities).
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` — Poisson NLL and APGM optimizer

use nereids_core::constants::{PIVOT_FLOOR, POISSON_EPSILON};

use crate::error::FittingError;
use crate::lm::FitModel;
use crate::parameters::ParameterSet;

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

/// Project a point onto the feasible region (parameter bounds).
fn project(params: &mut ParameterSet) {
    for p in &mut params.params {
        p.clamp();
    }
}

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
/// `Some(new_nll)` if a step was accepted, `None` if the line search exhausted
/// all backtracking attempts without finding an acceptable step.
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
    search_dir: &[f64],
    initial_alpha: f64,
    config: &PoissonConfig,
    grad: &[f64],
    nll: f64,
    all_vals_buf: &mut Vec<f64>,
    free_vals_buf: &mut Vec<f64>,
    trial_free_buf: &mut Vec<f64>,
) -> Option<f64> {
    let mut alpha = initial_alpha;
    for _ in 0..50 {
        // Trial step: x_new = project(x - alpha * search_dir)
        trial_free_buf.clear();
        trial_free_buf.extend(
            old_free
                .iter()
                .zip(search_dir.iter())
                .map(|(&v, &d)| v - alpha * d),
        );
        params.set_free_values(trial_free_buf);
        project(params);

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
        let descent = grad
            .iter()
            .zip(old_free.iter())
            .zip(free_vals_buf.iter())
            .map(|((&g, &old), &new)| g * (old - new))
            .sum::<f64>();

        if trial_nll.is_finite() && trial_nll <= nll - config.armijo_c * descent {
            return Some(trial_nll);
        }

        // Backtrack
        alpha *= config.backtrack;
    }
    params.set_free_values(old_free);
    None
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

/// Run Poisson-likelihood optimization using projected gradient descent.
///
/// Uses backtracking line search with Armijo condition and
/// projection onto parameter bounds after each step.
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

    params.all_values_into(&mut all_vals_buf);
    let y_model = model.evaluate(&all_vals_buf)?;
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

    for _ in 0..config.max_iter {
        iter += 1;

        // Compute gradient
        let grad = compute_gradient(
            model,
            params,
            y_obs,
            config.fd_step,
            &mut all_vals_buf,
            &mut free_idx_buf,
        )?;

        // Check gradient norm for convergence
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tol_param {
            converged = true;
            break;
        }

        // Diagonal preconditioning by parameter bound range.
        //
        // For joint density+temperature fitting, parameter scales differ by
        // 10^5+ (density ~0.001 vs temperature ~300). The raw gradient is
        // dominated by the large-scale parameter. Preconditioning by the
        // squared bound range gives Newton-like step sizes:
        //   search_dir_j = grad_j * (upper_j - lower_j)^2
        //
        // This makes the optimizer take steps proportional to the natural
        // range of each parameter, regardless of the gradient magnitude.
        params.free_indices_into(&mut free_idx_buf);
        let search_dir: Vec<f64> = grad
            .iter()
            .zip(free_idx_buf.iter())
            .map(|(&g, &idx)| {
                let p = &params.params[idx];
                let range = p.upper - p.lower;
                if range.is_finite() && range > 1e-10 {
                    // Bounded parameter: scale by range² for Newton-like step
                    g * range * range
                } else {
                    // Unbounded (e.g., density with upper=inf):
                    // Use current value as scale proxy (like relative step)
                    let scale = p.value.abs().max(1e-3);
                    g * scale * scale
                }
            })
            .collect();

        let search_norm: f64 = search_dir.iter().map(|d| d * d).sum::<f64>().sqrt();

        // Backtracking line search with preconditioned search direction.
        params.free_values_into(&mut free_vals_buf);
        old_free_buf.clear();
        old_free_buf.extend_from_slice(&free_vals_buf);
        let initial_alpha = config.step_size / search_norm.max(1.0);

        match backtracking_line_search(
            model,
            params,
            y_obs,
            &old_free_buf,
            &search_dir,
            initial_alpha,
            config,
            &grad,
            nll,
            &mut all_vals_buf,
            &mut free_vals_buf,
            &mut trial_free_buf,
        ) {
            Some(new_nll) => nll = new_nll,
            None => {
                // Can't improve from this point; stop without claiming convergence.
                // (params already restored by backtracking_line_search)
                break;
            }
        }

        // Convergence check: relative step size in parameter space.
        // Each parameter's step is normalized by its scale so that
        // temperature (range ~5000) doesn't dominate over density (range ~0.01).
        params.free_values_into(&mut free_vals_buf);
        let step_norm: f64 = old_free_buf
            .iter()
            .zip(free_vals_buf.iter())
            .zip(free_idx_buf.iter())
            .map(|((o, n), &idx)| {
                let range = params.params[idx].upper - params.params[idx].lower;
                let scale = if range.is_finite() && range > 1e-10 {
                    range
                } else {
                    o.abs().max(1e-3)
                };
                ((o - n) / scale).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        if step_norm < config.tol_param {
            converged = true;
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
        // CountsModel doesn't own the parameter vector — the count is
        // determined by ParameterSet. Return flux length as a proxy
        // (n_data), since n_params is only used by ForwardModel consumers
        // for buffer sizing, not by the Poisson optimizer.
        self.flux.len()
    }
}

use crate::lm::FlatMatrix;

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
        // The wrapper adds 2 background parameters (b0, b1).
        // n_data from inner is a reasonable proxy for the base param count.
        self.inv_sqrt_energies.len()
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
        };

        // T = 0.5 → counts = flux*0.5 + background
        let result = counts_model.evaluate(&[0.5]).unwrap();
        assert!((result[0] - 55.0).abs() < 1e-10);
        assert!((result[1] - 110.0).abs() < 1e-10);
        assert!((result[2] - 165.0).abs() < 1e-10);
    }
}
