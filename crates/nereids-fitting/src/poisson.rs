//! Poisson-likelihood optimizer for low-count neutron data.
//!
//! When neutron counts are low (< ~30 per bin), the Gaussian/chi-squared
//! assumption breaks down and Poisson statistics must be used directly.
//!
//! ## Negative log-likelihood
//!
//! L(θ) = Σᵢ [y_model(θ)ᵢ - y_obs,ᵢ · ln(y_model(θ)ᵢ)]
//!
//! This is minimized using L-BFGS with projected gradient for bound constraints
//! (non-negativity on densities).
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` — Poisson NLL and APGM optimizer

use nereids_endf::resonance::ResonanceData;
use nereids_physics::transmission::{self, InstrumentParams};

use nereids_core::constants::{PIVOT_FLOOR, POISSON_EPSILON};

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
    /// and gradient-norm convergence checks in `poisson_fit` and `poisson_fit_analytic`.
    pub tol_param: f64,
    /// Armijo line search parameter (sufficient decrease).
    pub armijo_c: f64,
    /// Line search backtracking factor.
    pub backtrack: f64,
    /// Number of L-BFGS correction pairs (memory parameter m).
    /// Only used by `poisson_fit_lbfgsb`. Default: 7.
    pub lbfgsb_memory: usize,
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
            lbfgsb_memory: 7,
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

/// Single-bin Poisson NLL gradient (∂NLL/∂y_model) with smooth extrapolation.
///
/// For mdl > 0: ∂NLL/∂mdl = 1 − obs/mdl
/// For mdl <= epsilon: linear extrapolation from epsilon,
///   NLL'(ε) + NLL''(ε)·(mdl−ε) = (1 − obs/ε) − (obs/ε²)·delta
///
/// Since delta = ε − mdl ≥ 0, the derivative of the NLL term w.r.t. mdl
/// is the derivative of the quadratic extrapolation in poisson_nll_term.
///
/// The minimum curvature floor (1/ε when obs == 0) must match
/// `poisson_nll_term` so the gradient is consistent with the objective.
#[inline]
fn poisson_nll_grad_term(obs: f64, mdl: f64) -> f64 {
    // #125.3: Negative observed counts would produce wrong-signed gradient terms.
    // Release builds skip this check; callers must ensure non-negative counts. See #125 item 3.
    debug_assert!(
        obs.is_finite() && obs >= 0.0,
        "poisson_nll_grad_term: obs must be finite and >= 0, got {obs}"
    );
    if mdl > POISSON_EPSILON {
        1.0 - obs / mdl
    } else {
        let eps = POISSON_EPSILON;
        let grad_eps = 1.0 - obs / eps;
        // Minimum curvature 1/eps, matching poisson_nll_term.
        let hess_eps = if obs > 0.0 {
            obs / (eps * eps)
        } else {
            1.0 / eps
        };
        let delta = eps - mdl;
        // g(eps) + g'(eps)*(mdl - eps) = g(eps) - g'(eps)*delta
        // where g'(x) = NLL''(x) = hess_eps at x=eps
        grad_eps - hess_eps * delta
    }
}

/// Compute gradient of Poisson NLL by finite differences.
///
/// `all_vals_buf` is a reusable scratch buffer for `params.all_values_into()`,
/// avoiding a fresh allocation on every `model.evaluate()` call inside the
/// per-parameter FD loop (N_free+1 allocations saved per gradient call).
fn compute_gradient(
    model: &dyn FitModel,
    params: &mut ParameterSet,
    y_obs: &[f64],
    fd_step: f64,
    all_vals_buf: &mut Vec<f64>,
) -> Vec<f64> {
    params.all_values_into(all_vals_buf);
    let base_model = model.evaluate(all_vals_buf);
    let base_nll = poisson_nll(y_obs, &base_model);

    let free_indices = params.free_indices();
    let mut grad = vec![0.0; free_indices.len()];

    for (j, &idx) in free_indices.iter().enumerate() {
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
        let perturbed_model = model.evaluate(all_vals_buf);
        let perturbed_nll = poisson_nll(y_obs, &perturbed_model);
        params.params[idx].value = original;

        grad[j] = (perturbed_nll - base_nll) / actual_step;
    }

    grad
}

/// Project a point onto the feasible region (parameter bounds).
fn project(params: &mut ParameterSet) {
    for p in &mut params.params {
        p.clamp();
    }
}

/// Backtracking line search with Armijo condition.
///
/// Both `poisson_fit` (vanilla gradient descent) and `poisson_fit_analytic`
/// (Fisher-preconditioned descent) share the same line-search loop structure:
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
        let trial_model = model.evaluate(all_vals_buf);

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
/// Returns `Some(PoissonResult)` if all parameters are fixed (either a valid
/// result with converged=true, or a non-finite NLL with converged=false).
/// Returns `None` if there are free parameters and optimization should proceed.
fn try_early_return_fixed(
    model: &dyn FitModel,
    y_obs: &[f64],
    params: &ParameterSet,
) -> Option<PoissonResult> {
    if params.n_free() != 0 {
        return None;
    }
    let y_model = model.evaluate(&params.all_values());
    let nll = poisson_nll(y_obs, &y_model);
    if !nll.is_finite() {
        return Some(PoissonResult {
            nll,
            iterations: 0,
            converged: false,
            params: params.all_values(),
        });
    }
    Some(PoissonResult {
        nll,
        iterations: 0,
        converged: true,
        params: params.all_values(),
    })
}

/// Validate inputs shared by `poisson_fit_analytic` and `poisson_fit_lbfgsb`.
///
/// Checks:
/// - `flux` and all `cross_sections[k]` have the same length as `y_obs`.
/// - Every free parameter is mapped by `density_indices` or `temperature_index`.
///
/// Panics on validation failure (same contract as the original inline asserts).
fn validate_analytic_inputs(
    n_e: usize,
    flux: &[f64],
    cross_sections: &[Vec<f64>],
    density_indices: &[usize],
    params: &ParameterSet,
    temp_ctx: Option<&TemperatureContext>,
    caller: &str,
) {
    assert_eq!(flux.len(), n_e, "flux length must match y_obs");
    assert_eq!(
        density_indices.len(),
        cross_sections.len(),
        "density_indices length must match cross_sections length, got {} density indices and {} cross sections",
        density_indices.len(),
        cross_sections.len(),
    );
    for (k, sigma) in cross_sections.iter().enumerate() {
        assert_eq!(
            sigma.len(),
            n_e,
            "cross_sections[{}] length ({}) must match energy grid length ({})",
            k,
            sigma.len(),
            n_e,
        );
    }

    // Validate that every free parameter is referenced by density_indices
    // or the temperature context.  Free parameters with no analytical
    // gradient mapping will never move, causing silent convergence to the
    // initial guess.
    let free_set: std::collections::HashSet<usize> = params.free_indices().into_iter().collect();
    let mut mapped_set: std::collections::HashSet<usize> =
        density_indices.iter().copied().collect();
    if let Some(ctx) = temp_ctx {
        mapped_set.insert(ctx.temperature_index);
    }
    let unmapped: Vec<usize> = free_set.difference(&mapped_set).copied().collect();
    assert!(
        unmapped.is_empty(),
        "{caller}: free parameters {unmapped:?} are not mapped by density_indices \
         or temperature_index; analytical gradient is zero for these params. \
         Use poisson_fit (FD) for mixed parameter sets.",
    );
}

/// Build a mapping from each free parameter index to the list of isotope
/// indices it controls.
///
/// `param_isotopes[j]` contains the isotope indices `k` such that
/// `density_indices[k] == free_indices[j]`.  This is precomputed once
/// outside the iteration loop since `free_indices` and `density_indices`
/// are invariant during optimization.
fn build_param_isotope_map(free_indices: &[usize], density_indices: &[usize]) -> Vec<Vec<usize>> {
    free_indices
        .iter()
        .map(|&pi| {
            density_indices
                .iter()
                .enumerate()
                .filter(|&(_, &di)| di == pi)
                .map(|(k, _)| k)
                .collect()
        })
        .collect()
}

/// Compute the Beer-Lambert transmission T(E) = exp(−Σₖ nₖ · σₖ(E))
/// from cross-sections and current parameter values.
///
/// `density_indices[k]` maps isotope k to the parameter index, so the
/// density lookup is correct even when the parameter vector contains
/// additional nuisance parameters before/after the densities.
fn compute_transmission(
    n_e: usize,
    xs: &[Vec<f64>],
    density_indices: &[usize],
    all_vals: &[f64],
) -> Vec<f64> {
    debug_assert!(
        density_indices.iter().all(|&idx| idx < all_vals.len()),
        "density_indices out of bounds for parameter vector"
    );
    (0..n_e)
        .map(|e| {
            let mut neg_opt = 0.0f64;
            for (k, xs_k) in xs.iter().enumerate() {
                let density = all_vals[density_indices[k]];
                neg_opt -= density * xs_k[e];
            }
            neg_opt.exp()
        })
        .collect()
}

/// Inputs for `compute_analytic_gradient` bundled to satisfy clippy's
/// too-many-arguments lint without suppression.
struct AnalyticGradientCtx<'a> {
    y_obs: &'a [f64],
    y_model: &'a [f64],
    flux: &'a [f64],
    t_now: &'a [f64],
    xs: &'a [Vec<f64>],
    param_isotopes: &'a [Vec<usize>],
    temp_free_pos: Option<usize>,
    density_indices: &'a [usize],
    all_vals: &'a [f64],
    dxs_dt: &'a [Vec<f64>],
}

/// Compute the analytical gradient of the Poisson NLL w.r.t. free parameters.
///
/// Density gradient:  ∂NLL/∂nₖ = Σ_E [(1 − y_obs/Y) · (−Φ · σₖ · T)]
/// Temperature gradient: ∂NLL/∂T = Σ_E [(1 − y_obs/Y) · Φ · T · (−Σₖ nₖ · ∂σₖ/∂T)]
///
/// Both use the smooth `poisson_nll_grad_term` for the residual factor,
/// consistent with the C¹ extrapolation in `poisson_nll_term` (#109.2).
fn compute_analytic_gradient(ctx: &AnalyticGradientCtx<'_>) -> Vec<f64> {
    let mut grad: Vec<f64> = ctx
        .param_isotopes
        .iter()
        .map(|iso_indices| {
            ctx.y_obs
                .iter()
                .zip(ctx.y_model.iter())
                .zip(ctx.flux.iter())
                .zip(ctx.t_now.iter())
                .enumerate()
                .map(|(e, (((&obs, &ym), &phi), &t))| {
                    let residual_factor = poisson_nll_grad_term(obs, ym);
                    let sigma_sum: f64 = iso_indices.iter().map(|&k| ctx.xs[k][e]).sum();
                    residual_factor * phi * t * (-sigma_sum)
                })
                .sum::<f64>()
        })
        .collect();

    // Temperature gradient: ∂NLL/∂T = Σ_E [(1 − y_obs/Y) · Φ · T · (−Σₖ nₖ · ∂σₖ/∂T)]
    if let Some(pos) = ctx.temp_free_pos {
        debug_assert!(
            !ctx.dxs_dt.is_empty(),
            "dxs_dt must be non-empty when temperature fitting is enabled"
        );
        let temp_grad: f64 = ctx
            .y_obs
            .iter()
            .zip(ctx.y_model.iter())
            .zip(ctx.flux.iter())
            .zip(ctx.t_now.iter())
            .enumerate()
            .map(|(e, (((&obs, &ym), &phi), &t))| {
                let residual_factor = poisson_nll_grad_term(obs, ym);
                let dsigma_sum: f64 = (0..ctx.density_indices.len())
                    .map(|k| {
                        let density = ctx.all_vals[ctx.density_indices[k]];
                        density * ctx.dxs_dt[k][e]
                    })
                    .sum();
                residual_factor * phi * t * (-dsigma_sum)
            })
            .sum();
        grad[pos] = temp_grad;
    }

    grad
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
/// Optimization result with final NLL, parameters, and convergence status.
pub fn poisson_fit(
    model: &dyn FitModel,
    y_obs: &[f64],
    params: &mut ParameterSet,
    config: &PoissonConfig,
) -> PoissonResult {
    if let Some(result) = try_early_return_fixed(model, y_obs, params) {
        return result;
    }

    // Scratch buffers reused across the entire optimization loop to avoid
    // per-iteration allocations inside compute_gradient (N_free+1 calls)
    // and backtracking_line_search (up to 50 calls).
    let mut all_vals_buf = Vec::with_capacity(params.params.len());
    let mut free_vals_buf = Vec::with_capacity(params.n_free());
    let mut old_free_buf: Vec<f64> = Vec::with_capacity(params.n_free());
    let mut trial_free_buf: Vec<f64> = Vec::with_capacity(params.n_free());

    params.all_values_into(&mut all_vals_buf);
    let y_model = model.evaluate(&all_vals_buf);
    let mut nll = poisson_nll(y_obs, &y_model);

    // Guard: if the initial NLL is non-finite, bail out immediately rather
    // than entering the optimization loop with garbage values.
    if !nll.is_finite() {
        return PoissonResult {
            nll,
            iterations: 0,
            converged: false,
            params: params.all_values(),
        };
    }

    let mut converged = false;
    let mut iter = 0;

    for _ in 0..config.max_iter {
        iter += 1;

        // Compute gradient
        let grad = compute_gradient(model, params, y_obs, config.fd_step, &mut all_vals_buf);

        // Check gradient norm for convergence
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tol_param {
            converged = true;
            break;
        }

        // Backtracking line search with projected gradient.
        //
        // Scale the initial step by the gradient norm so that the first trial
        // moves parameters by approximately `step_size` regardless of how large
        // the gradient is.  Without this normalisation, high-count Poisson data
        // produces a large gradient (∝ √I₀), the fixed alpha=step_size initial
        // step wildly overshoots, and 30 backtracking halvings are not enough to
        // recover—causing the line search to fail even far from the optimum.
        params.free_values_into(&mut free_vals_buf);
        old_free_buf.clear();
        old_free_buf.extend_from_slice(&free_vals_buf);
        let initial_alpha = config.step_size / grad_norm.max(1.0);

        match backtracking_line_search(
            model,
            params,
            y_obs,
            &old_free_buf,
            &grad,
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

        // Convergence check: step size in parameter space.
        // Using relative NLL change is unreliable for Poisson NLL — at high
        // photon counts the NLL is large (∝ I₀·n_bins) so even a productive
        // step has a tiny relative change.  Parameter displacement is a
        // scale-invariant and physically meaningful stopping criterion.
        params.free_values_into(&mut free_vals_buf);
        let step_norm: f64 = old_free_buf
            .iter()
            .zip(free_vals_buf.iter())
            .map(|(o, n)| (o - n).powi(2))
            .sum::<f64>()
            .sqrt();
        if step_norm < config.tol_param {
            converged = true;
            break;
        }
    }

    PoissonResult {
        nll,
        iterations: iter,
        converged,
        params: params.all_values(),
    }
}

/// Context for fitting temperature alongside densities in the analytical
/// Poisson optimizer.
///
/// When provided, `poisson_fit_analytic` recomputes cross-sections and their
/// temperature derivative at each iteration (via
/// `broadened_cross_sections_with_derivative`), adding the ∂NLL/∂T gradient
/// column so the optimizer moves temperature jointly with densities.
pub struct TemperatureContext {
    /// Index of the temperature parameter in the full parameter vector.
    pub temperature_index: usize,
    /// Resonance data for all isotopes (needed for recomputing σ at new T).
    pub resonance_data: Vec<ResonanceData>,
    /// Energy grid in eV (needed for recomputing σ at new T).
    pub energies: Vec<f64>,
    /// Optional instrument parameters (for resolution broadening during
    /// cross-section recomputation).
    pub instrument: Option<InstrumentParams>,
}

/// Run Poisson-likelihood optimization with an analytical gradient.
///
/// Equivalent to [`poisson_fit`] but replaces the finite-difference gradient
/// with the closed-form derivative of the Poisson NLL for the forward model
///
///   Y(E) = Φ(E) · T(n, E) + B(E),   T = exp(−Σₖ nₖ · σₖ(E))
///
/// Analytical gradient w.r.t. density nₖ:
///
///   ∂NLL/∂nₖ = Σ_E [(1 − y_obs(E)/Y(E)) · Φ(E) · T(E) · (−σₖ(E))]
///
/// This costs one forward evaluation per iteration (vs N_isotopes+1 for FD),
/// which is 3–5× faster for 2–4 isotopes at typical VENUS energy grids.
///
/// # Arguments
/// * `model`           — Forward model Y = Φ·T(θ) + B.
/// * `y_obs`           — Observed counts.
/// * `flux`            — Incident flux Φ(E) (from nuisance estimation).
/// * `cross_sections`  — Precomputed Doppler-broadened σₖ(E), one slice per isotope.
/// * `density_indices` — Maps isotope `k` → parameter index: the density for
///   isotope `k` is `params[density_indices[k]]`.  This decouples isotope
///   ordering from the parameter vector layout, supporting callers that include
///   additional nuisance parameters alongside densities.
/// * `params`          — Parameter set (modified in place).
/// * `config`          — Optimizer configuration.
/// * `temp_ctx`        — Optional temperature-fitting context.  When `Some`,
///   the optimizer recomputes cross-sections at the current temperature each
///   iteration and adds a ∂NLL/∂T gradient column.
// The 8th parameter (`temp_ctx`) extends an existing 7-param function with
// optional temperature-fitting context.  Bundling into a config struct would
// break all call sites for marginal gain; a single `Option` is the minimal
// extension that keeps the existing API intact.
#[allow(clippy::too_many_arguments)]
pub fn poisson_fit_analytic(
    model: &dyn FitModel,
    y_obs: &[f64],
    flux: &[f64],
    cross_sections: &[Vec<f64>],
    density_indices: &[usize],
    params: &mut ParameterSet,
    config: &PoissonConfig,
    temp_ctx: Option<&TemperatureContext>,
) -> PoissonResult {
    let n_e = y_obs.len();
    validate_analytic_inputs(
        n_e,
        flux,
        cross_sections,
        density_indices,
        params,
        temp_ctx,
        "poisson_fit_analytic",
    );

    if let Some(result) = try_early_return_fixed(model, y_obs, params) {
        return result;
    }

    let free_indices = params.free_indices();
    let param_isotopes = build_param_isotope_map(&free_indices, density_indices);

    // Position of the temperature parameter in the free-parameter vector.
    let temp_free_pos: Option<usize> = temp_ctx.as_ref().and_then(|ctx| {
        free_indices
            .iter()
            .position(|&fi| fi == ctx.temperature_index)
    });

    // Clone cross-sections only when temperature fitting is enabled (the
    // optimizer overwrites xs_owned each iteration with recomputed values).
    // When temperature is not being fitted, borrow the caller's data directly.
    let mut xs_owned: Vec<Vec<f64>> = if temp_ctx.is_some() {
        cross_sections.to_vec()
    } else {
        Vec::new()
    };
    // Temperature derivative dσ_k/dT — only allocated when fitting temperature.
    let mut dxs_dt: Vec<Vec<f64>> = if temp_ctx.is_some() {
        vec![vec![0.0; n_e]; density_indices.len()]
    } else {
        Vec::new()
    };

    // Scratch buffers reused across the entire optimization loop.
    let mut all_vals_buf = Vec::with_capacity(params.params.len());
    let mut free_vals_buf = Vec::with_capacity(params.n_free());
    let mut old_free_buf: Vec<f64> = Vec::with_capacity(params.n_free());
    let mut trial_free_buf: Vec<f64> = Vec::with_capacity(params.n_free());

    params.all_values_into(&mut all_vals_buf);
    let y_model = model.evaluate(&all_vals_buf);
    let mut nll = poisson_nll(y_obs, &y_model);

    // Guard: if the initial NLL is non-finite, bail out immediately rather
    // than entering the optimization loop with garbage values.
    if !nll.is_finite() {
        return PoissonResult {
            nll,
            iterations: 0,
            converged: false,
            params: params.all_values(),
        };
    }

    let mut converged = false;
    let mut iter = 0;

    for _ in 0..config.max_iter {
        iter += 1;

        // When fitting temperature, recompute cross-sections and their T
        // derivative at the current temperature.  This is expensive (3x
        // broadening) but necessary for an exact analytical gradient.
        if let Some(ctx) = &temp_ctx {
            params.all_values_into(&mut all_vals_buf);
            let t_current = all_vals_buf[ctx.temperature_index];
            let (xs_new, dxs_new) = transmission::broadened_cross_sections_with_derivative(
                &ctx.energies,
                &ctx.resonance_data,
                t_current,
                ctx.instrument.as_ref(),
            )
            .expect("poisson_fit_analytic: energy grid must be sorted ascending");
            xs_owned = xs_new;
            dxs_dt = dxs_new;
        }

        // Use owned cross-sections when temperature fitting (mutated each
        // iteration), otherwise borrow the caller's immutable data directly.
        let xs_ref: &[Vec<f64>] = if temp_ctx.is_some() {
            &xs_owned
        } else {
            cross_sections
        };

        // Evaluate the full model Y(E) and compute T(E) from Beer-Lambert.
        params.all_values_into(&mut all_vals_buf);
        let y_model_now = model.evaluate(&all_vals_buf);
        let t_now = compute_transmission(n_e, xs_ref, density_indices, &all_vals_buf);

        // Analytical gradient (density + optional temperature).
        let grad = compute_analytic_gradient(&AnalyticGradientCtx {
            y_obs,
            y_model: &y_model_now,
            flux,
            t_now: &t_now,
            xs: xs_ref,
            param_isotopes: &param_isotopes,
            temp_free_pos,
            density_indices,
            all_vals: &all_vals_buf,
            dxs_dt: &dxs_dt,
        });

        // Check gradient norm for convergence
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tol_param {
            converged = true;
            break;
        }

        // Backtracking line search with diagonal Fisher-information
        // preconditioning.
        //
        // When fitting temperature alongside densities, gradient components
        // differ by many orders of magnitude because parameter scales differ
        // (density ~0.001 vs temperature ~300 K) and the NLL curvature
        // w.r.t. each parameter differs enormously.  Vanilla gradient descent
        // normalises by ‖∇f‖, making the temperature step negligible.
        //
        // Fix: precondition by D = diag(1 / H_ii) where H_ii is the diagonal
        // of the Fisher information:
        //   H_{n_k, n_k} ≈ Σ_E (Φ σ_k T)² / Y
        //   H_{T, T}     ≈ Σ_E (Φ T Σ_k n_k ∂σ_k/∂T)² / Y
        //
        // This gives each parameter a Newton-like step size, handling the
        // extreme scale mismatch between density and temperature.
        // When temp_ctx is None, the preconditioner is still applied
        // (harmless: all density parameters have similar curvature).
        params.free_values_into(&mut free_vals_buf);
        old_free_buf.clear();
        old_free_buf.extend_from_slice(&free_vals_buf);

        // Compute diagonal Fisher information for preconditioning.
        let hessian_diag: Vec<f64> = param_isotopes
            .iter()
            .enumerate()
            .map(|(j, iso_indices)| {
                let is_temp = temp_free_pos == Some(j);
                y_model_now
                    .iter()
                    .zip(flux.iter())
                    .zip(t_now.iter())
                    .enumerate()
                    .map(|(e, ((&ym, &phi), &t))| {
                        // #125.2: Use ym.max(POISSON_EPSILON) instead of
                        // hard-returning 0.0, consistent with the gradient path
                        // which uses smooth extrapolation for ym <= epsilon.
                        // Hard-returning 0.0 loses curvature information.
                        let ym_safe = ym.max(POISSON_EPSILON);
                        let dy = if is_temp {
                            // ∂Y/∂T = Φ · T · (−Σ_k n_k · ∂σ_k/∂T)
                            let dsigma_sum: f64 = (0..density_indices.len())
                                .map(|k| {
                                    let density = all_vals_buf[density_indices[k]];
                                    density * dxs_dt[k][e]
                                })
                                .sum();
                            phi * t * (-dsigma_sum)
                        } else {
                            // ∂Y/∂n_k = −Φ · σ_k · T
                            let sigma_sum: f64 = iso_indices.iter().map(|&k| xs_ref[k][e]).sum();
                            phi * t * (-sigma_sum)
                        };
                        dy * dy / ym_safe
                    })
                    .sum::<f64>()
            })
            .collect();

        // search_dir = D * grad where D_ii = 1/H_ii (Newton scaling).
        let search_dir: Vec<f64> = grad
            .iter()
            .zip(hessian_diag.iter())
            .map(|(&g, &h)| {
                if h > PIVOT_FLOOR {
                    g / h
                } else {
                    // Pre-existing floor; when dy is extremely small, h can underflow.
                    // This falls back to unscaled gradient which is safe but slower to converge.
                    g
                }
            })
            .collect();
        let search_norm: f64 = search_dir.iter().map(|g| g * g).sum::<f64>().sqrt();
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
                // (params already restored by backtracking_line_search)
                break;
            }
        }

        // Convergence: parameter displacement (see poisson_fit for rationale).
        params.free_values_into(&mut free_vals_buf);
        let step_norm: f64 = old_free_buf
            .iter()
            .zip(free_vals_buf.iter())
            .map(|(o, n)| (o - n).powi(2))
            .sum::<f64>()
            .sqrt();
        if step_norm < config.tol_param {
            converged = true;
            break;
        }
    }

    PoissonResult {
        nll,
        iterations: iter,
        converged,
        params: params.all_values(),
    }
}

/// Convert transmission model + counts to a counts-based forward model.
///
/// Given:
///   Y_model = flux × T_model(θ) + background
///
/// This wraps a transmission model to predict observed counts.
pub struct CountsModel<'a> {
    /// Underlying transmission model.
    pub transmission_model: &'a dyn FitModel,
    /// Incident flux (counts per bin in open beam, after normalization).
    pub flux: &'a [f64],
    /// Background counts per bin.
    pub background: &'a [f64],
}

impl<'a> FitModel for CountsModel<'a> {
    fn evaluate(&self, params: &[f64]) -> Vec<f64> {
        let transmission = self.transmission_model.evaluate(params);
        transmission
            .iter()
            .zip(self.flux.iter())
            .zip(self.background.iter())
            .map(|((&t, &f), &b)| f * t + b)
            .collect()
    }
}

/// L-BFGS memory: stores last `m` correction pairs for inverse Hessian
/// approximation.
///
/// Reference: Nocedal & Wright, "Numerical Optimization", Algorithm 7.4
/// (two-loop recursion for limited-memory BFGS).
///
/// Uses a ring buffer to avoid shifting elements when the buffer is full.
// TODO(perf): consider flat buffer with stride-based indexing for cache locality.
// The current Vec<Vec<f64>> layout scatters s/y vectors across the heap; a single
// contiguous allocation with stride = n_free would improve L1/L2 cache utilization
// during the two-loop recursion, especially for large parameter vectors.
struct LbfgsbMemory {
    /// Maximum number of correction pairs to store.
    m: usize,
    /// Parameter difference vectors s_k = x_{k+1} - x_k (ring buffer).
    s: Vec<Vec<f64>>,
    /// Gradient difference vectors y_k = g_{k+1} - g_k (ring buffer).
    y: Vec<Vec<f64>>,
    /// Precomputed 1/(y_k · s_k) for each stored pair.
    rho: Vec<f64>,
    /// Number of currently stored pairs (<= m).
    len: usize,
    /// Next write position in the ring buffer.
    cursor: usize,
}

impl LbfgsbMemory {
    /// Create a new L-BFGS memory with capacity for `m` correction pairs.
    fn new(m: usize, n: usize) -> Self {
        Self {
            m,
            s: vec![vec![0.0; n]; m],
            y: vec![vec![0.0; n]; m],
            rho: vec![0.0; m],
            len: 0,
            cursor: 0,
        }
    }

    /// Store a new correction pair (s, y).
    ///
    /// The curvature condition y·s > 0 is enforced; pairs that violate it
    /// (e.g., near constraints or numerical noise) are silently discarded.
    fn push(&mut self, s_vec: Vec<f64>, y_vec: Vec<f64>) {
        let ys: f64 = y_vec
            .iter()
            .zip(s_vec.iter())
            .map(|(&yi, &si)| yi * si)
            .sum();
        if ys <= PIVOT_FLOOR {
            // Curvature condition violated — discard this pair.
            return;
        }
        let idx = self.cursor;
        self.s[idx] = s_vec;
        self.y[idx] = y_vec;
        self.rho[idx] = 1.0 / ys;
        self.cursor = (self.cursor + 1) % self.m;
        if self.len < self.m {
            self.len += 1;
        }
    }

    /// Apply the two-loop recursion (Algorithm 7.4, Nocedal & Wright) to
    /// compute H_k * q, where H_k is the L-BFGS approximation to the
    /// inverse Hessian.
    ///
    /// If no pairs are stored yet, returns q unchanged (steepest descent).
    fn apply_two_loop(&self, q: &[f64]) -> Vec<f64> {
        let mut r: Vec<f64> = q.to_vec();

        if self.len == 0 {
            return r;
        }

        let mut alpha = vec![0.0; self.len];

        // Backward pass: most recent to oldest.
        // Most recent is at ring index (cursor - 1 + m) % m,
        // second most recent at (cursor - 2 + m) % m, etc.
        for i in (0..self.len).rev() {
            let idx = (self.cursor + self.m - self.len + i) % self.m;
            let dot_sr: f64 = self.s[idx]
                .iter()
                .zip(r.iter())
                .map(|(&si, &ri)| si * ri)
                .sum();
            alpha[i] = self.rho[idx] * dot_sr;
            for (rj, &yj) in r.iter_mut().zip(self.y[idx].iter()) {
                *rj -= alpha[i] * yj;
            }
        }

        // Scale by initial Hessian approximation: gamma = (s·y) / (y·y)
        // using the most recent pair.
        let newest = (self.cursor + self.m - 1) % self.m;
        let sy: f64 = self.s[newest]
            .iter()
            .zip(self.y[newest].iter())
            .map(|(&si, &yi)| si * yi)
            .sum();
        let yy: f64 = self.y[newest].iter().map(|&yi| yi * yi).sum();
        if yy > PIVOT_FLOOR {
            let gamma = sy / yy;
            for ri in r.iter_mut() {
                *ri *= gamma;
            }
        }

        // Forward pass: oldest to most recent.
        for (i, &alpha_i) in alpha.iter().enumerate().take(self.len) {
            let idx = (self.cursor + self.m - self.len + i) % self.m;
            let dot_yr: f64 = self.y[idx]
                .iter()
                .zip(r.iter())
                .map(|(&yi, &ri)| yi * ri)
                .sum();
            let beta = self.rho[idx] * dot_yr;
            for (rj, &sj) in r.iter_mut().zip(self.s[idx].iter()) {
                *rj += (alpha_i - beta) * sj;
            }
        }

        r
    }
}

/// Run Poisson-likelihood optimization using L-BFGS-B (limited-memory BFGS
/// with box constraints).
///
/// Uses the same analytical gradient as [`poisson_fit_analytic`] but replaces
/// the diagonal Fisher preconditioning with a full L-BFGS inverse Hessian
/// Returns `true` if `direction` is a descent direction with respect to
/// `gradient`, i.e. the dot product `direction . gradient` is strictly
/// positive.
///
/// In the L-BFGS-B line search convention `x_new = x - alpha * search_dir`,
/// the search direction should point in the gradient direction for descent,
/// so a positive dot product indicates the step will reduce the objective.
///
/// Used by [`poisson_fit_lbfgsb`] to decide whether to keep the L-BFGS
/// direction or fall back to steepest descent (the projected gradient).
fn is_descent_direction(direction: &[f64], gradient: &[f64]) -> bool {
    let dot: f64 = direction
        .iter()
        .zip(gradient.iter())
        .map(|(&d, &g)| d * g)
        .sum();
    // Strictly positive: orthogonal (dot == 0) is NOT descent because the
    // step would not reduce the objective along the gradient.
    dot > 0.0
}

/// approximation.  This captures curvature across parameters (not just
/// per-parameter diagonal curvature), which is critical for problems with
/// correlated parameters such as joint density + temperature fitting.
///
/// The "B" (box constraints) is handled via projected gradient: gradient
/// components at active bounds are zeroed before computing the L-BFGS
/// direction, and the resulting step is projected back onto bounds.
///
/// Reference: Nocedal & Wright, "Numerical Optimization", Ch. 7 & 9.
///
/// # Arguments
/// * `model`           — Forward model Y = Phi * T(theta) + B.
/// * `y_obs`           — Observed counts.
/// * `flux`            — Incident flux Phi(E).
/// * `cross_sections`  — Precomputed Doppler-broadened sigma_k(E), one per isotope.
/// * `density_indices` — Maps isotope k -> parameter index for densities.
/// * `params`          — Parameter set (modified in place).
/// * `config`          — Optimizer configuration (including `lbfgsb_memory`).
/// * `temp_ctx`        — Optional temperature-fitting context.
#[allow(clippy::too_many_arguments)]
pub fn poisson_fit_lbfgsb(
    model: &dyn FitModel,
    y_obs: &[f64],
    flux: &[f64],
    cross_sections: &[Vec<f64>],
    density_indices: &[usize],
    params: &mut ParameterSet,
    config: &PoissonConfig,
    temp_ctx: Option<&TemperatureContext>,
) -> PoissonResult {
    let n_e = y_obs.len();
    assert!(
        config.lbfgsb_memory >= 1,
        "lbfgsb_memory must be >= 1, got {}",
        config.lbfgsb_memory,
    );
    validate_analytic_inputs(
        n_e,
        flux,
        cross_sections,
        density_indices,
        params,
        temp_ctx,
        "poisson_fit_lbfgsb",
    );

    if let Some(result) = try_early_return_fixed(model, y_obs, params) {
        return result;
    }

    let free_indices = params.free_indices();
    let n_free = free_indices.len();
    let param_isotopes = build_param_isotope_map(&free_indices, density_indices);

    // Position of the temperature parameter in the free-parameter vector.
    let temp_free_pos: Option<usize> = temp_ctx.as_ref().and_then(|ctx| {
        free_indices
            .iter()
            .position(|&fi| fi == ctx.temperature_index)
    });

    // Clone cross-sections only when temperature fitting is enabled.
    let mut xs_owned: Vec<Vec<f64>> = if temp_ctx.is_some() {
        cross_sections.to_vec()
    } else {
        Vec::new()
    };
    let mut dxs_dt: Vec<Vec<f64>> = if temp_ctx.is_some() {
        vec![vec![0.0; n_e]; density_indices.len()]
    } else {
        Vec::new()
    };

    let y_model = model.evaluate(&params.all_values());
    let mut nll = poisson_nll(y_obs, &y_model);

    // Guard: if the initial NLL is non-finite, bail out immediately rather
    // than entering the optimization loop with garbage values.
    if !nll.is_finite() {
        return PoissonResult {
            nll,
            iterations: 0,
            converged: false,
            params: params.all_values(),
        };
    }

    let mut converged = false;
    let mut iter = 0;

    // Initialize L-BFGS memory.
    let mut lbfgs = LbfgsbMemory::new(config.lbfgsb_memory, n_free);

    // Previous free parameters and gradient — needed for memory updates.
    let mut prev_free: Vec<f64> = params.free_values();
    let mut prev_grad: Option<Vec<f64>> = None;

    // Scratch buffers reused across the entire optimization loop to avoid
    // per-iteration allocations inside backtracking_line_search.
    let mut all_vals_buf = Vec::with_capacity(params.params.len());
    let mut free_vals_buf = Vec::with_capacity(params.n_free());
    let mut trial_free_buf: Vec<f64> = Vec::with_capacity(params.n_free());

    for _ in 0..config.max_iter {
        iter += 1;

        // ---- Temperature XS recomputation ----
        // Fill all_vals_buf once up front; reused by both the temperature
        // recomputation (if enabled) and the model evaluation below.
        params.all_values_into(&mut all_vals_buf);

        if let Some(ctx) = &temp_ctx {
            let t_current = all_vals_buf[ctx.temperature_index];
            let (xs_new, dxs_new) = transmission::broadened_cross_sections_with_derivative(
                &ctx.energies,
                &ctx.resonance_data,
                t_current,
                ctx.instrument.as_ref(),
            )
            .expect("poisson_fit_lbfgsb: energy grid must be sorted ascending");
            xs_owned = xs_new;
            dxs_dt = dxs_new;
        }

        // Use owned cross-sections when temperature fitting, otherwise
        // borrow the caller's immutable data directly.
        let xs_ref: &[Vec<f64>] = if temp_ctx.is_some() {
            &xs_owned
        } else {
            cross_sections
        };

        // ---- Evaluate model and compute transmission + gradient ----
        let y_model_now = model.evaluate(&all_vals_buf);
        let t_now = compute_transmission(n_e, xs_ref, density_indices, &all_vals_buf);
        let grad = compute_analytic_gradient(&AnalyticGradientCtx {
            y_obs,
            y_model: &y_model_now,
            flux,
            t_now: &t_now,
            xs: xs_ref,
            param_isotopes: &param_isotopes,
            temp_free_pos,
            density_indices,
            all_vals: &all_vals_buf,
            dxs_dt: &dxs_dt,
        });

        // Snapshot current free-parameter values once (used by both
        // the L-BFGS memory update and the projected direction).
        let current_free = params.free_values();

        // ---- Update L-BFGS memory with previous step's pair ----
        if let Some(ref pg) = prev_grad {
            let s_vec: Vec<f64> = current_free
                .iter()
                .zip(prev_free.iter())
                .map(|(&c, &p)| c - p)
                .collect();
            let y_vec: Vec<f64> = grad
                .iter()
                .zip(pg.iter())
                .map(|(&gc, &gp)| gc - gp)
                .collect();
            lbfgs.push(s_vec, y_vec);
        }

        // ---- Projected L-BFGS direction ----
        //
        // Identify active bounds: parameter at lower bound with positive
        // gradient (wants to go more negative, but can't), or at upper
        // bound with negative gradient.

        // Helper: zero components of `vec` at active bounds.  A bound is
        // "active" when the parameter sits on the bound and the gradient
        // points into the infeasible half-space.
        let zero_active_bounds = |vec: &mut [f64], g: &[f64]| {
            for (j, &fi) in free_indices.iter().enumerate() {
                let p = &params.params[fi];
                let at_lower = (p.value - p.lower).abs() < PIVOT_FLOOR && g[j] > 0.0;
                let at_upper = (p.upper - p.value).abs() < PIVOT_FLOOR && g[j] < 0.0;
                if at_lower || at_upper {
                    vec[j] = 0.0;
                }
            }
        };

        let mut projected_grad = grad.clone();
        zero_active_bounds(&mut projected_grad, &grad);

        // ---- Projected gradient norm convergence check ----
        // Use the projected gradient norm (KKT condition): if the projected
        // gradient is zero we are at a constrained optimum, regardless of how
        // large the raw gradient is at active bounds.
        let proj_grad_norm: f64 = projected_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if proj_grad_norm < config.tol_param {
            converged = true;
            break;
        }

        // Apply two-loop recursion to get search direction.
        let mut direction = lbfgs.apply_two_loop(&projected_grad);

        // Zero out direction components at active bounds.
        zero_active_bounds(&mut direction, &grad);

        // Ensure descent direction: if direction · gradient <= 0, the
        // L-BFGS direction is not a descent direction (ascending or
        // orthogonal to the gradient); fall back to steepest descent.
        // (The search_dir convention in backtracking_line_search is:
        //  x_new = x - alpha * search_dir, so search_dir should point
        //  in the gradient direction for descent.)
        if !is_descent_direction(&direction, &grad) {
            // L-BFGS direction is not a descent direction; fall back.
            direction = projected_grad.clone();
        }

        // ---- Line search ----
        //
        // L-BFGS-B memory update protocol:
        //   1. `prev_free` is saved BEFORE the line search so that
        //      s_k = x_{k+1} - x_k uses the pre-step position.
        //   2. `prev_grad` is saved from the CURRENT gradient (before the
        //      line search moves parameters) so that y_k = g_{k+1} - g_k
        //      uses the gradient at the pre-step point.
        //   3. At the top of the NEXT iteration, the memory update computes
        //      s_k and y_k using the new (post-step) position and gradient
        //      minus these saved values.
        let search_norm: f64 = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
        let initial_alpha = config.step_size / search_norm.max(1.0);

        prev_free = current_free.clone();
        let old_free = current_free;
        prev_grad = Some(grad.clone());

        match backtracking_line_search(
            model,
            params,
            y_obs,
            &old_free,
            &direction,
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
                // Line search exhausted; params restored by backtracking_line_search.
                break;
            }
        }

        // ---- Parameter displacement convergence check ----
        params.free_values_into(&mut free_vals_buf);
        let step_norm: f64 = old_free
            .iter()
            .zip(free_vals_buf.iter())
            .map(|(o, n)| (o - n).powi(2))
            .sum::<f64>()
            .sqrt();
        if step_norm < config.tol_param {
            converged = true;
            break;
        }
    }

    PoissonResult {
        nll,
        iterations: iter,
        converged,
        params: params.all_values(),
    }
}

// TODO(validation): add test comparing L-BFGS-B vs analytic on a SAMMY-derived
// spectrum (e.g., U-238 single resonance at 6.674 eV with realistic noise).
// This would verify that both optimizers recover the same density/temperature
// to the same tolerance, catching any divergence in the shared gradient path.
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
        fn evaluate(&self, params: &[f64]) -> Vec<f64> {
            let b = params[0]; // "density"
            self.x
                .iter()
                .zip(self.flux.iter())
                .map(|(&xi, &fi)| fi * (-b * xi).exp())
                .collect()
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
        let y_obs = model.evaluate(&[true_b]);

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("b", 1.0), // Initial guess 2× off
        ]);

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default());

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

        let y_obs = model.evaluate(&[true_b]);

        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", 0.1)]);

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default());

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

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default());

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
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                vec![params[0]; 3]
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
        let result = counts_model.evaluate(&[0.5]);
        assert!((result[0] - 55.0).abs() < 1e-10);
        assert!((result[1] - 110.0).abs() < 1e-10);
        assert!((result[2] - 165.0).abs() < 1e-10);
    }

    // ---- Tests for poisson_fit_analytic ----

    #[test]
    fn test_analytic_matches_fd_result() {
        // Both poisson_fit (finite-difference) and poisson_fit_analytic should
        // converge to the same answer on a clean exponential model.
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let true_b = 0.5;
        let flux: Vec<f64> = vec![1000.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[true_b]);
        let cross_sections = vec![x.clone()]; // σ(E) = x

        // FD path
        let mut params_fd = ParameterSet::new(vec![FitParameter::non_negative("b", 1.0)]);
        let res_fd = poisson_fit(&model, &y_obs, &mut params_fd, &PoissonConfig::default());

        // Analytic path
        let mut params_an = ParameterSet::new(vec![FitParameter::non_negative("b", 1.0)]);
        let res_an = poisson_fit_analytic(
            &model,
            &y_obs,
            &flux,
            &cross_sections,
            &[0],
            &mut params_an,
            &PoissonConfig::default(),
            None,
        );

        assert!(res_fd.converged, "FD did not converge");
        assert!(res_an.converged, "Analytic did not converge");
        assert!(
            (res_fd.params[0] - res_an.params[0]).abs() < 1e-4,
            "FD={}, Analytic={} should agree",
            res_fd.params[0],
            res_an.params[0],
        );
        // Analytic should use fewer or equal iterations (exact gradient).
        assert!(
            res_an.iterations <= res_fd.iterations,
            "Analytic ({} iters) should not need more iterations than FD ({})",
            res_an.iterations,
            res_fd.iterations,
        );
    }

    #[test]
    fn test_analytic_two_isotopes() {
        // Two-isotope model: Y = Φ · exp(−n₁·σ₁ − n₂·σ₂)
        let n_e = 30;
        let sigma1: Vec<f64> = (0..n_e).map(|i| 1.0 + 0.1 * i as f64).collect();
        let sigma2: Vec<f64> = (0..n_e).map(|i| 0.5 + 0.05 * (n_e - i) as f64).collect();
        let flux: Vec<f64> = vec![500.0; n_e];
        let true_n1 = 0.3;
        let true_n2 = 0.7;

        // Forward model using both cross-sections
        struct TwoIsotopeModel {
            sigma1: Vec<f64>,
            sigma2: Vec<f64>,
            flux: Vec<f64>,
        }
        impl FitModel for TwoIsotopeModel {
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                let (n1, n2) = (params[0], params[1]);
                self.sigma1
                    .iter()
                    .zip(self.sigma2.iter())
                    .zip(self.flux.iter())
                    .map(|((&s1, &s2), &f)| f * (-n1 * s1 - n2 * s2).exp())
                    .collect()
            }
        }

        let model = TwoIsotopeModel {
            sigma1: sigma1.clone(),
            sigma2: sigma2.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[true_n1, true_n2]);

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("n1", 0.1),
            FitParameter::non_negative("n2", 0.1),
        ]);

        let result = poisson_fit_analytic(
            &model,
            &y_obs,
            &flux,
            &[sigma1, sigma2],
            &[0, 1],
            &mut params,
            &PoissonConfig::default(),
            None,
        );

        assert!(result.converged, "Two-isotope analytic did not converge");
        assert!(
            (result.params[0] - true_n1).abs() / true_n1 < 0.01,
            "n1: fitted={}, true={}",
            result.params[0],
            true_n1,
        );
        assert!(
            (result.params[1] - true_n2).abs() / true_n2 < 0.01,
            "n2: fitted={}, true={}",
            result.params[1],
            true_n2,
        );
    }

    #[test]
    fn test_analytic_zero_density_convergence() {
        // True density is ~0; verify the optimizer converges near zero
        // without NaN or divergence.
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let flux: Vec<f64> = vec![1000.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };
        // True b = 0 → y_obs = flux (constant)
        let y_obs = model.evaluate(&[0.0]);
        let cross_sections = vec![x.clone()];

        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", 0.5)]);

        let result = poisson_fit_analytic(
            &model,
            &y_obs,
            &flux,
            &cross_sections,
            &[0],
            &mut params,
            &PoissonConfig::default(),
            None,
        );

        assert!(result.converged, "Zero-density fit did not converge");
        assert!(
            result.params[0] < 0.01,
            "b should be ~0, got {}",
            result.params[0],
        );
        assert!(
            result.params[0] >= 0.0,
            "b must be non-negative, got {}",
            result.params[0],
        );
    }

    #[test]
    fn test_analytic_nonzero_background() {
        // Verify the analytical gradient is correct when background B ≠ 0.
        // Forward model: Y(E) = Φ(E)·exp(−b·σ(E)) + B(E)
        let n_e = 20;
        let sigma: Vec<f64> = (0..n_e).map(|i| 1.0 + 0.2 * i as f64).collect();
        let flux: Vec<f64> = vec![500.0; n_e];
        let background: Vec<f64> = vec![20.0; n_e]; // 4% of flux
        let true_b = 0.4;

        // Transmission-only model (no background); CountsModel adds flux+bg.
        struct PureTransmission {
            sigma: Vec<f64>,
        }
        impl FitModel for PureTransmission {
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                let b = params[0];
                self.sigma.iter().map(|&s| (-b * s).exp()).collect()
            }
        }

        let t_model = PureTransmission {
            sigma: sigma.clone(),
        };
        let counts_model = CountsModel {
            transmission_model: &t_model,
            flux: &flux,
            background: &background,
        };

        // Generate observed counts Y = Φ·T + B at true density.
        let y_obs = counts_model.evaluate(&[true_b]);

        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", 0.2)]);

        // Background adds a floor to Y, weakening the gradient signal.
        // Allow more iterations than the default to ensure convergence
        // across platforms with different FP rounding behavior.
        let config = PoissonConfig {
            max_iter: 500,
            ..PoissonConfig::default()
        };

        let result = poisson_fit_analytic(
            &counts_model,
            &y_obs,
            &flux,
            &[sigma],
            &[0],
            &mut params,
            &config,
            None,
        );

        assert!(
            result.converged,
            "Nonzero-background fit did not converge after {} iters",
            result.iterations,
        );
        assert!(
            (result.params[0] - true_b).abs() / true_b < 0.01,
            "With B≠0: fitted={}, true={}, error={:.2}%",
            result.params[0],
            true_b,
            (result.params[0] - true_b).abs() / true_b * 100.0,
        );
    }

    #[test]
    fn test_poisson_analytic_temperature_recovery() {
        // Fit density + temperature jointly using poisson_fit_analytic with
        // TemperatureContext.  Synthetic data generated at T=300K; initial
        // guess T=200K.  Verify both converge to within 0.5%.
        use nereids_core::types::Isotope;
        use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};
        use nereids_physics::transmission;

        let resonance_data = vec![nereids_endf::resonance::ResonanceData {
            isotope: Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 6.674,
                        j: 0.5,
                        gn: 1.493e-3,
                        gg: 23.0e-3,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
            }],
        }];

        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();
        let n_e = energies.len();
        let true_density = 0.0005;
        let true_temp = 300.0;
        let flux: Vec<f64> = vec![10000.0; n_e];

        // Generate synthetic counts at true parameters.
        let xs_true = transmission::broadened_cross_sections(
            &energies,
            &resonance_data,
            true_temp,
            None,
            None,
        )
        .unwrap();
        let y_obs: Vec<f64> = (0..n_e)
            .map(|e| {
                let t = (-true_density * xs_true[0][e]).exp();
                flux[e] * t
            })
            .collect();

        // Forward model: Y = Φ · exp(−n · σ(E, T))
        // The model recomputes σ from the TransmissionFitModel path, but
        // for the analytical optimizer we provide precomputed σ + ∂σ/∂T
        // via TemperatureContext.
        //
        // We use a simple inline model that reads density and temperature
        // from the parameter vector and evaluates the full physics.
        use nereids_physics::transmission::SampleParams;
        struct PhysicsCountsModel {
            energies: Vec<f64>,
            resonance_data: Vec<nereids_endf::resonance::ResonanceData>,
            flux: Vec<f64>,
        }
        impl FitModel for PhysicsCountsModel {
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                let density = params[0];
                let temperature = params[1];
                let sample = SampleParams {
                    temperature_k: temperature,
                    isotopes: vec![(self.resonance_data[0].clone(), density)],
                };
                let transmission =
                    transmission::forward_model(&self.energies, &sample, None).unwrap();
                transmission
                    .iter()
                    .zip(self.flux.iter())
                    .map(|(&t, &f)| f * t)
                    .collect()
            }
        }

        let model = PhysicsCountsModel {
            energies: energies.clone(),
            resonance_data: resonance_data.clone(),
            flux: flux.clone(),
        };

        // Initial cross-sections at the initial guess temperature.
        let init_temp = 200.0;
        let xs_init = transmission::broadened_cross_sections(
            &energies,
            &resonance_data,
            init_temp,
            None,
            None,
        )
        .unwrap();

        // params[0] = density, params[1] = temperature
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.001),
            FitParameter {
                name: "temperature".into(),
                value: init_temp,
                lower: 50.0,
                upper: 1000.0,
                fixed: false,
            },
        ]);

        let temp_ctx = TemperatureContext {
            temperature_index: 1,
            resonance_data: resonance_data.clone(),
            energies: energies.clone(),
            instrument: None,
        };

        // Tight tolerance and many iterations: density and temperature are
        // correlated, so projected gradient descent needs many small steps to
        // navigate the curved valley in NLL space.
        let config = PoissonConfig {
            max_iter: 5000,
            tol_param: 1e-12,
            ..PoissonConfig::default()
        };

        let result = poisson_fit_analytic(
            &model,
            &y_obs,
            &flux,
            &xs_init,
            &[0],
            &mut params,
            &config,
            Some(&temp_ctx),
        );

        assert!(
            result.converged,
            "Temperature+density fit did not converge after {} iterations",
            result.iterations,
        );
        let fitted_density = result.params[0];
        let fitted_temp = result.params[1];
        assert!(
            (fitted_density - true_density).abs() / true_density < 0.01,
            "density: fitted={}, true={}, error={:.2}%",
            fitted_density,
            true_density,
            (fitted_density - true_density).abs() / true_density * 100.0,
        );
        assert!(
            (fitted_temp - true_temp).abs() / true_temp < 0.01,
            "temperature: fitted={:.1}, true={:.1}, error={:.2}%",
            fitted_temp,
            true_temp,
            (fitted_temp - true_temp).abs() / true_temp * 100.0,
        );
    }

    // ---- Edge-case tests for issue #125 ----

    #[test]
    fn test_poisson_nll_term_negative_model() {
        // #125.6: Negative model prediction should use smooth extrapolation,
        // not produce NaN or panic.
        let nll = poisson_nll_term(10.0, -5.0);
        assert!(
            nll.is_finite(),
            "NLL should be finite for negative model, got {nll}"
        );

        // Should be larger than the NLL at epsilon (penalty grows for negative mdl).
        let nll_at_eps = poisson_nll_term(10.0, POISSON_EPSILON);
        assert!(
            nll > nll_at_eps,
            "NLL at mdl=-5 ({nll}) should exceed NLL at epsilon ({nll_at_eps})"
        );
    }

    #[test]
    fn test_poisson_nll_term_zero_obs() {
        // Zero observed counts: NLL = mdl (no log term).
        let nll = poisson_nll_term(0.0, 10.0);
        assert!(
            (nll - 10.0).abs() < 1e-10,
            "NLL(obs=0, mdl=10) should be 10.0, got {nll}"
        );

        // Extrapolation region with obs=0: should still have upward curvature.
        let nll_neg = poisson_nll_term(0.0, -1.0);
        let nll_zero = poisson_nll_term(0.0, 0.0);
        assert!(
            nll_neg > nll_zero,
            "NLL should grow as mdl goes more negative: nll(-1)={nll_neg} vs nll(0)={nll_zero}"
        );
    }

    #[test]
    fn test_all_fixed_params_nan_model_poisson() {
        // #125.1: All-fixed parameters with NaN model → converged=false.
        struct NanModel;
        impl FitModel for NanModel {
            fn evaluate(&self, _params: &[f64]) -> Vec<f64> {
                vec![f64::NAN; 5]
            }
        }

        let y_obs = vec![10.0; 5];
        let mut params = ParameterSet::new(vec![FitParameter::fixed("a", 1.0)]);

        let result = poisson_fit(&NanModel, &y_obs, &mut params, &PoissonConfig::default());

        assert!(
            !result.converged,
            "All-fixed NaN model should not converge in Poisson fit"
        );
        assert!(!result.nll.is_finite(), "NLL should be non-finite");
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_all_fixed_params_nan_model_poisson_analytic() {
        // #125.1: Exercise the NaN guard in poisson_fit_analytic (not just poisson_fit).
        // When all parameters are fixed and the model produces NaN, the result
        // must report converged=false.
        struct NanModel;
        impl FitModel for NanModel {
            fn evaluate(&self, _params: &[f64]) -> Vec<f64> {
                vec![f64::NAN; 5]
            }
        }

        let y_obs = vec![10.0; 5];
        let flux = vec![1000.0; 5];
        // One isotope with trivial cross-sections.
        let cross_sections = vec![vec![1.0; 5]];
        let density_indices = vec![0];
        let mut params = ParameterSet::new(vec![FitParameter::fixed("density", 0.5)]);

        let result = poisson_fit_analytic(
            &NanModel,
            &y_obs,
            &flux,
            &cross_sections,
            &density_indices,
            &mut params,
            &PoissonConfig::default(),
            None,
        );

        assert!(
            !result.converged,
            "All-fixed NaN model should not converge in poisson_fit_analytic"
        );
        assert!(!result.nll.is_finite(), "NLL should be non-finite");
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_all_fixed_params_good_model_poisson() {
        // #125.1: All-fixed parameters with a valid model → converged=true.
        let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let flux: Vec<f64> = vec![100.0; x.len()];
        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[0.5]);

        let mut params = ParameterSet::new(vec![FitParameter::fixed("b", 0.5)]);

        let result = poisson_fit(&model, &y_obs, &mut params, &PoissonConfig::default());

        assert!(
            result.converged,
            "All-fixed good model should converge in Poisson fit"
        );
        assert!(result.nll.is_finite(), "NLL should be finite");
        assert_eq!(result.iterations, 0);
    }

    // ---- Tests for poisson_fit_lbfgsb ----

    #[test]
    fn test_all_fixed_params_nan_model_poisson_lbfgsb() {
        // Exercise the NaN guard in poisson_fit_lbfgsb.
        // When all parameters are fixed and the model produces NaN, the result
        // must report converged=false.
        struct NanModel;
        impl FitModel for NanModel {
            fn evaluate(&self, _params: &[f64]) -> Vec<f64> {
                vec![f64::NAN; 5]
            }
        }

        let y_obs = vec![10.0; 5];
        let flux = vec![1000.0; 5];
        let cross_sections = vec![vec![1.0; 5]];
        let density_indices = vec![0];
        let mut params = ParameterSet::new(vec![FitParameter::fixed("density", 0.5)]);

        let result = poisson_fit_lbfgsb(
            &NanModel,
            &y_obs,
            &flux,
            &cross_sections,
            &density_indices,
            &mut params,
            &PoissonConfig::default(),
            None,
        );

        assert!(
            !result.converged,
            "All-fixed NaN model should not converge in poisson_fit_lbfgsb"
        );
        assert!(!result.nll.is_finite(), "NLL should be non-finite");
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_lbfgsb_matches_analytic_result() {
        // Both poisson_fit_analytic and poisson_fit_lbfgsb should converge
        // to the same answer on a clean exponential model.
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let true_b = 0.5;
        let flux: Vec<f64> = vec![1000.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[true_b]);
        let cross_sections = vec![x.clone()];

        // Analytic path
        let mut params_an = ParameterSet::new(vec![FitParameter::non_negative("b", 1.0)]);
        let res_an = poisson_fit_analytic(
            &model,
            &y_obs,
            &flux,
            &cross_sections,
            &[0],
            &mut params_an,
            &PoissonConfig::default(),
            None,
        );

        // L-BFGS-B path
        let mut params_lb = ParameterSet::new(vec![FitParameter::non_negative("b", 1.0)]);
        let res_lb = poisson_fit_lbfgsb(
            &model,
            &y_obs,
            &flux,
            &cross_sections,
            &[0],
            &mut params_lb,
            &PoissonConfig::default(),
            None,
        );

        assert!(res_an.converged, "Analytic did not converge");
        assert!(res_lb.converged, "L-BFGS-B did not converge");
        assert!(
            (res_an.params[0] - res_lb.params[0]).abs() < 1e-4,
            "Analytic={}, L-BFGS-B={} should agree",
            res_an.params[0],
            res_lb.params[0],
        );
        // Note: L-BFGS-B is not guaranteed to use fewer iterations than the
        // Fisher-preconditioned gradient on all problems (especially 1D where
        // it starts with no curvature history), so no iteration-count comparison
        // is asserted here.
    }

    #[test]
    fn test_lbfgsb_two_isotopes() {
        // Two-isotope model: Y = Phi * exp(-n1*sigma1 - n2*sigma2)
        let n_e = 30;
        let sigma1: Vec<f64> = (0..n_e).map(|i| 1.0 + 0.1 * i as f64).collect();
        let sigma2: Vec<f64> = (0..n_e).map(|i| 0.5 + 0.05 * (n_e - i) as f64).collect();
        let flux: Vec<f64> = vec![500.0; n_e];
        let true_n1 = 0.3;
        let true_n2 = 0.7;

        struct TwoIsotopeModel {
            sigma1: Vec<f64>,
            sigma2: Vec<f64>,
            flux: Vec<f64>,
        }
        impl FitModel for TwoIsotopeModel {
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                let (n1, n2) = (params[0], params[1]);
                self.sigma1
                    .iter()
                    .zip(self.sigma2.iter())
                    .zip(self.flux.iter())
                    .map(|((&s1, &s2), &f)| f * (-n1 * s1 - n2 * s2).exp())
                    .collect()
            }
        }

        let model = TwoIsotopeModel {
            sigma1: sigma1.clone(),
            sigma2: sigma2.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[true_n1, true_n2]);

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("n1", 0.1),
            FitParameter::non_negative("n2", 0.1),
        ]);

        let result = poisson_fit_lbfgsb(
            &model,
            &y_obs,
            &flux,
            &[sigma1, sigma2],
            &[0, 1],
            &mut params,
            &PoissonConfig::default(),
            None,
        );

        assert!(
            result.converged,
            "L-BFGS-B two-isotope did not converge after {} iters",
            result.iterations,
        );
        assert!(
            (result.params[0] - true_n1).abs() / true_n1 < 0.01,
            "n1: fitted={}, true={}",
            result.params[0],
            true_n1,
        );
        assert!(
            (result.params[1] - true_n2).abs() / true_n2 < 0.01,
            "n2: fitted={}, true={}",
            result.params[1],
            true_n2,
        );
    }

    #[test]
    fn test_lbfgsb_temperature_recovery() {
        // Joint density + temperature fit using L-BFGS-B.
        // This is the KEY benchmark: L-BFGS-B should converge in <200
        // iterations (vs 5000 for the diagonal Fisher preconditioner).
        use nereids_core::types::Isotope;
        use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};
        use nereids_physics::transmission;

        let resonance_data = vec![nereids_endf::resonance::ResonanceData {
            isotope: Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 6.674,
                        j: 0.5,
                        gn: 1.493e-3,
                        gg: 23.0e-3,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
            }],
        }];

        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();
        let n_e = energies.len();
        let true_density = 0.0005;
        let true_temp = 300.0;
        let flux: Vec<f64> = vec![10000.0; n_e];

        let xs_true = transmission::broadened_cross_sections(
            &energies,
            &resonance_data,
            true_temp,
            None,
            None,
        )
        .unwrap();
        let y_obs: Vec<f64> = (0..n_e)
            .map(|e| {
                let t = (-true_density * xs_true[0][e]).exp();
                flux[e] * t
            })
            .collect();

        use nereids_physics::transmission::SampleParams;
        struct PhysicsCountsModel {
            energies: Vec<f64>,
            resonance_data: Vec<nereids_endf::resonance::ResonanceData>,
            flux: Vec<f64>,
        }
        impl FitModel for PhysicsCountsModel {
            fn evaluate(&self, params: &[f64]) -> Vec<f64> {
                let density = params[0];
                let temperature = params[1];
                let sample = SampleParams {
                    temperature_k: temperature,
                    isotopes: vec![(self.resonance_data[0].clone(), density)],
                };
                let transmission =
                    transmission::forward_model(&self.energies, &sample, None).unwrap();
                transmission
                    .iter()
                    .zip(self.flux.iter())
                    .map(|(&t, &f)| f * t)
                    .collect()
            }
        }

        let model = PhysicsCountsModel {
            energies: energies.clone(),
            resonance_data: resonance_data.clone(),
            flux: flux.clone(),
        };

        let init_temp = 200.0;
        let xs_init = transmission::broadened_cross_sections(
            &energies,
            &resonance_data,
            init_temp,
            None,
            None,
        )
        .unwrap();

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.001),
            FitParameter {
                name: "temperature".into(),
                value: init_temp,
                lower: 50.0,
                upper: 1000.0,
                fixed: false,
            },
        ]);

        let temp_ctx = TemperatureContext {
            temperature_index: 1,
            resonance_data: resonance_data.clone(),
            energies: energies.clone(),
            instrument: None,
        };

        let config = PoissonConfig {
            max_iter: 200,
            tol_param: 1e-12,
            ..PoissonConfig::default()
        };

        let result = poisson_fit_lbfgsb(
            &model,
            &y_obs,
            &flux,
            &xs_init,
            &[0],
            &mut params,
            &config,
            Some(&temp_ctx),
        );

        assert!(
            result.converged,
            "L-BFGS-B temperature+density fit did not converge after {} iterations",
            result.iterations,
        );
        let fitted_density = result.params[0];
        let fitted_temp = result.params[1];
        assert!(
            (fitted_density - true_density).abs() / true_density < 0.01,
            "density: fitted={}, true={}, error={:.2}%",
            fitted_density,
            true_density,
            (fitted_density - true_density).abs() / true_density * 100.0,
        );
        assert!(
            (fitted_temp - true_temp).abs() / true_temp < 0.01,
            "temperature: fitted={:.1}, true={:.1}, error={:.2}%",
            fitted_temp,
            true_temp,
            (fitted_temp - true_temp).abs() / true_temp * 100.0,
        );
        // The `converged` check above already verifies the optimizer did not
        // hit the iteration limit, so no separate iteration-count assertion is
        // needed (it would be fragile across platforms and BLAS implementations).
    }

    #[test]
    fn test_lbfgsb_zero_density() {
        // True density is ~0; verify convergence near the lower bound
        // without NaN or divergence.
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let flux: Vec<f64> = vec![1000.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[0.0]);
        let cross_sections = vec![x.clone()];

        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", 0.5)]);

        let result = poisson_fit_lbfgsb(
            &model,
            &y_obs,
            &flux,
            &cross_sections,
            &[0],
            &mut params,
            &PoissonConfig::default(),
            None,
        );

        assert!(
            result.converged,
            "L-BFGS-B zero-density fit did not converge"
        );
        assert!(
            result.params[0] < 0.01,
            "b should be ~0, got {}",
            result.params[0],
        );
        assert!(
            result.params[0] >= 0.0,
            "b must be non-negative, got {}",
            result.params[0],
        );
    }

    #[test]
    fn test_lbfgsb_all_fixed() {
        // All parameters fixed — should return immediately with 0 iterations.
        let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let flux: Vec<f64> = vec![100.0; x.len()];
        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[0.5]);
        let cross_sections = vec![x.clone()];

        let mut params = ParameterSet::new(vec![FitParameter::fixed("b", 0.5)]);

        let result = poisson_fit_lbfgsb(
            &model,
            &y_obs,
            &flux,
            &cross_sections,
            &[0],
            &mut params,
            &PoissonConfig::default(),
            None,
        );

        assert!(
            result.converged,
            "All-fixed L-BFGS-B should converge immediately"
        );
        assert!(result.nll.is_finite(), "NLL should be finite");
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_lbfgsb_memory_one() {
        // Edge case: L-BFGS-B with m=1 (minimal memory).  Should still
        // converge, possibly needing more iterations than default m=7.
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let true_b = 0.5;
        let flux: Vec<f64> = vec![1000.0; x.len()];

        let model = ExponentialModel {
            x: x.clone(),
            flux: flux.clone(),
        };
        let y_obs = model.evaluate(&[true_b]);
        let cross_sections = vec![x.clone()];

        let mut params = ParameterSet::new(vec![FitParameter::non_negative("b", 1.0)]);

        let config = PoissonConfig {
            lbfgsb_memory: 1,
            max_iter: 500, // allow more iterations for minimal memory
            ..PoissonConfig::default()
        };

        let result = poisson_fit_lbfgsb(
            &model,
            &y_obs,
            &flux,
            &cross_sections,
            &[0],
            &mut params,
            &config,
            None,
        );

        assert!(
            result.converged,
            "L-BFGS-B with m=1 did not converge after {} iterations",
            result.iterations,
        );
        assert!(
            (result.params[0] - true_b).abs() / true_b < 0.01,
            "m=1: fitted={}, true={}, error={:.2}%",
            result.params[0],
            true_b,
            (result.params[0] - true_b).abs() / true_b * 100.0,
        );
    }

    #[test]
    fn test_is_descent_direction_opposite_vectors() {
        // Direction opposes gradient => positive dot product => descent.
        assert!(is_descent_direction(&[1.0, 0.0], &[1.0, 0.0]));
        assert!(is_descent_direction(&[0.5, 0.3], &[2.0, 1.0]));
    }

    #[test]
    fn test_is_descent_direction_orthogonal_not_descent() {
        // Orthogonal direction => dot product is zero => NOT descent.
        // The step would not reduce the objective along the gradient.
        assert!(!is_descent_direction(&[0.0, 1.0], &[1.0, 0.0]));
        assert!(!is_descent_direction(&[1.0, 0.0], &[0.0, 1.0]));
    }

    #[test]
    fn test_is_descent_direction_ascending_not_descent() {
        // Direction has negative dot product with gradient => ascending.
        assert!(!is_descent_direction(&[-1.0, 0.0], &[1.0, 0.0]));
        assert!(!is_descent_direction(&[-0.5, -0.3], &[2.0, 1.0]));
    }
}
