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
    /// Convergence tolerance on relative NLL change.
    pub tol_nll: f64,
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
            tol_nll: 1e-8,
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
/// Terms where y_model ≤ 0 are penalized heavily.
fn poisson_nll(y_obs: &[f64], y_model: &[f64]) -> f64 {
    y_obs
        .iter()
        .zip(y_model.iter())
        .map(|(&obs, &mdl)| {
            if mdl > 0.0 {
                mdl - obs * mdl.ln()
            } else {
                1e30 // Large penalty for non-positive model
            }
        })
        .sum()
}

/// Compute gradient of Poisson NLL by finite differences.
fn compute_gradient(
    model: &dyn FitModel,
    params: &mut ParameterSet,
    y_obs: &[f64],
    fd_step: f64,
) -> Vec<f64> {
    let base_model = model.evaluate(&params.all_values());
    let base_nll = poisson_nll(y_obs, &base_model);

    let free_indices = params.free_indices();
    let mut grad = vec![0.0; free_indices.len()];

    for (j, &idx) in free_indices.iter().enumerate() {
        let original = params.params[idx].value;
        let step = fd_step * (1.0 + original.abs());

        params.params[idx].value = original + step;
        params.params[idx].clamp();
        let actual_step = params.params[idx].value - original;

        if actual_step.abs() < 1e-30 {
            params.params[idx].value = original;
            continue;
        }

        let perturbed_model = model.evaluate(&params.all_values());
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
    let y_model = model.evaluate(&params.all_values());
    let mut nll = poisson_nll(y_obs, &y_model);
    let mut converged = false;
    let mut iter = 0;

    for _ in 0..config.max_iter {
        iter += 1;

        // Compute gradient
        let grad = compute_gradient(model, params, y_obs, config.fd_step);

        // Check gradient norm for convergence
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tol_nll {
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
        let old_free = params.free_values();
        let mut alpha = config.step_size / grad_norm.max(1.0);
        let mut accepted = false;

        for _ in 0..50 {
            // Trial step: x_new = project(x - α·∇f)
            let trial_free: Vec<f64> = old_free
                .iter()
                .zip(grad.iter())
                .map(|(&v, &g)| v - alpha * g)
                .collect();
            params.set_free_values(&trial_free);
            project(params);

            let trial_model = model.evaluate(&params.all_values());
            let trial_nll = poisson_nll(y_obs, &trial_model);

            // Armijo condition: f(x_new) <= f(x) - c·α·∇f·d
            let descent = grad
                .iter()
                .zip(old_free.iter())
                .zip(params.free_values().iter())
                .map(|((&g, &old), &new)| g * (old - new))
                .sum::<f64>();

            if trial_nll <= nll - config.armijo_c * descent {
                nll = trial_nll;
                accepted = true;
                break;
            }

            // Backtrack
            alpha *= config.backtrack;
        }

        if !accepted {
            params.set_free_values(&old_free);
            // Can't improve from this point; stop without claiming convergence.
            break;
        }

        // Convergence check: step size in parameter space.
        // Using relative NLL change is unreliable for Poisson NLL — at high
        // photon counts the NLL is large (∝ I₀·n_bins) so even a productive
        // step has a tiny relative change.  Parameter displacement is a
        // scale-invariant and physically meaningful stopping criterion.
        let step_norm: f64 = old_free
            .iter()
            .zip(params.free_values().iter())
            .map(|(o, n)| (o - n).powi(2))
            .sum::<f64>()
            .sqrt();
        if step_norm < config.tol_nll {
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
///   additional nuisance parameters alongside densities (via `CountsModel`'s
///   `density_param_range`).
/// * `params`          — Parameter set (modified in place).
/// * `config`          — Optimizer configuration.
pub fn poisson_fit_analytic(
    model: &dyn FitModel,
    y_obs: &[f64],
    flux: &[f64],
    cross_sections: &[Vec<f64>],
    density_indices: &[usize],
    params: &mut ParameterSet,
    config: &PoissonConfig,
) -> PoissonResult {
    let n_e = y_obs.len();
    assert_eq!(flux.len(), n_e, "flux length must match y_obs");
    assert_eq!(
        density_indices.len(),
        cross_sections.len(),
        "density_indices length ({}) must match cross_sections length ({})",
        density_indices.len(),
        cross_sections.len(),
    );

    let y_model = model.evaluate(&params.all_values());
    let mut nll = poisson_nll(y_obs, &y_model);
    let mut converged = false;
    let mut iter = 0;

    for _ in 0..config.max_iter {
        iter += 1;

        // Evaluate the full model Y(E) = Φ·T + B so we can form (1 - y_obs/Y).
        let y_model_now = model.evaluate(&params.all_values());

        // Compute T(E) directly from cross-sections and current densities:
        //   T(E) = exp(−Σₖ nₖ · σₖ(E))
        //
        // An earlier version derived T = Y/Φ, which is only correct when
        // background B = 0.  Computing T from the Beer-Lambert definition
        // is exact regardless of background and avoids a silent bias whenever
        // a caller supplies a CountsModel with nonzero B(E).
        //
        // `density_indices[k]` maps isotope k → parameter index, so the
        // density lookup is correct even when the parameter vector contains
        // additional nuisance parameters before/after the densities.
        let all_vals = params.all_values();
        let t_now: Vec<f64> = (0..n_e)
            .map(|e| {
                let mut neg_opt = 0.0f64;
                for (k, xs) in cross_sections.iter().enumerate() {
                    let density = all_vals[density_indices[k]];
                    if density > 0.0 {
                        neg_opt -= density * xs[e];
                    }
                }
                neg_opt.exp()
            })
            .collect();

        // Analytical gradient: ∂NLL/∂nₖ = Σ_E [(1 − y_obs/Y) · (−Φ · σₖ · T)]
        //
        // Derivation:  Y = Φ·T + B,  T = exp(−Σ nₖ σₖ)
        //   ∂Y/∂nₖ = Φ · ∂T/∂nₖ = −Φ · σₖ · T
        //   ∂NLL/∂nₖ = Σ_E (1 − y_obs/Y) · ∂Y/∂nₖ
        //
        // For each free parameter, we sum the cross-sections of every isotope
        // whose density maps to that parameter index.  Free parameters that
        // don't correspond to any density (nuisance params) get zero gradient
        // from this formula — callers should ensure all free params are
        // densities, or use `poisson_fit` (FD) for mixed parameter sets.
        let free_indices = params.free_indices();
        let grad: Vec<f64> = free_indices
            .iter()
            .map(|&param_idx| {
                y_obs
                    .iter()
                    .zip(y_model_now.iter())
                    .zip(flux.iter())
                    .zip(t_now.iter())
                    .enumerate()
                    .map(|(e, (((&obs, &ym), &phi), &t))| {
                        let ym_safe = ym.max(1e-30);
                        let residual_factor = 1.0 - obs / ym_safe;
                        // Sum σₖ for all isotopes mapped to this param index.
                        // Handles the common case (one isotope per param) and
                        // tied parameters (multiple isotopes sharing one density).
                        let sigma_sum: f64 = density_indices
                            .iter()
                            .enumerate()
                            .filter(|&(_, di)| *di == param_idx)
                            .map(|(k, _)| cross_sections[k][e])
                            .sum();
                        residual_factor * phi * t * (-sigma_sum)
                    })
                    .sum::<f64>()
            })
            .collect();

        // Check gradient norm for convergence
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tol_nll {
            converged = true;
            break;
        }

        // Backtracking line search — same fix as poisson_fit:
        // scale initial alpha by gradient norm for scale invariance.
        let old_free = params.free_values();
        let mut alpha = config.step_size / grad_norm.max(1.0);
        let mut accepted = false;

        for _ in 0..50 {
            let trial_free: Vec<f64> = old_free
                .iter()
                .zip(grad.iter())
                .map(|(&v, &g)| v - alpha * g)
                .collect();
            params.set_free_values(&trial_free);
            project(params);

            let trial_model = model.evaluate(&params.all_values());
            let trial_nll = poisson_nll(y_obs, &trial_model);

            let descent = grad
                .iter()
                .zip(old_free.iter())
                .zip(params.free_values().iter())
                .map(|((&g, &old), &new)| g * (old - new))
                .sum::<f64>();

            if trial_nll <= nll - config.armijo_c * descent {
                nll = trial_nll;
                accepted = true;
                break;
            }

            alpha *= config.backtrack;
        }

        if !accepted {
            params.set_free_values(&old_free);
            break;
        }

        // Convergence: parameter displacement (see poisson_fit for rationale).
        let step_norm: f64 = old_free
            .iter()
            .zip(params.free_values().iter())
            .map(|(o, n)| (o - n).powi(2))
            .sum::<f64>()
            .sqrt();
        if step_norm < config.tol_nll {
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
    pub flux: Vec<f64>,
    /// Background counts per bin.
    pub background: Vec<f64>,
    /// Indices into the full parameter vector for density parameters.
    /// Other parameters in the full vector are nuisance params.
    pub density_param_range: std::ops::Range<usize>,
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
        let counts_model = CountsModel {
            transmission_model: &t_model,
            flux: vec![100.0, 200.0, 300.0],
            background: vec![5.0, 10.0, 15.0],
            density_param_range: 0..1,
        };

        // T = 0.5 → counts = flux*0.5 + background
        let result = counts_model.evaluate(&[0.5]);
        assert!((result[0] - 55.0).abs() < 1e-10);
        assert!((result[1] - 110.0).abs() < 1e-10);
        assert!((result[2] - 165.0).abs() < 1e-10);
    }
}
