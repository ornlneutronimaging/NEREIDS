//! Proximal penalty support for ADMM-based spatial regularization.
//!
//! Provides [`ProximalPenalty`] for adding quadratic penalty terms to the
//! per-pixel fitting objective during TV-ADMM outer iterations.
//!
//! For the LM solver, [`ProximalModel`] wraps any [`FitModel`] and appends
//! pseudo-residual rows that encode the proximal penalty as extra
//! sum-of-squares terms.
//!
//! For the Poisson solver, the penalty is added directly to the NLL and
//! its gradient (see [`poisson`](crate::poisson) integration).

use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};

/// Proximal penalty terms for ADMM integration.
///
/// Each term adds `(rho/2) · (params[idx] - target)²` to the objective.
/// Used by the TV-ADMM outer loop to couple per-pixel fits with the
/// spatial regularization constraints.
#[derive(Debug, Clone, Default)]
pub struct ProximalPenalty {
    /// `(param_index, target_value, rho)` triples.
    pub terms: Vec<(usize, f64, f64)>,
}

impl ProximalPenalty {
    /// Compute the total penalty value: `Σ (rho/2)(params[idx] - target)²`.
    pub fn value(&self, params: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|&(idx, target, rho)| 0.5 * rho * (params[idx] - target).powi(2))
            .sum()
    }

    /// Compute the penalty gradient contribution for a given parameter.
    ///
    /// Returns `rho * (params[idx] - target)` if `param_idx` has a proximal
    /// term, or 0.0 otherwise.
    pub fn gradient(&self, param_idx: usize, param_val: f64) -> f64 {
        self.terms
            .iter()
            .filter(|&&(idx, _, _)| idx == param_idx)
            .map(|&(_, target, rho)| rho * (param_val - target))
            .sum()
    }

    /// Compute the penalty Hessian diagonal contribution for a given parameter.
    ///
    /// Returns `rho` if `param_idx` has a proximal term, or 0.0 otherwise.
    pub fn hessian_diag(&self, param_idx: usize) -> f64 {
        self.terms
            .iter()
            .filter(|&&(idx, _, _)| idx == param_idx)
            .map(|&(_, _, rho)| rho)
            .sum()
    }
}

/// Wraps a [`FitModel`], appending proximal pseudo-residual rows for LM.
///
/// For each proximal term `(idx, target, rho)`, appends one extra output:
///   `model_output[extra_row] = sqrt(rho) * params[idx]`
///
/// When paired with extended observations `y_obs[extra_row] = sqrt(rho) * target`
/// and `sigma[extra_row] = sqrt(2)`, the LM objective gains:
///   `residual² / sigma² = (rho/2) * (params[idx] - target)²`
///
/// matching [`ProximalPenalty::value`] and the ADMM augmented Lagrangian.
pub struct ProximalModel<M: FitModel> {
    inner: M,
    /// `(param_index, target, sqrt_rho)` — note sqrt_rho, not rho.
    terms: Vec<(usize, f64, f64)>,
}

impl<M: FitModel> ProximalModel<M> {
    /// Wrap an inner model with proximal penalty terms.
    ///
    /// `penalty` terms use `(idx, target, rho)` — this constructor stores
    /// `sqrt(rho)` internally.
    pub fn new(inner: M, penalty: &ProximalPenalty) -> Self {
        let terms = penalty
            .terms
            .iter()
            .map(|&(idx, target, rho)| (idx, target, rho.sqrt()))
            .collect();
        Self { inner, terms }
    }
}

impl<M: FitModel> FitModel for ProximalModel<M> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let mut y = self.inner.evaluate(params)?;
        for &(idx, _target, sqrt_rho) in &self.terms {
            y.push(sqrt_rho * params[idx]);
        }
        Ok(y)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_data = y_current.len() - self.terms.len();
        let inner_y = &y_current[..n_data];
        let inner_jac = self
            .inner
            .analytical_jacobian(params, free_param_indices, inner_y)?;

        let n_rows = y_current.len();
        let n_free = free_param_indices.len();
        let mut jac = FlatMatrix::zeros(n_rows, n_free);

        // Copy inner Jacobian rows.
        for i in 0..n_data {
            for j in 0..n_free {
                *jac.get_mut(i, j) = inner_jac.get(i, j);
            }
        }

        // Proximal rows: ∂(sqrt_rho * params[idx]) / ∂params[free_j] = sqrt_rho if idx == free_j.
        for (p, &(idx, _target, sqrt_rho)) in self.terms.iter().enumerate() {
            let row = n_data + p;
            if let Some(col) = free_param_indices.iter().position(|&fp| fp == idx) {
                *jac.get_mut(row, col) = sqrt_rho;
            }
        }

        Some(jac)
    }
}

/// Extend observed data and uncertainty vectors with proximal pseudo-observations.
///
/// For each proximal term, appends:
///   `y_obs_ext = sqrt(rho) * target`
///   `sigma_ext = sqrt(2)`
///
/// so the LM residual `(y_obs - y_model)² / sigma²` equals
/// `(rho/2) * (target - param)²` for the proximal rows, matching
/// [`ProximalPenalty::value`] and the ADMM augmented Lagrangian.
pub fn extend_data_for_proximal(
    y_obs: &[f64],
    sigma: &[f64],
    penalty: &ProximalPenalty,
) -> (Vec<f64>, Vec<f64>) {
    let mut y_ext = y_obs.to_vec();
    let mut s_ext = sigma.to_vec();
    for &(_idx, target, rho) in &penalty.terms {
        y_ext.push(rho.sqrt() * target);
        s_ext.push(std::f64::consts::SQRT_2);
    }
    (y_ext, s_ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial model: y[i] = params[0] * (i+1) for testing.
    struct LinearModel {
        n_points: usize,
    }

    impl FitModel for LinearModel {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok((0..self.n_points)
                .map(|i| params[0] * (i as f64 + 1.0))
                .collect())
        }
    }

    #[test]
    fn test_proximal_penalty_value() {
        let penalty = ProximalPenalty {
            terms: vec![(0, 2.0, 4.0), (1, 3.0, 6.0)],
        };
        // params = [5.0, 1.0]
        // term 0: 0.5 * 4.0 * (5.0 - 2.0)^2 = 2.0 * 9.0 = 18.0
        // term 1: 0.5 * 6.0 * (1.0 - 3.0)^2 = 3.0 * 4.0 = 12.0
        let v = penalty.value(&[5.0, 1.0]);
        assert!((v - 30.0).abs() < 1e-12, "expected 30.0, got {v}");
    }

    #[test]
    fn test_proximal_penalty_gradient() {
        let penalty = ProximalPenalty {
            terms: vec![(0, 2.0, 4.0)],
        };
        // grad[0] = 4.0 * (5.0 - 2.0) = 12.0
        let g = penalty.gradient(0, 5.0);
        assert!((g - 12.0).abs() < 1e-12, "expected 12.0, got {g}");
        // No term for param 1.
        let g1 = penalty.gradient(1, 5.0);
        assert!((g1).abs() < 1e-12, "expected 0.0, got {g1}");
    }

    #[test]
    fn test_proximal_penalty_hessian_diag() {
        let penalty = ProximalPenalty {
            terms: vec![(0, 2.0, 4.0), (1, 3.0, 6.0)],
        };
        assert!((penalty.hessian_diag(0) - 4.0).abs() < 1e-12);
        assert!((penalty.hessian_diag(1) - 6.0).abs() < 1e-12);
        assert!((penalty.hessian_diag(2)).abs() < 1e-12);
    }

    #[test]
    fn test_proximal_model_evaluate() {
        let model = LinearModel { n_points: 3 };
        let penalty = ProximalPenalty {
            terms: vec![(0, 1.5, 4.0)],
        };
        let proximal = ProximalModel::new(model, &penalty);

        let y = proximal.evaluate(&[2.0]).unwrap();
        // Inner: [2, 4, 6], proximal: sqrt(4.0) * 2.0 = 4.0
        assert_eq!(y.len(), 4);
        assert!((y[0] - 2.0).abs() < 1e-12);
        assert!((y[1] - 4.0).abs() < 1e-12);
        assert!((y[2] - 6.0).abs() < 1e-12);
        assert!((y[3] - 4.0).abs() < 1e-12); // sqrt(4) * 2.0
    }

    #[test]
    fn test_extend_data_for_proximal() {
        let y_obs = vec![1.0, 2.0, 3.0];
        let sigma = vec![0.1, 0.1, 0.1];
        let penalty = ProximalPenalty {
            terms: vec![(0, 1.5, 4.0)],
        };

        let (y_ext, s_ext) = extend_data_for_proximal(&y_obs, &sigma, &penalty);
        assert_eq!(y_ext.len(), 4);
        assert_eq!(s_ext.len(), 4);
        // sqrt(4.0) * 1.5 = 3.0
        assert!((y_ext[3] - 3.0).abs() < 1e-12);
        assert!((s_ext[3] - std::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn test_proximal_penalty_empty() {
        let penalty = ProximalPenalty::default();
        assert!((penalty.value(&[1.0, 2.0])).abs() < 1e-12);
        assert!((penalty.gradient(0, 5.0)).abs() < 1e-12);
        assert!((penalty.hessian_diag(0)).abs() < 1e-12);
    }
}
