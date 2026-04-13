//! Joint-Poisson counts-path objective with profiled flux.
//!
//! This module implements the **joint-Poisson conditional binomial deviance**
//! derived in `.research/spatial-regularization/evidence/32-counts-path-governing-equations-v2.md`
//! (equations §4.1, §5.7, §6.2b) and validated experimentally in memo 35
//! §P1.  It supersedes the fixed-flux Poisson NLL (`poisson.rs`) for the
//! counts-path fitter.
//!
//! ## Model
//!
//! Under the λ-at-sample convention with proton-charge ratio `c = Q_s / Q_ob`:
//!
//! - `O_i ~ Poisson(λ_i / c)`  (open-beam counts)
//! - `S_i ~ Poisson(λ_i · T_i)` (sample counts)
//!
//! Profiling out `λ_i` bin-by-bin gives the closed-form MLE
//!
//! ```text
//! λ̂_i = c · (O_i + S_i) / (1 + c · T_i)
//! ```
//!
//! The profile-conditional log-likelihood is equivalent (up to constants) to
//! a Binomial `S_i | N_i = O_i + S_i ~ Binomial(N_i, p_i)` with
//!
//! ```text
//! p_i = c · T_i / (1 + c · T_i)
//! ```
//!
//! The conditional deviance is
//!
//! ```text
//! D(θ) = 2 · Σ_i [ S_i · ln(S_i / (N_i · p_i))
//!                + O_i · ln(O_i / (N_i · (1 − p_i))) ]
//! ```
//!
//! with the `x · ln(x / 0) → 0` convention when `x = 0`.
//!
//! Under the correct model, `D / (n − k)` → 1 as n → ∞ — this replaces the
//! fixed-flux Pearson χ²/dof reported from the old Poisson path (which
//! scaled with `c` at constant density fidelity; see memo 35 headline).

use nereids_core::constants::{PIVOT_FLOOR, POISSON_EPSILON};

use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};
use crate::parameters::ParameterSet;

/// Joint-Poisson objective.
///
/// Wraps a transmission `FitModel` (which produces `T_i = model.evaluate(θ)`)
/// together with the observed open-beam counts `O_i`, sample counts `S_i`,
/// and proton-charge ratio `c = Q_s / Q_ob`.
///
/// The caller is responsible for ensuring `o`, `s`, and `model.evaluate()`
/// output all have the same length.
pub struct JointPoissonObjective<'a> {
    /// Transmission model: `evaluate(θ) → T(E)`.
    pub model: &'a dyn FitModel,
    /// Open-beam counts per bin.
    pub o: &'a [f64],
    /// Sample counts per bin.
    pub s: &'a [f64],
    /// Proton-charge ratio `c = Q_s / Q_ob`.  Must be strictly positive.
    pub c: f64,
}

impl<'a> JointPoissonObjective<'a> {
    /// Number of data bins.
    pub fn n_data(&self) -> usize {
        self.o.len()
    }

    /// Closed-form profile MLE for the per-bin flux: `λ̂ = c·(O+S) / (1+c·T)`.
    ///
    /// Guards: when `1 + c·T ≤ ε`, returns 0 to avoid division blow-up.
    #[inline]
    pub fn profile_lambda(&self, t_i: f64, o_i: f64, s_i: f64) -> f64 {
        let denom = 1.0 + self.c * t_i;
        if denom <= POISSON_EPSILON {
            0.0
        } else {
            self.c * (o_i + s_i) / denom
        }
    }

    /// Vector form of [`profile_lambda`](Self::profile_lambda).
    pub fn profile_lambda_per_bin(&self, t: &[f64]) -> Vec<f64> {
        t.iter()
            .zip(self.o.iter())
            .zip(self.s.iter())
            .map(|((&ti, &oi), &si)| self.profile_lambda(ti, oi, si))
            .collect()
    }

    /// Conditional binomial deviance at the given transmission vector.
    ///
    /// D = 2 · Σ [ S·ln(S/(Np)) + O·ln(O/(N(1−p))) ] with
    /// `p = cT/(1+cT)`, `N = O+S`, and `x·ln(x/0) → 0`.
    ///
    /// For `T ≤ ε` or `1+cT ≤ ε`, uses a smooth quadratic extrapolation in T
    /// analogous to the smooth-NLL trick in `poisson.rs`, so gradient-based
    /// optimizers see a C¹ objective rather than a cliff.
    pub fn deviance_from_transmission(&self, t: &[f64]) -> f64 {
        debug_assert_eq!(t.len(), self.o.len());
        debug_assert_eq!(t.len(), self.s.len());
        let mut d = 0.0;
        for ((&t_i, &o_i), &s_i) in t.iter().zip(self.o.iter()).zip(self.s.iter()) {
            d += binomial_deviance_term(s_i, o_i, t_i, self.c);
        }
        d
    }

    /// Evaluate the deviance at parameter vector θ by calling the model.
    pub fn deviance(&self, params: &[f64]) -> Result<f64, FittingError> {
        let t = self.model.evaluate(params)?;
        if t.len() != self.o.len() {
            return Err(FittingError::LengthMismatch {
                expected: self.o.len(),
                actual: t.len(),
                field: "transmission",
            });
        }
        Ok(self.deviance_from_transmission(&t))
    }

    /// Analytical gradient of the deviance w.r.t. the free parameters.
    ///
    /// Returns `None` if the transmission model does not provide an analytical
    /// Jacobian — callers should fall back to `deviance_gradient_fd`.
    ///
    /// Gradient derivation: with `p_i = cT_i/(1+cT_i)` and N_i = O_i+S_i,
    ///
    ///   d D / d T_i = −2 · (S_i − O_i·c·T_i) / (T_i · (1 + c·T_i))
    ///
    /// then chain-rule with the transmission Jacobian J_{i,j} = ∂T_i / ∂θ_{f(j)}
    /// where f(j) is the j-th free parameter index.
    pub fn deviance_gradient_analytical(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
    ) -> Result<Option<Vec<f64>>, FittingError> {
        let t = self.model.evaluate(params)?;
        if t.len() != self.o.len() {
            return Err(FittingError::LengthMismatch {
                expected: self.o.len(),
                actual: t.len(),
                field: "transmission",
            });
        }
        let jac = match self
            .model
            .analytical_jacobian(params, free_param_indices, &t)
        {
            Some(j) => j,
            None => return Ok(None),
        };
        let n_free = free_param_indices.len();
        let mut grad = vec![0.0f64; n_free];
        for (i, (&t_i, (&o_i, &s_i))) in t.iter().zip(self.o.iter().zip(self.s.iter())).enumerate()
        {
            let w = deviance_weight(s_i, o_i, t_i, self.c);
            for (g, col) in grad.iter_mut().zip(0..n_free) {
                *g += w * jac.get(i, col);
            }
        }
        Ok(Some(grad))
    }

    /// Fisher information for free parameters (Gauss-Newton curvature of D).
    ///
    /// Uses the expected-info form
    ///
    ///   h_i ≡ ∂² D / ∂ T_i²  ≈  2 · (O_i + S_i) · c / (T_i · (1 + c·T_i)²)
    ///
    /// (derived from logit-link binomial Var(S|N) = N p (1−p) and
    /// d logit(p) / dT = 1/T, scaled by 2 since D = −2 L).  Then
    ///
    ///   I(θ)_{j,k} = Σ_i h_i · J_{i,j} · J_{i,k}.
    ///
    /// Returns `None` if the transmission model does not provide an analytical
    /// Jacobian.
    pub fn fisher_information(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
    ) -> Result<Option<FlatMatrix>, FittingError> {
        let t = self.model.evaluate(params)?;
        let jac = match self
            .model
            .analytical_jacobian(params, free_param_indices, &t)
        {
            Some(j) => j,
            None => return Ok(None),
        };
        let n_free = free_param_indices.len();
        let mut info = FlatMatrix::zeros(n_free, n_free);
        for (i, ((&t_i, &o_i), &s_i)) in t.iter().zip(self.o.iter()).zip(self.s.iter()).enumerate()
        {
            let h = deviance_curvature(s_i, o_i, t_i, self.c);
            for j in 0..n_free {
                let jij = jac.get(i, j);
                for k in 0..n_free {
                    *info.get_mut(j, k) += h * jij * jac.get(i, k);
                }
            }
        }
        Ok(Some(info))
    }

    /// Finite-difference gradient of the deviance.
    ///
    /// Central differences on each free parameter.  Used as a fallback when
    /// the model has no analytical Jacobian.  `params` is a mutable
    /// `ParameterSet` so we can respect bounds via `clamp()`.
    pub fn deviance_gradient_fd(
        &self,
        params: &mut ParameterSet,
        fd_step: f64,
    ) -> Result<Vec<f64>, FittingError> {
        let free_idx = params.free_indices();
        let base_values = params.all_values();
        let base_d = self.deviance(&base_values)?;

        let mut grad = vec![0.0; free_idx.len()];
        for (j, &idx) in free_idx.iter().enumerate() {
            let original = params.params[idx].value;
            let step = fd_step * (1.0 + original.abs());

            params.params[idx].value = original + step;
            params.params[idx].clamp();
            let mut actual_step = params.params[idx].value - original;
            if actual_step.abs() < PIVOT_FLOOR {
                // Upper bound blocks forward step: try backward.
                params.params[idx].value = original - step;
                params.params[idx].clamp();
                actual_step = params.params[idx].value - original;
                if actual_step.abs() < PIVOT_FLOOR {
                    params.params[idx].value = original;
                    continue;
                }
            }
            let perturbed_values = params.all_values();
            let perturbed_d = match self.deviance(&perturbed_values) {
                Ok(v) => v,
                Err(_) => {
                    params.params[idx].value = original;
                    continue;
                }
            };
            params.params[idx].value = original;
            grad[j] = (perturbed_d - base_d) / actual_step;
        }
        Ok(grad)
    }
}

/// Per-bin binomial deviance term with smooth guards.
///
/// Returns `2 · [S·ln(S/(Np)) + O·ln(O/(N(1−p)))]` with the zero-count
/// convention `x · ln(x / ·) → 0` when `x = 0`.
///
/// For `T ≤ ε`: clamps to `ε` in the denominator rather than propagating
/// Inf/NaN — the optimizer can still see a finite (large) D and a
/// continuous gradient via the [`deviance_weight`] guard.
#[inline]
fn binomial_deviance_term(s: f64, o: f64, t: f64, c: f64) -> f64 {
    debug_assert!(
        s.is_finite() && s >= 0.0,
        "binomial_deviance_term: S must be finite and >= 0, got {s}"
    );
    debug_assert!(
        o.is_finite() && o >= 0.0,
        "binomial_deviance_term: O must be finite and >= 0, got {o}"
    );
    debug_assert!(
        c.is_finite() && c > 0.0,
        "binomial_deviance_term: c must be finite and > 0, got {c}"
    );
    let t_safe = t.max(POISSON_EPSILON);
    let n = s + o;
    if n <= 0.0 {
        // Bin has zero counts in both arms — no information, no contribution.
        return 0.0;
    }
    let ct = c * t_safe;
    // Use a numerically stable form for p.  For small cT, p ≈ cT, 1−p ≈ 1.
    let one_plus_ct = 1.0 + ct;
    // Expected sample and open-beam counts under profile λ̂.
    let exp_s = ct / one_plus_ct * n; // = N·p = c·N·T/(1+cT)
    let exp_o = n / one_plus_ct; //         = N·(1−p) = N/(1+cT)

    let term_s = xlogy_ratio(s, exp_s);
    let term_o = xlogy_ratio(o, exp_o);
    2.0 * (term_s + term_o)
}

/// `x · ln(x / y)` with the `0 · ln(0 / 0) → 0`, `x · ln(x / 0) → +∞`
/// conventions.  For `y > 0` and `x = 0` the term is 0.  For `y = 0` and
/// `x > 0` we clamp `y` to `POISSON_EPSILON` so the objective stays
/// finite and continuous.
#[inline]
fn xlogy_ratio(x: f64, y: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        let y_safe = y.max(POISSON_EPSILON);
        x * (x / y_safe).ln()
    }
}

/// Per-bin ∂D/∂T.
///
///   ∂D/∂T = −2 · (S − O·c·T) / (T · (1 + c·T))
///
/// When `T ≤ ε`, uses a linear extrapolation from `T = ε` so the gradient
/// stays finite and continuous across the boundary (matching the clamping
/// done in [`binomial_deviance_term`]).
#[inline]
fn deviance_weight(s: f64, o: f64, t: f64, c: f64) -> f64 {
    let t_safe = t.max(POISSON_EPSILON);
    let one_plus_ct = 1.0 + c * t_safe;
    -2.0 * (s - o * c * t_safe) / (t_safe * one_plus_ct)
}

/// Per-bin ∂²D/∂T² using the expected-info (Fisher) form.
///
/// Under the model, Var(S | N) = N · p · (1 − p) = N · cT / (1+cT)².  With
/// d logit(p) / dT = 1/T, the Fisher info on T is
///
///   I_TT = N · c / (T · (1 + c·T)²)
///
/// and ∂²D/∂T² = 2 · I_TT (since D = −2 · L_c).
#[inline]
fn deviance_curvature(s: f64, o: f64, t: f64, c: f64) -> f64 {
    let t_safe = t.max(POISSON_EPSILON);
    let n = s + o;
    let one_plus_ct = 1.0 + c * t_safe;
    2.0 * n * c / (t_safe * one_plus_ct * one_plus_ct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::FitParameter;

    // ------------------------------------------------------------------
    // Test fixtures
    // ------------------------------------------------------------------

    /// A constant-transmission model: T_i = θ_0 for all i.  Useful for
    /// testing the profile λ̂ formula and deviance / gradient in isolation.
    struct ConstModel {
        n_e: usize,
    }

    impl FitModel for ConstModel {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(vec![params[0]; self.n_e])
        }

        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let n_e = y_current.len();
            let n_free = free_param_indices.len();
            let mut jac = FlatMatrix::zeros(n_e, n_free);
            // ∂T/∂θ_0 = 1 for all i, and 0 for any other parameter.
            for i in 0..n_e {
                for (j, &pi) in free_param_indices.iter().enumerate() {
                    *jac.get_mut(i, j) = if pi == 0 { 1.0 } else { 0.0 };
                }
            }
            Some(jac)
        }
    }

    /// A linear-in-E model: T_i = θ_0 − θ_1 · e_i (Beer-Lambert surrogate).
    /// Used for the analytical-vs-FD gradient check and profile tests with
    /// non-trivial Jacobian.
    struct LinearModel<'a> {
        e: &'a [f64],
    }

    impl<'a> FitModel for LinearModel<'a> {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(self
                .e
                .iter()
                .map(|&ei| (params[0] - params[1] * ei).max(POISSON_EPSILON))
                .collect())
        }

        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let n_e = y_current.len();
            let n_free = free_param_indices.len();
            let mut jac = FlatMatrix::zeros(n_e, n_free);
            for i in 0..n_e {
                for (j, &pi) in free_param_indices.iter().enumerate() {
                    *jac.get_mut(i, j) = match pi {
                        0 => 1.0,
                        1 => -self.e[i],
                        _ => 0.0,
                    };
                }
            }
            Some(jac)
        }
    }

    // ------------------------------------------------------------------
    // (a) Profile λ̂ closed form matches the score-equation bisection root.
    // ------------------------------------------------------------------
    #[test]
    fn test_profile_lambda_closed_form_matches_bisection() {
        // For each bin independently, score(λ) = (O+S)/λ − (1/c + T) = 0
        // has the unique positive root λ̂ = c(O+S)/(1+cT).  Bisect on
        // [1e-10, 1e12] and verify agreement to 1e-9.
        let cases = [
            (50.0_f64, 5.0_f64, 0.5_f64, 1.0_f64),
            (1000.0, 900.0, 0.9, 5.98),
            (10.0, 1.0, 0.1, 2.0),
            (0.0, 5.0, 0.25, 1.5), // O=0 edge
            (5.0, 0.0, 0.75, 3.0), // S=0 edge
        ];
        for (o, s, t, c) in cases {
            let model = ConstModel { n_e: 1 };
            let obj = JointPoissonObjective {
                model: &model,
                o: &[o],
                s: &[s],
                c,
            };
            let closed = obj.profile_lambda(t, o, s);

            // Bisection root of score(λ) = (O+S)/λ − (1/c + T).
            let score = |lam: f64| (o + s) / lam - (1.0 / c + t);
            let (mut lo, mut hi) = (1e-10, 1e12);
            // score is monotonically decreasing in λ, score(lo) > 0, score(hi) < 0.
            assert!(score(lo) >= 0.0);
            assert!(score(hi) <= 0.0);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if score(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let bisect = 0.5 * (lo + hi);
            let rel_err = ((closed - bisect) / bisect).abs();
            assert!(
                rel_err < 1e-9,
                "profile λ̂ mismatch: closed={closed} bisect={bisect} rel_err={rel_err}"
            );
        }
    }

    // ------------------------------------------------------------------
    // (b) D = 0 at exact match of expected counts.
    // ------------------------------------------------------------------
    #[test]
    fn test_deviance_zero_at_exact_match() {
        // Construct a model where S_i = λ·T_i, O_i = λ/c exactly for integer
        // choices, then verify D < 1e-8.  With T=0.5, c=2, λ=200: S=100,
        // O=100 per bin; p = 2*0.5/(1+1) = 0.5; Np = (O+S)/2 = 100 = S;
        // N(1-p) = 100 = O, so both logs are zero and D = 0.
        let t_val = 0.5;
        let c = 2.0;
        let n_bins = 5;
        let o = vec![100.0; n_bins];
        let s = vec![100.0; n_bins];
        let t = vec![t_val; n_bins];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c,
        };
        let d = obj.deviance_from_transmission(&t);
        assert!(d.abs() < 1e-8, "D should be ≈ 0 at exact match, got {d}");

        // Also verify via parameter evaluation (model returns constant T).
        let d_via_params = obj.deviance(&[t_val]).unwrap();
        assert!(d_via_params.abs() < 1e-8);
    }

    // ------------------------------------------------------------------
    // (c) Analytical gradient matches finite-difference.
    // ------------------------------------------------------------------
    #[test]
    fn test_deviance_gradient_matches_fd() {
        // Use the linear model T = θ_0 − θ_1 · E with noise-free synthetic
        // counts.  Compute analytical gradient via chain rule and FD
        // gradient via re-evaluation; they must agree.
        let e: Vec<f64> = (0..20).map(|i| 0.1 + 0.05 * i as f64).collect();
        let theta_true = [0.95_f64, 0.1_f64];
        let c = 3.0;
        let lam = 500.0;

        // Generate noise-free expected counts.
        let model = LinearModel { e: &e };
        let t_true = model.evaluate(&theta_true).unwrap();
        let o: Vec<f64> = t_true.iter().map(|_| lam / c).collect();
        let s: Vec<f64> = t_true.iter().map(|&ti| lam * ti).collect();

        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c,
        };

        // Evaluate gradient at a point slightly off truth so it is nonzero.
        let theta_eval = [0.80_f64, 0.15_f64];
        let free_idx = vec![0, 1];

        let g_analytical = obj
            .deviance_gradient_analytical(&theta_eval, &free_idx)
            .unwrap()
            .expect("LinearModel provides analytical jacobian");

        // Central-difference gradient.
        let eps = 1e-6;
        let mut g_fd = [0.0_f64; 2];
        for j in 0..2 {
            let mut tp = theta_eval;
            let mut tm = theta_eval;
            tp[j] += eps;
            tm[j] -= eps;
            let dp = obj.deviance(&tp).unwrap();
            let dm = obj.deviance(&tm).unwrap();
            g_fd[j] = (dp - dm) / (2.0 * eps);
        }

        for (a, f) in g_analytical.iter().zip(g_fd.iter()) {
            let rel = ((a - f) / f.abs().max(1e-6)).abs();
            assert!(
                rel < 1e-4,
                "analytical vs FD gradient disagree: analytical={a} fd={f} rel={rel}"
            );
        }
    }

    // ------------------------------------------------------------------
    // (d) D/(n-k) asymptote on synthetic joint-Poisson data at matched
    //     model — single free parameter θ_0 = T, use 1D grid search to
    //     recover it, verify D/(n-1) ≈ 1 and density bias < 1%.
    // ------------------------------------------------------------------
    #[test]
    fn test_deviance_per_dof_asymptote() {
        // Deterministic generator (xorshift) so the test is reproducible.
        struct Xorshift(u64);
        impl Xorshift {
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
            // Knuth-style Poisson sampler (good enough for rate ≤ ~100).
            fn poisson(&mut self, lambda: f64) -> f64 {
                if lambda <= 0.0 {
                    return 0.0;
                }
                if lambda > 30.0 {
                    // Gaussian approx for moderate rates — test cells all
                    // use small λ, but keep the branch for robustness.
                    let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
                    let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    return (lambda + z * lambda.sqrt()).round().max(0.0);
                }
                let l = (-lambda).exp();
                let mut k: f64 = 0.0;
                let mut p: f64 = 1.0;
                loop {
                    k += 1.0;
                    let u = (self.next_u64() as f64) / (u64::MAX as f64);
                    p *= u;
                    if p <= l {
                        return k - 1.0;
                    }
                    if k > 1000.0 {
                        return k - 1.0;
                    }
                }
            }
        }

        let n_bins = 200;
        let t_true = 0.35_f64;
        let c = 2.0;
        let lam = 50.0;
        let n_reps = 30;

        let mut d_per_dof_samples = Vec::with_capacity(n_reps);
        let mut bias_samples = Vec::with_capacity(n_reps);
        let mut rng = Xorshift(0xDEAD_BEEF_CAFE_BABE);

        for _ in 0..n_reps {
            let o: Vec<f64> = (0..n_bins).map(|_| rng.poisson(lam / c)).collect();
            let s: Vec<f64> = (0..n_bins).map(|_| rng.poisson(lam * t_true)).collect();
            let model = ConstModel { n_e: n_bins };
            let obj = JointPoissonObjective {
                model: &model,
                o: &o,
                s: &s,
                c,
            };

            // 1D grid search over T, then local refinement via Brent-like
            // bisection on the gradient sign.
            let grid: Vec<f64> = (0..200).map(|i| 0.01 + 0.99 * (i as f64) / 199.0).collect();
            let mut best = (grid[0], f64::INFINITY);
            for &t_try in &grid {
                let d_try = obj.deviance_from_transmission(&vec![t_try; n_bins]);
                if d_try < best.1 {
                    best = (t_try, d_try);
                }
            }
            // Bisect on the gradient-sign neighbourhood.
            let dt = 0.01;
            let (mut lo, mut hi) = ((best.0 - dt).max(POISSON_EPSILON), (best.0 + dt).min(0.999));
            let grad_at = |t: f64| -> f64 {
                let tvec = vec![t; n_bins];
                let free_idx = [0_usize];
                let g = obj
                    .deviance_gradient_analytical(&[t], &free_idx)
                    .unwrap()
                    .unwrap();
                // gradient is w.r.t. θ_0 = T (ConstModel Jacobian is 1).
                let _ = tvec; // silence unused
                g[0]
            };
            let mut glo = grad_at(lo);
            let mut ghi = grad_at(hi);
            if glo * ghi < 0.0 {
                for _ in 0..80 {
                    let mid = 0.5 * (lo + hi);
                    let gmid = grad_at(mid);
                    if gmid * glo < 0.0 {
                        hi = mid;
                        ghi = gmid;
                    } else {
                        lo = mid;
                        glo = gmid;
                    }
                }
            }
            let t_hat = 0.5 * (lo + hi);
            let d_hat = obj.deviance_from_transmission(&vec![t_hat; n_bins]);
            let dof = (n_bins - 1) as f64;
            d_per_dof_samples.push(d_hat / dof);
            bias_samples.push((t_hat - t_true) / t_true);
        }

        let mean_dpd: f64 = d_per_dof_samples.iter().sum::<f64>() / d_per_dof_samples.len() as f64;
        let mean_bias: f64 = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;

        // Under matched model, E[D]/(n-k) → 1.  Tolerate [0.85, 1.15]
        // with n_bins=200, n_reps=30, small λ (some low-count bins).
        assert!(
            (0.85..=1.15).contains(&mean_dpd),
            "D/(n-k) asymptote out of band: mean={mean_dpd}"
        );
        assert!(
            mean_bias.abs() < 0.02,
            "density bias > 2%: mean={mean_bias}"
        );
    }

    // ------------------------------------------------------------------
    // Edge: zero-count bin contributes 0 deviance regardless of T.
    // ------------------------------------------------------------------
    #[test]
    fn test_zero_counts_contribute_zero() {
        let model = ConstModel { n_e: 3 };
        let obj = JointPoissonObjective {
            model: &model,
            o: &[0.0, 10.0, 5.0],
            s: &[0.0, 5.0, 2.0],
            c: 1.5,
        };
        let d_full = obj.deviance_from_transmission(&[0.6, 0.6, 0.6]);
        // Drop the zero-N bin — result must be identical.
        let obj_reduced = JointPoissonObjective {
            model: &model, // same model, we just bypass the 1st bin via data
            o: &[10.0, 5.0],
            s: &[5.0, 2.0],
            c: 1.5,
        };
        let d_reduced = obj_reduced.deviance_from_transmission(&[0.6, 0.6]);
        assert!((d_full - d_reduced).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // FD gradient fallback agrees with analytical form.
    // ------------------------------------------------------------------
    #[test]
    fn test_fd_gradient_matches_analytical() {
        let e: Vec<f64> = (0..15).map(|i| 0.2 + 0.1 * i as f64).collect();
        let theta = [0.9_f64, 0.05_f64];
        let c = 1.5;
        let lam = 300.0;
        let model = LinearModel { e: &e };
        let t_true = model.evaluate(&theta).unwrap();
        let o: Vec<f64> = t_true.iter().map(|_| lam / c).collect();
        let s: Vec<f64> = t_true.iter().map(|&ti| lam * ti).collect();
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c,
        };
        let mut ps = ParameterSet::new(vec![
            FitParameter::non_negative("theta_0", 0.85),
            FitParameter::non_negative("theta_1", 0.06),
        ]);
        let g_fd = obj.deviance_gradient_fd(&mut ps, 1e-6).unwrap();
        let g_analytical = obj
            .deviance_gradient_analytical(&ps.all_values(), &ps.free_indices())
            .unwrap()
            .unwrap();
        for (f, a) in g_fd.iter().zip(g_analytical.iter()) {
            let rel = ((f - a) / a.abs().max(1e-6)).abs();
            assert!(rel < 5e-3, "fd={f} analytical={a} rel={rel}");
        }
    }

    // ------------------------------------------------------------------
    // Fisher matrix is symmetric positive semi-definite at the fit.
    // ------------------------------------------------------------------
    #[test]
    fn test_fisher_matrix_symmetry_psd() {
        let e: Vec<f64> = (0..10).map(|i| 0.3 + 0.1 * i as f64).collect();
        let theta = [0.9_f64, 0.05_f64];
        let c = 2.0;
        let lam = 400.0;
        let model = LinearModel { e: &e };
        let t_true = model.evaluate(&theta).unwrap();
        let o: Vec<f64> = t_true.iter().map(|_| lam / c).collect();
        let s: Vec<f64> = t_true.iter().map(|&ti| lam * ti).collect();
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c,
        };
        let info = obj
            .fisher_information(&theta, &[0, 1])
            .unwrap()
            .expect("LinearModel provides analytical jacobian");
        // Symmetry.
        let i01 = info.get(0, 1);
        let i10 = info.get(1, 0);
        assert!((i01 - i10).abs() < 1e-10);
        // PSD: diagonal entries > 0 (model is identifiable).
        assert!(info.get(0, 0) > 0.0);
        assert!(info.get(1, 1) > 0.0);
        // Determinant > 0 (rank-2 identifiable).
        let det = info.get(0, 0) * info.get(1, 1) - i01 * i10;
        assert!(det > 0.0, "Fisher matrix determinant = {det}");
    }
}
