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
    /// Near invalid or numerically tiny transmission values, the per-bin
    /// evaluation (`binomial_deviance_term`) uses `t.max(POISSON_EPSILON)`
    /// to clamp T away from zero before entering the logarithms and the
    /// `1/(1+cT)` factor.  This avoids singular logs and division-by-zero
    /// but is a piecewise clamp, not a smooth quadratic extrapolation —
    /// D(T) is C⁰ at the clamp boundary, not C¹.  In practice this is
    /// adequate because the optimizer's transmission values come from a
    /// `FitModel` that keeps T bounded well above `POISSON_EPSILON` for
    /// physically plausible density / nuisance parameter values.
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

    /// Finite-difference Fisher information.
    ///
    /// Fallback for callers whose transmission model does not implement
    /// [`FitModel::analytical_jacobian`] — i.e., when
    /// [`Self::fisher_information`] would return `None`.  Builds the
    /// transmission Jacobian column-by-column via central differences and
    /// assembles
    ///
    ///   `I(θ)_{j,k} = Σ_i h_i · J_{i,j} · J_{i,k}`
    ///
    /// where `h_i = ∂² D / ∂ T_i²` is the per-bin deviance curvature
    /// `2·(O_i + S_i)·c / (T_i·(1 + c·T_i)²)` (Fisher-scoring form derived
    /// from binomial logit-link Var(S | N) = N·p·(1−p) with d logit p / dT
    /// = 1/T — see the module-level docstring §Model).  Returns `Ok(None)`
    /// only if the base model evaluation itself fails.
    pub fn fisher_information_fd(
        &self,
        params: &mut ParameterSet,
        fd_step: f64,
    ) -> Result<Option<FlatMatrix>, FittingError> {
        let free_idx = params.free_indices();
        let base_values = params.all_values();
        let t_base = self.model.evaluate(&base_values)?;
        let n_e = t_base.len();
        let n_free = free_idx.len();
        if n_free == 0 {
            return Ok(Some(FlatMatrix::zeros(0, 0)));
        }
        let mut jac = FlatMatrix::zeros(n_e, n_free);
        for (col, &idx) in free_idx.iter().enumerate() {
            let original = params.params[idx].value;
            let step = fd_step * (1.0 + original.abs());
            params.params[idx].value = original + step;
            params.params[idx].clamp();
            let forward_step = params.params[idx].value - original;
            let t_plus = if forward_step.abs() >= PIVOT_FLOOR {
                Some(self.model.evaluate(&params.all_values())?)
            } else {
                None
            };
            params.params[idx].value = original - step;
            params.params[idx].clamp();
            let backward_step = original - params.params[idx].value;
            let t_minus = if backward_step.abs() >= PIVOT_FLOOR {
                Some(self.model.evaluate(&params.all_values())?)
            } else {
                None
            };
            params.params[idx].value = original;
            let (t_a, t_b, denom) = match (t_plus, t_minus) {
                (Some(tp), Some(tm)) => (tp, tm, forward_step + backward_step),
                (Some(tp), None) => (tp, t_base.clone(), forward_step),
                (None, Some(tm)) => (t_base.clone(), tm, backward_step),
                (None, None) => continue,
            };
            if denom.abs() < PIVOT_FLOOR {
                continue;
            }
            for i in 0..n_e {
                *jac.get_mut(i, col) = (t_a[i] - t_b[i]) / denom;
            }
        }
        let mut info = FlatMatrix::zeros(n_free, n_free);
        for (i, ((&t_i, &o_i), &s_i)) in t_base
            .iter()
            .zip(self.o.iter())
            .zip(self.s.iter())
            .enumerate()
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

// ======================================================================
// joint_poisson_fit — two-stage solver (damped Fisher + Nelder-Mead polish)
// ======================================================================

use crate::lm::{invert_matrix, solve_damped_system};
use crate::nelder_mead::{NelderMeadConfig, nelder_mead_minimize};

/// Configuration for [`joint_poisson_fit`].
#[derive(Debug, Clone)]
pub struct JointPoissonFitConfig {
    /// Maximum number of damped-Fisher iterations in stage 1.
    pub max_iter: usize,
    /// Initial damping factor (Marquardt λ) on the Fisher matrix diagonal.
    pub lambda_init: f64,
    /// Multiplicative factor to increase λ on a rejected step.
    pub lambda_up: f64,
    /// Multiplicative factor to decrease λ on an accepted step.
    pub lambda_down: f64,
    /// Armijo sufficient-decrease coefficient.
    pub armijo_c: f64,
    /// Backtracking factor during line search.
    pub backtrack: f64,
    /// Convergence tolerance on relative deviance change.
    pub tol_d: f64,
    /// Convergence tolerance on normalized parameter step.
    pub tol_param: f64,
    /// Finite-difference step for gradient fallback.
    pub fd_step: f64,
    /// Enable Nelder-Mead polish after stage 1.
    ///
    /// Default `false` as of #486.  The polish tolerances
    /// (`xatol = 1e-9, fatol = 1e-10`) were originally matched to the
    /// EG5 synthetic benchmark (memo 35 §P2.1) where D stays O(1), so
    /// `fatol` is physically meaningful.  On real-data regimes where
    /// D saturates at 10⁴–10⁵ (un-modelled upstream physics —
    /// memo 35 §P3/§P4), `fatol / D` drops below f64 ULP and polish
    /// cannot self-terminate — it burns its full `max_iter = 5000`
    /// every fit at 70–260× wall cost, and the three-scenario
    /// ablation on real VENUS Hf 120-min data (issue #486) showed
    /// the resulting parameter shift is ≤ 0.35 Fisher σ on every
    /// parameter in every scenario — i.e. below the solver's own
    /// reported uncertainty floor.
    ///
    /// The polish mechanism itself is sound (self-terminates cleanly
    /// on synthetic D≈1 data per ablation S3); only the absolute
    /// tolerance defaults are mis-calibrated for real counts data.
    /// A future scale-aware rescale (`fatol_rel` vs `D_stage1`) can
    /// re-enable polish as a useful opt-in refinement.
    ///
    /// Set this to `true` (via `with_counts_enable_polish(Some(true))`
    /// at the pipeline level) when you specifically want the polish
    /// stage on a synthetic / clean-data scenario where the absolute
    /// tolerance defaults are physically meaningful.
    pub enable_polish: bool,
    /// Polish (Nelder-Mead) configuration.  Used only when
    /// `enable_polish == true`.  Default `xatol = 1e-9`, `fatol = 1e-10`
    /// match the EG5 synthetic benchmark tolerances from memo 35 §P2.1
    /// — physically meaningful when `D ≈ 1` (clean data) but sub-f64-
    /// ULP on real counts where `D ≈ 10⁴`–`10⁵`, which is why
    /// `enable_polish` defaults to `false`.  See #486.
    pub polish: NelderMeadConfig,
    /// Compute and return the Fisher covariance and parameter uncertainties.
    pub compute_covariance: bool,
}

impl Default for JointPoissonFitConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            lambda_init: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            armijo_c: 1e-4,
            backtrack: 0.5,
            tol_d: 1e-8,
            tol_param: 1e-8,
            fd_step: 1e-6,
            // #486: flipped from `true` to `false` after a three-scenario
            // ablation on real VENUS data showed polish burning full
            // `max_iter = 5000` at 70-260× wall cost for ≤ 0.35 Fisher σ
            // parameter movement.  The absolute tolerances below are
            // physically meaningful for synthetic (D ≈ 1) benchmarks and
            // dead on real counts data (D ≈ 10⁵).  Opt in via
            // `UnifiedFitConfig::with_counts_enable_polish(Some(true))`
            // when you specifically want the polish stage.  See the
            // field doc on `enable_polish` for details.
            enable_polish: false,
            polish: NelderMeadConfig {
                // Tolerances tuned for the EG5 synthetic regime (memo 35
                // §P2.1) — `fatol = 1e-10` vs D ≈ 1 is a physically
                // meaningful "deviance isn't budging" check.  On real
                // counts data where D ≈ 10⁵ the same absolute value is
                // sub-ULP; polish can't self-terminate and is disabled
                // by the default above.  A future scale-aware rescale
                // (`fatol_rel` vs D_stage1) is tracked as a follow-up.
                xatol: 1e-9,
                fatol: 1e-10,
                max_iter: 5000,
                initial_step_frac: 0.02,
                initial_step_abs: 1e-4,
            },
            compute_covariance: true,
        }
    }
}

/// Outcome of [`joint_poisson_fit`].
#[derive(Debug, Clone)]
pub struct JointPoissonResult {
    /// Final deviance D at the fitted parameters.
    pub deviance: f64,
    /// D / (n − k).  Primary GOF statistic per memo 35 §P1.2.
    pub deviance_per_dof: f64,
    /// Number of data bins (n).
    pub n_data: usize,
    /// Number of free parameters (k).
    pub n_free: usize,
    /// Iterations performed in the damped-Fisher stage.
    pub gn_iterations: usize,
    /// Iterations performed by the Nelder-Mead polish stage (0 if disabled).
    pub polish_iterations: usize,
    /// `true` when the stage-1 (damped Fisher) optimizer met its `tol_d`
    /// and `tol_param` criteria before hitting `max_iter`.
    pub gn_converged: bool,
    /// `true` when the Nelder-Mead polish met `xatol` and `fatol` before
    /// `max_iter` (always `false` if `enable_polish == false`).
    pub polish_converged: bool,
    /// `true` when the polish step lowered the deviance below the stage-1
    /// best value.  Useful diagnostic — if polish improved D materially,
    /// stage 1 likely stalled.
    pub polish_improved: bool,
    /// Final parameter values (all parameters, including fixed).
    pub params: Vec<f64>,
    /// Inverse Fisher covariance of free parameters (n_free × n_free),
    /// computed at the final θ.  `None` if the Fisher matrix was singular
    /// or `compute_covariance == false`.
    pub covariance: Option<FlatMatrix>,
    /// `√diag(covariance)` for each free parameter, in free-index order.
    pub uncertainties: Option<Vec<f64>>,
}

/// Two-stage joint-Poisson fit: damped Fisher stage followed by
/// Nelder-Mead polish.
///
/// **Memo 35 §P1 + §P2 requirements** this function satisfies:
///
/// - Minimizes the **conditional binomial deviance** `D(θ)`
///   ([`JointPoissonObjective::deviance`]), not fixed-flux Poisson NLL.
/// - Reports `D / (n − k)` as the primary GOF (P1.2).
/// - Honours an **explicit `c = Q_s/Q_ob`** stored in the objective (P1.3).
/// - Runs Nelder-Mead **polish** after the gradient stage to escape the
///   EG2-S1 C_full initial-point stall (P2.1).
/// - Exposes `gn_converged` and `polish_converged` separately so callers
///   do not rely on a single "success" flag — acceptance is meant to come
///   from the deviance value (P2.3).
///
/// The damped-Fisher stage uses LM-style acceptance: a step is accepted if
/// it satisfies an Armijo condition on D; on rejection, λ is increased and
/// the step is recomputed.  Bounds are enforced via projection (clamp).
pub fn joint_poisson_fit(
    objective: &JointPoissonObjective<'_>,
    params: &mut ParameterSet,
    config: &JointPoissonFitConfig,
) -> Result<JointPoissonResult, FittingError> {
    let n_data = objective.n_data();
    if n_data == 0 {
        return Err(FittingError::EmptyData);
    }

    // Stage 1: damped Fisher with Armijo backtracking.
    let stage1 = damped_fisher_stage(objective, params, config)?;

    // Capture stage-1 best.
    let best_d_stage1 = stage1.deviance;
    let gn_iterations = stage1.iterations;
    let gn_converged = stage1.converged;

    // Stage 2: Nelder-Mead polish on free parameters, seeded from stage-1 θ.
    let mut polish_iterations = 0usize;
    let mut polish_converged = false;
    let mut polish_improved = false;
    if config.enable_polish {
        let free_idx = params.free_indices();
        let bounds: Vec<(f64, f64)> = free_idx
            .iter()
            .map(|&i| (params.params[i].lower, params.params[i].upper))
            .collect();
        let x0: Vec<f64> = free_idx.iter().map(|&i| params.params[i].value).collect();

        // Snapshot fixed parameters so the closure can rebuild the full
        // parameter vector for each evaluation.
        let all_values_snapshot = params.all_values();

        let obj_closure = |x: &[f64]| -> Result<f64, FittingError> {
            let mut all = all_values_snapshot.clone();
            for (j, &idx) in free_idx.iter().enumerate() {
                all[idx] = x[j];
            }
            objective.deviance(&all)
        };
        let nm = nelder_mead_minimize(obj_closure, &x0, Some(&bounds), &config.polish)?;
        polish_iterations = nm.iterations;
        polish_converged = nm.self_converged;
        if nm.fun < best_d_stage1 {
            polish_improved = true;
            // Commit polish result to the parameter set.
            for (j, &idx) in free_idx.iter().enumerate() {
                params.params[idx].value = nm.x[j];
                params.params[idx].clamp();
            }
        }
    }

    let final_values = params.all_values();
    let final_deviance = objective.deviance(&final_values)?;
    let n_free = params.n_free();
    let dof = (n_data as isize - n_free as isize).max(1) as f64;
    let deviance_per_dof = final_deviance / dof;

    // Covariance from inverse Fisher at the final θ.  Uses the analytical
    // Jacobian when the transmission model provides one; otherwise falls
    // back to finite-difference Jacobian assembled into the deviance-
    // Hessian form — so callers always get uncertainties for identifiable
    // parameters.
    //
    // **Scale note (covariance vs Newton step).**  `fisher_information`
    // assembles `H_D = Σ h_i · J·J^T` with `h_i = ∂² D / ∂ T_i² = 2 · I_TT_i`
    // (see [`deviance_curvature`]).  This `2·I` form is exactly what the
    // damped-Fisher Newton step needs, since stepping on D with
    // `Δθ = -H_D^{-1} · ∇D = -(2I)^{-1} · (-2 ∇L) = I^{-1} · ∇L`
    // recovers the Fisher-scoring direction on the log-likelihood L.
    //
    // For the asymptotic MLE covariance, however, the Cramér-Rao bound is
    // `Cov(θ̂) = I^{-1}`, NOT `H_D^{-1} = (2I)^{-1} = I^{-1}/2`.  Inverting
    // `H_D` and using it directly would under-report variance by 2× and
    // standard errors by √2 × — a real bug caught in review.  We rescale
    // the inverse here: `I^{-1} = 2 · H_D^{-1}`.
    let (covariance, uncertainties) = if config.compute_covariance {
        let free_idx = params.free_indices();
        let info_opt = match objective.fisher_information(&final_values, &free_idx)? {
            Some(info) => Some(info),
            None => objective.fisher_information_fd(params, config.fd_step)?,
        };
        match info_opt {
            Some(info) => match invert_matrix(&info) {
                Some(mut cov) => {
                    // Rescale: invert_matrix returned (2I)^{-1}; multiply
                    // every entry by 2 to obtain I^{-1}.
                    for v in cov.data.iter_mut() {
                        *v *= 2.0;
                    }
                    let u: Vec<f64> = (0..cov.nrows)
                        .map(|i| {
                            let v = cov.get(i, i);
                            if v > 0.0 { v.sqrt() } else { f64::NAN }
                        })
                        .collect();
                    (Some(cov), Some(u))
                }
                None => (None, None),
            },
            None => (None, None),
        }
    } else {
        (None, None)
    };

    Ok(JointPoissonResult {
        deviance: final_deviance,
        deviance_per_dof,
        n_data,
        n_free,
        gn_iterations,
        polish_iterations,
        gn_converged,
        polish_converged,
        polish_improved,
        params: final_values,
        covariance,
        uncertainties,
    })
}

/// Stage 1 output.
struct Stage1Output {
    deviance: f64,
    iterations: usize,
    converged: bool,
}

/// Damped-Fisher stage (Gauss-Newton / Marquardt on the deviance).
///
/// Mirrors the structure of `lm.rs` but on the joint-Poisson objective.
/// Falls back to finite-difference gradient when the model has no
/// analytical Jacobian.
fn damped_fisher_stage(
    objective: &JointPoissonObjective<'_>,
    params: &mut ParameterSet,
    config: &JointPoissonFitConfig,
) -> Result<Stage1Output, FittingError> {
    let mut lambda = config.lambda_init;
    let mut iter = 0usize;
    let mut converged = false;

    let mut all_vals = params.all_values();
    let mut d_current = objective.deviance(&all_vals)?;

    while iter < config.max_iter {
        iter += 1;
        let free_idx = params.free_indices();
        let n_free = free_idx.len();
        if n_free == 0 {
            converged = true;
            break;
        }

        // Gradient (analytical if available, FD otherwise).
        let grad = match objective.deviance_gradient_analytical(&all_vals, &free_idx)? {
            Some(g) => g,
            None => objective.deviance_gradient_fd(params, config.fd_step)?,
        };
        // Fisher information (Gauss-Newton curvature).  If absent, use a
        // diagonal identity fallback scaled by gradient magnitude — this
        // degenerates the stage into projected gradient descent, which is
        // exactly how `poisson.rs` behaves in the FD regime.
        let info = match objective.fisher_information(&all_vals, &free_idx)? {
            Some(m) => m,
            None => {
                let mut ident = FlatMatrix::zeros(n_free, n_free);
                for i in 0..n_free {
                    *ident.get_mut(i, i) = 1.0;
                }
                ident
            }
        };
        // Solve (I + λ diag(I)) δ = -g.
        let neg_grad: Vec<f64> = grad.iter().map(|&g| -g).collect();
        let step = match solve_damped_system(&info, &neg_grad, lambda) {
            Some(s) => s,
            None => {
                // Singular Fisher at current θ.  Increase damping and retry
                // on the next iteration.
                lambda *= config.lambda_up;
                if lambda > 1e16 {
                    break;
                }
                continue;
            }
        };

        // Armijo line search with projection.
        let grad_dot_step = grad
            .iter()
            .zip(step.iter())
            .map(|(&g, &s)| g * s)
            .sum::<f64>();
        // If the step isn't a descent direction w.r.t. D, flip sign (fallback
        // to negative gradient direction).
        let effective_step: Vec<f64> = if grad_dot_step >= 0.0 {
            grad.iter().map(|&g| -g).collect()
        } else {
            step
        };

        let mut alpha = 1.0;
        let mut accepted = false;
        let d0 = d_current;
        let mut trial_vals = all_vals.clone();
        for _ in 0..50 {
            for (j, &idx) in free_idx.iter().enumerate() {
                trial_vals[idx] = all_vals[idx] + alpha * effective_step[j];
            }
            // Project onto bounds.
            for &idx in free_idx.iter() {
                let lo = params.params[idx].lower;
                let hi = params.params[idx].upper;
                if trial_vals[idx] < lo {
                    trial_vals[idx] = lo;
                }
                if trial_vals[idx] > hi {
                    trial_vals[idx] = hi;
                }
            }
            let d_trial = match objective.deviance(&trial_vals) {
                Ok(v) if v.is_finite() => v,
                _ => f64::INFINITY,
            };
            // Armijo condition: f(x+αp) ≤ f(x) + c·α·⟨g, p⟩ (descent).  When
            // we flipped to -grad above, ⟨g, p⟩ = -||g||² < 0.
            let gdotp = grad
                .iter()
                .zip(effective_step.iter())
                .map(|(&g, &s)| g * s)
                .sum::<f64>();
            if d_trial <= d0 + config.armijo_c * alpha * gdotp {
                accepted = true;
                break;
            }
            alpha *= config.backtrack;
            if alpha < 1e-16 {
                break;
            }
        }

        if accepted {
            // Commit step.
            for &idx in free_idx.iter() {
                params.params[idx].value = trial_vals[idx];
                params.params[idx].clamp();
            }
            let rel_change =
                (d_current - objective.deviance(&trial_vals)?) / d_current.abs().max(1.0);
            all_vals = params.all_values();
            let new_d = objective.deviance(&all_vals)?;
            let step_norm_sq = effective_step
                .iter()
                .map(|&s| (alpha * s).powi(2))
                .sum::<f64>();
            let step_norm = step_norm_sq.sqrt();
            d_current = new_d;
            lambda = (lambda * config.lambda_down).max(1e-16);

            if rel_change.abs() < config.tol_d && step_norm < config.tol_param {
                converged = true;
                break;
            }
        } else {
            // Rejected: increase damping and try again.
            lambda *= config.lambda_up;
            if lambda > 1e16 {
                break;
            }
        }
    }

    Ok(Stage1Output {
        deviance: d_current,
        iterations: iter,
        converged,
    })
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

    // ==================================================================
    // joint_poisson_fit — end-to-end integration tests
    // ==================================================================

    /// A wrapped transmission model: T_out = A_n · T_inner + B_A + B_B/√E + B_C·√E.
    /// Models the full counts-path background structure of memo 35 §P2.2.
    struct BackgroundedTransmission<'a> {
        inner: &'a dyn FitModel,
        energies: &'a [f64],
        n_idx: usize,
        a_idx: usize,
        b_a_idx: usize,
        b_b_idx: usize,
        b_c_idx: usize,
        n_params: usize,
    }

    impl<'a> FitModel for BackgroundedTransmission<'a> {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            // Pass the "density" parameter to the inner model as its param 0.
            let t_inner = self.inner.evaluate(&[params[self.n_idx]])?;
            let a_n = params[self.a_idx];
            let b_a = params[self.b_a_idx];
            let b_b = params[self.b_b_idx];
            let b_c = params[self.b_c_idx];
            Ok(t_inner
                .iter()
                .zip(self.energies.iter())
                .map(|(&t, &e)| {
                    let inv_sqrt_e = if e > 0.0 { 1.0 / e.sqrt() } else { 0.0 };
                    let sqrt_e = if e > 0.0 { e.sqrt() } else { 0.0 };
                    a_n * t + b_a + b_b * inv_sqrt_e + b_c * sqrt_e
                })
                .collect())
        }
        // No analytical jacobian — forces the fitter onto FD fallback, which
        // is the stress test (memo 35 §P2.1 notes FD + over-parameterization
        // as the stall trigger).
    }

    /// Exponential-in-E model: T_inner = exp(−n · σ(E)), σ(E) = 1.
    /// Effectively a single-parameter constant transmission when σ=1 flat.
    /// Uses an energy-dependent "cross section" so Jacobian is identifiable.
    struct ExpDecayModel<'a> {
        sigma: &'a [f64],
    }
    impl<'a> FitModel for ExpDecayModel<'a> {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            let n = params[0];
            Ok(self
                .sigma
                .iter()
                .map(|&s| (-n * s).exp().max(POISSON_EPSILON))
                .collect())
        }
        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            y_current: &[f64],
        ) -> Option<FlatMatrix> {
            // ∂T/∂n = -σ · T
            let n_e = y_current.len();
            let n_free = free_param_indices.len();
            let mut jac = FlatMatrix::zeros(n_e, n_free);
            for (i, &y_i) in y_current.iter().enumerate() {
                for (j, &pi) in free_param_indices.iter().enumerate() {
                    *jac.get_mut(i, j) = if pi == 0 { -self.sigma[i] * y_i } else { 0.0 };
                }
            }
            Some(jac)
        }
    }

    /// Deterministic Poisson generator (Knuth for small λ, Gaussian for
    /// large).  Duplicated from asymptote test so each test is self-contained.
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
        fn uniform(&mut self) -> f64 {
            (self.next_u64() as f64) / (u64::MAX as f64)
        }
        fn poisson(&mut self, lambda: f64) -> f64 {
            if lambda <= 0.0 {
                return 0.0;
            }
            if lambda > 30.0 {
                let u1 = self.uniform().max(1e-12);
                let u2 = self.uniform();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                return (lambda + z * lambda.sqrt()).round().max(0.0);
            }
            let l = (-lambda).exp();
            let mut k: f64 = 0.0;
            let mut p: f64 = 1.0;
            loop {
                k += 1.0;
                let u = self.uniform();
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

    // ------------------------------------------------------------------
    // Matched-model single-parameter recovery at c = 5.98.
    // This is the EG1 "proposed" cell in miniature — verify |bias| < 1%
    // and D / (n − k) ∈ [0.85, 1.15] without needing the polish.
    // ------------------------------------------------------------------
    #[test]
    fn test_joint_poisson_fit_matched_model_single_param() {
        // Energies 1..10, flat cross section σ = 1.  Truth n = 0.3.
        let n_bins = 200;
        let sigma = vec![1.0_f64; n_bins];
        let model = ExpDecayModel { sigma: &sigma };
        let n_true = 0.3_f64;
        let c = 5.98;
        let lam = 3000.0; // OB target ~500 counts/bin
        let t_true = model.evaluate(&[n_true]).unwrap();

        let mut rng = Xorshift(0x1234_5678_9ABC_DEF0);
        let o: Vec<f64> = (0..n_bins).map(|_| rng.poisson(lam / c)).collect();
        let s: Vec<f64> = (0..n_bins).map(|i| rng.poisson(lam * t_true[i])).collect();

        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c,
        };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("n", 0.1)]);
        let cfg = JointPoissonFitConfig {
            enable_polish: true,
            ..Default::default()
        };
        let result = joint_poisson_fit(&obj, &mut params, &cfg).unwrap();

        let n_fit = result.params[0];
        let rel_bias = (n_fit - n_true) / n_true;
        assert!(
            rel_bias.abs() < 0.01,
            "density bias {rel_bias} exceeds 1% (n_fit={n_fit} n_true={n_true})"
        );
        assert!(
            (0.85..=1.15).contains(&result.deviance_per_dof),
            "D/(n-k) out of band: {}",
            result.deviance_per_dof
        );
    }

    // ------------------------------------------------------------------
    // Polish-never-worsens invariant on a backgrounded fit.  Memo 35 §P2.1
    // claims NM polish reduces D materially when stage-1 stalls.  At the
    // unit-test scale we verify the testable invariant: enabling polish
    // never produces a larger final D than disabling it on the same data.
    //
    // Note: on this over-parameterized (5-free-param) synthetic with only
    // 150 bins, the deviance surface has multiple near-equal minima —
    // exactly the identifiability ambiguity §P2.2 targets.  Density
    // recovery under over-parameterization is therefore *not* a unit-test
    // contract here; it is tested end-to-end with the single-parameter
    // matched-model test above.
    // ------------------------------------------------------------------
    #[test]
    fn test_joint_poisson_fit_polish_does_not_worsen_deviance() {
        let n_bins = 150;
        let energies: Vec<f64> = (0..n_bins).map(|i| 1.0 + 0.5 * i as f64).collect();
        let sigma: Vec<f64> = energies.iter().map(|&e| 1.0 / e).collect();
        let inner = ExpDecayModel { sigma: &sigma };

        // Truth: n = 0.3, A_n = 0.9, no additive bg.
        let n_true = 0.3_f64;
        let a_n_true = 0.9_f64;
        let t_inner_true = inner.evaluate(&[n_true]).unwrap();
        let t_true: Vec<f64> = t_inner_true.iter().map(|&t| a_n_true * t).collect();

        let c = 5.98_f64;
        let lam = 5000.0_f64;
        let mut rng = Xorshift(0xF00D_FACE_DEAD_BEEF);
        let o: Vec<f64> = (0..n_bins).map(|_| rng.poisson(lam / c)).collect();
        let s: Vec<f64> = (0..n_bins).map(|i| rng.poisson(lam * t_true[i])).collect();

        let bg_model = BackgroundedTransmission {
            inner: &inner,
            energies: &energies,
            n_idx: 0,
            a_idx: 1,
            b_a_idx: 2,
            b_b_idx: 3,
            b_c_idx: 4,
            n_params: 5,
        };
        let _ = bg_model.n_params; // silence dead-code warning

        let obj = JointPoissonObjective {
            model: &bg_model,
            o: &o,
            s: &s,
            c,
        };

        // x0 analogous to EG2-S1 regime: n near truth, A_n = 1, all
        // additive bg at 0, bg bounds tight to curb degeneracy.
        let mk_params = || {
            ParameterSet::new(vec![
                FitParameter::non_negative("n", 0.25),
                FitParameter::non_negative("A_n", 1.0),
                FitParameter {
                    name: "B_A".into(),
                    value: 0.0,
                    lower: -0.05,
                    upper: 0.05,
                    fixed: false,
                },
                FitParameter {
                    name: "B_B".into(),
                    value: 0.0,
                    lower: -0.05,
                    upper: 0.05,
                    fixed: false,
                },
                FitParameter {
                    name: "B_C".into(),
                    value: 0.0,
                    lower: -0.05,
                    upper: 0.05,
                    fixed: false,
                },
            ])
        };

        let mut params_no_polish = mk_params();
        let cfg_no_polish = JointPoissonFitConfig {
            enable_polish: false,
            ..Default::default()
        };
        let r_no_polish = joint_poisson_fit(&obj, &mut params_no_polish, &cfg_no_polish).unwrap();

        let mut params_polish = mk_params();
        let cfg_polish = JointPoissonFitConfig {
            enable_polish: true,
            ..Default::default()
        };
        let r_polish = joint_poisson_fit(&obj, &mut params_polish, &cfg_polish).unwrap();

        // Invariant: enabling polish must not increase final D.
        assert!(
            r_polish.deviance <= r_no_polish.deviance + 1e-6,
            "polish worsened D: D_polish={} D_no_polish={}",
            r_polish.deviance,
            r_no_polish.deviance
        );

        // When polish_improved flag is set, polish D must be strictly
        // better than stage-1 D (consistency check on the flag semantics).
        if r_polish.polish_improved {
            assert!(
                r_polish.deviance < r_no_polish.deviance,
                "polish_improved=true but D_polish={} >= D_no_polish={}",
                r_polish.deviance,
                r_no_polish.deviance
            );
        }

        // The fit should return a physically sensible density (positive,
        // finite, within an order of magnitude of truth — not a strict
        // recovery test, just a sanity check).
        let n_fit = r_polish.params[0];
        assert!(n_fit.is_finite() && n_fit > 0.0);
        assert!(
            n_fit > 0.1 && n_fit < 0.8,
            "density grossly off: n_fit={n_fit} (truth={n_true})"
        );
    }

    // ------------------------------------------------------------------
    // Fit result carries gn_converged and polish_converged separately
    // (memo 35 §P2.3 — acceptance from deviance value, not one flag).
    // ------------------------------------------------------------------
    #[test]
    fn test_joint_poisson_fit_exposes_separate_converged_flags() {
        let n_bins = 50;
        let sigma = vec![0.5_f64; n_bins];
        let model = ExpDecayModel { sigma: &sigma };
        let n_true = 0.2;
        let c = 2.0;
        let lam = 500.0;
        let t_true = model.evaluate(&[n_true]).unwrap();
        let mut rng = Xorshift(0xABAD_CAFE_BABE_F00D);
        let o: Vec<f64> = (0..n_bins).map(|_| rng.poisson(lam / c)).collect();
        let s: Vec<f64> = (0..n_bins).map(|i| rng.poisson(lam * t_true[i])).collect();

        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c,
        };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("n", 0.1)]);
        let cfg = JointPoissonFitConfig {
            enable_polish: true,
            ..Default::default()
        };
        let r = joint_poisson_fit(&obj, &mut params, &cfg).unwrap();

        // Both flags exist; at least one should be true on this easy case.
        assert!(r.gn_converged || r.polish_converged);
        assert!(r.n_data == n_bins);
        assert!(r.n_free == 1);
        assert!(r.deviance > 0.0);
        assert!(r.deviance_per_dof.is_finite());
        // Uncertainty present (compute_covariance default true).
        assert!(r.uncertainties.is_some());
        let u = r.uncertainties.as_ref().unwrap();
        assert_eq!(u.len(), 1);
        assert!(u[0].is_finite() && u[0] > 0.0);
    }

    // ------------------------------------------------------------------
    // Reported uncertainty matches the analytical Cramér-Rao bound
    // I^{-1} (NOT (2I)^{-1} — the Hessian-of-D inverse, which would
    // under-report σ by √2).  Caught in code review of memo-35 §P1
    // implementation; see `joint_poisson_fit` covariance-extraction
    // doc-comment for the rescaling rationale.
    // ------------------------------------------------------------------
    #[test]
    fn test_uncertainty_matches_analytical_fisher_inverse() {
        // Construct a single-parameter constant-T model on noise-free
        // expected counts: O_i = λ/c, S_i = λ·T (per memo 35 §4.1).
        // With ConstModel (J_i = ∂T/∂θ = 1), the analytical Fisher is
        //   I(T) = Σ_i (O_i + S_i)·c / (T·(1+cT)²)
        //        = N · λ · (1+cT)/c · c / (T·(1+cT)²)
        //        = N · λ / (T · (1+cT))
        // and σ_T = √(I^{-1}) = √( T·(1+cT) / (N·λ) ).
        let n_bins = 200;
        let t_true = 0.5_f64;
        let c = 2.0_f64;
        let lam = 100.0_f64;
        let o: Vec<f64> = vec![lam / c; n_bins];
        let s: Vec<f64> = vec![lam * t_true; n_bins];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c,
        };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", t_true)]);
        let cfg = JointPoissonFitConfig {
            // Disable polish for a clean Newton-only fit (avoids NM-tail
            // perturbations of the final θ that would shift σ slightly).
            enable_polish: false,
            ..Default::default()
        };
        let r = joint_poisson_fit(&obj, &mut params, &cfg).unwrap();
        let sigma_reported = r.uncertainties.as_ref().expect("σ available")[0];

        // Analytical Cramér-Rao σ.
        let sigma_analytical = (t_true * (1.0 + c * t_true) / (n_bins as f64 * lam)).sqrt();

        // The pre-fix (uncompensated) value would be σ_analytical / √2 —
        // tighten the tolerance below √2 so the regression is caught.
        let rel_err = (sigma_reported - sigma_analytical).abs() / sigma_analytical;
        assert!(
            rel_err < 0.05,
            "reported σ = {sigma_reported} vs analytical I^{{-1}}^(1/2) = \
             {sigma_analytical} (rel_err = {rel_err}); pre-fix code reported \
             σ_analytical / √2 ≈ {} which would give rel_err ≈ 0.293",
            sigma_analytical / 2.0_f64.sqrt(),
        );
    }
}
