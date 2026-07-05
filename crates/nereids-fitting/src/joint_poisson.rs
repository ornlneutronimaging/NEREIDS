//! Joint-Poisson counts-path objective with profiled flux.
//!
//! This module implements the **joint-Poisson conditional binomial deviance**:
//! the per-bin flux is profiled out of a two-arm Poisson model analytically
//! (derivation below).  The deviance is validated against synthetic
//! counts benchmarks and locked by a real-VENUS counts regression test
//! on the committed aggregated-Hf fixture.  It supersedes
//! the fixed-flux Poisson NLL (`poisson.rs`) for the counts-path fitter.
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
//! scaled with the proton-charge ratio `c` at constant density fidelity).

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
    /// Optional per-bin active mask (SAMMY EMIN/EMAX-equivalent
    /// fit-energy-range restriction).  When `Some(m)`, only bins where
    /// `m[i]` is `true` contribute to the deviance / gradient / Fisher
    /// information; the model is still evaluated on the full grid so
    /// resolution broadening at the boundaries is correct.  When
    /// `None`, all bins are active (default behaviour).
    ///
    /// Length must equal `o.len()`; the GUI / pipeline dispatch builds
    /// it from the configured `[E_min, E_max]` against the energy grid.
    pub active_mask: Option<&'a [bool]>,
}

impl<'a> JointPoissonObjective<'a> {
    /// Number of data bins.
    pub fn n_data(&self) -> usize {
        self.o.len()
    }

    /// Number of *active* data bins — `n_data` when no mask is set,
    /// or the count of `true` entries in `active_mask` otherwise.
    /// This is the count that should drive deviance-per-dof reporting.
    pub fn n_active(&self) -> usize {
        crate::active_mask::active_count(self.active_mask, self.o.len())
    }

    /// Predicate: is bin `i` active?  Returns `true` when no mask is
    /// set (full-grid default).
    #[inline]
    fn bin_active(&self, i: usize) -> bool {
        self.active_mask.is_none_or(|m| m[i])
    }

    /// Runtime guard for the public methods that bypass `joint_poisson_fit`'s
    /// up-front validation (callers may invoke `deviance_from_transmission`,
    /// `deviance_gradient_analytical`, `fisher_information[_fd]`, etc.
    /// directly for diagnostics).  Mirrors the entry-point checks in
    /// `joint_poisson_fit`: `o.len() == s.len()`, `c` finite and > 0, optional
    /// `active_mask` length agrees, all `o[i]` / `s[i]` finite and >= 0, and
    /// the caller-supplied transmission length agrees with `o.len()`.  The
    /// `debug_assert!`s in the per-bin helpers are no-ops in release builds —
    /// without this guard a length mismatch in `s` would silently truncate
    /// via `.zip()` and a non-positive / NaN `c` would produce finite
    /// garbage.
    ///
    /// **Error orientation.**  `FittingError::LengthMismatch` displays as
    /// `"{field} length ({actual}) must match expected length ({expected})"`.
    /// The objective's own invariants (`s.len()` vs `o.len()`, `mask.len()`
    /// vs `o.len()`) are checked first, with `expected = o.len()` so the
    /// message accurately names the offending field.  The caller-supplied
    /// `t` length is then checked against `o.len()` with `field =
    /// "transmission"` — pre-fix this branch reported `field =
    /// "open_beam_counts"` with `expected = t_len`, which read as "the
    /// open-beam array is wrong" when the actual fault was the caller's
    /// transmission slice.
    fn validate_inputs(&self, t_len: usize) -> Result<(), FittingError> {
        // Internal invariants of the objective itself — these must hold
        // regardless of what the caller passes for `t`.
        if self.s.len() != self.o.len() {
            return Err(FittingError::LengthMismatch {
                expected: self.o.len(),
                actual: self.s.len(),
                field: "sample_counts",
            });
        }
        if let Some(m) = self.active_mask
            && m.len() != self.o.len()
        {
            return Err(FittingError::LengthMismatch {
                expected: self.o.len(),
                actual: m.len(),
                field: "active_mask",
            });
        }
        if !self.c.is_finite() || self.c <= 0.0 {
            return Err(FittingError::InvalidConfig(format!(
                "proton-charge ratio c = Q_s/Q_ob must be finite and > 0, got {}",
                self.c
            )));
        }
        // Caller-supplied length: the transmission slice must match the
        // objective's bin count.
        if t_len != self.o.len() {
            return Err(FittingError::LengthMismatch {
                expected: self.o.len(),
                actual: t_len,
                field: "transmission",
            });
        }
        // Per-element count validation.  The entry point `joint_poisson_fit`
        // also calls `validate_counts` up-front so the user gets the error
        // before any LM work, but every public method that bypasses the
        // entry point (`deviance_from_transmission`, `fisher_information`,
        // `profile_lambda_per_bin`, …) must still reject non-finite /
        // negative counts — the inner `binomial_deviance_term` /
        // `xlogy_ratio` would otherwise propagate NaN past the zero-clamp
        // (`NaN <= 0.0` is `false`) or silently swallow a negative as zero.
        validate_counts(self.o, "open_beam_counts")?;
        validate_counts(self.s, "sample_counts")?;
        Ok(())
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
    ///
    /// Validates `t.len() == o.len() == s.len()` and `c > 0`; returns
    /// `FittingError::LengthMismatch` / `InvalidConfig` rather than the
    /// previous `.zip()` truncate-and-pretend behaviour (which would
    /// silently shrink the output to `min(t.len(), o.len(), s.len())`).
    pub fn profile_lambda_per_bin(&self, t: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.validate_inputs(t.len())?;
        Ok(t.iter()
            .zip(self.o.iter())
            .zip(self.s.iter())
            .map(|((&ti, &oi), &si)| self.profile_lambda(ti, oi, si))
            .collect())
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
    pub fn deviance_from_transmission(&self, t: &[f64]) -> Result<f64, FittingError> {
        self.validate_inputs(t.len())?;
        let mut d = 0.0;
        // Total active counts — sets the scale of legitimate xlogy round-off
        // (each term cancels quantities of magnitude ~(O+S), so the summed
        // error is O(ε · Σ(O+S))).
        let mut weight_scale = 0.0;
        for (i, ((&t_i, &o_i), &s_i)) in t.iter().zip(self.o.iter()).zip(self.s.iter()).enumerate()
        {
            if !self.bin_active(i) {
                continue;
            }
            d += binomial_deviance_term(s_i, o_i, t_i, self.c);
            weight_scale += o_i + s_i;
        }
        // Each conditional-binomial term is nonnegative by Gibbs' inequality,
        // so D ≥ 0 is a hard mathematical invariant — but the per-term xlogy
        // cancellations carry O(ULP·count) round-off, and when the optimizer
        // converges machine-exactly on noise-free (synthetic) data the sum
        // can land at ~−1e-13.  Clamp WITHIN the round-off envelope only
        // (review R2): an unbounded clamp would convert a genuine
        // deviance-term defect producing a large negative / −∞ sum into a
        // silent "perfect converged fit" (the D == 0 convergence shortcut in
        // damped_fisher_stage).  Beyond the envelope, surface the defect as
        // an Err — at the initial point that propagates to the caller, and
        // mid-iteration it rejects trials until the fit reports
        // non-converged, both honest failure signals.  NaN from a
        // non-finite model row passes through first: NaN.max(0.0) = 0.0
        // would be wrong (f64::max returns the OTHER operand for NaN).
        if d.is_nan() {
            return Ok(d);
        }
        // 64× margin over the observed magnitude class (−1.5e-13 at
        // Σ(O+S) ≈ 4e6, i.e. ~ε·Σ ≈ 1e-9 worst-case).
        let roundoff_envelope = 64.0 * f64::EPSILON * weight_scale.max(1.0);
        if d < -roundoff_envelope {
            return Err(FittingError::EvaluationFailed(format!(
                "conditional-binomial deviance D = {d} is negative beyond the \
                 accumulation round-off envelope ({roundoff_envelope:.3e}) — \
                 this indicates a deviance-term defect, not round-off; \
                 refusing to clamp it to a perfect fit"
            )));
        }
        Ok(d.max(0.0))
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
        self.deviance_from_transmission(&t)
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
        self.validate_inputs(t.len())?;
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
            if !self.bin_active(i) {
                continue;
            }
            let w = deviance_weight(s_i, o_i, t_i, self.c);
            // `deviance_weight` returns 0 for non-finite `t_i`, so a NaN
            // transmission row already contributes nothing — except that
            // `0.0 * NaN = NaN`.  If the upstream Jacobian column has a
            // NaN cell (common for FD-built Jacobians where the model
            // returns NaN at some probe point), the bare `0.0 * jac.get(...)`
            // would poison `grad[col]`.  Skip the row entirely when the
            // weight is zero, and skip any individual Jacobian cell that
            // is not finite.
            if w == 0.0 {
                continue;
            }
            for (g, col) in grad.iter_mut().zip(0..n_free) {
                let j = jac.get(i, col);
                if j.is_finite() {
                    *g += w * j;
                }
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
        self.validate_inputs(t.len())?;
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
            if !self.bin_active(i) {
                continue;
            }
            let h = deviance_curvature(s_i, o_i, t_i, self.c);
            // Mirror the gradient guard: `deviance_curvature` returns 0
            // for non-finite `t_i`, but `0.0 * NaN = NaN` would still
            // poison the Fisher matrix when an FD-built Jacobian has a
            // NaN cell.  Skip the row at h == 0, and skip cells that are
            // not finite.
            if h == 0.0 {
                continue;
            }
            for j in 0..n_free {
                let jij = jac.get(i, j);
                if !jij.is_finite() {
                    continue;
                }
                for k in 0..n_free {
                    let jik = jac.get(i, k);
                    if jik.is_finite() {
                        *info.get_mut(j, k) += h * jij * jik;
                    }
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
        self.validate_inputs(t_base.len())?;
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
            // Per-cell finiteness check.  The matching guard in lm.rs
            // `compute_jacobian` zeroes NaN entries instead of dropping
            // the column; the same pattern applies here because the
            // downstream Fisher accumulator below already skips inactive
            // rows (`bin_active(i)`), so a NaN at a masked / inactive
            // row must not block the column for active rows.  Active-row
            // NaN is handled by [`deviance_curvature`], which returns 0
            // on non-finite `t_i` so the assembly stays clean.
            for i in 0..n_e {
                let a = t_a[i];
                let b = t_b[i];
                if a.is_finite() && b.is_finite() {
                    *jac.get_mut(i, col) = (a - b) / denom;
                }
                // else: leave at the zero-default; masked rows are never
                // read by the active-bin filter in the Fisher loop.
            }
        }
        let mut info = FlatMatrix::zeros(n_free, n_free);
        for (i, ((&t_i, &o_i), &s_i)) in t_base
            .iter()
            .zip(self.o.iter())
            .zip(self.s.iter())
            .enumerate()
        {
            if !self.bin_active(i) {
                continue;
            }
            let h = deviance_curvature(s_i, o_i, t_i, self.c);
            // Same guard as the analytical `fisher_information`: avoid
            // `0.0 * NaN = NaN` poisoning the matrix from NaN Jacobian
            // cells (per-cell zero default from the FD loop above leaves
            // most NaN entries as 0, but a stale value from a partial
            // FD failure must still be defensively skipped).
            if h == 0.0 {
                continue;
            }
            for j in 0..n_free {
                let jij = jac.get(i, j);
                if !jij.is_finite() {
                    continue;
                }
                for k in 0..n_free {
                    let jik = jac.get(i, k);
                    if jik.is_finite() {
                        *info.get_mut(j, k) += h * jij * jik;
                    }
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
            // After the NaN-T contract in `binomial_deviance_term`,
            // `self.deviance` can legitimately return `Ok(NaN)` when a
            // probe lands in a region where the model produces a
            // non-finite transmission.  A non-finite `perturbed_d`
            // divided by `actual_step` would write NaN into `grad[j]`
            // and poison every subsequent step that consumes the
            // gradient — symmetric with the `Err` branch below.  Treat
            // both as "this probe is invalid; leave the column at 0".
            let perturbed_d = match self.deviance(&perturbed_values) {
                Ok(v) if v.is_finite() => v,
                _ => {
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
/// NaN-T contract (see also [`deviance_weight`] / [`deviance_curvature`]):
///
/// - For `0 ≤ T ≤ POISSON_EPSILON` (finite but numerically tiny or zero):
///   clamps `T` to `POISSON_EPSILON` in the denominator so the optimizer
///   sees a finite (large) D and a continuous gradient.  This is the
///   "smooth guard" path.
/// - For **non-finite** `T` (NaN or ±∞): returns `NaN` so the deviance
///   sum becomes `NaN` and the LM / damped-Fisher trial-step guards
///   (`Ok(v) if v.is_finite()`) reject the step.  This deliberately does
///   *not* clamp via `f64::max`, because `f64::max(NaN, ε)` returns `ε`
///   — which would silently masquerade as a valid bin.
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
    // `f64::max(NaN, ε)` returns `ε`, so a non-finite T would silently
    // masquerade as a tiny positive transmission and the deviance would
    // evaluate to a finite (but meaningless) value that the LM trial-step
    // guard `Ok(v) if v.is_finite()` would accept.  Return NaN so the
    // deviance sum becomes NaN and the trial step is rejected.  The
    // matching `deviance_weight` / `deviance_curvature` guards return 0,
    // which keeps the gradient / Fisher accumulators clean rather than
    // poisoning them with NaN contributions.
    if !t.is_finite() {
        return f64::NAN;
    }
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

/// Reject non-finite or negative count arrays at public entry points.
///
/// Two distinct failure modes motivate the up-front check:
///
/// - **Non-finite (NaN / ±∞).**  The per-bin `xlogy_ratio` helper treats
///   `x <= 0.0` as the zero-count branch and returns 0, but `NaN <= 0.0`
///   is `false`, so a NaN slips past the branch and propagates
///   `NaN · ln(NaN / y) = NaN` straight into the deviance sum.  The LM
///   trial-step guard then sees `Ok(NaN)` instead of a clean error.
/// - **Negative.**  `x <= 0.0` *is* true for `x < 0.0`, so the zero-count
///   branch silently swallows negatives and returns 0 — the deviance
///   stays finite but the bin is treated as "no data", which is
///   physically meaningless and conceals the upstream bug (subtraction
///   artefact in TOF normalisation, signed-int overflow in the loader,
///   etc.).  Negatives never produce NaN, but the "successful" fit
///   silently discards real data.
///
/// Validate up-front so callers get a typed `InvalidConfig` error
/// pointing at the offending bin instead of either failure mode.
fn validate_counts(counts: &[f64], field: &'static str) -> Result<(), FittingError> {
    for (i, &v) in counts.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Err(FittingError::InvalidConfig(format!(
                "{field}[{i}] must be finite and >= 0, got {v}"
            )));
        }
    }
    Ok(())
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
    // A non-finite T must not be folded into the gradient accumulator.
    // `f64::max(NaN, ε)` returns `ε`, which would turn a NaN bin into a
    // finite gradient contribution scaled by the Jacobian and silently
    // steer the optimizer.  Skip the bin (return 0) — the matching
    // `binomial_deviance_term` returns NaN so the step is rejected by
    // the trial-guard, but the gradient stays clean in case the caller
    // is using it for diagnostics on a partially-bad grid.
    if !t.is_finite() {
        return 0.0;
    }
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
    // See the matching guard in [`deviance_weight`].  A non-finite T
    // would otherwise contribute a huge spurious curvature via
    // `f64::max(NaN, ε) -> ε`, inflating the diagonal of the Fisher
    // matrix and underestimating the corresponding parameter
    // uncertainty (covariance = I⁻¹ entries shrink as I grows).
    if !t.is_finite() {
        return 0.0;
    }
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
    /// (`xatol = 1e-9, fatol = 1e-10`) were originally matched to a
    /// synthetic counts benchmark where D stays O(1), so `fatol` is
    /// physically meaningful.  On real-data regimes where D saturates
    /// at 10⁴–10⁵ (un-modelled upstream physics), `fatol / D` drops
    /// below f64 ULP and polish
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
    /// match the synthetic counts-benchmark tolerances — physically
    /// meaningful when `D ≈ 1` (clean data) but sub-f64-ULP on real
    /// counts where `D ≈ 10⁴`–`10⁵`, which is why `enable_polish`
    /// defaults to `false`.  See #486.
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
                // Tolerances tuned for the synthetic D ≈ 1 regime —
                // `fatol = 1e-10` vs D ≈ 1 is a physically
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
    /// D / (n − k).  The primary goodness-of-fit statistic for the counts path.
    pub deviance_per_dof: f64,
    /// Number of data bins on the configured grid (n).  This is the
    /// total bin count; when a fit-energy-range mask is in effect, the
    /// count of bins that actually contributed to the cost function is
    /// reported separately as [`Self::n_active`].
    pub n_data: usize,
    /// Number of *active* data bins — equal to `n_data` when no mask is
    /// set, or the count of `true` entries in the objective's
    /// `active_mask` otherwise.  The deviance / dof ratio uses
    /// `(n_active − n_free)` so reduced deviance is unbiased when a
    /// fit-energy-range mask is in effect (SAMMY EMIN/EMAX semantics, #514).
    pub n_active: usize,
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
/// **Counts-path contract** this function satisfies:
///
/// - Minimizes the **conditional binomial deviance** `D(θ)`
///   ([`JointPoissonObjective::deviance`]), not fixed-flux Poisson NLL.
/// - Reports `D / (n − k)` as the primary GOF.
/// - Honours an **explicit `c = Q_s/Q_ob`** stored in the objective.
/// - Runs Nelder-Mead **polish** after the gradient stage to escape the
///   initial-point stall seen on backgrounded fits.
/// - Exposes `gn_converged` and `polish_converged` separately so callers
///   do not rely on a single "success" flag — acceptance is meant to come
///   from the deviance value.
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

    // Validate `o` / `s` length and `c` up-front at the public entry
    // point.  The inner per-bin helpers (`binomial_deviance_term`,
    // `deviance_from_transmission`) use `debug_assert!` only, which is a
    // no-op in release builds.  Without these hard checks:
    //   - A length mismatch in `o` vs `s` silently truncates via `.zip()`,
    //     minimising deviance on a sub-range of bins.
    //   - A non-positive or non-finite `c` produces finite garbage
    //     (e.g. zero `cT`, NaN denominators) that the LM happily descends.
    //   - A NaN / negative `o[i]` or `s[i]` would slip past the inner
    //     `xlogy_ratio` zero-clamp (`x <= 0.0` swallows negatives, but
    //     `NaN <= 0.0` is `false` so a NaN count bleeds straight into the
    //     log and out into the deviance sum).
    // All surface as "the fit converged" with bogus parameter values —
    // exactly the failure mode the trial-step guard cannot catch because
    // the deviance value is finite.
    if objective.s.len() != n_data {
        return Err(FittingError::LengthMismatch {
            expected: n_data,
            actual: objective.s.len(),
            field: "sample_counts",
        });
    }
    if !objective.c.is_finite() || objective.c <= 0.0 {
        return Err(FittingError::InvalidConfig(format!(
            "proton-charge ratio c = Q_s/Q_ob must be finite and > 0, got {}",
            objective.c
        )));
    }
    validate_counts(objective.o, "open_beam_counts")?;
    validate_counts(objective.s, "sample_counts")?;

    // Validate active-mask length up-front, mirroring the LM solver's
    // length-mismatch early-return (#514).  A debug-assert deep in the
    // deviance routines would silently pass through in release builds
    // with a length mismatch, causing out-of-bounds index reads when
    // the masked accumulator iterates `o`/`s`/`mask` together.
    if let Some(m) = objective.active_mask
        && m.len() != n_data
    {
        return Err(FittingError::LengthMismatch {
            expected: n_data,
            actual: m.len(),
            field: "active_mask",
        });
    }

    // SAMMY EMIN/EMAX-equivalent fit-energy-range (#514): zero active
    // bins means the user's `[E_min, E_max]` does not overlap the
    // configured grid.  No data contributes to the deviance — return
    // non-converged with NaN before falling through.  Without this
    // the all-bins-masked path would compute deviance = 0 (sum over
    // zero rows) which combined with `n_free == 0` (all-fixed params)
    // would report `gn_converged: true, deviance: 0` from a fit that
    // saw no data.
    let n_free_initial = params.n_free();
    let n_active_initial = objective.n_active();
    if n_active_initial == 0 {
        return Ok(JointPoissonResult {
            deviance: f64::NAN,
            deviance_per_dof: f64::NAN,
            n_data,
            n_active: 0,
            n_free: n_free_initial,
            gn_iterations: 0,
            polish_iterations: 0,
            gn_converged: false,
            polish_converged: false,
            polish_improved: false,
            params: params.all_values(),
            covariance: None,
            uncertainties: None,
        });
    }

    // Underdetermined-check: when a fit-energy-range mask leaves fewer
    // active bins than free parameters, the problem is rank-deficient
    // and any deviance / dof ratio would be deceptive (the previous
    // `.max(1)` divisor produced a finite-looking deviance-per-dof for
    // empty / too-narrow masks).  Mirror LM's behaviour at
    // `lm.rs:578-588`: return a non-converged result up-front, before
    // wasting cycles on the damped-Fisher stage.
    if n_active_initial < n_free_initial {
        return Ok(JointPoissonResult {
            deviance: f64::NAN,
            deviance_per_dof: f64::NAN,
            n_data,
            n_active: n_active_initial,
            n_free: n_free_initial,
            gn_iterations: 0,
            polish_iterations: 0,
            gn_converged: false,
            polish_converged: false,
            polish_improved: false,
            params: params.all_values(),
            covariance: None,
            uncertainties: None,
        });
    }

    // Stage 1: damped Fisher with Armijo backtracking.
    let stage1 = damped_fisher_stage(objective, params, config)?;

    // Capture stage-1 best.
    let best_d_stage1 = stage1.deviance;
    let gn_iterations = stage1.iterations;
    let gn_converged = stage1.converged;

    // Stage 2: Nelder-Mead polish on free parameters, seeded from stage-1 θ.
    //
    // Guard against the all-fixed configuration: `nelder_mead_minimize`
    // requires a non-empty `x0` (asserts in `nelder_mead.rs`).  When every
    // parameter is fixed there is nothing to polish, so skip stage 2 and
    // leave the polish flags at their default `false` values.  This path
    // is reachable from pipeline callers that pin all params and set
    // `with_counts_enable_polish(Some(true))`.
    //
    // Also short-circuit polish when stage 1 ended on a non-finite
    // deviance: there is no meaningful starting deviance to refine, and
    // the acceptance test `nm.fun < best_d_stage1` would degrade to
    // `finite < NaN == false` (discarding the NM result) while
    // `nm.self_converged` could still be `true`, leaking a spurious
    // converged flag together with a NaN final deviance.  Mirrors the
    // LM `n_free == 0` early-return at `lm.rs:584-607`, which refuses to
    // report a converged fit when the model emits NaN at active bins.
    let mut polish_iterations = 0usize;
    let mut polish_converged = false;
    let mut polish_improved = false;
    let free_idx = params.free_indices();
    if config.enable_polish && !free_idx.is_empty() && best_d_stage1.is_finite() {
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
    // Active-bin masking (SAMMY EMIN/EMAX): when a fit-energy-range mask
    // is in effect, dof must use the count of bins that contributed to
    // the deviance — otherwise deviance-per-dof is biased low by the
    // ratio (n_active / n_data).  The `n_active < n_free` case has
    // already been short-circuited above; here `n_active >= n_free`,
    // so `dof` is non-negative and exactly-determined fits
    // (`n_active == n_free`) report `deviance_per_dof = NaN` (0/0)
    // as in LM (`lm.rs:784`).
    let n_active = objective.n_active();
    let dof = n_active.saturating_sub(n_free);
    let deviance_per_dof = if dof > 0 {
        final_deviance / dof as f64
    } else {
        f64::NAN
    };

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
    // standard errors by √2 × — a real ½-scaling bug.  We rescale
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
        n_active,
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

        // D is a sum of nonnegative conditional-binomial terms (clamped at
        // 0 in `deviance_from_transmission`), so D == 0 is the exact global
        // minimum — reachable on noise-free synthetic data once the model
        // converges machine-exactly.  Declare convergence here: the Armijo
        // test can never accept a step from D == 0 (it demands a strict
        // decrease), so without this check the stage inflates λ past its
        // ceiling and reports a PERFECT fit as non-converged.
        if d_current == 0.0 {
            converged = true;
            break;
        }

        let free_idx = params.free_indices();
        let n_free = free_idx.len();
        if n_free == 0 {
            // All parameters fixed: we are not optimizing; convergence is
            // well-defined only if the already-computed deviance at the
            // current parameters is finite.  If the model returned
            // non-finite transmission, `binomial_deviance_term` propagates
            // that as NaN deviance (see the non-finite-T contract documented
            // on `binomial_deviance_term`), and a non-finite deviance cannot
            // be reported as a converged fit.  LM applies the same guard in
            // the `n_free == 0` branch of `levenberg_marquardt_with_mask`;
            // the matching LM regression is `test_all_fixed_params_nan_model`.
            converged = d_current.is_finite();
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
                active_mask: None,
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
            active_mask: None,
        };
        let d = obj.deviance_from_transmission(&t).unwrap();
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
            active_mask: None,
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
        // `Xorshift` is defined at the module level below — Rust item order
        // is not significant inside a module.
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
                active_mask: None,
            };

            // 1D grid search over T, then local refinement via Brent-like
            // bisection on the gradient sign.
            let grid: Vec<f64> = (0..200).map(|i| 0.01 + 0.99 * (i as f64) / 199.0).collect();
            let mut best = (grid[0], f64::INFINITY);
            for &t_try in &grid {
                let d_try = obj
                    .deviance_from_transmission(&vec![t_try; n_bins])
                    .unwrap();
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
            let d_hat = obj
                .deviance_from_transmission(&vec![t_hat; n_bins])
                .unwrap();
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
            active_mask: None,
        };
        let d_full = obj.deviance_from_transmission(&[0.6, 0.6, 0.6]).unwrap();
        // Drop the zero-N bin — result must be identical.
        let obj_reduced = JointPoissonObjective {
            model: &model, // same model, we just bypass the 1st bin via data
            o: &[10.0, 5.0],
            s: &[5.0, 2.0],
            c: 1.5,
            active_mask: None,
        };
        let d_reduced = obj_reduced.deviance_from_transmission(&[0.6, 0.6]).unwrap();
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
            active_mask: None,
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
            active_mask: None,
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
    /// Models the full counts-path background structure (normalization
    /// plus the three-term energy-dependent background).
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
        // is the stress test (FD + over-parameterization is the
        // empirically established stall trigger).
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
    /// large).  Shared across the stochastic-asymptote and joint-Poisson
    /// fit tests in this module.
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
    // A miniature of the validated matched-model configuration — verify |bias| < 1%
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
            active_mask: None,
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
    // Polish-never-worsens invariant on a backgrounded fit.  NM polish
    // is meant to reduce D materially when stage-1 stalls.  At the
    // unit-test scale we verify the testable invariant: enabling polish
    // never produces a larger final D than disabling it on the same data.
    //
    // Note: on this over-parameterized (5-free-param) synthetic with only
    // 150 bins, the deviance surface has multiple near-equal minima —
    // exactly the over-parameterization identifiability ambiguity the
    // B_A-pairing rule targets.  Density
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
            active_mask: None,
        };

        // x0 analogous to the stall-prone backgrounded regime: n near truth, A_n = 1, all
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
    // (acceptance is judged from the deviance value, not one flag).
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
            active_mask: None,
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
    // under-report σ by √2).  A real bug in the original
    // implementation; see `joint_poisson_fit` covariance-extraction
    // doc-comment for the rescaling rationale.
    // ------------------------------------------------------------------
    #[test]
    fn test_uncertainty_matches_analytical_fisher_inverse() {
        // Construct a single-parameter constant-T model on noise-free
        // expected counts: O_i = λ/c, S_i = λ·T (the module-doc model).
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
            active_mask: None,
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

    // ------------------------------------------------------------------
    // Active-bin mask (SAMMY EMIN/EMAX-equivalent fit-energy-range, #514).
    // ------------------------------------------------------------------

    /// `deviance_from_transmission` with `active_mask` set must equal
    /// the same call computed only over the `true` bins (subset
    /// equivalence) — the masking is correct iff dropping out-of-mask
    /// bins from `o`, `s`, `t` produces the same value.
    #[test]
    fn test_jp_active_mask_subset_equivalence() {
        // 5-bin objective with an arbitrary mask — bins 1 and 3 active.
        let o_full = [10.0, 20.0, 5.0, 15.0, 25.0];
        let s_full = [4.0, 8.0, 1.0, 6.0, 12.0];
        let t_full = [0.4, 0.5, 0.7, 0.6, 0.45];
        let mask = [false, true, false, true, false];
        let c = 1.5;
        let model_full = ConstModel { n_e: 5 };
        let obj_full = JointPoissonObjective {
            model: &model_full,
            o: &o_full,
            s: &s_full,
            c,
            active_mask: Some(&mask),
        };
        let d_masked = obj_full.deviance_from_transmission(&t_full).unwrap();

        // Compare against an objective built directly on the active subset.
        let o_sub = [o_full[1], o_full[3]];
        let s_sub = [s_full[1], s_full[3]];
        let t_sub = [t_full[1], t_full[3]];
        let model_sub = ConstModel { n_e: 2 };
        let obj_sub = JointPoissonObjective {
            model: &model_sub,
            o: &o_sub,
            s: &s_sub,
            c,
            active_mask: None,
        };
        let d_subset = obj_sub.deviance_from_transmission(&t_sub).unwrap();

        assert!(
            (d_masked - d_subset).abs() < 1e-12,
            "masked deviance {d_masked} != subset deviance {d_subset}"
        );

        // Active-bin count should be 2, not 5.
        assert_eq!(obj_full.n_active(), 2);
        assert_eq!(obj_full.n_data(), 5);
    }

    /// Out-of-mask gradient contributions must drop to zero — verified
    /// by comparing against an unmasked subset gradient.
    #[test]
    fn test_jp_active_mask_gradient_subset_equivalence() {
        let e_full: Vec<f64> = (0..6).map(|i| 0.1 + 0.1 * i as f64).collect();
        let theta_true = [0.95_f64, 0.05_f64];
        let c = 2.0;
        let lam = 100.0;
        let model_full = LinearModel { e: &e_full };
        let t_full = model_full.evaluate(&theta_true).unwrap();
        let o_full: Vec<f64> = vec![lam / c; e_full.len()];
        let s_full: Vec<f64> = t_full.iter().map(|&ti| lam * ti).collect();

        // Mask = bins 2..5 active.
        let mask = vec![false, false, true, true, true, false];
        let obj_full = JointPoissonObjective {
            model: &model_full,
            o: &o_full,
            s: &s_full,
            c,
            active_mask: Some(&mask),
        };

        let params_full = ParameterSet::new(vec![
            FitParameter::non_negative("a", theta_true[0]),
            FitParameter::non_negative("b", theta_true[1]),
        ]);
        let free_idx = params_full.free_indices();
        let theta_eval = [0.9_f64, 0.07_f64];
        let grad_masked = obj_full
            .deviance_gradient_analytical(&theta_eval, &free_idx)
            .unwrap()
            .expect("analytical gradient");

        // Subset reference: only bins 2..5.
        let e_sub = e_full[2..5].to_vec();
        let o_sub = o_full[2..5].to_vec();
        let s_sub = s_full[2..5].to_vec();
        let model_sub = LinearModel { e: &e_sub };
        let obj_sub = JointPoissonObjective {
            model: &model_sub,
            o: &o_sub,
            s: &s_sub,
            c,
            active_mask: None,
        };
        let grad_sub = obj_sub
            .deviance_gradient_analytical(&theta_eval, &free_idx)
            .unwrap()
            .expect("analytical gradient");

        for (i, (&gm, &gs)) in grad_masked.iter().zip(grad_sub.iter()).enumerate() {
            assert!(
                (gm - gs).abs() < 1e-9,
                "grad component {i}: masked={gm} subset={gs}"
            );
        }
    }

    /// `joint_poisson_fit` must reject an underdetermined (n_active <
    /// n_free) configuration with a non-converged result and NaN
    /// deviance / per-dof, mirroring the LM solver.  An all-`false`
    /// active mask is the extreme case (`n_active == 0 < n_free`);
    /// the prior `.max(1)` divisor produced a deceptive
    /// finite-looking deviance-per-dof for empty / too-narrow masks.
    /// Regression for #514.
    #[test]
    fn test_joint_poisson_rejects_zero_active_mask() {
        let n_bins = 10;
        let o: Vec<f64> = vec![50.0; n_bins];
        let s: Vec<f64> = vec![25.0; n_bins];
        let mask = vec![false; n_bins]; // n_active = 0
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: Some(&mask),
        };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        let cfg = JointPoissonFitConfig::default();
        let r = joint_poisson_fit(&obj, &mut params, &cfg).unwrap();

        assert!(
            !r.gn_converged && !r.polish_converged,
            "underdetermined fit must report non-converged"
        );
        assert!(
            r.deviance.is_nan(),
            "underdetermined deviance must be NaN, got {}",
            r.deviance
        );
        assert!(
            r.deviance_per_dof.is_nan(),
            "underdetermined deviance-per-dof must be NaN, got {}",
            r.deviance_per_dof
        );
        assert_eq!(r.n_data, n_bins);
        assert_eq!(r.n_active, 0);
        assert_eq!(r.n_free, 1);
        assert!(r.covariance.is_none());
        assert!(r.uncertainties.is_none());
    }

    /// Zero active bins with **all parameters fixed** (`n_free == 0`)
    /// must still return non-converged.  Without the
    /// `n_active == 0` early-return, the underdetermined check
    /// `n_active < n_free` is `0 < 0` → false, so the function would
    /// fall through to the main loop, compute `deviance = 0` from the
    /// empty sum, and `dof = 0` → `deviance_per_dof = NaN` — but
    /// `gn_converged` could still be `true`, masquerading as a
    /// successful fit on no data.  Regression for #517 (#514).
    #[test]
    fn test_joint_poisson_rejects_zero_active_with_no_free_params() {
        let n_bins = 5;
        let o: Vec<f64> = vec![10.0; n_bins];
        let s: Vec<f64> = vec![5.0; n_bins];
        let mask = vec![false; n_bins];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: Some(&mask),
        };
        let mut params = ParameterSet::new(vec![FitParameter::fixed("T", 0.5)]);
        let r = joint_poisson_fit(&obj, &mut params, &JointPoissonFitConfig::default()).unwrap();
        assert!(!r.gn_converged);
        assert!(!r.polish_converged);
        assert!(r.deviance.is_nan());
        assert!(r.deviance_per_dof.is_nan());
        assert_eq!(r.n_active, 0);
        assert_eq!(r.n_free, 0);
    }

    /// `joint_poisson_fit` validates active-mask length up-front and
    /// returns `LengthMismatch` rather than relying on a debug-assert
    /// deep in the deviance routines (which silently passes through in
    /// release builds, then panics on out-of-bounds index reads).
    /// Regression for #514.
    #[test]
    fn test_joint_poisson_rejects_active_mask_length_mismatch() {
        let n_bins = 5;
        let o: Vec<f64> = vec![10.0; n_bins];
        let s: Vec<f64> = vec![5.0; n_bins];
        let mask_wrong = vec![true, true, true]; // wrong length
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: Some(&mask_wrong),
        };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        let cfg = JointPoissonFitConfig::default();
        let err = joint_poisson_fit(&obj, &mut params, &cfg).unwrap_err();
        assert!(
            matches!(
                err,
                FittingError::LengthMismatch {
                    field: "active_mask",
                    ..
                }
            ),
            "expected LengthMismatch on active_mask; got {err:?}"
        );
    }

    // ==================================================================
    // Release-mode input validation at joint_poisson_fit.
    //
    // The inner `binomial_deviance_term` and `deviance_from_transmission`
    // protect themselves with `debug_assert!` only.  Release builds skip
    // those, so a length mismatch in `o` vs `s` silently truncates via
    // `.zip()` and a non-positive `c` produces finite garbage that the
    // optimizer happily minimises.  Validate at the public entry point.
    // ==================================================================

    /// `joint_poisson_fit` rejects an `o`/`s` length mismatch with a
    /// `LengthMismatch` error rather than silently truncating via `.zip()`
    /// and minimising bogus deviance on a sub-range of bins.
    #[test]
    fn test_joint_poisson_rejects_o_s_length_mismatch() {
        let n_bins = 5;
        let o: Vec<f64> = vec![10.0; n_bins];
        // Deliberate mismatch: `s` has one fewer bin than `o`.
        let s: Vec<f64> = vec![5.0; n_bins - 1];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        let err =
            joint_poisson_fit(&obj, &mut params, &JointPoissonFitConfig::default()).unwrap_err();
        assert!(
            matches!(
                err,
                FittingError::LengthMismatch {
                    field: "sample_counts",
                    ..
                }
            ),
            "expected LengthMismatch on sample_counts; got {err:?}"
        );
    }

    /// `joint_poisson_fit` rejects a non-positive proton-charge ratio `c`
    /// with `InvalidConfig` rather than falling through to the inner
    /// `debug_assert!` (which is a no-op in release builds and lets the
    /// optimizer minimise a garbage deviance landscape).
    #[test]
    fn test_joint_poisson_rejects_non_positive_c() {
        let n_bins = 5;
        let o: Vec<f64> = vec![10.0; n_bins];
        let s: Vec<f64> = vec![5.0; n_bins];
        let model = ConstModel { n_e: n_bins };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        // c = 0 is the textbook degenerate case (no sample counts).
        let obj_zero = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 0.0,
            active_mask: None,
        };
        let err = joint_poisson_fit(&obj_zero, &mut params, &JointPoissonFitConfig::default())
            .unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(_)),
            "expected InvalidConfig on c=0; got {err:?}"
        );

        // Negative c is unphysical.
        let mut params2 = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        let obj_neg = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: -1.5,
            active_mask: None,
        };
        let err = joint_poisson_fit(&obj_neg, &mut params2, &JointPoissonFitConfig::default())
            .unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(_)),
            "expected InvalidConfig on c<0; got {err:?}"
        );

        // NaN c — caught by the same finiteness check.
        let mut params3 = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        let obj_nan = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: f64::NAN,
            active_mask: None,
        };
        let err = joint_poisson_fit(&obj_nan, &mut params3, &JointPoissonFitConfig::default())
            .unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(_)),
            "expected InvalidConfig on c=NaN; got {err:?}"
        );
    }

    // ==================================================================
    // `f64::max(NaN, ε) == ε` swallows active NaN T.
    //
    // Rust stdlib's `f64::max` returns the non-NaN argument when one is
    // NaN, so `t.max(POISSON_EPSILON)` silently turns a NaN transmission
    // into ε.  The deviance term then evaluates to a finite (large)
    // number which passes the trial-step's `v.is_finite()` guard, so the
    // optimizer accepts steps into regions where the model is broken.
    //
    // `binomial_deviance_term` returns NaN when T is non-finite (so the
    // deviance sum becomes NaN and the trial guard rejects the step),
    // and `deviance_weight` / `deviance_curvature` return 0 (so the
    // gradient / Fisher accumulators are not poisoned by the bad bin).
    // ==================================================================

    /// `binomial_deviance_term` returns NaN when `t` is non-finite — so
    /// the per-bin sum poisons the deviance and the trial-step guard
    /// (`Ok(v) if v.is_finite()`) rejects the step instead of silently
    /// accepting a bogus-but-finite value.
    #[test]
    fn test_binomial_deviance_term_nan_t_returns_nan() {
        // Pre-fix: `t.max(POISSON_EPSILON)` swallows NaN and returns a
        // finite (but meaningless) deviance.
        let d_nan_t = binomial_deviance_term(50.0, 10.0, f64::NAN, 2.0);
        assert!(
            d_nan_t.is_nan(),
            "non-finite T must produce NaN deviance, not a finite shim; got {d_nan_t}"
        );

        // +inf / -inf likewise — they are not physical transmission values.
        let d_inf_t = binomial_deviance_term(50.0, 10.0, f64::INFINITY, 2.0);
        assert!(
            d_inf_t.is_nan(),
            "+inf T must produce NaN deviance; got {d_inf_t}"
        );
        let d_neg_inf_t = binomial_deviance_term(50.0, 10.0, f64::NEG_INFINITY, 2.0);
        assert!(
            d_neg_inf_t.is_nan(),
            "-inf T must produce NaN deviance; got {d_neg_inf_t}"
        );
    }

    /// `deviance_weight` returns 0 for non-finite `t` so the gradient
    /// accumulator is not poisoned — bad bins drop out instead of
    /// becoming silent NaN contributions weighted by the Jacobian.
    #[test]
    fn test_deviance_weight_nan_t_returns_zero() {
        let w = deviance_weight(50.0, 10.0, f64::NAN, 2.0);
        assert_eq!(w, 0.0, "non-finite T must give zero weight; got {w}");
    }

    /// `deviance_curvature` returns 0 for non-finite `t` so the Fisher
    /// info accumulator is not poisoned.
    #[test]
    fn test_deviance_curvature_nan_t_returns_zero() {
        let h = deviance_curvature(50.0, 10.0, f64::NAN, 2.0);
        assert_eq!(h, 0.0, "non-finite T must give zero curvature; got {h}");
    }

    /// End-to-end: a model that returns NaN at some active bin makes the
    /// deviance non-finite, the trial-step guard rejects it (rather than
    /// accepting a bogus finite step), and the fit either bails out
    /// non-converged or recovers without committing the bad step.  Prior
    /// to the M14 fix the optimizer could silently accept the NaN step.
    #[test]
    fn test_joint_poisson_fit_rejects_nan_transmission() {
        // Model that returns NaN at θ < 0.1 and a constant 0.5 otherwise.
        struct NanAtSmallTheta;
        impl FitModel for NanAtSmallTheta {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let t = if params[0] < 0.1 { f64::NAN } else { 0.5 };
                Ok(vec![t; 4])
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
                        *jac.get_mut(i, j) = if pi == 0 { 1.0 } else { 0.0 };
                    }
                }
                Some(jac)
            }
        }

        let model = NanAtSmallTheta;
        let n = 4;
        let o = vec![10.0; n];
        let s = vec![5.0; n];
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        // Initial point lands in the NaN region.
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", 0.05)]);
        let cfg = JointPoissonFitConfig::default();
        let result = joint_poisson_fit(&obj, &mut params, &cfg);
        match result {
            Ok(r) => {
                // The optimizer must NOT report a finite deviance from a
                // NaN-T initial point — pre-fix it would do so by silently
                // converting NaN to POISSON_EPSILON.  After the fix the
                // deviance is NaN (initial eval propagates), or the fit
                // never accepts a NaN step, or it ascends out of the NaN
                // region and lands at the finite plateau (params[0] >= 0.1).
                if r.params[0] < 0.1 {
                    assert!(
                        r.deviance.is_nan() && !r.gn_converged,
                        "stayed in NaN region but reported finite deviance: {r:?}"
                    );
                }
            }
            Err(_) => {
                // Acceptable: hard error from the initial evaluation.
            }
        }
    }

    /// All-fixed parameters + NaN transmission must NOT be reported as
    /// `gn_converged = true`.
    ///
    /// The `n_free == 0` shortcut in `damped_fisher_stage` previously set
    /// `converged = true` unconditionally, so a fit with every parameter
    /// fixed and a model that returns NaN at active bins would return
    /// `deviance = NaN` together with `gn_converged = true`.  Downstream
    /// pipeline code (`pipeline.rs`'s `gn_converged || polish_converged`)
    /// would then surface that pixel as a "converged" fit in the spatial
    /// map.  The guard at the top of `damped_fisher_stage` now keys
    /// convergence off `d_current.is_finite()`.
    ///
    /// Mirrors `lm.rs::test_all_fixed_params_nan_model` (issue #125.1),
    /// which exercises the equivalent guard in
    /// `levenberg_marquardt_with_mask`.
    #[test]
    fn test_joint_poisson_all_fixed_nan_transmission_does_not_converge() {
        struct NanModel {
            n_e: usize,
        }
        impl FitModel for NanModel {
            fn evaluate(&self, _params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![f64::NAN; self.n_e])
            }
        }

        let n_bins = 5;
        let o = vec![10.0; n_bins];
        let s = vec![5.0; n_bins];
        let model = NanModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        let mut params = ParameterSet::new(vec![FitParameter::fixed("T", 0.5)]);
        let cfg = JointPoissonFitConfig::default();

        let r = joint_poisson_fit(&obj, &mut params, &cfg).unwrap();

        assert!(
            r.deviance.is_nan(),
            "expected NaN deviance from all-fixed NaN model; got {}",
            r.deviance
        );
        assert!(
            r.deviance_per_dof.is_nan(),
            "expected NaN deviance_per_dof; got {}",
            r.deviance_per_dof
        );
        assert!(
            !r.gn_converged,
            "all-fixed NaN deviance must not be reported as GN-converged",
        );
        assert_eq!(r.n_free, 0);
        assert_eq!(r.n_active, n_bins);
        // The damped-Fisher loop increments `iter` before the `n_free == 0`
        // branch hits `break`, so the all-fixed path always reports exactly
        // one iteration.  Lock that in so future loop refactors don't
        // silently drift the iteration count.
        assert_eq!(
            r.gn_iterations, 1,
            "all-fixed branch should report exactly one iteration",
        );
    }

    /// Companion to [`test_joint_poisson_all_fixed_nan_transmission_does_not_converge`]
    /// covering the polish-enabled path.
    ///
    /// `nelder_mead_minimize` asserts that `x0` is non-empty (see
    /// `nelder_mead.rs`), which used to panic when stage 2 was invoked with
    /// every parameter fixed.  The polish entry-point now short-circuits on
    /// `free_indices().is_empty()`, so the call must return cleanly with
    /// `polish_converged == false` and the stage-1 NaN deviance preserved.
    /// Mirrors the pipeline configuration in `nereids-pipeline` where
    /// `with_counts_enable_polish(Some(true))` is set independently of
    /// whether the parameter set has any free entries.
    #[test]
    fn test_joint_poisson_all_fixed_nan_transmission_with_polish_does_not_panic() {
        struct NanModel {
            n_e: usize,
        }
        impl FitModel for NanModel {
            fn evaluate(&self, _params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![f64::NAN; self.n_e])
            }
        }

        let n_bins = 5;
        let o = vec![10.0; n_bins];
        let s = vec![5.0; n_bins];
        let model = NanModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        let mut params = ParameterSet::new(vec![FitParameter::fixed("T", 0.5)]);
        let cfg = JointPoissonFitConfig {
            enable_polish: true,
            ..JointPoissonFitConfig::default()
        };

        // Must not panic — the empty-x0 guard short-circuits stage 2.
        let r = joint_poisson_fit(&obj, &mut params, &cfg).unwrap();

        assert!(
            r.deviance.is_nan(),
            "expected NaN deviance from all-fixed NaN model; got {}",
            r.deviance
        );
        assert!(
            !r.gn_converged,
            "all-fixed NaN deviance must not be reported as GN-converged",
        );
        assert!(
            !r.polish_converged,
            "polish stage must report not-converged when skipped on all-fixed params",
        );
        assert!(
            !r.polish_improved,
            "polish stage cannot have improved the deviance when it was skipped",
        );
        assert_eq!(
            r.polish_iterations, 0,
            "polish stage must report zero iterations when skipped",
        );
        assert_eq!(r.n_free, 0);
        assert_eq!(r.n_active, n_bins);
        assert_eq!(
            r.gn_iterations, 1,
            "all-fixed branch should report exactly one iteration",
        );
    }

    /// Polish path with at least one **free** parameter must not report
    /// `polish_converged = true` when stage 1 ended on a non-finite
    /// deviance.
    ///
    /// Without the `best_d_stage1.is_finite()` short-circuit in the polish
    /// guard, Nelder-Mead would still run and return a finite `nm.fun`
    /// (its infeasible-point handler maps NaN evaluations to `+∞` and
    /// contracts away from them).  The commit test `nm.fun < best_d_stage1`
    /// then reduces to `finite < NaN == false`, so the polish step is
    /// discarded — but `polish_converged` would inherit `nm.self_converged`
    /// regardless, leaking a spurious converged flag together with a NaN
    /// final deviance.  Downstream pipeline code (`pipeline.rs`'s
    /// `gn_converged || polish_converged`) would then surface that fit as
    /// converged in the spatial map.
    ///
    /// Symmetric to the all-fixed NaN guard above: stage 2 refuses to run
    /// when there is no finite stage-1 deviance to refine.
    #[test]
    fn test_joint_poisson_polish_does_not_report_converged_when_stage1_nan() {
        struct NanModel {
            n_e: usize,
        }
        impl FitModel for NanModel {
            fn evaluate(&self, _params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![f64::NAN; self.n_e])
            }
        }

        let n_bins = 5;
        let o = vec![10.0; n_bins];
        let s = vec![5.0; n_bins];
        let model = NanModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        // At least one FREE parameter so polish actually runs (unlike
        // `test_joint_poisson_all_fixed_nan_transmission_with_polish_does_not_panic`,
        // which exercises the empty-free-set short-circuit instead).
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        let cfg = JointPoissonFitConfig {
            enable_polish: true,
            ..JointPoissonFitConfig::default()
        };

        let r = joint_poisson_fit(&obj, &mut params, &cfg).unwrap();

        assert!(
            r.deviance.is_nan(),
            "expected NaN deviance from NaN model; got {}",
            r.deviance
        );
        assert!(!r.gn_converged, "stage 1 cannot converge on NaN deviance",);
        assert!(
            !r.polish_converged,
            "stage 2 must not report converged when stage 1 ended non-finite",
        );
        assert!(
            !r.polish_improved,
            "polish cannot have improved a NaN starting deviance",
        );
        assert_eq!(
            r.polish_iterations, 0,
            "polish must not run when stage 1 is non-finite",
        );
        assert_eq!(r.n_free, 1);
        assert_eq!(r.n_active, n_bins);
    }

    // ==================================================================
    // NaN-in-Jacobian during FD probes (Fisher info).
    //
    // The post-convergence Fisher / covariance path builds a Jacobian
    // via FD when the model has no analytical form.  If the FD probe
    // straddles a region where the model returns NaN, the resulting
    // column is poisoned and the inverse Fisher inherits NaN entries.
    // The main LM loop's trial guard does not run here (it only checks
    // the trial step in the main optimisation loop).
    //
    // Per-cell skip: when the FD probe output is non-finite, leave the
    // entry at its zero default rather than dividing NaN by `actual_step`
    // (consistent with the "model-evaluation-failed" branch in
    // `compute_jacobian`).
    // ==================================================================

    /// `fisher_information_fd` zeroes per-cell entries whose FD probe
    /// returned a non-finite model output, rather than baking NaN into
    /// the Fisher matrix (and from there into the inverse covariance).
    #[test]
    fn test_fisher_information_fd_skips_nan_probe() {
        // Model: T_i = θ_0 (constant).  Returns NaN whenever
        // |θ_0 - 0.6| > 1e-3 — i.e. a NaN ring around the FD probe,
        // but a finite value at the base point.
        struct NanFdProbe;
        impl FitModel for NanFdProbe {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                let t = if (params[0] - 0.6).abs() > 1e-3 {
                    f64::NAN
                } else {
                    params[0]
                };
                Ok(vec![t; 3])
            }
            // No analytical_jacobian -> Fisher info must use FD fallback.
        }
        let model = NanFdProbe;
        let n = 3;
        let o = vec![10.0; n];
        let s = vec![5.0; n];
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("T", 0.6)]);
        let info = obj
            .fisher_information_fd(&mut params, 1e-2)
            .expect("fisher_information_fd should not return Err on a finite base")
            .expect("fisher_information_fd should return Some(matrix)");
        // Every entry must be finite — column was skipped on NaN probe.
        for v in info.data.iter() {
            assert!(
                v.is_finite(),
                "fisher_information_fd produced non-finite entry: {v}"
            );
        }
    }

    // ==================================================================
    // Per-element count validation propagates through `validate_inputs`.
    //
    // An earlier version ran `validate_counts` only at the
    // `joint_poisson_fit` entry point.  Direct callers of
    // `deviance_from_transmission` / `fisher_information_fd` /
    // `profile_lambda_per_bin` (diagnostics paths) bypassed that check,
    // so a NaN in `o` would propagate straight into the deviance sum
    // via `NaN <= 0.0 == false` slipping past `xlogy_ratio`'s
    // zero-branch, and a negative count would be silently swallowed as
    // zero.  The per-element check therefore lives in
    // `validate_inputs`, which every public method already calls.
    // These tests run in release mode (no `debug_assert!`) and verify
    // the typed error reaches the caller.
    // ==================================================================

    /// `deviance_from_transmission` must reject a NaN open-beam count
    /// with `InvalidConfig` rather than returning `Ok(NaN)` (or, worse,
    /// `Ok(finite)` if a future `xlogy_ratio` rewrite handled NaN by
    /// falling through to the zero branch).  The inner `debug_assert!`
    /// is a no-op in release builds, so the typed error is the only
    /// real guard.
    #[test]
    fn test_deviance_from_transmission_rejects_non_finite_counts() {
        let n_bins = 4;
        let mut o = vec![10.0; n_bins];
        o[2] = f64::NAN;
        let s = vec![5.0; n_bins];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        let t = vec![0.5; n_bins];
        let err = obj.deviance_from_transmission(&t).unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(ref msg) if msg.contains("open_beam_counts")),
            "expected InvalidConfig naming open_beam_counts; got {err:?}"
        );

        // +inf likewise.
        let mut s_inf = vec![5.0; n_bins];
        s_inf[0] = f64::INFINITY;
        let obj_inf = JointPoissonObjective {
            model: &model,
            o: &vec![10.0; n_bins],
            s: &s_inf,
            c: 1.0,
            active_mask: None,
        };
        let err = obj_inf.deviance_from_transmission(&t).unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(ref msg) if msg.contains("sample_counts")),
            "expected InvalidConfig naming sample_counts; got {err:?}"
        );
    }

    /// `deviance_from_transmission` must reject a negative count with
    /// `InvalidConfig` rather than silently treating it as a zero-count
    /// bin (which `xlogy_ratio`'s `x <= 0.0` branch would do).  Negatives
    /// indicate an upstream loader / TOF-subtraction bug; swallowing
    /// them as "no data" conceals the failure mode.
    #[test]
    fn test_deviance_from_transmission_rejects_negative_counts() {
        let n_bins = 3;
        let mut o = vec![10.0; n_bins];
        o[1] = -2.0;
        let s = vec![5.0; n_bins];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        let t = vec![0.5; n_bins];
        let err = obj.deviance_from_transmission(&t).unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(ref msg) if msg.contains("open_beam_counts")),
            "expected InvalidConfig naming open_beam_counts; got {err:?}"
        );
    }

    /// The reorientation also reaches `profile_lambda_per_bin` and
    /// `fisher_information_fd`: every public method that calls
    /// `validate_inputs` now picks up the per-element check.
    #[test]
    fn test_other_public_methods_reject_non_finite_counts() {
        let n_bins = 4;
        let mut s = vec![5.0; n_bins];
        s[3] = f64::NAN;
        let o = vec![10.0; n_bins];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        let t = vec![0.5; n_bins];

        let err = obj.profile_lambda_per_bin(&t).unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(_)),
            "profile_lambda_per_bin: expected InvalidConfig; got {err:?}"
        );

        let params = vec![0.5];
        let free_idx = vec![0];
        let err = obj
            .deviance_gradient_analytical(&params, &free_idx)
            .unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(_)),
            "deviance_gradient_analytical: expected InvalidConfig; got {err:?}"
        );

        let err = obj.fisher_information(&params, &free_idx).unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(_)),
            "fisher_information: expected InvalidConfig; got {err:?}"
        );

        let mut ps = ParameterSet::new(vec![FitParameter::non_negative("T", 0.5)]);
        let err = obj.fisher_information_fd(&mut ps, 1e-2).unwrap_err();
        assert!(
            matches!(err, FittingError::InvalidConfig(_)),
            "fisher_information_fd: expected InvalidConfig; got {err:?}"
        );
    }

    /// `validate_inputs` now reports caller-supplied transmission length
    /// mismatches with `field = "transmission"` and `expected = o.len()`.
    /// Pre-fix this used `field = "open_beam_counts"` with reversed
    /// expected/actual, which read as "the open-beam array is wrong"
    /// when the actual fault was the caller's `t` slice.
    #[test]
    fn test_validate_inputs_reports_transmission_length_mismatch_correctly() {
        let n_bins = 5;
        let o = vec![10.0; n_bins];
        let s = vec![5.0; n_bins];
        let model = ConstModel { n_e: n_bins };
        let obj = JointPoissonObjective {
            model: &model,
            o: &o,
            s: &s,
            c: 1.0,
            active_mask: None,
        };
        // Caller passes `t` shorter than `o`/`s`.
        let t_short = vec![0.5; n_bins - 2];
        let err = obj.deviance_from_transmission(&t_short).unwrap_err();
        match err {
            FittingError::LengthMismatch {
                expected,
                actual,
                field,
            } => {
                assert_eq!(field, "transmission", "field must name `transmission`");
                assert_eq!(expected, n_bins, "expected must be o.len()");
                assert_eq!(actual, n_bins - 2, "actual must be t.len()");
            }
            other => panic!("expected LengthMismatch on transmission; got {other:?}"),
        }
    }
}
