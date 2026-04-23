//! Forward-model surrogates for multi-isotope accelerated fits.
//!
//! Currently exposes [`SparseEmpiricalCubaturePlan`] — a Jacobian-anchored
//! sparse empirical cubature on the joint σ-pushforward manifold.  Round-2
//! of the algorithm-design round-robin (contestant `codex04`) validated
//! this as the k ≥ 2 winner; see
//! `.research/algo_design_roundrobin_r2/JUDGMENT.md` and the independent
//! cross-family `JUDGMENT_CODEX.md`.
//!
//! # Mathematical basis
//!
//! Let `R` be the resolution operator on a fixed target grid, `σ_1(E'),
//! …, σ_k(E')` the per-isotope cross-sections, and `x_ℓ = (σ_1(E'_ℓ), …,
//! σ_k(E'_ℓ)) ∈ ℝ^k` the pushforward of a source point `E'_ℓ`.  For each
//! row `i`, exact evaluation is
//!
//! ```text
//! T_i(n) = Σ_ℓ R_{iℓ} exp(-n · x_ℓ)
//! ∂T_i/∂n_j = -Σ_ℓ R_{iℓ} x_{ℓ,j} exp(-n · x_ℓ)
//! ```
//!
//! The row support contains ~82 ℓ's on the VENUS 3471-bin production
//! grid.  By [Carathéodory / Tchakaloff], any nonneg combination of
//! feature vectors over this support is matched (in feature space) by an
//! equivalent nonneg combination supported on at most `d + 1` atoms,
//! where `d` is the feature dimension.  Choosing features = forward
//! evaluations at `S` training densities + Jacobian evaluations at one
//! anchor density gives `d = S + k` features, so each row collapses to
//! ≤ `S + k + 1` atoms while preserving positivity, row-stochasticity,
//! and the exact Jacobian at the anchor.
//!
//! # Empirical compression (real VENUS operator, codex04 measurements)
//!
//! | Scenario                          | k | avg atoms/row | max atoms/row | compression vs exact |
//! |-----------------------------------|---|---------------|---------------|----------------------|
//! | Hf (natural group)                | 1 | 3.53          | 67            | 23.3×                |
//! | Hf + W                            | 2 | 5.65          | 7             | 14.5×                |
//! | U-235 + U-238                     | 2 | 5.32          | 7             | 15.5×                |
//! | Gd + Eu + Sm                      | 3 | 8.59          | 9             | 9.6×                 |
//! | Hf-174/176/177/178/179/180 indep. | 6 | 9.03          | 15            | 9.1×                 |
//!
//! # LP solver
//!
//! Row-wise Tchakaloff reduction is framed as a feasibility LP (minimize
//! `0` subject to the equality constraints) and solved with `microlp`.
//! The problem is small (≤ S + k + 1 rows × |support| columns, here
//! typically ~ 10 × ~ 100) so a pure-Rust simplex is fast enough.

use std::fmt;

use microlp::{ComparisonOp, OptimizationDirection, Problem};

use crate::resolution::ResolutionMatrix;

/// Errors from [`SparseEmpiricalCubaturePlan`] construction.
#[derive(Debug)]
pub enum CubatureBuildError {
    /// Flat `sigmas` storage has the wrong total element count.
    ///
    /// `sigmas` is stored row-major as `sigmas[j * n_rows + ℓ] =
    /// σ_j(E'_ℓ)`, so the expected total length is `k * n_rows`.
    SigmaGridMismatch {
        /// Expected total element count (`k * n_rows`).
        expected: usize,
        /// Actual `sigmas.len()`.
        actual: usize,
    },
    /// Zero isotopes supplied — the cubature has no meaning for k = 0.
    ZeroIsotopes,
    /// Zero training densities supplied — the LP construction requires
    /// at least one forward feature per row.
    ZeroTrainingDensities,
    /// A training density vector has a length different from the
    /// isotope count.
    TrainingDensityLength {
        /// Expected (k).
        expected: usize,
        /// Actual (`training_densities[i].len()`).
        actual: usize,
        /// Offending index.
        index: usize,
    },
    /// The Jacobian anchor density has a length different from the
    /// isotope count.
    AnchorLength {
        /// Expected (k).
        expected: usize,
        /// Actual.
        actual: usize,
    },
    /// The row-wise LP failed to produce a feasible solution.  Should
    /// never fire on a well-formed problem because the uniform
    /// (non-sparse) weight is always feasible; if it does, it signals
    /// a numerical degeneracy (e.g., identical atoms in the row
    /// support) worth investigating.
    LpInfeasible {
        /// Row of the resolution matrix where the LP failed.
        row: usize,
    },
}

impl fmt::Display for CubatureBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SigmaGridMismatch { expected, actual } => write!(
                f,
                "sigmas flat length ({actual}) must equal k * n_rows ({expected})",
            ),
            Self::ZeroIsotopes => write!(f, "cubature requires at least one isotope"),
            Self::ZeroTrainingDensities => {
                write!(f, "cubature requires at least one training density sample",)
            }
            Self::TrainingDensityLength {
                expected,
                actual,
                index,
            } => write!(
                f,
                "training_densities[{index}] has length {actual} (expected k = {expected})",
            ),
            Self::AnchorLength { expected, actual } => write!(
                f,
                "jacobian_anchor has length {actual} (expected k = {expected})",
            ),
            Self::LpInfeasible { row } => write!(
                f,
                "row-wise LP failed to find a feasible cubature for row {row} — \
                 likely numerical degeneracy in the row support",
            ),
        }
    }
}

impl std::error::Error for CubatureBuildError {}

/// Row-wise Tchakaloff cubature of the joint σ-pushforward measure on a
/// fixed target grid.
///
/// Laid out in flat Struct-of-Arrays (SoA) form for cache-friendly online
/// evaluation:
///
/// * `row_starts[i]..row_starts[i+1]` indexes into `weights`/`atoms` for
///   row `i`.
/// * `weights[q]` is the per-atom nonneg weight (sums to 1.0 within each
///   row since the source measure is row-stochastic).
/// * `atoms[q]` is a flat row-major block of length `k` storing the
///   atom's joint σ coordinates.
///
/// Built once per `(grid, isotope_set, training_densities, anchor)`
/// tuple and applied repeatedly during LM / KL iterations via
/// [`Self::forward`] and [`Self::forward_and_jacobian`].
#[derive(Debug, Clone)]
pub struct SparseEmpiricalCubaturePlan {
    /// Target energy grid the plan was built for (owned copy, same
    /// pattern as [`crate::resolution::ResolutionPlan`] /
    /// [`crate::resolution::ResolutionMatrix`]).  Callers implementing
    /// plan caches compare this against their current grid to decide
    /// whether the plan is still valid.
    target_energies: Vec<f64>,
    /// Number of isotopes (per-atom dimensionality).
    k: usize,
    /// `row_starts[i]..row_starts[i+1]` — CSR-style row offsets.
    /// Length `target_energies.len() + 1`.
    row_starts: Vec<u32>,
    /// Per-atom nonneg weights.  Within each row, `Σ_q weights[q] = 1`.
    weights: Vec<f64>,
    /// Row-major flat storage of atom coordinates in ℝ^k.  Length
    /// `k * weights.len()`.  Atom `q` occupies indices `k*q .. k*(q+1)`.
    atoms: Vec<f64>,
    /// Optional training-density upper bound — the per-isotope
    /// `train_max` used to build the plan.  When set, dispatch
    /// layers can compare the current fit iterate against it and
    /// fall back to the exact path when the iterate strays beyond
    /// the box (with a tolerance multiplier to avoid thrashing).
    /// `None` means "no box information available; dispatch cannot
    /// safety-check against it".  Set via
    /// [`Self::with_density_box`].  Codex round-4 P1 on PR #480.
    density_box: Option<Vec<f64>>,
}

impl SparseEmpiricalCubaturePlan {
    /// Canonical default training-density rule from the codex04
    /// round-2 reference: for an upper-bound density vector
    /// `train_max ∈ ℝ^k`, return `S = 2 + k` training points
    /// consisting of `0.25 * train_max`, `0.75 * train_max`, and the
    /// k axis-aligned "unit" points `train_max[i] · e_i` (all other
    /// components zero).  Exposed as a helper so callers — including
    /// the wiring in PR #474b — don't have to hand-roll the rule.
    ///
    /// Duplicates are NOT removed.  In practice the rule produces
    /// `S = k + 2` distinct points for any `k ≥ 1` with all
    /// `train_max[i] > 0`.
    pub fn default_training_points(train_max: &[f64]) -> Vec<Vec<f64>> {
        let k = train_max.len();
        let mut points: Vec<Vec<f64>> = Vec::with_capacity(k + 2);
        points.push(train_max.iter().map(|&x| 0.25 * x).collect());
        points.push(train_max.iter().map(|&x| 0.75 * x).collect());
        for (i, &max_i) in train_max.iter().enumerate() {
            let mut p = vec![0.0_f64; k];
            p[i] = max_i;
            points.push(p);
        }
        points
    }

    /// Canonical default Jacobian anchor from the codex04 round-2
    /// reference: `0.5 * train_max`, the midpoint of the density
    /// box.
    pub fn default_jacobian_anchor(train_max: &[f64]) -> Vec<f64> {
        train_max.iter().map(|&x| 0.5 * x).collect()
    }

    /// Build a Tchakaloff sparse-cubature plan row-by-row from an exact
    /// [`ResolutionMatrix`] + isotope cross-section stack.
    ///
    /// # Arguments
    ///
    /// * `matrix` — exact sparse R (built via
    ///   [`crate::resolution::ResolutionPlan::compile_to_matrix`]).
    /// * `sigmas` — per-isotope cross-sections on the matrix's target
    ///   grid, flat row-major: `sigmas[j * n_rows + ℓ]` = σ_j(E'_ℓ).
    /// * `k` — number of isotopes (must match `sigmas.len() / n_rows`).
    /// * `training_densities` — a slice of density vectors `n^(s) ∈
    ///   ℝ^k` covering the density box the fit is expected to explore.
    ///   Codex04's default rule is `[0.25 * train_max, 0.75 *
    ///   train_max] ∪ {train_max_e_i : i=1..k}` which gives `S = 2 + k`
    ///   distinct training points.
    /// * `jacobian_anchor` — a single density `n* ∈ ℝ^k` at which the
    ///   Jacobian features are evaluated.  Codex04 uses `0.5 * train_max`.
    ///
    /// Per-row LP:
    ///
    /// ```text
    /// find   x ≥ 0 in ℝ^{|support|}
    /// s.t.   Σ_q x_q = 1
    ///        phi[s, q]  = exp(-n^(s) · σ_support[q])      for s = 1..S
    ///        phi[ℓ, q]  = σ_{ℓ, support[q]} · exp(-n* · σ_support[q])
    ///                                                     for ℓ = 1..k
    ///        phi @ x    = phi @ w_exact_support
    /// ```
    ///
    /// where `w_exact_support = R[i, support] / Σ_q R[i, support[q]]`
    /// is the **exact full-support row measure** (the existing
    /// non-sparse weight distribution — NOT uniform; the entries
    /// carry the kernel shape).  It serves as the feasibility
    /// fallback for the LP: the identity `x = w_exact_support`
    /// always satisfies the equality constraints, so a feasible
    /// solution exists.  The returned basic feasible solution has
    /// at most `S + k + 1` nonzero entries (Carathéodory).
    pub fn build(
        matrix: &ResolutionMatrix,
        sigmas: &[f64],
        k: usize,
        training_densities: &[Vec<f64>],
        jacobian_anchor: &[f64],
    ) -> Result<Self, CubatureBuildError> {
        if k == 0 {
            return Err(CubatureBuildError::ZeroIsotopes);
        }
        if training_densities.is_empty() {
            return Err(CubatureBuildError::ZeroTrainingDensities);
        }
        let n_rows = matrix.len();
        if sigmas.len() != k * n_rows {
            return Err(CubatureBuildError::SigmaGridMismatch {
                expected: k * n_rows,
                actual: sigmas.len(),
            });
        }
        for (idx, td) in training_densities.iter().enumerate() {
            if td.len() != k {
                return Err(CubatureBuildError::TrainingDensityLength {
                    expected: k,
                    actual: td.len(),
                    index: idx,
                });
            }
        }
        if jacobian_anchor.len() != k {
            return Err(CubatureBuildError::AnchorLength {
                expected: k,
                actual: jacobian_anchor.len(),
            });
        }

        // Empty matrix — return an empty plan.
        if n_rows == 0 {
            return Ok(Self {
                target_energies: matrix.target_energies().to_vec(),
                k,
                row_starts: vec![0],
                weights: Vec::new(),
                atoms: Vec::new(),
                density_box: None,
            });
        }

        let n_train = training_densities.len();
        // Per-row LP has `n_train + k` equality rows for `phi @ x =
        // target` plus 1 for `sum x = 1`.
        let phi_rows = n_train + k;

        let mut row_starts: Vec<u32> = Vec::with_capacity(n_rows + 1);
        row_starts.push(0);
        let mut weights: Vec<f64> = Vec::new();
        let mut atoms: Vec<f64> = Vec::new();

        // Reusable scratch across rows.  Per-row support widths differ,
        // but the max is bounded by `max(row_nnz) ≤ 132` on the real
        // VENUS operator; `clear()` reuses the `Vec` capacity.
        let mut support_sigma: Vec<f64> = Vec::new(); // k * |support|, row-major over atoms
        let mut w_exact: Vec<f64> = Vec::new(); // |support|
        let mut phi_fwd: Vec<f64> = Vec::new(); // n_train × |support|, row-major over rows
        let mut phi_grad: Vec<f64> = Vec::new(); // k × |support|
        let mut grad_base: Vec<f64> = Vec::new(); // |support| — exp(-anchor · σ_q) hoisted out of ell loop
        let mut target: Vec<f64> = Vec::new(); // phi_rows
        let mut phi_col_buf: Vec<(microlp::Variable, f64)> = Vec::new();

        for i in 0..n_rows {
            let start = matrix.row_starts()[i] as usize;
            let end = matrix.row_starts()[i + 1] as usize;
            let support_cols = &matrix.col_indices()[start..end];
            let support_vals = &matrix.values()[start..end];
            let support_len = support_cols.len();

            // Passthrough / empty row → emit uniform weight directly.
            // No LP needed.  (A single row with a single entry at col
            // i, value 1.0, stays as a single atom — its pushforward
            // coordinates are just σ at that column.)
            if support_len == 0 {
                row_starts.push(weights.len() as u32);
                continue;
            }

            // Shortcut: if the row support has only 1 column, the
            // cubature is that single atom with weight 1.  No LP and
            // no feature matrix needed.  Must check BEFORE building
            // w_exact / phi to avoid the work the shortcut then
            // discards.
            if support_len == 1 {
                let col = support_cols[0] as usize;
                weights.push(1.0);
                atoms.extend((0..k).map(|j| sigmas[j * n_rows + col]));
                row_starts.push(weights.len() as u32);
                continue;
            }

            // Non-trivial row (support_len ≥ 2).  Build normalized
            // exact-weight distribution + collect support-column σ
            // vectors.
            //
            // **Zero-weight CSR cells MUST be filtered out** before
            // they reach the LP.  [`ResolutionPlan::compile_to_matrix`]
            // deliberately retains `value == 0.0` entries for the
            // `frac == +0.0` branch to preserve downstream NaN-safety
            // when the matrix is re-applied to a spectrum containing
            // NaN at `lo + 1`.  But the cubature LP has a zero
            // objective, so the simplex is free to assign positive
            // mass to any zero-weight variable — the training
            // constraints pass trivially (w_exact = 0 → target
            // contribution = 0), yet held-out forward/Jacobian
            // predictions can pick up mass at energies the exact
            // resolution operator never samples.  Filter them here
            // so no zero-R column ever becomes an LP variable or a
            // stored atom.  Codex round-3 finding on PR #474a.
            //
            // Row sum guard: the source matrix is row-stochastic
            // (Σ_q R_{iq} = 1 to machine precision), so dropping
            // exactly-zero columns preserves `row_sum > 0`.
            let row_sum: f64 = support_vals.iter().sum();
            support_sigma.clear();
            support_sigma.reserve(k * support_len);
            w_exact.clear();
            w_exact.reserve(support_len);
            for (q, &col_u32) in support_cols.iter().enumerate() {
                if support_vals[q] == 0.0 {
                    continue;
                }
                let col = col_u32 as usize;
                for j in 0..k {
                    support_sigma.push(sigmas[j * n_rows + col]);
                }
                w_exact.push(support_vals[q] / row_sum);
            }
            // Effective support length after dropping zero-weight
            // CSR cells.  Subsequent LP / feature-matrix code uses
            // this, not the original `support_len` that included
            // zero-weight cells.
            let support_len = w_exact.len();

            // Re-check the degenerate cases on the filtered support.
            // If all CSR cells happened to be zero, treat like an
            // empty row.  If exactly one survives, take the shortcut.
            if support_len == 0 {
                row_starts.push(weights.len() as u32);
                continue;
            }
            if support_len == 1 {
                weights.push(1.0);
                atoms.extend_from_slice(&support_sigma[..k]);
                row_starts.push(weights.len() as u32);
                continue;
            }

            // Build per-row feature matrix phi (row-major over feature
            // rows, then support columns).
            phi_fwd.clear();
            phi_fwd.reserve(n_train * support_len);
            for td in training_densities.iter() {
                for q in 0..support_len {
                    let mut dot = 0.0_f64;
                    for j in 0..k {
                        dot += td[j] * support_sigma[q * k + j];
                    }
                    phi_fwd.push((-dot).exp());
                }
            }
            // Jacobian features `phi_grad[ℓ, q] = σ_{ℓ,q} · exp(-n* ·
            // σ_q)`.  The `exp(-n* · σ_q)` factor depends only on `q`,
            // not `ℓ`, so hoist it into a row-local `grad_base[q]`
            // buffer to avoid recomputing |support| × k exponentials
            // (matches the codex04 Python reference's `phi_grad_base`
            // layout).
            phi_grad.clear();
            phi_grad.reserve(k * support_len);
            grad_base.clear();
            grad_base.reserve(support_len);
            for q in 0..support_len {
                let mut dot = 0.0_f64;
                for j in 0..k {
                    dot += jacobian_anchor[j] * support_sigma[q * k + j];
                }
                grad_base.push((-dot).exp());
            }
            for ell in 0..k {
                for q in 0..support_len {
                    phi_grad.push(support_sigma[q * k + ell] * grad_base[q]);
                }
            }

            // Target = phi @ w_exact, built streaming per feature row.
            target.clear();
            target.reserve(phi_rows);
            for s in 0..n_train {
                let mut t = 0.0_f64;
                for q in 0..support_len {
                    t += phi_fwd[s * support_len + q] * w_exact[q];
                }
                target.push(t);
            }
            for ell in 0..k {
                let mut t = 0.0_f64;
                for q in 0..support_len {
                    t += phi_grad[ell * support_len + q] * w_exact[q];
                }
                target.push(t);
            }

            // Feasibility LP: minimize 0 subject to the equality
            // constraints.  Each column = one atom; coefficient on the
            // objective = 0.  `x_q ∈ [0, ∞)`.
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let vars: Vec<microlp::Variable> = (0..support_len)
                .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
                .collect();

            // sum x_q = 1
            phi_col_buf.clear();
            for &v in &vars {
                phi_col_buf.push((v, 1.0));
            }
            problem.add_constraint(&phi_col_buf, ComparisonOp::Eq, 1.0);

            // phi @ x = target, one equality per feature row.
            for s in 0..n_train {
                phi_col_buf.clear();
                for q in 0..support_len {
                    phi_col_buf.push((vars[q], phi_fwd[s * support_len + q]));
                }
                problem.add_constraint(&phi_col_buf, ComparisonOp::Eq, target[s]);
            }
            for ell in 0..k {
                phi_col_buf.clear();
                for q in 0..support_len {
                    phi_col_buf.push((vars[q], phi_grad[ell * support_len + q]));
                }
                problem.add_constraint(&phi_col_buf, ComparisonOp::Eq, target[n_train + ell]);
            }

            // Solve.  If `microlp` fails (it may on numerically
            // degenerate row supports — e.g., identical σ across the
            // row, which is physically rare but possible), fall back
            // to the exact full-support row measure `w_exact`.  This
            // preserves correctness at the cost of giving up
            // compression on that row.  The `LpInfeasible` error
            // variant (returned below) only fires if BOTH the LP
            // solution AND the `w_exact` fallback produce an empty
            // active set after the `WEIGHT_EPSILON` filter — which
            // is physically impossible on a valid row-stochastic
            // `ResolutionMatrix` row (`Σ w_exact = 1` implies at
            // least one entry exceeds `1 / support_len > 1e-12`).
            let sparse_weights: Vec<f64> = match problem.solve() {
                Ok(solution) => vars.iter().map(|&v| solution.var_value(v)).collect(),
                Err(_) => w_exact.clone(),
            };

            // Drop numerically-zero atoms and renormalize so the row
            // still sums to exactly 1.0 after simplex roundoff.
            const WEIGHT_EPSILON: f64 = 1e-12;
            let mut active: Vec<(usize, f64)> = sparse_weights
                .iter()
                .enumerate()
                .filter_map(|(q, &w)| (w > WEIGHT_EPSILON).then_some((q, w)))
                .collect();
            if active.is_empty() {
                // Extreme fallback — should never happen because
                // w_exact is already feasible with support_len > 0,
                // but defend against a corrupt LP result.
                active = w_exact
                    .iter()
                    .enumerate()
                    .filter_map(|(q, &w)| (w > WEIGHT_EPSILON).then_some((q, w)))
                    .collect();
                if active.is_empty() {
                    return Err(CubatureBuildError::LpInfeasible { row: i });
                }
            }
            let active_sum: f64 = active.iter().map(|&(_, w)| w).sum();
            // Note: rows with repeated σ patterns (physically
            // uncommon but possible) end up with multiple atoms at
            // identical x.  We emit them separately and rely on
            // online forward evaluation to sum the weighted
            // exponentials, which is algebraically identical to a
            // pre-merged atom.  Merging would be a micro-optimization
            // worth revisiting only if profiling shows the duplicate
            // work matters.

            for (q, w) in active {
                weights.push(w / active_sum);
                for j in 0..k {
                    atoms.push(support_sigma[q * k + j]);
                }
            }
            row_starts.push(weights.len() as u32);
        }

        Ok(Self {
            target_energies: matrix.target_energies().to_vec(),
            k,
            row_starts,
            weights,
            atoms,
            density_box: None,
        })
    }

    /// Number of rows (target-grid size) covered by this plan.
    pub fn len(&self) -> usize {
        self.target_energies.len()
    }

    /// True when the plan covers no target energies.
    pub fn is_empty(&self) -> bool {
        self.target_energies.is_empty()
    }

    /// Number of isotopes (per-atom dimensionality).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Total number of stored atoms across all rows.
    pub fn n_atoms(&self) -> usize {
        self.weights.len()
    }

    /// Target energy grid the plan was built for.
    ///
    /// Mirrors [`crate::resolution::ResolutionPlan::target_energies`]
    /// / [`crate::resolution::ResolutionMatrix::target_energies`] —
    /// callers implementing plan caches compare this against their
    /// current grid to decide whether the plan is still valid.
    pub fn target_energies(&self) -> &[f64] {
        &self.target_energies
    }

    /// CSR row-start offsets.  `row_starts()[i]..row_starts()[i+1]`
    /// names the atom range for row `i`.  Length `len() + 1`.
    pub fn row_starts(&self) -> &[u32] {
        &self.row_starts
    }

    /// Per-atom weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Per-atom σ coordinates, flat row-major.  Atom `q` at
    /// `atoms()[k * q .. k * (q + 1)]`.
    pub fn atoms(&self) -> &[f64] {
        &self.atoms
    }

    /// Training-density upper bound recorded at build time, if any.
    /// Dispatch layers use this to detect when the fit iterate
    /// escapes the training region and safely fall back to the
    /// exact path (cubature accuracy degrades quickly outside the
    /// trained box).  `None` when the caller chose not to record
    /// one — in that case dispatch cannot safety-check.
    pub fn density_box(&self) -> Option<&[f64]> {
        self.density_box.as_deref()
    }

    /// Attach the training-density upper bound (`train_max`) used
    /// during build, so dispatch can refuse to fire on iterates
    /// that escape the trained region.  Builder-style; returns
    /// `self` for chaining.  Callers in `spatial_map_typed`
    /// populate this with the same `train_max` vector fed into
    /// [`Self::default_training_points`] / [`Self::default_jacobian_anchor`].
    ///
    /// # Panics
    ///
    /// Panics if `train_max.len() != self.k()`.
    #[must_use]
    pub fn with_density_box(mut self, train_max: Vec<f64>) -> Self {
        assert_eq!(
            train_max.len(),
            self.k,
            "train_max length ({}) must equal k ({})",
            train_max.len(),
            self.k,
        );
        self.density_box = Some(train_max);
        self
    }

    /// Evaluate the surrogate forward model `T_i(n)` at density vector
    /// `n ∈ ℝ^k`.
    ///
    /// # Panics
    ///
    /// Panics if `n.len() != self.k()`.
    pub fn forward(&self, n: &[f64]) -> Vec<f64> {
        assert_eq!(
            n.len(),
            self.k,
            "density vector length ({}) must match plan isotope count ({})",
            n.len(),
            self.k,
        );
        let mut out = vec![0.0_f64; self.target_energies.len()];
        for (i, out_i) in out.iter_mut().enumerate() {
            let s = self.row_starts[i] as usize;
            let e = self.row_starts[i + 1] as usize;
            let mut acc = 0.0_f64;
            for q in s..e {
                let atom = &self.atoms[q * self.k..(q + 1) * self.k];
                let mut dot = 0.0_f64;
                for j in 0..self.k {
                    dot += n[j] * atom[j];
                }
                acc += self.weights[q] * (-dot).exp();
            }
            *out_i = acc;
        }
        out
    }

    /// Evaluate forward + per-density Jacobian at density vector `n`.
    /// Returns `(T, J)` where `T[i] = T_i(n)` and `J[i * k + ℓ] =
    /// ∂T_i/∂n_ℓ`, both computed from the same atom scan so the online
    /// cost is `(k + 1)` FLOPs per atom rather than `k + 1` separate
    /// passes.
    ///
    /// # Panics
    ///
    /// Panics if `n.len() != self.k()`.
    pub fn forward_and_jacobian(&self, n: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(
            n.len(),
            self.k,
            "density vector length ({}) must match plan isotope count ({})",
            n.len(),
            self.k,
        );
        let mut forward = vec![0.0_f64; self.target_energies.len()];
        let mut jac = vec![0.0_f64; self.target_energies.len() * self.k];
        for i in 0..self.target_energies.len() {
            let s = self.row_starts[i] as usize;
            let e = self.row_starts[i + 1] as usize;
            let mut t_i = 0.0_f64;
            let jac_row = &mut jac[i * self.k..(i + 1) * self.k];
            for q in s..e {
                let atom = &self.atoms[q * self.k..(q + 1) * self.k];
                let mut dot = 0.0_f64;
                for j in 0..self.k {
                    dot += n[j] * atom[j];
                }
                let term = self.weights[q] * (-dot).exp();
                t_i += term;
                for (ell, jac_slot) in jac_row.iter_mut().enumerate() {
                    *jac_slot -= term * atom[ell];
                }
            }
            forward[i] = t_i;
        }
        (forward, jac)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scalar (k = 1) surrogate — epic #472, PR #475.
// ═══════════════════════════════════════════════════════════════════
//
// The [`SparseEmpiricalCubaturePlan`] above is the k ≥ 2 production
// winner, but its generic atom construction over-damps the grouped
// Hf k = 1 KL scatter by ~27 % (codex04 round-2 measurement).  The
// scalar path gets a dedicated surrogate.  PR #475 built both
// round-2 candidates (Lanczos σ-pushforward Gauss quadrature,
// Chebyshev-in-density) side-by-side and benched them on the real
// VENUS 3471-bin production grid:
//
//     Lanczos Gauss (m = 6):  fwd = 34 µs,  max_err = 4e-15, 7.1× vs exact.
//     Chebyshev    (M = 16):  fwd = 20 µs,  max_err = 2e-15, 12.2× vs exact.
//
// **Chebyshev won**.  Lanczos + Gauss-pushforward machinery was
// deleted per the issue's "drop the loser" contract — no
// same-name-different-function duplication.  If future research
// finds a better scalar surrogate, the public
// [`ScalarSurrogatePlan`] alias below is the stable swap point.

/// Errors from scalar surrogate plan construction.
#[derive(Debug)]
pub enum ScalarSurrogateBuildError {
    /// `sigma` flat length disagrees with the matrix grid size.
    SigmaGridMismatch {
        /// Expected length (`n_rows`).
        expected: usize,
        /// Actual `sigma.len()`.
        actual: usize,
    },
    /// A Chebyshev-node build was given `n_max ≤ 0` or `M < 2`.
    InvalidChebyshevBox {
        /// Offending upper bound.
        n_max: f64,
        /// Requested node count.
        m: usize,
    },
}

impl fmt::Display for ScalarSurrogateBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SigmaGridMismatch { expected, actual } => write!(
                f,
                "scalar sigma length ({actual}) must equal n_rows ({expected})",
            ),
            Self::InvalidChebyshevBox { n_max, m } => write!(
                f,
                "Chebyshev plan requires n_max > 0 and M ≥ 2, got n_max = {n_max}, M = {m}",
            ),
        }
    }
}

impl std::error::Error for ScalarSurrogateBuildError {}

/// Chebyshev-in-density interpolant of `T_i(n)` for scalar (k = 1)
/// forward models.  For each row `i`, pre-samples `T_i(n_j)` at
/// `M` Chebyshev-of-the-first-kind nodes in `[0, n_max]`, then
/// stores the Chebyshev coefficients.  Online evaluation is
/// Clenshaw recurrence with `M` multiply-adds per row.
///
/// Unlike the Gauss quadrature, the Chebyshev representation is a
/// **scalar interpolant** in density space — one pass evaluates
/// the interpolant at `n`, and the derivative needs a separate
/// derivative-coefficient series.
#[derive(Debug, Clone)]
pub struct ScalarChebyshevPlan {
    /// Target energy grid the plan was built for.
    target_energies: Vec<f64>,
    /// Upper bound of the density box `[0, n_max]` the interpolant
    /// is valid on.
    n_max: f64,
    /// Number of Chebyshev nodes (order + 1).  Same for every row.
    m: usize,
    /// Row-major Chebyshev coefficients: `coeffs[i * m + k]` is the
    /// `k`-th Chebyshev coefficient of row `i`.
    coeffs: Vec<f64>,
    /// Optional training-density upper bound (defaults to `n_max`
    /// if the builder doesn't override).
    density_box: Option<f64>,
}

impl ScalarChebyshevPlan {
    /// Build an `M`-node Chebyshev-in-density plan from a
    /// [`ResolutionMatrix`] + scalar σ + density box `[0, n_max]`.
    /// Internally calls [`apply_r`] `M` times (one per Chebyshev
    /// node) to get exact row evaluations, then runs a per-row
    /// discrete cosine transform to extract Chebyshev coefficients.
    ///
    /// Cost: `M × N × avg_nnz_per_row` FMAs for the exact sampling
    /// pass, plus `M^2` per row for the DCT (matrix-vector form,
    /// simpler than FFT for small M).
    pub fn build(
        matrix: &ResolutionMatrix,
        sigma: &[f64],
        n_max: f64,
        m: usize,
    ) -> Result<Self, ScalarSurrogateBuildError> {
        let n_rows = matrix.len();
        if sigma.len() != n_rows {
            return Err(ScalarSurrogateBuildError::SigmaGridMismatch {
                expected: n_rows,
                actual: sigma.len(),
            });
        }
        if !n_max.is_finite() || n_max <= 0.0 || m < 2 {
            return Err(ScalarSurrogateBuildError::InvalidChebyshevBox { n_max, m });
        }

        // Chebyshev nodes of the first kind on [-1, 1]:
        //   x_j = cos(π (j + 0.5) / M)   for j = 0..M-1
        // Mapped to [0, n_max]:
        //   n_j = (n_max / 2) (x_j + 1)
        let nodes_x: Vec<f64> = (0..m)
            .map(|j| {
                let pj = (j as f64 + 0.5) * std::f64::consts::PI / m as f64;
                pj.cos()
            })
            .collect();
        let nodes_n: Vec<f64> = nodes_x.iter().map(|&x| 0.5 * n_max * (x + 1.0)).collect();

        // Evaluate T_i(n_j) exactly for each j.  `values[j * n_rows
        // + i]` = T_i(n_j).
        let mut samples = vec![0.0_f64; m * n_rows];
        for (j, &nj) in nodes_n.iter().enumerate() {
            let t_un: Vec<f64> = (0..n_rows).map(|i| (-nj * sigma[i]).exp()).collect();
            let t_res = crate::resolution::apply_r(matrix, &t_un);
            for (i, &v) in t_res.iter().enumerate() {
                samples[j * n_rows + i] = v;
            }
        }

        // DCT-II to extract Chebyshev coefficients per row.
        // c_k = (2 / M) Σ_j T_i(n_j) T_k(x_j)   for k ≥ 1
        // c_0 = (1 / M) Σ_j T_i(n_j)
        // where T_k(cos θ) = cos(k θ), θ_j = π (j + 0.5) / M.
        let mut coeffs = vec![0.0_f64; n_rows * m];
        for i in 0..n_rows {
            for k in 0..m {
                let mut sum = 0.0_f64;
                for j in 0..m {
                    let theta_j = (j as f64 + 0.5) * std::f64::consts::PI / m as f64;
                    sum += samples[j * n_rows + i] * (k as f64 * theta_j).cos();
                }
                let scale = if k == 0 { 1.0 } else { 2.0 } / m as f64;
                coeffs[i * m + k] = scale * sum;
            }
        }

        Ok(Self {
            target_energies: matrix.target_energies().to_vec(),
            n_max,
            m,
            coeffs,
            density_box: Some(n_max),
        })
    }

    pub fn len(&self) -> usize {
        self.target_energies.len()
    }
    pub fn is_empty(&self) -> bool {
        self.target_energies.is_empty()
    }
    pub fn target_energies(&self) -> &[f64] {
        &self.target_energies
    }
    pub fn n_max(&self) -> f64 {
        self.n_max
    }
    pub fn m(&self) -> usize {
        self.m
    }
    pub fn density_box(&self) -> Option<f64> {
        self.density_box
    }

    /// Evaluate the Chebyshev interpolant at density `n`.  Density
    /// outside `[0, n_max]` extrapolates (caller responsibility —
    /// dispatch should reject via the density-box check).
    pub fn forward_scalar(&self, n: f64) -> Vec<f64> {
        let n_rows = self.target_energies.len();
        let mut out = vec![0.0_f64; n_rows];
        if self.m == 0 {
            return out;
        }
        // Map n → x ∈ [-1, 1].
        let x = 2.0 * n / self.n_max - 1.0;
        // Clenshaw recurrence: evaluate Σ_k c_k T_k(x).
        // b_{M+1} = b_{M+2} = 0; b_k = 2 x b_{k+1} - b_{k+2} + c_k;
        // result = c_0 + x b_1 - b_2.
        for (i, out_i) in out.iter_mut().enumerate() {
            let row_start = i * self.m;
            let mut b_next = 0.0_f64;
            let mut b_next_next = 0.0_f64;
            for k in (1..self.m).rev() {
                let b_k = 2.0 * x * b_next - b_next_next + self.coeffs[row_start + k];
                b_next_next = b_next;
                b_next = b_k;
            }
            *out_i = self.coeffs[row_start] + x * b_next - b_next_next;
        }
        out
    }

    /// Evaluate forward + derivative in one pass.  The derivative
    /// of a Chebyshev series can be evaluated via a modified
    /// Clenshaw recurrence that internally tracks the derivative
    /// coefficients — or we use the standard identity
    /// `T_k'(x) = k · U_{k-1}(x)` (Chebyshev-of-the-second-kind
    /// recurrence).  Here we run two parallel Clenshaw sweeps: one
    /// for `T(x)` and one for `d/dx T(x)`, then scale by
    /// `dx/dn = 2 / n_max`.
    pub fn forward_and_derivative_scalar(&self, n: f64) -> (Vec<f64>, Vec<f64>) {
        let n_rows = self.target_energies.len();
        let mut forward = vec![0.0_f64; n_rows];
        let mut deriv = vec![0.0_f64; n_rows];
        if self.m == 0 {
            return (forward, deriv);
        }
        let x = 2.0 * n / self.n_max - 1.0;
        let dx_dn = 2.0 / self.n_max;

        // Derivative coefficients d_k such that Σ d_k T_k(x) =
        // d/dx Σ c_k T_k(x).  Standard recurrence:
        //   d_{M-1} = 0
        //   d_{M-2} = 2 (M-1) c_{M-1}
        //   d_k = d_{k+2} + 2 (k+1) c_{k+1}   for k = M-3..0
        // (then d_0 needs to be halved if we want a simple Clenshaw,
        // but it's cleaner to use the "recurrence with halved d_0"
        // convention; we apply the same Clenshaw as forward.)
        let m = self.m;
        let mut d_coeffs = vec![0.0_f64; m];

        for (i, (out_t, out_d)) in forward.iter_mut().zip(deriv.iter_mut()).enumerate() {
            let row_start = i * m;
            // Compute d_coeffs for this row.
            d_coeffs.fill(0.0);
            if m >= 2 {
                // d_{M-1} = 0 (already zero)
                // d_{M-2} = 2 (M-1) c_{M-1}  for M ≥ 2
                for k in (0..m - 1).rev() {
                    let prev = if k + 2 < m { d_coeffs[k + 2] } else { 0.0 };
                    d_coeffs[k] = prev + 2.0 * (k as f64 + 1.0) * self.coeffs[row_start + k + 1];
                }
                d_coeffs[0] *= 0.5; // Clenshaw convention: halve d_0.
            }

            // Clenshaw on forward coefficients.
            let mut b_next = 0.0_f64;
            let mut b_next_next = 0.0_f64;
            for k in (1..m).rev() {
                let b_k = 2.0 * x * b_next - b_next_next + self.coeffs[row_start + k];
                b_next_next = b_next;
                b_next = b_k;
            }
            *out_t = self.coeffs[row_start] + x * b_next - b_next_next;

            // Clenshaw on derivative coefficients (dx/dx side).
            let mut b_next = 0.0_f64;
            let mut b_next_next = 0.0_f64;
            for k in (1..m).rev() {
                let b_k = 2.0 * x * b_next - b_next_next + d_coeffs[k];
                b_next_next = b_next;
                b_next = b_k;
            }
            let deriv_dx = d_coeffs[0] + x * b_next - b_next_next;
            *out_d = deriv_dx * dx_dn;
        }
        (forward, deriv)
    }
}

/// Scalar (k = 1) surrogate used by the downstream dispatch
/// layers (see `TransmissionFitModel` / `PrecomputedTransmissionModel`).
///
/// This was an enum of `Gauss` vs `Chebyshev` during PR #475's
/// bench-off period; Chebyshev won the real-VENUS bench (12.2×
/// vs exact, 1e-15 accuracy at VENUS density) and Lanczos Gauss
/// was deleted per the issue's "drop the loser" contract.
/// The type alias is kept as a public stable name so callers and
/// downstream dispatch code aren't coupled to the winning impl's
/// concrete type — if a future research sprint finds a better
/// scalar surrogate, only the alias moves.
pub type ScalarSurrogatePlan = ScalarChebyshevPlan;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::{ResolutionPlan, TabulatedResolution};

    // ---------- Synthetic plan helpers (CI-hermetic) ----------

    /// Build a synthetic (energies, sigmas, ResolutionMatrix) triple
    /// with a uniform triangular-kernel resolution operator and a
    /// hand-designed multi-isotope σ pattern.  Avoids loading any
    /// fixture — these tests run on every `cargo test`.
    fn synthetic_setup(
        n_grid: usize,
        half_kernel: usize,
        k: usize,
    ) -> (Vec<f64>, Vec<f64>, crate::resolution::ResolutionMatrix) {
        assert!(n_grid > 2 * half_kernel);
        let energies: Vec<f64> = (0..n_grid).map(|i| 10.0 + i as f64).collect();
        // Build a ResolutionMatrix from a hand-constructed plan with
        // triangular-kernel rows — the same `make_synthetic_overlap_plan`
        // approach used in `resolution.rs` tests, inlined here to avoid
        // cross-module test visibility.
        let mut starts: Vec<u32> = Vec::with_capacity(n_grid + 1);
        starts.push(0);
        let mut lo_idx: Vec<u32> = Vec::new();
        let mut frac_arr: Vec<f64> = Vec::new();
        let mut weight_arr: Vec<f64> = Vec::new();
        let mut norm: Vec<f64> = Vec::with_capacity(n_grid);
        for i in 0..n_grid {
            let lo_min = i.saturating_sub(half_kernel);
            let lo_max = (i + half_kernel).min(n_grid - 2);
            let mut row_norm = 0.0_f64;
            for lo in lo_min..=lo_max {
                let d = (lo as i64 - i as i64).abs() as f64;
                let w = 1.0 - d / (half_kernel as f64 + 1.0);
                lo_idx.push(lo as u32);
                frac_arr.push(0.5);
                weight_arr.push(w);
                row_norm += w;
            }
            norm.push(row_norm);
            starts.push(lo_idx.len() as u32);
        }
        // Use the raw constructor via compile_to_matrix on a
        // manually-assembled plan.  ResolutionPlan's fields are
        // crate-private, so we build it via the canonical plan
        // constructor (`TabulatedResolution::plan`) would require a
        // kernel — so we instead invoke the test-visible constructor
        // pattern the resolution module already uses internally.
        //
        // For the surrogate tests we only need the compiled matrix,
        // not the plan; we therefore build the ResolutionMatrix
        // directly (mirroring compile_to_matrix's output format)
        // without going through ResolutionPlan.  This is done by
        // constructing the plan via the public `plan()` route from
        // a trivial TabulatedResolution proxy: a single-energy,
        // delta-kernel resolution that produces identity rows; then
        // overriding via a synthetic plan fixture would require
        // crate-private access.
        //
        // Simplest path: use a minimal `ResolutionPlan` surrogate by
        // directly building a `ResolutionMatrix`-equivalent CSR via
        // the `ResolutionPlan::compile_to_matrix` pathway.  Since
        // that method consumes only the public fields above, we
        // expose a test-only helper `from_raw_parts` on ResolutionPlan.
        // (Added in this module as `SyntheticPlanBuilder` below.)
        let plan =
            SyntheticPlanBuilder::new(energies.clone(), starts, lo_idx, frac_arr, weight_arr, norm)
                .build();
        let matrix = plan.compile_to_matrix();

        // Synthetic σ: k independent Gaussian resonances per isotope at
        // distinct energies, bounded in a physically plausible range.
        let mut sigmas = vec![0.0_f64; k * n_grid];
        for j in 0..k {
            let e_center = 10.0 + (j as f64 + 1.0) * (n_grid as f64) / (k as f64 + 1.0);
            let width = 3.0;
            for ell in 0..n_grid {
                let e = 10.0 + ell as f64;
                let g = (-((e - e_center).powi(2)) / (width * width)).exp();
                sigmas[j * n_grid + ell] = 100.0 * g + 5.0;
            }
        }
        (energies, sigmas, matrix)
    }

    /// Helper that exposes a way to build a `ResolutionPlan` from raw
    /// parts — needed because the fields are private to
    /// `resolution.rs`.  This test-only wrapper uses the same round-
    /// trip trick the resolution tests use: build via the public
    /// `TabulatedResolution::plan` surface on a trivial grid.  For the
    /// purpose of surrogate tests we don't care that the raw plan
    /// weights differ from what a real kernel would produce — what
    /// matters is that `compile_to_matrix` produces a valid CSR.
    struct SyntheticPlanBuilder {
        energies: Vec<f64>,
        starts: Vec<u32>,
        lo_idx: Vec<u32>,
        frac: Vec<f64>,
        weight: Vec<f64>,
        norm: Vec<f64>,
    }

    impl SyntheticPlanBuilder {
        fn new(
            energies: Vec<f64>,
            starts: Vec<u32>,
            lo_idx: Vec<u32>,
            frac: Vec<f64>,
            weight: Vec<f64>,
            norm: Vec<f64>,
        ) -> Self {
            Self {
                energies,
                starts,
                lo_idx,
                frac,
                weight,
                norm,
            }
        }

        /// Build a `ResolutionPlan` by going through the crate-public
        /// test-only constructor exposed on the resolution module.
        fn build(self) -> ResolutionPlan {
            crate::resolution::test_support::plan_from_raw_parts(
                self.energies,
                self.starts,
                self.lo_idx,
                self.frac,
                self.weight,
                self.norm,
            )
        }
    }

    // ---------- Tests ----------

    #[test]
    fn cubature_rejects_zero_isotopes() {
        let (_e, _s, matrix) = synthetic_setup(20, 3, 2);
        let err = SparseEmpiricalCubaturePlan::build(&matrix, &[], 0, &[vec![0.0]], &[0.0])
            .expect_err("k = 0 must reject");
        assert!(matches!(err, CubatureBuildError::ZeroIsotopes));
    }

    #[test]
    fn cubature_rejects_mismatched_sigmas() {
        let (_e, _s, matrix) = synthetic_setup(20, 3, 2);
        let err = SparseEmpiricalCubaturePlan::build(
            &matrix,
            &[0.0; 7], // wrong length
            2,
            &[vec![1e-4, 1e-4]],
            &[1e-4, 1e-4],
        )
        .expect_err("sigma grid mismatch must reject");
        assert!(matches!(err, CubatureBuildError::SigmaGridMismatch { .. }));
    }

    #[test]
    fn cubature_empty_matrix_empty_plan() {
        // Reuse the synthetic fabric but with n_grid = 0 — the helper
        // can't produce that directly (assertion), so build an empty
        // matrix via a zero-row plan.
        let plan = crate::resolution::test_support::plan_from_raw_parts(
            Vec::new(),
            vec![0_u32],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let matrix = plan.compile_to_matrix();
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &[], 3, &[vec![0.0; 3]], &[0.0; 3])
            .expect("empty matrix must build empty cubature");
        assert_eq!(cub.len(), 0);
        assert!(cub.is_empty());
        assert_eq!(cub.n_atoms(), 0);
        assert!(cub.target_energies().is_empty());
    }

    #[test]
    fn cubature_target_energies_mirror_matrix_grid() {
        let (energies, sigmas, matrix) = synthetic_setup(20, 3, 2);
        let train_max = [1e-4_f64, 1e-4];
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigmas, 2, &training, &anchor)
            .expect("build");
        // target_energies must byte-match the matrix's stored grid so
        // callers can use it as a cache key (same pattern as
        // ResolutionPlan / ResolutionMatrix).
        assert_eq!(cub.target_energies(), matrix.target_energies());
        assert_eq!(cub.target_energies(), energies.as_slice());
    }

    #[test]
    fn cubature_default_training_points_shape() {
        let train_max = [1e-4_f64, 2e-4, 5e-5];
        let pts = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        // S = k + 2 = 5 points for k = 3.
        assert_eq!(pts.len(), 5);
        for p in &pts {
            assert_eq!(p.len(), 3);
        }
        // First two points are quarter / three-quarter of train_max.
        for (i, &m) in train_max.iter().enumerate() {
            assert!((pts[0][i] - 0.25 * m).abs() < 1e-15);
            assert!((pts[1][i] - 0.75 * m).abs() < 1e-15);
        }
        // Remaining k points are axis-aligned.
        for (i, &max_i) in train_max.iter().enumerate() {
            for (j, &value) in pts[2 + i].iter().enumerate() {
                let expected = if i == j { max_i } else { 0.0 };
                assert!((value - expected).abs() < 1e-15);
            }
        }
        // Anchor is the midpoint.
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        for (i, &m) in train_max.iter().enumerate() {
            assert!((anchor[i] - 0.5 * m).abs() < 1e-15);
        }
    }

    /// Zero-weight CSR cells retained by
    /// [`crate::resolution::ResolutionPlan::compile_to_matrix`] for
    /// NaN-safety (the `frac == +0.0` branch) MUST NOT become
    /// cubature atoms, even though the LP's zero objective would let
    /// the simplex put arbitrary mass on them.  Codex round-3 finding
    /// on PR #474a — this test guards against regression.
    #[test]
    fn cubature_rejects_zero_weight_csr_cells_as_atoms() {
        // Hand-construct a 5-cell synthetic plan where every
        // regular-bracket entry has `frac = +0.0`, producing CSR
        // rows with an explicit `(lo + 1, 0.0)` zero-weight column.
        let energies: Vec<f64> = (0..5).map(|i| 10.0 + i as f64).collect();
        let mut starts: Vec<u32> = vec![0];
        let mut lo_idx: Vec<u32> = Vec::new();
        let mut frac: Vec<f64> = Vec::new();
        let mut weight: Vec<f64> = Vec::new();
        let mut norm: Vec<f64> = Vec::new();
        for i in 0..5 {
            // Row i: one regular-bracket entry at lo = i.min(3) with
            // frac = +0.0.  This produces CSR columns {i.min(3),
            // i.min(3) + 1} with values {1.0, 0.0} respectively.
            let lo = i.min(3);
            lo_idx.push(lo as u32);
            frac.push(0.0); // +0.0, not the -0.0 sentinel
            weight.push(1.0);
            norm.push(1.0);
            starts.push(lo_idx.len() as u32);
        }
        let plan = crate::resolution::test_support::plan_from_raw_parts(
            energies, starts, lo_idx, frac, weight, norm,
        );
        let matrix = plan.compile_to_matrix();

        // Confirm the matrix actually has zero-weight CSR cells.
        let total_nnz = matrix.nnz();
        let zero_weight_cells = matrix.values().iter().filter(|&&v| v == 0.0).count();
        assert!(
            zero_weight_cells > 0,
            "test fixture must include zero-weight CSR cells — got {total_nnz} nnz, {zero_weight_cells} zero",
        );

        // Build a cubature.  The resulting atoms must correspond ONLY
        // to CSR cells with non-zero weight.
        let sigmas = vec![0.5_f64, 1.0, 1.5, 2.0, 2.5];
        let train_max = [1e-4_f64];
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigmas, 1, &training, &anchor)
            .expect("build must succeed on zero-weight-cell fixture");

        // Collect the σ values retained as atoms; each must correspond
        // to a support column with non-zero CSR value.  With k = 1,
        // the atom sigma is either 0.5, 1.0, 1.5, 2.0, or 2.5 —
        // whichever column was non-zero in the source row.
        for (i, window) in cub.row_starts().windows(2).enumerate() {
            let (s, e) = (window[0] as usize, window[1] as usize);
            for q in s..e {
                let atom_sigma = cub.atoms()[q];
                // The corresponding CSR cell at the nearest source
                // column must have non-zero weight.
                let row_start = matrix.row_starts()[i] as usize;
                let row_end = matrix.row_starts()[i + 1] as usize;
                let row_cols = &matrix.col_indices()[row_start..row_end];
                let row_vals = &matrix.values()[row_start..row_end];
                let source_nonzero = row_cols
                    .iter()
                    .zip(row_vals)
                    .find(|&(&col, _)| (sigmas[col as usize] - atom_sigma).abs() < 1e-15)
                    .map(|(_, &v)| v);
                assert!(
                    source_nonzero.is_some() && source_nonzero.unwrap() > 0.0,
                    "row {i} atom sigma {atom_sigma} has no non-zero source in CSR row",
                );
            }
        }
    }

    #[test]
    fn cubature_build_error_display() {
        // Cover each error variant's Display message so a future
        // refactor that breaks the formatting fails loudly.
        let e = CubatureBuildError::ZeroIsotopes;
        assert!(format!("{e}").contains("at least one isotope"));

        let e = CubatureBuildError::ZeroTrainingDensities;
        assert!(format!("{e}").contains("at least one training density"));

        let e = CubatureBuildError::SigmaGridMismatch {
            expected: 100,
            actual: 50,
        };
        let s = format!("{e}");
        assert!(s.contains("sigmas") && s.contains("100") && s.contains("50"));

        let e = CubatureBuildError::TrainingDensityLength {
            expected: 3,
            actual: 2,
            index: 7,
        };
        let s = format!("{e}");
        assert!(s.contains("training_densities[7]") && s.contains("length 2"));

        let e = CubatureBuildError::AnchorLength {
            expected: 3,
            actual: 5,
        };
        let s = format!("{e}");
        assert!(s.contains("jacobian_anchor") && s.contains("length 5"));

        let e = CubatureBuildError::LpInfeasible { row: 42 };
        assert!(format!("{e}").contains("row 42"));
    }

    /// Forward equivalence at the training densities: the cubature's
    /// feasibility LP pins `phi_fwd @ x = phi_fwd @ w_exact`, so
    /// `cubature.forward(n^(s))` equals `sum_q R_{iq} exp(-n^(s)
    /// · σ_q)` (the exact surrogate output) at every training density
    /// `n^(s)` — row by row.
    #[test]
    fn cubature_forward_matches_exact_at_training_densities() {
        let (_e, sigmas, matrix) = synthetic_setup(40, 4, 2);
        let train_max = [2e-4_f64, 1.5e-4];
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigmas, 2, &training, &anchor)
            .expect("build");

        for (s, n) in training.iter().enumerate() {
            let t_cub = cub.forward(n);
            let t_exact = exact_forward(&matrix, &sigmas, 2, n);
            let max_err = max_hybrid_err(&t_cub, &t_exact);
            assert!(
                max_err < 1e-9,
                "training[{s}] n={n:?} max hybrid err = {max_err:.3e} (expected < 1e-9)",
            );
        }
    }

    /// Forward accuracy at a held-out density inside the training
    /// convex hull: the cubature's bias should be bounded (Jensen-like
    /// term on the missing feature directions) but still within the
    /// ≤1e-3 max abs error band that codex04 measured on real VENUS.
    #[test]
    fn cubature_forward_held_out_bounded_error() {
        let (_e, sigmas, matrix) = synthetic_setup(40, 4, 2);
        let train_max = [2e-4_f64, 1.5e-4];
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigmas, 2, &training, &anchor)
            .expect("build");

        // Moderate density at 50 % of the box, both isotopes active.
        let n_test = vec![0.5 * train_max[0], 0.5 * train_max[1]];
        let t_cub = cub.forward(&n_test);
        let t_exact = exact_forward(&matrix, &sigmas, 2, &n_test);
        let max_abs = t_cub
            .iter()
            .zip(t_exact.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-2,
            "held-out max abs err = {max_abs:.3e} (expected < 1e-2)",
        );
    }

    /// Jacobian at the anchor density: the cubature's LP pins
    /// `phi_grad @ x = phi_grad @ w_exact`, so the Jacobian columns at
    /// `n*` should match the exact Jacobian `-R[-σ_ℓ exp(-n* · σ)]` to
    /// LP tolerance.
    #[test]
    fn cubature_jacobian_matches_exact_at_anchor() {
        let (_e, sigmas, matrix) = synthetic_setup(30, 4, 3);
        let train_max = [2e-4_f64, 1.5e-4, 1e-4];
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigmas, 3, &training, &anchor)
            .expect("build");

        let (_t_cub, j_cub) = cub.forward_and_jacobian(&anchor);
        let j_exact = exact_jacobian(&matrix, &sigmas, 3, &anchor);
        let max_err = max_hybrid_err(&j_cub, &j_exact);
        // Looser than the forward-at-training-densities bound (1e-9)
        // because Jacobian features `σ_ℓ · exp(-n · σ)` have magnitudes
        // O(50) (σ in barns) vs forward features' O(1).  The simplex
        // solver's equality residuals accumulate ~1e-8 abs error which
        // is LP precision, not a cubature correctness issue — codex04's
        // Python reference implementation hits the same band.
        assert!(
            max_err < 1e-7,
            "Jacobian at anchor max hybrid err = {max_err:.3e} (expected < 1e-7)",
        );
    }

    /// Row weights sum to 1 after renormalization.
    #[test]
    fn cubature_rows_are_probability_measures() {
        let (_e, sigmas, matrix) = synthetic_setup(30, 4, 2);
        let train_max = [2e-4_f64, 1.5e-4];
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigmas, 2, &training, &anchor)
            .expect("build");
        for i in 0..cub.len() {
            let s = cub.row_starts()[i] as usize;
            let e = cub.row_starts()[i + 1] as usize;
            let row_sum: f64 = cub.weights()[s..e].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-12,
                "row {i} sum = {row_sum} (expected 1.0 within 1e-12)",
            );
        }
    }

    /// k = 6 curse-of-dim stress: confirm the build succeeds, atoms
    /// stay bounded (~S+k+1 per row), and held-out forward error stays
    /// modest.  Mirrors codex04's k = 6 independent-Hf scenario in
    /// structural shape.
    #[test]
    fn cubature_k6_builds_and_evaluates() {
        let (_e, sigmas, matrix) = synthetic_setup(30, 4, 6);
        let train_max: Vec<f64> = (0..6).map(|j| 1e-4 * (1.0 + 0.2 * j as f64)).collect();
        // S training points = 2 midpoints + k axis-aligned points = 8.
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigmas, 6, &training, &anchor)
            .expect("k=6 build");

        // Atom counts: codex04's Carathéodory bound is S + k + 1 = 15.
        // The LP may produce fewer (columns genuinely redundant).  Allow
        // a small slack above the theoretical bound for numerical edge
        // cases.
        let max_atoms = cub
            .row_starts()
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        assert!(
            max_atoms <= 18,
            "k=6 max atoms/row = {max_atoms} (expected ≤ 18 = S+k+1+slack)",
        );

        // Forward at held-out density inside the box.
        let n_test: Vec<f64> = train_max.iter().map(|&x| 0.4 * x).collect();
        let t_cub = cub.forward(&n_test);
        let t_exact = exact_forward(&matrix, &sigmas, 6, &n_test);
        let max_abs = t_cub
            .iter()
            .zip(t_exact.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-2,
            "k=6 held-out max abs err = {max_abs:.3e} (expected < 1e-2)",
        );
    }

    // ---------- helpers ----------

    fn exact_forward(
        matrix: &crate::resolution::ResolutionMatrix,
        sigmas: &[f64],
        k: usize,
        n: &[f64],
    ) -> Vec<f64> {
        let n_rows = matrix.len();
        // T_un[ℓ] = exp(-Σ_j n_j σ_j(ℓ)).
        let mut t_un = vec![0.0_f64; n_rows];
        for (ell, t) in t_un.iter_mut().enumerate() {
            let mut dot = 0.0_f64;
            for j in 0..k {
                dot += n[j] * sigmas[j * n_rows + ell];
            }
            *t = (-dot).exp();
        }
        crate::resolution::apply_r(matrix, &t_un)
    }

    fn exact_jacobian(
        matrix: &crate::resolution::ResolutionMatrix,
        sigmas: &[f64],
        k: usize,
        n: &[f64],
    ) -> Vec<f64> {
        let n_rows = matrix.len();
        let mut jac = vec![0.0_f64; n_rows * k];
        // ∂T_i/∂n_ℓ = -Σ_q R_{iq} σ_ℓ(q) exp(-n · σ_q).
        let mut t_un = vec![0.0_f64; n_rows];
        for (q, t) in t_un.iter_mut().enumerate() {
            let mut dot = 0.0_f64;
            for j in 0..k {
                dot += n[j] * sigmas[j * n_rows + q];
            }
            *t = (-dot).exp();
        }
        for ell in 0..k {
            let mut inner = vec![0.0_f64; n_rows];
            for q in 0..n_rows {
                inner[q] = -sigmas[ell * n_rows + q] * t_un[q];
            }
            let col = crate::resolution::apply_r(matrix, &inner);
            for (i, &v) in col.iter().enumerate() {
                jac[i * k + ell] = v;
            }
        }
        jac
    }

    fn max_hybrid_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let denom = x.abs().max(y.abs()).max(1e-12);
                (x - y).abs() / denom
            })
            .fold(0.0_f64, f64::max)
    }

    // ---------- Real-VENUS-kernel tests (#[ignore]-gated) ----------
    //
    // Gated on the PLEIADES resolution fixture per the established
    // pattern in `resolution.rs`.  These exercise the cubature on the
    // production-scale VENUS kernel at the actual grid size surrogates
    // will see in the wiring follow-up (PR #474b).

    /// k = 1 grouped case: the cubature should match the exact
    /// `ResolutionMatrix @ exp(-n σ)` forward output to LP precision
    /// at the training densities, and produce bounded error at
    /// held-out densities inside the training box.  Codex04's 20-seed
    /// KL follow-up showed 1.27× scatter inflation on grouped-Hf k=1
    /// — this test does not re-measure that; it only checks forward-
    /// model correctness.
    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn cubature_real_venus_k1_forward_equivalence() {
        let res_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("_fts_bl10_0p5meV_1keV_25pts.txt");
        let text = std::fs::read_to_string(&res_path).expect(
            "missing PLEIADES resolution file at repo root (see `#[ignore]` message for details)",
        );
        let tab = TabulatedResolution::from_text(&text, 25.0).expect("parse fixture");

        // Smaller production-ish grid (512 instead of 3471) to keep
        // LP-per-row cost tractable within a single test — the
        // cubature structure is grid-independent, so 512 suffices to
        // show correctness.  Full 3471 sweep lives in the wiring
        // follow-up.
        let n_grid = 512_usize;
        let energies: Vec<f64> = (0..n_grid)
            .map(|i| 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64))
            .collect();
        let plan = tab.plan(&energies).expect("build plan on sorted grid");
        let matrix = plan.compile_to_matrix();

        // Synthetic Gaussian-resonance σ on the real energy grid —
        // we don't need real ENDF σ to prove the cubature math; a
        // physically plausible σ shape is sufficient.  The real-
        // ENDF closed-loop validation belongs in PR #474b.
        let sigma: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let g = (-((e - 80.0).powi(2)) / 8.0).exp();
                100.0 * g + 5.0
            })
            .collect();

        let train_max = [2e-4_f64];
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigma, 1, &training, &anchor)
            .expect("build k=1 cubature on real VENUS kernel");

        // At each training density, cubature = exact.
        for n_s in training.iter() {
            let t_cub = cub.forward(n_s);
            let t_exact = exact_forward(&matrix, &sigma, 1, n_s);
            let max_err = max_hybrid_err(&t_cub, &t_exact);
            assert!(
                max_err < 1e-9,
                "VENUS k=1 training n={n_s:?} max err = {max_err:.3e}",
            );
        }

        // At held-out density (VENUS production ~ 1.6e-4), bounded
        // error.  Codex04's real-VENUS Hf aggregated fit showed a
        // density shift of 1.66e-4 relative — which means forward
        // error at the optimum is at least that small.  Here we
        // allow 1e-3 max abs error as a generous ceiling; the actual
        // value on the Beer-Lambert `T ∈ [0, 1]` output is typically
        // an order of magnitude tighter.
        let n_venus = vec![1.6e-4];
        let t_cub = cub.forward(&n_venus);
        let t_exact = exact_forward(&matrix, &sigma, 1, &n_venus);
        let max_abs = t_cub
            .iter()
            .zip(t_exact.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-3,
            "VENUS k=1 held-out n=1.6e-4 max abs err = {max_abs:.3e}",
        );

        // Log compression ratio for diagnostics.
        let exact_nnz = matrix.nnz();
        let cub_atoms = cub.n_atoms();
        eprintln!(
            "VENUS k=1 cubature compression: {exact_nnz} exact nnz → {cub_atoms} atoms ({:.1}× compression)",
            exact_nnz as f64 / cub_atoms as f64,
        );
    }

    // ── Scalar (k = 1) surrogate tests — PR #475 ───────────────────

    /// Build a 1-isotope synthetic σ + matrix pair, shared by both
    /// scalar surrogate tests.
    fn scalar_setup(
        n_grid: usize,
        half_kernel: usize,
    ) -> (Vec<f64>, crate::resolution::ResolutionMatrix) {
        let (_e, sigmas, matrix) = synthetic_setup(n_grid, half_kernel, 1);
        (sigmas, matrix) // sigmas for k=1 is flat length n_grid
    }

    #[test]
    fn scalar_chebyshev_matches_exact_at_multiple_densities() {
        let (sigmas_flat, matrix) = scalar_setup(40, 4);
        let sigma = &sigmas_flat;
        let n_max = 2e-4_f64;
        let plan =
            ScalarChebyshevPlan::build(&matrix, sigma, n_max, 16).expect("build chebyshev plan");
        for n in [1e-5_f64, 1e-4, 1.6e-4] {
            let t_plan = plan.forward_scalar(n);
            let t_un: Vec<f64> = sigma.iter().map(|&s| (-n * s).exp()).collect();
            let t_exact = crate::resolution::apply_r(&matrix, &t_un);
            let max_err = max_hybrid_err(&t_plan, &t_exact);
            // Chebyshev accuracy depends on M; for M = 16 on a
            // bounded T ∈ [0, 1] signal, expect ≤ 1e-8.
            assert!(
                max_err < 1e-8,
                "Chebyshev vs exact at n = {n:.1e}: max hybrid err = {max_err:.3e}",
            );
        }
    }

    #[test]
    fn scalar_chebyshev_derivative_matches_finite_difference() {
        let (sigmas_flat, matrix) = scalar_setup(30, 4);
        let sigma = &sigmas_flat;
        let n_max = 2e-4_f64;
        let plan = ScalarChebyshevPlan::build(&matrix, sigma, n_max, 16).expect("build");
        let n = 1.6e-4_f64;
        let h = 1e-8_f64;
        let (_t, dt_an) = plan.forward_and_derivative_scalar(n);
        let t_plus = plan.forward_scalar(n + h);
        let t_minus = plan.forward_scalar(n - h);
        for i in 0..plan.len() {
            let dt_fd = (t_plus[i] - t_minus[i]) / (2.0 * h);
            let denom = dt_an[i].abs().max(dt_fd.abs()).max(1e-12);
            let rel = (dt_an[i] - dt_fd).abs() / denom;
            assert!(
                rel < 1e-4,
                "row {i}: analytic {} vs FD {} rel = {:.3e}",
                dt_an[i],
                dt_fd,
                rel,
            );
        }
    }

    #[test]
    fn scalar_chebyshev_rejects_invalid_box() {
        let (sigmas_flat, matrix) = scalar_setup(20, 3);
        let err = ScalarChebyshevPlan::build(&matrix, &sigmas_flat, 0.0, 16)
            .expect_err("n_max = 0 must reject");
        assert!(matches!(
            err,
            ScalarSurrogateBuildError::InvalidChebyshevBox { .. }
        ));
        let err = ScalarChebyshevPlan::build(&matrix, &sigmas_flat, 1e-4, 1)
            .expect_err("M = 1 must reject");
        assert!(matches!(
            err,
            ScalarSurrogateBuildError::InvalidChebyshevBox { .. }
        ));
    }

    #[test]
    fn scalar_plan_rejects_sigma_size_mismatch() {
        let (_sigmas_flat, matrix) = scalar_setup(20, 3);
        let wrong = vec![0.0_f64; 15];
        let err = ScalarChebyshevPlan::build(&matrix, &wrong, 1e-4, 16)
            .expect_err("Chebyshev sigma mismatch must reject");
        assert!(matches!(
            err,
            ScalarSurrogateBuildError::SigmaGridMismatch { .. }
        ));
    }

    /// Real-VENUS regression test for the Chebyshev scalar
    /// surrogate on the 3471-bin VENUS production grid with
    /// synthetic Hf-like σ.  Asserts forward accuracy ≤ 1e-5 at
    /// VENUS density (issue #475 success criterion) and logs
    /// per-call wall time.  PR #475 benched this against a
    /// Lanczos σ-pushforward Gauss quadrature on the same kernel
    /// and picked Chebyshev (12.2× vs exact, 1.89e-15 accuracy)
    /// over Gauss (7.1× vs exact, 4.00e-15 accuracy).  Lanczos
    /// code has been deleted; this test now pins the winner's
    /// numbers as a regression guard.
    #[test]
    #[ignore = "requires PLEIADES resolution file `_fts_bl10_0p5meV_1keV_25pts.txt` at repo root (gitignored by policy)"]
    fn scalar_chebyshev_real_venus_k1_regression() {
        let res_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("_fts_bl10_0p5meV_1keV_25pts.txt");
        let text = std::fs::read_to_string(&res_path).expect(
            "missing PLEIADES resolution file at repo root (see `#[ignore]` message for details)",
        );
        let tab = TabulatedResolution::from_text(&text, 25.0).expect("parse fixture");

        // VENUS production grid size.
        let n_grid = 3471_usize;
        let energies: Vec<f64> = (0..n_grid)
            .map(|i| 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64))
            .collect();
        let plan = tab.plan(&energies).expect("build plan on sorted grid");
        let matrix = plan.compile_to_matrix();
        let matrix_nnz = matrix.nnz();

        // Synthetic Hf-like σ: Gaussian resonance peaks + baseline
        // potential scattering.  Rough ENDF-Hf magnitudes (peaks
        // up to O(1e3) barns, baseline ~10 barns).
        let sigma: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let peak_a = 1200.0 * (-((e - 80.0).powi(2)) / 8.0).exp();
                let peak_b = 600.0 * (-((e - 120.0).powi(2)) / 6.0).exp();
                let peak_c = 300.0 * (-((e - 160.0).powi(2)) / 10.0).exp();
                peak_a + peak_b + peak_c + 10.0
            })
            .collect();

        // Training box: 2 × VENUS density.  M = 16 Chebyshev nodes
        // — PR #475 bench showed this achieves 1e-15 forward
        // accuracy on the whole box.
        let n_max = 2e-4_f64;
        let t_build = std::time::Instant::now();
        let cheb =
            ScalarChebyshevPlan::build(&matrix, &sigma, n_max, 16).expect("build Chebyshev plan");
        let t_build = t_build.elapsed();

        // Forward accuracy at VENUS density.
        let n_venus = 1.6e-4_f64;
        let t_exact = exact_forward(&matrix, &sigma, 1, &[n_venus]);
        let t_cheb = cheb.forward_scalar(n_venus);
        let max_err = max_hybrid_err(&t_cheb, &t_exact);

        // Per-forward timing.
        let n_iters = 1000;
        let t0 = std::time::Instant::now();
        let mut sink = 0.0_f64;
        for _ in 0..n_iters {
            sink += cheb.forward_scalar(n_venus)[0];
        }
        let t_fwd = t0.elapsed() / n_iters as u32;
        std::hint::black_box(sink);

        let t_un: Vec<f64> = sigma.iter().map(|&s| (-n_venus * s).exp()).collect();
        let t0 = std::time::Instant::now();
        let mut sink = 0.0_f64;
        for _ in 0..n_iters {
            sink += crate::resolution::apply_r(&matrix, &t_un)[0];
        }
        let t_exact_fwd = t0.elapsed() / n_iters as u32;
        std::hint::black_box(sink);

        eprintln!();
        eprintln!(
            "=== scalar Chebyshev VENUS k=1 regression (n_grid = {n_grid}, matrix nnz = {matrix_nnz}) ==="
        );
        eprintln!(
            "Chebyshev (M=16):  build = {:>9.1?}  n_coeff = {:>7}  fwd = {:>8.2?}  max_err @ VENUS = {:.3e}",
            t_build,
            n_grid * 16,
            t_fwd,
            max_err,
        );
        eprintln!(
            "Exact apply_r:     build = {:>9}                     fwd = {:>8.2?}  (reference)",
            "n/a", t_exact_fwd,
        );
        eprintln!(
            "Cheb speedup vs exact: {:.1}×",
            t_exact_fwd.as_nanos() as f64 / t_fwd.as_nanos() as f64,
        );
        eprintln!();

        // Issue #475 success criteria.
        assert!(
            max_err < 1e-5,
            "Chebyshev forward max err {max_err:.3e} exceeds 1e-5 ceiling",
        );
    }
}
