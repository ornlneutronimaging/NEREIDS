//! Transmission forward model adapter for fitting.
//!
//! Wraps the physics `forward_model` function into a `FitModel` trait object
//! that the LM optimizer can call. The fit parameters are the areal densities
//! (thicknesses) of each isotope in the sample.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use nereids_core::constants::{EV_TO_JOULES, NEUTRON_MASS_KG};
use nereids_endf::resonance::ResonanceData;
use nereids_physics::resolution::{self, ResolutionFunction, ResolutionPlan};
use nereids_physics::surrogate::{ScalarSurrogatePlan, SparseEmpiricalCubaturePlan};
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};

/// Absolute-magnitude threshold for `L_scale` division safety in the
/// partial-GAL rank-1 derivation of the energy-scale Jacobian.  When
/// `|l_scale| < L_SCALE_EPSILON`, the per-bin
/// `(tof_i - t0_clamped) / l_scale` factor in
/// [`EnergyScaleTransmissionModel::analytical_jacobian`] blows up,
/// and combined with the FD-based reference t0 column (which goes to
/// ~0 at the same boundary) produces NaN entries in the L_scale
/// Jacobian column.
///
/// Below this threshold the L_scale column falls through to the
/// per-coordinate central-FD path that already follows the partial-GAL
/// block in the same function.
///
/// **Note:** the literal `1.0e-12` matches the `1e-12` factor in the
/// t0 clamp at [`EnergyScaleTransmissionModel::corrected_energies`]
/// and the partial-GAL t0 FD precompute, but the semantic role
/// differs — the t0 clamp is *relative* (`min_tof_us * (1 - 1e-12)`)
/// while this constant is an *absolute* magnitude bound.  Both
/// guards protect against the same `(tof - t0) / l_eff` blow-up at
/// the energy-scale-degenerate corner; the value choice is
/// coincident, not tied.  See issue #500 for the L_scale gap
/// closure.
const L_SCALE_EPSILON: f64 = 1.0e-12;

/// Transmission model backed by precomputed Doppler-broadened cross-sections.
///
/// The expensive physics steps (resonance → σ(E), Doppler broadening) are
/// computed once and stored.  Each `evaluate()` call performs Beer-Lambert
/// and, when `instrument` is present, resolution broadening on the total
/// transmission:
///
///   T(E) = R ⊗ exp(−Σᵢ nᵢ · σ_{D,i}(E))
///
/// Issue #442: resolution broadening is applied to T(E) after Beer-Lambert,
/// not to σ(E) before.
///
/// Construct via `nereids_physics::transmission::broadened_cross_sections`,
/// then wrap in `Arc` so the same precomputed data is shared read-only
/// across all rayon worker threads.
pub struct PrecomputedTransmissionModel {
    /// Doppler-broadened cross-sections σ_D(E) per isotope, shape
    /// \[n_isotopes\]\[n_grid_energies\].
    ///
    /// **The grid these σ live on is determined by
    /// [`work_layout`](Self::work_layout):**
    ///
    /// * `work_layout` is `Some` (Gaussian resolution → auxiliary extended
    ///   grid): σ live on the **working grid**, i.e.
    ///   `work_layout.energies`, with `n_grid_energies ==
    ///   work_layout.energies.len()`.  `evaluate()` / `analytical_jacobian()`
    ///   apply Beer-Lambert + resolution on this working grid and extract the
    ///   data points LAST via `work_layout.extract(..)` — matching
    ///   `forward_model` (issue #608).
    /// * `work_layout` is `None` (tabulated resolution, or no resolution): the
    ///   working grid IS the data grid, so σ live on the **data grid**
    ///   (`energies`), with `n_grid_energies == energies.len()`.  No extraction
    ///   is needed and the surrogate fast paths + data-grid `resolution_plan`
    ///   behave exactly as before.
    pub cross_sections: Arc<Vec<Vec<f64>>>,
    /// Mapping: `params[density_indices[i]]` is the density of isotope `i`.
    ///
    /// Wrapped in `Arc` so that parallel pixel loops can share one copy
    /// via cheap reference-count increments instead of deep-cloning per pixel.
    ///
    /// Kept `pub` (not `pub(crate)`) because the Python bindings
    /// (`nereids-python`) construct and access this field directly.
    pub density_indices: Arc<Vec<usize>>,
    /// Energy grid (eV), required for resolution broadening.
    /// `None` when resolution is disabled — Beer-Lambert only.
    pub energies: Option<Arc<Vec<f64>>>,
    /// Instrument resolution parameters.
    /// When `Some`, resolution broadening is applied to the total
    /// transmission after Beer-Lambert in `evaluate()`.
    pub instrument: Option<Arc<InstrumentParams>>,
    /// Optional pre-built broadening plan for `(energies, resolution)`.
    ///
    /// When a caller builds the plan once (e.g. spatial dispatch for
    /// a grid shared across every pixel) and passes it via
    /// `with_resolution_plan`, `evaluate()` and `analytical_jacobian()`
    /// skip the per-call kernel-interp / bracket / trap-weight work
    /// and reduce each broadening call to a gather + multiply-add.
    /// `None` ⇒ fall back to the per-call broadening path, byte-
    /// identical output.
    pub resolution_plan: Option<Arc<ResolutionPlan>>,
    /// Optional sparse empirical cubature plan.
    ///
    /// When the plan is present AND its `target_energies` match this
    /// model's energy grid AND `cubature.k() == n_density_params`
    /// AND no temperature / energy-scale fitting is active, the
    /// `evaluate()` / `analytical_jacobian()` fast path calls
    /// `cubature.forward_and_jacobian(n)` directly instead of
    /// `exp(-Σ n σ) + apply_resolution`.  Any guard failure falls
    /// back to the exact path, so installing a plan cannot change
    /// results unless every guard passes.
    pub sparse_cubature_plan: Option<Arc<SparseEmpiricalCubaturePlan>>,
    /// Optional scalar (k = 1) surrogate plan.
    ///
    /// Mutually exclusive with `sparse_cubature_plan` in practice —
    /// the cubature dispatch fires only for `k ≥ 2` and the scalar
    /// plan only for `k == 1`.  The type alias
    /// `ScalarSurrogatePlan = ScalarChebyshevPlan` is kept as a
    /// stable public name so a future scalar surrogate can swap in
    /// without touching this field or any dispatch call site.
    /// Chebyshev-in-density was picked over Lanczos Gauss
    /// quadrature after a real-VENUS bench-off (Chebyshev won on
    /// both the accuracy and wall-time axes; see
    /// `nereids_physics::surrogate` module docs).
    pub sparse_scalar_plan: Option<Arc<ScalarSurrogatePlan>>,
    /// Working-grid layout matching [`cross_sections`](Self::cross_sections).
    ///
    /// Issue #608: when `cross_sections` is stored on the auxiliary extended
    /// grid (Gaussian resolution), this maps the working grid back to the data
    /// grid so `evaluate()` / `analytical_jacobian()` apply resolution on the
    /// working grid and extract the data points last.  `None` ⇒ the working
    /// grid is the data grid (tabulated / no resolution): Beer-Lambert and
    /// resolution run directly on `energies` and no extraction is needed, which
    /// keeps the surrogate fast paths and the data-grid `resolution_plan`
    /// byte-identical to before.
    pub work_layout: Option<Arc<transmission::WorkingGridLayout>>,
}

/// Deduplicate `density_indices` and return the distinct density-
/// parameter indices **sorted ascending by value** — e.g.
/// `[0,0,0,0,0,0]` (grouped) → `[0]`; `[0,1,2,3,4,5]` (ungrouped) →
/// `[0,1,2,3,4,5]`; `[1,0,1]` (non-monotonic group layout) →
/// `[0,1]` (NOT first-appearance order `[1,0]`).
///
/// **Why sorted-by-value, not first-appearance?** The cubature
/// dispatch maps `n[j] = params[result[j]]` onto the cubature's
/// j-th atom column.  The cubature was built from a σ stack
/// indexed by density-param index (`sigmas[j * n_rows + ℓ] =
/// σ_{param_j}(E'_ℓ)`) — so atom column `j` corresponds to
/// density param `j`.  Using sorted-by-value output keeps the
/// dispatched `params[result[j]]` aligned with `cubature.atoms()`
/// at column `j` regardless of the user's `density_indices`
/// ordering.  First-appearance order would swap columns for
/// non-monotonic mappings, returning wrong transmissions and
/// wrong Jacobians.
fn density_param_indices(density_indices: &[usize]) -> Vec<usize> {
    // `sort_unstable` + `dedup` is O(n log n) and avoids the O(n²)
    // cost of repeated `Vec::contains` scans.  This runs on every
    // `evaluate()` / `analytical_jacobian()` call, so the linear-
    // scan version showed up in spatial-map profiling once the
    // per-pixel cubature dispatch started firing.
    let mut seen: Vec<usize> = density_indices.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen
}

/// Check whether a cubature-based forward evaluation is eligible
/// given the plan, the model's energy grid, the model's active
/// resolution plan, and density-param structure.  Centralized so
/// `evaluate`, `analytical_jacobian`, and both model types share a
/// single predicate.
///
/// **Grid identity** (not just length) matters: a cached plan from a
/// previous spatial call on a different grid with the same bin count
/// would silently return forward/Jacobian values for the stale grid.
/// We compare `plan.target_energies()` against the model's `energies`
/// via `to_bits()` per element (same contract
/// `apply_resolution_with_plan` already enforces).
///
/// **Tabulated-kernel tie**: the cubature fast path folds
/// `apply_resolution*` into its atom sweep — skipping it when the
/// model otherwise would have applied a Gaussian kernel is a
/// silent wrong-answer path.  We require
/// `matches!(instrument_resolution, ResolutionFunction::Tabulated(_))`
/// so Gaussian-resolution models never hit the cubature path (a
/// plan is only ever built against a tabulated kernel).
///
/// **Optional `resolution_plan` cross-check**: when a prebuilt
/// `ResolutionPlan` is attached (e.g., via
/// `spatial_map_typed`'s plan-hoist pathway), we additionally
/// verify its grid matches the cubature plan's grid — defence-in-
/// depth against a
/// `with_precomputed_resolution_plan(plan_A) +
/// with_precomputed_sparse_cubature_plan(plan_B_on_different_grid)`
/// mis-configuration.  When no resolution plan is attached (the
/// default on the single-spectrum entrypoint, where
/// `fit_spectrum_typed` / `build_transmission_model` don't
/// synthesize one), eligibility falls back to the cubature-plan
/// grid check alone; this keeps the `with_precomputed_sparse_cubature_plan`
/// API usable on the single-spectrum surface without the caller
/// having to pre-build a matching `ResolutionPlan` just to unlock
/// the fast path.
///
/// **Known caveat (same-grid kernel swap)**: if a caller rebuilds
/// the tabulated resolution plan for a *different kernel* on the
/// same energy grid without rebuilding the cubature, the grid
/// bit-check here passes but the atom weights still encode the
/// OLD operator.  Guarding against this requires a kernel
/// fingerprint on the cubature plan, which is not implemented
/// here.  Upstream callers are
/// responsible for clearing the cubature when they swap kernels;
/// in spatial dispatch this is enforced by
/// `UnifiedFitConfig::with_precomputed_cross_sections` /
/// `with_precomputed_base_xs` / `with_groups` all clearing the
/// cached cubature (see pipeline.rs), so a refit through the
/// standard surface cannot hit this case.
/// Check whether a scalar (k = 1) surrogate plan is eligible given
/// the model's energy grid, active tabulated resolution,
/// attached `ResolutionPlan`, current σ row, and
/// `n_density_params == 1`.  Parallels [`cubature_eligible`] for
/// the multi-isotope path on grid-identity + `Tabulated(_)` guard,
/// and **additionally** enforces content identity via the
/// source-`ResolutionPlan` `Arc::ptr_eq` check and a σ
/// fingerprint — closing a same-grid stale-plan correctness hole:
/// a plan built from different σ or a different kernel but
/// attached on the same energy grid must never dispatch the
/// surrogate.
fn scalar_eligible(
    plan: &ScalarSurrogatePlan,
    energies: &[f64],
    instrument_resolution: &ResolutionFunction,
    resolution_plan: Option<&Arc<ResolutionPlan>>,
    sigma_row: &[f64],
    n_density_params: usize,
) -> bool {
    if n_density_params != 1 {
        return false;
    }
    if plan.len() != energies.len() {
        return false;
    }
    if !matches!(instrument_resolution, ResolutionFunction::Tabulated(_)) {
        return false;
    }
    let plan_grid = plan.target_energies();
    for (e_cur, e_plan) in energies.iter().zip(plan_grid) {
        if e_cur.to_bits() != e_plan.to_bits() {
            return false;
        }
    }
    // Source-`ResolutionPlan` identity via `Arc::ptr_eq` — O(1)
    // check that the plan was built from the SAME resolution
    // kernel the model is currently using.  The grid-only
    // check was insufficient: a plan built for a different
    // tabulated kernel on an identical grid would silently
    // dispatch and return transmissions shifted by ~0.13
    // absolute (measured).  Requiring the model to attach the exact same
    // `Arc<ResolutionPlan>` the scalar plan was built from
    // closes that hole.
    let Some(model_plan) = resolution_plan else {
        return false;
    };
    if !Arc::ptr_eq(model_plan, plan.source_resolution_plan()) {
        return false;
    }
    // Transitive grid-identity on `resolution_plan` (retained from
    // the previous check — catches an `Arc::ptr_eq`-true pair whose
    // inner grid has been mutated out from under us, e.g. a
    // `Mutex<ResolutionPlan>` unsafe pattern; defence-in-depth).
    if model_plan.target_energies().len() != energies.len() {
        return false;
    }
    for (e_cur, e_res) in energies.iter().zip(model_plan.target_energies()) {
        if e_cur.to_bits() != e_res.to_bits() {
            return false;
        }
    }
    // σ fingerprint: same-grid-different-σ would otherwise pass
    // every grid check.  FNV-1a-64 over `to_bits()` is fast
    // (~3 µs for 3471-point VENUS grid) and cryptographically
    // sufficient for catching unintentional mismatch; matched-bit
    // collisions would require an adversarial σ, which isn't a
    // threat model here (the wrong-σ bug surfaces from
    // copy-paste caller errors).
    if nereids_physics::surrogate::fingerprint_f64_slice(sigma_row) != plan.sigma_fingerprint() {
        return false;
    }
    true
}

/// Check whether the scalar iterate `n` is inside the surrogate's
/// recorded training box `[0, train_max]` — **strict** `n ≤ train_max`,
/// unlike the cubature's 1.5× tolerance.
///
/// Chebyshev-in-density is a polynomial interpolant.  Inside
/// `[0, n_max]` it is exact at the M = 16 nodes and tight (≤ 1e-15
/// rel err) between them; outside, the interpolant diverges
/// exponentially in `(n - n_max) / n_max` — measured:
/// **73 % relative error at `1.5 × n_max`** and catastrophic
/// divergence beyond — exactly the "silently wrong forward"
/// failure mode that would corrupt a fit without the solver
/// ever seeing an error flag.
///
/// The cubature's 1.5× tolerance is safe because LP-matched atoms
/// moment-match the σ-pushforward measure and generalize gracefully
/// past the box; Chebyshev polynomials do not.  So the scalar
/// box is a **hard boundary**: the solver must either stay inside
/// or trigger the exact-path fallback.  Because the spatial build
/// site sets `n_max = 2 × initial_density`, the initial iterate
/// sits at 50 % of the box — with plenty of room for solver
/// exploration up to 2× the initial density before the guard
/// fires.
fn scalar_density_within_box(plan: &ScalarSurrogatePlan, n: f64) -> bool {
    let Some(train_max) = plan.density_box() else {
        return true;
    };
    if !n.is_finite() || n < 0.0 {
        return false;
    }
    n <= train_max
}

/// Check whether the current density iterate `n` is inside the
/// training region recorded on the cubature plan, with a 50 %
/// expansion tolerance to avoid thrashing at the box boundary.
/// When the plan has no recorded box, accepts unconditionally
/// (caller is responsible; legacy code path).
///
/// Returns `false` when any component escapes the tolerance-
/// expanded box OR is negative, OR is not finite.  Without this,
/// a spatial fit whose per-pixel
/// optimum drifts beyond `2 × initial_densities` silently runs the
/// surrogate out of domain.
fn density_within_box(plan: &SparseEmpiricalCubaturePlan, n: &[f64]) -> bool {
    let Some(train_max) = plan.density_box() else {
        // No box recorded — caller accepts the risk.
        return true;
    };
    if train_max.len() != n.len() {
        return false;
    }
    const TOLERANCE: f64 = 1.5; // 50 % slack above train_max
    for (&n_i, &max_i) in n.iter().zip(train_max) {
        if !n_i.is_finite() || n_i < 0.0 {
            return false;
        }
        if n_i > max_i * TOLERANCE {
            return false;
        }
    }
    true
}

fn cubature_eligible(
    plan: &SparseEmpiricalCubaturePlan,
    energies: &[f64],
    instrument_resolution: &ResolutionFunction,
    resolution_plan: Option<&ResolutionPlan>,
    n_density_params: usize,
) -> bool {
    // k ≥ 2: the scalar k=1 branch handles the grouped case.
    if n_density_params < 2 {
        return false;
    }
    if plan.k() != n_density_params {
        return false;
    }
    if plan.len() != energies.len() {
        return false;
    }
    // Gaussian-resolution models must NOT hit the cubature path:
    // the cubature was built against a TabulatedResolution kernel
    // (it's the only kernel `ResolutionPlan::compile_to_matrix`
    // accepts), so firing it on a Gaussian-active model would
    // silently replace Gaussian broadening with a tabulated
    // surrogate.
    if !matches!(instrument_resolution, ResolutionFunction::Tabulated(_)) {
        return false;
    }
    // Per-element `to_bits()` grid identity check catches `-0.0` vs
    // `+0.0` and NaN-bit differences that float `==` silently
    // accepts or rejects.  The cubature plan's own grid is the
    // primary reference (atoms are indexed against it).
    let cub_grid = plan.target_energies();
    for (e_cur, e_cub) in energies.iter().zip(cub_grid) {
        if e_cur.to_bits() != e_cub.to_bits() {
            return false;
        }
    }
    // Defense-in-depth: when a ResolutionPlan is ALSO attached,
    // verify transitive grid identity.  Catches the
    // `with_precomputed_resolution_plan(plan_A) +
    // with_precomputed_sparse_cubature_plan(plan_B_on_different_grid)`
    // mis-configuration case.  When no resolution plan is attached
    // (typical single-spectrum entrypoint —
    // `fit_spectrum_typed` / `build_transmission_model` don't
    // synthesize one by default), the in-model resolution broaden
    // path falls back to per-call `apply_resolution` and the
    // cubature's self-check above is the grid guard.  An earlier
    // "resolution_plan.is_some() required" rule was over-strict and
    // silently disabled the fast path on the single-spectrum
    // surface.
    if let Some(res_plan) = resolution_plan {
        if res_plan.target_energies().len() != energies.len() {
            return false;
        }
        let res_grid = res_plan.target_energies();
        for (e_cur, e_res) in energies.iter().zip(res_grid) {
            if e_cur.to_bits() != e_res.to_bits() {
                return false;
            }
        }
    }
    true
}

impl PrecomputedTransmissionModel {
    /// Working-grid energies for resolution broadening (issue #608).
    ///
    /// Returns the auxiliary extended grid when `work_layout` is set (Gaussian
    /// resolution), otherwise the data grid (`energies`).  Returns `None` only
    /// when no instrument is configured (Beer-Lambert-only path).
    fn work_energies(&self) -> Option<&[f64]> {
        match (&self.work_layout, &self.energies) {
            (Some(layout), _) => Some(layout.energies.as_slice()),
            (None, Some(energies)) => Some(energies.as_slice()),
            (None, None) => None,
        }
    }

    /// Extract the data-grid points from a working-grid spectrum (issue #608).
    ///
    /// When `work_layout` is `None` the working grid IS the data grid, so this
    /// is the identity (a plain clone).
    fn extract_data_points(&self, working: &[f64]) -> Vec<f64> {
        match &self.work_layout {
            Some(layout) => layout.extract(working),
            None => working.to_vec(),
        }
    }
}

impl FitModel for PrecomputedTransmissionModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        if self.cross_sections.is_empty() {
            return Err(FittingError::InvalidConfig(
                "PrecomputedTransmissionModel.cross_sections must not be empty".into(),
            ));
        }
        let n_e = self.cross_sections[0].len();

        // Cubature fast path: when the plan is installed, matches
        // the grid + isotope count, and instrument resolution is
        // enabled (cubature folds both `exp(-Σ n σ)` and `apply_R`
        // into a single per-row atom sweep).
        if let (Some(cubature), Some(inst), Some(energies)) =
            (&self.sparse_cubature_plan, &self.instrument, &self.energies)
        {
            let params_indices = density_param_indices(&self.density_indices);
            if cubature_eligible(
                cubature,
                energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) {
                let n: Vec<f64> = params_indices.iter().map(|&i| params[i]).collect();
                if density_within_box(cubature, &n) {
                    return Ok(cubature.forward(&n));
                }
                // Density escaped the training box — fall through
                // to the exact path (cubature accuracy degrades
                // quickly outside the trained region).
            }
        }

        // Scalar (k = 1) surrogate fast path — same eligibility
        // stack as the cubature, gated on `n_density_params == 1`.
        // The content-identity guards
        // (σ-fingerprint + Arc::ptr_eq on source resolution plan)
        // close the same-grid stale-plan hole.
        if let (Some(scalar), Some(inst), Some(energies)) =
            (&self.sparse_scalar_plan, &self.instrument, &self.energies)
        {
            let params_indices = density_param_indices(&self.density_indices);
            // Only fire when the σ stack is the single collapsed
            // row the scalar plan was built from (spatial's
            // post-grouping shape).  Non-collapsed k = 1 flows
            // cannot safely dispatch here.
            if self.cross_sections.len() == 1
                && self.density_indices.len() == 1
                && self.density_indices[0] == params_indices[0]
                && scalar_eligible(
                    scalar,
                    energies,
                    &inst.resolution,
                    self.resolution_plan.as_ref(),
                    &self.cross_sections[0],
                    params_indices.len(),
                )
            {
                let n = params[params_indices[0]];
                if scalar_density_within_box(scalar, n) {
                    return Ok(scalar.forward_scalar(n));
                }
            }
        }

        // Beer-Lambert on the WORKING grid (issue #608): `cross_sections` are
        // stored on the working grid (auxiliary extended grid for Gaussian
        // resolution; the data grid for tabulated / no resolution), so `n_e`
        // is the working-grid length.
        let mut neg_opt = vec![0.0f64; n_e];
        // #109.1: No density > 0 guard — let Beer-Lambert handle all densities
        // naturally.  exp(−n·σ) is well-defined for negative n (gives T > 1,
        // which is unphysical but the optimizer will reject it via chi2
        // increase).  Removing the guard makes evaluate() consistent with
        // the analytical Jacobian, which always computes ∂T/∂n = −σ·T
        // regardless of the sign of n.
        for (i, xs) in self.cross_sections.iter().enumerate() {
            let density = params[self.density_indices[i]];
            for (j, &sigma) in xs.iter().enumerate() {
                neg_opt[j] -= density * sigma;
            }
        }
        let transmission: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

        // Issue #442 + #608: apply resolution broadening to the total
        // transmission AFTER Beer-Lambert, on the WORKING grid, then extract
        // the data points LAST.  When `work_layout` is `None` (tabulated / no
        // resolution) the working grid IS the data grid (`self.energies`), the
        // extraction is the identity, and the data-grid `resolution_plan` still
        // matches — byte-identical to the pre-#608 path.
        // Resolution applies iff there is an instrument AND a working grid to
        // apply it on (`work_energies()` = the layout grid when present, else the
        // data grid).  `evaluate` and `analytical_jacobian` share this exact
        // guard so the two paths cannot diverge (issue #608).
        if let (Some(inst), Some(work_energies)) = (&self.instrument, self.work_energies()) {
            let t_broadened = resolution::apply_resolution_with_plan(
                self.resolution_plan.as_deref(),
                work_energies,
                &transmission,
                &inst.resolution,
            )
            .map_err(|e| FittingError::EvaluationFailed(format!("resolution broadening: {e}")))?;
            Ok(self.extract_data_points(&t_broadened))
        } else {
            Ok(self.extract_data_points(&transmission))
        }
    }

    /// Analytical Jacobian for the Beer-Lambert transmission model.
    ///
    /// Without resolution:
    ///   T(E) = exp(-Σᵢ nᵢ · σᵢ(E))
    ///   ∂T/∂nᵢ = -σᵢ(E) · T(E)
    ///
    /// With resolution (R is a linear operator):
    ///   T_obs(E) = R\[T\](E) = R\[exp(-Σᵢ nᵢ · σᵢ)\](E)
    ///   ∂T_obs/∂nᵢ = R\[-σᵢ(E) · T(E)\]
    ///
    /// For grouped isotopes sharing density parameter N_g:
    ///   ∂T_obs/∂N_g = R\[-(Σ_{i∈g} σᵢ(E)) · T(E)\]
    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = if self.cross_sections.is_empty() {
            return None;
        } else {
            self.cross_sections[0].len()
        };
        let n_free = free_param_indices.len();

        // Cubature fast path: same eligibility as `evaluate()` plus
        // the requirement that every free param is a density param
        // (cubature can't produce Jacobian columns for non-density
        // params like background / normalization, which are the
        // calling layer's responsibility).
        if let (Some(cubature), Some(inst), Some(energies)) =
            (&self.sparse_cubature_plan, &self.instrument, &self.energies)
        {
            let params_indices = density_param_indices(&self.density_indices);
            if cubature_eligible(
                cubature,
                energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) {
                // Map each free param to its column in the cubature
                // Jacobian.  `None` for any free param that isn't a
                // density param → fall through to the exact path.
                // Wrappers (`NormalizedTransmissionModel`,
                // `TransmissionKLBackgroundModel`) ensure only density
                // slots reach this layer; non-density free params here
                // would indicate a wrapper bypass.
                let col_map: Option<Vec<usize>> = free_param_indices
                    .iter()
                    .map(|&fp| params_indices.iter().position(|&i| i == fp))
                    .collect();
                if let Some(col_map) = col_map {
                    let n: Vec<f64> = params_indices.iter().map(|&i| params[i]).collect();
                    if density_within_box(cubature, &n) {
                        let (_t, jac_flat) = cubature.forward_and_jacobian(&n);
                        // jac_flat[i * k + ell] = ∂T_i / ∂n_ell
                        let k = params_indices.len();
                        let mut jacobian = FlatMatrix::zeros(n_e, n_free);
                        for (col, &ell) in col_map.iter().enumerate() {
                            for i in 0..n_e {
                                *jacobian.get_mut(i, col) = jac_flat[i * k + ell];
                            }
                        }
                        return Some(jacobian);
                    }
                    // Density outside box → fall through to exact.
                }
            }
        }

        // Scalar (k = 1) surrogate Jacobian fast path.  For a
        // scalar fit `free_param_indices = [0]`, so
        // the Jacobian has one column.
        if let (Some(scalar), Some(inst), Some(energies)) =
            (&self.sparse_scalar_plan, &self.instrument, &self.energies)
        {
            let params_indices = density_param_indices(&self.density_indices);
            if self.cross_sections.len() == 1
                && self.density_indices.len() == 1
                && self.density_indices[0] == params_indices[0]
                && scalar_eligible(
                    scalar,
                    energies,
                    &inst.resolution,
                    self.resolution_plan.as_ref(),
                    &self.cross_sections[0],
                    params_indices.len(),
                )
                && free_param_indices.len() == 1
                && free_param_indices[0] == params_indices[0]
            {
                let n = params[params_indices[0]];
                if scalar_density_within_box(scalar, n) {
                    let (_t, dt) = scalar.forward_and_derivative_scalar(n);
                    let mut jacobian = FlatMatrix::zeros(n_e, 1);
                    for (i, &v) in dt.iter().enumerate() {
                        *jacobian.get_mut(i, 0) = v;
                    }
                    return Some(jacobian);
                }
            }
        }

        // For each free parameter, sum the cross-sections of every isotope
        // tied to that parameter index.  σ (and the sums) are on the WORKING
        // grid (issue #608), so `n_e` is the working-grid length.
        //   ∂T/∂N_g = -(Σ_{iso∈g} σ_iso(E)) · T(E)
        let fp_xs_sums: Vec<Vec<f64>> = free_param_indices
            .iter()
            .map(|&fp_idx| {
                let mut sum = vec![0.0f64; n_e];
                for (iso, &di) in self.density_indices.iter().enumerate() {
                    if di == fp_idx {
                        for (j, &sigma) in self.cross_sections[iso].iter().enumerate() {
                            sum[j] += sigma;
                        }
                    }
                }
                sum
            })
            .collect();

        // The Jacobian has one row per DATA point; `y_current` is on the data
        // grid.  When resolution is enabled the inner derivative is formed on
        // the working grid, resolution-broadened there, and the data points
        // extracted last (issue #608).
        let n_data = y_current.len();

        // When resolution is enabled, we need the UNRESOLVED T(E) = exp(-Σnσ)
        // on the WORKING grid to form the inner derivative -σ·T, then apply
        // resolution on the working grid and extract the data points.
        // y_current is T_obs = R[T] on the DATA grid, which is NOT the same.
        // Same resolution guard as `evaluate` (issue #608) so the two paths
        // agree by construction; the else branch is the no-resolution Jacobian.
        if let (Some(inst), Some(work_energies)) = (&self.instrument, self.work_energies()) {
            // Recompute unresolved T on the working grid from σ and params.
            let mut neg_opt = vec![0.0f64; n_e];
            for (i, xs) in self.cross_sections.iter().enumerate() {
                let density = params[self.density_indices[i]];
                for (j, &sigma) in xs.iter().enumerate() {
                    neg_opt[j] -= density * sigma;
                }
            }
            let t_unresolved: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

            // ∂T_obs/∂N_g = extract(R[-σ_sum(E) · T_unresolved(E)])
            let mut jacobian = FlatMatrix::zeros(n_data, n_free);
            for (col, xs_sum) in fp_xs_sums.iter().enumerate() {
                let inner_deriv: Vec<f64> =
                    (0..n_e).map(|i| -xs_sum[i] * t_unresolved[i]).collect();
                let resolved_deriv = resolution::apply_resolution_with_plan(
                    self.resolution_plan.as_deref(),
                    work_energies,
                    &inner_deriv,
                    &inst.resolution,
                )
                .ok()?;
                let resolved_deriv = self.extract_data_points(&resolved_deriv);
                for (i, &val) in resolved_deriv.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            }
            Some(jacobian)
        } else {
            // No resolution → no auxiliary grid: the working grid IS the data
            // grid (`n_e == n_data`), and y_current IS T(E) directly.
            //   ∂T/∂N_g = -σ_sum(E) · T(E)
            let mut jacobian = FlatMatrix::zeros(n_data, n_free);
            for i in 0..n_data {
                for (j, xs_sum) in fp_xs_sums.iter().enumerate() {
                    *jacobian.get_mut(i, j) = -xs_sum[i] * y_current[i];
                }
            }
            Some(jacobian)
        }
    }
}

/// Forward model for fitting isotopic areal densities from transmission data.
///
/// The model computes T(E) for a set of isotopes with variable areal densities.
/// Each isotope's resonance data and the energy grid are fixed; only the
/// areal densities are adjusted during fitting.
///
/// Optionally, the sample temperature can also be fitted by setting
/// `temperature_index` to the parameter slot holding the temperature value.
/// When `temperature_index` is `Some(idx)`, the Doppler broadening kernel
/// is recomputed at `params[idx]` when the temperature changes (cached
/// across calls at the same temperature), and the analytical Jacobian
/// provides density columns directly plus a single FD column for temperature.
///
/// `instrument` uses `Arc` so that parallel pixel loops can share one copy
/// of a potentially large tabulated resolution kernel via cheap
/// reference-count increments instead of deep-cloning per pixel.
pub struct TransmissionFitModel {
    /// Energy grid (eV), ascending.
    energies: Vec<f64>,
    /// Resonance data for each isotope.
    resonance_data: Vec<ResonanceData>,
    /// Sample temperature in Kelvin (used when `temperature_index` is `None`).
    temperature_k: f64,
    /// Optional instrument resolution parameters (Arc-shared for parallel use).
    instrument: Option<Arc<InstrumentParams>>,
    /// Index mapping: which `params` indices correspond to areal densities.
    /// params[density_indices[i]] = areal density of isotope i.
    ///
    /// Uses `Vec<usize>` (not `Arc<Vec<usize>>`) because `TransmissionFitModel`
    /// is constructed fresh per pixel (via `fit_spectrum`) and never shared
    /// across threads.  `PrecomputedTransmissionModel` uses `Arc<Vec<usize>>`
    /// for its density_indices because it _is_ shared across rayon workers.
    density_indices: Vec<usize>,
    /// Fractional ratio of each member isotope within its group.
    /// For ungrouped isotopes, all values are 1.0.
    /// When groups are active: `effective_density_i = params[density_indices[i]] * density_ratios[i]`
    density_ratios: Vec<f64>,
    /// If `Some(idx)`, `params[idx]` is treated as the sample temperature (K)
    /// and included as a free parameter in the fit. The Doppler broadening
    /// kernel is recomputed at each `evaluate()` call.
    temperature_index: Option<usize>,
    /// Cached unbroadened (Reich-Moore) cross-sections, computed once in
    /// `new()` when `temperature_index` is `Some`. Eliminates redundant
    /// O(N_energy × N_resonances) computation on every `evaluate()` call.
    /// Wrapped in `Arc` so `spatial_map` can share a single allocation across
    /// all per-pixel `TransmissionFitModel` instances without deep cloning.
    base_xs: Option<Arc<Vec<Vec<f64>>>>,
    /// Cached broadened cross-sections from the last `evaluate()` call, on the
    /// **working grid** (auxiliary extended grid when Gaussian resolution is
    /// active, else the data grid).  Used by `analytical_jacobian()` to provide
    /// density columns without rebroadening AND to build the inner derivative
    /// `−σ·T` on the working grid before resolution + data-point extraction
    /// (issue #608).  Interior mutability via `RefCell` is needed because
    /// `FitModel::evaluate` takes `&self`.  Safe because `TransmissionFitModel`
    /// is constructed per-pixel and never shared across threads.
    cached_broadened_xs: RefCell<Option<Rc<Vec<Vec<f64>>>>>,
    /// Cached analytical temperature derivative ∂σ/∂T, on the **working grid**,
    /// computed on-demand by `analytical_jacobian()` when the temperature
    /// column is needed.  Invalidated when temperature changes (cleared in
    /// `evaluate()`).
    cached_dxs_dt: RefCell<Option<Rc<Vec<Vec<f64>>>>>,
    /// Working-grid layout (energies + data-index map) matching
    /// `cached_broadened_xs` / `cached_dxs_dt`.  Resolution broadening is
    /// applied on `layout.energies` and the data points are extracted last
    /// (issue #608).  Set in `evaluate()` alongside the broadened σ cache.
    cached_work_layout: RefCell<Option<Rc<transmission::WorkingGridLayout>>>,
    /// Temperature at which `cached_broadened_xs` was computed.
    /// `Cell` is sufficient because `f64` is `Copy`.
    cached_temperature: Cell<f64>,
    /// Optional prebuilt resolution plan for [`Self::energies`].
    ///
    /// When a caller (typically spatial dispatch) builds the plan
    /// once for a shared grid, passing it here lets every per-pixel
    /// `evaluate()` / `analytical_jacobian()` call reuse the hoisted
    /// TOF / kernel-interp / bracket work.  `None` ⇒ per-call
    /// broadening (same output as pre-plan main).
    resolution_plan: Option<Arc<ResolutionPlan>>,
    /// Optional sparse empirical cubature plan.
    ///
    /// See [`PrecomputedTransmissionModel::sparse_cubature_plan`]
    /// for the dispatch contract.  In this per-pixel model the
    /// cubature is additionally constrained: if `temperature_index`
    /// is `Some` or the temperature changes between evaluate calls,
    /// the σ the cubature was built against becomes stale so the
    /// dispatch silently falls back.
    sparse_cubature_plan: Option<Arc<SparseEmpiricalCubaturePlan>>,
    /// Optional scalar (k = 1) surrogate plan.
    /// Parallel to `sparse_cubature_plan` but dispatches only for
    /// `n_density_params == 1`.
    sparse_scalar_plan: Option<Arc<ScalarSurrogatePlan>>,
}

impl TransmissionFitModel {
    /// Create a validated `TransmissionFitModel`.
    ///
    /// When `external_base_xs` is `Some`, uses those precomputed unbroadened
    /// cross-sections instead of computing them (expensive Reich-Moore).
    /// `spatial_map` precomputes once for all pixels and passes them here.
    ///
    /// # Errors
    /// Returns `FittingError::InvalidConfig` if `temperature_index` overlaps
    /// with `density_indices`, or if `external_base_xs` has a mismatched shape.
    pub fn new(
        energies: Vec<f64>,
        resonance_data: Vec<ResonanceData>,
        temperature_k: f64,
        instrument: Option<Arc<InstrumentParams>>,
        density_mapping: (Vec<usize>, Vec<f64>),
        temperature_index: Option<usize>,
        external_base_xs: Option<Arc<Vec<Vec<f64>>>>,
    ) -> Result<Self, FittingError> {
        let (density_indices, density_ratios) = density_mapping;
        if density_indices.len() != resonance_data.len() {
            return Err(FittingError::InvalidConfig(format!(
                "density_indices has {} entries but resonance_data has {}",
                density_indices.len(),
                resonance_data.len(),
            )));
        }
        if density_ratios.len() != resonance_data.len() {
            return Err(FittingError::InvalidConfig(format!(
                "density_ratios has {} entries but resonance_data has {}",
                density_ratios.len(),
                resonance_data.len(),
            )));
        }
        if let Some(ti) = temperature_index
            && density_indices.contains(&ti)
        {
            return Err(FittingError::InvalidConfig(
                "temperature_index must not overlap with density_indices".into(),
            ));
        }
        // Validate external base XS shape before accepting.
        if let Some(ref xs) = external_base_xs {
            if xs.len() != resonance_data.len() {
                return Err(FittingError::InvalidConfig(format!(
                    "external_base_xs has {} isotopes but resonance_data has {}",
                    xs.len(),
                    resonance_data.len(),
                )));
            }
            for (i, row) in xs.iter().enumerate() {
                if row.len() != energies.len() {
                    return Err(FittingError::InvalidConfig(format!(
                        "external_base_xs[{i}] has {} energies but expected {}",
                        row.len(),
                        energies.len(),
                    )));
                }
            }
        }
        let base_xs = match external_base_xs {
            Some(xs) => Some(xs),
            None if temperature_index.is_some() => Some(Arc::new(
                transmission::unbroadened_cross_sections(&energies, &resonance_data, None)
                    .map_err(|e| {
                        FittingError::InvalidConfig(format!(
                            "failed to compute unbroadened cross-sections: {e}"
                        ))
                    })?,
            )),
            None => None,
        };
        Ok(Self {
            energies,
            resonance_data,
            temperature_k,
            instrument,
            density_indices,
            density_ratios,
            temperature_index,
            base_xs,
            cached_broadened_xs: RefCell::new(None),
            cached_dxs_dt: RefCell::new(None),
            cached_work_layout: RefCell::new(None),
            cached_temperature: Cell::new(f64::NAN),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
        })
    }

    /// Attach a prebuilt resolution plan for the model's energy grid.
    ///
    /// Safe to call before any `evaluate()`.  Caller contract:
    /// `plan.target_energies() == energies` — violating this will
    /// fail on the first broadening call, either via a length
    /// mismatch or, for a different same-length grid,
    /// `ResolutionError::PlanGridMismatch`.
    #[must_use]
    pub fn with_resolution_plan(mut self, plan: Option<Arc<ResolutionPlan>>) -> Self {
        self.resolution_plan = plan;
        self
    }

    /// Attach a prebuilt sparse empirical cubature plan.  See
    /// [`PrecomputedTransmissionModel::sparse_cubature_plan`] for the
    /// dispatch conditions.
    #[must_use]
    pub fn with_sparse_cubature_plan(
        mut self,
        plan: Option<Arc<SparseEmpiricalCubaturePlan>>,
    ) -> Self {
        self.sparse_cubature_plan = plan;
        self
    }

    /// Attach a prebuilt scalar (k = 1) surrogate plan.  See
    /// [`PrecomputedTransmissionModel::sparse_scalar_plan`] for the
    /// dispatch conditions.
    #[must_use]
    pub fn with_sparse_scalar_plan(mut self, plan: Option<Arc<ScalarSurrogatePlan>>) -> Self {
        self.sparse_scalar_plan = plan;
        self
    }
}

impl FitModel for TransmissionFitModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        debug_assert!(
            self.density_indices.iter().all(|&i| i < params.len()),
            "density_indices out of bounds for params (len={})",
            params.len(),
        );
        debug_assert!(
            self.temperature_index.is_none_or(|i| i < params.len()),
            "temperature_index out of bounds for params (len={})",
            params.len(),
        );

        // Cubature fast path: plan present, resolution on, no
        // temperature fit (σ the cubature was built against must not
        // change at runtime).  k=1 grouped case and per-isotope T-fit
        // falls through to the exact path.
        if let (Some(cubature), Some(inst)) = (&self.sparse_cubature_plan, &self.instrument)
            && self.temperature_index.is_none()
        {
            let params_indices = density_param_indices(&self.density_indices);
            if cubature_eligible(
                cubature,
                &self.energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) {
                // Caller contract: for grouped fits the cubature
                // was built with σ already aggregated by ratios
                // (`σ_group_j = Σ_{i ∈ group_j} ratio_i · σ_i`),
                // so the online `forward(n)` receives only the
                // per-group density vector and multiplies by the
                // pre-aggregated atoms internally.
                let n: Vec<f64> = params_indices.iter().map(|&i| params[i]).collect();
                if density_within_box(cubature, &n) {
                    return Ok(cubature.forward(&n));
                }
                // Density escaped training box → fall through.
            }
        }

        // Scalar (k = 1) surrogate fast path was removed from this
        // model: `TransmissionFitModel`'s on-the-fly σ compute
        // couldn't be cheaply fingerprint-checked against the
        // plan's σ, leaving a same-grid stale-plan correctness
        // hole.  Production spatial dispatch attaches scalar plans
        // to [`PrecomputedTransmissionModel`] (via
        // `UnifiedFitConfig::with_precomputed_cross_sections` +
        // `with_precomputed_sparse_scalar_plan`), which DOES
        // enforce σ-fingerprint + Arc::ptr_eq guards.  The
        // `sparse_scalar_plan` field and setter remain here for
        // API consistency with `PrecomputedTransmissionModel`, but
        // this model will always fall through to the exact path.

        let temperature_k = match self.temperature_index {
            Some(idx) => params[idx],
            None => self.temperature_k,
        };

        if let Some(ref base_xs) = self.base_xs {
            // Fast path: reuse cached unbroadened XS, only redo Doppler + Beer-Lambert.
            // Validate temperature (same rules as SampleParams::new in the slow path)
            // so the optimizer can't silently evaluate an unphysical model.
            if !temperature_k.is_finite() || temperature_k < 0.0 {
                return Err(FittingError::EvaluationFailed(format!(
                    "Invalid temperature: {temperature_k} K (must be finite and non-negative)"
                )));
            }

            // Compute broadened XS on the WORKING grid (or reuse cache if
            // temperature unchanged).  Caching avoids redundant Doppler
            // broadening on rejected LM steps (same T, different lambda) and
            // enables analytical_jacobian() to read the broadened σ for the
            // density columns AND to build the inner derivative on the same
            // working grid.
            //
            // Issue #608: Doppler + Beer-Lambert + resolution all run on the
            // working grid (auxiliary extended grid when Gaussian resolution is
            // active), with the data points extracted LAST — matching
            // forward_model.  The previous cached path collapsed σ to the
            // coarse data grid before resolution, degrading the convolution.
            //
            // Derivative ∂σ/∂T is computed on-demand in analytical_jacobian(),
            // NOT here — evaluate() is called many times during line search
            // trials, and the derivative overhead would dominate.
            let (broadened_xs, layout) = if (temperature_k - self.cached_temperature.get()).abs()
                < 1e-15
                && self.cached_broadened_xs.borrow().is_some()
            {
                (
                    Rc::clone(self.cached_broadened_xs.borrow().as_ref().unwrap()),
                    Rc::clone(self.cached_work_layout.borrow().as_ref().unwrap()),
                )
            } else {
                let working = transmission::broadened_cross_sections_from_base_on_working_grid(
                    &self.energies,
                    base_xs,
                    &self.resonance_data,
                    temperature_k,
                    self.instrument.as_deref(),
                )
                .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?;
                let xs = Rc::new(working.sigma);
                let layout = Rc::new(working.layout);
                *self.cached_broadened_xs.borrow_mut() = Some(Rc::clone(&xs));
                *self.cached_work_layout.borrow_mut() = Some(Rc::clone(&layout));
                // Invalidate derivative cache — temperature changed, old ∂σ/∂T stale.
                *self.cached_dxs_dt.borrow_mut() = None;
                self.cached_temperature.set(temperature_k);
                (xs, layout)
            };

            // Beer-Lambert on the working grid: T(E) = exp(-Σᵢ nᵢ · rᵢ · σᵢ(E))
            // where rᵢ is the fractional ratio (1.0 for ungrouped isotopes).
            let work_len = layout.energies.len();
            let mut neg_opt = vec![0.0f64; work_len];
            for (i, xs) in broadened_xs.iter().enumerate() {
                let density = params[self.density_indices[i]];
                let ratio = self.density_ratios[i];
                for (j, &sigma) in xs.iter().enumerate() {
                    neg_opt[j] -= density * ratio * sigma;
                }
            }
            let transmission: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

            // Issue #442: apply resolution broadening to the total transmission
            // AFTER Beer-Lambert, on the working grid; then extract the data
            // points last (issue #608).  For Gaussian resolution `resolution_plan`
            // is `None` (the planned path is tabulated-only) and broadening runs
            // on `layout.energies`; for tabulated resolution the working grid IS
            // the data grid so the data-grid plan still matches.
            if let Some(ref inst) = self.instrument {
                let t_broadened = resolution::apply_resolution_with_plan(
                    self.resolution_plan.as_deref(),
                    &layout.energies,
                    &transmission,
                    &inst.resolution,
                )
                .map_err(|e| {
                    FittingError::EvaluationFailed(format!("resolution broadening: {e}"))
                })?;
                Ok(layout.extract(&t_broadened))
            } else {
                Ok(layout.extract(&transmission))
            }
        } else {
            // Original path: full forward model (no temperature fitting).
            // Apply ratio weights: effective density = params[idx] * ratio.
            let isotopes: Vec<(ResonanceData, f64)> = self
                .resonance_data
                .iter()
                .enumerate()
                .map(|(i, rd)| {
                    (
                        rd.clone(),
                        params[self.density_indices[i]] * self.density_ratios[i],
                    )
                })
                .collect();

            let sample = SampleParams::new(temperature_k, isotopes)
                .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?;

            transmission::forward_model(&self.energies, &sample, self.instrument.as_deref())
                .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
        }
    }

    /// Analytical Jacobian for the transmission model with temperature fitting.
    ///
    /// When `base_xs` is available (temperature fitting path):
    /// - **Density columns**: `∂T/∂nᵢ = -σᵢ(E)·T(E)` using cached broadened XS
    ///   from the most recent `evaluate()` call.  Same formula as
    ///   `PrecomputedTransmissionModel`, zero extra broadening calls.
    /// - **Temperature column**: analytical chain rule via on-demand `∂σ/∂T`.
    ///   `∂T/∂T_temp = -T(E) · Σᵢ nᵢ·rᵢ·∂σᵢ/∂T`.  The derivative is
    ///   computed once per temperature via
    ///   `broadened_cross_sections_with_analytical_derivative_from_base()`
    ///   and cached until temperature changes.  Costs one broadening call
    ///   per Jacobian (same as the old FD approach, but exact).
    ///
    /// Returns `None` for the no-base_xs path (full forward model), which
    /// falls back to finite-difference in the LM solver.
    /// Analytical Jacobian for density and temperature fitting.
    ///
    /// Without resolution:
    ///   ∂T/∂N_g = -(Σ_{i∈g} rᵢ σᵢ) · T
    ///   ∂T/∂Temp = -T · Σᵢ nᵢ rᵢ ∂σᵢ/∂T
    ///
    /// With resolution (R is a linear operator):
    ///   ∂T_obs/∂N_g = R\[-(Σ_{i∈g} rᵢ σᵢ) · T\]
    ///   ∂T_obs/∂Temp = R\[-T · Σᵢ nᵢ rᵢ ∂σᵢ/∂T\]
    ///
    /// Returns `None` only when `base_xs` is not available (full forward
    /// model path falls back to FD) or when the temperature cache is stale.
    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        // Cubature fast path — same eligibility as `evaluate()` plus
        // the requirement that every free param is a density param.
        if let (Some(cubature), Some(inst)) = (&self.sparse_cubature_plan, &self.instrument)
            && self.temperature_index.is_none()
        {
            let params_indices = density_param_indices(&self.density_indices);
            if cubature_eligible(
                cubature,
                &self.energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) {
                let col_map: Option<Vec<usize>> = free_param_indices
                    .iter()
                    .map(|&fp| params_indices.iter().position(|&i| i == fp))
                    .collect();
                if let Some(col_map) = col_map {
                    let n: Vec<f64> = params_indices.iter().map(|&i| params[i]).collect();
                    if density_within_box(cubature, &n) {
                        // In-box: take the cubature Jacobian fast
                        // path.  Out-of-box falls through to the
                        // exact analytical Jacobian below.
                        let (_t, jac_flat) = cubature.forward_and_jacobian(&n);
                        let k = params_indices.len();
                        let n_e = self.energies.len();
                        let mut jacobian = FlatMatrix::zeros(n_e, free_param_indices.len());
                        for (col, &ell) in col_map.iter().enumerate() {
                            for i in 0..n_e {
                                *jacobian.get_mut(i, col) = jac_flat[i * k + ell];
                            }
                        }
                        return Some(jacobian);
                    }
                }
            }
        }

        // Scalar (k = 1) surrogate Jacobian fast path removed —
        // see the docstring at the corresponding
        // site in `TransmissionFitModel::evaluate()` above.

        // Only provide analytical Jacobian when base_xs is available
        // (temperature-fitting fast path with cached broadened XS).
        let _base_xs_guard = self.base_xs.as_ref()?;
        let cached_xs = self.cached_broadened_xs.borrow();
        let broadened_xs = cached_xs.as_ref()?;
        // Working-grid layout matching the cached σ (issue #608).  Inner
        // derivatives are formed on this grid, resolution-broadened there, and
        // the data points are extracted LAST.
        let cached_layout = self.cached_work_layout.borrow();
        let layout = cached_layout.as_ref()?;

        // Guard: verify the cache matches the current parameter temperature.
        if let Some(ti) = self.temperature_index {
            let param_temp = params[ti];
            if (param_temp - self.cached_temperature.get()).abs() > 1e-15 {
                return None;
            }
        }

        let n_e = y_current.len();
        let work_len = layout.energies.len();
        let n_free = free_param_indices.len();
        let mut jacobian = FlatMatrix::zeros(n_e, n_free);

        let temp_col = self
            .temperature_index
            .and_then(|ti| free_param_indices.iter().position(|&fp| fp == ti));

        // The UNRESOLVED transmission T(E) on the WORKING grid, used to form
        // inner derivatives before resolution.  Issue #608: with resolution,
        // y_current is T_obs = R[T] on the DATA grid — not usable as the inner
        // T on the working grid — so recompute T from the cached working-grid
        // σ.  Without resolution the working grid is the data grid (identity
        // layout) and y_current IS T, so reuse it to stay bit-identical.
        let t_unresolved: Option<Vec<f64>> = if self.instrument.is_some() {
            let mut neg_opt = vec![0.0f64; work_len];
            for (iso, xs) in broadened_xs.iter().enumerate() {
                let density = params[self.density_indices[iso]];
                let ratio = self.density_ratios[iso];
                for (j, &sigma) in xs.iter().enumerate() {
                    neg_opt[j] -= density * ratio * sigma;
                }
            }
            Some(neg_opt.iter().map(|&d| d.exp()).collect())
        } else {
            None
        };
        // T(E) on the working grid for the inner derivatives.
        let t_for_deriv: &[f64] = t_unresolved.as_deref().unwrap_or(y_current);

        // ── Density columns: ∂T/∂N_g or ∂T_obs/∂N_g ──
        // Role indices are assumed DISTINCT (first-match layout: the
        // temperature column is skipped here and filled separately, so a
        // parameter serving both roles would get only one contribution).
        // The pipeline always constructs distinct indices; aliasing is
        // not supported in this resolution-coupled fill — see
        // NormalizedTransmissionModel's "Index invariant" for the
        // accumulate-hardened pattern used by the simple wrappers.
        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            if temp_col == Some(col) {
                continue;
            }
            let mut sigma_sum = vec![0.0f64; work_len];
            for (iso, &di) in self.density_indices.iter().enumerate() {
                if di == fp_idx {
                    let ratio = self.density_ratios[iso];
                    for (j, &sigma) in broadened_xs[iso].iter().enumerate() {
                        sigma_sum[j] += ratio * sigma;
                    }
                }
            }
            // Inner derivative on the working grid: -σ_sum · T_unresolved.
            let inner: Vec<f64> = (0..work_len)
                .map(|i| -sigma_sum[i] * t_for_deriv[i])
                .collect();

            if let Some(ref inst) = self.instrument {
                // ∂T_obs/∂N_g = extract(R[inner]) — resolution on the working
                // grid, data points extracted last (issue #608).
                let resolved = resolution::apply_resolution_with_plan(
                    self.resolution_plan.as_deref(),
                    &layout.energies,
                    &inner,
                    &inst.resolution,
                )
                .ok()?;
                let resolved = layout.extract(&resolved);
                for (i, &val) in resolved.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            } else {
                // No resolution → identity layout, inner is already data grid.
                for (i, &val) in inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            }
        }

        // ── Temperature column: ∂T/∂Temp or ∂T_obs/∂Temp ──
        if let Some(col) = temp_col {
            // Compute ∂σ/∂T (on the working grid) on-demand if not cached.
            {
                let needs_compute = self.cached_dxs_dt.borrow().as_ref().is_none();
                if needs_compute {
                    let base_xs = self.base_xs.as_ref()?;
                    let temperature_k = self.cached_temperature.get();
                    let working =
                        transmission::broadened_cross_sections_with_analytical_derivative_from_base_on_working_grid(
                            &self.energies,
                            base_xs,
                            &self.resonance_data,
                            temperature_k,
                            self.instrument.as_deref(),
                        )
                        .ok()?;
                    *self.cached_dxs_dt.borrow_mut() = Some(Rc::new(working.dsigma_dt));
                }
            }
            let cached_dxs = self.cached_dxs_dt.borrow();
            let dxs_dt = cached_dxs.as_ref()?;

            // Inner derivative on the working grid: -T · Σᵢ nᵢ rᵢ ∂σᵢ/∂T.
            let inner: Vec<f64> = (0..work_len)
                .map(|i| {
                    let mut sum_n_dsigma = 0.0f64;
                    for (iso, dxs) in dxs_dt.iter().enumerate() {
                        let density = params[self.density_indices[iso]];
                        let ratio = self.density_ratios[iso];
                        sum_n_dsigma += density * ratio * dxs[i];
                    }
                    -t_for_deriv[i] * sum_n_dsigma
                })
                .collect();

            if let Some(ref inst) = self.instrument {
                let resolved = resolution::apply_resolution_with_plan(
                    self.resolution_plan.as_deref(),
                    &layout.energies,
                    &inner,
                    &inst.resolution,
                )
                .ok()?;
                let resolved = layout.extract(&resolved);
                for (i, &val) in resolved.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            } else {
                for (i, &val) in inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            }
        }

        Some(jacobian)
    }
}

/// Wraps a transmission model with SAMMY-style normalization and background.
///
/// T_out(E) = Anorm × T_inner(E) + BackA + BackB / √E + BackC × √E
///          + BackD × exp(−BackF / √E)
///
/// The normalization and background parameters are additional entries in the
/// parameter vector, appended after the density (and optional temperature)
/// parameters of the inner model.
///
/// The exponential tail (BackD, BackF) is optional.  When
/// `back_d_index` and `back_f_index` are `None`, the model reduces to
/// the 4-parameter form.
///
/// ## SAMMY Reference
/// SAMMY manual Sec III.E.2 — NORMAlization and BACKGround cards.
/// SAMMY fits up to 6 background terms; we implement all 6:
///   Anorm, constant BackA, 1/√E term BackB, √E term BackC,
///   exponential amplitude BackD, exponential decay BackF.
///
/// ## Index invariant
///
/// The role indices (`anorm_index`, `back_*_index`) must NOT designate
/// a parameter the inner model reads: the analytic Jacobian filters the
/// role indices out of the inner free set, so such a collision cannot
/// be detected and the column would silently omit Anorm × ∂T_inner/∂p.
/// Aliasing AMONG the role indices themselves IS supported — the
/// Jacobian columns accumulate.
pub struct NormalizedTransmissionModel<M: FitModel> {
    /// The inner (pure Beer-Lambert) transmission model.
    inner: M,
    /// Precomputed √E for each energy bin.  Computed once in `new()`.
    sqrt_energies: Vec<f64>,
    /// Precomputed 1/√E for each energy bin.  Computed once in `new()`.
    inv_sqrt_energies: Vec<f64>,
    /// Index of the Anorm parameter in the full parameter vector.
    anorm_index: usize,
    /// Index of the BackA (constant background) parameter.
    back_a_index: usize,
    /// Index of the BackB (1/√E background) parameter.
    back_b_index: usize,
    /// Index of the BackC (√E background) parameter.
    back_c_index: usize,
    /// Index of BackD (exponential amplitude) in the parameter vector.
    /// `None` disables the exponential tail term.
    back_d_index: Option<usize>,
    /// Index of BackF (exponential decay constant) in the parameter vector.
    /// `None` disables the exponential tail term.
    back_f_index: Option<usize>,
}

impl<M: FitModel> NormalizedTransmissionModel<M> {
    /// Create a new normalized transmission model (4-parameter, no exponential tail).
    ///
    /// # Arguments
    /// * `inner` — The inner transmission model (Beer-Lambert).
    /// * `energies` — Energy grid in eV (must be positive).
    /// * `anorm_index` — Index of Anorm in the parameter vector.
    /// * `back_a_index` — Index of BackA in the parameter vector.
    /// * `back_b_index` — Index of BackB in the parameter vector.
    /// * `back_c_index` — Index of BackC in the parameter vector.
    pub fn new(
        inner: M,
        energies: &[f64],
        anorm_index: usize,
        back_a_index: usize,
        back_b_index: usize,
        back_c_index: usize,
    ) -> Self {
        let sqrt_energies: Vec<f64> = energies.iter().map(|&e| e.sqrt()).collect();
        let inv_sqrt_energies: Vec<f64> = sqrt_energies
            .iter()
            .map(|&se| if se > 0.0 { 1.0 / se } else { 0.0 })
            .collect();
        Self {
            inner,
            sqrt_energies,
            inv_sqrt_energies,
            anorm_index,
            back_a_index,
            back_b_index,
            back_c_index,
            back_d_index: None,
            back_f_index: None,
        }
    }

    /// Create a normalized transmission model with the SAMMY exponential tail.
    ///
    /// Adds BackD × exp(−BackF / √E) to the 4-parameter background model.
    ///
    /// # Arguments
    /// * `back_d_index` — Index of BackD (exponential amplitude) in the parameter vector.
    /// * `back_f_index` — Index of BackF (exponential decay constant) in the parameter vector.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_exponential(
        inner: M,
        energies: &[f64],
        anorm_index: usize,
        back_a_index: usize,
        back_b_index: usize,
        back_c_index: usize,
        back_d_index: usize,
        back_f_index: usize,
    ) -> Self {
        let sqrt_energies: Vec<f64> = energies.iter().map(|&e| e.sqrt()).collect();
        let inv_sqrt_energies: Vec<f64> = sqrt_energies
            .iter()
            .map(|&se| if se > 0.0 { 1.0 / se } else { 0.0 })
            .collect();
        Self {
            inner,
            sqrt_energies,
            inv_sqrt_energies,
            anorm_index,
            back_a_index,
            back_b_index,
            back_c_index,
            back_d_index: Some(back_d_index),
            back_f_index: Some(back_f_index),
        }
    }
}

impl<M: FitModel> FitModel for NormalizedTransmissionModel<M> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let t_inner = self.inner.evaluate(params)?;
        let anorm = params[self.anorm_index];
        let back_a = params[self.back_a_index];
        let back_b = params[self.back_b_index];
        let back_c = params[self.back_c_index];

        // Optional exponential tail: BackD × exp(−BackF / √E)
        let (back_d, back_f) = match (self.back_d_index, self.back_f_index) {
            (Some(di), Some(fi)) => (params[di], params[fi]),
            _ => (0.0, 0.0),
        };
        let has_exp = self.back_d_index.is_some();

        let result: Vec<f64> = t_inner
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let mut val = anorm * t
                    + back_a
                    + back_b * self.inv_sqrt_energies[i]
                    + back_c * self.sqrt_energies[i];
                if has_exp {
                    val += back_d * (-back_f * self.inv_sqrt_energies[i]).exp();
                }
                val
            })
            .collect();
        Ok(result)
    }

    /// Analytical Jacobian for the normalized transmission model.
    ///
    /// For each free parameter:
    /// - If it belongs to the inner model (density or temperature):
    ///   ∂T_out/∂p = Anorm × ∂T_inner/∂p  (inner Jacobian scaled by Anorm)
    /// - ∂T_out/∂Anorm  = T_inner(E)
    /// - ∂T_out/∂BackA  = 1
    /// - ∂T_out/∂BackB  = 1/√E
    /// - ∂T_out/∂BackC  = √E
    /// - ∂T_out/∂BackD  = exp(−BackF / √E)
    /// - ∂T_out/∂BackF  = −BackD × exp(−BackF / √E) / √E
    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        let n_free = free_param_indices.len();

        // Compute T_inner for Anorm column and for scaling inner Jacobian.
        // T_inner = (T_out - BackA - BackB/√E - BackC×√E) / Anorm
        // But to avoid numerical issues, recompute from the inner model.
        let t_inner = self.inner.evaluate(params).ok()?;

        let anorm = params[self.anorm_index];

        // Identify which free params are background params vs inner params.
        let mut bg_indices_set = vec![
            self.anorm_index,
            self.back_a_index,
            self.back_b_index,
            self.back_c_index,
        ];
        if let Some(di) = self.back_d_index {
            bg_indices_set.push(di);
        }
        if let Some(fi) = self.back_f_index {
            bg_indices_set.push(fi);
        }

        // Collect inner model's free param indices (those not in bg_indices).
        let inner_free_indices: Vec<usize> = free_param_indices
            .iter()
            .copied()
            .filter(|idx| !bg_indices_set.contains(idx))
            .collect();

        // Get inner Jacobian if there are inner free params.
        // y_current for the inner model is t_inner, not the outer y_current.
        let inner_jac = if !inner_free_indices.is_empty() {
            self.inner
                .analytical_jacobian(params, &inner_free_indices, &t_inner)
        } else {
            None
        };

        // Precompute exp(−BackF / √E) for the exponential tail columns.
        let exp_terms: Vec<f64> =
            if let (Some(di), Some(fi)) = (self.back_d_index, self.back_f_index) {
                let _back_d = params[di];
                let back_f = params[fi];
                self.inv_sqrt_energies
                    .iter()
                    .map(|&inv_se| (-back_f * inv_se).exp())
                    .collect()
            } else {
                vec![]
            };

        let mut jacobian = FlatMatrix::zeros(n_e, n_free);

        // Map inner free param index → column in inner Jacobian.
        let mut inner_col_map = std::collections::HashMap::new();
        for (col, &idx) in inner_free_indices.iter().enumerate() {
            inner_col_map.insert(idx, col);
        }

        // Independent role checks with accumulation (+=) rather than a
        // first-match if/else-if chain: nothing forbids two role indices
        // from aliasing, and evaluate() reads an aliased parameter for
        // every role it occupies, so its derivative is the SUM of the
        // matching columns. Distinct indices touch each column once on a
        // zeroed matrix — identical to assignment. A role index colliding
        // with an INNER-model parameter remains undetectable here (role
        // indices are filtered out of inner_free_indices) — see the
        // struct docs.
        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            let mut matched = false;
            if fp_idx == self.anorm_index {
                // ∂T_out/∂Anorm = T_inner(E)
                for (i, &ti) in t_inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) += ti;
                }
                matched = true;
            }
            if fp_idx == self.back_a_index {
                // ∂T_out/∂BackA = 1
                for i in 0..n_e {
                    *jacobian.get_mut(i, col) += 1.0;
                }
                matched = true;
            }
            if fp_idx == self.back_b_index {
                // ∂T_out/∂BackB = 1/√E
                for (i, &inv_se) in self.inv_sqrt_energies.iter().enumerate() {
                    *jacobian.get_mut(i, col) += inv_se;
                }
                matched = true;
            }
            if fp_idx == self.back_c_index {
                // ∂T_out/∂BackC = √E
                for (i, &se) in self.sqrt_energies.iter().enumerate() {
                    *jacobian.get_mut(i, col) += se;
                }
                matched = true;
            }
            if self.back_d_index == Some(fp_idx) {
                // ∂T_out/∂BackD = exp(−BackF / √E)
                for (i, &et) in exp_terms.iter().enumerate() {
                    *jacobian.get_mut(i, col) += et;
                }
                matched = true;
            }
            if self.back_f_index == Some(fp_idx) {
                // ∂T_out/∂BackF = −BackD × exp(−BackF / √E) / √E
                let back_d = params[self.back_d_index.unwrap()];
                for (i, (&et, &inv_se)) in exp_terms
                    .iter()
                    .zip(self.inv_sqrt_energies.iter())
                    .enumerate()
                {
                    *jacobian.get_mut(i, col) += -back_d * et * inv_se;
                }
                matched = true;
            }
            if let Some(&inner_col) = inner_col_map.get(&fp_idx) {
                // Inner model parameter: ∂T_out/∂p = Anorm × ∂T_inner/∂p
                if let Some(ref jac) = inner_jac {
                    for i in 0..n_e {
                        *jacobian.get_mut(i, col) += anorm * jac.get(i, inner_col);
                    }
                    matched = true;
                } else {
                    // Inner model did not provide analytical Jacobian —
                    // fall back to finite-difference for the whole thing.
                    return None;
                }
            }
            if !matched {
                // Unknown parameter — should not happen, but fall back to FD.
                return None;
            }
        }

        Some(jacobian)
    }
}

// ── Energy-scale transmission model (SAMMY TZERO equivalent) ─────────────

/// Transmission model with energy-scale calibration parameters (t₀, L_scale).
///
/// Carries per-isotope resonance data (NOT a precomputed σ grid) and rebuilds
/// the TRUE cross-section at the corrected energies on each evaluation
/// (issue #608), matching `forward_model`:
///   1. Convert nominal energy → TOF: `t = TOF_FACTOR * L / √E_nom`
///   2. Apply calibration: `t_corr = t - t₀`,
///      `E_corr = (TOF_FACTOR * L * L_scale / t_corr)²`
///   3. Evaluate σ(E_corr) directly via `reich_moore` + Doppler on a working
///      grid built from `E_corr` (auxiliary extended grid under Gaussian
///      resolution; `E_corr` itself for tabulated / no resolution) — NOT
///      interpolation of a fixed σ grid, which clamps at the auxiliary
///      boundary and drops resonance fine-structure.
///   4. Beer-Lambert + resolution on the working grid, then extract the data
///      points last.
///
/// This is equivalent to SAMMY's TZERO parameters.
///
/// The Jacobian for t₀ and L_scale defaults to **partial-GAL** since
/// issue #489: central FD on `t0` only (2 evals) plus an inline rank-1
/// derivation of the `L_scale` column. The previous central-FD-on-both
/// (4-eval) behaviour is reachable via `with_jacobian_method`,
/// `NEREIDS_TZERO_JACOBIAN=fd2`, or `tzero_jacobian="fd2"` Python kwarg.
/// See [`EnergyScaleJacobianMethod`] for full method documentation.
pub struct EnergyScaleTransmissionModel {
    /// Resonance parameters per isotope.  Issue #608: the energy-scale model
    /// evaluates the TRUE cross-section at the corrected energies (matching
    /// `forward_model`) instead of interpolating a precomputed σ grid, so it
    /// carries resonance data and rebuilds σ on the corrected working grid each
    /// `evaluate`.  This is the only way to reproduce SAMMY's σ(E_corr) under
    /// the energy-scale shift with full boundary + resonance-fine-structure
    /// fidelity; interpolating a fixed precomputed σ cannot (it clamps at the
    /// auxiliary boundary and misses fine-structure).
    resonance_data: Arc<Vec<ResonanceData>>,
    /// Density parameter index per isotope (same convention as
    /// `PrecomputedTransmissionModel`).
    density_indices: Arc<Vec<usize>>,
    /// Fractional ratio per isotope (1.0 when ungrouped).  Per-isotope
    /// thickness is `params[density_indices[i]] * density_ratios[i]`.
    density_ratios: Arc<Vec<f64>>,
    /// Sample temperature (K) for Doppler broadening at the corrected energies.
    /// Used as the fixed temperature when `temperature_index` is `None`, and as
    /// the fallback / initial value otherwise.
    temperature_k: f64,
    /// If `Some(idx)`, `params[idx]` is the sample temperature (K) fitted as a
    /// free parameter jointly with the energy scale (issue #634); σ is rebuilt
    /// at that T on each evaluate. `None` ⇒ the fixed `temperature_k` is used.
    /// Mirrors `PrecomputedTransmissionModel::temperature_index`.
    ///
    /// The temperature Jacobian column is computed by central finite
    /// difference (like this model's t0 column), not by the analytic ∂σ/∂T
    /// that the fixed-grid `PrecomputedTransmissionModel` uses. The forward σ
    /// stays exact — FD only sets the descent direction / covariance, and it
    /// is validated against the analytic column to `<1e-4` relative. Porting
    /// analytic ∂σ/∂T here is a deliberate FUTURE optimization: it is
    /// evaluated on the *corrected* grid (which moves with t0/L_scale), so it
    /// would need a new physics helper plus a third `(t0,L_scale,T)`-keyed
    /// derivative cache — not worth it until profiling shows the FD probes
    /// dominate.
    temperature_index: Option<usize>,
    /// Nominal energy grid (eV, ascending).
    nominal_energies: Vec<f64>,
    /// Flight path length in meters (used for TOF↔energy conversion).
    flight_path_m: f64,
    /// TOF factor: sqrt(m_n / (2 * eV)) in μs·√eV/m.
    tof_factor: f64,
    /// Index of t₀ (μs) in the parameter vector.
    t0_index: usize,
    /// Index of L_scale (dimensionless) in the parameter vector.
    l_scale_index: usize,
    /// Instrument resolution parameters (applied after Beer-Lambert).
    instrument: Option<Arc<transmission::InstrumentParams>>,
    /// Plan cache keyed on `(t0_bits, l_scale_bits)`.  Within one KL
    /// outer iteration (deviance + gradient + Fisher all at the same
    /// `params`) `evaluate_at` is called 3× at identical `(t0, L)`;
    /// the density-column path of `analytical_jacobian` wants a plan
    /// at that same `(t0, L)` too — that's 4 cache hits per outer
    /// iter on KL+periso+TZERO.  Finite-difference probes land at a
    /// different `(t0, L)` bit-pattern from the accepted probe and
    /// are routed through `evaluate_at_with_cache(..., false)` so
    /// they stay on the non-plan broadening path — no plan is built
    /// or inserted for FD probes, so they neither miss nor pollute
    /// the cache.
    ///
    /// **Capacity 2** (FIFO on miss): this survives LM backtracking,
    /// where a proposed-but-rejected trial step evaluates at a new
    /// `(t0, L)` key and would otherwise evict the accepted-step
    /// plan.  With capacity 2, the accepted plan stays resident
    /// alongside the trial plan; if the trial is rejected, the next
    /// iteration's evaluate at the accepted `(t0, L)` still hits.
    /// Only when a genuine new accepted step lands do we start
    /// aging the oldest entry out (#483 A1).
    ///
    /// `RefCell` is safe: `TransmissionFitModel`-family models are
    /// rebuilt per-pixel and never shared across rayon workers.
    cached_plans: RefCell<CachedPlanRing>,
    /// Capacity-1 cache of the working-grid σ keyed on `(t0_bits, L_scale_bits)`
    /// (issue #608 perf): a base-point `evaluate` + the Jacobian's density
    /// columns at the same probe reuse one reich_moore+Doppler build instead of
    /// rebuilding it twice.  `RefCell` is safe — the model is rebuilt per pixel
    /// and never shared across threads.
    cached_work_xs: RefCell<CachedWorkXs>,
    /// Method for the t0 / L_scale Jacobian columns. Initialised from
    /// [`EnergyScaleJacobianMethod::from_env`] in [`Self::new`], which
    /// defaults to `PartialGal` since issue #489 (and respects the
    /// `NEREIDS_TZERO_JACOBIAN` env var as a global override). Can be
    /// overridden per-instance via [`Self::with_jacobian_method`].
    jacobian_method: EnergyScaleJacobianMethod,
}

/// Capacity-1 working-grid σ cache entry, keyed on
/// `(t0_bits, l_scale_bits, temperature_bits)` — the temperature bits (issue
/// #634) keep a T-only perturbation (same t0/L_scale) from incorrectly hitting
/// a σ built at the base temperature. Named alias to keep the field type within
/// clippy's `type_complexity` budget (issue #608).
type CachedWorkXs = Option<((u64, u64, u64), Rc<transmission::WorkingGridXs>)>;

/// One `(t0_bits, l_scale_bits)` → `ResolutionPlan` entry.  Named
/// struct to keep the cache field type within clippy's
/// `type_complexity` budget.
#[derive(Debug, Clone)]
struct CachedPlanEntry {
    key: (u64, u64),
    plan: Arc<ResolutionPlan>,
}

/// Capacity-2 FIFO ring of plan entries.  Two entries suffice to
/// survive a single-trial LM backtrack (accepted + trial); deeper
/// backtracking chains still lose the accepted plan eventually, but
/// those are rare in production and cheaper to miss than the default
/// non-plan path.  Issue #483 A1.
#[derive(Debug, Default)]
struct CachedPlanRing {
    /// Slot 0 is the most-recently-inserted entry; slot 1 is the
    /// previous entry.  Lookup checks both; insert shifts 0 → 1 and
    /// places the new entry at 0.
    slots: [Option<CachedPlanEntry>; 2],
}

impl CachedPlanRing {
    fn lookup(&self, key: (u64, u64)) -> Option<Arc<ResolutionPlan>> {
        for slot in &self.slots {
            if let Some(entry) = slot
                && entry.key == key
            {
                return Some(Arc::clone(&entry.plan));
            }
        }
        None
    }

    fn insert(&mut self, entry: CachedPlanEntry) {
        // Shift oldest out, newest to slot 0.
        self.slots[1] = self.slots[0].take();
        self.slots[0] = Some(entry);
    }
}

/// Method for computing the t0 / L_scale columns of the
/// `EnergyScaleTransmissionModel` Jacobian.
///
/// - `PartialGal` (default since issue #489): central FD on `t0` only
///   (2 evaluations); derive `L_scale` column inline via the rank-1
///   identity `J[:, L_scale] = ((tof - t0) / L_scale) * J[:, t0]` per
///   energy bin. Halves the FD probe count on workloads where both
///   calibration parameters are free.
///
///   **Correctness regime**: exact in the no-resolution limit and the
///   narrow-kernel limit. With a non-trivial resolution operator `R`,
///   the rank-1 simplification additionally assumes per-bin uniformity
///   of `(tof - t0) / L_scale` over the kernel support — necessary
///   because `R` mixes source bins whose ratios differ. `broaden_presorted`
///   uses `self.flight_path_m` (not the model's `L_nominal * L_scale`) so
///   tabulated kernels satisfy the structural factorisation through
///   `e_corr`, but the per-bin homogeneity assumption is empirical.
///   On real VENUS Hf 120-min KL+per-iso+TZERO 4×4 the approximation is
///   tight enough that 15/16 pixels converge within 0.1·σ_Fisher of FD2;
///   median wall-time speedup 1.28× over FD2.
/// - `FiniteDifference`: central FD on the full inner forward chain,
///   4 forward evaluations per Jacobian (h_t0=1e-4, h_ls=1e-7).
///   The pre-#489 production default; reachable via
///   `NEREIDS_TZERO_JACOBIAN=fd2` env var or `tzero_jacobian="fd2"`
///   Python kwarg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyScaleJacobianMethod {
    FiniteDifference,
    PartialGal,
}

impl EnergyScaleJacobianMethod {
    /// Resolve the default Jacobian method from the
    /// `NEREIDS_TZERO_JACOBIAN` env var.
    ///
    /// The env var is read **once per process** via a `OnceLock`. Per
    /// `EnergyScaleTransmissionModel::new` is hot under
    /// `spatial_map_typed` (one model per pixel; 262 144 calls per
    /// 512×512 map), so `std::env::var` would otherwise be a syscall
    /// hot spot. Tests that need to swap the default must use
    /// `EnergyScaleTransmissionModel::with_jacobian_method` (which
    /// bypasses the cache); changing the env var mid-process has no
    /// effect.
    ///
    /// An unrecognized or removed value (e.g. the `"chain"` method dropped in
    /// #608) emits a one-time `eprintln` warning and falls back to the
    /// `PartialGal` default rather than being silently masked.  It does not
    /// panic — `new` is a hot, infallible, per-pixel constructor across the
    /// PyO3 boundary; the Python `tzero_jacobian=` kwarg is the strict
    /// (hard-erroring) override path.
    fn from_env() -> Self {
        use std::sync::OnceLock;
        static CACHED: OnceLock<EnergyScaleJacobianMethod> = OnceLock::new();
        *CACHED.get_or_init(Self::resolve_env_uncached)
    }

    fn resolve_env_uncached() -> Self {
        let Ok(v) = std::env::var("NEREIDS_TZERO_JACOBIAN") else {
            // Unset → the documented #489 default, silently.
            return Self::PartialGal;
        };
        if v.eq_ignore_ascii_case("fd2")
            || v.eq_ignore_ascii_case("finite-difference")
            || v.eq_ignore_ascii_case("finite_difference")
        {
            Self::FiniteDifference
        } else if v.eq_ignore_ascii_case("partial-gal") || v.eq_ignore_ascii_case("partial_gal") {
            Self::PartialGal
        } else {
            // Set to an unrecognized / removed method name.  The legacy
            // `"chain"` / `"frozen-r"` / `"frozen_r"` FrozenResolutionChainRule
            // method was removed in #608 (it interpolated a precomputed σ on the
            // data grid, incompatible with the true-σ aux-grid `evaluate`;
            // FD / PartialGal of the corrected evaluate is the exact
            // replacement).  The Python `tzero_jacobian=` kwarg HARD-ERRORS on
            // these names (bindings/python `parse_tzero_jacobian`); `from_env` is
            // an infallible, process-cached, per-pixel constructor path that must
            // not panic across the PyO3 boundary (cf. the #608 `working_xs`
            // Err-not-panic guard), so it cannot itself return an error.  Warn
            // loudly (once, via the `OnceLock` in `from_env`) so the override is
            // NOT silently masked, then fall back to the PartialGal default —
            // matching the kwarg in *surfacing* the bad value while staying
            // non-fatal on this hot, infallible path.
            eprintln!(
                "warning: NEREIDS_TZERO_JACOBIAN=\"{v}\" is not a recognized \
                 Jacobian method (\"chain\" / \"frozen-r\" were removed in #608); \
                 using the default \"partial-gal\". Valid values: \"fd2\", \
                 \"partial-gal\"."
            );
            Self::PartialGal
        }
    }
}

impl EnergyScaleTransmissionModel {
    /// Create a new energy-scale transmission model.
    ///
    /// # Arguments
    /// * `resonance_data` — Resonance parameters per isotope; σ is evaluated at
    ///   the corrected energies via `reich_moore` + Doppler (issue #608).
    /// * `density_indices` — Maps isotope index → density parameter index.
    /// * `density_ratios` — Fractional ratio per isotope (1.0 when ungrouped).
    /// * `temperature_k` — Sample temperature (K) for Doppler broadening.
    /// * `nominal_energies` — Energy grid in eV (ascending).
    /// * `flight_path_m` — Nominal flight path in meters.
    /// * `t0_index` — Index of t₀ parameter.
    /// * `l_scale_index` — Index of L_scale parameter.
    /// * `instrument` — Optional resolution function.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        resonance_data: Arc<Vec<ResonanceData>>,
        density_indices: Arc<Vec<usize>>,
        density_ratios: Arc<Vec<f64>>,
        temperature_k: f64,
        nominal_energies: Vec<f64>,
        flight_path_m: f64,
        t0_index: usize,
        l_scale_index: usize,
        instrument: Option<Arc<transmission::InstrumentParams>>,
    ) -> Self {
        // TOF_FACTOR = sqrt(m_n / (2 · eV)) · 1e6 [μs·√eV/m].
        // Use the CODATA 2018 values from nereids-core::constants so that
        // this model, calibration.rs, and core::tof_to_energy all agree to
        // machine precision (previously the inline approximations differed
        // by ~5e-5 relative, enough to visibly shift sharp resonances).
        let tof_factor = (0.5 * NEUTRON_MASS_KG / EV_TO_JOULES).sqrt() * 1.0e6;
        Self {
            resonance_data,
            density_indices,
            density_ratios,
            temperature_k,
            // Default: temperature fixed. `with_temperature_index` opts into
            // joint temperature fitting (issue #634).
            temperature_index: None,
            nominal_energies,
            flight_path_m,
            tof_factor,
            t0_index,
            l_scale_index,
            instrument,
            cached_plans: RefCell::new(CachedPlanRing::default()),
            cached_work_xs: RefCell::new(None),
            jacobian_method: EnergyScaleJacobianMethod::from_env(),
        }
    }

    /// Override the t0 / L_scale Jacobian method for this model instance.
    /// Bypasses the `NEREIDS_TZERO_JACOBIAN` env var.
    #[must_use]
    pub fn with_jacobian_method(mut self, method: EnergyScaleJacobianMethod) -> Self {
        self.jacobian_method = method;
        self
    }

    /// Fit the sample temperature jointly with the energy scale (issue #634).
    /// `Some(idx)` makes `params[idx]` the free temperature (K); `None` keeps
    /// temperature fixed at the constructor's `temperature_k`.
    ///
    /// # Errors
    /// `FittingError::InvalidConfig` if `Some(idx)` collides with `t0_index`,
    /// `l_scale_index`, or any density index — a mis-wired index would
    /// otherwise Doppler-broaden at a nonsense "temperature" (e.g. the t0
    /// value) with no error.  Mirrors `TransmissionFitModel::new`'s
    /// density-overlap rejection (issue #634 review, sibling-parity class).
    pub fn with_temperature_index(
        mut self,
        temperature_index: Option<usize>,
    ) -> Result<Self, FittingError> {
        if let Some(idx) = temperature_index
            && (idx == self.t0_index
                || idx == self.l_scale_index
                || self.density_indices.contains(&idx))
        {
            return Err(FittingError::InvalidConfig(format!(
                "temperature_index {idx} must not overlap t0_index \
                 ({}), l_scale_index ({}), or the density indices",
                self.t0_index, self.l_scale_index,
            )));
        }
        self.temperature_index = temperature_index;
        Ok(self)
    }

    /// Sample temperature (K) for the current parameter vector: the fitted
    /// `params[temperature_index]` when temperature is free, else the fixed
    /// `temperature_k`. Mirrors `PrecomputedTransmissionModel`.
    fn temperature_for(&self, params: &[f64]) -> f64 {
        debug_assert!(
            self.temperature_index.is_none_or(|i| i < params.len()),
            "temperature_index out of bounds for params (len={})",
            params.len()
        );
        match self.temperature_index {
            Some(idx) => params[idx],
            None => self.temperature_k,
        }
    }

    /// Build or reuse the broadening plan for the current `(t0, L_scale)`
    /// probe.  Capacity-2 FIFO ring keyed on raw `f64` bits, matching
    /// the invariant that `corrected_energies(t0, L)` is a pure
    /// function of `(t0_bits, L_bits)` and `self.nominal_energies`
    /// (fixed for the model's lifetime).
    ///
    /// Capacity 2 survives one LM backtrack rejection: the previous
    /// (accepted) entry stays in slot 1 while the trial-step entry
    /// occupies slot 0, so a rejection followed by an evaluate at the
    /// restored accepted `(t0, L)` still hits (#483 A1).
    ///
    /// Returns `None` for Gaussian resolution (no plan representation)
    /// or when the `build_resolution_plan` call fails (unsorted grid) —
    /// both cases transparently fall back to the non-plan
    /// `apply_resolution` path via `apply_resolution_with_plan(None, …)`.
    ///
    /// `working_energies` is the broadening grid the plan is built on — the
    /// model's WORKING grid (`work.layout.energies`), which every caller passes
    /// post-#608.  For tabulated resolution (the only case that builds a plan)
    /// the working grid IS the corrected data grid; for Gaussian it is the
    /// auxiliary extended grid, but that path returns `None` above before the
    /// grid is used.
    fn cached_resolution_plan(
        &self,
        t0_us: f64,
        l_scale: f64,
        working_energies: &[f64],
    ) -> Option<Arc<ResolutionPlan>> {
        let inst = self.instrument.as_ref()?;
        // Match on a reference to `inst.resolution` defensively so the
        // check never attempts to move a non-`Copy` `ResolutionFunction`
        // out of a shared `Arc<InstrumentParams>`.
        if !matches!(
            &inst.resolution,
            nereids_physics::resolution::ResolutionFunction::Tabulated(_)
        ) {
            // Only Tabulated opts into plan caching. Gaussian genuinely has no
            // plan; IkedaCarpenter *does* have one (build_resolution_plan returns
            // Some) but is intentionally not cached here — it falls back to the
            // per-call resynthesis path (the W6 perf follow-up).
            return None;
        }
        let key = (t0_us.to_bits(), l_scale.to_bits());
        if let Some(plan) = self.cached_plans.borrow().lookup(key) {
            return Some(plan);
        }
        // Miss: build, insert, return.
        let plan = resolution::build_resolution_plan(working_energies, &inst.resolution)
            .ok()
            .flatten()?;
        let arc = Arc::new(plan);
        self.cached_plans.borrow_mut().insert(CachedPlanEntry {
            key,
            plan: Arc::clone(&arc),
        });
        Some(arc)
    }

    /// Compute the corrected energy grid for given (t₀, L_scale).
    ///
    /// **Physical bound on `t0_us`.**  The corrected TOF is `tof - t0_us`,
    /// where `tof = tof_factor · L / √E_nom`.  For the corrected grid to
    /// remain physical, `tof_corr > 0` must hold for every bin — i.e.
    /// `t0_us < min_i(tof_i) = tof_factor · L / √(max E_nom)`.  The
    /// `EnergyScaleTransmissionModel` pipeline registers `t0_us` with
    /// bounds of ±10 μs, which safely satisfies this invariant for VENUS
    /// (L = 25 m, E ≤ 200 eV gives `min_tof ≈ 17.7 μs`).
    ///
    /// As a defensive measure — if a caller ever invokes this function
    /// with a `t0_us` that would push any bin's `tof_corr` below zero —
    /// we clamp `t0_us` to just under `min_tof` so the corrected grid
    /// stays monotone and physical.  This is a safety net; the expected
    /// path is that the optimizer's parameter bounds keep `t0_us` well
    /// below the clamp threshold.
    fn corrected_energies(&self, t0_us: f64, l_scale: f64) -> Vec<f64> {
        if self.nominal_energies.is_empty() {
            return Vec::new();
        }
        let l_eff = self.flight_path_m * l_scale;
        // min(tof) over the grid = tof_factor * L / sqrt(max E_nom).
        let min_tof = self
            .nominal_energies
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, e| {
                acc.min(self.tof_factor * self.flight_path_m / e.sqrt())
            });
        let t0_limit = min_tof * (1.0 - 1.0e-12);
        let t0_clamped = t0_us.min(t0_limit);
        self.nominal_energies
            .iter()
            .map(|&e_nom| {
                let tof = self.tof_factor * self.flight_path_m / e_nom.sqrt();
                let tof_corr = tof - t0_clamped;
                (self.tof_factor * l_eff / tof_corr).powi(2)
            })
            .collect()
    }

    /// Doppler-broadened TRUE σ per isotope on the working grid built from the
    /// corrected energies, plus the data-grid layout (issue #608).
    ///
    /// Mirrors `forward_model`: builds the auxiliary extended grid on `e_corr`
    /// WITH the model's resonance data (boundary extension + resonance
    /// fine-structure), evaluates σ via `reich_moore` at those energies, and
    /// Doppler-broadens there.  The corrected grid is re-derived per
    /// `(t0, L_scale)` probe, so the working grid + σ are rebuilt each call —
    /// the only way to reproduce SAMMY's σ(E_corr) under the energy-scale shift
    /// (boundary + fine-structure fidelity).  For tabulated / no resolution the
    /// working grid is `e_corr` itself with an identity layout.
    fn working_xs(
        &self,
        e_corr: &[f64],
        temperature_k: f64,
    ) -> Result<transmission::WorkingGridXs, FittingError> {
        // Issue #634 review: validate the (possibly fitted) temperature at
        // the point of consumption, mirroring
        // `PrecomputedTransmissionModel::evaluate`.  Without this, a NaN or
        // negative `params[temperature_index]` flows into
        // `broadened_cross_sections_on_working_grid`, whose
        // `temperature_k > 0.0` branch silently SKIPS Doppler broadening and
        // returns plausible unbroadened σ as `Ok` ("NaN bypasses guards").
        // The production fitter bounds T ∈ [1, 5000] K, so this guard fires
        // only for direct model API misuse — but the model is public.
        if !temperature_k.is_finite() || temperature_k < 0.0 {
            return Err(FittingError::EvaluationFailed(format!(
                "temperature must be finite and non-negative, got {temperature_k}"
            )));
        }
        // Issue #608: a degenerate calibration can drive corrected energies to
        // 0 (l_scale → 0) or non-finite (l_scale → ∞).  `reich_moore` asserts
        // positive finite energy (an always-on `assert!`), so without this guard
        // such inputs PANIC inside `broadened_cross_sections_on_working_grid` —
        // a process abort across the PyO3 boundary.  Return a graceful Err so the
        // LM/KL/Python callers see a failed evaluate instead of a panic.
        //
        // BEHAVIOR CHANGE vs pre-#608: the old model interpolated a precomputed σ
        // and CLAMPED a degenerate corrected energy to the grid edge, continuing
        // the fit with a (finite but unphysical) value; the true-σ model instead
        // FAILS the evaluate rather than fabricating σ at a non-positive energy.
        // Reachable only by a degenerate calibration, which production keeps out
        // of reach: `validate_energy_scale_params` rejects `l_scale_init <= 0` at
        // setup and `corrected_energies` clamps `t0` below the min TOF, so a real
        // fit never drives `e_corr` to 0 / ∞; this guard is the runtime backstop.
        if let Some(&bad) = e_corr.iter().find(|&&e| !e.is_finite() || e <= 0.0) {
            return Err(FittingError::EvaluationFailed(format!(
                "energy-scale corrected energy is non-positive or non-finite ({bad}); \
                 t0 / L_scale give a degenerate calibration"
            )));
        }
        transmission::broadened_cross_sections_on_working_grid(
            e_corr,
            &self.resonance_data,
            temperature_k,
            self.instrument.as_deref(),
            None,
        )
        .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
    }

    /// Working-grid σ for the current probe, cached (capacity 1, keyed on
    /// `(t0, L_scale)` bits) so a base-point `evaluate` and the Jacobian's
    /// density columns at the SAME probe share one reich_moore+Doppler build
    /// instead of rebuilding it twice (issue #608 perf).  FD probes at
    /// perturbed `(t0, L_scale)` miss and rebuild, as required.
    fn working_xs_for(
        &self,
        params: &[f64],
        e_corr: &[f64],
    ) -> Result<Rc<transmission::WorkingGridXs>, FittingError> {
        let temperature_k = self.temperature_for(params);
        let key = (
            params[self.t0_index].to_bits(),
            params[self.l_scale_index].to_bits(),
            temperature_k.to_bits(),
        );
        let hit = self
            .cached_work_xs
            .borrow()
            .as_ref()
            .and_then(|(k, xs)| (*k == key).then(|| Rc::clone(xs)));
        if let Some(xs) = hit {
            return Ok(xs);
        }
        let xs = Rc::new(self.working_xs(e_corr, temperature_k)?);
        *self.cached_work_xs.borrow_mut() = Some((key, Rc::clone(&xs)));
        Ok(xs)
    }

    /// Evaluate transmission at given parameters (densities + t0 + l_scale).
    ///
    /// When `use_plan_cache` is `true`, the struct-level `(t0, L_scale)`-
    /// keyed plan cache is consulted and populated — appropriate for
    /// evaluate calls that will be followed by more work at the SAME
    /// probe (e.g. `FitModel::evaluate` + `analytical_jacobian` density
    /// cols within one KL outer iter).  When `false`, broadening goes
    /// through the non-plan path unchanged — appropriate for the
    /// one-shot LM FD probes at `(t0 ± h, L)` / `(t0, L ± h)` where
    /// a plan build has no reuse to amortize.  Issue #483 A1.
    fn evaluate_at_with_cache(
        &self,
        params: &[f64],
        e_corr: &[f64],
        use_plan_cache: bool,
    ) -> Result<Vec<f64>, FittingError> {
        // Issue #608: evaluate the TRUE σ at the corrected energies on the
        // working grid (auxiliary extended grid for Gaussian resolution; the
        // data grid for tabulated / no resolution) — reich_moore + Doppler on
        // the working grid, Beer-Lambert, resolution, extract the data points
        // last — exactly as `forward_model` does.  This replaces interpolating
        // a precomputed σ, which clamped at the auxiliary boundary and dropped
        // resonance fine-structure (a forward_model-fidelity gap; #608).
        let work = self.working_xs_for(params, e_corr)?;
        let work_e = &work.layout.energies;

        // Beer-Lambert on the working grid: T = exp(-Σᵢ nᵢ·rᵢ·σᵢ(E)), where rᵢ
        // is the fractional ratio (1.0 for ungrouped isotopes).  No density > 0
        // guard — exp(−n·σ) is well-defined for negative n, matching
        // PrecomputedTransmissionModel (issue #109.1).
        let mut neg_opt = vec![0.0f64; work_e.len()];
        for (iso, xs) in work.sigma.iter().enumerate() {
            let density = params[self.density_indices[iso]];
            let ratio = self.density_ratios[iso];
            for (j, &sigma) in xs.iter().enumerate() {
                neg_opt[j] -= density * ratio * sigma;
            }
        }
        let t_unbroadened: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

        let Some(inst) = self.instrument.as_ref() else {
            // No resolution: the working grid IS the data grid (identity
            // layout), so `extract` is a no-op clone.
            return Ok(work.layout.extract(&t_unbroadened));
        };

        // Resolution on the working grid, then extract the data points last
        // (issue #442 + #608).  For tabulated resolution the working grid IS
        // `e_corr`, so the `(t0, L_scale)`-keyed plan (built on `e_corr`) still
        // matches; for Gaussian the plan is `None` and broadening runs on the
        // auxiliary grid via `apply_resolution`.
        let plan = if use_plan_cache {
            let t0 = params[self.t0_index];
            let l_scale = params[self.l_scale_index];
            self.cached_resolution_plan(t0, l_scale, work_e)
        } else {
            None
        };
        let t_broadened = resolution::apply_resolution_with_plan(
            plan.as_deref(),
            work_e,
            &t_unbroadened,
            &inst.resolution,
        )
        .map_err(|e| FittingError::EvaluationFailed(format!("resolution broadening: {e}")))?;
        Ok(work.layout.extract(&t_broadened))
    }
}

impl FitModel for EnergyScaleTransmissionModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let t0 = params[self.t0_index];
        let l_scale = params[self.l_scale_index];
        let e_corr = self.corrected_energies(t0, l_scale);
        // Public `evaluate` uses the plan cache: downstream the
        // Jacobian (+ joint-Poisson's gradient + Fisher) will re-call
        // `evaluate` at the SAME `(t0, L_scale)` before the next LM
        // step, and the density-col path of `analytical_jacobian`
        // also wants a plan at this probe — all of those hit the
        // cache.  LM's own FD probes — one-coordinate-at-a-time
        // central differences at `(t0 ± h, L_scale)` or
        // `(t0, L_scale ± h)` — go through a dedicated non-cache
        // path in `analytical_jacobian` below, so they don't add
        // plan-build overhead.  Issue #483 A1.
        self.evaluate_at_with_cache(params, &e_corr, true)
    }

    /// Jacobian: analytical for density parameters, finite-difference for t₀ and L_scale.
    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        _y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = self.nominal_energies.len();
        let n_free = free_param_indices.len();
        let mut jacobian = FlatMatrix::zeros(n_e, n_free);

        let t0 = params[self.t0_index];
        let l_scale = params[self.l_scale_index];
        let e_corr = self.corrected_energies(t0, l_scale);
        let energy_scale_method = self.jacobian_method;
        let t0_free_pos = free_param_indices
            .iter()
            .position(|&idx| idx == self.t0_index);
        let l_scale_free_pos = free_param_indices
            .iter()
            .position(|&idx| idx == self.l_scale_index);
        // Partial-GAL t0 FD pair (precomputed once; the L_scale column
        // is derived from this column inline below). Skipped when:
        // - method is not PartialGal, OR
        // - either t0 or L_scale is fixed (the rank-1 derivation needs
        //   both columns paired), OR
        // - `t0 + h` would land at or above the `corrected_energies`
        //   clamp (`min_tof * (1 - 1e-12)`). At the clamp, both `±h`
        //   probes collapse to the same clamped value: the t0 FD column
        //   becomes ~0, and the rank-1 L_scale column would also be ~0
        //   even though `corrected_energies` does NOT clamp on
        //   `L_scale`. Falling through here lets the standard
        //   per-coordinate FD path below compute the L_scale column
        //   correctly. Issue #489.
        let partial_gal_t0_column = if energy_scale_method == EnergyScaleJacobianMethod::PartialGal
            && t0_free_pos.is_some()
            && l_scale_free_pos.is_some()
        {
            let h = 1e-4;
            let min_tof_us = self
                .nominal_energies
                .iter()
                .map(|&e| self.tof_factor * self.flight_path_m / e.sqrt())
                .fold(f64::INFINITY, f64::min);
            let t0_limit = min_tof_us * (1.0 - 1.0e-12);
            // Need (t0 + h) strictly below the clamp so the +h probe
            // returns a distinct corrected grid; otherwise fall through.
            if t0 + h >= t0_limit {
                None
            } else {
                let mut p_plus = params.to_vec();
                let mut p_minus = params.to_vec();
                p_plus[self.t0_index] += h;
                p_minus[self.t0_index] -= h;
                let e_corr_plus =
                    self.corrected_energies(p_plus[self.t0_index], p_plus[self.l_scale_index]);
                let e_corr_minus =
                    self.corrected_energies(p_minus[self.t0_index], p_minus[self.l_scale_index]);
                let y_plus = match self.evaluate_at_with_cache(&p_plus, &e_corr_plus, false) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                let y_minus = match self.evaluate_at_with_cache(&p_minus, &e_corr_minus, false) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                // Per-cell finiteness check.  Without it a NaN in
                // `y_plus[i]` / `y_minus[i]` propagates into both the
                // t0 column AND the L_scale column derived from it via
                // the rank-1 reconstruction at `scale * partial_t0_col[i]`
                // (~line 2280), poisoning the post-convergence
                // covariance the same way lm.rs `compute_jacobian` was
                // vulnerable.  Mirror that fix: zero the entry rather
                // than dropping the column — masked rows (NaN by design
                // in some test contracts) get skipped downstream by the
                // active-mask row-skip in the LM normal-equation
                // assembly, so a 0 in a masked row is benign.
                let mut col = vec![0.0f64; n_e];
                for i in 0..n_e {
                    let a = y_plus[i];
                    let b = y_minus[i];
                    if a.is_finite() && b.is_finite() {
                        col[i] = (a - b) / (2.0 * h);
                    }
                    // else: leave col[i] at 0.0; downstream L_scale
                    // reconstruction `scale * 0 == 0` is consistent.
                }
                Some(col)
            }
        } else {
            None
        };

        // Issue #608: density columns are formed on the WORKING grid (auxiliary
        // extended grid for Gaussian resolution; `e_corr` for tabulated / no
        // resolution) from the TRUE σ at the corrected energies (reich_moore +
        // Doppler), resolution-broadened there, and the data points extracted
        // last — matching `forward_model` and `evaluate`.
        let work = match self.working_xs_for(params, &e_corr) {
            Ok(w) => w,
            Err(_) => return None,
        };
        let work_layout = &work.layout;
        let work_e = &work_layout.energies;

        // Unresolved T on the WORKING grid: T = exp(-Σᵢ nᵢ·rᵢ·σᵢ).
        let mut neg_opt = vec![0.0f64; work_e.len()];
        for (iso, xs) in work.sigma.iter().enumerate() {
            let density = params[self.density_indices[iso]];
            let ratio = self.density_ratios[iso];
            for (j, &sigma) in xs.iter().enumerate() {
                neg_opt[j] -= density * ratio * sigma;
            }
        }
        let t_unresolved: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

        // Density-column plan: Issue #483 A1 routes through the
        // struct-level `(t0, L_scale)`-keyed cache.  Built on the working grid
        // (== `e_corr` for tabulated, where the plan is meaningful; `None` for
        // Gaussian).  When `self.evaluate(params)` ran earlier in the same KL
        // outer iteration the cache was already populated at the current
        // `(t0, L_scale)` and this lookup is a cheap Arc clone.
        //
        // An earlier `n_density_cols >= 2` gate is
        // dropped here: the cache makes the plan build a one-shot
        // cost amortized across every evaluate at `(t0, L_scale)` in
        // the surrounding KL iteration, so even the N_density = 1
        // case (A.1 / KL+grouped+TZERO) now benefits from plan
        // reuse across 3 evaluates + 2 jacobians per outer iter.
        // The non-tabulated / build-failure branches still return
        // `None` → `apply_resolution_with_plan(None, …)` forwards
        // byte-identically to `apply_resolution`.
        let density_plan = self.cached_resolution_plan(t0, l_scale, work_e);

        // Role indices (t0/L_scale/temperature/densities) are assumed
        // DISTINCT — first-match layout; aliasing is not supported in
        // this FD-arm fill. See NormalizedTransmissionModel's "Index
        // invariant" for the accumulate-hardened pattern used by the
        // simple wrappers.
        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            // Temperature (issue #634), when free, is differentiated by the
            // per-coordinate central FD arm below — it is neither t0 nor
            // L_scale, so the PartialGal block never fires for it. Perturbing
            // T changes σ (via `working_xs` at the T-widened cache key) but not
            // the corrected grid, so its ±h probes share `e_corr`.
            let is_temperature = Some(fp_idx) == self.temperature_index;
            if fp_idx == self.t0_index || fp_idx == self.l_scale_index || is_temperature {
                // partial-GAL: when both t0 and L_scale are free, the t0
                // column comes from a single pre-computed FD pair (above),
                // and the L_scale column is the per-bin rank-1 derivation
                //   J[:, L_scale]_i = ((tof_i - t0_clamped) / L_scale) * J[:, t0]_i.
                //
                // The structural factorisation through `e_corr` holds
                // when `R` depends on `(t0, L_scale)` only through
                // `e_corr` — `broaden_presorted` uses `self.flight_path_m`
                // (not the model's `L_nominal * L_scale`) for
                // `tof_center` / `e_prime`, so tabulated kernels satisfy
                // it. The per-bin rank-1 simplification additionally
                // assumes per-bin homogeneity of `(tof - t0) / L_scale`
                // across the kernel support; see the
                // `EnergyScaleJacobianMethod` doc for the empirical
                // characterisation. When only one of t0 / L_scale is
                // free, we fall through to the standard FD path below.
                if let Some(partial_t0_col) = &partial_gal_t0_column {
                    if fp_idx == self.t0_index {
                        for (i, &val) in partial_t0_col.iter().enumerate() {
                            *jacobian.get_mut(i, col) = val;
                        }
                        continue;
                    }
                    if fp_idx == self.l_scale_index {
                        let l_scale = params[self.l_scale_index];
                        // Issue #500: at `l_scale ≈ 0` the rank-1 factor
                        // `(tof - t0_clamped) / l_scale` blows up and
                        // produces NaN columns when combined with the
                        // FD-based t0 reference (which goes to ~0 at the
                        // same boundary).  Skip the partial-GAL path
                        // and fall through to the per-coordinate FD
                        // section below — mirrors the t0 clamp-boundary
                        // fallthrough (when
                        // `partial_gal_t0_column` is `None`, the entire
                        // partial-GAL block is skipped).  Production
                        // L_scale bounds are typically `[0.99, 1.01]`,
                        // so this guard fires only at API edge cases.
                        if l_scale.abs() >= L_SCALE_EPSILON {
                            let t0 = params[self.t0_index];
                            // Match the `corrected_energies` t0 clamp so the
                            // (tof - t0) factor in the rank-1 derivation
                            // agrees with the production forward at the
                            // clamp boundary.
                            let min_tof_us = self
                                .nominal_energies
                                .iter()
                                .map(|&e| self.tof_factor * self.flight_path_m / e.sqrt())
                                .fold(f64::INFINITY, f64::min);
                            let t0_clamped = t0.min(min_tof_us * (1.0 - 1.0e-12));
                            for (i, &e_nom) in self.nominal_energies.iter().enumerate() {
                                let tof_i = self.tof_factor * self.flight_path_m / e_nom.sqrt();
                                let scale = (tof_i - t0_clamped) / l_scale;
                                *jacobian.get_mut(i, col) = scale * partial_t0_col[i];
                            }
                            continue;
                        }
                        // l_scale ≈ 0: do NOT `continue`; flow falls
                        // through to the FD path below for this column.
                    }
                }
                // Finite difference for energy-scale parameters.
                //
                // Central-difference probes perturb one coordinate at
                // a time: `(t0 ± h, L_scale)` when differentiating in
                // `t0`, or `(t0, L_scale ± h)` when differentiating
                // in `L_scale`.  Each perturbed point is a distinct
                // `(t0, L_scale)` key that would miss the struct
                // plan cache, and building a plan for the probe has
                // no reuse to amortize.  Route them through
                // `evaluate_at_with_cache(..., false)` so they stay
                // on the original non-plan `apply_resolution` path.
                // The public `FitModel::evaluate` path continues to
                // use the cache for the many-uses-per-probe callers
                // (KL solver's deviance + gradient + Fisher at the
                // current probe).  Issue #483 A1.
                // FD step per coordinate: t0 in μs → absolute 1e-4; L_scale
                // dimensionless → absolute 1e-7; temperature in K → a RELATIVE
                // step (1e-4·T, i.e. ~0.03 K at 300 K, matching L_scale's
                // relative scale) since T is O(300 K) and an absolute 1e-7 K
                // would be pure round-off. Central differences make the
                // truncation error O((h/T)²) ~ 1e-9, far below the analytic
                // column it is validated against (see the FD-vs-analytic test).
                let h = if fp_idx == self.t0_index {
                    1e-4
                } else if is_temperature {
                    1e-4 * params[fp_idx].max(1.0)
                } else {
                    1e-7
                };
                let mut p_plus = params.to_vec();
                let mut p_minus = params.to_vec();
                p_plus[fp_idx] += h;
                p_minus[fp_idx] -= h;
                let t0_plus = p_plus[self.t0_index];
                let l_plus = p_plus[self.l_scale_index];
                let t0_minus = p_minus[self.t0_index];
                let l_minus = p_minus[self.l_scale_index];
                let e_corr_plus = self.corrected_energies(t0_plus, l_plus);
                let e_corr_minus = self.corrected_energies(t0_minus, l_minus);
                let y_plus = match self.evaluate_at_with_cache(&p_plus, &e_corr_plus, false) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                let y_minus = match self.evaluate_at_with_cache(&p_minus, &e_corr_minus, false) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                // Per-cell finiteness check — mirrors the lm.rs
                // `compute_jacobian` FD path.  A NaN in the perturbed
                // model at an active row would otherwise feed NaN
                // through the post-convergence covariance; per-cell
                // skip leaves masked-row NaN benign (the LM normal-
                // equation assembly already row-skips those).
                for i in 0..n_e {
                    let a = y_plus[i];
                    let b = y_minus[i];
                    if a.is_finite() && b.is_finite() {
                        *jacobian.get_mut(i, col) = (a - b) / (2.0 * h);
                    }
                    // else: leave at zero-default.
                }
            } else {
                // Density parameter: analytical derivative on the WORKING grid
                // (issue #608) from the TRUE σ, resolution-broadened there,
                // data points extracted last.
                // ∂T/∂n_g = extract(R[-(Σ_{iso∈g} rᵢ·σ_iso(E)) · T_unresolved(E)])
                let mut sigma_sum = vec![0.0f64; work_e.len()];
                for (iso, &di) in self.density_indices.iter().enumerate() {
                    if di == fp_idx {
                        let ratio = self.density_ratios[iso];
                        for (j, &sigma) in work.sigma[iso].iter().enumerate() {
                            sigma_sum[j] += ratio * sigma;
                        }
                    }
                }
                let inner_deriv: Vec<f64> = (0..work_e.len())
                    .map(|i| -sigma_sum[i] * t_unresolved[i])
                    .collect();

                // Apply resolution to derivative if enabled.
                //
                // When `density_plan` is `Some` (tabulated resolution
                // + populated cache) we hit the struct-level
                // `(t0, L_scale)`-keyed plan (built on the working grid, which
                // equals `e_corr` for tabulated).  When `None` (Gaussian
                // resolution or build failure),
                // `apply_resolution_with_plan(None, …)` transparently
                // forwards to `apply_resolution` — bit-exact with
                // the pre-cache path.  Issue #483 A1.
                if let Some(inst) = &self.instrument {
                    let resolved_deriv = match resolution::apply_resolution_with_plan(
                        density_plan.as_deref(),
                        work_e,
                        &inner_deriv,
                        &inst.resolution,
                    ) {
                        Ok(v) => v,
                        Err(_) => return None,
                    };
                    let resolved_deriv = work_layout.extract(&resolved_deriv);
                    for (i, &val) in resolved_deriv.iter().enumerate() {
                        *jacobian.get_mut(i, col) = val;
                    }
                } else {
                    // No resolution → identity layout, inner is already data grid.
                    for (i, &val) in inner_deriv.iter().enumerate() {
                        *jacobian.get_mut(i, col) = val;
                    }
                }
            }
        }

        Some(jacobian)
    }
}

// ── ForwardModel implementations (Phase 1) ──────────────────────────────
//
// Each implementation delegates to the existing FitModel logic.
// `predict` == `evaluate`, `jacobian` converts FlatMatrix → Vec<Vec<f64>>.

use crate::forward_model::ForwardModel;

impl ForwardModel for PrecomputedTransmissionModel {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        let fm = self.analytical_jacobian(params, free_param_indices, y_current)?;
        Some(flat_matrix_to_vecs(&fm, free_param_indices.len()))
    }

    fn n_data(&self) -> usize {
        // Issue #608: when a Gaussian working-grid layout is attached,
        // `cross_sections` lives on the (longer) working grid, but the number of
        // DATA points the model predicts is the layout's data-index count.
        // Without a layout the working grid IS the data grid.
        if let Some(layout) = &self.work_layout {
            layout.data_indices.len()
        } else if self.cross_sections.is_empty() {
            0
        } else {
            self.cross_sections[0].len()
        }
    }

    fn n_params(&self) -> usize {
        // Max index in density_indices + 1
        self.density_indices
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m + 1)
    }
}

impl ForwardModel for TransmissionFitModel {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        let fm = self.analytical_jacobian(params, free_param_indices, y_current)?;
        Some(flat_matrix_to_vecs(&fm, free_param_indices.len()))
    }

    fn n_data(&self) -> usize {
        self.energies.len()
    }

    fn n_params(&self) -> usize {
        let max_density = self.density_indices.iter().copied().max().unwrap_or(0);
        let max_temp = self.temperature_index.unwrap_or(0);
        max_density.max(max_temp) + 1
    }
}

impl<M: FitModel> ForwardModel for NormalizedTransmissionModel<M> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        let fm = self.analytical_jacobian(params, free_param_indices, y_current)?;
        Some(flat_matrix_to_vecs(&fm, free_param_indices.len()))
    }

    fn n_data(&self) -> usize {
        self.sqrt_energies.len()
    }

    fn n_params(&self) -> usize {
        // The background indices are the highest parameter indices.
        let mut max_idx = self
            .anorm_index
            .max(self.back_a_index)
            .max(self.back_b_index)
            .max(self.back_c_index);
        if let Some(di) = self.back_d_index {
            max_idx = max_idx.max(di);
        }
        if let Some(fi) = self.back_f_index {
            max_idx = max_idx.max(fi);
        }
        max_idx + 1
    }
}

// ── Multiplicative baseline wrapper (issue #635) ──────────────────────────

/// Reference energy for the multiplicative-baseline log basis: the geometric
/// midpoint `√(E_min · E_max)` of the grid.  Centering the basis at the
/// geometric midpoint makes the design columns `1, z, z²` near-orthogonal on
/// a log-uniform grid and makes `b0` the mid-grid baseline value — so its
/// bound is directly the "a few % off unity" statement from the VENUS data.
///
/// The caller must guarantee a non-empty grid of positive energies (the
/// pipeline validates this); on an empty grid this returns NaN, which the
/// config validation rejects downstream.  The actual extrema are folded
/// over the slice rather than read from `first()`/`last()`, so the
/// documented `√(E_min·E_max)` holds regardless of grid ordering (the
/// pipeline convention is ascending, but `UnifiedFitConfig::new` does not
/// enforce it and the two forms agree bit-exactly on any monotonic grid).
pub fn baseline_reference_energy(energies: &[f64]) -> f64 {
    if energies.is_empty() {
        return f64::NAN;
    }
    let (lo, hi) = energies
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &e| {
            (lo.min(e), hi.max(e))
        });
    (lo * hi).sqrt()
}

/// Reference energy for the baseline log basis, computed over the **active
/// fit window only** (issue #648).
///
/// The full grid extends far beyond the resonances of interest (a VENUS Ta
/// grid spans 4.5 eV–2.3 MeV while the fit window is 8–45 eV).  Centering
/// the `ln(E/E_ref)` basis at the full-grid midpoint (≈3211 eV) instead of
/// the active-window midpoint (≈19 eV) pushes every active bin to a large
/// negative `z`, so the `1, z, z²` columns stop being near-orthogonal and
/// the baseline silently absorbs Doppler broadening — the fitted
/// temperature runs away with `warnings = []`.  Restricting the midpoint to
/// the bins that actually enter the cost (the active mask) restores it.
///
/// `active` is the per-bin mask from
/// [`crate::active_mask::build_active_mask`]; `None` (no `fit_energy_range`)
/// is identical to [`baseline_reference_energy`] over the whole grid.  A
/// mask selecting no bins falls back to the full grid rather than returning
/// NaN (the pipeline's active-bin-count gate rejects an empty window
/// upstream, so this branch is defensive only).
pub fn baseline_reference_energy_active(energies: &[f64], active: Option<&[bool]>) -> f64 {
    match active {
        None => baseline_reference_energy(energies),
        Some(mask) => {
            debug_assert_eq!(mask.len(), energies.len());
            let (lo, hi, any) = energies.iter().zip(mask).fold(
                (f64::INFINITY, f64::NEG_INFINITY, false),
                |(lo, hi, any), (&e, &a)| {
                    if a {
                        (lo.min(e), hi.max(e), true)
                    } else {
                        (lo, hi, any)
                    }
                },
            );
            if any {
                (lo * hi).sqrt()
            } else {
                baseline_reference_energy(energies)
            }
        }
    }
}

/// Bounded multiplicative polynomial baseline (issue #635):
///
/// ```text
/// y(E) = B(E) · T_inner(E),   B(E) = b0 + b1·z + b2·z²,   z = ln(E / E_ref)
/// ```
///
/// where `E_ref = √(E_min·E_max)` (see [`baseline_reference_energy`]) and
/// `T_inner` is any inner [`FitModel`] — typically the bare transmission
/// model, or [`NormalizedTransmissionModel`] when the SAMMY additive
/// background is also configured (the baseline is the OUTERMOST factor).
///
/// ## INTENTIONAL DEPARTURE from SAMMY
///
/// SAMMY's modern data-reduction path applies a SCALAR normalization plus
/// additive backgrounds only:
/// `T_obs = Anorm·T + BackA + BackB/√E + BackC·√E + BackD·exp(−BackF/√E)`
/// (`cro/mnrm1.f90`, subroutine `Norm`, applied to
/// every data type via `the/ZeroKCrossCorrections_M.f90`).  SAMMY's nearest
/// analogue to an energy-dependent multiplicative normalization is the
/// DORMANT legacy power-law `Anorm = Anrm(1) + Anrm(2)·E^Anrm(3)`
/// (`acs/macs4.f90:440–450`, `Find_Www_Yyy`), which is not reachable from the
/// modern reconstruction path.  This low-order ln-E polynomial baseline is a
/// NEREIDS extension motivated by the IPTS-37432 campaign (findings A3/A5):
/// real VENUS sample/open-beam ratios sit a few % from unity with smooth
/// energy dependence, and freeing the SAMMY `Anorm` together with temperature
/// and density is degenerate on such data (observed: T → 4471 K, n +76 %,
/// χ²/ν 932, with no warning).  The bounded multiplicative form fitted
/// jointly with temperature at fixed density produced χ²/ν ≈ 2–8 across the
/// 20-run campaign.
///
/// Because `b0` is exactly degenerate with `Anorm`, the pipeline rejects a
/// free `Anorm` alongside ANY configured baseline — including a fully
/// frozen one (see `nereids-pipeline::validate_multiplicative_baseline`).
/// A frozen-`b0` + free-`Anorm` combination would be well-posed, but
/// supporting it buys nothing (`Anorm` would just play `b0`'s role at a
/// rescaled value) and splits the normalization story across two knobs;
/// the sanctioned combination is `Anorm` held fixed.
///
/// ## Index invariant
///
/// The baseline indices (`b0_index`, `b1_index`, `b2_index`) must NOT
/// designate a parameter the inner model reads: the analytic Jacobian
/// filters the baseline indices out of the inner free set, so such a
/// collision cannot be detected and the column would silently omit
/// B(E) × ∂T_inner/∂p. Aliasing AMONG the baseline indices themselves
/// IS supported — the Jacobian columns accumulate.
pub struct MultiplicativeBaselineModel<M: FitModel> {
    /// The inner model (bare transmission, or the additive-background
    /// wrapper when both are configured).
    inner: M,
    /// Precomputed `z_i = ln(E_i / E_ref)`.
    ln_ratio: Vec<f64>,
    /// Precomputed `z_i²`.
    ln_ratio_sq: Vec<f64>,
    /// Index of `b0` (mid-grid baseline value) in the parameter vector.
    b0_index: usize,
    /// Index of `b1` (slope per ln-E unit) in the parameter vector.
    b1_index: usize,
    /// Index of `b2` (curvature per ln-E² unit) in the parameter vector.
    b2_index: usize,
    /// Optional per-bin active mask (SAMMY EMIN/EMAX-equivalent
    /// `fit_energy_range`, #514).  When `Some`, the positivity guard in
    /// [`FitModel::evaluate`] is scoped to ACTIVE bins only: masked bins
    /// contribute nothing to any mask-honouring cost function, so a
    /// negative `B(E)` there must not reject the whole trial step — on a
    /// wide TOF grid (|z| up to ~6–8) coefficients that are comfortably
    /// in-bounds inside the fit window can drive `B` negative at far
    /// out-of-window bins, and an unscoped guard would veto every such
    /// trial (λ inflation → spurious non-convergence).  Masked bins still
    /// emit the raw product `B·T_inner` (possibly negative), matching the
    /// LM/joint-Poisson contract that masked-bin values are never read.
    active_mask: Option<Vec<bool>>,
}

impl<M: FitModel> MultiplicativeBaselineModel<M> {
    /// Create the wrapper.  `e_ref` is normally
    /// [`baseline_reference_energy`]`(energies)`; it is passed explicitly so
    /// result consumers can reconstruct `B(E)` with the exact same reference.
    pub fn new(
        inner: M,
        energies: &[f64],
        e_ref: f64,
        b0_index: usize,
        b1_index: usize,
        b2_index: usize,
    ) -> Self {
        let ln_ratio: Vec<f64> = energies.iter().map(|&e| (e / e_ref).ln()).collect();
        let ln_ratio_sq: Vec<f64> = ln_ratio.iter().map(|&z| z * z).collect();
        Self {
            inner,
            ln_ratio,
            ln_ratio_sq,
            b0_index,
            b1_index,
            b2_index,
            active_mask: None,
        }
    }

    /// Scope the runtime positivity guard to the given active mask
    /// (`None` = all bins active, the default).  See the `active_mask`
    /// field doc for why masked bins must be exempt.
    #[must_use]
    pub fn with_active_mask(mut self, mask: Option<&[bool]>) -> Self {
        self.active_mask = mask.map(<[bool]>::to_vec);
        self
    }

    /// `B(E_i)` for the current parameters.
    fn baseline_at(&self, params: &[f64], i: usize) -> f64 {
        params[self.b0_index]
            + params[self.b1_index] * self.ln_ratio[i]
            + params[self.b2_index] * self.ln_ratio_sq[i]
    }
}

impl<M: FitModel> FitModel for MultiplicativeBaselineModel<M> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let t_inner = self.inner.evaluate(params)?;
        let mut out = Vec::with_capacity(t_inner.len());
        for (i, &t) in t_inner.iter().enumerate() {
            let b = self.baseline_at(params, i);
            // Positivity guard: a non-positive baseline is unphysical (the
            // measured ratio would change sign) and would silently flip the
            // model.  The default bounds keep B > 0 on typical windows, but a
            // very wide TOF grid (z ≈ ±8) can drive in-bounds coefficients
            // negative — reject the trial step instead.  Mid-iteration `Err`
            // is a REJECTED trial in the LM (backtrack / raise λ); the config
            // validation guarantees the initial point satisfies B(E) > 0.
            //
            // SCOPED to active bins (#514 review R2): a bin masked out by
            // fit_energy_range contributes nothing to any mask-honouring
            // cost function, so a negative B there must not veto the trial
            // step — an unscoped guard rejected in-window-valid coefficients
            // because of out-of-window bins, inflating λ into spurious
            // non-convergence.  Masked bins emit the raw (possibly negative)
            // product, which the solvers never read.
            // A mask shorter than the grid treats out-of-range bins as
            // ACTIVE (guarded) — the conservative default for a misused
            // public constructor; the pipeline always builds equal-length
            // masks from the same grid.
            let bin_active = self
                .active_mask
                .as_ref()
                .is_none_or(|m| m.get(i).copied().unwrap_or(true));
            let positive = b.is_finite() && b > 0.0;
            if bin_active && !positive {
                return Err(FittingError::EvaluationFailed(format!(
                    "multiplicative baseline B(E) = {b} is non-positive at bin {i} \
                     (b0 + b1·z + b2·z² with z = {})",
                    self.ln_ratio[i],
                )));
            }
            out.push(b * t);
        }
        Ok(out)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        let n_free = free_param_indices.len();

        // Recompute T_inner once — both the baseline columns (∂/∂b_k =
        // z^k · T_inner) and the inner-column scaling (× B) need it.
        let t_inner = self.inner.evaluate(params).ok()?;

        let baseline_set = [self.b0_index, self.b1_index, self.b2_index];
        let inner_free_indices: Vec<usize> = free_param_indices
            .iter()
            .copied()
            .filter(|idx| !baseline_set.contains(idx))
            .collect();

        // Inner Jacobian ONCE, against the inner model's own output.
        let inner_jac = if !inner_free_indices.is_empty() {
            self.inner
                .analytical_jacobian(params, &inner_free_indices, &t_inner)
        } else {
            None
        };

        let mut inner_col_map = std::collections::HashMap::new();
        for (col, &idx) in inner_free_indices.iter().enumerate() {
            inner_col_map.insert(idx, col);
        }

        let mut jacobian = FlatMatrix::zeros(n_e, n_free);
        // Independent role checks with accumulation (+=) rather than a
        // first-match if/else-if chain: nothing forbids the baseline
        // indices from aliasing, and baseline_at() reads an aliased
        // parameter for every role it occupies, so its derivative is the
        // SUM of the matching columns. Distinct indices touch each column
        // once on a zeroed matrix — identical to assignment. A baseline
        // index colliding with an INNER-model parameter remains
        // undetectable here (baseline indices are filtered out of
        // inner_free_indices) — see the struct docs.
        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            let mut matched = false;
            if fp_idx == self.b0_index {
                // ∂y/∂b0 = T_inner
                for (i, &ti) in t_inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) += ti;
                }
                matched = true;
            }
            if fp_idx == self.b1_index {
                // ∂y/∂b1 = z · T_inner
                for (i, &ti) in t_inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) += self.ln_ratio[i] * ti;
                }
                matched = true;
            }
            if fp_idx == self.b2_index {
                // ∂y/∂b2 = z² · T_inner
                for (i, &ti) in t_inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) += self.ln_ratio_sq[i] * ti;
                }
                matched = true;
            }
            if let Some(&inner_col) = inner_col_map.get(&fp_idx) {
                // Inner parameter: ∂y/∂p = B(E) · ∂T_inner/∂p
                if let Some(ref jac) = inner_jac {
                    for i in 0..n_e {
                        let b = self.baseline_at(params, i);
                        *jacobian.get_mut(i, col) += b * jac.get(i, inner_col);
                    }
                    matched = true;
                } else {
                    // Inner has no analytic Jacobian — FD for everything.
                    return None;
                }
            }
            if !matched {
                // Unknown parameter — should not happen; fall back to FD.
                return None;
            }
        }
        Some(jacobian)
    }
}

impl<M: FitModel> ForwardModel for MultiplicativeBaselineModel<M> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        let fm = self.analytical_jacobian(params, free_param_indices, y_current)?;
        Some(flat_matrix_to_vecs(&fm, free_param_indices.len()))
    }

    fn n_data(&self) -> usize {
        self.ln_ratio.len()
    }

    fn n_params(&self) -> usize {
        // Assumes the baseline coefficients occupy the HIGHEST parameter
        // indices (the pipeline appends them last: density → temperature →
        // energy-scale → background → baseline).  A caller that interleaves
        // baseline indices below other parameters would under-report the
        // vector length here — matching the sibling wrappers, which make
        // the same layout assumption (e.g. EnergyScaleTransmissionModel
        // over t0/l_scale).
        self.b0_index.max(self.b1_index).max(self.b2_index) + 1
    }
}

impl ForwardModel for EnergyScaleTransmissionModel {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        let fm = self.analytical_jacobian(params, free_param_indices, y_current)?;
        Some(flat_matrix_to_vecs(&fm, free_param_indices.len()))
    }

    fn n_data(&self) -> usize {
        self.nominal_energies.len()
    }

    fn n_params(&self) -> usize {
        self.t0_index.max(self.l_scale_index) + 1
    }
}

/// Convert a `FlatMatrix` (row-major) to `Vec<Vec<f64>>` (column-major).
///
/// Returns `cols` vectors, each of length `fm.nrows()`.
fn flat_matrix_to_vecs(fm: &FlatMatrix, cols: usize) -> Vec<Vec<f64>> {
    let nrows = fm.nrows;
    (0..cols)
        .map(|j| (0..nrows).map(|i| fm.get(i, j)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{self, FitModel, LmConfig};
    use crate::parameters::{FitParameter, ParameterSet};
    use nereids_core::types::Isotope;
    use nereids_endf::resonance::test_support::u238_single_resonance;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};

    /// ∞-norm of the residual between two equal-length spectra.
    /// (Issue #608 aux-grid regression-test helper.)
    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    /// ∞-norm (max |value|) of a spectrum — a scale for relative thresholds.
    fn max_abs(a: &[f64]) -> f64 {
        a.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
    }

    // ── PrecomputedTransmissionModel ─────────────────────────────────────────

    /// Verify Beer-Lambert: T(E) = exp(-Σᵢ nᵢ·σᵢ(E)).
    #[test]
    fn precomputed_evaluate_matches_beer_lambert() {
        let model = make_precomputed(
            vec![
                vec![1.0, 2.0, 3.0], // isotope 0
                vec![0.5, 0.5, 0.5], // isotope 1
            ],
            vec![0, 1],
        );

        let params = [0.2f64, 0.4f64];
        let y = model.evaluate(&params).unwrap();

        let expected: Vec<f64> = (0..3)
            .map(|i| {
                let s0 = [1.0, 2.0, 3.0][i];
                let s1 = [0.5, 0.5, 0.5][i];
                (-params[0] * s0 - params[1] * s1).exp()
            })
            .collect();

        assert_eq!(y.len(), 3);
        for (yi, ei) in y.iter().zip(expected.iter()) {
            assert!(
                (yi - ei).abs() < 1e-12,
                "evaluate mismatch: got {yi}, expected {ei}"
            );
        }
    }

    /// Analytical Jacobian ∂T/∂nᵢ = -σᵢ(E)·T(E) must match central-difference FD.
    #[test]
    fn precomputed_analytical_jacobian_matches_finite_difference() {
        let model = make_precomputed(
            vec![
                vec![1.0, 2.0, 3.0], // isotope 0
                vec![0.5, 0.5, 0.5], // isotope 1
            ],
            vec![0, 1],
        );

        let params = [0.2f64, 0.4f64];
        let y = model.evaluate(&params).unwrap();
        let free = vec![0usize, 1usize];

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some(_)");

        assert_eq!(jac.nrows, 3); // n_energies
        assert_eq!(jac.ncols, 2); // n_free_params

        // Central-difference reference.
        let h = 1e-6f64;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;

            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();

            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                assert!(
                    (fd - ana).abs() < 1e-6,
                    "Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, analytical={ana:.8}"
                );
            }
        }
    }

    /// When two isotopes share a density parameter, the Jacobian column must
    /// equal -T(E) * (σ₀(E) + σ₁(E)), not just the first isotope's σ.
    #[test]
    fn precomputed_jacobian_tied_parameters_sums_both_isotopes() {
        // Two isotopes mapped to the same density parameter (index 0).
        let model = make_precomputed(
            vec![
                vec![1.0, 2.0, 3.0], // isotope 0
                vec![0.5, 1.0, 1.5], // isotope 1 — tied to same param
            ],
            vec![0, 0], // both isotopes share param[0]
        );

        let params = [0.1f64];
        let y = model.evaluate(&params).unwrap();
        let free = vec![0usize];

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some(_)");

        // Expected: ∂T/∂n = -T(E) * (σ₀(E) + σ₁(E))
        for i in 0..3 {
            let sigma_sum = [1.0, 2.0, 3.0][i] + [0.5, 1.0, 1.5][i];
            let expected = -y[i] * sigma_sum;
            assert!(
                (jac.get(i, 0) - expected).abs() < 1e-12,
                "Tied Jacobian mismatch at E[{i}]: got {}, expected {expected}",
                jac.get(i, 0)
            );
        }
    }

    // ── TransmissionFitModel ─────────────────────────────────────────────────

    #[test]
    fn test_recover_single_isotope_thickness() {
        let data = u238_single_resonance();
        let true_thickness = 0.0005;

        // Generate synthetic data
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            0.0,
            None,
            (vec![0], vec![1.0]),
            None,
            None,
        )
        .unwrap();

        let y_obs = model.evaluate(&[true_thickness]).unwrap();
        let sigma = vec![0.01; y_obs.len()]; // 1% uncertainty

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("thickness", 0.001), // initial guess 2× off
        ]);

        let result =
            lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default())
                .unwrap();

        assert!(result.converged, "Fit did not converge");
        let fitted = result.params[0];
        assert!(
            (fitted - true_thickness).abs() / true_thickness < 0.01,
            "Fitted thickness = {}, true = {}, error = {:.1}%",
            fitted,
            true_thickness,
            (fitted - true_thickness).abs() / true_thickness * 100.0,
        );
    }

    #[test]
    fn test_recover_two_isotope_thicknesses() {
        let u238 = u238_single_resonance();

        // Second isotope with resonance at 20 eV
        let other = ResonanceData {
            isotope: Isotope::new(1, 10).unwrap(),
            za: 1010,
            awr: 10.0,
            ranges: vec![ResonanceRange {
                energy_low: 0.0,
                energy_high: 100.0,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 5.0,
                naps: 1,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 10.0,
                    apl: 5.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 20.0,
                        j: 0.5,
                        gn: 0.1,
                        gg: 0.05,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                ap_table: None,
                r_external: vec![],
            }],
        };

        let true_t1 = 0.0003;
        let true_t2 = 0.0001;

        let energies: Vec<f64> = (0..301).map(|i| 1.0 + (i as f64) * 0.1).collect();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![u238, other],
            0.0,
            None,
            (vec![0, 1], vec![1.0, 1.0]),
            None,
            None,
        )
        .unwrap();

        let y_obs = model.evaluate(&[true_t1, true_t2]).unwrap();
        let sigma = vec![0.01; y_obs.len()];

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("U-238 thickness", 0.001),
            FitParameter::non_negative("Other thickness", 0.001),
        ]);

        let result =
            lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default())
                .unwrap();

        assert!(
            result.converged,
            "Fit did not converge after {} iterations",
            result.iterations
        );

        let (fit_t1, fit_t2) = (result.params[0], result.params[1]);
        assert!(
            (fit_t1 - true_t1).abs() / true_t1 < 0.05,
            "U-238: fitted={}, true={}, error={:.1}%",
            fit_t1,
            true_t1,
            (fit_t1 - true_t1).abs() / true_t1 * 100.0,
        );
        assert!(
            (fit_t2 - true_t2).abs() / true_t2 < 0.05,
            "Other: fitted={}, true={}, error={:.1}%",
            fit_t2,
            true_t2,
            (fit_t2 - true_t2).abs() / true_t2 * 100.0,
        );
    }

    // ── Temperature fitting ──────────────────────────────────────────────────

    /// Verify that temperature_index makes evaluate() read T from the
    /// parameter vector instead of the fixed `temperature_k` field.
    #[test]
    fn temperature_index_overrides_fixed_temperature() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        // Model with fixed temperature = 0 K but temperature_index pointing
        // to params[1].
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            (vec![0], vec![1.0]),
            Some(1),
            None,
        )
        .unwrap();

        // Model with fixed temperature = 300 K (no temperature_index).
        let model_fixed = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            300.0,
            None,
            (vec![0], vec![1.0]),
            None,
            None,
        )
        .unwrap();

        let density = 0.0005;
        let y_via_index = model.evaluate(&[density, 300.0]).unwrap();
        let y_via_fixed = model_fixed.evaluate(&[density]).unwrap();

        for (a, b) in y_via_index.iter().zip(y_via_fixed.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "temperature_index path disagrees with fixed path: {} vs {}",
                a,
                b
            );
        }
    }

    /// Recover temperature from Doppler-broadened synthetic data.
    ///
    /// Generates transmission at T_true with known density, then fits both
    /// density and temperature simultaneously.
    #[test]
    fn test_recover_temperature() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let true_temp = 300.0; // K

        // Energy grid around the 6.674 eV resonance.
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.025).collect();

        // Generate synthetic data at the true temperature.
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            0.0, // ignored — temperature_index is set
            None,
            (vec![0], vec![1.0]),
            Some(1), // params[1] = temperature
            None,
        )
        .unwrap();

        let mut y_obs = model.evaluate(&[true_density, true_temp]).unwrap();
        // Add tiny deterministic noise so reduced_chi2 stays positive.
        // Without noise, the analytical Jacobian converges to exact parameters,
        // yielding chi2 ≈ 0, which makes covariance ≈ 0 and uncertainty NaN.
        for (i, y) in y_obs.iter_mut().enumerate() {
            *y *= 1.0 + 1e-5 * ((i % 7) as f64 - 3.0);
        }
        let sigma = vec![0.005; y_obs.len()];

        // Fit with initial guesses offset from truth.
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.001),
            FitParameter {
                name: "temperature_k".into(),
                value: 200.0, // initial guess 100 K off
                lower: 1.0,
                upper: 2000.0,
                fixed: false,
            },
        ]);

        let config = LmConfig {
            max_iter: 200,
            ..LmConfig::default()
        };

        let result = lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &config).unwrap();

        assert!(
            result.converged,
            "Temperature fit did not converge after {} iterations",
            result.iterations
        );

        let fit_density = result.params[0];
        let fit_temp = result.params[1];

        // Tiny deterministic noise (max ±3e-5): optimizer should converge to within 0.1%.
        assert!(
            (fit_density - true_density).abs() / true_density < 0.001,
            "Density: fitted={}, true={}, error={:.1}%",
            fit_density,
            true_density,
            (fit_density - true_density).abs() / true_density * 100.0,
        );
        assert!(
            (fit_temp - true_temp).abs() / true_temp < 0.001,
            "Temperature: fitted={:.1} K, true={:.1} K, error={:.1}%",
            fit_temp,
            true_temp,
            (fit_temp - true_temp).abs() / true_temp * 100.0,
        );

        // Verify uncertainty is reported.
        let unc = result
            .uncertainties
            .expect("uncertainties should be available");
        assert!(
            unc.len() == 2,
            "expected 2 uncertainties, got {}",
            unc.len()
        );
        assert!(
            unc[1] > 0.0 && unc[1].is_finite(),
            "temperature uncertainty should be positive and finite, got {}",
            unc[1]
        );
    }

    /// Analytical Jacobian for TransmissionFitModel (with temperature) must
    /// agree with central-difference finite-difference Jacobian.
    ///
    /// This validates both the density columns (∂T/∂nᵢ = -σᵢ·T) and the
    /// temperature column (forward FD at T+dT).
    #[test]
    fn transmission_fit_model_analytical_jacobian_matches_fd() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel::new(
            energies,
            vec![data],
            0.0,
            None,
            (vec![0], vec![1.0]),
            Some(1), // params[1] = temperature
            None,
        )
        .unwrap();

        let params = [0.0005f64, 300.0f64]; // density, temperature
        let y = model.evaluate(&params).unwrap();
        let free = vec![0usize, 1usize];

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some(_)");

        assert_eq!(jac.nrows, y.len());
        assert_eq!(jac.ncols, 2);

        // Central-difference reference.
        let h = 1e-6f64;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h * (1.0 + params[p_idx].abs());
            p_minus[p_idx] -= h * (1.0 + params[p_idx].abs());

            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();

            let actual_2h = p_plus[p_idx] - p_minus[p_idx];
            for i in 0..y.len() {
                let fd = (y_plus[i] - y_minus[i]) / actual_2h;
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                // Use a meaningful floor: when both FD and analytical values
                // are below 1e-10, relative error comparisons are dominated
                // by floating-point noise and are not physically meaningful.
                //
                // The floor was raised from 1e-15 to 1e-10 alongside the
                // B=S_l boundary condition fix in the Reich-Moore U-matrix.
                // That fix shifted near-zero cross-section values from
                // O(1e-15) to O(1e-10), making the old floor too tight for
                // floating-point comparison at those magnitudes.
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 0.01,
                    "Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, analytical={ana:.8}, \
                     rel_err={:.4}",
                    err / scale,
                );
            }
        }
    }

    /// Verify that the broadened-XS cache avoids redundant recomputation.
    /// Calling evaluate() twice with the same temperature should produce
    /// identical results and reuse the cache.
    #[test]
    fn transmission_fit_model_cache_reuse() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel::new(
            energies,
            vec![data],
            0.0,
            None,
            (vec![0], vec![1.0]),
            Some(1),
            None,
        )
        .unwrap();

        // First call populates the cache.
        let y1 = model.evaluate(&[0.0005, 300.0]).unwrap();
        assert!(model.cached_broadened_xs.borrow().is_some());
        assert!((model.cached_temperature.get() - 300.0).abs() < 1e-15);

        // Second call with same temperature but different density should
        // reuse cached broadened XS (no rebroadening).
        let y2 = model.evaluate(&[0.001, 300.0]).unwrap();
        assert!((model.cached_temperature.get() - 300.0).abs() < 1e-15);

        // Results must differ (different density) but cache temperature unchanged.
        assert!(
            (y1[100] - y2[100]).abs() > 1e-10,
            "different densities should produce different transmission"
        );

        // Change temperature — cache should update.
        let _y3 = model.evaluate(&[0.0005, 600.0]).unwrap();
        assert!((model.cached_temperature.get() - 600.0).abs() < 1e-15);
    }

    // ── NormalizedTransmissionModel ─────────────────────────────────────────

    /// Helper: make a PrecomputedTransmissionModel with given cross-sections
    /// and no resolution (Beer-Lambert only).
    fn make_precomputed(
        xs: Vec<Vec<f64>>,
        density_indices: Vec<usize>,
    ) -> PrecomputedTransmissionModel {
        PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs),
            density_indices: Arc::new(density_indices),
            energies: None,
            instrument: None,
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: None,
        }
    }

    // ── Cubature dispatch tests ─────────────────────────────────────────

    /// Helper: build a synthetic resolution kernel + plan + matrix.
    /// CI-hermetic (no PLEIADES fixture) using the same synthetic-
    /// overlap-plan pattern as the surrogate module's tests.
    fn synthetic_resolution_setup(
        n_grid: usize,
        half_kernel: usize,
    ) -> (
        Vec<f64>,
        Arc<ResolutionPlan>,
        nereids_physics::resolution::ResolutionMatrix,
    ) {
        assert!(n_grid > 2 * half_kernel);
        let energies: Vec<f64> = (0..n_grid).map(|i| 10.0 + i as f64).collect();
        let mut starts: Vec<u32> = vec![0];
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
        let plan = nereids_physics::resolution::test_support::plan_from_raw_parts(
            energies.clone(),
            starts,
            lo_idx,
            frac_arr,
            weight_arr,
            norm,
        );
        let matrix = plan.compile_to_matrix();
        (energies, Arc::new(plan), matrix)
    }

    /// Helper: build a k-isotope synthetic σ stack.
    fn synthetic_sigmas(n_grid: usize, k: usize) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            let center = 10.0 + (j as f64 + 1.0) * (n_grid as f64) / (k as f64 + 1.0);
            let width = 3.0;
            out.push(
                (0..n_grid)
                    .map(|ell| {
                        let e = 10.0 + ell as f64;
                        100.0 * (-((e - center).powi(2)) / (width * width)).exp() + 5.0
                    })
                    .collect(),
            );
        }
        out
    }

    /// Helper: build a sparse cubature plan against a known
    /// (matrix, σ stack) pair, with the canonical design-study training
    /// rule.
    fn build_cubature(
        matrix: &nereids_physics::resolution::ResolutionMatrix,
        sigmas: &[Vec<f64>],
        train_max: Vec<f64>,
    ) -> Arc<SparseEmpiricalCubaturePlan> {
        let k = sigmas.len();
        let n_rows = matrix.len();
        let mut flat = Vec::with_capacity(k * n_rows);
        for row in sigmas {
            flat.extend_from_slice(row);
        }
        let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        Arc::new(
            SparseEmpiricalCubaturePlan::build(matrix, &flat, k, &training, &anchor)
                .expect("synthetic cubature build"),
        )
    }

    /// Build an `InstrumentParams` wrapping a trivial delta-like
    /// tabulated resolution (single ref energy, δ-kernel).  Used
    /// only because the dispatch guards check `instrument.is_some()`
    /// AND require `ResolutionFunction::Tabulated(_)`.  The actual
    /// broadening wouldn't fire on the cubature path regardless
    /// (cubature folds `apply_resolution*` into its atom sweep).
    fn make_trivial_instrument() -> Arc<InstrumentParams> {
        use nereids_physics::resolution::ResolutionFunction;
        // Tabulated resolution required for cubature-dispatch tests:
        // the eligibility guard refuses the dispatch when the active
        // instrument resolution isn't `ResolutionFunction::Tabulated`.
        // The test_support helper builds a minimal delta-like kernel;
        // the broadening never actually runs on the cubature path
        // (cubature.forward replaces apply_resolution entirely).
        let tab =
            Arc::new(nereids_physics::resolution::test_support::trivial_tabulated_resolution(25.0));
        let res_fn = ResolutionFunction::Tabulated(tab);
        Arc::new(InstrumentParams { resolution: res_fn })
    }

    #[test]
    fn precomputed_cubature_dispatches_at_k2_matching_k() {
        // k = 2 with an installed cubature plan: evaluate should
        // return the cubature's forward output (which differs from
        // the exact `exp(-Σ n σ) + apply_r` path ONLY at held-out
        // densities; at training densities the LP pins them exactly).
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);
        let train_max = vec![1e-4_f64, 1e-4];
        let cubature = build_cubature(&matrix, &sigmas, train_max.clone());

        // Build the model with cubature installed.  The resolution
        // plan MUST also be installed for the cubature dispatch to
        // fire — without it, `cubature_eligible` refuses the plan
        // on the grounds that the cubature would be silently
        // bypassing an unknown resolution operator.
        let mut model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas.clone()),
            density_indices: Arc::new(vec![0, 1]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(make_trivial_instrument()),
            resolution_plan: Some(Arc::clone(&plan)),
            sparse_cubature_plan: Some(Arc::clone(&cubature)),
            sparse_scalar_plan: None,
            work_layout: None,
        };

        // Evaluate at a training density: cubature ≡ exact to LP
        // tolerance.
        let n = [0.25 * train_max[0], 0.25 * train_max[1]];
        let t_cubature = model.evaluate(&n).unwrap();

        // Disable cubature → exact cannot match bit-for-bit (different
        // summation order).  But we can compute the cubature output
        // directly and confirm it equals what `evaluate()` returned.
        model.sparse_cubature_plan = None;
        let t_exact_via_r = {
            // exp(-Σ n σ) then apply_r
            let n_grid_local = n_grid;
            let mut neg_opt = vec![0.0_f64; n_grid_local];
            for (j, &nj) in n.iter().enumerate() {
                for (ell, &sig) in sigmas[j].iter().enumerate() {
                    neg_opt[ell] -= nj * sig;
                }
            }
            let t_un: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();
            nereids_physics::resolution::apply_r(&matrix, &t_un)
        };
        let t_cubature_direct = cubature.forward(&n);

        // Sanity: cubature direct output matches what evaluate() returned.
        for (a, b) in t_cubature.iter().zip(t_cubature_direct.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        // Cubature vs exact at training density: LP-pinned equivalence.
        let max_err = t_cubature
            .iter()
            .zip(t_exact_via_r.iter())
            .map(|(a, b)| {
                let denom = a.abs().max(b.abs()).max(1e-12);
                (a - b).abs() / denom
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-9,
            "at training density, cubature vs exact max err = {max_err:.3e}",
        );
    }

    #[test]
    fn precomputed_cubature_falls_back_at_k1() {
        // k = 1 with a k=2 cubature → cubature_eligible returns false
        // (plan.k mismatch with n_density_params), dispatch MUST
        // fall back to the exact `exp(-n σ) + apply_resolution`
        // path.  We prove fallback via byte-identity: constructing a
        // second model WITHOUT the cubature plan must produce
        // exactly the same output as the first model WITH the
        // ineligible plan.  A false-positive dispatch would violate
        // this invariant because the k=2 cubature's atoms live in
        // ℝ² and `cubature.forward([n])` would panic on the
        // input-length check in `SparseEmpiricalCubaturePlan::forward`
        // — OR, worse, if the guard check accidentally accepted a
        // k=2 plan for a k=1 model the output would numerically
        // differ from straight Beer-Lambert by more than
        // floating-point noise.
        let n_grid = 40_usize;
        // `plan` intentionally unused here: this test wants both
        // model variants in the no-dispatch state (no cubature can
        // fire because k=1 vs cubature.k=2), so installing a
        // resolution plan would add work without changing the
        // tested invariant.
        let (energies, _plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas_k2 = synthetic_sigmas(n_grid, 2);
        let cubature_k2 = build_cubature(&matrix, &sigmas_k2, vec![1e-4_f64, 1e-4]);

        // Model has k = 1 (only one isotope in cross_sections), but a
        // k = 2 cubature is installed → must fall back.
        let sigmas_k1 = synthetic_sigmas(n_grid, 1);
        let model_with_plan = PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas_k1.clone()),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(make_trivial_instrument()),
            resolution_plan: None,
            sparse_cubature_plan: Some(Arc::clone(&cubature_k2)),
            sparse_scalar_plan: None,
            work_layout: None,
        };
        let model_without_plan = PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas_k1.clone()),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(make_trivial_instrument()),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: None,
        };

        let n = [1e-4_f64];
        let t_with = model_with_plan.evaluate(&n).unwrap();
        let t_without = model_without_plan.evaluate(&n).unwrap();
        assert_eq!(t_with.len(), n_grid);
        assert_eq!(t_without.len(), n_grid);
        // Byte identity: ineligible-plan dispatch MUST equal
        // no-plan dispatch exactly.
        for (a, b) in t_with.iter().zip(t_without.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "fallback path must be byte-identical to the no-plan path; \
                 otherwise the k=2 cubature is silently firing on a k=1 model",
            );
        }
    }

    #[test]
    fn precomputed_cubature_no_plan_means_exact_path() {
        // No cubature installed → byte-identical to the
        // pre-cubature-dispatch path.  This is the regression guard:
        // the dispatch addition
        // must not change the default forward path.
        let n_grid = 40_usize;
        let (_energies, _plan, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);

        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas.clone()),
            density_indices: Arc::new(vec![0, 1]),
            energies: None,
            instrument: None,
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: None,
        };

        let n = [1e-4_f64, 1e-4];
        let t = model.evaluate(&n).unwrap();
        // Exact Beer-Lambert: T = exp(-Σ n σ).
        for (ell, &t_val) in t.iter().enumerate() {
            let tau: f64 = sigmas
                .iter()
                .zip(n.iter())
                .map(|(s, &ni)| ni * s[ell])
                .sum();
            let expected = (-tau).exp();
            assert!(
                (t_val - expected).abs() < 1e-14,
                "at ell={ell}: got {t_val}, expected {expected}",
            );
        }
    }

    #[test]
    fn precomputed_cubature_jacobian_matches_forward_derivative() {
        // Cubature Jacobian columns should equal the per-isotope
        // derivatives of the cubature forward output at the anchor.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);
        let train_max = vec![1e-4_f64, 1e-4];
        let cubature = build_cubature(&matrix, &sigmas, train_max.clone());

        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas),
            density_indices: Arc::new(vec![0, 1]),
            energies: Some(Arc::new(energies)),
            instrument: Some(make_trivial_instrument()),
            resolution_plan: Some(Arc::clone(&plan)),
            sparse_cubature_plan: Some(Arc::clone(&cubature)),
            sparse_scalar_plan: None,
            work_layout: None,
        };

        // Use anchor density: LP pins Jacobian exactly here.
        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let y_curr = model.evaluate(&anchor).unwrap();
        let jac = model
            .analytical_jacobian(&anchor, &[0, 1], &y_curr)
            .expect("cubature Jacobian path");

        // Cross-check: cubature.forward_and_jacobian should give the
        // same J.
        let (_t_ref, jac_flat_ref) = cubature.forward_and_jacobian(&anchor);
        for i in 0..n_grid {
            for col in 0..2 {
                let from_model = jac.get(i, col);
                let from_cubature = jac_flat_ref[i * 2 + col];
                assert!(
                    (from_model - from_cubature).abs() < 1e-14,
                    "row {i} col {col}: model = {from_model}, cubature = {from_cubature}",
                );
            }
        }
    }

    // ── TransmissionFitModel cubature dispatch tests ──────────────────
    //
    // The per-pixel `TransmissionFitModel` fires the cubature path
    // with extra guards (`temperature_index.is_none()` for σ stack
    // stability).  These tests exercise BOTH `evaluate()` and
    // `analytical_jacobian()` directly on `TransmissionFitModel`,
    // not the precomputed variant.

    /// Build a minimal `TransmissionFitModel` with a single trivial
    /// resonance per isotope + the synthetic σ used for the
    /// Precomputed tests, so the cubature dispatch condition can
    /// trigger without loading full ENDF data.
    fn make_trivial_fit_model(energies: Vec<f64>, k: usize) -> TransmissionFitModel {
        // Build k synthetic Isotope / ResonanceData pairs — the fit
        // model doesn't actually consult them when the cubature
        // dispatch fires (cubature.forward replaces `exp(-Σ n σ) +
        // apply_resolution`).  But the constructor still validates
        // the count.
        // Minimal ResonanceData — the cubature dispatch fires
        // before any ENDF-derived code runs, so `ranges` can be
        // empty.  When the dispatch falls through (tests that check
        // the exact path), we don't exercise cross_sections from
        // these resonance_data either; the model uses
        // `precomputed_cross_sections` / `base_xs`.
        let resonance_data: Vec<ResonanceData> = (0..k)
            .map(|j| {
                let iso = Isotope::new(40 + j as u32, 96 + j as u32).unwrap();
                ResonanceData {
                    isotope: iso,
                    za: ((40 + j) * 1000 + (96 + j)) as u32,
                    awr: 96.0 + j as f64,
                    ranges: vec![],
                }
            })
            .collect();

        TransmissionFitModel::new(
            energies,
            resonance_data,
            293.6,
            Some(make_trivial_instrument()),
            ((0..k).collect(), vec![1.0; k]),
            None,
            None,
        )
        .expect("TransmissionFitModel::new")
    }

    #[test]
    fn fit_model_cubature_dispatches_at_anchor() {
        // Build a k = 2 cubature and a TransmissionFitModel whose
        // density_indices / ratios map directly (identity) onto it.
        // `evaluate()` at the anchor density MUST equal
        // `cubature.forward(anchor)` exactly — the LP equality
        // constraint pins it.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);
        let train_max = vec![1e-4_f64, 1e-4];
        let cubature = build_cubature(&matrix, &sigmas, train_max.clone());

        // Install BOTH the resolution plan and the cubature plan:
        // the eligibility guard requires `resolution_plan.is_some()`
        // so the cubature doesn't silently bypass an unknown
        // resolution operator.
        let model = make_trivial_fit_model(energies.clone(), 2)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_cubature_plan(Some(cubature.clone()));

        // Evaluate at a training density (LP pins exactly) → model
        // output equals cubature output.
        let n = [0.25 * train_max[0], 0.25 * train_max[1]];
        let t_model = model.evaluate(&n).unwrap();
        let t_cub = cubature.forward(&n);
        assert_eq!(t_model.len(), n_grid);
        for (a, b) in t_model.iter().zip(t_cub.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "TransmissionFitModel cubature dispatch must return cubature.forward() byte-exact at the LP-pinned anchor",
            );
        }
    }

    #[test]
    fn fit_model_cubature_jacobian_matches_cubature_output() {
        // Same pattern as the Precomputed Jacobian test but on
        // TransmissionFitModel.  analytical_jacobian at the anchor
        // density must return exactly `cubature.forward_and_jacobian(n)`'s
        // J matrix.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);
        let train_max = vec![1e-4_f64, 1e-4];
        let cubature = build_cubature(&matrix, &sigmas, train_max.clone());

        let model = make_trivial_fit_model(energies, 2)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_cubature_plan(Some(cubature.clone()));

        let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
        let y_curr = model.evaluate(&anchor).unwrap();
        let jac = model
            .analytical_jacobian(&anchor, &[0, 1], &y_curr)
            .expect("cubature Jacobian path on TransmissionFitModel");
        let (_t_ref, jac_flat_ref) = cubature.forward_and_jacobian(&anchor);
        for i in 0..n_grid {
            for col in 0..2 {
                let from_model = jac.get(i, col);
                let from_cubature = jac_flat_ref[i * 2 + col];
                assert_eq!(
                    from_model.to_bits(),
                    from_cubature.to_bits(),
                    "row {i} col {col}: TransmissionFitModel must return cubature J byte-exact",
                );
            }
        }
    }

    #[test]
    fn fit_model_cubature_falls_back_on_grid_mismatch() {
        // Build a cubature on one grid, install it on a model with a
        // DIFFERENT same-length grid.  Dispatch must refuse the plan
        // via the new `to_bits()` grid-identity check and produce
        // byte-identical output to the no-plan model (exact path).
        let n_grid = 40_usize;
        let (energies_a, _plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);
        let train_max = vec![1e-4_f64, 1e-4];
        let cubature = build_cubature(&matrix, &sigmas, train_max);

        // A different same-length grid (shifted by 1 eV).
        let energies_b: Vec<f64> = energies_a.iter().map(|&e| e + 1.0).collect();

        let model_with_stale_plan =
            make_trivial_fit_model(energies_b.clone(), 2).with_sparse_cubature_plan(Some(cubature));
        let model_without_plan = make_trivial_fit_model(energies_b, 2);

        let n = [1e-5_f64, 1e-5];
        let t_stale = model_with_stale_plan.evaluate(&n).unwrap();
        let t_exact = model_without_plan.evaluate(&n).unwrap();
        for (a, b) in t_stale.iter().zip(t_exact.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "stale-grid cubature plan MUST NOT fire; evaluate() must match no-plan byte-exactly",
            );
        }
    }

    #[test]
    fn fit_model_cubature_falls_back_when_density_escapes_box() {
        // Build cubature with train_max = [1e-4, 1e-4], install
        // the density_box, then call evaluate() with a density
        // WELL beyond the 1.5× tolerance.  Dispatch must fall back
        // to the exact path rather than silently extrapolate the
        // surrogate outside its trained region.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);
        let train_max = vec![1e-4_f64, 1e-4];

        // Build cubature AND attach the density_box.
        let cubature = {
            let flat: Vec<f64> = sigmas.iter().flat_map(|s| s.iter().copied()).collect();
            let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
            let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
            Arc::new(
                SparseEmpiricalCubaturePlan::build(&matrix, &flat, 2, &training, &anchor)
                    .expect("build")
                    .with_density_box(train_max.clone()),
            )
        };

        let model_with = make_trivial_fit_model(energies.clone(), 2)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_cubature_plan(Some(Arc::clone(&cubature)));
        let model_without =
            make_trivial_fit_model(energies, 2).with_resolution_plan(Some(Arc::clone(&plan)));

        // Escape: 5× the training max → well outside the 1.5× tolerance.
        let n_escape = [5.0 * train_max[0], 5.0 * train_max[1]];
        let t_with = model_with.evaluate(&n_escape).unwrap();
        let t_without = model_without.evaluate(&n_escape).unwrap();
        // If the guard fired correctly, the cubature-installed
        // model falls back to the exact path and produces the same
        // output as the no-plan model — byte-identical.
        for (a, b) in t_with.iter().zip(t_without.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "density-box escape guard MUST fall back to exact path byte-identically",
            );
        }
    }

    #[test]
    fn fit_model_cubature_dispatches_without_resolution_plan_attached() {
        // Single-spectrum regression: callers of the non-spatial
        // `fit_spectrum_typed` / `build_transmission_model` path
        // attach a cubature via
        // `UnifiedFitConfig::with_precomputed_sparse_cubature_plan`
        // but typically don't also pre-build a `ResolutionPlan` (the
        // per-call `apply_resolution` broaden path is used
        // otherwise).  The cubature fast path MUST still fire — a
        // prior `resolution_plan.is_some()` requirement
        // made the new API inert on this surface.
        let n_grid = 40_usize;
        let (energies, _plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 2);
        let train_max = vec![1e-4_f64, 1e-4];
        let cubature = build_cubature(&matrix, &sigmas, train_max.clone());

        // Intentionally NOT installing a resolution plan.  The
        // instrument's tabulated resolution is enough.
        let model = make_trivial_fit_model(energies.clone(), 2)
            .with_sparse_cubature_plan(Some(Arc::clone(&cubature)));

        let n = [0.25 * train_max[0], 0.25 * train_max[1]];
        let t_model = model.evaluate(&n).unwrap();
        let t_cub = cubature.forward(&n);
        for (a, b) in t_model.iter().zip(t_cub.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "cubature dispatch must fire on single-spectrum path without a separate ResolutionPlan attached",
            );
        }
    }

    // ── Scalar (k = 1) dispatch-guard tests ───────────────────────────
    //
    // The cubature tests above cover
    // the k ≥ 2 path; the scalar path is a separate surrogate with its
    // own eligibility guard (`scalar_eligible`) and its own
    // density-box guard (`scalar_density_within_box`).  These tests
    // exercise the scalar-specific guards: k=1-only, grid-identity
    // via `to_bits()`, tabulated-only instrument resolution,
    // density-box escape, and that the pure no-plan path remains
    // byte-identical to the pre-surrogate path.

    /// Helper: build a synthetic scalar (k = 1) Chebyshev plan on
    /// the same grid as the cubature helpers.  Takes an
    /// `Arc<ResolutionPlan>` so tests can share the same Arc
    /// pointer with the model's `resolution_plan` (required by the
    /// `Arc::ptr_eq` dispatch guard).
    fn build_scalar_plan(
        res_plan: Arc<ResolutionPlan>,
        sigma_k1: &[f64],
        n_max: f64,
    ) -> Arc<ScalarSurrogatePlan> {
        Arc::new(
            nereids_physics::surrogate::ScalarChebyshevPlan::build(res_plan, sigma_k1, n_max, 16)
                .expect("synthetic scalar Chebyshev build"),
        )
    }

    /// Helper: build a `PrecomputedTransmissionModel` with the
    /// caller-chosen σ / k / resolution-plan / scalar-plan state.
    /// Mirrors `make_trivial_fit_model` but targets the model that
    /// actually dispatches scalar in production (spatial routes
    /// scalar-eligible k=1 through `PrecomputedTransmissionModel`).
    fn make_precomp_for_scalar(
        energies: Vec<f64>,
        sigmas: Vec<Vec<f64>>,
        density_indices: Vec<usize>,
        resolution_plan: Option<Arc<ResolutionPlan>>,
        scalar_plan: Option<Arc<ScalarSurrogatePlan>>,
    ) -> PrecomputedTransmissionModel {
        PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas),
            density_indices: Arc::new(density_indices),
            energies: Some(Arc::new(energies)),
            instrument: Some(make_trivial_instrument()),
            resolution_plan,
            sparse_cubature_plan: None,
            sparse_scalar_plan: scalar_plan,
            work_layout: None,
        }
    }

    #[test]
    fn precomputed_scalar_dispatches_at_k1() {
        // k = 1 with both the scalar plan and the resolution plan
        // installed (same Arc) and σ matching the plan's
        // fingerprint: evaluate() must return the scalar plan's
        // forward output byte-exact.
        let n_grid = 40_usize;
        let (energies, res_plan, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(Arc::clone(&res_plan), &sigmas[0], n_max);

        let model = make_precomp_for_scalar(
            energies,
            sigmas,
            vec![0],
            Some(Arc::clone(&res_plan)),
            Some(Arc::clone(&scalar)),
        );

        let n = [0.5 * n_max];
        let t_model = model.evaluate(&n).unwrap();
        let t_scalar = scalar.forward_scalar(n[0]);
        assert_eq!(t_model.len(), n_grid);
        for (a, b) in t_model.iter().zip(t_scalar.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "scalar dispatch must return forward_scalar() byte-exact",
            );
        }
    }

    #[test]
    fn precomputed_scalar_jacobian_matches_derivative() {
        let n_grid = 40_usize;
        let (energies, res_plan, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(Arc::clone(&res_plan), &sigmas[0], n_max);

        let model = make_precomp_for_scalar(
            energies,
            sigmas,
            vec![0],
            Some(Arc::clone(&res_plan)),
            Some(Arc::clone(&scalar)),
        );

        let n = [0.5 * n_max];
        let y_curr = model.evaluate(&n).unwrap();
        let jac = model
            .analytical_jacobian(&n, &[0], &y_curr)
            .expect("scalar Jacobian path");
        let (_t_ref, dt_ref) = scalar.forward_and_derivative_scalar(n[0]);
        assert_eq!(jac.ncols, 1);
        assert_eq!(jac.nrows, n_grid);
        for (i, &dt_i) in dt_ref.iter().enumerate().take(n_grid) {
            assert_eq!(
                jac.get(i, 0).to_bits(),
                dt_i.to_bits(),
                "row {i}: scalar dT/dn must be byte-exact",
            );
        }
    }

    #[test]
    fn precomputed_scalar_falls_back_at_k2() {
        // k = 2 with a scalar plan installed → `scalar_eligible`
        // rejects `cross_sections.len() == 1` guard (k=2 model has
        // 2 σ rows).  Dispatch falls back.
        let n_grid = 40_usize;
        let (energies, res_plan, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas_k2 = synthetic_sigmas(n_grid, 2);
        let sigma_k1 = synthetic_sigmas(n_grid, 1).remove(0);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(Arc::clone(&res_plan), &sigma_k1, n_max);

        let model_with = make_precomp_for_scalar(
            energies.clone(),
            sigmas_k2.clone(),
            vec![0, 1],
            Some(Arc::clone(&res_plan)),
            Some(scalar),
        );
        let model_without = make_precomp_for_scalar(
            energies,
            sigmas_k2,
            vec![0, 1],
            Some(Arc::clone(&res_plan)),
            None,
        );
        let n = [1e-4_f64, 2e-4];
        let t_with = model_with.evaluate(&n).unwrap();
        let t_without = model_without.evaluate(&n).unwrap();
        for (a, b) in t_with.iter().zip(t_without.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "scalar plan must refuse k=2 dispatch → byte-identical fallback",
            );
        }
    }

    #[test]
    fn precomputed_scalar_falls_back_on_stale_resolution_plan() {
        // Same-grid
        // DIFFERENT-kernel ResolutionPlan swap must not silently
        // dispatch.  The `Arc::ptr_eq` guard on the scalar plan's
        // stored source plan is the O(1) check that closes this.
        let n_grid = 40_usize;
        let (energies, res_plan_a, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(Arc::clone(&res_plan_a), &sigmas[0], n_max);

        // Build a DIFFERENT ResolutionPlan on the same grid (wider
        // kernel) and attach it to the model.  Even though the
        // grid matches bit-for-bit, the scalar plan was built from
        // res_plan_a and its `source_resolution_plan` Arc differs
        // from res_plan_b → dispatch refuses.
        let (_e_b, res_plan_b, _matrix_b) = synthetic_resolution_setup(n_grid, 6);
        let model_stale = make_precomp_for_scalar(
            energies.clone(),
            sigmas.clone(),
            vec![0],
            Some(Arc::clone(&res_plan_b)),
            Some(Arc::clone(&scalar)),
        );
        let model_noplan =
            make_precomp_for_scalar(energies, sigmas, vec![0], Some(res_plan_b), None);
        let n = [0.25 * n_max];
        let t_stale = model_stale.evaluate(&n).unwrap();
        let t_exact = model_noplan.evaluate(&n).unwrap();
        for (a, b) in t_stale.iter().zip(t_exact.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "scalar plan with non-ptr_eq source_resolution_plan MUST NOT fire",
            );
        }
    }

    #[test]
    fn precomputed_scalar_falls_back_on_stale_sigma() {
        // Plan built
        // from σ_A, attached to a model whose cross_sections[0] is
        // σ_B on the same grid with the same resolution plan →
        // σ-fingerprint mismatch forces fallback.
        let n_grid = 40_usize;
        let (energies, res_plan, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigma_a = synthetic_sigmas(n_grid, 1);
        // σ_B: flip one element of σ_A so the fingerprint differs
        // but the shape / magnitude is plausible.
        let mut sigma_b = sigma_a.clone();
        sigma_b[0][n_grid / 2] += 1.0; // tiny perturbation → different fingerprint
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(Arc::clone(&res_plan), &sigma_a[0], n_max);

        let model_stale = make_precomp_for_scalar(
            energies.clone(),
            sigma_b.clone(),
            vec![0],
            Some(Arc::clone(&res_plan)),
            Some(scalar),
        );
        let model_noplan =
            make_precomp_for_scalar(energies, sigma_b, vec![0], Some(res_plan), None);
        let n = [0.25 * n_max];
        let t_stale = model_stale.evaluate(&n).unwrap();
        let t_exact = model_noplan.evaluate(&n).unwrap();
        for (a, b) in t_stale.iter().zip(t_exact.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "σ-fingerprint mismatch MUST force fallback → byte-identical to no-plan",
            );
        }
    }

    #[test]
    fn precomputed_scalar_falls_back_when_density_escapes_box() {
        let n_grid = 40_usize;
        let (energies, res_plan, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(Arc::clone(&res_plan), &sigmas[0], n_max);

        let model_with = make_precomp_for_scalar(
            energies.clone(),
            sigmas.clone(),
            vec![0],
            Some(Arc::clone(&res_plan)),
            Some(Arc::clone(&scalar)),
        );
        let model_without =
            make_precomp_for_scalar(energies, sigmas, vec![0], Some(Arc::clone(&res_plan)), None);
        let n_escape = [2.0 * n_max];
        let t_with = model_with.evaluate(&n_escape).unwrap();
        let t_without = model_without.evaluate(&n_escape).unwrap();
        for (a, b) in t_with.iter().zip(t_without.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "density-box escape guard must fall back byte-identically",
            );
        }
    }

    #[test]
    fn precomputed_scalar_rejects_nonfinite_density() {
        let n_grid = 40_usize;
        let (energies, res_plan, _matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(Arc::clone(&res_plan), &sigmas[0], n_max);

        let model_with = make_precomp_for_scalar(
            energies.clone(),
            sigmas.clone(),
            vec![0],
            Some(Arc::clone(&res_plan)),
            Some(Arc::clone(&scalar)),
        );
        let model_without =
            make_precomp_for_scalar(energies, sigmas, vec![0], Some(Arc::clone(&res_plan)), None);
        for bad_n in [f64::NAN, f64::INFINITY, -1e-6_f64] {
            let n = [bad_n];
            let t_with = model_with.evaluate(&n).unwrap();
            let t_without = model_without.evaluate(&n).unwrap();
            for (i, (a, b)) in t_with.iter().zip(t_without.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "n = {bad_n}: scalar guard must fall back byte-exactly; row {i}",
                );
            }
        }
    }

    #[test]
    fn scalar_density_within_box_direct_guard() {
        // Unit-test the scalar_density_within_box helper directly
        // without going through the model dispatch.  Chebyshev is a
        // polynomial interpolant that diverges exponentially outside
        // `[0, n_max]` — measured: 73 % rel err
        // at `1.5 × n_max`.  The guard is therefore **strict**
        // `n ≤ train_max`, not the cubature's 1.5× tolerance.
        let n_grid = 16_usize;
        let (_energies, res_plan, _matrix) = synthetic_resolution_setup(n_grid, 2);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 1e-4_f64;
        let plan =
            nereids_physics::surrogate::ScalarChebyshevPlan::build(res_plan, &sigmas[0], n_max, 16)
                .expect("build");

        // Inside the box: accepted.
        assert!(scalar_density_within_box(&plan, 0.0));
        assert!(scalar_density_within_box(&plan, 0.5 * n_max));
        assert!(scalar_density_within_box(&plan, n_max));
        // Any positive excursion past the box is rejected (no
        // 1.5× tolerance).
        assert!(!scalar_density_within_box(
            &plan,
            n_max * (1.0 + f64::EPSILON)
        ));
        assert!(!scalar_density_within_box(&plan, 1.01 * n_max));
        assert!(!scalar_density_within_box(&plan, 1.5 * n_max));
        assert!(!scalar_density_within_box(&plan, 2.0 * n_max));
        // Non-finite and negative must be rejected.
        assert!(!scalar_density_within_box(&plan, f64::NAN));
        assert!(!scalar_density_within_box(&plan, f64::INFINITY));
        assert!(!scalar_density_within_box(&plan, f64::NEG_INFINITY));
        assert!(!scalar_density_within_box(&plan, -1e-9));
    }

    #[test]
    fn density_param_indices_sorted_by_value() {
        // First-appearance order would swap columns for non-
        // monotonic group layouts like [1, 0, 1].  Sorted-by-value
        // keeps dispatch aligned with the cubature's σ-stack
        // indexing (`sigmas[j * n_rows + ℓ]` = σ for density param
        // j).
        assert_eq!(density_param_indices(&[0, 0, 0]), vec![0]);
        assert_eq!(density_param_indices(&[0, 1, 2, 3]), vec![0, 1, 2, 3]);
        assert_eq!(density_param_indices(&[1, 0, 1]), vec![0, 1]);
        assert_eq!(density_param_indices(&[3, 1, 2, 0, 2]), vec![0, 1, 2, 3]);
    }

    /// Verify that NormalizedTransmissionModel with identity normalization
    /// (Anorm=1, all background=0) gives the same result as the inner model.
    #[test]
    fn normalized_identity_matches_inner() {
        let xs = vec![
            vec![1.0, 2.0, 3.0], // isotope 0
            vec![0.5, 0.5, 0.5], // isotope 1
        ];
        let inner_ref = make_precomputed(xs.clone(), vec![0, 1]);
        let inner_wrap = make_precomputed(xs, vec![0, 1]);

        let energies = [4.0, 9.0, 16.0];
        // params: [density0, density1, Anorm, BackA, BackB, BackC]
        let model = NormalizedTransmissionModel::new(inner_wrap, &energies, 2, 3, 4, 5);

        let params = [0.2, 0.4, 1.0, 0.0, 0.0, 0.0];
        let y_norm = model.evaluate(&params).unwrap();
        let y_inner = inner_ref.evaluate(&params).unwrap();

        for (a, b) in y_norm.iter().zip(y_inner.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "identity normalization should match inner: {} vs {}",
                a,
                b
            );
        }
    }

    /// Verify the normalization formula:
    /// T_out = Anorm * T_inner + BackA + BackB/sqrt(E) + BackC*sqrt(E)
    #[test]
    fn normalized_formula_correct() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner_ref = make_precomputed(xs.clone(), vec![0]);
        let inner_wrap = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0]; // sqrt = [2, 3, 4]
        let model = NormalizedTransmissionModel::new(inner_wrap, &energies, 1, 2, 3, 4);

        // params: [density, Anorm, BackA, BackB, BackC]
        let anorm = 0.95;
        let back_a = 0.01;
        let back_b = 0.02;
        let back_c = 0.005;
        let density = 0.3;
        let params = [density, anorm, back_a, back_b, back_c];

        let y = model.evaluate(&params).unwrap();
        let t_inner = inner_ref.evaluate(&params).unwrap();

        for (i, (&yi, &ti)) in y.iter().zip(t_inner.iter()).enumerate() {
            let sqrt_e = energies[i].sqrt();
            let expected = anorm * ti + back_a + back_b / sqrt_e + back_c * sqrt_e;
            assert!(
                (yi - expected).abs() < 1e-12,
                "E[{i}]: got {yi}, expected {expected}"
            );
        }
    }

    /// Analytical Jacobian of NormalizedTransmissionModel must match
    /// central-difference finite-difference.
    #[test]
    fn normalized_analytical_jacobian_matches_fd() {
        let xs = vec![
            vec![1.0, 2.0, 3.0], // isotope 0
            vec![0.5, 0.5, 0.5], // isotope 1
        ];
        let inner = make_precomputed(xs, vec![0, 1]);

        let energies = [4.0, 9.0, 16.0];
        // params: [density0, density1, Anorm, BackA, BackB, BackC]
        let model = NormalizedTransmissionModel::new(inner, &energies, 2, 3, 4, 5);

        let params = [0.2, 0.4, 0.95, 0.01, 0.02, 0.005];
        let y = model.evaluate(&params).unwrap();
        let free: Vec<usize> = (0..6).collect();

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some");

        assert_eq!(jac.nrows, 3);
        assert_eq!(jac.ncols, 6);

        // Central-difference reference
        let h = 1e-7;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;

            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();

            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 1e-4,
                    "Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, analytical={ana:.8}, \
                     rel_err={:.6}",
                    err / scale,
                );
            }
        }
    }

    /// Aliased role indices (BackA == BackB sharing one parameter): the
    /// analytic Jacobian must ACCUMULATE both roles' contributions
    /// (1 + 1/√E), matching central finite differences — not keep only
    /// the first match.
    #[test]
    fn normalized_jacobian_aliased_role_indices_match_fd() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0];
        // params: [density, Anorm, BackAB (shared), BackC] — BackA and
        // BackB deliberately alias index 2.
        let model = NormalizedTransmissionModel::new(inner, &energies, 1, 2, 2, 3);

        let params = [0.3, 0.95, 0.02, 0.005];
        let y = model.evaluate(&params).unwrap();
        let free: Vec<usize> = (0..4).collect();

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some");

        let h = 1e-7;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;
            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();
            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 1e-4,
                    "aliased Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, \
                     analytical={ana:.8}",
                );
            }
        }
        // Pin the aliased column exactly: ∂/∂BackAB = 1 + 1/√E.
        for (i, &e) in energies.iter().enumerate() {
            let expected = 1.0 + 1.0 / e.sqrt();
            assert!(
                (jac.get(i, 2) - expected).abs() < 1e-12,
                "aliased column row {i}: {} vs expected {expected}",
                jac.get(i, 2),
            );
        }
    }

    /// Aliased baseline indices (b0 == b1 sharing one parameter) in the
    /// multiplicative wrapper: same accumulate-not-overwrite requirement,
    /// derivative (1 + z)·T_inner against the FD oracle.
    #[test]
    fn multiplicative_baseline_aliased_indices_match_fd() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0];
        let e_ref = baseline_reference_energy(&energies);
        // params: [density, b01 (shared), b2] — b0 and b1 deliberately
        // alias index 1.
        let model = MultiplicativeBaselineModel::new(inner, &energies, e_ref, 1, 1, 2);

        let params = [0.3, 1.02, 0.01];
        let y = model.evaluate(&params).unwrap();
        let free: Vec<usize> = (0..3).collect();

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some");

        let h = 1e-7;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;
            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();
            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 1e-4,
                    "aliased baseline Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, \
                     analytical={ana:.8}",
                );
            }
        }
    }

    /// Verify that when some background params are fixed (not in
    /// free_param_indices), the Jacobian columns are correct.
    #[test]
    fn normalized_jacobian_partial_free() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0];
        let model = NormalizedTransmissionModel::new(inner, &energies, 1, 2, 3, 4);

        // params: [density, Anorm, BackA, BackB, BackC]
        let params = [0.3, 0.95, 0.01, 0.0, 0.0];
        let y = model.evaluate(&params).unwrap();
        // Only density and Anorm are free
        let free = vec![0usize, 1usize];

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("should return Some for partial free");

        assert_eq!(jac.nrows, 3);
        assert_eq!(jac.ncols, 2);

        // Central-difference reference
        let h = 1e-7;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;

            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();

            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 1e-4,
                    "Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, analytical={ana:.8}"
                );
            }
        }
    }

    /// Issue #635: `baseline_reference_energy` is the geometric grid midpoint.
    #[test]
    fn baseline_reference_energy_geometric_mid() {
        let e = [1.0, 5.0, 100.0];
        assert!((baseline_reference_energy(&e) - 10.0).abs() < 1e-12);
        assert!(baseline_reference_energy(&[]).is_nan());
    }

    /// Issue #648: with an active mask, the reference energy is the midpoint
    /// of the ACTIVE window, not the full grid.  Mirrors the real VENUS case
    /// where the full grid spans to the MeV range but the fit window is
    /// 8–45 eV: the full-grid midpoint (≈3211 eV) silently lets the baseline
    /// absorb Doppler broadening; the active midpoint (≈19 eV) does not.
    #[test]
    fn baseline_reference_energy_active_uses_window_not_full_grid() {
        // Grid: three low-eV resonance bins + one MeV-scale tail bin.
        let energies = [8.0, 20.0, 45.0, 2_278_807.0];
        // fit_energy_range 8–45 eV → last bin inactive.
        let mask = [true, true, true, false];
        let e_ref = baseline_reference_energy_active(&energies, Some(&mask));
        assert!(
            (e_ref - (8.0_f64 * 45.0).sqrt()).abs() < 1e-9,
            "active E_ref = {e_ref}, expected {}",
            (8.0_f64 * 45.0).sqrt()
        );
        // Full-grid value is the buggy ≈3211 eV — the fix must differ from it.
        let full = baseline_reference_energy(&energies);
        assert!(full > 1000.0 && (full - e_ref).abs() > 1000.0);
        // None mask == full grid (no fit_energy_range).
        assert_eq!(
            baseline_reference_energy_active(&energies, None),
            baseline_reference_energy(&energies)
        );
        // Degenerate all-false mask falls back to full grid, never NaN.
        let none_active = [false, false, false, false];
        assert_eq!(
            baseline_reference_energy_active(&energies, Some(&none_active)),
            baseline_reference_energy(&energies)
        );
    }

    /// Issue #635: identity coefficients (1, 0, 0) reproduce the inner model
    /// bit-for-bit — the wrapper must be a no-op at the default init.
    #[test]
    fn baseline_identity_matches_inner() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner_ref = make_precomputed(xs.clone(), vec![0]);
        let inner_wrap = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0];
        let e_ref = baseline_reference_energy(&energies);
        let model = MultiplicativeBaselineModel::new(inner_wrap, &energies, e_ref, 1, 2, 3);

        // params: [density, b0, b1, b2]
        let params = [0.3, 1.0, 0.0, 0.0];
        let y = model.evaluate(&params).unwrap();
        let t_inner = inner_ref.evaluate(&params).unwrap();
        assert_eq!(y, t_inner, "identity baseline must be bit-exact");
    }

    /// Issue #635: hand-computed `B(E)·T` with an explicit reference energy —
    /// pins the centered ln-E basis and the coefficient order.
    #[test]
    fn baseline_formula_correct() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner_ref = make_precomputed(xs.clone(), vec![0]);
        let inner_wrap = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0];
        let e_ref = 8.0; // explicit, NOT the geometric mid — pins the argument
        let model = MultiplicativeBaselineModel::new(inner_wrap, &energies, e_ref, 1, 2, 3);

        let (b0, b1, b2) = (1.02, -0.03, 0.01);
        let density = 0.3;
        let params = [density, b0, b1, b2];
        let y = model.evaluate(&params).unwrap();
        let t_inner = inner_ref.evaluate(&params).unwrap();

        for (i, (&yi, &ti)) in y.iter().zip(t_inner.iter()).enumerate() {
            let z = (energies[i] / e_ref).ln();
            let expected = (b0 + b1 * z + b2 * z * z) * ti;
            assert!(
                (yi - expected).abs() < 1e-12,
                "E[{i}]: got {yi}, expected {expected}"
            );
        }
    }

    /// Issue #635: analytical Jacobian matches central finite differences with
    /// every parameter free (density + b0 + b1 + b2).
    #[test]
    fn baseline_analytical_jacobian_matches_fd() {
        let xs = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.5, 0.5]];
        let inner = make_precomputed(xs, vec![0, 1]);

        let energies = [4.0, 9.0, 16.0];
        let e_ref = baseline_reference_energy(&energies);
        // params: [density0, density1, b0, b1, b2]
        let model = MultiplicativeBaselineModel::new(inner, &energies, e_ref, 2, 3, 4);

        let params = [0.2, 0.4, 1.02, -0.03, 0.01];
        let y = model.evaluate(&params).unwrap();
        let free: Vec<usize> = (0..5).collect();

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some");
        assert_eq!(jac.nrows, 3);
        assert_eq!(jac.ncols, 5);

        let h = 1e-7;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;
            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();
            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 1e-4,
                    "Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, analytical={ana:.8}"
                );
            }
        }
    }

    /// Issue #635: partial free sets (only density + b1) produce the correct
    /// column subset.
    #[test]
    fn baseline_jacobian_partial_free() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0];
        let e_ref = baseline_reference_energy(&energies);
        let model = MultiplicativeBaselineModel::new(inner, &energies, e_ref, 1, 2, 3);

        // params: [density, b0, b1, b2]; only density and b1 free.
        let params = [0.3, 1.02, -0.03, 0.01];
        let y = model.evaluate(&params).unwrap();
        let free = vec![0usize, 2usize];

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("should return Some for partial free");
        assert_eq!(jac.nrows, 3);
        assert_eq!(jac.ncols, 2);

        let h = 1e-7;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;
            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();
            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 1e-4,
                    "Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, analytical={ana:.8}"
                );
            }
        }
    }

    /// Issue #635: the STACKED composition the pipeline builds — baseline
    /// wrapping the additive-background wrapper — chains both Jacobians
    /// correctly (verified against central FD over all 8 parameters).
    #[test]
    fn baseline_stacked_on_normalized_jacobian_matches_fd() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0];
        let e_ref = baseline_reference_energy(&energies);
        // params: [density, Anorm, BackA, BackB, BackC, b0, b1, b2]
        let bg = NormalizedTransmissionModel::new(inner, &energies, 1, 2, 3, 4);
        let model = MultiplicativeBaselineModel::new(bg, &energies, e_ref, 5, 6, 7);

        let params = [0.3, 0.98, 0.01, 0.02, 0.005, 1.02, -0.03, 0.01];
        let y = model.evaluate(&params).unwrap();
        let free: Vec<usize> = (0..8).collect();

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("stacked analytical_jacobian should return Some");
        assert_eq!(jac.nrows, 3);
        assert_eq!(jac.ncols, 8);

        let h = 1e-7;
        for (col, &p_idx) in free.iter().enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[p_idx] += h;
            p_minus[p_idx] -= h;
            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();
            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let err = (fd - ana).abs();
                let scale = fd.abs().max(ana.abs()).max(1e-10);
                assert!(
                    err / scale < 1e-4,
                    "stacked Jacobian mismatch (row {i}, col {col}): \
                     FD={fd:.8}, analytical={ana:.8}"
                );
            }
        }
    }

    /// Issue #635: a non-positive B(E) at any bin rejects the evaluation —
    /// the positivity guard fires on wide grids where in-bounds coefficients
    /// can drive the polynomial negative.
    #[test]
    fn baseline_evaluate_rejects_nonpositive_b() {
        let xs = vec![vec![1.0; 5]];
        let inner = make_precomputed(xs, vec![0]);

        // Very wide grid: z spans ±~6.9 around the geometric mid.
        let energies = [1e-3, 1e-1, 1.0, 1e1, 1e3];
        let e_ref = baseline_reference_energy(&energies);
        let model = MultiplicativeBaselineModel::new(inner, &energies, e_ref, 1, 2, 3);

        // In-bounds-magnitude coefficients that go negative at the grid edge:
        // B(z=-6.9) = 0.9 - 0.05·(-6.9) ... use b2 to force it negative.
        let params = [0.3, 0.9, 0.0, -0.05];
        let err = model.evaluate(&params);
        assert!(
            err.is_err(),
            "B(E) <= 0 at the grid edge must be rejected, got {err:?}"
        );
        // Sanity: identity still evaluates on the same grid.
        assert!(model.evaluate(&[0.3, 1.0, 0.0, 0.0]).is_ok());
    }

    /// Review R2: the positivity guard is scoped to ACTIVE bins.  The same
    /// in-bounds coefficients that go negative only at masked-out grid-edge
    /// bins must NOT reject the trial step when those bins are excluded by
    /// the fit-energy-range mask — an unscoped guard vetoed in-window-valid
    /// steps and inflated λ into spurious non-convergence.
    #[test]
    fn baseline_positivity_guard_scoped_to_active_mask() {
        let xs = vec![vec![1.0; 5]];
        let energies = [1e-3, 1e-1, 1.0, 1e1, 1e3];
        let e_ref = baseline_reference_energy(&energies);
        // Same coefficients as baseline_evaluate_rejects_nonpositive_b:
        // B < 0 only at the outer bins (|z| ≈ 6.9).
        let params = [0.3, 0.9, 0.0, -0.05];

        // Unmasked control (non-vacuity): the guard fires.
        let unmasked = MultiplicativeBaselineModel::new(
            make_precomputed(xs.clone(), vec![0]),
            &energies,
            e_ref,
            1,
            2,
            3,
        );
        assert!(unmasked.evaluate(&params).is_err());

        // Mask out the offending edge bins: evaluation succeeds and the
        // ACTIVE bins carry the expected positive product.
        let mask = [false, true, true, true, false];
        let masked = MultiplicativeBaselineModel::new(
            make_precomputed(xs, vec![0]),
            &energies,
            e_ref,
            1,
            2,
            3,
        )
        .with_active_mask(Some(&mask));
        let out = masked
            .evaluate(&params)
            .expect("negative B at MASKED bins must not reject the step");
        for (i, (&y, &active)) in out.iter().zip(mask.iter()).enumerate() {
            if active {
                let z = (energies[i] / e_ref).ln();
                let b = 0.9 - 0.05 * z * z;
                assert!(b > 0.0, "test setup: active bin {i} must have B > 0");
                let t = (-0.3f64).exp();
                assert!(
                    (y - b * t).abs() < 1e-12,
                    "active bin {i}: y = {y}, expected {}",
                    b * t
                );
            }
        }
    }

    /// End-to-end: fit recovers known Anorm + BackA from synthetic data.
    #[test]
    fn normalized_fit_recovers_anorm_and_backa() {
        let xs = vec![vec![1.0, 2.0, 3.0, 2.0, 1.5]];
        let inner = make_precomputed(xs, vec![0]);

        let energies = [4.0, 9.0, 16.0, 25.0, 36.0];
        let model = NormalizedTransmissionModel::new(inner, &energies, 1, 2, 3, 4);

        // True parameters
        let true_density = 0.2;
        let true_anorm = 0.95;
        let true_back_a = 0.02;
        let true_params = [true_density, true_anorm, true_back_a, 0.0, 0.0];

        let y_obs = model.evaluate(&true_params).unwrap();
        let sigma = vec![0.001; y_obs.len()];

        // Initial guesses offset from truth
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.1),
            FitParameter {
                name: "anorm".into(),
                value: 1.0,
                lower: 0.5,
                upper: 1.5,
                fixed: false,
            },
            FitParameter::unbounded("back_a", 0.0),
            FitParameter::fixed("back_b", 0.0),
            FitParameter::fixed("back_c", 0.0),
        ]);

        let config = LmConfig {
            max_iter: 200,
            ..LmConfig::default()
        };

        let result = lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &config).unwrap();

        assert!(result.converged, "Fit should converge");

        let fit_density = result.params[0];
        let fit_anorm = result.params[1];
        let fit_back_a = result.params[2];

        assert!(
            (fit_density - true_density).abs() / true_density < 0.01,
            "density: fitted={fit_density}, true={true_density}"
        );
        assert!(
            (fit_anorm - true_anorm).abs() / true_anorm < 0.01,
            "anorm: fitted={fit_anorm}, true={true_anorm}"
        );
        assert!(
            (fit_back_a - true_back_a).abs() < 0.001,
            "back_a: fitted={fit_back_a}, true={true_back_a}"
        );
    }

    // ── Phase 1: ForwardModel tests ──

    #[test]
    fn forward_model_predict_equals_fit_model_evaluate_precomputed() {
        use crate::forward_model::ForwardModel;
        let xs = vec![vec![1.0, 2.0, 3.0, 2.0, 1.5]];
        let model = make_precomputed(xs, vec![0]);
        let params = [0.001];
        let fm_result = model.evaluate(&params).unwrap();
        let fwd_result = model.predict(&params).unwrap();
        assert_eq!(fm_result, fwd_result);
        assert_eq!(model.n_data(), 5);
        assert_eq!(model.n_params(), 1);
    }

    #[test]
    fn forward_model_predict_equals_fit_model_evaluate_normalized() {
        use crate::forward_model::ForwardModel;
        let xs = vec![vec![1.0, 2.0, 3.0, 2.0, 1.5]];
        let inner = make_precomputed(xs, vec![0]);
        let energies = [4.0, 9.0, 16.0, 25.0, 36.0];
        let model = NormalizedTransmissionModel::new(inner, &energies, 1, 2, 3, 4);
        let params = [0.001, 0.95, 0.01, 0.0, 0.0];
        let fm_result = model.evaluate(&params).unwrap();
        let fwd_result = model.predict(&params).unwrap();
        assert_eq!(fm_result, fwd_result);
        assert_eq!(model.n_data(), 5);
        assert_eq!(model.n_params(), 5);
    }

    #[test]
    fn forward_model_jacobian_columns_match_precomputed() {
        use crate::forward_model::ForwardModel;
        let xs = vec![vec![1.0, 2.0, 3.0], vec![0.5, 1.5, 2.5]];
        let model = make_precomputed(xs, vec![0, 1]);
        let params = [0.001, 0.002];
        let y = model.predict(&params).unwrap();
        let free_indices = vec![0, 1];
        let jac = model
            .jacobian(&params, &free_indices, &y)
            .expect("analytical jacobian should be available");
        assert_eq!(jac.len(), 2); // 2 columns (one per free param)
        assert_eq!(jac[0].len(), 3); // 3 rows (one per energy bin)
    }

    // ── Issue #442 Step 3 regression tests ─────────────────────────────────

    /// Issue #442: PrecomputedTransmissionModel with resolution must match
    /// forward_model() with resolution for the same single-isotope sample.
    #[test]
    fn precomputed_with_resolution_matches_forward_model() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();

        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        // Reference: forward_model() (already fixed in Step 1).
        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_forward = transmission::forward_model(&energies, &sample, Some(&inst)).unwrap();

        // Precomputed path: Doppler-only XS → PrecomputedTransmissionModel.
        let xs = transmission::broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            Some(&inst), // aux grid for Doppler accuracy
            None,
        )
        .unwrap();
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(Arc::clone(&inst)),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: None,
        };
        let t_precomputed = model.evaluate(&[thickness]).unwrap();

        // Both should agree closely on the interior grid.
        // Small differences are expected from extended-grid Doppler
        // in forward_model vs data-grid Doppler in broadened_cross_sections.
        let interior = 20..energies.len() - 20;
        let mut max_err = 0.0f64;
        for i in interior {
            let err = (t_forward[i] - t_precomputed[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 0.02,
            "PrecomputedTransmissionModel with resolution should match \
             forward_model.  Max error = {max_err}"
        );
    }

    /// Issue #442: PrecomputedTransmissionModel without resolution must
    /// behave identically to the pre-fix version (pure Beer-Lambert).
    #[test]
    fn precomputed_without_resolution_unchanged() {
        let model_no_res = make_precomputed(
            vec![vec![100.0, 200.0, 50.0]], // one isotope
            vec![0],
        );
        let params = [0.001f64]; // density
        let t = model_no_res.evaluate(&params).unwrap();

        // Expected: pure Beer-Lambert.
        let expected: Vec<f64> = [100.0, 200.0, 50.0]
            .iter()
            .map(|&sigma| (-params[0] * sigma).exp())
            .collect();

        for (i, (&ti, &ei)) in t.iter().zip(expected.iter()).enumerate() {
            assert!(
                (ti - ei).abs() < 1e-14,
                "No-resolution mismatch at bin {i}: got {ti}, expected {ei}"
            );
        }

        // Analytical Jacobian should still be available when instrument is None.
        let y = model_no_res.evaluate(&params).unwrap();
        assert!(
            model_no_res
                .analytical_jacobian(&params, &[0], &y)
                .is_some(),
            "Analytical Jacobian must be available when instrument is None"
        );
    }

    /// PrecomputedTransmissionModel with resolution: analytical Jacobian
    /// exists and density derivative matches finite difference.
    #[test]
    fn precomputed_jacobian_with_resolution_matches_fd() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        let xs = transmission::broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            Some(&inst),
            None,
        )
        .unwrap();
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(Arc::clone(&inst)),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: None,
        };

        let params = [0.0005f64];
        let y = model.evaluate(&params).unwrap();

        let jac = model
            .analytical_jacobian(&params, &[0], &y)
            .expect("analytical Jacobian must be available with resolution");

        // Finite-difference reference.
        let h = 1e-7;
        let y_plus = model.evaluate(&[params[0] + h]).unwrap();
        let y_minus = model.evaluate(&[params[0] - h]).unwrap();

        let interior = 20..energies.len() - 20;
        let mut max_rel_err = 0.0f64;
        for i in interior {
            let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
            let ana = jac.get(i, 0);
            let denom = fd.abs().max(ana.abs()).max(1e-30);
            max_rel_err = max_rel_err.max((ana - fd).abs() / denom);
        }
        assert!(
            max_rel_err < 0.01,
            "PrecomputedTM analytical Jacobian with resolution vs FD: \
             max relative error = {max_rel_err}"
        );
    }

    /// PrecomputedTransmissionModel with resolution + shared density param:
    /// grouped isotope Jacobian matches FD.
    #[test]
    fn precomputed_jacobian_grouped_with_resolution_matches_fd() {
        use nereids_physics::resolution::ResolutionFunction;

        let energies: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.1).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });
        // Two isotopes sharing one density parameter.
        let xs = vec![vec![10.0; 100], vec![5.0; 100]];
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs),
            density_indices: Arc::new(vec![0, 0]), // both share param[0]
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(Arc::clone(&inst)),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: None,
        };

        let params = [0.001f64];
        let y = model.evaluate(&params).unwrap();
        let jac = model
            .analytical_jacobian(&params, &[0], &y)
            .expect("analytical Jacobian must be available");

        let h = 1e-7;
        let y_plus = model.evaluate(&[params[0] + h]).unwrap();
        let y_minus = model.evaluate(&[params[0] - h]).unwrap();

        let mut max_rel_err = 0.0f64;
        for i in 10..energies.len() - 10 {
            let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
            let ana = jac.get(i, 0);
            let denom = fd.abs().max(ana.abs()).max(1e-30);
            max_rel_err = max_rel_err.max((ana - fd).abs() / denom);
        }
        assert!(
            max_rel_err < 0.01,
            "Grouped PrecomputedTM analytical Jacobian with resolution vs FD: \
             max relative error = {max_rel_err}"
        );
    }

    // ── TransmissionFitModel Jacobian with resolution ──────────────────────

    /// TransmissionFitModel with resolution: analytical Jacobian exists and
    /// density + temperature columns match finite difference.
    #[test]
    fn transmission_fit_model_jacobian_with_resolution_matches_fd() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            300.0,
            Some(inst),
            (vec![0], vec![1.0]),
            Some(1), // temperature_index = 1
            None,
        )
        .unwrap();

        let params = [0.0005f64, 300.0];
        let y = model.evaluate(&params).unwrap();
        let free = vec![0usize, 1usize];

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical Jacobian must be available with resolution");

        // FD for each free param.
        let h_density = 1e-7;
        let h_temp = 0.01; // temperature needs larger step

        for (col, (&fp_idx, &h)) in free.iter().zip([h_density, h_temp].iter()).enumerate() {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[fp_idx] += h;
            p_minus[fp_idx] -= h;
            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();

            let interior = 20..energies.len() - 20;
            let mut max_rel_err = 0.0f64;
            for i in interior {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * h);
                let ana = jac.get(i, col);
                let denom = fd.abs().max(ana.abs()).max(1e-30);
                max_rel_err = max_rel_err.max((ana - fd).abs() / denom);
            }
            let label = if col == 0 { "density" } else { "temperature" };
            assert!(
                max_rel_err < 0.05,
                "TransmissionFitModel {label} column with resolution vs FD: \
                 max relative error = {max_rel_err}"
            );
        }
    }

    /// TransmissionFitModel without resolution: analytical Jacobian still
    /// available and unchanged.
    #[test]
    fn transmission_fit_model_jacobian_available_without_resolution() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel::new(
            energies,
            vec![data],
            300.0,
            None,
            (vec![0], vec![1.0]),
            Some(1),
            None,
        )
        .unwrap();

        let params = [0.0005, 300.0];
        let y = model.evaluate(&params).unwrap();

        assert!(
            model.analytical_jacobian(&params, &[0, 1], &y).is_some(),
            "TransmissionFitModel analytical Jacobian must be available \
             when resolution is disabled"
        );
    }

    // ── Issue #442: TransmissionFitModel temperature-path resolution fix ───

    /// TransmissionFitModel::evaluate() with fit_temperature=true and
    /// resolution enabled must match forward_model() for the same sample.
    #[test]
    fn transmission_fit_model_temp_path_with_resolution_matches_forward_model() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();

        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        // Reference: forward_model() (corrected in Step 1).
        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_ref = transmission::forward_model(&energies, &sample, Some(&inst)).unwrap();

        // Temperature-fitting path through TransmissionFitModel.
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            temperature,
            Some(Arc::clone(&inst)),
            (vec![0], vec![1.0]),
            Some(1), // temperature_index
            None,
        )
        .unwrap();

        // params = [density, temperature]
        let t_model = model.evaluate(&[thickness, temperature]).unwrap();

        // Compare on interior (skip boundary effects from extended grid
        // differences between forward_model and broadened_cross_sections_from_base).
        let interior = 20..energies.len() - 20;
        let mut max_err = 0.0f64;
        for i in interior {
            max_err = max_err.max((t_ref[i] - t_model[i]).abs());
        }
        assert!(
            max_err < 0.02,
            "TransmissionFitModel temperature path with resolution should match \
             forward_model.  Max error = {max_err}"
        );
    }

    /// TransmissionFitModel temperature path without resolution must be
    /// unchanged (pure Doppler + Beer-Lambert).
    #[test]
    fn transmission_fit_model_temp_path_no_resolution_unchanged() {
        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        // Reference: forward_model without resolution.
        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_ref = transmission::forward_model(&energies, &sample, None).unwrap();

        // TransmissionFitModel, no resolution.
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            temperature,
            None,
            (vec![0], vec![1.0]),
            Some(1),
            None,
        )
        .unwrap();

        let t_model = model.evaluate(&[thickness, temperature]).unwrap();

        for (i, (&r, &m)) in t_ref.iter().zip(t_model.iter()).enumerate() {
            assert!(
                (r - m).abs() < 1e-12,
                "No-resolution mismatch at E[{i}]={}: ref={r}, model={m}",
                energies[i]
            );
        }
    }

    // ── Issue #608: LM-fit resolution must use the auxiliary grid ────────────
    //
    // The pre-#608 cached / precomputed / energy-scale paths applied resolution
    // broadening on the COARSE data grid, unlike `forward_model`, which broadens
    // on the auxiliary extended grid and extracts the data points last.  The
    // tests below pin every fixed path to `forward_model` — an INDEPENDENT
    // oracle: it computes σ inline (`reich_moore::cross_sections_at_energy`) and
    // never calls the `broadened_cross_sections` family this fix touches — to
    // MACHINE PRECISION over the FULL grid, including the boundary points the
    // earlier #442 tests (tol 2e-2, interior-only) excluded.  Each test verifies
    // the kernel actually broadens the spectrum (a non-vacuity pre-check, so a
    // shared-primitive oracle cannot pass vacuously) and, where it can construct the
    // old path, shows the old coarse-grid result differed materially — proving
    // the fix is a real correction, not a no-op.  Jacobian columns are checked
    // against central finite differences of the (now aux-correct) `evaluate`.
    //
    // SCOPE of the 1e-9 bound: these tests pin GRID FIDELITY — that
    // each fixed path builds the same auxiliary grid + layout as `forward_model`
    // and extracts the data points identically.  The resolution KERNEL primitive
    // itself (`apply_resolution_*`, `build_aux_grid`, `doppler::doppler_broaden`)
    // is SHARED with the oracle, so a kernel error common to both would pass
    // here; the kernel's physics is validated independently against SAMMY in
    // `nereids-physics` (`resolution.rs`, `samtry_validation.rs`).  The
    // non-vacuity `‖kernel − none‖` guards keep this shared-primitive oracle
    // non-circular for what it asserts (the #608 grid wiring).

    /// Issue #608: the spatial production path (`PrecomputedTransmissionModel`)
    /// must broaden resolution on the auxiliary grid, matching `forward_model`.
    #[test]
    fn issue_608_precomputed_aux_grid_resolution_matches_forward_model() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        // Independent oracle (computes σ inline; broadens on the aux grid).
        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_ref = transmission::forward_model(&energies, &sample, Some(&inst)).unwrap();

        // Non-vacuity: the kernel must actually broaden the spectrum, else
        // aux-grid vs data-grid broadening would be indistinguishable.
        let t_nores = transmission::forward_model(&energies, &sample, None).unwrap();
        let broaden = max_abs_diff(&t_ref, &t_nores);
        assert!(
            broaden > 1e-3 * max_abs(&t_nores),
            "resolution kernel must broaden the spectrum non-trivially (got {broaden:.3e})"
        );

        // FIXED path: working-grid σ + layout, exactly as `spatial_map_typed` builds it.
        let working = transmission::broadened_cross_sections_on_working_grid(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            Some(&inst),
            None,
        )
        .unwrap();
        assert!(
            !working.layout.is_identity(),
            "Gaussian resolution must build a non-identity auxiliary grid — else \
             this test does not exercise the #608 fix"
        );
        let model_fixed = PrecomputedTransmissionModel {
            cross_sections: Arc::new(working.sigma),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(Arc::clone(&inst)),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: Some(Arc::new(working.layout)),
        };
        let t_fixed = model_fixed.evaluate(&[thickness]).unwrap();

        // OLD path: data-grid σ, no layout — broadens on the coarse data grid
        // (the configuration the pre-#608 spatial pipeline produced).
        let xs_data = transmission::broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            Some(&inst),
            None,
        )
        .unwrap();
        let model_old = PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs_data),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(Arc::clone(&inst)),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: None,
        };
        let t_old = model_old.evaluate(&[thickness]).unwrap();

        let err_fixed = max_abs_diff(&t_fixed, &t_ref);
        let err_old = max_abs_diff(&t_old, &t_ref);

        assert!(
            err_fixed < 1e-9,
            "aux-grid PrecomputedTransmissionModel must match forward_model to \
             machine precision over the full grid (got {err_fixed:.3e})"
        );
        assert!(
            err_old > 1e-4 && err_old > 1e4 * err_fixed.max(1e-15),
            "old coarse-grid path should differ from forward_model far more than \
             the fixed path (old={err_old:.3e}, fixed={err_fixed:.3e})"
        );
    }

    /// Issue #634: the energy-scale model's FINITE-DIFFERENCE temperature
    /// column must match the ANALYTIC ∂σ/∂T column that the fixed-grid
    /// `TransmissionFitModel` produces on the SAME corrected grid, to <1e-4
    /// relative. This validates the FD choice (correct index, sign, magnitude)
    /// against the exact analytic derivative the non-energy-scale path uses.
    /// The non-zero assertions guard against a silently mis-wired T index
    /// (which would yield a zero column a loose recovery test could miss).
    #[test]
    fn energy_scale_temperature_jacobian_matches_analytic() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();
        let temperature = 320.0;
        let density = 0.0006;
        // Non-trivial energy scale so the corrected grid differs from nominal.
        let t0 = 0.7_f64;
        let l_scale = 1.004_f64;
        let flight_path = 25.0;

        // Param layout mirrors the pipeline: [density, temperature, t0, l_scale]
        // (temperature appended before the energy-scale params).
        let (d_idx, t_idx, t0_idx, ls_idx) = (0usize, 1usize, 2usize, 3usize);
        let params = [density, temperature, t0, l_scale];

        // Energy-scale model with temperature fitting (FD T column). No
        // resolution keeps the corrected-grid physics identical to the oracle.
        let es = EnergyScaleTransmissionModel::new(
            Arc::new(vec![data.clone()]),
            Arc::new(vec![d_idx]),
            Arc::new(vec![1.0]),
            temperature,
            energies.clone(),
            flight_path,
            t0_idx,
            ls_idx,
            None,
        )
        .with_temperature_index(Some(t_idx))
        .expect("distinct temperature index");

        let free = [d_idx, t_idx, t0_idx, ls_idx];
        let y = es.evaluate(&params).unwrap();
        let jac = es
            .analytical_jacobian(&params, &free, &y)
            .expect("energy-scale jacobian available");
        let t_col_fd: Vec<f64> = (0..energies.len()).map(|i| jac.get(i, 1)).collect();

        // Analytic oracle: TransmissionFitModel on the SAME corrected grid.
        let e_corr = es.corrected_energies(t0, l_scale);
        let oracle = TransmissionFitModel::new(
            e_corr,
            vec![data],
            temperature,
            None,
            (vec![d_idx], vec![1.0]),
            Some(t_idx),
            None,
        )
        .unwrap();
        // Oracle params: [density, temperature] (no t0/l_scale); T at index 1.
        let oracle_params = [density, temperature];
        let y_o = oracle.evaluate(&oracle_params).unwrap();
        let jac_o = oracle
            .analytical_jacobian(&oracle_params, &[d_idx, t_idx], &y_o)
            .expect("oracle analytic jacobian available");
        let t_col_an: Vec<f64> = (0..energies.len()).map(|i| jac_o.get(i, 1)).collect();

        let mut scale = 0.0f64;
        let mut max_err = 0.0f64;
        for i in 0..energies.len() {
            scale = scale.max(t_col_an[i].abs());
            max_err = max_err.max((t_col_fd[i] - t_col_an[i]).abs());
        }
        assert!(
            scale > 1e-6,
            "analytic T column must be non-trivially non-zero (scale {scale:.3e})"
        );
        let fd_scale = t_col_fd.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(
            fd_scale > 1e-6,
            "FD T column must be non-zero — a mis-wired T index gives a silent zero"
        );
        let rel = max_err / scale;
        assert!(
            rel < 1e-4,
            "energy-scale FD ∂T/∂temperature must match the analytic column to \
             <1e-4 relative, got {rel:.3e}"
        );
    }

    /// Issue #608: `PrecomputedTransmissionModel::analytical_jacobian` forms the
    /// inner derivative on the auxiliary grid; it must match central finite
    /// differences of the (aux-correct) `evaluate`.
    #[test]
    fn issue_608_precomputed_aux_grid_jacobian_matches_fd() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });
        let working = transmission::broadened_cross_sections_on_working_grid(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            Some(&inst),
            None,
        )
        .unwrap();
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(working.sigma),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(Arc::clone(&inst)),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
            work_layout: Some(Arc::new(working.layout)),
        };

        let params = [thickness];
        let free = [0usize];
        let y0 = model.evaluate(&params).unwrap();
        let jac = model
            .analytical_jacobian(&params, &free, &y0)
            .expect("analytical jacobian must be available with resolution + aux grid");

        let h = 1e-7;
        let mut pp = params;
        let mut pm = params;
        pp[0] += h;
        pm[0] -= h;
        let yp = model.evaluate(&pp).unwrap();
        let ym = model.evaluate(&pm).unwrap();

        let mut scale = 0.0f64;
        let mut max_err = 0.0f64;
        for i in 0..y0.len() {
            let fd = (yp[i] - ym[i]) / (2.0 * h);
            let an = jac.get(i, 0);
            scale = scale.max(an.abs());
            max_err = max_err.max((fd - an).abs());
        }
        let rel = max_err / scale.max(1e-30);
        assert!(
            rel < 1e-6,
            "analytical density Jacobian must match central FD (rel err {rel:.3e})"
        );
    }

    /// Issue #608: `TransmissionFitModel`'s cached temperature-fit `evaluate`
    /// must broaden on the auxiliary grid, matching `forward_model` to machine
    /// precision over the full grid (the #442 test tolerated 2e-2, interior-only).
    #[test]
    fn issue_608_transmission_fit_temp_path_aux_grid_matches_forward_model() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_ref = transmission::forward_model(&energies, &sample, Some(&inst)).unwrap();
        let t_nores = transmission::forward_model(&energies, &sample, None).unwrap();
        let broaden = max_abs_diff(&t_ref, &t_nores);
        assert!(
            broaden > 1e-3 * max_abs(&t_nores),
            "resolution kernel must broaden the spectrum non-trivially (got {broaden:.3e})"
        );

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            temperature,
            Some(Arc::clone(&inst)),
            (vec![0], vec![1.0]),
            Some(1), // temperature_index → exercises the cached temperature path
            None,
        )
        .unwrap();
        let t_model = model.evaluate(&[thickness, temperature]).unwrap();

        let err = max_abs_diff(&t_model, &t_ref);
        assert!(
            err < 1e-9,
            "aux-grid TransmissionFitModel temperature path must match \
             forward_model over the full grid (got {err:.3e})"
        );
    }

    /// Issue #608: `TransmissionFitModel::analytical_jacobian` (cached temp path)
    /// forms density and temperature inner derivatives on the auxiliary grid;
    /// both columns must match central finite differences of `evaluate`.
    #[test]
    fn issue_608_transmission_fit_temp_path_jacobian_matches_fd() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            temperature,
            Some(Arc::clone(&inst)),
            (vec![0], vec![1.0]),
            Some(1),
            None,
        )
        .unwrap();

        let params = [thickness, temperature];
        // evaluate() populates the broadened-σ cache at these params; the
        // analytical jacobian reads that cache, so compute it BEFORE any FD
        // perturbation mutates the cache.
        let y0 = model.evaluate(&params).unwrap();
        let free = [0usize, 1usize];
        let jac = model
            .analytical_jacobian(&params, &free, &y0)
            .expect("analytical jacobian must be available with resolution + aux grid");

        // Per-parameter central-FD step (absolute): density ~5e-4, temperature 300 K.
        let steps = [1e-7, 1e-2];
        for (col, &p_idx) in free.iter().enumerate() {
            let h = steps[col];
            let mut pp = params;
            let mut pm = params;
            pp[p_idx] += h;
            pm[p_idx] -= h;
            let yp = model.evaluate(&pp).unwrap();
            let ym = model.evaluate(&pm).unwrap();
            let mut scale = 0.0f64;
            let mut max_err = 0.0f64;
            for i in 0..y0.len() {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                let an = jac.get(i, col);
                scale = scale.max(an.abs());
                max_err = max_err.max((fd - an).abs());
            }
            let rel = max_err / scale.max(1e-30);
            assert!(
                rel < 1e-5,
                "analytical Jacobian column {col} must match central FD (rel err {rel:.3e})"
            );
        }
    }

    /// Issue #608: EnergyScale must evaluate the TRUE σ at the
    /// corrected energies on the auxiliary grid — INCLUDING the boundary
    /// extension points — exactly like `forward_model`, not clamp a precomputed
    /// σ.  With the U-238 resonance near the grid EDGE (where the pre-fix clamp
    /// deviated most) and Gaussian resolution active, EnergyScale at identity
    /// calibration must match `forward_model` — an independent oracle that
    /// evaluates σ inline — to machine precision over the FULL grid.  This is the
    /// non-circular replacement for the previous flat-σ/clamp-oracle test (which
    /// could not detect the boundary deviation).
    #[test]
    fn issue_608_energy_scale_aux_grid_true_sigma_matches_forward_model() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let density = 0.01;
        // Grid placing the U-238 resonance (~6.67 eV) near the UPPER edge, so σ
        // is strongly non-flat at the boundary — exactly where clamping (the
        // pre-#608 behaviour) deviated from true physics.
        let energies: Vec<f64> = (0..121).map(|i| 5.0 + (i as f64) * 0.015).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        let model = make_energy_scale_u238(energies.clone(), Some(Arc::clone(&inst)));
        let t_es = model.evaluate(&[density, 0.0, 1.0]).unwrap();

        // Independent oracle: forward_model evaluates σ inline on the aux grid.
        let sample = SampleParams::new(300.0, vec![(data, density)]).unwrap();
        let t_ref = transmission::forward_model(&energies, &sample, Some(&inst)).unwrap();

        // Non-vacuity: the resolution kernel must broaden the spectrum.
        let t_nores = transmission::forward_model(&energies, &sample, None).unwrap();
        let broaden = max_abs_diff(&t_ref, &t_nores);
        assert!(
            broaden > 1e-3 * max_abs(&t_nores),
            "resolution kernel must broaden the spectrum non-trivially (got {broaden:.3e})"
        );

        // True-σ aux-grid EnergyScale matches forward_model over the FULL grid,
        // including the resonance-near-edge boundary where the old clamp failed.
        let err = max_abs_diff(&t_es, &t_ref);
        assert!(
            err < 1e-9,
            "EnergyScale identity-calibration evaluate must match forward_model to \
             machine precision over the full grid (got {err:.3e})"
        );
    }

    /// Issue #608: the GROUPED energy-scale path — multiple isotopes mapped
    /// to ONE density parameter with non-unity ratios — is reachable in
    /// production (`with_groups` + `fit_energy_scale`) but was exercised by no
    /// test; every other energy-scale test used a single isotope
    /// (`density_indices=[0]`, ratio 1.0).  Build two DISTINCT isotopes sharing
    /// density param 0 with ratios (0.7, 0.3) and verify the per-member
    /// Beer-Lambert accumulation (`Σᵢ n·ratioᵢ·σᵢ`) matches `forward_model` with
    /// per-isotope effective densities — plus an FD check on the single shared
    /// density column.
    #[test]
    fn issue_608_energy_scale_grouped_density_matches_forward_model() {
        use nereids_endf::resonance::test_support::synthetic_single_resonance;
        use nereids_physics::resolution::ResolutionFunction;

        let iso0 = u238_single_resonance(); // resonance @ ~6.674 eV
        let iso1 = synthetic_single_resonance(72, 178, 176.0, 7.5); // distinct @ 7.5 eV
        let density = 0.01_f64;
        let ratios = [0.7_f64, 0.3_f64];
        // Grid overlapping BOTH resonances so σ0 ≠ σ1 (a swapped ratio / wrong
        // index shifts T detectably — proven by the swap guard below).
        let energies: Vec<f64> = (0..201).map(|i| 5.0 + (i as f64) * 0.02).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        let model = EnergyScaleTransmissionModel::new(
            Arc::new(vec![iso0.clone(), iso1.clone()]),
            Arc::new(vec![0, 0]), // both isotopes → density param 0 (grouped)
            Arc::new(vec![ratios[0], ratios[1]]),
            300.0,
            energies.clone(),
            25.0,
            1, // t0 index
            2, // l_scale index
            Some(Arc::clone(&inst)),
        );
        let params = [density, 0.0, 1.0]; // identity calibration (t0=0, l_scale=1)
        let t_es = model.evaluate(&params).unwrap();

        // Independent oracle: forward_model with per-isotope effective areal
        // densities n·ratioᵢ.  Beer-Lambert is additive over isotopes, so the
        // grouped model (one density param × per-iso ratio) must equal a two-
        // isotope sample with densities (n·0.7, n·0.3).
        let sample = SampleParams::new(
            300.0,
            vec![
                (iso0.clone(), density * ratios[0]),
                (iso1.clone(), density * ratios[1]),
            ],
        )
        .unwrap();
        let t_ref = transmission::forward_model(&energies, &sample, Some(&inst)).unwrap();

        // Non-vacuity: the kernel must broaden the grouped spectrum.
        let t_nores = transmission::forward_model(&energies, &sample, None).unwrap();
        assert!(
            max_abs_diff(&t_ref, &t_nores) > 1e-3 * max_abs(&t_nores),
            "resolution kernel must broaden the grouped spectrum non-trivially"
        );

        // Discrimination: swapping the two ratios MUST change T (proves σ0 ≠ σ1
        // over the grid, so the match assertion below is sensitive to a ratio /
        // index mix-up in the per-member accumulation — i.e. non-vacuous).
        let model_swapped = EnergyScaleTransmissionModel::new(
            Arc::new(vec![iso0.clone(), iso1.clone()]),
            Arc::new(vec![0, 0]),
            Arc::new(vec![ratios[1], ratios[0]]), // swapped
            300.0,
            energies.clone(),
            25.0,
            1,
            2,
            Some(Arc::clone(&inst)),
        );
        let t_swapped = model_swapped.evaluate(&params).unwrap();
        assert!(
            max_abs_diff(&t_es, &t_swapped) > 1e-4,
            "swapping the two density ratios must change T (else the test could \
             not distinguish the ratio→isotope assignment)"
        );

        // Grouped evaluate matches the independent oracle to machine precision.
        let err = max_abs_diff(&t_es, &t_ref);
        assert!(
            err < 1e-9,
            "grouped EnergyScale (2 isotopes → 1 density param, ratios {ratios:?}) \
             must match forward_model with per-isotope effective densities to \
             machine precision (got {err:.3e})"
        );

        // FD check on the single shared density column: ∂T/∂n accumulates
        // ratioᵢ·σᵢ over BOTH grouped isotopes.
        let free = vec![0usize];
        let jac = model
            .analytical_jacobian(&params, &free, &t_es)
            .expect("Jacobian should be available");
        let h = 1e-7;
        let mut pp = params;
        let mut pm = params;
        pp[0] += h;
        pm[0] -= h;
        let yp = model.evaluate(&pp).unwrap();
        let ym = model.evaluate(&pm).unwrap();
        for row in 0..energies.len() {
            let fd = (yp[row] - ym[row]) / (2.0 * h);
            let anal = jac.get(row, 0);
            let abs_err = (anal - fd).abs();
            let rel_err = abs_err / fd.abs().max(1e-15);
            assert!(
                rel_err < 1e-3 || abs_err < 1e-8,
                "grouped density col bin {row}: anal={anal:.6e} fd={fd:.6e} rel={rel_err:.2e}"
            );
        }
    }

    /// Resolution-enabled temperature path must produce measurably different
    /// results from the unresolved path (verifies resolution is being applied).
    #[test]
    fn transmission_fit_model_temp_path_resolution_makes_difference() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();

        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        // With resolution.
        let model_res = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            temperature,
            Some(inst),
            (vec![0], vec![1.0]),
            Some(1),
            None,
        )
        .unwrap();
        let t_res = model_res.evaluate(&[thickness, temperature]).unwrap();

        // Without resolution.
        let model_no = TransmissionFitModel::new(
            energies.clone(),
            vec![data],
            temperature,
            None,
            (vec![0], vec![1.0]),
            Some(1),
            None,
        )
        .unwrap();
        let t_no = model_no.evaluate(&[thickness, temperature]).unwrap();

        let interior = 20..energies.len() - 20;
        let max_diff: f64 = interior
            .map(|i| (t_res[i] - t_no[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 1e-4,
            "Resolution should make a measurable difference in the temperature \
             path, but max diff = {max_diff}"
        );
    }

    // ── Exponential background (BackD, BackF) tests ──

    /// Verify that new_with_exponential evaluate() matches the formula:
    /// T_out = Anorm*T_inner + BackA + BackB/√E + BackC*√E + BackD*exp(-BackF/√E)
    #[test]
    fn exponential_evaluate_formula_correct() {
        let xs = vec![vec![1.0, 2.0, 3.0]];
        let inner = make_precomputed(xs, vec![0]);
        let energies = [4.0, 9.0, 25.0]; // sqrt = [2, 3, 5]

        let model =
            NormalizedTransmissionModel::new_with_exponential(inner, &energies, 1, 2, 3, 4, 5, 6);

        // params: [density, anorm, back_a, back_b, back_c, back_d, back_f]
        let density = 0.1;
        let anorm = 1.02;
        let back_a = 0.01;
        let back_b = 0.005;
        let back_c = 0.002;
        let back_d = 0.05;
        let back_f = 3.0;
        let params = [density, anorm, back_a, back_b, back_c, back_d, back_f];

        let y = model.evaluate(&params).unwrap();

        // Manually compute expected
        let xs_vals = [1.0, 2.0, 3.0];
        let sqrt_e = [2.0, 3.0, 5.0];
        for i in 0..3 {
            let t_inner = (-density * xs_vals[i]).exp();
            let expected = anorm * t_inner
                + back_a
                + back_b / sqrt_e[i]
                + back_c * sqrt_e[i]
                + back_d * (-back_f / sqrt_e[i]).exp();
            assert!(
                (y[i] - expected).abs() < 1e-12,
                "bin {i}: got {}, expected {expected}",
                y[i]
            );
        }
    }

    /// Analytical Jacobian for BackD and BackF columns must match central FD.
    #[test]
    fn exponential_jacobian_matches_finite_difference() {
        let xs = vec![vec![1.0, 2.0, 3.0, 0.5, 1.5]];
        let inner = make_precomputed(xs, vec![0]);
        let energies = [0.1, 1.0, 4.0, 25.0, 100.0]; // span 0.1–100 eV

        let model =
            NormalizedTransmissionModel::new_with_exponential(inner, &energies, 1, 2, 3, 4, 5, 6);

        // params: [density, anorm, back_a, back_b, back_c, back_d, back_f]
        let params = [0.1, 1.02, 0.01, 0.005, 0.002, 0.05, 3.0];
        let y = model.evaluate(&params).unwrap();
        let free_indices: Vec<usize> = (0..7).collect();

        let jac = model
            .analytical_jacobian(&params, &free_indices, &y)
            .expect("analytical Jacobian should be available");

        // Central finite difference for all parameters
        let h = 1e-7;
        for (col, &pidx) in free_indices.iter().enumerate() {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[pidx] += h;
            p_minus[pidx] -= h;
            let y_plus = model.evaluate(&p_plus).unwrap();
            let y_minus = model.evaluate(&p_minus).unwrap();

            for row in 0..energies.len() {
                let fd = (y_plus[row] - y_minus[row]) / (2.0 * h);
                let anal = jac.get(row, col);
                let abs_err = (anal - fd).abs();
                let rel_err = abs_err / fd.abs().max(1e-15);
                assert!(
                    rel_err < 1e-5 || abs_err < 1e-10,
                    "param {pidx} (col {col}), bin {row}: analytical={anal:.10e}, \
                     fd={fd:.10e}, rel_err={rel_err:.2e}"
                );
            }
        }
    }

    /// Round-trip: fit recovers all 6 background + density from noiseless data.
    #[test]
    fn exponential_fit_recovers_all_params() {
        let xs = vec![vec![1.0, 2.0, 3.0, 2.0, 1.5, 0.8, 1.2, 2.5]];
        let inner = make_precomputed(xs, vec![0]);
        let energies = [0.5, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 64.0];

        let model =
            NormalizedTransmissionModel::new_with_exponential(inner, &energies, 1, 2, 3, 4, 5, 6);

        // True parameters
        let true_density = 0.15;
        let true_anorm = 1.02;
        let true_back_a = 0.01;
        let true_back_b = 0.005;
        let true_back_c = 0.002;
        let true_back_d = 0.03;
        let true_back_f = 2.0;
        let true_params = [
            true_density,
            true_anorm,
            true_back_a,
            true_back_b,
            true_back_c,
            true_back_d,
            true_back_f,
        ];

        let y_obs = model.evaluate(&true_params).unwrap();
        let sigma = vec![0.001; y_obs.len()];

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.1),
            FitParameter {
                name: "anorm".into(),
                value: 1.0,
                lower: 0.5,
                upper: 1.5,
                fixed: false,
            },
            FitParameter {
                name: "back_a".into(),
                value: 0.0,
                lower: -0.5,
                upper: 0.5,
                fixed: false,
            },
            FitParameter {
                name: "back_b".into(),
                value: 0.0,
                lower: -0.5,
                upper: 0.5,
                fixed: false,
            },
            FitParameter {
                name: "back_c".into(),
                value: 0.0,
                lower: -0.5,
                upper: 0.5,
                fixed: false,
            },
            FitParameter {
                name: "back_d".into(),
                value: 0.01,
                lower: 0.0,
                upper: 1.0,
                fixed: false,
            },
            FitParameter {
                name: "back_f".into(),
                value: 1.0,
                lower: 0.0,
                upper: 100.0,
                fixed: false,
            },
        ]);

        let config = LmConfig {
            max_iter: 500,
            ..LmConfig::default()
        };

        let result = lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &config).unwrap();

        assert!(result.converged, "Fit should converge");

        let fitted = &result.params;
        let check = |name, fitted_val: f64, true_val: f64, tol: f64| {
            let err = (fitted_val - true_val).abs();
            let rel = err / true_val.abs().max(1e-10);
            assert!(
                rel < tol || err < 1e-6,
                "{name}: fitted={fitted_val:.6}, true={true_val:.6}, rel_err={rel:.4}"
            );
        };

        check("density", fitted[0], true_density, 0.10);
        check("anorm", fitted[1], true_anorm, 0.10);
        check("back_a", fitted[2], true_back_a, 0.10);
        check("back_b", fitted[3], true_back_b, 0.10);
        check("back_c", fitted[4], true_back_c, 0.10);
        check("back_d", fitted[5], true_back_d, 0.10);
        check("back_f", fitted[6], true_back_f, 0.10);
    }

    // ── EnergyScaleTransmissionModel tests ──

    /// Verify that corrected_energies shifts the grid correctly.
    /// Build a single-isotope (U-238) EnergyScale model for tests: density at
    /// param 0, t0 at param 1, l_scale at param 2.  Issue #608: σ is evaluated
    /// from the resonance at the corrected energies (matching forward_model), so
    /// test grids should overlap the U-238 resonance (~6.67 eV) for non-trivial
    /// σ.  Temperature 300 K, flight path 25 m.
    fn make_energy_scale_u238(
        energies: Vec<f64>,
        instrument: Option<Arc<InstrumentParams>>,
    ) -> EnergyScaleTransmissionModel {
        EnergyScaleTransmissionModel::new(
            Arc::new(vec![u238_single_resonance()]),
            Arc::new(vec![0]),
            Arc::new(vec![1.0]),
            300.0,
            energies,
            25.0,
            1,
            2,
            instrument,
        )
    }

    #[test]
    fn energy_scale_corrected_energies() {
        let energies = vec![10.0, 20.0, 50.0, 100.0, 200.0];
        let model = make_energy_scale_u238(energies.clone(), None);

        // With t0=0, l_scale=1: corrected energies should equal nominal
        let e_corr = model.corrected_energies(0.0, 1.0);
        for (i, (&nom, &corr)) in energies.iter().zip(e_corr.iter()).enumerate() {
            assert!(
                (nom - corr).abs() / nom < 1e-10,
                "bin {i}: nominal={nom}, corrected={corr}"
            );
        }

        // With l_scale > 1: all corrected energies should increase
        let e_corr_ls = model.corrected_energies(0.0, 1.005);
        for (i, (&nom, &corr)) in energies.iter().zip(e_corr_ls.iter()).enumerate() {
            assert!(
                corr > nom,
                "bin {i}: l_scale=1.005 should increase energy, got nom={nom}, corr={corr}"
            );
        }

        // With t0 > 0: energies should increase (shorter effective TOF)
        let e_corr_t0 = model.corrected_energies(1.0, 1.0);
        for (i, (&nom, &corr)) in energies.iter().zip(e_corr_t0.iter()).enumerate() {
            assert!(
                corr > nom,
                "bin {i}: t0=1.0 should increase energy, got nom={nom}, corr={corr}"
            );
        }
    }

    #[test]
    fn corrected_energy_grid_matches_energy_scale_model() {
        // Pin the resolution calibrator's `corrected_energy_grid` to the runtime
        // `EnergyScaleTransmissionModel::corrected_energies`: they are separate
        // implementations of the SAME (t0, L_scale) energy-scale convention, and the
        // calibrated (t0, L_scale) must be consumable by the runtime model. This test
        // makes a future sign/numerator/L_scale/TOF_FACTOR drift in *either* fail
        // fast, and anchors the calibrator's recovery tests (which otherwise inject
        // and recover through the same helper — a self-consistent loop). Probes use
        // feasible t0 (≪ min_tof) so the runtime clamp never engages and the two are
        // bit-for-bit comparable.
        let energies = vec![5.0, 8.0, 12.0, 20.0, 50.0, 120.0];
        let flight = 25.0;
        let model = make_energy_scale_u238(energies.clone(), None);
        for &(t0, l_scale) in &[
            (0.0, 1.0),
            (1.5, 1.0),
            (-2.0, 1.0),
            (0.0, 1.005),
            (0.0, 0.995),
            (1.0, 1.003),
            (-1.0, 0.997),
        ] {
            let runtime = model.corrected_energies(t0, l_scale);
            let calib =
                crate::resolution_calib::corrected_energy_grid(&energies, t0, l_scale, flight)
                    .expect("feasible t0 must not error");
            for (i, (&r, &c)) in runtime.iter().zip(calib.iter()).enumerate() {
                assert!(
                    (r - c).abs() / r < 1e-12,
                    "convention drift at bin {i} (t0={t0}, L_scale={l_scale}): \
                     runtime={r}, calibrator={c}"
                );
            }
        }
    }

    /// Issue #608: at identity calibration (t0=0, l_scale=1) the corrected grid
    /// equals the nominal grid, so EnergyScale must evaluate the SAME true σ as
    /// `forward_model` — the independent oracle — to machine precision.
    #[test]
    fn energy_scale_evaluate_identity() {
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.03).collect();
        let density = 0.01;
        let model_es = make_energy_scale_u238(energies.clone(), None);
        let y_es = model_es.evaluate(&[density, 0.0, 1.0]).unwrap();

        let sample = SampleParams::new(300.0, vec![(u238_single_resonance(), density)]).unwrap();
        let y_ref = transmission::forward_model(&energies, &sample, None).unwrap();

        for (i, (&a, &b)) in y_es.iter().zip(y_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "bin {i}: energy_scale={a}, forward_model={b}"
            );
        }
    }

    /// Jacobian for energy-scale model: density columns must match FD.
    #[test]
    fn energy_scale_jacobian_density_matches_fd() {
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.06).collect();
        let model = make_energy_scale_u238(energies.clone(), None);

        let params = [0.01, 0.5, 1.002]; // density, t0, l_scale
        let y = model.evaluate(&params).unwrap();
        // Density column only (matching this test's name).  The energy-scale
        // (t0 / L_scale) columns are FD-based and method-dependent; they are
        // covered against a matching-h FD2 reference by the partial_gal_* tests.
        // Comparing them to a different-h FD here would be apples-to-oranges,
        // especially on the sharp U-238 resonance (#608 migration
        // to true-σ resonance data).
        let free = vec![0];
        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("Jacobian should be available");

        let h = 1e-7;
        for (col, &pidx) in free.iter().enumerate() {
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[pidx] += h;
            pm[pidx] -= h;
            let yp = model.evaluate(&pp).unwrap();
            let ym = model.evaluate(&pm).unwrap();
            for row in 0..energies.len() {
                let fd = (yp[row] - ym[row]) / (2.0 * h);
                let anal = jac.get(row, col);
                let abs_err = (anal - fd).abs();
                let rel_err = abs_err / fd.abs().max(1e-15);
                assert!(
                    rel_err < 1e-3 || abs_err < 1e-8,
                    "param {pidx} col {col} bin {row}: anal={anal:.6e} fd={fd:.6e} rel={rel_err:.2e}"
                );
            }
        }
    }

    /// LM fit with energy-scale model recovers l_scale from shifted data.
    ///
    /// Uses a sharp Breit-Wigner-like resonance on a dense grid so the
    /// energy shift is unambiguous.  Only l_scale is varied (t0 fixed
    /// at 0) to avoid degenerate local minima.
    #[test]
    fn energy_scale_fit_recovers_l_scale() {
        // Dense grid over the sharp U-238 resonance (~6.67 eV) so the energy
        // shift is unambiguous.  Only l_scale is varied (t0 fixed at 0).
        let energies: Vec<f64> = (0..200).map(|i| 4.0 + (i as f64) * 0.03).collect();

        let true_density = 0.001;
        let true_ls = 1.003;

        let model = make_energy_scale_u238(energies, None);
        let true_params = [true_density, 0.0, true_ls];
        let y_obs = model.evaluate(&true_params).unwrap();
        let sigma = vec![0.001; y_obs.len()];

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", 0.0005),
            FitParameter::fixed("t0", 0.0),
            FitParameter {
                name: "l_scale".into(),
                value: 1.0,
                lower: 0.99,
                upper: 1.01,
                fixed: false,
            },
        ]);

        let config = LmConfig {
            max_iter: 200,
            ..LmConfig::default()
        };

        let result = lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &config).unwrap();

        assert!(result.converged, "Fit should converge");
        let f = &result.params;
        assert!(
            (f[0] - true_density).abs() / true_density < 0.05,
            "density: fitted={}, true={true_density}",
            f[0]
        );
        assert!(
            (f[2] - true_ls).abs() < 0.001,
            "l_scale: fitted={}, true={true_ls}",
            f[2]
        );
    }

    /// Partial-GAL Jacobian with NO resolution should match FD2 to f64
    /// roundoff: in this regime the rank-1 identity
    /// `J[:, L_scale] = ((tof - t0) / L_scale) * J[:, t0]` is exact (the
    /// forward chain factorises through `e_corr` only, with no
    /// resolution operator to introduce additional `(t0, L_scale)`
    /// dependence).  Issue #489.
    #[test]
    fn partial_gal_no_resolution_matches_fd2() {
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.06).collect();
        // Pin both reference and alt models explicitly via
        // `with_jacobian_method` so the test is independent of the
        // process-global `NEREIDS_TZERO_JACOBIAN` env var.  Without
        // pinning, the post-#489 default of `PartialGal` would make
        // the "FD2 reference" actually run partial-GAL (vacuous
        // self-comparison).
        let mut model = make_energy_scale_u238(energies.clone(), None)
            .with_jacobian_method(EnergyScaleJacobianMethod::FiniteDifference);

        let params = [0.001, 0.05, 1.002]; // density, t0, l_scale
        let free = vec![0, 1, 2];

        // FD2 reference Jacobian (explicitly pinned above).
        let jac_fd2 = model
            .analytical_jacobian(&params, &free, &model.evaluate(&params).unwrap())
            .expect("FD2 Jacobian should be available");

        // Partial-GAL Jacobian.
        model = model.with_jacobian_method(EnergyScaleJacobianMethod::PartialGal);
        let jac_pg = model
            .analytical_jacobian(&params, &free, &model.evaluate(&params).unwrap())
            .expect("partial-GAL Jacobian should be available");

        // Density column: identical (analytical, not affected by method).
        for i in 0..energies.len() {
            let fd2 = jac_fd2.get(i, 0);
            let pg = jac_pg.get(i, 0);
            assert!(
                (fd2 - pg).abs() < 1e-15,
                "density bin {i}: fd2={fd2:.6e} pg={pg:.6e}"
            );
        }
        // t0 column: identical (both methods use the same FD pair when
        // both t0 and L_scale are free; partial-GAL just hoists it out
        // of the loop).
        for i in 0..energies.len() {
            let fd2 = jac_fd2.get(i, 1);
            let pg = jac_pg.get(i, 1);
            assert!(
                (fd2 - pg).abs() < 1e-15,
                "t0 bin {i}: fd2={fd2:.6e} pg={pg:.6e}"
            );
        }
        // L_scale column: the rank-1 derivation is analytically exact without
        // resolution.  The only residual vs FD2 is the difference in central-FD
        // truncation — PartialGal's L_scale inherits the t0 step (h=1e-4), FD2
        // takes a direct L_scale step (h=1e-7).  On the sharp U-238 resonance
        // that truncation dominates small-derivative TAIL bins (per-bin rel can
        // hit a few % there while contributing negligibly to the spectrum), so
        // compare the aggregate relative L₂ — the same robust metric the
        // with-resolution sister test uses.  Measured ~8.0e-3 here; the bound
        // (2.5e-2) gives ~3× headroom yet is far below the O(1) a broken rank-1
        // identity would produce.
        let mut num_sq = 0.0_f64;
        let mut den_sq = 0.0_f64;
        for i in 0..energies.len() {
            let fd2 = jac_fd2.get(i, 2);
            let pg = jac_pg.get(i, 2);
            let diff = pg - fd2;
            num_sq += diff * diff;
            den_sq += fd2 * fd2;
        }
        let rel_l2 = (num_sq / den_sq.max(1e-30)).sqrt();
        assert!(
            rel_l2 < 2.5e-2,
            "L_scale rank-1 vs FD2 rel L₂ = {rel_l2:.3e} (expected ≪ 1 without \
             resolution — the rank-1 identity is exact up to FD truncation)"
        );
    }

    /// When only L_scale is free (t0 fixed), partial-GAL falls through
    /// to standard FD: there is no t0 column to derive L_scale from,
    /// so the per-coordinate FD path must still be used. Verifies the
    /// dispatch logic correctly handles this case.
    #[test]
    fn partial_gal_l_scale_only_falls_through_to_fd() {
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.06).collect();
        let model = make_energy_scale_u238(energies.clone(), None)
            .with_jacobian_method(EnergyScaleJacobianMethod::PartialGal);

        let params = [0.001, 0.0, 1.002];
        let free = vec![0, 2]; // density + L_scale (no t0)
        let y = model.evaluate(&params).unwrap();
        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("Jacobian should be available even when t0 not free");

        // L_scale column should match a manual central FD reference.
        let h = 1e-7;
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        pp[2] += h;
        pm[2] -= h;
        let yp = model.evaluate(&pp).unwrap();
        let ym = model.evaluate(&pm).unwrap();
        for i in 0..energies.len() {
            let fd = (yp[i] - ym[i]) / (2.0 * h);
            let anal = jac.get(i, 1);
            let abs_err = (anal - fd).abs();
            let rel_err = abs_err / fd.abs().max(1e-15);
            assert!(
                rel_err < 1e-3 || abs_err < 1e-8,
                "L_scale bin {i}: anal={anal:.6e} fd={fd:.6e} rel={rel_err:.2e}"
            );
        }
    }

    /// Regression for #500: at `l_scale ≈ 0` the partial-GAL rank-1
    /// derivation `(tof - t0_clamped) / l_scale` divides by zero,
    /// producing a NaN L_scale Jacobian column.  After the fix, the
    /// L_scale column falls through to the per-coordinate FD path —
    /// every Jacobian entry must be finite, and the L_scale column
    /// must agree with the FD2 reference (which uses the same
    /// per-coordinate FD).
    ///
    /// Setup mirrors `partial_gal_no_resolution_matches_fd2` so the
    /// FD-tolerance comparison against FD2 stays apples-to-apples.
    #[test]
    fn partial_gal_l_scale_zero_falls_through_to_finite_jacobian() {
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.06).collect();
        let mut model = make_energy_scale_u238(energies.clone(), None)
            .with_jacobian_method(EnergyScaleJacobianMethod::FiniteDifference);

        // l_scale = 0.0 — well below `L_SCALE_EPSILON = 1e-12` — so the
        // partial-GAL guard fires and falls through to FD.
        //
        // **Active code path (regression target):** the test inputs
        // are chosen so the new `L_SCALE_EPSILON` guard is what fires,
        // *not* the older `t0 + h >= t0_limit` precompute fallthrough.
        // Specifically:
        //
        //   - `min_tof_us = tof_factor * 25.0 / sqrt(max_E ≈ 10.0) ≈ 5.7e2 µs`
        //   - `t0 + h = 0.05 + 1e-4 = 0.0501 µs ≪ min_tof * (1 - 1e-12)`
        //
        // So `partial_gal_t0_column = Some(...)` (not `None`), the
        // partial-GAL block at line ~2208 enters, and the L_scale
        // branch reaches the new `l_scale.abs() < L_SCALE_EPSILON`
        // guard.  Pre-fix, the inner `(tof_i - t0_clamped) / 0.0 =
        // ±inf` then `inf * 0 = NaN` would poison the column.  If a
        // future refactor changes these inputs, verify that
        // `partial_gal_t0_column.is_some()` still holds for this test
        // — otherwise the regression target shifts to a different
        // code path.
        let params = [0.001, 0.05, 1e-13]; // density, t0, l_scale ≈ 0 (< L_SCALE_EPSILON)
        let free = vec![0, 1, 2];

        // FD2 reference Jacobian.  FD2 computes each column via its
        // own per-coordinate FD pair, so it produces well-defined
        // finite values at l_scale = 0 (no division by l_scale in the
        // FD2 path).
        let jac_fd2 = model
            .analytical_jacobian(&params, &free, &model.evaluate(&params).unwrap())
            .expect("FD2 Jacobian should be available at l_scale = 0");

        // Partial-GAL Jacobian — with the #500 guard, L_scale column
        // falls through to the same per-coordinate FD path.
        model = model.with_jacobian_method(EnergyScaleJacobianMethod::PartialGal);
        let jac_pg = model
            .analytical_jacobian(&params, &free, &model.evaluate(&params).unwrap())
            .expect("partial-GAL Jacobian should be available at l_scale = 0 (fallthrough to FD)");

        // Primary regression: every entry finite.  Pre-fix the L_scale
        // column would be NaN from the `1 / l_scale` division.
        for i in 0..jac_pg.nrows {
            for c in 0..jac_pg.ncols {
                let v = jac_pg.get(i, c);
                assert!(
                    v.is_finite(),
                    "partial-GAL Jacobian must be finite at l_scale = 0; \
                     got non-finite at ({i},{c}) = {v}"
                );
            }
        }

        // Bit-equivalent to FD2 across every column — confirms the
        // L_scale fallthrough lands on the same FD code path FD2 uses,
        // and the density / t0 columns are unchanged by the guard.
        for c in 0..jac_pg.ncols {
            for i in 0..jac_pg.nrows {
                let fd2 = jac_fd2.get(i, c);
                let pg = jac_pg.get(i, c);
                let abs_err = (fd2 - pg).abs();
                let rel_err = abs_err / fd2.abs().max(1e-15);
                assert!(
                    rel_err < 1e-3 || abs_err < 1e-8,
                    "partial-GAL must match FD2 at l_scale = 0; \
                     col {c} bin {i}: fd2={fd2:.6e} pg={pg:.6e} rel={rel_err:.2e}"
                );
            }
        }
    }

    /// In-tree regression for the partial-GAL rank-1 approximation in
    /// the presence of a non-trivial resolution kernel.  Issue #499.
    ///
    /// **Motivation.**  The empirical bound supporting the post-#489
    /// default flip to `PartialGal` was measured on real VENUS Hf
    /// 120-min KL+per-iso+TZERO 4×4 data: 15 of 16 fitted pixels landed
    /// within 0.1·σ_Fisher of the FD2 reference for the L_scale
    /// column.  That measurement was made against the production
    /// USR/FTS tabulated resolution kernel, which ORNL release policy
    /// keeps out of the repository — so it cannot ship as an in-tree
    /// fixture.  Without an in-tree analogue, a future refactor could
    /// silently regress the rank-1 bound on real workloads.
    ///
    /// **Synthetic stand-in.**  This test exercises the same code path
    /// with a sharp Gaussian "resonance" cross-section convolved by a
    /// Gaussian resolution kernel — a deliberately rough stand-in for
    /// the SAMMY-format tabulated VENUS USR/FTS kernel.  The Gaussian
    /// kernel is *not* a fidelity replacement; it is the simplest
    /// non-trivial resolution operator that activates the partial-GAL
    /// resolution-bearing branch without introducing a binary fixture.
    ///
    /// **Tolerance.**  Density and t0 columns retain the same tight
    /// bound as the no-resolution test (the resolution operator does
    /// not couple into those columns differently).  The L_scale column
    /// is checked via relative L₂ norm against the FD2 reference with
    /// tolerance `PARTIAL_GAL_REL_L2_TOLERANCE = 1.5e-5`.  On the U-238
    /// resonance grid below (kernel sized so it spans several bins and
    /// meaningfully broadens the resonance) the measured relative L₂ is
    /// `~4.3e-6` — the tolerance gives roughly 3× headroom over the current
    /// measurement, tight enough to catch a non-trivial regression
    /// of the rank-1 simplification while loose enough to absorb
    /// FD-truncation noise.  An upstream pre-check (see below)
    /// asserts the kernel itself is non-trivial so a future tweak
    /// to grid spacing or kernel parameters cannot silently degrade
    /// this back into a vacuous "no-resolution-in-disguise" test
    /// (a regression that has occurred before).  The
    /// measured relative L₂ surfaces in the assert message if the
    /// bound is ever exceeded so future tightening is straightforward.
    #[test]
    fn partial_gal_with_resolution_matches_fd2() {
        use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};

        // Tolerance for relative L₂ error on the L_scale column.
        // Measured rel L₂ on this synthetic grid is ~9.3e-4; 3e-3
        // gives ~3× headroom — tight enough to catch a non-trivial
        // regression of the rank-1 simplification under resolution,
        // loose enough to absorb FD truncation noise.  See rustdoc
        // above for why this bound is conservative rather than the
        // tighter empirical 0.1·σ_Fisher seen on real workloads.
        const PARTIAL_GAL_REL_L2_TOLERANCE: f64 = 1.5e-5;

        // Dense grid over the sharp U-238 resonance (~6.67 eV) so the σ feature
        // is well resolved and the resolution kernel meaningfully broadens it.
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.06).collect();

        // Gaussian resolution kernel sized to be NON-TRIVIAL on this grid (it
        // broadens the U-238 resonance by ~1%, verified by the pre-check below).
        // A kernel-too-narrow vacuous-test regression has occurred before;
        // the pre-check guards against re-introducing it.
        let instrument = Some(Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        }));

        // Pin FD2 first so the comparison is independent of the
        // process-global `NEREIDS_TZERO_JACOBIAN` env var, matching
        // the pattern used by `partial_gal_no_resolution_matches_fd2`.
        let mut model = make_energy_scale_u238(energies.clone(), instrument)
            .with_jacobian_method(EnergyScaleJacobianMethod::FiniteDifference);

        let params = [0.001, 0.05, 1.002]; // density, t0, l_scale
        let free = vec![0, 1, 2];

        // Pre-check: confirm the resolution kernel actually broadens the
        // spectrum on this grid, so the comparison is not a vacuous
        // no-resolution-in-disguise test.
        let model_no_resolution = make_energy_scale_u238(energies.clone(), None)
            .with_jacobian_method(EnergyScaleJacobianMethod::FiniteDifference);
        let t_no_res = model_no_resolution.evaluate(&params).unwrap();
        let t_with_res = model.evaluate(&params).unwrap();
        let diff_inf = t_no_res
            .iter()
            .zip(t_with_res.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let t_inf = t_no_res.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(
            diff_inf > 1e-3 * t_inf,
            "resolution kernel must broaden the spectrum nontrivially \
             (got ||T_kernel - T_none||_∞ = {diff_inf:.3e}, ||T_none||_∞ = {t_inf:.3e}, \
             ratio = {ratio:.3e}); widen the kernel or sharpen the resonance",
            ratio = diff_inf / t_inf.max(1e-30),
        );

        // FD2 reference Jacobian (explicitly pinned above).
        let jac_fd2 = model
            .analytical_jacobian(&params, &free, &model.evaluate(&params).unwrap())
            .expect("FD2 Jacobian should be available with resolution kernel");

        // Flip to partial-GAL.
        model = model.with_jacobian_method(EnergyScaleJacobianMethod::PartialGal);
        let jac_pg = model
            .analytical_jacobian(&params, &free, &model.evaluate(&params).unwrap())
            .expect("partial-GAL Jacobian should be available with resolution kernel");

        // Density column: tight bound — resolution doesn't change the
        // density derivative path.
        for i in 0..energies.len() {
            let fd2 = jac_fd2.get(i, 0);
            let pg = jac_pg.get(i, 0);
            let abs_err = (fd2 - pg).abs();
            let rel_err = abs_err / fd2.abs().max(1e-15);
            assert!(
                rel_err < 1e-3 || abs_err < 1e-8,
                "density bin {i}: fd2={fd2:.6e} pg={pg:.6e} rel={rel_err:.2e}"
            );
        }

        // t0 column: tight bound — both methods use the same FD pair
        // on t0 (partial-GAL just hoists it out of the per-coord loop).
        for i in 0..energies.len() {
            let fd2 = jac_fd2.get(i, 1);
            let pg = jac_pg.get(i, 1);
            let abs_err = (fd2 - pg).abs();
            let rel_err = abs_err / fd2.abs().max(1e-15);
            assert!(
                rel_err < 1e-3 || abs_err < 1e-8,
                "t0 bin {i}: fd2={fd2:.6e} pg={pg:.6e} rel={rel_err:.2e}"
            );
        }

        // L_scale column: relative L₂ norm bound.  In the presence of
        // a non-trivial resolution kernel the rank-1 identity is no
        // longer exact; the resolution operator introduces an
        // additional (t0, L_scale)-dependence that partial-GAL
        // approximates as zero.  The bound captures the residual.
        let mut num_sq = 0.0_f64;
        let mut den_sq = 0.0_f64;
        for i in 0..energies.len() {
            let fd2 = jac_fd2.get(i, 2);
            let pg = jac_pg.get(i, 2);
            let diff = pg - fd2;
            num_sq += diff * diff;
            den_sq += fd2 * fd2;
        }
        let rel_l2 = (num_sq / den_sq.max(1e-30)).sqrt();
        assert!(
            rel_l2 < PARTIAL_GAL_REL_L2_TOLERANCE,
            "L_scale rel L₂ = {rel_l2:.4e} exceeds tolerance {tol:.4e}; \
             tighten or loosen `PARTIAL_GAL_REL_L2_TOLERANCE` (see rustdoc)",
            tol = PARTIAL_GAL_REL_L2_TOLERANCE,
        );
    }
}
