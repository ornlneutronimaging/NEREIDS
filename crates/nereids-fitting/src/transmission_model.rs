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
    /// Doppler-broadened cross-sections σ_D(E) per isotope,
    /// shape \[n_isotopes\]\[n_energies\].
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
    /// Optional sparse empirical cubature plan (epic #472).
    ///
    /// When the plan is present AND its `target_energies` match this
    /// model's energy grid AND `cubature.k() == n_density_params`
    /// AND no temperature / energy-scale fitting is active, the
    /// `evaluate()` / `analytical_jacobian()` fast path calls
    /// `cubature.forward_and_jacobian(n)` directly instead of
    /// `exp(-Σ n σ) + apply_resolution`.  Any guard failure falls
    /// back to the exact path, so the default behaviour is
    /// byte-identical to main.
    pub sparse_cubature_plan: Option<Arc<SparseEmpiricalCubaturePlan>>,
    /// Optional scalar (k = 1) surrogate plan (epic #472, PR #475).
    ///
    /// Mutually exclusive with `sparse_cubature_plan` in practice —
    /// the cubature dispatch fires only for `k ≥ 2` and the scalar
    /// plan only for `k == 1`.  The type alias
    /// `ScalarSurrogatePlan = ScalarChebyshevPlan` is kept as a
    /// stable public name so a future scalar surrogate can swap in
    /// without touching this field or any dispatch call site.
    /// PR #475 picked Chebyshev-in-density over Lanczos Gauss
    /// quadrature after a real-VENUS bench-off (Chebyshev won on
    /// both the accuracy and wall-time axes; see
    /// `nereids_physics::surrogate` module docs).
    pub sparse_scalar_plan: Option<Arc<ScalarSurrogatePlan>>,
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
/// wrong Jacobians.  Codex round-4 P2 on PR #480.
fn density_param_indices(density_indices: &[usize]) -> Vec<usize> {
    // `sort_unstable` + `dedup` is O(n log n) and avoids the O(n²)
    // cost of repeated `Vec::contains` scans.  This runs on every
    // `evaluate()` / `analytical_jacobian()` call, so the linear-
    // scan version showed up in spatial-map profiling once the
    // per-pixel cubature dispatch started firing.  Copilot Phase B
    // finding on PR #481.
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
/// `apply_resolution_with_plan` already enforces — Codex round-1 P1
/// on PR #480).
///
/// **Tabulated-kernel tie**: the cubature fast path folds
/// `apply_resolution*` into its atom sweep — skipping it when the
/// model otherwise would have applied a Gaussian kernel is a
/// silent wrong-answer path.  We require
/// `matches!(instrument_resolution, ResolutionFunction::Tabulated(_))`
/// so Gaussian-resolution models never hit the cubature path (a
/// plan is only ever built against a tabulated kernel).  Codex
/// round-3 P2 on PR #480.
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
/// fingerprint on the cubature plan, which is out of scope for
/// this PR (Codex round-4 P2 on PR #480).  Upstream callers are
/// responsible for clearing the cubature when they swap kernels;
/// in spatial dispatch this is enforced by
/// `UnifiedFitConfig::with_precomputed_cross_sections` /
/// `with_precomputed_base_xs` / `with_groups` all clearing the
/// cached cubature (see pipeline.rs), so a refit through the
/// standard surface cannot hit this case.
/// Check whether a scalar (k = 1) surrogate plan is eligible given
/// the model's energy grid, active tabulated resolution, and
/// `n_density_params == 1`.  Parallels
/// [`cubature_eligible`] for the multi-isotope path — same
/// grid-identity + `Tabulated(_)` variant guards, same
/// optional-ResolutionPlan transitive check.  Epic #472, PR #475.
fn scalar_eligible(
    plan: &ScalarSurrogatePlan,
    energies: &[f64],
    instrument_resolution: &ResolutionFunction,
    resolution_plan: Option<&ResolutionPlan>,
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
    if let Some(res_plan) = resolution_plan {
        if res_plan.target_energies().len() != energies.len() {
            return false;
        }
        for (e_cur, e_res) in energies.iter().zip(res_plan.target_energies()) {
            if e_cur.to_bits() != e_res.to_bits() {
                return false;
            }
        }
    }
    true
}

/// Check whether the scalar iterate `n` is inside the surrogate's
/// recorded training box `[0, train_max]` with 50 % tolerance.
fn scalar_density_within_box(plan: &ScalarSurrogatePlan, n: f64) -> bool {
    let Some(train_max) = plan.density_box() else {
        return true;
    };
    if !n.is_finite() || n < 0.0 {
        return false;
    }
    const TOLERANCE: f64 = 1.5;
    n <= train_max * TOLERANCE
}

/// Check whether the current density iterate `n` is inside the
/// training region recorded on the cubature plan, with a 50 %
/// expansion tolerance to avoid thrashing at the box boundary.
/// When the plan has no recorded box, accepts unconditionally
/// (caller is responsible; legacy code path).
///
/// Returns `false` when any component escapes the tolerance-
/// expanded box OR is negative, OR is not finite.  Codex round-4
/// P1 on PR #480: without this, a spatial fit whose per-pixel
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
    // k ≥ 2: PR #475 (scalar k=1 branch) handles the grouped case.
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
    // surrogate.  Codex round-3 P2 on PR #480.
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
    // cubature's self-check above is the grid guard.  Codex
    // separate-review finding on PR #481 inception: the round-2
    // "resolution_plan.is_some() required" was over-strict and
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
        // into a single per-row atom sweep).  See epic #472.
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
        // Epic #472, PR #475.
        if let (Some(scalar), Some(inst), Some(energies)) =
            (&self.sparse_scalar_plan, &self.instrument, &self.energies)
        {
            let params_indices = density_param_indices(&self.density_indices);
            if scalar_eligible(
                scalar,
                energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) {
                let n = params[params_indices[0]];
                if scalar_density_within_box(scalar, n) {
                    return Ok(scalar.forward_scalar(n));
                }
            }
        }

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

        // Issue #442: apply resolution broadening to total transmission
        // AFTER Beer-Lambert.  This is the SAMMY-correct ordering.
        if let (Some(inst), Some(energies)) = (&self.instrument, &self.energies) {
            let t_broadened = resolution::apply_resolution_with_plan(
                self.resolution_plan.as_deref(),
                energies,
                &transmission,
                &inst.resolution,
            )
            .map_err(|e| FittingError::EvaluationFailed(format!("resolution broadening: {e}")))?;
            Ok(t_broadened)
        } else {
            Ok(transmission)
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
                //
                // **Known caveat (Codex round-5 P2 on PR #480,
                // extended to scalar path in PR #475)**: if
                // `free_param_indices` contains non-density slots,
                // `col_map` is `None` and analytical_jacobian
                // falls through to the exact derivatives — but
                // `evaluate()` unconditionally dispatches the
                // cubature forward when eligible.  The scalar
                // (k = 1) Jacobian branch below has an analogous
                // stronger guard
                // (`free_param_indices == [density_param]`) that
                // similarly falls through when any nuisance slot
                // is free, while its `evaluate()` path fires on
                // `scalar_eligible` alone — the same eval/Jac
                // mix.  In the current layered architecture
                // (`NormalizedTransmissionModel` /
                // `TransmissionKLBackgroundModel` wrap the inner
                // `PrecomputedTransmissionModel` before any
                // non-density nuisance parameter reaches this
                // layer), `free_param_indices` here is always the
                // density slots, so neither mismatch can arise
                // via the standard pipeline.  A defence-in-depth
                // fix (route `evaluate()` through a context that
                // knows about free params) is deferred to the
                // trust-region PR (#476) that will revisit this
                // dispatch surface.
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

        // Scalar (k = 1) surrogate Jacobian fast path — epic #472 PR
        // #475.  For a scalar fit `free_param_indices = [0]`, so
        // the Jacobian has one column.
        if let (Some(scalar), Some(inst), Some(energies)) =
            (&self.sparse_scalar_plan, &self.instrument, &self.energies)
        {
            let params_indices = density_param_indices(&self.density_indices);
            if scalar_eligible(
                scalar,
                energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) && free_param_indices.len() == 1
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
        // tied to that parameter index.
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

        // When resolution is enabled, we need the UNRESOLVED T(E) = exp(-Σnσ)
        // to form the inner derivative -σ·T, then apply resolution.
        // y_current is T_obs = R[T], which is NOT the same as T.
        if let (Some(inst), Some(energies)) = (&self.instrument, &self.energies) {
            // Recompute unresolved T from cross-sections and current params.
            let mut neg_opt = vec![0.0f64; n_e];
            for (i, xs) in self.cross_sections.iter().enumerate() {
                let density = params[self.density_indices[i]];
                for (j, &sigma) in xs.iter().enumerate() {
                    neg_opt[j] -= density * sigma;
                }
            }
            let t_unresolved: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

            // ∂T_obs/∂N_g = R[-σ_sum(E) · T_unresolved(E)]
            let mut jacobian = FlatMatrix::zeros(n_e, n_free);
            for (col, xs_sum) in fp_xs_sums.iter().enumerate() {
                let inner_deriv: Vec<f64> =
                    (0..n_e).map(|i| -xs_sum[i] * t_unresolved[i]).collect();
                let resolved_deriv = resolution::apply_resolution_with_plan(
                    self.resolution_plan.as_deref(),
                    energies,
                    &inner_deriv,
                    &inst.resolution,
                )
                .ok()?;
                for (i, &val) in resolved_deriv.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            }
            Some(jacobian)
        } else {
            // No resolution: ∂T/∂N_g = -σ_sum(E) · T(E)
            // y_current IS T(E) directly.
            let mut jacobian = FlatMatrix::zeros(n_e, n_free);
            for i in 0..n_e {
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
    /// Cached broadened cross-sections from the last `evaluate()` call.
    /// Used by `analytical_jacobian()` to provide density columns without
    /// rebroadening.  Interior mutability via `RefCell` is needed because
    /// `FitModel::evaluate` takes `&self`.  Safe because `TransmissionFitModel`
    /// is constructed per-pixel and never shared across threads.
    cached_broadened_xs: RefCell<Option<Rc<Vec<Vec<f64>>>>>,
    /// Cached analytical temperature derivative ∂σ/∂T, computed on-demand
    /// by `analytical_jacobian()` when the temperature column is needed.
    /// Invalidated when temperature changes (cleared in `evaluate()`).
    cached_dxs_dt: RefCell<Option<Rc<Vec<Vec<f64>>>>>,
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
    /// Optional sparse empirical cubature plan (epic #472).
    ///
    /// See [`PrecomputedTransmissionModel::sparse_cubature_plan`]
    /// for the dispatch contract.  In this per-pixel model the
    /// cubature is additionally constrained: if `temperature_index`
    /// is `Some` or the temperature changes between evaluate calls,
    /// the σ the cubature was built against becomes stale so the
    /// dispatch silently falls back.
    sparse_cubature_plan: Option<Arc<SparseEmpiricalCubaturePlan>>,
    /// Optional scalar (k = 1) surrogate plan (epic #472, PR #475).
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
    /// dispatch conditions.  Epic #472 PR #475.
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
        // falls through to the exact path.  See epic #472.
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

        // Scalar (k = 1) surrogate fast path — epic #472, PR #475.
        if let (Some(scalar), Some(inst)) = (&self.sparse_scalar_plan, &self.instrument)
            && self.temperature_index.is_none()
        {
            let params_indices = density_param_indices(&self.density_indices);
            if scalar_eligible(
                scalar,
                &self.energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) {
                let n = params[params_indices[0]];
                if scalar_density_within_box(scalar, n) {
                    return Ok(scalar.forward_scalar(n));
                }
            }
        }

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

            // Compute broadened XS (or reuse cache if temperature unchanged).
            // Caching avoids redundant Doppler broadening on rejected LM steps
            // (same T, different lambda) and enables analytical_jacobian() to
            // read the broadened σ for density columns.
            //
            // Derivative ∂σ/∂T is computed on-demand in analytical_jacobian(),
            // NOT here — evaluate() is called many times during line search
            // trials, and the derivative overhead would dominate.
            let broadened_xs = if (temperature_k - self.cached_temperature.get()).abs() < 1e-15 {
                Rc::clone(self.cached_broadened_xs.borrow().as_ref().unwrap())
            } else {
                let xs = Rc::new(
                    transmission::broadened_cross_sections_from_base(
                        &self.energies,
                        base_xs,
                        &self.resonance_data,
                        temperature_k,
                        self.instrument.as_deref(),
                    )
                    .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?,
                );
                *self.cached_broadened_xs.borrow_mut() = Some(Rc::clone(&xs));
                // Invalidate derivative cache — temperature changed, old ∂σ/∂T stale.
                *self.cached_dxs_dt.borrow_mut() = None;
                self.cached_temperature.set(temperature_k);
                xs
            };

            // Beer-Lambert: T(E) = exp(-Σᵢ nᵢ · rᵢ · σᵢ(E))
            // where rᵢ is the fractional ratio (1.0 for ungrouped isotopes).
            let n_e = self.energies.len();
            let mut neg_opt = vec![0.0f64; n_e];
            for (i, xs) in broadened_xs.iter().enumerate() {
                let density = params[self.density_indices[i]];
                let ratio = self.density_ratios[i];
                for (j, &sigma) in xs.iter().enumerate() {
                    neg_opt[j] -= density * ratio * sigma;
                }
            }
            let transmission: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

            // Issue #442: apply resolution broadening to total transmission
            // AFTER Beer-Lambert.
            if let Some(ref inst) = self.instrument {
                resolution::apply_resolution_with_plan(
                    self.resolution_plan.as_deref(),
                    &self.energies,
                    &transmission,
                    &inst.resolution,
                )
                .map_err(|e| FittingError::EvaluationFailed(format!("resolution broadening: {e}")))
            } else {
                Ok(transmission)
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

        // Scalar (k = 1) surrogate Jacobian fast path — epic #472 PR
        // #475.  Must match `evaluate()` dispatch conditions + the
        // single-density constraint on free params.
        if let (Some(scalar), Some(inst)) = (&self.sparse_scalar_plan, &self.instrument)
            && self.temperature_index.is_none()
        {
            let params_indices = density_param_indices(&self.density_indices);
            if scalar_eligible(
                scalar,
                &self.energies,
                &inst.resolution,
                self.resolution_plan.as_deref(),
                params_indices.len(),
            ) && free_param_indices.len() == 1
                && free_param_indices[0] == params_indices[0]
            {
                let n = params[params_indices[0]];
                if scalar_density_within_box(scalar, n) {
                    let n_e = self.energies.len();
                    let (_t, dt) = scalar.forward_and_derivative_scalar(n);
                    let mut jacobian = FlatMatrix::zeros(n_e, 1);
                    for (i, &v) in dt.iter().enumerate() {
                        *jacobian.get_mut(i, 0) = v;
                    }
                    return Some(jacobian);
                }
            }
        }

        // Only provide analytical Jacobian when base_xs is available
        // (temperature-fitting fast path with cached broadened XS).
        let _base_xs_guard = self.base_xs.as_ref()?;
        let cached_xs = self.cached_broadened_xs.borrow();
        let broadened_xs = cached_xs.as_ref()?;

        // Guard: verify the cache matches the current parameter temperature.
        if let Some(ti) = self.temperature_index {
            let param_temp = params[ti];
            if (param_temp - self.cached_temperature.get()).abs() > 1e-15 {
                return None;
            }
        }

        let n_e = y_current.len();
        let n_free = free_param_indices.len();
        let mut jacobian = FlatMatrix::zeros(n_e, n_free);

        let temp_col = self
            .temperature_index
            .and_then(|ti| free_param_indices.iter().position(|&fp| fp == ti));

        // When resolution is enabled, we need the UNRESOLVED T to form
        // inner derivatives, then apply resolution.  y_current is T_obs = R[T].
        let t_unresolved: Option<Vec<f64>> = if self.instrument.is_some() {
            let mut neg_opt = vec![0.0f64; n_e];
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

        // Helper: the T(E) to use for inner derivatives.
        // With resolution: unresolved T.  Without: y_current IS T.
        let t_for_deriv: &[f64] = t_unresolved.as_deref().unwrap_or(y_current);

        // ── Density columns: ∂T/∂N_g or ∂T_obs/∂N_g ──
        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            if temp_col == Some(col) {
                continue;
            }
            let mut sigma_sum = vec![0.0f64; n_e];
            for (iso, &di) in self.density_indices.iter().enumerate() {
                if di == fp_idx {
                    let ratio = self.density_ratios[iso];
                    for (j, &sigma) in broadened_xs[iso].iter().enumerate() {
                        sigma_sum[j] += ratio * sigma;
                    }
                }
            }
            // Inner derivative: -σ_sum · T_unresolved
            let inner: Vec<f64> = (0..n_e).map(|i| -sigma_sum[i] * t_for_deriv[i]).collect();

            if let Some(ref inst) = self.instrument {
                // ∂T_obs/∂N_g = R[inner]
                let resolved = resolution::apply_resolution_with_plan(
                    self.resolution_plan.as_deref(),
                    &self.energies,
                    &inner,
                    &inst.resolution,
                )
                .ok()?;
                for (i, &val) in resolved.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            } else {
                for (i, &val) in inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) = val;
                }
            }
        }

        // ── Temperature column: ∂T/∂Temp or ∂T_obs/∂Temp ──
        if let Some(col) = temp_col {
            // Compute ∂σ/∂T on-demand if not cached.
            {
                let needs_compute = self.cached_dxs_dt.borrow().as_ref().is_none();
                if needs_compute {
                    let base_xs = self.base_xs.as_ref()?;
                    let temperature_k = self.cached_temperature.get();
                    let (_, dxs_vec) =
                        transmission::broadened_cross_sections_with_analytical_derivative_from_base(
                            &self.energies,
                            base_xs,
                            &self.resonance_data,
                            temperature_k,
                            self.instrument.as_deref(),
                        )
                        .ok()?;
                    *self.cached_dxs_dt.borrow_mut() = Some(Rc::new(dxs_vec));
                }
            }
            let cached_dxs = self.cached_dxs_dt.borrow();
            let dxs_dt = cached_dxs.as_ref()?;

            // Inner derivative: -T · Σᵢ nᵢ rᵢ ∂σᵢ/∂T
            let inner: Vec<f64> = (0..n_e)
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
                    &self.energies,
                    &inner,
                    &inst.resolution,
                )
                .ok()?;
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

        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            if fp_idx == self.anorm_index {
                // ∂T_out/∂Anorm = T_inner(E)
                for (i, &ti) in t_inner.iter().enumerate() {
                    *jacobian.get_mut(i, col) = ti;
                }
            } else if fp_idx == self.back_a_index {
                // ∂T_out/∂BackA = 1
                for i in 0..n_e {
                    *jacobian.get_mut(i, col) = 1.0;
                }
            } else if fp_idx == self.back_b_index {
                // ∂T_out/∂BackB = 1/√E
                for (i, &inv_se) in self.inv_sqrt_energies.iter().enumerate() {
                    *jacobian.get_mut(i, col) = inv_se;
                }
            } else if fp_idx == self.back_c_index {
                // ∂T_out/∂BackC = √E
                for (i, &se) in self.sqrt_energies.iter().enumerate() {
                    *jacobian.get_mut(i, col) = se;
                }
            } else if self.back_d_index == Some(fp_idx) {
                // ∂T_out/∂BackD = exp(−BackF / √E)
                for (i, &et) in exp_terms.iter().enumerate() {
                    *jacobian.get_mut(i, col) = et;
                }
            } else if self.back_f_index == Some(fp_idx) {
                // ∂T_out/∂BackF = −BackD × exp(−BackF / √E) / √E
                let back_d = params[self.back_d_index.unwrap()];
                for (i, (&et, &inv_se)) in exp_terms
                    .iter()
                    .zip(self.inv_sqrt_energies.iter())
                    .enumerate()
                {
                    *jacobian.get_mut(i, col) = -back_d * et * inv_se;
                }
            } else if let Some(&inner_col) = inner_col_map.get(&fp_idx) {
                // Inner model parameter: ∂T_out/∂p = Anorm × ∂T_inner/∂p
                if let Some(ref jac) = inner_jac {
                    for i in 0..n_e {
                        *jacobian.get_mut(i, col) = anorm * jac.get(i, inner_col);
                    }
                } else {
                    // Inner model did not provide analytical Jacobian —
                    // fall back to finite-difference for the whole thing.
                    return None;
                }
            } else {
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
/// Wraps precomputed cross-sections and re-maps the energy grid at each
/// evaluation:
///   1. Convert nominal energy → TOF: `t = TOF_FACTOR * L / √E_nom`
///   2. Apply calibration: `t_corr = t - t₀`, `E_corr = (TOF_FACTOR * L * L_scale / t_corr)²`
///   3. Interpolate cross-sections σ(E_nom) → σ(E_corr)
///   4. Beer-Lambert + resolution on the corrected grid
///
/// This is equivalent to SAMMY's TZERO parameters.
///
/// The Jacobian for t₀ and L_scale is computed via finite differences
/// (the energy-grid remapping is nonlinear and not easily differentiable
/// analytically through the interpolation + resolution pipeline).
pub struct EnergyScaleTransmissionModel {
    /// Precomputed cross-sections on the NOMINAL energy grid.
    cross_sections: Arc<Vec<Vec<f64>>>,
    /// Density parameter indices (same as PrecomputedTransmissionModel).
    density_indices: Arc<Vec<usize>>,
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
}

impl EnergyScaleTransmissionModel {
    /// Create a new energy-scale transmission model.
    ///
    /// # Arguments
    /// * `cross_sections` — Precomputed σ(E) on the nominal grid.
    /// * `density_indices` — Maps isotope index → parameter index.
    /// * `nominal_energies` — Energy grid in eV (ascending).
    /// * `flight_path_m` — Nominal flight path in meters.
    /// * `t0_index` — Index of t₀ parameter.
    /// * `l_scale_index` — Index of L_scale parameter.
    /// * `instrument` — Optional resolution function.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cross_sections: Arc<Vec<Vec<f64>>>,
        density_indices: Arc<Vec<usize>>,
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
            cross_sections,
            density_indices,
            nominal_energies,
            flight_path_m,
            tof_factor,
            t0_index,
            l_scale_index,
            instrument,
        }
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

    /// Interpolate a cross-section array from nominal grid to corrected grid.
    fn interpolate_xs(nominal_e: &[f64], xs: &[f64], corrected_e: &[f64]) -> Vec<f64> {
        corrected_e
            .iter()
            .map(|&e_corr| {
                // Binary search in the nominal (ascending) grid
                let n = nominal_e.len();
                if e_corr <= nominal_e[0] {
                    return xs[0];
                }
                if e_corr >= nominal_e[n - 1] {
                    return xs[n - 1];
                }
                let pos = nominal_e.partition_point(|&e| e < e_corr);
                let i = if pos == 0 { 0 } else { pos - 1 };
                let frac = (e_corr - nominal_e[i]) / (nominal_e[i + 1] - nominal_e[i]);
                xs[i] + frac * (xs[i + 1] - xs[i])
            })
            .collect()
    }

    /// Evaluate transmission at given parameters (densities + t0 + l_scale).
    fn evaluate_at(&self, params: &[f64], e_corr: &[f64]) -> Result<Vec<f64>, FittingError> {
        let n_e = self.nominal_energies.len();

        // Interpolate cross-sections to corrected energy grid
        let mut neg_opt = vec![0.0f64; n_e];
        for (i, xs) in self.cross_sections.iter().enumerate() {
            let density = params[self.density_indices[i]];
            let xs_interp = Self::interpolate_xs(&self.nominal_energies, xs, e_corr);
            for (j, &sigma) in xs_interp.iter().enumerate() {
                neg_opt[j] -= density * sigma;
            }
        }
        let transmission: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

        // Resolution broadening on the corrected grid.  The corrected
        // grid changes with (t0, l_scale), and within one LM iteration
        // the FD columns for t0 and L_scale rebuild the plan twice
        // each — on real-VENUS A.1 that miss rate outweighs the
        // density-column hits and leaves the plan cache net-negative.
        // The non-plan path stays here until the aux-grid hoist
        // (#459 B1/B2) gives us a grid that's fixed across the entire
        // Jacobian call.
        if let Some(inst) = &self.instrument {
            let t_broadened = resolution::apply_resolution(e_corr, &transmission, &inst.resolution)
                .map_err(|e| {
                    FittingError::EvaluationFailed(format!("resolution broadening: {e}"))
                })?;
            Ok(t_broadened)
        } else {
            Ok(transmission)
        }
    }
}

impl FitModel for EnergyScaleTransmissionModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let t0 = params[self.t0_index];
        let l_scale = params[self.l_scale_index];
        let e_corr = self.corrected_energies(t0, l_scale);
        self.evaluate_at(params, &e_corr)
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

        // Compute unresolved T and interpolated cross-sections for density derivatives
        let mut neg_opt = vec![0.0f64; n_e];
        let mut interp_xs_all: Vec<Vec<f64>> = Vec::new();
        for (i, xs) in self.cross_sections.iter().enumerate() {
            let density = params[self.density_indices[i]];
            let xs_interp = Self::interpolate_xs(&self.nominal_energies, xs, &e_corr);
            for (j, &sigma) in xs_interp.iter().enumerate() {
                neg_opt[j] -= density * sigma;
            }
            interp_xs_all.push(xs_interp);
        }
        let t_unresolved: Vec<f64> = neg_opt.iter().map(|&d| d.exp()).collect();

        // Density-column plan cache: when the Jacobian has ≥ 2 density
        // columns sharing the current (t0, l_scale) grid, build one
        // broadening plan here and reuse it across every density-col
        // `apply_resolution_with_plan`.  FD columns (t0, l_scale) go
        // through `self.evaluate(p_plus/minus)` → `apply_resolution`
        // unchanged — they each work at a DIFFERENT (t0, l_scale) so
        // a single-grid plan would miss every time.
        //
        // Hit-rate analysis (see PR body):
        //   - N_density = 1 (A.1 / B.1 / KL+grouped+TZERO): 1 hit, plan
        //     build ≈ broaden cost → net-neutral.  Guard `>= 2` makes
        //     this path a no-op: `density_plan` stays None, and
        //     `apply_resolution_with_plan(None, …)` forwards to the
        //     original `apply_resolution`, bit-exact.
        //   - N_density = 6 (B.3, KL+per-iso+TZERO): 6 hits → ~45%
        //     speedup on density-col broadening + ~30–40% wall on
        //     the Jacobian call.
        let n_density_cols = free_param_indices
            .iter()
            .filter(|&&fp_idx| fp_idx != self.t0_index && fp_idx != self.l_scale_index)
            .count();
        // Only tabulated resolution has a meaningful plan; Gaussian
        // would burn an O(n) sorted-grid validation only to return
        // `None`, so we short-circuit here and stay byte-identical to
        // the pre-cache Gaussian hot path (Copilot #1).
        let density_plan = if n_density_cols >= 2
            && let Some(inst) = &self.instrument
            && matches!(
                inst.resolution,
                nereids_physics::resolution::ResolutionFunction::Tabulated(_)
            ) {
            resolution::build_resolution_plan(&e_corr, &inst.resolution)
                .ok()
                .flatten()
        } else {
            None
        };

        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            if fp_idx == self.t0_index || fp_idx == self.l_scale_index {
                // Finite difference for energy-scale parameters
                let h = if fp_idx == self.t0_index { 1e-4 } else { 1e-7 };
                let mut p_plus = params.to_vec();
                let mut p_minus = params.to_vec();
                p_plus[fp_idx] += h;
                p_minus[fp_idx] -= h;
                let y_plus = match self.evaluate(&p_plus) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                let y_minus = match self.evaluate(&p_minus) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                for i in 0..n_e {
                    *jacobian.get_mut(i, col) = (y_plus[i] - y_minus[i]) / (2.0 * h);
                }
            } else {
                // Density parameter: analytical derivative
                // ∂T/∂n_g = -(Σ_{iso∈g} σ_iso(E)) · T_unresolved(E)
                let mut sigma_sum = vec![0.0f64; n_e];
                for (iso, &di) in self.density_indices.iter().enumerate() {
                    if di == fp_idx {
                        for (j, &sigma) in interp_xs_all[iso].iter().enumerate() {
                            sigma_sum[j] += sigma;
                        }
                    }
                }
                let inner_deriv: Vec<f64> =
                    (0..n_e).map(|i| -sigma_sum[i] * t_unresolved[i]).collect();

                // Apply resolution to derivative if enabled.
                //
                // When `density_plan` is Some (N_density ≥ 2) we
                // hit the hoisted broadening plan built above.  When
                // None (N_density ≤ 1 or Gaussian resolution),
                // `apply_resolution_with_plan(None, …)` transparently
                // forwards to `apply_resolution` — bit-exact with
                // pre-cache main.
                if let Some(inst) = &self.instrument {
                    let resolved_deriv = match resolution::apply_resolution_with_plan(
                        density_plan.as_ref(),
                        &e_corr,
                        &inner_deriv,
                        &inst.resolution,
                    ) {
                        Ok(v) => v,
                        Err(_) => return None,
                    };
                    for (i, &val) in resolved_deriv.iter().enumerate() {
                        *jacobian.get_mut(i, col) = val;
                    }
                } else {
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
        if self.cross_sections.is_empty() {
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
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};

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

    fn u238_single_resonance() -> ResonanceData {
        ResonanceData {
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
                naps: 1,
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
                r_external: vec![],
            }],
        }
    }

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
                urr: None,
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
        }
    }

    // ── Cubature dispatch tests (epic #472, PR #474b) ───────────────────

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
    /// (matrix, σ stack) pair, with the canonical codex04 training
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
        // round-3 P2 guard refuses the dispatch when the active
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
        // bypassing an unknown resolution operator (Codex round-2
        // P2 guard).
        let mut model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas.clone()),
            density_indices: Arc::new(vec![0, 1]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(make_trivial_instrument()),
            resolution_plan: Some(Arc::clone(&plan)),
            sparse_cubature_plan: Some(Arc::clone(&cubature)),
            sparse_scalar_plan: None,
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
        };
        let model_without_plan = PrecomputedTransmissionModel {
            cross_sections: Arc::new(sigmas_k1.clone()),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies.clone())),
            instrument: Some(make_trivial_instrument()),
            resolution_plan: None,
            sparse_cubature_plan: None,
            sparse_scalar_plan: None,
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
        // No cubature installed → byte-identical to pre-PR #474b
        // main.  This is the regression guard: the dispatch addition
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
    // not the precomputed variant.  Claude round-1 P1 on PR #480.

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
        // the round-2 eligibility guard requires `resolution_plan.is_some()`
        // so the cubature doesn't silently bypass an unknown
        // resolution operator (Codex round-2 P2 on PR #480).
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
                "stale-grid cubature plan MUST NOT fire; evaluate() must match no-plan byte-exactly \
                 (Codex PR #480 round-1 P1 regression guard)",
            );
        }
    }

    #[test]
    fn fit_model_cubature_falls_back_when_density_escapes_box() {
        // Build cubature with train_max = [1e-4, 1e-4], install
        // the density_box, then call evaluate() with a density
        // WELL beyond the 1.5× tolerance.  Dispatch must fall back
        // to the exact path rather than silently extrapolate the
        // surrogate outside its trained region.  Codex round-4 P1
        // on PR #480.
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
        // otherwise).  The cubature fast path MUST still fire — the
        // prior round-2 `resolution_plan.is_some()` requirement
        // made the new API inert on this surface.  Codex separate-
        // review finding on PR #481.
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
    // Claude round-1 P2-#8 on PR #475.  The cubature tests above cover
    // the k ≥ 2 path; the scalar path is a separate surrogate with its
    // own eligibility guard (`scalar_eligible`) and its own
    // density-box guard (`scalar_density_within_box`).  These tests
    // exercise the scalar-specific guards: k=1-only, grid-identity
    // via `to_bits()`, tabulated-only instrument resolution,
    // density-box escape, and that the pure no-plan path remains
    // byte-identical to pre-PR #475 main.

    /// Helper: build a synthetic scalar (k = 1) Chebyshev plan on the
    /// same grid as the cubature helpers.
    fn build_scalar_plan(
        matrix: &nereids_physics::resolution::ResolutionMatrix,
        sigma_k1: &[f64],
        n_max: f64,
    ) -> Arc<ScalarSurrogatePlan> {
        Arc::new(
            nereids_physics::surrogate::ScalarChebyshevPlan::build(matrix, sigma_k1, n_max, 16)
                .expect("synthetic scalar Chebyshev build"),
        )
    }

    #[test]
    fn fit_model_scalar_dispatches_at_k1() {
        // k = 1 with both the scalar plan and the resolution plan
        // installed: evaluate() must return the scalar plan's
        // forward output.  At the Chebyshev node density the plan
        // reproduces exact `apply_r ∘ exp(-nσ)` to machine
        // precision, so comparing to the plan's own forward is a
        // byte-identical check.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(&matrix, &sigmas[0], n_max);

        let model = make_trivial_fit_model(energies.clone(), 1)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_scalar_plan(Some(Arc::clone(&scalar)));

        let n = [0.5 * n_max];
        let t_model = model.evaluate(&n).unwrap();
        let t_scalar = scalar.forward_scalar(n[0]);
        assert_eq!(t_model.len(), n_grid);
        for (a, b) in t_model.iter().zip(t_scalar.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "TransmissionFitModel scalar dispatch must return forward_scalar() byte-exact",
            );
        }
    }

    #[test]
    fn fit_model_scalar_jacobian_matches_scalar_derivative() {
        // analytical_jacobian at a density inside the box must
        // return exactly `scalar.forward_and_derivative_scalar(n)`'s
        // derivative as the single column.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(&matrix, &sigmas[0], n_max);

        let model = make_trivial_fit_model(energies, 1)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_scalar_plan(Some(Arc::clone(&scalar)));

        let n = [0.5 * n_max];
        let y_curr = model.evaluate(&n).unwrap();
        let jac = model
            .analytical_jacobian(&n, &[0], &y_curr)
            .expect("scalar Jacobian path on TransmissionFitModel");
        let (_t_ref, dt_ref) = scalar.forward_and_derivative_scalar(n[0]);
        assert_eq!(jac.ncols, 1);
        assert_eq!(jac.nrows, n_grid);
        for (i, &dt_i) in dt_ref.iter().enumerate().take(n_grid) {
            assert_eq!(
                jac.get(i, 0).to_bits(),
                dt_i.to_bits(),
                "row {i}: TransmissionFitModel must return scalar dT/dn byte-exact",
            );
        }
    }

    #[test]
    fn fit_model_scalar_falls_back_at_k2() {
        // k = 2 with a scalar plan installed → `scalar_eligible`
        // returns false (`n_density_params != 1`), dispatch MUST
        // fall back to the exact `exp(-Σ n σ) + apply_resolution`
        // path.  Prove fallback via byte-identity vs a no-plan
        // model with the same k = 2 grid.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas_k2 = synthetic_sigmas(n_grid, 2);
        let sigma_k1_stale = synthetic_sigmas(n_grid, 1).remove(0);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(&matrix, &sigma_k1_stale, n_max);

        // Model has k = 2 but a scalar plan is installed → must
        // fall back (scalar_eligible rejects `n_density_params != 1`).
        let model_with_plan = make_trivial_fit_model(energies.clone(), 2)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_scalar_plan(Some(Arc::clone(&scalar)));
        let model_without_plan =
            make_trivial_fit_model(energies, 2).with_resolution_plan(Some(Arc::clone(&plan)));

        // Feed both models the same σ stack so the exact path is
        // identical.  `TransmissionFitModel` uses its own internal
        // computed σ path — in this synthetic test both models
        // build the same way from make_trivial_fit_model and no
        // σ is precomputed, so evaluate() builds σ from each
        // isotope's resonance data (empty ranges → σ ≡ 0).
        //   With σ ≡ 0: exp(-Σ n σ) = 1 everywhere, so both
        //   evaluate outputs are the broadened 1-vector.  Byte
        //   identity proves the scalar plan did NOT fire
        //   (if it had, output would use `sigma_k1_stale` and
        //   differ).
        let _ = sigmas_k2;
        let n = [1e-4_f64, 2e-4];
        let t_with = model_with_plan.evaluate(&n).unwrap();
        let t_without = model_without_plan.evaluate(&n).unwrap();
        for (a, b) in t_with.iter().zip(t_without.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "scalar plan MUST NOT fire at k=2; evaluate() must match no-plan byte-exactly",
            );
        }
    }

    #[test]
    fn fit_model_scalar_falls_back_on_grid_mismatch() {
        // Build a scalar plan on one grid, install on a model with
        // a DIFFERENT same-length grid.  Dispatch must refuse via
        // the `to_bits()` grid-identity check.
        let n_grid = 40_usize;
        let (energies_a, plan_a, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(&matrix, &sigmas[0], n_max);

        // Shifted grid — same length, different bit pattern.
        let energies_b: Vec<f64> = energies_a.iter().map(|&e| e + 1.0).collect();

        // Install the stale (grid-A) scalar plan on a grid-B model.
        // Note: install without a resolution plan so only the
        // scalar plan's grid-identity check gates dispatch.
        let _ = plan_a;
        let model_with_stale_plan =
            make_trivial_fit_model(energies_b.clone(), 1).with_sparse_scalar_plan(Some(scalar));
        let model_without_plan = make_trivial_fit_model(energies_b, 1);

        let n = [0.25 * n_max];
        let t_stale = model_with_stale_plan.evaluate(&n).unwrap();
        let t_exact = model_without_plan.evaluate(&n).unwrap();
        for (a, b) in t_stale.iter().zip(t_exact.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "stale-grid scalar plan MUST NOT fire; evaluate() must match no-plan byte-exactly",
            );
        }
    }

    #[test]
    fn fit_model_scalar_falls_back_when_density_escapes_box() {
        // scalar.density_box() auto-set to n_max during build.  A
        // density at 2× n_max is > 1.5× tolerance → guard fires →
        // fall back to exact path byte-identically.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(&matrix, &sigmas[0], n_max);

        let model_with = make_trivial_fit_model(energies.clone(), 1)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_scalar_plan(Some(Arc::clone(&scalar)));
        let model_without =
            make_trivial_fit_model(energies, 1).with_resolution_plan(Some(Arc::clone(&plan)));

        // Escape: 2× the training n_max → beyond the 1.5× tolerance.
        let n_escape = [2.0 * n_max];
        let t_with = model_with.evaluate(&n_escape).unwrap();
        let t_without = model_without.evaluate(&n_escape).unwrap();
        for (a, b) in t_with.iter().zip(t_without.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "scalar density-box escape guard MUST fall back to exact path byte-identically",
            );
        }
    }

    #[test]
    fn fit_model_scalar_rejects_nonfinite_and_negative_density() {
        // `scalar_density_within_box` rejects NaN, ±∞, and negative
        // densities — falling back to exact.  Prove via byte-identity
        // against a no-plan model.
        let n_grid = 40_usize;
        let (energies, plan, matrix) = synthetic_resolution_setup(n_grid, 4);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 2.0 * 1e-4_f64;
        let scalar = build_scalar_plan(&matrix, &sigmas[0], n_max);

        let model_with = make_trivial_fit_model(energies.clone(), 1)
            .with_resolution_plan(Some(Arc::clone(&plan)))
            .with_sparse_scalar_plan(Some(Arc::clone(&scalar)));
        let model_without =
            make_trivial_fit_model(energies, 1).with_resolution_plan(Some(Arc::clone(&plan)));

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
        // without going through the model dispatch.  Verifies the
        // 1.5× tolerance exactly matches the cubature's convention.
        let n_grid = 16_usize;
        let (_energies, _plan, matrix) = synthetic_resolution_setup(n_grid, 2);
        let sigmas = synthetic_sigmas(n_grid, 1);
        let n_max = 1e-4_f64;
        let plan =
            nereids_physics::surrogate::ScalarChebyshevPlan::build(&matrix, &sigmas[0], n_max, 16)
                .expect("build");

        // Inside the box.
        assert!(scalar_density_within_box(&plan, 0.0));
        assert!(scalar_density_within_box(&plan, 0.5 * n_max));
        assert!(scalar_density_within_box(&plan, n_max));
        // Inside the 1.5× tolerance.
        assert!(scalar_density_within_box(&plan, 1.49 * n_max));
        assert!(scalar_density_within_box(&plan, 1.5 * n_max));
        // Beyond the tolerance.
        assert!(!scalar_density_within_box(&plan, 1.5001 * n_max));
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
        // j).  Codex round-4 P2 on PR #480.
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
    #[test]
    fn energy_scale_corrected_energies() {
        let xs = vec![vec![1.0; 5]];
        let energies = vec![10.0, 20.0, 50.0, 100.0, 200.0];
        let model = EnergyScaleTransmissionModel::new(
            std::sync::Arc::new(xs),
            std::sync::Arc::new(vec![0]),
            energies.clone(),
            25.0,
            1, // t0_index
            2, // l_scale_index
            None,
        );

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

    /// Verify interpolate_xs produces correct results.
    #[test]
    fn energy_scale_interpolate_xs() {
        let nominal_e = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let xs = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        // Exact grid points
        let result =
            EnergyScaleTransmissionModel::interpolate_xs(&nominal_e, &xs, &[1.0, 3.0, 5.0]);
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 30.0).abs() < 1e-10);
        assert!((result[2] - 50.0).abs() < 1e-10);

        // Midpoints
        let result = EnergyScaleTransmissionModel::interpolate_xs(&nominal_e, &xs, &[1.5, 2.5]);
        assert!((result[0] - 15.0).abs() < 1e-10);
        assert!((result[1] - 25.0).abs() < 1e-10);

        // Out of range: clamp
        let result = EnergyScaleTransmissionModel::interpolate_xs(&nominal_e, &xs, &[0.5, 6.0]);
        assert!((result[0] - 10.0).abs() < 1e-10); // below range
        assert!((result[1] - 50.0).abs() < 1e-10); // above range
    }

    /// Verify evaluate returns identity at t0=0, l_scale=1.
    #[test]
    fn energy_scale_evaluate_identity() {
        let xs = vec![vec![1.0, 2.0, 3.0, 2.0, 1.5]];
        let energies = vec![4.0, 9.0, 16.0, 25.0, 36.0];
        let model_es = EnergyScaleTransmissionModel::new(
            std::sync::Arc::new(xs.clone()),
            std::sync::Arc::new(vec![0]),
            energies.clone(),
            25.0,
            1, // t0_index
            2, // l_scale_index
            None,
        );

        let model_pre = make_precomputed(xs, vec![0]);

        let density = 0.1;
        let params_es = [density, 0.0, 1.0]; // t0=0, l_scale=1
        let params_pre = [density];

        let y_es = model_es.evaluate(&params_es).unwrap();
        let y_pre = model_pre.evaluate(&params_pre).unwrap();

        for (i, (&a, &b)) in y_es.iter().zip(y_pre.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "bin {i}: energy_scale={a}, precomputed={b}"
            );
        }
    }

    /// Jacobian for energy-scale model: density columns must match FD.
    #[test]
    fn energy_scale_jacobian_density_matches_fd() {
        let xs = vec![vec![1.0, 2.0, 3.0, 2.0, 1.5]];
        let energies = vec![4.0, 9.0, 16.0, 25.0, 36.0];
        let model = EnergyScaleTransmissionModel::new(
            std::sync::Arc::new(xs),
            std::sync::Arc::new(vec![0]),
            energies.clone(),
            25.0,
            1,
            2,
            None,
        );

        let params = [0.1, 0.5, 1.002]; // density, t0, l_scale
        let y = model.evaluate(&params).unwrap();
        let free = vec![0, 1, 2];
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
        let n = 200;
        let energies: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64) * 0.5).collect();
        // Breit-Wigner-like cross-section: σ = 100 / ((E-50)^2 + 1)
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| 100.0 / ((e - 50.0).powi(2) + 1.0))
            .collect();

        let true_density = 0.001;
        let true_ls = 1.003;

        let model = EnergyScaleTransmissionModel::new(
            std::sync::Arc::new(vec![xs]),
            std::sync::Arc::new(vec![0]),
            energies,
            25.0,
            1, // t0_index (fixed at 0)
            2, // l_scale_index
            None,
        );
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
}
