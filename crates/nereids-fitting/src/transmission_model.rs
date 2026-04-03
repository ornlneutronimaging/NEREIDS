//! Transmission forward model adapter for fitting.
//!
//! Wraps the physics `forward_model` function into a `FitModel` trait object
//! that the LM optimizer can call. The fit parameters are the areal densities
//! (thicknesses) of each isotope in the sample.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_physics::resolution;
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
}

impl FitModel for PrecomputedTransmissionModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        if self.cross_sections.is_empty() {
            return Err(FittingError::InvalidConfig(
                "PrecomputedTransmissionModel.cross_sections must not be empty".into(),
            ));
        }
        let n_e = self.cross_sections[0].len();
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
            let t_broadened =
                resolution::apply_resolution(energies, &transmission, &inst.resolution).map_err(
                    |e| FittingError::EvaluationFailed(format!("resolution broadening: {e}")),
                )?;
            Ok(t_broadened)
        } else {
            Ok(transmission)
        }
    }

    /// Analytical Jacobian for the Beer-Lambert transmission model.
    ///
    /// T(E) = exp(-Σᵢ nᵢ · σᵢ(E))
    /// ∂T/∂nᵢ = -σᵢ(E) · T(E)
    ///
    /// When resolution is enabled, the correct derivative is
    /// ∂[R⊗T]/∂nᵢ = R⊗[-σᵢ·T] (resolution is a linear operator).
    /// This is not yet implemented (issue #442 Step 4); returns `None`
    /// to fall back to finite-difference Jacobians.
    ///
    /// Costs O(N_energy × N_isotopes) with zero extra evaluate() calls,
    /// because T(E) is already in `y_current` from the LM loop.
    /// This eliminates N_free extra evaluate() calls per LM iteration
    /// compared to finite-difference Jacobians.
    fn analytical_jacobian(
        &self,
        _params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        // Issue #442: fall back to FD when resolution is enabled.
        // The analytical form ∂[R⊗T]/∂nᵢ is deferred to Step 4.
        if self.instrument.is_some() {
            return None;
        }

        let n_e = y_current.len();
        let n_free = free_param_indices.len();

        // For each free parameter, sum the cross-sections of every isotope
        // tied to that parameter index.  The Beer-Lambert derivative is:
        //   ∂T/∂n_fp = -T(E) · Σ_{iso: density_indices[iso]==fp_idx} σ_iso(E)
        // Using only the first match (via .position) would give the wrong
        // gradient whenever multiple isotopes share one density parameter.
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

        // jacobian.get(i, j) = ∂T(E_i)/∂params[free_param_indices[j]]
        //                    = -(Σ σ_iso(E_i)) · T(E_i)   (Beer-Lambert derivative)
        let mut jacobian = FlatMatrix::zeros(n_e, n_free);
        for i in 0..n_e {
            for (j, xs_sum) in fp_xs_sums.iter().enumerate() {
                *jacobian.get_mut(i, j) = -xs_sum[i] * y_current[i];
            }
        }

        Some(jacobian)
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
        })
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
            Ok(neg_opt.iter().map(|&d| d.exp()).collect())
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
    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        // Issue #442 Step 4: when resolution is enabled, the analytical
        // Jacobian ∂T/∂nᵢ = -σᵢ·T assumes T = exp(-Σnσ), but the actual
        // model is T_out = R⊗exp(-Σnσ).  The correct form
        // ∂[R⊗T]/∂nᵢ = R⊗[-σᵢ·T] is not yet implemented.
        // Fall back to finite-difference.
        if self.instrument.is_some() {
            return None;
        }

        // Only provide analytical Jacobian when base_xs is available
        // (temperature-fitting fast path with cached broadened XS).
        let _base_xs_guard = self.base_xs.as_ref()?;
        let cached_xs = self.cached_broadened_xs.borrow();
        let broadened_xs = cached_xs.as_ref()?;

        // Guard: verify the cache matches the current parameter temperature.
        // After a rejected LM trial step, evaluate() may have updated the cache
        // to the trial temperature while the solver reverted params to the
        // accepted point.  Using stale XS would give an incorrect Jacobian.
        if let Some(ti) = self.temperature_index {
            let param_temp = params[ti];
            if (param_temp - self.cached_temperature.get()).abs() > 1e-15 {
                // Cache is stale — fall back to finite-difference Jacobian.
                return None;
            }
        }

        let n_e = y_current.len();
        let n_free = free_param_indices.len();
        let mut jacobian = FlatMatrix::zeros(n_e, n_free);

        // Identify which free parameter column is the temperature (if any).
        let temp_col = self
            .temperature_index
            .and_then(|ti| free_param_indices.iter().position(|&fp| fp == ti));

        // ── Density columns: ∂T/∂nᵢ = -σᵢ(E)·T(E) ──
        // Same formula as PrecomputedTransmissionModel::analytical_jacobian.
        for (col, &fp_idx) in free_param_indices.iter().enumerate() {
            if temp_col == Some(col) {
                continue; // temperature column handled below
            }
            // Sum ratio-weighted cross-sections of all isotopes tied to this free parameter.
            // ∂T/∂N_g = -T(E) · Σ_{i∈g} rᵢ · σᵢ(E)
            let mut sigma_sum = vec![0.0f64; n_e];
            for (iso, &di) in self.density_indices.iter().enumerate() {
                if di == fp_idx {
                    let ratio = self.density_ratios[iso];
                    for (j, &sigma) in broadened_xs[iso].iter().enumerate() {
                        sigma_sum[j] += ratio * sigma;
                    }
                }
            }
            for i in 0..n_e {
                *jacobian.get_mut(i, col) = -sigma_sum[i] * y_current[i];
            }
        }

        // ── Temperature column: analytical ∂T/∂T_temp ──
        //
        // Chain rule: ∂T(E)/∂T_temp = -T(E) · Σᵢ nᵢ · rᵢ · ∂σᵢ(E)/∂T
        //
        // Computed on-demand here (not in evaluate()) because evaluate()
        // is called many times during line search trials where the
        // derivative is not needed. Computing it here costs one extra
        // broadening call per Jacobian, same as the old FD approach but
        // with exact (not approximate) derivatives.
        if let Some(col) = temp_col {
            // Compute ∂σ/∂T on-demand if not cached at current temperature.
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
            for i in 0..n_e {
                let mut sum_n_dsigma = 0.0f64;
                for (iso, dxs) in dxs_dt.iter().enumerate() {
                    let density = params[self.density_indices[iso]];
                    let ratio = self.density_ratios[iso];
                    sum_n_dsigma += density * ratio * dxs[i];
                }
                *jacobian.get_mut(i, col) = -y_current[i] * sum_n_dsigma;
            }
        }

        Some(jacobian)
    }
}

/// Wraps a transmission model with SAMMY-style normalization and background.
///
/// T_out(E) = Anorm × T_inner(E) + BackA + BackB / √E + BackC × √E
///
/// The normalization and background parameters are additional entries in the
/// parameter vector, appended after the density (and optional temperature)
/// parameters of the inner model.
///
/// ## SAMMY Reference
/// SAMMY manual Sec III.E.2 — NORMAlization and BACKGround cards.
/// SAMMY fits up to 4 background terms; we implement the same 4:
///   Anorm, constant BackA, 1/√E term BackB, √E term BackC.
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
}

impl<M: FitModel> NormalizedTransmissionModel<M> {
    /// Create a new normalized transmission model.
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

        let result: Vec<f64> = t_inner
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                anorm * t
                    + back_a
                    + back_b * self.inv_sqrt_energies[i]
                    + back_c * self.sqrt_energies[i]
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
        let bg_indices = [
            self.anorm_index,
            self.back_a_index,
            self.back_b_index,
            self.back_c_index,
        ];

        // Collect inner model's free param indices (those not in bg_indices).
        let inner_free_indices: Vec<usize> = free_param_indices
            .iter()
            .copied()
            .filter(|idx| !bg_indices.contains(idx))
            .collect();

        // Get inner Jacobian if there are inner free params.
        // y_current for the inner model is t_inner, not the outer y_current.
        let inner_jac = if !inner_free_indices.is_empty() {
            self.inner
                .analytical_jacobian(params, &inner_free_indices, &t_inner)
        } else {
            None
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
        self.anorm_index
            .max(self.back_a_index)
            .max(self.back_b_index)
            .max(self.back_c_index)
            + 1
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
        }
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

    /// Issue #442: PrecomputedTransmissionModel analytical Jacobian must
    /// fall back to None when resolution is enabled.
    #[test]
    fn precomputed_jacobian_disabled_with_resolution() {
        use nereids_physics::resolution::ResolutionFunction;

        let energies: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.1).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(vec![vec![10.0; 100]]),
            density_indices: Arc::new(vec![0]),
            energies: Some(Arc::new(energies)),
            instrument: Some(inst),
        };
        let params = [0.001f64];
        let y = model.evaluate(&params).unwrap();
        assert!(
            model.analytical_jacobian(&params, &[0], &y).is_none(),
            "Analytical Jacobian must return None when resolution is enabled \
             (issue #442 Step 4 not yet implemented)"
        );
    }

    // ── Issue #442 Step 4: TransmissionFitModel Jacobian containment ───────

    /// Issue #442 Step 4: TransmissionFitModel analytical Jacobian must
    /// return None when resolution is enabled (density + temperature paths).
    #[test]
    fn transmission_fit_model_jacobian_disabled_with_resolution() {
        use nereids_physics::resolution::ResolutionFunction;

        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.05).collect();
        let inst = Arc::new(InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                nereids_physics::resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        });

        // Temperature-fitting path (base_xs present).
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            300.0,
            Some(inst),
            (vec![0], vec![1.0]),
            Some(1), // temperature_index
            None,    // external_base_xs — will be computed internally
        )
        .unwrap();

        // params = [density, temperature]
        let params = [0.0005, 300.0];
        let y = model.evaluate(&params).unwrap();

        assert!(
            model.analytical_jacobian(&params, &[0, 1], &y).is_none(),
            "TransmissionFitModel analytical Jacobian must return None \
             when resolution is enabled"
        );
    }

    /// Issue #442 Step 4: TransmissionFitModel analytical Jacobian must
    /// remain available when resolution is NOT enabled.
    #[test]
    fn transmission_fit_model_jacobian_available_without_resolution() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.05).collect();

        // Temperature-fitting path, no resolution.
        let model = TransmissionFitModel::new(
            energies,
            vec![data],
            300.0,
            None, // no instrument
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
}
