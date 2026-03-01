//! Transmission forward model adapter for fitting.
//!
//! Wraps the physics `forward_model` function into a `FitModel` trait object
//! that the LM optimizer can call. The fit parameters are the areal densities
//! (thicknesses) of each isotope in the sample.

use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};

/// Transmission model backed by precomputed broadened cross-sections.
///
/// The expensive physics steps (resonance → σ(E), Doppler broadening,
/// resolution broadening) are computed once and stored.  Each `evaluate()`
/// call performs only the Beer-Lambert step:
///
///   T(E) = exp(−Σᵢ nᵢ · σ_D,i(E))
///
/// which is O(N_energy) instead of O(N_energy × N_resonances).  For a
/// 128×128 spatial map this is ~100–1000× faster than `TransmissionFitModel`.
///
/// Construct via `nereids_physics::transmission::broadened_cross_sections`,
/// then wrap in `Arc` so the same precomputed data is shared read-only
/// across all rayon worker threads.
pub struct PrecomputedTransmissionModel {
    /// Broadened cross-sections σ_D(E) per isotope, shape [n_isotopes][n_energies].
    pub cross_sections: Arc<Vec<Vec<f64>>>,
    /// Mapping: `params[density_indices[i]]` is the density of isotope `i`.
    ///
    /// Wrapped in `Arc` so that parallel pixel loops can share one copy
    /// via cheap reference-count increments instead of deep-cloning per pixel.
    ///
    /// Kept `pub` (not `pub(crate)`) because the Python bindings
    /// (`nereids-python`) construct and access this field directly.
    pub density_indices: Arc<Vec<usize>>,
}

impl FitModel for PrecomputedTransmissionModel {
    fn evaluate(&self, params: &[f64]) -> Vec<f64> {
        assert!(
            !self.cross_sections.is_empty(),
            "PrecomputedTransmissionModel.cross_sections must not be empty"
        );
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
        neg_opt.iter().map(|&d| d.exp()).collect()
    }

    /// Analytical Jacobian for the Beer-Lambert transmission model.
    ///
    /// T(E) = exp(-Σᵢ nᵢ · σᵢ(E))
    /// ∂T/∂nᵢ = -σᵢ(E) · T(E)
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
/// is recomputed at `params[idx]` on every `evaluate()` call, and the LM
/// engine will compute a finite-difference Jacobian column for it.
///
/// `instrument` uses `Arc` so that parallel pixel loops (e.g. in `sparse.rs`)
/// can share one copy of a potentially large tabulated resolution kernel
/// via cheap reference-count increments instead of deep-cloning per pixel.
pub struct TransmissionFitModel {
    /// Energy grid (eV), ascending.
    pub energies: Vec<f64>,
    /// Resonance data for each isotope.
    pub resonance_data: Vec<ResonanceData>,
    /// Sample temperature in Kelvin (used when `temperature_index` is `None`).
    pub temperature_k: f64,
    /// Optional instrument resolution parameters (Arc-shared for parallel use).
    pub instrument: Option<Arc<InstrumentParams>>,
    /// Index mapping: which `params` indices correspond to areal densities.
    /// params[density_indices[i]] = areal density of isotope i.
    ///
    /// Uses `Vec<usize>` (not `Arc<Vec<usize>>`) because `TransmissionFitModel`
    /// is constructed fresh per pixel (via `fit_spectrum`) and never shared
    /// across threads.  `PrecomputedTransmissionModel` uses `Arc<Vec<usize>>`
    /// for its density_indices because it _is_ shared across rayon workers.
    pub density_indices: Vec<usize>,
    /// If `Some(idx)`, `params[idx]` is treated as the sample temperature (K)
    /// and included as a free parameter in the fit. The Doppler broadening
    /// kernel is recomputed at each `evaluate()` call.
    pub temperature_index: Option<usize>,
}

impl TransmissionFitModel {
    /// Create a validated `TransmissionFitModel`.
    ///
    /// # Errors
    /// Returns `FittingError::InvalidConfig` if `temperature_index` overlaps
    /// with `density_indices`.
    pub fn new(
        energies: Vec<f64>,
        resonance_data: Vec<ResonanceData>,
        temperature_k: f64,
        instrument: Option<Arc<InstrumentParams>>,
        density_indices: Vec<usize>,
        temperature_index: Option<usize>,
    ) -> Result<Self, FittingError> {
        if let Some(ti) = temperature_index
            && density_indices.contains(&ti)
        {
            return Err(FittingError::InvalidConfig(
                "temperature_index must not overlap with density_indices".into(),
            ));
        }
        Ok(Self {
            energies,
            resonance_data,
            temperature_k,
            instrument,
            density_indices,
            temperature_index,
        })
    }
}

impl FitModel for TransmissionFitModel {
    fn evaluate(&self, params: &[f64]) -> Vec<f64> {
        let isotopes: Vec<(ResonanceData, f64)> = self
            .resonance_data
            .iter()
            .zip(self.density_indices.iter())
            .map(|(rd, &idx): (&ResonanceData, &usize)| (rd.clone(), params[idx]))
            .collect();

        let temperature_k = match self.temperature_index {
            Some(idx) => params[idx],
            None => self.temperature_k,
        };

        let sample = SampleParams {
            temperature_k,
            isotopes,
        };

        // forward_model can fail for unsorted energies or invalid Doppler
        // params — both are configuration bugs (energies and isotope data are
        // set once at model construction).  The LM loop cannot fix these.
        transmission::forward_model(&self.energies, &sample, self.instrument.as_deref())
            .expect("TransmissionFitModel: forward_model failed (energy grid must be sorted ascending and Doppler params must be valid)")
    }
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
        let xs = Arc::new(vec![
            vec![1.0, 2.0, 3.0], // isotope 0
            vec![0.5, 0.5, 0.5], // isotope 1
        ]);
        let model = PrecomputedTransmissionModel {
            cross_sections: xs,
            density_indices: Arc::new(vec![0, 1]),
        };

        let params = [0.2f64, 0.4f64];
        let y = model.evaluate(&params);

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
        let xs = Arc::new(vec![
            vec![1.0, 2.0, 3.0], // isotope 0
            vec![0.5, 0.5, 0.5], // isotope 1
        ]);
        let model = PrecomputedTransmissionModel {
            cross_sections: xs,
            density_indices: Arc::new(vec![0, 1]),
        };

        let params = [0.2f64, 0.4f64];
        let y = model.evaluate(&params);
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

            let y_plus = model.evaluate(&p_plus);
            let y_minus = model.evaluate(&p_minus);

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
        let xs = Arc::new(vec![
            vec![1.0, 2.0, 3.0], // isotope 0
            vec![0.5, 1.0, 1.5], // isotope 1 — tied to same param
        ]);
        let model = PrecomputedTransmissionModel {
            cross_sections: xs,
            density_indices: Arc::new(vec![0, 0]), // both isotopes share param[0]
        };

        let params = [0.1f64];
        let y = model.evaluate(&params);
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
        }
    }

    #[test]
    fn test_recover_single_isotope_thickness() {
        let data = u238_single_resonance();
        let true_thickness = 0.0005;

        // Generate synthetic data
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
            temperature_index: None,
        };

        let y_obs = model.evaluate(&[true_thickness]);
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
                naps: 0,
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
            }],
        };

        let true_t1 = 0.0003;
        let true_t2 = 0.0001;

        let energies: Vec<f64> = (0..301).map(|i| 1.0 + (i as f64) * 0.1).collect();

        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![u238, other],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0, 1],
            temperature_index: None,
        };

        let y_obs = model.evaluate(&[true_t1, true_t2]);
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
        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data.clone()],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
            temperature_index: Some(1),
        };

        // Model with fixed temperature = 300 K (no temperature_index).
        let model_fixed = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data],
            temperature_k: 300.0,
            instrument: None,
            density_indices: vec![0],
            temperature_index: None,
        };

        let density = 0.0005;
        let y_via_index = model.evaluate(&[density, 300.0]);
        let y_via_fixed = model_fixed.evaluate(&[density]);

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
        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data],
            temperature_k: 0.0, // ignored — temperature_index is set
            instrument: None,
            density_indices: vec![0],
            temperature_index: Some(1), // params[1] = temperature
        };

        let y_obs = model.evaluate(&[true_density, true_temp]);
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

        // Noise-free synthetic data: optimizer should converge to within 0.1%.
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
}
