//! Transmission forward model adapter for fitting.
//!
//! Wraps the physics `forward_model` function into a `FitModel` trait object
//! that the LM optimizer can call. The fit parameters are the areal densities
//! (thicknesses) of each isotope in the sample.

use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

use crate::lm::FitModel;

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
    pub density_indices: Vec<usize>,
}

impl FitModel for PrecomputedTransmissionModel {
    fn evaluate(&self, params: &[f64]) -> Vec<f64> {
        assert!(
            !self.cross_sections.is_empty(),
            "PrecomputedTransmissionModel.cross_sections must not be empty"
        );
        let n_e = self.cross_sections[0].len();
        let mut neg_opt = vec![0.0f64; n_e];
        for (i, xs) in self.cross_sections.iter().enumerate() {
            let density = params[self.density_indices[i]];
            if density > 0.0 {
                for (j, &sigma) in xs.iter().enumerate() {
                    neg_opt[j] -= density * sigma;
                }
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
    ) -> Option<Vec<Vec<f64>>> {
        let n_e = y_current.len();

        // Build lookup: column j (free param index) → cross-section slice.
        // density_indices[iso] = parameter index that controls isotope `iso`.
        let fp_to_xs: Vec<Option<&[f64]>> = free_param_indices
            .iter()
            .map(|&fp_idx| {
                self.density_indices
                    .iter()
                    .position(|&di| di == fp_idx)
                    .map(|iso| self.cross_sections[iso].as_slice())
            })
            .collect();

        // jacobian[i][j] = ∂T(E_i)/∂params[free_param_indices[j]]
        //                = -σ_j(E_i) · T(E_i)   (Beer-Lambert derivative)
        let jacobian: Vec<Vec<f64>> = (0..n_e)
            .map(|i| {
                fp_to_xs
                    .iter()
                    .map(|xs_opt| match xs_opt {
                        Some(xs) => -xs[i] * y_current[i],
                        None => 0.0,
                    })
                    .collect()
            })
            .collect();

        Some(jacobian)
    }
}

/// Forward model for fitting isotopic areal densities from transmission data.
///
/// The model computes T(E) for a set of isotopes with variable areal densities.
/// Each isotope's resonance data and the energy grid are fixed; only the
/// areal densities are adjusted during fitting.
///
/// `instrument` uses `Arc` so that parallel pixel loops (e.g. in `sparse.rs`)
/// can share one copy of a potentially large tabulated resolution kernel
/// via cheap reference-count increments instead of deep-cloning per pixel.
pub struct TransmissionFitModel {
    /// Energy grid (eV), ascending.
    pub energies: Vec<f64>,
    /// Resonance data for each isotope.
    pub resonance_data: Vec<ResonanceData>,
    /// Sample temperature in Kelvin.
    pub temperature_k: f64,
    /// Optional instrument resolution parameters (Arc-shared for parallel use).
    pub instrument: Option<Arc<InstrumentParams>>,
    /// Index mapping: which `params` indices correspond to areal densities.
    /// params[density_indices[i]] = areal density of isotope i.
    pub density_indices: Vec<usize>,
}

impl FitModel for TransmissionFitModel {
    fn evaluate(&self, params: &[f64]) -> Vec<f64> {
        let isotopes: Vec<(ResonanceData, f64)> = self
            .resonance_data
            .iter()
            .zip(self.density_indices.iter())
            .map(|(rd, &idx): (&ResonanceData, &usize)| (rd.clone(), params[idx]))
            .collect();

        let sample = SampleParams {
            temperature_k: self.temperature_k,
            isotopes,
        };

        transmission::forward_model(&self.energies, &sample, self.instrument.as_deref())
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
            density_indices: vec![0, 1],
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
            density_indices: vec![0, 1],
        };

        let params = [0.2f64, 0.4f64];
        let y = model.evaluate(&params);
        let free = vec![0usize, 1usize];

        let jac = model
            .analytical_jacobian(&params, &free, &y)
            .expect("analytical_jacobian should return Some(_)");

        assert_eq!(jac.len(), 3); // n_energies
        for row in &jac {
            assert_eq!(row.len(), 2); // n_free_params
        }

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
                let ana = jac[i][col];
                assert!(
                    (fd - ana).abs() < 1e-6,
                    "Jacobian mismatch (row {i}, col {col}): FD={fd:.8}, analytical={ana:.8}"
                );
            }
        }
    }

    // ── TransmissionFitModel ─────────────────────────────────────────────────

    fn u238_single_resonance() -> ResonanceData {
        ResonanceData {
            isotope: Isotope::new(92, 238),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
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
        };

        let y_obs = model.evaluate(&[true_thickness]);
        let sigma = vec![0.01; y_obs.len()]; // 1% uncertainty

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("thickness", 0.001), // initial guess 2× off
        ]);

        let result =
            lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

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
            isotope: Isotope::new(1, 10),
            za: 1010,
            awr: 10.0,
            ranges: vec![ResonanceRange {
                energy_low: 0.0,
                energy_high: 100.0,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 5.0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 10.0,
                    apl: 5.0,
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
        };

        let y_obs = model.evaluate(&[true_t1, true_t2]);
        let sigma = vec![0.01; y_obs.len()];

        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("U-238 thickness", 0.001),
            FitParameter::non_negative("Other thickness", 0.001),
        ]);

        let result =
            lm::levenberg_marquardt(&model, &y_obs, &sigma, &mut params, &LmConfig::default());

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
}
