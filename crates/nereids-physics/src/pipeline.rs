//! Default forward model composing all pipeline stages.
//!
//! Pipeline order (Phase 1b):
//! 1. 0K cross sections (R-matrix)
//! 2. Beer-Lambert transmission
//! 3. Normalization + background
//!
//! Future phases will add:
//! - Doppler broadening (Phase 1c)
//! - Self-shielding (optional)
//! - Resolution convolution

use crate::rmatrix::cross_section::compute_0k_cross_sections;
use crate::transmission::beer_lambert::{compute_transmission, compute_transmission_with_jacobian};
use crate::transmission::normalization::{
    apply_normalization, apply_normalization_with_jacobian, NormalizationConfig,
};

use nereids_core::{
    EnergyGrid, ForwardModel, ForwardModelConfig, Parameter, PhysicsError, RMatrixParameters,
    ResolutionFunction,
};

/// Count the total number of free (vary=true) parameters in the R-matrix problem.
fn count_free_params(params: &RMatrixParameters) -> usize {
    let mut count = 0;
    for isotope in &params.isotopes {
        if isotope.abundance.vary {
            count += 1;
        }
        for sg in &isotope.spin_groups {
            for res in &sg.resonances {
                if res.energy.vary {
                    count += 1;
                }
                if res.gamma_n.vary {
                    count += 1;
                }
                if res.gamma_g.vary {
                    count += 1;
                }
                if let Some(ref f) = res.fission {
                    if f.gamma_f1.vary {
                        count += 1;
                    }
                    if f.gamma_f2.vary {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// The default forward model that composes all physics pipeline stages.
pub struct DefaultForwardModel {
    /// Optional resolution function. If `None`, no resolution broadening is applied.
    pub resolution: Option<Box<dyn ResolutionFunction>>,
}

impl ForwardModel for DefaultForwardModel {
    fn transmission(
        &self,
        energy: &EnergyGrid,
        params: &RMatrixParameters,
        config: &ForwardModelConfig,
    ) -> Result<Vec<f64>, PhysicsError> {
        // Phase 1b pipeline: cross sections → Beer-Lambert → normalization

        // 1. Compute 0K cross sections (Phase 1a)
        let cross_sections = compute_0k_cross_sections(energy, params, config)?;

        // 2. Apply Beer-Lambert law
        // For Phase 1b: use first isotope's thickness and number density
        // (assumes single-sample or all isotopes share same geometry)
        let (number_density, thickness_cm) = if params.isotopes.is_empty() {
            (0.0, 0.0) // Empty case
        } else {
            (
                params.isotopes[0].number_density,
                params.isotopes[0].thickness_cm,
            )
        };

        let transmission = compute_transmission(&cross_sections, number_density, thickness_cm)?;

        // 3. Apply normalization and background
        let norm_config = NormalizationConfig {
            normalization: Parameter::fixed(config.normalization),
            background: config.background.clone(),
        };
        let final_transmission = apply_normalization(&transmission, energy, &norm_config)?;

        Ok(final_transmission)
    }

    fn transmission_with_jacobian(
        &self,
        energy: &EnergyGrid,
        params: &RMatrixParameters,
        config: &ForwardModelConfig,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), PhysicsError> {
        // Phase 1b pipeline with Jacobians

        // 1. Compute 0K cross sections (Phase 1a)
        let cross_sections = compute_0k_cross_sections(energy, params, config)?;

        // Cross-section Jacobian not yet implemented (Phase 1a TODO)
        // For now, use empty Jacobian
        let cross_section_jacobian = vec![vec![0.0; 0]; energy.len()];

        // 2. Apply Beer-Lambert law with Jacobian
        let (number_density, thickness_cm) = if params.isotopes.is_empty() {
            (0.0, 0.0)
        } else {
            (
                params.isotopes[0].number_density,
                params.isotopes[0].thickness_cm,
            )
        };

        let (transmission, jac_beer) = compute_transmission_with_jacobian(
            &cross_sections,
            &cross_section_jacobian,
            number_density,
            thickness_cm,
        )?;

        // 3. Apply normalization with Jacobian
        let norm_config = NormalizationConfig {
            normalization: Parameter::fixed(config.normalization),
            background: config.background.clone(),
        };
        let (final_transmission, jac_norm) =
            apply_normalization_with_jacobian(&transmission, energy, &norm_config)?;

        // 4. Combine Jacobians via chain rule
        // dFinal/d(xs_params) = Norm * dT/d(xs_params)
        // Full Jacobian = [scaled_jac_beer | jac_norm]
        let jacobian = combine_jacobians(&jac_beer, &jac_norm, config.normalization);

        Ok((final_transmission, jacobian))
    }
}

/// Combine Beer-Lambert and normalization Jacobians via chain rule.
///
/// # Formula
///
/// ```text
/// dFinal/d(xs_param) = Norm * dT/d(xs_param)
/// Full Jacobian = [Norm * jac_beer | jac_norm]
/// ```
///
/// # Arguments
///
/// * `jac_beer` - Beer-Lambert Jacobian `dT/d(xs_params)` shape `[n_energy × n_xs_params]`
/// * `jac_norm` - Normalization Jacobian `dFinal/d(norm_params)` shape `[n_energy × n_norm_params]`
/// * `normalization` - Normalization factor (for chain rule scaling)
///
/// # Returns
///
/// Combined Jacobian shape `[n_energy × (n_xs_params + n_norm_params)]`
fn combine_jacobians(
    jac_beer: &[Vec<f64>],
    jac_norm: &[Vec<f64>],
    normalization: f64,
) -> Vec<Vec<f64>> {
    if jac_beer.is_empty() {
        return jac_norm.to_vec();
    }

    let n_energy = jac_beer.len();
    let n_xs_params = jac_beer[0].len();
    let n_norm_params = jac_norm[0].len();

    let mut result = vec![vec![0.0; n_xs_params + n_norm_params]; n_energy];

    for i in 0..n_energy {
        // Scale xs Jacobian by normalization: dFinal/dT = Norm
        for j in 0..n_xs_params {
            result[i][j] = normalization * jac_beer[i][j];
        }
        // Copy norm Jacobian directly
        for j in 0..n_norm_params {
            result[i][n_xs_params + j] = jac_norm[i][j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::{Background, Channel, IsotopeParams, Parameter, Resonance, SpinGroup};

    fn create_test_isotope() -> IsotopeParams {
        IsotopeParams {
            name: "Test-1".to_string(),
            awr: 1.0,
            abundance: Parameter::fixed(1.0),
            thickness_cm: 1.0,
            number_density: 0.05,
            spin_groups: vec![SpinGroup {
                j: 0.5,
                channels: vec![Channel {
                    l: 0,
                    channel_spin: 0.0,
                    radius: 5.0,
                    effective_radius: 5.0,
                }],
                resonances: vec![Resonance {
                    energy: Parameter::fixed(10.0),
                    gamma_n: Parameter::fixed(0.1),
                    gamma_g: Parameter::fixed(0.05),
                    fission: None,
                }],
            }],
        }
    }

    #[test]
    fn test_pipeline_empty() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![]).unwrap();
        let params = RMatrixParameters { isotopes: vec![] };
        let config = ForwardModelConfig::default();

        // Empty energy grid returns error from compute_0k_cross_sections
        let result = model.transmission(&energy, &params, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_basic() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![1.0, 5.0, 20.0]).unwrap();
        let params = RMatrixParameters {
            isotopes: vec![create_test_isotope()],
        };
        let config = ForwardModelConfig::default();

        let result = model.transmission(&energy, &params, &config).unwrap();

        // Should return transmission values (between 0 and 1)
        assert_eq!(result.len(), 3);
        for t in &result {
            assert!(*t >= 0.0 && *t <= 1.0, "Transmission out of range: {}", t);
        }
    }

    #[test]
    fn test_pipeline_with_normalization() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![1.0, 5.0, 20.0]).unwrap();
        let params = RMatrixParameters {
            isotopes: vec![create_test_isotope()],
        };
        let mut config = ForwardModelConfig::default();
        config.normalization = 0.9;

        let result = model.transmission(&energy, &params, &config).unwrap();

        // With normalization < 1.0, values should be scaled down
        assert_eq!(result.len(), 3);
        for t in &result {
            assert!(
                *t >= 0.0 && *t <= 1.0,
                "Normalized transmission out of range: {}",
                t
            );
        }
    }

    #[test]
    fn test_pipeline_with_background() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![1.0, 5.0, 20.0]).unwrap();
        let params = RMatrixParameters {
            isotopes: vec![create_test_isotope()],
        };
        let mut config = ForwardModelConfig::default();
        config.background = Background::Constant { value: 0.05 };

        let result = model.transmission(&energy, &params, &config).unwrap();

        // With background, values can exceed 1.0
        assert_eq!(result.len(), 3);
        for t in &result {
            assert!(*t >= 0.05, "Transmission less than background: {}", t);
        }
    }

    #[test]
    fn test_pipeline_with_jacobian_shape() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![1.0, 5.0, 20.0]).unwrap();
        let params = RMatrixParameters {
            isotopes: vec![create_test_isotope()],
        };
        let config = ForwardModelConfig::default();

        let (_t, jac) = model
            .transmission_with_jacobian(&energy, &params, &config)
            .unwrap();

        // Jacobian shape should be [3 energies × 0 params] (no free params)
        assert_eq!(jac.len(), 3);
        assert_eq!(jac[0].len(), 0);
    }

    #[test]
    fn test_combine_jacobians_empty() {
        let jac_beer: Vec<Vec<f64>> = vec![];
        let jac_norm = vec![vec![1.0]; 3];

        let result = combine_jacobians(&jac_beer, &jac_norm, 0.9);

        // Should return just the norm Jacobian
        assert_eq!(result, jac_norm);
    }

    #[test]
    fn test_combine_jacobians_both() {
        let jac_beer = vec![vec![1.0, 2.0]; 2]; // 2 energies × 2 xs params
        let jac_norm = vec![vec![0.5]; 2]; // 2 energies × 1 norm param

        let result = combine_jacobians(&jac_beer, &jac_norm, 0.9);

        // Should have shape [2 energies × 3 params]
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);

        // First 2 columns: scaled by normalization
        assert!((result[0][0] - 0.9 * 1.0).abs() < 1e-10);
        assert!((result[0][1] - 0.9 * 2.0).abs() < 1e-10);
        assert!((result[1][0] - 0.9 * 1.0).abs() < 1e-10);
        assert!((result[1][1] - 0.9 * 2.0).abs() < 1e-10);

        // Last column: copied directly from norm Jacobian
        assert!((result[0][2] - 0.5).abs() < 1e-10);
        assert!((result[1][2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_count_free_params() {
        let isotope = IsotopeParams {
            name: "Test-1".to_string(),
            awr: 1.0,
            abundance: Parameter::free(1.0),
            thickness_cm: 1.0,
            number_density: 0.05,
            spin_groups: vec![SpinGroup {
                j: 0.5,
                channels: vec![Channel {
                    l: 0,
                    channel_spin: 0.0,
                    radius: 5.0,
                    effective_radius: 5.0,
                }],
                resonances: vec![
                    Resonance {
                        energy: Parameter::free(10.0),
                        gamma_n: Parameter::fixed(0.1),
                        gamma_g: Parameter::free(0.05),
                        fission: None,
                    },
                    Resonance {
                        energy: Parameter::fixed(20.0),
                        gamma_n: Parameter::free(0.2),
                        gamma_g: Parameter::fixed(0.1),
                        fission: None,
                    },
                ],
            }],
        };

        let params = RMatrixParameters {
            isotopes: vec![isotope],
        };

        // 1 abundance + 1 energy + 1 gamma_g + 1 gamma_n = 4 free params
        assert_eq!(count_free_params(&params), 4);
    }
}
