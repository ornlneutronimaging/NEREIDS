//! Default forward model composing all pipeline stages.
//!
//! Pipeline order (Phase 1b):
//! 1. 0K cross sections (R-matrix)
//! 2. Beer-Lambert transmission
//! 3. Normalization + background
//!
//! Beer-Lambert is evaluated as:
//! `T(E) = exp(-Σ_i [n_i * d_i * σ_i(E)])`
//! where each isotope contributes its own areal density (`n_i*d_i`) and
//! abundance-weighted isotope cross section `σ_i(E)`.
//!
//! Future phases will add:
//! - Doppler broadening (Phase 1c)
//! - Self-shielding (optional)
//! - Resolution convolution

use crate::rmatrix::cross_section::compute_0k_cross_sections;
use crate::transmission::normalization::{
    apply_normalization, apply_normalization_with_jacobian, NormalizationConfig,
};

use nereids_core::{
    EnergyGrid, ForwardModel, ForwardModelConfig, Parameter, PhysicsError, RMatrixParameters,
    ResolutionFunction,
};

#[derive(Debug, Clone, Copy)]
enum FreeParamPath {
    Abundance {
        isotope: usize,
    },
    ResonanceEnergy {
        isotope: usize,
        spin_group: usize,
        resonance: usize,
    },
    GammaN {
        isotope: usize,
        spin_group: usize,
        resonance: usize,
    },
    GammaG {
        isotope: usize,
        spin_group: usize,
        resonance: usize,
    },
    GammaF1 {
        isotope: usize,
        spin_group: usize,
        resonance: usize,
    },
    GammaF2 {
        isotope: usize,
        spin_group: usize,
        resonance: usize,
    },
}

fn collect_free_param_paths(params: &RMatrixParameters) -> Vec<FreeParamPath> {
    let mut paths = Vec::new();
    for (iso_idx, isotope) in params.isotopes.iter().enumerate() {
        if isotope.abundance.vary {
            paths.push(FreeParamPath::Abundance { isotope: iso_idx });
        }
    }

    for (iso_idx, isotope) in params.isotopes.iter().enumerate() {
        for (sg_idx, sg) in isotope.spin_groups.iter().enumerate() {
            for (res_idx, res) in sg.resonances.iter().enumerate() {
                if res.energy.vary {
                    paths.push(FreeParamPath::ResonanceEnergy {
                        isotope: iso_idx,
                        spin_group: sg_idx,
                        resonance: res_idx,
                    });
                }
                if res.gamma_n.vary {
                    paths.push(FreeParamPath::GammaN {
                        isotope: iso_idx,
                        spin_group: sg_idx,
                        resonance: res_idx,
                    });
                }
                if res.gamma_g.vary {
                    paths.push(FreeParamPath::GammaG {
                        isotope: iso_idx,
                        spin_group: sg_idx,
                        resonance: res_idx,
                    });
                }
                if let Some(ref f) = res.fission {
                    if f.gamma_f1.vary {
                        paths.push(FreeParamPath::GammaF1 {
                            isotope: iso_idx,
                            spin_group: sg_idx,
                            resonance: res_idx,
                        });
                    }
                    if f.gamma_f2.vary {
                        paths.push(FreeParamPath::GammaF2 {
                            isotope: iso_idx,
                            spin_group: sg_idx,
                            resonance: res_idx,
                        });
                    }
                }
            }
        }
    }
    paths
}

/// Count the total number of free (vary=true) parameters in the R-matrix problem.
fn count_free_params(params: &RMatrixParameters) -> usize {
    collect_free_param_paths(params).len()
}

fn get_param_value(params: &RMatrixParameters, path: FreeParamPath) -> f64 {
    match path {
        FreeParamPath::Abundance { isotope } => params.isotopes[isotope].abundance.value,
        FreeParamPath::ResonanceEnergy {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .energy
                .value
        }
        FreeParamPath::GammaN {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .gamma_n
                .value
        }
        FreeParamPath::GammaG {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .gamma_g
                .value
        }
        FreeParamPath::GammaF1 {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .fission
                .as_ref()
                .expect("path only collected when present")
                .gamma_f1
                .value
        }
        FreeParamPath::GammaF2 {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .fission
                .as_ref()
                .expect("path only collected when present")
                .gamma_f2
                .value
        }
    }
}

fn set_param_value(params: &mut RMatrixParameters, path: FreeParamPath, value: f64) {
    match path {
        FreeParamPath::Abundance { isotope } => params.isotopes[isotope].abundance.value = value,
        FreeParamPath::ResonanceEnergy {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .energy
                .value = value
        }
        FreeParamPath::GammaN {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .gamma_n
                .value = value
        }
        FreeParamPath::GammaG {
            isotope,
            spin_group,
            resonance,
        } => {
            params.isotopes[isotope].spin_groups[spin_group].resonances[resonance]
                .gamma_g
                .value = value
        }
        FreeParamPath::GammaF1 {
            isotope,
            spin_group,
            resonance,
        } => {
            if let Some(ref mut f) =
                params.isotopes[isotope].spin_groups[spin_group].resonances[resonance].fission
            {
                f.gamma_f1.value = value;
            }
        }
        FreeParamPath::GammaF2 {
            isotope,
            spin_group,
            resonance,
        } => {
            if let Some(ref mut f) =
                params.isotopes[isotope].spin_groups[spin_group].resonances[resonance].fission
            {
                f.gamma_f2.value = value;
            }
        }
    }
}

fn finite_difference_step(value: f64) -> f64 {
    1e-6 * (value.abs() + 1.0)
}

fn compute_beer_lambert_transmission_multi_isotope(
    energy: &EnergyGrid,
    params: &RMatrixParameters,
    config: &ForwardModelConfig,
) -> Result<Vec<f64>, PhysicsError> {
    // Keep empty-grid behavior consistent across all code paths.
    if energy.is_empty() {
        return Err(PhysicsError::EmptyEnergyGrid);
    }

    // Preserve existing behavior for empty-isotope problems: no attenuation.
    if params.isotopes.is_empty() {
        return Ok(vec![1.0; energy.len()]);
    }

    let mut optical_depth = vec![0.0_f64; energy.len()];
    for isotope in &params.isotopes {
        if !isotope.number_density.is_finite() || !isotope.thickness_cm.is_finite() {
            return Err(PhysicsError::InvalidParameter(format!(
                "isotope '{}' has non-finite geometry (number_density={}, thickness_cm={})",
                isotope.name, isotope.number_density, isotope.thickness_cm
            )));
        }
        if isotope.number_density < 0.0 || isotope.thickness_cm < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "isotope '{}' has negative geometry (number_density={}, thickness_cm={})",
                isotope.name, isotope.number_density, isotope.thickness_cm
            )));
        }

        let areal_density = isotope.number_density * isotope.thickness_cm;
        if areal_density == 0.0 {
            continue;
        }

        // Compute this isotope's abundance-weighted cross section contribution.
        let single_params = RMatrixParameters {
            isotopes: vec![isotope.clone()],
        };
        let isotope_cs = compute_0k_cross_sections(energy, &single_params, config)?;
        for (i, cs) in isotope_cs.iter().enumerate() {
            if cs.total < 0.0 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "negative isotope total cross section for '{}' at energy index {}: {}",
                    isotope.name, i, cs.total
                )));
            }
            optical_depth[i] += areal_density * cs.total;
        }
    }

    Ok(optical_depth.into_iter().map(|tau| (-tau).exp()).collect())
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
        // 1. Beer-Lambert transmission with per-isotope areal density.
        let transmission = compute_beer_lambert_transmission_multi_isotope(energy, params, config)?;

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
        // 1. Base Beer-Lambert transmission.
        let transmission = compute_beer_lambert_transmission_multi_isotope(energy, params, config)?;

        // 2. Finite-difference Jacobian for free R-matrix parameters.
        // This keeps derivatives physically coupled to the current transmission path
        // until analytic cross-section Jacobians are introduced.
        let n_free_params = count_free_params(params);
        let free_paths = collect_free_param_paths(params);
        debug_assert_eq!(free_paths.len(), n_free_params);
        let mut jac_beer = vec![vec![0.0; n_free_params]; energy.len()];
        for (j, path) in free_paths.iter().copied().enumerate() {
            let base_value = get_param_value(params, path);
            let h = finite_difference_step(base_value);

            let mut params_plus = params.clone();
            set_param_value(&mut params_plus, path, base_value + h);
            let t_plus =
                compute_beer_lambert_transmission_multi_isotope(energy, &params_plus, config);

            let mut params_minus = params.clone();
            set_param_value(&mut params_minus, path, base_value - h);
            let t_minus =
                compute_beer_lambert_transmission_multi_isotope(energy, &params_minus, config);

            match (t_plus, t_minus) {
                (Ok(tp), Ok(tm)) => {
                    for i in 0..energy.len() {
                        jac_beer[i][j] = (tp[i] - tm[i]) / (2.0 * h);
                    }
                }
                (Ok(tp), Err(_)) => {
                    for i in 0..energy.len() {
                        jac_beer[i][j] = (tp[i] - transmission[i]) / h;
                    }
                }
                (Err(_), Ok(tm)) => {
                    for i in 0..energy.len() {
                        jac_beer[i][j] = (transmission[i] - tm[i]) / h;
                    }
                }
                (Err(e_plus), Err(e_minus)) => {
                    return Err(PhysicsError::InvalidParameter(format!(
                        "failed finite-difference Jacobian for parameter column {j} (base value {base_value}): plus error={e_plus}, minus error={e_minus}"
                    )));
                }
            }
        }

        // 3. Apply normalization with Jacobian.
        let norm_config = NormalizationConfig {
            normalization: Parameter::fixed(config.normalization),
            background: config.background.clone(),
        };
        let (final_transmission, jac_norm) =
            apply_normalization_with_jacobian(&transmission, energy, &norm_config)?;

        // 4. Combine Jacobians via chain rule.
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

        // Empty energy grid should always be rejected.
        let result = model.transmission(&energy, &params, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_empty_grid_with_zero_areal_isotopes_errors() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![]).unwrap();
        let mut isotope = create_test_isotope();
        isotope.number_density = 0.0;
        isotope.thickness_cm = 0.0;
        let params = RMatrixParameters {
            isotopes: vec![isotope],
        };
        let config = ForwardModelConfig::default();

        let t_result = model.transmission(&energy, &params, &config);
        assert!(matches!(t_result, Err(PhysicsError::EmptyEnergyGrid)));

        let j_result = model.transmission_with_jacobian(&energy, &params, &config);
        assert!(matches!(j_result, Err(PhysicsError::EmptyEnergyGrid)));
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
    fn test_pipeline_with_jacobian_preserves_free_param_width() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![1.0, 5.0, 20.0]).unwrap();
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
                resonances: vec![Resonance {
                    energy: Parameter::free(10.0),
                    gamma_n: Parameter::free(0.1),
                    gamma_g: Parameter::fixed(0.05),
                    fission: None,
                }],
            }],
        };
        let params = RMatrixParameters {
            isotopes: vec![isotope],
        };
        let config = ForwardModelConfig::default();

        let (_t, jac) = model
            .transmission_with_jacobian(&energy, &params, &config)
            .unwrap();

        // 3 free params: abundance + resonance energy + gamma_n
        assert_eq!(jac.len(), energy.len());
        assert_eq!(jac[0].len(), 3);
    }

    #[test]
    fn test_pipeline_with_jacobian_nonzero_for_free_parameters() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![1.0e6, 1.1e6]).unwrap();
        let isotope = IsotopeParams {
            name: "Test-1".to_string(),
            awr: 10.0,
            abundance: Parameter::free(1.0),
            thickness_cm: 1.0,
            number_density: 0.1,
            spin_groups: vec![SpinGroup {
                j: 0.5,
                channels: vec![Channel {
                    l: 0,
                    channel_spin: 0.5,
                    radius: 2.908,
                    effective_radius: 2.908,
                }],
                resonances: vec![],
            }],
        };
        let params = RMatrixParameters {
            isotopes: vec![isotope],
        };
        let config = ForwardModelConfig {
            include_potential_scattering: true,
            ..ForwardModelConfig::default()
        };

        let (_t, jac) = model
            .transmission_with_jacobian(&energy, &params, &config)
            .unwrap();

        assert_eq!(jac.len(), energy.len());
        assert_eq!(jac[0].len(), 1); // free abundance only
        assert!(
            jac.iter().any(|row| row[0] != 0.0),
            "expected nonzero derivative for free abundance parameter"
        );
    }

    #[test]
    fn test_pipeline_multi_isotope_areal_density_sum() {
        let model = DefaultForwardModel { resolution: None };
        let energy = EnergyGrid::new(vec![5.0]).unwrap();

        let mut iso1 = create_test_isotope();
        iso1.name = "Iso-1".to_string();
        iso1.number_density = 0.05;
        iso1.thickness_cm = 1.0;

        let mut iso2 = create_test_isotope();
        iso2.name = "Iso-2".to_string();
        iso2.number_density = 0.30;
        iso2.thickness_cm = 2.0;
        iso2.abundance = Parameter::fixed(0.5);

        let params = RMatrixParameters {
            isotopes: vec![iso1.clone(), iso2.clone()],
        };
        let config = ForwardModelConfig::default();

        let transmission = model.transmission(&energy, &params, &config).unwrap();

        // Expected Beer-Lambert optical depth:
        // τ(E) = Σ_i (n_i d_i σ_i(E)), where σ_i is abundance-weighted isotope XS.
        let mut expected_tau = 0.0;
        for isotope in [iso1, iso2] {
            let single = RMatrixParameters {
                isotopes: vec![isotope.clone()],
            };
            let xs = compute_0k_cross_sections(&energy, &single, &config).unwrap();
            expected_tau += isotope.number_density * isotope.thickness_cm * xs[0].total;
        }
        let expected_t = (-expected_tau).exp();
        assert!((transmission[0] - expected_t).abs() < 1e-12);
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
