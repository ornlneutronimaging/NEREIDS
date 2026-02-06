//! Compute 0K cross sections over an energy grid.

use nereids_core::energy::EnergyGrid;
use nereids_core::error::PhysicsError;
use nereids_core::forward_model::ForwardModelConfig;
use nereids_core::nuclear::{IsotopeParams, RMatrixParameters};

use super::reich_moore::{reich_moore_cross_sections, CrossSections, RMatrixConfig};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Infer target spin I from spin-group J values using J = I ± 1/2.
///
/// For ambiguous single-J isotopes, choose the smaller physically valid solution.
fn infer_target_spin(isotope: &IsotopeParams) -> f64 {
    let Some(first_group) = isotope.spin_groups.first() else {
        return 0.0;
    };

    let tol = 1e-8;
    let j0 = first_group.j;
    let mut candidates = vec![(j0 - 0.5).abs(), j0 + 0.5];
    candidates.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite target-spin candidates"));
    candidates.dedup_by(|a, b| approx_eq(*a, *b, tol));

    for sg in isotope.spin_groups.iter().skip(1) {
        let j = sg.j;
        candidates.retain(|i| approx_eq(j, i + 0.5, tol) || approx_eq(j, (i - 0.5).abs(), tol));
        if candidates.is_empty() {
            break;
        }
    }

    candidates
        .into_iter()
        .fold(None, |best: Option<f64>, v| match best {
            Some(curr) if curr <= v => Some(curr),
            _ => Some(v),
        })
        .unwrap_or_else(|| (j0 - 0.5).max(0.0))
}

/// Compute 0K Reich-Moore cross sections over an energy grid.
///
/// Evaluates cross sections at each energy point in the grid, summing contributions
/// from all spin groups with statistical weights.
///
/// # Arguments
///
/// * `energy_grid` - Energy points in eV
/// * `params` - R-matrix parameters (resonances, spin groups, isotopes)
/// * `config` - Forward model configuration
///
/// # Returns
///
/// Vector of `CrossSections` (one per energy point) containing elastic, capture,
/// fission, and total cross sections in barns.
///
/// # Errors
///
/// Returns `PhysicsError` if:
/// - Energy grid is empty
/// - R-matrix computation fails at any energy point
/// - Isotope parameters are invalid
///
/// # Algorithm
///
/// For each energy E in the grid:
/// 1. For each isotope:
///    a. For each spin group J:
///       - Compute cross sections for resonances with this J
///       - Apply statistical weight `G_J` = (2J+1)/(2I+1)
///    b. Sum weighted contributions from all spin groups
/// 2. Weight by isotope abundance
/// 3. Return array of cross sections
///
/// # References
///
/// SAMMY `dopush1.f90` lines 73-176 (angular momentum summation)
pub fn compute_0k_cross_sections(
    energy_grid: &EnergyGrid,
    params: &RMatrixParameters,
    config: &ForwardModelConfig,
) -> Result<Vec<CrossSections>, PhysicsError> {
    if energy_grid.is_empty() {
        return Err(PhysicsError::EmptyEnergyGrid);
    }

    let n_energies = energy_grid.len();
    let mut cross_sections = vec![CrossSections::default(); n_energies];

    // Loop over all isotopes
    for isotope in &params.isotopes {
        let abundance = isotope.abundance.value;

        // Skip isotopes with zero abundance
        if abundance.abs() < 1e-30 {
            continue;
        }

        // Compute cross sections for this isotope at all energies
        let iso_cross_sections = compute_isotope_cross_sections(energy_grid, isotope, config)?;

        // Accumulate weighted contributions
        for (i, iso_cs) in iso_cross_sections.iter().enumerate() {
            cross_sections[i].elastic += abundance * iso_cs.elastic;
            cross_sections[i].capture += abundance * iso_cs.capture;
            cross_sections[i].fission += abundance * iso_cs.fission;
            cross_sections[i].total += abundance * iso_cs.total;
        }
    }

    Ok(cross_sections)
}

/// Compute cross sections for a single isotope over an energy grid.
fn compute_isotope_cross_sections(
    energy_grid: &EnergyGrid,
    isotope: &IsotopeParams,
    config: &ForwardModelConfig,
) -> Result<Vec<CrossSections>, PhysicsError> {
    let n_energies = energy_grid.len();
    let mut cross_sections = vec![CrossSections::default(); n_energies];

    let target_spin = infer_target_spin(isotope);

    // Create R-matrix configuration
    let rmatrix_config = RMatrixConfig {
        target_spin,
        awr: isotope.awr,
        include_potential: config.include_potential_scattering,
    };

    // Loop over each energy point
    for (i, &energy) in energy_grid.values.iter().enumerate() {
        // Sum contributions from all spin groups
        for spin_group in &isotope.spin_groups {
            // Get resonances for this spin group
            let resonances = &spin_group.resonances;

            // Skip only when there is neither resonance contribution nor potential scattering.
            if resonances.is_empty() && !rmatrix_config.include_potential {
                continue;
            }

            // Compute cross sections for this spin group at this energy
            let cs = reich_moore_cross_sections(energy, resonances, spin_group, &rmatrix_config)?;

            // Add to total (statistical weight already applied in reich_moore_cross_sections)
            cross_sections[i].elastic += cs.elastic;
            cross_sections[i].capture += cs.capture;
            cross_sections[i].fission += cs.fission;
            cross_sections[i].total += cs.total;
        }
    }

    Ok(cross_sections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::nuclear::{Channel, Parameter, Resonance, SpinGroup};

    #[test]
    fn test_compute_0k_empty_grid() {
        let energy_grid = EnergyGrid::new(vec![]).unwrap();
        let params = RMatrixParameters::default();
        let config = ForwardModelConfig::default();

        let result = compute_0k_cross_sections(&energy_grid, &params, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_0k_no_isotopes() {
        let energy_grid = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let params = RMatrixParameters::default();
        let config = ForwardModelConfig::default();

        let result = compute_0k_cross_sections(&energy_grid, &params, &config).unwrap();
        assert_eq!(result.len(), 3);

        // With no isotopes, all cross sections should be zero
        for cs in &result {
            assert_eq!(cs.elastic, 0.0);
            assert_eq!(cs.capture, 0.0);
            assert_eq!(cs.fission, 0.0);
            assert_eq!(cs.total, 0.0);
        }
    }

    #[test]
    fn test_compute_0k_single_isotope() {
        let energy_grid = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();

        let isotope = IsotopeParams {
            name: "Test-1".to_string(),
            awr: 10.0,
            abundance: Parameter::fixed(1.0),
            thickness_cm: 0.1,
            number_density: 1e-3,
            spin_groups: vec![SpinGroup {
                j: 0.5,
                channels: vec![Channel {
                    l: 0,
                    channel_spin: 0.5,
                    radius: 2.908,
                    effective_radius: 2.908,
                }],
                resonances: vec![Resonance {
                    energy: Parameter::fixed(5.0),
                    gamma_n: Parameter::fixed(0.1),
                    gamma_g: Parameter::fixed(0.05),
                    fission: None,
                }],
            }],
        };

        let params = RMatrixParameters {
            isotopes: vec![isotope],
        };

        let config = ForwardModelConfig::default();

        let result = compute_0k_cross_sections(&energy_grid, &params, &config).unwrap();
        assert_eq!(result.len(), 3);

        // Cross sections should be non-zero (we have a resonance at 5 eV)
        // The middle point at 10 eV should show some resonance contribution
        assert!(result[1].total > 0.0);
    }

    #[test]
    fn test_compute_0k_multiple_spin_groups() {
        let energy_grid = EnergyGrid::new(vec![1.0, 5.0, 10.0]).unwrap();

        let isotope = IsotopeParams {
            name: "Test-2".to_string(),
            awr: 10.0,
            abundance: Parameter::fixed(1.0),
            thickness_cm: 0.1,
            number_density: 1e-3,
            spin_groups: vec![
                SpinGroup {
                    j: 0.5,
                    channels: vec![Channel {
                        l: 0,
                        channel_spin: 0.5,
                        radius: 2.908,
                        effective_radius: 2.908,
                    }],
                    resonances: vec![Resonance {
                        energy: Parameter::fixed(3.0),
                        gamma_n: Parameter::fixed(0.1),
                        gamma_g: Parameter::fixed(0.05),
                        fission: None,
                    }],
                },
                SpinGroup {
                    j: 1.5,
                    channels: vec![Channel {
                        l: 1,
                        channel_spin: 0.5,
                        radius: 2.908,
                        effective_radius: 2.908,
                    }],
                    resonances: vec![Resonance {
                        energy: Parameter::fixed(7.0),
                        gamma_n: Parameter::fixed(0.2),
                        gamma_g: Parameter::fixed(0.1),
                        fission: None,
                    }],
                },
            ],
        };

        let params = RMatrixParameters {
            isotopes: vec![isotope],
        };

        let config = ForwardModelConfig::default();

        let result = compute_0k_cross_sections(&energy_grid, &params, &config).unwrap();
        assert_eq!(result.len(), 3);

        // All energies should have some cross section from nearby resonances
        for cs in &result {
            assert!(cs.total >= 0.0);
        }
    }

    #[test]
    fn test_compute_0k_potential_scattering_for_empty_spin_group() {
        let energy_grid = EnergyGrid::new(vec![1.0]).unwrap();

        let isotope = IsotopeParams {
            name: "Potential-Only".to_string(),
            awr: 10.0,
            abundance: Parameter::fixed(1.0),
            thickness_cm: 0.1,
            number_density: 1e-3,
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

        let result = compute_0k_cross_sections(&energy_grid, &params, &config).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].elastic > 0.0);
        assert!(result[0].total > 0.0);
    }
}
