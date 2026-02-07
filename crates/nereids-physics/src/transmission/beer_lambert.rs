//! Beer-Lambert transmission law for neutron resonance imaging.
//!
//! # Physics
//!
//! The Beer-Lambert law describes how neutron transmission decreases with sample
//! thickness and cross section:
//!
//! ```text
//! T(E) = exp(-optical_depth)
//!
//! where optical_depth = n * d * σ_total(E)
//!
//! Units:
//!   n = number density [atoms/barn-cm]
//!   d = thickness [cm]
//!   σ_total(E) = total cross section [barns] (abundance-weighted for mixtures)
//!
//! Unit check:
//!   [atoms/barn-cm] × [cm] × [barns] = dimensionless ✓
//! ```
//!
//! # Jacobian Derivatives
//!
//! For gradient-based fitting, we compute partial derivatives:
//!
//! ```text
//! dT/dσ = -n * d * T(E)
//! dT/d(thickness) = -n * σ(E) * T(E)
//! dT/d(number_density) = -d * σ(E) * T(E)
//! ```
//!
//! # References
//!
//! SAMMY `mnrm2.f90` `Transm_sum` lines 19-39

use crate::rmatrix::reich_moore::CrossSections;
use nereids_core::PhysicsError;

/// Maximum optical depth to avoid numerical underflow in exp().
///
/// exp(-50) ≈ 1.9e-22 is effectively zero for transmission.
const MAX_OPTICAL_DEPTH: f64 = 50.0;

/// Compute Beer-Lambert transmission for a sample.
///
/// # Formula
///
/// `T(E) = exp(-n * d * σ_total(E))`
///
/// # Arguments
///
/// * `cross_sections` - Pre-computed 0K cross sections (one per energy point)
/// * `number_density` - Sample number density [atoms/barn-cm]
/// * `thickness_cm` - Sample thickness [cm]
///
/// # Returns
///
/// Transmission spectrum (dimensionless, range 0.0 to 1.0)
///
/// # Errors
///
/// Returns `PhysicsError` if any cross section is negative (unphysical)
///
/// # References
///
/// SAMMY `mnrm2.f90` `Transm_sum` lines 19-22
pub fn compute_transmission(
    cross_sections: &[CrossSections],
    number_density: f64,
    thickness_cm: f64,
) -> Result<Vec<f64>, PhysicsError> {
    let mut transmission = Vec::with_capacity(cross_sections.len());

    for xs in cross_sections {
        // Check for unphysical cross section
        if xs.total < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "Negative total cross section: {}",
                xs.total
            )));
        }

        // Compute optical depth: n * d * σ(E)
        let mut optical_depth: f64 = number_density * thickness_cm * xs.total;

        // Clamp optical depth to avoid underflow
        optical_depth = optical_depth.min(MAX_OPTICAL_DEPTH);

        // Apply Beer-Lambert law
        transmission.push((-optical_depth).exp());
    }

    Ok(transmission)
}

/// Compute Beer-Lambert transmission with Jacobian.
///
/// # Returns
///
/// `(transmission, jacobian)` where:
/// - `transmission[i]` = T(E_i)
/// - `jacobian[i][j]` = dT_i / dp_j
///
/// # Jacobian Structure
///
/// Columns correspond to free parameters (where cross-section derivatives are provided):
/// - Resonance parameters (energies, widths) from cross-section Jacobian
///
/// Shape: `[n_energy × n_free_params]`
///
/// # Arguments
///
/// * `cross_sections` - Pre-computed 0K cross sections
/// * `cross_section_jacobian` - Jacobian dσ/dp from Phase 1a (shape `[n_energy × n_xs_params]`)
/// * `number_density` - Sample number density [atoms/barn-cm]
/// * `thickness_cm` - Sample thickness [cm]
///
/// # Jacobian Formulas
///
/// Chain rule from cross sections:
/// ```text
/// dT/d(xs_param_j) = -n * d * T(E) * dσ/d(xs_param_j)
/// ```
///
/// # References
///
/// SAMMY `mnrm2.f90` `Transm_sum` lines 25-39
pub fn compute_transmission_with_jacobian(
    cross_sections: &[CrossSections],
    cross_section_jacobian: &[Vec<f64>],
    number_density: f64,
    thickness_cm: f64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), PhysicsError> {
    // First compute transmission (validates inputs)
    let transmission = compute_transmission(cross_sections, number_density, thickness_cm)?;

    // Handle empty case
    if transmission.is_empty() {
        return Ok((transmission, vec![]));
    }

    let n_energy = transmission.len();

    // Validate Jacobian shape: one row per energy point and consistent width.
    if cross_section_jacobian.len() != n_energy {
        return Err(PhysicsError::InvalidParameter(format!(
            "cross_section_jacobian row count ({}) must match number of energy points ({n_energy})",
            cross_section_jacobian.len()
        )));
    }
    let n_free_params = cross_section_jacobian.first().map_or(0, Vec::len);
    for (i, row) in cross_section_jacobian.iter().enumerate() {
        if row.len() != n_free_params {
            return Err(PhysicsError::InvalidParameter(format!(
                "cross_section_jacobian row {i} has length {}, expected {n_free_params}",
                row.len()
            )));
        }
    }

    // Initialize Jacobian matrix
    let mut jacobian = vec![vec![0.0; n_free_params]; n_energy];

    // Compute derivatives with respect to cross-section parameters
    for i_energy in 0..n_energy {
        for j_param in 0..n_free_params {
            let d_sigma_dp = cross_section_jacobian[i_energy][j_param];

            // Chain rule: dT/d(param) = -n * d * T * (dσ/d(param))
            jacobian[i_energy][j_param] =
                -number_density * thickness_cm * transmission[i_energy] * d_sigma_dp;
        }
    }

    Ok((transmission, jacobian))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create test cross sections with constant values
    fn const_cross_sections(n_energy: usize, sigma_total: f64) -> Vec<CrossSections> {
        vec![
            CrossSections {
                elastic: sigma_total,
                capture: 0.0,
                fission: 0.0,
                total: sigma_total,
            };
            n_energy
        ]
    }

    #[test]
    fn test_compute_transmission_empty() {
        let xs = vec![];
        let result = compute_transmission(&xs, 0.05, 1.0).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_compute_transmission_simple() {
        // n=0.05 atoms/barn-cm, d=1.0 cm, σ=10 barns
        // Optical depth = 0.05 * 1.0 * 10 = 0.5
        // T = exp(-0.5) ≈ 0.6065
        let xs = const_cross_sections(3, 10.0);
        let result = compute_transmission(&xs, 0.05, 1.0).unwrap();

        assert_eq!(result.len(), 3);
        let expected = (-0.5_f64).exp();
        for t in &result {
            assert!((t - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_transmission_large_optical_depth_clamped() {
        // Large cross section to trigger clamping
        // n=1.0, d=10.0, σ=100.0 → optical_depth = 1000 > 50 (clamped)
        // T = exp(-50) ≈ 1.9e-22
        let xs = const_cross_sections(1, 100.0);
        let result = compute_transmission(&xs, 1.0, 10.0).unwrap();

        let expected = (-MAX_OPTICAL_DEPTH).exp();
        assert!((result[0] - expected).abs() < 1e-30);
    }

    #[test]
    fn test_compute_transmission_zero_cross_section() {
        // Zero cross section → perfect transmission (T=1.0)
        let xs = const_cross_sections(2, 0.0);
        let result = compute_transmission(&xs, 0.05, 1.0).unwrap();
        assert_eq!(result, vec![1.0, 1.0]);
    }

    #[test]
    fn test_compute_transmission_negative_cross_section() {
        let xs = vec![CrossSections {
            elastic: -10.0,
            capture: 0.0,
            fission: 0.0,
            total: -10.0,
        }];

        let result = compute_transmission(&xs, 0.05, 1.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Negative total cross section"));
    }

    #[test]
    fn test_compute_transmission_with_jacobian_empty() {
        let xs: Vec<CrossSections> = vec![];
        let jac_xs: Vec<Vec<f64>> = vec![];

        let (t, jac) = compute_transmission_with_jacobian(&xs, &jac_xs, 0.05, 1.0).unwrap();
        assert_eq!(t, Vec::<f64>::new());
        assert_eq!(jac, Vec::<Vec<f64>>::new());
    }

    #[test]
    fn test_compute_transmission_with_jacobian_shape() {
        // 3 energy points, 2 xs parameters
        let xs = const_cross_sections(3, 10.0);
        let jac_xs = vec![vec![1.0, 2.0]; 3]; // 3 energies × 2 params

        let (_t, jac) = compute_transmission_with_jacobian(&xs, &jac_xs, 0.05, 1.0).unwrap();

        // Should have shape [3 energies × 2 params]
        assert_eq!(jac.len(), 3);
        assert_eq!(jac[0].len(), 2);
    }

    #[test]
    fn test_compute_transmission_with_jacobian_values() {
        // Test Jacobian values with known derivatives
        // n=0.1, d=1.0, σ=10 → optical_depth=1.0, T=exp(-1)≈0.3679
        // dT/dσ = -n * d * T = -0.1 * 1.0 * 0.3679 = -0.03679
        // If dσ/dp = 1.0, then dT/dp = -0.03679 * 1.0 = -0.03679
        let xs = const_cross_sections(1, 10.0);
        let jac_xs = vec![vec![1.0]]; // dσ/dp = 1.0

        let (t, jac) = compute_transmission_with_jacobian(&xs, &jac_xs, 0.1, 1.0).unwrap();

        let expected_t = (-1.0_f64).exp();
        assert!((t[0] - expected_t).abs() < 1e-10);

        // dT/dp = -n * d * T * (dσ/dp) = -0.1 * 1.0 * exp(-1) * 1.0
        let expected_deriv = -0.1 * 1.0 * expected_t * 1.0;
        assert!((jac[0][0] - expected_deriv).abs() < 1e-10);
    }

    #[test]
    fn test_compute_transmission_with_jacobian_row_count_mismatch_errors() {
        let xs = const_cross_sections(2, 10.0);
        let jac_xs = vec![vec![1.0]]; // 1 row, but 2 energies

        let result = compute_transmission_with_jacobian(&xs, &jac_xs, 0.1, 1.0);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("row count"));
    }

    #[test]
    fn test_compute_transmission_with_jacobian_ragged_rows_error() {
        let xs = const_cross_sections(2, 10.0);
        let jac_xs = vec![vec![1.0, 2.0], vec![1.0]]; // ragged

        let result = compute_transmission_with_jacobian(&xs, &jac_xs, 0.1, 1.0);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("row 1"));
    }

    #[test]
    fn test_compute_transmission_varying_cross_sections() {
        // Test with energy-dependent cross sections
        let xs = vec![
            CrossSections {
                elastic: 5.0,
                capture: 0.0,
                fission: 0.0,
                total: 5.0,
            },
            CrossSections {
                elastic: 10.0,
                capture: 0.0,
                fission: 0.0,
                total: 10.0,
            },
            CrossSections {
                elastic: 20.0,
                capture: 0.0,
                fission: 0.0,
                total: 20.0,
            },
        ];

        let result = compute_transmission(&xs, 0.1, 1.0).unwrap();
        assert_eq!(result.len(), 3);

        // n=0.1, d=1.0
        // E1: σ=5  → optical_depth=0.5  → T=exp(-0.5)
        // E2: σ=10 → optical_depth=1.0  → T=exp(-1.0)
        // E3: σ=20 → optical_depth=2.0  → T=exp(-2.0)
        assert!((result[0] - (-0.5_f64).exp()).abs() < 1e-10);
        assert!((result[1] - (-1.0_f64).exp()).abs() < 1e-10);
        assert!((result[2] - (-2.0_f64).exp()).abs() < 1e-10);
    }
}
