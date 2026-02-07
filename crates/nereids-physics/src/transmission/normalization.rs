//! Normalization and background corrections for transmission spectra.
//!
//! # Physics
//!
//! After computing the Beer-Lambert transmission, experimental corrections are applied:
//!
//! ```text
//! Final(E) = Norm * T(E) + Background(E)
//!
//! where:
//!   Norm = normalization factor (multiplicative, typically ~1.0)
//!   T(E) = theoretical transmission from Beer-Lambert
//!   Background(E) = additive background correction
//! ```
//!
//! # Background Models
//!
//! See `nereids_core::Background` for available background models:
//! - Constant: `BackA`
//! - Inverse sqrt: `BackB / √E`
//! - Sqrt: `BackC * √E`
//! - Exponential: `BackD * exp(-BackF / √E)`
//!
//! # Jacobian Derivatives
//!
//! For fitting, we compute partial derivatives:
//!
//! ```text
//! dFinal/d(Norm) = T(E)
//! dFinal/d(BackA) = 1
//! dFinal/d(BackB) = 1 / √E
//! dFinal/d(BackC) = √E
//! dFinal/d(BackD) = exp(-BackF / √E)
//! dFinal/d(BackF) = -BackD * exp(-BackF / √E) / √E
//! ```
//!
//! # References
//!
//! SAMMY `mnrm1.f90` lines 40-69

use nereids_core::{Background, EnergyGrid, Parameter, PhysicsError};

/// Configuration for normalization and background corrections.
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Global normalization factor (multiplicative). Default 1.0.
    pub normalization: Parameter,
    /// Background model. Default: None.
    pub background: Background,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            normalization: Parameter::fixed(1.0),
            background: Background::None,
        }
    }
}

/// Apply normalization and background to transmission.
///
/// # Formula
///
/// `Final(E) = Norm * T(E) + Background(E)`
///
/// # Arguments
///
/// * `transmission` - Theoretical transmission from Beer-Lambert
/// * `energy` - Energy grid [eV]
/// * `config` - Normalization and background configuration
///
/// # Returns
///
/// Final transmission spectrum with normalization and background applied
///
/// # Errors
///
/// Returns `PhysicsError` if transmission and energy grid lengths don't match
///
/// # References
///
/// SAMMY `mnrm1.f90` `Norm` lines 40-45, 64-69
pub fn apply_normalization(
    transmission: &[f64],
    energy: &EnergyGrid,
    config: &NormalizationConfig,
) -> Result<Vec<f64>, PhysicsError> {
    // Validate inputs
    if transmission.len() != energy.len() {
        return Err(PhysicsError::DimensionMismatch {
            expected: energy.len(),
            got: transmission.len(),
        });
    }

    // Evaluate background at all energies
    let background = config.background.evaluate(energy);

    // Apply normalization and background
    let mut final_spectrum = Vec::with_capacity(transmission.len());
    for (i, &t) in transmission.iter().enumerate() {
        final_spectrum.push(config.normalization.value * t + background[i]);
    }

    Ok(final_spectrum)
}

/// Apply normalization and background with Jacobian.
///
/// # Returns
///
/// `(final_spectrum, jacobian)` where:
/// - `final_spectrum[i]` = Final(E_i)
/// - `jacobian[i][j]` = dFinal_i / dp_j
///
/// # Jacobian Structure
///
/// Columns correspond to free parameters (where `vary=true`):
/// - Normalization (if varied)
/// - Background parameters (depending on model, if varied)
///
/// Shape: `[n_energy × n_norm_params]`
///
/// # Arguments
///
/// * `transmission` - Theoretical transmission from Beer-Lambert
/// * `energy` - Energy grid [eV]
/// * `config` - Normalization and background configuration
///
/// # Jacobian Formulas
///
/// See module documentation for derivatives of each background model.
///
/// # References
///
/// SAMMY `mnrm1.f90` `Norm` lines 48-57
pub fn apply_normalization_with_jacobian(
    transmission: &[f64],
    energy: &EnergyGrid,
    config: &NormalizationConfig,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), PhysicsError> {
    // First compute final spectrum (validates inputs)
    let final_spectrum = apply_normalization(transmission, energy, config)?;

    // Handle empty case
    if final_spectrum.is_empty() {
        return Ok((final_spectrum, vec![]));
    }

    let n_energy = final_spectrum.len();

    // Count free parameters
    let mut n_free_params = 0;
    if config.normalization.vary {
        n_free_params += 1;
    }
    // Note: Background parameters not yet implemented as fittable
    // Will add when needed for fitting Phase 2

    // Initialize Jacobian matrix
    let mut jacobian = vec![vec![0.0; n_free_params]; n_energy];

    // Column index tracker
    let mut col_idx = 0;

    // Derivative with respect to normalization
    if config.normalization.vary {
        for (i, &t) in transmission.iter().enumerate() {
            // dFinal/d(Norm) = T(E)
            jacobian[i][col_idx] = t;
        }
        col_idx += 1;
    }

    // TODO: Add derivatives for background parameters when needed
    // For Phase 1b, only normalization is varied

    Ok((final_spectrum, jacobian))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_normalization_empty() {
        let t = vec![];
        let energy = EnergyGrid::new(vec![]).unwrap();
        let config = NormalizationConfig::default();

        let result = apply_normalization(&t, &energy, &config).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_apply_normalization_no_background() {
        // Norm=1.0, no background → output should equal input
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let config = NormalizationConfig::default();

        let result = apply_normalization(&t, &energy, &config).unwrap();
        assert_eq!(result, vec![0.5, 0.6, 0.7]);
    }

    #[test]
    fn test_apply_normalization_with_scaling() {
        // Norm=0.9, no background
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::fixed(0.9),
            background: Background::None,
        };

        let result = apply_normalization(&t, &energy, &config).unwrap();
        assert!((result[0] - 0.45).abs() < 1e-10);
        assert!((result[1] - 0.54).abs() < 1e-10);
        assert!((result[2] - 0.63).abs() < 1e-10);
    }

    #[test]
    fn test_apply_normalization_constant_background() {
        // Norm=1.0, constant background=0.1
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::fixed(1.0),
            background: Background::Constant { value: 0.1 },
        };

        let result = apply_normalization(&t, &energy, &config).unwrap();
        assert!((result[0] - 0.6).abs() < 1e-10);
        assert!((result[1] - 0.7).abs() < 1e-10);
        assert!((result[2] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_apply_normalization_inverse_sqrt_background() {
        // Norm=1.0, BackB/√E with coefficient=2.0
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 4.0, 9.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::fixed(1.0),
            background: Background::InverseSqrt { coefficient: 2.0 },
        };

        let result = apply_normalization(&t, &energy, &config).unwrap();
        // E=1.0: T=0.5, Back=2.0/1.0=2.0 → 0.5+2.0=2.5
        // E=4.0: T=0.6, Back=2.0/2.0=1.0 → 0.6+1.0=1.6
        // E=9.0: T=0.7, Back=2.0/3.0=0.667 → 0.7+0.667=1.367
        assert!((result[0] - 2.5).abs() < 1e-10);
        assert!((result[1] - 1.6).abs() < 1e-10);
        assert!((result[2] - (0.7 + 2.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_apply_normalization_sqrt_background() {
        // Norm=1.0, BackC*√E with coefficient=0.1
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 4.0, 9.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::fixed(1.0),
            background: Background::Sqrt { coefficient: 0.1 },
        };

        let result = apply_normalization(&t, &energy, &config).unwrap();
        // E=1.0: T=0.5, Back=0.1*1.0=0.1 → 0.6
        // E=4.0: T=0.6, Back=0.1*2.0=0.2 → 0.8
        // E=9.0: T=0.7, Back=0.1*3.0=0.3 → 1.0
        assert!((result[0] - 0.6).abs() < 1e-10);
        assert!((result[1] - 0.8).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_normalization_exponential_background() {
        // Norm=1.0, BackD*exp(-BackF/√E)
        let t = vec![0.5, 0.6];
        let energy = EnergyGrid::new(vec![1.0, 4.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::fixed(1.0),
            background: Background::Exponential {
                amplitude: 1.0,
                decay_factor: 2.0,
            },
        };

        let result = apply_normalization(&t, &energy, &config).unwrap();
        // E=1.0: T=0.5, Back=exp(-2/1)=exp(-2) → 0.5+exp(-2)
        // E=4.0: T=0.6, Back=exp(-2/2)=exp(-1) → 0.6+exp(-1)
        assert!((result[0] - (0.5 + (-2.0_f64).exp())).abs() < 1e-10);
        assert!((result[1] - (0.6 + (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_apply_normalization_combined() {
        // Norm=0.9, constant background=0.05
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::fixed(0.9),
            background: Background::Constant { value: 0.05 },
        };

        let result = apply_normalization(&t, &energy, &config).unwrap();
        // Final = 0.9 * T + 0.05
        assert!((result[0] - (0.9 * 0.5 + 0.05)).abs() < 1e-10);
        assert!((result[1] - (0.9 * 0.6 + 0.05)).abs() < 1e-10);
        assert!((result[2] - (0.9 * 0.7 + 0.05)).abs() < 1e-10);
    }

    #[test]
    fn test_apply_normalization_dimension_mismatch() {
        let t = vec![0.5, 0.6];
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let config = NormalizationConfig::default();

        let result = apply_normalization(&t, &energy, &config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("dimension mismatch"));
    }

    #[test]
    fn test_apply_normalization_with_jacobian_empty() {
        let t: Vec<f64> = vec![];
        let energy = EnergyGrid::new(vec![]).unwrap();
        let config = NormalizationConfig::default();

        let (final_spec, jac) = apply_normalization_with_jacobian(&t, &energy, &config).unwrap();
        assert_eq!(final_spec, Vec::<f64>::new());
        assert_eq!(jac, Vec::<Vec<f64>>::new());
    }

    #[test]
    fn test_apply_normalization_with_jacobian_fixed_norm() {
        // Fixed normalization → no free parameters → empty Jacobian
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::fixed(1.0),
            background: Background::None,
        };

        let (_final_spec, jac) = apply_normalization_with_jacobian(&t, &energy, &config).unwrap();

        // No free parameters
        assert_eq!(jac.len(), 3); // 3 energies
        assert_eq!(jac[0].len(), 0); // 0 parameters
    }

    #[test]
    fn test_apply_normalization_with_jacobian_vary_norm() {
        // Vary normalization → 1 free parameter
        let t = vec![0.5, 0.6, 0.7];
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let config = NormalizationConfig {
            normalization: Parameter::free(0.9),
            background: Background::None,
        };

        let (_final_spec, jac) = apply_normalization_with_jacobian(&t, &energy, &config).unwrap();

        // 1 free parameter (normalization)
        assert_eq!(jac.len(), 3); // 3 energies
        assert_eq!(jac[0].len(), 1); // 1 parameter

        // dFinal/d(Norm) = T
        assert!((jac[0][0] - 0.5).abs() < 1e-10);
        assert!((jac[1][0] - 0.6).abs() < 1e-10);
        assert!((jac[2][0] - 0.7).abs() < 1e-10);
    }
}
