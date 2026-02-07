//! Background model types for transmission fitting.
//!
//! SAMMY supports multiple background models that add energy-dependent corrections
//! to the theoretical transmission spectrum. These account for detector response,
//! scattered neutrons, and other experimental effects.
//!
//! # Background Models
//!
//! - **None**: No background (zero contribution)
//! - **Constant**: Energy-independent offset `BackA`
//! - **`InverseSqrt`**: Inverse square-root term `BackB / √E`
//! - **Sqrt**: Square-root term `BackC * √E`
//! - **Exponential**: Exponential decay `BackD * exp(-BackF / √E)`
//!
//! # References
//!
//! SAMMY `mnrm1.f90` lines 64-68 (background evaluation)

use crate::energy::EnergyGrid;
use crate::error::PhysicsError;

/// Background model for transmission normalization.
///
/// The background is added to the normalized transmission:
/// `Final(E) = Norm * T(E) + Background(E)`
///
/// Each variant corresponds to a term in SAMMY's background model.
///
/// # Design Note
///
/// Uses enum instead of trait to enable `Clone` (required by `ForwardModelConfig`)
/// and to allow exhaustive pattern matching in derivative calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum Background {
    /// No background contribution.
    None,

    /// Constant background (energy-independent).
    ///
    /// Formula: `BackA`
    Constant {
        /// Background amplitude (dimensionless).
        value: f64,
    },

    /// Inverse square-root background.
    ///
    /// Formula: `BackB / √E`
    ///
    /// Common for detector efficiency corrections.
    InverseSqrt {
        /// Coefficient `BackB` [dimensionless or eV^(1/2)].
        coefficient: f64,
    },

    /// Square-root background.
    ///
    /// Formula: `BackC * √E`
    ///
    /// Models energy-dependent scattering.
    Sqrt {
        /// Coefficient `BackC` [dimensionless or eV^(-1/2)].
        coefficient: f64,
    },

    /// Exponential decay background.
    ///
    /// Formula: `BackD * exp(-BackF / √E)`
    ///
    /// Models fast-neutron contamination or detector dead-time effects.
    Exponential {
        /// Amplitude `BackD` (dimensionless).
        amplitude: f64,
        /// Decay factor `BackF` [eV^(1/2)].
        decay_factor: f64,
    },
}

impl Background {
    /// Evaluate the background at each energy point.
    ///
    /// # Arguments
    ///
    /// * `energy` - Energy grid [eV]
    ///
    /// # Returns
    ///
    /// Background values (same length as energy grid, dimensionless)
    ///
    /// # Errors
    ///
    /// Returns `PhysicsError::InvalidParameter` if any energy value is non-positive
    /// for background models that require `sqrt(E)`.
    ///
    /// # References
    ///
    /// SAMMY `mnrm1.f90` lines 64-68
    pub fn evaluate(&self, energy: &EnergyGrid) -> Result<Vec<f64>, PhysicsError> {
        match self {
            Background::None => Ok(vec![0.0; energy.len()]),

            Background::Constant { value } => Ok(vec![*value; energy.len()]),

            Background::InverseSqrt { coefficient } => {
                let mut out = Vec::with_capacity(energy.len());
                for (i, &e) in energy.values.iter().enumerate() {
                    if e <= 0.0 {
                        return Err(PhysicsError::InvalidParameter(format!(
                            "energy must be positive for InverseSqrt background at index {i}, got {e}"
                        )));
                    }
                    out.push(coefficient / e.sqrt());
                }
                Ok(out)
            }

            Background::Sqrt { coefficient } => {
                let mut out = Vec::with_capacity(energy.len());
                for (i, &e) in energy.values.iter().enumerate() {
                    if e <= 0.0 {
                        return Err(PhysicsError::InvalidParameter(format!(
                            "energy must be positive for Sqrt background at index {i}, got {e}"
                        )));
                    }
                    out.push(coefficient * e.sqrt());
                }
                Ok(out)
            }

            Background::Exponential {
                amplitude,
                decay_factor,
            } => {
                let mut out = Vec::with_capacity(energy.len());
                for (i, &e) in energy.values.iter().enumerate() {
                    if e <= 0.0 {
                        return Err(PhysicsError::InvalidParameter(format!(
                            "energy must be positive for Exponential background at index {i}, got {e}"
                        )));
                    }
                    out.push(amplitude * (-decay_factor / e.sqrt()).exp());
                }
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_none() {
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let bg = Background::None;
        let result = bg.evaluate(&energy).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_background_constant() {
        let energy = EnergyGrid::new(vec![1.0, 10.0, 100.0]).unwrap();
        let bg = Background::Constant { value: 0.5 };
        let result = bg.evaluate(&energy).unwrap();
        assert_eq!(result, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_background_inverse_sqrt() {
        let energy = EnergyGrid::new(vec![1.0, 4.0, 9.0]).unwrap();
        let bg = Background::InverseSqrt { coefficient: 2.0 };
        let result = bg.evaluate(&energy).unwrap();
        assert_eq!(result, vec![2.0, 1.0, 2.0 / 3.0]);
    }

    #[test]
    fn test_background_sqrt() {
        let energy = EnergyGrid::new(vec![1.0, 4.0, 9.0]).unwrap();
        let bg = Background::Sqrt { coefficient: 2.0 };
        let result = bg.evaluate(&energy).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_background_exponential() {
        let energy = EnergyGrid::new(vec![1.0, 4.0]).unwrap();
        let bg = Background::Exponential {
            amplitude: 1.0,
            decay_factor: 2.0,
        };
        let result = bg.evaluate(&energy).unwrap();
        // exp(-2/1) ≈ 0.1353, exp(-2/2) ≈ 0.3679
        assert!((result[0] - (-2.0_f64).exp()).abs() < 1e-10);
        assert!((result[1] - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_background_clone() {
        let bg1 = Background::Exponential {
            amplitude: 1.5,
            decay_factor: 3.0,
        };
        let bg2 = bg1.clone();
        assert_eq!(bg1, bg2);
    }

    #[test]
    fn test_background_negative_energy() {
        let energy = EnergyGrid::new(vec![1.0, -1.0, 100.0]).unwrap();
        let bg = Background::InverseSqrt { coefficient: 1.0 };
        let result = bg.evaluate(&energy);
        assert!(result.is_err());
    }
}
