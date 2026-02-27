//! Core domain types for neutron resonance imaging.

use crate::error::NereidsError;

/// Identifies an isotope by atomic number Z and mass number A.
///
/// Fields are private to enforce validation invariants (A > 0, Z <= A).
/// Use [`Isotope::new`] to construct and the getter methods to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Isotope {
    /// Atomic number (number of protons).
    z: u32,
    /// Mass number (protons + neutrons).
    a: u32,
}

impl Isotope {
    /// Create a new isotope with validation.
    ///
    /// # Errors
    /// Returns `NereidsError::InvalidParameter` if:
    /// - `a` is zero (mass number must be positive)
    /// - `z > a` (atomic number cannot exceed mass number)
    pub fn new(z: u32, a: u32) -> Result<Self, NereidsError> {
        if a == 0 {
            return Err(NereidsError::InvalidParameter(
                "mass number A must be positive".to_string(),
            ));
        }
        if z > a {
            return Err(NereidsError::InvalidParameter(format!(
                "atomic number Z ({}) cannot exceed mass number A ({})",
                z, a
            )));
        }
        Ok(Self { z, a })
    }

    /// Atomic number (number of protons).
    pub fn z(&self) -> u32 {
        self.z
    }

    /// Mass number (protons + neutrons).
    pub fn a(&self) -> u32 {
        self.a
    }
}

impl std::fmt::Display for Isotope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Z={}, A={}", self.z, self.a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isotope_rejects_z_greater_than_a() {
        // Z > A is a physical impossibility (more protons than nucleons).
        let err = Isotope::new(3, 2).unwrap_err();
        assert!(err.to_string().contains("cannot exceed"));
    }

    #[test]
    fn test_isotope_rejects_zero_mass_number() {
        let err = Isotope::new(0, 0).unwrap_err();
        assert!(err.to_string().contains("must be positive"));
    }

    #[test]
    fn test_isotope_valid() {
        let iso = Isotope::new(92, 238).unwrap();
        assert_eq!(iso.z(), 92);
        assert_eq!(iso.a(), 238);
    }

    #[test]
    fn test_isotope_neutron() {
        // Z=0, A=1 is a neutron — valid.
        let neutron = Isotope::new(0, 1).unwrap();
        assert_eq!(neutron.z(), 0);
        assert_eq!(neutron.a(), 1);
    }
}

/// A sample layer with known isotopic composition.
#[derive(Debug, Clone)]
pub struct SampleLayer {
    /// Isotopes present and their areal densities (atoms/barn).
    pub components: Vec<SampleComponent>,
    /// Sample temperature in Kelvin.
    pub temperature_k: f64,
}

/// A single isotopic component within a sample layer.
#[derive(Debug, Clone)]
pub struct SampleComponent {
    /// The isotope.
    pub isotope: Isotope,
    /// Areal density in atoms/barn (= number_density × thickness).
    pub areal_density: f64,
}
