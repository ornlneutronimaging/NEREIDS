//! Core domain types for neutron resonance imaging.

use crate::error::NereidsError;

/// Identifies an isotope by atomic number Z and mass number A.
///
/// Fields are private to enforce validation invariants (A > 0, Z <= A).
/// Use [`Isotope::new`] to construct and the getter methods to read.
///
/// Deserialization goes through [`Isotope::new`] so that invalid JSON
/// (e.g. `{"z": 100, "a": 5}`) is rejected at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Isotope {
    /// Atomic number (number of protons).
    z: u32,
    /// Mass number (protons + neutrons).
    a: u32,
}

/// Private helper for serde deserialization — accepts any z/a pair,
/// then validates via [`Isotope::new`].
#[derive(serde::Deserialize)]
struct IsotopeRaw {
    z: u32,
    a: u32,
}

impl<'de> serde::Deserialize<'de> for Isotope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = IsotopeRaw::deserialize(deserializer)?;
        Isotope::new(raw.z, raw.a).map_err(serde::de::Error::custom)
    }
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

    #[test]
    fn test_deserialize_valid_isotope() {
        let json = r#"{"z": 92, "a": 238}"#;
        let iso: Isotope = serde_json::from_str(json).unwrap();
        assert_eq!(iso.z(), 92);
        assert_eq!(iso.a(), 238);
    }

    #[test]
    fn test_deserialize_invalid_isotope_z_greater_than_a() {
        // Z > A must be rejected even when deserialized from JSON.
        let json = r#"{"z": 100, "a": 5}"#;
        let result: Result<Isotope, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot exceed"),
            "expected 'cannot exceed' in error: {err_msg}"
        );
    }

    #[test]
    fn test_deserialize_invalid_isotope_zero_mass() {
        // A == 0 must be rejected even when deserialized from JSON.
        let json = r#"{"z": 0, "a": 0}"#;
        let result: Result<Isotope, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("must be positive"),
            "expected 'must be positive' in error: {err_msg}"
        );
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
