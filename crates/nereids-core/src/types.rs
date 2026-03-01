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

    // --- SampleComponent validation tests ---

    #[test]
    fn test_sample_component_valid() {
        let iso = Isotope::new(92, 238).unwrap();
        let comp = SampleComponent::new(iso, 0.001).unwrap();
        assert_eq!(comp.isotope().z(), 92);
        assert!((comp.areal_density() - 0.001).abs() < 1e-15);
    }

    #[test]
    fn test_sample_component_zero_density() {
        let iso = Isotope::new(92, 238).unwrap();
        let comp = SampleComponent::new(iso, 0.0).unwrap();
        assert!((comp.areal_density()).abs() < 1e-15);
    }

    #[test]
    fn test_sample_component_rejects_negative_density() {
        let iso = Isotope::new(92, 238).unwrap();
        let err = SampleComponent::new(iso, -0.001).unwrap_err();
        assert_eq!(err, SampleError::InvalidArealDensity(-0.001));
    }

    #[test]
    fn test_sample_component_rejects_nan_density() {
        let iso = Isotope::new(92, 238).unwrap();
        let err = SampleComponent::new(iso, f64::NAN).unwrap_err();
        assert!(matches!(err, SampleError::InvalidArealDensity(_)));
    }

    #[test]
    fn test_sample_component_rejects_infinite_density() {
        let iso = Isotope::new(92, 238).unwrap();
        let err = SampleComponent::new(iso, f64::INFINITY).unwrap_err();
        assert!(matches!(err, SampleError::InvalidArealDensity(_)));
    }

    // --- SampleLayer validation tests ---

    #[test]
    fn test_sample_layer_valid() {
        let iso = Isotope::new(92, 238).unwrap();
        let comp = SampleComponent::new(iso, 0.001).unwrap();
        let layer = SampleLayer::new(vec![comp], 300.0).unwrap();
        assert_eq!(layer.components().len(), 1);
        assert!((layer.temperature_k() - 300.0).abs() < 1e-15);
    }

    #[test]
    fn test_sample_layer_rejects_empty_components() {
        let err = SampleLayer::new(vec![], 300.0).unwrap_err();
        assert_eq!(err, SampleError::EmptyComponents);
    }

    #[test]
    fn test_sample_layer_rejects_nan_temperature() {
        let iso = Isotope::new(92, 238).unwrap();
        let comp = SampleComponent::new(iso, 0.001).unwrap();
        let err = SampleLayer::new(vec![comp], f64::NAN).unwrap_err();
        assert!(matches!(err, SampleError::NonFiniteTemperature(_)));
    }

    #[test]
    fn test_sample_layer_rejects_infinite_temperature() {
        let iso = Isotope::new(92, 238).unwrap();
        let comp = SampleComponent::new(iso, 0.001).unwrap();
        let err = SampleLayer::new(vec![comp], f64::INFINITY).unwrap_err();
        assert!(matches!(err, SampleError::NonFiniteTemperature(_)));
    }

    #[test]
    fn test_sample_layer_rejects_negative_temperature() {
        let iso = Isotope::new(92, 238).unwrap();
        let comp = SampleComponent::new(iso, 0.001).unwrap();
        let err = SampleLayer::new(vec![comp], -1.0).unwrap_err();
        assert_eq!(err, SampleError::NegativeTemperature(-1.0));
    }

    #[test]
    fn test_sample_layer_zero_temperature_allowed() {
        let iso = Isotope::new(92, 238).unwrap();
        let comp = SampleComponent::new(iso, 0.001).unwrap();
        let layer = SampleLayer::new(vec![comp], 0.0).unwrap();
        assert!((layer.temperature_k()).abs() < 1e-15);
    }
}

/// Errors from `SampleComponent` and `SampleLayer` construction.
#[derive(Debug, PartialEq)]
pub enum SampleError {
    /// Areal density must be non-negative and finite.
    InvalidArealDensity(f64),
    /// Temperature must be finite.
    NonFiniteTemperature(f64),
    /// Temperature must be non-negative.
    NegativeTemperature(f64),
    /// A sample layer must contain at least one component.
    EmptyComponents,
}

impl std::fmt::Display for SampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArealDensity(v) => {
                write!(f, "areal density must be non-negative and finite, got {v}")
            }
            Self::NonFiniteTemperature(v) => {
                write!(f, "temperature must be finite, got {v}")
            }
            Self::NegativeTemperature(v) => {
                write!(f, "temperature must be non-negative, got {v}")
            }
            Self::EmptyComponents => write!(f, "sample layer must have at least one component"),
        }
    }
}

impl std::error::Error for SampleError {}

/// A single isotopic component within a sample layer.
#[derive(Debug, Clone)]
pub struct SampleComponent {
    /// The isotope.
    isotope: Isotope,
    /// Areal density in atoms/barn (= number_density × thickness).
    areal_density: f64,
}

impl SampleComponent {
    /// Create a validated sample component.
    ///
    /// # Errors
    /// Returns `SampleError::InvalidArealDensity` if `areal_density` is negative,
    /// NaN, or infinite.
    pub fn new(isotope: Isotope, areal_density: f64) -> Result<Self, SampleError> {
        if !areal_density.is_finite() || areal_density < 0.0 {
            return Err(SampleError::InvalidArealDensity(areal_density));
        }
        Ok(Self {
            isotope,
            areal_density,
        })
    }

    /// Returns the isotope.
    #[must_use]
    pub fn isotope(&self) -> &Isotope {
        &self.isotope
    }

    /// Returns the areal density in atoms/barn.
    #[must_use]
    pub fn areal_density(&self) -> f64 {
        self.areal_density
    }
}

/// A sample layer with known isotopic composition.
#[derive(Debug, Clone)]
pub struct SampleLayer {
    /// Isotopes present and their areal densities (atoms/barn).
    components: Vec<SampleComponent>,
    /// Sample temperature in Kelvin.
    temperature_k: f64,
}

impl SampleLayer {
    /// Create a validated sample layer.
    ///
    /// # Errors
    /// Returns `SampleError::EmptyComponents` if `components` is empty.
    /// Returns `SampleError::NonFiniteTemperature` if `temperature_k` is NaN or infinity.
    /// Returns `SampleError::NegativeTemperature` if `temperature_k < 0.0`.
    pub fn new(components: Vec<SampleComponent>, temperature_k: f64) -> Result<Self, SampleError> {
        if components.is_empty() {
            return Err(SampleError::EmptyComponents);
        }
        if !temperature_k.is_finite() {
            return Err(SampleError::NonFiniteTemperature(temperature_k));
        }
        if temperature_k < 0.0 {
            return Err(SampleError::NegativeTemperature(temperature_k));
        }
        Ok(Self {
            components,
            temperature_k,
        })
    }

    /// Returns the isotopic components in this layer.
    #[must_use]
    pub fn components(&self) -> &[SampleComponent] {
        &self.components
    }

    /// Returns the sample temperature in Kelvin.
    #[must_use]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }
}
