//! Core domain types for neutron resonance imaging.

use crate::elements;
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

    // --- IsotopeGroup tests ---

    #[test]
    fn test_natural_group_w() {
        let w = IsotopeGroup::natural(74).unwrap();
        assert_eq!(w.name(), "W (nat)");
        assert_eq!(w.n_members(), 5); // W-180, W-182, W-183, W-184, W-186
        let sum: f64 = w.members().iter().map(|(_, r)| r).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_natural_group_au_monoisotopic() {
        let au = IsotopeGroup::natural(79).unwrap();
        assert_eq!(au.n_members(), 1);
        assert_eq!(au.members()[0].0.a(), 197);
        assert!((au.members()[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_natural_group_unknown_z() {
        let err = IsotopeGroup::natural(0).unwrap_err();
        assert!(err.to_string().contains("no natural isotopes"));
    }

    #[test]
    fn test_subset_group_eu() {
        let eu = IsotopeGroup::subset(63, &[151, 153]).unwrap();
        assert_eq!(eu.name(), "Eu-151/153");
        assert_eq!(eu.n_members(), 2);
        let sum: f64 = eu.members().iter().map(|(_, r)| r).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_subset_group_invalid_a() {
        let err = IsotopeGroup::subset(63, &[999]).unwrap_err();
        assert!(err.to_string().contains("not a natural isotope"));
    }

    #[test]
    fn test_subset_group_empty() {
        let err = IsotopeGroup::subset(63, &[]).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_custom_group() {
        let u235 = Isotope::new(92, 235).unwrap();
        let u238 = Isotope::new(92, 238).unwrap();
        let g =
            IsotopeGroup::custom("U (enriched)".into(), vec![(u235, 0.95), (u238, 0.05)]).unwrap();
        assert_eq!(g.name(), "U (enriched)");
        assert_eq!(g.n_members(), 2);
    }

    #[test]
    fn test_custom_group_auto_normalizes() {
        let u235 = Isotope::new(92, 235).unwrap();
        let u238 = Isotope::new(92, 238).unwrap();
        // Sum = 1.0005, within tolerance → auto-normalized
        let g = IsotopeGroup::custom("U".into(), vec![(u235, 0.9505), (u238, 0.05)]).unwrap();
        let sum: f64 = g.members().iter().map(|(_, r)| r).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_group_rejects_bad_sum() {
        let u235 = Isotope::new(92, 235).unwrap();
        let err = IsotopeGroup::custom("bad".into(), vec![(u235, 0.5)]).unwrap_err();
        assert!(err.to_string().contains("sum to 1.0"));
    }

    #[test]
    fn test_custom_group_rejects_negative_ratio() {
        let u235 = Isotope::new(92, 235).unwrap();
        let u238 = Isotope::new(92, 238).unwrap();
        let err = IsotopeGroup::custom("bad".into(), vec![(u235, -0.5), (u238, 1.5)]).unwrap_err();
        assert!(err.to_string().contains("positive and finite"));
    }

    #[test]
    fn test_custom_group_rejects_empty() {
        let err = IsotopeGroup::custom("empty".into(), vec![]).unwrap_err();
        assert!(err.to_string().contains("at least one member"));
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

// ---------------------------------------------------------------------------
// Isotope groups
// ---------------------------------------------------------------------------

/// Maximum allowed deviation of ratio sum from 1.0 before rejecting.
const RATIO_SUM_TOLERANCE: f64 = 1e-3;
/// Threshold below which the sum is considered exact (no renormalization).
const RATIO_EXACT_TOLERANCE: f64 = 1e-6;

/// A group of isotopes sharing one fitted density parameter.
///
/// Members have fixed fractional ratios summing to 1.0. During fitting,
/// the effective cross-section `σ_eff(E) = Σ fᵢ · σᵢ(E)` reduces the
/// group to a single "virtual isotope" with one free density parameter.
///
/// # Examples
/// ```
/// use nereids_core::types::IsotopeGroup;
///
/// // All natural tungsten isotopes at IUPAC abundances
/// let w = IsotopeGroup::natural(74).unwrap();
/// assert_eq!(w.name(), "W (nat)");
/// assert_eq!(w.n_members(), 5); // W-180, W-182, W-183, W-184, W-186
///
/// // Subset: only Eu-151 and Eu-153
/// let eu = IsotopeGroup::subset(63, &[151, 153]).unwrap();
/// assert_eq!(eu.n_members(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct IsotopeGroup {
    name: String,
    members: Vec<(Isotope, f64)>,
}

impl IsotopeGroup {
    /// Create a group from all natural isotopes of element Z at IUPAC abundances.
    ///
    /// # Errors
    /// - Unknown element Z (no natural isotopes in database)
    pub fn natural(z: u32) -> Result<Self, NereidsError> {
        let isotopes = elements::natural_isotopes(z);
        if isotopes.is_empty() {
            return Err(NereidsError::InvalidParameter(format!(
                "no natural isotopes for Z={z}"
            )));
        }
        let sym = elements::element_symbol(z).unwrap_or("?");
        Ok(Self {
            name: format!("{sym} (nat)"),
            members: isotopes,
        })
    }

    /// Create a group from a subset of natural isotopes, re-normalized.
    ///
    /// # Errors
    /// - Empty `mass_numbers`
    /// - Any mass number not found among natural isotopes of Z
    /// - Unknown element Z
    pub fn subset(z: u32, mass_numbers: &[u32]) -> Result<Self, NereidsError> {
        if mass_numbers.is_empty() {
            return Err(NereidsError::InvalidParameter(
                "mass_numbers must not be empty".into(),
            ));
        }
        let all_natural = elements::natural_isotopes(z);
        if all_natural.is_empty() {
            return Err(NereidsError::InvalidParameter(format!(
                "no natural isotopes for Z={z}"
            )));
        }
        let mut selected: Vec<(Isotope, f64)> = Vec::with_capacity(mass_numbers.len());
        for &a in mass_numbers {
            let found = all_natural.iter().find(|(iso, _)| iso.a() == a);
            match found {
                Some(&pair) => selected.push(pair),
                None => {
                    return Err(NereidsError::InvalidParameter(format!(
                        "A={a} is not a natural isotope of Z={z}"
                    )));
                }
            }
        }
        // Re-normalize ratios to sum to 1.0
        let sum: f64 = selected.iter().map(|(_, r)| r).sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err(NereidsError::InvalidParameter(
                "selected isotope abundances sum to zero or non-finite".into(),
            ));
        }
        for entry in &mut selected {
            entry.1 /= sum;
        }
        let sym = elements::element_symbol(z).unwrap_or("?");
        let a_labels: Vec<String> = mass_numbers.iter().map(|a| a.to_string()).collect();
        let name = format!("{sym}-{}", a_labels.join("/"));
        Ok(Self {
            name,
            members: selected,
        })
    }

    /// Create a group with arbitrary isotope/ratio pairs.
    ///
    /// Ratios must be positive, finite, and sum to 1.0 (within tolerance).
    /// If the sum is within 0.1% of 1.0 but not exact, ratios are auto-normalized.
    ///
    /// # Errors
    /// - Empty members
    /// - Non-positive or non-finite ratio
    /// - Ratio sum deviates from 1.0 by more than 0.1%
    pub fn custom(name: String, members: Vec<(Isotope, f64)>) -> Result<Self, NereidsError> {
        if members.is_empty() {
            return Err(NereidsError::InvalidParameter(
                "isotope group must have at least one member".into(),
            ));
        }
        for (iso, ratio) in &members {
            if !ratio.is_finite() || *ratio <= 0.0 {
                return Err(NereidsError::InvalidParameter(format!(
                    "ratio for {} must be positive and finite, got {ratio}",
                    elements::isotope_to_string(iso),
                )));
            }
        }
        let sum: f64 = members.iter().map(|(_, r)| r).sum();
        if (sum - 1.0).abs() > RATIO_SUM_TOLERANCE {
            return Err(NereidsError::InvalidParameter(format!(
                "ratios must sum to 1.0 (got {sum})"
            )));
        }
        let mut members = members;
        if (sum - 1.0).abs() > RATIO_EXACT_TOLERANCE {
            for entry in &mut members {
                entry.1 /= sum;
            }
        }
        Ok(Self { name, members })
    }

    /// Group display name (e.g., "W (nat)", "Eu-151/153").
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Member isotopes with their fractional ratios.
    #[must_use]
    pub fn members(&self) -> &[(Isotope, f64)] {
        &self.members
    }

    /// Number of member isotopes.
    #[must_use]
    pub fn n_members(&self) -> usize {
        self.members.len()
    }
}
