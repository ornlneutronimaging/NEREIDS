//! Core domain types for neutron resonance imaging.

/// Identifies an isotope by atomic number Z and mass number A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Isotope {
    /// Atomic number (number of protons).
    pub z: u32,
    /// Mass number (protons + neutrons).
    pub a: u32,
}

impl Isotope {
    pub fn new(z: u32, a: u32) -> Self {
        debug_assert!(a > 0, "mass number A must be positive");
        debug_assert!(
            z <= a,
            "atomic number Z ({}) cannot exceed mass number A ({})",
            z,
            a
        );
        Self { z, a }
    }
}

impl std::fmt::Display for Isotope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Z={}, A={}", self.z, self.a)
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
