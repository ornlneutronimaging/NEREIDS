//! Nuclear data types for R-matrix parameterization.

/// A single fit parameter: a value paired with whether it varies during optimization.
#[derive(Debug, Clone, Copy)]
pub struct Parameter {
    pub value: f64,
    pub vary: bool,
}

impl Parameter {
    /// Create a fixed parameter (will not vary during fitting).
    pub fn fixed(value: f64) -> Self {
        Self { value, vary: false }
    }

    /// Create a free parameter (will vary during fitting).
    pub fn free(value: f64) -> Self {
        Self { value, vary: true }
    }
}

impl Default for Parameter {
    fn default() -> Self {
        Self::fixed(0.0)
    }
}

/// Fission widths for fissile isotopes.
#[derive(Debug, Clone, Copy)]
pub struct FissionWidths {
    /// First fission width (`Γ_f1`) in eV.
    pub gamma_f1: Parameter,
    /// Second fission width (`Γ_f2`) in eV.
    pub gamma_f2: Parameter,
}

/// A single nuclear resonance with its parameters.
#[derive(Debug, Clone)]
pub struct Resonance {
    /// Resonance energy in eV.
    pub energy: Parameter,
    /// Neutron width (`Γ_n`) in eV.
    pub gamma_n: Parameter,
    /// Capture width (`Γ_γ`) in eV.
    pub gamma_g: Parameter,
    /// Fission widths. `None` for non-fissile isotopes.
    pub fission: Option<FissionWidths>,
}

/// A reaction channel defined by angular momentum and spin quantum numbers.
#[derive(Debug, Clone)]
pub struct Channel {
    /// Orbital angular momentum quantum number.
    pub l: u32,
    /// Channel spin.
    pub channel_spin: f64,
    /// Channel radius in fm.
    pub radius: f64,
    /// Effective radius for potential scattering in fm.
    pub effective_radius: f64,
}

/// A spin group: a set of resonances sharing the same total angular momentum J.
#[derive(Debug, Clone)]
pub struct SpinGroup {
    /// Total angular momentum J.
    pub j: f64,
    /// Reaction channels for this spin group.
    pub channels: Vec<Channel>,
    /// Resonances belonging to this spin group.
    pub resonances: Vec<Resonance>,
}

/// Parameters for a single isotope.
#[derive(Debug, Clone)]
pub struct IsotopeParams {
    /// Isotope name (e.g., "Fe-56", "U-235").
    pub name: String,
    /// Atomic weight ratio (mass relative to neutron).
    pub awr: f64,
    /// Abundance (fraction, 0.0 to 1.0). This is a primary fit parameter in Mode 1.
    pub abundance: f64,
    /// Sample thickness in cm.
    pub thickness_cm: f64,
    /// Number density in atoms/barn-cm.
    pub number_density: f64,
    /// Spin groups with their resonances and channels.
    pub spin_groups: Vec<SpinGroup>,
}

/// Collection of isotope parameters defining the full R-matrix problem.
#[derive(Debug, Clone, Default)]
pub struct RMatrixParameters {
    /// All isotopes in the sample.
    pub isotopes: Vec<IsotopeParams>,
}
