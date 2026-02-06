//! Nuclear data types for R-matrix parameterization.

/// Flags indicating which resonance parameters are allowed to vary during fitting.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default)]
pub struct VaryFlags {
    pub energy: bool,
    pub gamma_n: bool,
    pub gamma_g: bool,
    pub gamma_f1: bool,
    pub gamma_f2: bool,
}

/// A single nuclear resonance with its parameters.
#[derive(Debug, Clone, Default)]
pub struct Resonance {
    /// Resonance energy in eV.
    pub energy: f64,
    /// Neutron width (`Γ_n`) in eV.
    pub gamma_n: f64,
    /// Capture width (`Γ_γ`) in eV.
    pub gamma_g: f64,
    /// First fission width (`Γ_f1`) in eV. Zero for non-fissile isotopes.
    pub gamma_f1: f64,
    /// Second fission width (`Γ_f2`) in eV. Zero for non-fissile isotopes.
    pub gamma_f2: f64,
    /// Which parameters may vary during fitting.
    pub vary: VaryFlags,
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
