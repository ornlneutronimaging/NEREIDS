//! Resonance parameter data structures.
//!
//! These types represent parsed ENDF-6 File 2 resonance data, organized
//! following the structure in SAMMY's `SammyRMatrixParameters.h`.
//!
//! ## SAMMY Reference
//! - `sammy/external/openScale/repo/packages/ScaleUtils/EndfLib/RMatResonanceParam.h`
//! - `sammy/src/endf/SammyRMatrixParameters.h`

use nereids_core::types::Isotope;
use serde::{Deserialize, Serialize};

/// Resonance formalism flag (ENDF LRF values).
///
/// Reference: ENDF-6 Formats Manual, File 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResonanceFormalism {
    /// Single-Level Breit-Wigner (LRF=1 with SLBW treatment, or SAMMY LRF=-1).
    SLBW,
    /// Multi-Level Breit-Wigner (LRF=2).
    MLBW,
    /// Reich-Moore (LRF=3). Primary formalism for light and actinide isotopes.
    ReichMoore,
    /// R-Matrix Limited (LRF=7). General multi-channel formalism; used for
    /// many medium-heavy isotopes (W, Ta, Zr, etc.) in ENDF/B-VIII.0.
    RMatrixLimited,
}

/// Top-level container for all resonance data parsed from an ENDF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceData {
    /// The isotope this data belongs to.
    pub isotope: Isotope,
    /// ZA identifier (Z*1000 + A).
    pub za: u32,
    /// Atomic weight ratio (mass of target / neutron mass).
    pub awr: f64,
    /// Energy ranges containing resonance parameters.
    pub ranges: Vec<ResonanceRange>,
}

/// A single energy range within the resolved resonance region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceRange {
    /// Lower energy bound (eV).
    pub energy_low: f64,
    /// Upper energy bound (eV).
    pub energy_high: f64,
    /// Resolved (true) or unresolved (false).
    pub resolved: bool,
    /// Resonance formalism used in this range.
    pub formalism: ResonanceFormalism,
    /// Target spin (I).
    pub target_spin: f64,
    /// Scattering radius (fm).
    pub scattering_radius: f64,
    /// Spin groups for LRF=1/2/3 (L-grouped). Empty for LRF=7.
    pub l_groups: Vec<LGroup>,
    /// R-Matrix Limited data for LRF=7. `None` for LRF=1/2/3.
    pub rml: Option<Box<RmlData>>,
}

/// Parameters grouped by orbital angular momentum L.
///
/// In ENDF File 2 (LRF=3, Reich-Moore), resonances are grouped by L-value.
/// Each L-group contains resonances with different J values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LGroup {
    /// Orbital angular momentum quantum number.
    pub l: u32,
    /// Atomic weight ratio for this group.
    pub awr: f64,
    /// Channel scattering radius for this L (fm). 0.0 means use the global value.
    pub apl: f64,
    /// Individual resonances in this L-group.
    pub resonances: Vec<Resonance>,
}

/// A single resonance entry.
///
/// The meaning of the width fields depends on the formalism:
///
/// ## Reich-Moore (LRF=3)
/// - `gn`: Neutron width Γn (eV)
/// - `gg`: Radiation (gamma) width Γγ (eV)
/// - `gfa`: First fission width Γf1 (eV), 0.0 if non-fissile
/// - `gfb`: Second fission width Γf2 (eV), 0.0 if non-fissile
///
/// ## SLBW/MLBW (LRF=1/2)
/// - `gn`: Neutron width Γn (eV)
/// - `gg`: Radiation width Γγ (eV)
/// - `gfa`: Fission width Γf (eV)
/// - `gfb`: Not used (0.0)
///
/// Reference: ENDF-6 Formats Manual, Section 2.2.1
/// Reference: SAMMY manual, Section 2 (Scattering Theory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resonance {
    /// Resonance energy (eV).
    pub energy: f64,
    /// Total angular momentum J.
    pub j: f64,
    /// Neutron width Γn (eV).
    pub gn: f64,
    /// Radiation (capture/gamma) width Γγ (eV).
    pub gg: f64,
    /// First fission width (eV). Zero for non-fissile isotopes.
    pub gfa: f64,
    /// Second fission width (eV). Zero for non-fissile isotopes.
    pub gfb: f64,
}

// ─── LRF=7 (R-Matrix Limited) Data Structures ────────────────────────────────
//
// LRF=7 organizes resonances by spin group (J,π) rather than L-value.
// Each spin group has multiple explicit reaction channels. Resonances carry
// reduced width amplitudes γ per channel, not formal widths Γ.
//
// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY manual Ch. 3
// SAMMY source: rml/mrml01.f (reader), rml/mrml11.f (cross-section calc)

/// Particle pair definition for LRF=7 R-Matrix Limited.
///
/// Identifies the two particles in a reaction channel (e.g., neutron + W-184,
/// or gamma + W-185). Used to determine which channels are entrance (neutron)
/// channels and which are exit (fission, capture) channels.
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6, Table 2.2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticlePair {
    /// Mass of particle a (neutron = 1.0, in neutron mass units).
    pub ma: f64,
    /// Mass of particle b (target nucleus, in neutron mass units).
    pub mb: f64,
    /// Z*A of particle a (0 for neutron).
    pub za: f64,
    /// Z*A of particle b (target).
    pub zb: f64,
    /// Spin of particle a (1/2 for neutron).
    pub ia: f64,
    /// Spin of particle b (target spin I).
    pub ib: f64,
    /// Q-value for this reaction (eV). 0 for elastic.
    pub q: f64,
    /// Penetrability flag: 0 = calculated, 1 = tabulated.
    pub pnt: i32,
    /// Shift factor flag: 0 = calculated, 1 = tabulated.
    pub shf: i32,
    /// ENDF MT number identifying the reaction (2=elastic, 18=fission, 102=capture).
    pub mt: u32,
    /// Parity of particle a.
    pub pa: f64,
    /// Parity of particle b.
    pub pb: f64,
}

/// A single reaction channel within an LRF=7 spin group.
///
/// Specifies which particle pair, what orbital angular momentum, and the
/// radii used to compute penetrabilities and hard-sphere phase shifts.
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6, Table 2.3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmlChannel {
    /// Index into the parent `RmlData::particle_pairs` vector.
    pub particle_pair_idx: usize,
    /// Orbital angular momentum quantum number L.
    pub l: u32,
    /// Channel spin S = |I ± 1/2|.
    pub channel_spin: f64,
    /// Boundary condition B (usually 0.0; shifts the shift factor reference).
    pub boundary: f64,
    /// Effective channel radius APE (fm), used to compute P_l and S_l.
    pub effective_radius: f64,
    /// True channel radius APT (fm), used to compute hard-sphere phase φ_l.
    pub true_radius: f64,
}

/// A single resonance in LRF=7 format.
///
/// For KRM=2 (standard R-matrix), `widths` contains reduced width amplitudes
/// γ_c (eV^{1/2}) and `gamma_gamma = 0.0`.
///
/// For KRM=3 (Reich-Moore approximation), `widths` contains formal partial widths
/// Γ_c (eV) and `gamma_gamma` is the capture width Γ_γ (eV) used to form complex
/// pole energies: Ẽ_n = E_n - i·Γ_γn/2. The reduced amplitudes are derived as
/// γ_nc = √(Γ_nc / (2·P_c(E_n))).
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY manual §3.1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmlResonance {
    /// Resonance energy (eV).
    pub energy: f64,
    /// Width amplitudes per channel (eV^{1/2} for KRM=2; eV for KRM=3).
    ///
    /// Sign convention: sign(γ) encodes interference between resonances.
    /// `widths.len()` equals the number of channels in the parent `SpinGroup`.
    pub widths: Vec<f64>,
    /// Capture (gamma) width Γ_γ (eV) for KRM=3 Reich-Moore approximation.
    ///
    /// Used to make the R-matrix denominator complex: E_n → E_n - i·Γ_γ/2.
    /// Zero for KRM=2 (standard R-matrix, no complex energy shift).
    pub gamma_gamma: f64,
}

/// A spin group (J, π) in LRF=7 R-Matrix Limited format.
///
/// Groups all resonances with the same total angular momentum J and parity π.
/// Each spin group has its own set of reaction channels.
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY rml/mrml01.f
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpinGroup {
    /// Total angular momentum J.
    pub j: f64,
    /// Parity: +1.0 (even) or -1.0 (odd).
    pub parity: f64,
    /// Reaction channels for this spin group.
    pub channels: Vec<RmlChannel>,
    /// Resonances in this spin group.
    pub resonances: Vec<RmlResonance>,
}

/// Complete R-Matrix Limited data for one energy range (LRF=7).
///
/// Stored in `ResonanceRange::rml` when the formalism is `RMatrixLimited`.
///
/// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY rml/mrml01.f
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmlData {
    /// Target spin I.
    pub target_spin: f64,
    /// Atomic weight ratio (mass of target / neutron mass).
    pub awr: f64,
    /// Global scattering radius AP (fm); used as fallback when per-channel APE = 0.
    pub scattering_radius: f64,
    /// R-matrix type flag from ENDF CONT header.
    ///
    /// KRM=2: Standard multi-channel R-matrix (widths are reduced amplitudes γ).
    /// KRM=3: Reich-Moore approximation (widths are formal partial widths Γ;
    ///        capture enters via complex pole energies Ẽ_n = E_n - i·Γ_γ/2).
    /// Reference: ENDF-6 Formats Manual §2.2.1.6; SAMMY rml/mrml01.f
    pub krm: u32,
    /// Particle pair definitions (NPP entries).
    pub particle_pairs: Vec<ParticlePair>,
    /// Spin groups (NJS entries), one per (J, π) combination.
    pub spin_groups: Vec<SpinGroup>,
}

impl ResonanceData {
    /// Total number of resonances across all ranges and groups.
    ///
    /// For LRF=7 ranges, counts resonances across all spin groups.
    pub fn total_resonance_count(&self) -> usize {
        self.ranges.iter().map(|r| r.resonance_count()).sum()
    }

    /// Get all resonances in the resolved region (LRF=1/2/3 only), sorted by energy.
    ///
    /// Returns an empty vec for LRF=7 ranges; use `ResonanceRange::rml` directly
    /// to access R-Matrix Limited resonances.
    pub fn all_resolved_resonances(&self) -> Vec<&Resonance> {
        let mut resonances: Vec<&Resonance> = self
            .ranges
            .iter()
            .filter(|r| r.resolved && r.rml.is_none())
            .flat_map(|r| &r.l_groups)
            .flat_map(|lg| &lg.resonances)
            .collect();
        resonances.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        resonances
    }
}

impl ResonanceRange {
    /// Total resonance count for this range (works for both LRF=1/2/3 and LRF=7).
    pub fn resonance_count(&self) -> usize {
        if let Some(rml) = &self.rml {
            rml.spin_groups.iter().map(|sg| sg.resonances.len()).sum()
        } else {
            self.l_groups.iter().map(|lg| lg.resonances.len()).sum()
        }
    }
}

impl std::fmt::Display for ResonanceData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ResonanceData(ZA={}, AWR={:.4}, ranges={}, total_resonances={})",
            self.za,
            self.awr,
            self.ranges.len(),
            self.total_resonance_count()
        )
    }
}
