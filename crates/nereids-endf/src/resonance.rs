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
    /// Multi-Level Breit-Wigner (LRF=1).
    MLBW,
    /// Reich-Moore (LRF=2). This is the primary formalism for NEREIDS.
    ReichMoore,
    /// R-Matrix Limited (LRF=3).
    RMatrixLimited,
    /// General R-Matrix (LRF=7).
    GeneralRMatrix,
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
    /// Spin groups containing the actual resonance parameters.
    pub l_groups: Vec<LGroup>,
}

/// Parameters grouped by orbital angular momentum L.
///
/// In ENDF File 2 (LRF=2, Reich-Moore), resonances are grouped by L-value.
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
/// ## Reich-Moore (LRF=2)
/// - `gn`: Neutron width Γn (eV)
/// - `gg`: Radiation (gamma) width Γγ (eV)
/// - `gfa`: First fission width Γf1 (eV), 0.0 if non-fissile
/// - `gfb`: Second fission width Γf2 (eV), 0.0 if non-fissile
///
/// ## SLBW/MLBW (LRF=1)
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

impl ResonanceData {
    /// Total number of resonances across all ranges and groups.
    pub fn total_resonance_count(&self) -> usize {
        self.ranges
            .iter()
            .flat_map(|r| &r.l_groups)
            .map(|lg| lg.resonances.len())
            .sum()
    }

    /// Get all resonances in the resolved region, sorted by energy.
    pub fn all_resolved_resonances(&self) -> Vec<&Resonance> {
        let mut resonances: Vec<&Resonance> = self
            .ranges
            .iter()
            .filter(|r| r.resolved)
            .flat_map(|r| &r.l_groups)
            .flat_map(|lg| &lg.resonances)
            .collect();
        resonances.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        resonances
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
