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

// ─── ENDF TAB1: one-dimensional interpolation table ──────────────────────────
//
// TAB1 records encode a piecewise function y(x) with up to 5 interpolation laws
// (ENDF INT codes 1–5).  Used here for the energy-dependent scattering radius
// AP(E) when NRO=1.
//
// Reference: ENDF-6 Formats Manual §0.5 (TAB1 record type)

/// One-dimensional interpolation table (ENDF TAB1 record).
///
/// Stores piecewise-interpolated y(x) data.  Multiple interpolation regions
/// are supported via ENDF NBT/INT boundary pairs.
///
/// Interpolation law codes (ENDF INT):
/// - 1: Histogram (y = y_left for all x in interval)
/// - 2: Linear-linear
/// - 3: Linear in x, log in y
/// - 4: Log in x, linear in y
/// - 5: Log-log
///
/// Reference: ENDF-6 Formats Manual §0.5; SAMMY `inp/minp2.f90`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tab1 {
    /// Interpolation region boundaries (NBT, 1-based index of the last point
    /// in each region).  `boundaries.len() == interp_codes.len()`.
    pub boundaries: Vec<usize>,
    /// Interpolation law codes (INT) for each region.
    pub interp_codes: Vec<u32>,
    /// Data points as (x, y) pairs, sorted ascending in x.
    pub points: Vec<(f64, f64)>,
}

impl Tab1 {
    /// Evaluate the tabulated function at `x` by piecewise interpolation.
    ///
    /// Values outside the tabulated range are extrapolated by clamping to the
    /// nearest boundary value (no extrapolation beyond the table ends).
    pub fn evaluate(&self, x: f64) -> f64 {
        let pts = &self.points;
        if pts.is_empty() {
            return 0.0;
        }
        if x <= pts[0].0 {
            return pts[0].1;
        }
        if x >= pts[pts.len() - 1].0 {
            return pts[pts.len() - 1].1;
        }

        // Binary search: find the first index where pts[i].0 > x.
        // The interval containing x is [pts[i-1], pts[i]].
        let i = pts.partition_point(|(xi, _)| *xi <= x);
        let (x0, y0) = pts[i - 1];
        let (x1, y1) = pts[i];

        match self.interp_code_for_interval(i - 1) {
            1 => y0, // histogram: constant left value
            3 => {
                // lin-log: linear in x, log in y
                let t = (x - x0) / (x1 - x0);
                (y0.ln() + t * (y1.ln() - y0.ln())).exp()
            }
            4 => {
                // log-lin: log in x, linear in y
                let t = (x.ln() - x0.ln()) / (x1.ln() - x0.ln());
                y0 + t * (y1 - y0)
            }
            5 => {
                // log-log
                let t = (x.ln() - x0.ln()) / (x1.ln() - x0.ln());
                (y0.ln() + t * (y1.ln() - y0.ln())).exp()
            }
            _ => {
                // INT=2 (lin-lin) and any unknown code: linear interpolation
                let t = (x - x0) / (x1 - x0);
                y0 + t * (y1 - y0)
            }
        }
    }

    /// Return the ENDF interpolation code for the interval [pts[idx], pts[idx+1]].
    ///
    /// ENDF NBT boundaries are 1-based indices of the *last point* in each region.
    /// Interval `idx` (0-based) belongs to the first region j where `idx + 2 <= NBT[j]`.
    fn interp_code_for_interval(&self, idx: usize) -> u32 {
        for (j, &nbt) in self.boundaries.iter().enumerate() {
            if idx + 2 <= nbt {
                return self.interp_codes[j];
            }
        }
        self.interp_codes.last().copied().unwrap_or(2)
    }
}

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
    ///
    /// Constant value from the ENDF CONT header AP field.
    /// When `ap_table` is `Some`, use `scattering_radius_at(energy_ev)` instead
    /// of reading this field directly — the table provides the energy-dependent
    /// value and only falls back to this constant outside the table range.
    pub scattering_radius: f64,
    /// Energy-dependent scattering radius AP(E) (fm), present when NRO=1.
    ///
    /// ENDF-6 §2.2.1: when NRO≠0 a TAB1 record immediately follows the range
    /// CONT header to give AP(E) as a piecewise function.  At each energy the
    /// table value replaces the constant `scattering_radius` in penetrability,
    /// shift, and hard-sphere phase calculations.
    ///
    /// `None` when the range has NRO=0 (constant AP).
    ///
    /// Reference: ENDF-6 Formats Manual §2.2.1; SAMMY `mlb/mmlb1.f90`
    pub ap_table: Option<Tab1>,
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
    /// Penetrability flag.
    ///
    /// `PNT=1`: calculate penetrability P_c analytically (Blatt-Weisskopf).
    /// Used for massive-particle channels (neutron elastic).
    /// `PNT=0`: do not calculate penetrability; set P_c = 0 (photon/massless channels).
    pub pnt: i32,
    /// Shift factor flag.
    ///
    /// `SHF=1`: calculate shift factor S_c analytically (Blatt-Weisskopf).
    /// `SHF=0`: do not calculate; treat S_c = B_c so (S_c − B_c) = 0 in level matrix.
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
    /// Scattering radius at a given neutron energy.
    ///
    /// Returns the interpolated value from `ap_table` when NRO=1 (energy-dependent
    /// radius), or the constant `scattering_radius` when NRO=0.
    ///
    /// Use this method in all physics calculations that need the channel radius,
    /// rather than reading `scattering_radius` directly.
    ///
    /// # Arguments
    /// * `energy_ev` — Lab-frame neutron energy in eV.
    pub fn scattering_radius_at(&self, energy_ev: f64) -> f64 {
        if let Some(table) = &self.ap_table {
            table.evaluate(energy_ev)
        } else {
            self.scattering_radius
        }
    }

    /// Total resonance count for this range (works for both LRF=1/2/3 and LRF=7).
    pub fn resonance_count(&self) -> usize {
        if let Some(rml) = &self.rml {
            rml.spin_groups.iter().map(|sg| sg.resonances.len()).sum()
        } else {
            self.l_groups.iter().map(|lg| lg.resonances.len()).sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linlin_table(points: Vec<(f64, f64)>) -> Tab1 {
        let n = points.len();
        Tab1 {
            boundaries: vec![n],
            interp_codes: vec![2],
            points,
        }
    }

    /// Linear-linear interpolation in the interior of the table.
    #[test]
    fn test_tab1_linlin_interior() {
        let table = make_linlin_table(vec![(1.0, 10.0), (5.0, 30.0), (10.0, 5.0)]);
        // midpoint of [1,5]: x=3 → 10 + (3-1)/(5-1) * (30-10) = 10 + 0.5*20 = 20
        let v = table.evaluate(3.0);
        assert!((v - 20.0).abs() < 1e-10, "lin-lin midpoint, got {v}");
        // midpoint of [5,10]: x=7.5 → 30 + (7.5-5)/(10-5) * (5-30) = 30 + 0.5*(-25) = 17.5
        let v2 = table.evaluate(7.5);
        assert!(
            (v2 - 17.5).abs() < 1e-10,
            "lin-lin second interval, got {v2}"
        );
    }

    /// Values outside the table range clamp to the boundary value.
    #[test]
    fn test_tab1_clamping() {
        let table = make_linlin_table(vec![(2.0, 5.0), (8.0, 15.0)]);
        assert_eq!(table.evaluate(0.0), 5.0, "below low bound");
        assert_eq!(table.evaluate(100.0), 15.0, "above high bound");
        assert_eq!(table.evaluate(2.0), 5.0, "at low bound");
        assert_eq!(table.evaluate(8.0), 15.0, "at high bound");
    }

    /// Histogram interpolation (INT=1): y stays constant from left endpoint.
    #[test]
    fn test_tab1_histogram() {
        let table = Tab1 {
            boundaries: vec![3],
            interp_codes: vec![1],
            points: vec![(0.0, 10.0), (5.0, 20.0), (10.0, 30.0)],
        };
        assert_eq!(
            table.evaluate(2.5),
            10.0,
            "histogram: should return left value"
        );
        assert_eq!(table.evaluate(7.5), 20.0, "histogram: second interval");
    }

    /// Two-region table: lin-lin for low energies, lin-log for high.
    #[test]
    fn test_tab1_multiregion() {
        // Region 0 (INT=2, lin-lin): points 0..2  (NBT=2)
        // Region 1 (INT=3, lin-log): points 2..4  (NBT=4)
        // Points: (1,1), (3,3), (10,3), (100,30)
        let table = Tab1 {
            boundaries: vec![2, 4],
            interp_codes: vec![2, 3],
            points: vec![(1.0, 1.0), (3.0, 3.0), (10.0, 3.0), (100.0, 30.0)],
        };
        // Interval 0 ([1,3], INT=2): x=2 → lin-lin: 1 + (2-1)/(3-1) * (3-1) = 2
        assert!(
            (table.evaluate(2.0) - 2.0).abs() < 1e-10,
            "region 0 lin-lin"
        );
        // Interval 2 ([10,100], INT=3 lin-log): x=31.62 (~sqrt(10*100))
        // t = (31.62-10)/(100-10) = 0.2402
        // y = exp(ln3 + 0.2402*(ln30 - ln3)) = exp(1.0986 + 0.2402*2.3026) = exp(1.6522) ≈ 5.22
        // Not exact; just verify result is strictly between 3 and 30.
        let v = table.evaluate(31.62);
        assert!(v > 3.0 && v < 30.0, "region 1 lin-log: {v}");
    }

    /// scattering_radius_at falls back to constant when ap_table is None.
    #[test]
    fn test_scattering_radius_at_constant() {
        let range = ResonanceRange {
            energy_low: 1e-5,
            energy_high: 1e4,
            resolved: true,
            formalism: crate::resonance::ResonanceFormalism::ReichMoore,
            target_spin: 0.0,
            scattering_radius: 9.4285,
            ap_table: None,
            l_groups: vec![],
            rml: None,
        };
        assert_eq!(range.scattering_radius_at(1.0), 9.4285);
        assert_eq!(range.scattering_radius_at(1000.0), 9.4285);
    }

    /// scattering_radius_at interpolates from ap_table when NRO=1.
    #[test]
    fn test_scattering_radius_at_energy_dependent() {
        // AP goes from 8.0 fm at 1 eV to 10.0 fm at 1000 eV (lin-lin).
        let table = make_linlin_table(vec![(1.0, 8.0), (1000.0, 10.0)]);
        let range = ResonanceRange {
            energy_low: 1e-5,
            energy_high: 1e4,
            resolved: true,
            formalism: crate::resonance::ResonanceFormalism::ReichMoore,
            target_spin: 0.0,
            scattering_radius: 9.0, // constant fallback (ignored when table is Some)
            ap_table: Some(table),
            l_groups: vec![],
            rml: None,
        };
        // At 1 eV: 8.0 fm
        assert!((range.scattering_radius_at(1.0) - 8.0).abs() < 1e-10);
        // At 1000 eV: 10.0 fm
        assert!((range.scattering_radius_at(1000.0) - 10.0).abs() < 1e-10);
        // At 500.5 eV (midpoint): 9.0 fm
        let mid = range.scattering_radius_at(500.5);
        assert!((mid - 9.0).abs() < 0.01, "midpoint AP ≈ 9.0, got {mid}");
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
