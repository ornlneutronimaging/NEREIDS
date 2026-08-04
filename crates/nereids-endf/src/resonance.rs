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
/// Interpolation law codes (ENDF INT), per ENDF-6 Formats Manual §0.5:
/// - 1: Histogram (y constant = y_left)
/// - 2: Linear-linear
/// - 3: Log in x, linear in y  (y linear in ln(x))
/// - 4: Linear in x, log in y  (ln(y) linear in x)
/// - 5: Log-log
///
/// Verified against SAMMY OpenScale `CELibrary/Interpolate.h`:
///   case 3 → `LinByLog` = log-x/linear-y
///   case 4 → `LogByLin` = linear-x/log-y
///
/// Reference: ENDF-6 Formats Manual §0.5; SAMMY OpenScale `CELibrary/Interpolate.h`
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
    /// Values outside the tabulated range are clamped to the nearest endpoint
    /// (no extrapolation).
    ///
    /// Log-interpolation modes (INT=3, 4, 5) require strictly positive
    /// arguments for the logarithm.  If a tabulated value or x-coordinate
    /// is non-positive where a logarithm would be taken, the function
    /// transparently falls back to lin-lin interpolation for that interval
    /// rather than producing NaN or panicking.  In practice, ENDF AP(E)
    /// tables always have positive x (energy) and positive y (radius in fm),
    /// so this guard is defensive only.
    pub fn evaluate(&self, x: f64) -> f64 {
        let pts = &self.points;
        if pts.is_empty() {
            // The parser rejects NP=0, so an empty table indicates a bug in
            // test-code construction.  Panic in debug builds; return 0.0 in
            // release to avoid UB.
            debug_assert!(
                !pts.is_empty(),
                "Tab1::evaluate called with empty points table"
            );
            return 0.0;
        }
        // NaN/±inf: partition_point's comparisons are all false for NaN,
        // returning index 0, and pts[0 - 1] would underflow.  Clamp to the
        // nearest finite endpoint instead.
        if !x.is_finite() {
            debug_assert!(x.is_finite(), "Tab1::evaluate: non-finite argument {x}");
            return if x > 0.0 {
                pts[pts.len() - 1].1
            } else {
                pts[0].1
            };
        }
        if x <= pts[0].0 {
            return pts[0].1;
        }
        if x >= pts[pts.len() - 1].0 {
            return pts[pts.len() - 1].1;
        }

        // Binary search: find the first index where pts[i].0 > x.
        // The interval containing x is [pts[i-1], pts[i]].
        // Because the outer clamps ensure pts[0].0 < x < pts[last].0,
        // we are guaranteed x0 < x1 (strict), so (x1 - x0) > 0.
        let i = pts.partition_point(|(xi, _)| *xi <= x);
        let (x0, y0) = pts[i - 1];
        let (x1, y1) = pts[i];

        // Fallback to lin-lin for any interval; used when log guards fire.
        let lin_lin = || {
            let t = (x - x0) / (x1 - x0);
            y0 + t * (y1 - y0)
        };

        match self.interp_code_for_interval(i - 1) {
            1 => y0, // histogram: constant left value
            3 => {
                // INT=3: y linear in ln(x) — log in x, linear in y.
                // SAMMY OpenScale: case 3 → LinByLog (requires x0, x1, x > 0).
                if x0 > 0.0 && x1 > 0.0 && x > 0.0 {
                    let t = (x.ln() - x0.ln()) / (x1.ln() - x0.ln());
                    y0 + t * (y1 - y0)
                } else {
                    lin_lin()
                }
            }
            4 => {
                // INT=4: ln(y) linear in x — linear in x, log in y.
                // SAMMY OpenScale: case 4 → LogByLin (requires y0, y1 > 0).
                if y0 > 0.0 && y1 > 0.0 {
                    let t = (x - x0) / (x1 - x0);
                    (y0.ln() + t * (y1.ln() - y0.ln())).exp()
                } else {
                    lin_lin()
                }
            }
            5 => {
                // log-log; requires x0, x1, x, y0, y1 > 0
                if x0 > 0.0 && x1 > 0.0 && x > 0.0 && y0 > 0.0 && y1 > 0.0 {
                    let t = (x.ln() - x0.ln()) / (x1.ln() - x0.ln());
                    (y0.ln() + t * (y1.ln() - y0.ln())).exp()
                } else {
                    lin_lin()
                }
            }
            _ => {
                // INT=2 (lin-lin) and any unknown code: linear interpolation
                lin_lin()
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
    /// R-Matrix Limited (LRF=7). General multi-channel formalism (W, Ta, Zr,
    /// etc. in ENDF/B-VIII.0). Parsed for cursor alignment but not evaluated:
    /// the RML physics was removed because its closed-channel treatment was
    /// incomplete (the Coulomb/SHF=1 closed-channel shift was unimplemented)
    /// and the evaluator was never validated against SAMMY. Ranges tagged
    /// `RMatrixLimited` are non-evaluable and resolve to Skip.
    RMatrixLimited,
    /// Unresolved Resonance Region (LRU=2). Parsed for cursor alignment but not
    /// evaluated: NEREIDS does not compute URR average cross sections. The
    /// Hauser-Feshbach path was removed because it lacked the ENDF
    /// width-fluctuation correction (a systematically wrong average). Ranges
    /// tagged `Unresolved` are non-evaluable and resolve to Skip.
    Unresolved,
    /// Scattering-radius-only range (LRU=0). ENDF-6 §2.1: the standard stanza
    /// for materials given a scattering radius but no resonance parameters. It
    /// carries no resonances, so there is nothing to evaluate — the range is a
    /// non-evaluable placeholder that resolves to Skip. Captured (rather than
    /// dropped) so a file whose only range is LRU=0 is rejected with an error
    /// that names the LRU=0 span instead of misreporting an empty file.
    ScatteringRadiusOnly,
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
    /// value, clamping to the nearest endpoint for energies outside the table
    /// range.  This constant is only used when `ap_table` is `None` (NRO=0).
    pub scattering_radius: f64,
    /// NAPS flag: scattering radius calculation control.
    ///
    /// NAPS=0: use the channel radius for penetrability/shift calculations.
    /// NAPS=1: use the scattering radius (AP or AP(E)) for penetrability/shift.
    /// Reference: ENDF-6 Formats Manual §2.2.1
    #[serde(default)]
    pub naps: i32,
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
    #[serde(default)]
    pub ap_table: Option<Tab1>,
    /// Spin groups for LRF=1/2/3 (L-grouped). Empty for LRF=7 and LRU=2 (both
    /// parsed-and-skipped, non-evaluable).
    pub l_groups: Vec<LGroup>,
    /// R-external (background R-matrix) entries per spin group.
    ///
    /// Diagonal, real-valued corrections to the R-matrix that approximate
    /// the effect of distant (unresolved) resonances.  Keyed by (L, J).
    ///
    /// Populated from SAMMY's "R-EXTERNAL PARAMETERS FOLLOW" section.
    /// Empty for ENDF-only data or SAMMY cases without R-external.
    ///
    /// SAMMY Ref: Manual Section II.B.1.d, mpar03.f90 Readrx
    #[serde(default)]
    pub r_external: Vec<RExternalEntry>,
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
    /// Q-value for competitive width (eV). Only meaningful for BW formalisms
    /// (LRF=1/2) where LRX=1; zero otherwise.
    /// Reference: ENDF-6 Formats Manual §2.2.1.1, L-value CONT record (C2 field).
    #[serde(default)]
    pub qx: f64,
    /// Competitive width flag. LRX=0: no competitive width; LRX=1: competitive
    /// reaction exists (width = GT - GN - GG - GF). Only used in BW formalisms.
    /// Reference: ENDF-6 Formats Manual §2.2.1.1, L-value CONT record (L2 field).
    #[serde(default)]
    pub lrx: i32,
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

// ─── R-External (Background R-Matrix) ─────────────────────────────────────────

/// R-external (background R-matrix) parameters for a single spin group channel.
///
/// Parameterizes smooth R-matrix contribution from distant (unresolved)
/// resonances.  The background R-matrix is diagonal and real-valued,
/// parameterized as a logarithmic polynomial in energy.
///
/// ## Formula
/// ```text
/// R_ext(E) = R_con + R_lin·E + R_quad·E²
///          + s_lin·(E_up − E_low)
///          − (s_con + s_lin·E)·ln[(E_up − E) / (E − E_low)]
/// ```
///
/// SAMMY Ref: Manual Section II.B.1.d, mcro2.f90 lines 180-193
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RExternalEntry {
    /// Orbital angular momentum L of the spin group.
    pub l: u32,
    /// Total angular momentum J (signed, per SAMMY convention).
    pub j: f64,
    /// Lower energy bound (eV).
    pub e_low: f64,
    /// Upper energy bound (eV).
    pub e_up: f64,
    /// Constant term in R-matrix polynomial.
    pub r_con: f64,
    /// Linear coefficient (eV⁻¹).
    pub r_lin: f64,
    /// Constant logarithmic coefficient.
    pub s_con: f64,
    /// Linear logarithmic coefficient (eV⁻¹).
    pub s_lin: f64,
    /// Quadratic coefficient (eV⁻²).
    pub r_quad: f64,
}

impl RExternalEntry {
    /// Evaluate R_ext(E) at the given energy.
    ///
    /// The polynomial part (`r_con + r_lin·E + r_quad·E²`) applies at all
    /// energies.  The logarithmic terms are only added when `E` is strictly
    /// inside `(e_low, e_up)`.
    ///
    /// SAMMY Ref: mcro2.f90 Setr_Cro, lines 180-193
    pub fn evaluate(&self, energy_ev: f64) -> f64 {
        let e = energy_ev;
        let mut r = self.r_con + self.r_lin * e + self.r_quad * e * e;

        let e_up_diff = self.e_up - e;
        let e_low_diff = e - self.e_low;
        if e_up_diff > 0.0 && e_low_diff > 0.0 {
            let log_val = (e_up_diff / e_low_diff).ln();
            r -= (self.s_con + self.s_lin * e) * log_val;
            r += self.s_lin * (self.e_up - self.e_low);
        }

        r
    }
}

impl ResonanceData {
    /// Total number of resonances across all ranges and groups.
    ///
    /// Counts the L-grouped resonances of evaluable LRF=1/2/3 ranges. LRF=7
    /// (and LRU=2) ranges are parsed-and-skipped with empty `l_groups`, so
    /// they contribute 0 — NEREIDS does not evaluate them.
    ///
    /// A low count for a given evaluation reflects that evaluation's
    /// resolved-resonance-region (RRR) extent, **not** a dropped energy range.
    /// The parser reads every NER range and errors on unconsumed MF2/MT151 data,
    /// so ranges are never silently discarded. For example, Ta-181 in
    /// ENDF/B-VIII.0 returns 76 (RRR only to 330 eV, plus an unresolved URR that
    /// contributes 0 discrete resonances), whereas ENDF/B-VIII.1 extended the RRR
    /// to 2554 eV and returns 565. See `test_parse_ta181_endf8_0_resonance_count`
    /// in `parser.rs`, which pins the VIII.0 count as a regression guard.
    pub fn total_resonance_count(&self) -> usize {
        self.ranges.iter().map(|r| r.resonance_count()).sum()
    }

    /// Ranges that are parsed but NOT evaluated (non-evaluable placeholders).
    ///
    /// These are the LRF=7 (R-Matrix Limited), LRU=2 (unresolved), and LRU=0
    /// (scattering-radius-only) ranges: the parser consumes their records for
    /// cursor alignment and discards any resonance parameters they carry in
    /// the file (LRF=7 and LRU=2 tapes do carry them; LRU=0 has none), so the
    /// stored placeholder holds none and contributes exactly zero to every
    /// cross-section. Any physics computed over their energy span reflects
    /// only the *other* ranges of the evaluation. Callers that surface data
    /// to users should warn when this list is non-empty.
    pub fn unevaluated_ranges(&self) -> Vec<&ResonanceRange> {
        self.ranges.iter().filter(|r| !r.is_evaluable()).collect()
    }

    /// Whether any range is a non-evaluable parse-and-skip placeholder.
    ///
    /// See [`Self::unevaluated_ranges`].
    pub fn has_unevaluated_ranges(&self) -> bool {
        self.ranges.iter().any(|r| !r.is_evaluable())
    }

    /// Whether at least one range can actually produce non-zero cross-sections.
    ///
    /// `false` means every range is a parse-and-skip placeholder (or there are
    /// no ranges at all): the evaluation would return zero cross-section over
    /// its full grid (transmission ≡ 1). This is the load-time acceptance
    /// predicate — the parser rejects such an evaluation, and the project
    /// loader drops it from the ENDF cache so a stale removed-physics payload
    /// cannot silently restore as a zero-cross-section isotope.
    pub fn has_evaluable_range(&self) -> bool {
        self.ranges.iter().any(|r| r.is_evaluable())
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

    /// Total discrete-resonance count for this range.
    ///
    /// Counts the L-grouped resonances of an evaluable LRF=1/2/3 range. LRF=7
    /// (and LRU=2) ranges are parsed-and-skipped with empty `l_groups`, so they
    /// contribute 0 — NEREIDS does not evaluate them.
    pub fn resonance_count(&self) -> usize {
        self.l_groups.iter().map(|lg| lg.resonances.len()).sum()
    }

    /// Can this range actually produce non-zero cross-sections?
    ///
    /// Evaluable means a resolved (LRU=1) range using one of the implemented
    /// formalisms — SLBW (LRF=1), MLBW (LRF=2), or Reich-Moore (LRF=3) — with
    /// at least one resonance-bearing L-group. A resolved range whose L-groups
    /// are all empty evaluates to exactly zero everywhere (including potential
    /// scattering: J-groups derive from the resonance list, so an empty list
    /// yields no J-groups), which is the silently-inert outcome this predicate
    /// exists to catch. LRF=7 (R-Matrix Limited), LRU=2 (unresolved), and
    /// LRU=0 (scattering-radius-only) ranges are parse-and-skip placeholders —
    /// consumed for cursor alignment, never evaluated — so they return `false`
    /// and contribute zero cross-section over their energy span.
    pub fn is_evaluable(&self) -> bool {
        self.resolved
            && matches!(
                self.formalism,
                ResonanceFormalism::SLBW
                    | ResonanceFormalism::MLBW
                    | ResonanceFormalism::ReichMoore
            )
            && self.l_groups.iter().any(|lg| !lg.resonances.is_empty())
    }

    /// One-line diagnostic for a parse-and-skip placeholder range, e.g.
    /// `"LRF=7 (R-Matrix Limited) over [1.000000e-5, 1.000000e3] eV"`.
    ///
    /// Shared by the parser's no-evaluable-content error, the Python-binding
    /// `UserWarning`, and the GUI load log so all three surfaces describe a
    /// skipped range identically. Only meaningful for ranges where
    /// [`Self::is_evaluable`] is `false`: the resolved-formalism arms label
    /// the accepted-but-inert no-resonance shape (callers never pass an
    /// evaluable range).
    pub fn skip_description(&self) -> String {
        let kind = match self.formalism {
            ResonanceFormalism::RMatrixLimited => "LRF=7 (R-Matrix Limited)",
            ResonanceFormalism::Unresolved => "LRU=2 (URR)",
            ResonanceFormalism::ScatteringRadiusOnly => {
                "LRU=0 (scattering-radius-only, no resonance parameters)"
            }
            // A resolved LRF=1/2/3 range reaches here only when it carries no
            // resonances (every L-group empty) — accepted, warn-and-skip.
            ResonanceFormalism::SLBW => "LRF=1 (SLBW) resolved range with no resonances",
            ResonanceFormalism::MLBW => "LRF=2 (MLBW) resolved range with no resonances",
            ResonanceFormalism::ReichMoore => {
                "LRF=3 (Reich-Moore) resolved range with no resonances"
            }
        };
        format!(
            "{kind} over [{:.6e}, {:.6e}] eV",
            self.energy_low, self.energy_high
        )
    }
}

/// Group resonances by their total angular momentum J value (test-only).
///
/// Returns a vector of `(J, resonances)` pairs. Two J values are considered
/// equal if they differ by less than [`nereids_core::constants::QUANTUM_NUMBER_EPS`].
///
/// Note: The physics crate uses `group_resonances_by_j` (in `reich_moore.rs`)
/// for cross-section precomputation, which builds per-resonance invariants
/// directly during grouping. This function is retained for unit-level tests
/// of the grouping logic itself.
#[cfg(test)]
fn group_by_j(resonances: &[Resonance]) -> Vec<(f64, Vec<&Resonance>)> {
    let mut groups: Vec<(f64, Vec<&Resonance>)> = Vec::new();
    for res in resonances {
        let j = res.j;
        if let Some(group) = groups
            .iter_mut()
            .find(|(gj, _)| (*gj - j).abs() < nereids_core::constants::QUANTUM_NUMBER_EPS)
        {
            group.1.push(res);
        } else {
            groups.push((j, vec![res]));
        }
    }
    groups
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

    /// Two-region table: lin-lin for low energies, log-x/lin-y (INT=3) for high.
    #[test]
    fn test_tab1_multiregion() {
        // Region 0 (INT=2, lin-lin): points 0..2  (NBT=2)
        // Region 1 (INT=3, log in x / linear in y): points 2..4  (NBT=4)
        // Points: (1,1), (3,3), (10,3), (100,30)
        let table = Tab1 {
            boundaries: vec![2, 4],
            interp_codes: vec![2, 3],
            points: vec![(1.0, 1.0), (3.0, 3.0), (10.0, 3.0), (100.0, 30.0)],
        };
        // Interval 0 ([1,3], INT=2 lin-lin): x=2 → 1 + (2-1)/(3-1) * (3-1) = 2
        assert!(
            (table.evaluate(2.0) - 2.0).abs() < 1e-10,
            "region 0 lin-lin"
        );
        // Interval 1 ([3,10], INT=3 log-x/lin-y): x=5.
        // y0==y1==3.0, so any interpolation mode yields 3.0 regardless.
        // This verifies the region boundary is crossed correctly and that
        // x=5 routes to interval 1 (not interval 0 or 2).
        assert!(
            (table.evaluate(5.0) - 3.0).abs() < 1e-10,
            "region 1 INT=3 (constant y segment): x=5 should give 3.0"
        );
        // Interval 2 ([10,100], INT=3 log-x/lin-y): x=31.62 ≈ sqrt(10*100) = geometric midpoint.
        // INT=3: t = ln(x/x0) / ln(x1/x0) = ln(31.62/10) / ln(100/10) = ln(3.162)/ln(10) ≈ 0.5
        // y = y0 + t*(y1 - y0) = 3 + 0.5*(30 - 3) = 16.5
        let v = table.evaluate(31.62);
        assert!(
            (v - 16.5).abs() < 0.1,
            "region 2 INT=3 at geometric midpoint: expected 16.5, got {v}"
        );
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
            naps: 1,
            ap_table: None,
            l_groups: vec![],
            r_external: vec![],
        };
        assert_eq!(range.scattering_radius_at(1.0), 9.4285);
        assert_eq!(range.scattering_radius_at(1000.0), 9.4285);
    }

    /// `is_evaluable` is content-sensitive: a resolved LRF=1/2/3 range whose
    /// L-groups are all empty is inert (zero cross-section everywhere,
    /// potential scattering included, because J-groups derive from the
    /// resonance list) and must not count as evaluable.
    #[test]
    fn test_is_evaluable_requires_resonances() {
        let mut range = ResonanceRange {
            energy_low: 1e-5,
            energy_high: 1e4,
            resolved: true,
            formalism: crate::resonance::ResonanceFormalism::MLBW,
            target_spin: 0.0,
            scattering_radius: 9.4,
            naps: 1,
            ap_table: None,
            l_groups: vec![LGroup {
                l: 0,
                awr: 236.0,
                apl: 0.0,
                qx: 0.0,
                lrx: 0,
                resonances: vec![],
            }],
            r_external: vec![],
        };
        assert!(!range.is_evaluable(), "all-empty L-groups must be inert");
        range.l_groups[0].resonances.push(Resonance {
            energy: 6.674,
            j: 0.5,
            gn: 1.5e-3,
            gg: 2.3e-2,
            gfa: 0.0,
            gfb: 0.0,
        });
        assert!(
            range.is_evaluable(),
            "a resonance-bearing L-group is evaluable"
        );
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
            naps: 1,
            ap_table: Some(table),
            l_groups: vec![],
            r_external: vec![],
        };
        // At 1 eV: 8.0 fm
        assert!((range.scattering_radius_at(1.0) - 8.0).abs() < 1e-10);
        // At 1000 eV: 10.0 fm
        assert!((range.scattering_radius_at(1000.0) - 10.0).abs() < 1e-10);
        // At 500.5 eV (midpoint): 9.0 fm
        let mid = range.scattering_radius_at(500.5);
        assert!((mid - 9.0).abs() < 0.01, "midpoint AP ≈ 9.0, got {mid}");
    }

    /// Log-guard fallback: if an x-coordinate is non-positive in an INT=3
    /// (log-x, linear-y) interval, evaluate() falls back to lin-lin.
    #[test]
    fn test_tab1_log_guard_nonpositive_x() {
        // INT=3 (log in x, linear in y) with x0=0.0 — 0.0_f64.ln() = -inf without guard.
        let table = Tab1 {
            boundaries: vec![2],
            interp_codes: vec![3], // log in x, linear in y
            points: vec![(0.0, 8.0), (10.0, 10.0)],
        };
        // x=0.0 is at the left boundary; evaluate() clamps to y=8.0 before interpolation.
        assert!((table.evaluate(0.0) - 8.0).abs() < 1e-10);
        // x=5.0 is interior; x0=0.0 triggers the log guard → lin-lin fallback.
        let result = table.evaluate(5.0);
        assert!(
            result.is_finite(),
            "fallback to lin-lin should give finite result, got {result}"
        );
    }

    /// Log-guard fallback: if a y-value is non-positive in an INT=4
    /// (linear-x, log-y) interval, evaluate() falls back to lin-lin.
    #[test]
    fn test_tab1_log_guard_nonpositive_y() {
        // INT=4 (linear in x, log in y) with y0=0.0 — 0.0_f64.ln() = -inf without guard.
        let table = Tab1 {
            boundaries: vec![2],
            interp_codes: vec![4], // linear in x, log in y
            points: vec![(1.0, 0.0), (10.0, 1.0)],
        };
        let result = table.evaluate(5.0);
        assert!(
            result.is_finite(),
            "fallback to lin-lin should give finite result, got {result}"
        );
    }

    /// INT=3 (log in x, linear in y): verify correct formula against analytic values.
    #[test]
    fn test_tab1_logx_linear_y() {
        // Points at x=1 (y=0) and x=100 (y=2.0).
        // At x=10: t = ln(10)/ln(100) = 1/2, y = 0 + 0.5*2 = 1.0
        let table = Tab1 {
            boundaries: vec![2],
            interp_codes: vec![3], // log in x, linear in y
            points: vec![(1.0, 0.0), (100.0, 2.0)],
        };
        let y = table.evaluate(10.0);
        assert!(
            (y - 1.0).abs() < 1e-12,
            "INT=3 at geometric midpoint x=10: expected y=1.0, got {y}"
        );
    }

    /// INT=4 (linear in x, log in y): verify correct formula against analytic values.
    #[test]
    fn test_tab1_linear_x_logy() {
        // Points at x=0 (y=1) and x=2 (y=e²).
        // At x=1 (midpoint): t=0.5, y = exp(0 + 0.5*2) = exp(1) = e
        let e = std::f64::consts::E;
        let table = Tab1 {
            boundaries: vec![2],
            interp_codes: vec![4], // linear in x, log in y
            points: vec![(0.0, 1.0), (2.0, e * e)],
        };
        let y = table.evaluate(1.0);
        assert!(
            (y - e).abs() < 1e-12,
            "INT=4 at midpoint x=1: expected y=e={e:.6}, got {y:.6}"
        );
    }

    #[test]
    fn test_group_by_j() {
        // Empty input
        let groups = group_by_j(&[]);
        assert!(groups.is_empty());

        // Single resonance
        let r1 = Resonance {
            energy: 6.67,
            j: 0.5,
            gn: 0.001,
            gg: 0.023,
            gfa: 0.0,
            gfb: 0.0,
        };
        let single = [r1.clone()];
        let groups = group_by_j(&single);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1.len(), 1);

        // Multiple J values
        let r2 = Resonance {
            j: 1.5,
            ..r1.clone()
        };
        let r3 = Resonance {
            j: 0.5,
            energy: 20.0,
            ..r1.clone()
        };
        let multi = [r1, r2, r3];
        let groups = group_by_j(&multi);
        assert_eq!(groups.len(), 2); // J=0.5 and J=1.5
        // J=0.5 group should have 2 resonances
        let j05 = groups
            .iter()
            .find(|(j, _)| (*j - 0.5).abs() < nereids_core::constants::QUANTUM_NUMBER_EPS)
            .unwrap();
        assert_eq!(j05.1.len(), 2);
    }
}

/// Synthetic [`ResonanceData`] / [`ResonanceRange`] builders for cross-crate
/// tests.  Gated on `#[cfg(any(test, feature = "test-support"))]`: visible to
/// in-crate `#[cfg(test)] mod tests` AND to integration tests in sibling crates
/// that enable the `test-support` feature in their `[dev-dependencies]`.  Never
/// compiled into release builds.  Consolidates previously-scattered ad-hoc
/// builders into one named API, mirroring PR #545's
/// `nereids_physics::resolution::test_support`.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::{LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange};
    use nereids_core::types::Isotope;

    /// Parameters for [`single_resonance`].  No `Default`: every field is
    /// required because different absorbing sites used different "defaults",
    /// and forcing callers to be explicit prevents silent drift.
    pub struct SingleResonanceParams {
        pub energy: f64,
        pub gamma_n: f64,
        pub gamma_g: f64,
        pub j: f64,
        pub l: u32,
        pub awr: f64,
        pub target_spin: f64,
        pub scattering_radius: f64,
    }

    // --- Private structural helpers ---
    //
    // The public fixtures below all build a single `ResonanceRange` with one
    // `LGroup` whose body varies only in well-defined ways.  The two private
    // helpers below absorb the structural skeleton (resolved/ap_table/
    // r_external/qx/lrx/gfa/gfb) so each public helper carries only the
    // physically-meaningful parameters.

    /// One resolved `ResonanceRange` with a single L-group.  All "structural
    /// invariants" (resolved, `ap_table`/`r_external`,
    /// `qx`/`lrx`) get the minimal-fixture defaults.
    #[allow(clippy::too_many_arguments)]
    fn make_range(
        energy_low: f64,
        energy_high: f64,
        formalism: ResonanceFormalism,
        target_spin: f64,
        scattering_radius: f64,
        naps: i32,
        l: u32,
        lgroup_awr: f64,
        apl: f64,
        resonances: Vec<Resonance>,
    ) -> ResonanceRange {
        ResonanceRange {
            energy_low,
            energy_high,
            resolved: true,
            formalism,
            target_spin,
            scattering_radius,
            naps,
            l_groups: vec![LGroup {
                l,
                awr: lgroup_awr,
                apl,
                qx: 0.0,
                lrx: 0,
                resonances,
            }],
            ap_table: None,
            r_external: vec![],
        }
    }

    /// Wrap a `ResonanceRange` in a `ResonanceData` for caller-chosen `(z, a, awr)`.
    fn wrap(z: u32, a: u32, awr: f64, range: ResonanceRange) -> ResonanceData {
        ResonanceData {
            isotope: Isotope::new(z, a).unwrap(),
            za: z * 1000 + a,
            awr,
            ranges: vec![range],
        }
    }

    /// One `Resonance` with `gfa = gfb = 0` (the common minimal-fixture case).
    fn res(energy: f64, j: f64, gn: f64, gg: f64) -> Resonance {
        Resonance {
            energy,
            j,
            gn,
            gg,
            gfa: 0.0,
            gfb: 0.0,
        }
    }

    // --- Public fixtures ---

    /// Canonical U-238 6.674 eV Reich-Moore single-resonance.  Byte-identical
    /// anchor for the most common synthetic case; absorbs four previously-
    /// duplicated copies across pipeline / physics / fitting.
    pub fn u238_single_resonance() -> ResonanceData {
        u238_with_formalism(ResonanceFormalism::ReichMoore)
    }

    /// Same as [`u238_single_resonance`] with a caller-chosen formalism.
    /// Default RM-style range `1e-5 .. 1e4` eV.
    pub fn u238_with_formalism(formalism: ResonanceFormalism) -> ResonanceData {
        wrap(
            92,
            238,
            236.006,
            make_range(
                1e-5,
                1e4,
                formalism,
                0.0,
                9.4285,
                1,
                0,
                236.006,
                0.0,
                vec![res(6.674, 0.5, 1.493e-3, 23.0e-3)],
            ),
        )
    }

    /// As [`u238_with_formalism`] with wider range `1e-6 .. 1e5` eV for the
    /// velocity-factor regression suite (`slbw_velocity_factor.rs`).
    pub fn u238_with_formalism_wide_range(formalism: ResonanceFormalism) -> ResonanceData {
        wrap(
            92,
            238,
            236.006,
            make_range(
                1e-6,
                1e5,
                formalism,
                0.0,
                9.4285,
                1,
                0,
                236.006,
                0.0,
                vec![res(6.674, 0.5, 1.493e-3, 23.0e-3)],
            ),
        )
    }

    /// U-238 with three well-separated s-wave resonances (6.674, 20.87,
    /// 36.68 eV), Reich-Moore.  Multiple dips at different energies break the
    /// (t0, L_scale) degeneracy that a single resonance leaves — a single dip
    /// cannot separate a TOF offset from a flight-path scale.  Used by the
    /// energy-scale calibration and joint temperature-recovery tests (#634);
    /// analogous to (not numerically identical with) the Python
    /// `TestFitEnergyScaleRecovery` fixture.
    pub fn u238_three_resonances() -> ResonanceData {
        wrap(
            92,
            238,
            236.006,
            make_range(
                1e-6,
                1e5,
                ResonanceFormalism::ReichMoore,
                0.0,
                9.4285,
                1,
                0,
                236.006,
                0.0,
                vec![
                    res(6.674, 0.5, 1.493e-3, 23.0e-3),
                    res(20.87, 0.5, 10.3e-3, 26.0e-3),
                    res(36.68, 0.5, 34.4e-3, 27.0e-3),
                ],
            ),
        )
    }

    /// Fully-parameterized U-238 ZA single-resonance, Reich-Moore.  For the
    /// RM-harness tests that vary (E_r, Γn, Γγ, J, L, AWR, I, AP) per case.
    pub fn single_resonance(p: SingleResonanceParams) -> ResonanceData {
        wrap(
            92,
            238,
            p.awr,
            make_range(
                1e-5,
                1e4,
                ResonanceFormalism::ReichMoore,
                p.target_spin,
                p.scattering_radius,
                1,
                p.l,
                p.awr,
                0.0,
                vec![res(p.energy, p.j, p.gamma_n, p.gamma_g)],
            ),
        )
    }

    /// Synthetic single-resonance for an arbitrary `(z, a, awr, energy)`.
    /// Hard-codes RM, AP=5, I=0, L=0, J=0.5, Γn=1e-3, Γγ=1e-2.  Used by
    /// multi-isotope group-fit / calibration tests.
    pub fn synthetic_single_resonance(z: u32, a: u32, awr: f64, energy: f64) -> ResonanceData {
        wrap(
            z,
            a,
            awr,
            make_range(
                1e-5,
                1e4,
                ResonanceFormalism::ReichMoore,
                0.0,
                5.0,
                1,
                0,
                awr,
                0.0,
                vec![res(energy, 0.5, 1e-3, 1e-2)],
            ),
        )
    }

    /// U-238-ZA single s-wave SLBW over the wider `1e-5 .. 1e6` eV range used
    /// by the elastic-oracle regression test (`slbw_elastic_oracle.rs`).
    /// I=0 so `g_J = 1` for J=1/2.
    pub fn synthetic_swave_slbw(
        awr: f64,
        e_r_ev: f64,
        gn_ev: f64,
        gg_ev: f64,
        scattering_radius_fm: f64,
    ) -> ResonanceData {
        wrap(
            92,
            238,
            awr,
            make_range(
                1e-5,
                1e6,
                ResonanceFormalism::SLBW,
                0.0,
                scattering_radius_fm,
                1,
                0,
                awr,
                0.0,
                vec![res(e_r_ev, 0.5, gn_ev, gg_ev)],
            ),
        )
    }

    /// Minimal single-resonance for offline detectability tests.  Auto-derives
    /// `awr ≈ a - 0.009` (rough neutron-mass correction); hard-codes RM, AP=6,
    /// I=0, L=0, J=0.5.
    pub fn synthetic_isotope(z: u32, a: u32, res_energy: f64, gn: f64, gg: f64) -> ResonanceData {
        let awr = a as f64 - 0.009;
        wrap(
            z,
            a,
            awr,
            make_range(
                1e-5,
                1e4,
                ResonanceFormalism::ReichMoore,
                0.0,
                6.0,
                1,
                0,
                awr,
                0.0,
                vec![res(res_energy, 0.5, gn, gg)],
            ),
        )
    }

    /// Multi-resonance sibling of [`synthetic_isotope`]: N s-wave resonances
    /// `(energy_eV, Γn_eV, Γγ_eV)` in **one** L-group of a **single** range —
    /// i.e. ONE potential-scattering term.  Deliberately NOT built by stacking
    /// N `synthetic_isotope` single-resonance isotopes in a sample: each such
    /// isotope carries its own AP hard-sphere background, so a stack N-folds
    /// the potential-scattering baseline.  Same structural defaults as
    /// [`synthetic_isotope`] (RM, AP=6, I=0, L=0, J=0.5, `awr ≈ a − 0.009`).
    pub fn synthetic_isotope_multi(
        z: u32,
        a: u32,
        resonances: &[(f64, f64, f64)],
    ) -> ResonanceData {
        let awr = a as f64 - 0.009;
        wrap(
            z,
            a,
            awr,
            make_range(
                1e-5,
                1e4,
                ResonanceFormalism::ReichMoore,
                0.0,
                6.0,
                1,
                0,
                awr,
                0.0,
                resonances
                    .iter()
                    .map(|&(e, gn, gg)| res(e, 0.5, gn, gg))
                    .collect(),
            ),
        )
    }

    /// Hf-178 MLBW: two s-waves at 7.8 and 16.9 eV in the same J=1/2 group.
    /// Range `0 .. 100` eV, AP=9.48, NAPS=0.  MLBW positivity and
    /// total-vs-components regression tests in `slbw.rs`.
    pub fn hf178_mlbw_two_resonances() -> ResonanceData {
        wrap(
            72,
            178,
            177.94,
            make_range(
                0.0,
                100.0,
                ResonanceFormalism::MLBW,
                0.0,
                9.48,
                0,
                0,
                177.94,
                0.0,
                vec![res(7.8, 0.5, 0.002, 0.060), res(16.9, 0.5, 0.004, 0.055)],
            ),
        )
    }

    /// Hf-177 MLBW: two s-waves at 2.386 and 5.89 eV in the same high-J group
    /// (J=4.0), target spin I=3.5.  Range `1e-5 .. 1e3` eV, AP=7.0, NAPS=0.
    /// MLBW coherent-vs-incoherent dispatcher regression (PR #465 root cause).
    pub fn hf177_mlbw_two_resonances_high_j() -> ResonanceData {
        wrap(
            72,
            177,
            175.4232,
            make_range(
                1e-5,
                1e3,
                ResonanceFormalism::MLBW,
                3.5,
                7.0,
                0,
                0,
                175.4232,
                0.0,
                vec![
                    res(2.386, 4.0, 2.0e-3, 60.0e-3),
                    res(5.89, 4.0, 3.5e-3, 62.0e-3),
                ],
            ),
        )
    }

    /// SAMMY ex001 hydrogen-anchor: SLBW single resonance at 10 eV on the
    /// synthetic ZA=1010 (AWR=10).  Doppler-broadening reference suite.
    /// Widths are in eV (SAMMY par file has them in meV; conversion baked in).
    pub fn ex001_hydrogen_single_resonance() -> ResonanceData {
        wrap(
            1,
            10,
            10.0,
            make_range(
                0.0,
                100.0,
                ResonanceFormalism::SLBW,
                0.0,
                2.908,
                1,
                0,
                10.0,
                2.908,
                vec![res(10.0, 0.5, 0.5e-3, 1.0e-3)],
            ),
        )
    }

    /// Minimal SLBW `ResonanceRange` (not a full `ResonanceData`) using
    /// U-238-like parameters with a single 6.674 eV s-wave.  For
    /// `slbw_cross_sections_for_range` panic tests at the range-level entry.
    pub fn minimal_slbw_range() -> ResonanceRange {
        make_range(
            1e-5,
            1e4,
            ResonanceFormalism::SLBW,
            0.0,
            9.4285,
            1,
            0,
            236.006,
            0.0,
            vec![res(6.674, 0.5, 1.493e-3, 23.0e-3)],
        )
    }
}

#[cfg(test)]
mod test_support_tests {
    use super::ResonanceFormalism;
    use super::test_support::*;

    #[test]
    fn u238_single_resonance_has_canonical_za_and_energy() {
        let d = u238_single_resonance();
        assert_eq!(d.za, 92238);
        assert_eq!(d.ranges[0].l_groups[0].resonances[0].energy, 6.674);
    }

    #[test]
    fn u238_with_formalism_slbw_returns_slbw() {
        let d = u238_with_formalism(ResonanceFormalism::SLBW);
        assert_eq!(d.ranges[0].formalism, ResonanceFormalism::SLBW);
        assert_eq!(d.ranges[0].energy_low, 1e-5);
        assert_eq!(d.ranges[0].energy_high, 1e4);
    }

    #[test]
    fn u238_with_formalism_wide_range_uses_wide_bounds() {
        let d = u238_with_formalism_wide_range(ResonanceFormalism::MLBW);
        assert_eq!(d.ranges[0].energy_low, 1e-6);
        assert_eq!(d.ranges[0].energy_high, 1e5);
        assert_eq!(d.ranges[0].formalism, ResonanceFormalism::MLBW);
    }

    #[test]
    fn single_resonance_param_struct_builds_rm() {
        let d = single_resonance(SingleResonanceParams {
            energy: 6.674,
            gamma_n: 1.493e-3,
            gamma_g: 23.0e-3,
            j: 0.5,
            l: 0,
            awr: 236.006,
            target_spin: 0.0,
            scattering_radius: 9.4285,
        });
        assert_eq!(d.ranges[0].formalism, ResonanceFormalism::ReichMoore);
        assert_eq!(d.za, 92238);
    }

    #[test]
    fn hf178_mlbw_two_resonances_returns_two() {
        let d = hf178_mlbw_two_resonances();
        assert_eq!(d.ranges[0].l_groups[0].resonances.len(), 2);
        assert_eq!(d.za, 72178);
    }

    #[test]
    fn synthetic_isotope_uses_caller_za() {
        let d = synthetic_isotope(74, 184, 10.0, 1e-3, 1e-2);
        assert_eq!(d.za, 74184);
        assert_eq!(d.ranges[0].l_groups[0].resonances[0].energy, 10.0);
    }

    #[test]
    fn synthetic_isotope_multi_puts_all_resonances_in_one_group() {
        // One range, one L-group, one potential-scattering term — NOT N
        // stacked single-resonance isotopes (which would N-fold the AP
        // background).
        let d = synthetic_isotope_multi(
            73,
            181,
            &[
                (10.36, 0.003, 0.058),
                (24.0, 0.009, 0.060),
                (39.1, 0.040, 0.060),
            ],
        );
        assert_eq!(d.za, 73181);
        assert_eq!(d.ranges.len(), 1);
        assert_eq!(d.ranges[0].l_groups.len(), 1);
        let rs = &d.ranges[0].l_groups[0].resonances;
        assert_eq!(rs.len(), 3);
        assert_eq!(rs[0].energy, 10.36);
        assert_eq!(rs[2].gn, 0.040);
        // Structural defaults match synthetic_isotope (same awr law, RM, J=0.5).
        let single = synthetic_isotope(73, 181, 10.36, 0.003, 0.058);
        assert_eq!(d.awr, single.awr);
        assert_eq!(d.ranges[0].formalism, single.ranges[0].formalism);
        assert_eq!(rs[0].j, single.ranges[0].l_groups[0].resonances[0].j);
    }
}
