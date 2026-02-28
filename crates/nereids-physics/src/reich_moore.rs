//! Multi-formalism cross-section dispatcher.
//!
//! `cross_sections_at_energy` is the primary entry point for computing
//! energy-dependent cross-sections from ENDF resonance data.  It
//! iterates over all resonance ranges in the data and dispatches each to
//! the appropriate formalism-specific calculator:
//!
//! | ENDF LRF | Formalism                  | Implemented as              |
//! |----------|----------------------------|-----------------------------|
//! | 1        | SLBW                       | `slbw::slbw_cross_sections_for_range` |
//! | 2        | MLBW (approx.)             | `slbw::slbw_cross_sections_for_range` (SLBW approximation; resonance interference ignored) |
//! | 3        | Reich-Moore                | `reich_moore_spin_group` (this module) |
//! | 7        | R-Matrix Limited           | `rmatrix_limited::cross_sections_for_rml_range` |
//! | URR      | Hauser-Feshbach average    | `urr::urr_cross_sections` |
//!
//! ## Reich-Moore Approximation
//! In the full R-matrix, all channels (neutron, capture, fission) appear
//! explicitly. The Reich-Moore approximation *eliminates the capture channel*
//! from the channel space, absorbing its effect into an imaginary part of
//! the energy denominator. This makes the level matrix smaller while
//! remaining highly accurate.
//!
//! For non-fissile isotopes (like U-238 below threshold), each spin group
//! has only ONE explicit channel (neutron elastic), making the R-matrix
//! a scalar — and the calculation is very efficient.
//!
//! ## SAMMY Reference
//! - `rml/mrml07.f` Setr: R-matrix construction
//! - `rml/mrml09.f` Yinvrs: level matrix inversion
//! - `rml/mrml11.f` Setxqx: X-matrix, Sectio: cross-sections
//! - `rml/mrml03.f` Betset: ENDF widths → reduced width amplitudes
//! - SAMMY manual Section 2.1 (R-matrix theory)

use num_complex::Complex64;

use nereids_core::constants::{DIVISION_FLOOR, LOG_FLOOR, PIVOT_FLOOR, QUANTUM_NUMBER_EPS};
use nereids_endf::resonance::{ResonanceData, ResonanceFormalism, ResonanceRange, Tab1};

use crate::channel;
use crate::penetrability;
use crate::rmatrix_limited;
use crate::slbw;
use crate::urr;

// ─── Per-resonance precomputed invariants ─────────────────────────────────────
//
// These quantities depend only on the resonance parameters and the channel
// radius at the resonance energy — both are energy-independent constants.
// Pre-computing them once (outside the energy loop) eliminates redundant
// `penetrability(l, rho_r)` and `group_by_j()` calls per energy point.
//
// Issue #87: "Perf: Pre-cache J-groups and per-resonance quantities"

/// Per-resonance invariants for the single-channel (non-fissile) Reich-Moore path.
///
/// Pre-computed once per resonance before the energy sweep.
/// Reference: SAMMY `rml/mrml03.f` Betset (lines 240-276)
struct PrecomputedResonanceSingle {
    /// Resonance energy E_r (eV).
    energy: f64,
    /// Capture width Γ_γ (eV).
    gamma_g: f64,
    /// Reduced width amplitude squared γ²_n = |Γ_n| / (2·P_l(E_r)).
    gamma_n_reduced_sq: f64,
}

/// Per-resonance invariants for the 2-channel (one fission) Reich-Moore path.
struct PrecomputedResonance2ch {
    /// Resonance energy E_r (eV).
    energy: f64,
    /// Capture width Γ_γ (eV).
    gamma_g: f64,
    /// Reduced width amplitude β_n = sign(Γ_n) × √(|Γ_n| / (2·P_l(E_r))).
    beta_n: f64,
    /// Fission width amplitude β_f = sign(Γ_f) × √(|Γ_f| / 2).
    beta_f: f64,
}

/// Per-resonance invariants for the 3-channel (two fission) Reich-Moore path.
struct PrecomputedResonance3ch {
    /// Resonance energy E_r (eV).
    energy: f64,
    /// Capture width Γ_γ (eV).
    gamma_g: f64,
    /// Reduced width amplitude β_n.
    beta_n: f64,
    /// Fission width amplitude β_fa.
    beta_fa: f64,
    /// Fission width amplitude β_fb.
    beta_fb: f64,
}

/// Pre-computed J-group for the single-channel Reich-Moore path.
///
/// Groups resonances by total angular momentum J, with per-resonance
/// invariants already computed. The J-grouping depends only on the
/// resonance data, not on the incident energy, so it is computed once
/// and reused for every energy point.
struct PrecomputedJGroupSingle {
    /// Statistical weight g_J = (2J+1) / ((2I+1)(2s+1)).
    g_j: f64,
    /// Pre-computed per-resonance quantities for this J-group.
    resonances: Vec<PrecomputedResonanceSingle>,
}

/// Pre-computed J-group for the 2-channel fission path.
struct PrecomputedJGroup2ch {
    g_j: f64,
    resonances: Vec<PrecomputedResonance2ch>,
}

/// Pre-computed J-group for the 3-channel fission path.
struct PrecomputedJGroup3ch {
    g_j: f64,
    resonances: Vec<PrecomputedResonance3ch>,
}

/// Compute penetrability at the resonance energy P_l(ρ_r).
///
/// ENDF widths are defined as Γ_n = 2·P_l(AP(E_r), E_r)·γ²_n,
/// so the penetrability must be evaluated at the resonance energy
/// using the channel radius AP(E_r) — not the incident-energy AP(E).
///
/// This function is the core quantity that Issue #87 caches: previously
/// it was recomputed for every resonance at every energy point.
///
/// When E_r ≈ 0, the penetrability is zero (matching SLBW behavior in
/// `slbw.rs`). This ensures the result depends only on resonance
/// parameters and is independent of the incident energy, enabling the
/// precompute to be hoisted above the energy loop.
fn penetrability_at_resonance(
    e_r: f64,
    l: u32,
    awr: f64,
    channel_radius: f64,
    ap_table: Option<&Tab1>,
) -> f64 {
    if e_r.abs() > PIVOT_FLOOR {
        let radius_at_er = ap_table.map_or(channel_radius, |t| t.evaluate(e_r.abs()));
        let rho_r = channel::rho(e_r.abs(), awr, radius_at_er);
        penetrability::penetrability(l, rho_r)
    } else {
        // E_r ≈ 0 → P_l(0) = 0, so γ²_n = 0 regardless.
        // Using 0.0 keeps this function energy-independent.
        0.0
    }
}

/// Build pre-computed J-groups for the single-channel (non-fissile) path.
///
/// Groups resonances by J, pre-computes γ²_n per resonance.
/// All quantities depend only on resonance parameters (not incident energy),
/// so the result can be computed once and reused across all energy points.
fn precompute_jgroups_single(
    resonances: &[nereids_endf::resonance::Resonance],
    l: u32,
    awr: f64,
    channel_radius: f64,
    ap_table: Option<&Tab1>,
    target_spin: f64,
) -> Vec<PrecomputedJGroupSingle> {
    // Group by J (same logic as `group_by_j` but builds PrecomputedResonanceSingle directly).
    let mut j_values: Vec<f64> = Vec::new();
    let mut groups: Vec<PrecomputedJGroupSingle> = Vec::new();

    for res in resonances {
        let j = res.j;

        // Precompute per-resonance invariants.
        let p_at_er = penetrability_at_resonance(res.energy, l, awr, channel_radius, ap_table);
        let gamma_n_reduced_sq = if p_at_er > PIVOT_FLOOR {
            res.gn.abs() / (2.0 * p_at_er)
        } else {
            0.0
        };
        let precomp = PrecomputedResonanceSingle {
            energy: res.energy,
            gamma_g: res.gg,
            gamma_n_reduced_sq,
        };

        // Find or create J-group.
        if let Some(idx) = j_values
            .iter()
            .position(|&gj| (gj - j).abs() < QUANTUM_NUMBER_EPS)
        {
            groups[idx].resonances.push(precomp);
        } else {
            j_values.push(j);
            groups.push(PrecomputedJGroupSingle {
                g_j: channel::statistical_weight(j, target_spin),
                resonances: vec![precomp],
            });
        }
    }
    groups
}

/// Build pre-computed J-groups for the 2-channel fission path.
///
/// All quantities depend only on resonance parameters (not incident energy),
/// so the result can be computed once and reused across all energy points.
fn precompute_jgroups_2ch(
    resonances: &[nereids_endf::resonance::Resonance],
    l: u32,
    awr: f64,
    channel_radius: f64,
    ap_table: Option<&Tab1>,
    target_spin: f64,
) -> Vec<PrecomputedJGroup2ch> {
    let mut j_values: Vec<f64> = Vec::new();
    let mut groups: Vec<PrecomputedJGroup2ch> = Vec::new();

    for res in resonances {
        let j = res.j;

        let p_at_er = penetrability_at_resonance(res.energy, l, awr, channel_radius, ap_table);

        let beta_n = if p_at_er > PIVOT_FLOOR {
            let sign = if res.gn >= 0.0 { 1.0 } else { -1.0 };
            sign * (res.gn.abs() / (2.0 * p_at_er)).sqrt()
        } else {
            0.0
        };

        let beta_f = {
            let sign = if res.gfa >= 0.0 { 1.0 } else { -1.0 };
            sign * (res.gfa.abs() / 2.0).sqrt()
        };

        let precomp = PrecomputedResonance2ch {
            energy: res.energy,
            gamma_g: res.gg,
            beta_n,
            beta_f,
        };

        if let Some(idx) = j_values
            .iter()
            .position(|&gj| (gj - j).abs() < QUANTUM_NUMBER_EPS)
        {
            groups[idx].resonances.push(precomp);
        } else {
            j_values.push(j);
            groups.push(PrecomputedJGroup2ch {
                g_j: channel::statistical_weight(j, target_spin),
                resonances: vec![precomp],
            });
        }
    }
    groups
}

/// Build pre-computed J-groups for the 3-channel fission path.
///
/// All quantities depend only on resonance parameters (not incident energy),
/// so the result can be computed once and reused across all energy points.
fn precompute_jgroups_3ch(
    resonances: &[nereids_endf::resonance::Resonance],
    l: u32,
    awr: f64,
    channel_radius: f64,
    ap_table: Option<&Tab1>,
    target_spin: f64,
) -> Vec<PrecomputedJGroup3ch> {
    let mut j_values: Vec<f64> = Vec::new();
    let mut groups: Vec<PrecomputedJGroup3ch> = Vec::new();

    for res in resonances {
        let j = res.j;

        let p_at_er = penetrability_at_resonance(res.energy, l, awr, channel_radius, ap_table);

        let beta_n = if p_at_er > PIVOT_FLOOR {
            let sign = if res.gn >= 0.0 { 1.0 } else { -1.0 };
            sign * (res.gn.abs() / (2.0 * p_at_er)).sqrt()
        } else {
            0.0
        };

        let beta_fa = {
            let sign = if res.gfa >= 0.0 { 1.0 } else { -1.0 };
            sign * (res.gfa.abs() / 2.0).sqrt()
        };

        let beta_fb = {
            let sign = if res.gfb >= 0.0 { 1.0 } else { -1.0 };
            sign * (res.gfb.abs() / 2.0).sqrt()
        };

        let precomp = PrecomputedResonance3ch {
            energy: res.energy,
            gamma_g: res.gg,
            beta_n,
            beta_fa,
            beta_fb,
        };

        if let Some(idx) = j_values
            .iter()
            .position(|&gj| (gj - j).abs() < QUANTUM_NUMBER_EPS)
        {
            groups[idx].resonances.push(precomp);
        } else {
            j_values.push(j);
            groups.push(PrecomputedJGroup3ch {
                g_j: channel::statistical_weight(j, target_spin),
                resonances: vec![precomp],
            });
        }
    }
    groups
}

/// Cross-section results at a single energy point.
#[derive(Debug, Clone, Copy)]
pub struct CrossSections {
    /// Total cross-section (barns).
    pub total: f64,
    /// Elastic scattering cross-section (barns).
    pub elastic: f64,
    /// Capture (n,γ) cross-section (barns).
    pub capture: f64,
    /// Fission cross-section (barns).
    pub fission: f64,
}

/// Compute cross-sections at a single energy.
///
/// Dispatches each resonance range to the appropriate formalism-specific
/// calculator (SLBW, MLBW, Reich-Moore, R-Matrix Limited, URR) based on the
/// formalism stored in that range.  See the module-level table for the full
/// dispatch map.
///
/// Adjacent ranges that share a boundary energy use half-open intervals
/// `[e_low, e_high)` so the boundary point is counted exactly once
/// (ENDF-6 §2 convention).
///
/// # Arguments
/// * `data` — Parsed resonance parameters from ENDF.
/// * `energy_ev` — Neutron energy in eV (lab frame).
///
/// # Returns
/// Cross-sections in barns.
pub fn cross_sections_at_energy(data: &ResonanceData, energy_ev: f64) -> CrossSections {
    let awr = data.awr;

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for (range_idx, range) in data.ranges.iter().enumerate() {
        // Use half-open [low, high) only when the *next* range begins exactly at
        // this range's upper endpoint AND that next range can actually produce
        // cross-sections.  Ranges that parse successfully but whose physics is
        // not yet wired up (URR with urr=None) must not steal the boundary —
        // otherwise the shared energy point falls into a gap.
        // ENDF-6 §2 — adjacent ranges share a single boundary energy.
        let next_starts_here = data
            .ranges
            .get(range_idx + 1)
            .is_some_and(|next| next.energy_low == range.energy_high && range_is_evaluable(next));
        let in_range = if next_starts_here {
            energy_ev >= range.energy_low && energy_ev < range.energy_high
        } else {
            energy_ev >= range.energy_low && energy_ev <= range.energy_high
        };
        if !in_range {
            continue;
        }

        // URR (LRU=2): Hauser-Feshbach average cross-sections.
        // These ranges have `resolved = false` so they must be dispatched before
        // the `!range.resolved` skip below.
        //
        // Note: `parse_urr_range` sets urr.e_low == range.energy_low and
        // urr.e_high == range.energy_high, so the outer `in_range` check and the
        // inner band guard in `urr_cross_sections` test the same interval.
        // The inner guard is kept as a safety net for direct calls.
        if let Some(urr_data) = &range.urr {
            debug_assert_eq!(
                urr_data.e_low, range.energy_low,
                "URR e_low must equal range.energy_low"
            );
            debug_assert_eq!(
                urr_data.e_high, range.energy_high,
                "URR e_high must equal range.energy_high"
            );
            let ap_fm = range.scattering_radius_at(energy_ev);
            let (t, e, c, f) = urr::urr_cross_sections(urr_data, energy_ev, ap_fm);
            total += t;
            elastic += e;
            capture += c;
            fission += f;
            continue;
        }

        if !range.resolved {
            continue;
        }

        // Each range carries its own target_spin — pass per-range, not
        // from the first range, to correctly compute statistical weights g_J.
        let (t, e, c, f) = cross_sections_for_range(range, energy_ev, awr, range.target_spin);
        total += t;
        elastic += e;
        capture += c;
        fission += f;
    }

    CrossSections {
        total,
        elastic,
        capture,
        fission,
    }
}

/// Compute cross-sections over a grid of energies.
///
/// Optimized batch evaluation: precomputes J-groups and per-resonance
/// invariants (reduced width amplitudes, penetrability at E_r) once per
/// resonance range, then evaluates each energy point using the cached data.
/// This avoids redundant `group_by_j` + `penetrability(l, rho_r)` calls
/// that the per-point API (`cross_sections_at_energy`) would repeat.
///
/// Issue #87: the precompute is hoisted above the energy loop so that
/// `precompute_jgroups_*` runs O(ranges) times total, not O(ranges × energies).
///
/// # Arguments
/// * `data` — Parsed resonance parameters from ENDF.
/// * `energies` — Slice of neutron energies in eV.
///
/// # Returns
/// Vector of cross-sections, one per energy point.
pub fn cross_sections_on_grid(data: &ResonanceData, energies: &[f64]) -> Vec<CrossSections> {
    if energies.is_empty() {
        return Vec::new();
    }

    let awr = data.awr;

    // Phase 1: precompute per-range data (J-groups, reduced widths, etc.).
    // This runs once for the entire grid, not per energy point.
    let precomputed: Vec<PrecomputedRangeData> = data
        .ranges
        .iter()
        .enumerate()
        .map(|(range_idx, range)| precompute_range_data(range, range_idx, data, awr))
        .collect();

    // Phase 2: evaluate each energy point using precomputed data.
    energies
        .iter()
        .map(|&energy_ev| {
            let mut total = 0.0;
            let mut elastic = 0.0;
            let mut capture = 0.0;
            let mut fission = 0.0;

            for pc in &precomputed {
                let in_range = if pc.half_open_upper {
                    energy_ev >= pc.energy_low && energy_ev < pc.energy_high
                } else {
                    energy_ev >= pc.energy_low && energy_ev <= pc.energy_high
                };
                if !in_range {
                    continue;
                }

                let (t, e, c, f) = evaluate_precomputed_range(pc, energy_ev, awr);
                total += t;
                elastic += e;
                capture += c;
                fission += f;
            }

            CrossSections {
                total,
                elastic,
                capture,
                fission,
            }
        })
        .collect()
}

// ─── Precomputed range data for batch grid evaluation ────────────────────────
//
// These types hold energy-independent invariants for a single resonance range,
// precomputed once by `precompute_range_data` and reused for every energy point
// in `cross_sections_on_grid`.

/// Precomputed data for a single L-group within a Reich-Moore range.
///
/// Holds the J-group cache and metadata needed to compute energy-dependent
/// channel parameters (rho, P_l, S_l, phi_l) at each energy point.
enum PrecomputedRmLGroupData {
    /// Non-fissile: single neutron channel, capture eliminated.
    Single {
        l: u32,
        awr_l: f64,
        /// L-group override radius (fm). 0.0 means use range radius.
        apl: f64,
        jgroups: Vec<PrecomputedJGroupSingle>,
    },
    /// One fission channel (gfa != 0, gfb == 0).
    TwoCh {
        l: u32,
        awr_l: f64,
        apl: f64,
        jgroups: Vec<PrecomputedJGroup2ch>,
    },
    /// Two fission channels (both gfa and gfb != 0).
    ThreeCh {
        l: u32,
        awr_l: f64,
        apl: f64,
        jgroups: Vec<PrecomputedJGroup3ch>,
    },
}

/// Precomputed data for a single SLBW L-group.
struct PrecomputedSlbwLGroupData {
    l: u32,
    awr_l: f64,
    /// L-group override radius (fm). 0.0 means use range radius.
    apl: f64,
    jgroups: Vec<slbw::PrecomputedSlbwJGroup>,
}

/// Precomputed data for a single resonance range.
///
/// Wraps the formalism-specific precomputed L-group data plus the
/// energy interval metadata needed for range dispatch.
struct PrecomputedRangeData<'a> {
    energy_low: f64,
    energy_high: f64,
    half_open_upper: bool,
    kind: PrecomputedRangeKind<'a>,
}

/// Formalism-specific precomputed data for a range.
enum PrecomputedRangeKind<'a> {
    /// Reich-Moore (LRF=3): precomputed J-groups per L-group.
    /// The range reference is kept for `scattering_radius_at(energy_ev)`.
    ReichMoore {
        range: &'a ResonanceRange,
        l_groups: Vec<PrecomputedRmLGroupData>,
    },
    /// SLBW/MLBW (LRF=1,2): precomputed J-groups per L-group.
    /// The range reference is kept for `scattering_radius_at(energy_ev)`.
    Slbw {
        range: &'a ResonanceRange,
        l_groups: Vec<PrecomputedSlbwLGroupData>,
    },
    /// R-Matrix Limited (LRF=7): no precompute, evaluate per-energy.
    RMatrixLimited(&'a nereids_endf::resonance::RmlData),
    /// URR: no precompute, evaluate per-energy.
    Urr {
        urr_data: &'a nereids_endf::resonance::UrrData,
        range: &'a ResonanceRange,
    },
    /// Not evaluable (skip).
    Skip,
}

/// Build precomputed range data for a single resonance range.
///
/// This extracts all energy-independent quantities (J-group structure,
/// reduced width amplitudes, penetrability at resonance energies) so they
/// can be reused across all energy points without redundant computation.
fn precompute_range_data<'a>(
    range: &'a ResonanceRange,
    range_idx: usize,
    data: &'a ResonanceData,
    awr: f64,
) -> PrecomputedRangeData<'a> {
    // Half-open upper bound logic (same as cross_sections_at_energy).
    let next_starts_here = data
        .ranges
        .get(range_idx + 1)
        .is_some_and(|next| next.energy_low == range.energy_high && range_is_evaluable(next));

    let make = |kind| PrecomputedRangeData {
        energy_low: range.energy_low,
        energy_high: range.energy_high,
        half_open_upper: next_starts_here,
        kind,
    };

    // URR ranges.
    if let Some(urr_data) = &range.urr {
        return make(PrecomputedRangeKind::Urr { urr_data, range });
    }

    if !range.resolved {
        return make(PrecomputedRangeKind::Skip);
    }

    // RML ranges.
    if let Some(rml) = &range.rml {
        return make(PrecomputedRangeKind::RMatrixLimited(rml));
    }

    // SLBW/MLBW ranges: precompute J-groups per L-group.
    if matches!(
        range.formalism,
        ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
    ) {
        let l_groups: Vec<PrecomputedSlbwLGroupData> = range
            .l_groups
            .iter()
            .map(|l_group| {
                let l = l_group.l;
                let awr_l = if l_group.awr > 0.0 { l_group.awr } else { awr };
                let jgroups = slbw::precompute_slbw_jgroups(
                    &l_group.resonances,
                    l,
                    awr_l,
                    range,
                    l_group,
                    range.target_spin,
                );
                PrecomputedSlbwLGroupData {
                    l,
                    awr_l,
                    apl: l_group.apl,
                    jgroups,
                }
            })
            .collect();
        return make(PrecomputedRangeKind::Slbw { range, l_groups });
    }

    // Reich-Moore ranges: precompute J-groups per L-group.
    if range.formalism == ResonanceFormalism::ReichMoore {
        let rm_l_groups: Vec<PrecomputedRmLGroupData> = range
            .l_groups
            .iter()
            .map(|l_group| {
                let l = l_group.l;
                let awr_l = if l_group.awr > 0.0 { l_group.awr } else { awr };

                // Channel radius for precompute: when APL > 0, use it; otherwise
                // use the constant scattering_radius. The ap_table (NRO=1) case
                // is handled inside penetrability_at_resonance, which evaluates
                // the table at E_r for each resonance.
                let channel_radius = if l_group.apl > 0.0 {
                    l_group.apl
                } else {
                    range.scattering_radius
                };
                let ap_table_ref: Option<&Tab1> = if l_group.apl > 0.0 {
                    None
                } else {
                    range.ap_table.as_ref()
                };

                let has_fission = l_group
                    .resonances
                    .iter()
                    .any(|r| r.gfa.abs() > PIVOT_FLOOR || r.gfb.abs() > PIVOT_FLOOR);
                let has_two_fission = l_group.resonances.iter().any(|r| r.gfb.abs() > PIVOT_FLOOR);

                if !has_fission {
                    let jgroups = precompute_jgroups_single(
                        &l_group.resonances,
                        l,
                        awr_l,
                        channel_radius,
                        ap_table_ref,
                        range.target_spin,
                    );
                    PrecomputedRmLGroupData::Single {
                        l,
                        awr_l,
                        apl: l_group.apl,
                        jgroups,
                    }
                } else if !has_two_fission {
                    let jgroups = precompute_jgroups_2ch(
                        &l_group.resonances,
                        l,
                        awr_l,
                        channel_radius,
                        ap_table_ref,
                        range.target_spin,
                    );
                    PrecomputedRmLGroupData::TwoCh {
                        l,
                        awr_l,
                        apl: l_group.apl,
                        jgroups,
                    }
                } else {
                    let jgroups = precompute_jgroups_3ch(
                        &l_group.resonances,
                        l,
                        awr_l,
                        channel_radius,
                        ap_table_ref,
                        range.target_spin,
                    );
                    PrecomputedRmLGroupData::ThreeCh {
                        l,
                        awr_l,
                        apl: l_group.apl,
                        jgroups,
                    }
                }
            })
            .collect();
        return make(PrecomputedRangeKind::ReichMoore {
            range,
            l_groups: rm_l_groups,
        });
    }

    // Unrecognized formalism: skip.
    make(PrecomputedRangeKind::Skip)
}

/// Evaluate cross-sections for a precomputed range at a single energy.
///
/// Uses the cached J-groups and per-resonance invariants to avoid
/// redundant precomputation. Only energy-dependent quantities (rho,
/// P_l, S_l, phi_l, pi/k^2) are computed per call.
fn evaluate_precomputed_range(
    pc: &PrecomputedRangeData,
    energy_ev: f64,
    awr: f64,
) -> (f64, f64, f64, f64) {
    match &pc.kind {
        PrecomputedRangeKind::Skip => (0.0, 0.0, 0.0, 0.0),

        PrecomputedRangeKind::RMatrixLimited(rml) => {
            rmatrix_limited::cross_sections_for_rml_range(rml, energy_ev)
        }

        PrecomputedRangeKind::Urr { urr_data, range } => {
            let ap_fm = range.scattering_radius_at(energy_ev);
            urr::urr_cross_sections(urr_data, energy_ev, ap_fm)
        }

        PrecomputedRangeKind::Slbw { range, l_groups } => {
            let pi_over_k2 = channel::pi_over_k_squared_barns(energy_ev, awr);
            let mut total = 0.0;
            let mut elastic = 0.0;
            let mut capture = 0.0;
            let mut fission = 0.0;

            for lg in l_groups {
                // Channel radius at incident energy: use APL if set,
                // otherwise the range's (possibly energy-dependent) radius.
                let channel_radius = if lg.apl > 0.0 {
                    lg.apl
                } else {
                    range.scattering_radius_at(energy_ev)
                };

                let rho = channel::rho(energy_ev, lg.awr_l, channel_radius);
                let phi = penetrability::phase_shift(lg.l, rho);
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                let sin2_phi = sin_phi * sin_phi;
                let p_at_e = penetrability::penetrability(lg.l, rho);

                let (t, e, c, f) = slbw::slbw_evaluate_with_cached_jgroups(
                    &lg.jgroups,
                    energy_ev,
                    pi_over_k2,
                    p_at_e,
                    sin_phi,
                    cos_phi,
                    sin2_phi,
                );
                total += t;
                elastic += e;
                capture += c;
                fission += f;
            }

            (total, elastic, capture, fission)
        }

        PrecomputedRangeKind::ReichMoore { range, l_groups } => {
            let mut total = 0.0;
            let mut elastic = 0.0;
            let mut capture = 0.0;
            let mut fission = 0.0;

            for lg in l_groups {
                let (l, awr_l, apl) = match lg {
                    PrecomputedRmLGroupData::Single { l, awr_l, apl, .. } => (*l, *awr_l, *apl),
                    PrecomputedRmLGroupData::TwoCh { l, awr_l, apl, .. } => (*l, *awr_l, *apl),
                    PrecomputedRmLGroupData::ThreeCh { l, awr_l, apl, .. } => (*l, *awr_l, *apl),
                };

                // Channel radius at incident energy: use APL if set,
                // otherwise the range's (possibly energy-dependent) radius.
                let channel_radius = if apl > 0.0 {
                    apl
                } else {
                    range.scattering_radius_at(energy_ev)
                };

                let rho = channel::rho(energy_ev, awr_l, channel_radius);
                let p_l = penetrability::penetrability(l, rho);
                let s_l = penetrability::shift_factor(l, rho);
                let phi_l = penetrability::phase_shift(l, rho);

                let (t, e, c, f) = match lg {
                    PrecomputedRmLGroupData::Single { jgroups, .. } => {
                        let mut t = 0.0;
                        let mut e = 0.0;
                        let mut c = 0.0;
                        let mut f = 0.0;
                        for jg in jgroups {
                            let (jt, je, jc, jf) = reich_moore_spin_group_precomputed(
                                &jg.resonances,
                                energy_ev,
                                awr_l,
                                jg.g_j,
                                p_l,
                                s_l,
                                phi_l,
                            );
                            t += jt;
                            e += je;
                            c += jc;
                            f += jf;
                        }
                        (t, e, c, f)
                    }
                    PrecomputedRmLGroupData::TwoCh { jgroups, .. } => {
                        let mut t = 0.0;
                        let mut e = 0.0;
                        let mut c = 0.0;
                        let mut f = 0.0;
                        for jg in jgroups {
                            let (jt, je, jc, jf) = reich_moore_2ch_precomputed(
                                &jg.resonances,
                                energy_ev,
                                awr_l,
                                jg.g_j,
                                p_l,
                                s_l,
                                phi_l,
                            );
                            t += jt;
                            e += je;
                            c += jc;
                            f += jf;
                        }
                        (t, e, c, f)
                    }
                    PrecomputedRmLGroupData::ThreeCh { jgroups, .. } => {
                        let mut t = 0.0;
                        let mut e = 0.0;
                        let mut c = 0.0;
                        let mut f = 0.0;
                        for jg in jgroups {
                            let (jt, je, jc, jf) = reich_moore_3ch_precomputed(
                                &jg.resonances,
                                energy_ev,
                                awr_l,
                                jg.g_j,
                                p_l,
                                s_l,
                                phi_l,
                            );
                            t += jt;
                            e += je;
                            c += jc;
                            f += jf;
                        }
                        (t, e, c, f)
                    }
                };

                total += t;
                elastic += e;
                capture += c;
                fission += f;
            }

            (total, elastic, capture, fission)
        }
    }
}

/// Can this range actually produce non-zero cross-sections?
///
/// Returns `true` for formalisms whose physics evaluation is implemented:
/// - SLBW (LRF=1) and MLBW (LRF=2) resolved ranges
/// - Reich-Moore (LRF=3) resolved ranges
/// - R-Matrix Limited (LRF=7) resolved ranges
/// - URR ranges with parsed data (`urr.is_some()`)
///
/// Returns `false` for URR placeholders created when unsupported INT
/// codes force a skip, or other unrecognized formalisms.
///
/// **Keep in sync with `cross_sections_for_range`.**  Whenever a new
/// formalism is dispatched there, add it to the `matches!` pattern here so
/// that energy boundary logic (`next_starts_here`) stays correct.
fn range_is_evaluable(range: &ResonanceRange) -> bool {
    if range.urr.is_some() {
        return true;
    }
    if !range.resolved {
        return false;
    }
    matches!(
        range.formalism,
        ResonanceFormalism::SLBW
            | ResonanceFormalism::MLBW
            | ResonanceFormalism::ReichMoore
            | ResonanceFormalism::RMatrixLimited
    )
}

/// Cross-sections for a single resolved resonance range.
///
/// Returns (total, elastic, capture, fission) in barns.
/// Dispatches to the R-Matrix Limited calculator for LRF=7 ranges.
fn cross_sections_for_range(
    range: &ResonanceRange,
    energy_ev: f64,
    awr: f64,
    target_spin: f64,
) -> (f64, f64, f64, f64) {
    // LRF=7 (R-Matrix Limited): dispatch to multi-channel calculator.
    if let Some(rml) = &range.rml {
        return rmatrix_limited::cross_sections_for_rml_range(rml, energy_ev);
    }

    // SLBW/MLBW: dispatch to the SLBW per-range calculator.
    //
    // IMPORTANT: MLBW is *not* implemented as a true multi-level
    // Breit–Wigner formalism.  MLBW ranges are evaluated using the
    // single-level Breit–Wigner (SLBW) formulas as an approximation.
    // This ignores resonance–resonance interference (the defining
    // difference between SLBW and MLBW), so results may be physically
    // incorrect for closely spaced or overlapping resonances.
    //
    // This check must precede the l_group loop because
    // `slbw_cross_sections_for_range` handles all L-groups and J-groups
    // internally (including potential scattering).
    if matches!(
        range.formalism,
        ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
    ) {
        return slbw::slbw_cross_sections_for_range(range, energy_ev, awr, target_spin);
    }

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for l_group in &range.l_groups {
        let l = l_group.l;
        let awr_l = if l_group.awr > 0.0 { l_group.awr } else { awr };

        // Channel radius: use L-dependent radius if available, else the
        // energy-dependent (or constant) global radius from the range.
        let channel_radius = if l_group.apl > 0.0 {
            l_group.apl
        } else {
            range.scattering_radius_at(energy_ev)
        };

        // AP table for per-resonance radius: only applicable when the global
        // NRO=1 table is in use (i.e., no L-group override via APL).
        // When NRO=1, ENDF widths are defined at AP(E_r), not AP(energy_ev),
        // so p_at_er must use the radius evaluated at the resonance energy.
        let ap_table_ref: Option<&Tab1> = if l_group.apl > 0.0 {
            None
        } else {
            range.ap_table.as_ref()
        };

        // Compute channel parameters at this energy.
        let rho = channel::rho(energy_ev, awr_l, channel_radius);
        let p_l = penetrability::penetrability(l, rho);
        let s_l = penetrability::shift_factor(l, rho);
        let phi_l = penetrability::phase_shift(l, rho);

        // Determine fission channel count for this L-group.
        let has_fission = l_group
            .resonances
            .iter()
            .any(|r| r.gfa.abs() > PIVOT_FLOOR || r.gfb.abs() > PIVOT_FLOOR);
        let has_two_fission = l_group.resonances.iter().any(|r| r.gfb.abs() > PIVOT_FLOOR);

        match range.formalism {
            ResonanceFormalism::ReichMoore => {
                if !has_fission {
                    // Single-channel (non-fissile): build J-groups.
                    // Note: in the per-point path (cross_sections_at_energy),
                    // this is rebuilt each call. The batch grid path
                    // (cross_sections_on_grid) hoists this above the energy loop.
                    let jgroups = precompute_jgroups_single(
                        &l_group.resonances,
                        l,
                        awr_l,
                        channel_radius,
                        ap_table_ref,
                        target_spin,
                    );
                    for jg in &jgroups {
                        let (t, e, c, f) = reich_moore_spin_group_precomputed(
                            &jg.resonances,
                            energy_ev,
                            awr_l,
                            jg.g_j,
                            p_l,
                            s_l,
                            phi_l,
                        );
                        total += t;
                        elastic += e;
                        capture += c;
                        fission += f;
                    }
                } else if !has_two_fission {
                    // 2-channel fission: build J-groups (see note above).
                    let jgroups = precompute_jgroups_2ch(
                        &l_group.resonances,
                        l,
                        awr_l,
                        channel_radius,
                        ap_table_ref,
                        target_spin,
                    );
                    for jg in &jgroups {
                        let (t, e, c, f) = reich_moore_2ch_precomputed(
                            &jg.resonances,
                            energy_ev,
                            awr_l,
                            jg.g_j,
                            p_l,
                            s_l,
                            phi_l,
                        );
                        total += t;
                        elastic += e;
                        capture += c;
                        fission += f;
                    }
                } else {
                    // 3-channel fission: build J-groups (see note above).
                    let jgroups = precompute_jgroups_3ch(
                        &l_group.resonances,
                        l,
                        awr_l,
                        channel_radius,
                        ap_table_ref,
                        target_spin,
                    );
                    for jg in &jgroups {
                        let (t, e, c, f) = reich_moore_3ch_precomputed(
                            &jg.resonances,
                            energy_ev,
                            awr_l,
                            jg.g_j,
                            p_l,
                            s_l,
                            phi_l,
                        );
                        total += t;
                        elastic += e;
                        capture += c;
                        fission += f;
                    }
                }
            }
            ResonanceFormalism::SLBW | ResonanceFormalism::MLBW => {
                // Unreachable: SLBW/MLBW ranges are dispatched before
                // entering this loop (see early return above).
                unreachable!("SLBW/MLBW dispatched before l_group loop");
            }
            _ => {
                // Other formalisms (e.g. Adler-Adler LRF=4) are not
                // implemented.  `range_is_evaluable` returns `false` for
                // these, so they should never reach here through the
                // normal dispatcher.  This arm exists only for
                // exhaustiveness; contribution is zero.
                continue;
            }
        }
    }

    (total, elastic, capture, fission)
}

/// Cross-sections for a single spin group (J, π) in the Reich-Moore formalism,
/// using pre-computed per-resonance invariants (γ²_n cached).
///
/// For non-fissile isotopes, the R-matrix has a single neutron channel
/// and the capture channel is eliminated (absorbed into the imaginary
/// part of the resonance denominator).
///
/// ## Mathematical Formulation
///
/// For a single neutron channel with eliminated capture:
///
/// R(E) = Σ_n γ²_n / (E_n - E - iΓ_γ,n/2)
///
/// where γ²_n = Γ_n,n / (2·P_l(E_n)) is the reduced width amplitude squared.
///
/// Level matrix (scalar): Y = (S - B + iP)⁻¹ - R
///
/// X-matrix (scalar): X = P · Y⁻¹ · R · (S - B + iP)⁻¹
///
/// The scattering matrix element is:
///   U = e^{2iφ} · (1 + 2i·X)
///
/// Cross-sections:
///   σ_elastic = (π/k²) · g_J · |1 - U|²
///   σ_total   = (2π/k²) · g_J · (1 - Re(U))
///   σ_capture = σ_total - σ_elastic (unitarity deficit)
///
/// Reference: SAMMY `rml/mrml11.f` Sectio routine
#[allow(clippy::too_many_arguments)]
fn reich_moore_spin_group_precomputed(
    resonances: &[PrecomputedResonanceSingle],
    energy_ev: f64,
    awr: f64,
    g_j: f64,
    p_l: f64,
    s_l: f64,
    phi_l: f64,
) -> (f64, f64, f64, f64) {
    let pi_over_k2 = channel::pi_over_k_squared_barns(energy_ev, awr);

    // Single-channel case (neutron only, capture eliminated).
    // This is the common case for non-fissile isotopes.

    // Boundary condition: B = S(E_n) at each resonance energy.
    // SAMMY typically uses B = 0 (Shift=0 flag in .par file).
    // ENDF convention: B = S(E_n) unless NRO/NAPS flags say otherwise.
    // For simplicity and following SAMMY ex027 (Shift=0), we use B = 0.
    let boundary = 0.0;

    // Build the R-matrix (scalar, complex) = Σ_n γ²_n / (E_n - E - iΓ_γ,n/2)
    //
    // Note: ENDF stores "observed" widths Γ_n. The reduced width amplitude is:
    //   γ²_n = Γ_n / (2 · P_l(ρ_n))
    // where ρ_n = k(E_n)·a, evaluated at the resonance energy.
    //
    // Issue #87: γ²_n is now pre-computed in PrecomputedResonanceSingle.
    //
    // Reference: SAMMY `rml/mrml03.f` Betset (lines 240-276)
    let mut r_real = 0.0;
    let mut r_imag = 0.0;

    for res in resonances {
        let e_r = res.energy;
        let gamma_g = res.gamma_g;
        let gamma_n_reduced_sq = res.gamma_n_reduced_sq;

        // Denominator: (E_n - E)² + (Γ_γ/2)²
        let de = e_r - energy_ev;
        let half_gg = gamma_g / 2.0;
        let denom = de * de + half_gg * half_gg;

        if denom > DIVISION_FLOOR {
            // R-matrix contribution:
            // R += γ²_n / (E_n - E - i·Γ_γ/2)
            //    = γ²_n · (E_n - E + i·Γ_γ/2) / denom
            r_real += gamma_n_reduced_sq * de / denom;
            r_imag += gamma_n_reduced_sq * half_gg / denom;
        }
    }

    // Level matrix Y = 1/(S - B + iP) - R  (scalar, complex)
    let l_real = s_l - boundary;
    let l_imag = p_l;
    let l_denom = l_real * l_real + l_imag * l_imag;

    // 1/(S - B + iP) = (S - B - iP) / |S - B + iP|²
    let l_inv_real = l_real / l_denom;
    let l_inv_imag = -l_imag / l_denom;

    let y_real = l_inv_real - r_real;
    let y_imag = l_inv_imag - r_imag;

    // Y⁻¹ = 1/Y
    let y_denom = y_real * y_real + y_imag * y_imag;
    let y_inv_real = y_real / y_denom;
    let y_inv_imag = -y_imag / y_denom;

    // X-matrix (scalar): X = P · Y⁻¹ · R · (1/(S-B+iP))
    // Actually: X = √P · Y⁻¹ · R · √P · (1/(S-B+iP))
    //
    // From SAMMY mrml11.f: XXXX = √P_J · (Y⁻¹·R)_JI · (√P_I / L_II)
    // For single channel: X = √P · Y⁻¹ · R · √P / L
    //                       = P · Y⁻¹ · R / (S-B+iP)
    //
    // Let's compute step by step:
    // 1. q = Y⁻¹ · R (complex multiply)
    let q_real = y_inv_real * r_real - y_inv_imag * r_imag;
    let q_imag = y_inv_real * r_imag + y_inv_imag * r_real;

    // 2. X = P · q / (S-B+iP) = P · q · (S-B-iP) / |S-B+iP|²
    let x_unscaled_real = q_real * l_real + q_imag * l_imag;
    let x_unscaled_imag = q_imag * l_real - q_real * l_imag;
    let x_real = p_l * x_unscaled_real / l_denom;
    let x_imag = p_l * x_unscaled_imag / l_denom;

    // Compute the collision matrix element U from X.
    //
    //   U = e^{2iφ} · (1 + 2iX)
    //
    // Reference: ENDF-102 Section 2, Lane & Thomas R-matrix theory
    let x = Complex64::new(x_real, x_imag);
    let phase = Complex64::new((2.0 * phi_l).cos(), (2.0 * phi_l).sin());
    let u = phase * (1.0 + 2.0 * Complex64::i() * x);

    // Cross-sections from the collision matrix U:
    //
    //   σ_total   = g_J · (2π/k²) · (1 - Re(U))
    //   σ_elastic = g_J · (π/k²) · |1 - U|²
    //   σ_capture = σ_total - σ_elastic  (unitarity deficit)
    //
    // Reference: standard R-matrix cross-section formulas
    let sigma_total = g_j * 2.0 * pi_over_k2 * (1.0 - u.re);
    let one_minus_u = 1.0 - u;
    let sigma_elastic = g_j * pi_over_k2 * one_minus_u.norm_sqr();
    let sigma_capture = sigma_total - sigma_elastic;

    // For non-fissile isotopes, all absorption is capture.
    (sigma_total, sigma_elastic, sigma_capture, 0.0)
}

/// Reich-Moore 2-channel (neutron + 1 fission) with pre-computed betas.
///
/// Reference: SAMMY `rml/mrml09.f` Twoch routine
#[allow(clippy::too_many_arguments)]
fn reich_moore_2ch_precomputed(
    resonances: &[PrecomputedResonance2ch],
    energy_ev: f64,
    awr: f64,
    g_j: f64,
    p_l: f64,
    s_l: f64,
    phi_l: f64,
) -> (f64, f64, f64, f64) {
    let pi_over_k2 = channel::pi_over_k_squared_barns(energy_ev, awr);
    let boundary = 0.0;

    // 2-channel: neutron + one fission channel.
    // R-matrix is 2x2 complex.
    let mut r_mat = [[Complex64::new(0.0, 0.0); 2]; 2];

    for res in resonances {
        // Denominator: (E_n - E) - i*Gamma_g/2
        let de = res.energy - energy_ev;
        let half_gg = res.gamma_g / 2.0;
        let inv_denom = 1.0 / Complex64::new(de, -half_gg);

        // R_ij += beta_i * beta_j / denom
        let betas = [res.beta_n, res.beta_f];
        for i in 0..2 {
            for j in 0..2 {
                r_mat[i][j] += betas[i] * betas[j] * inv_denom;
            }
        }
    }

    // Level matrix Y = diag(1/(S-B+iP)) - R
    // Channel 0 (neutron): L = S_l - B + i*P_l
    // Channel 1 (fission): L = 0 + i*1 (no penetrability, Pent=0)
    //   -> fission channel: P_f = 1, S_f = 0
    let l_n = Complex64::new(s_l - boundary, p_l);
    let l_f = Complex64::new(0.0, 1.0); // Fission: no barrier

    let l_inv = [1.0 / l_n, 1.0 / l_f];

    let mut y_mat = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            y_mat[i][j] = -r_mat[i][j];
        }
        y_mat[i][i] += l_inv[i];
    }

    // Invert 2x2 Y-matrix.
    // Guard against singular matrix.
    let det = y_mat[0][0] * y_mat[1][1] - y_mat[0][1] * y_mat[1][0];
    if det.norm() < LOG_FLOOR {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let y_inv = [
        [y_mat[1][1] * inv_det, -y_mat[0][1] * inv_det],
        [-y_mat[1][0] * inv_det, y_mat[0][0] * inv_det],
    ];

    // X-matrix: X_ij = sqrt(P_i) * (Y^-1 * R)_ij * sqrt(P_j) / L_jj
    let mut q = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                q[i][j] += y_inv[i][k] * r_mat[k][j];
            }
        }
    }

    let sqrt_p = [p_l.sqrt(), 1.0]; // sqrt(P_n), sqrt(P_f)
    let mut x_mat = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let l_jj = if j == 0 { l_n } else { l_f };
            x_mat[i][j] = sqrt_p[i] * q[i][j] * sqrt_p[j] / l_jj;
        }
    }

    // Collision matrix U from X-matrix.
    let phase2 = Complex64::new((2.0 * phi_l).cos(), (2.0 * phi_l).sin());
    let phase1 = Complex64::new(phi_l.cos(), phi_l.sin());

    let u_nn = phase2 * (1.0 + 2.0 * Complex64::i() * x_mat[0][0]);
    let u_nf = phase1 * 2.0 * Complex64::i() * x_mat[0][1];

    // Cross-sections from U-matrix.
    let sigma_total = g_j * 2.0 * pi_over_k2 * (1.0 - u_nn.re);
    let sigma_elastic = g_j * pi_over_k2 * (1.0 - u_nn).norm_sqr();
    let sigma_fission = g_j * pi_over_k2 * u_nf.norm_sqr();
    let sigma_capture = sigma_total - sigma_elastic - sigma_fission;

    (sigma_total, sigma_elastic, sigma_capture, sigma_fission)
}

/// 3-channel Reich-Moore (neutron + 2 fission channels) with pre-computed betas.
#[allow(clippy::too_many_arguments)]
fn reich_moore_3ch_precomputed(
    resonances: &[PrecomputedResonance3ch],
    energy_ev: f64,
    awr: f64,
    g_j: f64,
    p_l: f64,
    s_l: f64,
    phi_l: f64,
) -> (f64, f64, f64, f64) {
    let pi_over_k2 = channel::pi_over_k_squared_barns(energy_ev, awr);
    let boundary = 0.0;

    let mut r_mat = [[Complex64::new(0.0, 0.0); 3]; 3];

    for res in resonances {
        let de = res.energy - energy_ev;
        let half_gg = res.gamma_g / 2.0;
        let inv_denom = 1.0 / Complex64::new(de, -half_gg);

        let betas = [res.beta_n, res.beta_fa, res.beta_fb];
        for i in 0..3 {
            for j in 0..3 {
                r_mat[i][j] += betas[i] * betas[j] * inv_denom;
            }
        }
    }

    // Level matrix Y.
    let l_n = Complex64::new(s_l - boundary, p_l);
    let l_f = Complex64::new(0.0, 1.0);
    let l_vals = [l_n, l_f, l_f];
    let l_inv: Vec<Complex64> = l_vals.iter().map(|&li| 1.0 / li).collect();

    let mut y_mat = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            y_mat[i][j] = -r_mat[i][j];
        }
        y_mat[i][i] += l_inv[i];
    }

    // Invert 3x3 via cofactor expansion.
    let y_inv = match invert_3x3(y_mat) {
        Some(inv) => inv,
        None => return (0.0, 0.0, 0.0, 0.0),
    };

    // X-matrix.
    let sqrt_p = [p_l.sqrt(), 1.0, 1.0];
    let mut x_mat = [[Complex64::new(0.0, 0.0); 3]; 3];
    let mut q = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                q[i][j] += y_inv[i][k] * r_mat[k][j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            x_mat[i][j] = sqrt_p[i] * q[i][j] * sqrt_p[j] / l_vals[j];
        }
    }

    // Collision matrix U from X-matrix.
    let phase2 = Complex64::new((2.0 * phi_l).cos(), (2.0 * phi_l).sin());
    let phase1 = Complex64::new(phi_l.cos(), phi_l.sin());

    let u_nn = phase2 * (1.0 + 2.0 * Complex64::i() * x_mat[0][0]);
    let u_nf1 = phase1 * 2.0 * Complex64::i() * x_mat[0][1];
    let u_nf2 = phase1 * 2.0 * Complex64::i() * x_mat[0][2];

    // Cross-sections from U-matrix.
    let sigma_total = g_j * 2.0 * pi_over_k2 * (1.0 - u_nn.re);
    let sigma_elastic = g_j * pi_over_k2 * (1.0 - u_nn).norm_sqr();
    let sigma_fission = g_j * pi_over_k2 * (u_nf1.norm_sqr() + u_nf2.norm_sqr());
    let sigma_capture = sigma_total - sigma_elastic - sigma_fission;

    (sigma_total, sigma_elastic, sigma_capture, sigma_fission)
}

/// Invert a 3×3 complex matrix via cofactor expansion.
///
/// Returns `None` if the matrix is singular (|det| < LOG_FLOOR), preventing
/// NaN propagation from 1/det when det ≈ 0.
fn invert_3x3(m: [[Complex64; 3]; 3]) -> Option<[[Complex64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.norm() < LOG_FLOOR {
        return None; // singular — caller returns zero cross-sections
    }

    let inv_det = 1.0 / det;

    let mut result = [[Complex64::new(0.0, 0.0); 3]; 3];
    result[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    result[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    result[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    result[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    result[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    result[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    result[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    result[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    result[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    Some(result)
}

// J-group assembly is now done inline via the `precompute_jgroups_*` functions
// (Issue #87).  The old `group_by_j` import is no longer needed.

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceRange};

    /// Create a simple single-resonance test case for validation.
    #[allow(clippy::too_many_arguments)]
    fn make_single_resonance_data(
        energy: f64,
        gamma_n: f64,
        gamma_g: f64,
        j: f64,
        l: u32,
        awr: f64,
        target_spin: f64,
        scattering_radius: f64,
    ) -> ResonanceData {
        ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin,
                scattering_radius,
                naps: 0,
                l_groups: vec![LGroup {
                    l,
                    awr,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy,
                        j,
                        gn: gamma_n,
                        gg: gamma_g,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                ap_table: None,
                urr: None,
            }],
        }
    }

    #[test]
    fn test_capture_peak_single_resonance() {
        // U-238 6.674 eV resonance.
        // At the resonance energy, capture cross-section should peak at ~22,000 barns.
        let data = make_single_resonance_data(
            6.674,    // E_r (eV)
            1.493e-3, // Γ_n (eV)
            23.0e-3,  // Γ_γ (eV)
            0.5,      // J
            0,        // L
            236.006,  // AWR
            0.0,      // target spin I
            9.4285,   // scattering radius (fm)
        );

        let xs = cross_sections_at_energy(&data, 6.674);

        // The capture cross-section at peak should be approximately:
        // σ_c = g_J × π/k² × 4×Γ_n×Γ_γ / Γ² where Γ = Γ_n + Γ_γ
        // For the RM formalism the peak is very close to this BW estimate.
        // g_J = 1.0, π/k² ≈ 98,200 barns, Γ = 0.024493
        // σ_c ≈ 1.0 × 98200 × 4 × 1.493e-3 × 23.0e-3 / (24.493e-3)²
        //     ≈ 98200 × 0.2289 ≈ 22,478 barns
        println!("Capture at 6.674 eV: {} barns", xs.capture);
        println!("Total at 6.674 eV: {} barns", xs.total);
        println!("Elastic at 6.674 eV: {} barns", xs.elastic);

        assert!(
            xs.capture > 15000.0 && xs.capture < 30000.0,
            "Capture should be ~22000 barns, got {}",
            xs.capture
        );
        assert!(xs.total > xs.capture, "Total > capture");
        assert!(xs.elastic > 0.0, "Elastic should be positive");
        assert!(xs.fission.abs() < 1e-10, "No fission for U-238");
    }

    #[test]
    fn test_1_over_v_behavior() {
        // Far from resonances, capture cross-section should follow 1/v ∝ 1/√E.
        // The 6.674 eV resonance tail should dominate at low energies.
        let data =
            make_single_resonance_data(6.674, 1.493e-3, 23.0e-3, 0.5, 0, 236.006, 0.0, 9.4285);

        let xs_01 = cross_sections_at_energy(&data, 0.1);
        let xs_04 = cross_sections_at_energy(&data, 0.4);

        // At low E, σ ∝ 1/√E, so σ(0.1)/σ(0.4) ≈ √(0.4/0.1) = 2.0
        let ratio = xs_01.capture / xs_04.capture;
        println!("1/v ratio test: σ(0.1)/σ(0.4) = {}", ratio);
        assert!(
            (ratio - 2.0).abs() < 0.3,
            "Expected ~2.0 for 1/v behavior, got {}",
            ratio
        );
    }

    #[test]
    fn test_cross_sections_positive() {
        // All cross-sections must be non-negative at all energies.
        let data =
            make_single_resonance_data(6.674, 1.493e-3, 23.0e-3, 0.5, 0, 236.006, 0.0, 9.4285);

        for &e in &[0.01, 0.1, 1.0, 5.0, 6.0, 6.674, 7.0, 10.0, 100.0, 1000.0] {
            let xs = cross_sections_at_energy(&data, e);
            assert!(xs.total >= 0.0, "Total negative at E={}: {}", e, xs.total);
            assert!(
                xs.elastic >= 0.0,
                "Elastic negative at E={}: {}",
                e,
                xs.elastic
            );
            assert!(
                xs.capture >= -1e-10,
                "Capture negative at E={}: {}",
                e,
                xs.capture
            );
        }
    }

    /// Parse full U-238 ENDF file and compute cross-sections.
    ///
    /// Validates against SAMMY ex027 output (Doppler-broadened at 300K, but
    /// we compare unbroadened RM values which should bracket the broadened data).
    #[test]
    fn test_u238_full_endf_cross_sections() {
        let endf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../SAMMY/SAMMY/samexm_new/ex027_new/ex027.endf");

        if !endf_path.exists() {
            eprintln!("Skipping: SAMMY ENDF file not found at {:?}", endf_path);
            return;
        }

        let endf_text = std::fs::read_to_string(&endf_path).unwrap();
        let data = nereids_endf::parser::parse_endf_file2(&endf_text).unwrap();

        // Compute cross-sections at several energies near the 6.674 eV resonance.
        let energies = [1.0, 5.0, 6.0, 6.5, 6.674, 7.0, 8.0, 10.0, 20.0, 50.0, 100.0];

        println!("\nU-238 Reich-Moore cross-sections (unbroadened):");
        println!(
            "{:>10} {:>12} {:>12} {:>12} {:>12}",
            "E (eV)", "Total", "Elastic", "Capture", "Fission"
        );

        for &e in &energies {
            let xs = cross_sections_at_energy(&data, e);
            println!(
                "{:>10.3} {:>12.3} {:>12.3} {:>12.3} {:>12.6}",
                e, xs.total, xs.elastic, xs.capture, xs.fission
            );

            // Basic sanity: all cross-sections non-negative.
            assert!(xs.total >= 0.0, "Total negative at E={}", e);
            assert!(xs.elastic >= 0.0, "Elastic negative at E={}", e);
            // Capture can be very slightly negative due to floating point.
            assert!(
                xs.capture >= -0.01,
                "Capture negative at E={}: {}",
                e,
                xs.capture
            );
        }

        // Check the 6.674 eV resonance peak.
        // With the full ENDF file (all resonances), the peak capture
        // should still be dominated by the 6.674 eV resonance.
        let xs_peak = cross_sections_at_energy(&data, 6.674);
        assert!(
            xs_peak.capture > 10000.0,
            "Capture at 6.674 eV should be >10,000 barns (got {})",
            xs_peak.capture
        );

        // The 20.87 eV resonance should also show a significant peak.
        let xs_20 = cross_sections_at_energy(&data, 20.87);
        assert!(
            xs_20.capture > 1000.0,
            "Capture at 20.87 eV should be >1,000 barns (got {})",
            xs_20.capture
        );

        // SAMMY ex027 broadened output at ~6.674 eV gives ~339 barns capture.
        // Our UNBROADENED result should be MUCH larger (since Doppler broadening
        // spreads the peak). This confirms we're computing the correct physics.
        println!(
            "\n6.674 eV peak: capture={:.0} barns (unbroadened), SAMMY broadened ~339 barns",
            xs_peak.capture
        );
        assert!(
            xs_peak.capture > 339.0,
            "Unbroadened peak must exceed SAMMY broadened value"
        );
    }

    /// Build a single-resonance `ResonanceData` with a chosen formalism.
    fn make_slbw_data(formalism: ResonanceFormalism) -> ResonanceData {
        ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 6.674,
                        j: 0.5,
                        gn: 1.493e-3,
                        gg: 23.0e-3,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                ap_table: None,
                urr: None,
            }],
        }
    }

    /// `cross_sections_at_energy` with an SLBW-formalism range must give
    /// the same result as `slbw::slbw_cross_sections`.
    #[test]
    fn test_dispatcher_slbw_matches_slbw_module() {
        let data = make_slbw_data(ResonanceFormalism::SLBW);

        let test_energies = [0.1, 1.0, 5.0, 6.0, 6.674, 7.0, 10.0, 100.0];
        for &e in &test_energies {
            let via_dispatcher = cross_sections_at_energy(&data, e);
            let via_slbw = crate::slbw::slbw_cross_sections(&data, e);

            let eps = 1e-10;
            assert!(
                (via_dispatcher.total - via_slbw.total).abs() < eps,
                "total mismatch at {e} eV: dispatcher={} slbw={}",
                via_dispatcher.total,
                via_slbw.total
            );
            assert!(
                (via_dispatcher.capture - via_slbw.capture).abs() < eps,
                "capture mismatch at {e} eV: dispatcher={} slbw={}",
                via_dispatcher.capture,
                via_slbw.capture
            );
            assert!(
                (via_dispatcher.elastic - via_slbw.elastic).abs() < eps,
                "elastic mismatch at {e} eV: dispatcher={} slbw={}",
                via_dispatcher.elastic,
                via_slbw.elastic
            );
        }
    }

    /// MLBW is dispatched as SLBW (approximation).  Verify that the
    /// dispatcher returns the same values as the SLBW module when given
    /// an MLBW-formalism range, and that the results are physically
    /// reasonable (positive, peak at resonance energy).
    #[test]
    fn test_dispatcher_mlbw_uses_slbw_approximation() {
        let data_mlbw = make_slbw_data(ResonanceFormalism::MLBW);
        let data_slbw = make_slbw_data(ResonanceFormalism::SLBW);

        let test_energies = [1.0, 6.674, 10.0];
        for &e in &test_energies {
            let xs_mlbw = cross_sections_at_energy(&data_mlbw, e);
            let xs_slbw = cross_sections_at_energy(&data_slbw, e);
            let eps = 1e-10;
            assert!(
                (xs_mlbw.total - xs_slbw.total).abs() < eps,
                "MLBW/SLBW mismatch at {e} eV: mlbw={} slbw={}",
                xs_mlbw.total,
                xs_slbw.total
            );
        }

        // Sanity: peak capture at resonance energy should be large.
        let xs_peak = cross_sections_at_energy(&data_mlbw, 6.674);
        assert!(
            xs_peak.capture > 1000.0,
            "MLBW capture at 6.674 eV should be substantial (got {})",
            xs_peak.capture
        );
    }

    /// Singular Y-matrix guard: when the R-matrix contribution nearly
    /// cancels the L⁻¹ diagonal at a resonance energy, Y ≈ 0 and
    /// Y⁻¹ diverges.  The cross-sections must remain finite and
    /// non-negative (no NaN or Inf propagation).
    ///
    /// We construct a scenario where evaluation occurs exactly at E_r,
    /// maximizing the R-matrix contribution.  With a very narrow
    /// resonance (small Γ_γ), the imaginary denominator is tiny and
    /// the R-matrix peak is enormous, stressing the Y inversion.
    #[test]
    fn test_reich_moore_singular_y_matrix_guard() {
        // Very narrow resonance: Γ_γ = 1e-6 eV makes R huge at E = E_r.
        let data = make_single_resonance_data(
            10.0,    // E_r
            1.0e-3,  // Γ_n (eV)
            1.0e-6,  // Γ_γ (very small → large R at resonance)
            0.5,     // J
            0,       // L
            236.006, // AWR
            0.0,     // target spin
            9.4285,  // radius
        );

        // Evaluate exactly at E_r where R is maximized.
        let xs = cross_sections_at_energy(&data, 10.0);
        assert!(
            xs.total.is_finite() && xs.total >= 0.0,
            "Total must be finite and non-negative at resonance peak, got {}",
            xs.total
        );
        assert!(
            xs.elastic.is_finite() && xs.elastic >= 0.0,
            "Elastic must be finite and non-negative, got {}",
            xs.elastic
        );
        assert!(
            xs.capture.is_finite(),
            "Capture must be finite, got {}",
            xs.capture
        );
    }

    /// Reich-Moore 2-channel (fission) with a singular det guard:
    /// when both fission and neutron widths are tiny, the 2x2 Y-matrix
    /// determinant can be near zero.  Results must be finite.
    #[test]
    fn test_reich_moore_fission_near_singular() {
        let data = ResonanceData {
            isotope: nereids_core::types::Isotope::new(94, 239).unwrap(),
            za: 94239,
            awr: 236.998,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.5,
                scattering_radius: 9.41,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.998,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 10.0,
                        j: 1.0,
                        gn: 1.0e-8,  // very small neutron width
                        gg: 1.0e-8,  // very small capture width
                        gfa: 1.0e-8, // very small fission width
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                ap_table: None,
                urr: None,
            }],
        };

        // Evaluate at the resonance energy.
        let xs = cross_sections_at_energy(&data, 10.0);
        assert!(
            xs.total.is_finite() && xs.total >= 0.0,
            "Total must be finite, got {}",
            xs.total
        );
        assert!(
            xs.fission.is_finite() && xs.fission >= 0.0,
            "Fission must be finite and non-negative, got {}",
            xs.fission
        );
    }

    /// `cross_sections_on_grid` (batch, precompute hoisted) must produce
    /// identical results to `cross_sections_at_energy` (per-point) for
    /// Reich-Moore data.
    #[test]
    fn test_grid_matches_per_point_reich_moore() {
        let data =
            make_single_resonance_data(6.674, 1.493e-3, 23.0e-3, 0.5, 0, 236.006, 0.0, 9.4285);

        let energies = [0.01, 0.1, 1.0, 5.0, 6.0, 6.674, 7.0, 10.0, 100.0, 1000.0];
        let grid_results = cross_sections_on_grid(&data, &energies);

        for (i, &e) in energies.iter().enumerate() {
            let point = cross_sections_at_energy(&data, e);
            let grid = &grid_results[i];
            let eps = 1e-12;
            assert!(
                (point.total - grid.total).abs() < eps,
                "total mismatch at E={e}: per_point={} grid={}",
                point.total,
                grid.total
            );
            assert!(
                (point.elastic - grid.elastic).abs() < eps,
                "elastic mismatch at E={e}: per_point={} grid={}",
                point.elastic,
                grid.elastic
            );
            assert!(
                (point.capture - grid.capture).abs() < eps,
                "capture mismatch at E={e}: per_point={} grid={}",
                point.capture,
                grid.capture
            );
            assert!(
                (point.fission - grid.fission).abs() < eps,
                "fission mismatch at E={e}: per_point={} grid={}",
                point.fission,
                grid.fission
            );
        }
    }

    /// `cross_sections_on_grid` must match `cross_sections_at_energy` for
    /// SLBW-formalism data too (the batch path precomputes SLBW J-groups).
    #[test]
    fn test_grid_matches_per_point_slbw() {
        let data = make_slbw_data(ResonanceFormalism::SLBW);

        let energies = [0.1, 1.0, 5.0, 6.0, 6.674, 7.0, 10.0, 100.0];
        let grid_results = cross_sections_on_grid(&data, &energies);

        for (i, &e) in energies.iter().enumerate() {
            let point = cross_sections_at_energy(&data, e);
            let grid = &grid_results[i];
            let eps = 1e-12;
            assert!(
                (point.total - grid.total).abs() < eps,
                "total mismatch at E={e}: per_point={} grid={}",
                point.total,
                grid.total
            );
            assert!(
                (point.capture - grid.capture).abs() < eps,
                "capture mismatch at E={e}: per_point={} grid={}",
                point.capture,
                grid.capture
            );
        }
    }

    /// `cross_sections_on_grid` must match per-point for the full U-238
    /// ENDF file (many resonances, L-groups, J-groups).
    #[test]
    fn test_grid_matches_per_point_u238_full() {
        let endf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../SAMMY/SAMMY/samexm_new/ex027_new/ex027.endf");

        if !endf_path.exists() {
            eprintln!("Skipping: SAMMY ENDF file not found at {:?}", endf_path);
            return;
        }

        let endf_text = std::fs::read_to_string(&endf_path).unwrap();
        let data = nereids_endf::parser::parse_endf_file2(&endf_text).unwrap();

        let energies: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.5).collect();
        let grid_results = cross_sections_on_grid(&data, &energies);

        for (i, &e) in energies.iter().enumerate() {
            let point = cross_sections_at_energy(&data, e);
            let grid = &grid_results[i];
            let eps = 1e-10;
            assert!(
                (point.total - grid.total).abs() < eps * point.total.abs().max(1.0),
                "total mismatch at E={e}: per_point={} grid={}",
                point.total,
                grid.total
            );
            assert!(
                (point.capture - grid.capture).abs() < eps * point.capture.abs().max(1.0),
                "capture mismatch at E={e}: per_point={} grid={}",
                point.capture,
                grid.capture
            );
        }
    }
}
