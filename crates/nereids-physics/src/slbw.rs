//! Single-Level Breit-Wigner (SLBW) cross-section calculation.
//!
//! SLBW is the simplest resonance formalism. It treats each resonance
//! independently (no interference between resonances). It is useful as:
//! - A validation check against the more complex Reich-Moore calculation
//! - A quick approximation for isolated resonances
//! - An educational/debugging tool
//!
//! For isolated, well-separated resonances, SLBW and Reich-Moore should
//! give nearly identical results.
//!
//! This module also handles MLBW (Multi-Level Breit-Wigner) ranges, which are
//! evaluated using the SLBW formulas as an approximation (see `reich_moore`
//! module for details).
//!
//! ## SAMMY Reference
//! - `mlb/mmlb4.f90` Abpart_Mlb subroutine
//! - SAMMY manual Section 2.1.1
//!
//! ## Formulas
//! For each resonance at energy E_r with total J:
//!
//! σ_capture(E) = g_J × π/k² × Γ_n(E)·Γ_γ / ((E-E_r)² + (Γ/2)²)
//!
//! σ_elastic(E) = g_J × π/k² × [Γ_n(E)² / ((E-E_r)² + (Γ/2)²)
//!                + 2·sin(φ)·(Γ_n(E)·(E-E_r)·cos(φ) + Γ_n(E)·Γ/2·sin(φ))
//!                  / ((E-E_r)² + (Γ/2)²)
//!                + sin²(φ)]
//!
//! where Γ_n(E) = Γ_n(E_r) × √(E/E_r) × P_l(E)/P_l(E_r)
//! and Γ = Γ_n(E) + Γ_γ + Γ_f

use nereids_core::constants::{DIVISION_FLOOR, PIVOT_FLOOR};
use nereids_endf::resonance::ResonanceData;

use crate::channel;
use crate::penetrability;
use crate::reich_moore::{CrossSections, PrecomputedJGroup, group_resonances_by_j};

// ─── Per-resonance precomputed invariants for SLBW ────────────────────────────
//
// In SLBW, the energy-dependent neutron width is:
//   Γ_n(E) = Γ_n(E_r) × √(E/E_r) × P_l(E)/P_l(E_r)
//
// The quantities that depend only on resonance parameters (not on E):
//   - P_l(E_r): penetrability at resonance energy
//   - |Γ_n(E_r)|: neutron width magnitude
//   - Γ_γ, Γ_f: capture and fission widths
//   - E_r: resonance energy
//
// Issue #87: Pre-compute P_l(E_r) and J-groups once before the energy loop.

/// Per-resonance invariants for SLBW, pre-computed once per resonance.
pub(crate) struct PrecomputedSlbwResonance {
    /// Resonance energy E_r (eV).
    pub(crate) energy: f64,
    /// Capture width Γ_γ (eV).
    pub(crate) gamma_g: f64,
    /// Fission width |Γ_fa| + |Γ_fb| (eV).
    pub(crate) gamma_f: f64,
    /// Absolute neutron width |Γ_n(E_r)| (eV).
    pub(crate) gn_abs: f64,
    /// Penetrability at resonance energy P_l(ρ_r).
    /// Pre-computed to avoid redundant `penetrability(l, rho_r)` calls.
    pub(crate) p_at_er: f64,
}

/// Pre-computed J-group for SLBW (type alias for the generic J-group).
pub(crate) type PrecomputedSlbwJGroup = PrecomputedJGroup<PrecomputedSlbwResonance>;

/// Build pre-computed J-groups for SLBW.
///
/// All quantities depend only on resonance parameters (not incident energy),
/// so the result can be computed once and reused across all energy points.
///
/// Uses the shared `group_resonances_by_j` helper (Issue #158) with an
/// SLBW-specific closure for per-resonance invariant construction.
pub(crate) fn precompute_slbw_jgroups(
    resonances: &[nereids_endf::resonance::Resonance],
    l: u32,
    awr_l: f64,
    range: &nereids_endf::resonance::ResonanceRange,
    l_group: &nereids_endf::resonance::LGroup,
    target_spin: f64,
) -> Vec<PrecomputedSlbwJGroup> {
    group_resonances_by_j(resonances, target_spin, |res| {
        let e_r = res.energy;

        // Pre-compute P_l(E_r) — this is the redundant computation Issue #87 eliminates.
        let p_at_er = if e_r.abs() > PIVOT_FLOOR {
            let radius_at_er = if l_group.apl > 0.0 {
                l_group.apl
            } else if range.naps == 0 {
                // NAPS=0: use channel radius per ENDF-6 §2.2.1
                0.123 * awr_l.cbrt() + 0.08
            } else {
                // NAPS=1 (default): use scattering radius AP or AP(E)
                range.scattering_radius_at(e_r.abs())
            };
            let rho_r = channel::rho(e_r.abs(), awr_l, radius_at_er);
            penetrability::penetrability(l, rho_r)
        } else {
            0.0 // Will produce gamma_n = 0 in the energy loop
        };

        PrecomputedSlbwResonance {
            energy: e_r,
            gamma_g: res.gg,
            gamma_f: res.gfa.abs() + res.gfb.abs(),
            gn_abs: res.gn.abs(),
            p_at_er,
        }
    })
}

/// Compute SLBW cross-sections at a single energy.
///
/// Works with both SLBW and Reich-Moore ENDF data (uses the same
/// resonance parameters but applies the SLBW formulas).
pub fn slbw_cross_sections(data: &ResonanceData, energy_ev: f64) -> CrossSections {
    let awr = data.awr;

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    // Use simple closed-interval logic: energy_ev must be in [e_low, e_high].
    //
    // This function evaluates *only* resolved SLBW/MLBW ranges and skips
    // everything else (including URR ranges where !range.resolved).  Using
    // half-open [e_low, e_high) when the next range is a URR range would
    // exclude the shared boundary energy from the resolved range while still
    // skipping the URR range, producing zero XS at the boundary — an
    // artificial dip.  Closed intervals prevent that gap.  A double-count at
    // a shared boundary between two resolved ranges is a negligibly small
    // effect compared to the gap that half-open logic would introduce.
    for range in &data.ranges {
        if !range.resolved || energy_ev < range.energy_low || energy_ev > range.energy_high {
            continue;
        }

        // Each range carries its own target_spin — pass per-range, not
        // from the first range, to correctly compute statistical weights g_J.
        let (t, e, c, f) = slbw_cross_sections_for_range(range, energy_ev, awr, range.target_spin);
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

/// Compute SLBW cross-sections for a single resolved resonance range.
///
/// This is the per-range workhorse called by both `slbw_cross_sections`
/// (which iterates over all ranges) and the unified dispatcher in
/// `reich_moore::cross_sections_for_range`.
///
/// Returns `(total, elastic, capture, fission)` in barns.
pub fn slbw_cross_sections_for_range(
    range: &nereids_endf::resonance::ResonanceRange,
    energy_ev: f64,
    awr: f64,
    target_spin: f64,
) -> (f64, f64, f64, f64) {
    let pi_over_k2 = channel::pi_over_k_squared_barns(energy_ev, awr);

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for l_group in &range.l_groups {
        let l = l_group.l;
        let awr_l = if l_group.awr > 0.0 { l_group.awr } else { awr };

        // Scattering radius for phase shift (always AP/APL).
        let scatt_radius = if l_group.apl > 0.0 {
            l_group.apl
        } else {
            range.scattering_radius_at(energy_ev)
        };
        // Penetrability radius: NAPS=0 uses channel radius formula,
        // NAPS=1 uses scattering radius (ENDF-6 §2.2.1).
        let pen_radius = if l_group.apl > 0.0 {
            l_group.apl
        } else if range.naps == 0 {
            0.123 * awr_l.cbrt() + 0.08
        } else {
            scatt_radius
        };

        let rho_phase = channel::rho(energy_ev, awr_l, scatt_radius);
        let rho_pen = channel::rho(energy_ev, awr_l, pen_radius);
        let phi = penetrability::phase_shift(l, rho_phase);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let sin2_phi = sin_phi * sin_phi;
        let p_at_e = penetrability::penetrability(l, rho_pen);

        // Build J-groups and per-resonance invariants.
        // Note: in the per-point path (slbw_cross_sections_for_range), this
        // is rebuilt each call. The batch grid path (cross_sections_on_grid)
        // hoists this above the energy loop via precompute_range_data.
        let j_groups =
            precompute_slbw_jgroups(&l_group.resonances, l, awr_l, range, l_group, target_spin);

        for jg in &j_groups {
            let g_j = jg.g_j;

            // Potential scattering for this (l, J) group.
            // Added once per spin group.
            let pot_scatter = pi_over_k2 * g_j * 4.0 * sin2_phi;
            elastic += pot_scatter;
            total += pot_scatter;

            for res in &jg.resonances {
                let e_r = res.energy;

                // Energy-dependent neutron width:
                // Γ_n(E) = Γ_n(E_r) × √(E/E_r) × P_l(E)/P_l(E_r)
                // P_l(E_r) is pre-computed in res.p_at_er (Issue #87).
                let gamma_n = if e_r.abs() > PIVOT_FLOOR && res.p_at_er > PIVOT_FLOOR {
                    res.gn_abs * (energy_ev / e_r.abs()).sqrt() * p_at_e / res.p_at_er
                } else {
                    0.0
                };

                let gamma_total = gamma_n + res.gamma_g + res.gamma_f;
                let de = energy_ev - e_r;
                let denom = de * de + (gamma_total / 2.0).powi(2);

                if denom < DIVISION_FLOOR {
                    continue;
                }

                // Capture cross-section (symmetric Breit-Wigner).
                let sigma_c = pi_over_k2 * g_j * gamma_n * res.gamma_g / denom;
                capture += sigma_c;
                total += sigma_c;

                // Fission cross-section.
                let sigma_f = pi_over_k2 * g_j * gamma_n * res.gamma_f / denom;
                fission += sigma_f;
                total += sigma_f;

                // Elastic resonance scattering (interference term + resonance peak).
                // σ_el_res = g × π/k² × [Γ_n² / denom]
                let sigma_e_res = pi_over_k2 * g_j * gamma_n * gamma_n / denom;
                elastic += sigma_e_res;
                total += sigma_e_res;

                // Interference between resonance and potential scattering.
                let interf = pi_over_k2
                    * g_j
                    * 2.0
                    * gamma_n
                    * (de * cos_phi * 2.0 * sin_phi + (gamma_total / 2.0) * 2.0 * sin2_phi)
                    / denom;
                elastic += interf;
                total += interf;
            }
        }
    }

    (total, elastic, capture, fission)
}

/// Evaluate SLBW cross-sections for a single L-group using pre-cached J-groups.
///
/// This is the per-energy inner loop extracted from `slbw_cross_sections_for_range`,
/// used by the batch grid path (`cross_sections_on_grid`) to avoid redundant
/// J-group precomputation at every energy point.
///
/// # Arguments
/// * `jgroups` — Pre-computed J-groups (energy-independent invariants).
/// * `energy_ev` — Incident neutron energy (eV).
/// * `pi_over_k2` — π/k² in barns at this energy.
/// * `p_at_e` — Penetrability P_l(ρ) at incident energy.
/// * `sin_phi` — sin(φ_l) at incident energy.
/// * `cos_phi` — cos(φ_l) at incident energy.
/// * `sin2_phi` — sin²(φ_l) at incident energy.
///
/// # Returns
/// `(total, elastic, capture, fission)` in barns for this L-group.
#[allow(clippy::too_many_arguments)]
pub(crate) fn slbw_evaluate_with_cached_jgroups(
    jgroups: &[PrecomputedSlbwJGroup],
    energy_ev: f64,
    pi_over_k2: f64,
    p_at_e: f64,
    sin_phi: f64,
    cos_phi: f64,
    sin2_phi: f64,
) -> (f64, f64, f64, f64) {
    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for jg in jgroups {
        let g_j = jg.g_j;

        // Potential scattering for this (l, J) group.
        let pot_scatter = pi_over_k2 * g_j * 4.0 * sin2_phi;
        elastic += pot_scatter;
        total += pot_scatter;

        for res in &jg.resonances {
            let e_r = res.energy;

            // Energy-dependent neutron width:
            // Γ_n(E) = Γ_n(E_r) × √(E/E_r) × P_l(E)/P_l(E_r)
            // P_l(E_r) is pre-computed in res.p_at_er (Issue #87).
            let gamma_n = if e_r.abs() > PIVOT_FLOOR && res.p_at_er > PIVOT_FLOOR {
                res.gn_abs * (energy_ev / e_r.abs()).sqrt() * p_at_e / res.p_at_er
            } else {
                0.0
            };

            let gamma_total = gamma_n + res.gamma_g + res.gamma_f;
            let de = energy_ev - e_r;
            let denom = de * de + (gamma_total / 2.0).powi(2);

            if denom < DIVISION_FLOOR {
                continue;
            }

            // Capture cross-section (symmetric Breit-Wigner).
            let sigma_c = pi_over_k2 * g_j * gamma_n * res.gamma_g / denom;
            capture += sigma_c;
            total += sigma_c;

            // Fission cross-section.
            let sigma_f = pi_over_k2 * g_j * gamma_n * res.gamma_f / denom;
            fission += sigma_f;
            total += sigma_f;

            // Elastic resonance scattering (interference term + resonance peak).
            let sigma_e_res = pi_over_k2 * g_j * gamma_n * gamma_n / denom;
            elastic += sigma_e_res;
            total += sigma_e_res;

            // Interference between resonance and potential scattering.
            let interf = pi_over_k2
                * g_j
                * 2.0
                * gamma_n
                * (de * cos_phi * 2.0 * sin_phi + (gamma_total / 2.0) * 2.0 * sin2_phi)
                / denom;
            elastic += interf;
            total += interf;
        }
    }

    (total, elastic, capture, fission)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};

    #[test]
    fn test_slbw_capture_peak() {
        // Single resonance at 6.674 eV (U-238).
        // SLBW and RM should agree well for an isolated resonance.
        let data = ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                naps: 1,
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
                urr: None,
                ap_table: None,
                r_external: vec![],
            }],
        };

        let xs = slbw_cross_sections(&data, 6.674);
        println!("SLBW capture at 6.674 eV: {} barns", xs.capture);

        // Same estimate as RM: ~22,000 barns.
        assert!(
            xs.capture > 15000.0 && xs.capture < 30000.0,
            "Capture should be ~22000 barns, got {}",
            xs.capture
        );
    }

    #[test]
    fn test_slbw_vs_rm_single_resonance() {
        // For a single isolated resonance, SLBW and RM should be very close.
        use crate::reich_moore;

        let resonances = vec![LGroup {
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
        }];

        let rm_data = ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                naps: 1,
                l_groups: resonances.clone(),
                rml: None,
                urr: None,
                ap_table: None,
                r_external: vec![],
            }],
        };

        let slbw_data = ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                naps: 1,
                l_groups: resonances,
                rml: None,
                urr: None,
                ap_table: None,
                r_external: vec![],
            }],
        };

        // Compare at several energies near the resonance peak.
        // Note: RM and SLBW differ in how they handle energy-dependent
        // neutron widths away from the peak. SLBW includes an extra √(E/E_r)
        // velocity factor in Γ_n(E), leading to ~10-15% differences in the
        // resonance wings. At the peak and very near it, they agree well.
        for &e in &[6.5, 6.674, 7.0] {
            let rm = reich_moore::cross_sections_at_energy(&rm_data, e);
            let slbw = slbw_cross_sections(&slbw_data, e);

            let rel_diff_cap =
                (rm.capture - slbw.capture).abs() / rm.capture.max(slbw.capture).max(1e-10);

            println!(
                "E={:.3}: RM_cap={:.2}, SLBW_cap={:.2}, rel_diff={:.4}",
                e, rm.capture, slbw.capture, rel_diff_cap
            );

            // Near the peak, the formalisms should agree within ~5%.
            assert!(
                rel_diff_cap < 0.05,
                "RM vs SLBW capture differ by {:.1}% at E={} eV",
                rel_diff_cap * 100.0,
                e
            );
        }
    }
}
