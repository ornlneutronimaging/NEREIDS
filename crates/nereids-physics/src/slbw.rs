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
//! This module also provides true MLBW (Multi-Level Breit-Wigner) via
//! `mlbw_evaluate_with_cached_jgroups`, which includes resonance-resonance
//! interference in the elastic channel (see SAMMY `mlb/mmlb3.f90`).
//! MLBW is dispatched through `reich_moore::cross_sections_on_grid` (the
//! single public entry point) like every other formalism.
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
                channel::endf_channel_radius_fm(awr_l)
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
/// Thin wrapper: per L-group, builds the precomputed J-groups once and
/// delegates to `slbw_evaluate_with_cached_jgroups`.  The batch grid
/// path (`cross_sections_on_grid`) calls the same evaluator with
/// J-groups precomputed once per range, which is how we guarantee
/// batch/per-point bit-exact equivalence.
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
            channel::endf_channel_radius_fm(awr_l)
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

        let j_groups =
            precompute_slbw_jgroups(&l_group.resonances, l, awr_l, range, l_group, target_spin);

        let (t, e, c, f) = slbw_evaluate_with_cached_jgroups(
            &j_groups, energy_ev, pi_over_k2, p_at_e, sin_phi, cos_phi, sin2_phi,
        );
        total += t;
        elastic += e;
        capture += c;
        fission += f;
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

/// Evaluate **true MLBW** cross-sections for a single L-group using
/// pre-cached J-groups (resonance-resonance interference in elastic).
///
/// This is the per-energy MLBW inner loop.  The batch grid path
/// (`reich_moore::cross_sections_on_grid`) precomputes the J-groups
/// once per range and calls this evaluator at each energy; the
/// single-energy entry point (`reich_moore::cross_sections_at_energy`)
/// goes through the same precompute+evaluate pipeline.  Both callers
/// share exactly this evaluator — which is what prevents the #465
/// class of bug (the batch dispatcher previously routed MLBW ranges
/// through `slbw_evaluate_with_cached_jgroups`,
/// silently computing SLBW's incoherent elastic sum instead of MLBW's
/// coherent sum).
///
/// # Arguments
/// * `jgroups` — Pre-computed J-groups (energy-independent invariants).
///   Uses the same `PrecomputedSlbwJGroup` type as SLBW; the cached
///   `p_at_er` and `gn_abs` have identical meaning across both
///   formalisms.
/// * `energy_ev` — Incident neutron energy (eV).
/// * `pi_over_k2` — π/k² in barns at this energy.
/// * `p_at_e` — Penetrability P_l(ρ) at the incident energy.
/// * `phi` — Hard-sphere phase shift φ_l at the incident energy.
///
/// # Returns
/// `(total, elastic, capture, fission)` in barns for this L-group.
///
/// # Physics
///   U_nn = e^{-2iφ} · [1 + i · Σ_r Γ_n^r / (E_r - E - iΓ_tot^r/2)]
///   σ_elastic = (π/k²) · g_J · |1 - U_nn|²
///
/// Capture and fission are identical to SLBW (per-resonance incoherent
/// sums).
///
/// # SAMMY reference
/// `mlb/mmlb3.f90` Elastc_Mlb, `mlb/mmlb4.f90` Abpart_Mlb.
pub(crate) fn mlbw_evaluate_with_cached_jgroups(
    jgroups: &[PrecomputedSlbwJGroup],
    energy_ev: f64,
    pi_over_k2: f64,
    p_at_e: f64,
    phi: f64,
) -> (f64, f64, f64, f64) {
    use num_complex::Complex64;

    // Phase factor: e^{-2iφ} = cos(2φ) - i·sin(2φ).
    //
    // TRUTH SOURCE: SAMMY mlb/mmlb3.f90 Elastc_Mlb line 54.
    // Convention: U = e^{-2iφ} · (1 + iX), NOT e^{+2iφ} · (1 - iX).
    // The positive exponent was a bug (commit 5508fea) that caused
    // negative total cross-sections for all MLBW isotopes (Hf-176/177/
    // 178/179). Fixed in commit f0eadc1.
    let phase2 = Complex64::new((2.0 * phi).cos(), -(2.0 * phi).sin());

    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for jg in jgroups {
        let g_j = jg.g_j;

        // Coherent sum over resonances: X = Σ_r Γ_n^r / (E_r - E - iΓ_tot^r/2)
        let mut x_sum = Complex64::new(0.0, 0.0);

        for res in &jg.resonances {
            let e_r = res.energy;
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

            // Capture — identical to SLBW (incoherent per-resonance).
            let sigma_c = pi_over_k2 * g_j * gamma_n * res.gamma_g / denom;
            capture += sigma_c;

            // Fission — identical to SLBW (incoherent per-resonance).
            let sigma_f = pi_over_k2 * g_j * gamma_n * res.gamma_f / denom;
            fission += sigma_f;

            // Accumulate coherent sum for U-matrix:
            //   Γ_n / (E_r - E - iΓ_tot/2)
            //   = Γ_n · (E_r - E + iΓ_tot/2) / [(E_r-E)² + (Γ_tot/2)²]
            // Note: de = E - E_r, so E_r - E = -de.
            let x_r = Complex64::new(
                gamma_n * (-de) / denom,
                gamma_n * (gamma_total / 2.0) / denom,
            );
            x_sum += x_r;
        }

        // Collision matrix: U_nn = e^{-2iφ} · (1 + i·X)
        //   1 + i·X = (1 - X_im) + i·X_re
        let one_plus_ix = Complex64::new(1.0 - x_sum.im, x_sum.re);
        let u_nn = phase2 * one_plus_ix;

        // σ_elastic = (π/k²) · g_J · |1 - U_nn|²
        let one_minus_u = Complex64::new(1.0 - u_nn.re, -u_nn.im);
        let sigma_el = pi_over_k2 * g_j * one_minus_u.norm_sqr();
        elastic += sigma_el;

        // σ_total is the sum of the channel components (NOT the optical
        // theorem).  The optical theorem σ_total = 2π/k² · g · (1 - Re(U))
        // holds only for a UNITARY S-matrix.  In MLBW capture and fission
        // remove flux from the elastic channel, so |U| < 1 and the
        // optical theorem overestimates absorption.
    }

    let total = elastic + capture + fission;
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

    /// Verify the SLBW NAPS=0 code path uses the channel radius formula.
    ///
    /// Same structure as the RM `test_naps_zero_uses_channel_radius_formula`:
    /// 1. NAPS=0 with AP = formula_radius matches NAPS=1 with AP = formula_radius
    ///    (confirming the formula gives the expected value)
    /// 2. NAPS=0 with a different AP still produces valid XS (no NaN/panic)
    #[test]
    fn test_slbw_naps_zero_uses_channel_radius_formula() {
        let awr: f64 = 55.345; // Fe-56-like
        let formula_radius = channel::endf_channel_radius_fm(awr);

        // NAPS=0: penetrability uses formula, phase shift uses AP (= formula here)
        let data_naps0 = ResonanceData {
            isotope: nereids_core::types::Isotope::new(26, 56).unwrap(),
            za: 26056,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e5,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: formula_radius, // AP = formula → same as pen_radius
                naps: 0,
                l_groups: vec![LGroup {
                    l: 1,
                    awr,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 30000.0,
                        j: 1.5,
                        gn: 5.0,
                        gg: 1.0,
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

        // NAPS=1 with AP = formula_radius: both penetrability and phase use AP
        let data_naps1 = ResonanceData {
            isotope: nereids_core::types::Isotope::new(26, 56).unwrap(),
            za: 26056,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e5,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: formula_radius,
                naps: 1,
                l_groups: vec![LGroup {
                    l: 1,
                    awr,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 30000.0,
                        j: 1.5,
                        gn: 5.0,
                        gg: 1.0,
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

        let e = 30000.0;
        let xs_naps0 = slbw_cross_sections(&data_naps0, e);
        let xs_naps1 = slbw_cross_sections(&data_naps1, e);

        // When AP equals the formula radius, NAPS=0 and NAPS=1 should
        // give identical results (both use the same radius everywhere).
        assert!(
            (xs_naps0.total - xs_naps1.total).abs() < 1e-10 * xs_naps1.total.abs().max(1.0),
            "NAPS=0 total={} vs NAPS=1 total={}: should match when AP=formula",
            xs_naps0.total,
            xs_naps1.total,
        );
        assert!(
            (xs_naps0.capture - xs_naps1.capture).abs() < 1e-10 * xs_naps1.capture.abs().max(1.0),
            "NAPS=0 capture={} vs NAPS=1 capture={}: should match when AP=formula",
            xs_naps0.capture,
            xs_naps1.capture,
        );

        // Verify finite and positive cross-sections (no NaN from formula)
        assert!(xs_naps0.total.is_finite() && xs_naps0.total > 0.0);
        assert!(xs_naps0.capture.is_finite() && xs_naps0.capture > 0.0);

        // Also verify NAPS=0 with a different AP still produces valid XS
        let data_naps0_diff_ap = ResonanceData {
            isotope: nereids_core::types::Isotope::new(26, 56).unwrap(),
            za: 26056,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e5,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: 9.0, // Different from formula
                naps: 0,
                l_groups: vec![LGroup {
                    l: 1,
                    awr,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 30000.0,
                        j: 1.5,
                        gn: 5.0,
                        gg: 1.0,
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
        let xs_diff = slbw_cross_sections(&data_naps0_diff_ap, e);
        assert!(
            xs_diff.total.is_finite() && xs_diff.total > 0.0,
            "NAPS=0 with different AP should still produce valid XS"
        );
    }

    fn make_mlbw_multi_resonance_data() -> nereids_endf::resonance::ResonanceData {
        use nereids_endf::resonance::*;
        ResonanceData {
            isotope: nereids_core::types::Isotope::new(72, 178).unwrap(),
            za: 72178,
            awr: 177.94,
            ranges: vec![ResonanceRange {
                energy_low: 0.0,
                energy_high: 100.0,
                formalism: ResonanceFormalism::MLBW,
                naps: 0,
                resolved: true,
                scattering_radius: 9.48,
                target_spin: 0.0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 177.94,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![
                        Resonance {
                            energy: 7.8,
                            j: 0.5,
                            gn: 0.002,
                            gg: 0.060,
                            gfa: 0.0,
                            gfb: 0.0,
                        },
                        Resonance {
                            energy: 16.9,
                            j: 0.5,
                            gn: 0.004,
                            gg: 0.055,
                            gfa: 0.0,
                            gfb: 0.0,
                        },
                    ],
                }],
                rml: None,
                ap_table: None,
                urr: None,
                r_external: vec![],
            }],
        }
    }

    /// MLBW cross-sections must be non-negative for multi-resonance data.
    /// Guard: catches e^{+2iφ} phase convention bug (commit 5508fea, fixed f0eadc1).
    #[test]
    fn test_mlbw_multi_resonance_positive() {
        use crate::reich_moore::cross_sections_at_energy;
        let data = make_mlbw_multi_resonance_data();
        for &e in &[1.0, 5.0, 7.0, 7.8, 8.0, 10.0, 12.0, 16.9, 17.0, 20.0, 50.0] {
            let xs = cross_sections_at_energy(&data, e);
            assert!(xs.total >= 0.0, "MLBW total < 0 at E={e}: {:.4}", xs.total);
            assert!(
                xs.elastic >= 0.0,
                "MLBW elastic < 0 at E={e}: {:.4}",
                xs.elastic
            );
        }
    }

    /// Total = elastic + capture + fission for MLBW.
    /// Guard: catches optical theorem misuse (U not unitary for MLBW).
    #[test]
    fn test_mlbw_total_equals_components() {
        use crate::reich_moore::cross_sections_at_energy;
        let data = make_mlbw_multi_resonance_data();
        for &e in &[1.0, 5.0, 7.8, 10.0, 50.0] {
            let xs = cross_sections_at_energy(&data, e);
            let sum = xs.elastic + xs.capture + xs.fission;
            assert!(
                (xs.total - sum).abs() < 1e-10,
                "MLBW total ({:.6}) != el+cap+fis ({:.6}) at E={e}",
                xs.total,
                sum
            );
        }
    }
}
