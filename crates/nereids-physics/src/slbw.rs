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

use nereids_endf::resonance::ResonanceData;

use crate::channel;
use crate::penetrability;

/// SLBW cross-section results at a single energy.
#[derive(Debug, Clone, Copy)]
pub struct SlbwCrossSections {
    pub total: f64,
    pub elastic: f64,
    pub capture: f64,
    pub fission: f64,
}

/// Compute SLBW cross-sections at a single energy.
///
/// Works with both SLBW and Reich-Moore ENDF data (uses the same
/// resonance parameters but applies the SLBW formulas).
pub fn slbw_cross_sections(data: &ResonanceData, energy_ev: f64) -> SlbwCrossSections {
    let awr = data.awr;

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

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

    SlbwCrossSections {
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

        let channel_radius = if l_group.apl > 0.0 {
            l_group.apl
        } else {
            range.scattering_radius_at(energy_ev)
        };

        let rho = channel::rho(energy_ev, awr_l, channel_radius);
        let phi = penetrability::phase_shift(l, rho);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let sin2_phi = sin_phi * sin_phi;

        // Group resonances by J.
        let j_groups = group_by_j(&l_group.resonances);

        for (j_total, resonances) in &j_groups {
            let g_j = channel::statistical_weight(*j_total, target_spin);

            // Potential scattering for this (l, J) group.
            // Added once per spin group.
            let pot_scatter = pi_over_k2 * g_j * 4.0 * sin2_phi;
            elastic += pot_scatter;
            total += pot_scatter;

            for res in resonances {
                let e_r = res.energy;
                let gamma_g = res.gg;
                let gamma_f = res.gfa.abs() + res.gfb.abs();

                // Energy-dependent neutron width:
                // Γ_n(E) = Γ_n(E_r) × √(E/E_r) × P_l(E)/P_l(E_r)
                // P_l(E_r) uses the channel radius evaluated at the resonance
                // energy (ENDF §2.2.1 NRO=1: AP is tabulated, not constant).
                let gamma_n = if e_r.abs() > 1e-30 {
                    let radius_at_er = if l_group.apl > 0.0 {
                        l_group.apl
                    } else {
                        range.scattering_radius_at(e_r.abs())
                    };
                    let rho_r = channel::rho(e_r.abs(), awr_l, radius_at_er);
                    let p_at_e = penetrability::penetrability(l, rho);
                    let p_at_er = penetrability::penetrability(l, rho_r);
                    if p_at_er > 1e-30 {
                        res.gn.abs() * (energy_ev / e_r.abs()).sqrt() * p_at_e / p_at_er
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let gamma_total = gamma_n + gamma_g + gamma_f;
                let de = energy_ev - e_r;
                let denom = de * de + (gamma_total / 2.0).powi(2);

                if denom < 1e-50 {
                    continue;
                }

                // Capture cross-section (symmetric Breit-Wigner).
                let sigma_c = pi_over_k2 * g_j * gamma_n * gamma_g / denom;
                capture += sigma_c;
                total += sigma_c;

                // Fission cross-section.
                let sigma_f = pi_over_k2 * g_j * gamma_n * gamma_f / denom;
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

fn group_by_j(
    resonances: &[nereids_endf::resonance::Resonance],
) -> Vec<(f64, Vec<&nereids_endf::resonance::Resonance>)> {
    let mut groups: Vec<(f64, Vec<&nereids_endf::resonance::Resonance>)> = Vec::new();
    for res in resonances {
        let j = res.j;
        if let Some(group) = groups.iter_mut().find(|(gj, _)| (*gj - j).abs() < 1e-10) {
            group.1.push(res);
        } else {
            groups.push((j, vec![res]));
        }
    }
    groups
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
            isotope: nereids_core::types::Isotope::new(92, 238),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
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
            isotope: nereids_core::types::Isotope::new(92, 238),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                l_groups: resonances.clone(),
                rml: None,
                urr: None,
                ap_table: None,
            }],
        };

        let slbw_data = ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::SLBW,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                l_groups: resonances,
                rml: None,
                urr: None,
                ap_table: None,
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
