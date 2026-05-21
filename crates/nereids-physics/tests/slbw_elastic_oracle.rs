//! Independent oracle verification of the SLBW elastic cross-section.
//!
//! This integration test cross-checks `nereids_physics::slbw::slbw_cross_sections`
//! against a direct `|1 − U_nn|²` evaluation, exercising the high-ρ regime that
//! the existing `samtry` validation suite does not reach. It catches sign and
//! factor errors in the resonance-potential interference term that would be
//! numerically invisible at the low-ρ (s-wave, thermal-eV) energies the
//! `samtry` cases sample.
//!
//! Issue #549: SLBW elastic Γ_tot·sin²φ interference sign error.
//!
//! ## SAMMY reference
//! - `mlb/mmlb3.f90:56` — `Elastc_Mlb` SLBW branch:
//!   `Sum += ((1 − Cs)·A + Si·B + Aaathr) · D`
//! - `xxx/mxxx9.f90:68-71` — `Cs2sn2`: confirms `Cs = cos(2φ)`, `Si = sin(2φ)`.
//!
//! ## U-matrix convention
//! Per `nereids-physics/src/slbw.rs:347-349` (the MLBW evaluator's own
//! docstring), NEREIDS uses
//!
//!   U_nn = e^{−2iφ} · [1 + iΓ_n / (E_r − E − iΓ_tot/2)]
//!   σ_el = (π/k²) · g_J · |1 − U_nn|²

use nereids_core::types::Isotope;
use nereids_endf::resonance::{
    LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
};
use nereids_physics::channel::{pi_over_k_squared_barns, rho};
use nereids_physics::penetrability::{penetrability, phase_shift};
use nereids_physics::slbw::slbw_cross_sections;

/// Build a synthetic dataset with a single s-wave SLBW resonance.
///
/// Uses U-238-like target spin (I=0) so g_J = 1 for J = 1/2.
fn synthetic_swave_slbw(
    awr: f64,
    e_r_ev: f64,
    gn_ev: f64,
    gg_ev: f64,
    scattering_radius_fm: f64,
) -> ResonanceData {
    ResonanceData {
        isotope: Isotope::new(92, 238).unwrap(),
        za: 92238,
        awr,
        ranges: vec![ResonanceRange {
            energy_low: 1e-5,
            energy_high: 1e6,
            resolved: true,
            formalism: ResonanceFormalism::SLBW,
            target_spin: 0.0,
            scattering_radius: scattering_radius_fm,
            naps: 1,
            ap_table: None,
            l_groups: vec![LGroup {
                l: 0,
                awr,
                apl: 0.0,
                qx: 0.0,
                lrx: 0,
                resonances: vec![Resonance {
                    energy: e_r_ev,
                    j: 0.5,
                    gn: gn_ev,
                    gg: gg_ev,
                    gfa: 0.0,
                    gfb: 0.0,
                }],
            }],
            rml: None,
            urr: None,
            r_external: vec![],
        }],
    }
}

/// Direct `|1 − U_nn|²` oracle for the elastic cross-section.
///
/// Restricted to the single-s-wave-resonance, I=0, J=1/2 configuration the
/// helper above produces. Uses NEREIDS primitives for ρ, P_l, φ_l, and π/k²
/// so the test isolates the elastic-formula algebra from constant choices.
fn oracle_elastic_swave_single_resonance(data: &ResonanceData, energy_ev: f64) -> f64 {
    let awr = data.awr;
    let range = &data.ranges[0];
    let l_group = &range.l_groups[0];
    let res = &l_group.resonances[0];

    // ρ, P_l, φ_l at incident energy
    let radius_e = range.scattering_radius_at(energy_ev.abs());
    let rho_e = rho(energy_ev, awr, radius_e);
    let p_l_e = penetrability(0, rho_e);
    let phi_e = phase_shift(0, rho_e);

    // P_l at the resonance energy (needed for the SLBW Γ_n(E) scaling)
    let radius_r = range.scattering_radius_at(res.energy.abs());
    let rho_r = rho(res.energy.abs(), awr, radius_r);
    let p_l_r = penetrability(0, rho_r);

    // Γ_n(E) = Γ_n(E_r) · P_l(E)/P_l(E_r) per ENDF-6 §D.1.1 eq D.7.
    // The penetrability ratio already carries the full √E dependence
    // for s-wave; an extra √(E/E_r) multiplier would double-count it.
    let gamma_n = res.gn.abs() * p_l_e / p_l_r;
    let gamma_total = gamma_n + res.gg + res.gfa.abs() + res.gfb.abs();

    let de = energy_ev - res.energy;
    let den = de * de + (gamma_total / 2.0).powi(2);

    // Inner factor: 1 + iΓ_n / (E_r − E − iΓ_tot/2)
    //             = 1 + iΓ_n · (E_r − E + iΓ_tot/2) / Den
    //             = (1 − Γ_n·Γ_tot/(2·Den)) + i · (Γ_n·(E_r − E)/Den)
    let inner_re = 1.0 - gamma_n * gamma_total / (2.0 * den);
    let inner_im = -gamma_n * de / den;

    // e^{−2iφ} = cos(2φ) − i·sin(2φ)
    let c2 = (2.0 * phi_e).cos();
    let s2 = (2.0 * phi_e).sin();

    // U_nn = (c2 − i·s2) · (inner_re + i·inner_im)
    let u_re = c2 * inner_re + s2 * inner_im;
    let u_im = c2 * inner_im - s2 * inner_re;

    // |1 − U_nn|² = (1 − Re U)² + (Im U)²
    let one_minus_u_re = 1.0 - u_re;
    let abs_sq = one_minus_u_re * one_minus_u_re + u_im * u_im;

    let pi_over_k2 = pi_over_k_squared_barns(energy_ev, awr);
    // g_J = (2J+1) / [2(2I+1)] = 2/2 = 1 for I=0, J=1/2
    let g_j = 1.0;

    pi_over_k2 * g_j * abs_sq
}

/// Tight tolerance: the SLBW evaluator and the oracle share the same primitives,
/// so any remaining disagreement is floating-point noise.
const REL_TOL: f64 = 1e-10;

/// High-ρ s-wave at the resonance peak.
///
/// E_r = 25 keV, U-238 mass + 9.43 fm radius gives ρ ≈ 0.33 at peak,
/// so sin²φ ≈ 0.11 and the Γ_tot·sin²φ interference bias is large.
/// With the buggy sign (slbw.rs:309 = `+`), NEREIDS disagrees with the
/// oracle by several percent. With the corrected sign (`-`), agreement
/// is bit-noise.
#[test]
fn slbw_elastic_matches_u_matrix_oracle_high_rho_at_peak() {
    let data = synthetic_swave_slbw(236.006, 25_000.0, 1.0, 1.0, 9.4285);
    let energy_ev = 25_000.0;

    let nereids = slbw_cross_sections(&data, energy_ev);
    let oracle = oracle_elastic_swave_single_resonance(&data, energy_ev);

    let rel_err = (nereids.elastic - oracle).abs() / oracle.abs();
    assert!(
        rel_err < REL_TOL,
        "SLBW elastic disagrees with |1 − U_nn|² oracle at high ρ peak:\n  \
             E       = {} eV\n  \
             NEREIDS = {} barns\n  \
             Oracle  = {} barns\n  \
             abs Δ   = {:.6e}\n  \
             rel err = {:.6e}",
        energy_ev,
        nereids.elastic,
        oracle,
        (nereids.elastic - oracle).abs(),
        rel_err
    );
}

/// High-ρ s-wave off-peak (a few Γ_tot away).
///
/// Off-peak the interference term contributes a larger fraction of the
/// total elastic cross-section than at peak, so this point amplifies the
/// sign-error visibility.
#[test]
fn slbw_elastic_matches_u_matrix_oracle_high_rho_off_peak() {
    let data = synthetic_swave_slbw(236.006, 25_000.0, 1.0, 1.0, 9.4285);
    let energy_ev = 25_010.0; // 10 eV off-peak ≈ 5·Γ_tot

    let nereids = slbw_cross_sections(&data, energy_ev);
    let oracle = oracle_elastic_swave_single_resonance(&data, energy_ev);

    let rel_err = (nereids.elastic - oracle).abs() / oracle.abs();
    assert!(
        rel_err < REL_TOL,
        "SLBW elastic disagrees with |1 − U_nn|² oracle off-peak, high ρ:\n  \
             E       = {} eV\n  \
             NEREIDS = {} barns\n  \
             Oracle  = {} barns\n  \
             rel err = {:.6e}",
        energy_ev,
        nereids.elastic,
        oracle,
        rel_err
    );
}

/// Existing `samtry` regime: low-energy s-wave where ρ ≪ 1.
///
/// Pinned BEFORE the sign fix as a guard that the fix does not regress
/// the regime the samtry suite already validates. Should pass both
/// before and after the fix (the sign bias is below 1e-10 at these
/// energies, well within REL_TOL).
#[test]
fn slbw_elastic_matches_u_matrix_oracle_low_rho_swave() {
    let data = synthetic_swave_slbw(236.006, 6.674, 0.001, 0.025, 9.4285);
    let energy_ev = 6.674;

    let nereids = slbw_cross_sections(&data, energy_ev);
    let oracle = oracle_elastic_swave_single_resonance(&data, energy_ev);

    let rel_err = (nereids.elastic - oracle).abs() / oracle.abs();
    assert!(
        rel_err < REL_TOL,
        "SLBW elastic disagrees with |1 − U_nn|² oracle at low ρ:\n  \
             NEREIDS = {} barns\n  \
             Oracle  = {} barns\n  \
             rel err = {:.6e}",
        nereids.elastic,
        oracle,
        rel_err
    );
}
