//! Regression test for the SLBW/MLBW energy-dependent neutron-width formula.
//!
//! Per ENDF-6 §D.1.1 equation D.7 (ENDF-102 Formats Manual, IAEA public PDF
//! page 357), the SLBW/MLBW neutron width has the energy dependence
//!
//!   Γ_n(E) = Γ_n(E_r) × P_l(E) / P_l(E_r)
//!
//! with NO additional √(E/E_r) multiplier. For neutral s-wave (l=0),
//! P_0(ρ)=ρ ∝ √E so the penetrability ratio P_0(E)/P_0(E_r) = √(E/E_r)
//! already supplies the full √E low-energy velocity factor. Multiplying
//! by a separate √(E/E_r) would yield Γ_n ∝ E (and σ_capture ≈ constant
//! in the 1/v limit) instead of Γ_n ∝ √E (and σ_capture ∝ 1/v).
//!
//! Reference implementations:
//! - NJOY/RECONR `src/reconr.f90` `csslbw`/`csmlbw`: `gne = gn*pe*rper`.
//! - SAMMY `mlb/mmlb4.f90:88-100` `Abpart_Mlb` (via `γ_n² = GN/(2·P_l(E_r))`
//!   from `new/mnew3.f90:307-339` `Betset`).
//!
//! These tests pin the corrected formula and would have caught the
//! double-counted velocity factor on day one of the SLBW evaluator's
//! existence.

use nereids_core::types::Isotope;
use nereids_endf::resonance::{
    LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
};
use nereids_physics::reich_moore;
use nereids_physics::slbw::slbw_cross_sections;

/// Single isolated s-wave U-238 6.674 eV resonance, SLBW formalism.
///
/// I = 0 so g_J = 1 for J = 1/2. Naps = 1 (use AP for everything).
fn u238_slbw_single_resonance() -> ResonanceData {
    ResonanceData {
        isotope: Isotope::new(92, 238).unwrap(),
        za: 92238,
        awr: 236.006,
        ranges: vec![ResonanceRange {
            energy_low: 1e-6,
            energy_high: 1e5,
            resolved: true,
            formalism: ResonanceFormalism::SLBW,
            target_spin: 0.0,
            scattering_radius: 9.4285,
            naps: 1,
            ap_table: None,
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
            r_external: vec![],
        }],
    }
}

/// Same isolated resonance but as Reich-Moore. This is the "ground-truth"
/// formalism — its width formula is independently implemented and known
/// to match SAMMY/NJOY for isolated resonances.
fn u238_rm_single_resonance() -> ResonanceData {
    let mut data = u238_slbw_single_resonance();
    data.ranges[0].formalism = ResonanceFormalism::ReichMoore;
    data
}

/// At E = E_r the penetrability ratio is 1 by construction, so Γ_n(E_r) = GN.
/// Below the resonance, in the 1/v regime, capture must scale as 1/√E for an
/// s-wave neutron width Γ_n ∝ √E. With the buggy formula Γ_n ∝ E, the
/// 1/v slope is wiped out and capture becomes ~constant below the peak.
///
/// Test: σ_cap(E/4) / σ_cap(E) ≈ 2 at energies well below the resonance.
/// Buggy formula would give a ratio ≈ 1.
#[test]
fn slbw_capture_scales_as_one_over_v_below_resonance() {
    let data = u238_slbw_single_resonance();

    // Use thermal-energy probes far below the 6.674 eV resonance, where
    // the resonant denominator is dominated by (E-E_r)² ≈ E_r², so the
    // capture cross-section is essentially Γ_n(E) · Γ_γ × (π/k²) / E_r².
    // Both Γ_n ∝ √E and (π/k²) ∝ 1/E contribute to the 1/v low-E slope.
    let e_lo = 0.001_f64; // 1 meV
    let e_hi = 4.0 * e_lo; // 4 meV

    let xs_lo = slbw_cross_sections(&data, e_lo);
    let xs_hi = slbw_cross_sections(&data, e_hi);

    // For Γ_n ∝ √E and π/k² ∝ 1/E:
    //   σ_cap(E) ∝ Γ_n(E) / E ∝ √E / E = 1/√E (1/v)
    // so σ_cap(E_lo) / σ_cap(E_hi) ≈ √(E_hi / E_lo) = √4 = 2.
    let ratio = xs_lo.capture / xs_hi.capture;
    assert!(
        (ratio - 2.0).abs() < 0.02,
        "σ_cap(E/4)/σ_cap(E) = {ratio:.4} (expected ≈ 2.0 for 1/v); \
         a ratio near 1 indicates the double-counted velocity factor \
         (Γ_n ∝ E) is back."
    );
}

/// Γ_n(E) at E = E_r/4 must equal Γ_n(E_r)/2 for an s-wave resonance,
/// not Γ_n(E_r)/4. The capture cross-section is proportional to Γ_n at
/// fixed (π/k², g_J, denominator), so:
///
///   σ_cap(E_r/4) × (E_r/4) / [σ_cap(E_r) × E_r]  · ratio of denominators
///
/// gives a direct probe of Γ_n's energy scaling.
///
/// Here we compare the SLBW evaluator against Reich-Moore on the same
/// isolated s-wave resonance. Both formalisms apply Γ_n(E) = GN·P_l(E)/P_l(E_r);
/// the formula difference between them is only in the elastic
/// channel (resonance-resonance interference, RM uses U-matrix), so for
/// a SINGLE isolated resonance with capture-dominated kernel they must
/// agree on σ_cap below resonance to within Reich-Moore's resonance-mixing
/// terms (which vanish for one resonance).
///
/// With the buggy double-counted velocity factor, SLBW capture was off
/// by a factor of √(E/E_r) below the peak.
#[test]
fn slbw_matches_reich_moore_capture_below_resonance() {
    let slbw_data = u238_slbw_single_resonance();
    let rm_data = u238_rm_single_resonance();

    // Probe at thermal energies (well below the 6.674 eV peak).
    for &e in &[0.0253, 0.1, 0.5, 1.0, 3.0] {
        let slbw = slbw_cross_sections(&slbw_data, e);
        let rm = reich_moore::cross_sections_at_energy(&rm_data, e);
        let denom = rm.capture.abs().max(1e-12);
        let rel = (slbw.capture - rm.capture).abs() / denom;
        assert!(
            rel < 5e-3,
            "SLBW vs Reich-Moore capture disagree at E = {e} eV: \
             SLBW = {} barns, RM = {} barns, rel = {:.4}. \
             Disagreement at this scale below resonance indicates the \
             velocity-factor double-count is still present.",
            slbw.capture,
            rm.capture,
            rel
        );
    }
}

/// MLBW formalism: same width formula as SLBW. Use a synthetic single
/// s-wave resonance and verify 1/v capture below the peak.
fn u238_mlbw_single_resonance() -> ResonanceData {
    let mut data = u238_slbw_single_resonance();
    data.ranges[0].formalism = ResonanceFormalism::MLBW;
    data
}

#[test]
fn mlbw_capture_scales_as_one_over_v_below_resonance() {
    let data = u238_mlbw_single_resonance();
    let e_lo = 0.001_f64;
    let e_hi = 4.0 * e_lo;

    // MLBW is dispatched via reich_moore::cross_sections_at_energy.
    let xs_lo = reich_moore::cross_sections_at_energy(&data, e_lo);
    let xs_hi = reich_moore::cross_sections_at_energy(&data, e_hi);

    let ratio = xs_lo.capture / xs_hi.capture;
    assert!(
        (ratio - 2.0).abs() < 0.02,
        "MLBW σ_cap(E/4)/σ_cap(E) = {ratio:.4} (expected ≈ 2.0 for 1/v); \
         a ratio near 1 indicates the double-counted velocity factor \
         (Γ_n ∝ E) is back in the MLBW path."
    );
}
