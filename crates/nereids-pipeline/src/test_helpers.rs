//! Shared test helpers for the nereids-pipeline crate.

use nereids_core::types::Isotope;
use nereids_endf::resonance::{
    LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
};

/// Build a minimal U-238 `ResonanceData` with a single resonance at 6.674 eV.
///
/// Used across pipeline, spatial, and sparse tests to avoid triplicating the
/// same 30-line constructor.
pub fn u238_single_resonance() -> ResonanceData {
    ResonanceData {
        isotope: Isotope::new(92, 238).unwrap(),
        za: 92238,
        awr: 236.006,
        ranges: vec![ResonanceRange {
            energy_low: 1e-5,
            energy_high: 1e4,
            resolved: true,
            formalism: ResonanceFormalism::ReichMoore,
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
            urr: None,
            ap_table: None,
        }],
    }
}
