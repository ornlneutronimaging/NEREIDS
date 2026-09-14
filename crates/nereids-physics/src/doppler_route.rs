//! Which Doppler route one isotope took, and why.
//!
//! Doppler broadening is two-tier. Tier 1 integrates the free-gas kernel
//! over the resonance equation itself at error-controlled quadrature; tier 2
//! convolves the kernel with a cross-section sampled on the caller's grid,
//! which is exact for the table it is given but loses the area of any line
//! narrower than the grid spacing. The tier is chosen per isotope for the
//! whole grid and is never mixed within one isotope.
//!
//! A tier-2 result is a declared approximation boundary, never a silent
//! substitution, so the reason for it is part of the answer. This module is
//! only the vocabulary for saying so; [`crate::continuous_doppler`] decides.

use std::fmt;

use nereids_endf::resonance::ResonanceFormalism;

/// The Doppler route one isotope took, for the whole grid.
#[derive(Debug, Clone, PartialEq)]
pub enum DopplerRoute {
    /// No broadening was applied: the sample is at 0 K, the one temperature
    /// `DopplerParams` accepts as meaning no kernel at all (it rejects a
    /// negative one). Nothing is sampled and nothing is integrated, so this
    /// is neither tier.
    Unbroadened,
    /// Tier 1: the free-gas kernel integrated over the resonance equation.
    /// Grid-independent by construction.
    Continuous {
        /// Every formalism the grid was evaluated with, in order of first
        /// use. A grid may legitimately span adjacent resolved ranges of
        /// different formalisms, and the disclosure names all of them
        /// rather than only the lowest energy's.
        formalisms: Vec<ResonanceFormalism>,
    },
    /// Tier 2: the kernel convolved with a sampled zero-kelvin table.
    SampledTable {
        /// The first tier-1 condition that failed.
        reason: SampledTableReason,
    },
}

/// Why an isotope took the sampled-table route.
///
/// Each variant names the lowest grid energy at which its condition failed.
/// `#[non_exhaustive]` because a future source of tier-2 routing must not be
/// a breaking change for downstream matchers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SampledTableReason {
    /// The range covering `energy_ev` is not a resolved SLBW/MLBW range
    /// (`Some`), or the source has no tier-1 range at all (`None`).
    Formalism {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Formalism of the covering range, if one covers it.
        formalism: Option<ResonanceFormalism>,
    },
    /// The covering range is resolved SLBW/MLBW but carries no resonances,
    /// so it evaluates to nothing. Its formalism is not what demoted the
    /// isotope, and saying "SLBW formalism" here would read as though SLBW
    /// were a tier-2 formalism.
    EmptyResolvedRange {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Formalism of the empty range.
        formalism: ResonanceFormalism,
    },
    /// The source has a resolved SLBW/MLBW range, but this grid energy lies
    /// outside every range. Typical of an acquisition reaching past the
    /// resolved region, or of the auxiliary grid the resolution function
    /// adds beyond the data.
    GridLeavesResolvedRange {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Lower edge of the nearest resolved range (eV).
        range_low_ev: f64,
        /// Upper edge of the nearest resolved range (eV).
        range_high_ev: f64,
        /// Formalism of that range.
        formalism: ResonanceFormalism,
    },
    /// `√E ≤ 8u`: the thermal support window would fold through zero
    /// energy, where the kernel's reflected term is no longer negligible.
    ThermalWindowFoldsThroughZero {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Free-gas kernel width `u = √(k_B T / A)` in √eV.
        thermal_u: f64,
        /// Formalism of the covering range.
        formalism: ResonanceFormalism,
    },
    /// The thermal window `[(√E−8u)², (√E+8u)²]` is not wholly inside the
    /// range covering `energy_ev`, so the integral would evaluate source
    /// energies with a different range's formalism.
    WindowCrossesRangeBoundary {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Lower edge of the thermal window (eV).
        window_low_ev: f64,
        /// Upper edge of the thermal window (eV).
        window_high_ev: f64,
        /// Lower edge of the covering range (eV).
        range_low_ev: f64,
        /// Upper edge of the covering range (eV).
        range_high_ev: f64,
        /// Formalism of the covering range.
        formalism: ResonanceFormalism,
    },
    /// A second evaluable range overlaps the thermal window. ENDF-6 forbids
    /// overlapping ranges but nothing upstream rejects them, and the
    /// dispatcher sums every range containing a point, so tier 1 refuses
    /// rather than integrate a mixture of formalisms.
    OverlappingRange {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Index into `ResonanceData::ranges` of the overlapping range.
        other_range_index: usize,
        /// Formalism of the covering range.
        formalism: ResonanceFormalism,
    },
    /// A grid energy is not a positive, finite number, so no window can be
    /// placed around it. Rejected up front rather than per energy: NaN
    /// compares false against every other value, so one left in the grid
    /// would mask the genuine lowest failing energy.
    NonPhysicalEnergy {
        /// The offending grid energy (eV).
        energy_ev: f64,
    },
    /// The covering range carries a File-3 (MF=3) background term, which
    /// the resonance equation does not represent.
    File3Background {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Formalism of the covering range.
        formalism: ResonanceFormalism,
    },
}

/// Human name of a formalism, for disclosure lines.
fn formalism_name(formalism: ResonanceFormalism) -> &'static str {
    match formalism {
        ResonanceFormalism::SLBW => "SLBW",
        ResonanceFormalism::MLBW => "MLBW",
        ResonanceFormalism::ReichMoore => "Reich-Moore",
        ResonanceFormalism::RMatrixLimited => "LRF=7 (R-Matrix Limited)",
        ResonanceFormalism::Unresolved => "unresolved (LRU=2)",
        ResonanceFormalism::ScatteringRadiusOnly => "scattering-radius-only (LRU=0)",
    }
}

impl fmt::Display for DopplerRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DopplerRoute::Unbroadened => write!(f, "no Doppler broadening (temperature 0 K)"),
            DopplerRoute::Continuous { formalisms } => {
                let names: Vec<&str> = formalisms.iter().map(|&x| formalism_name(x)).collect();
                match names.len() {
                    0 => write!(
                        f,
                        "continuous free-gas integral over the resonance equation"
                    ),
                    1 => write!(
                        f,
                        "continuous free-gas integral over the {} resonance equation",
                        names[0]
                    ),
                    _ => write!(
                        f,
                        "continuous free-gas integral over the {} and {} resonance equations",
                        names[..names.len() - 1].join(", "),
                        names[names.len() - 1],
                    ),
                }
            }
            DopplerRoute::SampledTable { reason } => {
                write!(f, "sampled-table kernel-on-grid ({reason})")
            }
        }
    }
}

impl fmt::Display for SampledTableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Formalism {
                formalism: Some(formalism),
                ..
            } => write!(f, "{} formalism", formalism_name(*formalism)),
            Self::Formalism {
                energy_ev,
                formalism: None,
            } => write!(f, "no evaluable resolved range at {energy_ev:.2e} eV"),
            Self::EmptyResolvedRange {
                energy_ev,
                formalism,
            } => write!(
                f,
                "resolved {} range with no resonances at {energy_ev:.2e} eV",
                formalism_name(*formalism)
            ),
            Self::GridLeavesResolvedRange {
                energy_ev,
                range_low_ev,
                range_high_ev,
                formalism,
            } => write!(
                f,
                "grid energy {energy_ev:.2e} eV lies outside the resolved {} range \
                 [{range_low_ev:.2e}, {range_high_ev:.2e}] eV",
                formalism_name(*formalism)
            ),
            Self::ThermalWindowFoldsThroughZero {
                energy_ev,
                thermal_u,
                ..
            } => write!(
                f,
                "\u{221a}E \u{2264} 8u at {energy_ev:.2e} eV (u = {thermal_u:.2e} \u{221a}eV)"
            ),
            Self::WindowCrossesRangeBoundary {
                window_low_ev,
                window_high_ev,
                range_low_ev,
                range_high_ev,
                ..
            } => {
                let edge = if window_low_ev < range_low_ev {
                    range_low_ev
                } else {
                    range_high_ev
                };
                write!(
                    f,
                    "thermal window [{window_low_ev:.2e}, {window_high_ev:.2e}] eV leaves the \
                     resolved range at {edge:.2e} eV"
                )
            }
            Self::OverlappingRange {
                energy_ev,
                other_range_index,
                ..
            } => write!(
                f,
                "resonance range {other_range_index} overlaps the thermal window at \
                 {energy_ev:.2e} eV"
            ),
            Self::NonPhysicalEnergy { energy_ev } => {
                write!(
                    f,
                    "grid energy {energy_ev:.2e} eV is not positive and finite"
                )
            }
            Self::File3Background { energy_ev, .. } => {
                write!(f, "File-3 background term present at {energy_ev:.2e} eV")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every rendering, against the exact text a user reads. The reasons
    /// differ in wording, not only in payload, because each names a
    /// different physical diagnosis.
    #[test]
    fn every_route_and_reason_renders_its_own_diagnosis() {
        let cases: Vec<(DopplerRoute, &str)> = vec![
            (
                DopplerRoute::Unbroadened,
                "no Doppler broadening (temperature 0 K)",
            ),
            (
                sampled(SampledTableReason::NonPhysicalEnergy {
                    energy_ev: f64::NAN,
                }),
                "sampled-table kernel-on-grid (grid energy NaN eV is not positive and finite)",
            ),
            (
                sampled(SampledTableReason::NonPhysicalEnergy { energy_ev: -5.0 }),
                "sampled-table kernel-on-grid (grid energy -5.00e0 eV is not positive and \
                 finite)",
            ),
            (
                DopplerRoute::Continuous {
                    formalisms: vec![ResonanceFormalism::MLBW],
                },
                "continuous free-gas integral over the MLBW resonance equation",
            ),
            (
                DopplerRoute::Continuous {
                    formalisms: vec![ResonanceFormalism::SLBW, ResonanceFormalism::MLBW],
                },
                "continuous free-gas integral over the SLBW and MLBW resonance equations",
            ),
            (
                sampled(SampledTableReason::Formalism {
                    energy_ev: 1.0,
                    formalism: Some(ResonanceFormalism::ReichMoore),
                }),
                "sampled-table kernel-on-grid (Reich-Moore formalism)",
            ),
            (
                sampled(SampledTableReason::Formalism {
                    energy_ev: 2.5e4,
                    formalism: None,
                }),
                "sampled-table kernel-on-grid (no evaluable resolved range at 2.50e4 eV)",
            ),
            (
                sampled(SampledTableReason::EmptyResolvedRange {
                    energy_ev: 6.674,
                    formalism: ResonanceFormalism::SLBW,
                }),
                "sampled-table kernel-on-grid (resolved SLBW range with no resonances at \
                 6.67e0 eV)",
            ),
            (
                sampled(SampledTableReason::GridLeavesResolvedRange {
                    energy_ev: 255.0,
                    range_low_ev: 1e-5,
                    range_high_ev: 250.0,
                    formalism: ResonanceFormalism::MLBW,
                }),
                "sampled-table kernel-on-grid (grid energy 2.55e2 eV lies outside the resolved \
                 MLBW range [1.00e-5, 2.50e2] eV)",
            ),
            (
                sampled(SampledTableReason::ThermalWindowFoldsThroughZero {
                    energy_ev: 3.1e-3,
                    thermal_u: 1.2e-2,
                    formalism: ResonanceFormalism::SLBW,
                }),
                "sampled-table kernel-on-grid (\u{221a}E \u{2264} 8u at 3.10e-3 eV \
                 (u = 1.20e-2 \u{221a}eV))",
            ),
            (
                sampled(window_crossing(9.99e3, 9.81e3, 1.02e4)),
                "sampled-table kernel-on-grid (thermal window [9.81e3, 1.02e4] eV leaves the \
                 resolved range at 1.00e4 eV)",
            ),
            (
                sampled(SampledTableReason::OverlappingRange {
                    energy_ev: 5.0,
                    other_range_index: 1,
                    formalism: ResonanceFormalism::MLBW,
                }),
                "sampled-table kernel-on-grid (resonance range 1 overlaps the thermal window at \
                 5.00e0 eV)",
            ),
            (
                sampled(SampledTableReason::File3Background {
                    energy_ev: 7.0,
                    formalism: ResonanceFormalism::MLBW,
                }),
                "sampled-table kernel-on-grid (File-3 background term present at 7.00e0 eV)",
            ),
        ];
        for (route, expected) in cases {
            assert_eq!(route.to_string(), expected);
        }
    }

    /// A window leaving the range at the BOTTOM names the lower edge; the
    /// case above names the upper one. One expression decides which, so a
    /// test that only ever crossed the top would not see it reversed.
    #[test]
    fn a_window_crossing_the_lower_edge_names_the_lower_edge() {
        assert_eq!(
            window_crossing(1.5e-5, 5e-6, 3e-5).to_string(),
            "thermal window [5.00e-6, 3.00e-5] eV leaves the resolved range at 1.00e-5 eV"
        );
    }

    fn sampled(reason: SampledTableReason) -> DopplerRoute {
        DopplerRoute::SampledTable { reason }
    }

    fn window_crossing(energy_ev: f64, low: f64, high: f64) -> SampledTableReason {
        SampledTableReason::WindowCrossesRangeBoundary {
            energy_ev,
            window_low_ev: low,
            window_high_ev: high,
            range_low_ev: 1e-5,
            range_high_ev: 1e4,
            formalism: ResonanceFormalism::MLBW,
        }
    }
}
