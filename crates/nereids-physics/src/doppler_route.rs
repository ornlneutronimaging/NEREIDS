//! Per-isotope Doppler evaluation route and its disclosure.
//!
//! Doppler broadening is two-tier. Tier 1 integrates the free-gas kernel
//! over the resonance equation itself at error-controlled quadrature
//! (`continuous_doppler`); tier 2 convolves the kernel with a
//! zero-kelvin cross-section sampled on a grid (`doppler`). The tier is
//! chosen per isotope for the whole requested grid, never mixed within one
//! isotope, and every result that broadened something reports which tier
//! each isotope took. A sampled-table result is a declared approximation
//! boundary, never a silent substitution, so the reason for it is part of
//! the disclosure.
//!
//! This module holds only the disclosure types and their rendering; the
//! gate that decides the route lives in `continuous_doppler` and the
//! broadeners in `transmission`.

use std::fmt;

use nereids_core::elements::isotope_to_string;
use nereids_core::types::Isotope;
use nereids_endf::resonance::ResonanceFormalism;

/// Which Doppler evaluation route one isotope took, for the whole grid.
///
/// The three tiers are fixed by the physics contract, so the enum is
/// exhaustive: a matcher that forgets one is a bug the compiler should
/// report.
#[derive(Debug, Clone, PartialEq)]
pub enum DopplerRoute {
    /// No Doppler broadening was applied (temperature ≤ 0 K). Nothing is
    /// sampled or convolved, so this is neither tier.
    Unbroadened,
    /// Tier 1: the free-gas kernel integrated over the resonance equation
    /// at error-controlled quadrature. Grid-independent by construction.
    /// Only resolved SLBW and MLBW sources qualify.
    Continuous {
        /// Formalism of the resolved range the whole grid lies in.
        formalism: ResonanceFormalism,
    },
    /// Tier 2: kernel-on-grid convolution of a zero-kelvin table sampled
    /// on the working grid. Accuracy depends on the grid resolving every
    /// line inside each thermal window, which is why the route is disclosed.
    SampledTable {
        /// The first tier-1 condition that failed, or the caller's choice.
        reason: SampledTableReason,
    },
}

/// Why an isotope took the sampled-table route.
///
/// Gate reasons name the lowest grid energy at which the condition failed
/// and the formalism of the covering range where one exists. The enum is
/// `#[non_exhaustive]` because a future source of tier-2 routing (a File-3
/// term that is actually parsed, say) must not be a SemVer break for
/// downstream matchers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SampledTableReason {
    /// The caller supplied a zero-kelvin table. The engine never evaluated
    /// the resonance source, so no tier-1 condition was tested.
    ExplicitTable,
    /// The range covering `energy_ev` is not a resolved SLBW/MLBW range
    /// (`Some`), or no evaluable range covers `energy_ev` at all (`None`).
    Formalism {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Formalism of the covering range, if any.
        formalism: Option<ResonanceFormalism>,
    },
    /// `√E ≤ 8u` at `energy_ev`: the thermal support window would fold
    /// through zero energy.
    ThermalWindowFoldsThroughZero {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Free-gas kernel width `u = √(k_B T / A)` in √eV.
        thermal_u: f64,
        /// Formalism of the resolved range covering `energy_ev`.
        formalism: ResonanceFormalism,
    },
    /// The thermal window `[(√E − 8u)², (√E + 8u)²]` at `energy_ev` is not
    /// wholly inside the resolved range covering `energy_ev`.
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
    /// A second evaluable range overlaps the thermal window at `energy_ev`.
    /// ENDF-6 forbids overlapping ranges, but the parser does not validate
    /// it and the dispatcher sums every range containing a point, so tier 1
    /// refuses rather than integrate a mixture of formalisms.
    OverlappingRange {
        /// Grid energy at which the condition failed (eV).
        energy_ev: f64,
        /// Index into `ResonanceData::ranges` of the overlapping range.
        other_range_index: usize,
        /// Formalism of the covering range.
        formalism: ResonanceFormalism,
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

/// One isotope's route, labelled for disclosure.
///
/// Working-grid results align routes positionally with the resonance data;
/// the label is attached once, at the result boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct IsotopeDopplerRoute {
    /// The isotope the route belongs to.
    pub isotope: Isotope,
    /// The route it took.
    pub route: DopplerRoute,
}

impl DopplerRoute {
    /// Whether two routes are the same tier for the same kind of reason.
    ///
    /// Gate reasons carry the energy at which the condition failed (and the
    /// window or kernel width there); those describe one grid and move
    /// when the grid moves. A caller that must hold the route fixed across
    /// grids — the energy-scale model, whose working grid follows every
    /// `(t0, L_scale)` probe — compares kinds, not the reported energies.
    pub fn same_kind(&self, other: &Self) -> bool {
        match (self, other) {
            (DopplerRoute::Unbroadened, DopplerRoute::Unbroadened) => true,
            (
                DopplerRoute::Continuous { formalism: a },
                DopplerRoute::Continuous { formalism: b },
            ) => a == b,
            (
                DopplerRoute::SampledTable { reason: a },
                DopplerRoute::SampledTable { reason: b },
            ) => a.same_kind(b),
            _ => false,
        }
    }

    /// True when a resolved SLBW/MLBW isotope fell to tier 2 for a reason
    /// other than its formalism or the caller's explicit table.
    ///
    /// This is the one case where a reader of "SLBW/MLBW takes the
    /// continuous route" would be surprised, so it is the case that earns a
    /// line in the fit result's `warnings`. A Reich-Moore isotope on the
    /// sampled-table route is the documented behaviour and does not warn.
    pub fn is_edge_fallback(&self) -> bool {
        match self {
            DopplerRoute::Unbroadened | DopplerRoute::Continuous { .. } => false,
            DopplerRoute::SampledTable { reason } => !matches!(
                reason,
                SampledTableReason::ExplicitTable | SampledTableReason::Formalism { .. }
            ),
        }
    }
}

impl SampledTableReason {
    /// Same variant and, where the variant carries one, the same formalism
    /// or overlapping-range index; the energy at which the condition failed
    /// and the window or kernel width there are not compared (see
    /// [`DopplerRoute::same_kind`]).
    pub fn same_kind(&self, other: &Self) -> bool {
        use SampledTableReason as R;
        match (self, other) {
            (R::ExplicitTable, R::ExplicitTable) => true,
            (R::Formalism { formalism: a, .. }, R::Formalism { formalism: b, .. }) => a == b,
            (
                R::ThermalWindowFoldsThroughZero { formalism: a, .. },
                R::ThermalWindowFoldsThroughZero { formalism: b, .. },
            )
            | (
                R::WindowCrossesRangeBoundary { formalism: a, .. },
                R::WindowCrossesRangeBoundary { formalism: b, .. },
            )
            | (R::File3Background { formalism: a, .. }, R::File3Background { formalism: b, .. }) => {
                a == b
            }
            (
                R::OverlappingRange {
                    other_range_index: i,
                    formalism: a,
                    ..
                },
                R::OverlappingRange {
                    other_range_index: j,
                    formalism: b,
                    ..
                },
            ) => i == j && a == b,
            _ => false,
        }
    }
}

/// The `warnings` line for an edge fallback, or `None` for every other route.
///
/// `gate_temperature_k` is the temperature at which the route was decided:
/// the fit temperature for a fixed-temperature evaluation, the fit's upper
/// bound for a free-temperature fit. Naming it explains a demotion at the
/// bound of an isotope that would qualify at room temperature.
pub fn edge_fallback_warning(
    route: &IsotopeDopplerRoute,
    gate_temperature_k: f64,
) -> Option<String> {
    let DopplerRoute::SampledTable { reason } = &route.route else {
        return None;
    };
    let formalism = match reason {
        SampledTableReason::ExplicitTable | SampledTableReason::Formalism { .. } => return None,
        SampledTableReason::ThermalWindowFoldsThroughZero { formalism, .. }
        | SampledTableReason::WindowCrossesRangeBoundary { formalism, .. }
        | SampledTableReason::OverlappingRange { formalism, .. }
        | SampledTableReason::File3Background { formalism, .. } => *formalism,
    };
    Some(format!(
        "Doppler: {} took the sampled-table route although it is resolved {}: {reason} \
         (route gate at {gate_temperature_k} K)",
        isotope_to_string(&route.isotope),
        formalism_name(formalism),
    ))
}

/// Human name of a formalism for disclosure lines.
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
            DopplerRoute::Continuous { formalism } => write!(
                f,
                "continuous free-gas integral over the {} resonance equation",
                formalism_name(*formalism)
            ),
            DopplerRoute::SampledTable { reason } => {
                write!(f, "sampled-table kernel-on-grid ({reason})")
            }
        }
    }
}

impl fmt::Display for SampledTableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SampledTableReason::ExplicitTable => write!(f, "caller-supplied zero-kelvin table"),
            SampledTableReason::Formalism {
                formalism: Some(formalism),
                ..
            } => write!(f, "{} formalism", formalism_name(*formalism)),
            SampledTableReason::Formalism {
                energy_ev,
                formalism: None,
            } => write!(f, "no evaluable resolved range at {energy_ev:.2e} eV"),
            SampledTableReason::ThermalWindowFoldsThroughZero {
                energy_ev,
                thermal_u,
                ..
            } => write!(
                f,
                "\u{221a}E \u{2264} 8u at {energy_ev:.2e} eV (u = {thermal_u:.2e} \u{221a}eV)"
            ),
            SampledTableReason::WindowCrossesRangeBoundary {
                window_low_ev,
                window_high_ev,
                range_low_ev,
                range_high_ev,
                ..
            } => {
                let edge = if *window_low_ev < *range_low_ev {
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
            SampledTableReason::OverlappingRange {
                energy_ev,
                other_range_index,
                ..
            } => write!(
                f,
                "resonance range {other_range_index} overlaps the thermal window at \
                 {energy_ev:.2e} eV"
            ),
            SampledTableReason::File3Background { energy_ev, .. } => {
                write!(f, "File-3 background term present at {energy_ev:.2e} eV")
            }
        }
    }
}

impl fmt::Display for IsotopeDopplerRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", isotope_to_string(&self.isotope), self.route)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hf177() -> Isotope {
        Isotope::new(72, 177).unwrap()
    }

    fn window_reason() -> SampledTableReason {
        SampledTableReason::WindowCrossesRangeBoundary {
            energy_ev: 9.99e3,
            window_low_ev: 9.81e3,
            window_high_ev: 1.02e4,
            range_low_ev: 1e-5,
            range_high_ev: 1e4,
            formalism: ResonanceFormalism::MLBW,
        }
    }

    #[test]
    fn display_continuous_names_element_symbol_and_formalism() {
        let route = IsotopeDopplerRoute {
            isotope: hf177(),
            route: DopplerRoute::Continuous {
                formalism: ResonanceFormalism::MLBW,
            },
        };
        assert_eq!(
            route.to_string(),
            "Hf-177: continuous free-gas integral over the MLBW resonance equation"
        );
    }

    #[test]
    fn display_unbroadened_and_explicit_table() {
        assert_eq!(
            DopplerRoute::Unbroadened.to_string(),
            "no Doppler broadening (temperature 0 K)"
        );
        assert_eq!(
            DopplerRoute::SampledTable {
                reason: SampledTableReason::ExplicitTable
            }
            .to_string(),
            "sampled-table kernel-on-grid (caller-supplied zero-kelvin table)"
        );
    }

    #[test]
    fn display_each_gate_reason() {
        let cases: Vec<(SampledTableReason, &str)> = vec![
            (
                SampledTableReason::Formalism {
                    energy_ev: 1.0,
                    formalism: Some(ResonanceFormalism::ReichMoore),
                },
                "Reich-Moore formalism",
            ),
            (
                SampledTableReason::Formalism {
                    energy_ev: 2.5e4,
                    formalism: None,
                },
                "no evaluable resolved range at 2.50e4 eV",
            ),
            (
                SampledTableReason::ThermalWindowFoldsThroughZero {
                    energy_ev: 3.1e-3,
                    thermal_u: 1.2e-2,
                    formalism: ResonanceFormalism::SLBW,
                },
                "\u{221a}E \u{2264} 8u at 3.10e-3 eV (u = 1.20e-2 \u{221a}eV)",
            ),
            (
                window_reason(),
                "thermal window [9.81e3, 1.02e4] eV leaves the resolved range at 1.00e4 eV",
            ),
            (
                SampledTableReason::OverlappingRange {
                    energy_ev: 5.0,
                    other_range_index: 1,
                    formalism: ResonanceFormalism::MLBW,
                },
                "resonance range 1 overlaps the thermal window at 5.00e0 eV",
            ),
            (
                SampledTableReason::File3Background {
                    energy_ev: 7.0,
                    formalism: ResonanceFormalism::MLBW,
                },
                "File-3 background term present at 7.00e0 eV",
            ),
        ];
        for (reason, expected) in cases {
            assert_eq!(reason.to_string(), expected);
            assert_eq!(
                DopplerRoute::SampledTable { reason }.to_string(),
                format!("sampled-table kernel-on-grid ({expected})")
            );
        }
    }

    #[test]
    fn window_leaving_below_names_the_lower_edge() {
        let reason = SampledTableReason::WindowCrossesRangeBoundary {
            energy_ev: 1.5e-5,
            window_low_ev: 5e-6,
            window_high_ev: 3e-5,
            range_low_ev: 1e-5,
            range_high_ev: 1e4,
            formalism: ResonanceFormalism::SLBW,
        };
        assert_eq!(
            reason.to_string(),
            "thermal window [5.00e-6, 3.00e-5] eV leaves the resolved range at 1.00e-5 eV"
        );
    }

    #[test]
    fn unknown_element_falls_back_to_z_label() {
        let route = IsotopeDopplerRoute {
            isotope: Isotope::new(150, 300).unwrap(),
            route: DopplerRoute::Unbroadened,
        };
        assert!(route.to_string().starts_with("Z150-300: "));
    }

    #[test]
    fn is_edge_fallback_only_for_window_reasons() {
        assert!(!DopplerRoute::Unbroadened.is_edge_fallback());
        assert!(
            !DopplerRoute::Continuous {
                formalism: ResonanceFormalism::SLBW
            }
            .is_edge_fallback()
        );
        assert!(
            !DopplerRoute::SampledTable {
                reason: SampledTableReason::ExplicitTable
            }
            .is_edge_fallback()
        );
        assert!(
            !DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    energy_ev: 1.0,
                    formalism: Some(ResonanceFormalism::ReichMoore),
                }
            }
            .is_edge_fallback()
        );
        for reason in [
            window_reason(),
            SampledTableReason::ThermalWindowFoldsThroughZero {
                energy_ev: 1e-3,
                thermal_u: 0.1,
                formalism: ResonanceFormalism::MLBW,
            },
            SampledTableReason::OverlappingRange {
                energy_ev: 1.0,
                other_range_index: 2,
                formalism: ResonanceFormalism::MLBW,
            },
            SampledTableReason::File3Background {
                energy_ev: 1.0,
                formalism: ResonanceFormalism::SLBW,
            },
        ] {
            assert!(DopplerRoute::SampledTable { reason }.is_edge_fallback());
        }
    }

    #[test]
    fn edge_fallback_warning_names_isotope_formalism_reason_and_gate() {
        let route = IsotopeDopplerRoute {
            isotope: hf177(),
            route: DopplerRoute::SampledTable {
                reason: window_reason(),
            },
        };
        assert_eq!(
            edge_fallback_warning(&route, 5000.0).as_deref(),
            Some(
                "Doppler: Hf-177 took the sampled-table route although it is resolved MLBW: \
                 thermal window [9.81e3, 1.02e4] eV leaves the resolved range at 1.00e4 eV \
                 (route gate at 5000 K)"
            )
        );
        let fixed = edge_fallback_warning(&route, 293.6).unwrap();
        assert!(fixed.ends_with("(route gate at 293.6 K)"));
    }

    #[test]
    fn no_warning_for_non_edge_routes() {
        for route in [
            DopplerRoute::Unbroadened,
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::MLBW,
            },
            DopplerRoute::SampledTable {
                reason: SampledTableReason::ExplicitTable,
            },
            DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    energy_ev: 1.0,
                    formalism: Some(ResonanceFormalism::ReichMoore),
                },
            },
        ] {
            let labelled = IsotopeDopplerRoute {
                isotope: hf177(),
                route,
            };
            assert_eq!(edge_fallback_warning(&labelled, 5000.0), None);
        }
    }

    #[test]
    fn same_kind_ignores_the_reported_energy_but_not_the_reason_or_formalism() {
        let rm_at = |energy_ev: f64| DopplerRoute::SampledTable {
            reason: SampledTableReason::Formalism {
                energy_ev,
                formalism: Some(ResonanceFormalism::ReichMoore),
            },
        };
        assert!(rm_at(4.0).same_kind(&rm_at(4.1)));
        assert_ne!(rm_at(4.0), rm_at(4.1));
        assert!(!rm_at(4.0).same_kind(&DopplerRoute::SampledTable {
            reason: SampledTableReason::Formalism {
                energy_ev: 4.0,
                formalism: None,
            },
        }));

        let mut shifted = window_reason();
        if let SampledTableReason::WindowCrossesRangeBoundary {
            energy_ev,
            window_low_ev,
            window_high_ev,
            ..
        } = &mut shifted
        {
            *energy_ev += 0.5;
            *window_low_ev += 0.5;
            *window_high_ev += 0.5;
        }
        assert!(window_reason().same_kind(&shifted));
        let slbw_window = SampledTableReason::WindowCrossesRangeBoundary {
            energy_ev: 9.99e3,
            window_low_ev: 9.81e3,
            window_high_ev: 1.02e4,
            range_low_ev: 1e-5,
            range_high_ev: 1e4,
            formalism: ResonanceFormalism::SLBW,
        };
        assert!(!window_reason().same_kind(&slbw_window));
        assert!(!window_reason().same_kind(&SampledTableReason::ExplicitTable));

        let continuous = |formalism| DopplerRoute::Continuous { formalism };
        assert!(
            continuous(ResonanceFormalism::MLBW).same_kind(&continuous(ResonanceFormalism::MLBW))
        );
        assert!(
            !continuous(ResonanceFormalism::MLBW).same_kind(&continuous(ResonanceFormalism::SLBW))
        );
        assert!(!continuous(ResonanceFormalism::MLBW).same_kind(&DopplerRoute::Unbroadened));
        assert!(DopplerRoute::Unbroadened.same_kind(&DopplerRoute::Unbroadened));
    }
}
