//! The tier-1 Doppler gate: which isotopes may be broadened by integrating
//! the free-gas kernel over the resonance equation.
//!
//! ## The condition
//!
//! With `v = √E` the neutron speed in √eV and `u = √(k_B T / A)` the thermal
//! width in the same units, the free-gas kernel (SAMMY manual Sec. III.B.1)
//! carries a direct term in `√E − √E′` and a reflected term in `√E + √E′`.
//! Dropping the reflected term leaves a single Gaussian in `√E`, which is
//! what an integral over the resonance equation can evaluate directly. The
//! reflected term is below `exp(−64)` exactly when `√E > 8u`, which is also
//! the condition under which the truncated window `|x| ≤ 8` contains only
//! positive source energies. That one inequality is why [`SUPPORT_X`] is 8
//! and why it appears in both the gate and the window.
//!
//! The remaining conditions exist because the integral evaluates the
//! resonance equation at source energies spread across the whole window, not
//! only at the target: every one of those energies must be governed by the
//! same resolved SLBW or MLBW range, or the integral would silently mix
//! formalisms, or integrate a range whose parameters do not describe the
//! source there.
//!
//! One tier-1 condition of the contract is NOT implemented here. A range
//! carrying a File-3 (MF=3) smooth background must take the sampled-table
//! tier, because the resonance equation does not represent that background.
//! Only File 2 is parsed, so nothing can answer whether a range has one, and
//! a gate that cannot see the data cannot enforce it. Whichever change adds
//! MF=3 parsing owes this condition; until then an evaluation carrying a
//! File-3 background would be integrated without it.
//!
//! ## Why the verdict is per isotope and all-or-nothing
//!
//! Mixing tiers within one isotope would make the reported cross-section a
//! function of where in the grid each point happened to fall. The gate
//! therefore reports the first failing condition at the lowest failing
//! energy and demotes the whole isotope.
//!
//! The verdict does depend on the grid's EXTENT: a grid reaching past the
//! resolved region asks a different question from one that stops inside it,
//! and answering both the same way would hide the reach. The reported
//! reason is the LOWEST failing energy, which the ascending-grid contract
//! makes the same as the first one reached.
//!
//! Nothing in the workspace calls this yet; the integral it guards arrives
//! separately.

use nereids_endf::resonance::{ResonanceData, ResonanceFormalism, ResonanceRange};

use crate::doppler::{DopplerError, DopplerParams, validate_doppler_grid};
use crate::doppler_route::{DopplerRoute, SampledTableReason};
use crate::reich_moore::{covers, upper_bound_is_half_open};

/// Half-width of the kernel support in units of `u`, so the thermal window
/// is `[(√E − 8u)², (√E + 8u)²]`. `erfc(8) ≈ 1.1e-29` of the kernel mass
/// lies outside it, far below any tolerance the integral works to.
pub const SUPPORT_X: f64 = 8.0;

/// The Doppler route of one isotope over `work_energies` at `temperature_k`.
///
/// The kernel width `u = √(k_B T / A)` is built here from the source's OWN
/// `awr`, so it cannot be computed for a different nuclide than the one
/// being routed: the same grid and temperature against AWR 1 rather than
/// AWR 236 moves `8u` by a factor of 15 and flips the verdict.
///
/// The verdict covers the whole grid: the lowest energy that fails a tier-1
/// condition demotes the isotope and names the reason. Conditions are tested
/// in a fixed order so the reported reason is deterministic rather than an
/// artefact of which check happened to run first.
///
/// # Errors
///
/// Returns [`DopplerError`] when the temperature or the source's `awr` is
/// not a valid [`DopplerParams`], or when `work_energies` is empty or is not
/// the grid broadening requires — finite, strictly positive and strictly
/// ascending.
/// An unroutable input is an error and not a route: the sampled-table tier
/// rejects exactly these grids too, so reporting one as tier 2 would send
/// the caller down a path that cannot run.
pub fn classify_isotope(
    data: &ResonanceData,
    work_energies: &[f64],
    temperature_k: f64,
) -> Result<DopplerRoute, DopplerError> {
    let params = DopplerParams::new(temperature_k, data.awr)?;
    if work_energies.is_empty() {
        return Err(DopplerError::EmptyGrid);
    }
    // The same contract the broadening entry points enforce, checked by the
    // same function: an energy this rejects cannot be routed by EITHER tier,
    // so it is an error rather than a reason to prefer the sampled table.
    // It also leaves the reduction below free of values it cannot order —
    // NaN compares false against everything, so one left in the grid would
    // pin the reported reason to itself.
    validate_doppler_grid(work_energies)?;
    // At absolute zero there is no kernel to apply by either route, so the
    // tier question does not arise. `DopplerParams` rejects a negative
    // temperature outright, which is why this tests for equality.
    if params.temperature_k() == 0.0 {
        return Ok(DopplerRoute::Unbroadened);
    }
    let thermal_u = params.u();
    // Every formalism the grid was evaluated with, first use first. A grid
    // may legitimately span adjacent resolved ranges of different
    // formalisms; the disclosed route is the executed route, so it names
    // all of them rather than the lowest energy's alone. The grid is
    // non-empty and strictly ascending by the contract checked above, so
    // "first use" is energy order and at least one entry is produced.
    let mut formalisms: Vec<ResonanceFormalism> = Vec::new();
    for &energy in work_energies {
        match tier_one_check(data, energy, thermal_u) {
            Ok(formalism) => {
                if !formalisms.contains(&formalism) {
                    formalisms.push(formalism);
                }
            }
            // The grid ascends, so the first energy to fail is the lowest
            // one that fails, and there is nothing later that could report
            // a better reason.
            Err(reason) => return Ok(DopplerRoute::SampledTable { reason }),
        }
    }
    Ok(DopplerRoute::Continuous { formalisms })
}

/// A resolved SLBW or MLBW range that actually carries resonances.
fn is_tier_one_formalism(range: &ResonanceRange) -> bool {
    range.is_evaluable()
        && matches!(
            range.formalism,
            ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
        )
}

/// The tier-1 conditions at one energy, in order. `Ok` carries the covering
/// range's formalism.
fn tier_one_check(
    data: &ResonanceData,
    energy_ev: f64,
    thermal_u: f64,
) -> Result<ResonanceFormalism, SampledTableReason> {
    let covering = |(index, range): &(usize, &ResonanceRange)| {
        covers(
            range.energy_low,
            range.energy_high,
            upper_bound_is_half_open(data, *index),
            energy_ev,
        )
    };
    // Prefer the evaluable covering range: a parse-and-skip placeholder
    // spanning the same energies contributes nothing to the cross-section
    // and must not mask the range that does.
    let ranges = || data.ranges.iter().enumerate();
    let Some((index, range)) = ranges()
        .find(|entry| entry.1.is_evaluable() && covering(entry))
        .or_else(|| ranges().find(covering))
    else {
        return Err(uncovered_energy_reason(data, energy_ev));
    };

    if !is_tier_one_formalism(range) {
        // A resolved SLBW/MLBW range with no resonances is accepted by the
        // parser but evaluates to nothing. Its formalism is not what
        // demoted the isotope, so name the empty range instead of reporting
        // "SLBW formalism", which would read as though SLBW were tier 2.
        if range.resolved
            && matches!(
                range.formalism,
                ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
            )
        {
            return Err(SampledTableReason::EmptyResolvedRange {
                energy_ev,
                formalism: range.formalism,
            });
        }
        return Err(SampledTableReason::Formalism {
            energy_ev,
            formalism: Some(range.formalism),
        });
    }
    let formalism = range.formalism;

    let speed = energy_ev.sqrt();
    let low_speed = speed - SUPPORT_X * thermal_u;
    if low_speed <= 0.0 {
        return Err(SampledTableReason::ThermalWindowFoldsThroughZero {
            energy_ev,
            thermal_u,
            formalism,
        });
    }
    let window_low_ev = low_speed * low_speed;
    let window_high_ev = (speed + SUPPORT_X * thermal_u).powi(2);

    // The window's top must respect the same half-open convention the
    // cross-section dispatcher uses: at a bound shared with an evaluable
    // neighbour, a source energy exactly on it belongs to the next range,
    // which is the cross-formalism mixing this gate exists to prevent.
    let top_inside = if upper_bound_is_half_open(data, index) {
        window_high_ev < range.energy_high
    } else {
        window_high_ev <= range.energy_high
    };
    if window_low_ev < range.energy_low || !top_inside {
        return Err(SampledTableReason::WindowCrossesRangeBoundary {
            energy_ev,
            window_low_ev,
            window_high_ev,
            range_low_ev: range.energy_low,
            range_high_ev: range.energy_high,
            formalism,
        });
    }

    // ENDF-6 forbids overlapping ranges, but the parser does not validate
    // it and the dispatcher sums every range containing a point, so an
    // overlap would put a second formalism inside the window.
    if let Some((other_range_index, _)) = ranges().find(|(i, other)| {
        *i != index
            && other.is_evaluable()
            && other.energy_low < window_high_ev
            && other.energy_high > window_low_ev
    }) {
        return Err(SampledTableReason::OverlappingRange {
            energy_ev,
            other_range_index,
            formalism,
        });
    }

    Ok(formalism)
}

/// The reason for an energy that no range covers: the nearest resolved
/// SLBW/MLBW range when the source has one, so a grid reaching past the
/// resolved region says so; otherwise a formalism failure with no
/// formalism to name.
fn uncovered_energy_reason(data: &ResonanceData, energy_ev: f64) -> SampledTableReason {
    let distance = |range: &ResonanceRange| {
        (range.energy_low - energy_ev)
            .max(energy_ev - range.energy_high)
            .max(0.0)
    };
    match data
        .ranges
        .iter()
        .filter(|range| is_tier_one_formalism(range))
        .min_by(|a, b| distance(a).total_cmp(&distance(b)))
    {
        Some(range) => SampledTableReason::GridLeavesResolvedRange {
            energy_ev,
            range_low_ev: range.energy_low,
            range_high_ev: range.energy_high,
            formalism: range.formalism,
        },
        None => SampledTableReason::Formalism {
            energy_ev,
            formalism: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doppler::DopplerParams;
    use nereids_endf::resonance::test_support::{synthetic_swave_slbw, u238_with_formalism};

    /// Room temperature. For the U-238 fixtures 8u ≈ 0.083 √eV, so the
    /// window at 6.674 eV spans about ±0.43 eV.
    const ROOM_K: f64 = 293.6;

    /// The kernel width the gate derives for the U-238 fixtures.
    fn route(data: &ResonanceData, energies: &[f64]) -> DopplerRoute {
        classify_isotope(data, energies, ROOM_K).expect("valid params and grid")
    }

    fn continuous(formalisms: &[ResonanceFormalism]) -> DopplerRoute {
        DopplerRoute::Continuous {
            formalisms: formalisms.to_vec(),
        }
    }

    /// Condition 1, the eligible case: a resolved SLBW or MLBW source with
    /// the whole window inside its range integrates.
    #[test]
    fn a_resolved_breit_wigner_source_inside_its_range_is_continuous() {
        for formalism in [ResonanceFormalism::SLBW, ResonanceFormalism::MLBW] {
            let data = u238_with_formalism(formalism);
            assert_eq!(route(&data, &[6.5, 6.674, 6.9]), continuous(&[formalism]));
        }
    }

    /// Condition 1, refused: Reich-Moore is a tier-2 formalism, named as
    /// such rather than as a window or range failure.
    #[test]
    fn reich_moore_is_refused_by_formalism() {
        let data = u238_with_formalism(ResonanceFormalism::ReichMoore);
        assert_eq!(
            route(&data, &[6.674]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    energy_ev: 6.674,
                    formalism: Some(ResonanceFormalism::ReichMoore),
                }
            }
        );
    }

    /// A resolved SLBW range carrying no resonances is not a formalism
    /// failure: it evaluates to nothing, and reporting "SLBW formalism"
    /// would read as though SLBW itself were tier 2.
    #[test]
    fn an_empty_resolved_range_is_named_as_empty_not_as_its_formalism() {
        let mut data = u238_with_formalism(ResonanceFormalism::SLBW);
        data.ranges[0].l_groups[0].resonances.clear();
        assert_eq!(
            route(&data, &[6.674]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::EmptyResolvedRange {
                    energy_ev: 6.674,
                    formalism: ResonanceFormalism::SLBW,
                }
            }
        );
    }

    /// An energy past the resolved region names the range it left, which is
    /// the common real-data case: an acquisition, or the auxiliary grid a
    /// resolution function adds, reaching beyond the evaluation.
    #[test]
    fn a_grid_past_the_resolved_region_names_the_range_it_left() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        assert_eq!(
            route(&data, &[6.674, 2.5e4]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::GridLeavesResolvedRange {
                    energy_ev: 2.5e4,
                    range_low_ev: 1e-5,
                    range_high_ev: 1e4,
                    formalism: ResonanceFormalism::MLBW,
                }
            }
        );
        // Control: a source with no tier-1 range at all has no range to
        // name, and reports the formalism failure instead.
        let rm = u238_with_formalism(ResonanceFormalism::ReichMoore);
        assert_eq!(
            route(&rm, &[2.5e4]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    energy_ev: 2.5e4,
                    formalism: None,
                }
            }
        );
    }

    /// Condition 2: below `8u` the window folds through zero energy, where
    /// the kernel's reflected term the integral drops is no longer small.
    /// A mass-1 target at 300 K has 8u ≈ 1.29 √eV, so 0.01 eV fails and
    /// 100 eV passes.
    #[test]
    fn a_window_that_would_fold_through_zero_is_refused() {
        // synthetic_swave_slbw builds a source whose awr is its first
        // argument, so the gate derives u for mass 1 from that.
        let data = synthetic_swave_slbw(1.0, 10.0, 1e-3, 1e-3, 3.0);
        let thermal_u = DopplerParams::new(300.0, 1.0).unwrap().u();
        assert_eq!(
            classify_isotope(&data, &[0.01, 100.0], 300.0).unwrap(),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::ThermalWindowFoldsThroughZero {
                    energy_ev: 0.01,
                    thermal_u,
                    formalism: ResonanceFormalism::SLBW,
                }
            }
        );
        assert_eq!(
            classify_isotope(&data, &[100.0], 300.0).unwrap(),
            continuous(&[ResonanceFormalism::SLBW])
        );
    }

    /// Condition 3: the window must lie inside the range, not merely the
    /// target energy. At 9990 eV the target is inside a range ending at
    /// 1e4 eV but the window is not.
    #[test]
    fn a_window_crossing_the_range_edge_is_refused_though_the_energy_is_inside() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let DopplerRoute::SampledTable {
            reason:
                SampledTableReason::WindowCrossesRangeBoundary {
                    energy_ev,
                    window_high_ev,
                    range_high_ev,
                    ..
                },
        } = route(&data, &[9990.0])
        else {
            panic!("a window past the range top must be refused");
        };
        assert_eq!((energy_ev, range_high_ev), (9990.0, 1e4));
        assert!(window_high_ev > range_high_ev);
        // Control: the same range, with the window well inside it.
        assert_eq!(
            route(&data, &[6.674]),
            continuous(&[ResonanceFormalism::MLBW])
        );
    }

    /// Condition 4: a second evaluable range inside the window would put a
    /// second formalism under the integral, because the dispatcher sums
    /// every range containing a source energy.
    #[test]
    fn an_overlapping_evaluable_range_is_refused() {
        let mut data = u238_with_formalism(ResonanceFormalism::MLBW);
        let mut overlapping = data.ranges[0].clone();
        overlapping.energy_low = 5.0;
        overlapping.energy_high = 8.0;
        data.ranges.push(overlapping);
        assert_eq!(
            route(&data, &[6.674]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::OverlappingRange {
                    energy_ev: 6.674,
                    other_range_index: 1,
                    formalism: ResonanceFormalism::MLBW,
                }
            }
        );
        // Control: a non-evaluable neighbour contributes no cross-section,
        // so it is not an overlap — and placed FIRST it must not mask the
        // range that does carry the cross-section either.
        data.ranges[1].formalism = ResonanceFormalism::Unresolved;
        data.ranges[1].resolved = false;
        for _ in 0..2 {
            assert_eq!(
                route(&data, &[6.674]),
                continuous(&[ResonanceFormalism::MLBW])
            );
            data.ranges.swap(0, 1);
        }
    }

    /// A grid spanning adjacent resolved ranges of different formalisms is
    /// eligible, and the verdict names both, because both were evaluated.
    /// The window at a shared, half-open bound is refused for the same
    /// reason the dispatcher hands that energy to the next range.
    #[test]
    fn adjacent_ranges_are_both_eligible_and_both_named() {
        let mut data = u238_with_formalism(ResonanceFormalism::SLBW);
        data.ranges[0].energy_high = 100.0;
        let mut upper = u238_with_formalism(ResonanceFormalism::MLBW).ranges[0].clone();
        upper.energy_low = 100.0;
        data.ranges.push(upper);

        assert_eq!(
            route(&data, &[50.0]),
            continuous(&[ResonanceFormalism::SLBW])
        );
        assert_eq!(
            route(&data, &[500.0]),
            continuous(&[ResonanceFormalism::MLBW])
        );
        assert_eq!(
            route(&data, &[50.0, 500.0]),
            continuous(&[ResonanceFormalism::SLBW, ResonanceFormalism::MLBW])
        );
        // The grid contract is what makes "first reached" mean energy
        // order: a descending grid is not a differently-ordered request,
        // it is refused before any routing happens.
        assert!(matches!(
            classify_isotope(&data, &[500.0, 50.0], ROOM_K),
            Err(DopplerError::UnsortedEnergies { .. })
        ));
        // At 99.9 eV the window reaches the shared bound, where a source
        // energy would be evaluated with the MLBW range above.
        let DopplerRoute::SampledTable {
            reason: SampledTableReason::WindowCrossesRangeBoundary { range_high_ev, .. },
        } = route(&data, &[99.9])
        else {
            panic!("a window reaching a shared range bound must be refused");
        };
        assert_eq!(range_high_ev, 100.0);
    }

    /// The verdict is all-or-nothing and reports the LOWEST failing energy,
    /// so passing energies before or after the failure do not change what
    /// the user is told.
    #[test]
    fn the_lowest_failing_energy_decides_for_the_whole_isotope() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        for grid in [
            vec![6.674, 9990.0, 9999.0],
            vec![9990.0, 9999.0],
            vec![9990.0],
        ] {
            let DopplerRoute::SampledTable {
                reason: SampledTableReason::WindowCrossesRangeBoundary { energy_ev, .. },
            } = route(&data, &grid)
            else {
                panic!("the grid reaches past the range top and must be refused");
            };
            assert_eq!(energy_ev, 9990.0, "grid {grid:?}");
        }
    }

    /// 0 K is neither tier: `DopplerParams` accepts it as "no broadening",
    /// and the gate must not report a continuous integral over a kernel of
    /// zero width.
    #[test]
    fn absolute_zero_is_neither_tier() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        assert_eq!(
            classify_isotope(&data, &[6.674], 0.0).unwrap(),
            DopplerRoute::Unbroadened
        );
        // Control: the same source, the same grid, one kelvin up.
        assert_eq!(
            classify_isotope(&data, &[6.674], 1.0).unwrap(),
            continuous(&[ResonanceFormalism::MLBW])
        );
        // Grid validity is answered independently of the temperature: 0 K
        // must not become a way to smuggle a malformed grid past the check.
        assert!(matches!(
            classify_isotope(&data, &[f64::NAN], 0.0),
            Err(DopplerError::InvalidEnergy { .. })
        ));
        assert!(matches!(
            classify_isotope(&data, &[], 0.0),
            Err(DopplerError::EmptyGrid)
        ));
    }

    /// An unroutable grid is an error, not a route. The sampled-table tier
    /// rejects exactly these grids too (`validate_doppler_grid` is the same
    /// function both use), so answering "take tier 2" would send the caller
    /// somewhere that cannot run. Rejecting up front also keeps the
    /// lowest-failing-energy reduction free of values it cannot order: NaN
    /// compares false against everything, so one left in the grid would pin
    /// the reported reason to itself.
    #[test]
    fn an_unroutable_grid_is_an_error_and_not_a_route() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        for bad in [f64::NAN, f64::INFINITY, -5.0, 0.0] {
            assert!(
                matches!(
                    classify_isotope(&data, &[bad, 6.674], ROOM_K),
                    Err(DopplerError::InvalidEnergy { index: 0, .. })
                ),
                "energy {bad} must be refused"
            );
        }
        // The masking falsifier: 9990 eV alone is refused for crossing the
        // range edge, and a NaN in front of it must not quietly become the
        // reported reason.
        let DopplerRoute::SampledTable {
            reason: SampledTableReason::WindowCrossesRangeBoundary { energy_ev, .. },
        } = route(&data, &[9990.0])
        else {
            panic!("9990 eV must be refused at the range edge");
        };
        assert_eq!(energy_ev, 9990.0);
        // A temperature the parameters reject is likewise an error, and the
        // AWR is never a caller's to get wrong: it comes from the source.
        assert!(matches!(
            classify_isotope(&data, &[6.674], -1.0),
            Err(DopplerError::InvalidParams(_))
        ));
    }
}
