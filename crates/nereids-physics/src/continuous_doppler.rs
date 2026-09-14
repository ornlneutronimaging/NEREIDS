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
//! ## Why the verdict is per isotope and all-or-nothing
//!
//! Mixing tiers within one isotope would make the reported cross-section a
//! function of where in the grid each point happened to fall. The gate
//! therefore reports the first failing condition at the lowest failing
//! energy and demotes the whole isotope.
//!
//! The verdict does depend on the grid's EXTENT: a grid reaching past the
//! resolved region asks a different question from one that stops inside it,
//! and answering both the same way would hide the reach. What it does not
//! depend on is the grid's ORDER, nor on where in the grid a failing energy
//! sits — the same energies in any sequence give the same route, the same
//! reason and the same formalism list.
//!
//! Nothing in the workspace calls this yet; the integral it guards arrives
//! separately.

use nereids_endf::resonance::{ResonanceData, ResonanceFormalism, ResonanceRange};

use crate::doppler::DopplerParams;
use crate::doppler_route::{DopplerRoute, SampledTableReason};
use crate::reich_moore::{covers, upper_bound_is_half_open};

/// Half-width of the kernel support in units of `u`, so the thermal window
/// is `[(√E − 8u)², (√E + 8u)²]`. `erfc(8) ≈ 1.1e-29` of the kernel mass
/// lies outside it, far below any tolerance the integral works to.
pub const SUPPORT_X: f64 = 8.0;

/// The Doppler route of one isotope over `work_energies`.
///
/// `params` must be the broadening parameters of THIS isotope, since the
/// kernel width `u = √(k_B T / A)` is read from them. Taking the validated
/// type rather than a bare `u` is what keeps that width finite and
/// non-negative: a negative `u` puts the window's lower edge ABOVE its upper
/// edge, and both containment tests below then pass on a window that does
/// not exist.
///
/// The verdict covers the whole grid: the lowest energy that fails a tier-1
/// condition demotes the isotope and names the reason. Conditions are tested
/// in a fixed order so the reported reason is deterministic rather than an
/// artefact of which check happened to run first.
pub fn classify_isotope(
    data: &ResonanceData,
    work_energies: &[f64],
    params: &DopplerParams,
) -> DopplerRoute {
    classify_isotope_with(
        data,
        work_energies,
        params,
        &ResonanceRange::has_file3_background,
    )
}

/// [`classify_isotope`] with the File-3 predicate injected, so the gate can
/// be shown to consult it before any parser produces a range answering
/// `true`.
pub(crate) fn classify_isotope_with(
    data: &ResonanceData,
    work_energies: &[f64],
    params: &DopplerParams,
    file3_present: &dyn Fn(&ResonanceRange) -> bool,
) -> DopplerRoute {
    // At absolute zero there is no kernel to apply by either route, so the
    // tier question does not arise. `DopplerParams` rejects a negative
    // temperature outright, which is why this tests for equality.
    if params.temperature_k() == 0.0 {
        return DopplerRoute::Unbroadened;
    }
    // One pass up front, so the reduction below is provably free of values
    // it cannot order: NaN compares false against everything, so a single
    // NaN left in the grid would pin the reported reason to itself and mask
    // the genuine lowest failing energy.
    if let Some(&energy_ev) = work_energies.iter().find(|e| !e.is_finite() || **e <= 0.0) {
        return DopplerRoute::SampledTable {
            reason: SampledTableReason::NonPhysicalEnergy { energy_ev },
        };
    }
    let thermal_u = params.u();
    // Every range the grid was evaluated in, by range index. A grid may
    // legitimately span adjacent resolved ranges of different formalisms;
    // the disclosed route is the executed route, so it names all of them
    // rather than the lowest energy's alone. Indices rather than formalisms,
    // because the disclosure is then ordered by the source's own ranges
    // instead of by the order the caller happened to list its energies in.
    let mut range_indices: Vec<usize> = Vec::new();
    // The lowest failing energy, not the first one encountered: the grid is
    // a request, and its ORDER must not change the verdict either.
    let mut failure: Option<(f64, SampledTableReason)> = None;
    for &energy in work_energies {
        match tier_one_check(data, energy, thermal_u, file3_present) {
            Ok(index) => {
                if !range_indices.contains(&index) {
                    range_indices.push(index);
                }
            }
            Err(reason) => {
                if failure.as_ref().is_none_or(|(lowest, _)| energy < *lowest) {
                    failure = Some((energy, reason));
                }
            }
        }
    }
    if let Some((_, reason)) = failure {
        return DopplerRoute::SampledTable { reason };
    }
    // Only an empty grid gets here with nothing accumulated. With no energy
    // to test, the only condition that can be evaluated is the formalism
    // one: the window, overlap and File-3 conditions all need an energy, so
    // an empty grid reports whether the source has ANY tier-1 range, and not
    // whether a real grid over it would be routed continuously.
    if range_indices.is_empty() {
        return match data.ranges.iter().find(|r| is_tier_one_formalism(r)) {
            Some(range) => DopplerRoute::Continuous {
                formalisms: vec![range.formalism],
            },
            None => DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    energy_ev: data.ranges.first().map_or(0.0, |r| r.energy_low),
                    formalism: data.ranges.first().map(|r| r.formalism),
                },
            },
        };
    }
    range_indices.sort_unstable();
    let mut formalisms: Vec<ResonanceFormalism> = Vec::new();
    for index in range_indices {
        let formalism = data.ranges[index].formalism;
        if !formalisms.contains(&formalism) {
            formalisms.push(formalism);
        }
    }
    DopplerRoute::Continuous { formalisms }
}

/// A resolved SLBW or MLBW range that actually carries resonances.
fn is_tier_one_formalism(range: &ResonanceRange) -> bool {
    range.is_evaluable()
        && matches!(
            range.formalism,
            ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
        )
}

/// The tier-1 conditions at one energy, in order. `Ok` carries the index of
/// the covering range, which is what orders the disclosure.
fn tier_one_check(
    data: &ResonanceData,
    energy_ev: f64,
    thermal_u: f64,
    file3_present: &dyn Fn(&ResonanceRange) -> bool,
) -> Result<usize, SampledTableReason> {
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

    if file3_present(range) {
        return Err(SampledTableReason::File3Background {
            energy_ev,
            formalism,
        });
    }
    Ok(index)
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

    const ROOM_K: f64 = 293.6;

    /// Room-temperature U-238: 8u ≈ 0.083 √eV, so the window at 6.674 eV
    /// spans about ±0.43 eV.
    fn u238_params() -> DopplerParams {
        DopplerParams::new(ROOM_K, 236.006).unwrap()
    }

    fn route(data: &ResonanceData, energies: &[f64]) -> DopplerRoute {
        classify_isotope(data, energies, &u238_params())
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
        let data = synthetic_swave_slbw(1.0, 10.0, 1e-3, 1e-3, 3.0);
        let params = DopplerParams::new(300.0, 1.0).unwrap();
        assert_eq!(
            classify_isotope(&data, &[0.01, 100.0], &params),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::ThermalWindowFoldsThroughZero {
                    energy_ev: 0.01,
                    thermal_u: params.u(),
                    formalism: ResonanceFormalism::SLBW,
                }
            }
        );
        assert_eq!(
            classify_isotope(&data, &[100.0], &params),
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

    /// Condition 5. The predicate is `false` for every range today, so the
    /// gate is exercised through an injected one; the paired assertion is
    /// that the real predicate is what the public entry point reads.
    #[test]
    fn a_file3_background_is_refused_and_the_real_predicate_is_the_one_consulted() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        assert_eq!(
            classify_isotope_with(&data, &[6.674], &u238_params(), &|_| true),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::File3Background {
                    energy_ev: 6.674,
                    formalism: ResonanceFormalism::MLBW,
                }
            }
        );
        assert_eq!(
            classify_isotope_with(&data, &[6.674], &u238_params(), &|_| false),
            continuous(&[ResonanceFormalism::MLBW])
        );
        // The pair that must be updated together when MF=3 support lands.
        assert!(data.ranges.iter().all(|r| !r.has_file3_background()));
        assert_eq!(
            route(&data, &[6.674]),
            continuous(&[ResonanceFormalism::MLBW])
        );
        // The empty grid does NOT consult the predicate, because it has no
        // energy to consult it at. Pinned so the limitation is visible
        // rather than mistaken for a File-3 refusal that never ran.
        assert_eq!(
            classify_isotope_with(&data, &[], &u238_params(), &|_| true),
            continuous(&[ResonanceFormalism::MLBW])
        );
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
        // Ordered by the source's ranges, not by the caller's grid: a
        // TOF-derived grid runs DESCENDING in energy and must disclose the
        // same route, comparing equal to the ascending one.
        let ascending = route(&data, &[50.0, 500.0]);
        assert_eq!(
            ascending,
            continuous(&[ResonanceFormalism::SLBW, ResonanceFormalism::MLBW])
        );
        assert_eq!(route(&data, &[500.0, 50.0]), ascending);
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
    /// so neither the position of the failure in the request nor the order
    /// of the request changes what the user is told.
    #[test]
    fn the_lowest_failing_energy_decides_for_the_whole_isotope() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        for grid in [
            vec![6.674, 9990.0, 9999.0],
            vec![9999.0, 9990.0, 6.674],
            vec![9999.0, 9990.0],
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

    /// An empty grid has no energy to test, so only the formalism condition
    /// can be evaluated. It reports whether the source has any tier-1 range
    /// at all — NOT that a real grid over it would be routed continuously,
    /// which the narrowed-range case below makes explicit.
    #[test]
    fn an_empty_grid_reports_the_formalism_condition_alone() {
        assert_eq!(
            route(&u238_with_formalism(ResonanceFormalism::MLBW), &[]),
            continuous(&[ResonanceFormalism::MLBW])
        );
        assert!(matches!(
            route(&u238_with_formalism(ResonanceFormalism::ReichMoore), &[]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    formalism: Some(ResonanceFormalism::ReichMoore),
                    ..
                }
            }
        ));
        // A range too narrow to hold any window: every real grid over it is
        // refused, the empty grid still reports Continuous.
        let mut narrow = u238_with_formalism(ResonanceFormalism::MLBW);
        narrow.ranges[0].energy_low = 6.6;
        narrow.ranges[0].energy_high = 6.7;
        assert!(matches!(
            route(&narrow, &[6.674]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::WindowCrossesRangeBoundary { .. }
            }
        ));
        assert_eq!(route(&narrow, &[]), continuous(&[ResonanceFormalism::MLBW]));
    }

    /// 0 K is neither tier: `DopplerParams` accepts it as "no broadening",
    /// and the gate must not report a continuous integral over a kernel of
    /// zero width.
    #[test]
    fn absolute_zero_is_neither_tier() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let cold = DopplerParams::new(0.0, 236.006).unwrap();
        assert_eq!(
            classify_isotope(&data, &[6.674], &cold),
            DopplerRoute::Unbroadened
        );
        // Control: the same source, the same grid, one kelvin up.
        let warm = DopplerParams::new(1.0, 236.006).unwrap();
        assert_eq!(
            classify_isotope(&data, &[6.674], &warm),
            continuous(&[ResonanceFormalism::MLBW])
        );
    }

    /// A grid energy that is not positive and finite is named as such, and
    /// — the point of rejecting it up front — does NOT displace the real
    /// lowest failing energy, which is what a NaN left in the reduction
    /// would do, since NaN compares false against every other value.
    #[test]
    fn a_non_physical_grid_energy_is_named_and_does_not_mask_the_real_failure() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        for bad in [f64::NAN, f64::INFINITY, -5.0, 0.0] {
            let DopplerRoute::SampledTable {
                reason: SampledTableReason::NonPhysicalEnergy { energy_ev },
            } = route(&data, &[bad, 6.674])
            else {
                panic!("energy {bad} must be refused as non-physical");
            };
            assert_eq!(energy_ev.to_bits(), bad.to_bits(), "bad energy {bad}");
        }
        // The masking falsifier: 9990 eV alone is refused for crossing the
        // range edge, and putting NaN in front of it must not silently
        // convert that diagnosis into the NaN one.
        let DopplerRoute::SampledTable {
            reason: SampledTableReason::WindowCrossesRangeBoundary { energy_ev, .. },
        } = route(&data, &[9990.0])
        else {
            panic!("9990 eV must be refused at the range edge");
        };
        assert_eq!(energy_ev, 9990.0);
        assert!(matches!(
            route(&data, &[f64::NAN, 9990.0]),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::NonPhysicalEnergy { .. }
            }
        ));
    }
}
