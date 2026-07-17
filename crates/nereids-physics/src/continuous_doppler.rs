//! Error-controlled free-gas Doppler integration of the resonance equation.
//!
//! The legacy broadener interpolates a caller-supplied zero-temperature
//! table.  That is appropriate only when the table already resolves every
//! resonance inside each thermal kernel.  This module instead evaluates the
//! immutable ENDF resonance equation at quadrature points selected from the
//! physical resonance locations and widths.  It never creates or refines a
//! stored source grid.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use nereids_endf::resonance::{ResonanceData, ResonanceFormalism, ResonanceRange};

use crate::doppler::DopplerParams;
use crate::reich_moore::CrossSectionPlan;

const SUPPORT_X: f64 = 8.0;
const RELATIVE_TOLERANCE: f64 = 1.0e-8;
const ABSOLUTE_TOLERANCE_BARN: f64 = 1.0e-8;
const MAX_DEPTH: usize = 20;
const MAX_ACTIVE_PANELS: usize = 4_096;
const SQRT_PI: f64 = 1.772_453_850_905_516;
const BREAKPOINT_WIDTHS: [f64; 5] = [-4.0, -1.0, 0.0, 1.0, 4.0];

const GL16: [(f64, f64); 16] = [
    (-9.894_009_349_916_499e-1, 2.715_245_941_175_405_5e-2),
    (-9.445_750_230_732_326e-1, 6.225_352_393_864_761e-2),
    (-8.656_312_023_878_316e-1, 9.515_851_168_249_3e-2),
    (-7.554_044_083_550_031e-1, 1.246_289_712_555_339_5e-1),
    (-6.178_762_444_026_438e-1, 1.495_959_888_165_766_5e-1),
    (-4.580_167_776_572_273_7e-1, 1.691_565_193_950_026_2e-1),
    (-2.816_035_507_792_589e-1, 1.826_034_150_449_236e-1),
    (-9.501_250_983_763_744e-2, 1.894_506_104_550_685_9e-1),
    (9.501_250_983_763_744e-2, 1.894_506_104_550_685_9e-1),
    (2.816_035_507_792_589e-1, 1.826_034_150_449_236e-1),
    (4.580_167_776_572_273_7e-1, 1.691_565_193_950_026_2e-1),
    (6.178_762_444_026_438e-1, 1.495_959_888_165_766_5e-1),
    (7.554_044_083_550_031e-1, 1.246_289_712_555_339_5e-1),
    (8.656_312_023_878_316e-1, 9.515_851_168_249_3e-2),
    (9.445_750_230_732_326e-1, 6.225_352_393_864_761e-2),
    (9.894_009_349_916_499e-1, 2.715_245_941_175_405_5e-2),
];

const GL32: [(f64, f64); 32] = [
    (-9.972_638_618_494_816e-1, 7.018_610_009_470_362e-3),
    (-9.856_115_115_452_683e-1, 1.627_439_473_090_648_6e-2),
    (-9.647_622_555_875_064e-1, 2.539_206_530_926_273e-2),
    (-9.349_060_759_377_397e-1, 3.427_386_291_302_163e-2),
    (-8.963_211_557_660_521e-1, 4.283_589_802_222_692_6e-2),
    (-8.493_676_137_325_7e-1, 5.099_805_926_237_596e-2),
    (-7.944_837_959_679_424e-1, 5.868_409_347_853_546e-2),
    (-7.321_821_187_402_897e-1, 6.582_222_277_636_16e-2),
    (-6.630_442_669_302_152e-1, 7.234_579_410_884_82e-2),
    (-5.877_157_572_407_623e-1, 7.819_389_578_707_012e-2),
    (-5.068_999_089_322_294e-1, 8.331_192_422_694_662e-2),
    (-4.213_512_761_306_353_3e-1, 8.765_209_300_440_369e-2),
    (-3.318_686_022_821_276_7e-1, 9.117_387_869_576_371e-2),
    (-2.392_873_622_521_370_6e-1, 9.384_439_908_080_439e-2),
    (-1.444_719_615_827_965e-1, 9.563_872_007_927_46e-2),
    (-4.830_766_568_773_832e-2, 9.654_008_851_472_758e-2),
    (4.830_766_568_773_832e-2, 9.654_008_851_472_758e-2),
    (1.444_719_615_827_965e-1, 9.563_872_007_927_46e-2),
    (2.392_873_622_521_370_6e-1, 9.384_439_908_080_439e-2),
    (3.318_686_022_821_276_7e-1, 9.117_387_869_576_371e-2),
    (4.213_512_761_306_353_3e-1, 8.765_209_300_440_369e-2),
    (5.068_999_089_322_294e-1, 8.331_192_422_694_662e-2),
    (5.877_157_572_407_623e-1, 7.819_389_578_707_012e-2),
    (6.630_442_669_302_152e-1, 7.234_579_410_884_82e-2),
    (7.321_821_187_402_897e-1, 6.582_222_277_636_16e-2),
    (7.944_837_959_679_424e-1, 5.868_409_347_853_546e-2),
    (8.493_676_137_325_7e-1, 5.099_805_926_237_596e-2),
    (8.963_211_557_660_521e-1, 4.283_589_802_222_692_6e-2),
    (9.349_060_759_377_397e-1, 3.427_386_291_302_163e-2),
    (9.647_622_555_875_064e-1, 2.539_206_530_926_273e-2),
    (9.856_115_115_452_683e-1, 1.627_439_473_090_648_6e-2),
    (9.972_638_618_494_816e-1, 7.018_610_009_470_362e-3),
];

#[derive(Debug, Clone, PartialEq)]
pub enum ContinuousDopplerError {
    InvalidEnergy {
        index: usize,
        value: f64,
    },
    UnsortedEnergy {
        index: usize,
        previous: f64,
        current: f64,
    },
    InvalidIntegral {
        energy_ev: f64,
        value: f64,
    },
    PanelLimit {
        energy_ev: f64,
        limit: usize,
    },
    DepthLimit {
        energy_ev: f64,
        depth: usize,
    },
    MidpointStagnation {
        energy_ev: f64,
        left: f64,
        right: f64,
    },
}

impl fmt::Display for ContinuousDopplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEnergy { index, value } => write!(
                formatter,
                "continuous Doppler energy {index} must be finite and positive, got {value}"
            ),
            Self::UnsortedEnergy {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "continuous Doppler energies must increase strictly; values at {} and {index} are {previous} and {current}",
                index - 1
            ),
            Self::InvalidIntegral { energy_ev, value } => write!(
                formatter,
                "continuous Doppler integral at {energy_ev} eV is invalid: {value}"
            ),
            Self::PanelLimit { energy_ev, limit } => write!(
                formatter,
                "continuous Doppler integral at {energy_ev} eV exceeded {limit} active panels"
            ),
            Self::DepthLimit { energy_ev, depth } => write!(
                formatter,
                "continuous Doppler integral at {energy_ev} eV exceeded depth {depth}"
            ),
            Self::MidpointStagnation {
                energy_ev,
                left,
                right,
            } => write!(
                formatter,
                "continuous Doppler integral at {energy_ev} eV stagnated on [{left}, {right}]"
            ),
        }
    }
}

impl std::error::Error for ContinuousDopplerError {}

#[derive(Debug, Clone)]
struct Panel {
    left: f64,
    right: f64,
    depth: usize,
    value: f64,
    error: f64,
    sequence: usize,
}

impl PartialEq for Panel {
    fn eq(&self, other: &Self) -> bool {
        self.error.to_bits() == other.error.to_bits() && self.sequence == other.sequence
    }
}

impl Eq for Panel {}

impl PartialOrd for Panel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Panel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.error
            .total_cmp(&other.error)
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

fn supported_range(
    data: &ResonanceData,
    target_energy: f64,
    thermal_u: f64,
) -> Option<&ResonanceRange> {
    let speed = target_energy.sqrt();
    let low_speed = speed - SUPPORT_X * thermal_u;
    if low_speed <= 0.0 {
        return None;
    }
    let low_energy = low_speed * low_speed;
    let high_energy = (speed + SUPPORT_X * thermal_u).powi(2);
    data.ranges.iter().find(|range| {
        range.resolved
            && matches!(
                range.formalism,
                ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
            )
            && range.rml.is_none()
            && range.urr.is_none()
            && low_energy >= range.energy_low
            && high_energy <= range.energy_high
    })
}

fn breakpoints(range: &ResonanceRange, target_energy: f64, thermal_u: f64) -> Vec<f64> {
    let speed = target_energy.sqrt();
    let low_energy = (speed - SUPPORT_X * thermal_u).powi(2);
    let high_energy = (speed + SUPPORT_X * thermal_u).powi(2);
    let mut points = vec![-SUPPORT_X, SUPPORT_X];
    for group in &range.l_groups {
        for resonance in &group.resonances {
            let total_width =
                resonance.gn.abs() + resonance.gg.abs() + resonance.gfa.abs() + resonance.gfb.abs();
            if total_width <= 0.0
                || resonance.energy + 4.0 * total_width < low_energy
                || resonance.energy - 4.0 * total_width > high_energy
            {
                continue;
            }
            for multiplier in BREAKPOINT_WIDTHS {
                let source_energy = resonance.energy + multiplier * total_width;
                if source_energy <= 0.0 {
                    continue;
                }
                let coordinate = (source_energy.sqrt() - speed) / thermal_u;
                if coordinate > -SUPPORT_X && coordinate < SUPPORT_X {
                    points.push(coordinate);
                }
            }
        }
    }
    points.sort_by(f64::total_cmp);
    points.dedup_by(|left, right| left.to_bits() == right.to_bits());
    points
}

fn rule(
    plan: &CrossSectionPlan<'_>,
    target_energy: f64,
    thermal_u: f64,
    left: f64,
    right: f64,
    nodes: &[(f64, f64)],
) -> f64 {
    let middle = 0.5 * (left + right);
    let radius = 0.5 * (right - left);
    let target_speed = target_energy.sqrt();
    let source_energies: Vec<f64> = nodes
        .iter()
        .map(|(node, _)| {
            let coordinate = middle + radius * node;
            (target_speed + thermal_u * coordinate).powi(2)
        })
        .collect();
    let cross_sections = plan.evaluate(&source_energies);
    radius
        * nodes
            .iter()
            .zip(source_energies.iter().zip(cross_sections))
            .map(|((node, weight), (source_energy, cross_section))| {
                let coordinate = middle + radius * node;
                weight * (-coordinate * coordinate).exp() * source_energy * cross_section.total
                    / (SQRT_PI * target_energy)
            })
            .sum::<f64>()
}

fn evaluate_panel(
    plan: &CrossSectionPlan<'_>,
    target_energy: f64,
    thermal_u: f64,
    left: f64,
    right: f64,
    depth: usize,
    sequence: usize,
) -> Panel {
    let coarse = rule(plan, target_energy, thermal_u, left, right, &GL16);
    let fine = rule(plan, target_energy, thermal_u, left, right, &GL32);
    Panel {
        left,
        right,
        depth,
        value: fine,
        error: (fine - coarse).abs(),
        sequence,
    }
}

fn broaden_target(
    plan: &CrossSectionPlan<'_>,
    range: &ResonanceRange,
    target_energy: f64,
    thermal_u: f64,
) -> Result<f64, ContinuousDopplerError> {
    let points = breakpoints(range, target_energy, thermal_u);
    let mut heap = BinaryHeap::with_capacity(points.len());
    let mut value = 0.0;
    let mut error = 0.0;
    let mut sequence = 0usize;
    for pair in points.windows(2) {
        let panel = evaluate_panel(
            plan,
            target_energy,
            thermal_u,
            pair[0],
            pair[1],
            0,
            sequence,
        );
        sequence += 1;
        value += panel.value;
        error += panel.error;
        heap.push(panel);
    }

    while error > ABSOLUTE_TOLERANCE_BARN + RELATIVE_TOLERANCE * value.abs() {
        if heap.len() >= MAX_ACTIVE_PANELS {
            return Err(ContinuousDopplerError::PanelLimit {
                energy_ev: target_energy,
                limit: MAX_ACTIVE_PANELS,
            });
        }
        let panel = heap.pop().expect("an active panel while error is nonzero");
        if panel.depth >= MAX_DEPTH {
            return Err(ContinuousDopplerError::DepthLimit {
                energy_ev: target_energy,
                depth: MAX_DEPTH,
            });
        }
        let middle = 0.5 * (panel.left + panel.right);
        if !(panel.left < middle && middle < panel.right) {
            return Err(ContinuousDopplerError::MidpointStagnation {
                energy_ev: target_energy,
                left: panel.left,
                right: panel.right,
            });
        }
        let children = [
            evaluate_panel(
                plan,
                target_energy,
                thermal_u,
                panel.left,
                middle,
                panel.depth + 1,
                sequence,
            ),
            evaluate_panel(
                plan,
                target_energy,
                thermal_u,
                middle,
                panel.right,
                panel.depth + 1,
                sequence + 1,
            ),
        ];
        sequence += 2;
        value += children[0].value + children[1].value - panel.value;
        error = (error + children[0].error + children[1].error - panel.error).max(0.0);
        heap.extend(children);
    }

    if !value.is_finite() || value < 0.0 {
        return Err(ContinuousDopplerError::InvalidIntegral {
            energy_ev: target_energy,
            value,
        });
    }
    Ok(value)
}

/// Try the continuous source-aware free-gas transform.
///
/// `Ok(None)` means that at least one requested thermal support window is not
/// wholly inside a resolved SLBW/MLBW range.  Callers must then use their
/// explicitly named legacy route; unsupported data are never partially mixed
/// with the continuous result.
pub(crate) fn try_broaden(
    energies: &[f64],
    data: &ResonanceData,
    params: &DopplerParams,
) -> Result<Option<Vec<f64>>, ContinuousDopplerError> {
    for (index, &energy) in energies.iter().enumerate() {
        if !energy.is_finite() || energy <= 0.0 {
            return Err(ContinuousDopplerError::InvalidEnergy {
                index,
                value: energy,
            });
        }
        if index > 0 && energy <= energies[index - 1] {
            return Err(ContinuousDopplerError::UnsortedEnergy {
                index,
                previous: energies[index - 1],
                current: energy,
            });
        }
    }
    if energies.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let plan = CrossSectionPlan::new(data);
    if params.temperature_k() <= 0.0 {
        return Ok(Some(
            plan.evaluate(energies)
                .into_iter()
                .map(|cross_section| cross_section.total)
                .collect(),
        ));
    }
    let thermal_u = params.u();
    if thermal_u == 0.0 {
        return Ok(Some(
            plan.evaluate(energies)
                .into_iter()
                .map(|cross_section| cross_section.total)
                .collect(),
        ));
    }
    let ranges: Option<Vec<&ResonanceRange>> = energies
        .iter()
        .map(|&energy| supported_range(data, energy, thermal_u))
        .collect();
    let Some(ranges) = ranges else {
        return Ok(None);
    };

    energies
        .iter()
        .zip(ranges)
        .map(|(&energy, range)| broaden_target(&plan, range, energy, thermal_u))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::test_support::u238_with_formalism;

    #[test]
    fn zero_temperature_matches_the_resonance_equation() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = [6.5, 6.674, 6.9];
        let params = DopplerParams::new(0.0, data.awr).unwrap();
        let actual = try_broaden(&energies, &data, &params)
            .unwrap()
            .expect("supported MLBW source");
        let expected: Vec<f64> = CrossSectionPlan::new(&data)
            .evaluate(&energies)
            .into_iter()
            .map(|cross_section| cross_section.total)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn unsupported_reich_moore_is_one_named_fallback_not_partial_output() {
        let data = u238_with_formalism(ResonanceFormalism::ReichMoore);
        let params = DopplerParams::new(300.0, data.awr).unwrap();
        assert_eq!(try_broaden(&[6.674], &data, &params).unwrap(), None);
    }

    #[test]
    fn resolved_mlbw_result_is_finite_positive_and_reproducible() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let params = DopplerParams::new(300.0, data.awr).unwrap();
        let first = try_broaden(&[6.674], &data, &params)
            .unwrap()
            .expect("supported MLBW source");
        let second = try_broaden(&[6.674], &data, &params)
            .unwrap()
            .expect("supported MLBW source");
        assert!(first[0].is_finite() && first[0] > 0.0);
        assert_eq!(first, second);
    }

    #[test]
    fn result_at_a_target_does_not_depend_on_unrelated_output_points() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let params = DopplerParams::new(300.0, data.awr).unwrap();
        let isolated = try_broaden(&[6.674], &data, &params)
            .unwrap()
            .expect("supported MLBW source");
        let surrounding = try_broaden(&[6.5, 6.674, 6.9], &data, &params)
            .unwrap()
            .expect("supported MLBW source");

        assert_eq!(isolated[0], surrounding[1]);
    }

    #[test]
    fn underflowed_thermal_width_is_exactly_unbroadened() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = [6.5, 6.674, 6.9];
        let params = DopplerParams::new(f64::from_bits(1), data.awr).unwrap();
        assert_eq!(params.u(), 0.0);
        let actual = try_broaden(&energies, &data, &params)
            .unwrap()
            .expect("supported MLBW source");
        let expected: Vec<f64> = CrossSectionPlan::new(&data)
            .evaluate(&energies)
            .into_iter()
            .map(|cross_section| cross_section.total)
            .collect();

        assert_eq!(actual, expected);
    }
}
