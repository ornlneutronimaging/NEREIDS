//! Tier-1 Doppler broadening: the free-gas kernel integrated over the
//! resonance equation, and the gate deciding which isotopes may take it.
//!
//! ## What is integrated
//!
//! `σ_D(E) = (1/(√π·E)) ∫ e^{−x²} E′ σ(E′) dx` over `|x| ≤ 8`, with the
//! source energy `E′ = (√E + u·x)²`. Substituting the kernel into that form
//! is what removes the `1/v²` prefactor of the lab-frame convolution, so
//! the integrand is bounded wherever `σ` is.
//!
//! Differentiating at fixed source SPEED — the source energies do not move
//! with `T`, only the weight on them does — turns `d/dT` of `e^{−x²}` into
//! the same integrand times `(x² − ½)/T`. Value and derivative therefore
//! share every panel, and a derivative converged on those panels costs one
//! extra multiply per node rather than a second adaptive pass.
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
//! ## Quadrature
//!
//! Gauss–Kronrod G10/K21 (QUADPACK `qk21`) with adaptive bisection: the
//! panel with the largest error estimate is split until the total error
//! meets the tolerance. Initial panel edges are the window ends, the
//! resonance breakpoints, and every knot of an energy-dependent scattering
//! radius `AP(E′)` inside the window — the SLBW/MLBW evaluator interpolates
//! `AP` piecewise, so each knot is a kink in `σ(E′)` that both rules would
//! otherwise straddle and mis-estimate.
//!
//! Every failure is hard. A tier-1 broadening that cannot converge returns
//! an error rather than a degraded number, because the whole point of the
//! two tiers is that the caller is told which one ran.
//!
//! Nothing in the workspace calls this yet; `transmission.rs` is wired
//! separately.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use nereids_endf::resonance::{ResonanceData, ResonanceFormalism, ResonanceRange};
use rayon::prelude::*;

use crate::doppler::{DopplerError, DopplerParams, validate_doppler_grid, zero_negative_value};
use crate::doppler_route::{DopplerRoute, SampledTableReason};
use crate::reich_moore::{CrossSectionPlan, CrossSections, covers, upper_bound_is_half_open};

/// Half-width of the kernel support in units of `u`, so the thermal window
/// is `[(√E − 8u)², (√E + 8u)²]`. `erfc(8) ≈ 1.1e-29` of the kernel mass
/// lies outside it, far below any tolerance the integral works to.
pub const SUPPORT_X: f64 = 8.0;

/// Relative tolerance on each target integral. Two orders below the `1e-6`
/// relative level of the anchors and finite-difference gates downstream, so
/// quadrature error is not what those measure.
pub const RELATIVE_TOLERANCE: f64 = 1.0e-8;

/// Absolute tolerance (barn) on each target integral, for energies where
/// the cross-section itself is small and a relative test alone would chase
/// noise.
pub const ABSOLUTE_TOLERANCE_BARN: f64 = 1.0e-8;

/// Absolute tolerance (barn/K) on each temperature derivative. A typical
/// derivative is `σ/T ≈ 1e-2 barn/K`, so this sits two orders below the
/// `1e-6` relative level its consumers work to.
pub const ABSOLUTE_DERIVATIVE_TOLERANCE_BARN_PER_K: f64 = 1.0e-10;

/// Deepest bisection allowed. A panel of width `16/2^20 ≈ 1.5e-5` in `x` is
/// far narrower than any resonance the breakpoints did not already isolate,
/// so reaching this means the integrand is not being resolved at all.
pub const MAX_DEPTH: usize = 20;

/// Most panels alive for one target. Real MLBW sources on real grids need
/// tens; a limit two orders above that turns a runaway into an error
/// instead of an out-of-memory.
pub const MAX_ACTIVE_PANELS: usize = 4_096;

const SQRT_PI: f64 = 1.772_453_850_905_516;

/// Breakpoint offsets from each resonance energy in units of its total
/// width, so the Lorentzian core and both shoulders each start in their own
/// panel rather than being discovered by refinement.
const BREAKPOINT_WIDTHS: [f64; 5] = [-4.0, -1.0, 0.0, 1.0, 4.0];

/// Gauss–Kronrod 21-point abscissae on `[−1, 1]`, non-negative half
/// (QUADPACK `qk21`). Odd indices are the 10-point Gauss–Legendre nodes,
/// which is what lets one set of evaluations serve both rules.
const KRONROD_ABSCISSAE: [f64; 11] = [
    0.995_657_163_025_808_1,
    0.973_906_528_517_171_7,
    0.930_157_491_355_708_2,
    0.865_063_366_688_984_5,
    0.780_817_726_586_416_9,
    0.679_409_568_299_024_4,
    0.562_757_134_668_604_7,
    0.433_395_394_129_247_2,
    0.294_392_862_701_460_2,
    0.148_874_338_981_631_22,
    0.0,
];

/// Kronrod weights matching [`KRONROD_ABSCISSAE`]; the last is the centre.
const KRONROD_WEIGHTS: [f64; 11] = [
    0.011_694_638_867_371_874,
    0.032_558_162_307_964_725,
    0.054_755_896_574_351_995,
    0.075_039_674_810_919_96,
    0.093_125_454_583_697_6,
    0.109_387_158_802_297_64,
    0.123_491_976_262_065_84,
    0.134_709_217_311_473_34,
    0.142_775_938_577_060_09,
    0.147_739_104_901_338_49,
    0.149_445_554_002_916_9,
];

/// 10-point Gauss–Legendre weights for `KRONROD_ABSCISSAE[1, 3, 5, 7, 9]`.
const GAUSS_WEIGHTS: [f64; 5] = [
    0.066_671_344_308_688_14,
    0.149_451_349_150_580_6,
    0.219_086_362_515_982_04,
    0.269_266_719_309_996_35,
    0.295_524_224_714_752_87,
];

/// Hard limits of the adaptive quadrature. The defaults are the module
/// constants; a smaller budget lets a test force a limit deterministically
/// instead of hoping to construct a pathological source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadratureBudget {
    /// Deepest bisection allowed (see [`MAX_DEPTH`]).
    pub max_depth: usize,
    /// Most panels alive for one target (see [`MAX_ACTIVE_PANELS`]).
    pub max_active_panels: usize,
}

impl Default for QuadratureBudget {
    fn default() -> Self {
        Self {
            max_depth: MAX_DEPTH,
            max_active_panels: MAX_ACTIVE_PANELS,
        }
    }
}

/// Which cross-section channel an integral broadens.
///
/// Transmission needs `Total`. The SAMMY ex001 reference curve is a capture
/// cross-section, so validating against it needs `Capture`. All four come
/// out of one evaluation of the resonance equation, so choosing among them
/// costs nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Total cross-section.
    Total,
    /// Elastic scattering.
    Elastic,
    /// Radiative capture.
    Capture,
    /// Fission.
    Fission,
}

impl Channel {
    fn pick(self, xs: &CrossSections) -> f64 {
        match self {
            Channel::Total => xs.total,
            Channel::Elastic => xs.elastic,
            Channel::Capture => xs.capture,
            Channel::Fission => xs.fission,
        }
    }
}

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
            Ok(index) => {
                let formalism = data.ranges[index].formalism;
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

/// The tier-1 conditions at one energy, in order. `Ok` carries the index of
/// the covering range, which is both the formalism the route discloses and
/// the range the integral must evaluate over.
fn tier_one_check(
    data: &ResonanceData,
    energy_ev: f64,
    thermal_u: f64,
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

/// One panel's contribution, value and temperature derivative together.
#[derive(Debug, Clone, Copy, Default)]
struct Integral {
    value: f64,
    derivative: f64,
}

/// A live panel of the adaptive quadrature, ordered by error so the worst
/// one is refined next.
#[derive(Debug, Clone, Copy)]
struct Panel {
    left: f64,
    right: f64,
    depth: usize,
    value: f64,
    derivative: f64,
    value_error: f64,
    derivative_error: f64,
    /// Ordering key. `sequence` breaks ties so the heap is a total order
    /// and the refinement path does not depend on insertion accidents.
    priority: f64,
    sequence: usize,
}

impl PartialEq for Panel {
    fn eq(&self, other: &Self) -> bool {
        self.priority.to_bits() == other.priority.to_bits() && self.sequence == other.sequence
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
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

/// One converged target integral.
#[derive(Debug, Clone, Copy)]
struct TargetIntegral {
    value: f64,
    derivative: f64,
    /// Whether SAMMY's rule KEPT a negative value here.
    kept_negative: bool,
}

/// Everything one target integral needs. The quadrature is sequential
/// within a target, so a target's result is bit-reproducible under any
/// thread schedule.
struct TargetContext<'plan, 'data> {
    plan: &'plan CrossSectionPlan<'data>,
    channel: Channel,
    target_energy: f64,
    target_speed: f64,
    thermal_u: f64,
    temperature_k: f64,
    require_derivative: bool,
    budget: QuadratureBudget,
    /// Whether any quadrature node so far had a positive `σ`. These nodes
    /// ARE the contributing unbroadened points of SAMMY's negative rule.
    any_source_positive: std::cell::Cell<bool>,
}

impl TargetContext<'_, '_> {
    /// Value and derivative integrands at kernel coordinate `x`.
    ///
    /// `evaluate_one` panics on a non-positive energy; `(√E + u·x)²` is
    /// positive for `|x| ≤ 8` exactly because the gate already required
    /// `√E > 8u`. That condition is load-bearing here, not just a physics
    /// nicety — weakening it would turn a refusal into a panic.
    fn integrand(&self, x: f64) -> (f64, f64) {
        let source_energy = (self.target_speed + self.thermal_u * x).powi(2);
        let sigma = self.channel.pick(&self.plan.evaluate_one(source_energy));
        if sigma > 0.0 {
            self.any_source_positive.set(true);
        }
        let value = (-x * x).exp() * source_energy * sigma / (SQRT_PI * self.target_energy);
        (value, value * (x * x - 0.5) / self.temperature_k)
    }

    /// G10/K21 on `[left, right]`, returning `(kronrod, gauss)`. The Gauss
    /// rule reuses the Kronrod nodes at odd indices, so the pair costs 21
    /// evaluations rather than 31.
    fn gauss_kronrod(&self, left: f64, right: f64) -> (Integral, Integral) {
        let middle = 0.5 * (left + right);
        let radius = 0.5 * (right - left);
        let mut kronrod = Integral::default();
        let mut gauss = Integral::default();

        let (value, derivative) = self.integrand(middle);
        kronrod.value += KRONROD_WEIGHTS[10] * value;
        kronrod.derivative += KRONROD_WEIGHTS[10] * derivative;

        for (i, (&node, &weight)) in KRONROD_ABSCISSAE[..10]
            .iter()
            .zip(&KRONROD_WEIGHTS[..10])
            .enumerate()
        {
            let (value_left, derivative_left) = self.integrand(middle - radius * node);
            let (value_right, derivative_right) = self.integrand(middle + radius * node);
            let value = value_left + value_right;
            let derivative = derivative_left + derivative_right;
            kronrod.value += weight * value;
            kronrod.derivative += weight * derivative;
            if i % 2 == 1 {
                let gauss_weight = GAUSS_WEIGHTS[i / 2];
                gauss.value += gauss_weight * value;
                gauss.derivative += gauss_weight * derivative;
            }
        }

        let scale = |integral: Integral| Integral {
            value: radius * integral.value,
            derivative: radius * integral.derivative,
        };
        (scale(kronrod), scale(gauss))
    }

    fn evaluate_panel(&self, left: f64, right: f64, depth: usize, sequence: usize) -> Panel {
        let (fine, coarse) = self.gauss_kronrod(left, right);
        let value_error = (fine.value - coarse.value).abs();
        let derivative_error = (fine.derivative - coarse.derivative).abs();
        Panel {
            left,
            right,
            depth,
            value: fine.value,
            derivative: fine.derivative,
            value_error,
            derivative_error,
            // The derivative carries a 1/T, so comparing its raw error
            // against the value's would make the derivative dominate the
            // refinement order at low temperature for no reason.
            priority: if self.require_derivative {
                value_error.max(self.temperature_k * derivative_error)
            } else {
                value_error
            },
            sequence,
        }
    }

    /// Initial panel edges in `x`: the window ends, every resonance
    /// breakpoint inside it, and every `AP(E′)` knot inside it.
    fn breakpoints(&self, range: &ResonanceRange) -> Vec<f64> {
        let low_energy = (self.target_speed - SUPPORT_X * self.thermal_u).powi(2);
        let high_energy = (self.target_speed + SUPPORT_X * self.thermal_u).powi(2);
        let mut points = vec![-SUPPORT_X, SUPPORT_X];
        let mut push_source_energy = |source_energy: f64| {
            if source_energy <= 0.0 {
                return;
            }
            let coordinate = (source_energy.sqrt() - self.target_speed) / self.thermal_u;
            if coordinate > -SUPPORT_X && coordinate < SUPPORT_X {
                points.push(coordinate);
            }
        };
        for group in &range.l_groups {
            for resonance in &group.resonances {
                let total_width = resonance.gn.abs()
                    + resonance.gg.abs()
                    + resonance.gfa.abs()
                    + resonance.gfb.abs();
                // Skip resonances whose Lorentzian cannot reach the window;
                // their breakpoints would all be clipped anyway.
                if total_width <= 0.0
                    || resonance.energy + 4.0 * total_width < low_energy
                    || resonance.energy - 4.0 * total_width > high_energy
                {
                    continue;
                }
                for multiplier in BREAKPOINT_WIDTHS {
                    push_source_energy(resonance.energy + multiplier * total_width);
                }
            }
        }
        for &(knot_energy, _) in range.ap_table.iter().flat_map(|table| &table.points) {
            push_source_energy(knot_energy);
        }
        points.sort_by(f64::total_cmp);
        points.dedup_by(|left, right| left.to_bits() == right.to_bits());
        points
    }

    fn integrate(&self, range: &ResonanceRange) -> Result<TargetIntegral, DopplerError> {
        let points = self.breakpoints(range);
        // The budget bounds the INITIAL panels as well as the refined ones.
        // A source with many resonances, or a dense AP(E) table, produces
        // its panel count from the breakpoints alone, so checking only
        // inside the refinement loop would let a wide source allocate past
        // the limit and — if those panels happened to converge — never
        // consult it at all.
        if points.len() - 1 > self.budget.max_active_panels {
            return Err(DopplerError::PanelLimit {
                energy_ev: self.target_energy,
                limit: self.budget.max_active_panels,
            });
        }
        let mut heap = BinaryHeap::with_capacity(points.len());
        let mut value = 0.0;
        let mut derivative = 0.0;
        let mut value_error = 0.0;
        let mut derivative_error = 0.0;
        let mut sequence = 0usize;
        for pair in points.windows(2) {
            let panel = self.evaluate_panel(pair[0], pair[1], 0, sequence);
            sequence += 1;
            value += panel.value;
            derivative += panel.derivative;
            value_error += panel.value_error;
            derivative_error += panel.derivative_error;
            heap.push(panel);
        }

        while value_error > ABSOLUTE_TOLERANCE_BARN + RELATIVE_TOLERANCE * value.abs()
            || (self.require_derivative
                && derivative_error
                    > ABSOLUTE_DERIVATIVE_TOLERANCE_BARN_PER_K
                        + RELATIVE_TOLERANCE * derivative.abs())
        {
            if heap.len() >= self.budget.max_active_panels {
                return Err(DopplerError::PanelLimit {
                    energy_ev: self.target_energy,
                    limit: self.budget.max_active_panels,
                });
            }
            let panel = heap
                .pop()
                .expect("an active panel while the error is nonzero");
            if panel.depth >= self.budget.max_depth {
                return Err(DopplerError::DepthLimit {
                    energy_ev: self.target_energy,
                    depth: self.budget.max_depth,
                });
            }
            let middle = 0.5 * (panel.left + panel.right);
            if !(panel.left < middle && middle < panel.right) {
                return Err(DopplerError::MidpointStagnation {
                    energy_ev: self.target_energy,
                    left: panel.left,
                    right: panel.right,
                });
            }
            let children = [
                self.evaluate_panel(panel.left, middle, panel.depth + 1, sequence),
                self.evaluate_panel(middle, panel.right, panel.depth + 1, sequence + 1),
            ];
            sequence += 2;
            // Running totals are corrected by the delta rather than
            // recomputed, so the cost per bisection stays constant.
            value += children[0].value + children[1].value - panel.value;
            derivative += children[0].derivative + children[1].derivative - panel.derivative;
            value_error = (value_error + children[0].value_error + children[1].value_error
                - panel.value_error)
                .max(0.0);
            derivative_error =
                (derivative_error + children[0].derivative_error + children[1].derivative_error
                    - panel.derivative_error)
                    .max(0.0);
            heap.extend(children);
        }

        if !value.is_finite() {
            return Err(DopplerError::NonFiniteIntegral {
                energy_ev: self.target_energy,
                value,
                derivative: false,
            });
        }
        if self.require_derivative && !derivative.is_finite() {
            return Err(DopplerError::NonFiniteIntegral {
                energy_ev: self.target_energy,
                value: derivative,
                derivative: true,
            });
        }

        // SAMMY's negative-value rule, with the quadrature nodes as the
        // contributing unbroadened points. The integrand carries
        // `1/(√π·E)`, so `value·E` is the kernel-weighted mean of `E′·σ` in
        // barn·eV — exactly the quantity SAMMY tests before its `/Em`.
        if value < 0.0 {
            let zero = zero_negative_value(value * self.target_energy, || {
                self.any_source_positive.get()
            });
            if zero {
                return Ok(TargetIntegral {
                    value: 0.0,
                    derivative: 0.0,
                    kept_negative: false,
                });
            }
            return Ok(TargetIntegral {
                value,
                derivative,
                kept_negative: true,
            });
        }
        Ok(TargetIntegral {
            value,
            derivative,
            kept_negative: false,
        })
    }
}

/// A converged tier-1 broadening of one channel over a whole grid.
#[derive(Debug, Clone, PartialEq)]
pub struct TierOneBroadening {
    /// Broadened cross-section at each target energy (barn).
    pub values: Vec<f64>,
    /// Temperature derivative at each target (barn/K). All zero when the
    /// caller did not ask for one, because an unrequested derivative was
    /// never converged to its own tolerance.
    pub derivatives: Vec<f64>,
    /// Targets whose reported value is negative: SAMMY's rule KEPT them
    /// when broadening ran, and on the unbroadened path they are simply
    /// the negative values of the resonance equation.
    pub negative_values: usize,
}

/// Value and (optionally) temperature derivative of one channel at every
/// target energy, under an explicit quadrature budget.
///
/// The kernel width `u = √(k_B T / A)` is built from the source's OWN
/// `awr`, exactly as [`classify_isotope`] builds it, so the width and the
/// resonance equation can never describe different nuclides. Taking a
/// caller-supplied [`DopplerParams`] would allow that: an AWR wrong by
/// 0.87% moves the width by 0.43%, which is below every tolerance in this
/// module.
///
/// At zero temperature, or an underflowed `u`, there is no kernel and the
/// values are the unbroadened equation — for ANY source, matching
/// [`classify_isotope`] answering [`DopplerRoute::Unbroadened`] before it
/// gates. Above zero the gate runs, and a source that does not qualify is
/// refused with [`DopplerError::NotTierOne`] rather than integrated.
///
/// # Errors
/// [`DopplerError`] for an invalid grid or temperature, for a source that
/// is not tier-1, or for a quadrature that cannot converge inside `budget`.
pub fn broaden_with_budget(
    energies: &[f64],
    data: &ResonanceData,
    temperature_k: f64,
    channel: Channel,
    require_derivative: bool,
    budget: QuadratureBudget,
) -> Result<TierOneBroadening, DopplerError> {
    let params = DopplerParams::new(temperature_k, data.awr)?;
    validate_doppler_grid(energies)?;
    if energies.is_empty() {
        return Err(DopplerError::EmptyGrid);
    }
    let thermal_u = params.u();
    let plan = CrossSectionPlan::new(data);

    // No kernel: the "integral" is the resonance equation itself. This sits
    // ABOVE the gate because the gate is about which BROADENING tier to
    // take, and at 0 K neither runs — the route gate says `Unbroadened`
    // here, so refusing a Reich-Moore source would contradict it.
    if temperature_k <= 0.0 || thermal_u == 0.0 {
        let values: Vec<f64> = energies
            .iter()
            .map(|&energy| channel.pick(&plan.evaluate_one(energy)))
            .collect();
        // The unbroadened equation can itself be negative, so the count has
        // to be taken here too rather than assumed zero.
        let negative_values = values.iter().filter(|&&v| v < 0.0).count();
        return Ok(TierOneBroadening {
            values,
            derivatives: vec![0.0; energies.len()],
            negative_values,
        });
    }

    // Gate every target BEFORE integrating any of them: the verdict is
    // per isotope and all-or-nothing, so a grid that fails anywhere must
    // not return half a curve.
    let mut ranges = Vec::with_capacity(energies.len());
    for &energy in energies {
        match tier_one_check(data, energy, thermal_u) {
            Ok(index) => ranges.push(&data.ranges[index]),
            Err(reason) => return Err(DopplerError::NotTierOne { reason }),
        }
    }

    // Each target is an independent integral, so targets are the unit of
    // parallelism — the common thermometry case has ONE isotope, so
    // per-isotope parallelism would leave this serial. Results are gathered
    // in grid order so the reported error is always the lowest-index
    // failure, whatever order the threads finished in.
    let results: Vec<Result<TargetIntegral, DopplerError>> = energies
        .par_iter()
        .zip(ranges.par_iter())
        .map(|(&target_energy, range)| {
            TargetContext {
                plan: &plan,
                channel,
                target_energy,
                target_speed: target_energy.sqrt(),
                thermal_u,
                temperature_k,
                require_derivative,
                budget,
                any_source_positive: std::cell::Cell::new(false),
            }
            .integrate(range)
        })
        .collect();
    let integrals = results
        .into_iter()
        .collect::<Result<Vec<TargetIntegral>, DopplerError>>()?;
    let negative_values = integrals.iter().filter(|i| i.kept_negative).count();
    let (values, derivatives): (Vec<f64>, Vec<f64>) = integrals
        .into_iter()
        .map(|integral| (integral.value, integral.derivative))
        .unzip();
    Ok(TierOneBroadening {
        // A derivative the caller did not ask for was refined against the
        // VALUE's error criterion only, so it is not converged and its
        // finiteness was never checked. Returning it would look like an
        // answer; zeroing it matches the documented contract and the
        // zero-temperature path.
        derivatives: if require_derivative {
            derivatives
        } else {
            vec![0.0; values.len()]
        },
        values,
        negative_values,
    })
}

/// Tier-1 cross-section of one channel at every target energy.
///
/// # Errors
/// As [`broaden_with_budget`].
pub fn broaden_channel(
    energies: &[f64],
    data: &ResonanceData,
    temperature_k: f64,
    channel: Channel,
) -> Result<Vec<f64>, DopplerError> {
    broaden_with_budget(
        energies,
        data,
        temperature_k,
        channel,
        false,
        QuadratureBudget::default(),
    )
    .map(|broadening| broadening.values)
}

/// Tier-1 total cross-section at every target energy.
///
/// # Errors
/// As [`broaden_with_budget`].
pub fn broaden(
    energies: &[f64],
    data: &ResonanceData,
    temperature_k: f64,
) -> Result<Vec<f64>, DopplerError> {
    broaden_channel(energies, data, temperature_k, Channel::Total)
}

/// Tier-1 total cross-section and its exact temperature derivative
/// (barn/K), both converged on the same panels.
///
/// # Errors
/// As [`broaden_with_budget`].
pub fn broaden_with_derivative(
    energies: &[f64],
    data: &ResonanceData,
    temperature_k: f64,
) -> Result<(Vec<f64>, Vec<f64>), DopplerError> {
    broaden_with_budget(
        energies,
        data,
        temperature_k,
        Channel::Total,
        true,
        QuadratureBudget::default(),
    )
    .map(|broadening| (broadening.values, broadening.derivatives))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doppler::{DopplerParams, doppler_broaden};
    use nereids_core::constants::BOLTZMANN_EV_PER_K;
    use nereids_endf::parser::parse_endf_file2;
    use nereids_endf::resonance::Resonance;
    use nereids_endf::resonance::test_support::{
        ex001_hydrogen_single_resonance, synthetic_swave_slbw, u238_with_formalism,
    };

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

    // ── the quadrature rule itself ─────────────────────────────────────────

    /// The QUADPACK pair, checked by what defines it rather than by
    /// restating the table: K21 integrates polynomials exactly to degree 31
    /// and G10 to degree 19. A single mistyped digit in any node or weight
    /// breaks this far above 1e-14.
    #[test]
    fn the_gauss_kronrod_constants_are_the_quadpack_pair() {
        let kronrod_sum = 2.0 * KRONROD_WEIGHTS[..10].iter().sum::<f64>() + KRONROD_WEIGHTS[10];
        let gauss_sum = 2.0 * GAUSS_WEIGHTS.iter().sum::<f64>();
        assert!(
            (kronrod_sum - 2.0).abs() < 1e-15,
            "kronrod weights {kronrod_sum}"
        );
        assert!((gauss_sum - 2.0).abs() < 1e-15, "gauss weights {gauss_sum}");

        for degree in 0..=31u32 {
            let exact = if degree % 2 == 0 {
                2.0 / f64::from(degree + 1)
            } else {
                0.0
            };
            let symmetric = |x: f64| x.powi(degree as i32) + (-x).powi(degree as i32);
            let kronrod = KRONROD_WEIGHTS[10] * 0.0_f64.powi(degree as i32)
                + KRONROD_ABSCISSAE[..10]
                    .iter()
                    .zip(&KRONROD_WEIGHTS[..10])
                    .map(|(&x, &w)| w * symmetric(x))
                    .sum::<f64>();
            assert!(
                (kronrod - exact).abs() < 1e-14,
                "K21 degree {degree}: {kronrod} vs {exact}"
            );
            if degree <= 19 {
                let gauss = (0..5)
                    .map(|k| GAUSS_WEIGHTS[k] * symmetric(KRONROD_ABSCISSAE[2 * k + 1]))
                    .sum::<f64>();
                assert!(
                    (gauss - exact).abs() < 1e-14,
                    "G10 degree {degree}: {gauss} vs {exact}"
                );
            }
        }
    }

    // ── independent quadrature oracle ──────────────────────────────────────

    /// A uniform-speed trapezoid rule over the same window.
    ///
    /// This is an independent QUADRATURE, not an independent PHYSICS
    /// oracle. It deliberately reuses `CrossSectionPlan::evaluate_one`, the
    /// `(√E + u·x)²` mapping, the same `u`, and the same `E′σ/(√π E)`
    /// weighting, so it can only catch an error in the adaptive scheme —
    /// the panels, the error estimate, the refinement order. An error in
    /// the kernel itself would move both sides together and pass. SAMMY
    /// ex001 and the free-gas FWHM are what constrain the physics.
    fn trapezoid_oracle(
        data: &ResonanceData,
        target_energy: f64,
        temperature_k: f64,
        n_points: usize,
    ) -> (f64, f64) {
        let dx = 2.0 * SUPPORT_X / (n_points - 1) as f64;
        let thermal_u = (BOLTZMANN_EV_PER_K * temperature_k / data.awr).sqrt();
        let target_speed = target_energy.sqrt();
        let plan = CrossSectionPlan::new(data);
        let normalization = SQRT_PI * target_energy;
        let (mut value_sum, mut derivative_sum) = (0.0, 0.0);
        for index in 0..n_points {
            let x = -SUPPORT_X + index as f64 * dx;
            let source_energy = (target_speed + thermal_u * x).powi(2);
            let sigma = plan.evaluate_one(source_energy).total;
            let endpoint_weight = if index == 0 || index + 1 == n_points {
                0.5
            } else {
                1.0
            };
            let contribution =
                endpoint_weight * (-x * x).exp() * source_energy * sigma / normalization;
            value_sum += contribution;
            derivative_sum += contribution * (x * x - 0.5) / temperature_k;
        }
        (value_sum * dx, derivative_sum * dx)
    }

    fn hf177() -> ResonanceData {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/data/endf/Hf-177.endf");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("required Hf-177 fixture missing at {path:?}: {error}"));
        parse_endf_file2(&text).expect("tracked Hf-177 fixture must parse")
    }

    /// Real multi-resonance MLBW data against 400,001 trapezoid points, for
    /// BOTH the value and the temperature derivative.
    #[test]
    fn a_real_mlbw_source_matches_the_uniform_speed_trapezoid_oracle() {
        let data = hf177();
        let target_energy = 8.876_917_538_350_767_f64;
        let temperature_k = 300.0_f64;
        let (oracle_value, oracle_derivative) =
            trapezoid_oracle(&data, target_energy, temperature_k, 400_001);

        let (values, derivatives) =
            broaden_with_derivative(&[target_energy], &data, temperature_k).unwrap();
        let value_rel = (values[0] - oracle_value).abs() / oracle_value.abs();
        let derivative_rel = (derivatives[0] - oracle_derivative).abs() / oracle_derivative.abs();
        assert!(
            value_rel <= 1.0e-10,
            "value: adaptive={:.16e} trapezoid={oracle_value:.16e} rel={value_rel:.3e}",
            values[0]
        );
        assert!(
            derivative_rel <= 1.0e-10,
            "derivative: adaptive={:.16e} trapezoid={oracle_derivative:.16e} rel={derivative_rel:.3e}",
            derivatives[0]
        );
    }

    // ── SAMMY ───────────────────────────────────────────────────────────────

    /// SAMMY's own ex001a output, all 315 rows. Columns 1-3 of the vendored
    /// file are SAMMY echoing its input data and match
    /// `samexm/ex001/ex001a.dat` byte for byte; column 4 is its theory
    /// curve, the Doppler-broadened capture cross-section at 300 K.
    ///
    /// ## What the residual is, and what it is not
    ///
    /// We sit a FLAT +0.77% above SAMMY across the resonance line and agree
    /// to ~1e-6 in the wings. That shape is measured, not assumed, and it
    /// rules out the obvious causes: an AWR error would give an S-shaped
    /// residual changing sign across the peak (scanning AWR 9.90-10.00
    /// never brings the maximum below 7.7e-3), 300 K is the optimum over
    /// 295-305 K, and the two plausible alternative kernel weightings give
    /// 8.1e-2 and 4.3e-2 instead of 7.7e-3.
    ///
    /// The sampled-table tier sits at +0.80% on the same curve. The two
    /// tiers therefore agree with EACH OTHER to well under 0.1% while both
    /// stand 0.8% from SAMMY, so the residual is not this integrator. The
    /// core's integrated strength is ~0.77% larger than SAMMY's while the
    /// wings match — the signature of an effective width difference of
    /// about 11 μeV on Γ = 1.5 meV, most plausibly SAMMY's own `Dopfgm`
    /// under-resolving a 1.5 meV core on its grid.
    ///
    /// That last step is a hypothesis, not a measurement: this is an OPEN
    /// discrepancy against the only external reference available here, and
    /// it must be resolved before any claim of SAMMY-equivalent broadening
    /// is made in print. The assertion below is therefore a two-sided pin
    /// on the value we actually observe, not a loose bound — a loose
    /// tolerance here would absorb a sub-0.8% physics error, which is
    /// larger than the 0.43% kernel-width error this very branch removed.
    #[test]
    fn the_sammy_ex001_capture_curve_is_matched_to_a_measured_residual() {
        let data = ex001_hydrogen_single_resonance();
        let (energies, reference): (Vec<f64>, Vec<f64>) =
            include_str!("../tests/data/sammy_ex001a_answers.lst")
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    let columns: Vec<f64> = line
                        .split_whitespace()
                        .map(|c| c.parse::<f64>().unwrap())
                        .collect();
                    assert_eq!(
                        columns.len(),
                        4,
                        "ex001a rows are E, data, uncertainty, theory"
                    );
                    (columns[0], columns[3])
                })
                .unzip();
        assert_eq!(energies.len(), 315);
        // The whole curve must be eligible, or the comparison is vacuous.
        assert_eq!(
            classify_isotope(&data, &energies, 300.0).unwrap(),
            continuous(&[ResonanceFormalism::SLBW])
        );

        let ours = broaden_channel(&energies, &data, 300.0, Channel::Capture).unwrap();
        let (worst, max_rel) = ours
            .iter()
            .zip(&reference)
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs() / b))
            .fold((0, 0.0_f64), |acc, x| if x.1 > acc.1 { x } else { acc });
        eprintln!(
            "ex001 tier 1: max_rel={max_rel:.3e} at E={} eV (ours {} vs SAMMY {})",
            energies[worst], ours[worst], reference[worst]
        );
        // Two-sided: moving in EITHER direction is a change worth seeing.
        assert!(
            (7.5e-3..8.0e-3).contains(&max_rel),
            "max relative error {max_rel:.3e} left the measured band [7.5e-3, 8.0e-3]"
        );

        // The residual is in the line, not the wings, and it is one-signed.
        let wing = ours
            .iter()
            .zip(&reference)
            .zip(&energies)
            .filter(|(_, e)| **e < 8.5 || **e > 11.5)
            .map(|((a, b), _)| (a - b).abs() / b)
            .fold(0.0_f64, f64::max);
        assert!(
            wing < 1.0e-4,
            "the wings should agree closely, got {wing:.3e}"
        );
        assert!(
            ours.iter().zip(&reference).all(|(a, b)| a >= b),
            "the residual is one-signed: we sit above SAMMY everywhere"
        );
    }

    // ── analytic oracles ───────────────────────────────────────────────────

    /// A line far narrower than the kernel broadens to the kernel's own
    /// width, so the measured FWHM of `E·σ` must be the free-gas
    /// `2√(ln2)·Δ_D`. This tests the kernel, not the integrator.
    #[test]
    fn a_narrow_line_broadens_to_the_free_gas_fwhm() {
        let (resonance_ev, temperature_k, awr) = (10.0_f64, 300.0_f64, 10.0_f64);
        let data = synthetic_swave_slbw(awr, resonance_ev, 1.5e-6, 1.5e-6, 5.0);
        let doppler_width = (4.0 * resonance_ev * BOLTZMANN_EV_PER_K * temperature_k / awr).sqrt();
        let expected_fwhm = 2.0 * 2.0_f64.ln().sqrt() * doppler_width;
        assert!(
            doppler_width > 1e4 * 3e-6,
            "the line must be far narrower than the kernel for this to mean anything"
        );

        let energies: Vec<f64> = (0..=2400).map(|i| 9.4 + f64::from(i) * 5.0e-4).collect();
        let capture = broaden_channel(&energies, &data, temperature_k, Channel::Capture).unwrap();
        let profile: Vec<f64> = energies.iter().zip(&capture).map(|(e, s)| e * s).collect();
        let half = 0.5 * profile.iter().copied().fold(f64::MIN, f64::max);
        let crossing = |i: usize| {
            energies[i]
                + (half - profile[i]) / (profile[i + 1] - profile[i])
                    * (energies[i + 1] - energies[i])
        };
        let rise = (0..profile.len() - 1)
            .find(|&i| profile[i] < half && profile[i + 1] >= half)
            .unwrap();
        let fall = (rise + 1..profile.len() - 1)
            .find(|&i| profile[i] >= half && profile[i + 1] < half)
            .unwrap();
        let fwhm = crossing(fall) - crossing(rise);
        let rel = (fwhm - expected_fwhm).abs() / expected_fwhm;
        assert!(
            rel < 2.0e-3,
            "FWHM {fwhm:.6} vs free-gas {expected_fwhm:.6} (rel {rel:.3e})"
        );
    }

    /// The analytic temperature derivative against a five-point central
    /// difference of the VALUE path, which never touches the derivative
    /// integrand.
    #[test]
    fn the_analytic_temperature_derivative_matches_a_five_point_difference() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = [6.5, 6.674, 7.1];
        let t = 300.0_f64;
        let h = 2.0_f64;
        let (_, analytic) = broaden_with_derivative(&energies, &data, t).unwrap();

        let at = |temperature: f64| broaden(&energies, &data, temperature).unwrap();
        let (m2, m1, p1, p2) = (at(t - 2.0 * h), at(t - h), at(t + h), at(t + 2.0 * h));
        for i in 0..energies.len() {
            let fd = (m2[i] - 8.0 * m1[i] + 8.0 * p1[i] - p2[i]) / (12.0 * h);
            let rel = (analytic[i] - fd).abs() / fd.abs();
            assert!(
                rel < 1.0e-6,
                "E={} eV: analytic {} vs five-point {fd} (rel {rel:.3e})",
                energies[i],
                analytic[i]
            );
        }
    }

    // ── limits and degenerate cases ────────────────────────────────────────

    /// Zero temperature is the unbroadened equation exactly, not
    /// approximately: there is no kernel to apply, so returning anything
    /// else would be a quadrature artefact.
    #[test]
    fn zero_temperature_is_bit_exactly_the_resonance_equation() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = [6.5, 6.674, 6.9];
        let expected: Vec<f64> = CrossSectionPlan::new(&data)
            .evaluate(&energies)
            .into_iter()
            .map(|xs| xs.total)
            .collect();

        let (values, derivatives) = broaden_with_derivative(&energies, &data, 0.0).unwrap();
        assert_eq!(values, expected);
        assert_eq!(derivatives, vec![0.0; 3]);

        // A temperature so small that u underflows to zero takes the same
        // path, rather than dividing by a zero width.
        let tiny = f64::from_bits(1);
        assert_eq!(DopplerParams::new(tiny, data.awr).unwrap().u(), 0.0);
        assert_eq!(broaden(&energies, &data, tiny).unwrap(), expected);

        // At 0 K the route gate says `Unbroadened` for ANY source, so a
        // tier-2-only formalism must get the unbroadened equation here and
        // not a tier-1 refusal.
        let rm = u238_with_formalism(ResonanceFormalism::ReichMoore);
        assert_eq!(
            classify_isotope(&rm, &energies, 0.0).unwrap(),
            DopplerRoute::Unbroadened
        );
        assert!(broaden(&energies, &rm, 0.0).is_ok());
    }

    /// The sampled table converges to the integral as its grid is refined:
    /// the two tiers are answers to the same question, and this is what
    /// makes that claim testable rather than asserted.
    #[test]
    fn the_sampled_table_converges_to_the_integral_under_refinement() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let params = DopplerParams::new(293.6, data.awr).unwrap();
        let targets = [6.4, 6.674, 6.95];
        let exact = broaden(&targets, &data, 293.6).unwrap();

        let mut previous = f64::INFINITY;
        for refinement in [1usize, 4, 16] {
            let step = 0.01 / refinement as f64;
            let grid: Vec<f64> = (0..=((3.0 / step) as usize))
                .map(|i| 5.0 + i as f64 * step)
                .collect();
            let table: Vec<f64> = CrossSectionPlan::new(&data)
                .evaluate(&grid)
                .into_iter()
                .map(|xs| xs.total)
                .collect();
            let sampled = doppler_broaden(&grid, &table, &params).unwrap();
            let worst = targets
                .iter()
                .zip(&exact)
                .map(|(&target, &want)| {
                    let i = grid
                        .iter()
                        .position(|&g| (g - target).abs() < 0.5 * step)
                        .expect("target on the sampled grid");
                    (sampled[i] - want).abs() / want
                })
                .fold(0.0_f64, f64::max);
            assert!(
                worst < previous,
                "refinement {refinement} did not improve: {worst:.3e} vs {previous:.3e}"
            );
            previous = worst;
        }
        assert!(previous < 5.0e-3, "finest grid still {previous:.3e} away");
    }

    /// The value at one target does not depend on which grid asked for it.
    /// Targets are independent integrals, and this pins that they stay so
    /// once they are farmed out to threads.
    #[test]
    fn a_target_is_bit_identical_across_grids_that_contain_it() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let alone = broaden(&[6.674], &data, 293.6).unwrap()[0];
        for grid in [
            vec![6.674, 7.0],
            vec![6.0, 6.674],
            vec![6.0, 6.3, 6.674, 7.0, 7.5],
        ] {
            let index = grid.iter().position(|&e| e == 6.674).unwrap();
            let together = broaden(&grid, &data, 293.6).unwrap()[index];
            assert_eq!(
                together.to_bits(),
                alone.to_bits(),
                "grid {grid:?} moved the value at 6.674 eV"
            );
        }
    }

    /// A source the gate refuses is an error, not a silently degraded
    /// answer: the caller must learn which tier ran.
    #[test]
    fn a_source_the_gate_refuses_is_an_error() {
        let data = u238_with_formalism(ResonanceFormalism::ReichMoore);
        assert!(matches!(
            broaden(&[6.674], &data, 293.6),
            Err(DopplerError::NotTierOne {
                reason: SampledTableReason::Formalism { .. }
            })
        ));
    }

    /// A budget too small to converge reports the limit and the energy it
    /// was reached at, rather than returning an unconverged number.
    ///
    /// The target has to be one that actually refines: the limits are
    /// checked inside the refinement loop, so a target whose breakpoint
    /// panels already meet the tolerance never reaches them however small
    /// the budget. A line 1.5 μeV wide inside a 0.6 eV window is such a
    /// target — the breakpoints all collapse onto one kernel coordinate,
    /// so the spike must be found by bisection.
    #[test]
    fn an_exhausted_quadrature_budget_is_reported_not_absorbed() {
        let data = synthetic_swave_slbw(10.0, 10.0, 1.5e-6, 1.5e-6, 5.0);
        let limited =
            |budget| broaden_with_budget(&[10.0], &data, 300.0, Channel::Capture, false, budget);
        assert!(matches!(
            limited(QuadratureBudget {
                max_depth: 4,
                max_active_panels: MAX_ACTIVE_PANELS,
            }),
            Err(DopplerError::DepthLimit { depth: 4, .. })
        ));
        assert!(matches!(
            limited(QuadratureBudget {
                max_depth: MAX_DEPTH,
                max_active_panels: 8,
            }),
            Err(DopplerError::PanelLimit { limit: 8, .. })
        ));
        // Control: the default budget resolves the same spike.
        assert!(limited(QuadratureBudget::default()).is_ok());
    }

    /// SAMMY keeps a genuinely negative broadened value (`fgm/mfgm4.f90`
    /// 83-101) rather than clamping it: an SLBW total whose same-J
    /// interference outweighs the shared potential term really is negative.
    #[test]
    fn a_negative_slbw_total_survives_broadening() {
        let mut data = synthetic_swave_slbw(55.45, 20_095.0, 30.0, 0.5, 5.0);
        data.ranges[0].l_groups[0].resonances.push(Resonance {
            energy: 20_105.0,
            j: 0.5,
            gn: 30.0,
            gg: 0.5,
            gfa: 0.0,
            gfb: 0.0,
        });
        // The destructive-interference trough between the two same-J levels
        // sits near 20.00 keV, well BELOW both resonance energies.
        let energies: Vec<f64> = (0..=40).map(|i| 19_990.0 + f64::from(i) * 0.5).collect();

        // Non-vacuity: the UNBROADENED source must actually go negative
        // somewhere on this grid, or the test proves nothing.
        let plan = CrossSectionPlan::new(&data);
        assert!(
            energies.iter().any(|&e| plan.evaluate_one(e).total < 0.0),
            "fixture must have a negative unbroadened total"
        );

        let broadened = broaden_with_budget(
            &energies,
            &data,
            293.6,
            Channel::Total,
            false,
            QuadratureBudget::default(),
        )
        .unwrap();
        assert!(
            broadened.negative_values > 0,
            "SAMMY's rule must KEEP at least one negative value here"
        );
        assert!(
            broadened.values.iter().any(|&v| v < 0.0),
            "a kept negative must reach the caller, not be clamped to zero"
        );
        // The count and the values agree: every kept negative is a negative
        // the caller can see.
        assert_eq!(
            broadened.negative_values,
            broadened.values.iter().filter(|&&v| v < 0.0).count()
        );
    }

    /// An unrequested derivative was refined against the VALUE's error
    /// criterion only, so it is not converged. Returning it would look like
    /// an answer.
    #[test]
    fn a_derivative_that_was_not_requested_comes_back_zero() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = [6.5, 6.674, 7.1];
        let without = broaden_with_budget(
            &energies,
            &data,
            293.6,
            Channel::Total,
            false,
            QuadratureBudget::default(),
        )
        .unwrap();
        assert_eq!(without.derivatives, vec![0.0; energies.len()]);

        // Control: asking for it gives something that is NOT zero, so the
        // assertion above pins the opt-out rather than a dead code path.
        let with = broaden_with_budget(
            &energies,
            &data,
            293.6,
            Channel::Total,
            true,
            QuadratureBudget::default(),
        )
        .unwrap();
        assert!(with.derivatives.iter().all(|d| d.abs() > 0.0));
        // The values agree to rounding but NOT to the bit: asking for the
        // derivative puts its error into the refinement priority, so the
        // panels are split in a different order and the sum reassociates.
        for (a, b) in with.values.iter().zip(&without.values) {
            assert!((a - b).abs() / b.abs() < 1e-12, "{a} vs {b}");
        }
    }

    /// The budget bounds the panels the BREAKPOINTS produce, not only the
    /// ones refinement adds. Before this, a source wide enough to exceed
    /// the cap on its initial panels sailed past it, and if those panels
    /// converged the cap was never consulted at all.
    #[test]
    fn the_panel_budget_bounds_the_initial_panels_too() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        // One resonance yields five breakpoints, so the initial panel count
        // is above 1 but the target converges without any refinement — the
        // case that used to escape.
        assert!(matches!(
            broaden_with_budget(
                &[6.674],
                &data,
                293.6,
                Channel::Total,
                false,
                QuadratureBudget {
                    max_depth: MAX_DEPTH,
                    max_active_panels: 1,
                },
            ),
            Err(DopplerError::PanelLimit { limit: 1, .. })
        ));
        assert!(broaden(&[6.674], &data, 293.6).is_ok());
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
