//! Tier-1 Doppler broadening: the free-gas kernel integrated over the
//! resonance equation at error-controlled quadrature, and the gate that
//! decides which isotopes qualify.
//!
//! ## Physics
//!
//! The free-gas model broadens a cross-section by averaging over the
//! Maxwellian distribution of target velocities (SAMMY manual Sec. III.B.1,
//! `fgm/mfgm1.f90` `Dopfgm`). With `v = √E` the neutron speed in √eV units
//! and `u = √(k_B T / A)` the thermal width in the same units, the
//! broadened cross-section at target energy `E` is
//!
//! ```text
//! σ_D(E) = (1 / (√π · E)) ∫ exp(−x²) · E′ · σ(E′) dx,    E′ = (√E + u·x)²
//! ```
//!
//! once the reflected kernel term `exp(−(√E + √E′)²/u²)` is dropped. That
//! term is below `exp(−64)` whenever `√E > 8u`, which is exactly the
//! condition the route gate imposes; under it the window `|x| ≤ 8`
//! (`SUPPORT_X`) captures all but `erfc(8) ≈ 1e-29` of the kernel mass and
//! every source energy `E′` is positive.
//!
//! The temperature derivative is taken at fixed source speed, where `σ` is
//! temperature-independent, so `∂/∂T` acts on the kernel alone:
//! `du/dT = u / (2T)` gives `∂K/∂T = K · (x² − ½) / T`. The derivative
//! integrand is therefore the value integrand times `(x² − ½) / T`; no
//! `dσ/dE′` is needed.
//!
//! ## Why this tier exists
//!
//! The sampled-table broadener (`doppler`) convolves the kernel with a
//! cross-section sampled on the caller's grid. That is exact for the
//! piecewise-linear table it is given, but a resonance narrower than the
//! grid spacing loses area before the kernel ever sees it. Here the
//! resonance equation is evaluated at quadrature nodes chosen from the
//! resonance positions and widths, so the result at `E` depends only on
//! `E`, the source, and the temperature — never on which other energies
//! were requested. SAMMY's own `Dopfgm` integrates over an auto-refined
//! sampled grid, so agreement with SAMMY oracles is bounded by SAMMY's grid
//! error, not by this integral's.
//!
//! ## Quadrature
//!
//! Each target integral is split at `x` coordinates of `E_r + m·Γ_tot`
//! (`m ∈ BREAKPOINT_WIDTHS`) for every resonance whose Lorentzian reaches
//! into the window, then refined by global adaptive bisection: the panel
//! with the largest error estimate is halved until both the value and (when
//! requested) the derivative meet their tolerances. A panel is integrated
//! with the embedded Gauss–Kronrod pair G10/K21 (QUADPACK `qk21`); the
//! K21 result is kept and `|K21 − G10|` is the error estimate. Exceeding
//! a hard limit is an error, never a silently degraded value.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use nereids_endf::resonance::{ResonanceData, ResonanceFormalism, ResonanceRange};
use rayon::prelude::*;

use crate::doppler::DopplerParams;
use crate::doppler_route::{DopplerRoute, SampledTableReason};
use crate::reich_moore::{CrossSectionPlan, CrossSections, range_covers, upper_bound_is_half_open};

/// Half-width of the kernel support in units of `u`: the contract's
/// thermal window is `[(√E − 8u)², (√E + 8u)²]`. `erfc(8) ≈ 1.1e-29` of the
/// kernel mass lies outside it — far below every tolerance here.
pub const SUPPORT_X: f64 = 8.0;
/// Relative tolerance on each target integral. Two orders below the
/// `1e-6` relative anchors and finite-difference gates the result feeds.
pub const RELATIVE_TOLERANCE: f64 = 1.0e-8;
/// Absolute tolerance (barn) on each target integral, for energies where
/// the cross-section itself is small.
pub const ABSOLUTE_TOLERANCE_BARN: f64 = 1.0e-8;
/// Absolute tolerance (barn/K) on each temperature derivative. A typical
/// derivative is `σ / T ≈ 1e-2 barn/K`, so this is two orders below the
/// `1e-6` relative level of the Jacobian consumers.
pub const ABSOLUTE_DERIVATIVE_TOLERANCE_BARN_PER_K: f64 = 1.0e-10;
/// Deepest bisection allowed: a panel of width `16 / 2^20 ≈ 1.5e-5` in `x`
/// is far narrower than any resonance the breakpoints did not already
/// isolate, so reaching it means the integrand is not being resolved.
pub const MAX_DEPTH: usize = 20;
/// Most panels alive for one target. Real MLBW sources on real grids need
/// tens; a limit two orders above that turns a runaway into an error.
pub const MAX_ACTIVE_PANELS: usize = 4_096;

const SQRT_PI: f64 = 1.772_453_850_905_516;
/// Breakpoint offsets from each resonance energy in units of its total
/// width, chosen so the Lorentzian core and its shoulders each get their
/// own panel before adaptive refinement starts.
const BREAKPOINT_WIDTHS: [f64; 5] = [-4.0, -1.0, 0.0, 1.0, 4.0];

/// Gauss–Kronrod 21-point abscissae on `[−1, 1]`, non-negative half
/// (QUADPACK `qk21`). Odd indices are the 10-point Gauss–Legendre nodes.
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
/// Kronrod weights matching `KRONROD_ABSCISSAE` (the last is the centre).
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

/// Hard limits of the adaptive quadrature.
///
/// The defaults are the module constants; a smaller budget exists so a
/// test can force a limit error deterministically.
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
/// Transmission needs `Total`; a capture-yield comparison (the SAMMY
/// ex001 oracle) needs `Capture`. All four channels come out of one
/// evaluation of the resonance equation, so selecting one costs nothing.
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

/// Failure of a tier-1 broadening. Every variant is a hard error: the
/// integral never returns a degraded value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ContinuousDopplerError {
    /// A target energy is non-finite or non-positive.
    InvalidEnergy {
        /// Index into the target grid.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The target grid is not strictly increasing.
    UnsortedEnergy {
        /// Index of the first out-of-order energy.
        index: usize,
        /// The energy before it.
        previous: f64,
        /// The out-of-order energy.
        current: f64,
    },
    /// The source does not qualify for tier 1 on this grid. The reason is
    /// the same one the route gate discloses.
    NotTierOne {
        /// The first failing condition at the lowest failing energy.
        reason: SampledTableReason,
    },
    /// The converged integral is non-finite or negative.
    InvalidIntegral {
        /// Target energy (eV).
        energy_ev: f64,
        /// The offending value.
        value: f64,
    },
    /// The converged temperature derivative is non-finite.
    InvalidDerivative {
        /// Target energy (eV).
        energy_ev: f64,
        /// The offending value.
        value: f64,
    },
    /// Refinement would exceed the active-panel limit.
    PanelLimit {
        /// Target energy (eV).
        energy_ev: f64,
        /// The limit that was hit.
        limit: usize,
    },
    /// Refinement would exceed the bisection depth limit.
    DepthLimit {
        /// Target energy (eV).
        energy_ev: f64,
        /// The limit that was hit.
        depth: usize,
    },
    /// A panel is too narrow to bisect in floating point.
    MidpointStagnation {
        /// Target energy (eV).
        energy_ev: f64,
        /// Panel left edge in `x`.
        left: f64,
        /// Panel right edge in `x`.
        right: f64,
    },
    /// The caller's cancellation flag was observed.
    Cancelled,
}

impl fmt::Display for ContinuousDopplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEnergy { index, value } => write!(
                f,
                "continuous Doppler energy {index} must be finite and positive, got {value}"
            ),
            Self::UnsortedEnergy {
                index,
                previous,
                current,
            } => write!(
                f,
                "continuous Doppler energies must increase strictly; values at {} and {index} \
                 are {previous} and {current}",
                index - 1
            ),
            Self::NotTierOne { reason } => write!(
                f,
                "continuous Doppler route requested for a source that does not qualify: {reason}"
            ),
            Self::InvalidIntegral { energy_ev, value } => {
                write!(
                    f,
                    "continuous Doppler integral at {energy_ev} eV is invalid: {value}"
                )
            }
            Self::InvalidDerivative { energy_ev, value } => write!(
                f,
                "continuous Doppler temperature derivative at {energy_ev} eV is invalid: {value}"
            ),
            Self::PanelLimit { energy_ev, limit } => write!(
                f,
                "continuous Doppler integral at {energy_ev} eV exceeded {limit} active panels"
            ),
            Self::DepthLimit { energy_ev, depth } => write!(
                f,
                "continuous Doppler integral at {energy_ev} eV exceeded bisection depth {depth}"
            ),
            Self::MidpointStagnation {
                energy_ev,
                left,
                right,
            } => write!(
                f,
                "continuous Doppler integral at {energy_ev} eV stagnated on [{left}, {right}]"
            ),
            Self::Cancelled => write!(f, "continuous Doppler broadening cancelled"),
        }
    }
}

impl std::error::Error for ContinuousDopplerError {}

// ─── Route gate ──────────────────────────────────────────────────────────────

/// Decide the Doppler route of one isotope over a working grid.
///
/// `thermal_u` is the kernel width `u = √(k_B T / A)` at the gate
/// temperature ([`DopplerParams::u`]). The verdict is all-or-nothing: the
/// first tier-1 condition to fail at the lowest failing grid energy names
/// the reason, and the whole isotope takes the sampled-table route. The
/// conditions are tested in a fixed order so the reported reason is
/// deterministic:
///
/// 1. the range covering the energy is a resolved, evaluable SLBW or MLBW
///    range (under the same half-open boundary convention as the
///    cross-section dispatcher);
/// 2. `√E > 8u`, so the window does not fold through zero;
/// 3. the window `[(√E − 8u)², (√E + 8u)²]` lies inside that range — on a
///    half-open upper bound the window's top must stay strictly below it,
///    because a source energy on the bound would be evaluated with the next
///    range's formalism;
/// 4. no other evaluable range overlaps the window (the dispatcher sums
///    every range containing a point, and nothing upstream rejects
///    overlapping ranges);
/// 5. the range carries no File-3 background term.
///
/// An empty grid has no energy to fail at; its verdict is that of the
/// source's first resolved SLBW/MLBW range, if any. A source whose grid
/// spans an SLBW range and an MLBW range is still tier 1 at every energy;
/// the disclosed formalism is that of the lowest grid energy.
pub fn classify_isotope(
    data: &ResonanceData,
    work_energies: &[f64],
    thermal_u: f64,
) -> DopplerRoute {
    classify_isotope_with(
        data,
        work_energies,
        thermal_u,
        &ResonanceRange::has_file3_background,
    )
}

/// [`classify_isotope`] with the File-3 predicate injected, so the gate
/// can be shown to consult it before any parser produces a range that
/// answers `true`.
pub(crate) fn classify_isotope_with(
    data: &ResonanceData,
    work_energies: &[f64],
    thermal_u: f64,
    file3_present: &dyn Fn(&ResonanceRange) -> bool,
) -> DopplerRoute {
    let mut formalism = None;
    for &energy in work_energies {
        match tier_one_check(data, energy, thermal_u, file3_present) {
            Ok((_, range_formalism)) => {
                formalism.get_or_insert(range_formalism);
            }
            Err(reason) => return DopplerRoute::SampledTable { reason },
        }
    }
    match formalism.or_else(|| {
        data.ranges
            .iter()
            .find(|r| is_tier_one_formalism(r))
            .map(|r| r.formalism)
    }) {
        Some(formalism) => DopplerRoute::Continuous { formalism },
        None => DopplerRoute::SampledTable {
            reason: SampledTableReason::Formalism {
                energy_ev: f64::NAN,
                formalism: data.ranges.first().map(|r| r.formalism),
            },
        },
    }
}

fn is_tier_one_formalism(range: &ResonanceRange) -> bool {
    range.is_evaluable()
        && matches!(
            range.formalism,
            ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
        )
}

/// The tier-1 conditions at one energy; `Ok` carries the covering range's
/// index and formalism.
fn tier_one_check(
    data: &ResonanceData,
    energy_ev: f64,
    thermal_u: f64,
    file3_present: &dyn Fn(&ResonanceRange) -> bool,
) -> Result<(usize, ResonanceFormalism), SampledTableReason> {
    let covers = |(index, range): (usize, &ResonanceRange)| {
        range_covers(range, upper_bound_is_half_open(data, index), energy_ev)
    };
    // Prefer the evaluable covering range: a non-evaluable placeholder that
    // happens to span the same energies contributes nothing to the
    // cross-section and must not mask the range that does.
    let covering = data
        .ranges
        .iter()
        .enumerate()
        .find(|entry| entry.1.is_evaluable() && covers(*entry))
        .or_else(|| data.ranges.iter().enumerate().find(|entry| covers(*entry)));
    let Some((index, range)) = covering else {
        return Err(SampledTableReason::Formalism {
            energy_ev,
            formalism: None,
        });
    };
    if !is_tier_one_formalism(range) {
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

    for (other_range_index, other) in data.ranges.iter().enumerate() {
        if other_range_index != index
            && other.is_evaluable()
            && other.energy_low < window_high_ev
            && other.energy_high > window_low_ev
        {
            return Err(SampledTableReason::OverlappingRange {
                energy_ev,
                other_range_index,
                formalism,
            });
        }
    }

    if file3_present(range) {
        return Err(SampledTableReason::File3Background {
            energy_ev,
            formalism,
        });
    }
    Ok((index, formalism))
}

// ─── Adaptive quadrature ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
struct Integral {
    value: f64,
    derivative: f64,
}

#[derive(Debug, Clone)]
struct Panel {
    left: f64,
    right: f64,
    depth: usize,
    value: f64,
    derivative: f64,
    value_error: f64,
    derivative_error: f64,
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

/// Everything one target integral needs; the quadrature is sequential
/// within a target, so its result is bit-reproducible under any thread
/// schedule.
struct TargetContext<'plan, 'data> {
    plan: &'plan CrossSectionPlan<'data>,
    channel: Channel,
    target_energy: f64,
    target_speed: f64,
    thermal_u: f64,
    temperature_k: f64,
    require_derivative: bool,
    budget: QuadratureBudget,
}

impl TargetContext<'_, '_> {
    /// Value and derivative integrands at kernel coordinate `x`.
    fn integrand(&self, x: f64) -> (f64, f64) {
        let source_energy = (self.target_speed + self.thermal_u * x).powi(2);
        let sigma = self.channel.pick(&self.plan.evaluate_one(source_energy));
        let value = (-x * x).exp() * source_energy * sigma / (SQRT_PI * self.target_energy);
        (value, value * (x * x - 0.5) / self.temperature_k)
    }

    /// G10/K21 on `[left, right]`: returns `(kronrod, gauss)`.
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
            priority: if self.require_derivative {
                value_error.max(self.temperature_k * derivative_error)
            } else {
                value_error
            },
            sequence,
        }
    }

    /// Initial panel edges in `x`: the window ends plus the resonance
    /// breakpoints that fall inside it.
    fn breakpoints(&self, range: &ResonanceRange) -> Vec<f64> {
        let low_energy = (self.target_speed - SUPPORT_X * self.thermal_u).powi(2);
        let high_energy = (self.target_speed + SUPPORT_X * self.thermal_u).powi(2);
        let mut points = vec![-SUPPORT_X, SUPPORT_X];
        for group in &range.l_groups {
            for resonance in &group.resonances {
                let total_width = resonance.gn.abs()
                    + resonance.gg.abs()
                    + resonance.gfa.abs()
                    + resonance.gfb.abs();
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
                    let coordinate = (source_energy.sqrt() - self.target_speed) / self.thermal_u;
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

    fn integrate(&self, range: &ResonanceRange) -> Result<Integral, ContinuousDopplerError> {
        let points = self.breakpoints(range);
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
                return Err(ContinuousDopplerError::PanelLimit {
                    energy_ev: self.target_energy,
                    limit: self.budget.max_active_panels,
                });
            }
            let panel = heap.pop().expect("an active panel while error is nonzero");
            if panel.depth >= self.budget.max_depth {
                return Err(ContinuousDopplerError::DepthLimit {
                    energy_ev: self.target_energy,
                    depth: self.budget.max_depth,
                });
            }
            let middle = 0.5 * (panel.left + panel.right);
            if !(panel.left < middle && middle < panel.right) {
                return Err(ContinuousDopplerError::MidpointStagnation {
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

        if !value.is_finite() || value < 0.0 {
            return Err(ContinuousDopplerError::InvalidIntegral {
                energy_ev: self.target_energy,
                value,
            });
        }
        if self.require_derivative && !derivative.is_finite() {
            return Err(ContinuousDopplerError::InvalidDerivative {
                energy_ev: self.target_energy,
                value: derivative,
            });
        }
        Ok(Integral { value, derivative })
    }
}

fn validate_energies(energies: &[f64]) -> Result<(), ContinuousDopplerError> {
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
    Ok(())
}

/// Value and (optionally converged) derivative of one channel at every
/// target energy, under an explicit quadrature budget.
///
/// The gate runs first at the kernel width of `params`, so a source that
/// does not qualify is refused with [`ContinuousDopplerError::NotTierOne`]
/// rather than integrated. At zero temperature (or an underflowed `u`)
/// the values are the unbroadened equation and the derivatives are zero.
/// Targets are integrated in parallel; results are collected in grid
/// order and the first failing target by index is the one reported.
pub(crate) fn broaden_integrals(
    energies: &[f64],
    data: &ResonanceData,
    params: &DopplerParams,
    channel: Channel,
    require_derivative: bool,
    budget: QuadratureBudget,
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<f64>, Vec<f64>), ContinuousDopplerError> {
    validate_energies(energies)?;
    let thermal_u = params.u();
    let temperature_k = params.temperature_k();

    let mut ranges = Vec::with_capacity(energies.len());
    for &energy in energies {
        match tier_one_check(
            data,
            energy,
            thermal_u,
            &ResonanceRange::has_file3_background,
        ) {
            Ok((index, _)) => ranges.push(&data.ranges[index]),
            Err(reason) => return Err(ContinuousDopplerError::NotTierOne { reason }),
        }
    }
    if energies.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let plan = CrossSectionPlan::new(data);
    if temperature_k <= 0.0 || thermal_u == 0.0 {
        let values = energies
            .iter()
            .map(|&energy| channel.pick(&plan.evaluate_one(energy)))
            .collect();
        return Ok((values, vec![0.0; energies.len()]));
    }

    // Each target is an independent integral, so the targets are the unit
    // of parallelism (the common thermometry case has one isotope, so
    // per-isotope parallelism alone would leave this serial). Results are
    // gathered as a plain vector so that the error reported is always the
    // lowest-index failure, independent of thread scheduling.
    let results: Vec<Result<Integral, ContinuousDopplerError>> = energies
        .par_iter()
        .zip(ranges.par_iter())
        .map(|(&target_energy, range)| {
            if cancel.is_some_and(|flag| flag.load(AtomicOrdering::Relaxed)) {
                return Err(ContinuousDopplerError::Cancelled);
            }
            TargetContext {
                plan: &plan,
                channel,
                target_energy,
                target_speed: target_energy.sqrt(),
                thermal_u,
                temperature_k,
                require_derivative,
                budget,
            }
            .integrate(range)
        })
        .collect();
    let integrals = results
        .into_iter()
        .collect::<Result<Vec<Integral>, ContinuousDopplerError>>()?;
    Ok(integrals
        .into_iter()
        .map(|integral| (integral.value, integral.derivative))
        .unzip())
}

/// Tier-1 cross-section of one channel at every target energy.
///
/// # Errors
/// [`ContinuousDopplerError::NotTierOne`] if the source does not qualify on
/// this grid at the kernel width of `params`; the quadrature-limit and
/// validity errors otherwise. `cancel` is polled per target.
pub fn broaden_channel(
    energies: &[f64],
    data: &ResonanceData,
    params: &DopplerParams,
    channel: Channel,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<f64>, ContinuousDopplerError> {
    broaden_integrals(
        energies,
        data,
        params,
        channel,
        false,
        QuadratureBudget::default(),
        cancel,
    )
    .map(|(values, _)| values)
}

/// Tier-1 total cross-section at every target energy.
///
/// # Errors
/// As [`broaden_channel`].
pub fn broaden(
    energies: &[f64],
    data: &ResonanceData,
    params: &DopplerParams,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<f64>, ContinuousDopplerError> {
    broaden_channel(energies, data, params, Channel::Total, cancel)
}

/// Tier-1 total cross-section and its exact temperature derivative
/// (barn/K), both converged on the same panels.
///
/// # Errors
/// As [`broaden`].
pub fn broaden_with_derivative(
    energies: &[f64],
    data: &ResonanceData,
    params: &DopplerParams,
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<f64>, Vec<f64>), ContinuousDopplerError> {
    broaden_integrals(
        energies,
        data,
        params,
        Channel::Total,
        true,
        QuadratureBudget::default(),
        cancel,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doppler::doppler_broaden;
    use nereids_core::constants::BOLTZMANN_EV_PER_K;
    use nereids_endf::parser::parse_endf_file2;
    use nereids_endf::resonance::test_support::{
        ex001_hydrogen_single_resonance, synthetic_swave_slbw, u238_with_formalism,
    };
    use std::sync::atomic::AtomicBool;

    const ROOM_K: f64 = 293.6;

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

    /// The VENUS Hf regression grid is 3471 bins uniform in time of flight
    /// (uniform in 1/√E) over 7–200 eV. `refinement` inserts that many
    /// minus one extra points between consecutive base points, uniform in
    /// 1/√E, so every base energy is a bit-identical member of every
    /// refined grid.
    fn venus_like_grid(refinement: usize) -> Vec<f64> {
        const N_BASE: usize = 3471;
        let s_first = 1.0 / 7.002_342_76_f64.sqrt();
        let s_last = 1.0 / 199.895_085_36_f64.sqrt();
        let base: Vec<f64> = (0..N_BASE)
            .map(|i| {
                let s = s_first + (s_last - s_first) * i as f64 / (N_BASE - 1) as f64;
                1.0 / (s * s)
            })
            .collect();
        if refinement == 1 {
            return base;
        }
        let mut grid = Vec::with_capacity((N_BASE - 1) * refinement + 1);
        for pair in base.windows(2) {
            grid.push(pair[0]);
            let s_lo = 1.0 / pair[0].sqrt();
            let s_hi = 1.0 / pair[1].sqrt();
            for j in 1..refinement {
                let s = s_lo + (s_hi - s_lo) * j as f64 / refinement as f64;
                grid.push(1.0 / (s * s));
            }
        }
        grid.push(*base.last().unwrap());
        grid
    }

    /// The sampled-table route: the equation sampled on `energies`, then
    /// the kernel-on-grid broadener.
    fn sampled_total(energies: &[f64], data: &ResonanceData, params: &DopplerParams) -> Vec<f64> {
        let table: Vec<f64> = CrossSectionPlan::new(data)
            .evaluate(energies)
            .into_iter()
            .map(|xs| xs.total)
            .collect();
        doppler_broaden(energies, &table, params).unwrap()
    }

    fn continuous_total(
        energies: &[f64],
        data: &ResonanceData,
        params: &DopplerParams,
    ) -> Vec<f64> {
        broaden(energies, data, params, None).unwrap()
    }

    // ── quadrature rule ────────────────────────────────────────────────────

    #[test]
    fn gauss_kronrod_constants_are_the_quadpack_pair() {
        let kronrod_sum = 2.0 * KRONROD_WEIGHTS[..10].iter().sum::<f64>() + KRONROD_WEIGHTS[10];
        let gauss_sum = 2.0 * GAUSS_WEIGHTS.iter().sum::<f64>();
        assert!(
            (kronrod_sum - 2.0).abs() < 1e-15,
            "kronrod weights sum {kronrod_sum}"
        );
        assert!(
            (gauss_sum - 2.0).abs() < 1e-15,
            "gauss weights sum {gauss_sum}"
        );

        // K21 is exact for polynomials of degree ≤ 31, G10 for degree ≤ 19.
        // Any transcription error in a node or weight breaks exactness far
        // above 1e-14.
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

    // ── route gate ─────────────────────────────────────────────────────────

    #[test]
    fn reich_moore_is_refused_with_the_formalism_reason() {
        let data = u238_with_formalism(ResonanceFormalism::ReichMoore);
        let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
        let expected = SampledTableReason::Formalism {
            energy_ev: 6.674,
            formalism: Some(ResonanceFormalism::ReichMoore),
        };
        assert_eq!(
            classify_isotope(&data, &[6.674], params.u()),
            DopplerRoute::SampledTable {
                reason: expected.clone()
            }
        );
        assert_eq!(
            broaden(&[6.674], &data, &params, None),
            Err(ContinuousDopplerError::NotTierOne { reason: expected })
        );
    }

    #[test]
    fn resolved_slbw_and_mlbw_take_the_continuous_route() {
        for formalism in [ResonanceFormalism::SLBW, ResonanceFormalism::MLBW] {
            let data = u238_with_formalism(formalism);
            let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
            assert_eq!(
                classify_isotope(&data, &[6.5, 6.674, 6.9], params.u()),
                DopplerRoute::Continuous { formalism }
            );
        }
    }

    #[test]
    fn gate_reports_the_first_failing_condition_at_the_lowest_failing_energy() {
        // u238 MLBW range is [1e-5, 1e4] eV; at ROOM_K, 8u ≈ 0.083 √eV, so
        // the window at 9990 eV already reaches past 1e4 eV.
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let thermal_u = DopplerParams::new(ROOM_K, data.awr).unwrap().u();
        let route = classify_isotope(&data, &[6.674, 9990.0, 9999.0], thermal_u);
        match route {
            DopplerRoute::SampledTable {
                reason:
                    SampledTableReason::WindowCrossesRangeBoundary {
                        energy_ev,
                        window_high_ev,
                        range_high_ev,
                        formalism,
                        ..
                    },
            } => {
                assert_eq!(energy_ev, 9990.0, "lowest failing energy is reported");
                assert!(window_high_ev > range_high_ev);
                assert_eq!(range_high_ev, 1e4);
                assert_eq!(formalism, ResonanceFormalism::MLBW);
            }
            other => panic!("expected a window-boundary demotion, got {other:?}"),
        }
        assert!(route.is_edge_fallback());
        // The same grid without the offending energies is tier 1.
        assert_eq!(
            classify_isotope(&data, &[6.674], thermal_u),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::MLBW
            }
        );
    }

    #[test]
    fn gate_folds_through_zero_for_a_light_target_at_low_energy() {
        // AWR = 1: u ≈ 0.16 √eV at 300 K, so 8u ≈ 1.29 > √(0.01 eV) = 0.1.
        let data = synthetic_swave_slbw(1.0, 10.0, 1e-3, 1e-3, 3.0);
        let thermal_u = DopplerParams::new(300.0, 1.0).unwrap().u();
        let route = classify_isotope(&data, &[0.01, 100.0], thermal_u);
        match route {
            DopplerRoute::SampledTable {
                reason:
                    SampledTableReason::ThermalWindowFoldsThroughZero {
                        energy_ev,
                        thermal_u: reported_u,
                        formalism,
                    },
            } => {
                assert_eq!(energy_ev, 0.01);
                assert_eq!(reported_u, thermal_u);
                assert_eq!(formalism, ResonanceFormalism::SLBW);
            }
            other => panic!("expected a fold-through-zero demotion, got {other:?}"),
        }
        // Well above 8u the same source qualifies.
        assert_eq!(
            classify_isotope(&data, &[100.0], thermal_u),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::SLBW
            }
        );
    }

    #[test]
    fn gate_refuses_overlapping_evaluable_ranges() {
        let mut data = u238_with_formalism(ResonanceFormalism::MLBW);
        let mut second = data.ranges[0].clone();
        second.energy_low = 5.0;
        second.energy_high = 8.0;
        data.ranges.push(second);
        let thermal_u = DopplerParams::new(ROOM_K, data.awr).unwrap().u();
        assert_eq!(
            classify_isotope(&data, &[6.674], thermal_u),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::OverlappingRange {
                    energy_ev: 6.674,
                    other_range_index: 1,
                    formalism: ResonanceFormalism::MLBW,
                }
            }
        );
        // A non-evaluable second range does not count as an overlap.
        data.ranges[1].formalism = ResonanceFormalism::Unresolved;
        data.ranges[1].resolved = false;
        assert_eq!(
            classify_isotope(&data, &[6.674], thermal_u),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::MLBW
            }
        );
    }

    #[test]
    fn adjacent_ranges_use_the_dispatchers_half_open_boundary() {
        // SLBW [1e-5, 100) followed by MLBW [100, 1e4]: a window whose top
        // lands on 100 eV would be evaluated with the MLBW range, so tier 1
        // requires the top to stay strictly below the boundary.
        let mut data = u238_with_formalism(ResonanceFormalism::SLBW);
        data.ranges[0].energy_high = 100.0;
        let mut upper = u238_with_formalism(ResonanceFormalism::MLBW).ranges[0].clone();
        upper.energy_low = 100.0;
        data.ranges.push(upper);
        let thermal_u = DopplerParams::new(ROOM_K, data.awr).unwrap().u();

        match classify_isotope(&data, &[99.9], thermal_u) {
            DopplerRoute::SampledTable {
                reason:
                    SampledTableReason::WindowCrossesRangeBoundary {
                        range_high_ev,
                        formalism,
                        ..
                    },
            } => {
                assert_eq!(range_high_ev, 100.0);
                assert_eq!(formalism, ResonanceFormalism::SLBW);
            }
            other => panic!("expected a boundary demotion, got {other:?}"),
        }
        assert_eq!(
            classify_isotope(&data, &[50.0], thermal_u),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::SLBW
            }
        );
        assert_eq!(
            classify_isotope(&data, &[500.0], thermal_u),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::MLBW
            }
        );
        // Both halves are tier 1; the disclosed formalism is the lowest energy's.
        assert_eq!(
            classify_isotope(&data, &[50.0, 500.0], thermal_u),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::SLBW
            }
        );
    }

    #[test]
    fn energy_outside_every_range_is_a_formalism_failure_without_a_formalism() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let thermal_u = DopplerParams::new(ROOM_K, data.awr).unwrap().u();
        assert_eq!(
            classify_isotope(&data, &[2.5e4], thermal_u),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    energy_ev: 2.5e4,
                    formalism: None,
                }
            }
        );
    }

    #[test]
    fn empty_grid_verdict_is_the_leading_resolved_range() {
        let thermal_u = 0.01;
        assert_eq!(
            classify_isotope(
                &u238_with_formalism(ResonanceFormalism::MLBW),
                &[],
                thermal_u
            ),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::MLBW
            }
        );
        assert!(matches!(
            classify_isotope(
                &u238_with_formalism(ResonanceFormalism::ReichMoore),
                &[],
                thermal_u
            ),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::Formalism {
                    formalism: Some(ResonanceFormalism::ReichMoore),
                    ..
                }
            }
        ));
    }

    #[test]
    fn gate_consults_the_file3_predicate() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let thermal_u = DopplerParams::new(ROOM_K, data.awr).unwrap().u();
        assert_eq!(
            classify_isotope_with(&data, &[6.674], thermal_u, &|_| true),
            DopplerRoute::SampledTable {
                reason: SampledTableReason::File3Background {
                    energy_ev: 6.674,
                    formalism: ResonanceFormalism::MLBW,
                }
            }
        );
        assert_eq!(
            classify_isotope_with(&data, &[6.674], thermal_u, &|_| false),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::MLBW
            }
        );
    }

    /// The pair that must be updated together when MF=3 support lands:
    /// today no fixture carries a File-3 term, and the public gate reads
    /// exactly that predicate.
    #[test]
    fn no_fixture_carries_a_file3_term_and_the_public_gate_reads_the_predicate() {
        for data in [
            u238_with_formalism(ResonanceFormalism::SLBW),
            u238_with_formalism(ResonanceFormalism::MLBW),
            ex001_hydrogen_single_resonance(),
            hf177(),
        ] {
            assert!(data.ranges.iter().all(|r| !r.has_file3_background()));
            let thermal_u = DopplerParams::new(ROOM_K, data.awr).unwrap().u();
            let energy = data.ranges[0]
                .l_groups
                .iter()
                .flat_map(|g| g.resonances.iter())
                .map(|r| r.energy)
                .find(|&e| e > 1.0)
                .unwrap();
            assert!(matches!(
                classify_isotope(&data, &[energy], thermal_u),
                DopplerRoute::Continuous { .. }
            ));
        }
    }

    // ── quadrature behaviour ───────────────────────────────────────────────

    #[test]
    fn zero_temperature_is_exactly_the_resonance_equation() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = [6.5, 6.674, 6.9];
        let params = DopplerParams::new(0.0, data.awr).unwrap();
        let (values, derivatives) =
            broaden_with_derivative(&energies, &data, &params, None).unwrap();
        let expected: Vec<f64> = CrossSectionPlan::new(&data)
            .evaluate(&energies)
            .into_iter()
            .map(|xs| xs.total)
            .collect();
        assert_eq!(values, expected);
        assert_eq!(derivatives, vec![0.0; 3]);
    }

    #[test]
    fn underflowed_thermal_width_is_exactly_unbroadened() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let energies = [6.5, 6.674, 6.9];
        let params = DopplerParams::new(f64::from_bits(1), data.awr).unwrap();
        assert_eq!(params.u(), 0.0);
        let expected: Vec<f64> = CrossSectionPlan::new(&data)
            .evaluate(&energies)
            .into_iter()
            .map(|xs| xs.total)
            .collect();
        assert_eq!(continuous_total(&energies, &data, &params), expected);
    }

    #[test]
    fn rejects_invalid_and_unsorted_grids() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
        assert!(matches!(
            broaden(&[6.5, f64::NAN], &data, &params, None),
            Err(ContinuousDopplerError::InvalidEnergy { index: 1, .. })
        ));
        assert!(matches!(
            broaden(&[6.5, -1.0], &data, &params, None),
            Err(ContinuousDopplerError::InvalidEnergy { index: 1, .. })
        ));
        assert!(matches!(
            broaden(&[6.9, 6.5], &data, &params, None),
            Err(ContinuousDopplerError::UnsortedEnergy { index: 1, .. })
        ));
        assert_eq!(broaden(&[], &data, &params, None), Ok(Vec::new()));
    }

    #[test]
    fn cancellation_is_observed() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
        let cancel = AtomicBool::new(true);
        assert_eq!(
            broaden(&[6.674], &data, &params, Some(&cancel)),
            Err(ContinuousDopplerError::Cancelled)
        );
    }

    #[test]
    fn limit_error_names_the_lowest_failing_energy_regardless_of_scheduling() {
        let data = u238_with_formalism(ResonanceFormalism::MLBW);
        let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
        let budget = QuadratureBudget {
            max_depth: MAX_DEPTH,
            max_active_panels: 1,
        };
        let energies = [6.55, 6.6, 6.65, 6.674, 6.7, 6.75, 6.8];
        let failing: Vec<f64> = energies
            .iter()
            .copied()
            .filter(|&e| {
                broaden_integrals(&[e], &data, &params, Channel::Total, false, budget, None)
                    .is_err()
            })
            .collect();
        assert!(
            failing.len() >= 2,
            "the budget must fail several targets for the test to mean anything: {failing:?}"
        );
        for _ in 0..8 {
            let error = broaden_integrals(
                &energies,
                &data,
                &params,
                Channel::Total,
                false,
                budget,
                None,
            )
            .unwrap_err();
            assert_eq!(
                error,
                ContinuousDopplerError::PanelLimit {
                    energy_ev: failing[0],
                    limit: 1,
                }
            );
        }
    }

    #[test]
    fn result_at_a_target_is_bit_identical_across_grids() {
        // Oracle (e): grid independence, with the sampled route as the
        // control that does depend on the grid.
        let data = hf177();
        let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
        let venus = venus_like_grid(1);
        let index = 1000;
        let target = venus[index];
        let three = [venus[index - 1], target, venus[index + 1]];

        let alone = continuous_total(&[target], &data, &params)[0];
        let among_three = continuous_total(&three, &data, &params)[1];
        let on_venus = continuous_total(&venus, &data, &params)[index];
        assert_eq!(alone, among_three);
        assert_eq!(alone, on_venus);
        assert!(alone.is_finite() && alone > 0.0);

        let sampled_three = sampled_total(&three, &data, &params)[1];
        let sampled_venus = sampled_total(&venus, &data, &params)[index];
        let control = (sampled_three - sampled_venus).abs() / sampled_venus;
        assert!(
            control > 1e-6,
            "the sampled route must depend on the grid for this control to bite: {control:.3e}"
        );
    }

    #[test]
    fn sampled_table_converges_to_the_continuous_value_with_grid_refinement() {
        // Oracle (b): the re-anchor argument. On the VENUS grid itself the
        // sampled route misses area under Hf-177's narrow lines; refining
        // the sampled grid drives it monotonically onto the continuous value.
        let data = hf177();
        let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
        let base = venus_like_grid(1);
        let continuous = continuous_total(&base, &data, &params);

        // Compare only where the kernel window lies inside the sampled grid:
        // at the grid ends the sampled route has no table to convolve and
        // falls back to its declared tail extrapolation, which is a separate
        // behaviour from the sampling density this oracle is about.
        let (grid_low, grid_high) = (base[0], *base.last().unwrap());
        let interior: Vec<usize> = (0..base.len())
            .filter(|&i| {
                let speed = base[i].sqrt();
                (speed - SUPPORT_X * params.u()).powi(2) >= grid_low
                    && (speed + SUPPORT_X * params.u()).powi(2) <= grid_high
            })
            .collect();
        assert!(interior.len() > 3000, "interior points: {}", interior.len());

        let mut deviations = Vec::new();
        for refinement in [1usize, 4, 16, 64] {
            let grid = venus_like_grid(refinement);
            assert_eq!(grid.len(), (base.len() - 1) * refinement + 1);
            let sampled = sampled_total(&grid, &data, &params);
            let max_rel = interior
                .iter()
                .map(|&i| {
                    assert_eq!(grid[i * refinement], base[i]);
                    (sampled[i * refinement] - continuous[i]).abs() / continuous[i]
                })
                .fold(0.0_f64, f64::max);
            deviations.push((refinement, max_rel));
        }
        // Measured: 1.9 (k=1), 0.35 (k=4), 3.7e-3 (k=16), 1.9e-4 (k=64).
        eprintln!("sampled-vs-continuous max relative deviation: {deviations:?}");
        assert!(deviations[0].1 > 1e-2, "k=1 must miss: {deviations:?}");
        assert!(deviations[3].1 < 4e-4, "k=64 must agree: {deviations:?}");
        for pair in deviations.windows(2) {
            assert!(
                pair[1].1 < pair[0].1,
                "refinement must reduce the deviation monotonically: {deviations:?}"
            );
        }
    }

    // ── physics oracles ────────────────────────────────────────────────────

    /// Oracle (a): SAMMY ex001, every point of the reference curve, through
    /// tier 1 at SAMMY's own energies (no interpolation).
    ///
    /// Reference: `samexm_new/ex001_new/answers/ex001a.lst` column 4 (the
    /// Doppler-broadened capture cross-section), computed by SAMMY for a
    /// single resonance at 10 eV (Γγ = 1.0 meV, Γn = 0.5 meV) on a 10.0 amu
    /// target at 300 K. SAMMY's own broadening integrates over an
    /// auto-refined sampled grid and the par file declares an abundance of
    /// 0.999999, so agreement below about 1e-6 is not expected.
    ///
    /// Measured: 7.7e-3 maximum relative deviation over the 315 points, and
    /// the deviation is a smooth amplitude factor of about +0.7% across the
    /// whole curve (peak and both wings alike), not a shape difference. The
    /// quadrature is pinned to 1e-10 by the trapezoid oracle and the
    /// sampled route converges onto these values with grid refinement, so
    /// the residual is between the two codes' unbroadened line strengths,
    /// not in the broadening. The gate is held at 1e-2.
    #[test]
    fn sammy_ex001_full_curve_through_tier_one() {
        let data = ex001_hydrogen_single_resonance();
        let params = DopplerParams::new(300.0, data.awr).unwrap();
        let (energies, reference): (Vec<f64>, Vec<f64>) =
            include_str!("../tests/data/sammy_ex001a_answers.lst")
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    let columns: Vec<f64> = line
                        .split_whitespace()
                        .map(|c| c.parse::<f64>().unwrap())
                        .collect();
                    assert_eq!(columns.len(), 4, "ex001a.lst rows are E, data, unc, theory");
                    (columns[0], columns[3])
                })
                .unzip();
        assert_eq!(energies.len(), 315);
        assert_eq!(
            classify_isotope(&data, &energies, params.u()),
            DopplerRoute::Continuous {
                formalism: ResonanceFormalism::SLBW
            }
        );

        let ours = broaden_channel(&energies, &data, &params, Channel::Capture, None).unwrap();
        let (worst_index, max_rel) = ours
            .iter()
            .zip(&reference)
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs() / b))
            .fold((0, 0.0_f64), |acc, x| if x.1 > acc.1 { x } else { acc });
        eprintln!(
            "ex001 tier 1: max_rel_err={max_rel:.3e} at E={} eV (ours {} vs SAMMY {})",
            energies[worst_index], ours[worst_index], reference[worst_index]
        );
        assert!(max_rel < 0.01, "max relative error {max_rel:.3e}");
    }

    /// Oracle (c): an independent fixed-step trapezoid in kernel coordinates,
    /// with none of this module's panels, breakpoints, rule, or error
    /// estimator — value and temperature derivative.
    #[test]
    fn hf177_matches_uniform_speed_trapezoid_oracle() {
        let data = hf177();
        let target_energy = 8.876_917_538_350_767_f64;
        let temperature_k = 300.0_f64;
        let n_points = 400_001_usize;
        let dx = 2.0 * SUPPORT_X / (n_points - 1) as f64;
        let thermal_u = (BOLTZMANN_EV_PER_K * temperature_k / data.awr).sqrt();
        let target_speed = target_energy.sqrt();
        let plan = CrossSectionPlan::new(&data);
        let normalization = std::f64::consts::PI.sqrt() * target_energy;
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
        let (oracle_value, oracle_derivative) = (value_sum * dx, derivative_sum * dx);

        let params = DopplerParams::new(temperature_k, data.awr).unwrap();
        let (values, derivatives) =
            broaden_with_derivative(&[target_energy], &data, &params, None).unwrap();
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

    /// Oracle (d): the analytic temperature derivative against a five-point
    /// central difference of the value integral.
    #[test]
    fn analytical_temperature_derivative_matches_five_point_difference() {
        let cases: [(ResonanceData, f64, f64); 2] = [
            (u238_with_formalism(ResonanceFormalism::MLBW), 6.674, 300.0),
            (hf177(), 8.876_917_538_350_767, ROOM_K),
        ];
        for (data, energy, temperature) in cases {
            let step = 0.5;
            let params = DopplerParams::new(temperature, data.awr).unwrap();
            let (_, derivative) = broaden_with_derivative(&[energy], &data, &params, None).unwrap();
            let value_at = |offset: f64| {
                let shifted = DopplerParams::new(temperature + offset, data.awr).unwrap();
                continuous_total(&[energy], &data, &shifted)[0]
            };
            let five_point = (value_at(-2.0 * step) - 8.0 * value_at(-step) + 8.0 * value_at(step)
                - value_at(2.0 * step))
                / (12.0 * step);
            let difference = (derivative[0] - five_point).abs();
            let allowance = 1.0e-8 + 1.0e-6 * five_point.abs();
            assert!(
                difference <= allowance,
                "E={energy}: analytical={} five_point={five_point} difference={difference} allowance={allowance}",
                derivative[0]
            );
        }
    }
}
