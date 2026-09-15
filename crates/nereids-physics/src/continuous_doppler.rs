//! Doppler broadening by integrating the free-gas kernel over the resonance
//! equation.
//!
//! ## What is integrated
//!
//! SAMMY manual Eq. III B1.6/B1.7, in velocity space:
//!
//! ```text
//! σ_D(E) = (1/(√π·E)) ∫ e^{−x²} w² · s(w) dx      w = √E + u·x
//! s(w) = +σ(w²)   w > 0
//! s(w) = −σ(w²)   w < 0
//! ```
//!
//! with `u = √(k_B T / A)` the thermal width in √eV. The `w²` weight is what
//! removes the `1/v²` prefactor of the lab-frame convolution, so the
//! integrand is bounded wherever `σ` is.
//!
//! `s` is ODD through `w = 0`: the negative-`w` half is the reflected branch,
//! the target overtaking the neutron. It is not a correction to be dropped at
//! low energy — it is what makes the integral correct there. SAMMY manual
//! Sec. III.B.1: "Negative velocities are included as needed, in order to
//! properly evaluate the integral at low values of E". The share of the
//! kernel it carries is `erfc(√E/u)/2`, which is 24% at `√E/u = 0.5` and 7.9%
//! at 1. [`crate::doppler`] builds the same odd extension for a sampled
//! table.
//!
//! Differentiating at fixed source SPEED — the source energies do not move
//! with `T`, only the weight on them does — turns `d/dT` of `e^{−x²}` into
//! the same integrand times `(x² − ½)/T`. Value and derivative therefore
//! share every panel, and a derivative converged on those panels costs one
//! extra multiply per node rather than a second adaptive pass.
//!
//! ## Why there is no eligibility test
//!
//! Every source energy is evaluated through
//! [`CrossSectionPlan::evaluate_one`], which sums whichever ranges cover it
//! and dispatches SLBW, MLBW and Reich-Moore alike. So a window spanning two
//! ranges, or a Reich-Moore evaluation, needs nothing special: the
//! quadrature's only job is to know where the structure is, which is a
//! question about breakpoints.
//!
//! This matters beyond tidiness. An earlier revision chose between this
//! integral and the sampled table per isotope, from the working grid and the
//! temperature. Both are moved by a fit, so the choice could flip mid-fit and
//! σ stepped where the two methods disagreed. Selecting a method by anything
//! a fit can vary makes the forward model discontinuous in the parameter
//! being fitted; selecting it by what the INPUT IS cannot.
//!
//! ## Quadrature
//!
//! Gauss–Kronrod G10/K21 (QUADPACK `qk21`) with adaptive bisection: the panel
//! with the largest error estimate is split until the total error meets the
//! tolerance. Initial panel edges are the window ends, the zero crossing when
//! the window reaches it, each covering range's bounds, the resonance
//! breakpoints, and every knot of an energy-dependent scattering radius
//! `AP(E′)` inside the window — each is a kink both rules would otherwise
//! straddle and mis-estimate.
//!
//! Every failure is hard. A broadening that cannot converge returns an error
//! rather than a degraded number.
//!
//! ## Not implemented here
//!
//! A range carrying a File-3 (MF=3) smooth background is integrated without
//! it, because only File 2 is parsed and nothing can answer whether a range
//! has one. Whichever change adds MF=3 parsing owes this.
//!
//! Below a resolved range's lower bound the dispatcher returns zero, while
//! [`crate::doppler`] extrapolates 1/v. The two therefore disagree about a
//! window reaching under that bound. Both are approximations of a File-3
//! background neither can see.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use nereids_endf::resonance::ResonanceData;
use rayon::prelude::*;

use crate::doppler::{DopplerError, DopplerParams, validate_doppler_grid, zero_negative_value};
use crate::reich_moore::{CrossSectionPlan, CrossSections};

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
    /// SAMMY Eq. III B1.6 integrates `w²·s(w)` over the source speed
    /// `w = √E + u·x`, with
    ///
    /// ```text
    /// s(w) = +σ(w²)   w > 0
    /// s(w) = −σ(w²)   w < 0
    /// ```
    ///
    /// so the integrand is ODD through `w = 0` and passes through zero
    /// there. The negative-`w` half is the reflected branch: the target
    /// overtaking the neutron. It is what makes the integral correct at low
    /// energy, where the thermal window reaches below zero speed — SAMMY
    /// manual Sec. III.B.1, "Negative velocities are included as needed, in
    /// order to properly evaluate the integral at low values of E". The
    /// sampled path builds the same extension in
    /// [`build_extended_fgm_grid`](crate::doppler).
    ///
    /// `evaluate_one` is called with `w²`, which is positive whenever `w`
    /// is non-zero, and `w == 0` returns early — so its positive-energy
    /// assertion cannot fire for any `x`.
    fn integrand(&self, x: f64) -> (f64, f64) {
        let source_speed = self.target_speed + self.thermal_u * x;
        if source_speed == 0.0 {
            return (0.0, 0.0);
        }
        let source_energy = source_speed * source_speed;
        let sigma = self.channel.pick(&self.plan.evaluate_one(source_energy));
        if sigma > 0.0 {
            self.any_source_positive.set(true);
        }
        let value = (-x * x).exp() * source_speed.signum() * source_energy * sigma
            / (SQRT_PI * self.target_energy);
        // The derivative costs a multiply and a divide at every node, and
        // the value-only entry points discard it, so it is not computed
        // for them.
        let derivative = if self.require_derivative {
            value * (x * x - 0.5) / self.temperature_k
        } else {
            0.0
        };
        (value, derivative)
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
    fn breakpoints(&self, data: &ResonanceData) -> Vec<f64> {
        let speed_low = self.target_speed - SUPPORT_X * self.thermal_u;
        let speed_high = self.target_speed + SUPPORT_X * self.thermal_u;
        // A window reaching below zero speed covers every energy down to
        // zero on its reflected branch, so the culling bound below must not
        // be `speed_low²` — that would discard resonances the window
        // actually sees.
        let folds = speed_low < 0.0;
        let low_energy = if folds { 0.0 } else { speed_low * speed_low };
        let high_energy = speed_high * speed_high;
        let mut points = vec![-SUPPORT_X, SUPPORT_X];
        // `w²·s(w)` is continuous through `w = 0` but has a kink there, and
        // Gauss-Kronrod converges slowly across a kink it is not told
        // about. Make the crossing a panel boundary.
        if folds {
            points.push(-self.target_speed / self.thermal_u);
        }
        let mut push_source_energy = |source_energy: f64| {
            if source_energy <= 0.0 {
                return;
            }
            let speed = source_energy.sqrt();
            // A folded window reaches one source energy at BOTH ±√E. When
            // it does not fold, the reflected coordinate lands outside
            // `±SUPPORT_X` and the filter drops it, so no test is needed.
            for signed_speed in [speed, -speed] {
                let coordinate = (signed_speed - self.target_speed) / self.thermal_u;
                if coordinate > -SUPPORT_X && coordinate < SUPPORT_X {
                    points.push(coordinate);
                }
            }
        };
        // Every range the window reaches, not just the target's own. The
        // cross-section dispatcher already evaluates each source energy
        // with whichever range covers it, so the quadrature's only job is
        // to know where the structure is.
        for range in &data.ranges {
            if !range.is_evaluable()
                || range.energy_high < low_energy
                || range.energy_low > high_energy
            {
                continue;
            }
            // σ can step where one range stops contributing and the next
            // starts, so the edges are panel boundaries.
            push_source_energy(range.energy_low);
            push_source_energy(range.energy_high);
            for group in &range.l_groups {
                for resonance in &group.resonances {
                    let total_width = resonance.gn.abs()
                        + resonance.gg.abs()
                        + resonance.gfa.abs()
                        + resonance.gfb.abs();
                    // Skip resonances whose Lorentzian cannot reach the
                    // window; their breakpoints would all be clipped anyway.
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
        }
        points.sort_by(f64::total_cmp);
        points.dedup_by(|left, right| left.to_bits() == right.to_bits());
        points
    }

    fn integrate(&self, data: &ResonanceData) -> Result<TargetIntegral, DopplerError> {
        let points = self.breakpoints(data);
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
/// `awr`, so the kernel width and the resonance equation can never describe
/// different nuclides. Taking a caller-supplied [`DopplerParams`] would
/// allow that: an AWR wrong by 0.87% moves the width by 0.43%, which is
/// below every tolerance in this module.
///
/// At zero temperature, or an underflowed `u`, there is no kernel and the
/// values are the unbroadened equation.
///
/// # Errors
/// [`DopplerError`] for an invalid grid or temperature, or for a quadrature
/// that cannot converge inside `budget`.
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

    // Each target is an independent integral, so targets are the unit of
    // parallelism — the common thermometry case has ONE isotope, so
    // per-isotope parallelism would leave this serial. Results are gathered
    // in grid order so the reported error is always the lowest-index
    // failure, whatever order the threads finished in.
    let results: Vec<Result<TargetIntegral, DopplerError>> = energies
        .par_iter()
        .map(|&target_energy| {
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
            .integrate(data)
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
    use nereids_endf::resonance::test_support::{
        ex001_hydrogen_single_resonance, synthetic_swave_slbw, u238_with_formalism,
    };
    use nereids_endf::resonance::{Resonance, ResonanceFormalism};

    /// Room temperature. For the U-238 fixtures 8u ≈ 0.083 √eV, so the
    /// window at 6.674 eV spans about ±0.43 eV.
    const ROOM_K: f64 = 293.6;

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
    /// stand 0.8% from SAMMY, so the residual is not this integrator.
    ///
    /// ## Why SAMMY is the low one
    ///
    /// SAMMY's own run log for this case (`samexm/ex001/answers/
    /// ex001aa.lpt`) reports `** One resonance has fewer than 9 points
    /// across width` and `Number of points in auxiliary grid = 396`. That
    /// is 396 points spanning 8.04-11.96 eV, about 9.9 meV apart, against a
    /// natural width `Γ = Γn + Γγ = 1.5 meV` — SAMMY samples the Lorentzian
    /// at roughly 0.15 points across its OWN width before convolving.
    /// Under-sampling a narrow line loses line area, so SAMMY's broadened
    /// curve comes out low; we place quadrature breakpoints ON the
    /// resonance and integrate it to 1e-8, so we keep the area. A loss on
    /// SAMMY's side is the direction and the localisation we measure: the
    /// deficit lives in the core and the wings, where the line is
    /// resolved, agree to 1e-4.
    ///
    /// So the residual is SAMMY's grid, not our kernel — but it is still
    /// 0.77%, so it is pinned tightly rather than waved through. The
    /// assertion below is a two-sided band on the value actually observed;
    /// a loose tolerance would absorb a sub-0.8% physics error, which is
    /// larger than the 0.43% kernel-width error this very branch removed.
    ///
    /// The same log independently confirms the mass ratio this branch
    /// corrected: it prints `mass of neutron = 1.008664915600000 in amu`
    /// and `Dopp_FWHM` at 10 eV as 0.5378 eV, which is the width AWR
    /// 9.9141 gives (0.537766) and not the one AWR 10 gives (0.535451).
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
        // path, rather than dividing by a zero width — and the ROUTE must
        // say so too, or it would disclose a continuous integral over a
        // curve that was never broadened.
        let tiny = f64::from_bits(1);
        assert_eq!(DopplerParams::new(tiny, data.awr).unwrap().u(), 0.0);
        assert_eq!(broaden(&energies, &data, tiny).unwrap(), expected);

        // Every formalism the cross-section dispatcher evaluates goes
        // through the same integral, so Reich-Moore is not a special case
        // here or anywhere else.
        let rm = u238_with_formalism(ResonanceFormalism::ReichMoore);
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

    /// The integral must cover the part of the thermal window that folds
    /// through zero velocity.
    ///
    /// SAMMY Eq. III B1.6 integrates `w²·s(w)` with `s(w) = σ(w²)` for
    /// `w > 0` and `−σ(w²)` for `w < 0` — an ODD integrand through the
    /// origin. The sampled path builds exactly that odd extension
    /// (`build_extended_fgm_grid`), quoting the SAMMY manual Sec. III.B.1:
    /// negative velocities are included "in order to properly evaluate the
    /// integral at low values of E".
    ///
    /// How much the reflected branch matters is set by `√E/u`, and NOT by
    /// whether the truncated window happens to fold. The fraction of kernel
    /// mass on the reflected side is `erfc(√E/u)/2`: it is 24% at
    /// `√E/u = 0.5`, 7.9% at 1, and already 2e-3 at 2. Choosing targets by
    /// the old gate's `√E < 8u` instead would put them at `√E/u ≈ 4`, where
    /// the reflected mass is 1e-8 and the test measures nothing.
    ///
    /// `awr = 1` makes `u` large, so these ratios occur at energies well
    /// above the fixture's 1e-5 eV resolved-range floor — below that floor
    /// the dispatcher returns zero while the sampled path extrapolates
    /// 1/v, and keeping the targets clear of it keeps that disagreement out
    /// of this measurement.
    ///
    /// Measured against the sampled reference: with the sign the integral
    /// agrees to 2e-4 or better; without it the error is 34%, 11% and 3.7%
    /// at the three ratios.
    #[test]
    fn the_reflected_branch_carries_the_integral_at_low_energy() {
        let data = synthetic_swave_slbw(1.0, 5.0, 1.0e-3, 2.0e-2, 5.0);
        let params = DopplerParams::new(ROOM_K, data.awr).unwrap();
        let thermal_u = params.u();

        let step = 2.0e-5;
        let grid: Vec<f64> = (1..=20_000).map(|i| f64::from(i) * step).collect();
        let table: Vec<f64> = CrossSectionPlan::new(&data)
            .evaluate(&grid)
            .into_iter()
            .map(|xs| xs.total)
            .collect();
        let sampled = doppler_broaden(&grid, &table, &params).unwrap();

        for ratio in [0.5_f64, 0.75, 1.0] {
            let index = grid
                .iter()
                .position(|&e| (e - (ratio * thermal_u).powi(2)).abs() < 0.5 * step)
                .expect("target on the sampled grid");
            let target = grid[index];

            // Non-vacuity: at least 5% of the kernel must sit on the
            // reflected side, or this target proves nothing about it.
            let reflected_mass = 0.5 * erfc_approximation(ratio);
            assert!(
                reflected_mass > 0.05,
                "√E/u = {ratio} leaves only {reflected_mass:.2e} reflected mass"
            );

            let ours = broaden(&[target], &data, ROOM_K).unwrap()[0];
            let want = sampled[index];
            let relative = (ours - want).abs() / want.abs();
            assert!(
                relative < 1.0e-3,
                "at √E/u = {ratio} (E = {target:.3e} eV, {reflected_mass:.3e} of \
                 the kernel reflected) the integral gives {ours:.6e} against the \
                 sampled reference {want:.6e} — {relative:.3e} relative"
            );
        }
    }

    /// Abramowitz & Stegun 7.1.26. Only used to state how much kernel mass a
    /// test target puts on the reflected side, so ~1e-7 absolute is ample.
    fn erfc_approximation(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        poly * (-x * x).exp()
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

    /// Every formalism the cross-section dispatcher evaluates is broadened
    /// by the same integral.
    ///
    /// The integrand calls [`CrossSectionPlan::evaluate_one`], which
    /// dispatches SLBW, MLBW and Reich-Moore alike, so there is nothing for
    /// the broadening to special-case. Reich-Moore used to be refused here,
    /// which is what made the choice of method a runtime decision.
    ///
    /// The results must also DIFFER between formalisms, or the test would
    /// pass on an integrand that ignored the formalism entirely.
    #[test]
    fn every_formalism_the_dispatcher_evaluates_is_integrated() {
        let targets = [6.5, 6.674, 6.9];
        let mut results = Vec::new();
        for formalism in [
            ResonanceFormalism::SLBW,
            ResonanceFormalism::MLBW,
            ResonanceFormalism::ReichMoore,
        ] {
            let data = u238_with_formalism(formalism);
            let values = broaden(&targets, &data, ROOM_K)
                .unwrap_or_else(|e| panic!("{formalism:?} was not integrated: {e:?}"));
            assert!(
                values.iter().all(|v| v.is_finite() && *v > 0.0),
                "{formalism:?} produced {values:?}"
            );
            results.push((formalism, values));
        }
        for pair in results.windows(2) {
            let ((left, a), (right, b)) = (&pair[0], &pair[1]);
            assert!(
                a.iter().zip(b).any(|(x, y)| x != y),
                "{left:?} and {right:?} broadened identically, so the \
                 formalism never reached the integrand"
            );
        }
    }
}
