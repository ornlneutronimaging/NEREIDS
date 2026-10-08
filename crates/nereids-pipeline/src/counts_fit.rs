//! The areal densities and temperatures of the materials in one or more
//! regions of the detector, each region's background, the sample run's
//! normalization, the timing offset, the flight path and the pulse's numbers,
//! fitted to the counts of an open-beam run and a sample run recorded in the
//! same time bins.

use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use faer::Mat;
use nereids_endf::resonance::ResonanceData;
use nereids_fitting::error::FittingError;
use nereids_fitting::lm::{FitModel, FlatMatrix};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{Prior, Unbounded, information_inverse};
use nereids_fitting::statistics::{Consistency, consistency};
use nereids_physics::continuous_doppler::{SUPPORT_X, broaden_with_derivative};
use nereids_physics::doppler::DopplerParams;
use nereids_physics::flight_time_grid::{FlightTimeGrid, Rows};
use nereids_physics::ikeda_carpenter::IkedaCarpenterParams;
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::resonance_center_energies;
use rayon::prelude::*;

use crate::beam::BeamSpline;
use crate::error::PipelineError;
use crate::open_beam::{
    BOUND, COUNTS_TO_MEASURE_NOISE, Calibration, OpenBeamFit, PULSE_NUMBERS, Pulse, Recorded,
    combined, counted, fit_on_halved_grids, fit_open_beam, laws, overdispersion, validate_counts,
    validate_live, weights_of,
};
use crate::pipeline::TEMPERATURE_BOUNDS_K;

/// A count in a bin predicted fewer counts than this has a chance below it
/// under the model.
pub const NEGLIGIBLE_PREDICTION: f64 = 1e-10;

const SETTLED_OVERDISPERSION: f64 = 0.01;

const MOST_PASSES: usize = 20;

/// A quantity the fit holds at a known value, fits from a starting value
/// over the quantity's range, fits from `start` within `lower..=upper` inside
/// that range, or fits over the range from a `value` measured elsewhere with
/// standard deviation `sd`, a Gaussian prior on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Known(f64),
    Fitted(f64),
    Within { start: f64, lower: f64, upper: f64 },
    Measured { value: f64, sd: f64 },
}

impl Value {
    pub(crate) fn parameter(
        self,
        name: impl Into<Cow<'static, str>>,
        range: RangeInclusive<f64>,
        allowed: &str,
    ) -> Result<FitParameter, PipelineError> {
        let name = name.into();
        let (value, lower, upper) = match self {
            Self::Known(v) | Self::Fitted(v) | Self::Measured { value: v, .. } => {
                (v, *range.start(), *range.end())
            }
            Self::Within {
                start,
                lower,
                upper,
            } => (start, lower, upper),
        };
        if !(value.is_finite()
            && range.contains(&lower)
            && lower < upper
            && range.contains(&upper)
            && (lower..=upper).contains(&value)
            && !matches!(self, Self::Measured { sd, .. } if !(sd.is_finite() && sd > 0.0)))
        {
            return Err(PipelineError::InvalidParameter(match self {
                Self::Within { .. } => format!(
                    "{name} must be {allowed}, with bounds lower < upper and a finite start \
                     between them; got {self:?}"
                ),
                Self::Measured { .. } => format!(
                    "{name} must be finite and {allowed}, with a finite positive sd; got {self:?}"
                ),
                _ => format!("{name} must be finite and {allowed}; got {self:?}"),
            }));
        }
        Ok(FitParameter {
            name,
            value,
            lower,
            upper,
            fixed: matches!(self, Self::Known(_)),
        })
    }
}

/// An open-beam run and a sample run recorded in the same time bins, read in
/// one or more regions of the detector.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Time-bin edges in µs.
    pub time_edges_us: Vec<f64>,
    /// `c_q`, the sample run's proton charge over the open-beam run's.
    pub charge_ratio: f64,
    /// `a`, the normalization of the sample run, positive, the same in every
    /// region.
    pub normalization: Value,
    /// The regions, separate sets of pixels timed by one clock.
    pub regions: Vec<Region>,
}

/// The counts of a set of pixels in both runs, its background and the
/// material the beam crosses there.
#[derive(Debug, Clone)]
pub struct Region {
    /// Raw counts of the open-beam run, one per bin.
    pub open_counts: Vec<f64>,
    /// Raw counts of the sample run, one per bin.
    pub sample_counts: Vec<f64>,
    /// The fraction of the open-beam run's neutrons arriving in each bin that
    /// the detector records, in (0, 1]; `None` records every one.
    pub open_live: Option<Vec<f64>>,
    /// The fraction of the sample run's neutrons arriving in each bin that the
    /// detector records, in (0, 1]; `None` records every one.
    pub sample_live: Option<Vec<f64>>,
    /// `b0` (dimensionless), `b1` in √eV and `b2` in 1/√eV of the background
    /// `b(E) = b0 + b1/√E + b2·√E`, each any real number.
    pub background: [Value; 3],
    /// The material the beam crosses in the region; `None` for nothing, which
    /// transmits every neutron.
    pub material: Option<Material>,
}

/// A material's isotopes and temperature.
#[derive(Debug, Clone)]
pub struct Material {
    /// Each isotope with its areal density in atoms/barn, known, fitted or
    /// measured, at least 0.
    pub isotopes: Vec<(ResonanceData, Value)>,
    /// The temperature in K, within 1–5000 K.
    pub temperature_k: Value,
}

/// The fitted quantities of every region, and the normalization, timing
/// offset, flight path and pulse numbers the regions share.
#[derive(Debug, Clone)]
pub struct CountsFit {
    /// Each region's fitted quantities and counts, in the order given.
    pub regions: Vec<RegionFit>,
    /// The normalization `a`: the known one, or the fitted one.
    pub normalization: f64,
    /// The timing offset `t0` in µs: the known one, or the fitted one.
    pub t0_us: f64,
    /// The flight path in m: the known one, or the fitted one.
    pub flight_path_m: f64,
    /// The pulse's `[α₀, α₁]`, each the known or the fitted one.
    pub alpha: [f64; 2],
    /// The pulse's `[β₀, β₁]`, each the known or the fitted one.
    pub beta: [f64; 2],
    /// The pulse's `R`: the known one, or the fitted one.
    pub r: f64,
    /// The square of the pulse's triangle FWHM, `h²` in µs²: the known one, or
    /// the fitted one.
    pub fwhm_squared_us2: f64,
    /// Covariance of the fitted quantities among, in this order, each region's
    /// densities, in the order given, and temperature, region by region; the
    /// normalization; each region's `b0`, `b1` and `b2`, region by region;
    /// `t0`; the flight path; `α₀`, `α₁`, `β₀`, `β₁`, `R` and `h²`: the inverse
    /// of the information at the fit, each run's expected information over its
    /// overdispersion plus `1/sd²` for each measured quantity and `C⁻¹` over
    /// the pulse numbers a calibration's
    /// [`Pulse::prior`](crate::open_beam::Pulse::prior) covers.  The row and
    /// column of a quantity on one of its bounds, or that neither the counts
    /// nor a measurement determine, are NaN, and the other entries are
    /// conditional on every quantity that ended on a bound being held there;
    /// every entry is NaN when the fitted temperature of any region ends at
    /// 1 K or 5000 K.  `None` when the fit did not converge.
    ///
    /// The error bars take every known quantity as exact.
    /// They are not reliable where the counts barely determine a fitted
    /// temperature or barely separate it from a density, as at few counts or
    /// for a thin sample at modest counts.
    pub covariance: Option<FlatMatrix>,
    /// The fitted quantities' Gaussian without their bounds, in the
    /// covariance's order (see [`Unbounded`]), NaN where `covariance` is at a
    /// temperature edge.  `None` when `covariance` is, or when the fitter
    /// reports none.
    pub unbounded: Option<Unbounded>,
    /// Whether each fitted quantity, in the covariance's order, ended on one
    /// of its bounds.
    pub on_bound: Vec<bool>,
    /// Each run's half Poisson deviance over the overdispersion it was
    /// weighted with, summed over both runs of every region at the fit.
    pub deviance: f64,
    /// Whether the fitter converged, every sample run's overdispersion
    /// settled, the grid met its rule at the fitted temperatures and its
    /// flight times cover the fitted `t0`, flight path and pulse.
    pub converged: bool,
    /// For each measured quantity, in the covariance's order, its value
    /// without the fit's bounds ([`Self::unbounded`]) less its measurement
    /// over the standard deviation of that difference, `√(sd² − variance)`,
    /// the variance also without the bounds: near 0 ± 1 when the counts agree
    /// with the measurement.  NaN for a quantity the counts and measurements
    /// do not determine, and for every quantity when a fitted temperature ends
    /// at 1 K or 5000 K; it loses precision as the counts' information on the
    /// quantity vanishes beside the measurement's.  `None` when `unbounded`
    /// is.
    pub measured_pulls: Option<Vec<f64>>,
    /// Whether the counts accept the pulse's calibration on the numbers it
    /// covers, by [`consistency`] with the fit's numbers and covariance
    /// without their bounds ([`Self::unbounded`]).  The counts' information is
    /// divided by each run's overdispersion, so the test weakens as the
    /// overdispersion grows.  The
    /// overdispersion is at least 1, so with Poisson counts its estimate
    /// inflates the covariances the test uses, and `p` rejects a right
    /// calibration less often than its value says, the more so the fewer bins
    /// each run counts.  A calibration fitted to the same open-beam run is not
    /// independent of the fit, which the test does not account for.  `None`
    /// without a calibration, when `unbounded` is `None` or withheld, or when
    /// [`consistency`] gives none.
    pub pulse_consistency: Option<Consistency>,
    /// Step, in µs, of the fit's grid.
    pub step_us: f64,
    /// Number of points of that grid.
    pub points: usize,
    /// How many times the grid of
    /// [`FlightTimeGrid::new`](nereids_physics::flight_time_grid::FlightTimeGrid::new)
    /// was halved to reach it; that grid is built at the starting `t0`,
    /// flight path and pulse, or at fitted ones when the fit rebuilt it.
    pub halvings: usize,
}

/// One region's fitted quantities and counts.
#[derive(Debug, Clone)]
pub struct RegionFit {
    /// Areal density of each isotope of the region's material in atoms/barn,
    /// in the order given: the known one, or the fitted one; empty without a
    /// material.
    pub densities: Vec<f64>,
    /// The material's temperature in K: the known one, or the fitted one;
    /// `None` without a material.
    pub temperature_k: Option<f64>,
    /// `b0` (dimensionless), `b1` in √eV and `b2` in 1/√eV, each the known or
    /// the fitted one.
    pub background: [f64; 3],
    /// The region's beam per µs, as a function of the arrival time less the
    /// starting `t0`, fitted to both runs, with the intervals the region's
    /// open-beam fit chose.
    pub beam: BeamSpline,
    /// Whether the region's open-beam fit chose its richest beam; see
    /// [`OpenBeamFit::at_limit`](crate::open_beam::OpenBeamFit::at_limit).
    pub beam_at_limit: bool,
    /// The open-beam and sample counts the fit predicts in each bin at its
    /// answer, live fractions included, on the grid it accepted: the counts
    /// [`CountsFit::deviance`] compares with the measured ones.
    pub predicted: [Vec<f64>; 2],
    /// Variance of the counts of the open-beam run, then of the sample run,
    /// over their Poisson variance: the value each run's counts are divided by
    /// in the returned fit.  The open-beam run's is the region's open-beam
    /// fit's [`OpenBeamFit::overdispersion`](crate::open_beam::OpenBeamFit::overdispersion); the sample run's is measured the
    /// same way, on the bins the first fit predicts at least one count, by the
    /// previous fit, and within 1% by the returned one when the fit converged.
    /// `None` when the run's counts have not measured it; the open-beam run is
    /// then weighted with 1, and the sample run as the open-beam run.
    pub overdispersion: [Option<f64>; 2],
}

/// Fit the areal densities and temperatures of the regions' materials, each
/// region's background terms, the normalization, and the timing offset,
/// flight path and pulse numbers of `calibration`, that are not known, to the
/// raw counts of both runs in every region.  On a uniform grid of flight
/// times `u_i`, energies `E_i` and step `w`, the counts of region `r` in bin
/// `k` are
///
/// ```text
/// O_rk   = ℓ^O_rk · w Σ_i φ_ri P_ki
/// S_rk   = ℓ^S_rk · c_q · a · w Σ_i φ_ri [T_ri + b_r(E_i)] P_ki
/// T_ri   = exp(−Σ_m n_rm σ_m(E_i; T_r)), or 1 without a material
/// b_r(E) = b0_r + b1_r/√E + b2_r·√E
/// ```
///
/// with `φ_r` the region's beam per µs, `P_ki` the chance of a neutron at
/// `u_i` arriving in bin `k`, `n_rm` and `σ_m` each isotope's density and
/// total cross section Doppler-broadened at the material's temperature `T_r`,
/// `c_q` the charge ratio, `a` the normalization and `ℓ^O_rk`, `ℓ^S_rk` each
/// run's live fraction.  The background is beam neutrons that reach the
/// detector another way, so it passes through the pulse and scales with the
/// normalization; SAMMY's `BackA`, `BackB`, `BackC` (`cro/mnrm1.f90`) are
/// `a·b0`, `a·b1`, `a·b2`, to within the background's change over the pulse's
/// delay.  SAMMY's `BackD·exp(−BackF/√E)` term, and counts that bypass the
/// pulse, such as gammas, are not modelled.
///
/// The regions share the normalization, the run pair's scale; and `t0`, the
/// flight path and the pulse, which one clock, flight path and moderator set
/// for every pixel, and with them the grid and `P_ki`.  Each region has its
/// own beam, from its own open-beam counts, with the intervals its
/// [`fit_open_beam`] chooses, fitted with the rest to both runs; and its own
/// background, which depends on what surrounds its pixels.  A region without
/// a material counts `c_q·a·(1 + b_r)` of its beam in the sample run, so it
/// pins the normalization when its `b0` is known, as zero in the open-region
/// normalization of imaging, and nothing about it when its `b0` is fitted.
/// The regions must be separate pixels: the covariance takes every bin as
/// independent.
///
/// The [`Calibration`]'s `t0` and flight path `L` are fitted unless known.
/// The grid is built at a `t0₀` and `L₀`, at first the starting ones.  Grid
/// point `i` keeps its energy and arrives at `t0 + (L/L₀)·u_i`, with `u_i`
/// its flight time at `t0₀` and `L₀`, standing for `(L/L₀)·w` of flight time;
/// the beams are functions of the arrival time less the starting `t0`.  This
/// is SAMMY's energy scale: its `Tzero` is `t0`, and its `Elzero` times its
/// flight path is `L` (`dat/mdat0.f90`, `Mtzero`).
///
/// The pulse's numbers are fitted unless known, each point's `P_ki` taken at
/// the pulse's laws at its energy `E_i`.  A trial pulse whose `α` or `β` is 0
/// at some energy is outside the model and rejected; a fit driven there ends
/// unconverged.  The open-beam fits that start the beams and weight the
/// open-beam runs use the starting pulse.
///
/// The grid's first step is at most half the narrowest Doppler full width at
/// half maximum, in flight time, of any resonance of any region inside its
/// energy span, at that region's starting temperature; it is then halved
/// until the counts of every run of every region, together, meet
/// [`BOUND`].  The step is uniform and shared, so a
/// wide window whose span holds a narrow resonance at high energy, or a low
/// fitted temperature in any region, can exceed the grid's point cap; a
/// temperature a region's counts barely determine can run to 1 K and refuse
/// the fit that way.
///
/// The fit minimizes each run's half Poisson deviance over its
/// overdispersion, plus `½((x − value)/sd)²` for each [`Value::Measured`]
/// quantity `x`, and `½(θ − m)ᵀC⁻¹(θ − m)` over the pulse numbers `θ` a
/// calibration's [`Pulse::prior`](crate::open_beam::Pulse::prior) covers,
/// with `m` and `C` that prior's mean, which may lie past a number's bound,
/// and covariance.  Each region's open-beam run is weighted with its
/// open-beam fit's overdispersion, and its sample run first with the same.
/// The fit is repeated from its answer while any sample run's
/// overdispersion, measured on the bins the first fit predicts at least one
/// count, changes by more than 1%, the coarser grid of the accepted pair is
/// wider than the rule at the fitted temperatures, or its flight times miss
/// some that the fitted `t0`, `L` and pulse need.  The next first grid is the
/// finer of that pair's coarser grid and the rule's grid at the fitted
/// temperatures, or, when the flight times miss, the rule's grid of a grid
/// built at the fitted `t0`, `L` and pulse.  After twenty fits it is reported
/// unconverged.
///
/// The fitter finds a local minimum.  A thin sample hotter than about
/// 1,500 K fitted from room temperature can end in a false one, reported
/// converged with an overdispersion far above 1.  Where a black resonance
/// empties bins and the background is near zero, a fitted background with no
/// lower bound can drive a black bin's prediction to zero, and the fit ends
/// unconverged; with `b1` and `b2` known, `b0` bounded below by 0 ends on
/// that bound, converged.
///
/// # Errors
/// [`PipelineError::ShapeMismatch`] unless each run of each region has one
/// count, and one live fraction when given, per bin;
/// [`PipelineError::InvalidParameter`] if a count is not a whole non-negative
/// number, a run has no counts, a live fraction is not in (0, 1], the charge
/// ratio is not finite and positive, a known, starting or measured value is
/// not finite and in its quantity's range, a measured value's sd is not finite
/// and positive, bounds are not `lower < upper` in that range
/// with the start between them, the pulse's calibration covers a number that
/// is not fitted or is measured, no region holds a material, a material has
/// no isotopes or lists one twice, an isotope's resonance data are not
/// finite, an isotope not known to be absent has a resonance between the
/// energies of the last and first time edges, at the starting `t0` and flight
/// path or a converged fit's, outside the pulse's
/// [`line_span_ev`](crate::open_beam::Pulse::line_span_ev), or the energies
/// its broadened cross section reads on the grid at the starting `t0`, flight
/// path and pulse, or at the fitted ones of any pass that rebuilds it, at the
/// region's known temperature or at the upper bound of a fitted one, down to
/// zero for a window within the thermal spread of zero energy, are not inside
/// a single one of its evaluated (SLBW, MLBW or Reich–Moore) resolved ranges;
/// [`PipelineError::UnmodelledCounts`] if at the fit, converged or not, a bin
/// is predicted negative or non-finite counts, or holds counts predicted below
/// [`NEGLIGIBLE_PREDICTION`]: starting or known values the fitter cannot
/// leave;
/// everything [`fit_open_beam`] refuses for any region;
/// [`PipelineError::FlightTimeGrid`] for the grid's refusals at the starting
/// `t0`, flight path and pulse or the fitted ones of any pass, including more
/// points than it allows; [`PipelineError::Fitting`] if the fitter fails, the
/// cross sections fail at the start, or a decomposition of the pulse's
/// consistency test fails; a failure at a trial temperature is a rejected
/// step.
pub fn fit_counts(
    measurement: &Measurement,
    calibration: &Calibration,
) -> Result<CountsFit, PipelineError> {
    if measurement
        .regions
        .iter()
        .all(|region| region.material.is_none())
    {
        return Err(PipelineError::InvalidParameter(
            "no region of the measurement holds a material to fit".into(),
        ));
    }
    fit_counts_answer(measurement, calibration, None).map(|(fit, _)| fit)
}

pub(crate) struct Starts {
    pub(crate) beams: Vec<BeamStart>,
    pub(crate) origin_us: f64,
    pub(crate) shared_leverage: Option<Vec<f64>>,
    pub(crate) bound: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct BeamStart {
    pub(crate) beam: BeamSpline,
    pub(crate) overdispersion: Option<f64>,
    pub(crate) at_limit: bool,
}

impl From<OpenBeamFit> for BeamStart {
    fn from(open: OpenBeamFit) -> Self {
        Self {
            beam: open.beam,
            overdispersion: open.overdispersion,
            at_limit: open.at_limit,
        }
    }
}

pub(crate) struct Checked<'a> {
    shape: Vec<Option<usize>>,
    sample_parameters: Vec<FitParameter>,
    measured: Vec<(usize, f64, f64)>,
    lines: Vec<Lines<'a>>,
    instrument: Vec<FitParameter>,
    base: FlightTimeGrid,
    resonances_in_span: Vec<Vec<Vec<f64>>>,
}

pub(crate) fn checked<'a>(
    measurement: &'a Measurement,
    calibration: &Calibration,
) -> Result<Checked<'a>, PipelineError> {
    let Measurement {
        time_edges_us,
        charge_ratio,
        regions,
        ..
    } = measurement;
    let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
    let materials = || {
        regions
            .iter()
            .enumerate()
            .filter_map(|(r, region)| Some((r, region.material.as_ref()?)))
    };
    if !(charge_ratio.is_finite() && *charge_ratio > 0.0) {
        return invalid(format!(
            "the charge ratio must be finite and positive, got {charge_ratio}"
        ));
    }
    for (r, material) in materials() {
        if material.isotopes.is_empty() {
            return invalid(format!("the material of region {r} has no isotopes"));
        }
        for (i, (isotope, _)) in material.isotopes.iter().enumerate() {
            if material.isotopes[..i]
                .iter()
                .any(|(other, _)| other.za == isotope.za)
            {
                return invalid(format!(
                    "{} is listed twice in region {r}; its densities cannot be told apart",
                    isotope.isotope
                ));
            }
            if !finite(isotope) {
                return invalid(format!(
                    "the resonance data of {} in region {r} are not finite",
                    isotope.isotope
                ));
            }
        }
    }
    let shape = shape(measurement);
    let quantity_roles = roles(&vec![0; regions.len()], &shape);
    let sample_parameters = quantity_roles
        .iter()
        .filter_map(|&role| {
            let (value, name, range, allowed) = quantity(measurement, role)?;
            Some(value.parameter(name, range, &allowed))
        })
        .collect::<Result<Vec<FitParameter>, PipelineError>>()?;
    let measured: Vec<(usize, f64, f64)> = quantities(measurement, calibration)
        .enumerate()
        .filter_map(|(offset, (_, value))| match *value {
            Value::Measured { value, sd } => Some((offset, value, sd)),
            _ => None,
        })
        .collect();
    let lines: Vec<Lines<'_>> = materials()
        .map(|(region, material)| {
            let temperature = &sample_parameters[quantity_roles
                .iter()
                .position(|&role| role == Role::Temperature { region })
                .expect("a material has a temperature")];
            let reach_k = if temperature.fixed {
                temperature.value
            } else {
                temperature.upper
            };
            let dopplers = material
                .isotopes
                .iter()
                .map(|(isotope, _)| {
                    DopplerParams::new(reach_k, isotope.awr).map_err(|e| {
                        in_region(region, PipelineError::InvalidParameter(e.to_string()))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Lines {
                region,
                isotopes: &material.isotopes,
                dopplers,
            })
        })
        .collect::<Result<_, PipelineError>>()?;
    let instrument = calibration.instrument()?;
    validate_pulse_prior(calibration)?;
    let start: Vec<f64> = instrument.iter().map(|parameter| parameter.value).collect();
    let base = FlightTimeGrid::new(
        time_edges_us,
        start[0],
        start[1],
        &calibration.pulse.at(&start[2..])?,
    )?;
    calibrated_lines(
        &lines,
        &calibration.pulse,
        time_edges_us,
        start[0],
        start[1],
    )?;
    let resonances_in_span = in_span(&lines, &base)?;
    Ok(Checked {
        shape,
        sample_parameters,
        measured,
        lines,
        instrument,
        base,
        resonances_in_span,
    })
}

fn calibrated_lines(
    lines: &[Lines<'_>],
    pulse: &Pulse,
    time_edges_us: &[f64],
    t0_us: f64,
    flight_path_m: f64,
) -> Result<(), PipelineError> {
    for lines in lines {
        if let Some(((low, high), line)) = pulse.line_span_ev.zip(pulse.uncalibrated_line(
            lines.isotopes,
            time_edges_us,
            t0_us,
            flight_path_m,
        )) {
            return Err(PipelineError::InvalidParameter(format!(
                "region {} has a resonance at {line} eV, outside the {low}–{high} eV of the \
                 resonances the pulse was calibrated on",
                lines.region
            )));
        }
    }
    Ok(())
}

fn in_span(
    lines: &[Lines<'_>],
    grid: &FlightTimeGrid,
) -> Result<Vec<Vec<Vec<f64>>>, PipelineError> {
    let energies = grid.energies_ev();
    let span_ev = (energies[energies.len() - 1], energies[0]);
    lines
        .iter()
        .map(|lines| {
            lines
                .isotopes
                .iter()
                .zip(&lines.dopplers)
                .map(|((isotope, _), doppler)| {
                    let read = (
                        (span_ev.0.sqrt() - SUPPORT_X * doppler.u())
                            .max(0.0)
                            .powi(2),
                        (span_ev.1.sqrt() + SUPPORT_X * doppler.u()).powi(2),
                    );
                    if !isotope.ranges.iter().any(|range| {
                        range.is_evaluable()
                            && range.energy_low <= read.0
                            && read.1 <= range.energy_high
                    }) {
                        return Err(PipelineError::InvalidParameter(format!(
                            "the Doppler-broadened cross section of {} in region {} reads \
                             {:.6e}–{:.6e} eV, which no single one of its evaluated (SLBW, \
                             MLBW or Reich–Moore) resolved ranges holds",
                            isotope.isotope, lines.region, read.0, read.1
                        )));
                    }
                    Ok(resonance_center_energies(&[isotope])
                        .into_iter()
                        .filter(|e| (span_ev.0..=span_ev.1).contains(e))
                        .collect())
                })
                .collect()
        })
        .collect()
}

pub(crate) fn fit_counts_answer(
    measurement: &Measurement,
    calibration: &Calibration,
    starts: Option<&Starts>,
) -> Result<(CountsFit, Answer), PipelineError> {
    let Measurement {
        time_edges_us,
        charge_ratio,
        regions,
        ..
    } = measurement;
    let Checked {
        shape,
        sample_parameters,
        measured,
        lines,
        instrument,
        mut base,
        mut resonances_in_span,
    } = checked(measurement, calibration)?;
    let pulse = &calibration.pulse;
    let start: Vec<f64> = instrument.iter().map(|parameter| parameter.value).collect();
    let beam_origin_us = starts.map_or(start[0], |starts| starts.origin_us);
    let bins = time_edges_us.len() - 1;
    if let Some(starts) = starts {
        assert_eq!(starts.beams.len(), regions.len(), "one beam per region");
        if let Some(leverage) = &starts.shared_leverage {
            assert_eq!(leverage.len(), regions.len(), "one leverage per region");
        }
    }
    let mut live = Vec::with_capacity(2 * regions.len() * bins);
    for (r, region) in regions.iter().enumerate() {
        let checked = || -> Result<[Vec<f64>; 2], PipelineError> {
            validate_counts("open-beam", &region.open_counts, bins)?;
            validate_counts("sample", &region.sample_counts, bins)?;
            Ok([
                validate_live("open-beam", region.open_live.as_deref(), bins)?,
                validate_live("sample", region.sample_live.as_deref(), bins)?,
            ])
        };
        live.extend(checked().map_err(|e| in_region(r, e))?.concat());
    }

    let opens: Vec<BeamStart> = match starts {
        Some(starts) => starts.beams.clone(),
        None => regions
            .iter()
            .enumerate()
            .map(|(r, region)| {
                let open = fit_open_beam(
                    time_edges_us,
                    &region.open_counts,
                    calibration,
                    Some(&live[2 * r * bins..(2 * r + 1) * bins]),
                )
                .map_err(|e| in_region(r, e))?;
                Ok(BeamStart::from(open))
            })
            .collect::<Result<_, PipelineError>>()?,
    };
    let beams: Vec<BeamSpline> = opens.iter().map(|open| open.beam.clone()).collect();
    let sizes: Vec<usize> = beams.iter().map(|beam| beam.coefficients().len()).collect();
    let layout = Arc::new(Layout::new(roles(&sizes, &shape), regions.len()));
    let mut parameters = ParameterSet::new(
        beams
            .iter()
            .enumerate()
            .flat_map(|(r, beam)| {
                beam.coefficients().iter().enumerate().map(move |(i, &c)| {
                    FitParameter::unbounded(format!("beam {i} of region {r}"), c)
                })
            })
            .chain(sample_parameters)
            .chain(instrument)
            .collect(),
    );
    let resonances: Vec<Arc<[ResonanceData]>> = regions
        .iter()
        .map(|region| {
            region
                .material
                .iter()
                .flat_map(|material| material.isotopes.iter().map(|(data, _)| data.clone()))
                .collect()
        })
        .collect();
    let observed: Vec<f64> = regions
        .iter()
        .flat_map(|region| region.open_counts.iter().chain(&region.sample_counts))
        .copied()
        .collect();
    let calibrated_parameters: Vec<usize> = pulse
        .prior
        .iter()
        .flat_map(|prior| prior.numbers.iter().map(|n| layout.pulse + n))
        .collect();
    let pulse_prior = pulse
        .prior
        .as_ref()
        .map(|prior| Prior::correlated(&calibrated_parameters, &prior.mean, &prior.covariance))
        .transpose()?;
    let priors: Vec<Prior> = measured
        .iter()
        .map(|&(offset, mean, sd)| Prior::measured(layout.quantities + offset, mean, sd))
        .chain(pulse_prior.clone())
        .collect();
    let half_maximum = 2.0 * std::f64::consts::LN_2.sqrt();
    let narrowest_us = |grid: &FlightTimeGrid,
                        resonances_in_span: &[Vec<Vec<f64>>],
                        params: &[f64]|
     -> Result<f64, PipelineError> {
        let clock = TOF_FACTOR * grid.flight_path_m();
        let mut narrowest = f64::INFINITY;
        for (lines, energies) in lines.iter().zip(resonances_in_span) {
            let temperature_k = params[layout.regions[lines.region]
                .temperature
                .expect("a material has a temperature")];
            for ((isotope, _), energies) in lines.isotopes.iter().zip(energies) {
                let doppler = DopplerParams::new(temperature_k, isotope.awr)
                    .map_err(|e| PipelineError::InvalidParameter(e.to_string()))?;
                for &energy in energies {
                    let width_ev = half_maximum * doppler.doppler_width(energy);
                    narrowest = narrowest.min(clock / energy.sqrt() * width_ev / (2.0 * energy));
                }
            }
        }
        Ok(narrowest)
    };
    let first_grid = |grid: &FlightTimeGrid,
                      resonances_in_span: &[Vec<Vec<f64>>],
                      params: &[f64]|
     -> Result<(Arc<FlightTimeGrid>, usize), PipelineError> {
        let rule_us = 0.5 * narrowest_us(grid, resonances_in_span, params)?;
        let mut first = grid.clone();
        let mut halvings = 0;
        while first.step_us() > rule_us {
            first = first.halved()?;
            halvings += 1;
        }
        Ok((Arc::new(first), halvings))
    };
    let starting: Vec<f64> = parameters.params.iter().map(|p| p.value).collect();
    let mut first = first_grid(&base, &resonances_in_span, &starting)?;
    let mut weights: Vec<f64> = opens
        .iter()
        .flat_map(|open| [open.overdispersion.unwrap_or(1.0); 2])
        .collect();
    let mut sample_measured = vec![false; regions.len()];
    let mut noise_bins: Option<Vec<Vec<usize>>> = None;
    let mut passes = 0;
    let (fit, rule_halvings, overdispersions, settled) = loop {
        passes += 1;
        let dispersion: Vec<f64> = (0..observed.len()).map(|k| weights[k / bins]).collect();
        let fit = fit_on_halved_grids(
            &first.0,
            &mut parameters,
            &observed,
            &dispersion,
            &priors,
            |grid| {
                Ok(Recorded {
                    model: RegionsModel::new(
                        grid,
                        &beams,
                        &resonances,
                        *charge_ratio,
                        beam_origin_us,
                        &layout,
                    ),
                    live: &live,
                })
            },
            |spread, grid, params| {
                Ok(spread <= starts.map_or(BOUND, |starts| starts.bound)
                    || !grid.covers(
                        params[layout.t0],
                        params[layout.flight_path],
                        &laws(&params[layout.pulse..]),
                    )?)
            },
        )?;
        let fitted = |index: usize| fit.result.params[index];
        let noise_bins = noise_bins.get_or_insert_with(|| {
            (0..regions.len())
                .map(|r| counted(&fit, (2 * r + 1) * bins..(2 * r + 2) * bins))
                .collect()
        });
        let samples: Vec<Option<f64>> = noise_bins
            .iter()
            .enumerate()
            .map(|(r, counted)| {
                overdispersion(
                    &observed,
                    &fit,
                    counted,
                    starts
                        .and_then(|starts| starts.shared_leverage.as_ref())
                        .map_or(0.0, |extra| extra[r]),
                )
            })
            .collect();
        let next: Vec<f64> = samples
            .iter()
            .enumerate()
            .map(|(r, sample)| sample.unwrap_or(weights[2 * r]))
            .collect();
        let each_settled: Vec<bool> = next
            .iter()
            .enumerate()
            .map(|(r, next)| (next / weights[2 * r + 1] - 1.0).abs() <= SETTLED_OVERDISPERSION)
            .collect();
        let settled = each_settled.iter().all(|&s| s);
        let resolved = 2.0 * fit.step_us
            <= 0.5 * narrowest_us(&base, &resonances_in_span, &fit.result.params)?;
        let covered = fit.converged
            && fit.coarse.covers(
                fitted(layout.t0),
                fitted(layout.flight_path),
                &laws(&fit.result.params[layout.pulse..]),
            )?;
        if !fit.converged || (settled && resolved && covered) || passes == MOST_PASSES {
            let weighted: Vec<[Option<f64>; 2]> = (0..regions.len())
                .map(|r| {
                    [
                        opens[r].overdispersion,
                        (sample_measured[r] || (each_settled[r] && samples[r].is_some()))
                            .then_some(weights[2 * r + 1]),
                    ]
                })
                .collect();
            break (fit, first.1, weighted, settled && resolved && covered);
        }
        if covered {
            let resumed = (Arc::clone(&fit.coarse), first.1 + fit.halvings - 1);
            first = [
                first_grid(&base, &resonances_in_span, &fit.result.params)?,
                resumed,
            ]
            .into_iter()
            .min_by(|a, b| a.0.step_us().total_cmp(&b.0.step_us()))
            .expect("two grids");
        } else {
            base = FlightTimeGrid::new(
                time_edges_us,
                fitted(layout.t0),
                fitted(layout.flight_path),
                &calibration.pulse.at(&fit.result.params[layout.pulse..])?,
            )?;
            resonances_in_span = in_span(&lines, &base)?;
            first = first_grid(&base, &resonances_in_span, &fit.result.params)?;
        }
        for (r, sample) in samples.iter().enumerate() {
            weights[2 * r + 1] = next[r];
            sample_measured[r] = sample.is_some();
        }
    };
    let converged = fit.converged && settled;
    if converged {
        calibrated_lines(
            &lines,
            pulse,
            time_edges_us,
            fit.result.params[layout.t0],
            fit.result.params[layout.flight_path],
        )?;
    }

    if let Some((k, (&counts, &predicted))) =
        observed
            .iter()
            .zip(&fit.predicted)
            .enumerate()
            .find(|(_, (y, mu))| {
                !mu.is_finite() || **mu < 0.0 || (**y > 0.0 && **mu < NEGLIGIBLE_PREDICTION)
            })
    {
        return Err(PipelineError::UnmodelledCounts {
            region: k / (2 * bins),
            run: ["open-beam", "sample"][(k / bins) % 2],
            bin: k % bins,
            counts,
            predicted,
        });
    }

    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    let free = parameters.free_indices();
    let on_edge = layout
        .regions
        .iter()
        .filter_map(|region| region.temperature)
        .any(|t| free.contains(&t) && [t_low, t_high].contains(&fit.result.params[t]));
    let sample_quantities: Vec<usize> = (0..free.len())
        .filter(|&p| free[p] >= layout.quantities)
        .collect();
    let block = |full: &FlatMatrix| {
        let size = sample_quantities.len();
        let mut block = FlatMatrix::zeros(size, size);
        for (a, &p) in sample_quantities.iter().enumerate() {
            for (b, &q) in sample_quantities.iter().enumerate() {
                *block.get_mut(a, b) = if on_edge { f64::NAN } else { full.get(p, q) };
            }
        }
        block
    };
    let covariance = fit
        .result
        .covariance
        .as_ref()
        .filter(|_| converged)
        .map(block);
    let unbounded = fit
        .result
        .unbounded
        .as_ref()
        .filter(|_| converged)
        .map(|full| Unbounded {
            mean: sample_quantities
                .iter()
                .map(|&p| if on_edge { f64::NAN } else { full.mean[p] })
                .collect(),
            covariance: block(&full.covariance),
        });
    let params = &fit.result.params;
    let on_bound: Vec<bool> = sample_quantities
        .iter()
        .map(|&p| fit.result.on_bound[p])
        .collect();
    let position = |parameter: usize| {
        sample_quantities
            .iter()
            .position(|&p| free[p] == parameter)
            .expect("a measured or calibrated quantity is fitted")
    };
    let pulse_consistency = match (&pulse_prior, &unbounded) {
        (Some(prior), Some(unbounded)) => {
            let at: Vec<usize> = calibrated_parameters
                .iter()
                .map(|&parameter| position(parameter))
                .collect();
            let mut posterior = FlatMatrix::zeros(at.len(), at.len());
            for (a, &i) in at.iter().enumerate() {
                for (b, &j) in at.iter().enumerate() {
                    *posterior.get_mut(a, b) = unbounded.covariance.get(i, j);
                }
            }
            let estimate: Vec<f64> = at.iter().map(|&i| unbounded.mean[i]).collect();
            consistency(prior, &estimate, &posterior)?
        }
        _ => None,
    };
    let measured_pulls = unbounded.as_ref().map(|unbounded| {
        measured
            .iter()
            .map(|&(offset, mean, sd)| {
                let a = position(layout.quantities + offset);
                (unbounded.mean[a] - mean) / (sd * sd - unbounded.covariance.get(a, a)).sqrt()
            })
            .collect()
    });
    let region_fits = layout
        .regions
        .iter()
        .zip(&opens)
        .zip(overdispersions)
        .enumerate()
        .map(|(r, ((place, open), overdispersion))| RegionFit {
            densities: params[place.densities.clone()].to_vec(),
            temperature_k: place.temperature.map(|t| params[t]),
            background: [0, 1, 2].map(|i| params[place.background + i]),
            beam: open.beam.with_coefficients(&params[place.beam.clone()]),
            beam_at_limit: open.at_limit,
            predicted: [
                fit.predicted[2 * r * bins..(2 * r + 1) * bins].to_vec(),
                fit.predicted[(2 * r + 1) * bins..(2 * r + 2) * bins].to_vec(),
            ],
            overdispersion,
        })
        .collect();
    let mut fitted = vec![false; params.len()];
    let mut at_bound = vec![false; params.len()];
    for (&index, &bound) in free.iter().zip(&fit.result.on_bound) {
        fitted[index] = true;
        at_bound[index] = bound;
    }
    let answer = Answer {
        grid: Arc::clone(&fit.fine),
        dispersion: (0..observed.len()).map(|k| weights[k / bins]).collect(),
        params: params.clone(),
        fitted,
        at_bound,
        beams,
        resonances,
        charge_ratio: *charge_ratio,
        beam_origin_us,
        layout: Arc::clone(&layout),
        live,
        observed,
    };
    let fit = CountsFit {
        regions: region_fits,
        normalization: params[layout.normalization],
        t0_us: params[layout.t0],
        flight_path_m: params[layout.flight_path],
        alpha: [params[layout.pulse], params[layout.pulse + 1]],
        beta: [params[layout.pulse + 2], params[layout.pulse + 3]],
        r: params[layout.pulse + 4],
        fwhm_squared_us2: params[layout.pulse + 5],
        covariance,
        unbounded,
        on_bound,
        deviance: fit.result.deviance,
        converged,
        measured_pulls,
        pulse_consistency,
        step_us: fit.step_us,
        points: fit.points,
        halvings: rule_halvings + fit.halvings,
    };
    Ok((fit, answer))
}

pub(crate) struct Answer {
    grid: Arc<FlightTimeGrid>,
    beams: Vec<BeamSpline>,
    resonances: Vec<Arc<[ResonanceData]>>,
    charge_ratio: f64,
    beam_origin_us: f64,
    layout: Arc<Layout>,
    live: Vec<f64>,
    observed: Vec<f64>,
    dispersion: Vec<f64>,
    params: Vec<f64>,
    fitted: Vec<bool>,
    at_bound: Vec<bool>,
}

pub(crate) struct RegionTerms {
    pub(crate) quantities: Vec<Role>,
    pub(crate) gradient: Vec<f64>,
    pub(crate) information: FlatMatrix,
    pub(crate) diagonal: Vec<f64>,
    pub(crate) unbounded: (FlatMatrix, Vec<f64>),
    pub(crate) sensitivity: FlatMatrix,
    pub(crate) covariance: FlatMatrix,
    pub(crate) counted_information: FlatMatrix,
}

impl Answer {
    pub(crate) fn region_terms(
        &self,
        region: usize,
        shared: &[Role],
    ) -> Result<RegionTerms, PipelineError> {
        let roles = &self.layout.roles;
        let own: Vec<usize> = (0..roles.len())
            .filter(|&i| {
                self.fitted[i]
                    && matches!(
                        roles[i],
                        Role::Beam { region: r, .. }
                            | Role::Density { region: r, .. }
                            | Role::Temperature { region: r }
                            | Role::Background { region: r, .. }
                            if r == region
                    )
            })
            .collect();
        let shared: Vec<usize> = shared
            .iter()
            .map(|&role| {
                roles
                    .iter()
                    .position(|&r| r == role)
                    .expect("a shared role is in the layout")
            })
            .collect();
        let columns: Vec<usize> = own.iter().chain(&shared).copied().collect();
        let model = Recorded {
            model: RegionsModel::new(
                &self.grid,
                &self.beams,
                &self.resonances,
                self.charge_ratio,
                self.beam_origin_us,
                &self.layout,
            ),
            live: &self.live,
        };
        let predicted = model.evaluate(&self.params)?;
        let jacobian = model
            .analytical_jacobian(&self.params, &columns, &predicted)
            .expect("the counts model has an analytical Jacobian");
        let n = columns.len();
        let mut full = Mat::<f64>::zeros(n, n);
        let mut gradient = vec![0.0; n];
        let bins = self.observed.len() / (2 * self.layout.regions.len());
        let rows = 2 * region * bins..2 * (region + 1) * bins;
        let mut weights = vec![0.0; rows.len()];
        for ((((k, &mu), &dispersion), &observed), weight) in rows
            .clone()
            .zip(&predicted[rows.clone()])
            .zip(&self.dispersion[rows.clone()])
            .zip(&self.observed[rows.clone()])
            .zip(weights.iter_mut())
        {
            if mu > 0.0 {
                *weight = 1.0 / (mu * dispersion);
                for a in 0..n {
                    let slope = jacobian.get(k, a);
                    gradient[a] += *weight * slope * (mu - observed);
                    for b in 0..n {
                        full[(a, b)] += *weight * slope * jacobian.get(k, b);
                    }
                }
            } else {
                for (a, g) in gradient.iter_mut().enumerate() {
                    *g += jacobian.get(k, a) / dispersion;
                }
            }
        }
        let (o, s) = (own.len(), shared.len());
        let counted = rows.len();
        let profile = |set: Vec<usize>| -> Result<Profile, PipelineError> {
            let m = set.len();
            let mut block = FlatMatrix::zeros(m, m);
            for (a, &i) in set.iter().enumerate() {
                for (b, &j) in set.iter().enumerate() {
                    *block.get_mut(a, b) = full[(i, j)];
                }
            }
            let diagonal: Vec<f64> = set.iter().map(|&i| full[(i, i)]).collect();
            let inverse = information_inverse(&block, &diagonal, counted)?;
            let mut moved = FlatMatrix::zeros(m, s);
            for a in 0..m {
                for j in 0..s {
                    *moved.get_mut(a, j) = -(0..m)
                        .map(|b| inverse.spanned.get(a, b) * full[(set[b], o + j)])
                        .sum::<f64>();
                }
            }
            let profiled = |i: usize, j: usize| {
                full[(o + i, o + j)]
                    + (0..m)
                        .map(|a| full[(o + i, set[a])] * moved.get(a, j))
                        .sum::<f64>()
            };
            let mut information = FlatMatrix::zeros(s, s);
            for i in 0..s {
                for j in 0..s {
                    *information.get_mut(i, j) = 0.5 * (profiled(i, j) + profiled(j, i));
                }
            }
            let slope = (0..s)
                .map(|j| {
                    gradient[o + j]
                        + (0..m)
                            .map(|a| moved.get(a, j) * gradient[set[a]])
                            .sum::<f64>()
                })
                .collect();
            Ok(Profile {
                set,
                inverse: inverse.determined,
                resolved: inverse.resolved,
                moved,
                information,
                gradient: slope,
            })
        };
        let bounded = profile((0..o).filter(|&a| !self.at_bound[own[a]]).collect())?;
        let unbounded = profile((0..o).collect())?;
        let Profile {
            set,
            inverse,
            resolved,
            moved,
            information,
            gradient,
        } = bounded;
        let quantities: Vec<usize> = (0..set.len())
            .filter(|&a| !matches!(roles[own[set[a]]], Role::Beam { .. }))
            .collect();
        let mut sensitivity = FlatMatrix::zeros(quantities.len(), s);
        let mut covariance = FlatMatrix::zeros(quantities.len(), quantities.len());
        for (row, &a) in quantities.iter().enumerate() {
            for j in 0..s {
                *sensitivity.get_mut(row, j) = if resolved[a] {
                    moved.get(a, j)
                } else {
                    f64::NAN
                };
            }
            for (col, &b) in quantities.iter().enumerate() {
                *covariance.get_mut(row, col) = if resolved[a] && resolved[b] {
                    inverse.get(a, b)
                } else {
                    f64::NAN
                };
            }
        }
        let mut counted_information = FlatMatrix::zeros(s, s);
        let sample = rows.start + counted / 2..rows.end;
        for (k, &weight) in sample.clone().zip(&weights[counted / 2..]) {
            if predicted[k] < COUNTS_TO_MEASURE_NOISE {
                continue;
            }
            let slope: Vec<f64> = (0..s)
                .map(|j| {
                    jacobian.get(k, o + j)
                        + (0..set.len())
                            .map(|a| jacobian.get(k, set[a]) * moved.get(a, j))
                            .sum::<f64>()
                })
                .collect();
            for i in 0..s {
                for j in 0..s {
                    *counted_information.get_mut(i, j) += weight * slope[i] * slope[j];
                }
            }
        }
        Ok(RegionTerms {
            quantities: quantities.iter().map(|&a| roles[own[set[a]]]).collect(),
            gradient,
            information,
            diagonal: (0..s).map(|j| full[(o + j, o + j)]).collect(),
            unbounded: (unbounded.information, unbounded.gradient),
            sensitivity,
            covariance,
            counted_information,
        })
    }
}

struct Profile {
    set: Vec<usize>,
    inverse: FlatMatrix,
    resolved: Vec<bool>,
    moved: FlatMatrix,
    information: FlatMatrix,
    gradient: Vec<f64>,
}

pub(crate) const SHARED: [Role; 9] = [
    Role::Normalization,
    Role::T0,
    Role::FlightPath,
    Role::Pulse(0),
    Role::Pulse(1),
    Role::Pulse(2),
    Role::Pulse(3),
    Role::Pulse(4),
    Role::Pulse(5),
];

pub(crate) fn shared_parameters(
    measurement: &Measurement,
    calibration: &Calibration,
) -> Result<Vec<(Value, FitParameter)>, PipelineError> {
    validate_pulse_prior(calibration)?;
    let (value, name, range, allowed) =
        quantity(measurement, Role::Normalization).expect("the normalization is a quantity");
    let values = quantities(measurement, calibration)
        .filter(|(role, _)| SHARED.contains(role))
        .map(|(_, value)| *value);
    Ok(values
        .zip(
            std::iter::once(value.parameter(name, range, &allowed)?)
                .chain(calibration.instrument()?),
        )
        .collect())
}

fn validate_pulse_prior(calibration: &Calibration) -> Result<(), PipelineError> {
    let pulse = &calibration.pulse;
    let numbers = [
        pulse.alpha[0],
        pulse.alpha[1],
        pulse.beta[0],
        pulse.beta[1],
        pulse.r,
        pulse.fwhm_squared_us2,
    ];
    match pulse
        .prior
        .iter()
        .flat_map(|prior| &prior.numbers)
        .find(|&&n| !matches!(numbers[n], Value::Fitted(_) | Value::Within { .. }))
    {
        Some(&number) => Err(PipelineError::InvalidParameter(format!(
            "the pulse's calibration covers {}, which must be fitted, without a measurement; \
             got {:?}",
            PULSE_NUMBERS[number], numbers[number]
        ))),
        None => Ok(()),
    }
}

pub(crate) fn quantities<'a>(
    measurement: &'a Measurement,
    calibration: &'a Calibration,
) -> impl Iterator<Item = (Role, &'a Value)> {
    let shape = shape(measurement);
    let pulse = &calibration.pulse;
    let numbers = [
        &pulse.alpha[0],
        &pulse.alpha[1],
        &pulse.beta[0],
        &pulse.beta[1],
        &pulse.r,
        &pulse.fwhm_squared_us2,
    ];
    roles(&vec![0; shape.len()], &shape)
        .into_iter()
        .filter_map(move |role| {
            let value = match role {
                Role::T0 => &calibration.t0_us,
                Role::FlightPath => &calibration.flight_path_m,
                Role::Pulse(number) => numbers[number],
                _ => quantity(measurement, role)?.0,
            };
            Some((role, value))
        })
}

fn shape(measurement: &Measurement) -> Vec<Option<usize>> {
    measurement
        .regions
        .iter()
        .map(|region| region.material.as_ref().map(|m| m.isotopes.len()))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Role {
    Beam { region: usize, coefficient: usize },
    Density { region: usize, isotope: usize },
    Temperature { region: usize },
    Normalization,
    Background { region: usize, term: usize },
    T0,
    FlightPath,
    Pulse(usize),
}

fn roles(beams: &[usize], isotopes: &[Option<usize>]) -> Vec<Role> {
    let beam = beams.iter().enumerate().flat_map(|(region, &n)| {
        (0..n).map(move |coefficient| Role::Beam {
            region,
            coefficient,
        })
    });
    let materials = isotopes
        .iter()
        .enumerate()
        .filter_map(|(region, n)| Some((region, (*n)?)))
        .flat_map(|(region, n)| {
            (0..n)
                .map(move |isotope| Role::Density { region, isotope })
                .chain([Role::Temperature { region }])
        });
    let backgrounds = (0..isotopes.len())
        .flat_map(|region| (0..3).map(move |term| Role::Background { region, term }));
    beam.chain(materials)
        .chain([Role::Normalization])
        .chain(backgrounds)
        .chain([Role::T0, Role::FlightPath])
        .chain((0..6).map(Role::Pulse))
        .collect()
}

fn material(measurement: &Measurement, region: usize) -> &Material {
    measurement.regions[region]
        .material
        .as_ref()
        .expect("a density or temperature names a region with a material")
}

fn quantity(
    measurement: &Measurement,
    role: Role,
) -> Option<(&Value, String, RangeInclusive<f64>, String)> {
    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    Some(match role {
        Role::Density { region, isotope } => {
            let (data, density) = &material(measurement, region).isotopes[isotope];
            (
                density,
                format!("density of {} in region {region}", data.isotope),
                0.0..=f64::INFINITY,
                "0 or more".into(),
            )
        }
        Role::Temperature { region } => (
            &material(measurement, region).temperature_k,
            format!("temperature of region {region}"),
            t_low..=t_high,
            format!("within {t_low}–{t_high} K"),
        ),
        Role::Normalization => (
            &measurement.normalization,
            "normalization".into(),
            f64::MIN_POSITIVE..=f64::INFINITY,
            "positive".into(),
        ),
        Role::Background { region, term } => (
            &measurement.regions[region].background[term],
            format!("b{term} of region {region}"),
            f64::NEG_INFINITY..=f64::INFINITY,
            "of any sign".into(),
        ),
        Role::Beam { .. } | Role::T0 | Role::FlightPath | Role::Pulse(_) => return None,
    })
}

fn in_region(region: usize, error: PipelineError) -> PipelineError {
    labelled(&format!("region {region}"), error)
}

pub(crate) fn labelled(label: &str, error: PipelineError) -> PipelineError {
    match error {
        PipelineError::InvalidParameter(message) => {
            PipelineError::InvalidParameter(format!("{label}: {message}"))
        }
        PipelineError::ShapeMismatch(message) => {
            PipelineError::ShapeMismatch(format!("{label}: {message}"))
        }
        other => other,
    }
}

fn finite(isotope: &ResonanceData) -> bool {
    isotope.awr.is_finite()
        && isotope.ranges.iter().all(|range| {
            let radii = range
                .ap_table
                .iter()
                .flat_map(|table| table.points.iter().flat_map(|&(e, r)| [e, r]));
            let external = range.r_external.iter().flat_map(|r| {
                [
                    r.j, r.e_low, r.e_up, r.r_con, r.r_lin, r.s_con, r.s_lin, r.r_quad,
                ]
            });
            let groups = range.l_groups.iter().flat_map(|group| {
                [group.awr, group.apl, group.qx].into_iter().chain(
                    group
                        .resonances
                        .iter()
                        .flat_map(|r| [r.energy, r.j, r.gn, r.gg, r.gfa, r.gfb]),
                )
            });
            [
                range.energy_low,
                range.energy_high,
                range.target_spin,
                range.scattering_radius,
            ]
            .into_iter()
            .chain(radii)
            .chain(external)
            .chain(groups)
            .all(f64::is_finite)
        })
}

struct Lines<'a> {
    region: usize,
    isotopes: &'a [(ResonanceData, Value)],
    dopplers: Vec<DopplerParams>,
}

struct Layout {
    roles: Vec<Role>,
    regions: Vec<Place>,
    quantities: usize,
    normalization: usize,
    t0: usize,
    flight_path: usize,
    pulse: usize,
}

struct Place {
    beam: Range<usize>,
    densities: Range<usize>,
    temperature: Option<usize>,
    background: usize,
}

impl Layout {
    fn new(roles: Vec<Role>, regions: usize) -> Self {
        let at = |wanted: Role| {
            roles
                .iter()
                .position(|&role| role == wanted)
                .expect("every region has a background and the instrument its numbers")
        };
        let span = |of: &dyn Fn(Role) -> bool| {
            let first = roles.iter().position(|&role| of(role));
            let last = roles.iter().rposition(|&role| of(role));
            first
                .zip(last)
                .map_or(0..0, |(first, last)| first..last + 1)
        };
        let places = (0..regions)
            .map(|r| Place {
                beam: span(&|role| matches!(role, Role::Beam { region, .. } if region == r)),
                densities: span(
                    &|role| matches!(role, Role::Density { region, .. } if region == r),
                ),
                temperature: roles
                    .iter()
                    .position(|&role| role == Role::Temperature { region: r }),
                background: at(Role::Background { region: r, term: 0 }),
            })
            .collect();
        Self {
            quantities: roles
                .iter()
                .position(|role| !matches!(role, Role::Beam { .. }))
                .expect("the instrument's numbers follow the beams"),
            normalization: at(Role::Normalization),
            t0: at(Role::T0),
            flight_path: at(Role::FlightPath),
            pulse: at(Role::Pulse(0)),
            regions: places,
            roles,
        }
    }
}

struct RegionsModel {
    grid: Arc<FlightTimeGrid>,
    splines: Vec<BeamSpline>,
    beam_origin_us: f64,
    absorbers: Vec<Absorber>,
    energies: Vec<f64>,
    shapes: [Vec<f64>; 3],
    charge_ratio: f64,
    layout: Arc<Layout>,
    scale: RefCell<Option<Scale>>,
}

struct Absorber {
    isotopes: Arc<[ResonanceData]>,
    cross_sections: RefCell<Option<CrossSections>>,
}

struct Beams {
    open: Vec<f64>,
    normalized: Vec<f64>,
    transmitted: Vec<f64>,
    sample: Vec<f64>,
}

struct CrossSections {
    temperature_k: f64,
    values: Vec<Vec<f64>>,
    slopes: Vec<Vec<f64>>,
}

struct Scale {
    t0_us: f64,
    flight_path_m: f64,
    pulse: IkedaCarpenterParams,
    rows: Rows,
    basis: Vec<Vec<[(usize, f64); 5]>>,
    basis_slope: Vec<Vec<[(usize, f64); 5]>>,
}

impl Scale {
    fn new(
        (t0_us, flight_path_m): (f64, f64),
        pulse: IkedaCarpenterParams,
        rows: Rows,
        grid: &FlightTimeGrid,
        splines: &[BeamSpline],
        beam_origin_us: f64,
    ) -> Self {
        let (shift, stretch) = (t0_us - beam_origin_us, flight_path_m / grid.flight_path_m());
        let abscissae: Vec<f64> = grid
            .flight_times_us()
            .iter()
            .map(|u| shift + stretch * u)
            .collect();
        Self {
            t0_us,
            flight_path_m,
            pulse,
            rows,
            basis: splines
                .iter()
                .map(|spline| abscissae.iter().map(|&a| spline.basis(a)).collect())
                .collect(),
            basis_slope: splines
                .iter()
                .map(|spline| abscissae.iter().map(|&a| spline.basis_slope(a)).collect())
                .collect(),
        }
    }
}

fn predicted(rows: &Rows, values: &[f64]) -> Result<Vec<f64>, FittingError> {
    rows.predict(values)
        .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
}

impl RegionsModel {
    fn new(
        grid: &Arc<FlightTimeGrid>,
        beams: &[BeamSpline],
        isotopes: &[Arc<[ResonanceData]>],
        charge_ratio: f64,
        beam_origin_us: f64,
        layout: &Arc<Layout>,
    ) -> Self {
        let mut energies = grid.energies_ev();
        let shapes = [
            vec![1.0; energies.len()],
            energies.iter().map(|e| 1.0 / e.sqrt()).collect(),
            energies.iter().map(|e| e.sqrt()).collect(),
        ];
        energies.reverse();
        Self {
            grid: Arc::clone(grid),
            splines: beams.to_vec(),
            beam_origin_us,
            absorbers: isotopes
                .iter()
                .map(|isotopes| Absorber {
                    isotopes: Arc::clone(isotopes),
                    cross_sections: RefCell::new(None),
                })
                .collect(),
            energies,
            shapes,
            charge_ratio,
            layout: Arc::clone(layout),
            scale: RefCell::new(Some(Scale::new(
                (grid.t0_us(), grid.flight_path_m()),
                grid.pulse().params().clone(),
                grid.rows().clone(),
                grid,
                beams,
                beam_origin_us,
            ))),
        }
    }

    fn at(
        &self,
        region: usize,
        temperature_k: f64,
    ) -> Result<std::cell::Ref<'_, CrossSections>, FittingError> {
        let absorber = &self.absorbers[region];
        let current = absorber
            .cross_sections
            .borrow()
            .as_ref()
            .is_some_and(|c| c.temperature_k.to_bits() == temperature_k.to_bits());
        if !current {
            let energies = &self.energies;
            let (values, slopes) = absorber
                .isotopes
                .par_iter()
                .map(|isotope| {
                    let (mut sigma, mut slope) =
                        broaden_with_derivative(energies, isotope, temperature_k)
                            .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?;
                    sigma.reverse();
                    slope.reverse();
                    Ok((sigma, slope))
                })
                .collect::<Result<Vec<_>, FittingError>>()?
                .into_iter()
                .unzip();
            *absorber.cross_sections.borrow_mut() = Some(CrossSections {
                temperature_k,
                values,
                slopes,
            });
        }
        Ok(std::cell::Ref::map(absorber.cross_sections.borrow(), |c| {
            c.as_ref().expect("computed above")
        }))
    }

    fn scale(&self, params: &[f64]) -> Result<std::cell::Ref<'_, Scale>, FittingError> {
        let (t0_us, flight_path_m) = (params[self.layout.t0], params[self.layout.flight_path]);
        let pulse = laws(&params[self.layout.pulse..]);
        let key = |t0: f64, l: f64| (t0.to_bits(), l.to_bits());
        let current = self.scale.borrow().as_ref().is_some_and(|s| {
            key(s.t0_us, s.flight_path_m) == key(t0_us, flight_path_m) && s.pulse == pulse
        });
        if !current {
            let rows = self
                .grid
                .rows_at(t0_us, flight_path_m, &pulse)
                .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?;
            *self.scale.borrow_mut() = Some(Scale::new(
                (t0_us, flight_path_m),
                pulse,
                rows,
                &self.grid,
                &self.splines,
                self.beam_origin_us,
            ));
        }
        Ok(std::cell::Ref::map(self.scale.borrow(), |s| {
            s.as_ref().expect("computed above")
        }))
    }

    fn beams(&self, params: &[f64], scale: &Scale, region: usize) -> Result<Beams, FittingError> {
        let place = &self.layout.regions[region];
        let densities = &params[place.densities.clone()];
        let sigma = place
            .temperature
            .map(|t| self.at(region, params[t]))
            .transpose()?;
        let open: Vec<f64> = combined(&scale.basis[region], &params[place.beam.clone()])
            .into_iter()
            .map(f64::exp)
            .collect();
        let normalized: Vec<f64> = open
            .iter()
            .map(|phi| self.charge_ratio * params[self.layout.normalization] * phi)
            .collect();
        let terms = &params[place.background..place.background + 3];
        let (transmitted, sample) = normalized
            .iter()
            .enumerate()
            .map(|(j, beam)| {
                let depth: f64 = sigma
                    .iter()
                    .flat_map(|sigma| densities.iter().zip(&sigma.values))
                    .map(|(n, sigma)| n * sigma[j])
                    .sum();
                let background: f64 = terms.iter().zip(&self.shapes).map(|(b, g)| b * g[j]).sum();
                (beam * (-depth).exp(), beam * ((-depth).exp() + background))
            })
            .unzip();
        Ok(Beams {
            open,
            normalized,
            transmitted,
            sample,
        })
    }
}

impl FitModel for RegionsModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let scale = self.scale(params)?;
        let mut counts = Vec::new();
        for region in 0..self.absorbers.len() {
            let Beams { open, sample, .. } = self.beams(params, &scale, region)?;
            counts.extend(predicted(&scale.rows, &open)?);
            counts.extend(predicted(&scale.rows, &sample)?);
        }
        Ok(counts)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let layout = &*self.layout;
        let (t0_us, flight_path_m) = (params[layout.t0], params[layout.flight_path]);
        let scale = self.scale(params).ok()?;
        let regions = self.absorbers.len();
        let beams: Vec<Beams> = (0..regions)
            .map(|region| self.beams(params, &scale, region))
            .collect::<Result<_, _>>()
            .ok()?;
        let bins = y_current.len() / (2 * regions);
        let (open_rows, sample_rows) = (|r: usize| 2 * r * bins, |r: usize| (2 * r + 1) * bins);
        let counts = |values: &[f64]| predicted(&scale.rows, values).ok();
        let times = |beam: &[f64], slope: &[f64]| -> Vec<f64> {
            beam.iter().zip(slope).map(|(b, s)| b * s).collect()
        };
        let mut arrival: Option<Rows> = None;
        let mut pulse_slopes: Option<[Rows; 4]> = None;
        let ones = vec![1.0; self.energies.len()];
        let root_energy = &self.shapes[2];
        let pulse_columns: [(usize, &[f64]); 6] = [
            (0, root_energy),
            (0, &ones),
            (1, root_energy),
            (1, &ones),
            (2, &ones),
            (3, &ones),
        ];
        let mut jacobian = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        for (col, &index) in free_param_indices.iter().enumerate() {
            let blocks: Vec<(usize, Vec<f64>)> = match layout.roles[index] {
                Role::Beam {
                    region,
                    coefficient,
                } => {
                    let slope = weights_of(&scale.basis[region], coefficient);
                    let Beams { open, sample, .. } = &beams[region];
                    vec![
                        (open_rows(region), counts(&times(open, &slope))?),
                        (sample_rows(region), counts(&times(sample, &slope))?),
                    ]
                }
                Role::Density { region, isotope } => {
                    let sigma = self
                        .at(region, params[layout.regions[region].temperature?])
                        .ok()?;
                    let slope: Vec<f64> = sigma.values[isotope].iter().map(|s| -s).collect();
                    vec![(
                        sample_rows(region),
                        counts(&times(&beams[region].transmitted, &slope))?,
                    )]
                }
                Role::Temperature { region } => {
                    let place = &layout.regions[region];
                    let sigma = self.at(region, params[index]).ok()?;
                    let densities = &params[place.densities.clone()];
                    let broadening: Vec<f64> = (0..self.energies.len())
                        .map(|j| {
                            -densities
                                .iter()
                                .zip(&sigma.slopes)
                                .map(|(n, slope)| n * slope[j])
                                .sum::<f64>()
                        })
                        .collect();
                    vec![(
                        sample_rows(region),
                        counts(&times(&beams[region].transmitted, &broadening))?,
                    )]
                }
                Role::Normalization => {
                    let slope = vec![1.0 / params[index]; self.energies.len()];
                    (0..regions)
                        .map(|r| Some((sample_rows(r), counts(&times(&beams[r].sample, &slope))?)))
                        .collect::<Option<_>>()?
                }
                Role::Background { region, term } => vec![(
                    sample_rows(region),
                    counts(&times(&beams[region].normalized, &self.shapes[term]))?,
                )],
                Role::Pulse(number) => {
                    if pulse_slopes.is_none() {
                        pulse_slopes = Some(
                            self.grid
                                .pulse_slopes_at(t0_us, flight_path_m, &scale.pulse)
                                .ok()?,
                        );
                    }
                    let (law, weight) = pulse_columns[number];
                    let rows = &pulse_slopes.as_ref()?[law];
                    (0..regions)
                        .flat_map(|r| {
                            [
                                (open_rows(r), &beams[r].open),
                                (sample_rows(r), &beams[r].sample),
                            ]
                        })
                        .map(|(row, beam)| Some((row, predicted(rows, &times(beam, weight)).ok()?)))
                        .collect::<Option<_>>()?
                }
                Role::T0 | Role::FlightPath => {
                    if arrival.is_none() {
                        arrival = Some(
                            self.grid
                                .arrival_slopes_at(t0_us, flight_path_m, &scale.pulse)
                                .ok()?,
                        );
                    }
                    let arrival = arrival.as_ref()?;
                    let (reach, per_metre): (Vec<f64>, Option<f64>) = match layout.roles[index] {
                        Role::FlightPath => (
                            self.grid
                                .flight_times_us()
                                .iter()
                                .map(|u| u / self.grid.flight_path_m())
                                .collect(),
                            Some(flight_path_m),
                        ),
                        _ => (ones.clone(), None),
                    };
                    let mut blocks = Vec::with_capacity(2 * regions);
                    for r in 0..regions {
                        let log_beam_slope = combined(
                            &scale.basis_slope[r],
                            &params[layout.regions[r].beam.clone()],
                        );
                        for (row, beam) in [
                            (open_rows(r), &beams[r].open),
                            (sample_rows(r), &beams[r].sample),
                        ] {
                            let reached = times(beam, &reach);
                            let beam_part = counts(&times(&reached, &log_beam_slope))?;
                            let pulse_part = predicted(arrival, &reached).ok()?;
                            let mut column: Vec<f64> = beam_part
                                .iter()
                                .zip(&pulse_part)
                                .map(|(b, p)| b + p)
                                .collect();
                            if let Some(flight_path_m) = per_metre {
                                for (c, x) in column.iter_mut().zip(counts(beam)?) {
                                    *c += x / flight_path_m;
                                }
                            }
                            blocks.push((row, column));
                        }
                    }
                    blocks
                }
            };
            for (row, column) in blocks {
                for (offset, value) in column.into_iter().enumerate() {
                    *jacobian.get_mut(row + offset, col) = value;
                }
            }
        }
        Some(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use nereids_endf::resonance::test_support::synthetic_isotope;

    use faer::Side;
    use faer::linalg::solvers::DenseSolveCore;

    use super::*;
    use crate::open_beam::Pulse;
    use crate::open_beam::tests::{ALPHA, BETA, EDGES_US, FLIGHT_PATH_M, R, T0_US, grid};

    #[test]
    fn the_jacobian_is_the_slope_of_every_region_s_counts() {
        for channel_fwhm_us in [None, Some(0.35)] {
            jacobian_against_central_differences(&grid(channel_fwhm_us));
        }
    }

    fn jacobian_against_central_differences(grid: &Arc<FlightTimeGrid>) {
        let (_, u_hi) = grid.range_us();
        let coarse = BeamSpline::constant(347.0, u_hi, 1.0e4);
        let beams = [coarse.refined().refined(), coarse.clone(), coarse.refined()];
        let isotopes: [Arc<[ResonanceData]>; 3] = [
            Arc::new([
                synthetic_isotope(72, 180, 20.0, 0.01, 0.06),
                synthetic_isotope(74, 182, 20.3, 0.01, 0.06),
            ]),
            Arc::new([]),
            Arc::new([synthetic_isotope(74, 184, 20.6, 0.01, 0.06)]),
        ];
        let sizes: Vec<usize> = beams.iter().map(|b| b.coefficients().len()).collect();
        let layout = Arc::new(Layout::new(roles(&sizes, &[Some(2), None, Some(1)]), 3));
        let params: Vec<f64> = layout
            .roles
            .iter()
            .enumerate()
            .map(|(i, role)| match *role {
                Role::Beam { .. } => 9.0 + 0.3 * (i as f64).sin(),
                Role::Density { region, isotope } => {
                    [[3.0e-4, 5.0e-4], [0.0; 2], [4.0e-4, 0.0]][region][isotope]
                }
                Role::Temperature { region } => [300.0, 0.0, 250.0][region],
                Role::Normalization => 0.93,
                Role::Background { region, term } => {
                    [[0.05, 0.5, -0.01], [0.02, 0.1, 0.003], [0.04, -0.2, 0.005]][region][term]
                }
                Role::T0 => T0_US + 0.05,
                Role::FlightPath => FLIGHT_PATH_M + 0.003,
                Role::Pulse(n) => [0.37, 0.06, 0.01, 0.24, 0.16, 0.3][n],
            })
            .collect();
        let model = RegionsModel::new(grid, &beams, &isotopes, 1.2, T0_US, &layout);
        let live: Vec<f64> = (0..model.evaluate(&params).expect("counts").len())
            .map(|k| 0.9 + 0.1 * (0.3 * k as f64).sin())
            .collect();
        let model = Recorded { model, live: &live };
        let mut elsewhere = params.clone();
        for t in layout
            .regions
            .iter()
            .filter_map(|region| region.temperature)
        {
            elsewhere[t] -= 50.0;
        }
        elsewhere[layout.t0] = T0_US - 0.05;
        elsewhere[layout.flight_path] = FLIGHT_PATH_M - 0.003;
        elsewhere[layout.pulse + 4] = 0.2;
        model.evaluate(&elsewhere).expect("counts");
        let counts = model.evaluate(&params).expect("counts");
        let indices: Vec<usize> = (0..params.len()).collect();
        let jacobian = model
            .analytical_jacobian(&params, &indices, &counts)
            .expect("jacobian");
        let middle_us = grid.flight_times_us()[grid.flight_times_us().len() / 2];
        for index in indices {
            let h = if index == layout.flight_path {
                params[index] * 1e-4 * params[layout.t0] / middle_us
            } else {
                1e-4 * params[index].abs()
            };
            let shifted = |d: f64| {
                let mut p = params.clone();
                p[index] += d;
                model.evaluate(&p).expect("counts")
            };
            let (up, down) = (shifted(h), shifted(-h));
            let slopes: Vec<f64> = up
                .iter()
                .zip(&down)
                .map(|(u, d)| (u - d) / (2.0 * h))
                .collect();
            let column = slopes.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
            for (row, &slope) in slopes.iter().enumerate() {
                let analytic = jacobian.get(row, index);
                assert!(
                    (analytic - slope).abs() <= 1e-6 * column,
                    "{index} {row}: {analytic} vs {slope}"
                );
            }
        }
    }

    #[test]
    fn each_region_s_terms_rebuild_the_joint_covariance() {
        let grid = Arc::new(grid(None).halved().expect("grid").halved().expect("grid"));
        let (_, u_hi) = grid.range_us();
        let beam = BeamSpline::constant(347.0, u_hi, 1.0e4).refined();
        let beams = [beam.clone(), beam.clone(), beam];
        let materials = [
            Some(synthetic_isotope(72, 180, 20.0, 0.01, 0.06)),
            Some(synthetic_isotope(74, 182, 24.0, 0.01, 0.06)),
            None,
        ];
        let isotopes: Vec<Arc<[ResonanceData]>> = materials
            .iter()
            .map(|m| m.iter().cloned().collect())
            .collect();
        let shape: Vec<Option<usize>> = materials.iter().map(|m| m.as_ref().map(|_| 1)).collect();
        let sizes: Vec<usize> = beams.iter().map(|b| b.coefficients().len()).collect();
        let layout = Arc::new(Layout::new(roles(&sizes, &shape), 3));
        let truth: Vec<f64> = layout
            .roles
            .iter()
            .map(|role| match *role {
                Role::Beam {
                    region,
                    coefficient,
                } => beams[region].coefficients()[coefficient],
                Role::Density { region, .. } => [5.0e-4, 4.0e-4, 0.0][region],
                Role::Temperature { .. } => 300.0,
                Role::Normalization => 0.93,
                Role::Background { region, term } => {
                    [[0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0; 3]][region][term]
                }
                Role::T0 => T0_US,
                Role::FlightPath => FLIGHT_PATH_M,
                Role::Pulse(n) => [0.35, 0.05, 0.0, 0.25, 0.15, 0.0][n],
            })
            .collect();
        let model = RegionsModel::new(&grid, &beams, &isotopes, 1.2, T0_US, &layout);
        let bins = model.evaluate(&truth).expect("counts").len() / 6;
        let live: Vec<f64> = (0..6 * bins)
            .map(|k| 0.9 + 0.1 * (0.3 * k as f64).sin())
            .collect();
        let counts: Vec<f64> = Recorded { model, live: &live }
            .evaluate(&truth)
            .expect("counts")
            .iter()
            .enumerate()
            .map(|(k, mu)| (mu + 3.0 * mu.sqrt() * (1.7 * k as f64).sin()).round())
            .collect();
        let measurement = Measurement {
            time_edges_us: EDGES_US.map(f64::from).collect(),
            charge_ratio: 1.2,
            normalization: Value::Fitted(1.0),
            regions: materials
                .into_iter()
                .enumerate()
                .map(|(r, material)| Region {
                    open_counts: counts[2 * r * bins..(2 * r + 1) * bins].to_vec(),
                    sample_counts: counts[(2 * r + 1) * bins..(2 * r + 2) * bins].to_vec(),
                    open_live: Some(live[2 * r * bins..(2 * r + 1) * bins].to_vec()),
                    sample_live: Some(live[(2 * r + 1) * bins..(2 * r + 2) * bins].to_vec()),
                    background: match material {
                        Some(_) => [Value::Fitted(0.02), Value::Known(0.0), Value::Known(0.0)],
                        None => [Value::Known(0.0); 3],
                    },
                    material: material.map(|data| Material {
                        isotopes: vec![(data, Value::Fitted(3.0e-4))],
                        temperature_k: Value::Fitted(350.0),
                    }),
                })
                .collect(),
        };
        let calibration = Calibration {
            t0_us: Value::Fitted(T0_US),
            flight_path_m: Value::Fitted(FLIGHT_PATH_M),
            pulse: Pulse {
                alpha: [Value::Known(0.35), Value::Known(0.05)],
                beta: [Value::Known(0.0), Value::Known(0.25)],
                r: Value::Known(0.15),
                fwhm_squared_us2: Value::Known(0.0),
                energy_span_ev: (1.0, 200.0),
                n_tau: 256,
                line_span_ev: None,
                prior: None,
            },
        };
        let (fit, answer) = fit_counts_answer(&measurement, &calibration, None).expect("fit");
        assert!(fit.converged);
        let covariance = fit.covariance.as_ref().expect("covariance");
        let fitted: Vec<Role> = quantities(&measurement, &calibration)
            .filter(|(_, value)| !matches!(value, Value::Known(_)))
            .map(|(role, _)| role)
            .collect();
        let at = |role: Role| fitted.iter().position(|&r| r == role).expect("fitted");
        let close = |actual: f64, a: Role, b: Role| {
            let (i, j) = (at(a), at(b));
            let expected = covariance.get(i, j);
            let scale = (covariance.get(i, i) * covariance.get(j, j)).sqrt();
            assert!(
                (actual - expected).abs() <= 1e-8 * scale,
                "{a:?} {b:?}: {actual} vs {expected}"
            );
        };
        let shared = [Role::Normalization, Role::T0, Role::FlightPath];
        let terms: Vec<RegionTerms> = (0..3)
            .map(|r| answer.region_terms(r, &shared).expect("terms"))
            .collect();
        let k = shared.len();
        let joint = Mat::from_fn(k, k, |i, j| {
            terms.iter().map(|t| t.information.get(i, j)).sum::<f64>()
        })
        .llt(Side::Lower)
        .expect("positive definite")
        .inverse();
        for i in 0..k {
            for j in 0..k {
                close(joint[(i, j)], shared[i], shared[j]);
            }
        }
        let carried = |t: &RegionTerms, a: usize, j: usize| {
            (0..k)
                .map(|i| t.sensitivity.get(a, i) * joint[(i, j)])
                .sum::<f64>()
        };
        for (p, tp) in terms.iter().enumerate() {
            for (a, &ra) in tp.quantities.iter().enumerate() {
                for (j, &rs) in shared.iter().enumerate() {
                    close(carried(tp, a, j), ra, rs);
                }
                for (q, tq) in terms.iter().enumerate() {
                    for (b, &rb) in tq.quantities.iter().enumerate() {
                        let common: f64 = (0..k)
                            .map(|j| carried(tp, a, j) * tq.sensitivity.get(b, j))
                            .sum();
                        let own = if p == q { tp.covariance.get(a, b) } else { 0.0 };
                        close(own + common, ra, rb);
                    }
                }
            }
        }
        let leverage: f64 = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| terms[0].counted_information.get(i, j) * joint[(i, j)])
            .sum();
        let value = |role: Role| {
            answer.params[answer
                .layout
                .roles
                .iter()
                .position(|&r| r == role)
                .expect("a shared role")]
        };
        let alone = Measurement {
            normalization: Value::Known(value(Role::Normalization)),
            regions: vec![measurement.regions[0].clone()],
            ..measurement.clone()
        };
        let held = Calibration {
            t0_us: Value::Known(value(Role::T0)),
            flight_path_m: Value::Known(value(Role::FlightPath)),
            ..calibration.clone()
        };
        let joined = &fit.regions[0];
        let starts = Starts {
            beams: vec![BeamStart {
                beam: joined.beam.clone(),
                overdispersion: joined.overdispersion[0],
                at_limit: joined.beam_at_limit,
            }],
            origin_us: answer.beam_origin_us,
            bound: BOUND,
            shared_leverage: Some(vec![leverage]),
        };
        let (region, _) = fit_counts_answer(&alone, &held, Some(&starts)).expect("fit");
        let [Some(phi), Some(expected)] =
            [&region.regions[0], joined].map(|fit| fit.overdispersion[1])
        else {
            panic!("a sample overdispersion");
        };
        assert!((phi / expected - 1.0).abs() <= 1e-4, "{phi} vs {expected}");
        let gradient: Vec<f64> = (0..k)
            .map(|i| terms.iter().map(|t| t.gradient[i]).sum())
            .collect();
        let decrement: f64 = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * gradient[i] * joint[(i, j)] * gradient[j])
            .sum();
        assert!(decrement <= 1e-6, "{decrement}");
        let mut nudged = answer;
        let roles = nudged.layout.roles.clone();
        for (i, &role) in roles.iter().enumerate() {
            if matches!(
                role,
                Role::Density { region: 0, .. } | Role::Temperature { region: 0 }
            ) {
                nudged.params[i] += covariance.get(at(role), at(role)).sqrt();
            }
        }
        let nudged_terms: Vec<RegionTerms> = (0..3)
            .map(|r| nudged.region_terms(r, &shared).expect("terms"))
            .collect();
        let gradient: Vec<f64> = (0..k)
            .map(|i| nudged_terms.iter().map(|t| t.gradient[i]).sum())
            .collect();
        let decrement: f64 = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * gradient[i] * joint[(i, j)] * gradient[j])
            .sum();
        assert!(decrement <= 1e-2, "{decrement}");
    }

    #[test]
    fn a_map_under_a_pulse_calibration_is_the_joint_fit() {
        let grid = Arc::new(grid(None).halved().expect("grid").halved().expect("grid"));
        let (_, u_hi) = grid.range_us();
        let beam = BeamSpline::constant(347.0, u_hi, 1.0e4).refined();
        let beams = [beam.clone(), beam];
        let hafnium = synthetic_isotope(72, 180, 20.0, 0.01, 0.06);
        let isotopes: Vec<Arc<[ResonanceData]>> = vec![Arc::new([hafnium.clone()]), Arc::new([])];
        let sizes: Vec<usize> = beams.iter().map(|b| b.coefficients().len()).collect();
        let layout = Arc::new(Layout::new(roles(&sizes, &[Some(1), None]), 2));
        let truth: Vec<f64> = layout
            .roles
            .iter()
            .map(|role| match *role {
                Role::Beam {
                    region,
                    coefficient,
                } => beams[region].coefficients()[coefficient],
                Role::Density { .. } => 5.0e-4,
                Role::Temperature { .. } => 300.0,
                Role::Normalization => 0.93,
                Role::Background { region, term } => [[0.05, 0.0, 0.0], [0.0; 3]][region][term],
                Role::T0 => T0_US,
                Role::FlightPath => FLIGHT_PATH_M,
                Role::Pulse(n) => [0.35, 0.05, 0.0, 0.25, 0.15, 0.0][n],
            })
            .collect();
        let counts: Vec<f64> = RegionsModel::new(&grid, &beams, &isotopes, 1.2, T0_US, &layout)
            .evaluate(&truth)
            .expect("counts")
            .iter()
            .map(|c| c.round())
            .collect();
        let bins = counts.len() / 4;
        let run =
            |r: usize, run: usize| counts[(2 * r + run) * bins..(2 * r + run + 1) * bins].to_vec();
        let cube = |which: usize| {
            ndarray::Array3::from_shape_fn((bins, 2, 1), |(k, y, _)| run(y, which)[k])
        };
        let (open, sample) = (cube(0), cube(1));
        let sample_pixel = ndarray::Array2::from_shape_fn((2, 1), |(y, _)| y == 0);
        let empty_pixel = ndarray::Array2::from_shape_fn((2, 1), |(y, _)| y == 1);
        let none = ndarray::Array2::from_elem((2, 1), false);
        let material = Material {
            isotopes: vec![(hafnium, Value::Fitted(3.0e-4))],
            temperature_k: Value::Fitted(350.0),
        };
        let background = [Value::Fitted(0.02), Value::Known(0.0), Value::Known(0.0)];
        let map = crate::counts_map::MapMeasurement {
            time_edges_us: EDGES_US.map(f64::from).collect(),
            charge_ratio: 1.2,
            normalization: Value::Fitted(1.0),
            open_counts: open.view(),
            sample_counts: sample.view(),
            open_live: None,
            sample_live: None,
            excluded: none.view(),
            sample: sample_pixel.view(),
            empty: empty_pixel.view(),
            binning: 1,
            material: material.clone(),
            background,
            empty_background: [Value::Known(0.0); 3],
        };
        let calibration = Calibration {
            t0_us: Value::Measured {
                value: T0_US,
                sd: 0.01,
            },
            flight_path_m: Value::Fitted(FLIGHT_PATH_M),
            pulse: Pulse {
                alpha: [Value::Fitted(0.35), Value::Fitted(0.05)],
                beta: [Value::Known(0.0), Value::Known(0.25)],
                r: Value::Known(0.15),
                fwhm_squared_us2: Value::Known(0.0),
                energy_span_ev: (1.0, 200.0),
                n_tau: 256,
                line_span_ev: None,
                prior: Some(crate::pulse_calibration::PulsePrior {
                    numbers: vec![0, 1],
                    mean: vec![0.36, 0.05],
                    covariance: FlatMatrix {
                        data: vec![1.0e-4, -2.0e-5, -2.0e-5, 1.0e-5],
                        nrows: 2,
                        ncols: 2,
                    },
                }),
            },
        };
        let result = crate::counts_map::fit_map(&map, &calibration).expect("map");
        assert!(result.converged);
        let region = |r: usize, material: Option<Material>, background| Region {
            open_counts: run(r, 0),
            sample_counts: run(r, 1),
            open_live: None,
            sample_live: None,
            background,
            material,
        };
        let measurement = Measurement {
            time_edges_us: map.time_edges_us.clone(),
            charge_ratio: 1.2,
            normalization: map.normalization,
            regions: vec![
                region(0, Some(material), background),
                region(1, None, [Value::Known(0.0); 3]),
            ],
        };
        let joint = fit_counts(&measurement, &calibration).expect("joint fit");
        assert!(joint.converged);
        let covariance = joint.covariance.as_ref().expect("covariance");
        let fitted: Vec<Role> = quantities(&measurement, &calibration)
            .filter(|(_, value)| !matches!(value, Value::Known(_)))
            .map(|(role, _)| role)
            .collect();
        let at = |role: Role| fitted.iter().position(|&r| r == role).expect("fitted");
        let close = |actual: f64, a: Role, b: Role| {
            let (i, j) = (at(a), at(b));
            let scale = (covariance.get(i, i) * covariance.get(j, j)).sqrt();
            assert!(
                (actual - covariance.get(i, j)).abs() <= 1e-3 * scale,
                "{a:?} {b:?}: {actual} vs {}",
                covariance.get(i, j)
            );
        };
        let shared = [
            Role::Normalization,
            Role::T0,
            Role::FlightPath,
            Role::Pulse(0),
            Role::Pulse(1),
        ];
        let map_shared = result.shared_covariance.as_ref().expect("covariance");
        for (s, &a) in shared.iter().enumerate() {
            for (t, &b) in shared.iter().enumerate() {
                close(map_shared.get(s, t), a, b);
            }
        }
        let own = [
            Role::Density {
                region: 0,
                isotope: 0,
            },
            Role::Temperature { region: 0 },
            Role::Background { region: 0, term: 0 },
        ];
        let patch = result.covariance[[0, 0]].as_ref().expect("covariance");
        for (s, &a) in own.iter().enumerate() {
            for (t, &b) in own.iter().enumerate() {
                close(patch.get(s, t), a, b);
            }
        }
        let gaps = [
            (result.alpha[0] - joint.alpha[0])
                / covariance
                    .get(at(Role::Pulse(0)), at(Role::Pulse(0)))
                    .sqrt(),
            (result.t0_us - joint.t0_us) / covariance.get(at(Role::T0), at(Role::T0)).sqrt(),
            (result.densities[0][[0, 0]] - joint.regions[0].densities[0])
                / covariance.get(at(own[0]), at(own[0])).sqrt(),
        ];
        assert!(gaps.iter().all(|g| g.abs() <= 0.01), "{gaps:?}");
        let [Some(map_test), Some(joint_test)] =
            [result.pulse_consistency, joint.pulse_consistency]
        else {
            panic!("a pulse consistency");
        };
        assert_eq!(map_test.dof, joint_test.dof);
        assert!(
            (map_test.q - joint_test.q).abs() <= 1e-3 * joint_test.q,
            "{map_test:?} vs {joint_test:?}"
        );
    }

    #[test]
    fn the_background_lags_sammy_s_by_the_pulse_s_mean_delay() {
        let grid = grid(None);
        let (_, u_hi) = grid.range_us();
        let first_edge_us = f64::from(*EDGES_US.start());
        let beam = BeamSpline::constant(first_edge_us - T0_US, u_hi, 1.0e4);
        let isotopes: Arc<[ResonanceData]> = Arc::new([
            synthetic_isotope(72, 180, 20.0, 0.01, 0.06),
            synthetic_isotope(74, 182, 20.3, 0.01, 0.06),
        ]);
        let (charge_ratio, normalization) = (1.2, 0.93);
        let layout = Arc::new(Layout::new(
            roles(&[beam.coefficients().len()], &[Some(2)]),
            1,
        ));
        let model = RegionsModel::new(
            &grid,
            std::slice::from_ref(&beam),
            &[isotopes],
            charge_ratio,
            T0_US,
            &layout,
        );
        let counts = |background: [f64; 3]| {
            let params: Vec<f64> = beam
                .coefficients()
                .iter()
                .copied()
                .chain([3.0e-4, 5.0e-4, 300.0, normalization])
                .chain(background)
                .chain([T0_US, FLIGHT_PATH_M])
                .chain([0.35, 0.05, 0.0, 0.25, 0.15, 0.0])
                .collect();
            model.evaluate(&params).expect("counts")
        };
        let [b0, b1, b2] = [0.05, 0.5, -0.01];
        let (with, without) = (counts([b0, b1, b2]), counts([0.0; 3]));
        let bins = with.len() / 2;
        let clock = TOF_FACTOR * FLIGHT_PATH_M;
        for k in 0..bins {
            let u = first_edge_us + 0.5 + k as f64 - T0_US;
            let root_e = clock / u;
            let b = b0 + b1 / root_e + b2 * root_e;
            let slope = b1 / clock - b2 * clock / (u * u);
            let curvature = 2.0 * b2 * clock / u.powi(3);
            let energy = root_e * root_e;
            let (alpha, beta, r) = (ALPHA.eval(energy), BETA.eval(energy), R.eval(energy));
            let mean = 3.0 / alpha + r / beta;
            let square = mean * mean + 3.0 / (alpha * alpha) + r * (2.0 - r) / (beta * beta);
            let lagged = (with[bins + k] - without[bins + k]) / (charge_ratio * normalization);
            let residual = lagged / with[k] - b + slope * mean;
            let tolerance = 0.5 * slope.abs() + 0.5 * curvature.abs() * square;
            assert!(
                residual.abs() <= tolerance,
                "{k}: {residual} vs {tolerance}"
            );
        }
    }
}
