//! The areal densities of a sample's isotopes, its temperature, the sample
//! run's normalization and background, and the timing offset and flight path,
//! fitted to the counts of an open-beam run and a sample run recorded in the
//! same time bins.

use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::RangeInclusive;
use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::error::FittingError;
use nereids_fitting::lm::{FitModel, FlatMatrix};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::Prior;
use nereids_physics::continuous_doppler::{SUPPORT_X, broaden_with_derivative};
use nereids_physics::doppler::DopplerParams;
use nereids_physics::flight_time_grid::{FlightTimeGrid, Rows};
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::resonance_center_energies;
use rayon::prelude::*;

use crate::beam::BeamSpline;
use crate::error::PipelineError;
use crate::open_beam::{
    Calibration, Recorded, combined, counted, fit_on_halved_grids, fit_open_beam, overdispersion,
    validate_counts, validate_live, weights_of,
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

/// An open-beam run and a sample run recorded in the same time bins.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Time-bin edges in µs.
    pub time_edges_us: Vec<f64>,
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
    /// `c_q`, the sample run's proton charge over the open-beam run's.
    pub charge_ratio: f64,
    /// `a`, the normalization of the sample run, positive.
    pub normalization: Value,
    /// `b0` (dimensionless), `b1` in √eV and `b2` in 1/√eV of the background
    /// `b(E) = b0 + b1/√E + b2·√E`, each any real number.
    pub background: [Value; 3],
    /// Each isotope in the sample with its areal density in atoms/barn, known,
    /// fitted or measured, at least 0.
    pub isotopes: Vec<(ResonanceData, Value)>,
    /// The sample's temperature in K, within 1–5000 K.
    pub temperature_k: Value,
}

/// The fitted densities, temperature, normalization, background, timing offset
/// and flight path.
#[derive(Debug, Clone)]
pub struct CountsFit {
    /// Areal density of each isotope in atoms/barn, in the order given: the
    /// known one, or the fitted one.
    pub densities: Vec<f64>,
    /// The sample's temperature in K: the known one, or the fitted one.
    pub temperature_k: f64,
    /// The normalization `a`: the known one, or the fitted one.
    pub normalization: f64,
    /// `b0` (dimensionless), `b1` in √eV and `b2` in 1/√eV, each the known or
    /// the fitted one.
    pub background: [f64; 3],
    /// The timing offset `t0` in µs: the known one, or the fitted one.
    pub t0_us: f64,
    /// The flight path in m: the known one, or the fitted one.
    pub flight_path_m: f64,
    /// Covariance of the fitted quantities among the densities, in the order
    /// given, the temperature, the normalization, `b0`, `b1`, `b2`, `t0` and
    /// the flight path, in that order: the inverse of the information at the
    /// fit, each run's expected information over its overdispersion plus
    /// `1/sd²` for each measured quantity.  The row and column of a quantity
    /// on one of its bounds, or that neither the counts nor a measurement
    /// determine, are NaN, and the other entries are conditional on every
    /// quantity that ended on a bound being held there; every entry is NaN
    /// when a fitted temperature ends at 1 K or 5000 K.  `None` when the fit
    /// did not converge.
    ///
    /// The error bars take the pulse, and every known quantity, as exact.
    /// They are not reliable where the counts barely determine a fitted
    /// temperature or barely separate it from a density, as at few counts or
    /// for a thin sample at modest counts.
    pub covariance: Option<FlatMatrix>,
    /// Whether each fitted quantity, in the covariance's order, ended on one
    /// of its bounds.
    pub on_bound: Vec<bool>,
    /// The beam per µs, as a function of the arrival time less the starting
    /// `t0`, fitted to both runs, with the intervals the open-beam fit chose.
    pub beam: BeamSpline,
    /// Whether the open-beam fit chose its richest beam; see
    /// [`OpenBeamFit::at_limit`](crate::open_beam::OpenBeamFit::at_limit).
    pub beam_at_limit: bool,
    /// Each run's half Poisson deviance over the overdispersion it was
    /// weighted with, summed over both runs at the fit.
    pub deviance: f64,
    /// Whether the fitter converged, the sample run's overdispersion settled,
    /// the grid met its rule at the fitted temperature and its flight times
    /// cover the fitted `t0` and flight path.
    pub converged: bool,
    /// Variance of the counts of the open-beam run, then of the sample run,
    /// over their Poisson variance: the value each run's counts are divided by
    /// in the returned fit.  The open-beam run's is the open-beam fit's
    /// [`OpenBeamFit::overdispersion`](crate::open_beam::OpenBeamFit::overdispersion);
    /// the sample run's is measured the same way, on the bins the first fit
    /// predicts at least one count, by the previous fit, and within 1% by the
    /// returned one when `converged`.  `None` when the run's counts have not
    /// measured it; the open-beam run is then weighted with 1, and the sample
    /// run as the open-beam run.
    pub overdispersion: [Option<f64>; 2],
    /// For each measured quantity, in the covariance's order, its fitted value
    /// less its measurement over the standard deviation of that difference,
    /// `√(sd² − variance)`: near 0 ± 1 when the counts agree with the
    /// measurement.  NaN for a quantity on a bound, and for every quantity
    /// when a fitted temperature ends at 1 K or 5000 K; it loses precision as
    /// the counts' information on the quantity vanishes beside the
    /// measurement's.  `None` when `covariance` is.
    pub measured_pulls: Option<Vec<f64>>,
    /// Step, in µs, of the fit's grid.
    pub step_us: f64,
    /// Number of points of that grid.
    pub points: usize,
    /// How many times the grid of
    /// [`FlightTimeGrid::new`](nereids_physics::flight_time_grid::FlightTimeGrid::new)
    /// was halved to reach it; that grid is built at the starting `t0` and
    /// flight path, or at fitted ones when the fit rebuilt it.
    pub halvings: usize,
}

/// Fit the areal densities, temperature, normalization and background terms
/// of `measurement`, and the timing offset and flight path of `calibration`,
/// that are not known, to the raw counts of both runs.  On a
/// uniform grid of flight times `u_i`, energies `E_i` and step `w`, the counts
/// in bin `k` are
///
/// ```text
/// O_k  = ℓ^O_k · w Σ_i φ_i P_ki
/// S_k  = ℓ^S_k · c_q · a · w Σ_i φ_i [T_i + b(E_i)] P_ki
/// T_i  = exp(−Σ_m n_m σ_m(E_i))
/// b(E) = b0 + b1/√E + b2·√E
/// ```
///
/// with `φ_i` the beam per µs, `P_ki` the chance of a neutron at `u_i`
/// arriving in bin `k`, `n_m` and `σ_m` each isotope's density and
/// Doppler-broadened total cross section, `c_q` the charge ratio, `a` the
/// normalization and `ℓ^O_k`, `ℓ^S_k` each run's live fraction.  The
/// background is beam neutrons that reach the detector another way, so it
/// passes through the pulse and scales with the normalization; SAMMY's
/// `BackA`, `BackB`, `BackC` (`cro/mnrm1.f90`) are `a·b0`, `a·b1`, `a·b2`, to
/// within the background's change over the pulse's delay.  SAMMY's
/// `BackD·exp(−BackF/√E)` term, and counts that bypass the pulse, such as
/// gammas, are not modelled.  The beam `φ` has the intervals
/// [`fit_open_beam`] chooses and is fitted with the rest to both runs,
/// starting from the open-beam fit.
///
/// The [`Calibration`]'s `t0` and flight path `L` are fitted unless known.
/// The grid is built at a `t0₀` and `L₀`, at first the starting ones.  Grid
/// point `i` keeps its energy and arrives at `t0 + (L/L₀)·u_i`, with `u_i`
/// its flight time at `t0₀` and `L₀`, standing for `(L/L₀)·w` of flight time;
/// the beam is a function of the arrival time less the starting `t0`.
///
/// The grid's first step is at most half the narrowest Doppler full width at
/// half maximum, in flight time, of any resonance inside its energy span, at
/// the starting temperature; it is then halved until the counts of both runs
/// meet [`BOUND`](crate::open_beam::BOUND).  The step is uniform, so a wide
/// window whose span holds a narrow resonance at high energy, or a low fitted
/// temperature, can exceed the grid's point cap; a temperature the counts
/// barely determine can run to 1 K and refuse the fit that way.
///
/// The fit minimizes each run's half Poisson deviance over its
/// overdispersion, plus `½((x − value)/sd)²` for each [`Value::Measured`]
/// quantity `x`.  The open-beam run is weighted with the open-beam fit's
/// overdispersion, and the sample run first with the same.  The fit is
/// repeated from its answer while the sample run's overdispersion, measured on
/// the bins the first fit predicts at least one count, changes by more than
/// 1%, the coarser grid of the accepted pair is wider than the rule at the
/// fitted temperature, or its flight times miss, by more than one step at
/// either end, those the fitted `t0` and `L` need.  The next first grid is the
/// finer of that pair's coarser grid and the rule's grid at the fitted
/// temperature, or, when the flight times miss, the rule's grid of a grid
/// built at the fitted `t0` and `L`.  After twenty fits it is reported
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
/// The covariance assumes each run's bins are independent.
///
/// # Errors
/// [`PipelineError::ShapeMismatch`] unless each run has one count, and one
/// live fraction when given, per bin;
/// [`PipelineError::InvalidParameter`] if a count is not a whole non-negative
/// number, a run has no counts, a live fraction is not in (0, 1], the charge
/// ratio is not finite and positive, a known, starting or measured value is
/// not finite and in its quantity's range, a measured value's sd is not finite
/// and positive, bounds are not `lower < upper` in that range
/// with the start between them, there are no isotopes, an isotope is listed
/// twice, an isotope's resonance data are not finite, or the energies its
/// broadened cross section reads, at the known temperature or at the upper
/// bound of a fitted one, down to zero for a window within the thermal
/// spread of zero energy, are not inside a single one of its evaluated
/// (SLBW, MLBW or Reich–Moore) resolved ranges;
/// [`PipelineError::UnmodelledCounts`] if at the fit, converged or not, a bin
/// is predicted negative or non-finite counts, or holds counts predicted below
/// [`NEGLIGIBLE_PREDICTION`]: starting or known values the fitter cannot
/// leave;
/// everything [`fit_open_beam`] refuses; [`PipelineError::FlightTimeGrid`]
/// for the grid's refusals, including more points than it allows;
/// [`PipelineError::Fitting`] if the fitter fails, or the cross sections
/// fail at the start; a failure at a trial temperature is a rejected step.
pub fn fit_counts(
    measurement: &Measurement,
    calibration: &Calibration,
) -> Result<CountsFit, PipelineError> {
    let Measurement {
        time_edges_us,
        open_counts,
        sample_counts,
        open_live,
        sample_live,
        charge_ratio,
        normalization,
        background,
        isotopes,
        temperature_k,
    } = measurement;
    let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
    let measured: Vec<(usize, f64, f64)> = isotopes
        .iter()
        .map(|(_, density)| density)
        .chain([temperature_k, normalization])
        .chain(background)
        .chain([&calibration.t0_us, &calibration.flight_path_m])
        .enumerate()
        .filter_map(|(offset, value)| match *value {
            Value::Measured { value, sd } => Some((offset, value, sd)),
            _ => None,
        })
        .collect();
    if !(charge_ratio.is_finite() && *charge_ratio > 0.0) {
        return invalid(format!(
            "the charge ratio must be finite and positive, got {charge_ratio}"
        ));
    }
    let normalization = normalization.parameter(
        "normalization",
        f64::MIN_POSITIVE..=f64::INFINITY,
        "positive",
    )?;
    let background = ["b0", "b1", "b2"]
        .into_iter()
        .zip(background)
        .map(|(name, term)| term.parameter(name, f64::NEG_INFINITY..=f64::INFINITY, "of any sign"))
        .collect::<Result<Vec<FitParameter>, PipelineError>>()?;
    if isotopes.is_empty() {
        return invalid("the sample has no isotopes".into());
    }
    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    let temperature = temperature_k.parameter(
        "temperature",
        t_low..=t_high,
        &format!("within {t_low}–{t_high} K"),
    )?;
    let reach_k = if temperature.fixed {
        temperature.value
    } else {
        temperature.upper
    };
    let mut dopplers = Vec::with_capacity(isotopes.len());
    let mut densities = Vec::with_capacity(isotopes.len());
    for (i, (isotope, density)) in isotopes.iter().enumerate() {
        if isotopes[..i]
            .iter()
            .any(|(other, _)| other.za == isotope.za)
        {
            return invalid(format!(
                "{} is listed twice; its densities cannot be told apart",
                isotope.isotope
            ));
        }
        densities.push(density.parameter(
            format!("density of {}", isotope.isotope),
            0.0..=f64::INFINITY,
            "0 or more",
        )?);
        if !finite(isotope) {
            return invalid(format!(
                "the resonance data of {} are not finite",
                isotope.isotope
            ));
        }
        dopplers.push(
            DopplerParams::new(reach_k, isotope.awr)
                .map_err(|e| PipelineError::InvalidParameter(e.to_string()))?,
        );
    }
    let (t0, flight_path) = calibration.energy_scale()?;
    let beam_origin_us = t0.value;
    let mut base = FlightTimeGrid::new(
        time_edges_us,
        t0.value,
        flight_path.value,
        &calibration.pulse,
    )?;
    let bins = time_edges_us.len() - 1;
    validate_counts("open-beam", open_counts, bins)?;
    validate_counts("sample", sample_counts, bins)?;
    let open_live = validate_live("open-beam", open_live.as_deref(), bins)?;
    let sample_live = validate_live("sample", sample_live.as_deref(), bins)?;

    let in_span = |grid: &FlightTimeGrid| -> Result<Vec<Vec<f64>>, PipelineError> {
        let energies = grid.energies_ev();
        let span_ev = (energies[energies.len() - 1], energies[0]);
        let mut resonances_in_span = Vec::with_capacity(isotopes.len());
        for ((isotope, _), doppler) in isotopes.iter().zip(&dopplers) {
            let read = (
                (span_ev.0.sqrt() - SUPPORT_X * doppler.u())
                    .max(0.0)
                    .powi(2),
                (span_ev.1.sqrt() + SUPPORT_X * doppler.u()).powi(2),
            );
            if !isotope.ranges.iter().any(|range| {
                range.is_evaluable() && range.energy_low <= read.0 && read.1 <= range.energy_high
            }) {
                return Err(PipelineError::InvalidParameter(format!(
                    "the Doppler-broadened cross section of {} reads {:.6e}–{:.6e} eV, which no \
                     single one of its evaluated (SLBW, MLBW or Reich–Moore) resolved ranges holds",
                    isotope.isotope, read.0, read.1
                )));
            }
            resonances_in_span.push(
                resonance_center_energies(&[isotope])
                    .into_iter()
                    .filter(|e| (span_ev.0..=span_ev.1).contains(e))
                    .collect::<Vec<f64>>(),
            );
        }
        Ok(resonances_in_span)
    };
    let mut resonances_in_span = in_span(&base)?;
    let half_maximum = 2.0 * std::f64::consts::LN_2.sqrt();
    let narrowest_us = |grid: &FlightTimeGrid,
                        resonances_in_span: &[Vec<f64>],
                        temperature_k: f64|
     -> Result<f64, PipelineError> {
        let clock = TOF_FACTOR * grid.flight_path_m();
        let mut narrowest = f64::INFINITY;
        for ((isotope, _), energies) in isotopes.iter().zip(resonances_in_span) {
            let doppler = DopplerParams::new(temperature_k, isotope.awr)
                .map_err(|e| PipelineError::InvalidParameter(e.to_string()))?;
            for &energy in energies {
                let width_ev = half_maximum * doppler.doppler_width(energy);
                narrowest = narrowest.min(clock / energy.sqrt() * width_ev / (2.0 * energy));
            }
        }
        Ok(narrowest)
    };
    let first_grid = |grid: &FlightTimeGrid,
                      resonances_in_span: &[Vec<f64>],
                      temperature_k: f64|
     -> Result<(Arc<FlightTimeGrid>, usize), PipelineError> {
        let rule_us = 0.5 * narrowest_us(grid, resonances_in_span, temperature_k)?;
        let mut first = grid.clone();
        let mut halvings = 0;
        while first.step_us() > rule_us {
            first = first.halved()?;
            halvings += 1;
        }
        Ok((Arc::new(first), halvings))
    };

    let open = fit_open_beam(time_edges_us, open_counts, calibration, Some(&open_live))?;
    let layout = Layout::new(open.beam.coefficients().len(), isotopes.len());
    let mut parameters = ParameterSet::new(
        open.beam
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| FitParameter::unbounded(format!("beam {i}"), c))
            .chain(densities)
            .chain([temperature, normalization])
            .chain(background)
            .chain([t0, flight_path])
            .collect(),
    );
    let resonances: Arc<[ResonanceData]> = isotopes.iter().map(|(data, _)| data.clone()).collect();
    let observed: Vec<f64> = open_counts.iter().chain(sample_counts).copied().collect();
    let live: Vec<f64> = open_live.into_iter().chain(sample_live).collect();
    let priors: Vec<Prior> = measured
        .iter()
        .map(|&(offset, mean, sd)| Prior {
            parameter: layout.densities + offset,
            mean,
            sd,
        })
        .collect();
    let mut first = first_grid(
        &base,
        &resonances_in_span,
        parameters.params[layout.temperature].value,
    )?;
    let mut weights = [open.overdispersion.unwrap_or(1.0); 2];
    let mut sample_measured = false;
    let mut noise_bins = None;
    let mut passes = 0;
    let (fit, rule_halvings, overdispersion, settled) = loop {
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
                    model: TwoRunModel::new(
                        grid,
                        &open.beam,
                        &resonances,
                        *charge_ratio,
                        beam_origin_us,
                    ),
                    live: &live,
                })
            },
            |grid, params| Ok(grid.covers(params[layout.t0], params[layout.flight_path])?),
        )?;
        let fitted = |index: usize| fit.result.params[index];
        let fitted_k = fitted(layout.temperature);
        let noise_bins = noise_bins.get_or_insert_with(|| counted(&fit, bins..2 * bins));
        let sample = overdispersion(&observed, &fit, noise_bins);
        let next = sample.unwrap_or(weights[0]);
        let settled = (next / weights[1] - 1.0).abs() <= SETTLED_OVERDISPERSION;
        let resolved =
            2.0 * fit.step_us <= 0.5 * narrowest_us(&base, &resonances_in_span, fitted_k)?;
        let covered = fit.converged
            && fit
                .coarse
                .covers(fitted(layout.t0), fitted(layout.flight_path))?;
        if !fit.converged || (settled && resolved && covered) || passes == MOST_PASSES {
            let weighted = [
                open.overdispersion,
                (sample_measured || (settled && sample.is_some())).then_some(weights[1]),
            ];
            break (fit, first.1, weighted, settled && resolved && covered);
        }
        if covered {
            let resumed = (Arc::clone(&fit.coarse), first.1 + fit.halvings - 1);
            first = [first_grid(&base, &resonances_in_span, fitted_k)?, resumed]
                .into_iter()
                .min_by(|a, b| a.0.step_us().total_cmp(&b.0.step_us()))
                .expect("two grids");
        } else {
            base = FlightTimeGrid::new(
                time_edges_us,
                fitted(layout.t0),
                fitted(layout.flight_path),
                &calibration.pulse,
            )?;
            resonances_in_span = in_span(&base)?;
            first = first_grid(&base, &resonances_in_span, fitted_k)?;
        }
        weights[1] = next;
        sample_measured = sample.is_some();
    };
    let converged = fit.converged && settled;

    if let Some((k, (&counts, &predicted))) =
        observed
            .iter()
            .zip(&fit.predicted)
            .enumerate()
            .find(|(_, (y, mu))| {
                !mu.is_finite() || **mu < 0.0 || (**y > 0.0 && **mu < NEGLIGIBLE_PREDICTION)
            })
    {
        let run = if k < bins { "open-beam" } else { "sample" };
        return Err(PipelineError::UnmodelledCounts {
            run,
            bin: k % bins,
            counts,
            predicted,
        });
    }

    let free = parameters.free_indices();
    let on_edge = free.contains(&layout.temperature)
        && [t_low, t_high].contains(&fit.result.params[layout.temperature]);
    let sample_quantities: Vec<usize> = (0..free.len())
        .filter(|&p| free[p] >= layout.densities)
        .collect();
    let covariance = fit
        .result
        .covariance
        .as_ref()
        .filter(|_| converged)
        .map(|full| {
            let size = sample_quantities.len();
            let mut block = FlatMatrix::zeros(size, size);
            for (a, &p) in sample_quantities.iter().enumerate() {
                for (b, &q) in sample_quantities.iter().enumerate() {
                    *block.get_mut(a, b) = if on_edge { f64::NAN } else { full.get(p, q) };
                }
            }
            block
        });
    let params = &fit.result.params;
    let measured_pulls = covariance.as_ref().map(|block| {
        measured
            .iter()
            .map(|&(offset, mean, sd)| {
                let parameter = layout.densities + offset;
                let a = sample_quantities
                    .iter()
                    .position(|&p| free[p] == parameter)
                    .expect("a measured quantity is fitted");
                (params[parameter] - mean) / (sd * sd - block.get(a, a)).sqrt()
            })
            .collect()
    });
    Ok(CountsFit {
        densities: params[layout.densities..layout.temperature].to_vec(),
        temperature_k: params[layout.temperature],
        normalization: params[layout.normalization],
        background: [0, 1, 2].map(|i| params[layout.background + i]),
        t0_us: params[layout.t0],
        flight_path_m: params[layout.flight_path],
        covariance,
        on_bound: sample_quantities
            .iter()
            .map(|&p| fit.result.on_bound[p])
            .collect(),
        beam: open.beam.with_coefficients(&params[..layout.densities]),
        beam_at_limit: open.at_limit,
        deviance: fit.result.deviance,
        converged,
        overdispersion,
        measured_pulls,
        step_us: fit.step_us,
        points: fit.points,
        halvings: rule_halvings + fit.halvings,
    })
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

#[derive(Clone, Copy)]
struct Layout {
    densities: usize,
    temperature: usize,
    normalization: usize,
    background: usize,
    t0: usize,
    flight_path: usize,
}

impl Layout {
    fn new(beam_coefficients: usize, isotopes: usize) -> Self {
        let temperature = beam_coefficients + isotopes;
        Self {
            densities: beam_coefficients,
            temperature,
            normalization: temperature + 1,
            background: temperature + 2,
            t0: temperature + 5,
            flight_path: temperature + 6,
        }
    }
}

struct TwoRunModel {
    grid: Arc<FlightTimeGrid>,
    spline: BeamSpline,
    beam_origin_us: f64,
    isotopes: Arc<[ResonanceData]>,
    energies: Vec<f64>,
    shapes: [Vec<f64>; 3],
    charge_ratio: f64,
    layout: Layout,
    cross_sections: RefCell<Option<CrossSections>>,
    scale: RefCell<Option<Scale>>,
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
    rows: Rows,
    basis: Vec<[(usize, f64); 5]>,
    basis_slope: Vec<[(usize, f64); 5]>,
}

fn predicted(rows: &Rows, values: &[f64]) -> Result<Vec<f64>, FittingError> {
    rows.predict(values)
        .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
}

impl TwoRunModel {
    fn new(
        grid: &Arc<FlightTimeGrid>,
        beam: &BeamSpline,
        isotopes: &Arc<[ResonanceData]>,
        charge_ratio: f64,
        beam_origin_us: f64,
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
            spline: beam.clone(),
            beam_origin_us,
            isotopes: Arc::clone(isotopes),
            energies,
            shapes,
            charge_ratio,
            layout: Layout::new(beam.coefficients().len(), isotopes.len()),
            cross_sections: RefCell::new(None),
            scale: RefCell::new(None),
        }
    }

    fn at(&self, temperature_k: f64) -> Result<std::cell::Ref<'_, CrossSections>, FittingError> {
        let current = self
            .cross_sections
            .borrow()
            .as_ref()
            .is_some_and(|c| c.temperature_k.to_bits() == temperature_k.to_bits());
        if !current {
            let energies = &self.energies;
            let (values, slopes) = self
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
            *self.cross_sections.borrow_mut() = Some(CrossSections {
                temperature_k,
                values,
                slopes,
            });
        }
        Ok(std::cell::Ref::map(self.cross_sections.borrow(), |c| {
            c.as_ref().expect("computed above")
        }))
    }

    fn scale(
        &self,
        t0_us: f64,
        flight_path_m: f64,
    ) -> Result<std::cell::Ref<'_, Scale>, FittingError> {
        let key = |t0: f64, l: f64| (t0.to_bits(), l.to_bits());
        let current = self
            .scale
            .borrow()
            .as_ref()
            .is_some_and(|s| key(s.t0_us, s.flight_path_m) == key(t0_us, flight_path_m));
        if !current {
            let grid = &self.grid;
            let rows = if key(t0_us, flight_path_m) == key(grid.t0_us(), grid.flight_path_m()) {
                grid.rows().clone()
            } else {
                grid.rows_at(t0_us, flight_path_m)
                    .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?
            };
            let (shift, stretch) = (
                t0_us - self.beam_origin_us,
                flight_path_m / grid.flight_path_m(),
            );
            let abscissae: Vec<f64> = grid
                .flight_times_us()
                .iter()
                .map(|u| shift + stretch * u)
                .collect();
            *self.scale.borrow_mut() = Some(Scale {
                t0_us,
                flight_path_m,
                rows,
                basis: abscissae.iter().map(|&a| self.spline.basis(a)).collect(),
                basis_slope: abscissae
                    .iter()
                    .map(|&a| self.spline.basis_slope(a))
                    .collect(),
            });
        }
        Ok(std::cell::Ref::map(self.scale.borrow(), |s| {
            s.as_ref().expect("computed above")
        }))
    }

    fn beams(&self, params: &[f64], scale: &Scale) -> Result<Beams, FittingError> {
        let layout = self.layout;
        let densities = &params[layout.densities..layout.temperature];
        let sigma = self.at(params[layout.temperature])?;
        let open: Vec<f64> = combined(&scale.basis, &params[..layout.densities])
            .into_iter()
            .map(f64::exp)
            .collect();
        let normalized: Vec<f64> = open
            .iter()
            .map(|phi| self.charge_ratio * params[layout.normalization] * phi)
            .collect();
        let (transmitted, sample) = normalized
            .iter()
            .enumerate()
            .map(|(j, beam)| {
                let depth: f64 = densities
                    .iter()
                    .zip(&sigma.values)
                    .map(|(n, sigma)| n * sigma[j])
                    .sum();
                let background: f64 = params[layout.background..layout.t0]
                    .iter()
                    .zip(&self.shapes)
                    .map(|(b, g)| b * g[j])
                    .sum();
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

impl FitModel for TwoRunModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let layout = self.layout;
        let scale = self.scale(params[layout.t0], params[layout.flight_path])?;
        let Beams { open, sample, .. } = self.beams(params, &scale)?;
        let mut counts = predicted(&scale.rows, &open)?;
        counts.extend(predicted(&scale.rows, &sample)?);
        Ok(counts)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let layout = self.layout;
        let (t0_us, flight_path_m) = (params[layout.t0], params[layout.flight_path]);
        let scale = self.scale(t0_us, flight_path_m).ok()?;
        let Beams {
            open,
            normalized,
            transmitted,
            sample,
        } = self.beams(params, &scale).ok()?;
        let sigma = self.at(params[layout.temperature]).ok()?;
        let counts = |values: &[f64]| predicted(&scale.rows, values).ok();
        let times = |beam: &[f64], slope: &[f64]| -> Vec<f64> {
            beam.iter().zip(slope).map(|(b, s)| b * s).collect()
        };
        let mut arrival: Option<Rows> = None;
        let mut jacobian = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        for (col, &index) in free_param_indices.iter().enumerate() {
            let (open_column, sample_column) = if index < layout.densities {
                let slope = weights_of(&scale.basis, index);
                (
                    counts(&times(&open, &slope))?,
                    counts(&times(&sample, &slope))?,
                )
            } else if index < layout.temperature {
                let values = &sigma.values[index - layout.densities];
                let slope: Vec<f64> = values.iter().map(|s| -s).collect();
                (
                    vec![0.0; y_current.len() / 2],
                    counts(&times(&transmitted, &slope))?,
                )
            } else if index == layout.temperature {
                let broadening: Vec<f64> = (0..open.len())
                    .map(|j| {
                        -params[layout.densities..layout.temperature]
                            .iter()
                            .zip(&sigma.slopes)
                            .map(|(n, slope)| n * slope[j])
                            .sum::<f64>()
                    })
                    .collect();
                (
                    vec![0.0; y_current.len() / 2],
                    counts(&times(&transmitted, &broadening))?,
                )
            } else if index == layout.normalization {
                let slope = vec![1.0 / params[index]; open.len()];
                (
                    vec![0.0; y_current.len() / 2],
                    counts(&times(&sample, &slope))?,
                )
            } else if index < layout.t0 {
                let shape = &self.shapes[index - layout.background];
                (
                    vec![0.0; y_current.len() / 2],
                    counts(&times(&normalized, shape))?,
                )
            } else {
                if arrival.is_none() {
                    arrival = Some(self.grid.arrival_slopes_at(t0_us, flight_path_m).ok()?);
                }
                let arrival = arrival.as_ref()?;
                let log_beam_slope = combined(&scale.basis_slope, &params[..layout.densities]);
                let moved = |beam: &[f64], reach: &[f64]| -> Option<Vec<f64>> {
                    let reached = times(beam, reach);
                    let beam_part = counts(&times(&reached, &log_beam_slope))?;
                    let pulse_part = predicted(arrival, &reached).ok()?;
                    Some(
                        beam_part
                            .iter()
                            .zip(&pulse_part)
                            .map(|(b, p)| b + p)
                            .collect(),
                    )
                };
                if index == layout.t0 {
                    let reach = vec![1.0; open.len()];
                    (moved(&open, &reach)?, moved(&sample, &reach)?)
                } else {
                    let reach: Vec<f64> = self
                        .grid
                        .flight_times_us()
                        .iter()
                        .map(|u| u / self.grid.flight_path_m())
                        .collect();
                    let stretched = |beam: &[f64]| -> Option<Vec<f64>> {
                        let weight = counts(beam)?;
                        Some(
                            moved(beam, &reach)?
                                .iter()
                                .zip(&weight)
                                .map(|(m, x)| m + x / flight_path_m)
                                .collect(),
                        )
                    };
                    (stretched(&open)?, stretched(&sample)?)
                }
            };
            for (row, value) in open_column.into_iter().chain(sample_column).enumerate() {
                *jacobian.get_mut(row, col) = value;
            }
        }
        Some(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use nereids_endf::resonance::test_support::synthetic_isotope;

    use super::*;
    use crate::open_beam::tests::{ALPHA, BETA, EDGES_US, FLIGHT_PATH_M, R, T0_US, grid};

    #[test]
    fn the_jacobian_is_the_slope_of_both_runs_counts() {
        for channel_fwhm_us in [None, Some(0.35)] {
            jacobian_against_central_differences(&grid(channel_fwhm_us));
        }
    }

    fn jacobian_against_central_differences(grid: &Arc<FlightTimeGrid>) {
        let (_, u_hi) = grid.range_us();
        let beam = BeamSpline::constant(347.0, u_hi, 1.0e4).refined().refined();
        let isotopes: Arc<[ResonanceData]> = Arc::new([
            synthetic_isotope(72, 180, 20.0, 0.01, 0.06),
            synthetic_isotope(74, 182, 20.3, 0.01, 0.06),
        ]);
        let params: Vec<f64> = (0..beam.coefficients().len())
            .map(|i| 9.0 + 0.3 * (i as f64).sin())
            .chain([3.0e-4, 5.0e-4, 300.0, 0.93, 0.05, 0.5, -0.01])
            .chain([T0_US + 0.05, FLIGHT_PATH_M + 0.003])
            .collect();
        let model = TwoRunModel::new(grid, &beam, &isotopes, 1.2, T0_US);
        let live: Vec<f64> = (0..model.evaluate(&params).expect("counts").len())
            .map(|k| 0.9 + 0.1 * (0.3 * k as f64).sin())
            .collect();
        let model = Recorded { model, live: &live };
        let layout = model.model.layout;
        let mut elsewhere = params.clone();
        elsewhere[layout.temperature] = 250.0;
        elsewhere[layout.t0] = T0_US - 0.05;
        elsewhere[layout.flight_path] = FLIGHT_PATH_M - 0.003;
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
        let model = TwoRunModel::new(&grid, &beam, &isotopes, charge_ratio, T0_US);
        let counts = |background: [f64; 3]| {
            let params: Vec<f64> = beam
                .coefficients()
                .iter()
                .copied()
                .chain([3.0e-4, 5.0e-4, 300.0, normalization])
                .chain(background)
                .chain([T0_US, FLIGHT_PATH_M])
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
