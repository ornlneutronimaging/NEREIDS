//! The beam, in neutrons per µs of flight time, fitted to the open-beam counts
//! it produces through the instrument pulse.

use std::ops::RangeInclusive;
use std::sync::Arc;

use nereids_fitting::error::FittingError;
use nereids_fitting::lm::{FitModel, FlatMatrix};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{PoissonConfig, PoissonResult, Prior, poisson_fit};
use nereids_physics::flight_time_grid::{FlightTimeGrid, FlightTimeGridError};
use nereids_physics::ikeda_carpenter::{DetectorPulse, EnergyLaw, IkedaCarpenterParams};

use crate::beam::BeamSpline;
use crate::counts_fit::Value;
use crate::error::PipelineError;
use crate::pulse_calibration::PulsePrior;

/// Largest `Σ_k (μ_fine − μ_coarse)² / μ_fine`, over bins predicted non-empty,
/// by which halving the grid may change the predicted counts for the finer
/// grid to be accepted.
pub const BOUND: f64 = 0.01;

pub(crate) const PULSE_NUMBERS: [&str; 6] = ["α₀", "α₁", "β₀", "β₁", "R", "h²"];

const RATE: (RangeInclusive<f64>, &str) = (0.0..=f64::INFINITY, "0 or more");

pub(crate) const PULSE_RANGES: [(RangeInclusive<f64>, &str); 6] =
    [RATE, RATE, RATE, RATE, (0.0..=1.0, "within 0–1"), RATE];

/// A neutron of flight time `u` over the flight path `flight_path_m` (m)
/// arrives at `t0_us + u` (µs) plus a delay drawn from `pulse`.
/// [`fit_counts`](crate::counts_fit::fit_counts) fits `t0_us`,
/// `flight_path_m` and the pulse's numbers unless they are known;
/// [`fit_open_beam`] uses their starting values.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub t0_us: Value,
    pub flight_path_m: Value,
    pub pulse: Pulse,
}

impl Calibration {
    pub(crate) fn instrument(&self) -> Result<Vec<FitParameter>, PipelineError> {
        let pulse = &self.pulse;
        let numbers = [
            pulse.alpha[0],
            pulse.alpha[1],
            pulse.beta[0],
            pulse.beta[1],
            pulse.r,
            pulse.fwhm_squared_us2,
        ];
        let mut parameters = vec![
            self.t0_us
                .parameter("t0", f64::NEG_INFINITY..=f64::INFINITY, "any real number")?,
            self.flight_path_m.parameter(
                "flight path",
                f64::MIN_POSITIVE..=f64::INFINITY,
                "positive",
            )?,
        ];
        for ((name, value), (range, allowed)) in
            PULSE_NUMBERS.into_iter().zip(numbers).zip(PULSE_RANGES)
        {
            parameters.push(value.parameter(name, range, allowed)?);
        }
        Ok(parameters)
    }
}

/// The Ikeda–Carpenter pulse with `α = α₀√E + α₁` and `β = β₀√E + β₁` in
/// 1/µs, `E` in eV, a storage fraction `R` constant over `energy_span_ev`, and
/// the proton pulse's triangle of FWHM `h` in µs, given by `h²`, in which the
/// counts are smooth down to `h = 0`.  `α` and `β` must be at least 1e-9 µs⁻¹
/// across the span, `β` even where `R` is 0.
#[derive(Debug, Clone)]
pub struct Pulse {
    /// `[α₀, α₁]`, in 1/(µs·√eV) and 1/µs, each 0 or more.
    pub alpha: [Value; 2],
    /// `[β₀, β₁]`, in 1/(µs·√eV) and 1/µs, each 0 or more.
    pub beta: [Value; 2],
    /// `R`, within 0–1.
    pub r: Value,
    /// `h²` in µs², 0 or more.
    pub fwhm_squared_us2: Value,
    /// `(low, high)`, the energies in eV the laws hold over; a window that
    /// neutrons from outside them can reach is refused.
    pub energy_span_ev: (f64, f64),
    /// Samples across the prompt core that find the pulse's rise, which sets
    /// the grid's first step; at least 8.
    pub n_tau: usize,
    /// `(low, high)`, the energies in eV of the lowest and highest resonance
    /// a calibration measured the pulse on.
    /// [`fit_counts`](crate::counts_fit::fit_counts) refuses a sample with a
    /// resonance outside them, of an isotope not known to be absent, between
    /// the energies of its last and first time edges; a resonance outside
    /// those energies, whose neutrons reach the window only through the
    /// pulse's delay, is not checked.  `None` checks nothing.
    pub line_span_ev: Option<(f64, f64)>,
    /// A calibration of some of the numbers, from
    /// [`PulseCalibration::calibration`](crate::pulse_calibration::PulseCalibration::calibration),
    /// that [`fit_counts`](crate::counts_fit::fit_counts) fits them with;
    /// [`fit_open_beam`] uses their starting values.
    pub prior: Option<PulsePrior>,
}

impl Pulse {
    pub(crate) fn at(&self, numbers: &[f64]) -> Result<DetectorPulse, PipelineError> {
        DetectorPulse::new(laws(numbers), self.energy_span_ev, self.n_tau)
            .map_err(|e| PipelineError::InvalidParameter(e.to_string()))
    }
}

pub(crate) fn laws(numbers: &[f64]) -> IkedaCarpenterParams {
    IkedaCarpenterParams {
        alpha: EnergyLaw::SqrtE {
            a0: numbers[0],
            a1: numbers[1],
        },
        beta: EnergyLaw::SqrtE {
            a0: numbers[2],
            a1: numbers[3],
        },
        r: EnergyLaw::Const(numbers[4]),
        burst_sigma_us: None,
        channel_fwhm_us: Some(numbers[5].sqrt()),
    }
}

/// The fitted open beam.
#[derive(Debug, Clone)]
pub struct OpenBeamFit {
    /// The beam per µs of flight time over the flight-time grid's range; its
    /// knots span the flight times of the first and last edges.
    /// `covariance` shows how well the counts determine it.
    pub beam: BeamSpline,
    /// Half the Poisson deviance at the fit.
    pub deviance: f64,
    /// Whether the fitter converged.  It certifies the minimisation, not that
    /// the counts determine every coefficient; `covariance` shows that.
    pub converged: bool,
    /// Covariance of the beam's coefficients, scaled by `overdispersion`, or
    /// at the Poisson scale when that is `None`; rows and columns of a
    /// coefficient the counts leave undetermined are NaN.  `None` when the
    /// fit did not converge.
    pub covariance: Option<FlatMatrix>,
    /// Variance of the counts over their Poisson variance, at least 1: the
    /// richest fitted beam's Pearson χ² over the bins predicted at least one
    /// count, per degree of freedom (each bin's one less its leverage), divided
    /// by one plus the mean of `(y − μ)/μ` over those bins (D. Fletcher,
    /// Biometrika 99, 230–237, 2012), for independent bins.  Beam structure
    /// finer than every candidate is counted in it as noise; such a beam is
    /// not supported and is fitted smooth, which shows as an overdispersion
    /// far above the detector's.  `None` when the fit did not converge or
    /// those bins leave less than one degree of freedom after the
    /// coefficients' leverage.
    pub overdispersion: Option<f64>,
    /// Whether the chosen beam is the richest fitted; the counts may then hold
    /// structure finer than its intervals, whose misfit `overdispersion`
    /// includes.
    pub at_limit: bool,
    /// Step, in µs, of the returned fit's grid, which meets [`BOUND`] when the
    /// fit converged.
    pub step_us: f64,
    /// Number of points of that grid.
    pub points: usize,
    /// How many times the first grid was halved to reach it.
    pub halvings: usize,
}

/// Fit the beam to the raw open-beam counts `open_counts` of the time bins
/// `time_edges_us` (µs), each bin recording the fraction `open_live` of the
/// neutrons arriving in it, or every one when that is `None`.  `ln φ`, the
/// logarithm of the beam, is a cubic spline in `ln u` with knots from the
/// first edge's flight time to the last edge's; outside them the beam is the
/// spline's continuation, which reaches the bins only through the pulse's
/// delay or, for a folded pulse, its early tail.  Neutrons faster than the
/// grid's range, each reaching the bins with less than
/// [`NEGLIGIBLE_ARRIVAL_PROBABILITY`](nereids_physics::ikeda_carpenter::NEGLIGIBLE_ARRIVAL_PROBABILITY)
/// chance, are left out.
///
/// The number of spline intervals doubles from one while the coefficients
/// number at most half the bins.  Each candidate is fitted on a grid halved,
/// and the beam refitted on each finer grid, until the refitted beam's
/// predicted counts differ from the coarser grid's by at most [`BOUND`].  A
/// candidate that does not converge, or needs more grid points than allowed,
/// ends the ladder.  The candidate with the lowest `D / overdispersion + 2k` is
/// returned, `D` twice its deviance and `k` its coefficients.
///
/// Counts too sparse to determine the beam are not supported: the fit may not
/// converge, may leave coefficients undetermined (NaN in `covariance`), or,
/// when its beam varies faster than a grid within the point cap resolves, end
/// the ladder or, for the first candidate, refuse the fit.
///
/// Beam structure at or near the window's first edge is not supported: before
/// the edge, where neutrons reach the bins only through the pulse's delay, the
/// beam is the spline's continuation, which the counts cannot check.  Its error
/// biases the fitted beam near that edge and, through the continuation's mean
/// curvature, elsewhere; structure just before the edge can leave the first
/// bins unmatched, which shows as `at_limit` with an overdispersion far above
/// the detector's.  Start the window where the beam is smooth.
///
/// # Errors
/// [`PipelineError::ShapeMismatch`] unless there is one count, and one live
/// fraction when given, per bin;
/// [`PipelineError::InvalidParameter`] if a count is not a whole non-negative
/// number, every count is zero, a live fraction is not in (0, 1], there are
/// fewer than 8 bins (one interval's four coefficients and as many bins again
/// to measure the noise), a known, starting or measured value of the
/// calibration is not finite and in its quantity's range, a measured one's sd
/// is not finite and positive, bounds are not `lower < upper` in that range
/// with the start between them, the pulse's energy span is not `0 < low <
/// high`, its `n_tau` is below 8, or its starting `α` or `β` is below 1e-9
/// µs⁻¹ at an end of the span;
/// [`PipelineError::FlightTimeGrid`] for the grid's refusals, including the
/// first candidate's halving past the point cap; [`PipelineError::Fitting`] if
/// the fitter refuses.
pub fn fit_open_beam(
    time_edges_us: &[f64],
    open_counts: &[f64],
    calibration: &Calibration,
    open_live: Option<&[f64]>,
) -> Result<OpenBeamFit, PipelineError> {
    let start: Vec<f64> = calibration
        .instrument()?
        .iter()
        .map(|parameter| parameter.value)
        .collect();
    let t0_us = start[0];
    let grid = Arc::new(FlightTimeGrid::new(
        time_edges_us,
        t0_us,
        start[1],
        &calibration.pulse.at(&start[2..])?,
    )?);
    validate_counts("open-beam", open_counts, time_edges_us.len() - 1)?;
    let live = validate_live("open-beam", open_live, time_edges_us.len() - 1)?;

    let coefficients = |intervals: usize| intervals + 3;
    let admits = |intervals: usize| 2 * coefficients(intervals) <= open_counts.len();
    if !admits(1) {
        return Err(PipelineError::InvalidParameter(format!(
            "the open-beam fit needs at least {} time bins, one interval's {} coefficients \
             and as many again to measure the noise; got {}",
            2 * coefficients(1),
            coefficients(1),
            open_counts.len()
        )));
    }

    let u_first = time_edges_us[0] - t0_us;
    let u_last = time_edges_us[time_edges_us.len() - 1] - t0_us;
    let per_unit_beam = grid.predict(&vec![1.0; grid.flight_times_us().len()])?;
    let recorded: f64 = per_unit_beam.iter().zip(&live).map(|(c, l)| l * c).sum();
    let per_us = open_counts.iter().sum::<f64>() / recorded;
    let mut ladder = vec![fit_beam(
        &grid,
        &BeamSpline::constant(u_first, u_last, per_us),
        open_counts,
        &live,
    )?];
    while let Some(start) = ladder
        .last()
        .filter(|candidate| candidate.fit.converged && admits(2 * candidate.beam.intervals()))
        .map(|candidate| candidate.beam.refined())
    {
        match fit_beam(&grid, &start, open_counts, &live) {
            Ok(candidate) if candidate.fit.converged => ladder.push(candidate),
            Ok(_)
            | Err(PipelineError::FlightTimeGrid(FlightTimeGridError::TooManyPoints { .. })) => {
                break;
            }
            Err(error) => return Err(error),
        }
    }

    let richest = &ladder[ladder.len() - 1].fit;
    let overdispersion = overdispersion(
        open_counts,
        richest,
        &counted(richest, 0..open_counts.len()),
        None,
    );
    let scale = overdispersion.unwrap_or(1.0);
    let criterion = |candidate: &Candidate| {
        2.0 * candidate.fit.result.deviance / scale
            + 2.0 * candidate.beam.coefficients().len() as f64
    };
    let (chosen, _) =
        ladder
            .iter()
            .map(criterion)
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |best, (i, q)| {
                    if q < best.1 { (i, q) } else { best }
                },
            );
    let at_limit = chosen + 1 == ladder.len();
    let Candidate { beam, fit } = ladder.swap_remove(chosen);
    Ok(OpenBeamFit {
        beam,
        deviance: fit.result.deviance,
        converged: fit.converged,
        covariance: fit
            .result
            .covariance
            .filter(|_| fit.converged)
            .map(|mut covariance| {
                covariance.data.iter_mut().for_each(|v| *v *= scale);
                covariance
            }),
        overdispersion,
        at_limit,
        step_us: fit.step_us,
        points: fit.points,
        halvings: fit.halvings,
    })
}

pub(crate) fn validate_counts(run: &str, counts: &[f64], bins: usize) -> Result<(), PipelineError> {
    if counts.len() != bins {
        return Err(PipelineError::ShapeMismatch(format!(
            "{} {run} counts for {bins} time bins",
            counts.len()
        )));
    }
    if let Some((bin, count)) = counts.iter().enumerate().find(|(_, c)| !whole(**c)) {
        return Err(PipelineError::InvalidParameter(format!(
            "{run} counts must be whole non-negative numbers, got {count} in bin {bin}"
        )));
    }
    if counts.iter().all(|&c| c == 0.0) {
        return Err(PipelineError::InvalidParameter(format!(
            "the {run} run has no counts"
        )));
    }
    Ok(())
}

pub(crate) fn whole(count: f64) -> bool {
    count.is_finite() && count >= 0.0 && count.fract() == 0.0
}

pub(crate) fn validate_live(
    run: &str,
    live: Option<&[f64]>,
    bins: usize,
) -> Result<Vec<f64>, PipelineError> {
    let live = live.map_or_else(|| vec![1.0; bins], <[f64]>::to_vec);
    if live.len() != bins {
        return Err(PipelineError::ShapeMismatch(format!(
            "{} {run} live fractions for {bins} time bins",
            live.len()
        )));
    }
    if let Some((bin, fraction)) = live
        .iter()
        .enumerate()
        .find(|(_, l)| !(**l > 0.0 && **l <= 1.0))
    {
        return Err(PipelineError::InvalidParameter(format!(
            "{run} live fractions must be in (0, 1], got {fraction} in bin {bin}"
        )));
    }
    Ok(live)
}

const COUNTS_TO_MEASURE_NOISE: f64 = 1.0;

pub(crate) fn counted(fit: &GridFit, bins: std::ops::Range<usize>) -> Vec<usize> {
    bins.filter(|&k| fit.predicted[k] >= COUNTS_TO_MEASURE_NOISE)
        .collect()
}

pub(crate) fn overdispersion(
    observed: &[f64],
    fit: &GridFit,
    counted: &[usize],
    shared_leverage: Option<&[f64]>,
) -> Option<f64> {
    let leverage = fit.result.leverage.as_ref().filter(|_| fit.converged)?;
    let (pearson, skew, freedom) =
        counted
            .iter()
            .fold((0.0, 0.0, 0.0), |(pearson, skew, freedom), &k| {
                let (y, mu) = (observed[k], fit.predicted[k]);
                (
                    pearson + (y - mu).powi(2) / mu,
                    skew + (y - mu) / mu,
                    freedom + 1.0 - leverage[k] - shared_leverage.map_or(0.0, |extra| extra[k]),
                )
            });
    let fletcher = 1.0 + skew / counted.len() as f64;
    (freedom >= 1.0)
        .then(|| (pearson / freedom / fletcher).clamp(1.0, f64::INFINITY))
        .filter(|phi| phi.is_finite())
}

struct Candidate {
    beam: BeamSpline,
    fit: GridFit,
}

fn fit_beam(
    first_grid: &Arc<FlightTimeGrid>,
    start: &BeamSpline,
    open_counts: &[f64],
    live: &[f64],
) -> Result<Candidate, PipelineError> {
    let mut parameters = ParameterSet::new(
        start
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| FitParameter::unbounded(format!("beam {i}"), c))
            .collect(),
    );
    let dispersion = vec![1.0; open_counts.len()];
    let fit = fit_on_halved_grids(
        first_grid,
        &mut parameters,
        open_counts,
        &dispersion,
        &[],
        |grid| {
            Ok(Recorded {
                model: OpenBeamModel::new(grid, start),
                live,
            })
        },
        |spread, _, _| Ok(spread <= BOUND),
    )?;
    Ok(Candidate {
        beam: start.with_coefficients(&fit.result.params),
        fit,
    })
}

pub(crate) struct GridFit {
    pub(crate) result: PoissonResult,
    pub(crate) converged: bool,
    pub(crate) predicted: Vec<f64>,
    pub(crate) coarse: Arc<FlightTimeGrid>,
    pub(crate) fine: Arc<FlightTimeGrid>,
    pub(crate) step_us: f64,
    pub(crate) points: usize,
    pub(crate) halvings: usize,
}

pub(crate) fn fit_on_halved_grids<M: FitModel>(
    first_grid: &Arc<FlightTimeGrid>,
    parameters: &mut ParameterSet,
    observed: &[f64],
    dispersion: &[f64],
    priors: &[Prior],
    model_on: impl Fn(&Arc<FlightTimeGrid>) -> Result<M, PipelineError>,
    done: impl Fn(f64, &FlightTimeGrid, &[f64]) -> Result<bool, PipelineError>,
) -> Result<GridFit, PipelineError> {
    let dispersed: Vec<f64> = observed
        .iter()
        .zip(dispersion)
        .map(|(y, d)| y / d)
        .collect();
    let mut grid = Arc::clone(first_grid);
    let mut coarse = model_on(&grid)?;
    let mut halvings = 0;
    loop {
        let finer = Arc::new(grid.halved()?);
        let fine = model_on(&finer)?;
        let result = poisson_fit(
            &Dispersed {
                model: &fine,
                dispersion,
            },
            &dispersed,
            priors,
            parameters,
            &PoissonConfig::default(),
        )?;
        let converged = result.converged && result.params.iter().all(|p| p.is_finite());
        let predicted = fine.evaluate(&result.params)?;
        let spread: f64 = predicted
            .iter()
            .zip(coarse.evaluate(&result.params)?)
            .filter(|(fine, _)| **fine > 0.0)
            .map(|(fine, coarse)| (fine - coarse).powi(2) / fine)
            .sum();
        let coarse_grid = std::mem::replace(&mut grid, finer);
        coarse = fine;
        halvings += 1;
        if !converged || done(spread, &coarse_grid, &result.params)? {
            return Ok(GridFit {
                result,
                converged,
                predicted,
                coarse: coarse_grid,
                fine: Arc::clone(&grid),
                step_us: grid.step_us(),
                points: grid.flight_times_us().len(),
                halvings,
            });
        }
    }
}

struct Dispersed<'a, M> {
    model: &'a M,
    dispersion: &'a [f64],
}

impl<M: FitModel> FitModel for Dispersed<'_, M> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let counts = self.model.evaluate(params)?;
        Ok(counts
            .iter()
            .zip(self.dispersion)
            .map(|(c, d)| c / d)
            .collect())
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let mut jacobian = self
            .model
            .analytical_jacobian(params, free_param_indices, y_current)?;
        for (row, d) in self.dispersion.iter().enumerate() {
            for col in 0..free_param_indices.len() {
                *jacobian.get_mut(row, col) /= d;
            }
        }
        Some(jacobian)
    }
}

pub(crate) struct Recorded<'a, M> {
    pub(crate) model: M,
    pub(crate) live: &'a [f64],
}

impl<M: FitModel> FitModel for Recorded<'_, M> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let counts = self.model.evaluate(params)?;
        Ok(counts.iter().zip(self.live).map(|(c, l)| l * c).collect())
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let mut jacobian = self
            .model
            .analytical_jacobian(params, free_param_indices, y_current)?;
        for (row, l) in self.live.iter().enumerate() {
            for col in 0..free_param_indices.len() {
                *jacobian.get_mut(row, col) *= l;
            }
        }
        Some(jacobian)
    }
}

pub(crate) fn combined(basis: &[[(usize, f64); 5]], coefficients: &[f64]) -> Vec<f64> {
    basis
        .iter()
        .map(|pairs| pairs.iter().map(|&(i, w)| w * coefficients[i]).sum())
        .collect()
}

pub(crate) fn weights_of(basis: &[[(usize, f64); 5]], index: usize) -> Vec<f64> {
    basis
        .iter()
        .map(|pairs| {
            pairs
                .iter()
                .filter(|&&(i, _)| i == index)
                .map(|&(_, w)| w)
                .sum()
        })
        .collect()
}

pub(crate) struct OpenBeamModel {
    grid: Arc<FlightTimeGrid>,
    basis: Vec<[(usize, f64); 5]>,
}

impl OpenBeamModel {
    pub(crate) fn new(grid: &Arc<FlightTimeGrid>, beam: &BeamSpline) -> Self {
        Self {
            grid: Arc::clone(grid),
            basis: grid
                .flight_times_us()
                .iter()
                .map(|&u| beam.basis(u))
                .collect(),
        }
    }

    pub(crate) fn beam(&self, coefficients: &[f64]) -> Vec<f64> {
        combined(&self.basis, coefficients)
            .into_iter()
            .map(f64::exp)
            .collect()
    }

    pub(crate) fn log_slope(&self, index: usize) -> Vec<f64> {
        weights_of(&self.basis, index)
    }

    pub(crate) fn counts(&self, values: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.grid
            .predict(values)
            .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
    }
}

impl FitModel for OpenBeamModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.counts(&self.beam(params))
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let beam = self.beam(params);
        let mut jacobian = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        for (col, &index) in free_param_indices.iter().enumerate() {
            let values: Vec<f64> = self
                .log_slope(index)
                .iter()
                .zip(&beam)
                .map(|(slope, phi)| slope * phi)
                .collect();
            for (row, value) in self.counts(&values).ok()?.into_iter().enumerate() {
                *jacobian.get_mut(row, col) = value;
            }
        }
        Some(jacobian)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use nereids_physics::ikeda_carpenter::{IkedaCarpenter, SynthesisGrid};

    use super::*;

    pub(crate) const FLIGHT_PATH_M: f64 = 25.0;
    pub(crate) const T0_US: f64 = 3.0;
    pub(crate) const EDGES_US: std::ops::RangeInclusive<u32> = 350..=470;
    pub(crate) const ALPHA: EnergyLaw = EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 };
    pub(crate) const BETA: EnergyLaw = EnergyLaw::Const(0.25);
    pub(crate) const R: EnergyLaw = EnergyLaw::Const(0.15);

    pub(crate) fn grid(channel_fwhm_us: Option<f64>) -> Arc<FlightTimeGrid> {
        let pulse = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: ALPHA,
                beta: BETA,
                r: R,
                burst_sigma_us: None,
                channel_fwhm_us,
            },
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 200.0,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model");
        let edges: Vec<f64> = EDGES_US.map(f64::from).collect();
        Arc::new(
            FlightTimeGrid::new(&edges, T0_US, FLIGHT_PATH_M, &pulse.detector_pulse())
                .expect("grid"),
        )
    }

    #[test]
    fn the_jacobian_is_the_slope_of_the_counts() {
        for channel_fwhm_us in [None, Some(2.0)] {
            let grid = grid(channel_fwhm_us);
            let (_, u_hi) = grid.range_us();
            for beam in [
                BeamSpline::constant(347.0, u_hi, 1.0e4),
                BeamSpline::constant(347.0, u_hi, 1.0e4).refined().refined(),
            ] {
                let model = OpenBeamModel::new(&grid, &beam);
                let coefficients: Vec<f64> = (0..beam.coefficients().len())
                    .map(|i| 9.0 + 0.3 * (i as f64).sin())
                    .collect();
                let counts = model.evaluate(&coefficients).expect("counts");
                let indices: Vec<usize> = (0..coefficients.len()).collect();
                let jacobian = model
                    .analytical_jacobian(&coefficients, &indices, &counts)
                    .expect("jacobian");
                for index in indices {
                    let h = 1e-4;
                    let shifted = |d: f64| {
                        let mut c = coefficients.clone();
                        c[index] += d;
                        model.evaluate(&c).expect("counts")
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
                            "{channel_fwhm_us:?} {} {index} {row}: {analytic} vs {slope}",
                            beam.intervals()
                        );
                    }
                }
            }
        }
    }
}
