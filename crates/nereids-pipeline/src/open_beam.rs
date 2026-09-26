//! The beam, in neutrons per µs of flight time, fitted to the open-beam counts
//! it produces through the instrument pulse.

use std::sync::Arc;

use nereids_fitting::error::FittingError;
use nereids_fitting::lm::{FitModel, FlatMatrix};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{PoissonConfig, PoissonResult, poisson_fit};
use nereids_physics::flight_time_grid::{FlightTimeGrid, FlightTimeGridError};
use nereids_physics::ikeda_carpenter::IkedaCarpenter;

use crate::beam::BeamSpline;
use crate::error::PipelineError;

/// Largest `Σ_k (μ_fine − μ_coarse)² / μ_fine`, over bins predicted non-empty,
/// by which halving the grid may change the predicted counts for the finer
/// grid to be accepted.
pub const BOUND: f64 = 0.01;

/// A neutron of flight time `u` arrives at `t0_us + u` plus a delay drawn from
/// `pulse`, whose flight path sets `u` for each energy.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub t0_us: f64,
    pub pulse: Arc<IkedaCarpenter>,
}

/// The fitted open beam.
#[derive(Debug, Clone)]
pub struct OpenBeamFit {
    /// The beam per µs of flight time over the flight-time grid's range; its
    /// knots span the first edge's flight time to the grid's slow end.
    /// `covariance` shows how well the counts determine it.
    pub beam: BeamSpline,
    /// Half the Poisson deviance at the fit.
    pub deviance: f64,
    /// Whether the fitter converged.  It certifies the minimisation, not that
    /// the counts determine every coefficient; `covariance` shows that.
    pub converged: bool,
    /// Covariance of the beam's coefficients, scaled by `overdispersion`; rows
    /// and columns of a coefficient the counts leave undetermined are NaN.
    /// `None` when the fit did not converge.
    pub covariance: Option<FlatMatrix>,
    /// Variance of the counts over their Poisson variance, at least 1: the
    /// richest fitted beam's Pearson χ² per degree of freedom over the bins
    /// predicted at least one count, for independent bins.  Beam structure
    /// finer than every candidate is counted in it as noise; such a beam is
    /// not supported and is fitted smooth, which shows as an overdispersion
    /// far above the detector's.  `None` when the fit did not converge or
    /// those bins do not outnumber the coefficients.
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
/// `time_edges_us` (µs).  `ln φ`, the logarithm of the beam, is a cubic spline
/// in `ln u` with knots from the first edge's flight time to the flight-time
/// grid's slow end; neutrons faster than the grid's range, each reaching the
/// bins with less than
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
/// [`PipelineError::ShapeMismatch`] unless there is one count per bin;
/// [`PipelineError::InvalidParameter`] if a count is not a whole non-negative
/// number, every count is zero, or there are fewer than 8 bins (one interval's
/// four coefficients and as many bins again to measure the noise);
/// [`PipelineError::FlightTimeGrid`] for the grid's refusals, including the
/// first candidate's halving past the point cap; [`PipelineError::Fitting`] if
/// the fitter refuses.
pub fn fit_open_beam(
    time_edges_us: &[f64],
    open_counts: &[f64],
    calibration: &Calibration,
) -> Result<OpenBeamFit, PipelineError> {
    let grid = FlightTimeGrid::new(time_edges_us, calibration.t0_us, &calibration.pulse)?;
    validate_counts("open-beam", open_counts, time_edges_us.len() - 1)?;

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

    let (_, u_last) = grid.range_us();
    let u_first = time_edges_us[0] - calibration.t0_us;
    let per_unit_beam = grid.predict(&vec![1.0; grid.flight_times_us().len()])?;
    let per_us = open_counts.iter().sum::<f64>() / per_unit_beam.iter().sum::<f64>();
    let mut ladder = vec![fit_beam(
        &grid,
        &BeamSpline::constant(u_first, u_last, per_us),
        open_counts,
    )?];
    while let Some(start) = ladder
        .last()
        .filter(|candidate| candidate.fit.converged && admits(2 * candidate.beam.intervals()))
        .map(|candidate| candidate.beam.refined())
    {
        match fit_beam(&grid, &start, open_counts) {
            Ok(candidate) if candidate.fit.converged => ladder.push(candidate),
            Ok(_)
            | Err(PipelineError::FlightTimeGrid(FlightTimeGridError::TooManyPoints { .. })) => {
                break;
            }
            Err(error) => return Err(error),
        }
    }

    let richest = &ladder[ladder.len() - 1].fit;
    let overdispersion = overdispersion(open_counts, richest);
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
    if let Some((bin, count)) = counts
        .iter()
        .enumerate()
        .find(|(_, c)| !(c.is_finite() && **c >= 0.0 && c.fract() == 0.0))
    {
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

const COUNTS_TO_MEASURE_NOISE: f64 = 1.0;

pub(crate) fn overdispersion(observed: &[f64], fit: &GridFit) -> Option<f64> {
    let (pearson, bins) = observed
        .iter()
        .zip(&fit.predicted)
        .filter(|(_, mu)| **mu >= COUNTS_TO_MEASURE_NOISE)
        .fold((0.0, 0_usize), |(sum, bins), (y, mu)| {
            (sum + (y - mu).powi(2) / mu, bins + 1)
        });
    let parameters = fit.result.on_bound.iter().filter(|&&on| !on).count();
    let freedom = bins.checked_sub(parameters).filter(|&f| f > 0)?;
    fit.converged
        .then(|| (pearson / freedom as f64).clamp(1.0, f64::INFINITY))
}

struct Candidate {
    beam: BeamSpline,
    fit: GridFit,
}

fn fit_beam(
    first_grid: &FlightTimeGrid,
    start: &BeamSpline,
    open_counts: &[f64],
) -> Result<Candidate, PipelineError> {
    let mut parameters = ParameterSet::new(
        start
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| FitParameter::unbounded(format!("beam {i}"), c))
            .collect(),
    );
    let fit = fit_on_halved_grids(first_grid, &mut parameters, open_counts, |grid| {
        Ok(OpenBeamModel::new(grid, start))
    })?;
    Ok(Candidate {
        beam: start.with_coefficients(&fit.result.params),
        fit,
    })
}

pub(crate) struct GridFit {
    pub(crate) result: PoissonResult,
    pub(crate) converged: bool,
    pub(crate) predicted: Vec<f64>,
    pub(crate) step_us: f64,
    pub(crate) points: usize,
    pub(crate) halvings: usize,
}

pub(crate) fn fit_on_halved_grids<M: FitModel>(
    first_grid: &FlightTimeGrid,
    parameters: &mut ParameterSet,
    observed: &[f64],
    model_on: impl Fn(&FlightTimeGrid) -> Result<M, PipelineError>,
) -> Result<GridFit, PipelineError> {
    let mut grid = first_grid.clone();
    let mut coarse = model_on(&grid)?;
    let mut halvings = 0;
    loop {
        let finer = grid.halved()?;
        let fine = model_on(&finer)?;
        let result = poisson_fit(&fine, observed, parameters, &PoissonConfig::default())?;
        let converged = result.converged && result.params.iter().all(|p| p.is_finite());
        let predicted = fine.evaluate(&result.params)?;
        let spread: f64 = predicted
            .iter()
            .zip(coarse.evaluate(&result.params)?)
            .filter(|(fine, _)| **fine > 0.0)
            .map(|(fine, coarse)| (fine - coarse).powi(2) / fine)
            .sum();
        grid = finer;
        coarse = fine;
        halvings += 1;
        if spread <= BOUND || !converged {
            return Ok(GridFit {
                result,
                converged,
                predicted,
                step_us: grid.step_us(),
                points: grid.flight_times_us().len(),
                halvings,
            });
        }
    }
}

pub(crate) struct OpenBeamModel {
    grid: FlightTimeGrid,
    basis: Vec<[(usize, f64); 5]>,
}

impl OpenBeamModel {
    pub(crate) fn new(grid: &FlightTimeGrid, beam: &BeamSpline) -> Self {
        Self {
            grid: grid.clone(),
            basis: grid
                .flight_times_us()
                .iter()
                .map(|&u| beam.basis(u))
                .collect(),
        }
    }

    pub(crate) fn beam(&self, coefficients: &[f64]) -> Vec<f64> {
        self.basis
            .iter()
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|&(i, w)| w * coefficients[i])
                    .sum::<f64>()
                    .exp()
            })
            .collect()
    }

    pub(crate) fn log_slope(&self, index: usize) -> Vec<f64> {
        self.basis
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
    use nereids_physics::ikeda_carpenter::{EnergyLaw, IkedaCarpenterParams, SynthesisGrid};

    use super::*;

    pub(crate) fn grid(channel_fwhm_us: Option<f64>) -> FlightTimeGrid {
        let pulse = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                beta: EnergyLaw::Const(0.25),
                r: EnergyLaw::Const(0.15),
                burst_sigma_us: None,
                channel_fwhm_us,
            },
            25.0,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 200.0,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model");
        let edges: Vec<f64> = (350..=470).map(f64::from).collect();
        FlightTimeGrid::new(&edges, 3.0, &Arc::new(pulse)).expect("grid")
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
