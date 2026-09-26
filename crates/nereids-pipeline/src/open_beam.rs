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
    /// richest fitted beam's Pearson χ² per degree of freedom, for independent
    /// bins.  Beam structure finer than every candidate is counted in it as
    /// noise; such a beam is not supported and is fitted smooth, which shows
    /// as an overdispersion far above the detector's.  `None` when the fit did
    /// not converge.
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
/// Counts too sparse to determine the beam are not supported: a beam fitted to
/// them can vary faster than a grid within the point cap resolves, which ends
/// the ladder or, for the first candidate, refuses the fit.
///
/// Beam structure at or near the window's first edge is not supported: the
/// counts are still matched, but before the edge, where neutrons reach the bins
/// only through the pulse's delay, the beam is the spline's continuation, which
/// the counts cannot check, and its error biases the fitted beam near that
/// edge and, through the continuation's mean curvature, elsewhere.
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
    if open_counts.len() + 1 != time_edges_us.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "{} open-beam counts for {} time edges",
            open_counts.len(),
            time_edges_us.len()
        )));
    }
    if let Some((bin, count)) = open_counts
        .iter()
        .enumerate()
        .find(|(_, c)| !(c.is_finite() && **c >= 0.0 && c.fract() == 0.0))
    {
        return Err(PipelineError::InvalidParameter(format!(
            "open-beam counts must be whole non-negative numbers, got {count} in bin {bin}"
        )));
    }
    if open_counts.iter().all(|&c| c == 0.0) {
        return Err(PipelineError::InvalidParameter(
            "the open beam has no counts".into(),
        ));
    }

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
        .filter(|fit| fit.converged && admits(2 * fit.beam.intervals()))
        .map(|fit| fit.beam.refined())
    {
        match fit_beam(&grid, &start, open_counts) {
            Ok(fit) if fit.converged => ladder.push(fit),
            Ok(_)
            | Err(PipelineError::FlightTimeGrid(FlightTimeGridError::TooManyPoints { .. })) => {
                break;
            }
            Err(error) => return Err(error),
        }
    }

    let richest = &ladder[ladder.len() - 1];
    let overdispersion = richest.converged.then(|| {
        let pearson: f64 = open_counts
            .iter()
            .zip(&richest.predicted)
            .filter(|(_, mu)| **mu > 0.0)
            .map(|(y, mu)| (y - mu).powi(2) / mu)
            .sum();
        let freedom = open_counts.len() - richest.beam.coefficients().len();
        (pearson / freedom as f64).clamp(1.0, f64::INFINITY)
    });
    let scale = overdispersion.unwrap_or(1.0);
    let criterion = |fit: &Candidate| {
        2.0 * fit.result.deviance / scale + 2.0 * fit.beam.coefficients().len() as f64
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
    let fit = ladder.swap_remove(chosen);
    Ok(OpenBeamFit {
        beam: fit.beam,
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

struct Candidate {
    beam: BeamSpline,
    converged: bool,
    result: PoissonResult,
    predicted: Vec<f64>,
    step_us: f64,
    points: usize,
    halvings: usize,
}

fn fit_beam(
    first_grid: &FlightTimeGrid,
    start: &BeamSpline,
    open_counts: &[f64],
) -> Result<Candidate, PipelineError> {
    let mut grid = first_grid.clone();
    let mut beam = start.clone();
    let mut halvings = 0;
    loop {
        let finer = grid.halved()?;
        let (refit, result) = fit(&finer, &beam, open_counts)?;
        let converged = result.converged && refit.coefficients().iter().all(|c| c.is_finite());
        let predicted = counts(&finer, &refit)?;
        let spread: f64 = predicted
            .iter()
            .zip(counts(&grid, &refit)?)
            .filter(|(fine, _)| **fine > 0.0)
            .map(|(fine, coarse)| (fine - coarse).powi(2) / fine)
            .sum();
        grid = finer;
        beam = refit;
        halvings += 1;
        if spread <= BOUND || !converged {
            return Ok(Candidate {
                beam,
                converged,
                result,
                predicted,
                step_us: grid.step_us(),
                points: grid.flight_times_us().len(),
                halvings,
            });
        }
    }
}

fn counts(grid: &FlightTimeGrid, beam: &BeamSpline) -> Result<Vec<f64>, FlightTimeGridError> {
    let values: Vec<f64> = grid
        .flight_times_us()
        .iter()
        .map(|&u| beam.per_us(u))
        .collect();
    grid.predict(&values)
}

fn fit(
    grid: &FlightTimeGrid,
    start: &BeamSpline,
    open_counts: &[f64],
) -> Result<(BeamSpline, PoissonResult), PipelineError> {
    let model = OpenBeamModel::new(grid, start);
    let mut params = ParameterSet::new(
        start
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| FitParameter::unbounded(format!("beam {i}"), c))
            .collect(),
    );
    let result = poisson_fit(&model, open_counts, &mut params, &PoissonConfig::default())?;
    Ok((start.with_coefficients(&result.params), result))
}

struct OpenBeamModel<'a> {
    grid: &'a FlightTimeGrid,
    basis: Vec<[(usize, f64); 5]>,
}

impl<'a> OpenBeamModel<'a> {
    fn new(grid: &'a FlightTimeGrid, beam: &BeamSpline) -> Self {
        Self {
            grid,
            basis: grid
                .flight_times_us()
                .iter()
                .map(|&u| beam.basis(u))
                .collect(),
        }
    }

    fn beam(&self, coefficients: &[f64]) -> Vec<f64> {
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

    fn counts(&self, values: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.grid
            .predict(values)
            .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
    }
}

impl FitModel for OpenBeamModel<'_> {
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
                .basis
                .iter()
                .zip(&beam)
                .map(|(pairs, &phi)| {
                    phi * pairs
                        .iter()
                        .filter(|&&(i, _)| i == index)
                        .map(|&(_, w)| w)
                        .sum::<f64>()
                })
                .collect();
            for (row, value) in self.counts(&values).ok()?.into_iter().enumerate() {
                *jacobian.get_mut(row, col) = value;
            }
        }
        Some(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use nereids_physics::ikeda_carpenter::{EnergyLaw, IkedaCarpenterParams, SynthesisGrid};

    use super::*;

    fn grid(channel_fwhm_us: Option<f64>) -> FlightTimeGrid {
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
