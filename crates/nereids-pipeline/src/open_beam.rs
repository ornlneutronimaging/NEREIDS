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
    /// The beam per µs of flight time, on the accepted grid's flight-time
    /// range.
    pub beam: BeamSpline,
    /// Half the Poisson deviance at the fit.
    pub deviance: f64,
    /// Whether the fitter converged.  It certifies the minimisation, not that
    /// the counts determine every coefficient; `covariance` shows that.
    pub converged: bool,
    /// Covariance of the beam's coefficients; rows and columns of a
    /// coefficient the counts leave undetermined are NaN.  `None` when the fit
    /// did not converge.
    pub covariance: Option<FlatMatrix>,
    /// Step, in µs, of the accepted grid.
    pub step_us: f64,
    /// Number of points of the accepted grid.
    pub points: usize,
    /// How many times the first grid was halved to reach the accepted one.
    pub halvings: usize,
}

/// Fit the beam to the raw open-beam counts `open_counts` of the time bins
/// `time_edges_us` (µs).  The beam is one cubic in `ln u` over the flight-time
/// grid's range; neutrons faster than that range, each reaching the bins with
/// less than [`NEGLIGIBLE_ARRIVAL_PROBABILITY`](nereids_physics::ikeda_carpenter::NEGLIGIBLE_ARRIVAL_PROBABILITY)
/// chance, are left out.  The grid is halved, and the beam refitted on each
/// finer grid, until the refitted beam's predicted counts differ from the
/// coarser grid's by at most [`BOUND`]; that fit is returned.
///
/// # Errors
/// [`PipelineError::ShapeMismatch`] unless there is one count per bin;
/// [`PipelineError::InvalidParameter`] if a count is not a whole non-negative
/// number, or every count is zero; [`PipelineError::FlightTimeGrid`] for the
/// grid's refusals, including a halving past its point cap;
/// [`PipelineError::Fitting`] if the fitter refuses.
pub fn fit_open_beam(
    time_edges_us: &[f64],
    open_counts: &[f64],
    calibration: &Calibration,
) -> Result<OpenBeamFit, PipelineError> {
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

    let mut grid = FlightTimeGrid::new(time_edges_us, calibration.t0_us, &calibration.pulse)?;
    let (u_lo, u_hi) = grid.range_us();
    let per_unit_beam = grid.predict(&vec![1.0; grid.flight_times_us().len()])?;
    let per_us = open_counts.iter().sum::<f64>() / per_unit_beam.iter().sum::<f64>();
    let mut beam = BeamSpline::constant(u_lo, u_hi, per_us);
    let mut halvings = 0;
    loop {
        let finer = grid.halved()?;
        let (refit, result) = fit(&finer, &beam, open_counts)?;
        let spread: f64 = counts(&finer, &refit)?
            .iter()
            .zip(counts(&grid, &refit)?)
            .filter(|(fine, _)| **fine > 0.0)
            .map(|(fine, coarse)| (fine - coarse).powi(2) / fine)
            .sum();
        grid = finer;
        beam = refit;
        halvings += 1;
        if spread <= BOUND {
            return Ok(OpenBeamFit {
                beam,
                deviance: result.deviance,
                converged: result.converged,
                covariance: result.covariance,
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
    Ok((
        start.with_coefficients(std::array::from_fn(|i| result.params[i])),
        result,
    ))
}

struct OpenBeamModel<'a> {
    grid: &'a FlightTimeGrid,
    basis: Vec<[f64; 4]>,
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
            .map(|b| {
                b.iter()
                    .zip(coefficients)
                    .map(|(w, c)| w * c)
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
                .map(|(b, &phi)| phi * b[index])
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
            let (u_lo, u_hi) = grid.range_us();
            let beam = BeamSpline::constant(u_lo, u_hi, 1.0e4);
            let model = OpenBeamModel::new(&grid, &beam);
            let coefficients = [9.1, 9.4, 9.0, 8.7];
            let counts = model.evaluate(&coefficients).expect("counts");
            let jacobian = model
                .analytical_jacobian(&coefficients, &[0, 1, 2, 3], &counts)
                .expect("jacobian");
            for index in 0..4 {
                let h = 1e-4;
                let shifted = |d: f64| {
                    let mut c = coefficients;
                    c[index] += d;
                    model.evaluate(&c).expect("counts")
                };
                let (up, down) = (shifted(h), shifted(-h));
                for (row, (u, d)) in up.iter().zip(&down).enumerate() {
                    let slope = (u - d) / (2.0 * h);
                    let analytic = jacobian.get(row, index);
                    assert!(
                        (analytic - slope).abs() <= 1e-6 * slope.abs(),
                        "{channel_fwhm_us:?} {index} {row}: {analytic} vs {slope}"
                    );
                }
            }
        }
    }
}
