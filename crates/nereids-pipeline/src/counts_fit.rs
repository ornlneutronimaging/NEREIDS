//! Areal densities of a sample's isotopes, fitted to the counts of an
//! open-beam run and a sample run recorded in the same time bins.

use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::error::FittingError;
use nereids_fitting::lm::{FitModel, FlatMatrix};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_physics::continuous_doppler::SUPPORT_X;
use nereids_physics::doppler::DopplerParams;
use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::{broadened_cross_sections, resonance_center_energies};

use crate::beam::BeamSpline;
use crate::error::PipelineError;
use crate::open_beam::{
    Calibration, OpenBeamModel, fit_on_halved_grids, fit_open_beam, overdispersion, validate_counts,
};
use crate::pipeline::TEMPERATURE_BOUNDS_K;

/// A count in a bin predicted fewer counts than this has a chance below it
/// under the model.
pub const NEGLIGIBLE_PREDICTION: f64 = 1e-10;

/// An open-beam run and a sample run recorded in the same time bins.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Time-bin edges in µs.
    pub time_edges_us: Vec<f64>,
    /// Raw counts of the open-beam run, one per bin.
    pub open_counts: Vec<f64>,
    /// Raw counts of the sample run, one per bin.
    pub sample_counts: Vec<f64>,
    /// The sample run's beam over the open-beam run's, the ratio of their
    /// proton charges.
    pub charge_ratio: f64,
    /// Each isotope in the sample with the areal density, in atoms/barn, the
    /// fit starts from.
    pub isotopes: Vec<(ResonanceData, f64)>,
    /// The sample's temperature in K.
    pub temperature_k: f64,
}

/// The fitted densities.
#[derive(Debug, Clone)]
pub struct CountsFit {
    /// Areal density of each isotope in atoms/barn, in the order given.
    pub densities: Vec<f64>,
    /// Covariance of the densities, in the order given: the inverse of the
    /// expected information at the fit, which holds when the counts are
    /// large, scaled by `overdispersion`, or at the Poisson scale when that is
    /// `None`.  The row and column of a density on its bound of 0 or not
    /// determined are NaN.  `None` when the fit did not converge.
    pub covariance: Option<FlatMatrix>,
    /// The beam per µs of flight time, fitted to both runs, with the
    /// intervals the open-beam fit chose.
    pub beam: BeamSpline,
    /// Whether the open-beam fit chose its richest beam; see
    /// [`OpenBeamFit::at_limit`](crate::open_beam::OpenBeamFit::at_limit).
    pub beam_at_limit: bool,
    /// Half the Poisson deviance of both runs at the fit.
    pub deviance: f64,
    /// Whether the fitter converged.
    pub converged: bool,
    /// Variance of the counts of both runs over their Poisson variance, at
    /// least 1, measured on the bins predicted at least one count.  `None`
    /// when the fit did not converge or those bins do not outnumber the
    /// fitted parameters.
    pub overdispersion: Option<f64>,
    /// Step, in µs, of the fit's grid.
    pub step_us: f64,
    /// Number of points of that grid.
    pub points: usize,
    /// How many times the grid of
    /// [`FlightTimeGrid::new`](nereids_physics::flight_time_grid::FlightTimeGrid::new)
    /// was halved to reach it.
    pub halvings: usize,
}

/// Fit the isotopes' areal densities, at the known temperature, to the raw
/// counts of both runs of `measurement`.  The open-beam counts are
/// `O_k = Σ_j w φ(u_j) P_k(u_j)` and the sample counts
/// `S_k = c Σ_j w φ(u_j) T(E_j) P_k(u_j)` on a uniform flight-time grid,
/// with `T = exp(−Σ_i n_i σ_i)`, `σ_i` the isotope's Doppler-broadened total
/// cross section, `c` the charge ratio and `P_k` the chance of a neutron being
/// counted in bin `k`.  The beam `φ` has the intervals [`fit_open_beam`]
/// chooses and is fitted with the densities to both runs, starting from the
/// open-beam fit.  There is no background term: background counts bias the
/// densities, and are refused only where the model predicts none.
///
/// The grid's first step is at most half the narrowest Doppler full width at
/// half maximum, in flight time, of any resonance inside its energy span; it
/// is then halved until the counts of both runs meet
/// [`BOUND`](crate::open_beam::BOUND).  The step is uniform, so a wide window
/// whose span holds a narrow resonance at high energy can exceed the grid's
/// point cap.
///
/// The overdispersion scales the covariance; it assumes both runs share it
/// and their bins are independent.
///
/// # Errors
/// [`PipelineError::ShapeMismatch`] unless each run has one count per bin;
/// [`PipelineError::InvalidParameter`] if a count is not a whole non-negative
/// number, a run has no counts, the charge ratio is not finite and positive,
/// there are no isotopes, an isotope is listed twice, a starting density is
/// not finite and non-negative, the temperature is outside 1–5000 K, an
/// isotope's resonance data are not finite, or the energies its broadened
/// cross section reads, down to zero for a window within the thermal spread
/// of zero energy, are not inside a single one of its evaluated (SLBW, MLBW or
/// Reich–Moore) resolved ranges;
/// [`PipelineError::UnmodelledCounts`] if at the fit, converged or not, a bin
/// holds counts predicted below [`NEGLIGIBLE_PREDICTION`]: background, or
/// starting densities whose transmission vanishes where counts were recorded,
/// which the fitter cannot leave;
/// everything [`fit_open_beam`] refuses; [`PipelineError::FlightTimeGrid`]
/// for the grid's refusals, including more points than it allows;
/// [`PipelineError::Transmission`] and [`PipelineError::Fitting`] if the
/// cross sections or the fitter fail.
pub fn fit_counts(
    measurement: &Measurement,
    calibration: &Calibration,
) -> Result<CountsFit, PipelineError> {
    let Measurement {
        time_edges_us,
        open_counts,
        sample_counts,
        charge_ratio,
        isotopes,
        temperature_k,
    } = measurement;
    let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
    if !(charge_ratio.is_finite() && *charge_ratio > 0.0) {
        return invalid(format!(
            "the charge ratio must be finite and positive, got {charge_ratio}"
        ));
    }
    if isotopes.is_empty() {
        return invalid("the sample has no isotopes".into());
    }
    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    if !(t_low..=t_high).contains(temperature_k) {
        return invalid(format!(
            "the temperature must be within {t_low}–{t_high} K, got {temperature_k}"
        ));
    }
    let mut dopplers = Vec::with_capacity(isotopes.len());
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
        if !(density.is_finite() && *density >= 0.0) {
            return invalid(format!(
                "the starting density of {} must be finite and non-negative, got {density}",
                isotope.isotope
            ));
        }
        if !finite(isotope) {
            return invalid(format!(
                "the resonance data of {} are not finite",
                isotope.isotope
            ));
        }
        dopplers.push(
            DopplerParams::new(*temperature_k, isotope.awr)
                .map_err(|e| PipelineError::InvalidParameter(e.to_string()))?,
        );
    }
    let grid = FlightTimeGrid::new(time_edges_us, calibration.t0_us, &calibration.pulse)?;
    let bins = time_edges_us.len() - 1;
    validate_counts("open-beam", open_counts, bins)?;
    validate_counts("sample", sample_counts, bins)?;

    let energies = grid.energies_ev();
    let span_ev = (energies[energies.len() - 1], energies[0]);
    let clock = TOF_FACTOR * calibration.pulse.flight_path_m();
    let half_maximum = 2.0 * std::f64::consts::LN_2.sqrt();
    let mut narrowest_us = f64::INFINITY;
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
            return invalid(format!(
                "the Doppler-broadened cross section of {} reads {:.6e}–{:.6e} eV, which no \
                 single one of its evaluated (SLBW, MLBW or Reich–Moore) resolved ranges holds",
                isotope.isotope, read.0, read.1
            ));
        }
        for energy in resonance_center_energies(&[isotope])
            .into_iter()
            .filter(|e| (span_ev.0..=span_ev.1).contains(e))
        {
            let width_ev = half_maximum * doppler.doppler_width(energy);
            narrowest_us = narrowest_us.min(clock / energy.sqrt() * width_ev / (2.0 * energy));
        }
    }
    let mut first = grid;
    let mut halvings = 0;
    while first.step_us() > 0.5 * narrowest_us {
        first = first.halved()?;
        halvings += 1;
    }

    let open = fit_open_beam(time_edges_us, open_counts, calibration)?;
    let beam_coefficients = open.beam.coefficients().len();
    let mut parameters = ParameterSet::new(
        open.beam
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| FitParameter::unbounded(format!("beam {i}"), c))
            .chain(
                isotopes
                    .iter()
                    .enumerate()
                    .map(|(i, (_, n))| FitParameter::non_negative(format!("density {i}"), *n)),
            )
            .collect(),
    );
    let resonances: Vec<ResonanceData> = isotopes.iter().map(|(data, _)| data.clone()).collect();
    let observed: Vec<f64> = open_counts.iter().chain(sample_counts).copied().collect();
    let fit = fit_on_halved_grids(&Arc::new(first), &mut parameters, &observed, |grid| {
        TwoRunModel::new(grid, &open.beam, &resonances, *temperature_k, *charge_ratio)
    })?;

    if let Some((k, (&counts, &predicted))) = observed
        .iter()
        .zip(&fit.predicted)
        .enumerate()
        .find(|(_, (y, mu))| **y > 0.0 && **mu < NEGLIGIBLE_PREDICTION)
    {
        let run = if k < bins { "open-beam" } else { "sample" };
        return Err(PipelineError::UnmodelledCounts {
            run,
            bin: k % bins,
            counts,
            predicted,
        });
    }

    let overdispersion = overdispersion(&observed, &fit);
    let scale = overdispersion.unwrap_or(1.0);
    let densities = fit.result.params[beam_coefficients..].to_vec();
    let covariance = fit
        .result
        .covariance
        .as_ref()
        .filter(|_| fit.converged)
        .map(|full| {
            let mut block = FlatMatrix::zeros(densities.len(), densities.len());
            for i in 0..densities.len() {
                for j in 0..densities.len() {
                    *block.get_mut(i, j) =
                        scale * full.get(beam_coefficients + i, beam_coefficients + j);
                }
            }
            block
        });
    Ok(CountsFit {
        densities,
        covariance,
        beam: open
            .beam
            .with_coefficients(&fit.result.params[..beam_coefficients]),
        beam_at_limit: open.at_limit,
        deviance: fit.result.deviance,
        converged: fit.converged,
        overdispersion,
        step_us: fit.step_us,
        points: fit.points,
        halvings: halvings + fit.halvings,
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

struct TwoRunModel {
    beam: OpenBeamModel,
    cross_sections: Vec<Vec<f64>>,
    charge_ratio: f64,
}

impl TwoRunModel {
    fn new(
        grid: &Arc<FlightTimeGrid>,
        beam: &BeamSpline,
        isotopes: &[ResonanceData],
        temperature_k: f64,
        charge_ratio: f64,
    ) -> Result<Self, PipelineError> {
        let mut energies = grid.energies_ev();
        energies.reverse();
        let mut cross_sections =
            broadened_cross_sections(&energies, isotopes, temperature_k, None, None)?;
        for sigma in &mut cross_sections {
            sigma.reverse();
        }
        Ok(Self {
            beam: OpenBeamModel::new(grid, beam),
            cross_sections,
            charge_ratio,
        })
    }

    fn beams(&self, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (coefficients, densities) = params.split_at(params.len() - self.cross_sections.len());
        let open = self.beam.beam(coefficients);
        let sample = open
            .iter()
            .enumerate()
            .map(|(j, phi)| {
                let depth: f64 = densities
                    .iter()
                    .zip(&self.cross_sections)
                    .map(|(n, sigma)| n * sigma[j])
                    .sum();
                self.charge_ratio * phi * (-depth).exp()
            })
            .collect();
        (open, sample)
    }
}

impl FitModel for TwoRunModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let (open, sample) = self.beams(params);
        let mut counts = self.beam.counts(&open)?;
        counts.extend(self.beam.counts(&sample)?);
        Ok(counts)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let (open, sample) = self.beams(params);
        let beam_coefficients = params.len() - self.cross_sections.len();
        let mut jacobian = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        for (col, &index) in free_param_indices.iter().enumerate() {
            let (open_slope, sample_slope) = if index < beam_coefficients {
                let slope = self.beam.log_slope(index);
                (slope.clone(), slope)
            } else {
                let sigma = &self.cross_sections[index - beam_coefficients];
                (vec![0.0; open.len()], sigma.iter().map(|s| -s).collect())
            };
            let times = |beam: &[f64], slope: &[f64]| -> Vec<f64> {
                beam.iter().zip(slope).map(|(b, s)| b * s).collect()
            };
            let open_column = self.beam.counts(&times(&open, &open_slope)).ok()?;
            let sample_column = self.beam.counts(&times(&sample, &sample_slope)).ok()?;
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
    use crate::open_beam::tests::grid;

    #[test]
    fn the_jacobian_is_the_slope_of_both_runs_counts() {
        let grid = grid(None);
        let (_, u_hi) = grid.range_us();
        let beam = BeamSpline::constant(347.0, u_hi, 1.0e4).refined().refined();
        let isotopes = [
            synthetic_isotope(72, 180, 20.0, 0.01, 0.06),
            synthetic_isotope(74, 182, 20.3, 0.01, 0.06),
        ];
        let model = TwoRunModel::new(&grid, &beam, &isotopes, 300.0, 1.2).expect("model");
        let params: Vec<f64> = (0..beam.coefficients().len())
            .map(|i| 9.0 + 0.3 * (i as f64).sin())
            .chain([3.0e-4, 5.0e-4])
            .collect();
        let counts = model.evaluate(&params).expect("counts");
        let indices: Vec<usize> = (0..params.len()).collect();
        let jacobian = model
            .analytical_jacobian(&params, &indices, &counts)
            .expect("jacobian");
        for index in indices {
            let h = 1e-4 * params[index].abs();
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
}
