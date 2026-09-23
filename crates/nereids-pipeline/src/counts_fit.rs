use nereids_endf::resonance::ResonanceData;
use nereids_fitting::beam::BeamSpline;
use nereids_fitting::lm::FitModel;
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{PoissonConfig, poisson_fit};
use nereids_fitting::two_run::{OpenBeamModel, Points, TwoRunModel};
use nereids_physics::auxiliary_grid::with_resonance_points;
use nereids_physics::bin_weights::{BinWeights, BinWeightsError};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_physics::transmission::extract_resonance_widths;

use crate::error::PipelineError;
use crate::pipeline::TEMPERATURE_BOUNDS_K;

/// Largest change of the predicted counts, in counting noise over all bins
/// of both runs together, that the calculation points may leave: when
/// doubling them, and from the resonances given no points of their own.
pub const ACCURACY: f64 = 0.1;

const MAX_DOUBLINGS: usize = 8;

/// A quantity the fit determines from `Fitted`'s starting value, or holds at
/// its `Known` value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Known(f64),
    Fitted(f64),
}

impl Value {
    fn parameter(self, name: String, (lower, upper): (f64, f64)) -> FitParameter {
        match self {
            Self::Known(v) => FitParameter::fixed(name, v),
            Self::Fitted(v) => FitParameter {
                name: name.into(),
                value: v,
                lower,
                upper,
                fixed: false,
            },
        }
    }

    fn start(self) -> f64 {
        match self {
            Self::Known(v) | Self::Fitted(v) => v,
        }
    }
}

/// An open-beam run and a sample run recorded in the same time bins, each
/// counting neutrons alone: the fit has no background term.
pub struct Measurement {
    /// Time-bin edges in µs, strictly ascending.
    pub time_edges_us: Vec<f64>,
    pub open_counts: Vec<f64>,
    pub sample_counts: Vec<f64>,
    /// The sample run's beam relative to the open-beam run's, the ratio of
    /// their proton charges.
    pub charge_ratio: f64,
    /// Each isotope in the sample with its areal density in atoms/barn.
    pub isotopes: Vec<(ResonanceData, Value)>,
    pub temperature_k: Value,
}

/// The energy scale and the resolution: a neutron of energy `E` arrives at
/// `t0_us + TOF_FACTOR · flight_path_m / √E` plus an offset drawn from
/// `resolution`.
pub struct Calibration {
    pub flight_path_m: f64,
    pub t0_us: f64,
    pub resolution: ResolutionFunction,
}

#[derive(Debug, Clone)]
pub struct CountsFit {
    /// Areal density of each isotope in atoms/barn, in the order given.
    pub densities: Vec<f64>,
    /// Standard error of each fitted density; `None` for a known one or
    /// when the fit has no covariance.
    pub density_uncertainties: Vec<Option<f64>>,
    pub temperature_k: f64,
    pub temperature_uncertainty: Option<f64>,
    pub beam: BeamSpline,
    /// Poisson deviance of both runs.
    pub deviance: f64,
    /// Bins of both runs minus fitted parameters.
    pub degrees_of_freedom: usize,
    pub converged: bool,
    /// Number of calculation points.
    pub points: usize,
    /// The change of the predicted counts, in counting noise, each time the
    /// points were doubled; the last is at most [`ACCURACY`].
    pub accuracy: Vec<f64>,
    /// The counts, in counting noise, from the energies whose resonances
    /// were given no points, even with no sample in the beam; at most
    /// [`ACCURACY`].
    pub skipped: f64,
}

/// Fit `measurement` on `calibration`.
///
/// # Errors
/// [`PipelineError::InvalidParameter`] for invalid inputs,
/// [`PipelineError::ShapeMismatch`] when a run's counts do not match the time
/// bins, [`PipelineError::BinWeights`] when the resolution rejects an energy
/// (a Gaussian resolution always does),
/// [`PipelineError::Fitting`] when the model cannot be evaluated, and
/// [`PipelineError::PointsNotConverged`] when doubling the points does not
/// reach [`ACCURACY`].
pub fn fit_counts(
    measurement: &Measurement,
    calibration: &Calibration,
) -> Result<CountsFit, PipelineError> {
    validate(measurement, calibration)?;
    let resolution = calibration
        .resolution
        .with_flight_path(calibration.flight_path_m)
        .map_err(|e| PipelineError::InvalidParameter(e.to_string()))?;
    let edges = &measurement.time_edges_us;
    let t0 = calibration.t0_us;
    let kl = TOF_FACTOR * calibration.flight_path_m;
    let base = reach(edges, t0, &resolution)?;
    let weights_at = |grid: &[f64]| BinWeights::new(grid, edges, t0, &resolution);

    let base_weights = weights_at(&base)?;
    let beam = fit_beam(
        &base,
        &base_weights,
        (edges[0] - t0, edges[edges.len() - 1] - t0),
        &measurement.open_counts,
    )?;

    let isotopes: Vec<ResonanceData> = measurement.isotopes.iter().map(|i| i.0.clone()).collect();
    let refs: Vec<&ResonanceData> = isotopes.iter().collect();
    let resonances = extract_resonance_widths(&refs);
    let noise: Vec<f64> = measurement
        .sample_counts
        .iter()
        .map(|&c| c.max(1.0))
        .collect();
    let phi = beam_at(&base, &beam, kl);
    let mut kept_range =
        kept_energies(&base, &base_weights, &phi, measurement.charge_ratio, &noise);
    let mut grid = with_resonance_points(&base, &kept(&resonances, kept_range));
    let mut weights = weights_at(&grid)?;

    let mut params =
        ParameterSet::new(
            beam.coefficients()
                .iter()
                .enumerate()
                .map(|(i, &c)| FitParameter::unbounded(format!("beam {i}"), c))
                .chain(
                    measurement.isotopes.iter().enumerate().map(|(i, (_, n))| {
                        n.parameter(format!("density {i}"), (0.0, f64::INFINITY))
                    }),
                )
                .chain(std::iter::once(
                    measurement
                        .temperature_k
                        .parameter("temperature".into(), TEMPERATURE_BOUNDS_K),
                ))
                .collect(),
        );
    let observed: Vec<f64> = measurement
        .open_counts
        .iter()
        .chain(&measurement.sample_counts)
        .copied()
        .collect();
    let n_bins = edges.len() - 1;
    let mut accuracy = Vec::new();
    let (result, predicted, skipped) = loop {
        let points = Points::new(&grid, &weights, &beam);
        let model = TwoRunModel::new(&points, &isotopes, measurement.charge_ratio);
        let result = poisson_fit(&model, &observed, &mut params, &PoissonConfig::default())?;
        let predicted = model.evaluate(&result.params)?;
        let sample = &predicted[n_bins..];

        let fitted = beam.with_coefficients(&result.params[..beam.coefficients().len()]);
        let phi = beam_at(&grid, &fitted, kl);
        let c = measurement.charge_ratio;
        let skipped = skipped_counts(&grid, &weights, &phi, c, sample, kept_range);
        if skipped > ACCURACY {
            let fresh = kept_energies(&grid, &weights, &phi, c, sample);
            let wider = (kept_range.0.min(fresh.0), kept_range.1.max(fresh.1));
            let newly_kept: Vec<(f64, f64)> = kept(&resonances, wider)
                .into_iter()
                .filter(|&(e, _)| e < kept_range.0 || e > kept_range.1)
                .collect();
            kept_range = wider;
            grid = with_resonance_points(&grid, &newly_kept);
            weights = weights_at(&grid)?;
            continue;
        }

        let finer = doubled(&grid, kept_range, kl);
        let finer_weights = weights_at(&finer)?;
        let finer_points = Points::new(&finer, &finer_weights, &beam);
        let finer_model = TwoRunModel::new(&finer_points, &isotopes, measurement.charge_ratio);
        let change = noise_distance(&predicted, &finer_model.evaluate(&result.params)?);
        accuracy.push(change);
        if change <= ACCURACY {
            break (result, predicted, skipped);
        }
        if accuracy.len() == MAX_DOUBLINGS {
            return Err(PipelineError::PointsNotConverged(MAX_DOUBLINGS));
        }
        grid = finer;
        weights = finer_weights;
    };

    let free = params.free_indices();
    let uncertainty = |index: usize| {
        let column = free.iter().position(|&f| f == index)?;
        result.uncertainties.as_ref().map(|u| u[column])
    };
    let n_beam = beam.coefficients().len();
    let n_isotopes = isotopes.len();
    Ok(CountsFit {
        densities: result.params[n_beam..n_beam + n_isotopes].to_vec(),
        density_uncertainties: (n_beam..n_beam + n_isotopes).map(uncertainty).collect(),
        temperature_k: result.params[n_beam + n_isotopes],
        temperature_uncertainty: uncertainty(n_beam + n_isotopes),
        beam: beam.with_coefficients(&result.params[..n_beam]),
        deviance: deviance(&observed, &predicted),
        degrees_of_freedom: observed.len().saturating_sub(free.len()),
        converged: result.converged,
        points: grid.len(),
        accuracy,
        skipped,
    })
}

fn validate(measurement: &Measurement, calibration: &Calibration) -> Result<(), PipelineError> {
    let invalid = |msg: String| Err(PipelineError::InvalidParameter(msg));
    let edges = &measurement.time_edges_us;
    if edges.len() < 2
        || edges.iter().any(|t| !t.is_finite())
        || edges.windows(2).any(|w| w[0] >= w[1])
    {
        return invalid(
            "the time edges must be at least two, finite and strictly ascending".into(),
        );
    }
    if !(calibration.t0_us.is_finite() && calibration.t0_us < edges[0]) {
        return invalid(format!(
            "t0 must be finite and before the first time edge, got {}",
            calibration.t0_us
        ));
    }
    let n_bins = edges.len() - 1;
    for (name, counts) in [
        ("open", &measurement.open_counts),
        ("sample", &measurement.sample_counts),
    ] {
        if counts.len() != n_bins {
            return Err(PipelineError::ShapeMismatch(format!(
                "{} {name} counts for {n_bins} time bins",
                counts.len()
            )));
        }
        if counts.iter().any(|c| !(c.is_finite() && *c >= 0.0)) {
            return invalid(format!("the {name} counts must be finite and non-negative"));
        }
    }
    if !(measurement.charge_ratio.is_finite() && measurement.charge_ratio > 0.0) {
        return invalid(format!(
            "the charge ratio must be finite and positive, got {}",
            measurement.charge_ratio
        ));
    }
    let values = measurement
        .isotopes
        .iter()
        .map(|i| i.1)
        .chain(std::iter::once(measurement.temperature_k));
    if values
        .map(Value::start)
        .any(|v| !(v.is_finite() && v >= 0.0))
    {
        return invalid("densities and the temperature must be finite and non-negative".into());
    }
    let isotopes: Vec<&ResonanceData> = measurement.isotopes.iter().map(|i| &i.0).collect();
    if let Some((energy, width)) = extract_resonance_widths(&isotopes)
        .into_iter()
        .find(|(_, width)| !width.is_finite())
    {
        return invalid(format!(
            "the resonance at {energy} eV has a width of {width}"
        ));
    }
    if let Value::Fitted(t) = measurement.temperature_k {
        let (low, high) = TEMPERATURE_BOUNDS_K;
        if !(low..=high).contains(&t) {
            return invalid(format!(
                "a fitted temperature must start within [{low}, {high}] K, got {t}"
            ));
        }
        let broadens = measurement.isotopes.iter().any(|(isotope, density)| {
            *density != Value::Known(0.0) && !extract_resonance_widths(&[isotope]).is_empty()
        });
        if !broadens {
            return invalid(
                "a fitted temperature needs an isotope in the sample with resolved resonances"
                    .into(),
            );
        }
    }
    Ok(())
}

fn reach(
    edges: &[f64],
    t0: f64,
    resolution: &ResolutionFunction,
) -> Result<Vec<f64>, PipelineError> {
    let kl = TOF_FACTOR * resolution.flight_path_m();
    let recorded = |u: f64| -> Result<bool, PipelineError> {
        let probabilities = resolution
            .detector_bin_probabilities((kl / u).powi(2), edges, t0)
            .map_err(BinWeightsError::from)?;
        Ok(probabilities.iter().any(|&p| p > 0.0))
    };
    let flight_times: Vec<f64> = edges.iter().map(|t| t - t0).collect();
    let n = flight_times.len();
    let mut earlier = Vec::new();
    let step = flight_times[1] - flight_times[0];
    let mut u = flight_times[0] - step;
    while u > 0.0 {
        earlier.push(u);
        if !recorded(u)? {
            break;
        }
        u -= step;
    }
    let mut later = Vec::new();
    let step = flight_times[n - 1] - flight_times[n - 2];
    let mut u = flight_times[n - 1] + step;
    loop {
        later.push(u);
        if !recorded(u)? {
            break;
        }
        u += step;
    }
    Ok(later
        .iter()
        .rev()
        .chain(flight_times.iter().rev())
        .chain(&earlier)
        .map(|u| (kl / u).powi(2))
        .collect())
}

fn fit_beam(
    grid: &[f64],
    weights: &BinWeights,
    window_us: (f64, f64),
    open_counts: &[f64],
) -> Result<BeamSpline, PipelineError> {
    let total_weight: f64 = weights.apply(&vec![1.0; grid.len()]).iter().sum();
    let per_us = open_counts.iter().sum::<f64>().max(1.0) / total_weight;
    let fit = |spline: &BeamSpline| -> Result<(BeamSpline, f64), PipelineError> {
        let points = Points::new(grid, weights, spline);
        let model = OpenBeamModel::new(&points);
        let mut params = ParameterSet::new(
            spline
                .coefficients()
                .iter()
                .enumerate()
                .map(|(i, &c)| FitParameter::unbounded(format!("beam {i}"), c))
                .collect(),
        );
        let config = PoissonConfig {
            compute_covariance: false,
            ..PoissonConfig::default()
        };
        let result = poisson_fit(&model, open_counts, &mut params, &config)?;
        let predicted = model.evaluate(&result.params)?;
        Ok((
            spline.with_coefficients(&result.params),
            deviance(open_counts, &predicted),
        ))
    };
    let criterion = |(spline, deviance): &(BeamSpline, f64)| {
        deviance + 2.0 * spline.coefficients().len() as f64
    };
    let within_noise = |(spline, deviance): &(BeamSpline, f64)| {
        *deviance
            <= open_counts
                .len()
                .saturating_sub(spline.coefficients().len()) as f64
    };
    let mut current = fit(&BeamSpline::constant(window_us.0, window_us.1, per_us))?;
    let mut best = current.clone();
    let mut since_best = 0;
    while !within_noise(&current)
        && since_best < 2
        && current.0.refined().coefficients().len() <= open_counts.len()
    {
        current = fit(&current.0.refined())?;
        since_best += 1;
        if criterion(&current) < criterion(&best) {
            best = current.clone();
            since_best = 0;
        }
    }
    Ok(best.0)
}

fn beam_at(grid: &[f64], beam: &BeamSpline, kl: f64) -> Vec<f64> {
    grid.iter().map(|&e| beam.per_us(kl / e.sqrt())).collect()
}

fn unattenuated(
    weights: &BinWeights,
    phi: &[f64],
    charge_ratio: f64,
    j: usize,
    counts: &mut [f64],
) {
    for (k, w) in weights.row(j) {
        counts[k] += charge_ratio * phi[j] * w;
    }
}

fn kept_energies(
    grid: &[f64],
    weights: &BinWeights,
    phi: &[f64],
    charge_ratio: f64,
    expected: &[f64],
) -> (f64, f64) {
    let mut outside = vec![0.0; weights.n_bins()];
    let mut grow = |j: usize| {
        let mut trial = outside.clone();
        unattenuated(weights, phi, charge_ratio, j, &mut trial);
        let fits = noise_norm(&trial, expected) <= ACCURACY;
        if fits {
            outside = trial;
        }
        fits
    };
    let n = grid.len();
    let high = (1..n).rev().find(|&j| !grow(j)).unwrap_or(0);
    let low = (0..high).find(|&j| !grow(j)).unwrap_or(high);
    (
        low.checked_sub(1).map_or(f64::NEG_INFINITY, |j| grid[j]),
        grid.get(high + 1).copied().unwrap_or(f64::INFINITY),
    )
}

fn skipped_counts(
    grid: &[f64],
    weights: &BinWeights,
    phi: &[f64],
    charge_ratio: f64,
    expected: &[f64],
    kept_range: (f64, f64),
) -> f64 {
    let mut counts = vec![0.0; weights.n_bins()];
    for (j, &e) in grid.iter().enumerate() {
        if e <= kept_range.0 || e >= kept_range.1 {
            unattenuated(weights, phi, charge_ratio, j, &mut counts);
        }
    }
    noise_norm(&counts, expected)
}

fn noise_norm(counts: &[f64], expected: &[f64]) -> f64 {
    counts
        .iter()
        .zip(expected)
        .map(|(c, n)| if *c == 0.0 { 0.0 } else { c * c / n })
        .sum::<f64>()
        .sqrt()
}

fn kept(resonances: &[(f64, f64)], (low, high): (f64, f64)) -> Vec<(f64, f64)> {
    resonances
        .iter()
        .copied()
        .filter(|&(e, _)| low <= e && e <= high)
        .collect()
}

fn doubled(grid: &[f64], (low, high): (f64, f64), kl: f64) -> Vec<f64> {
    let mut finer = Vec::with_capacity(2 * grid.len());
    for pair in grid.windows(2) {
        finer.push(pair[0]);
        if low <= pair[0] && pair[1] <= high {
            let u = 0.5 * (kl / pair[0].sqrt() + kl / pair[1].sqrt());
            finer.push((kl / u).powi(2));
        }
    }
    finer.push(grid[grid.len() - 1]);
    finer
}

fn noise_distance(a: &[f64], b: &[f64]) -> f64 {
    let difference: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    noise_norm(&difference, b)
}

fn deviance(observed: &[f64], predicted: &[f64]) -> f64 {
    2.0 * observed
        .iter()
        .zip(predicted)
        .map(|(&y, &mu)| {
            if y > 0.0 {
                y * (y / mu).ln() - (y - mu)
            } else {
                mu
            }
        })
        .sum::<f64>()
}
