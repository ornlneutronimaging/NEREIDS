//! Non-negative additive background templates for two-arm count measurements.
//!
//! The neutron signal is evaluated before entering this module. A background
//! template is an expected detector-bin shape from an independent source such
//! as a blocked-beam or detector-only measurement. Its fitted amplitude is
//! non-negative, and the background is added after the neutron response.
//!
//! This module deliberately provides no smooth-curve generator. Fitting a
//! flexible curve to the same residual it is meant to explain would not
//! identify a physical background. The SAMMY transmission-level background is
//! also a separate model with different placement and meaning.

use std::collections::HashSet;

use nereids_physics::counts_response::{
    TwoArmCountPrediction, TwoArmCounts, add_count_backgrounds,
};

use crate::error::FittingError;
use crate::lm::{FlatMatrix, invert_matrix};
use crate::poisson::PoissonConfig;

/// One independently supplied detector-bin background shape.
///
/// A single non-negative amplitude multiplies both arms. For a component that
/// exists in only one acquisition, supply zeros for the other arm. Arrays are
/// counts per unit amplitude, already normalized to the corresponding run.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoArmBackgroundTemplate {
    /// Stable component name retained in fit output.
    pub name: String,
    /// Open-beam background counts per unit amplitude.
    pub open_beam: Vec<f64>,
    /// Sample background counts per unit amplitude.
    pub sample: Vec<f64>,
}

/// Result of fitting fixed neutron signals plus background templates.
#[derive(Debug, Clone)]
pub struct TwoArmBackgroundFitResult {
    /// Component names in the same order as `amplitudes`.
    pub names: Vec<String>,
    /// Fitted non-negative template amplitudes.
    pub amplitudes: Vec<f64>,
    /// Local one-sigma amplitude uncertainties from the expected (Fisher)
    /// information of the constrained objective at the solution.
    ///
    /// Free amplitudes are conditioned on the constraint: the information is
    /// restricted to the free set before inversion, so an amplitude whose
    /// correlated partner is held on its bound is not reported with the
    /// marginal over a direction the constraint removed. An amplitude on its
    /// zero bound (see `amplitude_at_bound`) reports `1 / sqrt(I_jj)`, the
    /// one-sided curvature scale of the objective at the boundary, not a
    /// symmetric interval.
    ///
    /// `None` when the fit did not converge, when the amplitudes are not
    /// identifiable, or when the free block of the information matrix is
    /// singular. An individual entry is `NaN` when its variance is
    /// non-positive, or when its template is sensitive on a bin with zero
    /// expectation — the boundary of the Poisson support, where the expected
    /// information diverges and no regular estimate exists. A reported number
    /// is therefore never zero.
    pub amplitude_uncertainties: Option<Vec<f64>>,
    /// Whether each fitted amplitude sits on its non-negativity bound with the
    /// gradient still pushing it negative — a one-sided limit rather than an
    /// interior estimate.
    pub amplitude_at_bound: Vec<bool>,
    /// Whether every named template amplitude is separately determined.
    ///
    /// `false` means that at least two supplied shapes are linearly dependent:
    /// the total fitted background can be valid, but the individual amplitudes
    /// are not physically interpretable. In that case uncertainties are not
    /// reported.
    pub amplitudes_identifiable: bool,
    /// Neutron signal, fitted background, and total for both arms.
    pub prediction: TwoArmCountPrediction,
    /// Poisson deviance for the concatenated open and sample arrays.
    pub poisson_deviance: f64,
    /// Deviance divided by `n_informative - template_rank`.
    pub deviance_per_dof: f64,
    /// Concatenated bins that contribute to the deviance: the observation,
    /// the neutron signal, or at least one template is nonzero there.
    ///
    /// This is not simply `2 * n_bins`: a bin where all three are exactly
    /// zero yields identically zero deviance for any amplitude vector, so it
    /// cannot be a degree of freedom.
    pub n_informative: usize,
    /// Whether the bounded optimizer converged.
    pub converged: bool,
    /// Optimizer iterations.
    pub iterations: usize,
}

/// Fit amplitudes of independently supplied detector-bin templates.
///
/// The neutron signal and every template shape remain fixed. This tests
/// whether an independently chosen shape explains the counts, but cannot prove
/// its provenance; callers must retain the independent measurement record.
///
/// `observed_open_counts` and `observed_sample_counts` are measured counts, so
/// they are plain slices rather than a [`TwoArmCounts`]: that type carries an
/// acquisition-window loss, which is a property of a prediction and has no
/// meaning for data read off a detector.
#[allow(clippy::too_many_arguments)]
pub fn fit_two_arm_background_templates(
    observed_open_counts: &[f64],
    observed_sample_counts: &[f64],
    neutron_signal: TwoArmCounts,
    open_exposure_scale: f64,
    sample_exposure_scale: f64,
    templates: &[TwoArmBackgroundTemplate],
    initial_amplitudes: &[f64],
    config: &PoissonConfig,
) -> Result<TwoArmBackgroundFitResult, FittingError> {
    validate_fit_inputs(
        observed_open_counts,
        observed_sample_counts,
        &neutron_signal,
        open_exposure_scale,
        sample_exposure_scale,
        templates,
        initial_amplitudes,
    )?;
    validate_config(config)?;
    let neutron_signal =
        scale_neutron_signal(neutron_signal, open_exposure_scale, sample_exposure_scale)?;
    let n_bins = observed_open_counts.len();
    let mut observed_joined = Vec::with_capacity(2 * n_bins);
    observed_joined.extend_from_slice(observed_open_counts);
    observed_joined.extend_from_slice(observed_sample_counts);

    // Fit a contribution measured in counts, rather than the caller's
    // arbitrary template units.  Without this normalization, multiplying a
    // template by (say) 1e-8 and its amplitude by 1e8 changes the optimizer's
    // stopping test even though the physical prediction is unchanged.
    let (normalized_templates, template_scales) = normalize_templates(templates);
    let normalized_initial_amplitudes: Vec<f64> = initial_amplitudes
        .iter()
        .zip(&template_scales)
        .map(|(&amplitude, &scale)| amplitude * scale)
        .collect();
    if normalized_initial_amplitudes
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(FittingError::InvalidConfig(
            "initial amplitude times template scale must be finite".into(),
        ));
    }

    let mut neutron_joined = Vec::with_capacity(2 * n_bins);
    neutron_joined.extend_from_slice(&neutron_signal.open_beam);
    neutron_joined.extend_from_slice(&neutron_signal.sample);
    let normalized_basis: Vec<Vec<f64>> = normalized_templates
        .iter()
        .map(|template| {
            let mut joined = Vec::with_capacity(2 * n_bins);
            joined.extend_from_slice(&template.open_beam);
            joined.extend_from_slice(&template.sample);
            joined
        })
        .collect();

    let n_informative =
        count_informative_bins(&observed_joined, &neutron_joined, &normalized_basis);
    let template_rank = background_template_rank(&normalized_basis);
    let amplitudes_identifiable = template_rank == normalized_basis.len();
    // Degrees of freedom are `n_informative - template_rank`, so the guard
    // keys on the rank the same way: a dependent template set spans fewer
    // directions than it has names and is still fittable (its total is
    // determined; its individual amplitudes are reported as unidentifiable).
    if n_informative <= template_rank {
        return Err(FittingError::InvalidConfig(format!(
            "{} background amplitudes spanning {template_rank} independent direction(s) cannot \
             be fitted from {n_informative} informative count value(s) with positive degrees of \
             freedom",
            templates.len(),
        )));
    }

    let fit = fit_non_negative_poisson_linear(
        &observed_joined,
        &neutron_joined,
        &normalized_basis,
        &normalized_initial_amplitudes,
        config,
    )?;

    let amplitudes: Vec<f64> = fit
        .amplitudes
        .iter()
        .zip(&template_scales)
        .map(|(&normalized_amplitude, &scale)| normalized_amplitude / scale)
        .collect();
    if amplitudes.iter().any(|amplitude| !amplitude.is_finite()) {
        return Err(FittingError::EvaluationFailed(
            "a fitted amplitude cannot be represented in the supplied template units; rescale the template counts"
                .into(),
        ));
    }
    let amplitude_uncertainties = amplitudes_identifiable
        .then(|| {
            fit.uncertainties.map(|uncertainties| {
                uncertainties
                    .iter()
                    .zip(&template_scales)
                    .map(|(&uncertainty, &scale)| {
                        let caller_units = uncertainty / scale;
                        if caller_units.is_finite() && caller_units > 0.0 {
                            caller_units
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
        })
        .flatten();

    let mut open_background = vec![0.0; n_bins];
    let mut sample_background = vec![0.0; n_bins];
    for (&amplitude, template) in fit.amplitudes.iter().zip(&normalized_templates) {
        add_scaled(&mut open_background, &template.open_beam, amplitude);
        add_scaled(&mut sample_background, &template.sample, amplitude);
    }
    let prediction = add_count_backgrounds(neutron_signal, &open_background, &sample_background)
        .map_err(|error| FittingError::EvaluationFailed(error.to_string()))?;
    let mut predicted_joined = Vec::with_capacity(2 * n_bins);
    predicted_joined.extend_from_slice(&prediction.open_beam.total);
    predicted_joined.extend_from_slice(&prediction.sample.total);
    let poisson_deviance = poisson_deviance(&observed_joined, &predicted_joined);
    // Dependent shapes add only `rank` independent fitted directions, and only
    // informative bins can constrain them. Counting every named row, or every
    // bin, would overstate the degrees of freedom and deflate the reported
    // goodness of fit.
    let dof = n_informative - template_rank;

    Ok(TwoArmBackgroundFitResult {
        names: templates
            .iter()
            .map(|template| template.name.clone())
            .collect(),
        amplitudes,
        amplitude_uncertainties,
        amplitude_at_bound: fit.at_bound,
        amplitudes_identifiable,
        prediction,
        poisson_deviance,
        deviance_per_dof: poisson_deviance / dof as f64,
        n_informative,
        converged: fit.converged,
        iterations: fit.iterations,
    })
}

/// Count concatenated bins that can discriminate between amplitude vectors.
///
/// Under this Poisson objective a bin contributes identically zero deviance
/// for *every* amplitude vector when its observation, its neutron signal, and
/// every template are all exactly zero there: the expectation is pinned at
/// zero whatever the amplitudes do. Such a bin carries no information and must
/// not inflate the degrees of freedom, exactly as dead bins are excluded from
/// the joint-Poisson dof in [`crate::joint_poisson`].
///
/// A bin with a positive observation but zero capacity is *not* degenerate —
/// it makes the deviance infinite, which is the correct verdict on a model
/// that cannot produce an observed count — so it is counted.
fn count_informative_bins(observed: &[f64], neutron_signal: &[f64], basis: &[Vec<f64>]) -> usize {
    (0..observed.len())
        .filter(|&bin| {
            observed[bin] > 0.0
                || neutron_signal[bin] > 0.0
                || basis.iter().any(|template| template[bin] > 0.0)
        })
        .count()
}

fn scale_neutron_signal(
    mut signal: TwoArmCounts,
    open_exposure_scale: f64,
    sample_exposure_scale: f64,
) -> Result<TwoArmCounts, FittingError> {
    for value in &mut signal.open_beam {
        *value *= open_exposure_scale;
    }
    for value in &mut signal.sample {
        *value *= sample_exposure_scale;
    }
    // The window-loss report is in the same units as the arm it belongs to, so
    // it must follow the same exposure conversion or the disclosed loss would
    // silently describe a different acquisition than the counts beside it.
    signal.open_beam_window_loss *= open_exposure_scale;
    signal.sample_window_loss *= sample_exposure_scale;
    if signal
        .open_beam
        .iter()
        .chain(&signal.sample)
        .chain(std::iter::once(&signal.open_beam_window_loss))
        .chain(std::iter::once(&signal.sample_window_loss))
        .any(|value| !value.is_finite())
    {
        return Err(FittingError::InvalidConfig(
            "exposure-scaled neutron signal must remain finite; rescale the reference signal"
                .into(),
        ));
    }
    Ok(signal)
}

struct LinearPoissonFit {
    amplitudes: Vec<f64>,
    uncertainties: Option<Vec<f64>>,
    at_bound: Vec<bool>,
    converged: bool,
    iterations: usize,
}

/// Minimize the exact count likelihood for a non-negative linear background.
///
/// With fixed neutron counts `s`, fixed non-negative templates `B`, and
/// non-negative amplitudes `a`, the expectation is `mu = s + B a` and the
/// Poisson objective is convex in `a`. Each iteration takes one joint
/// Fisher-scoring step on the free amplitudes — those off the zero bound, or
/// on it with the gradient pulling inward — projected back onto `a >= 0`
/// with backtracking on the objective.
///
/// The joint step matters for the ordinary inputs: a flat detector-dark
/// reference beside a slowly varying blocked-beam one are highly correlated,
/// and a coordinate-at-a-time sweep zig-zags between them for tens of
/// thousands of passes while the joint step resolves them in a handful. The
/// safeguarded per-coordinate solve remains as the fallback whenever the
/// reduced information matrix cannot be inverted (dependent templates, or
/// counts so large that `1/mu` underflows the pivot floor) or the joint step
/// fails to decrease the objective, so those cases still reach their
/// constrained minimum. Convergence is the scale-free KKT test on the
/// normalized gradient in every case.
fn fit_non_negative_poisson_linear(
    observed: &[f64],
    neutron_signal: &[f64],
    basis: &[Vec<f64>],
    initial_amplitudes: &[f64],
    config: &PoissonConfig,
) -> Result<LinearPoissonFit, FittingError> {
    let mut amplitudes = initial_amplitudes.to_vec();
    for (amplitude, template) in amplitudes.iter_mut().zip(basis) {
        let upper = amplitude_upper_bound(observed, template)?;
        *amplitude = amplitude.min(upper);
    }
    let mut converged = false;
    let mut iterations = 0;
    let mut gradient = vec![0.0; basis.len()];

    for iteration in 0..config.max_iter {
        let prediction = linear_prediction(neutron_signal, basis, &amplitudes)?;
        poisson_gradient_into(observed, &prediction, basis, &mut gradient);
        let free: Vec<usize> = (0..basis.len())
            .filter(|&component| amplitudes[component] > 0.0 || gradient[component] < 0.0)
            .collect();

        let joint_step =
            fisher_scoring_direction(&prediction, basis, &gradient, &free).and_then(|direction| {
                projected_backtracking_step(
                    observed,
                    neutron_signal,
                    basis,
                    &amplitudes,
                    &prediction,
                    &direction,
                )
            });
        match joint_step {
            Some(next) => amplitudes = next,
            None => {
                for component in 0..basis.len() {
                    let base = linear_prediction_without_component(
                        neutron_signal,
                        basis,
                        &amplitudes,
                        component,
                    )?;
                    amplitudes[component] = coordinate_minimum(
                        observed,
                        &base,
                        &basis[component],
                        amplitudes[component],
                    )?;
                }
            }
        }
        iterations = iteration + 1;

        let prediction = linear_prediction(neutron_signal, basis, &amplitudes)?;
        poisson_gradient_into(observed, &prediction, basis, &mut gradient);
        let maximum_violation = basis
            .iter()
            .enumerate()
            .map(|(component, template)| {
                let violation = if amplitudes[component] == 0.0 {
                    (-gradient[component]).max(0.0)
                } else {
                    gradient[component].abs()
                };
                violation / template.iter().sum::<f64>()
            })
            .fold(0.0_f64, f64::max);
        if maximum_violation <= config.tol_param {
            converged = true;
            break;
        }
    }

    let prediction = linear_prediction(neutron_signal, basis, &amplitudes)?;
    poisson_gradient_into(observed, &prediction, basis, &mut gradient);
    // An amplitude on its zero bound with the gradient still positive is held
    // there by the constraint, not by the data: report it as a one-sided
    // limit rather than an interior estimate.
    let at_bound: Vec<bool> = amplitudes
        .iter()
        .zip(&gradient)
        .map(|(&amplitude, &slope)| amplitude == 0.0 && slope > 0.0)
        .collect();
    let uncertainties = if converged && config.compute_covariance {
        poisson_linear_uncertainties(&prediction, basis, &at_bound)
    } else {
        None
    };
    Ok(LinearPoissonFit {
        amplitudes,
        uncertainties,
        at_bound,
        converged,
        iterations,
    })
}

fn poisson_gradient_into(
    observed: &[f64],
    prediction: &[f64],
    basis: &[Vec<f64>],
    gradient: &mut [f64],
) {
    for (slot, template) in gradient.iter_mut().zip(basis) {
        *slot = poisson_coordinate_gradient(observed, prediction, template);
    }
}

/// Expected (Fisher) information between two templates at the current
/// expectation: `sum_i B_ij B_ik / mu_i` over bins with `mu_i > 0`.
///
/// Unlike the observed information `sum_i B_ij B_ik y_i / mu_i^2`, this does
/// not vanish on zero-count bins, so the background-consistent-with-zero
/// case keeps a finite, meaningful bound. It is the same convention as the
/// joint-Poisson curvature elsewhere in this crate. Bins with `mu_i == 0`
/// carry either no counts (no information) or an infinite deviance already.
fn expected_information(prediction: &[f64], left: &[f64], right: &[f64]) -> f64 {
    prediction
        .iter()
        .zip(left)
        .zip(right)
        .filter(|&((&expected, &l), &r)| expected > 0.0 && l != 0.0 && r != 0.0)
        .map(|((&expected, &l), &r)| (l / expected) * r)
        .sum()
}

/// Fisher-scoring direction on the free amplitudes, zero elsewhere.
///
/// `None` when the reduced information matrix cannot be inverted or the
/// direction is not finite; the caller then falls back to a coordinate
/// sweep, which is always well posed one amplitude at a time.
fn fisher_scoring_direction(
    prediction: &[f64],
    basis: &[Vec<f64>],
    gradient: &[f64],
    free: &[usize],
) -> Option<Vec<f64>> {
    if free.is_empty() {
        return None;
    }
    let mut information = FlatMatrix::zeros(free.len(), free.len());
    for (row, &j) in free.iter().enumerate() {
        for (column, &k) in free.iter().enumerate() {
            *information.get_mut(row, column) =
                expected_information(prediction, &basis[j], &basis[k]);
        }
    }
    let inverse = invert_matrix(&information)?;
    let mut direction = vec![0.0; basis.len()];
    for (row, &j) in free.iter().enumerate() {
        direction[j] = -(0..free.len())
            .map(|column| inverse.get(row, column) * gradient[free[column]])
            .sum::<f64>();
    }
    direction
        .iter()
        .all(|value| value.is_finite())
        .then_some(direction)
}

/// Move along `direction`, projecting onto `a >= 0`, halving the step until
/// the Poisson deviance strictly decreases. `None` when no step length does —
/// including at the solution itself, where the fallback sweep then confirms
/// the KKT conditions at negligible cost.
fn projected_backtracking_step(
    observed: &[f64],
    neutron_signal: &[f64],
    basis: &[Vec<f64>],
    amplitudes: &[f64],
    prediction: &[f64],
    direction: &[f64],
) -> Option<Vec<f64>> {
    let current = poisson_deviance(observed, prediction);
    let mut step = 1.0;
    for _ in 0..40 {
        let trial: Vec<f64> = amplitudes
            .iter()
            .zip(direction)
            .map(|(&amplitude, &delta)| (amplitude + step * delta).max(0.0))
            .collect();
        if let Ok(trial_prediction) = linear_prediction(neutron_signal, basis, &trial)
            && poisson_deviance(observed, &trial_prediction) < current
        {
            return Some(trial);
        }
        step *= 0.5;
    }
    None
}

fn validate_config(config: &PoissonConfig) -> Result<(), FittingError> {
    // A non-finite tolerance would accept the first iteration unconditionally
    // and report convergence for an arbitrary amplitude vector.
    if !config.tol_param.is_finite() || config.tol_param <= 0.0 {
        return Err(FittingError::InvalidConfig(format!(
            "tol_param must be finite and > 0, got {}",
            config.tol_param
        )));
    }
    // Zero iterations would return the clamped initial guess as an
    // unconverged result with no error, which is a request that cannot be
    // meant.
    if config.max_iter == 0 {
        return Err(FittingError::InvalidConfig(
            "max_iter must be at least 1".into(),
        ));
    }
    Ok(())
}

fn coordinate_minimum(
    observed: &[f64],
    base: &[f64],
    template: &[f64],
    initial: f64,
) -> Result<f64, FittingError> {
    let gradient_at_zero = poisson_coordinate_gradient_at(observed, base, template, 0.0);
    if gradient_at_zero >= 0.0 {
        return Ok(0.0);
    }

    // For base >= 0, this is a mathematical upper bound on the root:
    // b*y/(base+x*b) <= y/x for every bin with b>0.
    let mut upper = amplitude_upper_bound(observed, template)?;
    if upper <= 0.0 {
        return Err(FittingError::EvaluationFailed(
            "could not form a finite upper bound for a background amplitude".into(),
        ));
    }
    let mut lower = 0.0;
    let mut value = initial.clamp(lower, upper);

    // Safeguarded Newton solves the monotone analytical derivative. The
    // bracket makes the result deterministic at a zero boundary and across
    // large changes in curvature.
    for _ in 0..80 {
        let (gradient, curvature) =
            poisson_coordinate_gradient_and_curvature_at(observed, base, template, value);
        if gradient == 0.0 {
            return Ok(value);
        }
        if gradient < 0.0 {
            lower = value;
        } else {
            upper = value;
        }
        let newton = value - gradient / curvature;
        let next = if curvature.is_finite()
            && curvature > 0.0
            && newton.is_finite()
            && newton > lower
            && newton < upper
        {
            newton
        } else {
            lower + 0.5 * (upper - lower)
        };
        if next == value {
            return Ok(next);
        }
        value = next;
    }
    Ok(value)
}

/// A finite upper bound for one normalized template amplitude.
///
/// Dividing two direct sums can overflow even when their ratio is finite
/// (for example, two observations near `f64::MAX`). Scaling the numerator by
/// its largest supported observation computes the same ratio without that
/// intermediate overflow.
fn amplitude_upper_bound(observed: &[f64], template: &[f64]) -> Result<f64, FittingError> {
    let maximum_observed = observed
        .iter()
        .zip(template)
        .filter_map(|(&count, &weight)| (weight > 0.0).then_some(count))
        .fold(0.0_f64, f64::max);
    if maximum_observed == 0.0 {
        return Ok(0.0);
    }
    let scaled_observed_sum: f64 = observed
        .iter()
        .zip(template)
        .filter_map(|(&count, &weight)| (weight > 0.0).then_some(count / maximum_observed))
        .sum();
    let template_sum: f64 = template.iter().sum();
    let upper = maximum_observed * (scaled_observed_sum / template_sum);
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(FittingError::EvaluationFailed(
            "could not form a finite upper bound for a background amplitude".into(),
        ))
    }
}

fn linear_prediction_without_component(
    neutron_signal: &[f64],
    basis: &[Vec<f64>],
    amplitudes: &[f64],
    excluded: usize,
) -> Result<Vec<f64>, FittingError> {
    let mut prediction = neutron_signal.to_vec();
    for (component, (&amplitude, template)) in amplitudes.iter().zip(basis).enumerate() {
        if component != excluded {
            add_scaled(&mut prediction, template, amplitude);
        }
    }
    validate_linear_prediction(&prediction)?;
    Ok(prediction)
}

fn linear_prediction(
    neutron_signal: &[f64],
    basis: &[Vec<f64>],
    amplitudes: &[f64],
) -> Result<Vec<f64>, FittingError> {
    let mut prediction = neutron_signal.to_vec();
    for (&amplitude, template) in amplitudes.iter().zip(basis) {
        add_scaled(&mut prediction, template, amplitude);
    }
    validate_linear_prediction(&prediction)?;
    Ok(prediction)
}

fn validate_linear_prediction(prediction: &[f64]) -> Result<(), FittingError> {
    if prediction.iter().any(|value| !value.is_finite()) {
        return Err(FittingError::EvaluationFailed(
            "background fit produced non-finite expected counts".into(),
        ));
    }
    Ok(())
}

fn poisson_coordinate_gradient(observed: &[f64], prediction: &[f64], template: &[f64]) -> f64 {
    observed
        .iter()
        .zip(prediction)
        .zip(template)
        .filter(|&((_, _), &weight)| weight > 0.0)
        .map(|((&count, &expected), &weight)| {
            weight - weighted_count_ratio(weight, count, expected)
        })
        .sum()
}

fn poisson_coordinate_gradient_at(
    observed: &[f64],
    base: &[f64],
    template: &[f64],
    amplitude: f64,
) -> f64 {
    poisson_coordinate_gradient_and_curvature_at(observed, base, template, amplitude).0
}

fn poisson_coordinate_gradient_and_curvature_at(
    observed: &[f64],
    base: &[f64],
    template: &[f64],
    amplitude: f64,
) -> (f64, f64) {
    observed.iter().zip(base).zip(template).fold(
        (0.0, 0.0),
        |(gradient, curvature), ((&count, &base_count), &weight)| {
            if weight == 0.0 {
                return (gradient, curvature);
            }
            let expected = base_count + amplitude * weight;
            let weighted_ratio = weighted_count_ratio(weight, count, expected);
            let curvature_term = if count == 0.0 {
                0.0
            } else {
                (weight / expected) * weighted_ratio
            };
            (
                gradient + weight - weighted_ratio,
                curvature + curvature_term,
            )
        },
    )
}

/// Compute `weight * count / expected` without avoidable intermediate
/// overflow. Exact zero observations contribute zero; a positive observation
/// with zero expectation contributes infinity, as required by Poisson counts.
fn weighted_count_ratio(weight: f64, count: f64, expected: f64) -> f64 {
    if weight == 0.0 || count == 0.0 {
        0.0
    } else if weight <= expected {
        count * (weight / expected)
    } else {
        weight * (count / expected)
    }
}

/// One-sigma amplitude uncertainties at the constrained solution.
///
/// The local curvature of the constrained objective is the expected
/// information restricted to the free amplitudes, so the free block is
/// inverted on its own. A partner held on its bound has been removed as a
/// direction; marginalizing over it would report the flat direction of a
/// near-collinear pair as the uncertainty of an amplitude the data in fact
/// pin to a fraction of a percent. An amplitude on its bound reports
/// `1 / sqrt(I_jj)`, the direct curvature scale of the objective at the
/// boundary.
///
/// A template that is sensitive on a bin with zero expectation sits on the
/// boundary of the Poisson support, where its expected information diverges
/// and the Fisher approximation is not a regular estimate. Such entries are
/// `NaN` rather than a number that would jump discontinuously as the
/// expectation reaches zero. `None` only when the free block is singular.
fn poisson_linear_uncertainties(
    prediction: &[f64],
    basis: &[Vec<f64>],
    at_bound: &[bool],
) -> Option<Vec<f64>> {
    let support_boundary: Vec<bool> = basis
        .iter()
        .map(|template| {
            prediction
                .iter()
                .zip(template)
                .any(|(&expected, &weight)| expected == 0.0 && weight != 0.0)
        })
        .collect();
    let free: Vec<usize> = (0..basis.len())
        .filter(|&j| !at_bound[j] && !support_boundary[j])
        .collect();

    let mut sigma = vec![f64::NAN; basis.len()];
    if !free.is_empty() {
        let mut information = FlatMatrix::zeros(free.len(), free.len());
        for (row, &j) in free.iter().enumerate() {
            for (column, &k) in free.iter().enumerate() {
                *information.get_mut(row, column) =
                    expected_information(prediction, &basis[j], &basis[k]);
            }
        }
        let covariance = invert_matrix(&information)?;
        for (row, &j) in free.iter().enumerate() {
            sigma[j] = positive_sqrt_or_nan(covariance.get(row, row));
        }
    }
    for (j, template) in basis.iter().enumerate() {
        if at_bound[j] && !support_boundary[j] {
            sigma[j] =
                positive_sqrt_or_nan(1.0 / expected_information(prediction, template, template));
        }
    }
    Some(sigma)
}

fn positive_sqrt_or_nan(variance: f64) -> f64 {
    if variance.is_finite() && variance > 0.0 {
        variance.sqrt()
    } else {
        f64::NAN
    }
}

fn normalize_templates(
    templates: &[TwoArmBackgroundTemplate],
) -> (Vec<TwoArmBackgroundTemplate>, Vec<f64>) {
    let scales: Vec<f64> = templates
        .iter()
        .map(|template| {
            template
                .open_beam
                .iter()
                .chain(&template.sample)
                .copied()
                .fold(0.0_f64, f64::max)
        })
        .collect();
    let normalized = templates
        .iter()
        .zip(&scales)
        .map(|(template, &scale)| TwoArmBackgroundTemplate {
            name: template.name.clone(),
            open_beam: template
                .open_beam
                .iter()
                .map(|value| value / scale)
                .collect(),
            sample: template.sample.iter().map(|value| value / scale).collect(),
        })
        .collect();
    (normalized, scales)
}

/// Number of linearly independent concatenated open/sample shapes.
///
/// Templates are first normalized to a maximum entry of one, so this test is
/// independent of the caller's amplitude units. The tolerance identifies only
/// dependence at floating-point resolution; it does not claim that two nearly
/// similar shapes can be separated in noisy data. That practical question is
/// handled by the later parameter-separation analysis.
fn background_template_rank(basis: &[Vec<f64>]) -> usize {
    let n_rows = basis.first().map_or(0, Vec::len);
    let dimension = n_rows.max(basis.len()) as f64;
    let tolerance = f64::EPSILON * dimension * (n_rows as f64).sqrt();
    let mut orthonormal: Vec<Vec<f64>> = Vec::with_capacity(basis.len());

    for column in basis {
        let mut residual = column.clone();
        // A second pass makes the result stable when several earlier columns
        // are close to one another.
        for _ in 0..2 {
            for direction in &orthonormal {
                let projection: f64 = residual.iter().zip(direction).map(|(a, b)| a * b).sum();
                for (value, &unit_value) in residual.iter_mut().zip(direction) {
                    *value -= projection * unit_value;
                }
            }
        }
        let norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm > tolerance {
            for value in &mut residual {
                *value /= norm;
            }
            orthonormal.push(residual);
        }
    }
    orthonormal.len()
}

fn add_scaled(output: &mut [f64], template: &[f64], amplitude: f64) {
    for (value, &basis) in output.iter_mut().zip(template) {
        *value += amplitude * basis;
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_fit_inputs(
    observed_open_counts: &[f64],
    observed_sample_counts: &[f64],
    neutron_signal: &TwoArmCounts,
    open_exposure_scale: f64,
    sample_exposure_scale: f64,
    templates: &[TwoArmBackgroundTemplate],
    initial_amplitudes: &[f64],
) -> Result<(), FittingError> {
    let n_bins = observed_open_counts.len();
    if n_bins == 0 {
        return Err(FittingError::EmptyData);
    }
    for (field, actual) in [
        ("observed_sample_counts", observed_sample_counts.len()),
        ("open_neutron_signal", neutron_signal.open_beam.len()),
        ("sample_neutron_signal", neutron_signal.sample.len()),
    ] {
        if actual != n_bins {
            return Err(FittingError::LengthMismatch {
                expected: n_bins,
                actual,
                field,
            });
        }
    }
    if templates.is_empty() {
        return Err(FittingError::InvalidConfig(
            "at least one independently supplied background template is required".into(),
        ));
    }
    if initial_amplitudes.len() != templates.len() {
        return Err(FittingError::LengthMismatch {
            expected: templates.len(),
            actual: initial_amplitudes.len(),
            field: "initial_amplitudes",
        });
    }

    validate_non_negative("observed_open_counts", observed_open_counts)?;
    validate_non_negative("observed_sample_counts", observed_sample_counts)?;
    validate_non_negative("open_neutron_signal", &neutron_signal.open_beam)?;
    validate_non_negative("sample_neutron_signal", &neutron_signal.sample)?;
    for (name, scale) in [
        ("open_exposure_scale", open_exposure_scale),
        ("sample_exposure_scale", sample_exposure_scale),
    ] {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(FittingError::InvalidConfig(format!(
                "{name} must be finite and > 0, got {scale}"
            )));
        }
    }
    // Validated here, before any optimization, so a bad value is reported as
    // malformed input (InvalidConfig) rather than surfacing from the physics
    // layer after a full fit as an evaluation failure.
    for (name, value) in [
        (
            "open_beam_window_loss",
            neutron_signal.open_beam_window_loss,
        ),
        ("sample_window_loss", neutron_signal.sample_window_loss),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(FittingError::InvalidConfig(format!(
                "{name} must be finite and >= 0 expected counts, got {value}"
            )));
        }
    }

    let mut names = HashSet::with_capacity(templates.len());
    for (index, (template, &initial)) in templates.iter().zip(initial_amplitudes).enumerate() {
        if template.name.trim().is_empty() {
            return Err(FittingError::InvalidConfig(format!(
                "background template {index} has an empty name"
            )));
        }
        if !names.insert(template.name.as_str()) {
            return Err(FittingError::InvalidConfig(format!(
                "background template name '{}' is duplicated",
                template.name
            )));
        }
        for (field, actual) in [
            ("open_background_template", template.open_beam.len()),
            ("sample_background_template", template.sample.len()),
        ] {
            if actual != n_bins {
                return Err(FittingError::LengthMismatch {
                    expected: n_bins,
                    actual,
                    field,
                });
            }
        }
        validate_non_negative("open_background_template", &template.open_beam)?;
        validate_non_negative("sample_background_template", &template.sample)?;
        if !template
            .open_beam
            .iter()
            .chain(&template.sample)
            .any(|&value| value > 0.0)
        {
            return Err(FittingError::InvalidConfig(format!(
                "background template '{}' is zero in both arms",
                template.name
            )));
        }
        if !initial.is_finite() || initial < 0.0 {
            return Err(FittingError::InvalidConfig(format!(
                "initial_amplitudes[{index}] must be finite and >= 0, got {initial}"
            )));
        }
    }
    Ok(())
}

fn validate_non_negative(field: &'static str, values: &[f64]) -> Result<(), FittingError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(FittingError::InvalidConfig(format!(
                "{field}[{index}] must be finite and >= 0 expected counts, got {value}"
            )));
        }
    }
    Ok(())
}

fn poisson_deviance(observed: &[f64], predicted: &[f64]) -> f64 {
    observed
        .iter()
        .zip(predicted)
        .map(|(&obs, &model)| {
            if obs > 0.0 {
                if model == 0.0 {
                    return f64::INFINITY;
                }
                // h(r) = (1+r) ln(1+r) - r, where r=(obs-model)/model.
                // A short series avoids subtracting nearly equal, very large
                // numbers when the fitted and observed counts almost match.
                let r = (obs - model) / model;
                let deviance = if r.abs() < 1.0e-3 {
                    let h = r
                        * r
                        * (0.5
                            + r * (-1.0 / 6.0
                                + r * (1.0 / 12.0 + r * (-1.0 / 20.0 + r * (1.0 / 30.0)))));
                    // The small factor is applied first: `(2 * model) * h`
                    // overflows for `model > MAX / 2` even when the exact
                    // term is representable.
                    2.0 * (model * h)
                } else {
                    // Away from equality the direct form does not suffer
                    // cancellation.  Subtracting logarithms also avoids an
                    // intermediate obs/model overflow or underflow.
                    2.0 * (obs * (obs.ln() - model.ln()) - (obs - model))
                };
                // Each exact deviance term is non-negative.  Guard only
                // against a final sub-ulp negative caused by floating point.
                deviance.max(0.0)
            } else {
                2.0 * model
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{count_informative_bins, poisson_deviance};

    #[test]
    fn exact_zero_observation_and_prediction_have_zero_deviance() {
        assert_eq!(poisson_deviance(&[0.0], &[0.0]), 0.0);
        assert!(poisson_deviance(&[1.0], &[0.0]).is_infinite());
    }

    #[test]
    fn nearly_equal_large_counts_have_non_negative_deviance() {
        let observed = [31_415_926_535_897.0, 27_182_818_284_590.0];
        let predicted = [31_415_926_535_896.0, 27_182_818_284_592.0];
        let deviance = poisson_deviance(&observed, &predicted);
        assert!(deviance.is_finite());
        assert!(deviance >= 0.0, "deviance = {deviance}");
    }

    /// `(2 * model) * h` overflows for `model > MAX / 2` although the exact
    /// term `2 * model * h` is representable; the factor order matters.
    #[test]
    fn near_equality_extreme_counts_keep_finite_deviance() {
        let deviance = poisson_deviance(&[1.0001e308], &[1.0e308]);
        assert!(
            deviance.is_finite() && deviance > 0.0,
            "deviance = {deviance}"
        );
    }

    #[test]
    fn extreme_finite_count_ratios_keep_finite_deviance() {
        let low_observation = poisson_deviance(&[1.0e-200], &[1.0e200]);
        let high_observation = poisson_deviance(&[1.0e200], &[1.0e-200]);
        assert!(low_observation.is_finite() && low_observation >= 0.0);
        assert!(high_observation.is_finite() && high_observation >= 0.0);
    }

    /// A bin whose observation, signal, and every template are exactly zero
    /// cannot discriminate between amplitude vectors, so it is not a degree of
    /// freedom. A positive observation against zero capacity is informative:
    /// it refutes the model outright.
    #[test]
    fn only_bins_that_can_discriminate_are_informative() {
        let basis = vec![vec![0.0, 1.0, 0.0, 0.0]];
        let observed = [0.0, 5.0, 0.0, 3.0];
        let neutron = [0.0, 2.0, 0.0, 0.0];
        assert_eq!(count_informative_bins(&observed, &neutron, &basis), 2);

        let all_dead = [0.0, 0.0];
        assert_eq!(
            count_informative_bins(&all_dead, &all_dead, &[vec![0.0, 0.0]]),
            0
        );
    }
}
