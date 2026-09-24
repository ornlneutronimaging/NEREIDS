//! Poisson-likelihood fitting of counts.
//!
//! Minimizes half the Poisson deviance
//!
//! ```text
//! D(θ) = Σᵢ [μᵢ(θ) − yᵢ + yᵢ ln(yᵢ / μᵢ(θ))]
//! ```
//!
//! within box bounds, by projected Levenberg–Marquardt steps in the Fisher
//! metric, for models with an analytical Jacobian.
//!
//! **Scope note.** The production pipeline does not apply this single-arm
//! objective to normalized transmission. Raw open/sample counts use the
//! joint-Poisson conditional-binomial-deviance solver in
//! [`crate::joint_poisson`]. This module remains available to the
//! `evaluate_jacobian_and_fisher` Fisher-information helper (via
//! [`CountsModel`], [`CountsBackgroundScaleModel`] and
//! [`TransmissionKLBackgroundModel`], all three of which that helper still
//! constructs) and to spatial-regularization research drivers; it is not a
//! public transmission fitting route.

use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};
use crate::parameters::{FitParameter, ParameterSet};

/// Configuration for the Poisson solvers.
#[derive(Debug, Clone)]
pub struct PoissonConfig {
    /// Maximum number of steps.
    pub max_iter: usize,
    /// Convergence tolerance of the count-background solver; [`poisson_fit`]
    /// stops on the Newton decrement instead.
    pub tol_param: f64,
    /// Whether to compute the covariance and error bars after convergence.
    pub compute_covariance: bool,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol_param: 1e-8,
            compute_covariance: true,
        }
    }
}

/// Result of [`poisson_fit`].
#[derive(Debug, Clone)]
pub struct PoissonResult {
    /// Half the Poisson deviance at `params`.
    pub deviance: f64,
    /// Number of steps taken.
    pub iterations: usize,
    /// Whether the expected-information Newton decrement fell below 1e-6
    /// inside the bounds: within about 0.0014 standard errors of a minimum
    /// where the deviance is quadratic there; see [`poisson_fit`].
    pub converged: bool,
    /// Final parameter values (all parameters, including fixed).
    pub params: Vec<f64>,
    /// Covariance of the free parameters; the rows and columns of a
    /// parameter without an error bar are NaN.  `None` when the fit did not
    /// converge or covariance computation is disabled.
    pub covariance: Option<FlatMatrix>,
    /// Standard error of each free parameter; `None` for a parameter on a
    /// bound or with a component along a direction the data do not
    /// determine.  `None` overall when `covariance` is.
    pub uncertainties: Option<Vec<Option<f64>>>,
    /// Whether each free parameter ended on one of its bounds.
    pub on_bound: Vec<bool>,
}

const NEWTON_DECREMENT_TOL: f64 = 1e-6;

const DEGENERATE_EIGENVALUE: f64 = 1e-12;

const MAX_JACOBI_SWEEPS: usize = 64;

const MAX_REJECTIONS: usize = 60;

const INITIAL_DAMPING: f64 = 1e-3;

const DAMPING_FACTOR: f64 = 10.0;

/// `obs·ln(obs/mean) + mean − obs`, by C. Loader's `bd0` ("Fast and
/// accurate computation of binomial probabilities", 2000): a series in
/// `v = (obs − mean)/(obs + mean)` when `|obs − mean| < 0.1 (obs + mean)`.
fn half_deviance(obs: f64, mean: f64) -> f64 {
    let half_sum = obs / 2.0 + mean / 2.0;
    let difference = obs - mean;
    if difference.abs() < 0.2 * half_sum {
        let v = difference / 2.0 / half_sum;
        let mut sum = difference * v;
        let mut term = obs * (2.0 * v);
        let mut j = 1.0;
        loop {
            term *= v * v;
            let next = sum + term / (2.0 * j + 1.0);
            if next == sum {
                return sum;
            }
            sum = next;
            j += 1.0;
        }
    } else if obs == 0.0 {
        mean
    } else {
        obs * (obs.ln() - mean.ln()) + mean - obs
    }
}

fn deviance(y_obs: &[f64], y_model: &[f64]) -> f64 {
    y_obs
        .iter()
        .zip(y_model)
        .map(|(&obs, &mean)| {
            if mean > 0.0 || (mean == 0.0 && obs == 0.0) {
                half_deviance(obs, mean)
            } else {
                f64::INFINITY
            }
        })
        .sum()
}

struct Linearization {
    weighted: FlatMatrix,
    residual: Vec<f64>,
    zero_slope: Vec<f64>,
    gradient: Vec<f64>,
}

fn linearize(
    model: &dyn FitModel,
    params: &ParameterSet,
    free: &[usize],
    y_obs: &[f64],
    y_model: &[f64],
) -> Result<Linearization, FittingError> {
    let mut weighted = model
        .analytical_jacobian(&params.all_values(), free, y_model)
        .ok_or_else(|| {
            FittingError::InvalidConfig(
                "poisson_fit needs a model with an analytical Jacobian".into(),
            )
        })?;
    for (expected, actual, field) in [
        (y_obs.len(), weighted.nrows, "analytical Jacobian rows"),
        (free.len(), weighted.ncols, "analytical Jacobian columns"),
    ] {
        if expected != actual {
            return Err(FittingError::LengthMismatch {
                expected,
                actual,
                field,
            });
        }
    }
    let root: Vec<f64> = y_model.iter().map(|mean| mean.sqrt()).collect();
    let mut zero_slope = vec![0.0; weighted.ncols];
    for (i, &r) in root.iter().enumerate() {
        for (j, z) in zero_slope.iter_mut().enumerate() {
            let slope = weighted.get(i, j);
            if r > 0.0 {
                *weighted.get_mut(i, j) = slope / r;
            } else {
                *z += slope;
                *weighted.get_mut(i, j) = 0.0;
            }
        }
    }
    let residual: Vec<f64> = y_obs
        .iter()
        .zip(y_model)
        .zip(&root)
        .map(|((&obs, &mean), &r)| if r > 0.0 { (mean - obs) / r } else { 0.0 })
        .collect();
    let gradient = zero_slope
        .iter()
        .enumerate()
        .map(|(j, z)| {
            z + (0..weighted.nrows)
                .map(|i| weighted.get(i, j) * residual[i])
                .sum::<f64>()
        })
        .collect();
    Ok(Linearization {
        weighted,
        residual,
        zero_slope,
        gradient,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn one_sided_jacobi(columns: &mut [Vec<f64>]) -> Vec<Vec<f64>> {
    let k = columns.len();
    let mut v: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let rotate = |a: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64| {
        for row in 0..a[p].len() {
            let (x, y) = (a[p][row], a[q][row]);
            a[p][row] = c * x - s * y;
            a[q][row] = s * x + c * y;
        }
    };
    for _ in 0..MAX_JACOBI_SWEEPS {
        let mut rotated = false;
        for p in 0..k {
            for q in p + 1..k {
                let alpha = dot(&columns[p], &columns[p]);
                let beta = dot(&columns[q], &columns[q]);
                let gamma = dot(&columns[p], &columns[q]);
                if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                rotate(columns, p, q, c, c * t);
                rotate(&mut v, p, q, c, c * t);
            }
        }
        if !rotated {
            break;
        }
    }
    v
}

fn column_scale(weighted: &FlatMatrix, col: usize) -> (f64, f64) {
    let peak = (0..weighted.nrows).fold(0.0_f64, |m, i| m.max(weighted.get(i, col).abs()));
    let length = (0..weighted.nrows)
        .map(|i| (weighted.get(i, col) / peak).powi(2))
        .sum::<f64>()
        .sqrt();
    (peak, length)
}

struct Decomposition {
    columns: Vec<usize>,
    largest: Vec<f64>,
    length: Vec<f64>,
    singular: Vec<f64>,
    left: Vec<Vec<f64>>,
    right: Vec<Vec<f64>>,
}

impl Decomposition {
    fn new(weighted: &FlatMatrix, columns: &[usize]) -> Self {
        let mut kept = vec![];
        let (mut largest, mut length, mut left) = (vec![], vec![], vec![]);
        for &col in columns {
            let (peak, norm) = column_scale(weighted, col);
            if peak > 0.0 {
                kept.push(col);
                largest.push(peak);
                length.push(norm);
                left.push(
                    (0..weighted.nrows)
                        .map(|i| weighted.get(i, col) / peak / norm)
                        .collect(),
                );
            }
        }
        let right = one_sided_jacobi(&mut left);
        let singular = left.iter().map(|u| dot(u, u).sqrt()).collect();
        Self {
            columns: kept,
            largest,
            length,
            singular,
            left,
            right,
        }
    }

    fn unscale(&self, i: usize, value: f64) -> f64 {
        value / self.length[i] / self.largest[i]
    }

    fn spanned(&self) -> impl Iterator<Item = usize> + '_ {
        let largest = self.singular.iter().fold(0.0_f64, |m, &s| m.max(s));
        let rows = self.left.first().map_or(0, Vec::len);
        let rank_floor = f64::EPSILON * rows.max(self.singular.len()) as f64 * largest;
        (0..self.singular.len()).filter(move |&k| self.singular[k] > rank_floor)
    }

    fn along(&self, k: usize, linear: &Linearization) -> f64 {
        let zero_part: f64 = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, &col)| self.right[k][i] * self.unscale(i, linear.zero_slope[col]))
            .sum();
        (dot(&self.left[k], &linear.residual) + zero_part) / self.singular[k]
    }

    fn newton_decrement(&self, linear: &Linearization) -> f64 {
        0.5 * self
            .spanned()
            .map(|k| self.along(k, linear).powi(2))
            .sum::<f64>()
    }

    fn step(&self, linear: &Linearization, n_free: usize, damping: f64) -> Vec<f64> {
        let mut direction = vec![0.0; n_free];
        for k in self.spanned() {
            let s = self.singular[k];
            let coefficient = self.along(k, linear) * s / (s * s + damping);
            for (i, &col) in self.columns.iter().enumerate() {
                direction[col] += self.unscale(i, self.right[k][i] * coefficient);
            }
        }
        direction
    }

    fn error_bars(&self, n_free: usize) -> (FlatMatrix, Vec<Option<f64>>) {
        let n = self.columns.len();
        let rows = self.left.first().map_or(0, Vec::len);
        let rounding = (f64::EPSILON * rows.max(n) as f64).powi(2);
        let determined: Vec<bool> = self
            .singular
            .iter()
            .map(|s| s * s >= DEGENERATE_EIGENVALUE)
            .collect();
        let resolved: Vec<bool> = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&k| !determined[k])
                    .map(|k| self.right[k][i].powi(2))
                    .sum::<f64>()
                    <= rounding
            })
            .collect();
        let variance = |i: usize, j: usize| -> f64 {
            (0..n)
                .filter(|&k| determined[k])
                .map(|k| self.right[k][i] * self.right[k][j] / self.singular[k].powi(2))
                .sum()
        };
        let mut covariance = FlatMatrix::zeros(n_free, n_free);
        covariance.data.fill(f64::NAN);
        let mut errors = vec![None; n_free];
        for i in (0..n).filter(|&i| resolved[i]) {
            for j in (0..n).filter(|&j| resolved[j]) {
                *covariance.get_mut(self.columns[i], self.columns[j]) =
                    self.unscale(j, self.unscale(i, variance(i, j)));
            }
            errors[self.columns[i]] = Some(self.unscale(i, variance(i, i).sqrt()));
        }
        (covariance, errors)
    }
}

fn on_bound(param: &FitParameter) -> bool {
    param.value == param.lower || param.value == param.upper
}

fn held_by_bound(param: &FitParameter, gradient: f64) -> bool {
    (param.value == param.lower && gradient > 0.0) || (param.value == param.upper && gradient < 0.0)
}

/// Fit `params` to the counts `y_obs` by minimizing half the Poisson
/// deviance within the parameter bounds.
///
/// The model must provide an analytical Jacobian at every point the fit
/// visits, and non-negative predictions; a bin predicted zero must have no
/// counts, and adds its slope to the gradient and nothing to the
/// information.  Each step is the Levenberg–Marquardt step
/// `(F + λI)⁻¹g` in coordinates scaled to unit Fisher information, projected
/// onto the box; `λ` is divided by 10 after a step that lowers the deviance
/// and multiplied by 10 before retrying one that does not (D. W. Marquardt,
/// J. Soc. Indust. Appl. Math. 11, 431–441, 1963).  A
/// free parameter on its bound whose gradient points out of the box is held
/// there for the step and left out of the convergence test.
///
/// The fit has converged when the Newton decrement `½ gᵀF⁺g` over the
/// parameters not held by a bound (on it, gradient pointing out),
/// `F = Jᵀ diag(1/μ) J` the expected information, is below 1e-6: the
/// quadratic model built from `F` predicts less than 1e-6 of further
/// decrease.  Where the deviance is quadratic near the minimum this puts the
/// fit within about 0.0014 standard errors of it, and within 0.01 where the
/// expected information understates the curvature, as at one count per bin.  Where it is not — weakly
/// determined parameters on a curved or flat likelihood ridge, a minimum at
/// infinity, or a parameter that adds counts to a bin predicted near zero —
/// the fit can stop short of the minimum, or at another local minimum, and
/// the error bars are not standard errors.
///
/// It stops unconverged when no step lowers the deviance, when the Jacobian
/// is not finite, when the start predicts a negative count or zero where
/// something was counted, or after `config.max_iter` steps.
///
/// # Errors
/// `FittingError::EmptyData` if `y_obs` is empty;
/// `FittingError::InvalidConfig` if an observation is negative or not
/// finite, a parameter value is not finite, a free parameter's bounds are
/// inverted or NaN, or the model returns no analytical Jacobian;
/// `FittingError::LengthMismatch` if the prediction or the Jacobian does not
/// match `y_obs` and the free parameters; the model's error if it fails at
/// the start.
pub fn poisson_fit(
    model: &dyn FitModel,
    y_obs: &[f64],
    params: &mut ParameterSet,
    config: &PoissonConfig,
) -> Result<PoissonResult, FittingError> {
    if let Some((bin, &obs)) = y_obs
        .iter()
        .enumerate()
        .find(|(_, obs)| !(obs.is_finite() && **obs >= 0.0))
    {
        return Err(FittingError::InvalidConfig(format!(
            "observed counts must be finite and non-negative, got {obs} in bin {bin}"
        )));
    }
    if y_obs.is_empty() {
        return Err(FittingError::EmptyData);
    }
    if let Some(p) = params.params.iter().find(|p| {
        !p.value.is_finite()
            || (!p.fixed && (p.lower.is_nan() || p.upper.is_nan() || p.lower > p.upper))
    }) {
        return Err(FittingError::InvalidConfig(format!(
            "parameter {} = {} with bounds [{}, {}]",
            p.name, p.value, p.lower, p.upper
        )));
    }
    params.set_free_values(&params.free_values());
    let mut y_model = model.evaluate(&params.all_values())?;
    if y_model.len() != y_obs.len() {
        return Err(FittingError::LengthMismatch {
            expected: y_model.len(),
            actual: y_obs.len(),
            field: "y_obs",
        });
    }
    let free = params.free_indices();
    let mut value = deviance(y_obs, &y_model);
    let mut iterations = 0;
    let mut at_minimum = None;
    let mut damping = INITIAL_DAMPING;
    while value.is_finite() {
        let linear = linearize(model, params, &free, y_obs, &y_model)?;
        if !linear.weighted.data.iter().all(|v| v.is_finite()) {
            break;
        }
        let movable: Vec<usize> = (0..free.len())
            .filter(|&j| !held_by_bound(&params.params[free[j]], linear.gradient[j]))
            .collect();
        let decomposition = Decomposition::new(&linear.weighted, &movable);
        if decomposition.newton_decrement(&linear) < NEWTON_DECREMENT_TOL {
            at_minimum = Some(linear);
            break;
        }
        if iterations == config.max_iter {
            break;
        }
        let start = params.free_values();
        let mut accepted = None;
        for _ in 0..MAX_REJECTIONS {
            let direction = decomposition.step(&linear, free.len(), damping);
            let unprojected: Vec<f64> = start.iter().zip(&direction).map(|(x, d)| x - d).collect();
            params.set_free_values(&unprojected);
            if let Ok(trial_model) = model.evaluate(&params.all_values()) {
                let trial_value = deviance(y_obs, &trial_model);
                if trial_value < value {
                    damping /= DAMPING_FACTOR;
                    accepted = Some((trial_model, trial_value));
                    break;
                }
            }
            params.set_free_values(&start);
            damping = (damping * DAMPING_FACTOR).max(INITIAL_DAMPING);
        }
        let Some((trial_model, trial_value)) = accepted else {
            break;
        };
        y_model = trial_model;
        value = trial_value;
        iterations += 1;
    }

    let bounded: Vec<bool> = free
        .iter()
        .map(|&idx| on_bound(&params.params[idx]))
        .collect();
    let (covariance, uncertainties) = match &at_minimum {
        Some(linear) if config.compute_covariance => {
            let interior: Vec<usize> = (0..free.len()).filter(|&j| !bounded[j]).collect();
            let (covariance, errors) =
                Decomposition::new(&linear.weighted, &interior).error_bars(free.len());
            (Some(covariance), Some(errors))
        }
        _ => (None, None),
    };
    Ok(PoissonResult {
        deviance: value,
        iterations,
        converged: at_minimum.is_some(),
        params: params.all_values(),
        covariance,
        uncertainties,
        on_bound: bounded,
    })
}

/// Fixed-flux counts-domain forward model: `Y_model = flux × T_model(θ) + background`.
///
/// **Retained for the research Fisher helper, not for production fitting.**
/// The production counts-KL dispatch (`SolverConfig::PoissonKL` on
/// `InputData::Counts` / `InputData::CountsWithNuisance`) goes through
/// the joint-Poisson conditional-binomial-deviance path in
/// [`crate::joint_poisson`].  `CountsModel` and
/// [`CountsBackgroundScaleModel`] below are consumed only by
/// `nereids_pipeline::pipeline::evaluate_jacobian_and_fisher` (the
/// Fisher-info research helper used by the spatial-regularization
/// epic #394) and by this module's `#[cfg(test)]` tests.  They assume
/// the caller has pre-computed `flux = c · O` (i.e. `c` is baked into
/// `flux` — a convention that proved error-prone for
/// end users, which is precisely why the production path no longer
/// uses this struct).
///
/// ## Physical count-response contract
///
/// This low-level wrapper cannot inspect `transmission_model` to determine how
/// instrument resolution was applied.  Its output must already be the
/// physically valid effective sample/open count response.  With response
/// operator `R` and incident spectrum `Φ`, the required ratio is
///
/// ```text
/// T_eff = R[Φ · T] / R[Φ].
/// ```
///
/// A post-hoc broadened transmission `R[T]` is not a valid substitute and
/// must never be wrapped here as though it were.  In particular, do not
/// directly wrap a resolution-bearing
/// [`TransmissionFitModel`](crate::transmission_model::TransmissionFitModel),
/// because that model returns `R[T]`.  An ordinary transmission is valid when
/// resolution is disabled, and a custom inner model that already returns the
/// exact effective ratio is also valid.  Otherwise, model the open and sample
/// response arms separately or call the guarded pipeline, which rejects
/// unsupported counts-plus-resolution combinations.  When resolution is
/// active, `flux` must be the matching resolved open-arm response `R[Φ]`.
///
/// The `flux` and `background` slices must have the same length as the
/// transmission vector returned by the inner model.  In debug builds,
/// `evaluate()` asserts this invariant.
pub struct CountsModel<'a> {
    /// Underlying effective count-ratio model.
    ///
    /// See the struct-level physical count-response contract.  In particular,
    /// this must not be a post-hoc broadened transmission `R[T]`.
    pub transmission_model: &'a dyn FitModel,
    /// Open-arm flux response (counts per bin, after normalization).
    pub flux: &'a [f64],
    /// Background counts per bin.
    pub background: &'a [f64],
    /// Total parameter count in the wrapped model.
    pub n_params: usize,
}

impl<'a> FitModel for CountsModel<'a> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let transmission = self.transmission_model.evaluate(params)?;
        debug_assert_eq!(
            transmission.len(),
            self.flux.len(),
            "CountsModel: transmission length ({}) != flux length ({})",
            transmission.len(),
            self.flux.len(),
        );
        debug_assert_eq!(
            self.flux.len(),
            self.background.len(),
            "CountsModel: flux length ({}) != background length ({})",
            self.flux.len(),
            self.background.len(),
        );
        Ok(transmission
            .iter()
            .zip(self.flux.iter())
            .zip(self.background.iter())
            .map(|((&t, &f), &b)| f * t + b)
            .collect())
    }

    /// Analytical Jacobian: ∂Y/∂θ = flux · ∂T_inner/∂θ.
    ///
    /// Background is constant w.r.t. θ and drops out.
    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        // Recover inner transmission: T = (Y - background) / flux
        let t_inner: Vec<f64> = y_current
            .iter()
            .zip(self.flux.iter())
            .zip(self.background.iter())
            .map(|((&y, &f), &b)| if f.abs() > 1e-30 { (y - b) / f } else { 0.0 })
            .collect();
        let inner_jac =
            self.transmission_model
                .analytical_jacobian(params, free_param_indices, &t_inner)?;
        let n_free = free_param_indices.len();
        let mut jac = FlatMatrix::zeros(n_e, n_free);
        for i in 0..n_e {
            for j in 0..n_free {
                *jac.get_mut(i, j) = self.flux[i] * inner_jac.get(i, j);
            }
        }
        Some(jac)
    }
}

// ── ForwardModel implementation for CountsModel (Phase 1) ────────────────

impl<'a> crate::forward_model::ForwardModel for CountsModel<'a> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    // No analytical jacobian — uses finite differences (same as FitModel).

    fn n_data(&self) -> usize {
        self.flux.len()
    }

    fn n_params(&self) -> usize {
        self.n_params
    }
}

/// Fixed-flux counts model with optional α₁ / α₂ nuisance scaling of
/// signal and detector background.
///
/// **Retained for the research Fisher helper, not for production fitting.**
/// See [`CountsModel`] for the scope note — the production counts-KL
/// dispatch does not use this struct; it is reached only from
/// `evaluate_jacobian_and_fisher` (Epic #394 spatial-regularization
/// prototype) and from this module's `#[cfg(test)]` tests.
///
/// Given an effective count-ratio model `T_eff(θ)`, predicts:
///
///   Y(E) = α₁ · [F_open(E) · T_eff(θ)] + α₂ · B(E)
///
/// where `F_open` is the observed open-arm count response and `α₁` and
/// `α₂` are parameter-vector entries.
///
/// ## Physical count-response contract
///
/// This low-level wrapper cannot inspect `transmission_model` to determine how
/// instrument resolution was applied.  The inner model must already return
/// the physically valid effective sample/open count response
/// `R[Φ · T] / R[Φ]`.  It must never return a post-hoc broadened
/// transmission `R[T]` as a substitute.  In particular, do not directly wrap
/// a resolution-bearing
/// [`TransmissionFitModel`](crate::transmission_model::TransmissionFitModel),
/// because that model returns `R[T]`.  No-resolution models and custom models
/// that already compute the exact effective ratio remain valid.  Otherwise,
/// model the two response arms separately or call the guarded pipeline, which
/// rejects unsupported counts-plus-resolution combinations.  When resolution
/// is active, `flux` must be the matching resolved open-arm response `R[Φ]`.
///
/// ## Index invariant
///
/// `alpha1_index` / `alpha2_index` must NOT designate a parameter index
/// the transmission model reads. The wrapper cannot detect such a
/// collision through `dyn FitModel`, and the analytic Jacobian excludes
/// the scale indices from the inner free set — a collided parameter
/// would get only the scale contribution, silently omitting ∂T/∂p.
/// (Sharing ONE parameter between the two scale roles,
/// `alpha1_index == alpha2_index`, IS supported: the columns
/// accumulate.)
pub struct CountsBackgroundScaleModel<'a> {
    /// Underlying effective count-ratio model.
    ///
    /// See the struct-level physical count-response contract.  In particular,
    /// this must not be a post-hoc broadened transmission `R[T]`.
    pub transmission_model: &'a dyn FitModel,
    /// Open-arm flux response.
    pub flux: &'a [f64],
    /// Detector background spectrum.
    pub background: &'a [f64],
    /// Index of α₁ in the parameter vector.
    /// Must not be a parameter the transmission model reads (see struct docs).
    pub alpha1_index: usize,
    /// Index of α₂ in the parameter vector.
    /// Must not be a parameter the transmission model reads (see struct docs).
    pub alpha2_index: usize,
    /// Total parameter count in the wrapped model.
    pub n_params: usize,
}

impl<'a> FitModel for CountsBackgroundScaleModel<'a> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let transmission = self.transmission_model.evaluate(params)?;
        let alpha1 = params[self.alpha1_index];
        let alpha2 = params[self.alpha2_index];
        debug_assert_eq!(transmission.len(), self.flux.len());
        debug_assert_eq!(self.flux.len(), self.background.len());
        Ok(transmission
            .iter()
            .zip(self.flux.iter())
            .zip(self.background.iter())
            .map(|((&t, &f), &b)| alpha1 * f * t + alpha2 * b)
            .collect())
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        let n_free = free_param_indices.len();
        let alpha1 = params[self.alpha1_index];
        let alpha1_col = free_param_indices
            .iter()
            .position(|&i| i == self.alpha1_index);
        let alpha2_col = free_param_indices
            .iter()
            .position(|&i| i == self.alpha2_index);
        let inner_free: Vec<usize> = free_param_indices
            .iter()
            .copied()
            .filter(|&i| i != self.alpha1_index && i != self.alpha2_index)
            .collect();

        // Evaluate the inner transmission model directly instead of
        // reconstructing from y_current — reconstruction via
        // (y - alpha2*b)/(alpha1*f) is undefined when alpha1 ≈ 0.
        let t_inner = match self.transmission_model.evaluate(params) {
            Ok(t) => t,
            Err(_) => return None,
        };

        let inner_jac = if !inner_free.is_empty() {
            self.transmission_model
                .analytical_jacobian(params, &inner_free, &t_inner)
        } else {
            None
        };

        let mut jacobian = FlatMatrix::zeros(n_e, n_free);
        if let Some(ref ij) = inner_jac {
            let mut inner_col = 0;
            for (col, &fp) in free_param_indices.iter().enumerate() {
                if fp == self.alpha1_index || fp == self.alpha2_index {
                    continue;
                }
                for row in 0..n_e {
                    *jacobian.get_mut(row, col) = alpha1 * self.flux[row] * ij.get(row, inner_col);
                }
                inner_col += 1;
            }
        } else if !inner_free.is_empty() {
            return None;
        }

        // Accumulate (+=) rather than assign: the struct does not forbid
        // alpha1_index == alpha2_index, and evaluate() reads the aliased
        // parameter for both roles, so its derivative is the SUM of both
        // column contributions (f·t + b). With distinct indices each
        // column is touched once and += on the zeroed matrix is identical
        // to assignment.
        if let Some(col) = alpha1_col {
            for (row, (&f, &t)) in self.flux.iter().zip(t_inner.iter()).enumerate() {
                *jacobian.get_mut(row, col) += f * t;
            }
        }
        if let Some(col) = alpha2_col {
            for (row, &bg) in self.background.iter().enumerate() {
                *jacobian.get_mut(row, col) += bg;
            }
        }

        Some(jacobian)
    }
}

impl<'a> crate::forward_model::ForwardModel for CountsBackgroundScaleModel<'a> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn n_data(&self) -> usize {
        self.flux.len()
    }

    fn n_params(&self) -> usize {
        self.n_params
    }
}

/// KL-compatible background model for transmission data.
///
/// Given a transmission model T_inner(θ), predicts:
///
///   T_out(E) = T_inner(E) + b₀ + b₁/√E
///
/// where b₀ and b₁ are the additive background parameters at indices
/// `b0_index` and `b1_index` in the parameter vector.
///
/// Unlike `NormalizedTransmissionModel` (which uses `Anorm * T + BackA +
/// BackB/√E + BackC√E` with 4 free parameters), this model:
/// - Has only 2 background parameters (b₀, b₁), reducing overfitting risk
/// - Constrains b₀, b₁ ≥ 0 via parameter bounds (physical: background
///   adds counts, never subtracts), ensuring T_out > 0 for valid Poisson NLL
/// - Does NOT multiply T_inner by a normalization factor — normalization
///   is handled separately (nuisance estimation for counts, or pre-processing
///   for transmission data)
///
/// ## Gradient
///
/// - ∂T_out/∂nₖ = ∂T_inner/∂nₖ = -σₖ(E)·T_inner(E)  (same as bare model)
/// - ∂T_out/∂b₀ = 1
/// - ∂T_out/∂b₁ = 1/√E
///
/// ## Index invariant
///
/// `b0_index` / `b1_index` must NOT designate a parameter index the
/// inner model reads. The wrapper cannot detect such a collision
/// through `dyn FitModel`, and the analytic Jacobian excludes the
/// background indices from the inner free set — a collided parameter
/// would get only the background contribution, silently omitting
/// ∂T_inner/∂p. (Sharing ONE parameter between the two background
/// roles, `b0_index == b1_index`, IS supported: the columns
/// accumulate.)
pub struct TransmissionKLBackgroundModel<'a> {
    /// Underlying transmission model (density parameters only).
    pub inner: &'a dyn FitModel,
    /// Precomputed 1/√E for each energy bin.
    pub inv_sqrt_energies: Vec<f64>,
    /// Index of b₀ (constant background) in the parameter vector.
    /// Must not be a parameter the inner model reads (see struct docs).
    pub b0_index: usize,
    /// Index of b₁ (1/√E background) in the parameter vector.
    /// Must not be a parameter the inner model reads (see struct docs).
    pub b1_index: usize,
    /// Total parameter count in the wrapped model.
    pub n_params: usize,
}

impl<'a> FitModel for TransmissionKLBackgroundModel<'a> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let t_inner = self.inner.evaluate(params)?;
        let b0 = params[self.b0_index];
        let b1 = params[self.b1_index];
        Ok(t_inner
            .iter()
            .zip(self.inv_sqrt_energies.iter())
            .map(|(&t, &inv_sqrt_e)| t + b0 + b1 * inv_sqrt_e)
            .collect())
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let n_e = y_current.len();
        let n_free = free_param_indices.len();

        // Identify which free params are background vs inner model.
        let b0_col = free_param_indices.iter().position(|&i| i == self.b0_index);
        let b1_col = free_param_indices.iter().position(|&i| i == self.b1_index);

        // Inner model free params (those not b0 or b1).
        let inner_free: Vec<usize> = free_param_indices
            .iter()
            .copied()
            .filter(|&i| i != self.b0_index && i != self.b1_index)
            .collect();

        // Get inner model Jacobian for density columns.
        let inner_jac = if !inner_free.is_empty() {
            // Evaluate inner model at current params to get T_inner for y_current.
            let t_inner = self.inner.evaluate(params).ok()?;
            self.inner
                .analytical_jacobian(params, &inner_free, &t_inner)
        } else {
            None
        };

        let mut jacobian = FlatMatrix::zeros(n_e, n_free);

        // Fill inner model columns (density, temperature).
        // Inner Jacobian is the same as bare model — background doesn't
        // affect ∂T_inner/∂nₖ.
        if let Some(ij) = inner_jac.as_ref() {
            let mut inner_col = 0;
            for (col, &fp) in free_param_indices.iter().enumerate() {
                if fp == self.b0_index || fp == self.b1_index {
                    continue;
                }
                for row in 0..n_e {
                    *jacobian.get_mut(row, col) = ij.get(row, inner_col);
                }
                inner_col += 1;
            }
        } else if !inner_free.is_empty() {
            // Inner params are free but the inner model has no analytical
            // Jacobian — fall back to FD for the entire model.
            return None;
        }

        // Background columns. Accumulate (+=) rather than assign: the
        // struct does not forbid b0_index == b1_index, and evaluate()
        // reads the aliased parameter for both roles, so its derivative
        // is the SUM of both column contributions (1 + 1/√E). With
        // distinct indices each column is touched once and += on the
        // zeroed matrix is identical to assignment.
        if let Some(col) = b0_col {
            for row in 0..n_e {
                *jacobian.get_mut(row, col) += 1.0; // ∂T_out/∂b₀ = 1
            }
        }
        if let Some(col) = b1_col {
            for row in 0..n_e {
                *jacobian.get_mut(row, col) += self.inv_sqrt_energies[row]; // ∂T_out/∂b₁ = 1/√E
            }
        }

        Some(jacobian)
    }
}

impl<'a> crate::forward_model::ForwardModel for TransmissionKLBackgroundModel<'a> {
    fn predict(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        self.evaluate(params)
    }

    fn n_data(&self) -> usize {
        self.inv_sqrt_energies.len()
    }

    fn n_params(&self) -> usize {
        self.n_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple model: y = a * exp(-b * x)
    /// This mimics transmission: counts = flux * exp(-density * sigma)
    struct ExponentialModel {
        x: Vec<f64>,
        flux: Vec<f64>,
    }

    impl FitModel for ExponentialModel {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            let b = params[0]; // "density"
            Ok(self
                .x
                .iter()
                .zip(self.flux.iter())
                .map(|(&xi, &fi)| fi * (-b * xi).exp())
                .collect())
        }
    }

    #[test]
    fn test_counts_model() {
        struct ConstTransmission;
        impl FitModel for ConstTransmission {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![params[0]; 3])
            }
        }

        let t_model = ConstTransmission;
        let flux = [100.0, 200.0, 300.0];
        let background = [5.0, 10.0, 15.0];
        let counts_model = CountsModel {
            transmission_model: &t_model,
            flux: &flux,
            background: &background,
            n_params: 1,
        };

        // T = 0.5 → counts = flux*0.5 + background
        let result = counts_model.evaluate(&[0.5]).unwrap();
        assert!((result[0] - 55.0).abs() < 1e-10);
        assert!((result[1] - 110.0).abs() < 1e-10);
        assert!((result[2] - 165.0).abs() < 1e-10);
        assert_eq!(
            crate::forward_model::ForwardModel::n_params(&counts_model),
            1
        );
    }

    #[test]
    fn test_counts_background_scale_model() {
        struct ConstTransmission;
        impl FitModel for ConstTransmission {
            fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
                Ok(vec![params[0]; 3])
            }

            fn analytical_jacobian(
                &self,
                _params: &[f64],
                free_param_indices: &[usize],
                _y_current: &[f64],
            ) -> Option<FlatMatrix> {
                let mut jac = FlatMatrix::zeros(3, free_param_indices.len());
                for (col, &fp) in free_param_indices.iter().enumerate() {
                    if fp == 0 {
                        for row in 0..3 {
                            *jac.get_mut(row, col) = 1.0;
                        }
                    }
                }
                Some(jac)
            }
        }

        let t_model = ConstTransmission;
        let flux = [100.0, 200.0, 300.0];
        let background = [5.0, 10.0, 15.0];
        let counts_model = CountsBackgroundScaleModel {
            transmission_model: &t_model,
            flux: &flux,
            background: &background,
            alpha1_index: 1,
            alpha2_index: 2,
            n_params: 3,
        };

        let params = [0.5, 0.8, 1.5];
        let result = counts_model.evaluate(&params).unwrap();
        assert!((result[0] - 47.5).abs() < 1e-10);
        assert!((result[1] - 95.0).abs() < 1e-10);
        assert!((result[2] - 142.5).abs() < 1e-10);
        assert_eq!(
            crate::forward_model::ForwardModel::n_params(&counts_model),
            3
        );
    }

    #[test]
    fn test_transmission_kl_background_has_no_jacobian_when_inner_lacks_one() {
        let x: Vec<f64> = (0..40).map(|i| 1.0 + 0.25 * i as f64).collect();
        let inner = ExponentialModel {
            x: x.clone(),
            flux: vec![1000.0; x.len()],
        };
        let inv_sqrt_energies: Vec<f64> = x.iter().map(|&e| 1.0 / e.sqrt()).collect();
        let wrapped = TransmissionKLBackgroundModel {
            inner: &inner,
            inv_sqrt_energies,
            b0_index: 1,
            b1_index: 2,
            n_params: 3,
        };

        let true_params = vec![0.4, 20.0, 10.0];
        let y_obs = wrapped.evaluate(&true_params).unwrap();

        assert!(
            wrapped
                .analytical_jacobian(&true_params, &[0, 1, 2], &y_obs)
                .is_none(),
            "inner param free without inner analytical_jacobian must give None"
        );
        assert!(
            wrapped
                .analytical_jacobian(&true_params, &[0], &y_obs)
                .is_none(),
            "inner-only free set without inner analytical_jacobian must give None"
        );
    }

    #[test]
    fn test_transmission_kl_background_background_only_analytic_jacobian() {
        let x: Vec<f64> = (0..40).map(|i| 1.0 + 0.25 * i as f64).collect();
        let inner = ExponentialModel {
            x: x.clone(),
            flux: vec![1000.0; x.len()],
        };
        let inv_sqrt_energies: Vec<f64> = x.iter().map(|&e| 1.0 / e.sqrt()).collect();
        let wrapped = TransmissionKLBackgroundModel {
            inner: &inner,
            inv_sqrt_energies: inv_sqrt_energies.clone(),
            b0_index: 1,
            b1_index: 2,
            n_params: 3,
        };

        let true_params = vec![0.4, 20.0, 10.0];
        let y_obs = wrapped.evaluate(&true_params).unwrap();

        let jac = wrapped
            .analytical_jacobian(&true_params, &[1, 2], &y_obs)
            .expect("background-only free set must stay on the analytic path");
        for (row, &inv_sqrt_e) in inv_sqrt_energies.iter().enumerate() {
            assert!(
                (jac.get(row, 0) - 1.0).abs() < 1e-15,
                "∂T/∂b₀ at row {row} = {}, expected 1.0",
                jac.get(row, 0),
            );
            assert!(
                (jac.get(row, 1) - inv_sqrt_e).abs() < 1e-15,
                "∂T/∂b₁ at row {row} = {}, expected {inv_sqrt_e}",
                jac.get(row, 1),
            );
        }
    }

    /// Central finite-difference column for one parameter, computed
    /// straight from `evaluate` — an oracle independent of the model's
    /// analytic Jacobian path.
    fn fd_column(model: &dyn FitModel, params: &[f64], param_index: usize, h: f64) -> Vec<f64> {
        let mut plus = params.to_vec();
        plus[param_index] += h;
        let mut minus = params.to_vec();
        minus[param_index] -= h;
        let y_plus = model.evaluate(&plus).unwrap();
        let y_minus = model.evaluate(&minus).unwrap();
        y_plus
            .iter()
            .zip(y_minus.iter())
            .map(|(&p, &m)| (p - m) / (2.0 * h))
            .collect()
    }

    /// Aliased background indices (b0_index == b1_index): the analytic
    /// Jacobian must ACCUMULATE both roles' contributions (1 + 1/√E),
    /// matching finite differences — not overwrite one with the other.
    #[test]
    fn test_transmission_kl_background_aliased_indices_jacobian_matches_fd() {
        let x: Vec<f64> = (0..10).map(|i| 1.0 + 0.5 * i as f64).collect();
        let inner = ExponentialModel {
            x: x.clone(),
            flux: vec![1000.0; x.len()],
        };
        let inv_sqrt_energies: Vec<f64> = x.iter().map(|&e| 1.0 / e.sqrt()).collect();
        let wrapped = TransmissionKLBackgroundModel {
            inner: &inner,
            inv_sqrt_energies: inv_sqrt_energies.clone(),
            b0_index: 1,
            b1_index: 1, // deliberately aliased with b0
            n_params: 2,
        };

        let params = vec![0.4, 15.0];
        let y = wrapped.evaluate(&params).unwrap();
        // Inner param fixed, only the aliased background param free →
        // analytic path.
        let jac = wrapped
            .analytical_jacobian(&params, &[1], &y)
            .expect("background-only free set must stay on the analytic path");

        let fd = fd_column(&wrapped, &params, 1, 1e-6);
        for (row, (&fd_val, &inv_sqrt_e)) in fd.iter().zip(inv_sqrt_energies.iter()).enumerate() {
            let expected = 1.0 + inv_sqrt_e;
            assert!(
                (jac.get(row, 0) - expected).abs() < 1e-12,
                "aliased ∂/∂b at row {row}: analytic {}, expected {expected}",
                jac.get(row, 0),
            );
            assert!(
                (jac.get(row, 0) - fd_val).abs() < 1e-5,
                "aliased ∂/∂b at row {row}: analytic {}, FD {fd_val}",
                jac.get(row, 0),
            );
        }
    }

    /// Aliased scale indices (alpha1_index == alpha2_index) in the
    /// sibling counts model: same accumulate-not-overwrite requirement,
    /// derivative f·T + B against the finite-difference oracle.
    #[test]
    fn test_counts_background_scale_aliased_indices_jacobian_matches_fd() {
        let x: Vec<f64> = (0..10).map(|i| 1.0 + 0.5 * i as f64).collect();
        let inner = ExponentialModel {
            x: x.clone(),
            flux: vec![1.0; x.len()], // inner transmission in [0,1]
        };
        let flux: Vec<f64> = vec![1000.0; x.len()];
        let background: Vec<f64> = x.iter().map(|&e| 5.0 + e).collect();
        let wrapped = CountsBackgroundScaleModel {
            transmission_model: &inner,
            flux: &flux,
            background: &background,
            alpha1_index: 1,
            alpha2_index: 1, // deliberately aliased with alpha1
            n_params: 2,
        };

        let params = vec![0.4, 1.2];
        let y = wrapped.evaluate(&params).unwrap();
        let t_inner = inner.evaluate(&params).unwrap();
        // Inner param fixed, only the aliased scale param free →
        // analytic path.
        let jac = wrapped
            .analytical_jacobian(&params, &[1], &y)
            .expect("scale-only free set must stay on the analytic path");

        let fd = fd_column(&wrapped, &params, 1, 1e-6);
        for (row, &fd_val) in fd.iter().enumerate() {
            let expected = flux[row] * t_inner[row] + background[row];
            assert!(
                (jac.get(row, 0) - expected).abs() < 1e-9,
                "aliased ∂/∂α at row {row}: analytic {}, expected {expected}",
                jac.get(row, 0),
            );
            assert!(
                (jac.get(row, 0) - fd_val).abs() < 1e-3,
                "aliased ∂/∂α at row {row}: analytic {}, FD {fd_val}",
                jac.get(row, 0),
            );
        }
    }

    struct Decay {
        t: Vec<f64>,
        jacobian_factor: f64,
    }

    impl FitModel for Decay {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(self
                .t
                .iter()
                .map(|&t| params[0] * (-params[1] * t).exp())
                .collect())
        }

        fn analytical_jacobian(
            &self,
            params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let mut jacobian = FlatMatrix::zeros(self.t.len(), free_param_indices.len());
            for (row, &t) in self.t.iter().enumerate() {
                let e = (-params[1] * t).exp();
                for (col, &index) in free_param_indices.iter().enumerate() {
                    let slope = [e, -params[0] * t * e][index];
                    *jacobian.get_mut(row, col) = slope * self.jacobian_factor;
                }
            }
            Some(jacobian)
        }
    }

    fn decay(jacobian_factor: f64) -> Decay {
        Decay {
            t: (0..25).map(|i| 0.2 * f64::from(i)).collect(),
            jacobian_factor,
        }
    }

    fn decay_params(a: f64, b: f64, a_upper: f64) -> ParameterSet {
        ParameterSet::new(vec![
            FitParameter {
                name: "a".into(),
                value: a,
                lower: 0.0,
                upper: a_upper,
                fixed: false,
            },
            FitParameter::non_negative("b", b),
        ])
    }

    fn fit_decay(
        model: &Decay,
        observed: &[f64],
        start: (f64, f64),
        max_iter: usize,
    ) -> PoissonResult {
        let mut params = decay_params(start.0, start.1, f64::INFINITY);
        let config = PoissonConfig {
            max_iter,
            ..PoissonConfig::default()
        };
        poisson_fit(model, observed, &mut params, &config).unwrap()
    }

    #[test]
    fn half_deviance_matches_high_precision_values() {
        for (obs, mean, exact) in [
            (1_000_001.0, 1_000_000.0, 4.999_998_333_334_167e-7),
            (3.0, 2.5, 0.046_964_670_381_863_88),
            (0.0, 2.5, 2.5),
            (1.0e308, 9.0e307, 5.360_515_657_826_301e305),
            (1.7e308, 1.65e308, 7.500_373_544_579_622e304),
            (1.0, 1.0e-310, 712.801_378_828_154_2),
        ] {
            let got = half_deviance(obs, mean);
            assert!(
                ((got - exact) / exact).abs() < 1e-12,
                "{obs}, {mean}: {got}"
            );
        }
    }

    #[test]
    fn inputs_that_are_not_counts_for_this_model_are_refused() {
        let model = decay(1.0);
        let observed = model.evaluate(&[100.0, 1.5]).unwrap();
        let fit = |model: &dyn FitModel, observed: &[f64]| {
            poisson_fit(
                model,
                observed,
                &mut decay_params(50.0, 1.0, f64::INFINITY),
                &PoissonConfig::default(),
            )
        };
        let mut bad = observed.clone();
        bad[3] = f64::NAN;
        assert!(fit(&model, &bad).is_err());
        bad[3] = -1.0;
        assert!(fit(&model, &bad).is_err());
        assert!(fit(&model, &observed[1..]).is_err());
        let empty = Decay {
            t: vec![],
            jacobian_factor: 1.0,
        };
        assert!(fit(&empty, &[]).is_err());
        let no_jacobian = ExponentialModel {
            x: model.t.clone(),
            flux: vec![100.0; model.t.len()],
        };
        assert!(fit(&no_jacobian, &observed).is_err());
    }

    #[test]
    fn a_start_with_a_zero_prediction_is_not_converged() {
        let model = decay(1.0);
        let observed = model.evaluate(&[100.0, 1.5]).unwrap();
        let result = fit_decay(&model, &observed, (0.0, 1.0), 200);
        assert!(!result.converged && result.iterations == 0, "{result:?}");
    }

    #[test]
    fn a_non_finite_slope_ends_the_fit_unconverged() {
        let model = decay(f64::NAN);
        let observed = decay(1.0).evaluate(&[100.0, 1.5]).unwrap();
        let result = fit_decay(&model, &observed, (50.0, 1.0), 200);
        assert!(
            !result.converged && result.uncertainties.is_none(),
            "{result:?}"
        );
    }

    #[test]
    fn a_fit_whose_step_cannot_lower_the_deviance_is_not_converged() {
        let model = decay(-1.0);
        let observed = decay(1.0).evaluate(&[100.0, 1.5]).unwrap();
        let result = fit_decay(&model, &observed, (50.0, 1.0), 200);
        assert!(
            !result.converged && result.iterations == 0 && result.uncertainties.is_none(),
            "{result:?}"
        );
    }

    #[test]
    fn a_start_outside_the_bounds_ends_inside_them() {
        let model = decay(1.0);
        let observed = model.evaluate(&[100.0, 1.5]).unwrap();
        let mut params = decay_params(100.0, 1.5, 80.0);
        let result =
            poisson_fit(&model, &observed, &mut params, &PoissonConfig::default()).unwrap();
        assert!(result.converged && result.params[0] == 80.0, "{result:?}");
        assert_eq!(result.on_bound, vec![true, false]);
    }

    #[test]
    fn a_fit_that_reaches_the_minimum_on_its_last_allowed_step_has_converged() {
        let model = decay(1.0);
        let observed = model.evaluate(&[100.0, 1.5]).unwrap();
        let steps = fit_decay(&model, &observed, (20.0, 0.3), 200).iterations;
        assert!(steps > 1);
        assert!(fit_decay(&model, &observed, (20.0, 0.3), steps).converged);
        assert!(!fit_decay(&model, &observed, (20.0, 0.3), steps - 1).converged);
    }

    struct Scaled {
        x: Vec<f64>,
        slope: f64,
        tiny: f64,
    }

    impl FitModel for Scaled {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(self
                .x
                .iter()
                .map(|&x| 100.0 * (self.slope * params[0] * x).exp() + self.tiny * params[1] * x)
                .collect())
        }

        fn analytical_jacobian(
            &self,
            params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let mut jacobian = FlatMatrix::zeros(self.x.len(), free_param_indices.len());
            for (row, &x) in self.x.iter().enumerate() {
                let slopes = [
                    100.0 * self.slope * x * (self.slope * params[0] * x).exp(),
                    self.tiny * x,
                ];
                for (col, &index) in free_param_indices.iter().enumerate() {
                    *jacobian.get_mut(row, col) = slopes[index];
                }
            }
            Some(jacobian)
        }
    }

    fn fit_scaled(slope: f64, tiny: f64) -> PoissonResult {
        let model = Scaled {
            x: vec![1.0, 2.0, 3.0],
            slope,
            tiny,
        };
        let observed = model.evaluate(&[0.3 / slope, 0.0]).unwrap();
        let mut params = ParameterSet::new(vec![
            FitParameter::unbounded("theta", 0.0),
            FitParameter::unbounded("weak", 0.0),
        ]);
        poisson_fit(&model, &observed, &mut params, &PoissonConfig::default()).unwrap()
    }

    #[test]
    fn a_slope_too_large_to_square_is_fitted() {
        let result = fit_scaled(1.0e160, 1.0);
        assert!(
            result.converged && result.deviance < NEWTON_DECREMENT_TOL,
            "{result:?}"
        );
    }

    #[test]
    fn a_slope_too_small_to_square_does_not_fake_convergence() {
        let result = fit_scaled(1.0, 1.0e-310);
        assert!(
            !result.converged || result.deviance < NEWTON_DECREMENT_TOL,
            "{result:?}"
        );
    }

    struct Line {
        x: Vec<f64>,
    }

    impl FitModel for Line {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(self.x.iter().map(|&x| 1.0 + params[0] * x).collect())
        }

        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let mut jacobian = FlatMatrix::zeros(self.x.len(), free_param_indices.len());
            jacobian.data.copy_from_slice(&self.x);
            Some(jacobian)
        }
    }

    #[test]
    fn predictions_stay_positive_where_nothing_was_counted() {
        let model = Line {
            x: vec![1.0, 2.0, 3.0],
        };
        let mut params = ParameterSet::new(vec![FitParameter::unbounded("a", 0.0)]);
        let result =
            poisson_fit(&model, &[0.0; 3], &mut params, &PoissonConfig::default()).unwrap();
        let predicted = model.evaluate(&result.params).unwrap();
        assert!(predicted.iter().all(|&mean| mean > 0.0), "{result:?}");
    }

    struct WrongShape;

    impl FitModel for WrongShape {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(vec![params[0]; 3])
        }

        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            Some(FlatMatrix::zeros(2, free_param_indices.len()))
        }
    }

    #[test]
    fn a_jacobian_of_the_wrong_shape_is_refused() {
        let mut params = ParameterSet::new(vec![FitParameter::non_negative("a", 1.0)]);
        let result = poisson_fit(
            &WrongShape,
            &[1.0; 3],
            &mut params,
            &PoissonConfig::default(),
        );
        assert!(
            matches!(result, Err(FittingError::LengthMismatch { .. })),
            "{result:?}"
        );
    }

    #[test]
    fn inverted_or_nan_bounds_are_refused() {
        let model = decay(1.0);
        let observed = model.evaluate(&[100.0, 1.5]).unwrap();
        for (lower, upper) in [(1.0, 0.0), (f64::NAN, 1.0), (0.0, f64::NAN)] {
            let mut params = ParameterSet::new(vec![
                FitParameter {
                    name: "a".into(),
                    value: 50.0,
                    lower,
                    upper,
                    fixed: false,
                },
                FitParameter::non_negative("b", 1.0),
            ]);
            let result = poisson_fit(&model, &observed, &mut params, &PoissonConfig::default());
            assert!(
                matches!(result, Err(FittingError::InvalidConfig(_))),
                "{lower}, {upper}"
            );
        }
    }

    struct Additive;

    impl FitModel for Additive {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(vec![2.0 * params[0], 1.0 - params[0]])
        }

        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let mut jacobian = FlatMatrix::zeros(2, free_param_indices.len());
            jacobian.data.copy_from_slice(&[2.0, -1.0]);
            Some(jacobian)
        }
    }

    #[test]
    fn a_bin_predicted_zero_keeps_its_slope_in_the_gradient() {
        let mut params = ParameterSet::new(vec![FitParameter {
            name: "theta".into(),
            value: 0.0,
            lower: 0.0,
            upper: 1.0,
            fixed: false,
        }]);
        let result = poisson_fit(
            &Additive,
            &[0.0, 0.0],
            &mut params,
            &PoissonConfig::default(),
        )
        .unwrap();
        assert!(
            result.converged && result.params[0] == 0.0 && result.on_bound == vec![true],
            "{result:?}"
        );
    }

    struct Line2 {
        offset: [f64; 2],
        slope: [f64; 2],
    }

    impl FitModel for Line2 {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok((0..2)
                .map(|i| self.offset[i] + self.slope[i] * params[0])
                .collect())
        }

        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let mut jacobian = FlatMatrix::zeros(2, free_param_indices.len());
            jacobian.data.copy_from_slice(&self.slope);
            Some(jacobian)
        }
    }

    fn bounded(value: f64, upper: f64) -> ParameterSet {
        ParameterSet::new(vec![FitParameter {
            name: "theta".into(),
            value,
            lower: 0.0,
            upper,
            fixed: false,
        }])
    }

    #[test]
    fn a_zero_bin_slope_enters_the_convergence_test() {
        let model = Line2 {
            offset: [0.0, 4.0],
            slope: [1.0, -2.0],
        };
        let result = poisson_fit(
            &model,
            &[0.0, 2.0],
            &mut bounded(0.0, 1.5),
            &PoissonConfig::default(),
        )
        .unwrap();
        assert!(
            result.converged && result.iterations == 0 && result.params[0] == 0.0,
            "{result:?}"
        );
    }

    #[test]
    fn a_non_finite_slope_in_a_zero_bin_ends_the_fit_unconverged() {
        let model = Line2 {
            offset: [0.0, 1.0],
            slope: [f64::NAN, 1.0],
        };
        let result = poisson_fit(
            &model,
            &[0.0, 1.0],
            &mut bounded(0.0, f64::INFINITY),
            &PoissonConfig::default(),
        )
        .unwrap();
        assert!(!result.converged, "{result:?}");
    }

    struct Wall;

    impl FitModel for Wall {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            let x = params[0];
            Ok(vec![(100.0 - x + (x - 80.0).max(0.0).powi(2)).exp()])
        }

        fn analytical_jacobian(
            &self,
            params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            let x = params[0];
            let mean = (100.0 - x + (x - 80.0).max(0.0).powi(2)).exp();
            let mut jacobian = FlatMatrix::zeros(1, free_param_indices.len());
            jacobian.data[0] = mean * (2.0 * (x - 80.0).max(0.0) - 1.0);
            Some(jacobian)
        }
    }

    #[test]
    fn damping_recovers_after_many_accepted_steps_to_reach_the_minimum() {
        let mut params = ParameterSet::new(vec![FitParameter::unbounded("x", 0.0)]);
        let result = poisson_fit(&Wall, &[0.0], &mut params, &PoissonConfig::default()).unwrap();
        assert!((result.params[0] - 80.5).abs() < 1e-3, "{result:?}");
    }

    #[test]
    fn non_finite_parameter_values_are_refused() {
        let model = decay(1.0);
        let observed = model.evaluate(&[100.0, 1.5]).unwrap();
        for (a, b) in [(f64::NAN, 1.0), (50.0, f64::INFINITY)] {
            let mut params = ParameterSet::new(vec![
                FitParameter::non_negative("a", a),
                FitParameter::fixed("b", b),
            ]);
            let result = poisson_fit(&model, &observed, &mut params, &PoissonConfig::default());
            assert!(
                matches!(result, Err(FittingError::InvalidConfig(_))),
                "{a}, {b}"
            );
        }
    }
}
