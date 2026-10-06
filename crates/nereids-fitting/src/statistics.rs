//! Consistency of a fit with its prior, and agreement between two
//! estimates.

use std::f64::consts::PI;

use faer::{Mat, Side};

use crate::error::FittingError;
use crate::lm::FlatMatrix;
use crate::poisson::{NEWTON_DECREMENT_TOL, Prior, symmetric};

const TERM_SHIFT: f64 = 0.1;

/// A χ² test: the statistic `q`, its degrees of freedom `dof`, and the
/// probability `p` of a `q` at least as large under the hypothesis tested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Consistency {
    pub q: f64,
    pub dof: usize,
    pub p: f64,
}

impl Consistency {
    /// The test of `q` against `χ²` with `dof` degrees of freedom.
    ///
    /// # Errors
    /// `FittingError::InvalidConfig` if `q` is negative or not a number, or
    /// `dof` is 0.
    pub fn new(q: f64, dof: usize) -> Result<Self, FittingError> {
        if q.is_nan() || q < 0.0 || dof == 0 {
            return Err(FittingError::InvalidConfig(format!(
                "a χ² test has a statistic of 0 or more on 1 or more degrees of freedom; got \
                 {q} on {dof}"
            )));
        }
        Ok(Self {
            q,
            dof,
            p: chi_squared_survival(q, dof),
        })
    }
}

fn chi_squared_survival(q: f64, dof: usize) -> f64 {
    if q == f64::INFINITY {
        return 0.0;
    }
    let x = q / 2.0;
    let (mut sum, mut log_term, offset) = if dof.is_multiple_of(2) {
        (0.0, -x, 1.0)
    } else {
        (
            libm::erfc(x.sqrt()),
            std::f64::consts::LN_2 - x + 0.5 * (x / PI).ln(),
            1.5,
        )
    };
    for n in 0..dof / 2 {
        sum += log_term.exp();
        log_term += x.ln() - (n as f64 + offset).ln();
    }
    sum
}

/// The consistency of the fitted values `estimate` of a `prior`'s
/// parameters with it, in the prior's order, given the fit's covariance
/// `posterior` of them.
///
/// With `C = LLᵀ` the prior covariance, `z = L⁻¹(estimate − mean)` and
/// `I − L⁻¹·posterior·L⁻ᵀ = Σᵢ rᵢuᵢuᵢᵀ`, `rᵢ` is the share of the prior's
/// variance along `uᵢ` that the counts remove, and `q = Σᵢ (uᵢᵀz)²/rᵢ` over
/// the `dof` directions with `rᵢ ≥ 8e-4`.  If the prior and the counts' model
/// are right and `estimate` and `posterior` are the fit's without its bounds
/// ([`Unbounded`](crate::poisson::Unbounded)), `q` follows `χ²` with `dof`
/// degrees of freedom, and `p = P(χ²_dof ≥ q)`.
///
/// `None` when `posterior` is not finite or no direction has `rᵢ ≥ 8e-4`.
///
/// # Errors
/// `FittingError::LengthMismatch` if `estimate` or `posterior` does not
/// match the prior's parameters; `FittingError::InvalidConfig` if the prior's
/// mean is not finite or a measured sd not finite and positive, `estimate` is
/// not finite while `posterior` is, or `posterior` is not symmetric to 1e-12
/// of `√(Σᵢᵢ Σⱼⱼ)` or not within the prior, with some `rᵢ` below `−8e-4` or
/// above `1 + 8e-4`, or `q` is not a number, as overflow in whitening can
/// give; `FittingError::EvaluationFailed` if a decomposition fails.
pub fn consistency(
    prior: &Prior,
    estimate: &[f64],
    posterior: &FlatMatrix,
) -> Result<Option<Consistency>, FittingError> {
    let n = prior.parameters.len();
    for (expected, actual, field) in [
        (n, estimate.len(), "estimate"),
        (n, posterior.nrows, "posterior rows"),
        (n, posterior.ncols, "posterior columns"),
        (n * n, posterior.data.len(), "posterior entries"),
    ] {
        if actual != expected {
            return Err(FittingError::LengthMismatch {
                expected,
                actual,
                field,
            });
        }
    }
    if !prior.is_valid() {
        return Err(FittingError::InvalidConfig(format!(
            "the consistency of a prior needs a finite mean and finite positive sds; got \
             {prior:?}"
        )));
    }
    let posterior = Mat::from_fn(n, n, |a, b| posterior.get(a, b));
    if !(0..n).all(|a| (0..n).all(|b| posterior[(a, b)].is_finite())) {
        return Ok(None);
    }
    if !estimate.iter().all(|x| x.is_finite()) {
        return Err(FittingError::InvalidConfig(format!(
            "the consistency of a prior needs finite estimates; got {estimate:?}"
        )));
    }
    if !symmetric(n, |a, b| posterior[(a, b)]) {
        return Err(FittingError::InvalidConfig(format!(
            "a posterior covariance must be symmetric; got {posterior:?}"
        )));
    }
    let mut z: Vec<f64> = estimate
        .iter()
        .zip(&prior.mean)
        .map(|(x, m)| x - m)
        .collect();
    prior.whiten(&mut z);
    let half: Vec<Vec<f64>> = (0..n)
        .map(|b| {
            let mut column: Vec<f64> = (0..n).map(|a| posterior[(a, b)]).collect();
            prior.whiten(&mut column);
            column
        })
        .collect();
    let whitened: Vec<Vec<f64>> = (0..n)
        .map(|a| {
            let mut row: Vec<f64> = half.iter().map(|column| column[a]).collect();
            prior.whiten(&mut row);
            row
        })
        .collect();
    let shrinkage = Mat::from_fn(n, n, |a, b| {
        f64::from(u8::from(a == b)) - 0.5 * (whitened[a][b] + whitened[b][a])
    });
    let eigen = shrinkage
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| FittingError::EvaluationFailed(format!("{e:?}")))?;
    let floor = (2.0 * (2.0 * NEWTON_DECREMENT_TOL).sqrt() / TERM_SHIFT).powi(2);
    let shares = eigen.S().column_vector();
    if (0..n).any(|d| !(-floor..=1.0 + floor).contains(&shares[d])) {
        return Err(FittingError::InvalidConfig(format!(
            "a posterior covariance must lie within the prior's; the shares of the prior's \
             variance it removes are {:?}",
            (0..n).map(|d| shares[d]).collect::<Vec<f64>>()
        )));
    }
    let (mut q, mut dof) = (0.0, 0);
    for d in 0..n {
        let share = shares[d];
        if share >= floor {
            let projection: f64 = (0..n).map(|a| eigen.U()[(a, d)] * z[a]).sum();
            q += projection.powi(2) / share;
            dof += 1;
        }
    }
    (dof > 0).then(|| Consistency::new(q, dof)).transpose()
}

/// The agreement of two independent estimates `a` and `b` of the same
/// parameters, each a mean with its covariance:
/// `q = dᵀ(C_a + C_b)⁻¹d`, with `d` the difference of the means, follows `χ²`
/// with as many degrees of freedom as parameters when both estimate the same
/// values.
///
/// # Errors
/// `FittingError::InvalidConfig` if `a` and `b` are over different
/// parameters, either has a mean that is not finite or a measured sd not
/// finite and positive, or `q` is not a number, as overflow in whitening can
/// give; `FittingError::EvaluationFailed` if `C_a + C_b` is
/// not positive definite in floating point.
pub fn agreement(a: &Prior, b: &Prior) -> Result<Consistency, FittingError> {
    if a.parameters != b.parameters || !a.is_valid() || !b.is_valid() {
        return Err(FittingError::InvalidConfig(format!(
            "two valid estimates of the same parameters are compared; got {a:?} and {b:?}"
        )));
    }
    let k = a.parameters.len();
    let mut sum = FlatMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            *sum.get_mut(i, j) = a.covariance(i, j) + b.covariance(i, j);
        }
    }
    let joint = Prior::factored(&a.parameters, &b.mean, &sum);
    if !joint.is_valid() {
        return Err(FittingError::EvaluationFailed(format!(
            "the sum of the two covariances is not positive definite: {sum:?}"
        )));
    }
    let mut d: Vec<f64> = a.mean.iter().zip(&b.mean).map(|(x, y)| x - y).collect();
    joint.whiten(&mut d);
    Consistency::new(d.iter().map(|w| w * w).sum(), k)
}

/// The part of `covariance` that the quantities at `shared` carry,
/// `C_·s C_ss⁻¹ C_s·`: the whole covariance between two quantities that
/// depend on each other only through those at `shared`.  A row of NaN in
/// `covariance` is a row of NaN in it.
///
/// `None` when the covariance of the quantities at `shared` is not finite and
/// positive definite.
///
/// # Panics
/// If an index in `shared` is not a row of `covariance`.
pub fn common_mode(covariance: &FlatMatrix, shared: &[usize]) -> Option<FlatMatrix> {
    let k = shared.len();
    let mut block = FlatMatrix::zeros(k, k);
    for (a, &i) in shared.iter().enumerate() {
        for (b, &j) in shared.iter().enumerate() {
            *block.get_mut(a, b) = covariance.get(i, j);
        }
    }
    let factor = Prior::factored(shared, &vec![0.0; k], &block);
    if !factor.is_valid() {
        return None;
    }
    let n = covariance.nrows;
    let whitened: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = shared.iter().map(|&s| covariance.get(i, s)).collect();
            factor.whiten(&mut row);
            row
        })
        .collect();
    let mut common = FlatMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *common.get_mut(i, j) = whitened[i]
                .iter()
                .zip(&whitened[j])
                .map(|(x, y)| x * y)
                .sum();
        }
    }
    Some(common)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chi_squared_survival_matches_scipy() {
        for (dof, q, p) in [
            (1, 3.841_458_820_694_120_5, 0.05),
            (2, 5.991_464_547_107_98, 0.05),
            (3, 7.814_727_903_251_176_5, 0.05),
            (4, 9.487_729_036_781_154, 0.05),
            (5, 11.070_497_693_516_351, 0.05),
            (6, 12.591_587_243_743_977, 0.05),
            (1, 2.0, 0.157_299_207_050_281_05),
            (5, 40.0, 1.493_367_900_050_396e-7),
            (4, 0.3, 0.989_814_172_888_816_5),
        ] {
            let survival = chi_squared_survival(q, dof);
            assert!((survival / p - 1.0).abs() <= 1e-13, "{dof} {q}: {survival}");
        }
        for (dof, q, p) in [
            (1500, 1500.0, 0.495_144_193_335_767_9),
            (1501, 1501.0, 0.495_145_811_152_950_14),
            (1500, 1700.0, 2.217_079_970_428_596_7e-4),
        ] {
            let survival = chi_squared_survival(q, dof);
            assert!((survival / p - 1.0).abs() <= 1e-10, "{dof} {q}: {survival}");
        }
        for dof in 1..=6 {
            assert_eq!(chi_squared_survival(0.0, dof), 1.0);
            assert_eq!(chi_squared_survival(2000.0, dof), 0.0);
            assert_eq!(chi_squared_survival(f64::INFINITY, dof), 0.0);
        }
    }

    fn two_by_two(data: [f64; 4]) -> FlatMatrix {
        FlatMatrix {
            data: data.to_vec(),
            nrows: 2,
            ncols: 2,
        }
    }

    fn rotated(shares: [f64; 2], angle: f64) -> [[f64; 2]; 2] {
        let (s, c) = angle.sin_cos();
        let u = [[c, -s], [s, c]];
        let mut m = [[0.0; 2]; 2];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = (0..2).map(|d| u[i][d] * shares[d] * u[j][d]).sum();
            }
        }
        m
    }

    #[test]
    fn the_statistic_weighs_each_direction_by_the_variance_the_counts_remove() {
        let factor = [[2.0, 0.0], [1.0, 2.0]];
        let covariance = FlatMatrix {
            data: vec![4.0, 2.0, 2.0, 5.0],
            nrows: 2,
            ncols: 2,
        };
        let prior = Prior::correlated(&[0, 1], &[3.0, -1.0], &covariance).unwrap();
        let angle: f64 = 0.5;
        let (s, c) = angle.sin_cos();
        let z = [1.0, 3.0];
        let estimate: Vec<f64> = (0..2)
            .map(|i| [3.0, -1.0][i] + (0..2).map(|j| factor[i][j] * z[j]).sum::<f64>())
            .collect();
        let posterior = |shares: [f64; 2]| {
            let m = rotated(shares, angle);
            let mut data = vec![0.0; 4];
            for i in 0..2 {
                for j in 0..2 {
                    data[i * 2 + j] = (0..2)
                        .flat_map(|a| (0..2).map(move |b| (a, b)))
                        .map(|(a, b)| {
                            factor[i][a] * (f64::from(u8::from(a == b)) - m[a][b]) * factor[j][b]
                        })
                        .sum();
                }
            }
            FlatMatrix {
                data,
                nrows: 2,
                ncols: 2,
            }
        };
        let along_first = c * z[0] + s * z[1];
        let both = consistency(&prior, &estimate, &posterior([0.5, 0.25]))
            .unwrap()
            .unwrap();
        let along_second = -s * z[0] + c * z[1];
        let q = along_first.powi(2) / 0.5 + along_second.powi(2) / 0.25;
        assert_eq!(both.dof, 2);
        assert!((both.q / q - 1.0).abs() <= 1e-12, "{} vs {q}", both.q);
        assert!((both.p / (-q / 2.0).exp() - 1.0).abs() <= 1e-12);
        let one = consistency(&prior, &estimate, &posterior([0.5, 1e-4]))
            .unwrap()
            .unwrap();
        assert_eq!(one.dof, 1);
        let q = along_first.powi(2) / 0.5;
        assert!((one.q / q - 1.0).abs() <= 1e-12, "{} vs {q}", one.q);
        assert!(
            consistency(&prior, &estimate, &posterior([1e-4, 1e-4]))
                .unwrap()
                .is_none()
        );
        let mut undetermined = posterior([0.5, 0.25]);
        undetermined.data[3] = f64::NAN;
        assert!(
            consistency(&prior, &estimate, &undetermined)
                .unwrap()
                .is_none()
        );
        assert!(consistency(&prior, &estimate[..1], &posterior([0.5, 0.25])).is_err());
        let mut truncated = posterior([0.5, 0.25]);
        truncated.data.pop();
        assert!(consistency(&prior, &estimate, &truncated).is_err());
        let nan = [estimate[0], f64::NAN];
        assert!(consistency(&prior, &nan, &posterior([0.5, 0.25])).is_err());
        assert!(consistency(&prior, &nan, &undetermined).unwrap().is_none());
        let negative = Prior::measured(0, 0.0, -1.0);
        let unit = FlatMatrix {
            data: vec![0.5],
            nrows: 1,
            ncols: 1,
        };
        assert!(consistency(&negative, &[1.0], &unit).is_err());
        let unknown = FlatMatrix {
            data: vec![f64::NAN],
            ..unit
        };
        assert!(consistency(&negative, &[1.0], &unknown).is_err());
        let identity =
            Prior::correlated(&[0, 1], &[0.0, 0.0], &two_by_two([1.0, 0.0, 0.0, 1.0])).unwrap();
        for impossible in [
            [0.5, 5.0, -5.0, 0.5],
            [1.5, 0.0, 0.0, 0.5],
            [-1.0, 0.0, 0.0, 0.5],
        ] {
            let result = consistency(&identity, &[2.0, 1.0], &two_by_two(impossible));
            assert!(result.is_err(), "{impossible:?}: {result:?}");
        }
        let far = consistency(
            &identity,
            &[1e155, 1e155],
            &two_by_two([0.5, 0.0, 0.0, 0.5]),
        )
        .unwrap()
        .unwrap();
        assert_eq!((far.q, far.p), (f64::INFINITY, 0.0));
    }

    #[test]
    fn two_estimates_agree_by_their_difference_over_the_sum_of_their_covariances() {
        let a = Prior::correlated(&[3, 7], &[1.0, 2.0], &two_by_two([4.0, 1.0, 1.0, 2.0])).unwrap();
        let b =
            Prior::correlated(&[3, 7], &[2.0, 0.5], &two_by_two([1.0, -0.5, -0.5, 3.0])).unwrap();
        let (sum, d) = ([5.0, 0.5, 0.5, 5.0], [-1.0, 1.5]);
        let determinant = sum[0] * sum[3] - sum[1] * sum[2];
        let q = (d[0] * d[0] * sum[3] - 2.0 * d[0] * d[1] * sum[1] + d[1] * d[1] * sum[0])
            / determinant;
        let result = agreement(&a, &b).unwrap();
        assert_eq!(result.dof, 2);
        assert!((result.q / q - 1.0).abs() <= 1e-14, "{} vs {q}", result.q);
        assert!((result.p / (-q / 2.0).exp() - 1.0).abs() <= 1e-14);
        let other =
            Prior::correlated(&[3, 8], &[2.0, 0.5], &two_by_two([1.0, 0.0, 0.0, 3.0])).unwrap();
        assert!(agreement(&a, &other).is_err());
        let invalid = Prior::measured(3, 0.0, -1.0);
        let one = Prior::measured(3, 0.0, 1.0);
        assert!(agreement(&invalid, &one).is_err());
        let far =
            |mean| Prior::correlated(&[3, 7], &[mean, 0.0], &two_by_two([1.0, 0.0, 0.0, 1.0]));
        assert!(agreement(&far(f64::MAX).unwrap(), &far(-f64::MAX).unwrap()).is_err());
        for (q, dof) in [(-1.0, 2), (f64::NAN, 2), (1.0, 0)] {
            assert!(Consistency::new(q, dof).is_err(), "{q} {dof}");
        }
    }
}
