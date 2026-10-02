//! Consistency of a fit with its prior.

use std::f64::consts::PI;

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

use crate::error::FittingError;
use crate::lm::FlatMatrix;
use crate::poisson::{NEWTON_DECREMENT_TOL, Prior, symmetric};

const TERM_SHIFT: f64 = 0.1;

/// Whether counts accept a prior: the statistic `q`, its degrees of freedom
/// `dof`, and the probability `p` of a `q` at least as large if the prior is
/// right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Consistency {
    pub q: f64,
    pub dof: usize,
    pub p: f64,
}

pub(crate) fn chi_squared_survival(q: f64, dof: usize) -> f64 {
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
/// parameters with it, in the prior's order, given which ended `on_bound`
/// and the fit's covariance `posterior` of them, conditional on those.
///
/// A parameter on a bound is held there: the rest have the prior
/// conditioned on it, with mean `m_R + C_RB C_BB⁻¹(θ_B − m_B)` and
/// covariance `C_RR − C_RB C_BB⁻¹ C_BR`, and the rows and columns of
/// `posterior` of a held parameter are not read.  With `C = LLᵀ` the rest's
/// prior covariance, `z = L⁻¹(estimate − mean)` and
/// `I − L⁻¹·posterior·L⁻ᵀ = Σᵢ rᵢuᵢuᵢᵀ` over the rest, `rᵢ` is the share of
/// the prior's variance along `uᵢ` that the counts remove, and
/// `q = Σᵢ (uᵢᵀz)²/rᵢ` over the `dof` directions with `rᵢ ≥ 8e-4`.  If the
/// prior and the counts' model are right, `q` follows `χ²` with `dof`
/// degrees of freedom, and `p = P(χ²_dof ≥ q)`.
///
/// `None` when every parameter is on a bound, `posterior` is not finite
/// over the rest, or no direction has `rᵢ ≥ 8e-4`.
///
/// # Errors
/// `FittingError::LengthMismatch` if `estimate`, `on_bound` or `posterior`
/// does not match the prior's parameters; `FittingError::InvalidConfig` if
/// `estimate` is not finite, the prior's mean is not finite or a measured sd
/// not finite and positive, or `posterior` over the rest is not symmetric to
/// 1e-12 of `√(Σᵢᵢ Σⱼⱼ)` or not within the prior there, with some `rᵢ` below
/// `−8e-4` or above `1 + 8e-4`; `FittingError::EvaluationFailed` if a
/// decomposition fails.
pub fn consistency(
    prior: &Prior,
    estimate: &[f64],
    on_bound: &[bool],
    posterior: &FlatMatrix,
) -> Result<Option<Consistency>, FittingError> {
    let k = prior.parameters.len();
    for (expected, actual, field) in [
        (k, estimate.len(), "estimate"),
        (k, on_bound.len(), "on_bound"),
        (k, posterior.nrows, "posterior rows"),
        (k, posterior.ncols, "posterior columns"),
        (k * k, posterior.data.len(), "posterior entries"),
    ] {
        if actual != expected {
            return Err(FittingError::LengthMismatch {
                expected,
                actual,
                field,
            });
        }
    }
    if !(prior.is_valid() && estimate.iter().all(|x| x.is_finite())) {
        return Err(FittingError::InvalidConfig(format!(
            "the consistency of a prior needs a finite mean, finite positive sds and \
             finite estimates; got {prior:?} and {estimate:?}"
        )));
    }
    let rest: Vec<usize> = (0..k).filter(|&i| !on_bound[i]).collect();
    let n = rest.len();
    let posterior = Mat::from_fn(n, n, |a, b| posterior.get(rest[a], rest[b]));
    if n == 0 || !(0..n).all(|a| (0..n).all(|b| posterior[(a, b)].is_finite())) {
        return Ok(None);
    }
    if !symmetric(n, |a, b| posterior[(a, b)]) {
        return Err(FittingError::InvalidConfig(format!(
            "a posterior covariance must be symmetric; got {posterior:?}"
        )));
    }
    let prior = conditioned(prior, estimate, &rest)?;
    let mut z: Vec<f64> = rest
        .iter()
        .zip(&prior.mean)
        .map(|(&i, m)| estimate[i] - m)
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
    Ok((dof > 0).then(|| Consistency {
        q,
        dof,
        p: chi_squared_survival(q, dof),
    }))
}

fn conditioned(prior: &Prior, estimate: &[f64], rest: &[usize]) -> Result<Prior, FittingError> {
    let k = prior.parameters.len();
    let held: Vec<usize> = (0..k).filter(|i| !rest.contains(i)).collect();
    let covariance = |i: usize, j: usize| -> f64 {
        (0..=i.min(j))
            .map(|l| prior.factor[i * k + l] * prior.factor[j * k + l])
            .sum()
    };
    let block = |rows: &[usize], columns: &[usize]| {
        Mat::from_fn(rows.len(), columns.len(), |a, b| {
            covariance(rows[a], columns[b])
        })
    };
    let cross = block(&held, rest);
    let shift = Mat::from_fn(held.len(), 1, |a, _| {
        estimate[held[a]] - prior.mean[held[a]]
    });
    let held_block = block(&held, &held)
        .llt(Side::Lower)
        .map_err(|e| FittingError::EvaluationFailed(format!("{e:?}")))?;
    let gain = held_block.solve(&cross);
    let along = held_block.solve(&shift);
    let n = rest.len();
    let mean: Vec<f64> = (0..n)
        .map(|b| {
            prior.mean[rest[b]]
                + (0..held.len())
                    .map(|a| cross[(a, b)] * along[(a, 0)])
                    .sum::<f64>()
        })
        .collect();
    let mut remaining = FlatMatrix::zeros(n, n);
    for b in 0..n {
        for c in 0..n {
            let removed = |b: usize, c: usize| -> f64 {
                (0..held.len()).map(|a| cross[(a, b)] * gain[(a, c)]).sum()
            };
            *remaining.get_mut(b, c) =
                covariance(rest[b], rest[c]) - 0.5 * (removed(b, c) + removed(c, b));
        }
    }
    let parameters: Vec<usize> = rest.iter().map(|&i| prior.parameters[i]).collect();
    let conditioned = Prior::factored(&parameters, &mean, &remaining);
    if conditioned.is_valid() {
        Ok(conditioned)
    } else {
        Err(FittingError::EvaluationFailed(format!(
            "the prior conditioned on the parameters on a bound is not positive definite: \
             {remaining:?}"
        )))
    }
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
        let both = consistency(&prior, &estimate, &[false; 2], &posterior([0.5, 0.25]))
            .unwrap()
            .unwrap();
        let along_second = -s * z[0] + c * z[1];
        let q = along_first.powi(2) / 0.5 + along_second.powi(2) / 0.25;
        assert_eq!(both.dof, 2);
        assert!((both.q / q - 1.0).abs() <= 1e-12, "{} vs {q}", both.q);
        assert!((both.p / (-q / 2.0).exp() - 1.0).abs() <= 1e-12);
        let one = consistency(&prior, &estimate, &[false; 2], &posterior([0.5, 1e-4]))
            .unwrap()
            .unwrap();
        assert_eq!(one.dof, 1);
        let q = along_first.powi(2) / 0.5;
        assert!((one.q / q - 1.0).abs() <= 1e-12, "{} vs {q}", one.q);
        assert!(
            consistency(&prior, &estimate, &[false; 2], &posterior([1e-4, 1e-4]))
                .unwrap()
                .is_none()
        );
        let mut undetermined = posterior([0.5, 0.25]);
        undetermined.data[3] = f64::NAN;
        assert!(
            consistency(&prior, &estimate, &[false; 2], &undetermined)
                .unwrap()
                .is_none()
        );
        assert!(consistency(&prior, &estimate[..1], &[false; 2], &posterior([0.5, 0.25])).is_err());
        let mut truncated = posterior([0.5, 0.25]);
        truncated.data.pop();
        assert!(consistency(&prior, &estimate, &[false; 2], &truncated).is_err());
        let nan = [estimate[0], f64::NAN];
        assert!(consistency(&prior, &nan, &[false; 2], &posterior([0.5, 0.25])).is_err());
        let negative = Prior::measured(0, 0.0, -1.0);
        let unit = FlatMatrix {
            data: vec![0.5],
            nrows: 1,
            ncols: 1,
        };
        assert!(consistency(&negative, &[1.0], &[false], &unit).is_err());
        let identity =
            Prior::correlated(&[0, 1], &[0.0, 0.0], &two_by_two([1.0, 0.0, 0.0, 1.0])).unwrap();
        for impossible in [
            [0.5, 5.0, -5.0, 0.5],
            [1.5, 0.0, 0.0, 0.5],
            [-1.0, 0.0, 0.0, 0.5],
        ] {
            let result = consistency(&identity, &[2.0, 1.0], &[false; 2], &two_by_two(impossible));
            assert!(result.is_err(), "{impossible:?}: {result:?}");
        }
        let far = consistency(
            &identity,
            &[1e155, 1e155],
            &[false; 2],
            &two_by_two([0.5, 0.0, 0.0, 0.5]),
        )
        .unwrap()
        .unwrap();
        assert_eq!((far.q, far.p), (f64::INFINITY, 0.0));
    }

    #[test]
    fn a_parameter_on_a_bound_conditions_the_prior_on_the_rest() {
        let covariance = FlatMatrix {
            data: vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0],
            nrows: 3,
            ncols: 3,
        };
        let prior = Prior::correlated(&[0, 1, 2], &[1.0, 2.0, 3.0], &covariance).unwrap();
        let (mean, remaining) = ([1.2, 3.3], [[3.2, -0.2], [-0.2, 4.2]]);
        let estimate = [2.0, 2.5, 2.0];
        let mut posterior = FlatMatrix::zeros(3, 3);
        posterior.data.fill(f64::NAN);
        for (a, i) in [0, 2].into_iter().enumerate() {
            for (b, j) in [0, 2].into_iter().enumerate() {
                *posterior.get_mut(i, j) = 0.5 * remaining[a][b];
            }
        }
        let d = [estimate[0] - mean[0], estimate[2] - mean[1]];
        let determinant = remaining[0][0] * remaining[1][1] - remaining[0][1] * remaining[1][0];
        let mahalanobis = (d[0] * d[0] * remaining[1][1] - 2.0 * d[0] * d[1] * remaining[0][1]
            + d[1] * d[1] * remaining[0][0])
            / determinant;
        let result = consistency(&prior, &estimate, &[false, true, false], &posterior)
            .unwrap()
            .unwrap();
        assert_eq!(result.dof, 2);
        assert!(
            (result.q / (2.0 * mahalanobis) - 1.0).abs() <= 1e-12,
            "{} vs {}",
            result.q,
            2.0 * mahalanobis
        );
        assert!(
            consistency(&prior, &estimate, &[true; 3], &posterior)
                .unwrap()
                .is_none()
        );
    }
}
