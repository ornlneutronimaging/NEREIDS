//! Counts the detector records away from where they arrive.
//!
//! The detector is a grid of square cells of side 1.  Counts arrive evenly
//! over a cell, and the share of them the detector spreads is recorded at a
//! displacement `u` drawn from a radial kernel normalized over the plane,
//!
//! ```text
//! K(r) = Σ_c w_c·exp(−r/s_c)/(2π s_c²),   Σ_c w_c = 1,
//! ```
//!
//! lengths `s_c` in cells.  The share of a cell's spread counts recorded in the
//! cell offset by `(dx, dy)` is
//!
//! ```text
//! T(dx, dy) = ∫ K(u)·Λ(u_x − dx)·Λ(u_y − dy) du,   Λ(t) = max(0, 1 − |t|),
//! ```
//!
//! `Λ(u_x − dx)·Λ(u_y − dy)` the overlap of the two cells displaced by `u`.

use std::f64::consts::{FRAC_PI_4, PI};

/// Eight-point Gauss–Legendre abscissae on `[−1, 1]`.
const ABSCISSAE: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
/// Eight-point Gauss–Legendre weights on `[−1, 1]`.
const WEIGHTS: [f64; 8] = [
    0.101_228_536_290_376_26,
    0.222_381_034_453_374_47,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_47,
    0.101_228_536_290_376_26,
];

/// `∫₀^R r^k e^(−r/s) dr` for `k` of 1, 2 or 3.
fn moment(k: i32, reach: f64, length: f64) -> f64 {
    let x = reach / length;
    let order = f64::from(k + 1);
    let gamma = if x < 1.0 {
        let mut term = x.powi(k + 1);
        let mut sum = term / order;
        for j in 1..40 {
            term *= -x / f64::from(j);
            sum += term / (order + f64::from(j));
        }
        sum
    } else {
        let mut term = 1.0;
        let mut partial = 1.0;
        for j in 1..=k {
            term *= x / f64::from(j);
            partial += term;
        }
        let factorial: f64 = (1..=k).map(f64::from).product();
        factorial * (1.0 - (-x).exp() * partial)
    };
    length.powi(k + 1) * gamma
}

/// `∫ exp(−r/s)/(2π s²)·(a + b·x + c·y + d·x·y)` over the unit square
/// `[0, 1]²`, in polar coordinates about its corner at the origin.
fn corner(length: f64, [a, b, c, d]: [f64; 4]) -> f64 {
    let mut sum = 0.0;
    for half in 0..2 {
        let (lo, hi) = (f64::from(half) * FRAC_PI_4, f64::from(half + 1) * FRAC_PI_4);
        let (mid, width) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        for (x, w) in ABSCISSAE.iter().zip(WEIGHTS) {
            let theta = mid + width * x;
            let (cos, sin) = (theta.cos(), theta.sin());
            let reach = 1.0 / if half == 0 { cos } else { sin };
            sum += width
                * w
                * (a * moment(1, reach, length)
                    + (b * cos + c * sin) * moment(2, reach, length)
                    + d * cos * sin * moment(3, reach, length));
        }
    }
    sum / (2.0 * PI * length * length)
}

/// `T(dx, dy)` of an exponential kernel of length `length`, in cells.
fn transfer_one(length: f64, [dx, dy]: [i64; 2]) -> f64 {
    let kernel = |x: f64, y: f64| (-(x * x + y * y).sqrt() / length).exp();
    let mut sum = 0.0;
    for side_x in [-1, 1] {
        for side_y in [-1, 1] {
            let x0 = dx + i64::from(side_x.min(0));
            let y0 = dy + i64::from(side_y.min(0));
            let (alpha_x, beta_x) = (1.0 + f64::from(side_x) * dx as f64, f64::from(-side_x));
            let (alpha_y, beta_y) = (1.0 + f64::from(side_y) * dy as f64, f64::from(-side_y));
            if (x0 == 0 || x0 == -1) && (y0 == 0 || y0 == -1) {
                let (flip_x, flip_y) = (
                    if x0 == 0 { 1.0 } else { -1.0 },
                    if y0 == 0 { 1.0 } else { -1.0 },
                );
                sum += corner(
                    length,
                    [
                        alpha_x * alpha_y,
                        beta_x * flip_x * alpha_y,
                        alpha_x * beta_y * flip_y,
                        beta_x * flip_x * beta_y * flip_y,
                    ],
                );
            } else {
                let mut cell = 0.0;
                for (x, w) in ABSCISSAE.iter().zip(WEIGHTS) {
                    let u = x0 as f64 + 0.5 * (x + 1.0);
                    for (y, v) in ABSCISSAE.iter().zip(WEIGHTS) {
                        let t = y0 as f64 + 0.5 * (y + 1.0);
                        cell += 0.25
                            * w
                            * v
                            * kernel(u, t)
                            * (alpha_x + beta_x * u)
                            * (alpha_y + beta_y * t);
                    }
                }
                sum += cell / (2.0 * PI * length * length);
            }
        }
    }
    sum
}

/// `T(dx, dy)` of the kernel `(w_c, s_c)`, weights normalized by their sum.
#[must_use]
pub fn transfer(kernel: &[(f64, f64)], offset: [i64; 2]) -> f64 {
    let total: f64 = kernel.iter().map(|&(w, _)| w).sum();
    kernel
        .iter()
        .map(|&(w, s)| w * transfer_one(s, offset))
        .sum::<f64>()
        / total
}

/// `F[r][q]`, the share of the spread counts arriving evenly over the cells of
/// region `q` that the detector records in the cells of region `r`, for the
/// regions' cells `cells` as `[row, column]` on a grid of `grid` rows and
/// columns.  Counts spread to cells in no region, or off the grid, are not
/// recorded in any region.
#[must_use]
pub fn region_shares(
    kernel: &[(f64, f64)],
    grid: [usize; 2],
    cells: &[Vec<[usize; 2]>],
) -> Vec<Vec<f64>> {
    let [rows, columns] = grid;
    let (reach_y, reach_x) = (rows as i64 - 1, columns as i64 - 1);
    let width = (2 * reach_x + 1) as usize;
    let table: Vec<f64> = (-reach_y..=reach_y)
        .flat_map(|dy| (-reach_x..=reach_x).map(move |dx| [dx, dy]))
        .map(|offset| transfer(kernel, offset))
        .collect();
    let at = |[dx, dy]: [i64; 2]| table[(dy + reach_y) as usize * width + (dx + reach_x) as usize];
    let mut region_of = vec![None; rows * columns];
    for (r, list) in cells.iter().enumerate() {
        for &[row, column] in list {
            region_of[row * columns + column] = Some(r);
        }
    }
    let mut shares = vec![vec![0.0; cells.len()]; cells.len()];
    for (q, sources) in cells.iter().enumerate() {
        for &[row, column] in sources {
            for (cell, region) in region_of.iter().enumerate() {
                if let Some(r) = *region {
                    let (target_row, target_column) = (cell / columns, cell % columns);
                    shares[r][q] += at([
                        target_column as i64 - column as i64,
                        target_row as i64 - row as i64,
                    ]);
                }
            }
        }
        let n = sources.len() as f64;
        for row in &mut shares {
            row[q] /= n;
        }
    }
    shares
}

#[cfg(test)]
mod tests {
    use super::*;

    fn midpoint(length: f64, [dx, dy]: [i64; 2], steps: usize) -> f64 {
        let h = 2.0 / steps as f64;
        let mut sum = 0.0;
        for i in 0..steps {
            let u = dx as f64 - 1.0 + h * (i as f64 + 0.5);
            let along_x = 1.0 - (u - dx as f64).abs();
            for j in 0..steps {
                let t = dy as f64 - 1.0 + h * (j as f64 + 0.5);
                let along_y = 1.0 - (t - dy as f64).abs();
                let r = (u * u + t * t).sqrt();
                sum += (-r / length).exp() * along_x * along_y;
            }
        }
        sum * h * h / (2.0 * PI * length * length)
    }

    #[test]
    fn the_transfer_is_the_kernel_integral_over_the_cell_overlap() {
        for length in [0.3, 1.0, 4.0] {
            for offset in [[0, 0], [1, 0], [0, -1], [1, 1], [-2, 1], [5, 3]] {
                let t = transfer(&[(1.0, length)], offset);
                let oracle = midpoint(length, offset, 4000);
                assert!(
                    (t - oracle).abs() <= 1e-6 * oracle.max(1e-12),
                    "s {length}, {offset:?}: {t} vs {oracle}"
                );
            }
        }
    }

    #[test]
    fn every_spread_count_lands_somewhere_on_the_plane() {
        let kernel = [(0.3, 0.4), (0.7, 2.0)];
        let total: f64 = (-60..=60)
            .flat_map(|dy| (-60..=60).map(move |dx| [dx, dy]))
            .map(|offset| transfer(&kernel, offset))
            .sum();
        assert!((total - 1.0).abs() <= 1e-9, "{total}");
    }
}
