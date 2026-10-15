//! Neutrons scattered once in a slab sample that leave through its back face.
//!
//! A neutron of energy `E` enters a slab of optical depth `D(E)` at normal
//! incidence.  It scatters elastically at depth fraction `s` with probability
//! `D_s·e^(−D(E)·s)·ds`, `D_s` the elastic optical depth, isotropically in the
//! centre-of-mass frame of a nucleus of mass ratio `A`.  At centre-of-mass
//! cosine `c` it leaves with energy `E'` along lab cosine `μ`,
//!
//! ```text
//! E' = E·(A² + 2Ac + 1)/(A + 1)²,   μ = (1 + Ac)/√(A² + 2Ac + 1),
//! ```
//!
//! and crosses the rest of the slab with transmission `e^(−D(E')·(1 − s)/μ)`.
//! The share of incident neutrons that scatter once and leave through the back
//! face, `c > −1/A`, is `D_s·G(E)`:
//!
//! ```text
//! G(E) = ½ ∫ g(D(E), D(E')/μ) dc  over −1/A < c < 1,
//! g(a, b) = ∫₀¹ e^(−a·s − b·(1 − s)) ds = (e^(−b) − e^(−a))/(a − b).
//! ```
//!
//! On a grid, `D(E')` is linear in `E'` between grid energies and equal to the
//! value at the lowest grid energy below it.

/// Four-point Gauss–Legendre abscissae on `[−1, 1]`.
const ABSCISSAE: [f64; 4] = [
    -0.861_136_311_594_052_6,
    -0.339_981_043_584_856_3,
    0.339_981_043_584_856_3,
    0.861_136_311_594_052_6,
];
/// Four-point Gauss–Legendre weights on `[−1, 1]`.
const WEIGHTS: [f64; 4] = [
    0.347_854_845_137_453_86,
    0.652_145_154_862_546_1,
    0.652_145_154_862_546_1,
    0.347_854_845_137_453_86,
];
/// Halvings of the interval that ends at grazing exit, `μ = 0`.
const HALVINGS: i32 = 16;

/// `g(a, b)`, `∂g/∂a` and `∂g/∂b` for optical depths `a`, `b` of 0 or more.
#[must_use]
pub fn slab_escape(a: f64, b: f64) -> (f64, f64, f64) {
    let gap = (a - b).abs();
    let base = (-a.min(b)).exp();
    let mean = if gap == 0.0 {
        1.0
    } else {
        -(-gap).exp_m1() / gap
    };
    let tilted = if gap < 0.1 {
        let mut term = 1.0;
        let mut sum = 0.5;
        for k in 1..14 {
            term *= -gap / f64::from(k);
            sum += term / f64::from(k + 2);
        }
        sum
    } else {
        (-(-gap).exp_m1() - gap * (-gap).exp()) / (gap * gap)
    };
    let (near, far) = (-base * (mean - tilted), -base * tilted);
    if a <= b {
        (base * mean, near, far)
    } else {
        (base * mean, far, near)
    }
}

/// `G` on a grid, with its slopes in the optical depths.
pub struct ForwardYield {
    /// `G(E_j)` at each grid energy.
    pub share: Vec<f64>,
    /// `∂G(E_j)/∂D(E_j)` through the depth at entry.
    pub per_entry: Vec<f64>,
    /// `∂G(E_j)/∂D(E_k)` through the depths at exit, as `(k, slope)`.
    pub per_exit: Vec<Vec<(usize, f64)>>,
}

/// [`ForwardYield`] at grid energies `energies_ev`, descending, with optical
/// depths `depth` and mass ratio `awr`.
#[must_use]
pub fn forward_yield(energies_ev: &[f64], depth: &[f64], awr: f64) -> ForwardYield {
    let n = energies_ev.len();
    let grazing = -1.0 / awr;
    let lift = (awr + 1.0).powi(2);
    let mut out = ForwardYield {
        share: vec![0.0; n],
        per_entry: vec![0.0; n],
        per_exit: vec![Vec::new(); n],
    };
    for j in 0..n {
        let energy = energies_ev[j];
        let cosine = |e: f64| (lift * e / energy - awr * awr - 1.0) / (2.0 * awr);
        let last = (j + 1..n)
            .take_while(|&k| cosine(energies_ev[k]) > grazing)
            .last()
            .unwrap_or(j);
        let mut exits = vec![0.0; last + 2 - j];
        let mut add = |lo: f64, hi: f64, upper: usize| {
            let lower = (upper + 1).min(n - 1);
            let span = energies_ev[upper] - energies_ev[lower];
            let (mid, half) = (0.5 * (hi + lo), 0.5 * (hi - lo));
            for (x, w) in ABSCISSAE.iter().zip(WEIGHTS) {
                let c = mid + half * x;
                let ratio = (awr * awr + 2.0 * awr * c + 1.0) / lift;
                let mu = (1.0 + awr * c) / ((awr + 1.0) * ratio.sqrt());
                let t = if span > 0.0 {
                    (energies_ev[upper] - energy * ratio) / span
                } else {
                    0.0
                };
                let exit = (1.0 - t) * depth[upper] + t * depth[lower];
                let (g, along_entry, along_exit) = slab_escape(depth[j], exit / mu);
                let weight = 0.5 * half * w;
                out.share[j] += weight * g;
                out.per_entry[j] += weight * along_entry;
                exits[upper - j] += weight * along_exit / mu * (1.0 - t);
                exits[lower - j] += weight * along_exit / mu * t;
            }
        };
        for k in j..last {
            add(
                cosine(energies_ev[k + 1]),
                cosine(energies_ev[k]).min(1.0),
                k,
            );
        }
        let top = if last == j {
            1.0
        } else {
            cosine(energies_ev[last])
        };
        let width = top - grazing;
        for i in 0..HALVINGS {
            add(
                grazing + width * 0.5_f64.powi(i + 1),
                grazing + width * 0.5_f64.powi(i),
                last,
            );
        }
        add(grazing, grazing + width * 0.5_f64.powi(HALVINGS), last);
        out.per_exit[j] = exits
            .into_iter()
            .enumerate()
            .filter(|&(_, s)| s != 0.0)
            .map(|(i, s)| (j + i, s))
            .collect();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simpson(f: impl Fn(f64) -> f64, lo: f64, hi: f64, steps: usize) -> f64 {
        let h = (hi - lo) / steps as f64;
        (0..=steps)
            .map(|i| {
                let weight = match i {
                    0 => 1.0,
                    i if i == steps => 1.0,
                    i if i % 2 == 1 => 4.0,
                    _ => 2.0,
                };
                weight * f(lo + h * i as f64)
            })
            .sum::<f64>()
            * h
            / 3.0
    }

    #[test]
    fn the_escape_is_the_depth_integral_and_its_slopes_are_its_derivatives() {
        for (a, b) in [
            (0.3, 0.3),
            (0.3, 0.3 + 1e-9),
            (0.3, 0.35),
            (2.0, 0.01),
            (0.01, 40.0),
            (25.0, 3.0),
        ] {
            let (g, along_a, along_b) = slab_escape(a, b);
            let oracle = simpson(|s| (-a * s - b * (1.0 - s)).exp(), 0.0, 1.0, 4000);
            assert!(
                (g / oracle - 1.0).abs() <= 1e-10,
                "{a} {b}: {g} vs {oracle}"
            );
            let h = 1e-6 * a.max(b);
            let da = (slab_escape(a + h, b).0 - slab_escape(a - h, b).0) / (2.0 * h);
            let db = (slab_escape(a, b + h).0 - slab_escape(a, b - h).0) / (2.0 * h);
            assert!(
                (along_a - da).abs() <= 1e-7 * g,
                "{a} {b}: {along_a} vs {da}"
            );
            assert!(
                (along_b - db).abs() <= 1e-7 * g,
                "{a} {b}: {along_b} vs {db}"
            );
        }
    }

    fn grid() -> (Vec<f64>, Vec<f64>) {
        let energies: Vec<f64> = (0..400).map(|i| 21.0 - 0.005 * f64::from(i)).collect();
        let depth = energies
            .iter()
            .map(|e| 0.05 + 6.0 / (1.0 + ((e - 20.2) / 0.08).powi(2)))
            .collect();
        (energies, depth)
    }

    #[test]
    fn a_thin_slab_sends_the_forward_share_of_its_scatters_out_the_back() {
        let awr = 180.0;
        let (energies, _) = grid();
        let depth = vec![1e-12; energies.len()];
        let yield_ = forward_yield(&energies, &depth, awr);
        for share in yield_.share {
            assert!((share - 0.5 * (1.0 + 1.0 / awr)).abs() <= 1e-8, "{share}");
        }
    }

    #[test]
    fn the_share_is_the_double_integral_over_depth_and_angle() {
        let awr = 180.0;
        let (energies, depth) = grid();
        let yield_ = forward_yield(&energies, &depth, awr);
        let interpolated = |e: f64| {
            let k = energies
                .iter()
                .position(|&x| x < e)
                .unwrap_or(energies.len());
            if k == 0 {
                return depth[0];
            }
            if k == energies.len() {
                return depth[k - 1];
            }
            let t = (energies[k - 1] - e) / (energies[k - 1] - energies[k]);
            (1.0 - t) * depth[k - 1] + t * depth[k]
        };
        for j in [0, 120, 160, 175, 230, 399] {
            let (e, a) = (energies[j], depth[j]);
            let inner = |c: f64| {
                let root = (awr * awr + 2.0 * awr * c + 1.0).sqrt();
                let mu = (1.0 + awr * c) / root;
                let b = interpolated(e * root * root / (awr + 1.0).powi(2)) / mu;
                if (a - b).abs() < 1e-6 {
                    (-0.5 * (a + b)).exp()
                } else {
                    ((-b).exp() - (-a).exp()) / (a - b)
                }
            };
            let oracle = 0.5 * simpson(inner, -1.0 / awr, 1.0, 200_000);
            assert!(
                (yield_.share[j] / oracle - 1.0).abs() <= 1e-6,
                "node {j}: {} vs {oracle}",
                yield_.share[j]
            );
        }
    }

    #[test]
    fn the_slopes_are_the_derivatives_of_the_share() {
        let awr = 180.0;
        let (energies, depth) = grid();
        let yield_ = forward_yield(&energies, &depth, awr);
        for j in [100, 158, 170] {
            let along = |k: usize| {
                let h = 1e-6 * depth[k];
                let mut up = depth.clone();
                let mut down = depth.clone();
                up[k] += h;
                down[k] -= h;
                (forward_yield(&energies, &up, awr).share[j]
                    - forward_yield(&energies, &down, awr).share[j])
                    / (2.0 * h)
            };
            let exit_at = |k: usize| {
                yield_.per_exit[j]
                    .iter()
                    .find(|&&(i, _)| i == k)
                    .map_or(0.0, |&(_, s)| s)
            };
            for k in j..j + 25 {
                let analytic = exit_at(k) + if k == j { yield_.per_entry[j] } else { 0.0 };
                assert!(
                    (analytic - along(k)).abs() <= 1e-6 * yield_.share[j],
                    "node {j}, depth {k}: {analytic} vs {}",
                    along(k)
                );
            }
        }
    }
}
