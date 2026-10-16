//! Transmission of a sample whose areal density varies over the pixels a
//! count sums.
//!
//! A region's pixels see the areal density `x·n`, `x` distributed with mean 1
//! and relative variance `v` over the beam that reaches them.  Their summed
//! counts transmit the mean `⟨exp(−x·D)⟩`, `D = Σ n σ` the optical depth at the
//! mean density.  For `x` Gamma distributed the mean is
//!
//! ```text
//! T(D, v) = (1 + v·D)^(−1/v) = exp(−D · ln(1 + v·D)/(v·D))
//! ```
//!
//! exact for that distribution only.  At `v = 0` it is `exp(−D)`.

/// `ln(1 + y)/y`, 1 at `y = 0`.
fn log_ratio(y: f64) -> f64 {
    if y.abs() < 1e-3 {
        1.0 - y / 2.0 + y * y / 3.0 - y * y * y / 4.0
    } else {
        y.ln_1p() / y
    }
}

/// `((1 + y)·ln(1 + y) − y)/y²`, 1/2 at `y = 0`.
fn variance_ratio(y: f64) -> f64 {
    if y.abs() < 1e-3 {
        0.5 - y / 6.0 + y * y / 12.0 - y * y * y / 20.0
    } else {
        ((1.0 + y) * y.ln_1p() - y) / (y * y)
    }
}

/// The mean transmission `T`, `−∂T/∂D` and `∂T/∂v` at optical depth `depth`
/// (`D`, 0 or more) and relative variance `variance` (`v`, 0 or more) of the
/// areal density.
#[must_use]
pub fn uneven_transmission(depth: f64, variance: f64) -> (f64, f64, f64) {
    let y = variance * depth;
    let transmission = (-depth * log_ratio(y)).exp();
    let per_depth = transmission / (1.0 + y);
    let per_variance = per_depth * depth * depth * variance_ratio(y);
    (transmission, per_depth, per_variance)
}

/// `q = (1 − T)/D`, the fraction of the beam that interacts per unit optical
/// depth, with `∂q/∂D` and `∂q/∂v`, at optical depth `depth` (`D`, 0 or more)
/// and relative variance `variance` (`v`, 0 or more): `Σ n σ_x · q` is the
/// chance that a neutron's first collision in the sample is of kind `x`.
/// `q` is 1 at `D = 0`.
#[must_use]
pub fn interacting_per_depth(depth: f64, variance: f64) -> (f64, f64, f64) {
    if depth < 1e-5 && variance * depth < 1e-5 {
        let (a, b) = (
            1.0 + variance,
            1.0 + 3.0 * variance + 2.0 * variance * variance,
        );
        return (
            1.0 - a * depth / 2.0 + b * depth * depth / 6.0,
            -a / 2.0 + b * depth / 3.0,
            -depth / 2.0 + (3.0 + 4.0 * variance) * depth * depth / 6.0,
        );
    }
    let (_, per_depth, per_variance) = uneven_transmission(depth, variance);
    let absorbed = -(-depth * log_ratio(variance * depth)).exp_m1();
    (
        absorbed / depth,
        (per_depth * depth - absorbed) / (depth * depth),
        -per_variance / depth,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `⟨exp(−x·D)⟩` over the Gamma density of mean 1 and variance `v`, by the
    /// trapezoid rule in `x` on a fine grid wide enough to hold the density.
    fn gamma_mean(depth: f64, variance: f64) -> f64 {
        let shape = 1.0 / variance;
        let log_norm = -libm_lgamma(shape) - shape * variance.ln();
        let (steps, top) = (400_000, 1.0 + 40.0 * variance.sqrt() + 40.0 * variance);
        let h = top / f64::from(steps);
        (0..=steps)
            .map(|i| {
                let x = h * f64::from(i);
                let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
                if x == 0.0 {
                    return 0.0;
                }
                let density = (log_norm + (shape - 1.0) * x.ln() - x / variance).exp();
                weight * h * density * (-x * depth).exp()
            })
            .sum()
    }

    fn libm_lgamma(z: f64) -> f64 {
        // ln Γ(z) by the Lanczos series with g = 7 and nine coefficients.
        const G: [f64; 9] = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_311_6e-7,
        ];
        let z = z - 1.0;
        let mut a = G[0];
        let t = z + 7.5;
        for (i, g) in G.iter().enumerate().skip(1) {
            a += g / (z + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
    }

    #[test]
    fn the_mean_transmission_is_the_gamma_average_of_exp_minus_x_d() {
        for (depth, variance) in [(0.3, 0.02), (2.0, 0.05), (8.0, 0.1), (20.0, 0.3)] {
            let (t, _, _) = uneven_transmission(depth, variance);
            let oracle = gamma_mean(depth, variance);
            assert!(
                (t / oracle - 1.0).abs() <= 1e-6,
                "D {depth} v {variance}: {t} vs {oracle}"
            );
        }
    }

    #[test]
    fn the_interacting_fraction_is_one_minus_t_over_d_with_its_slopes() {
        for variance in [0.0, 0.05, 0.4] {
            for depth in [0.0, 5e-6, 9.99e-6, 1.001e-5, 3e-3, 0.7, 12.0] {
                let (q, per_depth, per_variance) = interacting_per_depth(depth, variance);
                let (t, _, _) = uneven_transmission(depth, variance);
                if depth > 0.0 {
                    assert!(
                        (q * depth - (1.0 - t)).abs() <= 1e-12 * (1.0 - t).max(1e-300) + 1e-15,
                        "{depth} {variance}"
                    );
                } else {
                    assert_eq!(q, 1.0);
                }
                let (hd, hv) = (1e-5 * depth.max(1e-2), 1e-4 * variance.max(1e-3));
                let along_depth = (interacting_per_depth(depth + hd, variance).0
                    - interacting_per_depth((depth - hd).max(0.0), variance).0)
                    / (depth + hd - (depth - hd).max(0.0));
                let along_variance = (interacting_per_depth(depth, variance + hv).0
                    - interacting_per_depth(depth, (variance - hv).max(0.0)).0)
                    / (variance + hv - (variance - hv).max(0.0));
                assert!(
                    (per_depth - along_depth).abs() <= 1e-6,
                    "{depth} {variance}"
                );
                assert!(
                    (per_variance - along_variance).abs() <= 1e-6,
                    "{depth} {variance}"
                );
            }
        }
    }

    #[test]
    fn a_large_variance_over_a_small_depth_keeps_the_interacting_fraction() {
        for (depth, variance) in [(5e-6, 1e6), (9e-6, 2e3)] {
            let (q, per_depth, per_variance) = interacting_per_depth(depth, variance);
            let exact = -(-(variance * depth).ln_1p() / variance).exp_m1() / depth;
            assert!((q / exact - 1.0).abs() <= 1e-12, "{q} vs {exact}");
            let (hd, hv) = (1e-6 * depth, 1e-6 * variance);
            let along_depth = (interacting_per_depth(depth + hd, variance).0
                - interacting_per_depth(depth - hd, variance).0)
                / (2.0 * hd);
            let along_variance = (interacting_per_depth(depth, variance + hv).0
                - interacting_per_depth(depth, variance - hv).0)
                / (2.0 * hv);
            assert!((per_depth / along_depth - 1.0).abs() <= 1e-6, "{per_depth}");
            assert!(
                (per_variance / along_variance - 1.0).abs() <= 1e-6,
                "{per_variance}"
            );
        }
    }

    #[test]
    fn a_uniform_sample_transmits_exp_minus_d_and_the_slopes_are_its_derivatives() {
        for depth in [0.0, 0.5, 5.0, 30.0] {
            let (t, per_depth, per_variance) = uneven_transmission(depth, 0.0);
            assert_eq!(t.to_bits(), (-depth).exp().to_bits());
            assert_eq!(per_depth.to_bits(), t.to_bits());
            assert!((per_variance - 0.5 * depth * depth * t).abs() <= 1e-15 * (1.0 + t));
        }
        for (depth, variance) in [(0.4, 1e-5), (0.4, 2e-3), (6.0, 0.05), (25.0, 0.4)] {
            let (_, per_depth, per_variance) = uneven_transmission(depth, variance);
            let (hd, hv) = (1e-6 * depth, 1e-4 * variance.max(1e-3));
            let along_depth = (uneven_transmission(depth + hd, variance).0
                - uneven_transmission(depth - hd, variance).0)
                / (2.0 * hd);
            let along_variance = (uneven_transmission(depth, variance + hv).0
                - uneven_transmission(depth, variance - hv).0)
                / (2.0 * hv);
            assert!(
                (per_depth + along_depth).abs() <= 1e-7 * per_depth,
                "{depth} {variance}"
            );
            assert!(
                (per_variance - along_variance).abs() <= 1e-6 * per_variance.abs().max(1e-300),
                "{depth} {variance}: {per_variance} vs {along_variance}"
            );
        }
    }
}
