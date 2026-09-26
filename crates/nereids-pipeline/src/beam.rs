//! The beam before the sample: neutrons per µs of flight time.

/// `ln φ(u)`, the logarithm of the beam per µs at flight time `u`, as a
/// uniform cubic B-spline in `x = ln u` over
/// [`knot_span_us`](Self::knot_span_us).  Faster than that span, `ln φ` is the
/// spline's second-order Taylor expansion in `x` at the first knot; slower,
/// the last interval's cubic continues.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamSpline {
    x_low: f64,
    x_high: f64,
    coefficients: Vec<f64>,
}

impl BeamSpline {
    pub(crate) fn constant(u_low: f64, u_high: f64, per_us: f64) -> Self {
        Self {
            x_low: u_low.ln(),
            x_high: u_high.ln(),
            coefficients: vec![per_us.ln(); 4],
        }
    }

    pub(crate) fn with_coefficients(&self, coefficients: &[f64]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
            ..self.clone()
        }
    }

    pub(crate) fn refined(&self) -> Self {
        let c = &self.coefficients;
        let coefficients = (0..2 * c.len() - 3)
            .map(|p| {
                let i = p / 2;
                if p % 2 == 0 {
                    0.5 * (c[i] + c[i + 1])
                } else {
                    (c[i] + 6.0 * c[i + 1] + c[i + 2]) / 8.0
                }
            })
            .collect();
        Self {
            coefficients,
            ..self.clone()
        }
    }

    /// Number of spline intervals; there are three more coefficients.
    #[must_use]
    pub fn intervals(&self) -> usize {
        self.coefficients.len() - 3
    }

    /// The spline's coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The flight times, in µs, the knots span.
    #[must_use]
    pub fn knot_span_us(&self) -> (f64, f64) {
        (self.x_low.exp(), self.x_high.exp())
    }

    /// `(first, weights)` with `ln φ(u) = Σ weights[i] · coefficients[first + i]`,
    /// for `u_us > 0`.
    #[must_use]
    pub fn basis(&self, u_us: f64) -> (usize, [f64; 4]) {
        let h = (self.x_high - self.x_low) / self.intervals() as f64;
        let s = (u_us.ln() - self.x_low) / h;
        if s < 0.0 {
            return (
                0,
                [
                    1.0 / 6.0 - s / 2.0 + s * s / 2.0,
                    2.0 / 3.0 - s * s,
                    1.0 / 6.0 + s / 2.0 + s * s / 2.0,
                    0.0,
                ],
            );
        }
        let first = (s.floor() as usize).min(self.intervals() - 1);
        let t = s - first as f64;
        (
            first,
            [
                (1.0 - t).powi(3) / 6.0,
                (3.0 * t.powi(3) - 6.0 * t * t + 4.0) / 6.0,
                (-3.0 * t.powi(3) + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
                t.powi(3) / 6.0,
            ],
        )
    }

    /// The beam per µs at flight time `u_us > 0`.
    #[must_use]
    pub fn per_us(&self, u_us: f64) -> f64 {
        let (first, weights) = self.basis(u_us);
        weights
            .iter()
            .zip(&self.coefficients[first..])
            .map(|(w, c)| w * c)
            .sum::<f64>()
            .exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wavy() -> BeamSpline {
        let spline = BeamSpline::constant(280.0, 470.0, 1.0).refined().refined();
        let c: Vec<f64> = (0..spline.coefficients().len())
            .map(|i| (i as f64 * 0.7).sin())
            .collect();
        spline.with_coefficients(&c)
    }

    fn ln_beam(spline: &BeamSpline, u: f64) -> f64 {
        spline.per_us(u).ln()
    }

    #[test]
    fn the_coefficients_of_a_cubic_in_ln_u_reproduce_it() {
        let (u_low, u_high) = (280.0, 470.0);
        let [a, b, c, d] = [3.0, -0.7, 1.9, -2.3];
        let coefficients = [-1.0, 0.0, 1.0, 2.0]
            .map(|k: f64| a + b * k + c * (k * k - 1.0 / 3.0) + d * (k.powi(3) - k));
        let beam = BeamSpline::constant(u_low, u_high, 1.0).with_coefficients(&coefficients);
        for step in 0..=40 {
            let u = u_low * (u_high / u_low).powf(f64::from(step) / 40.0);
            let t = (u / u_low).ln() / (u_high / u_low).ln();
            let expected = a + b * t + c * t * t + d * t.powi(3);
            assert!((ln_beam(&beam, u) - expected).abs() < 1e-12, "{u}");
        }
    }

    #[test]
    fn refining_keeps_the_curve_inside_and_below_the_knots() {
        let spline = wavy();
        let refined = spline.refined();
        assert_eq!(refined.intervals(), 2 * spline.intervals());
        for k in 0..400 {
            let u = 150.0 * (600.0_f64 / 150.0).powf(f64::from(k) / 399.0);
            let (a, b) = (ln_beam(&spline, u), ln_beam(&refined, u));
            assert!((a - b).abs() < 1e-12, "at {u} µs: {a} became {b}");
        }
    }

    #[test]
    fn below_the_knots_the_curve_is_its_taylor_expansion_at_the_first_knot() {
        let spline = wavy();
        let x0 = 280.0_f64.ln();
        let f = |x: f64| ln_beam(&spline, x.exp());
        let h = (470.0_f64 / 280.0).ln() / spline.intervals() as f64;
        let nodes = [0.0, h / 3.0, 2.0 * h / 3.0, h];
        let values = nodes.map(|dx| f(x0 + dx));
        let others = |i: usize| (0..4).filter(move |&j| j != i);
        let cubic = |dx: f64| -> f64 {
            (0..4)
                .map(|i| {
                    values[i]
                        * others(i)
                            .map(|j| (dx - nodes[j]) / (nodes[i] - nodes[j]))
                            .product::<f64>()
                })
                .sum()
        };
        let leading: f64 = (0..4)
            .map(|i| values[i] / others(i).map(|j| nodes[i] - nodes[j]).product::<f64>())
            .sum();
        for dx in [-0.05, -0.2, -0.4] {
            let taylor = cubic(dx) - leading * dx.powi(3);
            assert!(
                (f(x0 + dx) - taylor).abs() < 1e-9 * (1.0 + taylor.abs()),
                "{dx}: {} vs {taylor}",
                f(x0 + dx)
            );
        }
    }
}
