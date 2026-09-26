//! The beam before the sample: neutrons per µs of flight time.

/// `ln φ(u)`, the logarithm of the beam per µs at flight time `u`, as a
/// uniform cubic B-spline in `x = ln u` over
/// [`knot_span_us`](Self::knot_span_us).  Faster than that span, `ln φ`
/// continues from the spline's value and slope at the first knot with the
/// spline's mean curvature over the span, so a beam whose curvature changes
/// across the span is continued with its mean; slower, the last interval's
/// cubic continues.  The curve is the beam only over the flight times a fit
/// integrates.
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

    /// `(index, weight)` pairs with `ln φ(u) = Σ weight · coefficients[index]`,
    /// for `u_us > 0`; an index may appear more than once.
    #[must_use]
    pub fn basis(&self, u_us: f64) -> [(usize, f64); 5] {
        let n = self.intervals();
        let h = (self.x_high - self.x_low) / n as f64;
        let s = (u_us.ln() - self.x_low) / h;
        if s < 0.0 {
            let mean_curvature = s * s / (4.0 * n as f64);
            return [
                (0, 1.0 / 6.0 - s / 2.0 + mean_curvature),
                (1, 2.0 / 3.0),
                (2, 1.0 / 6.0 + s / 2.0 - mean_curvature),
                (n, -mean_curvature),
                (n + 2, mean_curvature),
            ];
        }
        let first = (s.floor() as usize).min(n - 1);
        let t = s - first as f64;
        [
            (first, (1.0 - t).powi(3) / 6.0),
            (first + 1, (3.0 * t.powi(3) - 6.0 * t * t + 4.0) / 6.0),
            (
                first + 2,
                (-3.0 * t.powi(3) + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
            ),
            (first + 3, t.powi(3) / 6.0),
            (first, 0.0),
        ]
    }

    /// The beam per µs at flight time `u_us > 0`.
    #[must_use]
    pub fn per_us(&self, u_us: f64) -> f64 {
        self.basis(u_us)
            .iter()
            .map(|&(i, w)| w * self.coefficients[i])
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

    fn slope_at_first_node(f: &dyn Fn(f64) -> f64, nodes: [f64; 4]) -> f64 {
        let mut d = nodes.map(f);
        for k in 1..4 {
            for i in (k..4).rev() {
                d[i] = (d[i] - d[i - 1]) / (nodes[i] - nodes[i - k]);
            }
        }
        d[1] + (nodes[0] - nodes[1]) * (d[2] + d[3] * (nodes[0] - nodes[2]))
    }

    #[test]
    fn below_the_knots_the_curve_continues_with_the_mean_curvature() {
        let (x_low, x_high) = (280.0_f64.ln(), 470.0_f64.ln());
        let mut spline = BeamSpline::constant(280.0, 470.0, 1.0);
        for _ in 0..3 {
            let c: Vec<f64> = (0..spline.coefficients().len())
                .map(|i| (i as f64 * 0.7).sin())
                .collect();
            let wavy = spline.with_coefficients(&c);
            let f = |x: f64| ln_beam(&wavy, x.exp());
            let h = (x_high - x_low) / wavy.intervals() as f64;
            let within = |end: f64, toward: f64| {
                [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0].map(|k| end + toward * k * h)
            };
            let slope_low = slope_at_first_node(&f, within(x_low, 1.0));
            let slope_high = slope_at_first_node(&f, within(x_high, -1.0));
            let curvature = (slope_high - slope_low) / (x_high - x_low);
            for dx in [-0.05, -0.2, -0.4] {
                let expected = f(x_low) + slope_low * dx + curvature * dx * dx / 2.0;
                assert!(
                    (f(x_low + dx) - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                    "{} intervals, {dx}: {} vs {expected}",
                    wavy.intervals(),
                    f(x_low + dx)
                );
            }
            spline = spline.refined();
        }
    }
}
