//! The beam before the sample: neutrons per µs of flight time.

/// `ln φ(u)`, the logarithm of the beam per µs at flight time `u`, as one
/// interval of a uniform cubic B-spline in `x = ln u`, so any cubic in
/// `ln u`.  Defined on its knot span.
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

    /// The spline's four coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The flight times, in µs, the knots span.
    #[must_use]
    pub fn knot_span_us(&self) -> (f64, f64) {
        (self.x_low.exp(), self.x_high.exp())
    }

    /// `(index, weight)` pairs with `ln φ(u) = Σ weight · coefficients[index]`.
    #[must_use]
    pub fn basis(&self, u_us: f64) -> [(usize, f64); 4] {
        let t = (u_us.ln() - self.x_low) / (self.x_high - self.x_low);
        [
            (0, (1.0 - t).powi(3) / 6.0),
            (1, (3.0 * t.powi(3) - 6.0 * t * t + 4.0) / 6.0),
            (2, (-3.0 * t.powi(3) + 3.0 * t * t + 3.0 * t + 1.0) / 6.0),
            (3, t.powi(3) / 6.0),
        ]
    }

    /// The beam per µs at flight time `u_us`.
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

    #[test]
    fn the_coefficients_of_a_cubic_in_ln_u_reproduce_it() {
        let (u_low, u_high) = (280.0, 470.0);
        let [a, b, c, d] = [3.0, -0.7, 1.9, -2.3];
        let coefficients: Vec<f64> = (0..4)
            .map(|i| {
                let k = f64::from(i) - 1.0;
                a + b * k + c * (k * k - 1.0 / 3.0) + d * (k.powi(3) - k)
            })
            .collect();
        let beam = BeamSpline::constant(u_low, u_high, 1.0).with_coefficients(&coefficients);
        for step in 0..=40 {
            let u = u_low * (u_high / u_low).powf(f64::from(step) / 40.0);
            let t = (u / u_low).ln() / (u_high / u_low).ln();
            let expected = a + b * t + c * t * t + d * t.powi(3);
            assert!((beam.per_us(u).ln() - expected).abs() < 1e-12, "{u}");
        }
    }
}
