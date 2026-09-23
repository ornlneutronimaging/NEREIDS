//! The beam before the sample: neutrons per µs of flight time.

/// `ln φ(u)`, the logarithm of the beam per µs at flight time `u`, as a
/// uniform cubic B-spline in `x = ln u` between the flight times it was built
/// for, and outside them the straight line with the spline's value and slope
/// at the nearer end.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamSpline {
    x_low: f64,
    x_high: f64,
    coefficients: Vec<f64>,
}

impl BeamSpline {
    /// A beam of `per_us` neutrons per µs at every flight time, as one spline
    /// interval spanning flight times `u_low` to `u_high` in µs.
    ///
    /// # Panics
    /// Unless `0 < u_low < u_high`, both finite, and `per_us` is finite and
    /// positive.
    pub fn constant(u_low: f64, u_high: f64, per_us: f64) -> Self {
        assert!(
            u_low.is_finite() && u_high.is_finite() && 0.0 < u_low && u_low < u_high,
            "the flight times must satisfy 0 < low < high, got {u_low} and {u_high}"
        );
        assert!(
            per_us.is_finite() && per_us > 0.0,
            "the beam must be finite and positive, got {per_us}"
        );
        Self {
            x_low: u_low.ln(),
            x_high: u_high.ln(),
            coefficients: vec![per_us.ln(); 4],
        }
    }

    pub fn intervals(&self) -> usize {
        self.coefficients.len() - 3
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The same knots with `coefficients`.
    ///
    /// # Panics
    /// If `coefficients` has a different length.
    pub fn with_coefficients(&self, coefficients: &[f64]) -> Self {
        assert_eq!(coefficients.len(), self.coefficients.len());
        Self {
            coefficients: coefficients.to_vec(),
            ..self.clone()
        }
    }

    /// The same curve with every interval halved.
    pub fn refined(&self) -> Self {
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

    /// `(index, weight)` pairs with `ln φ(u) = Σ weight · coefficients[index]`.
    pub fn basis(&self, u_us: f64) -> [(usize, f64); 4] {
        let n = self.intervals();
        let h = (self.x_high - self.x_low) / n as f64;
        let x = u_us.ln();
        let line = |first: usize, d: f64| {
            let slope = d / (2.0 * h);
            [
                (first, 1.0 / 6.0 - slope),
                (first + 1, 4.0 / 6.0),
                (first + 2, 1.0 / 6.0 + slope),
            ]
        };
        if x < self.x_low {
            let [a, b, c] = line(0, x - self.x_low);
            return [a, b, c, (3, 0.0)];
        }
        if x > self.x_high {
            let [a, b, c] = line(n, x - self.x_high);
            return [(n - 1, 0.0), a, b, c];
        }
        let q = (((x - self.x_low) / h).floor() as usize).min(n - 1);
        let t = (x - self.x_low) / h - q as f64;
        [
            (q, (1.0 - t).powi(3) / 6.0),
            (q + 1, (3.0 * t.powi(3) - 6.0 * t * t + 4.0) / 6.0),
            (
                q + 2,
                (-3.0 * t.powi(3) + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
            ),
            (q + 3, t.powi(3) / 6.0),
        ]
    }

    /// The beam per µs at flight time `u_us`.
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
        let mut spline = BeamSpline::constant(50.0, 800.0, 1.0).refined().refined();
        let c: Vec<f64> = (0..spline.coefficients().len())
            .map(|i| (i as f64 * 0.7).sin())
            .collect();
        spline = spline.with_coefficients(&c);
        spline
    }

    #[test]
    fn refining_keeps_the_curve_inside_and_outside_the_knots() {
        let spline = wavy();
        let refined = spline.refined();
        assert_eq!(refined.intervals(), 2 * spline.intervals());
        for k in 0..400 {
            let u = 20.0 * (2000.0_f64 / 20.0).powf(f64::from(k) / 399.0);
            let (a, b) = (spline.per_us(u).ln(), refined.per_us(u).ln());
            assert!((a - b).abs() < 1e-12, "at {u} µs: {a} became {b}");
        }
    }

    #[test]
    fn a_power_law_is_a_straight_line_everywhere() {
        let spline = BeamSpline::constant(50.0, 800.0, 1.0);
        let c: Vec<f64> = (0..4).map(|i| 2.0 - 0.5 * f64::from(i)).collect();
        let spline = spline.with_coefficients(&c);
        let h = (800.0_f64 / 50.0).ln();
        let (at_low, slope) = (spline.per_us(50.0).ln(), -0.5 / h);
        for u in [10.0, 50.0, 120.0, 800.0, 5000.0] {
            let expected = at_low + slope * (u / 50.0_f64).ln();
            assert!((spline.per_us(u).ln() - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn the_curve_and_its_slope_are_continuous_at_the_ends() {
        let spline = wavy();
        for u in [50.0, 800.0] {
            let d = 1e-7;
            let value = |u: f64| spline.per_us(u).ln();
            let (below, above) = (value(u * (1.0 - d)), value(u * (1.0 + d)));
            assert!((below - value(u)).abs() < 1e-6 && (above - value(u)).abs() < 1e-6);
            let slope_below = (value(u) - below) / d;
            let slope_above = (above - value(u)) / d;
            assert!((slope_below - slope_above).abs() < 1e-5, "kink at {u} µs");
        }
    }
}
