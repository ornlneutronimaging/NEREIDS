//! Background model types for transmission fitting.

use crate::energy::EnergyGrid;

/// A background model that contributes an additive term to the transmission.
pub trait BackgroundModel: Send + Sync {
    /// Evaluate the background at each energy point.
    fn evaluate(&self, energy: &EnergyGrid) -> Vec<f64>;
}

/// Constant background (energy-independent).
#[derive(Debug, Clone)]
pub struct ConstantBackground {
    pub value: f64,
}

impl BackgroundModel for ConstantBackground {
    fn evaluate(&self, energy: &EnergyGrid) -> Vec<f64> {
        vec![self.value; energy.len()]
    }
}

/// Polynomial background: `b0 + b1*E + b2*E^2 + ...`
#[derive(Debug, Clone)]
pub struct PolynomialBackground {
    /// Coefficients in ascending power order: \[b0, b1, b2, ...\].
    pub coefficients: Vec<f64>,
}

impl BackgroundModel for PolynomialBackground {
    #[allow(clippy::cast_possible_wrap)]
    fn evaluate(&self, energy: &EnergyGrid) -> Vec<f64> {
        energy
            .values
            .iter()
            .map(|&e| {
                self.coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| c * e.powi(i as i32))
                    .sum()
            })
            .collect()
    }
}
