//! Resolution function trait for instrument response convolution.

use crate::energy::EnergyGrid;
use crate::error::PhysicsError;

/// A resolution function that convolves a spectrum with the instrument response.
///
/// For VENUS, this is a user-defined function from Monte Carlo simulation
/// capturing the SNS source signature and beamline optics. Without proper
/// resolution broadening, fitting fails (lesson from `ImagingReso`).
pub trait ResolutionFunction: Send + Sync {
    /// Convolve the given spectrum with the instrument resolution.
    fn convolve(&self, energy: &EnergyGrid, spectrum: &[f64]) -> Result<Vec<f64>, PhysicsError>;
}
