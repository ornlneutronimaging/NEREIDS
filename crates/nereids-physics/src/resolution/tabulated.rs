//! User-defined tabulated resolution function.
//!
//! For VENUS, the resolution function comes from a Monte Carlo simulation
//! capturing the SNS source signature and beamline optics. This is loaded
//! from a user-provided file (typically passed via SAMMY Card Set 16).

use nereids_core::{EnergyGrid, PhysicsError, ResolutionFunction};

/// Tabulated resolution function loaded from user-provided data.
#[derive(Debug, Clone)]
pub struct TabulatedResolution {
    /// Energy points of the tabulated kernel.
    pub energy: Vec<f64>,
    /// Resolution kernel values at each energy point.
    pub kernel: Vec<f64>,
}

impl ResolutionFunction for TabulatedResolution {
    fn convolve(&self, _energy: &EnergyGrid, spectrum: &[f64]) -> Result<Vec<f64>, PhysicsError> {
        // Stub: return spectrum unchanged.
        // Full implementation will interpolate and convolve.
        Ok(spectrum.to_vec())
    }
}
