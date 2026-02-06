//! Gaussian resolution function for testing and simple instruments.
//!
//! Teacher reference: `sammy/src/grp/mgrp1.f90` (DELTAL, DELTAG, DELTAB parameters)

use nereids_core::{EnergyGrid, PhysicsError, ResolutionFunction};

/// Gaussian resolution function with constant width.
#[derive(Debug, Clone)]
pub struct GaussianResolution {
    /// Gaussian sigma in eV.
    pub sigma: f64,
}

impl ResolutionFunction for GaussianResolution {
    fn convolve(&self, _energy: &EnergyGrid, spectrum: &[f64]) -> Result<Vec<f64>, PhysicsError> {
        // Stub: return spectrum unchanged.
        // Full implementation will convolve with Gaussian kernel.
        Ok(spectrum.to_vec())
    }
}
