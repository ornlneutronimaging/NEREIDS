//! Transmission spectrum and pixel data types.

use crate::energy::EnergyGrid;

/// A measured or computed transmission spectrum.
#[derive(Debug, Clone)]
pub struct TransmissionSpectrum {
    /// Energy grid for the spectrum.
    pub energy: EnergyGrid,
    /// Transmission values (dimensionless, typically 0.0 to 1.0).
    pub values: Vec<f64>,
    /// Uncertainty on transmission values (one sigma). `None` if unknown.
    pub uncertainty: Option<Vec<f64>>,
}

/// Data for a single detector pixel.
#[derive(Debug, Clone)]
pub struct PixelData {
    /// Pixel x coordinate.
    pub x: u32,
    /// Pixel y coordinate.
    pub y: u32,
    /// Observed transmission spectrum at this pixel.
    pub observed: TransmissionSpectrum,
}
