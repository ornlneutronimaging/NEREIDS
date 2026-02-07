//! Forward model trait — the central abstraction for transmission computation.

use crate::background::Background;
use crate::energy::EnergyGrid;
use crate::error::PhysicsError;
use crate::nuclear::RMatrixParameters;

/// Configuration for a forward model evaluation.
#[derive(Debug, Clone)]
pub struct ForwardModelConfig {
    /// Sample temperature in Kelvin. 0.0 means no Doppler broadening.
    pub temperature_k: f64,
    /// Normalization factor (multiplicative). Default 1.0.
    pub normalization: f64,
    /// Whether to include hard-sphere/potential scattering in 0K cross sections.
    pub include_potential_scattering: bool,
    /// Whether to apply self-shielding corrections.
    pub self_shielding: bool,
    /// Background model for additive corrections.
    pub background: Background,
}

impl Default for ForwardModelConfig {
    fn default() -> Self {
        Self {
            temperature_k: 0.0,
            normalization: 1.0,
            include_potential_scattering: false,
            self_shielding: false,
            background: Background::None,
        }
    }
}

/// The central abstraction for computing predicted transmission spectra.
///
/// Implementations compose the full physics pipeline:
/// 1. 0K cross sections (R-matrix)
/// 2. Doppler broadening
/// 3. Beer-Lambert transmission
/// 4. Self-shielding (optional)
/// 5. Resolution convolution
/// 6. Normalization + background
pub trait ForwardModel: Send + Sync {
    /// Compute predicted transmission spectrum.
    fn transmission(
        &self,
        energy: &EnergyGrid,
        params: &RMatrixParameters,
        config: &ForwardModelConfig,
    ) -> Result<Vec<f64>, PhysicsError>;

    /// Compute predicted transmission spectrum with Jacobian (partial derivatives
    /// with respect to fitted parameters).
    ///
    /// Returns `(transmission, jacobian)` where the Jacobian has shape
    /// `[n_energy][n_params]`: `jacobian[i][j]` = `dT_i` / `dp_j`.
    /// This follows the standard convention where rows are observations and
    /// columns are parameters.
    ///
    /// Performance note: implementations may use finite-difference fallback
    /// Jacobians when analytic derivatives are not yet available.
    fn transmission_with_jacobian(
        &self,
        energy: &EnergyGrid,
        params: &RMatrixParameters,
        config: &ForwardModelConfig,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), PhysicsError>;
}
