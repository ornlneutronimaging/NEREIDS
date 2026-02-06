//! Default forward model composing all pipeline stages.
//!
//! Pipeline order:
//! 1. 0K cross sections (R-matrix)
//! 2. Doppler broadening
//! 3. Beer-Lambert transmission
//! 4. Self-shielding (optional)
//! 5. Resolution convolution
//! 6. Normalization + background

use nereids_core::{
    EnergyGrid, ForwardModel, ForwardModelConfig, PhysicsError, RMatrixParameters,
    ResolutionFunction,
};

/// The default forward model that composes all physics pipeline stages.
pub struct DefaultForwardModel {
    /// Optional resolution function. If `None`, no resolution broadening is applied.
    pub resolution: Option<Box<dyn ResolutionFunction>>,
}

impl ForwardModel for DefaultForwardModel {
    fn transmission(
        &self,
        energy: &EnergyGrid,
        _params: &RMatrixParameters,
        _config: &ForwardModelConfig,
    ) -> Result<Vec<f64>, PhysicsError> {
        // Stub: return all 1.0 (perfect transmission, no sample).
        // Full implementation will compose all pipeline stages.
        Ok(vec![1.0; energy.len()])
    }

    fn transmission_with_jacobian(
        &self,
        energy: &EnergyGrid,
        params: &RMatrixParameters,
        config: &ForwardModelConfig,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), PhysicsError> {
        let t = self.transmission(energy, params, config)?;
        // Stub: return zero Jacobian.
        let n_params = params.isotopes.len();
        let jacobian = vec![vec![0.0; energy.len()]; n_params];
        Ok((t, jacobian))
    }
}
