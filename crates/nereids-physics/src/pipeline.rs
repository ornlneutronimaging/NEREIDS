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

/// Count the total number of free (vary=true) parameters in the R-matrix problem.
fn count_free_params(params: &RMatrixParameters) -> usize {
    let mut count = 0;
    for isotope in &params.isotopes {
        if isotope.abundance.vary {
            count += 1;
        }
        for sg in &isotope.spin_groups {
            for res in &sg.resonances {
                if res.energy.vary {
                    count += 1;
                }
                if res.gamma_n.vary {
                    count += 1;
                }
                if res.gamma_g.vary {
                    count += 1;
                }
                if let Some(ref f) = res.fission {
                    if f.gamma_f1.vary {
                        count += 1;
                    }
                    if f.gamma_f2.vary {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

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
        // Count free parameters across all isotopes and spin groups.
        let n_params = count_free_params(params);
        // Stub: return zero Jacobian with shape [n_energy][n_params].
        let jacobian = vec![vec![0.0; n_params]; energy.len()];
        Ok((t, jacobian))
    }
}
