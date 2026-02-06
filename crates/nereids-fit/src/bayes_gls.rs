//! Bayes/GLS optimizer with (M+W) inversion scheme.
//!
//! Implements the generalized least squares algorithm used in SAMMY:
//! - `Δu = (M⁻¹ + W)⁻¹ * Y`
//! - Posterior covariance: `M_new = (M⁻¹ + W)⁻¹`
//!
//! Teacher reference: `sammy/src/mpw/mmpw1.f90` (`Newpar_Mpw`)

use nereids_core::{FitConfig, FitResult, Optimizer};

/// Bayes/GLS optimizer using the (M+W) inversion scheme.
#[derive(Debug, Default)]
pub struct BayesGlsOptimizer;

impl Optimizer for BayesGlsOptimizer {
    fn fit(
        &self,
        _observed: &nereids_core::TransmissionSpectrum,
        _params: &nereids_core::RMatrixParameters,
        _config: &FitConfig,
    ) -> Result<FitResult, nereids_core::FitError> {
        Err(nereids_core::FitError::NotImplemented(
            "Bayes/GLS optimizer".into(),
        ))
    }
}
