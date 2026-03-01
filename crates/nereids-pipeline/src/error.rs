//! Error types for the nereids-pipeline crate.

use nereids_fitting::error::FittingError;
use nereids_physics::transmission::TransmissionError;

/// Errors that can occur during pipeline operations.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// Array shape or length mismatch between inputs.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Operation was cancelled via the cancellation token.
    #[error("Operation cancelled")]
    Cancelled,

    /// Error from the transmission forward model (e.g. unsorted energy grid).
    #[error("Transmission error: {0}")]
    Transmission(TransmissionError),

    /// Error from the fitting engine (e.g. empty data, length mismatch).
    #[error("Fitting error: {0}")]
    Fitting(FittingError),
}

impl From<TransmissionError> for PipelineError {
    fn from(e: TransmissionError) -> Self {
        match e {
            TransmissionError::Cancelled => PipelineError::Cancelled,
            other => PipelineError::Transmission(other),
        }
    }
}

impl From<FittingError> for PipelineError {
    fn from(e: FittingError) -> Self {
        PipelineError::Fitting(e)
    }
}
