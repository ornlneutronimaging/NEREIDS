//! Error types for the nereids-pipeline crate.

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
}
