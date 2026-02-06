//! Error types for the NEREIDS library.

use thiserror::Error;

/// Physics computation errors.
#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("not implemented: {0}")]
    NotImplemented(String),

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("energy grid is empty")]
    EmptyEnergyGrid,

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Fitting/optimization errors.
#[derive(Debug, Error)]
pub enum FitError {
    #[error("not implemented: {0}")]
    NotImplemented(String),

    #[error("convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("singular matrix encountered during fitting")]
    SingularMatrix,

    #[error("physics error during fitting: {0}")]
    Physics(#[from] PhysicsError),
}

/// I/O errors.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("not implemented: {0}")]
    NotImplemented(String),

    #[error("file not found: {path}")]
    FileNotFound { path: String },

    #[error("parse error: {0}")]
    Parse(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Top-level error type composing all error categories.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Physics(#[from] PhysicsError),

    #[error(transparent)]
    Fit(#[from] FitError),

    #[error(transparent)]
    Io(#[from] IoError),
}
