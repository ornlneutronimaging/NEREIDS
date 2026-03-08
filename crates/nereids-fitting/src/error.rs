//! Error types for the nereids-fitting crate.

use std::fmt;

/// Errors from fitting operations (input validation).
#[derive(Debug)]
pub enum FittingError {
    /// Observed data is empty.
    EmptyData,
    /// Array length mismatch.
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
        /// Name of the mismatched field.
        field: &'static str,
    },
    /// Invalid model configuration.
    InvalidConfig(String),
    /// Model evaluation failed (e.g. broadening error, invalid physics state).
    EvaluationFailed(String),
}

impl fmt::Display for FittingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => write!(f, "observed data must not be empty"),
            Self::LengthMismatch {
                expected,
                actual,
                field,
            } => write!(
                f,
                "{field} length ({actual}) must match expected length ({expected})"
            ),
            Self::InvalidConfig(msg) => write!(f, "invalid model configuration: {msg}"),
            Self::EvaluationFailed(msg) => write!(f, "model evaluation failed: {msg}"),
        }
    }
}

impl std::error::Error for FittingError {}
