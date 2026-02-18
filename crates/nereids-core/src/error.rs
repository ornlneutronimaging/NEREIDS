//! Error types for the NEREIDS library.

/// Top-level error type for all NEREIDS operations.
#[derive(Debug, thiserror::Error)]
pub enum NereidsError {
    #[error("ENDF error: {0}")]
    Endf(String),

    #[error("Physics error: {0}")]
    Physics(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Fitting error: {0}")]
    Fitting(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Convenience type alias for NEREIDS results.
pub type NereidsResult<T> = Result<T, NereidsError>;
