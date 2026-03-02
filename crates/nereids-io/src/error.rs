//! Error types for the nereids-io crate.

/// Errors that can occur during I/O operations.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// File not found or could not be opened.
    #[error("File not found: {0}: {1}")]
    FileNotFound(String, #[source] std::io::Error),

    /// Path is not a directory.
    #[error("Not a directory: {0}")]
    NotADirectory(String),

    /// TIFF decoding error.
    #[error("TIFF decode error: {0}")]
    TiffDecode(String),

    /// Frame dimensions do not match.
    #[error("Dimension mismatch at frame {frame}: expected ({ew}×{eh}), got ({gw}×{gh})",
        ew = expected.0, eh = expected.1, gw = got.0, gh = got.1)]
    DimensionMismatch {
        expected: (u32, u32),
        got: (u32, u32),
        frame: usize,
    },

    /// Array shape error.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// No files matched the given pattern.
    #[error("No files matching pattern '{pattern}' in directory: {directory}")]
    NoMatchingFiles { directory: String, pattern: String },

    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// TIFF encoding/writing error.
    #[error("TIFF encode error: {0}")]
    TiffEncode(String),

    /// HDF5 format or access error.
    #[error("HDF5 error: {0}")]
    Hdf5Error(String),
}
