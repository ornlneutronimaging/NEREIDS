//! Spectrum file parser for TOF/energy bin edges or centers.
//!
//! Parses CSV/TXT files containing TOF or energy values that define the
//! spectral bins of a neutron imaging dataset.
//!
//! ## Supported formats
//! - Single-column: one value per line
//! - Two-column (CSV/TSV): first column used, rest ignored
//! - Comment lines starting with `#` are skipped
//! - First non-comment line skipped if it cannot be parsed as a number (header)

use std::path::Path;

use crate::error::IoError;

/// Whether spectrum values represent TOF or energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumUnit {
    /// Values are TOF bin edges/centers in microseconds.
    TofMicroseconds,
    /// Values are energy bin edges/centers in eV.
    EnergyEv,
}

/// Whether values are bin edges (N+1 for N bins) or bin centers (N for N bins).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumValueKind {
    /// N+1 values defining the boundaries of N bins.
    BinEdges,
    /// N values at the center of each bin.
    BinCenters,
}

/// Parse a spectrum file from disk.
///
/// Returns the first column of numeric values, skipping comment and header lines.
/// Supports comma, tab, and whitespace as delimiters.
pub fn parse_spectrum_file(path: &Path) -> Result<Vec<f64>, IoError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
    parse_spectrum_text(&content)
}

/// Parse spectrum values from a string.
///
/// Extracts the first numeric column. Lines starting with `#` are comments.
/// The first non-comment line that cannot be parsed as a number is treated
/// as a header and skipped (only one such line is allowed).
pub fn parse_spectrum_text(text: &str) -> Result<Vec<f64>, IoError> {
    let mut values = Vec::new();
    let mut skipped_header = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Extract first token (split by comma, tab, or whitespace)
        let first_token = trimmed
            .split(|c: char| c == ',' || c == '\t' || c.is_ascii_whitespace())
            .next()
            .unwrap_or("")
            .trim();

        match first_token.parse::<f64>() {
            Ok(val) => {
                if !val.is_finite() {
                    return Err(IoError::InvalidParameter(format!(
                        "Non-finite value in spectrum file: {}",
                        val
                    )));
                }
                values.push(val);
            }
            Err(_) => {
                if !skipped_header && values.is_empty() {
                    skipped_header = true;
                    continue;
                }
                return Err(IoError::InvalidParameter(format!(
                    "Unparseable value in spectrum file: '{}'",
                    first_token
                )));
            }
        }
    }

    if values.len() < 2 {
        return Err(IoError::InvalidParameter(
            "Spectrum file must contain at least 2 values".into(),
        ));
    }

    Ok(values)
}

/// Validate that spectrum values are compatible with the TIFF frame count.
///
/// For bin edges: `n_values == n_frames + 1`.
/// For bin centers: `n_values == n_frames`.
pub fn validate_spectrum_frame_count(
    n_values: usize,
    n_frames: usize,
    kind: SpectrumValueKind,
) -> Result<(), IoError> {
    let expected = match kind {
        SpectrumValueKind::BinEdges => n_frames + 1,
        SpectrumValueKind::BinCenters => n_frames,
    };
    if n_values != expected {
        return Err(IoError::InvalidParameter(format!(
            "Spectrum has {} values but TIFF has {} frames (expected {} for {:?})",
            n_values, n_frames, expected, kind,
        )));
    }
    Ok(())
}

/// Validate that values are strictly monotonically increasing.
pub fn validate_monotonic(values: &[f64]) -> Result<(), IoError> {
    for window in values.windows(2) {
        match window[0].partial_cmp(&window[1]) {
            Some(std::cmp::Ordering::Less) => {} // strictly increasing — OK
            _ => {
                // Equal, decreasing, or NaN (partial_cmp returns None)
                return Err(IoError::InvalidParameter(format!(
                    "Spectrum values must be strictly increasing, but found {} >= {}",
                    window[0], window[1],
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_column() {
        let text = "1000.0\n2000.0\n3000.0\n4000.0\n";
        let values = parse_spectrum_text(text).unwrap();
        assert_eq!(values, vec![1000.0, 2000.0, 3000.0, 4000.0]);
    }

    #[test]
    fn test_parse_two_column_csv() {
        let text = "1000.0,0.5\n2000.0,0.6\n3000.0,0.7\n";
        let values = parse_spectrum_text(text).unwrap();
        assert_eq!(values, vec![1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn test_parse_whitespace_separated() {
        let text = "1000.0  0.5\n2000.0  0.6\n3000.0  0.7\n";
        let values = parse_spectrum_text(text).unwrap();
        assert_eq!(values, vec![1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn test_parse_comments_and_header() {
        let text = "\
# This is a comment
# Another comment
TOF_us, intensity
1000.0, 0.5
2000.0, 0.6
3000.0, 0.7
";
        let values = parse_spectrum_text(text).unwrap();
        assert_eq!(values, vec![1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn test_parse_tab_separated() {
        let text = "1000.0\t0.5\n2000.0\t0.6\n3000.0\t0.7\n";
        let values = parse_spectrum_text(text).unwrap();
        assert_eq!(values, vec![1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn test_parse_empty_lines_ignored() {
        let text = "\n1000.0\n\n2000.0\n\n3000.0\n\n";
        let values = parse_spectrum_text(text).unwrap();
        assert_eq!(values, vec![1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn test_parse_too_few_values() {
        let text = "1000.0\n";
        let result = parse_spectrum_text(text);
        assert!(result.is_err());
        assert!(
            format!("{}", result.unwrap_err()).contains("at least 2"),
            "Expected 'at least 2' error"
        );
    }

    #[test]
    fn test_parse_non_finite_value() {
        let text = "1000.0\nNaN\n3000.0\n";
        let result = parse_spectrum_text(text);
        assert!(result.is_err());
        assert!(
            format!("{}", result.unwrap_err()).contains("Non-finite"),
            "Expected non-finite error"
        );
    }

    #[test]
    fn test_parse_unparseable_after_data() {
        let text = "1000.0\n2000.0\nbad_value\n";
        let result = parse_spectrum_text(text);
        assert!(result.is_err());
        assert!(
            format!("{}", result.unwrap_err()).contains("Unparseable"),
            "Expected unparseable error"
        );
    }

    #[test]
    fn test_validate_frame_count_edges() {
        // 5 frames need 6 edges
        assert!(validate_spectrum_frame_count(6, 5, SpectrumValueKind::BinEdges).is_ok());
        assert!(validate_spectrum_frame_count(5, 5, SpectrumValueKind::BinEdges).is_err());
        assert!(validate_spectrum_frame_count(7, 5, SpectrumValueKind::BinEdges).is_err());
    }

    #[test]
    fn test_validate_frame_count_centers() {
        // 5 frames need 5 centers
        assert!(validate_spectrum_frame_count(5, 5, SpectrumValueKind::BinCenters).is_ok());
        assert!(validate_spectrum_frame_count(6, 5, SpectrumValueKind::BinCenters).is_err());
    }

    #[test]
    fn test_validate_monotonic_ok() {
        assert!(validate_monotonic(&[1.0, 2.0, 3.0, 4.0]).is_ok());
    }

    #[test]
    fn test_validate_monotonic_equal() {
        let result = validate_monotonic(&[1.0, 2.0, 2.0, 4.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_monotonic_decreasing() {
        let result = validate_monotonic(&[1.0, 3.0, 2.0, 4.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_monotonic_nan() {
        let result = validate_monotonic(&[1.0, f64::NAN, 3.0]);
        assert!(result.is_err(), "NaN should fail monotonicity check");
    }

    #[test]
    fn test_parse_spectrum_file_not_found() {
        let result = parse_spectrum_file(Path::new("/nonexistent/spectrum.csv"));
        assert!(result.is_err());
    }
}
