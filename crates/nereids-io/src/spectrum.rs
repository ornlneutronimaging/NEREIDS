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
//!
//! ## VENUS `*_Spectra.txt` sidecars
//!
//! Autoreduced VENUS TIFF folders ship a `<run>_Spectra.txt` sidecar whose
//! first column is each frame's *start time in seconds* (N rows for N
//! frames; the second column is counts).  [`read_tof_sidecar`] converts it
//! to the N+1 ascending TOF bin edges **in microseconds** that
//! [`crate::tof::tof_edges_to_energy_centers`] expects.

use std::path::Path;

use crate::error::IoError;

/// Microseconds per second — sidecar start times are recorded in seconds,
/// while every NEREIDS TOF axis is in microseconds.
pub const MICROSECONDS_PER_SECOND: f64 = 1e6;

/// Read a VENUS `*_Spectra.txt` TOF sidecar into bin edges (µs).
///
/// See [`parse_tof_sidecar_text`] for format semantics and validation.
///
/// # Arguments
/// * `path`     — Path to the sidecar file.
/// * `n_frames` — When `Some(n)`, the resulting edge count is validated
///   against the TIFF stack's frame count (`n + 1` edges for `n` frames).
pub fn read_tof_sidecar(path: &Path, n_frames: Option<usize>) -> Result<Vec<f64>, IoError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
    parse_tof_sidecar_text(&content, n_frames)
}

/// Parse VENUS `*_Spectra.txt` sidecar text into TOF bin edges (µs).
///
/// Format: CSV `shutter_time,counts` where column 0 is the frame **start
/// time in seconds** — one row per TOF frame.  Comment lines (`#`), blank
/// lines, and a single header row are tolerated (the
/// [`parse_spectrum_text`] rules).
///
/// Processing:
/// 1. start times must be finite, the first must be `>= 0`, and the
///    sequence must be strictly increasing;
/// 2. values are converted to microseconds ([`MICROSECONDS_PER_SECOND`]);
/// 3. the closing edge of the last frame is synthesized by extrapolating
///    the *last* frame width (`last + (last − prev)`), yielding N+1
///    ascending edges for N rows.
///
/// Bin **uniformity is deliberately not enforced**: VENUS MCP shutter
/// segments change the frame width mid-run, so a sidecar with several
/// distinct widths is valid.  The last-segment-width extrapolation is
/// exact whenever the final two frames belong to the same shutter segment
/// (always the case in practice — segments are many frames long).
///
/// The returned edges plug directly into
/// [`crate::tof::tof_edges_to_energy_centers`].
///
/// # Errors
/// [`IoError::InvalidParameter`] on fewer than 2 rows, non-finite or
/// unparseable values, a negative first start time, a non-increasing
/// sequence, or (when `n_frames` is `Some`) an edge/frame count mismatch.
pub fn parse_tof_sidecar_text(text: &str, n_frames: Option<usize>) -> Result<Vec<f64>, IoError> {
    let starts_s = parse_spectrum_text(text)?;
    if starts_s[0] < 0.0 {
        return Err(IoError::InvalidParameter(format!(
            "TOF sidecar start times must be >= 0 s, but the first is {}",
            starts_s[0],
        )));
    }
    validate_monotonic(&starts_s)?;

    let mut edges: Vec<f64> = Vec::with_capacity(starts_s.len() + 1);
    edges.extend(starts_s.iter().map(|s| s * MICROSECONDS_PER_SECOND));
    // parse_spectrum_text guarantees >= 2 values, so [n-2] is in bounds.
    let last = edges[edges.len() - 1];
    let last_width = last - edges[edges.len() - 2];
    edges.push(last + last_width);

    if let Some(frames) = n_frames {
        validate_spectrum_frame_count(edges.len(), frames, SpectrumValueKind::BinEdges)?;
    }
    Ok(edges)
}

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
///
/// # Assumptions
///
/// - **Column semantics**: only the first numeric column is extracted; any
///   additional columns (e.g., counts, intensity) are silently ignored.
/// - **Units are not inferred**: the caller must know whether values represent
///   TOF in microseconds or energy in eV and set [`SpectrumUnit`] accordingly.
/// - **Malformed lines**: comment lines (`#`-prefixed) and blank lines are
///   skipped.  The first non-comment, non-numeric line is treated as a header
///   and skipped; a second such line is a hard error.
/// - **Non-finite values** (NaN, Inf) produce a hard error.
/// - **Minimum length**: at least 2 values are required.
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

    /// T23: 3 rows of start times in seconds become 4 ascending µs edges.
    /// Values chosen binary-exact so seconds → µs conversion is exact.
    #[test]
    fn test_sidecar_three_rows_exact_edges() {
        let text = "0.5,100\n1.0,200\n1.5,300\n";
        let edges = parse_tof_sidecar_text(text, None).unwrap();
        assert_eq!(
            edges,
            vec![500_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
        );
        assert_eq!(edges.len(), 3 + 1);
    }

    /// T24: a header row is tolerated (one non-numeric first line).
    #[test]
    fn test_sidecar_header_row_tolerated() {
        let text = "shutter_time,counts\n0.5,100\n1.0,200\n1.5,300\n";
        let edges = parse_tof_sidecar_text(text, None).unwrap();
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[0], 500_000.0);
    }

    /// T25: n_frames validation — matching count passes, mismatch errors.
    #[test]
    fn test_sidecar_frame_count_validation() {
        let text = "0.5,100\n1.0,200\n1.5,300\n";
        assert!(parse_tof_sidecar_text(text, Some(3)).is_ok());
        let err = parse_tof_sidecar_text(text, Some(4)).unwrap_err();
        assert!(
            matches!(err, IoError::InvalidParameter(_)),
            "Expected InvalidParameter, got: {:?}",
            err,
        );
    }

    /// T26: non-monotonic start times are rejected.
    #[test]
    fn test_sidecar_non_monotonic_rejected() {
        let text = "0.5,100\n1.5,200\n1.0,300\n";
        assert!(parse_tof_sidecar_text(text, None).is_err());
    }

    /// T27: a NaN row is rejected.
    #[test]
    fn test_sidecar_nan_rejected() {
        let text = "0.5,100\nNaN,200\n1.5,300\n";
        assert!(parse_tof_sidecar_text(text, None).is_err());
    }

    /// T28: a single row cannot define a bin width — rejected.
    #[test]
    fn test_sidecar_single_row_rejected() {
        let text = "0.5,100\n";
        assert!(parse_tof_sidecar_text(text, None).is_err());
    }

    /// T29: a negative first start time is rejected.
    #[test]
    fn test_sidecar_negative_first_start_rejected() {
        let text = "-0.5,100\n0.5,200\n1.0,300\n";
        let err = parse_tof_sidecar_text(text, None).unwrap_err();
        assert!(
            format!("{}", err).contains(">= 0"),
            "Expected >= 0 message, got: {}",
            err,
        );
    }

    /// T30: non-uniform shutter segments (64 µs then 128 µs frames) are
    /// accepted, and the synthesized final edge extrapolates the *last*
    /// segment's width.
    #[test]
    fn test_sidecar_shutter_segments_last_width_extrapolation() {
        // Starts (s): 0, 64 µs, 192 µs — widths 64 µs then 128 µs.
        let text = "0.0,10\n0.000064,20\n0.000192,30\n";
        let edges = parse_tof_sidecar_text(text, None).unwrap();
        assert_eq!(edges.len(), 4);
        // The synthesized edge uses exactly the last frame width
        // (edges[2] - edges[1]), not the first segment's 64 µs.
        assert_eq!(edges[3], edges[2] + (edges[2] - edges[1]));
        let expected = [0.0, 64.0, 192.0, 320.0];
        for (edge, want) in edges.iter().zip(expected.iter()) {
            assert!(
                (edge - want).abs() < 1e-9,
                "edge {} != expected {}",
                edge,
                want,
            );
        }
    }

    /// T31: a missing sidecar file surfaces as FileNotFound.
    #[test]
    fn test_sidecar_missing_file() {
        let err = read_tof_sidecar(Path::new("/nonexistent/run_Spectra.txt"), None).unwrap_err();
        assert!(
            matches!(err, IoError::FileNotFound(..)),
            "Expected FileNotFound, got: {:?}",
            err,
        );
    }
}
