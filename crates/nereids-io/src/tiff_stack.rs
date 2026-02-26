//! Multi-frame TIFF stack loading for neutron imaging data.
//!
//! VENUS beamline data is typically stored as multi-frame TIFF files where each
//! frame corresponds to a time-of-flight (TOF) bin.  The result is a 3D array
//! with dimensions (n_tof, height, width).
//!
//! ## Supported formats
//! - Single multi-frame TIFF (all TOF bins in one file)
//! - Directory of single-frame TIFFs (one file per TOF bin, sorted by name)
//!
//! ## Data types
//! - 16-bit unsigned integer (common for neutron detectors)
//! - 32-bit float (normalized data)

use std::path::Path;

use ndarray::Array3;
use tiff::decoder::Decoder;
use tiff::decoder::DecodingResult;

use crate::error::IoError;

/// Load a multi-frame TIFF into a 3D array (n_frames, height, width).
///
/// Each TIFF frame becomes one slice along the first axis.
/// Data is converted to `f64` regardless of the source pixel type.
///
/// # Arguments
/// * `path` — Path to the multi-frame TIFF file.
///
/// # Returns
/// 3D array with shape (n_frames, height, width) and f64 values.
pub fn load_tiff_stack(path: &Path) -> Result<Array3<f64>, IoError> {
    let file = std::fs::File::open(path)
        .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
    let mut decoder = Decoder::new(file).map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

    let mut frames: Vec<Vec<f64>> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;

    loop {
        let (w, h) = decoder
            .dimensions()
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

        if frames.is_empty() {
            width = w;
            height = h;
        } else if w != width || h != height {
            return Err(IoError::DimensionMismatch {
                expected: (width, height),
                got: (w, h),
                frame: frames.len(),
            });
        }

        let data = decoder
            .read_image()
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

        let pixels = decode_to_f64(data)?;
        let expected_len = (width as usize) * (height as usize);
        if pixels.len() != expected_len {
            return Err(IoError::TiffDecode(format!(
                "Frame {} has {} pixels, expected {}",
                frames.len(),
                pixels.len(),
                expected_len
            )));
        }
        frames.push(pixels);

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;
    }

    let n_frames = frames.len();
    if n_frames == 0 {
        return Err(IoError::TiffDecode("TIFF file contains no frames".into()));
    }

    // Flatten all frames into a single Vec and reshape to 3D
    let flat: Vec<f64> = frames.into_iter().flatten().collect();
    Array3::from_shape_vec((n_frames, height as usize, width as usize), flat)
        .map_err(|e| IoError::TiffDecode(format!("Shape error: {}", e)))
}

/// Load a directory of single-frame TIFFs as a 3D stack.
///
/// Files are sorted by name (lexicographic), so they should be named with
/// zero-padded indices (e.g., `frame_0001.tiff`, `frame_0002.tiff`, ...).
///
/// # Arguments
/// * `dir` — Path to the directory containing TIFF files.
///
/// # Returns
/// 3D array with shape (n_files, height, width) and f64 values.
pub fn load_tiff_directory(dir: &Path) -> Result<Array3<f64>, IoError> {
    if !dir.is_dir() {
        return Err(IoError::NotADirectory(dir.to_string_lossy().into_owned()));
    }

    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| IoError::FileNotFound(dir.to_string_lossy().into_owned(), e))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_lowercase().as_str(), "tif" | "tiff"))
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();

    paths.sort();

    if paths.is_empty() {
        return Err(IoError::TiffDecode(
            "No TIFF files found in directory".into(),
        ));
    }

    load_frames_from_paths(&paths)
}

/// Load a directory of TIFFs matching a glob pattern as a 3D stack.
///
/// Files are sorted lexicographically by name, so they should be named with
/// zero-padded indices (e.g., `frame_0001.tif`, `frame_0002.tif`, ...).
///
/// Only files with `.tif` or `.tiff` extensions (case-insensitive) are considered.
/// When `pattern` is `None`, all such files are loaded.  When `Some`, the pattern
/// is additionally matched against each filename (not the full path) and supports
/// `*` (matches any sequence of characters) and `?` (matches a single character).
/// Examples: `"*.tif"`, `"frame_*.tiff"`, `"scan_*"` (the extension guard still
/// applies, so non-TIFF files are never decoded).
///
/// # Arguments
/// * `dir`     — Path to the directory containing TIFF files.
/// * `pattern` — Optional glob pattern to filter filenames.
///
/// # Returns
/// 3D array with shape (n_files, height, width) and f64 values.
///
/// # Errors
/// * [`IoError::NoMatchingFiles`] if no files match the pattern.
/// * [`IoError::DimensionMismatch`] if frames have inconsistent dimensions.
pub fn load_tiff_folder(dir: &Path, pattern: Option<&str>) -> Result<Array3<f64>, IoError> {
    if !dir.is_dir() {
        return Err(IoError::NotADirectory(dir.to_string_lossy().into_owned()));
    }

    // Collect directory entries, propagating per-entry read errors instead of
    // silently dropping them (which could produce incomplete stacks).
    let entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| IoError::FileNotFound(dir.to_string_lossy().into_owned(), e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| IoError::FileNotFound(dir.to_string_lossy().into_owned(), e))?;

    let mut paths: Vec<_> = entries
        .iter()
        .filter(|entry| {
            // Use path().is_file() which follows symlinks, unlike file_type().is_file()
            let is_file = entry.path().is_file();
            let is_tiff = entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_lowercase().as_str(), "tif" | "tiff"))
                .unwrap_or(false);
            let matches_pattern = match pattern {
                Some(pat) => entry
                    .file_name()
                    .to_str()
                    .map(|name| glob_match(pat, name))
                    .unwrap_or(false),
                None => true,
            };
            is_file && is_tiff && matches_pattern
        })
        .map(|entry| entry.path())
        .collect();

    paths.sort();

    if paths.is_empty() {
        return Err(IoError::NoMatchingFiles {
            directory: dir.to_string_lossy().into_owned(),
            pattern: pattern.unwrap_or("*.tif / *.tiff").to_string(),
        });
    }

    load_frames_from_paths(&paths)
}

/// Shared helper: load a sorted slice of single-frame TIFF paths into a 3D array.
///
/// Each file must contain exactly one frame.  Dimensions are checked for
/// consistency across all files and pixel counts are validated against the
/// reported image dimensions.
fn load_frames_from_paths(paths: &[std::path::PathBuf]) -> Result<Array3<f64>, IoError> {
    let mut frames: Vec<Vec<f64>> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;

    for (i, path) in paths.iter().enumerate() {
        let file = std::fs::File::open(path)
            .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
        let mut decoder = Decoder::new(file).map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

        let (w, h) = decoder
            .dimensions()
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

        if i == 0 {
            width = w;
            height = h;
        } else if w != width || h != height {
            return Err(IoError::DimensionMismatch {
                expected: (width, height),
                got: (w, h),
                frame: i,
            });
        }

        let data = decoder
            .read_image()
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

        let pixels = decode_to_f64(data)?;
        let expected_len = (width as usize) * (height as usize);
        if pixels.len() != expected_len {
            return Err(IoError::TiffDecode(format!(
                "Frame {} has {} pixels, expected {}",
                i,
                pixels.len(),
                expected_len
            )));
        }
        frames.push(pixels);
    }

    let n_frames = frames.len();
    let flat: Vec<f64> = frames.into_iter().flatten().collect();
    Array3::from_shape_vec((n_frames, height as usize, width as usize), flat)
        .map_err(|e| IoError::TiffDecode(format!("Shape error: {}", e)))
}

/// Simple glob pattern matching against a filename.
///
/// Supports `*` (matches zero or more characters) and `?` (matches exactly one
/// Unicode character).  The match is case-insensitive to handle mixed-case
/// extensions (`.TIF`, `.Tiff`, etc.).
///
/// Uses an iterative two-pointer algorithm (O(p*n) worst case) to avoid
/// exponential blowup on pathological patterns like `*a*a*a*b`.
fn glob_match(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let n: Vec<char> = name.to_lowercase().chars().collect();

    let (mut pi, mut ni) = (0usize, 0usize);
    // Saved backtrack positions when we encounter a '*'.
    let (mut star_pi, mut star_ni) = (None::<usize>, 0usize);

    while ni < n.len() {
        if pi < p.len() && p[pi] == '*' {
            // Record the star position and current name index for backtracking.
            star_pi = Some(pi);
            star_ni = ni;
            pi += 1; // Try matching '*' with zero characters first.
        } else if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if let Some(sp) = star_pi {
            // Mismatch — backtrack: let the last '*' consume one more character.
            star_ni += 1;
            ni = star_ni;
            pi = sp + 1;
        } else {
            return false;
        }
    }

    // Consume any trailing '*' characters in the pattern.
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }

    pi == p.len()
}

/// Convert TIFF decoded data to f64 values.
fn decode_to_f64(data: DecodingResult) -> Result<Vec<f64>, IoError> {
    match data {
        DecodingResult::U8(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::U16(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::U32(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::U64(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::F32(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::F64(v) => Ok(v),
        DecodingResult::I8(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::I16(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::I32(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::I64(v) => Ok(v.into_iter().map(|x| x as f64).collect()),
        DecodingResult::F16(v) => Ok(v.into_iter().map(f64::from).collect()),
    }
}

/// Metadata about a loaded TIFF stack.
#[derive(Debug, Clone)]
pub struct TiffStackInfo {
    /// Number of TOF frames.
    pub n_frames: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
}

impl TiffStackInfo {
    /// Extract info from a loaded 3D array.
    pub fn from_array(arr: &Array3<f64>) -> Self {
        let shape = arr.shape();
        Self {
            n_frames: shape[0],
            height: shape[1],
            width: shape[2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiff::encoder::TiffEncoder;

    /// Create a minimal multi-frame TIFF for testing.
    fn write_test_tiff(path: &Path, frames: &[Vec<u16>], width: u32, height: u32) {
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = TiffEncoder::new(file).unwrap();
        for frame in frames {
            encoder
                .write_image::<tiff::encoder::colortype::Gray16>(width, height, frame)
                .unwrap();
        }
    }

    #[test]
    fn test_load_single_frame_tiff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tiff");

        // 3x2 image, single frame, values 1-6
        let data: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
        write_test_tiff(&path, &[data], 3, 2);

        let arr = load_tiff_stack(&path).unwrap();
        assert_eq!(arr.shape(), &[1, 2, 3]);
        assert_eq!(arr[[0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 0, 2]], 3.0);
        assert_eq!(arr[[0, 1, 0]], 4.0);
        assert_eq!(arr[[0, 1, 2]], 6.0);
    }

    #[test]
    fn test_load_multi_frame_tiff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tiff");

        let frame1: Vec<u16> = vec![10, 20, 30, 40];
        let frame2: Vec<u16> = vec![50, 60, 70, 80];
        let frame3: Vec<u16> = vec![90, 100, 110, 120];
        write_test_tiff(&path, &[frame1, frame2, frame3], 2, 2);

        let arr = load_tiff_stack(&path).unwrap();
        assert_eq!(arr.shape(), &[3, 2, 2]);
        // First frame
        assert_eq!(arr[[0, 0, 0]], 10.0);
        assert_eq!(arr[[0, 1, 1]], 40.0);
        // Third frame
        assert_eq!(arr[[2, 0, 0]], 90.0);
        assert_eq!(arr[[2, 1, 1]], 120.0);
    }

    #[test]
    fn test_load_tiff_directory() {
        let dir = tempfile::tempdir().unwrap();

        // Write 3 single-frame TIFFs
        for i in 0..3u16 {
            let path = dir.path().join(format!("frame_{:04}.tiff", i));
            let data: Vec<u16> = (0..4).map(|j| (i + 1) * 10 + j).collect();
            write_test_tiff(&path, &[data], 2, 2);
        }

        let arr = load_tiff_directory(dir.path()).unwrap();
        assert_eq!(arr.shape(), &[3, 2, 2]);
        // frame_0000: 10, 11, 12, 13
        assert_eq!(arr[[0, 0, 0]], 10.0);
        // frame_0002: 30, 31, 32, 33
        assert_eq!(arr[[2, 0, 0]], 30.0);
        assert_eq!(arr[[2, 1, 1]], 33.0);
    }

    #[test]
    fn test_load_tiff_folder_no_pattern() {
        let dir = tempfile::tempdir().unwrap();

        // Mix of .tif and .tiff — both should be picked up
        for i in 0..2u16 {
            let path = dir.path().join(format!("frame_{:04}.tif", i));
            let data: Vec<u16> = (0..4).map(|j| (i + 1) * 10 + j).collect();
            write_test_tiff(&path, &[data], 2, 2);
        }
        let path = dir.path().join("frame_0002.tiff");
        write_test_tiff(&path, &[vec![30, 31, 32, 33]], 2, 2);

        // Non-TIFF sidecar should be ignored
        std::fs::write(dir.path().join("frame_0001.tif.bak"), b"not a tiff").unwrap();

        let arr = load_tiff_folder(dir.path(), None).unwrap();
        assert_eq!(arr.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_load_tiff_folder_with_pattern() {
        let dir = tempfile::tempdir().unwrap();

        for i in 0..3u16 {
            let path = dir.path().join(format!("frame_{:04}.tif", i));
            let data: Vec<u16> = (0..4).map(|j| (i + 1) * 10 + j).collect();
            write_test_tiff(&path, &[data], 2, 2);
        }

        let arr = load_tiff_folder(dir.path(), Some("*.tif")).unwrap();
        assert_eq!(arr.shape(), &[3, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 10.0);
        assert_eq!(arr[[2, 1, 1]], 33.0);
    }

    #[test]
    fn test_load_tiff_folder_custom_pattern() {
        let dir = tempfile::tempdir().unwrap();

        // Write files matching "scan_*.tif" and a non-matching file
        for i in 0..2u16 {
            let path = dir.path().join(format!("scan_{:04}.tif", i));
            let data: Vec<u16> = (0..4).map(|j| (i + 1) * 10 + j).collect();
            write_test_tiff(&path, &[data], 2, 2);
        }
        // This file should NOT be matched by "scan_*.tif"
        let extra = dir.path().join("other_0001.tif");
        write_test_tiff(&extra, &[vec![99, 99, 99, 99]], 2, 2);

        let arr = load_tiff_folder(dir.path(), Some("scan_*.tif")).unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 10.0);
    }

    #[test]
    fn test_load_tiff_folder_no_matching_files() {
        let dir = tempfile::tempdir().unwrap();

        // Write a .tiff file but search for .png
        let path = dir.path().join("frame_0001.tiff");
        write_test_tiff(&path, &[vec![1, 2, 3, 4]], 2, 2);

        let result = load_tiff_folder(dir.path(), Some("*.png"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, IoError::NoMatchingFiles { .. }),
            "Expected NoMatchingFiles, got: {:?}",
            err,
        );
    }

    #[test]
    fn test_load_tiff_folder_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();

        // Write a file with uppercase extension
        let path = dir.path().join("frame_0001.TIF");
        write_test_tiff(&path, &[vec![1, 2, 3, 4]], 2, 2);

        // Pattern with lowercase should still match
        let arr = load_tiff_folder(dir.path(), Some("*.tif")).unwrap();
        assert_eq!(arr.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_glob_match_basic() {
        assert!(glob_match("*.tif", "frame_0001.tif"));
        assert!(glob_match("*.tif", "a.tif"));
        assert!(!glob_match("*.tif", "frame_0001.tiff"));
        assert!(!glob_match("*.tif", "frame_0001.png"));
    }

    #[test]
    fn test_glob_match_question_mark() {
        assert!(glob_match("frame_?.tif", "frame_1.tif"));
        assert!(!glob_match("frame_?.tif", "frame_12.tif"));
        // '?' should match a single Unicode character, not a single byte
        assert!(glob_match("?.tif", "\u{00e9}.tif")); // é is multi-byte in UTF-8
    }

    #[test]
    fn test_glob_match_case_insensitive() {
        assert!(glob_match("*.tif", "FILE.TIF"));
        assert!(glob_match("*.TIF", "file.tif"));
    }

    #[test]
    fn test_glob_match_pattern_longer_than_name() {
        assert!(!glob_match("abcdef.tif", "a.tif"));
    }

    #[test]
    fn test_glob_match_pathological_pattern() {
        // Verify the iterative matcher handles patterns that would cause
        // exponential blowup in a naive recursive implementation.
        let pattern = "*a*a*a*a*a*b";
        let name = "aaaaaaaaaaaaaaaaaaaac";
        assert!(!glob_match(pattern, name));
    }

    #[test]
    fn test_load_tiff_folder_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_tiff_folder(dir.path(), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, IoError::NoMatchingFiles { .. }),
            "Expected NoMatchingFiles, got: {:?}",
            err,
        );
    }

    #[test]
    fn test_load_tiff_folder_not_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("frame_0001.tif");
        write_test_tiff(&file_path, &[vec![1, 2, 3, 4]], 2, 2);

        let result = load_tiff_folder(&file_path, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, IoError::NotADirectory(..)),
            "Expected NotADirectory, got: {:?}",
            err,
        );
    }

    #[test]
    fn test_load_tiff_folder_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();

        // Frame 0: 2x2
        write_test_tiff(
            &dir.path().join("frame_0000.tif"),
            &[vec![1, 2, 3, 4]],
            2,
            2,
        );
        // Frame 1: 3x2 — different width
        write_test_tiff(
            &dir.path().join("frame_0001.tif"),
            &[vec![1, 2, 3, 4, 5, 6]],
            3,
            2,
        );

        let result = load_tiff_folder(dir.path(), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, IoError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {:?}",
            err,
        );
    }

    #[test]
    fn test_nonexistent_file() {
        let result = load_tiff_stack(Path::new("/nonexistent/file.tiff"));
        assert!(result.is_err());
    }

    #[test]
    fn test_tiff_stack_info() {
        let arr = Array3::<f64>::zeros((10, 512, 512));
        let info = TiffStackInfo::from_array(&arr);
        assert_eq!(info.n_frames, 10);
        assert_eq!(info.height, 512);
        assert_eq!(info.width, 512);
    }
}
