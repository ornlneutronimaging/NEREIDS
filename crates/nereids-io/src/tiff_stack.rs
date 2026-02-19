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
    let mut decoder = Decoder::new(file)
        .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

    let mut frames: Vec<Vec<f64>> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;

    loop {
        let (w, h) = decoder.dimensions()
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

        let data = decoder.read_image()
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
        decoder.next_image()
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;
    }

    let n_frames = frames.len();
    if n_frames == 0 {
        return Err(IoError::TiffDecode("TIFF file contains no frames".into()));
    }

    // Flatten all frames into a single Vec and reshape to 3D
    let flat: Vec<f64> = frames.into_iter().flatten().collect();
    Array3::from_shape_vec(
        (n_frames, height as usize, width as usize),
        flat,
    )
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
            entry.path().extension()
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

    let mut frames: Vec<Vec<f64>> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;

    for (i, path) in paths.iter().enumerate() {
        let file = std::fs::File::open(path)
            .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
        let mut decoder = Decoder::new(file)
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

        let (w, h) = decoder.dimensions()
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

        let data = decoder.read_image()
            .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;
        frames.push(decode_to_f64(data)?);
    }

    let n_frames = frames.len();
    let flat: Vec<f64> = frames.into_iter().flatten().collect();
    Array3::from_shape_vec(
        (n_frames, height as usize, width as usize),
        flat,
    )
    .map_err(|e| IoError::TiffDecode(format!("Shape error: {}", e)))
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
        DecodingResult::F16(v) => Ok(v.into_iter().map(|x| f64::from(x)).collect()),
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
            encoder.write_image::<tiff::encoder::colortype::Gray16>(
                width, height, frame,
            ).unwrap();
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
