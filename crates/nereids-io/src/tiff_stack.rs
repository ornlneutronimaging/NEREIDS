//! Multi-frame TIFF stack loading for neutron imaging data.
//!
//! VENUS beamline data is typically stored as multi-frame TIFF files where each
//! frame corresponds to a time-of-flight (TOF) bin.  The result is a 3D array
//! with dimensions (n_tof, height, width).
//!
//! ## Supported formats
//! - Single multi-frame TIFF (all TOF bins in one file)
//! - Directory of single-frame TIFFs (one file per TOF bin, sorted by name)
//! - Chunked VENUS autoreduced folder: files named
//!   `<prefix>_<chunk>_<frame>.tif(f)` where the DAQ split one run into
//!   several chunks that each cover the full TOF frame range
//!
//! ## Chunked folders
//!
//! When *every* filename stem in the (extension- and pattern-filtered) folder
//! parses as `<prefix>_<chunk>_<frame>` with a single common prefix, the
//! folder is treated as chunked:
//! - one chunk → frames are ordered by *numeric* frame index (identical to
//!   lexicographic order for zero-padded names, and strictly better for
//!   unpadded ones where `_10` sorts before `_2` lexicographically);
//! - two or more chunks with identical frame-index sequences → chunks are
//!   summed element-wise by default (each chunk covers the same TOF bins, so
//!   the physical stack is the sum, not a concatenation).  Opt out with
//!   [`TiffFolderOptions::sum_chunks`]` = false` to get the legacy
//!   lexicographic concatenation;
//! - ragged chunks (differing frame counts or frame sets) or duplicate
//!   (chunk, frame) pairs → hard [`IoError::ChunkMismatch`] error, never a
//!   silent stack or partial sum.
//!
//! Folders with two or more distinct prefixes fall back to legacy
//! lexicographic stacking — summing across different prefixes would merge
//! different runs.  Use the `pattern` argument to select one run.
//!
//! ## Data types
//! - 16-bit unsigned integer (common for neutron detectors)
//! - 32-bit float (normalized data)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ndarray::{Array3, s};
use tiff::decoder::Decoder;
use tiff::decoder::DecodingResult;

use crate::error::IoError;

/// Options controlling how a TIFF folder (or file) is loaded.
#[derive(Debug, Clone, Copy)]
pub struct TiffFolderOptions {
    /// Sum DAQ chunks element-wise when a chunked folder is detected
    /// (default `true`).  When `false`, a chunked folder is loaded as the
    /// legacy lexicographic concatenation of all files.
    pub sum_chunks: bool,
}

impl Default for TiffFolderOptions {
    fn default() -> Self {
        Self { sum_chunks: true }
    }
}

/// Provenance metadata about a completed TIFF load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiffLoadInfo {
    /// Number of TIFF files read (1 for a single multi-frame file).
    pub n_files: usize,
    /// Number of DAQ chunks detected (0 when the folder does not follow the
    /// chunked `<prefix>_<chunk>_<frame>` naming convention).
    pub n_chunks: usize,
    /// Detected chunk identifiers, ascending (empty when `n_chunks == 0`).
    pub chunk_ids: Vec<u64>,
    /// Whether chunks were summed element-wise into a single stack.
    pub chunks_summed: bool,
}

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

/// Load TIFF data from either a single multi-frame file or a directory.
///
/// Auto-detects based on whether `path` is a file or directory:
/// - File → [`load_tiff_stack`] (multi-frame TIFF)
/// - Directory → [`load_tiff_directory`] (one file per frame)
///
/// # Arguments
/// * `path` — Path to either a multi-frame TIFF file or a directory of TIFFs.
///
/// # Returns
/// 3D array with shape (n_frames, height, width) and f64 values.
pub fn load_tiff_auto(path: &Path) -> Result<Array3<f64>, IoError> {
    match std::fs::metadata(path) {
        Ok(meta) => {
            if meta.is_file() {
                load_tiff_stack(path)
            } else if meta.is_dir() {
                load_tiff_directory(path)
            } else {
                Err(IoError::FileNotFound(
                    path.to_string_lossy().into_owned(),
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "path is neither a regular file nor a directory",
                    ),
                ))
            }
        }
        Err(e) => Err(IoError::FileNotFound(
            path.to_string_lossy().into_owned(),
            e,
        )),
    }
}

/// Load TIFF data from a file or directory, returning provenance metadata.
///
/// Auto-detects based on whether `path` is a file or directory, like
/// [`load_tiff_auto`], but additionally applies [`TiffFolderOptions`] (chunk
/// summing for directories) and reports what was done via [`TiffLoadInfo`].
///
/// # Arguments
/// * `path`    — Path to either a multi-frame TIFF file or a directory of TIFFs.
/// * `options` — Loading options (chunk summing).
///
/// # Returns
/// `(stack, info)` where `stack` has shape (n_frames, height, width).
pub fn load_tiff_auto_with_options(
    path: &Path,
    options: &TiffFolderOptions,
) -> Result<(Array3<f64>, TiffLoadInfo), IoError> {
    match std::fs::metadata(path) {
        Ok(meta) => {
            if meta.is_file() {
                let arr = load_tiff_stack(path)?;
                Ok((
                    arr,
                    TiffLoadInfo {
                        n_files: 1,
                        n_chunks: 0,
                        chunk_ids: Vec::new(),
                        chunks_summed: false,
                    },
                ))
            } else if meta.is_dir() {
                load_tiff_folder_with_options(path, None, options)
            } else {
                Err(IoError::FileNotFound(
                    path.to_string_lossy().into_owned(),
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "path is neither a regular file nor a directory",
                    ),
                ))
            }
        }
        Err(e) => Err(IoError::FileNotFound(
            path.to_string_lossy().into_owned(),
            e,
        )),
    }
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
    load_tiff_folder(dir, None).map_err(|e| match e {
        // Preserve the original error message for backward compatibility.
        IoError::NoMatchingFiles { .. } => {
            IoError::TiffDecode("No TIFF files found in directory".into())
        }
        other => other,
    })
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
/// * [`IoError::ChunkMismatch`] if a chunked folder is internally inconsistent.
pub fn load_tiff_folder(dir: &Path, pattern: Option<&str>) -> Result<Array3<f64>, IoError> {
    load_tiff_folder_with_options(dir, pattern, &TiffFolderOptions::default()).map(|(arr, _)| arr)
}

/// Load a directory of TIFFs matching a glob pattern, returning provenance
/// metadata.
///
/// Behaves like [`load_tiff_folder`] (same extension guard and glob pattern
/// semantics), with two additions:
/// - chunked-folder detection and element-wise chunk summing (see the
///   [module docs](self) and [`TiffFolderOptions::sum_chunks`]);
/// - a [`TiffLoadInfo`] report of what was loaded.
///
/// # Arguments
/// * `dir`     — Path to the directory containing TIFF files.
/// * `pattern` — Optional glob pattern to filter filenames.
/// * `options` — Loading options (chunk summing).
///
/// # Returns
/// `(stack, info)` where `stack` has shape (n_frames, height, width).
///
/// # Errors
/// * [`IoError::NoMatchingFiles`] if no files match the pattern.
/// * [`IoError::DimensionMismatch`] if frames have inconsistent dimensions.
/// * [`IoError::ChunkMismatch`] if a chunked folder is internally
///   inconsistent (ragged chunks or duplicate (chunk, frame) pairs).
pub fn load_tiff_folder_with_options(
    dir: &Path,
    pattern: Option<&str>,
    options: &TiffFolderOptions,
) -> Result<(Array3<f64>, TiffLoadInfo), IoError> {
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
        .filter_map(|entry| {
            // Compute path once to avoid repeated PathBuf allocations.
            let p = entry.path();
            // Use path().is_file() which follows symlinks, unlike file_type().is_file()
            if !p.is_file() {
                return None;
            }
            let is_tiff = p
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_lowercase().as_str(), "tif" | "tiff"))
                .unwrap_or(false);
            if !is_tiff {
                return None;
            }
            if let Some(pat) = pattern {
                let matches = entry
                    .file_name()
                    .to_str()
                    .map(|name| glob_match(pat, name))
                    .unwrap_or(false);
                if !matches {
                    return None;
                }
            }
            Some(p)
        })
        .collect();

    if paths.is_empty() {
        return Err(IoError::NoMatchingFiles {
            directory: dir.to_string_lossy().into_owned(),
            pattern: pattern.unwrap_or("*.tif / *.tiff").to_string(),
        });
    }

    let n_files = paths.len();

    match detect_chunk_layout(dir, &paths)? {
        ChunkLayout::Legacy => {
            paths.sort();
            let arr = load_frames_from_paths(&paths)?;
            Ok((
                arr,
                TiffLoadInfo {
                    n_files,
                    n_chunks: 0,
                    chunk_ids: Vec::new(),
                    chunks_summed: false,
                },
            ))
        }
        ChunkLayout::Chunked(chunks) => {
            let chunk_ids: Vec<u64> = chunks.keys().copied().collect();
            let n_chunks = chunks.len();
            if n_chunks == 1 || options.sum_chunks {
                // Single chunk loads in ascending numeric frame order (a
                // strict improvement over lexicographic order for unpadded
                // frame numbers); multiple chunks additionally sum
                // element-wise across chunks.
                let arr = load_chunked_sum(&chunks)?;
                Ok((
                    arr,
                    TiffLoadInfo {
                        n_files,
                        n_chunks,
                        chunk_ids,
                        chunks_summed: n_chunks > 1,
                    },
                ))
            } else {
                // Chunk summing opted out: legacy lexicographic concatenation
                // of every matching file (chunk structure is still reported).
                paths.sort();
                let arr = load_frames_from_paths(&paths)?;
                Ok((
                    arr,
                    TiffLoadInfo {
                        n_files,
                        n_chunks,
                        chunk_ids,
                        chunks_summed: false,
                    },
                ))
            }
        }
    }
}

/// Detected layout of a TIFF folder's (filtered) file list.
enum ChunkLayout {
    /// Not a chunked folder — load in lexicographic filename order.
    Legacy,
    /// Chunked folder: chunk id → frames as `(frame index, path)`, with
    /// chunk ids ascending (BTreeMap) and frames sorted ascending by index.
    /// Every chunk is validated to cover the identical frame-index sequence.
    Chunked(BTreeMap<u64, Vec<(u64, PathBuf)>>),
}

/// Parse a numeric filename field: non-empty, all ASCII digits, within `u64`
/// range.  Overflow (or any non-digit) yields `None` so the caller falls back
/// to legacy lexicographic loading rather than guessing.
fn parse_ascii_digits(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<u64>().ok()
}

/// Parse a filename stem of the chunked form `<prefix>_<chunk>_<frame>`.
///
/// Splits from the right (the prefix itself may contain underscores), and
/// requires both numeric fields to be non-empty ASCII digit runs.  Returns
/// `None` when the stem does not follow the convention.
fn parse_chunked_stem(stem: &str) -> Option<(&str, u64, u64)> {
    let (rest, frame_str) = stem.rsplit_once('_')?;
    let (prefix, chunk_str) = rest.rsplit_once('_')?;
    let frame = parse_ascii_digits(frame_str)?;
    let chunk = parse_ascii_digits(chunk_str)?;
    Some((prefix, chunk, frame))
}

/// Classify a filtered file list as legacy or chunked.
///
/// Legacy fallbacks (no error): any stem that does not parse as
/// `<prefix>_<chunk>_<frame>`, or two or more distinct prefixes (summing
/// across prefixes would merge different runs; use `pattern` to select one).
///
/// Hard errors ([`IoError::ChunkMismatch`]): duplicate (chunk, frame) pairs
/// (e.g. the same stem with both `.tif` and `.tiff` extensions, or `_764_`
/// alongside `_0764_`), and ragged chunks (differing frame counts or frame
/// sets).  Chunk ids need *not* be consecutive — a dropped middle chunk is
/// still the same run.
fn detect_chunk_layout(dir: &Path, paths: &[PathBuf]) -> Result<ChunkLayout, IoError> {
    let mut parsed: Vec<(&str, u64, u64, &PathBuf)> = Vec::with_capacity(paths.len());
    for path in paths {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            return Ok(ChunkLayout::Legacy);
        };
        let Some((prefix, chunk, frame)) = parse_chunked_stem(stem) else {
            return Ok(ChunkLayout::Legacy);
        };
        parsed.push((prefix, chunk, frame, path));
    }

    let first_prefix = parsed[0].0;
    if parsed.iter().any(|(prefix, ..)| *prefix != first_prefix) {
        return Ok(ChunkLayout::Legacy);
    }

    let mut chunks: BTreeMap<u64, Vec<(u64, PathBuf)>> = BTreeMap::new();
    for (_, chunk, frame, path) in parsed {
        chunks.entry(chunk).or_default().push((frame, path.clone()));
    }

    // Sort each chunk's frames by numeric index and reject duplicates.
    for (chunk_id, frames) in &mut chunks {
        frames.sort_by_key(|(frame, _)| *frame);
        if let Some(pair) = frames.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(IoError::ChunkMismatch {
                directory: dir.to_string_lossy().into_owned(),
                details: format!(
                    "duplicate frame {} in chunk {}: '{}' and '{}'",
                    pair[0].0,
                    chunk_id,
                    pair[0].1.display(),
                    pair[1].1.display(),
                ),
            });
        }
    }

    // Every chunk must cover the identical frame-index sequence; summing
    // ragged chunks would silently corrupt counts in the missing frames.
    let mut iter = chunks.iter();
    let (first_id, first_frames) = iter.next().expect("paths is non-empty");
    for (chunk_id, frames) in iter {
        if frames.len() != first_frames.len() {
            return Err(IoError::ChunkMismatch {
                directory: dir.to_string_lossy().into_owned(),
                details: format!(
                    "chunk {} has {} frames but chunk {} has {} frames",
                    first_id,
                    first_frames.len(),
                    chunk_id,
                    frames.len(),
                ),
            });
        }
        if let Some((a, b)) = first_frames
            .iter()
            .zip(frames.iter())
            .find(|(a, b)| a.0 != b.0)
        {
            return Err(IoError::ChunkMismatch {
                directory: dir.to_string_lossy().into_owned(),
                details: format!(
                    "chunks {} and {} cover different frame indices: first difference {} vs {}",
                    first_id, chunk_id, a.0, b.0,
                ),
            });
        }
    }

    Ok(ChunkLayout::Chunked(chunks))
}

/// Load a validated chunked layout: first chunk becomes the stack, remaining
/// chunks are decoded frame-by-frame and added element-wise.
///
/// Peak memory is one full stack plus one frame — VENUS stacks run to
/// several GB, so materialising every chunk simultaneously is not an option.
fn load_chunked_sum(chunks: &BTreeMap<u64, Vec<(u64, PathBuf)>>) -> Result<Array3<f64>, IoError> {
    let mut iter = chunks.values();
    let first = iter.next().expect("detect_chunk_layout yields >= 1 chunk");
    let first_paths: Vec<PathBuf> = first.iter().map(|(_, path)| path.clone()).collect();
    let mut acc = load_frames_from_paths(&first_paths)?;
    let (_, height, width) = acc.dim();

    for frames in iter {
        for (i, (_, path)) in frames.iter().enumerate() {
            let (pixels, w, h) = read_single_frame(path, i)?;
            if w as usize != width || h as usize != height {
                return Err(IoError::DimensionMismatch {
                    expected: (width as u32, height as u32),
                    got: (w, h),
                    frame: i,
                });
            }
            let mut slice = acc.slice_mut(s![i, .., ..]);
            for (dst, src) in slice.iter_mut().zip(pixels.iter()) {
                *dst += src;
            }
        }
    }

    Ok(acc)
}

/// Shared helper: load a sorted slice of single-frame TIFF paths into a 3D array.
///
/// Each file must contain exactly one frame.  Dimensions are checked for
/// consistency across all files and pixel counts are validated against the
/// reported image dimensions.
fn load_frames_from_paths(paths: &[std::path::PathBuf]) -> Result<Array3<f64>, IoError> {
    debug_assert!(
        !paths.is_empty(),
        "load_frames_from_paths called with empty paths"
    );
    let mut frames: Vec<Vec<f64>> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;

    for (i, path) in paths.iter().enumerate() {
        let (pixels, w, h) = read_single_frame(path, i)?;

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

        frames.push(pixels);
    }

    let n_frames = frames.len();
    let flat: Vec<f64> = frames.into_iter().flatten().collect();
    Array3::from_shape_vec((n_frames, height as usize, width as usize), flat)
        .map_err(|e| IoError::TiffDecode(format!("Shape error: {}", e)))
}

/// Decode one single-frame TIFF file to `(pixels, width, height)`.
///
/// Rejects files containing more than one frame — each file in a directory is
/// expected to contain exactly one frame; use [`load_tiff_stack`] for
/// multi-frame TIFFs.  The pixel count is validated against the reported
/// image dimensions.  `frame_label` is the frame's position in the stack
/// being assembled, used only for error messages.
fn read_single_frame(path: &Path, frame_label: usize) -> Result<(Vec<f64>, u32, u32), IoError> {
    let file = std::fs::File::open(path)
        .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
    let mut decoder = Decoder::new(file).map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

    let (w, h) = decoder
        .dimensions()
        .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

    let data = decoder
        .read_image()
        .map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

    // Reject multi-frame TIFFs in folder loading mode — each file
    // in a directory is expected to contain exactly one frame.
    // Use load_tiff_stack() for multi-frame TIFFs.
    if decoder.more_images() {
        return Err(IoError::InvalidParameter(format!(
            "File '{}' contains multiple frames; use load_tiff_stack() for multi-frame TIFFs",
            path.display()
        )));
    }

    let pixels = decode_to_f64(data)?;
    let expected_len = (w as usize) * (h as usize);
    if pixels.len() != expected_len {
        return Err(IoError::TiffDecode(format!(
            "Frame {} has {} pixels, expected {}",
            frame_label,
            pixels.len(),
            expected_len
        )));
    }
    Ok((pixels, w, h))
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
    fn test_glob_match_empty_strings() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "foo"));
        assert!(glob_match("*", ""));
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

    #[test]
    fn test_load_tiff_auto_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tiff");

        let frame1: Vec<u16> = vec![10, 20, 30, 40];
        let frame2: Vec<u16> = vec![50, 60, 70, 80];
        write_test_tiff(&path, &[frame1, frame2], 2, 2);

        let arr = load_tiff_auto(&path).unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 10.0);
        assert_eq!(arr[[1, 1, 1]], 80.0);
    }

    #[test]
    fn test_load_tiff_auto_directory() {
        let dir = tempfile::tempdir().unwrap();

        for i in 0..2u16 {
            let path = dir.path().join(format!("frame_{:04}.tif", i));
            let data: Vec<u16> = (0..4).map(|j| (i + 1) * 10 + j).collect();
            write_test_tiff(&path, &[data], 2, 2);
        }

        let arr = load_tiff_auto(dir.path()).unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 10.0);
    }

    #[test]
    fn test_load_tiff_auto_nonexistent() {
        let result = load_tiff_auto(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    /// Write a chunked-run test folder: `<prefix>_<chunk>_<frame>.tif` files
    /// with 2x2 pixels valued `base + frame*10 + pixel` so per-element sums
    /// are easy to assert.
    fn write_chunk_files(dir: &Path, prefix: &str, chunk: u64, base: u16, frames: &[u64]) {
        for &f in frames {
            let path = dir.join(format!("{}_{}_{:04}.tif", prefix, chunk, f));
            let data: Vec<u16> = (0..4).map(|j| base + (f as u16) * 10 + j).collect();
            write_test_tiff(&path, &[data], 2, 2);
        }
    }

    /// T1: two chunks with identical frame sequences sum element-wise.
    #[test]
    fn test_chunked_two_chunks_summed() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1, 2, 3]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1, 2, 3]);

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        assert_eq!(arr.shape(), &[4, 2, 2]);
        for f in 0..4usize {
            for j in 0..4usize {
                let expected = (100 + f * 10 + j) as f64 + (200 + f * 10 + j) as f64;
                assert_eq!(arr[[f, j / 2, j % 2]], expected, "frame {f} pixel {j}");
            }
        }
        assert_eq!(
            info,
            TiffLoadInfo {
                n_files: 8,
                n_chunks: 2,
                chunk_ids: vec![764, 765],
                chunks_summed: true,
            }
        );
    }

    /// T2: sum_chunks=false loads the legacy lexicographic concatenation.
    #[test]
    fn test_chunked_sum_opt_out() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1, 2, 3]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1, 2, 3]);

        let options = TiffFolderOptions { sum_chunks: false };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        assert_eq!(arr.shape(), &[8, 2, 2]);
        // Lexicographic: run_764_0000 .. run_764_0003, run_765_0000 ..
        assert_eq!(arr[[0, 0, 0]], 100.0);
        assert_eq!(arr[[4, 0, 0]], 200.0);
        assert_eq!(info.n_chunks, 2);
        assert_eq!(info.chunk_ids, vec![764, 765]);
        assert!(!info.chunks_summed);
    }

    /// T3: a single chunk loads identically to legacy (zero-padded names).
    #[test]
    fn test_chunked_single_chunk_matches_legacy() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1, 2]);

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        let legacy = {
            let mut paths: Vec<_> = std::fs::read_dir(dir.path())
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            paths.sort();
            load_frames_from_paths(&paths).unwrap()
        };
        assert_eq!(arr, legacy);
        assert_eq!(
            info,
            TiffLoadInfo {
                n_files: 3,
                n_chunks: 1,
                chunk_ids: vec![764],
                chunks_summed: false,
            }
        );
    }

    /// T4: non-chunked names (single `_<num>` field) use the legacy path.
    #[test]
    fn test_non_chunked_names_legacy() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..3u16 {
            let path = dir.path().join(format!("frame_{:04}.tif", i));
            write_test_tiff(&path, &[vec![i * 10, 1, 2, 3]], 2, 2);
        }

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        assert_eq!(arr.shape(), &[3, 2, 2]);
        assert_eq!(info.n_chunks, 0);
        assert!(info.chunk_ids.is_empty());
        assert!(!info.chunks_summed);
    }

    /// T5: ragged chunks (differing frame counts) are a hard error naming
    /// the per-chunk counts.
    #[test]
    fn test_chunked_ragged_counts_error() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1, 2]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1]);

        let err = load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default())
            .unwrap_err();
        assert!(
            matches!(err, IoError::ChunkMismatch { .. }),
            "Expected ChunkMismatch, got: {:?}",
            err,
        );
        let msg = format!("{}", err);
        assert!(msg.contains("3 frames"), "counts missing: {msg}");
        assert!(msg.contains("2 frames"), "counts missing: {msg}");
    }

    /// T6: equal counts but differing frame sets are a hard error naming the
    /// first differing frame.
    #[test]
    fn test_chunked_differing_frame_sets_error() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 2]);

        let err = load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default())
            .unwrap_err();
        assert!(
            matches!(err, IoError::ChunkMismatch { .. }),
            "Expected ChunkMismatch, got: {:?}",
            err,
        );
        let msg = format!("{}", err);
        assert!(
            msg.contains("1 vs 2"),
            "first differing frame missing: {msg}"
        );
    }

    /// T7: two distinct prefixes fall back to legacy (never sum across runs).
    #[test]
    fn test_chunked_mixed_prefixes_legacy() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run_a", 764, 100, &[0, 1]);
        write_chunk_files(dir.path(), "run_b", 764, 200, &[0, 1]);

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        assert_eq!(arr.shape(), &[4, 2, 2]);
        assert_eq!(info.n_chunks, 0);
        assert!(!info.chunks_summed);
    }

    /// T8: duplicate (chunk, frame) via `.tif` + `.tiff` of the same stem.
    #[test]
    fn test_chunked_duplicate_frame_error() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1]);
        let dup = dir.path().join("run_764_0001.tiff");
        write_test_tiff(&dup, &[vec![9, 9, 9, 9]], 2, 2);

        let err = load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default())
            .unwrap_err();
        assert!(
            matches!(err, IoError::ChunkMismatch { .. }),
            "Expected ChunkMismatch, got: {:?}",
            err,
        );
        let msg = format!("{}", err);
        assert!(msg.contains("duplicate frame 1"), "got: {msg}");
    }

    /// T9: unpadded frame numbers order numerically (`_2` before `_10`),
    /// unlike lexicographic order where `run_1_10` sorts before `run_1_2`.
    #[test]
    fn test_chunked_numeric_frame_order() {
        let dir = tempfile::tempdir().unwrap();
        write_test_tiff(&dir.path().join("run_1_2.tif"), &[vec![20, 0, 0, 0]], 2, 2);
        write_test_tiff(
            &dir.path().join("run_1_10.tif"),
            &[vec![100, 0, 0, 0]],
            2,
            2,
        );

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        // Numeric order: frame 2 first, then frame 10.
        assert_eq!(arr[[0, 0, 0]], 20.0);
        assert_eq!(arr[[1, 0, 0]], 100.0);
        assert_eq!(info.n_chunks, 1);
        assert_eq!(info.chunk_ids, vec![1]);
    }

    /// T10: dimension mismatch across chunks is surfaced as DimensionMismatch.
    #[test]
    fn test_chunked_cross_chunk_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1]);
        // Chunk 765 has the same frames but 3x2 images.
        for f in 0..2u64 {
            let path = dir.path().join(format!("run_765_{:04}.tif", f));
            write_test_tiff(&path, &[vec![1, 2, 3, 4, 5, 6]], 3, 2);
        }

        let err = load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default())
            .unwrap_err();
        assert!(
            matches!(err, IoError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {:?}",
            err,
        );
    }

    /// T11: three chunks sum element-wise.
    #[test]
    fn test_chunked_three_chunks() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 1, 100, &[0, 1]);
        write_chunk_files(dir.path(), "run", 2, 200, &[0, 1]);
        // Chunk ids need not be consecutive — a dropped middle chunk is fine.
        write_chunk_files(dir.path(), "run", 7, 400, &[0, 1]);

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        for f in 0..2usize {
            for j in 0..4usize {
                let expected = (700 + 3 * (f * 10 + j)) as f64;
                assert_eq!(arr[[f, j / 2, j % 2]], expected, "frame {f} pixel {j}");
            }
        }
        assert_eq!(info.n_chunks, 3);
        assert_eq!(info.chunk_ids, vec![1, 2, 7]);
        assert!(info.chunks_summed);
    }

    /// T12: a glob pattern that selects one chunk yields n_chunks == 1.
    #[test]
    fn test_chunked_pattern_selects_one_chunk() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1]);

        let (arr, info) = load_tiff_folder_with_options(
            dir.path(),
            Some("run_764_*"),
            &TiffFolderOptions::default(),
        )
        .unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 100.0);
        assert_eq!(info.n_chunks, 1);
        assert_eq!(info.chunk_ids, vec![764]);
        assert!(!info.chunks_summed);
    }

    /// T13: load_tiff_auto on a chunked directory sums by default.
    #[test]
    fn test_load_tiff_auto_chunked_directory() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1]);

        let arr = load_tiff_auto(dir.path()).unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 300.0);

        let (arr2, info) =
            load_tiff_auto_with_options(dir.path(), &TiffFolderOptions::default()).unwrap();
        assert_eq!(arr2, arr);
        assert!(info.chunks_summed);
        assert_eq!(info.n_chunks, 2);
    }

    /// Folder loading should reject files containing multiple frames.
    #[test]
    fn test_load_tiff_folder_rejects_multi_frame() {
        let dir = tempfile::tempdir().unwrap();

        // Write a multi-frame TIFF into the directory.
        let path = dir.path().join("multi.tiff");
        let frame1: Vec<u16> = vec![1, 2, 3, 4];
        let frame2: Vec<u16> = vec![5, 6, 7, 8];
        write_test_tiff(&path, &[frame1, frame2], 2, 2);

        let result = load_tiff_folder(dir.path(), None);
        assert!(
            result.is_err(),
            "Multi-frame TIFF in folder should be rejected"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("multiple frames"),
            "Error should mention multiple frames, got: {err}"
        );
    }
}
