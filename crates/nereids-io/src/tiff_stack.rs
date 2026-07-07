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
//!   lexicographic concatenation (the flag only affects folders with two or
//!   more chunks — single-chunk folders always load in numeric frame
//!   order);
//! - ragged chunks (differing frame counts or frame sets) or duplicate
//!   (chunk, frame) pairs → dispatched on the summing flag.  On the default
//!   summing path ([`TiffFolderOptions::sum_chunks`]` = true`) they are a
//!   hard [`IoError::ChunkMismatch`] error, never a silent stack or partial
//!   sum (summing ragged chunks would corrupt counts).  With summing opted
//!   out ([`sum_chunks`](TiffFolderOptions::sum_chunks)` = false`) there is
//!   nothing to corrupt, so the documented legacy lexicographic
//!   concatenation loads even for inconsistent chunks, with the irregularity
//!   surfaced through [`TiffLoadInfo::chunk_inconsistent`].
//!
//! Folders with two or more distinct prefixes fall back to legacy
//! lexicographic stacking — summing across different prefixes would merge
//! different runs.  Use the `pattern` argument to select one run.
//!
//! *Mixed* folders — where at least one stem parses as
//! `<prefix>_<chunk>_<frame>` but others do not (a stray overview TIFF, a
//! misnamed frame) — also fall back to legacy lexicographic stacking, but
//! the fallback is counted: [`TiffLoadInfo::n_unrecognized_files`] reports
//! how many files disabled chunk detection and
//! [`TiffLoadInfo::unrecognized_examples`] names up to
//! [`MAX_UNRECOGNIZED_EXAMPLES`] of them, so consumers (the GUI provenance
//! log, the Python `UserWarning`) can surface that a chunked-looking run
//! folder was *not* chunk-loaded.  Remove the stray files or use `pattern`
//! to exclude them.
//!
//! ### One acquisition per folder
//!
//! The chunk heuristic assumes the folder holds **one acquisition** — the
//! VENUS autoreduce layout, where each run gets its own directory (verified
//! on IPTS-37432 output; note the `<chunk>` field in real names is a
//! run-ish id, e.g. `..._ob_0_116_00000.tif`).  The heuristic cannot
//! distinguish same-prefix sibling *runs* co-located in one folder from DAQ
//! chunks of a single run: such siblings would be summed.  When a folder
//! may hold multiple runs, select one with `pattern` or pass
//! [`TiffFolderOptions::sum_chunks`]` = false`.
//!
//! ## Pixel-value policy
//!
//! Raw detector counts are non-negative by construction, so a negative or
//! non-finite pixel signals file corruption or a signed-type readout bug.
//! By default every loader rejects such values with
//! [`IoError::BadPixelValue`] ([`PixelValuePolicy::Reject`]).  Two escape
//! hatches exist:
//! - [`PixelValuePolicy::ClipToZero`] clamps negative values to `0.0`
//!   (counted in [`TiffLoadInfo::n_clipped_pixels`]); non-finite values
//!   still error, because clipping a NaN would invent data;
//! - [`PixelValuePolicy::Allow`] accepts all values verbatim — required for
//!   pre-normalized transmission stacks, where noise around zero can
//!   legitimately produce small negative values.
//!
//! For *corrupt readout pixels* in raw counts (e.g. a railed pixel stuck at
//! a signed sentinel), the right tool is a per-acquisition mask from
//! `nereids_io::normalization::detect_bad_pixels`, not a load-time clamp.
//!
//! ## Data types
//! - 16-bit unsigned integer (common for neutron detectors)
//! - 32-bit float (normalized data)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ndarray::{Array3, ArrayView2, s};
use tiff::decoder::Decoder;
use tiff::decoder::DecodingResult;

use crate::error::IoError;

/// Policy for negative or non-finite pixel values encountered during load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelValuePolicy {
    /// Reject the load with [`IoError::BadPixelValue`] (default).  Raw
    /// detector counts are non-negative by construction, so a negative or
    /// non-finite pixel signals corruption that must be surfaced, not
    /// silently imported.
    #[default]
    Reject,
    /// Clamp negative values to `0.0`, counting them in
    /// [`TiffLoadInfo::n_clipped_pixels`].  Non-finite values still error —
    /// clipping a NaN would invent data.
    ClipToZero,
    /// Accept all values verbatim.  Needed for pre-normalized transmission
    /// stacks, where noise around zero legitimately produces small negative
    /// values.
    Allow,
}

/// Options controlling how a TIFF folder (or file) is loaded.
#[derive(Debug, Clone, Copy)]
pub struct TiffFolderOptions {
    /// Sum DAQ chunks element-wise when a chunked folder is detected
    /// (default `true`).  When `false`, a *multi-chunk* folder is loaded
    /// as the legacy lexicographic concatenation of all files.  The flag
    /// only affects folders with two or more chunks: single-chunk (and
    /// non-chunk-patterned) folders load identically either way —
    /// chunk-patterned names in numeric frame order, others
    /// lexicographically.
    ///
    /// The flag also decides how *inconsistent* chunks (ragged frame
    /// counts/sets or duplicate (chunk, frame) pairs) are handled.  With
    /// summing on (the default), inconsistency is a hard
    /// [`IoError::ChunkMismatch`] error — summing ragged chunks would
    /// silently corrupt counts.  With summing off, the legacy lexicographic
    /// concatenation loads even for inconsistent chunks (there is nothing to
    /// corrupt), and the irregularity is reported via
    /// [`TiffLoadInfo::chunk_inconsistent`] rather than raised — inspecting
    /// raw frames of a ragged folder is exactly when `sum_chunks = false` is
    /// reached for.
    pub sum_chunks: bool,
    /// Policy for negative / non-finite pixel values (default
    /// [`PixelValuePolicy::Reject`]).
    pub pixel_policy: PixelValuePolicy,
}

impl Default for TiffFolderOptions {
    fn default() -> Self {
        Self {
            sum_chunks: true,
            pixel_policy: PixelValuePolicy::default(),
        }
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
    /// Number of negative pixels clamped to zero.  Only ever nonzero under
    /// [`PixelValuePolicy::ClipToZero`].
    pub n_clipped_pixels: usize,
    /// Number of files that did **not** parse as `<prefix>_<chunk>_<frame>`
    /// while at least one other file in the same folder did — a *mixed*
    /// folder, where the non-conforming files disabled chunk detection and
    /// forced the legacy lexicographic load.  `0` in every other path:
    /// single-file loads, fully chunk-patterned folders, folders where no
    /// file matches the convention (the normal `frame_0000.tif` world), and
    /// multi-prefix folders (every stem parses; a different, documented
    /// fallback).
    pub n_unrecognized_files: usize,
    /// Lexicographically-first filenames of the non-conforming files, capped
    /// at [`MAX_UNRECOGNIZED_EXAMPLES`] entries so the provenance stays
    /// message-sized.  Empty iff `n_unrecognized_files == 0`.
    pub unrecognized_examples: Vec<String>,
    /// Whether the folder's chunk-patterned files were internally
    /// inconsistent (ragged frame counts/sets or a duplicate (chunk, frame)
    /// pair) yet were still loaded, as the legacy lexicographic
    /// concatenation, because the caller opted out of summing
    /// ([`TiffFolderOptions::sum_chunks`]` = false`).  This is a distinct
    /// signal from [`n_unrecognized_files`](Self::n_unrecognized_files): the
    /// files *do* follow `<prefix>_<chunk>_<frame>`, they just do not agree
    /// on a common frame set.  Always `false` on the summing path — there the
    /// same inconsistency is a hard [`IoError::ChunkMismatch`] error, because
    /// summing ragged chunks would silently corrupt counts.
    pub chunk_inconsistent: bool,
}

/// Maximum number of offending filenames retained in
/// [`TiffLoadInfo::unrecognized_examples`] when a mixed folder disables
/// chunk detection (the count in [`TiffLoadInfo::n_unrecognized_files`] is
/// never capped).
pub const MAX_UNRECOGNIZED_EXAMPLES: usize = 3;

/// Apply the pixel-value policy to one decoded frame, in place.
///
/// Returns the number of pixels clamped to zero (only ever nonzero under
/// [`PixelValuePolicy::ClipToZero`]).  `frame` is the frame's position in
/// the stack being assembled, used only for error reporting.
fn enforce_pixel_policy(
    pixels: &mut [f64],
    policy: PixelValuePolicy,
    file: &Path,
    frame: usize,
) -> Result<usize, IoError> {
    match policy {
        PixelValuePolicy::Allow => Ok(0),
        PixelValuePolicy::Reject => {
            nereids_core::validation::first_non_finite_or_negative(pixels.iter().copied())
                .map_err(|(index, value)| IoError::BadPixelValue {
                    file: file.to_string_lossy().into_owned(),
                    frame,
                    index,
                    value,
                })?;
            Ok(0)
        }
        PixelValuePolicy::ClipToZero => {
            let mut clipped = 0usize;
            for (index, v) in pixels.iter_mut().enumerate() {
                // NaN bypasses `<`, so the finiteness check must come first
                // and cannot be folded into the comparison below.
                if !v.is_finite() {
                    return Err(IoError::BadPixelValue {
                        file: file.to_string_lossy().into_owned(),
                        frame,
                        index,
                        value: *v,
                    });
                }
                if *v < 0.0 {
                    *v = 0.0;
                    clipped += 1;
                }
            }
            Ok(clipped)
        }
    }
}

/// Load a multi-frame TIFF into a 3D array (n_frames, height, width).
///
/// Each TIFF frame becomes one slice along the first axis.
/// Data is converted to `f64` regardless of the source pixel type.
/// Negative or non-finite pixels are rejected (the default
/// [`PixelValuePolicy::Reject`]); use [`load_tiff_stack_with_options`] to
/// choose a different policy.
///
/// # Arguments
/// * `path` — Path to the multi-frame TIFF file.
///
/// # Returns
/// 3D array with shape (n_frames, height, width) and f64 values.
pub fn load_tiff_stack(path: &Path) -> Result<Array3<f64>, IoError> {
    load_tiff_stack_with_options(path, PixelValuePolicy::default()).map(|(arr, _)| arr)
}

/// Load a multi-frame TIFF with an explicit pixel-value policy, returning
/// provenance metadata.
///
/// Behaves like [`load_tiff_stack`], with the pixel-value policy applied to
/// every frame as it is decoded (see the [module docs](self)).
///
/// # Arguments
/// * `path`         — Path to the multi-frame TIFF file.
/// * `pixel_policy` — Policy for negative / non-finite pixel values.
///
/// # Returns
/// `(stack, info)` where `stack` has shape (n_frames, height, width).
pub fn load_tiff_stack_with_options(
    path: &Path,
    pixel_policy: PixelValuePolicy,
) -> Result<(Array3<f64>, TiffLoadInfo), IoError> {
    let file = std::fs::File::open(path)
        .map_err(|e| IoError::FileNotFound(path.to_string_lossy().into_owned(), e))?;
    let mut decoder = Decoder::new(file).map_err(|e| IoError::TiffDecode(format!("{}", e)))?;

    let mut frames: Vec<Vec<f64>> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut n_clipped_pixels = 0usize;

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

        let mut pixels = decode_to_f64(data)?;
        n_clipped_pixels += enforce_pixel_policy(&mut pixels, pixel_policy, path, frames.len())?;
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
    let arr = Array3::from_shape_vec((n_frames, height as usize, width as usize), flat)
        .map_err(|e| IoError::TiffDecode(format!("Shape error: {}", e)))?;
    Ok((
        arr,
        TiffLoadInfo {
            n_files: 1,
            n_chunks: 0,
            chunk_ids: Vec::new(),
            chunks_summed: false,
            n_clipped_pixels,
            n_unrecognized_files: 0,
            unrecognized_examples: Vec::new(),
            chunk_inconsistent: false,
        },
    ))
}

/// Load TIFF data from either a single multi-frame file or a directory.
///
/// Auto-detects based on whether `path` is a file or directory:
/// - File → [`load_tiff_stack`] (multi-frame TIFF)
/// - Directory → [`load_tiff_directory`] (one file per frame)
///
/// Directories are **not** loaded purely lexicographically: chunked VENUS
/// folders (`<prefix>_<chunk>_<frame>.tif`) are detected automatically,
/// ordered by numeric frame index, and chunks are summed element-wise
/// (the [`TiffFolderOptions`] defaults — see the [module docs](self)).
/// No provenance is returned; use [`load_tiff_auto_with_options`] to get
/// a [`TiffLoadInfo`] and to control chunk summing.
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
                load_tiff_stack_with_options(path, options.pixel_policy)
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
/// Delegates to [`load_tiff_folder`] with default options: chunked VENUS
/// folders (`<prefix>_<chunk>_<frame>.tif`) are detected automatically,
/// ordered by numeric frame index, and chunks covering identical frame
/// ranges are **summed element-wise**.  Only folders *not* following the
/// chunk convention load in lexicographic filename order — name legacy
/// files with zero-padded indices (e.g., `frame_0001.tiff`,
/// `frame_0002.tiff`, ...).  No provenance is returned; use
/// [`load_tiff_folder_with_options`] to get a [`TiffLoadInfo`] and to
/// control chunk summing.
///
/// # Arguments
/// * `dir` — Path to the directory containing TIFF files.
///
/// # Returns
/// 3D array with shape (n_frames, height, width) and f64 values.
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
/// Applies the default [`TiffFolderOptions`]: chunked VENUS folders
/// (`<prefix>_<chunk>_<frame>.tif`) are detected automatically, ordered by
/// numeric frame index, and chunks covering identical frame ranges are
/// **summed element-wise**.  Only folders *not* following the chunk
/// convention load in lexicographic filename order — name legacy files
/// with zero-padded indices (e.g., `frame_0001.tif`, `frame_0002.tif`,
/// ...).  No provenance is returned; use
/// [`load_tiff_folder_with_options`] to get a [`TiffLoadInfo`] and to
/// control chunk summing.
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
/// * [`IoError::FileNotFound`] if `dir` does not exist.
/// * [`IoError::NotADirectory`] if `dir` exists but is not a directory.
/// * [`IoError::NoMatchingFiles`] if no files match the pattern.
/// * [`IoError::DimensionMismatch`] if frames have inconsistent dimensions.
/// * [`IoError::ChunkMismatch`] if a chunked folder is internally
///   inconsistent *and* chunk summing is enabled (the default); with
///   `sum_chunks = false` the inconsistency is reported via
///   [`TiffLoadInfo::chunk_inconsistent`] instead of raised.
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
/// * [`IoError::FileNotFound`] if `dir` does not exist.
/// * [`IoError::NotADirectory`] if `dir` exists but is not a directory.
/// * [`IoError::NoMatchingFiles`] if no files match the pattern.
/// * [`IoError::DimensionMismatch`] if frames have inconsistent dimensions.
/// * [`IoError::ChunkMismatch`] if a chunked folder is internally
///   inconsistent (ragged chunks or duplicate (chunk, frame) pairs) *and*
///   `options.sum_chunks` is `true`.  With `sum_chunks = false` the same
///   inconsistency is not raised: the files load as the legacy lexicographic
///   concatenation and [`TiffLoadInfo::chunk_inconsistent`] is set.
pub fn load_tiff_folder_with_options(
    dir: &Path,
    pattern: Option<&str>,
    options: &TiffFolderOptions,
) -> Result<(Array3<f64>, TiffLoadInfo), IoError> {
    // Distinguish "does not exist" from "exists but is not a directory":
    // the Python binding maps `FileNotFound` *whose source kind is
    // `NotFound`* to `FileNotFoundError` and `NotADirectory` to
    // `NotADirectoryError`, and its docstring promises exactly that split.
    //
    // A single `metadata` probe (mirroring `load_tiff_auto_with_options`) is
    // the honest test: `Path::exists()`/`is_dir()` collapse *every* metadata
    // failure to `false`, so a permission-denied parent (EACCES) would be
    // mislabeled `FileNotFound(NotFound)` → Python `FileNotFoundError` — the
    // exact confusion `is_genuine_not_found` exists to prevent.  Wrapping the
    // real `io::Error` preserves its true kind, so only a genuine `NotFound`
    // reaches `FileNotFoundError` while EACCES falls through to `OSError`.
    match std::fs::metadata(dir) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => return Err(IoError::NotADirectory(dir.to_string_lossy().into_owned())),
        Err(e) => return Err(IoError::FileNotFound(dir.to_string_lossy().into_owned(), e)),
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
    let mut n_clipped_pixels = 0usize;

    match detect_chunk_layout(dir, &paths, options.sum_chunks)? {
        ChunkLayout::Legacy {
            n_unrecognized_files,
            unrecognized_examples,
        } => {
            paths.sort();
            let arr = load_frames_from_paths(&paths, options.pixel_policy, &mut n_clipped_pixels)?;
            Ok((
                arr,
                TiffLoadInfo {
                    n_files,
                    n_chunks: 0,
                    chunk_ids: Vec::new(),
                    chunks_summed: false,
                    n_clipped_pixels,
                    n_unrecognized_files,
                    unrecognized_examples,
                    chunk_inconsistent: false,
                },
            ))
        }
        ChunkLayout::InconsistentChunks { chunk_ids } => {
            // Only reachable with `sum_chunks == false` (detect_chunk_layout
            // yields this variant *instead of* a hard `ChunkMismatch` exactly
            // when summing was opted out).  There is nothing to corrupt when
            // we are not summing, so honor the documented `sum_chunks=false`
            // contract: load every matching file as the legacy lexicographic
            // concatenation (frame count = sum of all files) and surface the
            // irregularity through `chunk_inconsistent` — the chunk ids are
            // still reported so consumers can name what was inconsistent.
            let n_chunks = chunk_ids.len();
            paths.sort();
            let arr = load_frames_from_paths(&paths, options.pixel_policy, &mut n_clipped_pixels)?;
            Ok((
                arr,
                TiffLoadInfo {
                    n_files,
                    n_chunks,
                    chunk_ids,
                    chunks_summed: false,
                    n_clipped_pixels,
                    n_unrecognized_files: 0,
                    unrecognized_examples: Vec::new(),
                    chunk_inconsistent: true,
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
                let arr =
                    load_chunked_sum(dir, &chunks, options.pixel_policy, &mut n_clipped_pixels)?;
                Ok((
                    arr,
                    TiffLoadInfo {
                        n_files,
                        n_chunks,
                        chunk_ids,
                        chunks_summed: n_chunks > 1,
                        n_clipped_pixels,
                        n_unrecognized_files: 0,
                        unrecognized_examples: Vec::new(),
                        chunk_inconsistent: false,
                    },
                ))
            } else {
                // Chunk summing opted out (only reachable with >= 2 chunks):
                // legacy lexicographic concatenation of every matching file
                // (chunk structure is still reported in the info).
                paths.sort();
                let arr =
                    load_frames_from_paths(&paths, options.pixel_policy, &mut n_clipped_pixels)?;
                Ok((
                    arr,
                    TiffLoadInfo {
                        n_files,
                        n_chunks,
                        chunk_ids,
                        chunks_summed: false,
                        n_clipped_pixels,
                        n_unrecognized_files: 0,
                        unrecognized_examples: Vec::new(),
                        chunk_inconsistent: false,
                    },
                ))
            }
        }
    }
}

/// Detected layout of a TIFF folder's (filtered) file list.
enum ChunkLayout {
    /// Not a chunked folder — load in lexicographic filename order.
    /// The fields are nonzero/non-empty only for *mixed* folders (some
    /// stems parsed as `<prefix>_<chunk>_<frame>` but others did not),
    /// where the non-conforming files disabled chunk detection; see
    /// [`TiffLoadInfo::n_unrecognized_files`].
    Legacy {
        n_unrecognized_files: usize,
        unrecognized_examples: Vec<String>,
    },
    /// Chunk-conforming filenames that are internally inconsistent — ragged
    /// frame counts/sets or a duplicate (chunk, frame) pair.  Produced *only*
    /// when `sum_chunks` is `false`: with summing requested the identical
    /// inconsistency is a hard [`IoError::ChunkMismatch`] (summing ragged
    /// chunks would silently corrupt counts).  The caller loads these files
    /// as the legacy lexicographic concatenation and records the irregularity
    /// in [`TiffLoadInfo::chunk_inconsistent`]; the detected chunk ids are
    /// carried so the provenance can still name them.
    InconsistentChunks { chunk_ids: Vec<u64> },
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
/// `<prefix>_<chunk>_<frame>` (including non-UTF-8 stems), or two or more
/// distinct prefixes (summing across prefixes would merge different runs;
/// use `pattern` to select one).  *Mixed* folders — at least one stem
/// parsed but others did not — still fall back (a hand-assembled folder is
/// legitimate) but are counted in the returned
/// [`ChunkLayout::Legacy`] fields so every consumer can surface that a
/// stray file disabled chunk detection; all-non-conforming folders (the
/// normal `frame_0000.tif` world) report a count of 0.
///
/// Internally *inconsistent* chunks — duplicate (chunk, frame) pairs (e.g.
/// the same stem with both `.tif` and `.tiff` extensions, or `_764_`
/// alongside `_0764_`) or ragged chunks (differing frame counts or frame
/// sets) — are dispatched on `sum_chunks`:
/// - `sum_chunks == true` (the default summing path): a hard
///   [`IoError::ChunkMismatch`] error, because summing ragged chunks would
///   silently corrupt counts in the missing frames.  This guard is airtight
///   — the only path that ever *sums* rejects inconsistency before a single
///   frame is added.
/// - `sum_chunks == false` (the opt-out): [`ChunkLayout::InconsistentChunks`]
///   instead of an error.  Nothing is summed, so nothing can be corrupted;
///   the caller loads the documented legacy lexicographic concatenation and
///   flags [`TiffLoadInfo::chunk_inconsistent`].
///
/// Chunk ids need *not* be consecutive — a dropped middle chunk is still the
/// same run.
fn detect_chunk_layout(
    dir: &Path,
    paths: &[PathBuf],
    sum_chunks: bool,
) -> Result<ChunkLayout, IoError> {
    let mut parsed: Vec<(&str, u64, u64, &PathBuf)> = Vec::with_capacity(paths.len());
    let mut unrecognized: Vec<String> = Vec::new();
    for path in paths {
        // Non-UTF-8 stems cannot follow the ASCII naming convention, so
        // they count as non-conforming like any other unparseable stem
        // (displayed lossily in the examples).
        match path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(parse_chunked_stem)
        {
            Some((prefix, chunk, frame)) => parsed.push((prefix, chunk, frame, path)),
            None => unrecognized.push(
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned()),
            ),
        }
    }

    if parsed.is_empty() || !unrecognized.is_empty() {
        // Mixed folders (some stems parsed, some did not) must be loud:
        // without the count, ONE stray overview TIFF silently reinstates
        // the doubled-stack load — n_chunks reports 0 and neither the
        // Python warning nor the GUI provenance ever mentions chunks.
        let (n_unrecognized_files, unrecognized_examples) = if parsed.is_empty() {
            (0, Vec::new())
        } else {
            let n = unrecognized.len();
            // Sort for deterministic examples (read_dir order is arbitrary).
            unrecognized.sort();
            unrecognized.truncate(MAX_UNRECOGNIZED_EXAMPLES);
            (n, unrecognized)
        };
        return Ok(ChunkLayout::Legacy {
            n_unrecognized_files,
            unrecognized_examples,
        });
    }

    let first_prefix = parsed[0].0;
    if parsed.iter().any(|(prefix, ..)| *prefix != first_prefix) {
        // Every stem parsed but prefixes differ — a multi-run folder, the
        // documented legacy fallback, not a stray-file situation: count 0.
        return Ok(ChunkLayout::Legacy {
            n_unrecognized_files: 0,
            unrecognized_examples: Vec::new(),
        });
    }

    let mut chunks: BTreeMap<u64, Vec<(u64, PathBuf)>> = BTreeMap::new();
    for (_, chunk, frame, path) in parsed {
        chunks.entry(chunk).or_default().push((frame, path.clone()));
    }

    // One source of truth for "what makes chunks inconsistent"; the caller
    // decides whether that inconsistency is fatal (summing) or a soft
    // fall-back (opt-out).  Keeping the check in one place stops the two
    // paths from ever drifting apart.
    match validate_chunk_consistency(&mut chunks) {
        Ok(()) => Ok(ChunkLayout::Chunked(chunks)),
        Err(details) if sum_chunks => Err(IoError::ChunkMismatch {
            directory: dir.to_string_lossy().into_owned(),
            details,
        }),
        Err(_) => Ok(ChunkLayout::InconsistentChunks {
            chunk_ids: chunks.keys().copied().collect(),
        }),
    }
}

/// Validate a parsed chunk map for internal consistency, sorting each
/// chunk's frames by numeric index in place (needed both here and by
/// [`load_chunked_sum`]).
///
/// Returns `Ok(())` when every chunk covers the identical frame-index
/// sequence with no duplicate (chunk, frame) pair, or `Err(details)`
/// describing the first inconsistency found (a duplicate frame within a
/// chunk, differing per-chunk frame counts, or differing frame sets).  The
/// caller maps `details` onto either a hard [`IoError::ChunkMismatch`] (the
/// summing path — summing ragged chunks would silently corrupt counts) or a
/// [`ChunkLayout::InconsistentChunks`] soft fall-back (`sum_chunks = false`,
/// nothing to corrupt).
fn validate_chunk_consistency(
    chunks: &mut BTreeMap<u64, Vec<(u64, PathBuf)>>,
) -> Result<(), String> {
    // Sort each chunk's frames by numeric index and reject duplicates.
    for (chunk_id, frames) in chunks.iter_mut() {
        frames.sort_by_key(|(frame, _)| *frame);
        if let Some(pair) = frames.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(format!(
                "duplicate frame {} in chunk {}: '{}' and '{}'",
                pair[0].0,
                chunk_id,
                pair[0].1.display(),
                pair[1].1.display(),
            ));
        }
    }

    // Every chunk must cover the identical frame-index sequence.
    let mut iter = chunks.iter();
    let (first_id, first_frames) = iter.next().expect("chunks is non-empty");
    for (chunk_id, frames) in iter {
        if frames.len() != first_frames.len() {
            return Err(format!(
                "chunk {} has {} frames but chunk {} has {} frames",
                first_id,
                first_frames.len(),
                chunk_id,
                frames.len(),
            ));
        }
        if let Some((a, b)) = first_frames
            .iter()
            .zip(frames.iter())
            .find(|(a, b)| a.0 != b.0)
        {
            return Err(format!(
                "chunks {} and {} cover different frame indices: first difference {} vs {}",
                first_id, chunk_id, a.0, b.0,
            ));
        }
    }

    Ok(())
}

/// Load a validated chunked layout: first chunk becomes the stack, remaining
/// chunks are decoded frame-by-frame and added element-wise.
///
/// Peak memory is one full stack plus one frame's decode buffers: the first
/// chunk is decoded straight into its preallocated stack (see
/// [`load_frames_from_paths`]) and every later chunk is added one frame at
/// a time.  VENUS stacks run to several GB, so materialising every chunk
/// (or a transient second copy of one chunk's stack) is not an option.
fn load_chunked_sum(
    dir: &Path,
    chunks: &BTreeMap<u64, Vec<(u64, PathBuf)>>,
    pixel_policy: PixelValuePolicy,
    n_clipped_pixels: &mut usize,
) -> Result<Array3<f64>, IoError> {
    let mut iter = chunks.iter();
    let (&first_id, first) = iter.next().expect("detect_chunk_layout yields >= 1 chunk");
    let first_paths: Vec<PathBuf> = first.iter().map(|(_, path)| path.clone()).collect();
    let mut acc = load_frames_from_paths(&first_paths, pixel_policy, n_clipped_pixels)?;
    let (_, height, width) = acc.dim();

    // Duplicate-chunk guard (issue #653).  On real VENUS data every observed
    // multi-chunk folder is a byte-identical *duplicate write* of one
    // exposure, not sequential DAQ segments — summing them (the default)
    // silently doubles every count.  We fingerprint each chunk with an
    // FNV-1a hash over the raw f64 bits (O(1) extra memory — the stacks run
    // to several GB, so a second copy for an equality check is not an
    // option) and refuse to sum a chunk that is identical to ANY earlier
    // chunk (not just the first — otherwise a duplicate pair that excludes
    // the first chunk, e.g. [A, B, B], would still be double-summed).
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let fnv_step = |h: u64, bits: u64| (h ^ bits).wrapping_mul(FNV_PRIME);

    // Single-chunk folders (the dominant real case) never enter the loop, so
    // skip the full hash pass over the multi-GB first chunk entirely.
    let multi_chunk = chunks.len() > 1;
    let mut seen_hashes: Vec<(u64, u64)> = Vec::new(); // (chunk_id, hash)
    if multi_chunk {
        let first_hash = acc
            .iter()
            .fold(FNV_OFFSET, |h, &v| fnv_step(h, v.to_bits()));
        seen_hashes.push((first_id, first_hash));
    }

    for (&chunk_id, frames) in iter {
        let mut chunk_hash = FNV_OFFSET;
        for (i, (_, path)) in frames.iter().enumerate() {
            let (pixels, w, h) = read_single_frame(path, i, pixel_policy, n_clipped_pixels)?;
            if w as usize != width || h as usize != height {
                return Err(IoError::DimensionMismatch {
                    expected: (width as u32, height as u32),
                    got: (w, h),
                    frame: i,
                });
            }
            // Fold in the SAME element order as `acc.iter()` (row-major
            // frame,y,x) so identical content yields identical hashes.
            for &p in &pixels {
                chunk_hash = fnv_step(chunk_hash, p.to_bits());
            }
            let mut slice = acc.slice_mut(s![i, .., ..]);
            for (dst, src) in slice.iter_mut().zip(pixels.iter()) {
                *dst += src;
            }
        }
        if let Some(&(dup_of, _)) = seen_hashes.iter().find(|&&(_, h)| h == chunk_hash) {
            return Err(IoError::ChunkMismatch {
                directory: dir.to_string_lossy().into_owned(),
                details: format!(
                    "DAQ chunk {chunk_id} is byte-identical to chunk {dup_of} — a \
                     duplicate write of the same exposure, not a distinct DAQ segment. \
                     Summing them (default sum_chunks=true) would double every count \
                     (and inflate proton-charge normalisation by the chunk multiplicity). \
                     Pass sum_chunks=false to load all frames without summing (issue #653)."
                ),
            });
        }
        seen_hashes.push((chunk_id, chunk_hash));
    }

    Ok(acc)
}

/// Shared helper: load a sorted slice of single-frame TIFF paths into a 3D array.
///
/// Each file must contain exactly one frame.  Dimensions are checked for
/// consistency across all files and pixel counts are validated against the
/// reported image dimensions.  The stack is preallocated once the first
/// frame reveals the dimensions and every frame is copied straight into
/// its slice, so peak memory is the full stack plus one frame's decode
/// buffers — never a transient second copy of the stack.
fn load_frames_from_paths(
    paths: &[std::path::PathBuf],
    pixel_policy: PixelValuePolicy,
    n_clipped_pixels: &mut usize,
) -> Result<Array3<f64>, IoError> {
    debug_assert!(
        !paths.is_empty(),
        "load_frames_from_paths called with empty paths"
    );
    let mut width = 0u32;
    let mut height = 0u32;
    // Placeholder until the first frame reveals the dimensions (returned
    // as-is only in the release-mode empty-input case the debug_assert
    // above rules out in tests).
    let mut arr = Array3::<f64>::zeros((0, 0, 0));

    for (i, path) in paths.iter().enumerate() {
        let (pixels, w, h) = read_single_frame(path, i, pixel_policy, n_clipped_pixels)?;

        if i == 0 {
            width = w;
            height = h;
            arr = Array3::zeros((paths.len(), h as usize, w as usize));
        } else if w != width || h != height {
            return Err(IoError::DimensionMismatch {
                expected: (width, height),
                got: (w, h),
                frame: i,
            });
        }

        // read_single_frame validated pixels.len() == w × h, so this shape
        // check cannot fail in practice; map it anyway rather than unwrap.
        let view = ArrayView2::from_shape((h as usize, w as usize), &pixels)
            .map_err(|e| IoError::TiffDecode(format!("Shape error: {}", e)))?;
        arr.slice_mut(s![i, .., ..]).assign(&view);
    }

    Ok(arr)
}

/// Decode one single-frame TIFF file to `(pixels, width, height)`.
///
/// Rejects files containing more than one frame — each file in a directory is
/// expected to contain exactly one frame; use [`load_tiff_stack`] for
/// multi-frame TIFFs.  The pixel count is validated against the reported
/// image dimensions and the pixel-value policy is enforced (clipped pixels
/// accumulate into `n_clipped_pixels`).  `frame_label` is the frame's
/// position in the stack being assembled, used only for error messages.
fn read_single_frame(
    path: &Path,
    frame_label: usize,
    pixel_policy: PixelValuePolicy,
    n_clipped_pixels: &mut usize,
) -> Result<(Vec<f64>, u32, u32), IoError> {
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

    let mut pixels = decode_to_f64(data)?;
    *n_clipped_pixels += enforce_pixel_policy(&mut pixels, pixel_policy, path, frame_label)?;
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

    /// Create a signed-16-bit TIFF (native GrayI16 encoding) for pixel-value
    /// policy tests — a railed/corrupt readout pixel shows up as a negative
    /// signed sentinel such as -32554.
    fn write_test_tiff_i16(path: &Path, frames: &[Vec<i16>], width: u32, height: u32) {
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = TiffEncoder::new(file).unwrap();
        for frame in frames {
            encoder
                .write_image::<tiff::encoder::colortype::GrayI16>(width, height, frame)
                .unwrap();
        }
    }

    /// Create a 32-bit float TIFF (native Gray32Float encoding) so NaN and
    /// negative float pixels can be synthesized directly.
    fn write_test_tiff_f32(path: &Path, frames: &[Vec<f32>], width: u32, height: u32) {
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = TiffEncoder::new(file).unwrap();
        for frame in frames {
            encoder
                .write_image::<tiff::encoder::colortype::Gray32Float>(width, height, frame)
                .unwrap();
        }
    }

    /// The corrupt-readout sentinel used across the pixel-policy tests.
    const BAD_I16: i16 = -32554;

    /// T15: a negative signed pixel is rejected by default with a message
    /// naming the file, frame, index, value, and the detect_bad_pixels()
    /// escape hatch.
    #[test]
    fn test_pixel_policy_reject_negative_i16() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame_0000.tif");
        write_test_tiff_i16(&path, &[vec![10, BAD_I16, 30, 40]], 2, 2);

        let err = load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default())
            .unwrap_err();
        assert!(
            matches!(err, IoError::BadPixelValue { .. }),
            "Expected BadPixelValue, got: {:?}",
            err,
        );
        let msg = format!("{}", err);
        assert!(msg.contains("frame_0000.tif"), "file missing: {msg}");
        assert!(msg.contains("frame 0"), "frame missing: {msg}");
        assert!(msg.contains("index 1"), "index missing: {msg}");
        assert!(msg.contains("-32554"), "value missing: {msg}");
        assert!(
            msg.contains("detect_bad_pixels"),
            "detect_bad_pixels hint missing: {msg}"
        );
    }

    /// T16: ClipToZero clamps the negative pixel to 0.0 and counts it.
    #[test]
    fn test_pixel_policy_clip_to_zero() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame_0000.tif");
        write_test_tiff_i16(&path, &[vec![10, BAD_I16, 30, 40]], 2, 2);

        let options = TiffFolderOptions {
            pixel_policy: PixelValuePolicy::ClipToZero,
            ..Default::default()
        };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        assert_eq!(arr[[0, 0, 0]], 10.0);
        assert_eq!(arr[[0, 0, 1]], 0.0);
        assert_eq!(info.n_clipped_pixels, 1);
    }

    /// T17: Allow passes the negative value through verbatim.
    #[test]
    fn test_pixel_policy_allow_negative() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame_0000.tif");
        write_test_tiff_i16(&path, &[vec![10, BAD_I16, 30, 40]], 2, 2);

        let options = TiffFolderOptions {
            pixel_policy: PixelValuePolicy::Allow,
            ..Default::default()
        };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        assert_eq!(arr[[0, 0, 1]], f64::from(BAD_I16));
        assert_eq!(info.n_clipped_pixels, 0);
    }

    /// T18: a NaN float pixel is rejected by default.
    #[test]
    fn test_pixel_policy_reject_nan_f32() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame_0000.tif");
        write_test_tiff_f32(&path, &[vec![1.0, f32::NAN, 3.0, 4.0]], 2, 2);

        let err = load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default())
            .unwrap_err();
        assert!(
            matches!(err, IoError::BadPixelValue { .. }),
            "Expected BadPixelValue, got: {:?}",
            err,
        );
    }

    /// T19: ClipToZero still errors on NaN — clipping NaN would invent data.
    #[test]
    fn test_pixel_policy_clip_still_rejects_nan() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame_0000.tif");
        write_test_tiff_f32(&path, &[vec![1.0, f32::NAN, 3.0, 4.0]], 2, 2);

        let options = TiffFolderOptions {
            pixel_policy: PixelValuePolicy::ClipToZero,
            ..Default::default()
        };
        let err = load_tiff_folder_with_options(dir.path(), None, &options).unwrap_err();
        assert!(
            matches!(err, IoError::BadPixelValue { .. }),
            "Expected BadPixelValue, got: {:?}",
            err,
        );
    }

    /// T20: Allow passes NaN and negative floats through verbatim.
    #[test]
    fn test_pixel_policy_allow_nan_and_negative() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame_0000.tif");
        write_test_tiff_f32(&path, &[vec![1.0, f32::NAN, -5.0, 4.0]], 2, 2);

        let options = TiffFolderOptions {
            pixel_policy: PixelValuePolicy::Allow,
            ..Default::default()
        };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        assert!(arr[[0, 0, 1]].is_nan());
        assert_eq!(arr[[0, 1, 0]], -5.0);
        assert_eq!(info.n_clipped_pixels, 0);
    }

    /// T21: multi-frame load_tiff_stack rejects negatives by default;
    /// load_tiff_stack_with_options can clip instead.
    #[test]
    fn test_pixel_policy_multi_frame_stack() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tiff");
        let frame1: Vec<i16> = vec![1, 2, 3, 4];
        let frame2: Vec<i16> = vec![5, BAD_I16, 7, 8];
        write_test_tiff_i16(&path, &[frame1, frame2], 2, 2);

        let err = load_tiff_stack(&path).unwrap_err();
        assert!(
            matches!(err, IoError::BadPixelValue { frame: 1, .. }),
            "Expected BadPixelValue at frame 1, got: {:?}",
            err,
        );

        let (arr, info) =
            load_tiff_stack_with_options(&path, PixelValuePolicy::ClipToZero).unwrap();
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert_eq!(arr[[1, 0, 1]], 0.0);
        assert_eq!(info.n_clipped_pixels, 1);
    }

    /// T22: clipped-pixel counts accumulate across summed chunks.
    #[test]
    fn test_pixel_policy_clip_counts_accumulate_across_chunks() {
        let dir = tempfile::tempdir().unwrap();
        // Chunk 1: one negative pixel in frame 0.
        write_test_tiff_i16(
            &dir.path().join("run_1_0000.tif"),
            &[vec![10, -1, 30, 40]],
            2,
            2,
        );
        write_test_tiff_i16(
            &dir.path().join("run_1_0001.tif"),
            &[vec![11, 21, 31, 41]],
            2,
            2,
        );
        // Chunk 2: two negative pixels in frame 1.
        write_test_tiff_i16(
            &dir.path().join("run_2_0000.tif"),
            &[vec![100, 200, 300, 400]],
            2,
            2,
        );
        write_test_tiff_i16(
            &dir.path().join("run_2_0001.tif"),
            &[vec![-2, 201, -3, 401]],
            2,
            2,
        );

        let options = TiffFolderOptions {
            pixel_policy: PixelValuePolicy::ClipToZero,
            ..Default::default()
        };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        assert_eq!(info.n_clipped_pixels, 3);
        assert!(info.chunks_summed);
        // Clipping applies per frame before summing: frame 0 pixel 1 is
        // 0 + 200, frame 1 pixel 0 is 11 + 0.
        assert_eq!(arr[[0, 0, 1]], 200.0);
        assert_eq!(arr[[1, 0, 0]], 11.0);
        assert_eq!(arr[[1, 1, 0]], 31.0);
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
                n_clipped_pixels: 0,
                n_unrecognized_files: 0,
                unrecognized_examples: vec![],
                chunk_inconsistent: false,
            }
        );
    }

    /// Issue #653: two chunks with byte-identical content are a duplicate DAQ
    /// write, not sequential segments — the default summing path must refuse
    /// them (silently doubling every count is the exact real-VENUS failure),
    /// while `sum_chunks=false` still loads every frame for inspection.
    #[test]
    fn test_chunked_duplicate_write_rejected_on_sum_path() {
        let dir = tempfile::tempdir().unwrap();
        // Same base ⇒ chunk 765 is byte-identical to chunk 764.
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1, 2, 3]);
        write_chunk_files(dir.path(), "run", 765, 100, &[0, 1, 2, 3]);

        let err = load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default())
            .expect_err("summing byte-identical duplicate chunks must be a hard error");
        match err {
            IoError::ChunkMismatch { details, .. } => {
                assert!(
                    details.contains("identical") && details.contains("765"),
                    "unexpected message: {details}"
                );
                assert!(
                    details.contains("sum_chunks=false"),
                    "must name the escape hatch"
                );
            }
            other => panic!("expected ChunkMismatch, got {other:?}"),
        }

        // The escape hatch loads all 8 frames (concatenation), no summing.
        let options = TiffFolderOptions {
            sum_chunks: false,
            ..TiffFolderOptions::default()
        };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        assert_eq!(arr.shape(), &[8, 2, 2]);
        assert!(!info.chunks_summed);
        // Distinct chunks (the T1 case) must still sum — the guard is
        // content-based, not a blanket multi-chunk refusal.
        let dir2 = tempfile::tempdir().unwrap();
        write_chunk_files(dir2.path(), "run", 764, 100, &[0, 1, 2, 3]);
        write_chunk_files(dir2.path(), "run", 765, 200, &[0, 1, 2, 3]);
        let (_, info2) =
            load_tiff_folder_with_options(dir2.path(), None, &TiffFolderOptions::default())
                .unwrap();
        assert!(info2.chunks_summed);

        // A duplicate pair that does NOT include the first chunk ([A, B, B])
        // must also be caught — the guard compares against ALL earlier
        // chunks, not just the first.
        let dir3 = tempfile::tempdir().unwrap();
        write_chunk_files(dir3.path(), "run", 764, 100, &[0, 1, 2, 3]); // A
        write_chunk_files(dir3.path(), "run", 765, 200, &[0, 1, 2, 3]); // B
        write_chunk_files(dir3.path(), "run", 766, 200, &[0, 1, 2, 3]); // B' == B
        let err3 = load_tiff_folder_with_options(dir3.path(), None, &TiffFolderOptions::default())
            .expect_err("[A, B, B] must be rejected — 766 duplicates 765");
        match err3 {
            IoError::ChunkMismatch { details, .. } => assert!(
                details.contains("766") && details.contains("765"),
                "must name 766 as identical to 765, got: {details}"
            ),
            other => panic!("expected ChunkMismatch, got {other:?}"),
        }
    }

    /// T2: sum_chunks=false loads the legacy lexicographic concatenation.
    #[test]
    fn test_chunked_sum_opt_out() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1, 2, 3]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1, 2, 3]);

        let options = TiffFolderOptions {
            sum_chunks: false,
            ..Default::default()
        };
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
            let mut clipped = 0usize;
            load_frames_from_paths(&paths, PixelValuePolicy::Reject, &mut clipped).unwrap()
        };
        assert_eq!(arr, legacy);
        assert_eq!(
            info,
            TiffLoadInfo {
                n_files: 3,
                n_chunks: 1,
                chunk_ids: vec![764],
                chunks_summed: false,
                n_clipped_pixels: 0,
                n_unrecognized_files: 0,
                unrecognized_examples: vec![],
                chunk_inconsistent: false,
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
        // All-non-conforming folders are the normal legacy world: no noise.
        assert_eq!(info.n_unrecognized_files, 0);
        assert!(info.unrecognized_examples.is_empty());
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
        // Every stem parsed (prefixes merely differ) — not a stray-file
        // situation, so no unrecognized-file noise.
        assert_eq!(info.n_unrecognized_files, 0);
        assert!(info.unrecognized_examples.is_empty());
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

    /// T25: with `sum_chunks = false`, ragged chunks are NOT a hard error —
    /// they load as the legacy lexicographic concatenation (frame count = the
    /// sum of every file) and the irregularity is surfaced through
    /// `chunk_inconsistent`, not raised.  Inspecting raw frames of a ragged
    /// folder is exactly what the opt-out is for.  (Contrast T5, where the
    /// same folder under the default summing path is a `ChunkMismatch`.)
    #[test]
    fn test_chunked_ragged_opt_out_loads_concatenation() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1, 2]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1]);

        let options = TiffFolderOptions {
            sum_chunks: false,
            ..Default::default()
        };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        // Legacy concatenation of all 5 files (3 from chunk 764, 2 from 765),
        // never a partial sum.
        assert_eq!(arr.shape(), &[5, 2, 2]);
        // Lexicographic: run_764_0000..0002, then run_765_0000..0001.
        assert_eq!(arr[[0, 0, 0]], 100.0);
        assert_eq!(arr[[3, 0, 0]], 200.0);
        assert_eq!(
            info,
            TiffLoadInfo {
                n_files: 5,
                n_chunks: 2,
                chunk_ids: vec![764, 765],
                chunks_summed: false,
                n_clipped_pixels: 0,
                n_unrecognized_files: 0,
                unrecognized_examples: vec![],
                chunk_inconsistent: true,
            }
        );
    }

    /// T26: with `sum_chunks = false`, a duplicate (chunk, frame) pair is
    /// likewise NOT a hard error — the files load as the legacy lexicographic
    /// concatenation and `chunk_inconsistent` is set.  (Contrast T8, where
    /// the same folder under the default summing path is a `ChunkMismatch`.)
    #[test]
    fn test_chunked_duplicate_opt_out_loads_concatenation() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1]);
        let dup = dir.path().join("run_764_0001.tiff");
        write_test_tiff(&dup, &[vec![9, 9, 9, 9]], 2, 2);

        let options = TiffFolderOptions {
            sum_chunks: false,
            ..Default::default()
        };
        let (arr, info) = load_tiff_folder_with_options(dir.path(), None, &options).unwrap();
        // A single chunk (764) with a duplicated frame 1 — three files load
        // verbatim in lexicographic order (`.tif` before `.tiff`).
        assert_eq!(arr.shape(), &[3, 2, 2]);
        assert_eq!(
            info,
            TiffLoadInfo {
                n_files: 3,
                n_chunks: 1,
                chunk_ids: vec![764],
                chunks_summed: false,
                n_clipped_pixels: 0,
                n_unrecognized_files: 0,
                unrecognized_examples: vec![],
                chunk_inconsistent: true,
            }
        );
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

    /// T10: a mixed folder (chunk-patterned files plus one stray) falls
    /// back to legacy lexicographic loading — but the fallback is counted:
    /// the stray is reported in `n_unrecognized_files` and named in
    /// `unrecognized_examples`, so a mis-picked raw chunked run folder can
    /// never load as a k× concatenated stack with zero provenance.
    #[test]
    fn test_mixed_folder_legacy_fallback_counts_unrecognized() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0, 1]);
        write_chunk_files(dir.path(), "run", 765, 200, &[0, 1]);
        write_test_tiff(&dir.path().join("overview.tif"), &[vec![7, 7, 7, 7]], 2, 2);

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        // Legacy lexicographic concatenation of all 5 files, stray first
        // ("overview.tif" < "run_..."), never a chunk sum.
        assert_eq!(arr.shape(), &[5, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 7.0);
        assert_eq!(arr[[1, 0, 0]], 100.0);
        assert_eq!(
            info,
            TiffLoadInfo {
                n_files: 5,
                n_chunks: 0,
                chunk_ids: vec![],
                chunks_summed: false,
                n_clipped_pixels: 0,
                n_unrecognized_files: 1,
                unrecognized_examples: vec!["overview.tif".to_string()],
                chunk_inconsistent: false,
            }
        );
    }

    /// T11: `unrecognized_examples` is capped at
    /// [`MAX_UNRECOGNIZED_EXAMPLES`] lexicographically-first names while
    /// `n_unrecognized_files` keeps the full count.
    #[test]
    fn test_unrecognized_examples_capped() {
        let dir = tempfile::tempdir().unwrap();
        write_chunk_files(dir.path(), "run", 764, 100, &[0]);
        for name in ["stray_d.tif", "stray_c.tif", "stray_b.tif", "stray_a.tif"] {
            write_test_tiff(&dir.path().join(name), &[vec![1, 1, 1, 1]], 2, 2);
        }

        let (arr, info) =
            load_tiff_folder_with_options(dir.path(), None, &TiffFolderOptions::default()).unwrap();
        assert_eq!(arr.shape(), &[5, 2, 2]);
        assert_eq!(info.n_unrecognized_files, 4);
        assert_eq!(info.unrecognized_examples.len(), MAX_UNRECOGNIZED_EXAMPLES);
        assert_eq!(
            info.unrecognized_examples,
            vec!["stray_a.tif", "stray_b.tif", "stray_c.tif"]
        );
    }

    /// A nonexistent folder path is `FileNotFound` carrying the *real* OS
    /// error kind `NotFound` (Python: `FileNotFoundError`); `NotADirectory`
    /// is reserved for paths that exist but are not directories (see
    /// `test_load_tiff_folder_not_a_directory`).  The kind is now the genuine
    /// `std::fs::metadata` error, not a synthesized sentinel, so a
    /// permission-denied parent (EACCES — not portably reproducible in a unit
    /// test) surfaces as its true `PermissionDenied` kind and falls through
    /// to `OSError` rather than being mislabeled `FileNotFoundError`.
    #[test]
    fn test_load_tiff_folder_missing_dir_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no_such_dir");

        let err = load_tiff_folder_with_options(&missing, None, &TiffFolderOptions::default())
            .unwrap_err();
        assert!(
            matches!(
                &err,
                IoError::FileNotFound(_, source)
                    if source.kind() == std::io::ErrorKind::NotFound
            ),
            "Expected FileNotFound with kind NotFound, got: {:?}",
            err,
        );
    }

    /// T23: dimension mismatch across chunks is surfaced as DimensionMismatch.
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

    /// T24: three chunks sum element-wise.
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
