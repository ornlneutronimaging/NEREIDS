//! NeXus/HDF5 reading for rustpix-processed neutron imaging data.
//!
//! Supports two data modalities from rustpix output files:
//! - **Histogram**: 4D counts array `(rot_angle, y, x, tof)`.  The loader
//!   requires the caller to choose how multi-angle files are handled via
//!   [`MultiAngleMode`] (error, sum, or select-angle) and transposes the
//!   chosen 3D slice to NEREIDS convention `(tof, y, x)`.
//! - **Events**: per-neutron `(event_time_offset, x, y)` histogrammed into
//!   a `(tof, y, x)` grid with user-specified binning parameters.
//!
//! ## Multi-angle handling (issue #430)
//!
//! Earlier revisions of this module silently summed multi-angle
//! histograms into a single `(tof, y, x)` volume at load time — an
//! irreversible data loss in the import path.  The default now is to
//! **refuse** multi-angle files via [`MultiAngleMode::Error`]; callers
//! who genuinely want the legacy sum-over-angles behaviour opt in
//! explicitly with [`MultiAngleMode::Sum`], and callers who want to
//! work with a single projection from a multi-angle acquisition
//! choose [`MultiAngleMode::SelectAngle`].
//!
//! ## HDF5 Schema (rustpix convention)
//!
//! ```text
//! /entry/histogram/counts          — u64 4D [rot_angle, y, x, tof]
//! /entry/histogram/time_of_flight  — f64 1D, TOF axis (see "Units" below)
//! /entry/neutrons/event_time_offset — u64 1D, TOF per event (see "Units" below)
//! /entry/neutrons/x                — f64 1D, pixel coordinate
//! /entry/neutrons/y                — f64 1D, pixel coordinate
//! /entry/pixel_masks/dead          — u8  2D [y, x]
//! ```
//!
//! Metadata attributes on `/entry` or group level:
//! - `flight_path_m` (f64)
//! - `tof_offset_ns` (f64)
//!
//! ## Units convention
//!
//! **Canonical internal TOF unit: microseconds (µs).**  Every TOF
//! quantity returned to NEREIDS callers — `tof_edges_us`,
//! `EventBinningParams::tof_min_us`/`tof_max_us`, downstream
//! `nereids_io::tof` energy conversions — is in µs.  All other parts of
//! the pipeline (energy mapping, normalization, fitting) assume µs.
//!
//! On read, both `time_of_flight` (histogram path) and
//! `event_time_offset` (events path) consult the HDF5 `units`
//! attribute on the dataset (the NeXus/NXtof convention) and rescale
//! to µs accordingly:
//!
//! | `units` attribute       | Rescale to µs |
//! |-------------------------|---------------|
//! | `ns`, `nanoseconds`     | `× 1e-3`      |
//! | `us`, `µs`, `microseconds` | `× 1`      |
//! | `ms`, `milliseconds`    | `× 1e3`       |
//! | `s`, `seconds`          | `× 1e6`       |
//! | (missing)               | `× 1e-3` (assume ns — rustpix legacy default) |
//! | anything else           | hard error    |
//!
//! The "missing → assume ns" fallback preserves backward compatibility
//! with the rustpix producer and the maintainers' VENUS fixture
//! extraction tooling, which write nanoseconds without a `units`
//! attribute.  Any file
//! that *does* set `units` is parsed strictly: an unrecognised value
//! is rejected rather than silently mis-scaled.  This closes a
//! 1000× silent-rescale bug on `units = "us"` (issue #554).

use std::path::Path;

use hdf5::types::VarLenUnicode;
use ndarray::{Array3, s};

use crate::error::IoError;

/// Multiplicative scale factor from a NeXus `units` attribute string to
/// the canonical internal unit (microseconds).
///
/// See the module-level "Units convention" table for the full mapping.
/// `None` for the `units` attribute means "attribute absent" and falls
/// back to the rustpix legacy assumption of nanoseconds.  Any
/// recognised unit is matched case-insensitively after trimming
/// surrounding whitespace.  An unrecognised non-empty string returns
/// an error rather than silently mis-scaling.
fn tof_scale_to_us(units: Option<&str>) -> Result<f64, IoError> {
    match units {
        // Absent attribute — rustpix legacy default.  The project's own
        // fixture producers (the maintainers' VENUS extraction tooling)
        // write nanoseconds without a `units` attribute, so we
        // preserve that contract for backward compatibility.
        None => Ok(1e-3),
        Some(raw) => {
            let normalised = raw.trim().to_ascii_lowercase();
            match normalised.as_str() {
                "ns" | "nanosecond" | "nanoseconds" => Ok(1e-3),
                // "µs" lowercases to "µs" — the only non-ASCII form we
                // accept.  The MICRO SIGN U+00B5 (the literal "µ"
                // appearing in source above) and the Greek small
                // letter MU U+03BC are visually identical but are
                // distinct Unicode codepoints; both are written
                // verbatim by various NeXus producers, so we accept
                // both.
                "us" | "µs" | "\u{03bc}s" | "microsecond" | "microseconds" => Ok(1.0),
                "ms" | "millisecond" | "milliseconds" => Ok(1e3),
                "s" | "sec" | "second" | "seconds" => Ok(1e6),
                _ => Err(IoError::InvalidParameter(format!(
                    "Unsupported NeXus TOF units attribute {raw:?}: expected one of \
                     'ns', 'us'/'µs', 'ms', 's' (case-insensitive); refusing to \
                     guess a scale factor (issue #554)"
                ))),
            }
        }
    }
}

/// Read a string-valued attribute from an HDF5 `Location` (Group or
/// Dataset both deref to `Location`), returning `None` if the
/// attribute is absent.  Both storage conventions decode: variable-length
/// (rustpix) and fixed-length (SNS/ADARA) strings, ASCII or UTF-8, with
/// trailing NUL/space padding trimmed.  `Err` when the attribute exists
/// but is not a string, is a fixed string longer than the 1024-byte read
/// buffer, or cannot be read/decoded.
///
/// Absence is detected via [`Location::attr_names`] (rather than
/// catching any error from [`Location::attr`]) so that genuine HDF5
/// errors — corrupt file, permission denied, internal failure —
/// surface as [`IoError::InvalidParameter`] instead of silently
/// becoming "attribute missing".  This was a latent bug:
/// the previous implementation mapped *every* `attr()` failure to
/// `Ok(None)`, including non-"not found" errors.
pub(crate) fn read_string_attr(
    loc: &hdf5::Location,
    name: &str,
) -> Result<Option<String>, IoError> {
    // Probe the attribute table first.  `attr_names()` is the only
    // discriminator the hdf5-metno 0.12 `Error` enum exposes for
    // "absent vs. other failure" — its `Error` is a flat
    // `HDF5(ErrorStack) | Internal(String)` with no typed
    // "attribute not found" variant.
    let names = loc.attr_names().map_err(|e| {
        IoError::InvalidParameter(format!(
            "Failed to list attributes while looking for {name:?}: {e}"
        ))
    })?;
    if !names.iter().any(|n| n == name) {
        return Ok(None);
    }
    let attr = loc.attr(name).map_err(|e| {
        IoError::InvalidParameter(format!(
            "Failed to open attribute {name:?} (listed but unreadable): {e}"
        ))
    })?;
    // Producers disagree on string storage: rustpix writes variable-length
    // UTF-8, while SNS/ADARA facility files write fixed-length ASCII
    // (e.g. the 35-byte ISO timestamp on `event_time_zero@offset`).
    // Dispatch on the stored type descriptor instead of assuming one
    // (issue #637; previously only variable-length UTF-8 was readable).
    use hdf5::types::{FixedAscii, FixedUnicode, TypeDescriptor, VarLenAscii};
    let td = attr.dtype().and_then(|d| d.to_descriptor()).map_err(|e| {
        IoError::InvalidParameter(format!("Failed to inspect type of attribute {name:?}: {e}"))
    })?;
    let read_err = |e: hdf5::Error| {
        IoError::InvalidParameter(format!(
            "Failed to read string attribute {name:?}: {e} (stored as {td:?})"
        ))
    };
    let value = match td {
        TypeDescriptor::VarLenUnicode => attr
            .read_scalar::<VarLenUnicode>()
            .map_err(read_err)?
            .as_str()
            .to_string(),
        TypeDescriptor::VarLenAscii => attr
            .read_scalar::<VarLenAscii>()
            .map_err(read_err)?
            .as_str()
            .to_string(),
        // Fixed-length strings: HDF5's string-to-string soft conversion
        // repacks any length into this generous fixed buffer; trim the
        // NUL/space padding it leaves behind.
        TypeDescriptor::FixedAscii(n) | TypeDescriptor::FixedUnicode(n) if n <= 1024 => match td {
            TypeDescriptor::FixedAscii(_) => attr
                .read_scalar::<FixedAscii<1024>>()
                .map_err(read_err)?
                .as_str()
                .to_string(),
            _ => attr
                .read_scalar::<FixedUnicode<1024>>()
                .map_err(read_err)?
                .as_str()
                .to_string(),
        },
        TypeDescriptor::FixedAscii(n) | TypeDescriptor::FixedUnicode(n) => {
            return Err(IoError::InvalidParameter(format!(
                "String attribute {name:?} is {n} bytes, exceeding the supported \
                 fixed-string read buffer (1024)"
            )));
        }
        other => {
            return Err(IoError::InvalidParameter(format!(
                "Attribute {name:?} is not a string (stored as {other:?})"
            )));
        }
    };
    let value = value.trim_end_matches(['\0', ' ']).to_string();
    Ok(Some(value))
}

/// Metadata probed from a NeXus/HDF5 file without loading full data.
#[derive(Debug, Clone)]
pub struct NexusMetadata {
    /// Whether `/entry/histogram/counts` exists.
    pub has_histogram: bool,
    /// Whether `/entry/neutrons` group exists with event data.
    pub has_events: bool,
    /// Shape of the histogram `(rot_angle, y, x, tof)`, if present.
    pub histogram_shape: Option<[usize; 4]>,
    /// Number of events in `/entry/neutrons/event_time_offset`, if present.
    pub n_events: Option<usize>,
    /// Flight path in meters (from attributes), if present.
    pub flight_path_m: Option<f64>,
    /// TOF offset in nanoseconds (from attributes), if present.
    pub tof_offset_ns: Option<f64>,
    /// TOF bin edges or centers in **microseconds**, if present.  The
    /// probe path consults the dataset's `units` attribute and
    /// rescales to µs the same way [`load_nexus_histogram`] does, so
    /// this field is unit-consistent with [`NexusHistogramData::tof_edges_us`].
    pub tof_edges_us: Option<Vec<f64>>,
}

/// An entry in the HDF5 group/dataset tree hierarchy.
#[derive(Debug, Clone)]
pub struct Hdf5TreeEntry {
    /// Full path within the HDF5 file (e.g., `/entry/histogram/counts`).
    pub path: String,
    /// Whether this entry is a group or dataset.
    pub kind: Hdf5EntryKind,
    /// Dataset shape, if this entry is a dataset.
    pub shape: Option<Vec<usize>>,
}

/// Kind of HDF5 tree entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hdf5EntryKind {
    Group,
    Dataset,
}

/// Histogram data loaded from a NeXus file, ready for NEREIDS processing.
#[derive(Debug, Clone)]
pub struct NexusHistogramData {
    /// Counts array in NEREIDS convention: `(n_tof, height, width)`.
    pub counts: Array3<f64>,
    /// TOF bin edges in microseconds.
    pub tof_edges_us: Vec<f64>,
    /// Flight path in meters, if available from the file.
    pub flight_path_m: Option<f64>,
    /// Dead pixel mask from `/entry/pixel_masks/dead`, if present.
    pub dead_pixels: Option<ndarray::Array2<bool>>,
    /// Number of rotation angles summed (D-5). 1 means no collapse occurred.
    pub n_rotation_angles: usize,
    /// Event retention statistics (only populated for event-mode loading).
    pub event_stats: Option<EventRetentionStats>,
}

/// Statistics on how many events were kept vs dropped during histogramming.
#[derive(Debug, Clone)]
pub struct EventRetentionStats {
    /// Total events read from the file.
    pub total: usize,
    /// Events successfully histogrammed.
    pub kept: usize,
    /// Events dropped due to non-finite values in TOF or spatial coordinates.
    ///
    /// For u64 TOF input (`event_time_offset`), the TOF channel is always
    /// finite, so the TOF path contributes zero to this counter. Non-finite
    /// values arise from the f64 x/y pixel coordinates (NaN or Inf from
    /// upstream processing or detector artifacts).
    pub dropped_non_finite: usize,
    /// Events dropped due to TOF outside `[tof_min, tof_max)`.
    pub dropped_tof_range: usize,
    /// Events dropped due to pixel coordinates outside detector bounds.
    pub dropped_spatial: usize,
}

/// Probe a NeXus/HDF5 file for available data modalities and metadata.
///
/// Opens the file read-only and checks for histogram and event groups
/// without loading any large datasets.
pub fn probe_nexus(path: &Path) -> Result<NexusMetadata, IoError> {
    let file = hdf5::File::open(path).map_err(|e| {
        IoError::FileNotFound(
            path.display().to_string(),
            std::io::Error::other(e.to_string()),
        )
    })?;

    let entry = file
        .group("entry")
        .map_err(|e| IoError::InvalidParameter(format!("Missing /entry group: {e}")))?;

    // Probe histogram
    let (has_histogram, histogram_shape, tof_edges_us) = probe_histogram_group(&entry);

    // Probe events
    let (has_events, n_events) = probe_event_group(&entry);

    // Read metadata attributes from the /entry group
    let flight_path_m = read_f64_attr(&entry, "flight_path_m");
    let tof_offset_ns = read_f64_attr(&entry, "tof_offset_ns");

    Ok(NexusMetadata {
        has_histogram,
        has_events,
        histogram_shape,
        n_events,
        flight_path_m,
        tof_offset_ns,
        tof_edges_us,
    })
}

/// Policy for handling multi-angle NeXus histogram files.
///
/// Issue #430: the loader must refuse to silently collapse the
/// rotation-angle dimension.  Callers choose explicitly which
/// projection (or combination of projections) they want.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MultiAngleMode {
    /// Reject files with more than one rotation angle with a clear
    /// [`IoError::InvalidParameter`].  Single-angle files (`n_rot == 1`)
    /// load normally.  This is the default — it prevents silent data
    /// loss for callers that aren't multi-angle-aware.
    #[default]
    Error,
    /// Sum across all rotation angles into a single `(tof, y, x)`
    /// volume.  This is the legacy auto-sum behaviour, preserved as an
    /// **explicit opt-in** so that callers can't invoke it by
    /// accident.  Multi-angle analysis information is irreversibly
    /// lost on this path.
    Sum,
    /// Extract a single rotation angle by index.  Returns an error if
    /// the index is out of range.
    SelectAngle(usize),
}

/// Load histogram data from a NeXus file, refusing multi-angle inputs.
///
/// Reads `/entry/histogram/counts` (u64 4D), converts to f64, and
/// transposes the chosen single-angle slice to NEREIDS convention
/// `(tof, y, x)`.  TOF values are converted from nanoseconds to
/// microseconds.
///
/// If the file has more than one rotation angle (`n_rot > 1`), the
/// call returns [`IoError::InvalidParameter`] pointing at
/// [`load_nexus_histogram_with_mode`] — silent sum-over-angles
/// was the pre-#430 behaviour and has been removed because it lost
/// projection-resolved information without the caller's knowledge.
///
/// Single-angle files (`n_rot == 1`) load normally and reach the same
/// output as before #430.
pub fn load_nexus_histogram(path: &Path) -> Result<NexusHistogramData, IoError> {
    load_nexus_histogram_with_mode(path, MultiAngleMode::Error)
}

/// Load histogram data from a NeXus file with an explicit multi-angle
/// handling policy.  See [`MultiAngleMode`] for the options.
///
/// This is the explicit-opt-in variant behind
/// [`load_nexus_histogram`].  Use it when you know the file may have
/// multiple rotation angles and you have made a deliberate choice
/// about how to combine them.
pub fn load_nexus_histogram_with_mode(
    path: &Path,
    mode: MultiAngleMode,
) -> Result<NexusHistogramData, IoError> {
    let file = hdf5::File::open(path).map_err(|e| {
        IoError::FileNotFound(
            path.display().to_string(),
            std::io::Error::other(e.to_string()),
        )
    })?;

    let entry = file
        .group("entry")
        .map_err(|e| IoError::InvalidParameter(format!("Missing /entry group: {e}")))?;

    let hist_group = entry
        .group("histogram")
        .map_err(|e| IoError::InvalidParameter(format!("Missing /entry/histogram group: {e}")))?;

    // Read counts: u64 4D [rot_angle, y, x, tof]
    let counts_ds = hist_group.dataset("counts").map_err(|e| {
        IoError::InvalidParameter(format!("Missing /entry/histogram/counts dataset: {e}"))
    })?;

    let shape = counts_ds.shape();
    if shape.len() != 4 {
        return Err(IoError::ShapeMismatch(format!(
            "Expected 4D histogram counts, got {}D",
            shape.len()
        )));
    }

    // Validate the rotation-angle policy BEFORE reading the full 4D
    // counts dataset: the check is purely metadata-driven and the
    // rejection paths should be cheap.  Reading the full u64 cube just
    // to error out is wasteful on production multi-angle NeXus files
    // (easily multi-GB), and historically caused OOM-before-error on
    // the default "refuse" code path.
    let n_rot = shape[0];
    if n_rot == 0 {
        // Degenerate file with a zero-sized rotation-angle axis.
        // Reject rather than produce an all-zero output (which would
        // look like a valid-but-empty measurement).
        return Err(IoError::InvalidParameter(
            "NeXus histogram has zero rotation angles; /entry/histogram/counts axis 0 must \
             be >= 1"
                .into(),
        ));
    }
    // Mirror the rotation-angle guard for the sibling y / x / tof axes
    // (counts layout is [rot_angle, y, x, tof]).  A zero-sized y, x, or tof
    // axis is just as degenerate: it would produce an empty detector plane or
    // an empty energy series that looks like a valid-but-empty measurement
    // downstream instead of a clear load error.
    for (axis, name) in [(1usize, "y"), (2, "x"), (3, "tof")] {
        if shape[axis] == 0 {
            return Err(IoError::InvalidParameter(format!(
                "NeXus histogram has a zero-sized {name} axis; \
                 /entry/histogram/counts axis {axis} must be >= 1 (shape {shape:?})"
            )));
        }
    }
    match mode {
        MultiAngleMode::Error if n_rot > 1 => {
            return Err(IoError::InvalidParameter(format!(
                "NeXus histogram has {n_rot} rotation angles — refusing to silently \
                 combine them (issue #430).  Call load_nexus_histogram_with_mode with \
                 MultiAngleMode::Sum to preserve the legacy sum-over-angles behaviour, \
                 or MultiAngleMode::SelectAngle(i) to extract a single projection."
            )));
        }
        MultiAngleMode::SelectAngle(idx) if idx >= n_rot => {
            return Err(IoError::InvalidParameter(format!(
                "MultiAngleMode::SelectAngle({idx}) out of range: file has {n_rot} \
                 rotation angle(s); valid indices are 0..{n_rot} (exclusive, i.e. \
                 last valid index is {last})",
                last = n_rot - 1
            )));
        }
        _ => {}
    }

    // Read only the rotation-angle slice(s) the caller actually needs.
    // Reading the full 4D cube when the caller wants one projection is
    // wasteful on production multi-angle files (multi-GB per
    // acquisition).
    //
    // - `Error` is guaranteed to have `n_rot == 1` (validated above),
    //   so we hyperslab-read the single projection.
    // - `Sum` on a single-angle file is identity with `Error`.
    // - `Sum` on a multi-angle file needs every angle; the full read
    //   is unavoidable and the legacy opt-in carries its memory cost.
    // - `SelectAngle(idx)` hyperslab-reads only the selected
    //   projection — the other angles' bytes never enter memory.
    //
    // All paths produce a `[y, x, tof]` 3D `u64` array ready for the
    // f64 conversion + transpose below.
    let combined_yxtof: ndarray::Array3<u64> = match mode {
        MultiAngleMode::Error | MultiAngleMode::Sum if n_rot == 1 => {
            counts_ds.read_slice(s![0, .., .., ..]).map_err(|e| {
                IoError::InvalidParameter(format!("Failed to read single-angle slice: {e}"))
            })?
        }
        MultiAngleMode::Sum => {
            let full: ndarray::Array4<u64> = counts_ds.read().map_err(|e| {
                IoError::InvalidParameter(format!("Failed to read histogram counts: {e}"))
            })?;
            full.sum_axis(ndarray::Axis(0))
        }
        MultiAngleMode::SelectAngle(idx) => {
            counts_ds.read_slice(s![idx, .., .., ..]).map_err(|e| {
                IoError::InvalidParameter(format!("Failed to read selected-angle slice: {e}"))
            })?
        }
        MultiAngleMode::Error => {
            // Unreachable: n_rot > 1 was rejected above, n_rot == 1 is
            // matched by the first arm, n_rot == 0 was rejected earlier.
            unreachable!("Error mode reached with n_rot = {n_rot}")
        }
    };

    // Convert to f64 and transpose [y, x, tof] → NEREIDS convention [tof, y, x]
    let counts_f64: Array3<f64> = combined_yxtof
        .mapv(|v| v as f64)
        .permuted_axes([2, 0, 1])
        .as_standard_layout()
        .into_owned();
    let n_tof = counts_f64.shape()[0];

    // Read TOF axis (nanoseconds → microseconds)
    let tof_edges_us = read_tof_axis(&hist_group)?;

    // Validate TOF edges count against histogram TOF dimension
    if tof_edges_us.len() != n_tof + 1 && tof_edges_us.len() != n_tof {
        return Err(IoError::InvalidParameter(format!(
            "TOF axis length {} is incompatible with {} histogram bins (expected {} or {})",
            tof_edges_us.len(),
            n_tof,
            n_tof,
            n_tof + 1
        )));
    }

    // Read flight path
    let flight_path_m = read_f64_attr(&hist_group, "flight_path_m")
        .or_else(|| read_f64_attr(&entry, "flight_path_m"));

    // Read dead pixel mask, validated against the detector's spatial dims.
    // counts_f64 is [tof, y, x], so (height, width) = (shape[1], shape[2]).
    let dead_pixels = read_dead_pixel_mask(&entry, (counts_f64.shape()[1], counts_f64.shape()[2]))?;

    Ok(NexusHistogramData {
        counts: counts_f64,
        tof_edges_us,
        flight_path_m,
        dead_pixels,
        n_rotation_angles: n_rot,
        event_stats: None, // histogram mode, not events
    })
}

/// Parameters for histogramming neutron event data into a 3D grid.
#[derive(Debug, Clone, PartialEq)]
pub struct EventBinningParams {
    /// Number of TOF bins.
    pub n_bins: usize,
    /// Minimum TOF in microseconds.
    pub tof_min_us: f64,
    /// Maximum TOF in microseconds.
    pub tof_max_us: f64,
    /// Detector height in pixels.
    pub height: usize,
    /// Detector width in pixels.
    pub width: usize,
}

/// Load neutron event data from a NeXus file and histogram into a 3D grid.
///
/// Reads `/entry/neutrons/event_time_offset` (u64), `x` (f64), `y` (f64),
/// rescales TOF to the canonical internal unit of microseconds based on
/// the `event_time_offset` dataset's `units` attribute (issue #554), then
/// bins events into a `(n_bins, height, width)` histogram grid.
///
/// # TOF units handling (issue #554)
///
/// The loader consults the NeXus `units` attribute on the
/// `event_time_offset` dataset and rescales the raw `u64` channel
/// counts to µs accordingly.  See the module-level "Units convention"
/// table for the recognised values.  If the `units` attribute is
/// absent, the loader falls back to the rustpix legacy assumption of
/// nanoseconds (`× 1e-3`); if it is present but unrecognised, the
/// call returns [`IoError::InvalidParameter`] rather than silently
/// guessing a scale factor.
///
/// # Binning behaviour (D-8)
///
/// - **Out-of-range events are dropped and counted**: events with TOF outside
///   `[tof_min_us, tof_max_us)`, pixel coordinates outside `[0, width)` /
///   `[0, height)`, or non-finite spatial coordinates are excluded. Per-category
///   drop counts are returned in [`EventRetentionStats`] via
///   [`NexusHistogramData::event_stats`].
/// - **Pixel coordinates are rounded to the nearest integer** (`f64::round()`
///   then cast to `isize`), snapping sub-pixel positions to a discrete grid.
///   Fractional coordinates exactly at 0.5 round up.
pub fn load_nexus_events(
    path: &Path,
    params: &EventBinningParams,
) -> Result<NexusHistogramData, IoError> {
    if params.n_bins == 0 {
        return Err(IoError::InvalidParameter("n_bins must be positive".into()));
    }
    if params.height == 0 || params.width == 0 {
        return Err(IoError::InvalidParameter(
            "height and width must be positive".into(),
        ));
    }
    if !params.tof_min_us.is_finite() || !params.tof_max_us.is_finite() {
        return Err(IoError::InvalidParameter(
            "TOF bounds must be finite".into(),
        ));
    }
    if params.tof_max_us <= params.tof_min_us {
        return Err(IoError::InvalidParameter(format!(
            "tof_max_us ({}) must be greater than tof_min_us ({})",
            params.tof_max_us, params.tof_min_us
        )));
    }

    let file = hdf5::File::open(path).map_err(|e| {
        IoError::FileNotFound(
            path.display().to_string(),
            std::io::Error::other(e.to_string()),
        )
    })?;

    let entry = file
        .group("entry")
        .map_err(|e| IoError::InvalidParameter(format!("Missing /entry group: {e}")))?;

    let neutrons = entry
        .group("neutrons")
        .map_err(|e| IoError::InvalidParameter(format!("Missing /entry/neutrons group: {e}")))?;

    // Read event arrays.  Open the dataset first so we can consult its
    // `units` attribute (issue #554) before reading the data.
    let tof_ds = neutrons.dataset("event_time_offset").map_err(|e| {
        IoError::InvalidParameter(format!("Missing event_time_offset dataset: {e}"))
    })?;
    let tof_units = read_string_attr(&tof_ds, "units")?;
    let tof_scale = tof_scale_to_us(tof_units.as_deref())?;
    let tof_raw: Vec<u64> = tof_ds
        .read_1d()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read event_time_offset: {e}")))?
        .to_vec();

    let x_coords: Vec<f64> = neutrons
        .dataset("x")
        .map_err(|e| IoError::InvalidParameter(format!("Missing x dataset: {e}")))?
        .read_1d()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read x: {e}")))?
        .to_vec();

    let y_coords: Vec<f64> = neutrons
        .dataset("y")
        .map_err(|e| IoError::InvalidParameter(format!("Missing y dataset: {e}")))?
        .read_1d()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read y: {e}")))?
        .to_vec();

    if tof_raw.len() != x_coords.len() || tof_raw.len() != y_coords.len() {
        return Err(IoError::ShapeMismatch(format!(
            "Event arrays have mismatched lengths: tof={}, x={}, y={}",
            tof_raw.len(),
            x_coords.len(),
            y_coords.len()
        )));
    }

    // Generate linear TOF bin edges
    let tof_edges_us =
        crate::tof::linspace_tof_edges(params.tof_min_us, params.tof_max_us, params.n_bins)?;

    // Histogram events with retention tracking.
    let dt_us = (params.tof_max_us - params.tof_min_us) / params.n_bins as f64;
    let mut counts = Array3::<f64>::zeros((params.n_bins, params.height, params.width));
    let total = tof_raw.len();
    let mut kept = 0usize;
    let mut dropped_non_finite = 0usize;
    let mut dropped_tof_range = 0usize;
    let mut dropped_spatial = 0usize;

    for i in 0..tof_raw.len() {
        // Convert raw TOF to canonical µs via the units-attribute scale
        // factor (issue #554).  For the default rustpix case (`units`
        // absent → ns assumed), `tof_scale` is `1e-3`, recovering the
        // pre-fix expression `tof_raw[i] / 1000.0`.
        let tof_us = tof_raw[i] as f64 * tof_scale;
        if !tof_us.is_finite() {
            dropped_non_finite += 1;
            continue;
        }

        if tof_us < params.tof_min_us || tof_us >= params.tof_max_us {
            dropped_tof_range += 1;
            continue;
        }

        let xf = x_coords[i];
        let yf = y_coords[i];
        if !xf.is_finite() || !yf.is_finite() {
            dropped_non_finite += 1;
            continue;
        }
        let px = xf.round() as isize;
        let py = yf.round() as isize;

        if px < 0 || py < 0 || px >= params.width as isize || py >= params.height as isize {
            dropped_spatial += 1;
            continue;
        }

        let tof_bin = ((tof_us - params.tof_min_us) / dt_us) as usize;
        let tof_bin = tof_bin.min(params.n_bins - 1);
        counts[[tof_bin, py as usize, px as usize]] += 1.0;
        kept += 1;
    }

    // Read flight path
    let flight_path_m = read_f64_attr(&neutrons, "flight_path_m")
        .or_else(|| read_f64_attr(&entry, "flight_path_m"));

    // Read dead pixel mask, validated against the requested detector dims.
    let dead_pixels = read_dead_pixel_mask(&entry, (params.height, params.width))?;

    debug_assert_eq!(
        total,
        kept + dropped_non_finite + dropped_tof_range + dropped_spatial,
        "event retention accounting mismatch"
    );

    Ok(NexusHistogramData {
        counts,
        tof_edges_us,
        flight_path_m,
        dead_pixels,
        n_rotation_angles: 1,
        event_stats: Some(EventRetentionStats {
            total,
            kept,
            dropped_non_finite,
            dropped_tof_range,
            dropped_spatial,
        }),
    })
}

// ---- Internal helpers ----

/// Probe the histogram group for shape and TOF axis without loading counts.
///
/// The returned TOF edges are in **microseconds**, rescaled from the
/// dataset's NeXus `units` attribute via [`tof_scale_to_us`] — the
/// same logic the full [`load_nexus_histogram`] uses.  If the `units`
/// attribute is unparseable, the TOF axis is dropped entirely
/// (returned as `None`) rather than silently propagated at the wrong
/// scale, matching the function's "any failure → no data for that
/// field" contract.  The previous implementation returned the raw
/// values verbatim — a silent 1000× error for any file written with
/// `units = "us"`, symmetric with the load-path bug closed by issue
/// #554.
fn probe_histogram_group(entry: &hdf5::Group) -> (bool, Option<[usize; 4]>, Option<Vec<f64>>) {
    let hist = match entry.group("histogram") {
        Ok(g) => g,
        Err(_) => return (false, None, None),
    };

    let counts = match hist.dataset("counts") {
        Ok(ds) => ds,
        Err(_) => return (false, None, None),
    };

    let shape = counts.shape();
    if shape.len() != 4 {
        return (false, None, None);
    }

    let histogram_shape = Some([shape[0], shape[1], shape[2], shape[3]]);

    // Try reading TOF axis and rescaling to µs via the `units`
    // attribute.  Any failure (missing dataset, read error,
    // unparseable units attr) collapses to `None` — the probe is
    // best-effort and must never poison the rest of the metadata.
    let tof_edges_us = hist.dataset("time_of_flight").ok().and_then(|ds| {
        let raw = ds.read_1d::<f64>().ok()?.to_vec();
        // `read_string_attr` returns Ok(None) for absent and Err for
        // genuine HDF5 failures; either should propagate to "no TOF
        // axis" rather than fall through to the wrong-scale raw
        // values.
        let units = read_string_attr(&ds, "units").ok()?;
        let scale = tof_scale_to_us(units.as_deref()).ok()?;
        Some(raw.into_iter().map(|v| v * scale).collect())
    });

    (true, histogram_shape, tof_edges_us)
}

/// Probe the neutron event group for event count.
fn probe_event_group(entry: &hdf5::Group) -> (bool, Option<usize>) {
    let neutrons = match entry.group("neutrons") {
        Ok(g) => g,
        Err(_) => return (false, None),
    };

    let n_events = neutrons
        .dataset("event_time_offset")
        .ok()
        .map(|ds| ds.shape().first().copied().unwrap_or(0));

    (n_events.is_some(), n_events)
}

/// Read TOF axis from the histogram group, rescaling to µs based on
/// the dataset's `units` attribute (see module docs / issue #554).
fn read_tof_axis(hist_group: &hdf5::Group) -> Result<Vec<f64>, IoError> {
    let tof_ds = hist_group.dataset("time_of_flight").map_err(|e| {
        IoError::InvalidParameter(format!(
            "Missing /entry/histogram/time_of_flight dataset: {e}"
        ))
    })?;

    let raw: Vec<f64> = tof_ds
        .read_1d::<f64>()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read time_of_flight: {e}")))?
        .to_vec();

    // Consult the dataset's NeXus `units` attribute.  Missing →
    // legacy nanoseconds assumption (rustpix); known value → use
    // table; unknown value → hard error (no silent mis-scale).
    let units = read_string_attr(&tof_ds, "units")?;
    let scale = tof_scale_to_us(units.as_deref())?;

    let edges: Vec<f64> = raw.iter().map(|&v| v * scale).collect();

    // Validate the TOF axis is finite, strictly positive, and strictly
    // increasing, mirroring the spectrum-file load path (`guided::load` runs
    // `validate_monotonic` on the parsed spectrum before use).  A non-finite,
    // non-positive, or non-increasing TOF edge produces a `tof_to_energy` NaN /
    // negative-energy downstream; reject it here at the I/O boundary instead.
    //
    // `validate_monotonic` alone is *not* sufficient for the finite/positive
    // half: a trailing `+∞` satisfies `prev < +∞` (so monotonicity passes), a
    // single-edge axis never enters `windows(2)` at all, and `first <= 0.0` is
    // bypassed by `NaN` (`NaN <= 0.0` is `false`).  Check every scaled edge
    // explicitly with `is_finite() && > 0.0` (the `is_finite()` half is what
    // catches `NaN` / `±∞`, which order comparisons silently pass), then defer
    // the strictly-increasing requirement to `validate_monotonic`.
    for (i, &edge) in edges.iter().enumerate() {
        if !edge.is_finite() || edge <= 0.0 {
            return Err(IoError::InvalidParameter(format!(
                "NeXus TOF axis edge {i} must be finite and positive, got {edge}"
            )));
        }
    }
    crate::spectrum::validate_monotonic(&edges)?;

    Ok(edges)
}

/// Read a scalar f64 attribute from a group.
fn read_f64_attr(group: &hdf5::Group, name: &str) -> Option<f64> {
    group
        .attr(name)
        .ok()
        .and_then(|a| a.read_scalar::<f64>().ok())
}

/// Read the dead-pixel mask from `/entry/pixel_masks/dead`, validating its
/// shape against the detector's `(height, width)`.
///
/// Returns `Ok(None)` when the mask group / dataset is simply *absent* (a file
/// without a dead-pixel mask is valid).  Returns `Err` when the mask is
/// *present but malformed*:
/// * `pixel_masks` exists but is not a group, or `dead` exists but is not a
///   readable dataset — surfaced as `InvalidParameter` rather than silently
///   treated as absence (a malformed mask is an upstream-writer bug, and
///   silently dropping it would mask the wrong pixels or none at all);
/// * the mask shape does not match the counts' spatial dimensions — surfaced
///   as `ShapeMismatch`.
///
/// Absence vs malformed is decided by link existence (`member_names`), not by
/// whether `group()` / `dataset()` *succeed*: those collapse "the link is not
/// there" and "the link is there but the wrong object kind / unreadable" into
/// the same `Err`, which would otherwise mask real corruption as absence.
fn read_dead_pixel_mask(
    entry: &hdf5::Group,
    expected_hw: (usize, usize),
) -> Result<Option<ndarray::Array2<bool>>, IoError> {
    // `pixel_masks` link absent → no mask (valid file).
    let entry_members = entry
        .member_names()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to list /entry members: {e}")))?;
    if !entry_members.iter().any(|n| n == "pixel_masks") {
        return Ok(None);
    }
    // Link present but not openable as a group → malformed, not absent.
    let masks = entry.group("pixel_masks").map_err(|e| {
        IoError::InvalidParameter(format!(
            "/entry/pixel_masks is present but is not a readable group: {e}"
        ))
    })?;

    // `dead` link absent → no mask (valid file).
    let mask_members = masks.member_names().map_err(|e| {
        IoError::InvalidParameter(format!("Failed to list /entry/pixel_masks members: {e}"))
    })?;
    if !mask_members.iter().any(|n| n == "dead") {
        return Ok(None);
    }
    // Link present but not openable as a dataset → malformed, not absent.
    let dead_ds = masks.dataset("dead").map_err(|e| {
        IoError::InvalidParameter(format!(
            "/entry/pixel_masks/dead is present but is not a readable dataset: {e}"
        ))
    })?;
    let dead_u8: ndarray::Array2<u8> = dead_ds.read().map_err(|e| {
        IoError::InvalidParameter(format!("Failed to read /entry/pixel_masks/dead: {e}"))
    })?;
    let (eh, ew) = expected_hw;
    if dead_u8.dim() != (eh, ew) {
        return Err(IoError::ShapeMismatch(format!(
            "dead-pixel mask shape {:?} != detector spatial dimensions ({eh}, {ew})",
            dead_u8.dim(),
        )));
    }
    Ok(Some(dead_u8.mapv(|v| v != 0)))
}

/// List the group/dataset tree structure of an HDF5 file.
///
/// Walks the file hierarchy recursively up to `max_depth` levels deep,
/// returning entries with their path, kind (group vs dataset), and shape
/// (for datasets).  Useful for displaying file structure in a GUI browser.
pub fn list_hdf5_tree(path: &Path, max_depth: usize) -> Result<Vec<Hdf5TreeEntry>, IoError> {
    let file = hdf5::File::open(path)
        .map_err(|e| IoError::Hdf5Error(format!("Cannot open HDF5 file: {e}")))?;
    let mut entries = Vec::new();
    walk_group(
        &file
            .as_group()
            .map_err(|e| IoError::Hdf5Error(format!("Cannot read root group: {e}")))?,
        "/",
        0,
        max_depth,
        &mut entries,
    );
    Ok(entries)
}

/// Recursively walk an HDF5 group, collecting tree entries.
fn walk_group(
    group: &hdf5::Group,
    prefix: &str,
    depth: usize,
    max_depth: usize,
    entries: &mut Vec<Hdf5TreeEntry>,
) {
    let Ok(members) = group.member_names() else {
        return;
    };
    let mut members = members;
    members.sort();
    for name in &members {
        let child_path = if prefix == "/" {
            format!("/{name}")
        } else {
            format!("{prefix}/{name}")
        };

        // Try dataset first (leaf nodes)
        if let Ok(ds) = group.dataset(name) {
            let shape = ds.shape();
            entries.push(Hdf5TreeEntry {
                path: child_path,
                kind: Hdf5EntryKind::Dataset,
                shape: Some(shape),
            });
        } else if let Ok(child_group) = group.group(name) {
            // It's a group — record it and recurse if within depth
            entries.push(Hdf5TreeEntry {
                path: child_path.clone(),
                kind: Hdf5EntryKind::Group,
                shape: None,
            });
            if depth < max_depth {
                walk_group(&child_group, &child_path, depth + 1, max_depth, entries);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal NeXus HDF5 file with histogram data for testing.
    fn create_test_histogram(
        path: &Path,
        counts: &[u64],
        shape: [usize; 4],
        tof_ns: &[f64],
        flight_path_m: Option<f64>,
    ) {
        let file = hdf5::File::create(path).expect("create test file");
        let entry = file.create_group("entry").expect("create entry");

        if let Some(fp) = flight_path_m {
            entry
                .new_attr::<f64>()
                .shape(())
                .create("flight_path_m")
                .expect("create attr")
                .write_scalar(&fp)
                .expect("write attr");
        }

        let hist = entry.create_group("histogram").expect("create histogram");
        hist.new_dataset::<u64>()
            .shape(shape)
            .create("counts")
            .expect("create counts")
            .write_raw(counts)
            .expect("write counts");

        hist.new_dataset::<f64>()
            .shape([tof_ns.len()])
            .create("time_of_flight")
            .expect("create tof")
            .write_raw(tof_ns)
            .expect("write tof");
    }

    #[test]
    fn test_probe_nexus_histogram() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");

        // 1 rot angle, 2x3 spatial, 4 TOF bins → shape [1, 2, 3, 4]
        let counts = vec![0u64; 24];
        let tof_ns = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0]; // 5 edges for 4 bins
        create_test_histogram(&path, &counts, [1, 2, 3, 4], &tof_ns, Some(25.0));

        let meta = probe_nexus(&path).unwrap();
        assert!(meta.has_histogram);
        assert!(!meta.has_events);
        assert_eq!(meta.histogram_shape, Some([1, 2, 3, 4]));
        assert_eq!(meta.flight_path_m, Some(25.0));
        // No `units` attribute on this fixture → rustpix legacy-ns
        // assumption, so the probe rescales 1000/2000/.../5000 ns
        // into 1/2/.../5 µs (× 1e-3).
        let edges = meta.tof_edges_us.expect("probe should return TOF edges");
        assert_eq!(edges.len(), 5);
        for (i, &expected_us) in [1.0_f64, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            assert!(
                (edges[i] - expected_us).abs() < 1e-12,
                "edge {i}: expected {expected_us} µs, got {} µs",
                edges[i]
            );
        }
    }

    /// `probe_nexus` must respect the `units`
    /// attribute on `time_of_flight` the same way `load_nexus_histogram`
    /// does.  A file written with `units = "us"` must surface µs
    /// values verbatim through the probe (no 1000× silent rescale).
    #[test]
    fn test_probe_nexus_histogram_units_us_no_rescale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe_units_us.h5");

        let counts = vec![0u64; 4];
        // Values that would be wrong by 1000× if treated as ns.
        let tof_us = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0];
        create_test_histogram_with_units(&path, &counts, [1, 1, 1, 4], &tof_us, Some("us"));

        let meta = probe_nexus(&path).expect("probe with units=us");
        let edges = meta.tof_edges_us.expect("TOF axis should be present");
        assert_eq!(edges.len(), 5);
        for (i, &expected_us) in tof_us.iter().enumerate() {
            assert!(
                (edges[i] - expected_us).abs() < 1e-9,
                "probe edge {i}: expected {expected_us} µs (no rescale), got {} µs",
                edges[i]
            );
        }
    }

    #[test]
    fn test_load_nexus_histogram_single_angle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");

        // 1 rot angle, 2x3 spatial, 2 TOF bins
        let mut counts = vec![0u64; 2 * 3 * 2];
        counts[0] = 15; // rot=0, y=0, x=0, tof=0

        let tof_ns = vec![1000.0, 2000.0, 3000.0]; // 3 edges for 2 bins
        create_test_histogram(&path, &counts, [1, 2, 3, 2], &tof_ns, Some(25.0));

        let data = load_nexus_histogram(&path).unwrap();

        // Shape should be (n_tof=2, n_y=2, n_x=3) after transposing
        assert_eq!(data.counts.shape(), &[2, 2, 3]);
        // Single angle: value is preserved exactly
        assert_eq!(data.counts[[0, 0, 0]], 15.0);

        // TOF edges converted ns → µs
        assert_eq!(data.tof_edges_us.len(), 3);
        assert!((data.tof_edges_us[0] - 1.0).abs() < 1e-10);
        assert!((data.tof_edges_us[1] - 2.0).abs() < 1e-10);
        assert!((data.tof_edges_us[2] - 3.0).abs() < 1e-10);
        assert_eq!(data.flight_path_m, Some(25.0));
        assert_eq!(data.n_rotation_angles, 1);
    }

    /// Issue #430: default `load_nexus_histogram` must refuse multi-angle
    /// files rather than silently collapse the rotation dimension.
    #[test]
    fn test_load_nexus_histogram_multi_angle_errors_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_angle.h5");

        let counts = vec![1u64; 2 * 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [2, 2, 3, 2], &tof_ns, Some(25.0));

        let err = load_nexus_histogram(&path)
            .expect_err("multi-angle file must be rejected by the default loader");
        let msg = err.to_string();
        assert!(
            msg.contains("2 rotation angles") && msg.contains("#430"),
            "error message should name the angle count and reference #430, got: {msg}"
        );
        assert!(
            msg.contains("MultiAngleMode::Sum") && msg.contains("MultiAngleMode::SelectAngle"),
            "error message should point at the explicit-opt-in APIs, got: {msg}"
        );
    }

    /// Issue #430: `MultiAngleMode::Sum` is the explicit opt-in for the
    /// legacy auto-sum behaviour.  Recovers the pre-#430 output exactly.
    #[test]
    fn test_load_nexus_histogram_multi_angle_sum_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_angle_sum.h5");

        let mut counts = vec![0u64; 2 * 2 * 3 * 2];
        counts[0] = 10; // rot=0, y=0, x=0, tof=0
        counts[12] = 5; // rot=1, y=0, x=0, tof=0
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [2, 2, 3, 2], &tof_ns, Some(25.0));

        let data = load_nexus_histogram_with_mode(&path, MultiAngleMode::Sum).unwrap();
        assert_eq!(data.counts.shape(), &[2, 2, 3]);
        // Summed: 10 + 5 = 15
        assert_eq!(data.counts[[0, 0, 0]], 15.0);
        assert_eq!(data.n_rotation_angles, 2);
    }

    /// Issue #430: `MultiAngleMode::SelectAngle(i)` extracts a single
    /// projection by index, leaving the other angles' data unread.
    #[test]
    fn test_load_nexus_histogram_multi_angle_select_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_angle_select.h5");

        let mut counts = vec![0u64; 3 * 2 * 3 * 2];
        counts[0] = 100; // rot=0, y=0, x=0, tof=0
        counts[12] = 200; // rot=1, y=0, x=0, tof=0
        counts[24] = 300; // rot=2, y=0, x=0, tof=0
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [3, 2, 3, 2], &tof_ns, Some(25.0));

        // Select angle 1 — should see 200, not 100 / 300 / 600.
        let data = load_nexus_histogram_with_mode(&path, MultiAngleMode::SelectAngle(1)).unwrap();
        assert_eq!(data.counts[[0, 0, 0]], 200.0);
        assert_eq!(data.n_rotation_angles, 3);

        // Out-of-range index → error.
        let err = load_nexus_histogram_with_mode(&path, MultiAngleMode::SelectAngle(3))
            .expect_err("out-of-range angle index must error");
        let msg = err.to_string();
        assert!(
            msg.contains("SelectAngle(3)") && msg.contains("3 rotation angle"),
            "error should name the bad index and the actual count, got: {msg}"
        );
    }

    /// `MultiAngleMode::Error` on a single-angle file is a no-op:
    /// `n_rot == 1` is the trivial non-collapsing case.  All three
    /// modes must produce identical output here.
    #[test]
    fn test_load_nexus_histogram_single_angle_mode_parity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single_parity.h5");
        let counts = vec![7u64; 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [1, 2, 3, 2], &tof_ns, None);

        let d_err = load_nexus_histogram_with_mode(&path, MultiAngleMode::Error).unwrap();
        let d_sum = load_nexus_histogram_with_mode(&path, MultiAngleMode::Sum).unwrap();
        let d_sel = load_nexus_histogram_with_mode(&path, MultiAngleMode::SelectAngle(0)).unwrap();
        // All three modes produce the same output on a single-angle file.
        assert_eq!(d_err.counts, d_sum.counts);
        assert_eq!(d_err.counts, d_sel.counts);
        // Value preserved (not doubled — single angle).
        assert_eq!(d_err.counts[[0, 0, 0]], 7.0);
        assert_eq!(d_err.n_rotation_angles, 1);
    }

    /// A zero-angle file (degenerate, `shape[0] == 0`) must be
    /// rejected on every mode — otherwise `Sum` would produce an
    /// all-zero output indistinguishable from a valid but dark
    /// measurement, and `Error` would silently accept the degenerate
    /// file.
    #[test]
    fn test_load_nexus_histogram_zero_angles_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zero_angles.h5");
        let counts: Vec<u64> = Vec::new();
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [0, 2, 3, 2], &tof_ns, None);

        for mode in [
            MultiAngleMode::Error,
            MultiAngleMode::Sum,
            MultiAngleMode::SelectAngle(0),
        ] {
            let err = load_nexus_histogram_with_mode(&path, mode).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("zero rotation angles"),
                "mode {mode:?} zero-angle rejection should name the axis, got: {msg}"
            );
        }
    }

    /// A zero-sized y / x / tof axis is just as degenerate as a zero-angle
    /// axis and must be rejected the same way, rather than producing an empty
    /// detector plane / energy series downstream.
    #[test]
    fn test_load_nexus_histogram_zero_sibling_axes_rejected() {
        for (shape, axis_name) in [
            ([1usize, 0, 3, 2], "y"),
            ([1, 2, 0, 2], "x"),
            ([1, 2, 3, 0], "tof"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(format!("zero_{axis_name}.h5"));
            let counts: Vec<u64> = Vec::new(); // any axis is 0 → empty cube
            let tof_ns = vec![1000.0, 2000.0, 3000.0];
            create_test_histogram(&path, &counts, shape, &tof_ns, None);

            let err = load_nexus_histogram(&path).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("zero-sized {axis_name} axis")),
                "axis {axis_name} ({shape:?}) should be rejected by name, got: {msg}"
            );
        }
    }

    /// The NeXus TOF axis must be strictly monotonic and positive, mirroring
    /// the spectrum-file path.  A non-increasing or non-positive axis would
    /// silently feed `tof_to_energy` a bad value downstream.
    #[test]
    fn test_load_nexus_histogram_rejects_non_monotonic_tof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonmono_tof.h5");
        // 1 angle, 1×1 spatial, 2 TOF bins → 3 edges, but they decrease.
        let counts = vec![1u64, 2u64];
        let tof_ns = vec![3000.0, 2000.0, 1000.0];
        create_test_histogram(&path, &counts, [1, 1, 1, 2], &tof_ns, None);

        let err = load_nexus_histogram(&path).unwrap_err();
        assert!(
            err.to_string().contains("strictly increasing"),
            "non-monotonic TOF should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_load_nexus_histogram_rejects_non_positive_tof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonpos_tof.h5");
        // First edge is zero → non-positive TOF axis.
        let counts = vec![1u64, 2u64];
        let tof_ns = vec![0.0, 1000.0, 2000.0];
        create_test_histogram(&path, &counts, [1, 1, 1, 2], &tof_ns, None);

        let err = load_nexus_histogram(&path).unwrap_err();
        assert!(
            err.to_string().contains("finite and positive"),
            "non-positive TOF should be rejected, got: {err}"
        );
    }

    /// A trailing `+∞` TOF edge satisfies `prev < +∞` so it passes a
    /// monotonicity-only check, but it is not a real time — the per-edge
    /// `is_finite()` guard must reject it.
    #[test]
    fn test_load_nexus_histogram_rejects_trailing_infinite_tof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("inf_tail_tof.h5");
        // 1 angle, 1×1 spatial, 2 TOF bins → 3 edges, last is +∞.
        let counts = vec![1u64, 2u64];
        let tof_ns = vec![1000.0, 2000.0, f64::INFINITY];
        create_test_histogram(&path, &counts, [1, 1, 1, 2], &tof_ns, None);

        let err = load_nexus_histogram(&path).unwrap_err();
        assert!(
            err.to_string().contains("finite and positive"),
            "trailing +inf TOF edge should be rejected, got: {err}"
        );
    }

    /// A single-bin axis has only 2 edges; if a malformed scale yields a
    /// degenerate axis the per-edge guard still fires.  A 1-edge axis never
    /// enters `windows(2)` at all, so `validate_monotonic` is vacuously OK —
    /// the per-edge `is_finite() && > 0` check is the only thing that rejects
    /// a lone `NaN` / `+∞` edge.  Exercise `read_tof_axis` directly so the
    /// single-edge case is reachable without tripping the bin-count
    /// cross-check in `load_nexus_histogram`.
    #[test]
    fn test_read_tof_axis_rejects_single_nan_or_inf_edge() {
        let dir = tempfile::tempdir().unwrap();

        for (name, edge) in [("nan", f64::NAN), ("inf", f64::INFINITY)] {
            let path = dir.path().join(format!("single_{name}_edge.h5"));
            let file = hdf5::File::create(&path).expect("create");
            let entry = file.create_group("entry").expect("entry");
            let hist = entry.create_group("histogram").expect("histogram");
            hist.new_dataset::<f64>()
                .shape([1])
                .create("time_of_flight")
                .expect("create tof")
                .write_raw(&[edge])
                .expect("write tof");
            // No `units` attr → legacy-ns scale (finite, so the bad edge is
            // preserved as bad, not normalised away).
            drop(file);

            let file = hdf5::File::open(&path).expect("reopen");
            let hist_group = file
                .group("entry")
                .expect("entry")
                .group("histogram")
                .expect("histogram");
            let err = read_tof_axis(&hist_group).expect_err("single bad edge must reject");
            assert!(
                err.to_string().contains("finite and positive"),
                "single {name} edge should be rejected, got: {err}"
            );
        }
    }

    /// Create a histogram fixture that also carries a `/entry/pixel_masks/dead`
    /// mask of the given shape, for dead-mask shape-validation tests.
    fn create_test_histogram_with_dead_mask(
        path: &Path,
        counts: &[u64],
        shape: [usize; 4],
        tof_ns: &[f64],
        dead: &[u8],
        dead_shape: [usize; 2],
    ) {
        create_test_histogram(path, counts, shape, tof_ns, None);
        let file = hdf5::File::append(path).expect("reopen test file");
        let entry = file.group("entry").expect("entry");
        let masks = entry.create_group("pixel_masks").expect("pixel_masks");
        masks
            .new_dataset::<u8>()
            .shape(dead_shape)
            .create("dead")
            .expect("create dead")
            .write_raw(dead)
            .expect("write dead");
    }

    /// A dead-pixel mask whose shape does not match the detector's spatial
    /// dimensions must be rejected — applying it would mask the wrong pixels.
    #[test]
    fn test_load_nexus_histogram_rejects_mismatched_dead_mask() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_mask.h5");
        // counts shape [1, 2, 3, 2] → detector is 2×3; write a 5×5 mask.
        let counts = vec![1u64; 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        let dead = vec![0u8; 25];
        create_test_histogram_with_dead_mask(&path, &counts, [1, 2, 3, 2], &tof_ns, &dead, [5, 5]);

        let err = load_nexus_histogram(&path).unwrap_err();
        assert!(
            matches!(err, IoError::ShapeMismatch(_)),
            "expected ShapeMismatch, got {err:?}"
        );
        assert!(err.to_string().contains("dead-pixel mask shape"));
    }

    /// A correctly-shaped dead-pixel mask still loads (no false rejection).
    #[test]
    fn test_load_nexus_histogram_accepts_matching_dead_mask() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok_mask.h5");
        let counts = vec![1u64; 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        // 2×3 mask matching the detector; mark one pixel (row 0, col 1) dead.
        let dead = vec![0u8, 1, 0, 0, 0, 0];
        create_test_histogram_with_dead_mask(&path, &counts, [1, 2, 3, 2], &tof_ns, &dead, [2, 3]);

        let data = load_nexus_histogram(&path).expect("matching mask should load");
        let mask = data.dead_pixels.expect("mask present");
        assert_eq!(mask.dim(), (2, 3));
        assert!(mask[[0, 1]]);
    }

    /// A file with *no* `pixel_masks` group is valid: the mask is absent, not
    /// malformed, so the load succeeds with `dead_pixels == None`.
    #[test]
    fn test_load_nexus_histogram_absent_dead_mask_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_mask.h5");
        let counts = vec![1u64; 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [1, 2, 3, 2], &tof_ns, None);

        let data = load_nexus_histogram(&path).expect("absent mask should load");
        assert!(
            data.dead_pixels.is_none(),
            "absent dead mask must map to None"
        );
    }

    /// A `/entry/pixel_masks/dead` link that exists but is the wrong object
    /// kind (a group, not a dataset) is *present-but-malformed*: it must be
    /// surfaced as an error, not silently swallowed as absence (which would
    /// drop a real-but-corrupt mask and mask no pixels).
    #[test]
    fn test_load_nexus_histogram_rejects_present_but_invalid_dead_mask() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid_mask.h5");
        let counts = vec![1u64; 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [1, 2, 3, 2], &tof_ns, None);

        // Write `dead` as a *group*, not a dataset — present but malformed.
        let file = hdf5::File::append(&path).expect("reopen");
        let entry = file.group("entry").expect("entry");
        let masks = entry.create_group("pixel_masks").expect("pixel_masks");
        masks.create_group("dead").expect("dead-as-group");
        drop(file);

        let err = load_nexus_histogram(&path).unwrap_err();
        assert!(
            matches!(err, IoError::InvalidParameter(_)),
            "present-but-malformed dead mask must be InvalidParameter, got {err:?}"
        );
        assert!(
            err.to_string().contains("dead") && err.to_string().contains("not a readable dataset"),
            "error should identify the malformed dead dataset, got: {err}"
        );
    }

    /// `MultiAngleMode::Error` must reject multi-angle
    /// files BEFORE reading the full 4D counts dataset.  On a real
    /// multi-angle file this dataset can be multi-GB; wasting a read
    /// to then error out is prohibitive.  This test uses metadata
    /// (shape is 4D, n_rot > 1) from a tiny synthetic fixture to
    /// assert the error is returned — the underlying file is
    /// small here, but the code-path assertion is that rejection
    /// happens via the shape check alone.  (We can't assert "no
    /// read happened" directly without hooking HDF5, but the
    /// structural guarantee is preserved by the order of
    /// statements in `load_nexus_histogram_with_mode`.)
    #[test]
    fn test_multi_angle_rejection_happens_before_counts_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big_shape.h5");
        // Small synthetic file, but with shape[0]=4 so we exercise the
        // rejection path.
        let counts = vec![1u64; 4 * 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [4, 2, 3, 2], &tof_ns, None);

        let err = load_nexus_histogram_with_mode(&path, MultiAngleMode::Error).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("4 rotation angles") && msg.contains("#430"),
            "error message should name angle count + reference the issue, got: {msg}"
        );
    }

    #[test]
    fn test_ns_to_us_conversion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");

        let counts = vec![0u64; 3];
        let tof_ns = vec![500_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0];
        create_test_histogram(&path, &counts, [1, 1, 1, 3], &tof_ns, None);

        let data = load_nexus_histogram(&path).unwrap();

        // 500_000 ns = 500 µs, etc.
        assert!((data.tof_edges_us[0] - 500.0).abs() < 1e-10);
        assert!((data.tof_edges_us[1] - 1000.0).abs() < 1e-10);
        assert!((data.tof_edges_us[2] - 1500.0).abs() < 1e-10);
        assert!((data.tof_edges_us[3] - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn test_probe_missing_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.h5");

        let file = hdf5::File::create(&path).expect("create");
        file.create_group("entry").expect("create entry");
        drop(file);

        let meta = probe_nexus(&path).unwrap();
        assert!(!meta.has_histogram);
        assert!(!meta.has_events);
        assert!(meta.histogram_shape.is_none());
        assert!(meta.n_events.is_none());
    }

    /// Create a minimal NeXus file with neutron event data.
    fn create_test_events(
        path: &Path,
        tof_ns: &[u64],
        x: &[f64],
        y: &[f64],
        flight_path_m: Option<f64>,
    ) {
        let file = hdf5::File::create(path).expect("create");
        let entry = file.create_group("entry").expect("create entry");

        if let Some(fp) = flight_path_m {
            entry
                .new_attr::<f64>()
                .shape(())
                .create("flight_path_m")
                .expect("create attr")
                .write_scalar(&fp)
                .expect("write attr");
        }

        let neutrons = entry.create_group("neutrons").expect("create neutrons");
        neutrons
            .new_dataset::<u64>()
            .shape([tof_ns.len()])
            .create("event_time_offset")
            .expect("create tof")
            .write_raw(tof_ns)
            .expect("write tof");
        neutrons
            .new_dataset::<f64>()
            .shape([x.len()])
            .create("x")
            .expect("create x")
            .write_raw(x)
            .expect("write x");
        neutrons
            .new_dataset::<f64>()
            .shape([y.len()])
            .create("y")
            .expect("create y")
            .write_raw(y)
            .expect("write y");
    }

    #[test]
    fn test_histogram_known_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.h5");

        // 3 events: all at pixel (1, 0), TOFs at 1500 µs, 2500 µs, 1800 µs (in ns)
        let tof_ns = vec![1_500_000, 2_500_000, 1_800_000];
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 0.0];
        create_test_events(&path, &tof_ns, &x, &y, Some(25.0));

        let params = EventBinningParams {
            n_bins: 2,
            tof_min_us: 1000.0,
            tof_max_us: 3000.0,
            height: 2,
            width: 3,
        };

        let data = load_nexus_events(&path, &params).unwrap();
        assert_eq!(data.counts.shape(), &[2, 2, 3]);

        // Bin 0: TOF [1000, 2000) µs → events at 1500 and 1800 µs → 2 counts
        assert_eq!(data.counts[[0, 0, 1]], 2.0);
        // Bin 1: TOF [2000, 3000) µs → event at 2500 µs → 1 count
        assert_eq!(data.counts[[1, 0, 1]], 1.0);

        assert_eq!(data.flight_path_m, Some(25.0));
        assert_eq!(data.tof_edges_us.len(), 3); // n_bins + 1 edges

        // All 3 events kept, none dropped
        let stats = data
            .event_stats
            .as_ref()
            .expect("event_stats should be Some");
        assert_eq!(stats.total, 3);
        assert_eq!(stats.kept, 3);
        assert_eq!(stats.dropped_non_finite, 0);
        assert_eq!(stats.dropped_tof_range, 0);
        assert_eq!(stats.dropped_spatial, 0);
    }

    #[test]
    fn test_filter_out_of_range_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events_oob.h5");

        // Events: one in range, one out of TOF range, one out of spatial range
        let tof_ns = vec![
            1_500_000, // in range
            500_000,   // below tof_min
            1_500_000, // in range but x out of bounds
        ];
        let x = vec![0.0, 0.0, 5.0]; // 5.0 is out of width=3
        let y = vec![0.0, 0.0, 0.0];
        create_test_events(&path, &tof_ns, &x, &y, None);

        let params = EventBinningParams {
            n_bins: 2,
            tof_min_us: 1000.0,
            tof_max_us: 3000.0,
            height: 2,
            width: 3,
        };

        let data = load_nexus_events(&path, &params).unwrap();

        // Only 1 event should be counted (the first one)
        let total: f64 = data.counts.iter().sum();
        assert_eq!(total, 1.0);
        assert_eq!(data.counts[[0, 0, 0]], 1.0);

        // 1 kept, 1 dropped by TOF range, 1 dropped by spatial bounds
        let stats = data
            .event_stats
            .as_ref()
            .expect("event_stats should be Some");
        assert_eq!(stats.total, 3);
        assert_eq!(stats.kept, 1);
        assert_eq!(stats.dropped_non_finite, 0);
        assert_eq!(stats.dropped_tof_range, 1);
        assert_eq!(stats.dropped_spatial, 1);
    }

    #[test]
    fn test_empty_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_events.h5");

        create_test_events(&path, &[], &[], &[], None);

        let params = EventBinningParams {
            n_bins: 10,
            tof_min_us: 1000.0,
            tof_max_us: 20000.0,
            height: 4,
            width: 4,
        };

        let data = load_nexus_events(&path, &params).unwrap();
        assert_eq!(data.counts.shape(), &[10, 4, 4]);

        let total: f64 = data.counts.iter().sum();
        assert_eq!(total, 0.0);

        // Zero events in, zero events out
        let stats = data
            .event_stats
            .as_ref()
            .expect("event_stats should be Some");
        assert_eq!(stats.total, 0);
        assert_eq!(stats.kept, 0);
        assert_eq!(stats.dropped_non_finite, 0);
        assert_eq!(stats.dropped_tof_range, 0);
        assert_eq!(stats.dropped_spatial, 0);
    }

    #[test]
    fn test_probe_with_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("with_events.h5");

        create_test_events(
            &path,
            &[1000, 2000, 3000],
            &[0.0, 1.0, 2.0],
            &[0.0, 0.0, 1.0],
            None,
        );

        let meta = probe_nexus(&path).unwrap();
        assert!(!meta.has_histogram);
        assert!(meta.has_events);
        assert_eq!(meta.n_events, Some(3));
    }

    #[test]
    fn test_list_hdf5_tree() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tree.h5");

        // Create a file with nested groups and a dataset
        {
            let file = hdf5::File::create(&path).expect("create file");
            let g1 = file.create_group("entry").expect("create entry");
            let g2 = g1.create_group("histogram").expect("create histogram");
            g2.new_dataset::<f64>()
                .shape([3])
                .create("data")
                .expect("create data")
                .write_raw(&[1.0, 2.0, 3.0])
                .expect("write data");
        }

        let tree = list_hdf5_tree(&path, 10).unwrap();
        assert!(!tree.is_empty());

        // Check that we find the expected paths
        let paths: Vec<&str> = tree.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"/entry"));
        assert!(paths.contains(&"/entry/histogram"));
        assert!(paths.contains(&"/entry/histogram/data"));

        // The dataset should have a shape
        let data_entry = tree
            .iter()
            .find(|e| e.path == "/entry/histogram/data")
            .unwrap();
        assert!(data_entry.shape.is_some());
    }

    #[test]
    fn test_nan_xy_coords_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan_xy.h5");

        // 4 events: 1 good, 1 NaN x, 1 Inf y, 1 good
        let tof_ns = vec![1_500_000, 1_500_000, 1_500_000, 2_500_000];
        let x = vec![0.0, f64::NAN, 0.0, 1.0];
        let y = vec![0.0, 0.0, f64::INFINITY, 0.0];
        create_test_events(&path, &tof_ns, &x, &y, None);

        let params = EventBinningParams {
            n_bins: 2,
            tof_min_us: 1000.0,
            tof_max_us: 3000.0,
            height: 2,
            width: 3,
        };

        let data = load_nexus_events(&path, &params).unwrap();

        // Only 2 good events should be counted
        let total_counts: f64 = data.counts.iter().sum();
        assert_eq!(total_counts, 2.0);

        let stats = data
            .event_stats
            .as_ref()
            .expect("event_stats should be Some");
        assert_eq!(stats.total, 4);
        assert_eq!(stats.kept, 2);
        assert_eq!(stats.dropped_non_finite, 2);
        assert_eq!(stats.dropped_tof_range, 0);
        assert_eq!(stats.dropped_spatial, 0);
    }

    // -----------------------------------------------------------------
    // Issue #554 — NeXus `units` attribute on TOF datasets must be
    // honoured.  A file written with `units = "us"` was previously
    // divided by 1000 silently, shifting the energy axis by 1000×.
    // -----------------------------------------------------------------

    /// Write a scalar string attribute on an HDF5 dataset.  Tests use
    /// this to inject `units = "ns"`, `units = "us"`, etc. on the
    /// `time_of_flight` / `event_time_offset` datasets.
    fn write_units_attr(ds: &hdf5::Dataset, units: &str) {
        let val: VarLenUnicode = units.parse().expect("parse units string");
        ds.new_attr::<VarLenUnicode>()
            .shape(())
            .create("units")
            .expect("create units attr")
            .write_scalar(&val)
            .expect("write units attr");
    }

    /// Variant of `create_test_histogram` that stamps a `units`
    /// attribute on the `time_of_flight` dataset.
    fn create_test_histogram_with_units(
        path: &Path,
        counts: &[u64],
        shape: [usize; 4],
        tof_values: &[f64],
        units: Option<&str>,
    ) {
        let file = hdf5::File::create(path).expect("create test file");
        let entry = file.create_group("entry").expect("create entry");
        let hist = entry.create_group("histogram").expect("create histogram");
        hist.new_dataset::<u64>()
            .shape(shape)
            .create("counts")
            .expect("create counts")
            .write_raw(counts)
            .expect("write counts");
        let tof_ds = hist
            .new_dataset::<f64>()
            .shape([tof_values.len()])
            .create("time_of_flight")
            .expect("create tof");
        tof_ds.write_raw(tof_values).expect("write tof");
        if let Some(u) = units {
            write_units_attr(&tof_ds, u);
        }
    }

    /// Variant of `create_test_events` that stamps a `units` attribute
    /// on the `event_time_offset` dataset.
    fn create_test_events_with_units(
        path: &Path,
        tof_values: &[u64],
        x: &[f64],
        y: &[f64],
        units: Option<&str>,
    ) {
        let file = hdf5::File::create(path).expect("create");
        let entry = file.create_group("entry").expect("create entry");
        let neutrons = entry.create_group("neutrons").expect("create neutrons");
        let tof_ds = neutrons
            .new_dataset::<u64>()
            .shape([tof_values.len()])
            .create("event_time_offset")
            .expect("create tof");
        tof_ds.write_raw(tof_values).expect("write tof");
        if let Some(u) = units {
            write_units_attr(&tof_ds, u);
        }
        neutrons
            .new_dataset::<f64>()
            .shape([x.len()])
            .create("x")
            .expect("create x")
            .write_raw(x)
            .expect("write x");
        neutrons
            .new_dataset::<f64>()
            .shape([y.len()])
            .create("y")
            .expect("create y")
            .write_raw(y)
            .expect("write y");
    }

    /// `tof_scale_to_us` table: each recognised spelling maps to the
    /// documented multiplier.  Pure-helper test — exercises the lookup
    /// without an HDF5 file.
    #[test]
    fn test_tof_scale_to_us_table() {
        // Missing → legacy ns assumption.
        assert!((tof_scale_to_us(None).unwrap() - 1e-3).abs() < 1e-15);
        for (spelling, expected) in &[
            ("ns", 1e-3),
            ("Ns", 1e-3),
            ("NS", 1e-3),
            ("nanoseconds", 1e-3),
            ("us", 1.0),
            ("US", 1.0),
            ("microseconds", 1.0),
            ("µs", 1.0),
            ("ms", 1e3),
            ("milliseconds", 1e3),
            ("s", 1e6),
            ("seconds", 1e6),
            ("  s  ", 1e6),
        ] {
            let got = tof_scale_to_us(Some(*spelling))
                .unwrap_or_else(|e| panic!("spelling {spelling:?} unexpectedly errored: {e}"));
            assert!(
                (got - expected).abs() < 1e-15,
                "spelling {spelling:?}: expected scale {expected}, got {got}"
            );
        }
        // Unknown units must error — no silent fallback.
        for bad in &["picoseconds", "ticks", "us per channel", "", "garbage"] {
            let err = tof_scale_to_us(Some(*bad)).expect_err("unknown units must error");
            let msg = err.to_string();
            assert!(
                msg.contains("Unsupported NeXus TOF units"),
                "error for {bad:?} should mention 'Unsupported NeXus TOF units', got: {msg}"
            );
        }
    }

    /// Histogram path with `units = "ns"` (current canonical
    /// assumption): explicit ns annotation must produce the same µs
    /// edges as the rustpix-legacy "no attribute, assume ns" path.
    #[test]
    fn test_load_nexus_histogram_units_ns_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hist_units_ns.h5");
        let counts = vec![0u64; 2];
        // 3 ns edges → 0.001, 0.002, 0.003 µs
        let tof_ns = vec![1.0, 2.0, 3.0];
        create_test_histogram_with_units(&path, &counts, [1, 1, 1, 2], &tof_ns, Some("ns"));

        let data = load_nexus_histogram(&path).expect("load with units=ns");
        assert_eq!(data.tof_edges_us.len(), 3);
        assert!((data.tof_edges_us[0] - 0.001).abs() < 1e-12);
        assert!((data.tof_edges_us[1] - 0.002).abs() < 1e-12);
        assert!((data.tof_edges_us[2] - 0.003).abs() < 1e-12);
    }

    /// Histogram path with `units = "us"` (NeXus-standard
    /// microseconds): values must be passed through unchanged, NOT
    /// divided by 1000.  Pre-#554 this produced a 1000× too-small
    /// energy axis silently.
    #[test]
    fn test_load_nexus_histogram_units_us_no_rescale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hist_units_us.h5");
        let counts = vec![0u64; 4];
        // µs values that would be catastrophically wrong if divided by 1000.
        let tof_us = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0];
        create_test_histogram_with_units(&path, &counts, [1, 1, 1, 4], &tof_us, Some("us"));

        let data = load_nexus_histogram(&path).expect("load with units=us");
        assert_eq!(data.tof_edges_us.len(), 5);
        for (i, &expected) in tof_us.iter().enumerate() {
            assert!(
                (data.tof_edges_us[i] - expected).abs() < 1e-9,
                "edge {i}: expected {expected} µs (no rescale), got {} µs",
                data.tof_edges_us[i]
            );
        }
    }

    /// Histogram path: NeXus `units = "s"` (seconds) must rescale
    /// ×1e6 → µs.
    #[test]
    fn test_load_nexus_histogram_units_seconds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hist_units_s.h5");
        let counts = vec![0u64; 2];
        // 1 ms = 1000 µs, written as 0.001 s
        let tof_s = vec![0.001, 0.002, 0.003];
        create_test_histogram_with_units(&path, &counts, [1, 1, 1, 2], &tof_s, Some("s"));

        let data = load_nexus_histogram(&path).expect("load with units=s");
        assert!((data.tof_edges_us[0] - 1000.0).abs() < 1e-9);
        assert!((data.tof_edges_us[1] - 2000.0).abs() < 1e-9);
        assert!((data.tof_edges_us[2] - 3000.0).abs() < 1e-9);
    }

    /// Histogram path: unknown `units` must hard-error rather than
    /// silently default to ns.  Without this check, a typo
    /// (e.g. `"microsecond"` vs `"microseconds"`) could be mis-scaled
    /// — and worse, an exotic-but-real unit (`"ticks"`) would be
    /// silently dropped on the floor.
    #[test]
    fn test_load_nexus_histogram_units_unknown_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hist_units_bad.h5");
        let counts = vec![0u64; 2];
        let tof = vec![1.0, 2.0, 3.0];
        create_test_histogram_with_units(&path, &counts, [1, 1, 1, 2], &tof, Some("picoseconds"));

        let err = load_nexus_histogram(&path).expect_err("unknown units must error");
        let msg = err.to_string();
        assert!(
            msg.contains("Unsupported NeXus TOF units") && msg.contains("picoseconds"),
            "error should name the offending value, got: {msg}"
        );
    }

    /// Histogram path: missing `units` attribute is allowed and
    /// preserves the rustpix-legacy ns assumption.  This is the
    /// backward-compatibility guarantee for files produced by the
    /// rustpix-era extraction tooling, which writes nanoseconds
    /// without a `units` attribute.
    #[test]
    fn test_load_nexus_histogram_units_missing_legacy_ns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hist_units_missing.h5");
        let counts = vec![0u64; 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        // No `units` attribute → assume ns → 1.0 / 2.0 / 3.0 µs.
        create_test_histogram_with_units(&path, &counts, [1, 1, 1, 2], &tof_ns, None);
        let data = load_nexus_histogram(&path).expect("load with no units attr");
        assert!((data.tof_edges_us[0] - 1.0).abs() < 1e-12);
        assert!((data.tof_edges_us[1] - 2.0).abs() < 1e-12);
        assert!((data.tof_edges_us[2] - 3.0).abs() < 1e-12);
    }

    /// Events path with `units = "us"`: the TOF binning must place
    /// events at the correct µs values, NOT divide them by 1000.
    /// Pre-#554 a file written in µs would have all events land
    /// 1000× lower than the user-specified bins, leaving the
    /// histogram empty / all dropped by `tof_range`.
    #[test]
    fn test_load_nexus_events_units_us_no_rescale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events_units_us.h5");

        // Events at 1500 µs and 2500 µs (written in µs, units=us).
        let tof_us = vec![1500u64, 2500u64, 1800u64];
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 0.0];
        create_test_events_with_units(&path, &tof_us, &x, &y, Some("us"));

        let params = EventBinningParams {
            n_bins: 2,
            tof_min_us: 1000.0,
            tof_max_us: 3000.0,
            height: 2,
            width: 3,
        };
        let data = load_nexus_events(&path, &params).expect("load events with units=us");

        // Bin 0: [1000, 2000) µs → 1500 + 1800 = 2 events
        assert_eq!(data.counts[[0, 0, 1]], 2.0);
        // Bin 1: [2000, 3000) µs → 2500 = 1 event
        assert_eq!(data.counts[[1, 0, 1]], 1.0);
        let stats = data.event_stats.as_ref().expect("event stats");
        assert_eq!(stats.kept, 3);
        assert_eq!(stats.dropped_tof_range, 0);
    }

    /// Events path with `units = "ns"` (explicit): same result as
    /// the legacy "no attribute" path.
    #[test]
    fn test_load_nexus_events_units_ns_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events_units_ns.h5");

        // Events at 1500/2500/1800 µs, written in ns with units=ns.
        let tof_ns = vec![1_500_000u64, 2_500_000u64, 1_800_000u64];
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 0.0];
        create_test_events_with_units(&path, &tof_ns, &x, &y, Some("ns"));

        let params = EventBinningParams {
            n_bins: 2,
            tof_min_us: 1000.0,
            tof_max_us: 3000.0,
            height: 2,
            width: 3,
        };
        let data = load_nexus_events(&path, &params).expect("load events with units=ns");
        assert_eq!(data.counts[[0, 0, 1]], 2.0);
        assert_eq!(data.counts[[1, 0, 1]], 1.0);
    }

    /// Events path: unknown `units` must hard-error.
    #[test]
    fn test_load_nexus_events_units_unknown_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events_units_bad.h5");
        let tof = vec![1_500_000u64];
        let x = vec![0.0];
        let y = vec![0.0];
        create_test_events_with_units(&path, &tof, &x, &y, Some("clock-ticks"));

        let params = EventBinningParams {
            n_bins: 2,
            tof_min_us: 1000.0,
            tof_max_us: 3000.0,
            height: 2,
            width: 3,
        };
        let err = load_nexus_events(&path, &params).expect_err("unknown units must error");
        let msg = err.to_string();
        assert!(
            msg.contains("Unsupported NeXus TOF units") && msg.contains("clock-ticks"),
            "error should name the offending value, got: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// NXevent_data bank spectra with wall-clock interval filtering (issue #637)
// ---------------------------------------------------------------------------

/// TOF binning parameters for a 1-D NXevent_data bank spectrum (issue #637).
///
/// NXevent_data banks (facility NeXus convention: `/entry/<bank>/` with
/// `event_time_offset`, `event_index`, `event_time_zero`) have no per-event
/// pixel coordinates in the general case (monitors never do), so the result
/// is a 1-D TOF spectrum rather than a `(tof, y, x)` cube.
#[derive(Debug, Clone, Copy)]
pub struct BankBinningParams {
    /// Number of TOF bins.
    pub n_bins: usize,
    /// Minimum TOF in microseconds (inclusive).
    pub tof_min_us: f64,
    /// Maximum TOF in microseconds (exclusive).
    pub tof_max_us: f64,
}

/// Result of [`load_nexus_bank_spectrum`]: a 1-D TOF spectrum plus pulse and
/// event retention statistics.
///
/// The drop counters (`dropped_tof_range`, `dropped_non_finite`) cover only
/// events belonging to **kept** pulses; events on pulses excluded by
/// `keep_intervals` are accounted for as `events_total - events_kept -
/// dropped_tof_range - dropped_non_finite` and are not itemised.
#[derive(Debug, Clone)]
pub struct BankSpectrum {
    /// TOF bin edges in microseconds (`n_bins + 1` values, linear grid).
    pub tof_edges_us: Vec<f64>,
    /// Histogrammed event counts per TOF bin (`n_bins` values).
    pub counts: Vec<u64>,
    /// Total number of pulses recorded in the bank.
    pub pulses_total: usize,
    /// Pulses whose `event_time_zero` fell inside `keep_intervals`
    /// (equals `pulses_total` when no filter was given).
    pub pulses_kept: usize,
    /// Total number of events recorded in the bank.
    pub events_total: usize,
    /// Events on kept pulses that landed inside the TOF window.
    pub events_kept: usize,
    /// Events on kept pulses dropped for TOF outside `[tof_min_us, tof_max_us)`.
    pub dropped_tof_range: usize,
    /// Events on kept pulses dropped for non-finite TOF.
    pub dropped_non_finite: usize,
    /// ISO-8601 `offset` attribute of `event_time_zero`, when recorded —
    /// the absolute wall-clock epoch that pulse times are relative to.
    /// Compare with [`crate::runlog::RunLog::offset_iso`] to confirm that
    /// interval and pulse clocks share a zero point (at SNS both are
    /// seconds since run start and the attributes match exactly).
    pub pulse_time_offset_iso: Option<String>,
}

/// Load one NXevent_data bank (e.g. a beam monitor) as a 1-D TOF spectrum,
/// optionally keeping only pulses inside wall-clock `keep_intervals`
/// (issue #637).
///
/// Reads `/entry/<bank>/{event_time_offset, event_index, event_time_zero}`:
///
/// - `event_time_offset` — TOF per event; its `units` attribute is
///   **required** on this path (facility files always write it; refusing to
///   guess closes the #554 silent-rescale class).  Recognised values are
///   the module-level table (ns/us/ms/s), scaled to canonical µs.
/// - `event_index` — cumulative first-event index per pulse (validated
///   non-decreasing, last entry ≤ total events).  Events of pulse `p` are
///   `event_index[p] .. event_index[p+1]` (last pulse runs to the end).
/// - `event_time_zero` — pulse wall-clock times; its `units` attribute is
///   also required (NXevent_data specifies no default; SNS writes
///   `"second"`), accepted via the same table and rescaled to seconds.
///
/// `keep_intervals` are `(t_start, t_end)` pairs in seconds on the same
/// clock as `event_time_zero` (at SNS: seconds since run start — the same
/// clock as `/entry/DASlogs/<pv>/time`, so lists from
/// [`crate::runlog::intervals_where`] /
/// [`crate::runlog::intervals_intersect`] apply directly).  Pulse `p` is
/// kept iff `t_start <= event_time_zero[p] < t_end` for some interval;
/// the list may be unsorted/overlapping (it is normalised internally), but
/// every pair must be finite with `t_end > t_start`.
///
/// **Empty-bank grace (issue #637)**: a bank with zero events (the normal
/// state of every imaging-detector bank on VENUS, where tpx1 is frame-mode)
/// loads to an all-zero spectrum with correct pulse statistics — it never
/// errors.
pub fn load_nexus_bank_spectrum(
    path: &Path,
    bank: &str,
    params: &BankBinningParams,
    keep_intervals: Option<&[(f64, f64)]>,
) -> Result<BankSpectrum, IoError> {
    if params.n_bins == 0 {
        return Err(IoError::InvalidParameter("n_bins must be positive".into()));
    }
    if !params.tof_min_us.is_finite() || !params.tof_max_us.is_finite() {
        return Err(IoError::InvalidParameter(
            "TOF bounds must be finite".into(),
        ));
    }
    if params.tof_max_us <= params.tof_min_us {
        return Err(IoError::InvalidParameter(format!(
            "tof_max_us ({}) must be greater than tof_min_us ({})",
            params.tof_max_us, params.tof_min_us
        )));
    }
    // Normalise the keep-list once: validate pairs, sort, merge overlaps.
    let intervals: Option<Vec<(f64, f64)>> = match keep_intervals {
        None => None,
        Some(raw) => Some(crate::runlog::normalize_intervals(raw)?),
    };

    let file = hdf5::File::open(path).map_err(|e| {
        IoError::FileNotFound(
            path.display().to_string(),
            std::io::Error::other(e.to_string()),
        )
    })?;
    let group = file
        .group(&format!("entry/{bank}"))
        .map_err(|e| IoError::InvalidParameter(format!("Missing /entry/{bank} group: {e}")))?;

    let etz_ds = group.dataset("event_time_zero").map_err(|e| {
        IoError::InvalidParameter(format!("Missing /entry/{bank}/event_time_zero: {e}"))
    })?;
    // NXevent_data specifies only the unit CATEGORY (NX_TIME) with no
    // default, so a missing attribute is an error, not a guess — the same
    // policy #554 established for event_time_offset.  Every surveyed SNS
    // file writes units="second" here.
    let etz_to_s = match read_string_attr(&etz_ds, "units")? {
        None => {
            return Err(IoError::InvalidParameter(format!(
                "/entry/{bank}/event_time_zero has no units attribute; refusing to \
                 guess a time scale (issues #554/#637)"
            )));
        }
        Some(u) => tof_scale_to_us(Some(&u))? * 1e-6,
    };
    let event_time_zero: Vec<f64> = etz_ds
        .read_1d::<f64>()
        .map_err(|e| IoError::Hdf5Error(format!("Failed to read {bank}/event_time_zero: {e}")))?
        .to_vec()
        .into_iter()
        .map(|t| t * etz_to_s)
        .collect();
    // Retention accounting must be exact (same policy as the
    // event_index guards below): a pulse with a non-finite wall-clock
    // time can never match a keep-interval, so its events would vanish
    // from the counts without being tallied — fail loud instead.
    if let Some(i) = event_time_zero.iter().position(|t| !t.is_finite()) {
        return Err(IoError::InvalidParameter(format!(
            "{bank}/event_time_zero[{i}] is not finite ({}); corrupt pulse \
             times would silently exclude events from the accounting",
            event_time_zero[i]
        )));
    }
    let pulse_time_offset_iso = read_string_attr(&etz_ds, "offset")?;

    let event_index: Vec<u64> = group
        .dataset("event_index")
        .map_err(|e| IoError::InvalidParameter(format!("Missing /entry/{bank}/event_index: {e}")))?
        .read_1d::<u64>()
        .map_err(|e| IoError::Hdf5Error(format!("Failed to read {bank}/event_index: {e}")))?
        .to_vec();
    if event_index.len() != event_time_zero.len() {
        return Err(IoError::ShapeMismatch(format!(
            "{bank}: event_index has {} entries but event_time_zero has {}",
            event_index.len(),
            event_time_zero.len()
        )));
    }
    if event_index.windows(2).any(|w| w[1] < w[0]) {
        return Err(IoError::InvalidParameter(format!(
            "{bank}/event_index must be non-decreasing (cumulative first-event index per pulse)"
        )));
    }

    let eto_ds = group.dataset("event_time_offset").map_err(|e| {
        IoError::InvalidParameter(format!("Missing /entry/{bank}/event_time_offset: {e}"))
    })?;
    let tof_scale = match read_string_attr(&eto_ds, "units")? {
        Some(u) => tof_scale_to_us(Some(&u))?,
        None => {
            return Err(IoError::InvalidParameter(format!(
                "/entry/{bank}/event_time_offset has no units attribute; NXevent_data \
                 producers declare TOF units explicitly and this loader refuses to \
                 guess a scale factor (issues #554/#637)"
            )));
        }
    };
    let tof_raw: Vec<f64> = eto_ds
        .read_1d::<f64>()
        .map_err(|e| IoError::Hdf5Error(format!("Failed to read {bank}/event_time_offset: {e}")))?
        .to_vec();
    let events_total = tof_raw.len();
    if let Some(&last) = event_index.last()
        && last as usize > events_total
    {
        return Err(IoError::InvalidParameter(format!(
            "{bank}/event_index last entry ({last}) exceeds total event count ({events_total})"
        )));
    }
    // Retention accounting must be exact: every event belongs to a pulse
    // slice or a drop counter.  A first index > 0 (events preceding the
    // first pulse) or events without any pulse record would vanish
    // silently — fail loud instead (issue #637).
    match event_index.first() {
        Some(&first) if first != 0 => {
            return Err(IoError::InvalidParameter(format!(
                "{bank}/event_index first entry ({first}) must be 0: {first} event(s) \
                 precede the first pulse and would be silently dropped"
            )));
        }
        None if events_total > 0 => {
            return Err(IoError::InvalidParameter(format!(
                "{bank} has {events_total} events but no pulses (empty event_index)"
            )));
        }
        _ => {}
    }

    let pulses_total = event_time_zero.len();
    let bin_w = (params.tof_max_us - params.tof_min_us) / params.n_bins as f64;
    let keep_pulse = |t: f64| -> bool {
        // Normalise -0.0 to +0.0: membership is defined numerically, but
        // total_cmp (needed for NaN robustness) orders -0.0 below +0.0.
        let t = if t == 0.0 { 0.0 } else { t };
        match &intervals {
            None => true,
            Some(iv) => match iv.binary_search_by(|&(a, _)| a.total_cmp(&t)) {
                Ok(i) => t < iv[i].1,
                Err(0) => false,
                Err(i) => t < iv[i - 1].1,
            },
        }
    };

    let mut counts = vec![0u64; params.n_bins];
    let mut pulses_kept = 0usize;
    let mut events_kept = 0usize;
    let mut dropped_tof_range = 0usize;
    let mut dropped_non_finite = 0usize;
    for p in 0..pulses_total {
        if !keep_pulse(event_time_zero[p]) {
            continue;
        }
        pulses_kept += 1;
        let e0 = event_index[p] as usize;
        let e1 = if p + 1 < pulses_total {
            event_index[p + 1] as usize
        } else {
            events_total
        };
        for &raw in &tof_raw[e0..e1] {
            let tof = raw * tof_scale;
            if !tof.is_finite() {
                dropped_non_finite += 1;
                continue;
            }
            if tof < params.tof_min_us || tof >= params.tof_max_us {
                dropped_tof_range += 1;
                continue;
            }
            // Guard the floating-point upper edge: tof < max is checked, but
            // (tof - min) / bin_w can still round up to n_bins.
            let bin = (((tof - params.tof_min_us) / bin_w) as usize).min(params.n_bins - 1);
            counts[bin] += 1;
            events_kept += 1;
        }
    }
    let tof_edges_us = (0..=params.n_bins)
        .map(|i| params.tof_min_us + i as f64 * bin_w)
        .collect();
    Ok(BankSpectrum {
        tof_edges_us,
        counts,
        pulses_total,
        pulses_kept,
        events_total,
        events_kept,
        dropped_tof_range,
        dropped_non_finite,
        pulse_time_offset_iso,
    })
}

#[cfg(test)]
mod bank_tests {
    use super::*;

    /// Write a synthetic NXevent_data bank: per-pulse wall times (s) and
    /// per-pulse event TOF lists (µs, stored in the given units).
    fn create_test_bank(
        path: &Path,
        bank: &str,
        pulse_times_s: &[f64],
        events_per_pulse: &[Vec<f64>],
        tof_units: Option<&str>,
        tof_store_scale: f64,
    ) {
        assert_eq!(pulse_times_s.len(), events_per_pulse.len());
        let file = hdf5::File::create(path).expect("create test file");
        let entry = if let Ok(g) = file.group("entry") {
            g
        } else {
            file.create_group("entry").expect("create entry")
        };
        let g = entry.create_group(bank).expect("create bank");
        let mut index: Vec<u64> = Vec::new();
        let mut tofs: Vec<f64> = Vec::new();
        for evs in events_per_pulse {
            index.push(tofs.len() as u64);
            tofs.extend(evs.iter().map(|t| t * tof_store_scale));
        }
        let etz = g
            .new_dataset_builder()
            .with_data(pulse_times_s)
            .create("event_time_zero")
            .expect("etz");
        etz.new_attr::<hdf5::types::VarLenUnicode>()
            .create("units")
            .expect("attr")
            .write_scalar(&"second".parse::<hdf5::types::VarLenUnicode>().unwrap())
            .expect("write");
        etz.new_attr::<hdf5::types::VarLenUnicode>()
            .create("offset")
            .expect("attr")
            .write_scalar(
                &"2026-06-22T19:01:07.183368667-04:00"
                    .parse::<hdf5::types::VarLenUnicode>()
                    .unwrap(),
            )
            .expect("write");
        g.new_dataset_builder()
            .with_data(&index)
            .create("event_index")
            .expect("ei");
        let eto = g
            .new_dataset_builder()
            .with_data(&tofs)
            .create("event_time_offset")
            .expect("eto");
        if let Some(u) = tof_units {
            eto.new_attr::<hdf5::types::VarLenUnicode>()
                .create("units")
                .expect("attr")
                .write_scalar(&u.parse::<hdf5::types::VarLenUnicode>().unwrap())
                .expect("write");
        }
    }

    fn params(n_bins: usize, lo: f64, hi: f64) -> BankBinningParams {
        BankBinningParams {
            n_bins,
            tof_min_us: lo,
            tof_max_us: hi,
        }
    }

    #[test]
    fn unfiltered_spectrum_counts_all_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bank.h5");
        create_test_bank(
            &path,
            "monitor1",
            &[0.0, 1.0, 2.0],
            &[vec![100.0, 900.0], vec![500.0], vec![100.0, 500.0, 900.0]],
            Some("microsecond"),
            1.0,
        );
        let s = load_nexus_bank_spectrum(&path, "monitor1", &params(2, 0.0, 1000.0), None)
            .expect("load");
        assert_eq!(s.pulses_total, 3);
        assert_eq!(s.pulses_kept, 3);
        assert_eq!(s.events_total, 6);
        assert_eq!(s.events_kept, 6);
        assert_eq!(s.counts, vec![2, 4]); // [0,500): the two 100s; [500,1000): 500,500,900,900
        assert_eq!(s.tof_edges_us, vec![0.0, 500.0, 1000.0]);
        assert!(s.pulse_time_offset_iso.unwrap().starts_with("2026-06-22"));
    }

    #[test]
    fn interval_filter_keeps_only_matching_pulses_with_boundary_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bank.h5");
        // Pulses at t = 0, 10, 20, 30 s with 1, 2, 4, 8 events.
        create_test_bank(
            &path,
            "monitor1",
            &[0.0, 10.0, 20.0, 30.0],
            &[vec![50.0], vec![50.0; 2], vec![50.0; 4], vec![50.0; 8]],
            Some("microsecond"),
            1.0,
        );
        // Half-open [10, 30): keeps pulses at 10 and 20, not 0 and not 30.
        let s = load_nexus_bank_spectrum(
            &path,
            "monitor1",
            &params(1, 0.0, 100.0),
            Some(&[(10.0, 30.0)]),
        )
        .expect("load");
        assert_eq!(s.pulses_kept, 2);
        assert_eq!(s.events_kept, 6);
        assert_eq!(s.counts, vec![6]);
        // Unsorted, overlapping intervals normalise to the same union.
        let s2 = load_nexus_bank_spectrum(
            &path,
            "monitor1",
            &params(1, 0.0, 100.0),
            Some(&[(15.0, 30.0), (10.0, 20.0)]),
        )
        .expect("load");
        assert_eq!(s2.events_kept, 6);
        // Empty keep-list keeps nothing.
        let s3 = load_nexus_bank_spectrum(&path, "monitor1", &params(1, 0.0, 100.0), Some(&[]))
            .expect("load");
        assert_eq!((s3.pulses_kept, s3.events_kept), (0, 0));
        assert_eq!(s3.counts, vec![0]);
    }

    #[test]
    fn empty_bank_loads_gracefully() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bank.h5");
        // The VENUS reality: pulses recorded, zero events (frame-mode tpx1).
        create_test_bank(
            &path,
            "bank100_events",
            &[0.0, 1.0, 2.0],
            &[vec![], vec![], vec![]],
            Some("microsecond"),
            1.0,
        );
        let s = load_nexus_bank_spectrum(
            &path,
            "bank100_events",
            &params(4, 0.0, 1000.0),
            Some(&[(0.5, 2.5)]),
        )
        .expect("empty bank must load");
        assert_eq!(s.pulses_total, 3);
        assert_eq!(s.pulses_kept, 2);
        assert_eq!(s.events_total, 0);
        assert_eq!(s.counts, vec![0, 0, 0, 0]);
    }

    #[test]
    fn tof_units_are_scaled_and_required() {
        let dir = tempfile::tempdir().unwrap();
        // Nanosecond storage scales to the same µs spectrum.
        let p_ns = dir.path().join("ns.h5");
        create_test_bank(&p_ns, "m", &[0.0], &[vec![250.0, 750.0]], Some("ns"), 1e3);
        let s = load_nexus_bank_spectrum(&p_ns, "m", &params(2, 0.0, 1000.0), None).unwrap();
        assert_eq!(s.counts, vec![1, 1]);
        // Missing units attribute on this path is an error, not a guess.
        let p_none = dir.path().join("none.h5");
        create_test_bank(&p_none, "m", &[0.0], &[vec![250.0]], None, 1.0);
        let err = load_nexus_bank_spectrum(&p_none, "m", &params(2, 0.0, 1000.0), None)
            .expect_err("must refuse to guess units");
        assert!(err.to_string().contains("units"), "{err}");
    }

    #[test]
    fn fixed_length_ascii_attributes_read_correctly() {
        // SNS/ADARA facility files store attributes as FIXED-length ASCII
        // (rustpix uses variable-length UTF-8) — both must read (issue #637).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixed.h5");
        create_test_bank(&path, "m", &[0.0], &[vec![250.0, 750.0]], None, 1.0);
        {
            let file = hdf5::File::open_rw(&path).expect("reopen");
            let eto = file.dataset("entry/m/event_time_offset").expect("eto");
            let units = hdf5::types::FixedAscii::<16>::from_ascii(b"microsecond").unwrap();
            eto.new_attr::<hdf5::types::FixedAscii<16>>()
                .create("units")
                .expect("attr")
                .write_scalar(&units)
                .expect("write");
        }
        let s = load_nexus_bank_spectrum(&path, "m", &params(2, 0.0, 1000.0), None)
            .expect("fixed-ascii units must parse");
        assert_eq!(s.counts, vec![1, 1]);
    }

    #[test]
    fn non_finite_pulse_time_fails_loud() {
        // A NaN event_time_zero can never match a keep-interval, so its
        // events would silently vanish from the retention accounting.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nanpulse.h5");
        create_test_bank(
            &path,
            "m",
            &[0.0, f64::NAN],
            &[vec![100.0], vec![200.0]],
            Some("us"),
            1.0,
        );
        let err = load_nexus_bank_spectrum(&path, "m", &params(1, 0.0, 1000.0), None)
            .expect_err("non-finite pulse time must error");
        assert!(err.to_string().contains("not finite"), "{err}");
    }

    #[test]
    fn orphan_head_events_fail_loud() {
        // event_index[0] != 0 means events precede the first pulse; they
        // belong to no pulse slice and would vanish from the accounting.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orphan.h5");
        create_test_bank(&path, "m", &[0.0], &[vec![100.0, 200.0]], Some("us"), 1.0);
        {
            let file = hdf5::File::open_rw(&path).expect("reopen");
            let ei = file.dataset("entry/m/event_index").expect("ei");
            ei.write(&ndarray::arr1(&[1u64])).expect("overwrite");
        }
        let err = load_nexus_bank_spectrum(&path, "m", &params(1, 0.0, 1000.0), None)
            .expect_err("orphan head events must error");
        assert!(err.to_string().contains("precede the first pulse"), "{err}");
    }

    #[test]
    fn malformed_inputs_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bank.h5");
        create_test_bank(
            &path,
            "m",
            &[0.0, 1.0],
            &[vec![1.0], vec![2.0]],
            Some("us"),
            1.0,
        );
        // Bad interval pairs.
        for bad in [(5.0, 5.0), (5.0, 1.0), (f64::NAN, 1.0)] {
            assert!(
                load_nexus_bank_spectrum(&path, "m", &params(1, 0.0, 10.0), Some(&[bad])).is_err()
            );
        }
        // Bad binning params.
        assert!(load_nexus_bank_spectrum(&path, "m", &params(0, 0.0, 10.0), None).is_err());
        assert!(load_nexus_bank_spectrum(&path, "m", &params(1, 10.0, 10.0), None).is_err());
        // Missing bank.
        assert!(load_nexus_bank_spectrum(&path, "nope", &params(1, 0.0, 10.0), None).is_err());
    }
}
