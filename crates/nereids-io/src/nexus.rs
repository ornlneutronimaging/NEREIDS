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
//! /entry/histogram/time_of_flight  — f64 1D, nanoseconds
//! /entry/neutrons/event_time_offset — u64 1D, nanoseconds
//! /entry/neutrons/x                — f64 1D, pixel coordinate
//! /entry/neutrons/y                — f64 1D, pixel coordinate
//! /entry/pixel_masks/dead          — u8  2D [y, x]
//! ```
//!
//! Metadata attributes on `/entry` or group level:
//! - `flight_path_m` (f64)
//! - `tof_offset_ns` (f64)

use std::path::Path;

use ndarray::Array3;

use crate::error::IoError;

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
    /// TOF bin edges or centers in nanoseconds, if present.
    pub tof_edges_ns: Option<Vec<f64>>,
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
    let (has_histogram, histogram_shape, tof_edges_ns) = probe_histogram_group(&entry);

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
        tof_edges_ns,
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

    let counts_u64: ndarray::Array4<u64> = counts_ds
        .read()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read histogram counts: {e}")))?;

    // Collapse the rotation-angle axis according to the caller's policy
    // (issue #430).  For single-angle files all three modes are
    // equivalent — they just take the only slice that exists.
    let n_rot = shape[0];
    let combined_yxtof = match mode {
        MultiAngleMode::Error => {
            if n_rot > 1 {
                return Err(IoError::InvalidParameter(format!(
                    "NeXus histogram has {n_rot} rotation angles — refusing to silently \
                     combine them (issue #430).  Call load_nexus_histogram_with_mode with \
                     MultiAngleMode::Sum to preserve the legacy sum-over-angles behaviour, \
                     or MultiAngleMode::SelectAngle(i) to extract a single projection."
                )));
            }
            counts_u64.sum_axis(ndarray::Axis(0))
        }
        MultiAngleMode::Sum => counts_u64.sum_axis(ndarray::Axis(0)),
        MultiAngleMode::SelectAngle(idx) => {
            if idx >= n_rot {
                return Err(IoError::InvalidParameter(format!(
                    "MultiAngleMode::SelectAngle({idx}) out of range: file has {n_rot} \
                     rotation angle(s), valid indices are 0..{n_rot}"
                )));
            }
            counts_u64.index_axis(ndarray::Axis(0), idx).to_owned()
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

    // Read dead pixel mask
    let dead_pixels = read_dead_pixel_mask(&entry);

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
#[derive(Debug, Clone)]
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
/// Reads `/entry/neutrons/event_time_offset` (u64 ns), `x` (f64), `y` (f64),
/// converts TOF from nanoseconds to microseconds, then bins events into a
/// `(n_bins, height, width)` histogram grid.
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

    // Read event arrays
    let tof_ns: Vec<u64> = neutrons
        .dataset("event_time_offset")
        .map_err(|e| IoError::InvalidParameter(format!("Missing event_time_offset dataset: {e}")))?
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

    if tof_ns.len() != x_coords.len() || tof_ns.len() != y_coords.len() {
        return Err(IoError::ShapeMismatch(format!(
            "Event arrays have mismatched lengths: tof={}, x={}, y={}",
            tof_ns.len(),
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
    let total = tof_ns.len();
    let mut kept = 0usize;
    let mut dropped_non_finite = 0usize;
    let mut dropped_tof_range = 0usize;
    let mut dropped_spatial = 0usize;

    for i in 0..tof_ns.len() {
        let tof_us = tof_ns[i] as f64 / 1000.0; // ns → µs
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

    // Read dead pixel mask
    let dead_pixels = read_dead_pixel_mask(&entry);

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

    // Try reading TOF axis
    let tof_edges = hist
        .dataset("time_of_flight")
        .ok()
        .and_then(|ds| ds.read_1d::<f64>().ok())
        .map(|a| a.to_vec());

    (true, histogram_shape, tof_edges)
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

/// Read TOF axis from the histogram group, converting ns → µs.
fn read_tof_axis(hist_group: &hdf5::Group) -> Result<Vec<f64>, IoError> {
    let tof_ds = hist_group.dataset("time_of_flight").map_err(|e| {
        IoError::InvalidParameter(format!(
            "Missing /entry/histogram/time_of_flight dataset: {e}"
        ))
    })?;

    let tof_ns: Vec<f64> = tof_ds
        .read_1d::<f64>()
        .map_err(|e| IoError::InvalidParameter(format!("Failed to read time_of_flight: {e}")))?
        .to_vec();

    // Convert nanoseconds → microseconds
    Ok(tof_ns.iter().map(|&ns| ns / 1000.0).collect())
}

/// Read a scalar f64 attribute from a group.
fn read_f64_attr(group: &hdf5::Group, name: &str) -> Option<f64> {
    group
        .attr(name)
        .ok()
        .and_then(|a| a.read_scalar::<f64>().ok())
}

/// Read dead pixel mask from `/entry/pixel_masks/dead`.
fn read_dead_pixel_mask(entry: &hdf5::Group) -> Option<ndarray::Array2<bool>> {
    let masks = entry.group("pixel_masks").ok()?;
    let dead_ds = masks.dataset("dead").ok()?;
    let dead_u8: ndarray::Array2<u8> = dead_ds.read().ok()?;
    Some(dead_u8.mapv(|v| v != 0))
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
        assert!(meta.tof_edges_ns.is_some());
        assert_eq!(meta.tof_edges_ns.unwrap().len(), 5);
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
    /// `n_rot == 1` is the trivial non-collapsing case.
    #[test]
    fn test_load_nexus_histogram_error_mode_allows_single_angle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single_error.h5");
        let counts = vec![7u64; 2 * 3 * 2];
        let tof_ns = vec![1000.0, 2000.0, 3000.0];
        create_test_histogram(&path, &counts, [1, 2, 3, 2], &tof_ns, None);

        let data = load_nexus_histogram_with_mode(&path, MultiAngleMode::Error).unwrap();
        assert_eq!(data.n_rotation_angles, 1);
        // Value preserved (not doubled or summed — single angle)
        assert_eq!(data.counts[[0, 0, 0]], 7.0);
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
}
