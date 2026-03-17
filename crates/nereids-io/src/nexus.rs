//! NeXus/HDF5 reading for rustpix-processed neutron imaging data.
//!
//! Supports two data modalities from rustpix output files:
//! - **Histogram**: 4D counts array `(rot_angle, y, x, tof)` summed over
//!   rotation angles and transposed to NEREIDS convention `(tof, y, x)`.
//! - **Events**: per-neutron `(event_time_offset, x, y)` histogrammed into
//!   a `(tof, y, x)` grid with user-specified binning parameters.
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

/// Load histogram data from a NeXus file.
///
/// Reads `/entry/histogram/counts` (u64 4D), sums over the rotation angle
/// axis (axis 0), converts to f64, and transposes to NEREIDS convention
/// `(tof, y, x)`. TOF values are converted from nanoseconds to microseconds.
pub fn load_nexus_histogram(path: &Path) -> Result<NexusHistogramData, IoError> {
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

    // D-5: Sum over rotation angle axis (axis 0): [rot, y, x, tof] → [y, x, tof]
    // When n_rot > 1, this silently collapses multi-angle acquisitions.
    // Log a warning so the user knows angles were combined.
    let n_rot = shape[0];
    if n_rot > 1 {
        eprintln!(
            "Warning: NeXus histogram has {n_rot} rotation angles — summing into a single \
             volume. Multi-angle analysis is not yet supported."
        );
    }
    let summed = counts_u64.sum_axis(ndarray::Axis(0));

    // Convert to f64 and transpose [y, x, tof] → NEREIDS convention [tof, y, x]
    let counts_f64: Array3<f64> = summed
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
/// Events outside the spatial bounds or TOF range are silently dropped.
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

    // Histogram events
    let dt_us = (params.tof_max_us - params.tof_min_us) / params.n_bins as f64;
    let mut counts = Array3::<f64>::zeros((params.n_bins, params.height, params.width));

    for i in 0..tof_ns.len() {
        let tof_us = tof_ns[i] as f64 / 1000.0; // ns → µs
        if !tof_us.is_finite() {
            continue;
        }

        // Skip events outside TOF range
        if tof_us < params.tof_min_us || tof_us >= params.tof_max_us {
            continue;
        }

        // Pixel coordinates (round to nearest integer)
        let px = x_coords[i].round() as isize;
        let py = y_coords[i].round() as isize;

        // Skip events outside spatial bounds
        if px < 0 || py < 0 || px >= params.width as isize || py >= params.height as isize {
            continue;
        }

        let tof_bin = ((tof_us - params.tof_min_us) / dt_us) as usize;
        // Clamp to last bin (guard against floating-point rounding near the upper boundary)
        let tof_bin = tof_bin.min(params.n_bins - 1);

        counts[[tof_bin, py as usize, px as usize]] += 1.0;
    }

    // Read flight path
    let flight_path_m = read_f64_attr(&neutrons, "flight_path_m")
        .or_else(|| read_f64_attr(&entry, "flight_path_m"));

    // Read dead pixel mask
    let dead_pixels = read_dead_pixel_mask(&entry);

    Ok(NexusHistogramData {
        counts,
        tof_edges_us,
        flight_path_m,
        dead_pixels,
        n_rotation_angles: 1, // Event data has no rotation dimension
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
    fn test_load_nexus_histogram() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");

        // 2 rot angles, 2x3 spatial, 2 TOF bins
        // Each element set to known value for verification
        let mut counts = vec![0u64; 2 * 2 * 3 * 2];
        // counts[rot, y, x, tof] — linearized in row-major order
        // Set some values: rot=0, y=0, x=0, tof=0 → index 0
        counts[0] = 10;
        // rot=1, y=0, x=0, tof=0 → index 1*2*3*2 = 12
        counts[12] = 5;

        let tof_ns = vec![1000.0, 2000.0, 3000.0]; // 3 edges for 2 bins
        create_test_histogram(&path, &counts, [2, 2, 3, 2], &tof_ns, Some(25.0));

        let data = load_nexus_histogram(&path).unwrap();

        // Shape should be (n_tof=2, n_y=2, n_x=3) after summing rot and transposing
        assert_eq!(data.counts.shape(), &[2, 2, 3]);

        // Summed over rot: counts[tof=0, y=0, x=0] = 10 + 5 = 15
        assert_eq!(data.counts[[0, 0, 0]], 15.0);

        // TOF edges converted ns → µs
        assert_eq!(data.tof_edges_us.len(), 3);
        assert!((data.tof_edges_us[0] - 1.0).abs() < 1e-10); // 1000 ns → 1 µs
        assert!((data.tof_edges_us[1] - 2.0).abs() < 1e-10);
        assert!((data.tof_edges_us[2] - 3.0).abs() < 1e-10);

        assert_eq!(data.flight_path_m, Some(25.0));

        // D-5: rotation angle count is carried through
        assert_eq!(data.n_rotation_angles, 2);
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
}
