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

    // Read metadata attributes (try group level first, then entry level)
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
        .map_err(|e| IoError::TiffDecode(format!("Failed to read histogram counts: {e}")))?;

    // Sum over rotation angle axis (axis 0): [rot, y, x, tof] → [y, x, tof]
    let summed = counts_u64.sum_axis(ndarray::Axis(0));

    // Convert to f64 and transpose to NEREIDS convention [tof, y, x]
    let (n_y, n_x, n_tof) = (summed.shape()[0], summed.shape()[1], summed.shape()[2]);
    let mut counts_f64 = Array3::<f64>::zeros((n_tof, n_y, n_x));
    for t in 0..n_tof {
        for y in 0..n_y {
            for x in 0..n_x {
                counts_f64[[t, y, x]] = summed[[y, x, t]] as f64;
            }
        }
    }

    // Read TOF axis (nanoseconds → microseconds)
    let tof_edges_us = read_tof_axis(&hist_group)?;

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
        let counts = vec![0u64; 1 * 2 * 3 * 4];
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
    }

    #[test]
    fn test_ns_to_us_conversion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");

        let counts = vec![0u64; 1 * 1 * 1 * 3];
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
}
