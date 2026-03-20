//! Project file save and load for `.nrd.h5` (NEREIDS HDF5 archive).
//!
//! The project file captures the full session state so users can persist
//! and share analysis sessions. This module defines [`ProjectSnapshot`]
//! (a serialization-friendly subset of the GUI's `AppState`) and the
//! [`save_project`] and [`load_project`] functions for HDF5 I/O.

use std::path::Path;

use hdf5::types::VarLenUnicode;
use ndarray::{Array2, Array3};

use nereids_endf::resonance::ResonanceData;

use crate::error::IoError;

/// Current schema version written to `/meta/version`.
pub const PROJECT_SCHEMA_VERSION: &str = "1.0";

/// Serialization-friendly snapshot of the full session state.
///
/// All GUI-specific enums are stored as plain strings so this struct
/// has no dependency on the GUI crate. The GUI handles
/// `AppState <-> ProjectSnapshot` conversion.
#[derive(Debug)]
pub struct ProjectSnapshot {
    // -- meta --
    pub schema_version: String,
    pub created_utc: String,
    pub software_version: String,
    /// "spatial" | "single"
    pub fitting_type: String,
    /// "events" | "pre_normalized" | "transmission"
    pub data_type: String,

    // -- config/beamline --
    pub flight_path_m: f64,
    pub delay_us: f64,
    pub proton_charge_sample: f64,
    pub proton_charge_ob: f64,

    // -- config/isotopes (parallel arrays) --
    pub isotope_z: Vec<u32>,
    pub isotope_a: Vec<u32>,
    pub isotope_symbol: Vec<String>,
    pub isotope_density: Vec<f64>,
    pub isotope_enabled: Vec<bool>,

    // -- config/solver --
    /// "lm" | "poisson_kl"
    pub solver_method: String,
    pub max_iter: u32,
    pub temperature_k: f64,
    pub fit_temperature: bool,

    // -- config/resolution --
    pub resolution_enabled: bool,
    /// "gaussian" | "tabulated"
    pub resolution_kind: String,
    pub delta_t_us: Option<f64>,
    pub delta_l_m: Option<f64>,
    pub tabulated_path: Option<String>,

    // -- config/rois --
    /// Each ROI: [y_start, y_end, x_start, x_end].
    pub rois: Vec<[u64; 4]>,

    // -- config/endf --
    pub endf_library: String,

    // -- data --
    /// "linked" | "embedded"
    pub data_mode: String,
    pub sample_path: Option<String>,
    pub open_beam_path: Option<String>,
    pub spectrum_path: Option<String>,
    pub hdf5_path: Option<String>,
    /// "tof_us" | "energy_ev"
    pub spectrum_unit: String,
    /// "bin_edges" | "bin_centers"
    pub spectrum_kind: String,
    pub rebin_factor: u32,
    pub rebin_applied: bool,

    // -- data (embedded mode, populated on load) --
    pub sample_data: Option<Array3<f64>>,
    pub open_beam_data: Option<Array3<f64>>,
    pub spectrum_values: Option<Vec<f64>>,

    // -- intermediate (always embedded) --
    pub normalized: Option<Array3<f64>>,
    /// D-1: Per-bin transmission uncertainty σ (same shape as normalized).
    /// Previously missing from the snapshot, causing reloaded projects to
    /// lose uncertainty information (reconstructed as zeros).
    pub normalized_uncertainty: Option<Array3<f64>>,
    pub energies: Option<Vec<f64>>,
    /// D-20: Dead-pixel mask (true = dead). Same spatial dimensions as the
    /// transmission data (height × width). `None` when no mask is available.
    pub dead_pixels: Option<Array2<bool>>,

    // -- results (always embedded) --
    pub density_maps: Option<Vec<Array2<f64>>>,
    pub uncertainty_maps: Option<Vec<Array2<f64>>>,
    pub chi_squared_map: Option<Array2<f64>>,
    pub converged_map: Option<Array2<bool>>,
    pub temperature_map: Option<Array2<f64>>,
    pub n_converged: Option<usize>,
    pub n_total: Option<usize>,
    pub result_isotope_labels: Option<Vec<String>>,
    /// Per-pixel normalization factor (background fitting).
    pub anorm_map: Option<Array2<f64>>,
    /// Per-pixel background [A, B, C] maps (background fitting).
    /// Stored as 3 separate Array2 maps (one per coefficient).
    pub background_maps: Option<[Array2<f64>; 3]>,

    // -- results/single_fit (single-pixel fit, optional) --
    pub single_fit_densities: Option<Vec<f64>>,
    pub single_fit_uncertainties: Option<Vec<f64>>,
    pub single_fit_chi_squared: Option<f64>,
    pub single_fit_temperature: Option<f64>,
    pub single_fit_temperature_unc: Option<f64>,
    pub single_fit_converged: Option<bool>,
    pub single_fit_iterations: Option<usize>,
    pub single_fit_pixel: Option<(usize, usize)>,
    pub single_fit_labels: Option<Vec<String>>,
    /// Fitted normalization factor from single-pixel fit (1.0 default).
    pub single_fit_anorm: Option<f64>,
    /// Fitted background [BackA, BackB, BackC] from single-pixel fit.
    pub single_fit_background: Option<[f64; 3]>,

    // -- endf_cache --
    /// (symbol, resonance_data) pairs for offline loading.
    pub endf_cache: Vec<(String, ResonanceData)>,

    // -- provenance --
    /// (timestamp, kind, message) triples.
    pub provenance: Vec<(String, String, String)>,
}

impl Default for ProjectSnapshot {
    fn default() -> Self {
        Self {
            schema_version: String::new(),
            created_utc: String::new(),
            software_version: String::new(),
            fitting_type: String::new(),
            data_type: String::new(),
            flight_path_m: 0.0,
            delay_us: 0.0,
            proton_charge_sample: 0.0,
            proton_charge_ob: 0.0,
            isotope_z: vec![],
            isotope_a: vec![],
            isotope_symbol: vec![],
            isotope_density: vec![],
            isotope_enabled: vec![],
            solver_method: String::new(),
            max_iter: 0,
            temperature_k: 0.0,
            fit_temperature: false,
            resolution_enabled: false,
            resolution_kind: String::new(),
            delta_t_us: None,
            delta_l_m: None,
            tabulated_path: None,
            rois: vec![],
            endf_library: String::new(),
            data_mode: String::new(),
            sample_path: None,
            open_beam_path: None,
            spectrum_path: None,
            hdf5_path: None,
            spectrum_unit: String::new(),
            spectrum_kind: String::new(),
            rebin_factor: 0,
            rebin_applied: false,
            sample_data: None,
            open_beam_data: None,
            spectrum_values: None,
            normalized: None,
            normalized_uncertainty: None,
            energies: None,
            dead_pixels: None,
            density_maps: None,
            uncertainty_maps: None,
            chi_squared_map: None,
            converged_map: None,
            temperature_map: None,
            n_converged: None,
            n_total: None,
            result_isotope_labels: None,
            anorm_map: None,
            background_maps: None,
            single_fit_densities: None,
            single_fit_uncertainties: None,
            single_fit_chi_squared: None,
            single_fit_temperature: None,
            single_fit_temperature_unc: None,
            single_fit_converged: None,
            single_fit_iterations: None,
            single_fit_pixel: None,
            single_fit_labels: None,
            single_fit_anorm: None,
            single_fit_background: None,
            endf_cache: vec![],
            provenance: vec![],
        }
    }
}

/// Borrowed references to raw data for embedded saves.
///
/// Avoids cloning multi-GB arrays into [`ProjectSnapshot`].
/// Pass `None` for linked-mode saves.
pub struct EmbeddedData<'a> {
    pub sample: Option<&'a Array3<f64>>,
    pub open_beam: Option<&'a Array3<f64>>,
    pub spectrum: Option<&'a [f64]>,
}

/// Estimated compression ratio for gzip-4 on float64 neutron data.
pub const EMBED_COMPRESSION_RATIO: f64 = 3.0;

/// Estimate (uncompressed, compressed) byte sizes for embedding raw data.
pub fn estimate_embedded_size(
    sample: Option<&Array3<f64>>,
    open_beam: Option<&Array3<f64>>,
    spectrum: Option<&[f64]>,
) -> (u64, u64) {
    let mut raw: u64 = 0;
    if let Some(s) = sample {
        raw += (s.len() as u64) * 8;
    }
    if let Some(ob) = open_beam {
        raw += (ob.len() as u64) * 8;
    }
    if let Some(sp) = spectrum {
        raw += (sp.len() as u64) * 8;
    }
    let compressed = (raw as f64 / EMBED_COMPRESSION_RATIO) as u64;
    (raw, compressed)
}

/// Write a project snapshot to an HDF5 file at `path` (linked mode).
pub fn save_project(path: &Path, snap: &ProjectSnapshot) -> Result<(), IoError> {
    save_project_with_data(path, snap, None)
}

/// Write a project snapshot with optional embedded raw data.
///
/// When `embedded` is `Some`, raw data arrays are written to `/data/embedded/`
/// and the mode attribute is set to `"embedded"`. File paths in `/data/links/`
/// are always written for provenance.
pub fn save_project_with_data(
    path: &Path,
    snap: &ProjectSnapshot,
    embedded: Option<&EmbeddedData<'_>>,
) -> Result<(), IoError> {
    let file = hdf5::File::create(path).map_err(|e| IoError::Hdf5Error(format!("create: {e}")))?;

    write_meta(&file, snap)?;
    write_config(&file, snap)?;
    write_data_links(&file, snap, embedded)?;
    write_intermediate(&file, snap)?;
    write_results(&file, snap)?;
    write_endf_cache(&file, snap)?;
    write_provenance(&file, snap)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn hdf5_err(context: &str, e: impl std::fmt::Display) -> IoError {
    IoError::Hdf5Error(format!("{context}: {e}"))
}

fn write_str_attr(loc: &hdf5::Group, name: &str, value: &str) -> Result<(), IoError> {
    let val: VarLenUnicode = value.parse().map_err(|e| hdf5_err(name, e))?;
    loc.new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .and_then(|a| a.write_scalar(&val))
        .map_err(|e| hdf5_err(name, e))
}

fn write_f64_attr(loc: &hdf5::Group, name: &str, value: f64) -> Result<(), IoError> {
    loc.new_attr::<f64>()
        .shape(())
        .create(name)
        .and_then(|a| a.write_scalar(&value))
        .map_err(|e| hdf5_err(name, e))
}

fn write_u32_attr(loc: &hdf5::Group, name: &str, value: u32) -> Result<(), IoError> {
    loc.new_attr::<u32>()
        .shape(())
        .create(name)
        .and_then(|a| a.write_scalar(&value))
        .map_err(|e| hdf5_err(name, e))
}

fn write_bool_attr(loc: &hdf5::Group, name: &str, value: bool) -> Result<(), IoError> {
    let v: u8 = u8::from(value);
    loc.new_attr::<u8>()
        .shape(())
        .create(name)
        .and_then(|a| a.write_scalar(&v))
        .map_err(|e| hdf5_err(name, e))
}

fn write_u64_attr(loc: &hdf5::Group, name: &str, value: u64) -> Result<(), IoError> {
    loc.new_attr::<u64>()
        .shape(())
        .create(name)
        .and_then(|a| a.write_scalar(&value))
        .map_err(|e| hdf5_err(name, e))
}

fn write_meta(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let g = file
        .create_group("meta")
        .map_err(|e| hdf5_err("create /meta", e))?;
    write_str_attr(&g, "version", &snap.schema_version)?;
    write_str_attr(&g, "created_utc", &snap.created_utc)?;
    write_str_attr(&g, "software_version", &snap.software_version)?;
    write_str_attr(&g, "fitting_type", &snap.fitting_type)?;
    write_str_attr(&g, "data_type", &snap.data_type)?;
    Ok(())
}

fn write_config(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let config = file
        .create_group("config")
        .map_err(|e| hdf5_err("create /config", e))?;

    // Beamline
    let bl = config
        .create_group("beamline")
        .map_err(|e| hdf5_err("create /config/beamline", e))?;
    write_f64_attr(&bl, "flight_path_m", snap.flight_path_m)?;
    write_f64_attr(&bl, "delay_us", snap.delay_us)?;
    write_f64_attr(&bl, "proton_charge_sample", snap.proton_charge_sample)?;
    write_f64_attr(&bl, "proton_charge_ob", snap.proton_charge_ob)?;

    // Isotopes (parallel arrays as datasets)
    let iso = config
        .create_group("isotopes")
        .map_err(|e| hdf5_err("create /config/isotopes", e))?;
    let n = snap.isotope_z.len();
    if n > 0 {
        iso.new_dataset::<u32>()
            .shape([n])
            .create("z")
            .and_then(|ds| ds.write_raw(&snap.isotope_z))
            .map_err(|e| hdf5_err("/config/isotopes/z", e))?;

        iso.new_dataset::<u32>()
            .shape([n])
            .create("a")
            .and_then(|ds| ds.write_raw(&snap.isotope_a))
            .map_err(|e| hdf5_err("/config/isotopes/a", e))?;

        let symbols: Vec<VarLenUnicode> = snap
            .isotope_symbol
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|e| hdf5_err("parse VarLenUnicode symbol", e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        iso.new_dataset::<VarLenUnicode>()
            .shape([n])
            .create("symbol")
            .and_then(|ds| ds.write_raw(&symbols))
            .map_err(|e| hdf5_err("/config/isotopes/symbol", e))?;

        iso.new_dataset::<f64>()
            .shape([n])
            .create("density")
            .and_then(|ds| ds.write_raw(&snap.isotope_density))
            .map_err(|e| hdf5_err("/config/isotopes/density", e))?;

        let enabled: Vec<u8> = snap.isotope_enabled.iter().map(|&b| u8::from(b)).collect();
        iso.new_dataset::<u8>()
            .shape([n])
            .create("enabled")
            .and_then(|ds| ds.write_raw(&enabled))
            .map_err(|e| hdf5_err("/config/isotopes/enabled", e))?;
    }

    // Solver
    let solver = config
        .create_group("solver")
        .map_err(|e| hdf5_err("create /config/solver", e))?;
    write_str_attr(&solver, "method", &snap.solver_method)?;
    write_u32_attr(&solver, "max_iter", snap.max_iter)?;
    write_f64_attr(&solver, "temperature_k", snap.temperature_k)?;
    write_bool_attr(&solver, "fit_temperature", snap.fit_temperature)?;

    // Resolution
    let res = config
        .create_group("resolution")
        .map_err(|e| hdf5_err("create /config/resolution", e))?;
    write_bool_attr(&res, "enabled", snap.resolution_enabled)?;
    write_str_attr(&res, "kind", &snap.resolution_kind)?;
    if let Some(dt) = snap.delta_t_us {
        write_f64_attr(&res, "delta_t_us", dt)?;
    }
    if let Some(dl) = snap.delta_l_m {
        write_f64_attr(&res, "delta_l_m", dl)?;
    }
    if let Some(ref tp) = snap.tabulated_path {
        write_str_attr(&res, "tabulated_path", tp)?;
    }

    // ROIs
    if !snap.rois.is_empty() {
        let n_rois = snap.rois.len();
        let flat: Vec<u64> = snap.rois.iter().flat_map(|r| r.iter().copied()).collect();
        config
            .new_dataset::<u64>()
            .shape([n_rois, 4])
            .create("rois")
            .and_then(|ds| ds.write_raw(&flat))
            .map_err(|e| hdf5_err("/config/rois", e))?;
    }

    // ENDF library
    write_str_attr(&config, "endf_library", &snap.endf_library)?;

    Ok(())
}

fn write_data_links(
    file: &hdf5::File,
    snap: &ProjectSnapshot,
    embedded: Option<&EmbeddedData<'_>>,
) -> Result<(), IoError> {
    let data = file
        .create_group("data")
        .map_err(|e| hdf5_err("create /data", e))?;

    let mode = if embedded.is_some() {
        "embedded"
    } else {
        &snap.data_mode
    };
    write_str_attr(&data, "mode", mode)?;
    write_str_attr(&data, "spectrum_unit", &snap.spectrum_unit)?;
    write_str_attr(&data, "spectrum_kind", &snap.spectrum_kind)?;
    write_u32_attr(&data, "rebin_factor", snap.rebin_factor)?;
    write_bool_attr(&data, "rebin_applied", snap.rebin_applied)?;

    // Always write links for provenance (original file paths)
    let links = data
        .create_group("links")
        .map_err(|e| hdf5_err("create /data/links", e))?;
    if let Some(ref p) = snap.sample_path {
        write_str_attr(&links, "sample_path", p)?;
    }
    if let Some(ref p) = snap.open_beam_path {
        write_str_attr(&links, "open_beam_path", p)?;
    }
    if let Some(ref p) = snap.spectrum_path {
        write_str_attr(&links, "spectrum_path", p)?;
    }
    if let Some(ref p) = snap.hdf5_path {
        write_str_attr(&links, "hdf5_path", p)?;
    }

    // Write embedded data if present
    if let Some(emb) = embedded {
        write_embedded_data(&data, emb)?;
    }

    Ok(())
}

fn write_embedded_data(data_group: &hdf5::Group, emb: &EmbeddedData<'_>) -> Result<(), IoError> {
    let embedded = data_group
        .create_group("embedded")
        .map_err(|e| hdf5_err("create /data/embedded", e))?;

    if let Some(sample) = emb.sample {
        write_chunked_3d(&embedded, "sample", sample, "/data/embedded")?;
    }

    if let Some(ob) = emb.open_beam {
        write_chunked_3d(&embedded, "open_beam", ob, "/data/embedded")?;
    }

    if let Some(spectrum) = emb.spectrum {
        embedded
            .new_dataset::<f64>()
            .shape([spectrum.len()])
            .deflate(4)
            .create("spectrum")
            .and_then(|ds| ds.write_raw(spectrum))
            .map_err(|e| hdf5_err("/data/embedded/spectrum", e))?;
    }

    Ok(())
}

/// Write a 3D f64 array as a chunked, gzip-compressed dataset.
///
/// Uses `as_standard_layout()` to get a contiguous view without allocating
/// when the array is already in standard (row-major) layout. Only copies
/// if the array has non-standard strides.
///
/// Zero-dimension arrays are silently skipped (nothing to write).
fn write_chunked_3d(
    group: &hdf5::Group,
    name: &str,
    arr: &Array3<f64>,
    path_prefix: &str,
) -> Result<(), IoError> {
    let shape = [arr.shape()[0], arr.shape()[1], arr.shape()[2]];
    if shape.contains(&0) {
        return Ok(());
    }
    let contiguous = arr.as_standard_layout();
    let slice = contiguous.as_slice().ok_or_else(|| {
        hdf5_err(
            &format!("{path_prefix}/{name}"),
            "array is not contiguous after as_standard_layout",
        )
    })?;
    group
        .new_dataset::<f64>()
        .shape(shape)
        .chunk(chunk_shape_3d(shape))
        .deflate(4)
        .create(name)
        .and_then(|ds| ds.write_raw(slice))
        .map_err(|e| hdf5_err(&format!("{path_prefix}/{name}"), e))?;
    Ok(())
}

fn write_intermediate(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let inter = file
        .create_group("intermediate")
        .map_err(|e| hdf5_err("create /intermediate", e))?;

    if let Some(ref norm) = snap.normalized {
        write_chunked_3d(&inter, "normalized", norm, "/intermediate")?;
    }

    // D-1: Save per-bin transmission uncertainty alongside normalized data.
    if let Some(ref unc) = snap.normalized_uncertainty {
        write_chunked_3d(&inter, "normalized_uncertainty", unc, "/intermediate")?;
    }

    if let Some(ref energies) = snap.energies {
        inter
            .new_dataset::<f64>()
            .shape([energies.len()])
            .create("energies")
            .and_then(|ds| ds.write_raw(energies))
            .map_err(|e| hdf5_err("/intermediate/energies", e))?;
    }

    // D-20: Persist dead-pixel mask as u8 (0 = live, 1 = dead).
    if let Some(ref dp) = snap.dead_pixels {
        let shape = [dp.shape()[0], dp.shape()[1]];
        if !shape.contains(&0) {
            let data: Vec<u8> = dp.iter().map(|&b| u8::from(b)).collect();
            inter
                .new_dataset::<u8>()
                .shape(shape)
                .create("dead_pixels")
                .and_then(|ds| ds.write_raw(&data))
                .map_err(|e| hdf5_err("/intermediate/dead_pixels", e))?;
        }
    }

    Ok(())
}

fn write_results(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let results = file
        .create_group("results")
        .map_err(|e| hdf5_err("create /results", e))?;

    if let Some(ref maps) = snap.density_maps {
        let density = results
            .create_group("density")
            .map_err(|e| hdf5_err("create /results/density", e))?;
        let labels = snap.result_isotope_labels.as_deref().unwrap_or_default();
        for (i, map) in maps.iter().enumerate() {
            let name = labels
                .get(i)
                .map_or_else(|| format!("isotope_{i}"), |s| s.clone());
            let shape = [map.shape()[0], map.shape()[1]];
            let data: Vec<f64> = map.iter().copied().collect();
            density
                .new_dataset::<f64>()
                .shape(shape)
                .chunk(shape)
                .deflate(4)
                .create(name.as_str())
                .and_then(|ds| ds.write_raw(&data))
                .map_err(|e| hdf5_err(&format!("/results/density/{name}"), e))?;
        }
    }

    if let Some(ref maps) = snap.uncertainty_maps {
        let unc = results
            .create_group("uncertainty")
            .map_err(|e| hdf5_err("create /results/uncertainty", e))?;
        let labels = snap.result_isotope_labels.as_deref().unwrap_or_default();
        for (i, map) in maps.iter().enumerate() {
            let name = labels
                .get(i)
                .map_or_else(|| format!("isotope_{i}"), |s| s.clone());
            let shape = [map.shape()[0], map.shape()[1]];
            let data: Vec<f64> = map.iter().copied().collect();
            unc.new_dataset::<f64>()
                .shape(shape)
                .chunk(shape)
                .deflate(4)
                .create(name.as_str())
                .and_then(|ds| ds.write_raw(&data))
                .map_err(|e| hdf5_err(&format!("/results/uncertainty/{name}"), e))?;
        }
    }

    if let Some(ref chi2) = snap.chi_squared_map {
        let shape = [chi2.shape()[0], chi2.shape()[1]];
        let data: Vec<f64> = chi2.iter().copied().collect();
        results
            .new_dataset::<f64>()
            .shape(shape)
            .chunk(shape)
            .deflate(4)
            .create("chi_squared")
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| hdf5_err("/results/chi_squared", e))?;
    }

    if let Some(ref conv) = snap.converged_map {
        let shape = [conv.shape()[0], conv.shape()[1]];
        let data: Vec<u8> = conv.iter().map(|&b| u8::from(b)).collect();
        results
            .new_dataset::<u8>()
            .shape(shape)
            .chunk(shape)
            .deflate(4)
            .create("converged")
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| hdf5_err("/results/converged", e))?;
    }

    if let Some(ref t_map) = snap.temperature_map {
        let shape = [t_map.shape()[0], t_map.shape()[1]];
        let data: Vec<f64> = t_map.iter().copied().collect();
        results
            .new_dataset::<f64>()
            .shape(shape)
            .chunk(shape)
            .deflate(4)
            .create("temperature")
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| hdf5_err("/results/temperature", e))?;
    }

    // Save anorm and background maps from spatial fitting.
    if let Some(ref a_map) = snap.anorm_map {
        let shape = [a_map.shape()[0], a_map.shape()[1]];
        let data: Vec<f64> = a_map.iter().copied().collect();
        results
            .new_dataset::<f64>()
            .shape(shape)
            .chunk(shape)
            .deflate(4)
            .create("anorm")
            .and_then(|ds| ds.write_raw(&data))
            .map_err(|e| hdf5_err("/results/anorm", e))?;
    }

    if let Some(ref bg_maps) = snap.background_maps {
        let bg_grp = results
            .create_group("background")
            .map_err(|e| hdf5_err("create /results/background", e))?;
        for (i, &label) in ["back_a", "back_b", "back_c"].iter().enumerate() {
            let m = &bg_maps[i];
            let shape = [m.shape()[0], m.shape()[1]];
            let data: Vec<f64> = m.iter().copied().collect();
            bg_grp
                .new_dataset::<f64>()
                .shape(shape)
                .chunk(shape)
                .deflate(4)
                .create(label)
                .and_then(|ds| ds.write_raw(&data))
                .map_err(|e| hdf5_err(&format!("/results/background/{label}"), e))?;
        }
    }

    if let Some(nc) = snap.n_converged {
        write_u64_attr(&results, "n_converged", nc as u64)?;
    }
    if let Some(nt) = snap.n_total {
        write_u64_attr(&results, "n_total", nt as u64)?;
    }

    if let Some(ref labels) = snap.result_isotope_labels
        && !labels.is_empty()
    {
        let vlu: Vec<VarLenUnicode> = labels
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|e| hdf5_err("parse VarLenUnicode label", e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        results
            .new_dataset::<VarLenUnicode>()
            .shape([labels.len()])
            .create("result_isotopes")
            .and_then(|ds| ds.write_raw(&vlu))
            .map_err(|e| hdf5_err("/results/result_isotopes", e))?;
    }

    // Single-pixel fit results (optional)
    if let Some(ref densities) = snap.single_fit_densities {
        let sf = results
            .create_group("single_fit")
            .map_err(|e| hdf5_err("create /results/single_fit", e))?;
        sf.new_dataset::<f64>()
            .shape([densities.len()])
            .create("densities")
            .and_then(|ds| ds.write_raw(densities))
            .map_err(|e| hdf5_err("/results/single_fit/densities", e))?;
        if let Some(ref unc) = snap.single_fit_uncertainties {
            sf.new_dataset::<f64>()
                .shape([unc.len()])
                .create("uncertainties")
                .and_then(|ds| ds.write_raw(unc))
                .map_err(|e| hdf5_err("/results/single_fit/uncertainties", e))?;
        }
        if let Some(chi2) = snap.single_fit_chi_squared {
            write_f64_attr(&sf, "chi_squared", chi2)?;
        }
        if let Some(temp) = snap.single_fit_temperature {
            write_f64_attr(&sf, "temperature_k", temp)?;
        }
        if let Some(temp_unc) = snap.single_fit_temperature_unc {
            write_f64_attr(&sf, "temperature_k_unc", temp_unc)?;
        }
        if let Some(iterations) = snap.single_fit_iterations {
            write_u32_attr(&sf, "iterations", iterations as u32)?;
        }
        if let Some(conv) = snap.single_fit_converged {
            write_bool_attr(&sf, "converged", conv)?;
        }
        if let Some((py, px)) = snap.single_fit_pixel {
            write_u32_attr(&sf, "pixel_y", py as u32)?;
            write_u32_attr(&sf, "pixel_x", px as u32)?;
        }
        if let Some(ref labels) = snap.single_fit_labels {
            let vlu: Vec<VarLenUnicode> = labels
                .iter()
                .map(|s| s.parse().map_err(|e| hdf5_err("parse single_fit label", e)))
                .collect::<Result<Vec<_>, _>>()?;
            sf.new_dataset::<VarLenUnicode>()
                .shape([labels.len()])
                .create("isotope_labels")
                .and_then(|ds| ds.write_raw(&vlu))
                .map_err(|e| hdf5_err("/results/single_fit/isotope_labels", e))?;
        }
        if let Some(anorm) = snap.single_fit_anorm {
            write_f64_attr(&sf, "anorm", anorm)?;
        }
        if let Some(bg) = snap.single_fit_background {
            sf.new_dataset::<f64>()
                .shape([3])
                .create("background")
                .and_then(|ds| ds.write_raw(&bg))
                .map_err(|e| hdf5_err("/results/single_fit/background", e))?;
        }
    }

    Ok(())
}

fn write_endf_cache(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let cache = file
        .create_group("endf_cache")
        .map_err(|e| hdf5_err("create /endf_cache", e))?;

    let mut written = std::collections::HashSet::new();
    for (symbol, rd) in &snap.endf_cache {
        if !written.insert(symbol.clone()) {
            continue; // skip duplicate symbol — first entry wins
        }
        let iso_group = cache
            .create_group(symbol)
            .map_err(|e| hdf5_err(&format!("create /endf_cache/{symbol}"), e))?;

        let json = serde_json::to_string(rd)
            .map_err(|e| hdf5_err(&format!("serialize /endf_cache/{symbol}"), e))?;
        let vlu: VarLenUnicode = json.parse().map_err(|e| hdf5_err(symbol, e))?;
        iso_group
            .new_dataset::<VarLenUnicode>()
            .shape(())
            .create("resonance_data")
            .and_then(|ds| ds.write_scalar(&vlu))
            .map_err(|e| hdf5_err(&format!("/endf_cache/{symbol}/resonance_data"), e))?;
    }

    Ok(())
}

fn write_provenance(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let prov = file
        .create_group("provenance")
        .map_err(|e| hdf5_err("create /provenance", e))?;

    if snap.provenance.is_empty() {
        return Ok(());
    }

    let n = snap.provenance.len();
    let timestamps: Vec<VarLenUnicode> = snap
        .provenance
        .iter()
        .map(|(ts, _, _)| {
            ts.parse()
                .map_err(|e| hdf5_err("parse VarLenUnicode timestamp", e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let kinds: Vec<VarLenUnicode> = snap
        .provenance
        .iter()
        .map(|(_, k, _)| {
            k.parse()
                .map_err(|e| hdf5_err("parse VarLenUnicode kind", e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let messages: Vec<VarLenUnicode> = snap
        .provenance
        .iter()
        .map(|(_, _, m)| {
            m.parse()
                .map_err(|e| hdf5_err("parse VarLenUnicode message", e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    prov.new_dataset::<VarLenUnicode>()
        .shape([n])
        .create("timestamps")
        .and_then(|ds| ds.write_raw(&timestamps))
        .map_err(|e| hdf5_err("/provenance/timestamps", e))?;

    prov.new_dataset::<VarLenUnicode>()
        .shape([n])
        .create("kinds")
        .and_then(|ds| ds.write_raw(&kinds))
        .map_err(|e| hdf5_err("/provenance/kinds", e))?;

    prov.new_dataset::<VarLenUnicode>()
        .shape([n])
        .create("messages")
        .and_then(|ds| ds.write_raw(&messages))
        .map_err(|e| hdf5_err("/provenance/messages", e))?;

    Ok(())
}

/// Pick a reasonable chunk shape for a 3D dataset.
fn chunk_shape_3d(shape: [usize; 3]) -> [usize; 3] {
    // One full frame per chunk, capped at 256 frames.
    // Guard zero dimensions — HDF5 rejects zero-sized chunks.
    let frames = shape[0].clamp(1, 256);
    [frames, shape[1].max(1), shape[2].max(1)]
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

fn read_str_attr(loc: &hdf5::Group, name: &str) -> Result<String, IoError> {
    let val: VarLenUnicode = loc
        .attr(name)
        .and_then(|a| a.read_scalar())
        .map_err(|e| hdf5_err(name, e))?;
    Ok(val.as_str().to_string())
}

fn read_f64_attr(loc: &hdf5::Group, name: &str) -> Result<f64, IoError> {
    loc.attr(name)
        .and_then(|a| a.read_scalar())
        .map_err(|e| hdf5_err(name, e))
}

fn read_u32_attr(loc: &hdf5::Group, name: &str) -> Result<u32, IoError> {
    loc.attr(name)
        .and_then(|a| a.read_scalar())
        .map_err(|e| hdf5_err(name, e))
}

fn read_bool_attr(loc: &hdf5::Group, name: &str) -> Result<bool, IoError> {
    let v: u8 = loc
        .attr(name)
        .and_then(|a| a.read_scalar())
        .map_err(|e| hdf5_err(name, e))?;
    Ok(v != 0)
}

fn read_u64_attr(loc: &hdf5::Group, name: &str) -> Result<u64, IoError> {
    loc.attr(name)
        .and_then(|a| a.read_scalar())
        .map_err(|e| hdf5_err(name, e))
}

/// Return `None` if the attribute does not exist.
fn read_str_attr_opt(loc: &hdf5::Group, name: &str) -> Option<String> {
    loc.attr(name)
        .and_then(|a| a.read_scalar::<VarLenUnicode>())
        .ok()
        .map(|v| v.as_str().to_string())
}

/// Return `None` if the attribute does not exist.
fn read_f64_attr_opt(loc: &hdf5::Group, name: &str) -> Option<f64> {
    loc.attr(name).and_then(|a| a.read_scalar::<f64>()).ok()
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Load a project snapshot from an HDF5 file at `path`.
pub fn load_project(path: &Path) -> Result<ProjectSnapshot, IoError> {
    let file = hdf5::File::open(path).map_err(|e| IoError::Hdf5Error(format!("open: {e}")))?;

    let mut snap = read_meta(&file)?;
    read_config(&file, &mut snap)?;
    read_data_links(&file, &mut snap)?;
    read_intermediate(&file, &mut snap)?;
    read_results(&file, &mut snap)?;
    read_endf_cache_into(&file, &mut snap)?;
    read_provenance_into(&file, &mut snap)?;

    Ok(snap)
}

fn read_meta(file: &hdf5::File) -> Result<ProjectSnapshot, IoError> {
    let g = file.group("meta").map_err(|_| {
        IoError::Hdf5Error("Not a valid NEREIDS project file: missing schema version".to_string())
    })?;

    let schema_version = read_str_attr(&g, "version").map_err(|_| {
        IoError::Hdf5Error("Not a valid NEREIDS project file: missing schema version".to_string())
    })?;
    let created_utc = read_str_attr(&g, "created_utc")?;
    let software_version = read_str_attr(&g, "software_version")?;
    let fitting_type = read_str_attr(&g, "fitting_type")?;
    let data_type = read_str_attr(&g, "data_type")?;

    Ok(ProjectSnapshot {
        schema_version,
        created_utc,
        software_version,
        fitting_type,
        data_type,
        ..Default::default()
    })
}

fn read_config(file: &hdf5::File, snap: &mut ProjectSnapshot) -> Result<(), IoError> {
    let config = file
        .group("config")
        .map_err(|e| hdf5_err("open /config", e))?;

    // Beamline
    let bl = config
        .group("beamline")
        .map_err(|e| hdf5_err("open /config/beamline", e))?;
    snap.flight_path_m = read_f64_attr(&bl, "flight_path_m")?;
    snap.delay_us = read_f64_attr(&bl, "delay_us")?;
    snap.proton_charge_sample = read_f64_attr(&bl, "proton_charge_sample")?;
    snap.proton_charge_ob = read_f64_attr(&bl, "proton_charge_ob")?;

    // Isotopes
    let iso = config
        .group("isotopes")
        .map_err(|e| hdf5_err("open /config/isotopes", e))?;
    if iso.dataset("z").is_ok() {
        let z_ds = iso
            .dataset("z")
            .map_err(|e| hdf5_err("/config/isotopes/z", e))?;
        snap.isotope_z = z_ds
            .read_raw()
            .map_err(|e| hdf5_err("/config/isotopes/z", e))?;

        let a_ds = iso
            .dataset("a")
            .map_err(|e| hdf5_err("/config/isotopes/a", e))?;
        snap.isotope_a = a_ds
            .read_raw()
            .map_err(|e| hdf5_err("/config/isotopes/a", e))?;

        let sym_ds = iso
            .dataset("symbol")
            .map_err(|e| hdf5_err("/config/isotopes/symbol", e))?;
        let symbols: Vec<VarLenUnicode> = sym_ds
            .read_raw()
            .map_err(|e| hdf5_err("/config/isotopes/symbol", e))?;
        snap.isotope_symbol = symbols.iter().map(|v| v.as_str().to_string()).collect();

        let d_ds = iso
            .dataset("density")
            .map_err(|e| hdf5_err("/config/isotopes/density", e))?;
        snap.isotope_density = d_ds
            .read_raw()
            .map_err(|e| hdf5_err("/config/isotopes/density", e))?;

        let en_ds = iso
            .dataset("enabled")
            .map_err(|e| hdf5_err("/config/isotopes/enabled", e))?;
        let en_raw: Vec<u8> = en_ds
            .read_raw()
            .map_err(|e| hdf5_err("/config/isotopes/enabled", e))?;
        snap.isotope_enabled = en_raw.iter().map(|&v| v != 0).collect();
    }

    // Solver
    let solver = config
        .group("solver")
        .map_err(|e| hdf5_err("open /config/solver", e))?;
    snap.solver_method = read_str_attr(&solver, "method")?;
    snap.max_iter = read_u32_attr(&solver, "max_iter")?;
    snap.temperature_k = read_f64_attr(&solver, "temperature_k")?;
    snap.fit_temperature = read_bool_attr(&solver, "fit_temperature")?;

    // Resolution
    let res = config
        .group("resolution")
        .map_err(|e| hdf5_err("open /config/resolution", e))?;
    snap.resolution_enabled = read_bool_attr(&res, "enabled")?;
    snap.resolution_kind = read_str_attr(&res, "kind")?;
    snap.delta_t_us = read_f64_attr_opt(&res, "delta_t_us");
    snap.delta_l_m = read_f64_attr_opt(&res, "delta_l_m");
    snap.tabulated_path = read_str_attr_opt(&res, "tabulated_path");

    // ROIs
    if let Ok(roi_ds) = config.dataset("rois") {
        let shape = roi_ds.shape();
        let flat: Vec<u64> = roi_ds.read_raw().map_err(|e| hdf5_err("/config/rois", e))?;
        if shape.len() == 2 && shape[1] == 4 {
            snap.rois = flat.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        }
    }

    // ENDF library
    snap.endf_library = read_str_attr(&config, "endf_library")?;

    Ok(())
}

fn read_data_links(file: &hdf5::File, snap: &mut ProjectSnapshot) -> Result<(), IoError> {
    let data = file.group("data").map_err(|e| hdf5_err("open /data", e))?;
    snap.data_mode = read_str_attr(&data, "mode")?;
    snap.spectrum_unit = read_str_attr(&data, "spectrum_unit")?;
    snap.spectrum_kind = read_str_attr(&data, "spectrum_kind")?;
    snap.rebin_factor = read_u32_attr(&data, "rebin_factor")?;
    snap.rebin_applied = read_bool_attr(&data, "rebin_applied")?;

    let links = data
        .group("links")
        .map_err(|e| hdf5_err("open /data/links", e))?;
    snap.sample_path = read_str_attr_opt(&links, "sample_path");
    snap.open_beam_path = read_str_attr_opt(&links, "open_beam_path");
    snap.spectrum_path = read_str_attr_opt(&links, "spectrum_path");
    snap.hdf5_path = read_str_attr_opt(&links, "hdf5_path");

    // Read embedded data if mode is "embedded"
    if snap.data_mode == "embedded" {
        read_embedded_data(&data, snap)?;
    }

    Ok(())
}

fn read_embedded_data(data_group: &hdf5::Group, snap: &mut ProjectSnapshot) -> Result<(), IoError> {
    let embedded = data_group
        .group("embedded")
        .map_err(|e| hdf5_err("open /data/embedded (file claims embedded mode)", e))?;

    if let Ok(ds) = embedded.dataset("sample") {
        let shape = ds.shape();
        if shape.len() != 3 {
            return Err(hdf5_err(
                "/data/embedded/sample",
                format!("expected 3D, got {}D", shape.len()),
            ));
        }
        let data: Vec<f64> = ds
            .read_raw()
            .map_err(|e| hdf5_err("/data/embedded/sample", e))?;
        snap.sample_data = Some(
            Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
                .map_err(|e| hdf5_err("/data/embedded/sample reshape", e))?,
        );
    }

    if let Ok(ds) = embedded.dataset("open_beam") {
        let shape = ds.shape();
        if shape.len() != 3 {
            return Err(hdf5_err(
                "/data/embedded/open_beam",
                format!("expected 3D, got {}D", shape.len()),
            ));
        }
        let data: Vec<f64> = ds
            .read_raw()
            .map_err(|e| hdf5_err("/data/embedded/open_beam", e))?;
        snap.open_beam_data = Some(
            Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
                .map_err(|e| hdf5_err("/data/embedded/open_beam reshape", e))?,
        );
    }

    if let Ok(ds) = embedded.dataset("spectrum") {
        let data: Vec<f64> = ds
            .read_raw()
            .map_err(|e| hdf5_err("/data/embedded/spectrum", e))?;
        snap.spectrum_values = Some(data);
    }

    Ok(())
}

fn read_intermediate(file: &hdf5::File, snap: &mut ProjectSnapshot) -> Result<(), IoError> {
    let inter = file
        .group("intermediate")
        .map_err(|e| hdf5_err("open /intermediate", e))?;

    if let Ok(norm_ds) = inter.dataset("normalized") {
        let shape = norm_ds.shape();
        if shape.len() == 3 {
            let data: Vec<f64> = norm_ds
                .read_raw()
                .map_err(|e| hdf5_err("/intermediate/normalized", e))?;
            snap.normalized = Some(
                Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
                    .map_err(|e| hdf5_err("/intermediate/normalized reshape", e))?,
            );
        }
    }

    // D-1: Load per-bin transmission uncertainty.
    if let Ok(unc_ds) = inter.dataset("normalized_uncertainty") {
        let shape = unc_ds.shape();
        if shape.len() == 3 {
            let data: Vec<f64> = unc_ds
                .read_raw()
                .map_err(|e| hdf5_err("/intermediate/normalized_uncertainty", e))?;
            snap.normalized_uncertainty = Some(
                Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
                    .map_err(|e| hdf5_err("/intermediate/normalized_uncertainty reshape", e))?,
            );
        }
    }

    if let Ok(e_ds) = inter.dataset("energies") {
        let data: Vec<f64> = e_ds
            .read_raw()
            .map_err(|e| hdf5_err("/intermediate/energies", e))?;
        snap.energies = Some(data);
    }

    // D-20: Load dead-pixel mask (u8 → bool).
    if let Ok(dp_ds) = inter.dataset("dead_pixels") {
        let shape = dp_ds.shape();
        if shape.len() == 2 {
            let data: Vec<u8> = dp_ds
                .read_raw()
                .map_err(|e| hdf5_err("/intermediate/dead_pixels", e))?;
            let bools: Vec<bool> = data.iter().map(|&v| v != 0).collect();
            snap.dead_pixels = Some(
                Array2::from_shape_vec((shape[0], shape[1]), bools)
                    .map_err(|e| hdf5_err("/intermediate/dead_pixels reshape", e))?,
            );
        }
    }

    Ok(())
}

fn read_results(file: &hdf5::File, snap: &mut ProjectSnapshot) -> Result<(), IoError> {
    let results = file
        .group("results")
        .map_err(|e| hdf5_err("open /results", e))?;

    // Read isotope labels first so we can use them to order density/uncertainty maps
    if let Ok(labels_ds) = results.dataset("result_isotopes") {
        let labels_vlu: Vec<VarLenUnicode> = labels_ds
            .read_raw()
            .map_err(|e| hdf5_err("/results/result_isotopes", e))?;
        let labels: Vec<String> = labels_vlu.iter().map(|v| v.as_str().to_string()).collect();
        snap.result_isotope_labels = Some(labels);
    }

    // Density maps
    if let Ok(density_grp) = results.group("density") {
        let names = density_grp
            .member_names()
            .map_err(|e| hdf5_err("/results/density member_names", e))?;

        // Order by result_isotope_labels if available, otherwise alphabetical.
        // Append any datasets not in labels so data isn't lost on corrupted files.
        let ordered: Vec<String> = if let Some(ref labels) = snap.result_isotope_labels {
            let mut ordered: Vec<String> = labels
                .iter()
                .filter(|l| names.contains(l))
                .cloned()
                .collect();
            let mut remaining: Vec<String> =
                names.into_iter().filter(|n| !labels.contains(n)).collect();
            remaining.sort();
            ordered.extend(remaining);
            ordered
        } else {
            let mut sorted = names;
            sorted.sort();
            sorted
        };

        let mut maps = Vec::with_capacity(ordered.len());
        for name in &ordered {
            let ds = density_grp
                .dataset(name)
                .map_err(|e| hdf5_err(&format!("/results/density/{name}"), e))?;
            let shape = ds.shape();
            if shape.len() == 2 {
                let data: Vec<f64> = ds
                    .read_raw()
                    .map_err(|e| hdf5_err(&format!("/results/density/{name}"), e))?;
                maps.push(
                    Array2::from_shape_vec((shape[0], shape[1]), data)
                        .map_err(|e| hdf5_err(&format!("/results/density/{name} reshape"), e))?,
                );
            }
        }
        if !maps.is_empty() {
            snap.density_maps = Some(maps);
        }
    }

    // Uncertainty maps
    if let Ok(unc_grp) = results.group("uncertainty") {
        let names = unc_grp
            .member_names()
            .map_err(|e| hdf5_err("/results/uncertainty member_names", e))?;

        let ordered: Vec<String> = if let Some(ref labels) = snap.result_isotope_labels {
            let mut ordered: Vec<String> = labels
                .iter()
                .filter(|l| names.contains(l))
                .cloned()
                .collect();
            let mut remaining: Vec<String> =
                names.into_iter().filter(|n| !labels.contains(n)).collect();
            remaining.sort();
            ordered.extend(remaining);
            ordered
        } else {
            let mut sorted = names;
            sorted.sort();
            sorted
        };

        let mut maps = Vec::with_capacity(ordered.len());
        for name in &ordered {
            let ds = unc_grp
                .dataset(name)
                .map_err(|e| hdf5_err(&format!("/results/uncertainty/{name}"), e))?;
            let shape = ds.shape();
            if shape.len() == 2 {
                let data: Vec<f64> = ds
                    .read_raw()
                    .map_err(|e| hdf5_err(&format!("/results/uncertainty/{name}"), e))?;
                maps.push(
                    Array2::from_shape_vec((shape[0], shape[1]), data).map_err(|e| {
                        hdf5_err(&format!("/results/uncertainty/{name} reshape"), e)
                    })?,
                );
            }
        }
        if !maps.is_empty() {
            snap.uncertainty_maps = Some(maps);
        }
    }

    // Chi-squared map
    if let Ok(chi2_ds) = results.dataset("chi_squared") {
        let shape = chi2_ds.shape();
        if shape.len() == 2 {
            let data: Vec<f64> = chi2_ds
                .read_raw()
                .map_err(|e| hdf5_err("/results/chi_squared", e))?;
            snap.chi_squared_map = Some(
                Array2::from_shape_vec((shape[0], shape[1]), data)
                    .map_err(|e| hdf5_err("/results/chi_squared reshape", e))?,
            );
        }
    }

    // Converged map
    if let Ok(conv_ds) = results.dataset("converged") {
        let shape = conv_ds.shape();
        if shape.len() == 2 {
            let data: Vec<u8> = conv_ds
                .read_raw()
                .map_err(|e| hdf5_err("/results/converged", e))?;
            snap.converged_map = Some(
                Array2::from_shape_vec(
                    (shape[0], shape[1]),
                    data.iter().map(|&v| v != 0).collect(),
                )
                .map_err(|e| hdf5_err("/results/converged reshape", e))?,
            );
        }
    }

    // Temperature map
    if let Ok(t_ds) = results.dataset("temperature") {
        let shape = t_ds.shape();
        if shape.len() == 2 {
            let data: Vec<f64> = t_ds
                .read_raw()
                .map_err(|e| hdf5_err("/results/temperature", e))?;
            snap.temperature_map = Some(
                Array2::from_shape_vec((shape[0], shape[1]), data)
                    .map_err(|e| hdf5_err("/results/temperature reshape", e))?,
            );
        }
    }

    // D-11/D-21: Anorm map
    if let Ok(a_ds) = results.dataset("anorm") {
        let shape = a_ds.shape();
        if shape.len() == 2 {
            let data: Vec<f64> = a_ds.read_raw().map_err(|e| hdf5_err("/results/anorm", e))?;
            snap.anorm_map = Some(
                Array2::from_shape_vec((shape[0], shape[1]), data)
                    .map_err(|e| hdf5_err("/results/anorm reshape", e))?,
            );
        }
    }

    // D-11/D-21: Background maps
    if let Ok(bg_grp) = results.group("background") {
        let mut maps: [Option<Array2<f64>>; 3] = [None, None, None];
        for (i, &label) in ["back_a", "back_b", "back_c"].iter().enumerate() {
            if let Ok(ds) = bg_grp.dataset(label) {
                let shape = ds.shape();
                if shape.len() == 2 {
                    let data: Vec<f64> = ds
                        .read_raw()
                        .map_err(|e| hdf5_err(&format!("/results/background/{label}"), e))?;
                    maps[i] = Some(Array2::from_shape_vec((shape[0], shape[1]), data).map_err(
                        |e| hdf5_err(&format!("/results/background/{label} reshape"), e),
                    )?);
                }
            }
        }
        // Only set if all three are present.
        if maps.iter().all(|m| m.is_some()) {
            snap.background_maps = Some([
                maps[0].take().unwrap(),
                maps[1].take().unwrap(),
                maps[2].take().unwrap(),
            ]);
        }
    }

    // Scalar attrs
    if let Ok(nc) = read_u64_attr(&results, "n_converged") {
        snap.n_converged = Some(nc as usize);
    }
    if let Ok(nt) = read_u64_attr(&results, "n_total") {
        snap.n_total = Some(nt as usize);
    }

    // Single-pixel fit results
    if let Ok(sf) = results.group("single_fit") {
        if let Ok(ds) = sf.dataset("densities") {
            let data: Vec<f64> = ds
                .read_raw()
                .map_err(|e| hdf5_err("/results/single_fit/densities", e))?;
            snap.single_fit_densities = Some(data);
        }
        if let Ok(ds) = sf.dataset("uncertainties") {
            let data: Vec<f64> = ds
                .read_raw()
                .map_err(|e| hdf5_err("/results/single_fit/uncertainties", e))?;
            snap.single_fit_uncertainties = Some(data);
        }
        snap.single_fit_chi_squared = read_f64_attr(&sf, "chi_squared").ok();
        snap.single_fit_temperature = read_f64_attr(&sf, "temperature_k").ok();
        snap.single_fit_temperature_unc = read_f64_attr(&sf, "temperature_k_unc").ok();
        snap.single_fit_converged = read_bool_attr(&sf, "converged").ok();
        snap.single_fit_iterations = read_u32_attr(&sf, "iterations").ok().map(|v| v as usize);
        if let (Ok(py), Ok(px)) = (read_u32_attr(&sf, "pixel_y"), read_u32_attr(&sf, "pixel_x")) {
            snap.single_fit_pixel = Some((py as usize, px as usize));
        }
        if let Ok(ds) = sf.dataset("isotope_labels") {
            let vlu: Vec<VarLenUnicode> = ds
                .read_raw()
                .map_err(|e| hdf5_err("/results/single_fit/isotope_labels", e))?;
            snap.single_fit_labels = Some(vlu.iter().map(|v| v.as_str().to_string()).collect());
        }
        snap.single_fit_anorm = read_f64_attr(&sf, "anorm").ok();
        if let Ok(ds) = sf.dataset("background") {
            let data: Vec<f64> = ds
                .read_raw()
                .map_err(|e| hdf5_err("/results/single_fit/background", e))?;
            if data.len() == 3 {
                snap.single_fit_background = Some([data[0], data[1], data[2]]);
            }
        }
    }

    Ok(())
}

fn read_endf_cache_into(file: &hdf5::File, snap: &mut ProjectSnapshot) -> Result<(), IoError> {
    let cache = file
        .group("endf_cache")
        .map_err(|e| hdf5_err("open /endf_cache", e))?;

    let names = cache
        .member_names()
        .map_err(|e| hdf5_err("/endf_cache member_names", e))?;

    for name in &names {
        let iso_grp = cache
            .group(name)
            .map_err(|e| hdf5_err(&format!("/endf_cache/{name}"), e))?;
        let ds = iso_grp
            .dataset("resonance_data")
            .map_err(|e| hdf5_err(&format!("/endf_cache/{name}/resonance_data"), e))?;
        let json: VarLenUnicode = ds
            .read_scalar()
            .map_err(|e| hdf5_err(&format!("/endf_cache/{name}/resonance_data"), e))?;
        let rd: ResonanceData = serde_json::from_str(json.as_str())
            .map_err(|e| hdf5_err(&format!("deserialize /endf_cache/{name}"), e))?;
        snap.endf_cache.push((name.clone(), rd));
    }

    Ok(())
}

fn read_provenance_into(file: &hdf5::File, snap: &mut ProjectSnapshot) -> Result<(), IoError> {
    let prov = file
        .group("provenance")
        .map_err(|e| hdf5_err("open /provenance", e))?;

    let ts_ds = match prov.dataset("timestamps") {
        Ok(ds) => ds,
        Err(_) => return Ok(()), // empty provenance
    };
    let timestamps: Vec<VarLenUnicode> = ts_ds
        .read_raw()
        .map_err(|e| hdf5_err("/provenance/timestamps", e))?;
    let kinds: Vec<VarLenUnicode> = prov
        .dataset("kinds")
        .and_then(|ds| ds.read_raw())
        .map_err(|e| hdf5_err("/provenance/kinds", e))?;
    let messages: Vec<VarLenUnicode> = prov
        .dataset("messages")
        .and_then(|ds| ds.read_raw())
        .map_err(|e| hdf5_err("/provenance/messages", e))?;

    for (i, ts_vlu) in timestamps.iter().enumerate() {
        let ts = ts_vlu.as_str().to_string();
        let kind = kinds
            .get(i)
            .map_or(String::new(), |v| v.as_str().to_string());
        let msg = messages
            .get(i)
            .map_or(String::new(), |v| v.as_str().to_string());
        snap.provenance.push((ts, kind, msg));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::VarLenUnicode;
    use ndarray::{Array2, Array3};

    fn minimal_snapshot() -> ProjectSnapshot {
        ProjectSnapshot {
            schema_version: PROJECT_SCHEMA_VERSION.to_string(),
            created_utc: "2026-03-07T12:00:00Z".into(),
            software_version: "0.1.0".into(),
            fitting_type: "spatial".into(),
            data_type: "transmission".into(),
            flight_path_m: 15.3,
            delay_us: 0.0,
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
            isotope_z: vec![],
            isotope_a: vec![],
            isotope_symbol: vec![],
            isotope_density: vec![],
            isotope_enabled: vec![],
            solver_method: "lm".into(),
            max_iter: 20,
            temperature_k: 300.0,
            fit_temperature: false,
            resolution_enabled: false,
            resolution_kind: "gaussian".into(),
            delta_t_us: Some(1.5),
            delta_l_m: Some(0.003),
            tabulated_path: None,
            rois: vec![],
            endf_library: "ENDF/B-VIII.0".into(),
            data_mode: "linked".into(),
            sample_path: Some("/data/sample".into()),
            open_beam_path: Some("/data/ob".into()),
            spectrum_path: Some("/data/spectrum.txt".into()),
            hdf5_path: None,
            spectrum_unit: "tof_us".into(),
            spectrum_kind: "bin_edges".into(),
            rebin_factor: 1,
            rebin_applied: false,
            sample_data: None,
            open_beam_data: None,
            spectrum_values: None,
            normalized: None,
            normalized_uncertainty: None,
            energies: None,
            dead_pixels: None,
            density_maps: None,
            uncertainty_maps: None,
            chi_squared_map: None,
            converged_map: None,
            temperature_map: None,
            n_converged: None,
            n_total: None,
            result_isotope_labels: None,
            anorm_map: None,
            background_maps: None,
            single_fit_densities: None,
            single_fit_uncertainties: None,
            single_fit_chi_squared: None,
            single_fit_temperature: None,
            single_fit_temperature_unc: None,
            single_fit_converged: None,
            single_fit_iterations: None,
            single_fit_pixel: None,
            single_fit_labels: None,
            single_fit_anorm: None,
            single_fit_background: None,
            endf_cache: vec![],
            provenance: vec![],
        }
    }

    #[test]
    fn test_save_minimal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.nrd.h5");
        let snap = minimal_snapshot();
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        // Verify top-level groups exist
        assert!(file.group("meta").is_ok());
        assert!(file.group("config").is_ok());
        assert!(file.group("data").is_ok());
        assert!(file.group("intermediate").is_ok());
        assert!(file.group("results").is_ok());
        assert!(file.group("endf_cache").is_ok());
        assert!(file.group("provenance").is_ok());

        // Verify meta attrs
        let meta = file.group("meta").unwrap();
        let ver: VarLenUnicode = meta.attr("version").unwrap().read_scalar().unwrap();
        assert_eq!(ver.as_str(), "1.0");
        let ft: VarLenUnicode = meta.attr("fitting_type").unwrap().read_scalar().unwrap();
        assert_eq!(ft.as_str(), "spatial");
    }

    #[test]
    fn test_save_with_results() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.density_maps = Some(vec![Array2::from_elem((3, 4), 0.001)]);
        snap.uncertainty_maps = Some(vec![Array2::from_elem((3, 4), 0.0001)]);
        snap.chi_squared_map = Some(Array2::from_elem((3, 4), 1.5));
        snap.converged_map = Some(Array2::from_elem((3, 4), true));
        snap.n_converged = Some(12);
        snap.n_total = Some(12);
        snap.result_isotope_labels = Some(vec!["W-182".into()]);
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let results = file.group("results").unwrap();

        // Check density map
        let density = results.group("density").unwrap();
        let ds = density.dataset("W-182").unwrap();
        assert_eq!(ds.shape(), vec![3, 4]);
        let data: Vec<f64> = ds.read_raw().unwrap();
        assert!((data[0] - 0.001).abs() < 1e-10);

        // Check converged map
        let conv_ds = results.dataset("converged").unwrap();
        let conv: Vec<u8> = conv_ds.read_raw().unwrap();
        assert_eq!(conv[0], 1);

        // Check attrs
        let nc: u64 = results.attr("n_converged").unwrap().read_scalar().unwrap();
        assert_eq!(nc, 12);
    }

    #[test]
    fn test_save_with_intermediate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("inter.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.normalized = Some(Array3::from_elem((10, 3, 4), 0.5));
        snap.energies = Some(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let inter = file.group("intermediate").unwrap();
        let norm_ds = inter.dataset("normalized").unwrap();
        assert_eq!(norm_ds.shape(), vec![10, 3, 4]);

        let e_ds = inter.dataset("energies").unwrap();
        let e: Vec<f64> = e_ds.read_raw().unwrap();
        assert_eq!(e.len(), 5);
        assert!((e[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_save_endf_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("endf.nrd.h5");
        let mut snap = minimal_snapshot();

        // Create a minimal ResonanceData (empty ranges)
        let rd = ResonanceData {
            isotope: nereids_core::types::Isotope::new(74, 182).unwrap(),
            za: 74182,
            awr: 180.948,
            ranges: vec![],
        };
        snap.endf_cache = vec![("W-182".into(), rd)];
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let cache = file.group("endf_cache").unwrap();
        let w182 = cache.group("W-182").unwrap();
        let ds = w182.dataset("resonance_data").unwrap();
        let json: VarLenUnicode = ds.read_scalar().unwrap();

        // Round-trip: deserialize back
        let rd2: ResonanceData = serde_json::from_str(json.as_str()).unwrap();
        assert_eq!(rd2.za, 74182);
        assert!((rd2.awr - 180.948).abs() < 1e-6);
    }

    #[test]
    fn test_save_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prov.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.provenance = vec![
            (
                "2026-03-07T12:00:00Z".into(),
                "DataLoaded".into(),
                "Loaded sample".into(),
            ),
            (
                "2026-03-07T12:01:00Z".into(),
                "AnalysisRun".into(),
                "Spatial map done".into(),
            ),
        ];
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let prov = file.group("provenance").unwrap();
        let ts: Vec<VarLenUnicode> = prov.dataset("timestamps").unwrap().read_raw().unwrap();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[0].as_str(), "2026-03-07T12:00:00Z");

        let kinds: Vec<VarLenUnicode> = prov.dataset("kinds").unwrap().read_raw().unwrap();
        assert_eq!(kinds[1].as_str(), "AnalysisRun");
    }

    #[test]
    fn test_save_isotope_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iso.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.isotope_z = vec![74, 26];
        snap.isotope_a = vec![182, 56];
        snap.isotope_symbol = vec!["W-182".into(), "Fe-56".into()];
        snap.isotope_density = vec![0.001, 0.002];
        snap.isotope_enabled = vec![true, false];
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let iso = file.group("config/isotopes").unwrap();
        let z: Vec<u32> = iso.dataset("z").unwrap().read_raw().unwrap();
        assert_eq!(z, vec![74, 26]);
        let en: Vec<u8> = iso.dataset("enabled").unwrap().read_raw().unwrap();
        assert_eq!(en, vec![1, 0]);
    }

    #[test]
    fn test_save_rois() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roi.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.rois = vec![[10, 20, 30, 40], [50, 60, 70, 80]];
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let config = file.group("config").unwrap();
        let ds = config.dataset("rois").unwrap();
        assert_eq!(ds.shape(), vec![2, 4]);
        let data: Vec<u64> = ds.read_raw().unwrap();
        assert_eq!(data, vec![10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn test_save_temperature_map_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("temp.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.temperature_map = Some(Array2::from_elem((3, 4), 295.0));
        snap.density_maps = Some(vec![Array2::from_elem((3, 4), 0.001)]);
        snap.converged_map = Some(Array2::from_elem((3, 4), true));
        snap.n_converged = Some(12);
        snap.n_total = Some(12);
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let results = file.group("results").unwrap();
        let t_ds = results.dataset("temperature").unwrap();
        assert_eq!(t_ds.shape(), vec![3, 4]);
        let data: Vec<f64> = t_ds.read_raw().unwrap();
        assert!((data[0] - 295.0).abs() < 1e-10);
    }

    #[test]
    fn test_save_temperature_map_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_temp.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.density_maps = Some(vec![Array2::from_elem((3, 4), 0.001)]);
        snap.converged_map = Some(Array2::from_elem((3, 4), true));
        snap.n_converged = Some(12);
        snap.n_total = Some(12);
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let results = file.group("results").unwrap();
        assert!(results.dataset("temperature").is_err());
    }

    #[test]
    fn test_save_beamline_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bl.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.flight_path_m = 15.3;
        snap.delay_us = 42.5;
        save_project(&path, &snap).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let bl = file.group("config/beamline").unwrap();
        let fp: f64 = bl.attr("flight_path_m").unwrap().read_scalar().unwrap();
        let delay: f64 = bl.attr("delay_us").unwrap().read_scalar().unwrap();
        assert!((fp - 15.3).abs() < 1e-10);
        assert!((delay - 42.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_minimal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt.nrd.h5");
        let snap = minimal_snapshot();
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();

        assert_eq!(loaded.schema_version, snap.schema_version);
        assert_eq!(loaded.created_utc, snap.created_utc);
        assert_eq!(loaded.software_version, snap.software_version);
        assert_eq!(loaded.fitting_type, snap.fitting_type);
        assert_eq!(loaded.data_type, snap.data_type);
        assert!((loaded.flight_path_m - snap.flight_path_m).abs() < 1e-10);
        assert!((loaded.delay_us - snap.delay_us).abs() < 1e-10);
        assert!((loaded.proton_charge_sample - snap.proton_charge_sample).abs() < 1e-10);
        assert!((loaded.proton_charge_ob - snap.proton_charge_ob).abs() < 1e-10);
        assert_eq!(loaded.solver_method, snap.solver_method);
        assert_eq!(loaded.max_iter, snap.max_iter);
        assert!((loaded.temperature_k - snap.temperature_k).abs() < 1e-10);
        assert_eq!(loaded.fit_temperature, snap.fit_temperature);
        assert_eq!(loaded.resolution_enabled, snap.resolution_enabled);
        assert_eq!(loaded.resolution_kind, snap.resolution_kind);
        assert_eq!(loaded.endf_library, snap.endf_library);
        assert_eq!(loaded.data_mode, snap.data_mode);
        assert_eq!(loaded.sample_path, snap.sample_path);
        assert_eq!(loaded.open_beam_path, snap.open_beam_path);
        assert_eq!(loaded.spectrum_path, snap.spectrum_path);
        assert_eq!(loaded.hdf5_path, snap.hdf5_path);
        assert_eq!(loaded.spectrum_unit, snap.spectrum_unit);
        assert_eq!(loaded.spectrum_kind, snap.spectrum_kind);
        assert_eq!(loaded.rebin_factor, snap.rebin_factor);
        assert_eq!(loaded.rebin_applied, snap.rebin_applied);
    }

    #[test]
    fn test_roundtrip_with_results() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt_results.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.density_maps = Some(vec![
            Array2::from_elem((3, 4), 0.001),
            Array2::from_elem((3, 4), 0.002),
        ]);
        snap.uncertainty_maps = Some(vec![
            Array2::from_elem((3, 4), 0.0001),
            Array2::from_elem((3, 4), 0.0002),
        ]);
        snap.chi_squared_map = Some(Array2::from_elem((3, 4), 1.5));
        snap.converged_map = Some(Array2::from_elem((3, 4), true));
        snap.temperature_map = Some(Array2::from_elem((3, 4), 295.0));
        snap.n_converged = Some(12);
        snap.n_total = Some(12);
        snap.result_isotope_labels = Some(vec!["W-182".into(), "Fe-56".into()]);
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();

        let dm = loaded.density_maps.unwrap();
        assert_eq!(dm.len(), 2);
        assert!((dm[0][[0, 0]] - 0.001).abs() < 1e-10);
        assert!((dm[1][[0, 0]] - 0.002).abs() < 1e-10);

        let um = loaded.uncertainty_maps.unwrap();
        assert_eq!(um.len(), 2);

        let chi2 = loaded.chi_squared_map.unwrap();
        assert!((chi2[[0, 0]] - 1.5).abs() < 1e-10);

        let conv = loaded.converged_map.unwrap();
        assert!(conv[[0, 0]]);

        let temp = loaded.temperature_map.unwrap();
        assert!((temp[[0, 0]] - 295.0).abs() < 1e-10);

        assert_eq!(loaded.n_converged, Some(12));
        assert_eq!(loaded.n_total, Some(12));
        assert_eq!(
            loaded.result_isotope_labels,
            Some(vec!["W-182".into(), "Fe-56".into()])
        );
    }

    #[test]
    fn test_roundtrip_with_intermediate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt_inter.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.normalized = Some(Array3::from_elem((10, 3, 4), 0.5));
        snap.energies = Some(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();

        let norm = loaded.normalized.unwrap();
        assert_eq!(norm.shape(), &[10, 3, 4]);
        assert!((norm[[0, 0, 0]] - 0.5).abs() < 1e-10);

        let en = loaded.energies.unwrap();
        assert_eq!(en.len(), 5);
        assert!((en[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_endf_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt_endf.nrd.h5");
        let mut snap = minimal_snapshot();
        let rd = ResonanceData {
            isotope: nereids_core::types::Isotope::new(74, 182).unwrap(),
            za: 74182,
            awr: 180.948,
            ranges: vec![],
        };
        snap.endf_cache = vec![("W-182".into(), rd)];
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();

        assert_eq!(loaded.endf_cache.len(), 1);
        assert_eq!(loaded.endf_cache[0].0, "W-182");
        assert_eq!(loaded.endf_cache[0].1.za, 74182);
        assert!((loaded.endf_cache[0].1.awr - 180.948).abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt_prov.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.provenance = vec![
            (
                "2026-03-07 12:00:00 UTC".into(),
                "DataLoaded".into(),
                "Loaded sample".into(),
            ),
            (
                "2026-03-07 12:01:00 UTC".into(),
                "AnalysisRun".into(),
                "Spatial map done".into(),
            ),
        ];
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();

        assert_eq!(loaded.provenance.len(), 2);
        assert_eq!(loaded.provenance[0].0, "2026-03-07 12:00:00 UTC");
        assert_eq!(loaded.provenance[0].1, "DataLoaded");
        assert_eq!(loaded.provenance[0].2, "Loaded sample");
        assert_eq!(loaded.provenance[1].1, "AnalysisRun");
    }

    #[test]
    fn test_roundtrip_isotope_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt_iso.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.isotope_z = vec![74, 26];
        snap.isotope_a = vec![182, 56];
        snap.isotope_symbol = vec!["W-182".into(), "Fe-56".into()];
        snap.isotope_density = vec![0.001, 0.002];
        snap.isotope_enabled = vec![true, false];
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();

        assert_eq!(loaded.isotope_z, vec![74, 26]);
        assert_eq!(loaded.isotope_a, vec![182, 56]);
        assert_eq!(loaded.isotope_symbol, vec!["W-182", "Fe-56"]);
        assert!((loaded.isotope_density[0] - 0.001).abs() < 1e-10);
        assert_eq!(loaded.isotope_enabled, vec![true, false]);
    }

    #[test]
    fn test_roundtrip_rois() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt_rois.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.rois = vec![[10, 20, 30, 40], [50, 60, 70, 80]];
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();

        assert_eq!(loaded.rois.len(), 2);
        assert_eq!(loaded.rois[0], [10, 20, 30, 40]);
        assert_eq!(loaded.rois[1], [50, 60, 70, 80]);
    }

    #[test]
    fn test_load_missing_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.h5");
        // Create an HDF5 file with no /meta group
        hdf5::File::create(&path).unwrap();
        let result = load_project(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("schema") || err.contains("meta") || err.contains("version"),
            "Error should mention missing schema: {err}"
        );
    }

    #[test]
    fn test_load_future_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.schema_version = "99.0".into();
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();
        assert_eq!(loaded.schema_version, "99.0");
    }

    #[test]
    fn test_roundtrip_empty_isotopes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt_empty_iso.nrd.h5");
        let snap = minimal_snapshot(); // has empty isotope arrays
        save_project(&path, &snap).unwrap();
        let loaded = load_project(&path).unwrap();
        assert!(loaded.isotope_z.is_empty());
        assert!(loaded.isotope_a.is_empty());
        assert!(loaded.isotope_symbol.is_empty());
        assert!(loaded.isotope_density.is_empty());
        assert!(loaded.isotope_enabled.is_empty());
    }

    // -- embedded mode tests --

    #[test]
    fn test_roundtrip_embedded_sample_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embed_sample.nrd.h5");
        let snap = minimal_snapshot();
        let sample = Array3::from_shape_fn((5, 3, 4), |(t, y, x)| (t * 12 + y * 4 + x) as f64);
        let spectrum = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let emb = EmbeddedData {
            sample: Some(&sample),
            open_beam: None,
            spectrum: Some(&spectrum),
        };
        save_project_with_data(&path, &snap, Some(&emb)).unwrap();
        let loaded = load_project(&path).unwrap();

        assert_eq!(loaded.data_mode, "embedded");
        let loaded_sample = loaded.sample_data.unwrap();
        assert_eq!(loaded_sample.shape(), [5, 3, 4]);
        assert_eq!(loaded_sample, sample);
        assert!(loaded.open_beam_data.is_none());
        assert_eq!(loaded.spectrum_values.unwrap(), spectrum);
    }

    #[test]
    fn test_roundtrip_embedded_full() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embed_full.nrd.h5");
        let snap = minimal_snapshot();
        let sample = Array3::from_shape_fn((4, 2, 3), |(t, y, x)| (t * 6 + y * 3 + x) as f64 + 0.5);
        let ob = Array3::from_shape_fn((4, 2, 3), |(t, y, x)| (t * 6 + y * 3 + x) as f64 * 2.0);
        let spectrum = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let emb = EmbeddedData {
            sample: Some(&sample),
            open_beam: Some(&ob),
            spectrum: Some(&spectrum),
        };
        save_project_with_data(&path, &snap, Some(&emb)).unwrap();
        let loaded = load_project(&path).unwrap();

        assert_eq!(loaded.data_mode, "embedded");
        assert_eq!(loaded.sample_data.unwrap(), sample);
        assert_eq!(loaded.open_beam_data.unwrap(), ob);
        assert_eq!(loaded.spectrum_values.unwrap(), spectrum);
    }

    #[test]
    fn test_roundtrip_embedded_preserves_links() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embed_links.nrd.h5");
        let snap = minimal_snapshot();
        let sample = Array3::from_elem((2, 2, 2), 1.0);
        let emb = EmbeddedData {
            sample: Some(&sample),
            open_beam: None,
            spectrum: None,
        };
        save_project_with_data(&path, &snap, Some(&emb)).unwrap();
        let loaded = load_project(&path).unwrap();

        // Links should still be present from the snapshot
        assert_eq!(loaded.sample_path, Some("/data/sample".into()));
        assert_eq!(loaded.open_beam_path, Some("/data/ob".into()));
        assert_eq!(loaded.spectrum_path, Some("/data/spectrum.txt".into()));
    }

    #[test]
    fn test_embedded_file_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embed_size.nrd.h5");
        let snap = minimal_snapshot();
        // 100 frames × 10 × 10 = 10,000 f64 values = 80 KB raw
        let sample = Array3::from_elem((100, 10, 10), 42.0);
        let emb = EmbeddedData {
            sample: Some(&sample),
            open_beam: None,
            spectrum: None,
        };
        save_project_with_data(&path, &snap, Some(&emb)).unwrap();
        let file_size = std::fs::metadata(&path).unwrap().len();
        let raw_size = 100 * 10 * 10 * 8; // 80,000 bytes
        // Compressed file should be smaller than raw data (gzip on uniform data)
        assert!(
            file_size < raw_size,
            "File size {file_size} should be < raw {raw_size}"
        );
    }

    #[test]
    fn test_estimate_embedded_size() {
        let sample = Array3::from_elem((10, 5, 4), 1.0); // 200 elements
        let ob = Array3::from_elem((10, 5, 4), 2.0); // 200 elements
        let spectrum = vec![1.0; 11]; // 11 elements
        let (raw, compressed) = estimate_embedded_size(Some(&sample), Some(&ob), Some(&spectrum));
        // 200 + 200 + 11 = 411 f64 values × 8 bytes = 3288 bytes
        assert_eq!(raw, 411 * 8);
        assert!(compressed < raw);
        assert!(compressed > 0);
    }

    #[test]
    fn test_linked_mode_no_embedded_group() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("linked_no_embed.nrd.h5");
        let snap = minimal_snapshot();
        save_project(&path, &snap).unwrap(); // linked mode, no embedded data

        // Verify /data/embedded group does NOT exist
        let file = hdf5::File::open(&path).unwrap();
        let data = file.group("data").unwrap();
        assert!(
            data.group("embedded").is_err(),
            "Linked-mode file should not have /data/embedded group"
        );
    }

    #[test]
    fn test_embedded_missing_group_errors() {
        // Save a linked-mode file, then patch data_mode to "embedded" and
        // verify that loading returns an error (missing /data/embedded group).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing_embedded.nrd.h5");
        let snap = minimal_snapshot();
        save_project(&path, &snap).unwrap();

        // Overwrite /data/mode attribute to "embedded" without adding an embedded group
        {
            let file = hdf5::File::open_rw(&path).unwrap();
            let data = file.group("data").unwrap();
            // Delete existing mode attribute, then recreate as "embedded"
            data.delete_attr("mode").unwrap();
            let val: hdf5::types::VarLenUnicode = "embedded".parse().unwrap();
            data.new_attr::<hdf5::types::VarLenUnicode>()
                .shape(())
                .create("mode")
                .and_then(|a| a.write_scalar(&val))
                .unwrap();
        }

        let err = load_project(&path);
        assert!(err.is_err(), "Should error when embedded group is missing");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("embedded"),
            "Error should mention 'embedded': {msg}"
        );
    }

    #[test]
    fn test_embedded_wrong_dimensionality_errors() {
        // Save a valid embedded file, then replace sample with a 1D dataset.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wrong_dim.nrd.h5");
        let sample = Array3::from_elem((2, 3, 4), 1.0);
        let spectrum = vec![1.0, 2.0];
        let mut snap = minimal_snapshot();
        snap.data_mode = "embedded".into();
        let emb = EmbeddedData {
            sample: Some(&sample),
            open_beam: None,
            spectrum: Some(&spectrum),
        };
        save_project_with_data(&path, &snap, Some(&emb)).unwrap();

        // Replace /data/embedded/sample with a 1D dataset
        {
            let file = hdf5::File::open_rw(&path).unwrap();
            let embedded = file.group("data/embedded").unwrap();
            let _ = embedded.unlink("sample");
            embedded
                .new_dataset::<f64>()
                .shape([24])
                .create("sample")
                .unwrap()
                .write_raw(&[0.0_f64; 24])
                .unwrap();
        }

        let err = load_project(&path);
        assert!(err.is_err(), "Should error on non-3D sample dataset");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("expected 3D"),
            "Error should mention dimensionality: {msg}"
        );
    }

    #[test]
    fn test_roundtrip_single_fit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single_fit.nrd.h5");
        let mut snap = minimal_snapshot();
        snap.single_fit_densities = Some(vec![0.001, 0.002]);
        snap.single_fit_uncertainties = Some(vec![1e-5, 2e-5]);
        snap.single_fit_chi_squared = Some(1.23);
        snap.single_fit_temperature = Some(296.0);
        snap.single_fit_temperature_unc = Some(5.0);
        snap.single_fit_converged = Some(true);
        snap.single_fit_iterations = Some(42);
        snap.single_fit_pixel = Some((10, 20));
        snap.single_fit_labels = Some(vec!["U-238".into(), "Fe-56".into()]);
        save_project(&path, &snap).unwrap();

        let loaded = load_project(&path).unwrap();
        assert_eq!(
            loaded.single_fit_densities.as_deref(),
            Some([0.001, 0.002].as_slice())
        );
        assert_eq!(
            loaded.single_fit_uncertainties.as_deref(),
            Some([1e-5, 2e-5].as_slice())
        );
        assert!((loaded.single_fit_chi_squared.unwrap() - 1.23).abs() < 1e-10);
        assert!((loaded.single_fit_temperature.unwrap() - 296.0).abs() < 1e-10);
        assert!((loaded.single_fit_temperature_unc.unwrap() - 5.0).abs() < 1e-10);
        assert_eq!(loaded.single_fit_converged, Some(true));
        assert_eq!(loaded.single_fit_iterations, Some(42));
        assert_eq!(loaded.single_fit_pixel, Some((10, 20)));
        let expected_labels: Vec<String> = vec!["U-238".into(), "Fe-56".into()];
        assert_eq!(loaded.single_fit_labels, Some(expected_labels));
    }

    #[test]
    fn test_roundtrip_no_single_fit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_single_fit.nrd.h5");
        let snap = minimal_snapshot();
        save_project(&path, &snap).unwrap();

        let loaded = load_project(&path).unwrap();
        assert!(loaded.single_fit_densities.is_none());
        assert!(loaded.single_fit_pixel.is_none());
        assert!(loaded.single_fit_labels.is_none());
    }
}
