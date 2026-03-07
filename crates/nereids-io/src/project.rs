//! Project file save for `.nrd.h5` (NEREIDS HDF5 archive).
//!
//! The project file captures the full session state so users can persist
//! and share analysis sessions. This module defines [`ProjectSnapshot`]
//! (a serialization-friendly subset of the GUI's `AppState`) and the
//! [`save_project`] function that writes it to HDF5.

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

    // -- data (linked mode) --
    /// "linked" (embed mode is a separate issue)
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

    // -- intermediate (always embedded) --
    pub normalized: Option<Array3<f64>>,
    pub energies: Option<Vec<f64>>,

    // -- results (always embedded) --
    pub density_maps: Option<Vec<Array2<f64>>>,
    pub uncertainty_maps: Option<Vec<Array2<f64>>>,
    pub chi_squared_map: Option<Array2<f64>>,
    pub converged_map: Option<Array2<bool>>,
    pub temperature_map: Option<Array2<f64>>,
    pub n_converged: Option<usize>,
    pub n_total: Option<usize>,
    pub result_isotope_labels: Option<Vec<String>>,

    // -- endf_cache --
    /// (symbol, resonance_data) pairs for offline loading.
    pub endf_cache: Vec<(String, ResonanceData)>,

    // -- provenance --
    /// (timestamp, kind, message) triples.
    pub provenance: Vec<(String, String, String)>,
}

/// Write a project snapshot to an HDF5 file at `path`.
pub fn save_project(path: &Path, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let file = hdf5::File::create(path).map_err(|e| IoError::Hdf5Error(format!("create: {e}")))?;

    write_meta(&file, snap)?;
    write_config(&file, snap)?;
    write_data_links(&file, snap)?;
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

fn write_data_links(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let data = file
        .create_group("data")
        .map_err(|e| hdf5_err("create /data", e))?;
    write_str_attr(&data, "mode", &snap.data_mode)?;
    write_str_attr(&data, "spectrum_unit", &snap.spectrum_unit)?;
    write_str_attr(&data, "spectrum_kind", &snap.spectrum_kind)?;
    write_u32_attr(&data, "rebin_factor", snap.rebin_factor)?;
    write_bool_attr(&data, "rebin_applied", snap.rebin_applied)?;

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

    Ok(())
}

fn write_intermediate(file: &hdf5::File, snap: &ProjectSnapshot) -> Result<(), IoError> {
    let inter = file
        .create_group("intermediate")
        .map_err(|e| hdf5_err("create /intermediate", e))?;

    if let Some(ref norm) = snap.normalized {
        let shape = [norm.shape()[0], norm.shape()[1], norm.shape()[2]];
        let write_result = if let Some(slice) = norm.as_slice() {
            inter
                .new_dataset::<f64>()
                .shape(shape)
                .chunk(chunk_shape_3d(shape))
                .deflate(4)
                .create("normalized")
                .and_then(|ds| ds.write_raw(slice))
        } else {
            let data: Vec<f64> = norm.iter().copied().collect();
            inter
                .new_dataset::<f64>()
                .shape(shape)
                .chunk(chunk_shape_3d(shape))
                .deflate(4)
                .create("normalized")
                .and_then(|ds| ds.write_raw(&data))
        };
        write_result.map_err(|e| hdf5_err("/intermediate/normalized", e))?;
    }

    if let Some(ref energies) = snap.energies {
        inter
            .new_dataset::<f64>()
            .shape([energies.len()])
            .create("energies")
            .and_then(|ds| ds.write_raw(energies))
            .map_err(|e| hdf5_err("/intermediate/energies", e))?;
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
    let frames = shape[0].min(256);
    [frames, shape[1], shape[2]]
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
            normalized: None,
            energies: None,
            density_maps: None,
            uncertainty_maps: None,
            chi_squared_map: None,
            converged_map: None,
            temperature_map: None,
            n_converged: None,
            n_total: None,
            result_isotope_labels: None,
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
}
