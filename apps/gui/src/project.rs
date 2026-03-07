//! Project file save — AppState → ProjectSnapshot conversion and save dialog.

use std::path::PathBuf;

use nereids_io::project::{PROJECT_SCHEMA_VERSION, ProjectSnapshot};

use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};

use crate::state::{
    AppState, DataType, FittingType, ProvenanceEventKind, ResolutionMode, SolverMethod,
};
use crate::widgets::design::library_name;

/// Build a [`ProjectSnapshot`] from the current [`AppState`].
pub fn snapshot_from_state(state: &AppState) -> ProjectSnapshot {
    let fitting_type = match state.fitting_type {
        Some(FittingType::Spatial) => "spatial",
        Some(FittingType::Single) => "single",
        None => "unknown",
    };
    let data_type = match state.data_type {
        Some(DataType::Events) => "events",
        Some(DataType::PreNormalized) => "pre_normalized",
        Some(DataType::Transmission) => "transmission",
        None => "unknown",
    };
    let solver_method = match state.solver_method {
        SolverMethod::LevenbergMarquardt => "lm",
        SolverMethod::PoissonKL => "poisson_kl",
    };

    let (resolution_kind, delta_t_us, delta_l_m, tabulated_path) = match &state.resolution_mode {
        ResolutionMode::Gaussian {
            delta_t_us: dt,
            delta_l_m: dl,
        } => {
            if state.resolution_enabled {
                ("gaussian", Some(*dt), Some(*dl), None)
            } else {
                ("none", None, None, None)
            }
        }
        ResolutionMode::Tabulated { path, .. } => {
            if state.resolution_enabled {
                (
                    "tabulated",
                    None,
                    None,
                    Some(path.to_string_lossy().into_owned()),
                )
            } else {
                ("none", None, None, None)
            }
        }
    };

    let rois: Vec<[u64; 4]> = state
        .rois
        .iter()
        .map(|r| {
            [
                r.y_start as u64,
                r.y_end as u64,
                r.x_start as u64,
                r.x_end as u64,
            ]
        })
        .collect();

    // Isotope config (parallel arrays)
    let n_iso = state.isotope_entries.len();
    let mut isotope_z = Vec::with_capacity(n_iso);
    let mut isotope_a = Vec::with_capacity(n_iso);
    let mut isotope_symbol = Vec::with_capacity(n_iso);
    let mut isotope_density = Vec::with_capacity(n_iso);
    let mut isotope_enabled = Vec::with_capacity(n_iso);
    for entry in &state.isotope_entries {
        isotope_z.push(entry.z);
        isotope_a.push(entry.a);
        isotope_symbol.push(entry.symbol.clone());
        isotope_density.push(entry.initial_density);
        isotope_enabled.push(entry.enabled);
    }

    // ENDF cache: collect all resonance_data from isotope entries
    let endf_cache: Vec<(String, nereids_endf::resonance::ResonanceData)> = state
        .isotope_entries
        .iter()
        .filter_map(|e| {
            e.resonance_data
                .as_ref()
                .map(|rd| (e.symbol.clone(), rd.clone()))
        })
        .collect();

    // Normalized data (transmission array)
    let normalized = state.normalized.as_ref().map(|nd| nd.transmission.clone());

    // Results from spatial mapping
    let (
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        temperature_map,
        n_converged,
        n_total,
        result_isotope_labels,
    ) = if let Some(ref sr) = state.spatial_result {
        let labels: Vec<String> = state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled)
            .map(|e| e.symbol.clone())
            .collect();
        (
            Some(sr.density_maps.clone()),
            Some(sr.uncertainty_maps.clone()),
            Some(sr.chi_squared_map.clone()),
            Some(sr.converged_map.clone()),
            sr.temperature_map.clone(),
            Some(sr.n_converged),
            Some(sr.n_total),
            Some(labels),
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    // Provenance log
    let provenance: Vec<(String, String, String)> = state
        .provenance_log
        .iter()
        .map(|ev| {
            (
                ev.formatted_timestamp(),
                format!("{:?}", ev.kind),
                ev.message.clone(),
            )
        })
        .collect();

    let endf_library_name = library_name(state.endf_library).to_string();

    let spectrum_unit = match state.spectrum_unit {
        SpectrumUnit::TofMicroseconds => "tof_us",
        SpectrumUnit::EnergyEv => "energy_ev",
    };
    let spectrum_kind = match state.spectrum_kind {
        SpectrumValueKind::BinEdges => "bin_edges",
        SpectrumValueKind::BinCenters => "bin_centers",
    };

    ProjectSnapshot {
        schema_version: PROJECT_SCHEMA_VERSION.to_string(),
        created_utc: chrono_utc_now(),
        software_version: env!("CARGO_PKG_VERSION").to_string(),
        fitting_type: fitting_type.into(),
        data_type: data_type.into(),
        flight_path_m: state.beamline.flight_path_m,
        delay_us: state.beamline.delay_us,
        proton_charge_sample: state.proton_charge_sample,
        proton_charge_ob: state.proton_charge_ob,
        isotope_z,
        isotope_a,
        isotope_symbol,
        isotope_density,
        isotope_enabled,
        solver_method: solver_method.into(),
        max_iter: state.lm_config.max_iter as u32,
        temperature_k: state.temperature_k,
        fit_temperature: state.fit_temperature,
        resolution_kind: resolution_kind.into(),
        delta_t_us,
        delta_l_m,
        tabulated_path,
        rois,
        endf_library: endf_library_name,
        data_mode: "linked".into(),
        sample_path: state
            .sample_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        open_beam_path: state
            .open_beam_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        spectrum_path: state
            .spectrum_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        hdf5_path: state
            .hdf5_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        spectrum_unit: spectrum_unit.into(),
        spectrum_kind: spectrum_kind.into(),
        rebin_factor: state.rebin_factor as u32,
        rebin_applied: state.rebin_applied,
        normalized,
        energies: state.energies.clone(),
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        temperature_map,
        n_converged,
        n_total,
        result_isotope_labels,
        endf_cache,
        provenance,
    }
}

/// Show a native save dialog and write the project file.
pub fn save_project_dialog(state: &mut AppState) {
    let mut dialog = rfd::FileDialog::new()
        .set_title("Save NEREIDS Project")
        .add_filter("NEREIDS Project", &["nrd.h5"]);

    if let Some(ref existing) = state.project_file_path {
        if let Some(parent) = existing.parent() {
            dialog = dialog.set_directory(parent);
        }
        if let Some(name) = existing.file_name() {
            dialog = dialog.set_file_name(name.to_string_lossy());
        }
    }

    if let Some(path) = dialog.save_file() {
        // Ensure .nrd.h5 extension
        let path = ensure_extension(path);

        let snap = snapshot_from_state(state);
        match nereids_io::project::save_project(&path, &snap) {
            Ok(()) => {
                state.project_file_path = Some(path.clone());
                state.status_message = format!("Project saved to {}", path.display());
                state.log_provenance(
                    ProvenanceEventKind::ProjectSaved,
                    format!("Saved to {}", path.display()),
                );
            }
            Err(e) => {
                state.status_message = format!("Save failed: {e}");
            }
        }
    }
}

/// Quick-save: re-save to the existing project file path without dialog.
pub fn save_project_quick(state: &mut AppState) {
    if let Some(ref path) = state.project_file_path.clone() {
        let snap = snapshot_from_state(state);
        match nereids_io::project::save_project(path, &snap) {
            Ok(()) => {
                state.status_message = format!("Project saved to {}", path.display());
                state.log_provenance(
                    ProvenanceEventKind::ProjectSaved,
                    format!("Saved to {}", path.display()),
                );
            }
            Err(e) => {
                state.status_message = format!("Save failed: {e}");
            }
        }
    } else {
        save_project_dialog(state);
    }
}

/// Ensure the path ends with `.nrd.h5`.
fn ensure_extension(path: PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if s.ends_with(".nrd.h5") {
        path
    } else if s.ends_with(".h5") || s.ends_with(".hdf5") {
        // Replace existing HDF5 extension
        let stem = path.with_extension("");
        let stem = stem.with_extension(""); // strip double ext
        PathBuf::from(format!("{}.nrd.h5", stem.display()))
    } else {
        PathBuf::from(format!("{}.nrd.h5", s))
    }
}

/// ISO 8601 UTC timestamp (no chrono dependency — use SystemTime).
fn chrono_utc_now() -> String {
    let now = std::time::SystemTime::now();
    let d = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Simple UTC formatting without chrono
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since epoch to Y-M-D (simplified — good enough for ISO 8601)
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
