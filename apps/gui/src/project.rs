//! Project file save and load — AppState ↔ ProjectSnapshot conversion and dialogs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_io::normalization::NormalizedData;
use nereids_io::project::{EmbeddedData, PROJECT_SCHEMA_VERSION, ProjectSnapshot};
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use nereids_pipeline::spatial::SpatialResult;

use crate::state::{
    AppState, DataType, EndfStatus, FitFeedback, FittingType, GuidedStep, InputMode, IsotopeEntry,
    ProvenanceEvent, ProvenanceEventKind, ResolutionMode, RoiSelection, SaveDataMode, SolverMethod,
    UiMode,
};
use crate::widgets::design::library_name;

/// Internal action for the save modal.
enum SaveModalAction {
    None,
    Save,
    Cancel,
}

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
        (
            Some(sr.density_maps.clone()),
            Some(sr.uncertainty_maps.clone()),
            Some(sr.chi_squared_map.clone()),
            Some(sr.converged_map.clone()),
            sr.temperature_map.clone(),
            Some(sr.n_converged),
            Some(sr.n_total),
            Some(sr.isotope_labels.clone()),
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    // Single-pixel fit results
    let (
        single_fit_densities,
        single_fit_uncertainties,
        single_fit_chi_squared,
        single_fit_temperature,
        single_fit_temperature_unc,
        single_fit_converged,
        single_fit_iterations,
        single_fit_pixel,
        single_fit_labels,
    ) = if let Some(ref pfr) = state.pixel_fit_result {
        // Prefer labels from FitFeedback (captured at fit time) to avoid
        // desync if isotope_entries are modified after the fit.
        let labels: Vec<String> = state
            .last_fit_feedback
            .as_ref()
            .map(|fb| fb.densities.iter().map(|(s, _)| s.clone()).collect())
            .unwrap_or_else(|| {
                state
                    .isotope_entries
                    .iter()
                    .filter(|e| e.enabled && e.resonance_data.is_some())
                    .map(|e| e.symbol.clone())
                    .collect()
            });
        (
            Some(pfr.densities.clone()),
            pfr.uncertainties.clone(),
            Some(pfr.reduced_chi_squared),
            pfr.temperature_k,
            pfr.temperature_k_unc,
            Some(pfr.converged),
            Some(pfr.iterations),
            state.selected_pixel,
            Some(labels),
        )
    } else {
        (None, None, None, None, None, None, None, None, None)
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
        max_iter: state.lm_config.max_iter.min(u32::MAX as usize) as u32,
        temperature_k: state.temperature_k,
        fit_temperature: state.fit_temperature,
        resolution_enabled: state.resolution_enabled,
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
        rebin_factor: state.rebin_factor.min(u32::MAX as usize) as u32,
        rebin_applied: state.rebin_applied,
        sample_data: None,
        open_beam_data: None,
        spectrum_values: None,
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
        single_fit_densities,
        single_fit_uncertainties,
        single_fit_chi_squared,
        single_fit_temperature,
        single_fit_temperature_unc,
        single_fit_converged,
        single_fit_iterations,
        single_fit_pixel,
        single_fit_labels,
        endf_cache,
        provenance,
    }
}

/// Open the save-mode chooser modal.
pub fn save_project_dialog(state: &mut AppState) {
    state.show_save_modal = true;
}

/// Quick-save: re-save to the existing project file path without dialog.
pub fn save_project_quick(state: &mut AppState) {
    if let Some(ref path) = state.project_file_path.clone() {
        // If last save was embedded but data is gone, fall back to linked
        let mode = if state.last_save_mode == SaveDataMode::Embedded && state.sample_data.is_none()
        {
            state.status_message = "Raw data no longer in memory — saving in linked mode.".into();
            SaveDataMode::Linked
        } else {
            state.last_save_mode
        };
        execute_save(state, path, mode);
    } else {
        save_project_dialog(state);
    }
}

/// Ensure the path ends with `.nrd.h5`.
fn ensure_extension(path: PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if s.ends_with(".nrd.h5") {
        path
    } else if s.ends_with(".hdf5") {
        let stripped = s.strip_suffix(".hdf5").unwrap();
        PathBuf::from(format!("{stripped}.nrd.h5"))
    } else if s.ends_with(".h5") {
        let stripped = s.strip_suffix(".h5").unwrap();
        PathBuf::from(format!("{stripped}.nrd.h5"))
    } else {
        PathBuf::from(format!("{s}.nrd.h5"))
    }
}

/// Render the save-mode chooser modal window.
pub fn save_modal(ctx: &egui::Context, state: &mut AppState) {
    if !state.show_save_modal {
        return;
    }

    let can_embed = state.sample_data.is_some();
    let (raw_bytes, compressed_bytes) = if can_embed {
        nereids_io::project::estimate_embedded_size(
            state.sample_data.as_ref(),
            state.open_beam_data.as_ref(),
            state.spectrum_values.as_deref(),
        )
    } else {
        (0, 0)
    };

    let mut action = SaveModalAction::None;

    egui::Window::new("Save Project")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label("Choose how to store raw data:");
            ui.add_space(8.0);

            ui.radio_value(
                &mut state.save_data_mode,
                SaveDataMode::Linked,
                "Link to source files (recommended)",
            );
            ui.add_space(4.0);

            if can_embed {
                ui.radio_value(
                    &mut state.save_data_mode,
                    SaveDataMode::Embedded,
                    "Embed all data (for sharing)",
                );
            } else {
                ui.add_enabled(
                    false,
                    egui::RadioButton::new(false, "Embed all data (for sharing)"),
                )
                .on_disabled_hover_text(
                    "Raw data not in memory. Reload data in the Load step to enable embedding.",
                );
            }

            if state.save_data_mode == SaveDataMode::Embedded && can_embed {
                let ratio = nereids_io::project::EMBED_COMPRESSION_RATIO;
                ui.add_space(8.0);
                crate::widgets::design::card(ui, |ui| {
                    if let Some(ref sample) = state.sample_data {
                        let s = (sample.len() as u64) * 8;
                        ui.label(format!(
                            "Sample: {} \u{2192} ~{}",
                            crate::telemetry::format_bytes(s),
                            crate::telemetry::format_bytes((s as f64 / ratio) as u64),
                        ));
                    }
                    if let Some(ref ob) = state.open_beam_data {
                        let s = (ob.len() as u64) * 8;
                        ui.label(format!(
                            "Open beam: {} \u{2192} ~{}",
                            crate::telemetry::format_bytes(s),
                            crate::telemetry::format_bytes((s as f64 / ratio) as u64),
                        ));
                    }
                    if let Some(ref sp) = state.spectrum_values {
                        let s = (sp.len() as u64) * 8;
                        ui.label(format!(
                            "Spectrum: {} \u{2192} ~{}",
                            crate::telemetry::format_bytes(s),
                            crate::telemetry::format_bytes((s as f64 / ratio) as u64),
                        ));
                    }
                    ui.separator();
                    ui.label(format!(
                        "Total: {} \u{2192} ~{}",
                        crate::telemetry::format_bytes(raw_bytes),
                        crate::telemetry::format_bytes(compressed_bytes),
                    ));
                });

                if state.data_type == Some(DataType::Events) {
                    ui.add_space(4.0);
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "\u{26A0} Raw event data is very large. Consider linked mode.",
                    );
                }
            }

            ui.add_space(12.0);

            ui.horizontal(|ui| {
                let label = if state.save_data_mode == SaveDataMode::Embedded && can_embed {
                    format!(
                        "Save (~{})",
                        crate::telemetry::format_bytes(compressed_bytes)
                    )
                } else {
                    "Save".to_string()
                };
                if ui.button(label).clicked() {
                    action = SaveModalAction::Save;
                }
                if ui.button("Cancel").clicked() {
                    action = SaveModalAction::Cancel;
                }
            });
        });

    match action {
        SaveModalAction::Save => {
            state.show_save_modal = false;
            execute_save_with_dialog(state);
        }
        SaveModalAction::Cancel => {
            state.show_save_modal = false;
        }
        SaveModalAction::None => {}
    }
}

/// Show the native file dialog and save with the currently selected mode.
fn execute_save_with_dialog(state: &mut AppState) {
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
        let path = ensure_extension(path);
        execute_save(state, &path, state.save_data_mode);
    }
}

/// Execute the actual save to `path` with the given mode.
fn execute_save(state: &mut AppState, path: &Path, mode: SaveDataMode) {
    // Fall back to linked if embedded requested but sample data not available
    let mode = if mode == SaveDataMode::Embedded && state.sample_data.is_none() {
        state.status_message = "Sample data required for embed — saving in linked mode.".into();
        SaveDataMode::Linked
    } else {
        mode
    };
    let snap = snapshot_from_state(state);
    let result = match mode {
        SaveDataMode::Linked => nereids_io::project::save_project(path, &snap),
        SaveDataMode::Embedded => {
            let emb = EmbeddedData {
                sample: state.sample_data.as_ref(),
                open_beam: state.open_beam_data.as_ref(),
                spectrum: state.spectrum_values.as_deref(),
            };
            nereids_io::project::save_project_with_data(path, &snap, Some(&emb))
        }
    };
    let mode_label = match mode {
        SaveDataMode::Linked => "linked",
        SaveDataMode::Embedded => "embedded",
    };
    match result {
        Ok(()) => {
            state.project_file_path = Some(path.to_path_buf());
            state.last_save_mode = mode;
            state.status_message = format!("Project saved ({mode_label}) to {}", path.display());
            state.log_provenance(
                ProvenanceEventKind::ProjectSaved,
                format!("Saved ({mode_label}) to {}", path.display()),
            );
        }
        Err(e) => {
            state.status_message = format!("Save failed: {e}");
        }
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

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Show a native open dialog and load a project file.
pub fn load_project_dialog(state: &mut AppState) {
    let dialog = rfd::FileDialog::new()
        .set_title("Open NEREIDS Project")
        .add_filter("NEREIDS Project", &["nrd.h5", "h5"]);

    if let Some(path) = dialog.pick_file() {
        load_project_from_path(state, &path);
    }
}

/// Load a project file from `path` and apply the snapshot to `state`.
pub fn load_project_from_path(state: &mut AppState, path: &Path) {
    match nereids_io::project::load_project(path) {
        Ok(snap) => {
            state_from_snapshot(snap, state, path);
        }
        Err(e) => {
            state.status_message = format!("Load failed: {e}");
        }
    }
}

/// Apply a [`ProjectSnapshot`] to [`AppState`], restoring the full session.
fn state_from_snapshot(snap: ProjectSnapshot, state: &mut AppState, path: &Path) {
    // 0. Cancel any in-flight background tasks (ENDF fetches, fitting, etc.)
    state.cancel_pending_tasks();

    // 1. Clear derived state
    state.spatial_result = None;
    state.pixel_fit_result = None;
    state.last_fit_feedback = None;
    state.normalized = None;
    state.energies = None;
    state.preview_image = None;
    state.tile_display.clear();
    state.residuals_cache = None;
    state.sample_data = None;
    state.open_beam_data = None;
    state.spectrum_values = None;
    state.dead_pixels = None;
    state.fm_spectrum = None;
    state.fm_per_isotope_spectra.clear();
    state.detect_results.clear();
    state.load_error = false;
    state.show_save_modal = false;
    state.save_data_mode = SaveDataMode::default();

    // 2. Parse fitting_type and data_type
    state.fitting_type = match snap.fitting_type.as_str() {
        "spatial" => Some(FittingType::Spatial),
        "single" => Some(FittingType::Single),
        _ => None,
    };
    state.data_type = match snap.data_type.as_str() {
        "events" => Some(DataType::Events),
        "pre_normalized" => Some(DataType::PreNormalized),
        "transmission" => Some(DataType::Transmission),
        _ => None,
    };

    // 3. Rebuild pipeline
    state.rebuild_pipeline();

    // 4. Derive input_mode heuristically
    state.input_mode = match (snap.data_type.as_str(), snap.hdf5_path.is_some()) {
        ("events", true) => InputMode::Hdf5Event,
        ("events", false) => InputMode::TiffPair,
        ("pre_normalized", _) => InputMode::TiffPair,
        ("transmission", _) => InputMode::TransmissionTiff,
        _ => InputMode::TiffPair,
    };

    // 5. Restore beamline
    state.beamline.flight_path_m = snap.flight_path_m;
    state.beamline.delay_us = snap.delay_us;
    state.proton_charge_sample = snap.proton_charge_sample;
    state.proton_charge_ob = snap.proton_charge_ob;

    // 6. Restore ENDF library
    state.endf_library = match snap.endf_library.as_str() {
        "ENDF/B-VIII.1" => nereids_endf::retrieval::EndfLibrary::EndfB8_1,
        "JEFF-3.3" => nereids_endf::retrieval::EndfLibrary::Jeff3_3,
        "JENDL-5" => nereids_endf::retrieval::EndfLibrary::Jendl5,
        _ => nereids_endf::retrieval::EndfLibrary::EndfB8_0,
    };

    // 7. ENDF cache priority — build lookup from snapshot
    let endf_cache: HashMap<String, ResonanceData> = snap.endf_cache.into_iter().collect();

    // 8. Restore isotope entries with ENDF cache
    // Use the minimum length across all parallel arrays to avoid panics on corrupted files.
    let n_iso = snap
        .isotope_z
        .len()
        .min(snap.isotope_a.len())
        .min(snap.isotope_symbol.len());
    state.isotope_entries = (0..n_iso)
        .map(|i| {
            let symbol = snap.isotope_symbol[i].clone();
            let rd = endf_cache.get(&symbol).cloned();
            let status = if rd.is_some() {
                EndfStatus::Loaded
            } else {
                EndfStatus::Pending
            };
            IsotopeEntry {
                z: snap.isotope_z[i],
                a: snap.isotope_a[i],
                symbol,
                initial_density: snap.isotope_density.get(i).copied().unwrap_or(0.0),
                resonance_data: rd,
                enabled: snap.isotope_enabled.get(i).copied().unwrap_or(true),
                endf_status: status,
            }
        })
        .collect();

    // 9. Restore solver
    state.solver_method = match snap.solver_method.as_str() {
        "poisson_kl" => SolverMethod::PoissonKL,
        _ => SolverMethod::LevenbergMarquardt,
    };
    state.lm_config.max_iter = snap.max_iter as usize;
    state.temperature_k = snap.temperature_k;
    state.fit_temperature = snap.fit_temperature;

    // 10. Restore resolution
    state.resolution_enabled = snap.resolution_enabled;
    state.resolution_mode = match snap.resolution_kind.as_str() {
        "tabulated" => {
            if let Some(p) = snap.tabulated_path {
                ResolutionMode::Tabulated {
                    path: PathBuf::from(p),
                    data: None,
                    error: None,
                }
            } else {
                ResolutionMode::Gaussian {
                    delta_t_us: snap.delta_t_us.unwrap_or(0.0),
                    delta_l_m: snap.delta_l_m.unwrap_or(0.0),
                }
            }
        }
        _ => ResolutionMode::Gaussian {
            delta_t_us: snap.delta_t_us.unwrap_or(0.0),
            delta_l_m: snap.delta_l_m.unwrap_or(0.0),
        },
    };

    // 11. Restore ROIs
    state.rois = snap
        .rois
        .iter()
        .map(|r| RoiSelection {
            y_start: r[0] as usize,
            y_end: r[1] as usize,
            x_start: r[2] as usize,
            x_end: r[3] as usize,
        })
        .collect();
    state.selected_roi = None;
    state.fitting_rois = state.rois.clone();

    // 12. Restore spectrum unit/kind
    state.spectrum_unit = match snap.spectrum_unit.as_str() {
        "energy_ev" => SpectrumUnit::EnergyEv,
        _ => SpectrumUnit::TofMicroseconds,
    };
    state.spectrum_kind = match snap.spectrum_kind.as_str() {
        "bin_centers" => SpectrumValueKind::BinCenters,
        _ => SpectrumValueKind::BinEdges,
    };

    // 13. Restore rebin state
    state.rebin_factor = snap.rebin_factor as usize;
    state.rebin_applied = snap.rebin_applied;

    // 14. Restore file paths (verify existence, collect warnings)
    let mut missing = Vec::new();
    state.sample_path = snap.sample_path.map(|s| {
        let p = PathBuf::from(&s);
        if !p.exists() {
            missing.push(s);
        }
        p
    });
    state.open_beam_path = snap.open_beam_path.map(|s| {
        let p = PathBuf::from(&s);
        if !p.exists() {
            missing.push(s);
        }
        p
    });
    state.spectrum_path = snap.spectrum_path.map(|s| {
        let p = PathBuf::from(&s);
        if !p.exists() {
            missing.push(s);
        }
        p
    });
    state.hdf5_path = snap.hdf5_path.map(|s| {
        let p = PathBuf::from(&s);
        if !p.exists() {
            missing.push(s);
        }
        p
    });

    // 14b. Restore embedded data if present
    if snap.data_mode == "embedded" {
        if let Some(data) = snap.sample_data {
            state.sample_data = Some(data);
        }
        if let Some(data) = snap.open_beam_data {
            state.open_beam_data = Some(data);
        }
        if let Some(data) = snap.spectrum_values {
            state.spectrum_values = Some(data);
        }
        state.last_save_mode = SaveDataMode::Embedded;
    } else {
        state.last_save_mode = SaveDataMode::Linked;
    }

    // 15. Restore intermediate data
    if let Some(transmission) = snap.normalized {
        let shape = transmission.raw_dim();
        let uncertainty = ndarray::Array3::zeros(shape);
        state.normalized = Some(Arc::new(NormalizedData {
            transmission,
            uncertainty,
        }));
    }
    state.energies = snap.energies;

    // 16. Restore results
    if let Some(density_maps) = snap.density_maps {
        let n_maps = density_maps.len();
        let uncertainty_maps = snap.uncertainty_maps.unwrap_or_else(|| {
            density_maps
                .iter()
                .map(|m| ndarray::Array2::zeros(m.raw_dim()))
                .collect()
        });
        let chi_squared_map = snap
            .chi_squared_map
            .unwrap_or_else(|| ndarray::Array2::zeros((1, 1)));
        let converged_map = snap
            .converged_map
            .unwrap_or_else(|| ndarray::Array2::from_elem((1, 1), false));
        let result = SpatialResult {
            density_maps,
            uncertainty_maps,
            chi_squared_map,
            converged_map,
            temperature_map: snap.temperature_map,
            isotope_labels: snap.result_isotope_labels.unwrap_or_else(|| {
                // Fallback for project files created before labels were stored
                // in SpatialResult — derive from the restored isotope entries.
                state
                    .isotope_entries
                    .iter()
                    .filter(|e| e.enabled && e.resonance_data.is_some())
                    .map(|e| e.symbol.clone())
                    .collect()
            }),
            n_converged: snap.n_converged.unwrap_or(0),
            n_total: snap.n_total.unwrap_or(0),
        };
        state.init_tile_display(n_maps);
        state.spatial_result = Some(result);
    }

    // 16b. Restore single-pixel fit results
    if let Some(densities) = snap.single_fit_densities {
        let uncertainties = snap.single_fit_uncertainties;
        let result = nereids_pipeline::pipeline::SpectrumFitResult {
            densities,
            uncertainties,
            reduced_chi_squared: snap.single_fit_chi_squared.unwrap_or(0.0),
            converged: snap.single_fit_converged.unwrap_or(false),
            iterations: snap.single_fit_iterations.unwrap_or(0),
            temperature_k: snap.single_fit_temperature,
            temperature_k_unc: snap.single_fit_temperature_unc,
        };
        // Rebuild FitFeedback from the restored result
        if let Some(ref labels) = snap.single_fit_labels {
            let fb_densities: Vec<(String, f64)> = labels
                .iter()
                .zip(result.densities.iter())
                .map(|(s, &d)| (s.clone(), d))
                .collect();
            let summary = if result.converged {
                format!("Converged, chi2_r = {:.4}", result.reduced_chi_squared)
            } else {
                "Did not converge".to_string()
            };
            state.last_fit_feedback = Some(FitFeedback {
                success: result.converged,
                summary,
                densities: fb_densities,
                temperature_k: result.temperature_k,
            });
        }
        state.selected_pixel = snap.single_fit_pixel;
        state.pixel_fit_result = Some(result);
        state.fit_result_gen += 1;
    }

    // 17. Restore provenance
    state.provenance_log = snap
        .provenance
        .iter()
        .map(|(ts, kind_str, msg)| {
            let kind = match kind_str.as_str() {
                "DataLoaded" => ProvenanceEventKind::DataLoaded,
                "Normalized" => ProvenanceEventKind::Normalized,
                "AnalysisRun" => ProvenanceEventKind::AnalysisRun,
                "Exported" => ProvenanceEventKind::Exported,
                "ProjectSaved" => ProvenanceEventKind::ProjectSaved,
                "ProjectLoaded" => ProvenanceEventKind::ProjectLoaded,
                _ => ProvenanceEventKind::ConfigChanged,
            };
            let timestamp = parse_timestamp(ts);
            ProvenanceEvent {
                timestamp,
                kind,
                message: msg.clone(),
            }
        })
        .collect();

    // 18. Set project file path
    state.project_file_path = Some(path.to_path_buf());

    // 19. Auto-navigate
    if state.spatial_result.is_some() {
        state.ui_mode = UiMode::Studio;
    } else if state.normalized.is_some() {
        state.ui_mode = UiMode::Guided;
        state.guided_step = GuidedStep::Analyze;
    } else if !state.pipeline.is_empty() {
        state.ui_mode = UiMode::Guided;
        state.guided_step = state.pipeline[0].step;
    }

    // 20. Status message
    let mut status = format!("Project loaded from {}", path.display());
    if !missing.is_empty() {
        status.push_str(&format!(" (missing files: {})", missing.join(", ")));
    }
    state.status_message = status;

    // 21. Clear stale wizard session cache and dirty tracking
    state.cached_session = None;
    state.clear_dirty();

    // 22. Log provenance
    state.log_provenance(
        ProvenanceEventKind::ProjectLoaded,
        format!("Loaded from {}", path.display()),
    );
}

/// Parse a timestamp string ("YYYY-MM-DD HH:MM:SS UTC" or ISO 8601) into SystemTime.
fn parse_timestamp(s: &str) -> std::time::SystemTime {
    // Try "YYYY-MM-DD HH:MM:SS UTC" or "YYYY-MM-DDTHH:MM:SSZ"
    let s = s
        .trim()
        .trim_end_matches("UTC")
        .trim_end_matches('Z')
        .trim();
    let parts: Vec<&str> = s.splitn(2, ['T', ' ']).collect();
    if parts.len() != 2 {
        return std::time::UNIX_EPOCH;
    }
    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|p| p.parse().ok()).collect();
    let time_parts: Vec<u64> = parts[1].split(':').filter_map(|p| p.parse().ok()).collect();
    if date_parts.len() != 3 || time_parts.len() != 3 {
        return std::time::UNIX_EPOCH;
    }
    let (y, m, d) = (date_parts[0], date_parts[1], date_parts[2]);
    let (h, mi, sec) = (time_parts[0], time_parts[1], time_parts[2]);
    let days = ymd_to_days(y, m, d);
    let total_secs = days
        .checked_mul(86400)
        .and_then(|v| v.checked_add(h * 3600 + mi * 60 + sec));
    match total_secs {
        Some(secs) => std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs),
        None => std::time::UNIX_EPOCH,
    }
}

/// Convert (year, month, day) to days since Unix epoch.
fn ymd_to_days(y: u64, m: u64, d: u64) -> u64 {
    // Inverse of days_to_ymd — Howard Hinnant's civil_from_days
    if m <= 2 && y == 0 {
        return 0;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = y / 400;
    let yoe = y - era * 400;
    let m_adj = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * m_adj + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let base = era * 146097 + doe;
    base.saturating_sub(719468)
}
