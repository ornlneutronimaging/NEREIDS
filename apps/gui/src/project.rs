//! Project file save and load — AppState ↔ ProjectSnapshot conversion and dialogs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_io::project::{EmbeddedData, PROJECT_SCHEMA_VERSION, ProjectSnapshot};
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};

use crate::state::{
    AppState, DataType, EndfStatus, FittingType, GroupMemberState, InputMode, IsotopeEntry,
    IsotopeGroupEntry, ProvenanceEvent, ProvenanceEventKind, RoiSelection, SaveDataMode, UiMode,
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
        None => "unknown",
    };
    let solver_method = "poisson_kl";

    let resolution_kind = "none";
    let delta_t_us: Option<f64> = None;
    let delta_l_m: Option<f64> = None;
    let tabulated_path: Option<String> = None;

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

    // Isotope group config
    let isotope_group_z: Vec<u32> = state.isotope_groups.iter().map(|g| g.z).collect();
    let isotope_group_names: Vec<String> = state
        .isotope_groups
        .iter()
        .map(|g| g.name.clone())
        .collect();
    let isotope_group_members_json: Vec<String> = state
        .isotope_groups
        .iter()
        .map(|g| {
            let members: Vec<serde_json::Value> = g
                .members
                .iter()
                .filter(|m| m.ratio.is_finite())
                .map(|m| serde_json::json!({"a": m.a, "symbol": m.symbol, "ratio": m.ratio}))
                .collect();
            serde_json::to_string(&members).unwrap_or_else(|_| "[]".to_string())
        })
        .collect();
    let isotope_group_density: Vec<f64> = state
        .isotope_groups
        .iter()
        .map(|g| g.initial_density)
        .collect();
    let isotope_group_enabled: Vec<bool> = state.isotope_groups.iter().map(|g| g.enabled).collect();

    // ENDF cache: collect all resonance_data from isotope entries AND group members
    let mut endf_cache: Vec<(String, nereids_endf::resonance::ResonanceData)> = state
        .isotope_entries
        .iter()
        .filter_map(|e| {
            e.resonance_data
                .as_ref()
                .map(|rd| (e.symbol.clone(), rd.clone()))
        })
        .collect();
    // Also include group members' resonance data (dedup handled by writer)
    for group in &state.isotope_groups {
        for member in &group.members {
            if let Some(ref rd) = member.resonance_data {
                endf_cache.push((member.symbol.clone(), rd.clone()));
            }
        }
    }

    // Results are not produced while the counts engine is re-pointed; the
    // snapshot keeps its result fields for schema compatibility.
    let normalized: Option<ndarray::Array3<f64>> = None;
    let normalized_uncertainty: Option<ndarray::Array3<f64>> = None;
    let density_maps: Option<Vec<ndarray::Array2<f64>>> = None;
    let uncertainty_maps: Option<Vec<ndarray::Array2<f64>>> = None;
    let chi_squared_map: Option<ndarray::Array2<f64>> = None;
    let converged_map: Option<ndarray::Array2<bool>> = None;
    let temperature_map: Option<ndarray::Array2<f64>> = None;
    let temperature_uncertainty_map: Option<ndarray::Array2<f64>> = None;
    let n_converged: Option<usize> = None;
    let n_total: Option<usize> = None;
    let n_failed: Option<usize> = None;
    let result_isotope_labels: Option<Vec<String>> = None;
    let anorm_map: Option<ndarray::Array2<f64>> = None;
    let background_maps: Option<[ndarray::Array2<f64>; 3]> = None;
    let baseline_global: Option<[f64; 3]> = None;
    let baseline_e_ref_ev: Option<f64> = None;
    let baseline_maps: Option<[ndarray::Array2<f64>; 3]> = None;
    let single_fit_densities: Option<Vec<f64>> = None;
    let single_fit_uncertainties: Option<Vec<f64>> = None;
    let single_fit_chi_squared: Option<f64> = None;
    let single_fit_temperature: Option<f64> = None;
    let single_fit_temperature_unc: Option<f64> = None;
    let single_fit_converged: Option<bool> = None;
    let single_fit_iterations: Option<usize> = None;
    let single_fit_pixel: Option<(usize, usize)> = None;
    let single_fit_labels: Option<Vec<String>> = None;
    let single_fit_anorm: Option<f64> = None;
    let single_fit_background: Option<[f64; 3]> = None;
    let single_fit_baseline: Option<[f64; 3]> = None;
    let single_fit_baseline_e_ref_ev: Option<f64> = None;

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
        isotope_group_z,
        isotope_group_names,
        isotope_group_members_json,
        isotope_group_density,
        isotope_group_enabled,
        input_mode: match state.input_mode {
            InputMode::TiffPair => "tiff_pair",
            InputMode::Hdf5Histogram => "hdf5_histogram",
            InputMode::Hdf5Event => "hdf5_event",
        }
        .into(),
        analysis_mode: "full_spatial".into(),
        spatial_binning_factor: None,
        event_n_bins: state.event_n_bins as u32,
        event_tof_min_us: state.event_tof_min_us,
        event_tof_max_us: state.event_tof_max_us,
        event_height: state.event_height as u32,
        event_width: state.event_width as u32,
        solver_method: solver_method.into(),
        max_iter: 200,
        temperature_k: state.temperature_k,
        fit_temperature: false,
        fit_energy_scale: Some(false),
        fit_energy_range: state.fit_energy_range,
        uncertainty_is_estimated: Some(state.uncertainty_is_estimated),
        lm_background_enabled: Some(false),
        baseline_enabled: Some(state.baseline_enabled),
        kl_background_enabled: Some(state.kl_background_enabled),
        kl_c_ratio: Some(state.kl_c_ratio),
        kl_enable_polish_override: Some(state.kl_enable_polish_override),
        resolution_enabled: false,
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
        hdf5_ob_path: state
            .hdf5_ob_path
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
        normalized_uncertainty,
        energies: state.energies.clone(),
        // Persist the two mask components SEPARATELY:
        // `/intermediate/dead_pixels` carries the DECLARED
        // component only: persisting the effective mask (declared ∪
        // detected) there would let the restore-side declared-promotion
        // bake each session's detections into the next session's declared
        // component — monotonic mask growth across save/load cycles.  Old
        // project files stored the effective mask at this same dataset
        // path (unchanged dtype and shape, no format marker needed) and
        // get a documented one-time promotion on load; new saves end the
        // accumulation.
        dead_pixels: state.file_dead_pixels.clone(),
        // The DETECTED component gets its own versioned, session-scoped
        // dataset (`/intermediate/detected_dead_pixels`): a restore
        // without raw stacks cannot re-run detection, and dropping the
        // component there would silently refit without the dead/hot
        // exclusions active at save time.  Restore rebuilds the effective
        // mask as declared ∪ detected — both components round-trip
        // unchanged, so cycles still cannot grow either one.
        detected_dead_pixels: state.detected_dead_pixels.clone(),
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        temperature_map,
        temperature_uncertainty_map,
        n_converged,
        n_total,
        n_failed,
        result_isotope_labels,
        anorm_map,
        background_maps,
        baseline_global,
        baseline_e_ref_ev,
        baseline_maps,
        single_fit_densities,
        single_fit_uncertainties,
        single_fit_chi_squared,
        single_fit_temperature,
        single_fit_temperature_unc,
        single_fit_converged,
        single_fit_iterations,
        single_fit_pixel,
        single_fit_labels,
        single_fit_anorm,
        single_fit_background,
        single_fit_baseline,
        single_fit_baseline_e_ref_ev,
        endf_cache,
        // Read-only diagnostic populated by `load_project`; never written back.
        endf_cache_dropped: vec![],
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
            state.sample_data.as_deref(),
            state.open_beam_data.as_deref(),
            state.spectrum_values.as_ref().map(|v| v.as_slice()),
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

/// Open the "Save As" dialog; the pick is applied by
/// [`on_save_project_picked`] via the dialog dispatcher.
fn execute_save_with_dialog(state: &mut AppState) {
    let mut opts = crate::file_dialog::DialogOptions {
        title: Some("Save NEREIDS Project"),
        filters: vec![("NEREIDS Project", &["nrd.h5"])],
        ..Default::default()
    };

    if let Some(ref existing) = state.project_file_path {
        if let Some(parent) = existing.parent() {
            opts.directory = Some(parent.to_path_buf());
        }
        if let Some(name) = existing.file_name() {
            opts.file_name = Some(name.to_string_lossy().into_owned());
        }
    }

    state.file_dialogs.save_file(
        crate::file_dialog::DialogIntent::SaveProjectAs {
            // Captured now, not read back at resolution: the dialog can
            // resolve much later (Linux native tier), and the user may
            // have reopened the save modal and changed the mode
            // meanwhile — that choice belongs to the NEXT save.
            mode: state.save_data_mode,
        },
        opts,
    );
}

/// Apply a picked "Save As" path: normalize the extension and save with
/// the mode the save modal carried at request time.
pub(crate) fn on_save_project_picked(state: &mut AppState, path: PathBuf, mode: SaveDataMode) {
    let path = ensure_extension(path);
    execute_save(state, &path, mode);
}

/// Execute the actual save to `path` with the given mode.
///
/// Spawns a background thread so large embedded saves do not block the UI.
/// The result is polled in `crate::app::poll_pending_tasks`.
fn execute_save(state: &mut AppState, path: &Path, mode: SaveDataMode) {
    // Guard: don't start another save while one is in progress
    if state.is_saving {
        state.status_message = "Save already in progress.".into();
        return;
    }

    // Fall back to linked if embedded requested but sample data not available
    let mode = if mode == SaveDataMode::Embedded && state.sample_data.is_none() {
        state.status_message = "Sample data required for embed — saving in linked mode.".into();
        SaveDataMode::Linked
    } else {
        mode
    };

    tracing::info!(path = %path.display(), mode = ?mode, "saving project");

    let snap = snapshot_from_state(state);
    let path = path.to_path_buf();

    // For embedded mode, clone the data arrays for the background thread.
    let embedded_data = if mode == SaveDataMode::Embedded {
        Some((
            state.sample_data.clone(),
            state.open_beam_data.clone(),
            state.spectrum_values.clone(),
        ))
    } else {
        None
    };

    let (tx, rx) = std::sync::mpsc::channel();
    let ctx = state.egui_ctx.clone();

    state.is_saving = true;
    state.pending_save = Some(rx);
    state.status_message = "Saving project...".into();

    let handle = std::thread::spawn(move || {
        let result = if let Some((ref sample, ref open_beam, ref spectrum)) = embedded_data {
            let emb = EmbeddedData {
                sample: sample.as_deref(),
                open_beam: open_beam.as_deref(),
                spectrum: spectrum.as_ref().map(|v| v.as_slice()),
            };
            nereids_io::project::save_project_with_data(&path, &snap, Some(&emb))
        } else {
            nereids_io::project::save_project(&path, &snap)
        };
        let _ = tx.send(result.map(|()| (path, mode)).map_err(|e| e.to_string()));
    });
    state.save_join_handle = Some(handle);

    // Watcher thread to poke the GUI event loop until save completes.
    if let Some(ctx) = ctx {
        std::thread::spawn(move || {
            for _ in 0..600 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                ctx.request_repaint();
            }
        });
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

/// Open the project-file dialog; the pick is routed to
/// [`load_project_from_path`] via the dialog dispatcher.
pub fn load_project_dialog(state: &mut AppState) {
    state.file_dialogs.pick_file(
        crate::file_dialog::DialogIntent::OpenProject,
        crate::file_dialog::DialogOptions {
            title: Some("Open NEREIDS Project"),
            filters: vec![("NEREIDS Project", &["nrd.h5", "h5"])],
            ..Default::default()
        },
    );
}

/// Load a project file from `path` and apply the snapshot to `state`.
pub fn load_project_from_path(state: &mut AppState, path: &Path) {
    tracing::info!(path = %path.display(), "loading project");
    match nereids_io::project::load_project(path) {
        Ok(snap) => {
            state_from_snapshot(snap, state, path);
            tracing::info!(path = %path.display(), "project loaded");
        }
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e, "project load failed");
            state.status_message = format!("Load failed: {e}");
        }
    }
}

/// Log a warning when parsed ENDF data carries parse-and-skip placeholder
/// ranges (LRU=2 URR or LRU=0 scattering-radius-only).
///
/// Those spans contribute exactly zero cross-section, so any physics computed
/// over them reflects only the evaluation's other ranges. Shared by the ENDF
/// fetch worker and the project-restore path so both surfaces describe skipped
/// ranges identically, reusing `ResonanceRange::skip_description`. No-op when
/// the evaluation has no unevaluated ranges.
pub(crate) fn warn_unevaluated_ranges(symbol: &str, data: &ResonanceData) {
    if !data.has_unevaluated_ranges() {
        return;
    }
    let spans: Vec<String> = data
        .unevaluated_ranges()
        .iter()
        .map(|r| r.skip_description())
        .collect();
    tracing::warn!(
        isotope = %symbol,
        skipped = %spans.join("; "),
        "ENDF ranges parsed but not evaluated (zero cross-section over these spans)"
    );
}

/// Apply a [`ProjectSnapshot`] to [`AppState`], restoring the full session.
fn state_from_snapshot(snap: ProjectSnapshot, state: &mut AppState, path: &Path) {
    // 0. Cancel any in-flight background tasks (ENDF fetches, fitting, etc.)
    state.cancel_pending_tasks();

    // cancel_pending_tasks deliberately leaves save state alone (save cannot be
    // cancelled safely). Since we are replacing the entire session, clear save
    // tracking explicitly — the background thread will still finish its HDF5
    // write; the result does not matter for this session, but we
    // still keep the JoinHandle so shutdown code can join the thread safely.
    state.is_saving = false;
    state.pending_save = None;
    // Note: save_join_handle is intentionally NOT cleared here.
    // Dropping it would detach the thread, making on_exit unable to join it.
    // The handle will be joined in on_exit() before the process terminates.

    // 1. Clear derived state
    state.energies = None;
    state.preview_image = None;
    state.sample_data = None;
    state.open_beam_data = None;
    state.spectrum_values = None;
    state.dead_pixels = None;
    state.file_dead_pixels = None;
    state.detected_dead_pixels = None;
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
        _ => None,
    };

    // 3. Rebuild pipeline
    state.rebuild_pipeline();

    // 4. Restore input_mode (use persisted value if available, fall back to heuristic for old files)
    state.input_mode = if snap.input_mode.is_empty() {
        // Old heuristic (backward compat)
        match (snap.data_type.as_str(), snap.hdf5_path.is_some()) {
            ("events", true) => InputMode::Hdf5Event,
            ("events", false) => InputMode::TiffPair,
            ("pre_normalized", _) => InputMode::TiffPair,
            _ => InputMode::TiffPair,
        }
    } else {
        match snap.input_mode.as_str() {
            "hdf5_histogram" => InputMode::Hdf5Histogram,
            "hdf5_event" => InputMode::Hdf5Event,
            _ => InputMode::TiffPair,
        }
    };

    // 4c. Restore event params
    if snap.event_n_bins > 0 {
        state.event_n_bins = snap.event_n_bins as usize;
        state.event_tof_min_us = snap.event_tof_min_us;
        state.event_tof_max_us = snap.event_tof_max_us;
        state.event_height = snap.event_height as usize;
        state.event_width = snap.event_width as usize;
    } else if matches!(state.input_mode, InputMode::Hdf5Event) {
        state.event_n_bins = 0;
        state.event_tof_min_us = 0.0;
        state.event_tof_max_us = 0.0;
        state.event_height = 0;
        state.event_width = 0;
    }

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
        "TENDL-2023" => nereids_endf::retrieval::EndfLibrary::Tendl2023,
        "CENDL-3.2" => nereids_endf::retrieval::EndfLibrary::Cendl3_2,
        _ => nereids_endf::retrieval::EndfLibrary::EndfB8_0,
    };

    // 7. ENDF cache priority — build lookup from snapshot.
    // Entries dropped by the loader (removed-physics payloads with no evaluable
    // range) are absent from `endf_cache`, so the isotopes below fall through
    // to `EndfStatus::Pending` and are re-fetched on demand — the fetch worker
    // then surfaces the parser's hard error. Warn so the drop is visible.
    // ENDF restore notices (dropped stale entries + parsed-but-skipped
    // spans), collected into the load status line + provenance log below (a
    // tracing::warn alone lands only in the log file, where a user re-running
    // a saved fit would miss that the isotope set changed).
    let mut endf_notices: Vec<String> = Vec::new();
    for symbol in &snap.endf_cache_dropped {
        tracing::warn!(
            isotope = %symbol,
            "cached ENDF evaluation dropped on project load (no evaluable \
             resonance range — stale removed-physics data); re-fetch required"
        );
        endf_notices.push(format!(
            "cached ENDF data for {symbol} was dropped (no evaluable resonance \
             range — stale removed-physics data); the isotope is Pending and \
             excluded from fits until re-fetched"
        ));
    }
    let endf_cache: HashMap<String, ResonanceData> = snap.endf_cache.into_iter().collect();
    // Mixed evaluations that survived the load still carry parsed-but-skipped
    // spans (zero cross-section). Emit the same warning the fetch worker
    // emits, and surface a one-line notice in the load status/provenance —
    // the same visibility the drop notices above get, for the same reason.
    for (symbol, rd) in &endf_cache {
        warn_unevaluated_ranges(symbol, rd);
        let skipped = rd.unevaluated_ranges();
        if !skipped.is_empty() {
            endf_notices.push(format!(
                "{symbol}: {} of {} resonance range(s) are parsed but NOT \
                 evaluated ({}); those spans contribute zero cross-section",
                skipped.len(),
                rd.ranges.len(),
                skipped
                    .iter()
                    .map(|r| r.skip_description())
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
    }

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

    // 8b. Restore isotope groups with ENDF cache
    state.isotope_groups.clear();
    let n_groups = snap
        .isotope_group_z
        .len()
        .min(snap.isotope_group_names.len())
        .min(snap.isotope_group_density.len())
        .min(snap.isotope_group_members_json.len());
    for i in 0..n_groups {
        let members: Vec<GroupMemberState> = snap
            .isotope_group_members_json
            .get(i)
            .and_then(|json| serde_json::from_str::<Vec<serde_json::Value>>(json).ok())
            .unwrap_or_default()
            .iter()
            .map(|v| {
                let a = v["a"].as_u64().unwrap_or(0) as u32;
                let symbol = v["symbol"].as_str().unwrap_or("").to_string();
                let ratio = v["ratio"].as_f64().unwrap_or(0.0);
                // Check ENDF cache for this symbol
                let rd = endf_cache.get(&symbol).cloned();
                let status = if rd.is_some() {
                    EndfStatus::Loaded
                } else {
                    EndfStatus::Pending
                };
                GroupMemberState {
                    a,
                    symbol,
                    ratio,
                    resonance_data: rd,
                    endf_status: status,
                }
            })
            .filter(|m| m.a > 0 && !m.symbol.is_empty())
            .collect();
        state.isotope_groups.push(IsotopeGroupEntry {
            z: snap.isotope_group_z[i],
            name: snap.isotope_group_names[i].clone(),
            members,
            initial_density: snap.isotope_group_density[i],
            enabled: snap.isotope_group_enabled.get(i).copied().unwrap_or(true),
        });
    }

    // 9. Restore solver-independent configuration
    state.temperature_k = snap.temperature_k;
    state.fit_energy_range = snap.fit_energy_range;
    state.uncertainty_is_estimated = snap.uncertainty_is_estimated.unwrap_or(false);
    state.baseline_enabled = snap.baseline_enabled.unwrap_or(false);
    state.kl_background_enabled = snap.kl_background_enabled.unwrap_or(false);
    state.kl_c_ratio = snap.kl_c_ratio.unwrap_or(1.0);
    state.kl_enable_polish_override = snap.kl_enable_polish_override.unwrap_or(None);

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
    state.hdf5_ob_path = snap.hdf5_ob_path.map(|s| {
        let p = PathBuf::from(&s);
        if !p.exists() {
            missing.push(s);
        }
        p
    });

    // 14b. Restore embedded data if present
    if snap.data_mode == "embedded" {
        if let Some(data) = snap.sample_data {
            state.sample_data = Some(Arc::new(data));
        }
        if let Some(data) = snap.open_beam_data {
            state.open_beam_data = Some(Arc::new(data));
        }
        if let Some(data) = snap.spectrum_values {
            state.spectrum_values = Some(Arc::new(data));
        }
        state.last_save_mode = SaveDataMode::Embedded;
    } else {
        state.last_save_mode = SaveDataMode::Linked;
    }

    state.energies = snap.energies;

    // 15b. Restore the dead-pixel mask components.
    // project files persist the DECLARED component at
    // `/intermediate/dead_pixels` (snapshot_from_state writes
    // state.file_dead_pixels) and the DETECTED component at
    // `/intermediate/detected_dead_pixels`; the effective
    // mask is rebuilt here as declared ∪ persisted-detected via the same
    // single-site helper every normalization uses — a restored project
    // may carry embedded normalized data WITHOUT raw stacks, where
    // detection never re-runs, and a declared-only restore would silently
    // refit without the exclusions active at save time.  OLD files stored
    // the effective mask at the declared dataset path and lack the
    // detected dataset; promoting the effective mask to declared is a
    // documented one-time promotion — the only lossless reading of a file
    // that never recorded the split.  Either way, the next normalization
    // recomputes the effective mask from scratch as declared ∪
    // freshly-detected (AppState::set_detected_dead_pixels), so
    // save→load→save cycles cannot grow either component.  If the linked
    // HDF5 is reloaded, the load site
    // overwrites the declared mask with the file's own.
    //
    // Geometry validation: both components now come from a
    // FILE, so a hand-corrupted project can carry a mask whose geometry
    // does not match the data it will be applied to.  The refit consumes
    // the effective mask against the restored normalized transmission
    // (guided::analyze returns early without state.normalized) — without
    // ROIs the pipeline rejects a wrong-geometry mask cleanly, but with
    // restored ROIs the analyze-side ROI∪mask merge (ndarray::Zip)
    // panics on the shape mismatch.  So when the snapshot carries
    // normalized data (restored at step 15 above), validate each
    // component against its spatial dims and DROP any that cannot apply,
    // surfaced via the same deferred-notice mechanism as the drift
    // warning below — never silently.  When the snapshot carries no
    // normalized data, geometry is unknown here, but no refit can run
    // either: the only route to a mask consumer is a fresh normalization,
    // which recomputes the detected component at the data's own geometry
    // and drops an inapplicable declared mask via the setter's returned
    // notice.
    let mut mask_notices: Vec<String> = Vec::new();
    let expected_dims: Option<(usize, usize)> = None;
    let mut declared = snap.dead_pixels;
    let mut detected = snap.detected_dead_pixels;
    if let Some(dims) = expected_dims {
        if let Some(ref m) = declared
            && m.dim() != dims
        {
            mask_notices.push(format!(
                "restored declared pixel mask {:?} does not match the \
                 restored transmission geometry {:?}; declared mask \
                 dropped",
                m.dim(),
                dims
            ));
            declared = None;
        }
        if let Some(ref m) = detected
            && m.dim() != dims
        {
            mask_notices.push(format!(
                "restored detected pixel mask {:?} does not match the \
                 restored transmission geometry {:?}; detected mask \
                 dropped",
                m.dim(),
                dims
            ));
            detected = None;
        }
    }
    state.file_dead_pixels = declared;
    // The setter RETURNS a declared-mask-drop notice instead of logging
    // it: step 17 below replaces the provenance log wholesale
    // with the snapshot's, which would erase an entry appended here.
    // Deferred — with the geometry notices above — to steps 20/22,
    // exactly like the drift warning (15c).
    if let Some(notice) = state.set_detected_dead_pixels(detected) {
        mask_notices.push(notice);
    }

    // 15c. Detection-drift check: when the raw stacks WERE
    // restored (embedded save), re-run the same detection normalization
    // would run and WARN if it does not reproduce the persisted detected
    // component — hidden state drift becomes observable instead of being
    // silently replaced (or silently trusted).  The persisted component
    // stays active either way; re-normalizing adopts the fresh detection
    // explicitly.  Mode gating: the TIFF pair detects over sample ∪ OB
    // (skip when the OB stack was not restored — recomputing on the sample
    // alone would warn spuriously); the HDF5 modes detect over whatever is
    // present.
    let detected_drift: Option<String> = match (&state.detected_dead_pixels, &state.sample_data) {
        (Some(persisted), Some(sample)) => {
            let ob = state.open_beam_data.as_deref();
            let stacks = match state.input_mode {
                InputMode::TiffPair => ob.map(|o| (sample.as_ref(), Some(o))),
                InputMode::Hdf5Histogram | InputMode::Hdf5Event => Some((sample.as_ref(), ob)),
            };
            stacks.and_then(|(s, o)| {
                match nereids_io::pixels::detect_bad_pixels(
                    s,
                    o,
                    Some(nereids_io::pixels::HOT_PIXEL_K_MAD),
                ) {
                    Ok(fresh) if fresh.dim() == persisted.dim() => {
                        let n_diff = fresh
                            .iter()
                            .zip(persisted.iter())
                            .filter(|(f, p)| f != p)
                            .count();
                        (n_diff > 0).then(|| {
                            format!(
                                "recomputed dead/hot detection on the restored raw stacks \
                                 differs from the saved detected mask at {n_diff} pixel(s); \
                                 saved mask kept active — re-run normalization to adopt \
                                 the fresh detection"
                            )
                        })
                    }
                    Ok(fresh) => Some(format!(
                        "recomputed dead/hot detection is {:?} but the saved detected \
                         mask is {:?}; saved mask kept active",
                        fresh.dim(),
                        persisted.dim()
                    )),
                    Err(e) => Some(format!(
                        "dead/hot detection drift check failed ({e}); saved detected \
                         mask kept active"
                    )),
                }
            })
        }
        _ => None,
    };
    // (Logged to provenance at step 22 — step 17 below REPLACES the
    // provenance log with the snapshot's, which would drop an entry
    // appended here.)

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
    if !state.pipeline.is_empty() {
        state.ui_mode = UiMode::Guided;
        state.guided_step = state.pipeline[0].step;
    }

    // 20. Status message — compose the mask-drop notices (15b) and the
    // drift warning (15c) in rather than leaving them provenance-only; a
    // plain "Project loaded" would bury them.
    let mut status = format!("Project loaded from {}", path.display());
    if !missing.is_empty() {
        status.push_str(&format!(" (missing files: {})", missing.join(", ")));
    }
    for msg in &mask_notices {
        status.push_str(&format!(" — WARNING: {msg}"));
    }
    for msg in &endf_notices {
        status.push_str(&format!(" — WARNING: {msg}"));
    }
    if let Some(ref msg) = detected_drift {
        status.push_str(&format!(" — WARNING: {msg}"));
    }
    state.status_message = status;

    // 21. Clear stale wizard session cache
    state.cached_session = None;

    // 22. Log provenance (mask-drop and drift entries after step 17's
    // wholesale replacement of the provenance log — see 15b/15c)
    state.log_provenance(
        ProvenanceEventKind::ProjectLoaded,
        format!("Loaded from {}", path.display()),
    );
    for msg in &mask_notices {
        state.log_provenance(
            ProvenanceEventKind::ConfigChanged,
            format!("Pixel-mask restore: {msg}"),
        );
    }
    for msg in &endf_notices {
        state.log_provenance(
            ProvenanceEventKind::ConfigChanged,
            format!("ENDF restore: {msg}"),
        );
    }
    if let Some(ref msg) = detected_drift {
        state.log_provenance(
            ProvenanceEventKind::ProjectLoaded,
            format!("Detected-mask drift: {msg}"),
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use nereids_io::project::{load_project, save_project, save_project_with_data};

    /// The mask the refit consumes: `guided::analyze` merges
    /// `state.dead_pixels` with the ROI mask, and with no ROIs (the case
    /// in these tests) uses `state.dead_pixels` as-is.
    fn refit_active_mask(state: &AppState) -> Option<Array2<bool>> {
        assert!(state.rois.is_empty());
        state.dead_pixels.clone()
    }

    /// Project SAVE persists the mask as two
    /// SEPARATE components (declared at `/intermediate/dead_pixels`,
    /// detected at `/intermediate/detected_dead_pixels`), so a
    /// save→load→save cycle must not grow EITHER component.
    #[test]
    fn test_snapshot_persists_split_mask_components_no_growth_across_cycles() {
        let mut declared = Array2::from_elem((3, 3), false);
        declared[[0, 0]] = true;
        let mut detected = Array2::from_elem((3, 3), false);
        detected[[1, 1]] = true;

        let mut state = AppState {
            file_dead_pixels: Some(declared.clone()),
            ..AppState::default()
        };
        assert!(
            state
                .set_detected_dead_pixels(Some(detected.clone()))
                .is_none()
        );
        // Effective mask in the session is declared ∪ detected...
        assert_eq!(
            state
                .dead_pixels
                .as_ref()
                .unwrap()
                .iter()
                .filter(|&&m| m)
                .count(),
            2
        );

        // ...and the snapshot persists the two components separately —
        // the declared dataset never absorbs the detection.
        let snap = snapshot_from_state(&state);
        assert_eq!(snap.dead_pixels.as_ref().unwrap(), &declared);
        assert_eq!(snap.detected_dead_pixels.as_ref().unwrap(), &detected);

        // Load into a fresh session: each component restores to its own
        // field and the effective mask is rebuilt as declared ∪ detected.
        let mut restored = AppState::default();
        state_from_snapshot(snap, &mut restored, Path::new("/nonexistent/p.nrd.h5"));
        assert_eq!(restored.file_dead_pixels.as_ref().unwrap(), &declared);
        assert_eq!(restored.detected_dead_pixels.as_ref().unwrap(), &detected);
        assert_eq!(
            restored
                .dead_pixels
                .as_ref()
                .unwrap()
                .iter()
                .filter(|&&m| m)
                .count(),
            2
        );

        // The next normalization re-detects; the next save STILL persists
        // the same two components — no growth of either across cycles.
        assert!(
            restored
                .set_detected_dead_pixels(Some(detected.clone()))
                .is_none()
        );
        let snap2 = snapshot_from_state(&restored);
        assert_eq!(snap2.dead_pixels.as_ref().unwrap(), &declared);
        assert_eq!(snap2.detected_dead_pixels.as_ref().unwrap(), &detected);
    }

    /// Acceptance (b): save→restore→refit WITHOUT raw stacks
    /// — the exact scenario the detected-component persistence exists
    /// for.  Normalization (and thus detection) cannot re-run, so the
    /// refit's active mask must still equal the at-save mask, through the
    /// real HDF5 writer/reader.
    #[test]
    fn test_save_restore_refit_active_mask_without_raw_stacks() {
        let mut declared = Array2::from_elem((3, 3), false);
        declared[[0, 0]] = true;
        let mut detected = Array2::from_elem((3, 3), false);
        detected[[1, 1]] = true;

        let mut state = AppState {
            input_mode: InputMode::Hdf5Histogram,
            file_dead_pixels: Some(declared.clone()),
            ..AppState::default()
        };
        assert!(
            state
                .set_detected_dead_pixels(Some(detected.clone()))
                .is_none()
        );
        let at_save_mask = refit_active_mask(&state).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_raw.nrd.h5");
        save_project(&path, &snapshot_from_state(&state)).unwrap();

        let mut restored = AppState::default();
        state_from_snapshot(load_project(&path).unwrap(), &mut restored, &path);

        // Scenario precondition: no raw stacks after restore.
        assert!(restored.sample_data.is_none());
        // The refit's active mask equals the at-save mask (before the
        // detected component was lost and this was declared-only).
        assert_eq!(refit_active_mask(&restored).unwrap(), at_save_mask);
        // No drift check without raw stacks: nothing to recompute from.
        assert!(!restored.status_message.contains("WARNING"));
        assert!(
            !restored
                .provenance_log
                .iter()
                .any(|e| e.message.contains("drift"))
        );
    }

    /// Acceptance (a): save→restore→refit WITH raw stacks
    /// embedded.  The persisted detected component still restores the
    /// at-save active mask, and — since the recomputed detection agrees
    /// with the persisted one — no drift warning is raised.
    #[test]
    fn test_save_restore_refit_active_mask_with_raw_stacks() {
        // Raw stack with a genuinely detectable hot pixel: uniform 100
        // counts, one railed pixel (see
        // test_detect_hot_pixels_uniform_image_poisson_floor_path).
        let mut sample = Array3::from_elem((1, 3, 3), 100.0);
        sample[[0, 1, 1]] = 65535.0;
        let detected = nereids_io::pixels::detect_bad_pixels(
            &sample,
            None,
            Some(nereids_io::pixels::HOT_PIXEL_K_MAD),
        )
        .unwrap();
        assert!(detected[[1, 1]], "fixture must detect the railed pixel");

        let mut declared = Array2::from_elem((3, 3), false);
        declared[[0, 0]] = true;

        let mut state = AppState {
            input_mode: InputMode::Hdf5Histogram,
            sample_data: Some(Arc::new(sample.clone())),
            file_dead_pixels: Some(declared),
            ..AppState::default()
        };
        assert!(state.set_detected_dead_pixels(Some(detected)).is_none());
        let at_save_mask = refit_active_mask(&state).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("with_raw.nrd.h5");
        let embedded = EmbeddedData {
            sample: Some(&sample),
            open_beam: None,
            spectrum: None,
        };
        save_project_with_data(&path, &snapshot_from_state(&state), Some(&embedded)).unwrap();

        let mut restored = AppState::default();
        state_from_snapshot(load_project(&path).unwrap(), &mut restored, &path);

        // Scenario precondition: raw stacks present after restore.
        assert!(restored.sample_data.is_some());
        assert_eq!(refit_active_mask(&restored).unwrap(), at_save_mask);
        // Recomputed detection reproduces the persisted component → no
        // drift warning.
        assert!(!restored.status_message.contains("WARNING"));
        assert!(
            !restored
                .provenance_log
                .iter()
                .any(|e| e.message.contains("drift"))
        );
    }

    /// When raw stacks are present on restore and the
    /// recomputed detection DISAGREES with the persisted detected
    /// component, the persisted mask stays active (never silently
    /// replaced) and the drift is surfaced in both the status message and
    /// the provenance log.
    #[test]
    fn test_restore_with_raw_stacks_warns_on_detection_drift_keeps_saved_mask() {
        let mut sample = Array3::from_elem((1, 3, 3), 100.0);
        sample[[0, 1, 1]] = 65535.0; // recompute will flag (1, 1)

        // Persisted detected component from a "different" detection run:
        // flags (0, 2) instead — drift at 2 pixels ((1,1) and (0,2)).
        let mut stale_detected = Array2::from_elem((3, 3), false);
        stale_detected[[0, 2]] = true;

        let mut state = AppState {
            input_mode: InputMode::Hdf5Histogram,
            sample_data: Some(Arc::new(sample.clone())),
            ..AppState::default()
        };
        assert!(
            state
                .set_detected_dead_pixels(Some(stale_detected.clone()))
                .is_none()
        );
        let at_save_mask = refit_active_mask(&state).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drift.nrd.h5");
        let embedded = EmbeddedData {
            sample: Some(&sample),
            open_beam: None,
            spectrum: None,
        };
        save_project_with_data(&path, &snapshot_from_state(&state), Some(&embedded)).unwrap();

        let mut restored = AppState::default();
        state_from_snapshot(load_project(&path).unwrap(), &mut restored, &path);

        // The saved mask is kept active — drift never silently replaces it.
        assert_eq!(refit_active_mask(&restored).unwrap(), at_save_mask);
        assert_eq!(
            restored.detected_dead_pixels.as_ref().unwrap(),
            &stale_detected
        );
        // ...and the drift is observable in status + provenance.
        assert!(
            restored.status_message.contains("WARNING"),
            "status must surface the drift: {}",
            restored.status_message
        );
        assert!(restored.status_message.contains("2 pixel(s)"));
        assert!(
            restored
                .provenance_log
                .iter()
                .any(|e| e.message.contains("Detected-mask drift")),
            "provenance must record the drift"
        );
    }

    /// A mismatched declared/detected pair on restore
    /// must stay observable THROUGH step 17's wholesale replacement of
    /// the session provenance log with the snapshot's — the erasure the
    /// direct-call setter test (state.rs) cannot see.  End-to-end through
    /// the real HDF5 writer/reader: no normalized data in the snapshot,
    /// so restore-side geometry validation cannot apply and the corrupt
    /// pair reaches `set_detected_dead_pixels`, whose drop notice is
    /// deferred past step 17 into the final provenance log AND the
    /// status message.
    #[test]
    fn test_restore_mismatched_mask_pair_drop_notice_survives_provenance_replacement() {
        let mut snap = snapshot_from_state(&AppState::default());
        // Hand-corrupted pair: declared (4, 4) vs detected (2, 2).
        snap.dead_pixels = Some(Array2::from_elem((4, 4), true));
        let mut detected = Array2::from_elem((2, 2), false);
        detected[[0, 0]] = true;
        snap.detected_dead_pixels = Some(detected.clone());
        // The snapshot carries its own provenance history — step 17
        // replaces the session log with exactly this, which would
        // otherwise erase the drop entry the setter appended at step 15b.
        snap.provenance = vec![(
            "2026-07-04 00:00:00 UTC".into(),
            "ProjectSaved".into(),
            "saved in an earlier session".into(),
        )];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mismatched_pair.nrd.h5");
        save_project(&path, &snap).unwrap();

        let mut restored = AppState::default();
        state_from_snapshot(load_project(&path).unwrap(), &mut restored, &path);

        // Effective mask: the inapplicable declared mask is dropped —
        // the refit runs with the detected component only.
        assert_eq!(refit_active_mask(&restored).unwrap(), detected);
        // The drop survives step 17: the FINAL provenance log carries it
        // (naming both geometries)...
        assert!(
            restored
                .provenance_log
                .iter()
                .any(|e| e.message.contains("(4, 4)") && e.message.contains("(2, 2)")),
            "final provenance log must carry the declared-mask drop"
        );
        // ...alongside the snapshot's own restored history (i.e. the
        // replacement DID happen and the notice still made it through)...
        assert!(
            restored
                .provenance_log
                .iter()
                .any(|e| e.message.contains("saved in an earlier session")),
            "snapshot provenance must have replaced the session log"
        );
        // ...and the status message surfaces it too.
        assert!(
            restored.status_message.contains("WARNING")
                && restored.status_message.contains("(4, 4)")
                && restored.status_message.contains("(2, 2)"),
            "status must surface the drop: {}",
            restored.status_message
        );
    }
}
