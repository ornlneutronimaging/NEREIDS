//! Step 1: Data loading — multi-format TIFF + spectrum file input.
//!
//! Prototype: `.content-area` → cards with drop zones, auto-load,
//! format hints, and a Continue button with data guard.

use crate::state::{AppState, InputMode, ProvenanceEventKind};
use crate::theme::ThemeColors;
use crate::widgets::design;
use ndarray::Axis;

const INPUT_MODE_LABELS: [&str; 4] = [
    "TIFF Pair + Spectrum",
    "Transmission TIFF",
    "HDF5 Histogram",
    "HDF5 Event",
];

const INPUT_MODES: [InputMode; 4] = [
    InputMode::TiffPair,
    InputMode::TransmissionTiff,
    InputMode::Hdf5Histogram,
    InputMode::Hdf5Event,
];

/// Draw the Load step content.
pub fn load_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(ui, "Load Data", "Select input format and load files");

    // Input mode tabs — invalidate results when switching modes
    let mut tab_idx = INPUT_MODES
        .iter()
        .position(|&m| m == state.input_mode)
        .unwrap_or(0);
    if design::underline_tabs(ui, &INPUT_MODE_LABELS, &mut tab_idx) {
        state.input_mode = INPUT_MODES[tab_idx];
        state.invalidate_results();
        state.sample_data = None;
        state.open_beam_data = None;
        state.load_error = false;
    }

    ui.add_space(8.0);

    match state.input_mode {
        InputMode::TiffPair => tiff_pair_tab(ui, state),
        InputMode::TransmissionTiff => transmission_tiff_tab(ui, state),
        InputMode::Hdf5Histogram => hdf5_histogram_tab(ui, state),
        InputMode::Hdf5Event => hdf5_event_tab(ui, state),
    }

    // ── Navigation ─────────────────────────────────────────────
    ui.add_space(12.0);
    let can_continue = has_required_data(state);
    let nav_hint = if state.load_error {
        "Loading failed \u{2014} fix files or retry"
    } else {
        "Select files to continue"
    };
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        can_continue,
        nav_hint,
    ) {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}

/// Check whether the minimum data for this input mode is loaded.
fn has_required_data(state: &AppState) -> bool {
    match state.input_mode {
        InputMode::TiffPair => {
            state.sample_data.is_some()
                && state.open_beam_data.is_some()
                && state.spectrum_values.is_some()
        }
        InputMode::TransmissionTiff => {
            state.sample_data.is_some() && state.spectrum_values.is_some()
        }
        InputMode::Hdf5Histogram => state.sample_data.is_some() && state.spectrum_values.is_some(),
        InputMode::Hdf5Event => {
            // For events, we only need the file selected here;
            // histogramming happens in the Bin step.
            state.hdf5_path.is_some() && state.nexus_metadata.as_ref().is_some_and(|m| m.has_events)
        }
    }
}

// ── TIFF Pair tab ──────────────────────────────────────────────

/// TIFF Pair tab: Sample + Open Beam drop zones + Spectrum.
fn tiff_pair_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Load raw sample + open beam TIFF stacks with TOF spectrum.")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);

        let sample_changed = tiff_drop_zone(ui, "Sample", &mut state.sample_path);
        if sample_changed {
            state.sample_data = None;
            state.normalized = None;
            state.dead_pixels = None;
            state.energies = None;
            state.pixel_fit_result = None;
            state.spatial_result = None;
            state.load_error = false;
        }

        ui.add_space(6.0);
        let ob_changed = tiff_drop_zone(ui, "Open Beam", &mut state.open_beam_path);
        if ob_changed {
            state.open_beam_data = None;
            state.normalized = None;
            state.dead_pixels = None;
            state.energies = None;
            state.pixel_fit_result = None;
            state.spatial_result = None;
            state.load_error = false;
        }

        ui.add_space(8.0);
        spectrum_section(ui, state);
    });

    // Auto-load when all files are selected
    let can_load = state.sample_path.is_some()
        && state.open_beam_path.is_some()
        && state.spectrum_path.is_some()
        && state.sample_data.is_none()
        && !state.load_error;
    if can_load {
        load_all_data(state);
    }

    load_status_ui(ui, state);
    show_loaded_info(ui, state);
}

// ── Transmission TIFF tab ──────────────────────────────────────

/// Transmission TIFF tab: pre-normalized TIFF + Spectrum.
fn transmission_tiff_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Load pre-normalized transmission TIFF stack with TOF spectrum.")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);

        let changed = tiff_drop_zone(ui, "Transmission", &mut state.sample_path);
        if changed {
            state.sample_data = None;
            state.normalized = None;
            state.dead_pixels = None;
            state.energies = None;
            state.pixel_fit_result = None;
            state.spatial_result = None;
            state.load_error = false;
        }

        ui.add_space(8.0);
        spectrum_section(ui, state);
    });

    // Auto-load when all files are selected
    let can_load = state.sample_path.is_some()
        && state.spectrum_path.is_some()
        && state.sample_data.is_none()
        && !state.load_error;
    if can_load {
        load_all_data(state);
    }

    load_status_ui(ui, state);
    show_loaded_info(ui, state);
}

// ── Drop zone + folder fallback ────────────────────────────────

/// TIFF drop zone: click-to-browse file, with "or browse folder" link below.
///
/// Returns `true` if the user selected a new path.
fn tiff_drop_zone(ui: &mut egui::Ui, label: &str, path: &mut Option<std::path::PathBuf>) -> bool {
    let loaded = path.is_some();
    let display = path
        .as_ref()
        .map_or(format!("Click to select {label}..."), |p| {
            p.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        });
    let hint = "TIFF file or folder of TIFFs";

    let resp = design::drop_zone(ui, loaded, &display, hint);
    let mut changed = false;
    if resp.clicked()
        && let Some(f) = rfd::FileDialog::new()
            .add_filter("TIFF", &["tif", "tiff"])
            .pick_file()
    {
        *path = Some(f);
        changed = true;
    }
    // Secondary: folder browse
    if ui.small_button("or browse folder\u{2026}").clicked()
        && let Some(d) = rfd::FileDialog::new().pick_folder()
    {
        *path = Some(d);
        changed = true;
    }
    changed
}

// ── Spectrum section ───────────────────────────────────────────

/// Simplified spectrum file section: drop zone, no unit/kind toggles.
fn spectrum_section(ui: &mut egui::Ui, state: &mut AppState) {
    ui.label(egui::RichText::new("Spectrum File").strong());

    let loaded = state.spectrum_path.is_some();
    let display =
        state
            .spectrum_path
            .as_ref()
            .map_or("Click to select spectrum file...".to_string(), |p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });
    let resp = design::drop_zone(
        ui,
        loaded,
        &display,
        "CSV/TXT/DAT with TOF bin edges or centers",
    );
    if resp.clicked()
        && let Some(f) = rfd::FileDialog::new()
            .add_filter("Spectrum", &["csv", "txt", "dat"])
            .pick_file()
    {
        state.spectrum_path = Some(f);
        state.spectrum_values = None;
        state.sample_data = None;
        state.open_beam_data = None;
        state.energies = None;
        state.normalized = None;
        state.load_error = false;
    }

    // Show parsed info (no unit/kind toggles — auto-detected in load_all_data)
    if let Some(ref vals) = state.spectrum_values {
        ui.label(
            egui::RichText::new(format!("Parsed: {} values", vals.len()))
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
    }
}

// ── Loaded info ────────────────────────────────────────────────

/// Display loaded data info.
fn show_loaded_info(ui: &mut egui::Ui, state: &AppState) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    if state.sample_data.is_none() && state.open_beam_data.is_none() {
        return;
    }
    ui.add_space(8.0);
    if let Some(ref data) = state.sample_data {
        let shape = data.shape();
        ui.label(
            egui::RichText::new(format!(
                "\u{2713} Sample: {} frames, {}×{} px",
                shape[0], shape[1], shape[2]
            ))
            .size(11.0)
            .color(tc.fg2),
        );
    }
    if let Some(ref data) = state.open_beam_data {
        let shape = data.shape();
        ui.label(
            egui::RichText::new(format!(
                "\u{2713} Open Beam: {} frames, {}×{} px",
                shape[0], shape[1], shape[2]
            ))
            .size(11.0)
            .color(tc.fg2),
        );
    }
}

// ── HDF5 Histogram tab ────────────────────────────────────────

/// HDF5 Histogram tab: load pre-histogrammed NeXus data.
fn hdf5_histogram_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Load pre-histogrammed NeXus/HDF5 data.")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);
        hdf5_drop_zone(ui, state);
        show_nexus_metadata(ui, state);
    });

    show_hdf5_tree(ui, state);

    // Auto-load histogram when file is selected and has histogram data
    let can_load = state.hdf5_path.is_some()
        && state
            .nexus_metadata
            .as_ref()
            .is_some_and(|m| m.has_histogram)
        && state.sample_data.is_none()
        && !state.load_error;
    if can_load {
        load_hdf5_histogram(state);
    }

    load_status_ui(ui, state);
    show_loaded_info(ui, state);
}

// ── HDF5 Event tab ─────────────────────────────────────────────

/// HDF5 Event tab: load raw neutron events and histogram them.
fn hdf5_event_tab(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Load raw neutron events from NeXus/HDF5 and histogram them.")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(8.0);
        hdf5_drop_zone(ui, state);
        show_nexus_metadata(ui, state);
    });

    show_hdf5_tree(ui, state);

    show_loaded_info(ui, state);
}

// ── HDF5 shared helpers ────────────────────────────────────────

/// HDF5 file drop zone with auto-probe on selection.
fn hdf5_drop_zone(ui: &mut egui::Ui, state: &mut AppState) {
    let loaded = state.hdf5_path.is_some();
    let display =
        state
            .hdf5_path
            .as_ref()
            .map_or("Click to select NeXus/HDF5 file...".to_string(), |p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });
    let resp = design::drop_zone(ui, loaded, &display, "HDF5/NeXus (.h5, .hdf5, .nxs, .nx5)");
    if resp.clicked()
        && let Some(file) = rfd::FileDialog::new()
            .add_filter("NeXus/HDF5", &["h5", "hdf5", "nxs", "nx5"])
            .pick_file()
    {
        state.hdf5_path = Some(file.clone());
        state.invalidate_results();
        state.sample_data = None;
        state.open_beam_data = None;
        state.load_error = false;
        state.nexus_probe_error = None;

        // Probe the file immediately
        match nereids_io::nexus::probe_nexus(&file) {
            Ok(meta) => {
                if let Some(shape) = meta.histogram_shape {
                    state.event_height = shape[1];
                    state.event_width = shape[2];
                }
                state.nexus_metadata = Some(meta);
                state.nexus_probe_error = None;
                state.status_message = "NeXus file probed".into();
            }
            Err(e) => {
                state.nexus_metadata = None;
                state.nexus_probe_error = Some(format!("Probe failed: {e}"));
                state.status_message = format!("Probe failed: {e}");
            }
        }

        // Build HDF5 tree structure for browser display
        match nereids_io::nexus::list_hdf5_tree(&file, 3) {
            Ok(tree) => state.hdf5_tree = Some(tree),
            Err(_) => state.hdf5_tree = None,
        }
    }
}

/// Display probed NeXus metadata (or inline probe error).
fn show_nexus_metadata(ui: &mut egui::Ui, state: &AppState) {
    // Show probe error inline in red
    if let Some(ref err) = state.nexus_probe_error {
        ui.add_space(4.0);
        ui.label(egui::RichText::new(err).size(11.0).color(crate::theme::semantic::RED));
        return;
    }

    if let Some(ref meta) = state.nexus_metadata {
        let tc = ThemeColors::from_ctx(ui.ctx());
        ui.add_space(4.0);
        if meta.has_histogram {
            if let Some(shape) = meta.histogram_shape {
                ui.label(
                    egui::RichText::new(format!(
                        "Histogram: {}×{}×{} (rot×y×x), {} TOF bins",
                        shape[0], shape[1], shape[2], shape[3]
                    ))
                    .size(11.0)
                    .color(tc.fg2),
                );
            }
        } else {
            ui.label(
                egui::RichText::new("No histogram data.")
                    .size(11.0)
                    .color(tc.fg3),
            );
        }

        if meta.has_events {
            if let Some(n) = meta.n_events {
                ui.label(
                    egui::RichText::new(format!("Events: {} neutrons", n))
                        .size(11.0)
                        .color(tc.fg2),
                );
            }
        } else {
            ui.label(
                egui::RichText::new("No event data.")
                    .size(11.0)
                    .color(tc.fg3),
            );
        }

        if let Some(fp) = meta.flight_path_m {
            ui.label(
                egui::RichText::new(format!("Flight path: {:.2} m", fp))
                    .size(11.0)
                    .color(tc.fg2),
            );
        }
    }
}

/// Display the HDF5 file tree structure in a collapsing header.
fn show_hdf5_tree(ui: &mut egui::Ui, state: &AppState) {
    let tree = match state.hdf5_tree {
        Some(ref t) if !t.is_empty() => t,
        _ => return,
    };

    ui.add_space(4.0);
    egui::CollapsingHeader::new("HDF5 Structure")
        .default_open(false)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    for entry in tree {
                        let depth = entry.path.matches('/').count().saturating_sub(1);
                        let indent = "  ".repeat(depth);
                        let name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
                        let label = match entry.kind {
                            nereids_io::nexus::Hdf5EntryKind::Group => {
                                format!("{indent}[G] {name}")
                            }
                            nereids_io::nexus::Hdf5EntryKind::Dataset => {
                                if let Some(ref shape) = entry.shape {
                                    format!("{indent}[D] {name} {:?}", shape)
                                } else {
                                    format!("{indent}[D] {name}")
                                }
                            }
                        };
                        ui.label(egui::RichText::new(label).monospace().small());
                    }
                });
        });
}

// ── Data loading logic ─────────────────────────────────────────

/// Show load error with retry button.
fn load_status_ui(ui: &mut egui::Ui, state: &mut AppState) {
    if state.load_error {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(&state.status_message).color(crate::theme::semantic::RED));
            if ui.button("Retry").clicked() {
                state.load_error = false;
                state.sample_data = None;
                state.open_beam_data = None;
                state.spectrum_values = None;
            }
        });
    }
}

/// Load histogram data from HDF5 file.
fn load_hdf5_histogram(state: &mut AppState) {
    let path = match state.hdf5_path {
        Some(ref p) => p.clone(),
        None => return,
    };

    state.invalidate_results();

    match nereids_io::nexus::load_nexus_histogram(&path) {
        Ok(data) => {
            let shape = data.counts.shape();
            state.preview_image = Some(data.counts.sum_axis(ndarray::Axis(0)));
            state.status_message = format!(
                "HDF5 histogram loaded: {} frames, {}×{} px",
                shape[0], shape[1], shape[2]
            );

            state.spectrum_values = Some(data.tof_edges_us.clone());
            state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;

            let n_frames = data.counts.shape()[0];
            let n_tof_vals = data.tof_edges_us.len();
            state.spectrum_kind = if n_tof_vals == n_frames + 1 {
                nereids_io::spectrum::SpectrumValueKind::BinEdges
            } else {
                nereids_io::spectrum::SpectrumValueKind::BinCenters
            };

            if let Some(fp) = data.flight_path_m
                && fp.is_finite()
                && fp > 0.0
            {
                state.beamline.flight_path_m = fp;
            }

            if let Some(offset_ns) = state.nexus_metadata.as_ref().and_then(|m| m.tof_offset_ns) {
                let delay_us = offset_ns / 1000.0;
                if delay_us.is_finite() {
                    state.beamline.delay_us = delay_us;
                }
            }

            if let Some(dead) = data.dead_pixels {
                state.dead_pixels = Some(dead);
            }

            let shape = data.counts.shape();
            state.log_provenance(
                ProvenanceEventKind::DataLoaded,
                format!(
                    "Loaded HDF5 histogram: {} frames ({}x{})",
                    shape[0], shape[1], shape[2]
                ),
            );
            state.sample_data = Some(data.counts);
        }
        Err(e) => {
            state.status_message = format!("HDF5 load failed: {e}");
            state.load_error = true;
        }
    }
}

/// Load all data: TIFF stacks + spectrum file with validation and auto-detect.
fn load_all_data(state: &mut AppState) {
    state.invalidate_results();

    // Load sample TIFF (auto-detect file vs directory)
    if let Some(ref path) = state.sample_path {
        match nereids_io::tiff_stack::load_tiff_auto(path) {
            Ok(data) => {
                state.preview_image = Some(data.sum_axis(Axis(0)));
                let n_frames = data.shape()[0];
                state.log_provenance(
                    ProvenanceEventKind::DataLoaded,
                    format!(
                        "Loaded sample TIFF: {n_frames} frames from {}",
                        path.display()
                    ),
                );
                state.sample_data = Some(data);
                state.status_message = "Sample loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to load sample: {}", e);
                state.load_error = true;
                return;
            }
        }
    }

    // Load open beam (TiffPair mode only)
    if state.input_mode == InputMode::TiffPair
        && let Some(ref path) = state.open_beam_path
    {
        match nereids_io::tiff_stack::load_tiff_auto(path) {
            Ok(data) => {
                state.open_beam_data = Some(data);
                state.status_message = "Sample and open beam loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to load open beam: {}", e);
                state.sample_data = None; // Clear partial data
                state.load_error = true;
                return;
            }
        }
    }

    // Validate frame count consistency between sample and open beam
    if let (Some(sample), Some(ob)) = (&state.sample_data, &state.open_beam_data)
        && sample.shape()[0] != ob.shape()[0]
    {
        state.status_message = format!(
            "Frame count mismatch: sample has {} frames, open beam has {}",
            sample.shape()[0],
            ob.shape()[0]
        );
        state.sample_data = None;
        state.open_beam_data = None;
        state.load_error = true;
        return;
    }

    // Parse spectrum file with auto-detect bin type
    if let Some(ref path) = state.spectrum_path {
        match nereids_io::spectrum::parse_spectrum_file(path) {
            Ok(values) => {
                // Validate monotonicity
                if let Err(e) = nereids_io::spectrum::validate_monotonic(&values) {
                    state.status_message = format!("Spectrum: {}", e);
                    state.load_error = true;
                    return;
                }

                // Auto-detect bin type from frame count
                let n_frames = state.sample_data.as_ref().map_or(0, |d| d.shape()[0]);
                let n_values = values.len();
                state.spectrum_kind = if n_values == n_frames + 1 {
                    nereids_io::spectrum::SpectrumValueKind::BinEdges
                } else if n_values == n_frames {
                    nereids_io::spectrum::SpectrumValueKind::BinCenters
                } else {
                    state.status_message = format!(
                        "Spectrum mismatch: {n_values} values vs {n_frames} frames \
                         (expected {f1} for BinEdges or {n_frames} for BinCenters)",
                        f1 = n_frames + 1,
                    );
                    state.load_error = true;
                    return;
                };

                // Default to TOF microseconds
                state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;

                // Validate with auto-detected kind
                if let Err(e) = nereids_io::spectrum::validate_spectrum_frame_count(
                    values.len(),
                    n_frames,
                    state.spectrum_kind,
                ) {
                    state.status_message = format!("Spectrum: {}", e);
                    state.load_error = true;
                    return;
                }

                state.spectrum_values = Some(values);
                state.status_message = "All data loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to parse spectrum: {}", e);
                state.load_error = true;
            }
        }
    }
}
