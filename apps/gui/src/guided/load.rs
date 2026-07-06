//! Step 1: Data loading — multi-format TIFF + spectrum file input.
//!
//! Prototype: `.content-area` → cards with drop zones, auto-load,
//! format hints, and a Continue button with data guard.

use std::sync::Arc;

use crate::file_dialog::{DialogIntent, DialogOptions};
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
        state.hdf5_ob_path = None;
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

        tiff_drop_zone(
            ui,
            &mut state.file_dialogs,
            DialogIntent::TiffSample,
            "Sample",
            &state.sample_path,
        );

        ui.add_space(6.0);
        tiff_drop_zone(
            ui,
            &mut state.file_dialogs,
            DialogIntent::TiffOpenBeam,
            "Open Beam",
            &state.open_beam_path,
        );

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

        // Same destination + invalidation as the raw sample stack, so the
        // pick shares DialogIntent::TiffSample.
        tiff_drop_zone(
            ui,
            &mut state.file_dialogs,
            DialogIntent::TiffSample,
            "Transmission",
            &state.sample_path,
        );

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
/// Opens a dialog tagged with `intent`; the pick (file or folder — both
/// land in the same path field) is applied by the dispatch handler.
fn tiff_drop_zone(
    ui: &mut egui::Ui,
    dialogs: &mut crate::file_dialog::FileDialogs,
    intent: DialogIntent,
    label: &str,
    path: &Option<std::path::PathBuf>,
) {
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
    if resp.clicked() {
        dialogs.pick_file(
            intent.clone(),
            DialogOptions {
                filters: vec![("TIFF", &["tif", "tiff"])],
                ..Default::default()
            },
        );
    }
    // Secondary: folder browse
    if ui.small_button("or browse folder\u{2026}").clicked() {
        dialogs.pick_folder(intent, Default::default());
    }
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
    if resp.clicked() {
        state.file_dialogs.pick_file(
            DialogIntent::SpectrumFile,
            DialogOptions {
                filters: vec![("Spectrum", &["csv", "txt", "dat"])],
                ..Default::default()
            },
        );
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

/// Display loaded data info with a Reload button.
fn show_loaded_info(ui: &mut egui::Ui, state: &mut AppState) {
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
    // Reload button: force re-read from disk
    if ui.small_button("\u{21bb} Reload").clicked() {
        state.sample_data = None;
        state.open_beam_data = None;
        state.load_error = false;
        state.nexus_probe_error = None;
        state.invalidate_results();
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

    // Open beam file (optional — enables counts-domain KL fitting)
    hdf5_ob_picker(ui, state);

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

    // Open beam file (optional — enables counts-domain KL fitting)
    hdf5_ob_picker(ui, state);

    show_hdf5_tree(ui, state);

    show_loaded_info(ui, state);
}

/// Optional open beam NeXus file picker for HDF5 modes.
/// When provided, enables proper normalization (T = sample/OB) and
/// counts-domain KL fitting.
fn hdf5_ob_picker(ui: &mut egui::Ui, state: &mut AppState) {
    design::card(ui, |ui| {
        ui.label(
            egui::RichText::new("Open beam (optional — enables counts-domain fitting)")
                .size(10.0)
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        ui.add_space(4.0);
        let loaded = state.open_beam_data.is_some();
        let display = state
            .hdf5_ob_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Click to select open beam...".to_string());
        let resp = design::drop_zone(ui, loaded, &display, "Open beam HDF5 (.h5, .hdf5, .nxs)");
        if resp.clicked() {
            // Capture the interpretation inputs (mode + binning) at
            // click time — see the `DialogIntent::Hdf5OpenBeam` doc
            // comment. Params are captured unconditionally: cheap, and
            // correct regardless of which mode is active at resolution.
            state.file_dialogs.pick_file(
                DialogIntent::Hdf5OpenBeam {
                    mode: state.input_mode,
                    event_params: nereids_io::nexus::EventBinningParams {
                        n_bins: state.event_n_bins,
                        tof_min_us: state.event_tof_min_us,
                        tof_max_us: state.event_tof_max_us,
                        height: state.event_height,
                        width: state.event_width,
                    },
                },
                DialogOptions {
                    filters: vec![("HDF5", &["h5", "hdf5", "nxs", "nx5"])],
                    ..Default::default()
                },
            );
        }
        if loaded && ui.small_button("Clear OB").clicked() {
            state.hdf5_ob_path = None;
            state.open_beam_data = None;
            // Same reasoning as the load path above — OB removal
            // invalidates downstream fit/normalization state only.
            state.invalidate_fit_results();
            state.status_message = "Open beam cleared".into();
        }
    });
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
    if resp.clicked() {
        state.file_dialogs.pick_file(
            DialogIntent::Hdf5Sample,
            DialogOptions {
                filters: vec![("NeXus/HDF5", &["h5", "hdf5", "nxs", "nx5"])],
                ..Default::default()
            },
        );
    }
}

/// Display probed NeXus metadata (or inline probe error).
fn show_nexus_metadata(ui: &mut egui::Ui, state: &AppState) {
    // Show probe error inline in red
    if let Some(ref err) = state.nexus_probe_error {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(err)
                .size(11.0)
                .color(crate::theme::semantic::RED),
        );
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

    // Issue #430: explicit opt-in to the legacy sum-over-angles
    // behaviour on the sample-data load.  The status banner below
    // tells the user how many angles were combined.
    match nereids_io::nexus::load_nexus_histogram_with_mode(
        &path,
        nereids_io::nexus::MultiAngleMode::Sum,
    ) {
        Ok(data) => {
            let shape = data.counts.shape();
            state.preview_image = Some(data.counts.sum_axis(ndarray::Axis(0)));
            // D-5: Report rotation angle count when angles were collapsed.
            let angle_note = if data.n_rotation_angles > 1 {
                format!(
                    " ({} rotation angles summed — multi-angle analysis not yet supported, \
                     see #430)",
                    data.n_rotation_angles
                )
            } else {
                String::new()
            };
            state.status_message = format!(
                "HDF5 histogram loaded: {} frames, {}×{} px{angle_note}",
                shape[0], shape[1], shape[2]
            );

            state.spectrum_values = Some(Arc::new(data.tof_edges_us.clone()));
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

            // Declared-mask provenance (#646): the file-declared mask gets
            // its own field (assigned unconditionally — a file without a
            // mask must not inherit a previous file's), and the effective
            // mask starts as exactly that.  Every normalization recomputes
            // dead_pixels = declared ∪ detected from scratch
            // (AppState::set_detected_dead_pixels).  No detection has run
            // on the fresh data yet.
            state.file_dead_pixels = data.dead_pixels.clone();
            state.dead_pixels = data.dead_pixels;
            state.detected_dead_pixels = None;

            let shape = data.counts.shape();
            state.log_provenance(
                ProvenanceEventKind::DataLoaded,
                format!(
                    "Loaded HDF5 histogram: {} frames ({}x{})",
                    shape[0], shape[1], shape[2]
                ),
            );
            state.sample_data = Some(Arc::new(data.counts));
        }
        Err(e) => {
            state.status_message = format!("HDF5 load failed: {e}");
            state.load_error = true;
        }
    }
}

/// Format the chunk-summing / chunk-concatenation / stray-file /
/// inconsistent-chunk / pixel-clipping provenance suffix for a TIFF load, or
/// an empty string when nothing noteworthy happened.
fn tiff_load_suffix(info: &nereids_io::tiff_stack::TiffLoadInfo) -> String {
    let mut suffix = String::new();
    if info.n_chunks > 1 {
        let ids = info
            .chunk_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        if info.chunks_summed {
            suffix.push_str(&format!(", summed {} DAQ chunks ({ids})", info.n_chunks));
        } else {
            // Transmission mode loads with sum_chunks=false; without this
            // line a mis-picked raw chunked run folder loads as a k×
            // concatenated stack with zero provenance — the only downstream
            // symptom is a spectrum-count mismatch that never mentions
            // chunks.
            suffix.push_str(&format!(
                ", detected {} DAQ chunks ({ids}) — NOT summed (transmission \
                 mode); frames are concatenated",
                info.n_chunks,
            ));
        }
    }
    if info.n_unrecognized_files > 0 {
        suffix.push_str(&format!(
            ", {} file(s) did not match the chunk naming pattern (e.g. {}) — \
             chunk detection disabled, frames concatenated lexicographically",
            info.n_unrecognized_files,
            info.unrecognized_examples.join(", "),
        ));
    }
    if info.chunk_inconsistent {
        // The files ARE chunk-patterned but internally inconsistent (ragged
        // frame counts/sets or a duplicate (chunk, frame) pair).  Under the
        // default summing path this is a hard error; because transmission
        // mode loads with sum_chunks=false, it instead loaded verbatim as a
        // lexicographic concatenation — which the operator must see, since a
        // ragged raw run folder that would otherwise fail is now a silent
        // plain stack whose only symptom is a spectrum-count mismatch.
        suffix.push_str(
            " — WARNING: DAQ chunks are internally inconsistent (ragged frame \
             counts/sets or duplicate frames); loaded as a plain lexicographic \
             concatenation, not summed",
        );
    }
    if info.n_clipped_pixels > 0 {
        suffix.push_str(&format!(
            ", {} negative pixels clipped to 0",
            info.n_clipped_pixels
        ));
    }
    suffix
}

/// Load all data: TIFF stacks + spectrum file with validation and auto-detect.
fn load_all_data(state: &mut AppState) {
    state.invalidate_results();

    // Pre-normalized transmission stacks legitimately contain small
    // negative values (noise around zero), so they bypass the raw-counts
    // pixel guard; raw counts keep the strict default.  Chunk summing is
    // likewise counts semantics: element-wise summing k per-chunk
    // *transmission* stacks yields values near k (e.g. ~2.0 for 2 chunks),
    // which is physically meaningless — so TransmissionTiff mode also
    // disables it, exactly mirroring the pixel-policy switch.
    let options = if state.input_mode == InputMode::TransmissionTiff {
        nereids_io::tiff_stack::TiffFolderOptions {
            pixel_policy: nereids_io::tiff_stack::PixelValuePolicy::Allow,
            sum_chunks: false,
        }
    } else {
        // Raw counts: Reject bad pixels, sum DAQ chunks (the defaults).
        nereids_io::tiff_stack::TiffFolderOptions::default()
    };

    // Load sample TIFF (auto-detect file vs directory)
    if let Some(ref path) = state.sample_path {
        match nereids_io::tiff_stack::load_tiff_auto_with_options(path, &options) {
            Ok((data, info)) => {
                state.preview_image = Some(data.sum_axis(Axis(0)));
                let n_frames = data.shape()[0];
                state.log_provenance(
                    ProvenanceEventKind::DataLoaded,
                    format!(
                        "Loaded sample TIFF: {n_frames} frames from {}{}",
                        path.display(),
                        tiff_load_suffix(&info),
                    ),
                );
                state.sample_data = Some(Arc::new(data));
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
        match nereids_io::tiff_stack::load_tiff_auto_with_options(path, &options) {
            Ok((data, info)) => {
                let suffix = tiff_load_suffix(&info);
                if !suffix.is_empty() {
                    state.log_provenance(
                        ProvenanceEventKind::DataLoaded,
                        format!(
                            "Loaded open-beam TIFF: {} frames from {}{}",
                            data.shape()[0],
                            path.display(),
                            suffix,
                        ),
                    );
                }
                state.open_beam_data = Some(Arc::new(data));
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
        // VENUS sidecar auto-detect: *_Spectra.txt files hold frame START
        // times in SECONDS, not verbatim-µs edges — feeding them to the
        // plain parser would be a 10^6 unit error, so the sidecar reader
        // handles them and a sidecar parse failure is surfaced, never
        // silently retried with the verbatim parser.  Detection is by
        // filename suffix only; an explicit format selector in the UI is a
        // possible follow-up, so the provenance log records which parser
        // was chosen and why.
        let is_sidecar = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.to_lowercase().ends_with("_spectra.txt"));
        if is_sidecar {
            // Cross-check the sidecar frame count against the sample stack only
            // when it is already loaded; with no stack yet, pass `None` so a
            // valid sidecar parses instead of failing the count check against a
            // spurious n_frames = 0.
            let n_frames = state.sample_data.as_ref().map(|d| d.shape()[0]);
            match nereids_io::spectrum::read_tof_sidecar(path, n_frames) {
                Ok(edges) => {
                    state.spectrum_kind = nereids_io::spectrum::SpectrumValueKind::BinEdges;
                    state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;
                    state.log_provenance(
                        ProvenanceEventKind::DataLoaded,
                        format!(
                            "Spectrum parser: TOF sidecar (filename ends in \
                             _Spectra.txt): {} frame start times in seconds \
                             converted to {} µs bin edges",
                            edges.len() - 1,
                            edges.len(),
                        ),
                    );
                    state.spectrum_values = Some(Arc::new(edges));
                    state.status_message = "All data loaded".into();
                }
                Err(e) => {
                    state.status_message = format!("Failed to parse TOF sidecar: {}", e);
                    state.load_error = true;
                }
            }
            return;
        }
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

                state.log_provenance(
                    ProvenanceEventKind::DataLoaded,
                    format!(
                        "Spectrum parser: verbatim spectrum file (no \
                         _Spectra.txt suffix): {} values used as-is (µs/eV), \
                         detected as {:?}",
                        values.len(),
                        state.spectrum_kind,
                    ),
                );
                state.spectrum_values = Some(Arc::new(values));
                state.status_message = "All data loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to parse spectrum: {}", e);
                state.load_error = true;
            }
        }
    }
}

// ── Dialog-dispatch handlers ───────────────────────────────────
// Invoked from `crate::file_dialog::dispatch_results` when a Load-step
// picker resolves. Each mirrors the invalidation set the corresponding
// inline block used before the picks were routed through the facade.

/// Sample TIFF stack picked (raw pair tab), or pre-normalized
/// transmission stack picked — both land in `sample_path` with the same
/// downstream invalidation.
pub(crate) fn on_tiff_sample_picked(state: &mut AppState, path: std::path::PathBuf) {
    state.sample_path = Some(path);
    state.sample_data = None;
    state.normalized = None;
    state.dead_pixels = None;
    state.file_dead_pixels = None;
    state.detected_dead_pixels = None;
    state.energies = None;
    state.pixel_fit_result = None;
    state.spatial_result = None;
    state.load_error = false;
}

/// Open-beam TIFF stack picked.
pub(crate) fn on_tiff_open_beam_picked(state: &mut AppState, path: std::path::PathBuf) {
    state.open_beam_path = Some(path);
    state.open_beam_data = None;
    state.normalized = None;
    state.dead_pixels = None;
    state.file_dead_pixels = None;
    state.detected_dead_pixels = None;
    state.energies = None;
    state.pixel_fit_result = None;
    state.spatial_result = None;
    state.load_error = false;
}

/// TOF spectrum file picked.
pub(crate) fn on_spectrum_picked(state: &mut AppState, path: std::path::PathBuf) {
    state.spectrum_path = Some(path);
    state.spectrum_values = None;
    state.sample_data = None;
    state.open_beam_data = None;
    state.energies = None;
    state.normalized = None;
    state.load_error = false;
}

/// Sample NeXus/HDF5 file picked: reset derived state, probe the file,
/// and build the tree view.
pub(crate) fn on_hdf5_sample_picked(state: &mut AppState, file: std::path::PathBuf) {
    state.hdf5_path = Some(file.clone());
    state.invalidate_results();
    state.sample_data = None;
    state.open_beam_data = None;
    state.hdf5_ob_path = None;
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

/// Optional open-beam NeXus/HDF5 file picked: load (event or histogram
/// per the mode captured at request time), validate against the
/// current sample, and install.
pub(crate) fn on_hdf5_ob_picked(
    state: &mut AppState,
    path: std::path::PathBuf,
    mode: InputMode,
    event_params: &nereids_io::nexus::EventBinningParams,
) {
    state.hdf5_ob_path = Some(path.clone());
    // Event-vs-histogram dispatch and binning use ONLY the values
    // captured when the dialog was requested (`mode`, `event_params`) —
    // state edits made while the dialog is pending must not change how
    // the picked file is loaded. The shape/TOF validation below reads
    // CURRENT state on purpose: the OB must match whatever sample is
    // loaded now.
    let ob_result = if mode == InputMode::Hdf5Event {
        nereids_io::nexus::load_nexus_events(&path, event_params)
            .map(|d| (d.counts, d.tof_edges_us, d.n_rotation_angles))
    } else {
        // Issue #430: the loader refuses multi-angle files by
        // default to prevent silent sum-over-angles.  The GUI
        // makes the explicit opt-in to preserve existing
        // single-volume analysis behaviour on OB input.
        //
        // Issue #462: surface the rotation-angle count to the
        // user via the status banner, mirroring the sample-load
        // path.  `data.n_rotation_angles` is carried through the
        // tuple so the success arm can build the same `angle_note`
        // suffix.
        nereids_io::nexus::load_nexus_histogram_with_mode(
            &path,
            nereids_io::nexus::MultiAngleMode::Sum,
        )
        .map(|d| (d.counts, d.tof_edges_us, d.n_rotation_angles))
    };
    match ob_result {
        Ok((ob_counts, ob_tof_edges, n_rotation_angles)) => {
            // P1-7: Validate shape matches sample
            if let Some(ref sample) = state.sample_data
                && sample.shape() != ob_counts.shape()
            {
                state.status_message = format!(
                    "OB shape {:?} != sample {:?}",
                    ob_counts.shape(),
                    sample.shape()
                );
                state.hdf5_ob_path = None;
                state.open_beam_data = None;
                return;
            }
            // Validate TOF grid matches sample — reject if edges differ.
            // Positional bin pairing in the counts path is only correct
            // when sample and OB share the exact same TOF grid.
            if let Some(ref sv) = state.spectrum_values {
                if ob_tof_edges.len() != sv.len() {
                    state.status_message = format!(
                        "OB rejected: {} TOF edges vs sample {} — grids must match",
                        ob_tof_edges.len(),
                        sv.len()
                    );
                    state.hdf5_ob_path = None;
                    state.open_beam_data = None;
                    return;
                }
                let max_edge_diff: f64 = ob_tof_edges
                    .iter()
                    .zip(sv.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                let edge_scale = sv.last().copied().unwrap_or(1.0).max(1.0);
                if max_edge_diff > 1e-6 * edge_scale {
                    state.status_message = format!(
                        "OB rejected: TOF edges differ (max delta = {:.3} µs) — \
                         sample and OB must use the same TOF grid",
                        max_edge_diff
                    );
                    state.hdf5_ob_path = None;
                    state.open_beam_data = None;
                    return;
                }
            }
            state.open_beam_data = Some(Arc::new(ob_counts));
            // OB swap invalidates normalization + fit results,
            // but NOT the sample's TOF spectrum, energies,
            // preview image, ROIs, or rebin state.  The broad
            // `invalidate_results()` would clear `spectrum_values`,
            // which gates the Continue button on Hdf5Histogram
            // mode (`has_required_data`) — using the narrow
            // variant keeps Continue enabled.
            state.invalidate_fit_results();
            // #462: mirror the sample-load rotation-angle note
            // so OB and sample banners use identical wording
            // when angles were collapsed.
            let angle_note = if n_rotation_angles > 1 {
                format!(
                    " ({} rotation angles summed — multi-angle analysis not yet supported, \
                     see #430)",
                    n_rotation_angles
                )
            } else {
                String::new()
            };
            state.status_message = format!("Open beam loaded{angle_note}");
        }
        Err(e) => {
            state.status_message = format!("Open beam load failed: {e}");
            state.hdf5_ob_path = None;
            // P1-8: Clear data on failure
            state.open_beam_data = None;
        }
    }
}
