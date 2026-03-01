//! Step 1: Data loading — multi-format TIFF + spectrum file input.

use crate::state::{AppState, InputMode};
use ndarray::Axis;

/// Draw the Load step content.
pub fn load_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Load Data");
    ui.separator();

    // Input mode tabs — invalidate results when switching modes
    let prev_mode = state.input_mode;
    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut state.input_mode,
            InputMode::TiffPair,
            "TIFF Pair + Spectrum",
        );
        ui.selectable_value(
            &mut state.input_mode,
            InputMode::TransmissionTiff,
            "Transmission TIFF",
        );
        ui.add_enabled_ui(false, |ui| {
            ui.label("HDF5 Event");
            ui.label("HDF5 Histogram");
        });
    });
    if state.input_mode != prev_mode {
        state.invalidate_results();
        state.sample_data = None;
        state.open_beam_data = None;
    }

    ui.add_space(8.0);

    match state.input_mode {
        InputMode::TiffPair => tiff_pair_tab(ui, state),
        InputMode::TransmissionTiff => transmission_tiff_tab(ui, state),
    }
}

/// TIFF Pair tab: Sample + Open Beam + Spectrum file.
fn tiff_pair_tab(ui: &mut egui::Ui, state: &mut AppState) {
    if tiff_browse_row(ui, "Sample", &mut state.sample_path) {
        state.sample_data = None;
        state.normalized = None;
        state.dead_pixels = None;
        state.energies = None;
        state.pixel_fit_result = None;
        state.spatial_result = None;
    }
    if tiff_browse_row(ui, "Open Beam", &mut state.open_beam_path) {
        state.open_beam_data = None;
        state.normalized = None;
        state.dead_pixels = None;
        state.energies = None;
        state.pixel_fit_result = None;
        state.spatial_result = None;
    }

    ui.add_space(8.0);
    spectrum_input_section(ui, state);

    ui.add_space(8.0);

    // Load button
    let can_load = state.sample_path.is_some()
        && state.open_beam_path.is_some()
        && state.spectrum_path.is_some();
    ui.add_enabled_ui(can_load, |ui| {
        if ui.button("Load All").clicked() {
            load_all_data(state);
        }
    });

    show_loaded_info(ui, state);
}

/// Transmission TIFF tab: pre-normalized TIFF + Spectrum file.
fn transmission_tiff_tab(ui: &mut egui::Ui, state: &mut AppState) {
    if tiff_browse_row(ui, "Transmission", &mut state.sample_path) {
        state.sample_data = None;
        state.normalized = None;
        state.dead_pixels = None;
        state.energies = None;
        state.pixel_fit_result = None;
        state.spatial_result = None;
    }

    ui.add_space(8.0);
    spectrum_input_section(ui, state);

    ui.add_space(8.0);

    // Load button
    let can_load = state.sample_path.is_some() && state.spectrum_path.is_some();
    ui.add_enabled_ui(can_load, |ui| {
        if ui.button("Load All").clicked() {
            load_all_data(state);
        }
    });

    show_loaded_info(ui, state);
}

/// A TIFF browse row with both "Browse Folder..." and "File..." buttons.
///
/// Returns `true` if the user selected a new path (callers should invalidate
/// dependent state).
fn tiff_browse_row(ui: &mut egui::Ui, label: &str, path: &mut Option<std::path::PathBuf>) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(format!("{}:", label));
        match path.as_ref() {
            Some(p) => {
                let kind = if p.is_file() { "file" } else { "dir" };
                let name = p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                ui.label(format!("[{}] {}", kind, name));
            }
            None => {
                ui.label("(none)");
            }
        }

        if ui.button("Browse Folder...").clicked()
            && let Some(dir) = rfd::FileDialog::new().pick_folder()
        {
            *path = Some(dir);
            changed = true;
        }
        if ui.small_button("File...").clicked()
            && let Some(file) = rfd::FileDialog::new()
                .add_filter("TIFF", &["tif", "tiff"])
                .pick_file()
        {
            *path = Some(file);
            changed = true;
        }
    });
    changed
}

/// Spectrum file input section: browse, unit selector, value kind selector.
fn spectrum_input_section(ui: &mut egui::Ui, state: &mut AppState) {
    ui.label(egui::RichText::new("Spectrum File").strong());

    ui.horizontal(|ui| {
        ui.label("File:");
        if let Some(ref p) = state.spectrum_path {
            ui.label(
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            );
        } else {
            ui.label("(none)");
        }
        if ui.button("Browse...").clicked()
            && let Some(file) = rfd::FileDialog::new()
                .add_filter("Spectrum", &["csv", "txt", "dat"])
                .pick_file()
        {
            state.spectrum_path = Some(file);
            state.spectrum_values = None;
            state.energies = None;
            state.normalized = None;
        }
    });

    let prev_unit = state.spectrum_unit;
    ui.horizontal(|ui| {
        ui.label("Values are:");
        ui.selectable_value(
            &mut state.spectrum_unit,
            nereids_io::spectrum::SpectrumUnit::TofMicroseconds,
            "TOF (\u{03bc}s)",
        );
        ui.selectable_value(
            &mut state.spectrum_unit,
            nereids_io::spectrum::SpectrumUnit::EnergyEv,
            "Energy (eV)",
        );
    });

    let prev_kind = state.spectrum_kind;
    ui.horizontal(|ui| {
        ui.label("Value type:");
        ui.selectable_value(
            &mut state.spectrum_kind,
            nereids_io::spectrum::SpectrumValueKind::BinEdges,
            "Bin edges",
        );
        ui.selectable_value(
            &mut state.spectrum_kind,
            nereids_io::spectrum::SpectrumValueKind::BinCenters,
            "Bin centers",
        );
    });

    // Invalidate derived data when unit/kind settings change
    let kind_changed = state.spectrum_kind != prev_kind;
    if state.spectrum_unit != prev_unit || kind_changed {
        state.energies = None;
        state.normalized = None;
    }
    // Changing edges↔centers alters the expected value count vs frame count,
    // so previously loaded spectrum values may no longer be valid.
    if kind_changed {
        state.spectrum_values = None;
    }

    if let Some(ref vals) = state.spectrum_values {
        ui.label(format!(
            "Parsed: {} values, range [{:.2}, {:.2}]",
            vals.len(),
            vals.first().copied().unwrap_or(0.0),
            vals.last().copied().unwrap_or(0.0),
        ));
    }
}

/// Display loaded data info.
fn show_loaded_info(ui: &mut egui::Ui, state: &AppState) {
    ui.add_space(8.0);
    if let Some(ref data) = state.sample_data {
        let shape = data.shape();
        ui.label(format!(
            "Sample: {} frames, {}×{} px",
            shape[0], shape[1], shape[2]
        ));
    }
    if let Some(ref data) = state.open_beam_data {
        let shape = data.shape();
        ui.label(format!(
            "Open Beam: {} frames, {}×{} px",
            shape[0], shape[1], shape[2]
        ));
    }
}

/// Load all data: TIFF stacks + spectrum file with validation.
fn load_all_data(state: &mut AppState) {
    state.invalidate_results();

    // Load sample TIFF (auto-detect file vs directory)
    if let Some(ref path) = state.sample_path {
        match nereids_io::tiff_stack::load_tiff_auto(path) {
            Ok(data) => {
                state.preview_image = Some(data.sum_axis(Axis(0)));
                state.sample_data = Some(data);
                state.status_message = "Sample loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to load sample: {}", e);
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
        return;
    }

    // Parse spectrum file
    if let Some(ref path) = state.spectrum_path {
        match nereids_io::spectrum::parse_spectrum_file(path) {
            Ok(values) => {
                // Validate monotonicity
                if let Err(e) = nereids_io::spectrum::validate_monotonic(&values) {
                    state.status_message = format!("Spectrum: {}", e);
                    return;
                }

                // Validate frame count compatibility
                let n_frames = state.sample_data.as_ref().map_or(0, |d| d.shape()[0]);
                if let Err(e) = nereids_io::spectrum::validate_spectrum_frame_count(
                    values.len(),
                    n_frames,
                    state.spectrum_kind,
                ) {
                    state.status_message = format!("Spectrum: {}", e);
                    return;
                }

                state.spectrum_values = Some(values);
                state.status_message = "All data loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to parse spectrum: {}", e);
            }
        }
    }
}
