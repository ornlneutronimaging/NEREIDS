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
        ui.selectable_value(
            &mut state.input_mode,
            InputMode::Hdf5Histogram,
            "HDF5 Histogram",
        );
        ui.selectable_value(&mut state.input_mode, InputMode::Hdf5Event, "HDF5 Event");
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
        InputMode::Hdf5Histogram => hdf5_histogram_tab(ui, state),
        InputMode::Hdf5Event => hdf5_event_tab(ui, state),
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

/// HDF5 Histogram tab: load pre-histogrammed NeXus data.
fn hdf5_histogram_tab(ui: &mut egui::Ui, state: &mut AppState) {
    ui.label("Browse for a NeXus/HDF5 file containing histogram data.");
    hdf5_browse_row(ui, state);
    show_nexus_metadata(ui, state);

    if state.hdf5_path.is_some()
        && state
            .nexus_metadata
            .as_ref()
            .is_some_and(|m| m.has_histogram)
    {
        ui.add_space(8.0);
        if ui.button("Load Histogram").clicked() {
            load_hdf5_histogram(state);
        }
    }

    show_loaded_info(ui, state);
}

/// HDF5 Event tab: load raw neutron events and histogram them.
fn hdf5_event_tab(ui: &mut egui::Ui, state: &mut AppState) {
    ui.label("Browse for a NeXus/HDF5 file containing neutron event data.");
    hdf5_browse_row(ui, state);
    show_nexus_metadata(ui, state);

    if state.hdf5_path.is_some() && state.nexus_metadata.as_ref().is_some_and(|m| m.has_events) {
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Event Binning Parameters").strong());
        ui.horizontal(|ui| {
            ui.label("TOF bins:");
            ui.add(egui::DragValue::new(&mut state.event_n_bins).range(1..=10000));
        });
        ui.horizontal(|ui| {
            ui.label("TOF min (µs):");
            ui.add(egui::DragValue::new(&mut state.event_tof_min_us).speed(10.0));
        });
        ui.horizontal(|ui| {
            ui.label("TOF max (µs):");
            ui.add(egui::DragValue::new(&mut state.event_tof_max_us).speed(10.0));
        });
        ui.horizontal(|ui| {
            ui.label("Height (px):");
            ui.add(egui::DragValue::new(&mut state.event_height).range(1..=4096));
        });
        ui.horizontal(|ui| {
            ui.label("Width (px):");
            ui.add(egui::DragValue::new(&mut state.event_width).range(1..=4096));
        });

        ui.add_space(4.0);
        if ui.button("Histogram Events").clicked() {
            load_hdf5_events(state);
        }
    }

    show_loaded_info(ui, state);
}

/// HDF5 file browse row with probe.
fn hdf5_browse_row(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label("File:");
        if let Some(ref p) = state.hdf5_path {
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
                .add_filter("NeXus/HDF5", &["h5", "hdf5", "nxs", "nx5"])
                .pick_file()
        {
            state.hdf5_path = Some(file.clone());
            state.invalidate_results();
            state.sample_data = None;
            state.open_beam_data = None;

            // Probe the file immediately
            match nereids_io::nexus::probe_nexus(&file) {
                Ok(meta) => {
                    // Auto-fill event binning from metadata if available
                    if let Some(shape) = meta.histogram_shape {
                        state.event_height = shape[1];
                        state.event_width = shape[2];
                    }
                    state.nexus_metadata = Some(meta);
                    state.status_message = "NeXus file probed".into();
                }
                Err(e) => {
                    state.nexus_metadata = None;
                    state.status_message = format!("Probe failed: {e}");
                }
            }
        }
    });
}

/// Display probed NeXus metadata.
fn show_nexus_metadata(ui: &mut egui::Ui, state: &AppState) {
    if let Some(ref meta) = state.nexus_metadata {
        ui.add_space(4.0);
        if meta.has_histogram {
            if let Some(shape) = meta.histogram_shape {
                ui.label(format!(
                    "Histogram: {}×{}×{} (rot×y×x), {} TOF bins",
                    shape[0], shape[1], shape[2], shape[3]
                ));
            }
        } else {
            ui.label("No histogram data.");
        }

        if meta.has_events {
            if let Some(n) = meta.n_events {
                ui.label(format!("Events: {} neutrons", n));
            }
        } else {
            ui.label("No event data.");
        }

        if let Some(fp) = meta.flight_path_m {
            ui.label(format!("Flight path: {:.2} m", fp));
        }
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

            // Populate spectrum values from TOF edges
            state.spectrum_values = Some(data.tof_edges_us.clone());
            state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;

            // Determine whether TOF values are bin edges or bin centers
            let n_frames = data.counts.shape()[0]; // (tof, y, x)
            let n_tof_vals = data.tof_edges_us.len();
            state.spectrum_kind = if n_tof_vals == n_frames + 1 {
                nereids_io::spectrum::SpectrumValueKind::BinEdges
            } else {
                nereids_io::spectrum::SpectrumValueKind::BinCenters
            };

            // Set flight path if available and valid
            if let Some(fp) = data.flight_path_m
                && fp.is_finite()
                && fp > 0.0
            {
                state.beamline.flight_path_m = fp;
            }

            // Apply TOF offset as beamline delay
            if let Some(offset_ns) = state.nexus_metadata.as_ref().and_then(|m| m.tof_offset_ns) {
                let delay_us = offset_ns / 1000.0;
                if delay_us.is_finite() {
                    state.beamline.delay_us = delay_us;
                }
            }

            if let Some(dead) = data.dead_pixels {
                state.dead_pixels = Some(dead);
            }

            state.sample_data = Some(data.counts);
        }
        Err(e) => {
            state.status_message = format!("HDF5 load failed: {e}");
        }
    }
}

/// Load event data from HDF5 file with histogramming.
fn load_hdf5_events(state: &mut AppState) {
    let path = match state.hdf5_path {
        Some(ref p) => p.clone(),
        None => return,
    };

    state.invalidate_results();

    let params = nereids_io::nexus::EventBinningParams {
        n_bins: state.event_n_bins,
        tof_min_us: state.event_tof_min_us,
        tof_max_us: state.event_tof_max_us,
        height: state.event_height,
        width: state.event_width,
    };

    match nereids_io::nexus::load_nexus_events(&path, &params) {
        Ok(data) => {
            let shape = data.counts.shape();
            state.preview_image = Some(data.counts.sum_axis(ndarray::Axis(0)));
            state.status_message = format!(
                "Events histogrammed: {} bins, {}×{} px",
                shape[0], shape[1], shape[2]
            );

            // Populate spectrum values from TOF edges
            state.spectrum_values = Some(data.tof_edges_us.clone());
            state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;
            state.spectrum_kind = nereids_io::spectrum::SpectrumValueKind::BinEdges;

            // Set flight path if available and valid
            if let Some(fp) = data.flight_path_m
                && fp.is_finite()
                && fp > 0.0
            {
                state.beamline.flight_path_m = fp;
            }

            // Apply TOF offset as beamline delay
            if let Some(offset_ns) = state.nexus_metadata.as_ref().and_then(|m| m.tof_offset_ns) {
                let delay_us = offset_ns / 1000.0;
                if delay_us.is_finite() {
                    state.beamline.delay_us = delay_us;
                }
            }

            if let Some(dead) = data.dead_pixels {
                state.dead_pixels = Some(dead);
            }

            state.sample_data = Some(data.counts);
        }
        Err(e) => {
            state.status_message = format!("Event histogramming failed: {e}");
        }
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
