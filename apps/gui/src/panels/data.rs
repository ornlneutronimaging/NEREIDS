//! Data loading panel: TIFF directory selection, normalization, TOF→energy.

use crate::state::AppState;
use ndarray::Axis;

/// Draw the data loading panel in the left sidebar.
pub fn data_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Data Loading");
    ui.separator();

    // --- Sample directory ---
    ui.horizontal(|ui| {
        ui.label("Sample:");
        if let Some(ref path) = state.sample_path {
            ui.label(path.file_name().unwrap_or_default().to_string_lossy().to_string());
        } else {
            ui.label("(none)");
        }
        if ui.button("Browse...").clicked() {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                state.sample_path = Some(dir);
                state.sample_data = None;
                state.normalized = None;
                state.status_message = "Sample directory selected".into();
            }
        }
    });

    // --- Open beam directory ---
    ui.horizontal(|ui| {
        ui.label("Open Beam:");
        if let Some(ref path) = state.open_beam_path {
            ui.label(path.file_name().unwrap_or_default().to_string_lossy().to_string());
        } else {
            ui.label("(none)");
        }
        if ui.button("Browse...").clicked() {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                state.open_beam_path = Some(dir);
                state.open_beam_data = None;
                state.normalized = None;
                state.status_message = "Open beam directory selected".into();
            }
        }
    });

    ui.add_space(8.0);

    // --- Load button ---
    let can_load = state.sample_path.is_some() && state.open_beam_path.is_some();
    ui.add_enabled_ui(can_load, |ui| {
        if ui.button("Load TIFF Stacks").clicked() {
            load_tiff_data(state);
        }
    });

    // --- Show loaded data info ---
    if let Some(ref data) = state.sample_data {
        let shape = data.shape();
        ui.label(format!(
            "Sample: {} frames, {}x{} px",
            shape[0], shape[1], shape[2]
        ));
    }
    if let Some(ref data) = state.open_beam_data {
        let shape = data.shape();
        ui.label(format!(
            "Open Beam: {} frames, {}x{} px",
            shape[0], shape[1], shape[2]
        ));
    }

    ui.add_space(8.0);
    ui.separator();

    // --- Beamline parameters ---
    ui.heading("Beamline");
    ui.horizontal(|ui| {
        ui.label("Flight path (m):");
        ui.add(egui::DragValue::new(&mut state.beamline.flight_path_m).range(1.0..=100.0).speed(0.1));
    });
    ui.horizontal(|ui| {
        ui.label("Delay (us):");
        ui.add(egui::DragValue::new(&mut state.beamline.delay_us).range(0.0..=1000.0).speed(0.1));
    });
    ui.horizontal(|ui| {
        ui.label("PC sample:");
        ui.add(egui::DragValue::new(&mut state.proton_charge_sample).range(0.001..=1e6).speed(0.01));
    });
    ui.horizontal(|ui| {
        ui.label("PC open beam:");
        ui.add(egui::DragValue::new(&mut state.proton_charge_ob).range(0.001..=1e6).speed(0.01));
    });

    ui.add_space(8.0);

    // --- Normalize button ---
    let can_normalize = state.sample_data.is_some() && state.open_beam_data.is_some();
    ui.add_enabled_ui(can_normalize, |ui| {
        if ui.button("Normalize").clicked() {
            normalize_data(state);
        }
    });

    if state.normalized.is_some() {
        ui.label("Transmission computed");
        if let Some(ref dead) = state.dead_pixels {
            let n_dead = dead.iter().filter(|&&d| d).count();
            ui.label(format!("Dead pixels: {}", n_dead));
        }
    }
}

fn load_tiff_data(state: &mut AppState) {
    if let Some(ref sample_dir) = state.sample_path {
        match nereids_io::tiff_stack::load_tiff_directory(sample_dir) {
            Ok(data) => {
                // Build preview image: sum over TOF axis
                let sum = data.sum_axis(Axis(0));
                state.preview_image = Some(sum);
                state.sample_data = Some(data);
                state.status_message = "Sample loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to load sample: {}", e);
                return;
            }
        }
    }

    if let Some(ref ob_dir) = state.open_beam_path {
        match nereids_io::tiff_stack::load_tiff_directory(ob_dir) {
            Ok(data) => {
                state.open_beam_data = Some(data);
                state.status_message = "Sample and open beam loaded".into();
            }
            Err(e) => {
                state.status_message = format!("Failed to load open beam: {}", e);
            }
        }
    }
}

fn normalize_data(state: &mut AppState) {
    let sample = match state.sample_data {
        Some(ref d) => d,
        None => return,
    };
    let open_beam = match state.open_beam_data {
        Some(ref d) => d,
        None => return,
    };

    let params = nereids_io::normalization::NormalizationParams {
        proton_charge_sample: state.proton_charge_sample,
        proton_charge_ob: state.proton_charge_ob,
    };

    match nereids_io::normalization::normalize(sample, open_beam, &params, None) {
        Ok(norm) => {
            // Detect dead pixels
            state.dead_pixels = Some(nereids_io::normalization::detect_dead_pixels(sample));

            // Compute energy grid from TOF bins
            let n_tof = sample.shape()[0];
            let tof_edges = nereids_io::tof::linspace_tof_edges(
                state.tof_min_us,
                state.tof_max_us,
                n_tof,
            );
            match nereids_io::tof::tof_edges_to_energy_centers(&tof_edges, &state.beamline) {
                Ok(e) => state.energies = Some(e.to_vec()),
                Err(e) => {
                    state.status_message = format!("TOF→energy error: {}", e);
                    return;
                }
            }

            state.normalized = Some(norm);
            state.status_message = "Normalization complete".into();
        }
        Err(e) => {
            state.status_message = format!("Normalization error: {}", e);
        }
    }
}
