//! Step 1: Data loading — TIFF directory selection and loading.

use crate::state::AppState;
use ndarray::Axis;

/// Draw the Load step content.
pub fn load_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Load Data");
    ui.separator();

    // --- Sample directory ---
    ui.horizontal(|ui| {
        ui.label("Sample:");
        if let Some(ref path) = state.sample_path {
            ui.label(
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            );
        } else {
            ui.label("(none)");
        }
        if ui.button("Browse...").clicked()
            && let Some(dir) = rfd::FileDialog::new().pick_folder()
        {
            state.sample_path = Some(dir);
            state.sample_data = None;
            state.normalized = None;
            state.invalidate_results();
            state.status_message = "Sample directory selected".into();
        }
    });

    // --- Open beam directory ---
    ui.horizontal(|ui| {
        ui.label("Open Beam:");
        if let Some(ref path) = state.open_beam_path {
            ui.label(
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            );
        } else {
            ui.label("(none)");
        }
        if ui.button("Browse...").clicked()
            && let Some(dir) = rfd::FileDialog::new().pick_folder()
        {
            state.open_beam_path = Some(dir);
            state.open_beam_data = None;
            state.normalized = None;
            state.invalidate_results();
            state.status_message = "Open beam directory selected".into();
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
}

fn load_tiff_data(state: &mut AppState) {
    state.invalidate_results();
    if let Some(ref sample_dir) = state.sample_path {
        match nereids_io::tiff_stack::load_tiff_directory(sample_dir) {
            Ok(data) => {
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
