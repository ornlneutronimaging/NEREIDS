//! Step 3: Normalize — transmission computation.

use crate::state::AppState;
use std::sync::Arc;

/// Draw the Normalize step content.
pub fn normalize_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Normalize");
    ui.separator();

    // Normalize button
    let can_normalize =
        state.sample_data.is_some() && state.open_beam_data.is_some() && !state.is_fitting;
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
    } else if state.sample_data.is_some() && state.open_beam_data.is_some() {
        ui.label("Click Normalize to compute transmission.");
    } else {
        ui.label("Load sample and open beam data first (Step 1).");
    }

    ui.add_space(16.0);
    ui.label(
        egui::RichText::new(
            "Full normalize preview (spectrum viewer, analysis mode selection) coming in Phase 2.",
        )
        .italics()
        .color(crate::theme::semantic::ORANGE),
    );
}

fn normalize_data(state: &mut AppState) {
    state.cancel_pending_tasks();
    state.pixel_fit_result = None;
    state.spatial_result = None;

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
            state.dead_pixels = Some(nereids_io::normalization::detect_dead_pixels(sample));

            let n_tof = sample.shape()[0];
            let tof_edges = match nereids_io::tof::linspace_tof_edges(
                state.tof_min_us,
                state.tof_max_us,
                n_tof,
            ) {
                Ok(edges) => edges,
                Err(e) => {
                    state.normalized = None;
                    state.energies = None;
                    state.status_message = format!("TOF linspace error: {}", e);
                    return;
                }
            };
            match nereids_io::tof::tof_edges_to_energy_centers(&tof_edges, &state.beamline) {
                Ok(e) => state.energies = Some(e.to_vec()),
                Err(e) => {
                    state.normalized = None;
                    state.energies = None;
                    state.status_message = format!("TOF→energy error: {}", e);
                    return;
                }
            }

            state.normalized = Some(Arc::new(norm));
            state.status_message = "Normalization complete".into();
        }
        Err(e) => {
            state.status_message = format!("Normalization error: {}", e);
        }
    }
}
