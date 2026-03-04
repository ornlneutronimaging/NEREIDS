//! Bin step: configure event histogramming parameters (stub).
//!
//! This step is shown for event-based pipelines. The actual bin
//! parameters (n_bins, tof_min, tof_max, etc.) currently live in
//! load.rs HDF5Event tab and will be extracted here in a future batch.

use crate::state::AppState;
use crate::widgets::design;

/// Render the Bin step (stub).
pub fn bin_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Bin Events",
        "Set event histogramming parameters before normalization",
    );

    design::card(ui, |ui| {
        ui.label(
            "Event histogramming parameters will be configured here. \
             For now, bin settings are configured in the Load step.",
        );
    });

    let can_continue = state.sample_data.is_some();
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        can_continue,
        "Load event data first",
    ) {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}
