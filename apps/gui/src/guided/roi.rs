//! ROI step: region of interest selection (stub).
//!
//! This step appears in most pipelines (except single-transmission).
//! The actual ROI selection UI currently lives in analyze.rs and
//! will be extracted here in a future batch.

use crate::state::AppState;
use crate::widgets::design;

/// Render the ROI step (stub).
pub fn roi_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Region of Interest",
        "Select the spatial region to analyze",
    );

    design::card(ui, |ui| {
        ui.label(
            "ROI selection will be configured here. \
             For now, ROI can be set in the Analyze step.",
        );
    });

    // ROI is always passable (full image is the default)
    match design::nav_buttons(ui, Some("\u{2190} Back"), "Continue \u{2192}", true, "") {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}
