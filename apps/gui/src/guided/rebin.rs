//! Rebin step: optional energy rebinning (stub).
//!
//! This step appears in pre-normalized and transmission pipelines
//! as an optional step that can be skipped.

use crate::state::AppState;
use crate::widgets::design;

/// Render the Rebin step (stub, optional).
pub fn rebin_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Rebin",
        "Optionally rebin the energy axis (this step can be skipped)",
    );

    design::card(ui, |ui| {
        ui.label(
            "Energy rebinning is not yet implemented. \
             Click Continue to skip this step.",
        );
    });

    // Optional steps are always passable
    match design::nav_buttons(ui, Some("\u{2190} Back"), "Continue \u{2192}", true, "") {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}
