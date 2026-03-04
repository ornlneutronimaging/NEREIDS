//! Rebin step: optional energy rebinning.
//!
//! This step appears in pre-normalized and transmission pipelines as an
//! optional step. Energy rebinning is not yet implemented — the step
//! shows current bin info and allows the user to skip.

use crate::state::AppState;
use crate::theme::ThemeColors;
use crate::widgets::design;

/// Render the Rebin step (optional, not yet implemented).
pub fn rebin_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Rebin Energy Axis",
        "Optionally coarsen the energy binning (this step can be skipped)",
    );

    design::card_with_header(ui, "Status", None, |ui| {
        let tc = ThemeColors::from_ctx(ui.ctx());
        ui.label(
            egui::RichText::new("Energy rebinning is not yet implemented.")
                .size(12.0)
                .color(tc.fg2),
        );
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("This is an optional step \u{2014} click Continue to skip.")
                .size(11.0)
                .color(tc.fg3),
        );
    });

    // Show current bin count if spectrum data is available
    if let Some(ref vals) = state.spectrum_values {
        let n = format!("{}", vals.len());
        design::stat_row(ui, &[(&n, "energy bins")]);
    }

    // Optional steps are always passable
    match design::nav_buttons(ui, Some("\u{2190} Back"), "Continue \u{2192}", true, "") {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}
