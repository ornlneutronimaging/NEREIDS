//! Studio mode.

use crate::state::{AppState, UiMode};
use crate::widgets::design;

/// Render the Studio mode content.
///
/// The counts-KL fit engine is not yet wired into the GUI, so there is no
/// result surface here.
pub fn studio_content(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Studio",
        "Counts-fit results will appear here once the counts engine is wired in.",
    );

    ui.label(
        "Use Guided mode to load raw sample and open-beam counts, bin/rebin, \
         and configure isotopes.",
    );

    ui.add_space(12.0);
    if design::btn_primary(ui, "\u{2190} Back to Guided").clicked() {
        state.ui_mode = UiMode::Guided;
    }
}
