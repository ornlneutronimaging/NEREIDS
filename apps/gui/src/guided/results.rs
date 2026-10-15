//! Results step.

use crate::state::{AppState, GuidedStep};
use crate::widgets::design;

/// Draw the Results step content.
///
/// The legacy transmission/spatial result path was removed and the counts-KL
/// engine is not yet connected, so no fit outputs are available.
pub fn results_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Results",
        "Counts-fit results will appear here once the counts engine is wired in.",
    );

    ui.label(
        "The counts-KL fit engine is not yet connected to the GUI, so no fit \
         outputs are available. Load raw sample and open-beam counts, then \
         configure isotopes to prepare a counts fit.",
    );

    ui.add_space(12.0);
    if design::btn_primary(ui, "\u{2190} Back to Analyze").clicked() {
        state.guided_step = GuidedStep::Analyze;
    }
}
