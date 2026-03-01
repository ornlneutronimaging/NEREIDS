//! Step 5: Results — output display and export (Phase 4).

use crate::state::AppState;

/// Draw the Results step content.
pub fn results_step(ui: &mut egui::Ui, _state: &AppState) {
    ui.heading("Results");
    ui.separator();
    ui.label("Export and analysis results will appear here.");
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Results display and export coming in Phase 4.")
            .italics()
            .color(crate::theme::semantic::ORANGE),
    );
}
