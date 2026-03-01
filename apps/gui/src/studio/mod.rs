//! Studio mode — power-user split-pane layout (Phase 5).

use crate::state::AppState;
use crate::theme::ThemeColors;

/// Render the Studio mode placeholder.
pub fn studio_content(ctx: &egui::Context, _state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(colors.bg))
        .show(ctx, |ui| {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Studio Mode — Coming in Phase 5")
                        .heading()
                        .color(colors.fg3),
                );
            });
        });
}
