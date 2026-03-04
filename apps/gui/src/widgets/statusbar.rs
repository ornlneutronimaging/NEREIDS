//! Bottom status bar: mode, dimensions, isotope count, beamline, version.

use crate::state::{AppState, UiMode};
use crate::theme::{ThemeColors, semantic};

/// Render the bottom status bar.
pub fn status_bar(ctx: &egui::Context, state: &AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::TopBottomPanel::bottom("status_bar")
        .frame(
            egui::Frame::NONE
                .fill(colors.bg2)
                .inner_margin(egui::Margin::symmetric(12, 4))
                .stroke(egui::Stroke::new(1.0, colors.border)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Status dot
                let dot_color = if state.is_fitting || state.is_fetching_endf {
                    semantic::ORANGE
                } else {
                    semantic::GREEN
                };
                let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 4.0, dot_color);

                // Mode + step
                let mode_text = match state.ui_mode {
                    UiMode::Guided => {
                        if let Some(n) = state.guided_step.number() {
                            format!("Guided — Step {}: {}", n, state.guided_step.label())
                        } else {
                            format!("Guided — {}", state.guided_step.label())
                        }
                    }
                    UiMode::Studio => "Studio Mode".into(),
                };
                ui.label(egui::RichText::new(mode_text).small());
                ui.separator();

                // Data dimensions
                if let Some(ref data) = state.sample_data {
                    let s = data.shape();
                    ui.label(
                        egui::RichText::new(format!("{} × {} × {}", s[1], s[2], s[0]))
                            .small()
                            .color(colors.fg2),
                    );
                    ui.separator();
                }

                // Isotope count
                let n_enabled = state.isotope_entries.iter().filter(|e| e.enabled).count();
                if n_enabled > 0 {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} isotope{}",
                            n_enabled,
                            if n_enabled == 1 { "" } else { "s" }
                        ))
                        .small()
                        .color(colors.fg2),
                    );
                    ui.separator();
                }

                // Beamline
                let fp = state.beamline.flight_path_m;
                let fp_text = if fp.fract() == 0.0 {
                    format!("VENUS {} m", fp as i64)
                } else {
                    format!("VENUS {} m", fp)
                };
                ui.label(egui::RichText::new(fp_text).small().color(colors.fg2));

                // Right-aligned version
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("NEREIDS v{}", env!("CARGO_PKG_VERSION")))
                            .small()
                            .color(colors.fg3),
                    );
                });
            });
        });
}
