//! Bottom status bar: mode, dimensions, isotope count, beamline, version.

use crate::state::{AppState, UiMode};
use crate::theme::{ThemeColors, semantic};

/// Render the bottom status bar.
pub fn status_bar(ctx: &egui::Context, state: &AppState, rss_bytes: u64) {
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
                        if let Some(n) = state.step_display_number(state.guided_step) {
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
                let fp_text = if !fp.is_finite() {
                    "VENUS \u{2014}".to_owned()
                } else if (fp - fp.round()).abs() < 1e-6 {
                    format!("VENUS {} m", fp.round() as i64)
                } else {
                    format!("VENUS {} m", fp)
                };
                ui.label(egui::RichText::new(fp_text).small().color(colors.fg2));

                // Right-aligned: RAM + version
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("NEREIDS v{}", env!("CARGO_PKG_VERSION")))
                            .small()
                            .color(colors.fg3),
                    );

                    if rss_bytes > 0 {
                        let ram_text =
                            format!("RAM: {}", crate::telemetry::format_bytes(rss_bytes));
                        let resp =
                            ui.label(egui::RichText::new(&ram_text).small().color(colors.fg3));
                        resp.on_hover_ui(|ui| {
                            ui.label(egui::RichText::new("Process Memory").strong());
                            ui.label(format!(
                                "RSS: {}",
                                crate::telemetry::format_bytes(rss_bytes)
                            ));
                            ui.separator();
                            ui.label(egui::RichText::new("Estimated Buffers").strong());
                            let breakdown = crate::telemetry::memory_breakdown(state);
                            if breakdown.is_empty() {
                                ui.label("(no data loaded)");
                            } else {
                                let total: u64 = breakdown.iter().map(|(_, b)| b).sum();
                                egui::Grid::new("mem_breakdown").show(ui, |ui| {
                                    for (name, bytes) in &breakdown {
                                        ui.label(*name);
                                        ui.label(crate::telemetry::format_bytes(*bytes));
                                        ui.end_row();
                                    }
                                    ui.separator();
                                    ui.separator();
                                    ui.end_row();
                                    ui.label(egui::RichText::new("Total estimated").strong());
                                    ui.label(
                                        egui::RichText::new(crate::telemetry::format_bytes(total))
                                            .strong(),
                                    );
                                    ui.end_row();
                                });
                            }
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    "Excludes allocator overhead, textures, and OS caches",
                                )
                                .small()
                                .weak(),
                            );
                        });
                        ui.separator();
                    }
                });
            });
        });
}
