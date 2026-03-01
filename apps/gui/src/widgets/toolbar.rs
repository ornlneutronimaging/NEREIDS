//! Top toolbar: logo, mode toggle, progress, actions, theme selector.

use crate::state::{AppState, ThemePreference, UiMode};
use crate::theme::{ThemeColors, semantic};

/// Render the top toolbar.
pub fn toolbar(ctx: &egui::Context, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::TopBottomPanel::top("toolbar")
        .frame(
            egui::Frame::NONE
                .fill(colors.bg2)
                .inner_margin(egui::Margin::symmetric(12, 6))
                .stroke(egui::Stroke::new(1.0, colors.border)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 10.0;

                // Logo
                let logo = egui::Image::from_bytes(
                    "bytes://nereids-logo.svg",
                    include_bytes!("../../../../nereids-logo.svg"),
                )
                .fit_to_exact_size(egui::vec2(22.0, 22.0));
                ui.add(logo);

                // App name
                ui.label(egui::RichText::new("NEREIDS").strong().size(14.0));

                ui.add_space(20.0);

                // Mode toggle
                ui.selectable_value(&mut state.ui_mode, UiMode::Guided, "Guided");
                ui.selectable_value(&mut state.ui_mode, UiMode::Studio, "Studio");

                // Trailing controls right-aligned
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Theme selector (rightmost)
                    egui::ComboBox::from_id_salt("theme_toggle")
                        .width(60.0)
                        .selected_text(match state.theme_preference {
                            ThemePreference::Auto => "Auto",
                            ThemePreference::Light => "Light",
                            ThemePreference::Dark => "Dark",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut state.theme_preference,
                                ThemePreference::Auto,
                                "Auto",
                            );
                            ui.selectable_value(
                                &mut state.theme_preference,
                                ThemePreference::Light,
                                "Light",
                            );
                            ui.selectable_value(
                                &mut state.theme_preference,
                                ThemePreference::Dark,
                                "Dark",
                            );
                        });

                    // Progress indicator
                    if state.is_fitting {
                        ui.label(
                            egui::RichText::new("Fitting...")
                                .small()
                                .color(semantic::ORANGE),
                        );
                        ui.spinner();
                    } else if state.is_fetching_endf {
                        ui.label(
                            egui::RichText::new("Fetching ENDF...")
                                .small()
                                .color(semantic::ORANGE),
                        );
                        ui.spinner();
                    }
                });
            });
        });
}
