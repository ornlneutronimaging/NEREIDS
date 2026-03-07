//! Top toolbar: logo, mode toggle, studio tools, progress, home, theme.
//!
//! Prototype: `.top-toolbar { height: 48px; backdrop-filter: blur(20px); }`

use crate::state::{AppState, GuidedStep, ThemePreference, UiMode};
use crate::theme::ThemeColors;
use crate::widgets::design;

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
                    // Theme toggle (rightmost) — cycles ☀ → ☽ → A
                    let icon = match state.theme_preference {
                        ThemePreference::Light => "\u{2600}", // ☀
                        ThemePreference::Dark => "\u{263D}",  // ☽
                        ThemePreference::Auto => "A",
                    };
                    if design::btn_icon(ui, icon, false).clicked() {
                        state.theme_preference = match state.theme_preference {
                            ThemePreference::Light => ThemePreference::Dark,
                            ThemePreference::Dark => ThemePreference::Auto,
                            ThemePreference::Auto => ThemePreference::Light,
                        };
                    }

                    // Home button — returns to Landing page
                    if design::btn_primary(ui, "\u{2302} Home").clicked() {
                        state.guided_step = GuidedStep::Landing;
                        state.ui_mode = UiMode::Guided;
                    }

                    // Save button — visible when spatial results exist
                    let has_results = state.spatial_result.is_some();
                    if has_results && design::btn_primary(ui, "\u{1F4BE} Save").clicked() {
                        crate::project::save_project_dialog(state);
                    }

                    // Progress indicator
                    if state.is_fitting {
                        if let Some(ref fp) = state.fitting_progress {
                            let frac = fp.fraction();
                            let done = fp.done();
                            let total = fp.total();
                            design::progress_mini(
                                ui,
                                frac,
                                &format!("{:.0}% \u{2014} {done}/{total}", frac * 100.0),
                            );
                        } else {
                            design::progress_mini(ui, 0.0, "Fitting...");
                        }
                    } else if state.is_fetching_endf {
                        design::progress_mini(ui, 0.0, "Fetching ENDF...");
                    }
                });
            });
        });
}
