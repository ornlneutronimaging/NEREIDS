//! Guided mode sidebar: step navigator with completion indicators.

use crate::state::{AppState, GuidedStep, InputMode};
use crate::theme::{ThemeColors, semantic};

/// Render the guided mode sidebar with step navigation.
pub fn guided_sidebar(ctx: &egui::Context, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::SidePanel::left("guided_sidebar")
        .default_width(200.0)
        .resizable(false)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::symmetric(12, 16))
                .stroke(egui::Stroke::new(1.0, colors.border)),
        )
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("Workflow").strong().size(14.0));
            ui.add_space(12.0);

            // Main workflow steps
            for step in GuidedStep::WORKFLOW {
                let is_current = state.guided_step == step;
                let is_complete = step_is_complete(step, state);
                // SAFETY: WORKFLOW only contains numbered steps
                let number = step.number().unwrap_or(0);

                let response = ui
                    .horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;

                        // Number badge
                        let badge_color = if is_current {
                            colors.accent
                        } else if is_complete {
                            semantic::GREEN
                        } else {
                            colors.bg3
                        };
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 12.0, badge_color);
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            if is_complete && !is_current {
                                "\u{2713}".to_string() // checkmark
                            } else {
                                number.to_string()
                            },
                            egui::FontId::proportional(12.0),
                            if is_current || is_complete {
                                egui::Color32::WHITE
                            } else {
                                colors.fg
                            },
                        );

                        // Step label
                        let text = egui::RichText::new(step.label());
                        let text = if is_current {
                            text.strong().color(colors.accent)
                        } else {
                            text.color(colors.fg)
                        };
                        ui.label(text);
                    })
                    .response;

                if response.interact(egui::Sense::click()).clicked() {
                    state.guided_step = step;
                }

                ui.add_space(4.0);
            }

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            // Tools section
            ui.label(
                egui::RichText::new("Tools")
                    .strong()
                    .size(13.0)
                    .color(colors.fg2),
            );
            ui.add_space(4.0);

            if ui
                .selectable_label(
                    state.guided_step == GuidedStep::ForwardModel,
                    "Forward Model",
                )
                .clicked()
            {
                state.guided_step = GuidedStep::ForwardModel;
            }
            if ui
                .selectable_label(
                    state.guided_step == GuidedStep::Detectability,
                    "Detectability",
                )
                .clicked()
            {
                state.guided_step = GuidedStep::Detectability;
            }

            // Status message at bottom
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(&state.status_message)
                        .small()
                        .color(colors.fg3),
                );
            });
        });
}

/// Check whether a guided step has been completed based on state.
fn step_is_complete(step: GuidedStep, state: &AppState) -> bool {
    match step {
        GuidedStep::Load => match state.input_mode {
            InputMode::TiffPair => {
                state.sample_data.is_some()
                    && state.open_beam_data.is_some()
                    && state.spectrum_values.is_some()
            }
            InputMode::TransmissionTiff => {
                state.sample_data.is_some() && state.spectrum_values.is_some()
            }
        },
        GuidedStep::Configure => state
            .isotope_entries
            .iter()
            .any(|e| e.enabled && e.resonance_data.is_some()),
        GuidedStep::Normalize => state.normalized.is_some(),
        GuidedStep::Analyze => state.pixel_fit_result.is_some() || state.spatial_result.is_some(),
        GuidedStep::Results => state.spatial_result.is_some(),
        GuidedStep::ForwardModel | GuidedStep::Detectability => false,
    }
}
