//! Guided mode sidebar: step navigator with completion indicators,
//! dynamic subtitles, tool links, and provenance history.

use crate::state::{AppState, GuidedStep, InputMode};
use crate::theme::{ThemeColors, semantic};
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText, Sense, Stroke};

/// Render the guided mode sidebar with step navigation.
pub fn guided_sidebar(ctx: &egui::Context, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::SidePanel::left("guided_sidebar")
        .default_width(220.0)
        .resizable(false)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(Margin::symmetric(12, 16))
                .stroke(Stroke::new(1.0, colors.border)),
        )
        .show(ctx, |ui| {
            ui.label(RichText::new("Workflow").strong().size(14.0));
            ui.add_space(12.0);

            // ── Workflow steps ──────────────────────────────────
            for step in GuidedStep::WORKFLOW {
                step_row(ui, step, state, &colors);
                ui.add_space(2.0);
            }

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            // ── Tools section ──────────────────────────────────
            ui.label(RichText::new("Tools").strong().size(13.0).color(colors.fg2));
            ui.add_space(4.0);

            for &(tool_step, title, subtitle, icon) in &[
                (
                    GuidedStep::ForwardModel,
                    "Forward Model",
                    "Preview spectra",
                    "\u{21D2}",
                ),
                (
                    GuidedStep::Detectability,
                    "Detectability",
                    "Pre-experiment",
                    "\u{2605}",
                ),
            ] {
                tool_row(ui, tool_step, title, subtitle, icon, state, &colors);
                ui.add_space(2.0);
            }

            // ── History section (pushed to bottom) ─────────────
            ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                // bottom_up reverses render order: first item rendered
                // appears at the very bottom. So we render events first
                // (newest at bottom), then label, then separator.
                if !state.provenance_log.is_empty() {
                    let events: Vec<_> = state
                        .provenance_log
                        .iter()
                        .rev()
                        .take(4)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    // Render bottom-up: newest event first (visually at bottom)
                    for event in events.iter().rev() {
                        let ts = event.formatted_timestamp();
                        let short = &ts[11..16]; // "HH:MM"
                        ui.label(
                            RichText::new(format!("{short} \u{2014} {}", event.message))
                                .size(10.0)
                                .color(colors.fg3),
                        );
                    }
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("HISTORY")
                            .size(10.0)
                            .strong()
                            .color(colors.fg3),
                    );
                    ui.separator();
                }
            });
        });
}

/// Render a single workflow step row with badge, title, and subtitle.
fn step_row(ui: &mut egui::Ui, step: GuidedStep, state: &mut AppState, colors: &ThemeColors) {
    let is_active = state.guided_step == step;
    let is_complete = step_is_complete(step, state);
    let number = step.number().unwrap_or(0);

    let row_fill = if is_active {
        colors.accent
    } else {
        Color32::TRANSPARENT
    };

    let resp = egui::Frame::NONE
        .fill(row_fill)
        .corner_radius(CornerRadius::same(7))
        .inner_margin(Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                // Number badge (20×20)
                let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), Sense::hover());
                let center = rect.center();

                if is_active {
                    ui.painter()
                        .circle_filled(center, 10.0, Color32::from_white_alpha(60));
                    ui.painter().text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        number.to_string(),
                        egui::FontId::proportional(11.0),
                        Color32::WHITE,
                    );
                } else if is_complete {
                    ui.painter().circle_filled(center, 10.0, semantic::GREEN);
                    ui.painter().text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        "\u{2713}",
                        egui::FontId::proportional(11.0),
                        Color32::WHITE,
                    );
                } else {
                    // Pending: bg3 fill + border
                    ui.painter().circle_filled(center, 10.0, colors.bg3);
                    ui.painter()
                        .circle_stroke(center, 10.0, Stroke::new(1.0, colors.border));
                    ui.painter().text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        number.to_string(),
                        egui::FontId::proportional(11.0),
                        colors.fg3,
                    );
                }

                // Title + subtitle
                ui.vertical(|ui| {
                    let title_color = if is_active { Color32::WHITE } else { colors.fg };
                    ui.label(RichText::new(step.label()).color(title_color).strong());

                    let sub = step_subtitle(step, state);
                    let sub_color = if is_active {
                        Color32::from_white_alpha(180)
                    } else {
                        colors.fg3
                    };
                    ui.label(RichText::new(sub).size(10.0).color(sub_color));
                });
            });
        })
        .response;

    if resp.interact(Sense::click()).clicked() {
        state.guided_step = step;
    }
}

/// Render a tool row (Forward Model, Detectability) with icon and subtitle.
fn tool_row(
    ui: &mut egui::Ui,
    step: GuidedStep,
    title: &str,
    subtitle: &str,
    icon: &str,
    state: &mut AppState,
    colors: &ThemeColors,
) {
    let is_active = state.guided_step == step;
    let row_fill = if is_active {
        colors.accent
    } else {
        Color32::TRANSPARENT
    };

    let resp = egui::Frame::NONE
        .fill(row_fill)
        .corner_radius(CornerRadius::same(7))
        .inner_margin(Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                // Icon badge (20×20)
                let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), Sense::hover());
                let badge_color = if is_active {
                    Color32::from_white_alpha(60)
                } else {
                    colors.bg3
                };
                ui.painter().circle_filled(rect.center(), 10.0, badge_color);
                let icon_color = if is_active {
                    Color32::WHITE
                } else {
                    colors.fg2
                };
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    icon,
                    egui::FontId::proportional(11.0),
                    icon_color,
                );

                // Title + subtitle
                ui.vertical(|ui| {
                    let title_color = if is_active { Color32::WHITE } else { colors.fg };
                    ui.label(RichText::new(title).color(title_color).strong());
                    let sub_color = if is_active {
                        Color32::from_white_alpha(180)
                    } else {
                        colors.fg3
                    };
                    ui.label(RichText::new(subtitle).size(10.0).color(sub_color));
                });
            });
        })
        .response;

    if resp.interact(Sense::click()).clicked() {
        state.guided_step = step;
    }
}

/// Dynamic subtitle for each workflow step based on current state.
fn step_subtitle(step: GuidedStep, state: &AppState) -> String {
    match step {
        GuidedStep::Load => {
            if let Some(ref data) = state.sample_data {
                let shape = data.shape();
                let mode = match state.input_mode {
                    InputMode::TiffPair => "TIFF pair",
                    InputMode::TransmissionTiff => "Transmission",
                    InputMode::Hdf5Histogram => "HDF5 histogram",
                    InputMode::Hdf5Event => "HDF5 event",
                };
                format!("{mode} \u{2014} {}×{}×{}", shape[0], shape[1], shape[2])
            } else {
                "No data".into()
            }
        }
        GuidedStep::Configure => {
            let n = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .count();
            if n > 0 {
                format!("{n} isotope(s)")
            } else {
                "Not configured".into()
            }
        }
        GuidedStep::Normalize => {
            if let Some(ref norm) = state.normalized {
                let bins = norm.transmission.shape()[0];
                let px = norm.transmission.shape()[1] * norm.transmission.shape()[2];
                format!("{bins} bins, {px} px")
            } else {
                "Pending".into()
            }
        }
        GuidedStep::Analyze => {
            if let Some(ref sr) = state.spatial_result {
                let total = sr.density_maps[0].len();
                let converged = sr.converged_map.iter().filter(|&&v| v).count();
                format!("{converged}/{total}")
            } else if state.is_fitting {
                "Running...".into()
            } else {
                "Not started".into()
            }
        }
        GuidedStep::Results => "Maps & export".into(),
        GuidedStep::ForwardModel | GuidedStep::Detectability => String::new(),
    }
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
            InputMode::Hdf5Histogram | InputMode::Hdf5Event => {
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
