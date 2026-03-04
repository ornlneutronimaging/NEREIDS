//! Guided mode sidebar: dynamic pipeline steps, tool links, and provenance history.

use crate::state::{AppState, GuidedStep, InputMode, PipelineEntry};
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
            let on_landing_or_wizard =
                matches!(state.guided_step, GuidedStep::Landing | GuidedStep::Wizard);

            // ── Dynamic pipeline steps or placeholder ────────────
            if state.pipeline.is_empty() {
                ui.label(
                    RichText::new("WELCOME")
                        .size(10.0)
                        .strong()
                        .color(colors.fg3),
                );
                ui.add_space(8.0);
                ui.label(
                    RichText::new("Choose a workflow from the main panel to begin.")
                        .size(11.0)
                        .color(colors.fg3),
                );
            } else if on_landing_or_wizard {
                // Dimmed previous pipeline — user navigated Home
                ui.label(
                    RichText::new("PREVIOUS")
                        .size(10.0)
                        .strong()
                        .color(colors.fg3),
                );
                ui.add_space(4.0);
                let pathway = pathway_label(state);
                ui.label(RichText::new(pathway).size(12.0).color(colors.fg3));
                ui.add_space(8.0);

                // Dimmed step list
                let pipeline: Vec<PipelineEntry> = state.pipeline.clone();
                for entry in &pipeline {
                    dimmed_pipeline_step_row(ui, entry, state, &colors);
                    ui.add_space(2.0);
                }
                ui.add_space(8.0);

                // Resume button
                if ui.button("\u{21B5} Resume").clicked() {
                    // Jump back to the first incomplete step, or the first step
                    let target = pipeline
                        .iter()
                        .find(|e| !step_is_complete(e.step, state))
                        .unwrap_or(&pipeline[0]);
                    state.guided_step = target.step;
                }
            } else {
                // Active pipeline — full rendering
                let pathway_label = pathway_label(state);
                ui.label(RichText::new(pathway_label).strong().size(14.0));
                ui.add_space(12.0);

                let pipeline: Vec<PipelineEntry> = state.pipeline.clone();
                for entry in &pipeline {
                    pipeline_step_row(ui, entry, state, &colors);
                    ui.add_space(2.0);
                }
            }

            // ── Tools section (hidden on Landing/Wizard) ─────
            if !state.pipeline.is_empty() && !on_landing_or_wizard {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);

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
            }

            // ── Status + History (pushed to bottom) ────────────
            ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                // Status message (bottom-most element)
                ui.add_space(4.0);
                ui.label(
                    RichText::new(&state.status_message)
                        .small()
                        .color(colors.fg3),
                );

                // History: last 4 provenance events
                // bottom_up reverses visual order; .rev() renders newest
                // first (bottom-most), giving chronological top-to-bottom.
                if !state.provenance_log.is_empty() {
                    let start = state.provenance_log.len().saturating_sub(4);
                    let events = &state.provenance_log[start..];
                    for event in events.iter().rev() {
                        let ts = event.formatted_timestamp();
                        let short = ts.get(11..16).unwrap_or("??:??");
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

/// Render a pipeline step row with dynamic numbering.
fn pipeline_step_row(
    ui: &mut egui::Ui,
    entry: &PipelineEntry,
    state: &mut AppState,
    colors: &ThemeColors,
) {
    let step = entry.step;
    let is_active = state.guided_step == step;
    let is_complete = step_is_complete(step, state);
    let display_num = state.step_display_number(step);

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

                // Badge text: number for required, "—" for optional
                let badge_text = match display_num {
                    Some(n) => n.to_string(),
                    None => "\u{2014}".to_string(), // em dash
                };

                if is_active {
                    ui.painter()
                        .circle_filled(center, 10.0, Color32::from_white_alpha(60));
                    ui.painter().text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        &badge_text,
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
                        &badge_text,
                        egui::FontId::proportional(11.0),
                        colors.fg3,
                    );
                }

                // Title + subtitle + optional tag
                ui.vertical(|ui| {
                    let title_color = if is_active { Color32::WHITE } else { colors.fg };
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(step.label()).color(title_color).strong());
                        if entry.optional {
                            let tag_color = if is_active {
                                Color32::from_white_alpha(120)
                            } else {
                                colors.fg3
                            };
                            ui.label(RichText::new("skip").size(9.0).color(tag_color));
                        }
                    });

                    let sub = step_subtitle(step, state);
                    if !sub.is_empty() {
                        let sub_color = if is_active {
                            Color32::from_white_alpha(180)
                        } else {
                            colors.fg3
                        };
                        ui.label(RichText::new(sub).size(10.0).color(sub_color));
                    }
                });
            });
        })
        .response;

    if resp.interact(Sense::click()).clicked() {
        state.guided_step = step;
    }
}

/// Render a dimmed pipeline step row (for inactive/previous pipeline on Landing).
fn dimmed_pipeline_step_row(
    ui: &mut egui::Ui,
    entry: &PipelineEntry,
    state: &AppState,
    colors: &ThemeColors,
) {
    let step = entry.step;
    let is_complete = step_is_complete(step, state);
    let display_num = state.step_display_number(step);

    egui::Frame::NONE
        .corner_radius(CornerRadius::same(7))
        .inner_margin(Margin::symmetric(10, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                // Dimmed number badge (16×16, smaller)
                let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), Sense::hover());
                let center = rect.center();

                let badge_text = match display_num {
                    Some(n) => n.to_string(),
                    None => "\u{2014}".to_string(),
                };

                if is_complete {
                    ui.painter()
                        .circle_filled(center, 8.0, semantic::GREEN.gamma_multiply(0.4));
                    ui.painter().text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        "\u{2713}",
                        egui::FontId::proportional(9.0),
                        colors.fg3,
                    );
                } else {
                    ui.painter()
                        .circle_filled(center, 8.0, colors.bg3.gamma_multiply(0.5));
                    ui.painter().text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        &badge_text,
                        egui::FontId::proportional(9.0),
                        colors.fg3,
                    );
                }

                ui.label(RichText::new(step.label()).size(11.0).color(colors.fg3));
            });
        });
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
                format!("{}/{}", sr.n_converged, sr.n_total)
            } else if state.is_fitting {
                "Running...".into()
            } else {
                "Not started".into()
            }
        }
        GuidedStep::Results => "Maps & export".into(),
        GuidedStep::Bin => "Event histogramming".into(),
        GuidedStep::Rebin => "Energy rebin".into(),
        GuidedStep::Roi => "Region selection".into(),
        GuidedStep::Landing | GuidedStep::Wizard => String::new(),
        GuidedStep::ForwardModel | GuidedStep::Detectability => String::new(),
    }
}

/// Check whether a guided step has been completed based on state (public accessor).
pub fn step_is_complete_pub(step: GuidedStep, state: &AppState) -> bool {
    step_is_complete(step, state)
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
        GuidedStep::Configure => {
            let has_enabled = state.isotope_entries.iter().any(|e| e.enabled);
            has_enabled
                && state
                    .isotope_entries
                    .iter()
                    .filter(|e| e.enabled)
                    .all(|e| e.resonance_data.is_some())
        }
        GuidedStep::Normalize => state.normalized.is_some(),
        GuidedStep::Analyze => state.pixel_fit_result.is_some() || state.spatial_result.is_some(),
        GuidedStep::Results => state.spatial_result.is_some(),
        GuidedStep::Bin => state.sample_data.is_some(),
        GuidedStep::Rebin => true, // optional, skip counts as done
        GuidedStep::Roi => true,   // placeholder, always passable
        GuidedStep::Landing | GuidedStep::Wizard => false,
        GuidedStep::ForwardModel | GuidedStep::Detectability => false,
    }
}

/// Build a short pathway label for the sidebar header.
fn pathway_label(state: &AppState) -> String {
    let fitting = match state.fitting_type {
        Some(crate::state::FittingType::Spatial) => "Spatial",
        Some(crate::state::FittingType::Single) => "Single",
        None => "Workflow",
    };
    let data = match state.data_type {
        Some(crate::state::DataType::Events) => "Events",
        Some(crate::state::DataType::PreNormalized) => "Pre-norm",
        Some(crate::state::DataType::Transmission) => "Transmission",
        None => "",
    };
    if data.is_empty() {
        fitting.to_string()
    } else {
        format!("{fitting} \u{00B7} {data}")
    }
}
