//! Landing page: three entry cards in a horizontal grid.

use crate::state::{AppState, GuidedStep};
use crate::theme::ThemeColors;
use crate::widgets::design;
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText, Sense, Shadow, Stroke};

/// Render the landing page with three entry cards in a centered grid.
pub fn landing_step(ui: &mut egui::Ui, state: &mut AppState) {
    let tc = ThemeColors::from_ctx(ui.ctx());

    // Center the content vertically and horizontally
    ui.with_layout(Layout::top_down(Align::Center), |ui| {
        ui.add_space(40.0);

        // Title
        ui.label(RichText::new("NEREIDS").size(28.0).strong());
        ui.add_space(4.0);
        ui.label(
            RichText::new("Neutron Resonance Imaging & Elemental Density Identification System")
                .size(13.0)
                .color(tc.fg2),
        );
        ui.add_space(32.0);

        // Constrain the card grid to a max width
        let avail = ui.available_width();
        let max_grid = 900.0_f32.min(avail);

        let cards: [(GuidedStep, &str, &str, &str); 3] = [
            (
                GuidedStep::Wizard,
                "\u{1F4C8}", // 📈
                "Load & Fit Data",
                "Import neutron transmission data and fit isotope \
                 densities. Supports events, histograms, and \
                 pre-normalized formats.",
            ),
            (
                GuidedStep::ForwardModel,
                "\u{21A6}", // ↦
                "Forward Model",
                "Explore theoretical transmission spectra for any combination \
                 of isotopes and densities. No data required.",
            ),
            (
                GuidedStep::Detectability,
                "\u{2605}", // ★
                "Detectability",
                "Assess whether trace isotopes are detectable in a \
                 given matrix before running an experiment. \
                 Helps plan beamtime allocation.",
            ),
        ];

        // Use a fixed-width sub-UI so columns divide evenly
        ui.allocate_ui(egui::vec2(max_grid, 0.0), |ui| {
            ui.columns(3, |cols| {
                for (i, (target, icon, title, desc)) in cards.iter().enumerate() {
                    let clicked = landing_card(&mut cols[i], &tc, icon, title, desc);
                    if clicked {
                        if *target == GuidedStep::Wizard {
                            state.wizard_step = 0;
                        }
                        state.guided_step = *target;
                    }
                }
            });
        });

        // ── Resume bar ───────────────────────────────────
        // Show when there's an active pipeline (within-session)
        // or a cached session from a previous run.
        let has_active_pipeline = !state.pipeline.is_empty();
        let has_cached = state.cached_session.is_some();

        if has_active_pipeline || has_cached {
            ui.add_space(24.0);

            let summary = if has_active_pipeline {
                // Within-session: build summary from live state
                let fitting = match state.fitting_type {
                    Some(crate::state::FittingType::Spatial) => "Spatial",
                    Some(crate::state::FittingType::Single) => "Single",
                    None => "Unknown",
                };
                let data = match state.data_type {
                    Some(crate::state::DataType::Events) => "Events",
                    Some(crate::state::DataType::PreNormalized) => "Pre-norm",
                    Some(crate::state::DataType::Transmission) => "Transmission",
                    None => "Unknown",
                };
                let n_iso = state
                    .isotope_entries
                    .iter()
                    .filter(|e| e.enabled && e.resonance_data.is_some())
                    .count();
                if n_iso > 0 {
                    format!("{fitting} + {data}, {n_iso} isotope(s)")
                } else {
                    format!("{fitting} + {data}")
                }
            } else {
                state.cached_session.as_ref().unwrap().summary()
            };

            let label = if has_active_pipeline {
                "Resume current session"
            } else {
                "Restore previous session"
            };

            ui.allocate_ui(egui::vec2(max_grid, 0.0), |ui| {
                let frame_out = egui::Frame::NONE
                    .fill(tc.bg2)
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(16, 10))
                    .stroke(Stroke::new(1.0, tc.border))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new(label).size(12.0).strong().color(tc.fg2));
                                ui.label(RichText::new(&summary).size(11.0).color(tc.fg3));
                            });
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                design::btn_primary(ui, "\u{21B5} Resume").clicked()
                            })
                            .inner
                        })
                        .inner
                    });

                // Check btn_primary click (inner) OR frame background click
                let btn_clicked = frame_out.inner;
                let frame_clicked = frame_out.response.interact(Sense::click()).clicked();
                if btn_clicked || frame_clicked {
                    if has_active_pipeline {
                        // Resume within session: jump to first incomplete step
                        let target = state
                            .pipeline
                            .iter()
                            .find(|e| !crate::guided::sidebar::step_is_complete_pub(e.step, state))
                            .unwrap_or(&state.pipeline[0]);
                        state.guided_step = target.step;
                    } else if let Some(cache) = state.cached_session.take() {
                        // Restore from persisted cache
                        cache.apply_to(state);
                        if let Some(first) = state.pipeline.first() {
                            state.guided_step = first.step;
                        }
                    }
                }
            });
        }
    });
}

/// A single landing card: icon, title, description. Returns true if clicked.
fn landing_card(
    ui: &mut egui::Ui,
    tc: &ThemeColors,
    icon: &str,
    title: &str,
    description: &str,
) -> bool {
    let resp = egui::Frame::NONE
        .fill(tc.bg2)
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::same(20))
        .stroke(Stroke::new(1.0, tc.border))
        .shadow(Shadow {
            offset: [0, 1],
            blur: 3,
            spread: 0,
            color: Color32::from_black_alpha(12),
        })
        .show(ui, |ui| {
            ui.with_layout(Layout::top_down(Align::Center), |ui| {
                ui.add_space(8.0);
                ui.label(RichText::new(icon).size(32.0));
                ui.add_space(8.0);
                ui.label(RichText::new(title).size(15.0).strong());
                ui.add_space(6.0);
                ui.label(RichText::new(description).size(11.5).color(tc.fg2));
                ui.add_space(8.0);
            });
        })
        .response;

    resp.interact(Sense::click()).clicked()
}
