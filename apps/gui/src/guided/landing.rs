//! Landing page: three entry cards in a horizontal grid.

use crate::state::{AppState, GuidedStep};
use crate::theme::ThemeColors;
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

        // Center the grid by wrapping in a fixed-width layout
        let left_pad = ((avail - max_grid) / 2.0).max(0.0);
        if left_pad > 0.0 {
            ui.add_space(0.0); // no-op, centering handled by columns below
        }

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
        ui.allocate_ui(egui::vec2(max_grid, ui.available_height()), |ui| {
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
