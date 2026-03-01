//! Guided mode: wizard-style workflow with numbered steps.

pub mod analyze;
pub mod configure;
pub mod load;
pub mod normalize;
pub mod results;
pub mod sidebar;

use crate::state::{AppState, GuidedStep};

/// Render the guided mode content area for the current step.
pub fn guided_content(ui: &mut egui::Ui, state: &mut AppState) {
    match state.guided_step {
        GuidedStep::Load => load::load_step(ui, state),
        GuidedStep::Configure => configure::configure_step(ui, state),
        GuidedStep::Normalize => normalize::normalize_step(ui, state),
        GuidedStep::Analyze => analyze::analyze_step(ui, state),
        GuidedStep::Results => results::results_step(ui, state),
        GuidedStep::ForwardModel => {
            ui.heading("Forward Model");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Forward Model tool coming in Phase 4.")
                    .italics()
                    .color(crate::theme::semantic::ORANGE),
            );
        }
        GuidedStep::Detectability => {
            ui.heading("Detectability");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Detectability tool coming in Phase 4.")
                    .italics()
                    .color(crate::theme::semantic::ORANGE),
            );
        }
    }
}
