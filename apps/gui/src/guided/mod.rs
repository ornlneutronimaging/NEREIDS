//! Guided mode: wizard-style workflow with numbered steps.

pub mod analyze;
pub mod bin;
pub mod configure;
pub mod detectability;
pub mod forward_model;
pub mod landing;
pub mod load;
pub mod normalize;
pub mod rebin;
pub mod result_widgets;
pub mod results;
pub mod roi;
pub mod sidebar;
pub mod wizard;

use crate::state::{AppState, GuidedStep};

/// Render the guided mode content area for the current step.
pub fn guided_content(ui: &mut egui::Ui, state: &mut AppState) {
    match state.guided_step {
        GuidedStep::Landing => landing::landing_step(ui, state),
        GuidedStep::Wizard => wizard::wizard_step(ui, state),
        GuidedStep::Configure => configure::configure_step(ui, state),
        GuidedStep::Load => load::load_step(ui, state),
        GuidedStep::Bin => bin::bin_step(ui, state),
        GuidedStep::Rebin => rebin::rebin_step(ui, state),
        GuidedStep::Normalize => normalize::normalize_step(ui, state),
        GuidedStep::Roi => roi::roi_step(ui, state),
        GuidedStep::Analyze => analyze::analyze_step(ui, state),
        GuidedStep::Results => results::results_step(ui, state),
        GuidedStep::ForwardModel => forward_model::forward_model_step(ui, state),
        GuidedStep::Detectability => detectability::detectability_step(ui, state),
    }
}
