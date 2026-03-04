//! ROI step: region of interest selection with interactive image preview.
//!
//! The user draws rectangle ROIs directly on the preview image via
//! click-and-drag. Multiple ROIs are supported (union semantics for pipeline).
//! Full image is the default (no ROI = entire detector).

use crate::state::{AppState, Colormap, FittingType, RoiSelection};
use crate::theme::ThemeColors;
use crate::widgets::design;
use crate::widgets::image_view::{RoiEditorResult, show_image_with_roi_editor};

/// Render the ROI step.
pub fn roi_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Region of Interest",
        "Select the spatial region to analyze",
    );

    if let Some((height, width)) = image_dims(state) {
        // --- Image preview with drag-to-draw ROI ---
        // Check preview availability before entering closure to avoid borrow conflict.
        let preview_ok = state
            .preview_image
            .as_ref()
            .is_some_and(|p| p.shape()[0] == height && p.shape()[1] == width);

        // Snapshot ROI state for the widget (avoids borrow conflict with state mutation)
        let rois_snapshot: Vec<RoiSelection> = state.rois.clone();
        let selected_snapshot = state.selected_roi;

        let editor_result = if preview_ok {
            let preview = state.preview_image.as_ref().unwrap();
            let mut result = RoiEditorResult::None;
            design::card(ui, |ui| {
                let tc = ThemeColors::from_ctx(ui.ctx());
                let hint = if rois_snapshot.is_empty() {
                    "Drag on the image to draw an ROI rectangle"
                } else if selected_snapshot.is_some() {
                    "Drag selected ROI to move \u{2022} Draw outside to add \u{2022} Click empty to deselect"
                } else {
                    "Click an ROI to select \u{2022} Drag outside to add new"
                };
                ui.label(egui::RichText::new(hint).size(11.0).color(tc.fg3));
                ui.add_space(4.0);
                let (r, _rect) = show_image_with_roi_editor(
                    ui,
                    preview,
                    "roi_preview_tex",
                    Colormap::Viridis,
                    &rois_snapshot,
                    selected_snapshot,
                );
                result = r;
            });
            result
        } else {
            // Generate preview on-the-fly from sample_data if missing
            let can_generate = state.sample_data.is_some() && state.preview_image.is_none();
            if can_generate && let Some(ref data) = state.sample_data {
                state.preview_image = Some(data.sum_axis(ndarray::Axis(0)));
            }
            design::card(ui, |ui| {
                let tc = ThemeColors::from_ctx(ui.ctx());
                ui.label(
                    egui::RichText::new(
                        "No preview image available. Complete earlier steps first.",
                    )
                    .color(tc.fg2),
                );
            });
            RoiEditorResult::None
        };

        // Apply editor result — only clear downstream fit results, NOT preview_image
        match editor_result {
            RoiEditorResult::DrawnNew(roi) => {
                state.rois.push(roi);
                state.selected_roi = Some(state.rois.len() - 1);
                clear_downstream(state);
                state.log_provenance(
                    crate::state::ProvenanceEventKind::ConfigChanged,
                    format!(
                        "ROI #{} drawn: y=[{}, {}] x=[{}, {}]",
                        state.rois.len(),
                        roi.y_start,
                        roi.y_end,
                        roi.x_start,
                        roi.x_end,
                    ),
                );
            }
            RoiEditorResult::Moved { index, new_roi } => {
                if index < state.rois.len() {
                    state.rois[index] = new_roi;
                    clear_downstream(state);
                    state.log_provenance(
                        crate::state::ProvenanceEventKind::ConfigChanged,
                        format!(
                            "ROI #{} moved: y=[{}, {}] x=[{}, {}]",
                            index + 1,
                            new_roi.y_start,
                            new_roi.y_end,
                            new_roi.x_start,
                            new_roi.x_end,
                        ),
                    );
                }
            }
            RoiEditorResult::Selected(idx) => {
                state.selected_roi = Some(idx);
            }
            RoiEditorResult::Deselected => {
                state.selected_roi = None;
            }
            RoiEditorResult::None => {}
        }

        // Toolbar: Delete Selected + Reset All
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let has_selection =
                state.selected_roi.is_some() && state.selected_roi.unwrap() < state.rois.len();
            if ui
                .add_enabled(has_selection, egui::Button::new("Delete Selected"))
                .clicked()
                && let Some(idx) = state.selected_roi
                && idx < state.rois.len()
            {
                state.rois.remove(idx);
                state.selected_roi = None;
                clear_downstream(state);
            }
            if ui
                .add_enabled(!state.rois.is_empty(), egui::Button::new("Clear All ROIs"))
                .clicked()
            {
                state.rois.clear();
                state.selected_roi = None;
                clear_downstream(state);
            }
        });

        // Info about ROI meaning
        let tc = ThemeColors::from_ctx(ui.ctx());
        let roi_hint = match state.fitting_type {
            Some(FittingType::Single) => {
                "Single spectrum mode: ROI determines which pixels are accumulated."
            }
            _ => "Spatial mode: ROI crops the fitting region (each pixel fitted independently).",
        };
        ui.label(egui::RichText::new(roi_hint).size(11.0).color(tc.fg3));

        // Stat row
        let active = state.bounding_roi().unwrap_or(RoiSelection {
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
        });
        let roi_h = active.y_end.saturating_sub(active.y_start);
        let roi_w = active.x_end.saturating_sub(active.x_start);
        let img_str = format!("{}x{}", height, width);
        let roi_str = if state.rois.is_empty() {
            "full".to_string()
        } else {
            format!("{}x{}", roi_h, roi_w)
        };
        let n_rois_str = format!("{}", state.rois.len());
        design::stat_row(
            ui,
            &[
                (&img_str, "image"),
                (&roi_str, "bounding"),
                (&n_rois_str, "ROIs"),
            ],
        );
    } else {
        design::card(ui, |ui| {
            let tc = ThemeColors::from_ctx(ui.ctx());
            ui.label(
                egui::RichText::new("No image data available yet. Complete earlier steps first.")
                    .color(tc.fg2),
            );
        });
    }

    // ROI is always passable (full image is the default)
    match design::nav_buttons(ui, Some("\u{2190} Back"), "Continue \u{2192}", true, "") {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}

/// Clear downstream fit results without touching preview_image or upstream data.
fn clear_downstream(state: &mut AppState) {
    state.pixel_fit_result = None;
    state.spatial_result = None;
    state.selected_pixel = None;
}

/// Get image dimensions from whatever data source is available.
/// Checks normalized data first (ROI after Normalize), then raw sample data
/// (ROI before Normalize in single-events pipeline).
fn image_dims(state: &AppState) -> Option<(usize, usize)> {
    if let Some(ref norm) = state.normalized {
        let shape = norm.transmission.shape();
        Some((shape[1], shape[2]))
    } else if let Some(ref data) = state.sample_data {
        let shape = data.shape();
        Some((shape[1], shape[2]))
    } else {
        None
    }
}
