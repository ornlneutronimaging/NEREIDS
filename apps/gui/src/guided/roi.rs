//! ROI step: region of interest selection with interactive image preview.
//!
//! The user draws a rectangle ROI directly on the preview image via
//! click-and-drag. Full image is the default (no ROI = entire detector).

use crate::state::{AppState, Colormap, FittingType, RoiSelection};
use crate::theme::ThemeColors;
use crate::widgets::design;
use crate::widgets::image_view::show_image_with_roi_editor;

/// Render the ROI step.
pub fn roi_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Region of Interest",
        "Select the spatial region to analyze",
    );

    if let Some((height, width)) = image_dims(state) {
        let roi = state.roi.unwrap_or(RoiSelection {
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
        });

        // --- Image preview with drag-to-draw ROI ---
        // Check preview availability before entering closure to avoid borrow conflict.
        let preview_ok = state
            .preview_image
            .as_ref()
            .is_some_and(|p| p.shape()[0] == height && p.shape()[1] == width);

        let drawn_roi = if preview_ok {
            let preview = state.preview_image.as_ref().unwrap();
            let mut result = None;
            design::card(ui, |ui| {
                let tc = ThemeColors::from_ctx(ui.ctx());
                ui.label(
                    egui::RichText::new("Drag on the image to draw an ROI rectangle")
                        .size(11.0)
                        .color(tc.fg3),
                );
                ui.add_space(4.0);
                let (new_roi, _rect) = show_image_with_roi_editor(
                    ui,
                    preview,
                    "roi_preview_tex",
                    Colormap::Viridis,
                    Some(&roi),
                );
                result = new_roi;
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
            None
        };

        // Apply drawn ROI — only clear downstream fit results, NOT preview_image
        if let Some(drawn) = drawn_roi {
            state.roi = Some(drawn);
            state.pixel_fit_result = None;
            state.spatial_result = None;
            state.selected_pixel = None;
            state.log_provenance(
                crate::state::ProvenanceEventKind::ConfigChanged,
                format!(
                    "ROI drawn: y=[{}, {}] x=[{}, {}]",
                    drawn.y_start, drawn.y_end, drawn.x_start, drawn.x_end
                ),
            );
        }

        // Reset button
        ui.add_space(4.0);
        if ui.button("Reset to Full Image").clicked() {
            state.roi = None;
            state.pixel_fit_result = None;
            state.spatial_result = None;
            state.selected_pixel = None;
        }

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
        let active = state.roi.unwrap_or(RoiSelection {
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
        });
        let roi_h = active.y_end.saturating_sub(active.y_start);
        let roi_w = active.x_end.saturating_sub(active.x_start);
        let img_str = format!("{}x{}", height, width);
        let roi_str = format!("{}x{}", roi_h, roi_w);
        let px_str = format!("{}", roi_h * roi_w);
        design::stat_row(
            ui,
            &[(&img_str, "image"), (&roi_str, "ROI"), (&px_str, "pixels")],
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
