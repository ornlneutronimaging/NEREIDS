//! ROI step: region of interest selection with interactive image preview.
//!
//! This step appears in most pipelines. The user draws a rectangle ROI
//! directly on the preview image (drag-to-draw), or adjusts coordinates
//! via DragValues. Full image is the default (no ROI = entire detector).

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
        let mut roi = state.roi.unwrap_or(RoiSelection {
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
        });

        // --- Image preview with drag-to-draw ROI ---
        // Check preview availability and dimensions before entering closures
        // to avoid simultaneous immutable borrow of preview + mutable borrow of state.
        let preview_status = match &state.preview_image {
            Some(p) if p.shape()[0] == height && p.shape()[1] == width => 0u8, // ok
            Some(_) => 1,                                                      // dimension mismatch
            None => 2,                                                         // no preview
        };

        let drawn_roi = if preview_status == 0 {
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
        } else if preview_status == 1 {
            design::card(ui, |ui| {
                let tc = ThemeColors::from_ctx(ui.ctx());
                ui.label(
                    egui::RichText::new(
                        "Preview dimensions mismatch. Reload data to update preview.",
                    )
                    .color(tc.fg2),
                );
            });
            None
        } else {
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

        // Apply drawn ROI outside the borrow scope
        if let Some(drawn) = drawn_roi {
            roi = drawn;
            state.roi = Some(roi);
            state.pixel_fit_result = None;
            state.spatial_result = None;
            state.selected_pixel = None;
            state.invalidate_results();
            state.log_provenance(
                crate::state::ProvenanceEventKind::ConfigChanged,
                format!(
                    "ROI drawn: y=[{}, {}] x=[{}, {}]",
                    roi.y_start, roi.y_end, roi.x_start, roi.x_end
                ),
            );
        }

        // --- DragValue coordinate fields ---
        design::card_with_header(ui, "ROI Coordinates", None, |ui| {
            let changed = ui
                .horizontal(|ui| {
                    let mut changed = false;
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut roi.y_start)
                                .prefix("y\u{2080}=")
                                .range(0..=height),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut roi.y_end)
                                .prefix("y\u{2081}=")
                                .range(0..=height),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut roi.x_start)
                                .prefix("x\u{2080}=")
                                .range(0..=width),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut roi.x_end)
                                .prefix("x\u{2081}=")
                                .range(0..=width),
                        )
                        .changed();
                    changed
                })
                .inner;

            // Clamp to valid ranges: ensure start <= end
            if roi.y_start > roi.y_end {
                std::mem::swap(&mut roi.y_start, &mut roi.y_end);
            }
            if roi.x_start > roi.x_end {
                std::mem::swap(&mut roi.x_start, &mut roi.x_end);
            }

            if changed || state.roi.is_none() {
                state.roi = Some(roi);
            }

            ui.add_space(4.0);
            if ui.button("Reset to Full Image").clicked() {
                state.roi = None;
                // Only clear downstream fit results; preserve normalized/spectral data
                state.pixel_fit_result = None;
                state.spatial_result = None;
                state.selected_pixel = None;
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
        let active = state.roi.unwrap_or(RoiSelection {
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
        });
        let roi_h = active.y_end.saturating_sub(active.y_start);
        let roi_w = active.x_end.saturating_sub(active.x_start);
        let img_str = format!("{}×{}", height, width);
        let roi_str = format!("{}×{}", roi_h, roi_w);
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
