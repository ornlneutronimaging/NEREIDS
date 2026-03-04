//! ROI step: region of interest selection.
//!
//! This step appears in most pipelines. The user sets the spatial
//! region to analyze via coordinate DragValues. Full image is the
//! default (no ROI = entire detector).

use crate::state::{AppState, FittingType, RoiSelection};
use crate::theme::ThemeColors;
use crate::widgets::design;

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

            if changed || state.roi.is_none() {
                state.roi = Some(roi);
            }

            ui.add_space(4.0);
            if ui.button("Reset to Full Image").clicked() {
                state.roi = None;
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
