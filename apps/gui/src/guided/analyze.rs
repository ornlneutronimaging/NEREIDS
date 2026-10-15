//! Step 4: Analyze — counts preview and ROI selection.
//!
//! The counts-KL fit engine is not yet wired into the GUI, so the fit action
//! is disabled and this step presents the image/spectrum cockpit only.

use crate::state::{AppState, SpectrumAxis};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::{
    RoiEditorResult, apply_colormap, data_range, show_image_with_roi_editor,
};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use ndarray::Axis;

/// Draw the Analyze step content.
pub fn analyze_step(ui: &mut egui::Ui, state: &mut AppState) {
    if state.energies.is_none() && state.sample_data.is_some() && state.spectrum_values.is_some() {
        crate::guided::normalize::prepare_counts(state);
    }
    ensure_selected_pixel(state);

    let available_width = ui.available_width();
    let spacing_x = ui.spacing().item_spacing.x;
    let col_height = (ui.available_height() - 40.0).max(120.0);
    let (image_width, spectrum_width) = cockpit_column_widths(available_width, spacing_x);

    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(image_width, col_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| image_panel(ui, state),
        );

        ui.separator();

        ui.allocate_ui_with_layout(
            egui::vec2(spectrum_width, col_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| spectrum_panel(ui, state),
        );
    });

    ui.add_space(8.0);
    ui.separator();
    ui.add_enabled_ui(false, |ui| {
        let _ = ui.button("Run counts fit");
    });
    ui.weak("Counts engine re-point pending \u{2014} the fit action is disabled.");

    let can_continue = state.energies.is_some();
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Results \u{2192}",
        can_continue,
        "Prepare data to continue",
    ) {
        NavAction::Back => state.nav_prev(),
        NavAction::Continue => state.nav_next(),
        NavAction::None => {}
    }
}

fn cockpit_column_widths(available_width: f32, spacing_x: f32) -> (f32, f32) {
    const IMAGE_MIN: f32 = 340.0;
    const IMAGE_MAX: f32 = 720.0;
    const CENTER_MIN: f32 = 520.0;

    let chrome = spacing_x * 4.0;
    let usable = (available_width - chrome).max(120.0);
    let minimum_total = CENTER_MIN + IMAGE_MIN;

    if usable < minimum_total {
        let image = usable * 0.42;
        let center = usable - image;
        return (image, center);
    }

    let image = (usable * 0.38).clamp(IMAGE_MIN, IMAGE_MAX);
    let center = usable - image;
    (image, center)
}

fn ensure_selected_pixel(state: &mut AppState) {
    if state.selected_pixel.is_some() {
        return;
    }
    let Some(ref sample) = state.sample_data else {
        return;
    };
    let shape = sample.shape();
    if shape.len() < 3 || shape[1] == 0 || shape[2] == 0 {
        return;
    }
    state.selected_pixel = Some((shape[1] / 2, shape[2] / 2));
}

fn image_panel(ui: &mut egui::Ui, state: &mut AppState) {
    let slice: Option<ndarray::Array2<f64>> = if let Some(ref sample) = state.sample_data {
        let n_tof = sample.shape()[0];
        if n_tof == 0 {
            None
        } else {
            if state.analyze_tof_slice_index >= n_tof {
                state.analyze_tof_slice_index = n_tof - 1;
            }
            Some(
                sample
                    .index_axis(Axis(0), state.analyze_tof_slice_index)
                    .to_owned(),
            )
        }
    } else {
        state.preview_image.clone()
    };

    let Some(slice) = slice else {
        ui.label("Load counts to see a preview.");
        return;
    };

    ui.label("Sample counts (TOF slice):");
    let sel_roi = state.selected_roi;
    let sel_px = if sel_roi.is_some() {
        None
    } else {
        state.selected_pixel
    };
    let (editor_result, image_rect) = show_image_with_roi_editor(
        ui,
        &slice,
        "analyze_preview_tex",
        crate::state::Colormap::Viridis,
        &state.rois,
        sel_roi,
        sel_px,
    );
    apply_roi_editor_result(state, editor_result);
    image_color_bar(
        ui,
        &slice,
        crate::state::Colormap::Viridis,
        image_rect.width(),
    );

    ui.add(
        egui::Slider::new(
            &mut state.analyze_tof_slice_index,
            0..=state
                .sample_data
                .as_ref()
                .map_or(0, |s| s.shape()[0].saturating_sub(1)),
        )
        .text("TOF slice"),
    );
}

fn image_color_bar(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    colormap: crate::state::Colormap,
    target_width: f32,
) {
    let width = target_width.min(ui.available_width()).max(96.0).round();
    if !width.is_finite() {
        return;
    }
    let (vmin, vmax) = data_range(data);
    let (response, painter) = ui.allocate_painter(egui::vec2(width, 10.0), egui::Sense::hover());
    let steps = 96;
    for i in 0..steps {
        let t0 = i as f32 / steps as f32;
        let t1 = (i + 1) as f32 / steps as f32;
        let (r, g, b) = apply_colormap(colormap, t0 as f64);
        let x0 = response.rect.left() + t0 * response.rect.width();
        let x1 = response.rect.left() + t1 * response.rect.width();
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x0, response.rect.top()),
                egui::pos2(x1, response.rect.bottom()),
            ),
            0.0,
            egui::Color32::from_rgb(r, g, b),
        );
    }
    painter.rect_stroke(
        response.rect,
        0.0,
        egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
        egui::StrokeKind::Inside,
    );
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format_value(vmin)).small().weak());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(format_value(vmax)).small().weak());
        });
    });
}

fn format_value(value: f64) -> String {
    if !value.is_finite() {
        "NaN".to_string()
    } else if value.abs() >= 1.0e4 || (value != 0.0 && value.abs() < 1.0e-3) {
        format!("{value:.3e}")
    } else {
        format!("{value:.4}")
    }
}

fn apply_roi_editor_result(state: &mut AppState, result: RoiEditorResult) {
    match result {
        RoiEditorResult::DrawnNew(roi) => {
            state.rois.push(roi);
            state.selected_roi = Some(state.rois.len() - 1);
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
                state.selected_roi = Some(index);
            }
        }
        RoiEditorResult::Moving { index, new_roi } => {
            if index < state.rois.len() {
                state.rois[index] = new_roi;
                state.selected_roi = Some(index);
            }
        }
        RoiEditorResult::Selected(idx) => {
            state.selected_roi = Some(idx);
        }
        RoiEditorResult::Deselected => {
            state.selected_roi = None;
        }
        RoiEditorResult::ClickedPixel(y, x) => {
            state.selected_pixel = Some((y, x));
            state.selected_roi = None;
        }
        RoiEditorResult::None => {}
    }
}

#[derive(Clone, Copy)]
enum SpectrumSelection {
    Pixel {
        y: usize,
        x: usize,
    },
    Roi {
        index: usize,
        roi: crate::state::RoiSelection,
    },
}

fn active_spectrum_selection(state: &AppState) -> Option<SpectrumSelection> {
    if let Some(index) = state.selected_roi
        && let Some(&roi) = state.rois.get(index)
        && roi.y_end > roi.y_start
        && roi.x_end > roi.x_start
    {
        return Some(SpectrumSelection::Roi { index, roi });
    }

    state
        .selected_pixel
        .map(|(y, x)| SpectrumSelection::Pixel { y, x })
}

fn spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    let selection = active_spectrum_selection(state);

    ui.horizontal(|ui| {
        ui.label("Selection:");
        match selection {
            Some(SpectrumSelection::Roi { index, roi }) => {
                ui.label(format!(
                    "ROI {}  y:{}-{} x:{}-{}",
                    index + 1,
                    roi.y_start,
                    roi.y_end.saturating_sub(1),
                    roi.x_start,
                    roi.x_end.saturating_sub(1)
                ));
            }
            Some(SpectrumSelection::Pixel { y, x }) => {
                ui.label(format!("Pixel ({y}, {x})"));
            }
            None => {
                ui.label("none");
            }
        }
    });

    let Some(sample) = state.sample_data.as_ref() else {
        ui.label("Load counts to see a spectrum.");
        return;
    };

    let spectrum: Option<Vec<f64>> = match selection {
        Some(SpectrumSelection::Pixel { y, x }) => {
            let shape = sample.shape();
            (y < shape[1] && x < shape[2])
                .then(|| nereids_io::pixels::extract_spectrum(sample, y, x).to_vec())
        }
        Some(SpectrumSelection::Roi { roi, .. }) => {
            nereids_io::pixels::average_roi(sample, roi.y_start..roi.y_end, roi.x_start..roi.x_end)
                .ok()
                .map(|a| a.to_vec())
        }
        None => sample
            .mean_axis(Axis(2))
            .and_then(|a| a.mean_axis(Axis(1)))
            .map(|a| a.to_vec()),
    };

    let Some(spectrum) = spectrum else {
        ui.label("No valid selection.");
        return;
    };

    let n_tof = sample.shape()[0];
    let Some((x_values, x_label)) = design::build_spectrum_x_axis(&design::SpectrumXAxisParams {
        axis: state.analyze_spectrum_axis,
        energies: state.energies.as_deref(),
        spectrum_values: state.spectrum_values.as_ref().map(|v| v.as_slice()),
        spectrum_unit: state.spectrum_unit,
        spectrum_kind: state.spectrum_kind,
        flight_path_m: state.beamline.flight_path_m,
        delay_us: state.beamline.delay_us,
        n_tof,
    }) else {
        return;
    };

    ui.horizontal(|ui| {
        ui.label("Axis:");
        ui.selectable_value(
            &mut state.analyze_spectrum_axis,
            SpectrumAxis::EnergyEv,
            "Energy (eV)",
        );
        ui.selectable_value(
            &mut state.analyze_spectrum_axis,
            SpectrumAxis::TofMicroseconds,
            "TOF (\u{03bc}s)",
        );
    });

    let n_plot = spectrum.len().min(x_values.len());
    if n_plot == 0 {
        return;
    }

    let points: PlotPoints = (0..n_plot).map(|i| [x_values[i], spectrum[i]]).collect();
    Plot::new("analyze_spectrum_plot")
        .x_axis_label(x_label)
        .y_axis_label("Counts")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new("Counts", points));
            if let Some(x) = x_values.get(state.analyze_tof_slice_index.min(n_plot - 1)) {
                plot_ui.vline(
                    VLine::new("TOF slice", *x)
                        .color(egui::Color32::from_rgb(255, 165, 0))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            }
        });
}
