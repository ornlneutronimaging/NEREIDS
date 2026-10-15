//! Step 4: Analyze — counts cockpit and the counts-domain fit.

use std::sync::Arc;
use std::sync::mpsc;

use crate::counts::{self, FitOutcome, FitRequest, RegionRequest};
use crate::state::{AppState, SpectrumAxis};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::{
    RoiEditorResult, apply_colormap, data_range, show_image_with_roi_editor,
};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use ndarray::{Array2, Axis};
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};

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
    fit_controls(ui, state);
    ui.add_space(4.0);
    fit_buttons(ui, state);

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

fn fit_ready(state: &AppState) -> bool {
    state.sample_data.is_some()
        && state.open_beam_data.is_some()
        && state.spectrum_values.is_some()
        && state.energies.is_some()
        && (state
            .isotope_entries
            .iter()
            .any(|e| e.enabled && e.resonance_data.is_some())
            || state
                .isotope_groups
                .iter()
                .any(|g| g.enabled && g.members.iter().any(|m| m.resonance_data.is_some())))
}

fn fit_controls(ui: &mut egui::Ui, state: &mut AppState) {
    egui::CollapsingHeader::new("Counts fit settings")
        .default_open(true)
        .show(ui, |ui| {
            egui::Grid::new("counts_fit_grid")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Temperature (K)");
                    ui.add(
                        egui::DragValue::new(&mut state.temperature_k)
                            .range(1.0..=5000.0)
                            .speed(1.0),
                    );
                    ui.end_row();
                    ui.label("Map patch (px)");
                    ui.add(egui::DragValue::new(&mut state.map_binning).range(1..=64usize));
                    ui.end_row();
                    ui.label("Proton-charge ratio c");
                    ui.add(egui::DragValue::new(&mut state.kl_c_ratio).speed(0.01));
                    ui.end_row();
                    ui.checkbox(&mut state.kl_background_enabled, "Fit beam background b(E)");
                    ui.end_row();
                });
            egui::CollapsingHeader::new("Pulse")
                .default_open(false)
                .show(ui, |ui| {
                    egui::Grid::new("pulse_grid").num_columns(2).show(ui, |ui| {
                        ui.label("alpha0 (1/µs/√eV)");
                        ui.add(egui::DragValue::new(&mut state.pulse.alpha0).speed(0.001));
                        ui.end_row();
                        ui.label("alpha1 (1/µs)");
                        ui.add(egui::DragValue::new(&mut state.pulse.alpha1).speed(0.001));
                        ui.end_row();
                        ui.label("beta0 (1/µs/√eV)");
                        ui.add(egui::DragValue::new(&mut state.pulse.beta0).speed(0.001));
                        ui.end_row();
                        ui.label("beta1 (1/µs)");
                        ui.add(egui::DragValue::new(&mut state.pulse.beta1).speed(0.001));
                        ui.end_row();
                        ui.label("R");
                        ui.add(
                            egui::DragValue::new(&mut state.pulse.r)
                                .range(0.0..=1.0)
                                .speed(0.001),
                        );
                        ui.end_row();
                        ui.label("FWHM (µs)");
                        ui.add(
                            egui::DragValue::new(&mut state.pulse.fwhm_us)
                                .range(0.0..=100.0)
                                .speed(0.01),
                        );
                        ui.end_row();
                    });
                });
        });
}

fn fit_buttons(ui: &mut egui::Ui, state: &mut AppState) {
    let ready = fit_ready(state);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(
                ready && !state.is_fitting,
                egui::Button::new("Run counts fit"),
            )
            .on_hover_text("Fit the selected pixel or ROI")
            .clicked()
            && let Err(message) = start_fit(state, false)
        {
            state.status_message = message;
        }
        if ui
            .add_enabled(
                ready && !state.is_fitting,
                egui::Button::new("Run counts map"),
            )
            .on_hover_text("Fit every patch of the loaded cube")
            .clicked()
            && let Err(message) = start_fit(state, true)
        {
            state.status_message = message;
        }
        if state.is_fitting {
            ui.spinner();
        }
        if !ready {
            ui.weak("Load sample and open-beam counts, a TOF bin-edge spectrum and one isotope");
        }
    });
    if let Some(summary) = &state.pixel_fit {
        ui.add_space(4.0);
        let densities = summary
            .labels
            .iter()
            .zip(&summary.densities)
            .map(|(label, density)| format!("{label} = {density:.6} at/barn"))
            .collect::<Vec<_>>()
            .join(", ");
        ui.label(format!(
            "Pixel/ROI fit: {densities}; T = {:.3} K; deviance {:.4}; converged {}",
            summary.temperature_k, summary.deviance, summary.converged
        ));
    }
    if let Some(summary) = &state.map_fit {
        ui.add_space(4.0);
        ui.label(format!(
            "Counts map: {} isotope(s), {}x{} px patches, deviance {:.4}, converged {}, numerically accepted {} of {}",
            summary.labels.len(),
            summary.binning,
            summary.binning,
            summary.deviance,
            summary.converged,
            summary.trusted.iter().filter(|&&t| t).count(),
            summary.trusted.len(),
        ));
    }
}

fn build_request(state: &AppState) -> Result<FitRequest, String> {
    let sample = state
        .sample_data
        .as_ref()
        .ok_or("load sample counts first")?;
    let open = state
        .open_beam_data
        .as_ref()
        .ok_or("load open-beam counts first")?;
    let edges = state
        .spectrum_values
        .as_ref()
        .ok_or("load a TOF bin-edge spectrum")?;
    if state.spectrum_unit != SpectrumUnit::TofMicroseconds
        || state.spectrum_kind != SpectrumValueKind::BinEdges
    {
        return Err("the counts fit needs TOF bin edges".into());
    }
    let n_bins = sample.shape()[0];
    if edges.len() != n_bins + 1 {
        return Err(format!(
            "the spectrum has {} edges for {n_bins} bins; a counts fit needs one edge per bin boundary",
            edges.len()
        ));
    }
    if open.shape() != sample.shape() {
        return Err("sample and open-beam counts have different shapes".into());
    }
    let energies = state
        .energies
        .as_ref()
        .ok_or("prepare the energy grid first")?;
    let (e_min, e_max) = energies
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &e| {
            (lo.min(e), hi.max(e))
        });
    if !(e_min.is_finite() && e_max.is_finite() && e_min < e_max) {
        return Err("the energy grid is not usable".into());
    }
    let mut isotopes = Vec::new();
    for entry in &state.isotope_entries {
        if !entry.enabled {
            continue;
        }
        let data = entry
            .resonance_data
            .clone()
            .ok_or_else(|| format!("{} has no ENDF data", entry.symbol))?;
        isotopes.push(counts::IsotopeInput {
            label: entry.symbol.clone(),
            data,
            density: entry.initial_density.max(0.0),
        });
    }
    for group in &state.isotope_groups {
        if !group.enabled {
            continue;
        }
        for member in &group.members {
            let data = member
                .resonance_data
                .clone()
                .ok_or_else(|| format!("{} has no ENDF data", member.symbol))?;
            isotopes.push(counts::IsotopeInput {
                label: format!("{} ({})", member.symbol, group.name),
                data,
                density: (group.initial_density * member.ratio).max(0.0),
            });
        }
    }
    if isotopes.is_empty() {
        return Err("enable at least one isotope with loaded ENDF data".into());
    }
    let (height, width) = (sample.shape()[1], sample.shape()[2]);
    let excluded = state
        .dead_pixels
        .clone()
        .unwrap_or_else(|| Array2::from_elem((height, width), false));
    let region = match state.selected_roi.and_then(|index| state.rois.get(index)) {
        Some(roi) if roi.y_end > roi.y_start && roi.x_end > roi.x_start => RegionRequest::Roi {
            y0: roi.y_start,
            y1: roi.y_end,
            x0: roi.x_start,
            x1: roi.x_end,
        },
        _ => {
            let (y, x) = state.selected_pixel.unwrap_or((height / 2, width / 2));
            RegionRequest::Pixel { y, x }
        }
    };
    let charge_ratio = if state.kl_c_ratio.is_finite() && state.kl_c_ratio > 0.0 {
        state.kl_c_ratio
    } else if state.proton_charge_ob > 0.0 {
        state.proton_charge_sample / state.proton_charge_ob
    } else {
        1.0
    };
    Ok(FitRequest {
        time_edges_us: edges.as_ref().clone(),
        charge_ratio,
        normalization_fitted: true,
        background_fitted: state.kl_background_enabled,
        isotopes,
        temperature_k: state.temperature_k,
        library: state.endf_library,
        t0_us: state.beamline.delay_us,
        flight_path_m: state.beamline.flight_path_m,
        pulse: state.pulse,
        energy_span_ev: (e_min, e_max),
        sample_counts: Arc::clone(sample),
        open_counts: Arc::clone(open),
        excluded: Arc::new(excluded),
        region,
        binning: state.map_binning.max(1),
    })
}

fn start_fit(state: &mut AppState, map: bool) -> Result<(), String> {
    let request = build_request(state)?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let outcome = if map {
            FitOutcome::Map(counts::run_map_fit(&request).map(Box::new))
        } else {
            FitOutcome::Pixel(counts::run_pixel_fit(&request))
        };
        let _ = tx.send(outcome);
    });
    state.pending_fit = Some(rx);
    state.is_fitting = true;
    state.status_message = if map {
        "Fitting the counts map...".into()
    } else {
        "Fitting the counts...".into()
    };
    if let Some(ctx) = &state.egui_ctx {
        ctx.request_repaint();
    }
    Ok(())
}
