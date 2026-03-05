//! Step 4: Analyze -- solver configuration, fit execution, spectrum/map display.
//!
//! Layout: 3-column simultaneous view (controls | image | spectrum+results).
//! The previous tab-based layout hid the map and spectrum in separate tabs;
//! this redesign shows both side-by-side so the user can click a pixel on the
//! map and immediately see its spectrum.

use crate::state::{
    AppState, GuidedStep, InputMode, IsotopeEntry, ResolutionMode, SolverMethod, SpectrumAxis,
};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::{
    RoiEditorResult, show_image_with_roi_editor, show_viridis_image, show_viridis_image_with_roi,
};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use ndarray::Axis;
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};
use nereids_pipeline::pipeline::FitConfig;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Draw the Analyze step content.
///
/// Three-column layout:
/// ```text
/// +-- Controls (scroll) --+-- Image (clickable) --+-- Spectrum + Results --+
/// | Fit Parameters        | viridis density map   | Pixel: (y, x)         |
/// | ROI controls          | OR preview image      | [y DragValue] [x DV]  |
/// | Run buttons           |                       |                       |
/// | Fitting spinner       | Click to select pixel | [Spectrum Plot]       |
/// | [Isotope selector     | Pixel: (y, x) shown   | Measured + Fit lines  |
/// |  for map display]     |                       |                       |
/// |                       | [Convergence map      | Fit results:          |
/// |                       |  below if available]  | chi2_r, densities     |
/// +-----------------------+-----------------------+-----------------------+
/// ```
pub fn analyze_step(ui: &mut egui::Ui, state: &mut AppState) {
    // Auto-prepare pre-normalized data if the user skipped the Normalize step.
    if matches!(
        state.input_mode,
        InputMode::TransmissionTiff | InputMode::Hdf5Histogram | InputMode::Hdf5Event
    ) && state.normalized.is_none()
        && state.sample_data.is_some()
        && state.spectrum_values.is_some()
    {
        crate::guided::normalize::prepare_transmission(state);
    }

    ui.horizontal(|ui| {
        ui.heading("Analyze");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            design::teleport_pill(ui, "← Forward Model", GuidedStep::ForwardModel, state);
        });
    });
    ui.separator();

    let available_width = ui.available_width();
    let controls_width = 220.0_f32.min(available_width * 0.2);

    ui.horizontal(|ui| {
        // Column 1: fit controls (scrollable, with min height to avoid clipping)
        ui.allocate_ui_with_layout(
            egui::vec2(controls_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("analyze_controls")
                    .show(ui, |ui| {
                        fit_controls(ui, state);
                        convergence_summary(ui, state);
                    });
            },
        );

        ui.separator();

        // Remaining width split between image and spectrum panels.
        // Account for separators + item spacing + right-edge padding.
        let remaining = (available_width - controls_width - 48.0).max(200.0);
        let image_width = remaining * 0.45;
        let spectrum_width = remaining * 0.55;

        // Column 2: image viewer (map or preview, clickable)
        ui.allocate_ui_with_layout(
            egui::vec2(image_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("analyze_images")
                    .show(ui, |ui| {
                        image_panel(ui, state);
                    });
            },
        );

        ui.separator();

        // Column 3: spectrum + results
        ui.allocate_ui_with_layout(
            egui::vec2(spectrum_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                spectrum_panel(ui, state);
            },
        );
    });

    // -- Navigation --
    let can_continue = state.spatial_result.is_some();
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Results \u{2192}",
        can_continue,
        "Run analysis to continue",
    ) {
        NavAction::Back => state.nav_prev(),
        NavAction::Continue => state.nav_next(),
        NavAction::None => {}
    }
}

// ---- Fit Controls ----

fn fit_controls(ui: &mut egui::Ui, state: &mut AppState) {
    // -- Solver configuration --
    ui.label(egui::RichText::new("Solver").strong());

    ui.horizontal(|ui| {
        ui.label("Method:");
        egui::ComboBox::from_id_salt("solver_method")
            .selected_text(match state.solver_method {
                SolverMethod::LevenbergMarquardt => "Levenberg-Marquardt",
                SolverMethod::PoissonKL => "Poisson KL",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut state.solver_method,
                    SolverMethod::LevenbergMarquardt,
                    "Levenberg-Marquardt",
                );
                ui.selectable_value(
                    &mut state.solver_method,
                    SolverMethod::PoissonKL,
                    "Poisson KL",
                );
            });
    });

    ui.horizontal(|ui| {
        ui.label("Max iter:");
        ui.add(egui::DragValue::new(&mut state.lm_config.max_iter).range(1..=10000));
    });

    // Advanced solver controls (collapsible)
    let gear = if state.show_advanced_solver {
        "\u{2699} Advanced \u{25b2}"
    } else {
        "\u{2699} Advanced \u{25bc}"
    };
    if ui
        .add(egui::Button::new(egui::RichText::new(gear).small()).frame(false))
        .clicked()
    {
        state.show_advanced_solver = !state.show_advanced_solver;
    }

    if state.show_advanced_solver {
        ui.indent("advanced_solver", |ui| {
            ui.checkbox(
                &mut state.fit_temperature,
                "Fit temperature (slow for Spatial Map)",
            );
            ui.checkbox(
                &mut state.lm_config.compute_covariance,
                "Compute covariance (single-pixel/ROI only)",
            );
            ui.horizontal(|ui| {
                ui.label("Tol (param):");
                ui.add(
                    egui::DragValue::new(&mut state.lm_config.tol_param)
                        .speed(1e-9)
                        .range(1e-12..=1.0),
                );
            });
            if state.solver_method == SolverMethod::LevenbergMarquardt {
                ui.horizontal(|ui| {
                    ui.label("Lambda init:");
                    ui.add(
                        egui::DragValue::new(&mut state.lm_config.lambda_init)
                            .speed(1e-4)
                            .range(1e-10..=1e6),
                    );
                });
            }
        });
    }

    ui.add_space(8.0);
    ui.separator();

    // Run buttons
    ui.label(egui::RichText::new("Run").strong());

    let ready = state.normalized.is_some()
        && state.energies.is_some()
        && state
            .isotope_entries
            .iter()
            .any(|e| e.enabled && e.resonance_data.is_some());

    let can_run = ready && !state.is_fitting;

    ui.add_enabled_ui(can_run, |ui| {
        if ui.button("Fit Selected Pixel").clicked() {
            fit_pixel(state);
        }
        if ui.button("Fit ROI Average").clicked() {
            fit_roi(state);
        }
        if ui.button("Spatial Map (all pixels)").clicked() {
            run_spatial_map(state);
        }
    });

    if state.is_fitting {
        if let Some((done, total)) = state.fitting_progress {
            let frac = done as f32 / total.max(1) as f32;
            crate::widgets::design::progress_mini(ui, frac, &format!("{done}/{total} px"));
        } else {
            ui.spinner();
            ui.label("Fitting...");
        }
    }

    // -- Fit feedback card --
    if let Some(ref fb) = state.last_fit_feedback {
        ui.add_space(8.0);
        let (border_color, bg_color) = if fb.success {
            (
                egui::Color32::from_rgb(60, 160, 60),
                egui::Color32::from_rgba_premultiplied(30, 80, 30, 40),
            )
        } else {
            (
                egui::Color32::from_rgb(200, 60, 60),
                egui::Color32::from_rgba_premultiplied(80, 30, 30, 40),
            )
        };
        egui::Frame::default()
            .fill(bg_color)
            .stroke(egui::Stroke::new(1.5, border_color))
            .corner_radius(4.0)
            .inner_margin(6.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(&fb.summary)
                        .color(border_color)
                        .strong(),
                );
                for (symbol, density) in &fb.densities {
                    ui.label(format!("  {symbol}: {density:.4e} at/barn"));
                }
            });
    }
}

/// Small convergence map + summary, shown in the controls column after spatial map.
fn convergence_summary(ui: &mut egui::Ui, state: &AppState) {
    let result = match state.spatial_result {
        Some(ref r) => r,
        None => return,
    };

    ui.add_space(12.0);
    ui.label(egui::RichText::new("Convergence").strong());
    let conv_f64 = result.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
    let _ = show_viridis_image(ui, &conv_f64, "conv_tex");
    ui.label(
        egui::RichText::new(format!(
            "{}/{} converged",
            result.n_converged, result.n_total
        ))
        .size(11.0),
    );
}

// ---- Image Panel (Column 2) ----

/// Shows spatial result density maps if available, otherwise the preview image
/// with interactive ROI editor. Click to select pixel, Shift+drag to draw ROIs.
fn image_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if let Some(ref result) = state.spatial_result {
        // -- Density map display (read-only ROI overlay) --
        let n_isotopes = result.density_maps.len();
        if n_isotopes > 1 {
            let current_name = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .nth(state.map_display_isotope.min(n_isotopes - 1))
                .map(|e| e.symbol.as_str())
                .unwrap_or("?");
            ui.horizontal(|ui| {
                ui.label("Isotope:");
                egui::ComboBox::from_id_salt("isotope_map_select")
                    .selected_text(current_name)
                    .show_ui(ui, |ui| {
                        for i in 0..n_isotopes {
                            let name = state
                                .isotope_entries
                                .iter()
                                .filter(|e| e.enabled && e.resonance_data.is_some())
                                .nth(i)
                                .map(|e| e.symbol.as_str())
                                .unwrap_or("?");
                            ui.selectable_value(&mut state.map_display_isotope, i, name);
                        }
                    });
            });
        }

        if n_isotopes == 0 {
            ui.label("No density maps available.");
            return;
        }

        let idx = state.map_display_isotope.min(n_isotopes - 1);

        ui.label("Density (atoms/barn):");
        let (clicked, _rect) = show_viridis_image_with_roi(
            ui,
            &result.density_maps[idx],
            "density_tex",
            &state.rois,
            state.selected_pixel,
        );
        if let Some((y, x)) = clicked {
            state.selected_pixel = Some((y, x));
            state.pixel_fit_result = None;
            state.last_fit_feedback = None;
        }
    } else if let Some(ref norm) = state.normalized {
        // -- TOF-sliced preview with interactive ROI editor --
        let n_tof = norm.transmission.shape()[0];
        if n_tof == 0 {
            ui.label("(no data)");
        } else {
            if state.analyze_tof_slice_index >= n_tof {
                state.analyze_tof_slice_index = n_tof - 1;
            }

            ui.label("Transmission (TOF slice):");
            let slice = norm
                .transmission
                .index_axis(Axis(0), state.analyze_tof_slice_index)
                .to_owned();

            // Snapshot for borrow-free editor call
            let rois_snap: Vec<_> = state.rois.clone();
            let sel_roi = state.selected_roi;
            let sel_px = state.selected_pixel;
            let (editor_result, _rect) = show_image_with_roi_editor(
                ui,
                &slice,
                "analyze_preview_tex",
                crate::state::Colormap::Viridis,
                &rois_snap,
                sel_roi,
                sel_px,
            );
            apply_roi_editor_result(state, editor_result);

            ui.add(
                egui::Slider::new(
                    &mut state.analyze_tof_slice_index,
                    0..=n_tof.saturating_sub(1),
                )
                .text("TOF bin"),
            );
        }
    } else if let Some(ref preview) = state.preview_image {
        // -- Raw preview with interactive ROI editor --
        ui.label("Preview (summed counts):");
        let preview = preview.clone();
        let rois_snap: Vec<_> = state.rois.clone();
        let sel_roi = state.selected_roi;
        let sel_px = state.selected_pixel;
        let (editor_result, _rect) = show_image_with_roi_editor(
            ui,
            &preview,
            "preview_tex",
            crate::state::Colormap::Viridis,
            &rois_snap,
            sel_roi,
            sel_px,
        );
        apply_roi_editor_result(state, editor_result);
    } else {
        ui.label("Load and normalize data to see preview.");
    }

    // ROI toolbar (only in preview mode, not when viewing density maps)
    if state.spatial_result.is_none()
        && (state.normalized.is_some() || state.preview_image.is_some())
    {
        ui.horizontal(|ui| {
            let has_sel = state.selected_roi.is_some_and(|i| i < state.rois.len());
            if ui
                .add_enabled(has_sel, egui::Button::new("Delete ROI"))
                .clicked()
                && let Some(idx) = state.selected_roi
                && idx < state.rois.len()
            {
                state.log_provenance(
                    crate::state::ProvenanceEventKind::ConfigChanged,
                    format!("ROI #{} deleted", idx + 1),
                );
                state.rois.remove(idx);
                // Adjust selection: if deleted ROI was selected, clear.
                // If a higher-indexed ROI was selected, shift down.
                state.selected_roi = match state.selected_roi {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                clear_analyze_downstream(state);
            }
            if ui
                .add_enabled(!state.rois.is_empty(), egui::Button::new("Clear ROIs"))
                .clicked()
            {
                let n = state.rois.len();
                state.log_provenance(
                    crate::state::ProvenanceEventKind::ConfigChanged,
                    format!("Cleared all {n} ROIs"),
                );
                state.rois.clear();
                state.selected_roi = None;
                clear_analyze_downstream(state);
            }
            if !state.rois.is_empty() {
                ui.label(
                    egui::RichText::new(format!("{} ROI(s)", state.rois.len()))
                        .small()
                        .weak(),
                );
            }
        });
        ui.label(
            egui::RichText::new("Click to select pixel · Shift+drag to draw ROI")
                .small()
                .weak(),
        );
    }

    // Show selected pixel coordinates as feedback
    if let Some((y, x)) = state.selected_pixel {
        ui.add_space(4.0);
        ui.label(format!("Selected pixel: ({}, {})", y, x));
    }
}

/// Apply the result of the ROI editor interaction to the state.
fn apply_roi_editor_result(state: &mut AppState, result: RoiEditorResult) {
    match result {
        RoiEditorResult::DrawnNew(roi) => {
            state.rois.push(roi);
            state.selected_roi = Some(state.rois.len() - 1);
            clear_analyze_downstream(state);
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
                clear_analyze_downstream(state);
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
        RoiEditorResult::ClickedPixel(y, x) => {
            state.selected_pixel = Some((y, x));
            state.pixel_fit_result = None;
            state.last_fit_feedback = None;
        }
        RoiEditorResult::None => {}
    }
}

/// Clear downstream fit results when ROI changes.
fn clear_analyze_downstream(state: &mut AppState) {
    state.pixel_fit_result = None;
    state.spatial_result = None;
    state.last_fit_feedback = None;
    state.fitting_rois.clear();
}

// ---- Spectrum Panel (Column 3) ----

fn spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    // Pixel selector (manual override via DragValues)
    ui.horizontal(|ui| {
        ui.label("Pixel:");
        if let Some((y, x)) = state.selected_pixel {
            ui.label(format!("({}, {})", y, x));
        } else {
            ui.label("(none selected)");
        }
        if let Some(ref norm) = state.normalized {
            let shape = norm.transmission.shape();
            let height = shape[1];
            let width = shape[2];

            let mut y_val = state.selected_pixel.map_or(0, |(y, _)| y);
            let mut x_val = state.selected_pixel.map_or(0, |(_, x)| x);

            let y_changed = ui
                .add(
                    egui::DragValue::new(&mut y_val)
                        .prefix("y: ")
                        .range(0..=height.saturating_sub(1)),
                )
                .changed();
            let x_changed = ui
                .add(
                    egui::DragValue::new(&mut x_val)
                        .prefix("x: ")
                        .range(0..=width.saturating_sub(1)),
                )
                .changed();

            if y_changed || x_changed {
                state.selected_pixel = Some((y_val, x_val));
                state.pixel_fit_result = None;
            }
        }
    });

    // Axis toggle + resonance dips
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
        ui.separator();
        ui.checkbox(&mut state.show_resonance_dips, "Resonances");
    });

    let norm = match state.normalized {
        Some(ref n) => n,
        None => {
            ui.label("No normalized data available.");
            return;
        }
    };

    let n_tof = norm.transmission.shape()[0];

    // Build x-axis values
    let (x_values, x_label): (Vec<f64>, &str) = match state.analyze_spectrum_axis {
        SpectrumAxis::EnergyEv => match state.energies {
            Some(ref e) => (e.clone(), "Energy (eV)"),
            None => {
                ui.label("Load and normalize data to see spectrum.");
                return;
            }
        },
        SpectrumAxis::TofMicroseconds => match state.spectrum_values {
            Some(ref v) => match (state.spectrum_unit, state.spectrum_kind) {
                (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinEdges) => {
                    let centers: Vec<f64> = v
                        .windows(2)
                        .take(n_tof)
                        .map(|w| 0.5 * (w[0] + w[1]))
                        .collect();
                    (centers, "TOF (\u{03bc}s)")
                }
                (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinCenters) => {
                    (v.iter().take(n_tof).copied().collect(), "TOF (\u{03bc}s)")
                }
                (SpectrumUnit::EnergyEv, SpectrumValueKind::BinEdges) => {
                    if state.beamline.flight_path_m.is_finite()
                        && state.beamline.flight_path_m > 0.0
                    {
                        // Compute bin centers from edges, then convert to TOF.
                        let tof_vals: Vec<f64> = v
                            .windows(2)
                            .take(n_tof)
                            .map(|w| {
                                let center = 0.5 * (w[0] + w[1]);
                                if center > 0.0 {
                                    nereids_core::constants::energy_to_tof(
                                        center,
                                        state.beamline.flight_path_m,
                                    ) + state.beamline.delay_us
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect();
                        (tof_vals, "TOF (\u{03bc}s)")
                    } else {
                        // Fallback: show bin centers in energy units.
                        let centers: Vec<f64> = v
                            .windows(2)
                            .take(n_tof)
                            .map(|w| 0.5 * (w[0] + w[1]))
                            .collect();
                        (centers, "Energy (eV)")
                    }
                }
                (SpectrumUnit::EnergyEv, SpectrumValueKind::BinCenters) => {
                    if state.beamline.flight_path_m.is_finite()
                        && state.beamline.flight_path_m > 0.0
                    {
                        let tof_vals: Vec<f64> = v
                            .iter()
                            .take(n_tof)
                            .map(|&e| {
                                if e > 0.0 {
                                    nereids_core::constants::energy_to_tof(
                                        e,
                                        state.beamline.flight_path_m,
                                    ) + state.beamline.delay_us
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect();
                        (tof_vals, "TOF (\u{03bc}s)")
                    } else {
                        (v.iter().take(n_tof).copied().collect(), "Energy (eV)")
                    }
                }
            },
            None => {
                let indices: Vec<f64> = (0..n_tof).map(|i| i as f64).collect();
                (indices, "Frame index")
            }
        },
    };

    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => {
            ui.label("Select a pixel to view its spectrum.");
            return;
        }
    };

    let shape = norm.transmission.shape();
    if y >= shape[1] || x >= shape[2] {
        ui.label("Selected pixel is out of bounds. Click the image to select a new pixel.");
        return;
    }

    let n_plot = n_tof.min(x_values.len());
    if n_plot == 0 {
        return;
    }

    // Measured spectrum (skip points where x is non-finite, e.g. from non-positive energies)
    let measured_points: PlotPoints = (0..n_plot)
        .filter(|&i| x_values[i].is_finite())
        .map(|i| [x_values[i], norm.transmission[[i, y, x]]])
        .collect();
    let measured_line = Line::new("Measured", measured_points);

    // Fit result (if available)
    let energies = state.energies.as_ref();
    let fit_line = state.pixel_fit_result.as_ref().and_then(|result| {
        if !result.converged {
            return None;
        }
        let energies = energies?;
        let enabled: Vec<_> = state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .collect();
        if enabled.is_empty() {
            return None;
        }

        let resonance_data: Vec<_> = enabled
            .iter()
            .filter_map(|e| e.resonance_data.clone())
            .collect();
        let model = nereids_fitting::transmission_model::TransmissionFitModel::new(
            energies.clone(),
            resonance_data,
            state.temperature_k,
            None,
            (0..result.densities.len()).collect(),
            None,
        )
        .ok()?;

        use nereids_fitting::lm::FitModel;
        let fitted_t = model.evaluate(&result.densities);
        let n_fit = n_plot.min(fitted_t.len());
        let fit_points: PlotPoints = (0..n_fit)
            .filter(|&i| x_values[i].is_finite())
            .map(|i| [x_values[i], fitted_t[i]])
            .collect();
        Some(Line::new("Fit", fit_points).width(2.0))
    });

    // TOF position marker x-value
    let tof_marker_x = x_values
        .get(state.analyze_tof_slice_index.min(n_plot.saturating_sub(1)))
        .copied();

    // Plot
    Plot::new("spectrum_plot")
        .x_axis_label(x_label)
        .y_axis_label("Transmission")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(measured_line);
            if let Some(fit) = fit_line {
                plot_ui.line(fit);
            }

            // Current TOF position marker
            if let Some(xv) = tof_marker_x {
                plot_ui.vline(
                    VLine::new("TOF position", xv)
                        .color(egui::Color32::from_rgb(255, 165, 0))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            }

            // Resonance dip markers (energy axis only, toggled)
            if state.show_resonance_dips && state.analyze_spectrum_axis == SpectrumAxis::EnergyEv {
                let (x_min, x_max) = x_values
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(v), hi.max(v))
                    });
                for entry in &state.isotope_entries {
                    if !entry.enabled {
                        continue;
                    }
                    let Some(ref res_data) = entry.resonance_data else {
                        continue;
                    };
                    for range in &res_data.ranges {
                        for lg in &range.l_groups {
                            for res in &lg.resonances {
                                if res.energy >= x_min && res.energy <= x_max {
                                    plot_ui.vline(
                                        VLine::new("", res.energy)
                                            .color(egui::Color32::from_rgba_premultiplied(
                                                180, 80, 80, 50,
                                            ))
                                            .width(0.5),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });

    // Fit results below the plot
    if let Some(ref result) = state.pixel_fit_result {
        ui.separator();
        ui.horizontal(|ui| {
            let (label, color) = if result.converged {
                ("Converged", crate::theme::semantic::GREEN)
            } else {
                ("NOT converged", crate::theme::semantic::RED)
            };
            ui.label(egui::RichText::new(label).color(color).strong());
            ui.label(format!("chi2_r = {:.4}", result.reduced_chi_squared));
            ui.label(format!("iter = {}", result.iterations));
        });

        for (i, entry) in state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .enumerate()
        {
            if i < result.densities.len() {
                let unc_str = result
                    .uncertainties
                    .as_ref()
                    .and_then(|u| u.get(i))
                    .map_or("N/A".to_string(), |u| format!("{:.2e}", u));
                ui.label(format!(
                    "  {}: rho = {:.6e} +/- {} atoms/barn",
                    entry.symbol, result.densities[i], unc_str
                ));
            }
        }
    }
}

// ---- Fit Helpers ----

fn build_fit_config(state: &AppState) -> Result<FitConfig, String> {
    let energies = state
        .energies
        .as_ref()
        .ok_or_else(|| "No energy grid loaded".to_string())?
        .clone();

    let enabled: Vec<&IsotopeEntry> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .collect();

    if enabled.is_empty() {
        return Err("No enabled isotopes with resonance data".into());
    }

    let resonance_data: Vec<_> = enabled
        .iter()
        .filter_map(|e| e.resonance_data.clone())
        .collect();
    let isotope_names: Vec<_> = enabled.iter().map(|e| e.symbol.clone()).collect();
    let initial_densities: Vec<_> = enabled.iter().map(|e| e.initial_density).collect();

    let resolution = if state.resolution_enabled {
        match &state.resolution_mode {
            ResolutionMode::Gaussian {
                delta_t_us,
                delta_l_m,
            } => {
                let params =
                    ResolutionParams::new(state.beamline.flight_path_m, *delta_t_us, *delta_l_m)
                        .map_err(|e| format!("Invalid Gaussian resolution parameters: {e}"))?;
                Some(ResolutionFunction::Gaussian(params))
            }
            ResolutionMode::Tabulated {
                data: Some(tab), ..
            } => Some(ResolutionFunction::Tabulated(Arc::clone(tab))),
            ResolutionMode::Tabulated { data: None, .. } => {
                return Err(
                    "Tabulated resolution enabled but no file loaded — load a resolution file or disable broadening".into(),
                );
            }
        }
    } else {
        None
    };

    let mut config = FitConfig::new(
        energies,
        resonance_data,
        isotope_names,
        state.temperature_k,
        resolution,
        initial_densities,
        state.lm_config.clone(),
    )
    .map_err(|e| format!("FitConfig validation error: {e}"))?;

    config = config.with_compute_covariance(state.lm_config.compute_covariance);

    if state.solver_method == SolverMethod::PoissonKL {
        config = config.with_solver(nereids_pipeline::pipeline::SolverChoice::PoissonKL(
            nereids_fitting::poisson::PoissonConfig {
                max_iter: state.lm_config.max_iter,
                ..Default::default()
            },
        ));
    }

    if state.fit_temperature {
        config = config
            .with_fit_temperature(true)
            .map_err(|e| format!("FitConfig temperature error: {e}"))?;
    }

    Ok(config)
}

fn fit_pixel(state: &mut AppState) {
    state.last_fit_feedback = None;

    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: e.clone(),
                densities: vec![],
            });
            state.status_message = e;
            return;
        }
    };

    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => {
            let msg = "No pixel selected".to_string();
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
            });
            state.status_message = msg;
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let shape = norm.transmission.shape();
    if y >= shape[1] || x >= shape[2] {
        let msg = "Selected pixel is out of bounds".to_string();
        state.last_fit_feedback = Some(crate::state::FitFeedback {
            success: false,
            summary: msg.clone(),
            densities: vec![],
        });
        state.status_message = msg;
        return;
    }

    let n_energies = shape[0];
    let t_spectrum: Vec<f64> = (0..n_energies)
        .map(|e| norm.transmission[[e, y, x]])
        .collect();
    let sigma: Vec<f64> = (0..n_energies)
        .map(|e| norm.uncertainty[[e, y, x]].max(1e-10))
        .collect();

    let enabled_symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();

    let result = match nereids_pipeline::pipeline::fit_spectrum(&t_spectrum, &sigma, &config) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("Fit error: {e}");
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
            });
            state.status_message = msg;
            return;
        }
    };

    let summary = if result.converged {
        format!(
            "Pixel ({},{}) converged, \u{03C7}\u{00B2}\u{1D63} = {:.4}",
            y, x, result.reduced_chi_squared
        )
    } else {
        format!("Pixel ({},{}) did NOT converge", y, x)
    };

    let densities: Vec<(String, f64)> = enabled_symbols
        .iter()
        .zip(result.densities.iter())
        .map(|(s, &d)| (s.clone(), d))
        .collect();

    state.last_fit_feedback = Some(crate::state::FitFeedback {
        success: result.converged,
        summary: summary.clone(),
        densities,
    });
    state.status_message = summary;
    state.pixel_fit_result = Some(result);
}

fn fit_roi(state: &mut AppState) {
    state.last_fit_feedback = None;

    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: e.clone(),
                densities: vec![],
            });
            state.status_message = e;
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    if state.rois.is_empty() {
        let msg = "No ROI defined".to_string();
        state.last_fit_feedback = Some(crate::state::FitFeedback {
            success: false,
            summary: msg.clone(),
            densities: vec![],
        });
        state.status_message = msg;
        return;
    }

    let shape = norm.transmission.shape();
    let (n_tof, height, width) = (shape[0], shape[1], shape[2]);

    // Build union mask: average transmission across all pixels inside any ROI
    let mut sum_t = vec![0.0f64; n_tof];
    let mut sum_w = vec![0.0f64; n_tof]; // inverse-variance weights
    let mut n_pixels = 0usize;

    for y in 0..height {
        for x in 0..width {
            if !state.rois.iter().any(|r| r.contains(y, x)) {
                continue;
            }
            n_pixels += 1;
            for t in 0..n_tof {
                let val = norm.transmission[[t, y, x]];
                let sig = norm.uncertainty[[t, y, x]];
                if val.is_finite() && sig.is_finite() && sig > 0.0 {
                    let w = 1.0 / (sig * sig);
                    sum_t[t] += val * w;
                    sum_w[t] += w;
                }
            }
        }
    }

    if n_pixels == 0 {
        let msg = "No valid pixels in ROI".to_string();
        state.last_fit_feedback = Some(crate::state::FitFeedback {
            success: false,
            summary: msg.clone(),
            densities: vec![],
        });
        state.status_message = msg;
        return;
    }

    // Weighted average transmission and propagated uncertainty
    let avg_t: Vec<f64> = sum_t
        .iter()
        .zip(sum_w.iter())
        .map(|(&s, &w)| if w > 0.0 { s / w } else { 1.0 })
        .collect();
    let sigma: Vec<f64> = sum_w
        .iter()
        .map(|&w| if w > 0.0 { 1.0 / w.sqrt() } else { 1.0 })
        .collect();

    let enabled_symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();

    let result = match nereids_pipeline::pipeline::fit_spectrum(&avg_t, &sigma, &config) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("ROI fit error: {e}");
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
            });
            state.status_message = msg;
            return;
        }
    };

    let summary = if result.converged {
        format!(
            "ROI fit ({n_pixels} px) converged, \u{03C7}\u{00B2}\u{1D63} = {:.4}",
            result.reduced_chi_squared
        )
    } else {
        format!("ROI fit ({n_pixels} px) did NOT converge")
    };

    let densities: Vec<(String, f64)> = enabled_symbols
        .iter()
        .zip(result.densities.iter())
        .map(|(s, d)| (s.clone(), *d))
        .collect();

    state.last_fit_feedback = Some(crate::state::FitFeedback {
        success: result.converged,
        summary: summary.clone(),
        densities,
    });
    state.status_message = summary;
    state.pixel_fit_result = Some(result);
}

pub fn run_spatial_map(state: &mut AppState) {
    state.last_fit_feedback = None;

    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.status_message = e;
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => Arc::clone(n),
        None => return,
    };

    // Combine dead_pixels with ROI mask: pixels outside all ROIs are treated as dead.
    let shape = norm.transmission.shape();
    let (height, width) = (shape[1], shape[2]);
    let dead_pixels = if !state.rois.is_empty() {
        let mut mask = ndarray::Array2::from_elem((height, width), true);
        for roi in &state.rois {
            for y in roi.y_start..roi.y_end.min(height) {
                for x in roi.x_start..roi.x_end.min(width) {
                    mask[[y, x]] = false;
                }
            }
        }
        // Merge with existing dead_pixels
        if let Some(ref dp) = state.dead_pixels {
            ndarray::Zip::from(&mut mask)
                .and(dp)
                .for_each(|m, &d| *m = *m || d);
        }
        Some(mask)
    } else {
        state.dead_pixels.clone()
    };

    // Snapshot ROIs at fit time for overlay rendering in Results
    state.fitting_rois = state.rois.clone();

    let (tx, rx) = mpsc::channel();
    state.pending_spatial = Some(rx);
    state.is_fitting = true;
    state.status_message = "Running spatial mapping...".into();
    let cancel = Arc::clone(&state.cancel_token);

    // Progress counter: GUI polls this each frame via fitting_progress_counter.
    let progress = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    state.fitting_progress_counter = Some(Arc::clone(&progress));
    let n_total_pixels = height * width;
    let n_live = match dead_pixels {
        Some(ref dp) => dp.iter().filter(|&&d| !d).count(),
        None => n_total_pixels,
    };
    state.fitting_progress = Some((0, n_live));

    std::thread::spawn(move || {
        let result = nereids_pipeline::spatial::spatial_map(
            norm.transmission.view(),
            norm.uncertainty.view(),
            &config,
            dead_pixels.as_ref(),
            Some(&cancel),
            Some(&progress),
        );
        match result {
            Ok(r) => {
                if !cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(Ok(r));
                }
            }
            Err(nereids_pipeline::error::PipelineError::Cancelled) => {}
            Err(e) => {
                let _ = tx.send(Err(format!("{e}")));
            }
        }
    });
}
