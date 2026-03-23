//! Step 4: Analyze -- solver configuration, fit execution, spectrum/map display.
//!
//! Layout: 3-column simultaneous view (controls | image | spectrum+results).
//! The previous tab-based layout hid the map and spectrum in separate tabs;
//! this redesign shows both side-by-side so the user can click a pixel on the
//! map and immediately see its spectrum.

use crate::state::{
    AppState, EndfStatus, GuidedStep, InputMode, IsotopeEntry, SolverMethod, SpectrumAxis,
};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::{
    RoiEditorResult, show_image_with_roi_editor, show_viridis_image, show_viridis_image_with_roi,
};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use ndarray::Axis;
use nereids_pipeline::pipeline::{BackgroundConfig, InputData, SolverConfig, UnifiedFitConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

    // Reserve height for nav buttons (~40px) below the 3-column region.
    let col_height = (ui.available_height() - 40.0).max(300.0);

    ui.horizontal(|ui| {
        // Column 1: fit controls (scrollable — content can exceed viewport)
        ui.allocate_ui_with_layout(
            egui::vec2(controls_width, col_height),
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

        // Column 2: image viewer (NO ScrollArea — image uses available_height
        // to fill the column vertically instead of floating in the top half).
        ui.allocate_ui_with_layout(
            egui::vec2(image_width, col_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                image_panel(ui, state);
            },
        );

        ui.separator();

        // Column 3: spectrum + results
        ui.allocate_ui_with_layout(
            egui::vec2(spectrum_width, col_height),
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
                &mut state.background_enabled,
                "Background normalization (2cm detector)",
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

    let has_enabled_iso = state
        .isotope_entries
        .iter()
        .any(|e| e.enabled && e.resonance_data.is_some());
    let has_enabled_grp = state
        .isotope_groups
        .iter()
        .any(|g| g.enabled && g.overall_status() == EndfStatus::Loaded);
    let ready = state.normalized.is_some()
        && state.energies.is_some()
        && (has_enabled_iso || has_enabled_grp);

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
        if let Some(ref fp) = state.fitting_progress {
            let done = fp.done();
            let total = fp.total();
            let frac = fp.fraction();
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
                if let Some(t) = fb.temperature_k {
                    ui.label(format!("  T = {t:.1} K"));
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
            let labels = &result.isotope_labels;
            let current_name = labels
                .get(state.map_display_isotope.min(n_isotopes - 1))
                .map(|s| s.as_str())
                .unwrap_or("?");
            ui.horizontal(|ui| {
                ui.label("Isotope:");
                egui::ComboBox::from_id_salt("isotope_map_select")
                    .selected_text(current_name)
                    .show_ui(ui, |ui| {
                        for i in 0..n_isotopes {
                            let name = labels.get(i).map(|s| s.as_str()).unwrap_or("?");
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
            state.residuals_cache = None;
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

            let sel_roi = state.selected_roi;
            let sel_px = state.selected_pixel;
            let (editor_result, _rect) = show_image_with_roi_editor(
                ui,
                &slice,
                "analyze_preview_tex",
                crate::state::Colormap::Viridis,
                &state.rois,
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
        let sel_roi = state.selected_roi;
        let sel_px = state.selected_pixel;
        let (editor_result, _rect) = show_image_with_roi_editor(
            ui,
            preview,
            "preview_tex",
            crate::state::Colormap::Viridis,
            &state.rois,
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
            state.residuals_cache = None;
            state.last_fit_feedback = None;
        }
        RoiEditorResult::None => {}
    }
}

/// Collect all resonance data (without mapping) for draw_resonance_dips.
fn collect_all_resonance_data(state: &AppState) -> Vec<nereids_endf::resonance::ResonanceData> {
    design::collect_all_resonance_data_with_mapping(state).0
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
                state.residuals_cache = None;
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
        ui.label("Load and normalize data to see spectrum.");
        return;
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
    let fit_line = state.pixel_fit_result.as_ref().and_then(|result| {
        let energies = state.energies.as_ref()?;
        let (all_rd, density_indices, density_ratios) =
            design::collect_all_resonance_data_with_mapping(state);
        design::build_fit_line(&design::FitLineParams {
            result,
            resonance_data: &all_rd,
            density_indices: &density_indices,
            density_ratios: &density_ratios,
            energies,
            temperature_k: state.temperature_k,
            x_values: &x_values,
            n_plot,
        })
    });

    // TOF position marker x-value
    let tof_marker_x = x_values
        .get(state.analyze_tof_slice_index.min(n_plot.saturating_sub(1)))
        .copied();

    // Reserve space below the plot for the fit-result summary so it
    // doesn't get clipped at the column bottom.
    let plot_height = (ui.available_height() - 100.0).max(200.0);

    // Plot
    Plot::new("spectrum_plot")
        .height(plot_height)
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
                let all_rd = collect_all_resonance_data(state);
                design::draw_resonance_dips(plot_ui, &all_rd, &x_values);
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
            if state.uncertainty_is_estimated {
                ui.label(
                    egui::RichText::new(format!(
                        "chi2_r = {:.4} (approx.)",
                        result.reduced_chi_squared
                    ))
                    .color(crate::theme::semantic::ORANGE),
                );
            } else {
                ui.label(format!("chi2_r = {:.4}", result.reduced_chi_squared));
            }
            ui.label(format!("iter = {}", result.iterations));
            if let Some(t) = result.temperature_k {
                if let Some(u) = result.temperature_k_unc {
                    ui.label(format!("T = {t:.1} \u{00b1} {u:.1} K"));
                } else {
                    ui.label(format!("T = {t:.1} K"));
                }
            }
        });

        // Display per-entity density results — build labels in the same order
        // as build_fit_config (individuals first, then groups).
        let mut fit_labels: Vec<String> = state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .map(|e| e.symbol.clone())
            .collect();
        for g in &state.isotope_groups {
            if g.enabled && g.overall_status() == EndfStatus::Loaded {
                fit_labels.push(g.name.clone());
            }
        }
        for i in 0..result.densities.len() {
            let name = fit_labels.get(i).map(|s| s.as_str()).unwrap_or("?");
            let unc_str = result
                .uncertainties
                .as_ref()
                .and_then(|u| u.get(i))
                .map_or("N/A".to_string(), |u| format!("{:.2e}", u));
            ui.label(format!(
                "  {name}: rho = {:.6e} +/- {unc_str} atoms/barn",
                result.densities[i]
            ));
        }
    }
}

// ---- Fit Helpers ----

fn build_fit_config(state: &AppState) -> Result<UnifiedFitConfig, String> {
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

    // Collect enabled groups with all members loaded
    let enabled_groups: Vec<_> = state
        .isotope_groups
        .iter()
        .filter(|g| g.enabled && g.overall_status() == EndfStatus::Loaded)
        .collect();

    if enabled.is_empty() && enabled_groups.is_empty() {
        return Err("No enabled isotopes with resonance data".into());
    }

    let resolution = design::build_resolution_function(
        state.resolution_enabled,
        &state.resolution_mode,
        state.beamline.flight_path_m,
    )
    .map_err(|e| format!("{e} \u{2014} load a resolution file or disable broadening"))?;

    // When groups are present, use with_groups() to build the config.
    // Individual isotopes are wrapped as single-member groups for uniformity.
    let mut config = if !enabled_groups.is_empty() {
        use nereids_core::types::{Isotope, IsotopeGroup};
        use nereids_endf::resonance::ResonanceData;

        let mut group_specs: Vec<(IsotopeGroup, Vec<ResonanceData>)> = Vec::new();
        let mut group_densities: Vec<f64> = Vec::new();

        // Wrap individual isotopes as single-member groups
        for iso in &enabled {
            let isotope = Isotope::new(iso.z, iso.a)
                .map_err(|e| format!("Invalid isotope {}: {e}", iso.symbol))?;
            let group = IsotopeGroup::custom(iso.symbol.clone(), vec![(isotope, 1.0)])
                .map_err(|e| format!("Group wrap error for {}: {e}", iso.symbol))?;
            group_specs.push((group, vec![iso.resonance_data.clone().unwrap()]));
            group_densities.push(iso.initial_density);
        }

        // Add actual groups
        for g in &enabled_groups {
            let members: Vec<(Isotope, f64)> = g
                .members
                .iter()
                .map(|m| Isotope::new(g.z, m.a).map(|iso| (iso, m.ratio)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Group {} isotope error: {e}", g.name))?;
            let group = IsotopeGroup::custom(g.name.clone(), members)
                .map_err(|e| format!("Group config error for {}: {e}", g.name))?;
            let rd: Vec<ResonanceData> = g
                .members
                .iter()
                .filter_map(|m| m.resonance_data.clone())
                .collect();
            group_specs.push((group, rd));
            group_densities.push(g.initial_density);
        }

        let refs: Vec<(&IsotopeGroup, &[ResonanceData])> = group_specs
            .iter()
            .map(|(g, rd)| (g, rd.as_slice()))
            .collect();

        // Guard: group_specs must have at least one group with at least one member.
        let (first_group, first_rd) = group_specs
            .first()
            .filter(|(_, rd)| !rd.is_empty())
            .ok_or_else(|| {
                "No groups with loaded resonance data — cannot build fit config".to_string()
            })?;

        // Build a base config using the first group's real data, then replace via with_groups.
        // with_groups() replaces everything, so the base values are overwritten immediately,
        // but using real data avoids sentinel values and documents the provenance.
        let base_rd = vec![first_rd[0].clone()];
        let base_names = vec![first_group.name().to_string()];
        let base_densities = vec![group_densities[0]];
        let base = UnifiedFitConfig::new(
            energies,
            base_rd,
            base_names,
            state.temperature_k,
            resolution,
            base_densities,
        )
        .map_err(|e| format!("Config validation error: {e}"))?;

        base.with_groups(&refs, group_densities)
            .map_err(|e| format!("Group config error: {e}"))?
    } else {
        // No groups — use the standard per-isotope path
        let resonance_data: Vec<_> = enabled
            .iter()
            .filter_map(|e| e.resonance_data.clone())
            .collect();
        let isotope_names: Vec<_> = enabled.iter().map(|e| e.symbol.clone()).collect();
        let initial_densities: Vec<_> = enabled.iter().map(|e| e.initial_density).collect();

        UnifiedFitConfig::new(
            energies,
            resonance_data,
            isotope_names,
            state.temperature_k,
            resolution,
            initial_densities,
        )
        .map_err(|e| format!("Config validation error: {e}"))?
    };

    config = config.with_compute_covariance(state.lm_config.compute_covariance);

    let solver = if state.solver_method == SolverMethod::PoissonKL {
        SolverConfig::PoissonKL(nereids_fitting::poisson::PoissonConfig {
            max_iter: state.lm_config.max_iter,
            ..Default::default()
        })
    } else {
        SolverConfig::LevenbergMarquardt(state.lm_config.clone())
    };
    config = config.with_solver(solver);

    if state.fit_temperature {
        config = config.with_fit_temperature(true);
    }

    if state.background_enabled {
        config = config.with_transmission_background(BackgroundConfig::default());
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
                temperature_k: None,
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
                temperature_k: None,
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
            temperature_k: None,
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

    let mut enabled_symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();
    // Append group names in the same order as build_fit_config
    // (individuals first as single-member groups, then actual groups).
    for g in &state.isotope_groups {
        if g.enabled && g.overall_status() == EndfStatus::Loaded {
            enabled_symbols.push(g.name.clone());
        }
    }

    let input = if state.input_mode == InputMode::TiffPair {
        if let (Some(sample), Some(open_beam)) = (&state.sample_data, &state.open_beam_data) {
            let sample_counts: Vec<f64> = (0..n_energies).map(|e| sample[[e, y, x]]).collect();
            let open_beam_counts: Vec<f64> =
                (0..n_energies).map(|e| open_beam[[e, y, x]]).collect();
            InputData::Counts {
                sample_counts,
                open_beam_counts,
            }
        } else {
            InputData::Transmission {
                transmission: t_spectrum,
                uncertainty: sigma,
            }
        }
    } else {
        InputData::Transmission {
            transmission: t_spectrum,
            uncertainty: sigma,
        }
    };

    let result = match nereids_pipeline::pipeline::fit_spectrum_typed(&input, &config) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("Fit error: {e}");
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
                temperature_k: None,
            });
            state.status_message = msg;
            return;
        }
    };

    let chi2_suffix = if state.uncertainty_is_estimated {
        " (approx.)"
    } else {
        ""
    };
    let summary = if result.converged {
        format!(
            "Pixel ({},{}) converged, \u{03C7}\u{00B2}\u{1D63} = {:.4}{}",
            y, x, result.reduced_chi_squared, chi2_suffix
        )
    } else {
        format!("Pixel ({},{}) did NOT converge", y, x)
    };

    let densities: Vec<(String, f64)> = enabled_symbols
        .iter()
        .zip(result.densities.iter())
        .map(|(s, &d)| (s.clone(), d))
        .collect();

    let fitted_temp = result.temperature_k;
    state.last_fit_feedback = Some(crate::state::FitFeedback {
        success: result.converged,
        summary: summary.clone(),
        densities,
        temperature_k: fitted_temp,
    });
    state.status_message = summary;
    state.pixel_fit_result = Some(result);
    state.fit_result_gen = state.fit_result_gen.wrapping_add(1);
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
                temperature_k: None,
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
            temperature_k: None,
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
    let mut pixels: Vec<(usize, usize)> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if !state.rois.iter().any(|r| r.contains(y, x)) {
                continue;
            }
            n_pixels += 1;
            pixels.push((y, x));
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
            temperature_k: None,
        });
        state.status_message = msg;
        return;
    }

    // Weighted average transmission and propagated uncertainty
    let avg_t: Vec<f64> = sum_t
        .iter()
        .zip(sum_w.iter())
        .map(|(&s, &w)| if w > 0.0 { s / w } else { 0.0 })
        .collect();
    // Bins with no valid pixels get negligible weight so the fitter ignores them.
    let sigma: Vec<f64> = sum_w
        .iter()
        .map(|&w| if w > 0.0 { 1.0 / w.sqrt() } else { 1.0e30 })
        .collect();

    let mut enabled_symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();
    // Append group names in the same order as build_fit_config
    for g in &state.isotope_groups {
        if g.enabled && g.overall_status() == EndfStatus::Loaded {
            enabled_symbols.push(g.name.clone());
        }
    }

    let roi_input = if state.input_mode == InputMode::TiffPair {
        if let (Some(sample), Some(open_beam)) = (&state.sample_data, &state.open_beam_data) {
            let sample_counts: Vec<f64> = (0..shape[0])
                .map(|t| {
                    pixels
                        .iter()
                        .map(|&(y, x)| sample[[t, y, x]].max(0.0))
                        .sum::<f64>()
                })
                .collect();
            let open_beam_counts: Vec<f64> = (0..shape[0])
                .map(|t| {
                    pixels
                        .iter()
                        .map(|&(y, x)| open_beam[[t, y, x]].max(0.0))
                        .sum::<f64>()
                })
                .collect();
            InputData::Counts {
                sample_counts,
                open_beam_counts,
            }
        } else {
            InputData::Transmission {
                transmission: avg_t,
                uncertainty: sigma,
            }
        }
    } else {
        InputData::Transmission {
            transmission: avg_t,
            uncertainty: sigma,
        }
    };

    let result = match nereids_pipeline::pipeline::fit_spectrum_typed(&roi_input, &config) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("ROI fit error: {e}");
            state.last_fit_feedback = Some(crate::state::FitFeedback {
                success: false,
                summary: msg.clone(),
                densities: vec![],
                temperature_k: None,
            });
            state.status_message = msg;
            return;
        }
    };

    let chi2_suffix = if state.uncertainty_is_estimated {
        " (approx.)"
    } else {
        ""
    };
    let summary = if result.converged {
        format!(
            "ROI fit ({n_pixels} px) converged, \u{03C7}\u{00B2}\u{1D63} = {:.4}{}",
            result.reduced_chi_squared, chi2_suffix
        )
    } else {
        format!("ROI fit ({n_pixels} px) did NOT converge")
    };

    let densities: Vec<(String, f64)> = enabled_symbols
        .iter()
        .zip(result.densities.iter())
        .map(|(s, d)| (s.clone(), *d))
        .collect();

    let fitted_temp = result.temperature_k;
    state.last_fit_feedback = Some(crate::state::FitFeedback {
        success: result.converged,
        summary: summary.clone(),
        densities,
        temperature_k: fitted_temp,
    });
    state.status_message = summary;
    state.pixel_fit_result = Some(result);
    state.fit_result_gen = state.fit_result_gen.wrapping_add(1);
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

    // Progress: single FittingProgress struct holds the Arc<AtomicUsize> counter
    // and the pixel total.  Display code reads the atomic directly each frame.
    let n_live = match dead_pixels {
        Some(ref dp) => dp.iter().filter(|&&d| !d).count(),
        None => height * width,
    };
    let (fp, progress) = crate::state::FittingProgress::new(n_live);
    state.fitting_progress = Some(fp);
    let input_mode = state.input_mode;
    let sample_data = state.sample_data.clone();
    let open_beam_data = state.open_beam_data.clone();

    // Clone the egui context so the background thread can poke the GUI
    // event loop directly via ctx.request_repaint().  This sends an
    // OS-level wake signal — far more reliable than timer-based repaints
    // when rayon saturates the CPU.
    let ctx = state.egui_ctx.clone();

    // Build a DEDICATED rayon pool for per-pixel fitting.
    //
    // spatial_map uses par_iter over pixels, and each pixel's forward-model
    // evaluation calls broadened_cross_sections / unbroadened_cross_sections
    // which ALSO use par_iter on the global rayon pool.  If both the outer
    // pixel loop and the inner physics functions share the global pool, all
    // pool threads block on inner par_iter tasks that can only run when outer
    // tasks yield — creating massive contention and effectively deadlocking
    // the pool (especially with heavy temperature fitting).
    //
    // By running the outer pixel loop on a dedicated pool, the inner physics
    // par_iter calls dispatch to the *global* pool instead.  The two pools
    // are independent, so there is no nested contention.
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
    {
        Ok(p) => p,
        Err(e) => {
            state.status_message = format!("Failed to create thread pool: {e}");
            state.is_fitting = false;
            state.fitting_progress = None;
            return;
        }
    };

    std::thread::spawn(move || {
        // Watcher thread: poke the GUI every 100ms so the progress bar
        // repaints.  Uses near-zero CPU (sleep + one syscall per wake).
        let done_flag = Arc::new(AtomicBool::new(false));
        let watcher = {
            let done = Arc::clone(&done_flag);
            let repaint_ctx = ctx.clone();
            std::thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    if let Some(ref c) = repaint_ctx {
                        c.request_repaint();
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                // One final poke so the completion frame renders immediately.
                if let Some(ref c) = repaint_ctx {
                    c.request_repaint();
                }
            })
        };

        // Run spatial_map_typed on the dedicated pool so its par_iter doesn't
        // share the global pool with inner physics par_iter calls.
        let input = if input_mode == InputMode::TiffPair {
            if let (Some(sample), Some(open_beam)) = (&sample_data, &open_beam_data) {
                nereids_pipeline::spatial::InputData3D::Counts {
                    sample_counts: sample.view(),
                    open_beam_counts: open_beam.view(),
                }
            } else {
                nereids_pipeline::spatial::InputData3D::Transmission {
                    transmission: norm.transmission.view(),
                    uncertainty: norm.uncertainty.view(),
                }
            }
        } else {
            nereids_pipeline::spatial::InputData3D::Transmission {
                transmission: norm.transmission.view(),
                uncertainty: norm.uncertainty.view(),
            }
        };
        let result = pool.install(|| {
            nereids_pipeline::spatial::spatial_map_typed(
                &input,
                &config,
                dead_pixels.as_ref(),
                Some(&cancel),
                Some(&progress),
            )
        });

        // Signal watcher to stop, then send result.
        done_flag.store(true, Ordering::Relaxed);
        watcher.join().ok();

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
