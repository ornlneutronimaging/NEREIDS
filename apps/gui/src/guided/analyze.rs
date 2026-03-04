//! Step 4: Analyze -- solver configuration, fit execution, spectrum/map display.
//!
//! Layout: 3-column simultaneous view (controls | image | spectrum+results).
//! The previous tab-based layout hid the map and spectrum in separate tabs;
//! this redesign shows both side-by-side so the user can click a pixel on the
//! map and immediately see its spectrum.

use crate::state::{
    AppState, GuidedStep, InputMode, IsotopeEntry, ResolutionMode, RoiSelection, SolverMethod,
    SpectrumAxis,
};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::{show_viridis_image, show_viridis_image_with_roi};
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
            teleport_pill(ui, "← Forward Model", GuidedStep::ForwardModel, state);
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
                    });
            },
        );

        ui.separator();

        // Remaining width split between image and spectrum panels
        let remaining = (available_width - controls_width - 20.0).max(200.0);
        let image_width = remaining * 0.45;
        let spectrum_width = remaining * 0.55;

        // Column 2: image viewer (map or preview, clickable)
        ui.allocate_ui_with_layout(
            egui::vec2(image_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                image_panel(ui, state);
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
        NavAction::Back => state.guided_step = GuidedStep::Normalize,
        NavAction::Continue => state.guided_step = GuidedStep::Results,
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

    // ROI
    ui.label(egui::RichText::new("Region of Interest").strong());
    if let Some(ref norm) = state.normalized {
        let shape = norm.transmission.shape();
        let height = shape[1];
        let width = shape[2];

        let mut roi = state.roi.unwrap_or(RoiSelection {
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
        });

        let changed = ui
            .horizontal(|ui| {
                let mut changed = false;
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.y_start)
                            .prefix("y0=")
                            .range(0..=height),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.y_end)
                            .prefix("y1=")
                            .range(0..=height),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.x_start)
                            .prefix("x0=")
                            .range(0..=width),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.x_end)
                            .prefix("x1=")
                            .range(0..=width),
                    )
                    .changed();
                changed
            })
            .inner;

        if changed || state.roi.is_none() {
            state.roi = Some(roi);
        }
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
        if let Some(ref counter) = state.fitting_progress_counter {
            let done = counter.load(Ordering::Relaxed);
            if let Some((_, total)) = state.fitting_progress {
                state.fitting_progress = Some((done, total));
                let frac = done as f32 / total.max(1) as f32;
                crate::widgets::design::progress_mini(ui, frac, &format!("{done}/{total} px"));
            }
        } else {
            ui.spinner();
            ui.label("Fitting...");
        }
    }
}

// ---- Image Panel (Column 2) ----

/// Shows spatial result density maps if available, otherwise the preview image.
/// Handles click-to-select-pixel via the return value of `show_viridis_image`.
fn image_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if let Some(ref result) = state.spatial_result {
        // Isotope selector (when multiple density maps)
        let n_isotopes = result.density_maps.len();
        if n_isotopes > 1 {
            ui.horizontal(|ui| {
                ui.label("Isotope:");
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
            state.roi.as_ref(),
            state.selected_pixel,
        );
        if let Some((y, x)) = clicked {
            state.selected_pixel = Some((y, x));
            state.pixel_fit_result = None;
        }

        ui.add_space(8.0);

        ui.label("Convergence map:");
        let conv_f64 = result.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
        let _ = show_viridis_image(ui, &conv_f64, "conv_tex");

        ui.label(format!(
            "{}/{} pixels converged",
            result.n_converged, result.n_total
        ));
    } else if let Some(ref norm) = state.normalized {
        // TOF-sliced transmission preview with slider
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
            let (clicked, _rect) = show_viridis_image_with_roi(
                ui,
                &slice,
                "analyze_preview_tex",
                state.roi.as_ref(),
                state.selected_pixel,
            );
            if let Some((y, x)) = clicked {
                state.selected_pixel = Some((y, x));
                state.pixel_fit_result = None;
            }

            ui.add(
                egui::Slider::new(
                    &mut state.analyze_tof_slice_index,
                    0..=n_tof.saturating_sub(1),
                )
                .text("TOF bin"),
            );
        }
    } else if let Some(ref preview) = state.preview_image {
        ui.label("Preview (summed counts):");
        if let Some((y, x)) = show_viridis_image(ui, preview, "preview_tex") {
            state.selected_pixel = Some((y, x));
            state.pixel_fit_result = None;
        }
    } else {
        ui.label("Run spatial mapping to see density maps.");
    }

    // Show selected pixel coordinates as feedback
    if let Some((y, x)) = state.selected_pixel {
        ui.add_space(4.0);
        ui.label(format!("Selected pixel: ({}, {})", y, x));
    }
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

            // Resonance dip markers (energy axis only)
            if state.analyze_spectrum_axis == SpectrumAxis::EnergyEv {
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
                                            .color(egui::Color32::from_rgb(180, 80, 80))
                                            .style(egui_plot::LineStyle::dashed_loose()),
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
            } => ResolutionParams::new(state.beamline.flight_path_m, *delta_t_us, *delta_l_m)
                .ok()
                .map(ResolutionFunction::Gaussian),
            ResolutionMode::Tabulated {
                data: Some(tab), ..
            } => Some(ResolutionFunction::Tabulated(Arc::clone(tab))),
            ResolutionMode::Tabulated { data: None, .. } => None,
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
    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.status_message = e;
            return;
        }
    };

    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => {
            state.status_message = "No pixel selected".into();
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let shape = norm.transmission.shape();
    if y >= shape[1] || x >= shape[2] {
        state.status_message = "Selected pixel is out of bounds".into();
        return;
    }

    let n_energies = shape[0];
    let t_spectrum: Vec<f64> = (0..n_energies)
        .map(|e| norm.transmission[[e, y, x]])
        .collect();
    let sigma: Vec<f64> = (0..n_energies)
        .map(|e| norm.uncertainty[[e, y, x]].max(1e-10))
        .collect();

    let result = match nereids_pipeline::pipeline::fit_spectrum(&t_spectrum, &sigma, &config) {
        Ok(r) => r,
        Err(e) => {
            state.status_message = format!("Fit error: {}", e);
            return;
        }
    };

    state.status_message = if result.converged {
        format!(
            "Pixel ({},{}) fit converged, chi2_r = {:.4}",
            y, x, result.reduced_chi_squared
        )
    } else {
        format!("Pixel ({},{}) fit did NOT converge", y, x)
    };

    state.pixel_fit_result = Some(result);
}

fn fit_roi(state: &mut AppState) {
    let config = match build_fit_config(state) {
        Ok(c) => c,
        Err(e) => {
            state.status_message = e;
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let roi = match state.roi {
        Some(r) => r,
        None => {
            state.status_message = "No ROI defined".into();
            return;
        }
    };

    if roi.y_start >= roi.y_end || roi.x_start >= roi.x_end {
        state.status_message = "Invalid ROI: start must be less than end".into();
        return;
    }

    let shape = norm.transmission.shape();
    if roi.y_end > shape[1] || roi.x_end > shape[2] {
        state.status_message = "ROI exceeds image dimensions".into();
        return;
    }

    let result = match nereids_pipeline::spatial::fit_roi(
        norm.transmission.view(),
        norm.uncertainty.view(),
        roi.y_start..roi.y_end,
        roi.x_start..roi.x_end,
        &config,
    ) {
        Ok(r) => r,
        Err(e) => {
            state.status_message = format!("ROI fit error: {}", e);
            return;
        }
    };

    state.status_message = if result.converged {
        format!(
            "ROI fit converged, chi2_r = {:.4}",
            result.reduced_chi_squared
        )
    } else {
        "ROI fit did NOT converge".into()
    };

    state.pixel_fit_result = Some(result);
}

pub fn run_spatial_map(state: &mut AppState) {
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

    let dead_pixels = state.dead_pixels.clone();

    let (tx, rx) = mpsc::channel();
    state.pending_spatial = Some(rx);
    state.is_fitting = true;
    state.status_message = "Running spatial mapping...".into();
    let cancel = Arc::clone(&state.cancel_token);

    // Progress counter: GUI polls this each frame via fitting_progress_counter.
    let progress = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    state.fitting_progress_counter = Some(Arc::clone(&progress));
    let shape = norm.transmission.shape();
    let n_total_pixels = shape[1] * shape[2];
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

fn teleport_pill(ui: &mut egui::Ui, label: &str, target: GuidedStep, state: &mut AppState) {
    let accent = crate::theme::ThemeColors::from_ctx(ui.ctx()).accent;
    let btn = egui::Button::new(
        egui::RichText::new(label)
            .small()
            .color(egui::Color32::WHITE),
    )
    .fill(accent)
    .corner_radius(12.0);
    if ui.add(btn).clicked() {
        state.guided_step = target;
    }
}
