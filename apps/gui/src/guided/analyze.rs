//! Step 4: Analyze — solver configuration, fit execution, spectrum/map display.

use crate::state::{AppState, IsotopeEntry, RoiSelection, Tab};
use crate::widgets::image_view::show_grayscale_image;
use egui_plot::{Line, Plot, PlotPoints};
use nereids_pipeline::pipeline::FitConfig;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Draw the Analyze step content.
pub fn analyze_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Analyze");
    ui.separator();

    // Horizontal layout: left controls | right spectrum/map
    let available_width = ui.available_width();
    let controls_width = 260.0_f32.min(available_width * 0.35);

    ui.horizontal(|ui| {
        // Left: fit controls
        ui.allocate_ui_with_layout(
            egui::vec2(controls_width, ui.available_height()),
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

        // Right: spectrum/map tabs
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.active_tab, Tab::Spectrum, "Spectrum");
                ui.selectable_value(&mut state.active_tab, Tab::Map, "Map");
            });
            ui.separator();

            match state.active_tab {
                Tab::Spectrum => spectrum_panel(ui, state),
                Tab::Map => map_panel(ui, state),
            }
        });
    });
}

// ---- Fit Controls ----

fn fit_controls(ui: &mut egui::Ui, state: &mut AppState) {
    ui.label(egui::RichText::new("Fit Parameters").strong());
    ui.horizontal(|ui| {
        ui.label("Max iter:");
        ui.add(egui::DragValue::new(&mut state.lm_config.max_iter).range(1..=10000));
    });

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

    ui.add_enabled_ui(ready && !state.is_fitting, |ui| {
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
        ui.spinner();
        ui.label("Fitting...");
    }
}

// ---- Spectrum Panel ----

fn spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    // Pixel selector
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

    let energies = match state.energies {
        Some(ref e) => e,
        None => {
            ui.label("Load and normalize data to see spectrum.");
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => {
            ui.label("No normalized data available.");
            return;
        }
    };

    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => {
            ui.label("Select a pixel to view its spectrum.");
            return;
        }
    };

    let n_energies = norm.transmission.shape()[0];
    let n_plot = n_energies.min(energies.len());

    // Measured spectrum
    let measured_points: PlotPoints = (0..n_plot)
        .map(|i| [energies[i], norm.transmission[[i, y, x]]])
        .collect();
    let measured_line = Line::new("Measured T(E)", measured_points);

    // Fit result (if available)
    let fit_line = state.pixel_fit_result.as_ref().and_then(|result| {
        if !result.converged {
            return None;
        }
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
        let fit_points: PlotPoints = (0..n_plot).map(|i| [energies[i], fitted_t[i]]).collect();
        Some(Line::new("Fit", fit_points).width(2.0))
    });

    // Plot
    Plot::new("spectrum_plot")
        .x_axis_label("Energy (eV)")
        .y_axis_label("Transmission")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(measured_line);
            if let Some(fit) = fit_line {
                plot_ui.line(fit);
            }
        });

    // Fit results below the plot
    if let Some(ref result) = state.pixel_fit_result {
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(if result.converged {
                "Converged"
            } else {
                "Did NOT converge"
            });
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

// ---- Map Panel ----

fn map_panel(ui: &mut egui::Ui, state: &mut AppState) {
    let result = match state.spatial_result {
        Some(ref r) => r,
        None => {
            if let Some(ref preview) = state.preview_image {
                ui.label("Preview (summed counts):");
                show_grayscale_image(ui, preview, "preview_tex");
            } else {
                ui.label("Run spatial mapping to see density maps.");
            }
            return;
        }
    };

    // Isotope selector
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
    show_grayscale_image(ui, &result.density_maps[idx], "density_tex");

    ui.add_space(8.0);

    ui.label("Convergence map:");
    let conv_f64 = result.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
    show_grayscale_image(ui, &conv_f64, "conv_tex");

    ui.label(format!(
        "{}/{} pixels converged",
        result.n_converged, result.n_total
    ));
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

    FitConfig::new(
        energies,
        resonance_data,
        isotope_names,
        state.temperature_k,
        None,
        initial_densities,
        state.lm_config.clone(),
    )
    .map_err(|e| format!("FitConfig validation error: {e}"))
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

    let n_energies = norm.transmission.shape()[0];
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
    state.active_tab = Tab::Spectrum;
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
    state.active_tab = Tab::Spectrum;
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

    std::thread::spawn(move || {
        let result = nereids_pipeline::spatial::spatial_map(
            norm.transmission.view(),
            norm.uncertainty.view(),
            &config,
            dead_pixels.as_ref(),
            Some(&cancel),
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
