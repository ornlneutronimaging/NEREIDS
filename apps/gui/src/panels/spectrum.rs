//! Spectrum visualization panel using egui_plot.

use crate::state::AppState;
use egui_plot::{Line, Plot, PlotPoints};

/// Draw the spectrum plot in the central area.
pub fn spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Transmission Spectrum");

    // Pixel / ROI selector
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

    // Extract and plot spectrum
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
        // Reconstruct fitted transmission from densities
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
        let model = nereids_fitting::transmission_model::TransmissionFitModel {
            energies: energies.clone(),
            resonance_data,
            temperature_k: state.temperature_k,
            instrument: None,
            density_indices: (0..result.densities.len()).collect(),
            temperature_index: None,
        };

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

    // Show fit results below the plot
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
                ui.label(format!(
                    "  {}: rho = {:.6e} +/- {:.2e} atoms/barn",
                    entry.symbol, result.densities[i], result.uncertainties[i]
                ));
            }
        }
    }
}
