//! Results step.

use crate::state::{AppState, GuidedStep};
use crate::widgets::design;

/// Draw the Results step content.
pub fn results_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(ui, "Results", "Counts-fit results");

    let mut shown = false;
    if let Some(summary) = &state.pixel_fit {
        shown = true;
        ui.heading("Single counts fit");
        egui::Grid::new("pixel_fit_results")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Density");
                ui.label(
                    summary
                        .labels
                        .iter()
                        .zip(&summary.densities)
                        .map(|(label, density)| format!("{label} = {density:.6} at/barn"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
                ui.end_row();
                ui.label("Pixels");
                ui.label(format!("{:.0}", summary.pixels));
                ui.end_row();
                ui.label("Temperature (K)");
                ui.label(format!("{:.3}", summary.temperature_k));
                ui.end_row();
                ui.label("Deviance");
                ui.label(format!("{:.4}", summary.deviance));
                ui.end_row();
                ui.label("Converged");
                ui.label(summary.converged.to_string());
                ui.end_row();
            });
    }
    if let Some(summary) = &state.map_fit {
        shown = true;
        ui.add_space(8.0);
        ui.heading("Counts map");
        egui::Grid::new("map_fit_results")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Isotopes");
                ui.label(summary.labels.join(", "));
                ui.end_row();
                ui.label("Patch");
                ui.label(format!("{}x{} px", summary.binning, summary.binning));
                ui.end_row();
                ui.label("Deviance");
                ui.label(format!("{:.4}", summary.deviance));
                ui.end_row();
                ui.label("Converged");
                ui.label(summary.converged.to_string());
                ui.end_row();
                ui.label("Identified patches").on_hover_text(
                    "Patches whose fitted densities and temperature each have a finite uncertainty. The overdispersion says whether the model describes their counts.",
                );
                ui.label(format!(
                    "{} of {}",
                    summary.identified.iter().filter(|&&t| t).count(),
                    summary.identified.len()
                ));
                ui.end_row();
            });
        for ((label, density), sd) in summary
            .labels
            .iter()
            .zip(&summary.densities)
            .zip(&summary.density_sd)
        {
            ui.label(format!(
                "{label}: mean {:.4e} +/- {:.2e} at/barn",
                mean_of(density),
                mean_of(sd)
            ));
        }
        ui.label(format!(
            "Temperature: mean {:.3} +/- {:.3} K",
            mean_of(&summary.temperature_k),
            mean_of(&summary.temperature_sd_k)
        ));
    }
    if !shown {
        ui.label(
            "No counts fit yet. Load counts and isotopes, then run a fit on the Analyze step.",
        );
    }

    ui.add_space(12.0);
    if design::btn_primary(ui, "\u{2190} Back to Analyze").clicked() {
        state.guided_step = GuidedStep::Analyze;
    }
}

fn mean_of(values: &ndarray::Array2<f64>) -> f64 {
    let (mut sum, mut count) = (0.0, 0usize);
    for &value in values.iter() {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}
