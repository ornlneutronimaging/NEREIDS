//! Step 5: Results — spatial map results display and pixel inspector.

use crate::state::AppState;
use crate::widgets::image_view::show_viridis_image;

/// Draw the Results step content.
pub fn results_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Results");
    ui.separator();

    let result = match state.spatial_result {
        Some(ref r) => r,
        None => {
            ui.label("Run spatial mapping (Analyze step) to see results here.");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Export functionality coming in Phase 4.")
                    .italics()
                    .color(crate::theme::semantic::ORANGE),
            );
            return;
        }
    };

    // -- Summary Statistics Card --
    summary_card(ui, result, &state.isotope_entries);
    ui.add_space(12.0);

    // -- Density Map Grid --
    density_map_grid(ui, state);
    ui.add_space(12.0);

    // -- Pixel Inspector --
    pixel_inspector(ui, state);
    ui.add_space(12.0);

    ui.label(
        egui::RichText::new("Export functionality coming in Phase 4.")
            .italics()
            .color(crate::theme::semantic::ORANGE),
    );
}

/// Summary statistics card showing convergence and density stats.
fn summary_card(
    ui: &mut egui::Ui,
    result: &nereids_pipeline::spatial::SpatialResult,
    isotope_entries: &[crate::state::IsotopeEntry],
) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Summary Statistics").strong());
            ui.add_space(4.0);

            let pct = if result.n_total > 0 {
                100.0 * result.n_converged as f64 / result.n_total as f64
            } else {
                0.0
            };
            ui.label(format!(
                "Converged: {} / {} ({:.1}%)",
                result.n_converged, result.n_total, pct
            ));

            // Mean chi2_r
            let chi2_vals: Vec<f64> = result
                .chi_squared_map
                .iter()
                .zip(result.converged_map.iter())
                .filter(|&(_, &conv)| conv)
                .map(|(&c, _)| c)
                .filter(|&c| c.is_finite())
                .collect();
            if !chi2_vals.is_empty() {
                let mean_chi2: f64 = chi2_vals.iter().sum::<f64>() / chi2_vals.len() as f64;
                ui.label(format!("Mean chi2_r (converged): {:.4}", mean_chi2));
            }

            // Per-isotope mean density
            let enabled: Vec<_> = isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .collect();
            for (i, entry) in enabled.iter().enumerate() {
                if i < result.density_maps.len() {
                    let map = &result.density_maps[i];
                    let conv_vals: Vec<f64> = map
                        .iter()
                        .zip(result.converged_map.iter())
                        .filter(|&(_, &conv)| conv)
                        .map(|(&d, _)| d)
                        .filter(|&d| d.is_finite())
                        .collect();
                    if !conv_vals.is_empty() {
                        let mean: f64 = conv_vals.iter().sum::<f64>() / conv_vals.len() as f64;
                        ui.label(format!(
                            "{}: mean density = {:.4e} atoms/barn",
                            entry.symbol, mean
                        ));
                    }
                }
            }
        });
}

/// Grid of density map tiles (one per isotope + convergence map).
fn density_map_grid(ui: &mut egui::Ui, state: &mut AppState) {
    let result = match state.spatial_result {
        Some(ref r) => r,
        None => return,
    };

    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Density Maps").strong());
            ui.add_space(4.0);

            let enabled: Vec<_> = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .collect();

            ui.horizontal_wrapped(|ui| {
                for (i, entry) in enabled.iter().enumerate() {
                    if i < result.density_maps.len() {
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new(format!("{} density", entry.symbol)).small(),
                            );
                            if let Some((y, x)) = show_viridis_image(
                                ui,
                                &result.density_maps[i],
                                &format!("result_density_{}", i),
                            ) {
                                state.selected_pixel = Some((y, x));
                            }
                        });
                    }
                }

                // Convergence map
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Convergence").small());
                    let conv_f64 = result.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
                    let _ = show_viridis_image(ui, &conv_f64, "result_conv_map");
                    ui.label(
                        egui::RichText::new(format!("{}/{}", result.n_converged, result.n_total))
                            .small(),
                    );
                });
            });
        });
}

/// Pixel inspector: shows per-isotope density at the selected pixel.
fn pixel_inspector(ui: &mut egui::Ui, state: &AppState) {
    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => return,
    };

    let result = match state.spatial_result {
        Some(ref r) => r,
        None => return,
    };

    // Check bounds
    let shape = result.converged_map.shape();
    if y >= shape[0] || x >= shape[1] {
        return;
    }

    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(format!("Pixel Inspector ({}, {})", y, x)).strong());
            ui.add_space(4.0);

            let converged = result.converged_map[[y, x]];
            let (label, color) = if converged {
                ("Converged", crate::theme::semantic::GREEN)
            } else {
                ("NOT converged", crate::theme::semantic::RED)
            };
            ui.label(egui::RichText::new(label).color(color));

            ui.label(format!("chi2_r = {:.4}", result.chi_squared_map[[y, x]]));

            let enabled: Vec<_> = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .collect();

            for (i, entry) in enabled.iter().enumerate() {
                if i < result.density_maps.len() {
                    let density = result.density_maps[i][[y, x]];
                    let unc = result
                        .uncertainty_maps
                        .get(i)
                        .map(|u| u[[y, x]])
                        .map_or("N/A".to_string(), |u| format!("{:.2e}", u));
                    ui.label(format!(
                        "  {}: {:.6e} +/- {} atoms/barn",
                        entry.symbol, density, unc
                    ));
                }
            }
        });
}
