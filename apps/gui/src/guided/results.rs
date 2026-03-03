//! Step 5: Results — spatial map results display, pixel inspector, and export.

use super::result_widgets;
use crate::state::{AppState, Colormap, GuidedStep};
use crate::widgets::design;
use crate::widgets::image_view::show_colormapped_image_with_roi;

/// Draw the Results step content.
pub fn results_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(ui, "Results", "Density maps and export");

    // Ensure tile_display is populated
    match state.spatial_result {
        Some(ref r) => {
            if state.tile_display.len() < r.density_maps.len() + 1 {
                let n = r.density_maps.len();
                state.init_tile_display(n);
            }
        }
        None => {
            ui.label("Run spatial mapping (Analyze step) to see results here.");
            return;
        }
    }

    let result = state.spatial_result.as_ref().unwrap();

    // -- Summary Statistics Card --
    result_widgets::summary_card(ui, result, &state.isotope_entries);
    ui.add_space(4.0);

    // -- Stat Row --
    {
        let conv_pct = if result.n_total > 0 {
            100.0 * result.n_converged as f64 / result.n_total as f64
        } else {
            0.0
        };
        let chi2_vals: Vec<f64> = result
            .chi_squared_map
            .iter()
            .zip(result.converged_map.iter())
            .filter(|(v, c)| **c && v.is_finite())
            .map(|(v, _)| *v)
            .collect();
        let mean_chi2 = if chi2_vals.is_empty() {
            0.0
        } else {
            chi2_vals.iter().sum::<f64>() / chi2_vals.len() as f64
        };
        let n_iso = result.density_maps.len();
        let pct = format!("{conv_pct:.1}%");
        let chi2 = format!("{mean_chi2:.2}");
        let n_iso_str = format!("{n_iso}");
        design::stat_row(
            ui,
            &[
                (&pct, "Converged"),
                (&chi2, "Mean \u{03C7}\u{00B2}\u{1D63}"),
                (&n_iso_str, "Isotopes"),
            ],
        );
    }
    ui.add_space(12.0);

    // -- Density Map Grid --
    density_map_grid(ui, state);
    ui.add_space(12.0);

    // -- Pixel Inspector --
    result_widgets::pixel_inspector(ui, state);
    ui.add_space(12.0);

    // -- Export Panel --
    result_widgets::export_panel(ui, state);
    ui.add_space(12.0);

    // -- Provenance Log --
    result_widgets::provenance_section(ui, state);
    ui.add_space(12.0);

    // -- Navigation --
    ui.add_space(8.0);
    if ui.button("\u{2190} Back to Analyze").clicked() {
        state.guided_step = GuidedStep::Analyze;
    }
}

/// Grid of density map tiles (one per isotope + convergence map).
fn density_map_grid(ui: &mut egui::Ui, state: &mut AppState) {
    // Extract data we need before taking mutable borrows
    let (density_maps, conv_f64, n_converged, n_total) = match state.spatial_result {
        Some(ref r) => {
            let conv = r.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
            (r.density_maps.clone(), conv, r.n_converged, r.n_total)
        }
        None => return,
    };

    let symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();
    let n_density = density_maps.len().min(symbols.len());

    let n_tiles = n_density + 1; // isotopes + convergence

    design::card_with_header(ui, "Density Maps", None, |ui| {
        // Compute layout inside the card so available_width accounts for card padding.
        let available_width = ui.available_width();
        let tile_min = 180.0_f32;
        let n_cols = ((available_width / tile_min) as usize).max(1).min(n_tiles);
        let tile_width = ((available_width - (n_cols - 1) as f32 * 8.0) / n_cols as f32).max(1.0);

        egui::Grid::new("density_map_grid")
            .num_columns(n_cols)
            .spacing([8.0, 8.0])
            .show(ui, |ui| {
                for i in 0..n_tiles {
                    ui.allocate_ui(egui::vec2(tile_width, tile_width + 40.0), |ui| {
                        if i < n_density {
                            // Isotope density tile
                            let data = &density_maps[i];
                            let label = &symbols[i];
                            let tex_id = format!("result_density_{i}");

                            ui.label(egui::RichText::new(format!("{label} density")).small());

                            let colormap = state
                                .tile_display
                                .get(i)
                                .map_or(Colormap::Viridis, |t| t.colormap);
                            let show_bar =
                                state.tile_display.get(i).is_some_and(|t| t.show_colorbar);

                            let selected = state.selected_pixel;
                            if show_bar {
                                ui.horizontal(|ui| {
                                    if let Some((y, x)) = show_colormapped_image_with_roi(
                                        ui, data, &tex_id, colormap, None, selected,
                                    )
                                    .0
                                    {
                                        state.selected_pixel = Some((y, x));
                                        state.pixel_fit_result = None;
                                    }
                                    result_widgets::draw_colorbar(ui, data, colormap);
                                });
                            } else if let Some((y, x)) = show_colormapped_image_with_roi(
                                ui, data, &tex_id, colormap, None, selected,
                            )
                            .0
                            {
                                state.selected_pixel = Some((y, x));
                                state.pixel_fit_result = None;
                            }

                            result_widgets::tile_toolbelt(ui, data, i, label, state);
                        } else {
                            // Convergence map tile
                            let conv_idx = n_density;

                            ui.label(egui::RichText::new("Convergence").small());

                            let colormap = state
                                .tile_display
                                .get(conv_idx)
                                .map_or(Colormap::Viridis, |t| t.colormap);
                            let show_bar = state
                                .tile_display
                                .get(conv_idx)
                                .is_some_and(|t| t.show_colorbar);

                            if show_bar {
                                ui.horizontal(|ui| {
                                    let _ = show_colormapped_image_with_roi(
                                        ui,
                                        &conv_f64,
                                        "result_conv_map",
                                        colormap,
                                        None,
                                        None,
                                    );
                                    result_widgets::draw_colorbar(ui, &conv_f64, colormap);
                                });
                            } else {
                                let _ = show_colormapped_image_with_roi(
                                    ui,
                                    &conv_f64,
                                    "result_conv_map",
                                    colormap,
                                    None,
                                    None,
                                );
                            }

                            ui.label(
                                egui::RichText::new(format!("{n_converged}/{n_total}")).small(),
                            );
                            result_widgets::tile_toolbelt(
                                ui,
                                &conv_f64,
                                conv_idx,
                                "convergence",
                                state,
                            );
                        }
                    });
                    if (i + 1) % n_cols == 0 {
                        ui.end_row();
                    }
                }
            });
    });
}
