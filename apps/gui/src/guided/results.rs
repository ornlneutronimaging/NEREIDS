//! Step 5: Results — spatial map results display, pixel inspector, and export.

use super::result_widgets;
use crate::state::{AppState, Colormap};
use crate::widgets::image_view::show_colormapped_image;

/// Draw the Results step content.
pub fn results_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Results");
    ui.separator();

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

    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Density Maps").strong());
            ui.add_space(4.0);

            ui.horizontal_wrapped(|ui| {
                for i in 0..n_density {
                    let data = &density_maps[i];
                    let label = &symbols[i];
                    let tex_id = format!("result_density_{i}");

                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(format!("{label} density")).small());

                        let colormap = state
                            .tile_display
                            .get(i)
                            .map_or(Colormap::Viridis, |t| t.colormap);
                        let show_bar = state.tile_display.get(i).is_some_and(|t| t.show_colorbar);

                        if show_bar {
                            ui.horizontal(|ui| {
                                if let Some((y, x)) =
                                    show_colormapped_image(ui, data, &tex_id, colormap)
                                {
                                    state.selected_pixel = Some((y, x));
                                    state.pixel_fit_result = None;
                                }
                                result_widgets::draw_colorbar(ui, data, colormap);
                            });
                        } else if let Some((y, x)) =
                            show_colormapped_image(ui, data, &tex_id, colormap)
                        {
                            state.selected_pixel = Some((y, x));
                            state.pixel_fit_result = None;
                        }

                        result_widgets::tile_toolbelt(ui, data, i, label, state);
                    });
                }

                // Convergence map
                let conv_idx = n_density;
                ui.vertical(|ui| {
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
                            let _ =
                                show_colormapped_image(ui, &conv_f64, "result_conv_map", colormap);
                            result_widgets::draw_colorbar(ui, &conv_f64, colormap);
                        });
                    } else {
                        let _ = show_colormapped_image(ui, &conv_f64, "result_conv_map", colormap);
                    }

                    ui.label(egui::RichText::new(format!("{n_converged}/{n_total}")).small());
                    result_widgets::tile_toolbelt(ui, &conv_f64, conv_idx, "convergence", state);
                });
            });
        });
}
