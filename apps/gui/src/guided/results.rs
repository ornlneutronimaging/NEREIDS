//! Step 5: Results — spatial map results display, pixel inspector, and export.

use super::result_widgets;
use crate::state::{AppState, Colormap};
use crate::widgets::design;
use crate::widgets::image_view::{show_colormapped_image_with_roi, show_density_overlay};

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
        // Single-pass fold: compute sum + count without allocating a Vec.
        let (chi2_sum, chi2_count) = result
            .chi_squared_map
            .iter()
            .zip(result.converged_map.iter())
            .filter(|(v, c)| **c && v.is_finite())
            .fold((0.0_f64, 0usize), |(s, n), (v, _)| (s + *v, n + 1));
        let mean_chi2 = if chi2_count == 0 {
            0.0
        } else {
            chi2_sum / chi2_count as f64
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
    if ui.button("\u{2190} Back").clicked() {
        state.nav_prev();
    }
}

/// Grid of density map tiles (one per isotope + convergence map).
fn density_map_grid(ui: &mut egui::Ui, state: &mut AppState) {
    let result = match state.spatial_result {
        Some(ref r) => r,
        None => return,
    };
    let n_density_maps = result.density_maps.len();
    let n_converged = result.n_converged;
    let n_total = result.n_total;

    // Cache converged_map → f64 conversion in egui temp data to avoid per-frame mapv.
    let conv_cache_id = egui::Id::new("result_conv_f64_cache");
    let conv_ptr = result.converged_map.as_ptr() as usize;
    let conv_f64: ndarray::Array2<f64> = {
        let cached: Option<(usize, ndarray::Array2<f64>)> = ui.data(|d| d.get_temp(conv_cache_id));
        if let Some((ptr, arr)) = cached
            && ptr == conv_ptr
        {
            arr
        } else {
            let arr = result.converged_map.mapv(|b| if b { 1.0 } else { 0.0 });
            ui.data_mut(|d| d.insert_temp(conv_cache_id, (conv_ptr, arr.clone())));
            arr
        }
    };

    let symbols: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();
    let n_density = n_density_maps.min(symbols.len());
    let n_tiles = n_density + 1; // isotopes + convergence

    // Overlay toggle (only when fitting_rois is non-empty)
    let has_rois = !state.fitting_rois.is_empty();
    if has_rois {
        ui.horizontal(|ui| {
            ui.checkbox(&mut state.show_density_overlay, "Overlay on preview");
            if state.show_density_overlay {
                ui.label(
                    egui::RichText::new("Density shown only in fitted ROI regions")
                        .size(11.0)
                        .weak(),
                );
            }
        });
        ui.add_space(4.0);
    }

    let use_overlay = has_rois && state.show_density_overlay && state.preview_image.is_some();

    // Collect click events from tiles, apply to state after rendering.
    let mut new_pixel: Option<(usize, usize)> = None;

    design::card_with_header(ui, "Density Maps", None, |ui| {
        let available_width = ui.available_width();
        let tile_min = 180.0_f32;
        let n_cols = ((available_width / tile_min) as usize).max(1).min(n_tiles);

        let n_rows = n_tiles.div_ceil(n_cols);
        for row in 0..n_rows {
            ui.columns(n_cols, |columns| {
                for (col, ui) in columns.iter_mut().enumerate().take(n_cols) {
                    let i = row * n_cols + col;
                    if i >= n_tiles {
                        continue;
                    }

                    if i < n_density {
                        // Isotope density tile — render image in a short borrow scope,
                        // then call tile_toolbelt with a fresh borrow.
                        let label = &symbols[i];
                        let tex_id = format!("result_density_{i}");

                        ui.label(egui::RichText::new(format!("{label} density")).small());

                        let colormap = state
                            .tile_display
                            .get(i)
                            .map_or(Colormap::Viridis, |t| t.colormap);
                        let show_bar = state.tile_display.get(i).is_some_and(|t| t.show_colorbar);
                        let selected = state.selected_pixel;

                        // Render image (borrows state.spatial_result immutably).
                        {
                            let data = &state.spatial_result.as_ref().unwrap().density_maps[i];
                            if use_overlay {
                                let preview = state.preview_image.as_ref().unwrap();
                                let (clicked, _rect) = show_density_overlay(
                                    ui,
                                    preview,
                                    data,
                                    &state.fitting_rois,
                                    &tex_id,
                                    colormap,
                                    selected,
                                );
                                if clicked.is_some() {
                                    new_pixel = clicked;
                                }
                            } else if show_bar {
                                ui.horizontal(|ui| {
                                    if let Some(px) = show_colormapped_image_with_roi(
                                        ui,
                                        data,
                                        &tex_id,
                                        colormap,
                                        &[],
                                        selected,
                                    )
                                    .0
                                    {
                                        new_pixel = Some(px);
                                    }
                                    result_widgets::draw_colorbar(ui, data, colormap);
                                });
                            } else if let Some(px) = show_colormapped_image_with_roi(
                                ui,
                                data,
                                &tex_id,
                                colormap,
                                &[],
                                selected,
                            )
                            .0
                            {
                                new_pixel = Some(px);
                            }
                        }

                        // Toolbelt — uses split borrows to avoid conflicting with
                        // the spatial_result reference used for image rendering.
                        let data = &state.spatial_result.as_ref().unwrap().density_maps[i];
                        result_widgets::tile_toolbelt(
                            ui,
                            data,
                            i,
                            label,
                            &mut state.tile_display,
                            &mut state.status_message,
                        );
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
                                    &[],
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
                                &[],
                                None,
                            );
                        }

                        ui.label(egui::RichText::new(format!("{n_converged}/{n_total}")).small());
                        result_widgets::tile_toolbelt(
                            ui,
                            &conv_f64,
                            conv_idx,
                            "convergence",
                            &mut state.tile_display,
                            &mut state.status_message,
                        );
                    }
                }
            });
            if row + 1 < n_rows {
                ui.add_space(8.0);
            }
        }
    });

    // Apply deferred click outside the rendering loop.
    if let Some((y, x)) = new_pixel {
        state.selected_pixel = Some((y, x));
        state.pixel_fit_result = None;
    }
}
