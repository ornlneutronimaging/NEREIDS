//! Step 5: Results — spatial map results display, pixel inspector, and export.

use crate::state::{AppState, Colormap};
use crate::widgets::image_view::{
    apply_colormap, data_range, render_to_rgba, show_colormapped_image,
};

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
    summary_card(ui, result, &state.isotope_entries);
    ui.add_space(12.0);

    // -- Density Map Grid --
    density_map_grid(ui, state);
    ui.add_space(12.0);

    // -- Pixel Inspector --
    pixel_inspector(ui, state);
    ui.add_space(12.0);
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
                                draw_colorbar(ui, data, colormap);
                            });
                        } else if let Some((y, x)) =
                            show_colormapped_image(ui, data, &tex_id, colormap)
                        {
                            state.selected_pixel = Some((y, x));
                            state.pixel_fit_result = None;
                        }

                        tile_toolbelt(ui, data, i, label, state);
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
                            draw_colorbar(ui, &conv_f64, colormap);
                        });
                    } else {
                        let _ = show_colormapped_image(ui, &conv_f64, "result_conv_map", colormap);
                    }

                    ui.label(egui::RichText::new(format!("{n_converged}/{n_total}")).small());
                    tile_toolbelt(ui, &conv_f64, conv_idx, "convergence", state);
                });
            });
        });
}

/// Per-tile toolbelt: colormap selector, colorbar toggle, save PNG.
fn tile_toolbelt(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tile_idx: usize,
    label: &str,
    state: &mut AppState,
) {
    ui.horizontal(|ui| {
        // Colormap selector
        if let Some(tile) = state.tile_display.get_mut(tile_idx) {
            let current_label = tile.colormap.label();
            egui::ComboBox::from_id_salt(format!("cmap_{tile_idx}"))
                .selected_text(current_label)
                .width(80.0)
                .show_ui(ui, |ui| {
                    for cmap in Colormap::ALL {
                        ui.selectable_value(&mut tile.colormap, cmap, cmap.label());
                    }
                });

            // Colorbar toggle
            ui.checkbox(&mut tile.show_colorbar, "Bar");
        }

        // Save PNG button
        if ui.small_button("Save PNG").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name(format!("{label}.png"))
                .add_filter("PNG", &["png"])
                .save_file()
            {
                let colormap = state
                    .tile_display
                    .get(tile_idx)
                    .map_or(Colormap::Viridis, |t| t.colormap);
                let rgba = render_to_rgba(data, colormap);
                let (h, w) = (data.shape()[0] as u32, data.shape()[1] as u32);
                match image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8) {
                    Ok(()) => {
                        state.status_message = format!("Saved PNG: {}", path.display());
                    }
                    Err(e) => {
                        state.status_message = format!("PNG save error: {e}");
                    }
                }
            }
        }
    });
}

/// Draw a vertical colorbar gradient strip next to an image.
fn draw_colorbar(ui: &mut egui::Ui, data: &ndarray::Array2<f64>, colormap: Colormap) {
    let bar_width = 16.0;
    let bar_height = 128.0;
    let n_steps = 64;

    let (vmin, vmax) = data_range(data);

    let (response, painter) = ui.allocate_painter(
        egui::Vec2::new(bar_width + 50.0, bar_height),
        egui::Sense::hover(),
    );
    let rect = response.rect;
    let bar_rect = egui::Rect::from_min_size(rect.min, egui::Vec2::new(bar_width, bar_height));

    // Draw gradient steps
    let step_height = bar_height / n_steps as f32;
    for i in 0..n_steps {
        // Map from top (high) to bottom (low)
        let t = 1.0 - (i as f64 / (n_steps - 1) as f64);
        let (r, g, b) = apply_colormap(colormap, t);
        let y0 = bar_rect.top() + i as f32 * step_height;
        let step_rect = egui::Rect::from_min_size(
            egui::pos2(bar_rect.left(), y0),
            egui::Vec2::new(bar_width, step_height),
        );
        painter.rect_filled(step_rect, 0.0, egui::Color32::from_rgb(r, g, b));
    }

    // Border
    painter.rect_stroke(
        bar_rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::GRAY),
        egui::StrokeKind::Outside,
    );

    // Min/max labels
    let label_x = bar_rect.right() + 4.0;
    painter.text(
        egui::pos2(label_x, bar_rect.top()),
        egui::Align2::LEFT_TOP,
        format!("{vmax:.2e}"),
        egui::FontId::proportional(10.0),
        ui.visuals().text_color(),
    );
    painter.text(
        egui::pos2(label_x, bar_rect.bottom()),
        egui::Align2::LEFT_BOTTOM,
        format!("{vmin:.2e}"),
        egui::FontId::proportional(10.0),
        ui.visuals().text_color(),
    );
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
                    let unc = result.uncertainty_maps.get(i).map(|u| u[[y, x]]).map_or(
                        "N/A".to_string(),
                        |u| {
                            if u.is_finite() {
                                format!("{:.2e}", u)
                            } else {
                                "N/A".to_string()
                            }
                        },
                    );
                    ui.label(format!(
                        "  {}: {:.6e} +/- {} atoms/barn",
                        entry.symbol, density, unc
                    ));
                }
            }
        });
}
