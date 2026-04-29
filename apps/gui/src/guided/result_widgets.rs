//! Shared result display widgets used by both Guided Results step and Studio mode.

use crate::state::{AppState, Colormap, ExportFormat, ProvenanceEventKind, TileDisplayState};
use crate::widgets::design::{self, BadgeVariant};
use crate::widgets::image_view::{apply_colormap, data_range, render_to_rgba};

/// Summary statistics card showing convergence and density stats.
///
/// When `uncertainty_is_estimated` is true, chi-squared values are displayed
/// with a warning badge since they are based on estimated (not measured) uncertainty.
pub fn summary_card(
    ui: &mut egui::Ui,
    result: &nereids_pipeline::spatial::SpatialResult,
    uncertainty_is_estimated: bool,
) {
    let pct = if result.n_total > 0 {
        100.0 * result.n_converged as f64 / result.n_total as f64
    } else {
        0.0
    };
    let conv_badge = if pct > 95.0 {
        BadgeVariant::Green
    } else if pct > 50.0 {
        BadgeVariant::Orange
    } else {
        BadgeVariant::Red
    };

    design::card_with_header(
        ui,
        "Summary Statistics",
        Some((&format!("{:.0}%", pct), conv_badge)),
        |ui| {
            ui.label(format!(
                "Converged: {} / {} ({:.1}%)",
                result.n_converged, result.n_total, pct
            ));

            // Mean GOF.  Label swaps to "D/dof" for the counts-KL
            // dispatch (when deviance_per_dof_map is populated) —
            // memo 35 §P1.2 naming.
            let gof_label = if result.deviance_per_dof_map.is_some() {
                "D/dof"
            } else {
                "chi2_r"
            };
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
                ui.horizontal(|ui| {
                    if uncertainty_is_estimated {
                        ui.label(
                            egui::RichText::new(format!(
                                "Mean {}: {:.4} (approx.)",
                                gof_label, mean_chi2
                            ))
                            .color(crate::theme::semantic::ORANGE),
                        );
                    } else {
                        ui.label(format!("Mean {}: {:.4}", gof_label, mean_chi2));
                    }
                    let chi2_variant = if mean_chi2 < 2.0 {
                        BadgeVariant::Green
                    } else if mean_chi2 < 5.0 {
                        BadgeVariant::Orange
                    } else {
                        BadgeVariant::Red
                    };
                    let chi2_text = format!("{:.2}", mean_chi2);
                    design::badge(ui, &chi2_text, chi2_variant);
                });
                if uncertainty_is_estimated {
                    ui.label(
                        egui::RichText::new(
                            "⚠ chi² approximate (uncertainty estimated, not measured)",
                        )
                        .small()
                        .color(crate::theme::semantic::ORANGE),
                    );
                }
            }

            // Mean temperature (if available)
            if let Some(ref t_map) = result.temperature_map {
                let (sum_t, count_t) = t_map.iter().zip(result.converged_map.iter()).fold(
                    (0.0_f64, 0_usize),
                    |(sum, count), (&t, &conv)| {
                        if conv && t.is_finite() {
                            (sum + t, count + 1)
                        } else {
                            (sum, count)
                        }
                    },
                );
                if count_t > 0 {
                    let mean_t = sum_t / count_t as f64;
                    ui.label(format!("Mean temperature: {mean_t:.1} K"));
                }
            }

            // Per-isotope mean density
            for (i, label) in result.isotope_labels.iter().enumerate() {
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
                        ui.label(format!("{label}: mean density = {:.4e} atoms/barn", mean));
                    }
                }
            }
        },
    );
}

/// Per-tile toolbelt: colormap selector, colorbar toggle, save PNG.
///
/// Takes split borrows to avoid conflicting with `&state.spatial_result`
/// references held by the caller (density map data lives in spatial_result).
pub fn tile_toolbelt(
    ui: &mut egui::Ui,
    data: &ndarray::Array2<f64>,
    tile_idx: usize,
    label: &str,
    tile_display: &mut [TileDisplayState],
    status_message: &mut String,
) {
    ui.horizontal(|ui| {
        // Colormap selector
        if let Some(tile) = tile_display.get_mut(tile_idx) {
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
        if ui.small_button("Save PNG").clicked()
            && let Some(path) = rfd::FileDialog::new()
                .set_file_name(format!("{label}.png"))
                .add_filter("PNG", &["png"])
                .save_file()
        {
            let colormap = tile_display
                .get(tile_idx)
                .map_or(Colormap::Viridis, |t| t.colormap);
            let rgba = render_to_rgba(data, colormap);
            let (h, w) = (data.shape()[0] as u32, data.shape()[1] as u32);
            match image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8) {
                Ok(()) => {
                    *status_message = format!("Saved PNG: {}", path.display());
                }
                Err(e) => {
                    *status_message = format!("PNG save error: {e}");
                }
            }
        }
    });
}

/// Draw a vertical colorbar gradient strip next to an image.
pub fn draw_colorbar(ui: &mut egui::Ui, data: &ndarray::Array2<f64>, colormap: Colormap) {
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
pub fn pixel_inspector(ui: &mut egui::Ui, state: &AppState) {
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

    let converged = result.converged_map[[y, x]];
    let conv_badge = if converged {
        Some(("Converged", BadgeVariant::Green))
    } else {
        Some(("NOT converged", BadgeVariant::Red))
    };

    design::card_with_header(
        ui,
        &format!("Pixel Inspector ({}, {})", y, x),
        conv_badge,
        |ui| {
            let pixel_label = if result.deviance_per_dof_map.is_some() {
                "D/dof"
            } else {
                "chi2_r"
            };
            // Per-pixel maps carry NaN at unconverged pixels (NaN-on-failure
            // contract from #458 / PR-A B1/B2). Render an em-dash placeholder
            // instead of the literal string "NaN" so the inspector stays
            // legible — the "NOT converged" badge above already tells the
            // user the fit failed.
            let chi = result.chi_squared_map[[y, x]];
            if state.uncertainty_is_estimated {
                let text = if chi.is_finite() {
                    format!("{pixel_label} = {chi:.4} (approx.)")
                } else {
                    format!("{pixel_label} = \u{2014}")
                };
                ui.label(egui::RichText::new(text).color(crate::theme::semantic::ORANGE));
            } else if chi.is_finite() {
                ui.label(format!("{pixel_label} = {chi:.4}"));
            } else {
                ui.label(format!("{pixel_label} = \u{2014}"));
            }

            if let Some(ref t_map) = result.temperature_map {
                let t = t_map[[y, x]];
                if t.is_finite() {
                    ui.label(format!("  T = {t:.1} K"));
                }
            }

            let enabled: Vec<_> = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .collect();

            for (i, entry) in enabled.iter().enumerate() {
                if i < result.density_maps.len() {
                    let density = result.density_maps[i][[y, x]];
                    if !density.is_finite() {
                        ui.label(format!("  {}: \u{2014} atoms/barn", entry.symbol));
                    } else if state.uncertainty_is_estimated {
                        ui.label(format!("  {}: {:.6e} atoms/barn", entry.symbol, density));
                    } else {
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
            }
        },
    );
}

/// Execute the export based on the selected format.
pub fn run_export(state: &mut AppState) {
    let dir = match state.export_directory {
        Some(ref d) => d.clone(),
        None => return,
    };

    // Extract data needed from spatial_result
    let (
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        temperature_map,
        n_converged,
        n_total,
    ) = match state.spatial_result {
        Some(ref r) => (
            r.density_maps.clone(),
            r.uncertainty_maps.clone(),
            r.chi_squared_map.clone(),
            r.converged_map.clone(),
            r.temperature_map.clone(),
            r.n_converged,
            r.n_total,
        ),
        None => return,
    };

    let labels: Vec<String> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .map(|e| e.symbol.clone())
        .collect();

    let result = match state.export_format {
        ExportFormat::Tiff => export_tiff(&dir, &density_maps, &labels, temperature_map.as_ref()),
        ExportFormat::Hdf5 => export_hdf5(
            &dir,
            &density_maps,
            &uncertainty_maps,
            &chi_squared_map,
            &converged_map,
            &labels,
            temperature_map.as_ref(),
        ),
        ExportFormat::Markdown => export_markdown(
            &dir,
            &labels,
            &density_maps,
            &converged_map,
            n_converged,
            n_total,
            &state.provenance_log,
        ),
    };

    match result {
        Ok(msg) => {
            state.export_status = Some(msg.clone());
            state.log_provenance(
                ProvenanceEventKind::Exported,
                format!("Exported {:?} to {}", state.export_format, dir.display()),
            );
        }
        Err(e) => {
            state.export_status = Some(format!("Error: {e}"));
        }
    }
}

fn export_tiff(
    dir: &std::path::Path,
    density_maps: &[ndarray::Array2<f64>],
    labels: &[String],
    temperature_map: Option<&ndarray::Array2<f64>>,
) -> Result<String, String> {
    for (i, map) in density_maps.iter().enumerate() {
        let label = labels.get(i).map_or("unknown", |s| s.as_str());
        nereids_io::export::export_density_tiff(dir, map, label).map_err(|e| e.to_string())?;
    }
    if let Some(t_map) = temperature_map {
        nereids_io::export::export_map_tiff(dir, t_map, "temperature")
            .map_err(|e| e.to_string())?;
    }
    let n = density_maps.len() + temperature_map.is_some() as usize;
    Ok(format!("Exported {n} TIFF files to {}", dir.display()))
}

fn export_hdf5(
    dir: &std::path::Path,
    density_maps: &[ndarray::Array2<f64>],
    uncertainty_maps: &[ndarray::Array2<f64>],
    chi_squared_map: &ndarray::Array2<f64>,
    converged_map: &ndarray::Array2<bool>,
    labels: &[String],
    temperature_map: Option<&ndarray::Array2<f64>>,
) -> Result<String, String> {
    let path = dir.join("nereids_results.hdf5");
    nereids_io::export::export_results_hdf5(
        &path,
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        labels,
        temperature_map,
    )
    .map_err(|e| e.to_string())?;
    Ok(format!("Exported HDF5 to {}", path.display()))
}

fn export_markdown(
    dir: &std::path::Path,
    labels: &[String],
    density_maps: &[ndarray::Array2<f64>],
    converged_map: &ndarray::Array2<bool>,
    n_converged: usize,
    n_total: usize,
    provenance_log: &[crate::state::ProvenanceEvent],
) -> Result<String, String> {
    let path = dir.join("nereids_report.md");
    let provenance: Vec<(String, String)> = provenance_log
        .iter()
        .map(|ev| (ev.formatted_timestamp(), ev.message.clone()))
        .collect();
    nereids_io::export::export_markdown_report(
        &path,
        labels,
        density_maps,
        converged_map,
        n_converged,
        n_total,
        &provenance,
    )
    .map_err(|e| e.to_string())?;
    Ok(format!("Exported report to {}", path.display()))
}
