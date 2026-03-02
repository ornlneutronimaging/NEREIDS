//! Shared result display widgets used by both Guided Results step and Studio mode.

use crate::state::{AppState, Colormap, ExportFormat, ProvenanceEventKind};
use crate::widgets::image_view::{apply_colormap, data_range, render_to_rgba};

/// Summary statistics card showing convergence and density stats.
pub fn summary_card(
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

/// Per-tile toolbelt: colormap selector, colorbar toggle, save PNG.
pub fn tile_toolbelt(
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
        if ui.small_button("Save PNG").clicked()
            && let Some(path) = rfd::FileDialog::new()
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

/// Export panel: format selector, directory picker, export button.
pub fn export_panel(ui: &mut egui::Ui, state: &mut AppState) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Export Results").strong());
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                // Format selector
                ui.label("Format:");
                let current_label = state.export_format.label();
                egui::ComboBox::from_id_salt("export_format")
                    .selected_text(current_label)
                    .show_ui(ui, |ui| {
                        for fmt in ExportFormat::ALL {
                            ui.selectable_value(&mut state.export_format, fmt, fmt.label());
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Directory:");
                let dir_label = state
                    .export_directory
                    .as_ref()
                    .map_or("(not set)".to_string(), |p| p.display().to_string());
                ui.label(egui::RichText::new(dir_label).monospace());

                if ui.button("Browse...").clicked()
                    && let Some(path) = rfd::FileDialog::new().pick_folder()
                {
                    state.export_directory = Some(path);
                }
            });

            ui.add_space(4.0);

            let can_export = state.spatial_result.is_some() && state.export_directory.is_some();
            if ui
                .add_enabled(can_export, egui::Button::new("Export Results"))
                .clicked()
            {
                run_export(state);
            }

            if let Some(ref status) = state.export_status {
                ui.add_space(4.0);
                let color = if status.starts_with("Error") {
                    crate::theme::semantic::RED
                } else {
                    crate::theme::semantic::GREEN
                };
                ui.label(egui::RichText::new(status.as_str()).color(color));
            }
        });
}

/// Execute the export based on the selected format.
fn run_export(state: &mut AppState) {
    let dir = match state.export_directory {
        Some(ref d) => d.clone(),
        None => return,
    };

    // Extract data needed from spatial_result
    let (density_maps, uncertainty_maps, chi_squared_map, converged_map, n_converged, n_total) =
        match state.spatial_result {
            Some(ref r) => (
                r.density_maps.clone(),
                r.uncertainty_maps.clone(),
                r.chi_squared_map.clone(),
                r.converged_map.clone(),
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
        ExportFormat::Tiff => export_tiff(&dir, &density_maps, &labels),
        ExportFormat::Hdf5 => export_hdf5(
            &dir,
            &density_maps,
            &uncertainty_maps,
            &chi_squared_map,
            &converged_map,
            &labels,
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
) -> Result<String, String> {
    for (i, map) in density_maps.iter().enumerate() {
        let label = labels.get(i).map_or("unknown", |s| s.as_str());
        nereids_io::export::export_density_tiff(dir, map, label).map_err(|e| e.to_string())?;
    }
    Ok(format!(
        "Exported {} TIFF files to {}",
        density_maps.len(),
        dir.display()
    ))
}

fn export_hdf5(
    dir: &std::path::Path,
    density_maps: &[ndarray::Array2<f64>],
    uncertainty_maps: &[ndarray::Array2<f64>],
    chi_squared_map: &ndarray::Array2<f64>,
    converged_map: &ndarray::Array2<bool>,
    labels: &[String],
) -> Result<String, String> {
    let path = dir.join("nereids_results.hdf5");
    nereids_io::export::export_results_hdf5(
        &path,
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        labels,
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

/// Provenance log section (collapsible).
pub fn provenance_section(ui: &mut egui::Ui, state: &AppState) {
    if state.provenance_log.is_empty() {
        return;
    }

    egui::CollapsingHeader::new(egui::RichText::new("Provenance Log").strong())
        .default_open(false)
        .show(ui, |ui| {
            for event in state.provenance_log.iter().rev() {
                let ts = event.formatted_timestamp();

                let (kind_label, kind_color) = match event.kind {
                    ProvenanceEventKind::DataLoaded => ("LOAD", crate::theme::semantic::YELLOW),
                    ProvenanceEventKind::ConfigChanged => {
                        ("CONFIG", crate::theme::semantic::ORANGE)
                    }
                    ProvenanceEventKind::Normalized => ("NORM", crate::theme::semantic::GREEN),
                    ProvenanceEventKind::AnalysisRun => ("ANALYZE", crate::theme::semantic::ORANGE),
                    ProvenanceEventKind::Exported => ("EXPORT", crate::theme::semantic::GREEN),
                };

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&ts).monospace().small());
                    ui.label(egui::RichText::new(kind_label).small().color(kind_color));
                    ui.label(egui::RichText::new(&event.message).small());
                });
            }
        });
}
