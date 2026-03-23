//! Studio mode — "Final Cut"-style IDE workspace with document tabs,
//! resizable bottom dock, and shared tool embedding.
//!
//! Layout:
//!   1. Bottom dock (resizable, 4 tabs: Isotopes/Residuals/Provenance/Export)
//!   2. Mini-inspector sidebar (right, 220px, Analysis tab only)
//!   3. Central panel with document tabs (Analysis / Forward Model / Detectability)

use crate::guided::{detectability, forward_model, result_widgets};
use crate::state::{AppState, Colormap, EndfStatus, SpectrumAxis, StudioDocTab};
use crate::theme::ThemeColors;
use crate::widgets::design;
use crate::widgets::image_view::show_colormapped_image;
use egui_plot::{Line, Plot, PlotPoints};

/// Render the Studio mode content.
pub fn studio_content(ctx: &egui::Context, state: &mut AppState) {
    let has_results = state.spatial_result.is_some();

    // Ensure tile_display is populated when results exist
    if let Some(ref r) = state.spatial_result {
        let has_temp = r.temperature_map.is_some();
        let needed = r.density_maps.len() + 1 + has_temp as usize;
        if state.tile_display.len() < needed {
            state.init_tile_display(r.density_maps.len());
        }
    }

    // 1. Bottom dock (before side panel and central panel — egui ordering)
    if state.studio_show_dock {
        bottom_dock(ctx, state);
    }

    // 2. Parameter sidebar (left, Analysis tab only)
    if state.studio_doc_tab == StudioDocTab::Analysis {
        parameter_sidebar(ctx, state);
    }

    // 3. Central panel: doc tab bar + routed content
    let colors = ThemeColors::from_ctx(ctx);
    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::same(12)),
        )
        .show(ctx, |ui| {
            doc_tab_bar(ui, state);
            ui.add_space(4.0);

            match state.studio_doc_tab {
                StudioDocTab::Analysis => {
                    if has_results {
                        analysis_tab(ui, state);
                    } else {
                        no_results_placeholder(ui);
                    }
                }
                StudioDocTab::ForwardModel => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        studio_forward_model(ui, state);
                    });
                }
                StudioDocTab::Detectability => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        studio_detectability(ui, state);
                    });
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Document tab bar
// ---------------------------------------------------------------------------

fn doc_tab_bar(ui: &mut egui::Ui, state: &mut AppState) {
    let labels = &["Analysis", "Forward Model", "Detectability"];
    let mut idx = state.studio_doc_tab as usize;
    design::underline_tabs(ui, labels, &mut idx);
    state.studio_doc_tab = match idx {
        1 => StudioDocTab::ForwardModel,
        2 => StudioDocTab::Detectability,
        _ => StudioDocTab::Analysis,
    };
}

// ---------------------------------------------------------------------------
// Analysis tab: split-pane density map + spectrum
// ---------------------------------------------------------------------------

fn analysis_tab(ui: &mut egui::Ui, state: &mut AppState) {
    // Extract data before mutable borrows
    let (n_density, symbols, density_maps) = match state.spatial_result {
        Some(ref r) => {
            let symbols = r.isotope_labels.clone();
            let n_density = r.density_maps.len().min(symbols.len());
            let density_maps = r.density_maps.clone();
            (n_density, symbols, density_maps)
        }
        None => return,
    };

    if n_density == 0 {
        ui.label("No density maps available.");
        return;
    }

    // Clamp isotope index (account for optional temperature map entry).
    // Also reset to 0 when the selected index no longer maps to the same
    // isotope (e.g. after enable/disable toggling in the isotopes card).
    let has_temp_map_early = state
        .spatial_result
        .as_ref()
        .is_some_and(|r| r.temperature_map.is_some());
    let n_options_early = n_density + has_temp_map_early as usize;
    if state.studio_analysis_isotope >= n_options_early {
        state.studio_analysis_isotope = 0;
    }
    // If the symbol at the selected index doesn't match what the user last
    // saw, reset to 0 to avoid silently showing a different isotope's data.
    if let Some(prev_sym) = &state.studio_analysis_prev_symbol
        && symbols.get(state.studio_analysis_isotope) != Some(prev_sym)
    {
        state.studio_analysis_isotope = 0;
    }

    let available = ui.available_width();
    let left_width = (available * 0.55).max(200.0);
    let right_width = (available - left_width - 12.0).max(200.0);

    ui.horizontal(|ui| {
        // ---- Left column: density map ----
        ui.allocate_ui_with_layout(
            egui::vec2(left_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                analysis_map_column(ui, state, n_density, &symbols, &density_maps);
            },
        );

        ui.separator();

        // ---- Right column: spectrum ----
        ui.allocate_ui_with_layout(
            egui::vec2(right_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                analysis_spectrum_column(ui, state);
            },
        );
    });
}

/// Left column of Analysis tab: isotope selector + density map + toolbelt.
fn analysis_map_column(
    ui: &mut egui::Ui,
    state: &mut AppState,
    n_density: usize,
    symbols: &[String],
    density_maps: &[ndarray::Array2<f64>],
) {
    let has_temp_map = state
        .spatial_result
        .as_ref()
        .is_some_and(|r| r.temperature_map.is_some());
    // Index mapping: 0..n_density = isotopes, n_density = temperature (if present)
    let n_options = n_density + has_temp_map as usize;

    // Header: map selector (colormap/save controls are in tile_toolbelt below)
    ui.horizontal(|ui| {
        ui.label("Map:");
        let sel = state.studio_analysis_isotope;
        let sel_text = if sel < n_density {
            symbols.get(sel).map_or("—", |s| s.as_str())
        } else if has_temp_map && sel == n_density {
            "Temperature"
        } else {
            "—"
        };
        egui::ComboBox::from_id_salt("analysis_isotope_sel")
            .selected_text(sel_text)
            .show_ui(ui, |ui| {
                for (i, sym) in symbols.iter().enumerate().take(n_density) {
                    ui.selectable_value(&mut state.studio_analysis_isotope, i, sym);
                }
                if has_temp_map {
                    ui.selectable_value(
                        &mut state.studio_analysis_isotope,
                        n_density,
                        "Temperature",
                    );
                }
            });
    });
    ui.add_space(4.0);

    // Clamp
    if state.studio_analysis_isotope >= n_options {
        state.studio_analysis_isotope = 0;
    }

    // Track which symbol is displayed so we detect isotope list mutations.
    state.studio_analysis_prev_symbol = symbols.get(state.studio_analysis_isotope).cloned();

    let tile_idx = state.studio_analysis_isotope;
    let colormap = state
        .tile_display
        .get(tile_idx)
        .map_or(Colormap::Viridis, |t| t.colormap);
    let show_bar = state
        .tile_display
        .get(tile_idx)
        .is_some_and(|t| t.show_colorbar);

    // Select the appropriate map data
    let map_data: Option<&ndarray::Array2<f64>> = if tile_idx < n_density {
        density_maps.get(tile_idx)
    } else if has_temp_map && tile_idx == n_density {
        state
            .spatial_result
            .as_ref()
            .and_then(|r| r.temperature_map.as_ref())
    } else {
        None
    };

    if let Some(data) = map_data {
        // Colorbar width: 16px bar + 50px labels = 66px + 4px spacing.
        let colorbar_reserved = if show_bar { 70.0 } else { 0.0 };

        // Compute image display size using the parent's available dimensions
        // (BEFORE entering a horizontal, where available_height collapses to
        // one line height and produces a tiny image).
        let (dh, dw) = (data.shape()[0], data.shape()[1]);
        let img_avail_w = (ui.available_width() - colorbar_reserved).max(32.0);
        let img_avail_h = ui.available_height();
        let scale_w = img_avail_w / dw.max(1) as f32;
        let scale = if img_avail_h > 2000.0 {
            scale_w
        } else {
            let scale_h = img_avail_h / dh.max(1) as f32;
            scale_w.min(scale_h)
        };
        let img_height = dh as f32 * scale;

        // Use allocate_ui_with_layout with the pre-computed height so the
        // horizontal strip has enough room for the image.
        let strip_height = img_height.max(128.0); // at least colorbar height
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), strip_height),
            egui::Layout::left_to_right(egui::Align::TOP),
            |ui| {
                if let Some((y, x)) =
                    show_colormapped_image(ui, data, "studio_analysis_map", colormap)
                {
                    state.selected_pixel = Some((y, x));
                    state.pixel_fit_result = None;
                    state.residuals_cache = None;
                }
                if show_bar {
                    result_widgets::draw_colorbar(ui, data, colormap);
                }
            },
        );

        // Toolbelt
        let label = if tile_idx < n_density {
            symbols
                .get(tile_idx)
                .map_or("unknown", |s| s.as_str())
                .to_string()
        } else {
            "temperature".to_string()
        };
        result_widgets::tile_toolbelt(
            ui,
            data,
            tile_idx,
            &label,
            &mut state.tile_display,
            &mut state.status_message,
        );
    }
}

/// Right column of Analysis tab: axis toggle + spectrum plot + fit results.
fn analysis_spectrum_column(ui: &mut egui::Ui, state: &mut AppState) {
    // Axis toggle
    ui.horizontal(|ui| {
        ui.label("Axis:");
        ui.selectable_value(
            &mut state.analyze_spectrum_axis,
            SpectrumAxis::EnergyEv,
            "Energy (eV)",
        );
        ui.selectable_value(
            &mut state.analyze_spectrum_axis,
            SpectrumAxis::TofMicroseconds,
            "TOF (\u{03bc}s)",
        );
    });
    ui.add_space(4.0);

    // Need normalized data for spectrum
    let norm = match state.normalized {
        Some(ref n) => n.clone(),
        None => {
            ui.label("No normalized data available for spectrum.");
            return;
        }
    };

    let n_tof = norm.transmission.shape()[0];

    // Build x-axis values
    let Some((x_values, x_label)) = design::build_spectrum_x_axis(&design::SpectrumXAxisParams {
        axis: state.analyze_spectrum_axis,
        energies: state.energies.as_deref(),
        spectrum_values: state.spectrum_values.as_ref().map(|v| v.as_slice()),
        spectrum_unit: state.spectrum_unit,
        spectrum_kind: state.spectrum_kind,
        flight_path_m: state.beamline.flight_path_m,
        delay_us: state.beamline.delay_us,
        n_tof,
    }) else {
        ui.label("No energy grid loaded.");
        return;
    };

    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => {
            ui.label("Click the density map to view the spectrum at a pixel.");
            return;
        }
    };

    let shape = norm.transmission.shape();
    if y >= shape[1] || x >= shape[2] {
        ui.label("Selected pixel is out of bounds.");
        return;
    }

    let n_plot = n_tof.min(x_values.len());
    if n_plot == 0 {
        return;
    }

    // Measured spectrum
    let measured_points: PlotPoints = (0..n_plot)
        .filter(|&i| x_values[i].is_finite())
        .map(|i| [x_values[i], norm.transmission[[i, y, x]]])
        .collect();
    let measured_line = Line::new("Measured", measured_points);

    // Fit curve (if available)
    let fit_line = state.pixel_fit_result.as_ref().and_then(|result| {
        let energies = state.energies.as_ref()?;
        let (all_rd, density_indices, density_ratios) =
            design::collect_all_resonance_data_with_mapping(state);
        design::build_fit_line(&design::FitLineParams {
            result,
            resonance_data: &all_rd,
            density_indices: &density_indices,
            density_ratios: &density_ratios,
            energies,
            temperature_k: state.temperature_k,
            x_values: &x_values,
            n_plot,
        })
    });

    // Spectrum plot
    let plot_height = ui.available_height().clamp(200.0, 400.0);
    Plot::new("studio_spectrum")
        .height(plot_height)
        .x_axis_label(x_label)
        .y_axis_label("Transmission")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(measured_line);
            if let Some(fit) = fit_line {
                plot_ui.line(fit);
            }
        });

    // Fit results below plot
    if let Some(ref result) = state.pixel_fit_result {
        ui.separator();
        ui.horizontal(|ui| {
            let (label, color) = if result.converged {
                ("Converged", crate::theme::semantic::GREEN)
            } else {
                ("NOT converged", crate::theme::semantic::RED)
            };
            ui.label(egui::RichText::new(label).color(color).strong());
            if state.uncertainty_is_estimated {
                ui.label(
                    egui::RichText::new(format!(
                        "chi2_r = {:.4} (approx.)",
                        result.reduced_chi_squared
                    ))
                    .color(crate::theme::semantic::ORANGE),
                );
            } else {
                ui.label(format!("chi2_r = {:.4}", result.reduced_chi_squared));
            }
            ui.label(format!("iter = {}", result.iterations));
            if let Some(t) = result.temperature_k {
                if !state.uncertainty_is_estimated {
                    if let Some(u) = result.temperature_k_unc {
                        ui.label(format!("T = {t:.1} \u{00b1} {u:.1} K"));
                    } else {
                        ui.label(format!("T = {t:.1} K"));
                    }
                } else {
                    ui.label(format!("T = {t:.1} K"));
                }
            }
        });

        for (i, entry) in state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .enumerate()
        {
            if i < result.densities.len() {
                let dot_color = design::isotope_dot_color(&entry.symbol);
                ui.horizontal(|ui| {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                    if state.uncertainty_is_estimated {
                        ui.label(format!("{}: {:.4e}", entry.symbol, result.densities[i]));
                    } else {
                        let unc_str = result
                            .uncertainties
                            .as_ref()
                            .and_then(|u| u.get(i))
                            .map_or("N/A".to_string(), |u| format!("{:.2e}", u));
                        ui.label(format!(
                            "{}: {:.4e} \u{00b1} {}",
                            entry.symbol, result.densities[i], unc_str
                        ));
                    }
                });
            }
        }

        // Group density results
        let enabled_individual_count = state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .count();
        let mut group_idx = enabled_individual_count;
        for group in &state.isotope_groups {
            if group.enabled && group.overall_status() == EndfStatus::Loaded {
                if group_idx < result.densities.len() {
                    let dot_color = design::isotope_dot_color(&group.name);
                    ui.horizontal(|ui| {
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                        if state.uncertainty_is_estimated {
                            ui.label(format!(
                                "{}: {:.4e}",
                                group.name, result.densities[group_idx]
                            ));
                        } else {
                            let unc_str = result
                                .uncertainties
                                .as_ref()
                                .and_then(|u| u.get(group_idx))
                                .map_or("N/A".to_string(), |u| format!("{:.2e}", u));
                            ui.label(format!(
                                "{}: {:.4e} \u{00b1} {}",
                                group.name, result.densities[group_idx], unc_str
                            ));
                        }
                    });
                }
                group_idx += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FM and Detect wrappers (skip guided headers/teleport pills)
// ---------------------------------------------------------------------------

fn studio_forward_model(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(ui, "Forward Model", "Simulated transmission spectrum");

    ui.horizontal(|ui| {
        // Sync buttons (disabled during ENDF fetches to prevent index corruption).
        // Guard "Copy" by both FM and main fetch flags — copying mid-fetch
        // from main would create orphaned Fetching entries in FM.
        ui.add_enabled_ui(
            !state.is_fetching_fm_endf && !state.is_fetching_endf,
            |ui| {
                if ui.button("Copy from Config").clicked() {
                    forward_model::copy_config_to_fm(state);
                }
            },
        );
        ui.add_enabled_ui(!state.is_fetching_endf, |ui| {
            if ui.button("Push to Config").clicked() {
                forward_model::push_fm_to_config(state);
            }
        });
        ui.separator();
        ui.selectable_value(
            &mut state.fm_spectrum_axis,
            SpectrumAxis::EnergyEv,
            "Energy (eV)",
        );
        ui.selectable_value(
            &mut state.fm_spectrum_axis,
            SpectrumAxis::TofMicroseconds,
            "TOF (\u{03bc}s)",
        );
    });
    ui.add_space(8.0);

    forward_model::fm_spectrum_panel(ui, state);
    ui.add_space(12.0);

    forward_model::fm_resolution_card(ui, state);
    ui.add_space(8.0);

    forward_model::fm_isotopes_card(ui, state);
}

fn studio_detectability(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(ui, "Detectability", "Trace element sensitivity analysis");

    // Sync button: copy main isotope config as matrix composition.
    // Guard by both detect and main fetch flags — copying mid-fetch
    // from main would create orphaned Fetching entries in matrix.
    ui.horizontal(|ui| {
        ui.add_enabled_ui(
            !state.is_fetching_detect_endf && !state.is_fetching_endf,
            |ui| {
                if ui.button("Copy matrix from Config").clicked() {
                    detectability::copy_config_to_detect_matrix(state);
                }
            },
        );
    });
    ui.add_space(4.0);

    let locked = state.is_fetching_detect_endf;
    detectability::detect_library_selector(ui, state, locked);
    ui.add_space(8.0);

    detectability::detect_matrix_card(ui, state, locked);
    ui.add_space(8.0);

    detectability::detect_trace_card(ui, state, locked);
    ui.add_space(8.0);

    detectability::detect_resolution_card(ui, state);
    ui.add_space(8.0);

    detectability::detect_advanced_config(ui, state);
    ui.add_space(8.0);

    detectability::detect_action_buttons(ui, state);
    ui.add_space(12.0);

    detectability::detect_results_panel(ui, state);
}

// ---------------------------------------------------------------------------
// Bottom dock (resizable panel with 4 tabs)
// ---------------------------------------------------------------------------

fn bottom_dock(ctx: &egui::Context, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::TopBottomPanel::bottom("studio_dock")
        .resizable(true)
        .default_height(200.0)
        .min_height(140.0)
        .max_height(400.0)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::symmetric(12, 8))
                .stroke(egui::Stroke::new(1.0, colors.border)),
        )
        .show(ctx, |ui| {
            let labels = &["Isotopes", "Residuals", "Provenance", "Export"];
            design::underline_tabs(ui, labels, &mut state.studio_dock_tab);
            ui.add_space(4.0);

            egui::ScrollArea::vertical().show(ui, |ui| match state.studio_dock_tab {
                0 => dock_isotopes(ui, state),
                1 => dock_residuals(ui, state),
                2 => dock_provenance(ui, state),
                3 => dock_export(ui, state),
                _ => {}
            });
        });
}

/// Isotopes table (read-only reference view; editing is in the left sidebar).
fn dock_isotopes(ui: &mut egui::Ui, state: &AppState) {
    if state.isotope_entries.is_empty() && state.isotope_groups.is_empty() {
        ui.label(
            egui::RichText::new(
                "No isotopes configured. Add isotopes in Guided \u{2192} Configure.",
            )
            .small()
            .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        return;
    }

    ui.label(
        egui::RichText::new(
            "Read-only reference \u{2014} edit densities in the sidebar Isotopes card.",
        )
        .small()
        .color(ThemeColors::from_ctx(ui.ctx()).fg3),
    );
    ui.add_space(4.0);

    egui::Grid::new("dock_isotope_grid")
        .num_columns(6)
        .spacing([12.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            // Header
            ui.label(egui::RichText::new("On").small().strong());
            ui.label(egui::RichText::new("Symbol").small().strong());
            ui.label(egui::RichText::new("Z").small().strong());
            ui.label(egui::RichText::new("A").small().strong());
            ui.label(egui::RichText::new("Density").small().strong());
            ui.label(egui::RichText::new("ENDF").small().strong());
            ui.end_row();

            for entry in &state.isotope_entries {
                // Enabled indicator
                let icon = if entry.enabled {
                    "\u{2611}"
                } else {
                    "\u{2610}"
                };
                ui.label(egui::RichText::new(icon).small());

                // Symbol with colored dot
                ui.horizontal(|ui| {
                    let dot_color = design::isotope_dot_color(&entry.symbol);
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 3.0, dot_color);
                    ui.label(egui::RichText::new(&entry.symbol).small());
                });

                ui.label(egui::RichText::new(entry.z.to_string()).small());
                ui.label(egui::RichText::new(entry.a.to_string()).small());
                ui.label(egui::RichText::new(format!("{:.4e}", entry.initial_density)).small());

                // ENDF status badge
                let (badge_text, badge_variant) = match entry.endf_status {
                    EndfStatus::Pending => ("Pending", design::BadgeVariant::Orange),
                    EndfStatus::Fetching => ("...", design::BadgeVariant::Orange),
                    EndfStatus::Loaded => ("OK", design::BadgeVariant::Green),
                    EndfStatus::Failed => ("ERR", design::BadgeVariant::Red),
                };
                design::badge(ui, badge_text, badge_variant);

                ui.end_row();
            }
        });

    // Isotope groups
    if !state.isotope_groups.is_empty() {
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Groups").small().strong());
        ui.add_space(4.0);

        egui::Grid::new("dock_group_grid")
            .num_columns(4)
            .spacing([12.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label(egui::RichText::new("On").small().strong());
                ui.label(egui::RichText::new("Name").small().strong());
                ui.label(egui::RichText::new("Density").small().strong());
                ui.label(egui::RichText::new("ENDF").small().strong());
                ui.end_row();

                for group in &state.isotope_groups {
                    let icon = if group.enabled {
                        "\u{2611}"
                    } else {
                        "\u{2610}"
                    };
                    ui.label(egui::RichText::new(icon).small());

                    ui.horizontal(|ui| {
                        let dot_color = design::isotope_dot_color(&group.name);
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 3.0, dot_color);
                        ui.label(egui::RichText::new(&group.name).small());
                    });

                    ui.label(egui::RichText::new(format!("{:.4e}", group.initial_density)).small());

                    let status = group.overall_status();
                    let (badge_text, badge_variant) = match status {
                        EndfStatus::Pending => ("Pending", design::BadgeVariant::Orange),
                        EndfStatus::Fetching => ("...", design::BadgeVariant::Orange),
                        EndfStatus::Loaded => ("OK", design::BadgeVariant::Green),
                        EndfStatus::Failed => ("ERR", design::BadgeVariant::Red),
                    };
                    design::badge(ui, badge_text, badge_variant);

                    ui.end_row();
                }
            });
    }
}

/// Residuals dock tab — shows residual plot + statistics for the selected pixel.
///
/// Uses `state.residuals_cache` to avoid rebuilding the `TransmissionFitModel`
/// and recomputing the forward model on every frame. The cache is keyed by
/// `(fit_result_gen, pixel, resolution_enabled, resolution_mode, flight_path_m, temperature_k)`.
///
/// Densities come from `pixel_fit_result` (single-pixel fit) when available,
/// otherwise from `spatial_result` density maps at the selected pixel.
fn dock_residuals(ui: &mut egui::Ui, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ui.ctx());

    let (py, px) = match state.selected_pixel {
        Some(p) => p,
        None => {
            ui.label(
                egui::RichText::new(
                    "Click a pixel in the density map to view residuals for that fit.",
                )
                .small()
                .color(colors.fg3),
            );
            return;
        }
    };

    // Extract densities, temperature, chi2, and convergence from either
    // pixel_fit_result (priority) or spatial_result.
    let pixel_info = extract_pixel_fit_info(state, py, px);
    let (densities, effective_temp, chi2_r, converged) = match pixel_info {
        Some(info) => info,
        None => {
            ui.label(
                egui::RichText::new("No fit data for this pixel.")
                    .small()
                    .color(colors.fg3),
            );
            return;
        }
    };

    if !converged {
        ui.label(
            egui::RichText::new("Fit did not converge \u{2014} no residuals to display.")
                .small()
                .color(colors.fg3),
        );
        return;
    }

    // Check if the cache is still valid.
    let cache_valid = state.residuals_cache.as_ref().is_some_and(|c| {
        c.fit_gen == state.fit_result_gen
            && c.pixel == (py, px)
            && c.resolution_enabled == state.resolution_enabled
            && c.resolution_mode == state.resolution_mode
            && c.flight_path_m == state.beamline.flight_path_m
            && c.temperature_k == effective_temp
    });

    if !cache_valid {
        let new_cache = build_residuals_cache(state, (py, px), &densities, effective_temp, chi2_r);
        state.residuals_cache = new_cache;
    }

    let cache = match &state.residuals_cache {
        Some(c) => c,
        None => {
            ui.label(
                egui::RichText::new(
                    "Could not compute residuals (missing data or isotope mismatch).",
                )
                .small()
                .color(colors.fg3),
            );
            return;
        }
    };

    // Stats row
    design::stat_row(
        ui,
        &[
            (&format!("{:.2e}", cache.rms), "RMS"),
            (&format!("{:.2e}", cache.max_abs), "Max |r|"),
            (&cache.n_points.to_string(), "Points"),
            (
                &if state.uncertainty_is_estimated {
                    format!("{:.4}~", cache.chi2_r)
                } else {
                    format!("{:.4}", cache.chi2_r)
                },
                "\u{03c7}\u{00b2}_r",
            ),
        ],
    );
    ui.add_space(4.0);

    // Residual plot
    let plot_height = ui.available_height().clamp(100.0, 250.0);
    let res_points: egui_plot::PlotPoints = cache.residuals.iter().map(|&(e, r)| [e, r]).collect();
    egui_plot::Plot::new("dock_residuals_plot")
        .height(plot_height)
        .x_axis_label("Energy (eV)")
        .y_axis_label("Residual")
        .show(ui, |plot_ui| {
            plot_ui.line(
                egui_plot::Line::new("Residual", res_points)
                    .color(egui::Color32::from_rgb(100, 160, 255)),
            );
            plot_ui.hline(
                egui_plot::HLine::new("zero", 0.0)
                    .color(egui::Color32::from_rgba_premultiplied(150, 150, 150, 80))
                    .style(egui_plot::LineStyle::dashed_loose()),
            );
        });
}

/// Extract per-pixel fit info from `pixel_fit_result` or `spatial_result`.
///
/// Returns `(densities, effective_temperature, chi2_r, converged)`.
fn extract_pixel_fit_info(
    state: &AppState,
    py: usize,
    px: usize,
) -> Option<(Vec<f64>, f64, f64, bool)> {
    // Prefer pixel_fit_result (from explicit Fit Pixel / Fit ROI).
    if let Some(ref result) = state.pixel_fit_result {
        let temp = result.temperature_k.unwrap_or(state.temperature_k);
        return Some((
            result.densities.clone(),
            temp,
            result.reduced_chi_squared,
            result.converged,
        ));
    }

    // Fall back to spatial_result maps.
    let sr = state.spatial_result.as_ref()?;
    if py >= sr.converged_map.nrows() || px >= sr.converged_map.ncols() {
        return None;
    }
    let n_isotopes = sr.density_maps.len();
    if n_isotopes == 0 {
        return None;
    }
    let densities: Vec<f64> = (0..n_isotopes)
        .map(|i| sr.density_maps[i][[py, px]])
        .collect();
    let temp = sr
        .temperature_map
        .as_ref()
        .map_or(state.temperature_k, |m| m[[py, px]]);
    let chi2_r = sr.chi_squared_map[[py, px]];
    let converged = sr.converged_map[[py, px]];
    Some((densities, temp, chi2_r, converged))
}

/// Build the residuals cache by constructing a `TransmissionFitModel`, evaluating
/// the forward model, and computing residuals against measured data.
///
/// Returns `None` if any prerequisite is missing (no energies, etc.).
fn build_residuals_cache(
    state: &AppState,
    pixel: (usize, usize),
    densities: &[f64],
    temperature_k: f64,
    chi2_r: f64,
) -> Option<crate::state::CachedResiduals> {
    let energies = state.energies.as_ref()?;
    let norm = state.normalized.as_ref()?;
    let (py, px) = pixel;

    let shape = norm.transmission.shape();
    let n_tof = shape[0];
    if py >= shape[1] || px >= shape[2] {
        return None;
    }

    // Collect resonance data for all enabled isotopes and groups.
    let (resonance_data, density_indices, density_ratios) =
        design::collect_all_resonance_data_with_mapping(state);
    if resonance_data.is_empty() {
        return None;
    }

    // Guard: density parameter count must match the mapping's expected count.
    let n_density_params = density_indices.iter().max().map_or(0, |&m| m + 1);
    if densities.len() < n_density_params {
        return None; // stale result with different isotope config
    }

    // Build instrument params from current resolution settings.
    let instrument = {
        use nereids_physics::transmission::InstrumentParams;
        design::build_resolution_function(
            state.resolution_enabled,
            &state.resolution_mode,
            state.beamline.flight_path_m,
        )
        .ok()?
        .map(|resolution| std::sync::Arc::new(InstrumentParams { resolution }))
    };

    let model = nereids_fitting::transmission_model::TransmissionFitModel::new(
        energies.clone(),
        resonance_data,
        temperature_k,
        instrument,
        (density_indices, density_ratios),
        None,
        None,
    )
    .ok()?;

    use nereids_fitting::lm::FitModel;
    let fitted = model.evaluate(densities).ok()?;
    let n_plot = n_tof.min(energies.len()).min(fitted.len());
    if n_plot == 0 {
        return None;
    }

    // Compute residuals and statistics.
    let mut residuals = Vec::with_capacity(n_plot);
    let mut sum_sq = 0.0;
    let mut max_abs = 0.0f64;
    for i in 0..n_plot {
        let meas = norm.transmission[[i, py, px]];
        let res = meas - fitted[i];
        if res.is_finite() {
            residuals.push((energies[i], res));
            sum_sq += res * res;
            max_abs = max_abs.max(res.abs());
        }
    }
    let n_points = residuals.len();
    let rms = if n_points > 0 {
        (sum_sq / n_points as f64).sqrt()
    } else {
        0.0
    };

    Some(crate::state::CachedResiduals {
        fit_gen: state.fit_result_gen,
        pixel,
        resolution_enabled: state.resolution_enabled,
        resolution_mode: state.resolution_mode.clone(),
        flight_path_m: state.beamline.flight_path_m,
        temperature_k,
        chi2_r,
        residuals,
        rms,
        max_abs,
        n_points,
    })
}

/// Provenance log (flat list, no collapsing header).
fn dock_provenance(ui: &mut egui::Ui, state: &AppState) {
    if state.provenance_log.is_empty() {
        ui.label(
            egui::RichText::new("No events recorded yet.")
                .small()
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        return;
    }

    for event in state.provenance_log.iter().rev() {
        ui.horizontal(|ui| {
            let ts = event.formatted_timestamp();
            let time_str = ts.get(11..19).unwrap_or("??:??:??");
            ui.label(egui::RichText::new(time_str).small().monospace());

            let kind_color = match event.kind {
                crate::state::ProvenanceEventKind::DataLoaded => crate::theme::semantic::YELLOW,
                crate::state::ProvenanceEventKind::ConfigChanged => crate::theme::semantic::ORANGE,
                crate::state::ProvenanceEventKind::Normalized => crate::theme::semantic::GREEN,
                crate::state::ProvenanceEventKind::AnalysisRun => crate::theme::semantic::ORANGE,
                crate::state::ProvenanceEventKind::Exported => crate::theme::semantic::GREEN,
                crate::state::ProvenanceEventKind::ProjectSaved => crate::theme::semantic::GREEN,
                crate::state::ProvenanceEventKind::ProjectLoaded => crate::theme::semantic::GREEN,
            };
            ui.label(
                egui::RichText::new(format!("{:?}", event.kind))
                    .small()
                    .color(kind_color),
            );
            ui.label(egui::RichText::new(&event.message).small());
        });
    }
}

/// Export panel — flat layout without card wrapper to fit in the dock.
fn dock_export(ui: &mut egui::Ui, state: &mut AppState) {
    use crate::state::ExportFormat;

    ui.horizontal(|ui| {
        ui.label("Format:");
        let current_label = state.export_format.label();
        egui::ComboBox::from_id_salt("dock_export_format")
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
        ui.label(egui::RichText::new(dir_label).monospace().small());

        if ui.button("Browse\u{2026}").clicked()
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
        result_widgets::run_export(state);
    }

    if let Some(ref status) = state.export_status {
        let color = if status.starts_with("Error") {
            crate::theme::semantic::RED
        } else {
            crate::theme::semantic::GREEN
        };
        ui.label(egui::RichText::new(status.as_str()).small().color(color));
    }
}

// ---------------------------------------------------------------------------
// Mini-inspector sidebar (right, Analysis tab only)
// ---------------------------------------------------------------------------

/// Left parameter sidebar — editable beamline, solver, isotope, and ROI
/// controls with dirty tracking + re-run button.
fn parameter_sidebar(ctx: &egui::Context, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::SidePanel::left("studio_params")
        .resizable(true)
        .default_width(240.0)
        .min_width(200.0)
        .max_width(360.0)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::symmetric(10, 8))
                .stroke(egui::Stroke::new(1.0, colors.border)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                rerun_card(ui, state);
                ui.add_space(6.0);

                beamline_card(ui, state);
                ui.add_space(6.0);

                solver_card(ui, state);
                ui.add_space(6.0);

                isotopes_card(ui, state);
                ui.add_space(6.0);

                result_widgets::pixel_inspector(ui, state);
                ui.add_space(6.0);

                if let Some(ref result) = state.spatial_result {
                    result_widgets::summary_card(ui, result, state.uncertainty_is_estimated);
                }
            });
        });
}

/// Re-run pipeline card with dirty indicator.
fn rerun_card(ui: &mut egui::Ui, state: &mut AppState) {
    use crate::state::GuidedStep;

    design::card_with_header(ui, "Pipeline", None, |ui| {
        if state.is_fitting {
            if let Some(ref fp) = state.fitting_progress {
                let done = fp.done();
                let total = fp.total();
                let frac = fp.fraction();
                design::progress_mini(ui, frac, &format!("{done}/{total} px"));
            } else {
                ui.spinner();
                ui.label("Running...");
            }
        } else if let Some(step) = state.dirty_from {
            ui.horizontal(|ui| {
                design::badge(
                    ui,
                    &format!("Dirty: {}", step.label()),
                    design::BadgeVariant::Orange,
                );
            });
            if ui
                .button(format!("\u{25b6} Re-run from {}", step.label()))
                .clicked()
            {
                match crate::pipeline::run_from_dirty(state) {
                    Ok(_) => {}
                    Err(e) => {
                        state.status_message = e;
                    }
                }
            }
        } else {
            ui.label(
                egui::RichText::new("Up to date")
                    .small()
                    .color(ThemeColors::from_ctx(ui.ctx()).fg3),
            );
        }
    });

    // Quick-access: re-run from Analyze (always available when not fitting)
    if !state.is_fitting
        && state.spatial_result.is_some()
        && ui
            .add_enabled(
                state.dirty_from.is_none(),
                egui::Button::new("\u{1f504} Re-run Spatial Map"),
            )
            .on_hover_text("Force re-run the spatial map with current parameters")
            .clicked()
    {
        state.mark_dirty(GuidedStep::Analyze);
        match crate::pipeline::run_from_dirty(state) {
            Ok(_) => {}
            Err(e) => {
                state.status_message = e;
            }
        }
    }
}

/// Editable beamline parameters card.
fn beamline_card(ui: &mut egui::Ui, state: &mut AppState) {
    use crate::state::GuidedStep;

    design::card_with_header(ui, "Beamline", None, |ui| {
        ui.horizontal(|ui| {
            ui.label("Flight path:");
            let prev = state.beamline.flight_path_m;
            ui.add(
                egui::DragValue::new(&mut state.beamline.flight_path_m)
                    .speed(0.01)
                    .range(0.1..=100.0)
                    .suffix(" m"),
            );
            if state.beamline.flight_path_m != prev {
                state.mark_dirty(GuidedStep::Normalize);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Delay:");
            let prev = state.beamline.delay_us;
            ui.add(
                egui::DragValue::new(&mut state.beamline.delay_us)
                    .speed(0.1)
                    .range(-1000.0..=1000.0)
                    .suffix(" \u{03bc}s"),
            );
            if state.beamline.delay_us != prev {
                state.mark_dirty(GuidedStep::Normalize);
            }
        });
    });
}

/// Editable solver parameters card.
fn solver_card(ui: &mut egui::Ui, state: &mut AppState) {
    use crate::state::{GuidedStep, SolverMethod};

    design::card_with_header(ui, "Solver", None, |ui| {
        ui.horizontal(|ui| {
            ui.label("Method:");
            let prev = state.solver_method;
            egui::ComboBox::from_id_salt("studio_solver_method")
                .selected_text(match state.solver_method {
                    SolverMethod::LevenbergMarquardt => "LM",
                    SolverMethod::PoissonKL => "Poisson KL",
                })
                .width(90.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.solver_method,
                        SolverMethod::LevenbergMarquardt,
                        "Levenberg-Marquardt",
                    );
                    ui.selectable_value(
                        &mut state.solver_method,
                        SolverMethod::PoissonKL,
                        "Poisson KL",
                    );
                });
            if state.solver_method != prev {
                state.mark_dirty(GuidedStep::Analyze);
            }
        });

        ui.horizontal(|ui| {
            ui.label("Max iter:");
            let prev = state.lm_config.max_iter;
            ui.add(egui::DragValue::new(&mut state.lm_config.max_iter).range(1..=10000));
            if state.lm_config.max_iter != prev {
                state.mark_dirty(GuidedStep::Analyze);
            }
        });

        ui.horizontal(|ui| {
            ui.label("Temp:");
            let prev = state.temperature_k;
            ui.add(
                egui::DragValue::new(&mut state.temperature_k)
                    .speed(1.0)
                    .range(1.0..=2000.0)
                    .suffix(" K"),
            );
            if state.temperature_k != prev {
                state.mark_dirty(GuidedStep::Analyze);
            }
        });

        let prev_fit_temp = state.fit_temperature;
        ui.checkbox(&mut state.fit_temperature, "Fit temperature");
        if state.fit_temperature != prev_fit_temp {
            state.mark_dirty(GuidedStep::Analyze);
        }
    });
}

/// Isotope list card (enable/disable + densities).
fn isotopes_card(ui: &mut egui::Ui, state: &mut AppState) {
    use crate::state::GuidedStep;

    design::card_with_header(ui, "Isotopes", None, |ui| {
        if state.isotope_entries.is_empty() && state.isotope_groups.is_empty() {
            ui.label(
                egui::RichText::new("No isotopes — configure in Guided mode.")
                    .small()
                    .color(ThemeColors::from_ctx(ui.ctx()).fg3),
            );
            return;
        }

        let locked = state.is_fetching_endf || state.is_fitting;

        for i in 0..state.isotope_entries.len() {
            ui.horizontal(|ui| {
                // Enable checkbox
                let prev_enabled = state.isotope_entries[i].enabled;
                ui.add_enabled(
                    !locked,
                    egui::Checkbox::without_text(&mut state.isotope_entries[i].enabled),
                );
                if state.isotope_entries[i].enabled != prev_enabled {
                    state.mark_dirty(GuidedStep::Analyze);
                }

                // Colored dot + symbol
                let dot_color = design::isotope_dot_color(&state.isotope_entries[i].symbol);
                let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 3.0, dot_color);
                ui.label(egui::RichText::new(&state.isotope_entries[i].symbol).small());

                // Editable density
                let prev_density = state.isotope_entries[i].initial_density;
                ui.add_enabled(
                    !locked,
                    egui::DragValue::new(&mut state.isotope_entries[i].initial_density)
                        .speed(1e-5)
                        .range(0.0..=1.0),
                );
                if state.isotope_entries[i].initial_density != prev_density {
                    state.mark_dirty(GuidedStep::Analyze);
                }

                // ENDF status
                let (badge_text, badge_variant) = match state.isotope_entries[i].endf_status {
                    crate::state::EndfStatus::Pending => ("?", design::BadgeVariant::Orange),
                    crate::state::EndfStatus::Fetching => ("..", design::BadgeVariant::Orange),
                    crate::state::EndfStatus::Loaded => ("\u{2713}", design::BadgeVariant::Green),
                    crate::state::EndfStatus::Failed => ("!", design::BadgeVariant::Red),
                };
                design::badge(ui, badge_text, badge_variant);
            });
        }

        // Isotope groups
        for i in 0..state.isotope_groups.len() {
            ui.horizontal(|ui| {
                // Enable checkbox
                let prev_enabled = state.isotope_groups[i].enabled;
                ui.add_enabled(
                    !locked,
                    egui::Checkbox::without_text(&mut state.isotope_groups[i].enabled),
                );
                if state.isotope_groups[i].enabled != prev_enabled {
                    state.mark_dirty(GuidedStep::Analyze);
                }

                // Colored dot + name
                let dot_color = design::isotope_dot_color(&state.isotope_groups[i].name);
                let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 3.0, dot_color);
                ui.label(egui::RichText::new(&state.isotope_groups[i].name).small());

                // Editable density
                let prev_density = state.isotope_groups[i].initial_density;
                ui.add_enabled(
                    !locked,
                    egui::DragValue::new(&mut state.isotope_groups[i].initial_density)
                        .speed(1e-5)
                        .range(0.0..=1.0),
                );
                if state.isotope_groups[i].initial_density != prev_density {
                    state.mark_dirty(GuidedStep::Analyze);
                }

                // ENDF status
                let status = state.isotope_groups[i].overall_status();
                let (badge_text, badge_variant) = match status {
                    crate::state::EndfStatus::Pending => ("?", design::BadgeVariant::Orange),
                    crate::state::EndfStatus::Fetching => ("..", design::BadgeVariant::Orange),
                    crate::state::EndfStatus::Loaded => ("\u{2713}", design::BadgeVariant::Green),
                    crate::state::EndfStatus::Failed => ("!", design::BadgeVariant::Red),
                };
                design::badge(ui, badge_text, badge_variant);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// No-results placeholder
// ---------------------------------------------------------------------------

fn no_results_placeholder(ui: &mut egui::Ui) {
    let colors = ThemeColors::from_ctx(ui.ctx());
    ui.centered_and_justified(|ui| {
        ui.label(
            egui::RichText::new(
                "No results yet \u{2014} run spatial mapping in Guided mode,\nor use the Forward Model and Detectability tabs.",
            )
            .heading()
            .color(colors.fg3),
        );
    });
}
