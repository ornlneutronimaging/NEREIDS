//! Studio mode — "Final Cut"-style IDE workspace with document tabs,
//! resizable bottom dock, and shared tool embedding.
//!
//! Layout:
//!   1. Bottom dock (resizable, 4 tabs: Isotopes/Residuals/Provenance/Export)
//!   2. Mini-inspector sidebar (right, 220px, Analysis tab only)
//!   3. Central panel with document tabs (Analysis / Forward Model / Detectability)

use std::sync::Arc;

use crate::guided::{detectability, forward_model, result_widgets};
use crate::state::{AppState, Colormap, EndfStatus, SpectrumAxis, StudioDocTab};
use crate::theme::ThemeColors;
use crate::widgets::design;
use crate::widgets::image_view::show_colormapped_image;
use egui_plot::{Line, Plot, PlotPoints};

/// Render the Studio mode content.
pub fn studio_content(ui: &mut egui::Ui, state: &mut AppState) {
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
        bottom_dock(ui, state);
    }

    // 2. Parameter sidebar (left, Analysis tab only)
    if state.studio_doc_tab == StudioDocTab::Analysis {
        parameter_sidebar(ui, state);
    }

    // 3. Central panel: doc tab bar + routed content
    let colors = ThemeColors::from_ctx(ui.ctx());
    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::same(12)),
        )
        .show_inside(ui, |ui| {
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
            tile_idx,
            &label,
            &mut state.tile_display,
            &mut state.file_dialogs,
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

    // Fit curve (if available), evaluated on the grid the fit ran on.
    let fit_line = state.pixel_fit_result.as_ref().and_then(|result| {
        let energies = state.energies.as_ref()?;
        let (all_rd, density_indices, density_ratios) =
            design::collect_all_resonance_data_with_mapping(state);
        let instrument = design::build_resolution_function(
            state.resolution_enabled,
            &state.resolution_mode,
            state.beamline.flight_path_m,
        )
        .ok()
        .flatten()
        .map(|r| Arc::new(nereids_physics::transmission::InstrumentParams { resolution: r }));
        let range = crate::guided::analyze::fit_grid_range(
            energies,
            state.fit_energy_range,
            instrument.as_ref().map(|i| &i.resolution),
        )
        .ok()?;
        let plot = range.start.min(n_plot)..range.end.min(n_plot);
        design::build_fit_line(&design::FitLineParams {
            result,
            resonance_data: &all_rd,
            density_indices: &density_indices,
            density_ratios: &density_ratios,
            energies: &energies[range],
            temperature_k: state.temperature_k,
            x_values: &x_values[plot.clone()],
            n_plot: plot.len(),
            instrument,
            y_multiplier: None,
        })
    });
    let (fit_line, route_mismatch) = match fit_line {
        Some(fit) => (fit.line, fit.warning),
        None => (None, None),
    };
    if let Some(message) = &route_mismatch {
        ui.colored_label(crate::theme::semantic::ORANGE, message);
    }

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
            let gof_label = if result.deviance_per_dof.is_some() {
                "D/dof"
            } else {
                "chi2_r"
            };
            if state.uncertainty_is_estimated {
                ui.label(
                    egui::RichText::new(format!(
                        "{} = {:.4} (approx.)",
                        gof_label, result.reduced_chi_squared
                    ))
                    .color(crate::theme::semantic::ORANGE),
                );
            } else {
                ui.label(format!("{} = {:.4}", gof_label, result.reduced_chi_squared));
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

        // Group density results — only display when the result length matches
        // the current config (guards against stale results after config change).
        let enabled_individual_count = state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .count();
        let enabled_group_count = state
            .isotope_groups
            .iter()
            .filter(|g| g.enabled && g.overall_status() == EndfStatus::Loaded)
            .count();
        if enabled_individual_count + enabled_group_count == result.densities.len() {
            let mut group_idx = enabled_individual_count;
            for group in &state.isotope_groups {
                if group.enabled && group.overall_status() == EndfStatus::Loaded {
                    if group_idx < result.densities.len() {
                        let dot_color = design::isotope_dot_color(&group.name);
                        ui.horizontal(|ui| {
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
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

fn bottom_dock(ui: &mut egui::Ui, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ui.ctx());
    egui::Panel::bottom("studio_dock")
        .resizable(true)
        .default_size(200.0)
        .min_size(140.0)
        .max_size(400.0)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::symmetric(12, 8))
                .stroke(egui::Stroke::new(1.0_f32, colors.border)),
        )
        .show_inside(ui, |ui| {
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

    // The whole fit result for this pixel — from `pixel_fit_result` when a
    // single-pixel fit is loaded, else assembled from the spatial maps.
    // The residual needs every piece of it, not just the densities: the
    // fitted Anorm, background and baseline are part of what the fit
    // predicted, and the fitted energy scale decides where the physics was
    // evaluated.
    let result = match crate::guided::analyze::selected_pixel_fit_result_for_overlay(state, py, px)
    {
        Some(result) => result,
        None => {
            ui.label(
                egui::RichText::new("No converged fit for this pixel.")
                    .small()
                    .color(colors.fg3),
            );
            return;
        }
    };
    let effective_temp = result.temperature_k.unwrap_or(state.temperature_k);

    if !result.converged {
        ui.label(
            egui::RichText::new("Fit did not converge \u{2014} no residuals to display.")
                .small()
                .color(colors.fg3),
        );
        return;
    }

    // Take the cache out and hand it over, putting it back at the end,
    // rather than rebuilding and then re-reading the field: re-reading needs
    // a `None` arm that cannot be reached and would render an empty dock if
    // it ever were.
    let held = state.residuals_cache.take();
    let cache = match residuals_to_render(state, &result, effective_temp, held) {
        Ok(cache) => cache,
        Err(reason) => {
            // Say which refusal fired: a degenerate energy scale, a window
            // holding no measured bin and a wholly non-finite one are not
            // "missing data", and the user can only act on the cause that
            // actually happened.
            state.residuals_cache = None;
            ui.label(
                egui::RichText::new(reason.message())
                    .small()
                    .color(colors.fg3),
            );
            return;
        }
    };

    let view = dock_residual_view(&cache, state.uncertainty_is_estimated);
    if let Some(message) = &view.warning {
        ui.colored_label(crate::theme::semantic::ORANGE, message);
    }
    let stats: Vec<(&str, &str)> = view
        .stats
        .iter()
        .map(|(value, label)| (value.as_str(), *label))
        .collect();
    design::stat_row(ui, &stats);
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
    state.residuals_cache = Some(cache);
}

/// Everything the residual dock shows out of a computed cache, before any
/// of it is painted.
///
/// Separated from the painting so that WHAT is shown is testable while only
/// the egui calls that show it are not. The warning in particular is not
/// decoration: it says the residuals were taken against a model that is
/// not, or is not known to be, the fitted one.
#[derive(Debug, PartialEq, Eq)]
struct DockResidualView {
    /// The route warning carried on the cache, when there is one.
    warning: Option<String>,
    /// `(value, label)` for each statistic, already formatted.
    stats: Vec<(String, &'static str)>,
}

/// What the dock renders for `cache`.
///
/// `uncertainty_is_estimated` appends a tilde to χ²_r: the fit's σ was
/// estimated from the data rather than measured, so the number is a
/// goodness-of-fit indicator and not a χ² against known errors, and the
/// mark is the only thing that says so.
fn dock_residual_view(
    cache: &crate::state::CachedResiduals,
    uncertainty_is_estimated: bool,
) -> DockResidualView {
    DockResidualView {
        warning: cache.warning.clone(),
        stats: vec![
            (format!("{:.2e}", cache.rms), "RMS"),
            (format!("{:.2e}", cache.max_abs), "Max |r|"),
            (cache.n_points.to_string(), "Points"),
            (
                if uncertainty_is_estimated {
                    format!("{:.4}~", cache.chi2_r)
                } else {
                    format!("{:.4}", cache.chi2_r)
                },
                "\u{03c7}\u{00b2}_r",
            ),
        ],
    }
}

/// The residuals to render this frame: `held` when it still describes this
/// state, otherwise a freshly built cache.
///
/// The reuse decision is the difference between showing this fit's
/// residuals and showing the previous state's, so it lives here rather than
/// in the dock where no test can reach it. `held` is only good while every
/// input [`build_residuals_cache`] reads is unchanged, and the fit
/// generation alone does not say that: the selected pixel, the resolution
/// settings, the flight path and the temperature the model is evaluated at
/// all move without it.
///
/// # Errors
/// The [`design::ResidualsUnavailable`] that fired while rebuilding.
fn residuals_to_render(
    state: &AppState,
    result: &nereids_pipeline::pipeline::SpectrumFitResult,
    temperature_k: f64,
    held: Option<crate::state::CachedResiduals>,
) -> Result<crate::state::CachedResiduals, design::ResidualsUnavailable> {
    if let Some(cache) = held
        && cache.fit_gen == state.fit_result_gen
        && Some(cache.pixel) == state.selected_pixel
        && cache.resolution_enabled == state.resolution_enabled
        && cache.resolution_mode == state.resolution_mode
        && cache.flight_path_m == state.beamline.flight_path_m
        && cache.temperature_k == temperature_k
    {
        return Ok(cache);
    }
    build_residuals_cache(state, result, temperature_k)
}

/// Which bins of the loaded grid the dock's residuals are formed on, and
/// whose spectrum they are taken from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResidualWindow {
    /// The fit window's index range into the LOADED energy grid.
    range: std::ops::Range<usize>,
    /// The cube pixel, in the cube's own `[t, row, column]` order.
    pixel: (usize, usize),
}

/// Choose the window and the pixel the dock's residuals belong to.
///
/// These two are decisions about WHAT is computed, and they are the two a
/// caller can get wrong without the compiler noticing, so they are made
/// once here where a test can call them and then travel together into
/// [`fit_window_measurement`]:
///
/// - the window is the fit's own slice of the loaded grid
///   ([`crate::guided::analyze::fit_grid_range`]), not `0..range.end`,
///   which pairs every residual with a different energy and a different
///   measured value for any fit that does not start at bin 0;
/// - the pixel keeps the cube's `(row, column)` order — the transposed
///   pair reads a different pixel's spectrum and reports its residuals as
///   this one's.
///
/// # Errors
/// [`design::ResidualsUnavailable::MissingData`] when no energy grid is
/// loaded or no pixel is selected;
/// [`design::ResidualsUnavailable::ModelUnavailable`] when the fit-energy
/// range selects nothing on this grid.
fn residual_window(
    state: &AppState,
    resolution: Option<&nereids_physics::resolution::ResolutionFunction>,
) -> Result<ResidualWindow, design::ResidualsUnavailable> {
    use design::ResidualsUnavailable::{MissingData, ModelUnavailable};

    let energies = state.energies.as_ref().ok_or(MissingData)?;
    let pixel = state.selected_pixel.ok_or(MissingData)?;
    let range =
        crate::guided::analyze::fit_grid_range(energies, state.fit_energy_range, resolution)
            .map_err(|_| ModelUnavailable)?;
    Ok(ResidualWindow { range, pixel })
}

/// Gather what [`design::residuals_for_fit`] needs out of the application
/// state, call it, and key the answer to the state it was computed from.
///
/// Everything physical — which grid the physics is evaluated on, what curve
/// the measurement is taken against, and which refusals are possible —
/// lives in `design::residuals_for_fit`, where it is tested directly. This
/// function is the state plumbing around that one call: the loaded grid and
/// cube, the fit window, the enabled isotopes, the resolution settings, and
/// the cache key.
///
/// # Errors
/// The [`design::ResidualsUnavailable`] that fired, so the dock can name
/// the cause instead of guessing one.
fn build_residuals_cache(
    state: &AppState,
    result: &nereids_pipeline::pipeline::SpectrumFitResult,
    temperature_k: f64,
) -> Result<crate::state::CachedResiduals, design::ResidualsUnavailable> {
    use design::ResidualsUnavailable::{MissingData, ModelUnavailable};

    let energies = state.energies.as_ref().ok_or(MissingData)?;
    let norm = state.normalized.as_ref().ok_or(MissingData)?;

    // Collect resonance data for all enabled isotopes and groups.
    let (resonance_data, density_indices, density_ratios) =
        design::collect_all_resonance_data_with_mapping(state);
    if resonance_data.is_empty() {
        return Err(MissingData);
    }

    // Guard: density parameter count must match the mapping's expected count.
    let n_density_params = density_indices.iter().max().map_or(0, |&m| m + 1);
    if result.densities.len() != n_density_params {
        return Err(MissingData); // stale result with different isotope config
    }

    // Build instrument params from current resolution settings.
    let instrument = {
        use nereids_physics::transmission::InstrumentParams;
        design::build_resolution_function(
            state.resolution_enabled,
            &state.resolution_mode,
            state.beamline.flight_path_m,
        )
        .map_err(|_| ModelUnavailable)?
        .map(|resolution| std::sync::Arc::new(InstrumentParams { resolution }))
    };

    // Residuals are formed on the grid the fit ran on (the fit-energy
    // slice), so the model is gated the way the fit gated: a
    // free-temperature fit gated its routes at the fit's upper bound.
    let window = residual_window(state, instrument.as_ref().map(|i| &i.resolution))?;
    let (py, px) = window.pixel;
    let shape = norm.transmission.shape();
    if py >= shape[1] || px >= shape[2] {
        return Err(MissingData);
    }
    let (fit_energies, measured) = fit_window_measurement(energies, &norm.transmission, &window);

    let design::DockResiduals { stats, warning } =
        design::residuals_for_fit(design::ResidualsRequest {
            result,
            nominal_energies: fit_energies,
            measured: &measured,
            resonance_data,
            density_mapping: (density_indices, density_ratios),
            temperature_k,
            instrument,
        })?;
    let n_points = stats.residuals.len();

    Ok(crate::state::CachedResiduals {
        fit_gen: state.fit_result_gen,
        pixel: window.pixel,
        resolution_enabled: state.resolution_enabled,
        resolution_mode: state.resolution_mode.clone(),
        flight_path_m: state.beamline.flight_path_m,
        temperature_k,
        chi2_r: result.reduced_chi_squared,
        residuals: stats.residuals,
        rms: stats.rms,
        max_abs: stats.max_abs,
        n_points,
        warning,
    })
}

/// The fit window's nominal energies, and the measurement on exactly those
/// bins.
///
/// Two choices live here, and both decide what the dock computes rather
/// than how it is wired:
///
/// - The energies are the fit window's SLICE of the loaded grid (see
///   [`crate::guided::analyze::fit_grid_range`]), because that is the grid
///   the fit ran on — it is the residual plot's x-axis, the grid the
///   composed background and baseline are evaluated on, and the grid the
///   Doppler route gate reads. Handing over the whole loaded grid instead
///   pairs every residual with the wrong energy and the wrong measured
///   value for any window that does not start at bin 0.
/// - The measurement is read at `range.start + i`, the cube's
///   time-of-flight bin belonging to window bin `i`. Reading it at `i`
///   would subtract the fitted curve from the first bins of the
///   acquisition rather than from the fitted ones.
///
/// The window and the pixel arrive together in a [`ResidualWindow`] built by
/// [`residual_window`], so neither can be picked ad hoc at the call site.
///
/// The measurement stops where the cube's time-of-flight axis stops, so a
/// loaded energy grid that outruns the cube yields a shorter — possibly
/// empty — measurement instead of an out-of-bounds index.
fn fit_window_measurement<'a>(
    energies: &'a [f64],
    transmission: &ndarray::Array3<f64>,
    window: &ResidualWindow,
) -> (&'a [f64], Vec<f64>) {
    let (py, px) = window.pixel;
    let range = &window.range;
    let fit_energies = &energies[range.clone()];
    let n_measured = transmission.shape()[0]
        .saturating_sub(range.start)
        .min(fit_energies.len());
    let measured = (0..n_measured)
        .map(|i| transmission[[range.start + i, py, px]])
        .collect();
    (fit_energies, measured)
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

        if ui.button("Browse\u{2026}").clicked() {
            state.file_dialogs.pick_folder(
                crate::file_dialog::DialogIntent::ExportDirectory,
                Default::default(),
            );
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
fn parameter_sidebar(ui: &mut egui::Ui, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ui.ctx());
    egui::Panel::left("studio_params")
        .resizable(true)
        .default_size(240.0)
        .min_size(200.0)
        .max_size(360.0)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::symmetric(10, 8))
                .stroke(egui::Stroke::new(1.0_f32, colors.border)),
        )
        .show_inside(ui, |ui| {
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
                // Same treatment as the other resolution-change paths: the
                // outputs (and any in-flight spatial worker) belong to the
                // previous flight path.
                state.invalidate_analysis_outputs();
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
            // Same availability rule as the Analyze panel: KL is a raw-count
            // likelihood, and without both count arms a KL request would land
            // on the rejected transmission+Poisson route.
            let counts_available = crate::guided::analyze::display_as_counts(state);
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
                    ui.add_enabled_ui(counts_available, |ui| {
                        ui.selectable_value(
                            &mut state.solver_method,
                            SolverMethod::PoissonKL,
                            "Poisson KL (raw counts)",
                        );
                    });
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
            // The widget writes to a local value and the state change goes
            // through `set_temperature_k`, so the edit and the invalidation
            // it implies cannot come apart: this is the temperature both
            // redraw paths fall back to for every fit that did not fit one,
            // and editing the field here without dropping the stored fit
            // leaves the dock re-computing at the new temperature and
            // reporting the answer as the fit's.
            let mut temperature_k = state.temperature_k;
            ui.add(
                egui::DragValue::new(&mut temperature_k)
                    .speed(1.0)
                    .range(1.0..=2000.0)
                    .suffix(" K"),
            );
            state.set_temperature_k(temperature_k);
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
                // Enable checkbox. The widget writes to a local flag and the
                // state change goes through `set_isotope_enabled`, so the
                // flip and the invalidation it implies cannot come apart:
                // flipping the field here and forgetting the invalidation
                // leaves the dock on the previous isotope set's numbers.
                let mut enabled = state.isotope_entries[i].enabled;
                ui.add_enabled(!locked, egui::Checkbox::without_text(&mut enabled));
                state.set_isotope_enabled(i, enabled);

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
                // Enable checkbox — see the per-isotope checkbox above for
                // why the flip goes through the state method.
                let mut enabled = state.isotope_groups[i].enabled;
                ui.add_enabled(!locked, egui::Checkbox::without_text(&mut enabled));
                state.set_isotope_group_enabled(i, enabled);

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
                    state.pixel_fit_result = None;
                    state.residuals_cache = None;
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

#[cfg(test)]
mod tests {
    use super::{
        AppState, DockResidualView, ResidualWindow, dock_residual_view, fit_window_measurement,
        residual_window, residuals_to_render,
    };
    use crate::state::CachedResiduals;
    use crate::widgets::design::ResidualsUnavailable;
    use ndarray::Array3;

    fn window(range: std::ops::Range<usize>, pixel: (usize, usize)) -> ResidualWindow {
        ResidualWindow { range, pixel }
    }

    /// A cube whose value at `[t, y, x]` is `t*100 + y*10 + x` — a marker
    /// per (time bin, pixel) rather than a physical transmission, so reading
    /// the wrong bin or the wrong pixel is visible in the value itself.
    fn marked_cube(n_tof: usize) -> Array3<f64> {
        Array3::from_shape_fn((n_tof, 2, 2), |(t, y, x)| (t * 100 + y * 10 + x) as f64)
    }

    /// The dock's residuals are formed on the FIT WINDOW, and each one is
    /// taken against the cube's time-of-flight bin that window bin belongs
    /// to. A window starting at bin 0 cannot tell either choice apart from
    /// its opposite, so this one starts at bin 3: handing over the whole
    /// loaded grid, or reading the cube from bin 0, each pairs every
    /// residual with a different energy and a different measurement.
    #[test]
    fn the_residual_window_is_the_fit_window_and_its_own_measured_bins() {
        let energies: Vec<f64> = (0..10).map(|i| 6.0 + i as f64 * 0.01).collect();
        let cube = marked_cube(10);

        let (bins, measured) = fit_window_measurement(&energies, &cube, &window(3..7, (1, 0)));

        assert_eq!(bins, &energies[3..7]);
        assert_eq!(measured, vec![310.0, 410.0, 510.0, 610.0]);
        // Non-vacuity: this window is neither the whole grid nor the
        // cube's leading bins, so both mutations change what is returned.
        assert!(bins.len() < energies.len());
        assert_ne!(measured[0], cube[[0, 1, 0]]);
    }

    /// A loaded energy grid can outrun the cube's time-of-flight axis. The
    /// measurement then stops where the cube stops — and a window lying
    /// entirely past it yields nothing, which `residuals_for_fit` reports as
    /// the empty-window refusal rather than as non-finite numbers.
    #[test]
    fn a_window_past_the_cube_truncates_instead_of_indexing_past_it() {
        let energies: Vec<f64> = (0..14).map(|i| 6.0 + i as f64 * 0.01).collect();
        let cube = marked_cube(10);

        let (bins, measured) = fit_window_measurement(&energies, &cube, &window(8..14, (1, 1)));
        assert_eq!(bins.len(), 6);
        assert_eq!(measured, vec![811.0, 911.0]);

        let (bins, measured) = fit_window_measurement(&energies, &cube, &window(10..14, (1, 1)));
        assert_eq!(bins.len(), 4);
        assert!(measured.is_empty());
    }

    /// The window and the pixel are the two arguments the dock could hand
    /// over wrong without the compiler noticing, so they are chosen here.
    /// The fit-energy range picks a window that starts well inside the
    /// grid, which is what tells the fit's own slice apart from `0..end`;
    /// the pixel is `(row, column)` and must come back in that order, not
    /// transposed.
    #[test]
    fn the_dock_takes_the_fit_window_and_the_selected_pixel() {
        let energies: Vec<f64> = (0..100).map(|i| 6.0 + i as f64 * 0.01).collect();
        let state = AppState {
            energies: Some(energies.clone()),
            fit_energy_range: Some((6.30, 6.60)),
            selected_pixel: Some((7, 3)),
            ..AppState::default()
        };

        let chosen = residual_window(&state, None).expect("a loaded grid and a selected pixel");
        assert!(
            chosen.range.start > 0 && chosen.range.end < energies.len(),
            "a window that starts at bin 0 cannot tell the fit slice from 0..end: {chosen:?}"
        );
        assert!(energies[chosen.range.start] >= 6.30);
        assert!(energies[chosen.range.end - 1] <= 6.60);
        assert_eq!(
            chosen.pixel,
            (7, 3),
            "the cube is indexed [t, row, column]; the transposed pair is another pixel"
        );

        // With no fit-energy range the window is the whole grid — the one
        // case where `0..end` happens to be right.
        let unrestricted = AppState {
            fit_energy_range: None,
            ..AppState {
                energies: Some(energies.clone()),
                selected_pixel: Some((7, 3)),
                ..AppState::default()
            }
        };
        assert_eq!(
            residual_window(&unrestricted, None),
            Ok(window(0..energies.len(), (7, 3)))
        );

        // The prerequisites are named apart: nothing loaded and nothing
        // selected are both "missing data", a range that selects nothing on
        // this grid is a model failure.
        let no_grid = AppState {
            selected_pixel: Some((7, 3)),
            ..AppState::default()
        };
        assert_eq!(
            residual_window(&no_grid, None),
            Err(ResidualsUnavailable::MissingData)
        );
        let no_pixel = AppState {
            energies: Some(energies.clone()),
            ..AppState::default()
        };
        assert_eq!(
            residual_window(&no_pixel, None),
            Err(ResidualsUnavailable::MissingData)
        );
        let empty_range = AppState {
            fit_energy_range: Some((100.0, 200.0)),
            ..state
        };
        assert_eq!(
            residual_window(&empty_range, None),
            Err(ResidualsUnavailable::ModelUnavailable)
        );
    }

    /// A cache whose every key still matches is handed straight back; one
    /// whose key has moved is not, whatever else is true of it. The state
    /// here cannot build anything (no energy grid), so a rebuild is visible
    /// as the refusal — which is exactly the difference between showing
    /// this fit's residuals and showing the previous state's.
    #[test]
    fn a_cache_is_reused_only_while_it_describes_this_state() {
        let cache = CachedResiduals {
            fit_gen: 4,
            pixel: (2, 5),
            resolution_enabled: false,
            resolution_mode: crate::state::ResolutionMode::Gaussian {
                delta_t_us: 0.0,
                delta_l_m: 0.0,
            },
            flight_path_m: 25.0,
            temperature_k: 293.6,
            chi2_r: 1.0,
            // A value nothing could recompute, so a returned cache is
            // provably the one that went in.
            residuals: vec![(6.0, 0.25)],
            rms: 12345.0,
            max_abs: 12345.0,
            n_points: 1,
            warning: None,
        };
        // The state the cache was computed from. `energies: None` (the
        // default) means nothing can be rebuilt, so a rebuild shows up as
        // the refusal and reuse shows up as the sentinel RMS.
        let matching = || {
            let mut state = AppState {
                fit_result_gen: 4,
                selected_pixel: Some((2, 5)),
                resolution_enabled: false,
                resolution_mode: cache.resolution_mode.clone(),
                ..AppState::default()
            };
            state.beamline.flight_path_m = 25.0;
            state
        };
        let result = fit_result_shell();

        let reused = residuals_to_render(&matching(), &result, 293.6, Some(cache.clone()))
            .expect("the key still describes this state");
        assert_eq!(reused.rms, 12345.0, "the held cache is what came back");

        // Each key field ON ITS OWN forces the rebuild, which this state
        // cannot do — so the stale cache is refused rather than shown.
        let refused = |state: &AppState, temperature_k: f64| {
            residuals_to_render(state, &result, temperature_k, Some(cache.clone())).unwrap_err()
        };
        assert_eq!(
            refused(&matching(), 500.0),
            ResidualsUnavailable::MissingData,
            "the temperature the model is evaluated at moved"
        );

        let mut moved = matching();
        moved.selected_pixel = Some((5, 2));
        assert_eq!(refused(&moved, 293.6), ResidualsUnavailable::MissingData);

        let mut moved = matching();
        moved.fit_result_gen = 5;
        assert_eq!(refused(&moved, 293.6), ResidualsUnavailable::MissingData);

        let mut moved = matching();
        moved.resolution_enabled = true;
        assert_eq!(refused(&moved, 293.6), ResidualsUnavailable::MissingData);

        let mut moved = matching();
        moved.beamline.flight_path_m = 30.0;
        assert_eq!(refused(&moved, 293.6), ResidualsUnavailable::MissingData);

        let mut moved = matching();
        moved.resolution_mode = crate::state::ResolutionMode::Gaussian {
            delta_t_us: 1.0,
            delta_l_m: 0.0,
        };
        assert_eq!(refused(&moved, 293.6), ResidualsUnavailable::MissingData);

        // And with nothing held there is nothing to reuse.
        assert_eq!(
            residuals_to_render(&matching(), &result, 293.6, None).unwrap_err(),
            ResidualsUnavailable::MissingData
        );
    }

    /// What the dock puts on screen out of a cache: the four statistics as
    /// they are formatted and labelled, and the route warning — which is
    /// not decoration, it says these residuals were taken against a model
    /// that is not, or is not known to be, the fitted one. The tilde on
    /// χ²_r is the only mark saying the σ behind it was estimated from the
    /// data rather than measured.
    ///
    /// The egui calls that paint this view are not covered: no UI harness
    /// exists in this crate, so deleting a paint call still compiles and
    /// still passes.
    #[test]
    fn the_dock_shows_the_statistics_and_the_warning_it_was_given() {
        let cache = CachedResiduals {
            fit_gen: 0,
            pixel: (0, 0),
            resolution_enabled: false,
            resolution_mode: crate::state::ResolutionMode::Gaussian {
                delta_t_us: 0.0,
                delta_l_m: 0.0,
            },
            flight_path_m: 25.0,
            temperature_k: 293.6,
            chi2_r: 1.25,
            residuals: vec![(6.0, 0.01), (6.1, -0.02)],
            rms: 0.0158,
            max_abs: 0.02,
            n_points: 2,
            warning: Some("Fit overlay is unchecked".to_string()),
        };

        let view = dock_residual_view(&cache, false);
        assert_eq!(
            view,
            DockResidualView {
                warning: Some("Fit overlay is unchecked".to_string()),
                stats: vec![
                    ("1.58e-2".to_string(), "RMS"),
                    ("2.00e-2".to_string(), "Max |r|"),
                    ("2".to_string(), "Points"),
                    ("1.2500".to_string(), "\u{03c7}\u{00b2}_r"),
                ],
            }
        );

        let estimated = dock_residual_view(&cache, true);
        assert_eq!(estimated.stats[3].0, "1.2500~");

        let silent = CachedResiduals {
            warning: None,
            ..cache
        };
        assert_eq!(dock_residual_view(&silent, false).warning, None);
    }

    /// A converged single-pixel result with nothing composed on top.
    fn fit_result_shell() -> nereids_pipeline::pipeline::SpectrumFitResult {
        nereids_pipeline::pipeline::SpectrumFitResult {
            densities: Vec::new(),
            uncertainties: None,
            reduced_chi_squared: 1.0,
            converged: true,
            iterations: 1,
            temperature_k: None,
            temperature_k_unc: None,
            anorm: 1.0,
            background: [0.0; 3],
            back_d: None,
            back_f: None,
            t0_us: None,
            l_scale: None,
            energy_scale_flight_path_m: None,
            deviance_per_dof: None,
            baseline: None,
            baseline_e_ref_ev: None,
            warnings: Vec::new(),
            doppler_routes: None,
        }
    }
}
