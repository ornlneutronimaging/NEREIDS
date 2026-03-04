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

/// Format a float without unnecessary trailing zeros; non-finite → "—".
fn format_compact_f64(value: f64) -> String {
    if !value.is_finite() {
        return "\u{2014}".to_owned();
    }
    if value.fract().abs() < 1e-9 {
        return format!("{:.0}", value);
    }
    let s = format!("{:.3}", value);
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// Render the Studio mode content.
pub fn studio_content(ctx: &egui::Context, state: &mut AppState) {
    let has_results = state.spatial_result.is_some();

    // Ensure tile_display is populated when results exist
    if let Some(ref r) = state.spatial_result
        && state.tile_display.len() < r.density_maps.len() + 1
    {
        state.init_tile_display(r.density_maps.len());
    }

    // 1. Bottom dock (before side panel and central panel — egui ordering)
    if state.studio_show_dock {
        bottom_dock(ctx, state);
    }

    // 2. Mini-inspector sidebar (right, only for Analysis tab with results)
    if state.studio_doc_tab == StudioDocTab::Analysis && has_results {
        mini_inspector(ctx, state);
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
            let symbols: Vec<String> = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .map(|e| e.symbol.clone())
                .collect();
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

    // Clamp isotope index
    if state.studio_analysis_isotope >= n_density {
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
    // Header: isotope selector (colormap/save controls are in tile_toolbelt below)
    ui.horizontal(|ui| {
        ui.label("Isotope:");
        egui::ComboBox::from_id_salt("analysis_isotope_sel")
            .selected_text(
                symbols
                    .get(state.studio_analysis_isotope)
                    .map_or("—", |s| s.as_str()),
            )
            .show_ui(ui, |ui| {
                for (i, sym) in symbols.iter().enumerate().take(n_density) {
                    ui.selectable_value(&mut state.studio_analysis_isotope, i, sym);
                }
            });
    });
    ui.add_space(4.0);

    // Density map image
    let tile_idx = state.studio_analysis_isotope;
    let colormap = state
        .tile_display
        .get(tile_idx)
        .map_or(Colormap::Viridis, |t| t.colormap);
    let show_bar = state
        .tile_display
        .get(tile_idx)
        .is_some_and(|t| t.show_colorbar);

    if let Some(data) = density_maps.get(tile_idx) {
        if show_bar {
            ui.horizontal(|ui| {
                if let Some((y, x)) =
                    show_colormapped_image(ui, data, "studio_analysis_map", colormap)
                {
                    state.selected_pixel = Some((y, x));
                    state.pixel_fit_result = None;
                }
                result_widgets::draw_colorbar(ui, data, colormap);
            });
        } else if let Some((y, x)) =
            show_colormapped_image(ui, data, "studio_analysis_map", colormap)
        {
            state.selected_pixel = Some((y, x));
            state.pixel_fit_result = None;
        }

        // Toolbelt
        let label = symbols
            .get(tile_idx)
            .map_or("unknown", |s| s.as_str())
            .to_string();
        result_widgets::tile_toolbelt(ui, data, tile_idx, &label, state);
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

    // Build x-axis values (same pattern as analyze.rs)
    let (x_values, x_label): (Vec<f64>, &str) = match state.analyze_spectrum_axis {
        SpectrumAxis::EnergyEv => match state.energies {
            Some(ref e) => (e.clone(), "Energy (eV)"),
            None => {
                ui.label("No energy grid loaded.");
                return;
            }
        },
        SpectrumAxis::TofMicroseconds => match state.spectrum_values {
            Some(ref v) => {
                use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
                match (state.spectrum_unit, state.spectrum_kind) {
                    (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinEdges) => {
                        let centers: Vec<f64> = v
                            .windows(2)
                            .take(n_tof)
                            .map(|w| 0.5 * (w[0] + w[1]))
                            .collect();
                        (centers, "TOF (\u{03bc}s)")
                    }
                    (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinCenters) => {
                        (v.iter().take(n_tof).copied().collect(), "TOF (\u{03bc}s)")
                    }
                    (SpectrumUnit::EnergyEv, _) => {
                        // Spectrum file is in energy units — convert to TOF for the plot axis.
                        if state.beamline.flight_path_m.is_finite()
                            && state.beamline.flight_path_m > 0.0
                        {
                            let tof_vals: Vec<f64> = v
                                .iter()
                                .take(n_tof)
                                .map(|&e| {
                                    if e > 0.0 {
                                        nereids_core::constants::energy_to_tof(
                                            e,
                                            state.beamline.flight_path_m,
                                        ) + state.beamline.delay_us
                                    } else {
                                        f64::NAN
                                    }
                                })
                                .collect();
                            (tof_vals, "TOF (\u{03bc}s)")
                        } else {
                            // Cannot convert without valid beamline params — show energy instead.
                            (v.iter().take(n_tof).copied().collect(), "Energy (eV)")
                        }
                    }
                }
            }
            None => {
                let indices: Vec<f64> = (0..n_tof).map(|i| i as f64).collect();
                (indices, "Frame index")
            }
        },
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
    let energies = state.energies.as_ref();
    let fit_line = state.pixel_fit_result.as_ref().and_then(|result| {
        if !result.converged {
            return None;
        }
        let energies = energies?;
        let enabled: Vec<_> = state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .collect();
        if enabled.is_empty() {
            return None;
        }
        let resonance_data: Vec<_> = enabled
            .iter()
            .filter_map(|e| e.resonance_data.clone())
            .collect();
        let model = nereids_fitting::transmission_model::TransmissionFitModel::new(
            energies.clone(),
            resonance_data,
            state.temperature_k,
            None,
            (0..result.densities.len()).collect(),
            None,
        )
        .ok()?;

        use nereids_fitting::lm::FitModel;
        let fitted_t = model.evaluate(&result.densities);
        let n_fit = n_plot.min(fitted_t.len());
        let fit_points: PlotPoints = (0..n_fit)
            .filter(|&i| x_values[i].is_finite())
            .map(|i| [x_values[i], fitted_t[i]])
            .collect();
        Some(Line::new("Fit", fit_points).width(2.0))
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
            ui.label(format!("chi2_r = {:.4}", result.reduced_chi_squared));
            ui.label(format!("iter = {}", result.iterations));
        });

        for (i, entry) in state
            .isotope_entries
            .iter()
            .filter(|e| e.enabled && e.resonance_data.is_some())
            .enumerate()
        {
            if i < result.densities.len() {
                let unc_str = result
                    .uncertainties
                    .as_ref()
                    .and_then(|u| u.get(i))
                    .map_or("N/A".to_string(), |u| format!("{:.2e}", u));
                let dot_color = design::isotope_dot_color(&entry.symbol);
                ui.horizontal(|ui| {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                    ui.label(format!(
                        "{}: {:.4e} \u{00b1} {}",
                        entry.symbol, result.densities[i], unc_str
                    ));
                });
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
        .default_height(170.0)
        .min_height(80.0)
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
                1 => dock_residuals(ui),
                2 => dock_provenance(ui, state),
                3 => dock_export(ui, state),
                _ => {}
            });
        });
}

/// Isotopes table (read-only view of configure isotope list).
fn dock_isotopes(ui: &mut egui::Ui, state: &AppState) {
    if state.isotope_entries.is_empty() {
        ui.label(
            egui::RichText::new("No isotopes configured. Add isotopes in Guided → Configure.")
                .small()
                .color(ThemeColors::from_ctx(ui.ctx()).fg3),
        );
        return;
    }

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
}

/// Residuals placeholder.
fn dock_residuals(ui: &mut egui::Ui) {
    ui.label(
        egui::RichText::new("Residual analysis \u{2014} coming in a future update.")
            .small()
            .color(ThemeColors::from_ctx(ui.ctx()).fg3),
    );
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
            ui.label(egui::RichText::new(&ts[11..19]).small().monospace());

            let kind_color = match event.kind {
                crate::state::ProvenanceEventKind::DataLoaded => crate::theme::semantic::YELLOW,
                crate::state::ProvenanceEventKind::ConfigChanged => crate::theme::semantic::ORANGE,
                crate::state::ProvenanceEventKind::Normalized => crate::theme::semantic::GREEN,
                crate::state::ProvenanceEventKind::AnalysisRun => crate::theme::semantic::ORANGE,
                crate::state::ProvenanceEventKind::Exported => crate::theme::semantic::GREEN,
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

/// Export panel (reuse result_widgets).
fn dock_export(ui: &mut egui::Ui, state: &mut AppState) {
    result_widgets::export_panel(ui, state);
}

// ---------------------------------------------------------------------------
// Mini-inspector sidebar (right, Analysis tab only)
// ---------------------------------------------------------------------------

fn mini_inspector(ctx: &egui::Context, state: &mut AppState) {
    let colors = ThemeColors::from_ctx(ctx);
    egui::SidePanel::right("studio_mini_inspector")
        .resizable(true)
        .default_width(220.0)
        .min_width(180.0)
        .max_width(320.0)
        .frame(
            egui::Frame::NONE
                .fill(colors.bg)
                .inner_margin(egui::Margin::symmetric(10, 8))
                .stroke(egui::Stroke::new(1.0, colors.border)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                beamline_summary(ui, state);
                ui.add_space(8.0);

                solver_summary(ui, state);
                ui.add_space(8.0);

                result_widgets::pixel_inspector(ui, state);
                ui.add_space(8.0);

                if let Some(ref result) = state.spatial_result {
                    result_widgets::summary_card(ui, result, &state.isotope_entries);
                }
            });
        });
}

/// Compact beamline config summary (read-only).
fn beamline_summary(ui: &mut egui::Ui, state: &AppState) {
    let colors = ThemeColors::from_ctx(ui.ctx());
    design::card_with_header(ui, "Beamline", None, |ui| {
        ui.label(
            egui::RichText::new(format!(
                "Flight path: {} m",
                format_compact_f64(state.beamline.flight_path_m)
            ))
            .small()
            .color(colors.fg2),
        );
        ui.label(
            egui::RichText::new(format!(
                "Delay: {} \u{03bc}s",
                format_compact_f64(state.beamline.delay_us)
            ))
            .small()
            .color(colors.fg2),
        );
    });
}

/// Compact solver config summary (read-only).
fn solver_summary(ui: &mut egui::Ui, state: &AppState) {
    let colors = ThemeColors::from_ctx(ui.ctx());
    let method = match state.solver_method {
        crate::state::SolverMethod::LevenbergMarquardt => "Levenberg-Marquardt",
        crate::state::SolverMethod::PoissonKL => "Poisson KL",
    };
    design::card_with_header(ui, "Solver", None, |ui| {
        ui.label(egui::RichText::new(method).small().color(colors.fg2));
        ui.label(
            egui::RichText::new(format!("Max iter: {}", state.lm_config.max_iter))
                .small()
                .color(colors.fg2),
        );
        ui.label(
            egui::RichText::new(format!("Temp: {:.0} K", state.temperature_k))
                .small()
                .color(colors.fg2),
        );
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
