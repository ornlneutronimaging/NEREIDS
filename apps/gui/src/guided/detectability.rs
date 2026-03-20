//! Detectability tool — multi-matrix + trace isotope analysis with resolution broadening,
//! verdict badges, hero stats, and delta-T spectrum plot.

use crate::state::{AppState, EndfStatus, GuidedStep, PeriodicTableTarget};
use crate::widgets::design;
use egui_plot::{HLine, Line, Plot, PlotPoints};
use nereids_endf::retrieval::EndfLibrary;
use std::sync::Arc;
use std::sync::mpsc;

/// Fraction of opaque energy bins above which we show a warning.
const OPAQUE_WARN_FRACTION: f64 = 0.5;

/// Draw the Detectability tool content.
pub fn detectability_step(ui: &mut egui::Ui, state: &mut AppState) {
    // -- Header row --
    ui.horizontal(|ui| {
        design::content_header(ui, "Detectability", "Trace element sensitivity analysis");
    });
    ui.horizontal(|ui| {
        design::teleport_pill(ui, "← Configure", GuidedStep::Configure, state);
    });

    ui.add_space(8.0);

    // -- ENDF library selector (shared by matrix + trace) --
    let isotope_locked = state.is_fetching_detect_endf;
    detect_library_selector(ui, state, isotope_locked);

    ui.add_space(8.0);

    // -- Matrix isotopes card --
    detect_matrix_card(ui, state, isotope_locked);
    ui.add_space(8.0);

    // -- Trace isotopes card --
    detect_trace_card(ui, state, isotope_locked);
    ui.add_space(8.0);

    // -- Resolution card --
    detect_resolution_card(ui, state);
    ui.add_space(8.0);

    // -- Advanced config --
    detect_advanced_config(ui, state);
    ui.add_space(8.0);

    // -- Fetch ENDF + Run buttons --
    detect_action_buttons(ui, state);
    ui.add_space(12.0);

    // -- Results: hero stats + verdict table + spectrum --
    detect_results_panel(ui, state);
}

/// ENDF library selector, shared by matrix and trace isotopes.
pub(crate) fn detect_library_selector(ui: &mut egui::Ui, state: &mut AppState, locked: bool) {
    let prev_lib = state.detect_endf_library;
    ui.add_enabled_ui(!locked, |ui| {
        ui.horizontal(|ui| {
            ui.label("Library:");
            egui::ComboBox::from_id_salt("detect_endf_lib")
                .selected_text(design::library_name(state.detect_endf_library))
                .show_ui(ui, |ui| {
                    for (val, label) in [
                        (EndfLibrary::EndfB8_0, "ENDF/B-VIII.0"),
                        (EndfLibrary::EndfB8_1, "ENDF/B-VIII.1"),
                        (EndfLibrary::Jeff3_3, "JEFF-3.3"),
                        (EndfLibrary::Jendl5, "JENDL-5"),
                    ] {
                        ui.selectable_value(&mut state.detect_endf_library, val, label);
                    }
                });
        });
    });
    if state.detect_endf_library != prev_lib {
        for m in &mut state.detect_matrix_entries {
            m.resonance_data = None;
            m.endf_status = EndfStatus::Pending;
        }
        for t in &mut state.detect_trace_entries {
            t.resonance_data = None;
            t.endf_status = EndfStatus::Pending;
        }
        state.detect_results.clear();
    }
}

/// Card: matrix isotopes list with add/remove, density, ENDF badges.
pub(crate) fn detect_matrix_card(ui: &mut egui::Ui, state: &mut AppState, locked: bool) {
    design::card_with_header(ui, "Matrix Isotopes", None, |ui| {
        let mut matrix_remove = None;
        for (idx, matrix) in state.detect_matrix_entries.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                // Colored dot
                let dot_color = design::isotope_dot_color(&matrix.symbol);
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, dot_color);

                ui.add_enabled_ui(!locked, |ui| {
                    let z_changed = ui
                        .add(
                            egui::DragValue::new(&mut matrix.z)
                                .prefix("Z=")
                                .range(1..=118),
                        )
                        .changed();
                    let a_changed = ui
                        .add(
                            egui::DragValue::new(&mut matrix.a)
                                .prefix("A=")
                                .range(1..=300),
                        )
                        .changed();
                    if z_changed && matrix.z > matrix.a {
                        matrix.a = matrix.z;
                    }
                    if a_changed && matrix.a < matrix.z {
                        matrix.z = matrix.a;
                    }
                    if z_changed || a_changed {
                        matrix.symbol = format!(
                            "{}-{}",
                            nereids_core::elements::element_symbol(matrix.z).unwrap_or("??"),
                            matrix.a
                        );
                        matrix.resonance_data = None;
                        matrix.endf_status = EndfStatus::Pending;
                        state.detect_results.clear();
                    }
                });
                ui.label(&matrix.symbol);
                ui.add_enabled_ui(!locked, |ui| {
                    if ui
                        .add(
                            egui::DragValue::new(&mut matrix.initial_density)
                                .speed(0.0001)
                                .range(1e-6..=1.0)
                                .max_decimals(6),
                        )
                        .changed()
                    {
                        state.detect_results.clear();
                    }
                });

                // ENDF status badge
                match matrix.endf_status {
                    EndfStatus::Loaded => design::badge(ui, "OK", design::BadgeVariant::Green),
                    EndfStatus::Fetching => {
                        ui.spinner();
                    }
                    EndfStatus::Failed => design::badge(ui, "ERR", design::BadgeVariant::Red),
                    EndfStatus::Pending => design::badge(ui, "...", design::BadgeVariant::Orange),
                }

                ui.add_enabled_ui(!locked, |ui| {
                    if ui.small_button("\u{00d7}").clicked() {
                        matrix_remove = Some(idx);
                    }
                });
            });
        }
        if let Some(idx) = matrix_remove {
            state.detect_matrix_entries.remove(idx);
            state.detect_results.clear();
        }

        ui.add_space(4.0);
        ui.add_enabled_ui(!locked, |ui| {
            if ui.button("Add Matrix Isotope...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::DetectMatrix;
                state.periodic_table_selected_z = None;
                state.periodic_table_density = 0.001; // at/barn default
            }
        });
    });
}

/// Card: trace isotopes list with add/remove, concentration, ENDF status.
pub(crate) fn detect_trace_card(ui: &mut egui::Ui, state: &mut AppState, locked: bool) {
    design::card_with_header(ui, "Trace Isotopes", None, |ui| {
        let mut to_remove = None;
        for (idx, entry) in state.detect_trace_entries.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                // Colored dot
                let dot_color = design::isotope_dot_color(&entry.symbol);
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, dot_color);

                ui.add_enabled_ui(!locked, |ui| {
                    let z_changed = ui
                        .add(
                            egui::DragValue::new(&mut entry.z)
                                .prefix("Z=")
                                .range(1..=118),
                        )
                        .changed();
                    let a_changed = ui
                        .add(
                            egui::DragValue::new(&mut entry.a)
                                .prefix("A=")
                                .range(1..=300),
                        )
                        .changed();
                    if z_changed && entry.z > entry.a {
                        entry.a = entry.z;
                    }
                    if a_changed && entry.a < entry.z {
                        entry.z = entry.a;
                    }
                    if z_changed || a_changed {
                        entry.symbol = format!(
                            "{}-{}",
                            nereids_core::elements::element_symbol(entry.z).unwrap_or("??"),
                            entry.a
                        );
                        entry.resonance_data = None;
                        entry.endf_status = EndfStatus::Pending;
                        state.detect_results.clear();
                    }
                });
                ui.label(&entry.symbol);
                ui.add_enabled_ui(!locked, |ui| {
                    if ui
                        .add(
                            egui::DragValue::new(&mut entry.concentration_ppm)
                                .prefix("ppm=")
                                .speed(10.0)
                                .range(0.1..=1e6),
                        )
                        .changed()
                    {
                        state.detect_results.clear();
                    }
                });

                match entry.endf_status {
                    EndfStatus::Loaded => design::badge(ui, "OK", design::BadgeVariant::Green),
                    EndfStatus::Fetching => design::badge(ui, "…", design::BadgeVariant::Orange),
                    EndfStatus::Failed => design::badge(ui, "ERR", design::BadgeVariant::Red),
                    EndfStatus::Pending => design::badge(ui, "...", design::BadgeVariant::Orange),
                }

                ui.add_enabled_ui(!locked, |ui| {
                    if ui.small_button("\u{00d7}").clicked() {
                        to_remove = Some(idx);
                    }
                });
            });
        }
        if let Some(idx) = to_remove {
            state.detect_trace_entries.remove(idx);
            state.detect_results.clear();
        }

        ui.add_space(4.0);
        ui.add_enabled_ui(!locked, |ui| {
            if ui.button("Add Trace...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::DetectTrace;
                state.periodic_table_selected_z = None;
                state.periodic_table_density = 1000.0; // ppm default
            }
        });
    });
}

/// Resolution card: Gaussian (parametric) or tabulated file, with shared widget.
pub(crate) fn detect_resolution_card(ui: &mut egui::Ui, state: &mut AppState) {
    let res = design::resolution_card(
        ui,
        &mut state.detect_resolution_enabled,
        &mut state.detect_resolution_mode,
        state.beamline.flight_path_m,
    );
    if res.changed {
        state.detect_results.clear();
    }
}

/// Advanced config: SNR threshold, I0, energy range, points, temperature.
pub(crate) fn detect_advanced_config(ui: &mut egui::Ui, state: &mut AppState) {
    let prev_snr = state.detect_snr_threshold;
    let prev_i0 = state.detect_i0;
    let prev_emin = state.detect_energy_min;
    let prev_emax = state.detect_energy_max;
    let prev_npts = state.detect_n_energy_points;
    let prev_temp = state.detect_temperature_k;

    egui::CollapsingHeader::new("Advanced")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("SNR threshold:");
                ui.add(
                    egui::DragValue::new(&mut state.detect_snr_threshold)
                        .speed(0.1)
                        .range(0.1..=10.0),
                );
            });
            ui.horizontal(|ui| {
                ui.label("I\u{2080} (counts/bin):");
                ui.add(
                    egui::DragValue::new(&mut state.detect_i0)
                        .speed(100.0)
                        .range(1.0..=1e8),
                );
            });
            ui.horizontal(|ui| {
                ui.label("E_min (eV):");
                ui.add(
                    egui::DragValue::new(&mut state.detect_energy_min)
                        .speed(0.1)
                        .range(0.001..=1e4),
                );
                ui.label("E_max:");
                ui.add(
                    egui::DragValue::new(&mut state.detect_energy_max)
                        .speed(1.0)
                        .range(0.01..=1e6),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Energy points:");
                ui.add(
                    egui::DragValue::new(&mut state.detect_n_energy_points)
                        .speed(10.0)
                        .range(100..=50000),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Temperature (K):");
                ui.add(
                    egui::DragValue::new(&mut state.detect_temperature_k)
                        .range(1.0..=5000.0)
                        .speed(1.0),
                );
            });
        });

    if state.detect_snr_threshold != prev_snr
        || state.detect_i0 != prev_i0
        || state.detect_energy_min != prev_emin
        || state.detect_energy_max != prev_emax
        || state.detect_n_energy_points != prev_npts
        || state.detect_temperature_k != prev_temp
    {
        state.detect_results.clear();
    }
}

/// Fetch ENDF + Run Analysis buttons.
pub(crate) fn detect_action_buttons(ui: &mut egui::Ui, state: &mut AppState) {
    // Auto-fetch ENDF data when entries are pending
    let has_pending = state
        .detect_matrix_entries
        .iter()
        .any(|m| m.endf_status == EndfStatus::Pending)
        || state
            .detect_trace_entries
            .iter()
            .any(|t| t.endf_status == EndfStatus::Pending);
    if has_pending && !state.is_fetching_detect_endf {
        detect_fetch_endf_data(state);
    }

    ui.horizontal(|ui| {
        let can_run = !state.detect_matrix_entries.is_empty()
            && state
                .detect_matrix_entries
                .iter()
                .all(|m| m.endf_status == EndfStatus::Loaded)
            && state
                .detect_trace_entries
                .iter()
                .any(|t| t.endf_status == EndfStatus::Loaded);

        ui.add_enabled_ui(can_run, |ui| {
            if design::btn_primary(ui, "Run Analysis").clicked() {
                run_detectability(state);
            }
        });

        if state.is_fetching_detect_endf {
            ui.spinner();
            ui.label("Fetching ENDF data…");
        }
    });

    let has_failed = state
        .detect_matrix_entries
        .iter()
        .any(|m| m.endf_status == EndfStatus::Failed)
        || state
            .detect_trace_entries
            .iter()
            .any(|t| t.endf_status == EndfStatus::Failed);
    if has_failed && !state.is_fetching_detect_endf && ui.button("Retry failed").clicked() {
        for e in &mut state.detect_matrix_entries {
            if e.endf_status == EndfStatus::Failed {
                e.endf_status = EndfStatus::Pending;
            }
        }
        for e in &mut state.detect_trace_entries {
            if e.endf_status == EndfStatus::Failed {
                e.endf_status = EndfStatus::Pending;
            }
        }
    }
}

/// Results panel: hero stat row, verdict table with badges, delta-T spectrum.
pub(crate) fn detect_results_panel(ui: &mut egui::Ui, state: &AppState) {
    if state.detect_results.is_empty() {
        design::card(ui, |ui| {
            ui.label(
                egui::RichText::new(
                    "Configure matrix and trace isotopes, then click Run Analysis.",
                )
                .italics()
                .color(crate::theme::semantic::ORANGE),
            );
        });
        return;
    }

    // Summary banner
    let n_total = state.detect_results.len();
    let n_detect = state
        .detect_results
        .iter()
        .filter(|(_, r)| r.detectable)
        .count();
    let summary = if n_detect == n_total {
        format!(
            "ALL DETECTABLE (>{:.0}\u{03c3})",
            state.detect_snr_threshold
        )
    } else {
        format!("{}/{} DETECTABLE", n_detect, n_total)
    };
    let summary_variant = if n_detect == n_total {
        design::BadgeVariant::Green
    } else if n_detect > 0 {
        design::BadgeVariant::Orange
    } else {
        design::BadgeVariant::Red
    };
    design::badge(ui, &summary, summary_variant);
    ui.add_space(8.0);

    // Hero stat row — best result
    if let Some((best_name, best)) = state.detect_results.iter().max_by(|(_, a), (_, b)| {
        a.peak_snr
            .partial_cmp(&b.peak_snr)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        let snr_str = format!("{:.1}\u{03c3}", best.peak_snr);
        let energy_str = format!("{:.2} eV", best.peak_energy_ev);
        design::stat_row(
            ui,
            &[
                (&snr_str, "Best SNR"),
                (best_name, "Best Isotope"),
                (&energy_str, "Peak Energy"),
            ],
        );
    }

    ui.add_space(8.0);

    // Opaque matrix warning
    let max_opaque = state
        .detect_results
        .iter()
        .map(|(_, r)| r.opaque_fraction)
        .fold(0.0f64, f64::max);
    if max_opaque > OPAQUE_WARN_FRACTION {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "\u{26a0} Matrix is opaque at {:.0}% of energies — reduce matrix density",
                    max_opaque * 100.0,
                ))
                .color(crate::theme::semantic::ORANGE),
            );
        });
        ui.add_space(4.0);
    }

    // Verdict table with badges
    design::card_with_header(ui, "Verdict Table", None, |ui| {
        egui::Grid::new("detect_verdict_grid")
            .striped(true)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Isotope").strong());
                ui.label(egui::RichText::new("Peak E (eV)").strong());
                ui.label(egui::RichText::new("Peak SNR").strong());
                ui.label(egui::RichText::new("\u{0394}T/ppm").strong());
                ui.label(egui::RichText::new("Verdict").strong());
                ui.end_row();

                for (name, report) in &state.detect_results {
                    ui.label(name);
                    ui.label(format!("{:.2}", report.peak_energy_ev));
                    ui.label(format!("{:.2}\u{03c3}", report.peak_snr));
                    ui.label(format!("{:.2e}", report.peak_delta_t_per_ppm));
                    if report.detectable {
                        design::badge(ui, "DETECTABLE", design::BadgeVariant::Green);
                    } else if report.opaque_fraction > OPAQUE_WARN_FRACTION {
                        design::badge(ui, "OPAQUE MATRIX", design::BadgeVariant::Red);
                    } else {
                        design::badge(ui, "NOT DETECTED", design::BadgeVariant::Red);
                    }
                    ui.end_row();
                }
            });
    });

    ui.add_space(12.0);

    // Delta-T spectrum plot
    if state
        .detect_results
        .iter()
        .any(|(_, r)| !r.delta_t_spectrum.is_empty())
    {
        design::card_with_header(ui, "\u{0394}T Spectrum", None, |ui| {
            ui.label(
                egui::RichText::new("Transmission difference per trace")
                    .small()
                    .weak(),
            );
            let plot_height = ui.available_height().clamp(200.0, 300.0);
            Plot::new("detect_delta_t_plot")
                .height(plot_height)
                .x_axis_label("Energy (eV)")
                .y_axis_label("|\u{0394}T|")
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    // Detection threshold line: snr_threshold / sqrt(i0)
                    let threshold = state.detect_snr_threshold / state.detect_i0.sqrt();
                    plot_ui.hline(
                        HLine::new("threshold", threshold)
                            .color(egui::Color32::from_rgb(200, 200, 200))
                            .style(egui_plot::LineStyle::dashed_loose()),
                    );

                    for (name, report) in &state.detect_results {
                        if report.delta_t_spectrum.is_empty() {
                            continue;
                        }
                        let n = report.energies.len().min(report.delta_t_spectrum.len());
                        let points: PlotPoints = (0..n)
                            .filter(|&i| {
                                report.energies[i].is_finite()
                                    && report.delta_t_spectrum[i].is_finite()
                            })
                            .map(|i| [report.energies[i], report.delta_t_spectrum[i]])
                            .collect();
                        let color = design::isotope_dot_color(name);
                        plot_ui.line(Line::new(name.as_str(), points).color(color).width(1.5));
                    }
                });
        });
    }
}

/// Run detectability analysis for all trace candidates against the matrix.
fn run_detectability(state: &mut AppState) {
    use nereids_pipeline::detectability::{TraceDetectabilityConfig, trace_detectability};

    let matrix_isotopes: Vec<_> = state
        .detect_matrix_entries
        .iter()
        .filter_map(|e| e.resonance_data.clone().map(|rd| (rd, e.initial_density)))
        .collect();

    if matrix_isotopes.is_empty() {
        state.status_message = "Matrix isotope(s) need ENDF data".into();
        return;
    }

    let e_min = state.detect_energy_min;
    let e_max = state.detect_energy_max;
    if e_min >= e_max {
        state.status_message = format!(
            "E_min ({:.3} eV) must be less than E_max ({:.3} eV)",
            e_min, e_max
        );
        return;
    }

    let n = state.detect_n_energy_points;
    let energies: Vec<f64> = (0..n)
        .map(|i| e_min * (e_max / e_min).powf(i as f64 / (n - 1).max(1) as f64))
        .collect();

    // Build resolution function if enabled
    let res_fn = match design::build_resolution_function(
        state.detect_resolution_enabled,
        &state.detect_resolution_mode,
        state.beamline.flight_path_m,
    ) {
        Ok(rf) => rf,
        Err(msg) => {
            state.status_message = msg;
            None
        }
    };

    let config = TraceDetectabilityConfig {
        matrix_isotopes: &matrix_isotopes,
        energies: &energies,
        i0: state.detect_i0,
        temperature_k: state.detect_temperature_k,
        resolution: res_fn.as_ref(),
        snr_threshold: state.detect_snr_threshold,
    };

    let mut results = Vec::new();
    let mut n_errors = 0usize;
    for trace_entry in &state.detect_trace_entries {
        let trace_rd = match &trace_entry.resonance_data {
            Some(rd) => rd,
            None => continue,
        };
        match trace_detectability(&config, trace_rd, trace_entry.concentration_ppm) {
            Ok(report) => results.push((trace_entry.symbol.clone(), report)),
            Err(e) => {
                state.status_message =
                    format!("Detectability error for {}: {}", trace_entry.symbol, e);
                n_errors += 1;
            }
        }
    }

    // Sort by peak_snr descending
    results.sort_by(|(_, a), (_, b)| {
        b.peak_snr
            .partial_cmp(&a.peak_snr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if results.is_empty() && n_errors == 0 {
        state.status_message = "No trace isotopes had data to analyze".into();
        return;
    }

    state.detect_results = results;
    if n_errors == 0 {
        state.status_message = "Detectability analysis complete".into();
    } else {
        state.status_message = format!(
            "Detectability analysis complete ({} trace(s) failed)",
            n_errors
        );
    }
}

/// Copy main Configure isotopes into the Detectability matrix list.
///
/// When the main config uses a different ENDF library, trace entries are
/// invalidated (resonance data cleared, status reset to Pending) to prevent
/// mixed-library detectability runs.
pub(crate) fn copy_config_to_detect_matrix(state: &mut AppState) {
    let library_changed = state.endf_library != state.detect_endf_library;

    state.detect_matrix_entries = state
        .isotope_entries
        .iter()
        .map(|e| e.clone_with_normalized_status())
        .collect();
    state.detect_endf_library = state.endf_library;
    state.detect_results.clear();

    // If the library changed, invalidate trace entries so they get re-fetched
    // from the new library (same pattern as detect_library_selector).
    if library_changed {
        for t in &mut state.detect_trace_entries {
            t.resonance_data = None;
            t.endf_status = EndfStatus::Pending;
        }
    }
}

/// Fetch ENDF data for matrix + trace isotopes.
/// Index convention: 0..N = matrix entries, N.. = trace entries at (index - N).
pub(crate) fn detect_fetch_endf_data(state: &mut AppState) {
    use nereids_core::types::Isotope;
    use nereids_endf::retrieval;

    let mut work: Vec<design::EndfWorkItem> = Vec::new();
    let mut failed_matrix: Vec<usize> = Vec::new();
    let mut failed_trace: Vec<usize> = Vec::new();

    for (i, entry) in state.detect_matrix_entries.iter().enumerate() {
        if entry.endf_status == EndfStatus::Pending {
            match Isotope::new(entry.z, entry.a) {
                Ok(isotope) => {
                    if retrieval::mat_number(&isotope).is_some() {
                        work.push(design::EndfWorkItem {
                            z: entry.z,
                            a: entry.a,
                            is_detect_matrix: true,
                            isotope,
                            symbol: entry.symbol.clone(),
                            library: state.detect_endf_library,
                        });
                    } else {
                        state.status_message = format!(
                            "No MAT number for matrix {} \u{2014} isotope not in database",
                            entry.symbol
                        );
                        failed_matrix.push(i);
                    }
                }
                Err(e) => {
                    state.status_message = format!(
                        "Matrix isotope Z={} A={} is not supported: {}",
                        entry.z, entry.a, e
                    );
                    failed_matrix.push(i);
                }
            }
        }
    }

    for (i, entry) in state.detect_trace_entries.iter().enumerate() {
        if entry.endf_status == EndfStatus::Pending {
            match Isotope::new(entry.z, entry.a) {
                Ok(isotope) => {
                    if retrieval::mat_number(&isotope).is_some() {
                        work.push(design::EndfWorkItem {
                            z: entry.z,
                            a: entry.a,
                            is_detect_matrix: false,
                            isotope,
                            symbol: entry.symbol.clone(),
                            library: state.detect_endf_library,
                        });
                    } else {
                        state.status_message = format!(
                            "No MAT number for {} \u{2014} isotope not in database",
                            entry.symbol
                        );
                        failed_trace.push(i);
                    }
                }
                Err(e) => {
                    state.status_message = format!("Invalid isotope {}: {}", entry.symbol, e);
                    failed_trace.push(i);
                }
            }
        }
    }
    for i in failed_matrix {
        state.detect_matrix_entries[i].endf_status = EndfStatus::Failed;
    }
    for i in failed_trace {
        state.detect_trace_entries[i].endf_status = EndfStatus::Failed;
    }

    if work.is_empty() {
        state.status_message =
            "No supported isotopes found \u{2014} none have MAT numbers in the ENDF database"
                .into();
        return;
    }

    for item in &work {
        if item.is_detect_matrix {
            for entry in state.detect_matrix_entries.iter_mut() {
                if entry.z == item.z && entry.a == item.a {
                    entry.endf_status = EndfStatus::Fetching;
                }
            }
        } else {
            for entry in state.detect_trace_entries.iter_mut() {
                if entry.z == item.z && entry.a == item.a {
                    entry.endf_status = EndfStatus::Fetching;
                }
            }
        }
    }

    let (tx, rx) = mpsc::channel();
    state.pending_detect_endf = Some(rx);
    state.is_fetching_detect_endf = true;
    state.status_message = "Fetching ENDF data (Detect)...".into();
    let cancel = Arc::clone(&state.cancel_token);

    std::thread::spawn(move || design::endf_fetch_worker(work, cancel, tx));
}
