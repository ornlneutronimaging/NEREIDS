//! Detectability tool — matrix/trace isotope analysis with verdict table.

use crate::state::{
    AppState, DetectTraceEntry, EndfFetchResult, EndfStatus, GuidedStep, IsotopeEntry,
    PeriodicTableTarget,
};
use nereids_endf::retrieval::EndfLibrary;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Draw the Detectability tool content.
pub fn detectability_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.heading("Detectability");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            teleport_pill(ui, "← Configure", GuidedStep::Configure, state);
        });
    });
    ui.separator();

    ui.add_space(8.0);

    // Two-column layout: controls (left) | results (right)
    let available_width = ui.available_width();
    let controls_width = 300.0_f32.min(available_width * 0.4);

    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(controls_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("detect_controls")
                    .show(ui, |ui| {
                        detect_controls(ui, state);
                    });
            },
        );
        ui.separator();
        ui.vertical(|ui| {
            detect_results_panel(ui, state);
        });
    });
}

/// Controls column: matrix isotope, trace isotopes, advanced config, run button.
fn detect_controls(ui: &mut egui::Ui, state: &mut AppState) {
    let isotope_locked = state.is_fetching_detect_endf;

    // --- Matrix isotope ---
    ui.label(egui::RichText::new("Matrix Isotope").strong());
    ui.add_space(4.0);

    // ENDF library selector (disabled during active fetch to prevent stale results)
    let prev_lib = state.detect_endf_library;
    ui.add_enabled_ui(!state.is_fetching_detect_endf, |ui| {
        ui.horizontal(|ui| {
            ui.label("Library:");
            egui::ComboBox::from_id_salt("detect_endf_lib")
                .selected_text(library_name(state.detect_endf_library))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.detect_endf_library,
                        EndfLibrary::EndfB8_0,
                        "ENDF/B-VIII.0",
                    );
                    ui.selectable_value(
                        &mut state.detect_endf_library,
                        EndfLibrary::EndfB8_1,
                        "ENDF/B-VIII.1",
                    );
                    ui.selectable_value(
                        &mut state.detect_endf_library,
                        EndfLibrary::Jeff3_3,
                        "JEFF-3.3",
                    );
                    ui.selectable_value(
                        &mut state.detect_endf_library,
                        EndfLibrary::Jendl5,
                        "JENDL-5",
                    );
                });
        });
    });
    // Library change invalidates all resonance data — must re-fetch
    if state.detect_endf_library != prev_lib {
        for m in &mut state.detect_matrix_entries {
            m.resonance_data = None;
            m.endf_status = EndfStatus::Pending;
        }
        for t in &mut state.detect_trace_entries {
            t.resonance_data = None;
        }
        state.detect_results.clear();
    }

    // Matrix isotope list
    let mut matrix_remove = None;
    for (idx, matrix) in state.detect_matrix_entries.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!isotope_locked, |ui| {
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
            ui.add_enabled_ui(!isotope_locked, |ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut matrix.initial_density)
                            .speed(0.0001)
                            .range(1e-6..=1.0),
                    )
                    .changed()
                {
                    state.detect_results.clear();
                }
            });
            if matrix.resonance_data.is_some() {
                ui.label("OK");
            }
            ui.add_enabled_ui(!isotope_locked, |ui| {
                if ui.small_button("X").clicked() {
                    matrix_remove = Some(idx);
                }
            });
        });
    }
    if let Some(idx) = matrix_remove {
        state.detect_matrix_entries.remove(idx);
        state.detect_results.clear();
    }

    ui.add_enabled_ui(!isotope_locked, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Add Matrix Isotope").clicked() {
                state.detect_matrix_entries.push(IsotopeEntry {
                    z: 26,
                    a: 56,
                    symbol: "Fe-56".into(),
                    initial_density: 0.001,
                    resonance_data: None,
                    enabled: true,
                    endf_status: EndfStatus::Pending,
                });
                state.detect_results.clear();
            }
            if ui.button("Periodic Table...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::DetectMatrix;
                state.periodic_table_selected_z = None;
            }
        });
    });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    // --- Trace isotopes ---
    ui.label(egui::RichText::new("Trace Isotopes").strong());
    ui.add_space(4.0);

    ui.add_enabled_ui(!isotope_locked, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Add Trace").clicked() {
                state.detect_trace_entries.push(DetectTraceEntry {
                    z: 72,
                    a: 178,
                    symbol: "Hf-178".into(),
                    concentration_ppm: 1000.0,
                    resonance_data: None,
                });
                state.detect_results.clear();
            }
            if ui.button("Periodic Table...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::DetectTrace;
                state.periodic_table_selected_z = None;
            }
        });
    });

    let mut to_remove = None;
    for (idx, entry) in state.detect_trace_entries.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!isotope_locked, |ui| {
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
                    state.detect_results.clear();
                }
            });
            ui.label(&entry.symbol);
            ui.add_enabled_ui(!isotope_locked, |ui| {
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
            if entry.resonance_data.is_some() {
                ui.label("OK");
            }
            ui.add_enabled_ui(!isotope_locked, |ui| {
                if ui.small_button("X").clicked() {
                    to_remove = Some(idx);
                }
            });
        });
    }
    if let Some(idx) = to_remove {
        state.detect_trace_entries.remove(idx);
        state.detect_results.clear();
    }

    ui.add_space(8.0);

    // --- Advanced config ---
    // Snapshot values to detect changes
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
                ui.label("I_0 (counts/bin):");
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

    // Invalidate stale results when any config param changes
    if state.detect_snr_threshold != prev_snr
        || state.detect_i0 != prev_i0
        || state.detect_energy_min != prev_emin
        || state.detect_energy_max != prev_emax
        || state.detect_n_energy_points != prev_npts
        || state.detect_temperature_k != prev_temp
    {
        state.detect_results.clear();
    }

    ui.add_space(8.0);

    // Fetch ENDF + Run buttons
    let has_missing_endf = state
        .detect_matrix_entries
        .iter()
        .any(|m| m.resonance_data.is_none())
        || state
            .detect_trace_entries
            .iter()
            .any(|t| t.resonance_data.is_none());

    ui.add_enabled_ui(has_missing_endf && !state.is_fetching_detect_endf, |ui| {
        if ui.button("Fetch ENDF Data").clicked() {
            detect_fetch_endf_data(state);
        }
    });
    if state.is_fetching_detect_endf {
        ui.spinner();
    }

    let can_run = !state.detect_matrix_entries.is_empty()
        && state
            .detect_matrix_entries
            .iter()
            .all(|m| m.resonance_data.is_some())
        && state
            .detect_trace_entries
            .iter()
            .any(|t| t.resonance_data.is_some());

    ui.add_enabled_ui(can_run, |ui| {
        if ui.button("Run Analysis").clicked() {
            run_detectability(state);
        }
    });
}

/// Results column: verdict table and optional spectrum plot.
fn detect_results_panel(ui: &mut egui::Ui, state: &AppState) {
    if state.detect_results.is_empty() {
        ui.label("Configure matrix and trace isotopes, then click Run Analysis.");
        return;
    }

    ui.label(egui::RichText::new("Verdict Table").strong());
    ui.add_space(4.0);

    egui::Grid::new("detect_verdict_grid")
        .striped(true)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            // Header
            ui.label(egui::RichText::new("Isotope").strong());
            ui.label(egui::RichText::new("Peak E (eV)").strong());
            ui.label(egui::RichText::new("Peak SNR").strong());
            ui.label(egui::RichText::new("dT/ppm").strong());
            ui.label(egui::RichText::new("Verdict").strong());
            ui.end_row();

            for (name, report) in &state.detect_results {
                ui.label(name);
                ui.label(format!("{:.2}", report.peak_energy_ev));
                ui.label(format!("{:.2}", report.peak_snr));
                ui.label(format!("{:.2e}", report.peak_delta_t_per_ppm));
                if report.detectable {
                    ui.label(
                        egui::RichText::new("PASS")
                            .color(crate::theme::semantic::GREEN)
                            .strong(),
                    );
                } else {
                    ui.label(
                        egui::RichText::new("FAIL")
                            .color(crate::theme::semantic::RED)
                            .strong(),
                    );
                }
                ui.end_row();
            }
        });
}

/// Run detectability analysis for all trace candidates against the matrix.
fn run_detectability(state: &mut AppState) {
    use nereids_pipeline::detectability::{TraceDetectabilityConfig, trace_detectability};

    // Collect matrix isotopes with loaded ENDF data
    let matrix_isotopes: Vec<_> = state
        .detect_matrix_entries
        .iter()
        .filter_map(|e| e.resonance_data.clone().map(|rd| (rd, e.initial_density)))
        .collect();

    if matrix_isotopes.is_empty() {
        state.status_message = "Matrix isotope(s) need ENDF data".into();
        return;
    }

    // Validate energy range
    let e_min = state.detect_energy_min;
    let e_max = state.detect_energy_max;
    if e_min >= e_max {
        state.status_message = format!(
            "E_min ({:.3} eV) must be less than E_max ({:.3} eV)",
            e_min, e_max
        );
        return;
    }

    // Build log-spaced energy grid
    let n = state.detect_n_energy_points;
    let energies: Vec<f64> = (0..n)
        .map(|i| e_min * (e_max / e_min).powf(i as f64 / (n - 1).max(1) as f64))
        .collect();

    let config = TraceDetectabilityConfig {
        matrix_isotopes: &matrix_isotopes,
        energies: &energies,
        i0: state.detect_i0,
        temperature_k: state.detect_temperature_k,
        resolution: None,
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

    // Guard: all traces skipped (no data) but no computation errors
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

fn library_name(lib: EndfLibrary) -> &'static str {
    match lib {
        EndfLibrary::EndfB8_0 => "ENDF/B-VIII.0",
        EndfLibrary::EndfB8_1 => "ENDF/B-VIII.1",
        EndfLibrary::Jeff3_3 => "JEFF-3.3",
        EndfLibrary::Jendl5 => "JENDL-5",
    }
}

/// Fetch ENDF data for matrix + trace isotopes.
/// Index convention: 0..N = matrix entries, N.. = trace entries at (index - N).
fn detect_fetch_endf_data(state: &mut AppState) {
    use nereids_core::types::Isotope;
    use nereids_endf::retrieval;

    let n_matrix = state.detect_matrix_entries.len();
    let mut work: Vec<(usize, Isotope, String, EndfLibrary)> = Vec::new();

    // Matrix entries at indices 0..N
    for (i, entry) in state.detect_matrix_entries.iter().enumerate() {
        if entry.resonance_data.is_none() {
            match Isotope::new(entry.z, entry.a) {
                Ok(isotope) => {
                    if retrieval::mat_number(&isotope).is_some() {
                        work.push((i, isotope, entry.symbol.clone(), state.detect_endf_library));
                    } else {
                        state.status_message = format!(
                            "No MAT number for matrix {} — isotope not in database",
                            entry.symbol
                        );
                    }
                }
                Err(e) => {
                    state.status_message = format!(
                        "Matrix isotope Z={} A={} is not supported: {}",
                        entry.z, entry.a, e
                    );
                }
            }
        }
    }

    // Traces at indices N+
    for (i, entry) in state.detect_trace_entries.iter().enumerate() {
        if entry.resonance_data.is_none() {
            match Isotope::new(entry.z, entry.a) {
                Ok(isotope) => {
                    if retrieval::mat_number(&isotope).is_some() {
                        work.push((
                            n_matrix + i,
                            isotope,
                            entry.symbol.clone(),
                            state.detect_endf_library,
                        ));
                    } else {
                        state.status_message = format!(
                            "No MAT number for {} — isotope not in database",
                            entry.symbol
                        );
                    }
                }
                Err(e) => {
                    state.status_message = format!("Invalid isotope {}: {}", entry.symbol, e);
                }
            }
        }
    }

    if work.is_empty() {
        state.status_message =
            "No supported isotopes found — none have MAT numbers in the ENDF database".into();
        return;
    }

    // Mark matrix entries as Fetching before spawning the background thread
    for (idx, _, _, _) in &work {
        if *idx < n_matrix {
            state.detect_matrix_entries[*idx].endf_status = EndfStatus::Fetching;
        }
    }

    // Store matrix count at fetch time for poll_pending_tasks dispatch
    state.detect_n_matrix_at_fetch = n_matrix;

    let (tx, rx) = mpsc::channel();
    state.pending_detect_endf = Some(rx);
    state.is_fetching_detect_endf = true;
    state.status_message = "Fetching ENDF data (Detect)...".into();
    let cancel = Arc::clone(&state.cancel_token);

    std::thread::spawn(move || {
        let retriever = nereids_endf::retrieval::EndfRetriever::new();
        for (index, isotope, symbol, library) in work {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let Some(mat) = retrieval::mat_number(&isotope) else {
                continue;
            };
            let result = match retriever.get_endf_file(&isotope, library, mat) {
                Ok((_path, endf_text)) => {
                    match nereids_endf::parser::parse_endf_file2(&endf_text) {
                        Ok(data) => Ok(data),
                        Err(e) => Err(format!("Parse error for {}: {}", symbol, e)),
                    }
                }
                Err(e) => Err(format!("Fetch error for {}: {}", symbol, e)),
            };
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let _ = tx.send(EndfFetchResult {
                index,
                symbol,
                result,
            });
        }
    });
}

fn teleport_pill(ui: &mut egui::Ui, label: &str, target: GuidedStep, state: &mut AppState) {
    let accent = crate::theme::ThemeColors::from_ctx(ui.ctx()).accent;
    let btn = egui::Button::new(
        egui::RichText::new(label)
            .small()
            .color(egui::Color32::WHITE),
    )
    .fill(accent)
    .corner_radius(12.0);
    if ui.add(btn).clicked() {
        state.guided_step = target;
    }
}
