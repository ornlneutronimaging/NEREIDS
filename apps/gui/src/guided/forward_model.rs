//! Forward Model tool — independent isotope sandbox with live spectrum preview.

use crate::state::{
    AppState, EndfFetchResult, EndfStatus, GuidedStep, IsotopeEntry, PeriodicTableTarget,
    SpectrumAxis,
};
use egui_plot::{Line, Plot, PlotPoints};
use nereids_endf::retrieval::EndfLibrary;
use nereids_physics::transmission::{self, SampleParams};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Draw the Forward Model tool content.
pub fn forward_model_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.heading("Forward Model");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            teleport_pill(ui, "Analyze →", GuidedStep::Analyze, state);
            teleport_pill(ui, "← Configure", GuidedStep::Configure, state);
        });
    });
    ui.separator();

    ui.add_space(8.0);

    // Sync buttons row (disabled during active ENDF fetches to prevent index corruption)
    ui.horizontal(|ui| {
        ui.add_enabled_ui(!state.is_fetching_fm_endf, |ui| {
            if ui.button("Copy from Config").clicked() {
                state.fm_isotope_entries = state
                    .isotope_entries
                    .iter()
                    .map(|e| IsotopeEntry {
                        z: e.z,
                        a: e.a,
                        symbol: e.symbol.clone(),
                        initial_density: e.initial_density,
                        resonance_data: e.resonance_data.clone(),
                        enabled: e.enabled,
                        endf_status: e.endf_status,
                    })
                    .collect();
                state.fm_endf_library = state.endf_library;
                state.fm_temperature_k = state.temperature_k;
                state.fm_spectrum = None;
                state.fm_per_isotope_spectra.clear();
            }
        });
        ui.add_enabled_ui(!state.is_fetching_endf, |ui| {
            if ui.button("Push to Config").clicked() {
                state.isotope_entries = state
                    .fm_isotope_entries
                    .iter()
                    .map(|e| IsotopeEntry {
                        z: e.z,
                        a: e.a,
                        symbol: e.symbol.clone(),
                        initial_density: e.initial_density,
                        resonance_data: e.resonance_data.clone(),
                        enabled: e.enabled,
                        endf_status: if e.resonance_data.is_some() {
                            EndfStatus::Loaded
                        } else {
                            EndfStatus::Pending
                        },
                    })
                    .collect();
                state.endf_library = state.fm_endf_library;
                state.temperature_k = state.fm_temperature_k;
                // Invalidate stale fit results
                state.spatial_result = None;
                state.pixel_fit_result = None;
            }
        });
    });

    ui.add_space(8.0);

    // Two-column layout: controls (left) | spectrum plot (right)
    let available_width = ui.available_width();
    let controls_width = 300.0_f32.min(available_width * 0.4);

    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(controls_width, ui.available_height().max(400.0)),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("fm_controls")
                    .show(ui, |ui| {
                        fm_isotope_controls(ui, state);
                    });
            },
        );
        ui.separator();
        ui.vertical(|ui| {
            fm_spectrum_panel(ui, state);
        });
    });
}

/// Isotope table controls for the Forward Model (independent from Configure).
fn fm_isotope_controls(ui: &mut egui::Ui, state: &mut AppState) {
    // ENDF library selector (disabled during active fetch to prevent stale results)
    let prev_lib = state.fm_endf_library;
    ui.add_enabled_ui(!state.is_fetching_fm_endf, |ui| {
        ui.horizontal(|ui| {
            ui.label("Library:");
            egui::ComboBox::from_id_salt("fm_endf_lib")
                .selected_text(library_name(state.fm_endf_library))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.fm_endf_library,
                        EndfLibrary::EndfB8_0,
                        "ENDF/B-VIII.0",
                    );
                    ui.selectable_value(
                        &mut state.fm_endf_library,
                        EndfLibrary::EndfB8_1,
                        "ENDF/B-VIII.1",
                    );
                    ui.selectable_value(
                        &mut state.fm_endf_library,
                        EndfLibrary::Jeff3_3,
                        "JEFF-3.3",
                    );
                    ui.selectable_value(&mut state.fm_endf_library, EndfLibrary::Jendl5, "JENDL-5");
                });
        });
    });
    // Library change invalidates all resonance data — must re-fetch
    if state.fm_endf_library != prev_lib {
        for e in &mut state.fm_isotope_entries {
            e.resonance_data = None;
            e.endf_status = EndfStatus::Pending;
        }
        state.fm_spectrum = None;
        state.fm_per_isotope_spectra.clear();
    }

    // Temperature
    ui.horizontal(|ui| {
        ui.label("Temperature (K):");
        if ui
            .add(
                egui::DragValue::new(&mut state.fm_temperature_k)
                    .range(1.0..=5000.0)
                    .speed(1.0),
            )
            .changed()
        {
            state.fm_spectrum = None;
            state.fm_per_isotope_spectra.clear();
        }
    });

    ui.add_space(8.0);

    let isotope_locked = state.is_fetching_fm_endf;

    // Add isotope button
    ui.add_enabled_ui(!isotope_locked, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Add Isotope").clicked() {
                state.fm_isotope_entries.push(IsotopeEntry {
                    z: 92,
                    a: 238,
                    symbol: "U-238".into(),
                    initial_density: 0.001,
                    resonance_data: None,
                    enabled: true,
                    endf_status: EndfStatus::Pending,
                });
                state.fm_spectrum = None;
                state.fm_per_isotope_spectra.clear();
            }
            if ui.button("Periodic Table...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::ForwardModel;
                state.periodic_table_selected_z = None;
            }
        });
    });

    ui.add_space(4.0);

    // Isotope list
    let mut to_remove = None;
    let mut changed = false;
    for (idx, entry) in state.fm_isotope_entries.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!isotope_locked, |ui| {
                if ui.checkbox(&mut entry.enabled, "").changed() {
                    changed = true;
                }

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

                // Enforce Z <= A
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
                    changed = true;
                }
            });

            ui.label(&entry.symbol);

            ui.add_enabled_ui(!isotope_locked, |ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut entry.initial_density)
                            .prefix("rho=")
                            .speed(0.0001)
                            .range(0.0..=1.0),
                    )
                    .changed()
                {
                    changed = true;
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
        state.fm_isotope_entries.remove(idx);
        changed = true;
    }

    if changed {
        state.fm_spectrum = None;
        state.fm_per_isotope_spectra.clear();
    }

    // Fetch ENDF data
    let has_missing = state
        .fm_isotope_entries
        .iter()
        .any(|e| e.enabled && e.resonance_data.is_none());
    ui.add_enabled_ui(has_missing && !state.is_fetching_fm_endf, |ui| {
        if ui.button("Fetch ENDF Data").clicked() {
            fm_fetch_endf_data(state);
        }
    });
    if state.is_fetching_fm_endf {
        ui.spinner();
    }
}

/// Spectrum plot with energy/TOF axis toggle and per-isotope contribution lines.
fn fm_spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    // Axis toggle (Energy vs TOF)
    ui.horizontal(|ui| {
        ui.label("Axis:");
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

    // Collect enabled isotopes that have resonance data
    let enabled: Vec<_> = state
        .fm_isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .collect();

    if enabled.is_empty() {
        ui.label(
            egui::RichText::new("Add isotopes and fetch ENDF data to see the forward model.")
                .italics()
                .color(crate::theme::semantic::ORANGE),
        );
        return;
    }

    // Energy grid: use loaded energies if available, otherwise generate log-spaced 1..100 eV
    let energies: Vec<f64> = if let Some(ref e) = state.energies {
        e.clone()
    } else {
        let n = 2000;
        let e_min = 1.0_f64;
        let e_max = 100.0_f64;
        (0..n)
            .map(|i| e_min * (e_max / e_min).powf(i as f64 / (n - 1).max(1) as f64))
            .collect()
    };

    // Invalidate cache when energy grid changes (e.g. new data loaded or energies cleared)
    let current_grid = state.energies.as_deref();
    let cached_grid = state.fm_energies.as_deref();
    if current_grid != cached_grid {
        state.fm_spectrum = None;
        state.fm_per_isotope_spectra.clear();
        state.fm_energies = None;
    }

    // Compute combined forward model if cache is stale
    if state.fm_spectrum.is_none() {
        let isotopes: Vec<_> = enabled
            .iter()
            .filter_map(|e| e.resonance_data.clone().map(|rd| (rd, e.initial_density)))
            .collect();

        match SampleParams::new(state.fm_temperature_k, isotopes) {
            Ok(sample) => match transmission::forward_model(&energies, &sample, None) {
                Ok(combined) => {
                    state.fm_spectrum = Some(combined);
                }
                Err(e) => {
                    state.status_message = format!("Forward model error: {e}");
                    // Sentinel: empty vec stops recomputation every frame
                    state.fm_spectrum = Some(Vec::new());
                }
            },
            Err(e) => {
                state.status_message = format!("Sample params error: {e}");
                // Sentinel: empty vec stops recomputation every frame
                state.fm_spectrum = Some(Vec::new());
            }
        }

        // Compute per-isotope contributions (each isotope alone)
        state.fm_per_isotope_spectra.clear();
        for entry in &enabled {
            let Some(rd) = entry.resonance_data.clone() else {
                continue;
            };
            let single = vec![(rd, entry.initial_density)];
            match SampleParams::new(state.fm_temperature_k, single) {
                Ok(sample) => match transmission::forward_model(&energies, &sample, None) {
                    Ok(t) => {
                        state.fm_per_isotope_spectra.push((entry.symbol.clone(), t));
                    }
                    Err(e) => {
                        state.status_message =
                            format!("Forward model error for {}: {e}", entry.symbol);
                        // Skip this isotope but don't leave cache in recomputation state
                    }
                },
                Err(e) => {
                    state.status_message = format!("Sample params error for {}: {e}", entry.symbol);
                    // Skip this isotope but don't leave cache in recomputation state
                }
            }
        }

        state.fm_energies = state.energies.clone();
    }

    // Use the cached energy grid for plotting
    let plot_energies = state.fm_energies.as_ref().unwrap_or(&energies);

    // Build x-axis values based on axis selection
    let (x_values, x_label): (Vec<f64>, &str) = match state.fm_spectrum_axis {
        SpectrumAxis::EnergyEv => (plot_energies.clone(), "Energy (eV)"),
        SpectrumAxis::TofMicroseconds => {
            let fp = state.beamline.flight_path_m;
            if fp.is_finite() && fp > 0.0 {
                let delay = state.beamline.delay_us;
                let tof: Vec<f64> = plot_energies
                    .iter()
                    .map(|&e| {
                        if e > 0.0 {
                            nereids_core::constants::energy_to_tof(e, fp) + delay
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                (tof, "TOF (\u{03bc}s)")
            } else {
                // Fallback to energy if flight path is not configured
                (plot_energies.clone(), "Energy (eV)")
            }
        }
    };

    // Distinct colors for per-isotope lines
    let isotope_colors = [
        egui::Color32::from_rgb(230, 100, 50), // orange
        egui::Color32::from_rgb(50, 180, 50),  // green
        egui::Color32::from_rgb(180, 50, 180), // purple
        egui::Color32::from_rgb(50, 150, 220), // blue
        egui::Color32::from_rgb(220, 50, 50),  // red
        egui::Color32::from_rgb(50, 200, 200), // cyan
        egui::Color32::from_rgb(200, 200, 50), // yellow
        egui::Color32::from_rgb(150, 100, 50), // brown
    ];

    Plot::new("fm_spectrum_plot")
        .x_axis_label(x_label)
        .y_axis_label("Transmission")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            // Combined transmission line
            if let Some(ref combined) = state.fm_spectrum {
                let n_plot = x_values.len().min(combined.len());
                let points: PlotPoints = (0..n_plot)
                    .filter(|&i| x_values[i].is_finite())
                    .map(|i| [x_values[i], combined[i]])
                    .collect();
                plot_ui.line(Line::new("Combined T(E)", points).width(2.0));
            }

            // Per-isotope contribution lines (dashed)
            for (idx, (symbol, spectrum)) in state.fm_per_isotope_spectra.iter().enumerate() {
                let n_plot = x_values.len().min(spectrum.len());
                let points: PlotPoints = (0..n_plot)
                    .filter(|&i| x_values[i].is_finite())
                    .map(|i| [x_values[i], spectrum[i]])
                    .collect();
                let color = isotope_colors[idx % isotope_colors.len()];
                plot_ui.line(
                    Line::new(symbol.as_str(), points)
                        .color(color)
                        .style(egui_plot::LineStyle::dashed_loose()),
                );
            }
        });
}

fn library_name(lib: EndfLibrary) -> &'static str {
    match lib {
        EndfLibrary::EndfB8_0 => "ENDF/B-VIII.0",
        EndfLibrary::EndfB8_1 => "ENDF/B-VIII.1",
        EndfLibrary::Jeff3_3 => "JEFF-3.3",
        EndfLibrary::Jendl5 => "JENDL-5",
    }
}

fn fm_fetch_endf_data(state: &mut AppState) {
    use nereids_core::types::Isotope;
    use nereids_endf::retrieval;

    let mut work: Vec<(usize, Isotope, String, EndfLibrary)> = Vec::new();
    for (i, entry) in state.fm_isotope_entries.iter().enumerate() {
        if entry.enabled && entry.resonance_data.is_none() {
            let isotope = match Isotope::new(entry.z, entry.a) {
                Ok(iso) => iso,
                Err(e) => {
                    state.status_message = format!("Invalid isotope {}: {}", entry.symbol, e);
                    continue;
                }
            };
            if retrieval::mat_number(&isotope).is_none() {
                state.status_message = format!(
                    "No MAT number for {} — isotope not in database",
                    entry.symbol
                );
                continue;
            }
            work.push((i, isotope, entry.symbol.clone(), state.fm_endf_library));
        }
    }

    if work.is_empty() {
        return;
    }

    // Mark entries as Fetching before spawning the background thread
    for (i, _, _, _) in &work {
        if let Some(entry) = state.fm_isotope_entries.get_mut(*i) {
            entry.endf_status = EndfStatus::Fetching;
        }
    }

    let (tx, rx) = mpsc::channel();
    state.pending_fm_endf = Some(rx);
    state.is_fetching_fm_endf = true;
    state.status_message = "Fetching ENDF data (FM)...".into();
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
