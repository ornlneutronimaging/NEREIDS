//! Step 2: Configuration — beamline parameters, isotope selection, ENDF fetch.

use crate::state::{
    AppState, EndfFetchResult, EndfStatus, GuidedStep, IsotopeEntry, PeriodicTableTarget,
};
use nereids_endf::retrieval::EndfLibrary;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Draw the Configure step content.
pub fn configure_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.heading("Configure");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            teleport_pill(ui, "Forward Model →", GuidedStep::ForwardModel, state);
        });
    });
    ui.separator();

    // --- Beamline parameters ---
    ui.label(egui::RichText::new("Beamline Parameters").strong());
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Flight path (m):");
        ui.add(
            egui::DragValue::new(&mut state.beamline.flight_path_m)
                .range(1.0..=100.0)
                .speed(0.1),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Delay (us):");
        ui.add(
            egui::DragValue::new(&mut state.beamline.delay_us)
                .range(0.0..=1000.0)
                .speed(0.1),
        );
    });
    ui.horizontal(|ui| {
        ui.label("PC sample:");
        ui.add(
            egui::DragValue::new(&mut state.proton_charge_sample)
                .range(0.001..=1e6)
                .speed(0.01),
        );
    });
    ui.horizontal(|ui| {
        ui.label("PC open beam:");
        ui.add(
            egui::DragValue::new(&mut state.proton_charge_ob)
                .range(0.001..=1e6)
                .speed(0.01),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Temperature (K):");
        ui.add(
            egui::DragValue::new(&mut state.temperature_k)
                .range(1.0..=5000.0)
                .speed(1.0),
        );
    });

    ui.add_space(12.0);
    ui.separator();

    // --- Isotope selection ---
    ui.label(egui::RichText::new("Isotopes").strong());
    ui.add_space(4.0);

    // ENDF library selector (disabled during active fetch to prevent stale results)
    let prev_lib = state.endf_library;
    ui.add_enabled_ui(!state.is_fetching_endf, |ui| {
        ui.horizontal(|ui| {
            ui.label("Library:");
            egui::ComboBox::from_id_salt("endf_lib")
                .selected_text(library_name(state.endf_library))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.endf_library,
                        EndfLibrary::EndfB8_0,
                        "ENDF/B-VIII.0",
                    );
                    ui.selectable_value(
                        &mut state.endf_library,
                        EndfLibrary::EndfB8_1,
                        "ENDF/B-VIII.1",
                    );
                    ui.selectable_value(&mut state.endf_library, EndfLibrary::Jeff3_3, "JEFF-3.3");
                    ui.selectable_value(&mut state.endf_library, EndfLibrary::Jendl5, "JENDL-5");
                });
        });
    });
    // Library change invalidates all resonance data — must re-fetch
    if state.endf_library != prev_lib {
        for e in &mut state.isotope_entries {
            e.resonance_data = None;
            e.endf_status = EndfStatus::Pending;
        }
    }

    ui.add_space(4.0);

    // Disable isotope add/remove/edit while ENDF fetch is in progress
    let isotope_locked = state.is_fetching_endf;

    // Add isotope row
    ui.add_enabled_ui(!isotope_locked, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Add Isotope").clicked() {
                state.isotope_entries.push(IsotopeEntry {
                    z: 92,
                    a: 238,
                    symbol: "U-238".into(),
                    initial_density: 0.001,
                    resonance_data: None,
                    enabled: true,
                    endf_status: EndfStatus::Pending,
                });
                // Invalidate stale results — isotope order may have changed.
                state.spatial_result = None;
                state.pixel_fit_result = None;
            }
            if ui.button("Periodic Table...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::Configure;
                state.periodic_table_selected_z = None;
            }
        });
    });

    // Isotope list
    let mut to_remove = None;
    let mut isotope_changed = false;
    for (idx, entry) in state.isotope_entries.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!isotope_locked, |ui| {
                let enabled_before = entry.enabled;
                ui.checkbox(&mut entry.enabled, "");
                if entry.enabled != enabled_before {
                    isotope_changed = true;
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

                // Enforce Z <= A (physical constraint: protons cannot exceed nucleons).
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
                    isotope_changed = true;
                }
            });

            ui.label(&entry.symbol);

            ui.add_enabled_ui(!isotope_locked, |ui| {
                ui.add(
                    egui::DragValue::new(&mut entry.initial_density)
                        .prefix("rho0=")
                        .speed(0.0001)
                        .range(0.0..=1.0),
                );
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
        state.isotope_entries.remove(idx);
        isotope_changed = true;
    }

    // Invalidate stale results when isotope entries change (enabled toggled,
    // Z/A edited, or entry removed).  density_maps are indexed positionally
    // by the isotope order at compute time, so any change invalidates them.
    if isotope_changed {
        state.spatial_result = None;
        state.pixel_fit_result = None;
    }

    // Fetch ENDF data for isotopes missing resonance data
    let has_missing = state
        .isotope_entries
        .iter()
        .any(|e| e.enabled && e.resonance_data.is_none());
    ui.add_enabled_ui(has_missing && !state.is_fetching_endf, |ui| {
        if ui.button("Fetch ENDF Data").clicked() {
            fetch_endf_data(state);
        }
    });
    if state.is_fetching_endf {
        ui.spinner();
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

fn fetch_endf_data(state: &mut AppState) {
    use nereids_core::types::Isotope;
    use nereids_endf::retrieval;

    let mut work: Vec<(usize, Isotope, String, EndfLibrary)> = Vec::new();
    for (i, entry) in state.isotope_entries.iter().enumerate() {
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
            work.push((i, isotope, entry.symbol.clone(), state.endf_library));
        }
    }

    if work.is_empty() {
        return;
    }

    // Mark entries as Fetching before spawning the background thread
    for (i, _, _, _) in &work {
        if let Some(entry) = state.isotope_entries.get_mut(*i) {
            entry.endf_status = EndfStatus::Fetching;
        }
    }

    let (tx, rx) = mpsc::channel();
    state.pending_endf = Some(rx);
    state.is_fetching_endf = true;
    state.status_message = "Fetching ENDF data...".into();
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
