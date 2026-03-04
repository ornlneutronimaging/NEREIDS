//! Step 2: Configuration — beamline parameters, isotope selection, ENDF fetch.

use crate::state::{AppState, EndfFetchResult, EndfStatus, GuidedStep, PeriodicTableTarget};
use crate::widgets::design::{self, ChipAction, NavAction};
use nereids_endf::retrieval::EndfLibrary;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Draw the Configure step content.
pub fn configure_step(ui: &mut egui::Ui, state: &mut AppState) {
    // Content header with teleport pill
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            design::content_header(
                ui,
                "Configure",
                "Set beamline parameters and select isotopes",
            );
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            teleport_pill(ui, "Forward Model →", GuidedStep::ForwardModel, state);
        });
    });

    // --- Beamline Parameters card ---
    design::card_with_header(ui, "Beamline Parameters", None, |ui| {
        egui::Grid::new("beamline_grid")
            .num_columns(4)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("Flight Path (m):");
                ui.add(
                    egui::DragValue::new(&mut state.beamline.flight_path_m)
                        .range(1.0..=100.0)
                        .speed(0.1),
                );
                ui.label("Delay (μs):");
                ui.add(
                    egui::DragValue::new(&mut state.beamline.delay_us)
                        .range(0.0..=1000.0)
                        .speed(0.1),
                );
                ui.end_row();

                ui.label("Temperature (K):");
                ui.add(
                    egui::DragValue::new(&mut state.temperature_k)
                        .range(1.0..=5000.0)
                        .speed(1.0),
                );
                ui.label("PC sample:");
                ui.add(
                    egui::DragValue::new(&mut state.proton_charge_sample)
                        .range(0.001..=1e6)
                        .speed(0.01),
                );
                ui.end_row();

                ui.label("PC open beam:");
                ui.add(
                    egui::DragValue::new(&mut state.proton_charge_ob)
                        .range(0.001..=1e6)
                        .speed(0.01),
                );
                ui.end_row();
            });
    });

    // --- Isotopes card ---
    // Custom header: title left, library ComboBox right
    let prev_lib = state.endf_library;
    design::card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Isotopes").size(14.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_enabled_ui(!state.is_fetching_endf, |ui| {
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
                            ui.selectable_value(
                                &mut state.endf_library,
                                EndfLibrary::Jeff3_3,
                                "JEFF-3.3",
                            );
                            ui.selectable_value(
                                &mut state.endf_library,
                                EndfLibrary::Jendl5,
                                "JENDL-5",
                            );
                        });
                });
            });
        });
        // Library change invalidates all resonance data and stale results
        if state.endf_library != prev_lib {
            for e in &mut state.isotope_entries {
                e.resonance_data = None;
                e.endf_status = EndfStatus::Pending;
            }
            state.spatial_result = None;
            state.pixel_fit_result = None;
        }

        ui.add_space(8.0);

        // Isotope chips
        let chip_result = isotope_chips_flow(ui, state);
        if chip_result.changed {
            state.spatial_result = None;
            state.pixel_fit_result = None;
        }

        ui.add_space(6.0);

        // Add buttons (disabled during fetch)
        ui.add_enabled_ui(!state.is_fetching_endf, |ui| {
            if ui.button("Add Isotope...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::Configure;
                state.periodic_table_selected_z = None;
                state.periodic_table_density = 0.001; // at/barn default
            }
        });

        // Fetch ENDF button
        let has_missing = state
            .isotope_entries
            .iter()
            .any(|e| e.enabled && e.endf_status != EndfStatus::Loaded);
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_enabled_ui(has_missing && !state.is_fetching_endf, |ui| {
                if ui.button("Fetch ENDF Data").clicked() {
                    fetch_endf_data(state);
                }
            });
            if state.is_fetching_endf {
                ui.spinner();
            }
        });
    });

    // Density editing popup
    density_edit_window(ui, state);

    // --- Navigation buttons ---
    let has_enabled = state.isotope_entries.iter().any(|e| e.enabled);
    let all_loaded = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled)
        .all(|e| e.endf_status == EndfStatus::Loaded);
    let can_continue = has_enabled && all_loaded;
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        can_continue,
        "Fetch ENDF data to continue",
    ) {
        NavAction::Back => state.guided_step = GuidedStep::Load,
        NavAction::Continue => state.guided_step = GuidedStep::Normalize,
        NavAction::None => {}
    }
}

// ---------------------------------------------------------------------------
// Isotope chips flow
// ---------------------------------------------------------------------------

struct ChipFlowResult {
    changed: bool,
}

fn isotope_chips_flow(ui: &mut egui::Ui, state: &mut AppState) -> ChipFlowResult {
    let mut to_remove = None;
    let mut changed = false;
    let locked = state.is_fetching_endf;

    ui.horizontal_wrapped(|ui| {
        for (idx, entry) in state.isotope_entries.iter_mut().enumerate() {
            ui.add_enabled_ui(!locked, |ui| {
                let action = design::isotope_chip(
                    ui,
                    &entry.symbol,
                    entry.initial_density,
                    entry.endf_status,
                    entry.enabled,
                    egui::Id::new(("iso_chip", idx)),
                );
                match action {
                    ChipAction::Remove => {
                        to_remove = Some(idx);
                    }
                    ChipAction::ToggleEnabled => {
                        entry.enabled = !entry.enabled;
                        changed = true;
                    }
                    ChipAction::None => {}
                }
            });
        }
    });

    if let Some(idx) = to_remove {
        state.isotope_entries.remove(idx);
        changed = true;
    }

    // Click density in a chip to open the editor
    // (Handled via right-click or double-click on the chip in the future;
    //  for now, provide a simple "Edit densities" button if entries exist.)
    if !state.isotope_entries.is_empty()
        && !locked
        && ui.small_button("Edit densities...").clicked()
    {
        state.editing_isotope_density = Some(0);
    }

    ChipFlowResult { changed }
}

// ---------------------------------------------------------------------------
// Density editing window
// ---------------------------------------------------------------------------

fn density_edit_window(ui: &mut egui::Ui, state: &mut AppState) {
    if state.editing_isotope_density.is_none() {
        return;
    }
    // Auto-close if a fetch started while the window was open
    if state.is_fetching_endf {
        state.editing_isotope_density = None;
        return;
    }

    let mut close = false;
    let mut open = true;
    egui::Window::new("Edit Densities")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ui.ctx(), |ui| {
            for entry in state.isotope_entries.iter_mut() {
                ui.horizontal(|ui| {
                    ui.label(&entry.symbol);
                    ui.add(
                        egui::DragValue::new(&mut entry.initial_density)
                            .prefix("ρ₀=")
                            .speed(0.0001)
                            .range(0.0..=1.0),
                    );
                });
            }
            ui.add_space(4.0);
            if ui.button("Done").clicked() {
                close = true;
            }
        });

    if !open || close {
        state.editing_isotope_density = None;
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
