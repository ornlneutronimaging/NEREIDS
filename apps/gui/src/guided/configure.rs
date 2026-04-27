//! Step 2: Configuration — beamline parameters, isotope selection, ENDF fetch.

use crate::state::{AppState, EndfStatus, FetchTarget, GuidedStep, PeriodicTableTarget};
use crate::widgets::design::{self, ChipAction, NavAction};
use nereids_endf::retrieval::EndfLibrary;
use std::sync::Arc;
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
            design::teleport_pill(ui, "Forward Model →", GuidedStep::ForwardModel, state);
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

    // --- Instrument Resolution card ---
    let res = design::resolution_card(
        ui,
        &mut state.resolution_enabled,
        &mut state.resolution_mode,
        state.beamline.flight_path_m,
    );
    if res.changed {
        state.spatial_result = None;
        state.pixel_fit_result = None;
    }

    // --- Isotopes card ---
    // Custom header: title left, library ComboBox right
    let prev_lib = state.endf_library;
    design::card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Isotopes").size(14.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_enabled_ui(!state.is_fetching_endf, |ui| {
                    egui::ComboBox::from_id_salt("endf_lib")
                        .selected_text(design::library_name(state.endf_library))
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
                            ui.selectable_value(
                                &mut state.endf_library,
                                EndfLibrary::Tendl2023,
                                "TENDL-2023",
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
            for g in &mut state.isotope_groups {
                for m in &mut g.members {
                    m.resonance_data = None;
                    m.endf_status = EndfStatus::Pending;
                }
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
            ui.horizontal(|ui| {
                if ui.button("Add Isotope...").clicked() {
                    state.periodic_table_open = true;
                    state.periodic_table_target = PeriodicTableTarget::Configure;
                    state.periodic_table_selected_z = None;
                    state.periodic_table_density = 0.001; // at/barn default
                }
                if ui.button("Add Element...").clicked() {
                    state.periodic_table_open = true;
                    state.periodic_table_target = PeriodicTableTarget::ConfigureGroup;
                    state.periodic_table_selected_z = None;
                    state.periodic_table_density = 0.001; // at/barn default
                }
            });
        });

        // Auto-fetch ENDF data when isotopes are pending
        let has_pending = state
            .isotope_entries
            .iter()
            .any(|e| e.enabled && e.endf_status == EndfStatus::Pending)
            || state.isotope_groups.iter().any(|g| {
                g.enabled
                    && g.members
                        .iter()
                        .any(|m| m.endf_status == EndfStatus::Pending)
            });
        if has_pending && !state.is_fetching_endf {
            fetch_endf_data(state);
        }
        if state.is_fetching_endf {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Fetching ENDF data…");
            });
        }
        let has_failed = state
            .isotope_entries
            .iter()
            .any(|e| e.enabled && e.endf_status == EndfStatus::Failed)
            || state.isotope_groups.iter().any(|g| {
                g.enabled
                    && g.members
                        .iter()
                        .any(|m| m.endf_status == EndfStatus::Failed)
            });
        if has_failed && !state.is_fetching_endf && ui.button("Retry failed").clicked() {
            for e in &mut state.isotope_entries {
                if e.endf_status == EndfStatus::Failed {
                    e.endf_status = EndfStatus::Pending;
                }
            }
            for g in &mut state.isotope_groups {
                for m in &mut g.members {
                    if m.endf_status == EndfStatus::Failed {
                        m.endf_status = EndfStatus::Pending;
                    }
                }
            }
        }
    });

    // Density editing popup
    density_edit_window(ui, state);

    // --- Navigation buttons ---
    let has_enabled_iso = state.isotope_entries.iter().any(|e| e.enabled);
    let has_enabled_grp = state.isotope_groups.iter().any(|g| g.enabled);
    let has_enabled = has_enabled_iso || has_enabled_grp;
    let all_iso_loaded = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled)
        .all(|e| e.endf_status == EndfStatus::Loaded);
    let all_grp_loaded = state
        .isotope_groups
        .iter()
        .filter(|g| g.enabled)
        .all(|g| g.overall_status() == EndfStatus::Loaded);
    let all_loaded = all_iso_loaded && all_grp_loaded;
    let can_continue = has_enabled && all_loaded;
    let has_any_failed = state
        .isotope_entries
        .iter()
        .any(|e| e.enabled && e.endf_status == EndfStatus::Failed)
        || state.isotope_groups.iter().any(|g| {
            g.enabled
                && g.members
                    .iter()
                    .any(|m| m.endf_status == EndfStatus::Failed)
        });
    let nav_hint = if !has_enabled {
        "Add an isotope or element group to continue"
    } else if has_any_failed {
        "Some isotopes failed \u{2014} remove or retry"
    } else {
        "Waiting for ENDF data\u{2026}"
    };
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        can_continue,
        nav_hint,
    ) {
        NavAction::Back => state.nav_prev(),
        NavAction::Continue => state.nav_next(),
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
    let mut group_to_remove = None;
    let mut changed = false;
    let locked = state.is_fetching_endf;

    // Render chips in a dynamic grid that adapts to the available width.
    // egui's horizontal_wrapped doesn't wrap Frame-based widgets, so we
    // compute the number of columns from available width and chip size,
    // then use egui::Grid for a clean multi-row layout.
    let avail_w = ui.available_width();
    let chip_width = 170.0; // approximate width per chip including spacing
    let n_cols = ((avail_w / chip_width).floor() as usize).max(1);

    if locked {
        ui.disable();
    }

    let n_individual = state.isotope_entries.len();
    let n_groups = state.isotope_groups.len();
    let total = n_individual + n_groups;

    egui::Grid::new("isotope_chip_grid")
        .num_columns(n_cols)
        .spacing([6.0, 6.0])
        .show(ui, |ui| {
            // Individual isotope chips
            for (idx, entry) in state.isotope_entries.iter_mut().enumerate() {
                if idx > 0 && idx.is_multiple_of(n_cols) {
                    ui.end_row();
                }
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
            }

            // Group chips (continue same grid)
            for (gidx, group) in state.isotope_groups.iter_mut().enumerate() {
                let flat_idx = n_individual + gidx;
                if flat_idx > 0 && flat_idx.is_multiple_of(n_cols) {
                    ui.end_row();
                }
                let action = design::group_chip(
                    ui,
                    &group.name,
                    group.members.len(),
                    group.initial_density,
                    group.overall_status(),
                    group.enabled,
                    egui::Id::new(("grp_chip", gidx)),
                );
                match action {
                    ChipAction::Remove => {
                        group_to_remove = Some(gidx);
                    }
                    ChipAction::ToggleEnabled => {
                        group.enabled = !group.enabled;
                        changed = true;
                    }
                    ChipAction::None => {}
                }
            }
        });

    if let Some(idx) = to_remove {
        state.isotope_entries.remove(idx);
        changed = true;
    }
    if let Some(idx) = group_to_remove {
        state.isotope_groups.remove(idx);
        changed = true;
    }

    // Click density in a chip to open the editor
    // (Handled via right-click or double-click on the chip in the future;
    //  for now, provide a simple "Edit densities" button if entries exist.)
    if (total > 0) && !locked && ui.small_button("Edit densities...").clicked() {
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
            // Individual isotope densities
            for entry in state.isotope_entries.iter_mut() {
                ui.horizontal(|ui| {
                    ui.label(&entry.symbol);
                    ui.add(
                        egui::DragValue::new(&mut entry.initial_density)
                            .prefix("\u{03c1}\u{2080}=")
                            .speed(0.0001)
                            .range(0.0..=1.0),
                    );
                });
            }
            // Group densities
            if !state.isotope_groups.is_empty() && !state.isotope_entries.is_empty() {
                ui.separator();
            }
            for group in state.isotope_groups.iter_mut() {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&group.name).strong());
                    ui.add(
                        egui::DragValue::new(&mut group.initial_density)
                            .prefix("\u{03c1}\u{2080}=")
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

fn fetch_endf_data(state: &mut AppState) {
    use nereids_core::types::Isotope;
    use nereids_endf::retrieval;

    let mut work: Vec<design::EndfWorkItem> = Vec::new();
    let mut failed_indices: Vec<usize> = Vec::new();
    for (i, entry) in state.isotope_entries.iter().enumerate() {
        if entry.enabled && entry.endf_status == EndfStatus::Pending {
            let isotope = match Isotope::new(entry.z, entry.a) {
                Ok(iso) => iso,
                Err(e) => {
                    state.status_message = format!("Invalid isotope {}: {}", entry.symbol, e);
                    failed_indices.push(i);
                    continue;
                }
            };
            if retrieval::mat_number(&isotope, state.endf_library).is_none() {
                state.status_message = format!(
                    "No MAT number for {} — isotope not in database",
                    entry.symbol
                );
                failed_indices.push(i);
                continue;
            }
            work.push(design::EndfWorkItem {
                z: entry.z,
                a: entry.a,
                target: FetchTarget::Configure,
                isotope,
                symbol: entry.symbol.clone(),
                library: state.endf_library,
            });
        }
    }
    for i in failed_indices {
        state.isotope_entries[i].endf_status = EndfStatus::Failed;
    }

    // Also queue group members with Pending status
    let mut failed_group_members: Vec<(usize, usize)> = Vec::new();
    for (gi, group) in state.isotope_groups.iter().enumerate() {
        if !group.enabled {
            continue;
        }
        for (mi, member) in group.members.iter().enumerate() {
            if member.endf_status != EndfStatus::Pending {
                continue;
            }
            let isotope = match Isotope::new(group.z, member.a) {
                Ok(iso) => iso,
                Err(e) => {
                    state.status_message = format!("Invalid isotope {}: {}", member.symbol, e);
                    failed_group_members.push((gi, mi));
                    continue;
                }
            };
            if retrieval::mat_number(&isotope, state.endf_library).is_none() {
                state.status_message = format!(
                    "No MAT number for {} \u{2014} isotope not in database",
                    member.symbol
                );
                failed_group_members.push((gi, mi));
                continue;
            }
            work.push(design::EndfWorkItem {
                z: group.z,
                a: member.a,
                target: FetchTarget::Configure,
                isotope,
                symbol: member.symbol.clone(),
                library: state.endf_library,
            });
        }
    }
    for (gi, mi) in failed_group_members {
        state.isotope_groups[gi].members[mi].endf_status = EndfStatus::Failed;
    }

    if work.is_empty() {
        return;
    }

    // Mark entries as Fetching before spawning the background thread
    for item in &work {
        for entry in state.isotope_entries.iter_mut() {
            if entry.z == item.z && entry.a == item.a {
                entry.endf_status = EndfStatus::Fetching;
            }
        }
        for group in state.isotope_groups.iter_mut() {
            if group.z == item.z {
                for member in &mut group.members {
                    if member.a == item.a {
                        member.endf_status = EndfStatus::Fetching;
                    }
                }
            }
        }
    }

    let (tx, rx) = mpsc::channel();
    state.pending_endf = Some(rx);
    state.is_fetching_endf = true;
    state.status_message = "Fetching ENDF data...".into();
    let cancel = Arc::clone(&state.cancel_token);

    std::thread::spawn(move || design::endf_fetch_worker(work, cancel, tx));
}
