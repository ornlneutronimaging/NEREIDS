//! Periodic Table modal widget for interactive element/isotope selection.
//!
//! Renders a standard 18-column periodic table as a grid of colored buttons.
//! When an element is selected, its natural isotopes are shown as clickable
//! chips that add the isotope to the appropriate target list (Configure,
//! Forward Model, or Detectability).

use crate::state::{
    AppState, DetectTraceEntry, EndfStatus, GroupMemberState, IsotopeEntry, IsotopeGroupEntry,
    PeriodicTableTarget,
};
use egui::{Color32, CornerRadius};
use nereids_core::types::Isotope;
use nereids_endf::retrieval::{self, EndfLibrary};

// ---------------------------------------------------------------------------
// Element categories for color coding
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum ElementCategory {
    AlkaliMetal,
    AlkalineEarth,
    TransitionMetal,
    PostTransitionMetal,
    Metalloid,
    NonMetal,
    Halogen,
    NobleGas,
    Lanthanide,
    Actinide,
}

impl ElementCategory {
    fn color(self) -> Color32 {
        match self {
            Self::AlkaliMetal => Color32::from_rgb(255, 200, 200),
            Self::AlkalineEarth => Color32::from_rgb(255, 222, 173),
            Self::TransitionMetal => Color32::from_rgb(255, 255, 200),
            Self::PostTransitionMetal => Color32::from_rgb(200, 230, 200),
            Self::Metalloid => Color32::from_rgb(200, 200, 255),
            Self::NonMetal => Color32::from_rgb(180, 230, 255),
            Self::Halogen => Color32::from_rgb(200, 255, 255),
            Self::NobleGas => Color32::from_rgb(230, 200, 255),
            Self::Lanthanide => Color32::from_rgb(255, 180, 220),
            Self::Actinide => Color32::from_rgb(255, 200, 160),
        }
    }
}

fn element_category(z: u32) -> ElementCategory {
    use ElementCategory::*;
    match z {
        // Alkali metals
        3 | 11 | 19 | 37 | 55 | 87 => AlkaliMetal,
        // Alkaline earth metals
        4 | 12 | 20 | 38 | 56 | 88 => AlkalineEarth,
        // Transition metals (3d, 4d, 5d, 6d blocks minus post-transition/metalloid)
        21..=30 | 39..=48 | 72..=80 | 104..=112 => TransitionMetal,
        // Post-transition metals
        13 | 31 | 49 | 50 | 81 | 82 | 83 | 113 | 114 | 115 | 116 => PostTransitionMetal,
        // Metalloids
        5 | 14 | 32 | 33 | 51 | 52 | 84 => Metalloid,
        // Halogens
        9 | 17 | 35 | 53 | 85 | 117 => Halogen,
        // Noble gases
        2 | 10 | 18 | 36 | 54 | 86 | 118 => NobleGas,
        // Lanthanides
        57..=71 => Lanthanide,
        // Actinides
        89..=103 => Actinide,
        // Remaining nonmetals (H, C, N, O, P, S, Se)
        1 | 6 | 7 | 8 | 15 | 16 | 34 => NonMetal,
        _ => TransitionMetal,
    }
}

// ---------------------------------------------------------------------------
// Static layout data: (Z, row, col) for the standard 18-column periodic table
// ---------------------------------------------------------------------------

/// Periodic table positions: (atomic_number, row, column).
/// Row 7 is an empty gap between the main table and the f-block rows.
/// Row 8 = lanthanides (La-Lu), row 9 = actinides (Ac-Lr).
const ELEMENT_POSITIONS: [(u32, u32, u32); 118] = [
    // Row 0
    (1, 0, 0),
    (2, 0, 17),
    // Row 1 (period 2)
    (3, 1, 0),
    (4, 1, 1),
    (5, 1, 12),
    (6, 1, 13),
    (7, 1, 14),
    (8, 1, 15),
    (9, 1, 16),
    (10, 1, 17),
    // Row 2 (period 3)
    (11, 2, 0),
    (12, 2, 1),
    (13, 2, 12),
    (14, 2, 13),
    (15, 2, 14),
    (16, 2, 15),
    (17, 2, 16),
    (18, 2, 17),
    // Row 3 (period 4: K-Kr, includes 3d block)
    (19, 3, 0),
    (20, 3, 1),
    (21, 3, 2),
    (22, 3, 3),
    (23, 3, 4),
    (24, 3, 5),
    (25, 3, 6),
    (26, 3, 7),
    (27, 3, 8),
    (28, 3, 9),
    (29, 3, 10),
    (30, 3, 11),
    (31, 3, 12),
    (32, 3, 13),
    (33, 3, 14),
    (34, 3, 15),
    (35, 3, 16),
    (36, 3, 17),
    // Row 4 (period 5: Rb-Xe, includes 4d block)
    (37, 4, 0),
    (38, 4, 1),
    (39, 4, 2),
    (40, 4, 3),
    (41, 4, 4),
    (42, 4, 5),
    (43, 4, 6),
    (44, 4, 7),
    (45, 4, 8),
    (46, 4, 9),
    (47, 4, 10),
    (48, 4, 11),
    (49, 4, 12),
    (50, 4, 13),
    (51, 4, 14),
    (52, 4, 15),
    (53, 4, 16),
    (54, 4, 17),
    // Row 5 (period 6: Cs, Ba, then 5d block Hf-Hg, then Tl-Rn)
    // La-Lu go to row 8 (lanthanides)
    (55, 5, 0),
    (56, 5, 1),
    (72, 5, 3),
    (73, 5, 4),
    (74, 5, 5),
    (75, 5, 6),
    (76, 5, 7),
    (77, 5, 8),
    (78, 5, 9),
    (79, 5, 10),
    (80, 5, 11),
    (81, 5, 12),
    (82, 5, 13),
    (83, 5, 14),
    (84, 5, 15),
    (85, 5, 16),
    (86, 5, 17),
    // Row 6 (period 7: Fr, Ra, then 6d block Rf-Cn, then Nh-Og)
    // Ac-Lr go to row 9 (actinides)
    (87, 6, 0),
    (88, 6, 1),
    (104, 6, 3),
    (105, 6, 4),
    (106, 6, 5),
    (107, 6, 6),
    (108, 6, 7),
    (109, 6, 8),
    (110, 6, 9),
    (111, 6, 10),
    (112, 6, 11),
    (113, 6, 12),
    (114, 6, 13),
    (115, 6, 14),
    (116, 6, 15),
    (117, 6, 16),
    (118, 6, 17),
    // Row 8: Lanthanides La(57)-Lu(71) at cols 3-17
    (57, 8, 3),
    (58, 8, 4),
    (59, 8, 5),
    (60, 8, 6),
    (61, 8, 7),
    (62, 8, 8),
    (63, 8, 9),
    (64, 8, 10),
    (65, 8, 11),
    (66, 8, 12),
    (67, 8, 13),
    (68, 8, 14),
    (69, 8, 15),
    (70, 8, 16),
    (71, 8, 17),
    // Row 9: Actinides Ac(89)-Lr(103) at cols 3-17
    (89, 9, 3),
    (90, 9, 4),
    (91, 9, 5),
    (92, 9, 6),
    (93, 9, 7),
    (94, 9, 8),
    (95, 9, 9),
    (96, 9, 10),
    (97, 9, 11),
    (98, 9, 12),
    (99, 9, 13),
    (100, 9, 14),
    (101, 9, 15),
    (102, 9, 16),
    (103, 9, 17),
];

// ---------------------------------------------------------------------------
// Grid lookup helper
// ---------------------------------------------------------------------------

/// Look up which element is at a given (row, col), if any.
fn element_at(row: u32, col: u32) -> Option<u32> {
    ELEMENT_POSITIONS
        .iter()
        .find(|(_, r, c)| *r == row && *c == col)
        .map(|(z, _, _)| *z)
}

// ---------------------------------------------------------------------------
// Main modal function
// ---------------------------------------------------------------------------

/// Render the periodic table as a modal window overlay.
///
/// Does nothing if `state.periodic_table_open` is `false`.
pub fn periodic_table_modal(ctx: &egui::Context, state: &mut AppState) {
    if !state.periodic_table_open {
        return;
    }

    let accent = crate::theme::ThemeColors::from_ctx(ctx).accent;
    let mut open = true;
    egui::Window::new("Periodic Table")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            // --- Element grid ---
            let prev_z = state.periodic_table_selected_z;
            egui::Grid::new("pt_grid")
                .spacing([2.0, 2.0])
                .show(ui, |ui| {
                    for row in 0..=9 {
                        if row == 7 {
                            ui.add_space(6.0);
                            ui.end_row();
                            continue;
                        }

                        for col in 0..18 {
                            if let Some(z) = element_at(row, col) {
                                let sym = nereids_core::elements::element_symbol(z).unwrap_or("??");
                                let cat = element_category(z);
                                let bg = cat.color();

                                let is_selected = state.periodic_table_selected_z == Some(z);
                                let btn = egui::Button::new(
                                    egui::RichText::new(sym).size(10.0).color(Color32::BLACK),
                                )
                                .min_size(egui::vec2(28.0, 28.0))
                                .fill(if is_selected {
                                    Color32::from_rgb(100, 149, 237)
                                } else {
                                    bg
                                });

                                if ui.add(btn).clicked() {
                                    state.periodic_table_selected_z = Some(z);
                                }
                            } else if row == 5 && col == 2 {
                                ui.label(egui::RichText::new("*").size(10.0).strong());
                            } else if row == 6 && col == 2 {
                                ui.label(egui::RichText::new("**").size(10.0).strong());
                            } else if row == 8 && col == 2 {
                                ui.label(egui::RichText::new("*").size(10.0).strong());
                            } else if row == 9 && col == 2 {
                                ui.label(egui::RichText::new("**").size(10.0).strong());
                            } else {
                                ui.allocate_space(egui::vec2(28.0, 28.0));
                            }
                        }
                        ui.end_row();
                    }
                });

            // Element change clears isotope selection
            if state.periodic_table_selected_z != prev_z {
                state.periodic_table_selected_isotopes.clear();
            }

            ui.add_space(8.0);
            ui.separator();

            // --- ConfigureGroup: element-level group add (no individual isotope selection) ---
            if state.periodic_table_target == PeriodicTableTarget::ConfigureGroup {
                if let Some(z) = state.periodic_table_selected_z {
                    let name = nereids_core::elements::element_name(z).unwrap_or("Unknown");
                    let sym = nereids_core::elements::element_symbol(z).unwrap_or("??");
                    ui.label(
                        egui::RichText::new(format!("Z={z}  {sym} - {name}"))
                            .strong()
                            .size(14.0),
                    );

                    // Resolve the library this modal is filtering against so we can
                    // gate group construction on library coverage.  A "natural group"
                    // implicitly claims natural composition; if any natural member is
                    // missing in the selected library, building the truncated group
                    // would either renormalize partial ratios (changing the physics
                    // — pure Cd-113 ≠ natural Cd) or auto-fail at fetch time.  Refuse
                    // up front and tell the user instead.
                    let lib: EndfLibrary = *state
                        .periodic_table_library
                        .get_or_insert(target_library(state));
                    let known_a: std::collections::HashSet<u32> =
                        retrieval::known_isotopes_for(z, lib).into_iter().collect();
                    let natural = nereids_core::elements::natural_isotopes(z);
                    let missing_a: Vec<u32> = natural
                        .iter()
                        .map(|(iso, _)| iso.a())
                        .filter(|a| !known_a.contains(a))
                        .collect();
                    let lib_label = super::design::library_name(lib);

                    if natural.is_empty() {
                        ui.label("No natural isotopes for this element.");
                    } else {
                        ui.label("Natural isotopes in group:");
                        let parts: Vec<String> = natural
                            .iter()
                            .map(|(iso, frac)| format!("{sym}-{} ({:.2}%)", iso.a(), frac * 100.0))
                            .collect();
                        ui.label(parts.join(", "));

                        if !missing_a.is_empty() {
                            let missing_list: Vec<String> =
                                missing_a.iter().map(|a| format!("{sym}-{a}")).collect();
                            ui.add_space(2.0);
                            ui.label(
                                egui::RichText::new(format!(
                                    "⚠ {lib_label} is missing {}: {}. \
                                     Switch library or add isotopes individually.",
                                    if missing_a.len() == 1 {
                                        "1 natural isotope"
                                    } else {
                                        "natural isotopes"
                                    },
                                    missing_list.join(", "),
                                ))
                                .color(Color32::from_rgb(0xC0, 0x80, 0x00))
                                .size(11.0),
                            );
                        }

                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Density (at/barn):");
                            ui.add(
                                egui::DragValue::new(&mut state.periodic_table_density)
                                    .speed(0.0001)
                                    .range(1e-6..=1.0)
                                    .max_decimals(6),
                            );

                            let fetching = state.is_fetching_endf;
                            ui.add_enabled_ui(!fetching, |ui| {
                                let lib = state
                                    .periodic_table_library
                                    .get_or_insert(target_library(state));
                                egui::ComboBox::from_id_salt("pt_lib")
                                    .selected_text(super::design::library_name(*lib))
                                    .show_ui(ui, |ui| {
                                        for (val, label) in [
                                            (EndfLibrary::EndfB8_0, "ENDF/B-VIII.0"),
                                            (EndfLibrary::EndfB8_1, "ENDF/B-VIII.1"),
                                            (EndfLibrary::Jeff3_3, "JEFF-3.3"),
                                            (EndfLibrary::Jendl5, "JENDL-5"),
                                            (EndfLibrary::Tendl2023, "TENDL-2023"),
                                            (EndfLibrary::Cendl3_2, "CENDL-3.2"),
                                        ] {
                                            ui.selectable_value(lib, val, label);
                                        }
                                    });
                            });

                            let already = state.isotope_groups.iter().any(|g| g.z == z);
                            let covered = missing_a.is_empty();
                            if ui
                                .add_enabled(
                                    !already && covered,
                                    egui::Button::new(
                                        egui::RichText::new(format!("Add {sym} Group"))
                                            .color(Color32::WHITE),
                                    )
                                    .fill(accent),
                                )
                                .clicked()
                            {
                                add_element_group(state, z);
                            }
                            if already {
                                ui.label(
                                    egui::RichText::new("(already added)")
                                        .size(10.0)
                                        .color(Color32::GRAY),
                                );
                            } else if !covered {
                                ui.label(
                                    egui::RichText::new(format!("(not in {lib_label})"))
                                        .size(10.0)
                                        .color(Color32::GRAY),
                                );
                            }
                        });
                    }
                } else {
                    ui.label("Click an element to add it as a natural isotope group.");
                }
            } else {
                // --- Standard isotope selection (all other targets) ---
                // Resolve the library this modal is filtering against.  The
                // get_or_insert pattern matches the dropdown sites below; the
                // dereference releases the &mut borrow before we read state
                // for natural_isotopes / known_isotopes_for.
                let lib: EndfLibrary = {
                    let target = target_library(state);
                    *state.periodic_table_library.get_or_insert(target)
                };
                if let Some(z) = state.periodic_table_selected_z {
                    let name = nereids_core::elements::element_name(z).unwrap_or("Unknown");
                    let sym = nereids_core::elements::element_symbol(z).unwrap_or("??");
                    ui.label(
                        egui::RichText::new(format!("Z={z}  {sym} - {name}"))
                            .strong()
                            .size(14.0),
                    );

                    // Filter natural isotopes by what the selected library actually
                    // covers — otherwise the user gets clickable chips that fail at
                    // fetch time with no MAT number (e.g. Br-79/Br-81 chips under
                    // CENDL-3.2, which has no Br entries; or partial Cd coverage
                    // under CENDL-3.2 where only Cd-113 is present).
                    let known_a: std::collections::HashSet<u32> =
                        retrieval::known_isotopes_for(z, lib).into_iter().collect();
                    let natural: Vec<(Isotope, f64)> = nereids_core::elements::natural_isotopes(z)
                        .into_iter()
                        .filter(|(iso, _)| known_a.contains(&iso.a()))
                        .collect();
                    let known: Vec<Isotope> = known_a
                        .iter()
                        .filter_map(|&a| Isotope::new(z, a).ok())
                        .collect();

                    if natural.is_empty() && known.is_empty() {
                        ui.label("No ENDF evaluations available.");
                    } else {
                        // --- Natural isotopes (with abundance %) ---
                        if !natural.is_empty() {
                            ui.label("Natural isotopes:");
                            ui.horizontal_wrapped(|ui| {
                                for (iso, frac) in &natural {
                                    let a = iso.a();
                                    isotope_chip(ui, state, z, a, sym, Some(*frac), accent);
                                }
                            });
                        }

                        // --- Extra ENDF evaluations (no abundance) ---
                        let natural_a: Vec<u32> = natural.iter().map(|(iso, _)| iso.a()).collect();
                        let extra: Vec<u32> = known
                            .iter()
                            .map(|iso| iso.a())
                            .filter(|a| !natural_a.contains(a))
                            .collect();

                        if !extra.is_empty() {
                            let label = if natural.is_empty() {
                                "ENDF evaluations:"
                            } else {
                                "Additional ENDF evaluations:"
                            };
                            ui.label(label);
                            ui.horizontal_wrapped(|ui| {
                                for a in &extra {
                                    isotope_chip(ui, state, z, *a, sym, None, accent);
                                }
                            });
                        }
                    }
                } else {
                    ui.label("Click an element to see its natural isotopes.");
                }

                // --- Custom isotope entry ---
                ui.add_space(4.0);
                egui::CollapsingHeader::new("Custom Isotope (manual Z/A)")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Z:");
                            ui.add(
                                egui::DragValue::new(&mut state.periodic_table_custom_z)
                                    .range(1..=118_u32),
                            );
                            ui.label("A:");
                            ui.add(
                                egui::DragValue::new(&mut state.periodic_table_custom_a)
                                    .range(1..=999_u32),
                            );
                        });
                        if state.periodic_table_custom_z > state.periodic_table_custom_a {
                            state.periodic_table_custom_a = state.periodic_table_custom_z;
                        }
                        let cz = state.periodic_table_custom_z;
                        let ca = state.periodic_table_custom_a;
                        let csym = nereids_core::elements::element_symbol(cz).unwrap_or("??");
                        let has_eval = retrieval::has_endf_evaluation_for(cz, ca, lib);
                        let lib_label = super::design::library_name(lib);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("{csym}-{ca}")).strong());
                            if has_eval {
                                ui.label(
                                    egui::RichText::new(format!("{lib_label} eval available"))
                                        .color(Color32::from_rgb(34, 139, 34)),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new(format!("No {lib_label} eval"))
                                        .color(Color32::from_rgb(200, 130, 0)),
                                );
                            }
                        });
                        let already = state.periodic_table_selected_isotopes.contains(&(cz, ca));
                        if ui
                            .add_enabled(!already, egui::Button::new(format!("Add {csym}-{ca}")))
                            .clicked()
                        {
                            state.periodic_table_selected_isotopes.push((cz, ca));
                        }
                    });

                // --- Density + library + Add Selected ---
                if !state.periodic_table_selected_isotopes.is_empty() {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let is_ppm =
                            state.periodic_table_target == PeriodicTableTarget::DetectTrace;
                        let label = if is_ppm {
                            "Conc. (ppm):"
                        } else {
                            "Density (at/barn):"
                        };
                        ui.label(label);
                        ui.add(
                            egui::DragValue::new(&mut state.periodic_table_density)
                                .speed(if is_ppm { 10.0 } else { 0.0001 })
                                .range(if is_ppm { 0.1..=1e6 } else { 1e-6..=1.0 })
                                .max_decimals(if is_ppm { 1 } else { 6 }),
                        );

                        // Disable library changes while a fetch is in progress
                        // to prevent stale results from overwriting new entries.
                        let fetching = match state.periodic_table_target {
                            PeriodicTableTarget::Configure
                            | PeriodicTableTarget::ConfigureGroup => state.is_fetching_endf,
                            PeriodicTableTarget::ForwardModel => state.is_fetching_fm_endf,
                            PeriodicTableTarget::DetectMatrix
                            | PeriodicTableTarget::DetectTrace => state.is_fetching_detect_endf,
                        };
                        // Snapshot the library before the combobox so we can detect
                        // a user-driven library change and prune already-selected
                        // isotopes that aren't in the new library's coverage.
                        // Without this, selecting Br-79 under ENDF/B then switching
                        // this dropdown to CENDL-3.2 would still queue Br-79 (which
                        // CENDL doesn't cover) on click of Add Selected.
                        let prev_lib = *state
                            .periodic_table_library
                            .get_or_insert(target_library(state));
                        ui.add_enabled_ui(!fetching, |ui| {
                            let lib = state
                                .periodic_table_library
                                .as_mut()
                                .expect("initialized via get_or_insert above");
                            egui::ComboBox::from_id_salt("pt_lib")
                                .selected_text(super::design::library_name(*lib))
                                .show_ui(ui, |ui| {
                                    for (val, label) in [
                                        (EndfLibrary::EndfB8_0, "ENDF/B-VIII.0"),
                                        (EndfLibrary::EndfB8_1, "ENDF/B-VIII.1"),
                                        (EndfLibrary::Jeff3_3, "JEFF-3.3"),
                                        (EndfLibrary::Jendl5, "JENDL-5"),
                                        (EndfLibrary::Tendl2023, "TENDL-2023"),
                                        (EndfLibrary::Cendl3_2, "CENDL-3.2"),
                                    ] {
                                        ui.selectable_value(lib, val, label);
                                    }
                                });
                        });
                        let new_lib = state
                            .periodic_table_library
                            .expect("initialized via get_or_insert above");
                        if new_lib != prev_lib {
                            let before = state.periodic_table_selected_isotopes.len();
                            state.periodic_table_selected_isotopes.retain(|&(z, a)| {
                                retrieval::has_endf_evaluation_for(z, a, new_lib)
                            });
                            let dropped = before - state.periodic_table_selected_isotopes.len();
                            if dropped > 0 {
                                let lib_label = super::design::library_name(new_lib);
                                state.status_message = format!(
                                    "Dropped {} isotope{} not in {lib_label}",
                                    dropped,
                                    if dropped == 1 { "" } else { "s" },
                                );
                            }
                        }

                        let n_sel = state.periodic_table_selected_isotopes.len();
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(format!("Add Selected ({n_sel})"))
                                        .color(Color32::WHITE),
                                )
                                .fill(accent),
                            )
                            .clicked()
                        {
                            add_selected_isotopes(state);
                        }
                    });
                }
            }

            ui.add_space(8.0);

            // --- Close button ---
            ui.horizontal(|ui| {
                if ui.button("Close").clicked() {
                    close_modal(state);
                }
            });
        });

    if !open {
        close_modal(state);
    }
}

/// Clean up modal state on close.
fn close_modal(state: &mut AppState) {
    state.periodic_table_open = false;
    state.periodic_table_selected_z = None;
    state.periodic_table_selected_isotopes.clear();
    state.periodic_table_library = None;
}

// ---------------------------------------------------------------------------
// Isotope insertion logic
// ---------------------------------------------------------------------------

/// Batch-add all selected isotopes to the appropriate target list.
/// Closes the modal afterwards.
fn add_selected_isotopes(state: &mut AppState) {
    let density = state.periodic_table_density;
    // Defense-in-depth: filter the selected set against the target library's
    // coverage so a stale selection (e.g. Br-79 carried over from ENDF/B-VIII.0
    // before the user switched to CENDL-3.2) can't reach the fetch queue.  The
    // PT-library combobox prunes on change, but we re-check here in case any
    // future caller mutates `periodic_table_selected_isotopes` after the user's
    // last library interaction.
    let lib = state
        .periodic_table_library
        .unwrap_or_else(|| target_library(state));
    let selected: Vec<(u32, u32)> = state
        .periodic_table_selected_isotopes
        .iter()
        .copied()
        .filter(|&(z, a)| retrieval::has_endf_evaluation_for(z, a, lib))
        .collect();

    // Propagate PT library choice to the target's library field.
    // If the user selected a different library, existing resonance data
    // must be re-fetched (same pattern as the main library ComboBox).
    if let Some(lib) = state.periodic_table_library {
        let target_lib = match state.periodic_table_target {
            PeriodicTableTarget::Configure | PeriodicTableTarget::ConfigureGroup => {
                &mut state.endf_library
            }
            PeriodicTableTarget::ForwardModel => &mut state.fm_endf_library,
            PeriodicTableTarget::DetectMatrix | PeriodicTableTarget::DetectTrace => {
                &mut state.detect_endf_library
            }
        };
        if *target_lib != lib {
            *target_lib = lib;
            // Clear existing resonance data so they get re-fetched with the new library
            match state.periodic_table_target {
                PeriodicTableTarget::Configure | PeriodicTableTarget::ConfigureGroup => {
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
                }
                PeriodicTableTarget::ForwardModel => {
                    for e in &mut state.fm_isotope_entries {
                        e.resonance_data = None;
                        e.endf_status = EndfStatus::Pending;
                    }
                    state.fm_spectrum = None;
                    state.fm_per_isotope_spectra.clear();
                }
                PeriodicTableTarget::DetectMatrix | PeriodicTableTarget::DetectTrace => {
                    for e in &mut state.detect_matrix_entries {
                        e.resonance_data = None;
                        e.endf_status = EndfStatus::Pending;
                    }
                    for e in &mut state.detect_trace_entries {
                        e.resonance_data = None;
                        e.endf_status = EndfStatus::Pending;
                    }
                    state.detect_results.clear();
                }
            }
        }
    }

    let mut added = 0usize;
    let mut skipped_dup = 0usize;
    let mut skipped_group = 0usize;
    for (z, a) in &selected {
        let sym = nereids_core::elements::element_symbol(*z).unwrap_or("??");
        let symbol = format!("{}-{}", sym, a);
        match state.periodic_table_target {
            PeriodicTableTarget::Configure => {
                if state.isotope_entries.iter().any(|e| e.z == *z && e.a == *a) {
                    skipped_dup += 1;
                    continue; // already present
                }
                // Skip if this element already has a group (avoid group+individual overlap)
                if state.isotope_groups.iter().any(|g| g.z == *z) {
                    skipped_group += 1;
                    continue;
                }
                // Always add with Pending status — the auto-fetch loop in
                // configure_step will pick up new Pending entries once any
                // active fetch completes.
                state.isotope_entries.push(IsotopeEntry {
                    z: *z,
                    a: *a,
                    symbol,
                    initial_density: density,
                    resonance_data: None,
                    enabled: true,
                    endf_status: EndfStatus::Pending,
                });
                state.spatial_result = None;
                state.pixel_fit_result = None;
            }
            // ConfigureGroup uses add_element_group(), not add_selected_isotopes()
            PeriodicTableTarget::ConfigureGroup => continue,
            PeriodicTableTarget::ForwardModel => {
                if state
                    .fm_isotope_entries
                    .iter()
                    .any(|e| e.z == *z && e.a == *a)
                {
                    skipped_dup += 1;
                    continue; // already present
                }
                state.fm_isotope_entries.push(IsotopeEntry {
                    z: *z,
                    a: *a,
                    symbol,
                    initial_density: density,
                    resonance_data: None,
                    enabled: true,
                    endf_status: EndfStatus::Pending,
                });
                state.fm_spectrum = None;
                state.fm_per_isotope_spectra.clear();
            }
            PeriodicTableTarget::DetectMatrix => {
                if state
                    .detect_matrix_entries
                    .iter()
                    .any(|e| e.z == *z && e.a == *a)
                {
                    skipped_dup += 1;
                    continue; // already present
                }
                state.detect_matrix_entries.push(IsotopeEntry {
                    z: *z,
                    a: *a,
                    symbol,
                    initial_density: density,
                    resonance_data: None,
                    enabled: true,
                    endf_status: EndfStatus::Pending,
                });
                state.detect_results.clear();
            }
            PeriodicTableTarget::DetectTrace => {
                if state
                    .detect_trace_entries
                    .iter()
                    .any(|e| e.z == *z && e.a == *a)
                {
                    skipped_dup += 1;
                    continue; // already present
                }
                state.detect_trace_entries.push(DetectTraceEntry {
                    z: *z,
                    a: *a,
                    symbol,
                    concentration_ppm: density,
                    resonance_data: None,
                    endf_status: EndfStatus::Pending,
                });
                state.detect_results.clear();
            }
        }
        added += 1;
    }

    // Provide user feedback about what was added.
    let is_fetching = match state.periodic_table_target {
        PeriodicTableTarget::Configure | PeriodicTableTarget::ConfigureGroup => {
            state.is_fetching_endf
        }
        PeriodicTableTarget::ForwardModel => state.is_fetching_fm_endf,
        PeriodicTableTarget::DetectMatrix | PeriodicTableTarget::DetectTrace => {
            state.is_fetching_detect_endf
        }
    };
    // Build status message from scratch (don't append to stale old message).
    let mut msg = String::new();
    if added > 0 && is_fetching {
        msg = format!("Added {added} isotope(s) \u{2014} will fetch after current batch completes");
    } else if added > 0 {
        msg = format!("Added {added} isotope(s)");
    }
    // Report separate skip reasons so users understand why isotopes were not added.
    let mut skip_parts: Vec<String> = Vec::new();
    if skipped_dup > 0 {
        skip_parts.push(format!("{skipped_dup} duplicate(s) skipped"));
    }
    if skipped_group > 0 {
        skip_parts.push(format!("{skipped_group} skipped (element group exists)"));
    }
    if !skip_parts.is_empty() {
        let skip_msg = skip_parts.join(", ");
        if msg.is_empty() {
            msg = skip_msg;
        } else {
            msg = format!("{msg} ({skip_msg})");
        }
    }
    if !msg.is_empty() {
        state.status_message = msg;
    }

    close_modal(state);
}

/// Get the current ENDF library for the target context (default for library selector).
fn target_library(state: &AppState) -> EndfLibrary {
    match state.periodic_table_target {
        PeriodicTableTarget::Configure | PeriodicTableTarget::ConfigureGroup => state.endf_library,
        PeriodicTableTarget::ForwardModel => state.fm_endf_library,
        PeriodicTableTarget::DetectMatrix | PeriodicTableTarget::DetectTrace => {
            state.detect_endf_library
        }
    }
}

/// Add an element as a natural-isotope group (e.g., all natural W isotopes
/// sharing one density parameter).
///
/// Creates an `IsotopeGroupEntry` with members set to `Pending` so the
/// auto-fetch loop in `configure_step` will pick them up.
///
/// Refuses to construct a group if the selected library doesn't cover all
/// natural isotopes — silently dropping members and renormalizing partial
/// ratios would break the "natural composition" semantic (pure Cd-113 is
/// not natural Cd).  The modal UI gates the trigger button on the same
/// check; this is defense-in-depth.
fn add_element_group(state: &mut AppState, z: u32) {
    let lib = state
        .periodic_table_library
        .unwrap_or_else(|| target_library(state));
    let natural = nereids_core::elements::natural_isotopes(z);
    if natural.is_empty() {
        return;
    }
    let known_a: std::collections::HashSet<u32> =
        retrieval::known_isotopes_for(z, lib).into_iter().collect();
    if !natural.iter().all(|(iso, _)| known_a.contains(&iso.a())) {
        // UI should have prevented this, but refuse defensively rather than
        // build a partial group that misrepresents natural composition.
        let lib_label = super::design::library_name(lib);
        state.status_message = format!(
            "Cannot build natural group — {lib_label} is missing one or more natural isotopes"
        );
        return;
    }
    // Duplicate check: group already exists
    if state.isotope_groups.iter().any(|g| g.z == z) {
        state.status_message = "Element group already added".into();
        close_modal(state);
        return;
    }
    // Cross-check: warn if any of this element's natural isotopes exist as individuals
    let overlapping: Vec<String> = natural
        .iter()
        .filter(|(iso, _)| {
            state
                .isotope_entries
                .iter()
                .any(|e| e.z == z && e.a == iso.a())
        })
        .map(|(iso, _)| {
            let sym = nereids_core::elements::element_symbol(z).unwrap_or("??");
            format!("{sym}-{}", iso.a())
        })
        .collect();
    if !overlapping.is_empty() {
        // Remove individual entries that overlap with the group
        state
            .isotope_entries
            .retain(|e| !(e.z == z && natural.iter().any(|(iso, _)| iso.a() == e.a)));
        state.status_message = format!(
            "Removed individual {} (now in group)",
            overlapping.join(", ")
        );
    }

    // Propagate library choice
    if let Some(lib) = state.periodic_table_library
        && state.endf_library != lib
    {
        state.endf_library = lib;
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
    }

    let sym = nereids_core::elements::element_symbol(z).unwrap_or("??");
    let members: Vec<GroupMemberState> = natural
        .iter()
        .map(|(iso, ratio)| GroupMemberState {
            a: iso.a(),
            symbol: format!("{sym}-{}", iso.a()),
            ratio: *ratio,
            resonance_data: None,
            endf_status: EndfStatus::Pending,
        })
        .collect();
    let n = members.len();

    state.isotope_groups.push(IsotopeGroupEntry {
        z,
        name: format!("{sym} (nat)"),
        members,
        initial_density: state.periodic_table_density,
        enabled: true,
    });
    state.spatial_result = None;
    state.pixel_fit_result = None;
    state.status_message = format!("Added {sym} group ({n} isotopes)");
    close_modal(state);
}

/// Toggleable isotope chip. Shows abundance % for natural isotopes,
/// plain label for ENDF-only. Clicking toggles the (z, a) pair in
/// `periodic_table_selected_isotopes`.
fn isotope_chip(
    ui: &mut egui::Ui,
    state: &mut AppState,
    z: u32,
    a: u32,
    sym: &str,
    frac: Option<f64>,
    accent: Color32,
) {
    let pair = (z, a);
    let selected = state.periodic_table_selected_isotopes.contains(&pair);

    let label = match frac {
        Some(f) => format!("{sym}-{a} ({:.2}%)", f * 100.0),
        None => format!("{sym}-{a}"),
    };

    let bg = if selected {
        accent
    } else if frac.is_some() {
        Color32::from_gray(235)
    } else {
        Color32::from_gray(220)
    };
    let text_color = if selected {
        Color32::WHITE
    } else {
        Color32::BLACK
    };

    let btn = egui::Button::new(egui::RichText::new(label).size(12.0).color(text_color))
        .fill(bg)
        .corner_radius(CornerRadius::same(12));

    if ui.add(btn).clicked() {
        if selected {
            state
                .periodic_table_selected_isotopes
                .retain(|p| *p != pair);
        } else {
            state.periodic_table_selected_isotopes.push(pair);
        }
    }
}
