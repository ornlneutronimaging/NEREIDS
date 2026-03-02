//! Periodic Table modal widget for interactive element/isotope selection.
//!
//! Renders a standard 18-column periodic table as a grid of colored buttons.
//! When an element is selected, its natural isotopes are shown as clickable
//! chips that add the isotope to the appropriate target list (Configure,
//! Forward Model, or Detectability).

use crate::state::{AppState, DetectTraceEntry, IsotopeEntry, PeriodicTableTarget};
use egui::Color32;

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

    let mut open = true;
    egui::Window::new("Periodic Table")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            // --- Element grid ---
            egui::Grid::new("pt_grid")
                .spacing([2.0, 2.0])
                .show(ui, |ui| {
                    for row in 0..=9 {
                        // Row 7 is a visual gap between the main table and f-block
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
                                    // Highlight selected element
                                    Color32::from_rgb(100, 149, 237)
                                } else {
                                    bg
                                });

                                if ui.add(btn).clicked() {
                                    state.periodic_table_selected_z = Some(z);
                                }
                            } else {
                                // Placeholder labels for lanthanide/actinide markers
                                if row == 5 && col == 2 {
                                    ui.label(egui::RichText::new("*").size(10.0).strong());
                                } else if row == 6 && col == 2 {
                                    ui.label(egui::RichText::new("**").size(10.0).strong());
                                } else if row == 8 && col == 2 {
                                    ui.label(egui::RichText::new("*").size(10.0).strong());
                                } else if row == 9 && col == 2 {
                                    ui.label(egui::RichText::new("**").size(10.0).strong());
                                } else {
                                    // Empty cell
                                    ui.allocate_space(egui::vec2(28.0, 28.0));
                                }
                            }
                        }
                        ui.end_row();
                    }
                });

            ui.add_space(8.0);
            ui.separator();

            // --- Selected element info + natural isotopes ---
            if let Some(z) = state.periodic_table_selected_z {
                let name = nereids_core::elements::element_name(z).unwrap_or("Unknown");
                let sym = nereids_core::elements::element_symbol(z).unwrap_or("??");
                ui.label(
                    egui::RichText::new(format!("Z={z}  {sym} - {name}"))
                        .strong()
                        .size(14.0),
                );

                let isotopes = nereids_core::elements::natural_isotopes(z);
                if isotopes.is_empty() {
                    ui.label("No natural isotopes in database.");
                } else {
                    ui.label("Natural isotopes (click to add):");
                    ui.horizontal_wrapped(|ui| {
                        for (iso, frac) in &isotopes {
                            let a = iso.a();
                            let chip_label = format!("{}-{}  ({:.2}%)", sym, a, frac * 100.0);
                            if ui.button(&chip_label).clicked() {
                                add_isotope_to_target(state, z, a, sym);
                            }
                        }
                    });
                }
            } else {
                ui.label("Click an element to see its natural isotopes.");
            }

            ui.add_space(8.0);

            // --- Close button ---
            ui.horizontal(|ui| {
                if ui.button("Close").clicked() {
                    state.periodic_table_open = false;
                    state.periodic_table_selected_z = None;
                }
            });
        });

    // Handle the window X button
    if !open {
        state.periodic_table_open = false;
        state.periodic_table_selected_z = None;
    }
}

// ---------------------------------------------------------------------------
// Isotope insertion logic
// ---------------------------------------------------------------------------

/// Add a selected isotope to the appropriate target list based on
/// `state.periodic_table_target`.
///
/// Blocked when an ENDF fetch is in-flight for the corresponding target,
/// preventing stale fetch results from landing on newly-added isotopes.
fn add_isotope_to_target(state: &mut AppState, z: u32, a: u32, sym: &str) {
    let symbol = format!("{}-{}", sym, a);
    match state.periodic_table_target {
        PeriodicTableTarget::Configure => {
            if state.is_fetching_endf {
                return;
            }
            state.isotope_entries.push(IsotopeEntry {
                z,
                a,
                symbol,
                initial_density: 0.001,
                resonance_data: None,
                enabled: true,
            });
            // Invalidate stale results -- isotope list changed.
            state.spatial_result = None;
            state.pixel_fit_result = None;
        }
        PeriodicTableTarget::ForwardModel => {
            if state.is_fetching_fm_endf {
                return;
            }
            state.fm_isotope_entries.push(IsotopeEntry {
                z,
                a,
                symbol,
                initial_density: 0.001,
                resonance_data: None,
                enabled: true,
            });
            state.fm_spectrum = None;
            state.fm_per_isotope_spectra.clear();
        }
        PeriodicTableTarget::DetectMatrix => {
            if state.is_fetching_detect_endf {
                return;
            }
            state.detect_matrix = Some(IsotopeEntry {
                z,
                a,
                symbol,
                initial_density: state.detect_matrix_density,
                resonance_data: None,
                enabled: true,
            });
            state.detect_results.clear();
        }
        PeriodicTableTarget::DetectTrace => {
            if state.is_fetching_detect_endf {
                return;
            }
            state.detect_trace_entries.push(DetectTraceEntry {
                z,
                a,
                symbol,
                concentration_ppm: 1000.0,
                resonance_data: None,
            });
            state.detect_results.clear();
        }
    }
}
