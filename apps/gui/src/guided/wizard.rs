//! Decision wizard: Q1 (fitting type), Q2 (data format), Confirm (pipeline).

use crate::state::{
    AnalysisMode, AppState, DataType, FittingType, GuidedStep, InputMode, PipelineEntry,
};
use crate::theme::ThemeColors;
use crate::widgets::design;
use egui::{Color32, CornerRadius, Margin, RichText, Stroke};

/// Render the decision wizard (dispatches to the current sub-step).
pub fn wizard_step(ui: &mut egui::Ui, state: &mut AppState) {
    let max_width = 700.0_f32.min(ui.available_width());
    ui.vertical(|ui| {
        ui.set_max_width(max_width);

        // Breadcrumb bar
        breadcrumb_bar(ui, state);
        ui.add_space(12.0);

        match state.wizard_step {
            0 => wizard_q1(ui, state),
            1 => wizard_q2(ui, state),
            _ => wizard_confirm(ui, state),
        }
    });
}

// ── Breadcrumb ──────────────────────────────────────────────────

fn breadcrumb_bar(ui: &mut egui::Ui, state: &mut AppState) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    ui.horizontal(|ui| {
        for (idx, label) in ["Q1: Fitting", "Q2: Data", "Confirm"].iter().enumerate() {
            let is_active = state.wizard_step == idx as u8;
            let is_past = (idx as u8) < state.wizard_step;

            let (fill, text_color) = if is_active {
                (tc.accent, Color32::WHITE)
            } else if is_past {
                (tc.bg3, tc.fg)
            } else {
                (Color32::TRANSPARENT, tc.fg3)
            };

            let resp = egui::Frame::NONE
                .fill(fill)
                .corner_radius(CornerRadius::same(4))
                .inner_margin(Margin::symmetric(10, 4))
                .stroke(if is_active || is_past {
                    Stroke::NONE
                } else {
                    Stroke::new(1.0, tc.border)
                })
                .show(ui, |ui| {
                    ui.label(RichText::new(*label).size(11.0).strong().color(text_color));
                })
                .response;

            // Allow clicking past breadcrumbs to go back
            if is_past && resp.interact(egui::Sense::click()).clicked() {
                state.wizard_step = idx as u8;
            }

            if idx < 2 {
                ui.label(RichText::new("\u{203A}").color(tc.fg3));
            }
        }
    });
}

// ── Q1: Fitting Type ────────────────────────────────────────────

fn wizard_q1(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "What kind of fitting?",
        "This determines how results are presented and how ROI is interpreted.",
    );

    // Spatial option
    let selected_spatial = state.fitting_type == Some(FittingType::Spatial);
    if wizard_option_card(
        ui,
        "Spatially Resolved Mapping",
        "Fit each pixel independently to produce full density maps. \
         ROI crops the result region.",
        Some("Most common"),
        selected_spatial,
    ) {
        state.fitting_type = Some(FittingType::Spatial);
        state.analysis_mode = AnalysisMode::FullSpatialMap;
        state.wizard_step = 1;
    }

    // Single option
    let selected_single = state.fitting_type == Some(FittingType::Single);
    if wizard_option_card(
        ui,
        "Single Spectrum (ROI Accumulation)",
        "Accumulate all detector pixels (or an ROI) into one high-statistics \
         spectrum. Produces a single density result per isotope.",
        Some("Higher statistics"),
        selected_single,
    ) {
        state.fitting_type = Some(FittingType::Single);
        state.analysis_mode = AnalysisMode::RoiSingleSpectrum;
        state.wizard_step = 1;
    }

    ui.add_space(12.0);
    if ui.button("\u{2190} Back to Home").clicked() {
        state.guided_step = GuidedStep::Landing;
    }
}

// ── Q2: Data Format ─────────────────────────────────────────────

fn wizard_q2(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "What data format?",
        "This determines which processing steps are needed before fitting.",
    );

    // Events
    let selected_events = state.data_type == Some(DataType::Events);
    if wizard_option_card(
        ui,
        "Raw Events (HDF5/NeXus)",
        "Time-stamped neutron detection events. Requires binning into \
         histogram before normalization. Both sample and open beam files needed.",
        Some("Full flexibility"),
        selected_events,
    ) {
        state.data_type = Some(DataType::Events);
        state.input_mode = InputMode::Hdf5Event;
        state.wizard_step = 2;
    }

    // Pre-normalized
    let selected_prenorm = state.data_type == Some(DataType::PreNormalized);
    if wizard_option_card(
        ui,
        "Histogram, Pre-Normalization",
        "Pre-binned count data \u{2014} sample + open beam pair. \
         Supports HDF5 histogram or TIFF stack + spectrum file.",
        None,
        selected_prenorm,
    ) {
        state.data_type = Some(DataType::PreNormalized);
        // Default to TiffPair; user can switch to Hdf5Histogram in Load step
        state.input_mode = InputMode::TiffPair;
        state.wizard_step = 2;
    }

    // Transmission
    let selected_trans = state.data_type == Some(DataType::Transmission);
    if wizard_option_card(
        ui,
        "Transmission (Already Normalized)",
        "Pre-computed T(E) = I/I\u{2080}. Supports HDF5 histogram or \
         TIFF stack + spectrum file. Skips normalization entirely.",
        None,
        selected_trans,
    ) {
        state.data_type = Some(DataType::Transmission);
        state.input_mode = InputMode::TransmissionTiff;
        state.wizard_step = 2;
    }

    ui.add_space(12.0);
    if ui.button("\u{2190} Back").clicked() {
        state.wizard_step = 0;
    }
}

// ── Q3: Confirm ─────────────────────────────────────────────────

fn wizard_confirm(ui: &mut egui::Ui, state: &mut AppState) {
    let (Some(ft), Some(dt)) = (state.fitting_type, state.data_type) else {
        // Should not reach Confirm without both selections — redirect to Q1
        state.wizard_step = 0;
        return;
    };

    let fitting_label = match ft {
        FittingType::Spatial => "Spatially Resolved",
        FittingType::Single => "Single Spectrum",
    };
    let data_label = match dt {
        DataType::Events => "Events",
        DataType::PreNormalized => "Pre-norm Histogram",
        DataType::Transmission => "Transmission",
    };

    design::content_header(
        ui,
        "Your Analysis Pipeline",
        &format!("Pathway: {fitting_label} + {data_label}"),
    );

    let steps = GuidedStep::pipeline(ft, dt);

    // Pipeline chips
    design::card_with_header(ui, "Pipeline Steps", None, |ui| {
        pipeline_chips(ui, &steps);
    });

    // Limitations
    let limitations = limitations_for(ft, dt);
    if !limitations.is_empty() {
        let tc = ThemeColors::from_ctx(ui.ctx());
        egui::Frame::NONE
            .fill(tc.bg2)
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::same(14))
            .stroke(Stroke::new(1.0, tc.accent))
            .show(ui, |ui| {
                ui.label(
                    RichText::new("\u{26A0} Limitations & Notes")
                        .strong()
                        .size(13.0),
                );
                ui.add_space(4.0);
                for note in limitations {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("\u{2022}").color(tc.fg2));
                        ui.label(RichText::new(note).size(12.0).color(tc.fg2));
                    });
                }
            });
        ui.add_space(14.0);
    }

    // Action buttons
    ui.horizontal(|ui| {
        if ui.button("\u{2190} Change").clicked() {
            state.wizard_step = 1;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if design::btn_primary(ui, "Start \u{2192}").clicked() {
                // fitting_type and data_type are already set by Q1/Q2
                state.rebuild_pipeline();
                state.cached_session = None; // new pipeline supersedes cache
                if let Some(first) = state.pipeline.first() {
                    state.guided_step = first.step;
                }
            }
        });
    });
}

// ── Helpers ─────────────────────────────────────────────────────

/// Render pipeline chips joined by arrows.
fn pipeline_chips(ui: &mut egui::Ui, steps: &[PipelineEntry]) {
    let tc = ThemeColors::from_ctx(ui.ctx());
    ui.horizontal_wrapped(|ui| {
        for (i, entry) in steps.iter().enumerate() {
            if i > 0 {
                ui.label(RichText::new("\u{2192}").color(tc.fg3));
            }
            let label = entry.step.label();
            let text = if entry.optional {
                format!("{label} (opt)")
            } else {
                label.to_string()
            };
            egui::Frame::NONE
                .fill(if entry.optional { tc.bg3 } else { tc.accent })
                .corner_radius(CornerRadius::same(4))
                .inner_margin(Margin::symmetric(8, 3))
                .show(ui, |ui| {
                    let color = if entry.optional {
                        tc.fg
                    } else {
                        Color32::WHITE
                    };
                    ui.label(RichText::new(text).size(11.0).strong().color(color));
                });
        }
    });
}

/// A clickable wizard option card. Returns `true` if clicked.
fn wizard_option_card(
    ui: &mut egui::Ui,
    title: &str,
    description: &str,
    badge: Option<&str>,
    selected: bool,
) -> bool {
    let tc = ThemeColors::from_ctx(ui.ctx());
    let fill = if selected { tc.bg3 } else { tc.bg2 };
    let border = if selected {
        Stroke::new(2.0, tc.accent)
    } else {
        Stroke::new(1.0, tc.border)
    };

    let resp = egui::Frame::NONE
        .fill(fill)
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(16))
        .stroke(border)
        .shadow(if selected {
            egui::Shadow::NONE
        } else {
            egui::Shadow {
                offset: [0, 1],
                blur: 3,
                spread: 0,
                color: Color32::from_black_alpha(12),
            }
        })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(title).strong().size(14.0));
                if let Some(b) = badge {
                    ui.add_space(4.0);
                    design::badge(ui, b, design::BadgeVariant::Green);
                }
            });
            ui.add_space(4.0);
            ui.label(RichText::new(description).size(12.0).color(tc.fg2));
        })
        .response;

    ui.add_space(8.0);
    resp.interact(egui::Sense::click()).clicked()
}

/// Pathway-specific limitations and notes for the Confirm step.
fn limitations_for(fitting: FittingType, data: DataType) -> Vec<&'static str> {
    match (fitting, data) {
        (FittingType::Spatial, DataType::Events) => vec![
            "Full flexibility: choose bin count, mode, and TOF range.",
            "Requires both sample and open beam event files.",
            "ROI crops the result region (each pixel fitted independently).",
        ],
        (FittingType::Single, DataType::Events) => vec![
            "Events are binned, then ROI pixels are accumulated into one spectrum.",
            "Single density result per isotope (no spatial map).",
            "Higher statistics than per-pixel fitting.",
        ],
        (FittingType::Spatial, DataType::PreNormalized) => vec![
            "Histogram data is already binned \u{2014} limited to existing bins.",
            "Optional spatial rebin (2\u{00D7}2, 3\u{00D7}3, etc.) to boost per-pixel counts.",
            "Normalization computes T(E) from sample/OB pair.",
            "ROI crops the result region (each pixel fitted independently).",
        ],
        (FittingType::Single, DataType::PreNormalized) => vec![
            "Physics note: raw counts must be accumulated FIRST, then normalized.",
            "Division is nonlinear: mean(a/b) \u{2260} mean(a)/mean(b).",
            "Optional spatial rebin before accumulation for better statistics.",
            "ROI pixels accumulated into one high-statistics spectrum.",
        ],
        (FittingType::Spatial, DataType::Transmission) => vec![
            "Data is already normalized \u{2014} skips normalization step entirely.",
            "Optional spatial rebin (requires I\u{2080} estimate to approximate counts).",
            "ROI crops the result region (each pixel fitted independently).",
        ],
        (FittingType::Single, DataType::Transmission) => vec![
            "Transmission data: to accumulate correctly, needs I\u{2080} estimate.",
            "Optional spatial rebin via counts approximation (T \u{00D7} I\u{2080} \u{2192} sum \u{2192} \u{00F7} I\u{2080}).",
            "Approximates counts as T \u{00D7} I\u{2080}_est, sums, then re-normalizes.",
        ],
    }
}
