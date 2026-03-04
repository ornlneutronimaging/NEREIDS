//! Rebin step: optional energy rebinning by integer factor.
//!
//! - **Counts data** (TiffPair, HDF5): sum adjacent bins (conserves counts).
//! - **Transmission data** (TransmissionTiff): average adjacent bins
//!   (assumes uniform I₀ — this is an approximation).

use crate::state::{AppState, InputMode};
use crate::theme::ThemeColors;
use crate::widgets::design;
use nereids_io::spectrum::SpectrumValueKind;

/// Render the Rebin step.
pub fn rebin_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Rebin Energy Axis",
        "Optionally coarsen the energy binning (this step can be skipped)",
    );

    let is_transmission = state.input_mode == InputMode::TransmissionTiff;
    let n_bins = current_bin_count(state);

    design::card_with_header(ui, "Rebinning", None, |ui| {
        let tc = ThemeColors::from_ctx(ui.ctx());

        // Current bin count
        if let Some(n) = n_bins {
            ui.label(format!("Current bins: {n}"));
        } else {
            ui.label(egui::RichText::new("No spectrum data available.").color(tc.fg3));
            return;
        }
        let n = n_bins.unwrap();

        ui.add_space(4.0);

        // Transmission warning
        if is_transmission {
            ui.label(
                egui::RichText::new(
                    "\u{26a0} Transmission data: rebinning uses bin-averaging \
                     (assumes uniform I\u{2080}). This is an approximation.",
                )
                .size(11.0)
                .color(tc.fg3),
            );
            ui.add_space(4.0);
        }

        if state.rebin_applied {
            // Show applied state
            ui.horizontal(|ui| {
                design::badge(
                    ui,
                    &format!("Rebinned \u{00d7}{}", state.rebin_factor),
                    design::BadgeVariant::Green,
                );
                if ui.button("Undo").clicked() {
                    // Clear data to trigger auto-load from disk
                    state.sample_data = None;
                    state.open_beam_data = None;
                    state.spectrum_values = None;
                    state.rebin_applied = false;
                    state.rebin_factor = 1;
                    state.load_error = false;
                    state.invalidate_results();
                    state.log_provenance(
                        crate::state::ProvenanceEventKind::ConfigChanged,
                        "Rebin undone — reloading original data",
                    );
                    // Navigate back to Load so auto-load re-triggers
                    state.guided_step = crate::state::GuidedStep::Load;
                }
            });
        } else {
            // Factor selector
            ui.horizontal(|ui| {
                ui.label("Factor:");
                ui.add(
                    egui::DragValue::new(&mut state.rebin_factor)
                        .range(1..=n)
                        .speed(1),
                );
                for &f in &[2, 4, 8] {
                    if f <= n
                        && ui
                            .selectable_label(state.rebin_factor == f, format!("{f}x"))
                            .clicked()
                    {
                        state.rebin_factor = f;
                    }
                }
            });

            // Preview result
            if state.rebin_factor > 1 {
                let n_new = n.div_ceil(state.rebin_factor);
                ui.label(
                    egui::RichText::new(format!("{n} \u{2192} {n_new} bins"))
                        .size(12.0)
                        .color(tc.fg2),
                );
            }

            ui.add_space(4.0);

            // Apply button
            let can_apply = state.rebin_factor > 1 && state.sample_data.is_some();
            ui.add_enabled_ui(can_apply, |ui| {
                if ui.button("Apply Rebinning").clicked() {
                    apply_rebin(state, is_transmission);
                }
            });
        }
    });

    // Stat row
    if let Some(n) = n_bins {
        let label = format!("{n}");
        design::stat_row(ui, &[(&label, "energy bins")]);
    }

    // Nav buttons — always passable (optional step)
    match design::nav_buttons(ui, Some("\u{2190} Back"), "Continue \u{2192}", true, "") {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}

fn current_bin_count(state: &AppState) -> Option<usize> {
    let vals = state.spectrum_values.as_ref()?;
    Some(if state.spectrum_kind == SpectrumValueKind::BinEdges {
        vals.len().saturating_sub(1)
    } else {
        vals.len()
    })
}

fn apply_rebin(state: &mut AppState, is_transmission: bool) {
    let factor = state.rebin_factor;

    // Cancel any in-flight fitting tasks before mutating data arrays
    state.cancel_pending_tasks();

    // Validate spectrum axis matches data axis-0
    if let (Some(data), Some(vals)) = (&state.sample_data, &state.spectrum_values) {
        let expected = if state.spectrum_kind == SpectrumValueKind::BinEdges {
            vals.len().saturating_sub(1)
        } else {
            vals.len()
        };
        if expected != data.shape()[0] {
            state.status_message = format!(
                "Spectrum length ({}) doesn't match data bins ({})",
                expected,
                data.shape()[0]
            );
            return;
        }
    }

    // Rebin sample_data
    if let Some(ref data) = state.sample_data {
        let rebinned = if is_transmission {
            nereids_io::rebin::rebin_transmission(data, factor)
        } else {
            nereids_io::rebin::rebin_counts(data, factor)
        };
        state.sample_data = Some(rebinned);
    }

    // Rebin open_beam_data (always counts)
    if let Some(ref data) = state.open_beam_data {
        let rebinned = nereids_io::rebin::rebin_counts(data, factor);
        state.open_beam_data = Some(rebinned);
    }

    // Rebin spectrum axis
    if let Some(ref vals) = state.spectrum_values {
        let new_vals = if state.spectrum_kind == SpectrumValueKind::BinEdges {
            nereids_io::rebin::rebin_edges(vals, factor)
        } else {
            nereids_io::rebin::rebin_centers(vals, factor)
        };
        state.spectrum_values = Some(new_vals);
    }

    // Update preview image (sum along TOF axis of new data)
    if let Some(ref data) = state.sample_data {
        state.preview_image = Some(data.sum_axis(ndarray::Axis(0)));
    }

    state.rebin_applied = true;
    state.energies = None;
    state.normalized = None;
    state.spatial_result = None;
    state.pixel_fit_result = None;
    state.status_message = format!("Rebinned by factor {factor}");
    state.log_provenance(
        crate::state::ProvenanceEventKind::ConfigChanged,
        format!("Rebinned by factor {factor}"),
    );
}
