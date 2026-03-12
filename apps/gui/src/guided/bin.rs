//! Bin step: configure and run event histogramming.
//!
//! This step appears only in event-based pipelines (after Load, before
//! Normalize). The user sets bin parameters and triggers histogramming
//! of the raw HDF5 event data loaded in the previous step.

use std::sync::Arc;

use crate::state::{AppState, ProvenanceEventKind};
use crate::theme::ThemeColors;
use crate::widgets::design;

/// Render the Bin step.
pub fn bin_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Bin Events",
        "Configure histogram parameters for raw event data",
    );

    let has_events =
        state.hdf5_path.is_some() && state.nexus_metadata.as_ref().is_some_and(|m| m.has_events);

    if !has_events {
        design::card(ui, |ui| {
            let tc = ThemeColors::from_ctx(ui.ctx());
            ui.label(
                egui::RichText::new(
                    "No event data loaded. Go back to Load and select an HDF5 event file.",
                )
                .color(tc.fg2),
            );
        });
    } else {
        design::card_with_header(ui, "Binning Parameters", None, |ui| {
            ui.horizontal(|ui| {
                ui.label("TOF bins:");
                ui.add(egui::DragValue::new(&mut state.event_n_bins).range(1..=10000));
            });
            ui.horizontal(|ui| {
                ui.label("TOF min (\u{00B5}s):");
                ui.add(egui::DragValue::new(&mut state.event_tof_min_us).speed(10.0));
            });
            ui.horizontal(|ui| {
                ui.label("TOF max (\u{00B5}s):");
                ui.add(egui::DragValue::new(&mut state.event_tof_max_us).speed(10.0));
            });
            ui.horizontal(|ui| {
                ui.label("Height (px):");
                ui.add(egui::DragValue::new(&mut state.event_height).range(1..=4096));
            });
            ui.horizontal(|ui| {
                ui.label("Width (px):");
                ui.add(egui::DragValue::new(&mut state.event_width).range(1..=4096));
            });

            ui.add_space(4.0);
            let valid_range = state.event_tof_min_us < state.event_tof_max_us;
            ui.add_enabled_ui(valid_range, |ui| {
                if design::btn_primary(ui, "Histogram Events").clicked() {
                    histogram_events(state);
                }
                if !valid_range {
                    ui.label(
                        egui::RichText::new("TOF min must be less than TOF max")
                            .small()
                            .color(crate::theme::semantic::RED),
                    );
                }
            });
        });

        // Show result summary if histogrammed
        if let Some(ref data) = state.sample_data {
            let shape = data.shape();
            let bins = format!("{}", shape[0]);
            let px = format!("{}×{}", shape[1], shape[2]);
            design::stat_row(ui, &[(&bins, "bins"), (&px, "pixels")]);
        }
    }

    let can_continue = state.sample_data.is_some();
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        can_continue,
        "Histogram events first",
    ) {
        design::NavAction::Back => state.nav_prev(),
        design::NavAction::Continue => state.nav_next(),
        design::NavAction::None => {}
    }
}

/// Histogram raw events from HDF5 file using configured bin parameters.
fn histogram_events(state: &mut AppState) {
    let path = match state.hdf5_path {
        Some(ref p) => p.clone(),
        None => return,
    };

    state.invalidate_results();

    let params = nereids_io::nexus::EventBinningParams {
        n_bins: state.event_n_bins,
        tof_min_us: state.event_tof_min_us,
        tof_max_us: state.event_tof_max_us,
        height: state.event_height,
        width: state.event_width,
    };

    match nereids_io::nexus::load_nexus_events(&path, &params) {
        Ok(data) => {
            let shape = data.counts.shape();
            state.preview_image = Some(data.counts.sum_axis(ndarray::Axis(0)));
            state.status_message = format!(
                "Events histogrammed: {} bins, {}×{} px",
                shape[0], shape[1], shape[2]
            );

            state.spectrum_values = Some(Arc::new(data.tof_edges_us.clone()));
            state.spectrum_unit = nereids_io::spectrum::SpectrumUnit::TofMicroseconds;
            state.spectrum_kind = nereids_io::spectrum::SpectrumValueKind::BinEdges;

            if let Some(fp) = data.flight_path_m
                && fp.is_finite()
                && fp > 0.0
            {
                state.beamline.flight_path_m = fp;
            }

            if let Some(offset_ns) = state.nexus_metadata.as_ref().and_then(|m| m.tof_offset_ns) {
                let delay_us = offset_ns / 1000.0;
                if delay_us.is_finite() {
                    state.beamline.delay_us = delay_us;
                }
            }

            if let Some(dead) = data.dead_pixels {
                state.dead_pixels = Some(dead);
            }

            state.log_provenance(
                ProvenanceEventKind::DataLoaded,
                format!(
                    "Histogrammed HDF5 events: {} frames ({}x{})",
                    shape[0], shape[1], shape[2]
                ),
            );
            state.sample_data = Some(Arc::new(data.counts));
        }
        Err(e) => {
            state.status_message = format!("Event histogramming failed: {e}");
        }
    }
}
