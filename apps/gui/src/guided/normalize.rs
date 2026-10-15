//! Step 3: Prepare — energy grid, pixel-mask detection, and counts preview.

use crate::state::{
    AppState, EndfStatus, InputMode, ProvenanceEventKind, SpectrumAxis, SpectrumDataSource,
};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::show_viridis_image;
use egui_plot::{Line, Plot, PlotPoints, VLine};
use ndarray::Axis;
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use std::sync::Arc;

/// Draw the Prepare step content.
pub fn normalize_step(ui: &mut egui::Ui, state: &mut AppState) {
    design::content_header(
        ui,
        "Prepare Counts",
        "Build the energy grid from the spectrum file and detect bad pixels.",
    );

    if state.energies.is_none() && state.sample_data.is_some() && state.spectrum_values.is_some() {
        prepare_counts(state);
    }

    let has_data = state.energies.is_some();
    if has_data {
        if let (Some(sample), Some(energies)) = (&state.sample_data, &state.energies) {
            let shape = sample.shape();
            let bins = format!("{}", shape[0]);
            let pixels = format!("{}", shape[1] * shape[2]);
            let e_min = format!("{:.2}", energies.first().copied().unwrap_or(0.0));
            let e_max = format!("{:.2}", energies.last().copied().unwrap_or(0.0));
            design::stat_row(
                ui,
                &[
                    (&bins, "Energy Bins"),
                    (&pixels, "Pixels"),
                    (&e_min, "E_min (eV)"),
                    (&e_max, "E_max (eV)"),
                ],
            );
        }
    } else if state.sample_data.is_some() {
        ui.label("Load a spectrum file to define the energy axis.");
    } else {
        ui.label("Load sample and open-beam counts first.");
    }

    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        has_data,
        "Prepare data to continue",
    ) {
        NavAction::Back => state.nav_prev(),
        NavAction::Continue => state.nav_next(),
        NavAction::None => {}
    }

    if has_data {
        ui.add_space(8.0);
        counts_preview_card(ui, state);
    }
}

fn counts_preview_card(ui: &mut egui::Ui, state: &mut AppState) {
    design::card_with_header(ui, "Counts Preview", None, |ui| {
        let available_width = ui.available_width();
        let image_width = 220.0_f32.min(available_width * 0.35);

        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(image_width, ui.available_height()),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| preview_image_panel(ui, state),
            );

            ui.separator();

            ui.vertical(|ui| preview_spectrum_panel(ui, state));
        });
    });
}

fn preview_image_panel(ui: &mut egui::Ui, state: &mut AppState) {
    let Some(sample) = state.sample_data.as_ref() else {
        return;
    };
    let n_tof = sample.shape()[0];
    if n_tof == 0 {
        ui.label("(no data)");
        return;
    }

    if state.tof_slice_index >= n_tof {
        state.tof_slice_index = n_tof - 1;
    }

    let slice = sample.index_axis(Axis(0), state.tof_slice_index).to_owned();
    let _ = show_viridis_image(ui, &slice, "prepare_preview_tex");

    ui.add(
        egui::Slider::new(&mut state.tof_slice_index, 0..=n_tof.saturating_sub(1)).text("TOF bin"),
    );
}

fn preview_spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label("Source:");
        ui.selectable_value(
            &mut state.normalize_spectrum_source,
            SpectrumDataSource::FullImage,
            "Full Image",
        );
        ui.selectable_value(
            &mut state.normalize_spectrum_source,
            SpectrumDataSource::RoiAverage,
            "ROI Average",
        );
    });

    ui.horizontal(|ui| {
        ui.label("Axis:");
        ui.selectable_value(
            &mut state.normalize_spectrum_axis,
            SpectrumAxis::EnergyEv,
            "Energy (eV)",
        );
        ui.selectable_value(
            &mut state.normalize_spectrum_axis,
            SpectrumAxis::TofMicroseconds,
            "TOF (\u{03bc}s)",
        );
        ui.separator();
        ui.checkbox(&mut state.show_resonance_dips, "Resonance dips");
    });

    let Some(sample) = state.sample_data.as_ref() else {
        return;
    };
    let n_tof = sample.shape()[0];

    let spectrum: Vec<f64> = match state.normalize_spectrum_source {
        SpectrumDataSource::FullImage => {
            match sample.mean_axis(Axis(2)).and_then(|a| a.mean_axis(Axis(1))) {
                Some(a) => a.to_vec(),
                None => {
                    ui.label("(empty spatial dimensions)");
                    return;
                }
            }
        }
        SpectrumDataSource::RoiAverage => {
            if let Some(roi) = state.bounding_roi() {
                match nereids_io::pixels::average_roi(
                    sample,
                    roi.y_start..roi.y_end,
                    roi.x_start..roi.x_end,
                ) {
                    Ok(a) => a.to_vec(),
                    Err(_) => {
                        ui.label("Invalid ROI \u{2014} using full image.");
                        return;
                    }
                }
            } else {
                ui.label("No ROI set \u{2014} showing full image average.");
                return;
            }
        }
    };

    let Some((x_values, x_label)) = design::build_spectrum_x_axis(&design::SpectrumXAxisParams {
        axis: state.normalize_spectrum_axis,
        energies: state.energies.as_deref(),
        spectrum_values: state.spectrum_values.as_ref().map(|v| v.as_slice()),
        spectrum_unit: state.spectrum_unit,
        spectrum_kind: state.spectrum_kind,
        flight_path_m: state.beamline.flight_path_m,
        delay_us: state.beamline.delay_us,
        n_tof,
    }) else {
        return;
    };

    let n_plot = spectrum.len().min(x_values.len());
    if n_plot == 0 {
        return;
    }

    let points: PlotPoints = (0..n_plot).map(|i| [x_values[i], spectrum[i]]).collect();
    let line = Line::new("Counts", points);

    let tof_marker_x = x_values
        .get(state.tof_slice_index.min(n_plot.saturating_sub(1)))
        .copied();

    Plot::new("prepare_spectrum_plot")
        .x_axis_label(x_label)
        .y_axis_label("Counts")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(line);

            if let Some(x) = tof_marker_x {
                plot_ui.vline(
                    VLine::new("TOF position", x)
                        .color(egui::Color32::from_rgb(255, 165, 0))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            }

            if state.show_resonance_dips && state.normalize_spectrum_axis == SpectrumAxis::EnergyEv
            {
                let all_rd = resonance_data(state);
                design::draw_resonance_dips(plot_ui, &all_rd, &x_values);
            }
        });
}

fn resonance_data(state: &AppState) -> Vec<nereids_endf::resonance::ResonanceData> {
    let mut all = Vec::new();
    for e in &state.isotope_entries {
        if e.enabled
            && let Some(rd) = &e.resonance_data
        {
            all.push(rd.clone());
        }
    }
    for g in &state.isotope_groups {
        if g.enabled && g.overall_status() == EndfStatus::Loaded {
            for m in &g.members {
                if let Some(rd) = &m.resonance_data {
                    all.push(rd.clone());
                }
            }
        }
    }
    all
}

/// Build the energy grid and pixel mask for the loaded counts.
pub(crate) fn prepare_counts(state: &mut AppState) {
    state.cancel_pending_tasks();

    let sample = match state.sample_data {
        Some(ref d) => Arc::clone(d),
        None => return,
    };
    let open_beam = state.open_beam_data.as_ref().map(Arc::clone);

    if state.input_mode == InputMode::TiffPair && open_beam.is_none() {
        state.status_message = "Load an open-beam stack to prepare counts".into();
        return;
    }

    let n_tof = sample.shape()[0];
    match compute_energies(state, n_tof) {
        Ok(energies) => state.energies = Some(energies),
        Err(e) => {
            state.energies = None;
            state.status_message = format!("Energy conversion: {e}");
            return;
        }
    }

    match nereids_io::pixels::detect_bad_pixels(
        &sample,
        open_beam.as_deref(),
        Some(nereids_io::pixels::HOT_PIXEL_K_MAD),
    ) {
        Ok(mask) => {
            if let Some(notice) = state.set_detected_dead_pixels(Some(mask)) {
                state.log_provenance(ProvenanceEventKind::ConfigChanged, notice);
            }
        }
        Err(e) => {
            let _ = state.set_detected_dead_pixels(None);
            state.status_message = format!("Pixel-mask detection failed: {e}");
            return;
        }
    }

    state.log_provenance(
        ProvenanceEventKind::Normalized,
        "Counts prepared (energy grid + pixel mask)",
    );
    state.status_message = "Counts prepared — counts-KL fit engine re-point pending".into();
}

/// Compute energy bin centers from the spectrum file loaded in state.
fn compute_energies(state: &AppState, n_tof: usize) -> Result<Vec<f64>, String> {
    let values = state.spectrum_values.as_ref().ok_or_else(|| {
        "No spectrum file loaded — a spectrum file is required to define \
         the energy axis. Load a spectrum file in the Load step before \
         proceeding."
            .to_string()
    })?;

    let energies = match (state.spectrum_unit, state.spectrum_kind) {
        (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinEdges) => {
            nereids_io::tof::tof_edges_to_energy_centers(values, &state.beamline)
                .map(|a| a.to_vec())
                .map_err(|e| format!("{e}"))?
        }
        (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinCenters) => {
            if !state.beamline.flight_path_m.is_finite() || state.beamline.flight_path_m <= 0.0 {
                return Err("Flight path must be positive and finite".into());
            }
            let mut energies: Vec<f64> = values
                .iter()
                .map(|&tof| {
                    let corrected = tof - state.beamline.delay_us;
                    if corrected <= 0.0 || !corrected.is_finite() {
                        return Err(format!(
                            "TOF {tof:.2} \u{03bc}s - delay {:.2} \u{03bc}s = {corrected:.2} \u{03bc}s is not positive",
                            state.beamline.delay_us
                        ));
                    }
                    Ok(nereids_core::constants::tof_to_energy(
                        corrected,
                        state.beamline.flight_path_m,
                    ))
                })
                .collect::<Result<Vec<f64>, String>>()?;
            energies.reverse();
            energies
        }
        (SpectrumUnit::EnergyEv, SpectrumValueKind::BinEdges) => {
            if values.iter().any(|&v| v <= 0.0) {
                return Err("Energy bin edges must be positive for geometric mean".into());
            }
            values.windows(2).map(|w| (w[0] * w[1]).sqrt()).collect()
        }
        (SpectrumUnit::EnergyEv, SpectrumValueKind::BinCenters) => {
            if values.iter().any(|&v| v <= 0.0) {
                return Err("Energy bin centers must be positive".into());
            }
            values.to_vec()
        }
    };

    if energies.len() != n_tof {
        return Err(format!(
            "Energy grid has {} points but data has {} frames — check spectrum unit/kind settings",
            energies.len(),
            n_tof
        ));
    }

    Ok(energies)
}
