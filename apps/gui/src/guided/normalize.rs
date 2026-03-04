//! Step 3: Normalize — transmission computation, preview, and analysis mode selection.

use crate::state::{
    AnalysisMode, AppState, GuidedStep, InputMode, ProvenanceEventKind, SpectrumAxis,
    SpectrumDataSource,
};
use crate::widgets::design::{self, NavAction};
use crate::widgets::image_view::show_viridis_image;
use egui_plot::{Line, Plot, PlotPoints, VLine};
use ndarray::{Array3, Axis};
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use std::sync::Arc;

/// Draw the Normalize step content.
///
/// No inner ScrollArea -- the guided content area in app.rs already wraps
/// everything in a vertical ScrollArea, so nesting would clip content.
pub fn normalize_step(ui: &mut egui::Ui, state: &mut AppState) {
    let subtitle = match state.input_mode {
        InputMode::TiffPair => "Normalize raw data and preview transmission",
        _ => "Preview transmission and select analysis mode",
    };
    design::content_header(ui, "Normalize", subtitle);

    let has_data = state.normalized.is_some() && state.energies.is_some();

    // --- Normalization status + Analysis Mode (compact, full width) ---
    normalization_controls_card(ui, state);

    if has_data {
        analysis_mode_card(ui, state);

        if let (Some(norm), Some(energies)) = (&state.normalized, &state.energies) {
            let shape = norm.transmission.shape();
            let bins = format!("{}", shape[0]);
            let pixels = format!("{}", shape[1] * shape[2]);
            let e_min = format!("{:.2}", energies.first().unwrap_or(&0.0));
            let e_max = format!("{:.2}", energies.last().unwrap_or(&0.0));
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
    }

    let can_continue = has_data;
    match design::nav_buttons(
        ui,
        Some("\u{2190} Back"),
        "Continue \u{2192}",
        can_continue,
        "Normalize data to continue",
    ) {
        NavAction::Back => state.guided_step = GuidedStep::Configure,
        NavAction::Continue => state.guided_step = GuidedStep::Analyze,
        NavAction::None => {}
    }

    // --- Transmission preview (full width, tall) ---
    if has_data {
        ui.add_space(8.0);
        transmission_preview_card(ui, state);
    }
}

// ---- Normalization Controls Card ----

fn normalization_controls_card(ui: &mut egui::Ui, state: &mut AppState) {
    design::card_with_header(ui, "Normalization", None, |ui| {
        match state.input_mode {
            InputMode::TransmissionTiff | InputMode::Hdf5Histogram | InputMode::Hdf5Event => {
                if state.normalized.is_some() && state.energies.is_some() {
                    let label = match state.input_mode {
                        InputMode::Hdf5Histogram => "HDF5 histogram data ready.",
                        InputMode::Hdf5Event => "Histogrammed event data ready.",
                        _ => "Transmission data ready (pre-normalized).",
                    };
                    ui.label(label);
                } else if state.sample_data.is_some() && state.spectrum_values.is_some() {
                    // Auto-prepare: pre-normalized/HDF5 data needs no user action.
                    prepare_transmission(state);
                    ui.label("Preparing data...");
                } else {
                    ui.label("Load data first (Step 1).");
                }
            }
            InputMode::TiffPair => {
                // Editable proton charge parameters
                egui::Grid::new("norm_params")
                    .num_columns(4)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("PC sample:");
                        ui.add(
                            egui::DragValue::new(&mut state.proton_charge_sample)
                                .range(0.001..=1e6)
                                .speed(0.01),
                        );
                        ui.label("PC open beam:");
                        ui.add(
                            egui::DragValue::new(&mut state.proton_charge_ob)
                                .range(0.001..=1e6)
                                .speed(0.01),
                        );
                        ui.end_row();
                    });
                ui.add_space(6.0);

                let can_normalize = state.sample_data.is_some()
                    && state.open_beam_data.is_some()
                    && !state.is_fitting;
                ui.add_enabled_ui(can_normalize, |ui| {
                    if ui.button("Normalize").clicked() {
                        normalize_data(state);
                    }
                });

                if state.normalized.is_some() {
                    ui.label("Transmission computed.");
                    if let Some(ref dead) = state.dead_pixels {
                        let n_dead = dead.iter().filter(|&&d| d).count();
                        ui.label(format!("Dead pixels: {}", n_dead));
                    }
                } else if state.sample_data.is_some() && state.open_beam_data.is_some() {
                    ui.label("Click Normalize to compute transmission.");
                } else {
                    ui.label("Load sample and open beam data first (Step 1).");
                }
            }
        }
    });
}

// ---- Transmission Preview Card ----

fn transmission_preview_card(ui: &mut egui::Ui, state: &mut AppState) {
    design::card_with_header(ui, "Transmission Preview", None, |ui| {
        let available_width = ui.available_width();
        let image_width = 220.0_f32.min(available_width * 0.35);

        ui.horizontal(|ui| {
            // Left: image + TOF slicer
            ui.allocate_ui_with_layout(
                egui::vec2(image_width, ui.available_height()),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    preview_image_panel(ui, state);
                },
            );

            ui.separator();

            // Right: spectrum plot + controls
            ui.vertical(|ui| {
                preview_spectrum_panel(ui, state);
            });
        });
    });
}

fn preview_image_panel(ui: &mut egui::Ui, state: &mut AppState) {
    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let n_tof = norm.transmission.shape()[0];
    if n_tof == 0 {
        ui.label("(no data)");
        return;
    }

    // Clamp TOF index
    if state.tof_slice_index >= n_tof {
        state.tof_slice_index = n_tof - 1;
    }

    // Extract 2D slice at current TOF index
    let slice = norm
        .transmission
        .index_axis(Axis(0), state.tof_slice_index)
        .to_owned();
    let _ = show_viridis_image(ui, &slice, "norm_preview_tex");

    // TOF slicer
    ui.add(
        egui::Slider::new(&mut state.tof_slice_index, 0..=n_tof.saturating_sub(1)).text("TOF bin"),
    );
}

/// Compute the spatially-averaged transmission spectrum (mean over y, x axes).
///
/// Returns `None` when either spatial dimension has length 0, which would cause
/// `ndarray::mean_axis()` to return `None`.
fn full_image_average(transmission: &Array3<f64>) -> Option<Vec<f64>> {
    let avg_x = transmission.mean_axis(Axis(2))?;
    let avg_xy = avg_x.mean_axis(Axis(1))?;
    Some(avg_xy.to_vec())
}

fn preview_spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    // Controls row
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

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let n_tof = norm.transmission.shape()[0];

    // Compute averaged spectrum based on data source
    let spectrum: Vec<f64> = match state.normalize_spectrum_source {
        SpectrumDataSource::FullImage => match full_image_average(&norm.transmission) {
            Some(avg) => avg,
            None => {
                ui.label("(empty spatial dimensions)");
                return;
            }
        },
        SpectrumDataSource::RoiAverage => {
            if let Some(roi) = state.roi {
                match nereids_io::normalization::average_roi(
                    &norm.transmission,
                    roi.y_start..roi.y_end,
                    roi.x_start..roi.x_end,
                ) {
                    Ok(avg) => avg.to_vec(),
                    Err(_) => {
                        ui.label("Invalid ROI \u{2014} using full image.");
                        match full_image_average(&norm.transmission) {
                            Some(avg) => avg,
                            None => {
                                ui.label("(empty spatial dimensions)");
                                return;
                            }
                        }
                    }
                }
            } else {
                ui.label("No ROI set \u{2014} showing full image average.");
                match full_image_average(&norm.transmission) {
                    Some(avg) => avg,
                    None => {
                        ui.label("(empty spatial dimensions)");
                        return;
                    }
                }
            }
        }
    };

    // Build x-axis values and label, respecting spectrum_unit and spectrum_kind.
    let (x_values, x_label): (Vec<f64>, &str) = match state.normalize_spectrum_axis {
        SpectrumAxis::EnergyEv => match state.energies {
            Some(ref e) => (e.clone(), "Energy (eV)"),
            None => return,
        },
        SpectrumAxis::TofMicroseconds => match state.spectrum_values {
            Some(ref v) => match (state.spectrum_unit, state.spectrum_kind) {
                (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinEdges) => {
                    // Compute arithmetic mean bin centers from adjacent edges.
                    let centers: Vec<f64> = v
                        .windows(2)
                        .take(n_tof)
                        .map(|w| 0.5 * (w[0] + w[1]))
                        .collect();
                    (centers, "TOF (\u{03bc}s)")
                }
                (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinCenters) => {
                    (v.iter().take(n_tof).copied().collect(), "TOF (\u{03bc}s)")
                }
                (SpectrumUnit::EnergyEv, _) => {
                    // Spectrum file is in energy units — convert to TOF for the plot axis.
                    if state.beamline.flight_path_m.is_finite()
                        && state.beamline.flight_path_m > 0.0
                    {
                        let tof_vals: Vec<f64> = v
                            .iter()
                            .take(n_tof)
                            .map(|&e| {
                                nereids_core::constants::energy_to_tof(
                                    e,
                                    state.beamline.flight_path_m,
                                ) + state.beamline.delay_us
                            })
                            .collect();
                        (tof_vals, "TOF (\u{03bc}s)")
                    } else {
                        // Cannot convert without valid beamline params — show energy instead.
                        (v.iter().take(n_tof).copied().collect(), "Energy (eV)")
                    }
                }
            },
            None => {
                // No spectrum values — use frame indices with honest label.
                let indices: Vec<f64> = (0..n_tof).map(|i| i as f64).collect();
                (indices, "Frame index")
            }
        },
    };

    let n_plot = spectrum.len().min(x_values.len());
    if n_plot == 0 {
        return;
    }

    let points: PlotPoints = (0..n_plot).map(|i| [x_values[i], spectrum[i]]).collect();
    let line = Line::new("Transmission", points);

    // TOF marker line
    let tof_marker_x = x_values
        .get(state.tof_slice_index.min(n_plot.saturating_sub(1)))
        .copied();

    Plot::new("normalize_spectrum_plot")
        .x_axis_label(x_label)
        .y_axis_label("Transmission")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(line);

            // Current TOF position marker
            if let Some(x) = tof_marker_x {
                plot_ui.vline(
                    VLine::new("TOF position", x)
                        .color(egui::Color32::from_rgb(255, 165, 0))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            }

            // Resonance dip markers (energy axis only)
            if state.show_resonance_dips && state.normalize_spectrum_axis == SpectrumAxis::EnergyEv
            {
                let x_min = x_values[0].min(x_values[n_plot - 1]);
                let x_max = x_values[0].max(x_values[n_plot - 1]);
                for entry in &state.isotope_entries {
                    if !entry.enabled {
                        continue;
                    }
                    let Some(ref res_data) = entry.resonance_data else {
                        continue;
                    };
                    for range in &res_data.ranges {
                        for lg in &range.l_groups {
                            for res in &lg.resonances {
                                if res.energy >= x_min && res.energy <= x_max {
                                    plot_ui.vline(
                                        VLine::new(
                                            format!("{} {:.1}eV", entry.symbol, res.energy),
                                            res.energy,
                                        )
                                        .color(egui::Color32::from_rgb(180, 80, 80))
                                        .style(egui_plot::LineStyle::dashed_loose()),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });
}

// ---- Analysis Mode Card ----

fn analysis_mode_card(ui: &mut egui::Ui, state: &mut AppState) {
    design::card_with_header(ui, "Analysis Mode", None, |ui| {
        ui.horizontal(|ui| {
            // Full Spatial Map
            let is_spatial = state.analysis_mode == AnalysisMode::FullSpatialMap;
            if ui.radio(is_spatial, "Full Spatial Map").clicked() {
                state.analysis_mode = AnalysisMode::FullSpatialMap;
            }
            ui.label(
                egui::RichText::new("Fit every pixel independently.")
                    .small()
                    .weak(),
            );

            ui.add_space(16.0);

            // ROI Single Spectrum
            let is_roi = state.analysis_mode == AnalysisMode::RoiSingleSpectrum;
            if ui.radio(is_roi, "ROI \u{2192} Single Spectrum").clicked() {
                state.analysis_mode = AnalysisMode::RoiSingleSpectrum;
            }
            ui.label(egui::RichText::new("Average ROI, fit once.").small().weak());

            ui.add_space(16.0);

            // Spatial Binning
            let is_binning = matches!(state.analysis_mode, AnalysisMode::SpatialBinning(_));
            if ui.radio(is_binning, "Spatial Binning").clicked() && !is_binning {
                state.analysis_mode = AnalysisMode::SpatialBinning(2);
            }
            if is_binning {
                let mut bin_size = match state.analysis_mode {
                    AnalysisMode::SpatialBinning(n) => n,
                    _ => 2,
                };
                egui::ComboBox::from_id_salt("bin_size")
                    .selected_text(format!("{}x{}", bin_size, bin_size))
                    .width(50.0)
                    .show_ui(ui, |ui| {
                        for &n in &[2u8, 4, 8] {
                            ui.selectable_value(&mut bin_size, n, format!("{}x{}", n, n));
                        }
                    });
                state.analysis_mode = AnalysisMode::SpatialBinning(bin_size);
            } else {
                ui.label(egui::RichText::new("Bin NxN, fit map.").small().weak());
            }
        });
    });
}

// ---- Normalization Logic (unchanged from Phase 2a) ----

/// Standard normalization: sample + open beam → transmission.
fn normalize_data(state: &mut AppState) {
    state.cancel_pending_tasks();
    state.pixel_fit_result = None;
    state.spatial_result = None;

    let sample = match state.sample_data {
        Some(ref d) => d,
        None => return,
    };
    let open_beam = match state.open_beam_data {
        Some(ref d) => d,
        None => return,
    };

    let params = nereids_io::normalization::NormalizationParams {
        proton_charge_sample: state.proton_charge_sample,
        proton_charge_ob: state.proton_charge_ob,
    };

    match nereids_io::normalization::normalize(sample, open_beam, &params, None) {
        Ok(norm) => {
            state.dead_pixels = Some(nereids_io::normalization::detect_dead_pixels(sample));

            let n_tof = sample.shape()[0];
            match compute_energies(state, n_tof) {
                Ok(energies) => state.energies = Some(energies),
                Err(e) => {
                    state.normalized = None;
                    state.energies = None;
                    state.status_message = format!("Energy conversion: {}", e);
                    return;
                }
            }

            state.normalized = Some(Arc::new(norm));
            state.log_provenance(
                ProvenanceEventKind::Normalized,
                "Normalization complete (Method 2)",
            );
            state.status_message = "Normalization complete".into();
        }
        Err(e) => {
            state.status_message = format!("Normalization error: {}", e);
        }
    }
}

/// Prepare pre-normalized transmission data (TransmissionTiff mode).
///
/// Called automatically when TransmissionTiff data is ready but not yet
/// wrapped as `NormalizedData`.  Idempotent — does nothing if already prepared.
pub(crate) fn prepare_transmission(state: &mut AppState) {
    state.cancel_pending_tasks();
    state.pixel_fit_result = None;
    state.spatial_result = None;

    let sample = match state.sample_data {
        Some(ref d) => d.clone(),
        None => return,
    };

    // Only clear dead pixels for TransmissionTiff — HDF5 modes load them from the file.
    if state.input_mode == InputMode::TransmissionTiff {
        state.dead_pixels = None;
    }

    let n_tof = sample.shape()[0];
    // TODO(Phase 2b): estimate uncertainty from data or allow user to specify.
    // Using uniform synthetic uncertainty since no open beam is available.
    let uncertainty = ndarray::Array3::from_elem(sample.raw_dim(), 0.01);

    match compute_energies(state, n_tof) {
        Ok(energies) => state.energies = Some(energies),
        Err(e) => {
            state.status_message = format!("Energy conversion: {}", e);
            return;
        }
    }

    state.normalized = Some(Arc::new(nereids_io::normalization::NormalizedData {
        transmission: sample,
        uncertainty,
    }));
    state.log_provenance(
        ProvenanceEventKind::Normalized,
        "Transmission data prepared",
    );
    state.status_message = "Transmission ready (synthetic uncertainty — see docs)".into();
}

/// Compute energy bin centers from spectrum file or synthetic TOF edges.
///
/// Uses the spectrum file values and unit/kind settings from state.
/// Falls back to synthetic linear TOF edges if no spectrum file is loaded
/// (backward compatibility).
fn compute_energies(state: &AppState, n_tof: usize) -> Result<Vec<f64>, String> {
    if let Some(ref values) = state.spectrum_values {
        let energies = match (state.spectrum_unit, state.spectrum_kind) {
            (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinEdges) => {
                // TOF edges → energy centers via geometric mean
                nereids_io::tof::tof_edges_to_energy_centers(values, &state.beamline)
                    .map(|a| a.to_vec())
                    .map_err(|e| format!("{}", e))?
            }
            (SpectrumUnit::TofMicroseconds, SpectrumValueKind::BinCenters) => {
                // Convert each TOF center to energy directly
                if !state.beamline.flight_path_m.is_finite() || state.beamline.flight_path_m <= 0.0
                {
                    return Err("Flight path must be positive and finite".into());
                }
                let mut energies: Vec<f64> = values
                    .iter()
                    .map(|&tof| {
                        let corrected = tof - state.beamline.delay_us;
                        if corrected <= 0.0 || !corrected.is_finite() {
                            return Err(format!(
                                "TOF {:.2} \u{03bc}s - delay {:.2} \u{03bc}s = {:.2} \u{03bc}s is not positive",
                                tof, state.beamline.delay_us, corrected
                            ));
                        }
                        Ok(nereids_core::constants::tof_to_energy(
                            corrected,
                            state.beamline.flight_path_m,
                        ))
                    })
                    .collect::<Result<Vec<f64>, String>>()?;
                // TOF ascending → energy descending, so reverse
                energies.reverse();
                energies
            }
            (SpectrumUnit::EnergyEv, SpectrumValueKind::BinEdges) => {
                // Energy edges → geometric mean centers
                if values.iter().any(|&v| v <= 0.0) {
                    return Err("Energy bin edges must be positive for geometric mean".into());
                }
                values.windows(2).map(|w| (w[0] * w[1]).sqrt()).collect()
            }
            (SpectrumUnit::EnergyEv, SpectrumValueKind::BinCenters) => {
                // Direct: energy centers
                if values.iter().any(|&v| v <= 0.0) {
                    return Err("Energy bin centers must be positive".into());
                }
                values.clone()
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
    } else {
        // Fallback: synthetic linear TOF edges (backward compatibility)
        let tof_edges =
            nereids_io::tof::linspace_tof_edges(state.tof_min_us, state.tof_max_us, n_tof)
                .map_err(|e| format!("{}", e))?;
        nereids_io::tof::tof_edges_to_energy_centers(&tof_edges, &state.beamline)
            .map(|a| a.to_vec())
            .map_err(|e| format!("{}", e))
    }
}
