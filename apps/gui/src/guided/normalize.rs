//! Step 3: Normalize — transmission computation and energy axis derivation.

use crate::state::{AppState, InputMode};
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use std::sync::Arc;

/// Draw the Normalize step content.
pub fn normalize_step(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Normalize");
    ui.separator();

    match state.input_mode {
        InputMode::TransmissionTiff => {
            // Pre-normalized data: skip normalization, just derive energy axis
            if state.normalized.is_some() && state.energies.is_some() {
                ui.label("Transmission data ready (pre-normalized).");
                show_energy_info(ui, state);
            } else if state.sample_data.is_some() && state.spectrum_values.is_some() {
                if ui.button("Prepare Transmission").clicked() {
                    prepare_transmission(state);
                }
            } else {
                ui.label("Load transmission TIFF and spectrum file first (Step 1).");
            }
        }
        InputMode::TiffPair => {
            // Standard normalization: sample + open beam
            let can_normalize =
                state.sample_data.is_some() && state.open_beam_data.is_some() && !state.is_fitting;
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
                show_energy_info(ui, state);
            } else if state.sample_data.is_some() && state.open_beam_data.is_some() {
                ui.label("Click Normalize to compute transmission.");
            } else {
                ui.label("Load sample and open beam data first (Step 1).");
            }
        }
    }

    ui.add_space(16.0);
    ui.label(
        egui::RichText::new(
            "Full normalize preview (spectrum viewer, analysis mode selection) coming in Phase 2b.",
        )
        .italics()
        .color(crate::theme::semantic::ORANGE),
    );
}

/// Show energy axis info after normalization.
fn show_energy_info(ui: &mut egui::Ui, state: &AppState) {
    if let Some(ref energies) = state.energies {
        ui.label(format!(
            "Energy axis: {} bins, [{:.2}, {:.2}] eV",
            energies.len(),
            energies.first().copied().unwrap_or(0.0),
            energies.last().copied().unwrap_or(0.0),
        ));
    }
}

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
            state.status_message = "Normalization complete".into();
        }
        Err(e) => {
            state.status_message = format!("Normalization error: {}", e);
        }
    }
}

/// Prepare pre-normalized transmission data (TransmissionTiff mode).
fn prepare_transmission(state: &mut AppState) {
    let sample = match state.sample_data {
        Some(ref d) => d.clone(),
        None => return,
    };

    // No open beam in transmission mode — dead pixel detection not applicable
    state.dead_pixels = None;

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
                                "TOF {:.2} µs - delay {:.2} µs = {:.2} µs is not positive",
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
