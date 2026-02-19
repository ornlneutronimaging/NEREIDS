//! Fitting controls panel: isotope selection, parameters, run fit.

use crate::state::{AppState, EndfFetchResult, IsotopeEntry, RoiSelection, Tab};
use nereids_endf::retrieval::EndfLibrary;
use nereids_pipeline::pipeline::FitConfig;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Draw the fitting controls panel.
pub fn fitting_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Isotopes");
    ui.separator();

    // ENDF library selector
    ui.horizontal(|ui| {
        ui.label("Library:");
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
                ui.selectable_value(&mut state.endf_library, EndfLibrary::Jeff3_3, "JEFF-3.3");
                ui.selectable_value(&mut state.endf_library, EndfLibrary::Jendl5, "JENDL-5");
            });
    });

    ui.add_space(4.0);

    // Disable isotope add/remove/edit while ENDF fetch is in progress
    // to prevent index mismatch between fetch results and modified list.
    let isotope_locked = state.is_fetching_endf;

    // Add isotope row
    ui.add_enabled_ui(!isotope_locked, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Add Isotope").clicked() {
                state.isotope_entries.push(IsotopeEntry {
                    z: 92,
                    a: 238,
                    symbol: "U-238".into(),
                    initial_density: 0.001,
                    resonance_data: None,
                    enabled: true,
                });
            }
        });
    });

    // Isotope list
    let mut to_remove = None;
    for (idx, entry) in state.isotope_entries.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!isotope_locked, |ui| {
                ui.checkbox(&mut entry.enabled, "");

                let z_changed = ui
                    .add(
                        egui::DragValue::new(&mut entry.z)
                            .prefix("Z=")
                            .range(1..=118),
                    )
                    .changed();
                let a_changed = ui
                    .add(
                        egui::DragValue::new(&mut entry.a)
                            .prefix("A=")
                            .range(1..=300),
                    )
                    .changed();

                if z_changed || a_changed {
                    entry.symbol = format!(
                        "{}-{}",
                        nereids_core::elements::element_symbol(entry.z).unwrap_or("??"),
                        entry.a
                    );
                    entry.resonance_data = None;
                }
            });

            ui.label(&entry.symbol);

            ui.add_enabled_ui(!isotope_locked, |ui| {
                ui.add(
                    egui::DragValue::new(&mut entry.initial_density)
                        .prefix("rho0=")
                        .speed(0.0001)
                        .range(0.0..=1.0),
                );
            });

            if entry.resonance_data.is_some() {
                ui.label("OK");
            }

            ui.add_enabled_ui(!isotope_locked, |ui| {
                if ui.small_button("X").clicked() {
                    to_remove = Some(idx);
                }
            });
        });
    }
    if let Some(idx) = to_remove {
        state.isotope_entries.remove(idx);
    }

    // Fetch ENDF data for isotopes missing resonance data
    let has_missing = state
        .isotope_entries
        .iter()
        .any(|e| e.enabled && e.resonance_data.is_none());
    ui.add_enabled_ui(has_missing && !state.is_fetching_endf, |ui| {
        if ui.button("Fetch ENDF Data").clicked() {
            fetch_endf_data(state);
        }
    });
    if state.is_fetching_endf {
        ui.spinner();
    }

    ui.add_space(8.0);
    ui.separator();

    // --- Fit parameters ---
    ui.heading("Fit Parameters");
    ui.horizontal(|ui| {
        ui.label("Temperature (K):");
        ui.add(
            egui::DragValue::new(&mut state.temperature_k)
                .range(0.0..=3000.0)
                .speed(1.0),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Max iter:");
        ui.add(egui::DragValue::new(&mut state.lm_config.max_iter).range(1..=10000));
    });

    ui.add_space(8.0);
    ui.separator();

    // --- ROI ---
    ui.heading("Region of Interest");
    if let Some(ref norm) = state.normalized {
        let shape = norm.transmission.shape();
        let height = shape[1];
        let width = shape[2];

        let mut roi = state.roi.unwrap_or(RoiSelection {
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
        });

        let changed = ui
            .horizontal(|ui| {
                let mut changed = false;
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.y_start)
                            .prefix("y0=")
                            .range(0..=height),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.y_end)
                            .prefix("y1=")
                            .range(0..=height),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.x_start)
                            .prefix("x0=")
                            .range(0..=width),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut roi.x_end)
                            .prefix("x1=")
                            .range(0..=width),
                    )
                    .changed();
                changed
            })
            .inner;

        if changed || state.roi.is_none() {
            state.roi = Some(roi);
        }
    }

    ui.add_space(8.0);
    ui.separator();

    // --- Run buttons ---
    ui.heading("Run");

    let ready = state.normalized.is_some()
        && state.energies.is_some()
        && state
            .isotope_entries
            .iter()
            .any(|e| e.enabled && e.resonance_data.is_some());

    ui.add_enabled_ui(ready && !state.is_fitting, |ui| {
        if ui.button("Fit Selected Pixel").clicked() {
            fit_pixel(state);
        }
        if ui.button("Fit ROI Average").clicked() {
            fit_roi(state);
        }
        if ui.button("Spatial Map (all pixels)").clicked() {
            run_spatial_map(state);
        }
    });

    if state.is_fitting {
        ui.spinner();
        ui.label("Fitting...");
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

    // Collect work items: (index, isotope, symbol, mat_number)
    let mut work: Vec<(usize, Isotope, String, EndfLibrary)> = Vec::new();
    for (i, entry) in state.isotope_entries.iter().enumerate() {
        if entry.enabled && entry.resonance_data.is_none() {
            let isotope = Isotope::new(entry.z, entry.a);
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
            let mat = retrieval::mat_number(&isotope).unwrap();
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
        // tx drops here, closing the channel
    });
}

fn build_fit_config(state: &AppState) -> Option<FitConfig> {
    let energies = state.energies.as_ref()?.clone();

    let enabled: Vec<&IsotopeEntry> = state
        .isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .collect();

    if enabled.is_empty() {
        return None;
    }

    let resonance_data: Vec<_> = enabled
        .iter()
        .filter_map(|e| e.resonance_data.clone())
        .collect();
    let isotope_names: Vec<_> = enabled.iter().map(|e| e.symbol.clone()).collect();
    let initial_densities: Vec<_> = enabled.iter().map(|e| e.initial_density).collect();

    Some(FitConfig {
        energies,
        resonance_data,
        isotope_names,
        temperature_k: state.temperature_k,
        resolution: None,
        initial_densities,
        lm_config: state.lm_config.clone(),
    })
}

fn fit_pixel(state: &mut AppState) {
    let config = match build_fit_config(state) {
        Some(c) => c,
        None => {
            state.status_message = "Missing fit configuration".into();
            return;
        }
    };

    let (y, x) = match state.selected_pixel {
        Some(px) => px,
        None => {
            state.status_message = "No pixel selected".into();
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let n_energies = norm.transmission.shape()[0];
    let t_spectrum: Vec<f64> = (0..n_energies)
        .map(|e| norm.transmission[[e, y, x]])
        .collect();
    let sigma: Vec<f64> = (0..n_energies)
        .map(|e| norm.uncertainty[[e, y, x]].max(1e-10))
        .collect();

    let result = nereids_pipeline::pipeline::fit_spectrum(&t_spectrum, &sigma, &config);

    state.status_message = if result.converged {
        format!(
            "Pixel ({},{}) fit converged, chi2_r = {:.4}",
            y, x, result.reduced_chi_squared
        )
    } else {
        format!("Pixel ({},{}) fit did NOT converge", y, x)
    };

    state.pixel_fit_result = Some(result);
    state.active_tab = Tab::Spectrum;
}

fn fit_roi(state: &mut AppState) {
    let config = match build_fit_config(state) {
        Some(c) => c,
        None => {
            state.status_message = "Missing fit configuration".into();
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => n,
        None => return,
    };

    let roi = match state.roi {
        Some(r) => r,
        None => {
            state.status_message = "No ROI defined".into();
            return;
        }
    };

    if roi.y_start >= roi.y_end || roi.x_start >= roi.x_end {
        state.status_message = "Invalid ROI: start must be less than end".into();
        return;
    }

    let shape = norm.transmission.shape();
    if roi.y_end > shape[1] || roi.x_end > shape[2] {
        state.status_message = "ROI exceeds image dimensions".into();
        return;
    }

    let result = nereids_pipeline::spatial::fit_roi(
        &norm.transmission,
        &norm.uncertainty,
        roi.y_start..roi.y_end,
        roi.x_start..roi.x_end,
        &config,
    );

    state.status_message = if result.converged {
        format!(
            "ROI fit converged, chi2_r = {:.4}",
            result.reduced_chi_squared
        )
    } else {
        "ROI fit did NOT converge".into()
    };

    state.pixel_fit_result = Some(result);
    state.active_tab = Tab::Spectrum;
}

fn run_spatial_map(state: &mut AppState) {
    let config = match build_fit_config(state) {
        Some(c) => c,
        None => {
            state.status_message = "Missing fit configuration".into();
            return;
        }
    };

    let norm = match state.normalized {
        Some(ref n) => Arc::clone(n),
        None => return,
    };

    let dead_pixels = state.dead_pixels.clone();

    let (tx, rx) = mpsc::channel();
    state.pending_spatial = Some(rx);
    state.is_fitting = true;
    state.status_message = "Running spatial mapping...".into();
    let cancel = Arc::clone(&state.cancel_token);

    std::thread::spawn(move || {
        let result = nereids_pipeline::spatial::spatial_map(
            &norm.transmission,
            &norm.uncertainty,
            &config,
            dead_pixels.as_ref(),
        );
        // Only send result if not cancelled — receiver may already be dropped
        if !cancel.load(Ordering::Relaxed) {
            let _ = tx.send(result);
        }
    });
}
