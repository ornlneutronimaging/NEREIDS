//! Forward Model tool — independent isotope sandbox with live spectrum preview.

use crate::state::{
    AppState, EndfStatus, GuidedStep, IsotopeEntry, PeriodicTableTarget, SpectrumAxis,
};
use crate::widgets::design;
use egui_plot::{Line, Plot, PlotPoints};
use nereids_endf::retrieval::EndfLibrary;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};
use std::sync::Arc;
use std::sync::mpsc;

/// Draw the Forward Model tool content.
pub fn forward_model_step(ui: &mut egui::Ui, state: &mut AppState) {
    // -- Header row: title + teleport pills + sync + axis toggle --
    ui.horizontal(|ui| {
        design::content_header(ui, "Forward Model", "Simulated transmission");
    });
    ui.horizontal(|ui| {
        design::teleport_pill(ui, "← Configure", GuidedStep::Configure, state);
        design::teleport_pill(ui, "Analyze →", GuidedStep::Analyze, state);
        ui.separator();
        // Sync buttons (disabled during active ENDF fetches to prevent index corruption)
        ui.add_enabled_ui(
            !state.is_fetching_fm_endf && !state.is_fetching_endf,
            |ui| {
                if ui.button("Copy from Config").clicked() {
                    copy_config_to_fm(state);
                }
            },
        );
        ui.add_enabled_ui(!state.is_fetching_endf, |ui| {
            if ui.button("Push to Config").clicked() {
                push_fm_to_config(state);
            }
        });
        ui.separator();
        // Axis toggle
        ui.label("Axis:");
        ui.selectable_value(
            &mut state.fm_spectrum_axis,
            SpectrumAxis::EnergyEv,
            "Energy (eV)",
        );
        ui.selectable_value(
            &mut state.fm_spectrum_axis,
            SpectrumAxis::TofMicroseconds,
            "TOF (\u{03bc}s)",
        );
    });

    ui.add_space(8.0);

    // -- Hero Spectrum (full width) --
    fm_spectrum_panel(ui, state);
    ui.add_space(12.0);

    // -- Instrument Resolution Card --
    fm_resolution_card(ui, state);
    ui.add_space(12.0);

    // -- Isotopes Card --
    fm_isotopes_card(ui, state);
}

/// Build an optional InstrumentParams from the FM resolution state.
/// Returns `(instrument, warning)` — warning is set if resolution is enabled
/// but parameters are invalid (e.g., flight path not configured).
pub(crate) fn fm_instrument(state: &AppState) -> (Option<InstrumentParams>, Option<&'static str>) {
    match design::build_resolution_function(
        state.fm_resolution_enabled,
        &state.fm_resolution_mode,
        state.beamline.flight_path_m,
    ) {
        Ok(Some(resolution)) => (Some(InstrumentParams { resolution }), None),
        Ok(None) => (None, None),
        Err(_) => (
            None,
            Some("Resolution enabled but parameters invalid \u{2014} broadening disabled"),
        ),
    }
}

/// Resolution card using the shared design widget.
pub(crate) fn fm_resolution_card(ui: &mut egui::Ui, state: &mut AppState) {
    let res = design::resolution_card(
        ui,
        &mut state.fm_resolution_enabled,
        &mut state.fm_resolution_mode,
        state.beamline.flight_path_m,
    );
    if res.changed {
        state.fm_spectrum = None;
        state.fm_per_isotope_spectra.clear();
    }
}

/// Isotopes card: library, temperature, isotope list with density sliders.
pub(crate) fn fm_isotopes_card(ui: &mut egui::Ui, state: &mut AppState) {
    let isotope_locked = state.is_fetching_fm_endf;

    // Card header: Library + Temperature inline
    let lib_label = design::library_name(state.fm_endf_library);
    let header = format!(
        "Isotopes  \u{2014}  {lib_label}  T={:.0}K",
        state.fm_temperature_k
    );
    design::card_with_header(ui, &header, None, |ui| {
        // Library selector
        let prev_lib = state.fm_endf_library;
        ui.add_enabled_ui(!isotope_locked, |ui| {
            ui.horizontal(|ui| {
                ui.label("Library:");
                egui::ComboBox::from_id_salt("fm_endf_lib")
                    .selected_text(design::library_name(state.fm_endf_library))
                    .show_ui(ui, |ui| {
                        for (val, label) in [
                            (EndfLibrary::EndfB8_0, "ENDF/B-VIII.0"),
                            (EndfLibrary::EndfB8_1, "ENDF/B-VIII.1"),
                            (EndfLibrary::Jeff3_3, "JEFF-3.3"),
                            (EndfLibrary::Jendl5, "JENDL-5"),
                        ] {
                            ui.selectable_value(&mut state.fm_endf_library, val, label);
                        }
                    });
                ui.label("Temp (K):");
                if ui
                    .add(
                        egui::DragValue::new(&mut state.fm_temperature_k)
                            .range(1.0..=5000.0)
                            .speed(1.0),
                    )
                    .changed()
                {
                    state.fm_spectrum = None;
                    state.fm_per_isotope_spectra.clear();
                }
            });
        });
        if state.fm_endf_library != prev_lib {
            for e in &mut state.fm_isotope_entries {
                e.resonance_data = None;
                e.endf_status = EndfStatus::Pending;
            }
            state.fm_spectrum = None;
            state.fm_per_isotope_spectra.clear();
        }

        ui.add_space(4.0);

        // Isotope list with density sliders
        let mut to_remove = None;
        let mut changed = false;
        for (idx, entry) in state.fm_isotope_entries.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!isotope_locked, |ui| {
                    if ui.checkbox(&mut entry.enabled, "").changed() {
                        changed = true;
                    }
                });

                // Colored dot matching isotope_dot_color
                let dot_color = design::isotope_dot_color(&entry.symbol);
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, dot_color);

                ui.label(&entry.symbol);

                ui.add_enabled_ui(!isotope_locked, |ui| {
                    // DragValue for precise density entry
                    if ui
                        .add(
                            egui::DragValue::new(&mut entry.initial_density)
                                .speed(0.0001)
                                .range(1e-6..=0.05)
                                .max_decimals(6),
                        )
                        .changed()
                    {
                        changed = true;
                    }

                    // Logarithmic slider for visual density adjustment
                    if ui
                        .add(
                            egui::Slider::new(&mut entry.initial_density, 1e-6..=0.05)
                                .logarithmic(true)
                                .show_value(false)
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });

                // ENDF status badge
                match entry.endf_status {
                    EndfStatus::Loaded => {
                        design::badge(ui, "OK", design::BadgeVariant::Green);
                    }
                    EndfStatus::Fetching => {
                        ui.spinner();
                    }
                    EndfStatus::Failed => {
                        design::badge(ui, "ERR", design::BadgeVariant::Red);
                    }
                    EndfStatus::Pending => {
                        design::badge(ui, "...", design::BadgeVariant::Orange);
                    }
                }

                ui.add_enabled_ui(!isotope_locked, |ui| {
                    if ui.small_button("\u{00d7}").clicked() {
                        to_remove = Some(idx);
                    }
                });
            });
        }
        if let Some(idx) = to_remove {
            state.fm_isotope_entries.remove(idx);
            changed = true;
        }
        if changed {
            state.fm_spectrum = None;
            state.fm_per_isotope_spectra.clear();
        }

        ui.add_space(4.0);

        // Add Isotope + Fetch buttons
        ui.add_enabled_ui(!isotope_locked, |ui| {
            if ui.button("Add Isotope...").clicked() {
                state.periodic_table_open = true;
                state.periodic_table_target = PeriodicTableTarget::ForwardModel;
                state.periodic_table_selected_z = None;
                state.periodic_table_density = 0.001; // at/barn default
            }
        });

        // Auto-fetch ENDF data when isotopes are pending
        let has_pending = state
            .fm_isotope_entries
            .iter()
            .any(|e| e.enabled && e.endf_status == EndfStatus::Pending);
        if has_pending && !state.is_fetching_fm_endf {
            fm_fetch_endf_data(state);
        }
        if state.is_fetching_fm_endf {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Fetching ENDF data…");
            });
        }
        let has_failed = state
            .fm_isotope_entries
            .iter()
            .any(|e| e.enabled && e.endf_status == EndfStatus::Failed);
        if has_failed && !state.is_fetching_fm_endf && ui.button("Retry failed").clicked() {
            for e in &mut state.fm_isotope_entries {
                if e.endf_status == EndfStatus::Failed {
                    e.endf_status = EndfStatus::Pending;
                }
            }
        }
    });
}

/// Hero spectrum plot with energy/TOF axis and per-isotope contribution lines.
pub(crate) fn fm_spectrum_panel(ui: &mut egui::Ui, state: &mut AppState) {
    // Collect enabled isotopes that have resonance data
    let enabled: Vec<_> = state
        .fm_isotope_entries
        .iter()
        .filter(|e| e.enabled && e.resonance_data.is_some())
        .collect();

    if enabled.is_empty() {
        design::card(ui, |ui| {
            ui.label(
                egui::RichText::new("Add isotopes and fetch ENDF data to see the forward model.")
                    .italics()
                    .color(crate::theme::semantic::ORANGE),
            );
        });
        return;
    }

    // Energy grid: use loaded energies if available, otherwise generate log-spaced 1..100 eV
    let energies: Vec<f64> = if let Some(ref e) = state.energies {
        e.clone()
    } else {
        let n = 2000;
        let e_min = 1.0_f64;
        let e_max = 100.0_f64;
        (0..n)
            .map(|i| e_min * (e_max / e_min).powf(i as f64 / (n - 1).max(1) as f64))
            .collect()
    };

    // Invalidate cache when energy grid changes
    let current_grid = state.energies.as_deref();
    let cached_grid = state.fm_energies.as_deref();
    if current_grid != cached_grid {
        state.fm_spectrum = None;
        state.fm_per_isotope_spectra.clear();
        state.fm_energies = None;
    }

    // Compute combined forward model if cache is stale
    if state.fm_spectrum.is_none() {
        let (instrument, res_warning) = fm_instrument(state);
        if let Some(msg) = res_warning {
            state.status_message = msg.into();
        }
        let isotopes: Vec<_> = enabled
            .iter()
            .filter_map(|e| e.resonance_data.clone().map(|rd| (rd, e.initial_density)))
            .collect();

        match SampleParams::new(state.fm_temperature_k, isotopes) {
            Ok(sample) => {
                match transmission::forward_model(&energies, &sample, instrument.as_ref()) {
                    Ok(combined) => {
                        state.fm_spectrum = Some(combined);
                    }
                    Err(e) => {
                        state.status_message = format!("Forward model error: {e}");
                        state.fm_spectrum = Some(Vec::new());
                    }
                }
            }
            Err(e) => {
                state.status_message = format!("Sample params error: {e}");
                state.fm_spectrum = Some(Vec::new());
            }
        }

        // Compute per-isotope contributions (each isotope alone)
        state.fm_per_isotope_spectra.clear();
        for entry in &enabled {
            let Some(rd) = entry.resonance_data.clone() else {
                continue;
            };
            let single = vec![(rd, entry.initial_density)];
            match SampleParams::new(state.fm_temperature_k, single) {
                Ok(sample) => {
                    match transmission::forward_model(&energies, &sample, instrument.as_ref()) {
                        Ok(t) => {
                            state.fm_per_isotope_spectra.push((entry.symbol.clone(), t));
                        }
                        Err(e) => {
                            state.status_message =
                                format!("Forward model error for {}: {e}", entry.symbol);
                        }
                    }
                }
                Err(e) => {
                    state.status_message = format!("Sample params error for {}: {e}", entry.symbol);
                }
            }
        }

        state.fm_energies = state.energies.clone();
    }

    // Use the cached energy grid for plotting
    let plot_energies = state.fm_energies.as_ref().unwrap_or(&energies);

    // Build x-axis values based on axis selection
    let (x_values, x_label): (Vec<f64>, &str) = match state.fm_spectrum_axis {
        SpectrumAxis::EnergyEv => (plot_energies.clone(), "Energy (eV)"),
        SpectrumAxis::TofMicroseconds => {
            let fp = state.beamline.flight_path_m;
            if fp.is_finite() && fp > 0.0 {
                let delay = state.beamline.delay_us;
                let tof: Vec<f64> = plot_energies
                    .iter()
                    .map(|&e| {
                        if e > 0.0 {
                            nereids_core::constants::energy_to_tof(e, fp) + delay
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                (tof, "TOF (\u{03bc}s)")
            } else {
                (plot_energies.clone(), "Energy (eV)")
            }
        }
    };

    // Hero plot — full width, min 300px tall
    let plot_height = ui.available_height().clamp(300.0, 350.0);
    Plot::new("fm_spectrum_plot")
        .height(plot_height)
        .x_axis_label(x_label)
        .y_axis_label("Transmission")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            // Combined transmission line
            if let Some(ref combined) = state.fm_spectrum {
                let n_plot = x_values.len().min(combined.len());
                let points: PlotPoints = (0..n_plot)
                    .filter(|&i| x_values[i].is_finite())
                    .map(|i| [x_values[i], combined[i]])
                    .collect();
                plot_ui.line(Line::new("Combined T(E)", points).width(2.0));
            }

            // Per-isotope contribution lines (dashed, colored by isotope hash)
            for (symbol, spectrum) in state.fm_per_isotope_spectra.iter() {
                let n_plot = x_values.len().min(spectrum.len());
                let points: PlotPoints = (0..n_plot)
                    .filter(|&i| x_values[i].is_finite())
                    .map(|i| [x_values[i], spectrum[i]])
                    .collect();
                let color = design::isotope_dot_color(symbol);
                plot_ui.line(
                    Line::new(symbol.as_str(), points)
                        .color(color)
                        .style(egui_plot::LineStyle::dashed_loose()),
                );
            }
        });
}

/// Copy main Configure isotopes + settings into the FM sandbox.
pub(crate) fn copy_config_to_fm(state: &mut AppState) {
    state.fm_isotope_entries = state
        .isotope_entries
        .iter()
        .map(|e| IsotopeEntry {
            z: e.z,
            a: e.a,
            symbol: e.symbol.clone(),
            initial_density: e.initial_density,
            resonance_data: e.resonance_data.clone(),
            enabled: e.enabled,
            endf_status: if e.resonance_data.is_some() {
                EndfStatus::Loaded
            } else {
                EndfStatus::Pending
            },
        })
        .collect();
    state.fm_endf_library = state.endf_library;
    state.fm_temperature_k = state.temperature_k;
    state.fm_resolution_enabled = state.resolution_enabled;
    state.fm_resolution_mode = state.resolution_mode.clone();
    state.fm_spectrum = None;
    state.fm_per_isotope_spectra.clear();
}

/// Push FM sandbox isotopes + settings back to main Configure.
pub(crate) fn push_fm_to_config(state: &mut AppState) {
    state.isotope_entries = state
        .fm_isotope_entries
        .iter()
        .map(|e| IsotopeEntry {
            z: e.z,
            a: e.a,
            symbol: e.symbol.clone(),
            initial_density: e.initial_density,
            resonance_data: e.resonance_data.clone(),
            enabled: e.enabled,
            endf_status: if e.resonance_data.is_some() {
                EndfStatus::Loaded
            } else {
                EndfStatus::Pending
            },
        })
        .collect();
    state.endf_library = state.fm_endf_library;
    state.temperature_k = state.fm_temperature_k;
    state.resolution_enabled = state.fm_resolution_enabled;
    state.resolution_mode = state.fm_resolution_mode.clone();
    state.spatial_result = None;
    state.pixel_fit_result = None;
    // Mark pipeline dirty so the Studio re-run button becomes active.
    state.mark_dirty(GuidedStep::Analyze);
}

pub(crate) fn fm_fetch_endf_data(state: &mut AppState) {
    use nereids_core::types::Isotope;
    use nereids_endf::retrieval;

    let mut work: Vec<(usize, Isotope, String, EndfLibrary)> = Vec::new();
    let mut failed_indices: Vec<usize> = Vec::new();
    for (i, entry) in state.fm_isotope_entries.iter().enumerate() {
        if entry.enabled && entry.endf_status == EndfStatus::Pending {
            let isotope = match Isotope::new(entry.z, entry.a) {
                Ok(iso) => iso,
                Err(e) => {
                    state.status_message = format!("Invalid isotope {}: {}", entry.symbol, e);
                    failed_indices.push(i);
                    continue;
                }
            };
            if retrieval::mat_number(&isotope).is_none() {
                state.status_message = format!(
                    "No MAT number for {} — isotope not in database",
                    entry.symbol
                );
                failed_indices.push(i);
                continue;
            }
            work.push((i, isotope, entry.symbol.clone(), state.fm_endf_library));
        }
    }
    for i in failed_indices {
        state.fm_isotope_entries[i].endf_status = EndfStatus::Failed;
    }

    if work.is_empty() {
        return;
    }

    // Mark entries as Fetching before spawning the background thread
    for (i, _, _, _) in &work {
        if let Some(entry) = state.fm_isotope_entries.get_mut(*i) {
            entry.endf_status = EndfStatus::Fetching;
        }
    }

    let (tx, rx) = mpsc::channel();
    state.pending_fm_endf = Some(rx);
    state.is_fetching_fm_endf = true;
    state.status_message = "Fetching ENDF data (FM)...".into();
    let cancel = Arc::clone(&state.cancel_token);

    std::thread::spawn(move || design::endf_fetch_worker(work, cancel, tx));
}
