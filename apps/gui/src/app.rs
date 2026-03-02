//! Main application structure and egui App implementation.

use crate::guided;
use crate::state::{AppState, Tab, UiMode};
use crate::studio;
use crate::theme;
use crate::widgets;

/// NEREIDS desktop application.
pub struct NereidsApp {
    pub state: AppState,
}

impl NereidsApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: AppState::default(),
        }
    }
}

impl eframe::App for NereidsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply theme
        theme::apply_theme(ctx, self.state.theme_preference);

        // Poll background tasks
        poll_pending_tasks(&mut self.state);

        // Keep repainting while background work is in progress
        if self.state.is_fitting
            || self.state.is_fetching_endf
            || self.state.is_fetching_fm_endf
            || self.state.is_fetching_detect_endf
        {
            ctx.request_repaint();
        }

        // Top toolbar
        widgets::toolbar::toolbar(ctx, &mut self.state);

        // Bottom status bar
        widgets::statusbar::status_bar(ctx, &self.state);

        // Main content area
        match self.state.ui_mode {
            UiMode::Guided => {
                guided::sidebar::guided_sidebar(ctx, &mut self.state);
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        guided::guided_content(ui, &mut self.state);
                    });
                });
            }
            UiMode::Studio => {
                studio::studio_content(ctx, &mut self.state);
            }
        }

        // Periodic table modal overlay
        crate::widgets::periodic_table::periodic_table_modal(ctx, &mut self.state);
    }
}

/// Poll background task channels and apply results to state.
fn poll_pending_tasks(state: &mut AppState) {
    // Poll spatial map result
    if let Some(ref rx) = state.pending_spatial {
        match rx.try_recv() {
            Ok(Ok(result)) => {
                state.status_message = format!(
                    "Spatial map: {}/{} converged",
                    result.n_converged, result.n_total
                );
                state.spatial_result = Some(result);
                state.is_fitting = false;
                state.active_tab = Tab::Map;
                state.pending_spatial = None;
            }
            Ok(Err(err_msg)) => {
                state.status_message = format!("Spatial map error: {err_msg}");
                state.is_fitting = false;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                state.status_message = "Spatial map task failed".into();
                state.is_fitting = false;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {} // Still running
        }
    }

    // Poll ENDF fetch results (streamed one per isotope)
    if let Some(ref rx) = state.pending_endf {
        let mut disconnected = false;
        // Drain all available results this frame
        loop {
            match rx.try_recv() {
                Ok(fetch) => {
                    if let Some(entry) = state.isotope_entries.get_mut(fetch.index) {
                        match fetch.result {
                            Ok(data) => {
                                entry.resonance_data = Some(data);
                                state.status_message = format!("Loaded {}", fetch.symbol);
                                // Invalidate stale results — isotope data changed.
                                state.spatial_result = None;
                                state.pixel_fit_result = None;
                            }
                            Err(msg) => {
                                state.status_message = msg;
                            }
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        // Only finalize when the sender is dropped (thread finished)
        if disconnected {
            if !state
                .isotope_entries
                .iter()
                .any(|e| e.enabled && e.resonance_data.is_none())
            {
                state.status_message = "All ENDF data loaded".into();
            }
            state.is_fetching_endf = false;
            state.pending_endf = None;
        }
    }

    // Poll Forward Model ENDF fetch results
    if let Some(ref rx) = state.pending_fm_endf {
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(fetch) => {
                    if let Some(entry) = state.fm_isotope_entries.get_mut(fetch.index) {
                        match fetch.result {
                            Ok(data) => {
                                entry.resonance_data = Some(data);
                                state.status_message = format!("FM: loaded {}", fetch.symbol);
                                state.fm_spectrum = None;
                                state.fm_per_isotope_spectra.clear();
                            }
                            Err(msg) => {
                                state.status_message = msg;
                            }
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            if !state
                .fm_isotope_entries
                .iter()
                .any(|e| e.enabled && e.resonance_data.is_none())
            {
                state.status_message = "FM: all ENDF data loaded".into();
            }
            state.is_fetching_fm_endf = false;
            state.pending_fm_endf = None;
        }
    }

    // Poll Detectability ENDF fetch results
    // Index convention: 0 = matrix isotope, 1+ = trace entries at position (index - 1)
    if let Some(ref rx) = state.pending_detect_endf {
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(fetch) => {
                    if fetch.index == 0 {
                        // Matrix isotope
                        if let Some(ref mut matrix) = state.detect_matrix {
                            match fetch.result {
                                Ok(data) => {
                                    matrix.resonance_data = Some(data);
                                    state.status_message =
                                        format!("Detect: loaded matrix {}", fetch.symbol);
                                }
                                Err(msg) => {
                                    state.status_message = msg;
                                }
                            }
                        }
                    } else {
                        // Trace isotope at position (index - 1)
                        let trace_idx = fetch.index - 1;
                        if let Some(entry) = state.detect_trace_entries.get_mut(trace_idx) {
                            match fetch.result {
                                Ok(data) => {
                                    entry.resonance_data = Some(data);
                                    state.status_message =
                                        format!("Detect: loaded trace {}", fetch.symbol);
                                }
                                Err(msg) => {
                                    state.status_message = msg;
                                }
                            }
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            // Count how many trace isotopes have data vs total
            let total_traces = state.detect_trace_entries.len();
            let loaded_traces = state
                .detect_trace_entries
                .iter()
                .filter(|t| t.resonance_data.is_some())
                .count();
            let matrix_loaded = state
                .detect_matrix
                .as_ref()
                .is_some_and(|m| m.resonance_data.is_some());

            if matrix_loaded && loaded_traces == total_traces {
                state.status_message = "Detect: all ENDF data loaded".into();
            } else {
                let mut parts = Vec::new();
                if !matrix_loaded && let Some(ref m) = state.detect_matrix {
                    parts.push(format!("matrix {} not supported", m.symbol));
                }
                if loaded_traces < total_traces {
                    let unsupported = total_traces - loaded_traces;
                    parts.push(format!(
                        "ENDF data loaded for {} of {} trace isotopes ({} not supported)",
                        loaded_traces, total_traces, unsupported
                    ));
                } else {
                    parts.push(format!(
                        "ENDF data loaded for all {} trace isotopes",
                        total_traces
                    ));
                }
                state.status_message = format!("Detect: {}", parts.join("; "));
            }
            state.is_fetching_detect_endf = false;
            state.pending_detect_endf = None;
        }
    }
}
