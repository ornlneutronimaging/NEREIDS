//! Main application structure and egui App implementation.

use crate::guided;
use crate::state::{AppState, EndfStatus, ProvenanceEventKind, SessionCache, Tab, UiMode};
use crate::studio;
use crate::theme;
use crate::widgets;

/// NEREIDS desktop application.
pub struct NereidsApp {
    pub state: AppState,
}

const SESSION_CACHE_KEY: &str = "nereids_session_cache";

impl NereidsApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut state = AppState::default();

        // Restore cached session from previous run (if any)
        if let Some(storage) = cc.storage
            && let Some(cache) = eframe::get_value::<SessionCache>(storage, SESSION_CACHE_KEY)
        {
            state.cached_session = Some(cache);
        }

        Self { state }
    }
}

impl eframe::App for NereidsApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Some(cache) = SessionCache::from_state(&self.state) {
            eframe::set_value(storage, SESSION_CACHE_KEY, &cache);
        } else {
            // Clear stale cache when no pipeline is configured
            storage.set_string(SESSION_CACHE_KEY, String::new());
        }
    }

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
                state.log_provenance(
                    ProvenanceEventKind::AnalysisRun,
                    format!(
                        "Spatial mapping: {}/{} converged",
                        result.n_converged, result.n_total
                    ),
                );
                state.init_tile_display(result.density_maps.len());
                state.spatial_result = Some(result);
                state.is_fitting = false;
                state.fitting_progress = None;
                state.fitting_progress_counter = None;
                state.active_tab = Tab::Map;
                state.pending_spatial = None;
            }
            Ok(Err(err_msg)) => {
                state.status_message = format!("Spatial map error: {err_msg}");
                state.is_fitting = false;
                state.fitting_progress = None;
                state.fitting_progress_counter = None;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                state.status_message = "Spatial map task failed".into();
                state.is_fitting = false;
                state.fitting_progress = None;
                state.fitting_progress_counter = None;
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
                                entry.endf_status = EndfStatus::Loaded;
                                state.status_message = format!("Loaded {}", fetch.symbol);
                                // Invalidate stale results — isotope data changed.
                                state.spatial_result = None;
                                state.pixel_fit_result = None;
                            }
                            Err(msg) => {
                                entry.endf_status = EndfStatus::Failed;
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
            let loaded_count = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.resonance_data.is_some())
                .count();
            if !state
                .isotope_entries
                .iter()
                .any(|e| e.enabled && e.resonance_data.is_none())
            {
                state.status_message = "All ENDF data loaded".into();
            }
            state.log_provenance(
                ProvenanceEventKind::ConfigChanged,
                format!("Fetched ENDF data for {loaded_count} isotopes"),
            );
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
                                entry.endf_status = EndfStatus::Loaded;
                                state.status_message = format!("FM: loaded {}", fetch.symbol);
                                state.fm_spectrum = None;
                                state.fm_per_isotope_spectra.clear();
                            }
                            Err(msg) => {
                                entry.endf_status = EndfStatus::Failed;
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
    // Index convention: 0..N = matrix entries, N.. = trace entries at (index - N)
    // where N = state.detect_n_matrix_at_fetch (captured at fetch spawn time).
    if let Some(ref rx) = state.pending_detect_endf {
        let mut disconnected = false;
        let n_matrix = state.detect_n_matrix_at_fetch;
        loop {
            match rx.try_recv() {
                Ok(fetch) => {
                    if fetch.index < n_matrix {
                        // Matrix entry
                        if let Some(entry) = state.detect_matrix_entries.get_mut(fetch.index) {
                            match fetch.result {
                                Ok(data) => {
                                    entry.resonance_data = Some(data);
                                    entry.endf_status = EndfStatus::Loaded;
                                    state.status_message =
                                        format!("Detect: loaded matrix {}", fetch.symbol);
                                }
                                Err(msg) => {
                                    entry.endf_status = EndfStatus::Failed;
                                    state.status_message = msg;
                                }
                            }
                        }
                    } else {
                        // Trace entry at (index - N)
                        let trace_idx = fetch.index - n_matrix;
                        if let Some(entry) = state.detect_trace_entries.get_mut(trace_idx) {
                            match fetch.result {
                                Ok(data) => {
                                    entry.resonance_data = Some(data);
                                    entry.endf_status = EndfStatus::Loaded;
                                    state.status_message =
                                        format!("Detect: loaded trace {}", fetch.symbol);
                                }
                                Err(msg) => {
                                    entry.endf_status = EndfStatus::Failed;
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
            let total_traces = state.detect_trace_entries.len();
            let loaded_traces = state
                .detect_trace_entries
                .iter()
                .filter(|t| t.resonance_data.is_some())
                .count();
            let total_matrix = state.detect_matrix_entries.len();
            let loaded_matrix = state
                .detect_matrix_entries
                .iter()
                .filter(|m| m.resonance_data.is_some())
                .count();

            if loaded_matrix == total_matrix && loaded_traces == total_traces {
                state.status_message = "Detect: all ENDF data loaded".into();
            } else {
                let mut parts = Vec::new();
                if loaded_matrix < total_matrix {
                    let unsupported = total_matrix - loaded_matrix;
                    parts.push(format!(
                        "{} of {} matrix isotopes loaded ({} not supported)",
                        loaded_matrix, total_matrix, unsupported
                    ));
                }
                if loaded_traces < total_traces {
                    let unsupported = total_traces - loaded_traces;
                    parts.push(format!(
                        "{} of {} trace isotopes loaded ({} not supported)",
                        loaded_traces, total_traces, unsupported
                    ));
                }
                if !parts.is_empty() {
                    state.status_message = format!("Detect: {}", parts.join("; "));
                }
            }
            state.is_fetching_detect_endf = false;
            state.pending_detect_endf = None;
        }
    }
}
