//! Main application structure and egui App implementation.

use crate::guided;
use crate::state::{
    AppState, EndfStatus, GuidedStep, ProvenanceEventKind, SaveDataMode, SessionCache, Tab, UiMode,
};
use crate::studio;
use crate::theme;
use crate::widgets;

/// NEREIDS desktop application.
pub struct NereidsApp {
    pub state: AppState,
    pub memory: crate::telemetry::MemoryTelemetry,
}

const SESSION_CACHE_KEY: &str = "nereids_session_cache";

impl NereidsApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Load DejaVuSans as a fallback font for Unicode symbols (arrows, math, etc.)
        // that egui's built-in font doesn't cover.
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "dejavu".to_owned(),
            std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
                "../assets/DejaVuSans.ttf"
            ))),
        );
        // Append as fallback to Proportional — egui tries fonts in order,
        // so the default font renders most glyphs and DejaVu fills the gaps.
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .push("dejavu".to_owned());
        cc.egui_ctx.set_fonts(fonts);

        let mut state = AppState::default();

        // Restore cached session from previous run (if any)
        if let Some(storage) = cc.storage
            && let Some(cache) = eframe::get_value::<SessionCache>(storage, SESSION_CACHE_KEY)
        {
            state.cached_session = Some(cache);
        }

        Self {
            state,
            memory: crate::telemetry::MemoryTelemetry::new(),
        }
    }
}

impl eframe::App for NereidsApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // If a background save is in progress, block until it completes
        // to avoid corrupting the HDF5 file.
        if let Some(handle) = self.state.save_join_handle.take() {
            handle.join().ok();
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Some(cache) = SessionCache::from_state(&self.state) {
            eframe::set_value(storage, SESSION_CACHE_KEY, &cache);
        } else {
            // Clear stale cache when no pipeline is configured
            storage.set_string(SESSION_CACHE_KEY, String::new());
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply theme (skip if unchanged to avoid 80+ color assignments per frame)
        let resolved = theme::resolve_dark_mode(ctx, self.state.theme_preference);
        if self.state.last_applied_dark_mode != Some(resolved) {
            theme::apply_theme(ctx, self.state.theme_preference);
            self.state.last_applied_dark_mode = Some(resolved);
        }

        // Store context so background threads can request repaints.
        self.state.egui_ctx = Some(ctx.clone());

        // Poll background tasks
        poll_pending_tasks(&mut self.state);

        // Refresh memory telemetry (750ms interval)
        self.memory.refresh(ctx.input(|i| i.time));

        // Keep repainting while background work is in progress.
        // Fitting also has a dedicated watcher thread that pokes via
        // ctx.request_repaint(), but we keep the timer as a fallback.
        if self.state.is_fitting
            || self.state.is_fetching_endf
            || self.state.is_fetching_fm_endf
            || self.state.is_fetching_detect_endf
            || self.state.is_saving
        {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Cmd+S / Ctrl+S — save project
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S)) {
            let has_results = self.state.spatial_result.is_some();
            if has_results && !self.state.is_saving {
                crate::project::save_project_quick(&mut self.state);
            }
        }

        // Cmd+O / Ctrl+O — open project
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::O)) {
            crate::project::load_project_dialog(&mut self.state);
        }

        // Top toolbar
        widgets::toolbar::toolbar(ctx, &mut self.state);

        // Bottom status bar
        widgets::statusbar::status_bar(ctx, &self.state, self.memory.rss_bytes);

        // Main content area
        match self.state.ui_mode {
            UiMode::Guided => {
                guided::sidebar::guided_sidebar(ctx, &mut self.state);
                guided::sidebar::history_window(ctx, &mut self.state);
                egui::CentralPanel::default().show(ctx, |ui| {
                    // Analyze needs the real viewport height so its image
                    // column can fill vertically (ScrollArea makes
                    // available_height() return infinity). All other steps
                    // use ScrollArea for overflowing content.
                    if self.state.guided_step == GuidedStep::Analyze {
                        guided::guided_content(ui, &mut self.state);
                    } else {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            guided::guided_content(ui, &mut self.state);
                        });
                    }
                });
            }
            UiMode::Studio => {
                studio::studio_content(ctx, &mut self.state);
            }
        }

        // Save-mode chooser modal
        crate::project::save_modal(ctx, &mut self.state);

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
                state.residuals_cache = None;
                state.active_tab = Tab::Map;
                state.pending_spatial = None;
                // Pipeline re-run completed successfully — clear dirty state.
                state.clear_dirty();
            }
            Ok(Err(err_msg)) => {
                state.status_message = format!("Spatial map error: {err_msg}");
                state.is_fitting = false;
                state.fitting_progress = None;
                state.residuals_cache = None;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                state.status_message = "Spatial map task failed".into();
                state.is_fitting = false;
                state.fitting_progress = None;
                state.residuals_cache = None;
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

    // Poll project save result
    if let Some(ref rx) = state.pending_save {
        match rx.try_recv() {
            Ok(Ok((path, mode))) => {
                let mode_label = match mode {
                    SaveDataMode::Linked => "linked",
                    SaveDataMode::Embedded => "embedded",
                };
                state.project_file_path = Some(path.clone());
                state.last_save_mode = mode;
                state.status_message =
                    format!("Project saved ({mode_label}) to {}", path.display());
                state.log_provenance(
                    ProvenanceEventKind::ProjectSaved,
                    format!("Saved ({mode_label}) to {}", path.display()),
                );
                state.is_saving = false;
                state.pending_save = None;
                state.save_join_handle = None;
            }
            Ok(Err(msg)) => {
                state.status_message = format!("Save failed: {msg}");
                state.is_saving = false;
                state.pending_save = None;
                state.save_join_handle = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                state.status_message = "Save task failed unexpectedly".into();
                state.is_saving = false;
                state.pending_save = None;
                state.save_join_handle = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {} // Still saving
        }
    }
}
