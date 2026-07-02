//! Main application structure and egui App implementation.

use crate::guided;
use crate::state::{
    AppState, EndfStatus, FetchTarget, GuidedStep, ProvenanceEventKind, SaveDataMode, SessionCache,
    Tab, UiMode,
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

    fn save_session_cache(&self, storage: &mut dyn eframe::Storage) {
        if let Some(cache) = SessionCache::from_state(&self.state) {
            eframe::set_value(storage, SESSION_CACHE_KEY, &cache);
        } else {
            // Clear stale cache when no pipeline is configured
            storage.set_string(SESSION_CACHE_KEY, String::new());
        }
    }

    fn wait_for_background_save(&mut self) {
        // If a background save is in progress, block until it completes
        // to avoid corrupting the HDF5 file.
        if let Some(handle) = self.state.save_join_handle.take() {
            handle.join().ok();
        }
    }
}

impl eframe::App for NereidsApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.wait_for_background_save();

        // Flush pending log records before any abrupt exit. `process::exit`
        // skips stack unwinding, so the `WorkerGuard` stashed inside
        // `logging` would otherwise never run its Drop.
        crate::logging::shutdown();

        // macOS/AppKit can abort after eframe returns from `on_exit` while
        // tearing down winit's NSView touch-bar observer.
        #[cfg(target_os = "macos")]
        std::process::exit(0);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.save_session_cache(storage);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        #[cfg(not(target_os = "macos"))]
        let _ = frame;

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

        #[cfg(target_os = "macos")]
        if ctx.input(|i| i.viewport().close_requested()) {
            if let Some(storage) = frame.storage_mut() {
                self.save_session_cache(storage);
            }
            self.wait_for_background_save();
            // Flush log records before the abrupt exit (see on_exit).
            crate::logging::shutdown();
            std::process::exit(0);
        }

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

        // Native-dialog failure banner: the log bridge latches rfd
        // backend errors (which never fire on user-cancel), so a dead
        // file-picker environment is surfaced instead of silent (#526).
        // The in-app fallback tier only exists on Linux — the banner
        // mentions the switch only when the facade actually made it.
        if let Some(msg) = crate::logging::take_dialog_backend_failure() {
            let switched = self.state.file_dialogs.note_backend_failure();
            let suffix = if switched {
                " — switched to the built-in file browser."
            } else {
                ""
            };
            self.state.native_dialog_warning =
                Some(format!("Native file dialog failed: {msg}{suffix}"));
        }
        // Environment problems found by the facade's probe/canary.
        if let Some(msg) = self.state.file_dialogs.take_warning() {
            self.state.native_dialog_warning = Some(msg);
        }
        if let Some(msg) = self.state.native_dialog_warning.clone() {
            egui::TopBottomPanel::top("native_dialog_warning").show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("\u{26A0}").color(ui.visuals().warn_fg_color));
                    ui.label(msg);
                    if ui.small_button("Dismiss").clicked() {
                        self.state.native_dialog_warning = None;
                    }
                });
            });
        }

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

        // File dialogs: drive any retained dialog, then route completed
        // picks. Runs after all panels so results are applied exactly
        // once per frame regardless of which panel opened the dialog.
        self.state.file_dialogs.update(ctx);
        crate::file_dialog::dispatch_results(&mut self.state);
    }
}

/// Poll background task channels and apply results to state.
fn poll_pending_tasks(state: &mut AppState) {
    // Poll spatial map result
    if let Some(ref rx) = state.pending_spatial {
        match rx.try_recv() {
            Ok(Ok(result)) => {
                tracing::info!(
                    converged = result.n_converged,
                    total = result.n_total,
                    "spatial map completed"
                );
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
                tracing::error!(error = %err_msg, "spatial map failed");
                state.status_message = format!("Spatial map error: {err_msg}");
                state.is_fitting = false;
                state.fitting_progress = None;
                state.residuals_cache = None;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                tracing::error!("spatial map task disconnected without result");
                state.status_message = "Spatial map task failed".into();
                state.is_fitting = false;
                state.fitting_progress = None;
                state.residuals_cache = None;
                state.pending_spatial = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {} // Still running
        }
    }

    // Poll ENDF fetch results (streamed one per isotope, matched by (z, a))
    if let Some(ref rx) = state.pending_endf {
        let mut disconnected = false;
        // Drain all available results this frame
        loop {
            match rx.try_recv() {
                Ok(fetch) => {
                    // Match by (z, a) — stable even if isotope list was mutated.
                    // Check BOTH individual entries AND group members for every
                    // fetch result, because the same (z, a) can appear in both
                    // lists (e.g., W-182 individually AND as part of a W group).
                    if let Some(entry) = state
                        .isotope_entries
                        .iter_mut()
                        .find(|e| e.z == fetch.z && e.a == fetch.a)
                    {
                        match &fetch.result {
                            Ok(data) => {
                                entry.resonance_data = Some(data.clone());
                                entry.endf_status = EndfStatus::Loaded;
                                state.status_message = format!("Loaded {}", fetch.symbol);
                                state.spatial_result = None;
                                state.pixel_fit_result = None;
                            }
                            Err(msg) => {
                                // Per-item failures stay at `debug` so a 20-isotope
                                // batch with an expired auth token doesn't flood the
                                // log; the batch-completion arm emits an aggregate
                                // warn when the failed count is non-zero.
                                tracing::debug!(symbol = %fetch.symbol, error = %msg, "ENDF fetch failed");
                                entry.endf_status = EndfStatus::Failed;
                                state.status_message = msg.clone();
                            }
                        }
                    }
                    // Also check group members (z on group, a on member)
                    for group in &mut state.isotope_groups {
                        if group.z != fetch.z {
                            continue;
                        }
                        if let Some(member) = group.members.iter_mut().find(|m| m.a == fetch.a) {
                            match &fetch.result {
                                Ok(data) => {
                                    member.resonance_data = Some(data.clone());
                                    member.endf_status = EndfStatus::Loaded;
                                    state.status_message = format!("Loaded {}", fetch.symbol);
                                    state.spatial_result = None;
                                    state.pixel_fit_result = None;
                                }
                                Err(msg) => {
                                    tracing::debug!(symbol = %fetch.symbol, error = %msg, "ENDF group-member fetch failed");
                                    member.endf_status = EndfStatus::Failed;
                                    state.status_message = msg.clone();
                                }
                            }
                            break;
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
            let group_loaded_count: usize = state
                .isotope_groups
                .iter()
                .filter(|g| g.enabled)
                .map(|g| {
                    g.members
                        .iter()
                        .filter(|m| m.resonance_data.is_some())
                        .count()
                })
                .sum();
            let any_missing_iso = state
                .isotope_entries
                .iter()
                .any(|e| e.enabled && e.resonance_data.is_none());
            let any_missing_grp = state
                .isotope_groups
                .iter()
                .any(|g| g.enabled && g.members.iter().any(|m| m.resonance_data.is_none()));
            if !any_missing_iso && !any_missing_grp {
                state.status_message = "All ENDF data loaded".into();
            }
            let total = loaded_count + group_loaded_count;
            let failed_iso = state
                .isotope_entries
                .iter()
                .filter(|e| e.enabled && e.endf_status == EndfStatus::Failed)
                .count();
            let failed_grp: usize = state
                .isotope_groups
                .iter()
                .filter(|g| g.enabled)
                .map(|g| {
                    g.members
                        .iter()
                        .filter(|m| m.endf_status == EndfStatus::Failed)
                        .count()
                })
                .sum();
            let failed = failed_iso + failed_grp;
            if failed > 0 {
                tracing::warn!(
                    failed,
                    loaded = total,
                    "ENDF batch (configure) had failures"
                );
            }
            tracing::info!(loaded = total, "ENDF batch fetch finished (configure)");
            state.log_provenance(
                ProvenanceEventKind::ConfigChanged,
                format!("Fetched ENDF data for {total} isotopes"),
            );
            state.is_fetching_endf = false;
            state.pending_endf = None;
        }
    }

    // Poll Forward Model ENDF fetch results (matched by (z, a))
    if let Some(ref rx) = state.pending_fm_endf {
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(fetch) => {
                    if let Some(entry) = state
                        .fm_isotope_entries
                        .iter_mut()
                        .find(|e| e.z == fetch.z && e.a == fetch.a)
                    {
                        match fetch.result {
                            Ok(data) => {
                                entry.resonance_data = Some(data);
                                entry.endf_status = EndfStatus::Loaded;
                                state.status_message = format!("FM: loaded {}", fetch.symbol);
                                state.fm_spectrum = None;
                                state.fm_per_isotope_spectra.clear();
                            }
                            Err(msg) => {
                                tracing::debug!(symbol = %fetch.symbol, error = %msg, "FM ENDF fetch failed");
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
            let failed_fm = state
                .fm_isotope_entries
                .iter()
                .filter(|e| e.enabled && e.endf_status == EndfStatus::Failed)
                .count();
            if failed_fm > 0 {
                tracing::warn!(
                    failed = failed_fm,
                    "ENDF batch (forward model) had failures"
                );
            }
            tracing::info!("ENDF batch fetch finished (forward model)");
            state.is_fetching_fm_endf = false;
            state.pending_fm_endf = None;
        }
    }

    // Poll Detectability ENDF fetch results (matched by (z, a) + FetchTarget)
    if let Some(ref rx) = state.pending_detect_endf {
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(fetch) => match fetch.target {
                    FetchTarget::DetectMatrix => {
                        if let Some(entry) = state
                            .detect_matrix_entries
                            .iter_mut()
                            .find(|e| e.z == fetch.z && e.a == fetch.a)
                        {
                            match fetch.result {
                                Ok(data) => {
                                    entry.resonance_data = Some(data);
                                    entry.endf_status = EndfStatus::Loaded;
                                    state.status_message =
                                        format!("Detect: loaded matrix {}", fetch.symbol);
                                }
                                Err(msg) => {
                                    tracing::debug!(symbol = %fetch.symbol, error = %msg, "Detect matrix ENDF fetch failed");
                                    entry.endf_status = EndfStatus::Failed;
                                    state.status_message = msg;
                                }
                            }
                        }
                    }
                    FetchTarget::DetectTrace => {
                        if let Some(entry) = state
                            .detect_trace_entries
                            .iter_mut()
                            .find(|e| e.z == fetch.z && e.a == fetch.a)
                        {
                            match fetch.result {
                                Ok(data) => {
                                    entry.resonance_data = Some(data);
                                    entry.endf_status = EndfStatus::Loaded;
                                    state.status_message =
                                        format!("Detect: loaded trace {}", fetch.symbol);
                                }
                                Err(msg) => {
                                    tracing::debug!(symbol = %fetch.symbol, error = %msg, "Detect trace ENDF fetch failed");
                                    entry.endf_status = EndfStatus::Failed;
                                    state.status_message = msg;
                                }
                            }
                        }
                    }
                    // Configure/ForwardModel results never appear on this channel
                    _ => {}
                },
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
            let failed_matrix = total_matrix.saturating_sub(loaded_matrix);
            let failed_traces = total_traces.saturating_sub(loaded_traces);
            if failed_matrix > 0 || failed_traces > 0 {
                tracing::warn!(
                    failed_matrix,
                    failed_traces,
                    "ENDF batch (detectability) had failures"
                );
            }
            tracing::info!(
                matrix_loaded = loaded_matrix,
                matrix_total = total_matrix,
                trace_loaded = loaded_traces,
                trace_total = total_traces,
                "ENDF batch fetch finished (detectability)"
            );
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
                tracing::info!(path = %path.display(), mode = %mode_label, "project saved");
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
                tracing::error!(error = %msg, "project save failed");
                state.status_message = format!("Save failed: {msg}");
                state.is_saving = false;
                state.pending_save = None;
                state.save_join_handle = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                tracing::error!("save task disconnected without result");
                state.status_message = "Save task failed unexpectedly".into();
                state.is_saving = false;
                state.pending_save = None;
                state.save_join_handle = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {} // Still saving
        }
    }
}
