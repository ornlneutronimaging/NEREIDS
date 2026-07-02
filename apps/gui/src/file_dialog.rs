//! Cross-platform file-dialog facade.
//!
//! Every file/folder picker in the GUI goes through [`FileDialogs`]: a
//! call site opens a dialog tagged with a [`DialogIntent`], and the
//! completed pick is routed by [`dispatch_results`] at the end of
//! `NereidsApp::update`. This decouples "the user clicked a picker
//! button" (UI code, often deep inside a panel with partial borrows of
//! `AppState`) from "a path was chosen" (state mutation, always with
//! full `&mut AppState`), so the dialog backend is free to resolve
//! immediately or on a later frame without the call sites caring.
//!
//! Backend tiers:
//! - **macOS/Windows** — native rfd dialog, blocking inside `open()`;
//!   the result is dispatched the same frame (identical behaviour to
//!   the previous direct `rfd::FileDialog` calls).
//! - **Linux, native tier** — rfd's xdg-portal backend (with rfd's own
//!   zenity CLI fallback) run on a **detached worker thread** and
//!   polled each frame. rfd 0.17's portal wait loop has no timeout
//!   (`wait_for_response` blocks forever if the portal accepts the
//!   request but its backend never responds), so the sync call must
//!   never run on the UI thread — the worker converts "app frozen"
//!   into "dialog didn't appear", which the escape-hatch overlay and
//!   the log-bridge latch (see `logging.rs`) then surface.
//! - **Linux, in-app tier** — pure-egui `egui-file-dialog`, used when
//!   the startup probe / portal canary finds no working native chain,
//!   when an rfd backend failure is latched at runtime, or when the
//!   user clicks the escape hatch. Works in every environment
//!   (containers, root, no D-Bus, `ssh -X`) with zero system
//!   dependencies — the environments of issue #526.

use std::path::PathBuf;

use crate::state::AppState;

/// Which UI feature requested the dialog — routes the picked path in
/// [`dispatch_results`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DialogIntent {
    /// Open a `.nrd.h5` project (toolbar, Ctrl/Cmd+O).
    OpenProject,
    /// "Save As" target for the current project (save modal).
    SaveProjectAs,
    /// Export directory for spatial-map results (Studio dock).
    ExportDirectory,
    /// Save one result tile as a colormapped PNG (tile toolbelt).
    SaveTilePng { tile_idx: usize, label: String },
    /// Install a local ENDF file into the cache (Configure step, #523).
    InstallLocalEndf,
    /// Sample TIFF stack — file or folder (Load step). Also used for the
    /// pre-normalized transmission stack, which lands in the same
    /// `sample_path` field with the same invalidation set.
    TiffSample,
    /// Open-beam TIFF stack — file or folder (Load step).
    TiffOpenBeam,
    /// TOF spectrum file (Load step).
    SpectrumFile,
    /// Sample NeXus/HDF5 file (Load step, histogram + event tabs).
    Hdf5Sample,
    /// Optional open-beam NeXus/HDF5 file (Load step).
    Hdf5OpenBeam,
    /// Tabulated instrument-resolution file for one of the three
    /// resolution cards.
    ResolutionFile(ResolutionTarget),
}

/// Which of the three resolution cards asked for a tabulated file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolutionTarget {
    Configure,
    ForwardModel,
    Detectability,
}

/// Backend-neutral dialog options (mirror of the rfd builder calls in use).
#[derive(Clone, Default)]
pub struct DialogOptions {
    pub title: Option<&'static str>,
    /// `(name, extensions)`, e.g. `("TIFF", &["tif", "tiff"])`.
    pub filters: Vec<(&'static str, &'static [&'static str])>,
    /// Pre-filled file name (save dialogs).
    pub file_name: Option<String>,
    /// Initial directory. `None` = backend default (native dialogs
    /// remember their last location themselves; the in-app tier falls
    /// back to the facade-tracked last-used directory).
    pub directory: Option<PathBuf>,
}

#[derive(Clone, Copy)]
enum Mode {
    PickFile,
    PickFolder,
    SaveFile,
}

/// Run a blocking rfd dialog. On macOS/Windows this is called directly
/// on the UI thread (native modal behaviour); on Linux it runs on a
/// worker thread because the portal backend can block indefinitely.
fn run_rfd_blocking(mode: Mode, opts: &DialogOptions) -> Option<PathBuf> {
    let mut dlg = rfd::FileDialog::new();
    if let Some(t) = opts.title {
        dlg = dlg.set_title(t);
    }
    for (name, exts) in &opts.filters {
        dlg = dlg.add_filter(*name, exts);
    }
    if let Some(d) = &opts.directory {
        dlg = dlg.set_directory(d);
    }
    if let Some(n) = &opts.file_name {
        dlg = dlg.set_file_name(n);
    }
    match mode {
        Mode::PickFile => dlg.pick_file(),
        Mode::PickFolder => dlg.pick_folder(),
        Mode::SaveFile => dlg.save_file(),
    }
}

/// A native (portal/zenity) dialog request in flight on a worker thread.
#[cfg(target_os = "linux")]
struct NativeRequest {
    mode: Mode,
    intent: DialogIntent,
    /// Kept for reopening in-app if the native chain fails or the user
    /// clicks the escape hatch.
    opts: DialogOptions,
    rx: std::sync::mpsc::Receiver<Option<PathBuf>>,
    started: std::time::Instant,
}

/// Poll-based dialog service stored in [`AppState`].
#[derive(Default)]
pub struct FileDialogs {
    /// Completed pick awaiting dispatch.
    pending: Option<(DialogIntent, PathBuf)>,
    /// Environment problem discovered by the probe/canary — surfaced by
    /// the app as a banner (consume-once).
    warning: Option<String>,
    /// Directory of the last pick; initial directory for the in-app
    /// tier when the request has no explicit one (session-scoped parity
    /// with native dialogs remembering their last location).
    last_dir: Option<PathBuf>,
    #[cfg(target_os = "linux")]
    linux: LinuxTiers,
}

/// Linux-only backend state: tier decision + the two dialog mechanisms.
#[cfg(target_os = "linux")]
#[derive(Default)]
struct LinuxTiers {
    /// `None` until the first `update()` runs the probe.
    tier: Option<LinuxTier>,
    /// Cloned egui context so worker threads can request a repaint.
    ctx: Option<egui::Context>,
    /// Native request in flight (worker thread).
    active_native: Option<NativeRequest>,
    /// Retained in-app dialog in flight.
    active_in_app: Option<(DialogIntent, egui_file_dialog::FileDialog)>,
    /// Portal-health canary result channel (worker thread).
    canary_rx: Option<std::sync::mpsc::Receiver<Result<(), String>>>,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum LinuxTier {
    /// rfd portal backend (+ its zenity fallback) on a worker thread.
    Native,
    /// Pure-egui in-app browser.
    InApp,
}

impl FileDialogs {
    pub fn pick_file(&mut self, intent: DialogIntent, opts: DialogOptions) {
        self.open(Mode::PickFile, intent, opts);
    }

    pub fn pick_folder(&mut self, intent: DialogIntent, opts: DialogOptions) {
        self.open(Mode::PickFolder, intent, opts);
    }

    pub fn save_file(&mut self, intent: DialogIntent, opts: DialogOptions) {
        self.open(Mode::SaveFile, intent, opts);
    }

    fn open(&mut self, mode: Mode, intent: DialogIntent, opts: DialogOptions) {
        // A new request supersedes any unconsumed earlier result.
        self.pending = None;

        #[cfg(not(target_os = "linux"))]
        {
            if let Some(path) = run_rfd_blocking(mode, &opts) {
                self.remember_dir(&path);
                self.pending = Some((intent, path));
            }
        }

        #[cfg(target_os = "linux")]
        {
            // Dropping a previous in-flight request also drops its
            // receiver, so a late result from an orphaned worker is
            // discarded instead of resolving the wrong intent.
            self.linux.active_native = None;
            self.linux.active_in_app = None;
            match self.linux.tier.unwrap_or(LinuxTier::Native) {
                LinuxTier::Native => self.open_native_worker(mode, intent, opts),
                LinuxTier::InApp => self.open_in_app(mode, intent, opts),
            }
        }
    }

    /// Per-frame driver, called once from `NereidsApp::update`: decides
    /// the Linux tier on first run, polls the canary and any in-flight
    /// native request, drives the retained in-app dialog, and renders
    /// the escape-hatch overlay. No-op on macOS/Windows (their dialogs
    /// resolve inside `open()`).
    pub fn update(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_os = "linux"))]
        let _ = ctx;

        #[cfg(target_os = "linux")]
        {
            self.linux.ctx = Some(ctx.clone());
            self.decide_tier_once();
            self.poll_canary();
            self.poll_native(ctx);
            self.drive_in_app(ctx);
        }
    }

    /// Take the completed pick, if any (consume-once).
    pub fn take_any(&mut self) -> Option<(DialogIntent, PathBuf)> {
        self.pending.take()
    }

    /// Take the probe/canary warning, if any (consume-once) — shown by
    /// the app in the native-dialog warning banner.
    pub fn take_warning(&mut self) -> Option<String> {
        self.warning.take()
    }

    /// A native-dialog backend failure was latched by the log bridge:
    /// downgrade to the in-app tier and, if a native request is still
    /// in flight, reopen the same request in-app so the user's click
    /// still lands in a working dialog.
    pub fn note_backend_failure(&mut self) {
        #[cfg(target_os = "linux")]
        {
            self.linux.tier = Some(LinuxTier::InApp);
            if let Some(req) = self.linux.active_native.take() {
                self.open_in_app(req.mode, req.intent, req.opts);
            }
        }
    }

    fn remember_dir(&mut self, path: &std::path::Path) {
        self.last_dir = if path.is_dir() {
            Some(path.to_path_buf())
        } else {
            path.parent().map(std::path::Path::to_path_buf)
        };
    }

    /// Test-only: inject a completed pick as if a dialog resolved.
    #[cfg(test)]
    pub fn inject(&mut self, intent: DialogIntent, path: PathBuf) {
        self.pending = Some((intent, path));
    }
}

// ── Linux tiers ────────────────────────────────────────────────
#[cfg(target_os = "linux")]
impl FileDialogs {
    /// First-frame tier decision from cheap env/filesystem checks (no
    /// D-Bus traffic, cannot block), plus an async portal canary when
    /// the native chain looks viable.
    fn decide_tier_once(&mut self) {
        if self.linux.tier.is_some() {
            return;
        }
        match probe_native_prerequisites() {
            Ok(()) => {
                self.linux.tier = Some(LinuxTier::Native);
                self.linux.canary_rx = Some(spawn_portal_canary());
            }
            Err(msg) => {
                self.linux.tier = Some(LinuxTier::InApp);
                self.warning = Some(msg);
            }
        }
    }

    /// Portal canary result: reading the FileChooser interface version
    /// succeeds only when a real portal backend implements it, so a
    /// failure means the exact portal-accepts-then-hangs environment —
    /// downgrade before any dialog can get stuck.
    fn poll_canary(&mut self) {
        let Some(rx) = &self.linux.canary_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(())) => {
                self.linux.canary_rx = None;
            }
            Ok(Err(msg)) => {
                self.linux.canary_rx = None;
                // Zenity alone still makes the native chain viable —
                // rfd falls back to it without touching the portal.
                if which_on_path("zenity") {
                    tracing::info!(
                        reason = %msg,
                        "portal canary failed; keeping native dialogs via zenity fallback"
                    );
                } else {
                    self.linux.tier = Some(LinuxTier::InApp);
                    self.warning = Some(format!(
                        "Native file dialogs disabled: {msg}. Using the built-in \
                         file browser. Install zenity or run inside a desktop \
                         session for native dialogs."
                    ));
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.linux.canary_rx = None;
            }
        }
    }

    fn open_native_worker(&mut self, mode: Mode, intent: DialogIntent, opts: DialogOptions) {
        let (tx, rx) = std::sync::mpsc::channel();
        let worker_opts = opts.clone();
        let ctx = self.linux.ctx.clone();
        std::thread::spawn(move || {
            let picked = run_rfd_blocking(mode, &worker_opts);
            let _ = tx.send(picked);
            if let Some(ctx) = ctx {
                ctx.request_repaint();
            }
        });
        self.linux.active_native = Some(NativeRequest {
            mode,
            intent,
            opts,
            rx,
            started: std::time::Instant::now(),
        });
    }

    /// Poll the worker; if the native dialog is taking suspiciously
    /// long to resolve, offer the built-in browser as an escape hatch
    /// (a hung portal never delivers a result, and rfd gives us no way
    /// to cancel it — the orphaned worker thread is the accepted cost).
    fn poll_native(&mut self, ctx: &egui::Context) {
        use std::sync::mpsc::TryRecvError;

        let mut escape = false;
        if let Some(req) = &self.linux.active_native {
            match req.rx.try_recv() {
                Ok(Some(path)) => {
                    let intent = req.intent.clone();
                    self.linux.active_native = None;
                    self.remember_dir(&path);
                    self.pending = Some((intent, path));
                }
                Ok(None) => {
                    // User cancel, or backend failure — the log-bridge
                    // latch (app banner + note_backend_failure) is the
                    // failure discriminator; nothing to do here.
                    self.linux.active_native = None;
                }
                Err(TryRecvError::Disconnected) => {
                    self.linux.active_native = None;
                }
                Err(TryRecvError::Empty) => {
                    // Still open. After a short grace period, offer the
                    // in-app fallback without killing the native dialog
                    // (we cannot tell "hung portal" from "user is
                    // browsing a big directory").
                    let elapsed = req.started.elapsed();
                    if elapsed >= std::time::Duration::from_secs(1) {
                        egui::Window::new("native_dialog_pending")
                            .title_bar(false)
                            .resizable(false)
                            .anchor(egui::Align2::CENTER_TOP, [0.0, 8.0])
                            .show(ctx, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spinner();
                                    ui.label("Waiting for the system file dialog\u{2026}");
                                    if ui.small_button("Use built-in browser").clicked() {
                                        escape = true;
                                    }
                                });
                            });
                    } else {
                        // Wake up in time to show the overlay.
                        ctx.request_repaint_after(std::time::Duration::from_secs(1) - elapsed);
                    }
                }
            }
        }

        if escape && let Some(req) = self.linux.active_native.take() {
            self.linux.tier = Some(LinuxTier::InApp);
            self.open_in_app(req.mode, req.intent, req.opts);
        }
    }

    fn open_in_app(&mut self, mode: Mode, intent: DialogIntent, opts: DialogOptions) {
        use egui_file_dialog::FileDialog;

        let mut dlg = FileDialog::new()
            .as_modal(true) // blocking-parity: shield app state while open
            .allow_file_overwrite(true); // built-in overwrite-confirm modal
        if let Some(t) = opts.title {
            dlg = dlg.title(t);
        }
        match mode {
            Mode::SaveFile => {
                // Save dialogs use an extension dropdown instead of filters.
                for (name, exts) in &opts.filters {
                    if let Some(ext) = exts.first() {
                        dlg = dlg.add_save_extension(name, ext);
                    }
                }
                if let Some((name, _)) = opts.filters.first() {
                    dlg = dlg.default_save_extension(name);
                }
            }
            _ => {
                for (name, exts) in &opts.filters {
                    dlg = dlg.add_file_filter_extensions(name, exts.to_vec());
                }
                if let Some((name, _)) = opts.filters.first() {
                    dlg = dlg.default_file_filter(name);
                }
            }
        }
        if let Some(dir) = opts.directory.or_else(|| self.last_dir.clone()) {
            dlg = dlg.initial_directory(dir);
        }
        if let Some(name) = &opts.file_name {
            dlg = dlg.default_file_name(name);
        }
        match mode {
            Mode::PickFile => dlg.pick_file(),
            Mode::PickFolder => dlg.pick_directory(),
            Mode::SaveFile => dlg.save_file(),
        }
        self.linux.active_in_app = Some((intent, dlg));
    }

    fn drive_in_app(&mut self, ctx: &egui::Context) {
        let Some((intent, dlg)) = self.linux.active_in_app.as_mut() else {
            return;
        };
        dlg.update(ctx);
        if let Some(path) = dlg.take_picked() {
            let intent = intent.clone();
            self.linux.active_in_app = None;
            self.remember_dir(&path);
            self.pending = Some((intent, path));
        } else if matches!(
            dlg.state(),
            egui_file_dialog::DialogState::Cancelled | egui_file_dialog::DialogState::Closed
        ) {
            self.linux.active_in_app = None;
        }
    }
}

/// Cheap native-chain viability check: env vars + file existence only,
/// no D-Bus traffic, cannot block. Either a session bus with an
/// installed portal (rfd's primary path) or zenity on PATH (rfd's
/// fallback path) makes native dialogs viable.
#[cfg(target_os = "linux")]
fn probe_native_prerequisites() -> Result<(), String> {
    let has_zenity = which_on_path("zenity");
    let has_bus = std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_some()
        || std::env::var_os("XDG_RUNTIME_DIR")
            .map(|dir| std::path::Path::new(&dir).join("bus").exists())
            .unwrap_or(false);
    let portal_installed = [
        "/usr/share/dbus-1/services/org.freedesktop.portal.Desktop.service",
        "/usr/libexec/xdg-desktop-portal",
        "/usr/lib/xdg-desktop-portal",
    ]
    .iter()
    .any(|p| std::path::Path::new(p).exists());

    native_chain_viable(has_zenity, has_bus, portal_installed)
}

/// Pure decision core of [`probe_native_prerequisites`] (unit-tested).
#[cfg(target_os = "linux")]
fn native_chain_viable(
    has_zenity: bool,
    has_bus: bool,
    portal_installed: bool,
) -> Result<(), String> {
    if has_zenity || (has_bus && portal_installed) {
        return Ok(());
    }
    let detail = if !has_bus {
        "no D-Bus session bus"
    } else {
        "xdg-desktop-portal is not installed"
    };
    Err(format!(
        "Native file dialogs unavailable ({detail}, and no 'zenity' on PATH). \
         Using the built-in file browser. Install zenity (dnf/apt install \
         zenity) or run inside a desktop session for native dialogs."
    ))
}

/// Does `bin` exist as an executable file on `$PATH`?
#[cfg(target_os = "linux")]
fn which_on_path(bin: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(bin);
        std::fs::metadata(&candidate)
            .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    })
}

/// Async portal-health canary: read the FileChooser portal interface
/// version over the session bus via `busctl` (ships with systemd, so
/// present on every RHEL/Alma/Fedora/Debian/Ubuntu target). The
/// interface is only exposed when a real backend (e.g.
/// xdg-desktop-portal-gtk) implements it, so this detects the exact
/// environment where rfd's portal call would hang: xdg-desktop-portal
/// accepts OpenFile but no backend ever sends a Response. `--timeout`
/// bounds the D-Bus call; the subprocess isolates us from any hang.
#[cfg(target_os = "linux")]
fn spawn_portal_canary() -> std::sync::mpsc::Receiver<Result<(), String>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = match std::process::Command::new("busctl")
            .args([
                "--user",
                "--timeout=2",
                "get-property",
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.FileChooser",
                "version",
            ])
            .output()
        {
            Ok(out) if out.status.success() => Ok(()),
            Ok(out) => Err(format!(
                "portal FileChooser backend not responding ({})",
                String::from_utf8_lossy(&out.stderr).trim()
            )),
            // busctl missing (non-systemd distro): inconclusive — keep
            // the probe verdict rather than downgrading.
            Err(_) => Ok(()),
        };
        let _ = tx.send(result);
    });
    rx
}

/// Route a completed pick to its consumer. Called once per frame from
/// `NereidsApp::update`, after all panels have run, so results are
/// applied exactly once regardless of which panel opened the dialog —
/// even if that panel is no longer visible.
pub fn dispatch_results(state: &mut AppState) {
    let Some((intent, path)) = state.file_dialogs.take_any() else {
        return;
    };
    match intent {
        DialogIntent::OpenProject => crate::project::load_project_from_path(state, &path),
        DialogIntent::SaveProjectAs => crate::project::on_save_project_picked(state, path),
        DialogIntent::ExportDirectory => state.export_directory = Some(path),
        DialogIntent::SaveTilePng { tile_idx, label } => {
            crate::guided::result_widgets::save_tile_png(state, tile_idx, &label, &path);
        }
        DialogIntent::InstallLocalEndf => {
            crate::guided::configure::install_local_endf(state, &path);
        }
        DialogIntent::TiffSample => crate::guided::load::on_tiff_sample_picked(state, path),
        DialogIntent::TiffOpenBeam => crate::guided::load::on_tiff_open_beam_picked(state, path),
        DialogIntent::SpectrumFile => crate::guided::load::on_spectrum_picked(state, path),
        DialogIntent::Hdf5Sample => crate::guided::load::on_hdf5_sample_picked(state, path),
        DialogIntent::Hdf5OpenBeam => crate::guided::load::on_hdf5_ob_picked(state, path),
        DialogIntent::ResolutionFile(target) => on_resolution_file_picked(state, target, path),
    }
}

/// Apply a picked tabulated-resolution file to the card identified by
/// `target`, mirroring the invalidation each card performs for its
/// other (non-file) changes.
fn on_resolution_file_picked(state: &mut AppState, target: ResolutionTarget, path: PathBuf) {
    let flight_path_m = state.beamline.flight_path_m;
    match target {
        ResolutionTarget::Configure => {
            if crate::widgets::design::apply_resolution_file(
                &mut state.resolution_mode,
                path,
                flight_path_m,
            ) {
                state.spatial_result = None;
                state.pixel_fit_result = None;
            }
        }
        ResolutionTarget::ForwardModel => {
            if crate::widgets::design::apply_resolution_file(
                &mut state.fm_resolution_mode,
                path,
                flight_path_m,
            ) {
                state.fm_spectrum = None;
                state.fm_per_isotope_spectra.clear();
            }
        }
        ResolutionTarget::Detectability => {
            if crate::widgets::design::apply_resolution_file(
                &mut state.detect_resolution_mode,
                path,
                flight_path_m,
            ) {
                state.detect_results.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ResolutionMode;

    #[test]
    fn take_any_is_consume_once() {
        let mut dialogs = FileDialogs::default();
        dialogs.inject(DialogIntent::ExportDirectory, PathBuf::from("/tmp/x"));
        assert!(dialogs.take_any().is_some());
        assert!(dialogs.take_any().is_none());
    }

    #[test]
    fn dispatch_export_directory_sets_state() {
        let mut state = AppState::default();
        state
            .file_dialogs
            .inject(DialogIntent::ExportDirectory, PathBuf::from("/tmp/exports"));
        dispatch_results(&mut state);
        assert_eq!(state.export_directory, Some(PathBuf::from("/tmp/exports")));
    }

    #[test]
    fn dispatch_without_pending_is_noop() {
        let mut state = AppState::default();
        dispatch_results(&mut state);
        assert!(state.export_directory.is_none());
    }

    #[test]
    fn dispatch_tiff_sample_sets_path_and_invalidates() {
        let mut state = AppState {
            load_error: true,
            ..Default::default()
        };
        state
            .file_dialogs
            .inject(DialogIntent::TiffSample, PathBuf::from("/data/sample.tif"));
        dispatch_results(&mut state);
        assert_eq!(state.sample_path, Some(PathBuf::from("/data/sample.tif")));
        assert!(!state.load_error);
        assert!(state.sample_data.is_none());
        assert!(state.normalized.is_none());
        assert!(state.spatial_result.is_none());
    }

    #[test]
    fn dispatch_spectrum_clears_derived_data() {
        let mut state = AppState {
            load_error: true,
            ..Default::default()
        };
        state
            .file_dialogs
            .inject(DialogIntent::SpectrumFile, PathBuf::from("/data/spec.csv"));
        dispatch_results(&mut state);
        assert_eq!(state.spectrum_path, Some(PathBuf::from("/data/spec.csv")));
        assert!(state.spectrum_values.is_none());
        assert!(state.energies.is_none());
        assert!(!state.load_error);
    }

    #[test]
    fn dispatch_resolution_file_bad_file_sets_error() {
        let mut state = AppState::default();
        // An empty temp file cannot parse as a tabulated resolution.
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        state.file_dialogs.inject(
            DialogIntent::ResolutionFile(ResolutionTarget::Configure),
            tmp.path().to_path_buf(),
        );
        dispatch_results(&mut state);
        match &state.resolution_mode {
            ResolutionMode::Tabulated { data, error, path } => {
                assert!(data.is_none());
                assert!(error.is_some(), "empty file must produce a parse error");
                assert_eq!(path, tmp.path());
            }
            other => panic!("expected Tabulated, got {other:?}"),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_chain_viability_matrix() {
        // zenity alone suffices (rfd's fallback path, direct child
        // process — the reliable path for ssh -X and containers).
        assert!(native_chain_viable(true, false, false).is_ok());
        // Bus + installed portal suffices (the desktop path).
        assert!(native_chain_viable(false, true, true).is_ok());
        // Bus without portal, portal without bus, or nothing: in-app.
        assert!(native_chain_viable(false, true, false).is_err());
        assert!(native_chain_viable(false, false, true).is_err());
        let err = native_chain_viable(false, false, false).unwrap_err();
        assert!(err.contains("zenity"), "message must name the fix: {err}");
    }
}
