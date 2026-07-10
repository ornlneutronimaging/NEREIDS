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
//!   when a resolved native request combined with the latch shows the
//!   final zenity leg failed (see `FileDialogs::apply_native_outcome`),
//!   or when the user clicks the escape hatch. Works in every environment
//!   (containers, root, no D-Bus, `ssh -X`) with zero system
//!   dependencies — the environments of issue #526.

use std::path::PathBuf;

use crate::state::{AppState, InputMode, SaveDataMode};
use nereids_endf::retrieval::EndfLibrary;

/// Which UI feature requested the dialog — routes the picked path in
/// [`dispatch_results`].
///
/// Capture rule: any state that determines how a picked path is
/// INTERPRETED (save mode, input mode, binning, target library) is
/// carried in the intent, snapshotted when the dialog is requested —
/// the dialog can resolve frames (or, on the Linux native tier,
/// seconds) later, and reading such state at resolution time would let
/// mid-dialog edits retroactively change what the pick does. State the
/// result is VALIDATED AGAINST (current selection, loaded sample) is
/// deliberately read at dispatch: validation must reflect what is
/// loaded when the pick lands.
#[derive(Clone, Debug, PartialEq)]
pub enum DialogIntent {
    /// Open a `.nrd.h5` project (toolbar, Ctrl/Cmd+O).
    OpenProject,
    /// "Save As" target for the current project (save modal). Carries
    /// the data mode chosen in the modal at request time.
    SaveProjectAs { mode: SaveDataMode },
    /// Export directory for spatial-map results (Studio dock).
    ExportDirectory,
    /// Save one result tile as a colormapped PNG (tile toolbelt).
    SaveTilePng { tile_idx: usize, label: String },
    /// Install a local ENDF file into the cache (Configure step, #523).
    /// Carries the library selected when the picker was opened; the
    /// isotope-in-selection check stays on current state.
    InstallLocalEndf { library: EndfLibrary },
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
    /// Optional open-beam NeXus/HDF5 file (Load step). Carries the
    /// input mode and event-binning parameters captured at request
    /// time (a same-shape, differently-binned OB would otherwise
    /// install silently); validation against the sample stays on
    /// current state — the OB must match whatever is loaded when it
    /// lands.
    Hdf5OpenBeam {
        mode: InputMode,
        event_params: nereids_io::nexus::EventBinningParams,
    },
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
    /// Show the waiting overlay immediately instead of after the grace
    /// period — set when a second request arrives while this one is
    /// still on screen, so the refusal to open another dialog is
    /// visible rather than silent.
    overlay_forced: bool,
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
    canary_rx: Option<std::sync::mpsc::Receiver<CanaryVerdict>>,
    /// Probe-failure message held back until the canary rules: if the
    /// canary proves the portal answers, the probe was a false negative
    /// (non-FHS install paths) and the warning is discarded unshown.
    pending_probe_warning: Option<String>,
    /// The in-app tier came from the (best-effort) probe, so a
    /// `Working` canary verdict may upgrade it to native. Cleared by
    /// the runtime latch and the escape hatch — those downgrades are
    /// evidence of an actually-broken chain and are never overturned.
    canary_may_upgrade: bool,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum LinuxTier {
    /// rfd portal backend (+ its zenity fallback) on a worker thread.
    Native,
    /// Pure-egui in-app browser.
    InApp,
}

/// What the portal canary learned about the FileChooser backend.
#[cfg(target_os = "linux")]
enum CanaryVerdict {
    /// The FileChooser interface answered: a portal backend exists.
    Working,
    /// The bus call ran but the interface did not answer: no backend
    /// implements FileChooser.
    Broken(String),
    /// `busctl` itself could not run (non-systemd distro): no evidence
    /// either way.
    Inconclusive,
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
        #[cfg(not(target_os = "linux"))]
        {
            // A new request supersedes any unconsumed earlier result.
            self.pending = None;
            if let Some(path) = run_rfd_blocking(mode, &opts) {
                self.remember_dir(&path);
                self.pending = Some((intent, path));
            }
        }

        #[cfg(target_os = "linux")]
        {
            // One native dialog at a time — regardless of intent. rfd
            // gives us no way to close an on-screen portal/zenity
            // dialog programmatically, so superseding the in-flight
            // request would orphan a live dialog whose eventual pick is
            // silently discarded (its receiver is gone). Refuse the new
            // request and make the refusal visible by forcing the
            // waiting overlay. The in-app tier needs no such guard for
            // pointer input — its dialog is modal (`as_modal(true)`) so
            // clicks cannot reach picker buttons — and the only
            // modality bypass, `ctx.input()`-based keyboard shortcuts,
            // is gated in app.rs via `dialog_in_flight()`. Superseding
            // an in-app dialog is safe regardless (it is fully owned;
            // no orphaned window, no lost pick).
            if let Some(req) = self.linux.active_native.as_mut() {
                req.overlay_forced = true;
                return;
            }
            // A new request supersedes any unconsumed earlier result.
            self.pending = None;
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
    /// the escape-hatch overlay. On macOS/Windows (dialogs resolve
    /// inside `open()`) it only surfaces a latched rfd error as a
    /// warning.
    pub fn update(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = ctx;
            // Dialogs resolve synchronously inside `open()`, so any
            // latched rfd error is already final here. There is no
            // fallback tier off-Linux — the warning must not claim one.
            if let Some(msg) = crate::logging::take_dialog_backend_failure() {
                self.warning = Some(format!("Native file dialog failed: {msg}"));
            }
        }

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

    /// Take the pending dialog warning, if any (consume-once) — probe/
    /// canary verdicts and native-backend failures alike, shown by the
    /// app in the native-dialog warning banner.
    pub fn take_warning(&mut self) -> Option<String> {
        self.warning.take()
    }

    /// Is a dialog currently open? Used to gate keyboard shortcuts:
    /// pointer input cannot reach picker buttons while a dialog is up
    /// (native dialogs refuse re-open; the in-app dialog is modal), but
    /// `ctx.input()`-based shortcuts bypass widget modality entirely.
    /// Always `false` off-Linux, where dialogs block inside `open()`.
    pub fn dialog_in_flight(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.linux.active_native.is_some() || self.linux.active_in_app.is_some()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
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
    /// D-Bus traffic, cannot block). The async portal canary always
    /// runs afterwards: on a viable-looking chain it can catch a
    /// backend-less portal, and on a failed probe it can rescue a
    /// false negative (the probe's path list is best-effort — see
    /// [`probe_native_prerequisites`]), so the probe's warning is
    /// stashed until the canary rules.
    fn decide_tier_once(&mut self) {
        if self.linux.tier.is_some() {
            return;
        }
        match probe_native_prerequisites() {
            Ok(()) => {
                self.linux.tier = Some(LinuxTier::Native);
            }
            Err(msg) => {
                self.linux.tier = Some(LinuxTier::InApp);
                self.linux.pending_probe_warning = Some(msg);
                self.linux.canary_may_upgrade = true;
            }
        }
        self.linux.canary_rx = Some(spawn_portal_canary(self.linux.ctx.clone()));
    }

    /// Drain the portal-canary channel and apply its verdict. The
    /// canary detects a MISSING FileChooser backend (interface not
    /// exposed); an installed-but-hung backend still answers the
    /// property read — that case is covered by the dialog worker
    /// thread + escape hatch, not here.
    fn poll_canary(&mut self) {
        let Some(rx) = &self.linux.canary_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(verdict) => {
                self.linux.canary_rx = None;
                self.apply_canary_verdict(verdict, which_on_path("zenity"));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Canary worker died without reporting: no evidence
                // either way.
                self.linux.canary_rx = None;
                self.apply_canary_verdict(CanaryVerdict::Inconclusive, which_on_path("zenity"));
            }
        }
    }

    /// Apply the canary verdict to the current tier.
    ///
    /// - Native tier: `Broken` downgrades (unless zenity keeps rfd's
    ///   portal-free fallback viable) and reopens any in-flight native
    ///   request in-app; `Working`/`Inconclusive` change nothing.
    /// - In-app tier reached via a failed probe (`canary_may_upgrade`):
    ///   `Working` proves the portal answers despite the probe missing
    ///   it, so upgrade to native and discard the stashed probe
    ///   warning; `Broken`/`Inconclusive` confirm the probe, so
    ///   surface the stashed warning now.
    /// - A latch or escape-hatch downgrade (`canary_may_upgrade`
    ///   cleared) is never overturned.
    ///
    /// `has_zenity` is passed in (rather than read from `$PATH` here)
    /// so both `Broken` arms are deterministically testable.
    fn apply_canary_verdict(&mut self, verdict: CanaryVerdict, has_zenity: bool) {
        match self.linux.tier {
            Some(LinuxTier::Native) => {
                if let CanaryVerdict::Broken(msg) = verdict {
                    // Zenity alone still makes the native chain viable —
                    // rfd falls back to it without touching the portal.
                    if has_zenity {
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
                        // A native request may already be waiting on the
                        // dead portal — reopen it in-app so the user's
                        // click still lands in a working dialog.
                        if let Some(req) = self.linux.active_native.take() {
                            self.open_in_app(req.mode, req.intent, req.opts);
                        }
                    }
                }
            }
            Some(LinuxTier::InApp) if self.linux.canary_may_upgrade => {
                self.linux.canary_may_upgrade = false;
                match verdict {
                    CanaryVerdict::Working => {
                        self.linux.tier = Some(LinuxTier::Native);
                        self.linux.pending_probe_warning = None;
                    }
                    CanaryVerdict::Broken(_) | CanaryVerdict::Inconclusive => {
                        self.warning = self.linux.pending_probe_warning.take();
                    }
                }
            }
            _ => {}
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
            overlay_forced: false,
        });
    }

    /// Poll the worker; if the native dialog is taking suspiciously
    /// long to resolve, offer the built-in browser as an escape hatch
    /// (a hung portal never delivers a result, and rfd gives us no way
    /// to cancel it — the orphaned worker thread is the accepted cost).
    /// A resolved request is combined with the log-bridge latch in
    /// [`Self::apply_native_outcome`].
    fn poll_native(&mut self, ctx: &egui::Context) {
        use std::sync::mpsc::TryRecvError;

        let Some(req) = &self.linux.active_native else {
            return;
        };
        let picked = match req.rx.try_recv() {
            Ok(picked) => picked,
            // Worker panicked without sending: no pick — classified
            // against the latch exactly like a returned `None`.
            Err(TryRecvError::Disconnected) => None,
            Err(TryRecvError::Empty) => {
                // Still open. After a short grace period (or right away
                // when a refused second request forced it), offer the
                // in-app fallback without killing the native dialog
                // (we cannot tell "hung portal" from "user is
                // browsing a big directory").
                let mut escape = false;
                let elapsed = req.started.elapsed();
                if req.overlay_forced || elapsed >= std::time::Duration::from_secs(1) {
                    egui::Window::new("native_dialog_pending")
                        .title_bar(false)
                        .resizable(false)
                        .anchor(egui::Align2::CENTER_TOP, [0.0_f32, 8.0_f32])
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
                if escape && let Some(req) = self.linux.active_native.take() {
                    self.linux.tier = Some(LinuxTier::InApp);
                    // The user judged the native chain unusable — a
                    // later canary verdict must not upgrade back to it.
                    self.linux.canary_may_upgrade = false;
                    self.open_in_app(req.mode, req.intent, req.opts);
                }
                return;
            }
        };
        if let Some(req) = self.linux.active_native.take() {
            let latched = crate::logging::take_dialog_backend_failure();
            self.apply_native_outcome(req, picked, latched);
        }
    }

    /// Combine a resolved native request with the log-bridge latch to
    /// decide what actually happened. rfd's Linux chain tries the
    /// portal first and falls back to zenity, log-erroring on every
    /// failed leg (rfd 0.17.2: "Can't connect to a portal: ...",
    /// "Failed to connect to session bus: ..." from
    /// src/backend/xdg_desktop_portal/portal/libdbus.rs, "OpenFile
    /// failed: ..." from .../portal/mod.rs), so a latched error alone
    /// does not mean the dialog failed — the zenity leg may have gone
    /// on to serve the user. Only the final zenity leg's errors name
    /// zenity ("Failed to pick file with zenity: ...", "Failed to save
    /// file with zenity: ...", "Failed to open zenity dialog: ..." from
    /// src/backend/xdg_desktop_portal.rs), which makes "mentions
    /// zenity" the end-of-chain discriminator. Wording pinned via
    /// rfd "=0.17.2" — see the workspace Cargo.toml.
    fn apply_native_outcome(
        &mut self,
        req: NativeRequest,
        picked: Option<PathBuf>,
        latched: Option<String>,
    ) {
        if let Some(path) = picked {
            // A pick means the chain worked end-to-end; any latched
            // error was a non-final leg falling through. Discard it.
            self.remember_dir(&path);
            self.pending = Some((req.intent, path));
            return;
        }
        match latched {
            Some(msg) if msg.contains("zenity") => {
                // The final leg failed: no dialog ever served the user.
                // Downgrade for good and reopen the same request in the
                // built-in browser so the click still lands somewhere.
                self.linux.tier = Some(LinuxTier::InApp);
                self.linux.canary_may_upgrade = false;
                self.warning = Some(format!(
                    "Native file dialog failed: {msg} — switched to the built-in file browser."
                ));
                self.open_in_app(req.mode, req.intent, req.opts);
            }
            Some(msg) => {
                // Portal leg failed but the zenity leg then ran and the
                // user cancelled: the chain works, keep the native tier
                // (mirror of `apply_canary_verdict`'s zenity-keeps-
                // native rule). Record the portal trouble for
                // diagnosis.
                tracing::info!(
                    error = %msg,
                    "portal dialog leg failed; zenity fallback served the request"
                );
            }
            // Plain user cancel: nothing to do.
            None => {}
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
                // Save dialogs use an extension dropdown instead of
                // filters — but only when no name was pre-filled:
                // egui-file-dialog 0.12.0 applies the default save
                // extension to the pre-filled name via
                // `PathBuf::set_extension`, which replaces only the
                // final dot-component ("x.nrd" + "nrd.h5" ->
                // "x.nrd.nrd.h5"). A supplied default name already
                // carries its extension; for typed-in names,
                // `ensure_extension` at dispatch remains the safety
                // net.
                if opts.file_name.is_none() {
                    for (name, exts) in &opts.filters {
                        if let Some(ext) = exts.first() {
                            dlg = dlg.add_save_extension(name, ext);
                        }
                    }
                    if let Some((name, _)) = opts.filters.first() {
                        dlg = dlg.default_save_extension(name);
                    }
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
///
/// The portal path list is best-effort FHS locations — non-FHS distros
/// (NixOS, Guix) install the portal elsewhere and read as a false
/// negative here. The canary is the authoritative check and can rescue
/// such a false negative (see [`FileDialogs::apply_canary_verdict`]).
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
/// interface is only exposed when a backend (e.g.
/// xdg-desktop-portal-gtk) implements it, so a failed read means the
/// no-backend case — rfd's portal call would never produce a dialog.
/// An installed-but-broken/hung backend still answers the property
/// read; that case is covered by the dialog worker thread + escape
/// hatch, not by this canary. `--timeout` bounds the D-Bus call; the
/// subprocess isolates us from any hang. After sending the verdict the
/// worker requests a repaint (like `open_native_worker`'s), so
/// `poll_canary` runs promptly instead of waiting for the next
/// user-driven frame.
#[cfg(target_os = "linux")]
fn spawn_portal_canary(ctx: Option<egui::Context>) -> std::sync::mpsc::Receiver<CanaryVerdict> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let verdict = match std::process::Command::new("busctl")
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
            Ok(out) if out.status.success() => CanaryVerdict::Working,
            Ok(out) => CanaryVerdict::Broken(format!(
                "portal FileChooser backend not responding ({})",
                String::from_utf8_lossy(&out.stderr).trim()
            )),
            // busctl missing (non-systemd distro): no evidence either
            // way — keep the probe verdict.
            Err(_) => CanaryVerdict::Inconclusive,
        };
        let _ = tx.send(verdict);
        if let Some(ctx) = ctx {
            ctx.request_repaint();
        }
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
        DialogIntent::SaveProjectAs { mode } => {
            crate::project::on_save_project_picked(state, path, mode);
        }
        DialogIntent::ExportDirectory => state.export_directory = Some(path),
        DialogIntent::SaveTilePng { tile_idx, label } => {
            crate::guided::result_widgets::save_tile_png(state, tile_idx, &label, &path);
        }
        DialogIntent::InstallLocalEndf { library } => {
            crate::guided::configure::install_local_endf(state, &path, library);
        }
        DialogIntent::TiffSample => crate::guided::load::on_tiff_sample_picked(state, path),
        DialogIntent::TiffOpenBeam => crate::guided::load::on_tiff_open_beam_picked(state, path),
        DialogIntent::SpectrumFile => crate::guided::load::on_spectrum_picked(state, path),
        DialogIntent::Hdf5Sample => crate::guided::load::on_hdf5_sample_picked(state, path),
        DialogIntent::Hdf5OpenBeam { mode, event_params } => {
            crate::guided::load::on_hdf5_ob_picked(state, path, mode, &event_params);
        }
        DialogIntent::ResolutionFile(target) => on_resolution_file_picked(state, target, path),
    }
}

/// Apply a picked tabulated-resolution file to the card identified by
/// `target`, mirroring the invalidation each card performs for its
/// other (non-file) changes.
///
/// Pick-wins semantics: the resolving file replaces the whole mode. On
/// the Linux native tier the dialog can resolve seconds later, so the
/// user may have switched the card to Gaussian and tuned it meanwhile —
/// the replacement is then announced in the status bar instead of
/// silently discarding those edits.
fn on_resolution_file_picked(state: &mut AppState, target: ResolutionTarget, path: PathBuf) {
    use crate::state::ResolutionMode;

    let flight_path_m = state.beamline.flight_path_m;
    let (mode, invalidate): (&mut ResolutionMode, fn(&mut AppState)) = match target {
        ResolutionTarget::Configure => (&mut state.resolution_mode, |s| {
            s.spatial_result = None;
            s.pixel_fit_result = None;
        }),
        ResolutionTarget::ForwardModel => (&mut state.fm_resolution_mode, |s| {
            s.fm_spectrum = None;
            s.fm_per_isotope_spectra.clear();
        }),
        ResolutionTarget::Detectability => (&mut state.detect_resolution_mode, |s| {
            s.detect_results.clear();
        }),
    };
    let replaced_gaussian = matches!(mode, ResolutionMode::Gaussian { .. });
    if crate::widgets::design::apply_resolution_file(mode, path, flight_path_m) {
        invalidate(state);
        if replaced_gaussian {
            state.status_message =
                "Tabulated resolution selected — replaced the Gaussian settings edited \
                 while the dialog was open"
                    .into();
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

    /// Minimal fitted result for SaveTilePng dispatch tests: one 2x2
    /// density map per label, plus an optional temperature map.
    fn spatial_result_with(
        labels: &[&str],
        with_temperature: bool,
    ) -> nereids_pipeline::spatial::SpatialResult {
        let map = || ndarray::Array2::from_shape_fn((2, 2), |(y, x)| (y * 2 + x) as f64);
        nereids_pipeline::spatial::SpatialResult {
            density_maps: labels.iter().map(|_| map()).collect(),
            uncertainty_maps: labels.iter().map(|_| map()).collect(),
            chi_squared_map: map(),
            deviance_per_dof_map: None,
            converged_map: ndarray::Array2::from_elem((2, 2), true),
            temperature_map: with_temperature.then(map),
            temperature_uncertainty_map: None,
            isotope_labels: labels.iter().map(|s| s.to_string()).collect(),
            anorm_map: None,
            background_maps: None,
            back_d_map: None,
            back_f_map: None,
            t0_us_map: None,
            l_scale_map: None,
            energy_scale_flight_path_m: None,
            baseline_global: None,
            baseline_e_ref_ev: None,
            baseline_maps: None,
            warnings: Vec::new(),
            n_converged: 4,
            n_total: 4,
            n_failed: 0,
        }
    }

    fn dispatch_save_tile_png(state: &mut AppState, tile_idx: usize, label: &str, path: PathBuf) {
        state.file_dialogs.inject(
            DialogIntent::SaveTilePng {
                tile_idx,
                label: label.to_string(),
            },
            path,
        );
        dispatch_results(state);
    }

    #[test]
    fn dispatch_save_tile_png_without_result_errors() {
        let mut state = AppState::default();
        dispatch_save_tile_png(
            &mut state,
            0,
            "Fe56",
            PathBuf::from("/tmp/never-written.png"),
        );
        assert_eq!(state.status_message, "PNG save error: no results available");
    }

    #[test]
    fn dispatch_save_tile_png_refuses_stale_label() {
        // The dialog can resolve frames after it was opened; if the
        // results were replaced meanwhile, the carried label no longer
        // names the tile at that index and nothing must be written.
        let mut state = AppState {
            spatial_result: Some(spatial_result_with(&["Fe56"], false)),
            ..Default::default()
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stale.png");
        dispatch_save_tile_png(&mut state, 0, "Gd157", path.clone());
        assert!(
            state.status_message.starts_with("PNG not saved:"),
            "expected refusal, got {:?}",
            state.status_message
        );
        assert!(!path.exists(), "stale-label save must not write a file");
    }

    #[test]
    fn dispatch_save_tile_png_saves_matching_density_tile() {
        let mut state = AppState {
            spatial_result: Some(spatial_result_with(&["Fe56"], false)),
            ..Default::default()
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("Fe56.png");
        dispatch_save_tile_png(&mut state, 0, "Fe56", path.clone());
        assert!(
            state.status_message.starts_with("Saved PNG:"),
            "expected success, got {:?}",
            state.status_message
        );
        assert!(path.exists());
    }

    #[test]
    fn dispatch_save_tile_png_maps_tiles_like_studio() {
        // Index mapping mirror of the Studio analysis column:
        // 0..n_density = isotopes, n_density = temperature (if present),
        // anything past that does not exist — even with a temperature
        // map available.
        let mut state = AppState {
            spatial_result: Some(spatial_result_with(&["Fe56"], true)),
            ..Default::default()
        };
        let dir = tempfile::tempdir().expect("tempdir");

        let temp_path = dir.path().join("temperature.png");
        dispatch_save_tile_png(&mut state, 1, "temperature", temp_path.clone());
        assert!(
            state.status_message.starts_with("Saved PNG:"),
            "expected temperature tile at n_density to save, got {:?}",
            state.status_message
        );
        assert!(temp_path.exists());

        let ghost_path = dir.path().join("ghost.png");
        dispatch_save_tile_png(&mut state, 2, "temperature", ghost_path.clone());
        assert_eq!(
            state.status_message,
            "PNG save error: tile 2 no longer exists"
        );
        assert!(!ghost_path.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn canary_rescues_probe_false_negative() {
        // Probe-failed state (e.g. non-FHS portal paths) + a Working
        // canary: upgrade to native, discard the stashed warning.
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::InApp);
        dialogs.linux.canary_may_upgrade = true;
        dialogs.linux.pending_probe_warning = Some("probe failed".into());
        dialogs.apply_canary_verdict(CanaryVerdict::Working, false);
        assert!(dialogs.linux.tier == Some(LinuxTier::Native));
        assert!(dialogs.warning.is_none(), "rescued probe must not warn");
        assert!(dialogs.linux.pending_probe_warning.is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn broken_canary_with_zenity_keeps_native_tier() {
        // Portal backend missing but zenity present: rfd's fallback
        // still serves native dialogs — no downgrade, no warning, and
        // any in-flight request keeps waiting on its worker.
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::Native);
        dialogs.linux.active_native = Some(native_request(DialogIntent::ExportDirectory));
        dialogs.apply_canary_verdict(CanaryVerdict::Broken("no backend".into()), true);
        assert!(dialogs.linux.tier == Some(LinuxTier::Native));
        assert!(dialogs.warning.is_none());
        assert!(dialogs.linux.active_native.is_some());
        assert!(dialogs.linux.active_in_app.is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn broken_canary_without_zenity_downgrades_and_reopens_in_app() {
        // Portal backend missing and no zenity: the native chain is
        // dead — downgrade, warn, and reopen the in-flight request in
        // the built-in browser so the user's click still lands.
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::Native);
        dialogs.linux.active_native = Some(native_request(DialogIntent::ExportDirectory));
        dialogs.apply_canary_verdict(CanaryVerdict::Broken("no backend".into()), false);
        assert!(dialogs.linux.tier == Some(LinuxTier::InApp));
        assert!(
            dialogs
                .warning
                .as_deref()
                .is_some_and(|w| w.contains("zenity")),
            "warning must name the fix"
        );
        assert!(dialogs.linux.active_native.is_none());
        assert!(
            dialogs.linux.active_in_app.is_some(),
            "pending request must reopen in-app"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn inconclusive_canary_surfaces_stashed_probe_warning() {
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::InApp);
        dialogs.linux.canary_may_upgrade = true;
        dialogs.linux.pending_probe_warning = Some("probe failed".into());
        dialogs.apply_canary_verdict(CanaryVerdict::Inconclusive, false);
        assert!(dialogs.linux.tier == Some(LinuxTier::InApp));
        assert_eq!(dialogs.warning.as_deref(), Some("probe failed"));
        assert!(!dialogs.linux.canary_may_upgrade);
    }

    /// Build an in-flight native request for outcome tests. The
    /// receiver never gets a message — `apply_native_outcome` takes the
    /// resolution as a parameter, mirroring how `poll_native` hands it
    /// the drained channel + latch.
    ///
    /// ORACLE COUPLING: the latched-message strings used by the
    /// outcome tests below are hand-copies of rfd 0.17.2 emissions,
    /// valid only under the exact `rfd = "=0.17.2"` pin (workspace
    /// Cargo.toml) — re-verify and update them on any deliberate bump.
    #[cfg(target_os = "linux")]
    fn native_request(intent: DialogIntent) -> NativeRequest {
        let (_tx, rx) = std::sync::mpsc::channel();
        NativeRequest {
            mode: Mode::PickFile,
            intent,
            opts: DialogOptions::default(),
            rx,
            started: std::time::Instant::now(),
            overlay_forced: false,
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_pick_discards_portal_leg_latch() {
        // Portal leg errored, zenity leg served a pick: the chain
        // worked — deliver the pick, keep the native tier, no warning.
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::Native);
        dialogs.apply_native_outcome(
            native_request(DialogIntent::ExportDirectory),
            Some(PathBuf::from("/tmp/picked")),
            Some("Failed to connect to session bus: org.freedesktop.DBus.Error.NoServer".into()),
        );
        assert_eq!(
            dialogs.take_any(),
            Some((DialogIntent::ExportDirectory, PathBuf::from("/tmp/picked")))
        );
        assert!(dialogs.linux.tier == Some(LinuxTier::Native));
        assert!(dialogs.take_warning().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn zenity_leg_failure_downgrades_and_reopens_in_app() {
        // No pick and the latch names zenity: the whole chain failed.
        // Downgrade, warn with the switch notice, and reopen the same
        // request in the built-in browser.
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::Native);
        dialogs.linux.canary_may_upgrade = true;
        dialogs.apply_native_outcome(
            native_request(DialogIntent::OpenProject),
            None,
            Some("Failed to pick file with zenity: not found".into()),
        );
        assert!(dialogs.linux.tier == Some(LinuxTier::InApp));
        assert!(!dialogs.linux.canary_may_upgrade);
        let warning = dialogs.take_warning().expect("warning must be set");
        assert!(warning.contains("zenity"), "got {warning:?}");
        assert!(
            warning.contains("switched to the built-in file browser"),
            "got {warning:?}"
        );
        assert!(
            matches!(
                &dialogs.linux.active_in_app,
                Some((DialogIntent::OpenProject, _))
            ),
            "the pending request must reopen in-app"
        );
        assert!(dialogs.take_any().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn portal_only_latch_on_cancel_keeps_native_tier() {
        // No pick but the latch is portal-leg-only: zenity ran and the
        // user cancelled — a working chain. Keep the native tier
        // (parity with apply_canary_verdict's zenity-keeps-native
        // rule), no warning, nothing reopened.
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::Native);
        dialogs.apply_native_outcome(
            native_request(DialogIntent::OpenProject),
            None,
            Some("OpenFile failed: org.freedesktop.DBus.Error.ServiceUnknown".into()),
        );
        assert!(dialogs.linux.tier == Some(LinuxTier::Native));
        assert!(dialogs.take_warning().is_none());
        assert!(dialogs.linux.active_in_app.is_none());
        assert!(dialogs.take_any().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn second_request_while_native_pending_does_not_supersede() {
        // A second click while a native dialog is on screen must not
        // drop the in-flight request (its dialog would be orphaned);
        // it forces the waiting overlay instead, and an unconsumed
        // earlier result survives the refused request.
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::Native);
        dialogs.linux.active_native = Some(native_request(DialogIntent::OpenProject));
        dialogs.pending = Some((DialogIntent::ExportDirectory, PathBuf::from("/tmp/x")));
        dialogs.pick_file(DialogIntent::TiffSample, DialogOptions::default());
        let req = dialogs
            .linux
            .active_native
            .as_ref()
            .expect("request must stay in flight");
        assert_eq!(req.intent, DialogIntent::OpenProject);
        assert!(req.overlay_forced, "refusal must force the overlay");
        assert_eq!(
            dialogs.take_any(),
            Some((DialogIntent::ExportDirectory, PathBuf::from("/tmp/x")))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn latched_downgrade_is_never_overturned_by_canary() {
        let mut dialogs = FileDialogs::default();
        dialogs.linux.tier = Some(LinuxTier::Native);
        dialogs.apply_native_outcome(
            native_request(DialogIntent::OpenProject),
            None,
            Some("Failed to pick file with zenity: exit status 1".into()),
        );
        assert!(dialogs.linux.tier == Some(LinuxTier::InApp));
        dialogs.apply_canary_verdict(CanaryVerdict::Working, true);
        assert!(dialogs.linux.tier == Some(LinuxTier::InApp));
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
