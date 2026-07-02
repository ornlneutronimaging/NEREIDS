//! Cross-platform file-dialog facade.
//!
//! Every file/folder picker in the GUI goes through [`FileDialogs`]: a
//! call site opens a dialog tagged with a [`DialogIntent`], and the
//! completed pick is routed by [`dispatch_results`] at the end of
//! `NereidsApp::update`. This decouples "the user clicked a picker
//! button" (UI code, often deep inside a panel with partial borrows of
//! `AppState`) from "a path was chosen" (state mutation, always with
//! full `&mut AppState`), so the dialog backend is free to resolve
//! immediately (blocking native dialog) or on a later frame (retained
//! in-app dialog, worker-thread native dialog) without the call sites
//! caring which happened.
//!
//! On macOS/Windows the native rfd dialog blocks inside `open()` and
//! the result is dispatched the same frame — identical behaviour to
//! the previous direct `rfd::FileDialog` calls.

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
#[derive(Default)]
pub struct DialogOptions {
    pub title: Option<&'static str>,
    /// `(name, extensions)`, e.g. `("TIFF", &["tif", "tiff"])`.
    pub filters: Vec<(&'static str, &'static [&'static str])>,
    /// Pre-filled file name (save dialogs).
    pub file_name: Option<String>,
    /// Initial directory. `None` = backend default (native dialogs
    /// remember their last location themselves).
    pub directory: Option<PathBuf>,
}

enum Mode {
    PickFile,
    PickFolder,
    SaveFile,
}

/// Poll-based dialog service stored in [`AppState`].
#[derive(Default)]
pub struct FileDialogs {
    /// Completed pick awaiting dispatch.
    pending: Option<(DialogIntent, PathBuf)>,
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

        let mut dlg = rfd::FileDialog::new();
        if let Some(t) = opts.title {
            dlg = dlg.set_title(t);
        }
        for (name, exts) in &opts.filters {
            dlg = dlg.add_filter(*name, exts);
        }
        if let Some(d) = opts.directory {
            dlg = dlg.set_directory(d);
        }
        if let Some(n) = &opts.file_name {
            dlg = dlg.set_file_name(n);
        }
        let picked = match mode {
            Mode::PickFile => dlg.pick_file(),
            Mode::PickFolder => dlg.pick_folder(),
            Mode::SaveFile => dlg.save_file(),
        };
        if let Some(path) = picked {
            self.pending = Some((intent, path));
        }
    }

    /// Per-frame driver hook, called once from `NereidsApp::update`.
    /// The blocking rfd tiers resolve inside `open()`, so this is
    /// currently a no-op; retained backends (in-app dialog, worker
    /// threads) plug in here.
    pub fn update(&mut self, _ctx: &egui::Context) {}

    /// Take the completed pick, if any (consume-once).
    pub fn take_any(&mut self) -> Option<(DialogIntent, PathBuf)> {
        self.pending.take()
    }

    /// Test-only: inject a completed pick as if a dialog resolved.
    #[cfg(test)]
    pub fn inject(&mut self, intent: DialogIntent, path: PathBuf) {
        self.pending = Some((intent, path));
    }
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
}
