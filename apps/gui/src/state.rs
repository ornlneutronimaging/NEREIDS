//! Application state shared across all GUI panels.

use ndarray::{Array2, Array3};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use nereids_endf::resonance::ResonanceData;
use nereids_endf::retrieval::EndfLibrary;
use nereids_io::nexus::{Hdf5TreeEntry, NexusMetadata};
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use nereids_io::tof::BeamlineParams;

/// Lightweight session cache for persistence across app restarts.
/// Only stores the subset of state needed to resume a previous pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionCache {
    pub fitting_type: Option<FittingType>,
    pub data_type: Option<DataType>,
    pub input_mode: InputMode,
    /// Beamline flight path (m).
    pub flight_path_m: f64,
    /// Beamline delay (μs).
    pub delay_us: f64,
    /// Temperature (K).
    pub temperature_k: f64,
    /// Proton charge sample.
    pub proton_charge_sample: f64,
    /// Proton charge open beam.
    pub proton_charge_ob: f64,
    /// Isotope list: (z, a, symbol, density, enabled).
    pub isotopes: Vec<CachedIsotope>,
    /// ENDF library name (e.g. "ENDF/B-VIII.0").
    pub endf_library_name: String,
    /// Energy rebin factor (1 = no rebinning).
    #[serde(default = "default_rebin_factor")]
    pub rebin_factor: usize,
    /// Whether rebinning has been applied.
    #[serde(default)]
    pub rebin_applied: bool,
    /// Whether the sidebar is collapsed (icon-only mode).
    #[serde(default)]
    pub sidebar_collapsed: bool,
    /// KL background enabled (b₀ + b₁/√E).
    #[serde(default)]
    pub kl_background_enabled: bool,
    /// Bounded multiplicative baseline enabled.
    #[serde(default)]
    pub baseline_enabled: bool,
    /// Counts-KL proton-charge ratio `c = Q_s / Q_ob`.
    /// Defaults to 1.0 (caller PC-normalized the flux upstream).
    #[serde(default = "default_kl_c_ratio")]
    pub kl_c_ratio: f64,
    /// Counts-KL Nelder-Mead polish override.  `None` = dispatcher
    /// auto-disables polish for multi-pixel spatial fits.
    /// `Some(true/false)` forces polish on/off.
    #[serde(default)]
    pub kl_enable_polish_override: Option<bool>,
    /// Fit energy range restriction (SAMMY EMIN/EMAX equivalent).
    /// `None` (default) = full grid.
    #[serde(default)]
    pub fit_energy_range: Option<(f64, f64)>,
    /// Isotope groups: (z, name, members, density, enabled).
    /// ResonanceData is not serialized — members get Pending status on restore.
    #[serde(default)]
    pub isotope_groups: Vec<CachedGroupEntry>,
}

fn default_rebin_factor() -> usize {
    1
}

/// Default for SessionCache::kl_c_ratio when a saved session predates the field.
fn default_kl_c_ratio() -> f64 {
    1.0
}

/// A cached isotope entry (serializable).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedIsotope {
    pub z: u32,
    pub a: u32,
    pub symbol: String,
    pub density: f64,
    pub enabled: bool,
}

/// A cached isotope group entry (serializable, without resonance data).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedGroupEntry {
    pub z: u32,
    pub name: String,
    pub members: Vec<CachedGroupMember>,
    pub initial_density: f64,
    pub enabled: bool,
}

/// A cached group member (serializable, without resonance data).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedGroupMember {
    pub a: u32,
    pub symbol: String,
    pub ratio: f64,
}

impl SessionCache {
    /// Build a session cache from the current application state.
    pub fn from_state(state: &AppState) -> Option<Self> {
        // Only cache if a pipeline has been configured
        if state.fitting_type.is_none() || state.data_type.is_none() || state.pipeline.is_empty() {
            return None;
        }
        Some(Self {
            fitting_type: state.fitting_type,
            data_type: state.data_type,
            input_mode: state.input_mode,
            flight_path_m: state.beamline.flight_path_m,
            delay_us: state.beamline.delay_us,
            temperature_k: state.temperature_k,
            proton_charge_sample: state.proton_charge_sample,
            proton_charge_ob: state.proton_charge_ob,
            isotopes: state
                .isotope_entries
                .iter()
                .map(|e| CachedIsotope {
                    z: e.z,
                    a: e.a,
                    symbol: e.symbol.clone(),
                    density: e.initial_density,
                    enabled: e.enabled,
                })
                .collect(),
            endf_library_name: crate::widgets::design::library_name(state.endf_library).to_string(),
            rebin_factor: state.rebin_factor,
            rebin_applied: state.rebin_applied,
            sidebar_collapsed: state.sidebar_collapsed,
            kl_background_enabled: state.kl_background_enabled,
            baseline_enabled: state.baseline_enabled,
            kl_c_ratio: state.kl_c_ratio,
            kl_enable_polish_override: state.kl_enable_polish_override,
            fit_energy_range: state.fit_energy_range,
            isotope_groups: state
                .isotope_groups
                .iter()
                .map(|g| CachedGroupEntry {
                    z: g.z,
                    name: g.name.clone(),
                    members: g
                        .members
                        .iter()
                        .map(|m| CachedGroupMember {
                            a: m.a,
                            symbol: m.symbol.clone(),
                            ratio: m.ratio,
                        })
                        .collect(),
                    initial_density: g.initial_density,
                    enabled: g.enabled,
                })
                .collect(),
        })
    }

    /// Apply cached settings to app state (restores pipeline + config).
    pub fn apply_to(&self, state: &mut AppState) {
        state.fitting_type = self.fitting_type;
        state.data_type = self.data_type;
        state.input_mode = self.input_mode;
        state.beamline.flight_path_m = self.flight_path_m;
        state.beamline.delay_us = self.delay_us;
        state.temperature_k = self.temperature_k;
        state.proton_charge_sample = self.proton_charge_sample;
        state.proton_charge_ob = self.proton_charge_ob;

        // Restore isotope entries (without resonance data — needs re-fetch)
        state.isotope_entries = self
            .isotopes
            .iter()
            .map(|c| IsotopeEntry {
                z: c.z,
                a: c.a,
                symbol: c.symbol.clone(),
                initial_density: c.density,
                resonance_data: None,
                enabled: c.enabled,
                endf_status: EndfStatus::Pending,
            })
            .collect();

        // Restore isotope groups (without resonance data — needs re-fetch)
        state.isotope_groups = self
            .isotope_groups
            .iter()
            .map(|g| IsotopeGroupEntry {
                z: g.z,
                name: g.name.clone(),
                members: g
                    .members
                    .iter()
                    .map(|m| GroupMemberState {
                        a: m.a,
                        symbol: m.symbol.clone(),
                        ratio: m.ratio,
                        resonance_data: None,
                        endf_status: EndfStatus::Pending,
                    })
                    .collect(),
                initial_density: g.initial_density,
                enabled: g.enabled,
            })
            .collect();

        // Restore library by matching label
        state.endf_library = match self.endf_library_name.as_str() {
            "ENDF/B-VIII.1" => nereids_endf::retrieval::EndfLibrary::EndfB8_1,
            "JEFF-3.3" => nereids_endf::retrieval::EndfLibrary::Jeff3_3,
            "JENDL-5" => nereids_endf::retrieval::EndfLibrary::Jendl5,
            "TENDL-2023" => nereids_endf::retrieval::EndfLibrary::Tendl2023,
            "CENDL-3.2" => nereids_endf::retrieval::EndfLibrary::Cendl3_2,
            _ => nereids_endf::retrieval::EndfLibrary::EndfB8_0,
        };

        // Restore rebin state
        state.rebin_factor = self.rebin_factor;
        state.rebin_applied = self.rebin_applied;

        // Restore sidebar state
        state.sidebar_collapsed = self.sidebar_collapsed;

        // Restore background normalization state
        state.kl_background_enabled = self.kl_background_enabled;
        state.baseline_enabled = self.baseline_enabled;
        state.kl_c_ratio = self.kl_c_ratio;
        state.kl_enable_polish_override = self.kl_enable_polish_override;
        state.fit_energy_range = self.fit_energy_range;

        // Rebuild pipeline
        state.rebuild_pipeline();
    }

    /// Summary label for display (e.g. "Spatial + Events, 3 isotopes").
    pub fn summary(&self) -> String {
        let fitting = match self.fitting_type {
            Some(FittingType::Spatial) => "Spatial",
            Some(FittingType::Single) => "Single",
            None => "Unknown",
        };
        let data = match self.data_type {
            Some(DataType::Events) => "Events",
            Some(DataType::PreNormalized) => "Pre-norm",
            None => "Unknown",
        };
        let n_iso = self.isotopes.iter().filter(|i| i.enabled).count();
        if n_iso > 0 {
            format!("{fitting} + {data}, {n_iso} isotope(s)")
        } else {
            format!("{fitting} + {data}")
        }
    }
}

/// Result of a background ENDF fetch for a single isotope.
///
/// Uses `(z, a)` to identify which isotope entry the result belongs to,
/// rather than positional index which is fragile if the list is mutated
/// during a background fetch.
pub struct EndfFetchResult {
    pub z: u32,
    pub a: u32,
    pub symbol: String,
    pub result: Result<ResonanceData, String>,
}

/// Input mode: which type of data is being loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InputMode {
    /// Sample TIFF stack + Open beam TIFF stack + Spectrum file.
    TiffPair,
    /// HDF5/NeXus file with pre-histogrammed counts.
    Hdf5Histogram,
    /// HDF5/NeXus file with raw neutron events (histogrammed on load).
    Hdf5Event,
}

/// Data source for the normalize-preview spectrum plot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumDataSource {
    /// Average over all pixels in the full image.
    FullImage,
    /// Average over the selected ROI.
    RoiAverage,
}

/// X-axis unit for the spectrum plot in the normalize preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumAxis {
    EnergyEv,
    TofMicroseconds,
}

/// A single provenance event in the session audit trail.
#[derive(Debug, Clone)]
pub struct ProvenanceEvent {
    pub timestamp: std::time::SystemTime,
    pub kind: ProvenanceEventKind,
    pub message: String,
}

/// Classification of provenance events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceEventKind {
    DataLoaded,
    ConfigChanged,
    Normalized,
    AnalysisRun,
    Exported,
    ProjectSaved,
    ProjectLoaded,
}

impl ProvenanceEvent {
    /// Format the timestamp as "YYYY-MM-DD HH:MM:SS UTC".
    pub fn formatted_timestamp(&self) -> String {
        self.timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| {
                let secs = d.as_secs();
                // Days since epoch
                let days = secs / 86400;
                let time_secs = secs % 86400;
                let h = time_secs / 3600;
                let m = (time_secs / 60) % 60;
                let s = time_secs % 60;
                // Convert days to Y-M-D (civil calendar from epoch 1970-01-01)
                let (y, mo, day) = days_to_civil(days);
                format!("{y:04}-{mo:02}-{day:02} {h:02}:{m:02}:{s:02} UTC")
            })
            .unwrap_or_else(|_| "????-??-?? ??:??:?? UTC".to_string())
    }
}

/// Convert days since Unix epoch to (year, month, day).
///
/// Algorithm from Howard Hinnant's `chrono`-compatible civil date conversion.
fn days_to_civil(days: u64) -> (i32, u32, u32) {
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

/// Colormap for image rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Viridis,
}

/// Data mode for project file save.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaveDataMode {
    #[default]
    Linked,
    Embedded,
}

/// Target context for the periodic table modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicTableTarget {
    Configure,
    ConfigureGroup,
}

/// Main application state.
pub struct AppState {
    // -- Data loading --
    pub input_mode: InputMode,
    pub sample_path: Option<PathBuf>,
    pub open_beam_path: Option<PathBuf>,
    pub sample_data: Option<Arc<Array3<f64>>>,
    pub open_beam_data: Option<Arc<Array3<f64>>>,
    /// Effective pixel mask applied by the analysis:
    /// `file_dead_pixels ∪ latest detection`.  Recomputed FROM SCRATCH on
    /// every normalization via [`AppState::set_detected_dead_pixels`] —
    /// never unioned with its own previous value, so detections can not
    /// accumulate across open-beam swaps / re-normalizations.
    pub dead_pixels: Option<Array2<bool>>,
    /// Declared-mask provenance: the pixel mask carried by an input
    /// artifact — the HDF5 file's dead-pixel dataset (set where the sample
    /// is loaded: `guided::load::load_hdf5_histogram`, `guided::bin`) or a saved
    /// project's `/intermediate/dead_pixels` (set on project restore).
    /// Lives exactly as long as the loaded sample: cleared by
    /// [`AppState::invalidate_results`] and the file-pick handlers.  This
    /// is the only mask component that persists across re-normalizations;
    /// detected masks are always recomputed
    /// ([`AppState::set_detected_dead_pixels`] is the single site that
    /// combines the two).
    pub file_dead_pixels: Option<Array2<bool>>,
    /// Detected-mask component: the dead ∪ hot mask from
    /// the most recent detection run, exactly as
    /// [`AppState::set_detected_dead_pixels`] received it.  Kept separate
    /// from the effective mask so project SAVE can persist it as its own
    /// session-scoped, versioned dataset
    /// (`/intermediate/detected_dead_pixels`): a restored project may
    /// carry embedded normalized data WITHOUT raw stacks — detection
    /// never re-runs there, and without this component a refit would
    /// silently lose the dead/hot exclusions active at save time.
    /// Restore rebuilds the effective mask as declared ∪ this component.
    /// Cleared wherever the mask pair is cleared/replaced (load sites,
    /// [`AppState::invalidate_results`]).
    pub detected_dead_pixels: Option<Array2<bool>>,

    // -- Spectrum file --
    pub spectrum_path: Option<PathBuf>,
    pub spectrum_values: Option<Arc<Vec<f64>>>,
    pub spectrum_unit: SpectrumUnit,
    pub spectrum_kind: SpectrumValueKind,

    // -- Beamline parameters --
    pub beamline: BeamlineParams,
    pub proton_charge_sample: f64,
    pub proton_charge_ob: f64,

    // -- Energy grid --
    pub energies: Option<Vec<f64>>,

    // -- Isotope selection --
    pub isotope_entries: Vec<IsotopeEntry>,
    pub isotope_groups: Vec<IsotopeGroupEntry>,
    pub endf_library: EndfLibrary,

    // -- Fitting --
    pub temperature_k: f64,
    /// Restrict the fit to `[min_eV, max_eV]` (SAMMY EMIN/EMAX equivalent —
    /// INPut-file card set 2, manual Table VI A.1).
    /// `None` = full grid (default).  Both bounds must lie inside the
    /// loaded energy grid; the resolution-kernel margin (~5×FWHM beyond
    /// each boundary) is applied automatically in `build_fit_config`
    /// so resonances near the boundaries are correctly broadened.
    pub fit_energy_range: Option<(f64, f64)>,

    // -- Uncertainty provenance --
    /// True when per-bin uncertainty was estimated rather than measured
    /// from sample + open-beam counting statistics. When true, chi-squared
    /// values are approximate and should be displayed with a warning.
    pub uncertainty_is_estimated: bool,

    // -- Background normalization --
    /// KL background: SAMMY 4-term wrapper (joint-Poisson compatible).
    /// Model: Anorm * T_inner(E) + BackA + BackB/sqrt(E) + BackC*sqrt(E)
    pub kl_background_enabled: bool,
    /// Proton-charge ratio c = Q_s / Q_ob for the counts-KL solver.
    /// Default 1.0 = caller PC-normalized the flux
    /// upstream.  For raw-count VENUS data, set to the actual
    /// Q_sample/Q_open_beam ratio (typically ~5–6).
    pub kl_c_ratio: f64,
    /// Retained project setting; the counts engine has no Nelder-Mead polish.
    pub kl_enable_polish_override: Option<bool>,
    /// Bounded multiplicative baseline:
    /// B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref), applied OUTERMOST.
    /// When combined with a background, Anorm is held fixed (b0/Anorm are
    /// degenerate normalizations) — `build_fit_config` wires that.
    pub baseline_enabled: bool,

    // -- Pixel / ROI selection --
    pub selected_pixel: Option<(usize, usize)>,
    pub rois: Vec<RoiSelection>,
    pub selected_roi: Option<usize>,

    /// Toggle: show provenance history popup window.
    pub show_history_window: bool,

    // -- Pipeline / wizard --
    pub fitting_type: Option<FittingType>,
    pub data_type: Option<DataType>,
    pub pipeline: Vec<PipelineEntry>,
    pub wizard_step: u8,

    // -- UI state --
    pub ui_mode: UiMode,
    pub guided_step: GuidedStep,
    pub theme_preference: ThemePreference,
    /// Cached resolved dark-mode boolean; used to skip redundant `apply_theme`.
    pub last_applied_dark_mode: Option<bool>,
    /// Whether the guided-mode sidebar is collapsed (icon-only mode).
    pub sidebar_collapsed: bool,
    pub status_message: String,
    pub is_fitting: bool,
    pub is_fetching_endf: bool,
    /// Cloned egui context for background threads to request repaints.
    pub egui_ctx: Option<egui::Context>,
    /// Poll-based file-dialog service; picks are routed by
    /// `crate::file_dialog::dispatch_results` each frame.
    pub file_dialogs: crate::file_dialog::FileDialogs,
    /// Dismissible banner shown when native file dialogs are unavailable
    /// or failed (portal/zenity chain). Set from the startup probe and
    /// the log-bridge latch.
    pub native_dialog_warning: Option<String>,

    /// Prevents auto-load retry after a loading failure; cleared when file paths change.
    pub load_error: bool,

    // -- Rebinning --
    /// Integer rebin factor (1 = no rebinning).
    pub rebin_factor: usize,
    /// True after rebinning has been applied to sample_data/open_beam_data.
    pub rebin_applied: bool,

    // -- HDF5/NeXus --
    pub hdf5_path: Option<PathBuf>,
    /// Open beam NeXus file for HDF5 histogram/event modes.
    /// When set, enables counts-domain fitting (T = sample/OB).
    pub hdf5_ob_path: Option<PathBuf>,
    pub nexus_metadata: Option<NexusMetadata>,
    /// Inline error message from NeXus probe (shown in red below metadata).
    pub nexus_probe_error: Option<String>,
    pub event_n_bins: usize,
    pub event_tof_min_us: f64,
    pub event_tof_max_us: f64,
    pub event_height: usize,
    pub event_width: usize,

    // -- Normalize preview --
    pub normalize_spectrum_source: SpectrumDataSource,
    pub normalize_spectrum_axis: SpectrumAxis,
    pub tof_slice_index: usize,
    pub show_resonance_dips: bool,

    // -- Analyze viewer --
    pub analyze_spectrum_axis: SpectrumAxis,
    pub analyze_tof_slice_index: usize,

    // -- Counts fit --
    /// The Ikeda-Carpenter pulse numbers used as the known calibration.
    pub pulse: crate::counts::PulseParams,
    /// Patch side of the counts map, in pixels.
    pub map_binning: usize,
    /// The last single counts fit.
    pub pixel_fit: Option<crate::counts::PixelFitSummary>,
    /// The last counts-map fit.
    pub map_fit: Option<crate::counts::MapFitSummary>,

    // -- Background task receivers and cancellation --
    pub pending_endf: Option<mpsc::Receiver<EndfFetchResult>>,
    pub pending_fit: Option<mpsc::Receiver<crate::counts::FitOutcome>>,
    pub cancel_token: Arc<AtomicBool>,

    // -- Preview image texture --
    pub preview_image: Option<Array2<f64>>,

    // -- Isotope density editor --
    pub editing_isotope_density: Option<usize>,

    // -- Periodic Table modal --
    pub periodic_table_open: bool,
    pub periodic_table_target: PeriodicTableTarget,
    pub periodic_table_selected_z: Option<u32>,
    pub periodic_table_selected_isotopes: Vec<(u32, u32)>,
    pub periodic_table_density: f64,
    pub periodic_table_library: Option<EndfLibrary>,
    pub periodic_table_custom_z: u32,
    pub periodic_table_custom_a: u32,

    // -- HDF5 tree browser --
    pub hdf5_tree: Option<Vec<Hdf5TreeEntry>>,

    // -- Provenance --
    pub provenance_log: Vec<ProvenanceEvent>,

    // -- Project file --
    /// Path of the last saved/loaded project file (.nrd.h5).
    pub project_file_path: Option<PathBuf>,
    /// Whether the save-mode chooser modal is open.
    pub show_save_modal: bool,
    /// Selected data mode for the next save operation.
    pub save_data_mode: SaveDataMode,
    /// Data mode used in the last explicit save (for Cmd+S re-save).
    pub last_save_mode: SaveDataMode,
    /// Whether a background save is in progress.
    pub is_saving: bool,
    /// Channel to receive the save result from the background thread.
    pub pending_save: Option<mpsc::Receiver<Result<(PathBuf, SaveDataMode), String>>>,
    /// Join handle for the background save thread (used to block on shutdown).
    pub save_join_handle: Option<std::thread::JoinHandle<()>>,

    // -- Session persistence --
    /// Cached session from a previous run (loaded at startup, cleared on use).
    pub cached_session: Option<SessionCache>,
}

/// ENDF fetch lifecycle for an isotope entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EndfStatus {
    #[default]
    Pending,
    Fetching,
    Loaded,
    Failed,
}

/// An isotope the user wants to include in the fit.
pub struct IsotopeEntry {
    pub z: u32,
    pub a: u32,
    pub symbol: String,
    pub initial_density: f64,
    pub resonance_data: Option<ResonanceData>,
    pub enabled: bool,
    pub endf_status: EndfStatus,
}

/// A group of isotopes sharing one density parameter (e.g., all natural W).
pub struct IsotopeGroupEntry {
    pub z: u32,
    pub name: String,
    pub members: Vec<GroupMemberState>,
    pub initial_density: f64,
    pub enabled: bool,
}

/// State for a single member of an isotope group.
pub struct GroupMemberState {
    pub a: u32,
    pub symbol: String,
    pub ratio: f64,
    pub resonance_data: Option<ResonanceData>,
    pub endf_status: EndfStatus,
}

impl IsotopeGroupEntry {
    /// Derived ENDF status: Loaded if all members Loaded, Failed if any Failed, etc.
    pub fn overall_status(&self) -> EndfStatus {
        if self.members.is_empty() {
            return EndfStatus::Pending;
        }
        if self
            .members
            .iter()
            .all(|m| m.endf_status == EndfStatus::Loaded)
        {
            EndfStatus::Loaded
        } else if self
            .members
            .iter()
            .any(|m| m.endf_status == EndfStatus::Failed)
        {
            EndfStatus::Failed
        } else if self
            .members
            .iter()
            .any(|m| m.endf_status == EndfStatus::Fetching)
        {
            EndfStatus::Fetching
        } else {
            EndfStatus::Pending
        }
    }
}

/// ROI rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct RoiSelection {
    pub y_start: usize,
    pub y_end: usize,
    pub x_start: usize,
    pub x_end: usize,
}

/// Application UI mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Guided,
    Studio,
}

/// Fitting type chosen in the wizard (Q1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FittingType {
    Spatial,
    Single,
}

/// Data type chosen in the wizard (Q2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DataType {
    Events,
    PreNormalized,
}

/// A single entry in the dynamic pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PipelineEntry {
    pub step: GuidedStep,
    pub optional: bool,
}

/// Step within the Guided workflow.
///
/// The discriminant values define pipeline ordering for dirty tracking:
/// earlier stages have lower values, so `min()` on the discriminant
/// gives "earliest dirty stage" semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GuidedStep {
    Landing = 0,
    Wizard = 1,
    Configure = 2,
    Load = 3,
    Bin = 4,
    Rebin = 5,
    Normalize = 6,
    Analyze = 7,
    Results = 8,
}

impl GuidedStep {
    /// Human-readable label for display.
    pub fn label(self) -> &'static str {
        match self {
            Self::Landing => "Home",
            Self::Wizard => "Setup",
            Self::Configure => "Configure",
            Self::Load => "Load",
            Self::Bin => "Bin",
            Self::Rebin => "Rebin",
            Self::Normalize => "Normalize",
            Self::Analyze => "Analyze",
            Self::Results => "Results",
        }
    }

    /// Compute the pipeline steps for the given fitting type and data type.
    pub fn pipeline(fitting: FittingType, data: DataType) -> Vec<PipelineEntry> {
        let req = |s| PipelineEntry {
            step: s,
            optional: false,
        };
        let opt = |s| PipelineEntry {
            step: s,
            optional: true,
        };
        match (fitting, data) {
            (FittingType::Spatial, DataType::Events) => vec![
                req(Self::Configure),
                req(Self::Load),
                req(Self::Bin),
                req(Self::Normalize),
                req(Self::Analyze),
                req(Self::Results),
            ],
            (FittingType::Single, DataType::Events) => vec![
                req(Self::Configure),
                req(Self::Load),
                req(Self::Bin),
                req(Self::Normalize),
                req(Self::Analyze),
                req(Self::Results),
            ],
            (FittingType::Spatial, DataType::PreNormalized) => vec![
                req(Self::Configure),
                req(Self::Load),
                opt(Self::Rebin),
                req(Self::Normalize),
                req(Self::Analyze),
                req(Self::Results),
            ],
            (FittingType::Single, DataType::PreNormalized) => vec![
                req(Self::Configure),
                req(Self::Load),
                opt(Self::Rebin),
                req(Self::Normalize),
                req(Self::Analyze),
                req(Self::Results),
            ],
        }
    }
}

/// Theme preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemePreference {
    Auto,
    Light,
    Dark,
}

impl AppState {
    /// Cancel any in-flight background tasks.
    /// Signals the cancellation token so threads exit early, drops receivers,
    /// and issues a fresh token for future tasks.
    pub fn cancel_pending_tasks(&mut self) {
        // Signal existing threads to stop
        self.cancel_token.store(true, Ordering::Relaxed);
        // Replace with a fresh token for future tasks
        self.cancel_token = Arc::new(AtomicBool::new(false));
        self.pending_endf = None;
        self.pending_fit = None;
        self.is_fitting = false;
        self.is_fetching_endf = false;
        // Note: is_saving / pending_save / save_join_handle are NOT cleared here.
        // Save cannot be safely cancelled — the background thread must finish its
        // HDF5 write. poll_pending_tasks handles cleanup when it completes.
        // Reset any Fetching entries back to Pending (cancellation interrupted them)
        for e in &mut self.isotope_entries {
            if e.endf_status == EndfStatus::Fetching {
                e.endf_status = EndfStatus::Pending;
            }
        }
        for g in &mut self.isotope_groups {
            for m in &mut g.members {
                if m.endf_status == EndfStatus::Fetching {
                    m.endf_status = EndfStatus::Pending;
                }
            }
        }
    }

    /// Drop pending background work that is downstream of a source or
    /// configuration change.
    pub fn invalidate_fit_results(&mut self) {
        self.cancel_pending_tasks();
    }

    /// Invalidate analysis outputs after a fit-control change.
    pub fn invalidate_analysis_outputs(&mut self) {
        self.invalidate_fit_results();
    }

    /// Recompute the effective pixel mask FROM SCRATCH:
    /// `dead_pixels = file/persisted-declared ∪ freshly detected`.
    ///
    /// The single site where normalization installs a detected mask — the
    /// uniform semantics across all paths: detected masks are always
    /// recomputed, only the declared component (`file_dead_pixels`)
    /// persists.  Never unions with the previous `dead_pixels`, which may
    /// hold a detection from an earlier run (e.g. before an open-beam
    /// swap) — unioning would accumulate stale flags monotonically.
    ///
    /// `detected == None` means detection failed or was skipped: the
    /// effective mask falls back to the declared component alone; a
    /// previous detection is never kept (it may be stale for the current
    /// open beam).  The dimension-mismatch arm is **defensive**: every
    /// load site installs the declared mask from the same artifact the
    /// data comes from, so in the current wiring the two components
    /// always agree in shape.  It is reachable in principle through
    /// state drift this method cannot rule out — e.g. a hand-edited or
    /// corrupt project file whose declared mask was saved from a
    /// different detector or ROI-cropped geometry than its own detected
    /// mask / embedded data.  A declared mask that cannot apply to the
    /// data is dropped for this recomputation (detected-only mask), and
    /// the drop is RETURNED as a notice the caller must surface — masking
    /// decisions stay observable, never silent.
    ///
    /// The notice is returned rather than logged here because one
    /// caller — project restore (`project::state_from_snapshot`, step
    /// 15b) — runs BEFORE the snapshot's provenance history replaces
    /// the session log wholesale (step 17), which would erase an entry
    /// appended by this method; a `#[must_use]` return value cannot be
    /// silently erased, and forces every caller to route the drop into
    /// its own provenance/status surface (immediately at the
    /// normalization sites, deferred past the log replacement on
    /// restore).
    #[must_use = "declared-mask-drop notice: surface it (provenance/status), never drop it"]
    pub fn set_detected_dead_pixels(&mut self, detected: Option<Array2<bool>>) -> Option<String> {
        // Keep the raw detected component: project SAVE
        // persists it as /intermediate/detected_dead_pixels so a restore
        // without raw stacks can still rebuild the effective mask.
        self.detected_dead_pixels = detected.clone();
        let mut notice = None;
        self.dead_pixels = match (self.file_dead_pixels.clone(), detected) {
            (Some(mut declared), Some(det)) if declared.dim() == det.dim() => {
                ndarray::Zip::from(&mut declared)
                    .and(&det)
                    .for_each(|m, &d| *m = *m || d);
                Some(declared)
            }
            // Defensive arm (see rustdoc): a declared mask of a different
            // geometry cannot apply — detected only, drop surfaced via
            // the returned notice.
            (Some(declared), Some(det)) => {
                notice = Some(format!(
                    "Declared pixel mask {:?} does not match the data \
                     geometry {:?}; declared mask ignored for this \
                     recomputation — detected-only mask applied",
                    declared.dim(),
                    det.dim()
                ));
                Some(det)
            }
            (Some(declared), None) => Some(declared),
            (None, det) => det,
        };
        notice
    }

    /// Clear pixel selection, ROI, results, normalization, and cancel pending tasks.
    /// Called when the underlying data changes.
    pub fn invalidate_results(&mut self) {
        self.cancel_pending_tasks();
        self.selected_pixel = None;
        self.rois.clear();
        self.selected_roi = None;
        self.preview_image = None;
        self.energies = None;
        self.dead_pixels = None;
        // The declared mask belongs to the invalidated data; every caller
        // of invalidate_results either replaces the sample or triggers a
        // reload, and the load sites reinstall the file-declared mask.
        self.file_dead_pixels = None;
        // The detected component belongs to the invalidated data too.
        self.detected_dead_pixels = None;
        self.spectrum_values = None;
        self.rebin_applied = false;
        self.rebin_factor = 1;
        self.analyze_tof_slice_index = 0;
    }

    /// Compute the bounding box of all ROIs, or `None` if no ROIs exist.
    pub fn bounding_roi(&self) -> Option<RoiSelection> {
        if self.rois.is_empty() {
            return None;
        }
        let mut y_start = usize::MAX;
        let mut y_end = 0;
        let mut x_start = usize::MAX;
        let mut x_end = 0;
        for r in &self.rois {
            y_start = y_start.min(r.y_start);
            y_end = y_end.max(r.y_end);
            x_start = x_start.min(r.x_start);
            x_end = x_end.max(r.x_end);
        }
        Some(RoiSelection {
            y_start,
            y_end,
            x_start,
            x_end,
        })
    }

    /// Append a provenance event to the session audit trail.
    pub fn log_provenance(&mut self, kind: ProvenanceEventKind, message: impl Into<String>) {
        self.provenance_log.push(ProvenanceEvent {
            timestamp: std::time::SystemTime::now(),
            kind,
            message: message.into(),
        });
    }

    /// Index of the current step in the pipeline, or `None` if not a pipeline step.
    pub fn pipeline_index(&self) -> Option<usize> {
        self.pipeline
            .iter()
            .position(|e| e.step == self.guided_step)
    }

    /// 1-based display number for a pipeline step (skipping optional steps).
    /// Returns `None` for optional steps (displayed as "—").
    pub fn step_display_number(&self, step: GuidedStep) -> Option<u8> {
        let mut n = 0u8;
        for entry in &self.pipeline {
            if !entry.optional {
                n += 1;
            }
            if entry.step == step {
                return if entry.optional { None } else { Some(n) };
            }
        }
        None
    }

    /// Navigate to the next step in the pipeline.
    pub fn nav_next(&mut self) {
        if let Some(idx) = self.pipeline_index()
            && idx + 1 < self.pipeline.len()
        {
            self.guided_step = self.pipeline[idx + 1].step;
            self.status_message = String::new();
        }
    }

    /// Navigate to the previous step in the pipeline.
    /// From the first pipeline step, returns to the Wizard.
    pub fn nav_prev(&mut self) {
        if let Some(idx) = self.pipeline_index() {
            if idx > 0 {
                self.guided_step = self.pipeline[idx - 1].step;
            } else {
                self.guided_step = GuidedStep::Wizard;
                self.wizard_step = 2; // return to Confirm page
            }
            self.status_message = String::new();
        }
    }

    /// Recompute the pipeline from the current fitting_type and data_type.
    pub fn rebuild_pipeline(&mut self) {
        if let (Some(ft), Some(dt)) = (self.fitting_type, self.data_type) {
            self.pipeline = GuidedStep::pipeline(ft, dt);
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            input_mode: InputMode::TiffPair,
            sample_path: None,
            open_beam_path: None,
            sample_data: None,
            open_beam_data: None,
            dead_pixels: None,
            file_dead_pixels: None,
            detected_dead_pixels: None,
            load_error: false,
            rebin_factor: 1,
            rebin_applied: false,

            spectrum_path: None,
            spectrum_values: None,
            spectrum_unit: SpectrumUnit::TofMicroseconds,
            spectrum_kind: SpectrumValueKind::BinEdges,

            beamline: BeamlineParams::default(),
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,

            energies: None,

            isotope_entries: Vec::new(),
            isotope_groups: Vec::new(),
            endf_library: EndfLibrary::EndfB8_1,

            temperature_k: 296.0,
            fit_energy_range: None,
            uncertainty_is_estimated: false,
            kl_background_enabled: false,
            baseline_enabled: false,
            kl_c_ratio: 1.0,
            kl_enable_polish_override: None,

            selected_pixel: None,
            rois: Vec::new(),
            selected_roi: None,
            show_history_window: false,

            fitting_type: None,
            data_type: None,
            pipeline: Vec::new(),
            wizard_step: 0,

            ui_mode: UiMode::Guided,
            guided_step: GuidedStep::Landing,
            theme_preference: ThemePreference::Auto,
            last_applied_dark_mode: None,
            sidebar_collapsed: false,
            status_message: "Ready".into(),
            is_fitting: false,
            is_fetching_endf: false,
            egui_ctx: None,
            file_dialogs: Default::default(),
            native_dialog_warning: None,

            hdf5_path: None,
            hdf5_ob_path: None,
            nexus_metadata: None,
            nexus_probe_error: None,
            event_n_bins: 500,
            event_tof_min_us: 1000.0,
            event_tof_max_us: 20000.0,
            event_height: 512,
            event_width: 512,

            normalize_spectrum_source: SpectrumDataSource::FullImage,
            normalize_spectrum_axis: SpectrumAxis::EnergyEv,
            tof_slice_index: 0,
            show_resonance_dips: false,

            analyze_spectrum_axis: SpectrumAxis::EnergyEv,
            analyze_tof_slice_index: 0,

            pulse: crate::counts::PulseParams::default(),
            map_binning: 4,
            pixel_fit: None,
            map_fit: None,

            pending_endf: None,
            pending_fit: None,
            cancel_token: Arc::new(AtomicBool::new(false)),

            preview_image: None,

            editing_isotope_density: None,

            periodic_table_open: false,
            periodic_table_target: PeriodicTableTarget::Configure,
            periodic_table_selected_z: None,
            periodic_table_selected_isotopes: Vec::new(),
            periodic_table_density: 0.001,
            periodic_table_library: None,
            periodic_table_custom_z: 94,
            periodic_table_custom_a: 239,

            hdf5_tree: None,

            provenance_log: Vec::new(),

            project_file_path: None,
            show_save_modal: false,
            save_data_mode: SaveDataMode::Linked,
            last_save_mode: SaveDataMode::Linked,
            is_saving: false,
            pending_save: None,
            save_join_handle: None,

            cached_session: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accumulation pin: re-detection REPLACES the previous
    /// detection instead of unioning with it — only the file-declared
    /// component persists across runs (e.g. open-beam swaps).
    #[test]
    fn test_set_detected_dead_pixels_no_accumulation_across_runs() {
        let mut declared = Array2::from_elem((3, 3), false);
        declared[[0, 0]] = true;
        let mut state = AppState {
            file_dead_pixels: Some(declared),
            ..AppState::default()
        };

        // First normalization detects (1, 1).
        let mut det1 = Array2::from_elem((3, 3), false);
        det1[[1, 1]] = true;
        assert!(state.set_detected_dead_pixels(Some(det1)).is_none());
        let mask = state.dead_pixels.as_ref().unwrap();
        assert!(mask[[0, 0]] && mask[[1, 1]]);
        assert_eq!(mask.iter().filter(|&&m| m).count(), 2);

        // OB swap: the fresh detection flags (2, 2) instead.  (1, 1) is
        // stale and must be GONE; (0, 0) is file-declared and persists.
        let mut det2 = Array2::from_elem((3, 3), false);
        det2[[2, 2]] = true;
        assert!(state.set_detected_dead_pixels(Some(det2)).is_none());
        let mask = state.dead_pixels.as_ref().unwrap();
        assert!(mask[[0, 0]], "file-declared flag must persist");
        assert!(!mask[[1, 1]], "stale detection must not accumulate");
        assert!(mask[[2, 2]], "fresh detection must be present");
        assert_eq!(mask.iter().filter(|&&m| m).count(), 2);
    }

    /// Detection failure falls back to the declared component alone —
    /// a previous detection is never kept.
    #[test]
    fn test_set_detected_dead_pixels_failure_keeps_declared_only() {
        let mut declared = Array2::from_elem((2, 2), false);
        declared[[0, 1]] = true;
        let mut state = AppState {
            file_dead_pixels: Some(declared.clone()),
            ..AppState::default()
        };

        let mut det = Array2::from_elem((2, 2), false);
        det[[1, 0]] = true;
        assert!(state.set_detected_dead_pixels(Some(det)).is_none());
        assert_eq!(
            state
                .dead_pixels
                .as_ref()
                .unwrap()
                .iter()
                .filter(|&&m| m)
                .count(),
            2
        );

        assert!(state.set_detected_dead_pixels(None).is_none());
        assert_eq!(state.dead_pixels.as_ref().unwrap(), &declared);

        // No declared mask either: the effective mask is absent, not stale.
        state.file_dead_pixels = None;
        assert!(state.set_detected_dead_pixels(None).is_none());
        assert!(state.dead_pixels.is_none());
    }

    /// The DEFENSIVE dimension-mismatch arm (reachable via a
    /// hand-edited/corrupt project whose declared mask was saved from a
    /// different detector or ROI-cropped geometry): the inapplicable
    /// declared mask is dropped for the recomputation (detected-only
    /// mask) and the drop is RETURNED as a must-surface notice, never
    /// silent.  The setter itself
    /// does not log to provenance — project restore replaces the
    /// provenance log wholesale AFTER calling it (step 17), which would
    /// erase such an entry; see
    /// `test_restore_mismatched_mask_pair_drop_notice_survives_provenance_replacement`
    /// (project.rs) for the end-to-end restore-path property this
    /// direct-call test cannot see.
    #[test]
    fn test_set_detected_dead_pixels_dim_mismatch_uses_detected_and_returns_notice() {
        let mut state = AppState {
            file_dead_pixels: Some(Array2::from_elem((4, 4), true)),
            ..AppState::default()
        };
        let provenance_len_before = state.provenance_log.len();

        let mut det = Array2::from_elem((2, 2), false);
        det[[0, 0]] = true;
        let notice = state.set_detected_dead_pixels(Some(det.clone()));
        assert_eq!(state.dead_pixels.as_ref().unwrap(), &det);
        // The drop is observable: the returned notice names both
        // geometries...
        let msg = notice.expect("dim mismatch must return a drop notice");
        assert!(msg.contains("(4, 4)") && msg.contains("(2, 2)"), "{msg}");
        // ...and surfacing is the CALLER's job — the setter appends
        // nothing itself (a caller-side log would be erased by restore's
        // provenance replacement; the return value cannot be).
        assert_eq!(state.provenance_log.len(), provenance_len_before);

        // The matching-geometry path returns no notice — no drop.
        state.file_dead_pixels = Some(Array2::from_elem((2, 2), false));
        assert!(state.set_detected_dead_pixels(Some(det)).is_none());
    }
}
