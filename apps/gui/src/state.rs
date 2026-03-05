//! Application state shared across all GUI panels.

use ndarray::{Array2, Array3};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use nereids_endf::resonance::ResonanceData;
use nereids_endf::retrieval::EndfLibrary;
use nereids_fitting::lm::LmConfig;
use nereids_io::nexus::{Hdf5TreeEntry, NexusMetadata};
use nereids_io::normalization::NormalizedData;
use nereids_io::spectrum::{SpectrumUnit, SpectrumValueKind};
use nereids_io::tof::BeamlineParams;
use nereids_pipeline::detectability::TraceDetectabilityReport;
use nereids_pipeline::pipeline::SpectrumFitResult;
use nereids_pipeline::spatial::SpatialResult;

/// Lightweight session cache for persistence across app restarts.
/// Only stores the subset of state needed to resume a previous pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionCache {
    pub fitting_type: Option<FittingType>,
    pub data_type: Option<DataType>,
    pub input_mode: InputMode,
    pub analysis_mode: AnalysisMode,
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
    /// Solver method.
    pub solver_method: CachedSolverMethod,
    /// Resolution broadening: "gaussian" or "tabulated".
    pub resolution_kind: String,
    /// Gaussian Δt (μs), only used when resolution_kind == "gaussian".
    pub resolution_delta_t_us: f64,
    /// Gaussian ΔL (m), only used when resolution_kind == "gaussian".
    pub resolution_delta_l_m: f64,
    /// Tabulated resolution file path (if applicable).
    pub resolution_path: Option<String>,
}

/// Serializable solver method for session cache.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CachedSolverMethod {
    LevenbergMarquardt,
    PoissonKL,
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
            analysis_mode: state.analysis_mode,
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
            solver_method: match state.solver_method {
                SolverMethod::LevenbergMarquardt => CachedSolverMethod::LevenbergMarquardt,
                SolverMethod::PoissonKL => CachedSolverMethod::PoissonKL,
            },
            resolution_kind: match &state.resolution_mode {
                ResolutionMode::Gaussian { .. } => "gaussian".to_string(),
                ResolutionMode::Tabulated { .. } => "tabulated".to_string(),
            },
            resolution_delta_t_us: match &state.resolution_mode {
                ResolutionMode::Gaussian { delta_t_us, .. } => *delta_t_us,
                _ => 0.0,
            },
            resolution_delta_l_m: match &state.resolution_mode {
                ResolutionMode::Gaussian { delta_l_m, .. } => *delta_l_m,
                _ => 0.0,
            },
            resolution_path: match &state.resolution_mode {
                ResolutionMode::Tabulated { path, .. } => Some(path.to_string_lossy().to_string()),
                _ => None,
            },
        })
    }

    /// Apply cached settings to app state (restores pipeline + config).
    pub fn apply_to(&self, state: &mut AppState) {
        state.fitting_type = self.fitting_type;
        state.data_type = self.data_type;
        state.input_mode = self.input_mode;
        state.analysis_mode = self.analysis_mode;
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

        // Restore library by matching label
        state.endf_library = match self.endf_library_name.as_str() {
            "ENDF/B-VIII.1" => nereids_endf::retrieval::EndfLibrary::EndfB8_1,
            "JEFF-3.3" => nereids_endf::retrieval::EndfLibrary::Jeff3_3,
            "JENDL-5" => nereids_endf::retrieval::EndfLibrary::Jendl5,
            _ => nereids_endf::retrieval::EndfLibrary::EndfB8_0,
        };

        // Restore solver method
        state.solver_method = match self.solver_method {
            CachedSolverMethod::LevenbergMarquardt => SolverMethod::LevenbergMarquardt,
            CachedSolverMethod::PoissonKL => SolverMethod::PoissonKL,
        };

        // Restore resolution mode
        state.resolution_mode = if self.resolution_kind == "tabulated" {
            if let Some(ref p) = self.resolution_path {
                ResolutionMode::Tabulated {
                    path: PathBuf::from(p),
                    data: None, // will need re-parse on first use
                    error: None,
                }
            } else {
                ResolutionMode::Gaussian {
                    delta_t_us: self.resolution_delta_t_us,
                    delta_l_m: self.resolution_delta_l_m,
                }
            }
        } else {
            ResolutionMode::Gaussian {
                delta_t_us: self.resolution_delta_t_us,
                delta_l_m: self.resolution_delta_l_m,
            }
        };

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
            Some(DataType::Transmission) => "Transmission",
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
pub struct EndfFetchResult {
    pub index: usize,
    pub symbol: String,
    pub result: Result<ResonanceData, String>,
}

/// Input mode: which type of data is being loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InputMode {
    /// Sample TIFF stack + Open beam TIFF stack + Spectrum file.
    TiffPair,
    /// Pre-normalized transmission TIFF + Spectrum file.
    TransmissionTiff,
    /// HDF5/NeXus file with pre-histogrammed counts.
    Hdf5Histogram,
    /// HDF5/NeXus file with raw neutron events (histogrammed on load).
    Hdf5Event,
}

/// Analysis mode — determines how fitting operates in the Analyze step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AnalysisMode {
    /// Fit every pixel independently (full spatial map).
    FullSpatialMap,
    /// Average ROI into one spectrum, fit once.
    RoiSingleSpectrum,
    /// Bin NxN pixels, fit the binned map.
    SpatialBinning(u8),
}

/// Solver method for fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMethod {
    LevenbergMarquardt,
    PoissonKL,
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

/// Prominent feedback from a fit operation, displayed in the Analyze controls.
#[derive(Debug, Clone)]
pub struct FitFeedback {
    /// True if fit converged.
    pub success: bool,
    /// Summary line (e.g. "Pixel (3,5) converged, chi2_r = 1.23").
    pub summary: String,
    /// Per-isotope densities: (symbol, density_atoms_per_barn).
    pub densities: Vec<(String, f64)>,
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

/// Available colormaps for density map rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Viridis,
    Inferno,
    Plasma,
    Grayscale,
}

impl Colormap {
    pub const ALL: [Colormap; 4] = [
        Colormap::Viridis,
        Colormap::Inferno,
        Colormap::Plasma,
        Colormap::Grayscale,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Viridis => "Viridis",
            Self::Inferno => "Inferno",
            Self::Plasma => "Plasma",
            Self::Grayscale => "Grayscale",
        }
    }
}

/// Export format for spatial mapping results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Tiff,
    Hdf5,
    Markdown,
}

impl ExportFormat {
    pub const ALL: [ExportFormat; 3] = [
        ExportFormat::Tiff,
        ExportFormat::Hdf5,
        ExportFormat::Markdown,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Tiff => "TIFF (f64)",
            Self::Hdf5 => "HDF5",
            Self::Markdown => "Markdown Report",
        }
    }
}

/// Per-tile display settings for density map rendering.
#[derive(Debug, Clone)]
pub struct TileDisplayState {
    pub colormap: Colormap,
    pub show_colorbar: bool,
}

impl Default for TileDisplayState {
    fn default() -> Self {
        Self {
            colormap: Colormap::Viridis,
            show_colorbar: false,
        }
    }
}

/// Target context for the periodic table modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicTableTarget {
    Configure,
    ForwardModel,
    DetectMatrix,
    DetectTrace,
}

/// Resolution broadening mode: parametric Gaussian or tabulated from file.
#[derive(Clone, Debug)]
pub enum ResolutionMode {
    /// Analytical Gaussian: Δt (μs) and ΔL (m).
    Gaussian { delta_t_us: f64, delta_l_m: f64 },
    /// Tabulated from a VENUS/FTS resolution file.
    Tabulated {
        path: PathBuf,
        data: Option<Arc<nereids_physics::resolution::TabulatedResolution>>,
        error: Option<String>,
    },
}

impl PartialEq for ResolutionMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Gaussian {
                    delta_t_us: dt1,
                    delta_l_m: dl1,
                },
                Self::Gaussian {
                    delta_t_us: dt2,
                    delta_l_m: dl2,
                },
            ) => dt1 == dt2 && dl1 == dl2,
            (
                Self::Tabulated {
                    path: p1,
                    data: d1,
                    error: e1,
                },
                Self::Tabulated {
                    path: p2,
                    data: d2,
                    error: e2,
                },
            ) => match (d1, d2) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => p1 == p2 && e1 == e2,
                _ => false,
            },
            _ => false,
        }
    }
}

impl Default for ResolutionMode {
    fn default() -> Self {
        Self::Gaussian {
            delta_t_us: 1.0,
            delta_l_m: 0.01,
        }
    }
}

/// A trace isotope entry for detectability analysis.
pub struct DetectTraceEntry {
    pub z: u32,
    pub a: u32,
    pub symbol: String,
    pub concentration_ppm: f64,
    pub resonance_data: Option<ResonanceData>,
    pub endf_status: EndfStatus,
}

/// Main application state.
pub struct AppState {
    // -- Data loading --
    pub input_mode: InputMode,
    pub sample_path: Option<PathBuf>,
    pub open_beam_path: Option<PathBuf>,
    pub sample_data: Option<Array3<f64>>,
    pub open_beam_data: Option<Array3<f64>>,
    pub normalized: Option<Arc<NormalizedData>>,
    pub dead_pixels: Option<Array2<bool>>,

    // -- Spectrum file --
    pub spectrum_path: Option<PathBuf>,
    pub spectrum_values: Option<Vec<f64>>,
    pub spectrum_unit: SpectrumUnit,
    pub spectrum_kind: SpectrumValueKind,

    // -- Beamline parameters --
    pub beamline: BeamlineParams,
    pub proton_charge_sample: f64,
    pub proton_charge_ob: f64,

    // -- Energy grid --
    pub energies: Option<Vec<f64>>,
    pub tof_min_us: f64,
    pub tof_max_us: f64,

    // -- Isotope selection --
    pub isotope_entries: Vec<IsotopeEntry>,
    pub endf_library: EndfLibrary,

    // -- Instrument Resolution --
    pub resolution_enabled: bool,
    pub resolution_mode: ResolutionMode,

    // -- Fitting --
    pub temperature_k: f64,
    pub lm_config: LmConfig,
    pub solver_method: SolverMethod,
    pub fit_temperature: bool,
    pub show_advanced_solver: bool,

    // -- Pixel / ROI selection --
    pub selected_pixel: Option<(usize, usize)>,
    pub rois: Vec<RoiSelection>,
    pub selected_roi: Option<usize>,

    /// Snapshot of ROIs at the time spatial_map was launched.
    /// Used to render density overlays only on fitted pixels.
    pub fitting_rois: Vec<RoiSelection>,
    /// Toggle: show density overlay on preview image in Results.
    pub show_density_overlay: bool,
    /// Toggle: show provenance history popup window.
    pub show_history_window: bool,

    // -- Results --
    pub pixel_fit_result: Option<SpectrumFitResult>,
    pub spatial_result: Option<SpatialResult>,
    /// Prominent feedback from last fit attempt (pixel or ROI).
    pub last_fit_feedback: Option<FitFeedback>,

    // -- Pipeline / wizard --
    pub fitting_type: Option<FittingType>,
    pub data_type: Option<DataType>,
    pub pipeline: Vec<PipelineEntry>,
    pub wizard_step: u8,

    // -- UI state --
    pub ui_mode: UiMode,
    pub guided_step: GuidedStep,
    pub theme_preference: ThemePreference,
    pub active_tab: Tab,
    pub status_message: String,
    pub is_fitting: bool,
    pub is_fetching_endf: bool,

    /// Prevents auto-load retry after a loading failure; cleared when file paths change.
    pub load_error: bool,

    // -- Rebinning --
    /// Integer rebin factor (1 = no rebinning).
    pub rebin_factor: usize,
    /// True after rebinning has been applied to sample_data/open_beam_data.
    pub rebin_applied: bool,

    // -- HDF5/NeXus --
    pub hdf5_path: Option<PathBuf>,
    pub nexus_metadata: Option<NexusMetadata>,
    pub event_n_bins: usize,
    pub event_tof_min_us: f64,
    pub event_tof_max_us: f64,
    pub event_height: usize,
    pub event_width: usize,

    // -- Normalize preview --
    pub analysis_mode: AnalysisMode,
    pub normalize_spectrum_source: SpectrumDataSource,
    pub normalize_spectrum_axis: SpectrumAxis,
    pub tof_slice_index: usize,
    pub show_resonance_dips: bool,

    // -- Analyze viewer --
    pub analyze_spectrum_axis: SpectrumAxis,
    pub analyze_tof_slice_index: usize,

    // -- Background task receivers and cancellation --
    pub pending_spatial: Option<mpsc::Receiver<Result<SpatialResult, String>>>,
    pub pending_endf: Option<mpsc::Receiver<EndfFetchResult>>,
    pub cancel_token: Arc<AtomicBool>,

    // -- Preview image texture --
    pub preview_image: Option<Array2<f64>>,
    pub map_display_isotope: usize,

    // -- Forward Model tool --
    pub fm_isotope_entries: Vec<IsotopeEntry>,
    pub fm_endf_library: EndfLibrary,
    pub pending_fm_endf: Option<mpsc::Receiver<EndfFetchResult>>,
    pub is_fetching_fm_endf: bool,
    pub fm_temperature_k: f64,
    pub fm_spectrum_axis: SpectrumAxis,
    pub fm_spectrum: Option<Vec<f64>>,
    pub fm_per_isotope_spectra: Vec<(String, Vec<f64>)>,
    pub fm_energies: Option<Vec<f64>>,
    pub fm_resolution_enabled: bool,
    pub fm_resolution_mode: ResolutionMode,

    // -- Detectability tool --
    pub detect_matrix_entries: Vec<IsotopeEntry>,
    pub detect_trace_entries: Vec<DetectTraceEntry>,
    pub detect_snr_threshold: f64,
    pub detect_i0: f64,
    pub detect_energy_min: f64,
    pub detect_energy_max: f64,
    pub detect_n_energy_points: usize,
    pub detect_results: Vec<(String, TraceDetectabilityReport)>,
    pub pending_detect_endf: Option<mpsc::Receiver<EndfFetchResult>>,
    pub is_fetching_detect_endf: bool,
    pub detect_endf_library: EndfLibrary,
    pub detect_temperature_k: f64,
    /// Number of matrix entries at the time the ENDF fetch was spawned.
    /// Used by `poll_pending_tasks` to dispatch results correctly.
    pub detect_n_matrix_at_fetch: usize,
    pub detect_resolution_enabled: bool,
    pub detect_resolution_mode: ResolutionMode,

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

    // -- Studio mode --
    pub studio_selected_tile: usize,
    pub studio_tool: StudioTool,
    pub studio_doc_tab: StudioDocTab,
    pub studio_dock_tab: usize,
    pub studio_show_dock: bool,
    pub studio_analysis_isotope: usize,

    // -- Progress --
    pub fitting_progress: Option<(usize, usize)>,
    pub fitting_progress_counter: Option<Arc<AtomicUsize>>,

    // -- Provenance --
    pub provenance_log: Vec<ProvenanceEvent>,

    // -- Per-tile display state (indexed same as density_maps + 1 for convergence) --
    pub tile_display: Vec<TileDisplayState>,

    // -- Export --
    pub export_format: ExportFormat,
    pub export_directory: Option<PathBuf>,
    pub export_status: Option<String>,

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

/// ROI rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct RoiSelection {
    pub y_start: usize,
    pub y_end: usize,
    pub x_start: usize,
    pub x_end: usize,
}

/// Active tab in the main view area.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Tab {
    Spectrum,
    Map,
}

/// Application UI mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Guided,
    Studio,
}

/// Studio interaction tool (toolbar selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StudioTool {
    #[default]
    Select,
    Roi,
    Probe,
    Zoom,
}

impl StudioTool {
    pub fn label(self) -> &'static str {
        match self {
            Self::Select => "Sel",
            Self::Roi => "ROI",
            Self::Probe => "Prb",
            Self::Zoom => "Zm",
        }
    }
}

/// Document tab in Studio mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StudioDocTab {
    #[default]
    Analysis,
    ForwardModel,
    Detectability,
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
    Transmission,
}

/// A single entry in the dynamic pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PipelineEntry {
    pub step: GuidedStep,
    pub optional: bool,
}

/// Step within the Guided workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidedStep {
    Landing,
    Wizard,
    Configure,
    Load,
    Bin,
    Rebin,
    Normalize,
    Analyze,
    Results,
    ForwardModel,
    Detectability,
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
            Self::ForwardModel => "Forward Model",
            Self::Detectability => "Detectability",
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
            (FittingType::Spatial, DataType::Transmission) => vec![
                req(Self::Configure),
                req(Self::Load),
                opt(Self::Rebin),
                req(Self::Analyze),
                req(Self::Results),
            ],
            (FittingType::Single, DataType::Transmission) => vec![
                req(Self::Configure),
                req(Self::Load),
                opt(Self::Rebin),
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
        self.pending_spatial = None;
        self.pending_endf = None;
        self.pending_fm_endf = None;
        self.pending_detect_endf = None;
        self.is_fitting = false;
        self.fitting_progress = None;
        self.fitting_progress_counter = None;
        self.is_fetching_endf = false;
        self.is_fetching_fm_endf = false;
        self.is_fetching_detect_endf = false;
        // Reset any Fetching entries back to Pending (cancellation interrupted them)
        for e in &mut self.isotope_entries {
            if e.endf_status == EndfStatus::Fetching {
                e.endf_status = EndfStatus::Pending;
            }
        }
        for e in &mut self.fm_isotope_entries {
            if e.endf_status == EndfStatus::Fetching {
                e.endf_status = EndfStatus::Pending;
            }
        }
        for e in &mut self.detect_matrix_entries {
            if e.endf_status == EndfStatus::Fetching {
                e.endf_status = EndfStatus::Pending;
            }
        }
        for e in &mut self.detect_trace_entries {
            if e.endf_status == EndfStatus::Fetching {
                e.endf_status = EndfStatus::Pending;
            }
        }
        // Clear stale FM spectrum caches
        self.fm_spectrum = None;
        self.fm_per_isotope_spectra.clear();
        // Clear stale detectability results
        self.detect_results.clear();
    }

    /// Clear pixel selection, ROI, results, normalization, and cancel pending tasks.
    /// Called when the underlying data changes.
    pub fn invalidate_results(&mut self) {
        self.cancel_pending_tasks();
        self.selected_pixel = None;
        self.rois.clear();
        self.selected_roi = None;
        self.pixel_fit_result = None;
        self.spatial_result = None;
        self.last_fit_feedback = None;
        self.fitting_rois.clear();
        self.preview_image = None;
        self.energies = None;
        self.normalized = None;
        self.dead_pixels = None;
        self.spectrum_values = None;
        self.tile_display.clear();
        self.studio_selected_tile = 0;
        self.export_status = None;
        self.rebin_applied = false;
        self.rebin_factor = 1;
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
        }
    }

    /// Recompute the pipeline from the current fitting_type and data_type.
    pub fn rebuild_pipeline(&mut self) {
        if let (Some(ft), Some(dt)) = (self.fitting_type, self.data_type) {
            self.pipeline = GuidedStep::pipeline(ft, dt);
        }
    }

    /// Ensure `tile_display` has enough entries for the current result.
    /// Call after spatial analysis completes.
    pub fn init_tile_display(&mut self, n_density_maps: usize) {
        // +1 for the convergence map tile
        let needed = n_density_maps + 1;
        self.tile_display
            .resize_with(needed, TileDisplayState::default);
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
            normalized: None,
            dead_pixels: None,
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
            tof_min_us: 1000.0,
            tof_max_us: 20000.0,

            isotope_entries: Vec::new(),
            endf_library: EndfLibrary::EndfB8_0,

            resolution_enabled: false,
            resolution_mode: ResolutionMode::default(),

            temperature_k: 296.0,
            lm_config: LmConfig::default(),
            solver_method: SolverMethod::LevenbergMarquardt,
            fit_temperature: false,
            show_advanced_solver: false,

            selected_pixel: None,
            rois: Vec::new(),
            selected_roi: None,
            fitting_rois: Vec::new(),
            show_density_overlay: true,
            show_history_window: false,

            pixel_fit_result: None,
            spatial_result: None,
            last_fit_feedback: None,

            fitting_type: None,
            data_type: None,
            pipeline: Vec::new(),
            wizard_step: 0,

            ui_mode: UiMode::Guided,
            guided_step: GuidedStep::Landing,
            theme_preference: ThemePreference::Auto,
            active_tab: Tab::Spectrum,
            status_message: "Ready".into(),
            is_fitting: false,
            is_fetching_endf: false,

            hdf5_path: None,
            nexus_metadata: None,
            event_n_bins: 500,
            event_tof_min_us: 1000.0,
            event_tof_max_us: 20000.0,
            event_height: 512,
            event_width: 512,

            analysis_mode: AnalysisMode::FullSpatialMap,
            normalize_spectrum_source: SpectrumDataSource::FullImage,
            normalize_spectrum_axis: SpectrumAxis::EnergyEv,
            tof_slice_index: 0,
            show_resonance_dips: false,

            analyze_spectrum_axis: SpectrumAxis::EnergyEv,
            analyze_tof_slice_index: 0,

            pending_spatial: None,
            pending_endf: None,
            cancel_token: Arc::new(AtomicBool::new(false)),

            preview_image: None,
            map_display_isotope: 0,

            fm_isotope_entries: Vec::new(),
            fm_endf_library: EndfLibrary::EndfB8_0,
            pending_fm_endf: None,
            is_fetching_fm_endf: false,
            fm_temperature_k: 296.0,
            fm_spectrum_axis: SpectrumAxis::EnergyEv,
            fm_spectrum: None,
            fm_per_isotope_spectra: Vec::new(),
            fm_energies: None,
            fm_resolution_enabled: false,
            fm_resolution_mode: ResolutionMode::default(),

            detect_matrix_entries: Vec::new(),
            detect_trace_entries: Vec::new(),
            detect_snr_threshold: 3.0,
            detect_i0: 10_000.0,
            detect_energy_min: 1.0,
            detect_energy_max: 100.0,
            detect_n_energy_points: 2000,
            detect_results: Vec::new(),
            pending_detect_endf: None,
            is_fetching_detect_endf: false,
            detect_endf_library: EndfLibrary::EndfB8_0,
            detect_temperature_k: 296.0,
            detect_n_matrix_at_fetch: 0,
            detect_resolution_enabled: false,
            detect_resolution_mode: ResolutionMode::default(),

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

            studio_selected_tile: 0,
            studio_tool: StudioTool::Select,
            studio_doc_tab: StudioDocTab::Analysis,
            studio_dock_tab: 0,
            studio_show_dock: true,
            studio_analysis_isotope: 0,
            fitting_progress: None,
            fitting_progress_counter: None,

            provenance_log: Vec::new(),
            tile_display: Vec::new(),
            export_format: ExportFormat::Tiff,
            export_directory: None,
            export_status: None,

            cached_session: None,
        }
    }
}
