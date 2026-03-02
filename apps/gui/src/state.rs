//! Application state shared across all GUI panels.

use ndarray::{Array2, Array3};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Result of a background ENDF fetch for a single isotope.
pub struct EndfFetchResult {
    pub index: usize,
    pub symbol: String,
    pub result: Result<ResonanceData, String>,
}

/// Input mode: which type of data is being loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Target context for the periodic table modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicTableTarget {
    Configure,
    ForwardModel,
    DetectMatrix,
    DetectTrace,
}

/// A trace isotope entry for detectability analysis.
pub struct DetectTraceEntry {
    pub z: u32,
    pub a: u32,
    pub symbol: String,
    pub concentration_ppm: f64,
    pub resonance_data: Option<ResonanceData>,
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

    // -- Fitting --
    pub temperature_k: f64,
    pub lm_config: LmConfig,
    pub solver_method: SolverMethod,
    pub fit_temperature: bool,
    pub show_advanced_solver: bool,

    // -- Pixel / ROI selection --
    pub selected_pixel: Option<(usize, usize)>,
    pub roi: Option<RoiSelection>,

    // -- Results --
    pub pixel_fit_result: Option<SpectrumFitResult>,
    pub spatial_result: Option<SpatialResult>,

    // -- UI state --
    pub ui_mode: UiMode,
    pub guided_step: GuidedStep,
    pub theme_preference: ThemePreference,
    pub active_tab: Tab,
    pub status_message: String,
    pub is_fitting: bool,
    pub is_fetching_endf: bool,

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

    // -- Detectability tool --
    pub detect_matrix: Option<IsotopeEntry>,
    pub detect_matrix_density: f64,
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

    // -- Periodic Table modal --
    pub periodic_table_open: bool,
    pub periodic_table_target: PeriodicTableTarget,
    pub periodic_table_selected_z: Option<u32>,

    // -- HDF5 tree browser --
    pub hdf5_tree: Option<Vec<Hdf5TreeEntry>>,
}

/// An isotope the user wants to include in the fit.
pub struct IsotopeEntry {
    pub z: u32,
    pub a: u32,
    pub symbol: String,
    pub initial_density: f64,
    pub resonance_data: Option<ResonanceData>,
    pub enabled: bool,
}

/// ROI rectangle in pixel coordinates.
#[derive(Clone, Copy)]
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

/// Step within the Guided workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidedStep {
    Load,
    Configure,
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
            Self::Load => "Load",
            Self::Configure => "Configure",
            Self::Normalize => "Normalize",
            Self::Analyze => "Analyze",
            Self::Results => "Results",
            Self::ForwardModel => "Forward Model",
            Self::Detectability => "Detectability",
        }
    }

    /// 1-based step number (None for tool steps).
    pub fn number(self) -> Option<u8> {
        match self {
            Self::Load => Some(1),
            Self::Configure => Some(2),
            Self::Normalize => Some(3),
            Self::Analyze => Some(4),
            Self::Results => Some(5),
            Self::ForwardModel | Self::Detectability => None,
        }
    }

    /// Ordered list of the five main workflow steps.
    pub const WORKFLOW: [GuidedStep; 5] = [
        Self::Load,
        Self::Configure,
        Self::Normalize,
        Self::Analyze,
        Self::Results,
    ];
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
        self.is_fetching_endf = false;
        self.is_fetching_fm_endf = false;
        self.is_fetching_detect_endf = false;
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
        self.roi = None;
        self.pixel_fit_result = None;
        self.spatial_result = None;
        self.preview_image = None;
        self.energies = None;
        self.normalized = None;
        self.dead_pixels = None;
        self.spectrum_values = None;
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

            temperature_k: 296.0,
            lm_config: LmConfig::default(),
            solver_method: SolverMethod::LevenbergMarquardt,
            fit_temperature: false,
            show_advanced_solver: false,

            selected_pixel: None,
            roi: None,

            pixel_fit_result: None,
            spatial_result: None,

            ui_mode: UiMode::Guided,
            guided_step: GuidedStep::Load,
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

            detect_matrix: None,
            detect_matrix_density: 0.001,
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

            periodic_table_open: false,
            periodic_table_target: PeriodicTableTarget::Configure,
            periodic_table_selected_z: None,

            hdf5_tree: None,
        }
    }
}
