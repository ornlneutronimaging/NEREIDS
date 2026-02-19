//! Application state shared across all GUI panels.

use ndarray::{Array2, Array3};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use nereids_endf::resonance::ResonanceData;
use nereids_endf::retrieval::EndfLibrary;
use nereids_fitting::lm::LmConfig;
use nereids_io::normalization::NormalizedData;
use nereids_io::tof::BeamlineParams;
use nereids_pipeline::pipeline::SpectrumFitResult;
use nereids_pipeline::spatial::SpatialResult;

/// Result of a background ENDF fetch for a single isotope.
pub struct EndfFetchResult {
    pub index: usize,
    pub symbol: String,
    pub result: Result<ResonanceData, String>,
}

/// Main application state.
pub struct AppState {
    // -- Data loading --
    pub sample_path: Option<PathBuf>,
    pub open_beam_path: Option<PathBuf>,
    pub sample_data: Option<Array3<f64>>,
    pub open_beam_data: Option<Array3<f64>>,
    pub normalized: Option<Arc<NormalizedData>>,
    pub dead_pixels: Option<Array2<bool>>,

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

    // -- Pixel / ROI selection --
    pub selected_pixel: Option<(usize, usize)>,
    pub roi: Option<RoiSelection>,

    // -- Results --
    pub pixel_fit_result: Option<SpectrumFitResult>,
    pub spatial_result: Option<SpatialResult>,

    // -- UI state --
    pub active_tab: Tab,
    pub status_message: String,
    pub is_fitting: bool,
    pub is_fetching_endf: bool,

    // -- Background task receivers --
    pub pending_spatial: Option<mpsc::Receiver<SpatialResult>>,
    pub pending_endf: Option<mpsc::Receiver<EndfFetchResult>>,

    // -- Preview image texture --
    pub preview_image: Option<Array2<f64>>,
    pub map_display_isotope: usize,
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

impl AppState {
    /// Cancel any in-flight background tasks by dropping their receivers.
    /// The background threads will notice the closed channel and exit.
    pub fn cancel_pending_tasks(&mut self) {
        self.pending_spatial = None;
        self.pending_endf = None;
        self.is_fitting = false;
        self.is_fetching_endf = false;
    }

    /// Clear pixel selection, ROI, results, and cancel pending tasks.
    /// Called when the underlying data changes.
    pub fn invalidate_results(&mut self) {
        self.cancel_pending_tasks();
        self.selected_pixel = None;
        self.roi = None;
        self.pixel_fit_result = None;
        self.spatial_result = None;
        self.preview_image = None;
        self.energies = None;
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            sample_path: None,
            open_beam_path: None,
            sample_data: None,
            open_beam_data: None,
            normalized: None,
            dead_pixels: None,

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

            selected_pixel: None,
            roi: None,

            pixel_fit_result: None,
            spatial_result: None,

            active_tab: Tab::Spectrum,
            status_message: "Ready".into(),
            is_fitting: false,
            is_fetching_endf: false,

            pending_spatial: None,
            pending_endf: None,

            preview_image: None,
            map_display_isotope: 0,
        }
    }
}
