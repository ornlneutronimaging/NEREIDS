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
    /// Energy rebin factor (1 = no rebinning).
    #[serde(default = "default_rebin_factor")]
    pub rebin_factor: usize,
    /// Whether rebinning has been applied.
    #[serde(default)]
    pub rebin_applied: bool,
    /// Whether the sidebar is collapsed (icon-only mode).
    #[serde(default)]
    pub sidebar_collapsed: bool,
    /// LM background enabled (SAMMY 4-param).
    #[serde(default, alias = "background_enabled")]
    pub lm_background_enabled: bool,
    /// KL background enabled (b₀ + b₁/√E).
    #[serde(default)]
    pub kl_background_enabled: bool,
    /// Bounded multiplicative baseline enabled (issue #635).
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
    /// Fit energy-scale calibration (TZERO `t₀` + flight-path `L_scale`)
    /// as free parameters.  Composes with `fit_temperature` (issue #634).
    #[serde(default)]
    pub fit_energy_scale: bool,
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

/// Serializable solver method for session cache.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CachedSolverMethod {
    LevenbergMarquardt,
    PoissonKL,
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
            rebin_factor: state.rebin_factor,
            rebin_applied: state.rebin_applied,
            sidebar_collapsed: state.sidebar_collapsed,
            lm_background_enabled: state.lm_background_enabled,
            kl_background_enabled: state.kl_background_enabled,
            baseline_enabled: state.baseline_enabled,
            kl_c_ratio: state.kl_c_ratio,
            kl_enable_polish_override: state.kl_enable_polish_override,
            fit_energy_scale: state.fit_energy_scale,
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

        // Restore rebin state
        state.rebin_factor = self.rebin_factor;
        state.rebin_applied = self.rebin_applied;

        // Restore sidebar state
        state.sidebar_collapsed = self.sidebar_collapsed;

        // Restore background normalization state
        state.lm_background_enabled = self.lm_background_enabled;
        state.kl_background_enabled = self.kl_background_enabled;
        state.baseline_enabled = self.baseline_enabled;
        state.kl_c_ratio = self.kl_c_ratio;
        state.kl_enable_polish_override = self.kl_enable_polish_override;
        state.fit_energy_scale = self.fit_energy_scale;
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

/// Which target list an ENDF fetch result belongs to.
///
/// Replaces the former `is_detect_matrix: bool` flag, giving each call-site
/// a self-documenting label instead of an opaque boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchTarget {
    Configure,
    ForwardModel,
    DetectMatrix,
    DetectTrace,
}

/// Result of a background ENDF fetch for a single isotope.
///
/// Uses `(z, a)` to identify which isotope entry the result belongs to,
/// rather than positional index which is fragile if the list is mutated
/// during a background fetch.
///
/// `target` distinguishes which list the result belongs to (Configure,
/// ForwardModel, DetectMatrix, or DetectTrace).
pub struct EndfFetchResult {
    pub z: u32,
    pub a: u32,
    pub target: FetchTarget,
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
    /// Fitted temperature (K), when temperature fitting was enabled.
    pub temperature_k: Option<f64>,
    /// Structured fit-configuration warnings from the pipeline (issue
    /// #635 — e.g. the degenerate free-Anorm + free-T + free-density
    /// trio).  Rendered as amber lines under the summary.
    pub warnings: Vec<String>,
    /// The Doppler route each isotope took, one line per isotope, as
    /// disclosed on the fit result. Empty when nothing was broadened.
    pub doppler_routes: Vec<String>,
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
            Self::Tiff => "TIFF (f32)",
            Self::Hdf5 => "HDF5",
            Self::Markdown => "Markdown Report",
        }
    }
}

/// Data mode for project file save.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaveDataMode {
    #[default]
    Linked,
    Embedded,
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
    ConfigureGroup,
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

/// Progress state for spatial mapping.
///
/// Holds an `Arc<AtomicUsize>` shared with the background thread and the
/// pixel total.  Display code reads the atomic directly each frame — no
/// intermediate polling step, no sync issues.
pub struct FittingProgress {
    counter: Arc<AtomicUsize>,
    total: usize,
}

impl FittingProgress {
    pub fn new(total: usize) -> (Self, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        let handle = Arc::clone(&counter);
        (Self { counter, total }, handle)
    }

    /// Current number of completed pixels (reads atomic).
    pub fn done(&self) -> usize {
        self.counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn fraction(&self) -> f32 {
        self.done() as f32 / self.total.max(1) as f32
    }
}

/// Cached residuals data for the Studio dock, avoiding per-frame model rebuild.
#[derive(Clone, Debug)]
pub struct CachedResiduals {
    /// Generation counter of the `pixel_fit_result` that produced this cache.
    pub fit_gen: u64,
    /// Selected pixel at cache time.
    pub pixel: (usize, usize),
    /// Resolution enabled flag at cache time.
    pub resolution_enabled: bool,
    /// Resolution mode at cache time (compared via PartialEq).
    pub resolution_mode: ResolutionMode,
    /// Flight path at cache time (affects Gaussian resolution).
    pub flight_path_m: f64,
    /// Effective temperature used for the forward model overlay.
    pub temperature_k: f64,
    /// Reduced chi-squared (stored so display doesn't need to re-extract).
    pub chi2_r: f64,
    /// Residual points: (energy_ev, residual_value).
    pub residuals: Vec<(f64, f64)>,
    /// RMS of residuals.
    pub rms: f64,
    /// Maximum absolute residual.
    pub max_abs: f64,
    /// Number of finite residual points.
    pub n_points: usize,
    /// The dock's copy of the overlay's route line (see
    /// `design::OverlayModel::route_mismatch`), which carries either of two
    /// things: that the model these residuals were taken against is not the
    /// one the fit disclosed, or that the fit disclosed no routes at all and
    /// this redraw could not be checked against anything. The residual dock
    /// shows whichever fired: residuals taken against a model that is not
    /// known to be the fitted one are as untrustworthy as the curve the
    /// spectrum panel warns about.
    pub warning: Option<String>,
}

/// Main application state.
pub struct AppState {
    // -- Data loading --
    pub input_mode: InputMode,
    pub sample_path: Option<PathBuf>,
    pub open_beam_path: Option<PathBuf>,
    pub sample_data: Option<Arc<Array3<f64>>>,
    pub open_beam_data: Option<Arc<Array3<f64>>>,
    pub normalized: Option<Arc<NormalizedData>>,
    /// Effective pixel mask applied by the analysis:
    /// `file_dead_pixels ∪ latest detection`.  Recomputed FROM SCRATCH on
    /// every normalization via [`AppState::set_detected_dead_pixels`] —
    /// never unioned with its own previous value, so detections can not
    /// accumulate across open-beam swaps / re-normalizations (#646).
    pub dead_pixels: Option<Array2<bool>>,
    /// Declared-mask provenance (#646): the pixel mask carried by an input
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
    /// Detected-mask component (#646 R4, P1-1): the dead ∪ hot mask from
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

    // -- Instrument Resolution --
    pub resolution_enabled: bool,
    pub resolution_mode: ResolutionMode,

    // -- Fitting --
    pub temperature_k: f64,
    pub lm_config: LmConfig,
    pub solver_method: SolverMethod,
    pub fit_temperature: bool,
    /// Fit residual energy-scale calibration (TZERO `t₀` μs + flight-path
    /// `L_scale`) as free parameters per SAMMY equivalent.  Composes with
    /// `fit_temperature` since issue #634 — the energy-scale model carries
    /// a fitted temperature column (joint thermometry).
    ///
    /// Initial values: `t₀ = 0.0` and `L_scale = 1.0` (identity seeds).
    /// The configured Delay has already been subtracted when the energy
    /// grid was built (`nereids-io::tof::tof_edges_to_energy`), so `t₀`
    /// represents the residual offset on top of the corrected grid;
    /// `L_scale` multiplies the nominal `flight_path_m`.
    pub fit_energy_scale: bool,
    /// Restrict the fit to `[min_eV, max_eV]` (SAMMY EMIN/EMAX equivalent —
    /// INPut-file card set 2, manual Table VI A.1).
    /// `None` = full grid (default).  Both bounds must lie inside the
    /// loaded energy grid; the resolution-kernel margin (~5×FWHM beyond
    /// each boundary) is applied automatically in `build_fit_config`
    /// so resonances near the boundaries are correctly broadened.
    pub fit_energy_range: Option<(f64, f64)>,
    pub show_advanced_solver: bool,

    // -- Uncertainty provenance --
    /// True when per-bin uncertainty was estimated from transmission shape
    /// (TransmissionTiff, HDF5 auto-prepare) rather than measured from
    /// sample + open-beam counting statistics. When true, chi-squared
    /// values are approximate and should be displayed with a warning.
    pub uncertainty_is_estimated: bool,

    // -- Background normalization --
    /// LM background: SAMMY 4-param model.
    /// Model: Anorm * T_inner(E) + BackA + BackB/sqrt(E) + BackC*sqrt(E)
    pub lm_background_enabled: bool,
    /// KL background: SAMMY 4-term wrapper (joint-Poisson compatible).
    /// Model: Anorm * T_inner(E) + BackA + BackB/sqrt(E) + BackC*sqrt(E)
    pub kl_background_enabled: bool,
    /// Proton-charge ratio c = Q_s / Q_ob for the counts-KL solver.
    /// Default 1.0 = caller PC-normalized the flux
    /// upstream.  For raw-count VENUS data, set to the actual
    /// Q_sample/Q_open_beam ratio (typically ~5–6).
    pub kl_c_ratio: f64,
    /// Override for Nelder-Mead polish on the counts-KL path.  `None`
    /// lets `spatial_map_typed` auto-disable polish when n_pixels > 1
    /// (measured ~17 min/pixel polish cost).  `Some(true)` forces
    /// polish on even at spatial scale (research use only);
    /// `Some(false)` forces off.
    pub kl_enable_polish_override: Option<bool>,
    /// Bounded multiplicative baseline (issue #635):
    /// B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref), applied OUTERMOST.
    /// When combined with a background, Anorm is held fixed (b0/Anorm are
    /// degenerate normalizations) — `build_fit_config` wires that.
    pub baseline_enabled: bool,

    // -- Pixel / ROI selection --
    pub selected_pixel: Option<(usize, usize)>,
    pub rois: Vec<RoiSelection>,
    pub selected_roi: Option<usize>,

    /// Snapshot of ROIs at the time spatial_map was launched.
    /// Used to render density overlays only on fitted pixels.
    pub fitting_rois: Vec<RoiSelection>,
    /// Toggle: show provenance history popup window.
    pub show_history_window: bool,
    /// Toggle: show the on-demand Analyze fit-details drawer.
    pub show_analyze_fit_info: bool,
    /// Toggle: show the Analyze isotope tick-strip picker popover.
    pub show_isotope_track_picker: bool,
    /// Per-`(z, a)` set of isotope tick strips the user has hidden via the
    /// picker. Hidden tracks are still part of the fit; only the diagnostic
    /// strip is suppressed so the visible track list stays focused when many
    /// isotopes are loaded.
    pub hidden_isotope_tracks: std::collections::HashSet<(u32, u32)>,

    // -- Results --
    pub pixel_fit_result: Option<SpectrumFitResult>,
    /// Generation counter; incremented each time `pixel_fit_result` is replaced.
    pub fit_result_gen: u64,
    /// Cached residuals for the Studio dock (keyed by `fit_result_gen` + resolution config).
    pub residuals_cache: Option<CachedResiduals>,
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
    /// Cached resolved dark-mode boolean; used to skip redundant `apply_theme`.
    pub last_applied_dark_mode: Option<bool>,
    /// Whether the guided-mode sidebar is collapsed (icon-only mode).
    pub sidebar_collapsed: bool,
    pub active_tab: Tab,
    pub status_message: String,
    pub is_fitting: bool,
    pub is_fetching_endf: bool,
    /// Cloned egui context for background threads to request repaints.
    pub egui_ctx: Option<egui::Context>,
    /// Poll-based file-dialog service; picks are routed by
    /// `crate::file_dialog::dispatch_results` each frame.
    pub file_dialogs: crate::file_dialog::FileDialogs,
    /// Dismissible banner shown when native file dialogs are unavailable
    /// or failed (portal/zenity chain) — #526's silent failure made
    /// visible. Set from the startup probe and the log-bridge latch.
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
    /// Cancellation token owned by the current in-flight spatial run, if any.
    /// Dedicated per run so an obsolete spatial worker can be stopped without
    /// cancelling the unrelated ENDF / forward-model / detectability workers
    /// that share `cancel_token`.
    pub spatial_cancel_token: Option<Arc<AtomicBool>>,
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

    // -- Dirty tracking for Studio re-run --
    /// When `Some`, indicates the earliest pipeline stage that needs re-running.
    /// Set by parameter edits in Studio; cleared after a successful re-run.
    pub dirty_from: Option<GuidedStep>,

    // -- Studio mode --
    pub studio_selected_tile: usize,
    pub studio_doc_tab: StudioDocTab,
    pub studio_dock_tab: usize,
    pub studio_show_dock: bool,
    pub studio_analysis_isotope: usize,
    /// Symbol of the last-displayed isotope in Studio Analysis tab.
    /// Used to detect when the isotope list changes under the selected index.
    pub studio_analysis_prev_symbol: Option<String>,

    // -- Progress --
    pub fitting_progress: Option<FittingProgress>,

    // -- Provenance --
    pub provenance_log: Vec<ProvenanceEvent>,

    // -- Per-tile display state (indexed same as density_maps + 1 for convergence) --
    pub tile_display: Vec<TileDisplayState>,

    // -- Export --
    pub export_format: ExportFormat,
    pub export_directory: Option<PathBuf>,
    pub export_status: Option<String>,

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

impl IsotopeEntry {
    /// Clone with `endf_status` normalized to match `resonance_data` presence.
    ///
    /// Used when copying entries between contexts (Configure → FM, Configure → Detect)
    /// to avoid propagating stale `Fetching` or `Failed` statuses.
    pub fn clone_with_normalized_status(&self) -> Self {
        Self {
            z: self.z,
            a: self.a,
            symbol: self.symbol.clone(),
            initial_density: self.initial_density,
            resonance_data: self.resonance_data.clone(),
            enabled: self.enabled,
            endf_status: if self.resonance_data.is_some() {
                EndfStatus::Loaded
            } else {
                EndfStatus::Pending
            },
        }
    }
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

impl RoiSelection {
    /// Check whether pixel (y, x) falls inside this ROI rectangle.
    pub fn contains(&self, y: usize, x: usize) -> bool {
        y >= self.y_start && y < self.y_end && x >= self.x_start && x < self.x_end
    }
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
    ForwardModel = 9,
    Detectability = 10,
}

impl GuidedStep {
    /// Numeric stage index for ordering comparisons.
    fn stage_order(self) -> u8 {
        self as u8
    }
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
        // The spatial worker listens on its own per-run token, not the
        // shared one; stop it explicitly.
        if let Some(token) = self.spatial_cancel_token.take() {
            token.store(true, Ordering::Relaxed);
        }
        self.pending_spatial = None;
        self.pending_endf = None;
        self.pending_fm_endf = None;
        self.pending_detect_endf = None;
        self.is_fitting = false;
        self.fitting_progress = None;
        self.is_fetching_endf = false;
        self.is_fetching_fm_endf = false;
        self.is_fetching_detect_endf = false;
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

    /// Clear only fit / normalization-downstream state.  Use this when
    /// something changes that invalidates the current normalization +
    /// fit (e.g. open-beam swap) but leaves the source sample data,
    /// spectrum values, energies, ROIs, and selection valid.
    ///
    /// Contrast with `invalidate_results`, which additionally nukes
    /// the data layer (sample, spectrum, energies, preview, ROIs,
    /// rebin state) and is appropriate when the **source** changes.
    pub fn invalidate_fit_results(&mut self) {
        self.cancel_pending_tasks();
        self.normalized = None;
        self.clear_pixel_fit_output();
        self.clear_spatial_fit_output();
    }

    /// Clear the cached single-spectrum result before starting a new pixel or
    /// ROI fit.  Input data, normalization, selection, and spatial-map results
    /// remain valid and are deliberately preserved.
    pub fn clear_pixel_fit_output(&mut self) {
        self.pixel_fit_result = None;
        self.residuals_cache = None;
        self.last_fit_feedback = None;
        self.show_analyze_fit_info = false;
    }

    /// Drop the single-spectrum fit output because the ENABLED ISOTOPE SET —
    /// or the resonance data behind it — changed.
    ///
    /// Enabling or disabling an isotope or a group, adding or removing one,
    /// or swapping the ENDF library changes the model a stored result was
    /// fitted with, but not `fit_result_gen` — which the residual cache is
    /// keyed on. Without this the cache still tests as
    /// valid and the dock goes on subtracting a curve built from the
    /// previous isotope set, reporting its RMS and Max|r| as this one's.
    ///
    /// That is the whole guarantee: the single-pixel result and its residual
    /// cache are gone. `spatial_result` is deliberately left alone — a map
    /// costs minutes to recompute — so the dock's spatial fallback
    /// ([`crate::guided::analyze::selected_pixel_fit_result_for_overlay`])
    /// can still assemble a per-pixel result that was fitted with the
    /// previous set. Three things stand between that result and a silent
    /// wrong answer: the density-parameter count guard in the dock's cache
    /// builder, which refuses a stored result whose density count no longer
    /// matches the enabled set; the identity-aware route comparison in
    /// `design::build_overlay_model`, which reports a redraw whose routes
    /// are not the disclosed ones; and — since a spatial result discloses no
    /// per-pixel routes — the undisclosed-redraw warning, which says the
    /// redraw was never checked against anything.
    pub fn clear_pixel_fit_for_isotope_change(&mut self) {
        self.clear_pixel_fit_output();
    }

    /// Enable or disable one isotope entry, with everything that change
    /// implies.
    ///
    /// The flip and the invalidation are one operation, not two steps a call
    /// site can get half right. A handler that flips the flag and forgets
    /// [`Self::clear_pixel_fit_for_isotope_change`] leaves the residual dock
    /// reporting the previous isotope set's RMS and Max|r|, because the
    /// residual cache is keyed on `fit_result_gen` and a toggle does not
    /// bump it — [`Self::mark_dirty`] only records which pipeline step went
    /// stale.
    ///
    /// A no-op for an out-of-range index, or when the flag already has the
    /// requested value.
    pub fn set_isotope_enabled(&mut self, index: usize, enabled: bool) {
        let Some(entry) = self.isotope_entries.get_mut(index) else {
            return;
        };
        if entry.enabled == enabled {
            return;
        }
        entry.enabled = enabled;
        self.on_enabled_isotope_set_changed();
    }

    /// Enable or disable one isotope group — the group equivalent of
    /// [`Self::set_isotope_enabled`], with the same coupling of the flip to
    /// the invalidation.
    pub fn set_isotope_group_enabled(&mut self, index: usize, enabled: bool) {
        let Some(group) = self.isotope_groups.get_mut(index) else {
            return;
        };
        if group.enabled == enabled {
            return;
        }
        group.enabled = enabled;
        self.on_enabled_isotope_set_changed();
    }

    /// What every change to the enabled isotope set costs: the Analyze step
    /// is stale, and so is any single-spectrum fit performed with the old
    /// set.
    fn on_enabled_isotope_set_changed(&mut self) {
        self.mark_dirty(GuidedStep::Analyze);
        self.clear_pixel_fit_for_isotope_change();
    }

    /// Set the live sample temperature, with everything that change implies.
    ///
    /// This box is a MODEL input, not a display setting.
    /// `SpectrumFitResult::temperature_k` is `Some` only for a fit that
    /// fitted the temperature, so every ordinary fixed-temperature result
    /// carries `None` and both redraw paths — the spectrum overlay
    /// (`design::build_fit_line`) and the residual dock
    /// (`studio::dock_residuals`) — fall back to this value for the
    /// temperature the physics is re-evaluated at.
    ///
    /// Nothing else would catch the change. `fit_result_gen` is not bumped
    /// by an edit here, and [`Self::mark_dirty`] only records which pipeline
    /// step went stale — it drops no output. The residual cache IS keyed on
    /// the effective temperature, so the dock does not show stale numbers;
    /// it silently RE-COMPUTES at the new temperature and presents the
    /// answer as the stored fit's RMS and Max|r|, while the overlay redraws
    /// the same way still labelled "Fit". Dropping the stored fit is what
    /// stops that, exactly as an isotope toggle drops it
    /// ([`Self::set_isotope_enabled`]).
    ///
    /// The spatial map is deliberately left alone, for the same reason and
    /// with the same protection as an isotope toggle: a map costs minutes to
    /// recompute, and a per-pixel result assembled from one discloses no
    /// Doppler routes, so `design::build_overlay_model` reports every
    /// broadening redraw of it as unchecked rather than passing it off as
    /// the fitted curve.
    ///
    /// A no-op when the value does not change: this runs on every repaint.
    pub fn set_temperature_k(&mut self, temperature_k: f64) {
        if self.temperature_k == temperature_k {
            return;
        }
        self.temperature_k = temperature_k;
        self.mark_dirty(GuidedStep::Analyze);
        self.clear_pixel_fit_output();
    }

    /// Drop every fit because the ISOTOPE LIST itself changed — a chip
    /// added or removed, or the ENDF library swapped underneath the
    /// resonance data.
    ///
    /// Stronger than [`Self::clear_pixel_fit_for_isotope_change`], which a
    /// mere enable/disable raises: there the map's own resonance data still
    /// exists and its per-pixel results can still be assembled and reported
    /// as unchecked, so the map is kept. Here the resonance data behind the
    /// map is gone or replaced, so the map cannot be redrawn against
    /// anything at all and is dropped with the single-pixel fit, its
    /// residual cache, its feedback and its panel flag.
    pub fn clear_fits_for_isotope_list_change(&mut self) {
        self.spatial_result = None;
        self.clear_pixel_fit_for_isotope_change();
    }

    /// Clear the cached map before starting a new spatial fit.  This prevents
    /// Results and export actions from reusing an older map while the new run
    /// is pending or after it fails.  An in-flight spatial worker is stopped
    /// through its dedicated per-run token so its obsolete computation does
    /// not keep burning CPU after the UI has dropped the receiver.
    pub fn clear_spatial_fit_output(&mut self) {
        if let Some(token) = self.spatial_cancel_token.take() {
            token.store(true, Ordering::Relaxed);
        }
        self.pending_spatial = None;
        self.is_fitting = false;
        self.fitting_progress = None;
        self.spatial_result = None;
        self.residuals_cache = None;
        self.fitting_rois.clear();
        self.export_status = None;
    }

    /// Invalidate all fit outputs after a fit-control change while preserving
    /// loaded inputs, normalization, energy grid, pixel selection, and ROIs.
    /// Any in-flight spatial worker is cancelled through its dedicated
    /// per-run token and its receiver detached, so an old-configuration
    /// result can neither arrive nor keep computing.  The shared cancellation
    /// token is deliberately left alone because it belongs to unrelated
    /// ENDF, forward-model, and detectability workers.
    pub fn invalidate_analysis_outputs(&mut self) {
        self.clear_pixel_fit_output();
        self.clear_spatial_fit_output();
    }

    /// Recompute the effective pixel mask FROM SCRATCH (#646):
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
    /// the drop is RETURNED as a notice the caller must surface (#646
    /// R4 F5; return-value design per review F1) — masking decisions
    /// stay observable, never silent.
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
    #[must_use = "declared-mask-drop notice: surface it (provenance/status), never drop it (#646 F1)"]
    pub fn set_detected_dead_pixels(&mut self, detected: Option<Array2<bool>>) -> Option<String> {
        // Keep the raw detected component (#646 R4, P1-1): project SAVE
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
        self.pixel_fit_result = None;
        self.residuals_cache = None;
        self.spatial_result = None;
        self.last_fit_feedback = None;
        self.fitting_rois.clear();
        self.preview_image = None;
        self.energies = None;
        self.normalized = None;
        self.dead_pixels = None;
        // The declared mask belongs to the invalidated data; every caller
        // of invalidate_results either replaces the sample or triggers a
        // reload, and the load sites reinstall the file-declared mask.
        self.file_dead_pixels = None;
        // The detected component belongs to the invalidated data too.
        self.detected_dead_pixels = None;
        self.spectrum_values = None;
        self.tile_display.clear();
        self.studio_selected_tile = 0;
        self.export_status = None;
        self.rebin_applied = false;
        self.rebin_factor = 1;
        self.analyze_tof_slice_index = 0;
        self.dirty_from = None;
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

    /// Mark the pipeline as dirty from the given step.
    ///
    /// Uses min-semantics: if already dirty from an earlier step, keeps the
    /// earlier one. Called by Studio parameter edits to track what needs
    /// re-running.
    pub fn mark_dirty(&mut self, step: GuidedStep) {
        self.dirty_from = Some(match self.dirty_from {
            Some(existing) if existing.stage_order() <= step.stage_order() => existing,
            _ => step,
        });
    }

    /// Clear dirty state after a successful pipeline re-run.
    pub fn clear_dirty(&mut self) {
        self.dirty_from = None;
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

    /// Ensure `tile_display` has enough entries for the current result.
    /// Call after spatial analysis completes.
    pub fn init_tile_display(&mut self, n_density_maps: usize) {
        // +1 for convergence, +1 for temperature map (if present)
        let has_temp = self
            .spatial_result
            .as_ref()
            .is_some_and(|r| r.temperature_map.is_some());
        let needed = n_density_maps + 1 + has_temp as usize;
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
            endf_library: EndfLibrary::EndfB8_0,

            resolution_enabled: false,
            resolution_mode: ResolutionMode::default(),

            temperature_k: 296.0,
            lm_config: LmConfig::default(),
            solver_method: SolverMethod::PoissonKL,
            fit_temperature: false,
            fit_energy_scale: false,
            fit_energy_range: None,
            show_advanced_solver: false,
            uncertainty_is_estimated: false,
            lm_background_enabled: false,
            kl_background_enabled: false,
            baseline_enabled: false,
            kl_c_ratio: 1.0,
            kl_enable_polish_override: None,

            selected_pixel: None,
            rois: Vec::new(),
            selected_roi: None,
            fitting_rois: Vec::new(),
            show_history_window: false,
            show_analyze_fit_info: false,
            show_isotope_track_picker: false,
            hidden_isotope_tracks: std::collections::HashSet::new(),

            pixel_fit_result: None,
            fit_result_gen: 0,
            residuals_cache: None,
            spatial_result: None,
            last_fit_feedback: None,

            fitting_type: None,
            data_type: None,
            pipeline: Vec::new(),
            wizard_step: 0,

            ui_mode: UiMode::Guided,
            guided_step: GuidedStep::Landing,
            theme_preference: ThemePreference::Auto,
            last_applied_dark_mode: None,
            sidebar_collapsed: false,
            active_tab: Tab::Spectrum,
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

            analysis_mode: AnalysisMode::FullSpatialMap,
            normalize_spectrum_source: SpectrumDataSource::FullImage,
            normalize_spectrum_axis: SpectrumAxis::EnergyEv,
            tof_slice_index: 0,
            show_resonance_dips: false,

            analyze_spectrum_axis: SpectrumAxis::EnergyEv,
            analyze_tof_slice_index: 0,

            pending_spatial: None,
            spatial_cancel_token: None,
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
            detect_i0: 100_000.0,
            detect_energy_min: 1.0,
            detect_energy_max: 100.0,
            detect_n_energy_points: 2000,
            detect_results: Vec::new(),
            pending_detect_endf: None,
            is_fetching_detect_endf: false,
            detect_endf_library: EndfLibrary::EndfB8_0,
            detect_temperature_k: 296.0,
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

            dirty_from: None,
            studio_selected_tile: 0,
            studio_doc_tab: StudioDocTab::Analysis,
            studio_dock_tab: 0,
            studio_show_dock: true,
            studio_analysis_isotope: 0,
            studio_analysis_prev_symbol: None,
            fitting_progress: None,

            provenance_log: Vec::new(),
            tile_display: Vec::new(),
            export_format: ExportFormat::Tiff,
            export_directory: None,
            export_status: None,

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

    /// A one-pixel spatial map, present only so that "the map was kept" and
    /// "the map was dropped" are distinguishable.
    fn a_spatial_map() -> SpatialResult {
        let map = || Array2::from_elem((1, 1), 1e-3);
        SpatialResult {
            density_maps: vec![map()],
            uncertainty_maps: vec![map()],
            chi_squared_map: map(),
            deviance_per_dof_map: None,
            converged_map: Array2::from_elem((1, 1), true),
            temperature_map: None,
            temperature_uncertainty_map: None,
            isotope_labels: vec!["U-238".into()],
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
            doppler_routes: None,
            n_converged: 1,
            n_total: 1,
            n_failed: 0,
        }
    }

    /// A state carrying one isotope, one group, a spatial map, and a
    /// single-pixel fit with its residual cache — all computed with that
    /// isotope set.
    fn state_with_a_cached_pixel_fit() -> AppState {
        AppState {
            spatial_result: Some(a_spatial_map()),
            isotope_entries: vec![IsotopeEntry {
                z: 92,
                a: 238,
                symbol: "U-238".into(),
                initial_density: 0.001,
                resonance_data: None,
                enabled: true,
                endf_status: EndfStatus::Pending,
            }],
            isotope_groups: vec![IsotopeGroupEntry {
                z: 72,
                name: "Hf".into(),
                members: Vec::new(),
                initial_density: 0.001,
                enabled: true,
            }],
            pixel_fit_result: Some(SpectrumFitResult {
                densities: vec![0.001],
                uncertainties: None,
                reduced_chi_squared: 1.0,
                converged: true,
                iterations: 3,
                temperature_k: None,
                temperature_k_unc: None,
                anorm: 1.0,
                background: [0.0; 3],
                back_d: None,
                back_f: None,
                t0_us: None,
                l_scale: None,
                energy_scale_flight_path_m: None,
                deviance_per_dof: None,
                baseline: None,
                baseline_e_ref_ev: None,
                warnings: Vec::new(),
                doppler_routes: None,
            }),
            residuals_cache: Some(CachedResiduals {
                fit_gen: 0,
                pixel: (1, 1),
                resolution_enabled: false,
                resolution_mode: ResolutionMode::Gaussian {
                    delta_t_us: 0.0,
                    delta_l_m: 0.0,
                },
                flight_path_m: 25.0,
                temperature_k: 293.6,
                chi2_r: 1.0,
                residuals: vec![(6.0, 0.01)],
                rms: 0.01,
                max_abs: 0.01,
                n_points: 1,
                warning: None,
            }),
            ..AppState::default()
        }
    }

    /// Enabling or disabling an isotope changes the model a stored fit was
    /// performed with, and the residual cache is keyed on `fit_result_gen`,
    /// which the toggle does NOT bump. So the toggle has to drop the result
    /// and the cache itself; without that the dock goes on subtracting a
    /// curve built from the previous isotope set and reporting its RMS and
    /// Max|r| as this one's.
    #[test]
    fn an_isotope_toggle_drops_the_fit_it_invalidates() {
        let mut state = state_with_a_cached_pixel_fit();

        state.clear_pixel_fit_for_isotope_change();

        assert!(
            state.pixel_fit_result.is_none(),
            "the result was fitted with the previous isotope set"
        );
        assert!(
            state.residuals_cache.is_none(),
            "residuals against the previous isotope set are not this one's"
        );
        // The cache key cannot be what protects it: the toggle leaves the
        // generation counter exactly where the stale cache's key is.
        assert_eq!(state.fit_result_gen, 0);
    }

    /// The toggle handlers flip the flag THROUGH these methods, which is
    /// what keeps the flip and the invalidation from coming apart: a
    /// handler that writes the field itself and forgets the invalidation
    /// compiles, renders, and leaves the residual dock reporting the
    /// previous isotope set's RMS and Max|r|.
    #[test]
    fn setting_an_isotope_enabled_flips_it_and_invalidates_the_fit() {
        let mut state = state_with_a_cached_pixel_fit();
        state.set_isotope_enabled(0, false);
        assert!(!state.isotope_entries[0].enabled, "the flag has to flip");
        assert!(
            state.pixel_fit_result.is_none(),
            "the result was fitted with the isotope that was just disabled"
        );
        assert!(
            state.residuals_cache.is_none(),
            "the residuals were taken against that same model"
        );
        assert_eq!(state.dirty_from, Some(GuidedStep::Analyze));
        // Nothing else can catch this: the key the cache is checked on is
        // exactly where it was.
        assert_eq!(state.fit_result_gen, 0);

        // A group is the same event and takes the same route.
        let mut state = state_with_a_cached_pixel_fit();
        state.set_isotope_group_enabled(0, false);
        assert!(!state.isotope_groups[0].enabled, "the flag has to flip");
        assert!(state.pixel_fit_result.is_none());
        assert!(state.residuals_cache.is_none());
        assert_eq!(state.dirty_from, Some(GuidedStep::Analyze));

        // A call that changes nothing is not a change. These run on every
        // repaint, so a stored fit must survive the frames where the user
        // touched nothing, and an out-of-range index must do nothing at all.
        let mut state = state_with_a_cached_pixel_fit();
        state.set_isotope_enabled(0, true);
        state.set_isotope_group_enabled(0, true);
        state.set_isotope_enabled(7, false);
        state.set_isotope_group_enabled(7, false);
        assert!(state.pixel_fit_result.is_some());
        assert!(state.residuals_cache.is_some());
        assert_eq!(state.dirty_from, None);
    }

    /// The live temperature box is a model input: every fixed-temperature
    /// result carries `temperature_k: None`, so both redraw paths fall back
    /// to it. Moving it therefore changes the curve the dock subtracts and
    /// the overlay draws — and because the residual cache is KEYED on that
    /// temperature, the dock does not go stale, it silently re-computes at
    /// the new value and reports the answer as the stored fit's RMS and
    /// Max|r|. Dropping the fit is the only thing that stops that: the
    /// generation counter the cache is checked on does not move.
    #[test]
    fn a_temperature_edit_drops_the_fit_it_would_otherwise_recompute() {
        let mut state = state_with_a_cached_pixel_fit();
        let before = state.temperature_k;

        state.set_temperature_k(before + 100.0);

        assert_eq!(state.temperature_k, before + 100.0, "the value has to move");
        assert!(
            state.pixel_fit_result.is_none(),
            "the result was fitted at the previous temperature"
        );
        assert!(
            state.residuals_cache.is_none(),
            "the residuals were taken against a curve evaluated at it"
        );
        assert_eq!(state.dirty_from, Some(GuidedStep::Analyze));
        // Nothing else can catch this: the key the cache is checked on is
        // exactly where it was, and the cache's own temperature field moves
        // WITH the box, so the cache would test as valid at the new value.
        assert_eq!(state.fit_result_gen, 0);
        // Same trade as an isotope toggle: the map costs minutes and its
        // per-pixel redraws are reported as unchecked, so it is kept.
        assert!(state.spatial_result.is_some());

        // A call that changes nothing is not a change. This runs on every
        // repaint, so a stored fit must survive the frames where the user
        // touched nothing.
        let mut state = state_with_a_cached_pixel_fit();
        state.set_temperature_k(state.temperature_k);
        assert!(state.pixel_fit_result.is_some());
        assert!(state.residuals_cache.is_some());
        assert_eq!(state.dirty_from, None);
    }

    /// Adding or removing an isotope chip, or swapping the ENDF library,
    /// destroys or replaces the resonance data the MAP was computed from —
    /// unlike an enable/disable, after which the map's own data still exists
    /// and its per-pixel results can still be reported as unchecked. So this
    /// one drops the map as well, and it drops the whole single-fit output
    /// with it: the residual cache, the feedback and the panel flag, not
    /// just `pixel_fit_result`.
    #[test]
    fn an_isotope_list_change_drops_the_map_and_the_whole_single_fit_output() {
        let mut state = state_with_a_cached_pixel_fit();
        state.last_fit_feedback = Some(FitFeedback {
            success: true,
            summary: "converged".into(),
            densities: vec![("U-238".into(), 1e-3)],
            temperature_k: None,
            warnings: Vec::new(),
            doppler_routes: Vec::new(),
        });
        state.show_analyze_fit_info = true;

        state.clear_fits_for_isotope_list_change();

        assert!(state.spatial_result.is_none());
        assert!(state.pixel_fit_result.is_none());
        assert!(
            state.residuals_cache.is_none(),
            "residuals against the previous resonance data are not this one's"
        );
        assert!(
            state.last_fit_feedback.is_none(),
            "the feedback describes a fit that no longer exists"
        );
        assert!(
            !state.show_analyze_fit_info,
            "the fit-info panel has nothing left to show"
        );

        // The weaker isotope-set invalidation keeps the map on purpose, so
        // the two are not interchangeable and this one is not vacuous.
        let mut state = state_with_a_cached_pixel_fit();
        state.clear_pixel_fit_for_isotope_change();
        assert!(state.pixel_fit_result.is_none());
        assert!(
            state.spatial_result.is_some(),
            "an enable/disable keeps the map: it costs minutes and its \
             per-pixel redraws are reported as unchecked"
        );
    }

    /// The #646 accumulation pin: re-detection REPLACES the previous
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
    /// silent (#646 R4 F5; return-value design F1).  The setter itself
    /// no longer logs to provenance — project restore replaces the
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
