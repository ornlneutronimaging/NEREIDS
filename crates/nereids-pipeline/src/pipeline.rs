//! Single-spectrum analysis pipeline.
//!
//! Orchestrates the full analysis chain for a single transmission spectrum:
//! ENDF loading → cross-section calculation → broadening → fitting.
//!
//! This is the building block for the spatial mapping pipeline.
//!
//! Uses `UnifiedFitConfig` with `SolverConfig` and typed `InputData` variants.

use std::fmt;
use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::joint_poisson::{self, JointPoissonFitConfig, JointPoissonObjective};
use nereids_fitting::lm::{self, FitModel, LmConfig, LmResult};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{self, PoissonConfig};
use nereids_fitting::transmission_model::{
    EnergyScaleTransmissionModel, NormalizedTransmissionModel, PrecomputedTransmissionModel,
    TransmissionFitModel,
};
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::{self as nereids_transmission, InstrumentParams};

use crate::error::PipelineError;

/// SAMMY-style normalization and background configuration.
///
/// When enabled, the transmission model becomes:
///   T_out(E) = Anorm × T_inner(E) + BackA + BackB / √E + BackC × √E
///            + BackD × exp(−BackF / √E)
///
/// The first 4 background parameters (Anorm, BackA, BackB, BackC) are always
/// available.  The exponential tail (BackD, BackF) is optional and disabled
/// by default (`fit_back_d = false`, `fit_back_f = false`).
///
/// ## SAMMY Reference
/// SAMMY manual Sec III.E.2 — NORMAlization and BACKGround cards.
/// SAMMY fits up to 6 background terms; we implement all 6.
#[derive(Debug, Clone)]
pub struct BackgroundConfig {
    /// Initial value for the normalization factor (default 1.0).
    pub anorm_init: f64,
    /// Initial value for the constant background (default 0.0).
    pub back_a_init: f64,
    /// Initial value for the 1/√E background term (default 0.0).
    pub back_b_init: f64,
    /// Initial value for the √E background term (default 0.0).
    pub back_c_init: f64,
    /// Initial value for the exponential amplitude (default 0.01).
    ///
    /// Must be > 0 when `fit_back_f` is true, otherwise the Jacobian
    /// column for BackF is identically zero and BackF cannot be learned.
    pub back_d_init: f64,
    /// Initial value for the exponential decay constant (default 1.0).
    ///
    /// Units: √eV.  Must be > 0 when `fit_back_d` is true, otherwise
    /// BackD is indistinguishable from BackA (both become constants).
    pub back_f_init: f64,
    /// Whether Anorm is free (true) or fixed (false).
    pub fit_anorm: bool,
    /// Whether BackA is free (true) or fixed (false).
    pub fit_back_a: bool,
    /// Whether BackB is free (true) or fixed (false).
    pub fit_back_b: bool,
    /// Whether BackC is free (true) or fixed (false).
    pub fit_back_c: bool,
    /// Whether BackD (exponential amplitude) is free (true) or fixed (false).
    pub fit_back_d: bool,
    /// Whether BackF (exponential decay constant) is free (true) or fixed (false).
    pub fit_back_f: bool,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            anorm_init: 1.0,
            back_a_init: 0.0,
            back_b_init: 0.0,
            back_c_init: 0.0,
            back_d_init: 0.01,
            back_f_init: 1.0,
            fit_anorm: true,
            fit_back_a: true,
            fit_back_b: true,
            fit_back_c: true,
            fit_back_d: false,
            fit_back_f: false,
        }
    }
}

/// Indices of SAMMY background parameters in the full parameter vector.
///
/// Replaces the previous 4-tuple representation and supports the optional
/// exponential tail terms (BackD, BackF).
#[derive(Debug, Clone, Copy)]
struct BackgroundIndices {
    anorm: usize,
    back_a: usize,
    back_b: usize,
    back_c: usize,
    back_d: Option<usize>,
    back_f: Option<usize>,
}

// ── New typed pipeline API (Phase 0) ─────────────────────────────────────

/// Typed input data — makes the data format explicit at the API boundary.
///
/// The two variants carry genuinely different data:
/// - **Counts**: raw detector counts + open beam → Poisson statistics native
/// - **Transmission**: normalized T = sample/open_beam + uncertainty → Gaussian statistics
///
/// `spatial_map_typed` dispatches to the correct fitting engine based on
/// which variant is provided.  This eliminates the overloaded positional
/// arguments that caused silent misinterpretation in the old API.
#[derive(Debug, Clone)]
pub enum InputData {
    /// Pre-normalized transmission with Gaussian uncertainties.
    ///
    /// Use with LM (default) or Poisson KL (opt-in for low-count T data).
    Transmission {
        /// Measured transmission T(E), values typically in [0, 2].
        transmission: Vec<f64>,
        /// Per-bin uncertainty σ_T(E).
        uncertainty: Vec<f64>,
    },
    /// Raw detector counts with open beam reference.
    ///
    /// Always uses Poisson KL (statistically optimal for count data).
    /// The fitting engine works directly on counts — no information-losing
    /// normalization to transmission.
    Counts {
        /// Sample counts per energy bin.
        sample_counts: Vec<f64>,
        /// Open-beam counts per energy bin (normalization reference).
        open_beam_counts: Vec<f64>,
    },
    /// Counts with pre-estimated nuisance parameters (power users).
    ///
    /// Use when you want to inspect or override the flux estimate before fitting.
    CountsWithNuisance {
        /// Sample counts per energy bin.
        sample_counts: Vec<f64>,
        /// Estimated flux spectrum (from open beam spatial average).
        flux: Vec<f64>,
        /// Estimated detector background spectrum.
        background: Vec<f64>,
    },
}

impl InputData {
    /// Number of energy bins.
    pub fn n_energies(&self) -> usize {
        match self {
            Self::Transmission { transmission, .. } => transmission.len(),
            Self::Counts { sample_counts, .. } => sample_counts.len(),
            Self::CountsWithNuisance { sample_counts, .. } => sample_counts.len(),
        }
    }

    /// Whether this is count data (Poisson-native).
    pub fn is_counts(&self) -> bool {
        matches!(self, Self::Counts { .. } | Self::CountsWithNuisance { .. })
    }
}

/// Solver-specific configuration.
///
/// Carries the full solver config inside each variant, making invalid
/// combinations unrepresentable.
#[derive(Debug, Clone, Default)]
pub enum SolverConfig {
    /// Levenberg-Marquardt chi-squared minimizer.
    LevenbergMarquardt(LmConfig),
    /// Fixed-flux Poisson KL (legacy counts path; see memo 35 §P1 for the
    /// GOF-reporting limitations compared to [`SolverConfig::JointPoisson`]).
    PoissonKL(PoissonConfig),
    /// Joint-Poisson profile-binomial-deviance fitter (memo 35 §P1/§P2).
    ///
    /// Uses an explicit proton-charge ratio `c = Q_s/Q_ob`
    /// (from `CountsBackgroundConfig::c`) and reports `D/(n − k)` as the
    /// primary GOF.  Stage-1 damped Fisher followed by optional Nelder-Mead
    /// polish (see [`JointPoissonFitConfig`]).
    JointPoisson(JointPoissonFitConfig),
    /// Automatic: Counts → PoissonKL (legacy default), Transmission → LM.
    #[default]
    Auto,
}

/// Background model for the counts fitting engine.
///
/// In the counts domain, the forward model is:
///   Y(E) = α₁ · [Φ(E) · exp(-Σ nᵢσᵢ(E))] + α₂ · B(E)
///
/// where Φ(E) is the incident flux and B(E) is detector/gamma background.
/// The reference Φ(E) / B(E) spectra are supplied by the caller or by
/// spatial pre-processing; this config only controls the fitted scale factors.
///
/// Important distinction:
/// - This is a detector-space counts background model `B(E)`.
/// - It is NOT the same as the transmission-lift background used by
///   `BackgroundConfig`, which models additive uplift of the apparent
///   transmission curve (for example gamma-tail structure that pushes
///   transmission upward).
///
/// For VENUS MCP/TPX event detectors, the current working assumption is:
/// - raw/open-beam is the correct normalization baseline
/// - dark-current / CCD-style electronic offset is not modeled
/// - rare ghost counts may exist at the hardware level, but are currently
///   treated as negligible unless a detector-background reference spectrum
///   is explicitly provided
///
/// This is structurally different from the transmission background model
/// ([`BackgroundConfig`]) because:
/// - Φ and B are reference spectra, not fitted per pixel
/// - α₁ and α₂ are optional per-pixel scale corrections
/// - All terms are non-negative (required for valid Poisson NLL)
#[derive(Debug, Clone)]
pub struct CountsBackgroundConfig {
    /// Initial normalization scale (default 1.0).
    pub alpha_1_init: f64,
    /// Initial background scale (default 1.0).
    pub alpha_2_init: f64,
    /// Whether α₁ is free (true) or fixed (false).
    pub fit_alpha_1: bool,
    /// Whether α₂ is free (true) or fixed (false).
    pub fit_alpha_2: bool,
    /// Proton-charge ratio `c = Q_s / Q_ob` for the joint-Poisson solver
    /// (memo 35 §P1.3 — "make `c` a first-class API parameter").
    ///
    /// Default `1.0`, which is **only** correct when the caller has already
    /// PC-normalized the open-beam counts so that `flux = c · O`.  For
    /// `SolverConfig::JointPoisson`, set this to the actual
    /// `Q_sample / Q_open_beam` ratio and pass raw open-beam counts —
    /// the solver will profile out λ itself.
    ///
    /// Ignored by `SolverConfig::PoissonKL` and LM paths.
    pub c: f64,
}

impl Default for CountsBackgroundConfig {
    fn default() -> Self {
        Self {
            alpha_1_init: 1.0,
            alpha_2_init: 1.0,
            fit_alpha_1: false,
            fit_alpha_2: false,
            c: 1.0,
        }
    }
}

// ── Phase 2: UnifiedFitConfig + fit_spectrum_typed ───────────────────────

/// Unified fit configuration for all data types and solvers.
///
/// Carries both transmission and counts background configs, and uses
/// [`SolverConfig`] (which embeds solver-specific tuning).
#[derive(Debug, Clone)]
pub struct UnifiedFitConfig {
    // ── Physics (shared by both engines) ──
    energies: Vec<f64>,
    resonance_data: Vec<ResonanceData>,
    isotope_names: Vec<String>,
    temperature_k: f64,
    resolution: Option<ResolutionFunction>,
    initial_densities: Vec<f64>,
    fit_temperature: bool,
    compute_covariance: bool,

    // ── Solver ──
    solver: SolverConfig,

    // ── Background models (engine-specific) ──
    /// SAMMY-style background for the transmission engine.
    transmission_background: Option<BackgroundConfig>,
    /// Counts-domain background for the counts engine.
    counts_background: Option<CountsBackgroundConfig>,

    // ── Precomputed caches (injected by spatial_map_typed) ──
    precomputed_cross_sections: Option<Arc<Vec<Vec<f64>>>>,
    precomputed_base_xs: Option<Arc<Vec<Vec<f64>>>>,

    // ── Energy-scale calibration (SAMMY TZERO equivalent) ──
    /// When true, fit t₀ (μs) and L_scale (dimensionless) parameters.
    /// These adjust the energy axis during fitting:
    ///   E_corr = (TOF_FACTOR * L * L_scale / (t_nom - t₀))²
    fit_energy_scale: bool,
    /// Initial t₀ value in microseconds (default 0.0).
    t0_init_us: f64,
    /// Initial L_scale value (dimensionless, default 1.0).
    l_scale_init: f64,
    /// Flight path in meters for TOF↔energy conversion (default from resolution or 25.0).
    flight_path_m: f64,

    // ── Isotope group mapping (optional) ──
    /// Maps member isotope index → density parameter index.
    /// `None` = identity mapping (one param per isotope, backward compat).
    pub(crate) density_indices: Option<Vec<usize>>,
    /// Fractional ratio per member isotope.
    /// `None` = all 1.0 (backward compat).
    pub(crate) density_ratios: Option<Vec<f64>>,
    /// Number of density parameters (groups or isotopes).
    /// `None` = `resonance_data.len()` (backward compat).
    n_density_params: Option<usize>,
}

impl UnifiedFitConfig {
    /// Construct a new config with validation.
    pub fn new(
        energies: Vec<f64>,
        resonance_data: Vec<ResonanceData>,
        isotope_names: Vec<String>,
        temperature_k: f64,
        resolution: Option<ResolutionFunction>,
        initial_densities: Vec<f64>,
    ) -> Result<Self, FitConfigError> {
        if energies.is_empty() {
            return Err(FitConfigError::EmptyEnergies);
        }
        if resonance_data.is_empty() {
            return Err(FitConfigError::EmptyResonanceData);
        }
        if initial_densities.len() != resonance_data.len() {
            return Err(FitConfigError::DensityCountMismatch {
                densities: initial_densities.len(),
                isotopes: resonance_data.len(),
            });
        }
        if isotope_names.len() != resonance_data.len() {
            return Err(FitConfigError::NameCountMismatch {
                names: isotope_names.len(),
                isotopes: resonance_data.len(),
            });
        }
        if !temperature_k.is_finite() {
            return Err(FitConfigError::NonFiniteTemperature(temperature_k));
        }
        if temperature_k < 0.0 {
            return Err(FitConfigError::NegativeTemperature(temperature_k));
        }
        Ok(Self {
            energies,
            resonance_data,
            isotope_names,
            temperature_k,
            resolution,
            initial_densities,
            fit_temperature: false,
            compute_covariance: true,
            solver: SolverConfig::Auto,
            transmission_background: None,
            counts_background: None,
            precomputed_cross_sections: None,
            precomputed_base_xs: None,
            fit_energy_scale: false,
            t0_init_us: 0.0,
            l_scale_init: 1.0,
            flight_path_m: 25.0,
            density_indices: None,
            density_ratios: None,
            n_density_params: None,
        })
    }

    // ── Builder methods ──

    #[must_use]
    pub fn with_solver(mut self, solver: SolverConfig) -> Self {
        self.solver = solver;
        self
    }

    #[must_use]
    pub fn with_fit_temperature(mut self, v: bool) -> Self {
        self.fit_temperature = v;
        self
    }

    #[must_use]
    pub fn with_compute_covariance(mut self, v: bool) -> Self {
        self.compute_covariance = v;
        self
    }

    /// Enable energy-scale fitting (SAMMY TZERO equivalent).
    ///
    /// Adds t₀ (μs) and L_scale (dimensionless) as fit parameters.
    /// These adjust the energy axis during fitting to correct for
    /// flight-path and timing-offset uncertainties.
    #[must_use]
    pub fn with_energy_scale(
        mut self,
        t0_init_us: f64,
        l_scale_init: f64,
        flight_path_m: f64,
    ) -> Self {
        self.fit_energy_scale = true;
        self.t0_init_us = t0_init_us;
        self.l_scale_init = l_scale_init;
        self.flight_path_m = flight_path_m;
        self
    }

    #[must_use]
    pub fn with_transmission_background(mut self, bg: BackgroundConfig) -> Self {
        self.transmission_background = Some(bg);
        self
    }

    #[must_use]
    pub fn with_counts_background(mut self, bg: CountsBackgroundConfig) -> Self {
        self.counts_background = Some(bg);
        self
    }

    #[must_use]
    pub fn with_precomputed_cross_sections(mut self, xs: Arc<Vec<Vec<f64>>>) -> Self {
        self.precomputed_cross_sections = Some(xs);
        self
    }

    #[must_use]
    pub fn with_precomputed_base_xs(mut self, xs: Arc<Vec<Vec<f64>>>) -> Self {
        self.precomputed_base_xs = Some(xs);
        self
    }

    /// Configure isotope groups with ratio constraints.
    ///
    /// Each group binds multiple isotopes to one fitted density parameter.
    /// `groups` is a slice of `(IsotopeGroup, member_resonance_data)` pairs.
    /// `initial_densities` must have one entry per group.
    ///
    /// Replaces the existing per-isotope configuration with the expanded
    /// group mapping (flattened resonance_data + density_indices + density_ratios).
    pub fn with_groups(
        mut self,
        groups: &[(&nereids_core::types::IsotopeGroup, &[ResonanceData])],
        initial_densities: Vec<f64>,
    ) -> Result<Self, FitConfigError> {
        if groups.is_empty() {
            return Err(FitConfigError::EmptyResonanceData);
        }
        if initial_densities.len() != groups.len() {
            return Err(FitConfigError::DensityCountMismatch {
                densities: initial_densities.len(),
                isotopes: groups.len(),
            });
        }
        let mut all_resonance_data = Vec::new();
        let mut all_indices = Vec::new();
        let mut all_ratios = Vec::new();
        let mut names = Vec::new();
        for (g_idx, (group, rd_list)) in groups.iter().enumerate() {
            if rd_list.len() != group.n_members() {
                return Err(FitConfigError::GroupMemberCountMismatch {
                    group_name: group.name().to_string(),
                    rd_count: rd_list.len(),
                    member_count: group.n_members(),
                });
            }
            names.push(group.name().to_string());
            for ((isotope, ratio), rd) in group.members().iter().zip(rd_list.iter()) {
                // Validate that the ResonanceData matches the expected member isotope.
                if rd.isotope != *isotope {
                    return Err(FitConfigError::GroupMemberIsotopeMismatch {
                        group_name: group.name().to_string(),
                        expected_z: isotope.z(),
                        expected_a: isotope.a(),
                        got_z: rd.isotope.z(),
                        got_a: rd.isotope.a(),
                    });
                }
                all_resonance_data.push(rd.clone());
                all_indices.push(g_idx);
                all_ratios.push(*ratio);
            }
        }
        self.resonance_data = all_resonance_data;
        self.isotope_names = names;
        self.initial_densities = initial_densities;
        self.n_density_params = Some(groups.len());
        self.density_indices = Some(all_indices);
        self.density_ratios = Some(all_ratios);
        // Clear stale caches — the isotope set changed.
        self.precomputed_cross_sections = None;
        self.precomputed_base_xs = None;
        Ok(self)
    }

    // ── Accessors ──

    pub fn energies(&self) -> &[f64] {
        &self.energies
    }
    pub fn resonance_data(&self) -> &[ResonanceData] {
        &self.resonance_data
    }
    pub fn isotope_names(&self) -> &[String] {
        &self.isotope_names
    }
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }
    pub fn resolution(&self) -> Option<&ResolutionFunction> {
        self.resolution.as_ref()
    }
    pub fn initial_densities(&self) -> &[f64] {
        &self.initial_densities
    }
    pub fn solver(&self) -> &SolverConfig {
        &self.solver
    }
    pub fn fit_temperature(&self) -> bool {
        self.fit_temperature
    }
    pub fn transmission_background(&self) -> Option<&BackgroundConfig> {
        self.transmission_background.as_ref()
    }
    pub fn counts_background(&self) -> Option<&CountsBackgroundConfig> {
        self.counts_background.as_ref()
    }
    pub fn precomputed_cross_sections(&self) -> Option<&Arc<Vec<Vec<f64>>>> {
        self.precomputed_cross_sections.as_ref()
    }
    /// Number of density parameters (one per group or per isotope).
    pub fn n_density_params(&self) -> usize {
        self.n_density_params.unwrap_or(self.resonance_data.len())
    }

    /// Resolve `SolverConfig::Auto` into a concrete solver for the given input.
    pub(crate) fn effective_solver(&self, input: &InputData) -> SolverConfig {
        match &self.solver {
            SolverConfig::Auto => {
                if input.is_counts() {
                    SolverConfig::PoissonKL(PoissonConfig::default())
                } else {
                    SolverConfig::LevenbergMarquardt(LmConfig::default())
                }
            }
            other => other.clone(),
        }
    }
}

/// Fit a single spectrum using the typed input data API.
///
/// Dispatches to the correct fitting engine based on the `InputData` variant
/// and solver configuration:
///
/// | Input | Solver | Path |
/// |-------|--------|------|
/// | Transmission | LM | LM chi-squared with optional SAMMY background |
/// | Transmission | KL | Poisson NLL on transmission with optional background |
/// | Counts | KL | Poisson NLL on raw counts (statistically optimal) |
/// | Counts | LM | Convert to T internally and route to LM |
/// | CountsWithNuisance | KL | Direct Poisson with user-supplied nuisance |
pub fn fit_spectrum_typed(
    input: &InputData,
    config: &UnifiedFitConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let n_e = config.energies().len();

    // Validate temperature when fitting is requested
    if config.fit_temperature && config.temperature_k < 1.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "temperature must be >= 1.0 K when fit_temperature is true, got {}",
            config.temperature_k,
        )));
    }

    // Validate input length matches energy grid
    if input.n_energies() != n_e {
        return Err(PipelineError::ShapeMismatch(format!(
            "input data has {} energy bins but config.energies has {}",
            input.n_energies(),
            n_e,
        )));
    }

    // Validate auxiliary array lengths match the primary data
    match input {
        InputData::Transmission {
            transmission,
            uncertainty,
        } => {
            if uncertainty.len() != transmission.len() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "uncertainty length {} != transmission length {}",
                    uncertainty.len(),
                    transmission.len(),
                )));
            }
        }
        InputData::Counts {
            sample_counts,
            open_beam_counts,
        } => {
            if open_beam_counts.len() != sample_counts.len() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "open_beam_counts length {} != sample_counts length {}",
                    open_beam_counts.len(),
                    sample_counts.len(),
                )));
            }
        }
        InputData::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        } => {
            if flux.len() != sample_counts.len() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "flux length {} != sample_counts length {}",
                    flux.len(),
                    sample_counts.len(),
                )));
            }
            if background.len() != sample_counts.len() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "background length {} != sample_counts length {}",
                    background.len(),
                    sample_counts.len(),
                )));
            }
        }
    }

    let effective_solver = config.effective_solver(input);

    match (input, &effective_solver) {
        // ── Transmission + LM: the well-tested path ──
        (
            InputData::Transmission {
                transmission,
                uncertainty,
            },
            SolverConfig::LevenbergMarquardt(lm_cfg),
        ) => fit_transmission_lm(transmission, uncertainty, config, lm_cfg),

        // ── Transmission + KL: Poisson NLL on transmission values ──
        (
            InputData::Transmission {
                transmission,
                uncertainty,
            },
            SolverConfig::PoissonKL(poisson_cfg),
        ) => fit_transmission_poisson(transmission, uncertainty, config, poisson_cfg),

        // ── Counts + KL: statistically optimal path ──
        (
            InputData::Counts {
                sample_counts,
                open_beam_counts,
            },
            SolverConfig::PoissonKL(poisson_cfg),
        ) => {
            // Convenience counts path:
            // - open beam is used as the flux reference Φ(E)
            // - detector-space counts background B_det(E) is currently
            //   assumed zero unless the caller explicitly supplies nuisance
            //   spectra via CountsWithNuisance
            //
            // This does NOT disable transmission background fitting.
            // If config.transmission_background is enabled, fit_counts_poisson
            // still fits the additive transmission-lift terms [b0, b1] inside
            // the bracketed transmission model:
            //   Y(E) = Φ(E) * [T(E) + b0 + b1/sqrt(E)] + B_det(E)
            //
            // That transmission-lift background is the mechanism currently
            // used to absorb gamma-tail structure in VENUS-style data.
            let flux: Vec<f64> = open_beam_counts.to_vec();
            let background = vec![0.0f64; n_e];
            fit_counts_poisson(sample_counts, &flux, &background, config, poisson_cfg)
        }

        // ── CountsWithNuisance + KL: user-supplied nuisance ──
        (
            InputData::CountsWithNuisance {
                sample_counts,
                flux,
                background,
            },
            SolverConfig::PoissonKL(poisson_cfg),
        ) => fit_counts_poisson(sample_counts, flux, background, config, poisson_cfg),

        // ── Counts + LM: convert to transmission (approximate path) ──
        //
        // This is NOT a native counts-domain LM engine.  Counts are divided
        // (sample/OB) to produce transmission, with σ ≈ √max(sample,1)/OB
        // as a simplified Poisson-to-Gaussian conversion.  Poisson structure
        // is lost.  For statistically correct low-count fitting, use the
        // Poisson KL solver (`solver="kl"` or `SolverConfig::Auto`).
        (
            InputData::Counts {
                sample_counts,
                open_beam_counts,
            },
            SolverConfig::LevenbergMarquardt(lm_cfg),
        ) => {
            let (transmission, uncertainty) =
                counts_to_transmission(sample_counts, open_beam_counts);
            fit_transmission_lm(&transmission, &uncertainty, config, lm_cfg)
        }

        // ── CountsWithNuisance + LM: not meaningful ──
        (InputData::CountsWithNuisance { .. }, SolverConfig::LevenbergMarquardt(_)) => {
            Err(PipelineError::InvalidParameter(
                "CountsWithNuisance requires a counts-domain solver (LM cannot use nuisance parameters)"
                    .into(),
            ))
        }

        // ── Joint-Poisson dispatch (memo 35 §P1/§P2) ──
        (
            InputData::Counts {
                sample_counts,
                open_beam_counts,
            },
            SolverConfig::JointPoisson(jp_cfg),
        ) => {
            // Use raw open-beam counts as O; joint-Poisson profiles λ̂ out
            // using the explicit c from CountsBackgroundConfig.
            let bg = vec![0.0f64; n_e];
            fit_counts_joint_poisson(sample_counts, open_beam_counts, &bg, config, jp_cfg)
        }
        (
            InputData::CountsWithNuisance {
                sample_counts,
                flux,
                background,
            },
            SolverConfig::JointPoisson(jp_cfg),
        ) => fit_counts_joint_poisson(sample_counts, flux, background, config, jp_cfg),

        // Joint-Poisson requires counts input — transmission data doesn't
        // have the (O, S) pair needed for the conditional-binomial form.
        (InputData::Transmission { .. }, SolverConfig::JointPoisson(_)) => {
            Err(PipelineError::InvalidParameter(
                "JointPoisson solver requires raw counts (sample + open-beam), \
                 not a transmission ratio.  Provide InputData::Counts or use LM/PoissonKL \
                 on transmission data."
                    .into(),
            ))
        }

        // Auto should be resolved by effective_solver
        (_, SolverConfig::Auto) => unreachable!("Auto should be resolved before dispatch"),
    }
}

/// Convert counts to transmission: T = sample/open_beam, σ = √(max(sample,1))/open_beam.
///
/// Zero-count bins (sample == 0) get σ = 1e10 so the fitter effectively ignores them.
/// Near-zero open beam bins use a floor of 1e-10 to avoid division by zero.
/// Convert raw counts to transmission with approximate Poisson uncertainty.
///
/// This is a simplified conversion for the Counts+LM fallback path.
/// The uncertainty σ ≈ √max(sample,1)/OB is a Gaussian approximation of
/// Poisson statistics, valid when counts are high (≥ ~20).  At low counts,
/// this overestimates confidence relative to the Poisson KL solver.
///
/// Zero-count and zero-OB bins are marked with sentinel uncertainties
/// (1e10 and 1e30 respectively) so the LM solver effectively ignores them.
fn counts_to_transmission(sample: &[f64], open_beam: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let transmission: Vec<f64> = sample
        .iter()
        .zip(open_beam.iter())
        .map(|(&s, &ob)| if ob > 0.0 { s / ob } else { 0.0 })
        .collect();
    let uncertainty: Vec<f64> = sample
        .iter()
        .zip(open_beam.iter())
        .map(|(&s, &ob)| {
            if ob <= 0.0 {
                // No open beam signal — treat as dead bin
                1e30
            } else if s <= 0.0 {
                // Zero sample counts — large σ so the fitter ignores this bin
                1e10
            } else {
                s.max(1.0).sqrt() / ob.max(1e-10)
            }
        })
        .collect();
    (transmission, uncertainty)
}

/// Transmission + LM path.
fn fit_transmission_lm(
    measured_t: &[f64],
    sigma: &[f64],
    config: &UnifiedFitConfig,
    lm_config: &LmConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let n_density_params = config.n_density_params();

    // Build parameter vector
    let mut param_vec = build_density_params(config);

    // Temperature parameter
    let _temperature_index = if config.fit_temperature {
        param_vec.push(FitParameter {
            name: "temperature_k".into(),
            value: config.temperature_k,
            lower: 1.0,
            upper: 5000.0,
            fixed: false,
        });
        Some(n_density_params)
    } else {
        None
    };

    // Guard: energy-scale + temperature fitting is not yet supported.
    // The EnergyScaleTransmissionModel does not wire the temperature parameter.
    if config.fit_energy_scale && config.fit_temperature {
        return Err(PipelineError::InvalidParameter(
            "fit_energy_scale and fit_temperature cannot both be true: \
             EnergyScaleTransmissionModel does not support temperature fitting yet"
                .into(),
        ));
    }

    // Energy-scale parameters (SAMMY TZERO equivalent)
    let energy_scale_indices = if config.fit_energy_scale {
        let t0_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "t0_us".into(),
            value: config.t0_init_us,
            lower: -10.0,
            upper: 10.0,
            fixed: false,
        });
        let ls_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "l_scale".into(),
            value: config.l_scale_init,
            lower: 0.99,
            upper: 1.01,
            fixed: false,
        });
        Some((t0_idx, ls_idx))
    } else {
        None
    };

    // Background parameters
    let bg_indices = config
        .transmission_background
        .as_ref()
        .map(|bg| append_background_params(&mut param_vec, bg));

    let mut params = ParameterSet::new(param_vec);
    let mut lm_cfg = lm_config.clone();
    lm_cfg.compute_covariance = config.compute_covariance;

    // Build model — use EnergyScaleTransmissionModel when energy-scale is enabled
    let model: Box<dyn FitModel> = if let Some((t0_idx, ls_idx)) = energy_scale_indices {
        // Energy-scale model needs precomputed Doppler-broadened cross-sections.
        // Precompute them if not already available.
        let n_params = config.n_density_params();
        let xs = if let Some(xs) = &config.precomputed_cross_sections {
            Arc::clone(xs)
        } else {
            // Precompute Doppler-broadened σ(E) on the nominal energy grid.
            // Resolution is NOT applied here — it's done inside the model's evaluate().
            let instrument = config
                .resolution
                .clone()
                .map(|r| Arc::new(InstrumentParams { resolution: r }));
            let xs_raw = nereids_transmission::broadened_cross_sections(
                config.energies(),
                &config.resonance_data,
                config.temperature_k,
                instrument.as_deref(),
                None,
            )
            .map_err(PipelineError::Transmission)?;
            Arc::new(xs_raw)
        };
        // Collapse grouped isotopes if needed
        let effective_xs =
            if let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios) {
                if xs.len() == di.len() && di.len() == dr.len() {
                    let n_e = xs[0].len();
                    let mut eff = vec![vec![0.0f64; n_e]; n_params];
                    for ((&idx, &ratio), member_xs) in di.iter().zip(dr.iter()).zip(xs.iter()) {
                        for (j, &sigma) in member_xs.iter().enumerate() {
                            eff[idx][j] += ratio * sigma;
                        }
                    }
                    Arc::new(eff)
                } else {
                    xs
                }
            } else {
                xs
            };
        // After group-collapsing, effective_xs has n_params entries.
        // Use identity mapping because each XS entry maps to its own
        // density parameter (same as PrecomputedTransmissionModel line 1443).
        let density_indices: Vec<usize> = (0..n_params).collect();
        let instrument = config
            .resolution
            .clone()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        Box::new(EnergyScaleTransmissionModel::new(
            effective_xs,
            Arc::new(density_indices),
            config.energies.clone(),
            config.flight_path_m,
            t0_idx,
            ls_idx,
            instrument,
        ))
    } else {
        build_transmission_model(config, n_density_params, _temperature_index)?
    };

    // Dispatch with optional background wrapping
    let result = if let Some(bi) = bg_indices {
        let wrapped = if let (Some(di), Some(fi)) = (bi.back_d, bi.back_f) {
            NormalizedTransmissionModel::new_with_exponential(
                &*model,
                config.energies(),
                bi.anorm,
                bi.back_a,
                bi.back_b,
                bi.back_c,
                di,
                fi,
            )
        } else {
            NormalizedTransmissionModel::new(
                &*model,
                config.energies(),
                bi.anorm,
                bi.back_a,
                bi.back_b,
                bi.back_c,
            )
        };
        lm::levenberg_marquardt(&wrapped, measured_t, sigma, &mut params, &lm_cfg)?
    } else {
        lm::levenberg_marquardt(&*model, measured_t, sigma, &mut params, &lm_cfg)?
    };

    let mut sr = extract_result(config, &result, n_density_params, bg_indices)?;

    // Populate energy-scale results if fitted.
    if let Some((t0_idx, ls_idx)) = energy_scale_indices {
        sr.t0_us = Some(result.params[t0_idx]);
        sr.l_scale = Some(result.params[ls_idx]);
    }

    Ok(sr)
}

/// Transmission + Poisson KL path.
///
/// Uses the same model architecture as the LM path:
/// - `EnergyScaleTransmissionModel` when energy-scale fitting is enabled
/// - `NormalizedTransmissionModel` for SAMMY-style background (Anorm + BackA/B/C)
/// - Poisson NLL handles negative model predictions via smooth extrapolation
fn fit_transmission_poisson(
    measured_t: &[f64],
    sigma: &[f64],
    config: &UnifiedFitConfig,
    poisson_cfg: &PoissonConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let mut poisson_cfg = poisson_cfg.clone();
    poisson_cfg.compute_covariance = config.compute_covariance;
    let poisson_cfg = &poisson_cfg;

    let n_density_params = config.n_density_params();
    let mut param_vec = build_density_params(config);

    // Temperature parameter
    if config.fit_temperature && config.fit_energy_scale {
        return Err(PipelineError::InvalidParameter(
            "fit_energy_scale and fit_temperature cannot both be true: \
             EnergyScaleTransmissionModel does not support temperature fitting yet"
                .into(),
        ));
    }
    let temperature_index = if config.fit_temperature {
        let idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "temperature_k".into(),
            value: config.temperature_k,
            lower: 1.0,
            upper: 5000.0,
            fixed: false,
        });
        Some(idx)
    } else {
        None
    };

    // Energy-scale parameters (same as LM path)
    let energy_scale_indices = if config.fit_energy_scale {
        let t0_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "t0_us".into(),
            value: config.t0_init_us,
            lower: -10.0,
            upper: 10.0,
            fixed: false,
        });
        let ls_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "l_scale".into(),
            value: config.l_scale_init,
            lower: 0.99,
            upper: 1.01,
            fixed: false,
        });
        Some((t0_idx, ls_idx))
    } else {
        None
    };

    // Background parameters — use same SAMMY-style model as LM
    let bg_indices = config
        .transmission_background
        .as_ref()
        .map(|bg| append_background_params(&mut param_vec, bg));

    let mut params = ParameterSet::new(param_vec);

    // Build inner model (energy-scale or precomputed)
    let model: Box<dyn FitModel> = if let Some((t0_idx, ls_idx)) = energy_scale_indices {
        let n_params = config.n_density_params();
        let xs = if let Some(xs) = &config.precomputed_cross_sections {
            Arc::clone(xs)
        } else {
            let instrument = config
                .resolution
                .clone()
                .map(|r| Arc::new(InstrumentParams { resolution: r }));
            let xs_raw = nereids_transmission::broadened_cross_sections(
                config.energies(),
                &config.resonance_data,
                config.temperature_k,
                instrument.as_deref(),
                None,
            )
            .map_err(PipelineError::Transmission)?;
            Arc::new(xs_raw)
        };
        let effective_xs =
            if let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios) {
                if xs.len() == di.len() && di.len() == dr.len() {
                    let n_e = xs[0].len();
                    let mut eff = vec![vec![0.0f64; n_e]; n_params];
                    for ((&idx, &ratio), member_xs) in di.iter().zip(dr.iter()).zip(xs.iter()) {
                        for (j, &sigma_val) in member_xs.iter().enumerate() {
                            eff[idx][j] += ratio * sigma_val;
                        }
                    }
                    Arc::new(eff)
                } else {
                    xs
                }
            } else {
                xs
            };
        let density_indices: Vec<usize> = (0..n_params).collect();
        let instrument = config
            .resolution
            .clone()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        Box::new(EnergyScaleTransmissionModel::new(
            effective_xs,
            Arc::new(density_indices),
            config.energies.clone(),
            config.flight_path_m,
            t0_idx,
            ls_idx,
            instrument,
        ))
    } else {
        build_transmission_model(config, n_density_params, temperature_index)?
    };

    // Wrap with NormalizedTransmissionModel for background (same as LM)
    let result = if let Some(bi) = bg_indices {
        let wrapped = if let (Some(di), Some(fi)) = (bi.back_d, bi.back_f) {
            NormalizedTransmissionModel::new_with_exponential(
                &*model,
                config.energies(),
                bi.anorm,
                bi.back_a,
                bi.back_b,
                bi.back_c,
                di,
                fi,
            )
        } else {
            NormalizedTransmissionModel::new(
                &*model,
                config.energies(),
                bi.anorm,
                bi.back_a,
                bi.back_b,
                bi.back_c,
            )
        };
        let pr = poisson::poisson_fit(&wrapped, measured_t, &mut params, poisson_cfg)?;
        poisson_to_lm_result(&wrapped, measured_t, sigma, &pr, &params)
    } else {
        let pr = poisson::poisson_fit(&*model, measured_t, &mut params, poisson_cfg)?;
        poisson_to_lm_result(&*model, measured_t, sigma, &pr, &params)
    }?;

    let mut sr = extract_result(config, &result, n_density_params, bg_indices)?;

    // Populate temperature
    if let Some(idx) = temperature_index {
        sr.temperature_k = Some(result.params[idx]);
        sr.temperature_k_unc = temperature_index.and_then(|i| {
            result
                .uncertainties
                .as_ref()
                .and_then(|u| u.get(i).copied())
        });
    }

    // Populate energy-scale results
    if let Some((t0_idx, ls_idx)) = energy_scale_indices {
        sr.t0_us = Some(result.params[t0_idx]);
        sr.l_scale = Some(result.params[ls_idx]);
    }

    Ok(sr)
}

/// Counts + Poisson KL path (statistically optimal).
///
/// Uses the same model architecture as the LM and KL-transmission paths:
/// - `EnergyScaleTransmissionModel` when energy-scale fitting is enabled
/// - `NormalizedTransmissionModel` for SAMMY-style background (Anorm + BackA/B/C)
/// - `CountsModel` / `CountsBackgroundScaleModel` for the counts wrapper
fn fit_counts_poisson(
    sample_counts: &[f64],
    flux: &[f64],
    background: &[f64],
    config: &UnifiedFitConfig,
    poisson_cfg: &PoissonConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let mut poisson_cfg = poisson_cfg.clone();
    poisson_cfg.compute_covariance = config.compute_covariance;
    let poisson_cfg = &poisson_cfg;

    let n_density_params = config.n_density_params();
    let mut param_vec = build_density_params(config);

    // Temperature parameter
    if config.fit_temperature && config.fit_energy_scale {
        return Err(PipelineError::InvalidParameter(
            "fit_energy_scale and fit_temperature cannot both be true: \
             EnergyScaleTransmissionModel does not support temperature fitting yet"
                .into(),
        ));
    }
    let temperature_index = if config.fit_temperature {
        let idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "temperature_k".into(),
            value: config.temperature_k,
            lower: 1.0,
            upper: 5000.0,
            fixed: false,
        });
        Some(idx)
    } else {
        None
    };

    // Energy-scale parameters (same as LM/KL-transmission paths)
    let energy_scale_indices = if config.fit_energy_scale {
        let t0_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "t0_us".into(),
            value: config.t0_init_us,
            lower: -10.0,
            upper: 10.0,
            fixed: false,
        });
        let ls_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "l_scale".into(),
            value: config.l_scale_init,
            lower: 0.99,
            upper: 1.01,
            fixed: false,
        });
        Some((t0_idx, ls_idx))
    } else {
        None
    };

    // Transmission background — use same SAMMY-style model as LM
    let bg_indices = config
        .transmission_background
        .as_ref()
        .map(|bg| append_background_params(&mut param_vec, bg));

    // Counts-domain background (alpha_1, alpha_2 scaling)
    let counts_bg = if let Some(bg) = config.counts_background() {
        if bg.fit_alpha_1 && flux.iter().all(|&v| v.abs() <= 1e-12) {
            return Err(PipelineError::InvalidParameter(
                "counts background alpha_1 cannot be fitted with zero flux reference".into(),
            ));
        }
        if bg.fit_alpha_2 && background.iter().all(|&v| v.abs() <= 1e-12) {
            return Err(PipelineError::InvalidParameter(
                "counts background alpha_2 cannot be fitted with zero detector background reference"
                    .into(),
            ));
        }

        let alpha1_idx = param_vec.len();
        param_vec.push(if bg.fit_alpha_1 {
            FitParameter {
                name: "alpha_1".into(),
                value: bg.alpha_1_init,
                lower: 0.0,
                upper: 10.0,
                fixed: false,
            }
        } else {
            FitParameter::fixed("alpha_1", bg.alpha_1_init)
        });

        let alpha2_idx = param_vec.len();
        param_vec.push(if bg.fit_alpha_2 {
            FitParameter {
                name: "alpha_2".into(),
                value: bg.alpha_2_init,
                lower: 0.0,
                upper: 10.0,
                fixed: false,
            }
        } else {
            FitParameter::fixed("alpha_2", bg.alpha_2_init)
        });

        Some((alpha1_idx, alpha2_idx))
    } else {
        None
    };

    let mut params = ParameterSet::new(param_vec);

    // Build inner transmission model (energy-scale or precomputed)
    let t_model: Box<dyn FitModel> = if let Some((t0_idx, ls_idx)) = energy_scale_indices {
        let n_params = config.n_density_params();
        let xs = if let Some(xs) = &config.precomputed_cross_sections {
            Arc::clone(xs)
        } else {
            let instrument = config
                .resolution
                .clone()
                .map(|r| Arc::new(InstrumentParams { resolution: r }));
            let xs_raw = nereids_transmission::broadened_cross_sections(
                config.energies(),
                &config.resonance_data,
                config.temperature_k,
                instrument.as_deref(),
                None,
            )
            .map_err(PipelineError::Transmission)?;
            Arc::new(xs_raw)
        };
        let effective_xs =
            if let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios) {
                if xs.len() == di.len() && di.len() == dr.len() {
                    let n_e = xs[0].len();
                    let mut eff = vec![vec![0.0f64; n_e]; n_params];
                    for ((&idx, &ratio), member_xs) in di.iter().zip(dr.iter()).zip(xs.iter()) {
                        for (j, &sigma_val) in member_xs.iter().enumerate() {
                            eff[idx][j] += ratio * sigma_val;
                        }
                    }
                    Arc::new(eff)
                } else {
                    xs
                }
            } else {
                xs
            };
        let density_indices: Vec<usize> = (0..n_params).collect();
        let instrument = config
            .resolution
            .clone()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        Box::new(EnergyScaleTransmissionModel::new(
            effective_xs,
            Arc::new(density_indices),
            config.energies.clone(),
            config.flight_path_m,
            t0_idx,
            ls_idx,
            instrument,
        ))
    } else {
        build_transmission_model(config, n_density_params, temperature_index)?
    };

    let n_free = params.n_free();
    let dof = sample_counts.len().saturating_sub(n_free).max(1);

    // Build the model chain: transmission → background → counts
    // The transmission background wraps the inner model with Anorm + BackA/B/C
    // The counts model wraps that with flux * T + detector_background
    let (pr, y_model) = if let Some(bi) = bg_indices {
        let wrapped = if let (Some(di), Some(fi)) = (bi.back_d, bi.back_f) {
            NormalizedTransmissionModel::new_with_exponential(
                &*t_model,
                config.energies(),
                bi.anorm,
                bi.back_a,
                bi.back_b,
                bi.back_c,
                di,
                fi,
            )
        } else {
            NormalizedTransmissionModel::new(
                &*t_model,
                config.energies(),
                bi.anorm,
                bi.back_a,
                bi.back_b,
                bi.back_c,
            )
        };
        if let Some((alpha1_idx, alpha2_idx)) = counts_bg {
            let counts_model = poisson::CountsBackgroundScaleModel {
                transmission_model: &wrapped,
                flux,
                background,
                alpha1_index: alpha1_idx,
                alpha2_index: alpha2_idx,
                n_params: params.params.len(),
            };
            let pr = poisson::poisson_fit(&counts_model, sample_counts, &mut params, poisson_cfg)?;
            let y_model = counts_model.evaluate(&pr.params)?;
            (pr, y_model)
        } else {
            let counts_model = poisson::CountsModel {
                transmission_model: &wrapped,
                flux,
                background,
                n_params: params.params.len(),
            };
            let pr = poisson::poisson_fit(&counts_model, sample_counts, &mut params, poisson_cfg)?;
            let y_model = counts_model.evaluate(&pr.params)?;
            (pr, y_model)
        }
    } else {
        if let Some((alpha1_idx, alpha2_idx)) = counts_bg {
            let counts_model = poisson::CountsBackgroundScaleModel {
                transmission_model: &*t_model,
                flux,
                background,
                alpha1_index: alpha1_idx,
                alpha2_index: alpha2_idx,
                n_params: params.params.len(),
            };
            let pr = poisson::poisson_fit(&counts_model, sample_counts, &mut params, poisson_cfg)?;
            let y_model = counts_model.evaluate(&pr.params)?;
            (pr, y_model)
        } else {
            let counts_model = poisson::CountsModel {
                transmission_model: &*t_model,
                flux,
                background,
                n_params: params.params.len(),
            };
            let pr = poisson::poisson_fit(&counts_model, sample_counts, &mut params, poisson_cfg)?;
            let y_model = counts_model.evaluate(&pr.params)?;
            (pr, y_model)
        }
    };

    // Pearson chi-squared for counts domain: Σ (obs - model)² / model
    let chi_sq: f64 = sample_counts
        .iter()
        .zip(y_model.iter())
        .map(|(&obs, &mdl)| {
            let expected = mdl.max(1e-10);
            (obs - expected).powi(2) / expected
        })
        .sum();

    let densities: Vec<f64> = (0..n_density_params).map(|i| pr.params[i]).collect();
    let fitted_temp = temperature_index.map(|idx| pr.params[idx]);

    let (anorm, bg_array, back_d, back_f) = if let Some(bi) = bg_indices {
        let bd = bi.back_d.map_or(0.0, |i| pr.params[i]);
        let bf = bi.back_f.map_or(0.0, |i| pr.params[i]);
        (
            pr.params[bi.anorm],
            [
                pr.params[bi.back_a],
                pr.params[bi.back_b],
                pr.params[bi.back_c],
            ],
            bd,
            bf,
        )
    } else {
        let fitted_alpha1 = counts_bg.map_or(1.0, |(a1, _)| pr.params[a1]);
        let fitted_alpha2 = counts_bg.map_or(0.0, |(_, a2)| pr.params[a2]);
        (fitted_alpha1, [0.0, 0.0, fitted_alpha2], 0.0, 0.0)
    };

    let (uncertainties, temperature_k_unc) = if let Some(ref unc_all) = pr.uncertainties {
        let dens_unc: Vec<f64> = (0..n_density_params)
            .map(|i| *unc_all.get(i).unwrap_or(&f64::NAN))
            .collect();
        let t_unc = temperature_index.and_then(|idx| {
            params
                .free_indices()
                .iter()
                .position(|&fi| fi == idx)
                .and_then(|pos| unc_all.get(pos).copied())
        });
        (Some(dens_unc), t_unc)
    } else {
        (None, None)
    };

    Ok(SpectrumFitResult {
        densities,
        uncertainties,
        reduced_chi_squared: chi_sq / dof as f64,
        converged: pr.converged,
        iterations: pr.iterations,
        temperature_k: fitted_temp,
        temperature_k_unc,
        anorm,
        background: bg_array,
        back_d,
        back_f,
        t0_us: energy_scale_indices.map(|(t0_idx, _)| pr.params[t0_idx]),
        l_scale: energy_scale_indices.map(|(_, ls_idx)| pr.params[ls_idx]),
        deviance_per_dof: None,
    })
}

/// Joint-Poisson counts-path fitter (memo 35 §P1/§P2).
///
/// Builds a pure transmission `FitModel` (density + optional temperature +
/// optional energy-scale) and feeds it to [`joint_poisson::joint_poisson_fit`]
/// together with explicit `(O, S, c)`.  Returns a [`SpectrumFitResult`] with
/// `deviance_per_dof = Some(...)` as the primary GOF (memo 35 §P1.2).
/// `reduced_chi_squared` is set to the same value so GUI consumers that
/// still read the legacy field see a deviance-based metric.
///
/// Current scope (P1 + P2, including P2.2): `fit_alpha_1`, `fit_alpha_2`,
/// and non-zero `detector_background` remain rejected (`λ̂` absorbs the
/// global flux scale, `B_det` / alpha_2 wiring is memo 35 §P3.2 deferred).
/// `transmission_background` with `A_n` + `B_A` / `B_B` / `B_C` is
/// supported as of P2.2, subject to the operational rule that `B_A` must
/// be enabled if any of `B_A` / `B_B` / `B_C` is enabled (memo 35 §P2.2,
/// EG2 S2 C_An shows A_n alone cannot absorb a constant offset — density
/// bias −23%).  Exponential-tail terms `BackD` / `BackF` are rejected
/// (memo 35 §P4-deferred).
fn fit_counts_joint_poisson(
    sample_counts: &[f64],
    flux: &[f64],
    detector_background: &[f64],
    config: &UnifiedFitConfig,
    jp_cfg: &JointPoissonFitConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    // ── Compatibility gates (memo 35 §P3 items are out of scope here) ──
    if let Some(bg) = config.counts_background()
        && (bg.fit_alpha_1 || bg.fit_alpha_2)
    {
        return Err(PipelineError::InvalidParameter(
            "joint-Poisson solver does not support fit_alpha_1/fit_alpha_2: \
             the profile lambda-hat absorbs the global flux scale (alpha_1 redundant); \
             alpha_2 / B_det wiring is deferred to memo 35 §P3."
                .into(),
        ));
    }
    if detector_background.iter().any(|&v| v.abs() > 1e-12) {
        return Err(PipelineError::InvalidParameter(
            "joint-Poisson solver with non-zero detector_background is not yet supported \
             (B_det wiring deferred to memo 35 §P3.2)."
                .into(),
        ));
    }

    // ── §P2.2 operational rule: B_A required if any additive term enabled ──
    if let Some(bg) = config.transmission_background.as_ref() {
        if bg.fit_back_d || bg.fit_back_f {
            return Err(PipelineError::InvalidParameter(
                "joint-Poisson solver does not support the BackD/BackF exponential \
                 tail (memo 35 §P4-deferred)."
                    .into(),
            ));
        }
        if (bg.fit_back_b || bg.fit_back_c) && !bg.fit_back_a {
            return Err(PipelineError::InvalidParameter(
                "joint-Poisson transmission_background: B_A (fit_back_a) must be \
                 enabled whenever any of B_B / B_C is enabled (memo 35 §P2.2 — \
                 A_n alone cannot absorb a constant offset; EG2 S2 C_An → −23% \
                 density bias)."
                    .into(),
            ));
        }
    }

    let c = config.counts_background().map(|b| b.c).unwrap_or(1.0);
    if !(c.is_finite() && c > 0.0) {
        return Err(PipelineError::InvalidParameter(format!(
            "joint-Poisson solver requires finite c > 0 in CountsBackgroundConfig, got {c}",
        )));
    }

    // ── Build parameter vector (mirrors fit_counts_poisson structure) ──
    let n_density_params = config.n_density_params();
    let mut param_vec = build_density_params(config);

    if config.fit_temperature && config.fit_energy_scale {
        return Err(PipelineError::InvalidParameter(
            "fit_energy_scale and fit_temperature cannot both be true: \
             EnergyScaleTransmissionModel does not support temperature fitting yet"
                .into(),
        ));
    }
    let temperature_index = if config.fit_temperature {
        let idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "temperature_k".into(),
            value: config.temperature_k,
            lower: 1.0,
            upper: 5000.0,
            fixed: false,
        });
        Some(idx)
    } else {
        None
    };

    let energy_scale_indices = if config.fit_energy_scale {
        let t0_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "t0_us".into(),
            value: config.t0_init_us,
            lower: -10.0,
            upper: 10.0,
            fixed: false,
        });
        let ls_idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "l_scale".into(),
            value: config.l_scale_init,
            lower: 0.99,
            upper: 1.01,
            fixed: false,
        });
        Some((t0_idx, ls_idx))
    } else {
        None
    };

    // ── Transmission background (A_n + B_A/B/C) parameters, P2.2 ──
    // Use the same SAMMY-style param block as the LM transmission path.
    // If BackD/BackF were enabled, we would have already errored out above.
    let bg_indices = config
        .transmission_background
        .as_ref()
        .map(|bg| append_background_params(&mut param_vec, bg));

    let mut params = ParameterSet::new(param_vec);

    // ── Build pure transmission model ──
    let t_model: Box<dyn FitModel> = if let Some((t0_idx, ls_idx)) = energy_scale_indices {
        let n_params = config.n_density_params();
        let xs = if let Some(xs) = &config.precomputed_cross_sections {
            Arc::clone(xs)
        } else {
            let instrument = config
                .resolution
                .clone()
                .map(|r| Arc::new(InstrumentParams { resolution: r }));
            let xs_raw = nereids_transmission::broadened_cross_sections(
                config.energies(),
                &config.resonance_data,
                config.temperature_k,
                instrument.as_deref(),
                None,
            )
            .map_err(PipelineError::Transmission)?;
            Arc::new(xs_raw)
        };
        let effective_xs =
            if let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios) {
                if xs.len() == di.len() && di.len() == dr.len() {
                    let n_e = xs[0].len();
                    let mut eff = vec![vec![0.0f64; n_e]; n_params];
                    for ((&idx, &ratio), member_xs) in di.iter().zip(dr.iter()).zip(xs.iter()) {
                        for (j, &sigma_val) in member_xs.iter().enumerate() {
                            eff[idx][j] += ratio * sigma_val;
                        }
                    }
                    Arc::new(eff)
                } else {
                    xs
                }
            } else {
                xs
            };
        let density_indices: Vec<usize> = (0..n_params).collect();
        let instrument = config
            .resolution
            .clone()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        Box::new(EnergyScaleTransmissionModel::new(
            effective_xs,
            Arc::new(density_indices),
            config.energies.clone(),
            config.flight_path_m,
            t0_idx,
            ls_idx,
            instrument,
        ))
    } else {
        build_transmission_model(config, n_density_params, temperature_index)?
    };

    // ── Wrap with NormalizedTransmissionModel if bg is active (P2.2) ──
    // The wrapper adds `T_out = A_n · T_inner + B_A + B_B/√E + B_C·√E`,
    // exactly matching the SAMMY form used by the LM transmission path.
    // Its analytical Jacobian chains through the inner model correctly,
    // so JointPoissonObjective picks up gradients for both density and
    // background parameters without further wiring.
    let result;
    if let Some(bi) = bg_indices {
        let wrapped = NormalizedTransmissionModel::new(
            &*t_model,
            config.energies(),
            bi.anorm,
            bi.back_a,
            bi.back_b,
            bi.back_c,
        );
        let objective = JointPoissonObjective {
            model: &wrapped,
            o: flux,
            s: sample_counts,
            c,
        };
        let mut cfg = jp_cfg.clone();
        cfg.compute_covariance = config.compute_covariance;
        result = joint_poisson::joint_poisson_fit(&objective, &mut params, &cfg).map_err(|e| {
            PipelineError::InvalidParameter(format!("joint-Poisson fit failed: {e}"))
        })?;
    } else {
        let objective = JointPoissonObjective {
            model: &*t_model,
            o: flux,
            s: sample_counts,
            c,
        };
        let mut cfg = jp_cfg.clone();
        cfg.compute_covariance = config.compute_covariance;
        result = joint_poisson::joint_poisson_fit(&objective, &mut params, &cfg).map_err(|e| {
            PipelineError::InvalidParameter(format!("joint-Poisson fit failed: {e}"))
        })?;
    }

    // ── Extract fitted quantities ──
    let densities: Vec<f64> = (0..n_density_params).map(|i| result.params[i]).collect();

    let (uncertainties, temperature_k_unc) = if let Some(ref unc_all) = result.uncertainties {
        let dens_unc: Vec<f64> = (0..n_density_params)
            .map(|i| *unc_all.get(i).unwrap_or(&f64::NAN))
            .collect();
        let t_unc = temperature_index.and_then(|idx| {
            params
                .free_indices()
                .iter()
                .position(|&fi| fi == idx)
                .and_then(|pos| unc_all.get(pos).copied())
        });
        (Some(dens_unc), t_unc)
    } else {
        (None, None)
    };
    let fitted_temp = temperature_index.map(|idx| result.params[idx]);

    // Convergence signal per memo 35 §P2.3: the deviance value is the
    // acceptance criterion, but we expose a boolean to preserve the
    // existing SpectrumFitResult shape.  Report True when EITHER stage
    // self-flagged convergence (whichever accepts).
    let converged = result.gn_converged || result.polish_converged;

    // ── Background parameter readout ──
    // When bg is active, read A_n / B_A / B_B / B_C from the fitted
    // parameter vector at their registered indices.  When bg is absent,
    // use the memo 35 §P1 convention: A_n = 1 (subsumed into λ̂), bg = 0.
    let (anorm_out, bg_abc_out) = if let Some(bi) = bg_indices {
        (
            result.params[bi.anorm],
            [
                result.params[bi.back_a],
                result.params[bi.back_b],
                result.params[bi.back_c],
            ],
        )
    } else {
        (1.0, [0.0, 0.0, 0.0])
    };

    Ok(SpectrumFitResult {
        densities,
        uncertainties,
        // Back-compat bridge: reduced_chi_squared carries D/(n−k) for the
        // joint-Poisson path.  Memo 35 §P1.2 — Pearson χ² is secondary.
        reduced_chi_squared: result.deviance_per_dof,
        converged,
        iterations: result.gn_iterations + result.polish_iterations,
        temperature_k: fitted_temp,
        temperature_k_unc,
        anorm: anorm_out,
        background: bg_abc_out,
        back_d: 0.0,
        back_f: 0.0,
        t0_us: energy_scale_indices.map(|(t0_idx, _)| result.params[t0_idx]),
        l_scale: energy_scale_indices.map(|(_, ls_idx)| result.params[ls_idx]),
        deviance_per_dof: Some(result.deviance_per_dof),
    })
}

// ── Shared helpers for fit_spectrum_typed ──

fn build_density_params(config: &UnifiedFitConfig) -> Vec<FitParameter> {
    config
        .initial_densities
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            FitParameter::non_negative(
                config
                    .isotope_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("isotope_{i}")),
                d,
            )
        })
        .collect()
}

fn append_background_params(
    param_vec: &mut Vec<FitParameter>,
    bg: &BackgroundConfig,
) -> BackgroundIndices {
    // Anorm bounded to [0.5, 2.0] — physically reasonable normalization range.
    // Previously unbounded [0, ∞), which allowed the fitter to absorb signal
    // into anorm (e.g., anorm=15.9 with density=0.03×true).
    // SAMMY also bounds normalization to a reasonable range.
    let anorm = param_vec.len();
    param_vec.push(if bg.fit_anorm {
        FitParameter {
            name: "anorm".into(),
            value: bg.anorm_init,
            lower: 0.5,
            upper: 2.0,
            fixed: false,
        }
    } else {
        FitParameter::fixed("anorm", bg.anorm_init)
    });
    // Background terms bounded to [-0.5, 0.5].
    // These are small corrections to the transmission baseline.
    // Unbounded background allows the fitter to absorb resonance signal
    // into the background polynomial, producing meaningless densities.
    // SAMMY also constrains background to reasonable ranges.
    let back_a = param_vec.len();
    param_vec.push(if bg.fit_back_a {
        FitParameter {
            name: "back_a".into(),
            value: bg.back_a_init,
            lower: -0.5,
            upper: 0.5,
            fixed: false,
        }
    } else {
        FitParameter::fixed("back_a", bg.back_a_init)
    });
    let back_b = param_vec.len();
    param_vec.push(if bg.fit_back_b {
        FitParameter {
            name: "back_b".into(),
            value: bg.back_b_init,
            lower: -0.5,
            upper: 0.5,
            fixed: false,
        }
    } else {
        FitParameter::fixed("back_b", bg.back_b_init)
    });
    let back_c = param_vec.len();
    param_vec.push(if bg.fit_back_c {
        FitParameter {
            name: "back_c".into(),
            value: bg.back_c_init,
            lower: -0.5,
            upper: 0.5,
            fixed: false,
        }
    } else {
        FitParameter::fixed("back_c", bg.back_c_init)
    });

    // Exponential tail: BackD × exp(−BackF / √E).
    // SAMMY manual Sec III.E.2 — terms 5-6.
    // BackD (amplitude): non-negative, bounded [0, 1].
    // BackF (decay constant in √eV units): non-negative, bounded [0, 100].
    // Note: if BackD_init = 0, the Jacobian column for BackF is identically
    // zero and the optimizer cannot learn BackF.  The default init (0.01)
    // avoids this.
    let back_d = if bg.fit_back_d {
        let idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "back_d".into(),
            value: bg.back_d_init,
            lower: 0.0,
            upper: 1.0,
            fixed: false,
        });
        Some(idx)
    } else {
        None
    };
    let back_f = if bg.fit_back_f {
        let idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "back_f".into(),
            value: bg.back_f_init,
            lower: 0.0,
            upper: 100.0,
            fixed: false,
        });
        Some(idx)
    } else {
        None
    };

    BackgroundIndices {
        anorm,
        back_a,
        back_b,
        back_c,
        back_d,
        back_f,
    }
}

/// Build the transmission forward model, selecting precomputed or full path.
fn build_transmission_model(
    config: &UnifiedFitConfig,
    n_density_params: usize,
    temperature_index: Option<usize>,
) -> Result<Box<dyn FitModel>, PipelineError> {
    let n_params = config.n_density_params();
    if !config.fit_temperature
        && let Some(xs) = &config.precomputed_cross_sections
    {
        // When groups are active, compute σ_eff per group from member XS.
        // For ungrouped isotopes, this is a no-op (identity mapping, ratio=1.0).
        let effective_xs =
            if let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios) {
                // Only collapse when XS is per-member (shape matches mapping).
                // If XS is already group-collapsed (len == n_params), skip.
                if xs.len() == di.len() && di.len() == dr.len() {
                    let n_e = xs[0].len();
                    let mut eff = vec![vec![0.0f64; n_e]; n_params];
                    for ((&idx, &ratio), member_xs) in di.iter().zip(dr.iter()).zip(xs.iter()) {
                        for (j, &sigma) in member_xs.iter().enumerate() {
                            eff[idx][j] += ratio * sigma;
                        }
                    }
                    Arc::new(eff)
                } else {
                    Arc::clone(xs)
                }
            } else {
                Arc::clone(xs)
            };
        // Issue #442: pass energies + instrument so evaluate() applies
        // resolution after Beer-Lambert on total transmission.
        let instrument = config
            .resolution
            .clone()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        return Ok(Box::new(PrecomputedTransmissionModel {
            cross_sections: effective_xs,
            density_indices: Arc::new((0..n_params).collect()),
            energies: instrument
                .as_ref()
                .map(|_| Arc::new(config.energies.clone())),
            instrument,
        }));
    }

    let instrument = config
        .resolution
        .clone()
        .map(|r| Arc::new(InstrumentParams { resolution: r }));

    let base_xs = config.precomputed_base_xs.clone();
    let density_ratios = config
        .density_ratios
        .clone()
        .unwrap_or_else(|| vec![1.0; n_density_params]);
    let density_indices = config
        .density_indices
        .clone()
        .unwrap_or_else(|| (0..n_density_params).collect());
    Ok(Box::new(TransmissionFitModel::new(
        config.energies.clone(),
        config.resonance_data.clone(),
        config.temperature_k,
        instrument,
        (density_indices, density_ratios),
        temperature_index,
        base_xs,
    )?))
}

/// Convert PoissonResult to LmResult with Pearson chi-squared.
fn poisson_to_lm_result(
    model: &dyn FitModel,
    measured_t: &[f64],
    sigma: &[f64],
    pr: &poisson::PoissonResult,
    params: &ParameterSet,
) -> Result<LmResult, PipelineError> {
    let n_free = params.n_free();
    let dof = measured_t.len().saturating_sub(n_free).max(1);
    let y_model = model.evaluate(&pr.params)?;
    let chi_sq: f64 = measured_t
        .iter()
        .zip(y_model.iter())
        .zip(sigma.iter())
        .map(|((obs, mdl), s)| {
            let residual = obs - mdl;
            (residual * residual) / (s * s).max(1e-30)
        })
        .sum();
    Ok(LmResult {
        chi_squared: chi_sq,
        reduced_chi_squared: chi_sq / dof as f64,
        iterations: pr.iterations,
        converged: pr.converged,
        params: pr.params.clone(),
        covariance: pr.covariance.clone(),
        uncertainties: pr.uncertainties.clone(),
    })
}

/// Extract SpectrumFitResult from solver output.
fn extract_result(
    config: &UnifiedFitConfig,
    result: &LmResult,
    n_density_params: usize,
    bg_indices: Option<BackgroundIndices>,
) -> Result<SpectrumFitResult, PipelineError> {
    let densities: Vec<f64> = (0..n_density_params).map(|i| result.params[i]).collect();

    let (anorm, background, back_d, back_f) = if let Some(bi) = bg_indices {
        let bd = bi.back_d.map_or(0.0, |i| result.params[i]);
        let bf = bi.back_f.map_or(0.0, |i| result.params[i]);
        (
            result.params[bi.anorm],
            [
                result.params[bi.back_a],
                result.params[bi.back_b],
                result.params[bi.back_c],
            ],
            bd,
            bf,
        )
    } else {
        (1.0, [0.0, 0.0, 0.0], 0.0, 0.0)
    };

    let (uncertainties, temperature_k, temperature_k_unc) = if result.converged {
        match &result.uncertainties {
            Some(unc_all) => {
                let (temp_k, temp_unc) = if config.fit_temperature {
                    (
                        Some(result.params[n_density_params]),
                        Some(*unc_all.get(n_density_params).unwrap_or(&f64::NAN)),
                    )
                } else {
                    (None, None)
                };
                let unc = unc_all
                    .get(..n_density_params)
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![f64::NAN; n_density_params]);
                (Some(unc), temp_k, temp_unc)
            }
            None => {
                let temp_k = if config.fit_temperature {
                    Some(result.params[n_density_params])
                } else {
                    None
                };
                (None, temp_k, None)
            }
        }
    } else {
        let temp_k = if config.fit_temperature {
            Some(result.params[n_density_params])
        } else {
            None
        };
        (None, temp_k, None)
    };

    Ok(SpectrumFitResult {
        densities,
        uncertainties,
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
        temperature_k,
        temperature_k_unc,
        anorm,
        background,
        back_d,
        back_f,
        t0_us: None,
        l_scale: None,
        deviance_per_dof: None,
    })
}

// ── Research: exact Jacobian/Fisher at arbitrary parameters ──────────────

/// Result of Jacobian/Fisher evaluation at given parameters.
///
/// Produced by [`evaluate_jacobian_and_fisher`], which builds the same model
/// chain as the production fitting pipeline but evaluates at the caller's
/// parameter values instead of optimising.
pub struct ModelJacobianResult {
    /// Analytical Jacobian J (n_data × n_free), row-major.
    pub jacobian: lm::FlatMatrix,
    /// Expected Poisson Fisher F = Jᵀ diag(1/μ) J (n_free × n_free).
    pub fisher: lm::FlatMatrix,
    /// Model prediction μ(E) at the evaluation point.
    pub model_prediction: Vec<f64>,
    /// Names of free parameters, in Jacobian column order.
    pub param_names: Vec<String>,
}

/// Evaluate the exact resolved analytical Jacobian and expected Poisson Fisher
/// at given parameter values, using the same model construction as the
/// production counts-domain fitting pipeline.
///
/// This is a research-oriented function: it builds the full model chain
/// (transmission model → optional background wrappers → counts model),
/// evaluates once at the provided parameters, computes the analytical
/// Jacobian, and assembles the expected Fisher information matrix.
///
/// No optimisation is performed.
///
/// # Arguments
///
/// * `config` — Unified fit configuration (energies, resonance data,
///   resolution, initial_densities used as evaluation densities, etc.)
/// * `flux` — Open-beam counts Φ(E) (length = n_energy)
/// * `background` — Detector background B(E) (length = n_energy, zeros if none)
///
/// Density evaluation values come from `config.initial_densities`.
/// Temperature evaluation value comes from `config.temperature_k`.
/// α₁/α₂ evaluation values come from `config.counts_background` init fields.
pub fn evaluate_jacobian_and_fisher(
    config: &UnifiedFitConfig,
    flux: &[f64],
    background: &[f64],
) -> Result<ModelJacobianResult, PipelineError> {
    let n_density_params = config.n_density_params();

    // ── Build parameter vector (mirrors fit_counts_poisson) ─────────
    let mut param_vec = build_density_params(config);

    let temperature_index = if config.fit_temperature {
        let idx = param_vec.len();
        param_vec.push(FitParameter {
            name: "temperature_k".into(),
            value: config.temperature_k,
            lower: 1.0,
            upper: 5000.0,
            fixed: false,
        });
        Some(idx)
    } else {
        None
    };

    let kl_bg = if config.transmission_background.is_some() {
        let base = param_vec.len();
        param_vec.push(FitParameter {
            name: "kl_b0".into(),
            value: 0.0,
            lower: 0.0,
            upper: 0.5,
            fixed: false,
        });
        param_vec.push(FitParameter {
            name: "kl_b1".into(),
            value: 0.0,
            lower: 0.0,
            upper: 0.5,
            fixed: false,
        });
        Some((base, base + 1))
    } else {
        None
    };

    let counts_bg = if let Some(bg) = config.counts_background() {
        let alpha1_idx = param_vec.len();
        param_vec.push(if bg.fit_alpha_1 {
            FitParameter {
                name: "alpha_1".into(),
                value: bg.alpha_1_init,
                lower: 0.0,
                upper: 10.0,
                fixed: false,
            }
        } else {
            FitParameter::fixed("alpha_1", bg.alpha_1_init)
        });
        let alpha2_idx = param_vec.len();
        param_vec.push(if bg.fit_alpha_2 {
            FitParameter {
                name: "alpha_2".into(),
                value: bg.alpha_2_init,
                lower: 0.0,
                upper: 10.0,
                fixed: false,
            }
        } else {
            FitParameter::fixed("alpha_2", bg.alpha_2_init)
        });
        Some((alpha1_idx, alpha2_idx))
    } else {
        None
    };

    let params = ParameterSet::new(param_vec);
    let all_vals = params.all_values();
    let free_idx = params.free_indices();
    let n_free = free_idx.len();

    // Collect free parameter names.
    let param_names: Vec<String> = free_idx
        .iter()
        .map(|&i| params.params[i].name.to_string())
        .collect();

    // ── Precompute cross-sections so that analytical Jacobian is available ──
    // For the density-only case (no temperature fitting), the model uses
    // PrecomputedTransmissionModel which requires precomputed XS.
    // For the temperature case, TransmissionFitModel computes base_xs in
    // its constructor.  Either way, precomputing here ensures the analytical
    // Jacobian path is always available.
    let config_with_xs;
    let effective_config = if config.precomputed_cross_sections.is_none() && !config.fit_temperature
    {
        let instrument = config
            .resolution
            .clone()
            .map(|r| Arc::new(InstrumentParams { resolution: r }));
        let xs = nereids_physics::transmission::broadened_cross_sections(
            config.energies(),
            &config.resonance_data,
            config.temperature_k,
            instrument.as_deref(),
            None,
        )
        .map_err(PipelineError::Transmission)?;
        config_with_xs = config.clone().with_precomputed_cross_sections(Arc::new(xs));
        &config_with_xs
    } else {
        config
    };

    // ── Build transmission model (same as production path) ──────────
    let t_model = build_transmission_model(effective_config, n_density_params, temperature_index)?;

    // ── Build counts model chain and evaluate ───────────────────────
    // Use a closure that evaluates and computes Jacobian for any FitModel.
    let evaluate_and_jacobian =
        |model: &dyn FitModel| -> Result<(Vec<f64>, lm::FlatMatrix), PipelineError> {
            let y_model = model.evaluate(&all_vals)?;
            let jac = model
                .analytical_jacobian(&all_vals, &free_idx, &y_model)
                .ok_or_else(|| {
                    PipelineError::InvalidParameter(
                        "analytical Jacobian not available for this model configuration".into(),
                    )
                })?;
            Ok((y_model, jac))
        };

    let (y_model, jac) = if let Some((b0_idx, b1_idx)) = kl_bg {
        let inv_sqrt_e: Vec<f64> = config
            .energies()
            .iter()
            .map(|&e| 1.0 / e.max(1e-10).sqrt())
            .collect();
        let wrapped = poisson::TransmissionKLBackgroundModel {
            inner: &*t_model,
            inv_sqrt_energies: inv_sqrt_e,
            b0_index: b0_idx,
            b1_index: b1_idx,
            n_params: params.params.len(),
        };
        if let Some((a1, a2)) = counts_bg {
            let cm = poisson::CountsBackgroundScaleModel {
                transmission_model: &wrapped,
                flux,
                background,
                alpha1_index: a1,
                alpha2_index: a2,
                n_params: params.params.len(),
            };
            evaluate_and_jacobian(&cm)?
        } else {
            let cm = poisson::CountsModel {
                transmission_model: &wrapped,
                flux,
                background,
                n_params: params.params.len(),
            };
            evaluate_and_jacobian(&cm)?
        }
    } else if let Some((a1, a2)) = counts_bg {
        let cm = poisson::CountsBackgroundScaleModel {
            transmission_model: &*t_model,
            flux,
            background,
            alpha1_index: a1,
            alpha2_index: a2,
            n_params: params.params.len(),
        };
        evaluate_and_jacobian(&cm)?
    } else {
        let cm = poisson::CountsModel {
            transmission_model: &*t_model,
            flux,
            background,
            n_params: params.params.len(),
        };
        evaluate_and_jacobian(&cm)?
    };

    // ── Assemble expected Poisson Fisher: F = Jᵀ diag(1/μ) J ───────
    let mut fisher = lm::FlatMatrix::zeros(n_free, n_free);
    for (i, &mu_i) in y_model.iter().enumerate() {
        let mu_inv = 1.0 / mu_i.max(1e-30);
        for a in 0..n_free {
            let ja = jac.get(i, a);
            for b in 0..=a {
                let jb = jac.get(i, b);
                *fisher.get_mut(a, b) += ja * jb * mu_inv;
                if a != b {
                    *fisher.get_mut(b, a) += ja * jb * mu_inv;
                }
            }
        }
    }

    Ok(ModelJacobianResult {
        jacobian: jac,
        fisher,
        model_prediction: y_model,
        param_names,
    })
}

// ── End Phase 2 ──────────────────────────────────────────────────────────

/// Errors from `FitConfig` construction.
#[derive(Debug, PartialEq)]
pub enum FitConfigError {
    /// Energy grid must be non-empty.
    EmptyEnergies,
    /// Resonance data must be non-empty.
    EmptyResonanceData,
    /// initial_densities length must match resonance_data length.
    DensityCountMismatch { densities: usize, isotopes: usize },
    /// isotope_names length must match resonance_data length.
    NameCountMismatch { names: usize, isotopes: usize },
    /// resonance_data count must match group member count.
    GroupMemberCountMismatch {
        group_name: String,
        rd_count: usize,
        member_count: usize,
    },
    /// ResonanceData isotope doesn't match expected group member.
    GroupMemberIsotopeMismatch {
        group_name: String,
        expected_z: u32,
        expected_a: u32,
        got_z: u32,
        got_a: u32,
    },
    /// Temperature must be finite.
    NonFiniteTemperature(f64),
    /// Temperature must be non-negative.
    NegativeTemperature(f64),
}

impl fmt::Display for FitConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEnergies => write!(f, "energy grid must be non-empty"),
            Self::EmptyResonanceData => write!(f, "resonance_data must be non-empty"),
            Self::DensityCountMismatch {
                densities,
                isotopes,
            } => write!(
                f,
                "initial_densities length ({densities}) must match number of density parameters ({isotopes})"
            ),
            Self::NameCountMismatch { names, isotopes } => write!(
                f,
                "isotope_names length ({names}) must match resonance_data length ({isotopes})"
            ),
            Self::GroupMemberCountMismatch {
                group_name,
                rd_count,
                member_count,
            } => write!(
                f,
                "group '{group_name}': provided {rd_count} ResonanceData but group has {member_count} members"
            ),
            Self::GroupMemberIsotopeMismatch {
                group_name,
                expected_z,
                expected_a,
                got_z,
                got_a,
            } => write!(
                f,
                "group '{group_name}': expected Z={expected_z} A={expected_a} but got Z={got_z} A={got_a}"
            ),
            Self::NonFiniteTemperature(v) => {
                write!(f, "temperature must be finite, got {v}")
            }
            Self::NegativeTemperature(v) => {
                write!(f, "temperature must be non-negative, got {v}")
            }
        }
    }
}

impl std::error::Error for FitConfigError {}

/// Result of fitting a single spectrum.
#[derive(Debug, Clone)]
pub struct SpectrumFitResult {
    /// Fitted areal densities (atoms/barn), one per isotope.
    pub densities: Vec<f64>,
    /// Uncertainty on each density.
    ///
    /// `None` when covariance computation was skipped.
    pub uncertainties: Option<Vec<f64>>,
    /// Reduced chi-squared of the fit.
    pub reduced_chi_squared: f64,
    /// Whether the fit converged.
    pub converged: bool,
    /// Number of iterations.
    pub iterations: usize,
    /// Fitted temperature in Kelvin (only when `fit_temperature` is true).
    pub temperature_k: Option<f64>,
    /// 1-sigma uncertainty on the fitted temperature (from covariance matrix).
    pub temperature_k_unc: Option<f64>,
    /// Fitted normalization / signal-scale parameter.
    /// Transmission LM uses `Anorm`; counts background scaling uses `alpha_1`.
    pub anorm: f64,
    /// Fitted background parameter triplet.
    /// Transmission LM uses `[BackA, BackB, BackC]`.
    /// Counts KL background uses `[b0, b1, alpha_2]`.
    pub background: [f64; 3],
    /// Fitted exponential background amplitude (SAMMY BackD).
    /// Zero when the exponential tail is not fitted.
    pub back_d: f64,
    /// Fitted exponential background decay constant (SAMMY BackF).
    /// Zero when the exponential tail is not fitted.
    pub back_f: f64,
    /// Fitted TOF offset in microseconds (SAMMY TZERO t₀).
    /// `None` when energy-scale fitting is not enabled.
    pub t0_us: Option<f64>,
    /// Fitted flight-path scale factor (SAMMY TZERO L₀, dimensionless).
    /// `None` when energy-scale fitting is not enabled.
    pub l_scale: Option<f64>,
    /// Joint-Poisson conditional binomial deviance divided by `(n − k)`
    /// (memo 35 §P1.2 — primary GOF for `SolverConfig::JointPoisson`).
    ///
    /// `Some(D/dof)` when the joint-Poisson solver was used;
    /// `None` for LM and legacy Poisson-KL paths (those populate
    /// `reduced_chi_squared` with Pearson χ² / (n−k) instead).
    pub deviance_per_dof: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::synthetic_single_resonance;
    use nereids_fitting::lm::FitModel;
    use nereids_physics::transmission as phys_transmission;

    use crate::test_helpers::u238_single_resonance;

    // ── Phase 0: InputData + SolverConfig + CountsBackgroundConfig tests ──

    #[test]
    fn test_input_data_transmission_n_energies() {
        let data = InputData::Transmission {
            transmission: vec![0.9, 0.8, 0.7],
            uncertainty: vec![0.01, 0.01, 0.01],
        };
        assert_eq!(data.n_energies(), 3);
        assert!(!data.is_counts());
    }

    #[test]
    fn test_input_data_counts_n_energies() {
        let data = InputData::Counts {
            sample_counts: vec![10.0, 20.0, 30.0, 40.0],
            open_beam_counts: vec![100.0, 100.0, 100.0, 100.0],
        };
        assert_eq!(data.n_energies(), 4);
        assert!(data.is_counts());
    }

    #[test]
    fn test_input_data_counts_with_nuisance() {
        let data = InputData::CountsWithNuisance {
            sample_counts: vec![5.0, 6.0],
            flux: vec![100.0, 100.0],
            background: vec![0.5, 0.5],
        };
        assert_eq!(data.n_energies(), 2);
        assert!(data.is_counts());
    }

    #[test]
    fn test_solver_config_default_is_auto() {
        let cfg = SolverConfig::default();
        assert!(matches!(cfg, SolverConfig::Auto));
    }

    #[test]
    fn test_counts_background_config_default() {
        let cfg = CountsBackgroundConfig::default();
        assert!((cfg.alpha_1_init - 1.0).abs() < f64::EPSILON);
        assert!((cfg.alpha_2_init - 1.0).abs() < f64::EPSILON);
        assert!(!cfg.fit_alpha_1);
        assert!(!cfg.fit_alpha_2);
    }

    // ── Phase 2: fit_spectrum_typed tests ──

    /// Helper: build synthetic transmission data from known density.
    fn synthetic_transmission(
        data: &ResonanceData,
        true_density: f64,
        energies: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(vec![
                phys_transmission::broadened_cross_sections(
                    energies,
                    std::slice::from_ref(data),
                    0.0,
                    None,
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap(),
            ]),
            density_indices: Arc::new(vec![0]),
            energies: None,
            instrument: None,
        };
        let t = model.evaluate(&[true_density]).unwrap();
        let sigma: Vec<f64> = t.iter().map(|&v| 0.01 * v.max(0.01)).collect();
        (t, sigma)
    }

    /// Helper: build synthetic counts from known density.
    fn synthetic_counts(
        data: &ResonanceData,
        true_density: f64,
        energies: &[f64],
        i0: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let (t, _) = synthetic_transmission(data, true_density, energies);
        let open_beam: Vec<f64> = vec![i0; energies.len()];
        let sample: Vec<f64> = t.iter().map(|&v| (v * i0).round().max(0.0)).collect();
        (sample, open_beam)
    }

    #[test]
    fn test_typed_transmission_lm_recovers_density() {
        let data = u238_single_resonance();
        let true_density = 0.002;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, sigma) = synthetic_transmission(&data, true_density, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData::Transmission {
            transmission: t,
            uncertainty: sigma,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged, "LM should converge");
        let fitted = result.densities[0];
        assert!(
            (fitted - true_density).abs() / true_density < 0.05,
            "density: fitted={fitted}, true={true_density}"
        );
    }

    #[test]
    fn test_extract_result_drops_uncertainties_when_unconverged() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..21).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            293.6,
            None,
            vec![0.001],
        )
        .unwrap();

        let result = LmResult {
            chi_squared: 1.0,
            reduced_chi_squared: 1.0,
            iterations: 5,
            converged: false,
            params: vec![0.001],
            covariance: Some(lm::FlatMatrix::zeros(1, 1)),
            uncertainties: Some(vec![0.123]),
        };

        let extracted = extract_result(&config, &result, 1, None).unwrap();
        assert!(!extracted.converged);
        assert!(
            extracted.uncertainties.is_none(),
            "pipeline must not surface uncertainties from an unconverged fit"
        );
        assert!(extracted.temperature_k_unc.is_none());
    }

    #[test]
    fn test_typed_counts_kl_recovers_density() {
        let data = u238_single_resonance();
        let true_density = 0.002;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, open_beam) = synthetic_counts(&data, true_density, &energies, 1000.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let input = InputData::Counts {
            sample_counts: sample,
            open_beam_counts: open_beam,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged, "KL on counts should converge");
        let fitted = result.densities[0];
        assert!(
            (fitted - true_density).abs() / true_density < 0.10,
            "density: fitted={fitted}, true={true_density}"
        );
    }

    #[test]
    fn test_typed_counts_kl_low_counts_recovers_density() {
        // I0=10 counts per bin — the regime where KL excels and LM fails
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, open_beam) = synthetic_counts(&data, true_density, &energies, 10.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let input = InputData::Counts {
            sample_counts: sample,
            open_beam_counts: open_beam,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged, "KL on low counts should converge");
        let fitted = result.densities[0];
        // Wider tolerance for low counts
        assert!(
            (fitted - true_density).abs() / true_density < 0.30,
            "density: fitted={fitted}, true={true_density}"
        );
    }

    #[test]
    fn test_typed_transmission_kl_recovers_density() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, sigma) = synthetic_transmission(&data, true_density, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let input = InputData::Transmission {
            transmission: t,
            uncertainty: sigma,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged, "KL on transmission should converge");
        let fitted = result.densities[0];
        assert!(
            (fitted - true_density).abs() / true_density < 0.05,
            "density: fitted={fitted}, true={true_density}"
        );
    }

    #[test]
    fn test_typed_counts_lm_auto_converts() {
        // Counts + LM should auto-convert to transmission and fit
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, open_beam) = synthetic_counts(&data, true_density, &energies, 1000.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData::Counts {
            sample_counts: sample,
            open_beam_counts: open_beam,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(
            result.converged,
            "LM on auto-converted counts should converge"
        );
        let fitted = result.densities[0];
        assert!(
            (fitted - true_density).abs() / true_density < 0.10,
            "density: fitted={fitted}, true={true_density}"
        );
    }

    #[test]
    fn test_typed_auto_solver_selects_kl_for_counts() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, open_beam) = synthetic_counts(&data, true_density, &energies, 1000.0);

        // Auto solver (default)
        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap(); // SolverConfig::Auto by default

        let input = InputData::Counts {
            sample_counts: sample,
            open_beam_counts: open_beam,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(
            result.converged,
            "Auto solver on counts should use KL and converge"
        );
    }

    #[test]
    fn test_typed_transmission_with_background_lm() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let true_anorm = 0.95;
        let true_back_a = 0.02;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();

        // Generate synthetic data with background
        let (t_pure, _) = synthetic_transmission(&data, true_density, &energies);
        let t_bg: Vec<f64> = t_pure
            .iter()
            .map(|&v| true_anorm * v + true_back_a)
            .collect();
        let sigma: Vec<f64> = t_bg.iter().map(|&v| 0.01 * v.max(0.01)).collect();

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig {
            max_iter: 500,
            ..LmConfig::default()
        }))
        .with_transmission_background(BackgroundConfig::default());

        let input = InputData::Transmission {
            transmission: t_bg,
            uncertainty: sigma,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(
            result.converged,
            "LM+BG should converge on noiseless synthetic data (chi2r={}, iter={})",
            result.reduced_chi_squared, result.iterations
        );
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.05,
            "density: fitted={}, true={true_density}",
            result.densities[0]
        );
        assert!(
            (result.anorm - true_anorm).abs() / true_anorm < 0.05,
            "anorm: fitted={}, true={true_anorm}",
            result.anorm
        );
    }

    #[test]
    fn test_typed_counts_with_nuisance_rejects_lm() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..10).map(|i| 1.0 + (i as f64) * 0.5).collect();

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData::CountsWithNuisance {
            sample_counts: vec![10.0; 10],
            flux: vec![100.0; 10],
            background: vec![0.0; 10],
        };

        let result = fit_spectrum_typed(&input, &config);
        assert!(result.is_err(), "CountsWithNuisance + LM should error");
    }

    /// Helper: build synthetic transmission at a given temperature.
    fn synthetic_transmission_at_temp(
        data: &ResonanceData,
        true_density: f64,
        temperature_k: f64,
        energies: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let xs = phys_transmission::broadened_cross_sections(
            energies,
            std::slice::from_ref(data),
            temperature_k,
            None,
            None,
        )
        .unwrap();
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs),
            density_indices: Arc::new(vec![0]),
            energies: None,
            instrument: None,
        };
        let t = model.evaluate(&[true_density]).unwrap();
        let sigma: Vec<f64> = t.iter().map(|&v| 0.01 * v.max(0.01)).collect();
        (t, sigma)
    }

    #[test]
    fn test_typed_poisson_kl_with_temperature() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let true_temp = 350.0;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, sigma) = synthetic_transmission_at_temp(&data, true_density, true_temp, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            300.0, // initial guess (off by 50 K)
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig {
            max_iter: 500,
            ..PoissonConfig::default()
        }))
        .with_fit_temperature(true);

        let input = InputData::Transmission {
            transmission: t,
            uncertainty: sigma,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();

        // Check density recovery (within 1%)
        let fitted_density = result.densities[0];
        assert!(
            (fitted_density - true_density).abs() / true_density < 0.01,
            "density: fitted={fitted_density}, true={true_density}, ratio={}",
            (fitted_density - true_density).abs() / true_density,
        );

        // Check temperature recovery (within 1 K)
        let fitted_temp = result
            .temperature_k
            .expect("temperature_k should be Some when fit_temperature=true");
        assert!(
            (fitted_temp - true_temp).abs() < 1.0,
            "temperature: fitted={fitted_temp}, true={true_temp}, delta={}",
            (fitted_temp - true_temp).abs(),
        );
    }

    #[test]
    fn test_typed_poisson_kl_with_temperature_and_background() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let true_temp = 350.0;
        let true_b0 = 0.012;
        let true_b1 = 0.008;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, sigma) = synthetic_transmission_at_temp(&data, true_density, true_temp, &energies);
        let measured_t: Vec<f64> = t
            .iter()
            .zip(energies.iter())
            .map(|(&ti, &e)| ti + true_b0 + true_b1 / e.sqrt())
            .collect();

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig {
            max_iter: 120,
            gauss_newton_lambda: 1e-4,
            ..PoissonConfig::default()
        }))
        .with_fit_temperature(true)
        .with_transmission_background(BackgroundConfig::default());

        let input = InputData::Transmission {
            transmission: measured_t,
            uncertainty: sigma,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();

        assert!(result.converged, "fit did not converge: {result:?}");
        assert!(
            result.iterations <= 80,
            "expected KL background+temperature fit to converge well before max_iter; got {}",
            result.iterations,
        );

        let fitted_density = result.densities[0];
        assert!(
            (fitted_density - true_density).abs() / true_density < 0.02,
            "density: fitted={fitted_density}, true={true_density}, ratio={}",
            (fitted_density - true_density).abs() / true_density,
        );

        let fitted_temp = result
            .temperature_k
            .expect("temperature_k should be Some when fit_temperature=true");
        assert!(
            (fitted_temp - true_temp).abs() < 3.0,
            "temperature: fitted={fitted_temp}, true={true_temp}, delta={}",
            (fitted_temp - true_temp).abs(),
        );

        assert!(
            (result.background[0] - true_b0).abs() < 5e-3,
            "background b0: fitted={}, true={}",
            result.background[0],
            true_b0,
        );
        assert!(
            (result.background[1] - true_b1).abs() < 5e-3,
            "background b1: fitted={}, true={}",
            result.background[1],
            true_b1,
        );
    }

    #[test]
    fn test_typed_counts_poisson_kl_with_background() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let true_b0 = 0.012;
        let true_b1 = 0.008;
        let detector_bg = 2.0;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, _) = synthetic_transmission(&data, true_density, &energies);
        let flux = vec![1000.0; energies.len()];
        let background = vec![detector_bg; energies.len()];
        let sample_counts: Vec<f64> = t
            .iter()
            .zip(energies.iter())
            .zip(flux.iter())
            .zip(background.iter())
            .map(|(((&ti, &e), &f), &bg)| f * (ti + true_b0 + true_b1 / e.sqrt()) + bg)
            .collect();

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig {
            max_iter: 120,
            gauss_newton_lambda: 1e-4,
            ..PoissonConfig::default()
        }))
        .with_transmission_background(BackgroundConfig::default());

        let input = InputData::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();

        assert!(result.converged, "fit did not converge: {result:?}");
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.02,
            "density: fitted={}, true={true_density}",
            result.densities[0]
        );
        assert!(
            (result.background[0] - true_b0).abs() < 5e-3,
            "background b0: fitted={}, true={true_b0}",
            result.background[0]
        );
        assert!(
            (result.background[1] - true_b1).abs() < 5e-3,
            "background b1: fitted={}, true={true_b1}",
            result.background[1]
        );
    }

    #[test]
    fn test_typed_counts_poisson_kl_with_temperature_and_background() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let true_temp = 350.0;
        let true_b0 = 0.012;
        let true_b1 = 0.008;
        let detector_bg = 2.0;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, _) = synthetic_transmission_at_temp(&data, true_density, true_temp, &energies);
        let flux = vec![1000.0; energies.len()];
        let background = vec![detector_bg; energies.len()];
        let sample_counts: Vec<f64> = t
            .iter()
            .zip(energies.iter())
            .zip(flux.iter())
            .zip(background.iter())
            .map(|(((&ti, &e), &f), &bg)| f * (ti + true_b0 + true_b1 / e.sqrt()) + bg)
            .collect();

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig {
            max_iter: 120,
            gauss_newton_lambda: 1e-4,
            ..PoissonConfig::default()
        }))
        .with_fit_temperature(true)
        .with_transmission_background(BackgroundConfig::default());

        let input = InputData::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();

        assert!(result.converged, "fit did not converge: {result:?}");
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.02,
            "density: fitted={}, true={true_density}",
            result.densities[0]
        );

        let fitted_temp = result
            .temperature_k
            .expect("temperature_k should be Some when fit_temperature=true");
        assert!(
            (fitted_temp - true_temp).abs() < 3.0,
            "temperature: fitted={fitted_temp}, true={true_temp}, delta={}",
            (fitted_temp - true_temp).abs(),
        );
        assert!(
            (result.background[0] - true_b0).abs() < 5e-3,
            "background b0: fitted={}, true={true_b0}",
            result.background[0]
        );
        assert!(
            (result.background[1] - true_b1).abs() < 5e-3,
            "background b1: fitted={}, true={true_b1}",
            result.background[1]
        );
    }

    #[test]
    fn test_typed_counts_poisson_kl_with_counts_background_scales() {
        let data = u238_single_resonance();
        let true_density = 0.002;
        let true_alpha1 = 0.92;
        let true_alpha2 = 1.35;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, _) = synthetic_transmission(&data, true_density, &energies);
        let flux = vec![120.0; energies.len()];
        let background: Vec<f64> = energies.iter().map(|&e| 30.0 + 8.0 / e.sqrt()).collect();
        let sample_counts: Vec<f64> = t
            .iter()
            .zip(flux.iter())
            .zip(background.iter())
            .map(|((&ti, &f), &bg)| true_alpha1 * f * ti + true_alpha2 * bg)
            .collect();

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig {
            max_iter: 120,
            gauss_newton_lambda: 1e-4,
            ..PoissonConfig::default()
        }))
        .with_counts_background(CountsBackgroundConfig {
            alpha_1_init: 1.0,
            alpha_2_init: 1.0,
            fit_alpha_1: true,
            fit_alpha_2: true,
            c: 1.0,
        });

        let input = InputData::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();

        assert!(result.converged, "fit did not converge: {result:?}");
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.02,
            "density: fitted={}, true={true_density}",
            result.densities[0]
        );
        assert!(
            (result.anorm - true_alpha1).abs() < 5e-3,
            "alpha_1/anorm: fitted={}, true={true_alpha1}",
            result.anorm
        );
        assert!(
            (result.background[2] - true_alpha2).abs() < 5e-3,
            "alpha_2/background[2]: fitted={}, true={true_alpha2}",
            result.background[2]
        );
    }

    /// Round-trip test: create a group of 2 isotopes with known ratios,
    /// generate synthetic transmission, fit with group constraints,
    /// verify the fitted group density matches the true value.
    #[test]
    fn test_grouped_fit_spectrum_round_trip() {
        use nereids_core::types::IsotopeGroup;

        // Two synthetic isotopes with resonances at different energies
        let rd1 = synthetic_single_resonance(92, 235, 233.025, 5.0);
        let rd2 = synthetic_single_resonance(92, 238, 236.006, 7.0);

        // Group with 60/40 ratio
        let iso1 = nereids_core::types::Isotope::new(92, 235).unwrap();
        let iso2 = nereids_core::types::Isotope::new(92, 238).unwrap();
        let group =
            IsotopeGroup::custom("U (60/40)".into(), vec![(iso1, 0.6), (iso2, 0.4)]).unwrap();

        let energies: Vec<f64> = (0..301).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let true_density = 0.0005;

        // Generate synthetic transmission using effective densities
        let sample = nereids_physics::transmission::SampleParams::new(
            0.0,
            vec![
                (rd1.clone(), true_density * 0.6),
                (rd2.clone(), true_density * 0.4),
            ],
        )
        .unwrap();
        let transmission =
            nereids_physics::transmission::forward_model(&energies, &sample, None).unwrap();
        let uncertainty: Vec<f64> = transmission.iter().map(|&t| 0.01 * t.max(0.01)).collect();

        // Build config with group
        let config = UnifiedFitConfig::new(
            energies.clone(),
            vec![rd1.clone()],
            vec!["placeholder".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_groups(&[(&group, &[rd1, rd2])], vec![0.001])
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData::Transmission {
            transmission,
            uncertainty,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();

        // Should recover true density within 1%
        assert_eq!(result.densities.len(), 1, "should have 1 group density");
        let fitted = result.densities[0];
        let rel_error = (fitted - true_density).abs() / true_density;
        assert!(
            rel_error < 0.01,
            "group density: fitted={fitted}, true={true_density}, rel_error={rel_error}"
        );
        assert!(result.converged, "fit should converge");
    }

    #[test]
    fn test_grouped_poisson_kl_with_temperature_and_background_noiseless() {
        use nereids_core::types::IsotopeGroup;

        let rd1 = synthetic_single_resonance(72, 176, 8.5, 5.0);
        let rd2 = synthetic_single_resonance(72, 178, 17.0, 7.5);
        let rd3 = synthetic_single_resonance(72, 180, 29.0, 6.0);

        let hf176 = nereids_core::types::Isotope::new(72, 176).unwrap();
        let hf178 = nereids_core::types::Isotope::new(72, 178).unwrap();
        let hf180 = nereids_core::types::Isotope::new(72, 180).unwrap();
        let group = IsotopeGroup::custom(
            "Hf-like (3 member)".into(),
            vec![(hf176, 0.2), (hf178, 0.5), (hf180, 0.3)],
        )
        .unwrap();

        let energies: Vec<f64> = (0..300).map(|i| 1.0 + (49.0 * i as f64) / 299.0).collect();
        let true_density = 0.001;
        let true_temp = 400.0;
        let true_b0 = 0.012;
        let true_b1 = 0.008;

        let sample = nereids_physics::transmission::SampleParams::new(
            true_temp,
            vec![
                (rd1.clone(), true_density * 0.2),
                (rd2.clone(), true_density * 0.5),
                (rd3.clone(), true_density * 0.3),
            ],
        )
        .unwrap();
        let pure_t =
            nereids_physics::transmission::forward_model(&energies, &sample, None).unwrap();
        let measured_t: Vec<f64> = pure_t
            .iter()
            .zip(energies.iter())
            .map(|(&t, &e)| t + true_b0 + true_b1 / e.sqrt())
            .collect();
        let sigma = vec![0.001; energies.len()];

        let config = UnifiedFitConfig::new(
            energies.clone(),
            vec![rd1.clone()],
            vec!["placeholder".into()],
            293.6,
            None,
            vec![0.0008],
        )
        .unwrap()
        .with_groups(&[(&group, &[rd1, rd2, rd3])], vec![0.0008])
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig {
            max_iter: 200,
            gauss_newton_lambda: 1e-4,
            ..PoissonConfig::default()
        }))
        .with_fit_temperature(true)
        .with_transmission_background(BackgroundConfig::default());

        let input = InputData::Transmission {
            transmission: measured_t,
            uncertainty: sigma,
        };

        let result = fit_spectrum_typed(&input, &config).unwrap();

        assert!(result.converged, "fit did not converge: {result:?}");
        // Tolerance: 1% — the NormalizedTransmissionModel (4 background params)
        // has slightly different convergence than the old 2-param KL model.
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.01,
            "density: fitted={}, true={true_density}",
            result.densities[0]
        );
        let fitted_temp = result
            .temperature_k
            .expect("temperature_k should be Some when fit_temperature=true");
        assert!(
            (fitted_temp - true_temp).abs() < 8.0,
            "temperature: fitted={fitted_temp}, true={true_temp}",
        );
        // Background: NormalizedTransmissionModel distributes the additive
        // background across Anorm + BackA/B/C.  Check that the total background
        // contribution is reasonable, not individual parameters.
        let e_mid: f64 = 10.0;
        let bg_total = (result.anorm - 1.0)
            + result.background[0]
            + result.background[1] / e_mid.sqrt()
            + result.background[2] * e_mid.sqrt();
        let true_bg_mid = true_b0 + true_b1 / e_mid.sqrt();
        assert!(
            (bg_total - true_bg_mid).abs() < 0.02,
            "total bg at E={e_mid}: fitted={bg_total:.6}, true={true_bg_mid:.6}",
        );
    }

    // ── Phase 2: KL fitting uncertainty tests ──────────────────────────────

    #[test]
    fn test_kl_counts_returns_density_uncertainty() {
        let data = u238_single_resonance();
        let true_density = 0.002;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (sample, open_beam) = synthetic_counts(&data, true_density, &energies, 1000.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let input = InputData::Counts {
            sample_counts: sample,
            open_beam_counts: open_beam,
        };
        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged);
        let unc = result
            .uncertainties
            .as_ref()
            .expect("KL 1D fit should return density uncertainties");
        assert_eq!(unc.len(), 1);
        assert!(
            unc[0].is_finite() && unc[0] > 0.0,
            "density unc = {}",
            unc[0]
        );
        assert!(
            unc[0] < result.densities[0],
            "unc ({}) should be < density ({}) for high-count data",
            unc[0],
            result.densities[0]
        );
    }

    #[test]
    fn test_kl_counts_returns_temperature_uncertainty() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.05).collect();
        let (sample, open_beam) = synthetic_counts(&data, 0.001, &energies, 1000.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            350.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_fit_temperature(true);

        let input = InputData::Counts {
            sample_counts: sample,
            open_beam_counts: open_beam,
        };
        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged);
        let unc = result
            .uncertainties
            .as_ref()
            .expect("KL+temp fit should return density uncertainties");
        assert!(
            unc[0].is_finite() && unc[0] > 0.0,
            "density unc = {}",
            unc[0]
        );
        let t_unc = result
            .temperature_k_unc
            .expect("KL+temp fit should return temperature uncertainty");
        assert!(
            t_unc.is_finite() && t_unc > 0.0,
            "temperature unc = {t_unc}"
        );
    }

    #[test]
    fn test_kl_counts_with_background_returns_uncertainty() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.05).collect();
        let (sample, open_beam) = synthetic_counts(&data, 0.001, &energies, 1000.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_fit_temperature(true)
        .with_transmission_background(BackgroundConfig::default());

        let input = InputData::Counts {
            sample_counts: sample,
            open_beam_counts: open_beam,
        };
        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged);
        let unc = result
            .uncertainties
            .as_ref()
            .expect("KL+bg fit should return density uncertainties");
        assert!(
            unc[0].is_finite() && unc[0] > 0.0,
            "density unc = {}",
            unc[0]
        );
    }

    #[test]
    fn test_lm_uncertainty_not_regressed() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, sigma) = synthetic_transmission(&data, 0.001, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let input = InputData::Transmission {
            transmission: t,
            uncertainty: sigma,
        };
        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged);
        let unc = result
            .uncertainties
            .as_ref()
            .expect("LM should still return uncertainties");
        assert!(unc[0].is_finite() && unc[0] > 0.0);
    }

    // ── Energy-scale fitting tests ──

    /// fit_energy_scale + fit_temperature must be rejected.
    #[test]
    fn test_energy_scale_rejects_temperature() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_obs, sigma) = synthetic_transmission(&data, 0.002, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            293.6,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(Default::default()))
        .with_fit_temperature(true)
        .with_energy_scale(0.0, 1.0, 25.0);

        let input = InputData::Transmission {
            transmission: t_obs,
            uncertainty: sigma,
        };
        let err = fit_spectrum_typed(&input, &config);
        assert!(err.is_err(), "energy_scale + temperature should fail");
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("fit_energy_scale") && msg.contains("fit_temperature"),
            "error should mention both flags: {msg}"
        );
    }

    /// Energy-scale fitting returns t0_us and l_scale in the result.
    #[test]
    fn test_energy_scale_returns_fitted_params() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t_obs, sigma) = synthetic_transmission(&data, 0.002, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            293.6,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(Default::default()))
        .with_transmission_background(BackgroundConfig::default())
        .with_energy_scale(0.0, 1.0, 25.0);

        let input = InputData::Transmission {
            transmission: t_obs,
            uncertainty: sigma,
        };
        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(result.converged, "Fit should converge");
        assert!(
            result.t0_us.is_some(),
            "t0_us should be Some when energy-scale is fitted"
        );
        assert!(
            result.l_scale.is_some(),
            "l_scale should be Some when energy-scale is fitted"
        );
        let t0 = result.t0_us.unwrap();
        let ls = result.l_scale.unwrap();
        // Values should be finite (not NaN/Inf) and within bounds
        assert!(t0.is_finite(), "t0 should be finite, got {t0}");
        assert!(ls.is_finite(), "l_scale should be finite, got {ls}");
        assert!(t0.abs() < 10.0, "t0 should be within bounds, got {t0}");
        assert!(
            ls > 0.98 && ls < 1.02,
            "l_scale should be within bounds, got {ls}"
        );
    }

    /// Without energy-scale fitting, t0_us and l_scale should be None.
    #[test]
    fn test_no_energy_scale_returns_none() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t_obs, sigma) = synthetic_transmission(&data, 0.002, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            293.6,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(Default::default()))
        .with_transmission_background(BackgroundConfig::default());

        let input = InputData::Transmission {
            transmission: t_obs,
            uncertainty: sigma,
        };
        let result = fit_spectrum_typed(&input, &config).unwrap();
        assert!(
            result.t0_us.is_none(),
            "t0_us should be None without energy-scale"
        );
        assert!(
            result.l_scale.is_none(),
            "l_scale should be None without energy-scale"
        );
    }

    // ==================================================================
    // Joint-Poisson solver integration tests (memo 35 §P1/§P2)
    // ==================================================================

    /// End-to-end: joint-Poisson density recovery at c = 5.98 on synthetic
    /// matched-model counts, via `fit_spectrum_typed`.  Verifies that
    /// `SpectrumFitResult.deviance_per_dof` is populated (P1.2) and that
    /// density is recovered to within 5% on a single-resonance spectrum
    /// under expected (noise-free) counts.
    #[test]
    fn test_joint_poisson_density_recovery_c_5_98() {
        let data = u238_single_resonance();
        let true_density = 0.0005_f64;
        let c = 5.98_f64;
        let lam_ob = 1000.0_f64; // expected open-beam counts per bin (O rate)
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, _) = synthetic_transmission(&data, true_density, &energies);

        // Noise-free expectations under joint-Poisson model:
        //   E[O] = lam_ob, E[S] = c · lam_ob · T
        let open_beam_counts: Vec<f64> = vec![lam_ob; energies.len()];
        let sample_counts: Vec<f64> = t.iter().map(|&ti| c * lam_ob * ti).collect();

        let jp_cfg = JointPoissonFitConfig::default();
        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::JointPoisson(jp_cfg))
        .with_counts_background(CountsBackgroundConfig {
            c,
            ..Default::default()
        });

        let input = InputData::Counts {
            sample_counts,
            open_beam_counts,
        };
        let result = fit_spectrum_typed(&input, &config).unwrap();

        // Deviance-based GOF is populated (P1.2).
        let d_per_dof = result
            .deviance_per_dof
            .expect("joint-Poisson solver must populate deviance_per_dof");
        assert!(
            d_per_dof.is_finite() && d_per_dof >= 0.0,
            "deviance_per_dof = {d_per_dof} is not a valid GOF"
        );
        // On noise-free expected counts, D should be very small (approaches
        // zero in the matched-model limit; allow some slack for numerical
        // error in the forward model).
        assert!(
            d_per_dof < 0.5,
            "noise-free expected-counts fit should give D/dof ≈ 0, got {d_per_dof}"
        );
        // Density recovery.
        let rel_bias = (result.densities[0] - true_density) / true_density;
        assert!(
            rel_bias.abs() < 0.05,
            "density bias {rel_bias} > 5%: fitted={} truth={true_density}",
            result.densities[0]
        );
        // Back-compat: reduced_chi_squared mirrors deviance_per_dof.
        assert!((result.reduced_chi_squared - d_per_dof).abs() < 1e-12);
    }

    /// `JointPoisson` rejects `fit_alpha_1` / `fit_alpha_2` — the profile
    /// `λ̂` absorbs the global flux scale, so `alpha_1` is redundant;
    /// `alpha_2` / B_det wiring is P3-deferred.
    #[test]
    fn test_joint_poisson_rejects_alpha_fit() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, _) = synthetic_transmission(&data, 0.0005, &energies);
        let open_beam_counts: Vec<f64> = vec![500.0; energies.len()];
        let sample_counts: Vec<f64> = t.iter().map(|&ti| 500.0 * ti).collect();

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::JointPoisson(JointPoissonFitConfig::default()))
        .with_counts_background(CountsBackgroundConfig {
            fit_alpha_1: true,
            c: 1.0,
            ..Default::default()
        });

        let input = InputData::Counts {
            sample_counts,
            open_beam_counts,
        };
        let err = fit_spectrum_typed(&input, &config).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("fit_alpha_1") || msg.contains("alpha_1"),
            "expected alpha_1 rejection message, got: {msg}"
        );
    }

    /// `JointPoisson` rejects transmission input (no O/S pair available).
    #[test]
    fn test_joint_poisson_rejects_transmission_input() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, u) = synthetic_transmission(&data, 0.0005, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::JointPoisson(JointPoissonFitConfig::default()));

        let input = InputData::Transmission {
            transmission: t,
            uncertainty: u,
        };
        let err = fit_spectrum_typed(&input, &config).unwrap_err();
        assert!(err.to_string().contains("JointPoisson"));
    }

    // ──────────────────────────────────────────────────────────────────
    // P2.2: transmission_background through the joint-Poisson path.
    // ──────────────────────────────────────────────────────────────────

    /// End-to-end: joint-Poisson with A_n + B_A + B_B + B_C free on
    /// noise-free synthetic counts with known background.  On 201 bins
    /// with 5 free params the (n, A_n) correlation is non-trivial so we
    /// assert the *wiring* is correct (bg reaches the fit, D/dof → 0,
    /// A_n + B_A near truth, density within 10%) rather than EG2-grade
    /// recovery.  The real VENUS evaluation in the companion memo is
    /// the stress test.
    #[test]
    fn test_joint_poisson_with_transmission_background() {
        let data = u238_single_resonance();
        let true_density = 0.0005_f64;
        let true_anorm = 0.85_f64;
        let true_ba = 0.03_f64;
        let true_bb = -0.01_f64;
        let true_bc = 0.0_f64;
        let c = 5.98_f64;
        let lam_ob = 2000.0_f64;
        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t_inner, _) = synthetic_transmission(&data, true_density, &energies);
        let t_out: Vec<f64> = t_inner
            .iter()
            .zip(energies.iter())
            .map(|(&ti, &e)| true_anorm * ti + true_ba + true_bb / e.sqrt() + true_bc * e.sqrt())
            .collect();
        let open_beam_counts: Vec<f64> = vec![lam_ob; energies.len()];
        let sample_counts: Vec<f64> = t_out.iter().map(|&ti| c * lam_ob * ti).collect();

        let bg = BackgroundConfig {
            anorm_init: 1.0,
            back_a_init: 0.0,
            back_b_init: 0.0,
            back_c_init: 0.0,
            back_d_init: 0.01,
            back_f_init: 1.0,
            fit_anorm: true,
            fit_back_a: true,
            fit_back_b: true,
            fit_back_c: true,
            fit_back_d: false,
            fit_back_f: false,
        };

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::JointPoisson(JointPoissonFitConfig::default()))
        .with_counts_background(CountsBackgroundConfig {
            c,
            ..Default::default()
        })
        .with_transmission_background(bg);

        let input = InputData::Counts {
            sample_counts,
            open_beam_counts,
        };
        let r = fit_spectrum_typed(&input, &config).unwrap();

        // The invariant P2.2 wiring is supposed to produce is: the 4 bg
        // parameters *actually reach the objective* (the fit produces a
        // near-zero deviance on noise-free expected counts) and the
        // fitter moves them off their initial values.  Density / A_n /
        // B_A recovery at unit-test scale (201 bins, 5 free params)
        // inherits the classic n ↔ A_n correlation — the realistic
        // stress test is the VENUS evaluation in the companion memo.

        // Deviance-based GOF populated and → 0 on noise-free expected counts.
        let dpd = r.deviance_per_dof.expect("joint-Poisson must report D/dof");
        assert!(
            dpd < 1.0,
            "D/dof = {dpd} unexpectedly large on noise-free fit — bg params not reaching objective?"
        );
        // Density didn't rail to zero.
        assert!(r.densities[0] > 1e-5, "density railed: {}", r.densities[0]);
        // A_n moved off its initial 1.0 toward truth 0.85.
        assert!(
            (r.anorm - 1.0).abs() > 0.05,
            "A_n did not move from init 1.0 (fitted={})",
            r.anorm
        );
        // Background triplet moved off zero at least in one component.
        let bg_moved = r.background.iter().any(|v| v.abs() > 1e-4);
        assert!(
            bg_moved,
            "no bg parameter moved from init 0: {:?}",
            r.background
        );
    }

    /// §P2.2 operational rule: `B_B` or `B_C` free → `B_A` must be free too.
    #[test]
    fn test_joint_poisson_p2_2_requires_back_a_when_back_b_enabled() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, _) = synthetic_transmission(&data, 0.0005, &energies);
        let ob: Vec<f64> = vec![500.0; energies.len()];
        let s: Vec<f64> = t.iter().map(|&ti| 500.0 * ti).collect();

        let bg = BackgroundConfig {
            // B_B enabled without B_A → must be rejected.
            fit_anorm: true,
            fit_back_a: false,
            fit_back_b: true,
            fit_back_c: false,
            fit_back_d: false,
            fit_back_f: false,
            ..BackgroundConfig::default()
        };

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::JointPoisson(JointPoissonFitConfig::default()))
        .with_counts_background(CountsBackgroundConfig {
            c: 1.0,
            ..Default::default()
        })
        .with_transmission_background(bg);

        let input = InputData::Counts {
            sample_counts: s,
            open_beam_counts: ob,
        };
        let err = fit_spectrum_typed(&input, &config).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("§P2.2") || msg.contains("B_A"),
            "expected §P2.2 rejection message, got: {msg}"
        );
    }

    /// Joint-Poisson rejects BackD/BackF exponential tail (§P4-deferred).
    #[test]
    fn test_joint_poisson_rejects_back_d_f() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let (t, _) = synthetic_transmission(&data, 0.0005, &energies);
        let ob: Vec<f64> = vec![500.0; energies.len()];
        let s: Vec<f64> = t.iter().map(|&ti| 500.0 * ti).collect();

        let bg = BackgroundConfig {
            fit_back_d: true,
            ..BackgroundConfig::default()
        };

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::JointPoisson(JointPoissonFitConfig::default()))
        .with_counts_background(CountsBackgroundConfig {
            c: 1.0,
            ..Default::default()
        })
        .with_transmission_background(bg);

        let input = InputData::Counts {
            sample_counts: s,
            open_beam_counts: ob,
        };
        let err = fit_spectrum_typed(&input, &config).unwrap_err();
        assert!(
            err.to_string().contains("BackD") || err.to_string().contains("§P4"),
            "expected BackD/BackF rejection, got: {err}"
        );
    }
}
