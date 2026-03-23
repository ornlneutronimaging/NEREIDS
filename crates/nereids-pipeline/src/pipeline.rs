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
use nereids_fitting::lm::{self, FitModel, LmConfig, LmResult};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{self, PoissonConfig};
use nereids_fitting::transmission_model::{
    NormalizedTransmissionModel, PrecomputedTransmissionModel, TransmissionFitModel,
};
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::InstrumentParams;

use crate::error::PipelineError;

/// SAMMY-style normalization and background configuration.
///
/// When enabled, the transmission model becomes:
///   T_out(E) = Anorm × T_inner(E) + BackA + BackB / √E + BackC × √E
///
/// These 4 parameters are fitted jointly with the isotope densities.
///
/// ## SAMMY Reference
/// SAMMY manual Sec III.E.2 — NORMAlization and BACKGround cards.
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
    /// Whether Anorm is free (true) or fixed (false).
    pub fit_anorm: bool,
    /// Whether BackA is free (true) or fixed (false).
    pub fit_back_a: bool,
    /// Whether BackB is free (true) or fixed (false).
    pub fit_back_b: bool,
    /// Whether BackC is free (true) or fixed (false).
    pub fit_back_c: bool,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            anorm_init: 1.0,
            back_a_init: 0.0,
            back_b_init: 0.0,
            back_c_init: 0.0,
            fit_anorm: true,
            fit_back_a: true,
            fit_back_b: true,
            fit_back_c: true,
        }
    }
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
    /// Poisson KL divergence minimizer (projected gradient + Armijo).
    PoissonKL(PoissonConfig),
    /// Automatic: Counts → PoissonKL, Transmission → LM.
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
}

impl Default for CountsBackgroundConfig {
    fn default() -> Self {
        Self {
            alpha_1_init: 1.0,
            alpha_2_init: 1.0,
            fit_alpha_1: false,
            fit_alpha_2: false,
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

        // ── Counts + LM: convert to transmission ──
        (
            InputData::Counts {
                sample_counts,
                open_beam_counts,
            },
            SolverConfig::LevenbergMarquardt(lm_cfg),
        ) => {
            // Convert counts to transmission: T = sample/open_beam
            let (transmission, uncertainty) =
                counts_to_transmission(sample_counts, open_beam_counts);
            fit_transmission_lm(&transmission, &uncertainty, config, lm_cfg)
        }

        // ── CountsWithNuisance + LM: not meaningful ──
        (InputData::CountsWithNuisance { .. }, SolverConfig::LevenbergMarquardt(_)) => {
            Err(PipelineError::InvalidParameter(
                "CountsWithNuisance requires PoissonKL solver (LM cannot use nuisance parameters)"
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

    // Background parameters
    let bg_base = param_vec.len();
    let bg_indices = if let Some(bg) = &config.transmission_background {
        append_background_params(&mut param_vec, bg);
        Some((bg_base, bg_base + 1, bg_base + 2, bg_base + 3))
    } else {
        None
    };

    let mut params = ParameterSet::new(param_vec);
    let mut lm_cfg = lm_config.clone();
    lm_cfg.compute_covariance = config.compute_covariance;

    // Build model
    let model = build_transmission_model(config, n_density_params, _temperature_index)?;

    // Dispatch with optional background wrapping
    let result = if let Some((ai, bai, bbi, bci)) = bg_indices {
        let wrapped =
            NormalizedTransmissionModel::new(&*model, config.energies(), ai, bai, bbi, bci);
        lm::levenberg_marquardt(&wrapped, measured_t, sigma, &mut params, &lm_cfg)?
    } else {
        lm::levenberg_marquardt(&*model, measured_t, sigma, &mut params, &lm_cfg)?
    };

    extract_result(config, &result, n_density_params, bg_indices)
}

/// Transmission + Poisson KL path.
fn fit_transmission_poisson(
    measured_t: &[f64],
    sigma: &[f64],
    config: &UnifiedFitConfig,
    poisson_cfg: &PoissonConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let n_density_params = config.n_density_params();

    let mut param_vec = build_density_params(config);

    // Temperature parameter (appended after densities, before background).
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

    // KL transmission background model: T_out = T_inner + b₀ + b₁/√E
    let bg_base = param_vec.len();
    let kl_bg = if config.transmission_background.is_some() {
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
        Some((bg_base, bg_base + 1))
    } else {
        None
    };

    let mut params = ParameterSet::new(param_vec);

    let model = build_transmission_model(config, n_density_params, temperature_index)?;

    let polish_cfg = if kl_bg.is_some() && temperature_index.is_some() {
        let mut cfg = poisson_cfg.clone();
        cfg.max_iter = cfg.max_iter.clamp(20, 80);
        cfg.tol_param = (cfg.tol_param * 1e-3).max(1e-12);
        cfg.gauss_newton_lambda = (cfg.gauss_newton_lambda * 0.1).max(1e-8);
        Some(cfg)
    } else {
        None
    };

    // Dispatch with KL-native background model or bare model.
    let result = if let Some((b0_idx, b1_idx)) = kl_bg {
        let inv_sqrt_e: Vec<f64> = config
            .energies()
            .iter()
            .map(|&e| 1.0 / e.max(1e-10).sqrt())
            .collect();
        let wrapped = poisson::TransmissionKLBackgroundModel {
            inner: &*model,
            inv_sqrt_energies: inv_sqrt_e,
            b0_index: b0_idx,
            b1_index: b1_idx,
            n_params: params.params.len(),
        };
        let mut pr = poisson::poisson_fit(&wrapped, measured_t, &mut params, poisson_cfg)?;
        if pr.converged
            && let Some(ref polish) = polish_cfg
        {
            let polish_result = poisson::poisson_fit(&wrapped, measured_t, &mut params, polish)?;
            if polish_result.converged {
                pr = poisson::PoissonResult {
                    iterations: pr.iterations + polish_result.iterations,
                    ..polish_result
                };
            }
        }
        poisson_to_lm_result(&wrapped, measured_t, sigma, &pr, &params)
    } else {
        let pr = poisson::poisson_fit(&*model, measured_t, &mut params, poisson_cfg)?;
        poisson_to_lm_result(&*model, measured_t, sigma, &pr, &params)
    }?;

    let densities: Vec<f64> = (0..n_density_params).map(|i| result.params[i]).collect();
    let uncertainties = if result.converged {
        result.covariance.as_ref().map(|cov| {
            (0..n_density_params)
                .map(|i| cov.get(i, i).sqrt())
                .collect::<Vec<_>>()
        })
    } else {
        None
    };

    // Extract temperature from result if fitted.
    let fitted_temp = temperature_index.map(|idx| result.params[idx]);

    if let Some((b0_idx, b1_idx)) = kl_bg {
        let b0 = result.params[b0_idx];
        let b1 = result.params[b1_idx];
        Ok(SpectrumFitResult {
            densities,
            uncertainties,
            reduced_chi_squared: result.reduced_chi_squared,
            converged: result.converged,
            iterations: result.iterations,
            temperature_k: fitted_temp,
            temperature_k_unc: None,
            anorm: 1.0,
            background: [b0, b1, 0.0],
        })
    } else {
        let mut sr = extract_result(config, &result, n_density_params, None)?;
        if let Some(t) = fitted_temp {
            sr.temperature_k = Some(t);
        }
        Ok(sr)
    }
}

/// Counts + Poisson KL path (statistically optimal).
fn fit_counts_poisson(
    sample_counts: &[f64],
    flux: &[f64],
    background: &[f64],
    config: &UnifiedFitConfig,
    poisson_cfg: &PoissonConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let n_density_params = config.n_density_params();

    let mut param_vec = build_density_params(config);

    // Temperature parameter (after densities).
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

    // KL background model: T_out = T_inner + b₀ + b₁/√E
    let bg_base = param_vec.len();
    let kl_bg = if config.transmission_background.is_some() {
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
        Some((bg_base, bg_base + 1))
    } else {
        None
    };

    let counts_bg = if let Some(bg) = config.counts_background() {
        // CountsBackgroundConfig only scales a supplied detector/background
        // reference spectrum. It does not invent one from the open beam.
        //
        // If the caller wants to fit alpha_2, they must provide a nonzero
        // background reference. This is deliberate: for MCP/TPX event data,
        // any residual ghost/detector counts are currently treated as
        // negligible unless independently characterized.
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

    let t_model = build_transmission_model(config, n_density_params, temperature_index)?;
    let n_free = params.n_free();
    let dof = sample_counts.len().saturating_sub(n_free).max(1);

    let (pr, y_model) = if let Some((b0_idx, b1_idx)) = kl_bg {
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
        let (pr, y_model) = if let Some((alpha1_idx, alpha2_idx)) = counts_bg {
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
        };
        (pr, y_model)
    } else {
        let (pr, y_model) = if let Some((alpha1_idx, alpha2_idx)) = counts_bg {
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
            // Wrap in counts model: Y = flux * T(theta) + background
            let counts_model = poisson::CountsModel {
                transmission_model: &*t_model,
                flux,
                background,
                n_params: params.params.len(),
            };
            let pr = poisson::poisson_fit(&counts_model, sample_counts, &mut params, poisson_cfg)?;
            let y_model = counts_model.evaluate(&pr.params)?;
            (pr, y_model)
        };
        (pr, y_model)
    };

    // Compute Pearson chi-squared for display
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
    let fitted_alpha1 = counts_bg.map_or(1.0, |(alpha1_idx, _)| pr.params[alpha1_idx]);
    let fitted_alpha2 = counts_bg.map_or(0.0, |(_, alpha2_idx)| pr.params[alpha2_idx]);
    let fitted_background = if let Some((b0_idx, b1_idx)) = kl_bg {
        [pr.params[b0_idx], pr.params[b1_idx], fitted_alpha2]
    } else {
        [0.0, 0.0, fitted_alpha2]
    };

    Ok(SpectrumFitResult {
        densities,
        uncertainties: None,
        reduced_chi_squared: chi_sq / dof as f64,
        converged: pr.converged,
        iterations: pr.iterations,
        temperature_k: fitted_temp,
        temperature_k_unc: None,
        anorm: fitted_alpha1,
        background: fitted_background,
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

fn append_background_params(param_vec: &mut Vec<FitParameter>, bg: &BackgroundConfig) {
    // Anorm bounded to [0.5, 2.0] — physically reasonable normalization range.
    // Previously unbounded [0, ∞), which allowed the fitter to absorb signal
    // into anorm (e.g., anorm=15.9 with density=0.03×true).
    // SAMMY also bounds normalization to a reasonable range.
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
        return Ok(Box::new(PrecomputedTransmissionModel {
            cross_sections: effective_xs,
            density_indices: Arc::new((0..n_params).collect()),
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
        covariance: None,
        uncertainties: None,
    })
}

/// Extract SpectrumFitResult from solver output.
fn extract_result(
    config: &UnifiedFitConfig,
    result: &LmResult,
    n_density_params: usize,
    bg_indices: Option<(usize, usize, usize, usize)>,
) -> Result<SpectrumFitResult, PipelineError> {
    let densities: Vec<f64> = (0..n_density_params).map(|i| result.params[i]).collect();

    let (anorm, background) = if let Some((ai, bai, bbi, bci)) = bg_indices {
        (
            result.params[ai],
            [result.params[bai], result.params[bbi], result.params[bci]],
        )
    } else {
        (1.0, [0.0, 0.0, 0.0])
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
        let background: Vec<f64> = energies
            .iter()
            .map(|&e| 30.0 + 8.0 / e.sqrt())
            .collect();
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
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.005,
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
}
