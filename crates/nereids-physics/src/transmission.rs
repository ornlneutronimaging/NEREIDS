//! Transmission forward model via the Beer-Lambert law.
//!
//! Computes theoretical neutron transmission spectra from resonance parameters,
//! applying cross-section calculation, Doppler broadening, resolution broadening,
//! and the Beer-Lambert attenuation law.
//!
//! ## Beer-Lambert Law
//!
//! For a single isotope:
//!   T(E) = exp(-n·d·σ(E))
//!
//! For multiple isotopes:
//!   T(E) = exp(-Σᵢ nᵢ·dᵢ·σᵢ(E))
//!
//! where n is number density (atoms/cm³), d is thickness (cm),
//! and σ(E) is the total cross-section in barns (1 barn = 10⁻²⁴ cm²).
//!
//! In practice, the product n·d is expressed as "areal density" in
//! atoms/barn, so T(E) = exp(-thickness × σ(E)) with thickness in atoms/barn.
//!
//! ## SAMMY Reference
//! - `cro/` and `xxx/` modules — cross-section to transmission conversion
//! - Manual Section 2 (transmission definition), Section 5 (experimental corrections)

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use nereids_endf::resonance::ResonanceData;

use crate::doppler::{self, DopplerParams, DopplerParamsError};
use crate::reich_moore;
use crate::resolution::{self, ResolutionError, ResolutionFunction};

/// Build the auxiliary extended grid for resolution broadening.
///
/// Shared helper that extracts Gaussian resolution params and resonance info
/// to build the extended grid with boundary extension + adaptive intermediate
/// points.  Returns `None` if no extension is needed (no resolution, or grid
/// unchanged).
///
/// Intermediate points are inserted only when the resolution broadening at
/// the grid midpoint uses the PW-linear Gaussian path (exp tail negligible
/// or absent).  For genuine combined-kernel cases, intermediates create
/// non-uniform spacing transitions that degrade the Xcoef quadrature.
fn build_aux_grid(
    energies: &[f64],
    instrument: Option<&InstrumentParams>,
    resonance_data: &[ResonanceData],
) -> Option<(Vec<f64>, Vec<usize>)> {
    instrument.and_then(|inst| {
        if let ResolutionFunction::Gaussian(ref params) = inst.resolution {
            // Check the Gaussian-to-exp-tail ratio at the grid midpoint to
            // decide whether intermediates help or hurt.  The ratio C =
            // W_g/(2·W_e) determines which broadening path is used per-energy
            // in resolution_broaden_presorted.  When C > 2.5 the PW-linear
            // Gaussian path is used, which benefits from intermediates.
            let use_intermediates = if energies.len() >= 2 {
                let e_mid = energies[energies.len() / 2];
                let wg = params.gaussian_width(e_mid);
                let we = params.exp_width(e_mid);
                // Matches EXP_TAIL_NEGLIGIBLE_C = 2.5 in resolution.rs
                we < 1e-60 || wg / (2.0 * we) > 2.5
            } else {
                true
            };

            // Extract (energy_eV, gd_eV) pairs for fine-structure densification.
            // gd = total resonance width, used by Fspken to identify regions
            // needing denser grid points around narrow resonances.
            // SAMMY Ref: dat/mdat4.f90 Fspken lines 243-284
            let resonances = extract_resonance_widths(resonance_data);

            let (ext_e, di) = if use_intermediates {
                crate::auxiliary_grid::build_extended_grid(energies, Some(params), &resonances)
            } else {
                crate::auxiliary_grid::build_extended_grid_boundary_only(energies, Some(params))
            };
            if ext_e.len() > energies.len() {
                Some((ext_e, di))
            } else {
                None
            }
        } else {
            None
        }
    })
}

/// Extract (energy_eV, gd_eV) pairs from resonance data for fine-structure
/// grid densification.
///
/// For LRF=1/2/3 (BW and Reich-Moore): `gd = |Γn| + |Γγ| + |Γf1| + |Γf2|`
/// For LRF=7 (R-Matrix Limited): `gd = |Γγ| + Σ|γ_i|²` (approximate)
///
/// SAMMY Ref: dat/mdat4.f90 Fspken — uses total width to define the region
/// [E_res − gd, E_res + gd] for fine-structure point insertion.
fn extract_resonance_widths(resonance_data: &[ResonanceData]) -> Vec<(f64, f64)> {
    let mut pairs = Vec::new();
    for rd in resonance_data {
        for range in &rd.ranges {
            if !range.resolved {
                continue;
            }
            // LRF=1/2/3: resonances grouped by L
            for lg in &range.l_groups {
                for res in &lg.resonances {
                    let gd = res.gn.abs() + res.gg.abs() + res.gfa.abs() + res.gfb.abs();
                    if gd > 0.0 {
                        pairs.push((res.energy, gd));
                    }
                }
            }
            // LRF=7: resonances in spin groups
            if let Some(ref rml) = range.rml {
                for sg in &rml.spin_groups {
                    for res in &sg.resonances {
                        let mut gd = res.gamma_gamma.abs();
                        for &w in &res.widths {
                            gd += w * w; // γ² approximates Γ
                        }
                        if gd > 0.0 {
                            pairs.push((res.energy, gd));
                        }
                    }
                }
            }
        }
    }
    pairs
}

/// Errors from the transmission forward model.
#[derive(Debug)]
pub enum TransmissionError {
    /// The energy grid is not sorted or has a length mismatch with data.
    Resolution(ResolutionError),
    /// Doppler broadening parameter validation failed.
    Doppler(DopplerParamsError),
    /// Doppler broadening input validation failed (e.g. length mismatch).
    DopplerBroadening(crate::doppler::DopplerError),
    /// Computation was cancelled via the cancel token.
    Cancelled,
    /// Input array mismatch (e.g. cross-sections vs thicknesses length).
    InputMismatch(String),
}

impl fmt::Display for TransmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolution(e) => write!(f, "resolution broadening error: {}", e),
            Self::Doppler(e) => write!(f, "Doppler parameter error: {}", e),
            Self::DopplerBroadening(e) => write!(f, "Doppler broadening error: {}", e),
            Self::Cancelled => write!(f, "computation cancelled"),
            Self::InputMismatch(msg) => write!(f, "input mismatch: {}", msg),
        }
    }
}

impl std::error::Error for TransmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resolution(e) => Some(e),
            Self::Doppler(e) => Some(e),
            Self::DopplerBroadening(e) => Some(e),
            Self::Cancelled => None,
            Self::InputMismatch(_) => None,
        }
    }
}

impl From<ResolutionError> for TransmissionError {
    fn from(e: ResolutionError) -> Self {
        Self::Resolution(e)
    }
}

impl From<DopplerParamsError> for TransmissionError {
    fn from(e: DopplerParamsError) -> Self {
        Self::Doppler(e)
    }
}

impl From<crate::doppler::DopplerError> for TransmissionError {
    fn from(e: crate::doppler::DopplerError) -> Self {
        Self::DopplerBroadening(e)
    }
}

/// Broadened cross-sections and their temperature derivative.
///
/// `xs[k][e]` is the Doppler+resolution-broadened cross-section for isotope
/// `k` at energy index `e`; `dxs_dt[k][e]` is the central finite-difference
/// derivative with respect to temperature.
pub type BroadenedXsWithDerivative = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Compute transmission from cross-sections via Beer-Lambert law.
///
/// T(E) = exp(-thickness × σ(E))
///
/// # Arguments
/// * `cross_sections` — Total cross-sections in barns at each energy point.
/// * `thickness` — Areal density in atoms/barn (= number_density × path_length).
///
/// # Returns
/// Transmission values (0 to 1) at each energy point.
pub fn beer_lambert(cross_sections: &[f64], thickness: f64) -> Vec<f64> {
    cross_sections
        .iter()
        .map(|&sigma| (-thickness * sigma).exp())
        .collect()
}

/// Compute transmission for multiple isotopes.
///
/// T(E) = exp(-Σᵢ thicknessᵢ × σᵢ(E))
///
/// # Arguments
/// * `cross_sections_per_isotope` — Vec of cross-section arrays, one per isotope.
///   Each inner slice has the same length as the energy grid.
/// * `thicknesses` — Areal density (atoms/barn) for each isotope.
///
/// # Returns
/// Combined transmission values at each energy point.
pub fn beer_lambert_multi(
    cross_sections_per_isotope: &[&[f64]],
    thicknesses: &[f64],
) -> Result<Vec<f64>, TransmissionError> {
    if cross_sections_per_isotope.len() != thicknesses.len() {
        return Err(TransmissionError::InputMismatch(format!(
            "cross_sections_per_isotope length ({}) must match thicknesses length ({})",
            cross_sections_per_isotope.len(),
            thicknesses.len()
        )));
    }
    if cross_sections_per_isotope.is_empty() {
        return Err(TransmissionError::InputMismatch(
            "cross_sections_per_isotope must not be empty".into(),
        ));
    }

    let n_energies = cross_sections_per_isotope[0].len();
    for (k, sigma) in cross_sections_per_isotope.iter().enumerate() {
        if sigma.len() != n_energies {
            return Err(TransmissionError::InputMismatch(format!(
                "cross_sections_per_isotope[{}] length ({}) must match [0] length ({})",
                k,
                sigma.len(),
                n_energies
            )));
        }
    }

    Ok((0..n_energies)
        .map(|i| {
            let total_attenuation: f64 = cross_sections_per_isotope
                .iter()
                .zip(thicknesses.iter())
                .map(|(sigma, &thick)| thick * sigma[i])
                .sum();
            (-total_attenuation).exp()
        })
        .collect())
}

/// Errors from `SampleParams` construction.
#[derive(Debug, PartialEq)]
pub enum SampleParamsError {
    /// Temperature must be finite.
    NonFiniteTemperature(f64),
    /// Temperature must be non-negative.
    NegativeTemperature(f64),
}

impl fmt::Display for SampleParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteTemperature(v) => {
                write!(f, "temperature must be finite, got {v}")
            }
            Self::NegativeTemperature(v) => {
                write!(f, "temperature must be non-negative, got {v}")
            }
        }
    }
}

impl std::error::Error for SampleParamsError {}

/// Sample description for the forward model.
#[derive(Debug, Clone)]
pub struct SampleParams {
    /// Temperature in Kelvin (for Doppler broadening).
    temperature_k: f64,
    /// Isotope compositions: (resonance data, areal density in atoms/barn).
    isotopes: Vec<(ResonanceData, f64)>,
}

impl SampleParams {
    /// Create validated sample parameters.
    ///
    /// # Errors
    /// Returns `SampleParamsError::NonFiniteTemperature` if `temperature_k` is
    /// NaN or infinity.
    /// Returns `SampleParamsError::NegativeTemperature` if `temperature_k < 0.0`.
    pub fn new(
        temperature_k: f64,
        isotopes: Vec<(ResonanceData, f64)>,
    ) -> Result<Self, SampleParamsError> {
        if !temperature_k.is_finite() {
            return Err(SampleParamsError::NonFiniteTemperature(temperature_k));
        }
        if temperature_k < 0.0 {
            return Err(SampleParamsError::NegativeTemperature(temperature_k));
        }
        Ok(Self {
            temperature_k,
            isotopes,
        })
    }

    /// Returns the sample temperature in Kelvin.
    #[must_use]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Returns the isotope compositions: (resonance data, areal density).
    #[must_use]
    pub fn isotopes(&self) -> &[(ResonanceData, f64)] {
        &self.isotopes
    }
}

/// Optional instrument resolution parameters.
#[derive(Debug, Clone)]
pub struct InstrumentParams {
    /// Resolution broadening function (Gaussian or tabulated).
    pub resolution: ResolutionFunction,
}

/// Compute a complete theoretical transmission spectrum.
///
/// This is the main forward model that chains:
///   ENDF parameters → cross-sections → Doppler broadening → resolution → transmission
///
/// # Arguments
/// * `energies` — Energy grid in eV (sorted ascending).
/// * `sample` — Sample parameters (isotopes with areal densities, temperature).
/// * `instrument` — Optional instrument parameters (resolution broadening).
///
/// # Returns
/// Theoretical transmission spectrum on the energy grid.
///
/// # Errors
/// * [`TransmissionError::Resolution`] — if resolution broadening is
///   enabled (`instrument` is `Some`) and `energies` is not sorted ascending.
/// * [`TransmissionError::Doppler`] — if Doppler broadening is enabled
///   (`temperature_k > 0.0`) and `DopplerParams` validation fails
///   (e.g., non-positive or non-finite AWR).
///
/// **Note**: isotopes with thickness <= 0.0 are silently skipped
/// (they contribute zero attenuation). This allows callers to include
/// inactive isotopes in `SampleParams` without causing errors.
pub fn forward_model(
    energies: &[f64],
    sample: &SampleParams,
    instrument: Option<&InstrumentParams>,
) -> Result<Vec<f64>, TransmissionError> {
    let n = energies.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Validate energy grid once before the per-isotope loop so that
    // resolution broadening can use the presorted (unchecked) path,
    // avoiding redundant O(N) sort checks per isotope.
    if instrument.is_some() && !energies.windows(2).all(|w| w[0] <= w[1]) {
        return Err(ResolutionError::UnsortedEnergies.into());
    }

    // Build auxiliary grid with boundary extension + resonance fine-structure.
    // SAMMY Ref: dat/mdat4.f90 Escale, Fspken, Add_Pnts
    let active_rd: Vec<ResonanceData> = sample
        .isotopes()
        .iter()
        .filter(|(_, t)| *t > 0.0)
        .map(|(rd, _)| rd.clone())
        .collect();
    let ext_grid = build_aux_grid(energies, instrument, &active_rd);

    // Compute broadened cross-sections for all isotopes in parallel.
    // Each isotope's Doppler + resolution broadening is independent.
    // Skip isotopes with non-positive thickness (zero attenuation).
    let broadened: Result<Vec<(Vec<f64>, f64)>, TransmissionError> = sample
        .isotopes()
        .par_iter()
        .filter(|(_, thickness)| *thickness > 0.0)
        .map(|(res_data, thickness)| {
            let xs = if let Some((ref ext_energies, ref data_indices)) = ext_grid {
                let inst = instrument.unwrap();
                let unbroadened: Vec<f64> = ext_energies
                    .iter()
                    .map(|&e| reich_moore::cross_sections_at_energy(res_data, e).total)
                    .collect();
                let after_doppler = if sample.temperature_k() > 0.0 {
                    let params = DopplerParams::new(sample.temperature_k(), res_data.awr)?;
                    doppler::doppler_broaden(ext_energies, &unbroadened, &params)?
                } else {
                    unbroadened
                };
                let broadened = resolution::apply_resolution_presorted(
                    ext_energies,
                    &after_doppler,
                    &inst.resolution,
                );
                data_indices.iter().map(|&i| broadened[i]).collect()
            } else {
                let unbroadened: Vec<f64> = energies
                    .iter()
                    .map(|&e| reich_moore::cross_sections_at_energy(res_data, e).total)
                    .collect();
                let after_doppler = if sample.temperature_k() > 0.0 {
                    let params = DopplerParams::new(sample.temperature_k(), res_data.awr)?;
                    doppler::doppler_broaden(energies, &unbroadened, &params)?
                } else {
                    unbroadened
                };
                if let Some(inst) = instrument {
                    resolution::apply_resolution_presorted(
                        energies,
                        &after_doppler,
                        &inst.resolution,
                    )
                } else {
                    after_doppler
                }
            };
            Ok((xs, *thickness))
        })
        .collect();
    let broadened = broadened?;

    // 4. Accumulate total attenuation: Σᵢ thicknessᵢ × σᵢ(E)
    let mut total_attenuation = vec![0.0f64; n];
    for (xs, thickness) in &broadened {
        for i in 0..n {
            total_attenuation[i] += thickness * xs[i];
        }
    }

    // 5. Beer-Lambert: T = exp(-attenuation)
    Ok(total_attenuation.iter().map(|&att| (-att).exp()).collect())
}

/// Compute Doppler- and resolution-broadened cross-sections for each isotope.
///
/// This is the expensive physics step that should be done **once** before
/// fitting many pixels with the same isotopes and energy grid.  The result
/// feeds directly into `nereids_fitting::transmission_model::PrecomputedTransmissionModel`,
/// making per-pixel Beer-Lambert evaluation trivial.
///
/// # Arguments
/// * `energies`        — Energy grid in eV (sorted ascending).
/// * `resonance_data`  — Resonance parameters for each isotope.
/// * `temperature_k`   — Sample temperature for Doppler broadening.
/// * `instrument`      — Optional instrument resolution parameters.
/// * `cancel`          — Optional cancellation token.  Cancellation is checked
///   at the start of each isotope's parallel task; in-flight tasks run to
///   completion (consistent with the rayon pattern in `spatial.rs`).
///
/// # Returns
/// One cross-section vector per isotope on success.
///
/// # Errors
/// * [`TransmissionError::Cancelled`] — if the `cancel` flag was observed
///   during parallel execution (either before an isotope started or after
///   all tasks completed).
/// * [`TransmissionError::Resolution`] — if resolution broadening is enabled
///   (`instrument` is `Some`) and `energies` is not sorted ascending.
/// * [`TransmissionError::Doppler`] — if Doppler broadening is enabled
///   (`temperature_k > 0.0`) and `DopplerParams` validation fails
///   (e.g., non-positive or non-finite AWR).
pub fn broadened_cross_sections(
    energies: &[f64],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Vec<f64>>, TransmissionError> {
    // Validate energy grid once before the per-isotope loop.
    if instrument.is_some() && !energies.windows(2).all(|w| w[0] <= w[1]) {
        return Err(ResolutionError::UnsortedEnergies.into());
    }

    // Build auxiliary grid with boundary extension + resonance fine-structure.
    // SAMMY extends the energy grid beyond the data range and adds dense points
    // around narrow resonances so the broadening convolution integrals have
    // adequate quadrature points.
    // SAMMY Ref: dat/mdat4.f90 Escale+Fspken+Add_Pnts, dat/mdata.f90 Vqcon
    let ext_grid = build_aux_grid(energies, instrument, resonance_data);

    // Parallelize across isotopes — Doppler + resolution broadening for each
    // isotope is independent and this is the dominant cost in the forward model
    // pipeline.  Cancellation is checked per-isotope inside the parallel map.
    let result: Result<Vec<Vec<f64>>, TransmissionError> = resonance_data
        .par_iter()
        .map(|rd| {
            // Check cancellation before starting this isotope.
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(TransmissionError::Cancelled);
            }

            // When an extended grid is available, evaluate XS + Doppler +
            // resolution on the extended grid, then extract at data positions.
            // The boundary extension ensures the resolution broadening
            // convolution kernel is fully supported at all data points.
            let xs = if let Some((ref ext_energies, ref data_indices)) = ext_grid {
                let inst = instrument.unwrap();
                let unbroadened: Vec<f64> = ext_energies
                    .iter()
                    .map(|&e| reich_moore::cross_sections_at_energy(rd, e).total)
                    .collect();
                let after_doppler = if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden(ext_energies, &unbroadened, &params)?
                } else {
                    unbroadened
                };
                let broadened = resolution::apply_resolution_presorted(
                    ext_energies,
                    &after_doppler,
                    &inst.resolution,
                );
                data_indices.iter().map(|&i| broadened[i]).collect()
            } else {
                // No resolution broadening: original pipeline on data grid.
                let unbroadened: Vec<f64> = energies
                    .iter()
                    .map(|&e| reich_moore::cross_sections_at_energy(rd, e).total)
                    .collect();
                if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden(energies, &unbroadened, &params)?
                } else {
                    unbroadened
                }
            };

            Ok(xs)
        })
        .collect();

    // Final cancellation check: if cancel was set during parallel execution,
    // some tasks may have completed before observing it.
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(TransmissionError::Cancelled);
    }

    result
}

/// Compute Doppler+resolution-broadened cross-sections using SAMMY's
/// Beer-Lambert-aware pipeline for transmission data.
///
/// For transmission data, SAMMY applies resolution broadening to the
/// transmission T = exp(-nd×σ_D) rather than to σ_D directly.  Due to
/// Jensen's inequality (the exponential is convex), direct σ broadening
/// overestimates the effective cross section at resonance peaks.  This
/// function implements SAMMY's correct pipeline:
///
/// 1. Evaluate unbroadened σ on extended grid
/// 2. Doppler-broaden σ → σ_D
/// 3. Convert to transmission: T = exp(-nd × σ_D)
/// 4. Resolution-broaden T → T_broadened
/// 5. Convert back: σ_eff = -ln(T_broadened) / nd
///
/// SAMMY Ref: DopplerAndResolutionBroadener.cpp — resolution broadening is
/// applied after Beer-Lambert conversion in the SAMMY pipeline.
///
/// # Arguments
/// * `energies`             — Energy grid in eV (sorted ascending).
/// * `resonance_data`       — Resonance parameters for each isotope.
/// * `temperature_k`        — Sample temperature for Doppler broadening.
/// * `instrument`           — Instrument resolution parameters.
/// * `thickness_atoms_barn`  — Sample thickness n×d (atoms/barn).  Must be > 0.
/// * `cancel`               — Optional cancellation token.
///
/// # Returns
/// One effective cross-section vector per isotope on success.
pub fn broadened_cross_sections_for_transmission(
    energies: &[f64],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: &InstrumentParams,
    thickness_atoms_barn: f64,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Vec<f64>>, TransmissionError> {
    if !energies.windows(2).all(|w| w[0] <= w[1]) {
        return Err(ResolutionError::UnsortedEnergies.into());
    }

    let ext_grid = build_aux_grid(energies, Some(instrument), resonance_data);
    let nd = thickness_atoms_barn;

    let result: Result<Vec<Vec<f64>>, TransmissionError> = resonance_data
        .par_iter()
        .map(|rd| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(TransmissionError::Cancelled);
            }

            let (ext_energies, data_indices) = ext_grid
                .as_ref()
                .expect("instrument provided but no ext grid built");

            // 1. Unbroadened cross sections on extended grid.
            let unbroadened: Vec<f64> = ext_energies
                .iter()
                .map(|&e| reich_moore::cross_sections_at_energy(rd, e).total)
                .collect();

            // 2. Doppler broadening.
            let after_doppler = if temperature_k > 0.0 {
                let params = DopplerParams::new(temperature_k, rd.awr)?;
                doppler::doppler_broaden(ext_energies, &unbroadened, &params)?
            } else {
                unbroadened
            };

            // 3. Convert to transmission: T = exp(-nd × σ_D).
            let transmission: Vec<f64> = after_doppler
                .iter()
                .map(|&sigma| (-nd * sigma).exp())
                .collect();

            // 4. Resolution-broaden T.
            let t_broadened = resolution::apply_resolution_presorted(
                ext_energies,
                &transmission,
                &instrument.resolution,
            );

            // 5. Convert back to effective σ: σ_eff = -ln(T_broad) / nd.
            let sigma_eff: Vec<f64> = data_indices
                .iter()
                .map(|&i| {
                    let t = t_broadened[i].max(1e-30); // Prevent ln(0)
                    -t.ln() / nd
                })
                .collect();

            Ok(sigma_eff)
        })
        .collect();

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(TransmissionError::Cancelled);
    }

    result
}

/// Compute broadened cross-sections and their temperature derivative.
///
/// Returns `(sigma_k, dsigma_k_dT)` where `sigma_k[k][e]` is the
/// Doppler+resolution-broadened cross-section for isotope `k` at energy
/// index `e`, and `dsigma_k_dT[k][e]` is its central finite-difference
/// derivative with respect to temperature.
///
/// The derivative uses step size `dT = 1e-4 * (1 + T)`, which balances
/// truncation error and roundoff for the T ~ 1..2000 K regime relevant
/// to neutron resonance experiments.
///
/// # Cost
/// Three calls to `broadened_cross_sections` (at T, T+dT, T-dT).
///
/// # Errors
/// * [`TransmissionError::Resolution`] — if resolution broadening is
///   enabled (`instrument` is `Some`) and `energies` is not sorted ascending.
/// * [`TransmissionError::Doppler`] — if Doppler broadening is enabled
///   (`temperature_k > 0.0`) and `DopplerParams` validation fails
///   (e.g., non-positive or non-finite AWR).
pub fn broadened_cross_sections_with_derivative(
    energies: &[f64],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
) -> Result<BroadenedXsWithDerivative, TransmissionError> {
    let dt = 1e-4 * (1.0 + temperature_k);
    let t_up = temperature_k + dt;
    let t_down = (temperature_k - dt).max(0.0); // stay physical
    let actual_2dt = t_up - t_down;

    // The three broadening calls at T, T+dT, T-dT are independent —
    // run them concurrently with rayon::join.  No cancel token passed;
    // TransmissionError::{Resolution, Doppler} can occur.
    let (center_result, (up_result, down_result)) = rayon::join(
        || broadened_cross_sections(energies, resonance_data, temperature_k, instrument, None),
        || {
            rayon::join(
                || broadened_cross_sections(energies, resonance_data, t_up, instrument, None),
                || broadened_cross_sections(energies, resonance_data, t_down, instrument, None),
            )
        },
    );
    let xs_center = center_result?;
    let xs_up = up_result?;
    let xs_down = down_result?;

    let dxs_dt: Vec<Vec<f64>> = xs_up
        .iter()
        .zip(xs_down.iter())
        .map(|(up, down)| {
            up.iter()
                .zip(down.iter())
                .map(|(&u, &d)| (u - d) / actual_2dt)
                .collect()
        })
        .collect();

    Ok((xs_center, dxs_dt))
}

/// Compute unbroadened (raw Reich-Moore) cross-sections for each isotope.
///
/// This is the temperature-independent first step of the forward model.
/// The result can be cached and reused across multiple temperature evaluations
/// (e.g., during LM iterations where temperature is a free parameter).
///
/// # Returns
/// One total cross-section vector per isotope: `result[k][e]` is the
/// unbroadened total cross-section (barns) for isotope `k` at energy `e`.
pub fn unbroadened_cross_sections(
    energies: &[f64],
    resonance_data: &[ResonanceData],
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Vec<f64>>, TransmissionError> {
    let result: Result<Vec<Vec<f64>>, TransmissionError> = resonance_data
        .par_iter()
        .map(|rd| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(TransmissionError::Cancelled);
            }
            let xs: Vec<f64> = energies
                .iter()
                .map(|&e| reich_moore::cross_sections_at_energy(rd, e).total)
                .collect();
            Ok(xs)
        })
        .collect();

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(TransmissionError::Cancelled);
    }
    result
}

/// Compute Doppler- and resolution-broadened cross-sections from precomputed
/// unbroadened cross-sections.
///
/// Like [`broadened_cross_sections`] but skips the expensive Reich-Moore
/// calculation (step 1). Use [`unbroadened_cross_sections`] to compute
/// `base_xs` once, then call this function repeatedly with different
/// temperatures.
pub fn broadened_cross_sections_from_base(
    energies: &[f64],
    base_xs: &[Vec<f64>],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
) -> Result<Vec<Vec<f64>>, TransmissionError> {
    if base_xs.len() != resonance_data.len() {
        return Err(TransmissionError::InputMismatch(format!(
            "base_xs has {} isotopes but resonance_data has {}",
            base_xs.len(),
            resonance_data.len(),
        )));
    }
    for (i, row) in base_xs.iter().enumerate() {
        if row.len() != energies.len() {
            return Err(TransmissionError::InputMismatch(format!(
                "base_xs[{i}] has {} energies but expected {}",
                row.len(),
                energies.len(),
            )));
        }
    }
    if instrument.is_some() && !energies.windows(2).all(|w| w[0] <= w[1]) {
        return Err(ResolutionError::UnsortedEnergies.into());
    }

    // Build auxiliary grid with boundary extension + resonance fine-structure.
    // base_xs is on the data grid; we extend it to the aux grid by evaluating
    // cross-sections at the auxiliary-only points (cheap: only the few hundred
    // extra points, not the full grid).
    // SAMMY Ref: dat/mdat4.f90 Escale+Fspken+Add_Pnts
    let ext_grid = build_aux_grid(energies, instrument, resonance_data);

    // Build a bool mask to identify data-grid positions in the extended grid.
    let is_data_point: Option<Vec<bool>> = ext_grid.as_ref().map(|(ext_e, di)| {
        let mut mask = vec![false; ext_e.len()];
        for &idx in di {
            mask[idx] = true;
        }
        mask
    });

    base_xs
        .par_iter()
        .zip(resonance_data.par_iter())
        .map(|(xs_raw, rd)| {
            if let Some((ref ext_energies, ref data_indices)) = ext_grid {
                let inst = instrument.unwrap();
                let mask = is_data_point.as_ref().unwrap();

                // Build extended XS: copy cached data-grid values, evaluate new points.
                let mut xs_ext = vec![0.0f64; ext_energies.len()];
                for (data_i, &ext_i) in data_indices.iter().enumerate() {
                    xs_ext[ext_i] = xs_raw[data_i];
                }
                for (j, &e) in ext_energies.iter().enumerate() {
                    if !mask[j] {
                        xs_ext[j] = reich_moore::cross_sections_at_energy(rd, e).total;
                    }
                }

                let after_doppler = if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden(ext_energies, &xs_ext, &params)?
                } else {
                    xs_ext
                };
                let broadened = resolution::apply_resolution_presorted(
                    ext_energies,
                    &after_doppler,
                    &inst.resolution,
                );
                Ok(data_indices.iter().map(|&i| broadened[i]).collect())
            } else {
                // No auxiliary grid — original path on data grid.
                let after_doppler = if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden(energies, xs_raw, &params)?
                } else {
                    xs_raw.clone()
                };
                let xs = if let Some(inst) = instrument {
                    resolution::apply_resolution_presorted(
                        energies,
                        &after_doppler,
                        &inst.resolution,
                    )
                } else {
                    after_doppler
                };
                Ok(xs)
            }
        })
        .collect()
}

/// Compute broadened cross-sections and their temperature derivative from
/// precomputed unbroadened cross-sections.
///
/// Like [`broadened_cross_sections_with_derivative`] but skips the expensive
/// Reich-Moore calculation. Three calls to [`broadened_cross_sections_from_base`]
/// at T, T+dT, T−dT.
pub fn broadened_cross_sections_with_derivative_from_base(
    energies: &[f64],
    base_xs: &[Vec<f64>],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
) -> Result<BroadenedXsWithDerivative, TransmissionError> {
    let dt = 1e-4 * (1.0 + temperature_k);
    let t_up = temperature_k + dt;
    let t_down = (temperature_k - dt).max(0.0);
    let actual_2dt = t_up - t_down;

    // The three broadening calls at T, T+dT, T-dT are independent —
    // run them concurrently with rayon::join.
    let (center_result, (up_result, down_result)) = rayon::join(
        || {
            broadened_cross_sections_from_base(
                energies,
                base_xs,
                resonance_data,
                temperature_k,
                instrument,
            )
        },
        || {
            rayon::join(
                || {
                    broadened_cross_sections_from_base(
                        energies,
                        base_xs,
                        resonance_data,
                        t_up,
                        instrument,
                    )
                },
                || {
                    broadened_cross_sections_from_base(
                        energies,
                        base_xs,
                        resonance_data,
                        t_down,
                        instrument,
                    )
                },
            )
        },
    );
    let xs_center = center_result?;
    let xs_up = up_result?;
    let xs_down = down_result?;

    let dxs_dt: Vec<Vec<f64>> = xs_up
        .iter()
        .zip(xs_down.iter())
        .map(|(up, down)| {
            up.iter()
                .zip(down.iter())
                .map(|(&u, &d)| (u - d) / actual_2dt)
                .collect()
        })
        .collect();

    Ok((xs_center, dxs_dt))
}

/// Compute a transmission spectrum from precomputed unbroadened cross-sections.
///
/// Applies Doppler broadening, resolution broadening, and Beer-Lambert
/// using cached base XS. This skips the expensive Reich-Moore calculation,
/// making it suitable for use inside `TransmissionFitModel::evaluate()` when
/// temperature is a free parameter.
pub fn forward_model_from_base_xs(
    energies: &[f64],
    base_xs: &[Vec<f64>],
    resonance_data: &[ResonanceData],
    thicknesses: &[f64],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
) -> Result<Vec<f64>, TransmissionError> {
    if base_xs.len() != resonance_data.len() || thicknesses.len() != resonance_data.len() {
        return Err(TransmissionError::InputMismatch(format!(
            "forward_model_from_base_xs: base_xs({})/thicknesses({})/resonance_data({}) length mismatch",
            base_xs.len(),
            thicknesses.len(),
            resonance_data.len(),
        )));
    }
    let n = energies.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let broadened = broadened_cross_sections_from_base(
        energies,
        base_xs,
        resonance_data,
        temperature_k,
        instrument,
    )?;

    let mut total_attenuation = vec![0.0f64; n];
    for (xs, &thickness) in broadened.iter().zip(thicknesses.iter()) {
        if thickness <= 0.0 {
            continue;
        }
        for i in 0..n {
            total_attenuation[i] += thickness * xs[i];
        }
    }

    Ok(total_attenuation.iter().map(|&att| (-att).exp()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::types::Isotope;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};

    fn u238_single_resonance() -> ResonanceData {
        ResonanceData {
            isotope: Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 6.674,
                        j: 0.5,
                        gn: 1.493e-3,
                        gg: 23.0e-3,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
                r_external: vec![],
            }],
        }
    }

    #[test]
    fn test_beer_lambert_zero_thickness() {
        let xs = vec![100.0, 200.0, 300.0];
        let t = beer_lambert(&xs, 0.0);
        assert_eq!(t, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_beer_lambert_basic() {
        // σ = 100 barns, thickness = 0.01 atoms/barn
        // T = exp(-1.0) ≈ 0.3679
        let xs = vec![100.0];
        let t = beer_lambert(&xs, 0.01);
        assert!(
            (t[0] - (-1.0_f64).exp()).abs() < 1e-10,
            "T = {}, expected {}",
            t[0],
            (-1.0_f64).exp()
        );
    }

    #[test]
    fn test_beer_lambert_opaque() {
        // Very thick sample: T should be 0 (exp(-1000) underflows)
        let xs = vec![1000.0];
        let t = beer_lambert(&xs, 1.0);
        assert_eq!(t[0], 0.0, "T = {}, expected 0.0", t[0]);
    }

    #[test]
    fn test_beer_lambert_multi_additive() {
        // Two isotopes should combine additively in the exponent.
        // σ₁ = 100 barns, t₁ = 0.01 → att₁ = 1.0
        // σ₂ = 200 barns, t₂ = 0.005 → att₂ = 1.0
        // T = exp(-(1.0 + 1.0)) = exp(-2.0)
        let xs1 = vec![100.0];
        let xs2 = vec![200.0];
        let t = beer_lambert_multi(&[&xs1, &xs2], &[0.01, 0.005]).unwrap();
        assert!(
            (t[0] - (-2.0_f64).exp()).abs() < 1e-10,
            "T = {}, expected {}",
            t[0],
            (-2.0_f64).exp()
        );
    }

    #[test]
    fn test_transmission_dip_at_resonance() {
        // U-238 has a huge capture resonance at 6.674 eV.
        // A thin sample should show a transmission dip there.
        let data = u238_single_resonance();
        let thickness = 0.001; // atoms/barn (thin)

        // Evaluate at a few energies
        let energies = [1.0, 3.0, 6.674, 10.0, 20.0];
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| reich_moore::cross_sections_at_energy(&data, e).total)
            .collect();
        let trans = beer_lambert(&xs, thickness);

        // At 6.674 eV (on resonance), transmission should be much lower
        let t_on_res = trans[2];
        let t_off_res = trans[0]; // 1 eV, off resonance

        assert!(
            t_on_res < t_off_res,
            "On-resonance T ({}) should be < off-resonance T ({})",
            t_on_res,
            t_off_res
        );

        // On-resonance with huge σ (~25000 barns), T ≈ exp(-25) ≈ 0
        assert!(
            t_on_res < 0.01,
            "On-resonance T ({}) should be very small",
            t_on_res
        );
    }

    #[test]
    fn test_forward_model_no_broadening() {
        // Forward model at T=0 with no resolution should give
        // the same result as direct Beer-Lambert on unbroadened σ.
        let data = u238_single_resonance();
        let thickness = 0.001;

        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();

        // Direct calculation
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| reich_moore::cross_sections_at_energy(&data, e).total)
            .collect();
        let t_direct = beer_lambert(&xs, thickness);

        // Forward model
        let sample = SampleParams::new(0.0, vec![(data, thickness)]).unwrap();
        let t_forward = forward_model(&energies, &sample, None).unwrap();

        for i in 0..energies.len() {
            assert!(
                (t_direct[i] - t_forward[i]).abs() < 1e-10,
                "Mismatch at E={}: direct={}, forward={}",
                energies[i],
                t_direct[i],
                t_forward[i]
            );
        }
    }

    #[test]
    fn test_forward_model_with_broadening() {
        // Forward model with Doppler broadening should smooth out the
        // transmission dip, making it wider and shallower.
        let data = u238_single_resonance();
        let thickness = 0.0001; // Very thin (to avoid total absorption)

        let energies: Vec<f64> = (0..401).map(|i| 5.0 + (i as f64) * 0.01).collect();

        // Cold (no broadening)
        let sample_cold = SampleParams::new(0.0, vec![(data.clone(), thickness)]).unwrap();
        let t_cold = forward_model(&energies, &sample_cold, None).unwrap();

        // Hot (300 K Doppler)
        let sample_hot = SampleParams::new(300.0, vec![(data, thickness)]).unwrap();
        let t_hot = forward_model(&energies, &sample_hot, None).unwrap();

        // Find minima
        let min_cold = t_cold.iter().cloned().fold(f64::MAX, f64::min);
        let min_hot = t_hot.iter().cloned().fold(f64::MAX, f64::min);

        // Broadened dip should be shallower (higher minimum transmission)
        assert!(
            min_hot > min_cold,
            "Broadened min T ({}) should be > unbroadened min T ({})",
            min_hot,
            min_cold
        );
    }

    #[test]
    fn test_forward_model_multi_isotope() {
        // Two isotopes with different resonances should create two dips.
        let u238 = u238_single_resonance();

        // Create a fictitious second isotope with a resonance at 20 eV
        let other = ResonanceData {
            isotope: Isotope::new(1, 10).unwrap(),
            za: 1010,
            awr: 10.0,
            ranges: vec![ResonanceRange {
                energy_low: 0.0,
                energy_high: 100.0,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 5.0,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 10.0,
                    apl: 5.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: 20.0,
                        j: 0.5,
                        gn: 0.1,
                        gg: 0.05,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
                r_external: vec![],
            }],
        };

        let energies: Vec<f64> = (0..301).map(|i| 1.0 + (i as f64) * 0.1).collect();

        let sample = SampleParams::new(0.0, vec![(u238, 0.0001), (other, 0.0001)]).unwrap();
        let t = forward_model(&energies, &sample, None).unwrap();

        // Find the transmission near 6.674 eV (U-238 resonance)
        let idx_u238 = energies
            .iter()
            .position(|&e| (e - 6.7).abs() < 0.05)
            .unwrap();
        // Find the transmission near 20 eV (other resonance)
        let idx_other = energies
            .iter()
            .position(|&e| (e - 20.0).abs() < 0.05)
            .unwrap();
        // Off-resonance
        let idx_off = energies
            .iter()
            .position(|&e| (e - 15.0).abs() < 0.05)
            .unwrap();

        // Both dips should be visible
        assert!(
            t[idx_u238] < t[idx_off],
            "U-238 dip at 6.7 eV: T={}, off-res: T={}",
            t[idx_u238],
            t[idx_off]
        );
        assert!(
            t[idx_other] < t[idx_off],
            "Other dip at 20 eV: T={}, off-res: T={}",
            t[idx_other],
            t[idx_off]
        );
    }

    #[test]
    fn test_broadened_xs_derivative() {
        // Verify ∂σ/∂T via Richardson-like consistency: compute the derivative
        // at two different step sizes and check they agree to reasonable
        // tolerance (the internal step dT = 1e-4*(1+T) is O(h²)-accurate).
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();
        let temperature = 300.0;

        let (xs, dxs_dt) = broadened_cross_sections_with_derivative(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            None,
        )
        .unwrap();

        // Basic shape checks
        assert_eq!(xs.len(), 1, "one isotope");
        assert_eq!(dxs_dt.len(), 1, "one isotope derivative");
        assert_eq!(xs[0].len(), energies.len());
        assert_eq!(dxs_dt[0].len(), energies.len());

        // The derivative should be non-zero near the resonance at 6.674 eV
        // where Doppler broadening has a strong effect.
        let idx_res = energies
            .iter()
            .position(|&e| (e - 6.674).abs() < 0.05)
            .unwrap();
        assert!(
            dxs_dt[0][idx_res].abs() > 0.0,
            "dσ/dT should be non-zero near resonance, got {}",
            dxs_dt[0][idx_res]
        );

        // Cross-check: compute a manual FD at a larger step (10×) and verify
        // the two derivatives are consistent (within ~1% relative error on
        // the derivative near the resonance peak).
        let big_dt = 1.0; // 1 K step — much larger than internal 0.03 K
        let xs_up = broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature + big_dt,
            None,
            None,
        )
        .unwrap();
        let xs_down =
            broadened_cross_sections(&energies, &[data], temperature - big_dt, None, None).unwrap();

        let manual_deriv: Vec<f64> = xs_up[0]
            .iter()
            .zip(xs_down[0].iter())
            .map(|(&u, &d)| (u - d) / (2.0 * big_dt))
            .collect();

        // Compare near the resonance where the derivative is large.
        // Allow up to 5% relative difference due to O(h²) truncation.
        let deriv_fine = dxs_dt[0][idx_res];
        let deriv_coarse = manual_deriv[idx_res];
        let rel_err =
            (deriv_fine - deriv_coarse).abs() / deriv_fine.abs().max(deriv_coarse.abs()).max(1e-30);
        assert!(
            rel_err < 0.05,
            "FD derivatives at two step sizes disagree: fine={}, coarse={}, rel_err={}",
            deriv_fine,
            deriv_coarse,
            rel_err,
        );
    }

    #[test]
    fn test_broadened_xs_derivative_low_temperature() {
        // Regression test: derivative must have correct sign at low temperature.
        // Before fix, t_down was clamped to 0.1 K, causing actual_2dt < 0
        // and flipping the derivative sign for T < 0.1 K.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 5.0 + (i as f64) * 0.1).collect();

        // T = 0.05 K: was broken (t_down=0.1 > t_up=0.050105)
        let (xs_low, dxs_low) = broadened_cross_sections_with_derivative(
            &energies,
            std::slice::from_ref(&data),
            0.05,
            None,
        )
        .unwrap();
        assert!(!xs_low.is_empty());
        // Derivative should be finite and mostly positive (Doppler broadening
        // increases with temperature for narrow resonances).
        for deriv_vec in &dxs_low {
            for &d in deriv_vec {
                assert!(d.is_finite(), "derivative must be finite at T=0.05 K");
            }
        }

        // T = 0.0 K: edge case (forward difference only)
        let (xs_zero, dxs_zero) = broadened_cross_sections_with_derivative(
            &energies,
            std::slice::from_ref(&data),
            0.0,
            None,
        )
        .unwrap();
        assert!(!xs_zero.is_empty());
        for deriv_vec in &dxs_zero {
            for &d in deriv_vec {
                assert!(d.is_finite(), "derivative must be finite at T=0.0 K");
            }
        }
    }

    // --- SampleParams validation tests ---

    #[test]
    fn test_sample_params_valid() {
        let sample = SampleParams::new(300.0, vec![]).unwrap();
        assert!((sample.temperature_k() - 300.0).abs() < 1e-15);
        assert!(sample.isotopes().is_empty());
    }

    #[test]
    fn test_sample_params_zero_temperature() {
        let sample = SampleParams::new(0.0, vec![]).unwrap();
        assert!((sample.temperature_k()).abs() < 1e-15);
    }

    #[test]
    fn test_sample_params_rejects_negative_temperature() {
        let err = SampleParams::new(-1.0, vec![]).unwrap_err();
        assert_eq!(err, SampleParamsError::NegativeTemperature(-1.0));
    }

    #[test]
    fn test_sample_params_rejects_nan_temperature() {
        let err = SampleParams::new(f64::NAN, vec![]).unwrap_err();
        assert!(matches!(err, SampleParamsError::NonFiniteTemperature(_)));
    }

    #[test]
    fn test_sample_params_rejects_infinite_temperature() {
        let err = SampleParams::new(f64::INFINITY, vec![]).unwrap_err();
        assert!(matches!(err, SampleParamsError::NonFiniteTemperature(_)));
    }

    #[test]
    fn test_sample_params_rejects_neg_infinite_temperature() {
        let err = SampleParams::new(f64::NEG_INFINITY, vec![]).unwrap_err();
        assert!(matches!(err, SampleParamsError::NonFiniteTemperature(_)));
    }

    // --- Base XS caching tests ---

    #[test]
    fn test_forward_model_from_base_xs_matches_forward_model() {
        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        // Reference: full forward model
        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_ref = forward_model(&energies, &sample, None).unwrap();

        // Cached path: unbroadened XS → forward_model_from_base_xs
        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();
        let t_cached = forward_model_from_base_xs(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            &[thickness],
            temperature,
            None,
        )
        .unwrap();

        for (i, (&r, &c)) in t_ref.iter().zip(t_cached.iter()).enumerate() {
            assert!(
                (r - c).abs() < 1e-12,
                "Mismatch at E[{}]={}: ref={}, cached={}",
                i,
                energies[i],
                r,
                c
            );
        }
    }

    #[test]
    fn test_broadened_from_base_matches_broadened() {
        let data = u238_single_resonance();
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        let xs_ref = broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            None,
            None,
        )
        .unwrap();
        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();
        let xs_cached = broadened_cross_sections_from_base(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            temperature,
            None,
        )
        .unwrap();

        assert_eq!(xs_ref.len(), xs_cached.len());
        for (r, c) in xs_ref[0].iter().zip(xs_cached[0].iter()) {
            assert!(
                (r - c).abs() < 1e-12,
                "broadened_from_base mismatch: ref={}, cached={}",
                r,
                c
            );
        }
    }

    #[test]
    fn test_derivative_from_base_matches_derivative() {
        let data = u238_single_resonance();
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        let (xs_ref, dxs_ref) = broadened_cross_sections_with_derivative(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            None,
        )
        .unwrap();

        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();
        let (xs_cached, dxs_cached) = broadened_cross_sections_with_derivative_from_base(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            temperature,
            None,
        )
        .unwrap();

        for (r, c) in xs_ref[0].iter().zip(xs_cached[0].iter()) {
            assert!(
                (r - c).abs() < 1e-12,
                "XS mismatch: ref={}, cached={}",
                r,
                c
            );
        }
        for (r, c) in dxs_ref[0].iter().zip(dxs_cached[0].iter()) {
            assert!(
                (r - c).abs() < 1e-12,
                "dXS/dT mismatch: ref={}, cached={}",
                r,
                c
            );
        }
    }
}
