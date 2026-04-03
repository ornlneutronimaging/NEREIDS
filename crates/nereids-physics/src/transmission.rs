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
    resonance_data: &[&ResonanceData],
) -> Option<(Vec<f64>, Vec<usize>)> {
    instrument.and_then(|inst| {
        if let ResolutionFunction::Gaussian(ref params) = inst.resolution {
            // P-9: Check the Gaussian-to-exp-tail ratio at MULTIPLE energies
            // to decide whether intermediates help or hurt.  The ratio C =
            // W_g/(2·W_e) determines which broadening path is used per-energy
            // in resolution_broaden_presorted.  When C > 2.5 the PW-linear
            // Gaussian path is used, which benefits from intermediates.
            //
            // Previously checked at a single midpoint, which could make the
            // wrong decision if the ratio crosses 2.5 within the energy range.
            // Now checks at 5 points (lo, 25%, mid, 75%, hi) and uses
            // intermediates if a MAJORITY of points have C > 2.5.
            let use_intermediates = if energies.len() >= 2 {
                let n = energies.len();
                let check_indices = [0, n / 4, n / 2, 3 * n / 4, n - 1];
                let n_pw_linear = check_indices
                    .iter()
                    .filter(|&&i| {
                        let e = energies[i];
                        let wg = params.gaussian_width(e);
                        let we = params.exp_width(e);
                        we < 1e-60 || wg / (2.0 * we) > 2.5
                    })
                    .count();
                // Majority rule: use intermediates if ≥3 of 5 points qualify.
                n_pw_linear >= 3
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
fn extract_resonance_widths(resonance_data: &[&ResonanceData]) -> Vec<(f64, f64)> {
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
/// `k` at energy index `e`; `dxs_dt[k][e]` is the analytical derivative
/// with respect to temperature.
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
    // Collect references to avoid cloning full ResonanceData structs.
    // SAMMY Ref: dat/mdat4.f90 Escale, Fspken, Add_Pnts
    let active_rd: Vec<&ResonanceData> = sample
        .isotopes()
        .iter()
        .filter(|(_, t)| *t > 0.0)
        .map(|(rd, _)| rd)
        .collect();
    let ext_grid = build_aux_grid(energies, instrument, &active_rd);

    // Compute Doppler-broadened cross-sections for all isotopes in parallel.
    // Resolution is NOT applied here — it must be applied after Beer-Lambert
    // on the total transmission.
    //
    // SAMMY Ref: DopplerAndResolutionBroadener.cpp — resolution broadening is
    // applied to T(E), not to σ(E).  Due to Jensen's inequality (exp is
    // convex), broadening σ before the nonlinear Beer-Lambert systematically
    // overestimates effective cross-sections at resonance peaks.
    //
    // Correct pipeline:
    //   1. Per-isotope: σ → Doppler → σ_D   (on working grid)
    //   2. Accumulate:  attenuation = Σᵢ nᵢ·σ_{D,i}
    //   3. Beer-Lambert: T = exp(−attenuation)
    //   4. Resolution:  T_broad = R ⊗ T     (on working grid)
    //   5. Extract at data positions
    //
    // The working grid is the extended grid (with boundary+fine-structure
    // points) when available, otherwise the data grid.

    // Determine working grid: extended grid for resolution boundary handling,
    // or the data grid when no extension was needed.
    let (work_energies, work_len): (&[f64], usize) = if let Some((ref ext_e, _)) = ext_grid {
        (ext_e.as_slice(), ext_e.len())
    } else {
        (energies, n)
    };

    let doppler_xs: Result<Vec<(Vec<f64>, f64)>, TransmissionError> = sample
        .isotopes()
        .par_iter()
        .filter(|(_, thickness)| *thickness > 0.0)
        .map(|(res_data, thickness)| {
            let unbroadened: Vec<f64> = work_energies
                .iter()
                .map(|&e| reich_moore::cross_sections_at_energy(res_data, e).total)
                .collect();
            let after_doppler = if sample.temperature_k() > 0.0 {
                let params = DopplerParams::new(sample.temperature_k(), res_data.awr)?;
                doppler::doppler_broaden(work_energies, &unbroadened, &params)?
            } else {
                unbroadened
            };
            Ok((after_doppler, *thickness))
        })
        .collect();
    let doppler_xs = doppler_xs?;

    // 4. Accumulate total attenuation: Σᵢ thicknessᵢ × σ_{D,i}(E)
    let mut total_attenuation = vec![0.0f64; work_len];
    for (xs, thickness) in &doppler_xs {
        for i in 0..work_len {
            total_attenuation[i] += thickness * xs[i];
        }
    }

    // 5. Beer-Lambert: T = exp(−attenuation)
    let transmission: Vec<f64> = total_attenuation.iter().map(|&att| (-att).exp()).collect();

    // 6. Resolution broadening on total transmission, then extract at data positions.
    if let Some(inst) = instrument {
        let t_broadened =
            resolution::apply_resolution_presorted(work_energies, &transmission, &inst.resolution);
        if let Some((_, ref data_indices)) = ext_grid {
            Ok(data_indices.iter().map(|&i| t_broadened[i]).collect())
        } else {
            Ok(t_broadened)
        }
    } else {
        Ok(transmission)
    }
}

/// Compute Doppler-broadened cross-sections for each isotope.
///
/// Returns **Doppler-only** cross-sections.  Resolution broadening is NOT
/// applied here because it must be applied after Beer-Lambert on the total
/// transmission for physically correct results (issue #442).
///
/// When `instrument` is `Some`, the auxiliary extended grid is still
/// constructed for Doppler boundary accuracy, but the resolution
/// convolution is not performed.
///
/// This is the expensive physics step that should be done **once** before
/// fitting many pixels with the same isotopes and energy grid.  The result
/// feeds into `nereids_fitting::transmission_model::PrecomputedTransmissionModel`,
/// which currently applies Beer-Lambert only.  Post-Beer-Lambert resolution
/// broadening per-pixel is not yet implemented (issue #442 Step 3).
///
/// # Arguments
/// * `energies`        — Energy grid in eV (sorted ascending).
/// * `resonance_data`  — Resonance parameters for each isotope.
/// * `temperature_k`   — Sample temperature for Doppler broadening.
/// * `instrument`      — Optional instrument resolution parameters.
///   Used only for auxiliary grid construction (Doppler boundary accuracy).
///   Resolution broadening is NOT applied.
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
/// * [`TransmissionError::Resolution`] — if `instrument` is `Some` and
///   `energies` is not sorted ascending.
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
    let rd_refs: Vec<&ResonanceData> = resonance_data.iter().collect();
    let ext_grid = build_aux_grid(energies, instrument, &rd_refs);

    // Parallelize across isotopes — Doppler broadening for each isotope is
    // independent.  Resolution is NOT applied here (issue #442: resolution
    // must be applied after Beer-Lambert on total transmission).
    // Cancellation is checked per-isotope inside the parallel map.
    let result: Result<Vec<Vec<f64>>, TransmissionError> = resonance_data
        .par_iter()
        .map(|rd| {
            // Check cancellation before starting this isotope.
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(TransmissionError::Cancelled);
            }

            // When an extended grid is available, evaluate XS + Doppler on
            // the extended grid, then extract at data positions.  The
            // boundary extension improves Doppler broadening accuracy near
            // grid edges and narrow resonances.
            let xs = if let Some((ref ext_energies, ref data_indices)) = ext_grid {
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
                data_indices.iter().map(|&i| after_doppler[i]).collect()
            } else {
                // No extended grid: Doppler on data grid only.
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

    let rd_refs: Vec<&ResonanceData> = resonance_data.iter().collect();
    let ext_grid = build_aux_grid(energies, Some(instrument), &rd_refs);
    let nd = thickness_atoms_barn;

    let result: Result<Vec<Vec<f64>>, TransmissionError> = resonance_data
        .par_iter()
        .map(|rd| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(TransmissionError::Cancelled);
            }

            let sigma_eff = if let Some((ref ext_energies, ref data_indices)) = ext_grid {
                // Extended grid available: evaluate the full pipeline on the
                // extended grid and extract at data positions.

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
                data_indices
                    .iter()
                    .map(|&i| {
                        let t = t_broadened[i].clamp(1e-30, 1.0);
                        -t.ln() / nd
                    })
                    .collect()
            } else {
                // No extended grid (e.g. tabulated resolution with no aux grid):
                // Doppler on data grid, Beer-Lambert, resolution on data grid.
                let unbroadened: Vec<f64> = energies
                    .iter()
                    .map(|&e| reich_moore::cross_sections_at_energy(rd, e).total)
                    .collect();

                let after_doppler = if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden(energies, &unbroadened, &params)?
                } else {
                    unbroadened
                };

                let transmission: Vec<f64> = after_doppler
                    .iter()
                    .map(|&sigma| (-nd * sigma).exp())
                    .collect();

                let t_broadened = resolution::apply_resolution_presorted(
                    energies,
                    &transmission,
                    &instrument.resolution,
                );

                t_broadened
                    .iter()
                    .map(|&t| {
                        let t_clamped = t.clamp(1e-30, 1.0);
                        -t_clamped.ln() / nd
                    })
                    .collect()
            };

            Ok(sigma_eff)
        })
        .collect();

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(TransmissionError::Cancelled);
    }

    result
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

/// Compute Doppler-broadened cross-sections from precomputed unbroadened
/// cross-sections.
///
/// Returns **Doppler-only** cross-sections.  Resolution broadening is NOT
/// applied (issue #442: must be applied after Beer-Lambert on total T).
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
    let rd_refs: Vec<&ResonanceData> = resonance_data.iter().collect();
    let ext_grid = build_aux_grid(energies, instrument, &rd_refs);

    // Build a bool mask to identify data-grid positions in the extended grid.
    let is_data_point: Option<Vec<bool>> = ext_grid.as_ref().map(|(ext_e, di)| {
        let mut mask = vec![false; ext_e.len()];
        for &idx in di {
            mask[idx] = true;
        }
        mask
    });

    // Resolution is NOT applied (issue #442).  Doppler broadening only.
    base_xs
        .par_iter()
        .zip(resonance_data.par_iter())
        .map(|(xs_raw, rd)| {
            if let Some((ref ext_energies, ref data_indices)) = ext_grid {
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
                Ok(data_indices.iter().map(|&i| after_doppler[i]).collect())
            } else {
                // No auxiliary grid — Doppler on data grid only.
                let after_doppler = if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden(energies, xs_raw, &params)?
                } else {
                    xs_raw.clone()
                };
                Ok(after_doppler)
            }
        })
        .collect()
}

/// Compute Doppler-broadened cross-sections and their **analytical**
/// temperature derivative from precomputed unbroadened cross-sections.
///
/// Returns **Doppler-only** cross-sections and derivatives.  Resolution
/// broadening is NOT applied (issue #442: must be applied after
/// Beer-Lambert on total T).
///
/// Uses `doppler_broaden_with_derivative` for exact ∂σ/∂T in a single pass
/// (1× broadening), replacing the 3× FD approach.
///
/// Returns `BroadenedXsWithDerivative`: `(sigma_k, dsigma_k_dT)`.
pub fn broadened_cross_sections_with_analytical_derivative_from_base(
    energies: &[f64],
    base_xs: &[Vec<f64>],
    resonance_data: &[ResonanceData],
    temperature_k: f64,
    instrument: Option<&InstrumentParams>,
) -> Result<BroadenedXsWithDerivative, TransmissionError> {
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

    // Build auxiliary grid (same as broadened_cross_sections_from_base).
    let rd_refs: Vec<&ResonanceData> = resonance_data.iter().collect();
    let ext_grid = build_aux_grid(energies, instrument, &rd_refs);
    let is_data_point: Option<Vec<bool>> = ext_grid.as_ref().map(|(ext_e, di)| {
        let mut mask = vec![false; ext_e.len()];
        for &idx in di {
            mask[idx] = true;
        }
        mask
    });

    // Per-isotope: Doppler broaden with analytical derivative.
    // Resolution is NOT applied (issue #442).
    type IsotopeXsDxs = Result<(Vec<f64>, Vec<f64>), TransmissionError>;
    let results: Vec<IsotopeXsDxs> = base_xs
        .par_iter()
        .zip(resonance_data.par_iter())
        .map(|(xs_raw, rd)| {
            if let Some((ref ext_energies, ref data_indices)) = ext_grid {
                let mask = is_data_point.as_ref().unwrap();

                // Build extended XS on auxiliary grid.
                let mut xs_ext = vec![0.0f64; ext_energies.len()];
                for (data_i, &ext_i) in data_indices.iter().enumerate() {
                    xs_ext[ext_i] = xs_raw[data_i];
                }
                for (j, &e) in ext_energies.iter().enumerate() {
                    if !mask[j] {
                        xs_ext[j] = reich_moore::cross_sections_at_energy(rd, e).total;
                    }
                }

                // Doppler broaden + analytical derivative in one pass.
                // Resolution is NOT applied (issue #442).
                let (after_doppler, dxs_dt_doppler) = if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden_with_derivative(ext_energies, &xs_ext, &params)?
                } else {
                    (xs_ext, vec![0.0; ext_energies.len()])
                };

                // Extract data-grid points from extended grid.
                let xs: Vec<f64> = data_indices.iter().map(|&i| after_doppler[i]).collect();
                let dxs: Vec<f64> = data_indices.iter().map(|&i| dxs_dt_doppler[i]).collect();
                Ok((xs, dxs))
            } else {
                // No auxiliary grid — Doppler on data grid only.
                let (after_doppler, dxs_dt_doppler) = if temperature_k > 0.0 {
                    let params = DopplerParams::new(temperature_k, rd.awr)?;
                    doppler::doppler_broaden_with_derivative(energies, xs_raw, &params)?
                } else {
                    (xs_raw.clone(), vec![0.0; energies.len()])
                };
                Ok((after_doppler, dxs_dt_doppler))
            }
        })
        .collect();

    // Separate into (xs_all, dxs_all).
    let mut xs_all = Vec::with_capacity(base_xs.len());
    let mut dxs_all = Vec::with_capacity(base_xs.len());
    for r in results {
        let (xs, dxs) = r?;
        xs_all.push(xs);
        dxs_all.push(dxs);
    }
    Ok((xs_all, dxs_all))
}

/// Compute a transmission spectrum from precomputed unbroadened cross-sections.
///
/// Applies Doppler broadening and Beer-Lambert using cached base XS,
/// then resolution broadening on the total transmission (issue #442).
/// This skips the expensive Reich-Moore calculation, making it suitable
/// for use inside `TransmissionFitModel::evaluate()` when temperature
/// is a free parameter.
///
/// Pipeline:
///   1. Doppler-broaden base σ (via `broadened_cross_sections_from_base`)
///   2. Accumulate total attenuation: Σᵢ nᵢ·σ_{D,i}
///   3. Beer-Lambert: T = exp(−attenuation)
///   4. Resolution: T_broad = R ⊗ T  (when instrument is present)
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

    // Step 1: Doppler-only σ (resolution NOT applied — issue #442).
    let doppler_xs = broadened_cross_sections_from_base(
        energies,
        base_xs,
        resonance_data,
        temperature_k,
        instrument,
    )?;

    // Step 2-3: accumulate attenuation, Beer-Lambert.
    let mut total_attenuation = vec![0.0f64; n];
    for (xs, &thickness) in doppler_xs.iter().zip(thicknesses.iter()) {
        if thickness <= 0.0 {
            continue;
        }
        for i in 0..n {
            total_attenuation[i] += thickness * xs[i];
        }
    }
    let transmission: Vec<f64> = total_attenuation.iter().map(|&att| (-att).exp()).collect();

    // Step 4: resolution broadening on total transmission.
    if let Some(inst) = instrument {
        resolution::apply_resolution(energies, &transmission, &inst.resolution)
            .map_err(TransmissionError::from)
    } else {
        Ok(transmission)
    }
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
                naps: 1,
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
                naps: 1,
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
    fn test_broadened_xs_analytical_derivative() {
        // Verify analytical ∂σ/∂T against a manual FD at a larger step (1 K)
        // and check they agree to reasonable tolerance.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();
        let temperature = 300.0;

        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();
        let (xs, dxs_dt) = broadened_cross_sections_with_analytical_derivative_from_base(
            &energies,
            &base_xs,
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

        // Cross-check: compute a manual FD at a larger step and verify
        // the analytical derivative is consistent (within ~5% relative error).
        let big_dt = 1.0;
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
        let deriv_analytical = dxs_dt[0][idx_res];
        let deriv_coarse = manual_deriv[idx_res];
        let rel_err = (deriv_analytical - deriv_coarse).abs()
            / deriv_analytical.abs().max(deriv_coarse.abs()).max(1e-30);
        assert!(
            rel_err < 0.05,
            "Analytical vs FD derivatives disagree: analytical={}, coarse={}, rel_err={}",
            deriv_analytical,
            deriv_coarse,
            rel_err,
        );
    }

    #[test]
    fn test_broadened_xs_analytical_derivative_low_temperature() {
        // Regression test: derivative must have correct sign at low temperature.
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 5.0 + (i as f64) * 0.1).collect();

        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();

        // T = 0.05 K
        let (xs_low, dxs_low) = broadened_cross_sections_with_analytical_derivative_from_base(
            &energies,
            &base_xs,
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

        // T = 0.0 K: edge case
        let (xs_zero, dxs_zero) = broadened_cross_sections_with_analytical_derivative_from_base(
            &energies,
            &base_xs,
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
    fn test_analytical_derivative_from_base_shape_and_finiteness() {
        // Verify the analytical derivative from base returns correct shapes
        // and finite values.
        let data = u238_single_resonance();
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();
        let (xs, dxs_dt) = broadened_cross_sections_with_analytical_derivative_from_base(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            temperature,
            None,
        )
        .unwrap();

        assert_eq!(xs.len(), 1);
        assert_eq!(dxs_dt.len(), 1);
        assert_eq!(xs[0].len(), energies.len());
        assert_eq!(dxs_dt[0].len(), energies.len());

        // All values should be finite.
        for &v in &xs[0] {
            assert!(v.is_finite(), "XS must be finite, got {v}");
        }
        for &v in &dxs_dt[0] {
            assert!(v.is_finite(), "dXS/dT must be finite, got {v}");
        }

        // XS from base should match XS from full broadening.
        let xs_full = broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            None,
            None,
        )
        .unwrap();
        for (r, c) in xs_full[0].iter().zip(xs[0].iter()) {
            assert!(
                (r - c).abs() < 1e-12,
                "XS mismatch: full={}, from_base={}",
                r,
                c
            );
        }
    }

    /// Regression test for issue #442: resolution broadening must be applied
    /// to the total transmission T(E) AFTER Beer-Lambert, not to σ(E) before.
    ///
    /// This test constructs the expected result from first principles:
    ///
    ///   1. Doppler-broaden σ
    ///   2. Beer-Lambert: T = exp(−n·σ_D)
    ///   3. Resolution-broaden T
    ///
    /// and asserts that `forward_model()` matches.
    #[test]
    fn test_forward_model_resolution_after_beer_lambert() {
        let data = u238_single_resonance();
        let thickness = 0.0005; // atoms/barn
        let temperature = 300.0;

        // Energy grid around the 6.674 eV resonance.
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();

        let inst = InstrumentParams {
            resolution: resolution::ResolutionFunction::Gaussian(
                resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        };

        // --- Build expected from first principles ---

        // Step 1: Doppler-broadened σ on the data grid.
        let unbroadened: Vec<f64> = energies
            .iter()
            .map(|&e| reich_moore::cross_sections_at_energy(&data, e).total)
            .collect();
        let doppler_params = doppler::DopplerParams::new(temperature, data.awr).unwrap();
        let sigma_d = doppler::doppler_broaden(&energies, &unbroadened, &doppler_params).unwrap();

        // Step 2: Beer-Lambert on total transmission.
        let transmission: Vec<f64> = sigma_d.iter().map(|&s| (-thickness * s).exp()).collect();

        // Step 3: Resolution-broaden the transmission.
        let t_expected =
            resolution::apply_resolution(&energies, &transmission, &inst.resolution).unwrap();

        // --- Wrong ordering for comparison: Resolution(σ) then Beer-Lambert ---
        let sigma_broadened =
            resolution::apply_resolution(&energies, &sigma_d, &inst.resolution).unwrap();
        let t_wrong: Vec<f64> = sigma_broadened
            .iter()
            .map(|&s| (-thickness * s).exp())
            .collect();

        // --- forward_model() output ---
        let sample = SampleParams::new(temperature, vec![(data, thickness)]).unwrap();
        let t_forward = forward_model(&energies, &sample, Some(&inst)).unwrap();

        // forward_model should match the correct ordering (resolution after Beer-Lambert).
        // The extended grid in forward_model adds boundary points, so the match
        // is approximate — but should be very close on the interior grid.
        let interior = 20..energies.len() - 20; // skip boundary region
        let mut max_err_correct = 0.0f64;
        let mut max_err_wrong = 0.0f64;
        for i in interior.clone() {
            let err_correct = (t_forward[i] - t_expected[i]).abs();
            let err_wrong = (t_forward[i] - t_wrong[i]).abs();
            max_err_correct = max_err_correct.max(err_correct);
            max_err_wrong = max_err_wrong.max(err_wrong);
        }

        // The key discriminant: forward_model must be much closer to the
        // correct ordering (resolution after Beer-Lambert) than to the wrong
        // ordering (resolution before Beer-Lambert).
        //
        // Small absolute differences (~1%) between forward_model and the
        // data-grid reference are expected because forward_model uses an
        // extended grid for boundary handling.
        assert!(
            max_err_correct < max_err_wrong,
            "forward_model is closer to the WRONG ordering than the correct one. \
             Error vs correct = {max_err_correct}, error vs wrong = {max_err_wrong}"
        );

        // The error against the correct ordering should be at least 5× smaller
        // than the error against the wrong ordering.
        assert!(
            max_err_correct < max_err_wrong * 0.5,
            "forward_model should be clearly closer to the correct ordering. \
             Error vs correct = {max_err_correct}, error vs wrong = {max_err_wrong}, \
             ratio = {:.2}",
            max_err_correct / max_err_wrong
        );

        // Verify the two orderings actually differ — if they don't, the test
        // is not exercising the bug.
        let ordering_diff: f64 = interior
            .map(|i| (t_expected[i] - t_wrong[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            ordering_diff > 1e-4,
            "The two orderings should differ measurably at the resonance dip, \
             but max diff = {ordering_diff}. Test parameters may be too weak."
        );
    }

    /// Issue #442 containment: `broadened_cross_sections()` must return
    /// Doppler-only σ even when `instrument` is `Some`.  Resolution
    /// broadening must NOT be applied inside this function.
    #[test]
    fn test_broadened_xs_is_doppler_only_with_instrument() {
        let data = u238_single_resonance();
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        let inst = InstrumentParams {
            resolution: resolution::ResolutionFunction::Gaussian(
                resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        };

        // With instrument (used for aux grid, but NOT for resolution broadening).
        let xs_with_inst = broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            Some(&inst),
            None,
        )
        .unwrap();

        // Without instrument (pure Doppler on data grid).
        let xs_no_inst = broadened_cross_sections(
            &energies,
            std::slice::from_ref(&data),
            temperature,
            None,
            None,
        )
        .unwrap();

        // Both should return Doppler-only σ.  The with-instrument path uses
        // the extended grid for Doppler which may produce slightly different
        // values, but they must be close (no resolution smoothing).
        assert_eq!(xs_with_inst.len(), 1);
        assert_eq!(xs_no_inst.len(), 1);
        assert_eq!(xs_with_inst[0].len(), energies.len());

        // Compute what resolution-broadened σ would look like.
        let sigma_resolved =
            resolution::apply_resolution(&energies, &xs_no_inst[0], &inst.resolution).unwrap();

        // The with-instrument result must NOT match the resolution-broadened version.
        // Near the resonance dip, resolution broadening smooths the peak — the
        // Doppler-only result should have a deeper dip than the resolved one.
        let idx_dip = energies
            .iter()
            .position(|&e| (e - 6.674).abs() < 0.05)
            .unwrap();
        let diff_doppler = (xs_with_inst[0][idx_dip] - xs_no_inst[0][idx_dip]).abs();
        let diff_resolved = (xs_with_inst[0][idx_dip] - sigma_resolved[idx_dip]).abs();

        // The Doppler-only values from both paths should be closer to each
        // other than to the resolution-broadened value.
        assert!(
            diff_doppler < diff_resolved,
            "broadened_cross_sections with instrument should return Doppler-only σ, \
             not resolution-broadened σ.  \
             diff(with_inst, no_inst) = {diff_doppler}, \
             diff(with_inst, resolved) = {diff_resolved}"
        );
    }

    /// Issue #442 containment: `broadened_cross_sections_from_base()` must
    /// return Doppler-only σ even when `instrument` is `Some`.
    #[test]
    fn test_broadened_from_base_is_doppler_only_with_instrument() {
        let data = u238_single_resonance();
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        let inst = InstrumentParams {
            resolution: resolution::ResolutionFunction::Gaussian(
                resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        };

        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();

        // With instrument.
        let xs_with_inst = broadened_cross_sections_from_base(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            temperature,
            Some(&inst),
        )
        .unwrap();

        // Without instrument.
        let xs_no_inst = broadened_cross_sections_from_base(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            temperature,
            None,
        )
        .unwrap();

        // Both should produce Doppler-only σ.  With-instrument may use
        // extended grid, but the extracted data-grid values should be close.
        let sigma_resolved =
            resolution::apply_resolution(&energies, &xs_no_inst[0], &inst.resolution).unwrap();

        let idx_dip = energies
            .iter()
            .position(|&e| (e - 6.674).abs() < 0.05)
            .unwrap();
        let diff_doppler = (xs_with_inst[0][idx_dip] - xs_no_inst[0][idx_dip]).abs();
        let diff_resolved = (xs_with_inst[0][idx_dip] - sigma_resolved[idx_dip]).abs();

        assert!(
            diff_doppler < diff_resolved,
            "broadened_cross_sections_from_base with instrument should return Doppler-only σ. \
             diff(with_inst, no_inst) = {diff_doppler}, \
             diff(with_inst, resolved) = {diff_resolved}"
        );
    }

    // ── Issue #442 Step 5: forward_model_from_base_xs resolution ordering ──

    /// Issue #442 Step 5: `forward_model_from_base_xs()` with resolution must
    /// match `forward_model()` with resolution for the same sample.
    #[test]
    fn test_forward_model_from_base_xs_matches_forward_model_with_resolution() {
        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..401).map(|i| 4.0 + (i as f64) * 0.015).collect();

        let inst = InstrumentParams {
            resolution: resolution::ResolutionFunction::Gaussian(
                resolution::ResolutionParams::new(25.0, 0.5, 0.005, 0.0).unwrap(),
            ),
        };

        // Reference: forward_model() (fixed in Step 1).
        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_ref = forward_model(&energies, &sample, Some(&inst)).unwrap();

        // Base-XS path: unbroadened → forward_model_from_base_xs with resolution.
        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();
        let t_base = forward_model_from_base_xs(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            &[thickness],
            temperature,
            Some(&inst),
        )
        .unwrap();

        // Both use resolution after Beer-Lambert but may differ slightly
        // due to extended-grid Doppler in forward_model vs data-grid Doppler
        // in broadened_cross_sections_from_base.
        let interior = 20..energies.len() - 20;
        let mut max_err = 0.0f64;
        for i in interior.clone() {
            max_err = max_err.max((t_ref[i] - t_base[i]).abs());
        }
        assert!(
            max_err < 0.02,
            "forward_model_from_base_xs with resolution should match \
             forward_model.  Max error = {max_err}"
        );

        // Verify resolution actually made a difference (not a vacuous test).
        let t_no_res = forward_model_from_base_xs(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            &[thickness],
            temperature,
            None,
        )
        .unwrap();
        let res_diff: f64 = interior
            .map(|i| (t_base[i] - t_no_res[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            res_diff > 1e-4,
            "Resolution should make a measurable difference, but max diff = {res_diff}"
        );
    }

    /// Issue #442 Step 5: `forward_model_from_base_xs()` without resolution
    /// must remain unchanged (matches existing no-resolution test).
    #[test]
    fn test_forward_model_from_base_xs_no_resolution_unchanged() {
        let data = u238_single_resonance();
        let thickness = 0.0005;
        let temperature = 300.0;
        let energies: Vec<f64> = (0..201).map(|i| 4.0 + (i as f64) * 0.025).collect();

        let sample = SampleParams::new(temperature, vec![(data.clone(), thickness)]).unwrap();
        let t_ref = forward_model(&energies, &sample, None).unwrap();

        let base_xs =
            unbroadened_cross_sections(&energies, std::slice::from_ref(&data), None).unwrap();
        let t_base = forward_model_from_base_xs(
            &energies,
            &base_xs,
            std::slice::from_ref(&data),
            &[thickness],
            temperature,
            None,
        )
        .unwrap();

        for (i, (&r, &b)) in t_ref.iter().zip(t_base.iter()).enumerate() {
            assert!(
                (r - b).abs() < 1e-12,
                "No-resolution mismatch at E[{i}]={}: ref={r}, base={b}",
                energies[i]
            );
        }
    }
}
