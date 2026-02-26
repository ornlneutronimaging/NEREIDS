//! TRINIDI-inspired two-stage reconstruction for low-count data.
//!
//! When per-pixel neutron counts are very low (< ~10 counts/bin), direct
//! fitting per pixel is unreliable. This module implements a two-stage approach:
//!
//! ## Stage 1: Nuisance Parameter Estimation
//! Average spectra over a high-statistics region (or full image) to estimate
//! nuisance parameters: flux spectrum, background, and normalization.
//!
//! ## Stage 2: Per-Pixel Density Reconstruction
//! With nuisance parameters fixed, reconstruct per-pixel areal densities
//! using Poisson-likelihood fitting. The forward model is:
//!
//!   Y_s(E) = α₁ · [Φ(E) · exp(-Σᵢ ρᵢ · σᵢ(E)) + α₂ · B(E)]
//!
//! where:
//! - Φ(E) = incident flux spectrum (estimated in Stage 1)
//! - ρᵢ = areal density of isotope i (fit parameter)
//! - σᵢ(E) = total cross-section of isotope i
//! - B(E) = background spectrum (estimated in Stage 1)
//! - α₁, α₂ = normalization scalers (estimated in Stage 1)
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` — Two-stage reconstruction with APGM

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array2, Array3};
use rayon::prelude::*;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{self, CountsModel, PoissonConfig};
use nereids_fitting::transmission_model::PrecomputedTransmissionModel;
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::{InstrumentParams, broadened_cross_sections};

/// Configuration for two-stage sparse reconstruction.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Energy grid in eV (ascending).
    pub energies: Vec<f64>,
    /// Resonance data for each isotope.
    pub resonance_data: Vec<ResonanceData>,
    /// Isotope names (for reporting).
    pub isotope_names: Vec<String>,
    /// Sample temperature in Kelvin.
    pub temperature_k: f64,
    /// Optional resolution function (Gaussian or tabulated).
    pub resolution: Option<ResolutionFunction>,
    /// Initial guess for areal densities.
    pub initial_densities: Vec<f64>,
    /// Poisson optimizer configuration.
    pub poisson_config: PoissonConfig,
}

/// Nuisance parameters estimated in Stage 1.
#[derive(Debug, Clone)]
pub struct NuisanceParams {
    /// Estimated incident flux spectrum (counts per energy bin).
    pub flux: Vec<f64>,
    /// Estimated background spectrum (counts per energy bin).
    pub background: Vec<f64>,
}

/// Result of sparse reconstruction for a single pixel.
#[derive(Debug, Clone)]
pub struct PixelResult {
    /// Fitted areal densities.
    pub densities: Vec<f64>,
    /// Final negative log-likelihood.
    pub nll: f64,
    /// Whether the fit converged.
    pub converged: bool,
}

/// Result of the full two-stage reconstruction.
#[derive(Debug)]
pub struct SparseResult {
    /// Density maps, one per isotope. Shape (height, width).
    pub density_maps: Vec<Array2<f64>>,
    /// Per-pixel Poisson NLL map (analogous to chi_squared_map in SpatialResult).
    pub nll_map: Array2<f64>,
    /// Convergence map.
    pub converged_map: Array2<bool>,
    /// Number converged.
    pub n_converged: usize,
    /// Total pixels fitted.
    pub n_total: usize,
    /// Nuisance parameters from Stage 1.
    pub nuisance: NuisanceParams,
}

/// Stage 1: Estimate nuisance parameters from region-averaged data.
///
/// Averages open-beam counts over the ROI (excluding dead pixels) to
/// produce a per-energy-bin flux estimate.  Background is currently set
/// to zero; a future version may fit it from a sample-free region.
///
/// # Arguments
/// * `sample_counts` — Raw sample counts (n_energies, height, width).
/// * `open_beam_counts` — Raw open-beam counts (n_energies, height, width).
/// * `roi` — Optional (y_range, x_range) for high-statistics region.
///   If None, uses the entire image.
/// * `dead_pixels` — Optional dead-pixel mask.  Dead pixels are excluded
///   from the ROI average so that suppressed open-beam counts do not bias
///   the flux estimate.
///
/// # Errors
/// Returns `Err` if every pixel in the ROI is dead, leaving zero live
/// pixels to average over.
pub fn estimate_nuisance(
    _sample_counts: &Array3<f64>,
    open_beam_counts: &Array3<f64>,
    roi: Option<(std::ops::Range<usize>, std::ops::Range<usize>)>,
    dead_pixels: Option<&Array2<bool>>,
) -> Result<NuisanceParams, String> {
    let n_energies = open_beam_counts.shape()[0];
    let height = open_beam_counts.shape()[1];
    let width = open_beam_counts.shape()[2];

    // Average open-beam over ROI to get flux estimate
    let (y_range, x_range) = match roi {
        Some((y_r, x_r)) => {
            if y_r.start >= y_r.end || x_r.start >= x_r.end {
                return Err("ROI y-range or x-range is empty".into());
            }
            if y_r.start >= height || y_r.end > height || x_r.start >= width || x_r.end > width {
                return Err("ROI is out of bounds of the open-beam data".into());
            }
            (y_r, x_r)
        }
        None => (0..height, 0..width),
    };

    let mut flux = vec![0.0f64; n_energies];
    let mut background = vec![0.0f64; n_energies];
    let mut n_pixels: usize = 0;

    for y in y_range {
        for x in x_range.clone() {
            if dead_pixels.is_some_and(|m| m[[y, x]]) {
                continue;
            }
            for e in 0..n_energies {
                flux[e] += open_beam_counts[[e, y, x]];
            }
            n_pixels += 1;
        }
    }

    if n_pixels == 0 {
        return Err(
            "no live pixels in ROI after dead-pixel filtering; cannot estimate nuisance flux"
                .into(),
        );
    }

    let n_pix_f = n_pixels as f64;
    for f in &mut flux {
        *f /= n_pix_f;
    }

    // Background estimation: currently hardcoded to zero.  A future version
    // could fit background from a sample-free ROI, but that requires knowing
    // which pixels are sample-free.  Dead-pixel filtering above is still
    // applied to both flux and background accumulators so that when real
    // background estimation is added, the infrastructure is already in place.
    background.fill(0.0);

    Ok(NuisanceParams { flux, background })
}

/// Stage 2: Per-pixel density reconstruction with Poisson likelihood.
///
/// # Arguments
/// * `sample_counts` — Raw sample counts (n_energies, height, width).
/// * `nuisance` — Nuisance parameters from Stage 1.
/// * `config` — Sparse reconstruction configuration.
/// * `dead_pixels` — Optional dead pixel mask.
/// * `cancel` — Optional cancellation token. Checked per-pixel to stop early.
pub fn sparse_reconstruct(
    sample_counts: &Array3<f64>,
    nuisance: &NuisanceParams,
    config: &SparseConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
) -> SparseResult {
    let shape = sample_counts.shape();
    let (n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data.len();

    // Guard against zero-isotope calls: with no isotopes, PrecomputedTransmissionModel
    // would receive an empty cross_sections Vec and the first evaluate() would panic.
    // Return a neutral result (T=1 everywhere) rather than crashing.
    if n_isotopes == 0 {
        return SparseResult {
            density_maps: vec![],
            nll_map: Array2::from_elem((height, width), f64::NAN),
            converged_map: Array2::from_elem((height, width), false),
            n_converged: 0,
            n_total: 0,
            nuisance: nuisance.clone(),
        };
    }

    assert_eq!(n_energies, config.energies.len());
    assert_eq!(
        nuisance.flux.len(),
        n_energies,
        "nuisance flux length ({}) must match n_energies ({})",
        nuisance.flux.len(),
        n_energies,
    );
    assert_eq!(
        nuisance.background.len(),
        n_energies,
        "nuisance background length ({}) must match n_energies ({})",
        nuisance.background.len(),
        n_energies,
    );

    // Precompute Doppler-broadened cross-sections once, outside the pixel loop.
    // The same XS apply to every pixel (same isotopes, same temperature, same energy grid).
    // Mirrors the pattern used in spatial.rs to avoid repeating expensive broadening work.
    let instrument_params = config.resolution.as_ref().map(|r| InstrumentParams {
        resolution: r.clone(),
    });
    let xs_raw = broadened_cross_sections(
        &config.energies,
        &config.resonance_data,
        config.temperature_k,
        instrument_params.as_ref(),
        cancel,
    );
    // If cancelled during XS precomputation, return an empty result immediately.
    let xs_raw = match xs_raw {
        Some(v) => v,
        None => {
            return SparseResult {
                density_maps: (0..n_isotopes)
                    .map(|_| Array2::zeros((height, width)))
                    .collect(),
                nll_map: Array2::from_elem((height, width), f64::NAN),
                converged_map: Array2::from_elem((height, width), false),
                n_converged: 0,
                n_total: 0,
                nuisance: nuisance.clone(),
            };
        }
    };
    let xs: Arc<Vec<Vec<f64>>> = Arc::new(xs_raw);
    let density_idx: Vec<usize> = (0..n_isotopes).collect();

    // Collect pixel coordinates
    let mut pixel_coords: Vec<(usize, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let is_dead = dead_pixels.is_some_and(|m| m[[y, x]]);
            if !is_dead {
                pixel_coords.push((y, x));
            }
        }
    }

    // Fit all pixels in parallel, skipping new work when cancelled
    let results: Vec<((usize, usize), PixelResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            // Check cancellation before starting each pixel
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            // Extract counts for this pixel
            let y_obs: Vec<f64> = (0..n_energies)
                .map(|e| sample_counts[[e, y, x]].max(0.0))
                .collect();

            // Build per-pixel transmission model reusing precomputed XS.
            // Arc::clone shares the cross-section data (zero-copy) across all pixels,
            // avoiding expensive repeated Doppler/resolution broadening.
            let t_model = PrecomputedTransmissionModel {
                cross_sections: Arc::clone(&xs),
                density_indices: density_idx.clone(),
            };

            // Wrap in counts model: Y = flux * T(θ) + background.
            // flux/background are borrowed (zero-copy) — no per-pixel allocation.
            let counts_model = CountsModel {
                transmission_model: &t_model,
                flux: &nuisance.flux,
                background: &nuisance.background,
                density_param_range: 0..n_isotopes,
            };

            // Fit with Poisson likelihood using analytical gradient.
            // Passing the precomputed cross-sections and flux enables
            // poisson_fit_analytic to compute the full ∂NLL/∂nₖ gradient vector
            // in one model evaluation per iteration instead of N_isotopes+1
            // evaluations with finite differences (one base + one per parameter).
            let mut params = ParameterSet::new(
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
                                .unwrap_or_else(|| format!("isotope_{}", i)),
                            d,
                        )
                    })
                    .collect(),
            );

            let result = poisson::poisson_fit_analytic(
                &counts_model,
                &y_obs,
                &nuisance.flux,
                &xs,
                t_model.density_indices.as_slice(),
                &mut params,
                &config.poisson_config,
            );

            let pixel_result = PixelResult {
                densities: (0..n_isotopes).map(|i| result.params[i]).collect(),
                nll: result.nll,
                converged: result.converged,
            };

            Some(((y, x), pixel_result))
        })
        .collect();

    // Assemble output
    let mut density_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::zeros((height, width)))
        .collect();
    let mut nll_map = Array2::from_elem((height, width), f64::NAN);
    let mut converged_map = Array2::from_elem((height, width), false);
    let mut n_converged = 0;

    for ((y, x), result) in &results {
        for (i, map) in density_maps.iter_mut().enumerate() {
            map[[*y, *x]] = result.densities[i];
        }
        nll_map[[*y, *x]] = result.nll;
        converged_map[[*y, *x]] = result.converged;
        if result.converged {
            n_converged += 1;
        }
    }

    SparseResult {
        density_maps,
        nll_map,
        converged_map,
        n_converged,
        n_total: results.len(),
        nuisance: nuisance.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::types::Isotope;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};
    use nereids_fitting::lm::FitModel;
    use nereids_fitting::transmission_model::TransmissionFitModel;

    fn u238_single_resonance() -> ResonanceData {
        ResonanceData {
            isotope: Isotope::new(92, 238),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
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
            }],
        }
    }

    #[test]
    fn test_estimate_nuisance_basic() {
        let n_e = 5;
        let ob = Array3::from_elem((n_e, 2, 2), 100.0);
        let sample = Array3::from_elem((n_e, 2, 2), 50.0);

        let nuisance = estimate_nuisance(&sample, &ob, None, None).unwrap();
        assert_eq!(nuisance.flux.len(), n_e);
        assert!((nuisance.flux[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_reconstruct_synthetic() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let n_energies = energies.len();

        // Build transmission model to generate synthetic data
        let t_model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data.clone()],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
        };
        let transmission = t_model.evaluate(&[true_density]);

        // Create synthetic counts: flux = 1000, counts = flux * T
        let flux = 1000.0;
        let height = 2;
        let width = 2;
        let mut sample_counts = Array3::<f64>::zeros((n_energies, height, width));
        let ob_counts = Array3::from_elem((n_energies, height, width), flux);

        for y in 0..height {
            for x in 0..width {
                for e in 0..n_energies {
                    sample_counts[[e, y, x]] = flux * transmission[e];
                }
            }
        }

        // Stage 1: estimate nuisance
        let nuisance = estimate_nuisance(&sample_counts, &ob_counts, None, None).unwrap();

        // Stage 2: reconstruct
        let config = SparseConfig {
            energies,
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            poisson_config: PoissonConfig::default(),
        };

        let result = sparse_reconstruct(&sample_counts, &nuisance, &config, None, None);

        assert_eq!(result.n_total, 4);
        assert!(
            result.n_converged >= 3,
            "Only {}/{} pixels converged",
            result.n_converged,
            result.n_total,
        );

        // Check that most pixels recovered approximately the true density
        for y in 0..height {
            for x in 0..width {
                if result.converged_map[[y, x]] {
                    let fitted = result.density_maps[0][[y, x]];
                    assert!(
                        (fitted - true_density).abs() / true_density < 0.2,
                        "Pixel ({},{}) density = {}, expected ~{}",
                        y,
                        x,
                        fitted,
                        true_density,
                    );
                }
            }
        }
    }
}
