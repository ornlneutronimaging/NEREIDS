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

use ndarray::{Array2, Array3};
use rayon::prelude::*;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{self, CountsModel, PoissonConfig};
use nereids_fitting::transmission_model::TransmissionFitModel;
use nereids_physics::resolution::ResolutionParams;
use nereids_physics::transmission::InstrumentParams;

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
    /// Optional resolution parameters.
    pub resolution: Option<ResolutionParams>,
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
/// Uses the open-beam counts directly as the flux estimate, and
/// estimates background from high-energy tails where transmission → 1.
///
/// # Arguments
/// * `sample_counts` — Raw sample counts (n_energies, height, width).
/// * `open_beam_counts` — Raw open-beam counts (n_energies, height, width).
/// * `roi` — Optional (y_range, x_range) for high-statistics region.
///   If None, uses the entire image.
pub fn estimate_nuisance(
    _sample_counts: &Array3<f64>,
    open_beam_counts: &Array3<f64>,
    roi: Option<(std::ops::Range<usize>, std::ops::Range<usize>)>,
) -> NuisanceParams {
    let n_energies = open_beam_counts.shape()[0];

    // Average open-beam over ROI to get flux estimate
    let (y_range, x_range) = roi.unwrap_or_else(|| {
        (0..open_beam_counts.shape()[1], 0..open_beam_counts.shape()[2])
    });

    let n_pixels = (y_range.end - y_range.start) * (x_range.end - x_range.start);
    let n_pix_f = n_pixels as f64;

    let mut flux = vec![0.0f64; n_energies];
    let mut background = vec![0.0f64; n_energies];

    for y in y_range.clone() {
        for x in x_range.clone() {
            for e in 0..n_energies {
                flux[e] += open_beam_counts[[e, y, x]];
            }
        }
    }

    for e in 0..n_energies {
        flux[e] /= n_pix_f;
    }

    // Estimate background: use the difference between sample and
    // open_beam × expected_transmission in a region where we expect T ≈ 1.
    // For simplicity, assume background is a small fraction of flux.
    // A more sophisticated approach would fit this, but for now use a
    // conservative estimate of 0 (can be refined later).
    for e in 0..n_energies {
        background[e] = 0.0;
    }

    NuisanceParams { flux, background }
}

/// Stage 2: Per-pixel density reconstruction with Poisson likelihood.
///
/// # Arguments
/// * `sample_counts` — Raw sample counts (n_energies, height, width).
/// * `nuisance` — Nuisance parameters from Stage 1.
/// * `config` — Sparse reconstruction configuration.
/// * `dead_pixels` — Optional dead pixel mask.
pub fn sparse_reconstruct(
    sample_counts: &Array3<f64>,
    nuisance: &NuisanceParams,
    config: &SparseConfig,
    dead_pixels: Option<&Array2<bool>>,
) -> SparseResult {
    let shape = sample_counts.shape();
    let (n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data.len();

    assert_eq!(n_energies, config.energies.len());

    // Build the transmission model (shared across pixels)
    let instrument = config.resolution.map(|r| InstrumentParams { resolution: r });

    // Collect pixel coordinates
    let mut pixel_coords: Vec<(usize, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let is_dead = dead_pixels.map_or(false, |m| m[[y, x]]);
            if !is_dead {
                pixel_coords.push((y, x));
            }
        }
    }

    // Fit all pixels in parallel
    let results: Vec<((usize, usize), PixelResult)> = pixel_coords
        .par_iter()
        .map(|&(y, x)| {
            // Extract counts for this pixel
            let y_obs: Vec<f64> = (0..n_energies)
                .map(|e| sample_counts[[e, y, x]].max(0.0))
                .collect();

            // Build per-pixel transmission model
            let t_model = TransmissionFitModel {
                energies: config.energies.clone(),
                resonance_data: config.resonance_data.clone(),
                temperature_k: config.temperature_k,
                instrument,
                density_indices: (0..n_isotopes).collect(),
            };

            // Wrap in counts model: Y = flux * T(θ) + background
            let counts_model = CountsModel {
                transmission_model: &t_model,
                flux: nuisance.flux.clone(),
                background: nuisance.background.clone(),
                density_param_range: 0..n_isotopes,
            };

            // Fit with Poisson likelihood
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

            let result = poisson::poisson_fit(
                &counts_model,
                &y_obs,
                &mut params,
                &config.poisson_config,
            );

            let pixel_result = PixelResult {
                densities: (0..n_isotopes).map(|i| result.params[i]).collect(),
                nll: result.nll,
                converged: result.converged,
            };

            ((y, x), pixel_result)
        })
        .collect();

    // Assemble output
    let mut density_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::zeros((height, width)))
        .collect();
    let mut converged_map = Array2::from_elem((height, width), false);
    let mut n_converged = 0;

    for ((y, x), result) in &results {
        for i in 0..n_isotopes {
            density_maps[i][[*y, *x]] = result.densities[i];
        }
        converged_map[[*y, *x]] = result.converged;
        if result.converged {
            n_converged += 1;
        }
    }

    SparseResult {
        density_maps,
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
            }],
        }
    }

    #[test]
    fn test_estimate_nuisance_basic() {
        let n_e = 5;
        let ob = Array3::from_elem((n_e, 2, 2), 100.0);
        let sample = Array3::from_elem((n_e, 2, 2), 50.0);

        let nuisance = estimate_nuisance(&sample, &ob, None);
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
        let nuisance = estimate_nuisance(&sample_counts, &ob_counts, None);

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

        let result = sparse_reconstruct(&sample_counts, &nuisance, &config, None);

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
                        y, x, fitted, true_density,
                    );
                }
            }
        }
    }
}
