//! Spatial mapping: per-pixel fitting with rayon parallelization.
//!
//! Applies the single-spectrum fitting pipeline across all pixels in
//! a hyperspectral neutron imaging dataset to produce 2D composition maps.

use ndarray::{Array2, Array3};
use rayon::prelude::*;

use crate::pipeline::{fit_spectrum, FitConfig, SpectrumFitResult};

/// Result of spatial mapping over a 2D image.
#[derive(Debug)]
pub struct SpatialResult {
    /// Fitted areal density maps, one per isotope.
    /// Each Array2 has shape (height, width).
    pub density_maps: Vec<Array2<f64>>,
    /// Uncertainty maps, one per isotope.
    pub uncertainty_maps: Vec<Array2<f64>>,
    /// Reduced chi-squared map.
    pub chi_squared_map: Array2<f64>,
    /// Convergence map (true = converged).
    pub converged_map: Array2<bool>,
    /// Number of pixels that converged.
    pub n_converged: usize,
    /// Total number of pixels fitted.
    pub n_total: usize,
}

/// Run per-pixel fitting across a transmission image stack.
///
/// # Arguments
/// * `transmission` — 3D array (n_energies, height, width) of measured transmission.
/// * `uncertainty` — 3D array (n_energies, height, width) of measurement uncertainties.
/// * `config` — Fit configuration (shared across all pixels).
/// * `dead_pixels` — Optional dead pixel mask. Dead pixels are skipped.
///
/// # Returns
/// Spatial result with density maps, uncertainty maps, and fit quality.
pub fn spatial_map(
    transmission: &Array3<f64>,
    uncertainty: &Array3<f64>,
    config: &FitConfig,
    dead_pixels: Option<&Array2<bool>>,
) -> SpatialResult {
    let shape = transmission.shape();
    let (n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data.len();

    assert_eq!(n_energies, config.energies.len());

    // Collect pixel coordinates to fit
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
    let results: Vec<((usize, usize), SpectrumFitResult)> = pixel_coords
        .par_iter()
        .map(|&(y, x)| {
            // Extract spectrum for this pixel
            let t_spectrum: Vec<f64> = (0..n_energies)
                .map(|e| transmission[[e, y, x]])
                .collect();
            let sigma: Vec<f64> = (0..n_energies)
                .map(|e| uncertainty[[e, y, x]].max(1e-10)) // Avoid zero uncertainty
                .collect();

            let result = fit_spectrum(&t_spectrum, &sigma, config);
            ((y, x), result)
        })
        .collect();

    // Assemble output maps
    let mut density_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::zeros((height, width)))
        .collect();
    let mut uncertainty_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::from_elem((height, width), f64::NAN))
        .collect();
    let mut chi_squared_map = Array2::from_elem((height, width), f64::NAN);
    let mut converged_map = Array2::from_elem((height, width), false);
    let mut n_converged = 0;

    for ((y, x), result) in &results {
        for i in 0..n_isotopes {
            density_maps[i][[*y, *x]] = result.densities[i];
            uncertainty_maps[i][[*y, *x]] = result.uncertainties[i];
        }
        chi_squared_map[[*y, *x]] = result.reduced_chi_squared;
        converged_map[[*y, *x]] = result.converged;
        if result.converged {
            n_converged += 1;
        }
    }

    SpatialResult {
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        n_converged,
        n_total: results.len(),
    }
}

/// Fit a single spectrum averaged over a region of interest.
///
/// This is useful for getting high-statistics results from a spatial region
/// before attempting per-pixel fitting.
///
/// # Arguments
/// * `transmission` — 3D array (n_energies, height, width).
/// * `uncertainty` — 3D array (n_energies, height, width).
/// * `y_range` — Row range for the ROI.
/// * `x_range` — Column range for the ROI.
/// * `config` — Fit configuration.
pub fn fit_roi(
    transmission: &Array3<f64>,
    uncertainty: &Array3<f64>,
    y_range: std::ops::Range<usize>,
    x_range: std::ops::Range<usize>,
    config: &FitConfig,
) -> SpectrumFitResult {
    let n_energies = transmission.shape()[0];
    let n_pixels = (y_range.end - y_range.start) * (x_range.end - x_range.start);

    // Average transmission over ROI
    let mut avg_t = vec![0.0f64; n_energies];
    let mut avg_unc2 = vec![0.0f64; n_energies]; // Sum of squared uncertainties

    for y in y_range.clone() {
        for x in x_range.clone() {
            for e in 0..n_energies {
                avg_t[e] += transmission[[e, y, x]];
                avg_unc2[e] += uncertainty[[e, y, x]].powi(2);
            }
        }
    }

    let n_pix_f = n_pixels as f64;
    for e in 0..n_energies {
        avg_t[e] /= n_pix_f;
        // Uncertainty of mean: σ_mean = √(Σσ²)/N
        avg_unc2[e] = (avg_unc2[e]).sqrt() / n_pix_f;
    }

    fit_spectrum(&avg_t, &avg_unc2, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::types::Isotope;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};
    use nereids_fitting::lm::{FitModel, LmConfig};
    use nereids_fitting::transmission_model::TransmissionFitModel;

    fn u238_single_resonance() -> nereids_endf::resonance::ResonanceData {
        nereids_endf::resonance::ResonanceData {
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
    fn test_spatial_map_uniform_sample() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let n_energies = energies.len();

        // Generate synthetic spectrum
        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data.clone()],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
        };
        let spectrum = model.evaluate(&[true_density]);

        // Create a 3×3 image with uniform transmission
        let height = 3;
        let width = 3;
        let mut transmission = Array3::<f64>::zeros((n_energies, height, width));
        let uncertainty = Array3::from_elem((n_energies, height, width), 0.01);

        for y in 0..height {
            for x in 0..width {
                for e in 0..n_energies {
                    transmission[[e, y, x]] = spectrum[e];
                }
            }
        }

        let config = FitConfig {
            energies,
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
        };

        let result = spatial_map(&transmission, &uncertainty, &config, None);

        assert_eq!(result.n_total, 9);
        assert_eq!(result.n_converged, 9);
        assert_eq!(result.density_maps.len(), 1);

        // All pixels should recover the true density
        for y in 0..height {
            for x in 0..width {
                let fitted = result.density_maps[0][[y, x]];
                assert!(
                    (fitted - true_density).abs() / true_density < 0.05,
                    "Pixel ({},{}) density = {}, expected {}",
                    y, x, fitted, true_density,
                );
            }
        }
    }

    #[test]
    fn test_spatial_map_with_dead_pixels() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let n_energies = energies.len();

        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data.clone()],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
        };
        let spectrum = model.evaluate(&[0.0005]);

        let height = 2;
        let width = 2;
        let mut transmission = Array3::<f64>::zeros((n_energies, height, width));
        let uncertainty = Array3::from_elem((n_energies, height, width), 0.01);

        for y in 0..height {
            for x in 0..width {
                for e in 0..n_energies {
                    transmission[[e, y, x]] = spectrum[e];
                }
            }
        }

        // Mark pixel (0,0) as dead
        let mut dead = Array2::from_elem((height, width), false);
        dead[[0, 0]] = true;

        let config = FitConfig {
            energies,
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
        };

        let result = spatial_map(&transmission, &uncertainty, &config, Some(&dead));

        assert_eq!(result.n_total, 3); // 4 pixels - 1 dead = 3
        assert_eq!(result.density_maps[0][[0, 0]], 0.0); // dead pixel stays at 0
    }

    #[test]
    fn test_fit_roi() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let n_energies = energies.len();

        let model = TransmissionFitModel {
            energies: energies.clone(),
            resonance_data: vec![data.clone()],
            temperature_k: 0.0,
            instrument: None,
            density_indices: vec![0],
        };
        let spectrum = model.evaluate(&[true_density]);

        // 4×4 image
        let height = 4;
        let width = 4;
        let mut transmission = Array3::<f64>::zeros((n_energies, height, width));
        let uncertainty = Array3::from_elem((n_energies, height, width), 0.01);

        for y in 0..height {
            for x in 0..width {
                for e in 0..n_energies {
                    transmission[[e, y, x]] = spectrum[e];
                }
            }
        }

        let config = FitConfig {
            energies,
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
        };

        // Fit a 2×2 ROI
        let result = fit_roi(
            &transmission,
            &uncertainty,
            1..3,
            1..3,
            &config,
        );

        assert!(result.converged);
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.05,
            "ROI density = {}, expected {}",
            result.densities[0],
            true_density,
        );
    }
}
