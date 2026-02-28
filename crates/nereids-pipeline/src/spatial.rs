//! Spatial mapping: per-pixel fitting with rayon parallelization.
//!
//! Applies the single-spectrum fitting pipeline across all pixels in
//! a hyperspectral neutron imaging dataset to produce 2D composition maps.

use ndarray::{Array2, Array3};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use nereids_physics::transmission::{InstrumentParams, broadened_cross_sections};

use crate::error::PipelineError;
use crate::pipeline::{FitConfig, SpectrumFitResult, fit_spectrum};

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
/// * `cancel` — Optional cancellation token. When set to `true`, in-flight
///   rayon tasks finish but no new pixels are started. The function returns
///   `Err(PipelineError::Cancelled)`.
///
/// # Cancellation semantics
/// If cancellation occurs mid-flight, pixels that already completed are
/// included in the result.  The function returns `Err(Cancelled)` only if
/// **no** pixels completed before cancellation was detected.  When at least
/// one pixel finished, the partial result is returned as `Ok(SpatialResult)`
/// with `n_total` reflecting only the completed pixels.
///
/// # Errors
/// Returns `PipelineError::ShapeMismatch` if array dimensions are inconsistent,
/// or `PipelineError::Cancelled` if the cancellation token is set before any
/// pixel completes.
///
/// # Returns
/// Spatial result with density maps, uncertainty maps, and fit quality.
pub fn spatial_map(
    transmission: &Array3<f64>,
    uncertainty: &Array3<f64>,
    config: &FitConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
) -> Result<SpatialResult, PipelineError> {
    let shape = transmission.shape();
    let (n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data.len();

    // Validate shapes
    if uncertainty.shape() != transmission.shape() {
        return Err(PipelineError::ShapeMismatch(format!(
            "uncertainty shape {:?} != transmission shape {:?}",
            uncertainty.shape(),
            transmission.shape(),
        )));
    }
    if n_energies != config.energies.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "transmission spectral axis ({}) != config.energies length ({})",
            n_energies,
            config.energies.len(),
        )));
    }
    if let Some(dp) = dead_pixels
        && dp.shape() != [height, width]
    {
        return Err(PipelineError::ShapeMismatch(format!(
            "dead_pixels shape {:?} != spatial dimensions ({}, {})",
            dp.shape(),
            height,
            width,
        )));
    }

    // Up-front config validation: catch configuration-level errors before
    // entering the expensive precompute + pixel loop.  Per-pixel fit failures
    // are extremely rare once these hold, but without this gate a config bug
    // would be silently swallowed by the filter_map below.
    if n_energies == 0 {
        return Err(PipelineError::InvalidParameter(
            "spectral axis length is zero; at least one energy bin is required".into(),
        ));
    }
    if config.resonance_data.is_empty() {
        return Err(PipelineError::InvalidParameter(
            "resonance_data is empty — nothing to fit".into(),
        ));
    }
    if config.initial_densities.len() != n_isotopes {
        return Err(PipelineError::ShapeMismatch(format!(
            "initial_densities length ({}) != resonance_data length ({})",
            config.initial_densities.len(),
            n_isotopes,
        )));
    }
    if config.fit_temperature && config.temperature_k < 1.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "fit_temperature requires temperature_k >= 1.0, got {}",
            config.temperature_k,
        )));
    }
    if !config.temperature_k.is_finite() {
        return Err(PipelineError::InvalidParameter(format!(
            "temperature_k must be finite, got {}",
            config.temperature_k,
        )));
    }

    // Collect live pixel coordinates first (cheap).  We must know there is
    // actual work to do before paying the expensive precompute cost below.
    let mut pixel_coords: Vec<(usize, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let is_dead = dead_pixels.is_some_and(|m| m[[y, x]]);
            if !is_dead {
                pixel_coords.push((y, x));
            }
        }
    }

    // Bail out before the expensive precompute if:
    //   (a) the cancel flag is already set, or
    //   (b) every pixel is masked dead — nothing to fit.
    let empty_result = || SpatialResult {
        density_maps: (0..n_isotopes)
            .map(|_| Array2::zeros((height, width)))
            .collect(),
        uncertainty_maps: (0..n_isotopes)
            .map(|_| Array2::from_elem((height, width), f64::NAN))
            .collect(),
        chi_squared_map: Array2::from_elem((height, width), f64::NAN),
        converged_map: Array2::from_elem((height, width), false),
        n_converged: 0,
        n_total: 0,
    };
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(PipelineError::Cancelled);
    }
    if pixel_coords.is_empty() {
        return Ok(empty_result());
    }

    // When temperature is free, cross-sections are recomputed each iteration
    // inside the forward model, so precomputing here would be wasted work.
    // Only precompute when temperature is fixed.
    let fast_config = if config.fit_temperature {
        config.clone()
    } else {
        // Use caller-supplied precomputed cross-sections when available; only call
        // broadened_cross_sections when none are provided.  This lets repeated
        // spatial_map calls with the same isotopes/energy grid share one precompute
        // result and avoids redundant Doppler+resolution broadening work.
        let xs: Arc<Vec<Vec<f64>>> = match config.precomputed_cross_sections.clone() {
            Some(cached) => cached,
            None => {
                let instrument_params = config.resolution.as_ref().map(|r| InstrumentParams {
                    resolution: r.clone(),
                });
                // Pass the cancel token so precompute can bail between isotopes.
                let xs = broadened_cross_sections(
                    &config.energies,
                    &config.resonance_data,
                    config.temperature_k,
                    instrument_params.as_ref(),
                    cancel,
                )?;
                Arc::new(xs)
            }
        };

        // Build a config variant with the precomputed cross-sections injected.
        // fit_spectrum will use PrecomputedTransmissionModel when this field is Some.
        FitConfig {
            precomputed_cross_sections: Some(xs),
            ..config.clone()
        }
    };

    // Fit all pixels in parallel, skipping new work when cancelled
    let results: Vec<((usize, usize), SpectrumFitResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            // Check cancellation before starting each pixel
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            // Extract spectrum for this pixel
            let t_spectrum: Vec<f64> = (0..n_energies).map(|e| transmission[[e, y, x]]).collect();
            let sigma: Vec<f64> = (0..n_energies)
                .map(|e| uncertainty[[e, y, x]].max(1e-10)) // Avoid zero uncertainty
                .collect();

            // Config-level validation has passed up-front, so per-pixel
            // fit failures should be extremely rare (numerical edge cases
            // only).  Use an explicit match to make the skip intent clear.
            match fit_spectrum(&t_spectrum, &sigma, &fast_config) {
                Ok(result) => Some(((y, x), result)),
                Err(_) => None, // per-pixel fit failure; skip pixel
            }
        })
        .collect();

    // If cancellation was triggered during pixel fitting, return Cancelled.
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) && results.is_empty() {
        return Err(PipelineError::Cancelled);
    }

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

    Ok(SpatialResult {
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        n_converged,
        n_total: results.len(),
    })
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
///
/// # Errors
/// Returns `PipelineError::InvalidParameter` if the ROI is empty or out of bounds,
/// or `PipelineError::ShapeMismatch` if config dimensions are inconsistent.
pub fn fit_roi(
    transmission: &Array3<f64>,
    uncertainty: &Array3<f64>,
    y_range: std::ops::Range<usize>,
    x_range: std::ops::Range<usize>,
    config: &FitConfig,
) -> Result<SpectrumFitResult, PipelineError> {
    let n_energies = transmission.shape()[0];

    // Validate that uncertainty and transmission have the same shape.
    if uncertainty.shape() != transmission.shape() {
        return Err(PipelineError::ShapeMismatch(format!(
            "uncertainty shape {:?} != transmission shape {:?}",
            uncertainty.shape(),
            transmission.shape(),
        )));
    }
    // Validate spectral axis matches config.
    if n_energies != config.energies.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "transmission spectral axis ({}) != config.energies length ({})",
            n_energies,
            config.energies.len(),
        )));
    }

    if n_energies == 0 {
        return Err(PipelineError::InvalidParameter(
            "spectral axis length is zero; at least one energy bin is required".into(),
        ));
    }
    if y_range.start >= y_range.end || x_range.start >= x_range.end {
        return Err(PipelineError::InvalidParameter(format!(
            "ROI ranges must be non-empty: y={}..{}, x={}..{}",
            y_range.start, y_range.end, x_range.start, x_range.end,
        )));
    }
    let shape = transmission.shape();
    if y_range.end > shape[1] || x_range.end > shape[2] {
        return Err(PipelineError::InvalidParameter(format!(
            "ROI exceeds image dimensions: y_end={} (max {}), x_end={} (max {})",
            y_range.end, shape[1], x_range.end, shape[2],
        )));
    }

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
    use nereids_fitting::lm::{FitModel, LmConfig};
    use nereids_fitting::transmission_model::TransmissionFitModel;

    use crate::test_helpers::u238_single_resonance;

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
            temperature_index: None,
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
            precomputed_cross_sections: None,
            fit_temperature: false,
        };

        let result = spatial_map(&transmission, &uncertainty, &config, None, None).unwrap();

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
                    y,
                    x,
                    fitted,
                    true_density,
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
            temperature_index: None,
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
            precomputed_cross_sections: None,
            fit_temperature: false,
        };

        let result = spatial_map(&transmission, &uncertainty, &config, Some(&dead), None).unwrap();

        assert_eq!(result.n_total, 3); // 4 pixels - 1 dead = 3
        assert_eq!(result.density_maps[0][[0, 0]], 0.0); // dead pixel stays at 0
    }

    #[test]
    fn test_spatial_map_rejects_shape_mismatch() {
        let data = u238_single_resonance();
        let config = FitConfig {
            energies: vec![1.0, 2.0, 3.0],
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
            precomputed_cross_sections: None,
            fit_temperature: false,
        };

        let transmission = Array3::from_elem((3, 2, 2), 0.5);
        let uncertainty = Array3::from_elem((3, 2, 3), 0.01); // different width

        let result = spatial_map(&transmission, &uncertainty, &config, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_map_rejects_dead_pixel_shape_mismatch() {
        let data = u238_single_resonance();
        let config = FitConfig {
            energies: vec![1.0, 2.0, 3.0],
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
            precomputed_cross_sections: None,
            fit_temperature: false,
        };

        let transmission = Array3::from_elem((3, 2, 2), 0.5);
        let uncertainty = Array3::from_elem((3, 2, 2), 0.01);
        let dead = Array2::from_elem((3, 2), false); // wrong shape

        let result = spatial_map(&transmission, &uncertainty, &config, Some(&dead), None);
        assert!(result.is_err());
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
            temperature_index: None,
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
            precomputed_cross_sections: None,
            fit_temperature: false,
        };

        // Fit a 2×2 ROI
        let result = fit_roi(&transmission, &uncertainty, 1..3, 1..3, &config).unwrap();

        assert!(result.converged);
        assert!(
            (result.densities[0] - true_density).abs() / true_density < 0.05,
            "ROI density = {}, expected {}",
            result.densities[0],
            true_density,
        );
    }

    #[test]
    fn test_fit_roi_rejects_empty_range() {
        let data = u238_single_resonance();
        let config = FitConfig {
            energies: vec![1.0, 2.0, 3.0],
            resonance_data: vec![data],
            isotope_names: vec!["U-238".into()],
            temperature_k: 0.0,
            resolution: None,
            initial_densities: vec![0.001],
            lm_config: LmConfig::default(),
            precomputed_cross_sections: None,
            fit_temperature: false,
        };

        let transmission = Array3::from_elem((3, 4, 4), 0.5);
        let uncertainty = Array3::from_elem((3, 4, 4), 0.01);

        let result = fit_roi(&transmission, &uncertainty, 2..2, 0..2, &config);
        assert!(result.is_err());
    }
}
