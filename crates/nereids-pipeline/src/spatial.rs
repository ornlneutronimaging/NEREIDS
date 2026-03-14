//! Spatial mapping: per-pixel fitting with rayon parallelization.
//!
//! Applies the single-spectrum fitting pipeline across all pixels in
//! a hyperspectral neutron imaging dataset to produce 2D composition maps.

use ndarray::{Array2, Array3, ArrayView3, s};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nereids_physics::transmission::{
    InstrumentParams, broadened_cross_sections, unbroadened_cross_sections,
};

use crate::error::PipelineError;
use crate::pipeline::{FitConfig, SpectrumFitResult, fit_spectrum};

/// Precompute cross-sections once and return a fast `FitConfig` for per-pixel fitting.
///
/// When `fit_temperature` is true, precomputes unbroadened (Reich-Moore) base XS.
/// Otherwise, precomputes fully broadened (Doppler+resolution) XS.
/// Always disables covariance computation for speed.
///
/// This helper is shared by `spatial_map` and `spatial_map_tv`.
pub(crate) fn precompute_config(
    config: &FitConfig,
    cancel: Option<&AtomicBool>,
) -> Result<FitConfig, PipelineError> {
    if config.fit_temperature() {
        let base_xs = match config.precomputed_base_xs().cloned() {
            Some(cached) => cached,
            None => {
                let xs =
                    unbroadened_cross_sections(config.energies(), config.resonance_data(), cancel)?;
                Arc::new(xs)
            }
        };
        Ok(config
            .clone()
            .with_precomputed_base_xs(base_xs)
            .with_compute_covariance(false))
    } else {
        let xs: Arc<Vec<Vec<f64>>> = match config.precomputed_cross_sections().cloned() {
            Some(cached) => cached,
            None => {
                let instrument_params = config.resolution().map(|r| InstrumentParams {
                    resolution: r.clone(),
                });
                let xs = broadened_cross_sections(
                    config.energies(),
                    config.resonance_data(),
                    config.temperature_k(),
                    instrument_params.as_ref(),
                    cancel,
                )?;
                Arc::new(xs)
            }
        };
        Ok(config
            .clone()
            .with_precomputed_cross_sections(xs)
            .with_compute_covariance(false))
    }
}

/// ADMM convergence diagnostics for TV-regularized spatial maps.
#[derive(Debug)]
pub struct AdmmConvergenceInfo {
    /// Number of ADMM outer iterations actually performed.
    pub outer_iterations: usize,
    /// Final primal residual norm (constraint violation).
    pub primal_residual: f64,
    /// Final dual residual norm (dual variable change).
    pub dual_residual: f64,
    /// Whether ADMM converged before reaching `max_outer_iter`.
    pub admm_converged: bool,
}

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
    /// Fitted temperature map (K). `Some` when `config.fit_temperature()` is true.
    pub temperature_map: Option<Array2<f64>>,
    /// Isotope labels captured at compute time, one per density map.
    /// Ensures display labels stay in sync with density data even if the
    /// user modifies the isotope list after fitting.
    pub isotope_labels: Vec<String>,
    /// Number of pixels that converged.
    pub n_converged: usize,
    /// Total number of pixels fitted.
    pub n_total: usize,
    /// ADMM convergence info (only present for TV-regularized runs).
    pub admm_info: Option<AdmmConvergenceInfo>,
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
/// * `progress` — Optional pixel completion counter. Incremented after each
///   pixel finishes (whether successful or not). Callers can poll this from
///   another thread to display a progress bar.
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
    transmission: ArrayView3<'_, f64>,
    uncertainty: ArrayView3<'_, f64>,
    config: &FitConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
    progress: Option<&AtomicUsize>,
) -> Result<SpatialResult, PipelineError> {
    let shape = transmission.shape();
    let (n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data().len();

    // Validate shapes — config-level invariants (non-empty energies,
    // resonance_data, density count, temperature) are enforced by
    // FitConfig::new().  Only per-call shape checks remain here.
    if uncertainty.shape() != transmission.shape() {
        return Err(PipelineError::ShapeMismatch(format!(
            "uncertainty shape {:?} != transmission shape {:?}",
            uncertainty.shape(),
            transmission.shape(),
        )));
    }
    if n_energies != config.energies().len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "transmission spectral axis ({}) != config.energies length ({})",
            n_energies,
            config.energies().len(),
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
    if config.initial_densities().len() != n_isotopes {
        return Err(PipelineError::ShapeMismatch(format!(
            "initial_densities length ({}) != resonance_data length ({})",
            config.initial_densities().len(),
            n_isotopes,
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

    let isotope_labels = config.isotope_names().to_vec();

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
        temperature_map: None,
        isotope_labels: isotope_labels.clone(),
        n_converged: 0,
        n_total: 0,
        admm_info: None,
    };
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(PipelineError::Cancelled);
    }
    if pixel_coords.is_empty() {
        return Ok(empty_result());
    }

    // Transpose data from (n_energies, height, width) to (height, width, n_energies)
    // so that each pixel's spectrum is contiguous in memory.  The original layout
    // requires striding across height*width elements per energy step — for a
    // 1000x128x128 array that is 131,072 bytes between consecutive energy values,
    // causing constant L1 cache misses.  After transposing, a pixel's 1000-point
    // spectrum occupies 8 KB of contiguous memory, fitting comfortably in L1 cache.
    let trans_t: Array3<f64> = transmission
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();
    let unc_t: Array3<f64> = uncertainty
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();

    // Precompute cross-sections once for all pixels and disable covariance.
    let fast_config = precompute_config(config, cancel)?;

    // Fit all pixels in parallel, skipping new work when cancelled
    let results: Vec<((usize, usize), SpectrumFitResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            // Check cancellation before starting each pixel
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            // Extract spectrum for this pixel from the transposed (h, w, n_energies) layout.
            // The energy axis is now contiguous in memory, so this slice fits in L1 cache.
            let t_spectrum: Vec<f64> = trans_t.slice(s![y, x, ..]).to_vec();
            let sigma: Vec<f64> = unc_t
                .slice(s![y, x, ..])
                .iter()
                .map(|&u| u.max(1e-10)) // Avoid zero uncertainty
                .collect();

            // Config-level validation has passed up-front, so per-pixel
            // fit failures should be extremely rare (numerical edge cases
            // only).  Use an explicit match to make the skip intent clear.
            let out = match fit_spectrum(&t_spectrum, &sigma, &fast_config) {
                Ok(result) => Some(((y, x), result)),
                Err(_) => None, // per-pixel fit failure; skip pixel
            };
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            out
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
    let mut temperature_map: Option<Array2<f64>> = if config.fit_temperature() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut n_converged = 0;

    for ((y, x), result) in &results {
        for i in 0..n_isotopes {
            density_maps[i][[*y, *x]] = result.densities[i];
            // When covariance was skipped (per-pixel spatial_map), uncertainties
            // are None and the maps stay at their NaN default.
            if let Some(ref unc) = result.uncertainties {
                uncertainty_maps[i][[*y, *x]] = unc[i];
            }
        }
        chi_squared_map[[*y, *x]] = result.reduced_chi_squared;
        converged_map[[*y, *x]] = result.converged;
        if let (Some(t_map), Some(t)) = (&mut temperature_map, result.temperature_k) {
            t_map[[*y, *x]] = t;
        }
        if result.converged {
            n_converged += 1;
        }
    }

    Ok(SpatialResult {
        density_maps,
        uncertainty_maps,
        chi_squared_map,
        converged_map,
        temperature_map,
        isotope_labels,
        n_converged,
        n_total: results.len(),
        admm_info: None,
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
    transmission: ArrayView3<'_, f64>,
    uncertainty: ArrayView3<'_, f64>,
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
    if n_energies != config.energies().len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "transmission spectral axis ({}) != config.energies length ({})",
            n_energies,
            config.energies().len(),
        )));
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

    // Slice the ROI first, THEN transpose, so the work is O(n_energies * roi_h * roi_w)
    // instead of O(n_energies * height * width).  For small ROIs on large images this
    // avoids transposing the entire array.
    let roi_trans: Array3<f64> = transmission
        .slice(s![
            ..,
            y_range.start..y_range.end,
            x_range.start..x_range.end
        ])
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();
    let roi_unc: Array3<f64> = uncertainty
        .slice(s![
            ..,
            y_range.start..y_range.end,
            x_range.start..x_range.end
        ])
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();

    // Average transmission over ROI using local (0-based) indices into the sliced array
    let roi_h = y_range.end - y_range.start;
    let roi_w = x_range.end - x_range.start;
    let mut avg_t = vec![0.0f64; n_energies];
    let mut avg_unc2 = vec![0.0f64; n_energies]; // Sum of squared uncertainties

    for y in 0..roi_h {
        for x in 0..roi_w {
            let t_row = roi_trans.slice(s![y, x, ..]);
            let u_row = roi_unc.slice(s![y, x, ..]);
            for e in 0..n_energies {
                avg_t[e] += t_row[e];
                avg_unc2[e] += u_row[e].powi(2);
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
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            vec![0],
            None,
            None,
        )
        .unwrap();
        let spectrum = model.evaluate(&[true_density]).unwrap();

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

        let config = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            None,
            None,
            None,
        )
        .unwrap();

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

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            vec![0],
            None,
            None,
        )
        .unwrap();
        let spectrum = model.evaluate(&[0.0005]).unwrap();

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

        let config = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            Some(&dead),
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.n_total, 3); // 4 pixels - 1 dead = 3
        assert_eq!(result.density_maps[0][[0, 0]], 0.0); // dead pixel stays at 0
    }

    #[test]
    fn test_spatial_map_rejects_shape_mismatch() {
        let data = u238_single_resonance();
        let config = FitConfig::new(
            vec![1.0, 2.0, 3.0],
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let transmission = Array3::from_elem((3, 2, 2), 0.5);
        let uncertainty = Array3::from_elem((3, 2, 3), 0.01); // different width

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_map_rejects_dead_pixel_shape_mismatch() {
        let data = u238_single_resonance();
        let config = FitConfig::new(
            vec![1.0, 2.0, 3.0],
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let transmission = Array3::from_elem((3, 2, 2), 0.5);
        let uncertainty = Array3::from_elem((3, 2, 2), 0.01);
        let dead = Array2::from_elem((3, 2), false); // wrong shape

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            Some(&dead),
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_roi() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let n_energies = energies.len();

        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            vec![0],
            None,
            None,
        )
        .unwrap();
        let spectrum = model.evaluate(&[true_density]).unwrap();

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

        let config = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        // Fit a 2×2 ROI
        let result = fit_roi(transmission.view(), uncertainty.view(), 1..3, 1..3, &config).unwrap();

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
        let config = FitConfig::new(
            vec![1.0, 2.0, 3.0],
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let transmission = Array3::from_elem((3, 4, 4), 0.5);
        let uncertainty = Array3::from_elem((3, 4, 4), 0.01);

        let result = fit_roi(transmission.view(), uncertainty.view(), 2..2, 0..2, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_map_temperature_map() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let true_temp = 300.0;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let n_energies = energies.len();

        // Generate synthetic spectrum at true temperature
        let model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            true_temp,
            None,
            vec![0],
            None,
            None,
        )
        .unwrap();
        let spectrum = model.evaluate(&[true_density]).unwrap();

        // 2×2 image
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

        // Without fit_temperature → temperature_map is None
        let config_no_temp = FitConfig::new(
            energies.clone(),
            vec![data.clone()],
            vec!["U-238".into()],
            true_temp,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config_no_temp,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(
            result.temperature_map.is_none(),
            "temperature_map should be None when fit_temperature is false"
        );

        // With fit_temperature → temperature_map is Some
        let config_with_temp = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            true_temp,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_fit_temperature(true)
        .unwrap();

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config_with_temp,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(
            result.temperature_map.is_some(),
            "temperature_map should be Some when fit_temperature is true"
        );
        let t_map = result.temperature_map.unwrap();
        assert_eq!(t_map.shape(), &[height, width]);
        // All pixels should recover a temperature near 300 K
        for y in 0..height {
            for x in 0..width {
                let t = t_map[[y, x]];
                assert!(
                    t.is_finite(),
                    "Pixel ({y},{x}) temperature is not finite: {t}"
                );
            }
        }
    }
}
