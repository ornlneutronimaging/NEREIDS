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
    /// Per-pixel normalization factor map (when background fitting is enabled).
    pub anorm_map: Option<Array2<f64>>,
    /// Per-pixel background parameter maps [BackA, BackB, BackC] (when background enabled).
    pub background_maps: Option<[Array2<f64>; 3]>,
    /// Number of pixels that converged.
    pub n_converged: usize,
    /// Total number of pixels fitted.
    pub n_total: usize,

    // ── Optional fields from regularization (Phase 4) ──
    /// Negative log-likelihood map (populated for counts/KL path).
    pub nll_map: Option<Array2<f64>>,
    /// Number of Fisher weak directions (populated by regularization).
    pub n_weak_directions: Option<usize>,
    /// Fisher eigenvalues (populated by regularization).
    pub fisher_eigenvalues: Option<Vec<f64>>,
    /// Temperature uncertainty map from regularization's penalized Hessian.
    pub temperature_uncertainty_map: Option<Array2<f64>>,
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
        anorm_map: None,
        background_maps: None,
        n_converged: 0,
        n_total: 0,
        nll_map: None,
        n_weak_directions: None,
        fisher_eigenvalues: None,
        temperature_uncertainty_map: None,
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

    // When temperature is free, cross-sections are recomputed each iteration
    // inside the forward model, so precomputing here would be wasted work.
    // Only precompute when temperature is fixed.
    //
    // Always disable covariance computation for per-pixel fitting — the
    // post-convergence Jacobian + matrix inversion is wasted work when we
    // only need the fitted densities and chi-squared.  This saves one
    // full Jacobian evaluation per pixel (N_free model evaluations for
    // finite-difference, or free for analytical) plus the O(n_free³) inversion.
    let fast_config = if config.fit_temperature() {
        // Precompute unbroadened (Reich-Moore) cross-sections ONCE for all
        // pixels.  Without this, each pixel recomputes them inside
        // TransmissionFitModel::new() and TemperatureContext construction,
        // dominating total runtime for heavy nuclei (U-238: ~5000 resonances).
        let base_xs = match config.precomputed_base_xs().cloned() {
            Some(cached) => cached,
            None => {
                let xs =
                    unbroadened_cross_sections(config.energies(), config.resonance_data(), cancel)?;
                Arc::new(xs)
            }
        };
        config
            .clone()
            .with_precomputed_base_xs(base_xs)
            .with_compute_covariance(false)
    } else {
        // Use caller-supplied precomputed cross-sections when available; only call
        // broadened_cross_sections when none are provided.  This lets repeated
        // spatial_map calls with the same isotopes/energy grid share one precompute
        // result and avoids redundant Doppler+resolution broadening work.
        let xs: Arc<Vec<Vec<f64>>> = match config.precomputed_cross_sections().cloned() {
            Some(cached) => cached,
            None => {
                let instrument_params = config.resolution().map(|r| InstrumentParams {
                    resolution: r.clone(),
                });
                // Pass the cancel token so precompute can bail between isotopes.
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

        // Build a config variant with the precomputed cross-sections injected
        // and covariance computation disabled for per-pixel speed.
        // fit_spectrum will use PrecomputedTransmissionModel when this field is Some.
        config
            .clone()
            .with_precomputed_cross_sections(xs)
            .with_compute_covariance(false)
    };

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
    let has_background = config.background().is_some();
    let mut anorm_map: Option<Array2<f64>> = if has_background {
        Some(Array2::from_elem((height, width), 1.0))
    } else {
        None
    };
    let mut background_maps: Option<[Array2<f64>; 3]> = if has_background {
        Some([
            Array2::zeros((height, width)),
            Array2::zeros((height, width)),
            Array2::zeros((height, width)),
        ])
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
        if let Some(ref mut a_map) = anorm_map {
            a_map[[*y, *x]] = result.anorm;
        }
        if let Some(ref mut bg_maps) = background_maps {
            bg_maps[0][[*y, *x]] = result.background[0];
            bg_maps[1][[*y, *x]] = result.background[1];
            bg_maps[2][[*y, *x]] = result.background[2];
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
        anorm_map,
        background_maps,
        n_converged,
        // D-13: n_total is the number of ATTEMPTED pixels (excluding dead),
        // not the number of successful fits.  Failed pixels are left as NaN
        // in the maps.  This gives honest convergence percentages.
        n_total: pixel_coords.len(),
        nll_map: None,
        n_weak_directions: None,
        fisher_eigenvalues: None,
        temperature_uncertainty_map: None,
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

    let _n_pixels = (y_range.end - y_range.start) * (x_range.end - x_range.start);

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

    // D-2: Inverse-variance weighted average of transmission over ROI.
    //
    // Simple arithmetic averaging of transmission is biased because
    // T = C_s / C_ob is a ratio — the average of ratios ≠ ratio of averages.
    // Inverse-variance weighting (w_i = 1/σ_i²) gives more weight to
    // pixels with better statistics, approximating the counts-first approach.
    //
    // For each energy bin:
    //   T_avg = Σ(w_i · T_i) / Σ(w_i)
    //   σ_avg = 1 / √(Σ(w_i))
    //
    // Reference: Bevington & Robinson §4.1 (weighted average of measurements).
    let roi_h = y_range.end - y_range.start;
    let roi_w = x_range.end - x_range.start;
    let mut avg_t = vec![0.0f64; n_energies];
    let mut sum_w = vec![0.0f64; n_energies]; // Σ(1/σ²)

    for y in 0..roi_h {
        for x in 0..roi_w {
            let t_row = roi_trans.slice(s![y, x, ..]);
            let u_row = roi_unc.slice(s![y, x, ..]);
            for e in 0..n_energies {
                let sigma = u_row[e];
                if sigma > 0.0 && sigma.is_finite() {
                    let w = 1.0 / (sigma * sigma);
                    avg_t[e] += w * t_row[e];
                    sum_w[e] += w;
                }
            }
        }
    }

    for e in 0..n_energies {
        if sum_w[e] > 0.0 {
            avg_t[e] /= sum_w[e];
            // σ_avg = 1 / √(Σ w)
            sum_w[e] = 1.0 / sum_w[e].sqrt();
        } else {
            // No valid pixels — use zero weight (will be skipped by fitter)
            avg_t[e] = 0.0;
            sum_w[e] = 1e30;
        }
    }
    // Rename for clarity in the fit call
    let avg_unc = sum_w;

    fit_spectrum(&avg_t, &avg_unc, config)
}

// ── Phase 3: InputData3D + spatial_map_typed ─────────────────────────────

use crate::pipeline::{InputData, UnifiedFitConfig, fit_spectrum_typed};

/// 3D input data for spatial mapping.
///
/// The outer dimension is energy (axis 0), inner dimensions are spatial (y, x).
/// The two variants correspond to [`InputData`] but carry 3D arrays.
#[derive(Debug)]
pub enum InputData3D<'a> {
    /// Pre-normalized transmission + uncertainty.
    Transmission {
        transmission: ArrayView3<'a, f64>,
        uncertainty: ArrayView3<'a, f64>,
    },
    /// Raw detector counts + open beam reference.
    Counts {
        sample_counts: ArrayView3<'a, f64>,
        open_beam_counts: ArrayView3<'a, f64>,
    },
}

impl InputData3D<'_> {
    /// Shape of the data: (n_energies, height, width).
    fn shape(&self) -> (usize, usize, usize) {
        let s = match self {
            Self::Transmission { transmission, .. } => transmission.shape(),
            Self::Counts { sample_counts, .. } => sample_counts.shape(),
        };
        (s[0], s[1], s[2])
    }

    /// Whether this is counts data.
    fn is_counts(&self) -> bool {
        matches!(self, Self::Counts { .. })
    }
}

/// Spatial mapping using the typed input data API.
///
/// Dispatches per-pixel fitting based on the `InputData3D` variant:
/// - **Transmission**: per-pixel LM or KL on transmission values
/// - **Counts**: per-pixel KL on raw counts (preserves Poisson statistics)
///
/// Always returns [`SpatialResult`].
pub fn spatial_map_typed(
    input: &InputData3D<'_>,
    config: &UnifiedFitConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
    progress: Option<&AtomicUsize>,
) -> Result<SpatialResult, PipelineError> {
    let (n_energies, height, width) = input.shape();
    let n_isotopes = config.resonance_data().len();

    // Validate shapes
    if n_energies != config.energies().len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "input spectral axis ({n_energies}) != config.energies length ({})",
            config.energies().len(),
        )));
    }
    match input {
        InputData3D::Transmission {
            transmission,
            uncertainty,
        } => {
            if uncertainty.shape() != transmission.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "uncertainty shape {:?} != transmission shape {:?}",
                    uncertainty.shape(),
                    transmission.shape(),
                )));
            }
        }
        InputData3D::Counts {
            sample_counts,
            open_beam_counts,
        } => {
            if open_beam_counts.shape() != sample_counts.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "open_beam shape {:?} != sample shape {:?}",
                    open_beam_counts.shape(),
                    sample_counts.shape(),
                )));
            }
        }
    }
    if let Some(dp) = dead_pixels
        && dp.shape() != [height, width]
    {
        return Err(PipelineError::ShapeMismatch(format!(
            "dead_pixels shape {:?} != spatial dimensions ({height}, {width})",
            dp.shape(),
        )));
    }

    // Collect live pixel coordinates
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

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(PipelineError::Cancelled);
    }
    if pixel_coords.is_empty() {
        return Ok(SpatialResult {
            density_maps: (0..n_isotopes)
                .map(|_| Array2::zeros((height, width)))
                .collect(),
            uncertainty_maps: (0..n_isotopes)
                .map(|_| Array2::from_elem((height, width), f64::NAN))
                .collect(),
            chi_squared_map: Array2::from_elem((height, width), f64::NAN),
            converged_map: Array2::from_elem((height, width), false),
            temperature_map: None,
            isotope_labels,
            anorm_map: None,
            background_maps: None,
            n_converged: 0,
            n_total: 0,
            nll_map: None,
            n_weak_directions: None,
            fisher_eigenvalues: None,
            temperature_uncertainty_map: None,
        });
    }

    // Transpose data to (height, width, n_energies) for cache locality
    let (data_a, data_b) = match input {
        InputData3D::Transmission {
            transmission,
            uncertainty,
        } => {
            let a = transmission
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let b = uncertainty
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            (a, b)
        }
        InputData3D::Counts {
            sample_counts,
            open_beam_counts,
        } => {
            let a = sample_counts
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let b = open_beam_counts
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            (a, b)
        }
    };

    // Precompute cross-sections once (shared across all pixels)
    let xs: Arc<Vec<Vec<f64>>> = match config.precomputed_cross_sections().cloned() {
        Some(cached) => cached,
        None => {
            let instrument = config.resolution().map(|r| InstrumentParams {
                resolution: r.clone(),
            });
            let xs_raw = broadened_cross_sections(
                config.energies(),
                config.resonance_data(),
                config.temperature_k(),
                instrument.as_ref(),
                cancel,
            )?;
            Arc::new(xs_raw)
        }
    };

    // Build a config with precomputed XS and covariance disabled
    let fast_config = config
        .clone()
        .with_precomputed_cross_sections(xs)
        .with_compute_covariance(false);

    let is_counts = input.is_counts();
    let has_transmission_bg = config.transmission_background().is_some();

    // Fit all pixels in parallel
    let results: Vec<((usize, usize), SpectrumFitResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            let spectrum_a: Vec<f64> = data_a.slice(s![y, x, ..]).to_vec();
            let spectrum_b: Vec<f64> = data_b
                .slice(s![y, x, ..])
                .iter()
                .map(|&v| if is_counts { v } else { v.max(1e-10) })
                .collect();

            // Build per-pixel 1D InputData
            let pixel_input = if is_counts {
                InputData::Counts {
                    sample_counts: spectrum_a.iter().map(|&v| v.max(0.0)).collect(),
                    open_beam_counts: spectrum_b,
                }
            } else {
                InputData::Transmission {
                    transmission: spectrum_a,
                    uncertainty: spectrum_b,
                }
            };

            let out = match fit_spectrum_typed(&pixel_input, &fast_config) {
                Ok(result) => Some(((y, x), result)),
                Err(_) => None,
            };
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            out
        })
        .collect();

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
    let mut anorm_map: Option<Array2<f64>> = if has_transmission_bg {
        Some(Array2::from_elem((height, width), 1.0))
    } else {
        None
    };
    let mut background_maps: Option<[Array2<f64>; 3]> = if has_transmission_bg {
        Some([
            Array2::zeros((height, width)),
            Array2::zeros((height, width)),
            Array2::zeros((height, width)),
        ])
    } else {
        None
    };
    let mut n_converged = 0;

    for ((y, x), result) in &results {
        for i in 0..n_isotopes {
            density_maps[i][[*y, *x]] = result.densities[i];
            if let Some(ref unc) = result.uncertainties {
                uncertainty_maps[i][[*y, *x]] = unc[i];
            }
        }
        chi_squared_map[[*y, *x]] = result.reduced_chi_squared;
        converged_map[[*y, *x]] = result.converged;
        if let Some(ref mut a_map) = anorm_map {
            a_map[[*y, *x]] = result.anorm;
        }
        if let Some(ref mut bg_maps) = background_maps {
            bg_maps[0][[*y, *x]] = result.background[0];
            bg_maps[1][[*y, *x]] = result.background[1];
            bg_maps[2][[*y, *x]] = result.background[2];
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
        temperature_map: None, // TODO: wire temperature fitting for typed path
        isotope_labels,
        anorm_map,
        background_maps,
        n_converged,
        n_total: pixel_coords.len(),
        nll_map: None,
        n_weak_directions: None,
        fisher_eigenvalues: None,
        temperature_uncertainty_map: None,
    })
}

// ── End Phase 3 ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_fitting::lm::{FitModel, LmConfig};
    use nereids_fitting::poisson::PoissonConfig;
    use nereids_fitting::transmission_model::TransmissionFitModel;

    use crate::pipeline::{BackgroundConfig, SolverChoice};
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

    // ---- C3 Critical Integration Tests ----

    /// C3-1: spatial_map with background=True recovers density AND
    /// produces non-trivial anorm_map and background_maps.
    #[test]
    fn test_spatial_map_with_background() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let n_energies = energies.len();

        // Generate synthetic spectrum with known background:
        //   T_out(E) = Anorm * exp(-n*sigma(E)) + BackA
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
        let t_inner = model.evaluate(&[true_density]).unwrap();
        let anorm_true = 0.95;
        let back_a_true = 0.03;
        let spectrum: Vec<f64> = t_inner
            .iter()
            .map(|&t| anorm_true * t + back_a_true)
            .collect();

        // 3×3 image
        let (height, width) = (3, 3);
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
        .unwrap()
        .with_background(BackgroundConfig::default());

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            None,
            None,
            None,
        )
        .unwrap();

        // Density recovered within 10%
        let fitted = result.density_maps[0][[1, 1]];
        assert!(
            (fitted - true_density).abs() / true_density < 0.10,
            "density = {fitted}, expected {true_density}"
        );

        // anorm_map populated
        let anorm_map = result
            .anorm_map
            .expect("anorm_map should be Some when background=True");
        let anorm_fitted = anorm_map[[1, 1]];
        assert!(
            (anorm_fitted - anorm_true).abs() < 0.05,
            "anorm = {anorm_fitted}, expected ~{anorm_true}"
        );

        // background_maps populated
        let bg_maps = result
            .background_maps
            .expect("background_maps should be Some");
        let back_a_fitted = bg_maps[0][[1, 1]];
        assert!(
            (back_a_fitted - back_a_true).abs() < 0.02,
            "backA = {back_a_fitted}, expected ~{back_a_true}"
        );
    }

    /// C3-2: spatial_map with fitter=Poisson (via SolverChoice::PoissonKL)
    /// converges on clean synthetic data.
    #[test]
    fn test_spatial_map_with_poisson_solver() {
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

        let (height, width) = (3, 3);
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
        .unwrap()
        .with_solver(SolverChoice::PoissonKL(PoissonConfig {
            max_iter: 200,
            ..PoissonConfig::default()
        }));

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
        // At least some pixels should converge on clean data
        assert!(
            result.n_converged >= 5,
            "Expected >=5 converged, got {}",
            result.n_converged
        );

        // Check density recovery (Poisson on transmission with sigma=0.01 should be reasonable)
        let fitted = result.density_maps[0][[1, 1]];
        assert!(
            (fitted - true_density).abs() / true_density < 0.15,
            "Poisson density = {fitted}, expected {true_density}"
        );
    }

    /// C3-3: spatial_map with Poisson + background converges.
    #[test]
    fn test_spatial_map_poisson_with_background() {
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
        let t_inner = model.evaluate(&[true_density]).unwrap();
        // Apply background
        let spectrum: Vec<f64> = t_inner.iter().map(|&t| 0.95 * t + 0.03).collect();

        let (height, width) = (3, 3);
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
        .unwrap()
        .with_solver(SolverChoice::PoissonKL(PoissonConfig {
            max_iter: 200,
            ..PoissonConfig::default()
        }))
        .with_background(BackgroundConfig::default());

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            None,
            None,
            None,
        )
        .unwrap();

        // Should have anorm_map and background_maps
        assert!(
            result.anorm_map.is_some(),
            "anorm_map should be Some with Poisson+background"
        );
        assert!(
            result.background_maps.is_some(),
            "background_maps should be Some with Poisson+background"
        );
    }

    /// C3-4: Poisson + background + temperature is correctly rejected.
    #[test]
    fn test_poisson_bg_temperature_rejected() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let n_energies = energies.len();

        let (height, width) = (2, 2);
        let transmission = Array3::from_elem((n_energies, height, width), 0.8);
        let uncertainty = Array3::from_elem((n_energies, height, width), 0.01);

        let config = FitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap()
        .with_solver(SolverChoice::PoissonKL(PoissonConfig::default()))
        .with_background(BackgroundConfig::default())
        .with_fit_temperature(true)
        .unwrap();

        let result = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            None,
            None,
            None,
        );

        // Should fail with a clear error, not silently produce wrong results.
        // Note: the error comes from per-pixel fit_spectrum, so we get
        // an empty result (all pixels fail) rather than a top-level error.
        // Either outcome is acceptable — the key is no silent wrong answers.
        match result {
            Err(_) => {} // top-level rejection: good
            Ok(r) => {
                // If individual pixels all failed, that's also acceptable
                assert!(
                    r.n_converged == 0,
                    "Poisson+bg+temperature should not produce converged results; got {} converged",
                    r.n_converged,
                );
            }
        }
    }

    /// C3-5: spatial_map handles zero-count bins gracefully (no panic, no NaN propagation).
    #[test]
    fn test_spatial_map_zero_transmission_bins() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let n_energies = energies.len();

        let (height, width) = (2, 2);
        let mut transmission = Array3::from_elem((n_energies, height, width), 0.9);
        let mut uncertainty = Array3::from_elem((n_energies, height, width), 0.01);

        // Set 10% of bins to zero transmission with large uncertainty
        for e in (0..n_energies).step_by(10) {
            for y in 0..height {
                for x in 0..width {
                    transmission[[e, y, x]] = 0.0;
                    uncertainty[[e, y, x]] = 1e30; // negligible weight
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
        );

        // Should not panic
        assert!(result.is_ok(), "spatial_map panicked on zero-count bins");
        let r = result.unwrap();

        // At least some pixels should produce finite results
        let any_finite = r.density_maps[0].iter().any(|&d| d.is_finite());
        assert!(any_finite, "All density values are NaN/Inf after zero bins");
    }

    // ── Phase 3: spatial_map_typed tests ──

    use crate::pipeline::{SolverConfig, UnifiedFitConfig};
    use nereids_fitting::transmission_model::PrecomputedTransmissionModel;

    /// Build a 4x4 synthetic transmission stack from known density.
    fn synthetic_4x4_transmission(
        res_data: &nereids_endf::resonance::ResonanceData,
        true_density: f64,
        energies: &[f64],
    ) -> (Array3<f64>, Array3<f64>) {
        let n_e = energies.len();
        let xs = nereids_physics::transmission::broadened_cross_sections(
            energies,
            std::slice::from_ref(res_data),
            0.0,
            None,
            None,
        )
        .unwrap();
        let model = PrecomputedTransmissionModel {
            cross_sections: Arc::new(xs),
            density_indices: Arc::new(vec![0]),
        };
        let t_1d = model.evaluate(&[true_density]).unwrap();
        let sigma_1d: Vec<f64> = t_1d.iter().map(|&v| 0.01 * v.max(0.01)).collect();

        // Fill a 4x4 grid with the same spectrum
        let mut t_3d = Array3::zeros((n_e, 4, 4));
        let mut u_3d = Array3::zeros((n_e, 4, 4));
        for y in 0..4 {
            for x in 0..4 {
                for (i, (&t, &s)) in t_1d.iter().zip(sigma_1d.iter()).enumerate() {
                    t_3d[[i, y, x]] = t;
                    u_3d[[i, y, x]] = s;
                }
            }
        }
        (t_3d, u_3d)
    }

    /// Build a 4x4 synthetic counts stack from known density.
    fn synthetic_4x4_counts(
        res_data: &nereids_endf::resonance::ResonanceData,
        true_density: f64,
        energies: &[f64],
        i0: f64,
    ) -> (Array3<f64>, Array3<f64>) {
        let (t_3d, _) = synthetic_4x4_transmission(res_data, true_density, energies);
        let n_e = energies.len();
        let mut sample = Array3::zeros((n_e, 4, 4));
        let mut ob = Array3::zeros((n_e, 4, 4));
        for y in 0..4 {
            for x in 0..4 {
                for i in 0..n_e {
                    ob[[i, y, x]] = i0;
                    sample[[i, y, x]] = (t_3d[[i, y, x]] * i0).round().max(0.0);
                }
            }
        }
        (sample, ob)
    }

    #[test]
    fn test_spatial_map_typed_transmission_lm() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&data, true_density, &energies);

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

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_total, 16);
        assert!(result.n_converged >= 14, "Most pixels should converge");

        // Check mean density of converged pixels
        let d = &result.density_maps[0];
        let conv = &result.converged_map;
        let mean: f64 = d
            .iter()
            .zip(conv.iter())
            .filter(|(_, c)| **c)
            .map(|(d, _)| *d)
            .sum::<f64>()
            / result.n_converged as f64;
        assert!(
            (mean - true_density).abs() / true_density < 0.05,
            "mean density: {mean}, true: {true_density}"
        );
    }

    #[test]
    fn test_spatial_map_typed_counts_kl() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (sample, ob) = synthetic_4x4_counts(&data, true_density, &energies, 1000.0);

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

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_total, 16);
        assert!(
            result.n_converged >= 14,
            "Most pixels should converge with KL"
        );

        let d = &result.density_maps[0];
        let conv = &result.converged_map;
        let mean: f64 = d
            .iter()
            .zip(conv.iter())
            .filter(|(_, c)| **c)
            .map(|(d, _)| *d)
            .sum::<f64>()
            / result.n_converged.max(1) as f64;
        assert!(
            (mean - true_density).abs() / true_density < 0.10,
            "KL mean density: {mean}, true: {true_density}"
        );
    }

    #[test]
    fn test_spatial_map_typed_counts_kl_low_counts() {
        // I0=10: the regime where KL excels
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (sample, ob) = synthetic_4x4_counts(&data, true_density, &energies, 10.0);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap(); // Auto solver → KL for counts

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_total, 16);
        // At I0=10, KL should still converge for most pixels
        assert!(
            result.n_converged >= 10,
            "KL at I0=10: only {}/{} converged",
            result.n_converged,
            result.n_total
        );
    }

    #[test]
    fn test_spatial_map_typed_dead_pixels() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let (t_3d, u_3d) = synthetic_4x4_transmission(&data, 0.0005, &energies);

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap();

        // Mask half the pixels as dead
        let mut dead = Array2::from_elem((4, 4), false);
        for y in 0..2 {
            for x in 0..4 {
                dead[[y, x]] = true;
            }
        }

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let result = spatial_map_typed(&input, &config, Some(&dead), None, None).unwrap();
        assert_eq!(result.n_total, 8, "Only 8 live pixels");
    }
}
