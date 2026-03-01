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
//!   Y_s(E) = alpha_1 * [Phi(E) * exp(-sum_i rho_i * sigma_i(E)) + alpha_2 * B(E)]
//!
//! where:
//! - Phi(E) = incident flux spectrum (estimated in Stage 1)
//! - rho_i = areal density of isotope i (fit parameter)
//! - sigma_i(E) = total cross-section of isotope i
//! - B(E) = background spectrum (estimated in Stage 1)
//! - alpha_1, alpha_2 = normalization scalers (estimated in Stage 1)
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` — Two-stage reconstruction with APGM

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array2, Array3, s};
use rayon::prelude::*;

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{self, CountsModel, PoissonConfig};
use nereids_fitting::transmission_model::PrecomputedTransmissionModel;
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::{InstrumentParams, broadened_cross_sections};

use crate::error::PipelineError;

/// Errors from `SparseConfig` construction.
#[derive(Debug, PartialEq)]
pub enum SparseConfigError {
    /// Energy grid must be non-empty.
    EmptyEnergies,
    /// Resonance data must be non-empty.
    EmptyResonanceData,
    /// initial_densities length must match resonance_data length.
    DensityCountMismatch { densities: usize, isotopes: usize },
    /// isotope_names length must match resonance_data length.
    NameCountMismatch { names: usize, isotopes: usize },
    /// Temperature must be finite.
    NonFiniteTemperature(f64),
    /// Temperature must be non-negative.
    NegativeTemperature(f64),
}

impl std::fmt::Display for SparseConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyEnergies => write!(f, "energy grid must be non-empty"),
            Self::EmptyResonanceData => write!(f, "resonance_data must be non-empty"),
            Self::DensityCountMismatch {
                densities,
                isotopes,
            } => write!(
                f,
                "initial_densities length ({densities}) must match resonance_data length ({isotopes})"
            ),
            Self::NameCountMismatch { names, isotopes } => write!(
                f,
                "isotope_names length ({names}) must match resonance_data length ({isotopes})"
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

impl std::error::Error for SparseConfigError {}

/// Configuration for two-stage sparse reconstruction.
///
/// Fields are private to enforce validation invariants.
/// Use [`SparseConfig::new`] to construct.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Energy grid in eV (ascending).
    energies: Vec<f64>,
    /// Resonance data for each isotope.
    resonance_data: Vec<ResonanceData>,
    /// Isotope names (for reporting).
    isotope_names: Vec<String>,
    /// Sample temperature in Kelvin.
    temperature_k: f64,
    /// Optional resolution function (Gaussian or tabulated).
    resolution: Option<ResolutionFunction>,
    /// Initial guess for areal densities.
    initial_densities: Vec<f64>,
    /// Poisson optimizer configuration.
    poisson_config: PoissonConfig,
}

impl SparseConfig {
    /// Create a validated sparse reconstruction configuration.
    ///
    /// # Errors
    /// Returns `SparseConfigError` if any invariant is violated.
    pub fn new(
        energies: Vec<f64>,
        resonance_data: Vec<ResonanceData>,
        isotope_names: Vec<String>,
        temperature_k: f64,
        resolution: Option<ResolutionFunction>,
        initial_densities: Vec<f64>,
        poisson_config: PoissonConfig,
    ) -> Result<Self, SparseConfigError> {
        if energies.is_empty() {
            return Err(SparseConfigError::EmptyEnergies);
        }
        if resonance_data.is_empty() {
            return Err(SparseConfigError::EmptyResonanceData);
        }
        if initial_densities.len() != resonance_data.len() {
            return Err(SparseConfigError::DensityCountMismatch {
                densities: initial_densities.len(),
                isotopes: resonance_data.len(),
            });
        }
        if isotope_names.len() != resonance_data.len() {
            return Err(SparseConfigError::NameCountMismatch {
                names: isotope_names.len(),
                isotopes: resonance_data.len(),
            });
        }
        if !temperature_k.is_finite() {
            return Err(SparseConfigError::NonFiniteTemperature(temperature_k));
        }
        if temperature_k < 0.0 {
            return Err(SparseConfigError::NegativeTemperature(temperature_k));
        }
        Ok(Self {
            energies,
            resonance_data,
            isotope_names,
            temperature_k,
            resolution,
            initial_densities,
            poisson_config,
        })
    }

    /// Returns the energy grid in eV.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Returns the resonance data for each isotope.
    #[must_use]
    pub fn resonance_data(&self) -> &[ResonanceData] {
        &self.resonance_data
    }

    /// Returns the isotope names.
    #[must_use]
    pub fn isotope_names(&self) -> &[String] {
        &self.isotope_names
    }

    /// Returns the sample temperature in Kelvin.
    #[must_use]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Returns the optional resolution function.
    #[must_use]
    pub fn resolution(&self) -> Option<&ResolutionFunction> {
        self.resolution.as_ref()
    }

    /// Returns the initial density guesses.
    #[must_use]
    pub fn initial_densities(&self) -> &[f64] {
        &self.initial_densities
    }

    /// Returns the Poisson optimizer configuration.
    #[must_use]
    pub fn poisson_config(&self) -> &PoissonConfig {
        &self.poisson_config
    }
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
/// * `open_beam_counts` — Raw open-beam counts (n_energies, height, width).
/// * `roi` — Optional (y_range, x_range) for high-statistics region.
///   If None, uses the entire image.
/// * `dead_pixels` — Optional dead-pixel mask.  Dead pixels are excluded
///   from the ROI average so that suppressed open-beam counts do not bias
///   the flux estimate.
///
/// # Errors
/// Returns `PipelineError::InvalidParameter` if:
/// - The ROI y-range or x-range is empty.
/// - The ROI exceeds the spatial bounds of `open_beam_counts`.
/// - Every pixel in the ROI is dead, leaving zero live pixels to average over.
///
/// Returns `PipelineError::ShapeMismatch` if the `dead_pixels` mask shape
/// does not match the spatial dimensions of `open_beam_counts`.
pub fn estimate_nuisance(
    open_beam_counts: &Array3<f64>,
    roi: Option<(std::ops::Range<usize>, std::ops::Range<usize>)>,
    dead_pixels: Option<&Array2<bool>>,
) -> Result<NuisanceParams, PipelineError> {
    let n_energies = open_beam_counts.shape()[0];
    let height = open_beam_counts.shape()[1];
    let width = open_beam_counts.shape()[2];

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

    // Average open-beam over ROI to get flux estimate
    let (y_range, x_range) = match roi {
        Some((y_r, x_r)) => {
            if y_r.start >= y_r.end || x_r.start >= x_r.end {
                return Err(PipelineError::InvalidParameter(
                    "ROI y-range or x-range is empty".into(),
                ));
            }
            if y_r.start >= height || y_r.end > height || x_r.start >= width || x_r.end > width {
                return Err(PipelineError::InvalidParameter(
                    "ROI is out of bounds of the open-beam data".into(),
                ));
            }
            (y_r, x_r)
        }
        None => (0..height, 0..width),
    };

    // Slice the ROI first, THEN transpose, so the work is O(n_energies * roi_h * roi_w)
    // instead of O(n_energies * height * width).  For small ROIs on large images this
    // avoids transposing the entire open-beam array.
    let ob_roi: Array3<f64> = open_beam_counts
        .slice(s![
            ..,
            y_range.start..y_range.end,
            x_range.start..x_range.end
        ])
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();

    let roi_h = y_range.end - y_range.start;
    let roi_w = x_range.end - x_range.start;
    let mut flux = vec![0.0f64; n_energies];
    let mut n_pixels: usize = 0;

    for ly in 0..roi_h {
        for lx in 0..roi_w {
            // Map local ROI indices back to global coordinates for dead-pixel lookup
            if dead_pixels.is_some_and(|m| m[[y_range.start + ly, x_range.start + lx]]) {
                continue;
            }
            let row = ob_roi.slice(s![ly, lx, ..]);
            for e in 0..n_energies {
                flux[e] += row[e];
            }
            n_pixels += 1;
        }
    }

    if n_pixels == 0 {
        return Err(PipelineError::InvalidParameter(
            "no live pixels in ROI after dead-pixel filtering; cannot estimate nuisance flux"
                .into(),
        ));
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
    let background = vec![0.0f64; n_energies];

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
///
/// # Cancellation semantics
/// If cancellation occurs mid-flight, pixels that already completed are
/// included in the result.  The function returns `Err(Cancelled)` only if
/// **no** pixels completed before cancellation was detected.  When at least
/// one pixel finished, the partial result is returned as `Ok(SparseResult)`
/// with `n_total` reflecting only the completed pixels.
///
/// # Errors
/// Returns `PipelineError::ShapeMismatch` if array dimensions are inconsistent,
/// or `PipelineError::Cancelled` if the cancellation token is set before any
/// pixel completes.
pub fn sparse_reconstruct(
    sample_counts: &Array3<f64>,
    nuisance: &NuisanceParams,
    config: &SparseConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
) -> Result<SparseResult, PipelineError> {
    let shape = sample_counts.shape();
    let (n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data().len();

    // Config-level invariants (non-empty energies, resonance_data, density
    // count, temperature) are enforced by SparseConfig::new().  Only per-call
    // shape checks remain here.
    if n_energies != config.energies().len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "sample_counts spectral axis ({}) != config.energies length ({})",
            n_energies,
            config.energies().len(),
        )));
    }
    if nuisance.flux.len() != n_energies {
        return Err(PipelineError::ShapeMismatch(format!(
            "nuisance flux length ({}) must match n_energies ({})",
            nuisance.flux.len(),
            n_energies,
        )));
    }
    if nuisance.background.len() != n_energies {
        return Err(PipelineError::ShapeMismatch(format!(
            "nuisance background length ({}) must match n_energies ({})",
            nuisance.background.len(),
            n_energies,
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

    // Transpose sample_counts from (n_energies, height, width) to (height, width, n_energies)
    // so that each pixel's spectrum is contiguous in memory.  See spatial.rs for the
    // full cache-friendliness rationale.
    let counts_t: Array3<f64> = sample_counts
        .view()
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();

    // Precompute Doppler-broadened cross-sections once, outside the pixel loop.
    // The same XS apply to every pixel (same isotopes, same temperature, same energy grid).
    // Mirrors the pattern used in spatial.rs to avoid repeating expensive broadening work.
    let instrument_params = config.resolution().map(|r| InstrumentParams {
        resolution: r.clone(),
    });
    let xs_raw = broadened_cross_sections(
        config.energies(),
        config.resonance_data(),
        config.temperature_k(),
        instrument_params.as_ref(),
        cancel,
    )?;
    let xs: Arc<Vec<Vec<f64>>> = Arc::new(xs_raw);
    let density_idx: Arc<Vec<usize>> = Arc::new((0..n_isotopes).collect());

    // Pre-build parameter template outside the pixel loop so that per-pixel
    // iterations only need a cheap Clone (no format!() or name lookup).
    //
    // Why clone instead of reconstruct?  ParameterSet::new + FitParameter
    // constructors convert the name via `Into<Cow<'static, str>>` and
    // re-evaluate the isotope name lookup per pixel.  Cloning an existing
    // ParameterSet copies the Vec of FitParameter (which clones each
    // Cow — free for Borrowed, heap-copy for Owned), avoiding the
    // per-pixel format!/Into conversion overhead.  The template is the
    // single source of truth for parameter layout, bounds, and initial values.
    //
    // isotope_names.len() == n_isotopes is validated above, so direct indexing
    // is safe — the previous .get(i).unwrap_or_else(|| format!(...)) fallback
    // was unreachable dead code.
    let param_template = ParameterSet::new(
        config
            .initial_densities()
            .iter()
            .enumerate()
            .map(|(i, &d)| FitParameter::non_negative(config.isotope_names()[i].clone(), d))
            .collect(),
    );

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

    // Check cancellation before pixel loop
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(PipelineError::Cancelled);
    }

    // Fit all pixels in parallel, skipping new work when cancelled
    let results: Vec<((usize, usize), PixelResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            // Check cancellation before starting each pixel
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            // Extract counts for this pixel from the transposed (h, w, n_energies) layout.
            // The energy axis is now contiguous in memory, fitting in L1 cache.
            let y_obs: Vec<f64> = counts_t
                .slice(s![y, x, ..])
                .iter()
                .map(|&v| v.max(0.0))
                .collect();

            // Build per-pixel transmission model reusing precomputed XS.
            // Arc::clone shares the cross-section data (zero-copy) across all pixels,
            // avoiding expensive repeated Doppler/resolution broadening.
            let t_model = PrecomputedTransmissionModel {
                cross_sections: Arc::clone(&xs),
                density_indices: Arc::clone(&density_idx),
            };

            // Wrap in counts model: Y = flux * T(theta) + background.
            // flux/background are borrowed (zero-copy) — no per-pixel allocation.
            let counts_model = CountsModel {
                transmission_model: &t_model,
                flux: &nuisance.flux,
                background: &nuisance.background,
            };

            // Fit with Poisson likelihood using analytical gradient.
            // Passing the precomputed cross-sections and flux enables
            // poisson_fit_analytic to compute the full dNLL/dn_k gradient vector
            // in one model evaluation per iteration instead of N_isotopes+1
            // evaluations with finite differences (one base + one per parameter).
            let mut params = param_template.clone();

            let result = poisson::poisson_fit_analytic(
                &counts_model,
                &y_obs,
                &nuisance.flux,
                &xs,
                t_model.density_indices.as_slice(),
                &mut params,
                config.poisson_config(),
                None,
            )
            .ok()?;

            let pixel_result = PixelResult {
                densities: (0..n_isotopes).map(|i| result.params[i]).collect(),
                nll: result.nll,
                converged: result.converged,
            };

            Some(((y, x), pixel_result))
        })
        .collect();

    // If cancellation was triggered during pixel fitting and no results, return Cancelled.
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) && results.is_empty() {
        return Err(PipelineError::Cancelled);
    }

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

    Ok(SparseResult {
        density_maps,
        nll_map,
        converged_map,
        n_converged,
        n_total: results.len(),
        nuisance: nuisance.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_fitting::lm::FitModel;
    use nereids_fitting::transmission_model::TransmissionFitModel;

    use crate::test_helpers::u238_single_resonance;

    #[test]
    fn test_estimate_nuisance_basic() {
        let n_e = 5;
        let ob = Array3::from_elem((n_e, 2, 2), 100.0);

        let nuisance = estimate_nuisance(&ob, None, None).unwrap();
        assert_eq!(nuisance.flux.len(), n_e);
        assert!((nuisance.flux[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_nuisance_nonzero_roi_with_dead_pixel() {
        // 3 energies × 4 height × 4 width
        let n_e = 3;
        let h = 4;
        let w = 4;
        let mut ob = Array3::from_elem((n_e, h, w), 50.0);

        // Give ROI pixels distinct values so the average is verifiable.
        // ROI: y=1..3, x=1..3 → a 2×2 sub-grid of the 4×4 image.
        //   (1,1)=100  (1,2)=200
        //   (2,1)=300  (2,2)=400   ← this pixel will be dead
        for e in 0..n_e {
            ob[[e, 1, 1]] = 100.0;
            ob[[e, 1, 2]] = 200.0;
            ob[[e, 2, 1]] = 300.0;
            ob[[e, 2, 2]] = 400.0;
        }

        // Dead-pixel mask: only (2,2) is dead (global coords, inside the ROI).
        let mut dead = Array2::from_elem((h, w), false);
        dead[[2, 2]] = true;

        let roi = Some((1..3usize, 1..3usize));
        let nuisance = estimate_nuisance(&ob, roi, Some(&dead)).unwrap();

        // Live pixels in ROI: (1,1)=100, (1,2)=200, (2,1)=300.
        // Expected average = (100 + 200 + 300) / 3 = 200.
        assert_eq!(nuisance.flux.len(), n_e);
        for &f in &nuisance.flux {
            assert!((f - 200.0).abs() < 1e-10, "expected flux ~200.0, got {f}",);
        }
        // Background is zero (hardcoded).
        for &b in &nuisance.background {
            assert!((b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_estimate_nuisance_rejects_dead_pixel_shape_mismatch() {
        let n_e = 5;
        let ob = Array3::from_elem((n_e, 2, 2), 100.0);
        let dead = Array2::from_elem((3, 2), false); // wrong shape

        let result = estimate_nuisance(&ob, None, Some(&dead));
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_reconstruct_synthetic() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
        let n_energies = energies.len();

        // Build transmission model to generate synthetic data
        let t_model = TransmissionFitModel::new(
            energies.clone(),
            vec![data.clone()],
            0.0,
            None,
            vec![0],
            None,
        )
        .unwrap();
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
        let nuisance = estimate_nuisance(&ob_counts, None, None).unwrap();

        // Stage 2: reconstruct
        let config = SparseConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            PoissonConfig::default(),
        )
        .unwrap();

        let result = sparse_reconstruct(&sample_counts, &nuisance, &config, None, None).unwrap();

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

    #[test]
    fn test_sparse_config_rejects_density_len_mismatch() {
        let data = u238_single_resonance();
        let err = SparseConfig::new(
            vec![1.0, 2.0, 3.0],
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![], // wrong: should be 1 element
            PoissonConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SparseConfigError::DensityCountMismatch { .. }
        ));
    }

    #[test]
    fn test_sparse_reconstruct_rejects_dead_pixel_shape_mismatch() {
        let data = u238_single_resonance();
        let config = SparseConfig::new(
            vec![1.0, 2.0, 3.0],
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
            PoissonConfig::default(),
        )
        .unwrap();

        let sample = Array3::from_elem((3, 2, 2), 50.0);
        let nuisance = NuisanceParams {
            flux: vec![100.0; 3],
            background: vec![0.0; 3],
        };
        let dead = Array2::from_elem((3, 2), false); // wrong shape

        let result = sparse_reconstruct(&sample, &nuisance, &config, Some(&dead), None);
        assert!(result.is_err());
    }

    // --- SparseConfig validation tests ---

    #[test]
    fn test_sparse_config_valid() {
        let data = u238_single_resonance();
        let config = SparseConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
            PoissonConfig::default(),
        );
        assert!(config.is_ok());
    }

    #[test]
    fn test_sparse_config_rejects_empty_energies() {
        let data = u238_single_resonance();
        let err = SparseConfig::new(
            vec![],
            vec![data],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.001],
            PoissonConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, SparseConfigError::EmptyEnergies);
    }

    #[test]
    fn test_sparse_config_rejects_empty_resonance_data() {
        let err = SparseConfig::new(
            vec![1.0, 2.0],
            vec![],
            vec![],
            300.0,
            None,
            vec![],
            PoissonConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, SparseConfigError::EmptyResonanceData);
    }

    #[test]
    fn test_sparse_config_rejects_name_count_mismatch() {
        let data = u238_single_resonance();
        let err = SparseConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into(), "extra".into()], // 2 names but only 1 isotope
            300.0,
            None,
            vec![0.001],
            PoissonConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, SparseConfigError::NameCountMismatch { .. }));
    }

    #[test]
    fn test_sparse_config_rejects_nan_temperature() {
        let data = u238_single_resonance();
        let err = SparseConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            f64::NAN,
            None,
            vec![0.001],
            PoissonConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, SparseConfigError::NonFiniteTemperature(_)));
    }

    #[test]
    fn test_sparse_config_rejects_negative_temperature() {
        let data = u238_single_resonance();
        let err = SparseConfig::new(
            vec![1.0, 2.0],
            vec![data],
            vec!["U-238".into()],
            -1.0,
            None,
            vec![0.001],
            PoissonConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, SparseConfigError::NegativeTemperature(-1.0));
    }
}
