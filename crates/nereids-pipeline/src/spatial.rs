//! Spatial mapping: per-pixel fitting with rayon parallelization.
//!
//! Applies the single-spectrum fitting pipeline across all pixels in
//! a hyperspectral neutron imaging dataset to produce 2D composition maps.

use ndarray::{Array2, ArrayView3, s};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nereids_physics::transmission::{
    InstrumentParams, broadened_cross_sections, unbroadened_cross_sections,
};

use crate::error::PipelineError;
use crate::pipeline::SpectrumFitResult;

/// Result of spatial mapping over a 2D image.
#[derive(Debug)]
pub struct SpatialResult {
    /// Fitted areal density maps, one per isotope.
    /// Each Array2 has shape (height, width).
    pub density_maps: Vec<Array2<f64>>,
    /// Uncertainty maps, one per isotope.
    pub uncertainty_maps: Vec<Array2<f64>>,
    /// Reduced chi-squared map.  For the counts-KL dispatch (joint-Poisson
    /// deviance per memo 35 §P1.2) this is back-compat-mirrored to
    /// `D/(n−k)`; the semantically-correct per-pixel value is also
    /// exposed as [`Self::deviance_per_dof_map`].
    pub chi_squared_map: Array2<f64>,
    /// Per-pixel conditional binomial deviance `D/(n−k)` map.  `Some` when
    /// the effective per-pixel solver is the counts-KL dispatch
    /// (joint-Poisson); `None` for LM-only runs and transmission+PoissonKL
    /// where Pearson χ²/dof is the GOF.
    pub deviance_per_dof_map: Option<Array2<f64>>,
    /// Convergence map (true = converged).
    pub converged_map: Array2<bool>,
    /// Fitted temperature map (K). `Some` when `config.fit_temperature()` is true.
    pub temperature_map: Option<Array2<f64>>,
    /// Per-pixel temperature uncertainty map (K, 1-sigma).
    /// `Some` when `config.fit_temperature()` is true.
    /// Entries are NaN where uncertainty was unavailable for that pixel.
    pub temperature_uncertainty_map: Option<Array2<f64>>,
    /// Isotope labels captured at compute time, one per density map.
    /// Ensures display labels stay in sync with density data even if the
    /// user modifies the isotope list after fitting.
    pub isotope_labels: Vec<String>,
    /// Per-pixel normalization / signal-scale map (when background fitting is enabled).
    pub anorm_map: Option<Array2<f64>>,
    /// Per-pixel background parameter maps.
    /// Transmission LM uses `[BackA, BackB, BackC]`.
    /// Counts KL background uses `[b0, b1, alpha_2]`.
    pub background_maps: Option<[Array2<f64>; 3]>,
    /// Number of pixels that converged.
    pub n_converged: usize,
    /// Total number of pixels fitted.
    pub n_total: usize,
    /// Number of pixels where the fitter returned an error (not just
    /// non-convergence — a hard failure like invalid parameters or NaN
    /// model output). These pixels have NaN density and false convergence.
    pub n_failed: usize,
}

// ── Phase 3: InputData3D + spatial_map_typed ─────────────────────────────

use crate::pipeline::{InputData, SolverConfig, UnifiedFitConfig, fit_spectrum_typed};

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
    /// Raw detector counts with explicit nuisance spectra.
    CountsWithNuisance {
        sample_counts: ArrayView3<'a, f64>,
        flux: ArrayView3<'a, f64>,
        background: ArrayView3<'a, f64>,
    },
}

impl InputData3D<'_> {
    /// Shape of the data: (n_energies, height, width).
    pub(crate) fn shape(&self) -> (usize, usize, usize) {
        let s = match self {
            Self::Transmission { transmission, .. } => transmission.shape(),
            Self::Counts { sample_counts, .. } => sample_counts.shape(),
            Self::CountsWithNuisance { sample_counts, .. } => sample_counts.shape(),
        };
        (s[0], s[1], s[2])
    }

    /// `true` when the input is a counts variant (Counts or CountsWithNuisance)
    /// — i.e. the per-pixel dispatch goes through the counts-KL path
    /// (joint-Poisson deviance) rather than transmission.
    pub fn is_counts(&self) -> bool {
        matches!(self, Self::Counts { .. } | Self::CountsWithNuisance { .. })
    }
}

/// Spatial mapping using the typed input data API.
///
/// Dispatches per-pixel fitting based on the `InputData3D` variant:
/// - **Transmission**: per-pixel LM (or KL, opt-in) on transmission values.
/// - **Counts**: per-pixel counts-KL dispatch (joint-Poisson conditional
///   binomial deviance per memo 35 §P1) on the sample cube, paired
///   against the **spatially-averaged open-beam flux**.  See the inline
///   comment on `averaged_flux` for the rationale: this is a deliberate
///   bias-variance trade that reduces per-pixel OB shot-noise at the
///   cost of the exact per-pixel paired joint-Poisson observation model.
///   Callers needing the exact paired form should supply per-pixel
///   nuisance spectra via [`InputData3D::CountsWithNuisance`] instead.
/// - **CountsWithNuisance**: per-pixel counts-KL dispatch with the
///   caller-supplied per-pixel flux and background cubes.  No averaging.
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
    // n_maps = number of density maps to return (one per group or per isotope).
    let n_maps = config.n_density_params();

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
        InputData3D::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        } => {
            if flux.shape() != sample_counts.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "flux shape {:?} != sample shape {:?}",
                    flux.shape(),
                    sample_counts.shape(),
                )));
            }
            if background.shape() != sample_counts.shape() {
                return Err(PipelineError::ShapeMismatch(format!(
                    "background shape {:?} != sample shape {:?}",
                    background.shape(),
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
    let has_background_outputs =
        config.transmission_background().is_some() || config.counts_background().is_some();

    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(PipelineError::Cancelled);
    }
    if pixel_coords.is_empty() {
        return Ok(SpatialResult {
            density_maps: (0..n_maps)
                .map(|_| Array2::zeros((height, width)))
                .collect(),
            uncertainty_maps: (0..n_maps)
                .map(|_| Array2::from_elem((height, width), f64::NAN))
                .collect(),
            chi_squared_map: Array2::from_elem((height, width), f64::NAN),
            deviance_per_dof_map: if input.is_counts() {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            converged_map: Array2::from_elem((height, width), false),
            temperature_map: if config.fit_temperature() {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            temperature_uncertainty_map: if config.fit_temperature() {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            isotope_labels,
            anorm_map: if has_background_outputs {
                Some(Array2::from_elem((height, width), f64::NAN))
            } else {
                None
            },
            background_maps: if has_background_outputs {
                Some([
                    Array2::from_elem((height, width), f64::NAN),
                    Array2::from_elem((height, width), f64::NAN),
                    Array2::from_elem((height, width), f64::NAN),
                ])
            } else {
                None
            },
            n_converged: 0,
            n_total: 0,
            n_failed: 0,
        });
    }

    // Transpose data to (height, width, n_energies) for cache locality.
    let (data_a, data_b, data_c) = match input {
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
            (a, b, None)
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
            (a, b, None)
        }
        InputData3D::CountsWithNuisance {
            sample_counts,
            flux,
            background,
        } => {
            let a = sample_counts
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let b = flux
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            let c = background
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            (a, b, Some(c))
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

    // When groups are active and temperature is NOT being fitted, collapse
    // per-member broadened XS into per-group σ_eff once here.  This avoids
    // redundant O(n_members × n_energies) collapsing inside
    // build_transmission_model on every per-pixel call.
    let xs = if !config.fit_temperature()
        && let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios)
        && xs.len() == di.len()
        && di.len() == dr.len()
    {
        let n_e = xs[0].len();
        let mut eff = vec![vec![0.0f64; n_e]; n_maps];
        for ((&idx, &ratio), member_xs) in di.iter().zip(dr.iter()).zip(xs.iter()) {
            for (j, &sigma) in member_xs.iter().enumerate() {
                eff[idx][j] += ratio * sigma;
            }
        }
        Arc::new(eff)
    } else {
        xs
    };

    // Precompute unbroadened (base) cross-sections for temperature fitting.
    // This avoids 74× overhead from redundant Reich-Moore evaluation per
    // KL iteration (112ms Reich-Moore vs 1.5ms Doppler rebroadening).
    let fast_config = if config.fit_temperature() {
        let base_xs: Vec<Vec<f64>> =
            unbroadened_cross_sections(config.energies(), config.resonance_data(), cancel)
                .map_err(PipelineError::Transmission)?;
        config
            .clone()
            .with_precomputed_cross_sections(xs)
            .with_precomputed_base_xs(Arc::new(base_xs))
            .with_compute_covariance(true)
    } else {
        // For non-temperature path: xs is already collapsed to σ_eff when
        // groups are active, so clear group mapping to prevent double-collapse
        // inside build_transmission_model.
        let mut cfg = config.clone();
        if cfg.density_indices.is_some() {
            cfg.density_indices = None;
            cfg.density_ratios = None;
        }
        cfg.with_precomputed_cross_sections(xs)
            .with_compute_covariance(true)
    };

    // Auto-disable Nelder-Mead polish for multi-pixel counts-KL spatial
    // maps (memo 38 §6 recommendation).  Polish is a single-spectrum
    // research knob — on the VENUS Hf 120min aggregated fit it took
    // ~1 000 s; at 512 × 512 pixels that is untenable even with rayon.
    // Per-pixel fits also rarely hit the over-parameterized stall regime
    // polish targets.  The caller can force polish back on via
    // [`UnifiedFitConfig::with_counts_enable_polish(Some(true))`].
    let fast_config = if pixel_coords.len() > 1 && fast_config.counts_enable_polish().is_none() {
        fast_config.with_counts_enable_polish(Some(false))
    } else {
        fast_config
    };

    // ── Modeling choice: spatially-averaged open-beam flux ──
    //
    // For `InputData3D::Counts`, every pixel's sample spectrum is paired
    // with the **same** open-beam spectrum: the spatial average across
    // all live pixels (`pixel_coords`).  This is INTENTIONAL, not a
    // per-pixel paired observation.  The rationale:
    //
    // 1. The open-beam counts `O(E)` are a *reference flux* that is
    //    approximately spatially uniform (the sample casts a shadow
    //    on an otherwise flat beam profile).  Averaging reduces the
    //    shot-noise contamination of the flux estimate by √n_pixels.
    // 2. In the joint-Poisson profile-deviance form
    //    (`λ̂_i = c·(O_i + S_i) / (1 + c·T_i)`), a noisy per-pixel
    //    `O_i` propagates directly into `λ̂_i`, which in turn inflates
    //    the deviance without improving density recovery.
    //
    // This is a bias-variance trade: we lose the exact per-pixel paired
    // likelihood structure in exchange for a tighter density-fidelity
    // variance across pixels.  Empirically (evidence/37-…json), this is
    // the right call for the VENUS-style "flat beam with a masking
    // sample" geometry.
    //
    // **If this isn't the right assumption for your data** — e.g. you
    // have a genuinely spatially-varying beam profile and pre-estimated
    // per-pixel flux + detector-background spectra — use
    // [`InputData3D::CountsWithNuisance`] instead.  That variant
    // bypasses the averaging and pairs each pixel's sample with the
    // caller-supplied per-pixel flux and bg spectra.
    //
    // TODO(future): expose a config flag to switch the counts dispatch
    // between "averaged OB" (current, stability-oriented) and "raw
    // per-pixel OB" (exact paired joint-Poisson) if a use case arises
    // where both options are needed at call sites.
    let averaged_flux: Option<Vec<f64>> = if matches!(input, InputData3D::Counts { .. }) {
        let n_e = data_b.shape()[2]; // data_b is transposed: (h, w, n_e)
        let mut flux = vec![0.0f64; n_e];
        let n_live = pixel_coords.len() as f64;
        if n_live > 0.0 {
            for &(y, x) in &pixel_coords {
                let ob_spectrum = data_b.slice(s![y, x, ..]);
                for (e, &v) in ob_spectrum.iter().enumerate() {
                    flux[e] += v;
                }
            }
            for v in &mut flux {
                *v /= n_live;
            }
        }
        Some(flux)
    } else {
        None
    };
    let background_zeros: Vec<f64> = if matches!(input, InputData3D::Counts { .. }) {
        vec![0.0f64; data_b.shape()[2]]
    } else {
        Vec::new()
    };

    // Fit all pixels in parallel
    let failed_count = AtomicUsize::new(0);
    let results: Vec<((usize, usize), SpectrumFitResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            let spectrum_a: Vec<f64> = data_a.slice(s![y, x, ..]).to_vec();

            // Build per-pixel 1D InputData
            let pixel_input = match input {
                InputData3D::Counts { .. } => {
                    let sample_clamped: Vec<f64> = spectrum_a.iter().map(|&v| v.max(0.0)).collect();
                    let ob_spectrum: Vec<f64> = data_b.slice(s![y, x, ..]).to_vec();

                    // Check effective solver: KL uses CountsWithNuisance
                    // (averaged flux), LM uses raw Counts (auto-converts to
                    // transmission inside fit_spectrum_typed).
                    let effective = fast_config.effective_solver(&InputData::Counts {
                        sample_counts: sample_clamped.clone(),
                        open_beam_counts: ob_spectrum.clone(),
                    });
                    match effective {
                        SolverConfig::PoissonKL(_) => InputData::CountsWithNuisance {
                            sample_counts: sample_clamped,
                            flux: averaged_flux.as_ref().unwrap().clone(),
                            // Raw-count spatial path currently assumes zero
                            // detector background unless the caller provides
                            // explicit nuisance spectra.
                            background: background_zeros.clone(),
                        },
                        _ => InputData::Counts {
                            sample_counts: sample_clamped,
                            open_beam_counts: ob_spectrum,
                        },
                    }
                }
                InputData3D::CountsWithNuisance { .. } => InputData::CountsWithNuisance {
                    sample_counts: spectrum_a.iter().map(|&v| v.max(0.0)).collect(),
                    flux: data_b.slice(s![y, x, ..]).to_vec(),
                    background: data_c
                        .as_ref()
                        .expect("CountsWithNuisance requires background cube")
                        .slice(s![y, x, ..])
                        .to_vec(),
                },
                InputData3D::Transmission { .. } => {
                    let spectrum_b: Vec<f64> = data_b
                        .slice(s![y, x, ..])
                        .iter()
                        .map(|&v| v.max(1e-10))
                        .collect();
                    InputData::Transmission {
                        transmission: spectrum_a,
                        uncertainty: spectrum_b,
                    }
                }
            };

            let out = match fit_spectrum_typed(&pixel_input, &fast_config) {
                Ok(result) => Some(((y, x), result)),
                Err(_) => {
                    failed_count.fetch_add(1, Ordering::Relaxed);
                    None
                }
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
    let mut density_maps: Vec<Array2<f64>> = (0..n_maps)
        .map(|_| Array2::from_elem((height, width), f64::NAN))
        .collect();
    let mut uncertainty_maps: Vec<Array2<f64>> = (0..n_maps)
        .map(|_| Array2::from_elem((height, width), f64::NAN))
        .collect();
    let mut chi_squared_map = Array2::from_elem((height, width), f64::NAN);
    let mut deviance_per_dof_map: Option<Array2<f64>> = if input.is_counts() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut converged_map = Array2::from_elem((height, width), false);
    let mut anorm_map: Option<Array2<f64>> = if has_background_outputs {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut background_maps: Option<[Array2<f64>; 3]> = if has_background_outputs {
        Some([
            Array2::from_elem((height, width), f64::NAN),
            Array2::from_elem((height, width), f64::NAN),
            Array2::from_elem((height, width), f64::NAN),
        ])
    } else {
        None
    };
    let mut n_converged = 0;
    let mut temperature_map: Option<Array2<f64>> = if config.fit_temperature() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let mut temperature_uncertainty_map: Option<Array2<f64>> = if config.fit_temperature() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };

    for ((y, x), result) in &results {
        for i in 0..n_maps {
            density_maps[i][[*y, *x]] = result.densities[i];
            if let Some(ref unc) = result.uncertainties {
                uncertainty_maps[i][[*y, *x]] = unc[i];
            }
        }
        chi_squared_map[[*y, *x]] = result.reduced_chi_squared;
        if let (Some(dpd), Some(v)) = (&mut deviance_per_dof_map, result.deviance_per_dof) {
            dpd[[*y, *x]] = v;
        }
        converged_map[[*y, *x]] = result.converged;
        if let (Some(t_map), Some(t)) = (&mut temperature_map, result.temperature_k) {
            t_map[[*y, *x]] = t;
        }
        if let (Some(tu_map), Some(tu)) =
            (&mut temperature_uncertainty_map, result.temperature_k_unc)
        {
            tu_map[[*y, *x]] = tu;
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
        deviance_per_dof_map,
        converged_map,
        temperature_map,
        temperature_uncertainty_map,
        isotope_labels,
        anorm_map,
        background_maps,
        n_converged,
        n_total: pixel_coords.len(),
        n_failed: failed_count.load(Ordering::Relaxed),
    })
}

// ── End Phase 3 ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use nereids_fitting::lm::{FitModel, LmConfig};
    use nereids_fitting::poisson::PoissonConfig;
    use nereids_fitting::transmission_model::PrecomputedTransmissionModel;

    use crate::pipeline::{SolverConfig, UnifiedFitConfig};
    use crate::test_helpers::{synthetic_single_resonance, u238_single_resonance};

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
            energies: None,
            instrument: None,
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

    // Removed as part of the counts-KL collapse (Phase 0):
    //   test_spatial_map_typed_counts_with_nuisance_surfaces_background_maps
    // Tested fit_alpha_1 / fit_alpha_2 nuisance fitting on a 4×4 spatial
    // grid, which is no longer supported on the counts-KL dispatch (the
    // joint-Poisson profile λ̂ absorbs alpha_1 and alpha_2 / B_det is
    // P3.2-deferred; memo 35 §P3).  The SAMMY-style A_n + B_A/B/C wiring
    // on counts input is covered at pipeline scale by
    // `test_joint_poisson_with_transmission_background` (pipeline.rs);
    // a dedicated 3D-grid counterpart is not currently a test invariant
    // (spatial_map_typed is a thin per-pixel dispatcher over
    // fit_spectrum_typed, which is already covered).

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

    #[test]
    fn test_spatial_map_failed_pixels_remain_nan() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.2).collect();
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
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_counts_background(crate::pipeline::CountsBackgroundConfig {
            alpha_1_init: 1.0,
            alpha_2_init: 1.0,
            fit_alpha_1: false,
            fit_alpha_2: true,
            c: 1.0,
        });

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: ob.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();
        assert_eq!(result.n_converged, 0);
        assert!(
            result.density_maps[0].iter().all(|v| v.is_nan()),
            "failed pixels must remain NaN rather than looking like zero-density fits"
        );
        assert!(
            result.chi_squared_map.iter().all(|v| v.is_nan()),
            "failed pixels must retain NaN chi-squared"
        );
    }

    /// Spatial map with isotope groups: 2 isotopes in 1 group on a 2×2 grid.
    /// Verifies group-level density recovery and that only 1 density map is returned.
    #[test]
    fn test_spatial_map_grouped() {
        let rd1 = synthetic_single_resonance(92, 235, 233.025, 5.0);
        let rd2 = synthetic_single_resonance(92, 238, 236.006, 7.0);

        let iso1 = nereids_core::types::Isotope::new(92, 235).unwrap();
        let iso2 = nereids_core::types::Isotope::new(92, 238).unwrap();
        let group = nereids_core::types::IsotopeGroup::custom(
            "U (60/40)".into(),
            vec![(iso1, 0.6), (iso2, 0.4)],
        )
        .unwrap();

        let energies: Vec<f64> = (0..201).map(|i| 1.0 + (i as f64) * 0.05).collect();
        let n_e = energies.len();
        let true_density = 0.0005;

        // Generate synthetic transmission for the group
        let sample = nereids_physics::transmission::SampleParams::new(
            0.0,
            vec![
                (rd1.clone(), true_density * 0.6),
                (rd2.clone(), true_density * 0.4),
            ],
        )
        .unwrap();
        let t_1d = nereids_physics::transmission::forward_model(&energies, &sample, None).unwrap();
        let s_1d: Vec<f64> = t_1d.iter().map(|&v| 0.01 * v.max(0.01)).collect();

        // Fill 2×2 grid
        let mut t_3d = Array3::zeros((n_e, 2, 2));
        let mut u_3d = Array3::zeros((n_e, 2, 2));
        for y in 0..2 {
            for x in 0..2 {
                for (i, (&t, &s)) in t_1d.iter().zip(s_1d.iter()).enumerate() {
                    t_3d[[i, y, x]] = t;
                    u_3d[[i, y, x]] = s;
                }
            }
        }

        let config = UnifiedFitConfig::new(
            energies,
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

        let input = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };

        let result = spatial_map_typed(&input, &config, None, None, None).unwrap();

        // Should have 1 density map (1 group), not 2
        assert_eq!(
            result.density_maps.len(),
            1,
            "should have 1 group density map"
        );
        assert_eq!(result.isotope_labels, vec!["U (60/40)"]);
        assert_eq!(result.n_total, 4);

        // All pixels should recover true density within 5%
        for y in 0..2 {
            for x in 0..2 {
                let fitted = result.density_maps[0][[y, x]];
                let rel_error = (fitted - true_density).abs() / true_density;
                assert!(
                    rel_error < 0.05,
                    "pixel ({y},{x}): fitted={fitted}, true={true_density}, rel_error={rel_error}"
                );
            }
        }
    }

    // ── Phase 3: Spatial uncertainty propagation tests ──────────────────────

    /// Spatial LM transmission fit populates density uncertainty maps.
    #[test]
    fn test_spatial_lm_populates_density_uncertainty() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        // Add deterministic pseudo-noise so reduced chi-squared > 0
        // (a perfect fit gives chi2r=0, zeroing covariance).
        for y in 0..4 {
            for x in 0..4 {
                for e in 0..energies.len() {
                    let noise = 0.002 * ((e * 7 + y * 13 + x * 29) % 17) as f64 / 17.0 - 0.001;
                    t_3d[[e, y, x]] = (t_3d[[e, y, x]] + noise).max(0.001);
                }
            }
        }
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(result.n_converged > 0, "some pixels should converge");
        // Uncertainty maps should have finite positive values for converged pixels.
        let unc_map = &result.uncertainty_maps[0];
        let conv_map = &result.converged_map;
        let mut n_finite = 0;
        for y in 0..4 {
            for x in 0..4 {
                if conv_map[[y, x]] {
                    let u = unc_map[[y, x]];
                    assert!(
                        u.is_finite() && u > 0.0,
                        "LM density unc at ({y},{x}) should be finite+positive, got {u}"
                    );
                    n_finite += 1;
                }
            }
        }
        assert!(
            n_finite > 0,
            "at least one converged pixel should have finite unc"
        );
    }

    /// Spatial KL counts fit populates density uncertainty maps.
    #[test]
    fn test_spatial_kl_populates_density_uncertainty() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, _) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        // Convert to counts: OB=1000, sample = OB * T
        let ob_3d = Array3::from_elem(t_3d.raw_dim(), 1000.0);
        let sample_3d = &t_3d * &ob_3d;
        let data = InputData3D::Counts {
            sample_counts: sample_3d.view(),
            open_beam_counts: ob_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()));

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(result.n_converged > 0);
        let unc_map = &result.uncertainty_maps[0];
        let conv_map = &result.converged_map;
        let mut n_finite = 0;
        for y in 0..4 {
            for x in 0..4 {
                if conv_map[[y, x]] {
                    let u = unc_map[[y, x]];
                    assert!(
                        u.is_finite() && u > 0.0,
                        "KL density unc at ({y},{x}) should be finite+positive, got {u}"
                    );
                    n_finite += 1;
                }
            }
        }
        assert!(n_finite > 0);
    }

    /// Spatial temperature-fitting populates temperature_uncertainty_map.
    #[test]
    fn test_spatial_temperature_uncertainty_map() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 4.0 + (i as f64) * 0.05).collect();
        let (mut t_3d, u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        // Add pseudo-noise for nonzero chi2r.
        for y in 0..4 {
            for x in 0..4 {
                for e in 0..energies.len() {
                    let noise = 0.002 * ((e * 7 + y * 13 + x * 29) % 17) as f64 / 17.0 - 0.001;
                    t_3d[[e, y, x]] = (t_3d[[e, y, x]] + noise).max(0.001);
                }
            }
        }
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            300.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()))
        .with_fit_temperature(true);

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        assert!(result.temperature_map.is_some());
        let tu_map = result
            .temperature_uncertainty_map
            .as_ref()
            .expect("temperature_uncertainty_map should be Some when fit_temperature=true");
        assert_eq!(tu_map.shape(), [4, 4]);
        // At least some converged pixels should have finite temperature uncertainty.
        let mut n_finite = 0;
        for y in 0..4 {
            for x in 0..4 {
                if result.converged_map[[y, x]] {
                    let tu = tu_map[[y, x]];
                    if tu.is_finite() && tu > 0.0 {
                        n_finite += 1;
                    }
                }
            }
        }
        assert!(
            n_finite > 0,
            "at least one converged pixel should have finite temperature uncertainty"
        );
    }

    /// Unconverged pixels remain NaN, not zero-filled.
    #[test]
    fn test_spatial_unconverged_pixels_are_nan() {
        let rd = u238_single_resonance();
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        // Create data where pixel (0,0) is dead (all zeros)
        let (mut t_3d, mut u_3d) = synthetic_4x4_transmission(&rd, 0.001, &energies);
        for e in 0..energies.len() {
            t_3d[[e, 0, 0]] = 0.0;
            u_3d[[e, 0, 0]] = f64::INFINITY;
        }
        let data = InputData3D::Transmission {
            transmission: t_3d.view(),
            uncertainty: u_3d.view(),
        };
        let config = UnifiedFitConfig::new(
            energies,
            vec![rd],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.0005],
        )
        .unwrap()
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig::default()));

        let result = spatial_map_typed(&data, &config, None, None, None).unwrap();
        // Pixel (0,0) should not converge and uncertainty should remain NaN.
        if !result.converged_map[[0, 0]] {
            let u = result.uncertainty_maps[0][[0, 0]];
            assert!(
                u.is_nan(),
                "unconverged pixel uncertainty should be NaN, got {u}"
            );
        }
    }

    // ── Counts-KL spatial path (post-collapse) ────────────────────────

    /// Spatial counts-KL dispatch routes through `fit_counts_joint_poisson`
    /// and populates `deviance_per_dof_map`.  Polish auto-disable makes
    /// the per-pixel fits fast enough to run in a unit test; the result
    /// still recovers density on noise-free synthetic.
    #[test]
    fn test_spatial_map_typed_counts_kl_populates_deviance_per_dof_map() {
        let data = u238_single_resonance();
        let true_density = 0.0005;
        let energies: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, _) = synthetic_4x4_transmission(&data, true_density, &energies);
        let n_e = energies.len();

        // Synthesize counts: c=2.0, lam_ob=500.  E[O]=lam_ob, E[S]=c·lam_ob·T.
        let c_val = 2.0_f64;
        let lam_ob = 500.0_f64;
        let mut sample = Array3::zeros((n_e, 4, 4));
        let mut open_beam = Array3::from_elem((n_e, 4, 4), lam_ob);
        for y in 0..4 {
            for x in 0..4 {
                for (i, _) in energies.iter().enumerate() {
                    open_beam[[i, y, x]] = lam_ob;
                    sample[[i, y, x]] = c_val * lam_ob * t_3d[[i, y, x]];
                }
            }
        }

        let config = UnifiedFitConfig::new(
            energies,
            vec![data],
            vec!["U-238".into()],
            0.0,
            None,
            vec![0.001],
        )
        .unwrap()
        .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
        .with_counts_background(crate::pipeline::CountsBackgroundConfig {
            c: c_val,
            ..Default::default()
        });

        let input = InputData3D::Counts {
            sample_counts: sample.view(),
            open_beam_counts: open_beam.view(),
        };
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        // Deviance map populated (counts-KL path).
        let dpd = r
            .deviance_per_dof_map
            .as_ref()
            .expect("counts-KL spatial should populate deviance_per_dof_map");
        assert_eq!(dpd.shape(), &[4, 4]);
        let sample_val = dpd[[0, 0]];
        assert!(
            sample_val.is_finite(),
            "deviance_per_dof_map[0,0] = {sample_val} (should be finite)"
        );
        // Density recovery (noise-free).
        let density_mean: f64 = r.density_maps[0].iter().copied().sum::<f64>() / 16.0;
        assert!(
            (density_mean - true_density).abs() / true_density < 0.05,
            "mean density {density_mean} vs truth {true_density}",
        );
    }

    /// Polish auto-disable: the per-pixel dispatcher disables Nelder-Mead
    /// polish when n_pixels > 1 and the caller hasn't overridden.
    /// Verified indirectly by timing — with polish on the fit would hit
    /// the ~1000-iter cap; auto-disable keeps it fast.
    #[test]
    fn test_spatial_map_typed_counts_kl_auto_disables_polish() {
        let data = u238_single_resonance();
        let energies: Vec<f64> = (0..51).map(|i| 1.0 + (i as f64) * 0.1).collect();
        let (t_3d, _) = synthetic_4x4_transmission(&data, 0.0005, &energies);
        let n_e = energies.len();

        let mut sample = Array3::zeros((n_e, 4, 4));
        let open_beam = Array3::from_elem((n_e, 4, 4), 500.0);
        for y in 0..4 {
            for x in 0..4 {
                for i in 0..n_e {
                    sample[[i, y, x]] = 500.0 * t_3d[[i, y, x]];
                }
            }
        }

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
            open_beam_counts: open_beam.view(),
        };
        let start = std::time::Instant::now();
        let r = spatial_map_typed(&input, &config, None, None, None).unwrap();
        let elapsed = start.elapsed();
        // With polish auto-disabled, the 4x4 grid (16 pixels) on a tiny
        // 51-bin spectrum should complete well under 10 s even in debug
        // builds.  With polish enabled per pixel we'd expect ≥ several
        // seconds per pixel × 16 pixels = much longer.
        assert!(
            elapsed.as_secs() < 30,
            "spatial counts-KL ran for {elapsed:?} — polish autodisable may not be in effect",
        );
        assert!(r.deviance_per_dof_map.is_some());
    }
}
