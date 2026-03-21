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
}

impl InputData3D<'_> {
    /// Shape of the data: (n_energies, height, width).
    pub(crate) fn shape(&self) -> (usize, usize, usize) {
        let s = match self {
            Self::Transmission { transmission, .. } => transmission.shape(),
            Self::Counts { sample_counts, .. } => sample_counts.shape(),
        };
        (s[0], s[1], s[2])
    }

    /// Whether this is counts data.
    pub(crate) fn is_counts(&self) -> bool {
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
            density_maps: (0..n_maps)
                .map(|_| Array2::zeros((height, width)))
                .collect(),
            uncertainty_maps: (0..n_maps)
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

    // When groups are active and temperature is NOT being fitted, collapse
    // per-member broadened XS into per-group σ_eff once here.  This avoids
    // redundant O(n_members × n_energies) collapsing inside
    // build_transmission_model on every per-pixel call.
    let xs = if !config.fit_temperature()
        && let (Some(di), Some(dr)) = (&config.density_indices, &config.density_ratios)
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
            .with_compute_covariance(false)
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
            .with_compute_covariance(false)
    };

    let is_counts = input.is_counts();
    let has_transmission_bg = config.transmission_background().is_some();

    // For counts data: spatially average the open beam to get a stable flux
    // estimate, reducing per-pixel open-beam shot noise.
    // Without this, per-pixel open-beam shot noise contaminates the flux
    // estimate and makes KL fits materially noisier.
    let averaged_flux: Option<Vec<f64>> = if is_counts {
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
    let background_zeros: Vec<f64> = if is_counts {
        vec![0.0f64; data_b.shape()[2]]
    } else {
        Vec::new()
    };

    // Fit all pixels in parallel
    let results: Vec<((usize, usize), SpectrumFitResult)> = pixel_coords
        .par_iter()
        .filter_map(|&(y, x)| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }

            let spectrum_a: Vec<f64> = data_a.slice(s![y, x, ..]).to_vec();

            // Build per-pixel 1D InputData
            let pixel_input = if is_counts {
                let sample_clamped: Vec<f64> = spectrum_a.iter().map(|&v| v.max(0.0)).collect();
                let ob_spectrum: Vec<f64> = data_b.slice(s![y, x, ..]).to_vec();

                // Check effective solver: KL uses CountsWithNuisance (averaged
                // flux), LM uses raw Counts (auto-converts to transmission
                // inside fit_spectrum_typed).
                let effective = fast_config.effective_solver(&InputData::Counts {
                    sample_counts: sample_clamped.clone(),
                    open_beam_counts: ob_spectrum.clone(),
                });
                match effective {
                    SolverConfig::PoissonKL(_) => InputData::CountsWithNuisance {
                        sample_counts: sample_clamped,
                        flux: averaged_flux.as_ref().unwrap().clone(),
                        background: background_zeros.clone(),
                    },
                    _ => InputData::Counts {
                        sample_counts: sample_clamped,
                        open_beam_counts: ob_spectrum,
                    },
                }
            } else {
                let spectrum_b: Vec<f64> = data_b
                    .slice(s![y, x, ..])
                    .iter()
                    .map(|&v| v.max(1e-10))
                    .collect();
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
    let mut density_maps: Vec<Array2<f64>> = (0..n_maps)
        .map(|_| Array2::zeros((height, width)))
        .collect();
    let mut uncertainty_maps: Vec<Array2<f64>> = (0..n_maps)
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
    let mut temperature_map: Option<Array2<f64>> = if config.fit_temperature() {
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
        n_total: pixel_coords.len(),
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
}
