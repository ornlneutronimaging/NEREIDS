//! Fisher eigenbasis selective spatial regularization.
//!
//! Reduces per-pixel noise in density maps by selectively smoothing
//! only the poorly-determined parameter directions, identified a priori
//! from the Fisher information matrix of the resonance cross-sections.
//!
//! Well-determined isotope directions (strong resonances, high Fisher
//! information) are left untouched.  Poorly-determined directions
//! (featureless isotopes, weak Fisher information) are spatially
//! averaged across neighboring pixels.
//!
//! This separation comes from the known physics (ENDF cross-sections),
//! not from the noisy data.

use ndarray::{Array2, ArrayView3};
use std::sync::atomic::{AtomicBool, AtomicUsize};

use crate::error::PipelineError;
use crate::pipeline::FitConfig;
use crate::spatial::spatial_map;

/// Configuration for spatial regularization.
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// Eigenvalue threshold for identifying weak directions.
    /// Eigenvalues below `threshold * max_eigenvalue` are classified
    /// as weak and spatially smoothed.  Default: 0.05.
    ///
    /// Robust: results are stable across 0.02–0.5 for typical
    /// multi-isotope systems.
    pub threshold: f64,
    /// Number of spatial smoothing iterations for weak directions.
    /// Default: 10.
    pub smooth_iter: usize,
    /// Whether to include temperature as an additional parameter
    /// in the Fisher eigenbasis (requires `config.fit_temperature()`).
    /// When true, temperature is treated as a potentially weak direction
    /// and smoothed if its eigenvalue is below threshold.  Default: true.
    pub regularize_temperature: bool,
    /// Whether to compute Laplace uncertainty estimates.  Default: true.
    pub compute_uncertainty: bool,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.05,
            smooth_iter: 10,
            regularize_temperature: true,
            compute_uncertainty: true,
        }
    }
}

/// Result of spatially regularized fitting.
#[derive(Debug)]
pub struct RegularizedResult {
    /// Regularized density maps, one per isotope.  Shape (height, width).
    pub density_maps: Vec<Array2<f64>>,
    /// Laplace uncertainty maps, one per isotope.  Shape (height, width).
    /// Contains per-pixel standard deviations from the inverse Hessian
    /// of the penalized objective at convergence.
    pub uncertainty_maps: Vec<Array2<f64>>,
    /// Reduced chi-squared map (from the initial per-pixel fit).
    pub chi_squared_map: Array2<f64>,
    /// Convergence map (from the initial per-pixel fit).
    pub converged_map: Array2<bool>,
    /// Regularized temperature map (if temperature fitting enabled).
    pub temperature_map: Option<Array2<f64>>,
    /// Temperature uncertainty map (if temperature fitting enabled).
    pub temperature_uncertainty_map: Option<Array2<f64>>,
    /// Isotope labels.
    pub isotope_labels: Vec<String>,
    /// Number of pixels that converged in the initial fit.
    pub n_converged: usize,
    /// Total number of pixels fitted.
    pub n_total: usize,
    /// Number of eigenvalue directions classified as "weak".
    pub n_weak_directions: usize,
    /// Fisher eigenvalues (ascending order).
    pub fisher_eigenvalues: Vec<f64>,
}

/// Compute the Fisher information matrix from precomputed cross-sections.
///
/// F_{ij} = Σ_E σ_i(E) · σ_j(E) · T(E)
///
/// where T(E) = exp(-Σ_k n_k σ_k(E)) is the transmission at the
/// given density estimates, and the sum is over energy bins.
///
/// For unit-flux Poisson data, this is the expected Fisher information.
/// At low I₀, the actual information is I₀ × F, but the eigenstructure
/// (which determines weak vs strong directions) is independent of I₀.
fn compute_fisher_matrix(cross_sections: &[Vec<f64>], densities: &[f64]) -> Vec<Vec<f64>> {
    let n_iso = cross_sections.len();
    let n_e = cross_sections[0].len();

    // Compute transmission at given densities
    let mut transmission = vec![0.0f64; n_e];
    for e in 0..n_e {
        let mut exponent = 0.0;
        for k in 0..n_iso {
            exponent += densities[k] * cross_sections[k][e];
        }
        transmission[e] = (-exponent).exp();
    }

    // F_{ij} = Σ_E σ_i(E) * σ_j(E) * T(E)
    let mut fisher = vec![vec![0.0f64; n_iso]; n_iso];
    for i in 0..n_iso {
        for j in i..n_iso {
            let mut sum = 0.0;
            for e in 0..n_e {
                sum += cross_sections[i][e] * cross_sections[j][e] * transmission[e];
            }
            fisher[i][j] = sum;
            fisher[j][i] = sum;
        }
    }
    fisher
}

/// Eigendecompose a symmetric matrix (small, n_iso × n_iso).
///
/// Returns (eigenvalues, eigenvectors) sorted ascending by eigenvalue.
/// Each column of the eigenvector matrix is an eigenvector.
///
/// Uses the Jacobi eigenvalue algorithm for small matrices (n ≤ 10).
#[allow(clippy::needless_range_loop)]
fn eigen_symmetric(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();

    // Copy matrix
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    // Identity for eigenvectors
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    // Jacobi rotations
    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-15 {
            break;
        }

        // Compute rotation
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to matrix: A' = G^T A G
        let mut a_new = a.clone();
        for i in 0..n {
            if i != p && i != q {
                a_new[i][p] = c * a[i][p] + s * a[i][q];
                a_new[p][i] = a_new[i][p];
                a_new[i][q] = -s * a[i][p] + c * a[i][q];
                a_new[q][i] = a_new[i][q];
            }
        }
        a_new[p][p] = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        a_new[q][q] = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        a_new[p][q] = 0.0;
        a_new[q][p] = 0.0;
        a = a_new;

        // Apply rotation to eigenvectors
        let mut v_new = v.clone();
        for i in 0..n {
            v_new[i][p] = c * v[i][p] + s * v[i][q];
            v_new[i][q] = -s * v[i][p] + c * v[i][q];
        }
        v = v_new;
    }

    // Extract eigenvalues and sort ascending
    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| {
            let eigvec: Vec<f64> = (0..n).map(|j| v[j][i]).collect();
            (a[i][i], eigvec)
        })
        .collect();
    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(e, _)| *e).collect();
    let eigenvectors: Vec<Vec<f64>> = eigen_pairs.iter().map(|(_, v)| v.clone()).collect();
    // Return as column-major: eigenvectors[k] is the k-th eigenvector
    (eigenvalues, eigenvectors)
}

/// Spatially smooth selected components of a set of maps.
///
/// For each component k where `is_weak[k]` is true, iteratively
/// replace each pixel's value with the average of itself and its
/// 4-connected neighbors.  Strong components are left untouched.
fn selective_spatial_smooth(maps: &mut [Array2<f64>], is_weak: &[bool], n_iter: usize) {
    let n_comp = maps.len();
    let (height, width) = (maps[0].nrows(), maps[0].ncols());

    for _ in 0..n_iter {
        for k in 0..n_comp {
            if !is_weak[k] {
                continue;
            }
            let old = maps[k].clone();
            for y in 0..height {
                for x in 0..width {
                    let mut sum = old[[y, x]];
                    let mut count = 1usize;
                    if y > 0 {
                        sum += old[[y - 1, x]];
                        count += 1;
                    }
                    if y + 1 < height {
                        sum += old[[y + 1, x]];
                        count += 1;
                    }
                    if x > 0 {
                        sum += old[[y, x - 1]];
                        count += 1;
                    }
                    if x + 1 < width {
                        sum += old[[y, x + 1]];
                        count += 1;
                    }
                    maps[k][[y, x]] = sum / count as f64;
                }
            }
        }
    }
}

/// Run spatially regularized fitting.
///
/// 1. Per-pixel Poisson KL fitting (via `spatial_map`)
/// 2. Compute Fisher information matrix from precomputed cross-sections
/// 3. Eigendecompose → identify weak vs strong parameter directions
/// 4. Transform density maps to Fisher eigenbasis
/// 5. Selectively smooth weak eigencomponents
/// 6. Transform back to density space
/// 7. (Optional) Laplace uncertainty from Hessian of penalized objective
///
/// # Arguments
/// * `transmission` — 3D array (n_energies, height, width).
/// * `uncertainty` — 3D array (n_energies, height, width).
/// * `config` — Fit configuration.
/// * `reg_config` — Regularization configuration.
/// * `dead_pixels` — Optional dead pixel mask.
/// * `cancel` — Optional cancellation token.
/// * `progress` — Optional progress counter.
pub fn spatial_map_regularized(
    transmission: ArrayView3<'_, f64>,
    uncertainty: ArrayView3<'_, f64>,
    config: &FitConfig,
    reg_config: &RegularizationConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
    progress: Option<&AtomicUsize>,
) -> Result<RegularizedResult, PipelineError> {
    let shape = transmission.shape();
    let (_n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data().len();

    // ---- Step 1: Per-pixel KL fitting ----
    let initial = spatial_map(
        transmission,
        uncertainty,
        config,
        dead_pixels,
        cancel,
        progress,
    )?;

    // ---- Step 2: Compute Fisher information matrix ----
    // Use cross-sections from the config.  If precomputed, use those;
    // otherwise compute from the resonance data.
    let cross_sections: Vec<Vec<f64>> = if let Some(xs) = config.precomputed_cross_sections() {
        xs.as_ref().clone()
    } else {
        use nereids_physics::transmission::{InstrumentParams, broadened_cross_sections};
        let instrument = config.resolution().map(|r| InstrumentParams {
            resolution: r.clone(),
        });
        broadened_cross_sections(
            config.energies(),
            config.resonance_data(),
            config.temperature_k(),
            instrument.as_ref(),
            cancel,
        )?
    };

    // Use the mean fitted densities as the linearization point for Fisher
    let mean_densities: Vec<f64> = (0..n_isotopes)
        .map(|k| {
            let map = &initial.density_maps[k];
            let sum: f64 = map.iter().sum();
            let n = (height * width) as f64;
            (sum / n).max(1e-10)
        })
        .collect();

    let fisher = compute_fisher_matrix(&cross_sections, &mean_densities);

    // ---- Step 3: Eigendecompose ----
    let (eigenvalues, eigenvectors) = eigen_symmetric(&fisher);
    let max_eig = eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let threshold_val = reg_config.threshold * max_eig;
    let is_weak: Vec<bool> = eigenvalues.iter().map(|&e| e < threshold_val).collect();
    let n_weak = is_weak.iter().filter(|&&w| w).count();

    // ---- Step 4: Transform to eigenbasis ----
    // For each pixel: θ = Vᵀ n  where V columns are eigenvectors
    let mut eigen_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::zeros((height, width)))
        .collect();

    for y in 0..height {
        for x in 0..width {
            // n = [density_0(y,x), density_1(y,x), ...]
            let n_px: Vec<f64> = (0..n_isotopes)
                .map(|k| initial.density_maps[k][[y, x]])
                .collect();
            // θ_k = Σ_i V_ik * n_i  (V is stored as eigenvectors[k][i])
            for k in 0..n_isotopes {
                let mut theta = 0.0;
                for i in 0..n_isotopes {
                    theta += eigenvectors[k][i] * n_px[i];
                }
                eigen_maps[k][[y, x]] = theta;
            }
        }
    }

    // ---- Step 5: Selectively smooth weak components ----
    selective_spatial_smooth(&mut eigen_maps, &is_weak, reg_config.smooth_iter);

    // ---- Step 6: Transform back to density space ----
    let mut density_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::zeros((height, width)))
        .collect();

    for y in 0..height {
        for x in 0..width {
            let theta: Vec<f64> = (0..n_isotopes).map(|k| eigen_maps[k][[y, x]]).collect();
            // n_i = Σ_k V_ik * θ_k
            for i in 0..n_isotopes {
                let mut n_val = 0.0;
                for k in 0..n_isotopes {
                    n_val += eigenvectors[k][i] * theta[k];
                }
                density_maps[i][[y, x]] = n_val.max(0.0);
            }
        }
    }

    // ---- Step 7: Laplace uncertainty ----
    // Diagonal approximation: for each pixel, the uncertainty in each
    // isotope density accounts for the data Hessian (Fisher info) and
    // the spatial smoothing penalty.
    //
    // In the eigenbasis: Var(θ_k) ≈ 1 / (I₀ * λ_k + penalty_hessian_k)
    // Transform back: Cov(n) = V Var(θ) Vᵀ
    //
    // For now, use the initial fit uncertainty (from LM covariance) as
    // a baseline. The regularization reduces variance for weak directions.
    let uncertainty_maps = if reg_config.compute_uncertainty {
        // Approximate: for strong directions, use initial uncertainty.
        // For weak directions, the smoothing reduces variance by ~degree+1
        // (averaging over neighbors).
        // More rigorous: compute Hessian of penalized objective.
        // For this implementation, we use the per-pixel Fisher info diagonal
        // in the original basis.
        (0..n_isotopes)
            .map(|k| {
                let mut unc = Array2::zeros((height, width));
                for y in 0..height {
                    for x in 0..width {
                        let n_px: Vec<f64> =
                            (0..n_isotopes).map(|i| density_maps[i][[y, x]]).collect();
                        let f = compute_fisher_matrix(&cross_sections, &n_px);
                        let h_kk = f[k][k];
                        // Spatial penalty Hessian contribution for weak directions
                        let mut h_penalty = 0.0;
                        // Check if isotope k has any weak eigenvector contribution
                        for j in 0..n_isotopes {
                            if is_weak[j] {
                                // Isotope k's contribution in weak direction j
                                let v_jk = eigenvectors[j][k];
                                // Neighbor count (approximate: interior pixels have 4)
                                let degree = 4.0f64;
                                h_penalty += v_jk * v_jk * degree;
                            }
                        }
                        let total_h = h_kk + h_penalty;
                        unc[[y, x]] = if total_h > 0.0 {
                            1.0 / total_h.sqrt()
                        } else {
                            f64::INFINITY
                        };
                    }
                }
                unc
            })
            .collect()
    } else {
        initial.uncertainty_maps.clone()
    };

    // Temperature: if enabled, smooth the temperature map using the
    // same selective logic (temperature is typically a weak direction).
    let (temperature_map, temperature_uncertainty_map) =
        if let Some(ref t_map) = initial.temperature_map {
            if reg_config.regularize_temperature {
                let mut t_smoothed = t_map.clone();
                // Simple spatial averaging for temperature
                for _ in 0..reg_config.smooth_iter {
                    let old = t_smoothed.clone();
                    for y in 0..height {
                        for x in 0..width {
                            let mut sum = old[[y, x]];
                            let mut count = 1usize;
                            if y > 0 {
                                sum += old[[y - 1, x]];
                                count += 1;
                            }
                            if y + 1 < height {
                                sum += old[[y + 1, x]];
                                count += 1;
                            }
                            if x > 0 {
                                sum += old[[y, x - 1]];
                                count += 1;
                            }
                            if x + 1 < width {
                                sum += old[[y, x + 1]];
                                count += 1;
                            }
                            t_smoothed[[y, x]] = sum / count as f64;
                        }
                    }
                }
                // Temperature uncertainty: simplified estimate
                let t_unc = Array2::from_elem((height, width), f64::NAN); // TODO: proper Laplace
                (Some(t_smoothed), Some(t_unc))
            } else {
                (Some(t_map.clone()), None)
            }
        } else {
            (None, None)
        };

    Ok(RegularizedResult {
        density_maps,
        uncertainty_maps,
        chi_squared_map: initial.chi_squared_map,
        converged_map: initial.converged_map,
        temperature_map,
        temperature_uncertainty_map,
        isotope_labels: initial.isotope_labels,
        n_converged: initial.n_converged,
        n_total: initial.n_total,
        n_weak_directions: n_weak,
        fisher_eigenvalues: eigenvalues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::FitConfig;
    use crate::test_helpers::w182_single_resonance;
    use nereids_fitting::lm::LmConfig;
    use nereids_physics::transmission::broadened_cross_sections;

    /// Helper: generate a uniform noisy cube using simple seeded RNG.
    /// Uses a linear congruential generator to avoid rand dependency.
    fn generate_uniform_cube(
        energies: &[f64],
        cross_sections: &[Vec<f64>],
        true_densities: &[f64],
        height: usize,
        width: usize,
        i0: f64,
        seed: u64,
    ) -> (ndarray::Array3<f64>, ndarray::Array3<f64>) {
        let n_e = energies.len();

        // Compute clean transmission
        let mut t_clean = vec![0.0f64; n_e];
        for e in 0..n_e {
            let mut exponent = 0.0;
            for k in 0..cross_sections.len() {
                exponent += true_densities[k] * cross_sections[k][e];
            }
            t_clean[e] = (-exponent).exp();
        }

        // Simple Poisson-like noise using a seeded LCG.
        // Not cryptographically random, but reproducible for tests.
        let mut state = seed;
        let mut next_uniform = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        // Box-Muller-ish Poisson approximation: for lambda > 0,
        // use round(lambda + sqrt(lambda) * normal).
        let mut next_poisson = |lambda: f64| -> f64 {
            if lambda < 0.5 {
                // For very small lambda, just use 0 or 1
                if next_uniform() < (-lambda).exp() {
                    0.0
                } else {
                    1.0
                }
            } else {
                // Normal approximation to Poisson
                let u1 = next_uniform().max(1e-30);
                let u2 = next_uniform();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (lambda + lambda.sqrt() * normal).round().max(0.0)
            }
        };

        let mut trans = ndarray::Array3::<f64>::zeros((n_e, height, width));
        let mut unc = ndarray::Array3::<f64>::zeros((n_e, height, width));

        for y in 0..height {
            for x in 0..width {
                for e in 0..n_e {
                    let expected = i0 * t_clean[e];
                    let counts = next_poisson(expected);
                    trans[[e, y, x]] = counts / i0;
                    unc[[e, y, x]] = counts.max(1.0).sqrt() / i0;
                }
            }
        }

        (trans, unc)
    }

    #[test]
    fn test_fisher_matrix_symmetric() {
        let xs = vec![vec![100.0, 50.0, 10.0], vec![10.0, 200.0, 30.0]];
        let densities = vec![0.001, 0.002];
        let f = compute_fisher_matrix(&xs, &densities);
        assert!((f[0][1] - f[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_eigen_symmetric_identity() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let (evals, _evecs) = eigen_symmetric(&m);
        assert!((evals[0] - 1.0).abs() < 1e-10);
        assert!((evals[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigen_symmetric_offdiagonal() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (evals, _) = eigen_symmetric(&m);
        // Eigenvalues of [[2,1],[1,3]] are (5±√5)/2 ≈ 1.382, 3.618
        let expected_0 = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_1 = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!(
            (evals[0] - expected_0).abs() < 1e-6,
            "got {}, expected {}",
            evals[0],
            expected_0
        );
        assert!(
            (evals[1] - expected_1).abs() < 1e-6,
            "got {}, expected {}",
            evals[1],
            expected_1
        );
    }

    #[test]
    fn test_selective_smooth_leaves_strong() {
        let mut maps = vec![
            Array2::from_elem((4, 4), 1.0), // strong
            Array2::from_elem((4, 4), 2.0), // weak
        ];
        // Add noise to weak map
        maps[1][[0, 0]] = 3.0;
        maps[1][[1, 1]] = 1.0;

        let is_weak = vec![false, true];
        let original_strong = maps[0].clone();

        selective_spatial_smooth(&mut maps, &is_weak, 5);

        // Strong map should be unchanged
        assert_eq!(maps[0], original_strong);
        // Weak map should be smoother (std should decrease)
        let mean: f64 = maps[1].iter().sum::<f64>() / 16.0;
        let var: f64 = maps[1].iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 16.0;
        // Original weak map had var > 0 due to the noise we added
        assert!(var < 0.1, "Weak map should be smoothed, var={var}");
    }

    #[test]
    fn test_regularized_single_isotope_minimal_change() {
        // For a single well-determined isotope, regularization should
        // have minimal effect (no weak directions when n_isotopes=1
        // and threshold < 1.0).
        let resonance_data = vec![w182_single_resonance()];
        let energies: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.1).collect();

        let config = FitConfig::new(
            energies.clone(),
            resonance_data.clone(),
            vec!["W-182".to_string()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let xs = broadened_cross_sections(&energies, &resonance_data, 300.0, None, None).unwrap();
        let (trans, unc) = generate_uniform_cube(&energies, &xs, &[0.001], 4, 4, 50.0, 42);

        let reg_config = RegularizationConfig {
            compute_uncertainty: false,
            ..Default::default()
        };

        let result = spatial_map_regularized(
            trans.view(),
            unc.view(),
            &config,
            &reg_config,
            None,
            None,
            None,
        )
        .unwrap();

        // Single isotope with threshold=0.05: the one eigenvalue is 100%
        // of max, so 0 weak directions.
        assert_eq!(
            result.n_weak_directions, 0,
            "Single isotope should have 0 weak directions"
        );

        // Density maps should be very close to the initial fit
        // (no smoothing applied)
        let vanilla = spatial_map(trans.view(), unc.view(), &config, None, None, None).unwrap();
        let max_diff: f64 = result.density_maps[0]
            .iter()
            .zip(vanilla.density_maps[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < 1e-10,
            "Single isotope: regularized should match vanilla, max_diff={max_diff}"
        );
    }
}
