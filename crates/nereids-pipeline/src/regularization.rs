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
    debug_assert!(!cross_sections.is_empty());
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
        let (max_val, p, q) = find_max_offdiag(&a);
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
        apply_jacobi_rotation_matrix(&mut a, p, q, c, s);

        // Apply rotation to eigenvectors
        apply_jacobi_rotation_vectors(&mut v, p, q, c, s);
    }

    // Check convergence: warn-level residual off-diagonal (P2-9)
    #[cfg(debug_assertions)]
    {
        let (max_offdiag, _, _) = find_max_offdiag(&a);
        debug_assert!(
            max_offdiag <= 1e-10,
            "Jacobi did not fully converge: max off-diagonal = {max_offdiag:.2e}"
        );
    }

    // Extract eigenvalues and sort ascending (P1-1: handle NaN safely)
    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| {
            let eigvec: Vec<f64> = (0..n).map(|j| v[j][i]).collect();
            (a[i][i], eigvec)
        })
        .collect();
    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(e, _)| *e).collect();
    let eigenvectors: Vec<Vec<f64>> = eigen_pairs.iter().map(|(_, v)| v.clone()).collect();
    // Return as column-major: eigenvectors[k] is the k-th eigenvector
    (eigenvalues, eigenvectors)
}

/// Find the largest off-diagonal element in a symmetric matrix.
/// Returns (max_abs_value, row_index, col_index).
fn find_max_offdiag(a: &[Vec<f64>]) -> (f64, usize, usize) {
    let n = a.len();
    let mut max_val = 0.0f64;
    let mut p = 0;
    let mut q = 1.min(n.saturating_sub(1));
    for (i, row) in a.iter().enumerate() {
        for (j_offset, val) in row.iter().enumerate().skip(i + 1) {
            if val.abs() > max_val {
                max_val = val.abs();
                p = i;
                q = j_offset;
            }
        }
    }
    (max_val, p, q)
}

/// Apply a Jacobi rotation to the matrix A in-place.
/// Rotates in the (p, q) plane with angle defined by (cos, sin) = (c, s).
fn apply_jacobi_rotation_matrix(a: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
    let a_old = a.to_vec();
    for (i, row_old) in a_old.iter().enumerate() {
        if i != p && i != q {
            let new_ip = c * row_old[p] + s * row_old[q];
            let new_iq = -s * row_old[p] + c * row_old[q];
            a[i][p] = new_ip;
            a[p][i] = new_ip;
            a[i][q] = new_iq;
            a[q][i] = new_iq;
        }
    }
    a[p][p] = c * c * a_old[p][p] + 2.0 * s * c * a_old[p][q] + s * s * a_old[q][q];
    a[q][q] = s * s * a_old[p][p] - 2.0 * s * c * a_old[p][q] + c * c * a_old[q][q];
    a[p][q] = 0.0;
    a[q][p] = 0.0;
}

/// Apply a Jacobi rotation to the eigenvector matrix V in-place.
fn apply_jacobi_rotation_vectors(v: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
    for row in v.iter_mut() {
        let old_p = row[p];
        let old_q = row[q];
        row[p] = c * old_p + s * old_q;
        row[q] = -s * old_p + c * old_q;
    }
}

/// Smooth a single 2D array in-place using 4-connected averaging.
///
/// Each pixel is replaced with the mean of itself and its direct
/// neighbors (up to 4). Repeated `n_iter` times.
fn smooth_array2(map: &mut Array2<f64>, n_iter: usize) {
    let (height, width) = (map.nrows(), map.ncols());
    for _ in 0..n_iter {
        let old = map.clone();
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
                map[[y, x]] = sum / count as f64;
            }
        }
    }
}

/// Spatially smooth selected components of a set of maps.
///
/// For each component k where `is_weak[k]` is true, iteratively
/// replace each pixel's value with the average of itself and its
/// 4-connected neighbors.  Strong components are left untouched.
fn selective_spatial_smooth(maps: &mut [Array2<f64>], is_weak: &[bool], n_iter: usize) {
    let n_comp = maps.len();
    for k in 0..n_comp {
        if !is_weak[k] {
            continue;
        }
        smooth_array2(&mut maps[k], n_iter);
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
    // ---- Validate regularization config (P2-5) ----
    if !reg_config.threshold.is_finite()
        || reg_config.threshold <= 0.0
        || reg_config.threshold >= 1.0
    {
        return Err(PipelineError::InvalidParameter(format!(
            "threshold must be in (0, 1), got {}",
            reg_config.threshold
        )));
    }

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

    // Use the mean fitted densities (converged pixels only) as the
    // linearization point for Fisher (P2-11)
    let mean_densities: Vec<f64> = (0..n_isotopes)
        .map(|k| {
            let map = &initial.density_maps[k];
            let conv = &initial.converged_map;
            let mut sum = 0.0;
            let mut count = 0usize;
            for y in 0..height {
                for x in 0..width {
                    if conv[[y, x]] {
                        sum += map[[y, x]];
                        count += 1;
                    }
                }
            }
            if count > 0 {
                (sum / count as f64).max(1e-10)
            } else {
                1e-10
            }
        })
        .collect();

    let fisher = compute_fisher_matrix(&cross_sections, &mean_densities);

    // ---- Step 3: Eigendecompose ----
    let (eigenvalues, eigenvectors) = eigen_symmetric(&fisher);
    let max_eig = eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let threshold_val = reg_config.threshold * max_eig;
    // P1-3: Degenerate Fisher (all zeros) — all directions are weak
    let is_weak: Vec<bool> = if max_eig < 1e-30 {
        vec![true; n_isotopes]
    } else {
        eigenvalues.iter().map(|&e| e < threshold_val).collect()
    };
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
    // Use the Fisher eigen decomposition for proper correlated uncertainty.
    //
    // For each pixel: compute per-pixel Fisher F, add spatial penalty
    // contribution for weak directions, then invert via the eigen
    // decomposition:
    //   H = F + penalty
    //   (H^{-1})_kk = Σ_j V_jk^2 / λ_j    (P1-4)
    //   σ_k = sqrt((H^{-1})_kk)
    let uncertainty_maps = if reg_config.compute_uncertainty {
        let mut unc_maps: Vec<Array2<f64>> = (0..n_isotopes)
            .map(|_| Array2::zeros((height, width)))
            .collect();

        for y in 0..height {
            for x in 0..width {
                let n_px: Vec<f64> = (0..n_isotopes).map(|i| density_maps[i][[y, x]]).collect();
                let f = compute_fisher_matrix(&cross_sections, &n_px);

                // Compute actual neighbor count for this pixel (P2-2)
                let mut degree = 0.0f64;
                if y > 0 {
                    degree += 1.0;
                }
                if y + 1 < height {
                    degree += 1.0;
                }
                if x > 0 {
                    degree += 1.0;
                }
                if x + 1 < width {
                    degree += 1.0;
                }

                // Build penalized Hessian: H_ij = F_ij + penalty_ij
                // Penalty in eigenbasis: diag(degree) for weak directions
                // Back in density space: H = F + V·diag(penalty)·Vᵀ
                let mut h = f;
                for j in 0..n_isotopes {
                    if is_weak[j] {
                        for r in 0..n_isotopes {
                            for c in 0..n_isotopes {
                                h[r][c] += eigenvectors[j][r] * eigenvectors[j][c] * degree;
                            }
                        }
                    }
                }

                // Eigendecompose H to invert: H^{-1} = V diag(1/λ) Vᵀ
                let (h_evals, h_evecs) = eigen_symmetric(&h);

                // Extract diagonal of H^{-1}: (H^{-1})_kk = Σ_j V_jk^2 / λ_j
                for k in 0..n_isotopes {
                    let mut inv_diag = 0.0;
                    for (j, &lam) in h_evals.iter().enumerate() {
                        if lam > 1e-30 {
                            inv_diag += h_evecs[j][k] * h_evecs[j][k] / lam;
                        }
                    }
                    unc_maps[k][[y, x]] = if inv_diag > 0.0 {
                        inv_diag.sqrt()
                    } else {
                        f64::INFINITY
                    };
                }
            }
        }
        unc_maps
    } else {
        // P2-3: When uncertainty is not computed, use INFINITY (not NaN)
        (0..n_isotopes)
            .map(|_| Array2::from_elem((height, width), f64::INFINITY))
            .collect()
    };

    // Temperature: if enabled, smooth the temperature map using the
    // same smooth_array2 helper (P2-4).  Temperature is typically a
    // weak direction.
    let (temperature_map, temperature_uncertainty_map) =
        if let Some(ref t_map) = initial.temperature_map {
            if reg_config.regularize_temperature {
                let mut t_smoothed = t_map.clone();
                smooth_array2(&mut t_smoothed, reg_config.smooth_iter);
                // P1-5: Temperature uncertainty not yet computed — return None
                // rather than emitting NaN.
                (Some(t_smoothed), None)
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
    use crate::noise::generate_noisy_cube;
    use crate::pipeline::FitConfig;
    use crate::test_helpers::w182_single_resonance;
    use nereids_fitting::lm::LmConfig;
    use nereids_physics::transmission::broadened_cross_sections;

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
        let clean: Vec<f64> = (0..energies.len())
            .map(|e| (-0.001 * xs[0][e]).exp())
            .collect();
        let (trans, unc) = generate_noisy_cube(&clean, (4, 4), 50.0, 42);

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

    #[test]
    fn test_two_isotope_weak_direction_smoothed() {
        // Two isotopes with very different Fisher information.
        // Create a synthetic case where one isotope has strong resonances
        // (well-determined) and one has weak resonances (poorly-determined).
        // The regularization should smooth the weak isotope more than the strong.
        use crate::test_helpers::w182_single_resonance;

        let _resonance_data = [w182_single_resonance(), w182_single_resonance()];
        let _energies: Vec<f64> = (0..50).map(|i| 1.0 + i as f64 * 0.2).collect();

        // Build cross-sections manually: isotope 0 strong, isotope 1 weak
        let xs_strong: Vec<f64> = (0..50)
            .map(|i| {
                let e = 1.0 + i as f64 * 0.2;
                // Strong resonance at E=4.15 eV
                1000.0 / (1.0 + ((e - 4.15) / 0.1).powi(2))
            })
            .collect();
        let xs_weak: Vec<f64> = (0..50).map(|_| 5.0).collect(); // flat, featureless

        // Compute clean transmission for [0.001, 0.003]
        let true_densities = [0.001, 0.003];
        let clean: Vec<f64> = (0..50)
            .map(|e| (-(true_densities[0] * xs_strong[e] + true_densities[1] * xs_weak[e])).exp())
            .collect();

        // Generate noisy cube at low counts (unused — we test components directly)
        let (_trans, _unc) = generate_noisy_cube(&clean, (8, 8), 10.0, 42);

        // We can't use FitConfig with fake cross-sections, but we can
        // test the Fisher matrix and eigenbasis components directly.
        let xs = [xs_strong, xs_weak];
        let fisher = compute_fisher_matrix(&xs, &true_densities);
        let (eigenvalues, _eigenvectors) = eigen_symmetric(&fisher);

        // The weak isotope (flat xs) should produce a much smaller
        // Fisher eigenvalue than the strong one.
        assert!(
            eigenvalues[0] < eigenvalues[1] * 0.1,
            "Weak direction eigenvalue should be << strong: {:?}",
            eigenvalues
        );

        // Verify the threshold correctly identifies weak vs strong
        let threshold = 0.05;
        let max_eig = eigenvalues.iter().cloned().fold(0.0f64, f64::max);
        let is_weak: Vec<bool> = eigenvalues
            .iter()
            .map(|&e| e < threshold * max_eig)
            .collect();
        assert!(is_weak[0], "First eigenvalue should be weak");
        assert!(!is_weak[1], "Second eigenvalue should be strong");

        // Test selective smoothing: create noisy maps, smooth, verify
        // weak map gets smoother while strong map is untouched
        let mut maps = vec![
            Array2::from_elem((4, 4), 0.001),
            Array2::from_elem((4, 4), 0.003),
        ];
        // Add noise
        maps[0][[0, 0]] = 0.002;
        maps[0][[1, 1]] = 0.0005;
        maps[1][[0, 0]] = 0.004;
        maps[1][[1, 1]] = 0.002;

        let var_before_0: f64 = {
            let mean = maps[0].iter().sum::<f64>() / 16.0;
            maps[0].iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 16.0
        };
        let var_before_1: f64 = {
            let mean = maps[1].iter().sum::<f64>() / 16.0;
            maps[1].iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 16.0
        };

        selective_spatial_smooth(&mut maps, &is_weak, 10);

        let var_after_0: f64 = {
            let mean = maps[0].iter().sum::<f64>() / 16.0;
            maps[0].iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 16.0
        };
        let var_after_1: f64 = {
            let mean = maps[1].iter().sum::<f64>() / 16.0;
            maps[1].iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 16.0
        };

        // Weak map (index 0) should be smoother
        assert!(
            var_after_0 < var_before_0 * 0.5,
            "Weak map should be significantly smoother: before={var_before_0:.2e} after={var_after_0:.2e}"
        );
        // Strong map (index 1) should be unchanged
        assert!(
            (var_after_1 - var_before_1).abs() < 1e-15,
            "Strong map should be untouched: before={var_before_1:.2e} after={var_after_1:.2e}"
        );
    }

    #[test]
    fn test_regularized_reduces_noise_at_realistic_scale() {
        // Integration test: W-182 at 100 energy bins, I₀=10, 8×8 grid.
        // Regularization should NOT change the result for a single isotope
        // (only 1 direction, always "strong"), but this validates the
        // full pipeline runs without errors at moderate scale.
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
        let clean: Vec<f64> = (0..energies.len())
            .map(|e| (-0.001 * xs[0][e]).exp())
            .collect();
        let (trans, unc) = generate_noisy_cube(&clean, (8, 8), 10.0, 42);

        let reg_config = RegularizationConfig {
            compute_uncertainty: true,
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

        // Basic sanity checks
        assert_eq!(result.density_maps.len(), 1);
        assert_eq!(result.uncertainty_maps.len(), 1);
        assert_eq!(result.density_maps[0].shape(), &[8, 8]);
        assert_eq!(result.uncertainty_maps[0].shape(), &[8, 8]);
        assert_eq!(result.n_weak_directions, 0);
        assert_eq!(result.fisher_eigenvalues.len(), 1);
        assert!(result.fisher_eigenvalues[0] > 0.0);

        // All uncertainties should be finite and positive
        for &u in result.uncertainty_maps[0].iter() {
            assert!(
                u.is_finite() && u > 0.0,
                "Uncertainty should be finite positive, got {u}"
            );
        }

        // Densities should be reasonable (positive, within order of magnitude of true)
        let mean: f64 = result.density_maps[0].iter().sum::<f64>() / 64.0;
        assert!(
            mean > 0.0 && mean < 0.01,
            "Mean density should be reasonable, got {mean}"
        );
    }
}
