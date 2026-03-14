//! Total Variation regularization via ADMM for spatial density maps.
//!
//! Produces spatially coherent density maps while preserving material
//! boundaries (edges).  For each isotope k independently, minimizes:
//!
//! ```text
//! Σ_pixels L_pixel(n_k)  +  λ_k · Σ_edges |n_k(i) - n_k(j)|
//! ```
//!
//! where L_pixel is the per-pixel data-fidelity term (LM chi² or Poisson NLL)
//! and the second term is the isotropic Total Variation penalty on the
//! 4-connected pixel grid.
//!
//! ## Algorithm: linearized ADMM
//!
//! The per-pixel subproblem (n-update) freezes neighbors at the previous
//! iteration.  This decomposes the ADMM augmented Lagrangian into
//! independent per-pixel objectives, each with a single proximal penalty
//! term: `(ρ · deg(p) / 2) · ||n_k - target_k(p)||²`.
//!
//! This maps directly to the existing per-pixel fitting API via
//! [`ProximalPenalty`].
//!
//! ## Reference
//! - Boyd et al., "Distributed Optimization and Statistical Learning via
//!   the Alternating Direction Method of Multipliers", Foundations and
//!   Trends in Machine Learning, 2011.

use ndarray::{Array2, ArrayView3, s};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nereids_fitting::proximal::ProximalPenalty;

use crate::error::PipelineError;
use crate::pipeline::{FitConfig, ParameterRole, SolverChoice, fit_spectrum};
use crate::spatial::{AdmmConvergenceInfo, SpatialResult, precompute_config, spatial_map};

/// Relative density change threshold for ADMM-level convergence.
///
/// When per-pixel densities change by less than this between ADMM
/// iterations, the pixel is considered converged at the ADMM level.
/// This is a fallback for inner solvers (particularly Poisson FD) that
/// may fail to declare convergence near the warm-started optimum due
/// to FD gradient noise.
const DENSITY_STABILIZATION_TOL: f64 = 1e-6;

/// Configuration for TV-ADMM spatial regularization.
#[derive(Debug, Clone)]
pub struct TvAdmmConfig {
    /// TV strength per isotope. Length must match n_isotopes.
    pub lambda: Vec<f64>,
    /// ADMM penalty parameter. Default: 1.0.
    pub rho: f64,
    /// Maximum ADMM outer iterations. Default: 20.
    pub max_outer_iter: usize,
    /// Primal residual convergence tolerance. Default: 1e-4.
    pub tol_primal: f64,
    /// Dual residual convergence tolerance. Default: 1e-4.
    pub tol_dual: f64,
}

impl Default for TvAdmmConfig {
    fn default() -> Self {
        Self {
            lambda: Vec::new(),
            rho: 1.0,
            max_outer_iter: 20,
            tol_primal: 1e-4,
            tol_dual: 1e-4,
        }
    }
}

/// An edge between two pixels in the 4-connected grid.
#[derive(Debug, Clone, Copy)]
struct Edge {
    /// Linear index of pixel A.
    pixel_a: usize,
    /// Linear index of pixel B.
    pixel_b: usize,
}

/// Build the 4-connected edge list for an (h, w) grid.
///
/// Each edge connects horizontally or vertically adjacent pixels.
/// Dead pixels and their incident edges are excluded.
fn build_edge_list(h: usize, w: usize, dead_pixels: Option<&Array2<bool>>) -> Vec<Edge> {
    let is_dead = |y: usize, x: usize| -> bool { dead_pixels.is_some_and(|m| m[[y, x]]) };

    let mut edges = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if is_dead(y, x) {
                continue;
            }
            // Right neighbor.
            if x + 1 < w && !is_dead(y, x + 1) {
                edges.push(Edge {
                    pixel_a: y * w + x,
                    pixel_b: y * w + (x + 1),
                });
            }
            // Down neighbor.
            if y + 1 < h && !is_dead(y + 1, x) {
                edges.push(Edge {
                    pixel_a: y * w + x,
                    pixel_b: (y + 1) * w + x,
                });
            }
        }
    }
    edges
}

/// Per-pixel neighbor lookup: `(edge_index, neighbor_linear_idx, sign)`.
///
/// `sign` is +1 when the pixel is `pixel_a` of the edge (diff = a - b),
/// and -1 when the pixel is `pixel_b` (diff = b - a).
fn build_pixel_neighbors(edges: &[Edge], n_pixels: usize) -> Vec<Vec<(usize, usize, f64)>> {
    let mut neighbors = vec![Vec::new(); n_pixels];
    for (e_idx, edge) in edges.iter().enumerate() {
        neighbors[edge.pixel_a].push((e_idx, edge.pixel_b, 1.0));
        neighbors[edge.pixel_b].push((e_idx, edge.pixel_a, -1.0));
    }
    neighbors
}

/// Soft-thresholding operator: `sign(v) * max(|v| - κ, 0)`.
fn soft_threshold(v: f64, kappa: f64) -> f64 {
    v.signum() * (v.abs() - kappa).max(0.0)
}

/// Spatial mapping with TV-ADMM regularization.
///
/// Produces spatially smooth density maps that preserve material boundaries.
/// For each isotope independently, minimizes data-fidelity + TV penalty.
///
/// # Arguments
/// * `transmission` — 3D array (n_energies, height, width).
/// * `uncertainty` — 3D array (n_energies, height, width).
/// * `config` — Base fit configuration (shared across all pixels).
/// * `tv_config` — TV-ADMM parameters (lambda per isotope, rho, iterations).
/// * `dead_pixels` — Optional dead pixel mask.
/// * `cancel` — Optional cancellation token.
/// * `progress` — Optional pixel completion counter.
///
/// # Errors
/// Returns `PipelineError::InvalidParameter` if configuration is inconsistent,
/// `PipelineError::Cancelled` if cancelled.
pub fn spatial_map_tv(
    transmission: ArrayView3<'_, f64>,
    uncertainty: ArrayView3<'_, f64>,
    config: &FitConfig,
    tv_config: &TvAdmmConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
    progress: Option<&AtomicUsize>,
) -> Result<SpatialResult, PipelineError> {
    let shape = transmission.shape();
    let (n_energies, height, width) = (shape[0], shape[1], shape[2]);
    let n_isotopes = config.resonance_data().len();
    let n_pixels = height * width;

    // ---- Validate ----
    if tv_config.lambda.len() != n_isotopes {
        return Err(PipelineError::InvalidParameter(format!(
            "tv_config.lambda length ({}) != n_isotopes ({})",
            tv_config.lambda.len(),
            n_isotopes,
        )));
    }
    if !tv_config.rho.is_finite() || tv_config.rho <= 0.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "tv_config.rho must be > 0, got {}",
            tv_config.rho,
        )));
    }
    for (k, &lam) in tv_config.lambda.iter().enumerate() {
        if !lam.is_finite() || lam < 0.0 {
            return Err(PipelineError::InvalidParameter(format!(
                "tv_config.lambda[{k}] must be >= 0, got {lam}",
            )));
        }
    }
    if !tv_config.tol_primal.is_finite() || tv_config.tol_primal < 0.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "tv_config.tol_primal must be >= 0 and finite, got {}",
            tv_config.tol_primal,
        )));
    }
    if !tv_config.tol_dual.is_finite() || tv_config.tol_dual < 0.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "tv_config.tol_dual must be >= 0 and finite, got {}",
            tv_config.tol_dual,
        )));
    }
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

    // ---- Build edge graph ----
    let edges = build_edge_list(height, width, dead_pixels);
    let pixel_neighbors = build_pixel_neighbors(&edges, n_pixels);
    let n_edges = edges.len();

    // Determine which isotopes are free (get TV regularization).
    let free_isotopes: Vec<bool> = (0..n_isotopes)
        .map(|k| {
            config
                .constraints()
                .is_none_or(|c| c.densities[k] == ParameterRole::Free)
        })
        .collect();

    // Short-circuit: if no isotope has both free status and non-zero lambda,
    // the ADMM loop would do nothing. Return vanilla spatial_map directly.
    let has_active_tv = free_isotopes
        .iter()
        .zip(tv_config.lambda.iter())
        .any(|(&free, &lam)| free && lam > 0.0);
    if !has_active_tv {
        return spatial_map(
            transmission,
            uncertainty,
            config,
            dead_pixels,
            cancel,
            progress,
        );
    }

    // Precompute cross-sections once — shared by initial spatial_map and
    // all ADMM iterations, avoiding redundant broadening computation.
    let fast_config = precompute_config(config, cancel)?;

    // ---- Iteration 0: vanilla spatial_map ----
    // Pass the progress counter so the initial pass updates the GUI progress bar.
    let initial_result = spatial_map(
        transmission,
        uncertainty,
        &fast_config,
        dead_pixels,
        cancel,
        progress,
    )?;

    // Transpose data once for per-pixel access.
    let trans_t = transmission
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();
    let unc_t = uncertainty
        .permuted_axes([1, 2, 0])
        .as_standard_layout()
        .into_owned();

    // Extract per-pixel densities from initial result: n[pixel][isotope].
    let mut densities: Vec<Vec<f64>> = vec![vec![0.0; n_isotopes]; n_pixels];
    for (k, map) in initial_result.density_maps.iter().enumerate() {
        for y in 0..height {
            for x in 0..width {
                densities[y * width + x][k] = map[[y, x]];
            }
        }
    }

    // ---- Initialize ADMM dual variables ----
    // z[isotope][edge] — auxiliary variable.
    // u[isotope][edge] — scaled dual variable.
    let mut z = vec![vec![0.0f64; n_edges]; n_isotopes];
    let mut u = vec![vec![0.0f64; n_edges]; n_isotopes];

    // Collect live pixel indices.
    let live_pixels: Vec<usize> = (0..n_pixels)
        .filter(|&p| {
            let y = p / width;
            let x = p % width;
            !dead_pixels.is_some_and(|m| m[[y, x]])
        })
        .collect();

    // Short-circuit when all pixels are dead — the initial spatial_map already
    // returned an empty result, so just pass it through.
    if live_pixels.is_empty() {
        return Ok(initial_result);
    }

    // Track per-pixel convergence from the ADMM inner loop.
    // Updated each outer iteration; the last iteration's values are used for
    // the convergence map.  The final evaluation pass uses max_iter=0 (no
    // solver iteration), which always reports converged=false — so we must
    // track convergence from the inner loop where actual fitting happens.
    let mut pixel_converged = vec![false; n_pixels];

    // Track ADMM convergence metrics for reporting.
    let mut admm_outer_iterations = 0usize;
    let mut last_primal_norm = f64::INFINITY;
    let mut last_dual_norm = f64::INFINITY;
    let mut admm_converged = false;

    // ---- ADMM outer loop ----
    for _outer in 0..tv_config.max_outer_iter {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(PipelineError::Cancelled);
        }
        admm_outer_iterations += 1;

        // Snapshot densities before n-update for stabilization check.
        let prev_densities: Vec<Vec<f64>> = densities.clone();

        // ---- n-update: per-pixel fit with proximal penalty ----
        //
        // For each pixel p and free isotope k, the proximal target is:
        //   target_k(p) = (1/deg(p)) * Σ_{q ∈ N(p)} [n_k(q) + z_{pq} - u_{pq}]
        // where z_{pq} uses the sign convention matching the edge direction.
        //
        // The proximal penalty is:
        //   (rho * deg(p) / 2) * (n_k - target_k(p))²

        // Tuple: (pixel_index, densities, inner_converged, had_error).
        let pixel_results: Vec<(usize, Vec<f64>, bool, bool)> = live_pixels
            .par_iter()
            .filter_map(|&p| {
                if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                    return None;
                }

                let y = p / width;
                let x = p % width;
                let neighbors = &pixel_neighbors[p];
                let degree = neighbors.len();
                if degree == 0 {
                    // Isolated pixel (all neighbors dead): no regularization.
                    // Preserve convergence status from the initial spatial_map.
                    return Some((p, densities[p].clone(), pixel_converged[p], false));
                }

                // Build proximal penalty for this pixel.
                let mut penalty_terms = Vec::new();
                for k in 0..n_isotopes {
                    if !free_isotopes[k] || tv_config.lambda[k] == 0.0 {
                        continue;
                    }

                    // Compute target from neighbors.
                    let mut target_sum = 0.0;
                    for &(e_idx, neighbor_idx, sign) in neighbors {
                        // When sign=+1: pixel is A, diff = n_A - n_B
                        //   target contribution = n_B + z_e - u_e
                        // When sign=-1: pixel is B, diff = n_B - n_A
                        //   target contribution = n_A - z_e + u_e
                        target_sum +=
                            densities[neighbor_idx][k] + sign * (z[k][e_idx] - u[k][e_idx]);
                    }
                    let target = target_sum / degree as f64;
                    let rho_eff = tv_config.rho * degree as f64;

                    // param_index for isotope k is k (densities are params 0..n_isotopes).
                    penalty_terms.push((k, target, rho_eff));
                }

                let penalty = ProximalPenalty {
                    terms: penalty_terms,
                };

                // Build per-pixel FitConfig with warm-start + proximal.
                let mut pixel_config = fast_config.clone().with_proximal_penalty(penalty);
                pixel_config.set_initial_densities(densities[p].clone());

                // Extract per-pixel spectrum.
                let t_spectrum: Vec<f64> = trans_t.slice(s![y, x, ..]).to_vec();
                let sigma: Vec<f64> = unc_t
                    .slice(s![y, x, ..])
                    .iter()
                    .map(|&u| u.max(1e-10))
                    .collect();

                match fit_spectrum(&t_spectrum, &sigma, &pixel_config) {
                    Ok(result) => {
                        if let Some(prog) = progress {
                            prog.fetch_add(1, Ordering::Relaxed);
                        }
                        Some((p, result.densities, result.converged, false))
                    }
                    Err(_) => {
                        if let Some(prog) = progress {
                            prog.fetch_add(1, Ordering::Relaxed);
                        }
                        // Keep previous densities on failure.
                        Some((p, densities[p].clone(), false, true))
                    }
                }
            })
            .collect();

        // Update densities and convergence from pixel results.
        // ADMM-level density stabilization: if densities stopped changing
        // between iterations, the pixel is converged even if the inner
        // solver didn't declare it (common with warm-started Poisson FD).
        // Pixels with inner solver errors are excluded from stabilization
        // to prevent falsely marking them converged via unchanged densities.
        for (p, new_densities, inner_converged, had_error) in &pixel_results {
            let density_stable = _outer > 0
                && !had_error
                && new_densities
                    .iter()
                    .zip(prev_densities[*p].iter())
                    .all(|(new, old)| {
                        (new - old).abs() <= DENSITY_STABILIZATION_TOL * (old.abs() + 1e-30)
                    });
            densities[*p] = new_densities.clone();
            pixel_converged[*p] = *inner_converged || density_stable;
        }

        // ---- z-update: soft-thresholding per edge per isotope ----
        let z_prev = z.clone();
        for k in 0..n_isotopes {
            if !free_isotopes[k] || tv_config.lambda[k] == 0.0 {
                continue;
            }
            let kappa = tv_config.lambda[k] / tv_config.rho;
            for (e_idx, edge) in edges.iter().enumerate() {
                let diff = densities[edge.pixel_a][k] - densities[edge.pixel_b][k];
                z[k][e_idx] = soft_threshold(diff + u[k][e_idx], kappa);
            }
        }

        // ---- u-update: dual ascent per edge per isotope ----
        for k in 0..n_isotopes {
            if !free_isotopes[k] || tv_config.lambda[k] == 0.0 {
                continue;
            }
            for (e_idx, edge) in edges.iter().enumerate() {
                let diff = densities[edge.pixel_a][k] - densities[edge.pixel_b][k];
                u[k][e_idx] += diff - z[k][e_idx];
            }
        }

        // ---- Convergence check ----
        let mut primal_sq = 0.0;
        let mut dual_sq = 0.0;
        for k in 0..n_isotopes {
            if !free_isotopes[k] || tv_config.lambda[k] == 0.0 {
                continue;
            }
            for (e_idx, edge) in edges.iter().enumerate() {
                let diff = densities[edge.pixel_a][k] - densities[edge.pixel_b][k];
                let r = diff - z[k][e_idx]; // primal residual
                primal_sq += r * r;
                let s = tv_config.rho * (z[k][e_idx] - z_prev[k][e_idx]); // dual residual
                dual_sq += s * s;
            }
        }
        last_primal_norm = primal_sq.sqrt();
        last_dual_norm = dual_sq.sqrt();

        if last_primal_norm < tv_config.tol_primal && last_dual_norm < tv_config.tol_dual {
            admm_converged = true;
            break;
        }
    }

    // ---- Final pass: evaluate ancillary maps at ADMM densities ----
    // For each pixel, run fit_spectrum with ADMM density as warm-start
    // and max_iter=0 so the solver evaluates (but doesn't re-optimize).
    // This gives chi²/uncertainty/convergence AT the regularized solution,
    // not at the unconstrained per-pixel optimum.
    let trans_t_ref = &trans_t;
    let unc_t_ref = &unc_t;
    let densities_ref = &densities;

    // Build eval-only config: max_iter=0 prevents both LM and Poisson
    // from moving away from ADMM densities. compute_covariance=true for
    // uncertainty maps (LM path only; Poisson always returns None).
    let mut eval_lm_config = fast_config.lm_config().clone();
    eval_lm_config.max_iter = 0;
    let eval_config = fast_config
        .clone()
        .with_compute_covariance(true)
        .with_lm_config(eval_lm_config);
    // Also zero out Poisson max_iter when the solver is PoissonKL.
    let eval_config = if let SolverChoice::PoissonKL(ref pc) = *eval_config.solver() {
        let mut eval_pc = pc.clone();
        eval_pc.max_iter = 0;
        eval_config.with_solver(SolverChoice::PoissonKL(eval_pc))
    } else {
        eval_config
    };

    let final_results: Vec<(usize, crate::pipeline::SpectrumFitResult)> = live_pixels
        .par_iter()
        .filter_map(|&p| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return None;
            }
            let y = p / width;
            let x = p % width;

            let mut pixel_config = eval_config.clone();
            pixel_config.set_initial_densities(densities_ref[p].clone());

            let t_spectrum: Vec<f64> = trans_t_ref.slice(s![y, x, ..]).to_vec();
            let sigma: Vec<f64> = unc_t_ref
                .slice(s![y, x, ..])
                .iter()
                .map(|&u| u.max(1e-10))
                .collect();

            match fit_spectrum(&t_spectrum, &sigma, &pixel_config) {
                Ok(result) => Some((p, result)),
                Err(_) => None,
            }
        })
        .collect();

    // Assemble final SpatialResult with ADMM densities + fresh ancillary maps.
    let mut density_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::zeros((height, width)))
        .collect();
    let mut uncertainty_maps: Vec<Array2<f64>> = (0..n_isotopes)
        .map(|_| Array2::from_elem((height, width), f64::NAN))
        .collect();
    let mut chi_squared_map = Array2::from_elem((height, width), f64::NAN);
    // Use per-pixel LM convergence from the final evaluation, not the global
    // ADMM convergence flag.  This gives users the same granularity as the
    // vanilla spatial_map path and avoids the all-or-nothing convergence report.
    let mut converged_map = Array2::from_elem((height, width), false);
    let mut temperature_map: Option<Array2<f64>> = if config.fit_temperature() {
        Some(Array2::from_elem((height, width), f64::NAN))
    } else {
        None
    };
    let isotope_labels = config.isotope_names().to_vec();

    // Fill ADMM densities for all live pixels.
    for &p in &live_pixels {
        let y = p / width;
        let x = p % width;
        for k in 0..n_isotopes {
            density_maps[k][[y, x]] = densities[p][k];
        }
    }

    // Fill convergence from the last ADMM inner iteration (not the final
    // evaluation pass, which uses max_iter=0 and always reports false).
    let mut n_converged = 0usize;
    for &p in &live_pixels {
        let y = p / width;
        let x = p % width;
        converged_map[[y, x]] = pixel_converged[p];
        if pixel_converged[p] {
            n_converged += 1;
        }
    }

    // Fill ancillary maps (chi², uncertainties, temperature) from final
    // evaluation results.
    for (p, result) in &final_results {
        let y = p / width;
        let x = p % width;
        if let Some(ref unc) = result.uncertainties {
            for k in 0..n_isotopes {
                uncertainty_maps[k][[y, x]] = unc[k];
            }
        }
        chi_squared_map[[y, x]] = result.reduced_chi_squared;
        if let (Some(t_map), Some(temp)) = (&mut temperature_map, result.temperature_k) {
            t_map[[y, x]] = temp;
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
        n_total: live_pixels.len(),
        admm_info: Some(AdmmConvergenceInfo {
            outer_iterations: admm_outer_iterations,
            primal_residual: last_primal_norm,
            dual_residual: last_dual_norm,
            admm_converged,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-12);
        assert!((soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-12);
        assert!((soft_threshold(1.0, 2.0)).abs() < 1e-12);
        assert!((soft_threshold(0.0, 5.0)).abs() < 1e-12);
        assert!((soft_threshold(2.0, 2.0)).abs() < 1e-12); // exactly at threshold
    }

    #[test]
    fn test_edge_list_no_dead() {
        // 3x3 grid → 12 edges (6 horizontal + 6 vertical).
        let edges = build_edge_list(3, 3, None);
        assert_eq!(edges.len(), 12);
    }

    #[test]
    fn test_edge_list_with_dead() {
        // 3x3 grid, center pixel dead → 8 edges (center has 4 incident edges removed).
        let mut dead = Array2::from_elem((3, 3), false);
        dead[[1, 1]] = true;
        let edges = build_edge_list(3, 3, Some(&dead));
        assert_eq!(edges.len(), 8);
    }

    #[test]
    fn test_pixel_neighbors() {
        // 2x2 grid, no dead pixels → 4 edges.
        let edges = build_edge_list(2, 2, None);
        assert_eq!(edges.len(), 4);

        let neighbors = build_pixel_neighbors(&edges, 4);
        // Corner pixel (0,0) has 2 neighbors: right and down.
        assert_eq!(neighbors[0].len(), 2);
        // All pixels in a 2x2 grid have exactly 2 neighbors.
        for n in &neighbors {
            assert_eq!(n.len(), 2);
        }
    }

    #[test]
    fn test_validation_lambda_mismatch() {
        // lambda length != n_isotopes should fail.
        let n_e = 10;
        let energies: Vec<f64> = (0..n_e).map(|i| 1.0 + i as f64 * 0.5).collect();
        let transmission = Array3::from_elem((n_e, 2, 2), 0.9);
        let uncertainty = Array3::from_elem((n_e, 2, 2), 0.01);

        use crate::pipeline::FitConfig;
        use nereids_fitting::lm::LmConfig;

        // Create a minimal config with 1 isotope using test helper.
        let resonance_data = vec![crate::test_helpers::w182_single_resonance()];
        let config = FitConfig::new(
            energies,
            resonance_data,
            vec!["W-182".to_string()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let tv_config = TvAdmmConfig {
            lambda: vec![0.1, 0.2], // 2 lambdas, but only 1 isotope
            ..TvAdmmConfig::default()
        };

        let result = spatial_map_tv(
            transmission.view(),
            uncertainty.view(),
            &config,
            &tv_config,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_negative_rho() {
        let n_e = 10;
        let energies: Vec<f64> = (0..n_e).map(|i| 1.0 + i as f64 * 0.5).collect();
        let transmission = Array3::from_elem((n_e, 2, 2), 0.9);
        let uncertainty = Array3::from_elem((n_e, 2, 2), 0.01);

        use nereids_fitting::lm::LmConfig;

        let resonance_data = vec![crate::test_helpers::w182_single_resonance()];
        let config = FitConfig::new(
            energies,
            resonance_data,
            vec!["W-182".to_string()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let tv_config = TvAdmmConfig {
            lambda: vec![0.1],
            rho: -1.0,
            ..TvAdmmConfig::default()
        };

        let result = spatial_map_tv(
            transmission.view(),
            uncertainty.view(),
            &config,
            &tv_config,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_lambda_zero_matches_vanilla() {
        // λ=0 for all isotopes → TV penalty is zero → result should match
        // vanilla spatial_map (within numerical tolerance from warm-starting).
        use crate::test_helpers::w182_single_resonance;
        use nereids_fitting::lm::LmConfig;

        let n_e = 50;
        let energies: Vec<f64> = (0..n_e).map(|i| 1.0 + i as f64 * 0.2).collect();
        let true_density = 0.001;

        // Build a uniform 3x3 transmission cube.
        let resonance_data = vec![w182_single_resonance()];
        let config = FitConfig::new(
            energies.clone(),
            resonance_data.clone(),
            vec!["W-182".to_string()],
            300.0,
            None,
            vec![true_density],
            LmConfig::default(),
        )
        .unwrap();

        // Generate synthetic transmission using the physics model.
        use nereids_physics::transmission::broadened_cross_sections;
        let xs = broadened_cross_sections(&energies, &resonance_data, 300.0, None, None).unwrap();
        let mut transmission = Array3::zeros((n_e, 3, 3));
        let uncertainty = Array3::from_elem((n_e, 3, 3), 0.01);
        for e in 0..n_e {
            let t = (-true_density * xs[0][e]).exp();
            for y in 0..3 {
                for x in 0..3 {
                    transmission[[e, y, x]] = t;
                }
            }
        }

        // Vanilla result.
        let vanilla = spatial_map(
            transmission.view(),
            uncertainty.view(),
            &config,
            None,
            None,
            None,
        )
        .unwrap();

        // TV-ADMM with λ=0.
        let tv_config = TvAdmmConfig {
            lambda: vec![0.0],
            max_outer_iter: 1,
            ..TvAdmmConfig::default()
        };
        let tv_result = spatial_map_tv(
            transmission.view(),
            uncertainty.view(),
            &config,
            &tv_config,
            None,
            None,
            None,
        )
        .unwrap();

        // λ=0 means the ADMM loop does nothing beyond the initial spatial_map.
        for y in 0..3 {
            for x in 0..3 {
                let v = vanilla.density_maps[0][[y, x]];
                let tv = tv_result.density_maps[0][[y, x]];
                assert!(
                    (v - tv).abs() < 1e-6,
                    "λ=0 should match vanilla: pixel ({y},{x}) vanilla={v}, tv={tv}"
                );
            }
        }
    }

    #[test]
    fn test_tv_denoises_uniform() {
        // Noisy uniform image: TV should reduce variance compared to vanilla.
        use crate::noise::generate_noisy_cube;
        use crate::test_helpers::w182_single_resonance;
        use nereids_fitting::lm::LmConfig;

        let n_e = 50;
        let energies: Vec<f64> = (0..n_e).map(|i| 1.0 + i as f64 * 0.2).collect();
        let true_density = 0.001;
        let height = 4;
        let width = 4;

        let resonance_data = vec![w182_single_resonance()];
        let config = FitConfig::new(
            energies.clone(),
            resonance_data.clone(),
            vec!["W-182".to_string()],
            300.0,
            None,
            vec![true_density],
            LmConfig::default(),
        )
        .unwrap();

        // Generate clean transmission spectrum (1D, uniform across all pixels).
        use nereids_physics::transmission::broadened_cross_sections;
        let xs = broadened_cross_sections(&energies, &resonance_data, 300.0, None, None).unwrap();
        let clean_spectrum: Vec<f64> = (0..n_e).map(|e| (-true_density * xs[0][e]).exp()).collect();

        // Add Poisson noise with very low flux (noisy per-pixel estimates).
        let (noisy, unc) = generate_noisy_cube(&clean_spectrum, (height, width), 10.0, 42);

        // Vanilla fit.
        let vanilla = spatial_map(noisy.view(), unc.view(), &config, None, None, None).unwrap();

        // TV-ADMM fit with very strong regularization to force smoothing.
        let tv_config = TvAdmmConfig {
            lambda: vec![100.0],
            rho: 100.0,
            max_outer_iter: 10,
            ..TvAdmmConfig::default()
        };
        let tv_result = spatial_map_tv(
            noisy.view(),
            unc.view(),
            &config,
            &tv_config,
            None,
            None,
            None,
        )
        .unwrap();

        // Compute variance of fitted densities.
        let mean_vanilla: f64 =
            vanilla.density_maps[0].iter().sum::<f64>() / (height * width) as f64;
        let var_vanilla: f64 = vanilla.density_maps[0]
            .iter()
            .map(|&v| (v - mean_vanilla).powi(2))
            .sum::<f64>()
            / (height * width) as f64;

        let mean_tv: f64 = tv_result.density_maps[0].iter().sum::<f64>() / (height * width) as f64;
        let var_tv: f64 = tv_result.density_maps[0]
            .iter()
            .map(|&v| (v - mean_tv).powi(2))
            .sum::<f64>()
            / (height * width) as f64;

        assert!(
            var_tv < var_vanilla,
            "TV should reduce variance: var_tv={var_tv:.2e}, var_vanilla={var_vanilla:.2e}"
        );
    }

    #[test]
    fn test_cancellation() {
        use crate::test_helpers::w182_single_resonance;
        use nereids_fitting::lm::LmConfig;
        use std::sync::atomic::AtomicBool;

        let n_e = 10;
        let energies: Vec<f64> = (0..n_e).map(|i| 1.0 + i as f64 * 0.5).collect();
        let transmission = Array3::from_elem((n_e, 2, 2), 0.9);
        let uncertainty = Array3::from_elem((n_e, 2, 2), 0.01);

        let resonance_data = vec![w182_single_resonance()];
        let config = FitConfig::new(
            energies,
            resonance_data,
            vec!["W-182".to_string()],
            300.0,
            None,
            vec![0.001],
            LmConfig::default(),
        )
        .unwrap();

        let tv_config = TvAdmmConfig {
            lambda: vec![0.1],
            ..TvAdmmConfig::default()
        };

        // Set cancel before starting.
        let cancel = AtomicBool::new(true);
        let result = spatial_map_tv(
            transmission.view(),
            uncertainty.view(),
            &config,
            &tv_config,
            None,
            Some(&cancel),
            None,
        );
        assert!(result.is_err());
    }
}
