//! Auxiliary energy grid construction for resolution broadening.
//!
//! SAMMY extends the energy grid before computing cross-sections and applying
//! broadening.  This module reproduces SAMMY's default grid construction:
//!
//! 1. **Boundary extension** (Eqcon/Vqcon): Extend below E_min and above
//!    E_max using spacing of the first/last 5 data points, uniform in √E
//!    (FGM Doppler convention).
//! 2. **Resonance fine-structure** (Fspken/Add_Pnts): Add dense points around
//!    narrow resonances where the existing grid has fewer than 10 points per
//!    resonance width (SAMMY default iptdop=9).
//! 3. **Intermediate points** (Eqxtra): Insert extra points between each pair.
//!    Default is 0 — none of our test cases override this.
//!
//! ## SAMMY Reference
//! - `dat/mdat4.f90` — Escale (main entry), Fspken (resonance scan),
//!   Add_Pnts (fine-structure insertion)
//! - `dat/mdata.f90` — Eqxtra (intermediate points), Eqcon/Vqcon (boundary
//!   extension)
//! - `inp/InputInfoData.cpp` — Default iptdop=9, iptwid=5, nxtra=0

use crate::resolution::ResolutionParams;
use nereids_core::constants::NEAR_ZERO_FLOOR;

/// Number of boundary data points used to compute extension spacing.
///
/// SAMMY Ref: `dat/mdat4.f90` Escale lines 56-97
const N_BOUNDARY_REF: usize = 5;

/// Relative tolerance for duplicate detection during grid merge.
/// Points closer than `tol * E` are considered duplicates.
const MERGE_RELATIVE_TOL: f64 = 1e-10;

/// SAMMY default iptdop: controls fine-structure point density.
///
/// SAMMY Ref: `inp/InputInfoData.cpp` line 23
const IPTDOP: usize = 9;

/// Minimum grid points required within one resonance width [E_res−Gd, E_res+Gd].
/// If the existing grid has fewer, fine-structure points are added.
///
/// SAMMY Ref: `dat/mdat4.f90` Fspken lines 276-279
const MIN_POINTS_PER_WIDTH: usize = IPTDOP + 1;

/// Fraction of resonance width used as fine-structure spacing.
/// `Eg = FRACTN * Gd` gives ~14 uniformly-spaced points across 2·Gd.
///
/// SAMMY Ref: `dat/mdat4.f90` Fspken line 310
const FRACTN: f64 = 2.0 / (IPTDOP as f64 + 5.0);

/// Build an extended energy grid with boundary extension and resonance
/// fine-structure for resolution broadening.
///
/// Returns `(extended_energies, data_indices)` where:
/// - `extended_energies` is sorted ascending and includes all `data_energies`
/// - `data_indices[i]` is the index of `data_energies[i]` in `extended_energies`
///
/// When `data_energies` has fewer than 2 points or `resolution` is `None`,
/// returns a copy of the data grid with identity indices.
///
/// # Arguments
/// * `data_energies` — Experimental energy grid (sorted ascending, eV).
/// * `resolution` — Resolution parameters (for computing boundary width).
/// * `resonances` — (energy_eV, gd_eV) pairs for fine-structure densification.
///   `gd = 0.001 * Σ|Γ_i|` is the resonance half-width parameter from SAMMY's
///   Fspken convention.
///
/// # SAMMY Reference
/// `dat/mdat4.f90` Escale+Fspken+Add_Pnts, `dat/mdata.f90` Vqcon
pub fn build_extended_grid(
    data_energies: &[f64],
    resolution: Option<&ResolutionParams>,
    resonances: &[(f64, f64)],
) -> (Vec<f64>, Vec<usize>) {
    build_extended_grid_inner(data_energies, resolution, resonances, true)
}

/// Build extended grid without intermediate point densification.
///
/// Used for exponential tail broadening where the trapezoidal × kernel
/// quadrature is sensitive to grid density changes.
pub fn build_extended_grid_boundary_only(
    data_energies: &[f64],
    resolution: Option<&ResolutionParams>,
) -> (Vec<f64>, Vec<usize>) {
    build_extended_grid_inner(data_energies, resolution, &[], false)
}

fn build_extended_grid_inner(
    data_energies: &[f64],
    resolution: Option<&ResolutionParams>,
    resonances: &[(f64, f64)],
    add_intermediate: bool,
) -> (Vec<f64>, Vec<usize>) {
    if data_energies.is_empty() {
        return (vec![], vec![]);
    }
    if data_energies.len() == 1 {
        return (data_energies.to_vec(), vec![0]);
    }

    let res = match resolution {
        Some(r) => r,
        None => {
            let indices: Vec<usize> = (0..data_energies.len()).collect();
            return (data_energies.to_vec(), indices);
        }
    };

    let n = data_energies.len();

    // ── Step 1: Boundary extension ──────────────────────────────────────
    // Extend by 5σ of Gaussian width at each boundary, matching SAMMY's
    // Escale lines 54-97.
    let n_sigma = 5.0;

    let e_min = data_energies[0];
    let wg_low = res.gaussian_width(e_min);
    let extend_low = n_sigma * wg_low;

    let e_max = data_energies[n - 1];
    let wg_high = res.gaussian_width(e_max);
    let we_high = res.exp_width(e_max);
    let extend_high = if we_high > 1e-30 {
        let rwid = wg_high / we_high;
        if rwid <= 1.0 {
            6.25 * we_high
        } else if rwid <= 2.0 {
            n_sigma * (3.0 - rwid) * wg_high
        } else {
            n_sigma * wg_high
        }
    } else {
        n_sigma * wg_high
    };

    let mut grid = Vec::with_capacity(n + 200);

    // Low-side extension: equally spaced in √E (SAMMY FGM convention).
    // SAMMY Ref: dat/mdata.f90 Vqcon
    if extend_low > 0.0 && e_min > 0.0 {
        let n_ref = N_BOUNDARY_REF.min(n);
        let e_ref = data_energies[n_ref - 1];
        let d_sqrt = (e_ref.sqrt() - e_min.sqrt()) / (n_ref as f64 - 1.0).max(1.0);

        if d_sqrt > 1e-30 {
            let target_low = (e_min - extend_low).max(0.001);
            let sqrt_min = e_min.sqrt();
            let sqrt_target = target_low.sqrt();
            let n_ext = ((sqrt_min - sqrt_target) / d_sqrt).ceil() as usize;
            for k in 1..=n_ext {
                let sqrt_e = sqrt_min - d_sqrt * k as f64;
                if sqrt_e > 0.0 {
                    grid.push(sqrt_e * sqrt_e);
                }
            }
        }
    }

    // Add all data points.
    grid.extend_from_slice(data_energies);

    // High-side extension: equally spaced in √E (SAMMY FGM convention).
    if extend_high > 0.0 {
        let n_ref = N_BOUNDARY_REF.min(n);
        let e_ref = data_energies[n - n_ref];
        let d_sqrt = (e_max.sqrt() - e_ref.sqrt()) / (n_ref as f64 - 1.0).max(1.0);

        if d_sqrt > 1e-30 {
            let target_high = e_max + extend_high;
            let sqrt_max = e_max.sqrt();
            let sqrt_target = target_high.sqrt();
            let n_ext = ((sqrt_target - sqrt_max) / d_sqrt).ceil() as usize;
            for k in 1..=n_ext {
                let sqrt_e = sqrt_max + d_sqrt * k as f64;
                grid.push(sqrt_e * sqrt_e);
            }
        }
    }

    // Sort and deduplicate before inserting intermediate points.
    grid.sort_unstable_by(|a, b| a.total_cmp(b));
    dedup(&mut grid);

    // ── Step 2: Adaptive intermediate points ────────────────────────────
    // Insert intermediate points where the grid spacing exceeds a fraction
    // of the local resolution width.  This ensures the resolution broadening
    // integral has enough quadrature points even on coarse grids.
    //
    // Target: spacing ≤ W/4 (at least ~20 points per 5σ window).
    // SAMMY analogue: dat/mdata.f90 Eqxtra (with nxtra=0 default, but
    // SAMMY's fine-structure + Xcoef quadrature compensates).
    //
    // Skipped for exponential tail broadening where the trapezoidal × kernel
    // approach is sensitive to grid density changes (build_extended_grid_boundary_only).
    if add_intermediate {
        let mut extra: Vec<f64> = Vec::new();
        for k in 0..grid.len() - 1 {
            let e_lo = grid[k];
            let e_hi = grid[k + 1];
            let h = e_hi - e_lo;
            let e_mid = (e_lo + e_hi) * 0.5;
            let w = res.gaussian_width(e_mid);
            if w < NEAR_ZERO_FLOOR {
                continue;
            }
            let max_spacing = w * 0.25;
            if h > max_spacing {
                // Insert enough uniformly-spaced points.
                let n_ins = (h / max_spacing).ceil() as usize;
                let step = h / n_ins as f64;
                for j in 1..n_ins {
                    extra.push(e_lo + step * j as f64);
                }
            }
        }
        if !extra.is_empty() {
            grid.extend(extra);
            grid.sort_unstable_by(|a, b| a.total_cmp(b));
            dedup(&mut grid);
        }
    }

    // ── Step 3: Resonance fine-structure (Fspken) ───────────────────────
    // For each resonance within the grid range, check if the grid has at
    // least MIN_POINTS_PER_WIDTH points across [E_res-Gd, E_res+Gd].
    // If not, add uniformly-spaced points with spacing Eg = FRACTN * Gd,
    // plus exponentially-graded tail/transition points.
    // SAMMY Ref: dat/mdat4.f90 Fspken lines 243-284, Add_Pnts lines 333-532
    if !resonances.is_empty() {
        let mut fine_pts: Vec<f64> = Vec::new();
        for &(eres, gd) in resonances {
            let pts = fine_structure_points(&grid, eres, gd);
            fine_pts.extend(pts);
        }
        if !fine_pts.is_empty() {
            grid.extend(fine_pts);
            grid.sort_unstable_by(|a, b| a.total_cmp(b));
            dedup(&mut grid);
        }
    }

    // Filter to positive energies.
    grid.retain(|&e| e > 0.0);

    // Build data_indices.
    let data_indices = build_data_indices(&grid, data_energies);

    (grid, data_indices)
}

/// Generate fine-structure points around a single resonance.
///
/// SAMMY's `Fspken` identifies resonances where the existing grid has fewer
/// than `IPTDOP+1` (=10) points within [E_res−Gd, E_res+Gd].  For each such
/// resonance, `Add_Pnts` inserts:
/// - Uniform points across [E_res−Gd, E_res+Gd] with spacing `Eg = FRACTN * Gd`
/// - Exponentially graded transition points beyond ±Gd (spacing doubles each step)
///   up to ±3·Gd, preventing abrupt density jumps at the fine-structure boundary.
///
/// SAMMY Ref: `dat/mdat4.f90` Fspken lines 243-284, Add_Pnts lines 333-532,
///            DgradV/UgradV (graded transition)
fn fine_structure_points(grid: &[f64], eres: f64, gd: f64) -> Vec<f64> {
    if gd < 1e-30 || eres <= 0.0 {
        return vec![];
    }

    let xmin = (eres - gd).max(1e-6);
    let xmax = eres + gd;

    // Skip resonances outside the grid range.
    // SAMMY Ref: Fspken line 253: `IF (eres.LT.el_energb .OR. eres.GT.eh_energb) cycle`
    if grid.is_empty() || eres < grid[0] || eres > *grid.last().unwrap() {
        return vec![];
    }

    // Count existing grid points in [xmin, xmax].
    // SAMMY Ref: Fspken lines 269-279 (Pointr + K+iptdop+1 check)
    let lo = grid.partition_point(|&e| e < xmin);
    let hi = grid.partition_point(|&e| e <= xmax);
    let count = hi - lo;

    if count >= MIN_POINTS_PER_WIDTH {
        return vec![];
    }

    let eg = FRACTN * gd;
    if eg < 1e-30 {
        return vec![];
    }

    let mut new_points = Vec::new();

    // Uniform points across [xmin, xmax] with spacing eg.
    // SAMMY Ref: Add_Pnts — uniform fill within resonance width
    let n_pts = ((xmax - xmin) / eg).ceil() as usize;
    for i in 0..=n_pts {
        let e = xmin + eg * i as f64;
        if e > 0.0 && e <= xmax + eg * 0.01 {
            new_points.push(e);
        }
    }

    // Exponentially graded transition points beyond ±Gd.
    // Bridge from fine-structure spacing to the surrounding grid spacing
    // with doubling steps, preventing the abrupt spacing jumps that cause
    // Xcoef quadrature weight instability.
    // SAMMY Ref: Add_Pnts — DgradV/UgradV calls at lines 551-553, 659-661

    // Down-side: bridge from xmin to the nearest grid point below.
    let idx_below = lo; // lo is the first grid index >= xmin
    if idx_below > 0 {
        let e_below = grid[idx_below - 1];
        let gap = xmin - e_below;
        if gap > eg * 2.0 {
            let mut spacing = eg;
            let mut e = xmin;
            for _ in 0..20 {
                spacing *= 2.0;
                e -= spacing;
                if e <= e_below + MERGE_RELATIVE_TOL * e_below.abs().max(1e-30) {
                    break;
                }
                new_points.push(e);
            }
        }
    }

    // Up-side: bridge from xmax to the nearest grid point above.
    if hi < grid.len() {
        let e_above = grid[hi];
        let gap = e_above - xmax;
        if gap > eg * 2.0 {
            let mut spacing = eg;
            let mut e = xmax;
            for _ in 0..20 {
                spacing *= 2.0;
                e += spacing;
                if e >= e_above - MERGE_RELATIVE_TOL * e_above.abs().max(1e-30) {
                    break;
                }
                new_points.push(e);
            }
        }
    }

    new_points
}

/// Sort and deduplicate within tolerance.
fn dedup(grid: &mut Vec<f64>) {
    if grid.len() < 2 {
        return;
    }
    let mut deduped = Vec::with_capacity(grid.len());
    deduped.push(grid[0]);
    for &val in grid.iter().skip(1) {
        let prev = *deduped.last().unwrap();
        let tol = MERGE_RELATIVE_TOL * prev.abs().max(1e-30);
        if (val - prev).abs() > tol {
            deduped.push(val);
        }
    }
    *grid = deduped;
}

/// Build mapping from data energies to their indices in the extended grid.
///
/// Each data energy must appear exactly in the grid (guaranteed by
/// construction — data points are always included and never dropped by dedup).
///
/// Uses binary search for O(N log M) where N = data points, M = grid size.
fn build_data_indices(grid: &[f64], data_energies: &[f64]) -> Vec<usize> {
    data_energies
        .iter()
        .map(|&e| {
            let idx = grid.partition_point(|&ae| ae < e);
            // Search nearby for exact match (floating-point tolerance).
            let search_range = idx.saturating_sub(1)..grid.len().min(idx + 2);
            let mut best_idx = idx.min(grid.len() - 1);
            let mut best_dist = (grid[best_idx] - e).abs();
            for j in search_range {
                let dist = (grid[j] - e).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            best_idx
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_grid() {
        let (ext, indices) = build_extended_grid(&[], None, &[]);
        assert!(ext.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn test_single_point() {
        let energies = vec![100.0];
        let (ext, indices) = build_extended_grid(&energies, None, &[]);
        assert_eq!(ext, vec![100.0]);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_no_resolution_identity() {
        let data = vec![1.0, 5.0, 10.0];
        let (ext, indices) = build_extended_grid(&data, None, &[]);
        assert_eq!(ext, data);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_data_indices_roundtrip() {
        let data = vec![1.0, 5.0, 10.0, 100.0];
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let (ext, indices) = build_extended_grid(&data, Some(&res), &[]);
        assert!(ext.len() >= data.len());
        for (i, &e) in data.iter().enumerate() {
            assert!(
                (ext[indices[i]] - e).abs() < 1e-10,
                "data[{i}]={e} not at ext[{}]={}",
                indices[i],
                ext[indices[i]]
            );
        }
    }

    #[test]
    fn test_extension_covers_5sigma() {
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 5.0).collect();
        let res = ResolutionParams::new(10.0, 0.1, 0.01, 0.0).unwrap();
        let (ext, _) = build_extended_grid(&data, Some(&res), &[]);

        assert!(
            ext[0] < data[0],
            "expected extension below data[0]={}, got ext[0]={}",
            data[0],
            ext[0]
        );
        assert!(
            *ext.last().unwrap() > *data.last().unwrap(),
            "expected extension above data max"
        );
    }

    #[test]
    fn test_grid_is_sorted() {
        let data: Vec<f64> = (0..10).map(|i| 1000.0 + i as f64 * 100.0).collect();
        let res = ResolutionParams::new(50.0, 0.05, 0.01, 0.0).unwrap();
        let (ext, _) = build_extended_grid(&data, Some(&res), &[]);
        for pair in ext.windows(2) {
            assert!(
                pair[0] < pair[1],
                "grid not sorted: {} >= {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn test_grid_all_positive() {
        let data = vec![1.0, 2.0, 3.0];
        let res = ResolutionParams::new(10.0, 0.1, 0.01, 0.0).unwrap();
        let (ext, _) = build_extended_grid(&data, Some(&res), &[]);
        for &e in &ext {
            assert!(e > 0.0, "non-positive energy: {e}");
        }
    }

    #[test]
    fn test_fine_structure_adds_points() {
        // Test fine-structure in isolation (no intermediate points) by calling
        // build_extended_grid_inner directly.
        // Sparse grid with a narrow resonance at 500 eV, Gd = 1 eV.
        // Grid has ~5 eV spacing → only ~0-1 point in [499, 501].
        let data: Vec<f64> = (0..20).map(|i| 490.0 + i as f64 * 5.0).collect();
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let resonances = vec![(500.0, 1.0)]; // E_res=500 eV, Gd=1 eV

        // Without fine-structure, boundary-only:
        let (ext_without, _) = build_extended_grid_inner(&data, Some(&res), &[], false);
        // With fine-structure, still no intermediates:
        let (ext_with, _) = build_extended_grid_inner(&data, Some(&res), &resonances, false);

        assert!(
            ext_with.len() > ext_without.len(),
            "fine-structure should add points: {} vs {}",
            ext_with.len(),
            ext_without.len()
        );

        // Check that there are now ≥10 points in [499, 501].
        let lo = ext_with.partition_point(|&e| e < 499.0);
        let hi = ext_with.partition_point(|&e| e <= 501.0);
        assert!(
            hi - lo >= MIN_POINTS_PER_WIDTH,
            "expected ≥{MIN_POINTS_PER_WIDTH} points in resonance width, got {}",
            hi - lo
        );
    }

    #[test]
    fn test_fine_structure_skips_dense_grid() {
        // Dense grid: 0.1 eV spacing around a resonance with Gd=1.0 eV.
        // Already has ~20 points in [499, 501] → no fine-structure needed.
        let data: Vec<f64> = (0..100).map(|i| 495.0 + i as f64 * 0.1).collect();
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let resonances = vec![(500.0, 1.0)];

        let (ext_without, _) = build_extended_grid(&data, Some(&res), &[]);
        let (ext_with, _) = build_extended_grid(&data, Some(&res), &resonances);

        assert_eq!(
            ext_without.len(),
            ext_with.len(),
            "dense grid should not get extra fine-structure points"
        );
    }

    #[test]
    fn test_fine_structure_data_indices_valid() {
        // Verify data points are still found correctly after fine-structure insertion.
        let data: Vec<f64> = (0..20).map(|i| 490.0 + i as f64 * 5.0).collect();
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let resonances = vec![(500.0, 1.0), (520.0, 0.5)];

        let (ext, indices) = build_extended_grid(&data, Some(&res), &resonances);
        assert_eq!(indices.len(), data.len());
        for (i, &e) in data.iter().enumerate() {
            assert!(
                (ext[indices[i]] - e).abs() < 1e-10,
                "data[{i}]={e} not at ext[{}]={}",
                indices[i],
                ext[indices[i]]
            );
        }
    }

    #[test]
    fn test_fine_structure_outside_range_ignored() {
        // Resonance outside data range should not add points.
        let data: Vec<f64> = (0..10).map(|i| 100.0 + i as f64 * 10.0).collect();
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let resonances = vec![(50.0, 1.0), (300.0, 1.0)]; // Both outside [100, 190]

        let (ext_without, _) = build_extended_grid(&data, Some(&res), &[]);
        let (ext_with, _) = build_extended_grid(&data, Some(&res), &resonances);

        // May differ slightly due to boundary extension, but the resonance
        // outside the extended range should not add fine-structure.
        // Just verify the grid is valid.
        assert!(ext_with.len() >= data.len());
        for pair in ext_with.windows(2) {
            assert!(
                pair[0] < pair[1],
                "grid not sorted: {} >= {}",
                pair[0],
                pair[1]
            );
        }
        // Both resonances are far outside data range, should have same grid.
        assert_eq!(ext_without.len(), ext_with.len());
    }
}
