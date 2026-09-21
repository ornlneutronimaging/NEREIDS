//! Auxiliary energy grid construction for resolution broadening.
//!
//! SAMMY extends the energy grid before computing cross-sections and applying
//! broadening.  This module reproduces SAMMY's default grid construction:
//!
//! 1. **Boundary extension** (Eqcon/Vqcon): Extend below E_min and above
//!    E_max using spacing of the first/last 5 data points, uniform in √E
//!    (FGM Doppler convention) for the Gaussian family and uniform in time
//!    of flight for kernels tabulated there.
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

use crate::resolution::{ResolutionFunction, ResolutionParams};
use nereids_core::constants::NEAR_ZERO_FLOOR;

/// Number of boundary data points used to compute extension spacing.
///
/// SAMMY Ref: `dat/mdat4.f90` Escale lines 56-97 for the spacing; the
/// amounts come from `rsl/mrsl4.f90` Wdsint
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

/// Build extended grid with boundary extension only (no intermediate points).
///
/// Used when the combined Gaussian+exponential kernel is active, where
/// non-uniform spacing from adaptive intermediates degrades the Xcoef
/// quadrature accuracy.
pub fn build_extended_grid_boundary_only(
    data_energies: &[f64],
    resolution: Option<&ResolutionParams>,
) -> (Vec<f64>, Vec<usize>) {
    build_extended_grid_inner(data_energies, resolution, &[], false)
}

/// Extend a data grid past both ends by the reach of any resolution family.
/// Boundary extension only: the intermediate points and resonance fine
/// structure of [`build_extended_grid`] are built for the Gaussian family alone.
pub fn build_extended_grid_for(
    data_energies: &[f64],
    resolution: &ResolutionFunction,
) -> (Vec<f64>, Vec<usize>) {
    if data_energies.len() < 2 {
        let indices: Vec<usize> = (0..data_energies.len()).collect();
        return (data_energies.to_vec(), indices);
    }
    let (low, high) = resolution.grid_bounds_ev(data_energies);
    let spacing = match resolution {
        // The Gaussian's width is an energy.
        ResolutionFunction::Gaussian(_) => Spacing::SqrtEnergy,
        // These kernels are tabulated in time of flight.
        ResolutionFunction::Tabulated(_) | ResolutionFunction::IkedaCarpenter(_) => {
            Spacing::TimeOfFlight
        }
    };
    extend_boundaries(data_energies, low, high, spacing)
}

/// The variable in which the added boundary points are evenly spaced, at the
/// average spacing of the five data points nearest the edge.
///
/// SAMMY Ref: `dat/mdat4.f90` Escale, `sqrt(E)` for free-gas Doppler
/// (`dat/mdata.f90` Vqcon); time of flight is `1/sqrt(E)` up to the flight path.
#[derive(Clone, Copy)]
enum Spacing {
    SqrtEnergy,
    TimeOfFlight,
}

impl Spacing {
    fn to_u(self, e: f64) -> f64 {
        match self {
            Spacing::SqrtEnergy => e.sqrt(),
            Spacing::TimeOfFlight => 1.0 / e.sqrt(),
        }
    }

    fn to_e(self, u: f64) -> f64 {
        match self {
            Spacing::SqrtEnergy => u * u,
            Spacing::TimeOfFlight => 1.0 / (u * u),
        }
    }
}

/// The points stepping outward from `e_edge` to `target_e`, evenly spaced in
/// `spacing` at the average spacing of the `n_ref` data points from `e_edge`
/// to `e_ref`, ending at `target_e` itself; a lattice point within the merge
/// tolerance of the target is left out.
fn step_outward(
    spacing: Spacing,
    e_edge: f64,
    e_ref: f64,
    n_ref: usize,
    target_e: f64,
) -> Vec<f64> {
    let u_edge = spacing.to_u(e_edge);
    let u_target = spacing.to_u(target_e);
    let step = (u_edge - spacing.to_u(e_ref)) / (n_ref as f64 - 1.0).max(1.0);
    let steps = (u_target - u_edge) / step;
    if step.abs() <= 1e-30 || !steps.is_finite() || steps <= 0.0 {
        return Vec::new();
    }
    let n_between = (steps - MERGE_RELATIVE_TOL * (u_target / step).abs()).floor() as usize;
    let mut points: Vec<f64> = (1..=n_between)
        .map(|k| spacing.to_e(u_edge + step * k as f64))
        .collect();
    points.push(target_e);
    points
}

/// Extend a grid so it spans `[low, high]`, at the edge spacing of the data
/// itself, ending exactly at `low` and `high`.  The data points are carried
/// unchanged between the two extensions.
fn extend_boundaries(
    data_energies: &[f64],
    low: f64,
    high: f64,
    spacing: Spacing,
) -> (Vec<f64>, Vec<usize>) {
    let n = data_energies.len();
    let e_min = data_energies[0];
    let e_max = data_energies[n - 1];
    let n_ref = N_BOUNDARY_REF.min(n);

    let mut below = if low < e_min && e_min > 0.0 {
        step_outward(spacing, e_min, data_energies[n_ref - 1], n_ref, low)
    } else {
        Vec::new()
    };
    below.reverse();
    let above = if high > e_max {
        step_outward(spacing, e_max, data_energies[n - n_ref], n_ref, high)
    } else {
        Vec::new()
    };

    let indices = (below.len()..below.len() + n).collect();
    let mut grid = below;
    grid.extend_from_slice(data_energies);
    grid.extend(above);
    (grid, indices)
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

    // ── Step 1: Boundary extension ──────────────────────────────────────
    let (low, high) = ResolutionFunction::Gaussian(*res).grid_bounds_ev(data_energies);
    let (mut grid, _) = extend_boundaries(data_energies, low, high, Spacing::SqrtEnergy);

    // ── Step 2: Adaptive intermediate points ────────────────────────────
    // Insert intermediate points where the grid spacing exceeds a fraction
    // of the local resolution width.  This ensures the resolution broadening
    // integral has enough quadrature points even on coarse grids.
    //
    // Target: spacing ≤ W/4 (at least ~20 points per 5σ window).
    //
    // W/4 is sufficient: the PW-linear Gaussian integration is exact for
    // linear cross-section segments, so the error depends on the cross-section
    // curvature × h², not on quadrature point count.  Fine-structure points
    // (Step 3) densify around narrow resonances where curvature is high.
    //
    // SAMMY analogue: dat/mdata.f90 Eqxtra (with nxtra=0 default, but
    // SAMMY's fine-structure + Xcoef quadrature compensates).
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

    /// The grid ends exactly at the bounds it was asked to span and keeps
    /// every data point, even when a bound lies within the merge tolerance of
    /// a lattice point or of the data's own end.
    #[test]
    fn grid_ends_exactly_at_its_bounds_through_the_merge() {
        let data: Vec<f64> = (0..5).map(|i| 100.0 + f64::from(i)).collect();
        let e_min = data[0];
        let e_max = data[4];
        // A high end seven lattice steps out plus a sliver the merge cannot
        // resolve, in each spacing.
        for spacing in [Spacing::SqrtEnergy, Spacing::TimeOfFlight] {
            let u_max = spacing.to_u(e_max);
            let step = (u_max - spacing.to_u(data[0])) / 4.0;
            let high = spacing.to_e(u_max + 7.0 * step * (1.0 + 3.0e-11));
            let (grid, indices) = extend_boundaries(&data, e_min, high, spacing);
            assert_eq!(*grid.last().unwrap(), high);
            assert_eq!(grid.len(), data.len() + 7);
            assert_eq!(indices, vec![0, 1, 2, 3, 4]);

            let u_min = spacing.to_u(e_min);
            let step = (spacing.to_u(data[4]) - u_min) / 4.0;
            let low = spacing.to_e(u_min - 7.0 * step * (1.0 + 3.0e-11));
            let (grid, indices) = extend_boundaries(&data, low, e_max, spacing);
            assert_eq!(grid[0], low);
            assert_eq!(grid.len(), data.len() + 7);
            assert_eq!(indices, vec![7, 8, 9, 10, 11]);
        }
        // Bounds within the merge tolerance of the data's own ends.
        let high = e_max * (1.0 + 5.0e-11);
        let (grid, indices) = extend_boundaries(&data, e_min, high, Spacing::SqrtEnergy);
        assert_eq!(*grid.last().unwrap(), high);
        assert_eq!(grid.len(), data.len() + 1);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
        let low = e_min * (1.0 - 5.0e-11);
        let (grid, indices) = extend_boundaries(&data, low, e_max, Spacing::TimeOfFlight);
        assert_eq!(grid[0], low);
        assert_eq!(grid[1], e_min);
        assert_eq!(grid.len(), data.len() + 1);
        assert_eq!(indices, vec![1, 2, 3, 4, 5]);
    }

    /// A delayed tail that approaches the nominal flight time does not make the
    /// working grid grow without bound: the extension needs no more points than
    /// the instrument has time-of-flight channels over the same span, and it
    /// moves smoothly as the tail crosses the flight time.
    #[test]
    fn extension_stays_bounded_as_the_tail_nears_the_flight_time() {
        use crate::resolution::{TOF_FACTOR, TabulatedResolution};
        use std::sync::Arc;

        let offsets: Vec<f64> = (0..=140).map(|k| -20.0 + k as f64).collect();
        let weights = vec![1.0; offsets.len()];
        let table = TabulatedResolution::from_kernels(
            vec![100.0, 300.0],
            vec![(offsets.clone(), weights.clone()), (offsets, weights)],
            25.0,
        )
        .expect("valid two-block table");
        let resolution = ResolutionFunction::Tabulated(Arc::new(table));

        let mut previous: Option<usize> = None;
        for tenths in 2200..=2300 {
            let e_max = tenths as f64 / 10.0;
            let data: Vec<f64> = (0..400).map(|i| e_max - 40.0 + i as f64 * 0.1).collect();
            let (grid, _) = build_extended_grid_for(&data, &resolution);
            let added = grid.len() - data.len();
            let tof = |e: f64| TOF_FACTOR * 25.0 / e.sqrt();
            let e_top = data[data.len() - 1];
            let channels_above = tof(e_top) / (tof(data[data.len() - 2]) - tof(e_top));
            let channels_below = 20.0 / (tof(data[0]) - tof(data[1]));
            assert!(
                (added as f64) <= channels_above + channels_below + 2.0,
                "at e_max = {e_max} eV the extension added {added} points, more than the \
                 {channels_above:.0} + {channels_below:.0} channels the instrument has there"
            );
            if let Some(p) = previous {
                assert!(
                    added <= 2 * p + 8 && p <= 2 * added + 8,
                    "the extension jumped from {p} to {added} points between neighbouring \
                     windows ending near {e_max} eV"
                );
            }
            previous = Some(added);
        }
    }
}
