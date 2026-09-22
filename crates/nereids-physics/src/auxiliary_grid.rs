use crate::resolution::{ResolutionFunction, ResolutionParams};

/// Number of boundary data points used to compute extension spacing.
///
/// SAMMY Ref: `dat/mdat4.f90` Escale lines 56-97 for the spacing; the
/// amounts come from `rsl/mrsl4.f90` Wdsint
const N_BOUNDARY_REF: usize = 5;

/// Relative tolerance for duplicate detection during grid merge.
/// Points closer than `tol * E` are considered duplicates.
const MERGE_RELATIVE_TOL: f64 = 1e-10;

/// SAMMY default iptdop: controls resonance point density.
///
/// SAMMY Ref: `inp/InputInfoData.cpp` line 23
const IPTDOP: usize = 9;

/// Grid points within `[E_res − D, E_res + D]` that make resonance points
/// unnecessary.
///
/// SAMMY Ref: `dat/mdat4.f90` Fspken lines 276-279
const MIN_POINTS_PER_WIDTH: usize = IPTDOP + 1;

/// Largest spacing across a resonance, as a fraction of its total width.
///
/// SAMMY Ref: `dat/mdat4.f90` Fspken line 310
pub const FRACTN: f64 = 2.0 / (IPTDOP as f64 + 5.0);

/// Distance, as a fraction of the resonance spacing, within which an existing
/// point stands in for a resonance point.
///
/// SAMMY Ref: `dat/mdat4.f90` Add_Pnts lines 380, 400
const STAND_IN_FRACTION: f64 = 0.1;

/// Largest ratio between neighbouring intervals.
///
/// SAMMY Ref: `dat/mdat5.f90` RefineGrid line 264
const MAX_SPACING_RATIO: f64 = 2.5;

/// Build the working grid for broadening a spectrum measured at
/// `data_energies` with `resolution`.
///
/// Returns `(energies, data_indices)`: `energies` ascending and containing
/// every data energy unchanged, and `data_indices[i]` the index of
/// `data_energies[i]` in it.  Fewer than two data energies come back as they
/// are.
///
/// # Arguments
/// * `data_energies` — Data energies in eV, sorted ascending.
/// * `resolution` — The blur; sets how far the grid extends.
/// * `resonances` — `(E_res, D)` pairs in eV, `D` the resonance's total width.
pub fn build_working_grid(
    data_energies: &[f64],
    resolution: &ResolutionFunction,
    resonances: &[(f64, f64)],
) -> (Vec<f64>, Vec<usize>) {
    if data_energies.len() < 2 {
        return (data_energies.to_vec(), (0..data_energies.len()).collect());
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
    let (mut grid, _) = extend_boundaries(data_energies, low, high, spacing);
    if let ResolutionFunction::Gaussian(params) = resolution {
        let points = quarter_width_points(&grid, params);
        merge_points(&mut grid, points);
    }
    for &(energy, width) in resonances {
        let points = resonance_points(&grid, energy, width);
        merge_points(&mut grid, points);
    }
    grade_spacing(&mut grid);
    let data_indices = build_data_indices(&grid, data_energies);
    (grid, data_indices)
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
    let n = ((steps - MERGE_RELATIVE_TOL * (u_target / step).abs()).ceil() as usize).max(1);
    let stride = (u_target - u_edge) / n as f64;
    let mut points: Vec<f64> = (1..n)
        .map(|k| spacing.to_e(u_edge + stride * k as f64))
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

fn quarter_width_points(grid: &[f64], params: &ResolutionParams) -> Vec<f64> {
    grid.windows(2)
        .flat_map(|pair| {
            let (lo, hi) = (pair[0], pair[1]);
            let parts = ((hi - lo) / (0.25 * params.gaussian_width(0.5 * (lo + hi)))).ceil();
            let parts = if parts.is_finite() { parts as usize } else { 1 };
            (1..parts).map(move |k| lo + (hi - lo) * k as f64 / parts as f64)
        })
        .collect()
}

/// Points that sample the resonance at `energy` of total width `width`: its
/// centre and the ends of `[energy − width, energy + width]` within the grid,
/// and enough points between them and the grid's own points that no interval
/// there exceeds `FRACTN · width`.  An existing point within
/// `STAND_IN_FRACTION` of that spacing stands in for a centre or an end.
/// Returns none when the centre lies off the grid or the grid already holds
/// `MIN_POINTS_PER_WIDTH` points across the interval.
///
/// SAMMY Ref: `dat/mdat4.f90` Fspken lines 243-284, Add_Pnts lines 333-532
fn resonance_points(grid: &[f64], energy: f64, width: f64) -> Vec<f64> {
    let (first, last) = (grid[0], grid[grid.len() - 1]);
    if energy < first || energy > last {
        return Vec::new();
    }
    let lo = (energy - width).max(first);
    let hi = (energy + width).min(last);
    let start = grid.partition_point(|&e| e < lo);
    let end = grid.partition_point(|&e| e <= hi);
    if end - start >= MIN_POINTS_PER_WIDTH {
        return Vec::new();
    }
    let step = FRACTN * width;
    let neighbours = &grid[start.saturating_sub(1)..(end + 1).min(grid.len())];
    let mut anchors = grid[start..end].to_vec();
    let mut points = Vec::new();
    for e in [lo, energy, hi] {
        if neighbours
            .iter()
            .all(|&g| (g - e).abs() >= STAND_IN_FRACTION * step)
        {
            anchors.push(e);
            points.push(e);
        }
    }
    anchors.sort_by(f64::total_cmp);
    for pair in anchors.windows(2) {
        let gap = pair[1] - pair[0];
        let n = (gap / step).ceil() as usize;
        points.extend((1..n).map(|k| pair[0] + gap * k as f64 / n as f64));
    }
    points
}

fn merge_points(grid: &mut Vec<f64>, mut points: Vec<f64>) -> usize {
    points.sort_by(f64::total_cmp);
    let close = |a: f64, b: f64| (a - b).abs() <= MERGE_RELATIVE_TOL * a.abs().max(b.abs());
    let mut kept: Vec<f64> = Vec::with_capacity(points.len());
    for p in points {
        let i = grid.partition_point(|&g| g < p);
        let on_grid = grid.get(i).is_some_and(|&g| close(p, g)) || (i > 0 && close(p, grid[i - 1]));
        if !on_grid && kept.last().is_none_or(|&k| !close(p, k)) {
            kept.push(p);
        }
    }
    let added = kept.len();
    grid.extend(kept);
    grid.sort_by(f64::total_cmp);
    added
}

/// Add points until no interval is more than `MAX_SPACING_RATIO` times either
/// neighbour, or no further point lies outside the merge tolerance.  A longer
/// interval is split from its shorter neighbour's side at distances halving
/// from its middle down to that neighbour's length, so the new intervals
/// double outward.
///
/// SAMMY Ref: `dat/mdat5.f90` RefineGrid lines 264-328, 440-459
fn grade_spacing(grid: &mut Vec<f64>) {
    loop {
        let mut points = Vec::new();
        for k in 1..grid.len() - 1 {
            let below = grid[k] - grid[k - 1];
            let above = grid[k + 1] - grid[k];
            let finest = MERGE_RELATIVE_TOL * grid[k].abs();
            if below > MAX_SPACING_RATIO * above {
                let mut d = below / 2.0;
                while d >= above.max(finest) {
                    points.push(grid[k] - d);
                    d /= 2.0;
                }
            } else if above > MAX_SPACING_RATIO * below {
                let mut d = above / 2.0;
                while d >= below.max(finest) {
                    points.push(grid[k] + d);
                    d /= 2.0;
                }
            }
        }
        if merge_points(grid, points) == 0 {
            return;
        }
    }
}

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

    fn gaussian(res: ResolutionParams) -> ResolutionFunction {
        ResolutionFunction::Gaussian(res)
    }

    #[test]
    fn test_empty_grid() {
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let (ext, indices) = build_working_grid(&[], &gaussian(res), &[]);
        assert!(ext.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn test_single_point() {
        let energies = vec![100.0];
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let (ext, indices) = build_working_grid(&energies, &gaussian(res), &[]);
        assert_eq!(ext, vec![100.0]);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_data_indices_roundtrip() {
        let data = vec![1.0, 5.0, 10.0, 100.0];
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let (ext, indices) = build_working_grid(&data, &gaussian(res), &[]);
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
        let (ext, _) = build_working_grid(&data, &gaussian(res), &[]);

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
        let (ext, _) = build_working_grid(&data, &gaussian(res), &[]);
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
        let (ext, _) = build_working_grid(&data, &gaussian(res), &[]);
        for &e in &ext {
            assert!(e > 0.0, "non-positive energy: {e}");
        }
    }

    #[test]
    fn test_fine_structure_adds_points() {
        let data: Vec<f64> = (0..20).map(|i| 490.0 + i as f64 * 5.0).collect();
        let kernel = (vec![0.0, 1.0, 2.0], vec![1.0, 1.0, 1.0]);
        let table = crate::resolution::TabulatedResolution::from_kernels(
            vec![100.0, 1000.0],
            vec![kernel.clone(), kernel],
            25.0,
        )
        .unwrap();
        let resolution = ResolutionFunction::Tabulated(std::sync::Arc::new(table));
        let resonances = vec![(500.0, 1.0)];

        let (ext_without, _) = build_working_grid(&data, &resolution, &[]);
        let (ext_with, _) = build_working_grid(&data, &resolution, &resonances);

        assert!(
            ext_with.len() > ext_without.len(),
            "resonance points should be added: {} vs {}",
            ext_with.len(),
            ext_without.len()
        );

        let lo = ext_with.partition_point(|&e| e < 499.0);
        let hi = ext_with.partition_point(|&e| e <= 501.0);
        assert!(
            hi - lo >= MIN_POINTS_PER_WIDTH,
            "expected ≥{MIN_POINTS_PER_WIDTH} points in resonance width, got {}",
            hi - lo
        );
    }

    #[test]
    fn a_repeated_energy_and_a_gap_below_the_merge_tolerance_keep_every_data_point() {
        let mut data: Vec<f64> = (0..20).map(|i| 490.0 + i as f64 * 5.0).collect();
        data.insert(11, data[10]);
        data.insert(6, data[5] * (1.0 + 1e-12));
        let kernel = (vec![0.0, 1.0, 2.0], vec![1.0, 1.0, 1.0]);
        let table = crate::resolution::TabulatedResolution::from_kernels(
            vec![100.0, 1000.0],
            vec![kernel.clone(), kernel],
            25.0,
        )
        .unwrap();
        let resolution = ResolutionFunction::Tabulated(std::sync::Arc::new(table));

        let (ext, indices) = build_working_grid(&data, &resolution, &[(500.0, 1.0)]);
        for (i, &e) in data.iter().enumerate() {
            assert_eq!(ext[indices[i]], e, "data[{i}] = {e} is not in the grid");
        }
    }

    #[test]
    fn test_fine_structure_skips_dense_grid() {
        // Dense grid: 0.1 eV spacing around a resonance with Gd=1.0 eV.
        // Already has ~20 points in [499, 501] → no fine-structure needed.
        let data: Vec<f64> = (0..100).map(|i| 495.0 + i as f64 * 0.1).collect();
        let res = ResolutionParams::new(10.0, 0.01, 0.001, 0.0).unwrap();
        let resonances = vec![(500.0, 1.0)];

        let (ext_without, _) = build_working_grid(&data, &gaussian(res), &[]);
        let (ext_with, _) = build_working_grid(&data, &gaussian(res), &resonances);

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

        let (ext, indices) = build_working_grid(&data, &gaussian(res), &resonances);
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

        let (ext_without, _) = build_working_grid(&data, &gaussian(res), &[]);
        let (ext_with, _) = build_working_grid(&data, &gaussian(res), &resonances);

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
            let (grid, _) = build_working_grid(&data, &resolution, &[]);
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
