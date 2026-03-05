//! Energy rebinning: coarsen the TOF/energy axis by integer factor.
//!
//! Two modes:
//! - **Counts**: sum adjacent bins (conserves total counts).
//! - **Transmission**: average adjacent bins (T = I/I₀ is a ratio,
//!   so bin-averaging with uniform I₀ gives the correct result).

use ndarray::{Array3, Axis, s};

/// Rebin a 3D counts array along axis 0 by summing groups of `factor` slices.
///
/// Remainder slices at the end are summed into the last output bin.
/// Returns the original array unchanged if `factor < 2`.
///
/// **NaN handling**: NaN values propagate through the sum — any NaN in a
/// group produces NaN in that output bin. This matches ndarray semantics.
pub fn rebin_counts(data: &Array3<f64>, factor: usize) -> Array3<f64> {
    if factor < 2 {
        return data.clone();
    }
    let n_old = data.shape()[0];
    let height = data.shape()[1];
    let width = data.shape()[2];
    let n_new = n_old.div_ceil(factor);
    let mut out = Array3::<f64>::zeros((n_new, height, width));
    for i in 0..n_new {
        let start = i * factor;
        let end = (start + factor).min(n_old);
        out.slice_mut(s![i, .., ..])
            .assign(&data.slice(s![start..end, .., ..]).sum_axis(Axis(0)));
    }
    out
}

/// Rebin a 3D transmission array along axis 0 by averaging groups of
/// `factor` slices.
///
/// Transmission T = I/I₀ is a ratio. With uniform I₀ across bins,
/// the correct rebinned transmission is the arithmetic mean of the
/// group. **This is an approximation** — real I₀ varies per pixel
/// and energy bin.
///
/// Returns the original array unchanged if `factor < 2`.
///
/// **NaN handling**: NaN values propagate through the average — any NaN
/// in a group produces NaN in that output bin. This matches ndarray semantics.
pub fn rebin_transmission(data: &Array3<f64>, factor: usize) -> Array3<f64> {
    if factor < 2 {
        return data.clone();
    }
    let n_old = data.shape()[0];
    let height = data.shape()[1];
    let width = data.shape()[2];
    let n_new = n_old.div_ceil(factor);
    let mut out = Array3::<f64>::zeros((n_new, height, width));
    for i in 0..n_new {
        let start = i * factor;
        let end = (start + factor).min(n_old);
        let group_len = (end - start) as f64;
        let sum = data.slice(s![start..end, .., ..]).sum_axis(Axis(0));
        out.slice_mut(s![i, .., ..]).assign(&(sum / group_len));
    }
    out
}

/// Rebin bin edges by keeping every `factor`-th edge, always including
/// the final edge.
///
/// Input: N+1 edges for N bins.
/// Output: ceil(N/factor)+1 edges for ceil(N/factor) bins.
///
/// Returns the original edges unchanged if `factor <= 1`.
pub fn rebin_edges(edges: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 || edges.len() <= 1 {
        return edges.to_vec();
    }
    let mut out: Vec<f64> = edges.iter().step_by(factor).copied().collect();
    // Always include the final edge
    if out.last() != edges.last() {
        out.push(*edges.last().unwrap());
    }
    out
}

/// Rebin bin centers by averaging each group of `factor` centers.
///
/// Returns the original centers unchanged if `factor <= 1`.
pub fn rebin_centers(centers: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 || centers.is_empty() {
        return centers.to_vec();
    }
    centers
        .chunks(factor)
        .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn counts_conservation() {
        let data = Array3::from_shape_fn((6, 2, 3), |(t, y, x)| (t + y + x) as f64);
        let original_sum: f64 = data.sum();
        let rebinned = rebin_counts(&data, 2);
        assert_eq!(rebinned.shape(), &[3, 2, 3]);
        assert!((rebinned.sum() - original_sum).abs() < 1e-10);
    }

    #[test]
    fn counts_conservation_remainder() {
        // 7 bins / 3 = 3 output bins (2+2+3 or ceil division)
        let data = Array3::from_shape_fn((7, 2, 2), |(t, _y, _x)| t as f64 + 1.0);
        let original_sum: f64 = data.sum();
        let rebinned = rebin_counts(&data, 3);
        assert_eq!(rebinned.shape(), &[3, 2, 2]);
        assert!((rebinned.sum() - original_sum).abs() < 1e-10);
    }

    #[test]
    fn counts_factor_one_is_identity() {
        let data = Array3::from_shape_fn((5, 2, 2), |(t, y, x)| (t * 10 + y + x) as f64);
        let rebinned = rebin_counts(&data, 1);
        assert_eq!(rebinned, data);
    }

    #[test]
    fn counts_factor_ge_nbins_single_bin() {
        let data = Array3::from_shape_fn((4, 2, 2), |(t, _y, _x)| t as f64);
        let rebinned = rebin_counts(&data, 100);
        assert_eq!(rebinned.shape(), &[1, 2, 2]);
        assert!((rebinned.sum() - data.sum()).abs() < 1e-10);
    }

    #[test]
    fn transmission_averaging() {
        // Uniform transmission of 0.5 should stay 0.5 after rebinning
        let data = Array3::from_elem((6, 2, 2), 0.5);
        let rebinned = rebin_transmission(&data, 3);
        assert_eq!(rebinned.shape(), &[2, 2, 2]);
        for &v in rebinned.iter() {
            assert!((v - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn transmission_averaging_remainder() {
        // 7 bins / 3: groups of [0,1,2], [3,4,5], [6]
        // Per-pixel values = bin index → means = 1.0, 4.0, 6.0
        let data = Array3::from_shape_fn((7, 1, 1), |(t, _y, _x)| t as f64);
        let rebinned = rebin_transmission(&data, 3);
        assert_eq!(rebinned.shape(), &[3, 1, 1]);
        assert!((rebinned[[0, 0, 0]] - 1.0).abs() < 1e-10); // mean(0,1,2)
        assert!((rebinned[[1, 0, 0]] - 4.0).abs() < 1e-10); // mean(3,4,5)
        assert!((rebinned[[2, 0, 0]] - 6.0).abs() < 1e-10); // mean(6)
    }

    #[test]
    fn edges_even_division() {
        let edges = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // 4 bins
        let rebinned = rebin_edges(&edges, 2);
        assert_eq!(rebinned, vec![0.0, 2.0, 4.0]); // 2 bins
    }

    #[test]
    fn edges_remainder() {
        let edges = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // 5 bins
        let rebinned = rebin_edges(&edges, 2);
        // step_by(2): [0, 2, 4] → last edge 5.0 appended
        assert_eq!(rebinned, vec![0.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn edges_factor_one() {
        let edges = vec![1.0, 2.0, 3.0];
        assert_eq!(rebin_edges(&edges, 1), edges);
    }

    #[test]
    fn centers_averaging() {
        let centers = vec![1.0, 3.0, 5.0, 7.0]; // 4 bins
        let rebinned = rebin_centers(&centers, 2);
        assert_eq!(rebinned, vec![2.0, 6.0]);
    }

    #[test]
    fn centers_remainder() {
        let centers = vec![1.0, 3.0, 5.0]; // 3 bins
        let rebinned = rebin_centers(&centers, 2);
        assert_eq!(rebinned, vec![2.0, 5.0]); // mean(1,3)=2, mean(5)=5
    }

    #[test]
    fn counts_empty_array() {
        let data = Array3::<f64>::zeros((0, 2, 2));
        let rebinned = rebin_counts(&data, 3);
        assert_eq!(rebinned.shape(), &[0, 2, 2]);
    }

    #[test]
    fn counts_factor_zero_returns_clone() {
        let data = Array3::from_shape_fn((4, 1, 1), |(t, _, _)| t as f64);
        let rebinned = rebin_counts(&data, 0);
        assert_eq!(rebinned, data);
    }

    #[test]
    fn transmission_factor_zero_returns_clone() {
        let data = Array3::from_elem((4, 1, 1), 0.5);
        let rebinned = rebin_transmission(&data, 0);
        assert_eq!(rebinned, data);
    }

    #[test]
    fn counts_nan_propagation() {
        let mut data = Array3::from_shape_fn((4, 1, 1), |(t, _, _)| t as f64);
        data[[1, 0, 0]] = f64::NAN; // bin 1 is NaN
        let rebinned = rebin_counts(&data, 2);
        // Group [0, NaN] should produce NaN
        assert!(rebinned[[0, 0, 0]].is_nan());
        // Group [2, 3] should produce 5.0
        assert!((rebinned[[1, 0, 0]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn transmission_nan_propagation() {
        let mut data = Array3::from_shape_fn((4, 1, 1), |(t, _, _)| t as f64);
        data[[1, 0, 0]] = f64::NAN; // bin 1 is NaN
        let rebinned = rebin_transmission(&data, 2);
        // Group [0, NaN] → mean is NaN
        assert!(rebinned[[0, 0, 0]].is_nan());
        // Group [2, 3] → mean = 2.5
        assert!((rebinned[[1, 0, 0]] - 2.5).abs() < 1e-10);
    }
}
