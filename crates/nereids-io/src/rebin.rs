//! Rebinning: coarsen the TOF axis by an integer factor.
//!
//! Counts in adjacent bins are summed, and the merged bins keep every
//! `factor`-th edge. Rebinning needs bin edges: a merged bin's position
//! depends on the widths of the bins it merges, which centres do not give.

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

/// Rebin bin edges by keeping every `factor`-th edge, always including
/// the final edge.
///
/// Input: N+1 edges for N bins.
/// Output: ceil(N/factor)+1 edges for ceil(N/factor) bins.
///
/// Returns the original edges unchanged if `factor < 2`.
pub fn rebin_edges(edges: &[f64], factor: usize) -> Vec<f64> {
    if factor < 2 || edges.len() <= 1 {
        return edges.to_vec();
    }
    let mut out: Vec<f64> = edges.iter().step_by(factor).copied().collect();
    // Always include the final edge
    if out.last() != edges.last() {
        out.push(*edges.last().unwrap());
    }
    out
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
    fn edges_factor_zero_returns_clone() {
        let edges = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(rebin_edges(&edges, 0), edges);
    }
}
