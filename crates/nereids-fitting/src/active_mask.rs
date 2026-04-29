//! Active-bin masking for fit-energy-range restriction.
//!
//! SAMMY REGION equivalent (`MIN ENERGY` / `MAX ENERGY` SAM52 cards).
//! When a user restricts the fit to `[E_min, E_max]`, the GUI extends
//! the energy grid by ~5×FWHM beyond each boundary so resonances near
//! the boundaries are correctly broadened, and the cost-function paths
//! (LM transmission, joint-Poisson PBD) consult the mask returned here
//! to skip residual contributions outside `[E_min, E_max]`.
//!
//! Returns `None` when no range is set; callers treat that as
//! "all bins active" (default behaviour).

/// Build a per-bin active mask from the energy grid and an optional
/// user-specified `[E_min, E_max]` range.  Bin `i` is active iff
/// `E_min ≤ energies[i] ≤ E_max`.
///
/// Returns `None` when `range` is `None` so the caller can short-circuit
/// the all-active common case without allocating.
pub fn build_active_mask(energies: &[f64], range: Option<(f64, f64)>) -> Option<Vec<bool>> {
    let (lo, hi) = range?;
    Some(energies.iter().map(|&e| e >= lo && e <= hi).collect())
}

/// Active-bin count for a mask.  `None` = all bins active.
pub fn active_count(mask: Option<&[bool]>, n_total: usize) -> usize {
    match mask {
        Some(m) => m.iter().filter(|&&b| b).count(),
        None => n_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_returns_none_for_full_grid() {
        let energies = [1.0, 2.0, 3.0];
        assert!(build_active_mask(&energies, None).is_none());
    }

    #[test]
    fn build_marks_bins_inside_range_inclusive() {
        let energies = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = build_active_mask(&energies, Some((2.0, 4.0))).unwrap();
        assert_eq!(mask, vec![false, true, true, true, false]);
    }

    #[test]
    fn build_handles_empty_active_region() {
        let energies = [1.0, 2.0, 3.0];
        let mask = build_active_mask(&energies, Some((10.0, 20.0))).unwrap();
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn active_count_with_mask_counts_true_bins() {
        let mask = [true, false, true, true, false];
        assert_eq!(active_count(Some(&mask), 5), 3);
    }

    #[test]
    fn active_count_without_mask_returns_total() {
        assert_eq!(active_count(None, 42), 42);
    }
}
