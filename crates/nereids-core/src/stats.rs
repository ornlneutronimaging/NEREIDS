//! Small robust-statistics helpers shared across NEREIDS crates.
//!
//! These live in `nereids-core` (the dependency-free foundation crate) so that
//! the same estimator is computed identically everywhere instead of being
//! re-implemented per crate.  Each helper returns a plain `Option`/value
//! rather than a formatted error, so the calling crate can map degenerate
//! inputs onto its own error type / message wording without `nereids-core`
//! having to know about `IoError`, `FittingError`, etc.
//!
//! Precondition: callers pass **finite** inputs (they enforce this via
//! [`crate::validation`] before calling in).  The helpers nevertheless sort
//! with [`f64::total_cmp`], which is a total order even over NaN (NaN sorts
//! last), so a violated precondition degrades to a deterministic — if
//! meaningless — result instead of a `partial_cmp` panic path.

/// Consistency factor converting a median absolute deviation into a Gaussian
/// standard-deviation estimate: `sigma ≈ MAD_TO_SIGMA * MAD`.
///
/// Derivation: for X ~ N(μ, σ²), the MAD about the median satisfies
/// P(|X − μ| ≤ MAD) = 1/2, i.e. MAD = σ·Φ⁻¹(3/4) where Φ is the standard
/// normal CDF.  The consistency factor is therefore
///
///   1 / Φ⁻¹(3/4) = 1 / 0.674489750196081… = 1.482602218505601…
///
/// so that `MAD_TO_SIGMA * MAD` is an unbiased estimate of σ for Gaussian
/// data while staying robust (50% breakdown point) against outliers.
pub const MAD_TO_SIGMA: f64 = 1.482_602_218_505_601_8;

/// Median of a slice of values.
///
/// Copies the input and sorts with [`f64::total_cmp`] (total, deterministic —
/// see the module-level precondition note on NaN).  For even `n` the median
/// is the midpoint mean of the two central order statistics.
///
/// # Arguments
/// * `values` — Sample values.  Precondition: finite (callers enforce via
///   [`crate::validation`]).
///
/// # Returns
/// `Some(median)`, or `None` if `values` is empty.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        // Even n: midpoint mean of the two central order statistics.
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Median absolute deviation of a slice about a given center.
///
/// Computes `median(|v − center|)` with the same conventions as [`median`]
/// (total-order sort, even-`n` midpoint mean).  Note this returns the *raw*
/// MAD — multiply by [`MAD_TO_SIGMA`] to obtain a Gaussian-consistent σ
/// estimate.
///
/// # Arguments
/// * `values` — Sample values.  Precondition: finite (callers enforce via
///   [`crate::validation`]).
/// * `center` — Center to take deviations about (typically the sample
///   median).
///
/// # Returns
/// `Some(mad)`, or `None` if `values` is empty.
pub fn median_abs_deviation(values: &[f64], center: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let deviations: Vec<f64> = values.iter().map(|v| (v - center).abs()).collect();
    median(&deviations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_empty_is_none() {
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn median_single_element() {
        assert_eq!(median(&[3.5]), Some(3.5));
    }

    #[test]
    fn median_odd_n() {
        // Unsorted input; median of {1, 2, 9} is 2.
        assert_eq!(median(&[9.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn median_even_n_midpoint_mean() {
        // Sorted {1, 2, 3, 10} → midpoint mean of 2 and 3 = 2.5.
        assert_eq!(median(&[10.0, 3.0, 1.0, 2.0]), Some(2.5));
    }

    #[test]
    fn mad_empty_is_none() {
        assert_eq!(median_abs_deviation(&[], 0.0), None);
    }

    #[test]
    fn mad_basic() {
        // Values {1, 2, 4, 8}, center 3 → |dev| = {2, 1, 1, 5} → sorted
        // {1, 1, 2, 5} → midpoint mean of 1 and 2 = 1.5.
        assert_eq!(median_abs_deviation(&[1.0, 2.0, 4.0, 8.0], 3.0), Some(1.5));
    }

    #[test]
    fn mad_about_own_median() {
        // Values {1, 2, 3, 4, 100}, median 3 → |dev| = {2, 1, 0, 1, 97}
        // → median 1: the outlier does not blow up the scale estimate.
        let vals = [1.0, 2.0, 3.0, 4.0, 100.0];
        let med = median(&vals).unwrap();
        assert_eq!(med, 3.0);
        assert_eq!(median_abs_deviation(&vals, med), Some(1.0));
    }

    #[test]
    fn mad_of_constant_slice_is_zero() {
        let vals = [7.0; 5];
        let med = median(&vals).unwrap();
        assert_eq!(median_abs_deviation(&vals, med), Some(0.0));
    }

    #[test]
    fn mad_to_sigma_matches_inverse_normal_quantile() {
        // Φ(1/MAD_TO_SIGMA) must equal 3/4: check via the error function
        // relation Φ(x) = (1 + erf(x/√2))/2 evaluated numerically with a
        // rational-approximation-free identity — instead verify the inverse
        // direction: for the standard normal, P(|X| <= q) = 1/2 at
        // q = 1/MAD_TO_SIGMA ≈ 0.67449.  We hard-check the published value
        // of Φ⁻¹(3/4) to 15 significant digits.
        let q = 1.0 / MAD_TO_SIGMA;
        assert!((q - 0.674_489_750_196_081_7).abs() < 1e-15);
    }
}
