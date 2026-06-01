//! Small numeric-invariant helpers shared across NEREIDS crates.
//!
//! These live in `nereids-core` (the dependency-free foundation crate) so that
//! the same invariant is enforced identically everywhere instead of being
//! re-implemented per crate.  Each helper returns the *first* offending
//! `(index, value)` rather than a formatted error, so the calling crate can
//! map the failure onto its own error type / message wording without
//! `nereids-core` having to know about `IoError`, `FittingError`, etc.

/// Locate the first element that is not finite-and-non-negative.
///
/// Detector counts (and other count-like quantities) are non-negative by
/// construction — zero is legitimate, but a NaN, ±∞, or negative entry signals
/// an upstream loader / normalisation bug.  A bare `v < 0.0` test is *not*
/// sufficient because `NaN < 0.0` is `false`; this helper therefore rejects on
/// `!v.is_finite() || v < 0.0`, pairing the order comparison with a finiteness
/// check (NaN bypasses `<`).
///
/// Returns `Err((i, v))` for the first offending element at flat index `i`, or
/// `Ok(())` if every element is finite and `>= 0.0`.  An empty iterator is
/// vacuously `Ok(())`.
pub fn first_non_finite_or_negative<I>(values: I) -> Result<(), (usize, f64)>
where
    I: IntoIterator<Item = f64>,
{
    for (i, v) in values.into_iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Err((i, v));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_ok() {
        assert!(first_non_finite_or_negative(std::iter::empty::<f64>()).is_ok());
    }

    #[test]
    fn all_finite_non_negative_is_ok() {
        assert!(first_non_finite_or_negative([0.0, 1.0, 1e9, 2.5]).is_ok());
    }

    #[test]
    fn rejects_negative_with_index() {
        assert_eq!(
            first_non_finite_or_negative([1.0, 2.0, -3.0, 4.0]),
            Err((2, -3.0))
        );
    }

    #[test]
    fn rejects_nan_negative_does_not_bypass() {
        // `NaN < 0.0` is `false`; the `is_finite()` half of the guard is what
        // catches it.  Reported value compares unequal to itself, so match on
        // index + NaN-ness rather than `assert_eq!` on the tuple.
        let err = first_non_finite_or_negative([1.0, f64::NAN, 3.0]).unwrap_err();
        assert_eq!(err.0, 1);
        assert!(err.1.is_nan());
    }

    #[test]
    fn rejects_positive_and_negative_infinity() {
        assert_eq!(
            first_non_finite_or_negative([1.0, f64::INFINITY]),
            Err((1, f64::INFINITY))
        );
        assert_eq!(
            first_non_finite_or_negative([f64::NEG_INFINITY]),
            Err((0, f64::NEG_INFINITY))
        );
    }

    #[test]
    fn reports_first_offender_only() {
        // -1.0 at index 1 comes before NaN at index 3.
        assert_eq!(
            first_non_finite_or_negative([1.0, -1.0, 2.0, f64::NAN]),
            Err((1, -1.0))
        );
    }
}
