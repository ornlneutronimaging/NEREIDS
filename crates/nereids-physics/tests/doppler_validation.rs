//! Regression tests for the `doppler_broaden` energy-grid contract.
//!
//! `doppler_broaden` and `doppler_broaden_with_derivative` are public
//! physics leaves that previously trusted their callers to supply a
//! positive, finite, strictly ascending energy grid. A malformed grid
//! would not panic but would silently propagate NaN through the FGM
//! convolution kernel (because `NaN < FLOOR` is false, bypassing the
//! per-point velocity guard) or return unspecified `partition_point`
//! indices on an unsorted slice — both yielding wrong outputs rather
//! than typed errors.
//!
//! These tests pin the typed-error contract:
//! - NaN / ±∞ / 0 / negative energies → `DopplerError::InvalidEnergy`.
//! - Non-strictly-ascending grids → `DopplerError::UnsortedEnergies`.
//! - The same contract applies to the derivative variant (which
//!   delegates to `doppler_broaden` for its forward pass).

use nereids_physics::doppler::{
    DopplerError, DopplerParams, doppler_broaden, doppler_broaden_with_derivative,
};

fn params() -> DopplerParams {
    // Pick any valid params; we never reach the convolution because
    // validation rejects the bad grid first.
    DopplerParams::new(300.0, 238.0).expect("valid params")
}

#[test]
fn test_doppler_rejects_nan_energy() {
    let energies = vec![1.0, f64::NAN, 3.0];
    let xs = vec![10.0, 20.0, 30.0];
    let err = doppler_broaden(&energies, &xs, &params()).expect_err("NaN must be rejected");
    match err {
        DopplerError::InvalidEnergy { index, value } => {
            assert_eq!(index, 1);
            assert!(value.is_nan(), "expected NaN value, got {value}");
        }
        other => panic!("expected InvalidEnergy, got {other:?}"),
    }
}

#[test]
fn test_doppler_rejects_negative_energy() {
    let energies = vec![1.0, -2.0, 3.0];
    let xs = vec![10.0, 20.0, 30.0];
    let err =
        doppler_broaden(&energies, &xs, &params()).expect_err("negative energy must be rejected");
    assert!(
        matches!(
            err,
            DopplerError::InvalidEnergy {
                index: 1,
                value: v
            } if v == -2.0
        ),
        "expected InvalidEnergy at index 1 with value -2.0, got {err:?}"
    );
}

#[test]
fn test_doppler_rejects_zero_energy() {
    // Doppler broadening transforms to velocity space v = √E. Zero energy
    // would give v = 0, and the per-point convolution skips v < FLOOR — so
    // it would not crash, but the contract is "strictly positive" since
    // E = 0 has no physical meaning as an incident neutron energy.
    let energies = vec![0.0, 1.0, 2.0];
    let xs = vec![10.0, 20.0, 30.0];
    let err = doppler_broaden(&energies, &xs, &params()).expect_err("zero energy must be rejected");
    assert!(
        matches!(
            err,
            DopplerError::InvalidEnergy {
                index: 0,
                value: 0.0
            }
        ),
        "expected InvalidEnergy at index 0 with value 0.0, got {err:?}"
    );
}

#[test]
fn test_doppler_rejects_positive_infinity_energy() {
    let energies = vec![1.0, 2.0, f64::INFINITY];
    let xs = vec![10.0, 20.0, 30.0];
    let err = doppler_broaden(&energies, &xs, &params()).expect_err("+inf must be rejected");
    assert!(
        matches!(
            err,
            DopplerError::InvalidEnergy { index: 2, value } if value == f64::INFINITY
        ),
        "expected InvalidEnergy at index 2 with value +inf, got {err:?}"
    );
}

#[test]
fn test_doppler_rejects_unsorted_energies() {
    // 4.0 > 3.0 satisfies strictly-ascending against the previous entry;
    // the violation is at index 3 where 2.5 < 4.0.
    let energies = vec![1.0, 2.0, 4.0, 2.5, 5.0];
    let xs = vec![10.0, 20.0, 30.0, 25.0, 50.0];
    let err = doppler_broaden(&energies, &xs, &params())
        .expect_err("descending segment must be rejected");
    match err {
        DopplerError::UnsortedEnergies {
            index,
            previous,
            current,
        } => {
            assert_eq!(index, 3);
            assert_eq!(previous, 4.0);
            assert_eq!(current, 2.5);
        }
        other => panic!("expected UnsortedEnergies, got {other:?}"),
    }
}

#[test]
fn test_doppler_rejects_duplicate_energies() {
    // Strict ascending: equal neighbouring points are also rejected so
    // that the partition_point binary searches never get duplicate
    // boundary energies (which would produce zero-width FGM segments).
    let energies = vec![1.0, 2.0, 2.0, 3.0];
    let xs = vec![10.0, 20.0, 20.0, 30.0];
    let err = doppler_broaden(&energies, &xs, &params())
        .expect_err("duplicate energies must be rejected");
    match err {
        DopplerError::UnsortedEnergies {
            index,
            previous,
            current,
        } => {
            assert_eq!(index, 2);
            assert_eq!(previous, 2.0);
            assert_eq!(current, 2.0);
        }
        other => panic!("expected UnsortedEnergies, got {other:?}"),
    }
}

#[test]
fn test_doppler_accepts_valid_grid() {
    let energies = vec![1.0, 2.0, 5.0, 10.0];
    let xs = vec![10.0, 5.0, 1.0, 0.5];
    let result = doppler_broaden(&energies, &xs, &params()).expect("valid grid must succeed");
    assert_eq!(result.len(), energies.len());
    for v in &result {
        assert!(v.is_finite() && *v >= 0.0, "got non-finite/neg result {v}");
    }
}

#[test]
fn test_doppler_with_derivative_inherits_validation_nan() {
    let energies = vec![1.0, f64::NAN, 3.0];
    let xs = vec![10.0, 20.0, 30.0];
    let err = doppler_broaden_with_derivative(&energies, &xs, &params())
        .expect_err("derivative variant must reject NaN energies");
    assert!(
        matches!(
            err,
            DopplerError::InvalidEnergy { index: 1, value } if value.is_nan()
        ),
        "expected InvalidEnergy at index 1, got {err:?}"
    );
}

#[test]
fn test_doppler_with_derivative_inherits_validation_unsorted() {
    let energies = vec![1.0, 3.0, 2.0];
    let xs = vec![10.0, 30.0, 20.0];
    let err = doppler_broaden_with_derivative(&energies, &xs, &params())
        .expect_err("derivative variant must reject unsorted energies");
    assert!(
        matches!(
            err,
            DopplerError::UnsortedEnergies {
                index: 2,
                previous,
                current
            } if previous == 3.0 && current == 2.0
        ),
        "expected UnsortedEnergies at index 2, got {err:?}"
    );
}
