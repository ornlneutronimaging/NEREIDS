//! Fixture-gated regression tests for the VENUS USR resolution
//! operator and CSR-compiled [`crate::resolution::ResolutionMatrix`].
//!
//! These tests load the SAMMY-format tabulated resolution kernel for
//! the VENUS instrument (SNS Beam Line 10) from a gitignored file at
//! the workspace root.  When the fixture is absent (CI, fresh
//! checkouts), each test early-returns and is reported as passing —
//! no `#[ignore]` noise.  When the fixture is present, the tests
//! pin bit-exactness (where the test name says so) or hybrid
//! abs+rel equivalence within `1e-12` against the slow oracle
//! [`broaden_presorted_reference`].
//!
//! See `tests/README.md` for the rationale behind the early-return
//! idiom and issue #497 for the move from `#[ignore]`'d tests in
//! `src/resolution.rs`.

mod common;

use nereids_physics::resolution::{
    ResolutionError, ResolutionMatrix, ResolutionPlan, TabulatedResolution, apply_r,
    apply_resolution_with_matrix, test_support,
};

// ── Helpers (duplicated from `src/resolution.rs` tests — keeping
//    the crate's public surface minimal per issue #497 scope) ─────

fn interp_spectrum(energies: &[f64], spectrum: &[f64], e: f64) -> Option<f64> {
    use nereids_core::constants::NEAR_ZERO_FLOOR;
    if e < energies[0] || e > *energies.last().unwrap() {
        return None;
    }
    let lo = match energies
        .binary_search_by(|x| x.partial_cmp(&e).unwrap_or(std::cmp::Ordering::Equal))
    {
        Ok(idx) => return Some(spectrum[idx]),
        Err(idx) => idx.saturating_sub(1),
    };
    let hi = lo + 1;
    if hi >= energies.len() {
        return Some(spectrum[lo]);
    }
    let span = energies[hi] - energies[lo];
    if span.abs() < NEAR_ZERO_FLOOR {
        return Some(spectrum[lo]);
    }
    let frac = (e - energies[lo]) / span;
    Some(spectrum[lo] + frac * (spectrum[hi] - spectrum[lo]))
}

/// Reference implementation — the pre-optimization
/// `broaden_presorted`.  Used solely as the equivalence oracle in
/// the bit-exact test below.  Duplicated from `src/resolution.rs`
/// rather than promoting the crate-internal helper to `pub`; uses
/// `test_support::TOF_FACTOR` + `test_support::interpolated_kernel`
/// so the oracle and SUT share the exact same constants and
/// interior math.
fn broaden_presorted_reference(
    tab: &TabulatedResolution,
    energies: &[f64],
    spectrum: &[f64],
) -> Vec<f64> {
    use nereids_core::constants::DIVISION_FLOOR;
    let tof_factor = test_support::TOF_FACTOR;

    let n = energies.len();
    if n == 0 {
        return vec![];
    }
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let e = energies[i];
        if e <= 0.0 {
            result[i] = spectrum[i];
            continue;
        }
        let tof_center = tof_factor * tab.flight_path_m() / e.sqrt();
        let (offsets, weights) = test_support::interpolated_kernel(tab, e);
        let mut sum = 0.0;
        let mut norm = 0.0;
        for k in 0..offsets.len() {
            let dt = offsets[k];
            let w = weights[k];
            if w <= 0.0 {
                continue;
            }
            let tof_prime = tof_center + dt;
            if tof_prime <= 0.0 {
                continue;
            }
            let e_prime = (tof_factor * tab.flight_path_m() / tof_prime).powi(2);
            let s = match interp_spectrum(energies, spectrum, e_prime) {
                Some(v) => v,
                None => continue,
            };
            let dt_width = if k > 0 && k < offsets.len() - 1 {
                (offsets[k + 1] - offsets[k - 1]) * 0.5
            } else if k == 0 && offsets.len() > 1 {
                offsets[1] - offsets[0]
            } else if k == offsets.len() - 1 && offsets.len() > 1 {
                offsets[k] - offsets[k - 1]
            } else {
                1.0
            };
            let weight = w * dt_width.abs();
            sum += weight * s;
            norm += weight;
        }
        result[i] = if norm > DIVISION_FLOOR {
            sum / norm
        } else {
            spectrum[i]
        };
    }
    result
}

fn assert_bit_exact(reference: &[f64], actual: &[f64], label: &str) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    for (i, (&a, &b)) in reference.iter().zip(actual.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: element {i} mismatch: reference={a:.17e} actual={b:.17e}"
        );
    }
}

/// Hybrid abs+rel tolerance used across equivalence tests.  Guards
/// against the `a ≈ 0` trap where `a.abs().max(1e-300)` produces
/// meaningless relative errors for genuinely-zero reference values.
fn max_hybrid_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-12);
            (x - y).abs() / denom
        })
        .fold(0.0_f64, f64::max)
}

/// Build a `TabulatedResolution` + plan + matrix on a uniform energy
/// grid using the VENUS USR fixture kernel.  Helper duplicated from
/// `src/resolution.rs` rather than promoted to the public API.
///
/// Panics if the fixture is missing — callers must early-return via
/// [`common::venus_usr_resolution_path`] BEFORE invoking this.
fn build_fixture_plan_and_matrix(
    fixture_path: &std::path::Path,
    n_grid: usize,
) -> (Vec<f64>, ResolutionPlan, ResolutionMatrix) {
    let text = std::fs::read_to_string(fixture_path).expect("read VENUS USR fixture");
    let res = TabulatedResolution::from_text(&text, 25.0).expect("parse VENUS USR fixture");
    let energies: Vec<f64> = (0..n_grid)
        .map(|i| 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64))
        .collect();
    let plan = res.plan(&energies).expect("build plan on sorted grid");
    let matrix = plan.compile_to_matrix();
    (energies, plan, matrix)
}

// ── Tests ─────────────────────────────────────────────────────────

/// Real VENUS BL10 (SNS) resolution kernel, SAMMY USR format, on a
/// real Hf-like resonance spectrum over the full VENUS analysis
/// grid.  This is the closest regression of the production A.1 /
/// B.2 workload.
#[test]
fn test_broaden_presorted_bit_exact_on_venus_usr() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read VENUS USR fixture");
    let tab = TabulatedResolution::from_text(&text, 25.0).unwrap();

    // Production-like grid: uniform 7..200 eV with ~3500 bins
    let n = 3471;
    let energies: Vec<f64> = (0..n)
        .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
        .collect();
    // Resonance-dip spectrum (toy model, exercises the math regardless
    // of actual Hf σ, which is what we want for an interp test).
    let spectrum: Vec<f64> = energies
        .iter()
        .map(|&e| {
            1.0 - 0.8 * (-((e - 7.8).powi(2) / 0.01)).exp()
                - 0.5 * (-((e - 13.9).powi(2) / 0.04)).exp()
                - 0.6 * (-((e - 22.4).powi(2) / 0.1)).exp()
        })
        .collect();

    let reference = broaden_presorted_reference(&tab, &energies, &spectrum);
    let actual = test_support::broaden_presorted(&tab, &energies, &spectrum);
    assert_bit_exact(&reference, &actual, "venus_usr_real_resolution");
}

/// End-to-end row-stochasticity on the real VENUS kernel,
/// 512-point grid.  Gated on the VENUS USR resolution fixture
/// (SAMMY-format).
#[test]
fn resolution_matrix_is_row_stochastic_on_venus_kernel() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let (_energies, _plan, matrix) = build_fixture_plan_and_matrix(&path, 512);
    for i in 0..matrix.len() {
        let start = matrix.row_starts()[i] as usize;
        let end = matrix.row_starts()[i + 1] as usize;
        let row_sum: f64 = matrix.values()[start..end].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-13,
            "row {} sum = {} (expected 1.0 within 1e-13)",
            i,
            row_sum,
        );
    }
}

#[test]
fn resolution_matrix_apply_equivalent_to_plan_apply_on_venus_kernel() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let (_energies, plan, matrix) = build_fixture_plan_and_matrix(&path, 512);
    let n_grid = matrix.len();
    let spec: Vec<f64> = (0..n_grid)
        .map(|i| {
            let e = 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64);
            let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
            (-1.6e-4 * sigma).exp()
        })
        .collect();
    let plan_out = plan.apply(&spec);
    let matrix_out = apply_r(&matrix, &spec);
    let max_err = max_hybrid_err(&plan_out, &matrix_out);
    assert!(
        max_err < 1e-12,
        "apply_r vs plan.apply max hybrid err = {:.3e} (expected < 1e-12)",
        max_err,
    );
}

/// Production-grid guardrail for the `1e-12` tolerance documented
/// on `ResolutionPlan::compile_to_matrix`.  The 3471-bin VENUS grid
/// has ~82 entries per row, so accumulation error is an order of
/// magnitude larger than on the synthetic multi-row tests in
/// `src/resolution.rs`; this test pins the equivalence bound at
/// production scale so a future regression in either `apply` or
/// `apply_r` summation order fails loudly.  Logs the observed
/// `max_hybrid_err` via `eprintln!` so `-- --nocapture` runs surface
/// the actual headroom against the 1e-12 ceiling.
#[test]
fn resolution_matrix_apply_equivalent_at_production_grid() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let (_energies, plan, matrix) = build_fixture_plan_and_matrix(&path, 3471);
    let n_grid = matrix.len();
    // Same Beer-Lambert test spectrum as the 512-point test.
    let spec: Vec<f64> = (0..n_grid)
        .map(|i| {
            let e = 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64);
            let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
            (-1.6e-4 * sigma).exp()
        })
        .collect();
    let plan_out = plan.apply(&spec);
    let matrix_out = apply_r(&matrix, &spec);
    let max_err = max_hybrid_err(&plan_out, &matrix_out);
    eprintln!(
        "3471-grid apply_r vs plan.apply observed max_hybrid_err = {:.3e} \
         (ceiling 1e-12; theoretical bound ~1e-13 per row × 82 rows/entry)",
        max_err,
    );
    assert!(
        max_err < 1e-12,
        "3471-grid apply_r vs plan.apply max hybrid err = {:.3e} (expected < 1e-12)",
        max_err,
    );
}

#[test]
fn resolution_matrix_apply_equivalent_across_densities() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let (_energies, plan, matrix) = build_fixture_plan_and_matrix(&path, 512);
    let n_grid = matrix.len();
    for &n_density in &[1e-5_f64, 1e-4, 1.6e-4, 1e-3] {
        let spec: Vec<f64> = (0..n_grid)
            .map(|i| {
                let e = 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64);
                let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                    + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
                (-n_density * sigma).exp()
            })
            .collect();
        let plan_out = plan.apply(&spec);
        let matrix_out = apply_r(&matrix, &spec);
        let max_err = max_hybrid_err(&plan_out, &matrix_out);
        assert!(
            max_err < 1e-12,
            "density n={:.1e}: max hybrid err {:.3e} (expected < 1e-12)",
            n_density,
            max_err,
        );
    }
}

#[test]
fn resolution_matrix_csr_column_indices_sorted_per_row() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let (_energies, _plan, matrix) = build_fixture_plan_and_matrix(&path, 256);
    for i in 0..matrix.len() {
        let start = matrix.row_starts()[i] as usize;
        let end = matrix.row_starts()[i + 1] as usize;
        let row_cols = &matrix.col_indices()[start..end];
        for w in row_cols.windows(2) {
            assert!(
                w[0] < w[1],
                "row {} col_indices not strictly ascending: {:?}",
                i,
                row_cols,
            );
        }
    }
}

#[test]
fn resolution_matrix_grid_mismatch_detected() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let (energies, _plan, matrix) = build_fixture_plan_and_matrix(&path, 128);
    let spec = vec![1.0_f64; matrix.len()];

    // Same grid → passes.
    let ok = apply_resolution_with_matrix(&energies, &matrix, &spec);
    assert!(ok.is_ok());

    // Perturb one energy → MatrixGridMismatch with the offending index.
    let mut mutated = energies.clone();
    mutated[37] += 1e-12;
    let err = apply_resolution_with_matrix(&mutated, &matrix, &spec)
        .expect_err("grid mismatch must error");
    assert_eq!(
        err,
        ResolutionError::MatrixGridMismatch {
            first_diff_index: 37,
        }
    );
}

#[test]
fn resolution_matrix_length_mismatch_detected() {
    let Some(path) = common::venus_usr_resolution_path() else {
        return;
    };
    let (energies, _plan, matrix) = build_fixture_plan_and_matrix(&path, 64);
    let short_spec = vec![1.0_f64; matrix.len() - 1];
    let err = apply_resolution_with_matrix(&energies, &matrix, &short_spec)
        .expect_err("length mismatch must error");
    assert!(matches!(err, ResolutionError::LengthMismatch { .. }));
}
