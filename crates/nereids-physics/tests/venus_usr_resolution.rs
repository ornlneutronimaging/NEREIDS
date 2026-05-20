//! Regression tests for the VENUS-like USR resolution operator and
//! CSR-compiled [`nereids_physics::resolution::ResolutionMatrix`].
//!
//! These tests build a synthetic SAMMY USR-format kernel via
//! [`common::synthetic_venus_usr_tab`] (see that module for the
//! ORNL-release-policy rationale that rules out vendoring the real
//! VENUS BL10 fixture). The kernel is parsed via the same
//! [`TabulatedResolution::from_text`] entry point the production
//! pipeline uses, so every stage of the SAMMY USR path is exercised
//! on every `cargo test` run.
//!
//! Each broadening-equivalence test calls [`common::assert_kernel_broadens`]
//! up front so a future regression that collapses the synthetic
//! kernel toward a delta — which would silently turn every
//! equivalence test into a vacuous identity — fails loudly at the
//! first test instead. See PR #544 for the silent-no-op-via-kernel-
//! shrink failure mode this pre-check guards against, and issue #557
//! for the original CI-coverage gap that motivated the synthetic
//! replacement.

mod common;

use nereids_physics::resolution::{
    ResolutionError, ResolutionMatrix, ResolutionPlan, TabulatedResolution, apply_r,
    apply_resolution_with_matrix, test_support,
};

// ── Helpers (duplicated from `src/resolution.rs` tests — keeping
//    the crate's public surface minimal per issue #497 scope) ─────

fn interp_spectrum(energies: &[f64], spectrum: &[f64], e: f64) -> Option<f64> {
    // Verbatim copy of the canonical helper at
    // crates/nereids-physics/src/resolution.rs:2541 (used by the
    // in-src `broaden_presorted_reference` oracle).  Codex flagged
    // an earlier rewritten variant in PR #545 round-1 review as
    // semantically divergent (binary_search_by + early-return on
    // exact hits vs upper-bound search + always-interpolate) —
    // either path of divergence can flip the bit-exact comparison
    // on exact-grid-hit / duplicate-grid edge cases.  Keep this in
    // sync with the in-src version verbatim; if the production
    // helper changes, mirror the change here.
    let n = energies.len();
    if n == 0 {
        return None;
    }
    if e < energies[0] || e > energies[n - 1] {
        return None;
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if energies[mid] <= e {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = energies[hi] - energies[lo];
    if span.abs() < nereids_core::constants::NEAR_ZERO_FLOOR {
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
/// grid using the synthetic VENUS-like USR kernel from
/// [`common::synthetic_venus_usr_tab`]. Helper duplicated from
/// `src/resolution.rs` rather than promoted to the public API.
///
/// **Does not** call `common::assert_kernel_broadens` itself — that
/// pre-check is meaningless at small grid sizes (the synthetic
/// kernel's energy-space FWHM is sub-bin once the grid coarsens to
/// `n_grid ≤ 128` on the 7 – 200 eV range), and small-grid callers
/// like `resolution_matrix_grid_mismatch_detected` only need a valid
/// matrix to test validation-path errors. Tests that exercise the
/// actual broadening math must call
/// [`common::assert_kernel_broadens`] themselves; see the equivalence
/// tests below for the pattern.
fn build_fixture_plan_and_matrix(n_grid: usize) -> (Vec<f64>, ResolutionPlan, ResolutionMatrix) {
    let res = common::synthetic_venus_usr_tab();
    let energies: Vec<f64> = (0..n_grid)
        .map(|i| 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64))
        .collect();
    let plan = res.plan(&energies).expect("build plan on sorted grid");
    let matrix = plan.compile_to_matrix();
    (energies, plan, matrix)
}

// ── Tests ─────────────────────────────────────────────────────────

/// VENUS-like USR resolution kernel (synthetic SAMMY-format
/// triangular kernel, see `common::synthetic_venus_usr_text`) on a
/// Hf-like resonance-dip spectrum over a production-scale analysis
/// grid. This is the closest regression of the production A.1 / B.2
/// workload that we can ship as a public test — the real VENUS BL10
/// fixture is not approved for public release (see `common/mod.rs`).
#[test]
fn test_broaden_presorted_bit_exact_on_venus_usr() {
    let tab = common::synthetic_venus_usr_tab();

    // Production-like grid: uniform 7..200 eV with ~3500 bins
    let n = 3471;
    let energies: Vec<f64> = (0..n)
        .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
        .collect();
    let plan = tab.plan(&energies).expect("build plan on sorted grid");
    common::assert_kernel_broadens(&plan, &energies);
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
    assert_bit_exact(&reference, &actual, "venus_usr_synthetic_resolution");
}

/// End-to-end row-stochasticity on the synthetic VENUS-like kernel,
/// 512-point grid.
#[test]
fn resolution_matrix_is_row_stochastic_on_venus_kernel() {
    let (energies, plan, matrix) = build_fixture_plan_and_matrix(512);
    common::assert_kernel_broadens(&plan, &energies);
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
    let (energies, plan, matrix) = build_fixture_plan_and_matrix(512);
    common::assert_kernel_broadens(&plan, &energies);
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
/// on `ResolutionPlan::compile_to_matrix`.  The 3471-bin grid has
/// many entries per row, so accumulation error is an order of
/// magnitude larger than on the small multi-row tests in
/// `src/resolution.rs`; this test pins the equivalence bound at
/// production scale so a future regression in either `apply` or
/// `apply_r` summation order fails loudly.  Logs the observed
/// `max_hybrid_err` via `eprintln!` so `-- --nocapture` runs surface
/// the actual headroom against the 1e-12 ceiling.
#[test]
fn resolution_matrix_apply_equivalent_at_production_grid() {
    let (energies, plan, matrix) = build_fixture_plan_and_matrix(3471);
    common::assert_kernel_broadens(&plan, &energies);
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
         (ceiling 1e-12; theoretical bound ~1e-13 per row × NNZ/entry)",
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
    let (energies, plan, matrix) = build_fixture_plan_and_matrix(512);
    common::assert_kernel_broadens(&plan, &energies);
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
    let (_energies, _plan, matrix) = build_fixture_plan_and_matrix(256);
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
    let (energies, _plan, matrix) = build_fixture_plan_and_matrix(128);
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
    let (energies, _plan, matrix) = build_fixture_plan_and_matrix(64);
    let short_spec = vec![1.0_f64; matrix.len() - 1];
    let err = apply_resolution_with_matrix(&energies, &matrix, &short_spec)
        .expect_err("length mismatch must error");
    assert!(matches!(err, ResolutionError::LengthMismatch { .. }));
}
