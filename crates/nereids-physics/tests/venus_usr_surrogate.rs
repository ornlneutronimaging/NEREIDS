//! Regression tests for the surrogate cubature and scalar-Chebyshev
//! plans on a synthetic VENUS-like SAMMY USR resolution kernel.
//!
//! The kernel is the synthetic SAMMY USR-format kernel from
//! [`common::synthetic_venus_usr_tab`]; the real VENUS BL10 fixture
//! is not approved for public release (see `common/mod.rs` for the
//! full rationale and issue #557 for the CI-coverage gap this
//! replaces).

mod common;

use nereids_physics::resolution::{ResolutionMatrix, apply_r};
use nereids_physics::surrogate::{ScalarChebyshevPlan, SparseEmpiricalCubaturePlan};

// ── Helpers duplicated from `src/surrogate.rs` ───────────────────

fn exact_forward(matrix: &ResolutionMatrix, sigmas: &[f64], k: usize, n: &[f64]) -> Vec<f64> {
    let n_rows = matrix.len();
    let mut t_un = vec![0.0_f64; n_rows];
    for (ell, t) in t_un.iter_mut().enumerate() {
        let mut dot = 0.0_f64;
        for j in 0..k {
            dot += n[j] * sigmas[j * n_rows + ell];
        }
        *t = (-dot).exp();
    }
    apply_r(matrix, &t_un)
}

fn max_hybrid_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-12);
            (x - y).abs() / denom
        })
        .fold(0.0_f64, f64::max)
}

// ── Tests ─────────────────────────────────────────────────────────

/// k = 1 grouped case: the cubature should match the exact
/// `ResolutionMatrix @ exp(-n σ)` forward output to LP precision at
/// the training densities, and produce bounded error at held-out
/// densities inside the training box.  The design study's 20-seed KL
/// follow-up showed 1.27× scatter inflation on grouped-Hf k=1 — this
/// test does not re-measure that; it only checks forward-model
/// correctness.
#[test]
fn cubature_venus_like_k1_forward_equivalence() {
    let tab = common::synthetic_venus_usr_tab();

    // Smaller production-ish grid (512 instead of 3471) to keep
    // LP-per-row cost tractable within a single test — the cubature
    // structure is grid-independent, so 512 suffices to show
    // correctness.
    let n_grid = 512_usize;
    let energies: Vec<f64> = (0..n_grid)
        .map(|i| 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64))
        .collect();
    let plan = tab.plan(&energies).expect("build plan on sorted grid");
    common::assert_kernel_broadens(&plan, &energies);
    let matrix = plan.compile_to_matrix();

    // Synthetic Gaussian-resonance σ on the real energy grid — we
    // don't need real ENDF σ to prove the cubature math; a physically
    // plausible σ shape is sufficient.
    let sigma: Vec<f64> = energies
        .iter()
        .map(|&e| {
            let g = (-((e - 80.0).powi(2)) / 8.0).exp();
            100.0 * g + 5.0
        })
        .collect();

    let train_max = [2e-4_f64];
    let training = SparseEmpiricalCubaturePlan::default_training_points(&train_max);
    let anchor = SparseEmpiricalCubaturePlan::default_jacobian_anchor(&train_max);
    let cub = SparseEmpiricalCubaturePlan::build(&matrix, &sigma, 1, &training, &anchor)
        .expect("build k=1 cubature on VENUS-like kernel");

    // At each training density, cubature = exact.
    for n_s in training.iter() {
        let t_cub = cub.forward(n_s);
        let t_exact = exact_forward(&matrix, &sigma, 1, n_s);
        let max_err = max_hybrid_err(&t_cub, &t_exact);
        assert!(
            max_err < 1e-9,
            "VENUS k=1 training n={n_s:?} max err = {max_err:.3e}",
        );
    }

    // At held-out density (VENUS production ~ 1.6e-4), bounded
    // error.  The design study's real-VENUS Hf aggregated fit showed a density
    // shift of 1.66e-4 relative — which means forward error at the
    // optimum is at least that small.  Here we allow 1e-3 max abs
    // error as a generous ceiling; the actual value on the
    // Beer-Lambert `T ∈ [0, 1]` output is typically an order of
    // magnitude tighter.
    let n_venus = vec![1.6e-4];
    let t_cub = cub.forward(&n_venus);
    let t_exact = exact_forward(&matrix, &sigma, 1, &n_venus);
    let max_abs = t_cub
        .iter()
        .zip(t_exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs < 1e-3,
        "VENUS k=1 held-out n=1.6e-4 max abs err = {max_abs:.3e}",
    );

    // Log compression ratio for diagnostics.
    let exact_nnz = matrix.nnz();
    let cub_atoms = cub.n_atoms();
    eprintln!(
        "VENUS k=1 cubature compression: {exact_nnz} exact nnz → {cub_atoms} atoms ({:.1}× compression)",
        exact_nnz as f64 / cub_atoms as f64,
    );
}

/// VENUS-like regression test for the Chebyshev scalar surrogate on
/// the 3471-bin VENUS production grid with synthetic Hf-like σ.
/// Asserts forward accuracy ≤ 1e-5 at VENUS density (issue #475
/// success criterion) and logs per-call wall time.  A bench-off ran
/// this against a Lanczos σ-pushforward Gauss quadrature on the
/// same kernel and picked Chebyshev — it won on both accuracy
/// (≤ 2e-15 vs ≤ 4e-15) and wall-time.  Exact speedup ratios are
/// hardware-dependent and intentionally not pinned here; the
/// accuracy bound stays portable.
#[test]
fn scalar_chebyshev_venus_like_k1_regression() {
    let tab = common::synthetic_venus_usr_tab();

    // VENUS production grid size.
    let n_grid = 3471_usize;
    let energies: Vec<f64> = (0..n_grid)
        .map(|i| 7.0 + (200.0 - 7.0) * (i as f64) / ((n_grid - 1) as f64))
        .collect();
    let plan = std::sync::Arc::new(tab.plan(&energies).expect("build plan on sorted grid"));
    common::assert_kernel_broadens(&plan, &energies);
    let matrix = plan.compile_to_matrix();
    let matrix_nnz = matrix.nnz();

    // Synthetic Hf-like σ: Gaussian resonance peaks + baseline
    // potential scattering.  Rough ENDF-Hf magnitudes (peaks up to
    // O(1e3) barns, baseline ~10 barns).
    let sigma: Vec<f64> = energies
        .iter()
        .map(|&e| {
            let peak_a = 1200.0 * (-((e - 80.0).powi(2)) / 8.0).exp();
            let peak_b = 600.0 * (-((e - 120.0).powi(2)) / 6.0).exp();
            let peak_c = 300.0 * (-((e - 160.0).powi(2)) / 10.0).exp();
            peak_a + peak_b + peak_c + 10.0
        })
        .collect();

    // Training box: 2 × VENUS density.  M = 16 Chebyshev nodes —
    // a bench run showed this achieves 1e-15 forward accuracy on
    // the whole box.
    let n_max = 2e-4_f64;
    let t_build = std::time::Instant::now();
    let cheb = ScalarChebyshevPlan::build(std::sync::Arc::clone(&plan), &sigma, n_max, 16)
        .expect("build Chebyshev plan");
    let t_build = t_build.elapsed();

    // Forward accuracy at VENUS density.
    let n_venus = 1.6e-4_f64;
    let t_exact = exact_forward(&matrix, &sigma, 1, &[n_venus]);
    let t_cheb = cheb.forward_scalar(n_venus);
    let max_err = max_hybrid_err(&t_cheb, &t_exact);

    // Per-forward timing.
    let n_iters = 1000;
    let t0 = std::time::Instant::now();
    let mut sink = 0.0_f64;
    for _ in 0..n_iters {
        sink += cheb.forward_scalar(n_venus)[0];
    }
    let t_fwd = t0.elapsed() / n_iters as u32;
    std::hint::black_box(sink);

    let t_un: Vec<f64> = sigma.iter().map(|&s| (-n_venus * s).exp()).collect();
    let t0 = std::time::Instant::now();
    let mut sink = 0.0_f64;
    for _ in 0..n_iters {
        sink += apply_r(&matrix, &t_un)[0];
    }
    let t_exact_fwd = t0.elapsed() / n_iters as u32;
    std::hint::black_box(sink);

    eprintln!();
    eprintln!(
        "=== scalar Chebyshev VENUS k=1 regression (n_grid = {n_grid}, matrix nnz = {matrix_nnz}) ==="
    );
    eprintln!(
        "Chebyshev (M=16):  build = {:>9.1?}  n_coeff = {:>7}  fwd = {:>8.2?}  max_err @ VENUS = {:.3e}",
        t_build,
        n_grid * 16,
        t_fwd,
        max_err,
    );
    eprintln!(
        "Exact apply_r:     build = {:>9}                     fwd = {:>8.2?}  (reference)",
        "n/a", t_exact_fwd,
    );
    eprintln!(
        "Cheb speedup vs exact: {:.1}×",
        t_exact_fwd.as_nanos() as f64 / t_fwd.as_nanos() as f64,
    );
    eprintln!();

    // Issue #475 success criteria.
    assert!(
        max_err < 1e-5,
        "Chebyshev forward max err {max_err:.3e} exceeds 1e-5 ceiling",
    );
}
