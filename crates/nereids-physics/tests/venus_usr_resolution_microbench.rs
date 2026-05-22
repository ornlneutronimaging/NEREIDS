//! Microbenchmarks for the synthetic VENUS-like USR resolution
//! operator and CSR `ResolutionMatrix` apply paths.
//!
//! These are slow tests intentionally separated from the regression
//! suite in `venus_usr_resolution.rs`: they print timing tables to
//! stderr and exist to validate optimization-PR claims (two-pointer
//! walk vs binary search, plan-reuse vs per-call, CSR matvec vs
//! plan apply).
//!
//! The kernel is the synthetic SAMMY USR-format kernel from
//! [`common::synthetic_venus_usr_tab`]; see that module for the
//! ORNL-release-policy rationale ruling out the real VENUS BL10
//! fixture. Wall-clock numbers reported here are **lower bounds on
//! the optimization payoff** because the synthetic kernel is narrower
//! than the production fixture (~41 sample points vs ~500), but
//! the relative-speedup math is robust to kernel size.
//!
//! Run manually with `--nocapture --release`, e.g.:
//!
//! ```text
//! cargo test --release -p nereids-physics --test venus_usr_resolution_microbench \
//!     -- --nocapture
//! ```

mod common;

use nereids_physics::resolution::{apply_r, test_support};

/// Two-pointer `broaden_presorted` vs binary-search reference.  Pins
/// the bit-exact sink equality between the two paths so a future
/// regression in the two-pointer walk surfaces here as well as in
/// the bit-exact regression test.
#[test]
fn test_broaden_presorted_bench() {
    let tab = common::synthetic_venus_usr_tab();

    let n = 3471;
    let energies: Vec<f64> = (0..n)
        .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
        .collect();
    {
        // No-op-regression pre-check: must precede the bit-exact
        // sink-equality assertion below, which would pass vacuously
        // if the kernel collapsed to a delta.
        let pre_plan = tab.plan(&energies).expect("build plan on sorted grid");
        common::assert_kernel_broadens(&pre_plan, &energies);
    }
    let spectrum: Vec<f64> = energies
        .iter()
        .map(|&e| {
            1.0 - 0.8 * (-((e - 7.8).powi(2) / 0.01)).exp()
                - 0.6 * (-((e - 22.4).powi(2) / 0.1)).exp()
        })
        .collect();

    let repeats = 30;

    let start = std::time::Instant::now();
    let mut sink_ref = 0.0f64;
    for _ in 0..repeats {
        let r = test_support::broaden_presorted_reference(&tab, &energies, &spectrum);
        sink_ref += r.iter().sum::<f64>();
    }
    let t_ref = start.elapsed();

    let start = std::time::Instant::now();
    let mut sink_new = 0.0f64;
    for _ in 0..repeats {
        let r = test_support::broaden_presorted(&tab, &energies, &spectrum);
        sink_new += r.iter().sum::<f64>();
    }
    let t_new = start.elapsed();

    let speedup = t_ref.as_secs_f64() / t_new.as_secs_f64();
    eprintln!(
        "broaden_presorted microbench (n_grid={n}, repeats={repeats}, synthetic kernel):\n\
         reference (binary search): {t_ref:?}  (sink={sink_ref:.3})\n\
         two-pointer walk         : {t_new:?}  (sink={sink_new:.3})\n\
         speedup                  : {speedup:.2}x"
    );
    assert_eq!(sink_ref.to_bits(), sink_new.to_bits());
}

/// Plan-reuse path vs per-call `broaden_presorted`.  The payoff
/// `plan()` + `ResolutionPlan::apply()` is designed to deliver: when
/// broadening many spectra on the same target grid (LM iterations
/// with fixed TZERO, spatial maps with pre-calibrated energies),
/// building the plan once and applying it N times beats rebuilding
/// the plan internally on every call.
#[test]
fn test_plan_reuse_bench() {
    let tab = common::synthetic_venus_usr_tab();

    let n = 3471;
    let energies: Vec<f64> = (0..n)
        .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
        .collect();
    {
        // No-op-regression pre-check: must precede the bit-exact
        // sink-equality assertion below, which would pass vacuously
        // if the kernel collapsed to a delta.
        let pre_plan = tab.plan(&energies).expect("build plan on sorted grid");
        common::assert_kernel_broadens(&pre_plan, &energies);
    }

    // Many spectra simulating an LM fit's sequence of evaluations.
    let repeats = 100;
    let mut state: u64 = 0xA5A5_A5A5_DEAD_BEEF;
    let spectra: Vec<Vec<f64>> = (0..repeats)
        .map(|_| {
            energies
                .iter()
                .map(|&e| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let noise = ((state >> 33) as f64) / (u32::MAX as f64);
                    1.0 - 0.8 * (-((e - 7.8).powi(2) / 0.01)).exp() + 1e-3 * noise
                })
                .collect()
        })
        .collect();

    // Per-call path: same pipeline as today.  Build cost paid every call.
    let start = std::time::Instant::now();
    let mut sink_percall = 0.0f64;
    for spec in &spectra {
        let r = test_support::broaden_presorted(&tab, &energies, spec);
        sink_percall += r.iter().sum::<f64>();
    }
    let t_percall = start.elapsed();

    // Plan-reuse path: one build, many applies.
    let start = std::time::Instant::now();
    let plan = tab.plan(&energies).expect("sorted grid must validate");
    let t_build = start.elapsed();
    let mut sink_plan = 0.0f64;
    for spec in &spectra {
        let r = plan.apply(spec);
        sink_plan += r.iter().sum::<f64>();
    }
    let t_apply_total = start.elapsed() - t_build;

    let speedup = t_percall.as_secs_f64() / (t_build + t_apply_total).as_secs_f64();
    eprintln!(
        "plan-reuse microbench (n_grid={n}, {repeats} spectra, synthetic kernel):\n\
         per-call broaden_presorted : {t_percall:?}  (sink={sink_percall:.3})\n\
         plan build (once)          : {t_build:?}\n\
         apply × {repeats}          : {t_apply_total:?}\n\
         total plan path            : {:?}  (sink={sink_plan:.3})\n\
         speedup vs per-call        : {speedup:.2}x",
        t_build + t_apply_total,
    );
    assert_eq!(sink_percall.to_bits(), sink_plan.to_bits());
}

/// `apply_r` (ResolutionMatrix CSR) vs `ResolutionPlan::apply`,
/// 3471-bin production grid × 100 spectra.  Exercised manually to
/// decide whether the CSR compile + CSR matvec beats the plan's
/// two-pointer walk at the no-SIMD-no-unsafe baseline promised in
/// #473.
#[test]
fn resolution_matrix_apply_microbench() {
    let tab = common::synthetic_venus_usr_tab();

    let n = 3471_usize;
    let energies: Vec<f64> = (0..n)
        .map(|i| 7.0 + i as f64 * ((200.0 - 7.0) / (n - 1) as f64))
        .collect();
    let plan = tab.plan(&energies).expect("sorted grid must validate");
    common::assert_kernel_broadens(&plan, &energies);

    let t_compile = std::time::Instant::now();
    let matrix = plan.compile_to_matrix();
    let t_compile = t_compile.elapsed();

    let spec: Vec<f64> = energies
        .iter()
        .map(|&e| {
            let sigma = 50.0 * (-((e - 80.0).powi(2)) / 8.0).exp()
                + 10.0 * (-((e - 150.0).powi(2)) / 4.0).exp();
            (-1.6e-4 * sigma).exp()
        })
        .collect();

    let repeats = 100_usize;

    // Warm both paths so the first call's cache-miss latency does
    // not skew the micro-times.
    for _ in 0..5 {
        let _ = plan.apply(&spec);
        let _ = apply_r(&matrix, &spec);
    }

    let start = std::time::Instant::now();
    let mut sink_plan = 0.0f64;
    for _ in 0..repeats {
        sink_plan += plan.apply(&spec).iter().sum::<f64>();
    }
    let t_plan = start.elapsed();

    let start = std::time::Instant::now();
    let mut sink_matrix = 0.0f64;
    for _ in 0..repeats {
        sink_matrix += apply_r(&matrix, &spec).iter().sum::<f64>();
    }
    let t_matrix = start.elapsed();

    let speedup = t_plan.as_secs_f64() / t_matrix.as_secs_f64();
    eprintln!(
        "ResolutionMatrix microbench (n_grid={n}, {repeats} spectra):\n\
         compile (once)       : {:?}  ({} nnz)\n\
         plan.apply × {repeats} : {:?}\n\
         apply_r   × {repeats} : {:?}\n\
         speedup vs plan      : {:.2}x\n\
         sinks (plan/matrix)  : {:.6e} / {:.6e}",
        t_compile,
        matrix.nnz(),
        t_plan,
        t_matrix,
        speedup,
        sink_plan,
        sink_matrix,
    );
}

// Helpers `interp_spectrum` + `broaden_presorted_reference` live in
// `test_support` so this microbench, the in-src tests, and
// `venus_usr_resolution.rs` share one byte-identical oracle.
