//! Common helpers for integration tests exercising the SAMMY USR
//! (tabulated user-supplied resolution) parser + broadening pipeline
//! on a synthetic VENUS-like kernel.
//!
//! # Why synthetic instead of the real VENUS fixture?
//!
//! The production VENUS BL10 kernel file
//! (`_fts_bl10_0p5meV_1keV_25pts.txt`) is gitignored per `.gitignore:49`
//! — ORNL has **not approved that file for public release**, so we
//! cannot vendor it under `tests/data/` to make CI exercise the real
//! kernel.
//!
//! Previously, a `venus_usr_resolution_path` helper returned
//! `Option<PathBuf>::None` whenever the gitignored file was absent
//! (i.e., on every CI run and every fresh checkout), and the 13
//! fixture-gated tests in this directory all early-returned via
//! `let Some(path) = … else { return; };`. The net effect was that
//! the SAMMY USR-format kernel path that the SoftwareX paper documents
//! had **zero CI coverage**, as confirmed empirically when issue #557
//! moved the fixture aside and saw all 13 tests still report
//! `passed; 0 ignored`.
//!
//! The fix is to generate a SAMMY USR-format kernel **in-memory** with
//! shape comparable to the production fixture (triangular kernels,
//! 25-point reference-energy grid spanning meV → keV) and parse it via
//! the same [`TabulatedResolution::from_text`] entry point the
//! production fixture uses. This exercises every stage of the
//! production pipeline (`from_text` parser → `plan` + `apply` →
//! `compile_to_matrix` + `apply_r`) on a kernel that is **non-trivially
//! broadening** (verified by [`assert_kernel_broadens`] — see PR #544
//! for the silent-no-op-regression-via-kernel-shrink failure mode this
//! pre-check guards against).
//!
//! Tests that previously used the early-return idiom now unconditionally
//! call [`synthetic_venus_usr_tab`] and assert real expectations on
//! every `cargo test` run.

use nereids_physics::resolution::{ResolutionPlan, TabulatedResolution};

/// Flight path (m) the synthetic kernel is built for. Matches the
/// `flight_path_m = 25.0` argument previously used with the real
/// VENUS BL10 fixture so the test grids (7 – 200 eV uniform) sit at
/// production-like TOF values.
pub const SYNTHETIC_FLIGHT_PATH_M: f64 = 25.0;

/// Half-width of the synthetic triangular kernel in TOF microseconds.
///
/// The energy-space half-width `ΔE = 2·E / TOF · w` at the lower edge
/// of the test grid (E = 7 eV, L = 25 m → TOF ≈ 683 µs) is only
/// ≈ 0.010 eV, well below the 512-point uniform 7 – 200 eV grid bin
/// width of ≈ 0.378 eV — i.e., the kernel is sub-bin at the low-energy
/// edge. Broadening still visibly perturbs the dip-shape probe in
/// [`assert_kernel_broadens`] because the probe is centred at the
/// middle of the grid (E ≈ 103 eV), where the energy-space half-width
/// grows like `E^(3/2)` and the kernel straddles several bins. The
/// 0.5 μs scale is the same order of magnitude as the production
/// VENUS kernel's central-lobe half-width.
const SYNTHETIC_HALF_WIDTH_US: f64 = 0.5;

/// Number of (offset, weight) samples per reference-energy block.
/// Matches the order-of-magnitude density of the production fixture
/// (which has ~500 samples per block over a wider TOF span; we use
/// fewer samples because the support here is narrower).
const SYNTHETIC_POINTS_PER_BLOCK: usize = 41;

/// Generate a SAMMY USR-format resolution-kernel text block with a
/// VENUS-like reference-energy grid and triangular kernels.
///
/// The returned string is byte-feedable to
/// [`TabulatedResolution::from_text`] — it follows the SAMMY USR
/// format exactly:
///
/// ```text
/// <header line>
/// <separator line>
/// <ref_energy_1>   0.000000000000000e+000
/// <dt_offset>      <weight>
/// <dt_offset>      <weight>
/// ...
/// <blank line>
/// <ref_energy_2>   0.000000000000000e+000
/// ...
/// ```
///
/// Reference-energy grid spans 5 meV to 1 keV in 25 log-spaced points
/// (mirroring the production file's `0p5meV_1keV_25pts` shape, except
/// we start at 5 meV instead of 0.5 meV — the test grids only probe
/// 7 – 200 eV, so the low-E padding is for parser-path coverage of
/// the bracketing-kernel interpolation logic, not for actual use).
///
/// Each block contains a symmetric triangular kernel of half-width
/// [`SYNTHETIC_HALF_WIDTH_US`] μs sampled at
/// [`SYNTHETIC_POINTS_PER_BLOCK`] points. The kernel is intentionally
/// identical across ref energies so the bracketing-kernel
/// log-interpolation path in [`TabulatedResolution::interpolated_kernel`]
/// is exercised on a stable shape (energy-dependent kernel widths
/// would test the same code paths but add unnecessary variability
/// to the equivalence-tolerance arithmetic).
pub fn synthetic_venus_usr_text() -> String {
    let mut out = String::new();
    out.push_str("SYNTHETIC VENUS-like USR kernel — triangular base 1us / FWHM 0.5us PSR\n");
    out.push_str("-----\n");

    // 25 log-spaced reference energies between 5e-3 eV and 1e3 eV.
    let n_ref = 25usize;
    let e_min = 5.0e-3_f64;
    let e_max = 1.0e3_f64;
    let log_e_min = e_min.ln();
    let log_e_max = e_max.ln();
    for i in 0..n_ref {
        let frac = (i as f64) / ((n_ref - 1) as f64);
        let e_ref = (log_e_min + frac * (log_e_max - log_e_min)).exp();
        // Ref-energy line: <E>   0.0  (the parser ignores the second
        // column on the first line of a block — see
        // `TabulatedResolution::from_text`, which captures only `x`
        // as `current_energy`; we emit `0.0` as a conventional
        // placeholder).
        out.push_str(&format!("{e_ref:.15e}   0.000000000000000e+000\n"));

        // Triangular kernel: weights w(dt) = max(0, 1 − |dt|/half_width).
        // Symmetric about dt = 0. Since `SYNTHETIC_POINTS_PER_BLOCK`
        // is odd (41), the inner loop emits the dt = 0 sample at
        // k = (n_pts − 1) / 2 in addition to the symmetric dt ≠ 0
        // pairs; the parser accepts it as an ordinary (offset,
        // weight) entry.
        let half = SYNTHETIC_HALF_WIDTH_US;
        let n_pts = SYNTHETIC_POINTS_PER_BLOCK;
        let dt_step = 2.0 * half / ((n_pts - 1) as f64);
        for k in 0..n_pts {
            let dt = -half + (k as f64) * dt_step;
            let w = (1.0 - dt.abs() / half).max(0.0);
            // SAMMY format: free-form whitespace-separated doubles.
            out.push_str(&format!("{dt:.15e}   {w:.15e}\n"));
        }
        // Block separator (blank line).
        out.push('\n');
    }

    out
}

/// Parse the synthetic kernel and return the live
/// [`TabulatedResolution`]. Goes through the production parser
/// [`TabulatedResolution::from_text`] so the parsing code path is
/// exercised on every test invocation.
///
/// Panics if the synthetic text is malformed — that would be a
/// generator bug, not a test failure.
pub fn synthetic_venus_usr_tab() -> TabulatedResolution {
    let text = synthetic_venus_usr_text();
    TabulatedResolution::from_text(&text, SYNTHETIC_FLIGHT_PATH_M)
        .expect("synthetic VENUS USR text must parse via from_text")
}

/// Pre-check (PR #544 pattern) that the synthetic kernel actually
/// broadens a probe spectrum. Guards against future tweaks to
/// [`SYNTHETIC_HALF_WIDTH_US`] or the grid that would silently turn
/// the kernel into a no-op (delta-kernel), which would let
/// equivalence-style tests pass vacuously while no longer exercising
/// the broadening math.
///
/// Asserts `‖T_kernel − T_none‖_∞ > THRESHOLD · ‖T_none‖_∞` where
/// `T_none` is the input probe spectrum and `T_kernel` is the same
/// spectrum after broadening with the supplied plan. The probe
/// spectrum is a sharp Gaussian dip — its post-broadening shape is
/// strictly different from the input unless the kernel collapses to
/// a delta, so the contrast bound below is a robust silent-no-op
/// detector.
///
/// Called from every test that broadens via `plan.apply` /
/// `compile_to_matrix` so a regression to the synthetic kernel
/// breaks loudly at the first test instead of letting an entire
/// `cargo test` run pass while exercising zero broadening code.
pub fn assert_kernel_broadens(plan: &ResolutionPlan, energies: &[f64]) {
    // Sharp Gaussian dip at the centre of the grid. The σ_E = 0.3 eV
    // dip is much narrower than the broadened lobe (which spans
    // ~ΔE = 2·E^(3/2)/(TOF_FACTOR·L) · 0.5 μs ≈ 0.4 eV at E = 100
    // eV) so broadening visibly reshapes the dip.
    let n = energies.len();
    assert!(n >= 8, "probe needs at least 8 grid points");
    let e_centre = 0.5 * (energies[0] + energies[n - 1]);
    let sigma_e = 0.3_f64;
    let probe: Vec<f64> = energies
        .iter()
        .map(|&e| 1.0 - 0.8 * (-((e - e_centre).powi(2)) / (2.0 * sigma_e * sigma_e)).exp())
        .collect();

    let broadened = plan.apply(&probe);
    assert_eq!(broadened.len(), probe.len(), "plan.apply length mismatch");

    let probe_inf = probe.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let diff_inf = probe
        .iter()
        .zip(broadened.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // Threshold: 1% of ‖probe‖_∞. The dip depth is 0.8 so this
    // requires the broadening to move at least one grid point by
    // ≥ 0.008 in absolute terms — a delta-kernel produces exactly
    // zero deviation, well below this floor.
    const THRESHOLD: f64 = 0.01;
    let floor = THRESHOLD * probe_inf;
    assert!(
        diff_inf > floor,
        "synthetic VENUS kernel is acting as a no-op on the probe \
         spectrum: ‖T_kernel − T_none‖_∞ = {diff_inf:.3e}, expected \
         > {floor:.3e} (= {THRESHOLD} · ‖T_none‖_∞ = {probe_inf:.3e}). \
         A future tweak to SYNTHETIC_HALF_WIDTH_US or the kernel shape \
         has collapsed the kernel toward a delta — every equivalence \
         test downstream will pass vacuously. See PR #544 for the \
         silent-no-op-regression failure mode this pre-check guards \
         against."
    );
}
