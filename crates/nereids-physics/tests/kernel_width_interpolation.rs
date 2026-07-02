//! End-to-end width oracle for between-reference kernel interpolation
//! (issue #632).
//!
//! Symmetric Gaussian kernels (mode 0, so the convolution orientation
//! does not contaminate the width measurement) at reference energies
//! 10 and 50 eV, with TOF-space widths following the physical
//! σ_t ∝ E^{−1/2} law: σ_t(10 eV) = 2.0 µs, σ_t(50 eV) = 2/√5 µs. A
//! width-correct interpolation at 20 eV must give σ_t = 2/√2 =
//! 1.4142 µs. The pre-fix element-wise blend produced the arithmetic
//! chord 1.523 µs (+7.8 %) — the width surplus measured at +4.1…+7.2 %
//! across the production VENUS file's 5→50 eV reference gap.
//!
//! The fixture goes through [`TabulatedResolution::from_text`] and the
//! width is measured on the APPLIED broadening (a near-delta dip's
//! absorption-weighted σ_E mapped back to TOF space), so the whole
//! parse → interpolate → convolve chain is the system under test —
//! non-circular with respect to the interpolation algorithm.

use std::fmt::Write as _;
use std::sync::Arc;

use nereids_physics::resolution::{
    ResolutionFunction, TOF_FACTOR, TabulatedResolution, apply_resolution,
    apply_resolution_with_plan, build_resolution_plan,
};
const FLIGHT_PATH_M: f64 = 25.0;
const SIGMA_10EV_US: f64 = 2.0;

/// Two-block Gaussian fixture in the VENUS/FTS text format with exact
/// σ_t ∝ E^{−1/2} widths at 10 and 50 eV.
fn gaussian_kernel_text() -> String {
    let mut text = String::from("synthetic Gaussian kernels, sigma_t ~ E^-1/2\n-----\n");
    for eref in [10.0f64, 50.0] {
        let sigma = SIGMA_10EV_US * (eref / 10.0).powf(-0.5);
        writeln!(text, "   {eref:.5e}   0.00000e+000").unwrap();
        let n = 499;
        for i in 0..n {
            let d = -6.0 * sigma + 12.0 * sigma * i as f64 / (n - 1) as f64;
            let a = (-0.5 * (d / sigma).powi(2)).exp();
            writeln!(text, "{d:.15} {a:.15e}").unwrap();
        }
        text.push('\n');
    }
    text
}

/// Broaden a near-delta dip at `e0` and return the measured TOF-space
/// width of the applied kernel.
fn measured_sigma_t(tab: &TabulatedResolution, e0: f64, use_plan: bool) -> f64 {
    let n = 30_001;
    let energies: Vec<f64> = (0..n)
        .map(|i| e0 * 0.7 + e0 * 0.6 * i as f64 / (n - 1) as f64)
        .collect();
    let spectrum: Vec<f64> = energies
        .iter()
        .map(|&e| 1.0 - 0.8 * (-0.5 * ((e - e0) / (e0 * 1e-4)).powi(2)).exp())
        .collect();
    let res = ResolutionFunction::Tabulated(Arc::new(tab.clone()));
    let broadened = if use_plan {
        let plan = build_resolution_plan(&energies, &res)
            .expect("plan build")
            .expect("tabulated resolution must yield a plan");
        apply_resolution_with_plan(Some(&plan), &energies, &spectrum, &res).expect("plan broaden")
    } else {
        apply_resolution(&energies, &spectrum, &res).expect("direct broaden")
    };

    // Non-vacuity: the kernel must visibly reshape the near-delta dip.
    let diff_inf = spectrum
        .iter()
        .zip(&broadened)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        diff_inf > 0.01,
        "kernel is acting as a no-op at {e0} eV: ‖broadened − input‖∞ = {diff_inf:.3e}"
    );

    let absorption: Vec<f64> = broadened.iter().map(|&t| (1.0 - t).max(0.0)).collect();
    let total: f64 = absorption.iter().sum();
    let mu = energies
        .iter()
        .zip(&absorption)
        .map(|(&e, &a)| e * a)
        .sum::<f64>()
        / total;
    let var = energies
        .iter()
        .zip(&absorption)
        .map(|(&e, &a)| (e - mu).powi(2) * a)
        .sum::<f64>()
        / total;
    let sigma_e = var.sqrt();
    // |dt/dE| = t/(2E) at e0 maps the energy-space width back to TOF.
    let tof = TOF_FACTOR * FLIGHT_PATH_M / e0.sqrt();
    sigma_e * tof / (2.0 * e0)
}

#[test]
fn applied_width_follows_power_law_at_and_between_references() {
    let tab = TabulatedResolution::from_text(&gaussian_kernel_text(), FLIGHT_PATH_M)
        .expect("synthetic Gaussian fixture must parse");

    for (e0, expected) in [
        (10.0, SIGMA_10EV_US),
        (50.0, SIGMA_10EV_US / 5.0f64.sqrt()),
        // The between-reference point the arithmetic chord missed by
        // +7.8 % — width-correct interpolation must hit the power law.
        (20.0, SIGMA_10EV_US / 2.0f64.sqrt()),
    ] {
        let got = measured_sigma_t(&tab, e0, false);
        let rel = (got / expected - 1.0).abs();
        assert!(
            rel < 0.01,
            "applied width at {e0} eV: measured σ_t = {got:.4} µs, power law \
             expects {expected:.4} µs ({:+.1} % — the pre-fix chord gave +7.8 % \
             at 20 eV)",
            (got / expected - 1.0) * 100.0
        );
    }

    // Plan path must agree with the direct path on the midpoint width.
    let direct = measured_sigma_t(&tab, 20.0, false);
    let planned = measured_sigma_t(&tab, 20.0, true);
    assert!(
        (direct - planned).abs() / direct < 1e-12,
        "plan and direct widths must agree: {direct} vs {planned}"
    );
}
