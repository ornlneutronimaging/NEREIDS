//! Sign-pinning regression for the tabulated-kernel convolution
//! orientation (issue #631).
//!
//! Kernels store their delayed-emission tail at POSITIVE TOF offsets
//! (mode at 0). A delayed neutron measured at TOF `t` really flew
//! `t − dt` — it is faster than nominal — so broadening a symmetric
//! absorption dip must push its tail toward later TOF = LOWER apparent
//! energy: the broadened dip's absorption-weighted centroid shift and
//! skew are both strictly NEGATIVE. The pre-fix code gathered at
//! `t + dt` (a correlation, time-mirroring every kernel), which put the
//! tail on the high-energy side; none of the earlier tests pinned the
//! sign (the kernel-array test never applied the kernel, and the IC
//! centering test took `.abs()` of the shift). SAMMY reference:
//! `sammy/src/udr/mudr4.f90` `Ud_Convolute` gathers theory at `Tc − τ`.
//!
//! The synthetic kernel goes through [`TabulatedResolution::from_text`]
//! so the parser is part of the pinned path, and the assertions run
//! through BOTH the direct path (`apply_resolution`) and the plan path
//! (`build_resolution_plan` + `apply_resolution_with_plan`).

use std::fmt::Write as _;
use std::sync::Arc;

use nereids_physics::resolution::{
    ResolutionFunction, TabulatedResolution, apply_resolution, apply_resolution_with_plan,
    build_resolution_plan,
};

const FLIGHT_PATH_M: f64 = 25.0;
const E0: f64 = 20.0;

/// Synthetic asymmetric kernel in the VENUS/FTS text format: mode at
/// offset 0, exponential tail at positive offsets only (identical
/// blocks at 10 and 100 eV so `interpolated_kernel` blending is exact).
fn asymmetric_kernel_text() -> String {
    let n = 499;
    let (dt_min, dt_max) = (-1.0_f64, 15.0_f64);
    let mut text =
        String::from("synthetic asymmetric kernel, tail at positive TOF offsets\n-----\n");
    for eref in [10.0_f64, 100.0] {
        writeln!(text, "   {eref:.5e}   0.00000e+000").unwrap();
        let mut nearest = (f64::MAX, 0usize);
        let dts: Vec<f64> = (0..n)
            .map(|i| dt_min + (dt_max - dt_min) * i as f64 / (n - 1) as f64)
            .collect();
        for (i, &dt) in dts.iter().enumerate() {
            if dt.abs() < nearest.0 {
                nearest = (dt.abs(), i);
            }
        }
        for (i, &dt) in dts.iter().enumerate() {
            let amp = if i == nearest.1 {
                1.0
            } else if dt >= 0.0 {
                (-dt / 3.0).exp()
            } else {
                0.0
            };
            writeln!(text, "{dt:.15} {amp:.15e}").unwrap();
        }
        text.push('\n');
    }
    text
}

/// Absorption-weighted centroid shift and skew of a broadened
/// symmetric dip at `E0`.
fn dip_moments(broadened: &[f64], energies: &[f64]) -> (f64, f64) {
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
    let sd = var.sqrt();
    let skew = energies
        .iter()
        .zip(&absorption)
        .map(|(&e, &a)| ((e - mu) / sd).powi(3) * a)
        .sum::<f64>()
        / total;
    (mu - E0, skew)
}

fn grid_and_dip() -> (Vec<f64>, Vec<f64>) {
    let n = 8001;
    let energies: Vec<f64> = (0..n)
        .map(|i| (E0 - 3.0) + 6.0 * i as f64 / (n - 1) as f64)
        .collect();
    let spectrum: Vec<f64> = energies
        .iter()
        .map(|&e| 1.0 - 0.6 * (-0.5 * ((e - E0) / 0.02).powi(2)).exp())
        .collect();
    (energies, spectrum)
}

#[test]
fn delayed_tail_shifts_broadened_dip_to_lower_energy() {
    let tab = TabulatedResolution::from_text(&asymmetric_kernel_text(), FLIGHT_PATH_M)
        .expect("synthetic kernel must parse");
    let (energies, spectrum) = grid_and_dip();

    let res = ResolutionFunction::Tabulated(Arc::new(tab));

    // Direct path.
    let direct = apply_resolution(&energies, &spectrum, &res).expect("direct broaden");

    // Non-vacuity pre-check (same guard idea as
    // `common::assert_kernel_broadens` in the VENUS USR tests): the
    // kernel must visibly reshape the dip, or the sign assertions
    // below pass vacuously on a no-op broadener.
    let probe_inf = spectrum.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let diff_inf = spectrum
        .iter()
        .zip(&direct)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        diff_inf > 0.01 * probe_inf,
        "kernel is acting as a no-op on the probe dip: \
         ‖broadened − input‖∞ = {diff_inf:.3e} ≤ 1% of ‖input‖∞ = {probe_inf:.3e}"
    );

    let (shift, skew) = dip_moments(&direct, &energies);
    assert!(
        shift < 0.0,
        "centroid must shift to LOWER energy (delayed arrival): got {shift:+.4e} eV \
         — a positive shift means the kernel was applied time-mirrored"
    );
    assert!(
        skew < 0.0,
        "broadened dip must be skewed toward LOWER energy: got skew {skew:+.3e}"
    );

    // Plan path must agree in sign (bit-exactness with the direct path
    // is pinned elsewhere; this guards the orientation specifically).
    let plan = build_resolution_plan(&energies, &res)
        .expect("plan build")
        .expect("tabulated resolution must yield a plan");
    let planned =
        apply_resolution_with_plan(Some(&plan), &energies, &spectrum, &res).expect("plan broaden");
    let (shift_p, skew_p) = dip_moments(&planned, &energies);
    assert!(
        shift_p < 0.0 && skew_p < 0.0,
        "plan path orientation: shift {shift_p:+.4e}, skew {skew_p:+.3e}"
    );
}
