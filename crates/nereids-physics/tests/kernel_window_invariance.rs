//! A broadened value must not depend on where the data window ends.
//!
//! The broadened spectrum at a target energy is built from theory within the
//! kernel's reach of that energy. Whether the caller happened to stop
//! collecting data just past it is not a property of the sample, so widening
//! the window must leave the shared targets alone.

use nereids_endf::resonance::test_support::synthetic_isotope;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    RETAINED_KERNEL_MASS, ResolutionFunction, TOF_FACTOR, TabulatedResolution, test_support,
};
use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};

const FLIGHT_PATH_M: f64 = 25.0;
const E_LO: f64 = 5.0;
const E_HI: f64 = 9.0;
const STEP: f64 = 0.01;

/// The VENUS-like moderator the pipeline's own joint-recovery fixture uses.
fn ic_resolution() -> ResolutionFunction {
    let params = IkedaCarpenterParams {
        alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
        beta: EnergyLaw::Const(0.25),
        r: EnergyLaw::Const(0.15),
        burst_sigma_us: None,
        channel_fwhm_us: Some(0.35),
    };
    ResolutionFunction::IkedaCarpenter(std::sync::Arc::new(
        IkedaCarpenter::new(
            params,
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: E_LO * 0.5,
                e_max_ev: 64.0,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model"),
    ))
}

/// A mode-anchored kernel with mass on both sides of zero, so it gathers
/// from below as well as above, at two reference energies bracketing the
/// window.
fn tabulated_resolution() -> ResolutionFunction {
    let text = "header\n---\n\
        3.0 0.0\n\
        -1.5 0.0\n\
        -1.0 0.5\n\
        0.0 1.0\n\
        2.0 0.6\n\
        4.0 0.2\n\
        5.0 0.0\n\
        \n\
        12.0 0.0\n\
        -1.0 0.0\n\
        -0.6 0.5\n\
        0.0 1.0\n\
        1.2 0.6\n\
        2.4 0.2\n\
        3.0 0.0\n";
    ResolutionFunction::Tabulated(std::sync::Arc::new(
        TabulatedResolution::from_text(text, FLIGHT_PATH_M).expect("valid kernel text"),
    ))
}

fn grid(e_hi: f64) -> Vec<f64> {
    let n = ((e_hi - E_LO) / STEP).round() as usize + 1;
    (0..n).map(|i| E_LO + i as f64 * STEP).collect()
}

fn transmission(energies: &[f64], resolution: &ResolutionFunction) -> Vec<f64> {
    // A resonance placed AT the window edge: the case the truncation hurts
    // most, because the kernel there reaches entirely outside the data.
    let iso = synthetic_isotope(72, 178, E_HI, 0.05, 0.06);
    let sample = SampleParams::new(300.0, vec![(iso, 1.2e-5)]).expect("valid sample");
    forward_model(
        energies,
        &sample,
        Some(&InstrumentParams {
            resolution: resolution.clone(),
        }),
    )
    .expect("forward model")
}

/// Extending the window past the kernel's reach leaves the shared targets
/// unchanged, whichever way the kernel reaches.
///
/// The Ikeda-Carpenter kernel is causal on the emission clock, so its offsets
/// are positive and it gathers theory from HIGHER energy only; a mode-anchored
/// table straddles zero and gathers from both sides. Without a boundary
/// extension the edge of the window has nothing to gather on that side and
/// the surviving fragment is renormalized to full weight, so the model there
/// is built from a kernel that is narrower, lighter and mis-centred.
#[test]
fn a_wider_window_does_not_change_the_model_inside_it() {
    let mut moved = Vec::new();
    for (label, resolution) in [
        ("ikeda-carpenter", ic_resolution()),
        ("tabulated", tabulated_resolution()),
    ] {
        let narrow = grid(E_HI);
        let (below, above) = resolution.boundary_reach_ev(&narrow);
        assert!(
            above > 10.0 * STEP,
            "{label}: the kernel must reach past several grid steps above for this to test \
             anything, got {above}"
        );
        // Both windows must sit on the same STEP lattice, or the shared
        // targets are compared against neighbours rather than themselves.
        let e_lo = E_LO - STEP * (2.0 * below.max(10.0 * STEP) / STEP).ceil();
        let e_hi = E_HI + STEP * (2.0 * above / STEP).ceil();
        let n = ((e_hi - e_lo) / STEP).round() as usize + 1;
        let wide: Vec<f64> = (0..n).map(|i| e_lo + i as f64 * STEP).collect();
        let first = wide.partition_point(|&e| e < E_LO - 0.5 * STEP);

        let t_narrow = transmission(&narrow, &resolution);
        let t_wide = transmission(&wide, &resolution);

        // Transmission is the observable and lives in [0, 1], so the
        // difference is taken as it is measured, not relative to a value
        // that a black resonance drives to zero.
        let mut worst = 0.0_f64;
        let mut worst_at = 0.0_f64;
        for (i, &e) in narrow.iter().enumerate() {
            let d = (t_narrow[i] - t_wide[first + i]).abs();
            if d > worst {
                worst = d;
                worst_at = e;
            }
        }
        let t_min = t_narrow.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            t_min > 0.2,
            "{label}: the resonance must stay transparent enough for a moved value to show, \
             got a minimum transmission of {t_min}"
        );
        moved.push((label, worst, worst_at));
    }
    // The two runs convolve on different point sets - the wide window
    // carries its own data where the narrow one carries extension points -
    // so what is left is the trapezoid's own discretization, not lost
    // kernel mass. Truncation shows up here orders of magnitude above it.
    let moved: Vec<String> = moved
        .into_iter()
        .filter(|&(_, worst, _)| worst >= 1.0e-4)
        .map(|(label, worst, at)| format!("{label} moved by {worst:.3e} at {at} eV"))
        .collect();
    assert!(
        moved.is_empty(),
        "the model changed when the window was widened ({}); the value at a target must \
         come from the kernel's reach, not from where the data happens to stop",
        moved.join("; ")
    );
}

/// The working grid carries essentially all of the kernel's mass at every
/// data point.
///
/// The oracle maps the kernel's own samples through the time-of-flight
/// relation and sums the trapezoid weight of those that land inside the
/// working grid; it does not ask the reach code where the kernel ends. A
/// kernel renormalized after losing mass is a different kernel from the one
/// the file describes, and what the grid must guarantee is that the loss is
/// below what the quadrature can resolve.
#[test]
fn the_working_grid_carries_the_whole_kernel_at_every_data_point() {
    for (label, resolution) in [
        ("ikeda-carpenter", ic_resolution()),
        ("tabulated", tabulated_resolution()),
    ] {
        let data = grid(E_HI);
        let layout = nereids_physics::transmission::resolution_working_grid(
            &data,
            Some(&InstrumentParams {
                resolution: resolution.clone(),
            }),
            &[],
        )
        .expect("working grid");
        assert!(
            !layout.is_identity(),
            "{label}: a kernel that reaches past the data must get a wider working grid"
        );
        let work = &layout.energies;
        let (w_lo, w_hi) = (work[0], work[work.len() - 1]);

        let table = match &resolution {
            ResolutionFunction::Tabulated(t) => std::sync::Arc::clone(t),
            ResolutionFunction::IkedaCarpenter(ic) => std::sync::Arc::new(ic.tabulated().clone()),
            ResolutionFunction::Gaussian(_) => unreachable!(),
        };
        for &e in &data {
            let (offsets, weights) = test_support::interpolated_kernel(&table, e);
            let t = TOF_FACTOR * FLIGHT_PATH_M / e.sqrt();
            let n_k = offsets.len();
            let width = |k: usize| -> f64 {
                if n_k <= 1 {
                    1.0
                } else if k == 0 {
                    offsets[1] - offsets[0]
                } else if k == n_k - 1 {
                    offsets[k] - offsets[k - 1]
                } else {
                    (offsets[k + 1] - offsets[k - 1]) * 0.5
                }
            };
            let (mut total, mut inside) = (0.0_f64, 0.0_f64);
            for k in 0..n_k {
                let (dt, w) = (offsets[k], weights[k]);
                if w <= 0.0 || dt >= t {
                    continue;
                }
                let mass = w * width(k).abs();
                total += mass;
                let e_prime = e * (t / (t - dt)).powi(2);
                if e_prime >= w_lo && e_prime <= w_hi {
                    inside += mass;
                }
            }
            assert!(total > 0.0, "{label}: no kernel mass at {e} eV");
            assert!(
                inside >= RETAINED_KERNEL_MASS * total,
                "{label}: at {e} eV the working grid [{w_lo}, {w_hi}] carries {:.6} of the \
                 kernel's mass, below {RETAINED_KERNEL_MASS}",
                inside / total
            );
        }
    }
}
