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

/// Where a window's points sit: a grid uniform in energy, or the instrument's
/// own time-of-flight channels. Two windows cut from the same lattice share
/// their common points exactly.
#[derive(Clone, Copy)]
enum Lattice {
    Energy { step_ev: f64 },
    TimeOfFlight { channel_us: f64 },
}

impl Lattice {
    /// Every lattice point inside `[lo, hi]`, ascending.
    fn points(self, lo: f64, hi: f64) -> Vec<f64> {
        const SNAP: f64 = 1.0e-9;
        match self {
            Lattice::Energy { step_ev } => {
                let k0 = (lo / step_ev - SNAP).ceil() as i64;
                let k1 = (hi / step_ev + SNAP).floor() as i64;
                (k0..=k1).map(|k| k as f64 * step_ev).collect()
            }
            Lattice::TimeOfFlight { channel_us } => {
                let tof = |e: f64| TOF_FACTOR * FLIGHT_PATH_M / e.sqrt();
                let k_hi = (tof(lo) / channel_us + SNAP).floor() as i64;
                let k_lo = (tof(hi) / channel_us - SNAP).ceil() as i64;
                (k_lo..=k_hi)
                    .rev()
                    .map(|k| (TOF_FACTOR * FLIGHT_PATH_M / (k as f64 * channel_us)).powi(2))
                    .collect()
            }
        }
    }
}

struct Fixture {
    label: &'static str,
    resolution: ResolutionFunction,
    lattice: Lattice,
    e_lo: f64,
    e_hi: f64,
}

fn ikeda_carpenter(beta: f64, e_lo: f64, e_hi: f64) -> ResolutionFunction {
    let params = IkedaCarpenterParams {
        alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
        beta: EnergyLaw::Const(beta),
        r: EnergyLaw::Const(0.15),
        burst_sigma_us: None,
        channel_fwhm_us: Some(0.35),
    };
    // References bracket the window, as the calibration's own synthesis
    // grid does, so every data energy lies between two of them.
    ResolutionFunction::IkedaCarpenter(std::sync::Arc::new(
        IkedaCarpenter::new(
            params,
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: e_lo * 0.5,
                e_max_ev: e_hi * 2.0,
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

/// The three kernel regimes the working grid has to serve.
///
/// A fast moderator on a fine energy grid, where every kernel point is well
/// inside the flight time; a mode-anchored table that gathers from both
/// sides; and a slow moderator at high energy on the instrument's own
/// channels, where the kernel's late tail runs past the flight time and the
/// last surviving point gathers from far above the window.
fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            label: "ikeda-carpenter",
            resolution: ikeda_carpenter(0.25, 5.0, 9.0),
            lattice: Lattice::Energy { step_ev: 0.01 },
            e_lo: 5.0,
            e_hi: 9.0,
        },
        Fixture {
            label: "tabulated",
            resolution: tabulated_resolution(),
            lattice: Lattice::Energy { step_ev: 0.01 },
            e_lo: 5.0,
            e_hi: 9.0,
        },
        Fixture {
            label: "ikeda-carpenter past the flight time",
            resolution: ikeda_carpenter(0.05, 1000.0, 2000.0),
            lattice: Lattice::TimeOfFlight { channel_us: 0.5 },
            e_lo: 1000.0,
            e_hi: 2000.0,
        },
    ]
}

fn transmission(energies: &[f64], e_resonance: f64, resolution: &ResolutionFunction) -> Vec<f64> {
    // A resonance placed AT the window edge: the case the truncation hurts
    // most, because the kernel there reaches entirely outside the data.
    let iso = synthetic_isotope(72, 178, e_resonance, 0.05, 0.06);
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
    for f in fixtures() {
        let narrow = f.lattice.points(f.e_lo, f.e_hi);
        let n = narrow.len();
        let (below, above) = f.resolution.boundary_reach_ev(&narrow);
        assert!(
            above > 10.0 * (narrow[n - 1] - narrow[n - 2]),
            "{}: the kernel must reach past several grid steps above for this to test \
             anything, got {above}",
            f.label
        );
        let wide = f.lattice.points(
            (f.e_lo - 2.0 * below).min(0.9 * f.e_lo).max(0.25 * f.e_lo),
            (f.e_hi + 2.0 * above).max(1.1 * f.e_hi),
        );
        let first = wide.partition_point(|&e| e < narrow[0] * (1.0 - 1.0e-12));
        assert!(
            (wide[first] - narrow[0]).abs() <= 1.0e-12 * narrow[0],
            "{}: the wide window must contain the narrow one's points",
            f.label
        );

        let t_narrow = transmission(&narrow, f.e_hi, &f.resolution);
        let t_wide = transmission(&wide, f.e_hi, &f.resolution);

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
            "{}: the resonance must stay transparent enough for a moved value to show, \
             got a minimum transmission of {t_min}",
            f.label
        );
        moved.push((f.label, worst, worst_at));
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
    for f in fixtures() {
        let data = f.lattice.points(f.e_lo, f.e_hi);
        let layout = nereids_physics::transmission::resolution_working_grid(
            &data,
            Some(&InstrumentParams {
                resolution: f.resolution.clone(),
            }),
            &[],
        )
        .expect("working grid");
        assert!(
            !layout.is_identity(),
            "{}: a kernel that reaches past the data must get a wider working grid",
            f.label
        );
        let work = &layout.energies;
        // The grid ends where the last surviving kernel point gathers from,
        // so a point on the boundary is inside up to rounding.
        let (w_lo, w_hi) = (
            work[0] * (1.0 - 1.0e-12),
            work[work.len() - 1] * (1.0 + 1.0e-12),
        );

        let table = match &f.resolution {
            ResolutionFunction::Tabulated(t) => std::sync::Arc::clone(t),
            ResolutionFunction::IkedaCarpenter(ic) => std::sync::Arc::new(ic.tabulated().clone()),
            ResolutionFunction::Gaussian(_) => unreachable!(),
        };
        let mut dropped_past_flight_time = 0.0_f64;
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
                if w <= 0.0 {
                    continue;
                }
                let mass = w * width(k).abs();
                if dt >= t {
                    dropped_past_flight_time += mass;
                    continue;
                }
                total += mass;
                let e_prime = e * (t / (t - dt)).powi(2);
                if e_prime >= w_lo && e_prime <= w_hi {
                    inside += mass;
                }
            }
            assert!(total > 0.0, "{}: no kernel mass at {e} eV", f.label);
            assert!(
                inside >= RETAINED_KERNEL_MASS * total,
                "{}: at {e} eV the working grid [{w_lo}, {w_hi}] carries {:.6} of the \
                 kernel's mass, below {RETAINED_KERNEL_MASS}",
                f.label,
                inside / total
            );
        }
        if f.label.contains("past the flight time") {
            assert!(
                dropped_past_flight_time > 0.0,
                "{}: this fixture must put kernel points past the flight time",
                f.label
            );
        }
    }
}
