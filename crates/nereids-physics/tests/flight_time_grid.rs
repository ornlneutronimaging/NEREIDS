use std::sync::Arc;

use nereids_physics::flight_time_grid::{FlightTimeGrid, FlightTimeGridError};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, NEGLIGIBLE_ARRIVAL_PROBABILITY, SynthesisGrid,
};
use nereids_physics::resolution::{
    ResolutionFunction, ResolutionParams, TOF_FACTOR, TabulatedResolution,
};

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;

fn edges() -> Vec<f64> {
    (350..=470).map(f64::from).collect()
}

fn energy(flight_time_us: f64) -> f64 {
    (TOF_FACTOR * FLIGHT_PATH_M / flight_time_us).powi(2)
}

fn ikeda_carpenter(
    alpha: EnergyLaw,
    beta: f64,
    r: f64,
    channel_fwhm_us: Option<f64>,
) -> ResolutionFunction {
    ResolutionFunction::IkedaCarpenter(Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha,
                beta: EnergyLaw::Const(beta),
                r: EnergyLaw::Const(r),
                burst_sigma_us: None,
                channel_fwhm_us,
            },
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 200.0,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model"),
    ))
}

fn table(energies: Vec<f64>, kernels: &[&[(f64, f64)]]) -> ResolutionFunction {
    let kernels = kernels
        .iter()
        .map(|k| {
            (
                k.iter().map(|p| p.0).collect(),
                k.iter().map(|p| p.1).collect(),
            )
        })
        .collect();
    ResolutionFunction::Tabulated(Arc::new(
        TabulatedResolution::from_kernels(energies, kernels, FLIGHT_PATH_M).expect("valid table"),
    ))
}

const TRIANGLE: &[(f64, f64)] = &[(-1.0, 0.0), (0.0, 1.0), (3.0, 0.0)];
const LATE: &[(f64, f64)] = &[(2.0, 0.0), (3.0, 1.0), (5.0, 0.0)];

fn window_probability(resolution: &ResolutionFunction, flight_time_us: f64) -> f64 {
    resolution
        .detector_bin_probabilities(energy(flight_time_us), &edges(), T0_US)
        .expect("probabilities")
        .iter()
        .sum()
}

#[test]
fn the_range_holds_every_flight_time_whose_neutrons_reach_a_bin() {
    let pulses = [
        (
            "constant IC",
            ikeda_carpenter(EnergyLaw::Const(0.565), 0.25, 0.15, None),
            true,
        ),
        (
            "IC with an energy law",
            ikeda_carpenter(EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 }, 0.25, 0.15, None),
            true,
        ),
        (
            "IC with a channel fold",
            ikeda_carpenter(EnergyLaw::Const(0.565), 0.25, 0.15, Some(2.0)),
            false,
        ),
        (
            "triangle",
            table(vec![1.0, 200.0], &[TRIANGLE, TRIANGLE]),
            false,
        ),
        ("late table", table(vec![1.0, 200.0], &[LATE, LATE]), false),
    ];
    let nudge = 1e-6;
    for (name, pulse, analytic_tail) in pulses {
        let grid = FlightTimeGrid::new(&edges(), T0_US, &pulse).expect(name);
        let (u_lo, u_hi) = grid.range_us();
        assert_eq!(
            window_probability(&pulse, u_hi * (1.0 + nudge)),
            0.0,
            "{name}"
        );
        assert!(
            window_probability(&pulse, u_hi * (1.0 - nudge)) > 0.0,
            "{name}"
        );
        let before = window_probability(&pulse, u_lo * (1.0 - nudge));
        let after = window_probability(&pulse, u_lo * (1.0 + nudge));
        if analytic_tail {
            assert!(before < NEGLIGIBLE_ARRIVAL_PROBABILITY, "{name}: {before}");
            assert!(after >= NEGLIGIBLE_ARRIVAL_PROBABILITY, "{name}: {after}");
        } else {
            assert_eq!(before, 0.0, "{name}");
            assert!(after > 0.0, "{name}");
        }
    }
}

#[test]
fn the_step_resolves_the_narrowest_bin_and_the_pulse_rise() {
    let grid = FlightTimeGrid::new(
        &edges(),
        T0_US,
        &table(vec![1.0, 200.0], &[TRIANGLE, TRIANGLE]),
    )
    .unwrap();
    assert!(grid.step_us() <= 0.5);
    let narrow: Vec<f64> = (0..=240).map(|i| 350.0 + 0.25 * f64::from(i)).collect();
    let fine = FlightTimeGrid::new(
        &narrow,
        T0_US,
        &table(vec![1.0, 200.0], &[TRIANGLE, TRIANGLE]),
    )
    .unwrap();
    assert!(fine.step_us() <= 0.25);
    let sharp: &[(f64, f64)] = &[(-0.2, 0.0), (0.0, 1.0), (3.0, 0.0)];
    let sharp_inside = FlightTimeGrid::new(
        &edges(),
        T0_US,
        &table(
            vec![1.0, 18.0, 20.0, 22.0, 200.0],
            &[TRIANGLE, TRIANGLE, sharp, TRIANGLE, TRIANGLE],
        ),
    )
    .unwrap();
    assert!(sharp_inside.step_us() <= 0.1);
    assert!((grid.halved().unwrap().step_us() - 0.5 * grid.step_us()).abs() < 1e-12);
}

#[test]
fn windows_the_grid_cannot_describe_are_refused() {
    let triangle = table(vec![1.0, 200.0], &[TRIANGLE, TRIANGLE]);
    let refused = |edges: &[f64], t0: f64, pulse: &ResolutionFunction| {
        FlightTimeGrid::new(edges, t0, pulse).expect_err("refused")
    };
    let gaussian =
        ResolutionFunction::Gaussian(ResolutionParams::new(FLIGHT_PATH_M, 1.0, 0.01, 0.0).unwrap());
    assert!(matches!(
        refused(&edges(), T0_US, &gaussian),
        FlightTimeGridError::Resolution(_)
    ));
    let single = table(vec![1.0, 200.0], &[&[(0.0, 1.0)], &[(0.0, 1.0)]]);
    assert!(matches!(
        refused(&edges(), T0_US, &single),
        FlightTimeGridError::NoRise { .. }
    ));
    let slow_tail = ikeda_carpenter(EnergyLaw::Const(0.565), 0.02, 0.05, None);
    assert!(matches!(
        refused(&edges(), T0_US, &slow_tail),
        FlightTimeGridError::OutsideCalibration { .. }
    ));
    let fast_tail: &[(f64, f64)] = &[(-1.0, 0.0), (0.0, 1.0), (300.0, 0.0)];
    let (band_fast, band_slow) = (energy(148.0), energy(148.75));
    let centre = (band_fast * band_slow).sqrt();
    let stray = table(
        vec![
            1.0,
            band_slow.powi(2) / centre,
            centre,
            band_fast.powi(2) / centre,
            200.0,
        ],
        &[TRIANGLE, TRIANGLE, fast_tail, TRIANGLE, TRIANGLE],
    );
    assert!(matches!(
        refused(&edges(), T0_US, &stray),
        FlightTimeGridError::ReachedOutsideRange { .. }
    ));
    let tiny_bins: Vec<f64> = (0..=120_000).map(|i| 350.0 + 1e-3 * f64::from(i)).collect();
    assert!(matches!(
        refused(&tiny_bins, T0_US, &triangle),
        FlightTimeGridError::TooManyPoints(_)
    ));
    assert!(matches!(
        refused(&[350.0, 349.0], T0_US, &triangle),
        FlightTimeGridError::InvalidTimeEdges
    ));
    let past_slowest: Vec<f64> = (350..=2000).map(f64::from).collect();
    for (edges, t0) in [(past_slowest, T0_US), (edges(), 350.0)] {
        assert!(matches!(
            refused(&edges, t0, &triangle),
            FlightTimeGridError::OutsideCalibration { .. }
        ));
    }
    assert!(matches!(
        refused(&edges(), f64::NAN, &triangle),
        FlightTimeGridError::InvalidTimingOffset(_)
    ));
}
