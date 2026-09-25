use std::sync::Arc;

use nereids_physics::flight_time_grid::{FlightTimeGrid, FlightTimeGridError};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, NEGLIGIBLE_ARRIVAL_PROBABILITY, SynthesisGrid,
};
use nereids_physics::resolution::TOF_FACTOR;

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const CLOCK: f64 = TOF_FACTOR * FLIGHT_PATH_M;
const E_MIN_EV: f64 = 1.0;
const E_MAX_EV: f64 = 200.0;

fn edges() -> Vec<f64> {
    (350..=470).map(f64::from).collect()
}

fn pulse(
    alpha: EnergyLaw,
    beta: EnergyLaw,
    r: EnergyLaw,
    burst_sigma_us: Option<f64>,
    channel_fwhm_us: Option<f64>,
) -> Arc<IkedaCarpenter> {
    Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha,
                beta,
                r,
                burst_sigma_us,
                channel_fwhm_us,
            },
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: E_MIN_EV,
                e_max_ev: E_MAX_EV,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model"),
    )
}

fn unfolded() -> Vec<(&'static str, Arc<IkedaCarpenter>)> {
    vec![
        (
            "constant",
            pulse(
                EnergyLaw::Const(0.565),
                EnergyLaw::Const(0.25),
                EnergyLaw::Const(0.15),
                None,
                None,
            ),
        ),
        (
            "steep alpha",
            pulse(
                EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                EnergyLaw::Const(0.25),
                EnergyLaw::Const(0.15),
                None,
                None,
            ),
        ),
        (
            "alpha, beta and R laws",
            pulse(
                EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                EnergyLaw::SqrtE { a0: 0.02, a1: 0.2 },
                EnergyLaw::ExpMilliEv { kappa: 5.0e4 },
                None,
                None,
            ),
        ),
    ]
}

fn folded() -> Vec<(&'static str, Arc<IkedaCarpenter>)> {
    vec![
        (
            "folded constant",
            pulse(
                EnergyLaw::Const(0.565),
                EnergyLaw::Const(0.25),
                EnergyLaw::Const(0.15),
                None,
                Some(2.0),
            ),
        ),
        (
            "folded with laws",
            pulse(
                EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                EnergyLaw::SqrtE { a0: 0.02, a1: 0.2 },
                EnergyLaw::Const(0.15),
                Some(0.5),
                Some(2.0),
            ),
        ),
    ]
}

fn probability_in(pulse: &IkedaCarpenter, flight_time_us: f64, edges: &[f64], t0_us: f64) -> f64 {
    pulse
        .detector_bin_probabilities((CLOCK / flight_time_us).powi(2), edges, t0_us)
        .expect("probabilities")
        .iter()
        .sum()
}

fn window_probability(pulse: &IkedaCarpenter, flight_time_us: f64) -> f64 {
    probability_in(pulse, flight_time_us, &edges(), T0_US)
}

fn largest_outside(pulse: &IkedaCarpenter, grid: &FlightTimeGrid) -> f64 {
    let (u_lo, u_hi) = grid.range_us();
    let (u_fast, u_slow) = (CLOCK / E_MAX_EV.sqrt(), CLOCK / E_MIN_EV.sqrt());
    let scan = (0..)
        .map(|i| u_fast + 0.1 * f64::from(i))
        .take_while(|&u| u <= u_slow);
    scan.chain([u_lo * (1.0 - 1e-9), u_hi * (1.0 + 1e-9)])
        .filter(|&u| u < u_lo || u > u_hi)
        .map(|u| window_probability(pulse, u))
        .fold(0.0, f64::max)
}

#[test]
fn no_flight_time_outside_the_range_reaches_a_bin() {
    for (name, pulse) in unfolded().into_iter().chain(folded()) {
        let grid = FlightTimeGrid::new(&edges(), T0_US, &pulse).expect(name);
        let largest = largest_outside(&pulse, &grid);
        assert!(
            largest < NEGLIGIBLE_ARRIVAL_PROBABILITY,
            "{name}: {largest}"
        );
    }
}

fn assert_tight(name: &str, pulse: &Arc<IkedaCarpenter>, edges: &[f64], t0_us: f64) {
    let (u_lo, u_hi) = FlightTimeGrid::new(edges, t0_us, pulse)
        .expect(name)
        .range_us();
    let (first, last) = (edges[0], edges[edges.len() - 1]);
    let after_first = |u: f64| probability_in(pulse, u, &[first, first + 1.0e7], t0_us);
    let before_last = |u: f64| probability_in(pulse, u, &[last - 1.0e7, last], t0_us);
    assert!(
        after_first(u_lo * (1.0 - 1e-6)) < NEGLIGIBLE_ARRIVAL_PROBABILITY,
        "{name}"
    );
    assert!(
        after_first(u_lo * (1.0 + 1e-6)) >= NEGLIGIBLE_ARRIVAL_PROBABILITY,
        "{name}"
    );
    assert_eq!(before_last(u_hi * (1.0 + 1e-6)), 0.0, "{name}");
    assert!(before_last(u_hi * (1.0 - 1e-6)) > 0.0, "{name}");
}

#[test]
fn an_unfolded_pulse_reaches_the_bins_just_inside_both_ends() {
    for (name, pulse) in unfolded() {
        assert_tight(name, &pulse, &edges(), T0_US);
    }
}

#[test]
fn windows_the_synthesis_grid_does_not_cover_are_refused() {
    let constant = &unfolded()[0].1;
    let slow_tail = pulse(
        EnergyLaw::Const(0.565),
        EnergyLaw::Const(0.02),
        EnergyLaw::Const(0.05),
        None,
        None,
    );
    let past_slowest: Vec<f64> = (350..=2000).map(f64::from).collect();
    for (edges, t0, pulse) in [
        (edges(), T0_US, &slow_tail),
        (past_slowest, T0_US, constant),
        (edges(), 350.0, constant),
    ] {
        assert!(matches!(
            FlightTimeGrid::new(&edges, t0, pulse),
            Err(FlightTimeGridError::OutsideCalibration { .. })
        ));
    }
}

#[test]
fn invalid_windows_and_unaffordable_grids_are_refused() {
    let constant = &unfolded()[0].1;
    assert!(matches!(
        FlightTimeGrid::new(&edges(), f64::NAN, constant),
        Err(FlightTimeGridError::InvalidTimingOffset(_))
    ));
    assert!(matches!(
        FlightTimeGrid::new(&[350.0, 349.0], T0_US, constant),
        Err(FlightTimeGridError::InvalidTimeEdges)
    ));
    let sharp = |alpha: f64| {
        pulse(
            EnergyLaw::Const(alpha),
            EnergyLaw::Const(0.25),
            EnergyLaw::Const(0.0),
            None,
            None,
        )
    };
    assert!(matches!(
        FlightTimeGrid::new(&edges(), T0_US, &sharp(2000.0)),
        Err(FlightTimeGridError::TooManyPoints { .. })
    ));
    let near_cap = FlightTimeGrid::new(&edges(), T0_US, &sharp(700.0)).expect("near the cap");
    assert!(matches!(
        near_cap.halved(),
        Err(FlightTimeGridError::TooManyPoints { .. })
    ));
}

#[test]
fn a_neutron_arrives_between_its_delays_but_for_a_negligible_chance() {
    let (c, sqrt_e) = (EnergyLaw::Const, |a0, a1| EnergyLaw::SqrtE { a0, a1 });
    let witnesses = [
        (
            pulse(sqrt_e(0.35, 0.05), c(0.25), c(0.15), None, None),
            false,
        ),
        (
            pulse(c(0.565), sqrt_e(0.02, 0.05), c(0.5), None, None),
            false,
        ),
        (
            pulse(
                c(0.565),
                c(0.1),
                EnergyLaw::ExpMilliEv { kappa: 5.0e4 },
                None,
                None,
            ),
            false,
        ),
        (pulse(c(0.565), c(0.25), c(0.15), Some(1.0), None), true),
        (pulse(c(0.565), c(0.25), c(0.0), None, Some(2.0)), true),
        (
            pulse(
                sqrt_e(0.35, 0.05),
                sqrt_e(0.02, 0.2),
                c(0.15),
                Some(0.5),
                Some(2.0),
            ),
            true,
        ),
    ];
    for (w, (pulse, folded)) in witnesses.iter().enumerate() {
        for k in 0..=60 {
            let e = E_MIN_EV * (E_MAX_EV / E_MIN_EV).powf(f64::from(k) / 60.0);
            let (first, last) = pulse.delays_us(e).expect("delays");
            let nominal = -TOF_FACTOR * FLIGHT_PATH_M / e.sqrt();
            let chance = |edges: [f64; 2]| -> f64 {
                pulse
                    .detector_bin_probabilities(e, &edges, nominal)
                    .expect("probabilities")[0]
            };
            let before = chance([first - 1.0e4, first]);
            let after = chance([last, last + 1.0e6]);
            assert!(
                before <= NEGLIGIBLE_ARRIVAL_PROBABILITY,
                "{w} {e}: {before}"
            );
            assert!(after <= NEGLIGIBLE_ARRIVAL_PROBABILITY, "{w} {e}: {after}");
            assert_eq!(chance([first - 1.0e4, 0.0]) > 0.0, *folded, "{w} {e}");
            assert_eq!(after == 0.0, *folded, "{w} {e}");
        }
    }
}

#[test]
fn pulses_whose_arrival_can_fall_with_flight_time_are_refused() {
    let (c, sqrt_e) = (EnergyLaw::Const, |a0, a1| EnergyLaw::SqrtE { a0, a1 });
    for (parameter, pulse) in [
        ("α", pulse(sqrt_e(-0.05, 1.0), c(0.25), c(0.0), None, None)),
        (
            "β",
            pulse(c(0.565), sqrt_e(-0.02, 0.3), c(0.15), None, None),
        ),
        (
            "R",
            pulse(c(0.565), c(0.002), sqrt_e(0.07, 0.0), None, None),
        ),
    ] {
        let latest = |u: f64| u + pulse.delays_us((CLOCK / u).powi(2)).expect("delays").1;
        let fastest = CLOCK / E_MAX_EV.sqrt();
        let falls = (0..1000)
            .map(|i| fastest + 0.5 * f64::from(i))
            .any(|u| latest(u + 0.5) < latest(u));
        assert!(falls, "{parameter}");
        match FlightTimeGrid::new(&edges(), T0_US, &pulse) {
            Err(FlightTimeGridError::LengthensWithEnergy { parameter: found }) => {
                assert_eq!(found, parameter);
            }
            other => panic!("{parameter}: {other:?}"),
        }
    }
}
