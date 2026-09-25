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

fn pulse_on(
    params: IkedaCarpenterParams,
    (e_min_ev, e_max_ev): (f64, f64),
    n_tau: usize,
) -> Arc<IkedaCarpenter> {
    Arc::new(
        IkedaCarpenter::new(
            params,
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev,
                e_max_ev,
                n_energies: 32,
                n_tau,
            },
        )
        .expect("valid IC model"),
    )
}

fn pulse(
    alpha: EnergyLaw,
    beta: EnergyLaw,
    r: EnergyLaw,
    burst_sigma_us: Option<f64>,
    channel_fwhm_us: Option<f64>,
) -> Arc<IkedaCarpenter> {
    pulse_on(
        IkedaCarpenterParams {
            alpha,
            beta,
            r,
            burst_sigma_us,
            channel_fwhm_us,
        },
        (E_MIN_EV, E_MAX_EV),
        256,
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
        (
            "alpha falling as energy rises",
            pulse(
                EnergyLaw::SqrtE { a0: -0.05, a1: 1.2 },
                EnergyLaw::Const(0.25),
                EnergyLaw::Const(0.15),
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
    for (name, pulse) in unfolded() {
        let grid = FlightTimeGrid::new(&edges(), T0_US, &pulse).expect(name);
        let largest = largest_outside(&pulse, &grid);
        assert!(
            largest < NEGLIGIBLE_ARRIVAL_PROBABILITY,
            "{name}: {largest}"
        );
    }
    for (name, pulse) in folded() {
        let grid = FlightTimeGrid::new(&edges(), T0_US, &pulse).expect(name);
        assert_eq!(largest_outside(&pulse, &grid), 0.0, "{name}");
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
fn the_delay_bounds_hold_every_arrival_of_every_energy_between() {
    let (c, sqrt_e) = (EnergyLaw::Const, |a0, a1| EnergyLaw::SqrtE { a0, a1 });
    let r_law = |kappa| EnergyLaw::ExpMilliEv { kappa };
    let witnesses = [
        (
            "alpha law",
            pulse(sqrt_e(0.35, 0.05), c(0.25), c(0.15), None, None),
            NEGLIGIBLE_ARRIVAL_PROBABILITY,
        ),
        (
            "beta law",
            pulse(c(0.565), sqrt_e(0.02, 0.05), c(0.5), None, None),
            NEGLIGIBLE_ARRIVAL_PROBABILITY,
        ),
        (
            "R law",
            pulse(c(0.565), c(0.1), r_law(5.0e4), None, None),
            NEGLIGIBLE_ARRIVAL_PROBABILITY,
        ),
        (
            "folded alpha and beta laws",
            pulse(
                sqrt_e(0.35, 0.05),
                sqrt_e(0.02, 0.2),
                c(0.15),
                Some(0.5),
                Some(2.0),
            ),
            0.0,
        ),
        (
            "folded, R crossing its storage cutoff",
            pulse_on(
                IkedaCarpenterParams {
                    alpha: c(1.0),
                    beta: sqrt_e(-0.1, 1.0001),
                    r: r_law(4535.964588767297),
                    burst_sigma_us: Some(1.0),
                    channel_fwhm_us: None,
                },
                (1.0, 100.0),
                600,
            ),
            0.0,
        ),
    ];
    for (name, pulse, tail) in witnesses {
        let references = pulse.ref_energies();
        let whole = (references[0], references[references.len() - 1]);
        for (e_low, e_high) in references.windows(2).map(|w| (w[0], w[1])).chain([whole]) {
            let (first, last) = pulse.delay_bounds(e_low, e_high).expect(name);
            for k in 0..=20 {
                let e = e_low * (e_high / e_low).powf(f64::from(k) / 20.0);
                let nominal = -TOF_FACTOR * FLIGHT_PATH_M / e.sqrt();
                let chance = |edges: [f64; 2]| -> f64 {
                    pulse
                        .detector_bin_probabilities(e, &edges, nominal)
                        .expect("probabilities")[0]
                };
                assert_eq!(chance([first - 1.0e4, first]), 0.0, "{name} {e}");
                assert!(chance([last, last + 1.0e6]) <= tail, "{name} {e}");
            }
        }
    }
}

#[test]
fn the_search_backs_out_of_intervals_whose_bounds_reach_but_whose_neutrons_do_not() {
    let (fastest_us, slowest_us) = (120.0, 400.0);
    let pulse = pulse_on(
        IkedaCarpenterParams {
            alpha: EnergyLaw::SqrtE {
                a0: -100.0 / CLOCK,
                a1: 1.0,
            },
            beta: EnergyLaw::Const(0.25),
            r: EnergyLaw::Const(0.0),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        },
        ((CLOCK / slowest_us).powi(2), (CLOCK / fastest_us).powi(2)),
        256,
    );
    let window: Vec<f64> = (260..=270).map(f64::from).collect();
    assert_tight("alpha falling steeply", &pulse, &window, 0.0);
}
