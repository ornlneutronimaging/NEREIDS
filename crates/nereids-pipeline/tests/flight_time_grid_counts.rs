use std::iter::successors;
use std::sync::Arc;

use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, NEGLIGIBLE_ARRIVAL_PROBABILITY, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_pipeline::reference::Instrument;

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const CLOCK: f64 = TOF_FACTOR * FLIGHT_PATH_M;
const E_MIN_EV: f64 = 1.0;
const E_MAX_EV: f64 = 200.0;
const COUNTS_PER_US: f64 = 1.0e4;
const BOUND: f64 = 0.01;
const MAX_HALVINGS: usize = 6;
const SIMULATOR_STEP_US: f64 = 1.0 / 32.0;

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

fn pulses() -> Vec<(&'static str, Arc<IkedaCarpenter>)> {
    let (c, s) = (EnergyLaw::Const, |a0, a1| EnergyLaw::SqrtE { a0, a1 });
    vec![
        ("constant", pulse(c(0.565), c(0.25), c(0.15), None, None)),
        (
            "steep alpha",
            pulse(s(0.35, 0.05), c(0.25), c(0.15), None, None),
        ),
        (
            "alpha, beta and R laws",
            pulse(
                s(0.35, 0.05),
                s(0.02, 0.2),
                EnergyLaw::ExpMilliEv { kappa: 5.0e4 },
                None,
                None,
            ),
        ),
        (
            "folded constant",
            pulse(c(0.565), c(0.25), c(0.15), None, Some(2.0)),
        ),
        (
            "folded with laws",
            pulse(s(0.35, 0.05), s(0.02, 0.2), c(0.15), Some(0.5), Some(2.0)),
        ),
    ]
}

fn beam_per_us(u: f64) -> f64 {
    let x = (u / 400.0).ln();
    COUNTS_PER_US * (0.5 * x - 2.0 * x * x).exp()
}

fn counts(grid: &FlightTimeGrid) -> Vec<f64> {
    let beam: Vec<f64> = grid
        .flight_times_us()
        .iter()
        .map(|&u| beam_per_us(u))
        .collect();
    grid.predict(&beam).expect("one value per grid point")
}

fn spread(counts: &[f64], expected: &[f64]) -> f64 {
    counts
        .iter()
        .zip(expected)
        .map(|(c, e)| (c - e).powi(2) / e)
        .sum()
}

fn simulated(pulse: &Arc<IkedaCarpenter>, step_us: f64) -> Vec<f64> {
    let expected = Instrument {
        time_edges_us: edges(),
        flight_path_m: FLIGHT_PATH_M,
        t0_us: T0_US,
        resolution: ResolutionFunction::IkedaCarpenter(Arc::clone(pulse)),
    }
    .expected_counts(
        &|e| {
            let u = CLOCK / e.sqrt();
            beam_per_us(u) * u / (2.0 * e)
        },
        &|es| vec![1.0; es.len()],
        (E_MIN_EV, E_MAX_EV),
        step_us,
    );
    assert!(
        expected
            .edge_probability
            .iter()
            .all(|&p| p < NEGLIGIBLE_ARRIVAL_PROBABILITY / 100.0)
    );
    expected.counts
}

#[test]
fn the_grid_a_halving_accepts_matches_the_simulator() {
    for (name, pulse) in pulses() {
        let expected = simulated(&pulse, SIMULATOR_STEP_US);
        let simulator_spread = spread(&simulated(&pulse, 2.0 * SIMULATOR_STEP_US), &expected);
        assert!(simulator_spread <= BOUND / 100.0, "{name}");
        let grid = FlightTimeGrid::new(&edges(), T0_US, &pulse).expect(name);
        let predicted: Vec<Vec<f64>> =
            successors(Some(grid), |g| Some(g.halved().expect("halved grid")))
                .take(MAX_HALVINGS + 1)
                .map(|g| counts(&g))
                .collect();
        let accepted = 1 + predicted
            .windows(2)
            .position(|pair| spread(&pair[0], &pair[1]) <= BOUND)
            .expect(name);
        assert!(spread(&predicted[accepted], &expected) <= BOUND, "{name}");
    }
}
