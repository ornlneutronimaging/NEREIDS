use std::iter::successors;
use std::sync::Arc;

use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, NEGLIGIBLE_ARRIVAL_PROBABILITY, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR, TabulatedResolution};
use nereids_pipeline::reference::Instrument;

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const CLOCK: f64 = TOF_FACTOR * FLIGHT_PATH_M;
const COUNTS_PER_US: f64 = 1.0e4;
const BOUND: f64 = 0.01;
const MAX_HALVINGS: usize = 6;

fn edges() -> Vec<f64> {
    (350..=470).map(f64::from).collect()
}

fn ikeda_carpenter(alpha: EnergyLaw, channel_fwhm_us: Option<f64>) -> ResolutionFunction {
    ResolutionFunction::IkedaCarpenter(Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha,
                beta: EnergyLaw::Const(0.25),
                r: EnergyLaw::Const(0.15),
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

fn table(kernel: &[(f64, f64)]) -> ResolutionFunction {
    let kernel: (Vec<f64>, Vec<f64>) = kernel.iter().copied().unzip();
    ResolutionFunction::Tabulated(Arc::new(
        TabulatedResolution::from_kernels(
            vec![1.0, 200.0],
            vec![kernel.clone(), kernel],
            FLIGHT_PATH_M,
        )
        .expect("valid table"),
    ))
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
    grid.predict(&beam)
}

fn spread(counts: &[f64], expected: &[f64]) -> f64 {
    counts
        .iter()
        .zip(expected)
        .map(|(c, e)| (c - e).powi(2) / e)
        .sum()
}

fn simulated(grid: &FlightTimeGrid, pulse: ResolutionFunction) -> Vec<f64> {
    let (u_lo, u_hi) = grid.range_us();
    let margin = 0.2 * (u_hi - u_lo);
    let energy = |u: f64| (CLOCK / u).powi(2);
    let expected = Instrument {
        time_edges_us: edges(),
        flight_path_m: FLIGHT_PATH_M,
        t0_us: T0_US,
        resolution: pulse,
    }
    .expected_counts(
        &|e| {
            let u = CLOCK / e.sqrt();
            beam_per_us(u) * u / (2.0 * e)
        },
        &|es| vec![1.0; es.len()],
        (energy(u_hi + margin), energy(u_lo - margin)),
        grid.step_us() / 16.0,
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
fn a_grid_that_halving_leaves_unchanged_matches_the_simulator() {
    let pulses = [
        (
            "constant IC",
            ikeda_carpenter(EnergyLaw::Const(0.565), None),
            0,
        ),
        (
            "IC with a steep energy law",
            ikeda_carpenter(EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 }, None),
            1,
        ),
        (
            "IC with a channel fold",
            ikeda_carpenter(EnergyLaw::Const(0.565), Some(2.0)),
            0,
        ),
        ("triangle", table(&[(-1.0, 0.0), (0.0, 1.0), (3.0, 0.0)]), 0),
        (
            "late table",
            table(&[(2.0, 0.0), (3.0, 1.0), (5.0, 0.0)]),
            0,
        ),
    ];
    for (name, pulse, halvings) in pulses {
        let grid = FlightTimeGrid::new(&edges(), T0_US, &pulse).expect(name);
        let expected = simulated(&grid, pulse);
        let predicted: Vec<Vec<f64>> =
            successors(Some(grid), |g| Some(g.halved().expect("halved grid")))
                .take(MAX_HALVINGS + 1)
                .map(|g| counts(&g))
                .collect();
        let accepted = predicted
            .windows(2)
            .position(|pair| spread(&pair[0], &pair[1]) <= BOUND)
            .expect(name);
        assert_eq!(accepted, halvings, "{name}");
        assert!(spread(&predicted[accepted], &expected) <= BOUND, "{name}");
        assert!(
            predicted[..accepted]
                .iter()
                .all(|rejected| spread(rejected, &expected) > BOUND),
            "{name}"
        );
    }
}
