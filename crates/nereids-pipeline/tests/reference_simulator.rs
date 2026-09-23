use std::sync::Arc;

use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR, TabulatedResolution};
use nereids_pipeline::reference::Instrument;

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const COUNTS_PER_US: f64 = 1000.0;
const STEP_US: f64 = 1.0e-3;
const IC_ALPHA: f64 = 0.565;
const IC_BETA: f64 = 0.25;
const IC_R: f64 = 0.15;

const TRIANGLE: [(f64, f64); 3] = [(-1.0, 0.0), (0.0, 1.0), (3.0, 0.0)];

fn triangle() -> ResolutionFunction {
    let kernel = (
        TRIANGLE.iter().map(|p| p.0).collect(),
        TRIANGLE.iter().map(|p| p.1).collect(),
    );
    ResolutionFunction::Tabulated(Arc::new(
        TabulatedResolution::from_kernels(
            vec![1.0, 100.0],
            vec![kernel.clone(), kernel],
            FLIGHT_PATH_M,
        )
        .expect("valid table"),
    ))
}

fn ikeda_carpenter() -> ResolutionFunction {
    ResolutionFunction::IkedaCarpenter(Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::Const(IC_ALPHA),
                beta: EnergyLaw::Const(IC_BETA),
                r: EnergyLaw::Const(IC_R),
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 100.0,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model"),
    ))
}

fn instrument(resolution: ResolutionFunction, flight_path_m: f64, t0_us: f64) -> Instrument {
    Instrument {
        time_edges_us: (400..=470).map(f64::from).collect(),
        flight_path_m,
        t0_us,
        resolution,
    }
}

fn energy(flight_time_us: f64) -> f64 {
    (TOF_FACTOR * FLIGHT_PATH_M / flight_time_us).powi(2)
}

fn beam(e: f64) -> f64 {
    COUNTS_PER_US * (TOF_FACTOR * FLIGHT_PATH_M / e.sqrt()) / (2.0 * e)
}

fn open(e: &[f64]) -> Vec<f64> {
    vec![1.0; e.len()]
}

fn band(e: &[f64]) -> Vec<f64> {
    e.iter()
        .map(|&e| {
            if e >= energy(450.0) && e <= energy(440.0) {
                0.0
            } else {
                1.0
            }
        })
        .collect()
}

fn triangle_mean_us() -> f64 {
    TRIANGLE.iter().map(|p| p.0).sum::<f64>() / 3.0
}

fn removed(instrument: &Instrument) -> (f64, f64) {
    let range = (energy(560.0), energy(300.0));
    let o = instrument.expected_counts(&beam, &open, range, STEP_US);
    let s = instrument.expected_counts(&beam, &band, range, STEP_US);
    let edges = &instrument.time_edges_us;
    let (mut total, mut moment) = (0.0, 0.0);
    for k in 0..edges.len() - 1 {
        let dn = o.counts[k] - s.counts[k];
        total += dn;
        moment += dn * 0.5 * (edges[k] + edges[k + 1]);
    }
    (total, moment / total)
}

#[test]
fn a_beam_uniform_in_flight_time_fills_every_bin_in_proportion_to_its_width() {
    for (label, resolution) in [
        ("triangle", triangle()),
        ("ikeda-carpenter", ikeda_carpenter()),
    ] {
        let mut instrument = instrument(resolution, FLIGHT_PATH_M, T0_US);
        instrument.time_edges_us = vec![400.0, 401.0, 403.0, 407.0, 415.0, 431.0, 470.0];
        let result =
            instrument.expected_counts(&beam, &open, (energy(560.0), energy(250.0)), STEP_US);
        assert!(
            result.edge_probability.iter().all(|&p| p < 1.0e-12),
            "{label}: neutrons beyond the integrated energies reach the bins: {:?}",
            result.edge_probability
        );
        for (k, &c) in result.counts.iter().enumerate() {
            let width = instrument.time_edges_us[k + 1] - instrument.time_edges_us[k];
            assert!(
                (c - COUNTS_PER_US * width).abs() <= 1.0e-6 * COUNTS_PER_US * width,
                "{label}: bin {k}, {width} µs wide, holds {c}"
            );
        }
    }
}

#[test]
fn a_black_band_removes_the_beam_it_covers_where_it_arrives() {
    let (total, centroid) = removed(&instrument(triangle(), FLIGHT_PATH_M, T0_US));
    assert!(
        (total - 10.0 * COUNTS_PER_US).abs() <= 2.0 * STEP_US * COUNTS_PER_US,
        "the band removed {total} counts, not {}",
        10.0 * COUNTS_PER_US
    );
    let expected = T0_US + 445.0 + triangle_mean_us();
    assert!(
        (centroid - expected).abs() <= 1.0e-3,
        "the removed counts centre at {centroid} µs, not {expected} µs"
    );
}

#[test]
fn t0_delays_every_count_by_the_same_time() {
    let range = (energy(560.0), energy(300.0));
    let early =
        instrument(triangle(), FLIGHT_PATH_M, T0_US).expected_counts(&beam, &band, range, STEP_US);
    let late = instrument(triangle(), FLIGHT_PATH_M, T0_US + 1.0)
        .expected_counts(&beam, &band, range, STEP_US);
    for k in 1..late.counts.len() {
        assert!(
            (late.counts[k] - early.counts[k - 1]).abs() <= 1.0e-9 * early.counts[k - 1],
            "one more µs of t0 put {} in bin {k}, not the {} of bin {}",
            late.counts[k],
            early.counts[k - 1],
            k - 1
        );
    }
}

#[test]
fn the_flight_path_sets_where_the_band_arrives() {
    let longer = 1.01 * FLIGHT_PATH_M;
    let (_, centroid) = removed(&instrument(triangle(), longer, T0_US));
    let expected = T0_US + 445.0 * longer / FLIGHT_PATH_M + triangle_mean_us();
    assert!(
        (centroid - expected).abs() <= 1.0e-3,
        "over {longer} m the removed counts centre at {centroid} µs, not {expected} µs"
    );
}

#[test]
fn transmission_sees_ascending_energies_and_the_low_edge_is_first() {
    let ascending = |e: &[f64]| {
        assert!(
            e.windows(2).all(|w| w[0] < w[1]),
            "transmission was handed energies out of order"
        );
        open(e)
    };
    let result = instrument(triangle(), FLIGHT_PATH_M, T0_US).expected_counts(
        &beam,
        &ascending,
        (energy(470.5 - T0_US), energy(300.0)),
        STEP_US,
    );
    let early_side = (0.5 * 0.5 * 0.5) / (0.5 * 4.0 * 1.0);
    assert!(
        (result.edge_probability[0] - early_side).abs() <= 1.0e-12
            && result.edge_probability[1] == 0.0,
        "a neutron nominally 0.5 µs after the last edge lands in it with chance \
         {early_side}, one at 300 µs never does; got {:?}",
        result.edge_probability
    );
}

#[test]
#[should_panic(expected = "one value per energy")]
fn a_transmission_of_the_wrong_length_is_refused() {
    let short = |e: &[f64]| vec![1.0; e.len() / 2];
    instrument(triangle(), FLIGHT_PATH_M, T0_US).expected_counts(
        &beam,
        &short,
        (energy(560.0), energy(300.0)),
        STEP_US,
    );
}

#[test]
fn repeated_calls_give_identical_counts() {
    let instrument = instrument(ikeda_carpenter(), FLIGHT_PATH_M, T0_US);
    let range = (energy(560.0), energy(250.0));
    let first = instrument.expected_counts(&beam, &band, range, STEP_US);
    for _ in 0..5 {
        let again = instrument.expected_counts(&beam, &band, range, STEP_US);
        assert!(
            first
                .counts
                .iter()
                .zip(&again.counts)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the same call returned different counts"
        );
    }
}

#[test]
fn an_ikeda_carpenter_band_centres_at_its_mean_delay() {
    let mut instrument = instrument(ikeda_carpenter(), FLIGHT_PATH_M, T0_US);
    instrument.time_edges_us = (400..=540).map(f64::from).collect();
    let (_, centroid) = removed(&instrument);
    let expected = T0_US + 445.0 + 3.0 / IC_ALPHA + IC_R / IC_BETA;
    assert!(
        (centroid - expected).abs() <= 1.0e-3,
        "the removed counts centre at {centroid} µs, not {expected} µs"
    );
}

#[test]
#[should_panic(expected = "must bracket the time window")]
fn a_range_that_misses_the_window_is_refused() {
    instrument(triangle(), FLIGHT_PATH_M, T0_US).expected_counts(
        &beam,
        &open,
        (energy(300.0), energy(250.0)),
        STEP_US,
    );
}
