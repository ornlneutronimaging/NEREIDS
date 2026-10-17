use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::ikeda_carpenter::{
    DETECTOR_TAIL_DECAY_US, DetectorPulse, DetectorTail, IkedaCarpenter, IkedaCarpenterParams,
    NEGLIGIBLE_ARRIVAL_PROBABILITY, SynthesisGrid,
};
use nereids_physics::resolution::TOF_FACTOR;

const ENERGY_EV: f64 = 20.0;
const ARRIVAL_US: f64 = 100.0;
const FRACTION: f64 = 0.3;

fn edges() -> Vec<f64> {
    (0..=60).map(|k| 99.0 + 0.5 * f64::from(k)).collect()
}

fn params(
    [alpha, beta, r]: [f64; 3],
    fwhm_us: f64,
    tail: Option<DetectorTail>,
) -> IkedaCarpenterParams {
    IkedaCarpenterParams {
        channel_fwhm_us: Some(fwhm_us),
        detector_tail: tail,
        ..IkedaCarpenterParams::constant(alpha, beta, r)
    }
}

fn tail(fraction: f64, decay_us: f64) -> Option<DetectorTail> {
    Some(DetectorTail { fraction, decay_us })
}

fn probabilities(params: &IkedaCarpenterParams, arrival_us: f64) -> Vec<f64> {
    params
        .bin_probabilities_at(ENERGY_EV, arrival_us, &edges())
        .expect("probabilities")
}

/// `[α, β, R]`, the triangle's FWHM and `τ_D`: `1/τ_D` apart from `β`,
/// equal to it, within a relative 1e-3 of it and just beyond that, with no
/// storage, above `α`, and equal to both.
const CASES: [([f64; 3], f64, f64); 9] = [
    ([2.0, 0.3, 0.2], 0.0, 1.5),
    ([2.0, 0.3, 0.2], 0.4, 1.5),
    ([2.0, 0.5, 0.3], 0.0, 2.0),
    ([2.0, 0.5, 0.3], 0.4, 1.0 / (0.5 * 1.0005)),
    ([2.0, 0.5, 0.3], 0.0, 1.0 / (0.5 * 0.9995)),
    ([2.0, 0.5, 0.3], 0.0, 1.0 / (0.5 * 1.003)),
    ([2.0, 0.3, 0.0], 0.0, 0.8),
    ([2.0, 3.0, 0.4], 0.4, 0.25),
    ([1.0, 1.0, 0.3], 0.0, 1.0),
];

#[test]
fn a_detector_tail_delays_the_fraction_f_of_the_pulse_by_an_exponential() {
    for (rates, fwhm_us, decay_us) in CASES {
        let tailed = probabilities(
            &params(rates, fwhm_us, tail(FRACTION, decay_us)),
            ARRIVAL_US,
        );
        let plain = params(rates, fwhm_us, None);
        let step = 0.0025;
        let steps = (30.0 * decay_us / step).ceil() as usize * 2;
        let mut delayed = vec![0.0; tailed.len()];
        for j in 0..=steps {
            let s = step * j as f64;
            let simpson = match j {
                0 => 1.0,
                j if j == steps => 1.0,
                j if j % 2 == 1 => 4.0,
                _ => 2.0,
            } * step
                / 3.0;
            let weight = simpson * (-s / decay_us).exp() / decay_us;
            for (sum, p) in delayed
                .iter_mut()
                .zip(probabilities(&plain, ARRIVAL_US + s))
            {
                *sum += weight * p;
            }
        }
        let on_time = probabilities(&plain, ARRIVAL_US);
        let largest = tailed
            .iter()
            .zip(on_time.iter().zip(&delayed))
            .map(|(t, (p, d))| (t - ((1.0 - FRACTION) * p + FRACTION * d)).abs())
            .fold(0.0, f64::max);
        assert!(largest < 1e-11, "{rates:?} {fwhm_us} {decay_us}: {largest}");
    }
}

#[test]
fn the_slopes_are_those_of_the_probabilities() {
    for (rates, fwhm_us, decay_us) in CASES {
        let at = |rates: [f64; 3], h2: f64, fraction: f64, decay: f64, arrival: f64| {
            probabilities(&params(rates, h2.sqrt(), tail(fraction, decay)), arrival)
        };
        let h2 = fwhm_us * fwhm_us;
        let centred = |up: Vec<f64>, down: Vec<f64>, step: f64| -> Vec<f64> {
            up.iter()
                .zip(&down)
                .map(|(u, d)| (u - d) / (2.0 * step))
                .collect()
        };
        let by_rate = |n: usize| {
            let step = 1e-5 * rates[n].max(0.1);
            let moved = |by: f64| {
                let mut moved = rates;
                moved[n] += by * step;
                at(moved, h2, FRACTION, decay_us, ARRIVAL_US)
            };
            if rates[n] > 0.0 {
                centred(moved(1.0), moved(-1.0), step)
            } else {
                let [here, one, two] = [0.0, 1.0, 2.0].map(moved);
                (0..here.len())
                    .map(|k| (4.0 * one[k] - 3.0 * here[k] - two[k]) / (2.0 * step))
                    .collect()
            }
        };
        let differences = [
            by_rate(0),
            by_rate(1),
            by_rate(2),
            centred(
                at(rates, h2 + 1e-4, FRACTION, decay_us, ARRIVAL_US),
                at(rates, (h2 - 1e-4).max(0.0), FRACTION, decay_us, ARRIVAL_US),
                if h2 > 0.0 { 1e-4 } else { 5e-5 },
            ),
            centred(
                at(rates, h2, FRACTION + 1e-5, decay_us, ARRIVAL_US),
                at(rates, h2, FRACTION - 1e-5, decay_us, ARRIVAL_US),
                1e-5,
            ),
            centred(
                at(rates, h2, FRACTION, decay_us * (1.0 + 1e-5), ARRIVAL_US),
                at(rates, h2, FRACTION, decay_us * (1.0 - 1e-5), ARRIVAL_US),
                decay_us * 1e-5,
            ),
            centred(
                at(rates, h2, FRACTION, decay_us, ARRIVAL_US + 1e-5),
                at(rates, h2, FRACTION, decay_us, ARRIVAL_US - 1e-5),
                1e-5,
            ),
        ];
        let tailed = params(rates, fwhm_us, tail(FRACTION, decay_us));
        let [a, b, r, h] = tailed
            .bin_pulse_slopes_at(ENERGY_EV, ARRIVAL_US, &edges())
            .expect("pulse slopes");
        let [f, decay] = tailed
            .bin_tail_slopes_at(ENERGY_EV, ARRIVAL_US, &edges())
            .expect("tail slopes");
        let arrival = tailed
            .bin_arrival_slopes_at(ENERGY_EV, ARRIVAL_US, &edges())
            .expect("arrival slopes");
        let names = ["α", "β", "R", "h²", "f", "τ_D", "arrival"];
        for ((name, slopes), difference) in names
            .iter()
            .zip([a, b, r, h, f, decay, arrival])
            .zip(&differences)
        {
            if *name == "h²" && h2 == 0.0 {
                continue;
            }
            let scale = difference.iter().fold(0.0_f64, |m, d| m.max(d.abs()));
            let largest = slopes
                .iter()
                .zip(difference)
                .map(|(s, d)| (s - d).abs())
                .fold(0.0, f64::max);
            assert!(
                largest <= 1e-5 * scale,
                "{rates:?} {fwhm_us} {decay_us} {name}: {largest} of {scale}"
            );
        }
    }
}

fn everything_at(laws: &IkedaCarpenterParams, edges: &[f64]) -> Vec<Vec<f64>> {
    let mut all = vec![
        laws.bin_probabilities_at(ENERGY_EV, 0.0, edges)
            .expect("probabilities"),
    ];
    all.extend(
        laws.bin_pulse_slopes_at(ENERGY_EV, 0.0, edges)
            .expect("pulse slopes"),
    );
    all.extend(
        laws.bin_tail_slopes_at(ENERGY_EV, 0.0, edges)
            .expect("tail slopes"),
    );
    all.push(arrival_slopes(laws, 0.0, edges));
    all
}

fn arrival_slopes(laws: &IkedaCarpenterParams, arrival_us: f64, edges: &[f64]) -> Vec<f64> {
    laws.bin_arrival_slopes_at(ENERGY_EV, arrival_us, edges)
        .expect("arrival slopes")
}

fn assert_close(got: &[f64], expected: &[f64], tolerance: f64, name: &str) {
    let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (got, expected) in got.iter().zip(expected) {
        assert!(
            (got - expected).abs() <= tolerance * scale,
            "{name}: {got} against {expected}"
        );
    }
}

#[test]
fn the_pulse_at_the_lowest_rates_is_the_same_pulse_in_slower_time() {
    let ([alpha, beta, r], _, decay_us) = CASES[8];
    let slower = *DETECTOR_TAIL_DECAY_US.end();
    let at = |by: f64| {
        let laws = params(
            [alpha / by, beta / by, r],
            0.0,
            tail(FRACTION, decay_us * by),
        );
        let edges: Vec<f64> = edges().iter().map(|e| (e - ARRIVAL_US) * by).collect();
        everything_at(&laws, &edges)
    };
    let names = ["probability", "α", "β", "R", "h²", "f", "τ_D", "arrival"];
    let powers = [0, 1, 1, 0, -2, 0, -1, -1];
    for (((fast, slow), power), name) in at(1.0).iter().zip(at(slower)).zip(powers).zip(names) {
        let slow: Vec<f64> = slow.iter().map(|v| v * slower.powi(-power)).collect();
        assert_close(&slow, fast, 1e-8, name);
    }
}

#[test]
fn the_shortest_decay_moves_the_counts_by_its_mean() {
    let (rates, fwhm_us, _) = CASES[1];
    let decay_us = *DETECTOR_TAIL_DECAY_US.start();
    let tailed = params(rates, fwhm_us, tail(FRACTION, decay_us));
    let plain = params(rates, fwhm_us, None);
    let slopes = arrival_slopes(&plain, ARRIVAL_US, &edges());
    let moved: Vec<f64> = probabilities(&tailed, ARRIVAL_US)
        .iter()
        .zip(probabilities(&plain, ARRIVAL_US))
        .map(|(t, p)| (t - p) / (FRACTION * decay_us))
        .collect();
    let [by_f, by_decay] = tailed
        .bin_tail_slopes_at(ENERGY_EV, ARRIVAL_US, &edges())
        .expect("tail slopes");
    let by_f: Vec<f64> = by_f.iter().map(|v| v / decay_us).collect();
    let by_decay: Vec<f64> = by_decay.iter().map(|v| v / FRACTION).collect();
    let recorded = arrival_slopes(&tailed, ARRIVAL_US, &edges());
    for (got, name) in [
        (moved, "moved"),
        (by_f, "f"),
        (by_decay, "τ_D"),
        (recorded, "arrival"),
    ] {
        assert_close(&got, &slopes, 1e-4, name);
    }
    let pulse = |laws: &IkedaCarpenterParams| {
        laws.bin_pulse_slopes_at(ENERGY_EV, ARRIVAL_US, &edges())
            .expect("pulse slopes")
    };
    for (got, expected) in pulse(&tailed).iter().zip(pulse(&plain)) {
        assert_close(got, &expected, 1e-4, "pulse");
    }
}

#[test]
fn no_fraction_leaves_the_pulse_as_it_is() {
    let (rates, fwhm_us, decay_us) = CASES[1];
    let [plain, none] = [None, tail(0.0, decay_us)].map(|tail| params(rates, fwhm_us, tail));
    assert_eq!(
        probabilities(&plain, ARRIVAL_US),
        probabilities(&none, ARRIVAL_US)
    );
    let slopes = |params: &IkedaCarpenterParams| {
        params
            .bin_pulse_slopes_at(ENERGY_EV, ARRIVAL_US, &edges())
            .expect("slopes")
    };
    assert_eq!(slopes(&plain), slopes(&none));
    assert_eq!(
        plain.delays_us(ENERGY_EV).expect("delays"),
        none.delays_us(ENERGY_EV).expect("delays")
    );
}

#[test]
fn a_tailed_pulse_leaves_no_neutron_outside_the_range() {
    let (span, flight_path_m, t0_us) = ((1.0, 200.0), 25.0, 3.0);
    let laws = params([2.0, 0.3, 0.2], 0.4, tail(FRACTION, 4.0));
    let pulse = DetectorPulse::new(laws.clone(), span, 256).expect("pulse");
    let edges: Vec<f64> = (350..=470).map(f64::from).collect();
    let grid = FlightTimeGrid::new(&edges, t0_us, flight_path_m, &pulse).expect("grid");
    let (u_lo, u_hi) = grid.range_us();
    let clock = TOF_FACTOR * flight_path_m;
    let largest = (0..)
        .map(|i| clock / span.1.sqrt() + 0.1 * f64::from(i))
        .take_while(|&u| u <= clock / span.0.sqrt())
        .filter(|&u| u < u_lo || u > u_hi)
        .map(|u| {
            let energy = (clock / u).powi(2);
            laws.bin_probabilities_at(energy, t0_us + u, &edges)
                .expect("probabilities")
                .iter()
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    assert!(largest < NEGLIGIBLE_ARRIVAL_PROBABILITY, "{largest}");
    let untailed =
        DetectorPulse::new(params([2.0, 0.3, 0.2], 0.4, None), span, 256).expect("pulse");
    let without = FlightTimeGrid::new(&edges, t0_us, flight_path_m, &untailed).expect("grid");
    assert!(u_lo < without.range_us().0 - 4.0, "{u_lo}");
}

#[test]
fn a_tail_out_of_range_is_refused() {
    for (fraction, decay_us) in [
        (1.5, 1.0),
        (-0.1, 1.0),
        (f64::NAN, 1.0),
        (0.2, 0.0),
        (0.2, 1e-10),
        (0.2, 2e9),
    ] {
        let laws = params([2.0, 0.3, 0.2], 0.0, tail(fraction, decay_us));
        let message = laws
            .bin_probabilities_at(ENERGY_EV, ARRIVAL_US, &edges())
            .expect_err("refused")
            .to_string();
        assert!(message.contains("detector tail"), "{message}");
        assert!(DetectorPulse::new(laws, (1.0, 200.0), 256).is_err());
    }
    let plain = params([2.0, 0.3, 0.2], 0.0, None);
    assert!(
        plain
            .bin_tail_slopes_at(ENERGY_EV, ARRIVAL_US, &edges())
            .is_err()
    );
    let grid = SynthesisGrid {
        e_min_ev: 1.0,
        e_max_ev: 200.0,
        n_energies: 8,
        n_tau: 64,
    };
    let message = IkedaCarpenter::new(params([2.0, 0.3, 0.2], 0.0, tail(0.1, 1.0)), 25.0, &grid)
        .expect_err("refused")
        .to_string();
    assert!(message.contains("detector tail"), "{message}");
}
