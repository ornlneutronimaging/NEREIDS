use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::ikeda_carpenter::{
    DetectorPulse, DetectorTail, IkedaCarpenter, IkedaCarpenterParams,
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
/// storage, and above `α`.
const CASES: [([f64; 3], f64, f64); 8] = [
    ([2.0, 0.3, 0.2], 0.0, 1.5),
    ([2.0, 0.3, 0.2], 0.4, 1.5),
    ([2.0, 0.5, 0.3], 0.0, 2.0),
    ([2.0, 0.5, 0.3], 0.4, 1.0 / (0.5 * 1.0005)),
    ([2.0, 0.5, 0.3], 0.0, 1.0 / (0.5 * 0.9995)),
    ([2.0, 0.5, 0.3], 0.0, 1.0 / (0.5 * 1.003)),
    ([2.0, 0.3, 0.0], 0.0, 0.8),
    ([2.0, 3.0, 0.4], 0.4, 0.25),
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
