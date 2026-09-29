use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_endf::resonance::test_support::{synthetic_isotope, synthetic_isotope_multi};
use nereids_physics::continuous_doppler::SUPPORT_X;
use nereids_physics::doppler::DopplerParams;
use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_physics::transmission::broadened_cross_sections;
use nereids_pipeline::counts_fit::{
    CountsFit, Measurement, NEGLIGIBLE_PREDICTION, Value, fit_counts,
};
use nereids_pipeline::error::PipelineError;
use nereids_pipeline::open_beam::{BOUND, Calibration};
use nereids_pipeline::reference::Instrument;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Poisson};

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const CLOCK: f64 = TOF_FACTOR * FLIGHT_PATH_M;
const TEMPERATURE_K: f64 = 300.0;
const CHARGE_RATIO: f64 = 1.2;
const THIN: f64 = 7.687e-4;
const SATURATED: f64 = 3.844e-2;
const TERMS: [f64; 4] = [0.93, 0.05, 0.5, 0.0];

struct Setup {
    edges: Vec<f64>,
    pulse: Arc<IkedaCarpenter>,
    energy_range_ev: (f64, f64),
    simulator_step_us: f64,
}

fn pulse(r: f64, e_max_ev: f64) -> Arc<IkedaCarpenter> {
    Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                beta: EnergyLaw::Const(0.25),
                r: EnergyLaw::Const(r),
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model"),
    )
}

fn standard() -> Setup {
    Setup {
        edges: (350..=470).map(f64::from).collect(),
        pulse: pulse(0.15, 200.0),
        energy_range_ev: (10.0, 50.0),
        simulator_step_us: 1.0 / 32.0,
    }
}

fn hafnium_like(energy_ev: f64) -> ResonanceData {
    synthetic_isotope(72, 180, energy_ev, 0.01, 0.06)
}

fn beam(level: f64) -> impl Fn(f64) -> f64 {
    move |u: f64| {
        let x = (u / 400.0).ln();
        level * (0.5 * x - 2.0 * x * x).exp()
    }
}

fn expected(
    setup: &Setup,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
) -> (Vec<f64>, Vec<f64>) {
    expected_at(setup, beam, sample, TEMPERATURE_K)
}

fn expected_at(
    setup: &Setup,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
    temperature_k: f64,
) -> (Vec<f64>, Vec<f64>) {
    (
        run(setup, 1.0, beam, &[], temperature_k),
        run(setup, CHARGE_RATIO, beam, sample, temperature_k),
    )
}

fn run(
    setup: &Setup,
    scale: f64,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
    temperature_k: f64,
) -> Vec<f64> {
    let instrument = Instrument {
        time_edges_us: setup.edges.clone(),
        flight_path_m: FLIGHT_PATH_M,
        t0_us: T0_US,
        resolution: ResolutionFunction::IkedaCarpenter(Arc::clone(&setup.pulse)),
    };
    let per_ev = |e: f64| {
        let u = CLOCK / e.sqrt();
        beam(u) * u / (2.0 * e)
    };
    let isotopes: Vec<ResonanceData> = sample.iter().map(|(data, _)| data.clone()).collect();
    let transmission = |energies: &[f64]| -> Vec<f64> {
        let sigma = cross_sections(energies, &isotopes, temperature_k);
        (0..energies.len())
            .map(|j| {
                let depth: f64 = sample.iter().zip(&sigma).map(|((_, n), s)| n * s[j]).sum();
                (-depth).exp()
            })
            .collect()
    };
    instrument
        .expected_counts(
            &|e| scale * per_ev(e),
            &transmission,
            setup.energy_range_ev,
            setup.simulator_step_us,
        )
        .counts
}

fn cross_sections(energies: &[f64], isotopes: &[ResonanceData], kelvin: f64) -> Vec<Vec<f64>> {
    thread_local! {
        static COMPUTED: RefCell<HashMap<String, Vec<Vec<f64>>>> = RefCell::default();
    }
    let grid = (energies.len(), energies.first(), energies.last());
    let key = format!("{kelvin:?} {grid:?} {isotopes:?}");
    COMPUTED.with_borrow_mut(|computed| {
        let sigma = computed.entry(key).or_insert_with(|| {
            broadened_cross_sections(energies, isotopes, kelvin, None, None)
                .expect("cross sections")
        });
        sigma.clone()
    })
}

fn with_background(
    setup: &Setup,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
    temperature_k: f64,
    [a, back_a, back_b, back_c]: [f64; 4],
) -> (Vec<f64>, Vec<f64>) {
    let (open, transmitted) = expected_at(setup, beam, sample, temperature_k);
    let scattered = |u: f64| a * beam(u) * (back_a + back_b * u / CLOCK + back_c * CLOCK / u);
    let scattered = run(setup, CHARGE_RATIO, &scattered, &[], temperature_k);
    let sample = transmitted.iter().zip(&scattered).map(|(t, b)| a * t + b);
    (open, sample.collect())
}

fn rounded(counts: &[f64]) -> Vec<f64> {
    counts.iter().map(|c| c.round()).collect()
}

fn recorded(
    setup: &Setup,
    (open, sample): (Vec<f64>, Vec<f64>),
    isotopes: &[(ResonanceData, f64)],
) -> Measurement {
    let live = |mu: &[f64]| -> Vec<f64> {
        let most = mu.iter().fold(0.0_f64, |m, &c| m.max(c));
        mu.iter().map(|c| 1.0 / (1.0 + 0.25 * c / most)).collect()
    };
    let (open_live, sample_live) = (live(&open), live(&sample));
    let times =
        |mu: &[f64], live: &[f64]| mu.iter().zip(live).map(|(c, l)| (l * c).round()).collect();
    let counts = (times(&open, &open_live), times(&sample, &sample_live));
    let mut m = measurement(setup, counts, isotopes);
    m.open_live = Some(open_live);
    m.sample_live = Some(sample_live);
    m
}

fn calibration(setup: &Setup) -> Calibration {
    Calibration {
        t0_us: T0_US,
        pulse: Arc::clone(&setup.pulse),
    }
}

fn measurement(
    setup: &Setup,
    (open, sample): (Vec<f64>, Vec<f64>),
    isotopes: &[(ResonanceData, f64)],
) -> Measurement {
    Measurement {
        time_edges_us: setup.edges.clone(),
        open_counts: open,
        sample_counts: sample,
        open_live: None,
        sample_live: None,
        charge_ratio: CHARGE_RATIO,
        normalization: Value::Known(1.0),
        background: [Value::Known(0.0); 3],
        isotopes: isotopes.to_vec(),
        temperature_k: Value::Known(TEMPERATURE_K),
    }
}

fn fitted_from(mut measurement: Measurement, start_k: f64) -> Measurement {
    measurement.temperature_k = Value::Fitted(start_k);
    measurement
}

fn error_bar(fit: &CountsFit, i: usize) -> f64 {
    fit.covariance
        .as_ref()
        .expect("covariance")
        .get(i, i)
        .sqrt()
}

#[test]
fn densities_normalization_and_background_are_recovered_from_starts_on_either_side() {
    let setup = standard();
    for truth in [THIN, SATURATED] {
        let isotope = hafnium_like(20.0);
        let sample = [(isotope.clone(), truth)];
        let counts = with_background(&setup, &beam(1.0e6), &sample, TEMPERATURE_K, TERMS);
        for (start, a, back_a, back_b) in [
            (0.5 * truth, 1.0, 0.02, 0.2),
            (2.0 * truth, 0.85, 0.12, 0.8),
        ] {
            let mut m = recorded(&setup, counts.clone(), &[(isotope.clone(), start)]);
            m.normalization = Value::Fitted(a);
            m.background[0] = Value::Fitted(back_a);
            m.background[1] = Value::Fitted(back_b);
            let fit = fit_counts(&m, &calibration(&setup)).expect("fit");
            assert!(fit.converged, "{truth} from {start}");
            assert_eq!(fit.overdispersion, Some(1.0), "{truth} from {start}");
            let pull = |i: usize, x: f64, t: f64| (x - t) / error_bar(&fit, i);
            let pulls = [
                pull(0, fit.densities[0], truth),
                pull(1, fit.normalization, TERMS[0]),
                pull(2, fit.background[0], TERMS[1]),
                pull(3, fit.background[1], TERMS[2]),
            ];
            assert!(
                pulls.iter().all(|p| p.abs() <= BOUND.sqrt()),
                "{truth} from {start}: {pulls:?}"
            );
        }
    }
}

fn chain(setup: &Setup) -> Vec<FlightTimeGrid> {
    std::iter::successors(
        Some(FlightTimeGrid::new(&setup.edges, T0_US, &setup.pulse).expect("grid")),
        |g| g.halved().ok(),
    )
    .collect()
}

fn predicted(
    grid: &FlightTimeGrid,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
) -> Vec<f64> {
    predicted_at(grid, beam, sample, TEMPERATURE_K)
}

fn predicted_at(
    grid: &FlightTimeGrid,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
    temperature_k: f64,
) -> Vec<f64> {
    let mut energies = grid.energies_ev();
    energies.reverse();
    let isotopes: Vec<ResonanceData> = sample.iter().map(|(data, _)| data.clone()).collect();
    let sigma = broadened_cross_sections(&energies, &isotopes, temperature_k, None, None)
        .expect("cross sections");
    let points = energies.len();
    let open: Vec<f64> = grid.flight_times_us().iter().map(|&u| beam(u)).collect();
    let transmitted: Vec<f64> = open
        .iter()
        .enumerate()
        .map(|(j, phi)| {
            let depth: f64 = sample
                .iter()
                .zip(&sigma)
                .map(|((_, n), s)| n * s[points - 1 - j])
                .sum();
            CHARGE_RATIO * phi * (-depth).exp()
        })
        .collect();
    let mut counts = grid.predict(&open).expect("counts");
    counts.extend(grid.predict(&transmitted).expect("counts"));
    counts
}

fn spread(fine: &[f64], coarse: &[f64]) -> f64 {
    fine.iter()
        .zip(coarse)
        .filter(|(f, _)| **f > 0.0)
        .map(|(f, c)| (f - c).powi(2) / f)
        .sum()
}

fn first_accepted(grids: &[FlightTimeGrid], counts: impl Fn(&FlightTimeGrid) -> Vec<f64>) -> usize {
    1 + grids
        .windows(2)
        .position(|pair| spread(&counts(&pair[1]), &counts(&pair[0])) <= BOUND)
        .expect("a pair within the bound")
}

#[test]
fn the_fit_is_on_the_finer_grid_of_the_first_pair_both_runs_leave_unchanged() {
    let setup = standard();
    let isotope = hafnium_like(20.0);
    let counts = expected(&setup, &beam(1.0e8), &[(isotope.clone(), THIN)]);
    let observed: Vec<f64> = rounded(&counts.0)
        .into_iter()
        .chain(rounded(&counts.1))
        .collect();
    let (open, sample) = observed.split_at(counts.0.len());
    let fit = fit_counts(
        &measurement(
            &setup,
            (open.to_vec(), sample.to_vec()),
            &[(isotope.clone(), THIN)],
        ),
        &calibration(&setup),
    )
    .expect("fit");
    let fitted = [(isotope, fit.densities[0])];
    let beam = |u: f64| fit.beam.per_us(u);
    let grids = chain(&setup);
    let accepted = first_accepted(&grids, |grid| predicted(grid, &beam, &fitted));
    assert_eq!(fit.halvings, accepted);
    assert_eq!(fit.points, grids[accepted].flight_times_us().len());
    assert_eq!(fit.step_us, grids[accepted].step_us());
    let deviance: f64 = predicted(&grids[accepted], &beam, &fitted)
        .iter()
        .zip(&observed)
        .map(|(&mu, &y)| {
            let d = (y - mu) / mu;
            mu * d * d * (0.5 - d / 6.0 + d * d / 12.0)
        })
        .sum();
    assert!(
        (fit.deviance - deviance).abs() <= 1e-2 * deviance,
        "{} vs {deviance}",
        fit.deviance
    );
}

fn distance(fitted: &[f64], simulated: &[f64]) -> f64 {
    fitted
        .iter()
        .zip(simulated)
        .map(|(f, s)| (f - s).powi(2) / s)
        .sum()
}

#[test]
fn a_resonance_between_the_first_grid_s_points_is_resolved() {
    let setup = kev_window();
    let first = FlightTimeGrid::new(&setup.edges, T0_US, &setup.pulse).expect("grid");
    let j = first
        .flight_times_us()
        .iter()
        .position(|&u| u > 57.2)
        .expect("a point");
    let level = 1.0e4;
    let beam = move |u: f64| {
        let x = (u / 57.2).ln();
        level * (0.5 * x - 2.0 * x * x).exp()
    };
    let u_r = first.flight_times_us()[j] + 0.198 * first.step_us();
    let isotope = hafnium_like((CLOCK / u_r).powi(2));
    let sample = [(isotope.clone(), 0.17767)];
    let simulated = expected(&setup, &beam, &sample);
    let fit = fit_counts(
        &measurement(
            &setup,
            (rounded(&simulated.0), rounded(&simulated.1)),
            &sample,
        ),
        &calibration(&setup),
    )
    .expect("fit");
    let simulated: Vec<f64> = simulated.0.into_iter().chain(simulated.1).collect();
    let grids = chain(&setup);
    let halving_alone = first_accepted(&grids, |grid| predicted(grid, &beam, &sample));
    let missed = distance(
        &predicted(&grids[halving_alone], &beam, &sample),
        &simulated,
    );
    assert!(missed > BOUND, "the halving check alone misses {missed}");
    let resolved = distance(&predicted(&grids[fit.halvings], &beam, &sample), &simulated);
    assert!(resolved <= BOUND, "{resolved}");
}

fn draw(expected: &[f64], seed: u64, counts_per_neutron: f64) -> Vec<f64> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    expected
        .iter()
        .map(|&mu| {
            counts_per_neutron
                * Poisson::new(mu / counts_per_neutron)
                    .expect("positive rate")
                    .sample(&mut rng)
        })
        .collect()
}

fn draws(
    expected: &(Vec<f64>, Vec<f64>),
    seed: u64,
    counts_per_neutron: f64,
) -> (Vec<f64>, Vec<f64>) {
    (
        draw(&expected.0, seed, counts_per_neutron),
        draw(&expected.1, seed + 1_000_000, counts_per_neutron),
    )
}

#[test]
fn the_overdispersion_is_the_variance_of_both_runs_over_the_poisson_variance() {
    let setup = standard();
    let isotope = hafnium_like(20.0);
    let sample = [(isotope, THIN)];
    let expected = expected(&setup, &beam(1.0e6), &sample);
    let counts_per_neutron = 7.0;
    let fits: Vec<CountsFit> = (100..110)
        .map(|seed| {
            fit_counts(
                &measurement(&setup, draws(&expected, seed, counts_per_neutron), &sample),
                &calibration(&setup),
            )
            .expect("fit")
        })
        .collect();
    let mut ratios = Vec::new();
    for fit in &fits {
        assert!(fit.converged);
        ratios.push(fit.overdispersion.expect("measured") / counts_per_neutron);
        let pull = (fit.densities[0] - THIN) / error_bar(fit, 0);
        assert!(pull.abs() <= 4.0, "{pull}");
    }
    let freedom = (2 * expected.0.len() - fits[0].beam.coefficients().len() - 1) as f64;
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let bound = 3.0 * (2.0 / freedom).sqrt() / (ratios.len() as f64).sqrt();
    assert!((mean - 1.0).abs() <= bound, "{mean} vs 1 ± {bound}");
}

#[test]
fn counting_every_neutron_seven_times_scales_the_overdispersion_not_the_error_bars() {
    let setup = standard();
    let sample = [(hafnium_like(20.0), SATURATED)];
    let once = draws(&expected(&setup, &beam(1.0e6), &sample), 300, 3.0);
    let seven = (
        once.0.iter().map(|c| 7.0 * c).collect(),
        once.1.iter().map(|c| 7.0 * c).collect(),
    );
    let fit = |counts| {
        fit_counts(
            &fitted_from(measurement(&setup, counts, &sample), TEMPERATURE_K),
            &calibration(&setup),
        )
        .expect("fit")
    };
    let (a, b) = (fit(once), fit(seven));
    let ratio = b.overdispersion.expect("measured") / a.overdispersion.expect("measured");
    assert!((ratio / 7.0 - 1.0).abs() <= 1e-3, "{ratio}");
    assert!((b.densities[0] / a.densities[0] - 1.0).abs() <= 1e-6);
    assert!((b.temperature_k / a.temperature_k - 1.0).abs() <= 1e-6);
    let (a, b) = (
        a.covariance.expect("covariance"),
        b.covariance.expect("covariance"),
    );
    for (x, y) in a.data.iter().zip(&b.data) {
        assert!((y - x).abs() <= 1e-3 * x.abs(), "{x} vs {y}");
    }
}

#[test]
fn counts_the_model_cannot_make_are_refused_and_rare_ones_leave_the_noise_alone() {
    let setup = Setup {
        pulse: pulse(0.0, 200.0),
        ..standard()
    };
    let sample = [(hafnium_like(20.0), 4.0e3 / 7805.1)];
    let expected = expected(&setup, &beam(1.0e4), &sample);
    let (dark, rare) = (62, 57);
    assert!(expected.1[dark] < NEGLIGIBLE_PREDICTION);
    assert!((1e-8..1e-4).contains(&expected.1[rare]));

    let (open, mut counts) = (rounded(&expected.0), rounded(&expected.1));
    counts[dark] = 1.0;
    match fit_counts(
        &measurement(&setup, (open, counts), &sample),
        &calibration(&setup),
    ) {
        Err(PipelineError::UnmodelledCounts {
            run,
            bin,
            counts,
            predicted,
        }) => {
            assert_eq!((run, bin, counts), ("sample", dark, 1.0));
            assert!(predicted < NEGLIGIBLE_PREDICTION, "{predicted}");
        }
        other => panic!("{other:?}"),
    }

    let counts_per_neutron = 7.0;
    let (open, mut counts) = draws(&expected, 400, counts_per_neutron);
    counts[rare] = 1.0;
    let fit = fit_counts(
        &measurement(&setup, (open, counts), &sample),
        &calibration(&setup),
    )
    .expect("fit");
    let ratio = fit.overdispersion.expect("measured") / counts_per_neutron;
    let measured = expected.0.len() + expected.1.iter().filter(|&&mu| mu >= 1.0).count();
    let freedom = (measured - fit.beam.coefficients().len() - 1) as f64;
    assert!(
        (ratio - 1.0).abs() <= 3.0 * (2.0 / freedom).sqrt(),
        "{ratio}"
    );
}

#[test]
fn an_absent_isotope_is_fitted_on_its_bound() {
    let setup = standard();
    let isotope = hafnium_like(20.0);
    let expected = expected(&setup, &beam(1.0e4), &[(isotope.clone(), 0.0)]);
    for temperature_k in [Value::Known(TEMPERATURE_K), Value::Fitted(TEMPERATURE_K)] {
        let fits: Vec<CountsFit> = (500..506)
            .map(|seed| {
                let mut m = measurement(
                    &setup,
                    draws(&expected, seed, 1.0),
                    &[(isotope.clone(), THIN)],
                );
                m.temperature_k = temperature_k;
                fit_counts(&m, &calibration(&setup)).expect("fit")
            })
            .collect();
        assert!(fits.iter().all(|fit| fit.densities[0] >= 0.0));
        let absent: Vec<&CountsFit> = fits.iter().filter(|fit| fit.densities[0] == 0.0).collect();
        assert!(!absent.is_empty(), "{temperature_k:?}");
        for fit in absent {
            let covariance = fit.covariance.as_ref().expect("covariance");
            assert!(
                covariance.data.iter().all(|v| v.is_nan()),
                "{temperature_k:?}"
            );
        }
    }
}

fn inverse(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .expect("a row");
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let p = a[col][col];
        for j in 0..n {
            a[col][j] /= p;
            inv[col][j] /= p;
        }
        for row in (0..n).filter(|&r| r != col) {
            let f = a[row][col];
            for j in 0..n {
                a[row][j] -= f * a[col][j];
                inv[row][j] -= f * inv[col][j];
            }
        }
    }
    inv
}

#[test]
fn the_covariance_is_the_inverse_of_the_information_in_the_counts() {
    covariance_against_information(Value::Known(TEMPERATURE_K), Value::Fitted);
    covariance_against_information(Value::Fitted(1000.0), Value::Known);
}

fn covariance_against_information(temperature: Value, scale: fn(f64) -> Value) {
    let setup = standard();
    let truth = [
        (hafnium_like(20.0), 3.0 / 7805.1),
        (synthetic_isotope(74, 182, 20.3, 0.01, 0.06), 1.5 / 7805.1),
    ];
    let counts = with_background(&setup, &beam(1.0e6), &truth, TEMPERATURE_K, TERMS);
    let mut m = recorded(&setup, counts, &truth);
    m.temperature_k = temperature;
    m.normalization = scale(TERMS[0]);
    m.background[0] = scale(TERMS[1]);
    m.background[1] = scale(TERMS[2]);
    let fit = fit_counts(&m, &calibration(&setup)).expect("fit");
    let fitted_temperature = matches!(temperature, Value::Fitted(_));
    let scales = matches!(m.normalization, Value::Fitted(_));
    let quantities = truth.len() + usize::from(fitted_temperature) + 3 * usize::from(scales);
    let (low, high) = FlightTimeGrid::new(&setup.edges, T0_US, &setup.pulse)
        .expect("grid")
        .range_us();
    let beam_times = |index: Option<usize>, step: f64| {
        let beam = fit.beam.clone();
        move |u: f64| {
            if !(low..=high).contains(&u) {
                return 0.0;
            }
            let slope: f64 = index.map_or(0.0, |i| {
                beam.basis(u).iter().filter(|p| p.0 == i).map(|p| p.1).sum()
            });
            beam.per_us(u) * (step * slope).exp()
        }
    };
    for (i, (_, n)) in truth.iter().enumerate() {
        let pull = (fit.densities[i] - n) / error_bar(&fit, i);
        assert!(pull.abs() <= BOUND.sqrt(), "{i}: {pull}");
    }
    if fitted_temperature {
        let pull = (fit.temperature_k - TEMPERATURE_K) / error_bar(&fit, truth.len());
        assert!(pull.abs() <= BOUND.sqrt(), "temperature: {pull}");
    }
    let mut terms = [fit.normalization; 4];
    terms[1..].copy_from_slice(&fit.background);
    let first_term = truth.len() + usize::from(fitted_temperature);
    for i in (0..3).filter(|_| scales) {
        let pull = (terms[i] - TERMS[i]) / error_bar(&fit, first_term + i);
        assert!(pull.abs() <= BOUND.sqrt(), "term {i}: {pull}");
    }
    let fitted: Vec<(ResonanceData, f64)> = truth
        .iter()
        .zip(&fit.densities)
        .map(|((data, _), &n)| (data.clone(), n))
        .collect();
    let live: Vec<f64> = m
        .open_live
        .iter()
        .chain(&m.sample_live)
        .flatten()
        .copied()
        .collect();
    let joined = |(open, sample): (Vec<f64>, Vec<f64>)| -> Vec<f64> {
        let counts = open.into_iter().chain(sample);
        counts.zip(&live).map(|(c, l)| l * c).collect()
    };
    let expected_at = |setup: &Setup, beam: &dyn Fn(f64) -> f64, sample: &[_], kelvin: f64| {
        with_background(setup, beam, sample, kelvin, terms)
    };
    let temperature_k = fit.temperature_k;
    let mu = joined(expected_at(
        &setup,
        &beam_times(None, 0.0),
        &fitted,
        temperature_k,
    ));
    let coefficients = fit.beam.coefficients().len();
    let columns: Vec<Vec<f64>> = (0..coefficients + quantities)
        .map(|p| {
            let shifted = |sign: f64| {
                if p < coefficients {
                    let h = 1e-4;
                    let beam = beam_times(Some(p), sign * h);
                    (
                        joined(expected_at(&setup, &beam, &fitted, temperature_k)),
                        h,
                    )
                } else if p < coefficients + fitted.len() {
                    let mut sample = fitted.clone();
                    let h = 1e-4 * sample[p - coefficients].1;
                    sample[p - coefficients].1 += sign * h;
                    let beam = beam_times(None, 0.0);
                    (
                        joined(expected_at(&setup, &beam, &sample, temperature_k)),
                        h,
                    )
                } else if p < first_term + coefficients {
                    let h = 1e-4 * temperature_k;
                    let beam = beam_times(None, 0.0);
                    let shifted_k = temperature_k + sign * h;
                    (joined(expected_at(&setup, &beam, &fitted, shifted_k)), h)
                } else {
                    let mut shifted = terms;
                    let h = 1e-4 * terms[p - first_term - coefficients];
                    shifted[p - first_term - coefficients] += sign * h;
                    let beam = beam_times(None, 0.0);
                    let counts = with_background(&setup, &beam, &fitted, temperature_k, shifted);
                    (joined(counts), h)
                }
            };
            let ((up, h), (down, _)) = (shifted(1.0), shifted(-1.0));
            up.iter()
                .zip(&down)
                .map(|(a, b)| (a - b) / (2.0 * h))
                .collect()
        })
        .collect();
    let information: Vec<Vec<f64>> = columns
        .iter()
        .map(|a| {
            columns
                .iter()
                .map(|b| {
                    a.iter()
                        .zip(b)
                        .zip(&mu)
                        .filter(|(_, m)| **m > 0.0)
                        .map(|((x, y), m)| x * y / m)
                        .sum()
                })
                .collect()
        })
        .collect();
    let oracle = inverse(information);
    let covariance = fit.covariance.expect("covariance");
    for i in 0..quantities {
        for j in 0..quantities {
            let expected = oracle[coefficients + i][coefficients + j];
            let scale = (oracle[coefficients + i][coefficients + i]
                * oracle[coefficients + j][coefficients + j])
                .sqrt();
            assert!(
                (covariance.get(i, j) - expected).abs() <= 1e-2 * scale,
                "{temperature:?} {i}, {j}: {} vs {expected}",
                covariance.get(i, j)
            );
        }
    }
}

#[test]
fn measurements_the_fit_does_not_describe_are_refused() {
    let setup = standard();
    let isotope = hafnium_like(20.0);
    let counts = expected(&setup, &beam(1.0e4), &[(isotope.clone(), THIN)]);
    let good = measurement(
        &setup,
        (rounded(&counts.0), rounded(&counts.1)),
        &[(isotope.clone(), THIN)],
    );
    let refused = |change: &dyn Fn(&mut Measurement)| {
        let mut m = good.clone();
        change(&mut m);
        fit_counts(&m, &calibration(&setup)).expect_err("refused")
    };
    let invalid = |change: &dyn Fn(&mut Measurement)| {
        assert!(matches!(
            refused(change),
            PipelineError::InvalidParameter(_)
        ));
    };
    assert!(matches!(
        refused(&|m| {
            m.sample_counts.pop();
        }),
        PipelineError::ShapeMismatch(_)
    ));
    assert!(matches!(
        refused(&|m| m.sample_live = Some(vec![1.0; 3])),
        PipelineError::ShapeMismatch(_)
    ));
    invalid(&|m| m.sample_live = Some(vec![0.0; m.sample_counts.len()]));
    invalid(&|m| m.open_live = Some(vec![f64::NAN; m.open_counts.len()]));
    invalid(&|m| m.sample_counts[5] += 0.5);
    invalid(&|m| m.sample_counts[5] = -1.0);
    invalid(&|m| m.sample_counts.iter_mut().for_each(|c| *c = 0.0));
    invalid(&|m| m.charge_ratio = 0.0);
    invalid(&|m| m.charge_ratio = f64::NAN);
    invalid(&|m| m.normalization = Value::Fitted(0.0));
    invalid(&|m| m.normalization = Value::Known(f64::INFINITY));
    invalid(&|m| m.background[1] = Value::Fitted(f64::NAN));
    assert!(matches!(
        refused(&|m| m.background[0] = Value::Fitted(-2.0)),
        PipelineError::UnmodelledCounts { run: "sample", .. }
    ));
    invalid(&|m| m.isotopes.clear());
    invalid(&|m| m.isotopes.push((isotope.clone(), THIN)));
    invalid(&|m| m.isotopes[0].1 = -1.0);
    invalid(&|m| m.isotopes[0].1 = f64::NAN);
    for temperature_k in [0.5, 6000.0, f64::NAN] {
        invalid(&|m| m.temperature_k = Value::Known(temperature_k));
        invalid(&|m| m.temperature_k = Value::Fitted(temperature_k));
    }
    invalid(&|m| m.isotopes[0].0.ranges[0].l_groups[0].resonances[0].energy = f64::NAN);
    invalid(&|m| m.isotopes[0].0.ranges[0].l_groups[0].resonances[0].gg = f64::NAN);
    invalid(&|m| m.isotopes[0].0.ranges[0].target_spin = f64::NAN);
    invalid(&|m| m.isotopes[0].0.ranges[0].energy_high = 20.0);

    let top_ev = FlightTimeGrid::new(&setup.edges, T0_US, &setup.pulse)
        .expect("grid")
        .energies_ev()[0];
    let reach = |temperature_k: f64| {
        let u = DopplerParams::new(temperature_k, isotope.awr)
            .expect("doppler")
            .u();
        (top_ev.sqrt() + SUPPORT_X * u).powi(2)
    };
    let mut between = good.clone();
    between.isotopes[0].0.ranges[0].energy_high = 0.5 * (reach(TEMPERATURE_K) + reach(5000.0));
    assert!(fit_counts(&between, &calibration(&setup)).is_ok());
    assert!(matches!(
        fit_counts(&fitted_from(between, TEMPERATURE_K), &calibration(&setup)),
        Err(PipelineError::InvalidParameter(_))
    ));

    let slow = Setup {
        edges: (1350..=1420).map(f64::from).collect(),
        ..standard()
    };
    let light = synthetic_isotope(1, 1, 1.5, 0.01, 0.06);
    let bins = slow.edges.len() - 1;
    assert!(matches!(
        fit_counts(
            &measurement(
                &slow,
                (vec![100.0; bins], vec![100.0; bins]),
                &[(light, 1e-3)]
            ),
            &calibration(&slow),
        ),
        Err(PipelineError::InvalidParameter(_))
    ));
}

fn two_resonances() -> ResonanceData {
    synthetic_isotope_multi(72, 180, &[(20.0, 0.01, 0.06), (25.0, 0.0002, 0.06)])
}

fn rule_halvings(
    setup: &Setup,
    isotope: &ResonanceData,
    resonance_ev: f64,
    temperature_k: f64,
) -> usize {
    let width_ev = 2.0
        * std::f64::consts::LN_2.sqrt()
        * DopplerParams::new(temperature_k, isotope.awr)
            .expect("doppler")
            .doppler_width(resonance_ev);
    let rule_us = 0.5 * CLOCK / resonance_ev.sqrt() * width_ev / (2.0 * resonance_ev);
    chain(setup)
        .iter()
        .position(|grid| grid.step_us() <= rule_us)
        .expect("a grid within the rule")
}

fn kev_window() -> Setup {
    Setup {
        edges: (52..=69).map(f64::from).collect(),
        pulse: pulse(0.0, 3000.0),
        energy_range_ev: (300.0, 3000.0),
        simulator_step_us: 1.0 / 128.0,
    }
}

#[test]
fn density_and_temperature_are_recovered_from_starts_on_either_side() {
    let setup = standard();
    let cases = [
        (hafnium_like(20.0), THIN, 300.0, vec![200.0, 1000.0]),
        (two_resonances(), SATURATED, 300.0, vec![200.0, 1000.0]),
        (hafnium_like(20.0), THIN, 1500.0, vec![300.0, 3000.0]),
        (two_resonances(), SATURATED, 1500.0, vec![300.0, 3000.0]),
    ];
    for (isotope, density, truth_k, starts_k) in cases {
        let counts = expected_at(&setup, &beam(1.0e6), &[(isotope.clone(), density)], truth_k);
        let counts = (rounded(&counts.0), rounded(&counts.1));
        for start in [0.5 * density, 2.0 * density] {
            for &start_k in &starts_k {
                let fit = fit_counts(
                    &fitted_from(
                        measurement(&setup, counts.clone(), &[(isotope.clone(), start)]),
                        start_k,
                    ),
                    &calibration(&setup),
                )
                .expect("fit");
                let case = format!("{density} at {truth_k} K from {start}, {start_k} K");
                assert!(fit.converged, "{case}");
                assert_eq!(fit.overdispersion, Some(1.0), "{case}");
                let pulls = [
                    (fit.densities[0] - density) / error_bar(&fit, 0),
                    (fit.temperature_k - truth_k) / error_bar(&fit, 1),
                ];
                assert!(
                    pulls.iter().all(|p| p.abs() <= BOUND.sqrt()),
                    "{case}: {pulls:?}"
                );
            }
        }
    }
}

fn kev_beam(level: f64) -> impl Fn(f64) -> f64 {
    move |u: f64| {
        let x = (u / 57.2).ln();
        level * (0.5 * x - 2.0 * x * x).exp()
    }
}

fn kev_resonance_ev(setup: &Setup, offset: f64) -> f64 {
    let first = FlightTimeGrid::new(&setup.edges, T0_US, &setup.pulse).expect("grid");
    let j = first
        .flight_times_us()
        .iter()
        .position(|&u| u > 57.2)
        .expect("a point");
    (CLOCK / (first.flight_times_us()[j] + offset * first.step_us())).powi(2)
}

fn edge_fit(
    setup: &Setup,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
    truth_k: f64,
    start_k: f64,
) -> CountsFit {
    let counts = expected_at(setup, beam, sample, truth_k);
    let fit = fit_counts(
        &fitted_from(
            measurement(setup, (rounded(&counts.0), rounded(&counts.1)), sample),
            start_k,
        ),
        &calibration(setup),
    )
    .expect("fit");
    assert!(fit.converged, "{truth_k} K");
    let covariance = fit.covariance.as_ref().expect("covariance");
    assert!(covariance.data.iter().all(|v| v.is_nan()), "{truth_k} K");
    fit
}

#[test]
fn a_temperature_on_the_box_edge_withholds_the_covariance() {
    let setup = standard();
    let hot = edge_fit(
        &setup,
        &beam(1.0e6),
        &[(two_resonances(), SATURATED)],
        6000.0,
        TEMPERATURE_K,
    );
    assert_eq!(hot.temperature_k, 5000.0);

    let setup = Setup {
        simulator_step_us: 1.0 / 1024.0,
        ..kev_window()
    };
    let beam = kev_beam(1.0e7);
    let isotope = hafnium_like(kev_resonance_ev(&setup, 0.04));
    let cold = edge_fit(&setup, &beam, &[(isotope.clone(), 0.17767)], 0.0, 5000.0);
    assert_eq!(cold.temperature_k, 1.0);
    let fitted = [(isotope, cold.densities[0])];
    let (low, high) = chain(&setup)[0].range_us();
    let fitted_beam = |u: f64| {
        if (low..=high).contains(&u) {
            cold.beam.per_us(u)
        } else {
            0.0
        }
    };
    let reference = expected_at(&setup, &fitted_beam, &fitted, cold.temperature_k);
    let reference: Vec<f64> = reference.0.into_iter().chain(reference.1).collect();
    let on_grid = predicted_at(
        &chain(&setup)[cold.halvings],
        &|u| cold.beam.per_us(u),
        &fitted,
        cold.temperature_k,
    );
    let missed = distance(&on_grid, &reference);
    assert!(missed <= BOUND, "{missed}");
}

#[test]
fn the_grid_is_refined_when_the_fitted_temperature_narrows_the_resonance() {
    let setup = kev_window();
    let beam = kev_beam(1.0e4);
    let resonance_ev = kev_resonance_ev(&setup, 0.24);
    let isotope = hafnium_like(resonance_ev);
    let sample = [(isotope.clone(), 0.17767)];
    assert_eq!(rule_halvings(&setup, &isotope, resonance_ev, 5000.0), 0);
    let simulated = expected(&setup, &beam, &sample);
    let fit = fit_counts(
        &fitted_from(
            measurement(
                &setup,
                (rounded(&simulated.0), rounded(&simulated.1)),
                &sample,
            ),
            5000.0,
        ),
        &calibration(&setup),
    )
    .expect("fit");
    let pull = (fit.temperature_k - TEMPERATURE_K) / error_bar(&fit, 1);
    assert!(pull.abs() <= BOUND.sqrt(), "{pull}");
    let simulated: Vec<f64> = simulated.0.into_iter().chain(simulated.1).collect();
    let resolved = distance(
        &predicted(&chain(&setup)[fit.halvings], &beam, &sample),
        &simulated,
    );
    assert!(resolved <= BOUND, "{resolved}");
}

#[test]
fn a_false_minimum_shows_as_an_overdispersion_far_above_one() {
    let setup = standard();
    let isotope = hafnium_like(20.0);
    let counts = expected_at(&setup, &beam(1.0e6), &[(isotope.clone(), THIN)], 2000.0);
    let fit = fit_counts(
        &fitted_from(
            measurement(
                &setup,
                (rounded(&counts.0), rounded(&counts.1)),
                &[(isotope, 2.0 * THIN)],
            ),
            TEMPERATURE_K,
        ),
        &calibration(&setup),
    )
    .expect("fit");
    assert!(fit.converged);
    assert!(fit.temperature_k < 500.0, "{}", fit.temperature_k);
    let overdispersion = fit.overdispersion.expect("measured");
    assert!(overdispersion > 100.0, "{overdispersion}");
}

mod error_bar_pulls {
    use rayon::prelude::*;

    use super::*;

    const Z: f64 = 3.5;
    const MOST_FAILED: usize = 2;

    struct Ensemble {
        setup: Setup,
        sample: (ResonanceData, f64),
        temperature_k: Value,
        level: f64,
        counts_per_neutron: f64,
        open_run_fraction: f64,
        seeds: std::ops::Range<u64>,
    }

    struct Draw {
        pulls: Vec<f64>,
        estimates: Vec<f64>,
        reported_correlation: Option<f64>,
        overdispersion: f64,
        deviance: f64,
    }

    impl Ensemble {
        fn expected(&self) -> (Vec<f64>, Vec<f64>) {
            let (open, sample) = expected(
                &self.setup,
                &beam(self.level),
                std::slice::from_ref(&self.sample),
            );
            let open = open.iter().map(|mu| mu * self.open_run_fraction).collect();
            (open, sample)
        }

        fn measurement(
            &self,
            expected: &(Vec<f64>, Vec<f64>),
            seed: u64,
            start: (f64, Value),
        ) -> Measurement {
            let counts = draws(expected, seed, self.counts_per_neutron);
            let mut m = measurement(&self.setup, counts, &[(self.sample.0.clone(), start.0)]);
            m.charge_ratio = CHARGE_RATIO / self.open_run_fraction;
            m.temperature_k = start.1;
            m
        }

        fn draw(
            &self,
            expected: &(Vec<f64>, Vec<f64>),
            seed: u64,
            start: (f64, Value),
        ) -> Option<Draw> {
            let measurement = self.measurement(expected, seed, start);
            let fit = fit_counts(&measurement, &calibration(&self.setup)).ok()?;
            let covariance = fit.covariance.as_ref().filter(|_| fit.converged)?;
            let mut estimates = vec![fit.densities[0]];
            let mut truths = vec![self.sample.1];
            if matches!(self.temperature_k, Value::Fitted(_)) {
                estimates.push(fit.temperature_k);
                truths.push(TEMPERATURE_K);
            }
            let pulls: Vec<f64> = estimates
                .iter()
                .zip(&truths)
                .enumerate()
                .map(|(i, (x, truth))| (x - truth) / covariance.get(i, i).sqrt())
                .collect();
            let reported_correlation = (estimates.len() == 2).then(|| {
                covariance.get(0, 1) / (covariance.get(0, 0) * covariance.get(1, 1)).sqrt()
            });
            pulls.iter().all(|p| p.is_finite()).then_some(Draw {
                pulls,
                estimates,
                reported_correlation,
                overdispersion: fit.overdispersion?,
                deviance: fit.deviance,
            })
        }

        fn draws(&self, expected: &(Vec<f64>, Vec<f64>), start: (f64, Value)) -> Vec<Option<Draw>> {
            self.seeds
                .clone()
                .into_par_iter()
                .map(|seed| self.draw(expected, seed, start))
                .collect()
        }

        fn draws_from_truth(&self) -> Vec<Option<Draw>> {
            self.draws(&self.expected(), (self.sample.1, self.temperature_k))
        }
    }

    fn moments(values: &[f64]) -> (f64, f64) {
        let m = values.len() as f64;
        let mean = values.iter().sum::<f64>() / m;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0);
        (mean, variance.sqrt())
    }

    fn check(draws: &[Option<Draw>]) -> Vec<&Draw> {
        let kept: Vec<&Draw> = draws.iter().flatten().collect();
        let failed = draws.len() - kept.len();
        assert!(failed <= MOST_FAILED, "{failed} draws failed");
        let m = kept.len() as f64;
        for quantity in 0..kept[0].pulls.len() {
            let pulls: Vec<f64> = kept.iter().map(|d| d.pulls[quantity]).collect();
            let (mean, sd) = moments(&pulls);
            assert!(mean.abs() <= Z / m.sqrt(), "{quantity}: mean {mean}");
            let band = Z / (2.0 * (m - 1.0)).sqrt();
            assert!(
                (sd - 1.0).abs() <= band,
                "{quantity}: sd {sd} vs 1 ± {band}"
            );
        }
        if let Some(reported) = kept
            .iter()
            .map(|d| d.reported_correlation)
            .collect::<Option<Vec<f64>>>()
        {
            let columns = |i: usize| kept.iter().map(|d| d.estimates[i]).collect::<Vec<f64>>();
            let (n, t) = (columns(0), columns(1));
            let (mn, sn) = moments(&n);
            let (mt, st) = moments(&t);
            let measured = n
                .iter()
                .zip(&t)
                .map(|(a, b)| (a - mn) * (b - mt))
                .sum::<f64>()
                / ((m - 1.0) * sn * st);
            let reported = reported.iter().sum::<f64>() / m;
            let gap = (measured.atanh() - reported.atanh()).abs();
            assert!(
                gap <= Z / (m - 3.0).sqrt(),
                "correlation {measured} vs {reported}"
            );
        }
        kept
    }

    fn overdispersion_matches(kept: &[&Draw], counts_per_neutron: f64) {
        let ratios: Vec<f64> = kept
            .iter()
            .map(|d| d.overdispersion / counts_per_neutron)
            .collect();
        let (mean, sd) = moments(&ratios);
        let band = Z * sd / (ratios.len() as f64).sqrt();
        assert!((mean - 1.0).abs() <= band, "overdispersion {mean} ± {band}");
    }

    fn saturated(
        temperature_k: Value,
        level: f64,
        counts_per_neutron: f64,
        open_run_fraction: f64,
        seeds: std::ops::Range<u64>,
    ) -> Ensemble {
        Ensemble {
            setup: standard(),
            sample: (two_resonances(), SATURATED),
            temperature_k,
            level,
            counts_per_neutron,
            open_run_fraction,
            seeds,
        }
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_thin_sample_s_error_bars_are_its_scatter() {
        let ensemble = Ensemble {
            setup: standard(),
            sample: (hafnium_like(20.0), THIN),
            temperature_k: Value::Fitted(TEMPERATURE_K),
            level: 1.0e5,
            counts_per_neutron: 1.0,
            open_run_fraction: 1.0,
            seeds: 10_000..10_400,
        };
        let expected = ensemble.expected();
        let draws = ensemble.draws(&expected, (THIN, Value::Fitted(TEMPERATURE_K)));
        check(&draws);
        let restarted: Vec<Option<Draw>> = (10_000..10_200)
            .into_par_iter()
            .map(|seed| ensemble.draw(&expected, seed, (2.0 * THIN, Value::Fitted(1000.0))))
            .collect();
        let apart = draws[..restarted.len()]
            .iter()
            .zip(&restarted)
            .filter(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => (a.deviance - b.deviance).abs() > 1e-3,
                _ => true,
            })
            .count();
        assert!(apart <= MOST_FAILED, "{apart} draws depend on the start");
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn compound_counts_scale_the_error_bars_by_the_overdispersion() {
        let ensemble = saturated(
            Value::Fitted(TEMPERATURE_K),
            1.0e4,
            7.0,
            1.0,
            20_000..20_400,
        );
        let draws = ensemble.draws_from_truth();
        let kept = check(&draws);
        overdispersion_matches(&kept, 7.0);
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_short_open_run_widens_the_error_bars_by_its_noise() {
        let ensemble = saturated(
            Value::Fitted(TEMPERATURE_K),
            1.0e4,
            1.0,
            0.1,
            30_000..30_400,
        );
        check(&ensemble.draws_from_truth());
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_few_counts_per_bin_with_empty_bins_give_their_error_bars() {
        let ensemble = saturated(Value::Known(TEMPERATURE_K), 3.0, 1.0, 1.0, 40_000..40_400);
        check(&ensemble.draws_from_truth());
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn the_overdispersion_measures_compound_counts_at_a_few_counts_per_bin() {
        let ensemble = saturated(Value::Known(TEMPERATURE_K), 3.0, 7.0, 1.0, 50_000..50_400);
        let draws = ensemble.draws_from_truth();
        let kept = check(&draws);
        overdispersion_matches(&kept, 7.0);
    }
}
