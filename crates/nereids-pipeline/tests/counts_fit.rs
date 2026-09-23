use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_endf::resonance::test_support::{synthetic_isotope, synthetic_isotope_multi};
use nereids_physics::continuous_doppler;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, ResolutionParams, TOF_FACTOR};
use nereids_pipeline::counts_fit::{
    ACCURACY, Calibration, CountsFit, Measurement, Value, fit_counts,
};
use nereids_pipeline::error::PipelineError;
use nereids_pipeline::reference::Instrument;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Poisson};

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const CHARGE_RATIO: f64 = 1.2;
const DENSITY: f64 = 2.0e-4;
const TEMPERATURE_K: f64 = 300.0;
const STEP_US: f64 = 1.0e-3;
const FIRST_EDGE_US: i32 = 350;
const LAST_EDGE_US: i32 = 470;

fn energy(flight_time_us: f64) -> f64 {
    (TOF_FACTOR * FLIGHT_PATH_M / flight_time_us).powi(2)
}

fn resolution() -> ResolutionFunction {
    ResolutionFunction::IkedaCarpenter(Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                beta: EnergyLaw::Const(0.25),
                r: EnergyLaw::Const(0.15),
                burst_sigma_us: None,
                channel_fwhm_us: None,
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

fn calibration() -> Calibration {
    Calibration {
        flight_path_m: FLIGHT_PATH_M,
        t0_us: T0_US,
        resolution: resolution(),
    }
}

fn log_quadratic_beam(u: f64) -> f64 {
    let x = (u / 400.0).ln();
    1.0e4 * (-0.8 * x - 1.5 * x * x).exp()
}

fn beam_with_dip(u: f64) -> f64 {
    log_quadratic_beam(u) * (1.0 - 0.3 * (-((u - 420.0) / 6.0).powi(2)).exp())
}

struct Truth {
    edges: Vec<f64>,
    open: Vec<f64>,
    sample: Vec<f64>,
}

fn simulate(beam_per_us: fn(f64) -> f64, isotope: &ResonanceData) -> Truth {
    let edges: Vec<f64> = (FIRST_EDGE_US..=LAST_EDGE_US).map(f64::from).collect();
    let instrument = Instrument {
        time_edges_us: edges.clone(),
        flight_path_m: FLIGHT_PATH_M,
        t0_us: T0_US,
        resolution: resolution(),
    };
    let per_ev = |e: f64| {
        let u = TOF_FACTOR * FLIGHT_PATH_M / e.sqrt();
        beam_per_us(u) * u / (2.0 * e)
    };
    let range = (energy(480.0), energy(230.0));
    let open = instrument.expected_counts(&per_ev, &|e| vec![1.0; e.len()], range, STEP_US);
    let transmission = |energies: &[f64]| {
        continuous_doppler::broaden(energies, isotope, TEMPERATURE_K)
            .expect("Doppler")
            .iter()
            .map(|s| (-DENSITY * s).exp())
            .collect()
    };
    let sample =
        instrument.expected_counts(&|e| CHARGE_RATIO * per_ev(e), &transmission, range, STEP_US);
    for p in open.edge_probability.iter().chain(&sample.edge_probability) {
        assert!(
            *p < 1e-9,
            "the simulated range must hold every neutron, lost {p}"
        );
    }
    Truth {
        edges,
        open: open.counts,
        sample: sample.counts,
    }
}

fn fitted(truth: &Truth, isotope: &ResonanceData, density: Value, temperature: Value) -> CountsFit {
    fit_counts(
        &Measurement {
            time_edges_us: truth.edges.clone(),
            open_counts: truth.open.clone(),
            sample_counts: truth.sample.clone(),
            charge_ratio: CHARGE_RATIO,
            isotopes: vec![(isotope.clone(), density)],
            temperature_k: temperature,
        },
        &calibration(),
    )
    .expect("fit")
}

fn assert_sample_recovered(fit: &CountsFit, density: Value, temperature: Value) {
    assert!(
        fit.converged,
        "{density:?}, {temperature:?} did not converge"
    );
    for (label, value, got, sigma, truth) in [
        (
            "density",
            density,
            fit.densities[0],
            fit.density_uncertainties[0],
            DENSITY,
        ),
        (
            "temperature",
            temperature,
            fit.temperature_k,
            fit.temperature_uncertainty,
            TEMPERATURE_K,
        ),
    ] {
        match value {
            Value::Known(v) => assert_eq!(got, v, "a known {label} changed"),
            Value::Fitted(_) => {
                let sigma = sigma.unwrap_or_else(|| panic!("the fitted {label} has no error"));
                let pull = (got - truth) / sigma;
                assert!(pull.abs() < 0.1, "{label} {got} is {pull:.3}σ off");
            }
        }
    }
}

fn assert_recovered(fit: &CountsFit, density: Value, temperature: Value) {
    assert_sample_recovered(fit, density, temperature);
    assert!(
        fit.deviance < ACCURACY * ACCURACY,
        "the fitted counts differ from the simulated ones by a deviance of {}",
        fit.deviance
    );
}

fn assert_beam(fit: &CountsFit, truth: &Truth, beam_per_us: fn(f64) -> f64, noise_fraction: f64) {
    for (bin, &counts) in truth.edges.windows(2).zip(&truth.open) {
        let u = 0.5 * (bin[0] + bin[1]) - T0_US;
        let relative = fit.beam.per_us(u) / beam_per_us(u) - 1.0;
        let bound = noise_fraction / counts.sqrt();
        assert!(
            relative.abs() <= bound,
            "the beam at {u} µs is off by {relative:.2e}, more than {bound:.2e}"
        );
    }
}

#[test]
fn density_temperature_and_both_are_recovered_from_simulated_counts() {
    let isotope = synthetic_isotope(72, 178, 20.0, 0.02, 0.06);
    let truth = simulate(log_quadratic_beam, &isotope);
    for (density, temperature) in [
        (Value::Fitted(0.5 * DENSITY), Value::Known(TEMPERATURE_K)),
        (Value::Known(DENSITY), Value::Fitted(200.0)),
        (Value::Fitted(0.5 * DENSITY), Value::Fitted(200.0)),
        (Value::Fitted(0.5 * DENSITY), Value::Fitted(1000.0)),
    ] {
        let fit = fitted(&truth, &isotope, density, temperature);
        assert_recovered(&fit, density, temperature);
    }
}

#[test]
fn the_points_are_doubled_until_the_counts_stop_moving() {
    let isotope = synthetic_isotope(72, 178, 20.0, 0.02, 0.06);
    let truth = simulate(log_quadratic_beam, &isotope);
    let (density, temperature) = (Value::Fitted(0.5 * DENSITY), Value::Fitted(200.0));
    let fit = fitted(&truth, &isotope, density, temperature);
    assert!(
        fit.accuracy[0] > ACCURACY,
        "the first points must be too coarse for this to test anything, moved {}",
        fit.accuracy[0]
    );
    assert!(*fit.accuracy.last().unwrap() <= ACCURACY);
    assert_recovered(&fit, density, temperature);
}

#[test]
fn only_resonances_whose_neutrons_reach_the_bins_are_given_points() {
    let near = (energy(344.0), 0.05, 0.06);
    let central = (20.0, 0.02, 0.06);
    let far = (energy(305.0), 0.05, 0.06);
    let mut points = Vec::new();
    for resonances in [vec![near, central], vec![near, central, far]] {
        let isotope = synthetic_isotope_multi(72, 178, &resonances);
        let truth = simulate(log_quadratic_beam, &isotope);
        let (density, temperature) = (Value::Fitted(0.5 * DENSITY), Value::Fitted(200.0));
        let fit = fitted(&truth, &isotope, density, temperature);
        assert!(fit.skipped <= ACCURACY);
        assert_recovered(&fit, density, temperature);
        points.push(fit.points);
    }
    assert_eq!(points[0], points[1], "the far resonance was given points");
}

#[test]
fn the_fitted_beam_is_the_beam_before_the_blur() {
    let isotope = synthetic_isotope(72, 178, 20.0, 0.02, 0.06);
    let (density, temperature) = (Value::Fitted(0.5 * DENSITY), Value::Known(TEMPERATURE_K));
    for (beam, noise_fraction) in [
        (log_quadratic_beam as fn(f64) -> f64, ACCURACY),
        (beam_with_dip, 1.0),
    ] {
        let truth = simulate(beam, &isotope);
        let fit = fitted(&truth, &isotope, density, temperature);
        assert_beam(&fit, &truth, beam, noise_fraction);
        assert_sample_recovered(&fit, density, temperature);
    }
}

#[test]
fn the_reported_errors_match_the_scatter_of_repeated_measurements() {
    const DRAWS: usize = 40;
    let isotope = synthetic_isotope(72, 178, 20.0, 0.02, 0.06);
    let truth = simulate(log_quadratic_beam, &isotope);
    let mut rng = ChaCha12Rng::seed_from_u64(20260923);
    let mut draw = |expected: &[f64]| -> Vec<f64> {
        expected
            .iter()
            .map(|&mu| Poisson::new(mu).expect("positive rate").sample(&mut rng))
            .collect()
    };
    let (mut density_pulls, mut temperature_pulls) = (Vec::new(), Vec::new());
    for _ in 0..DRAWS {
        let noisy = Truth {
            edges: truth.edges.clone(),
            open: draw(&truth.open),
            sample: draw(&truth.sample),
        };
        let fit = fitted(
            &noisy,
            &isotope,
            Value::Fitted(DENSITY),
            Value::Fitted(TEMPERATURE_K),
        );
        assert!(fit.converged);
        density_pulls.push((fit.densities[0] - DENSITY) / fit.density_uncertainties[0].unwrap());
        temperature_pulls
            .push((fit.temperature_k - TEMPERATURE_K) / fit.temperature_uncertainty.unwrap());
    }
    for (label, pulls) in [
        ("density", density_pulls),
        ("temperature", temperature_pulls),
    ] {
        let n = pulls.len() as f64;
        let mean = pulls.iter().sum::<f64>() / n;
        let spread = (pulls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        assert!(
            mean.abs() < 3.0 / n.sqrt(),
            "{label} pulls average {mean:.3}"
        );
        assert!(
            (spread - 1.0).abs() < 3.0 / (2.0 * n).sqrt(),
            "{label} pulls spread {spread:.3}, not 1"
        );
    }
}

#[test]
fn a_gaussian_resolution_cannot_describe_counts_in_time_bins() {
    let isotope = synthetic_isotope(72, 178, 20.0, 0.02, 0.06);
    let edges: Vec<f64> = (FIRST_EDGE_US..=LAST_EDGE_US).map(f64::from).collect();
    let counts = vec![100.0; edges.len() - 1];
    let result = fit_counts(
        &Measurement {
            time_edges_us: edges,
            open_counts: counts.clone(),
            sample_counts: counts,
            charge_ratio: CHARGE_RATIO,
            isotopes: vec![(isotope, Value::Fitted(DENSITY))],
            temperature_k: Value::Known(TEMPERATURE_K),
        },
        &Calibration {
            resolution: ResolutionFunction::Gaussian(
                ResolutionParams::new(FLIGHT_PATH_M, 0.5, 0.005, 0.0).expect("valid"),
            ),
            ..calibration()
        },
    );
    assert!(matches!(result, Err(PipelineError::BinWeights(_))));
}

#[test]
fn a_temperature_the_counts_cannot_show_is_refused() {
    let isotope = synthetic_isotope(72, 178, 20.0, 0.02, 0.06);
    let edges: Vec<f64> = (FIRST_EDGE_US..=LAST_EDGE_US).map(f64::from).collect();
    let counts = vec![100.0; edges.len() - 1];
    for (density, temperature) in [
        (Value::Known(0.0), Value::Fitted(TEMPERATURE_K)),
        (Value::Fitted(DENSITY), Value::Fitted(0.5)),
    ] {
        let result = fit_counts(
            &Measurement {
                time_edges_us: edges.clone(),
                open_counts: counts.clone(),
                sample_counts: counts.clone(),
                charge_ratio: CHARGE_RATIO,
                isotopes: vec![(isotope.clone(), density)],
                temperature_k: temperature,
            },
            &calibration(),
        );
        assert!(
            matches!(result, Err(PipelineError::InvalidParameter(_))),
            "{density:?}, {temperature:?} was accepted"
        );
    }
}
