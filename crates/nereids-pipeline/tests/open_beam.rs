use std::sync::Arc;

use nereids_physics::flight_time_grid::{FlightTimeGrid, FlightTimeGridError};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_pipeline::beam::BeamSpline;
use nereids_pipeline::error::PipelineError;
use nereids_pipeline::open_beam::{BOUND, Calibration, OpenBeamFit, fit_open_beam};
use nereids_pipeline::reference::Instrument;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Poisson};

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const CLOCK: f64 = TOF_FACTOR * FLIGHT_PATH_M;
const E_MIN_EV: f64 = 1.0;
const E_MAX_EV: f64 = 200.0;
const SIMULATOR_STEP_US: f64 = 1.0 / 32.0;

fn edges() -> Vec<f64> {
    (350..=470).map(f64::from).collect()
}

fn pulse(
    alpha: EnergyLaw,
    beta: EnergyLaw,
    r: EnergyLaw,
    channel_fwhm_us: Option<f64>,
) -> Arc<IkedaCarpenter> {
    Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha,
                beta,
                r,
                burst_sigma_us: None,
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
    let c = EnergyLaw::Const;
    vec![
        ("constant", pulse(c(0.565), c(0.25), c(0.15), None)),
        (
            "energy laws",
            pulse(
                EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
                EnergyLaw::SqrtE { a0: 0.02, a1: 0.2 },
                EnergyLaw::ExpMilliEv { kappa: 5.0e4 },
                None,
            ),
        ),
        ("folded", pulse(c(0.565), c(0.25), c(0.15), Some(2.0))),
    ]
}

fn true_beam(counts_per_us: f64) -> impl Fn(f64) -> f64 {
    move |u: f64| {
        let x = (u / 400.0).ln();
        counts_per_us * (0.5 * x - 2.0 * x * x).exp()
    }
}

fn simulated(pulse: &Arc<IkedaCarpenter>, beam: &dyn Fn(f64) -> f64) -> Vec<f64> {
    Instrument {
        time_edges_us: edges(),
        flight_path_m: FLIGHT_PATH_M,
        t0_us: T0_US,
        resolution: ResolutionFunction::IkedaCarpenter(Arc::clone(pulse)),
    }
    .expected_counts(
        &|e| {
            let u = CLOCK / e.sqrt();
            beam(u) * u / (2.0 * e)
        },
        &|es| vec![1.0; es.len()],
        (E_MIN_EV, E_MAX_EV),
        SIMULATOR_STEP_US,
    )
    .counts
}

fn open_counts(pulse: &Arc<IkedaCarpenter>, beam: &dyn Fn(f64) -> f64) -> Vec<f64> {
    simulated(pulse, beam).into_iter().map(f64::round).collect()
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

fn dipped(centre_us: f64, fwhm_us: f64) -> impl Fn(f64) -> f64 {
    let smooth = true_beam(1.0e6);
    let sigma = fwhm_us / (8.0 * 2.0_f64.ln()).sqrt();
    move |u: f64| smooth(u) * (1.0 - 0.6 * (-(u - centre_us).powi(2) / (2.0 * sigma * sigma)).exp())
}

fn distance_from(pulse: &Arc<IkedaCarpenter>, beam: &BeamSpline, expected: &[f64]) -> f64 {
    let (low, high) = FlightTimeGrid::new(&edges(), T0_US, pulse)
        .expect("grid")
        .range_us();
    let on_range = |u: f64| {
        if (low..=high).contains(&u) {
            beam.per_us(u)
        } else {
            0.0
        }
    };
    simulated(pulse, &on_range)
        .iter()
        .zip(expected)
        .map(|(fitted, mu)| (fitted - mu).powi(2) / mu)
        .sum()
}

fn richest_coefficients(bins: usize) -> usize {
    (0..)
        .map(|p| (1_usize << p) + 3)
        .take_while(|k| 2 * k <= bins)
        .last()
        .expect("a candidate")
}

fn calibration(pulse: &Arc<IkedaCarpenter>) -> Calibration {
    Calibration {
        t0_us: T0_US,
        pulse: Arc::clone(pulse),
    }
}

fn bin_centres() -> Vec<f64> {
    edges()
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]) - T0_US)
        .collect()
}

fn predicted(grid: &FlightTimeGrid, beam: &dyn Fn(f64) -> f64) -> Vec<f64> {
    let values: Vec<f64> = grid.flight_times_us().iter().map(|&u| beam(u)).collect();
    grid.predict(&values).expect("one value per point")
}

#[test]
fn the_fitted_beam_is_the_beam_before_the_blur() {
    let beam = true_beam(1.0e6);
    for (name, pulse) in pulses() {
        let fit =
            fit_open_beam(&edges(), &open_counts(&pulse, &beam), &calibration(&pulse)).expect(name);
        assert!(fit.converged, "{name}");
        assert_eq!(fit.beam.intervals(), 1, "{name}");
        assert_eq!(fit.overdispersion, Some(1.0), "{name}");
        for u in bin_centres() {
            let error = fit.beam.per_us(u) / beam(u) - 1.0;
            assert!(error.abs() <= 1e-6, "{name} at {u} µs: {error}");
        }
    }
}

#[test]
fn the_fit_is_on_the_finer_grid_of_the_first_pair_halving_leaves_unchanged() {
    let pulse = &pulses()[0].1;
    let mut halvings = Vec::new();
    for level in [1.0e6, 1.0e8] {
        let counts = open_counts(pulse, &true_beam(level));
        let fit = fit_open_beam(&edges(), &counts, &calibration(pulse)).expect("fit");
        let beam = |u: f64| fit.beam.per_us(u);
        let chain: Vec<FlightTimeGrid> = std::iter::successors(
            Some(FlightTimeGrid::new(&edges(), T0_US, pulse).expect("grid")),
            |g| g.halved().ok(),
        )
        .collect();
        let accepted = 1 + chain
            .windows(2)
            .position(|pair| {
                let (coarse, fine) = (predicted(&pair[0], &beam), predicted(&pair[1], &beam));
                fine.iter()
                    .zip(&coarse)
                    .map(|(f, c)| (f - c).powi(2) / f)
                    .sum::<f64>()
                    <= BOUND
            })
            .expect("a pair within the bound");
        assert_eq!(fit.halvings, accepted, "{level:e}");
        assert_eq!(
            fit.points,
            chain[accepted].flight_times_us().len(),
            "{level:e}"
        );
        assert_eq!(fit.step_us, chain[accepted].step_us(), "{level:e}");
        let distance: f64 = predicted(&chain[accepted], &beam)
            .iter()
            .zip(simulated(pulse, &beam))
            .map(|(p, s)| (p - s).powi(2) / s)
            .sum();
        assert!(distance <= BOUND, "{level:e}: {distance}");
        let (knot_low, knot_high) = fit.beam.knot_span_us();
        let (u_low, u_high) = (edges()[0] - T0_US, chain[accepted].range_us().1);
        assert!(
            (knot_low / u_low - 1.0).abs() <= 1e-12 && (knot_high / u_high - 1.0).abs() <= 1e-12,
            "{level:e}"
        );
        let deviance: f64 = predicted(&chain[accepted], &beam)
            .iter()
            .zip(&counts)
            .map(|(&mu, &y)| {
                let d = (y - mu) / mu;
                mu * d * d * (0.5 - d / 6.0 + d * d / 12.0)
            })
            .sum();
        assert!(
            (fit.deviance - deviance).abs() <= 1e-2 * deviance,
            "{level:e}: {} vs {deviance}",
            fit.deviance
        );
        halvings.push(fit.halvings);
    }
    assert!(halvings[1] > halvings[0], "{halvings:?}");
}

#[test]
fn counts_that_are_not_an_open_beam_are_refused() {
    let pulse = &pulses()[0].1;
    let counts = open_counts(pulse, &true_beam(1.0e6));
    let with = |bin: usize, value: f64| {
        let mut c = counts.clone();
        c[bin] = value;
        c
    };
    for bad in [
        with(5, counts[5] + 0.5),
        with(5, -1.0),
        with(5, f64::NAN),
        vec![0.0; counts.len()],
    ] {
        assert!(matches!(
            fit_open_beam(&edges(), &bad, &calibration(pulse)),
            Err(PipelineError::InvalidParameter(_))
        ));
    }
    assert!(matches!(
        fit_open_beam(&edges(), &counts[1..], &calibration(pulse)),
        Err(PipelineError::ShapeMismatch(_))
    ));
    assert!(matches!(
        fit_open_beam(&edges()[..8], &counts[..7], &calibration(pulse)),
        Err(PipelineError::InvalidParameter(_))
    ));
    assert!(fit_open_beam(&edges()[..9], &counts[..8], &calibration(pulse)).is_ok());
}

#[test]
fn a_smooth_beam_keeps_one_interval_in_most_poisson_draws() {
    let pulse = &pulses()[0].1;
    let expected = simulated(pulse, &true_beam(1.0e6));
    let kept = (0..20)
        .filter(|&seed| {
            fit_open_beam(&edges(), &draw(&expected, seed, 1.0), &calibration(pulse))
                .expect("fit")
                .beam
                .intervals()
                == 1
        })
        .count();
    assert!(kept >= 12, "{kept} of 20");
}

#[test]
fn the_overdispersion_is_the_variance_over_the_poisson_variance() {
    let pulse = &pulses()[0].1;
    let counts_per_neutron = 7.0;
    let expected = simulated(pulse, &true_beam(1.0e6));
    let fits: Vec<OpenBeamFit> = (100..110)
        .map(|seed| {
            fit_open_beam(
                &edges(),
                &draw(&expected, seed, counts_per_neutron),
                &calibration(pulse),
            )
            .expect("fit")
        })
        .collect();
    let draws = fits.len() as f64;
    let mean = fits
        .iter()
        .map(|fit| fit.overdispersion.expect("converged") / counts_per_neutron)
        .sum::<f64>()
        / draws;
    let freedom = (expected.len() - richest_coefficients(expected.len())) as f64;
    let bound = 3.0 * (2.0 / freedom).sqrt() / draws.sqrt();
    assert!((mean - 1.0).abs() <= bound, "{mean} vs 1 ± {bound}");
    let at_limit = fits.iter().filter(|fit| fit.at_limit).count();
    assert!(at_limit <= 3, "{at_limit} of 10 at the limit");
}

#[test]
fn a_low_count_open_beam_is_fitted_and_its_overdispersion_measured() {
    let pulse = &pulses()[0].1;
    let counts_per_neutron = 3.0;
    let expected = simulated(pulse, &true_beam(5.0));
    let ratios: Vec<f64> = (1000..1020)
        .map(|seed| {
            fit_open_beam(
                &edges(),
                &draw(&expected, seed, counts_per_neutron),
                &calibration(pulse),
            )
            .expect("fit")
            .overdispersion
            .expect("converged")
                / counts_per_neutron
        })
        .collect();
    let draws = ratios.len() as f64;
    let mean = ratios.iter().sum::<f64>() / draws;
    let freedom = (expected.len() - richest_coefficients(expected.len())) as f64;
    let bound = 3.0 * (2.0 / freedom).sqrt() / draws.sqrt();
    assert!((mean - 1.0).abs() <= bound, "{mean} vs 1 ± {bound}");
}

#[test]
fn counting_every_neutron_seven_times_scales_the_overdispersion_not_the_error_bars() {
    let pulse = &pulses()[0].1;
    let once = draw(&simulated(pulse, &true_beam(1.0e6)), 300, 3.0);
    let seven: Vec<f64> = once.iter().map(|c| 7.0 * c).collect();
    let fit = |counts: &[f64]| fit_open_beam(&edges(), counts, &calibration(pulse)).expect("fit");
    let (a, b) = (fit(&once), fit(&seven));
    assert_eq!(a.beam.intervals(), b.beam.intervals());
    let ratio = b.overdispersion.expect("converged") / a.overdispersion.expect("converged");
    assert!((ratio / 7.0 - 1.0).abs() <= 1e-3, "{ratio}");
    let (a, b) = (
        a.covariance.expect("covariance"),
        b.covariance.expect("covariance"),
    );
    for (x, y) in a.data.iter().zip(&b.data) {
        assert!((y - x).abs() <= 1e-3 * x.abs(), "{x} vs {y}");
    }
}

#[test]
fn a_dip_the_candidates_can_follow_is_followed() {
    let pulse = &pulses()[0].1;
    for (centre_us, fwhm_us) in [(407.0, 40.0), (380.0, 30.0)] {
        let expected = simulated(pulse, &dipped(centre_us, fwhm_us));
        let rounded: Vec<f64> = expected.iter().map(|mu| mu.round()).collect();
        let noiseless = fit_open_beam(&edges(), &rounded, &calibration(pulse)).expect("fit");
        assert!(
            noiseless.beam.intervals() > 1 && !noiseless.at_limit,
            "{centre_us}"
        );
        let k = noiseless.beam.coefficients().len() as f64;
        let distance = distance_from(pulse, &noiseless.beam, &expected);
        assert!(
            distance <= k + 4.0 * (2.0 * k).sqrt(),
            "{centre_us}: {distance}"
        );
    }
    let expected = simulated(pulse, &dipped(407.0, 40.0));
    for seed in 200..206 {
        let fit =
            fit_open_beam(&edges(), &draw(&expected, seed, 1.0), &calibration(pulse)).expect("fit");
        assert!(fit.beam.intervals() > 1 && !fit.at_limit, "{seed}");
        let k = fit.beam.coefficients().len() as f64;
        let distance = distance_from(pulse, &fit.beam, &expected);
        assert!(distance <= k + 4.0 * (2.0 * k).sqrt(), "{seed}: {distance}");
    }
}

#[test]
fn a_dip_finer_than_every_candidate_is_reported_at_the_limit() {
    let pulse = &pulses()[0].1;
    let expected = simulated(pulse, &dipped(407.0, 4.7));
    let fit =
        fit_open_beam(&edges(), &draw(&expected, 200, 1.0), &calibration(pulse)).expect("fit");
    assert!(fit.at_limit);
    let k = fit.beam.coefficients().len() as f64;
    let distance = distance_from(pulse, &fit.beam, &expected);
    assert!(distance > k + 4.0 * (2.0 * k).sqrt(), "{distance}");
}

#[test]
fn counts_that_cannot_determine_the_beam_are_reported_undetermined() {
    let pulse = &pulses()[0].1;
    for (bin, count) in [(60, 1.0), (119, 3.0)] {
        let mut counts = vec![0.0; edges().len() - 1];
        counts[bin] = count;
        let fit = fit_open_beam(&edges(), &counts, &calibration(pulse)).expect("a fit");
        let determined = fit.converged
            && fit
                .covariance
                .is_some_and(|c| (0..4).all(|i| c.get(i, i).is_finite()));
        assert!(!determined, "bin {bin}");
        assert!(fit.overdispersion.is_none_or(f64::is_finite), "bin {bin}");
    }
}

#[test]
fn a_richer_beam_the_counts_or_the_grid_cannot_resolve_ends_the_ladder() {
    let pulse = &pulses()[0].1;
    let sparse = |bins: &[(usize, f64)]| {
        let mut counts = vec![0.0; edges().len() - 1];
        for &(bin, count) in bins {
            counts[bin] = count;
        }
        fit_open_beam(&edges(), &counts, &calibration(pulse)).expect("fit")
    };
    let unconverged_next = sparse(&[(10, 2.0), (90, 2.0), (100, 1.0)]);
    assert!(unconverged_next.converged && unconverged_next.at_limit);
    assert!(sparse(&[(0, 1.0), (30, 1.0), (60, 1.0)]).converged);
    let infinite_next = draw(&simulated(pulse, &true_beam(0.5)), 20069, 1.0);
    assert!(
        fit_open_beam(&edges(), &infinite_next, &calibration(pulse))
            .expect("fit")
            .converged
    );
    let past_the_point_cap: Vec<f64> = simulated(pulse, &dipped(360.0, 15.0))
        .iter()
        .map(|mu| mu.round())
        .collect();
    assert!(
        fit_open_beam(&edges(), &past_the_point_cap, &calibration(pulse))
            .expect("fit")
            .at_limit
    );
}

#[test]
fn the_grid_s_refusals_reach_the_caller() {
    let counts = open_counts(&pulses()[0].1, &true_beam(1.0e6));
    assert!(matches!(
        fit_open_beam(&[350.0], &[], &calibration(&pulses()[0].1)),
        Err(PipelineError::FlightTimeGrid(
            FlightTimeGridError::InvalidTimeEdges
        ))
    ));
    let c = EnergyLaw::Const;
    let lengthening = pulse(
        EnergyLaw::SqrtE { a0: -0.05, a1: 1.2 },
        c(0.25),
        c(0.15),
        None,
    );
    assert!(matches!(
        fit_open_beam(&edges(), &counts, &calibration(&lengthening)),
        Err(PipelineError::FlightTimeGrid(
            FlightTimeGridError::LengthensWithEnergy { .. }
        ))
    ));
    let near_the_point_cap = pulse(c(700.0), c(0.25), c(0.0), None);
    assert!(matches!(
        fit_open_beam(&edges(), &counts, &calibration(&near_the_point_cap)),
        Err(PipelineError::FlightTimeGrid(
            FlightTimeGridError::TooManyPoints { .. }
        ))
    ));
}
