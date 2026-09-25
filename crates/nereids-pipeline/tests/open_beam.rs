use std::sync::Arc;

use nereids_physics::flight_time_grid::{FlightTimeGrid, FlightTimeGridError};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_pipeline::error::PipelineError;
use nereids_pipeline::open_beam::{BOUND, Calibration, fit_open_beam};
use nereids_pipeline::reference::Instrument;

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

fn open_counts(pulse: &Arc<IkedaCarpenter>, beam: &dyn Fn(f64) -> f64) -> Vec<f64> {
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
    .into_iter()
    .map(f64::round)
    .collect()
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
        let (knot_low, knot_high) = fit.beam.knot_span_us();
        let (u_low, u_high) = chain[accepted].range_us();
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
}

#[test]
fn the_grid_s_refusals_reach_the_caller() {
    let counts = open_counts(&pulses()[0].1, &true_beam(1.0e6));
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
