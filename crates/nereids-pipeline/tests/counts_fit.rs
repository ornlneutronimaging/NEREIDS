use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, LazyLock};

use ndarray::{Array2, Array3, s};
use nereids_endf::resonance::ResonanceData;
use nereids_endf::resonance::test_support::{synthetic_isotope, synthetic_isotope_multi};
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::{Prior, Unbounded};
use nereids_fitting::statistics::{self, Consistency};
use nereids_physics::continuous_doppler::SUPPORT_X;
use nereids_physics::doppler::DopplerParams;
use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_physics::transmission::broadened_cross_sections;
use nereids_pipeline::counts_fit::{
    CountsFit, Material, Measurement, NEGLIGIBLE_PREDICTION, Region, RegionFit, Value, fit_counts,
};
use nereids_pipeline::counts_map::{CountsMap, MapMeasurement, Patch, fit_map};
use nereids_pipeline::error::PipelineError;
use nereids_pipeline::open_beam::{BOUND, Calibration, Pulse, fit_open_beam};
use nereids_pipeline::pulse_calibration::{Provenance, PulseCalibration};
use nereids_pipeline::reference::Instrument;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Normal, Poisson};

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;
const CLOCK: f64 = TOF_FACTOR * FLIGHT_PATH_M;
const TEMPERATURE_K: f64 = 300.0;
const CHARGE_RATIO: f64 = 1.2;
const THIN: f64 = 7.687e-4;
const SATURATED: f64 = 3.844e-2;
const TERMS: [f64; 4] = [0.93, 0.05, 0.5, 0.01];
const T0_STEP_US: f64 = 3e-3;
const PATH_STEP_M: f64 = 1e-4;

struct Setup {
    edges: Vec<f64>,
    pulse: Arc<IkedaCarpenter>,
    energy_range_ev: (f64, f64),
    simulator_step_us: f64,
    t0_us: f64,
    flight_path_m: f64,
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
        t0_us: T0_US,
        flight_path_m: FLIGHT_PATH_M,
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
        flight_path_m: setup.flight_path_m,
        t0_us: setup.t0_us,
        resolution: ResolutionFunction::IkedaCarpenter(Arc::clone(&setup.pulse)),
    };
    let clock = TOF_FACTOR * setup.flight_path_m;
    let per_ev = |e: f64| {
        let u = clock / e.sqrt();
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
    if let Some(sigma) = COMPUTED.with_borrow(|computed| computed.get(&key).cloned()) {
        return sigma;
    }
    let sigma =
        broadened_cross_sections(energies, isotopes, kelvin, None, None).expect("cross sections");
    COMPUTED.with_borrow_mut(|computed| computed.insert(key, sigma.clone()));
    sigma
}

fn with_background(
    setup: &Setup,
    beam: &dyn Fn(f64) -> f64,
    sample: &[(ResonanceData, f64)],
    temperature_k: f64,
    [a, b0, b1, b2]: [f64; 4],
) -> (Vec<f64>, Vec<f64>) {
    let (open, transmitted) = expected_at(setup, beam, sample, temperature_k);
    let clock = TOF_FACTOR * setup.flight_path_m;
    let scattered = |u: f64| a * beam(u) * (b0 + b1 * u / clock + b2 * clock / u);
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
    m.regions[0].open_live = Some(open_live);
    m.regions[0].sample_live = Some(sample_live);
    m
}

fn known(pulse: &IkedaCarpenter) -> Pulse {
    let coefficients = |law: &EnergyLaw| match *law {
        EnergyLaw::SqrtE { a0, a1 } => [a0, a1],
        EnergyLaw::Const(c) => [0.0, c],
        ref other => panic!("{other:?} is not a law in √E"),
    };
    let (params, detector) = (pulse.params(), pulse.detector_pulse());
    Pulse {
        alpha: coefficients(&params.alpha).map(Value::Known),
        beta: coefficients(&params.beta).map(Value::Known),
        r: Value::Known(coefficients(&params.r)[1]),
        fwhm_squared_us2: Value::Known(params.channel_fwhm_us.unwrap_or(0.0).powi(2)),
        energy_span_ev: detector.energy_span_ev(),
        n_tau: detector.n_tau(),
        line_span_ev: None,
        prior: None,
    }
}

fn calibration(setup: &Setup) -> Calibration {
    Calibration {
        t0_us: Value::Known(setup.t0_us),
        flight_path_m: Value::Known(setup.flight_path_m),
        pulse: known(&setup.pulse),
    }
}

fn measurement(
    setup: &Setup,
    (open, sample): (Vec<f64>, Vec<f64>),
    isotopes: &[(ResonanceData, f64)],
) -> Measurement {
    Measurement {
        time_edges_us: setup.edges.clone(),
        charge_ratio: CHARGE_RATIO,
        normalization: Value::Known(1.0),
        regions: vec![Region {
            open_counts: open,
            sample_counts: sample,
            open_live: None,
            sample_live: None,
            background: [Value::Known(0.0); 3],
            material: Some(Material {
                isotopes: isotopes
                    .iter()
                    .map(|(d, n)| (d.clone(), Value::Fitted(*n)))
                    .collect(),
                temperature_k: Value::Known(TEMPERATURE_K),
            }),
        }],
    }
}

trait OneMaterial {
    fn material_mut(&mut self) -> &mut Material;
}

impl OneMaterial for Measurement {
    fn material_mut(&mut self) -> &mut Material {
        self.regions[0].material.as_mut().expect("a material")
    }
}

trait OneTemperature {
    fn temperature(&self) -> f64;
}

impl OneTemperature for CountsFit {
    fn temperature(&self) -> f64 {
        self.regions[0].temperature_k.expect("a material")
    }
}

fn fitted_from(mut measurement: Measurement, start_k: f64) -> Measurement {
    measurement.material_mut().temperature_k = Value::Fitted(start_k);
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
fn densities_normalization_and_background_are_recovered_and_follow_a_density_on_its_bound() {
    let setup = standard();
    for (truth, b2) in [
        (THIN, Value::Known(TERMS[3])),
        (SATURATED, Value::Fitted(0.0)),
    ] {
        let isotope = hafnium_like(20.0);
        let sample = [(isotope.clone(), truth)];
        let counts = with_background(&setup, &beam(1.0e6), &sample, TEMPERATURE_K, TERMS);
        let fit = |(density, a, b0, b1): (Value, f64, f64, f64)| {
            let mut m = recorded(&setup, counts.clone(), &sample);
            m.material_mut().isotopes[0].1 = density;
            m.normalization = Value::Fitted(a);
            m.regions[0].background = [Value::Fitted(b0), Value::Fitted(b1), b2];
            let fit = fit_counts(&m, &calibration(&setup)).expect("fit");
            assert!(fit.converged, "{truth} from {density:?}");
            assert_eq!(
                fit.regions[0].overdispersion,
                [Some(1.0); 2],
                "{truth} from {density:?}"
            );
            fit
        };
        let estimates = |fit: &CountsFit| {
            [
                fit.regions[0].densities[0],
                fit.normalization,
                fit.regions[0].background[0],
                fit.regions[0].background[1],
                fit.regions[0].background[2],
            ]
        };
        let starts = [
            (Value::Fitted(0.5 * truth), 1.0, 0.02, 0.2),
            (Value::Fitted(2.0 * truth), 0.85, 0.12, 0.8),
        ];
        let free = starts.map(fit);
        for fit in &free {
            assert!(!fit.on_bound.contains(&true), "{truth}");
            let pulls: Vec<f64> = estimates(fit)
                .iter()
                .zip([truth, TERMS[0], TERMS[1], TERMS[2], TERMS[3]])
                .take(fit.on_bound.len())
                .enumerate()
                .map(|(i, (x, t))| (x - t) / error_bar(fit, i))
                .collect();
            assert!(
                pulls.iter().all(|p| p.abs() <= BOUND.sqrt()),
                "{truth}: {pulls:?}"
            );
        }

        let free = &free[0];
        let upper = truth - 3.0 * error_bar(free, 0);
        let density = Value::Within {
            start: 0.5 * truth,
            lower: 0.25 * truth,
            upper,
        };
        let bounded = fit((density, 1.0, 0.02, 0.2));
        assert_eq!(bounded.regions[0].densities[0], upper, "{truth}");
        assert!(bounded.on_bound[0], "{truth}");
        assert!(!bounded.on_bound[1..].contains(&true), "{truth}");
        let covariance = free.covariance.as_ref().expect("covariance");
        let (x, x_free) = (estimates(&bounded), estimates(free));
        for i in 1..free.on_bound.len() {
            let regression = covariance.get(i, 0) / covariance.get(0, 0);
            let conditional = x_free[i] + regression * (upper - x_free[0]);
            let pull = (x[i] - conditional) / error_bar(free, i);
            assert!(pull.abs() <= BOUND.sqrt(), "{truth} {i}: {pull}");
        }
    }
}

#[test]
fn an_empty_area_with_a_known_background_adds_its_counts_information_on_the_normalization() {
    let setup = standard();
    let (hot, other) = (
        hafnium_like(20.0),
        synthetic_isotope(74, 182, 20.3, 0.01, 0.06),
    );
    let terms = [TERMS, [TERMS[0], 0.08, 0.0, 0.0], [TERMS[0], 0.0, 0.0, 0.0]];
    let counts = [
        with_background(
            &setup,
            &beam(1.0e6),
            &[(hot.clone(), THIN)],
            1500.0,
            terms[0],
        ),
        with_background(
            &setup,
            &beam(1.5e6),
            &[(other.clone(), 2.0 * THIN)],
            TEMPERATURE_K,
            terms[1],
        ),
        with_background(&setup, &beam(2.0e6), &[], TEMPERATURE_K, terms[2]),
    ];
    let region = |r: usize, material: Option<Material>, background: [Value; 3]| Region {
        background,
        material,
        ..recorded(&setup, counts[r].clone(), &[]).regions.remove(0)
    };
    let regions = [
        region(
            0,
            Some(Material {
                isotopes: vec![(hot, Value::Fitted(0.5 * THIN))],
                temperature_k: Value::Fitted(1000.0),
            }),
            [
                Value::Fitted(0.02),
                Value::Fitted(0.2),
                Value::Known(TERMS[3]),
            ],
        ),
        region(
            1,
            Some(Material {
                isotopes: vec![(
                    other,
                    Value::Measured {
                        value: 2.0 * THIN,
                        sd: 0.1 * THIN,
                    },
                )],
                temperature_k: Value::Fitted(400.0),
            }),
            [Value::Fitted(0.02), Value::Known(0.0), Value::Known(0.0)],
        ),
        Region {
            open_counts: rounded(&counts[2].0),
            sample_counts: rounded(&counts[2].1),
            open_live: None,
            sample_live: None,
            background: [Value::Known(0.0); 3],
            material: None,
        },
    ];
    let fit = |regions: &[Region]| {
        let m = Measurement {
            time_edges_us: setup.edges.clone(),
            charge_ratio: CHARGE_RATIO,
            normalization: Value::Fitted(1.0),
            regions: regions.to_vec(),
        };
        let fit = fit_counts(&m, &calibration(&setup)).expect("fit");
        assert!(fit.converged);
        assert!(
            fit.regions
                .iter()
                .all(|r| r.overdispersion == [Some(1.0); 2])
        );
        let truth = [
            THIN,
            1500.0,
            2.0 * THIN,
            TEMPERATURE_K,
            TERMS[0],
            0.05,
            0.5,
            0.08,
        ];
        let estimates = [
            fit.regions[0].densities[0],
            fit.regions[0].temperature_k.expect("a material"),
            fit.regions[1].densities[0],
            fit.regions[1].temperature_k.expect("a material"),
            fit.normalization,
            fit.regions[0].background[0],
            fit.regions[0].background[1],
            fit.regions[1].background[0],
        ];
        let pulls: Vec<f64> = estimates
            .iter()
            .zip(truth)
            .enumerate()
            .map(|(i, (x, t))| (x - t) / error_bar(&fit, i))
            .collect();
        assert!(pulls.iter().all(|p| p.abs() <= BOUND.sqrt()), "{pulls:?}");
        fit
    };
    let (with, without) = (fit(&regions), fit(&regions[..2]));
    let variance = |fit: &CountsFit| fit.covariance.as_ref().expect("covariance").get(4, 4);
    let [n_open, n_sample] = with.regions[2]
        .predicted
        .each_ref()
        .map(|c| c.iter().sum::<f64>());
    let [phi_open, phi_sample] = with.regions[2]
        .overdispersion
        .map(|phi| phi.expect("measured"));
    let a = with.normalization;
    let added = n_open * n_sample / (a * a * (phi_sample * n_open + phi_open * n_sample));
    let gained = 1.0 / variance(&with) - 1.0 / variance(&without);
    assert!(
        (gained - added).abs() <= 1e-2 * added,
        "{gained} vs {added}"
    );
}

#[test]
fn a_map_recovers_each_patch_and_its_covariance_between_patches_is_the_shared_quantities() {
    let setup = Setup {
        pulse: pulse(0.0, 200.0),
        ..standard()
    };
    let isotopes = [
        hafnium_like(20.0),
        synthetic_isotope(74, 182, 24.0, 0.01, 0.06),
    ];
    let truths = [
        ([THIN, THIN], 300.0, 0.05, 1.0e6),
        ([1.5 * THIN, 0.0], 450.0, 0.05, 1.0e6),
        ([SATURATED, THIN], 300.0, 0.0, 2.0e4),
        ([THIN, 2.0 * THIN], 2500.0, 0.05, 1.0e6),
    ];
    let unit: Vec<(Vec<f64>, Vec<f64>)> = truths
        .iter()
        .map(|&(densities, t, b0, level)| {
            let sample: Vec<(ResonanceData, f64)> =
                isotopes.iter().cloned().zip(densities).collect();
            let terms = [TERMS[0], b0, 0.0, 0.0];
            with_background(&setup, &beam(level), &sample, t, terms)
        })
        .chain([with_background(
            &setup,
            &beam(1.0e6),
            &[],
            TEMPERATURE_K,
            [TERMS[0], 0.0, 0.0, 0.0],
        )])
        .collect();
    let bins = setup.edges.len() - 1;
    let most = unit[4].0.iter().fold(0.0_f64, |m, &c| m.max(c));
    let live: Vec<f64> = unit[4]
        .0
        .iter()
        .map(|c| 1.0 / (1.0 + 0.25 * c / most))
        .collect();
    let (hot, broken) = ((0, 2), (1, 7));
    let cube = |run: usize| {
        Array3::from_shape_fn((bins, 3, 12), |(k, y, x)| {
            if y < 2 && x >= 10 {
                return 0.0;
            }
            let (open, sample) = &unit[if y == 2 { 4 } else { x / 2 }];
            let level = [1.0, 1.3, 0.8, 1.1][2 * (y % 2) + x % 2];
            let count = (live[k] * level * [open, sample][run][k]).round();
            match (y, x) {
                pixel if pixel == hot => 100.0 * count,
                pixel if pixel == broken => f64::NAN,
                _ => count,
            }
        })
    };
    let counts = [cube(0), cube(1)];
    let excluded = Array2::from_shape_fn((3, 12), |pixel| pixel == hot || pixel == broken);
    let behind = Array2::from_shape_fn((3, 12), |(y, x)| y < 2 && x != 9);
    let empty = Array2::from_shape_fn((3, 12), |(y, _)| y == 2);
    let map = MapMeasurement {
        time_edges_us: setup.edges.clone(),
        charge_ratio: CHARGE_RATIO,
        normalization: Value::Measured {
            value: TERMS[0],
            sd: 4.0e-5,
        },
        open_counts: counts[0].view(),
        sample_counts: counts[1].view(),
        open_live: Some(live.clone()),
        sample_live: Some(live.clone()),
        excluded: excluded.view(),
        sample: behind.view(),
        empty: empty.view(),
        binning: 2,
        material: Material {
            isotopes: isotopes
                .iter()
                .map(|data| (data.clone(), Value::Fitted(THIN)))
                .collect(),
            temperature_k: Value::Within {
                start: 400.0,
                lower: 100.0,
                upper: 2000.0,
            },
        },
        background: [
            Value::Within {
                start: 0.02,
                lower: 0.0,
                upper: f64::INFINITY,
            },
            Value::Known(0.0),
            Value::Known(0.0),
        ],
        empty_background: [Value::Known(0.0); 3],
    };
    let mut shared = calibration(&setup);
    shared.t0_us = Value::Within {
        start: T0_US,
        lower: T0_US,
        upper: T0_US + 1.0,
    };
    shared.flight_path_m = Value::Fitted(FLIGHT_PATH_M);
    let result = fit_map(&map, &shared).expect("map");
    assert!(result.converged);
    let kinds = [
        Patch::Sample,
        Patch::Sample,
        Patch::Sample,
        Patch::Sample,
        Patch::Mixed,
        Patch::Sample,
    ];
    assert_eq!(
        result.patches,
        Array2::from_shape_fn((2, 6), |(i, j)| if i == 0 {
            kinds[j]
        } else {
            Patch::Outside
        })
    );
    assert_eq!(
        result.trusted,
        Array2::from_shape_fn((2, 6), |(i, j)| i == 0 && j < 3)
    );
    assert!(
        result
            .failed
            .indexed_iter()
            .all(|(patch, reason)| reason.is_some() == (patch == (0, 5)))
    );
    assert!(
        result.failed[[0, 5]]
            .as_deref()
            .is_some_and(|reason| reason.contains("has no counts"))
    );
    assert!(result.densities[0][[0, 4]].is_nan());
    assert_eq!(result.shared, ["normalization", "t0", "flight path"]);

    let shared_covariance = result.shared_covariance.as_ref().expect("covariance");
    let shared_sd = |s: usize| shared_covariance.get(s, s).sqrt();
    assert_eq!(result.t0_us, T0_US);
    assert!(shared_sd(1).is_nan());
    let mut pulls = vec![(result.flight_path_m - FLIGHT_PATH_M) / shared_sd(2)];
    let fits: Vec<&RegionFit> = (0..4)
        .map(|j| result.fits[[0, j]].as_ref().expect("a fit"))
        .collect();
    for (j, &(densities, t, _, _)) in truths.iter().enumerate() {
        let patch = result.covariance[[0, j]].as_ref().expect("covariance");
        let sd = |q: usize| patch.get(q, q).sqrt().to_bits();
        for (m, n) in densities.into_iter().enumerate() {
            let (density, error) = (result.densities[m][[0, j]], result.density_sd[m][[0, j]]);
            assert_eq!(error.to_bits(), sd(m));
            if n == 0.0 {
                assert_eq!(density, 0.0);
            } else if j < 3 {
                pulls.push((density - n) / error);
            }
        }
        let error = result.temperature_sd_k[[0, j]];
        assert_eq!(error.to_bits(), sd(2));
        if j < 3 {
            pulls.push((result.temperature_k[[0, j]] - t) / error);
        }
    }
    assert!(pulls.iter().all(|p| p.abs() <= BOUND.sqrt()), "{pulls:?}");
    assert_eq!(result.temperature_k[[0, 3]], 2000.0);

    let term = |y: f64, mu: f64| {
        if y == 0.0 {
            mu
        } else {
            y * (y / mu).ln() + mu - y
        }
    };
    let empty_fit = result.empty.as_ref().expect("empty pixels");
    let phi = |fit: &RegionFit, run: usize| fit.overdispersion[run].expect("measured");
    for (run, map) in result.overdispersion.iter().enumerate() {
        assert!((0..4).all(|j| map[[0, j]] == phi(fits[j], run)));
    }
    let (mut deviance, mut zeros) = (0.0, 0);
    for (run, residuals) in result.residuals.iter().enumerate() {
        for (j, fit) in fits.iter().enumerate() {
            for (k, &mu) in fit.predicted[run].iter().enumerate() {
                let y: f64 = (0..2)
                    .flat_map(|y| (2 * j..2 * j + 2).map(move |x| (y, x)))
                    .filter(|&pixel| !excluded[pixel])
                    .map(|(y, x)| counts[run][[k, y, x]])
                    .sum();
                let d = residuals[[k, 0, j]];
                let expected = 2.0 * term(y, mu);
                assert!(
                    (d * d - expected).abs() <= 1e-12 * (y + mu) && d * (y - mu) >= 0.0,
                    "{run} {j} {k}: {d} for {y} counts, {mu} predicted"
                );
                deviance += d * d / (2.0 * phi(fit, run));
                zeros += usize::from(y == 0.0);
            }
        }
        let empty_deviance: f64 = empty_fit.predicted[run]
            .iter()
            .enumerate()
            .map(|(k, &mu)| term(counts[run].slice(s![k, 2, ..]).sum(), mu))
            .sum();
        deviance += empty_deviance / phi(empty_fit, run);
    }
    assert!(zeros > 0);
    assert!(
        (deviance - result.deviance).abs() <= 1e-9 * result.deviance,
        "{deviance} vs {}",
        result.deviance
    );

    let summed = |run: usize, pixels: &dyn Fn(usize, usize) -> bool| -> Vec<f64> {
        (0..bins)
            .map(|k| {
                (0..3)
                    .flat_map(|y| (0..12).map(move |x| (y, x)))
                    .filter(|&(y, x)| !excluded[[y, x]] && pixels(y, x))
                    .map(|(y, x)| counts[run][[k, y, x]])
                    .sum()
            })
            .collect()
    };
    let region =
        |pixels: &dyn Fn(usize, usize) -> bool, material: Option<Material>, background| Region {
            open_counts: summed(0, pixels),
            sample_counts: summed(1, pixels),
            open_live: map.open_live.clone(),
            sample_live: map.sample_live.clone(),
            background,
            material,
        };
    let joint = fit_counts(
        &Measurement {
            time_edges_us: setup.edges.clone(),
            charge_ratio: CHARGE_RATIO,
            normalization: map.normalization,
            regions: (0..4)
                .map(|j| {
                    region(
                        &move |y, x| y < 2 && x / 2 == j,
                        Some(map.material.clone()),
                        map.background,
                    )
                })
                .chain([region(&|y, _| y == 2, None, map.empty_background)])
                .collect(),
        },
        &shared,
    )
    .expect("joint fit");
    assert!(joint.converged);
    let joint_covariance = joint.covariance.as_ref().expect("covariance");
    let joint_sd = |i: usize| joint_covariance.get(i, i).sqrt();
    let (normalization, flight_path) = (12, 18);
    let [Some(map_pulls), Some(joint_pulls)] = [&result.measured_pulls, &joint.measured_pulls]
    else {
        panic!("measured pulls");
    };
    assert!(
        (map_pulls[0] - joint_pulls[0]).abs() <= 1e-3,
        "{map_pulls:?} vs {joint_pulls:?}"
    );
    let mut gaps = vec![
        (result.normalization - joint.normalization) / joint_sd(normalization),
        (result.flight_path_m - joint.flight_path_m) / joint_sd(flight_path),
    ];
    let mut ratios = vec![
        shared_sd(0) / joint_sd(normalization),
        shared_sd(2) / joint_sd(flight_path),
    ];
    for (j, joint_region) in joint.regions[..4].iter().enumerate() {
        for m in 0..2 {
            let error = joint_sd(3 * j + m);
            if error.is_finite() {
                gaps.push((result.densities[m][[0, j]] - joint_region.densities[m]) / error);
                ratios.push(result.density_sd[m][[0, j]] / error);
            }
        }
        let error = joint_sd(3 * j + 2);
        if error.is_finite() {
            gaps.push(
                (result.temperature_k[[0, j]] - joint_region.temperature_k.expect("T")) / error,
            );
            ratios.push(result.temperature_sd_k[[0, j]] / error);
        }
    }
    assert!(gaps.iter().all(|g| g.abs() <= 0.01), "{gaps:?}");
    assert!(ratios.iter().all(|r| (r - 1.0).abs() <= 1e-3), "{ratios:?}");
    let cross = result
        .cross_covariance((0, 0), (0, 1))
        .expect("cross covariance");
    for (a, b) in [(0, 0), (1, 0), (2, 2), (0, 2)] {
        let joint = joint_covariance.get(a, 3 + b);
        assert!(
            (cross.get(a, b) - joint).abs() <= 1e-4 * joint.abs(),
            "{a} {b}: {} vs {joint}",
            cross.get(a, b)
        );
    }

    let area = |[open_counts, sample_counts]: [Vec<f64>; 2]| {
        let measurement = Measurement {
            time_edges_us: setup.edges.clone(),
            charge_ratio: CHARGE_RATIO,
            normalization: Value::Known(result.normalization),
            regions: vec![Region {
                open_counts,
                sample_counts,
                open_live: Some(live.clone()),
                sample_live: Some(live.clone()),
                background: map.background,
                material: Some(map.material.clone()),
            }],
        };
        let fixed = Calibration {
            t0_us: Value::Known(result.t0_us),
            flight_path_m: Value::Known(result.flight_path_m),
            ..calibration(&setup)
        };
        let fit = fit_counts(&measurement, &fixed).expect("area fit");
        assert!(fit.converged);
        fit
    };
    let measured = [0, 1].map(|run| {
        (0..bins)
            .map(|k| {
                (0..2)
                    .flat_map(|y| (0..4).map(move |x| (y, x)))
                    .filter(|&pixel| !excluded[pixel])
                    .map(|(y, x)| counts[run][[k, y, x]])
                    .sum()
            })
            .collect::<Vec<f64>>()
    });
    let predicted = [0, 1].map(|run| {
        (0..bins)
            .map(|k| (fits[0].predicted[run][k] + fits[1].predicted[run][k]).round())
            .collect::<Vec<f64>>()
    });
    let (one, two) = (area(measured), area(predicted));
    let weights = [0, 1].map(|j| fits[j].predicted[0].iter().sum::<f64>());
    for m in 0..2 {
        let (density, sd) = (one.regions[0].densities[m], error_bar(&one, m));
        let mean = (weights[0] * result.densities[m][[0, 0]]
            + weights[1] * result.densities[m][[0, 1]])
            / (weights[0] + weights[1]);
        assert!((density - two.regions[0].densities[m]).abs() <= BOUND.sqrt() * sd);
        assert!(
            (density - mean).abs() >= 5.0 * sd,
            "{density} vs {mean} ± {sd}"
        );
    }
    let sd = error_bar(&one, 2);
    let mean = (weights[0] * result.temperature_k[[0, 0]]
        + weights[1] * result.temperature_k[[0, 1]])
        / (weights[0] + weights[1]);
    assert!((one.temperature() - two.temperature()).abs() <= BOUND.sqrt() * sd);
    assert!((one.temperature() - mean).abs() >= 5.0 * sd);
}

#[test]
fn maps_the_fit_does_not_describe_are_refused() {
    let setup = standard();
    let edges: Vec<f64> = (0..=577).map(|k| 350.0 + 0.2 * f64::from(k)).collect();
    let counts = Array3::zeros((577, 1, 7));
    let mut broken = counts.clone();
    broken[[3, 0, 1]] = f64::NAN;
    let none = Array2::from_elem((1, 7), false);
    let all = Array2::from_elem((1, 7), true);
    let six = Array2::from_shape_fn((1, 7), |(_, x)| x < 6);
    let base = MapMeasurement {
        time_edges_us: edges,
        charge_ratio: CHARGE_RATIO,
        normalization: Value::Fitted(1.0),
        open_counts: counts.view(),
        sample_counts: counts.view(),
        open_live: None,
        sample_live: None,
        excluded: none.view(),
        sample: all.view(),
        empty: none.view(),
        binning: 1,
        material: Material {
            isotopes: vec![(hafnium_like(20.0), Value::Fitted(THIN))],
            temperature_k: Value::Known(TEMPERATURE_K),
        },
        background: [Value::Known(0.0); 3],
        empty_background: [Value::Known(0.0); 3],
    };
    let refused =
        |map: &MapMeasurement<'_>, expected: &str| match fit_map(map, &calibration(&setup)) {
            Err(
                PipelineError::InvalidParameter(message) | PipelineError::ShapeMismatch(message),
            ) => {
                assert!(message.contains(expected), "{message}")
            }
            other => panic!("{other:?}"),
        };
    refused(&base, "patch (0, 0): the open-beam run has no counts");
    let mut map = base.clone();
    map.sample = six.view();
    refused(&map, "patch (0, 0): the open-beam run has no counts");
    map.sample_counts = broken.view();
    refused(&map, "got NaN in bin 3 of pixel (0, 1)");
    map.sample = none.view();
    refused(&map, "no patch");
    map.sample = six.view();
    map.empty = all.view();
    refused(&map, "both behind the sample and empty");
    map.binning = 0;
    refused(&map, "at least one pixel");
    map.empty = all.slice(s![.., 1..]);
    refused(&map, "the empty mask has shape");
    map.sample_counts = counts.slice(s![1.., .., ..]);
    refused(&map, "sample counts of shape");
    let no_bins = Array3::zeros((0, 1, 7));
    let mut map = base.clone();
    map.time_edges_us.clear();
    map.open_counts = no_bins.view();
    map.sample_counts = no_bins.view();
    assert!(matches!(
        fit_map(&map, &calibration(&setup)),
        Err(PipelineError::FlightTimeGrid(_))
    ));
    let mut measured_empty = base.clone();
    measured_empty.empty_background[0] = Value::Measured {
        value: 0.0,
        sd: 0.1,
    };
    refused(&measured_empty, "empty pixels is not supported");
    let mut short_live = base.clone();
    short_live.sample_live = Some(vec![1.0; 2]);
    assert!(matches!(
        fit_map(&short_live, &calibration(&setup)),
        Err(PipelineError::ShapeMismatch(message)) if message.starts_with("2 sample live")
    ));
    let mut measured = base.clone();
    measured.background[0] = Value::Measured {
        value: 0.0,
        sd: 0.1,
    };
    refused(&measured, "once per patch");
}

fn chain(setup: &Setup) -> Vec<FlightTimeGrid> {
    std::iter::successors(
        Some(
            FlightTimeGrid::new(
                &setup.edges,
                T0_US,
                FLIGHT_PATH_M,
                &setup.pulse.detector_pulse(),
            )
            .expect("grid"),
        ),
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
    let fitted = [(isotope, fit.regions[0].densities[0])];
    let beam = |u: f64| fit.regions[0].beam.per_us(u);
    let grids = chain(&setup);
    let accepted = first_accepted(&grids, |grid| predicted(grid, &beam, &fitted));
    assert_eq!(fit.halvings, accepted);
    assert_eq!(fit.points, grids[accepted].flight_times_us().len());
    assert_eq!(fit.step_us, grids[accepted].step_us());
    let rebuilt = predicted(&grids[accepted], &beam, &fitted);
    for (run, rebuilt) in fit.regions[0]
        .predicted
        .iter()
        .zip(rebuilt.chunks(counts.0.len()))
    {
        assert_eq!(run.len(), rebuilt.len());
        assert!(
            run.iter()
                .zip(rebuilt)
                .all(|(f, r)| (f - r).abs() <= 1e-12 * r)
        );
    }
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
    let first = FlightTimeGrid::new(
        &setup.edges,
        T0_US,
        FLIGHT_PATH_M,
        &setup.pulse.detector_pulse(),
    )
    .expect("grid");
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
    counts_per_neutron: [f64; 2],
) -> (Vec<f64>, Vec<f64>) {
    (
        draw(&expected.0, seed, counts_per_neutron[0]),
        draw(&expected.1, seed + 1_000_000, counts_per_neutron[1]),
    )
}

#[test]
fn each_run_s_overdispersion_is_its_variance_over_its_poisson_variance() {
    let setup = standard();
    let isotope = hafnium_like(20.0);
    let sample = [(isotope, THIN)];
    let expected = [1.0e6, 2.0e6].map(|level| expected(&setup, &beam(level), &sample));
    let counts_per_neutron = [[3.0, 7.0], [5.0, 2.0]];
    let live = recorded(&setup, expected[1].clone(), &sample)
        .regions
        .remove(0);
    let (open_live, sample_live) = (
        live.open_live.expect("live"),
        live.sample_live.expect("live"),
    );
    let times = |mu: &[f64], live: &[f64]| mu.iter().zip(live).map(|(c, l)| c * l).collect();
    let thinned = (
        times(&expected[1].0, &open_live),
        times(&expected[1].1, &sample_live),
    );
    let fits: Vec<CountsFit> = (100..110)
        .map(|seed| {
            let mut m = measurement(
                &setup,
                draws(&expected[0], seed, counts_per_neutron[0]),
                &sample,
            );
            let mut second = measurement(
                &setup,
                draws(&thinned, seed + 500, counts_per_neutron[1]),
                &sample,
            );
            second.regions[0].open_live = Some(open_live.clone());
            second.regions[0].sample_live = Some(sample_live.clone());
            m.regions.extend(second.regions);
            let fit = fit_counts(&m, &calibration(&setup)).expect("fit");
            let open = fit_open_beam(
                &setup.edges,
                &m.regions[1].open_counts,
                &calibration(&setup),
                Some(&open_live),
            )
            .expect("open-beam fit");
            assert_eq!(fit.regions[1].overdispersion[0], open.overdispersion);
            fit
        })
        .collect();
    for (r, per_run) in counts_per_neutron.iter().enumerate() {
        let freedom =
            (expected[r].0.len() - fits[0].regions[r].beam.coefficients().len() - 1) as f64;
        let bound = 3.0 * (2.0 / freedom).sqrt() / (fits.len() as f64).sqrt();
        for (run, per_neutron) in per_run.iter().enumerate() {
            let mut ratios = Vec::new();
            for fit in &fits {
                assert!(fit.converged);
                ratios.push(fit.regions[r].overdispersion[run].expect("measured") / per_neutron);
            }
            let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
            assert!(
                (mean - 1.0).abs() <= bound,
                "region {r} run {run}: {mean} vs 1 ± {bound}"
            );
        }
    }
    for fit in &fits {
        let pulls = [0, 1].map(|i| (fit.regions[i].densities[0] - THIN) / error_bar(fit, i));
        assert!(pulls.iter().all(|p| p.abs() <= 4.0), "{pulls:?}");
    }
}

#[test]
fn black_bins_refuse_counts_hold_a_bounded_background_at_zero_and_rare_ones_leave_the_noise() {
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
    let mut m = measurement(&setup, (open.clone(), counts.clone()), &sample);
    counts[dark] = 1.0;
    m.regions
        .extend(measurement(&setup, (open, counts), &sample).regions);
    match fit_counts(&m, &calibration(&setup)) {
        Err(PipelineError::UnmodelledCounts {
            region,
            run,
            bin,
            counts,
            predicted,
        }) => {
            assert_eq!((region, run, bin, counts), (1, "sample", dark, 1.0));
            assert!(predicted < NEGLIGIBLE_PREDICTION, "{predicted}");
        }
        other => panic!("{other:?}"),
    }

    let counts = draws(&expected, 401, [1.0; 2]);
    let mut m = fitted_from(measurement(&setup, counts, &sample), TEMPERATURE_K);
    m.regions[0].background[0] = Value::Within {
        start: 1e-3,
        lower: 0.0,
        upper: f64::INFINITY,
    };
    let fit = fit_counts(&m, &calibration(&setup)).expect("fit");
    assert!(fit.converged);
    assert_eq!(fit.regions[0].background[0], 0.0);
    assert_eq!(fit.on_bound, [false, false, true]);
    assert!(error_bar(&fit, 0).is_finite() && error_bar(&fit, 1).is_finite());

    let counts_per_neutron = 7.0;
    let (open, mut counts) = draws(&expected, 400, [counts_per_neutron; 2]);
    counts[rare] = 1.0;
    let fit = fit_counts(
        &measurement(&setup, (open, counts), &sample),
        &calibration(&setup),
    )
    .expect("fit");
    let parameters = fit.regions[0].beam.coefficients().len() + 1;
    for (run, expected) in [&expected.0, &expected.1].into_iter().enumerate() {
        let ratio = fit.regions[0].overdispersion[run].expect("measured") / counts_per_neutron;
        let freedom = (expected.iter().filter(|&&mu| mu >= 1.0).count() - parameters) as f64;
        assert!(
            (ratio - 1.0).abs() <= 3.0 * (2.0 / freedom).sqrt(),
            "{run}: {ratio}"
        );
    }
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
                    draws(&expected, seed, [1.0; 2]),
                    &[(isotope.clone(), THIN)],
                );
                m.material_mut().temperature_k = temperature_k;
                fit_counts(&m, &calibration(&setup)).expect("fit")
            })
            .collect();
        assert!(fits.iter().all(|fit| fit.regions[0].densities[0] >= 0.0));
        let absent: Vec<&CountsFit> = fits
            .iter()
            .filter(|fit| fit.regions[0].densities[0] == 0.0)
            .collect();
        assert!(!absent.is_empty(), "{temperature_k:?}");
        for fit in absent {
            let covariance = fit.covariance.as_ref().expect("covariance");
            assert!(
                covariance.data.iter().all(|v| v.is_nan()),
                "{temperature_k:?}"
            );
        }
    }
    let measured: Vec<CountsFit> = (500..503)
        .map(|seed| {
            let mut m = measurement(
                &setup,
                draws(&expected, seed, [1.0; 2]),
                &[(isotope.clone(), THIN)],
            );
            m.material_mut().isotopes[0].1 = Value::Measured {
                value: 0.0,
                sd: THIN,
            };
            fit_counts(&m, &calibration(&setup)).expect("fit")
        })
        .filter(|fit| fit.regions[0].densities[0] == 0.0)
        .collect();
    assert!(!measured.is_empty());
    for fit in measured {
        let pulls = fit.measured_pulls.expect("measured pulls");
        assert!(pulls.iter().all(|pull| pull.is_finite()), "{pulls:?}");
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
fn the_covariance_is_the_inverse_of_the_information_in_the_counts_and_measurements() {
    covariance_against_information(Value::Known(TEMPERATURE_K), false);
    covariance_against_information(
        Value::Measured {
            value: TEMPERATURE_K,
            sd: 0.02 * TEMPERATURE_K,
        },
        true,
    );
}

fn covariance_against_information(temperature: Value, noisy: bool) {
    let setup = standard();
    let truth = [
        (hafnium_like(20.0), 3.0 / 7805.1),
        (synthetic_isotope(74, 182, 20.3, 0.01, 0.06), 1.5 / 7805.1),
    ];
    let counts = with_background(&setup, &beam(1.0e6), &truth, TEMPERATURE_K, TERMS);
    let mut m = recorded(&setup, counts, &truth);
    m.material_mut().temperature_k = temperature;
    m.normalization = Value::Fitted(TERMS[0]);
    m.regions[0].background[0] = Value::Fitted(TERMS[1]);
    m.regions[0].background[1] = Value::Fitted(TERMS[2]);
    m.regions[0].background[2] = Value::Known(TERMS[3]);
    let fitted_temperature = !matches!(temperature, Value::Known(_));
    let first_term = truth.len() + usize::from(fitted_temperature);
    let measured: Vec<(usize, f64, f64)> = if noisy {
        m.regions[0].sample_counts = draw(&m.regions[0].sample_counts, 500, 7.0);
        let (density_sd, a_sd, b0_sd) = (0.05 * truth[0].1, 0.01 * TERMS[0], 0.2 * TERMS[1]);
        m.material_mut().isotopes[0].1 = Value::Measured {
            value: truth[0].1,
            sd: density_sd,
        };
        m.normalization = Value::Measured {
            value: TERMS[0],
            sd: a_sd,
        };
        m.regions[0].background[0] = Value::Measured {
            value: TERMS[1],
            sd: b0_sd,
        };
        let temperature = match temperature {
            Value::Measured { value, sd } => Some((truth.len(), value, sd)),
            _ => None,
        };
        [(0, truth[0].1, density_sd)]
            .into_iter()
            .chain(temperature)
            .chain([
                (first_term, TERMS[0], a_sd),
                (first_term + 1, TERMS[1], b0_sd),
            ])
            .collect()
    } else {
        Vec::new()
    };
    let fit = fit_counts(&m, &calibration(&setup)).expect("fit");
    let overdispersion = fit.regions[0]
        .overdispersion
        .map(|phi| phi.expect("measured"));
    let deviance: f64 = [&m.regions[0].open_counts, &m.regions[0].sample_counts]
        .iter()
        .zip(&fit.regions[0].predicted)
        .zip(overdispersion)
        .map(|((y, mu), phi)| {
            let half: f64 = y
                .iter()
                .zip(mu)
                .map(|(&y, &mu)| {
                    let d = (y - mu) / mu;
                    mu * ((1.0 + d) * d.ln_1p() - d)
                })
                .sum();
            half / phi
        })
        .sum();
    assert!(
        (deviance / fit.deviance - 1.0).abs() <= 1e-9,
        "{deviance} vs {}",
        fit.deviance
    );
    let quantities = first_term + 3;
    let (low, high) = FlightTimeGrid::new(
        &setup.edges,
        T0_US,
        FLIGHT_PATH_M,
        &setup.pulse.detector_pulse(),
    )
    .expect("grid")
    .range_us();
    let beam_times = |index: Option<usize>, step: f64| {
        let beam = fit.regions[0].beam.clone();
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
    let mut terms = [fit.normalization; 4];
    terms[1..].copy_from_slice(&fit.regions[0].background);
    let estimates: Vec<f64> = fit.regions[0]
        .densities
        .iter()
        .chain(fitted_temperature.then_some(&fit.temperature()))
        .chain(&terms[..3])
        .copied()
        .collect();
    let truths = truth
        .iter()
        .map(|(_, n)| n)
        .chain(fitted_temperature.then_some(&TEMPERATURE_K))
        .chain(&TERMS[..3]);
    for (i, (x, truth)) in estimates.iter().zip(truths).enumerate().filter(|_| !noisy) {
        let pull = (x - truth) / error_bar(&fit, i);
        assert!(pull.abs() <= BOUND.sqrt(), "{i}: {pull}");
    }
    let fitted: Vec<(ResonanceData, f64)> = truth
        .iter()
        .zip(&fit.regions[0].densities)
        .map(|((data, _), &n)| (data.clone(), n))
        .collect();
    let live: Vec<f64> = m.regions[0]
        .open_live
        .iter()
        .chain(&m.regions[0].sample_live)
        .flatten()
        .copied()
        .collect();
    let joined = |(open, sample): (Vec<f64>, Vec<f64>)| -> Vec<f64> {
        let counts = open.into_iter().chain(sample);
        counts.zip(&live).map(|(c, l)| l * c).collect()
    };
    let temperature_k = fit.temperature();
    let beam = beam_times(None, 0.0);
    let mu = joined(with_background(
        &setup,
        &beam,
        &fitted,
        temperature_k,
        terms,
    ));
    let coefficients = fit.regions[0].beam.coefficients().len();
    let columns: Vec<Vec<f64>> = (0..coefficients + quantities)
        .map(|p| {
            let shifted = |sign: f64| {
                if p < coefficients {
                    let h = 1e-4;
                    let beam = beam_times(Some(p), sign * h);
                    let counts = with_background(&setup, &beam, &fitted, temperature_k, terms);
                    return (joined(counts), h);
                }
                let (mut sample, mut kelvin, mut shifted) = (fitted.clone(), temperature_k, terms);
                let q = p - coefficients;
                let value = if q < fitted.len() {
                    &mut sample[q].1
                } else if q < first_term {
                    &mut kelvin
                } else {
                    &mut shifted[q - first_term]
                };
                let h = 1e-4 * *value;
                *value += sign * h;
                let counts = with_background(&setup, &beam, &sample, kelvin, shifted);
                (joined(counts), h)
            };
            let ((up, h), (down, _)) = (shifted(1.0), shifted(-1.0));
            up.iter()
                .zip(&down)
                .map(|(a, b)| (a - b) / (2.0 * h))
                .collect()
        })
        .collect();
    let bins = mu.len() / 2;
    let mut information: Vec<Vec<f64>> = columns
        .iter()
        .map(|a| {
            columns
                .iter()
                .map(|b| {
                    a.iter()
                        .zip(b)
                        .zip(&mu)
                        .enumerate()
                        .filter(|(_, (_, m))| **m > 0.0)
                        .map(|(k, ((x, y), m))| x * y / (m * overdispersion[k / bins]))
                        .sum()
                })
                .collect()
        })
        .collect();
    let counts_only = inverse(information.clone());
    for &(q, _, sd) in &measured {
        let p = coefficients + q;
        let ratio = counts_only[p][p] / (sd * sd);
        assert!(
            (0.1..=10.0).contains(&ratio),
            "{q}: the counts' variance is {ratio} times the measurement's"
        );
        information[p][p] += sd.powi(-2);
    }
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
    let pulls = fit.measured_pulls.expect("measured pulls");
    assert_eq!(pulls.len(), measured.len());
    for (&(q, value, sd), pull) in measured.iter().zip(pulls) {
        let variance = oracle[coefficients + q][coefficients + q];
        let expected = (estimates[q] - value) / (sd * sd - variance).sqrt();
        assert!(
            (pull - expected).abs() <= 1e-2 * (1.0 + expected.abs()),
            "{q}: {pull} vs {expected}"
        );
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
            m.regions[0].sample_counts.pop();
        }),
        PipelineError::ShapeMismatch(_)
    ));
    assert!(matches!(
        refused(&|m| m.regions[0].sample_live = Some(vec![1.0; 3])),
        PipelineError::ShapeMismatch(_)
    ));
    invalid(&|m| m.regions[0].sample_live = Some(vec![0.0; m.regions[0].sample_counts.len()]));
    invalid(&|m| m.regions[0].open_live = Some(vec![f64::NAN; m.regions[0].open_counts.len()]));
    invalid(&|m| m.regions[0].sample_counts[5] += 0.5);
    invalid(&|m| m.regions[0].sample_counts[5] = -1.0);
    invalid(&|m| m.regions[0].sample_counts.iter_mut().for_each(|c| *c = 0.0));
    invalid(&|m| m.charge_ratio = 0.0);
    invalid(&|m| m.charge_ratio = f64::NAN);
    invalid(&|m| m.normalization = Value::Fitted(0.0));
    invalid(&|m| m.normalization = Value::Known(0.0));
    invalid(&|m| m.normalization = Value::Known(f64::INFINITY));
    invalid(&|m| m.regions[0].background[1] = Value::Fitted(f64::NAN));
    for sd in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        invalid(&|m| m.material_mut().temperature_k = Value::Measured { value: 300.0, sd });
    }
    let pulse = |change: &dyn Fn(&mut Pulse)| {
        let mut pulse = calibration(&setup).pulse;
        change(&mut pulse);
        pulse
    };
    for (t0_us, flight_path_m, pulse) in [
        (
            Value::Known(f64::NAN),
            Value::Known(FLIGHT_PATH_M),
            pulse(&|_| {}),
        ),
        (Value::Known(T0_US), Value::Known(0.0), pulse(&|_| {})),
        (
            Value::Known(T0_US),
            Value::Fitted(-FLIGHT_PATH_M),
            pulse(&|_| {}),
        ),
        (
            Value::Measured {
                value: T0_US,
                sd: 0.0,
            },
            Value::Known(FLIGHT_PATH_M),
            pulse(&|_| {}),
        ),
        (
            Value::Known(T0_US),
            Value::Known(FLIGHT_PATH_M),
            pulse(&|p| p.alpha[0] = Value::Known(-0.05)),
        ),
        (
            Value::Known(T0_US),
            Value::Known(FLIGHT_PATH_M),
            pulse(&|p| p.r = Value::Fitted(1.5)),
        ),
        (
            Value::Known(T0_US),
            Value::Known(FLIGHT_PATH_M),
            pulse(&|p| p.fwhm_squared_us2 = Value::Known(f64::NAN)),
        ),
        (
            Value::Known(T0_US),
            Value::Known(FLIGHT_PATH_M),
            pulse(&|p| {
                p.beta = [Value::Known(0.0); 2];
                p.r = Value::Known(0.0);
            }),
        ),
        (
            Value::Known(T0_US),
            Value::Known(FLIGHT_PATH_M),
            pulse(&|p| p.energy_span_ev = (0.0, 200.0)),
        ),
    ] {
        let calibration = Calibration {
            t0_us,
            flight_path_m,
            pulse,
        };
        assert!(matches!(
            fit_counts(&good, &calibration),
            Err(PipelineError::InvalidParameter(_))
        ));
        assert!(matches!(
            fit_open_beam(
                &good.time_edges_us,
                &good.regions[0].open_counts,
                &calibration,
                None
            ),
            Err(PipelineError::InvalidParameter(_))
        ));
    }
    let black = [(isotope.clone(), 4.0e3 / 7805.1)];
    let counts = expected(&setup, &beam(1.0e4), &black);
    let mut empty = measurement(&setup, (rounded(&counts.0), rounded(&counts.1)), &black);
    empty.regions[0].background[0] = Value::Fitted(-1e-5);
    match fit_counts(&empty, &calibration(&setup)) {
        Err(PipelineError::UnmodelledCounts {
            run: "sample",
            counts: 0.0,
            predicted,
            ..
        }) => assert!(predicted < 0.0, "{predicted}"),
        other => panic!("{other:?}"),
    }
    let mut huge = good.clone();
    huge.normalization = Value::Known(1e308);
    match fit_counts(&huge, &calibration(&setup)) {
        Err(PipelineError::UnmodelledCounts {
            run: "sample",
            predicted,
            ..
        }) => assert!(!predicted.is_finite(), "{predicted}"),
        other => panic!("{other:?}"),
    }
    invalid(&|m| m.material_mut().isotopes.clear());
    invalid(&|m| m.regions.clear());
    invalid(&|m| m.regions[0].material = None);
    invalid(&|m| {
        m.material_mut()
            .isotopes
            .push((isotope.clone(), Value::Fitted(THIN)))
    });
    invalid(&|m| m.material_mut().isotopes[0].1 = Value::Known(-1.0));
    invalid(&|m| m.material_mut().isotopes[0].1 = Value::Fitted(f64::NAN));
    for (lower, upper) in [(2.0 * THIN, 3.0 * THIN), (THIN, THIN), (-THIN, 2.0 * THIN)] {
        invalid(&|m| {
            m.material_mut().isotopes[0].1 = Value::Within {
                start: THIN,
                lower,
                upper,
            }
        });
    }
    for temperature_k in [0.5, 6000.0, f64::NAN] {
        invalid(&|m| m.material_mut().temperature_k = Value::Known(temperature_k));
        invalid(&|m| m.material_mut().temperature_k = Value::Fitted(temperature_k));
    }
    invalid(&|m| {
        m.material_mut().isotopes[0].0.ranges[0].l_groups[0].resonances[0].energy = f64::NAN
    });
    invalid(&|m| m.material_mut().isotopes[0].0.ranges[0].l_groups[0].resonances[0].gg = f64::NAN);
    invalid(&|m| m.material_mut().isotopes[0].0.ranges[0].target_spin = f64::NAN);
    invalid(&|m| m.material_mut().isotopes[0].0.ranges[0].energy_high = 20.0);

    let top_ev = FlightTimeGrid::new(
        &setup.edges,
        T0_US,
        FLIGHT_PATH_M,
        &setup.pulse.detector_pulse(),
    )
    .expect("grid")
    .energies_ev()[0];
    let reach = |temperature_k: f64| {
        let u = DopplerParams::new(temperature_k, isotope.awr)
            .expect("doppler")
            .u();
        (top_ev.sqrt() + SUPPORT_X * u).powi(2)
    };
    let mut between = good.clone();
    between.material_mut().isotopes[0].0.ranges[0].energy_high =
        0.5 * (reach(TEMPERATURE_K) + reach(5000.0));
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
        t0_us: T0_US,
        flight_path_m: FLIGHT_PATH_M,
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
                assert_eq!(fit.regions[0].overdispersion, [Some(1.0); 2], "{case}");
                let pulls = [
                    (fit.regions[0].densities[0] - density) / error_bar(&fit, 0),
                    (fit.temperature() - truth_k) / error_bar(&fit, 1),
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
    let first = FlightTimeGrid::new(
        &setup.edges,
        T0_US,
        FLIGHT_PATH_M,
        &setup.pulse.detector_pulse(),
    )
    .expect("grid");
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
    let unbounded = fit.unbounded.as_ref().expect("unbounded");
    assert!(
        covariance
            .data
            .iter()
            .chain(&unbounded.covariance.data)
            .chain(&unbounded.mean)
            .all(|v| v.is_nan()),
        "{truth_k} K"
    );
    fit
}

#[test]
fn only_a_temperature_on_the_box_edge_withholds_the_whole_covariance() {
    let setup = standard();
    let (cool, hot) = (
        [(hafnium_like(20.0), THIN)],
        [(two_resonances(), SATURATED)],
    );
    let counts = expected(&setup, &beam(1.0e6), &cool);
    let mut m = measurement(&setup, (rounded(&counts.0), rounded(&counts.1)), &cool);
    let counts = expected_at(&setup, &beam(1.0e6), &hot, 6000.0);
    let second = measurement(&setup, (rounded(&counts.0), rounded(&counts.1)), &hot);
    m.regions.extend(fitted_from(second, TEMPERATURE_K).regions);
    let both = fit_counts(&m, &calibration(&setup)).expect("fit");
    assert!(both.converged);
    assert_eq!(both.regions[1].temperature_k, Some(5000.0));
    let covariance = both.covariance.as_ref().expect("covariance");
    assert!(covariance.data.iter().all(|v| v.is_nan()));

    let sample = [(hafnium_like(20.0), THIN)];
    let counts = expected(&setup, &beam(1.0e6), &sample);
    let mut m = measurement(&setup, (rounded(&counts.0), rounded(&counts.1)), &sample);
    m.material_mut().temperature_k = Value::Within {
        start: 150.0,
        lower: 100.0,
        upper: 200.0,
    };
    let bounded = fit_counts(&m, &calibration(&setup)).expect("fit");
    assert!(bounded.converged);
    assert_eq!(bounded.temperature(), 200.0);
    assert_eq!(bounded.on_bound, [false, true]);
    assert!(error_bar(&bounded, 0).is_finite());

    let setup = Setup {
        simulator_step_us: 1.0 / 1024.0,
        ..kev_window()
    };
    let beam = kev_beam(1.0e7);
    let isotope = hafnium_like(kev_resonance_ev(&setup, 0.04));
    let cold = edge_fit(&setup, &beam, &[(isotope.clone(), 0.17767)], 0.0, 5000.0);
    assert_eq!(cold.temperature(), 1.0);
    let fitted = [(isotope, cold.regions[0].densities[0])];
    let (low, high) = chain(&setup)[0].range_us();
    let fitted_beam = |u: f64| {
        if (low..=high).contains(&u) {
            cold.regions[0].beam.per_us(u)
        } else {
            0.0
        }
    };
    let reference = expected_at(&setup, &fitted_beam, &fitted, cold.temperature());
    let reference: Vec<f64> = reference.0.into_iter().chain(reference.1).collect();
    let on_grid = predicted_at(
        &chain(&setup)[cold.halvings],
        &|u| cold.regions[0].beam.per_us(u),
        &fitted,
        cold.temperature(),
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
    let pull = (fit.temperature() - TEMPERATURE_K) / error_bar(&fit, 1);
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
    assert!(fit.temperature() < 500.0, "{}", fit.temperature());
    let overdispersion = fit.regions[0].overdispersion[1].expect("measured");
    assert!(overdispersion > 100.0, "{overdispersion}");
}

mod error_bar_pulls {
    use rayon::prelude::*;

    use super::*;

    pub(super) const Z: f64 = 3.5;
    const MOST_FAILED: usize = 2;

    struct Ensemble {
        setup: Setup,
        sample: (ResonanceData, f64),
        temperature_k: Value,
        terms: Option<[f64; 4]>,
        level: f64,
        counts_per_neutron: [f64; 2],
        open_run_fraction: f64,
        measured_density_sd: Option<f64>,
        seeds: std::ops::Range<u64>,
    }

    struct Draw {
        pulls: Vec<f64>,
        estimates: Vec<f64>,
        reported_correlation: Option<f64>,
        overdispersion: [f64; 2],
        deviance: f64,
    }

    impl Ensemble {
        fn expected(&self) -> (Vec<f64>, Vec<f64>) {
            let (open, sample) = with_background(
                &self.setup,
                &beam(self.level),
                std::slice::from_ref(&self.sample),
                TEMPERATURE_K,
                self.terms.unwrap_or([1.0, 0.0, 0.0, 0.0]),
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
            m.material_mut().temperature_k = start.1;
            if let Some([a, b0, b1, b2]) = self.terms {
                m.normalization = Value::Fitted(a);
                m.regions[0].background = [Value::Fitted(b0), Value::Fitted(b1), Value::Known(b2)];
            }
            if let Some(sd) = self.measured_density_sd {
                let mut rng = ChaCha12Rng::seed_from_u64(seed + 2_000_000);
                let value = Normal::new(self.sample.1, sd)
                    .expect("positive sd")
                    .sample(&mut rng);
                m.material_mut().isotopes[0].1 = Value::Measured { value, sd };
            }
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
            let mut estimates = vec![fit.regions[0].densities[0]];
            let mut truths = vec![self.sample.1];
            if matches!(self.temperature_k, Value::Fitted(_)) {
                estimates.push(fit.temperature());
                truths.push(TEMPERATURE_K);
            }
            if let Some(terms) = self.terms {
                estimates.extend([
                    fit.normalization,
                    fit.regions[0].background[0],
                    fit.regions[0].background[1],
                ]);
                truths.extend(&terms[..3]);
            }
            let mut pulls: Vec<f64> = estimates
                .iter()
                .zip(&truths)
                .enumerate()
                .map(|(i, (x, truth))| (x - truth) / covariance.get(i, i).sqrt())
                .collect();
            pulls.extend(fit.measured_pulls.as_ref()?);
            let reported_correlation = (estimates.len() >= 2).then(|| {
                covariance.get(0, 1) / (covariance.get(0, 0) * covariance.get(1, 1)).sqrt()
            });
            pulls.iter().all(|p| p.is_finite()).then_some(Draw {
                pulls,
                estimates,
                reported_correlation,
                overdispersion: [
                    fit.regions[0].overdispersion[0]?,
                    fit.regions[0].overdispersion[1]?,
                ],
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

    pub(super) fn moments(values: &[f64]) -> (f64, f64) {
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

    fn overdispersion_matches(kept: &[&Draw], counts_per_neutron: [f64; 2]) {
        for (run, per_neutron) in counts_per_neutron.iter().enumerate() {
            let ratios: Vec<f64> = kept
                .iter()
                .map(|d| d.overdispersion[run] / per_neutron)
                .collect();
            let (mean, sd) = moments(&ratios);
            let band = Z * sd / (ratios.len() as f64).sqrt();
            assert!(
                (mean - 1.0).abs() <= band,
                "run {run}: overdispersion {mean} ± {band}"
            );
        }
    }

    fn saturated(
        temperature_k: Value,
        level: f64,
        counts_per_neutron: [f64; 2],
        open_run_fraction: f64,
        seeds: std::ops::Range<u64>,
    ) -> Ensemble {
        Ensemble {
            setup: standard(),
            sample: (two_resonances(), SATURATED),
            temperature_k,
            terms: None,
            level,
            counts_per_neutron,
            open_run_fraction,
            measured_density_sd: None,
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
            terms: None,
            level: 1.0e5,
            counts_per_neutron: [1.0; 2],
            open_run_fraction: 1.0,
            measured_density_sd: None,
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
    fn a_measured_density_and_each_run_s_overdispersion_set_the_error_bars() {
        let counts_per_neutron = [3.0, 7.0];
        let mut ensemble = saturated(
            Value::Fitted(TEMPERATURE_K),
            1.0e4,
            counts_per_neutron,
            1.0,
            20_000..20_400,
        );
        let expected = ensemble.expected();
        let noiseless = fit_counts(
            &fitted_from(
                measurement(
                    &ensemble.setup,
                    (rounded(&expected.0), rounded(&expected.1)),
                    std::slice::from_ref(&ensemble.sample),
                ),
                TEMPERATURE_K,
            ),
            &calibration(&ensemble.setup),
        )
        .expect("fit");
        ensemble.measured_density_sd =
            Some(error_bar(&noiseless, 0) * counts_per_neutron[1].sqrt());
        let draws = ensemble.draws(&expected, (ensemble.sample.1, ensemble.temperature_k));
        let kept = check(&draws);
        overdispersion_matches(&kept, counts_per_neutron);
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_short_open_run_widens_the_error_bars_by_its_noise() {
        let ensemble = saturated(
            Value::Fitted(TEMPERATURE_K),
            1.0e4,
            [1.0; 2],
            0.1,
            30_000..30_400,
        );
        check(&ensemble.draws_from_truth());
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_few_counts_per_bin_with_empty_bins_give_their_error_bars() {
        let ensemble = saturated(
            Value::Known(TEMPERATURE_K),
            3.0,
            [1.0; 2],
            1.0,
            40_000..40_400,
        );
        check(&ensemble.draws_from_truth());
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn the_overdispersion_measures_compound_counts_at_a_few_counts_per_bin() {
        let ensemble = saturated(
            Value::Known(TEMPERATURE_K),
            3.0,
            [7.0; 2],
            1.0,
            50_000..50_400,
        );
        let draws = ensemble.draws_from_truth();
        let kept = check(&draws);
        overdispersion_matches(&kept, [7.0; 2]);
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn error_bars_with_the_normalization_and_background_fitted_are_the_scatter() {
        let ensemble = Ensemble {
            terms: Some(TERMS),
            ..saturated(
                Value::Fitted(TEMPERATURE_K),
                1.0e4,
                [1.0; 2],
                1.0,
                60_000..60_400,
            )
        };
        check(&ensemble.draws_from_truth());
    }
}

#[test]
fn density_temperature_timing_offset_and_flight_path_are_recovered_from_starts_on_either_side() {
    let setup = Setup {
        edges: (170..=280).map(|t| 2.0 * f64::from(t)).collect(),
        energy_range_ev: (5.0, 60.0),
        t0_us: T0_US + 0.02,
        flight_path_m: FLIGHT_PATH_M + 0.002,
        ..standard()
    };
    let sample = [(
        synthetic_isotope_multi(72, 180, &[(12.0, 0.01, 0.06), (24.0, 0.01, 0.06)]),
        THIN,
    )];
    let (open, transmitted) = expected(&setup, &beam(1.0e6), &sample);
    let counts = (rounded(&open), rounded(&transmitted));
    let empty = expected(&setup, &beam(2.0e6), &[]);
    let precision = 1.0 / open.iter().copied().fold(f64::INFINITY, f64::min).sqrt();
    let (t0_offset, path_offset) = (1.5, 0.08);
    for (sign, density, start_k) in [(1.0, 2.0 * THIN, 1000.0), (-1.0, 0.5 * THIN, 200.0)] {
        let calibration = Calibration {
            t0_us: Value::Fitted(setup.t0_us + sign * t0_offset),
            flight_path_m: Value::Fitted(setup.flight_path_m + sign * path_offset),
            ..calibration(&setup)
        };
        let start = [(sample[0].0.clone(), density)];
        let mut m = fitted_from(measurement(&setup, counts.clone(), &start), start_k);
        m.regions.push(Region {
            open_counts: rounded(&empty.0),
            sample_counts: rounded(&empty.1),
            open_live: None,
            sample_live: None,
            background: [Value::Known(0.0); 3],
            material: None,
        });
        let fit = fit_counts(&m, &calibration).expect("fit");
        assert!(fit.converged, "{sign}");
        let simulated = [(&open, &transmitted), (&empty.0, &empty.1)]
            .iter()
            .zip(&fit.regions)
            .map(|((o, s), region)| {
                distance(&region.predicted[0], o) + distance(&region.predicted[1], s)
            })
            .sum::<f64>();
        assert!(simulated <= BOUND, "{sign}: {simulated}");
        for (i, (estimate, truth)) in [
            (fit.regions[0].densities[0], THIN),
            (fit.temperature(), TEMPERATURE_K),
            (fit.t0_us, setup.t0_us),
            (fit.flight_path_m, setup.flight_path_m),
        ]
        .into_iter()
        .enumerate()
        {
            let pull = (estimate - truth) / error_bar(&fit, i);
            assert!(pull.abs() <= BOUND.sqrt(), "{sign} {i}: {pull}");
        }
        let beam_error = [1.0e6, 2.0e6]
            .iter()
            .zip(&fit.regions)
            .flat_map(|(&level, region)| {
                setup.edges.iter().map(move |&t| {
                    (region.beam.per_us(t - (setup.t0_us + sign * t0_offset))
                        / beam(level)(t - setup.t0_us)
                        - 1.0)
                        .abs()
                })
            })
            .fold(0.0, f64::max);
        assert!(
            beam_error <= precision,
            "{sign}: {beam_error} vs {precision}"
        );
    }
}

const CALIBRATION_PULSE: [f64; 6] = [0.5, 1.0, 0.08, 0.0, 0.2, 0.0];
const CALIBRATION_LINES: [(f64, f64, f64); 3] =
    [(10.0, 0.05, 0.06), (25.0, 0.005, 0.06), (50.0, 0.01, 0.06)];
const CALIBRATION_DENSITY: f64 = 2.0e-3;
const CALIBRATION_LEVEL: f64 = 2.0e6;
const DENSITY_RELATIVE_SD: f64 = 0.01;
const TEMPERATURE_SD_K: f64 = 10.0;

fn venus_pulse(numbers: &[f64]) -> Arc<IkedaCarpenter> {
    Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::SqrtE {
                    a0: numbers[0],
                    a1: numbers[1],
                },
                beta: EnergyLaw::SqrtE {
                    a0: numbers[2],
                    a1: numbers[3],
                },
                r: EnergyLaw::Const(numbers[4]),
                burst_sigma_us: None,
                channel_fwhm_us: Some(numbers[5]),
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
    )
}

fn calibration_foil(pulse: &[f64], t0_us: f64, flight_path_m: f64) -> Setup {
    foil_window(&CALIBRATION_LINES, pulse, t0_us, flight_path_m)
}

fn foil_window(lines: &[(f64, f64, f64)], pulse: &[f64], t0_us: f64, flight_path_m: f64) -> Setup {
    let lines_us: Vec<f64> = lines
        .iter()
        .map(|&(energy, _, _)| T0_US + CLOCK / energy.sqrt())
        .collect();
    let edges = std::iter::successors(Some(240.0), |&t: &f64| {
        let near_a_line = lines_us.iter().any(|line| (t - line).abs() < 10.0);
        (t < 600.0).then_some(t + if near_a_line { 1.0 } else { 8.0 })
    })
    .collect();
    Setup {
        edges,
        pulse: venus_pulse(pulse),
        energy_range_ev: (4.0, 120.0),
        simulator_step_us: 1.0 / 32.0,
        t0_us,
        flight_path_m,
    }
}

fn calibration_sample(density: f64) -> [(ResonanceData, f64); 1] {
    [(
        synthetic_isotope_multi(73, 181, &CALIBRATION_LINES),
        density,
    )]
}

fn calibration_measurement(setup: &Setup, counts: (Vec<f64>, Vec<f64>)) -> Measurement {
    foil_measurement(setup, counts, &calibration_sample(CALIBRATION_DENSITY))
}

fn foil_measurement(
    setup: &Setup,
    counts: (Vec<f64>, Vec<f64>),
    sample: &[(ResonanceData, f64)],
) -> Measurement {
    let mut m = measurement(setup, counts, sample);
    m.material_mut().isotopes[0].1 = Value::Measured {
        value: sample[0].1,
        sd: DENSITY_RELATIVE_SD * sample[0].1,
    };
    m.material_mut().temperature_k = Value::Measured {
        value: TEMPERATURE_K,
        sd: TEMPERATURE_SD_K,
    };
    m
}

fn calibration_start(truth: &[f64], sign: f64, fwhm_start_us: Option<f64>) -> Calibration {
    let start = |value: f64, offset: f64| Value::Fitted(value + sign * offset);
    Calibration {
        t0_us: start(T0_US, 0.05),
        flight_path_m: start(FLIGHT_PATH_M, 0.003),
        pulse: Pulse {
            alpha: [
                start(truth[0], 0.2 * truth[0]),
                start(truth[1], 0.2 * truth[1]),
            ],
            beta: [start(truth[2], 0.2 * truth[2]), Value::Known(truth[3])],
            r: start(truth[4], 0.05),
            fwhm_squared_us2: fwhm_start_us
                .map_or(Value::Known(truth[5].powi(2)), |h| Value::Fitted(h * h)),
            ..known(&venus_pulse(truth))
        },
    }
}

fn calibrated(m: &Measurement, truth: &[f64], sign: f64, fwhm_start_us: Option<f64>) -> CountsFit {
    fit_counts(m, &calibration_start(truth, sign, fwhm_start_us)).expect("fit")
}

fn calibration_estimates(fit: &CountsFit) -> [f64; 9] {
    [
        fit.regions[0].densities[0],
        fit.temperature(),
        fit.t0_us,
        fit.flight_path_m,
        fit.alpha[0],
        fit.alpha[1],
        fit.beta[0],
        fit.r,
        fit.fwhm_squared_us2,
    ]
}

fn calibration_truth(pulse: &[f64]) -> [f64; 9] {
    [
        CALIBRATION_DENSITY,
        TEMPERATURE_K,
        T0_US,
        FLIGHT_PATH_M,
        pulse[0],
        pulse[1],
        pulse[2],
        pulse[4],
        pulse[5].powi(2),
    ]
}

fn assert_recovered(fit: &CountsFit, pulse: &[f64], fitted: usize, case: &str) {
    assert!(fit.converged, "{case}");
    let estimates = calibration_estimates(fit);
    for (i, (estimate, truth)) in estimates
        .iter()
        .zip(calibration_truth(pulse))
        .enumerate()
        .take(fitted)
    {
        let pull = (estimate - truth) / error_bar(fit, i);
        assert!(pull.abs() <= BOUND.sqrt(), "{case} {i}: {pull}");
    }
}

fn foil_provenance() -> Provenance {
    Provenance {
        foil: "foil-a".into(),
        open: "open-a".into(),
        sample: "sample-a".into(),
    }
}

static ONE_FOIL: LazyLock<(PulseCalibration, CountsFit)> = LazyLock::new(|| {
    let setup = calibration_foil(&CALIBRATION_PULSE, T0_US, FLIGHT_PATH_M);
    let counts = expected(
        &setup,
        &beam(CALIBRATION_LEVEL),
        &calibration_sample(CALIBRATION_DENSITY),
    );
    let m = calibration_measurement(&setup, (rounded(&counts.0), rounded(&counts.1)));
    PulseCalibration::new(
        &m,
        &calibration_start(&CALIBRATION_PULSE, 1.0, None),
        foil_provenance(),
    )
    .expect("calibration")
});

#[test]
fn a_calibrated_pulse_reads_back_from_its_file_bit_for_bit() {
    let text = ONE_FOIL.0.to_json();
    let read = PulseCalibration::from_json(&text).expect("file");
    assert_eq!(read.to_json(), text);
    assert_eq!(
        format!("{:?}", read.calibration()),
        format!("{:?}", ONE_FOIL.0.calibration())
    );
}

#[test]
fn the_pulse_is_calibrated_on_one_foil() {
    assert_recovered(&ONE_FOIL.1, &CALIBRATION_PULSE, 8, "start above the truth");
}

const EXPERIMENT_K: f64 = 1500.0;

fn experiment_foil(pulse: &[f64], t0_us: f64, flight_path_m: f64) -> Setup {
    let mut setup = calibration_foil(pulse, t0_us, flight_path_m);
    setup.edges.retain(|&t| t <= 400.0);
    setup
}

fn experiment_setup(pulse: &[f64]) -> Setup {
    experiment_foil(pulse, T0_US + 0.03, FLIGHT_PATH_M + 0.002)
}

fn experiment_counts(pulse: &[f64]) -> (Vec<f64>, Vec<f64>) {
    expected_at(
        &experiment_setup(pulse),
        &beam(CALIBRATION_LEVEL),
        &calibration_sample(CALIBRATION_DENSITY),
        EXPERIMENT_K,
    )
}

fn experiment_with(
    pulse: &[f64],
    counts: (Vec<f64>, Vec<f64>),
    calibration: &Calibration,
) -> CountsFit {
    let sample = calibration_sample(CALIBRATION_DENSITY);
    let m = fitted_from(
        measurement(&experiment_setup(pulse), counts, &sample),
        EXPERIMENT_K,
    );
    fit_counts(&m, calibration).expect("fit")
}

fn experiment(pulse: &[f64]) -> CountsFit {
    let counts = experiment_counts(pulse);
    experiment_with(
        pulse,
        (rounded(&counts.0), rounded(&counts.1)),
        &ONE_FOIL.0.calibration(),
    )
}

#[test]
fn an_experiment_made_with_another_pulse_rejects_the_calibration() {
    let mut shifted = CALIBRATION_PULSE;
    shifted[4] += 0.01;
    let fit = experiment(&shifted);
    assert!(fit.converged);
    let consistency = fit.pulse_consistency.expect("consistency");
    assert!(consistency.p < 0.01, "{consistency:?}");
    let block = |unbounded: &Unbounded| {
        let mut covariance = FlatMatrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                *covariance.get_mut(i, j) = unbounded.covariance.get(4 + i, 4 + j);
            }
        }
        (unbounded.mean[4..8].to_vec(), covariance)
    };
    let (mean, covariance) = block(ONE_FOIL.1.unbounded.as_ref().expect("unbounded"));
    let prior = Prior::correlated(&[0, 1, 2, 3], &mean, &covariance).expect("prior");
    let (estimate, posterior) = block(fit.unbounded.as_ref().expect("unbounded"));
    let expected = statistics::consistency(&prior, &estimate, &posterior)
        .expect("consistency")
        .expect("statistic");
    assert!(
        (consistency.q / expected.q - 1.0).abs() <= 1e-9,
        "{consistency:?} vs {expected:?}"
    );
}

#[test]
fn a_sample_or_pulse_the_calibration_does_not_cover_is_refused() {
    let setup = calibration_foil(&CALIBRATION_PULSE, T0_US, FLIGHT_PATH_M);
    let bins = setup.edges.len() - 1;
    let flat = (vec![100.0; bins], vec![100.0; bins]);
    let refusal = |sample: &[(ResonanceData, f64)], calibration: &Calibration| match fit_counts(
        &measurement(&setup, flat.clone(), sample),
        calibration,
    ) {
        Err(PipelineError::InvalidParameter(message)) => message,
        other => panic!("{other:?}"),
    };
    let calibrated = ONE_FOIL.0.calibration();
    let wider = [(hafnium_like(55.0), CALIBRATION_DENSITY)];
    assert!(refusal(&wider, &calibrated).contains("55 eV, outside the 10–50 eV"));
    let mut known = calibrated;
    known.pulse.r = Value::Known(0.2);
    assert!(refusal(&calibration_sample(CALIBRATION_DENSITY), &known).contains("covers R"));
}

mod pulse_calibration {
    use rayon::prelude::*;

    use super::*;

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_calibration_from_below_converges_with_the_information_as_its_covariance() {
        let mut p = CALIBRATION_PULSE;
        p[5] = 0.35;
        let b0 = 0.05;
        let setup = calibration_foil(&p, T0_US, FLIGHT_PATH_M);
        let counts = with_background(
            &setup,
            &beam(CALIBRATION_LEVEL),
            &calibration_sample(CALIBRATION_DENSITY),
            TEMPERATURE_K,
            [1.0, b0, 0.0, 0.0],
        );
        let mut m = calibration_measurement(&setup, (rounded(&counts.0), rounded(&counts.1)));
        m.regions[0].background[0] = Value::Fitted(b0);
        let fit = calibrated(&m, &p, -1.0, Some(p[5] - 0.1));
        assert!(fit.converged);
        let fitted = [
            fit.regions[0].densities[0],
            fit.temperature(),
            fit.regions[0].background[0],
            fit.t0_us,
            fit.flight_path_m,
            fit.alpha[0],
            fit.alpha[1],
            fit.beta[0],
            fit.r,
            fit.fwhm_squared_us2,
        ];
        let truth = [
            CALIBRATION_DENSITY,
            TEMPERATURE_K,
            b0,
            T0_US,
            FLIGHT_PATH_M,
            p[0],
            p[1],
            p[2],
            p[4],
            p[5] * p[5],
        ];
        for (i, (estimate, truth)) in fitted.iter().zip(truth).enumerate() {
            let pull = (estimate - truth) / error_bar(&fit, i);
            assert!(pull.abs() <= BOUND.sqrt(), "{i}: {pull}");
        }

        let beam_origin_us = T0_US - 0.05;
        let counts_at = |quantities: &[f64], coefficient: Option<(usize, f64)>| -> Vec<f64> {
            let [
                density,
                temperature_k,
                b0,
                t0_us,
                flight_path_m,
                a0,
                a1,
                beta0,
                r,
                fwhm_squared,
            ] = quantities.try_into().expect("ten quantities");
            let at = calibration_foil(
                &[a0, a1, beta0, p[3], r, fwhm_squared.sqrt()],
                t0_us,
                flight_path_m,
            );
            let (open, sample) = with_background(
                &at,
                &fitted_beam(&fit, beam_origin_us, t0_us, coefficient),
                &calibration_sample(density),
                temperature_k,
                [1.0, b0, 0.0, 0.0],
            );
            open.into_iter().chain(sample).collect()
        };
        let coefficients = fit.regions[0].beam.coefficients().len();
        let mut information = information(&fit, &fitted, 3, counts_at);
        information[coefficients][coefficients] +=
            (DENSITY_RELATIVE_SD * CALIBRATION_DENSITY).powi(-2);
        information[coefficients + 1][coefficients + 1] += TEMPERATURE_SD_K.powi(-2);
        assert_covariance(&fit, &inverse(information));
    }

    fn fitted_beam(
        fit: &CountsFit,
        beam_origin_us: f64,
        t0_us: f64,
        coefficient: Option<(usize, f64)>,
    ) -> impl Fn(f64) -> f64 + '_ {
        move |u: f64| {
            let x = t0_us + u - beam_origin_us;
            let slope: f64 = coefficient.map_or(0.0, |(i, step)| {
                step * fit.regions[0]
                    .beam
                    .basis(x)
                    .iter()
                    .filter(|w| w.0 == i)
                    .map(|w| w.1)
                    .sum::<f64>()
            });
            fit.regions[0].beam.per_us(x) * slope.exp()
        }
    }

    fn information(
        fit: &CountsFit,
        fitted: &[f64],
        t0_at: usize,
        counts_at: impl Fn(&[f64], Option<(usize, f64)>) -> Vec<f64> + Sync,
    ) -> Vec<Vec<f64>> {
        let mu = counts_at(fitted, None);
        let coefficients = fit.regions[0].beam.coefficients().len();
        let columns: Vec<Vec<f64>> = (0..coefficients + fitted.len())
            .into_par_iter()
            .map(|c| {
                let shifted = |sign: f64| -> (Vec<f64>, f64) {
                    if c < coefficients {
                        let step = 1e-4;
                        return (counts_at(fitted, Some((c, sign * step))), step);
                    }
                    let q = c - coefficients;
                    let step = match q {
                        _ if q == t0_at => T0_STEP_US,
                        _ if q == t0_at + 1 => PATH_STEP_M,
                        _ => 1e-4 * fitted[q],
                    };
                    let mut quantities = fitted.to_vec();
                    quantities[q] += sign * step;
                    (counts_at(&quantities, None), step)
                };
                let ((up, step), (down, _)) = (shifted(1.0), shifted(-1.0));
                up.iter()
                    .zip(&down)
                    .map(|(a, b)| (a - b) / (2.0 * step))
                    .collect()
            })
            .collect();
        let bins = mu.len() / 2;
        let overdispersion = fit.regions[0].overdispersion.map(|phi| phi.unwrap_or(1.0));
        columns
            .iter()
            .map(|a| {
                columns
                    .iter()
                    .map(|b| {
                        a.iter()
                            .zip(b)
                            .zip(&mu)
                            .enumerate()
                            .filter(|(_, (_, m))| **m > 0.0)
                            .map(|(k, ((x, y), m))| x * y / (m * overdispersion[k / bins]))
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_covariance(fit: &CountsFit, oracle: &[Vec<f64>]) {
        let coefficients = fit.regions[0].beam.coefficients().len();
        let covariance = fit.covariance.as_ref().expect("covariance");
        let fitted = oracle.len() - coefficients;
        for i in 0..fitted {
            for j in 0..fitted {
                let expected = oracle[coefficients + i][coefficients + j];
                let scale = (oracle[coefficients + i][coefficients + i]
                    * oracle[coefficients + j][coefficients + j])
                    .sqrt();
                assert!(
                    (covariance.get(i, j) - expected).abs() <= 1e-2 * scale,
                    "{i}, {j}: {} vs {expected}",
                    covariance.get(i, j)
                );
            }
        }
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn the_triangle_is_calibrated_with_the_pulse() {
        let mut p = CALIBRATION_PULSE;
        p[5] = 0.35;
        let setup = calibration_foil(&p, T0_US, FLIGHT_PATH_M);
        let counts = expected(
            &setup,
            &beam(CALIBRATION_LEVEL),
            &calibration_sample(CALIBRATION_DENSITY),
        );
        let m = calibration_measurement(&setup, (rounded(&counts.0), rounded(&counts.1)));
        for (sign, fwhm_start_us) in [(1.0, p[5] + 0.1), (-1.0, p[5] - 0.1), (1.0, 0.0)] {
            let fit = calibrated(&m, &p, sign, Some(fwhm_start_us));
            assert_recovered(&fit, &p, 9, &format!("start {sign}, h {fwhm_start_us}"));
        }
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_moderation_rate_without_energy_dependence_ends_on_its_bound_half_the_time() {
        let mut p = CALIBRATION_PULSE;
        p[0] = 0.0;
        let setup = calibration_foil(&p, T0_US, FLIGHT_PATH_M);
        let expected = expected(
            &setup,
            &beam(CALIBRATION_LEVEL),
            &calibration_sample(CALIBRATION_DENSITY),
        );
        let truth = calibration_truth(&p);
        let fits: Vec<CountsFit> = (0..400_u64)
            .into_par_iter()
            .map(|seed| {
                let m = calibration_measurement(&setup, draws(&expected, 70_000 + seed, [1.0; 2]));
                calibrated(&m, &p, if seed % 2 == 0 { 1.0 } else { -1.0 }, None)
            })
            .filter(|fit| fit.converged)
            .collect();
        assert!(fits.len() >= 398, "{} converged", fits.len());
        let bounded = fits.iter().filter(|fit| fit.on_bound[4]).count() as f64 / fits.len() as f64;
        let spread = 0.5 / (fits.len() as f64).sqrt();
        assert!((bounded - 0.5).abs() <= 3.5 * spread, "{bounded}");
        let unbounded = |fit: &CountsFit, i: usize| {
            let unbounded = fit.unbounded.as_ref().expect("unbounded");
            (unbounded.mean[i] - truth[i]) / unbounded.covariance.get(i, i).sqrt()
        };
        let held = |fit: &CountsFit, i: usize| {
            (calibration_estimates(fit)[i] - truth[i]) / error_bar(fit, i)
        };
        let centred = super::error_bar_pulls::Z / (fits.len() as f64).sqrt();
        let cases = [2, 3, 5, 6, 7]
            .map(|i| (i, &held as &dyn Fn(&CountsFit, usize) -> f64, f64::INFINITY))
            .into_iter()
            .chain((2..8).map(|i| (i, &unbounded as &dyn Fn(&CountsFit, usize) -> f64, centred)));
        for (i, pull, bias) in cases {
            let pulls: Vec<f64> = fits.iter().map(|fit| pull(fit, i)).collect();
            let (mean, sd) = super::error_bar_pulls::moments(&pulls);
            let covered =
                pulls.iter().filter(|pull| pull.abs() <= 1.0).count() as f64 / pulls.len() as f64;
            assert!(
                mean.abs() <= bias && (0.9..=1.1).contains(&sd) && (0.61..=0.75).contains(&covered),
                "{i}: mean {mean} within {bias}, sd {sd}, within one sd {covered}"
            );
        }
    }

    const CHI_SQUARED_5_AT_0_05: f64 = 11.070_497_693_516_351;

    fn calibrated_numbers(fit: &CountsFit) -> [f64; 4] {
        [fit.alpha[0], fit.alpha[1], fit.beta[0], fit.r]
    }

    fn block(covariance: &FlatMatrix, indices: std::ops::Range<usize>) -> Vec<Vec<f64>> {
        indices
            .clone()
            .map(|i| indices.clone().map(|j| covariance.get(i, j)).collect())
            .collect()
    }

    fn calibration_block() -> Vec<Vec<f64>> {
        block(
            &ONE_FOIL.1.unbounded.as_ref().expect("unbounded").covariance,
            4..8,
        )
    }

    fn experiment_information(fit: &CountsFit) -> Vec<Vec<f64>> {
        let numbers = calibrated_numbers(fit);
        let fitted = [
            fit.regions[0].densities[0],
            fit.temperature(),
            fit.t0_us,
            fit.flight_path_m,
            numbers[0],
            numbers[1],
            numbers[2],
            numbers[3],
        ];
        let beam_origin_us = ONE_FOIL.1.t0_us;
        let counts_at = |quantities: &[f64], coefficient: Option<(usize, f64)>| -> Vec<f64> {
            let [
                density,
                temperature_k,
                t0_us,
                flight_path_m,
                a0,
                a1,
                beta0,
                r,
            ] = quantities.try_into().expect("eight quantities");
            let at = experiment_foil(&[a0, a1, beta0, 0.0, r, 0.0], t0_us, flight_path_m);
            let (open, sample) = expected_at(
                &at,
                &fitted_beam(fit, beam_origin_us, t0_us, coefficient),
                &calibration_sample(density),
                temperature_k,
            );
            open.into_iter().chain(sample).collect()
        };
        information(fit, &fitted, 2, counts_at)
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn the_statistic_compares_the_calibrated_pulse_with_the_experiment_s_own() {
        let mut shifted = CALIBRATION_PULSE;
        shifted[4] += 0.02;
        let counts = experiment_counts(&shifted);
        let counts = (rounded(&counts.0), rounded(&counts.1));
        let mut unmeasured = ONE_FOIL.0.calibration();
        unmeasured.pulse.prior = None;
        let (measured, alone) = rayon::join(
            || experiment_with(&shifted, counts.clone(), &ONE_FOIL.0.calibration()),
            || experiment_with(&shifted, counts.clone(), &unmeasured),
        );
        assert!(measured.converged && alone.converged);
        let mut weighted = alone.clone();
        weighted.regions[0].overdispersion = measured.regions[0].overdispersion;
        let coefficients = alone.regions[0].beam.coefficients().len();
        let own = inverse(experiment_information(&weighted));
        let start = coefficients + 4;
        let sum: Vec<Vec<f64>> = calibration_block()
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, c)| c + own[start + i][start + j])
                    .collect()
            })
            .collect();
        let weight = inverse(sum);
        let difference: Vec<f64> = calibrated_numbers(&alone)
            .iter()
            .zip(&ONE_FOIL.1.unbounded.as_ref().expect("unbounded").mean[4..8])
            .map(|(x, c)| x - c)
            .collect();
        let expected: f64 = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .map(|(i, j)| difference[i] * weight[i][j] * difference[j])
            .sum();
        let q = measured.pulse_consistency.expect("consistency").q;
        assert!((q / expected - 1.0).abs() <= 0.05, "{q} vs {expected}");
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn an_experiment_s_covariance_adds_the_calibration_s_inverse_to_its_information() {
        let fit = experiment(&CALIBRATION_PULSE);
        assert!(fit.converged);
        let coefficients = fit.regions[0].beam.coefficients().len();
        let mut information = experiment_information(&fit);
        let precision = inverse(calibration_block());
        for (a, row) in precision.iter().enumerate() {
            for (b, p) in row.iter().enumerate() {
                information[coefficients + 4 + a][coefficients + 4 + b] += p;
            }
        }
        assert_covariance(&fit, &inverse(information));
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn calibrated_experiments_have_error_bars_that_are_their_scatter() {
        let mut p = CALIBRATION_PULSE;
        p[5] = 0.35;
        let foil = calibration_foil(&p, T0_US, FLIGHT_PATH_M);
        let foil_counts = expected(
            &foil,
            &beam(CALIBRATION_LEVEL),
            &calibration_sample(CALIBRATION_DENSITY),
        );
        let counts = experiment_counts(&p);
        let fits: Vec<(CountsFit, [Option<f64>; 2])> = (0..400_u64)
            .into_par_iter()
            .filter_map(|seed| {
                let m =
                    calibration_measurement(&foil, draws(&foil_counts, 80_000 + seed, [1.0; 2]));
                let sign = if seed % 2 == 0 { 1.0 } else { -1.0 };
                let calibration = calibration_start(&p, sign, Some(p[5] + 0.05 * sign));
                let (calibrated, fit) =
                    PulseCalibration::new(&m, &calibration, foil_provenance()).ok()?;
                let experiment = calibrated.calibration();
                let drawn = draws(&counts, 90_000 + seed, [1.0; 2]);
                Some((
                    experiment_with(&p, drawn, &experiment),
                    fit.regions[0].overdispersion,
                ))
                .filter(|(fit, _)| fit.converged)
            })
            .collect();
        assert!(fits.len() >= 398, "{} converged", fits.len());
        let truth = [
            (0, CALIBRATION_DENSITY),
            (1, EXPERIMENT_K),
            (2, T0_US + 0.03),
            (3, FLIGHT_PATH_M + 0.002),
            (4, p[0]),
            (5, p[1]),
            (6, p[2]),
            (7, p[4]),
            (8, p[5] * p[5]),
        ];
        for (i, truth) in truth {
            let pulls: Vec<f64> = fits
                .iter()
                .map(|(fit, _)| {
                    let estimate = [
                        fit.regions[0].densities[0],
                        fit.temperature(),
                        fit.t0_us,
                        fit.flight_path_m,
                        fit.alpha[0],
                        fit.alpha[1],
                        fit.beta[0],
                        fit.r,
                        fit.fwhm_squared_us2,
                    ][i];
                    (estimate - truth) / error_bar(fit, i)
                })
                .collect();
            let (mean, sd) = super::error_bar_pulls::moments(&pulls);
            let m = pulls.len() as f64;
            let covered = pulls.iter().filter(|pull| pull.abs() <= 1.0).count() as f64 / m;
            assert!(
                mean.abs() <= super::error_bar_pulls::Z / m.sqrt()
                    && (0.9..=1.1).contains(&sd)
                    && (0.61..=0.75).contains(&covered),
                "{i}: mean {mean}, sd {sd}, within one sd {covered}"
            );
        }
        let pairs: Vec<_> = fits.iter().map(|(fit, phi)| (fit, *phi)).collect();
        pulse_consistencies_follow_chi_squared(&pairs);
    }

    fn pulse_consistencies_follow_chi_squared(fits: &[(&CountsFit, [Option<f64>; 2])]) {
        let weighted: Vec<f64> = fits
            .iter()
            .map(|(fit, calibration)| {
                let consistency = fit.pulse_consistency.expect("consistency");
                assert_eq!(consistency.dof, 5);
                let weights: Vec<f64> = fit.regions[0]
                    .overdispersion
                    .iter()
                    .chain(calibration)
                    .map(|phi| phi.unwrap_or(1.0))
                    .collect();
                consistency.q * weights.iter().sum::<f64>() / weights.len() as f64
            })
            .collect();
        let n = weighted.len() as f64;
        let reported = fits
            .iter()
            .filter(|(fit, _)| fit.pulse_consistency.expect("consistency").p < 0.05)
            .count() as f64
            / n;
        assert!(
            reported <= 0.05 + 3.0 * (0.05 * 0.95 / n).sqrt(),
            "{reported} rejected at a reported 0.05"
        );
        let q = weighted.iter().sum::<f64>() / n;
        let rejected = weighted
            .iter()
            .filter(|&&q| q > CHI_SQUARED_5_AT_0_05)
            .count() as f64
            / n;
        assert!(
            (q - 5.0).abs() <= 3.0 * (10.0 / n).sqrt()
                && (rejected - 0.05).abs() <= 3.0 * (0.05 * 0.95 / n).sqrt(),
            "mean q {q} on 5 degrees of freedom, {rejected} rejected at 0.05"
        );
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn experiments_calibrated_with_a_moderation_rate_at_its_bound_keep_density_and_temperature() {
        let mut bounded = 0;
        for (case, alpha0) in [0.0, 5e-4].into_iter().enumerate() {
            let mut p = CALIBRATION_PULSE;
            p[0] = alpha0;
            p[5] = 0.35;
            let foil = calibration_foil(&p, T0_US, FLIGHT_PATH_M);
            let foil_counts = expected(
                &foil,
                &beam(CALIBRATION_LEVEL),
                &calibration_sample(CALIBRATION_DENSITY),
            );
            let counts = experiment_counts(&p);
            let seeds = 1_000 * case as u64;
            let fits: Vec<(CountsFit, [Option<f64>; 2], bool)> = (0..400_u64)
                .into_par_iter()
                .filter_map(|seed| {
                    let drawn = draws(&foil_counts, 81_000 + seeds + seed, [1.0; 2]);
                    let m = calibration_measurement(&foil, drawn);
                    let sign = if seed % 2 == 0 { 1.0 } else { -1.0 };
                    let mut calibration = calibration_start(&p, sign, Some(p[5] + 0.05 * sign));
                    calibration.pulse.alpha[0] = Value::Fitted(0.05);
                    let (calibrated, calibration_fit) =
                        PulseCalibration::new(&m, &calibration, foil_provenance()).ok()?;
                    let drawn = draws(&counts, 91_000 + seeds + seed, [1.0; 2]);
                    Some((
                        experiment_with(&p, drawn, &calibrated.calibration()),
                        calibration_fit.regions[0].overdispersion,
                        calibration_fit.on_bound[4],
                    ))
                    .filter(|(fit, _, _)| fit.converged)
                })
                .collect();
            assert!(fits.len() >= 398, "{alpha0}: {} converged", fits.len());
            let n = fits.len() as f64;
            let held = fits.iter().filter(|(_, _, held)| *held).count();
            assert!(held > 0, "{alpha0}: no calibration ended on the bound");
            bounded += fits.iter().filter(|(fit, _, _)| fit.on_bound[4]).count();
            for (i, truth) in [(0, CALIBRATION_DENSITY), (1, EXPERIMENT_K)] {
                let pulls: Vec<f64> = fits
                    .iter()
                    .map(|(fit, _, _)| {
                        let estimate = [fit.regions[0].densities[0], fit.temperature()][i];
                        (estimate - truth) / error_bar(fit, i)
                    })
                    .collect();
                let (mean, sd) = super::error_bar_pulls::moments(&pulls);
                let covered = pulls.iter().filter(|pull| pull.abs() <= 1.0).count() as f64 / n;
                assert!(
                    mean.abs() <= super::error_bar_pulls::Z / n.sqrt()
                        && (0.9..=1.1).contains(&sd)
                        && (0.61..=0.75).contains(&covered),
                    "{alpha0} {i}: mean {mean}, sd {sd}, within one sd {covered}"
                );
            }
            let pairs: Vec<_> = fits.iter().map(|(fit, phi, _)| (fit, *phi)).collect();
            pulse_consistencies_follow_chi_squared(&pairs);
        }
        assert!(bounded > 0, "no experiment ended on the bound");
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_line_the_fitted_flight_path_brings_into_the_window_is_refused() {
        let flight_path_m = FLIGHT_PATH_M + 0.2;
        let setup = experiment_foil(&CALIBRATION_PULSE, T0_US + 0.03, flight_path_m);
        let high_ev =
            (CLOCK * flight_path_m / FLIGHT_PATH_M / (setup.edges[0] - T0_US - 0.03)).powi(2);
        let line_ev = 0.5 * (high_ev + (CLOCK / (setup.edges[0] - T0_US)).powi(2));
        let mut sample = calibration_sample(CALIBRATION_DENSITY).to_vec();
        sample.push((synthetic_isotope(74, 184, line_ev, 0.01, 0.06), 1e-3));
        let counts = expected_at(&setup, &beam(CALIBRATION_LEVEL), &sample, EXPERIMENT_K);
        let m = fitted_from(
            measurement(&setup, (rounded(&counts.0), rounded(&counts.1)), &sample),
            EXPERIMENT_K,
        );
        match fit_counts(&m, &ONE_FOIL.0.calibration()) {
            Err(PipelineError::InvalidParameter(message)) => {
                assert!(message.contains("outside the"), "{message}")
            }
            other => panic!("{line_ev} eV: {other:?}"),
        }
    }

    const SECOND_LINES: [(f64, f64, f64); 3] =
        [(15.0, 0.03, 0.06), (30.0, 0.006, 0.06), (45.0, 0.01, 0.06)];
    const SECOND_DENSITY: f64 = 3.0e-3;

    fn second_sample() -> [(ResonanceData, f64); 1] {
        [(
            synthetic_isotope_multi(74, 184, &SECOND_LINES),
            SECOND_DENSITY,
        )]
    }

    fn second_provenance() -> Provenance {
        Provenance {
            foil: "foil-b".into(),
            open: "open-b".into(),
            sample: "sample-b".into(),
        }
    }

    fn second_calibration(
        pulse: &[f64],
        counts: (Vec<f64>, Vec<f64>),
        start: &Calibration,
    ) -> Option<(PulseCalibration, CountsFit)> {
        let setup = foil_window(&SECOND_LINES, pulse, T0_US, FLIGHT_PATH_M);
        PulseCalibration::new(
            &foil_measurement(&setup, counts, &second_sample()),
            start,
            second_provenance(),
        )
        .ok()
    }

    fn second_counts(pulse: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let setup = foil_window(&SECOND_LINES, pulse, T0_US, FLIGHT_PATH_M);
        expected(&setup, &beam(CALIBRATION_LEVEL), &second_sample())
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_pulse_calibrated_on_one_foil_transfers_to_another_only_if_the_pulse_is_the_same() {
        let start = calibration_start(&CALIBRATION_PULSE, 1.0, None);
        let calibrate = |pulse: &[f64]| {
            let counts = second_counts(pulse);
            second_calibration(pulse, (rounded(&counts.0), rounded(&counts.1)), &start)
                .expect("calibration")
                .0
        };
        let mut shifted = CALIBRATION_PULSE;
        shifted[4] += 0.02;
        let (same, other) = rayon::join(|| calibrate(&CALIBRATION_PULSE), || calibrate(&shifted));
        let a = &ONE_FOIL.0;
        let failing = a.transfer(&other).expect("transfer");
        assert!(failing.agreement().p < 0.01, "{:?}", failing.agreement());
        assert!(a.clone().record_transfer(&other).is_err());
        let transfer = a.transfer(&same).expect("transfer");
        assert!(transfer.agreement().p > 0.01, "{:?}", transfer.agreement());
        let text = a
            .clone()
            .record_transfer(&same)
            .expect("recorded")
            .to_json();
        assert_eq!(
            PulseCalibration::from_json(&text).expect("file").to_json(),
            text
        );
    }

    fn transfer_replicas(pulse: &[f64; 6], seeds: u64) -> Vec<(Consistency, f64, bool)> {
        let first = calibration_foil(pulse, T0_US, FLIGHT_PATH_M);
        let first_counts = expected(
            &first,
            &beam(CALIBRATION_LEVEL),
            &calibration_sample(CALIBRATION_DENSITY),
        );
        let second_counts = second_counts(pulse);
        (0..400_u64)
            .into_par_iter()
            .filter_map(|seed| {
                let sign = if seed % 2 == 0 { 1.0 } else { -1.0 };
                let mut start = calibration_start(pulse, sign, Some(pulse[5] + 0.05 * sign));
                start.pulse.alpha[0] = Value::Fitted(pulse[0] + 0.05);
                let m =
                    calibration_measurement(&first, draws(&first_counts, seeds + seed, [1.0; 2]));
                let (a, a_fit) = PulseCalibration::new(&m, &start, foil_provenance()).ok()?;
                let drawn = draws(&second_counts, seeds + 500 + seed, [1.0; 2]);
                let (b, b_fit) = second_calibration(pulse, drawn, &start)?;
                let transfer = a.transfer(&b).ok()?;
                let weights: Vec<f64> = a_fit.regions[0]
                    .overdispersion
                    .iter()
                    .chain(&b_fit.regions[0].overdispersion)
                    .map(|phi| phi.unwrap_or(1.0))
                    .collect();
                Some((
                    transfer.agreement(),
                    weights.iter().sum::<f64>() / weights.len() as f64,
                    a_fit.on_bound[4] || b_fit.on_bound[4],
                ))
            })
            .collect()
    }

    fn agree_as_chi_squared(tests: &[(Consistency, f64, bool)]) {
        assert!(tests.len() >= 398, "{} transferred", tests.len());
        let n = tests.len() as f64;
        let weighted: Vec<f64> = tests
            .iter()
            .map(|(t, phi, _)| {
                assert_eq!(t.dof, 5);
                t.q * phi
            })
            .collect();
        let q = weighted.iter().sum::<f64>();
        let rejected = weighted
            .iter()
            .filter(|&&q| q > CHI_SQUARED_5_AT_0_05)
            .count() as f64
            / n;
        assert!(
            (q - 5.0 * n).abs() <= 3.0 * (10.0 * n).sqrt()
                && (rejected - 0.05).abs() <= 3.0 * (0.05 * 0.95 / n).sqrt(),
            "d2 {q} summed over {} degrees of freedom, {rejected} rejected at 0.05",
            5.0 * n
        );
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn two_foils_on_the_same_pulse_agree_as_chi_squared_says() {
        let mut p = CALIBRATION_PULSE;
        p[5] = 0.35;
        agree_as_chi_squared(&transfer_replicas(&p, 100_000));
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn two_foils_with_a_moderation_rate_near_its_bound_agree_as_chi_squared_says() {
        let mut p = CALIBRATION_PULSE;
        p[0] = 5e-4;
        p[5] = 0.35;
        let tests = transfer_replicas(&p, 200_000);
        assert!(
            tests.iter().any(|(_, _, bounded)| *bounded),
            "no calibration ended alpha0 on its bound"
        );
        agree_as_chi_squared(&tests);
    }
}

mod maps {
    use super::*;

    struct Tiled {
        setup: Setup,
        isotopes: [ResonanceData; 2],
        counts: [Array3<f64>; 2],
        behind: Array2<bool>,
        empty: Array2<bool>,
        excluded: Array2<bool>,
    }

    fn tiled(
        truths: &[([f64; 2], f64)],
        pick: impl Fn(usize, usize) -> usize,
        shape: (usize, usize),
    ) -> Tiled {
        let isotopes = [
            hafnium_like(20.0),
            synthetic_isotope(74, 182, 24.0, 0.01, 0.06),
        ];
        tiled_from(isotopes, 0.05, truths, pick, shape)
    }

    fn tiled_from(
        isotopes: [ResonanceData; 2],
        b0: f64,
        truths: &[([f64; 2], f64)],
        pick: impl Fn(usize, usize) -> usize,
        (rows, cols): (usize, usize),
    ) -> Tiled {
        let setup = Setup {
            pulse: pulse(0.0, 200.0),
            ..standard()
        };
        let unit: Vec<(Vec<f64>, Vec<f64>)> = truths
            .iter()
            .map(|&(densities, t)| {
                let sample: Vec<(ResonanceData, f64)> =
                    isotopes.iter().cloned().zip(densities).collect();
                with_background(&setup, &beam(1.0e6), &sample, t, [TERMS[0], b0, 0.0, 0.0])
            })
            .chain([with_background(
                &setup,
                &beam(1.0e6),
                &[],
                TEMPERATURE_K,
                [TERMS[0], 0.0, 0.0, 0.0],
            )])
            .collect();
        let bins = setup.edges.len() - 1;
        let shape = (bins, 2 * rows + 1, 2 * cols);
        let counts = [0, 1].map(|run| {
            Array3::from_shape_fn(shape, |(k, y, x)| {
                let source = if y == 2 * rows {
                    truths.len()
                } else {
                    pick(y / 2, x / 2)
                };
                let level = [1.0, 1.3, 0.8, 1.1][2 * (y % 2) + x % 2];
                let (open, sample) = &unit[source];
                (level * [open, sample][run][k]).round()
            })
        });
        let pixels = (2 * rows + 1, 2 * cols);
        Tiled {
            behind: Array2::from_shape_fn(pixels, |(y, _)| y < 2 * rows),
            empty: Array2::from_shape_fn(pixels, |(y, _)| y == 2 * rows),
            excluded: Array2::from_elem(pixels, false),
            setup,
            isotopes,
            counts,
        }
    }

    impl Tiled {
        fn map(&self, temperature_k: Value, empty_background: [Value; 3]) -> MapMeasurement<'_> {
            MapMeasurement {
                time_edges_us: self.setup.edges.clone(),
                charge_ratio: CHARGE_RATIO,
                normalization: Value::Fitted(1.0),
                open_counts: self.counts[0].view(),
                sample_counts: self.counts[1].view(),
                open_live: None,
                sample_live: None,
                excluded: self.excluded.view(),
                sample: self.behind.view(),
                empty: self.empty.view(),
                binning: 2,
                material: Material {
                    isotopes: self
                        .isotopes
                        .iter()
                        .map(|data| (data.clone(), Value::Fitted(THIN)))
                        .collect(),
                    temperature_k,
                },
                background: [Value::Fitted(0.02), Value::Known(0.0), Value::Known(0.0)],
                empty_background,
            }
        }

        fn calibration(&self) -> Calibration {
            Calibration {
                t0_us: Value::Fitted(T0_US),
                flight_path_m: Value::Fitted(FLIGHT_PATH_M),
                ..calibration(&self.setup)
            }
        }

        fn summed(&self, run: usize, ys: Range<usize>, xs: Range<usize>) -> Vec<f64> {
            (0..self.counts[run].dim().0)
                .map(|k| {
                    ys.clone()
                        .flat_map(|y| xs.clone().map(move |x| (y, x)))
                        .map(|(y, x)| self.counts[run][[k, y, x]])
                        .sum()
                })
                .collect()
        }

        fn region(&self, map: &MapMeasurement<'_>, (i, j): (usize, usize)) -> Region {
            let (ys, xs) = (2 * i..2 * i + 2, 2 * j..2 * j + 2);
            Region {
                open_counts: self.summed(0, ys.clone(), xs.clone()),
                sample_counts: self.summed(1, ys, xs),
                open_live: None,
                sample_live: None,
                background: map.background,
                material: Some(map.material.clone()),
            }
        }

        fn joint(&self, map: &MapMeasurement<'_>, calibration: &Calibration) -> CountsFit {
            let (_, height, width) = self.counts[0].dim();
            let regions = (0..height / 2)
                .flat_map(|i| (0..width / 2).map(move |j| (i, j)))
                .map(|patch| self.region(map, patch))
                .chain(map.empty.iter().any(|&e| e).then(|| Region {
                    open_counts: self.summed(0, height - 1..height, 0..width),
                    sample_counts: self.summed(1, height - 1..height, 0..width),
                    open_live: None,
                    sample_live: None,
                    background: map.empty_background,
                    material: None,
                }))
                .collect();
            fit_counts(
                &Measurement {
                    time_edges_us: map.time_edges_us.clone(),
                    charge_ratio: map.charge_ratio,
                    normalization: map.normalization,
                    regions,
                },
                calibration,
            )
            .expect("joint fit")
        }
    }

    fn pulls(
        result: &CountsMap,
        truths: &[([f64; 2], f64)],
        at: &[((usize, usize), usize)],
    ) -> Vec<f64> {
        let shared = result.shared_covariance.as_ref().expect("covariance");
        let mut pulls = vec![
            (result.normalization - TERMS[0]) / shared.get(0, 0).sqrt(),
            (result.flight_path_m - FLIGHT_PATH_M) / shared.get(2, 2).sqrt(),
        ];
        for &(patch, truth) in at {
            let (densities, t) = truths[truth];
            for (m, n) in densities.into_iter().enumerate() {
                pulls.push((result.densities[m][patch] - n) / result.density_sd[m][patch]);
            }
            pulls.push((result.temperature_k[patch] - t) / result.temperature_sd_k[patch]);
        }
        pulls
    }

    #[test]
    fn a_template_fit_counts_would_refuse_is_refused_before_any_patch_is_fitted() {
        let truths = [([THIN, THIN], 300.0)];
        let tiled = tiled(&truths, |_, _| 0, (1, 1));
        let map = tiled.map(Value::Fitted(0.5), [Value::Known(0.0); 3]);
        match fit_map(&map, &tiled.calibration()) {
            Err(PipelineError::InvalidParameter(message)) => {
                assert!(message.starts_with("temperature"), "{message}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_patch_at_a_known_5000_k_is_fitted() {
        let truths = [([THIN, THIN], 5000.0)];
        let tiled = tiled(&truths, |_, _| 0, (1, 1));
        let map = tiled.map(Value::Known(5000.0), [Value::Known(0.0); 3]);
        let result = fit_map(&map, &tiled.calibration()).expect("map");
        assert!(result.converged);
        assert!(result.failed[[0, 0]].is_none(), "{:?}", result.failed);
    }

    #[test]
    fn a_patch_whose_counts_say_nothing_of_its_temperature_is_fitted() {
        let truths = [([THIN, THIN], 300.0), ([0.0, 0.0], 300.0)];
        let mut tiled = tiled(&truths, |_, j| j, (1, 2));
        tiled.counts[1]
            .slice_mut(s![.., 0..2, 2..4])
            .mapv_inplace(|count| (1.1 * count).round());
        let mut map = tiled.map(Value::Fitted(400.0), [Value::Known(0.0); 3]);
        map.normalization = Value::Fitted(TERMS[0]);
        map.background = [Value::Known(0.05), Value::Known(0.0), Value::Known(0.0)];
        for (_, density) in &mut map.material.isotopes {
            *density = Value::Fitted(0.0);
        }
        let result = fit_map(&map, &tiled.calibration()).expect("map");
        assert!(result.converged);
        assert!(result.failed[[0, 1]].is_none(), "{:?}", result.failed);
        assert!((0..2).all(|m| result.densities[m][[0, 1]] == 0.0));
        assert!(result.temperature_sd_k[[0, 1]].is_nan());
    }

    #[test]
    fn a_map_of_scattered_counts_has_the_joint_fits_overdispersion() {
        let truths = [([THIN, THIN], 300.0), ([1.5 * THIN, 0.5 * THIN], 450.0)];
        let mut tiled = tiled(&truths, |_, j| j, (1, 2));
        for (run, counts) in tiled.counts.iter_mut().enumerate() {
            for (k, count) in counts.iter_mut().enumerate() {
                *count = (*count + 3.0 * count.sqrt() * (1.7 * (k + run) as f64).sin())
                    .round()
                    .max(0.0);
            }
        }
        let mut map = tiled.map(Value::Fitted(400.0), [Value::Known(0.0); 3]);
        let joint = tiled.joint(&map, &tiled.calibration());
        map.normalization = Value::Fitted(joint.normalization);
        let calibration = Calibration {
            t0_us: Value::Fitted(joint.t0_us),
            flight_path_m: Value::Fitted(joint.flight_path_m),
            ..tiled.calibration()
        };
        let result = fit_map(&map, &calibration).expect("map");
        assert_eq!(result.steps, 0);
        for (j, region) in joint.regions[..2].iter().enumerate() {
            let [phi, expected] = [
                result.overdispersion[1][[0, j]],
                region.overdispersion[1].unwrap(),
            ];
            assert!((phi / expected - 1.0).abs() <= 2e-4, "{phi} vs {expected}");
        }
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_map_of_sixteen_alike_patches_is_the_joint_fit_on_the_shared_quantities() {
        let side = 4;
        let truths = [([THIN, THIN], 300.0)];
        let tiled = tiled(&truths, |_, _| 0, (side, side));
        let map = tiled.map(Value::Fitted(400.0), [Value::Known(0.0); 3]);
        let calibration = tiled.calibration();
        let result = fit_map(&map, &calibration).expect("map");
        let joint = tiled.joint(&map, &calibration);
        assert!(result.converged && joint.converged);
        let shared = result.shared_covariance.as_ref().expect("covariance");
        let gaps = [
            (result.t0_us - joint.t0_us) / shared.get(1, 1).sqrt(),
            (result.flight_path_m - joint.flight_path_m) / shared.get(2, 2).sqrt(),
        ];
        assert!(gaps.iter().all(|g| g.abs() <= 2.5e-4), "{gaps:?}");
    }

    #[test]
    fn a_quantity_moving_with_an_undetermined_shared_one_is_undetermined() {
        let truths = [([0.0, 0.0], 300.0)];
        let tiled = tiled(&truths, |_, _| 0, (1, 2));
        let none = Array2::from_elem(tiled.empty.dim(), false);
        let mut map = tiled.map(Value::Known(300.0), [Value::Known(0.0); 3]);
        for (_, density) in &mut map.material.isotopes {
            *density = Value::Known(0.0);
        }
        map.empty = none.view();
        let joint = tiled.joint(&map, &tiled.calibration());
        let result = fit_map(&map, &tiled.calibration()).expect("map");
        let joint_covariance = joint.covariance.as_ref().expect("covariance");
        assert!((0..3).all(|i| joint_covariance.get(i, i).is_nan()));
        assert!(result.converged);
        let shared = result.shared_covariance.as_ref().expect("covariance");
        assert!(shared.get(0, 0).is_nan());
        for patch in [[0, 0], [0, 1]] {
            let covariance = result.covariance[patch].as_ref().expect("covariance");
            assert!(covariance.get(0, 0).is_nan(), "{covariance:?}");
        }
    }

    #[test]
    fn densities_the_counts_cannot_tell_apart_are_left_undetermined() {
        let first = hafnium_like(20.0);
        let mut twin = first.clone();
        twin.za += 1;
        let truths = [([THIN, THIN], 300.0)];
        let tiled = tiled_from([first, twin], 0.05, &truths, |_, _| 0, (1, 1));
        let mut map = tiled.map(Value::Fitted(400.0), [Value::Known(0.0); 3]);
        map.normalization = Value::Fitted(TERMS[0]);
        let result = fit_map(&map, &tiled.calibration()).expect("map");
        assert!(result.converged);
        assert!(result.failed[[0, 0]].is_none(), "{:?}", result.failed);
        assert!((0..2).all(|m| result.density_sd[m][[0, 0]].is_nan()));
        assert!(result.temperature_sd_k[[0, 0]].is_finite());
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_patch_failing_at_the_start_rejoins_and_one_failing_on_the_way_is_left_out() {
        let truths = [([THIN, THIN], 300.0), ([THIN, THIN], 5100.0)];
        let tiled = tiled(&truths, |_, j| j, (1, 2));
        let mut map = tiled.map(Value::Fitted(400.0), [Value::Known(0.0); 3]);
        map.normalization = Value::Fitted(0.8 * TERMS[0]);
        let alone = fit_counts(
            &Measurement {
                time_edges_us: map.time_edges_us.clone(),
                charge_ratio: map.charge_ratio,
                normalization: Value::Known(0.8 * TERMS[0]),
                regions: vec![tiled.region(&map, (0, 0))],
            },
            &Calibration {
                t0_us: Value::Known(T0_US),
                flight_path_m: Value::Known(FLIGHT_PATH_M),
                ..tiled.calibration()
            },
        );
        assert!(alone.map_or(true, |fit| !fit.converged));
        let result = fit_map(&map, &tiled.calibration()).expect("map");
        assert!(result.converged);
        assert!(result.failed[[0, 0]].is_none(), "{:?}", result.failed);
        let reason = result.failed[[0, 1]].as_deref().expect("failed");
        assert!(reason.contains("ended at 1 K or 5000 K"), "{reason}");
        let pull = (result.temperature_k[[0, 0]] - 300.0) / result.temperature_sd_k[[0, 0]];
        assert!(pull.abs() <= 0.01, "{pull}");
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_map_of_sixty_four_patches_recovers_each_one() {
        let truths = [
            ([THIN, THIN], 300.0),
            ([1.5 * THIN, 0.5 * THIN], 450.0),
            ([0.7 * THIN, 2.0 * THIN], 350.0),
            ([THIN, THIN], 600.0),
        ];
        let pick = |i: usize, j: usize| (i + j) % 4;
        let tiled = tiled(&truths, pick, (8, 8));
        let map = tiled.map(Value::Fitted(400.0), [Value::Known(0.0); 3]);
        let result = fit_map(&map, &tiled.calibration()).expect("map");
        assert!(result.converged);
        assert!(
            result.trusted.slice(s![..8, ..]).iter().all(|&t| t),
            "{:?}",
            result.trusted
        );
        let at: Vec<((usize, usize), usize)> = (0..8)
            .flat_map(|i| (0..8).map(move |j| ((i, j), pick(i, j))))
            .collect();
        let pulls = pulls(&result, &truths, &at);
        assert!(pulls.iter().all(|p| p.abs() <= BOUND.sqrt()), "{pulls:?}");
    }

    #[test]
    #[ignore = "slow; runs nightly"]
    fn a_patch_outside_its_temperature_bounds_leaves_the_rest_of_the_map_converged() {
        let truths = [([THIN, THIN], 300.0), ([THIN, 2.0 * THIN], 2500.0)];
        let tiled = tiled(&truths, |_, j| j, (1, 2));
        let bounded = Value::Within {
            start: 400.0,
            lower: 100.0,
            upper: 2000.0,
        };
        let map = tiled.map(
            bounded,
            [Value::Known(0.0), Value::Fitted(0.0), Value::Known(0.0)],
        );
        let result = fit_map(&map, &tiled.calibration()).expect("map");
        assert!(result.converged);
        assert_eq!(result.trusted.row(0).to_vec(), [true, false]);
        assert_eq!(result.temperature_k[[0, 1]], 2000.0);
        let pulls = pulls(&result, &truths, &[((0, 0), 0)]);
        assert!(pulls.iter().all(|p| p.abs() <= 1.0), "{pulls:?}");
    }
}
