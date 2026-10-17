use nereids_core::constants::BOLTZMANN_EV_PER_K;
use nereids_core::types::Isotope;
use nereids_endf::resonance::test_support::synthetic_isotope;
use nereids_endf::resonance::{ResonanceData, SmoothCrossSections, Tab1};
use nereids_endf::sammy::{parse_sammy_inp, parse_sammy_par, sammy_to_resonance_data};
use nereids_physics::continuous_doppler::Channel;
use nereids_physics::lattice_doppler::{LatticeError, PhononSpectrum, broaden_in_lattice};
use nereids_physics::reich_moore::CrossSectionPlan;

const KELVIN: f64 = 300.0;
const AWR: f64 = 179.4;
const STEP_EV: f64 = 2.5e-4;
const POINTS: usize = 81;

/// A spectrum rising as ω² and falling to zero at 20 meV.
fn tabulated() -> Vec<f64> {
    (0..POINTS)
        .map(|i| {
            let x = i as f64 / (POINTS - 1) as f64;
            x * x * (1.0 - x.powi(4))
        })
        .collect()
}

fn spectrum() -> PhononSpectrum {
    PhononSpectrum::new(STEP_EV, tabulated()).expect("spectrum")
}

/// `∫ρ(ω)·f(ω) dω` over the spectrum as [`PhononSpectrum`] defines it,
/// by Simpson's rule on each segment.
fn over_spectrum(f: impl Fn(f64) -> f64) -> f64 {
    over_spectrum_in(200, f)
}

fn over_spectrum_in(n: usize, f: impl Fn(f64) -> f64) -> f64 {
    let rho = tabulated();
    let density = |omega: f64| {
        let i = ((omega / STEP_EV) as usize).min(POINTS - 2);
        if i == 0 {
            rho[1] * (omega / STEP_EV).powi(2)
        } else {
            rho[i] + (rho[i + 1] - rho[i]) * (omega / STEP_EV - i as f64)
        }
    };
    let simpson = |g: &dyn Fn(f64) -> f64| -> f64 {
        let h = STEP_EV / n as f64;
        (0..POINTS - 1)
            .map(|i| {
                let a = i as f64 * STEP_EV;
                (0..n)
                    .map(|k| {
                        let x = a + k as f64 * h;
                        h / 6.0 * (g(x) + 4.0 * g(x + 0.5 * h) + g(x + h))
                    })
                    .sum::<f64>()
            })
            .sum()
    };
    let weighted = |w: f64| {
        let w = w.max(1e-12);
        density(w) * f(w)
    };
    simpson(&weighted) / simpson(&|w| density(w))
}

fn power(exponent: f64) -> ResonanceData {
    let (x0, x1) = (1e-3, 1e3);
    let table = Tab1 {
        boundaries: vec![2],
        interp_codes: vec![if exponent == 0.0 { 2 } else { 5 }],
        points: vec![(x0, x0.powf(exponent)), (x1, x1.powf(exponent))],
    };
    ResonanceData {
        isotope: Isotope::new(73, 181).expect("isotope"),
        za: 73181,
        awr: AWR,
        smooth: SmoothCrossSections {
            total: Some(table),
            ..SmoothCrossSections::default()
        },
        ranges: Vec::new(),
    }
}

#[test]
fn the_lattice_keeps_the_kernel_s_mass_mean_variance_and_skew() {
    let kt = BOLTZMANN_EV_PER_K * KELVIN;
    let coth = |w: f64| 1.0 / (w / (2.0 * kt)).tanh();
    let kt_eff = 0.5 * over_spectrum(|w| w * coth(w));
    let kt_eff_slope = 0.5
        * over_spectrum(|w| {
            let x = w / (2.0 * kt);
            w * x / (KELVIN * x.sinh().powi(2))
        });
    let skew = over_spectrum(|w| w * w);
    let energies = [8.0, 15.0, 30.0, 45.0];
    let broadened = |exponent: f64| {
        broaden_in_lattice(
            &energies,
            &power(exponent),
            KELVIN,
            &spectrum(),
            Channel::Total,
        )
        .expect("broadened")
    };
    let [one, first, second, third] = [0.0, 1.0, 2.0, 3.0].map(broadened);
    for (k, &e) in energies.iter().enumerate() {
        let r = e / (AWR + 1.0);
        let variance = 2.0 * r * kt_eff;
        assert!(
            (one.values[k] - 1.0).abs() <= 1e-10,
            "{e}: {}",
            one.values[k]
        );
        assert!(one.derivatives[k].abs() <= 1e-12, "{e}");
        assert!(
            (first.values[k] - e).abs() <= 1e-10 * e,
            "{e}: {}",
            first.values[k]
        );
        let spread = second.values[k] - e * e;
        assert!(
            (spread - variance).abs() <= 1e-8 * variance + 1e-14 * e * e,
            "{e}: {spread} {variance}"
        );
        let slope = second.derivatives[k];
        let expected = 2.0 * r * kt_eff_slope;
        assert!(
            (slope - expected).abs() <= 1e-8 * expected + 1e-14 * e * e,
            "{e}: {slope} {expected}"
        );
        let skewed = third.values[k] - e.powi(3) - 3.0 * e * variance;
        assert!(
            (skewed + r * skew).abs() <= 1e-5 * r * skew + 1e-13 * e.powi(3),
            "{e}: {skewed} {}",
            r * skew
        );
    }
}

fn simpson_weight(j: usize, last: usize, h: f64) -> f64 {
    let w = if j == 0 || j == last {
        1.0
    } else if j % 2 == 1 {
        4.0
    } else {
        2.0
    };
    w * h / 3.0
}

#[test]
fn a_resonance_broadened_in_the_lattice_is_the_direct_fourier_integral() {
    let isotope = synthetic_isotope(73, 181, 20.0, 0.01, 0.06);
    let plan = CrossSectionPlan::new(&isotope);
    let kt = BOLTZMANN_EV_PER_K * KELVIN;
    let coth = |w: f64| 1.0 / (w / (2.0 * kt)).tanh();
    let kt_eff = 0.5 * over_spectrum(|w| w * coth(w));
    let g0 = over_spectrum_in(40, |w| coth(w) / w);
    let energies = [19.8, 19.95, 20.0, 20.1, 20.4];
    let lattice = broaden_in_lattice(&energies, &isotope, KELVIN, &spectrum(), Channel::Total)
        .expect("broadened");
    for (k, &e) in energies.iter().enumerate() {
        let r = e / (isotope.awr + 1.0);
        assert!((-r * g0).exp() < 1e-20);
        let (end, dt) = ((40.0 / (r * kt_eff)).sqrt(), 0.04);
        let steps = 2 * (end / (2.0 * dt)).ceil() as usize;
        let psi: Vec<(f64, f64, f64, f64)> = (0..=steps)
            .map(|j| {
                let t = j as f64 * dt;
                let re = over_spectrum_in(40, |w| coth(w) * (w * t).cos() / w) - g0;
                let im = -over_spectrum_in(40, |w| (w * t).sin() / w);
                let magnitude = (r * re).exp();
                let weight = simpson_weight(j, steps, dt);
                (
                    t,
                    weight,
                    magnitude * (r * im).cos(),
                    magnitude * (r * im).sin(),
                )
            })
            .collect();
        let kernel = |epsilon: f64| -> f64 {
            psi.iter()
                .map(|&(t, w, re, im)| {
                    let (sin, cos) = (epsilon * t).sin_cos();
                    w * (cos * re - sin * im)
                })
                .sum::<f64>()
                / std::f64::consts::PI
        };
        let reach = 10.0 * (2.0 * r * kt_eff).sqrt();
        let (h, nodes) = (4e-4, 2 * (reach / 4e-4).ceil() as usize);
        let direct: f64 = (0..=nodes)
            .map(|m| {
                let epsilon = r - reach + m as f64 * h;
                simpson_weight(m, nodes, h)
                    * kernel(epsilon)
                    * plan.evaluate_one(e + r - epsilon).total
            })
            .sum();
        let value = lattice.values[k];
        assert!(
            (value - direct).abs() <= 1e-6 * direct,
            "{e}: {value} {direct}"
        );
    }
}

#[test]
fn the_temperature_slope_is_that_of_the_broadened_cross_section() {
    let isotope = synthetic_isotope(73, 181, 20.0, 0.01, 0.06);
    let energies = [19.7, 19.9, 20.0, 20.05, 20.3];
    let at = |kelvin: f64| {
        broaden_in_lattice(&energies, &isotope, kelvin, &spectrum(), Channel::Capture)
            .expect("broadened")
    };
    let (here, step) = (at(KELVIN), 0.05);
    let (up, down) = (at(KELVIN + step), at(KELVIN - step));
    for (k, energy) in energies.iter().enumerate() {
        let difference = (up.values[k] - down.values[k]) / (2.0 * step);
        let slope = here.derivatives[k];
        assert!(
            (slope - difference).abs() <= 1e-6 * slope.abs().max(1e-3 * here.values[k]),
            "{energy}: {slope} {difference}"
        );
    }
}

#[test]
fn spectra_temperatures_and_reaches_below_zero_are_refused() {
    for (step, density) in [
        (0.0, vec![0.0, 1.0]),
        (1e-3, vec![0.5, 1.0]),
        (1e-3, vec![0.0, -1.0, 1.0]),
        (1e-3, vec![0.0, 0.0]),
        (1e-3, vec![0.0]),
    ] {
        assert!(matches!(
            PhononSpectrum::new(step, density),
            Err(LatticeError::InvalidSpectrum(_))
        ));
    }
    assert!(PhononSpectrum::debye(0.0, 100).is_err());
    let isotope = synthetic_isotope(73, 181, 20.0, 0.01, 0.06);
    let broaden = |energies: &[f64], kelvin: f64| {
        broaden_in_lattice(energies, &isotope, kelvin, &spectrum(), Channel::Total)
    };
    assert!(matches!(
        broaden(&[20.0], 0.0),
        Err(LatticeError::InvalidTemperature(_))
    ));
    assert!(matches!(
        broaden(&[0.05, 20.0], KELVIN),
        Err(LatticeError::BelowZero(_))
    ));
    assert!(matches!(broaden(&[], KELVIN), Err(LatticeError::Grid(_))));
}

#[test]
fn a_resonance_broadened_in_the_lattice_matches_sammy_s_crystal_lattice_model() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/sammy-clm");
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).expect(name);
    let inp = parse_sammy_inp(&read("clm.inp")).expect("inp");
    assert_eq!(inp.temperature_k, 293.16);
    let data = sammy_to_resonance_data(&inp, &parse_sammy_par(&read("u238.par")).expect("par"))
        .expect("resonances");
    let spectrum_file = read("taper.clm");
    let mut lines = spectrum_file
        .lines()
        .skip_while(|l| !l.starts_with("CONTINUOUS"))
        .skip(1);
    let header: Vec<f64> = lines
        .next()
        .expect("step")
        .split_whitespace()
        .map(|x| x.parse().expect("number"))
        .collect();
    let density: Vec<f64> = lines
        .take_while(|l| !l.trim().is_empty())
        .flat_map(|l| {
            l.split_whitespace()
                .map(|x| x.parse::<f64>().expect("number"))
        })
        .collect();
    assert_eq!(density.len(), header[1] as usize);
    let spectrum = PhononSpectrum::new(header[0], density).expect("spectrum");
    let answers: Vec<[f64; 3]> = read("answers.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("number"))
                .collect();
            [f[0], f[1], f[2]]
        })
        .collect();
    let energies: Vec<f64> = answers.iter().map(|a| a[0]).collect();
    let ours = broaden_in_lattice(
        &energies,
        &data,
        inp.temperature_k,
        &spectrum,
        Channel::Total,
    )
    .expect("broadened");
    let plan = CrossSectionPlan::new(&data);
    for (k, &[energy, lattice, bare]) in answers.iter().enumerate() {
        let broadened = ours.values[k] / lattice - 1.0;
        let unbroadened = plan.evaluate_one(energy).total / bare - 1.0;
        assert!(
            (broadened - unbroadened).abs() <= 1e-5,
            "{energy}: {broadened:e} {unbroadened:e}"
        );
    }
}

fn constant_with(table: Vec<(f64, f64)>) -> ResonanceData {
    ResonanceData {
        smooth: SmoothCrossSections {
            total: Some(Tab1 {
                boundaries: vec![table.len()],
                interp_codes: vec![2],
                points: table,
            }),
            ..SmoothCrossSections::default()
        },
        ..power(0.0)
    }
}

#[test]
fn spectra_masses_bands_tables_and_spans_the_kernel_cannot_resolve_are_refused() {
    let scaled = PhononSpectrum::new(1e-3, vec![0.0, 1e308, 1e308]).expect("scaled");
    assert_eq!(
        scaled,
        PhononSpectrum::new(1e-3, vec![0.0, 1.0, 1.0]).expect("unit")
    );
    assert!(PhononSpectrum::new(1e-320, vec![0.0, 1.0]).is_err());
    let one = power(0.0);
    let broaden = |data: &ResonanceData, energies: &[f64], kelvin: f64, s: &PhononSpectrum| {
        broaden_in_lattice(energies, data, kelvin, s, Channel::Total)
    };
    let massless = ResonanceData {
        awr: f64::NAN,
        ..one.clone()
    };
    assert!(matches!(
        broaden(&massless, &[20.0], KELVIN, &spectrum()),
        Err(LatticeError::Grid(_))
    ));
    let mut band = vec![0.0; 301];
    band[200] = 1.0;
    let band = PhononSpectrum::new(1e-4, band).expect("band");
    assert!(matches!(
        broaden(&one, &[2.0], 1.0, &band),
        Err(LatticeError::Unresolved { .. })
    ));
    let jump = constant_with(vec![(1e-3, 1.0), (20.0, 1.0), (20.0, 2.0), (1e3, 2.0)]);
    assert!(matches!(
        broaden(&jump, &[20.1], KELVIN, &spectrum()),
        Err(LatticeError::Jump(_))
    ));
    let narrow = synthetic_isotope(73, 181, 20.0, 5e-6, 5e-6);
    let many: Vec<f64> = (0..2000).map(|k| 19.0 + 1e-3 * k as f64).collect();
    assert!(matches!(
        broaden(&narrow, &many, KELVIN, &spectrum()),
        Err(LatticeError::TooMuchWork(_))
    ));
    let wide = broaden(&one, &[1.0, 100.0], KELVIN, &spectrum()).expect("each in reach");
    for value in wide.values {
        assert!((value - 1.0).abs() <= 1e-7, "{value}");
    }
}

#[test]
fn a_file_3_peak_narrower_than_the_phonon_step_is_broadened_with_its_area() {
    let (centre, half_width, height) = (20.0, 1e-4, 1000.0);
    let peak = constant_with(vec![
        (1e-3, 0.0),
        (centre - half_width, 0.0),
        (centre, height),
        (centre + half_width, 0.0),
        (1e3, 0.0),
    ]);
    let step = 3e-2;
    let energies: Vec<f64> = (0..=40).map(|k| 19.4 + step * k as f64).collect();
    let broadened = broaden_in_lattice(&energies, &peak, KELVIN, &spectrum(), Channel::Total)
        .expect("broadened");
    let area: f64 = broadened
        .values
        .windows(2)
        .map(|w| 0.5 * step * (w[0] + w[1]))
        .sum();
    let expected = height * half_width;
    assert!((area / expected - 1.0).abs() <= 1e-3, "{area} {expected}");
}
