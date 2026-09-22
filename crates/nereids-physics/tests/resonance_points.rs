use std::sync::Arc;

use nereids_endf::resonance::test_support::synthetic_isotope;
use nereids_physics::auxiliary_grid::FRACTN;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    ResolutionFunction, ResolutionParams, TOF_FACTOR, TabulatedResolution,
};
use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};

const FLIGHT_PATH_M: f64 = 25.0;
const E_RESONANCE_EV: f64 = 100.0;
const CHANNEL_US: f64 = 0.5;
const SUBDIVISION: i64 = 64;

fn channels(lo: f64, hi: f64, channel_us: f64) -> Vec<f64> {
    let tof = |e: f64| TOF_FACTOR * FLIGHT_PATH_M / e.sqrt();
    let k_lo = (tof(hi) / channel_us).ceil() as i64;
    let k_hi = (tof(lo) / channel_us).floor() as i64;
    (k_lo..=k_hi)
        .rev()
        .map(|k| (TOF_FACTOR * FLIGHT_PATH_M / (k as f64 * channel_us)).powi(2))
        .collect()
}

fn blurs() -> Vec<(&'static str, ResolutionFunction)> {
    let ikeda_carpenter = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
            beta: EnergyLaw::Const(0.25),
            r: EnergyLaw::Const(0.15),
            burst_sigma_us: None,
            channel_fwhm_us: Some(0.35),
        },
        FLIGHT_PATH_M,
        &SynthesisGrid {
            e_min_ev: 45.0,
            e_max_ev: 220.0,
            n_energies: 32,
            n_tau: 256,
        },
    )
    .expect("valid IC model");
    let offsets: Vec<f64> = (0..81).map(|j| -0.987 + 0.05 * f64::from(j)).collect();
    let weights = offsets
        .iter()
        .map(|&dt| (dt + 1.0) * (-(dt + 1.0) / 0.8).exp())
        .collect();
    let kernel = (offsets, weights);
    let tabulated = TabulatedResolution::from_kernels(
        vec![50.0, 200.0],
        vec![kernel.clone(), kernel],
        FLIGHT_PATH_M,
    )
    .expect("valid table");
    vec![
        (
            "gaussian",
            ResolutionFunction::Gaussian(
                ResolutionParams::new(FLIGHT_PATH_M, 1.0, 0.0, 0.0).expect("valid"),
            ),
        ),
        (
            "gaussian with exponential tail",
            ResolutionFunction::Gaussian(
                ResolutionParams::new(FLIGHT_PATH_M, 1.0, 0.0, 1.0).expect("valid"),
            ),
        ),
        (
            "ikeda-carpenter",
            ResolutionFunction::IkedaCarpenter(Arc::new(ikeda_carpenter)),
        ),
        (
            "tabulated",
            ResolutionFunction::Tabulated(Arc::new(tabulated)),
        ),
    ]
}

fn peak_interpolation_error(spacing: f64) -> f64 {
    let lorentzian = |x: f64| 1.0 / (1.0 + 4.0 * x * x);
    let chord = |x: f64| 1.0 + (lorentzian(spacing) - 1.0) * x / spacing;
    (1..64)
        .map(|k| spacing * f64::from(k) / 64.0)
        .map(|x| lorentzian(x) - chord(x))
        .fold(0.0, f64::max)
}

fn transmission(energies: &[f64], resolution: &ResolutionFunction) -> Vec<f64> {
    let isotope = synthetic_isotope(72, 178, E_RESONANCE_EV, 0.05, 0.06);
    let sample = SampleParams::new(0.0, vec![(isotope, 1.0e-4)]).expect("valid sample");
    forward_model(
        energies,
        &sample,
        Some(&InstrumentParams {
            resolution: resolution.clone(),
        }),
    )
    .expect("forward model")
}

#[test]
fn a_narrow_resonance_is_broadened_as_on_a_dense_grid() {
    let coarse = channels(90.0, 110.0, CHANNEL_US);
    let dense = channels(90.0, 110.0, CHANNEL_US / SUBDIVISION as f64);
    let spacing = coarse[coarse.len() / 2 + 1] - coarse[coarse.len() / 2];
    assert!(
        spacing > 4.0 * (0.05 + 0.06),
        "the channels must be much wider than the resonance for this to test anything, \
         got {spacing} eV"
    );

    let mut failures = Vec::new();
    for (label, resolution) in blurs() {
        let t_coarse = transmission(&coarse, &resolution);
        let t_dense = transmission(&dense, &resolution);
        let mut worst = 0.0_f64;
        let mut dip = 0.0_f64;
        for (i, &e) in coarse.iter().enumerate() {
            let j = dense.partition_point(|&d| d < e * (1.0 - 1.0e-12));
            assert!(
                (dense[j] - e).abs() <= 1.0e-12 * e,
                "{label}: the dense grid must contain every coarse channel"
            );
            worst = worst.max((t_coarse[i] - t_dense[j]).abs());
            dip = dip.max(1.0 - t_dense[j]);
        }
        assert!(
            dip > 0.02,
            "{label}: the broadened dip must be deep enough to measure, got {dip}"
        );
        if worst > peak_interpolation_error(FRACTN) * dip {
            failures.push(format!(
                "{label}: off by {worst:.3e} against a dip of {dip:.3e}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "coarse channels misrepresent a resonance narrower than their spacing: {}",
        failures.join("; ")
    );
}
