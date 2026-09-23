use std::sync::Arc;

use nereids_physics::bin_weights::{BinWeights, BinWeightsError};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    ResolutionFunction, ResolutionParams, TOF_FACTOR, TabulatedResolution,
};

const FLIGHT_PATH_M: f64 = 25.0;
const T0_US: f64 = 3.0;

fn energy(flight_time_us: f64) -> f64 {
    (TOF_FACTOR * FLIGHT_PATH_M / flight_time_us).powi(2)
}

const IC_ALPHA: f64 = 0.565;
const IC_BETA: f64 = 0.25;
const IC_R: f64 = 0.15;
const TRIANGLE: [f64; 3] = [-1.0, 0.0, 3.0];

fn flight_time(energy_ev: f64) -> f64 {
    TOF_FACTOR * FLIGHT_PATH_M / energy_ev.sqrt()
}

fn energy_independent_pulses() -> Vec<(&'static str, ResolutionFunction, f64)> {
    let ikeda_carpenter = IkedaCarpenter::new(
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
    .expect("valid IC model");
    let triangle = (TRIANGLE.to_vec(), vec![0.0, 1.0, 0.0]);
    let tabulated = TabulatedResolution::from_kernels(
        vec![1.0, 100.0],
        vec![triangle.clone(), triangle],
        FLIGHT_PATH_M,
    )
    .expect("valid table");
    let single = (vec![0.0], vec![1.0]);
    let single_sample = TabulatedResolution::from_kernels(
        vec![1.0, 100.0],
        vec![single.clone(), single],
        FLIGHT_PATH_M,
    )
    .expect("valid table");
    vec![
        (
            "Ikeda–Carpenter",
            ResolutionFunction::IkedaCarpenter(Arc::new(ikeda_carpenter)),
            3.0 / IC_ALPHA + IC_R / IC_BETA,
        ),
        (
            "triangle",
            ResolutionFunction::Tabulated(Arc::new(tabulated)),
            TRIANGLE.iter().sum::<f64>() / 3.0,
        ),
        (
            "single sample",
            ResolutionFunction::Tabulated(Arc::new(single_sample)),
            0.0,
        ),
    ]
}

#[test]
fn beams_straight_in_flight_time_fill_each_bin_exactly_however_coarse_the_points() {
    let bins: Vec<f64> = (400..=470).map(f64::from).collect();
    let mut coarse: Vec<f64> = (0..=56)
        .map(|i| energy(480.0 - 5.0 * f64::from(i)))
        .collect();
    coarse.sort_by(f64::total_cmp);
    let one_piece = [energy(1000.0), energy(100.0)];
    let narrow_bin = [349.9, 350.1];
    for (label, resolution, mean_delay) in energy_independent_pulses() {
        for (energies, edges) in [(&coarse[..], &bins[..]), (&one_piece[..], &narrow_bin[..])] {
            let weights = BinWeights::new(energies, edges, T0_US, &resolution).expect("weights");
            let u: Vec<f64> = energies.iter().map(|&e| flight_time(e)).collect();
            let flat = weights.apply(&vec![1.0; energies.len()]);
            let linear = weights.apply(&u);
            for (k, bin) in edges.windows(2).enumerate() {
                let width = bin[1] - bin[0];
                let centre = 0.5 * (bin[0] + bin[1]);
                let expected = width * (centre - T0_US - mean_delay);
                assert!(
                    (flat[k] / width - 1.0).abs() < 1e-8,
                    "{label}: bin {bin:?} holds {}, not its width {width} µs",
                    flat[k]
                );
                assert!(
                    (linear[k] / expected - 1.0).abs() < 1e-8,
                    "{label}: bin {bin:?} holds {} for φ = u, not {expected}",
                    linear[k]
                );
            }
        }
    }
}

#[test]
fn a_gaussian_resolution_or_an_empty_bin_has_no_bin_weights() {
    let gaussian = ResolutionFunction::Gaussian(
        ResolutionParams::new(FLIGHT_PATH_M, 0.5, 0.005, 0.0).expect("valid"),
    );
    let energies = [energy(450.0), energy(440.0)];
    assert!(matches!(
        BinWeights::new(&energies, &[400.0, 401.0], T0_US, &gaussian),
        Err(BinWeightsError::Resolution(_))
    ));
    let (_, triangle, _) = energy_independent_pulses().swap_remove(1);
    assert!(matches!(
        BinWeights::new(&energies, &[400.0, 400.0, 401.0], T0_US, &triangle),
        Err(BinWeightsError::InvalidTimeEdges)
    ));
}
