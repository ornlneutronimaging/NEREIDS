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

fn blurs() -> Vec<(&'static str, ResolutionFunction)> {
    let ikeda_carpenter = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::Const(0.565),
            beta: EnergyLaw::Const(0.25),
            r: EnergyLaw::Const(0.15),
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
    let triangle = (vec![-1.0, 0.0, 3.0], vec![0.0, 1.0, 0.0]);
    let tabulated = TabulatedResolution::from_kernels(
        vec![1.0, 100.0],
        vec![triangle.clone(), triangle],
        FLIGHT_PATH_M,
    )
    .expect("valid table");
    vec![
        (
            "Ikeda–Carpenter",
            ResolutionFunction::IkedaCarpenter(Arc::new(ikeda_carpenter)),
        ),
        (
            "triangle",
            ResolutionFunction::Tabulated(Arc::new(tabulated)),
        ),
    ]
}

#[test]
fn a_flat_beam_fills_each_bin_by_its_width_on_coarse_points() {
    let edges: Vec<f64> = (400..=470).map(f64::from).collect();
    let mut energies: Vec<f64> = (0..=56)
        .map(|i| energy(480.0 - 5.0 * f64::from(i)))
        .collect();
    energies.sort_by(f64::total_cmp);
    for (label, resolution) in blurs() {
        let weights = BinWeights::new(&energies, &edges, T0_US, &resolution).expect("weights");
        let counts = weights.apply(&vec![1.0; energies.len()]);
        for (k, c) in counts.iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 1e-8,
                "{label}: bin {k} holds {c}, not its width 1 µs"
            );
        }
    }
}

#[test]
fn a_gaussian_resolution_has_no_bin_weights() {
    let gaussian = ResolutionFunction::Gaussian(
        ResolutionParams::new(FLIGHT_PATH_M, 0.5, 0.005, 0.0).expect("valid"),
    );
    let energies = [energy(450.0), energy(440.0)];
    assert!(matches!(
        BinWeights::new(&energies, &[400.0, 401.0], T0_US, &gaussian),
        Err(BinWeightsError::Resolution(_))
    ));
}
