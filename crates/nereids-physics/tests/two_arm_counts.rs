use std::sync::Arc;

use nereids_physics::counts_response::{CountsResponseError, two_arm_count_response};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    ResolutionFunction, ResolutionParams, TOF_FACTOR, TabulatedResolution,
};

fn triangle_response() -> ResolutionFunction {
    ResolutionFunction::Tabulated(Arc::new(
        TabulatedResolution::from_kernels(
            vec![25.0],
            vec![(vec![-1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0])],
            25.0,
        )
        .expect("valid triangle response"),
    ))
}

#[test]
fn two_arm_response_integrates_fluence_and_transmission_before_detector_binning() {
    let response = triangle_response();
    let arrival_0 = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
    let arrival_1 = arrival_0 + 1.0;
    let energy_1 = (TOF_FACTOR * 25.0 / arrival_1).powi(2);
    let edges = [arrival_0 - 1.0, arrival_0, arrival_0 + 1.0, arrival_0 + 2.0];

    let got = two_arm_count_response(
        &[25.0, energy_1],
        &[100.0, 200.0],
        &[0.2, 0.8],
        &edges,
        0.0,
        &response,
    )
    .expect("valid two-arm response");

    // E0 contributes [0.5, 0.5, 0.0]; E1 contributes [0.0, 0.5, 0.5].
    // The open and attenuated sample arms are summed separately.
    let want_open = [50.0, 150.0, 100.0];
    let want_sample = [10.0, 90.0, 80.0];
    for (index, ((&open, &sample), (&expected_open, &expected_sample))) in got
        .open_beam
        .iter()
        .zip(&got.sample)
        .zip(want_open.iter().zip(&want_sample))
        .enumerate()
    {
        assert!((open - expected_open).abs() < 2.0e-11, "open bin {index}");
        assert!(
            (sample - expected_sample).abs() < 2.0e-11,
            "sample bin {index}"
        );
    }
}

#[test]
fn acquisition_window_loss_is_not_renormalized() {
    let response = triangle_response();
    let arrival = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
    let got = two_arm_count_response(
        &[25.0],
        &[100.0],
        &[0.4],
        &[arrival, arrival + 1.0],
        0.0,
        &response,
    )
    .expect("valid truncated response");
    assert!((got.open_beam[0] - 50.0).abs() < 1.0e-12);
    assert!((got.sample[0] - 20.0).abs() < 1.0e-12);
}

#[test]
fn analytical_ic_is_evaluated_directly_in_detector_time() {
    let ic = Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::Const(1.2),
                beta: EnergyLaw::Const(0.2),
                r: EnergyLaw::Const(0.25),
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            25.0,
            &SynthesisGrid {
                e_min_ev: 20.0,
                e_max_ev: 30.0,
                n_energies: 8,
                n_tau: 600,
            },
        )
        .expect("valid IC response"),
    );
    let arrival = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
    let edges = [arrival, arrival + 1.0, arrival + 5.0, arrival + 20.0];
    let expected_probability = ic
        .detector_bin_probabilities(25.0, &edges, 0.0)
        .expect("direct IC probabilities");
    let got = two_arm_count_response(
        &[25.0],
        &[80.0],
        &[0.5],
        &edges,
        0.0,
        &ResolutionFunction::IkedaCarpenter(ic),
    )
    .expect("direct IC two-arm response");

    for (index, (&probability, (&open, &sample))) in expected_probability
        .iter()
        .zip(got.open_beam.iter().zip(&got.sample))
        .enumerate()
    {
        assert!((open - 80.0 * probability).abs() < 2.0e-12, "open {index}");
        assert!(
            (sample - 40.0 * probability).abs() < 2.0e-12,
            "sample {index}"
        );
    }
}

#[test]
fn unsupported_or_unphysical_inputs_fail_clearly() {
    let gaussian = ResolutionFunction::Gaussian(
        ResolutionParams::new(25.0, 1.0, 0.0, 0.0).expect("valid Gaussian parameters"),
    );
    let error = two_arm_count_response(&[25.0], &[100.0], &[0.5], &[100.0, 101.0], 0.0, &gaussian)
        .expect_err("Gaussian detector-time response must fail");
    assert!(error.to_string().contains("Gaussian energy broadening"));

    assert!(matches!(
        two_arm_count_response(
            &[25.0],
            &[100.0],
            &[1.01],
            &[100.0, 101.0],
            0.0,
            &triangle_response(),
        ),
        Err(CountsResponseError::InvalidTransmission { .. })
    ));
}
