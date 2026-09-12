use std::sync::Arc;

use nereids_physics::counts_response::{
    CountsResponseError, DetectorBinResponseMatrix, TwoArmCounts, add_count_backgrounds,
    two_arm_count_response,
};
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
    // Both pulses lie fully inside the acquisition window: the quantified
    // window-loss report must be (numerically) zero.
    assert!(
        got.open_beam_window_loss.abs() < 2.0e-11,
        "open window loss: {}",
        got.open_beam_window_loss
    );
    assert!(
        got.sample_window_loss.abs() < 2.0e-11,
        "sample window loss: {}",
        got.sample_window_loss
    );
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
    // Half of each pulse falls before the window start; the loss is reported
    // in expected counts per arm, not folded back into the window.
    assert!(
        (got.open_beam_window_loss - 50.0).abs() < 1.0e-12,
        "open window loss: {}",
        got.open_beam_window_loss
    );
    assert!(
        (got.sample_window_loss - 20.0).abs() < 1.0e-12,
        "sample window loss: {}",
        got.sample_window_loss
    );
}

// NOTE: this is a ROUTING and fluence/transmission-scaling check, not an
// independent IC oracle — `expected_probability` comes from the same
// `detector_bin_probabilities` law the operator dispatches to.  Independent
// verification of the IC probability law itself (closed forms, Simpson
// integrals) lives in `ic_causal_response.rs`.
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
        &ResolutionFunction::IkedaCarpenter(Arc::clone(&ic)),
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
    // The reported window loss must equal the fluence-weighted probability
    // mass the direct IC evaluation leaves outside the same edges.
    let outside = 1.0 - expected_probability.iter().sum::<f64>();
    assert!(
        (got.open_beam_window_loss - 80.0 * outside).abs() < 2.0e-12,
        "open window loss {} vs direct {}",
        got.open_beam_window_loss,
        80.0 * outside
    );
    assert!(
        (got.sample_window_loss - 40.0 * outside).abs() < 2.0e-12,
        "sample window loss {} vs direct {}",
        got.sample_window_loss,
        40.0 * outside
    );
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

#[test]
fn compact_response_keeps_every_nonzero_and_reconstructs_interior_zeros() {
    let response = triangle_response();
    let arrival_0 = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
    let arrival_1 = arrival_0 + 1.0;
    let energy_1 = (TOF_FACTOR * 25.0 / arrival_1).powi(2);
    let matrix = DetectorBinResponseMatrix::new(
        &[25.0, energy_1],
        &[arrival_0 - 1.0, arrival_0, arrival_0 + 1.0, arrival_0 + 2.0],
        0.0,
        &response,
    )
    .expect("valid compact response");

    assert_eq!(matrix.nnz(), 4);
    assert_eq!(
        matrix.storage_bytes(),
        3 * std::mem::size_of::<usize>()
            + 4 * std::mem::size_of::<u32>()
            + 4 * std::mem::size_of::<f64>()
    );
    assert_eq!(
        matrix.row_entries(0).collect::<Vec<_>>(),
        [(0, 0.5), (1, 0.5)]
    );
    assert_eq!(
        matrix.row_entries(1).collect::<Vec<_>>(),
        [(1, 0.5), (2, 0.5)]
    );
    assert_eq!(matrix.probability(0, 2), 0.0);
    assert_eq!(matrix.probability(1, 0), 0.0);
    assert_eq!(matrix.probability(0, 0), 0.5);

    let got = matrix
        .apply(&[100.0, 200.0], &[0.2, 0.8])
        .expect("valid compact response application");
    assert_eq!(got.open_beam, [50.0, 150.0, 100.0]);
    assert_eq!(got.sample, [10.0, 90.0, 80.0]);
    // Both pulses lie fully inside the window: the matrix must report the
    // same (numerically zero) window loss as the streaming operator.
    assert!(got.open_beam_window_loss.abs() < 2.0e-11);
    assert!(got.sample_window_loss.abs() < 2.0e-11);
}

#[test]
fn compact_response_reports_window_loss_like_the_streaming_operator() {
    let response = triangle_response();
    let arrival = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
    // Window starts at the pulse mode: half of the triangle falls outside.
    let edges = [arrival, arrival + 1.0];

    let matrix = DetectorBinResponseMatrix::new(&[25.0], &edges, 0.0, &response)
        .expect("valid truncated compact response");
    let from_matrix = matrix
        .apply(&[100.0], &[0.4])
        .expect("valid truncated application");
    let from_operator = two_arm_count_response(&[25.0], &[100.0], &[0.4], &edges, 0.0, &response)
        .expect("valid truncated streaming response");

    assert_eq!(from_matrix, from_operator);
    assert!((from_matrix.open_beam_window_loss - 50.0).abs() < 1.0e-12);
    assert!((from_matrix.sample_window_loss - 20.0).abs() < 1.0e-12);
}

fn signal_with_loss(open_loss: f64, sample_loss: f64) -> TwoArmCounts {
    TwoArmCounts {
        open_beam: vec![50.0, 150.0, 100.0],
        sample: vec![10.0, 90.0, 80.0],
        open_beam_window_loss: open_loss,
        sample_window_loss: sample_loss,
    }
}

#[test]
fn count_background_is_added_after_response_and_returned_separately() {
    let prediction = add_count_backgrounds(
        signal_with_loss(0.0, 0.0),
        &[2.0, 3.0, 4.0],
        &[5.0, 7.0, 11.0],
    )
    .expect("valid detector-bin backgrounds");

    assert_eq!(prediction.open_beam.neutron_signal, [50.0, 150.0, 100.0]);
    assert_eq!(prediction.open_beam.background, [2.0, 3.0, 4.0]);
    assert_eq!(prediction.open_beam.total, [52.0, 153.0, 104.0]);
    assert_eq!(prediction.sample.neutron_signal, [10.0, 90.0, 80.0]);
    assert_eq!(prediction.sample.background, [5.0, 7.0, 11.0]);
    assert_eq!(prediction.sample.total, [15.0, 97.0, 91.0]);
}

/// Pipeline-map R5·7 requires the acquisition-window loss to stay disclosed.
/// Forming the total expectation is precisely where that report is at risk of
/// being dropped, so it must survive into the prediction untouched — and it
/// must not be inflated by a background that is already inside the window.
#[test]
fn count_background_carries_window_loss_through_untouched() {
    let prediction = add_count_backgrounds(
        signal_with_loss(50.0, 20.0),
        &[2.0, 3.0, 4.0],
        &[5.0, 7.0, 11.0],
    )
    .expect("valid detector-bin backgrounds");

    assert_eq!(prediction.open_beam.window_loss, 50.0);
    assert_eq!(prediction.sample.window_loss, 20.0);
}

#[test]
fn count_background_rejects_shape_mismatch_and_negative_counts() {
    let signal = signal_with_loss(0.0, 0.0);
    assert!(matches!(
        add_count_backgrounds(signal.clone(), &[1.0], &[2.0, 3.0, 4.0]),
        Err(CountsResponseError::DetectorBinCountMismatch { .. })
    ));
    assert!(matches!(
        add_count_backgrounds(signal.clone(), &[1.0, 2.0, 3.0], &[0.0, -1.0, 0.0]),
        Err(CountsResponseError::InvalidExpectedCount {
            field: "sample_background_counts",
            index: 1,
            ..
        })
    ));
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(matches!(
            add_count_backgrounds(signal.clone(), &[1.0, bad, 3.0], &[1.0, 2.0, 3.0]),
            Err(CountsResponseError::InvalidExpectedCount {
                field: "open_background_counts",
                index: 1,
                ..
            })
        ));
    }
}

/// A non-finite window loss would otherwise ride into the prediction without
/// touching any per-bin value, since it never enters the totals.
#[test]
fn count_background_rejects_non_finite_window_loss() {
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(matches!(
            add_count_backgrounds(
                signal_with_loss(bad, 0.0),
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0, 3.0]
            ),
            Err(CountsResponseError::InvalidExpectedCount {
                field: "open_beam_window_loss",
                ..
            })
        ));
    }
}
