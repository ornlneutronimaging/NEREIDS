use nereids_physics::resolution::{TOF_FACTOR, TabulatedResolution};

fn one_asymmetric_pulse() -> TabulatedResolution {
    TabulatedResolution::from_kernels(
        vec![25.0],
        vec![(vec![-1.0, 0.0, 2.0], vec![0.0, 1.0, 0.0])],
        25.0,
    )
    .expect("valid one-energy tabulated response")
}

#[test]
fn tabulated_bins_integrate_piecewise_linear_density_without_window_renormalization() {
    let response = one_asymmetric_pulse();
    let nominal_arrival = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
    let relative_edges = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let detector_edges: Vec<f64> = relative_edges
        .iter()
        .map(|offset| nominal_arrival + offset)
        .collect();
    let got = response
        .detector_bin_probabilities(25.0, &detector_edges, 0.0)
        .expect("valid detector bins");
    let want = [0.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0, 0.0];
    for (index, (&actual, expected)) in got.iter().zip(want).enumerate() {
        assert!(
            (actual - expected).abs() < 2.0e-14,
            "bin {index}: {actual:.16e} != {expected:.16e}"
        );
    }
    assert!((got.iter().sum::<f64>() - 1.0).abs() < 2.0e-14);

    let truncated = response
        .detector_bin_probabilities(25.0, &[nominal_arrival, nominal_arrival + 1.0], 0.0)
        .expect("valid truncated detector window");
    assert_eq!(truncated.len(), 1);
    assert!(
        (truncated[0] - 0.5).abs() < 2.0e-14,
        "partial pulse was renormalized: {}",
        truncated[0]
    );
}

#[test]
fn tabulated_bins_select_the_pulse_by_true_energy() {
    let response = TabulatedResolution::from_kernels(
        vec![4.0, 16.0],
        vec![
            (vec![-2.0, 0.0, 2.0], vec![0.0, 1.0, 0.0]),
            (vec![-1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]),
        ],
        25.0,
    )
    .expect("valid two-energy tabulated response");

    let probability = |energy_ev: f64| {
        let arrival = TOF_FACTOR * 25.0 / energy_ev.sqrt();
        response
            .detector_bin_probabilities(energy_ev, &[arrival - 1.0, arrival + 1.0], 0.0)
            .expect("valid detector bin")[0]
    };
    assert!((probability(4.0) - 0.75).abs() < 2.0e-14);
    assert!((probability(16.0) - 1.0).abs() < 2.0e-14);
}

#[test]
fn tabulated_bins_reject_invalid_physical_inputs() {
    let response = one_asymmetric_pulse();
    assert!(
        response
            .detector_bin_probabilities(0.0, &[1.0, 2.0], 0.0)
            .is_err()
    );
    assert!(
        response
            .detector_bin_probabilities(25.0, &[2.0, 1.0], 0.0)
            .is_err()
    );
    assert!(
        response
            .detector_bin_probabilities(25.0, &[1.0, f64::NAN], 0.0)
            .is_err()
    );
    assert!(
        response
            .detector_bin_probabilities(25.0, &[1.0, 2.0], f64::INFINITY)
            .is_err()
    );
}

#[test]
fn tabulated_bins_support_delta_kernel_and_reject_invalid_flight_path() {
    let response =
        TabulatedResolution::from_kernels(vec![25.0], vec![(vec![0.0], vec![1.0])], 25.0)
            .expect("valid one-point delta response");
    let arrival = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
    let got = response
        .detector_bin_probabilities(25.0, &[arrival - 1.0, arrival, arrival + 1.0], 0.0)
        .expect("delta response must be integrable");
    assert_eq!(got, vec![0.0, 1.0]);

    for bad_path in [0.0, -25.0, f64::NAN, f64::INFINITY] {
        assert!(
            TabulatedResolution::from_kernels(
                vec![25.0],
                vec![(vec![-1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0])],
                bad_path,
            )
            .is_err(),
            "invalid flight path {bad_path} was accepted"
        );
    }
}
