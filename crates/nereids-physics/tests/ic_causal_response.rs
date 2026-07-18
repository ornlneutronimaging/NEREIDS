use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::TOF_FACTOR;

fn gamma3_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0 - (-x).exp() * (1.0 + x + 0.5 * x * x)
    }
}

#[test]
fn prompt_only_bins_match_closed_form_and_keep_physical_time() {
    let alpha = 2.0;
    let flight_path_m = 25.0_f64;
    let true_energy_ev = 25.0_f64;
    let timing_offset_us = 3.25_f64;
    let model = IkedaCarpenter::new(
        IkedaCarpenterParams::constant(alpha, 0.1, 0.0),
        flight_path_m,
        &SynthesisGrid::new(1.0, 100.0),
    )
    .expect("valid prompt-only IC model");

    let nominal_arrival = timing_offset_us + TOF_FACTOR * flight_path_m / true_energy_ev.sqrt();
    let delay_edges = [0.0, 0.5, 1.0, 2.0, 10.0];
    let detector_edges: Vec<f64> = delay_edges
        .iter()
        .map(|delay| nominal_arrival + delay)
        .collect();

    let actual = model
        .detector_bin_probabilities(true_energy_ev, &detector_edges, timing_offset_us)
        .expect("valid detector bins");
    let expected: Vec<f64> = delay_edges
        .windows(2)
        .map(|edge| gamma3_cdf(alpha * edge[1]) - gamma3_cdf(alpha * edge[0]))
        .collect();

    assert_eq!(actual.len(), expected.len());
    for (bin, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 2e-12,
            "bin {bin}: causal IC probability {got:.16e} != closed form {want:.16e}"
        );
    }

    let (delays, weights) = model
        .source_pulse_at(true_energy_ev)
        .expect("valid physical source pulse");
    let peak = weights
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("non-empty pulse");
    let step = delays[1] - delays[0];
    assert!(
        (delays[peak] - 2.0 / alpha).abs() <= step,
        "source pulse peak was moved from physical delay 2/alpha={} us to {} us",
        2.0 / alpha,
        delays[peak]
    );
}

#[test]
fn energy_law_is_evaluated_at_true_energy() {
    let flight_path_m = 25.0;
    let params = IkedaCarpenterParams {
        alpha: EnergyLaw::SqrtE { a0: 1.0, a1: 0.0 },
        beta: EnergyLaw::Const(0.1),
        r: EnergyLaw::Const(0.0),
        burst_sigma_us: None,
        channel_fwhm_us: None,
    };
    let model = IkedaCarpenter::new(params, flight_path_m, &SynthesisGrid::new(1.0, 100.0))
        .expect("valid energy-dependent IC model");

    for true_energy_ev in [1.0_f64, 4.0] {
        let nominal_arrival = TOF_FACTOR * flight_path_m / true_energy_ev.sqrt();
        let edges = [nominal_arrival, nominal_arrival + 1.0];
        let got = model
            .detector_bin_probabilities(true_energy_ev, &edges, 0.0)
            .expect("valid detector bin")[0];
        let alpha = true_energy_ev.sqrt();
        let want = gamma3_cdf(alpha);
        assert!(
            (got - want).abs() < 2e-12,
            "E={true_energy_ev} eV used the wrong pulse: {got:.16e} != {want:.16e}"
        );
    }
}

#[test]
fn moderator_rates_that_follow_neutron_speed_preserve_scaled_pulse_time() {
    let flight_path_m = 25.0;
    let params = IkedaCarpenterParams {
        alpha: EnergyLaw::SqrtE { a0: 2.0, a1: 0.0 },
        beta: EnergyLaw::SqrtE { a0: 0.5, a1: 0.0 },
        r: EnergyLaw::Const(0.3),
        burst_sigma_us: None,
        channel_fwhm_us: None,
    };
    let model = IkedaCarpenter::new(params, flight_path_m, &SynthesisGrid::new(1.0, 4.0))
        .expect("valid speed-scaled IC model");

    let probability_to_scaled_delay = |energy_ev: f64, delay_us: f64| {
        let arrival = TOF_FACTOR * flight_path_m / energy_ev.sqrt();
        model
            .detector_bin_probabilities(energy_ev, &[arrival, arrival + delay_us], 0.0)
            .expect("valid detector bin")[0]
    };

    let at_one_ev = probability_to_scaled_delay(1.0, 2.0);
    let at_four_ev = probability_to_scaled_delay(4.0, 1.0);
    assert!(
        (at_one_ev - at_four_ev).abs() < 2e-12,
        "speed-scaled moderator pulse changed shape: {at_one_ev:.16e} != {at_four_ev:.16e}"
    );
}

#[test]
fn probe_outside_synthesis_range_rejects_nonphysical_beta_law() {
    let model = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::Const(1.0),
            beta: EnergyLaw::SqrtE { a0: -1.0, a1: 2.0 },
            r: EnergyLaw::Const(0.3),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        },
        25.0,
        &SynthesisGrid::new(1.0, 2.0),
    )
    .expect("beta law is physical inside the synthesis range");

    let legacy_error = model
        .kernel_at(9.0)
        .expect_err("legacy kernel probe must reject a negative beta law")
        .to_string();
    assert!(
        legacy_error.contains("beta(9)"),
        "legacy kernel probe hid the invalid beta law behind: {legacy_error}"
    );
    assert!(
        model.source_pulse_at(9.0).is_err(),
        "physical source probe must reject a negative beta law"
    );
}

#[test]
fn invalid_detector_edges_fail_instead_of_being_reordered() {
    let model = IkedaCarpenter::new(
        IkedaCarpenterParams::constant(1.0, 0.1, 0.0),
        25.0,
        &SynthesisGrid::new(1.0, 100.0),
    )
    .expect("valid IC model");

    assert!(
        model
            .detector_bin_probabilities(10.0, &[100.0, 99.0], 0.0)
            .is_err()
    );
    assert!(
        model
            .detector_bin_probabilities(10.0, &[100.0, f64::NAN], 0.0)
            .is_err()
    );
    assert!(
        model
            .detector_bin_probabilities(0.0, &[100.0, 101.0], 0.0)
            .is_err()
    );
}

#[test]
fn equal_rate_storage_limit_matches_gamma4_closed_form() {
    let rate = 1.25_f64;
    let flight_path_m = 25.0_f64;
    let true_energy_ev = 16.0_f64;
    let model = IkedaCarpenter::new(
        IkedaCarpenterParams::constant(rate, rate, 1.0),
        flight_path_m,
        &SynthesisGrid::new(1.0, 100.0),
    )
    .expect("valid equal-rate storage model");
    let nominal_arrival = TOF_FACTOR * flight_path_m / true_energy_ev.sqrt();
    let delay_edges = [0.0, 0.2, 0.8, 2.0, 20.0];
    let detector_edges: Vec<f64> = delay_edges
        .iter()
        .map(|delay| nominal_arrival + delay)
        .collect();

    let got = model
        .detector_bin_probabilities(true_energy_ev, &detector_edges, 0.0)
        .expect("valid detector bins");
    let gamma4_cdf = |delay: f64| {
        let x = rate * delay;
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (-x).exp() * (1.0 + x + x * x / 2.0 + x.powi(3) / 6.0)
        }
    };
    for (index, edge) in delay_edges.windows(2).enumerate() {
        let want = gamma4_cdf(edge[1]) - gamma4_cdf(edge[0]);
        assert!(
            (got[index] - want).abs() < 3e-12,
            "equal-rate storage bin {index}: {} != {want}",
            got[index]
        );
    }
}

#[test]
fn symmetric_sns_pulse_fold_keeps_source_clock_and_preserves_missing_tail_mass() {
    let alpha = 2.0_f64;
    let flight_path_m = 25.0_f64;
    let true_energy_ev = 25.0_f64;
    let params = IkedaCarpenterParams {
        channel_fwhm_us: Some(0.35),
        ..IkedaCarpenterParams::constant(alpha, 0.1, 0.0)
    };
    let model = IkedaCarpenter::new(params, flight_path_m, &SynthesisGrid::new(1.0, 100.0))
        .expect("valid folded IC model");
    let (delays, weights) = model
        .source_pulse_at(true_energy_ev)
        .expect("valid folded source pulse");

    let mut area = 0.0;
    let mut first_moment = 0.0;
    for i in 0..delays.len() - 1 {
        let width = delays[i + 1] - delays[i];
        area += 0.5 * (weights[i] + weights[i + 1]) * width;
        first_moment += width
            * (delays[i] * (2.0 * weights[i] + weights[i + 1])
                + delays[i + 1] * (weights[i] + 2.0 * weights[i + 1]))
            / 6.0;
    }
    let mean = first_moment / area;
    assert!(
        (mean - 3.0 / alpha).abs() < 3e-3,
        "symmetric fold moved the physical source mean from {} us to {mean} us",
        3.0 / alpha
    );

    let nominal_arrival = TOF_FACTOR * flight_path_m / true_energy_ev.sqrt();
    let detector_edges: Vec<f64> = (0..=20)
        .map(|index| {
            nominal_arrival
                + delays[0]
                + (delays[delays.len() - 1] - delays[0]) * index as f64 / 20.0
        })
        .collect();
    let probabilities = model
        .detector_bin_probabilities(true_energy_ev, &detector_edges, 0.0)
        .expect("valid folded detector bins");
    let total: f64 = probabilities.iter().sum();

    // Independent oracle for X + C, where X is the prompt-only Gamma(3,
    // alpha) moderator delay and C is the symmetric triangular channel-time
    // density. Integrating the analytical moderator CDF over C predicts the
    // probability inside these finite detector edges without using the
    // implementation's sampled convolution or bin integration.
    let gamma3_cdf = |delay: f64| {
        let x = alpha * delay;
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (-x).exp() * (1.0 + x + x * x / 2.0)
        }
    };
    let half_base = 0.35_f64;
    let lower = delays[0];
    let upper = delays[delays.len() - 1];
    let integrand = |channel_delay: f64| {
        let channel_density = (1.0 - channel_delay.abs() / half_base) / half_base;
        channel_density * (gamma3_cdf(upper - channel_delay) - gamma3_cdf(lower - channel_delay))
    };
    let intervals = 20_000_usize;
    let step = 2.0 * half_base / intervals as f64;
    let mut oracle = integrand(-half_base) + integrand(half_base);
    for index in 1..intervals {
        let delay = -half_base + index as f64 * step;
        oracle += if index % 2 == 0 { 2.0 } else { 4.0 } * integrand(delay);
    }
    oracle *= step / 3.0;
    assert!(
        (total - oracle).abs() < 2.0e-7,
        "sampled folded pulse has probability {total}, independent analytical integral gives {oracle}"
    );
    assert!(
        oracle < 1.0,
        "finite detector edges must omit a physical tail"
    );
    assert!(probabilities.iter().all(|p| p.is_finite() && *p >= 0.0));
}

#[test]
fn gaussian_fold_preserves_finite_window_tail_mass() {
    let alpha = 100.0_f64;
    let sigma = 1.0_f64;
    let flight_path_m = 25.0_f64;
    let true_energy_ev = 25.0_f64;
    let params = IkedaCarpenterParams {
        burst_sigma_us: Some(sigma),
        ..IkedaCarpenterParams::constant(alpha, 1.0, 0.0)
    };
    let model = IkedaCarpenter::new(
        params,
        flight_path_m,
        &SynthesisGrid {
            e_min_ev: 24.0,
            e_max_ev: 26.0,
            n_energies: 2,
            n_tau: 8,
        },
    )
    .expect("valid Gaussian-folded IC model");
    let (delays, _) = model
        .source_pulse_at(true_energy_ev)
        .expect("valid folded source pulse");
    let lower = delays[0];
    let upper = delays[delays.len() - 1];
    let nominal_arrival = TOF_FACTOR * flight_path_m / true_energy_ev.sqrt();
    let got: f64 = model
        .detector_bin_probabilities(
            true_energy_ev,
            &[nominal_arrival + lower, nominal_arrival + upper],
            0.0,
        )
        .expect("valid folded detector window")
        .iter()
        .sum();

    // Independent oracle: integrate the exact Gamma(3) moderator CDF over a
    // standard normal variable. This does not use the sampled convolution or
    // detector-bin integrator under test.
    let normal_density = |delay: f64| {
        (-0.5 * (delay / sigma).powi(2)).exp() / (sigma * std::f64::consts::TAU.sqrt())
    };
    let moderator_cdf = |delay: f64| gamma3_cdf(alpha * delay);
    let integrand = |gaussian_delay: f64| {
        normal_density(gaussian_delay)
            * (moderator_cdf(upper - gaussian_delay) - moderator_cdf(lower - gaussian_delay))
    };
    let integration_limit = 10.0 * sigma;
    let intervals = 200_000_usize;
    let step = 2.0 * integration_limit / intervals as f64;
    let mut oracle = integrand(-integration_limit) + integrand(integration_limit);
    for index in 1..intervals {
        let delay = -integration_limit + index as f64 * step;
        oracle += if index % 2 == 0 { 2.0 } else { 4.0 } * integrand(delay);
    }
    oracle *= step / 3.0;

    assert!(
        1.0 - oracle > 4.0e-5,
        "oracle window must expose enough missing tail to reject silent renormalization"
    );
    assert!(
        (got - oracle).abs() < 2.0e-5,
        "sampled Gaussian-folded probability {got} disagrees with independent integral {oracle}"
    );
    assert!(got < 1.0, "finite window was silently renormalized to one");
}
