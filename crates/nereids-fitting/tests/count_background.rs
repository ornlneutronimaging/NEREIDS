use nereids_fitting::count_background::{
    TwoArmBackgroundTemplate, fit_two_arm_background_templates,
};
use nereids_fitting::poisson::PoissonConfig;
use nereids_physics::counts_response::{TwoArmCounts, add_count_backgrounds};

fn signals() -> TwoArmCounts {
    TwoArmCounts {
        open_beam: vec![1000.0, 900.0, 800.0, 700.0, 600.0],
        sample: vec![600.0, 540.0, 480.0, 420.0, 360.0],
    }
}

fn shaped_template(name: &str) -> TwoArmBackgroundTemplate {
    TwoArmBackgroundTemplate {
        name: name.into(),
        open_beam: vec![0.2, 0.5, 1.0, 2.0, 4.0],
        sample: vec![4.0, 2.0, 1.0, 0.5, 0.2],
    }
}

fn synthetic_observation(
    signal: TwoArmCounts,
    template: &TwoArmBackgroundTemplate,
    amplitude: f64,
) -> TwoArmCounts {
    let open: Vec<f64> = template
        .open_beam
        .iter()
        .map(|value| amplitude * value)
        .collect();
    let sample: Vec<f64> = template
        .sample
        .iter()
        .map(|value| amplitude * value)
        .collect();
    let prediction =
        add_count_backgrounds(signal, &open, &sample).expect("valid synthetic observation");
    TwoArmCounts {
        open_beam: prediction.open_beam.total,
        sample: prediction.sample.total,
    }
}

#[test]
fn recovers_fixed_independent_background_template() {
    let template = shaped_template("blocked_beam_reference");
    let true_amplitude = 25.0;
    let signal = signals();
    let observed = synthetic_observation(signal.clone(), &template, true_amplitude);

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        std::slice::from_ref(&template),
        &[1.0],
        &PoissonConfig::default(),
    )
    .expect("correct template fit");

    assert!(result.converged);
    assert!(result.amplitudes_identifiable);
    assert!((result.amplitudes[0] - true_amplitude).abs() < 1.0e-6);
    assert!(result.poisson_deviance < 1.0e-10);
    assert!((result.prediction.open_beam.background[4] - 100.0).abs() < 1.0e-6);
    assert!((result.prediction.sample.background[0] - 100.0).abs() < 1.0e-6);
}

#[test]
fn wrong_background_shape_cannot_silently_match_synthetic_counts() {
    let true_template = shaped_template("true_blocked_beam_reference");
    let signal = signals();
    let observed = synthetic_observation(signal.clone(), &true_template, 100.0);
    let wrong_template = TwoArmBackgroundTemplate {
        name: "wrong_flat_reference".into(),
        open_beam: vec![1.0; 5],
        sample: vec![1.0; 5],
    };

    let wrong = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &[wrong_template],
        &[1.0],
        &PoissonConfig::default(),
    )
    .expect("wrong shape still has a defined best fit");

    assert!(wrong.converged);
    assert!(
        wrong.deviance_per_dof > 5.0,
        "wrong template unexpectedly passed: D/dof = {}",
        wrong.deviance_per_dof
    );
}

#[test]
fn invalid_templates_fail_before_optimization() {
    let observed = signals();
    let signal = signals();
    let invalid = TwoArmBackgroundTemplate {
        name: "negative".into(),
        open_beam: vec![1.0, 1.0, -1.0, 1.0, 1.0],
        sample: vec![1.0; 5],
    };
    let error = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &[invalid],
        &[1.0],
        &PoissonConfig::default(),
    )
    .expect_err("negative expected counts must fail");
    assert!(error.to_string().contains("must be finite and >= 0"));
}

#[test]
fn template_units_do_not_change_the_physical_fit() {
    let signal = TwoArmCounts {
        open_beam: vec![100.0, 200.0, 300.0],
        sample: vec![80.0, 160.0, 240.0],
    };
    let tiny_units = TwoArmBackgroundTemplate {
        name: "same_reference_in_tiny_units".into(),
        open_beam: vec![1.0e-8; 3],
        sample: vec![1.0e-8; 3],
    };
    let true_amplitude = 1.0e9;
    let observed = synthetic_observation(signal.clone(), &tiny_units, true_amplitude);

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &[tiny_units],
        &[0.0],
        &PoissonConfig::default(),
    )
    .expect("template unit rescaling must not stall the optimizer");

    assert!(result.converged);
    assert!((result.amplitudes[0] / true_amplitude - 1.0).abs() < 1.0e-6);
    assert!(result.deviance_per_dof >= 0.0);
    assert!(result.deviance_per_dof < 1.0e-10);
}

#[test]
fn recovers_several_overlapping_components_from_distant_initial_values() {
    let signal = signals();
    let templates = vec![
        shaped_template("blocked_beam"),
        TwoArmBackgroundTemplate {
            name: "detector_dark".into(),
            open_beam: vec![1.0; 5],
            sample: vec![1.0; 5],
        },
        TwoArmBackgroundTemplate {
            name: "sample_scatter_reference".into(),
            open_beam: vec![4.0, 0.2, 2.0, 0.5, 1.0],
            sample: vec![0.5, 4.0, 0.2, 2.0, 1.0],
        },
    ];
    let truth = [25.0, 10.0, 3.0];
    let mut observed = signal.clone();
    for (&amplitude, template) in truth.iter().zip(&templates) {
        for (count, &basis) in observed.open_beam.iter_mut().zip(&template.open_beam) {
            *count += amplitude * basis;
        }
        for (count, &basis) in observed.sample.iter_mut().zip(&template.sample) {
            *count += amplitude * basis;
        }
    }

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &templates,
        &[1000.0, 0.0, 200.0],
        &PoissonConfig::default(),
    )
    .expect("convex multi-component fit");

    assert!(result.converged);
    for (&actual, &expected) in result.amplitudes.iter().zip(&truth) {
        assert!((actual / expected - 1.0).abs() < 1.0e-5);
    }
    assert!(result.deviance_per_dof < 1.0e-10);
    for (&actual, &expected) in result
        .prediction
        .open_beam
        .total
        .iter()
        .chain(&result.prediction.sample.total)
        .zip(observed.open_beam.iter().chain(&observed.sample))
    {
        assert!((actual - expected).abs() < 2.0e-5);
    }
}

#[test]
fn exact_poisson_fit_recovers_counts_below_the_general_solver_floor() {
    let signal = TwoArmCounts {
        open_beam: vec![0.0],
        sample: vec![0.0],
    };
    let template = TwoArmBackgroundTemplate {
        name: "tiny_count_reference".into(),
        open_beam: vec![1.0],
        sample: vec![1.0],
    };
    let observed = TwoArmCounts {
        open_beam: vec![1.0e-12],
        sample: vec![1.0e-12],
    };

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &[template],
        &[0.0],
        &PoissonConfig::default(),
    )
    .expect("exact count likelihood remains valid below 1e-10 count");

    assert!(result.converged);
    assert!((result.amplitudes[0] / 1.0e-12 - 1.0).abs() < 1.0e-8);
    assert!(result.poisson_deviance < 1.0e-24);
}

#[test]
fn zero_template_bins_do_not_contaminate_other_arm_gradient() {
    let signal = TwoArmCounts {
        open_beam: vec![0.0],
        sample: vec![0.0],
    };
    let template = TwoArmBackgroundTemplate {
        name: "open_only".into(),
        open_beam: vec![1.0],
        sample: vec![0.0],
    };
    let observed = TwoArmCounts {
        open_beam: vec![1.0],
        // This arm cannot be represented by the open-only template. Its
        // large finite value must not create 0 * infinity in the derivative.
        sample: vec![1.0e300],
    };

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &[template],
        &[0.0],
        &PoissonConfig::default(),
    )
    .expect("zero template weights are excluded from that coordinate");

    assert!(result.converged);
    assert!((result.amplitudes[0] - 1.0).abs() < 1.0e-8);
    assert!(result.poisson_deviance.is_infinite());
}

#[test]
fn unrepresentable_caller_amplitude_fails_instead_of_returning_infinity() {
    let signal = TwoArmCounts {
        open_beam: vec![0.0],
        sample: vec![0.0],
    };
    let template = TwoArmBackgroundTemplate {
        name: "unusable_units".into(),
        open_beam: vec![1.0e-320],
        sample: vec![1.0e-320],
    };
    let observed = TwoArmCounts {
        open_beam: vec![1.0],
        sample: vec![1.0],
    };

    let error = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &[template],
        &[0.0],
        &PoissonConfig::default(),
    )
    .expect_err("an infinite caller-unit amplitude must not escape the API");

    assert!(error.to_string().contains("rescale the template counts"));
}

#[test]
fn finite_ratio_of_large_count_sums_does_not_overflow() {
    let signal = TwoArmCounts {
        open_beam: vec![0.0],
        sample: vec![0.0],
    };
    let template = TwoArmBackgroundTemplate {
        name: "large_counts".into(),
        open_beam: vec![1.0],
        sample: vec![1.0],
    };
    let observed = TwoArmCounts {
        open_beam: vec![1.0e308],
        sample: vec![1.0e308],
    };

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &[template],
        &[1.0e308],
        &PoissonConfig::default(),
    )
    .expect("finite count ratio must not overflow through its direct sums");

    assert!(result.converged);
    assert_eq!(result.amplitudes[0], 1.0e308);
    assert_eq!(result.poisson_deviance, 0.0);
}

#[test]
fn unresolved_background_separation_never_reports_zero_uncertainty() {
    let n_bins = 3;
    let signal = TwoArmCounts {
        open_beam: vec![100.0; n_bins],
        sample: vec![100.0; n_bins],
    };
    let nearly_same = 1.0e-11;
    let first = TwoArmBackgroundTemplate {
        name: "first".into(),
        open_beam: vec![1.0; n_bins],
        sample: vec![1.0; n_bins],
    };
    let second = TwoArmBackgroundTemplate {
        name: "nearly_the_same".into(),
        open_beam: vec![1.0 - nearly_same, 1.0, 1.0 + nearly_same],
        sample: vec![1.0 + nearly_same, 1.0, 1.0 - nearly_same],
    };
    let templates = vec![first, second];
    let mut observed = signal.clone();
    for (count, (&a, &b)) in observed
        .open_beam
        .iter_mut()
        .zip(templates[0].open_beam.iter().zip(&templates[1].open_beam))
    {
        *count += 10.0 * a + 20.0 * b;
    }
    for (count, (&a, &b)) in observed
        .sample
        .iter_mut()
        .zip(templates[0].sample.iter().zip(&templates[1].sample))
    {
        *count += 10.0 * a + 20.0 * b;
    }

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &templates,
        &[10.0, 20.0],
        &PoissonConfig::default(),
    )
    .expect("nearly dependent templates still have a prediction");

    assert!(result.converged);
    assert!(result.amplitudes_identifiable);
    if let Some(uncertainties) = result.amplitude_uncertainties {
        assert!(uncertainties.iter().all(|value| *value != 0.0));
    }
}

#[test]
fn exposure_scales_prevent_run_normalization_from_becoming_background() {
    let reference_signal = TwoArmCounts {
        open_beam: vec![100.0, 100.0],
        sample: vec![50.0, 50.0],
    };
    let observed = TwoArmCounts {
        open_beam: vec![100.0, 100.0],
        sample: vec![100.0, 100.0],
    };
    let sample_only = TwoArmBackgroundTemplate {
        name: "sample_only".into(),
        open_beam: vec![0.0, 0.0],
        sample: vec![1.0, 1.0],
    };

    let result = fit_two_arm_background_templates(
        &observed,
        reference_signal,
        1.0,
        2.0,
        &[sample_only],
        &[10.0],
        &PoissonConfig::default(),
    )
    .expect("known exposure factors are part of the count prediction");

    assert!(result.converged);
    assert!(result.amplitudes[0] < 1.0e-10);
    assert_eq!(result.prediction.sample.neutron_signal, vec![100.0, 100.0]);
    assert_eq!(result.poisson_deviance, 0.0);
}

#[test]
fn dependent_templates_are_explicitly_marked_unidentifiable() {
    let signal = TwoArmCounts {
        open_beam: vec![100.0; 3],
        sample: vec![100.0; 3],
    };
    let observed = TwoArmCounts {
        open_beam: vec![120.0; 3],
        sample: vec![120.0; 3],
    };
    let templates = vec![
        TwoArmBackgroundTemplate {
            name: "dark".into(),
            open_beam: vec![1.0; 3],
            sample: vec![1.0; 3],
        },
        TwoArmBackgroundTemplate {
            name: "gamma".into(),
            open_beam: vec![2.0; 3],
            sample: vec![2.0; 3],
        },
    ];

    let result = fit_two_arm_background_templates(
        &observed,
        signal,
        1.0,
        1.0,
        &templates,
        &[0.0, 5.0],
        &PoissonConfig::default(),
    )
    .expect("the total prediction remains defined");

    assert!(result.converged);
    assert!(!result.amplitudes_identifiable);
    assert!(result.amplitude_uncertainties.is_none());
    assert_eq!(result.poisson_deviance, 0.0);
    for value in result.prediction.open_beam.background {
        assert!((value - 20.0).abs() < 1.0e-12);
    }
}
