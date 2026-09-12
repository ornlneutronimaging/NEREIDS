use nereids_fitting::count_background::{
    TwoArmBackgroundTemplate, fit_two_arm_background_templates,
};
use nereids_fitting::poisson::PoissonConfig;
use nereids_physics::counts_response::{TwoArmCounts, add_count_backgrounds};

/// Observations are measured counts, so they are plain slices: the
/// window-loss report on [`TwoArmCounts`] describes a prediction and has no
/// meaning for data read off a detector.
struct Observed {
    open_beam: Vec<f64>,
    sample: Vec<f64>,
}

fn signal(open_beam: Vec<f64>, sample: Vec<f64>) -> TwoArmCounts {
    TwoArmCounts {
        open_beam,
        sample,
        open_beam_window_loss: 0.0,
        sample_window_loss: 0.0,
    }
}

fn signals() -> TwoArmCounts {
    signal(
        vec![1000.0, 900.0, 800.0, 700.0, 600.0],
        vec![600.0, 540.0, 480.0, 420.0, 360.0],
    )
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
) -> Observed {
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
    Observed {
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
        &observed.open_beam,
        &observed.sample,
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
        &observed.open_beam,
        &observed.sample,
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
    let invalid = TwoArmBackgroundTemplate {
        name: "negative".into(),
        open_beam: vec![1.0, 1.0, -1.0, 1.0, 1.0],
        sample: vec![1.0; 5],
    };
    let error = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
        signals(),
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
    let signal = signal(vec![100.0, 200.0, 300.0], vec![80.0, 160.0, 240.0]);
    let tiny_units = TwoArmBackgroundTemplate {
        name: "same_reference_in_tiny_units".into(),
        open_beam: vec![1.0e-8; 3],
        sample: vec![1.0e-8; 3],
    };
    let true_amplitude = 1.0e9;
    let observed = synthetic_observation(signal.clone(), &tiny_units, true_amplitude);

    let result = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
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
    let mut observed = Observed {
        open_beam: signal.open_beam.clone(),
        sample: signal.sample.clone(),
    };
    for (&amplitude, template) in truth.iter().zip(&templates) {
        for (count, &basis) in observed.open_beam.iter_mut().zip(&template.open_beam) {
            *count += amplitude * basis;
        }
        for (count, &basis) in observed.sample.iter_mut().zip(&template.sample) {
            *count += amplitude * basis;
        }
    }

    let result = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
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
    let template = TwoArmBackgroundTemplate {
        name: "tiny_count_reference".into(),
        open_beam: vec![1.0],
        sample: vec![1.0],
    };

    let result = fit_two_arm_background_templates(
        &[1.0e-12],
        &[1.0e-12],
        signal(vec![0.0], vec![0.0]),
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
    let template = TwoArmBackgroundTemplate {
        name: "open_only".into(),
        open_beam: vec![1.0],
        sample: vec![0.0],
    };

    let result = fit_two_arm_background_templates(
        &[1.0],
        // This arm cannot be represented by the open-only template. Its
        // large finite value must not create 0 * infinity in the derivative.
        &[1.0e300],
        signal(vec![0.0], vec![0.0]),
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
    let template = TwoArmBackgroundTemplate {
        name: "unusable_units".into(),
        open_beam: vec![1.0e-320],
        sample: vec![1.0e-320],
    };

    let error = fit_two_arm_background_templates(
        &[1.0],
        &[1.0],
        signal(vec![0.0], vec![0.0]),
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
    let template = TwoArmBackgroundTemplate {
        name: "large_counts".into(),
        open_beam: vec![1.0],
        sample: vec![1.0],
    };

    let result = fit_two_arm_background_templates(
        &[1.0e308],
        &[1.0e308],
        signal(vec![0.0], vec![0.0]),
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

/// Two templates that differ only slightly in shape, observed with the
/// amplitudes they were generated from. Fit from a start far from the truth.
fn nearly_dependent_fixture(
    difference: f64,
    n_bins: usize,
) -> (Observed, TwoArmCounts, Vec<TwoArmBackgroundTemplate>) {
    let signal = signal(vec![100.0; n_bins], vec![100.0; n_bins]);
    let ramp = |i: usize| (i as f64 - (n_bins as f64 - 1.0) / 2.0) / (n_bins as f64);
    let first = TwoArmBackgroundTemplate {
        name: "first".into(),
        open_beam: vec![1.0; n_bins],
        sample: vec![1.0; n_bins],
    };
    let second = TwoArmBackgroundTemplate {
        name: "nearly_the_same".into(),
        open_beam: (0..n_bins).map(|i| 1.0 + difference * ramp(i)).collect(),
        sample: (0..n_bins).map(|i| 1.0 - difference * ramp(i)).collect(),
    };
    let templates = vec![first, second];
    let mut observed = Observed {
        open_beam: signal.open_beam.clone(),
        sample: signal.sample.clone(),
    };
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
    (observed, signal, templates)
}

/// Shapes that are close but resolvable must yield reported uncertainties that
/// are present, finite and strictly positive — asserted unconditionally, so
/// the test cannot pass by the uncertainties simply being absent.
#[test]
fn nearly_dependent_but_resolvable_templates_report_finite_positive_uncertainties() {
    let (observed, signal, templates) = nearly_dependent_fixture(1.0e-3, 9);

    let result = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
        signal,
        1.0,
        1.0,
        &templates,
        &[1.0, 1.0],
        &PoissonConfig::default(),
    )
    .expect("resolvable templates fit");

    assert!(result.converged, "iterations = {}", result.iterations);
    assert!(result.amplitudes_identifiable);
    assert!(
        (result.amplitudes[0] - 10.0).abs() < 1.0e-6,
        "{:?}",
        result.amplitudes
    );
    assert!(
        (result.amplitudes[1] - 20.0).abs() < 1.0e-6,
        "{:?}",
        result.amplitudes
    );
    let uncertainties = result
        .amplitude_uncertainties
        .expect("an invertible information matrix yields uncertainties");
    assert!(
        uncertainties
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "{uncertainties:?}"
    );
}

/// Shapes distinct at floating-point resolution but with a numerically
/// singular information matrix: the amplitudes are nominally identifiable, yet
/// no uncertainty can be reported. The documented outcome is withheld (NaN),
/// never a misleading zero.
#[test]
fn numerically_singular_information_withholds_uncertainties_rather_than_zero() {
    let (observed, signal, templates) = nearly_dependent_fixture(1.0e-11, 3);

    let result = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
        signal,
        1.0,
        1.0,
        &templates,
        &[1.0, 1.0],
        &PoissonConfig::default(),
    )
    .expect("nearly dependent templates still have a prediction");

    assert!(result.converged);
    assert!(result.amplitudes_identifiable);
    match result.amplitude_uncertainties {
        None => {}
        Some(uncertainties) => assert!(
            uncertainties.iter().all(|value| value.is_nan()),
            "a singular information matrix must not yield a numeric sigma: {uncertainties:?}"
        ),
    }
}

/// The intended use: a flat detector-dark template plus a slowly varying
/// blocked-beam template. The two are highly correlated, which is precisely
/// where a coordinate-at-a-time sweep zig-zags for tens of thousands of
/// iterations. The fit must converge within the default budget and recover
/// both amplitudes.
#[test]
fn correlated_dark_and_sloped_blocked_beam_converge_at_default_budget() {
    let n_bins = 400;
    let neutron: Vec<f64> = (0..n_bins)
        .map(|i| 5000.0 * (-(i as f64) / 160.0).exp())
        .collect();
    let signal = signal(neutron.clone(), neutron.iter().map(|v| 0.6 * v).collect());
    let dark = TwoArmBackgroundTemplate {
        name: "dark".into(),
        open_beam: vec![1.0; n_bins],
        sample: vec![1.0; n_bins],
    };
    let blocked = TwoArmBackgroundTemplate {
        name: "blocked_beam".into(),
        open_beam: (0..n_bins)
            .map(|i| 1.0 + 0.02 * i as f64 / n_bins as f64)
            .collect(),
        sample: (0..n_bins)
            .map(|i| 1.0 + 0.02 * i as f64 / n_bins as f64)
            .collect(),
    };
    let truth = [200.0, 300.0];
    let templates = vec![dark, blocked];
    let mut observed = Observed {
        open_beam: signal.open_beam.clone(),
        sample: signal.sample.clone(),
    };
    for (&amplitude, template) in truth.iter().zip(&templates) {
        for (count, &basis) in observed.open_beam.iter_mut().zip(&template.open_beam) {
            *count += amplitude * basis;
        }
        for (count, &basis) in observed.sample.iter_mut().zip(&template.sample) {
            *count += amplitude * basis;
        }
    }

    let result = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
        signal,
        1.0,
        1.0,
        &templates,
        &[1.0, 1.0],
        &PoissonConfig::default(),
    )
    .expect("correlated templates fit");

    assert!(
        result.converged,
        "not converged after {} iterations: amplitudes {:?}",
        result.iterations, result.amplitudes
    );
    assert!(
        result.iterations < 50,
        "a joint step should converge in a handful of iterations, took {}",
        result.iterations
    );
    for (&actual, &expected) in result.amplitudes.iter().zip(&truth) {
        assert!(
            (actual / expected - 1.0).abs() < 1.0e-6,
            "{:?}",
            result.amplitudes
        );
    }
    let uncertainties = result
        .amplitude_uncertainties
        .expect("converged and identifiable");
    assert!(
        uncertainties
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
}

#[test]
fn exposure_scales_prevent_run_normalization_from_becoming_background() {
    let reference_signal = signal(vec![100.0, 100.0], vec![50.0, 50.0]);
    let sample_only = TwoArmBackgroundTemplate {
        name: "sample_only".into(),
        open_beam: vec![0.0, 0.0],
        sample: vec![1.0, 1.0],
    };

    let result = fit_two_arm_background_templates(
        &[100.0, 100.0],
        &[100.0, 100.0],
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

/// The window-loss report is in the same units as the arm it belongs to, so a
/// run-normalization factor must convert it too. Leaving it unscaled would
/// disclose a loss describing a different acquisition than the counts beside it.
#[test]
fn exposure_scale_converts_the_window_loss_report_with_its_arm() {
    let mut reference = signal(vec![100.0, 100.0], vec![50.0, 50.0]);
    reference.open_beam_window_loss = 10.0;
    reference.sample_window_loss = 4.0;
    let sample_only = TwoArmBackgroundTemplate {
        name: "sample_only".into(),
        open_beam: vec![0.0, 0.0],
        sample: vec![1.0, 1.0],
    };

    let result = fit_two_arm_background_templates(
        &[300.0, 300.0],
        &[100.0, 100.0],
        reference,
        3.0,
        2.0,
        &[sample_only],
        &[1.0],
        &PoissonConfig::default(),
    )
    .expect("exposure-scaled fit");

    assert_eq!(result.prediction.open_beam.window_loss, 30.0);
    assert_eq!(result.prediction.sample.window_loss, 8.0);
}

#[test]
fn dependent_templates_are_explicitly_marked_unidentifiable() {
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
        &[120.0; 3],
        &[120.0; 3],
        signal(vec![100.0; 3], vec![100.0; 3]),
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

/// Bins with no observation, no signal, and no template capacity contribute
/// identically zero deviance for every amplitude vector. Counting them as
/// degrees of freedom would deflate the reported goodness of fit — the same
/// defect class corrected for the joint-Poisson dof. Padding a fit with such
/// dead bins must therefore not move `deviance_per_dof`.
#[test]
fn dead_bins_do_not_deflate_the_reported_goodness_of_fit() {
    let template = TwoArmBackgroundTemplate {
        name: "flat_reference".into(),
        open_beam: vec![1.0, 1.0, 1.0],
        sample: vec![1.0, 1.0, 1.0],
    };
    let padded_template = TwoArmBackgroundTemplate {
        name: "flat_reference".into(),
        open_beam: vec![1.0, 1.0, 1.0, 0.0, 0.0],
        sample: vec![1.0, 1.0, 1.0, 0.0, 0.0],
    };

    let compact = fit_two_arm_background_templates(
        &[130.0, 140.0, 120.0],
        &[125.0, 135.0, 145.0],
        signal(vec![100.0; 3], vec![100.0; 3]),
        1.0,
        1.0,
        &[template],
        &[10.0],
        &PoissonConfig::default(),
    )
    .expect("compact fit");

    let padded = fit_two_arm_background_templates(
        &[130.0, 140.0, 120.0, 0.0, 0.0],
        &[125.0, 135.0, 145.0, 0.0, 0.0],
        signal(
            vec![100.0, 100.0, 100.0, 0.0, 0.0],
            vec![100.0, 100.0, 100.0, 0.0, 0.0],
        ),
        1.0,
        1.0,
        &[padded_template],
        &[10.0],
        &PoissonConfig::default(),
    )
    .expect("padded fit");

    assert_eq!(compact.n_informative, 6);
    assert_eq!(padded.n_informative, 6);
    assert_eq!(compact.amplitudes[0], padded.amplitudes[0]);
    assert_eq!(compact.poisson_deviance, padded.poisson_deviance);
    assert_eq!(compact.deviance_per_dof, padded.deviance_per_dof);
}

/// A non-finite tolerance would accept the first iteration unconditionally
/// and certify an arbitrary amplitude vector as converged.
#[test]
fn non_finite_tolerance_is_rejected_before_fitting() {
    let template = shaped_template("blocked_beam");
    let signal = signals();
    let observed = synthetic_observation(signal.clone(), &template, 25.0);
    for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0e-8] {
        let config = PoissonConfig {
            tol_param: bad,
            ..PoissonConfig::default()
        };
        let error = fit_two_arm_background_templates(
            &observed.open_beam,
            &observed.sample,
            signal.clone(),
            1.0,
            1.0,
            std::slice::from_ref(&template),
            &[1.0],
            &config,
        )
        .expect_err("tolerance must be validated");
        assert!(
            matches!(
                error,
                nereids_fitting::error::FittingError::InvalidConfig(_)
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains("tol_param"), "{error}");
    }
}

/// A negative or non-finite window loss is malformed input and must be
/// reported as such before any optimization runs, not surface afterwards as a
/// model-evaluation failure.
#[test]
fn bad_window_loss_is_rejected_as_invalid_input_before_fitting() {
    let template = shaped_template("blocked_beam");
    let observed = synthetic_observation(signals(), &template, 25.0);
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let mut signal = signals();
        signal.sample_window_loss = bad;
        let error = fit_two_arm_background_templates(
            &observed.open_beam,
            &observed.sample,
            signal,
            1.0,
            1.0,
            std::slice::from_ref(&template),
            &[1.0],
            &PoissonConfig::default(),
        )
        .expect_err("bad window loss must be rejected");
        assert!(
            matches!(
                error,
                nereids_fitting::error::FittingError::InvalidConfig(_)
            ),
            "expected InvalidConfig for {bad}, got {error:?}"
        );
        assert!(error.to_string().contains("window_loss"), "{error}");
    }
}

/// Dependent templates span fewer directions than they have names. With two
/// informative bins and three proportional templates the raw count would
/// reject the fit, but the rank is one and the degrees of freedom are
/// positive: the total is determined and must be fitted, with the individual
/// amplitudes reported as unidentifiable.
#[test]
fn dependent_templates_with_positive_rank_dof_are_fitted_not_rejected() {
    let templates: Vec<TwoArmBackgroundTemplate> = [1.0, 2.0, 3.0]
        .iter()
        .enumerate()
        .map(|(index, &scale)| TwoArmBackgroundTemplate {
            name: format!("proportional_{index}"),
            open_beam: vec![scale],
            sample: vec![scale],
        })
        .collect();

    let result = fit_two_arm_background_templates(
        &[30.0],
        &[30.0],
        signal(vec![10.0], vec![10.0]),
        1.0,
        1.0,
        &templates,
        &[0.0, 0.0, 0.0],
        &PoissonConfig::default(),
    )
    .expect("rank-one templates with two informative bins have one degree of freedom");

    assert!(result.converged);
    assert!(!result.amplitudes_identifiable);
    assert!(result.amplitude_uncertainties.is_none());
    assert_eq!(result.n_informative, 2);
    assert!((result.prediction.open_beam.total[0] - 30.0).abs() < 1.0e-9);
    assert!(result.poisson_deviance < 1.0e-18);
}

/// When the unconstrained optimum is negative the amplitude is held at zero by
/// the bound, not chosen by the data. The result must say so, and the
/// reported sigma is then a one-sided curvature scale, not an interval.
#[test]
fn amplitude_held_on_the_zero_bound_is_flagged() {
    let template = TwoArmBackgroundTemplate {
        name: "over_predicted".into(),
        open_beam: vec![1.0; 4],
        sample: vec![1.0; 4],
    };

    // The neutron signal alone already over-predicts every bin, so any
    // positive background makes the fit worse.
    let result = fit_two_arm_background_templates(
        &[90.0; 4],
        &[90.0; 4],
        signal(vec![100.0; 4], vec![100.0; 4]),
        1.0,
        1.0,
        &[template],
        &[5.0],
        &PoissonConfig::default(),
    )
    .expect("bound-active fit");

    assert!(result.converged);
    assert_eq!(result.amplitudes, vec![0.0]);
    assert_eq!(result.amplitude_at_bound, vec![true]);
    let uncertainties = result
        .amplitude_uncertainties
        .expect("expected information is finite on the boundary");
    // Value-level oracle that discriminates the information convention:
    // eight bins with mu = 100 and unit weight give expected information
    // 8/100, so sigma = sqrt(12.5). The observed-information form would
    // give sqrt(100^2 / (8 * 90)) = 3.7268 instead.
    assert!(
        (uncertainties[0] - 12.5_f64.sqrt()).abs() < 1.0e-9,
        "sigma = {}",
        uncertainties[0]
    );
}

/// Independent oracle with `y != mu` on an interior two-template solution:
/// the reported sigmas must be the square roots of the diagonal of the
/// inverse expected-information matrix, computed here by hand from the
/// fitted expectation and the raw templates.
#[test]
fn interior_two_template_sigmas_match_the_inverse_information_diagonal() {
    let ramp = vec![0.25, 0.5, 0.75, 1.0];
    let templates = vec![
        TwoArmBackgroundTemplate {
            name: "dark".into(),
            open_beam: vec![1.0; 4],
            sample: vec![1.0; 4],
        },
        TwoArmBackgroundTemplate {
            name: "ramp".into(),
            open_beam: ramp.clone(),
            sample: ramp.clone(),
        },
    ];
    let result = fit_two_arm_background_templates(
        &[130.0, 110.0, 120.0, 140.0],
        &[125.0, 135.0, 115.0, 145.0],
        signal(vec![100.0; 4], vec![100.0; 4]),
        1.0,
        1.0,
        &templates,
        &[1.0, 1.0],
        &PoissonConfig::default(),
    )
    .expect("interior fit");
    assert!(result.converged);
    assert_eq!(
        result.amplitude_at_bound,
        vec![false, false],
        "{:?}",
        result.amplitudes
    );

    let mu: Vec<f64> = result
        .prediction
        .open_beam
        .total
        .iter()
        .chain(&result.prediction.sample.total)
        .copied()
        .collect();
    let joined = |t: &TwoArmBackgroundTemplate| -> Vec<f64> {
        t.open_beam.iter().chain(&t.sample).copied().collect()
    };
    let (b1, b2) = (joined(&templates[0]), joined(&templates[1]));
    let info = |x: &[f64], y: &[f64]| -> f64 {
        x.iter().zip(y).zip(&mu).map(|((a, b), m)| a * b / m).sum()
    };
    let (i11, i12, i22) = (info(&b1, &b1), info(&b1, &b2), info(&b2, &b2));
    let det = i11 * i22 - i12 * i12;
    let expected = [(i22 / det).sqrt(), (i11 / det).sqrt()];

    let sigma = result
        .amplitude_uncertainties
        .expect("interior, identifiable");
    for (got, want) in sigma.iter().zip(&expected) {
        assert!(
            (got / want - 1.0).abs() < 1.0e-9,
            "sigma {sigma:?} vs {expected:?}"
        );
    }
}

/// The module's stated ordinary inputs — flat dark plus slowly varying
/// blocked beam — with the dark amplitude pushed onto its bound. The free
/// amplitude is determined to a fraction of a percent; its sigma must be the
/// reduced-set value, not the marginal over the direction the constraint
/// removed (which is ~200x larger for this near-collinear pair).
#[test]
fn free_amplitude_sigma_is_conditioned_on_the_pinned_partner() {
    let n_bins = 400;
    let neutron: Vec<f64> = (0..n_bins)
        .map(|i| 5000.0 * (-(i as f64) / 160.0).exp())
        .collect();
    let dark = vec![1.0; n_bins];
    let blocked: Vec<f64> = (0..n_bins)
        .map(|i| 1.0 + 0.02 * i as f64 / n_bins as f64)
        .collect();
    let observed: Vec<f64> = (0..n_bins)
        .map(|i| neutron[i] + 300.0 * blocked[i] - 50.0 * dark[i])
        .collect();
    let templates = vec![
        TwoArmBackgroundTemplate {
            name: "dark".into(),
            open_beam: dark.clone(),
            sample: dark.clone(),
        },
        TwoArmBackgroundTemplate {
            name: "blocked_beam".into(),
            open_beam: blocked.clone(),
            sample: blocked.clone(),
        },
    ];

    let result = fit_two_arm_background_templates(
        &observed,
        &observed,
        signal(neutron.clone(), neutron.clone()),
        1.0,
        1.0,
        &templates,
        &[10.0, 10.0],
        &PoissonConfig::default(),
    )
    .expect("bound-active correlated fit");
    assert!(result.converged, "iterations {}", result.iterations);
    assert_eq!(
        result.amplitude_at_bound,
        vec![true, false],
        "{:?}",
        result.amplitudes
    );
    assert_eq!(result.amplitudes[0], 0.0);

    let mu: Vec<f64> = result
        .prediction
        .open_beam
        .total
        .iter()
        .chain(&result.prediction.sample.total)
        .copied()
        .collect();
    let both = |t: &[f64]| -> Vec<f64> { t.iter().chain(t).copied().collect() };
    let info = |x: &[f64], y: &[f64]| -> f64 {
        x.iter().zip(y).zip(&mu).map(|((a, b), m)| a * b / m).sum()
    };
    let (d, b) = (both(&dark), both(&blocked));
    let (i11, i12, i22) = (info(&d, &d), info(&d, &b), info(&b, &b));
    let reduced_free = 1.0 / i22.sqrt();
    let marginal_free = (i11 / (i11 * i22 - i12 * i12)).sqrt();
    let pinned_direct = 1.0 / i11.sqrt();

    let sigma = result
        .amplitude_uncertainties
        .expect("identifiable and converged");
    assert!(
        (sigma[1] / reduced_free - 1.0).abs() < 1.0e-9,
        "free sigma {} vs reduced {reduced_free}",
        sigma[1]
    );
    assert!(
        marginal_free > 50.0 * sigma[1],
        "the marginal ({marginal_free}) must be far above the conditioned value ({})",
        sigma[1]
    );
    assert!(
        (sigma[0] / pinned_direct - 1.0).abs() < 1.0e-9,
        "pinned sigma {} vs direct curvature {pinned_direct}",
        sigma[0]
    );
}

/// At the boundary of the Poisson support the expected information diverges.
/// The uncertainty must be withheld there, not reported as the number that
/// dropping the bin happens to produce — which jumps discontinuously as the
/// expectation reaches zero.
#[test]
fn zero_expectation_on_a_sensitive_bin_withholds_the_sigma() {
    let template = TwoArmBackgroundTemplate {
        name: "flat".into(),
        open_beam: vec![1.0, 1.0],
        sample: vec![1.0, 1.0],
    };
    let fit = |epsilon: f64| {
        fit_two_arm_background_templates(
            &[0.0, 90.0],
            &[0.0, 90.0],
            signal(vec![epsilon, 100.0], vec![epsilon, 100.0]),
            1.0,
            1.0,
            std::slice::from_ref(&template),
            &[0.0],
            &PoissonConfig::default(),
        )
        .expect("boundary fit")
    };

    let near = fit(1.0e-4);
    assert_eq!(near.amplitude_at_bound, vec![true]);
    let near_sigma = near.amplitude_uncertainties.expect("regular")[0];
    assert!(near_sigma.is_finite() && near_sigma < 0.1, "{near_sigma}");

    let boundary = fit(0.0);
    assert!(boundary.converged);
    assert_eq!(boundary.amplitude_at_bound, vec![true]);
    let boundary_sigma = boundary
        .amplitude_uncertainties
        .expect("entry withheld, not the array")[0];
    assert!(
        boundary_sigma.is_nan(),
        "got {boundary_sigma}, expected NaN"
    );
}

#[test]
fn zero_max_iter_is_rejected_before_fitting() {
    let template = shaped_template("blocked_beam");
    let observed = synthetic_observation(signals(), &template, 25.0);
    let error = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
        signals(),
        1.0,
        1.0,
        std::slice::from_ref(&template),
        &[1.0],
        &PoissonConfig {
            max_iter: 0,
            ..PoissonConfig::default()
        },
    )
    .expect_err("zero iterations is not a fit");
    assert!(
        matches!(
            error,
            nereids_fitting::error::FittingError::InvalidConfig(_)
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains("max_iter"), "{error}");
}

/// An interior solution is not on its bound.
#[test]
fn interior_amplitude_is_not_flagged_as_bound() {
    let template = shaped_template("blocked_beam");
    let signal = signals();
    let observed = synthetic_observation(signal.clone(), &template, 25.0);
    let result = fit_two_arm_background_templates(
        &observed.open_beam,
        &observed.sample,
        signal,
        1.0,
        1.0,
        std::slice::from_ref(&template),
        &[1.0],
        &PoissonConfig::default(),
    )
    .expect("interior fit");
    assert_eq!(result.amplitude_at_bound, vec![false]);
}

/// Three genuinely independent, near-collinear templates with one partner
/// whose true amplitude is negative, so its bound is active with a small
/// multiplier — smaller than the default stopping slack along that nearly
/// degenerate direction. The active set, and with it the conditioned sigma,
/// must not depend on where the iteration started or how tight the
/// tolerance was.
#[test]
fn active_set_and_sigma_are_invariant_to_start_and_tolerance() {
    let n_bins = 200;
    let x = |i: usize| i as f64 / n_bins as f64;
    let neutron: Vec<f64> = (0..n_bins)
        .map(|i| 5000.0 * (-(i as f64) / 80.0).exp())
        .collect();
    let dark = vec![1.0; n_bins];
    let blocked: Vec<f64> = (0..n_bins).map(|i| 1.0 + 0.02 * x(i)).collect();
    let third: Vec<f64> = (0..n_bins)
        .map(|i| 1.0 + 0.02 * x(i) + 0.003 * x(i) * x(i))
        .collect();
    let observed: Vec<f64> = (0..n_bins)
        .map(|i| neutron[i] + 100.0 * dark[i] - 30.0 * blocked[i] + 200.0 * third[i])
        .collect();
    let template = |name: &str, shape: &Vec<f64>| TwoArmBackgroundTemplate {
        name: name.into(),
        open_beam: shape.clone(),
        sample: shape.clone(),
    };
    let templates = vec![
        template("dark", &dark),
        template("blocked_beam", &blocked),
        template("third", &third),
    ];

    let mut reference: Option<(Vec<f64>, Vec<f64>)> = None;
    for start in [[0.0; 3], [10.0; 3], [500.0; 3]] {
        for tol in [1.0e-8, 1.0e-12] {
            let result = fit_two_arm_background_templates(
                &observed,
                &observed,
                signal(neutron.clone(), neutron.clone()),
                1.0,
                1.0,
                &templates,
                &start,
                &PoissonConfig {
                    tol_param: tol,
                    ..PoissonConfig::default()
                },
            )
            .expect("near-collinear fit");
            assert!(result.converged, "start {start:?} tol {tol}");
            assert_eq!(
                result.amplitude_at_bound,
                vec![false, true, false],
                "start {start:?} tol {tol}: amplitudes {:?}",
                result.amplitudes
            );
            let sigma = result.amplitude_uncertainties.expect("identifiable");
            match &reference {
                None => reference = Some((result.amplitudes.clone(), sigma)),
                Some((amplitudes_0, sigma_0)) => {
                    for ((a, a0), (s, s0)) in result
                        .amplitudes
                        .iter()
                        .zip(amplitudes_0)
                        .zip(sigma.iter().zip(sigma_0))
                    {
                        assert!(
                            (a - a0).abs() <= 1.0e-6 * a0.abs().max(1.0),
                            "amplitude drift at start {start:?} tol {tol}: {a} vs {a0}"
                        );
                        assert!(
                            (s / s0 - 1.0).abs() < 1.0e-6,
                            "sigma regime flipped at start {start:?} tol {tol}: {s} vs {s0}"
                        );
                    }
                }
            }
        }
    }
}

/// A conversion that underflows to zero would return an amplitude that cannot
/// rebuild the nonzero background the fit found. It is as unrepresentable as
/// an overflow and must be refused the same way.
#[test]
fn amplitude_conversion_underflow_is_rejected_like_overflow() {
    let template = TwoArmBackgroundTemplate {
        name: "enormous_units".into(),
        open_beam: vec![1.0e308],
        sample: vec![1.0e308],
    };
    let error = fit_two_arm_background_templates(
        &[1.0e-20],
        &[1.0e-20],
        signal(vec![0.0], vec![0.0]),
        1.0,
        1.0,
        &[template],
        &[5.0e-324],
        &PoissonConfig::default(),
    )
    .expect_err("an amplitude that underflows to zero must not escape the API");
    assert!(
        error.to_string().contains("rescale the template counts"),
        "{error}"
    );
}

/// The loose search bracket can overflow while the optimum itself is a
/// perfectly representable zero. The overflow must not abort the fit.
#[test]
fn overflowing_search_bracket_does_not_reject_a_representable_optimum() {
    let template = TwoArmBackgroundTemplate {
        name: "lopsided".into(),
        open_beam: vec![1.0],
        sample: vec![0.1],
    };
    let result = fit_two_arm_background_templates(
        &[1.0e308],
        &[1.0e308],
        signal(vec![1.0e308], vec![1.0e308]),
        1.0,
        1.0,
        &[template],
        &[0.0],
        &PoissonConfig::default(),
    )
    .expect("zero background fits these observations exactly");
    assert!(result.converged);
    assert_eq!(result.amplitudes, vec![0.0]);
    assert_eq!(result.poisson_deviance, 0.0);
}

/// `max_iter` is the caller's whole iteration budget: the post-convergence
/// active-set polish spends only what the main loop left, so the reported
/// count can never exceed it — including the tightest budget of one.
#[test]
fn max_iter_caps_the_reported_iterations_including_the_polish() {
    let template = shaped_template("blocked_beam");
    let signal = signals();
    let observed = synthetic_observation(signal.clone(), &template, 25.0);
    for max_iter in [1, 2, 3, 200] {
        let result = fit_two_arm_background_templates(
            &observed.open_beam,
            &observed.sample,
            signal.clone(),
            1.0,
            1.0,
            std::slice::from_ref(&template),
            &[1.0],
            &PoissonConfig {
                max_iter,
                ..PoissonConfig::default()
            },
        )
        .expect("fit within budget");
        assert!(
            result.iterations <= max_iter,
            "max_iter {max_iter} but reported {} iterations",
            result.iterations
        );
    }
}

/// Degrees of freedom must stay positive after dead bins are excluded, and the
/// rejection must count informative bins rather than raw array length.
#[test]
fn underdetermined_fit_is_rejected_on_informative_bins_not_array_length() {
    let templates = vec![
        TwoArmBackgroundTemplate {
            name: "first".into(),
            open_beam: vec![1.0, 0.0, 0.0, 0.0],
            sample: vec![0.0, 0.0, 0.0, 0.0],
        },
        TwoArmBackgroundTemplate {
            name: "second".into(),
            open_beam: vec![0.0, 1.0, 0.0, 0.0],
            sample: vec![0.0, 0.0, 0.0, 0.0],
        },
    ];

    // Eight raw concatenated values, but only the two bins the templates can
    // reach carry any information at all.
    let error = fit_two_arm_background_templates(
        &[5.0, 5.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0],
        signal(vec![0.0; 4], vec![0.0; 4]),
        1.0,
        1.0,
        &templates,
        &[1.0, 1.0],
        &PoissonConfig::default(),
    )
    .expect_err("two amplitudes cannot be fitted from two informative bins");

    assert!(
        error.to_string().contains("informative count"),
        "unexpected message: {error}"
    );
}
