use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid, ic_cdf, ic_pulse,
};
use nereids_physics::resolution::TOF_FACTOR;

/// Simpson integral of `f` over `[a, b]` with `2n` intervals, accumulated
/// with Neumaier compensation so the oracle's summation noise stays below the
/// tolerances that use it (naive summation of 2e5 terms costs ~1e-11).
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / (2 * n) as f64;
    let mut sum = f(a) + f(b);
    let mut compensation = 0.0_f64;
    for k in 1..2 * n {
        let w = if k % 2 == 1 { 4.0 } else { 2.0 };
        let value = w * f(a + k as f64 * h);
        let t = sum + value;
        compensation += if sum.abs() >= value.abs() {
            (sum - t) + value
        } else {
            (value - t) + sum
        };
        sum = t;
    }
    (sum + compensation) * h / 3.0
}

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

    // Independent oracle: integrate the exact Gamma(3) moderator CDF over a
    // standard normal variable. This does not use the sampled convolution or
    // detector-bin integrator under test.
    let normal_density = |delay: f64| {
        (-0.5 * (delay / sigma).powi(2)).exp() / (sigma * std::f64::consts::TAU.sqrt())
    };
    let moderator_cdf = |delay: f64| gamma3_cdf(alpha * delay);
    let window_oracle = |lo: f64, hi: f64| {
        simpson(
            |gaussian_delay: f64| {
                normal_density(gaussian_delay)
                    * (moderator_cdf(hi - gaussian_delay) - moderator_cdf(lo - gaussian_delay))
            },
            -10.0 * sigma,
            10.0 * sigma,
            100_000,
        )
    };
    let window_sum = |lo: f64, hi: f64| -> f64 {
        model
            .detector_bin_probabilities(
                true_energy_ev,
                &[nominal_arrival + lo, nominal_arrival + hi],
                0.0,
            )
            .expect("valid folded detector window")
            .iter()
            .sum()
    };

    // Broad window covering the whole sampled support: with the ±8σ reach the
    // truncated Gaussian mass is ~1e-15, so a broad window must no longer be
    // undercounted by the truncation bookkeeping.
    let got_broad = window_sum(lower, upper);
    let oracle_broad = window_oracle(lower, upper);
    assert!(
        (got_broad - oracle_broad).abs() < 2.0e-5,
        "sampled Gaussian-folded probability {got_broad} disagrees with independent integral {oracle_broad}"
    );
    assert!(
        1.0 - got_broad < 1.0e-6,
        "broad window undercounted: kernel-truncation bookkeeping discarded {:.3e}",
        1.0 - got_broad
    );
    assert!(
        got_broad < 1.0,
        "finite window was silently renormalized to one"
    );

    // Deliberately cut window: the upper edge sits on a sampled grid point
    // near +2σ, so real fold mass lies beyond it. Both the oracle and the
    // sampled path must SEE that loss — a renormalizing implementation would
    // report ~1 (comparison is loose because the n_tau = 8 grid is
    // deliberately coarse; the broad-window comparison above is the tight one).
    let cut = delays
        .iter()
        .copied()
        .min_by(|a, b| (a - 2.0 * sigma).abs().total_cmp(&(b - 2.0 * sigma).abs()))
        .expect("non-empty sampled support");
    let got_cut = window_sum(lower, cut);
    let oracle_cut = window_oracle(lower, cut);
    assert!(
        1.0 - oracle_cut > 1.0e-2,
        "cut-window oracle must expose real missing mass (got {oracle_cut})"
    );
    assert!(
        1.0 - got_cut > 1.0e-2,
        "cut window silently renormalized: reported {got_cut}"
    );
    assert!(
        got_cut < got_broad,
        "cutting the window must reduce the reported mass"
    );
}

#[test]
fn unequal_rate_storage_cdf_matches_pulse_quadrature() {
    // General α ≠ β storage CDF against Simpson integration of the density —
    // the two functions share no code path for the storage term, so this pins
    // both the bounded-bracket branch (|u| ≥ 0.05) and the Taylor branch
    // (0 < |u| < 0.05) against an independent quadrature oracle.
    let cases: [(f64, f64, f64, f64); 6] = [
        (1.7, 0.45, 0.35, 0.4),  // bounded branch, early time
        (1.7, 0.45, 0.35, 2.7),  // bounded branch, mid pulse
        (1.7, 0.45, 0.35, 12.0), // bounded branch, deep tail
        (0.6, 2.4, 0.8, 3.0),    // β > α (negative u)
        (1.0, 0.995, 0.5, 8.0),  // Taylor branch: u = 0.04
        (1.0, 0.995, 0.5, 1.0),  // Taylor branch: u = 0.005
    ];
    // Tolerance: near the |u| = 0.05 Taylor-branch boundary the two
    // implementations carry independent series truncation — measured against
    // a 60-digit reference at (1, 0.995, 0.5, 8): ic_cdf 4.7e-12, the
    // integrated pulse 1.4e-11 — so the comparison budget is their sum.
    for (alpha, beta, r, tau) in cases {
        let quad = simpson(|t| ic_pulse(alpha, beta, r, t), 0.0, tau, 100_000);
        let cdf = ic_cdf(alpha, beta, r, tau);
        assert!(
            (cdf - quad).abs() < 5.0e-11,
            "ic_cdf({alpha}, {beta}, {r}, {tau}) = {cdf:.16e} vs quadrature {quad:.16e}"
        );
    }
}

#[test]
fn storage_with_channel_fold_preserves_missing_tail_mass() {
    // The sampled fold path with storage active (r > 0 AND a channel fold):
    // the oracle is a double quadrature — the triangle channel density folded
    // with the pulse-density integral — sharing only `ic_pulse` with the code
    // under test (whose CDF equivalence the quadrature test above pins).
    let (alpha, beta, r) = (2.0, 0.25, 0.3);
    let fwhm = 0.7_f64; // triangle half-base = FWHM
    let flight_path_m = 25.0_f64;
    let true_energy_ev = 25.0_f64;
    let model = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::Const(alpha),
            beta: EnergyLaw::Const(beta),
            r: EnergyLaw::Const(r),
            burst_sigma_us: None,
            channel_fwhm_us: Some(fwhm),
        },
        flight_path_m,
        &SynthesisGrid::new(1.0, 100.0),
    )
    .expect("valid folded storage IC model");

    let nominal = TOF_FACTOR * flight_path_m / true_energy_ev.sqrt();
    // The window deliberately cuts the slow storage tail (β = 0.25 ⇒ mean
    // storage delay 4 µs) so real mass lies beyond the last edge.
    let edges = [nominal - 1.0, nominal + 1.0, nominal + 3.0, nominal + 6.0];
    let got: f64 = model
        .detector_bin_probabilities(true_energy_ev, &edges, 0.0)
        .expect("valid detector bins")
        .iter()
        .sum();

    let half_base = fwhm;
    let triangle = |c: f64| (1.0 - c.abs() / half_base) / half_base;
    let window_mass = |lo: f64, hi: f64| {
        simpson(
            |c| {
                let a = (lo - c).max(0.0);
                let b = (hi - c).max(0.0);
                if b <= a {
                    0.0
                } else {
                    triangle(c) * simpson(|t| ic_pulse(alpha, beta, r, t), a, b, 4_000)
                }
            },
            -half_base,
            half_base,
            600,
        )
    };
    let oracle = window_mass(-1.0, 6.0);
    assert!(
        1.0 - oracle > 0.05,
        "window must lose real storage-tail mass (oracle = {oracle:.6})"
    );
    assert!(
        (got - oracle).abs() < 5.0e-5,
        "folded storage window mass {got:.8} vs double-quadrature oracle {oracle:.8}"
    );
    assert!(got < 1.0, "finite window was silently renormalized to one");
}

#[test]
fn singular_inverse_lambda_laws_are_rejected() {
    let grid = SynthesisGrid::new(5.0, 50.0);
    // Construction: an all-zero (undefined at every energy) rate law must be
    // rejected loudly, not floored into a plausible ~1e9 µs⁻¹ rate.
    for (name, params) in [
        (
            "alpha",
            IkedaCarpenterParams {
                alpha: EnergyLaw::InverseLambda { a0: 0.0, a1: 0.0 },
                beta: EnergyLaw::Const(0.1),
                r: EnergyLaw::Const(0.0),
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
        ),
        (
            "beta",
            IkedaCarpenterParams {
                alpha: EnergyLaw::Const(2.0),
                beta: EnergyLaw::InverseLambda { a0: 0.0, a1: 0.0 },
                r: EnergyLaw::Const(0.5),
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
        ),
        (
            "tiny-negative beta",
            IkedaCarpenterParams {
                alpha: EnergyLaw::Const(2.0),
                beta: EnergyLaw::InverseLambda {
                    a0: -5.0e-10,
                    a1: 0.0,
                },
                r: EnergyLaw::Const(0.5),
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
        ),
    ] {
        let err = IkedaCarpenter::new(params, 25.0, &grid)
            .expect_err("singular law must not construct")
            .to_string();
        assert!(
            err.contains("singular"),
            "{name}: error must name the singularity, got: {err}"
        );
    }

    // Probe-time: a law that is regular over the whole synthesis range but
    // whose denominator crosses zero at a probe energy outside it must fail
    // at that probe, not evaluate through the floor. λ(4 eV) is recovered
    // through the public eval so the crossing is placed exactly.
    let lambda_at_4 = 1.0 / EnergyLaw::InverseLambda { a0: 0.0, a1: 1.0 }.eval(4.0);
    let crossing = EnergyLaw::InverseLambda {
        a0: 1.0,
        a1: -1.0 / lambda_at_4,
    };
    let model = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::Const(2.0),
            beta: crossing,
            r: EnergyLaw::Const(0.2),
            burst_sigma_us: None,
            channel_fwhm_us: None,
        },
        25.0,
        &grid,
    )
    .expect("regular over the synthesis range");
    let err = model
        .kernel_at(4.0)
        .expect_err("singular probe energy must be rejected")
        .to_string();
    assert!(
        err.contains("singular"),
        "probe error must name the singularity, got: {err}"
    );
}

#[test]
fn storage_cdf_matches_arbitrary_precision_convolution_reference() {
    // Reference values computed OUTSIDE this codebase from the DEFINING
    // convolution: CDF = (1−r)·Γ₃(ατ) + r·∫₀^τ (α³s²e^{−αs}/2)·(1−e^{−β(τ−s)}) ds
    // via 40-digit decimal Simpson (40 000 intervals; quadrature error ≲ 1e-14).
    // They share no code with ic_cdf or ic_pulse, so a storage-term error
    // common to both implementations cannot hide here. Tolerance covers the
    // f64 implementations' Taylor truncation near the |u| = 0.05 boundary
    // (measured 4.7e-12 at case 3 against a 60-digit closed-form check).
    let pins: [(f64, f64, f64, f64, f64); 4] = [
        (1.7, 0.45, 0.35, 2.7, 0.665_083_699_251_977_2),
        (0.6, 2.4, 0.8, 3.0, 0.219_541_992_689_109_74),
        (1.0, 0.995, 0.5, 8.0, 0.971_788_676_687_669_9),
        (2.0, 0.25, 0.3, 6.0, 0.899_740_379_887_985_2),
    ];
    for (alpha, beta, r, tau, reference) in pins {
        let cdf = ic_cdf(alpha, beta, r, tau);
        assert!(
            (cdf - reference).abs() < 1.0e-11,
            "ic_cdf({alpha}, {beta}, {r}, {tau}) = {cdf:.16e} vs external reference {reference:.16e}"
        );
    }
}

#[test]
fn non_finite_rate_parameters_return_visible_nan() {
    // Domain contract: garbage in stays visible. A NaN rate must not be
    // floored into a plausible CDF or pulse value.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(ic_cdf(bad, 0.5, 0.3, 2.0).is_nan());
        assert!(ic_cdf(2.0, bad, 0.3, 2.0).is_nan());
        assert!(ic_cdf(2.0, 0.5, bad, 2.0).is_nan());
        assert!(ic_pulse(bad, 0.5, 0.3, 2.0).is_nan());
        assert!(ic_pulse(2.0, bad, 0.3, 2.0).is_nan());
        assert!(ic_pulse(2.0, 0.5, bad, 2.0).is_nan());
    }
    // Finite non-positive rates keep the documented floor semantics.
    assert!(ic_cdf(-1.0, 0.5, 0.0, 2.0).is_finite());
}

#[test]
fn singular_exp_milli_ev_r_law_is_rejected() {
    // κ = 0 (undefined) and tiny-negative κ (divergent law) must not be
    // silently mapped to R = 0 by eval's floor.
    for kappa in [0.0, -5.0e-10] {
        let err = IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::Const(2.0),
                beta: EnergyLaw::Const(0.5),
                r: EnergyLaw::ExpMilliEv { kappa },
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            25.0,
            &SynthesisGrid::new(1.0, 100.0),
        )
        .expect_err("singular R law must not construct")
        .to_string();
        assert!(
            err.contains("singular"),
            "κ = {kappa}: error must name the singularity, got: {err}"
        );
    }
    // A small POSITIVE κ is the legitimate κ → 0⁺ limit (R → 0) and stays valid.
    IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::Const(2.0),
            beta: EnergyLaw::Const(0.5),
            r: EnergyLaw::ExpMilliEv { kappa: 5.0e-10 },
            burst_sigma_us: None,
            channel_fwhm_us: None,
        },
        25.0,
        &SynthesisGrid::new(1.0, 100.0),
    )
    .expect("positive tiny κ is the valid R → 0 limit");
}

#[test]
fn cdf_far_tail_values_are_exact_not_nan() {
    // Far-tail regression: at³ / u³ overflow used to produce inf·0 = NaN,
    // which detector-bin differencing then silently converted to zero mass.
    let deep_prompt = ic_cdf(1.0, 1.0e-6, 0.5, 1.0e103);
    assert_eq!(deep_prompt, 1.0, "deep prompt tail must saturate at 1");

    let instant_storage = ic_cdf(1.0, 1.0e101, 0.5, 1.0);
    assert!(
        instant_storage.is_finite() && (0.0..=1.0).contains(&instant_storage),
        "instant-storage limit must stay finite, got {instant_storage}"
    );
    assert!(
        (instant_storage - gamma3_cdf(1.0)).abs() < 1.0e-15,
        "instant storage decays to the prompt CDF"
    );

    let overflow_prompt = ic_cdf(1.0e155, 1.0, 0.0, 1.0);
    assert_eq!(overflow_prompt, 1.0, "Γ₃ far tail must saturate at 1");
}
