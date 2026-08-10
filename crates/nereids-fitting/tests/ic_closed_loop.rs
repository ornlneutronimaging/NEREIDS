//! Closed-loop acceptance test for the physics-complete bounded IC
//! calibration family (#642).
//!
//! The issue's failure mode: the old 2-parameter IC family (β pinned, R ≡ 0,
//! no PSR fold, α(E) allowed to go negative) lacked the storage-shape freedom,
//! which re-expressed as a ~90 K temperature degeneracy on real data. This
//! test closes the loop end-to-end on synthetic Ta-181:
//!
//! 1. generate noisy transmission from an IC truth kernel whose parameters are
//!    all INTERIOR to the new calibration boxes (storage tail present, PSR
//!    triangle folded);
//! 2. calibrate the IC family at the known (ρ, T) — the calibration must
//!    reach χ²/dof ≈ 1 with no pinned bounds and recover the truth;
//! 3. pin the calibrated resolution and re-fit the TEMPERATURE alone — the
//!    recovered T must agree with truth within its own uncertainty, and that
//!    uncertainty must be far below the 90 K degeneracy scale.
//!
//! Run at 300 K (ambient calibrant) and 1073 K (furnace condition).

use std::sync::Arc;

use nereids_endf::resonance::test_support::synthetic_isotope_multi;
use nereids_fitting::lm::{self, LmConfig};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::resolution_calib::{
    CalibrationConfig, ResolutionFamily, calibrate_resolution,
};
use nereids_fitting::transmission_model::TransmissionFitModel;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Normal};

/// IC truth: all coordinates interior to the calibration boxes (a0 ∈
/// [0.01, 5], a1 ∈ [1e-3, 2], β ∈ [0.02, 5], R ∈ [0, 1], PSR = config pin).
const A0_TRUE: f64 = 0.35;
const A1_TRUE: f64 = 0.05;
const BETA_TRUE: f64 = 0.25;
const R_TRUE: f64 = 0.15;
/// PSR triangle FWHM (µs) — equals the calibrator's default 350 ns pin.
const PSR_TRUE_US: f64 = 0.35;
/// Ta-181 areal density (at/b). Sized so the broadened dip minima land at
/// T ≈ 0.2–0.6 (non-black: at ~2e-3 the resonances saturate to T ≈ 1e-4 and
/// the calibrant carries no width information at the dip cores).
const DENSITY_TRUE: f64 = 2.0e-4;
/// Absolute transmission noise σ (0.3 % of the unit open-beam level).
const NOISE_SIGMA: f64 = 0.003;
/// Absolute ceiling (K) on the recovered-temperature bias |T_fit − T_true|
/// (#645 round 3, F3). The σ_T < 30 K gate plus the 3σ statistical window
/// jointly admit a worst-case bias of ~90 K — numerically the very
/// degeneracy scale of the old 2-parameter family this test exists to guard.
/// Observed biases are 0.5 K (300 K) / 1.2 K (1073 K); 30 K is far above
/// noise wander yet a third of the degeneracy scale.
const T_BIAS_MAX_K: f64 = 30.0;

/// Full calibrate → pin → refit-T loop at one truth temperature.
fn closed_loop(t_true_k: f64) {
    // Real Ta-181 resonance energies (10.36 / 24.0 / 39.1 eV; Γn, Γγ in eV) in
    // ONE L-group, so the potential-scattering background is not N-folded.
    let ta181 = synthetic_isotope_multi(
        73,
        181,
        &[
            (10.36, 0.003, 0.058),
            (24.0, 0.009, 0.060),
            (39.1, 0.040, 0.060),
        ],
    );
    let sample = SampleParams::new(t_true_k, vec![(ta181.clone(), DENSITY_TRUE)]).unwrap();
    // Grid density and synthesis resolution are sized for debug-build runtime
    // (tests run unoptimized): 500 points / 40×320 synthesis keeps one loop
    // under ~5 min; max_iter = 1200 plus the calibrator's built-in simplex
    // re-inflation lets the 4-parameter Nelder–Mead actually reach the χ²
    // minimum (a collapsed simplex once stalled the 300 K loop at
    // Δχ² ≈ +130, whose ~1.5 % kernel-width error re-expressed as a ~23 K
    // temperature bias in the pinned refit). The ASSERTIONS are the
    // acceptance spec and are not relaxed.
    let n_points = 500;
    let (e_lo, e_hi) = (6.0, 44.0);
    let energies: Vec<f64> = (0..n_points)
        .map(|i| e_lo + (e_hi - e_lo) * i as f64 / (n_points - 1) as f64)
        .collect();

    let cfg = CalibrationConfig {
        restarts: 2,
        ic_n_energies: 40,
        ic_n_tau: 320,
        max_iter: 1200,
        ..Default::default()
    };

    // Truth kernel on the SAME derived grid the calibrator synthesizes on
    // (e_min = E₀/2, e_max = 2·E_last, cfg.ic_n_energies × cfg.ic_n_tau), so
    // the loop closes exactly and any residual is noise, not grid mismatch.
    let ic_truth = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::SqrtE {
                a0: A0_TRUE,
                a1: A1_TRUE,
            },
            beta: EnergyLaw::Const(BETA_TRUE),
            r: EnergyLaw::Const(R_TRUE),
            burst_sigma_us: None,
            channel_fwhm_us: Some(PSR_TRUE_US),
        },
        cfg.flight_path_m,
        &SynthesisGrid {
            e_min_ev: (energies[0] * 0.5).max(1e-3),
            e_max_ev: energies.last().unwrap() * 2.0,
            n_energies: cfg.ic_n_energies,
            n_tau: cfg.ic_n_tau,
        },
    )
    .unwrap();
    let inst_truth = InstrumentParams {
        resolution: ResolutionFunction::IkedaCarpenter(Arc::new(ic_truth)),
    };
    // FGM Doppler broadening at t_true_k is applied inside forward_model.
    let clean = forward_model(&energies, &sample, Some(&inst_truth)).unwrap();

    // Non-black calibrant check: dip minima in the informative 0.2–0.6 band.
    let t_min = clean.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        (0.15..=0.65).contains(&t_min),
        "calibrant dip minimum T = {t_min} outside the non-black target band"
    );

    // Seeded absolute Gaussian noise. ChaCha12Rng by NAME (review #645 round
    // 2, F5): StdRng's stream is documented unstable across rand versions —
    // a rand upgrade would silently swap the noise realization under this
    // test's fixed tolerances. Naming ChaCha12Rng explicitly pins the exact
    // stream regardless of what StdRng aliases in any rand version (the
    // rand 0.9→0.10 bump left this realization unchanged — both closed-loop
    // cases still pass with no re-baseline).
    let mut rng = ChaCha12Rng::seed_from_u64(642);
    let normal = Normal::new(0.0, NOISE_SIGMA).unwrap();
    let data: Vec<f64> = clean.iter().map(|&t| t + normal.sample(&mut rng)).collect();
    let unc = vec![NOISE_SIGMA; data.len()];

    // --- Step 1: calibrate the bounded IC family at the KNOWN (ρ, T). ---
    let cal = calibrate_resolution(
        ResolutionFamily::IkedaCarpenter { fit_psr: false },
        &energies,
        &data,
        &unc,
        &sample,
        &cfg,
    )
    .unwrap();

    eprintln!(
        "closed_loop(T={t_true_k} K): calibration χ²/dof={:.4}, converged={}, \
         theta={:?}, bounds_hit={:?}",
        cal.chi2_dof, cal.converged, cal.theta, cal.bounds_hit
    );
    assert!(
        (0.7..=1.4).contains(&cal.chi2_dof),
        "calibration χ²/dof = {} not ≈ 1 at T = {t_true_k} K",
        cal.chi2_dof
    );
    assert_eq!(cal.n_free_params, 4, "IC family fits 4 parameters");
    assert!(
        cal.bounds_hit.is_empty(),
        "interior truth must not pin bounds at T = {t_true_k} K: {:?}",
        cal.bounds_hit
    );
    // Decoded recovery (loose windows: β and R act jointly via the storage
    // tail, so their marginals are broad under noise).
    let ResolutionFunction::IkedaCarpenter(ic_cal) = &cal.resolution else {
        panic!("expected an IC resolution");
    };
    let p = ic_cal.params();
    let EnergyLaw::SqrtE { a0, a1 } = p.alpha else {
        panic!("expected a SqrtE alpha law");
    };
    let EnergyLaw::Const(r_cal) = p.r else {
        panic!("expected a Const R law");
    };
    let EnergyLaw::Const(beta_cal) = p.beta else {
        panic!("expected a Const beta law");
    };
    assert!(a1 > 0.0, "α(E) positive by construction (a1 = {a1})");
    assert!(
        (a0 - A0_TRUE).abs() < 0.10,
        "a0 = {a0}, truth {A0_TRUE} (T = {t_true_k} K)"
    );
    assert!(
        (beta_cal - BETA_TRUE).abs() < 0.20,
        "β = {}, truth {BETA_TRUE} (T = {t_true_k} K)",
        beta_cal
    );
    assert!(
        (r_cal - R_TRUE).abs() < 0.12,
        "R = {r_cal}, truth {R_TRUE} (T = {t_true_k} K)"
    );
    let psr_us = p
        .channel_fwhm_us
        .expect("PSR fold must be active at the default config");
    // Approximate compare: the pin travels through ns → µs conversion
    // (350.0 · 1e-3 = 0.35000000000000003).
    assert!(
        (psr_us - PSR_TRUE_US).abs() < 1e-12,
        "PSR pin {psr_us} µs != config default {PSR_TRUE_US} µs"
    );

    // --- Step 2 (the point of #642): pin the calibrated resolution, free T. ---
    // Density fixed at truth; temperature free from a 20 % low start. If the
    // calibrated shape were wrong (the old 2-parameter family), the missing
    // width would re-express here as a ~90 K temperature bias.
    let model = TransmissionFitModel::new(
        energies.clone(),
        vec![ta181],
        0.0, // ignored — temperature_index is set
        Some(Arc::new(InstrumentParams {
            resolution: cal.resolution.clone(),
        })),
        (vec![0], vec![1.0]),
        Some(1), // params[1] = temperature
        None,
    )
    .unwrap();
    let mut params = ParameterSet::new(vec![
        FitParameter::fixed("density", DENSITY_TRUE),
        FitParameter {
            name: "temperature_k".into(),
            value: 0.8 * t_true_k,
            lower: 1.0,
            upper: 3000.0,
            fixed: false,
        },
    ]);
    let lm_cfg = LmConfig {
        max_iter: 200,
        ..LmConfig::default()
    };
    let result = lm::levenberg_marquardt(&model, &data, &unc, &mut params, &lm_cfg).unwrap();

    assert!(
        result.converged,
        "T-refit did not converge at T = {t_true_k} K ({} iterations)",
        result.iterations
    );
    let t_fit = result.params[1];
    // uncertainties covers FREE parameters only; temperature is the only one.
    let sigma_t = result
        .uncertainties
        .expect("temperature uncertainty available")[0];
    assert!(
        sigma_t.is_finite() && sigma_t > 0.0 && sigma_t < 30.0,
        "σ_T = {sigma_t} K not in (0, 30) at T = {t_true_k} K — the degeneracy \
         scale of the old family was ~90 K"
    );
    assert!(
        (t_fit - t_true_k).abs() <= 3.0 * sigma_t,
        "T recovered {t_fit} K vs truth {t_true_k} K exceeds 3σ = {} K",
        3.0 * sigma_t
    );
    // Absolute bias bound alongside the statistical one: the two assertions
    // above alone admit |bias| up to 3·30 = 90 K — see T_BIAS_MAX_K.
    assert!(
        (t_fit - t_true_k).abs() <= T_BIAS_MAX_K,
        "T recovered {t_fit} K vs truth {t_true_k} K exceeds the absolute \
         {T_BIAS_MAX_K} K bias ceiling (old-family degeneracy scale ~90 K)"
    );
    eprintln!(
        "closed_loop(T={t_true_k} K): χ²/dof={:.3}, a0={a0:.3}, a1={a1:.4}, β={:.3}, R={:.3}, \
         T_fit={t_fit:.1} ± {sigma_t:.1} K",
        cal.chi2_dof, beta_cal, r_cal
    );
}

#[test]
fn closed_loop_recovers_temperature_at_300k() {
    closed_loop(300.0);
}

#[test]
fn closed_loop_recovers_temperature_at_1073k() {
    closed_loop(1073.0);
}
