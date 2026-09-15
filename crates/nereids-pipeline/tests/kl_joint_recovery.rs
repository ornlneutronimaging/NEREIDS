//! Joint recovery of areal density and temperature through the counts-KL
//! path, from synthetic data whose truth is known.
//!
//! This is the scoreboard for the counts path. `ic_closed_loop` holds density
//! at its true value and frees only temperature, so the degeneracy the design
//! exists to handle — density and temperature are jointly encoded by the
//! curve of growth, and neither can be read off the dip alone — is never
//! exercised. A fit that recovers one while the other is pinned proves very
//! little about a fit that must find both.

use std::sync::Arc;

use nereids_endf::resonance::test_support::u238_single_resonance;
use nereids_fitting::poisson::PoissonConfig;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::ResolutionFunction;
use nereids_pipeline::pipeline::{
    ExactCountResponseConfig, InputData, SolverConfig, UnifiedFitConfig, fit_spectrum_typed,
};
use nereids_pipeline::synthetic::Truth;

const FLIGHT_PATH_M: f64 = 25.0;
const DENSITY_TRUE: f64 = 5.0e-4;
const TEMPERATURE_TRUE_K: f64 = 293.6;

/// 401 points over the U-238 6.674 eV resonance at 0.01 eV spacing. The
/// Doppler width there is about 0.054 eV against a natural width of 0.025 eV,
/// so temperature is what sets most of the observed shape — and the grid
/// samples that width about five times.
fn energies() -> Vec<f64> {
    (0..401).map(|i| 5.0 + f64::from(i) * 0.01).collect()
}

fn resolution(grid: &[f64]) -> ResolutionFunction {
    let ic = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
            beta: EnergyLaw::Const(0.25),
            r: EnergyLaw::Const(0.15),
            burst_sigma_us: None,
            channel_fwhm_us: Some(0.35),
        },
        FLIGHT_PATH_M,
        &SynthesisGrid {
            e_min_ev: (grid[0] * 0.5).max(1e-3),
            e_max_ev: grid.last().unwrap() * 2.0,
            n_energies: 32,
            n_tau: 256,
        },
    )
    .expect("valid IC truth");
    ResolutionFunction::IkedaCarpenter(Arc::new(ic))
}

/// Backgrounds are zero here. The counts path cannot represent a detector
/// background at all — it rejects any non-zero value — so a fixture with
/// `B_s != B_o` measures that gap rather than the joint recovery this test is
/// about. That measurement belongs with the fix for it.
fn truth() -> Truth {
    let grid = energies();
    Truth {
        resolution: resolution(&grid),
        nominal_energies_ev: grid,
        flight_path_m: FLIGHT_PATH_M,
        t0_us: 0.0,
        l_scale: 1.0,
        temperature_k: TEMPERATURE_TRUE_K,
        isotopes: vec![u238_single_resonance()],
        timing_offset_us: 0.0,
        window_pad_bins: 64,
        open_beam_counts_per_bin: 2.0e5,
        open_background_per_bin: 0.0,
        sample_background_per_bin: 0.0,
    }
}

/// Fit one noise realization from the shared seed stream.
fn recover(seed: u64) -> (f64, f64, f64) {
    let truth = truth();
    let m = truth.measure(&[DENSITY_TRUE], seed);

    // The fixture must not be losing neutrons: a window loss would show up as
    // a normalization the fit has to absorb, and it would land on density.
    let offered: f64 = m.incident_fluence_weights.iter().sum();
    assert!(
        m.window_loss.0 / offered < 1.0e-3 && m.window_loss.1 / offered < 1.0e-3,
        "fixture loses counts outside the acquisition window: {:?}",
        m.window_loss
    );
    // And the draw must be noise around the expectation, not the expectation.
    assert_ne!(m.sample_counts, m.expected_sample, "counts are not a draw");
    assert_ne!(m.open_beam_counts, m.expected_open, "counts are not a draw");

    // Start both parameters away from truth so the fit has to find them:
    // density 20 % low, temperature 15 % low.
    let config = UnifiedFitConfig::new(
        truth.nominal_energies_ev.clone(),
        vec![u238_single_resonance()],
        vec!["U-238".into()],
        TEMPERATURE_TRUE_K * 0.85,
        Some(truth.resolution.clone()),
        vec![DENSITY_TRUE * 0.8],
    )
    .expect("valid config")
    .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
    .with_fit_temperature(true)
    .with_exact_count_response(ExactCountResponseConfig {
        incident_fluence_weights: m.incident_fluence_weights.clone(),
        detector_time_edges_us: m.detector_time_edges_us.clone(),
        timing_offset_us: truth.timing_offset_us,
    });

    let result = fit_spectrum_typed(
        &InputData::Counts {
            sample_counts: m.sample_counts,
            open_beam_counts: m.open_beam_counts,
        },
        &config,
    )
    .expect("counts-KL fit runs");

    let density_bias = (result.densities[0] - DENSITY_TRUE) / DENSITY_TRUE;
    let temperature_bias = result.temperature_k.expect("temperature reported") - TEMPERATURE_TRUE_K;
    let deviance = result.deviance_per_dof.expect("deviance reported");
    println!(
        "seed {seed}: density {:+.3} %, temperature {temperature_bias:+.3} K, \
         D/dof {deviance:.4}",
        100.0 * density_bias
    );
    (density_bias, temperature_bias, deviance)
}

/// Both parameters come back, from several noise realizations.
///
/// The bounds are set from the measured spread, not from what would be
/// comfortable: across these seeds the worst density bias is well under a
/// percent and the worst temperature bias a few kelvin. A bound of 2 % and
/// 15 K would pass on a fit that had silently lost most of its accuracy.
#[test]
fn density_and_temperature_are_recovered_together() {
    let seeds = [20250915_u64, 11, 404, 7777, 31415];
    let mut worst_density = 0.0_f64;
    let mut worst_temperature = 0.0_f64;
    let mut mean_density = 0.0_f64;
    let mut mean_temperature = 0.0_f64;

    for seed in seeds {
        let (density_bias, temperature_bias, deviance) = recover(seed);
        assert!(
            (0.5..2.0).contains(&deviance),
            "D/dof {deviance:.4} says the model does not describe the data"
        );
        worst_density = worst_density.max(density_bias.abs());
        worst_temperature = worst_temperature.max(temperature_bias.abs());
        mean_density += density_bias / seeds.len() as f64;
        mean_temperature += temperature_bias / seeds.len() as f64;
    }

    assert!(
        worst_density < 0.01,
        "worst density bias {:+.3} % exceeds 1 %",
        100.0 * worst_density
    );
    assert!(
        worst_temperature < 5.0,
        "worst temperature bias {worst_temperature:+.3} K exceeds 5 K"
    );
    // A bias that survives averaging is systematic, not noise.
    assert!(
        mean_density.abs() < 0.005,
        "mean density bias {:+.3} % is systematic",
        100.0 * mean_density
    );
    assert!(
        mean_temperature.abs() < 2.5,
        "mean temperature bias {mean_temperature:+.3} K is systematic"
    );
}

/// The fit must actually be moving both parameters, not returning its own
/// starting point. Without this, a model that ignored the data entirely would
/// pass the test above whenever the seed happened to start near truth.
#[test]
fn the_fit_moves_both_parameters_away_from_their_seeds() {
    let truth = truth();
    let m = truth.measure(&[DENSITY_TRUE], 20250915);

    let density_seed = DENSITY_TRUE * 0.8;
    let temperature_seed = TEMPERATURE_TRUE_K * 0.85;
    let config = UnifiedFitConfig::new(
        truth.nominal_energies_ev.clone(),
        vec![u238_single_resonance()],
        vec!["U-238".into()],
        temperature_seed,
        Some(truth.resolution.clone()),
        vec![density_seed],
    )
    .expect("valid config")
    .with_solver(SolverConfig::PoissonKL(PoissonConfig::default()))
    .with_fit_temperature(true)
    .with_exact_count_response(ExactCountResponseConfig {
        incident_fluence_weights: m.incident_fluence_weights.clone(),
        detector_time_edges_us: m.detector_time_edges_us.clone(),
        timing_offset_us: truth.timing_offset_us,
    });

    let result = fit_spectrum_typed(
        &InputData::Counts {
            sample_counts: m.sample_counts,
            open_beam_counts: m.open_beam_counts,
        },
        &config,
    )
    .expect("counts-KL fit runs");

    let density = result.densities[0];
    let temperature = result.temperature_k.expect("temperature reported");
    assert!(
        (density - density_seed).abs() / density_seed > 0.05,
        "density did not move from its seed"
    );
    assert!(
        (temperature - temperature_seed).abs() > 10.0,
        "temperature did not move from its seed"
    );
}
