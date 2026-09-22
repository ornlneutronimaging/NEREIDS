//! The synthetic fixture must contain the physics the KL fit is meant to
//! recover. Each test below asserts one injected quantity actually moves the
//! counts — a fixture that silently omits one would let a recovery test pass
//! while measuring nothing.

use std::sync::Arc;

use nereids_endf::resonance::test_support::u238_single_resonance;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::ResolutionFunction;
use nereids_pipeline::synthetic::{Truth, detector_time_edges_around};

const FLIGHT_PATH_M: f64 = 25.0;
const TIMING_OFFSET_US: f64 = 0.0;
const WINDOW_PAD_BINS: usize = 64;
const NODES_PER_BIN: usize = 16;
/// U-238 areal density (at/b), sized so the 6.674 eV dip is deep enough to
/// carry information but far from black.
const DENSITY: f64 = 5.0e-4;
const TEMPERATURE_K: f64 = 293.6;

/// Grid straddling the U-238 6.674 eV resonance.
fn energies() -> Vec<f64> {
    (0..201).map(|i| 5.0 + f64::from(i) * 0.02).collect()
}

fn resolution(energies: &[f64]) -> ResolutionFunction {
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
            e_min_ev: (energies[0] * 0.5).max(1e-3),
            e_max_ev: energies.last().unwrap() * 2.0,
            n_energies: 32,
            n_tau: 256,
        },
    )
    .expect("valid IC truth");
    ResolutionFunction::IkedaCarpenter(Arc::new(ic))
}

fn truth() -> Truth {
    let grid = energies();
    let edges = detector_time_edges_around(&grid, FLIGHT_PATH_M, TIMING_OFFSET_US, WINDOW_PAD_BINS);
    let n_bins = edges.len() - 1;
    Truth {
        resolution: resolution(&grid),
        detector_time_edges_us: edges,
        source_bins: Some(WINDOW_PAD_BINS..n_bins - WINDOW_PAD_BINS),
        nodes_per_bin: NODES_PER_BIN,
        flight_path_m: FLIGHT_PATH_M,
        t0_us: 0.0,
        l_scale: 1.0,
        temperature_k: TEMPERATURE_K,
        isotopes: vec![u238_single_resonance()],
        timing_offset_us: TIMING_OFFSET_US,
        open_beam_counts_per_bin: 2.0e4,
        open_background_per_bin: 50.0,
        sample_background_per_bin: 120.0,
    }
}

#[test]
fn detector_time_edges_ascend_and_bracket_the_grid() {
    let truth = truth();
    let edges = &truth.detector_time_edges_us;
    assert_eq!(edges.len(), truth.fluence_per_bin().len() + 1);
    assert!(
        edges.windows(2).all(|w| w[0] < w[1]),
        "detector time edges must ascend"
    );

    let kl = nereids_physics::resolution::TOF_FACTOR * FLIGHT_PATH_M;
    for e in energies() {
        let tof = TIMING_OFFSET_US + kl / e.sqrt();
        assert!(
            tof > edges[0] && tof < *edges.last().unwrap(),
            "flight time {tof} for {e} eV falls outside the acquisition window"
        );
    }
}

/// The sample arm must be attenuated where the resonance is, and the two arms
/// must differ by more than their backgrounds.
#[test]
fn the_sample_arm_carries_the_resonance() {
    let truth = truth();
    let m = truth.measure(&[DENSITY], 1);

    let ratio: Vec<f64> = m
        .expected_sample
        .iter()
        .zip(&m.expected_open)
        .map(|(s, o)| s / o)
        .collect();
    let deepest = ratio.iter().copied().fold(f64::INFINITY, f64::min);
    let shallowest = ratio.iter().copied().fold(0.0_f64, f64::max);

    assert!(
        deepest < 0.8,
        "no resonance dip in the fixture: deepest ratio {deepest:.4}"
    );
    assert!(
        deepest > 0.05,
        "dip is black, so it carries no width information: {deepest:.4}"
    );
    assert!(
        shallowest - deepest > 0.1,
        "ratio is nearly flat ({deepest:.4}..{shallowest:.4}); the fixture \
         would not constrain density"
    );
}

/// The two arms carry DIFFERENT backgrounds. With no sample present the ratio
/// must still not be one, because the sample arm's own background remains.
#[test]
fn the_two_arms_carry_different_backgrounds() {
    let truth = truth();
    let m = truth.measure(&[0.0], 2);

    for (i, (s, o)) in m.expected_sample.iter().zip(&m.expected_open).enumerate() {
        let ratio = s / o;
        assert!(
            ratio > 1.0,
            "bin {i}: with no sample the arms differ only by background, and \
             the sample arm's is larger, so the ratio must exceed one; got \
             {ratio:.6}"
        );
    }

    // And the difference is exactly the background difference, which is what
    // makes B_s and B_o separately recoverable rather than only their sum.
    let expected_gap = truth.sample_background_per_bin - truth.open_background_per_bin;
    for (s, o) in m.expected_sample.iter().zip(&m.expected_open) {
        assert!(
            (s - o - expected_gap).abs() < 1e-6,
            "arm difference {} is not the injected background gap {expected_gap}",
            s - o
        );
    }
}

/// Resolution must actually broaden the recorded dip. Without this the
/// fixture would be consistent with a model that ignores the kernel.
#[test]
fn the_resolution_kernel_broadens_the_recorded_dip() {
    let grid = energies();
    let sharp = Truth {
        resolution: {
            let ic = IkedaCarpenter::new(
                IkedaCarpenterParams {
                    alpha: EnergyLaw::SqrtE { a0: 5.0, a1: 0.05 },
                    beta: EnergyLaw::Const(0.02),
                    r: EnergyLaw::Const(0.0),
                    burst_sigma_us: None,
                    channel_fwhm_us: None,
                },
                FLIGHT_PATH_M,
                &SynthesisGrid {
                    e_min_ev: (grid[0] * 0.5).max(1e-3),
                    e_max_ev: grid.last().unwrap() * 2.0,
                    n_energies: 32,
                    n_tau: 256,
                },
            )
            .expect("valid sharp IC");
            ResolutionFunction::IkedaCarpenter(Arc::new(ic))
        },
        ..truth()
    };

    let depth = |t: &Truth| {
        let m = t.measure(&[DENSITY], 3);
        m.expected_sample
            .iter()
            .zip(&m.expected_open)
            .map(|(s, o)| s / o)
            .fold(f64::INFINITY, f64::min)
    };

    let broad_depth = depth(&truth());
    let sharp_depth = depth(&sharp);
    assert!(
        broad_depth > sharp_depth,
        "a broader kernel must fill in the dip: broad {broad_depth:.4} is not \
         shallower than sharp {sharp_depth:.4}"
    );
}

/// Noise is Poisson around the expectation, reproducible for a seed and
/// different between seeds. A fixture whose "noise" was deterministic would
/// make every uncertainty claim meaningless.
#[test]
fn counts_are_a_reproducible_poisson_draw() {
    let truth = truth();
    let a = truth.measure(&[DENSITY], 7);
    let b = truth.measure(&[DENSITY], 7);
    let c = truth.measure(&[DENSITY], 8);

    assert_eq!(a.sample_counts, b.sample_counts, "seed 7 must reproduce");
    assert_ne!(a.sample_counts, c.sample_counts, "seed 8 must differ");

    // Counts are integers drawn around the expectation, and the total sits
    // within a few sigma of it.
    assert!(a.sample_counts.iter().all(|&n| n.fract() == 0.0));
    let observed: f64 = a.sample_counts.iter().sum();
    let expected: f64 = a.expected_sample.iter().sum();
    let sigma = expected.sqrt();
    assert!(
        (observed - expected).abs() < 5.0 * sigma,
        "total {observed} is more than 5 sigma from {expected}"
    );
}

#[test]
fn the_acquisition_window_keeps_the_counts_it_was_given() {
    let truth = truth();
    let m = truth.measure(&[DENSITY], 5);

    let offered: f64 = m.incident_fluence_weights.iter().sum();
    let (open_loss, sample_loss) = m.window_loss;
    assert!(
        open_loss / offered < 1.0e-3,
        "open arm loses {open_loss:.3e} of {offered:.3e} outside the window"
    );
    assert!(
        sample_loss <= open_loss,
        "the sample arm cannot lose more than the open arm: \
         {sample_loss:.3e} against {open_loss:.3e}"
    );

    assert_eq!(m.open_beam_counts.len(), m.sample_counts.len());
    assert_eq!(
        m.detector_time_edges_us.len(),
        m.sample_counts.len() + 1,
        "one more edge than bins"
    );
    assert!(
        m.sample_counts.len() > energies().len(),
        "the acquisition window must be wider than the region of interest"
    );
    assert_eq!(
        m.true_energies_ev.len(),
        m.incident_fluence_weights.len() * NODES_PER_BIN,
        "nodes_per_bin true energies per detector bin"
    );
    // At the identity energy scale the true energies ARE the nominal ones.
    assert_eq!(m.true_energies_ev, truth.nominal_energies_ev(NODES_PER_BIN));
}

/// A non-identity energy scale must move the true energies, or a calibration
/// recovery test built on this fixture would be measuring nothing.
#[test]
fn the_energy_scale_truth_moves_the_true_energies() {
    let shifted = Truth {
        t0_us: 0.35,
        l_scale: 1.004,
        ..truth()
    };
    let identity = truth();

    let a = identity.true_energies_ev();
    let b = shifted.true_energies_ev();
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs() / x)
        .fold(0.0_f64, f64::max);
    assert!(
        worst > 1.0e-3,
        "energy-scale truth barely moved the grid ({worst:.2e}); calibration \
         would be unrecoverable from it"
    );
}

/// Sixteen pixels, each with its own density and its own noise.
#[test]
fn a_detector_gives_one_measurement_per_pixel() {
    let truth = truth();
    let densities: Vec<Vec<f64>> = (0..16)
        .map(|p| vec![DENSITY * (1.0 + 0.05 * f64::from(p))])
        .collect();
    let pixels = truth.measure_detector(&densities, 100);

    assert_eq!(pixels.len(), 16);
    let depth = |m: &nereids_pipeline::synthetic::Measurement| {
        m.expected_sample
            .iter()
            .zip(&m.expected_open)
            .map(|(s, o)| s / o)
            .fold(f64::INFINITY, f64::min)
    };
    // Denser pixels attenuate more, so the dip deepens monotonically.
    for pair in pixels.windows(2) {
        assert!(
            depth(&pair[1]) < depth(&pair[0]),
            "a denser pixel must give a deeper dip"
        );
    }
    assert_ne!(
        pixels[0].sample_counts, pixels[1].sample_counts,
        "each pixel must get its own noise realization"
    );
}
