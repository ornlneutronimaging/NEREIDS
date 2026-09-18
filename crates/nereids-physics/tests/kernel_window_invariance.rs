//! A broadened value must not depend on where the data window ends.
//!
//! The broadened spectrum at a target energy is built from theory within the
//! kernel's reach of that energy. Whether the caller happened to stop
//! collecting data just past it is not a property of the sample, so widening
//! the window must leave the shared targets alone.

use nereids_endf::resonance::test_support::synthetic_isotope;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};

const FLIGHT_PATH_M: f64 = 25.0;
const E_LO: f64 = 5.0;
const E_HI: f64 = 9.0;
const STEP: f64 = 0.01;

/// The VENUS-like moderator the pipeline's own joint-recovery fixture uses.
fn ic_resolution() -> ResolutionFunction {
    let params = IkedaCarpenterParams {
        alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
        beta: EnergyLaw::Const(0.25),
        r: EnergyLaw::Const(0.15),
        burst_sigma_us: None,
        channel_fwhm_us: Some(0.35),
    };
    ResolutionFunction::IkedaCarpenter(std::sync::Arc::new(
        IkedaCarpenter::new(
            params,
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: E_LO * 0.5,
                e_max_ev: 64.0,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model"),
    ))
}

fn grid(e_hi: f64) -> Vec<f64> {
    let n = ((e_hi - E_LO) / STEP).round() as usize + 1;
    (0..n).map(|i| E_LO + i as f64 * STEP).collect()
}

fn transmission(energies: &[f64], resolution: &ResolutionFunction) -> Vec<f64> {
    // A resonance placed AT the window edge: the case the truncation hurts
    // most, because the kernel there reaches entirely outside the data.
    let iso = synthetic_isotope(72, 178, E_HI, 0.05, 0.06);
    let sample = SampleParams::new(300.0, vec![(iso, 2.0e-3)]).expect("valid sample");
    forward_model(
        energies,
        &sample,
        Some(&InstrumentParams {
            resolution: resolution.clone(),
        }),
    )
    .expect("forward model")
}

/// Extending the window past the kernel's reach leaves the shared targets
/// unchanged.
///
/// The Ikeda-Carpenter kernel is causal on the emission clock, so its offsets
/// are positive and it gathers theory from HIGHER energy only. Without a
/// boundary extension the top of the window has nothing to gather and the
/// surviving fragment is renormalized to full weight, so the model there is
/// built from a kernel that is narrower, lighter and mis-centred.
#[test]
fn a_wider_window_does_not_change_the_model_inside_it() {
    let resolution = ic_resolution();
    let narrow = grid(E_HI);
    let reach = resolution.boundary_reach_ev(E_LO, E_HI).1;
    assert!(
        reach.is_finite() && reach > 10.0 * STEP,
        "the kernel must reach past several grid steps for this to test anything, got {reach}"
    );
    let wide = grid(E_HI + 2.0 * reach);

    let t_narrow = transmission(&narrow, &resolution);
    let t_wide = transmission(&wide, &resolution);

    let mut worst = 0.0_f64;
    let mut worst_at = 0.0_f64;
    for (i, &e) in narrow.iter().enumerate() {
        let d = (t_narrow[i] - t_wide[i]).abs() / t_wide[i].abs().max(1e-300);
        if d > worst {
            worst = d;
            worst_at = e;
        }
    }
    // The two runs convolve on different point sets - the wide window carries
    // its own data where the narrow one carries extension points - so what is
    // left is the trapezoid's own discretization, not lost kernel mass.
    // Truncation shows up here four orders of magnitude above it.
    assert!(
        worst < 1.0e-4,
        "the model moved by {worst:.3e} at {worst_at} eV when the window was widened; \
         the value at a target must come from the kernel's reach, not from where \
         the data happens to stop"
    );
}

/// The working grid carries essentially all of the kernel's mass at every
/// data point.
///
/// Data-independent: it asks what fraction of the kernel survives the grid,
/// not what the spectrum does. A kernel renormalized after losing mass is a
/// different kernel from the one the file describes.
#[test]
fn the_working_grid_carries_the_whole_kernel_at_every_data_point() {
    let resolution = ic_resolution();
    let data = grid(E_HI);
    let layout = nereids_physics::transmission::resolution_working_grid(
        &data,
        Some(&InstrumentParams {
            resolution: resolution.clone(),
        }),
        &[],
    )
    .expect("working grid");

    assert!(
        !layout.is_identity(),
        "an Ikeda-Carpenter kernel must get a working grid wider than the data"
    );
    let work = &layout.energies;
    let (w_lo, w_hi) = (work[0], work[work.len() - 1]);

    let ResolutionFunction::IkedaCarpenter(ic) = &resolution else {
        unreachable!()
    };
    let table = ic.tabulated();
    for &e in &data {
        let (below, above) = table.kernel_support_directional_ev(e);
        assert!(
            e - below >= w_lo - 1e-9,
            "at {e} eV the kernel reaches down to {} but the working grid starts at {w_lo}",
            e - below
        );
        assert!(
            !above.is_finite() || e + above <= w_hi + 1e-9,
            "at {e} eV the kernel reaches up to {} but the working grid ends at {w_hi}",
            e + above
        );
    }
}
