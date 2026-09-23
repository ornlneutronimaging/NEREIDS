use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_endf::resonance::test_support::synthetic_isotope;
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    ResolutionFunction, ResolutionParams, TOF_FACTOR, TabulatedResolution,
};
use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};

const FLIGHT_PATH_M: f64 = 25.0;
const CHANNEL_US: f64 = 0.5;
const SUBDIVISION: i64 = 64;
const GAMMA_N_EV: f64 = 0.05;
const GAMMA_G_EV: f64 = 0.06;
const DENSITY: f64 = 1.0e-6;

/// SAMMY's default IPTDOP; resonance points sit `2 / (IPTDOP + 5)` of the
/// resonance's width apart.
///
/// SAMMY Ref: `inp/InputInfoData.cpp` line 23, `dat/mdat4.f90` Fspken line 310
const SAMMY_IPTDOP: f64 = 9.0;

fn channels(lo: f64, hi: f64, channel_us: f64) -> Vec<f64> {
    let tof = |e: f64| TOF_FACTOR * FLIGHT_PATH_M / e.sqrt();
    let k_lo = (tof(hi) / channel_us).ceil() as i64;
    let k_hi = (tof(lo) / channel_us).floor() as i64;
    (k_lo..=k_hi)
        .rev()
        .map(|k| (TOF_FACTOR * FLIGHT_PATH_M / (k as f64 * channel_us)).powi(2))
        .collect()
}

fn blurs() -> Vec<(&'static str, ResolutionFunction)> {
    let ikeda_carpenter = IkedaCarpenter::new(
        IkedaCarpenterParams {
            alpha: EnergyLaw::SqrtE { a0: 0.35, a1: 0.05 },
            beta: EnergyLaw::Const(0.25),
            r: EnergyLaw::Const(0.15),
            burst_sigma_us: None,
            channel_fwhm_us: Some(0.35),
        },
        FLIGHT_PATH_M,
        &SynthesisGrid {
            e_min_ev: 45.0,
            e_max_ev: 220.0,
            n_energies: 32,
            n_tau: 256,
        },
    )
    .expect("valid IC model");
    let offsets: Vec<f64> = (0..81).map(|j| -0.987 + 0.05 * f64::from(j)).collect();
    let weights = offsets
        .iter()
        .map(|&dt| (dt + 1.0) * (-(dt + 1.0) / 0.8).exp())
        .collect();
    let kernel = (offsets, weights);
    let tabulated = TabulatedResolution::from_kernels(
        vec![50.0, 200.0],
        vec![kernel.clone(), kernel],
        FLIGHT_PATH_M,
    )
    .expect("valid table");
    vec![
        (
            "gaussian",
            ResolutionFunction::Gaussian(
                ResolutionParams::new(FLIGHT_PATH_M, 1.0, 0.0, 0.0).expect("valid"),
            ),
        ),
        (
            "gaussian with exponential tail",
            ResolutionFunction::Gaussian(
                ResolutionParams::new(FLIGHT_PATH_M, 1.0, 0.0, 1.0).expect("valid"),
            ),
        ),
        (
            "ikeda-carpenter",
            ResolutionFunction::IkedaCarpenter(Arc::new(ikeda_carpenter)),
        ),
        (
            "tabulated",
            ResolutionFunction::Tabulated(Arc::new(tabulated)),
        ),
    ]
}

fn sammy_placement_area_error(width: f64, spacing: f64) -> f64 {
    let step = 2.0 * width / (SAMMY_IPTDOP + 5.0);
    let mut x: Vec<f64> = (0..=(SAMMY_IPTDOP as usize + 5))
        .map(|k| -width + step * k as f64)
        .collect();
    let reach = 400.0 * width;
    for side in [-1.0, 1.0] {
        let (mut edge, mut h) = (width, step);
        while h < spacing {
            h *= 2.0;
            edge += h;
            x.push(side * edge);
        }
        while edge < reach {
            edge += spacing;
            x.push(side * edge);
        }
    }
    x.sort_by(f64::total_cmp);
    let lorentzian = |e: f64| 1.0 / (1.0 + (2.0 * e / width).powi(2));
    let straight: f64 = x
        .windows(2)
        .map(|p| 0.5 * (p[1] - p[0]) * (lorentzian(p[0]) + lorentzian(p[1])))
        .sum();
    let exact = width * (2.0 * x[x.len() - 1] / width).atan();
    straight / exact - 1.0
}

fn transmission(
    energies: &[f64],
    isotopes: &[ResonanceData],
    resolution: &ResolutionFunction,
) -> Vec<f64> {
    let sample = SampleParams::new(
        0.0,
        isotopes.iter().map(|rd| (rd.clone(), DENSITY)).collect(),
    )
    .expect("valid sample");
    forward_model(
        energies,
        &sample,
        Some(&InstrumentParams {
            resolution: resolution.clone(),
        }),
    )
    .expect("forward model")
}

fn dip_sum_error(coarse: &[f64], t_coarse: &[f64], dense: &[f64], t_dense: &[f64]) -> f64 {
    let (mut on_coarse, mut on_dense) = (0.0, 0.0);
    for (i, &e) in coarse.iter().enumerate() {
        let j = dense.partition_point(|&d| d < e * (1.0 - 1.0e-12));
        assert!(
            (dense[j] - e).abs() <= 1.0e-12 * e,
            "the dense grid must contain every coarse point"
        );
        on_coarse += 1.0 - t_coarse[i];
        on_dense += 1.0 - t_dense[j];
    }
    assert!(on_dense > 1.0e-4, "the dip must be deep enough to measure");
    on_coarse / on_dense - 1.0
}

#[test]
fn a_narrow_resonance_is_broadened_as_on_a_dense_grid() {
    let coarse = channels(90.0, 110.0, CHANNEL_US);
    let dense = channels(90.0, 110.0, CHANNEL_US / SUBDIVISION as f64);
    let width = GAMMA_N_EV + GAMMA_G_EV;
    let above = coarse.partition_point(|&e| e < 100.0);
    let (c_lo, c_hi) = (coarse[above - 1], coarse[above]);
    assert!(
        c_hi - c_lo > 4.0 * width,
        "the channels must be much wider than the resonance for this to test anything, \
         got {} eV",
        c_hi - c_lo
    );
    let tolerance = sammy_placement_area_error(width, c_hi - c_lo).abs();

    let sweep = (0..12).map(|k| c_lo + (c_hi - c_lo) * (f64::from(k) + 0.5) / 12.0);
    let edges_on_channels = [c_lo, c_lo + width * 1.001, c_hi - width * 1.001];
    let mut failures = Vec::new();
    for e_res in sweep.chain(edges_on_channels) {
        let isotope = [synthetic_isotope(72, 178, e_res, GAMMA_N_EV, GAMMA_G_EV)];
        for (label, resolution) in blurs() {
            let t_coarse = transmission(&coarse, &isotope, &resolution);
            let t_dense = transmission(&dense, &isotope, &resolution);
            let error = dip_sum_error(&coarse, &t_coarse, &dense, &t_dense);
            if error.abs() > tolerance {
                failures.push(format!("{label} at {e_res:.4} eV: dip off by {error:+.3e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "coarse channels misrepresent a resonance narrower than their spacing by more \
         than SAMMY's placement does ({tolerance:.3e}): {}",
        failures.join("; ")
    );
}

#[test]
fn overlapping_resonances_give_the_same_transmission_in_either_order() {
    let coarse: Vec<f64> = (0..5).map(|k| 90.0 + 5.0 * f64::from(k)).collect();
    let dense: Vec<f64> = (0..=4096)
        .map(|k| 90.0 + 20.0 * f64::from(k) / 4096.0)
        .collect();
    let narrow = synthetic_isotope(72, 178, 100.0, 0.005, 0.005);
    let broad = synthetic_isotope(72, 178, 100.8, 0.5, 0.5);
    let resolution = ResolutionFunction::IkedaCarpenter(Arc::new(
        IkedaCarpenter::new(
            IkedaCarpenterParams {
                alpha: EnergyLaw::Const(1.0),
                beta: EnergyLaw::Const(0.25),
                r: EnergyLaw::Const(0.0),
                burst_sigma_us: None,
                channel_fwhm_us: None,
            },
            FLIGHT_PATH_M,
            &SynthesisGrid {
                e_min_ev: 45.0,
                e_max_ev: 220.0,
                n_energies: 32,
                n_tau: 256,
            },
        )
        .expect("valid IC model"),
    ));

    let forward = transmission(&coarse, &[narrow.clone(), broad.clone()], &resolution);
    let reversed = transmission(&coarse, &[broad.clone(), narrow.clone()], &resolution);
    for (i, (&a, &b)) in forward.iter().zip(&reversed).enumerate() {
        assert!(
            (a - b).abs() <= 1.0e-12,
            "the order of the isotopes moved T({}) from {a} to {b}",
            coarse[i]
        );
    }
    let t_dense = transmission(&dense, &[narrow, broad], &resolution);
    let error = dip_sum_error(&coarse, &forward, &dense, &t_dense);
    let tolerance = sammy_placement_area_error(0.01, 5.0)
        .abs()
        .max(sammy_placement_area_error(1.0, 5.0).abs());
    assert!(
        error.abs() <= tolerance,
        "dip off by {error:+.3e}, more than SAMMY's placement ({tolerance:.3e})"
    );
}
