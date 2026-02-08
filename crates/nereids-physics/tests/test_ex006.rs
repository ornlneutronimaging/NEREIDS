//! Integration tests for SAMMY ex006 (Gaussian resolution broadening).
//!
//! ex006 validates the legacy Gaussian-style resolution pathway controlled by
//! DELTAL/DELTAG/DELTAB in SAMMY. This test uses ex006a fixtures as a
//! phase-1d validation anchor.

mod parsers;

use nereids_core::energy::EnergyGrid;
use nereids_core::forward_model::ForwardModelConfig;
use nereids_core::nuclear::{
    Channel, FissionWidths, IsotopeParams, Parameter, RMatrixParameters, Resonance, SpinGroup,
};
use nereids_core::ResolutionFunction;
use nereids_physics::resolution::gaussian::GaussianResolution;
use nereids_physics::rmatrix::compute_0k_cross_sections;
use parsers::{parse_dat_file, parse_lpt_chi_squared, parse_par_file};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn compute_chi_squared(theory: &[f64], data: &[f64], uncertainties: &[f64]) -> f64 {
    assert_eq!(theory.len(), data.len());
    assert_eq!(theory.len(), uncertainties.len());
    theory
        .iter()
        .zip(data.iter())
        .zip(uncertainties.iter())
        .map(|((t, d), u)| {
            let r = (t - d) / u;
            r * r
        })
        .sum()
}

fn ex006_fixture_base_path() -> PathBuf {
    let base = PathBuf::from("tests/fixtures/sammy_reference/ex006");
    assert!(
        base.exists(),
        "Missing ex006 fixtures at {}",
        base.display()
    );
    base
}

fn ex006_config() -> ForwardModelConfig {
    ForwardModelConfig {
        include_potential_scattering: true,
        ..ForwardModelConfig::default()
    }
}

fn parse_ex006_par_grouped(
    path: &std::path::Path,
) -> Result<(Vec<Resonance>, Vec<Resonance>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    fn parse_fixed_11(line: &str, idx: usize) -> Option<f64> {
        let start = idx * 11;
        let end = start + 11;
        let f = line.get(start..end)?.trim();
        if f.is_empty() {
            Some(0.0)
        } else {
            f.parse::<f64>().ok()
        }
    }

    let mut g1 = Vec::new();
    let mut g2 = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Some(e_r) = parse_fixed_11(&line, 0) else {
            continue;
        };
        let Some(gamma_g_mev) = parse_fixed_11(&line, 1) else {
            continue;
        };
        let Some(gamma_n_mev) = parse_fixed_11(&line, 2) else {
            continue;
        };
        let Some(gamma_f1_mev) = parse_fixed_11(&line, 3) else {
            continue;
        };
        let Some(gamma_f2_mev) = parse_fixed_11(&line, 4) else {
            continue;
        };

        // Ex006 resonance records include the spin-group index as the last token.
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 6 {
            continue;
        }
        let Some(last) = tokens.last() else { continue };
        let Ok(group_idx) = last.parse::<usize>() else {
            continue;
        };
        if group_idx != 1 && group_idx != 2 {
            continue;
        }

        let gamma_g = gamma_g_mev / 1000.0;
        let gamma_n = gamma_n_mev / 1000.0;
        let gamma_f1 = gamma_f1_mev / 1000.0;
        let gamma_f2 = gamma_f2_mev / 1000.0;

        let resonance = Resonance {
            energy: Parameter::fixed(e_r),
            gamma_n: Parameter::fixed(gamma_n),
            gamma_g: Parameter::fixed(gamma_g),
            fission: if gamma_f1.abs() > 1e-30 || gamma_f2.abs() > 1e-30 {
                Some(FissionWidths {
                    gamma_f1: Parameter::fixed(gamma_f1),
                    gamma_f2: Parameter::fixed(gamma_f2),
                })
            } else {
                None
            },
        };

        if group_idx == 1 {
            g1.push(resonance);
        } else {
            g2.push(resonance);
        }
    }

    Ok((g1, g2))
}

fn build_ex006_params(group1: Vec<Resonance>, group2: Vec<Resonance>) -> RMatrixParameters {
    let isotope = IsotopeParams {
        name: "PU239".to_string(),
        awr: 239.0,
        abundance: Parameter::fixed(1.0),
        // Fission/cross-section data type for this exercise.
        thickness_cm: 0.0,
        number_density: 0.0,
        spin_groups: vec![
            SpinGroup {
                // ex006a.inp: spin group 1 with J=0.0
                j: 0.0,
                channels: vec![Channel {
                    l: 0,
                    channel_spin: 0.5,
                    radius: 9.54,
                    effective_radius: 9.54,
                }],
                resonances: group1,
            },
            SpinGroup {
                // ex006a.inp: spin group 2 with J=1.0
                j: 1.0,
                channels: vec![Channel {
                    l: 0,
                    channel_spin: 0.5,
                    radius: 9.54,
                    effective_radius: 9.54,
                }],
                resonances: group2,
            },
        ],
    };

    RMatrixParameters {
        isotopes: vec![isotope],
    }
}

fn sorted_experimental_triplets(
    energies: &[f64],
    data: &[f64],
    uncertainties: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rows: Vec<(f64, f64, f64)> = energies
        .iter()
        .copied()
        .zip(data.iter().copied())
        .zip(uncertainties.iter().copied())
        .map(|((e, d), u)| (e, d, u))
        .collect();
    rows.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut e = Vec::with_capacity(rows.len());
    let mut d = Vec::with_capacity(rows.len());
    let mut u = Vec::with_capacity(rows.len());
    for (ee, dd, uu) in rows {
        e.push(ee);
        d.push(dd);
        u.push(uu);
    }
    (e, d, u)
}

#[test]
fn test_ex006a_gaussian_resolution() {
    let base = ex006_fixture_base_path();
    let (group1, group2) = parse_ex006_par_grouped(&base.join("input/ex006a.par"))
        .expect("Failed to parse ex006a.par");
    assert!(!group1.is_empty() && !group2.is_empty());
    let parsed_flat = parse_par_file(&base.join("input/ex006a.par")).expect("Failed flat parse");
    assert_eq!(group1.len() + group2.len(), parsed_flat.len());
    let params = build_ex006_params(group1, group2);
    let exp = parse_dat_file(&base.join("input/ex006a.dat")).expect("Failed to parse ex006a.dat");
    let (energies, data, uncertainties) =
        sorted_experimental_triplets(&exp.energies, &exp.data, &exp.uncertainties);
    let energy_grid = EnergyGrid::new(energies.clone()).expect("Failed to build energy grid");

    let xs_0k = compute_0k_cross_sections(&energy_grid, &params, &ex006_config())
        .expect("0K cross section computation failed");
    let sigma_fission_0k: Vec<f64> = xs_0k.iter().map(|cs| cs.fission).collect();

    let chi2_no_res = compute_chi_squared(&sigma_fission_0k, &data, &uncertainties);

    let sigma_scan = [
        0.10_f64, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80,
    ];
    let mut best = (f64::INFINITY, 0.0_f64);
    for sigma in sigma_scan {
        let model = GaussianResolution { sigma };
        let broadened = model
            .convolve(&energy_grid, &sigma_fission_0k)
            .expect("resolution convolution failed");
        let chi2 = compute_chi_squared(&broadened, &data, &uncertainties);
        if chi2 < best.0 {
            best = (chi2, sigma);
        }
    }

    let expected = parse_lpt_chi_squared(&base.join("expected/ex006aa.lpt"))
        .expect("Failed to parse ex006aa.lpt")
        .chi_squared;

    eprintln!(
        "ex006a: chi2_no_res={:.3}, best_chi2={:.3} at sigma={:.3}, sammy={:.3}",
        chi2_no_res, best.0, best.1, expected
    );

    // Broadening should materially improve this benchmark over a no-resolution model.
    assert!(
        best.0 < chi2_no_res,
        "expected Gaussian resolution to improve chi²: no_res={chi2_no_res:.3}, best={:.3}",
        best.0
    );

    // Keep this as a coarse anchoring check for phase-1d. Exact SAMMY parity
    // requires time-domain DELTAL/DELTAG/DELTAB mapping that is not yet in the
    // current constant-sigma energy-domain implementation.
    let improvement_factor = chi2_no_res / best.0.max(1e-30);
    assert!(
        improvement_factor > 20.0,
        "expected substantial EX006 improvement from Gaussian broadening: no_res={chi2_no_res:.3}, best={:.3}, factor={improvement_factor:.2}",
        best.0,
    );
}
