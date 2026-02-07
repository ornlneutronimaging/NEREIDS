//! Integration tests for SAMMY ex003 validation.
//!
//! Tests 0K Reich-Moore R-matrix cross sections against SAMMY reference outputs.
//! Six test variants (a, c, e, f, x, t) validate different cross section types.

mod parsers;

use nereids_core::energy::EnergyGrid;
use nereids_core::forward_model::{ForwardModel, ForwardModelConfig};
use nereids_core::nuclear::{Channel, IsotopeParams, Parameter, RMatrixParameters, SpinGroup};
use nereids_physics::pipeline::DefaultForwardModel;
use nereids_physics::rmatrix::compute_0k_cross_sections;
use parsers::{parse_dat_file, parse_lpt_chi_squared, parse_par_file};
use std::path::PathBuf;

/// Compute chi-squared: χ² = Σ [(theory - data) / uncertainty]²
fn compute_chi_squared(theory: &[f64], data: &[f64], uncertainties: &[f64]) -> f64 {
    assert_eq!(theory.len(), data.len());
    assert_eq!(theory.len(), uncertainties.len());

    theory
        .iter()
        .zip(data.iter())
        .zip(uncertainties.iter())
        .map(|((t, d), u)| {
            let residual = (t - d) / u;
            residual * residual
        })
        .sum()
}

fn chi2_error(computed: f64, expected: f64) -> (f64, &'static str) {
    let scale = computed.abs().max(expected.abs());
    if scale < CHI2_ABS_CUTOFF {
        ((computed - expected).abs(), "absolute")
    } else {
        ((computed - expected).abs() / scale, "relative")
    }
}

const CHI2_ABS_CUTOFF: f64 = 1e-6;
const CHI2_ABS_TOLERANCE: f64 = 1e-6;
const CHI2_REL_TOLERANCE: f64 = 2e-2;

fn ex003_config() -> ForwardModelConfig {
    ForwardModelConfig {
        include_potential_scattering: true,
        ..ForwardModelConfig::default()
    }
}

fn ex003_fixture_base_path() -> PathBuf {
    let base_path = PathBuf::from("tests/fixtures/sammy_reference/ex003");
    assert!(
        base_path.exists(),
        "Missing ex003 fixtures at {}",
        base_path.display()
    );
    base_path
}

fn assert_chi2_close(variant: &str, computed: f64, expected: f64) {
    let (err, mode) = chi2_error(computed, expected);
    println!("{variant}:");
    println!("  Computed χ² = {computed}");
    println!("  Expected χ² = {expected}");
    println!("  Error mode = {mode}");
    println!("  Error value = {err:.6e}");

    if mode == "absolute" {
        assert!(
            err < CHI2_ABS_TOLERANCE,
            "Chi-squared mismatch ({mode} error {:.6e}): computed {}, expected {}",
            err,
            computed,
            expected
        );
    } else {
        assert!(
            err < CHI2_REL_TOLERANCE,
            "Chi-squared mismatch ({mode} error {:.6e}): computed {}, expected {}",
            err,
            computed,
            expected
        );
    }
}

/// Build R-matrix parameters from parsed resonances for ex003
fn build_ex003_params(resonances: Vec<nereids_core::nuclear::Resonance>) -> RMatrixParameters {
    // ex003 parameters (from documentation)
    let isotope = IsotopeParams {
        name: "Synthetic-10".to_string(),
        awr: 10.0,
        abundance: Parameter::fixed(1.0),
        thickness_cm: 0.1,
        number_density: 1e-3,
        spin_groups: vec![SpinGroup {
            // ex003 reference: single spin group with J=0.5 and three channels.
            j: 0.5,
            channels: vec![
                Channel {
                    l: 0, // entrance neutron channel
                    channel_spin: 0.5,
                    radius: 2.908,
                    effective_radius: 2.908,
                },
                Channel {
                    l: 0, // fission channel A
                    channel_spin: 0.5,
                    radius: 2.908,
                    effective_radius: 2.908,
                },
                Channel {
                    l: 0, // fission channel B
                    channel_spin: 0.5,
                    radius: 2.908,
                    effective_radius: 2.908,
                },
            ],
            resonances,
        }],
    };

    RMatrixParameters {
        isotopes: vec![isotope],
    }
}

#[test]
fn test_ex003a_absorption() {
    let base_path = ex003_fixture_base_path();

    // Load resonance parameters
    let par_path = base_path.join("input/ex003c.par");
    let resonances = parse_par_file(&par_path).expect("Failed to parse PAR file");
    let params = build_ex003_params(resonances);

    // Load experimental data
    let dat_path = base_path.join("input/ex003a.dat");
    let exp_data = parse_dat_file(&dat_path).expect("Failed to parse DAT file");

    // Create energy grid
    let energy_grid =
        EnergyGrid::new(exp_data.energies.clone()).expect("Failed to create energy grid");

    // Compute cross sections
    let config = ex003_config();
    let cross_sections = compute_0k_cross_sections(&energy_grid, &params, &config)
        .expect("Failed to compute cross sections");

    // Extract absorption cross section (σ_a = σ_c + σ_f)
    let sigma_absorption: Vec<f64> = cross_sections
        .iter()
        .map(|cs| cs.capture + cs.fission)
        .collect();

    // Compute chi-squared
    let chi2 = compute_chi_squared(&sigma_absorption, &exp_data.data, &exp_data.uncertainties);

    // Load expected chi-squared from LPT
    let lpt_path = base_path.join("expected/ex003a.lpt");
    let expected_stats = parse_lpt_chi_squared(&lpt_path).expect("Failed to parse LPT file");

    assert_chi2_close("ex003a (absorption)", chi2, expected_stats.chi_squared);
}

#[test]
fn test_ex003c_capture() {
    let base_path = ex003_fixture_base_path();

    // Load resonance parameters
    let par_path = base_path.join("input/ex003c.par");
    let resonances = parse_par_file(&par_path).expect("Failed to parse PAR file");
    let params = build_ex003_params(resonances);

    // Load experimental data
    let dat_path = base_path.join("input/ex003c.dat");
    let exp_data = parse_dat_file(&dat_path).expect("Failed to parse DAT file");

    // Create energy grid
    let energy_grid =
        EnergyGrid::new(exp_data.energies.clone()).expect("Failed to create energy grid");

    // Compute cross sections
    let config = ex003_config();
    let cross_sections = compute_0k_cross_sections(&energy_grid, &params, &config)
        .expect("Failed to compute cross sections");

    // Extract capture cross section
    let sigma_capture: Vec<f64> = cross_sections.iter().map(|cs| cs.capture).collect();

    // Compute chi-squared
    let chi2 = compute_chi_squared(&sigma_capture, &exp_data.data, &exp_data.uncertainties);

    // Load expected chi-squared from LPT
    let lpt_path = base_path.join("expected/ex003c.lpt");
    let expected_stats = parse_lpt_chi_squared(&lpt_path).expect("Failed to parse LPT file");

    assert_chi2_close("ex003c (capture)", chi2, expected_stats.chi_squared);
}

#[test]
fn test_ex003e_elastic() {
    let base_path = ex003_fixture_base_path();

    // Load resonance parameters
    let par_path = base_path.join("input/ex003c.par");
    let resonances = parse_par_file(&par_path).expect("Failed to parse PAR file");
    let params = build_ex003_params(resonances);

    // Load experimental data
    let dat_path = base_path.join("input/ex003e.dat");
    let exp_data = parse_dat_file(&dat_path).expect("Failed to parse DAT file");

    // Create energy grid
    let energy_grid =
        EnergyGrid::new(exp_data.energies.clone()).expect("Failed to create energy grid");

    // Compute cross sections
    let config = ex003_config();
    let cross_sections = compute_0k_cross_sections(&energy_grid, &params, &config)
        .expect("Failed to compute cross sections");

    // Extract elastic cross section
    let sigma_elastic: Vec<f64> = cross_sections.iter().map(|cs| cs.elastic).collect();

    // Compute chi-squared
    let chi2 = compute_chi_squared(&sigma_elastic, &exp_data.data, &exp_data.uncertainties);

    // Load expected chi-squared from LPT
    let lpt_path = base_path.join("expected/ex003e.lpt");
    let expected_stats = parse_lpt_chi_squared(&lpt_path).expect("Failed to parse LPT file");

    assert_chi2_close("ex003e (elastic)", chi2, expected_stats.chi_squared);
}

#[test]
fn test_ex003f_fission() {
    let base_path = ex003_fixture_base_path();

    // Load resonance parameters
    let par_path = base_path.join("input/ex003c.par");
    let resonances = parse_par_file(&par_path).expect("Failed to parse PAR file");
    let params = build_ex003_params(resonances);

    // Load experimental data
    let dat_path = base_path.join("input/ex003f.dat");
    let exp_data = parse_dat_file(&dat_path).expect("Failed to parse DAT file");

    // Create energy grid
    let energy_grid =
        EnergyGrid::new(exp_data.energies.clone()).expect("Failed to create energy grid");

    // Compute cross sections
    let config = ex003_config();
    let cross_sections = compute_0k_cross_sections(&energy_grid, &params, &config)
        .expect("Failed to compute cross sections");

    // Extract fission cross section
    let sigma_fission: Vec<f64> = cross_sections.iter().map(|cs| cs.fission).collect();

    // Compute chi-squared
    let chi2 = compute_chi_squared(&sigma_fission, &exp_data.data, &exp_data.uncertainties);

    // Load expected chi-squared from LPT
    let lpt_path = base_path.join("expected/ex003f.lpt");
    let expected_stats = parse_lpt_chi_squared(&lpt_path).expect("Failed to parse LPT file");

    assert_chi2_close("ex003f (fission)", chi2, expected_stats.chi_squared);
}

#[test]
fn test_ex003x_transmission() {
    let base_path = ex003_fixture_base_path();

    // Load resonance parameters
    let par_path = base_path.join("input/ex003c.par");
    let resonances = parse_par_file(&par_path).expect("Failed to parse PAR file");

    // Build parameters with ex003x-specific areal density
    // ex003x LPT reports Target Thickness = 0.5000E-04 (areal density n·d)
    let areal_density = 5.0e-5; // atoms/barn
    let number_density = 1.0; // atoms/(barn*cm) - arbitrary choice
    let thickness_cm = areal_density / number_density; // cm

    let mut params = build_ex003_params(resonances);
    params.isotopes[0].number_density = number_density;
    params.isotopes[0].thickness_cm = thickness_cm;

    // Load experimental data
    let dat_path = base_path.join("input/ex003x.dat");
    let exp_data = parse_dat_file(&dat_path).expect("Failed to parse DAT file");

    // Create energy grid
    let energy_grid =
        EnergyGrid::new(exp_data.energies.clone()).expect("Failed to create energy grid");

    // Compute transmission using full pipeline (Phase 1b)
    let config = ex003_config();
    let model = DefaultForwardModel { resolution: None };
    let transmission = model
        .transmission(&energy_grid, &params, &config)
        .expect("Failed to compute transmission");

    // Compute chi-squared
    let chi2 = compute_chi_squared(&transmission, &exp_data.data, &exp_data.uncertainties);

    // Load expected chi-squared from LPT
    let lpt_path = base_path.join("expected/ex003x.lpt");
    let expected_stats = parse_lpt_chi_squared(&lpt_path).expect("Failed to parse LPT file");

    assert_chi2_close("ex003x (transmission)", chi2, expected_stats.chi_squared);
}

#[test]
fn test_ex003t_total() {
    let base_path = ex003_fixture_base_path();

    // Load resonance parameters
    let par_path = base_path.join("input/ex003c.par");
    let resonances = parse_par_file(&par_path).expect("Failed to parse PAR file");
    let params = build_ex003_params(resonances);

    // Load experimental data
    let dat_path = base_path.join("input/ex003t.dat");
    let exp_data = parse_dat_file(&dat_path).expect("Failed to parse DAT file");

    // Create energy grid
    let energy_grid =
        EnergyGrid::new(exp_data.energies.clone()).expect("Failed to create energy grid");

    // Compute cross sections
    let config = ex003_config();
    let cross_sections = compute_0k_cross_sections(&energy_grid, &params, &config)
        .expect("Failed to compute cross sections");

    // Extract total cross section
    let sigma_total: Vec<f64> = cross_sections.iter().map(|cs| cs.total).collect();

    // Compute chi-squared
    let chi2 = compute_chi_squared(&sigma_total, &exp_data.data, &exp_data.uncertainties);

    // Load expected chi-squared from LPT
    let lpt_path = base_path.join("expected/ex003t.lpt");
    let expected_stats = parse_lpt_chi_squared(&lpt_path).expect("Failed to parse LPT file");

    assert_chi2_close("ex003t (total)", chi2, expected_stats.chi_squared);
}
