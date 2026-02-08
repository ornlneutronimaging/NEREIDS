//! Integration tests for SAMMY ex005 validation.
//!
//! Tests Doppler broadening against SAMMY reference outputs.
//! - ex005a: 0K capture cross sections (no broadening)
//! - ex005b: 50K capture cross sections (with Doppler broadening)
//!
//! Key physics: fictional element (AWR=10), single resonance at 10 eV,
//! no fission channels, capture data type.
//!
//! SAMMY PAR file widths are in meV; our parser converts to eV.

mod parsers;

use nereids_core::energy::EnergyGrid;
use nereids_core::forward_model::ForwardModelConfig;
use nereids_core::nuclear::{Channel, IsotopeParams, Parameter, RMatrixParameters, SpinGroup};
use nereids_physics::broadening::doppler::{broaden_cross_sections_to_grid, create_auxiliary_grid};
use nereids_physics::rmatrix::compute_0k_cross_sections;
use parsers::{parse_dat_file, parse_lpt_chi_squared, parse_par_file};
use std::path::PathBuf;

fn debug_log_enabled() -> bool {
    std::env::var_os("NEREIDS_TEST_DEBUG").is_some()
}

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

const CHI2_ABS_TOLERANCE: f64 = 1e-4;
const CHI2_REL_TOLERANCE: f64 = 2e-3;

fn ex005_fixture_base_path() -> PathBuf {
    let base = PathBuf::from("tests/fixtures/sammy_reference/ex005");
    assert!(
        base.exists(),
        "Missing ex005 fixtures at {}",
        base.display()
    );
    base
}

fn ex005_config() -> ForwardModelConfig {
    ForwardModelConfig {
        include_potential_scattering: true,
        ..ForwardModelConfig::default()
    }
}

/// Build R-matrix parameters from parsed resonances for ex005.
///
/// Key difference from ex003: ex005 has NO fission channels (per inp file),
/// so fission widths from the par file are stripped.
fn build_ex005_params(mut resonances: Vec<nereids_core::nuclear::Resonance>) -> RMatrixParameters {
    // Strip fission channels: ex005 inp defines only neutron+gamma
    for res in &mut resonances {
        res.fission = None;
    }

    let isotope = IsotopeParams {
        name: "element".to_string(),
        awr: 10.0,
        abundance: Parameter::fixed(1.0),
        // No Beer-Lambert for capture data type
        thickness_cm: 0.0,
        number_density: 0.0,
        spin_groups: vec![SpinGroup {
            j: 0.5,
            channels: vec![Channel {
                l: 0,
                channel_spin: 0.5,
                radius: 2.908,
                effective_radius: 2.908,
            }],
            resonances,
        }],
    };

    RMatrixParameters {
        isotopes: vec![isotope],
    }
}

fn assert_chi2_close(label: &str, computed: f64, expected: f64) {
    let abs_err = (computed - expected).abs();
    let rel_err = abs_err / expected.abs().max(1e-30);
    if debug_log_enabled() {
        println!("{label}:");
        println!("  Computed χ² = {computed:.1}");
        println!("  Expected χ² = {expected:.1}");
        println!("  Absolute error = {abs_err:.4e}");
        println!("  Relative error = {rel_err:.4e}");
    }
    assert!(
        abs_err < CHI2_ABS_TOLERANCE || rel_err < CHI2_REL_TOLERANCE,
        "{label}: chi-squared mismatch (abs_err={abs_err:.4e}, rel_err={rel_err:.4e}): computed={computed:.1}, expected={expected:.1}"
    );
}

/// ex005a: 0K capture cross sections, no Doppler broadening.
///
/// Expected: χ² ≈ 53241.1 (SAMMY ex005aa.lpt)
#[test]
fn test_ex005a_no_broadening() {
    let base = ex005_fixture_base_path();

    // Parse resonance parameters (meV → eV conversion in parser)
    let resonances =
        parse_par_file(&base.join("input/ex005a.par")).expect("Failed to parse ex005a.par");
    let params = build_ex005_params(resonances);

    // Parse experimental data (CSISRS format, absolute uncertainties)
    let exp_data =
        parse_dat_file(&base.join("input/ex005a.dat")).expect("Failed to parse ex005a.dat");
    assert_eq!(
        exp_data.energies.len(),
        315,
        "ex005a should have 315 data points"
    );

    // Compute 0K capture cross sections at data energies
    let grid = EnergyGrid::new(exp_data.energies.clone()).unwrap();
    let config = ex005_config();
    let xs = compute_0k_cross_sections(&grid, &params, &config)
        .expect("Failed to compute 0K cross sections");

    let sigma_capture: Vec<f64> = xs.iter().map(|cs| cs.capture).collect();

    // Print diagnostic values near the resonance
    if debug_log_enabled() {
        println!("\n=== ex005a diagnostics (0K, no broadening) ===");
        for (i, e) in exp_data.energies.iter().enumerate() {
            if (*e - 10.0).abs() < 0.05 {
                println!(
                    "  E={:.4}  σ_cap={:.4}  data={:.4}  unc={:.4}",
                    e, sigma_capture[i], exp_data.data[i], exp_data.uncertainties[i]
                );
            }
        }
    }

    // Compute chi-squared
    let chi2 = compute_chi_squared(&sigma_capture, &exp_data.data, &exp_data.uncertainties);

    // Compare against SAMMY reference
    let expected =
        parse_lpt_chi_squared(&base.join("expected/ex005aa.lpt")).expect("Failed to parse LPT");

    assert_chi2_close("ex005a (0K capture)", chi2, expected.chi_squared);
}

/// ex005b: 50K capture cross sections with Doppler broadening.
///
/// Expected: χ² ≈ 52514.8 (SAMMY ex005bb.lpt)
///
/// This test validates the full Doppler broadening pipeline:
/// 1. Create auxiliary fine grid that resolves the narrow resonance
/// 2. Compute 0K capture XS on fine grid
/// 3. Broaden at 50K using Gaussian convolution
/// 4. Interpolate broadened result to data energies
/// 5. Compare chi-squared against SAMMY reference
#[test]
fn test_ex005b_50k() {
    let base = ex005_fixture_base_path();

    // Parse resonance parameters
    let resonances =
        parse_par_file(&base.join("input/ex005a.par")).expect("Failed to parse ex005a.par");

    // Extract resonance info for auxiliary grid creation (widths already in eV)
    let res_energies: Vec<f64> = resonances.iter().map(|r| r.energy.value).collect();
    let res_widths: Vec<f64> = resonances
        .iter()
        .map(|r| r.gamma_g.value + r.gamma_n.value) // Total width (no fission)
        .collect();

    let params = build_ex005_params(resonances);

    // Parse experimental data
    let exp_data =
        parse_dat_file(&base.join("input/ex005a.dat")).expect("Failed to parse ex005a.dat");

    // Create auxiliary fine grid that resolves the narrow resonance (meV-scale)
    let aux_grid =
        create_auxiliary_grid(&exp_data.energies, &res_energies, &res_widths, 50.0, 10.0)
            .expect("Failed to create auxiliary grid");
    // SAMMY ex005bb reports 370 auxiliary points for this case.
    // Keep this tightly bounded near SAMMY's 370-point reference.
    assert!(
        (355..=375).contains(&aux_grid.len()),
        "unexpected auxiliary-grid size for ex005b: got {}, expected near SAMMY's 370",
        aux_grid.len()
    );
    if debug_log_enabled() {
        println!(
            "\n=== ex005b: auxiliary grid has {} points (data grid has {}) ===",
            aux_grid.len(),
            exp_data.energies.len()
        );
    }

    // Compute 0K capture cross sections on fine grid
    let fine_grid = EnergyGrid::new(aux_grid.clone()).unwrap();
    let config = ex005_config();
    let xs = compute_0k_cross_sections(&fine_grid, &params, &config)
        .expect("Failed to compute 0K cross sections on fine grid");

    let capture_0k: Vec<f64> = xs.iter().map(|cs| cs.capture).collect();

    // Print 0K peak value for diagnostics
    let max_0k = capture_0k.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_idx = capture_0k.iter().position(|&v| v == max_0k).unwrap_or(0);
    if debug_log_enabled() {
        println!(
            "  0K peak: σ_cap={:.1} barns at E={:.6} eV",
            max_0k, aux_grid[max_idx]
        );
    }

    // Apply Doppler broadening at 50K directly from auxiliary 0K grid
    // onto data energies (SAMMY source-grid -> data-grid workflow).
    let theory =
        broaden_cross_sections_to_grid(&capture_0k, &aux_grid, &exp_data.energies, 10.0, 50.0)
            .expect("Broadening failed");

    // Print diagnostics near the resonance
    if debug_log_enabled() {
        println!("=== ex005b diagnostics (50K broadened) ===");
        for (i, e) in exp_data.energies.iter().enumerate() {
            if (*e - 10.0).abs() < 0.05 {
                println!(
                    "  E={:.4}  σ_cap(50K)={:.4}  data={:.4}  unc={:.4}",
                    e, theory[i], exp_data.data[i], exp_data.uncertainties[i]
                );
            }
        }
    }

    // Compute chi-squared
    let chi2 = compute_chi_squared(&theory, &exp_data.data, &exp_data.uncertainties);

    // Compare against SAMMY reference
    let expected =
        parse_lpt_chi_squared(&base.join("expected/ex005bb.lpt")).expect("Failed to parse LPT");

    assert_chi2_close("ex005b (50K capture)", chi2, expected.chi_squared);
}
