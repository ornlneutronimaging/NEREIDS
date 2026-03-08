//! SAMMY `samtry` test suite validation.
//!
//! Validates NEREIDS physics against SAMMY's canonical reference output from
//! the `samtry/` test suite.  Each test case loads a SAMMY `.par` + `.inp` +
//! `.plt` file set, computes cross-sections (and optionally transmission) using
//! NEREIDS, and compares against SAMMY's `Th_initial` reference values.
//!
//! ## Phase 1: Transmission cases (issue #292)
//! - tr007: Fe-56, 3 resonances, 2 spin groups
//! - tr008: Ni-58, ~100 resonances, 5 spin groups
//! - tr006: Ni-60, ~270 resonances, pure forward model
//! - tr004: Ni-60, ~270 resonances, with transmission reference
//!
//! ## Phase 1 Tolerances
//!
//! SAMMY applies Doppler (FGM) + Gaussian resolution + exponential resolution
//! broadening.  NEREIDS currently applies only Doppler broadening in these
//! tests (no resolution parameters passed).  The remaining ~10-15% mean error
//! at resonance peaks is dominated by the missing resolution broadening:
//!
//! | Test  | Energy    | Doppler FWHM | Resolution FWHM | Mean error |
//! |-------|-----------|-------------|-----------------|------------|
//! | tr007 | 1.1 keV   | 2.5 eV      | 0.6 eV          | ~9%        |
//! | tr008 | 300 keV   | 39 eV       | 296 eV          | ~14%       |
//! | tr006 | 135 keV   | 26 eV       | 183 eV          | ~12%       |
//! | tr004 | 500 keV   | 50 eV       | 500+ eV         | ~15%       |
//!
//! Phase 2 will add resolution broadening and tighten tolerances to <1%.
//!
//! ## Reference
//! SAMMY source: `../SAMMY/SAMMY/sammy/samtry/`

use nereids_endf::sammy::{
    SammyInpConfig, SammyParFile, SammyPltRecord, parse_sammy_inp, parse_sammy_par,
    parse_sammy_plt, sammy_to_resonance_data,
};
use nereids_physics::reich_moore;
use nereids_physics::transmission;

use std::path::PathBuf;

// ─── Test infrastructure ───────────────────────────────────────────────────────

/// Root directory for samtry test data.
fn samtry_data_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop(); // crates/
    dir.pop(); // repo root
    dir.push("tests");
    dir.push("data");
    dir.push("samtry");
    dir
}

/// Load all files for a samtry test case.
fn load_samtry_case(
    test_id: &str,
    inp_name: &str,
    par_name: &str,
    plt_name: &str,
) -> (SammyInpConfig, SammyParFile, Vec<SammyPltRecord>) {
    let dir = samtry_data_dir().join(test_id);

    let inp_content = std::fs::read_to_string(dir.join(inp_name))
        .unwrap_or_else(|e| panic!("failed to read {inp_name}: {e}"));
    let par_content = std::fs::read_to_string(dir.join(par_name))
        .unwrap_or_else(|e| panic!("failed to read {par_name}: {e}"));
    let plt_content = std::fs::read_to_string(dir.join("answers").join(plt_name))
        .unwrap_or_else(|e| panic!("failed to read answers/{plt_name}: {e}"));

    let inp = parse_sammy_inp(&inp_content).unwrap();
    let par = parse_sammy_par(&par_content).unwrap();
    let plt = parse_sammy_plt(&plt_content).unwrap();

    (inp, par, plt)
}

/// Validation result for a single test case.
#[derive(Debug)]
struct ValidationResult {
    /// Maximum relative error across all reference points.
    max_rel_error: f64,
    /// Mean relative error.
    mean_rel_error: f64,
    /// Total number of reference points compared.
    n_points: usize,
    /// Number of points exceeding the tolerance.
    n_above_threshold: usize,
    /// Energy (keV) of the worst-case point.
    worst_energy_kev: f64,
}

/// Compare NEREIDS cross-sections against SAMMY Th_initial reference.
///
/// Uses unbroadened cross-sections first (no Doppler, no resolution).
fn validate_unbroadened_cross_sections(
    inp: &SammyInpConfig,
    par: &SammyParFile,
    reference: &[SammyPltRecord],
    tolerance_rel: f64,
) -> ValidationResult {
    let resonance_data = sammy_to_resonance_data(inp, par).unwrap();

    let mut max_rel_error = 0.0_f64;
    let mut sum_rel_error = 0.0;
    let mut n_above = 0;
    let mut worst_energy_kev = 0.0;

    for rec in reference {
        let energy_ev = rec.energy_kev * 1000.0; // .plt uses keV
        let xs = reich_moore::cross_sections_at_energy(&resonance_data, energy_ev);
        let nereids_total = xs.total;
        let sammy_total = rec.theory_initial;

        let rel_error = if sammy_total.abs() > 1e-6 {
            (nereids_total - sammy_total).abs() / sammy_total.abs()
        } else {
            // Near-zero: use absolute tolerance.
            (nereids_total - sammy_total).abs()
        };

        sum_rel_error += rel_error;
        if rel_error > max_rel_error {
            max_rel_error = rel_error;
            worst_energy_kev = rec.energy_kev;
        }
        if rel_error > tolerance_rel {
            n_above += 1;
        }
    }

    ValidationResult {
        max_rel_error,
        mean_rel_error: sum_rel_error / reference.len() as f64,
        n_points: reference.len(),
        n_above_threshold: n_above,
        worst_energy_kev,
    }
}

/// Compare NEREIDS broadened cross-sections against SAMMY Th_initial reference.
fn validate_broadened_cross_sections(
    inp: &SammyInpConfig,
    par: &SammyParFile,
    reference: &[SammyPltRecord],
    tolerance_rel: f64,
) -> ValidationResult {
    let resonance_data = sammy_to_resonance_data(inp, par).unwrap();

    // Build sorted energy grid from reference points (ascending).
    let mut energies: Vec<f64> = reference.iter().map(|r| r.energy_kev * 1000.0).collect();
    // SAMMY .plt may not be sorted; ensure ascending for broadening.
    let is_ascending = energies.windows(2).all(|w| w[0] <= w[1]);
    if !is_ascending {
        // Sort and track original order.
        let mut indexed: Vec<(usize, f64)> = energies.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        energies = indexed.iter().map(|(_, e)| *e).collect();
    }

    // Compute broadened cross-sections (Doppler + no resolution for now).
    let broadened = transmission::broadened_cross_sections(
        &energies,
        &[resonance_data],
        inp.temperature_k,
        None, // Skip resolution broadening for now.
        None, // No cancellation.
    )
    .unwrap();

    let xs_total = &broadened[0]; // Single isotope.

    // Build energy→xs map for lookup.
    let xs_map: std::collections::HashMap<u64, f64> = energies
        .iter()
        .zip(xs_total.iter())
        .map(|(&e, &xs)| (e.to_bits(), xs))
        .collect();

    let mut max_rel_error = 0.0_f64;
    let mut sum_rel_error = 0.0;
    let mut n_above = 0;
    let mut worst_energy_kev = 0.0;

    for rec in reference {
        let energy_ev = rec.energy_kev * 1000.0;
        let nereids_total = *xs_map.get(&energy_ev.to_bits()).unwrap_or(&0.0);
        let sammy_total = rec.theory_initial;

        let rel_error = if sammy_total.abs() > 1e-6 {
            (nereids_total - sammy_total).abs() / sammy_total.abs()
        } else {
            (nereids_total - sammy_total).abs()
        };

        sum_rel_error += rel_error;
        if rel_error > max_rel_error {
            max_rel_error = rel_error;
            worst_energy_kev = rec.energy_kev;
        }
        if rel_error > tolerance_rel {
            n_above += 1;
        }
    }

    ValidationResult {
        max_rel_error,
        mean_rel_error: sum_rel_error / reference.len() as f64,
        n_points: reference.len(),
        n_above_threshold: n_above,
        worst_energy_kev,
    }
}

/// Compare NEREIDS transmission against SAMMY Tr_initial reference (.plt2 file).
fn validate_transmission(
    inp: &SammyInpConfig,
    par: &SammyParFile,
    reference: &[SammyPltRecord],
    tolerance_rel: f64,
) -> ValidationResult {
    let resonance_data = sammy_to_resonance_data(inp, par).unwrap();

    // Build sorted energy grid.
    let mut energies: Vec<f64> = reference.iter().map(|r| r.energy_kev * 1000.0).collect();
    let is_ascending = energies.windows(2).all(|w| w[0] <= w[1]);
    if !is_ascending {
        energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    // Broadened cross-sections.
    let broadened = transmission::broadened_cross_sections(
        &energies,
        &[resonance_data],
        inp.temperature_k,
        None,
        None,
    )
    .unwrap();

    // Beer-Lambert transmission: T = exp(-n * σ)
    let trans = transmission::beer_lambert(&broadened[0], inp.thickness_atoms_barn);

    let trans_map: std::collections::HashMap<u64, f64> = energies
        .iter()
        .zip(trans.iter())
        .map(|(&e, &t)| (e.to_bits(), t))
        .collect();

    let mut max_rel_error = 0.0_f64;
    let mut sum_rel_error = 0.0;
    let mut n_above = 0;
    let mut worst_energy_kev = 0.0;

    for rec in reference {
        let energy_ev = rec.energy_kev * 1000.0;
        let nereids_trans = *trans_map.get(&energy_ev.to_bits()).unwrap_or(&0.0);
        let sammy_trans = rec.theory_initial;

        let rel_error = if sammy_trans.abs() > 1e-6 {
            (nereids_trans - sammy_trans).abs() / sammy_trans.abs()
        } else {
            (nereids_trans - sammy_trans).abs()
        };

        sum_rel_error += rel_error;
        if rel_error > max_rel_error {
            max_rel_error = rel_error;
            worst_energy_kev = rec.energy_kev;
        }
        if rel_error > tolerance_rel {
            n_above += 1;
        }
    }

    ValidationResult {
        max_rel_error,
        mean_rel_error: sum_rel_error / reference.len() as f64,
        n_points: reference.len(),
        n_above_threshold: n_above,
        worst_energy_kev,
    }
}

// ─── Test cases ────────────────────────────────────────────────────────────────

/// tr007: Fe-56 transmission, 1.13-1.17 keV, 3 resonances, 2 spin groups.
///
/// This is the simplest samtry test case. The .plt file contains broadened
/// cross-section reference values (Th_initial).
#[test]
fn test_tr007_fe56_parse() {
    let (inp, par, plt) = load_samtry_case("tr007", "t007a.inp", "t007a.par", "raa.plt");

    // Verify parsing.
    assert_eq!(par.resonances.len(), 3);
    assert_eq!(inp.spin_groups.len(), 2);
    assert!(!plt.is_empty(), "reference data should not be empty");

    // Verify isotope info.
    assert_eq!(inp.isotope_symbol, "FE56");
    assert!((inp.awr - 55.9).abs() < 0.1);
    assert!((inp.temperature_k - 329.0).abs() < 1.0);
}

#[test]
fn test_tr007_fe56_unbroadened() {
    let (inp, par, plt) = load_samtry_case("tr007", "t007a.inp", "t007a.par", "raa.plt");

    // Unbroadened cross-sections won't match the broadened reference exactly,
    // but they should be in the same ballpark — validate the physics is sane.
    let result = validate_unbroadened_cross_sections(&inp, &par, &plt, 1.0);
    eprintln!(
        "tr007 unbroadened: max_rel={:.4}, mean_rel={:.4}, n={}, worst@{:.4} keV",
        result.max_rel_error, result.mean_rel_error, result.n_points, result.worst_energy_kev
    );
    // Unbroadened should be within ~50% of broadened (resonance peaks differ).
    // This is a sanity check, not a precision test.
    assert!(
        result.mean_rel_error < 0.5,
        "unbroadened mean error {:.4} > 50%",
        result.mean_rel_error
    );
}

#[test]
fn test_tr007_fe56_broadened() {
    let (inp, par, plt) = load_samtry_case("tr007", "t007a.inp", "t007a.par", "raa.plt");

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr007 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // Phase 1: Doppler-only broadening.  SAMMY also applies Gaussian +
    // exponential resolution broadening (combined FWHM ~0.6 eV at 1.1 keV),
    // which we omit here.  The remaining ~9% mean error is concentrated at
    // the resonance peak where resolution broadening reduces the peak height.
    // Phase 2 will add resolution broadening and tighten to <1%.
    assert!(
        result.mean_rel_error < 0.15,
        "broadened mean error {:.4} > 15%",
        result.mean_rel_error
    );
}

/// tr008: Ni-58 transmission, 293-308 keV, ~100 resonances, 5 spin groups.
#[test]
fn test_tr008_ni58_parse() {
    let (inp, par, plt) = load_samtry_case("tr008", "t008a.inp", "t008a.par", "raa.plt");

    assert!(par.resonances.len() > 50, "expected many resonances");
    assert!(inp.spin_groups.len() >= 4, "expected >= 4 spin groups");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "58NI");
}

#[test]
fn test_tr008_ni58_broadened() {
    let (inp, par, plt) = load_samtry_case("tr008", "t008a.inp", "t008a.par", "raa.plt");

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr008 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // Phase 1: Resolution broadening dominates at 300 keV (FWHM ~296 eV
    // vs Doppler ~39 eV).  Without resolution, ~14% mean error expected.
    assert!(
        result.mean_rel_error < 0.20,
        "broadened mean error {:.4} > 20%",
        result.mean_rel_error
    );
}

/// tr006: Ni-60, 134-137 keV, ~270 resonances, DO NOT SOLVE BAYES (pure forward).
#[test]
fn test_tr006_ni60_parse() {
    let (inp, par, plt) = load_samtry_case("tr006", "t006a.inp", "t006a.par", "raa.plt");

    assert!(par.resonances.len() > 100, "expected many resonances");
    assert_eq!(inp.spin_groups.len(), 5);
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "60NI");
}

#[test]
fn test_tr006_ni60_broadened() {
    let (inp, par, plt) = load_samtry_case("tr006", "t006a.inp", "t006a.par", "raa.plt");

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr006 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // Phase 1: Resolution dominates at 135 keV (FWHM ~183 eV vs Doppler ~26 eV).
    assert!(
        result.mean_rel_error < 0.20,
        "broadened mean error {:.4} > 20%",
        result.mean_rel_error
    );
}

/// tr004: Ni-60, 505-508 keV, ~270 resonances, with transmission reference.
#[test]
fn test_tr004_ni60_parse() {
    let (inp, par, plt) = load_samtry_case("tr004", "t004a.inp", "t004a.par", "raa.plt");

    assert!(par.resonances.len() > 100, "expected many resonances");
    assert_eq!(inp.spin_groups.len(), 5);
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "60NI");
}

#[test]
fn test_tr004_ni60_broadened() {
    let (inp, par, plt) = load_samtry_case("tr004", "t004a.inp", "t004a.par", "raa.plt");

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr004 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // Phase 1: Resolution dominates at 500 keV (FWHM ~500+ eV vs Doppler ~50 eV).
    assert!(
        result.mean_rel_error < 0.20,
        "broadened mean error {:.4} > 20%",
        result.mean_rel_error
    );
}

#[test]
fn test_tr004_ni60_transmission() {
    let (inp, par, _) = load_samtry_case("tr004", "t004a.inp", "t004a.par", "raa.plt");

    // Load transmission reference (.plt2 file).
    let plt2_path = samtry_data_dir()
        .join("tr004")
        .join("answers")
        .join("raa2.plt");
    let plt2_content = std::fs::read_to_string(&plt2_path).unwrap();
    let plt2 = parse_sammy_plt(&plt2_content).unwrap();

    let result = validate_transmission(&inp, &par, &plt2, 0.01);
    eprintln!(
        "tr004 transmission: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    assert!(
        result.mean_rel_error < 0.05,
        "transmission mean error {:.4} > 5%",
        result.mean_rel_error
    );
}
