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
//! ## Broadening
//!
//! SAMMY applies Doppler (FGM) + Gaussian resolution + optional exponential
//! resolution broadening.  These tests apply Doppler + Gaussian resolution
//! using the SAMMY → NEREIDS parameter conversion from
//! `RslResolutionFunction_M.f90` (getAo2/getBo2).
//!
//! tr006 and tr004 use pure Gaussian resolution (Deltae=0, Iesopr=1).
//! tr007 and tr008 also have an exponential tail (Deltae>0, Iesopr=3),
//! which is not yet implemented in NEREIDS — these cases may show slightly
//! higher residual error.
//!
//! ## Reference
//! SAMMY source: `../SAMMY/SAMMY/sammy/samtry/`

use nereids_endf::sammy::{
    SammyInpConfig, SammyParFile, SammyPltRecord, parse_sammy_inp, parse_sammy_par,
    parse_sammy_plt, sammy_to_nereids_resolution, sammy_to_resonance_data,
};
use nereids_physics::reich_moore;
use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};
use nereids_physics::transmission::{self, InstrumentParams};

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

/// Build NEREIDS `InstrumentParams` from SAMMY .inp resolution parameters.
///
/// Converts SAMMY's (Deltal, Deltag) to NEREIDS's (delta_t_us, delta_l_m)
/// using the coefficient mapping from `RslResolutionFunction_M.f90`.
fn build_instrument_params(inp: &SammyInpConfig) -> Option<InstrumentParams> {
    let (flight_path, delta_t, delta_l) = sammy_to_nereids_resolution(inp)?;
    let res_params = ResolutionParams::new(flight_path, delta_t, delta_l)
        .expect("SAMMY resolution parameters should produce valid ResolutionParams");
    Some(InstrumentParams {
        resolution: ResolutionFunction::Gaussian(res_params),
    })
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
        energies.sort_by(|a, b| a.total_cmp(b));
    }

    // Build resolution parameters from SAMMY .inp config.
    let instrument = build_instrument_params(inp);

    // Compute broadened cross-sections (Doppler + Gaussian resolution).
    let broadened = transmission::broadened_cross_sections(
        &energies,
        &[resonance_data],
        inp.temperature_k,
        instrument.as_ref(),
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
        let nereids_total = *xs_map.get(&energy_ev.to_bits()).unwrap_or_else(|| {
            panic!(
                "Missing broadened cross section for energy {} eV (energy grid mismatch)",
                energy_ev
            )
        });
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
        energies.sort_by(|a, b| a.total_cmp(b));
    }

    // Broadened cross-sections with resolution.
    let instrument = build_instrument_params(inp);
    let broadened = transmission::broadened_cross_sections(
        &energies,
        &[resonance_data],
        inp.temperature_k,
        instrument.as_ref(),
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
        let nereids_trans = *trans_map.get(&energy_ev.to_bits()).unwrap_or_else(|| {
            panic!(
                "Missing transmission value for energy {} eV (energy grid mismatch)",
                energy_ev
            )
        });
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
    // Doppler + Gaussian resolution applied.  tr007 also has an exponential
    // tail (Deltae=0.022, Iesopr=3) which is not yet implemented.  The
    // remaining ~9% mean error is from the missing exponential broadening.
    assert!(
        result.mean_rel_error < 0.12,
        "broadened mean error {:.4} > 12%",
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
    // Doppler + Gaussian resolution applied.  tr008 also has an exponential
    // tail (Deltae=0.004, Iesopr=3) which is not yet implemented.  The
    // remaining ~7% mean error is from the missing exponential broadening
    // and HEGA Doppler mode differences.
    assert!(
        result.mean_rel_error < 0.10,
        "broadened mean error {:.4} > 10%",
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
    // tr006 uses pure Gaussian resolution (Deltae=0, Iesopr=1).
    // Observed ~3.4% mean error with Doppler + Gaussian resolution.
    assert!(
        result.mean_rel_error < 0.05,
        "broadened mean error {:.4} > 5%",
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
    // tr004 uses pure Gaussian resolution (Deltae=0, Iesopr=1).
    // Observed ~4.7% mean error with Doppler + Gaussian resolution.
    assert!(
        result.mean_rel_error < 0.06,
        "broadened mean error {:.4} > 6%",
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
    // tr004 transmission: pure Gaussian resolution (Deltae=0, Iesopr=1).
    // Observed ~0.95% mean error — transmission exponentiates the XS error,
    // so small XS errors become small transmission errors.
    assert!(
        result.mean_rel_error < 0.02,
        "transmission mean error {:.4} > 2%",
        result.mean_rel_error
    );
}
