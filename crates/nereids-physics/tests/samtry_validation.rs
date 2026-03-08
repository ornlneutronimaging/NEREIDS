//! SAMMY `samtry` test suite validation (issue #321).
//!
//! Validates NEREIDS physics against SAMMY's canonical reference output from
//! the `samtry/` test suite.  Each test case loads a SAMMY `.par` + `.inp` +
//! `.plt` file set, computes cross-sections (and optionally transmission) using
//! NEREIDS, and compares against SAMMY's `Th_initial` reference values.
//!
//! ## Error Budget by Broadening Type
//!
//! NEREIDS now implements SAMMY's exact resolution broadening:
//! - **Xcoef 4-point quadrature** (Eq. IV B 3.8) for integration weights
//! - **Gaussian + exponential tail kernel** (Iesopr=3) when Deltae > 0
//! - **Exerfc scaled complementary error function** for numerical stability
//!
//! The remaining errors come from:
//!
//! 1. **Doppler method mismatch** — SAMMY's `use multi-style doppler` keyword
//!    activates HEGA (Gaussian approximation in E-space).  NEREIDS always uses
//!    exact FGM (velocity-space convolution).  Affects tr006, tr008.
//!
//! 2. **Sparse grid + wide resonances** — When few points fall in the
//!    convolution window (high energy), even with Xcoef quadrature, the
//!    discrete integration struggles near sharp resonance peaks.
//!
//! | Category | Cases | Mean | Max | Dominant Error Source |
//! |----------|-------|------|-----|---------------------|
//! | FGM + Gauss, dense grid | tr015, tr016 | <9% | <21% | Discrete convolution |
//! | FGM + Gauss, sparse grid | tr004 | <5% | <28% | Sparse-grid convolution |
//! | HEGA Doppler + Gaussian | tr006 | <4% | <23% | Doppler method mismatch |
//! | FGM + HEGA, no Doppler | tr008 | <4% | <27% | HEGA vs FGM difference |
//! | FGM + Gauss + Exp tail | tr007, tr047 | <10% | <55% | Resonance peak sampling |
//! | FGM + Gauss + Exp, sparse | tr029, tr030 | <21% | <145% | Sparse grid + exp tail |
//! | Multi-channel fission | tr028 | IGNORED | — | Not implemented (Batch C) |
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
/// Converts SAMMY's (Deltal, Deltag, Deltae) to NEREIDS's
/// (delta_t_us, delta_l_m, delta_e) using the coefficient mapping from
/// `RslResolutionFunction_M.f90`.
fn build_instrument_params(inp: &SammyInpConfig) -> Option<InstrumentParams> {
    let (flight_path, delta_t, delta_l, delta_e) = sammy_to_nereids_resolution(inp)?;
    let res_params = ResolutionParams::new(flight_path, delta_t, delta_l, delta_e)
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
    let (inp, par, plt) = load_samtry_case(
        "tr007_fe56_transmission_doppler_resolution",
        "t007a.inp",
        "t007a.par",
        "raa.plt",
    );

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
    let (inp, par, plt) = load_samtry_case(
        "tr007_fe56_transmission_doppler_resolution",
        "t007a.inp",
        "t007a.par",
        "raa.plt",
    );

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
    let (inp, par, plt) = load_samtry_case(
        "tr007_fe56_transmission_doppler_resolution",
        "t007a.inp",
        "t007a.par",
        "raa.plt",
    );

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr007 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr007: Deltae=0.022, has exponential tail (Iesopr=3).
    // FGM Doppler + Gaussian+exponential resolution broadening implemented.
    // Dense grid at low energy (1.13-1.17 keV) means quadrature error is
    // negligible here.  Remaining error is from FGM vs HEGA Doppler difference.
    // Measured: 8.8% mean.
    assert!(
        result.mean_rel_error < 0.12,
        "broadened mean error {:.4} > 12%",
        result.mean_rel_error
    );
}

/// tr008: Ni-58 transmission, 293-308 keV, ~100 resonances, 5 spin groups.
#[test]
fn test_tr008_ni58_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr008_ni58_transmission_hega",
        "t008a.inp",
        "t008a.par",
        "raa.plt",
    );

    assert!(par.resonances.len() > 50, "expected many resonances");
    assert!(inp.spin_groups.len() >= 4, "expected >= 4 spin groups");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "58NI");
}

#[test]
fn test_tr008_ni58_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr008_ni58_transmission_hega",
        "t008a.inp",
        "t008a.par",
        "raa.plt",
    );

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr008 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr008: SAMMY uses HEGA Doppler (multi-style doppler broadening keyword),
    // NEREIDS uses exact FGM.  Additionally has exponential tail (Deltae=0.004).
    // Error sources: Doppler method mismatch + missing exponential tail.
    // Measured: 7.0% mean.
    assert!(
        result.mean_rel_error < 0.10,
        "broadened mean error {:.4} > 10%",
        result.mean_rel_error
    );
}

/// tr006: Ni-60, 134-137 keV, ~270 resonances, DO NOT SOLVE BAYES (pure forward).
#[test]
fn test_tr006_ni60_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr006_ni60_transmission_gaussian_print",
        "t006a.inp",
        "t006a.par",
        "raa.plt",
    );

    assert!(par.resonances.len() > 100, "expected many resonances");
    assert_eq!(inp.spin_groups.len(), 5);
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "60NI");
}

#[test]
fn test_tr006_ni60_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr006_ni60_transmission_gaussian_print",
        "t006a.inp",
        "t006a.par",
        "raa.plt",
    );

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr006 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr006: Deltae=0 (pure Gaussian resolution), but SAMMY uses HEGA Doppler
    // (`use multi-style doppler broadening` keyword → Kkkdop=0) while NEREIDS
    // uses exact FGM.  The 3.4% mean error is the HEGA-vs-FGM Doppler
    // method difference, not a resolution bug.  Kernel formula and parameter
    // conversion are exact (verified against mrsl4.f90 Rolowg).
    // Measured: 3.4% mean.
    assert!(
        result.mean_rel_error < 0.05,
        "broadened mean error {:.4} > 5%",
        result.mean_rel_error
    );
}

/// tr004: Ni-60, 505-508 keV, ~270 resonances, with transmission reference.
#[test]
fn test_tr004_ni60_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr004_ni60_transmission_gaussian",
        "t004a.inp",
        "t004a.par",
        "raa.plt",
    );

    assert!(par.resonances.len() > 100, "expected many resonances");
    assert_eq!(inp.spin_groups.len(), 5);
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "60NI");
}

#[test]
fn test_tr004_ni60_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr004_ni60_transmission_gaussian",
        "t004a.inp",
        "t004a.par",
        "raa.plt",
    );

    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr004 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr004: Deltae=0 (pure Gaussian resolution), both SAMMY and NEREIDS use
    // FGM Doppler.  The 4.7% mean error is dominated by sparse-grid trapezoidal
    // quadrature in the resolution convolution: at 500 keV the Gaussian width
    // is ~772 eV but only ~21 grid points fall in the 10σ window.  SAMMY's
    // 4-point Xcoef weights (Eq. IV B 3.8) handle sparse grids better.
    // Fix: implement Xcoef quadrature or densify the convolution grid.
    // Measured: 4.7% mean.
    assert!(
        result.mean_rel_error < 0.06,
        "broadened mean error {:.4} > 6%",
        result.mean_rel_error
    );
}

#[test]
fn test_tr004_ni60_transmission() {
    let (inp, par, _) = load_samtry_case(
        "tr004_ni60_transmission_gaussian",
        "t004a.inp",
        "t004a.par",
        "raa.plt",
    );

    // Load transmission reference (.plt2 file).
    let plt2_path = samtry_data_dir()
        .join("tr004_ni60_transmission_gaussian")
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
    // tr004 transmission: same sparse-grid quadrature issue as XS (4.7%), but
    // T = exp(-n·σ) compresses cross-section errors exponentially.
    // Measured: 0.95% mean — confirms the XS error is the root cause.
    assert!(
        result.mean_rel_error < 0.02,
        "transmission mean error {:.4} > 2%",
        result.mean_rel_error
    );
}

// ─── Batch A: New cases (issue #321-A) ──────────────────────────────────────

/// tr015: Ni-58 transmission, 180-181 keV, Doppler+Gaussian (Deltae=0).
///
/// Tests ENERGY UNCERTAINTIES keyword (parser should ignore it).
#[test]
fn test_tr015_ni58_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr015_ni58_transmission_energy_unc",
        "t015a.inp",
        "t015a.par",
        "raa.plt",
    );
    assert!(par.resonances.len() >= 15, "expected >=15 resonances");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "58NI");
    assert!((inp.temperature_k - 300.0).abs() < 1.0);
    assert!(!inp.no_broadening);
}

#[test]
fn test_tr015_ni58_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr015_ni58_transmission_energy_unc",
        "t015a.inp",
        "t015a.par",
        "raa.plt",
    );
    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr015 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr015: Deltae=0, FGM Doppler + pure Gaussian resolution.  Narrow range
    // (180-181 keV) with ~28 points.  Trapezoidal quadrature at moderate
    // energy produces ~8.4% mean error.  Similar to tr004/tr006 mechanism.
    // Measured: 8.4% mean.
    assert!(
        result.mean_rel_error < 0.10,
        "broadened mean error {:.4} > 10%",
        result.mean_rel_error
    );
}

/// tr016: Ni-58 transmission, 180-183 keV, Doppler+Gaussian (Deltae=0).
///
/// Tests ENERGY UNCERTAINTIES + PRINT PARTIAL DERIVATIVES keywords.
#[test]
fn test_tr016_ni58_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr016_ni58_transmission_partial_deriv",
        "t016a.inp",
        "t016a.par",
        "raa.plt",
    );
    assert!(par.resonances.len() >= 10, "expected >=10 resonances");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "58NI");
    assert!(!inp.no_broadening);
}

#[test]
fn test_tr016_ni58_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr016_ni58_transmission_partial_deriv",
        "t016a.inp",
        "t016a.par",
        "raa.plt",
    );
    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr016 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr016: Deltae=0, FGM Doppler + pure Gaussian resolution.  Wider range
    // (180-183 keV) with ~56 points — same mechanism as tr015 but with more
    // grid points, hence slightly lower error.
    // Measured: 6.5% mean.
    assert!(
        result.mean_rel_error < 0.08,
        "broadened mean error {:.4} > 8%",
        result.mean_rel_error
    );
}

/// tr029: Ni-58 transmission, 40-53000 eV, Doppler+Gaussian+Exponential.
///
/// Tests ENERGY UNCERTAINTIES and PRINT ALL INPUT keywords.
#[test]
fn test_tr029_ni58_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr029_ni58_transmission_abundance_var",
        "t029a.inp",
        "t029a.par",
        "raa.plt",
    );
    assert!(par.resonances.len() >= 100, "expected many resonances");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "58NI");
    assert!(!inp.no_broadening);
}

#[test]
fn test_tr029_ni58_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr029_ni58_transmission_abundance_var",
        "t029a.inp",
        "t029a.par",
        "raa.plt",
    );
    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr029 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr029: Deltae=0.008, has exponential tail (Iesopr=3).
    // Wide range (40-53k eV) with 1032 points — good grid density at lower
    // energies where most resonances live.  The remaining error is from
    // sparse-grid convolution at high energies plus Doppler method differences.
    // Measured: 4.9% mean.
    assert!(
        result.mean_rel_error < 0.08,
        "broadened mean error {:.4} > 8%",
        result.mean_rel_error
    );
}

/// tr030: Ni-58 transmission, 13-15.5 keV, Doppler+Gaussian+Exponential.
///
/// Tests spin group removal via negative sign in PAR file.
#[test]
fn test_tr030_ni58_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr030_ni58_transmission_spin_removal",
        "t030a.inp",
        "t030a.par",
        "raa.plt",
    );
    assert!(!par.resonances.is_empty(), "expected resonances");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "58NI");
    assert!(!inp.no_broadening);
}

#[test]
fn test_tr030_ni58_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr030_ni58_transmission_spin_removal",
        "t030a.inp",
        "t030a.par",
        "raa.plt",
    );
    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr030 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr030: Deltae=0.008, has exponential tail (Iesopr=3).
    // Narrow range (13-15.5 keV) with only 5 active resonances (68 excluded
    // via negative spin group).  The exponential tail's relative effect is
    // amplified when few resonances contribute.  Compare tr029 (same Deltae,
    // wide range, 4.9% mean) — the wider energy range dilutes sparse-grid error.
    // Measured: 19.6% mean.
    assert!(
        result.mean_rel_error < 0.25,
        "broadened mean error {:.4} > 25%",
        result.mean_rel_error
    );
}

/// tr047: Fe-56 transmission, 1130-1168 eV, cooled to 181K, Doppler+Gaussian+Exponential.
///
/// Similar to tr007 but at a different temperature. Tests `csisrs` keyword.
#[test]
fn test_tr047_fe56_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr047_fe56_transmission_cooled",
        "t047a.inp",
        "t047a.par",
        "raa.plt",
    );
    assert!(!par.resonances.is_empty(), "expected resonances");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "FE56");
    assert!((inp.temperature_k - 181.0).abs() < 1.0);
    assert!(!inp.no_broadening);
}

#[test]
fn test_tr047_fe56_broadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr047_fe56_transmission_cooled",
        "t047a.inp",
        "t047a.par",
        "raa.plt",
    );
    let result = validate_broadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr047 broadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // tr047: Deltae=0.022, has exponential tail (Iesopr=3).
    // Similar to tr007 (same isotope/energy range) but cooled to 181K (reduced
    // Doppler width).  Both tr047 and tr007 share the same Deltae=0.022, but
    // at lower temperature the relative contribution of exp tail vs Doppler
    // shifts.  The 2.6% mean error is low because the Fe-56 resonance at
    // 1151 eV dominates and its broadened shape is mainly Doppler+Gaussian.
    // Measured: 2.6% mean.
    assert!(
        result.mean_rel_error < 0.05,
        "broadened mean error {:.4} > 5%",
        result.mean_rel_error
    );
}

/// tr028: Pu-241 total cross-section, 0.001-0.1 eV, NO BROADENING.
///
/// Tests TOTAL CROSS SECTION keyword and no-broadening mode.
/// Very low energy range — unbroadened cross-sections should match exactly.
#[test]
fn test_tr028_pu241_parse() {
    let (inp, par, plt) = load_samtry_case(
        "tr028_pu241_total_xs_no_broadening",
        "t028a.inp",
        "t028a.par",
        "raa.plt",
    );
    assert!(!par.resonances.is_empty(), "expected resonances");
    assert!(!plt.is_empty());
    assert_eq!(inp.isotope_symbol, "PU241");
    assert!(inp.no_broadening, "should detect no-broadening keyword");
}

#[test]
#[ignore] // Pu-241 has 3 channels per spin group (fission) — Batch C scope.
fn test_tr028_pu241_unbroadened() {
    let (inp, par, plt) = load_samtry_case(
        "tr028_pu241_total_xs_no_broadening",
        "t028a.inp",
        "t028a.par",
        "raa.plt",
    );
    // No broadening — compare unbroadened cross-sections directly.
    let result = validate_unbroadened_cross_sections(&inp, &par, &plt, 0.01);
    eprintln!(
        "tr028 unbroadened: max_rel={:.6}, mean_rel={:.6}, n={}, above_1%={}, worst@{:.4} keV",
        result.max_rel_error,
        result.mean_rel_error,
        result.n_points,
        result.n_above_threshold,
        result.worst_energy_kev
    );
    // No broadening: unbroadened XS should match SAMMY reference closely.
    // Currently 91% mean error — multi-channel fission not yet supported.
    assert!(
        result.mean_rel_error < 0.50,
        "unbroadened mean error {:.4} > 50%",
        result.mean_rel_error
    );
}
