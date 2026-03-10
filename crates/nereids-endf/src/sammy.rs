//! SAMMY test-suite file parsers.
//!
//! Parses SAMMY `.par` (resonance parameters), `.inp` (input configuration),
//! and `.plt` (plot reference output) files from the `samtry/` test suite.
//! These parsers are intentionally minimal — they cover the subset of SAMMY's
//! format needed for Phase 1 transmission validation (issue #292).
//!
//! ## SAMMY Reference
//! - SAMMY Manual, Section 2 (input file format)
//! - SAMMY Manual, Section 8 (parameter file format)
//! - `samtry/tr007/`, `samtry/tr004/`, etc. for concrete examples

use std::fmt;

use nereids_core::types::Isotope;

use crate::resonance::{
    LGroup, RExternalEntry, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
};

// ─── Error type ────────────────────────────────────────────────────────────────

/// Error from parsing a SAMMY file.
#[derive(Debug)]
pub struct SammyParseError {
    pub message: String,
}

impl fmt::Display for SammyParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SAMMY parse error: {}", self.message)
    }
}

impl std::error::Error for SammyParseError {}

impl SammyParseError {
    fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

// ─── .par file types ───────────────────────────────────────────────────────────

/// A single resonance from a SAMMY `.par` file.
#[derive(Debug, Clone)]
pub struct SammyResonance {
    /// Resonance energy (eV).
    pub energy_ev: f64,
    /// Radiation (gamma) width Γ_γ (eV). Converted from meV in file.
    pub gamma_gamma_ev: f64,
    /// Neutron width Γ_n (eV). Converted from meV in file.
    pub gamma_n_ev: f64,
    /// First fission width (eV). Zero for non-fissile isotopes.
    pub gamma_f1_ev: f64,
    /// Second fission width (eV). Zero for non-fissile isotopes.
    pub gamma_f2_ev: f64,
    /// Spin group index (1-based, references `.inp` Card Set 10).
    pub spin_group: u32,
}

/// Per-isotope mass and abundance entry from "ISOTOPIC MASSES AND ABUNDANCES
/// FOLLOW" section in a SAMMY `.par` file.
///
/// SAMMY Ref: `IsotopicMassesAndAbundances.cpp`, Manual Section II.B.1.e
#[derive(Debug, Clone)]
pub struct IsotopicMass {
    /// Atomic weight ratio (target mass / neutron mass) for this isotope.
    pub awr: f64,
    /// Fractional abundance (number fraction) of this isotope in the sample.
    pub abundance: f64,
    /// 1-based spin group indices assigned to this isotope.
    pub spin_groups: Vec<u32>,
}

/// Parsed SAMMY `.par` file.
#[derive(Debug, Clone)]
pub struct SammyParFile {
    pub resonances: Vec<SammyResonance>,
    /// Per-spin-group channel radius overrides from "RADIUS PARAMETERS FOLLOW".
    /// Maps 1-based spin group index → effective channel radius in fm.
    /// When present, overrides the global scattering radius from the .inp file.
    pub radius_overrides: std::collections::HashMap<u32, f64>,
    /// Per-spin-group R-external parameters from "R-EXTERNAL PARAMETERS FOLLOW".
    /// Maps 1-based spin group index → 7 R-external parameters
    /// [E_low, E_up, R_con, R_lin, s_con, s_lin, R_quad].
    ///
    /// SAMMY Ref: mpar03.f90 Readrx, Manual Section II.B.1.d
    pub r_external: std::collections::HashMap<u32, [f64; 7]>,
    /// Per-isotope AWR and abundance from "ISOTOPIC MASSES AND ABUNDANCES FOLLOW".
    /// When present, overrides .inp spin group abundances and provides per-isotope
    /// AWR values (the .inp only has a global AWR from Card 2).
    ///
    /// SAMMY Ref: `IsotopicMassesAndAbundances.cpp`
    pub isotopic_masses: Vec<IsotopicMass>,
}

// ─── .inp file types ───────────────────────────────────────────────────────────

/// Spin group definition from `.inp` Card Set 10.
#[derive(Debug, Clone)]
pub struct SammySpinGroup {
    /// 1-based group index.
    pub index: u32,
    /// Total angular momentum J.  May be negative per SAMMY sign convention
    /// (negative J distinguishes spin groups with the same |J|).
    pub j: f64,
    /// Orbital angular momentum L (from the channel line).
    pub l: u32,
    /// Statistical weight / abundance.
    pub abundance: f64,
    /// Per-spin-group target spin I, parsed from columns [30:35] of each
    /// header line.  Defaults to 0.0 (even-even nuclei).
    pub target_spin: f64,
    /// Optional isotope label from the header line (columns ~52+),
    /// e.g. "Cu65", "Cu63".  Present in multi-isotope SAMMY cases.
    pub isotope_label: Option<String>,
}

/// Observation type parsed from the SAMMY `.inp` keyword block.
///
/// Determines what physical quantity SAMMY's Th_initial column represents:
/// - Transmission/TotalCrossSection → σ_total
/// - Fission → σ_fission
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SammyObservationType {
    #[default]
    Transmission,
    TotalCrossSection,
    Fission,
}

/// Beamline and sample configuration from a SAMMY `.inp` file.
#[derive(Debug, Clone)]
pub struct SammyInpConfig {
    pub title: String,
    /// Isotope symbol as written in the file (e.g. "60NI", "FE56").
    pub isotope_symbol: String,
    /// Atomic weight ratio (target mass / neutron mass).
    pub awr: f64,
    /// Lower energy bound for resonance range (eV).
    pub energy_min_ev: f64,
    /// Upper energy bound for resonance range (eV).
    pub energy_max_ev: f64,
    /// Sample temperature (K).
    pub temperature_k: f64,
    /// Flight path length (m).
    pub flight_path_m: f64,
    /// Card 5, field 3: Deltal — flight path uncertainty (SAMMY units).
    ///
    /// Maps to rslRes parameter 1 → Bo2 in SAMMY's resolution broadening.
    /// SAMMY Ref: `minp06.f90` line 226.
    pub delta_l_sammy: f64,
    /// Card 5, field 4: Deltae — exponential tail parameter (SAMMY units).
    ///
    /// Maps to rslRes parameter 3 → Co2 in SAMMY's resolution broadening.
    /// When zero, no exponential broadening is applied (Iesopr=1, pure Gaussian).
    pub delta_e_sammy: f64,
    /// Card 5, field 5: Deltag — timing uncertainty (SAMMY units).
    ///
    /// Maps to rslRes parameter 2 → Ao2 in SAMMY's resolution broadening.
    pub delta_g_sammy: f64,
    /// When true, broadening is explicitly disabled for this case
    /// (e.g., `BROADENING IS NOT WANTED` or `NO LOW-ENERGY BROADENING`).
    pub no_broadening: bool,
    /// BROADENING card override for Deltal (rslRes param 1).
    /// Non-zero values override Card 5 `delta_l_sammy`.
    /// SAMMY Ref: `minp18.f90` lines 89-94.
    pub broadening_delta_l: Option<f64>,
    /// BROADENING card override for Deltag (rslRes param 2).
    pub broadening_delta_g: Option<f64>,
    /// BROADENING card override for Deltae (rslRes param 3).
    pub broadening_delta_e: Option<f64>,
    /// Scattering radius (fm).
    pub scattering_radius_fm: f64,
    /// Sample thickness (atoms/barn).
    pub thickness_atoms_barn: f64,
    /// Target nuclear spin I.  Determines the statistical weight
    /// g_J = (2J+1) / ((2I+1)(2s+1)).  Parsed from field 6 of the
    /// spin group header in Card Set 10 (all headers should agree).
    /// Defaults to 0.0 (even-even nuclei like Fe-56, Ni-58, Ni-60).
    pub target_spin: f64,
    /// Spin group definitions.
    pub spin_groups: Vec<SammySpinGroup>,
    /// Observation type (Transmission, TotalCrossSection, or Fission).
    pub observation_type: SammyObservationType,
}

impl SammyInpConfig {
    /// Effective Deltal: BROADENING card override if non-zero, else Card 5.
    #[must_use]
    pub fn effective_delta_l(&self) -> f64 {
        self.broadening_delta_l
            .filter(|&v| v != 0.0)
            .unwrap_or(self.delta_l_sammy)
    }

    /// Effective Deltag: BROADENING card override if non-zero, else Card 5.
    #[must_use]
    pub fn effective_delta_g(&self) -> f64 {
        self.broadening_delta_g
            .filter(|&v| v != 0.0)
            .unwrap_or(self.delta_g_sammy)
    }

    /// Effective Deltae: BROADENING card override if non-zero, else Card 5.
    #[must_use]
    pub fn effective_delta_e(&self) -> f64 {
        self.broadening_delta_e
            .filter(|&v| v != 0.0)
            .unwrap_or(self.delta_e_sammy)
    }
}

/// Convert SAMMY resolution parameters to NEREIDS-convention values.
///
/// SAMMY stores resolution parameters in a different convention from NEREIDS:
///
/// | SAMMY coefficient | Formula | NEREIDS equivalent |
/// |---|---|---|
/// | Ao2 = (1.20112·Deltag / (Sm2·Dist))² | timing | `delta_t = Deltag / (2·√ln2)` |
/// | Bo2 = (0.81650·Deltal / Dist)² | path | `delta_l = Deltal / √6` |
///
/// where 1.20112 = 1/√ln2, 0.81650 = √(2/3), Sm2 = TOF_FACTOR = 72.298.
///
/// SAMMY Ref: `RslResolutionFunction_M.f90` (getAo2 lines 143-161, getBo2 lines 165-179)
///
/// Returns `None` if all effective Deltal, Deltag, and Deltae are zero
/// (no resolution broadening).
/// Otherwise returns `Some((flight_path_m, delta_t_us, delta_l_m, delta_e))`.
///
/// The fourth element `delta_e` is the exponential tail parameter (raw SAMMY
/// Deltae units, passed through without conversion). When non-zero, the
/// resolution kernel is the convolution of a Gaussian with an exponential
/// tail (SAMMY Iesopr=3).
#[must_use]
pub fn sammy_to_nereids_resolution(inp: &SammyInpConfig) -> Option<(f64, f64, f64, f64)> {
    if inp.no_broadening {
        return None;
    }

    let delta_l = inp.effective_delta_l();
    let delta_g = inp.effective_delta_g();
    let delta_e = inp.effective_delta_e();

    if delta_l == 0.0 && delta_g == 0.0 && delta_e == 0.0 {
        return None;
    }

    // Convert from SAMMY convention to NEREIDS convention.
    let delta_t_us = delta_g / (2.0 * 2.0_f64.ln().sqrt());
    let delta_l_m = delta_l / 6.0_f64.sqrt();
    // delta_e: no conversion needed — raw SAMMY Deltae maps directly to
    // NEREIDS's exp_width formula: Widexp = 2·Deltae·E^(3/2)/(TOF_FACTOR·L).
    // SAMMY Ref: RslResolutionFunction_M.f90 getCo2, mrsl4.f90 Wdsint.

    Some((inp.flight_path_m, delta_t_us, delta_l_m, delta_e))
}

// ─── .plt file types ───────────────────────────────────────────────────────────

/// A single record from a SAMMY `.plt` reference output file.
#[derive(Debug, Clone)]
pub struct SammyPltRecord {
    /// Energy (keV). SAMMY .plt files use keV, not eV.
    pub energy_kev: f64,
    /// Experimental data (cross-section in barns, or transmission).
    pub data: f64,
    /// Uncertainty on the data.
    pub uncertainty: f64,
    /// Theoretical value before fitting (our comparison target).
    pub theory_initial: f64,
    /// Theoretical value after fitting.
    pub theory_final: f64,
}

// ─── .plt parser ───────────────────────────────────────────────────────────────

/// Parse a SAMMY `.plt` file (5-column reference output with header).
///
/// Format:
/// ```text
///       Energy            Data      Uncertainty     Th_initial      Th_final
///    1.1330168       8.9078373      0.32005507       8.4964191       8.5014004
/// ```
pub fn parse_sammy_plt(content: &str) -> Result<Vec<SammyPltRecord>, SammyParseError> {
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip header line (contains non-numeric text).
        if trimmed.starts_with("Energy") || trimmed.contains("Th_initial") {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(SammyParseError::new(format!(
                "line {}: expected >=4 columns, got {}",
                i + 1,
                fields.len()
            )));
        }
        let parse = |s: &str, col: &str| {
            s.parse::<f64>().map_err(|e| {
                SammyParseError::new(format!("line {}: cannot parse {col}: {e}", i + 1))
            })
        };
        // Some .plt files have only 4 columns (no Th_final) when SAMMY
        // did not fit (e.g., "reconstruct cross sections" mode).
        records.push(SammyPltRecord {
            energy_kev: parse(fields[0], "energy")?,
            data: parse(fields[1], "data")?,
            uncertainty: parse(fields[2], "uncertainty")?,
            theory_initial: parse(fields[3], "theory_initial")?,
            theory_final: if fields.len() >= 5 {
                parse(fields[4], "theory_final")?
            } else {
                0.0
            },
        });
    }
    if records.is_empty() {
        return Err(SammyParseError::new("no data records found in .plt file"));
    }
    Ok(records)
}

// ─── .par parser ───────────────────────────────────────────────────────────────

/// Parse a Fortran-style floating-point literal.
///
/// Handles compact exponent notation without 'E': `5.400000-5` → `5.4e-5`,
/// `1.23+3` → `1.23e3`. Also handles Fortran D-exponent: `1.23D-5` → `1.23e-5`.
/// Standard notation (`1.23E-5`, `1.23`) is also handled.
fn parse_fortran_float(s: &str) -> Result<f64, std::num::ParseFloatError> {
    // Try standard parse first (fast path).
    if let Ok(v) = s.parse::<f64>() {
        return Ok(v);
    }
    // Handle Fortran D-exponent: "1.23D-5" or "1.23d+3" → replace D/d with E.
    if let Some(pos) = s.find(['D', 'd']) {
        let mut fixed = String::with_capacity(s.len());
        fixed.push_str(&s[..pos]);
        fixed.push('E');
        fixed.push_str(&s[pos + 1..]);
        return fixed.parse::<f64>();
    }
    // Look for a bare +/- exponent after a digit: "1.23-5" or "1.23+3".
    // Skip the first character to avoid matching a leading sign.
    if s.len() > 1 {
        for (i, c) in s[1..].char_indices() {
            let pos = i + 1; // position in original string
            if (c == '+' || c == '-') && s.as_bytes()[pos - 1].is_ascii_digit() {
                let mut fixed = String::with_capacity(s.len() + 1);
                fixed.push_str(&s[..pos]);
                fixed.push('E');
                fixed.push_str(&s[pos..]);
                return fixed.parse::<f64>();
            }
        }
    }
    s.parse::<f64>()
}

/// Parse a SAMMY `.par` file (resonance parameters).
///
/// Uses Fortran fixed-width column parsing (FORMAT 5F11.4, 5I2, I2):
///   cols  0-10: E_res (eV)         — 11 chars
///   cols 11-21: Γ_γ (meV)          — 11 chars
///   cols 22-32: Γ_n (meV)          — 11 chars
///   cols 33-43: Γ_f1 (meV)         — 11 chars
///   cols 44-54: Γ_f2 (meV)         — 11 chars
///   cols 55-64: vary flags (5×I2)  — 10 chars
///   cols 65-66: spin_group_id (I2) — 2 chars
///
/// Widths are in meV in the file; this function converts to eV.
///
/// SAMMY Ref: `ResonanceParameterIO.cpp`, `mrpti.f90`.
pub fn parse_sammy_par(content: &str) -> Result<SammyParFile, SammyParseError> {
    let mut resonances = Vec::new();

    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        // Stop at blank line or EXPLICIT keyword.
        if trimmed.is_empty() || trimmed.starts_with("EXPLICIT") {
            break;
        }

        // Need at least enough characters for E + Γ_γ + Γ_n (33 chars).
        if line.len() < 33 {
            return Err(SammyParseError::new(format!(
                "line {}: too short for .par format (need ≥33 chars, got {})",
                i + 1,
                line.len()
            )));
        }

        let parse_col = |start: usize, end: usize, name: &str| -> Result<f64, SammyParseError> {
            let s = if start >= line.len() {
                // Column starts past line end — treat missing as zero.
                ""
            } else {
                line[start..end.min(line.len())].trim()
            };
            if s.is_empty() {
                return Ok(0.0);
            }
            parse_fortran_float(s).map_err(|e| {
                SammyParseError::new(format!("line {}: cannot parse {name} ({s:?}): {e}", i + 1))
            })
        };

        let energy_ev = parse_col(0, 11, "E_res")?;
        let gamma_gamma_mev = parse_col(11, 22, "Γ_γ")?;
        let gamma_n_mev = parse_col(22, 33, "Γ_n")?;
        let gamma_f1_mev = parse_col(33, 44, "Γ_f1")?;
        let gamma_f2_mev = parse_col(44, 55, "Γ_f2")?;

        // Spin group index at cols 65-66 (I2).
        // The standard .par resonance line is FORMAT(5E11.4, 5I2, I2) = 67 columns.
        // Optional trailing data (e.g., energy uncertainties) starts at position 67.
        //
        // SAMMY convention (ResonanceParameterIO.cpp:217-220): negative spin group
        // means "exclude this resonance from the calculation" (setIncludeInCalc(false)).
        // We skip excluded resonances entirely.
        let spin_group_signed: i32 = if line.len() > 65 {
            let end = line.len().min(67);
            let sg_str = line[65..end].trim();
            if sg_str.is_empty() {
                1
            } else {
                sg_str.parse::<i32>().map_err(|e| {
                    SammyParseError::new(format!(
                        "line {}: cannot parse spin group ({sg_str:?}): {e}",
                        i + 1
                    ))
                })?
            }
        } else {
            1 // Default spin group if line is too short.
        };

        if spin_group_signed < 0 {
            // Negative spin group → excluded from calculation.
            continue;
        }
        let spin_group = spin_group_signed as u32;

        resonances.push(SammyResonance {
            energy_ev,
            gamma_gamma_ev: gamma_gamma_mev / 1000.0,
            gamma_n_ev: gamma_n_mev / 1000.0,
            gamma_f1_ev: gamma_f1_mev / 1000.0,
            gamma_f2_ev: gamma_f2_mev / 1000.0,
            spin_group,
        });
    }

    if resonances.is_empty() {
        return Err(SammyParseError::new("no resonances found in .par file"));
    }

    // Continue scanning for "RADIUS PARAMETERS FOLLOW" section after the
    // resonance block.  SAMMY par file layout:
    //   <resonances>
    //   <blank>
    //   <uncertainties/other sections>
    //   RADIUS PARAMETERS FOLLOW
    //   <radius lines>
    //   <blank>
    //
    // Ref: SAMMY AdjustedRadiusData.cpp readOldStyle
    let mut radius_overrides = std::collections::HashMap::new();
    let lines_vec: Vec<&str> = content.lines().collect();
    if let Some(rad_idx) = lines_vec
        .iter()
        .position(|l| l.trim().to_uppercase().starts_with("RADIUS"))
    {
        for rad_line in &lines_vec[rad_idx + 1..] {
            let trimmed = rad_line.trim();
            if trimmed.is_empty() || trimmed.starts_with(|c: char| c.is_alphabetic()) {
                break; // End of radius section.
            }

            // Old-style format (AdjustedRadiusData.cpp readOldStyle):
            //   Positions 0-9:   effective radius (F10)
            //   Positions 10-19: true radius (F10)
            //   Position  20:    chanOpt (I1) — ignored
            //   Position  21:    ifleff  (I1) — ignored
            //   Positions 22-23: ifltrue (I2) — ignored
            //   Positions 24+:   spin group indices (I2 each, 0 = terminator)
            if rad_line.len() < 20 {
                continue;
            }
            let r_eff: f64 = rad_line[..10].trim().parse().unwrap_or(0.0);
            if r_eff <= 0.0 {
                continue;
            }

            // Extract spin group indices from I2 fields starting at position 24.
            let group_str = if rad_line.len() > 24 {
                &rad_line[24..]
            } else {
                ""
            };
            let mut pos = 0;
            let mut found_group = false;
            while pos + 2 <= group_str.len() {
                let field = group_str[pos..pos + 2].trim();
                pos += 2;
                let val: i32 = match field.parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if val > 0 {
                    radius_overrides.insert(val as u32, r_eff);
                    found_group = true;
                } else if val == 0 && found_group {
                    break; // 0 after groups = terminator.
                }
            }

            // Fallback: if no I2 groups found but line has tokens after radii,
            // try whitespace-split parsing (handles short lines like "3.57 3.57 0 0 6").
            if !found_group {
                for tok in rad_line[20..].split_whitespace() {
                    if let Ok(v) = tok.parse::<i32>()
                        && v > 0
                    {
                        radius_overrides.insert(v as u32, r_eff);
                    }
                }
            }
        }
    }

    // Scan for "R-EXTERNAL PARAMETERS FOLLOW" section.
    //
    // SAMMY 7-parameter format (mpar03.f90 Readrx):
    //   I2 = spin group number (cols 1-2)
    //   I1 = channel number    (col  3)
    //   7×I1 = fit flags       (cols 4-10)
    //   7×F10.1 = parameters   (cols 11-80)
    //     P[0]=E_low, P[1]=E_up, P[2]=R_con, P[3]=R_lin,
    //     P[4]=s_con, P[5]=s_lin, P[6]=R_quad
    //
    // Ref: SAMMY Manual Section II.B.1.d, mpar03.f90 Readrx
    let mut r_external = std::collections::HashMap::new();
    if let Some(rext_idx) = lines_vec
        .iter()
        .position(|l| l.trim().starts_with("R-EXTERNAL PARAMETERS"))
    {
        for rext_line in &lines_vec[rext_idx + 1..] {
            let trimmed = rext_line.trim();
            if trimmed.is_empty() || trimmed.starts_with(|c: char| c.is_alphabetic()) {
                break; // End of R-external section.
            }

            // Need at least 10 chars (I2+I1+7×I1) + 70 chars (7×F10.1) = 80.
            if rext_line.len() < 40 {
                continue;
            }

            // Parse spin group index (I2, cols 0-1) and channel (I1, col 2).
            let sg: u32 = match rext_line[..2].trim().parse() {
                Ok(v) if v > 0 => v,
                _ => continue,
            };
            // Channel number (col 2) — we only use channel 1 (neutron elastic).
            // Fit flags (cols 3-9) — not needed for cross-section calculation.

            // Parse 7 parameters from cols 10+, each F10.1 (10 chars wide).
            // Allow flexible parsing: try fixed-width first, fall back to
            // whitespace-split for non-standard formatting.
            let param_str = &rext_line[10..];
            let mut params = [0.0f64; 7];
            let mut parsed_ok = true;

            // Try fixed-width F10.1 parsing (standard SAMMY format).
            if param_str.len() >= 70 {
                for (i, chunk) in (0..7).map(|i| (i, &param_str[i * 10..(i + 1) * 10])) {
                    match chunk.trim().parse::<f64>() {
                        Ok(v) => params[i] = v,
                        Err(_) => {
                            parsed_ok = false;
                            break;
                        }
                    }
                }
            } else {
                parsed_ok = false;
            }

            // Fallback: whitespace-delimited parsing.
            if !parsed_ok {
                let tokens: Vec<f64> = param_str
                    .split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect();
                // Require exactly 7 parameters; skip malformed entries.
                if tokens.len() != 7 {
                    continue;
                }
                for (i, &v) in tokens.iter().enumerate().take(7) {
                    params[i] = v;
                }
            }

            // Keep the FIRST entry per spin group (channel 1 = elastic).
            // Subsequent channels (fission, etc.) are intentionally ignored
            // since only the elastic R-external contribution is currently used.
            r_external.entry(sg).or_insert(params);
        }
    }

    // Scan for "ISOTOPIC MASSES AND ABUNDANCES FOLLOW" section.
    //
    // Format per line (fixed-width, same I2 spin group convention as RADIUS):
    //   cols  0- 9: AWR (F10)
    //   cols 10-19: abundance (F10)
    //   cols 20-29: uncertainty on abundance (F10, ignored)
    //   cols 30-31: flag (I2, ignored)
    //   cols 32+ : spin group indices (I2 each, 0 = terminator)
    //
    // SAMMY Ref: IsotopicMassesAndAbundances.cpp
    let mut isotopic_masses = Vec::new();
    if let Some(iso_idx) = lines_vec
        .iter()
        .position(|l| l.trim().to_uppercase().starts_with("ISOTOPIC"))
    {
        for iso_line in &lines_vec[iso_idx + 1..] {
            let trimmed = iso_line.trim();
            if trimmed.is_empty() || trimmed.starts_with(|c: char| c.is_alphabetic()) {
                break;
            }
            if iso_line.len() < 20 {
                continue;
            }

            let awr: f64 = iso_line[..10].trim().parse().map_err(|_| {
                SammyParseError::new(format!(
                    "ISOTOPIC MASSES: bad AWR field '{}'",
                    iso_line[..10].trim()
                ))
            })?;
            let abundance: f64 = iso_line[10..20].trim().parse().map_err(|_| {
                SammyParseError::new(format!(
                    "ISOTOPIC MASSES: bad abundance field '{}'",
                    iso_line[10..20].trim()
                ))
            })?;
            if awr <= 0.0 {
                continue;
            }

            // Parse spin group indices — I2 fields starting at position 32,
            // same convention as RADIUS PARAMETERS section.
            let group_start = 32.min(iso_line.len());
            let group_str = &iso_line[group_start..];
            let mut spin_groups = Vec::new();
            let mut pos = 0;
            let mut found_group = false;
            while pos + 2 <= group_str.len() {
                let field = group_str[pos..pos + 2].trim();
                pos += 2;
                let val: i32 = match field.parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if val > 0 {
                    spin_groups.push(val as u32);
                    found_group = true;
                } else if val == 0 && found_group {
                    break;
                }
            }

            // Fallback: whitespace-split parsing for non-standard formatting.
            if !found_group {
                for tok in iso_line[group_start..].split_whitespace() {
                    if let Ok(v) = tok.parse::<i32>()
                        && v > 0
                    {
                        spin_groups.push(v as u32);
                    }
                }
            }

            if !spin_groups.is_empty() {
                isotopic_masses.push(IsotopicMass {
                    awr,
                    abundance,
                    spin_groups,
                });
            }
        }
    }

    Ok(SammyParFile {
        resonances,
        radius_overrides,
        r_external,
        isotopic_masses,
    })
}

// ─── .inp parser ───────────────────────────────────────────────────────────────

/// Parse a SAMMY `.inp` file (minimal, for Phase 1 transmission cases).
///
/// Extracts: isotope info (Card 2), broadening params (Card 5), sample params
/// (Card 6), data type keyword, and spin group definitions (Card 10).
pub fn parse_sammy_inp(content: &str) -> Result<SammyInpConfig, SammyParseError> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 2 {
        return Err(SammyParseError::new(
            "file too short (need at least 2 lines)",
        ));
    }

    // Line 1: title.
    let title = lines[0].trim().to_string();

    // Line 2: isotope definition (Fortran FORMAT A10, F10.5, 2F10.1, I5, I5).
    //
    // Fixed-width columns:
    //   [0:10]  isotope symbol (may contain spaces, e.g. "CU 65")
    //   [10:20] AWR (atomic weight ratio)
    //   [20:30] Emin (lower energy bound, eV)
    //   [30:40] Emax (upper energy bound, eV)
    //
    // SAMMY Ref: `minp01.f90` Card Set 2 format.
    let iso_line = lines[1];
    if iso_line.len() < 30 {
        return Err(SammyParseError::new(format!(
            "line 2: too short for Card 2 fixed-width format (need ≥30 chars, got {})",
            iso_line.len()
        )));
    }
    let isotope_symbol = iso_line[..10.min(iso_line.len())].trim().to_string();
    let awr = iso_line[10..20.min(iso_line.len())]
        .trim()
        .parse::<f64>()
        .map_err(|e| SammyParseError::new(format!("line 2: AWR: {e}")))?;
    let energy_min_ev = iso_line[20..30.min(iso_line.len())]
        .trim()
        .parse::<f64>()
        .map_err(|e| SammyParseError::new(format!("line 2: Emin: {e}")))?;
    let energy_max_ev = if iso_line.len() > 30 {
        iso_line[30..40.min(iso_line.len())]
            .trim()
            .parse::<f64>()
            .map_err(|e| SammyParseError::new(format!("line 2: Emax: {e}")))?
    } else {
        // Some files may have a short line — fall back to Emin.
        energy_min_ev
    };

    // Scan for keyword commands (skip until blank line), then parse Card 5, 6.
    // Also look for TRANSMISSION keyword and spin group block.
    let mut temperature_k = 300.0;
    let mut flight_path_m = 0.0;
    let mut delta_l_sammy = 0.0;
    let mut delta_e_sammy = 0.0;
    let mut delta_g_sammy = 0.0;
    let mut scattering_radius_fm = 0.0;
    let mut thickness_atoms_barn = 0.0;
    let mut target_spin = 0.0;
    let mut spin_groups = Vec::new();
    let mut no_broadening = false;

    // State machine: find blank line after commands, then parse numeric cards.
    let mut idx = 2;
    // Scan command lines until first blank line, detecting special keywords.
    while idx < lines.len() {
        let trimmed = lines[idx].trim();
        if trimmed.is_empty() {
            idx += 1;
            break;
        }
        // Detect no-broadening keywords.
        let upper = trimmed.to_uppercase();
        // SAMMY allows abbreviated keywords: "BROADENING IS NOT WA" matches.
        if upper.starts_with("BROADENING IS NOT") || upper.contains("NO LOW-ENERGY BROADENING") {
            no_broadening = true;
        }
        // "INPUT IS ENDF/B FILE" and similar keywords are informational only.
        // NEREIDS uses its own ENDF parser; such keywords (and any optional
        // Mat= parameter on the same line) are ignored here.
        idx += 1;
    }

    // Card 5: broadening parameters (first numeric line after blank).
    //
    // SAMMY format (minp06.f90 line 226):
    //   READ (Iu22,10100) Temp, Dist, Deltal, Deltae, Deltag, ...
    //
    // - Temp: sample temperature (K)
    // - Dist: flight path length (m)
    // - Deltal: flight path uncertainty → rslRes param 1 → Bo2
    // - Deltae: exponential tail → rslRes param 3 → Co2
    // - Deltag: timing uncertainty → rslRes param 2 → Ao2
    //
    // When `no_broadening` is set AND the first numeric line has exactly 2
    // fields, Card 5 is absent — the line is Card 6 (scattering_radius,
    // thickness).  Example: tr034c.inp.
    // Card 5 (broadening params) is always present when broadening is active.
    // When "BROADENING IS NOT WANTED", Card 5 may or may not be present:
    // - tr028: no_broadening + Card 5 present (7 fields, next line is Card 6)
    // - tr018/tr034: no_broadening + Card 5 absent (2 fields, next is TRANSMISSION)
    // - tr037: no_broadening + Card 5 absent (4 fields, next is TRANSMISSION)
    //
    // Heuristic: when no_broadening, peek at the line AFTER the current one.
    // If that next line is also numeric (starts with digit/sign/dot), then
    // the current line is Card 5 (and the next is Card 6).  If the next
    // line is a keyword (starts with alpha), the current line is Card 6
    // (Card 5 was skipped).
    let card5_present = if idx < lines.len() {
        if no_broadening {
            // Look ahead to determine if Card 5 is present.
            if idx + 1 < lines.len() {
                let next_trimmed = lines[idx + 1].trim();
                next_trimmed
                    .bytes()
                    .next()
                    .is_some_and(|b| b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'+')
            } else {
                false
            }
        } else {
            true // Broadening active → Card 5 always present.
        }
    } else {
        false
    };

    if card5_present && idx < lines.len() {
        let card5_fields: Vec<&str> = lines[idx].split_whitespace().collect();
        if card5_fields.len() >= 2 {
            temperature_k = card5_fields[0]
                .parse::<f64>()
                .map_err(|e| SammyParseError::new(format!("Card 5: temperature: {e}")))?;
            flight_path_m = card5_fields[1]
                .parse::<f64>()
                .map_err(|e| SammyParseError::new(format!("Card 5: flight_path: {e}")))?;
            // SAMMY Card 5 field order: Temp, Dist, Deltal, Deltae, Deltag
            // Ref: SAMMY Users' Guide Table IVA.1.
            if card5_fields.len() >= 3 {
                delta_l_sammy = card5_fields[2].parse().unwrap_or(0.0);
            }
            if card5_fields.len() >= 4 {
                delta_e_sammy = card5_fields[3].parse().unwrap_or(0.0);
            }
            if card5_fields.len() >= 5 {
                delta_g_sammy = card5_fields[4].parse().unwrap_or(0.0);
            }
        }
        idx += 1;
    }

    // Card 6: scattering radius (required) and thickness (required).
    if idx < lines.len() {
        let card6_fields: Vec<&str> = lines[idx].split_whitespace().collect();
        if card6_fields.len() >= 2 {
            scattering_radius_fm = card6_fields[0]
                .parse::<f64>()
                .map_err(|e| SammyParseError::new(format!("Card 6: scattering_radius: {e}")))?;
            thickness_atoms_barn = card6_fields[1]
                .parse::<f64>()
                .map_err(|e| SammyParseError::new(format!("Card 6: thickness: {e}")))?;
        }
        idx += 1;
    }

    // Scan for observation keyword and spin group block, then BROADENING card.
    let mut broadening_delta_l: Option<f64> = None;
    let mut broadening_delta_g: Option<f64> = None;
    let mut broadening_delta_e: Option<f64> = None;
    let mut observation_type = SammyObservationType::default();

    while idx < lines.len() {
        let trimmed = lines[idx].trim().to_uppercase();
        // SAMMY allows keyword abbreviations: "TRANS" matches "TRANSMISSION",
        // "TOTAL" matches "TOTAL CROSS SECTION", "FISSI" matches "FISSION".
        if trimmed.starts_with("TRANS")
            || trimmed.starts_with("TOTAL")
            || trimmed.starts_with("FISSI")
            || trimmed.starts_with("CAPTU")
            || trimmed.starts_with("DIRAC")
        {
            if trimmed.starts_with("TOTAL") {
                observation_type = SammyObservationType::TotalCrossSection;
            } else if trimmed.starts_with("FISSI") {
                observation_type = SammyObservationType::Fission;
            }
            // NOTE: CAPTU (capture) and DIRAC observables are recognized so
            // the parser advances through their spin group definitions, but
            // observation_type intentionally remains Transmission (default).
            // Extend SammyObservationType when capture/Dirac physics are
            // implemented in NEREIDS.
            idx += 1;
            // Skip data-reduction parameter lines (SAMMY Card 8) that can
            // appear between the TRANSMISSION keyword and spin group
            // definitions.  Spin group headers have a positive integer in
            // columns 0-4; Card 8 lines have floats (e.g. "0.0 0.0 0 1"
            // in tr025).
            while idx < lines.len() {
                let l = lines[idx];
                let field = if l.len() >= 5 {
                    l[..5].trim()
                } else {
                    l.trim()
                };
                if field.is_empty() || field.parse::<u32>().is_ok() {
                    break;
                }
                idx += 1;
            }
            // Parse spin group definitions until blank line.
            let (groups, parsed_target_spin) = parse_spin_groups(&lines[idx..])?;
            spin_groups = groups;
            target_spin = parsed_target_spin;
            // Advance past spin group block.
            while idx < lines.len() && !lines[idx].trim().is_empty() {
                idx += 1;
            }
        } else if trimmed.starts_with("BROADENING") {
            // BROADENING card: next line has 6 floats + flags.
            //
            // SAMMY format (minp18.f90):
            //   Uuu(1-3) = channel_radius, Temp, thickness (broadenPars)
            //   Uuu(4-6) = Deltal, Deltag, Deltae (rslRes params 1-3)
            //
            // Non-zero Uuu(4-6) override Card 5 values.
            idx += 1;
            if idx < lines.len() {
                let brd_fields: Vec<&str> = lines[idx].split_whitespace().collect();
                if brd_fields.len() >= 6 {
                    if let Ok(v) = brd_fields[3].parse::<f64>()
                        && v != 0.0
                    {
                        broadening_delta_l = Some(v);
                    }
                    if let Ok(v) = brd_fields[4].parse::<f64>()
                        && v != 0.0
                    {
                        broadening_delta_g = Some(v);
                    }
                    if let Ok(v) = brd_fields[5].parse::<f64>()
                        && v != 0.0
                    {
                        broadening_delta_e = Some(v);
                    }
                }
            }
            // Only use the first BROADENING card (subsequent ones are for
            // AdjustableObject save/restore operations and may be empty).
            break;
        }
        idx += 1;
    }

    Ok(SammyInpConfig {
        title,
        isotope_symbol,
        awr,
        energy_min_ev,
        energy_max_ev,
        temperature_k,
        flight_path_m,
        delta_l_sammy,
        delta_e_sammy,
        delta_g_sammy,
        no_broadening,
        broadening_delta_l,
        broadening_delta_g,
        broadening_delta_e,
        scattering_radius_fm,
        thickness_atoms_barn,
        target_spin,
        spin_groups,
        observation_type,
    })
}

/// Parse spin group definitions from Card Set 10.
///
/// Each spin group has a header line and one or more channel lines:
/// ```text
///   1      1    0   .5       1.0   .0       <- group header
///     1    1    0    0      .500             <- channel line (indented)
///   2      1    0  -.5       1.0   .0
///     1    1    0    1      .500
/// ```
///
/// Header columns (unified for old and new format, verified against
/// SAMMY ResonanceParameterIO.cpp:931-938):
///   [0:5/0:3]   KKK   — spin group index
///   [5:10/7:10] NENT  — number of entrance channels
///   [10:15/12:15] NEXT — number of exit-only channels
///   [15:20] SPINJ     — J value (F5.1)
///   [20:30] ABNDNC    — abundance (F10.0)
///   [30:35] SPINI     — target spin (F5.1)
///   [36:]   label
///
/// Channel lines: L at cols 18-19 in both formats.
///
/// Returns (spin_groups, target_spin).
fn parse_spin_groups(lines: &[&str]) -> Result<(Vec<SammySpinGroup>, f64), SammyParseError> {
    let mut groups = Vec::new();
    let mut target_spin = 0.0;
    let mut i = 0;

    /// Extract a trimmed substring from a fixed-width field, returning "" if out of bounds.
    fn col(line: &str, start: usize, end: usize) -> &str {
        if start >= line.len() {
            return "";
        }
        let actual_end = end.min(line.len());
        line[start..actual_end].trim()
    }

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() {
            break; // End of spin group block.
        }

        let line = lines[i];

        // Spin group header line (see doc comment above for old/new layouts).
        //
        // Fields common to both formats: index, n_ent, n_exit occupy cols 0-14
        // (I5 in old, I3+gaps in new — but cols 0-14 parse identically by
        // trimming).  J is always at cols 15-19 (F5.1).
        //
        // Abundance: F10.0 at cols 20-29 (unified, both formats).
        // Target spin: F5.1 at cols 30-34 (unified, both formats).
        // SAMMY Ref: ResonanceParameterIO.cpp:931-938.
        //
        // Total channel lines following = n_ent + n_exit.
        let index: u32 = col(line, 0, 5)
            .parse()
            .map_err(|e| SammyParseError::new(format!("spin group index: {e}")))?;
        let n_ent: u32 = {
            let s = col(line, 5, 10);
            if s.is_empty() {
                1
            } else {
                s.parse()
                    .map_err(|e| SammyParseError::new(format!("spin group n_ent ({s:?}): {e}")))?
            }
        };
        let n_exit: u32 = {
            let s = col(line, 10, 15);
            if s.is_empty() {
                0
            } else {
                s.parse()
                    .map_err(|e| SammyParseError::new(format!("spin group n_exit ({s:?}): {e}")))?
            }
        };
        let j: f64 = col(line, 15, 20)
            .parse()
            .map_err(|e| SammyParseError::new(format!("spin group J: {e}")))?;
        // Preserve the sign: SAMMY uses negative J to distinguish spin
        // groups with the same |J|.  The sign keeps them in separate
        // J-groups during cross-section evaluation.
        // Abundance: 10 chars at cols 20-29 (F10.0) in both formats.
        // SAMMY Ref: ResonanceParameterIO.cpp:931 "10 spaces for abundance (21-30)"
        // (SAMMY uses 1-based column numbering; 0-based = 20-29).
        let abund_end = 30;
        let abundance: f64 = {
            let s = col(line, 20, abund_end);
            if s.is_empty() {
                1.0
            } else {
                s.parse().map_err(|e| {
                    SammyParseError::new(format!("spin group abundance ({s:?}): {e}"))
                })?
            }
        };

        // Target spin: 5 chars at cols 30-34 in both formats.
        // SAMMY Ref: ResonanceParameterIO.cpp:938 "5 spaces for target spin (31-35)"
        let tspin_start = 30;
        let group_target_spin = {
            let s = col(line, tspin_start, tspin_start + 5);
            if s.is_empty() {
                0.0
            } else {
                s.parse::<f64>().map_err(|e| {
                    SammyParseError::new(format!("spin group target spin ({s:?}): {e}"))
                })?
            }
        };
        // Use first group's target spin as the global default.
        if groups.is_empty() {
            target_spin = group_target_spin;
        }

        // Optional isotope label: text after column 36, trimmed.
        // Multi-isotope cases (e.g. tr034) store labels like "Cu65", "Cu63".
        let isotope_label = if line.len() > 36 {
            let s = line[36..].trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        } else {
            None
        };

        // Extract L from the first channel line following this header.
        //
        // SAMMY Card Set 10.1 (old format) channel card columns
        // (from ResonanceParameterIO.cpp:readOldChannelData):
        //   cols  0-2: spin group index (I3)
        //   cols  3-4: channel ID (I2)
        //   cols  5-7: kz1 (charge, usually blank)
        //   cols  8-9: lpent
        //   cols 10-12: kz2 (charge, usually blank)
        //   cols 13-14: ishift
        //   cols 15-17: ifexcl (usually blank)
        //   cols 18-19: L (orbital angular momentum)  ← this is what we need
        //   cols 20-29: channel spin
        //
        // Use fixed-width column extraction (cols 18-20) instead of
        // whitespace split, which breaks when blank fields (kz1, kz2,
        // ifexcl) are present or absent.
        let n_channels = n_ent + n_exit;
        let mut l = 0u32;
        if i + 1 < lines.len() {
            let l_str = col(lines[i + 1], 18, 20);
            if !l_str.is_empty() {
                l = l_str.parse().unwrap_or(0);
            }
        }

        groups.push(SammySpinGroup {
            index,
            j,
            l,
            abundance,
            target_spin: group_target_spin,
            isotope_label,
        });

        // Skip channel lines: n_ent + n_exit lines follow the header.
        i += 1 + n_channels as usize;
    }

    Ok((groups, target_spin))
}

// ─── Converter: SAMMY → NEREIDS ResonanceData ──────────────────────────────────

/// Small dimensionless perturbation added to J values to distinguish spin groups
/// with the same |J| but different channel structure.
const CHANNEL_OFFSET: f64 = 1e-6;

/// Compute J-value offsets for spin groups that share the same (L, |J|) pair.
///
/// For targets with non-zero spin (I > 0), multiple spin groups can have
/// the same (L, |J|) but represent different entrance channels (different
/// channel spin s = I +/- 1/2). These must be kept separate in the R-matrix
/// calculation. We assign a small perturbation (1e-6 per duplicate) to
/// distinguish them.
///
/// For I = 0 (even-even nuclei), duplicate (L, J) spin groups represent
/// pseudo-isotope contaminants, not extra channels, so no offset is applied.
///
/// Ref: SAMMY Manual Eq. III A.1, sum over entrance channels c.
fn compute_j_offsets(
    spin_groups: &[SammySpinGroup],
    target_spin: f64,
) -> std::collections::HashMap<u32, f64> {
    let mut sg_j_offset = std::collections::HashMap::new();
    if target_spin.abs() <= 1e-10 {
        return sg_j_offset;
    }
    let mut lj_seen: std::collections::HashMap<(u32, i64), u32> = std::collections::HashMap::new();
    for sg in spin_groups {
        let j_key = (sg.j.abs() * 1_000_000.0).round() as i64;
        let count = lj_seen.entry((sg.l, j_key)).or_insert(0);
        if *count > 0 {
            sg_j_offset.insert(sg.index, *count as f64 * CHANNEL_OFFSET);
        }
        *count += 1;
    }
    sg_j_offset
}

/// Convert parsed SAMMY `.par` + `.inp` into a NEREIDS `ResonanceData`.
///
/// Groups resonances by spin group, maps each group to an `LGroup` using the
/// spin group definitions from the `.inp` file.
pub fn sammy_to_resonance_data(
    inp: &SammyInpConfig,
    par: &SammyParFile,
) -> Result<ResonanceData, SammyParseError> {
    if inp.spin_groups.is_empty() {
        return Err(SammyParseError::new("no spin groups defined in .inp file"));
    }

    // Build a map: spin_group_index → (L, J, abundance).
    let group_map: std::collections::HashMap<u32, &SammySpinGroup> =
        inp.spin_groups.iter().map(|sg| (sg.index, sg)).collect();

    // Group resonances by L-value (NEREIDS groups by L, not by spin group).
    // Within each L-group, resonances carry their own J value.
    let mut l_group_map: std::collections::BTreeMap<u32, Vec<Resonance>> =
        std::collections::BTreeMap::new();

    // Multi-channel J offset: for targets with I > 0, perturb duplicate
    // (L, |J|) spin groups so each entrance channel gets its own J-group.
    // See `compute_j_offsets` doc comment for full explanation.
    let sg_j_offset = compute_j_offsets(&inp.spin_groups, inp.target_spin);

    for res in &par.resonances {
        let sg = group_map.get(&res.spin_group).ok_or_else(|| {
            SammyParseError::new(format!(
                "resonance at E={} eV references undefined spin group {}",
                res.energy_ev, res.spin_group
            ))
        })?;

        let j = sg.j + sg_j_offset.get(&sg.index).copied().unwrap_or(0.0);
        l_group_map.entry(sg.l).or_default().push(Resonance {
            energy: res.energy_ev,
            j,
            gn: res.gamma_n_ev,
            gg: res.gamma_gamma_ev,
            gfa: res.gamma_f1_ev,
            gfb: res.gamma_f2_ev,
        });
    }

    // Inject zero-width sentinel resonances for spin groups that have no
    // resonances in the .par file.  These spin groups still contribute
    // potential scattering: R=0 → U = e^{-2iφ_L} → σ_pot ≠ 0.
    // Without sentinels, the L-group (or J-group within it) is never
    // created, and the potential scattering is silently lost.
    let sg_indices_with_resonances: std::collections::HashSet<u32> =
        par.resonances.iter().map(|r| r.spin_group).collect();
    for sg in &inp.spin_groups {
        if !sg_indices_with_resonances.contains(&sg.index) {
            let j = sg.j + sg_j_offset.get(&sg.index).copied().unwrap_or(0.0);
            l_group_map.entry(sg.l).or_default().push(Resonance {
                energy: 0.0,
                j,
                gn: 0.0,
                gg: 0.0,
                gfa: 0.0,
                gfb: 0.0,
            });
        }
    }

    // Build per-L-group radius from par file "RADIUS PARAMETERS FOLLOW".
    // SAMMY assigns radii per spin group; NEREIDS uses per-L-group (apl).
    // Map: L → radius by looking up each spin group's L value and its
    // assigned radius.  All spin groups with the same L should have the
    // same radius (verified in SAMMY test cases).
    //
    // Ref: SAMMY AdjustedRadiusData.cpp readOldStyle
    let mut l_radius_map: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for sg in &inp.spin_groups {
        if let Some(&r) = par.radius_overrides.get(&sg.index) {
            l_radius_map.entry(sg.l).or_insert(r);
        }
    }

    // Build LGroups.
    let l_groups: Vec<LGroup> = l_group_map
        .into_iter()
        .map(|(l, resonances)| {
            // Use per-L-group radius if available, else fall back to global
            // (apl=0.0 means "use range.scattering_radius").
            let apl = l_radius_map.get(&l).copied().unwrap_or(0.0);
            LGroup {
                l,
                awr: inp.awr,
                apl,
                qx: 0.0,
                lrx: 0,
                resonances,
            }
        })
        .collect();

    // Use the target spin parsed from the spin group header (field 6).
    // For even-even nuclei (Fe-56, Ni-58, Ni-60) this is 0.0.
    let target_spin = inp.target_spin;

    // Infer Z and A from the isotope symbol (e.g. "60NI" → Z=28, A=60;
    // "FE56" → Z=26, A=56; "58NI" → Z=28, A=58; "NatFE" → Z=26, A=0).
    let (z, mut a) = parse_isotope_symbol(&inp.isotope_symbol)?;
    // For natural-element symbols (A=0), approximate A from AWR.
    if a == 0 {
        a = inp.awr.round() as u32;
    }
    let za = z * 1000 + a;

    // Use a wide energy range so that cross-sections can be evaluated at any
    // energy in the .plt reference file, which may extend beyond the .inp
    // analysis window.  SAMMY's .inp energy limits define the fitting region,
    // not the physics validity bounds of the resonance parameters.
    // R-external is NOT included in the single-isotope path.
    //
    // The single-isotope path merges ALL spin groups (including minor isotopes)
    // into one ResonanceRange.  R-external entries from one isotope's spin groups
    // could erroneously match another isotope's (L, J) group, corrupting the
    // cross section.  Use sammy_to_resonance_data_multi() for cases with R-ext.
    let range = ResonanceRange {
        energy_low: 1e-5,
        energy_high: 2e7,
        resolved: true,
        formalism: ResonanceFormalism::ReichMoore,
        target_spin,
        scattering_radius: inp.scattering_radius_fm,
        naps: 0,
        ap_table: None,
        l_groups,
        rml: None,
        urr: None,
        r_external: vec![],
    };

    Ok(ResonanceData {
        isotope: Isotope::new(z, a)
            .map_err(|e| SammyParseError::new(format!("invalid isotope: {e}")))?,
        za,
        awr: inp.awr,
        ranges: vec![range],
    })
}

/// Convert parsed SAMMY `.par` + `.inp` into multiple `(ResonanceData, abundance)` pairs.
///
/// Groups spin groups into "pseudo-isotopes" by isotope label (if present) or
/// by (abundance, target_spin).  Each group gets its own `ResonanceData` with
/// only the resonances belonging to that group's spin groups.
///
/// This is needed for multi-isotope SAMMY cases like tr024 (NatFe: Fe-56 +
/// Fe-54 + Fe-57) and tr034 (Cu-65 + Cu-63).
pub fn sammy_to_resonance_data_multi(
    inp: &SammyInpConfig,
    par: &SammyParFile,
) -> Result<Vec<(ResonanceData, f64)>, SammyParseError> {
    if inp.spin_groups.is_empty() {
        return Err(SammyParseError::new("no spin groups defined in .inp file"));
    }

    // Determine grouping key for each spin group.
    // If ANY spin group has an isotope_label, group by label.
    // Otherwise, group by (abundance, target_spin) using a string key.
    let has_labels = inp.spin_groups.iter().any(|sg| sg.isotope_label.is_some());

    // Build grouping: key → (spin_group_indices, abundance, target_spin).
    let mut group_order: Vec<String> = Vec::new();
    let mut group_members: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    let mut group_abundance: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    let mut group_target_spin: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();

    for (i, sg) in inp.spin_groups.iter().enumerate() {
        let key = if has_labels {
            sg.isotope_label
                .as_deref()
                .unwrap_or("unknown")
                .to_uppercase()
        } else {
            format!("{:.6}_{:.1}", sg.abundance, sg.target_spin)
        };

        if !group_members.contains_key(&key) {
            group_order.push(key.clone());
        }
        group_members.entry(key.clone()).or_default().push(i);
        if let Some(&existing) = group_abundance.get(&key)
            && (existing - sg.abundance).abs() >= 1e-12
        {
            return Err(SammyParseError::new(format!(
                "spin groups in isotope group '{}' disagree on abundance: {} vs {}",
                key, existing, sg.abundance
            )));
        }
        group_abundance.insert(key.clone(), sg.abundance);
        if let Some(&existing) = group_target_spin.get(&key)
            && (existing - sg.target_spin).abs() >= 1e-12
        {
            return Err(SammyParseError::new(format!(
                "spin groups in isotope group '{}' disagree on target_spin: {} vs {}",
                key, existing, sg.target_spin
            )));
        }
        group_target_spin.insert(key.clone(), sg.target_spin);
    }

    // Build spin_group_index → SammySpinGroup map.
    let sg_map: std::collections::HashMap<u32, &SammySpinGroup> =
        inp.spin_groups.iter().map(|sg| (sg.index, sg)).collect();

    // Build spin_group → (awr, abundance) from ISOTOPIC MASSES section if present.
    // These override the .inp spin group abundances and provide per-isotope AWR.
    let mut sg_iso_awr: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    let mut sg_iso_abund: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for im in &par.isotopic_masses {
        for &sg_idx in &im.spin_groups {
            sg_iso_awr.insert(sg_idx, im.awr);
            sg_iso_abund.insert(sg_idx, im.abundance);
        }
    }

    // Z from Card 2 (shared across all groups).
    let (card2_z, card2_a) = parse_isotope_symbol(&inp.isotope_symbol)?;

    let mut result = Vec::new();

    for key in &group_order {
        let member_indices = &group_members[key];
        let target_spin = group_target_spin[key];

        // Use ISOTOPIC MASSES abundance when available (overrides .inp spin
        // group headers).  Fall back to .inp abundance.
        let first_sg_idx = inp.spin_groups[member_indices[0]].index;
        let abundance = sg_iso_abund
            .get(&first_sg_idx)
            .copied()
            .unwrap_or(group_abundance[key]);

        // Collect the spin group indices (1-based) for this group.
        let sg_indices: std::collections::HashSet<u32> = member_indices
            .iter()
            .map(|&i| inp.spin_groups[i].index)
            .collect();

        // Filter resonances belonging to this group.
        let group_resonances: Vec<&SammyResonance> = par
            .resonances
            .iter()
            .filter(|r| sg_indices.contains(&r.spin_group))
            .collect();

        // Multi-channel J offset for this isotope group's spin groups.
        // See `compute_j_offsets` doc comment for full explanation.
        let member_sgs: Vec<SammySpinGroup> = member_indices
            .iter()
            .map(|&i| inp.spin_groups[i].clone())
            .collect();
        let sg_j_offset = compute_j_offsets(&member_sgs, target_spin);

        // Build L-groups from these resonances.
        let mut l_group_map: std::collections::BTreeMap<u32, Vec<Resonance>> =
            std::collections::BTreeMap::new();

        for res in &group_resonances {
            let sg = sg_map.get(&res.spin_group).ok_or_else(|| {
                SammyParseError::new(format!(
                    "resonance at E={} eV references undefined spin group {}",
                    res.energy_ev, res.spin_group
                ))
            })?;

            let j = sg.j + sg_j_offset.get(&sg.index).copied().unwrap_or(0.0);
            l_group_map.entry(sg.l).or_default().push(Resonance {
                energy: res.energy_ev,
                j,
                gn: res.gamma_n_ev,
                gg: res.gamma_gamma_ev,
                gfa: res.gamma_f1_ev,
                gfb: res.gamma_f2_ev,
            });
        }

        // Inject zero-width sentinels for spin groups in this isotope
        // group that have no resonances (same fix as sammy_to_resonance_data).
        let res_sg_indices: std::collections::HashSet<u32> =
            group_resonances.iter().map(|r| r.spin_group).collect();
        for &idx in &sg_indices {
            if !res_sg_indices.contains(&idx)
                && let Some(sg) = sg_map.get(&idx)
            {
                let j = sg.j + sg_j_offset.get(&sg.index).copied().unwrap_or(0.0);
                l_group_map.entry(sg.l).or_default().push(Resonance {
                    energy: 0.0,
                    j,
                    gn: 0.0,
                    gg: 0.0,
                    gfa: 0.0,
                    gfb: 0.0,
                });
            }
        }

        // Build per-L radius map for this isotope group's spin groups.
        let mut l_radius_map: std::collections::HashMap<u32, f64> =
            std::collections::HashMap::new();
        for &idx in &sg_indices {
            if let Some(sg) = sg_map.get(&idx)
                && let Some(&r) = par.radius_overrides.get(&sg.index)
            {
                l_radius_map.entry(sg.l).or_insert(r);
            }
        }

        // Per-isotope AWR: use ISOTOPIC MASSES value when available,
        // fall back to .inp global AWR from Card 2.
        let isotope_awr = sg_iso_awr.get(&first_sg_idx).copied().unwrap_or(inp.awr);

        let l_groups: Vec<LGroup> = l_group_map
            .into_iter()
            .map(|(l, resonances)| {
                let apl = l_radius_map.get(&l).copied().unwrap_or(0.0);
                LGroup {
                    l,
                    awr: isotope_awr,
                    apl,
                    qx: 0.0,
                    lrx: 0,
                    resonances,
                }
            })
            .collect();

        // Determine Z/A for this group.
        let (z, mut a) = if has_labels {
            // Try to parse the isotope label (e.g. "Cu65" → Z=29, A=65).
            parse_isotope_symbol(key).unwrap_or((card2_z, card2_a))
        } else {
            (card2_z, card2_a)
        };
        // For natural-element symbols (A=0), approximate A from AWR.
        if a == 0 {
            a = isotope_awr.round() as u32;
        }
        let za = z * 1000 + a;

        // Build R-external entries for this isotope group's spin groups.
        let mut r_external_entries = Vec::new();
        for &idx in &sg_indices {
            if let Some(sg) = sg_map.get(&idx)
                && let Some(params) = par.r_external.get(&sg.index)
            {
                let j = sg.j + sg_j_offset.get(&sg.index).copied().unwrap_or(0.0);
                r_external_entries.push(RExternalEntry {
                    l: sg.l,
                    j,
                    e_low: params[0],
                    e_up: params[1],
                    r_con: params[2],
                    r_lin: params[3],
                    s_con: params[4],
                    s_lin: params[5],
                    r_quad: params[6],
                });
            }
        }

        let range = ResonanceRange {
            energy_low: 1e-5,
            energy_high: 2e7,
            resolved: true,
            formalism: ResonanceFormalism::ReichMoore,
            target_spin,
            scattering_radius: inp.scattering_radius_fm,
            naps: 0,
            ap_table: None,
            l_groups,
            rml: None,
            urr: None,
            r_external: r_external_entries,
        };

        let resonance_data = ResonanceData {
            isotope: Isotope::new(z, a)
                .map_err(|e| SammyParseError::new(format!("invalid isotope: {e}")))?,
            za,
            awr: isotope_awr,
            ranges: vec![range],
        };

        result.push((resonance_data, abundance));
    }

    Ok(result)
}

/// Parse a SAMMY isotope symbol like "60NI", "FE56", "58NI", "NatFE",
/// "CU 65", "Cu65" into (Z, A).
///
/// For natural-element symbols like "NatFE", A=0 is returned (no specific
/// mass number).  Space-separated labels like "CU 65" are handled by
/// stripping internal spaces.
pub(crate) fn parse_isotope_symbol(symbol: &str) -> Result<(u32, u32), SammyParseError> {
    // Strip internal spaces and hyphens (handles "CU 65" → "CU65", "Rh-103" → "RH103").
    let joined: String = symbol
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .collect();
    let s = joined.to_uppercase();

    // Handle "NAT" prefix: "NATFE" → Z from element, A=0.
    if let Some(rest) = s.strip_prefix("NAT")
        && let Some(z) = element_symbol_to_z(rest)
    {
        return Ok((z, 0));
    }

    // Try format: "60NI" (mass number first, then element symbol).
    // Extract just the leading alphabetic chars as element symbol to handle
    // labels with trailing comments (e.g. "94Zr ... FAKES" → "ZR").
    if let Some(split_pos) = s.find(|c: char| c.is_ascii_alphabetic()) {
        let mass_str = &s[..split_pos];
        let remaining = &s[split_pos..];
        let alpha_len = remaining
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(remaining.len());
        let elem_str = &remaining[..alpha_len];
        if !mass_str.is_empty()
            && let Ok(a) = mass_str.parse::<u32>()
            && let Some(z) = element_symbol_to_z(elem_str)
        {
            return Ok((z, a));
        }
    }

    // Try format: "FE56" (element symbol first, then mass number).
    // Extract just the leading digit chars to handle trailing garbage.
    if let Some(split_pos) = s.find(|c: char| c.is_ascii_digit()) {
        let elem_str = &s[..split_pos];
        let remaining = &s[split_pos..];
        let digit_len = remaining
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(remaining.len());
        let mass_str = &remaining[..digit_len];
        if let Ok(a) = mass_str.parse::<u32>()
            && let Some(z) = element_symbol_to_z(elem_str)
        {
            return Ok((z, a));
        }
    }

    // Try bare element symbol without mass number (e.g. "Al" → Z=13, A=0).
    if let Some(z) = element_symbol_to_z(&s) {
        return Ok((z, 0));
    }

    // Try full element name (e.g. "ZIRCONIUM" → Z=40, A=0 natural).
    if let Some(z) = element_name_to_z(&s) {
        return Ok((z, 0));
    }

    Err(SammyParseError::new(format!(
        "cannot parse isotope symbol: {symbol}"
    )))
}

/// Map element symbol (uppercase) to atomic number Z.
///
/// Covers all elements Z=1 (H) through Z=100 (Fm).
fn element_symbol_to_z(symbol: &str) -> Option<u32> {
    match symbol {
        "H" => Some(1),
        "HE" => Some(2),
        "LI" => Some(3),
        "BE" => Some(4),
        "B" => Some(5),
        "C" => Some(6),
        "N" => Some(7),
        "O" => Some(8),
        "F" => Some(9),
        "NE" => Some(10),
        "NA" => Some(11),
        "MG" => Some(12),
        "AL" => Some(13),
        "SI" => Some(14),
        "P" => Some(15),
        "S" => Some(16),
        "CL" => Some(17),
        "AR" => Some(18),
        "K" => Some(19),
        "CA" => Some(20),
        "SC" => Some(21),
        "TI" => Some(22),
        "V" => Some(23),
        "CR" => Some(24),
        "MN" => Some(25),
        "FE" => Some(26),
        "CO" => Some(27),
        "NI" => Some(28),
        "CU" => Some(29),
        "ZN" => Some(30),
        "GA" => Some(31),
        "GE" => Some(32),
        "AS" => Some(33),
        "SE" => Some(34),
        "BR" => Some(35),
        "KR" => Some(36),
        "RB" => Some(37),
        "SR" => Some(38),
        "Y" => Some(39),
        "ZR" => Some(40),
        "NB" => Some(41),
        "MO" => Some(42),
        "TC" => Some(43),
        "RU" => Some(44),
        "RH" => Some(45),
        "PD" => Some(46),
        "AG" => Some(47),
        "CD" => Some(48),
        "IN" => Some(49),
        "SN" => Some(50),
        "SB" => Some(51),
        "TE" => Some(52),
        "I" => Some(53),
        "XE" => Some(54),
        "CS" => Some(55),
        "BA" => Some(56),
        "LA" => Some(57),
        "CE" => Some(58),
        "PR" => Some(59),
        "ND" => Some(60),
        "PM" => Some(61),
        "SM" => Some(62),
        "EU" => Some(63),
        "GD" => Some(64),
        "TB" => Some(65),
        "DY" => Some(66),
        "HO" => Some(67),
        "ER" => Some(68),
        "TM" => Some(69),
        "YB" => Some(70),
        "LU" => Some(71),
        "HF" => Some(72),
        "TA" => Some(73),
        "W" => Some(74),
        "RE" => Some(75),
        "OS" => Some(76),
        "IR" => Some(77),
        "PT" => Some(78),
        "AU" => Some(79),
        "HG" => Some(80),
        "TL" => Some(81),
        "PB" => Some(82),
        "BI" => Some(83),
        "PO" => Some(84),
        "AT" => Some(85),
        "RN" => Some(86),
        "FR" => Some(87),
        "RA" => Some(88),
        "AC" => Some(89),
        "TH" => Some(90),
        "PA" => Some(91),
        "U" => Some(92),
        "NP" => Some(93),
        "PU" => Some(94),
        "AM" => Some(95),
        "CM" => Some(96),
        "BK" => Some(97),
        "CF" => Some(98),
        "ES" => Some(99),
        "FM" => Some(100),
        _ => None,
    }
}

/// Map full element name (uppercase) to atomic number Z.
///
/// Handles SAMMY inp files that use full element names on Card 2
/// (e.g. "ZIRCONIUM" instead of "ZR").
fn element_name_to_z(name: &str) -> Option<u32> {
    match name {
        "HYDROGEN" => Some(1),
        "HELIUM" => Some(2),
        "LITHIUM" => Some(3),
        "BERYLLIUM" => Some(4),
        "BORON" => Some(5),
        "CARBON" => Some(6),
        "NITROGEN" => Some(7),
        "OXYGEN" => Some(8),
        "FLUORINE" => Some(9),
        "NEON" => Some(10),
        "SODIUM" => Some(11),
        "MAGNESIUM" => Some(12),
        "ALUMINUM" | "ALUMINIUM" => Some(13),
        "SILICON" => Some(14),
        "PHOSPHORUS" => Some(15),
        "SULFUR" | "SULPHUR" => Some(16),
        "CHLORINE" => Some(17),
        "ARGON" => Some(18),
        "POTASSIUM" => Some(19),
        "CALCIUM" => Some(20),
        "SCANDIUM" => Some(21),
        "TITANIUM" => Some(22),
        "VANADIUM" => Some(23),
        "CHROMIUM" => Some(24),
        "MANGANESE" => Some(25),
        "IRON" => Some(26),
        "COBALT" => Some(27),
        "NICKEL" => Some(28),
        "COPPER" => Some(29),
        "ZINC" => Some(30),
        "GALLIUM" => Some(31),
        "GERMANIUM" => Some(32),
        "ARSENIC" => Some(33),
        "SELENIUM" => Some(34),
        "BROMINE" => Some(35),
        "KRYPTON" => Some(36),
        "RUBIDIUM" => Some(37),
        "STRONTIUM" => Some(38),
        "YTTRIUM" => Some(39),
        "ZIRCONIUM" => Some(40),
        "NIOBIUM" => Some(41),
        "MOLYBDENUM" => Some(42),
        "TECHNETIUM" => Some(43),
        "RUTHENIUM" => Some(44),
        "RHODIUM" => Some(45),
        "PALLADIUM" => Some(46),
        "SILVER" => Some(47),
        "CADMIUM" => Some(48),
        "INDIUM" => Some(49),
        "TIN" => Some(50),
        "ANTIMONY" => Some(51),
        "TELLURIUM" => Some(52),
        "IODINE" => Some(53),
        "XENON" => Some(54),
        "CESIUM" | "CAESIUM" => Some(55),
        "BARIUM" => Some(56),
        "LANTHANUM" => Some(57),
        "CERIUM" => Some(58),
        "PRASEODYMIUM" => Some(59),
        "NEODYMIUM" => Some(60),
        "PROMETHIUM" => Some(61),
        "SAMARIUM" => Some(62),
        "EUROPIUM" => Some(63),
        "GADOLINIUM" => Some(64),
        "TERBIUM" => Some(65),
        "DYSPROSIUM" => Some(66),
        "HOLMIUM" => Some(67),
        "ERBIUM" => Some(68),
        "THULIUM" => Some(69),
        "YTTERBIUM" => Some(70),
        "LUTETIUM" => Some(71),
        "HAFNIUM" => Some(72),
        "TANTALUM" => Some(73),
        "TUNGSTEN" => Some(74),
        "RHENIUM" => Some(75),
        "OSMIUM" => Some(76),
        "IRIDIUM" => Some(77),
        "PLATINUM" => Some(78),
        "GOLD" => Some(79),
        "MERCURY" => Some(80),
        "THALLIUM" => Some(81),
        "LEAD" => Some(82),
        "BISMUTH" => Some(83),
        "THORIUM" => Some(90),
        "PROTACTINIUM" => Some(91),
        "URANIUM" => Some(92),
        "NEPTUNIUM" => Some(93),
        "PLUTONIUM" => Some(94),
        "AMERICIUM" => Some(95),
        "CURIUM" => Some(96),
        "BERKELIUM" => Some(97),
        "CALIFORNIUM" => Some(98),
        _ => None,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plt_header_and_data() {
        let content = "\
      Energy            Data      Uncertainty     Th_initial      Th_final
   1.1330168       8.9078373      0.32005507       8.4964191       8.5014004
   1.1336480       8.5677882      0.31321707       8.5003345       8.5048604
";
        let records = parse_sammy_plt(content).unwrap();
        assert_eq!(records.len(), 2);
        assert!((records[0].energy_kev - 1.1330168).abs() < 1e-7);
        assert!((records[0].theory_initial - 8.4964191).abs() < 1e-5);
        assert!((records[1].energy_kev - 1.1336480).abs() < 1e-7);
    }

    #[test]
    fn test_parse_par_tr007() {
        let content = "\
-2070.     1450.      186600.    0.         0.          0 0 0 0 0 1
27650.     1400.      1480000.   0.         0.          1 0 1 0 0 1
1151.07    600.       62.        0.         0.          1 1 1 0 0 2


EXPLICIT
";
        let par = parse_sammy_par(content).unwrap();
        assert_eq!(par.resonances.len(), 3);

        // First resonance: bound state at -2070 eV.
        let r0 = &par.resonances[0];
        assert!((r0.energy_ev - (-2070.0)).abs() < 1e-6);
        assert!((r0.gamma_gamma_ev - 1.45).abs() < 1e-6); // 1450 meV → 1.45 eV
        assert!((r0.gamma_n_ev - 186.6).abs() < 1e-6); // 186600 meV → 186.6 eV
        assert_eq!(r0.spin_group, 1);

        // Third resonance: 1151.07 eV (in-range for tr007).
        let r2 = &par.resonances[2];
        assert!((r2.energy_ev - 1151.07).abs() < 1e-6);
        assert!((r2.gamma_gamma_ev - 0.6).abs() < 1e-6); // 600 meV → 0.6 eV
        assert!((r2.gamma_n_ev - 0.062).abs() < 1e-6); // 62 meV → 0.062 eV
        assert_eq!(r2.spin_group, 2);
    }

    #[test]
    fn test_parse_inp_tr007() {
        let content = "\
 FE TRANSMISSION FROM 1.13 TO 1.18 KEV
      FE56 55.9      1133.01   1170.517    99
PRINT WEIGHTED RESIDUALS
DATA COVARIANCE FILE IS NAMED t007a.dcv
PRINT THEORETICAL VALUES
PRINT ALL INPUT PARAMETERS
PRINT INPUT DATA
WRITE NEW INPUT FILE = temp1_thick1.inp
#DO NOT SUPPRESS ANY INTERMEDIATE RESULTS
GENERATE PLOT FILE AUTOMATICALLY

329.      80.263     0.0301   0.0       .021994
6.0       0.2179
TRANSMISSION
  1      1    0  0.5    0.9999   .0
    1    1    0    0      .500      .000      .000
  2      1    0 -0.5    0.9172   .0
    1    1    0    1      .500      .000      .000

BROADENING
6.00      329.00    0.2179    0.025     0.022     0.022      1 1 1 1 1 1
";
        let inp = parse_sammy_inp(content).unwrap();
        assert_eq!(inp.isotope_symbol, "FE56");
        assert!((inp.awr - 55.9).abs() < 1e-6);
        assert!((inp.energy_min_ev - 1133.01).abs() < 1e-6);
        assert!((inp.energy_max_ev - 1170.517).abs() < 1e-6);
        assert!((inp.temperature_k - 329.0).abs() < 1e-6);
        assert!((inp.flight_path_m - 80.263).abs() < 1e-6);
        // Card 5 resolution params.
        assert!((inp.delta_l_sammy - 0.0301).abs() < 1e-6);
        assert!((inp.delta_e_sammy - 0.0).abs() < 1e-6);
        assert!((inp.delta_g_sammy - 0.021994).abs() < 1e-6);
        // BROADENING card overrides.
        assert_eq!(inp.broadening_delta_l, Some(0.025));
        assert_eq!(inp.broadening_delta_g, Some(0.022));
        assert_eq!(inp.broadening_delta_e, Some(0.022));
        // Effective values use BROADENING card when present.
        assert!((inp.effective_delta_l() - 0.025).abs() < 1e-6);
        assert!((inp.effective_delta_g() - 0.022).abs() < 1e-6);
        assert!((inp.effective_delta_e() - 0.022).abs() < 1e-6);
        assert!((inp.scattering_radius_fm - 6.0).abs() < 1e-6);
        assert!((inp.thickness_atoms_barn - 0.2179).abs() < 1e-6);
        // Target spin parsed from spin group header field 6 (".0" → 0.0).
        assert!((inp.target_spin - 0.0).abs() < 1e-6);

        // Spin groups.
        assert_eq!(inp.spin_groups.len(), 2);
        assert_eq!(inp.spin_groups[0].index, 1);
        assert!((inp.spin_groups[0].j - 0.5).abs() < 1e-6);
        assert_eq!(inp.spin_groups[0].l, 0);
        assert_eq!(inp.spin_groups[1].index, 2);
        assert!((inp.spin_groups[1].j - (-0.5)).abs() < 1e-6); // SAMMY sign preserved
        // SG2 has J=-0.5 which couples as L=1, s=1/2, J=|L-S|=1/2.
        // SAMMY Card 10.1 stores L at column 18-19, which is field[3] in
        // whitespace split (field[2] is ishift=0, not L).
        assert_eq!(inp.spin_groups[1].l, 1);
    }

    #[test]
    fn test_parse_isotope_symbol() {
        assert_eq!(parse_isotope_symbol("60NI").unwrap(), (28, 60));
        assert_eq!(parse_isotope_symbol("FE56").unwrap(), (26, 56));
        assert_eq!(parse_isotope_symbol("58NI").unwrap(), (28, 58));
        assert!(parse_isotope_symbol("UNKNOWN99").is_err());
        // Multi-isotope labels.
        assert_eq!(parse_isotope_symbol("Cu65").unwrap(), (29, 65));
        assert_eq!(parse_isotope_symbol("Cu63").unwrap(), (29, 63));
        assert_eq!(parse_isotope_symbol("CU 65").unwrap(), (29, 65));
        // Natural-element prefix.
        assert_eq!(parse_isotope_symbol("NatFE").unwrap(), (26, 0));
        assert_eq!(parse_isotope_symbol("NatFe").unwrap(), (26, 0));
    }

    #[test]
    fn test_sammy_to_resonance_data_tr007() {
        let inp = SammyInpConfig {
            title: "test".to_string(),
            isotope_symbol: "FE56".to_string(),
            awr: 55.9,
            energy_min_ev: 1133.01,
            energy_max_ev: 1170.517,
            temperature_k: 329.0,
            flight_path_m: 80.263,
            delta_l_sammy: 0.0301,
            delta_e_sammy: 0.0,
            delta_g_sammy: 0.021994,
            no_broadening: false,
            broadening_delta_l: None,
            broadening_delta_g: None,
            broadening_delta_e: None,
            scattering_radius_fm: 6.0,
            thickness_atoms_barn: 0.2179,
            target_spin: 0.0,
            observation_type: SammyObservationType::default(),
            spin_groups: vec![
                SammySpinGroup {
                    index: 1,
                    j: 0.5,
                    l: 0,
                    abundance: 0.9999,
                    target_spin: 0.0,
                    isotope_label: None,
                },
                SammySpinGroup {
                    index: 2,
                    j: -0.5, // Negative J per SAMMY convention
                    l: 1,    // p-wave: J=|L-S|=|1-0.5|=0.5
                    abundance: 0.9172,
                    target_spin: 0.0,
                    isotope_label: None,
                },
            ],
        };
        let par = SammyParFile {
            resonances: vec![
                SammyResonance {
                    energy_ev: -2070.0,
                    gamma_gamma_ev: 1.45,
                    gamma_n_ev: 186.6,
                    gamma_f1_ev: 0.0,
                    gamma_f2_ev: 0.0,
                    spin_group: 1,
                },
                SammyResonance {
                    energy_ev: 1151.07,
                    gamma_gamma_ev: 0.6,
                    gamma_n_ev: 0.062,
                    gamma_f1_ev: 0.0,
                    gamma_f2_ev: 0.0,
                    spin_group: 2,
                },
            ],
            radius_overrides: std::collections::HashMap::new(),
            r_external: std::collections::HashMap::new(),
            isotopic_masses: Vec::new(),
        };
        let rd = sammy_to_resonance_data(&inp, &par).unwrap();
        assert_eq!(rd.za, 26056); // Fe-56
        assert_eq!(rd.ranges.len(), 1);
        assert_eq!(rd.ranges[0].formalism, ResonanceFormalism::ReichMoore);
        assert!((rd.ranges[0].scattering_radius - 6.0).abs() < 1e-6);
        // SG1 (L=0) has 1 resonance, SG2 (L=1) has 1 resonance → two L-groups.
        assert_eq!(rd.ranges[0].l_groups.len(), 2);
        // L-groups are ordered by L (BTreeMap).
        assert_eq!(rd.ranges[0].l_groups[0].l, 0);
        assert_eq!(rd.ranges[0].l_groups[0].resonances.len(), 1);
        assert_eq!(rd.ranges[0].l_groups[1].l, 1);
        assert_eq!(rd.ranges[0].l_groups[1].resonances.len(), 1);
        // Resonances preserve signed J from their spin groups.
        let res_l0 = &rd.ranges[0].l_groups[0].resonances[0];
        let res_l1 = &rd.ranges[0].l_groups[1].resonances[0];
        assert!((res_l0.j - 0.5).abs() < 1e-6, "spin group 1 → J=+0.5");
        assert!((res_l1.j - (-0.5)).abs() < 1e-6, "spin group 2 → J=-0.5");
    }

    #[test]
    fn test_sammy_to_nereids_resolution_conversion() {
        // Use tr007's effective BROADENING card values: Deltag=0.022, Deltal=0.025.
        let inp = SammyInpConfig {
            title: String::new(),
            isotope_symbol: "FE56".to_string(),
            awr: 55.9,
            energy_min_ev: 1133.0,
            energy_max_ev: 1170.0,
            temperature_k: 329.0,
            flight_path_m: 80.263,
            delta_l_sammy: 0.0301,
            delta_e_sammy: 0.0,
            delta_g_sammy: 0.021994,
            no_broadening: false,
            broadening_delta_l: Some(0.025),
            broadening_delta_g: Some(0.022),
            broadening_delta_e: Some(0.022),
            scattering_radius_fm: 6.0,
            thickness_atoms_barn: 0.2179,
            target_spin: 0.0,
            observation_type: SammyObservationType::default(),
            spin_groups: vec![],
        };

        let (flight_path, delta_t, delta_l, delta_e) =
            sammy_to_nereids_resolution(&inp).expect("should return Some for non-zero params");

        assert!((flight_path - 80.263).abs() < 1e-10);
        // delta_t = Deltag / (2·√ln2) = 0.022 / (2·0.83255...) = 0.01321...
        let expected_dt = 0.022 / (2.0 * 2.0_f64.ln().sqrt());
        assert!(
            (delta_t - expected_dt).abs() < 1e-12,
            "delta_t={delta_t}, expected={expected_dt}"
        );
        // delta_l = Deltal / √6 = 0.025 / 2.44949... = 0.01021...
        let expected_dl = 0.025 / 6.0_f64.sqrt();
        assert!(
            (delta_l - expected_dl).abs() < 1e-12,
            "delta_l={delta_l}, expected={expected_dl}"
        );
        // delta_e = raw Deltae from BROADENING card (no conversion)
        assert!(
            (delta_e - 0.022).abs() < 1e-12,
            "delta_e={delta_e}, expected=0.022"
        );
    }

    #[test]
    fn test_sammy_to_nereids_resolution_zero_params() {
        let inp = SammyInpConfig {
            title: String::new(),
            isotope_symbol: "FE56".to_string(),
            awr: 55.9,
            energy_min_ev: 1133.0,
            energy_max_ev: 1170.0,
            temperature_k: 329.0,
            flight_path_m: 80.263,
            delta_l_sammy: 0.0,
            delta_e_sammy: 0.0,
            delta_g_sammy: 0.0,
            no_broadening: false,
            broadening_delta_l: None,
            broadening_delta_g: None,
            broadening_delta_e: None,
            scattering_radius_fm: 6.0,
            thickness_atoms_barn: 0.2179,
            target_spin: 0.0,
            observation_type: SammyObservationType::default(),
            spin_groups: vec![],
        };

        assert!(
            sammy_to_nereids_resolution(&inp).is_none(),
            "should return None when both Deltal and Deltag are zero"
        );
    }

    #[test]
    fn test_parse_fortran_d_exponent() {
        assert!((parse_fortran_float("1.23D-5").unwrap() - 1.23e-5).abs() < 1e-20);
        assert!((parse_fortran_float("1.23d-5").unwrap() - 1.23e-5).abs() < 1e-20);
        assert!((parse_fortran_float("5.4D+3").unwrap() - 5.4e3).abs() < 1e-6);
        assert!((parse_fortran_float("1.0D0").unwrap() - 1.0).abs() < 1e-15);
    }
}
