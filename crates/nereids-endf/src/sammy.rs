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

use crate::resonance::{LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange};

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

/// Parsed SAMMY `.par` file.
#[derive(Debug, Clone)]
pub struct SammyParFile {
    pub resonances: Vec<SammyResonance>,
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
/// **Note**: This function only converts the Gaussian resolution parameters
/// (Deltal → Bo2, Deltag → Ao2).  The exponential tail parameter Deltae
/// (`effective_delta_e()`) is not used — exponential resolution broadening
/// is not yet implemented in NEREIDS.
///
/// Returns `None` if both effective Deltal and Deltag are zero (no resolution broadening).
/// Otherwise returns `Some((flight_path_m, delta_t_us, delta_l_m))`.
#[must_use]
pub fn sammy_to_nereids_resolution(inp: &SammyInpConfig) -> Option<(f64, f64, f64)> {
    if inp.no_broadening {
        return None;
    }

    let delta_l = inp.effective_delta_l();
    let delta_g = inp.effective_delta_g();

    if delta_l == 0.0 && delta_g == 0.0 {
        return None;
    }

    // Convert from SAMMY convention to NEREIDS convention.
    let delta_t_us = delta_g / (2.0 * 2.0_f64.ln().sqrt());
    let delta_l_m = delta_l / 6.0_f64.sqrt();

    Some((inp.flight_path_m, delta_t_us, delta_l_m))
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
        if fields.len() < 5 {
            return Err(SammyParseError::new(format!(
                "line {}: expected 5 columns, got {}",
                i + 1,
                fields.len()
            )));
        }
        let parse = |s: &str, col: &str| {
            s.parse::<f64>().map_err(|e| {
                SammyParseError::new(format!("line {}: cannot parse {col}: {e}", i + 1))
            })
        };
        records.push(SammyPltRecord {
            energy_kev: parse(fields[0], "energy")?,
            data: parse(fields[1], "data")?,
            uncertainty: parse(fields[2], "uncertainty")?,
            theory_initial: parse(fields[3], "theory_initial")?,
            theory_final: parse(fields[4], "theory_final")?,
        });
    }
    if records.is_empty() {
        return Err(SammyParseError::new("no data records found in .plt file"));
    }
    Ok(records)
}

// ─── .par parser ───────────────────────────────────────────────────────────────

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
            s.parse::<f64>().map_err(|e| {
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
    Ok(SammyParFile { resonances })
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

    // Line 2: isotope definition.
    // Format: "      60NI    59.927  505000.0  508000.0        -1"
    let iso_line = lines[1];
    let iso_fields: Vec<&str> = iso_line.split_whitespace().collect();
    if iso_fields.len() < 4 {
        return Err(SammyParseError::new(format!(
            "line 2: expected at least 4 fields (symbol, AWR, Emin, Emax), got {}",
            iso_fields.len()
        )));
    }
    let isotope_symbol = iso_fields[0].to_string();
    let awr = iso_fields[1]
        .parse::<f64>()
        .map_err(|e| SammyParseError::new(format!("line 2: AWR: {e}")))?;
    let energy_min_ev = iso_fields[2]
        .parse::<f64>()
        .map_err(|e| SammyParseError::new(format!("line 2: Emin: {e}")))?;
    let energy_max_ev = iso_fields[3]
        .parse::<f64>()
        .map_err(|e| SammyParseError::new(format!("line 2: Emax: {e}")))?;

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
        if upper.contains("BROADENING IS NOT WANTED") || upper.contains("NO LOW-ENERGY BROADENING")
        {
            no_broadening = true;
        }
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
    if idx < lines.len() {
        let card5_fields: Vec<&str> = lines[idx].split_whitespace().collect();
        if card5_fields.len() >= 2 {
            temperature_k = card5_fields[0]
                .parse::<f64>()
                .map_err(|e| SammyParseError::new(format!("Card 5: temperature: {e}")))?;
            flight_path_m = card5_fields[1]
                .parse::<f64>()
                .map_err(|e| SammyParseError::new(format!("Card 5: flight_path: {e}")))?;
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

    // Scan for TRANSMISSION keyword and spin group block, then BROADENING card.
    let mut broadening_delta_l: Option<f64> = None;
    let mut broadening_delta_g: Option<f64> = None;
    let mut broadening_delta_e: Option<f64> = None;

    while idx < lines.len() {
        let trimmed = lines[idx].trim().to_uppercase();
        if trimmed.starts_with("TRANSMISSION") || trimmed.starts_with("TOTAL") {
            idx += 1;
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
/// Header line uses fixed-width Fortran FORMAT(I5, I5, I5, F5.1, F11.4, F5.1):
///   [0:5]   KKK    — spin group index
///   [5:10]  NENT   — number of entrance channels
///   [10:15] NEXT   — number of exit-only channels
///   [15:20] SPINJ  — J value for this group (F5.1)
///   [20:31] ABNDNC — abundance/weight (F11.4)
///   [31:36] SPINI  — target spin (F5.1)
///
/// Channel lines use split_whitespace (indented ≥4 spaces).
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

        // Distinguish header vs channel line by leading whitespace.
        // Header lines have ≤3 leading spaces; channel lines have ≥4.
        let leading_spaces = lines[i].len() - lines[i].trim_start().len();
        let is_header = leading_spaces <= 3 && lines[i].len() >= 20;

        if is_header {
            let line = lines[i];
            // Fixed-width column extraction.
            let index: u32 = col(line, 0, 5)
                .parse()
                .map_err(|e| SammyParseError::new(format!("spin group index: {e}")))?;
            let j: f64 = col(line, 15, 20)
                .parse()
                .map_err(|e| SammyParseError::new(format!("spin group J: {e}")))?;
            // Preserve the sign: SAMMY uses negative J to distinguish spin
            // groups with the same |J|.  The sign keeps them in separate
            // J-groups during cross-section evaluation.
            let abundance: f64 = {
                let s = col(line, 20, 31);
                if s.is_empty() {
                    1.0
                } else {
                    s.parse().map_err(|e| {
                        SammyParseError::new(format!("spin group abundance ({s:?}): {e}"))
                    })?
                }
            };

            // Target spin: read from the first header only.
            if groups.is_empty() {
                let s = col(line, 31, 36);
                if !s.is_empty() {
                    target_spin = s.parse::<f64>().map_err(|e| {
                        SammyParseError::new(format!("spin group target spin ({s:?}): {e}"))
                    })?;
                }
            }

            // The next line is the channel definition — extract L.
            //
            // SAMMY Card Set 10.1 (old format) channel card columns
            // (from ResonanceParameterIO.cpp:readOldChannelData):
            //   cols  3-4: channel ID
            //   cols  5-7: kz1 (charge, usually blank)
            //   cols  8-9: lpent
            //   cols 10-12: kz2 (charge, usually blank)
            //   cols 13-14: ishift
            //   cols 15-17: ifexcl (usually blank)
            //   cols 18-19: L (orbital angular momentum)  ← this is what we need
            //   cols 20-29: channel spin
            //
            // Blank integer fields (kz1, kz2, ifexcl) collapse in whitespace
            // splitting, so the token layout is:
            //   [0]=chan_id  [1]=lpent  [2]=ishift  [3]=L  [4]=channel_spin
            let mut l = 0u32;
            if i + 1 < lines.len() {
                let ch_fields: Vec<&str> = lines[i + 1].split_whitespace().collect();
                if ch_fields.len() >= 4 {
                    l = ch_fields[3].parse().unwrap_or(0);
                }
            }

            groups.push(SammySpinGroup {
                index,
                j,
                l,
                abundance,
            });
        }
        // Skip channel lines (they follow their header).
        i += 1;
    }

    Ok((groups, target_spin))
}

// ─── Converter: SAMMY → NEREIDS ResonanceData ──────────────────────────────────

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

    for res in &par.resonances {
        let sg = group_map.get(&res.spin_group).ok_or_else(|| {
            SammyParseError::new(format!(
                "resonance at E={} eV references undefined spin group {}",
                res.energy_ev, res.spin_group
            ))
        })?;

        l_group_map.entry(sg.l).or_default().push(Resonance {
            energy: res.energy_ev,
            j: sg.j,
            gn: res.gamma_n_ev,
            gg: res.gamma_gamma_ev,
            gfa: res.gamma_f1_ev,
            gfb: res.gamma_f2_ev,
        });
    }

    // Build LGroups.
    let l_groups: Vec<LGroup> = l_group_map
        .into_iter()
        .map(|(l, resonances)| LGroup {
            l,
            awr: inp.awr,
            apl: 0.0, // Use global scattering radius.
            qx: 0.0,
            lrx: 0,
            resonances,
        })
        .collect();

    // Use the target spin parsed from the spin group header (field 6).
    // For even-even nuclei (Fe-56, Ni-58, Ni-60) this is 0.0.
    let target_spin = inp.target_spin;

    // Infer Z and A from the isotope symbol (e.g. "60NI" → Z=28, A=60;
    // "FE56" → Z=26, A=56; "58NI" → Z=28, A=58).
    let (z, a) = parse_isotope_symbol(&inp.isotope_symbol)?;
    let za = z * 1000 + a;

    // Use a wide energy range so that cross-sections can be evaluated at any
    // energy in the .plt reference file, which may extend beyond the .inp
    // analysis window.  SAMMY's .inp energy limits define the fitting region,
    // not the physics validity bounds of the resonance parameters.
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
    };

    Ok(ResonanceData {
        isotope: Isotope::new(z, a)
            .map_err(|e| SammyParseError::new(format!("invalid isotope: {e}")))?,
        za,
        awr: inp.awr,
        ranges: vec![range],
    })
}

/// Parse a SAMMY isotope symbol like "60NI", "FE56", "58NI" into (Z, A).
fn parse_isotope_symbol(symbol: &str) -> Result<(u32, u32), SammyParseError> {
    let s = symbol.trim().to_uppercase();

    // Try format: "60NI" (mass number first, then element symbol).
    if let Some(split_pos) = s.find(|c: char| c.is_ascii_alphabetic()) {
        let mass_str = &s[..split_pos];
        let elem_str = &s[split_pos..];
        if !mass_str.is_empty()
            && let Ok(a) = mass_str.parse::<u32>()
            && let Some(z) = element_symbol_to_z(elem_str)
        {
            return Ok((z, a));
        }
    }

    // Try format: "FE56" (element symbol first, then mass number).
    if let Some(split_pos) = s.find(|c: char| c.is_ascii_digit()) {
        let elem_str = &s[..split_pos];
        let mass_str = &s[split_pos..];
        if let Ok(a) = mass_str.parse::<u32>()
            && let Some(z) = element_symbol_to_z(elem_str)
        {
            return Ok((z, a));
        }
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
            spin_groups: vec![
                SammySpinGroup {
                    index: 1,
                    j: 0.5,
                    l: 0,
                    abundance: 0.9999,
                },
                SammySpinGroup {
                    index: 2,
                    j: -0.5, // Negative J per SAMMY convention
                    l: 1,    // p-wave: J=|L-S|=|1-0.5|=0.5
                    abundance: 0.9172,
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
            spin_groups: vec![],
        };

        let (flight_path, delta_t, delta_l) =
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
            spin_groups: vec![],
        };

        assert!(
            sammy_to_nereids_resolution(&inp).is_none(),
            "should return None when both Deltal and Deltag are zero"
        );
    }
}
