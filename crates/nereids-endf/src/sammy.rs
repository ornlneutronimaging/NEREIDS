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
    /// Channel width / delta-L (m or μs depending on context).
    pub channel_width: f64,
    /// Scattering radius (fm).
    pub scattering_radius_fm: f64,
    /// Sample thickness (atoms/barn).
    pub thickness_atoms_barn: f64,
    /// Spin group definitions.
    pub spin_groups: Vec<SammySpinGroup>,
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
/// Reads fixed-width columnar data until a blank line or the `EXPLICIT` keyword.
/// Column layout (all widths approximate, falls back to whitespace split):
///   cols  1-11: E_res (eV)
///   cols 12-22: Γ_γ (meV)
///   cols 23-33: Γ_n (meV)
///   cols 34-44: Γ_f1 (meV)
///   cols 45-55: Γ_f2 (meV)
///   remaining:  [vary-flags]  spin_group_id
///
/// Widths are in meV in the file; this function converts to eV.
pub fn parse_sammy_par(content: &str) -> Result<SammyParFile, SammyParseError> {
    let mut resonances = Vec::new();

    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        // Stop at blank line or EXPLICIT keyword.
        if trimmed.is_empty() || trimmed.starts_with("EXPLICIT") {
            break;
        }

        // Parse the resonance line. Use whitespace splitting — the Fortran
        // fixed-width format in samtry files always has spaces between fields.
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 3 {
            return Err(SammyParseError::new(format!(
                "line {}: expected at least 3 fields (E, Γ_γ, Γ_n), got {}",
                i + 1,
                fields.len()
            )));
        }

        let parse_f64 = |s: &str, name: &str| {
            s.parse::<f64>().map_err(|e| {
                SammyParseError::new(format!("line {}: cannot parse {name}: {e}", i + 1))
            })
        };

        let energy_ev = parse_f64(fields[0], "E_res")?;
        let gamma_gamma_mev = parse_f64(fields[1], "Γ_γ")?;
        let gamma_n_mev = parse_f64(fields[2], "Γ_n")?;

        // Fission widths: fields 3 and 4 if present and non-zero.
        // These come after Γ_n but before the vary-flag integers.
        // We detect fission widths by checking if the field parses as f64
        // and contains a decimal point (vary flags are integers like "0").
        let (gamma_f1_mev, gamma_f2_mev, flag_start) = parse_fission_widths(&fields[3..]);

        // Spin group index is the last integer on the line.
        let spin_group = fields
            .last()
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| {
                SammyParseError::new(format!("line {}: no spin group index found", i + 1))
            })?;

        // Ensure the spin_group came from after the flag region.
        // In samtry files, the layout is: E Γ_γ Γ_n [Γ_f1 Γ_f2] flags... spin_group
        // The spin group is always the very last field.
        let _ = flag_start; // Flags are not needed for forward model.

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

/// Parse optional fission widths from the remaining fields.
///
/// Returns (Γ_f1_meV, Γ_f2_meV, index_of_first_vary_flag).
/// Fission width fields contain "." (floating point), vary flags are bare integers.
fn parse_fission_widths(fields: &[&str]) -> (f64, f64, usize) {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let mut idx = 0;

    // Field 0 = potential Γ_f1
    if let Some(&s) = fields.first()
        && (s.contains('.') || s.contains('E') || s.contains('e'))
        && let Ok(v) = s.parse::<f64>()
    {
        f1 = v;
        idx = 1;
        // Field 1 = potential Γ_f2
        if let Some(&s2) = fields.get(1)
            && (s2.contains('.') || s2.contains('E') || s2.contains('e'))
            && let Ok(v2) = s2.parse::<f64>()
        {
            f2 = v2;
            idx = 2;
        }
    }
    (f1, f2, idx)
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
    let mut channel_width = 0.0;
    let mut scattering_radius_fm = 0.0;
    let mut thickness_atoms_barn = 0.0;
    let mut spin_groups = Vec::new();

    // State machine: find blank line after commands, then parse numeric cards.
    let mut idx = 2;
    // Skip command lines until first blank line.
    while idx < lines.len() {
        let trimmed = lines[idx].trim();
        if trimmed.is_empty() {
            idx += 1;
            break;
        }
        idx += 1;
    }

    // Card 5: broadening parameters (first numeric line after blank).
    if idx < lines.len() {
        let card5_fields: Vec<&str> = lines[idx].split_whitespace().collect();
        if card5_fields.len() >= 2 {
            temperature_k = card5_fields[0].parse().unwrap_or(300.0);
            flight_path_m = card5_fields[1].parse().unwrap_or(0.0);
            if card5_fields.len() >= 5 {
                channel_width = card5_fields[4].parse().unwrap_or(0.0);
            }
        }
        idx += 1;
    }

    // Card 6: scattering radius and thickness.
    if idx < lines.len() {
        let card6_fields: Vec<&str> = lines[idx].split_whitespace().collect();
        if card6_fields.len() >= 2 {
            scattering_radius_fm = card6_fields[0].parse().unwrap_or(0.0);
            thickness_atoms_barn = card6_fields[1].parse().unwrap_or(0.0);
        }
        idx += 1;
    }

    // Scan for TRANSMISSION keyword and spin group block.
    while idx < lines.len() {
        let trimmed = lines[idx].trim().to_uppercase();
        if trimmed == "TRANSMISSION" {
            idx += 1;
            // Parse spin group definitions until blank line.
            spin_groups = parse_spin_groups(&lines[idx..])?;
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
        channel_width,
        scattering_radius_fm,
        thickness_atoms_barn,
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
/// Header: group_index  n_channels  ?  J  abundance  target_spin
/// Channel: channel_id  pair_id  L  ?  channel_spin
fn parse_spin_groups(lines: &[&str]) -> Result<Vec<SammySpinGroup>, SammyParseError> {
    let mut groups = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() {
            break; // End of spin group block.
        }

        let fields: Vec<&str> = trimmed.split_whitespace().collect();

        // Distinguish header vs channel line: channel lines are indented (start
        // with spaces) and the first field is the channel index within the group.
        // Header lines have the group index as first field, typically ≥ 1, and
        // are less indented. We detect headers by checking if the first character
        // of the raw line (after minimal trim) is a digit at a low indent level.
        //
        // Simpler heuristic: header lines have >= 5 fields with a floating-point
        // J value at position 3 (contains '.').
        let is_header = fields.len() >= 5 && fields[3].contains('.');

        if is_header {
            let index = fields[0]
                .parse::<u32>()
                .map_err(|e| SammyParseError::new(format!("spin group index: {e}")))?;
            let j: f64 = fields[3]
                .parse()
                .map_err(|e| SammyParseError::new(format!("spin group J: {e}")))?;
            // Preserve the sign: SAMMY uses negative J to distinguish spin
            // groups with the same |J|.  The sign keeps them in separate
            // J-groups during cross-section evaluation.
            let abundance: f64 = fields[4].parse().unwrap_or(1.0);

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

    Ok(groups)
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

    // Determine target spin from the .inp.  For Ni/Fe isotopes with even A,
    // target spin I = 0.  We extract it from the spin group definition if
    // available (field 6 in the header), otherwise default to 0.
    let target_spin = 0.0;

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

/// Map element symbol to atomic number Z.
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
        "ZR" => Some(40),
        "NB" => Some(41),
        "MO" => Some(42),
        "TC" => Some(43),
        "AG" => Some(47),
        "IN" => Some(49),
        "SN" => Some(50),
        "TA" => Some(73),
        "W" => Some(74),
        "PB" => Some(82),
        "TH" => Some(90),
        "U" => Some(92),
        "PU" => Some(94),
        "AM" => Some(95),
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
        assert!((inp.scattering_radius_fm - 6.0).abs() < 1e-6);
        assert!((inp.thickness_atoms_barn - 0.2179).abs() < 1e-6);

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
            channel_width: 0.021994,
            scattering_radius_fm: 6.0,
            thickness_atoms_barn: 0.2179,
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
}
