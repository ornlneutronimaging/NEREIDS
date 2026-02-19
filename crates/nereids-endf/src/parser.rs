//! ENDF-6 File 2 resonance parameter parser.
//!
//! Parses the fixed-width 80-character ENDF-6 format to extract resolved
//! resonance region (RRR) parameters.
//!
//! ## ENDF-6 Line Format
//! Each line is exactly 80 characters:
//! - Cols 1-11:  Field 1 (floating point or integer)
//! - Cols 12-22: Field 2
//! - Cols 23-33: Field 3
//! - Cols 34-44: Field 4
//! - Cols 45-55: Field 5
//! - Cols 56-66: Field 6
//! - Cols 67-70: MAT number
//! - Cols 71-72: MF (file number)
//! - Cols 73-75: MT (section number)
//! - Cols 76-80: Line sequence
//!
//! ## SAMMY Reference
//! - SAMMY manual Section 9 (ENDF-6 format)
//! - SAMMY source: `sammy/src/endf/` module

use crate::resonance::*;
use nereids_core::elements::isotope_from_za;

/// Parse ENDF-6 File 2 resonance parameters from raw ENDF text.
///
/// Extracts all MF=2, MT=151 lines and parses the resolved resonance region.
///
/// # Arguments
/// * `endf_text` — Full ENDF file contents as a string.
///
/// # Returns
/// `ResonanceData` containing all parsed resonance parameters.
pub fn parse_endf_file2(endf_text: &str) -> Result<ResonanceData, EndfParseError> {
    // Extract MF=2, MT=151 lines (resonance parameters).
    let lines: Vec<&str> = endf_text
        .lines()
        .filter(|line| {
            if line.len() < 75 {
                return false;
            }
            let mf = line[70..72].trim();
            let mt = line[72..75].trim();
            mf == "2" && mt == "151"
        })
        .collect();

    if lines.is_empty() {
        return Err(EndfParseError::MissingSection(
            "No MF=2, MT=151 data found".to_string(),
        ));
    }

    let mut pos = 0;

    // HEAD record: ZA, AWR, 0, 0, NIS, 0
    let head = parse_cont(&lines, &mut pos)?;
    let za = head.c1 as u32;
    let awr = head.c2;
    let nis = head.n1 as usize; // number of isotopes (usually 1)

    let isotope = isotope_from_za(za);
    let mut all_ranges = Vec::new();

    for _ in 0..nis {
        // Isotope CONT: ZAI, ABN, 0, LFW, NER, 0
        let iso_cont = parse_cont(&lines, &mut pos)?;
        let _zai = iso_cont.c1 as u32;
        let _abn = iso_cont.c2; // abundance
        let _lfw = iso_cont.l2; // fission width flag
        let ner = iso_cont.n1 as usize; // number of energy ranges

        for _ in 0..ner {
            // Range CONT: EL, EH, LRU, LRF, NRO, NAPS
            let range_cont = parse_cont(&lines, &mut pos)?;
            let energy_low = range_cont.c1;
            let energy_high = range_cont.c2;
            let lru = range_cont.l1; // 1=resolved, 2=unresolved
            let lrf = range_cont.l2; // resonance formalism

            if lru == 2 {
                // Unresolved resonance region — skip for now.
                // We need to skip all lines in this subsection.
                skip_unresolved_range(&lines, &mut pos, lrf)?;
                continue;
            }

            if lru != 1 {
                return Err(EndfParseError::UnsupportedFormat(format!(
                    "LRU={} not supported (expected 1=resolved)",
                    lru
                )));
            }

            let nro = range_cont.n1; // energy-dependent scattering radius flag
            let _naps = range_cont.n2; // scattering radius calculation flag

            // If NRO != 0, there's a TAB1 record for energy-dependent radius.
            if nro != 0 {
                skip_tab1(&lines, &mut pos)?;
            }

            // ENDF-6 Formats Manual: LRF values for resolved resonance region
            // LRF=1: Single-Level Breit-Wigner (SLBW)
            // LRF=2: Multi-Level Breit-Wigner (MLBW)
            // LRF=3: Reich-Moore
            // LRF=4: Adler-Adler (deprecated, not supported)
            // LRF=7: R-Matrix Limited (general)
            let formalism = match lrf {
                1 => ResonanceFormalism::SLBW,
                2 => ResonanceFormalism::MLBW,
                3 => ResonanceFormalism::ReichMoore,
                7 => ResonanceFormalism::RMatrixLimited,
                _ => {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "LRF={} not yet supported",
                        lrf
                    )));
                }
            };

            match formalism {
                ResonanceFormalism::MLBW | ResonanceFormalism::SLBW => {
                    let range =
                        parse_bw_range(&lines, &mut pos, energy_low, energy_high, formalism)?;
                    all_ranges.push(range);
                }
                ResonanceFormalism::ReichMoore => {
                    let range = parse_reich_moore_range(&lines, &mut pos, energy_low, energy_high)?;
                    all_ranges.push(range);
                }
                _ => {
                    return Err(EndfParseError::UnsupportedFormat(format!(
                        "Formalism {:?} parsing not yet implemented",
                        formalism
                    )));
                }
            }
        }
    }

    Ok(ResonanceData {
        isotope,
        za,
        awr,
        ranges: all_ranges,
    })
}

/// Parse a Breit-Wigner (SLBW or MLBW) resolved resonance range.
///
/// ENDF-6 File 2, LRF=1:
/// - CONT: SPI, AP, 0, 0, NLS, 0
/// - For each L-value:
///   - CONT: AWRI, 0.0, L, 0, 6*NRS, NRS
///   - LIST: NRS resonances, each 6 values: ER, AJ, GT, GN, GG, GF
///
/// Reference: ENDF-6 Formats Manual Section 2.2.1.1
fn parse_bw_range(
    lines: &[&str],
    pos: &mut usize,
    energy_low: f64,
    energy_high: f64,
    formalism: ResonanceFormalism,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    let cont = parse_cont(lines, pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2;
    let nls = cont.n1 as usize; // number of L-values

    let mut l_groups = Vec::with_capacity(nls);

    for _ in 0..nls {
        // CONT: AWRI, 0.0, L, 0, 6*NRS, NRS
        let l_cont = parse_cont(lines, pos)?;
        let awr_l = l_cont.c1;
        let l_val = l_cont.l1 as u32;
        let nrs = l_cont.n2 as usize; // number of resonances

        let mut resonances = Vec::with_capacity(nrs);

        // Each resonance is 6 values on one line (or spanning lines).
        // In ENDF format, LIST records pack 6 values per line.
        let total_values = nrs * 6;
        let values = parse_list_values(lines, pos, total_values)?;

        for i in 0..nrs {
            let base = i * 6;
            resonances.push(Resonance {
                energy: values[base],  // ER
                j: values[base + 1],   // AJ
                gn: values[base + 3],  // GN (neutron width)
                gg: values[base + 4],  // GG (gamma width)
                gfa: values[base + 5], // GF (fission width)
                gfb: 0.0,              // Not used in BW
                                       // Note: values[base+2] is GT (total width) — derived, not stored
            });
        }

        l_groups.push(LGroup {
            l: l_val,
            awr: awr_l,
            apl: 0.0, // Not in BW format
            resonances,
        });
    }

    Ok(ResonanceRange {
        energy_low,
        energy_high,
        resolved: true,
        formalism,
        target_spin,
        scattering_radius,
        l_groups,
    })
}

/// Parse a Reich-Moore resolved resonance range.
///
/// ENDF-6 File 2, LRF=2:
/// - CONT: SPI, AP, 0, 0, NLS, 0
/// - For each L-value:
///   - CONT: AWRI, APL, L, 0, 6*NRS, NRS
///   - LIST: NRS resonances, each 6 values: ER, AJ, GN, GG, GFA, GFB
///
/// Reference: ENDF-6 Formats Manual Section 2.2.1.2
/// Reference: SAMMY manual Section 2 (R-matrix theory)
fn parse_reich_moore_range(
    lines: &[&str],
    pos: &mut usize,
    energy_low: f64,
    energy_high: f64,
) -> Result<ResonanceRange, EndfParseError> {
    // CONT: SPI, AP, 0, 0, NLS, 0
    let cont = parse_cont(lines, pos)?;
    let target_spin = cont.c1;
    let scattering_radius = cont.c2;
    let nls = cont.n1 as usize; // number of L-values

    let mut l_groups = Vec::with_capacity(nls);

    for _ in 0..nls {
        // CONT: AWRI, APL, L, 0, 6*NRS, NRS
        let l_cont = parse_cont(lines, pos)?;
        let awr_l = l_cont.c1;
        let apl = l_cont.c2; // L-dependent scattering radius
        let l_val = l_cont.l1 as u32;
        let nrs = l_cont.n2 as usize; // number of resonances

        let mut resonances = Vec::with_capacity(nrs);

        // Each resonance is 6 values: ER, AJ, GN, GG, GFA, GFB
        let total_values = nrs * 6;
        let values = parse_list_values(lines, pos, total_values)?;

        for i in 0..nrs {
            let base = i * 6;
            resonances.push(Resonance {
                energy: values[base],  // ER (eV)
                j: values[base + 1],   // AJ (total J)
                gn: values[base + 2],  // GN (neutron width, eV)
                gg: values[base + 3],  // GG (gamma width, eV)
                gfa: values[base + 4], // GFA (fission width 1, eV)
                gfb: values[base + 5], // GFB (fission width 2, eV)
            });
        }

        l_groups.push(LGroup {
            l: l_val,
            awr: awr_l,
            apl,
            resonances,
        });
    }

    Ok(ResonanceRange {
        energy_low,
        energy_high,
        resolved: true,
        formalism: ResonanceFormalism::ReichMoore,
        target_spin,
        scattering_radius,
        l_groups,
    })
}

// ---------------------------------------------------------------------------
// Low-level ENDF line parsing helpers
// ---------------------------------------------------------------------------

/// Parsed CONT (control) record with 2 floats, 4 integers.
struct ContRecord {
    c1: f64,
    c2: f64,
    l1: i32,
    l2: i32,
    n1: i32,
    n2: i32,
}

/// Parse a CONT record from the current line.
fn parse_cont(lines: &[&str], pos: &mut usize) -> Result<ContRecord, EndfParseError> {
    if *pos >= lines.len() {
        return Err(EndfParseError::UnexpectedEof(
            "Expected CONT record but reached end of data".to_string(),
        ));
    }
    let line = lines[*pos];
    *pos += 1;

    Ok(ContRecord {
        c1: parse_endf_float(line, 0)?,
        c2: parse_endf_float(line, 1)?,
        l1: parse_endf_int(line, 2)?,
        l2: parse_endf_int(line, 3)?,
        n1: parse_endf_int(line, 4)?,
        n2: parse_endf_int(line, 5)?,
    })
}

/// Parse a LIST of floating-point values spanning multiple lines.
///
/// ENDF packs 6 values per line. We read ceil(n/6) lines.
fn parse_list_values(
    lines: &[&str],
    pos: &mut usize,
    n_values: usize,
) -> Result<Vec<f64>, EndfParseError> {
    let mut values = Vec::with_capacity(n_values);
    let n_lines = n_values.div_ceil(6);

    for _ in 0..n_lines {
        if *pos >= lines.len() {
            return Err(EndfParseError::UnexpectedEof(
                "Expected LIST data but reached end".to_string(),
            ));
        }
        let line = lines[*pos];
        *pos += 1;

        let remaining = n_values - values.len();
        let fields_on_line = remaining.min(6);

        for field in 0..fields_on_line {
            values.push(parse_endf_float(line, field)?);
        }
    }

    Ok(values)
}

/// Parse a floating-point value from an 11-character ENDF field.
///
/// ENDF uses Fortran-style floats that may omit 'E', e.g.:
/// - " 1.234567+2" means 1.234567e+2
/// - "-3.456789-1" means -3.456789e-1
/// - " 0.000000+0" means 0.0
fn parse_endf_float(line: &str, field_index: usize) -> Result<f64, EndfParseError> {
    let start = field_index * 11;
    let end = start + 11;

    if line.len() < end {
        // Short line — treat as zero.
        return Ok(0.0);
    }

    let field = &line[start..end];
    let trimmed = field.trim();

    if trimmed.is_empty() {
        return Ok(0.0);
    }

    // Try standard Rust float parsing first.
    if let Ok(v) = trimmed.parse::<f64>() {
        return Ok(v);
    }

    // Handle Fortran-style: "1.234567+2" or "-3.456789-1"
    // Look for +/- that is NOT the first character and NOT preceded by 'e'/'E'/'d'/'D'.
    let bytes = trimmed.as_bytes();
    for i in 1..bytes.len() {
        if (bytes[i] == b'+' || bytes[i] == b'-')
            && bytes[i - 1] != b'e'
            && bytes[i - 1] != b'E'
            && bytes[i - 1] != b'd'
            && bytes[i - 1] != b'D'
            && bytes[i - 1] != b'+'
            && bytes[i - 1] != b'-'
        {
            let mantissa = &trimmed[..i];
            let exponent = &trimmed[i..];
            let with_e = format!("{}E{}", mantissa, exponent);
            if let Ok(v) = with_e.parse::<f64>() {
                return Ok(v);
            }
        }
    }

    Err(EndfParseError::InvalidFloat(format!(
        "Cannot parse ENDF float: '{}'",
        field
    )))
}

/// Parse an integer from an 11-character ENDF field.
fn parse_endf_int(line: &str, field_index: usize) -> Result<i32, EndfParseError> {
    let start = field_index * 11;
    let end = start + 11;

    if line.len() < end {
        return Ok(0);
    }

    let field = &line[start..end];
    let trimmed = field.trim();

    if trimmed.is_empty() {
        return Ok(0);
    }

    // ENDF integers may have a decimal point (e.g., "1.000000+0" for 1).
    // Try integer parse first, then float-to-int.
    if let Ok(v) = trimmed.parse::<i32>() {
        return Ok(v);
    }

    // Try parsing as float then truncating.
    if let Ok(v) = parse_endf_float(line, field_index) {
        return Ok(v as i32);
    }

    Err(EndfParseError::InvalidFloat(format!(
        "Cannot parse ENDF int: '{}'",
        field
    )))
}

/// Skip an unresolved resonance range (we don't parse URR yet).
fn skip_unresolved_range(lines: &[&str], pos: &mut usize, _lrf: i32) -> Result<(), EndfParseError> {
    // Simplified: skip until we hit a SEND record (all zeros in fields 1-4)
    // or until the MF/MT changes. For robustness, skip the subsection CONT
    // and its data by reading NLS and skipping.
    let cont = parse_cont(lines, pos)?;
    let nls = cont.n1 as usize;

    for _ in 0..nls {
        let l_cont = parse_cont(lines, pos)?;
        let njs = l_cont.n2 as usize;
        for _ in 0..njs {
            let j_cont = parse_cont(lines, pos)?;
            let ne = j_cont.n2 as usize;
            // Each J-block has NE lines of data (6 values each).
            let n_lines = (ne * 6).div_ceil(6);
            *pos += n_lines;
        }
    }
    Ok(())
}

/// Skip a TAB1 record (interpolation table + function values).
fn skip_tab1(lines: &[&str], pos: &mut usize) -> Result<(), EndfParseError> {
    let cont = parse_cont(lines, pos)?;
    let nr = cont.n1 as usize; // number of interpolation ranges
    let np = cont.n2 as usize; // number of points

    // Interpolation ranges: 2 integers per range, 6 per line
    let interp_lines = (nr * 2).div_ceil(6);
    *pos += interp_lines;

    // Data points: 2 floats per point, 6 per line → np*2 values total
    let data_lines = (np * 2).div_ceil(6);
    *pos += data_lines;

    Ok(())
}

/// Errors from ENDF parsing.
#[derive(Debug, thiserror::Error)]
pub enum EndfParseError {
    #[error("Missing section: {0}")]
    MissingSection(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid float: {0}")]
    InvalidFloat(String),

    #[error("Unexpected end of file: {0}")]
    UnexpectedEof(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_endf_float_standard() {
        // ENDF fields are exactly 11 chars wide, no separators.
        // " 1.23456+2" in 11 chars = " 1.23456+02" (Fortran E11.4 style)
        //  01234567890  (field 0: cols 0-10, field 1: cols 11-21, etc.)
        let line = " 1.23456+02 2.34567-01 0.00000+00                                            ";
        assert!((parse_endf_float(line, 0).unwrap() - 123.456).abs() < 0.01);
        assert!((parse_endf_float(line, 1).unwrap() - 0.234567).abs() < 1e-6);
        assert!((parse_endf_float(line, 2).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_endf_float_with_e() {
        // 11-char fields: "1.23456E+02" "2.34567E-01"
        let line = "1.23456E+022.34567E-01                                                       ";
        assert!((parse_endf_float(line, 0).unwrap() - 123.456).abs() < 0.01);
        assert!((parse_endf_float(line, 1).unwrap() - 0.234567).abs() < 1e-6);
    }

    #[test]
    fn test_parse_endf_float_negative() {
        let line = "-1.23456+02-2.34567-01                                                       ";
        assert!((parse_endf_float(line, 0).unwrap() - (-123.456)).abs() < 0.01);
        assert!((parse_endf_float(line, 1).unwrap() - (-0.234567)).abs() < 1e-6);
    }

    #[test]
    fn test_parse_endf_int() {
        let line = "          0          1          2          3          4          5            ";
        assert_eq!(parse_endf_int(line, 0).unwrap(), 0);
        assert_eq!(parse_endf_int(line, 1).unwrap(), 1);
        assert_eq!(parse_endf_int(line, 2).unwrap(), 2);
    }

    /// Parse the SAMMY ex027 ENDF file for U-238 (Reich-Moore, LRF=3).
    ///
    /// This test validates against the SAMMY-distributed ENDF file.
    /// The first positive-energy resonance of U-238 is at 6.674 eV.
    #[test]
    fn test_parse_u238_sammy_endf() {
        let endf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../SAMMY/SAMMY/samexm_new/ex027_new/ex027.endf");

        if !endf_path.exists() {
            eprintln!(
                "Skipping test: SAMMY ENDF file not found at {:?}",
                endf_path
            );
            return;
        }

        let endf_text = std::fs::read_to_string(&endf_path).unwrap();
        let data = parse_endf_file2(&endf_text).unwrap();

        // Basic structure checks.
        assert_eq!(data.za, 92238, "Should be U-238");
        assert!((data.awr - 236.006).abs() < 0.01, "AWR should be ~236");
        assert!(!data.ranges.is_empty(), "Should have at least one range");

        let range = &data.ranges[0];
        assert!(range.resolved, "First range should be resolved");
        assert_eq!(
            range.formalism,
            ResonanceFormalism::ReichMoore,
            "U-238 ENDF uses Reich-Moore (LRF=3)"
        );
        assert!(
            (range.target_spin - 0.0).abs() < 1e-10,
            "U-238 target spin I=0"
        );
        assert!(
            (range.scattering_radius - 0.94285).abs() < 0.001,
            "Scattering radius ~0.94285 fm"
        );
        assert_eq!(range.l_groups.len(), 2, "Should have L=0 and L=1 groups");

        // Check first L-group (L=0).
        let l0 = &range.l_groups[0];
        assert_eq!(l0.l, 0, "First group should be L=0");
        assert!(
            l0.resonances.len() > 500,
            "L=0 should have hundreds of resonances"
        );

        // Find the famous 6.674 eV resonance of U-238.
        let first_positive = l0
            .resonances
            .iter()
            .find(|r| r.energy > 0.0)
            .expect("Should have positive-energy resonances");
        assert!(
            (first_positive.energy - 6.674).abs() < 0.01,
            "First positive resonance should be at 6.674 eV, got {}",
            first_positive.energy
        );
        assert!(
            (first_positive.j - 0.5).abs() < 1e-10,
            "6.674 eV resonance has J=0.5"
        );

        // The 6.674 eV resonance neutron width: ~1.493e-3 eV
        assert!(
            (first_positive.gn - 1.493e-3).abs() < 1e-5,
            "Neutron width should be ~1.493e-3 eV, got {}",
            first_positive.gn
        );
        // Gamma width: ~2.3e-2 eV
        assert!(
            (first_positive.gg - 2.3e-2).abs() < 1e-3,
            "Gamma width should be ~2.3e-2 eV, got {}",
            first_positive.gg
        );

        let total = data.total_resonance_count();
        println!(
            "U-238 ENDF parsed successfully: {} total resonances across {} L-groups",
            total,
            range.l_groups.len()
        );
        println!(
            "  L=0: {} resonances, L=1: {} resonances",
            l0.resonances.len(),
            range.l_groups[1].resonances.len()
        );
    }
}
