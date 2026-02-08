//! Parser for SAMMY .par files (resonance parameters).
//!
//! PAR format: 5 floating-point columns (11 characters each) plus variation flags.
//! - Column 1: `E_r` (resonance energy in eV)
//! - Column 2: `Γ_γ` (capture width in milliEV)
//! - Column 3: `Γ_n` (neutron width in milliEV)
//! - Column 4: `Γ_fa` (first fission width in milliEV)
//! - Column 5: `Γ_fb` (second fission width in milliEV)
//! - Columns 6+: Variation flags (1 = fixed, 0 = vary)

use nereids_core::nuclear::{FissionWidths, Parameter, Resonance};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn parse_par_first_five_fields(line: &str) -> Option<[f64; 5]> {
    parse_par_first_five_fields_fixed_width(line).or_else(|| parse_par_first_five_fields_ws(line))
}

fn parse_par_first_five_fields_fixed_width(line: &str) -> Option<[f64; 5]> {
    // Canonical SAMMY PAR layout: first 5 values in fixed-width 11-char fields.
    if line.len() >= 55 {
        let mut vals = [0.0_f64; 5];
        let mut ok = true;
        for (i, slot) in vals.iter_mut().enumerate() {
            let start = i * 11;
            let end = start + 11;
            let field = line.get(start..end).unwrap_or("").trim();
            if field.is_empty() {
                if i < 3 {
                    // E_r, gamma_g, and gamma_n are required fields.
                    ok = false;
                    break;
                }
                // Some SAMMY PAR variants leave optional fission-width columns blank.
                *slot = 0.0;
                continue;
            }
            match field.parse::<f64>() {
                Ok(v) => *slot = v,
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(vals);
        }
    }
    None
}

fn parse_par_first_five_fields_ws(line: &str) -> Option<[f64; 5]> {
    // Fallback for whitespace-delimited fixtures/variants.
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    Some([
        parts[0].parse::<f64>().ok()?,
        parts[1].parse::<f64>().ok()?,
        parts[2].parse::<f64>().ok()?,
        parts[3].parse::<f64>().ok()?,
        parts[4].parse::<f64>().ok()?,
    ])
}

/// Parse a SAMMY .par file and extract resonance parameters.
///
/// # Arguments
///
/// * `path` - Path to the .par file
///
/// # Returns
///
/// Vector of `Resonance` objects parsed from the file.
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be read
/// - Lines cannot be parsed as floats
/// - Format is invalid
pub fn parse_par_file(path: &Path) -> Result<Vec<Resonance>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut resonances = Vec::new();
    let mut seen_resonance_block = false;

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let from_fixed_width = parse_par_first_five_fields_fixed_width(&line).is_some();
        let Some([e_r, gamma_g_milliev, gamma_n_milliev, gamma_fa_milliev, gamma_fb_milliev]) =
            parse_par_first_five_fields(&line)
        else {
            // Resonance records are expected as one contiguous block at the
            // top of these SAMMY PAR fixtures. Once that block ends, stop so
            // later numeric control sections are not misinterpreted.
            if seen_resonance_block {
                break;
            }
            continue;
        };
        // Resonance records in our SAMMY fixtures carry additional control/index
        // fields beyond the first 5 numeric columns. This skips short numeric
        // records used by non-resonance sections (e.g., relative uncertainties).
        if !from_fixed_width && line.split_whitespace().count() < 6 {
            if seen_resonance_block {
                break;
            }
            continue;
        }
        seen_resonance_block = true;

        // Convert milliEV to eV
        let gamma_g = gamma_g_milliev / 1000.0;
        let gamma_n = gamma_n_milliev / 1000.0;
        let gamma_fa = gamma_fa_milliev / 1000.0;
        let gamma_fb = gamma_fb_milliev / 1000.0;

        // Create fission widths if non-zero
        let fission = if gamma_fa.abs() > 1e-30 || gamma_fb.abs() > 1e-30 {
            Some(FissionWidths {
                gamma_f1: Parameter::fixed(gamma_fa),
                gamma_f2: Parameter::fixed(gamma_fb),
            })
        } else {
            None
        };

        // Create resonance
        let resonance = Resonance {
            energy: Parameter::fixed(e_r),
            gamma_n: Parameter::fixed(gamma_n),
            gamma_g: Parameter::fixed(gamma_g),
            fission,
        };

        resonances.push(resonance);
    }

    Ok(resonances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(suffix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("nereids_par_{nanos}_{suffix}"))
    }

    #[test]
    fn test_parse_ex003c_par() {
        let path = PathBuf::from("tests/fixtures/sammy_reference/ex003/input/ex003c.par");

        // Skip test if file doesn't exist (e.g., CI without fixtures)
        if !path.exists() {
            eprintln!("Skipping test: file not found");
            return;
        }

        let resonances = parse_par_file(&path).unwrap();

        // ex003c should have 12 resonances
        assert_eq!(resonances.len(), 12);

        // First resonance should be at 0.25 eV
        assert!((resonances[0].energy.value - 0.25).abs() < 1e-10);

        // Check neutron width for first resonance (0.5 milliEV → 0.0005 eV)
        assert!((resonances[0].gamma_n.value - 0.0005).abs() < 1e-10);

        // Check capture width from PAR col2 (1.0 milliEV → 0.001 eV)
        assert!((resonances[0].gamma_g.value - 0.001).abs() < 1e-10);

        // Check fission widths exist
        assert!(resonances[0].fission.is_some());
        let fission = resonances[0].fission.as_ref().unwrap();
        assert!((fission.gamma_f1.value - 0.0005).abs() < 1e-10); // 0.5 milliEV
        assert!((fission.gamma_f2.value - 0.0005).abs() < 1e-10); // 0.5 milliEV
    }

    #[test]
    fn test_parse_fixed_width_par_without_whitespace_delimiters() {
        let path = unique_temp_path("fixed.par");
        let line = format!(
            "{:>11}{:>11}{:>11}{:>11}{:>11}{:>11}\n",
            "2.5E-1", "1.0", "0.5", "0.5", "0.5", "1"
        );
        fs::write(&path, line).unwrap();

        let resonances = parse_par_file(&path).unwrap();
        assert_eq!(resonances.len(), 1);
        assert!((resonances[0].energy.value - 0.25).abs() < 1e-12);
        assert!((resonances[0].gamma_g.value - 0.001).abs() < 1e-12);
        assert!((resonances[0].gamma_n.value - 0.0005).abs() < 1e-12);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_parse_compact_fixed_width_par_without_token_boundaries() {
        let path = unique_temp_path("compact_fixed.par");
        let line = format!(
            "{}{}{}{}{}{}\n",
            format!("{:011.4E}", 2.5e-1),
            format!("{:011.4E}", 1.0),
            format!("{:011.4E}", 0.5),
            format!("{:011.4E}", 0.5),
            format!("{:011.4E}", 0.5),
            "00000000001"
        );
        fs::write(&path, line).unwrap();

        let resonances = parse_par_file(&path).unwrap();
        assert_eq!(resonances.len(), 1);
        assert!((resonances[0].energy.value - 0.25).abs() < 1e-12);
        assert!((resonances[0].gamma_g.value - 0.001).abs() < 1e-12);
        assert!((resonances[0].gamma_n.value - 0.0005).abs() < 1e-12);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_parse_ignores_post_resonance_numeric_control_sections() {
        let path = unique_temp_path("post_block_controls.par");
        let contents = concat!(
            " 0.25      1.         0.5        0.5       0.5          1 0 0 0 0 1\n",
            "\n",
            ".150000000\n",
            "ORRESolution function parameters follow\n",
            "NORMAlization and \"constant\" background follow\n",
            "1.00191245 .09055000 0.        0.        0.        0.        1 1 0 0 0 0\n",
        );
        fs::write(&path, contents).unwrap();

        let resonances = parse_par_file(&path).unwrap();
        assert_eq!(resonances.len(), 1);
        assert!((resonances[0].energy.value - 0.25).abs() < 1e-12);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_parse_ignores_fixed_width_control_rows_with_blank_gamma_columns() {
        let path = unique_temp_path("fixed_width_controls.par");
        let resonance = format!(
            "{:>11}{:>11}{:>11}{:>11}{:>11}{:>11}\n",
            "2.5E-1", "1.0", "0.5", "0.5", "0.5", "1"
        );
        // Matches the ex007 control-row shape: fixed-width line with blank
        // gamma columns and a single numeric uncertainty term later in the row.
        let control = "-5.9951E+04                             0.05                       0 0 1\n";
        fs::write(&path, format!("{resonance}{control}")).unwrap();

        let resonances = parse_par_file(&path).unwrap();
        assert_eq!(resonances.len(), 1);
        assert!((resonances[0].energy.value - 0.25).abs() < 1e-12);

        let _ = fs::remove_file(path);
    }
}
