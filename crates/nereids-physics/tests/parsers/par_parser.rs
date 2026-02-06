//! Parser for SAMMY .par files (resonance parameters).
//!
//! PAR format: 5 floating-point columns (11 characters each) plus variation flags.
//! - Column 1: E_r (resonance energy in eV)
//! - Column 2: Γ_γ (capture width in milliEV)
//! - Column 3: Γ_n (neutron width in milliEV)
//! - Column 4: Γ_fa (first fission width in milliEV)
//! - Column 5: Γ_fb (second fission width in milliEV)
//! - Columns 6+: Variation flags (1 = fixed, 0 = vary)

use nereids_core::nuclear::{FissionWidths, Parameter, Resonance};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Parse the line
        let parts: Vec<&str> = trimmed.split_whitespace().collect();

        if parts.len() < 5 {
            return Err(format!(
                "Line {}: expected at least 5 columns, got {}",
                line_num + 1,
                parts.len()
            )
            .into());
        }

        // Parse floating point values
        let e_r: f64 = parts[0].parse().map_err(|e| {
            format!("Line {}: failed to parse energy: {}", line_num + 1, e)
        })?;

        let gamma_g_milliev: f64 = parts[1].parse().map_err(|e| {
            format!(
                "Line {}: failed to parse capture width: {}",
                line_num + 1,
                e
            )
        })?;

        let gamma_n_milliev: f64 = parts[2].parse().map_err(|e| {
            format!("Line {}: failed to parse neutron width: {}", line_num + 1, e)
        })?;

        let gamma_fa_milliev: f64 = parts[3].parse().map_err(|e| {
            format!(
                "Line {}: failed to parse fission width 1: {}",
                line_num + 1,
                e
            )
        })?;

        let gamma_fb_milliev: f64 = parts[4].parse().map_err(|e| {
            format!(
                "Line {}: failed to parse fission width 2: {}",
                line_num + 1,
                e
            )
        })?;

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
    use std::path::PathBuf;

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
}
