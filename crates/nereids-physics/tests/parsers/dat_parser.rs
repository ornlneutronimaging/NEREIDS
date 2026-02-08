//! Parser for SAMMY .dat files (experimental data).
//!
//! DAT format: "twenty" format with 20-character fields.
//! - Column 1: Energy (eV)
//! - Column 2: Data value (cross section or transmission)
//! - Column 3: Uncertainty (standard deviation)

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parsed experimental data from a SAMMY .dat file.
#[derive(Debug, Clone)]
pub struct ExperimentalData {
    pub energies: Vec<f64>,
    pub data: Vec<f64>,
    pub uncertainties: Vec<f64>,
}

fn parse_dat_values_from_line(line: &str) -> Option<Vec<(f64, f64, f64)>> {
    // Compact fixed-block format used by ex006/ex007 fixtures:
    // repeated records of width 37 = [15-char energy][15-char data][7-char unc].
    if line.len() >= 37 && line.len().is_multiple_of(37) {
        let mut out = Vec::with_capacity(line.len() / 37);
        let mut ok = true;
        for k in 0..(line.len() / 37) {
            let base = k * 37;
            let e_field = line.get(base..base + 15).unwrap_or("").trim();
            let d_field = line.get(base + 15..base + 30).unwrap_or("").trim();
            let u_field = line.get(base + 30..base + 37).unwrap_or("").trim();
            match (
                e_field.parse::<f64>(),
                d_field.parse::<f64>(),
                u_field.parse::<f64>(),
            ) {
                (Ok(e), Ok(d), Ok(u)) => out.push((e, d, u)),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && !out.is_empty() {
            return Some(out);
        }
    }

    // SAMMY "twenty" format: 3 fixed-width 20-char fields.
    if line.len() >= 60 {
        let mut vals = [0.0_f64; 3];
        let mut ok = true;
        for (i, slot) in vals.iter_mut().enumerate() {
            let start = i * 20;
            let end = start + 20;
            let field = line.get(start..end).unwrap_or("").trim();
            match field.parse::<f64>() {
                Ok(v) => *slot = v,
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(vec![(vals[0], vals[1], vals[2])]);
        }
    }

    // Fallback for whitespace-delimited variants (single row).
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let energy = parts[0].parse::<f64>().ok()?;
    let datum = parts[1].parse::<f64>().ok()?;
    let uncertainty = parts[2].parse::<f64>().ok()?;
    Some(vec![(energy, datum, uncertainty)])
}

/// Parse a SAMMY .dat file and extract experimental data.
///
/// # Arguments
///
/// * `path` - Path to the .dat file
///
/// # Returns
///
/// `ExperimentalData` containing energy points, data values, and uncertainties.
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be read
/// - Lines cannot be parsed as floats
/// - Format is invalid
pub fn parse_dat_file(path: &Path) -> Result<ExperimentalData, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut energies = Vec::new();
    let mut data = Vec::new();
    let mut uncertainties = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let rows = parse_dat_values_from_line(&line).ok_or_else(|| {
            format!(
                "Line {}: failed to parse DAT row as fixed-width or whitespace-delimited numeric fields",
                line_num + 1
            )
        })?;

        for (energy, datum, uncertainty) in rows {
            // Some SAMMY fixtures terminate with a sentinel row of zeros.
            if energy <= 0.0 {
                continue;
            }
            energies.push(energy);
            data.push(datum);
            uncertainties.push(uncertainty);
        }
    }

    Ok(ExperimentalData {
        energies,
        data,
        uncertainties,
    })
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
        std::env::temp_dir().join(format!("nereids_dat_{nanos}_{suffix}"))
    }

    #[test]
    fn test_parse_ex003c_dat() {
        let path = PathBuf::from("tests/fixtures/sammy_reference/ex003/input/ex003c.dat");

        // Skip test if file doesn't exist (e.g., CI without fixtures)
        if !path.exists() {
            eprintln!("Skipping test: file not found");
            return;
        }

        let data = parse_dat_file(&path).unwrap();

        // ex003c should have many data points (synthetic data)
        assert!(data.energies.len() > 100);
        assert_eq!(data.energies.len(), data.data.len());
        assert_eq!(data.energies.len(), data.uncertainties.len());

        // Energies should be positive
        for energy in &data.energies {
            assert!(*energy > 0.0);
        }

        // Data and uncertainties should be positive
        for datum in &data.data {
            assert!(*datum > 0.0);
        }

        for uncertainty in &data.uncertainties {
            assert!(*uncertainty > 0.0);
        }
    }

    #[test]
    fn test_parse_fixed_width_dat_without_whitespace_delimiters() {
        let path = unique_temp_path("fixed.dat");
        let content = format!(
            "{:>20}{:>20}{:>20}\n",
            "1.0000000000E+00", "2.5000000000E+01", "5.0000000000E-01"
        );
        fs::write(&path, content).unwrap();

        let data = parse_dat_file(&path).unwrap();
        assert_eq!(data.energies.len(), 1);
        assert!((data.energies[0] - 1.0).abs() < 1e-12);
        assert!((data.data[0] - 25.0).abs() < 1e-12);
        assert!((data.uncertainties[0] - 0.5).abs() < 1e-12);

        let _ = fs::remove_file(path);
    }
}
