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

        // Parse the line (whitespace-separated)
        let parts: Vec<&str> = trimmed.split_whitespace().collect();

        // Skip lines with fewer than 3 columns (likely blank lines or EOF markers)
        if parts.len() < 3 {
            continue;
        }

        // Parse floating point values
        let energy: f64 = parts[0]
            .parse()
            .map_err(|e| format!("Line {}: failed to parse energy: {}", line_num + 1, e))?;

        let datum: f64 = parts[1]
            .parse()
            .map_err(|e| format!("Line {}: failed to parse data: {}", line_num + 1, e))?;

        let uncertainty: f64 = parts[2].parse().map_err(|e| {
            format!(
                "Line {}: failed to parse uncertainty: {}",
                line_num + 1,
                e
            )
        })?;

        energies.push(energy);
        data.push(datum);
        uncertainties.push(uncertainty);
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
    use std::path::PathBuf;

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
}
