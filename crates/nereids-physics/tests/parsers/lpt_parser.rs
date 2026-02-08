//! Parser for SAMMY .lpt files (list output with chi-squared).
//!
//! LPT format: SAMMY text output containing chi-squared statistics.
//! We extract:
//! - "CUSTOMARY CHI SQUARED = <value>"
//! - "CUSTOMARY CHI SQUARED DIVIDED BY NDAT = <value>"

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parsed chi-squared statistics from a SAMMY .lpt file.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquaredStats {
    pub chi_squared: f64,
    pub chi_squared_per_point: f64,
}

/// One tabulated theoretical value from an LPT "THEORETICAL VALUES" block.
#[derive(Debug, Clone, Copy)]
pub struct TheoryPoint {
    pub energy_ev: f64,
    pub theory: f64,
}

fn parse_float_prefix(s: &str) -> Option<(f64, usize)> {
    let trimmed_start = s.len() - s.trim_start().len();
    let s = &s[trimmed_start..];
    if s.is_empty() {
        return None;
    }
    let end = s.find(char::is_whitespace).unwrap_or(s.len());
    let token = &s[..end];
    let val = token.parse::<f64>().ok()?;
    Some((val, trimmed_start + end))
}

fn parse_theory_points_from_line(mut line: &str) -> Vec<TheoryPoint> {
    let mut out = Vec::new();
    while let Some(close_idx) = line.find(')') {
        let after = &line[close_idx + 1..];
        let Some((energy, used_e)) = parse_float_prefix(after) else {
            line = after;
            continue;
        };
        let Some((theory, used_t)) = parse_float_prefix(&after[used_e..]) else {
            line = after;
            continue;
        };
        out.push(TheoryPoint {
            energy_ev: energy,
            theory,
        });
        line = &after[used_e + used_t..];
    }
    out
}

/// Parse a SAMMY .lpt file and extract chi-squared statistics.
///
/// # Arguments
///
/// * `path` - Path to the .lpt file
///
/// # Returns
///
/// `ChiSquaredStats` containing chi-squared and chi-squared per data point.
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be read
/// - Chi-squared line not found
/// - Values cannot be parsed
pub fn parse_lpt_chi_squared(path: &Path) -> Result<ChiSquaredStats, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut chi_squared = None;
    let mut chi_squared_per_point = None;

    for line_result in reader.lines() {
        let line = line_result?;

        // Search for "CUSTOMARY CHI SQUARED ="
        if line.contains("CUSTOMARY CHI SQUARED") && line.contains("DIVIDED BY NDAT") {
            // Extract value after "="
            if let Some(eq_pos) = line.rfind('=') {
                let value_str = line[eq_pos + 1..].trim();
                chi_squared_per_point = Some(
                    value_str
                        .parse::<f64>()
                        .map_err(|e| format!("Failed to parse chi²/NDAT: {e}"))?,
                );
            }
        } else if line.contains("CUSTOMARY CHI SQUARED") && !line.contains("DIVIDED BY NDAT") {
            // Extract value after "="
            if let Some(eq_pos) = line.rfind('=') {
                let value_str = line[eq_pos + 1..].trim();
                chi_squared = Some(
                    value_str
                        .parse::<f64>()
                        .map_err(|e| format!("Failed to parse chi²: {e}"))?,
                );
            }
        }

        // Keep scanning the full file and retain the latest values.
        // Iterative SAMMY outputs can contain multiple chi-squared blocks.
    }

    match (chi_squared, chi_squared_per_point) {
        (Some(chi2), Some(chi2_per_pt)) => Ok(ChiSquaredStats {
            chi_squared: chi2,
            chi_squared_per_point: chi2_per_pt,
        }),
        (None, _) => Err("Chi-squared value not found in LPT file".into()),
        (_, None) => Err("Chi-squared per point not found in LPT file".into()),
    }
}

/// Parse the final "THEORETICAL VALUES" table from a SAMMY LPT file.
pub fn parse_lpt_theory_points(
    path: &Path,
) -> Result<Vec<TheoryPoint>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut in_block = false;
    let mut current = Vec::new();
    let mut last_complete: Option<Vec<TheoryPoint>> = None;

    for line_result in reader.lines() {
        let line = line_result?;
        if line.contains("THEORETICAL VALUES") {
            in_block = true;
            current.clear();
            continue;
        }

        if !in_block {
            continue;
        }

        if line.contains("CUSTOMARY CHI SQUARED")
            || line.contains("### Array size used")
            || line.contains("BAYESIAN CHI SQUARED")
        {
            if !current.is_empty() {
                last_complete = Some(current.clone());
            }
            in_block = false;
            continue;
        }

        let parsed = parse_theory_points_from_line(&line);
        if !parsed.is_empty() {
            current.extend(parsed);
        }
    }

    if in_block && !current.is_empty() {
        last_complete = Some(current);
    }

    last_complete.ok_or_else(|| "No theoretical-values table found in LPT file".into())
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
        std::env::temp_dir().join(format!("nereids_lpt_{nanos}_{suffix}"))
    }

    #[test]
    fn test_parse_ex003c_lpt() {
        let path = PathBuf::from("tests/fixtures/sammy_reference/ex003/expected/ex003c.lpt");

        // Skip test if file doesn't exist (e.g., CI without fixtures)
        if !path.exists() {
            eprintln!("Skipping test: file not found");
            return;
        }

        let stats = parse_lpt_chi_squared(&path).unwrap();

        // Check that values are positive and reasonable
        assert!(stats.chi_squared > 0.0);
        assert!(stats.chi_squared_per_point > 0.0);

        // Chi-squared per point should be smaller than total chi-squared
        assert!(stats.chi_squared_per_point < stats.chi_squared);

        // Values should be finite
        assert!(stats.chi_squared.is_finite());
        assert!(stats.chi_squared_per_point.is_finite());
    }

    #[test]
    fn test_parse_all_ex003_lpt() {
        let variants = ["ex003a", "ex003c", "ex003e", "ex003f", "ex003x", "ex003t"];

        for variant in &variants {
            let path = PathBuf::from(format!(
                "tests/fixtures/sammy_reference/ex003/expected/{variant}.lpt"
            ));

            // Skip test if file doesn't exist
            if !path.exists() {
                eprintln!("Skipping test for {variant}: file not found");
                continue;
            }

            let stats = parse_lpt_chi_squared(&path).unwrap();

            // Basic sanity checks
            assert!(
                stats.chi_squared.is_finite(),
                "{variant}: chi² should be finite"
            );
            assert!(
                stats.chi_squared_per_point.is_finite(),
                "{variant}: chi²/NDAT should be finite"
            );
            assert!(
                stats.chi_squared > 0.0,
                "{variant}: chi² should be positive"
            );
        }
    }

    #[test]
    fn test_parse_lpt_uses_last_chi_squared_block() {
        let path = unique_temp_path("iter.lpt");
        let content = "\
CUSTOMARY CHI SQUARED = 1.0
CUSTOMARY CHI SQUARED DIVIDED BY NDAT = 0.1
CUSTOMARY CHI SQUARED = 2.0
CUSTOMARY CHI SQUARED DIVIDED BY NDAT = 0.2
";
        fs::write(&path, content).unwrap();

        let stats = parse_lpt_chi_squared(&path).unwrap();
        assert!((stats.chi_squared - 2.0).abs() < 1e-12);
        assert!((stats.chi_squared_per_point - 0.2).abs() < 1e-12);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_parse_ex007_theory_points() {
        let path = PathBuf::from("tests/fixtures/sammy_reference/ex007/expected/ex007awl.lpt");
        if !path.exists() {
            eprintln!("Skipping test: file not found");
            return;
        }

        let pts = parse_lpt_theory_points(&path).unwrap();
        assert_eq!(pts.len(), 56);
        assert!(pts[0].energy_ev > 180_000.0);
        assert!(pts[0].theory > 0.0);
        assert!(pts[55].energy_ev < 183_000.5);
        assert!(pts[55].theory > 0.0);
    }
}
