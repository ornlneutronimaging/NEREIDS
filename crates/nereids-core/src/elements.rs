//! Element and isotope reference data.
//!
//! Provides element symbols, names, and natural isotopic abundances
//! for all elements relevant to neutron resonance imaging.
//!
//! Delegates to the `endf-mat` crate for raw lookup data.

use crate::error::NereidsError;
use crate::types::Isotope;

/// Element symbol lookup by atomic number Z.
///
/// Returns `Some("n")` for Z=0 (neutron). Returns `None` for Z > 118.
pub fn element_symbol(z: u32) -> Option<&'static str> {
    endf_mat::element_symbol(z)
}

/// Element name lookup by atomic number Z.
pub fn element_name(z: u32) -> Option<&'static str> {
    endf_mat::element_name(z)
}

/// Parse an isotope string like "U-238", "Pu-239", "Fe-56".
///
/// Returns `None` if the string cannot be parsed or validation fails.
pub fn parse_isotope_str(s: &str) -> Option<Isotope> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        return None;
    }
    let symbol = parts[0].trim();
    let a: u32 = parts[1].trim().parse().ok()?;
    let z = symbol_to_z(symbol)?;
    Isotope::new(z, a).ok()
}

/// Look up atomic number Z from element symbol (case-insensitive).
pub fn symbol_to_z(symbol: &str) -> Option<u32> {
    endf_mat::symbol_to_z(symbol)
}

/// Natural isotopic abundance for a given isotope, as a fraction (0.0 to 1.0).
///
/// Returns `None` if the isotope is not in the database (e.g., synthetic isotopes).
/// Data from IUPAC 2016 recommended values (via NIST).
pub fn natural_abundance(isotope: &Isotope) -> Option<f64> {
    endf_mat::natural_abundance(isotope.z(), isotope.a())
}

/// Get all naturally occurring isotopes for element Z.
///
/// Silently skips any isotopes that fail validation (should never happen
/// for data from the `endf_mat` crate, but defensive nonetheless).
pub fn natural_isotopes(z: u32) -> Vec<(Isotope, f64)> {
    endf_mat::natural_isotopes(z)
        .into_iter()
        .filter_map(|(a, frac)| Isotope::new(z, a).ok().map(|iso| (iso, frac)))
        .collect()
}

/// Get all isotopes with ENDF evaluations for element Z.
///
/// Unlike [`natural_isotopes`], this includes synthetic and transuranic
/// isotopes (Tc, Pm, Np, Pu, Am, etc.) that have no natural abundance
/// but do have evaluated nuclear data files.
pub fn known_isotopes(z: u32) -> Vec<Isotope> {
    endf_mat::known_isotopes(z)
        .into_iter()
        .filter_map(|a| Isotope::new(z, a).ok())
        .collect()
}

/// Whether the ENDF/B-VIII.0 sublibrary has an evaluation for (Z, A).
pub fn has_endf_evaluation(z: u32, a: u32) -> bool {
    endf_mat::has_endf_evaluation(z, a)
}

/// Compute ENDF ZA identifier: Z * 1000 + A.
pub fn za_from_isotope(isotope: &Isotope) -> u32 {
    endf_mat::za(isotope.z(), isotope.a())
}

/// Parse ENDF ZA identifier back to an [`Isotope`].
///
/// # Errors
/// Returns `NereidsError::InvalidParameter` when the ZA value produces
/// an invalid isotope (Z > A, or A == 0 — e.g. ZA = 26000 for natural
/// iron).  Callers should propagate or handle this gracefully instead of
/// panicking, since real ENDF files may contain such entries.
pub fn isotope_from_za(za: u32) -> Result<Isotope, NereidsError> {
    Isotope::new(endf_mat::z_from_za(za), endf_mat::a_from_za(za))
}

/// Format isotope as standard string, e.g. "U-238".
pub fn isotope_to_string(isotope: &Isotope) -> String {
    match element_symbol(isotope.z()) {
        Some(sym) => format!("{}-{}", sym, isotope.a()),
        None => format!("Z{}-{}", isotope.z(), isotope.a()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_symbol() {
        assert_eq!(element_symbol(1), Some("H"));
        assert_eq!(element_symbol(92), Some("U"));
        assert_eq!(element_symbol(26), Some("Fe"));
        assert_eq!(element_symbol(0), Some("n"));
    }

    #[test]
    fn test_parse_isotope_str() {
        let u238 = parse_isotope_str("U-238").unwrap();
        assert_eq!(u238.z(), 92);
        assert_eq!(u238.a(), 238);

        let fe56 = parse_isotope_str("Fe-56").unwrap();
        assert_eq!(fe56.z(), 26);
        assert_eq!(fe56.a(), 56);

        assert!(parse_isotope_str("invalid").is_none());
    }

    #[test]
    fn test_za_roundtrip() {
        let iso = Isotope::new(92, 238).unwrap();
        let za = za_from_isotope(&iso);
        assert_eq!(za, 92238);
        let back = isotope_from_za(za).unwrap();
        assert_eq!(back, iso);
    }

    #[test]
    fn test_isotope_from_za_natural_element_returns_error() {
        // ZA=26000 → Z=26, A=0 (natural iron). A==0 fails validation.
        let result = isotope_from_za(26000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be positive"));
    }

    #[test]
    fn test_isotope_from_za_invalid_z_greater_than_a() {
        // Contrived ZA where Z > A (malformed data).
        // ZA=999001 → Z=999, A=1 → Z > A.
        let result = isotope_from_za(999001);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot exceed"));
    }

    #[test]
    fn test_natural_abundance() {
        let u238 = Isotope::new(92, 238).unwrap();
        let abund = natural_abundance(&u238).unwrap();
        assert!((abund - 0.992742).abs() < 1e-6);
    }

    #[test]
    fn test_natural_isotopes() {
        let fe_isotopes = natural_isotopes(26);
        assert_eq!(fe_isotopes.len(), 4);
        let total: f64 = fe_isotopes.iter().map(|(_, a)| a).sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_isotope_to_string() {
        let iso = Isotope::new(92, 238).unwrap();
        assert_eq!(isotope_to_string(&iso), "U-238");
    }

    #[test]
    fn test_known_isotopes_plutonium() {
        let pu = known_isotopes(94);
        assert!(!pu.is_empty());
        assert!(pu.iter().any(|iso| iso.a() == 239));
    }

    #[test]
    fn test_known_isotopes_synthetic_element() {
        // Tc has no natural isotopes but has ENDF evaluations
        assert!(natural_isotopes(43).is_empty());
        let tc = known_isotopes(43);
        assert!(!tc.is_empty());
    }

    #[test]
    fn test_known_isotopes_superset_of_natural() {
        let natural: Vec<Isotope> = natural_isotopes(26)
            .into_iter()
            .map(|(iso, _)| iso)
            .collect();
        let known = known_isotopes(26);
        for iso in &natural {
            assert!(known.contains(iso));
        }
        // Fe-55 is in ENDF but not natural
        assert!(known.iter().any(|iso| iso.a() == 55));
    }

    #[test]
    fn test_has_endf_evaluation() {
        assert!(has_endf_evaluation(94, 239));
        assert!(!has_endf_evaluation(94, 999));
    }
}
