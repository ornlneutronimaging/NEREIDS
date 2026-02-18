//! Element and isotope reference data.
//!
//! Provides element symbols, names, and natural isotopic abundances
//! for all elements relevant to neutron resonance imaging.

use crate::types::Isotope;

/// Element symbol lookup by atomic number Z.
///
/// Returns `None` for Z > 118 or Z == 0.
pub fn element_symbol(z: u32) -> Option<&'static str> {
    ELEMENT_SYMBOLS.get(z as usize).copied()
}

/// Element name lookup by atomic number Z.
pub fn element_name(z: u32) -> Option<&'static str> {
    ELEMENT_NAMES.get(z as usize).copied()
}

/// Parse an isotope string like "U-238", "Pu-239", "Fe-56".
///
/// Returns `None` if the string cannot be parsed.
pub fn parse_isotope_str(s: &str) -> Option<Isotope> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        return None;
    }
    let symbol = parts[0].trim();
    let a: u32 = parts[1].trim().parse().ok()?;
    let z = symbol_to_z(symbol)?;
    Some(Isotope::new(z, a))
}

/// Look up atomic number Z from element symbol (case-insensitive on first char).
pub fn symbol_to_z(symbol: &str) -> Option<u32> {
    let symbol_lower = symbol.to_lowercase();
    for (z, &sym) in ELEMENT_SYMBOLS.iter().enumerate() {
        if sym.to_lowercase() == symbol_lower {
            return Some(z as u32);
        }
    }
    None
}

/// Natural isotopic abundance for a given isotope, as a fraction (0.0 to 1.0).
///
/// Returns `None` if the isotope is not in the database (e.g., synthetic isotopes).
/// Data from IUPAC 2016 recommended values.
pub fn natural_abundance(isotope: &Isotope) -> Option<f64> {
    NATURAL_ABUNDANCES
        .iter()
        .find(|(z, a, _)| *z == isotope.z && *a == isotope.a)
        .map(|(_, _, abundance)| *abundance)
}

/// Get all naturally occurring isotopes for element Z.
pub fn natural_isotopes(z: u32) -> Vec<(Isotope, f64)> {
    NATURAL_ABUNDANCES
        .iter()
        .filter(|(zz, _, _)| *zz == z)
        .map(|(zz, a, abund)| (Isotope::new(*zz, *a), *abund))
        .collect()
}

/// Compute ENDF ZA identifier: Z * 1000 + A.
pub fn za_from_isotope(isotope: &Isotope) -> u32 {
    isotope.z * 1000 + isotope.a
}

/// Parse ENDF ZA identifier back to (Z, A).
pub fn isotope_from_za(za: u32) -> Isotope {
    Isotope::new(za / 1000, za % 1000)
}

/// Format isotope as standard string, e.g. "U-238".
pub fn isotope_to_string(isotope: &Isotope) -> String {
    match element_symbol(isotope.z) {
        Some(sym) => format!("{}-{}", sym, isotope.a),
        None => format!("Z{}-{}", isotope.z, isotope.a),
    }
}

// Element symbols indexed by Z (Z=0 is placeholder).
#[rustfmt::skip]
static ELEMENT_SYMBOLS: &[&str] = &[
    "n",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  // 0-9
    "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  // 10-19
    "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", // 20-29
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  // 30-39
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", // 40-49
    "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", // 50-59
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", // 60-69
    "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", // 70-79
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", // 80-89
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", // 90-99
    "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", // 100-109
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",       // 110-118
];

#[rustfmt::skip]
static ELEMENT_NAMES: &[&str] = &[
    "neutron",     "Hydrogen",    "Helium",      "Lithium",     "Beryllium",
    "Boron",       "Carbon",      "Nitrogen",    "Oxygen",      "Fluorine",
    "Neon",        "Sodium",      "Magnesium",   "Aluminum",    "Silicon",
    "Phosphorus",  "Sulfur",      "Chlorine",    "Argon",       "Potassium",
    "Calcium",     "Scandium",    "Titanium",    "Vanadium",    "Chromium",
    "Manganese",   "Iron",        "Cobalt",      "Nickel",      "Copper",
    "Zinc",        "Gallium",     "Germanium",   "Arsenic",     "Selenium",
    "Bromine",     "Krypton",     "Rubidium",    "Strontium",   "Yttrium",
    "Zirconium",   "Niobium",     "Molybdenum",  "Technetium",  "Ruthenium",
    "Rhodium",     "Palladium",   "Silver",      "Cadmium",     "Indium",
    "Tin",         "Antimony",    "Tellurium",   "Iodine",      "Xenon",
    "Cesium",      "Barium",      "Lanthanum",   "Cerium",      "Praseodymium",
    "Neodymium",   "Promethium",  "Samarium",    "Europium",    "Gadolinium",
    "Terbium",     "Dysprosium",  "Holmium",     "Erbium",      "Thulium",
    "Ytterbium",   "Lutetium",    "Hafnium",     "Tantalum",    "Tungsten",
    "Rhenium",     "Osmium",      "Iridium",     "Platinum",    "Gold",
    "Mercury",     "Thallium",    "Lead",        "Bismuth",     "Polonium",
    "Astatine",    "Radon",       "Francium",    "Radium",      "Actinium",
    "Thorium",     "Protactinium","Uranium",     "Neptunium",   "Plutonium",
    "Americium",   "Curium",      "Berkelium",   "Californium", "Einsteinium",
    "Fermium",     "Mendelevium", "Nobelium",    "Lawrencium",  "Rutherfordium",
    "Dubnium",     "Seaborgium",  "Bohrium",     "Hassium",     "Meitnerium",
    "Darmstadtium","Roentgenium", "Copernicium", "Nihonium",    "Flerovium",
    "Moscovium",   "Livermorium", "Tennessine",  "Oganesson",
];

/// Natural abundances: (Z, A, fraction).
/// Data from IUPAC 2016 for isotopes commonly encountered in neutron resonance imaging.
/// This is not exhaustive — covers elements most relevant to VENUS experiments.
#[rustfmt::skip]
static NATURAL_ABUNDANCES: &[(u32, u32, f64)] = &[
    // Hydrogen
    (1, 1, 0.999885), (1, 2, 0.000115),
    // Lithium
    (3, 6, 0.0759), (3, 7, 0.9241),
    // Boron
    (5, 10, 0.199), (5, 11, 0.801),
    // Carbon
    (6, 12, 0.9893), (6, 13, 0.0107),
    // Nitrogen
    (7, 14, 0.99636), (7, 15, 0.00364),
    // Oxygen
    (8, 16, 0.99757), (8, 17, 0.00038), (8, 18, 0.00205),
    // Aluminum
    (13, 27, 1.0),
    // Silicon
    (14, 28, 0.92223), (14, 29, 0.04685), (14, 30, 0.03092),
    // Iron
    (26, 54, 0.05845), (26, 56, 0.91754), (26, 57, 0.02119), (26, 58, 0.00282),
    // Nickel
    (28, 58, 0.68077), (28, 60, 0.26223), (28, 61, 0.01140), (28, 62, 0.03634), (28, 64, 0.00926),
    // Copper
    (29, 63, 0.6915), (29, 65, 0.3085),
    // Zirconium
    (40, 90, 0.5145), (40, 91, 0.1122), (40, 92, 0.1715), (40, 94, 0.1738), (40, 96, 0.0280),
    // Niobium
    (41, 93, 1.0),
    // Molybdenum
    (42, 92, 0.1477), (42, 94, 0.0923), (42, 95, 0.1590), (42, 96, 0.1668),
    (42, 97, 0.0956), (42, 98, 0.2419), (42, 100, 0.0967),
    // Silver
    (47, 107, 0.51839), (47, 109, 0.48161),
    // Cadmium
    (48, 106, 0.0125), (48, 108, 0.0089), (48, 110, 0.1249), (48, 111, 0.1280),
    (48, 112, 0.2413), (48, 113, 0.1222), (48, 114, 0.2873), (48, 116, 0.0749),
    // Indium
    (49, 113, 0.0429), (49, 115, 0.9571),
    // Tin
    (50, 112, 0.0097), (50, 114, 0.0066), (50, 115, 0.0034), (50, 116, 0.1454),
    (50, 117, 0.0768), (50, 118, 0.2422), (50, 119, 0.0859), (50, 120, 0.3258),
    (50, 122, 0.0463), (50, 124, 0.0579),
    // Hafnium
    (72, 174, 0.0016), (72, 176, 0.0526), (72, 177, 0.1860), (72, 178, 0.2728),
    (72, 179, 0.1362), (72, 180, 0.3508),
    // Tantalum
    (73, 180, 0.0001201), (73, 181, 0.9998799),
    // Tungsten
    (74, 180, 0.0012), (74, 182, 0.2650), (74, 183, 0.1431), (74, 184, 0.3064), (74, 186, 0.2843),
    // Gold
    (79, 197, 1.0),
    // Lead
    (82, 204, 0.014), (82, 206, 0.241), (82, 207, 0.221), (82, 208, 0.524),
    // Thorium
    (90, 232, 1.0),
    // Uranium
    (92, 234, 0.000054), (92, 235, 0.007204), (92, 238, 0.992742),
    // Plutonium (no natural abundance, but commonly encountered — use typical reactor values)
    (94, 239, 1.0),
];

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
        assert_eq!(u238.z, 92);
        assert_eq!(u238.a, 238);

        let fe56 = parse_isotope_str("Fe-56").unwrap();
        assert_eq!(fe56.z, 26);
        assert_eq!(fe56.a, 56);

        assert!(parse_isotope_str("invalid").is_none());
    }

    #[test]
    fn test_za_roundtrip() {
        let iso = Isotope::new(92, 238);
        let za = za_from_isotope(&iso);
        assert_eq!(za, 92238);
        let back = isotope_from_za(za);
        assert_eq!(back, iso);
    }

    #[test]
    fn test_natural_abundance() {
        let u238 = Isotope::new(92, 238);
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
        let iso = Isotope::new(92, 238);
        assert_eq!(isotope_to_string(&iso), "U-238");
    }
}
