//! Element symbol and name lookup by atomic number.
//!
//! Complete periodic table data for Z=0 (neutron) through Z=118 (Oganesson).

/// Element symbol from atomic number Z.
///
/// Returns `Some("H")` for Z=1, `Some("U")` for Z=92, etc.
/// Returns `Some("n")` for Z=0 (neutron).
/// Returns `None` for Z > 118.
pub fn element_symbol(z: u32) -> Option<&'static str> {
    ELEMENT_SYMBOLS.get(z as usize).copied()
}

/// Element name from atomic number Z.
///
/// Returns `Some("Hydrogen")` for Z=1, `Some("Uranium")` for Z=92, etc.
/// Returns `None` for Z > 118.
pub fn element_name(z: u32) -> Option<&'static str> {
    ELEMENT_NAMES.get(z as usize).copied()
}

/// Atomic number Z from element symbol (case-insensitive).
///
/// Accepts "U", "u", "Fe", "fe", etc.
pub fn symbol_to_z(symbol: &str) -> Option<u32> {
    // Neutron "n" is exact-match only to avoid shadowing Nitrogen "N"
    if symbol == "n" {
        return Some(0);
    }
    let lower = symbol.to_lowercase();
    for (z, &sym) in ELEMENT_SYMBOLS.iter().enumerate().skip(1) {
        if sym.to_lowercase() == lower {
            return Some(z as u32);
        }
    }
    None
}

// Element symbols indexed by Z. Z=0 is the neutron.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_symbol() {
        assert_eq!(element_symbol(0), Some("n"));
        assert_eq!(element_symbol(1), Some("H"));
        assert_eq!(element_symbol(26), Some("Fe"));
        assert_eq!(element_symbol(92), Some("U"));
        assert_eq!(element_symbol(118), Some("Og"));
        assert_eq!(element_symbol(119), None);
    }

    #[test]
    fn test_element_name() {
        assert_eq!(element_name(1), Some("Hydrogen"));
        assert_eq!(element_name(92), Some("Uranium"));
        assert_eq!(element_name(26), Some("Iron"));
        assert_eq!(element_name(119), None);
    }

    #[test]
    fn test_symbol_to_z() {
        assert_eq!(symbol_to_z("H"), Some(1));
        assert_eq!(symbol_to_z("Fe"), Some(26));
        assert_eq!(symbol_to_z("U"), Some(92));
        assert_eq!(symbol_to_z("fe"), Some(26)); // case-insensitive
        assert_eq!(symbol_to_z("Xx"), None);
    }

    #[test]
    fn test_symbol_roundtrip() {
        for z in 1..=118 {
            let sym = element_symbol(z).unwrap();
            assert_eq!(symbol_to_z(sym), Some(z));
        }
    }

    #[test]
    fn test_table_sizes() {
        assert_eq!(ELEMENT_SYMBOLS.len(), 119); // Z=0..118
        assert_eq!(ELEMENT_NAMES.len(), 119);
    }
}
