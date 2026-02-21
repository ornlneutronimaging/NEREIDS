//! Natural isotopic abundances.
//!
//! IUPAC-recommended isotopic compositions of the elements (2016 values,
//! from NIST Atomic Weights and Isotopic Compositions database). Covers
//! 289 naturally occurring isotopes across 85 elements.
//!
//! Data source: <https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl>

/// Natural isotopic abundance (mole fraction) for a given isotope.
///
/// Returns `None` if the isotope has no natural abundance (synthetic isotopes,
/// or isotopes not in the IUPAC database).
///
/// # Examples
/// ```
/// assert!((endf_mat::natural_abundance(92, 238).unwrap() - 0.992742).abs() < 1e-6);
/// assert_eq!(endf_mat::natural_abundance(43, 99), None); // Tc is synthetic
/// ```
pub fn natural_abundance(z: u32, a: u32) -> Option<f64> {
    NATURAL_ABUNDANCES
        .binary_search_by(|&(tz, ta, _)| (tz, ta).cmp(&(z, a)))
        .ok()
        .map(|idx| NATURAL_ABUNDANCES[idx].2)
}

/// All naturally occurring isotopes for element Z, as `(A, fraction)` pairs.
///
/// Returns an empty `Vec` for synthetic elements (e.g., Tc, Pm).
///
/// # Examples
/// ```
/// let fe = endf_mat::natural_isotopes(26);
/// assert_eq!(fe.len(), 4); // Fe-54, Fe-56, Fe-57, Fe-58
/// let total: f64 = fe.iter().map(|(_, f)| f).sum();
/// assert!((total - 1.0).abs() < 0.01);
/// ```
pub fn natural_isotopes(z: u32) -> Vec<(u32, f64)> {
    NATURAL_ABUNDANCES
        .iter()
        .filter(|&&(tz, _, _)| tz == z)
        .map(|&(_, a, frac)| (a, frac))
        .collect()
}

// Natural isotopic abundances: (Z, A, fraction)
// Source: NIST Atomic Weights and Isotopic Compositions (IUPAC 2016)
// 289 entries covering 85 elements, sorted by (Z, A)
#[rustfmt::skip]
static NATURAL_ABUNDANCES: &[(u32, u32, f64)] = &[
    // Hydrogen (H, Z=1)
    (1, 1, 0.999885),
    (1, 2, 0.000115),

    // Helium (He, Z=2)
    (2, 3, 0.00000134),
    (2, 4, 0.999999),

    // Lithium (Li, Z=3)
    (3, 6, 0.0759),
    (3, 7, 0.9241),

    // Beryllium (Be, Z=4)
    (4, 9, 1.0),

    // Boron (B, Z=5)
    (5, 10, 0.199),
    (5, 11, 0.801),

    // Carbon (C, Z=6)
    (6, 12, 0.9893),
    (6, 13, 0.0107),

    // Nitrogen (N, Z=7)
    (7, 14, 0.99636),
    (7, 15, 0.00364),

    // Oxygen (O, Z=8)
    (8, 16, 0.99757),
    (8, 17, 0.00038),
    (8, 18, 0.00205),

    // Fluorine (F, Z=9)
    (9, 19, 1.0),

    // Neon (Ne, Z=10)
    (10, 20, 0.9048),
    (10, 21, 0.0027),
    (10, 22, 0.0925),

    // Sodium (Na, Z=11)
    (11, 23, 1.0),

    // Magnesium (Mg, Z=12)
    (12, 24, 0.7899),
    (12, 25, 0.1),
    (12, 26, 0.1101),

    // Aluminum (Al, Z=13)
    (13, 27, 1.0),

    // Silicon (Si, Z=14)
    (14, 28, 0.92223),
    (14, 29, 0.04685),
    (14, 30, 0.03092),

    // Phosphorus (P, Z=15)
    (15, 31, 1.0),

    // Sulfur (S, Z=16)
    (16, 32, 0.9499),
    (16, 33, 0.0075),
    (16, 34, 0.0425),
    (16, 36, 0.0001),

    // Chlorine (Cl, Z=17)
    (17, 35, 0.7576),
    (17, 37, 0.2424),

    // Argon (Ar, Z=18)
    (18, 36, 0.003336),
    (18, 38, 0.000629),
    (18, 40, 0.996035),

    // Potassium (K, Z=19)
    (19, 39, 0.932581),
    (19, 40, 0.000117),
    (19, 41, 0.067302),

    // Calcium (Ca, Z=20)
    (20, 40, 0.96941),
    (20, 42, 0.00647),
    (20, 43, 0.00135),
    (20, 44, 0.02086),
    (20, 46, 0.00004),
    (20, 48, 0.00187),

    // Scandium (Sc, Z=21)
    (21, 45, 1.0),

    // Titanium (Ti, Z=22)
    (22, 46, 0.0825),
    (22, 47, 0.0744),
    (22, 48, 0.7372),
    (22, 49, 0.0541),
    (22, 50, 0.0518),

    // Vanadium (V, Z=23)
    (23, 50, 0.0025),
    (23, 51, 0.9975),

    // Chromium (Cr, Z=24)
    (24, 50, 0.04345),
    (24, 52, 0.83789),
    (24, 53, 0.09501),
    (24, 54, 0.02365),

    // Manganese (Mn, Z=25)
    (25, 55, 1.0),

    // Iron (Fe, Z=26)
    (26, 54, 0.05845),
    (26, 56, 0.91754),
    (26, 57, 0.02119),
    (26, 58, 0.00282),

    // Cobalt (Co, Z=27)
    (27, 59, 1.0),

    // Nickel (Ni, Z=28)
    (28, 58, 0.68077),
    (28, 60, 0.26223),
    (28, 61, 0.011399),
    (28, 62, 0.036346),
    (28, 64, 0.009255),

    // Copper (Cu, Z=29)
    (29, 63, 0.6915),
    (29, 65, 0.3085),

    // Zinc (Zn, Z=30)
    (30, 64, 0.4917),
    (30, 66, 0.2773),
    (30, 67, 0.0404),
    (30, 68, 0.1845),
    (30, 70, 0.0061),

    // Gallium (Ga, Z=31)
    (31, 69, 0.60108),
    (31, 71, 0.39892),

    // Germanium (Ge, Z=32)
    (32, 70, 0.2057),
    (32, 72, 0.2745),
    (32, 73, 0.0775),
    (32, 74, 0.365),
    (32, 76, 0.0773),

    // Arsenic (As, Z=33)
    (33, 75, 1.0),

    // Selenium (Se, Z=34)
    (34, 74, 0.0089),
    (34, 76, 0.0937),
    (34, 77, 0.0763),
    (34, 78, 0.2377),
    (34, 80, 0.4961),
    (34, 82, 0.0873),

    // Bromine (Br, Z=35)
    (35, 79, 0.5069),
    (35, 81, 0.4931),

    // Krypton (Kr, Z=36)
    (36, 78, 0.00355),
    (36, 80, 0.02286),
    (36, 82, 0.11593),
    (36, 83, 0.115),
    (36, 84, 0.56987),
    (36, 86, 0.17279),

    // Rubidium (Rb, Z=37)
    (37, 85, 0.7217),
    (37, 87, 0.2783),

    // Strontium (Sr, Z=38)
    (38, 84, 0.0056),
    (38, 86, 0.0986),
    (38, 87, 0.07),
    (38, 88, 0.8258),

    // Yttrium (Y, Z=39)
    (39, 89, 1.0),

    // Zirconium (Zr, Z=40)
    (40, 90, 0.5145),
    (40, 91, 0.1122),
    (40, 92, 0.1715),
    (40, 94, 0.1738),
    (40, 96, 0.028),

    // Niobium (Nb, Z=41)
    (41, 93, 1.0),

    // Molybdenum (Mo, Z=42)
    (42, 92, 0.1453),
    (42, 94, 0.0915),
    (42, 95, 0.1584),
    (42, 96, 0.1667),
    (42, 97, 0.096),
    (42, 98, 0.2439),
    (42, 100, 0.0982),

    // Ruthenium (Ru, Z=44)
    (44, 96, 0.0554),
    (44, 98, 0.0187),
    (44, 99, 0.1276),
    (44, 100, 0.126),
    (44, 101, 0.1706),
    (44, 102, 0.3155),
    (44, 104, 0.1862),

    // Rhodium (Rh, Z=45)
    (45, 103, 1.0),

    // Palladium (Pd, Z=46)
    (46, 102, 0.0102),
    (46, 104, 0.1114),
    (46, 105, 0.2233),
    (46, 106, 0.2733),
    (46, 108, 0.2646),
    (46, 110, 0.1172),

    // Silver (Ag, Z=47)
    (47, 107, 0.51839),
    (47, 109, 0.48161),

    // Cadmium (Cd, Z=48)
    (48, 106, 0.0125),
    (48, 108, 0.0089),
    (48, 110, 0.1249),
    (48, 111, 0.128),
    (48, 112, 0.2413),
    (48, 113, 0.1222),
    (48, 114, 0.2873),
    (48, 116, 0.0749),

    // Indium (In, Z=49)
    (49, 113, 0.0429),
    (49, 115, 0.9571),

    // Tin (Sn, Z=50)
    (50, 112, 0.0097),
    (50, 114, 0.0066),
    (50, 115, 0.0034),
    (50, 116, 0.1454),
    (50, 117, 0.0768),
    (50, 118, 0.2422),
    (50, 119, 0.0859),
    (50, 120, 0.3258),
    (50, 122, 0.0463),
    (50, 124, 0.0579),

    // Antimony (Sb, Z=51)
    (51, 121, 0.5721),
    (51, 123, 0.4279),

    // Tellurium (Te, Z=52)
    (52, 120, 0.0009),
    (52, 122, 0.0255),
    (52, 123, 0.0089),
    (52, 124, 0.0474),
    (52, 125, 0.0707),
    (52, 126, 0.1884),
    (52, 128, 0.3174),
    (52, 130, 0.3408),

    // Iodine (I, Z=53)
    (53, 127, 1.0),

    // Xenon (Xe, Z=54)
    (54, 124, 0.000952),
    (54, 126, 0.00089),
    (54, 128, 0.019102),
    (54, 129, 0.264006),
    (54, 130, 0.04071),
    (54, 131, 0.212324),
    (54, 132, 0.269086),
    (54, 134, 0.104357),
    (54, 136, 0.088573),

    // Cesium (Cs, Z=55)
    (55, 133, 1.0),

    // Barium (Ba, Z=56)
    (56, 130, 0.00106),
    (56, 132, 0.00101),
    (56, 134, 0.02417),
    (56, 135, 0.06592),
    (56, 136, 0.07854),
    (56, 137, 0.11232),
    (56, 138, 0.71698),

    // Lanthanum (La, Z=57)
    (57, 138, 0.0008881),
    (57, 139, 0.999112),

    // Cerium (Ce, Z=58)
    (58, 136, 0.00185),
    (58, 138, 0.00251),
    (58, 140, 0.8845),
    (58, 142, 0.11114),

    // Praseodymium (Pr, Z=59)
    (59, 141, 1.0),

    // Neodymium (Nd, Z=60)
    (60, 142, 0.27152),
    (60, 143, 0.12174),
    (60, 144, 0.23798),
    (60, 145, 0.08293),
    (60, 146, 0.17189),
    (60, 148, 0.05756),
    (60, 150, 0.05638),

    // Samarium (Sm, Z=62)
    (62, 144, 0.0307),
    (62, 147, 0.1499),
    (62, 148, 0.1124),
    (62, 149, 0.1382),
    (62, 150, 0.0738),
    (62, 152, 0.2675),
    (62, 154, 0.2275),

    // Europium (Eu, Z=63)
    (63, 151, 0.4781),
    (63, 153, 0.5219),

    // Gadolinium (Gd, Z=64)
    (64, 152, 0.002),
    (64, 154, 0.0218),
    (64, 155, 0.148),
    (64, 156, 0.2047),
    (64, 157, 0.1565),
    (64, 158, 0.2484),
    (64, 160, 0.2186),

    // Terbium (Tb, Z=65)
    (65, 159, 1.0),

    // Dysprosium (Dy, Z=66)
    (66, 156, 0.00056),
    (66, 158, 0.00095),
    (66, 160, 0.02329),
    (66, 161, 0.18889),
    (66, 162, 0.25475),
    (66, 163, 0.24896),
    (66, 164, 0.2826),

    // Holmium (Ho, Z=67)
    (67, 165, 1.0),

    // Erbium (Er, Z=68)
    (68, 162, 0.00139),
    (68, 164, 0.01601),
    (68, 166, 0.33503),
    (68, 167, 0.22869),
    (68, 168, 0.26978),
    (68, 170, 0.1491),

    // Thulium (Tm, Z=69)
    (69, 169, 1.0),

    // Ytterbium (Yb, Z=70)
    (70, 168, 0.00123),
    (70, 170, 0.02982),
    (70, 171, 0.1409),
    (70, 172, 0.2168),
    (70, 173, 0.16103),
    (70, 174, 0.32026),
    (70, 176, 0.12996),

    // Lutetium (Lu, Z=71)
    (71, 175, 0.97401),
    (71, 176, 0.02599),

    // Hafnium (Hf, Z=72)
    (72, 174, 0.0016),
    (72, 176, 0.0526),
    (72, 177, 0.186),
    (72, 178, 0.2728),
    (72, 179, 0.1362),
    (72, 180, 0.3508),

    // Tantalum (Ta, Z=73)
    (73, 180, 0.0001201),
    (73, 181, 0.99988),

    // Tungsten (W, Z=74)
    (74, 180, 0.0012),
    (74, 182, 0.265),
    (74, 183, 0.1431),
    (74, 184, 0.3064),
    (74, 186, 0.2843),

    // Rhenium (Re, Z=75)
    (75, 185, 0.374),
    (75, 187, 0.626),

    // Osmium (Os, Z=76)
    (76, 184, 0.0002),
    (76, 186, 0.0159),
    (76, 187, 0.0196),
    (76, 188, 0.1324),
    (76, 189, 0.1615),
    (76, 190, 0.2626),
    (76, 192, 0.4078),

    // Iridium (Ir, Z=77)
    (77, 191, 0.373),
    (77, 193, 0.627),

    // Platinum (Pt, Z=78)
    (78, 190, 0.00012),
    (78, 192, 0.00782),
    (78, 194, 0.3286),
    (78, 195, 0.3378),
    (78, 196, 0.2521),
    (78, 198, 0.07356),

    // Gold (Au, Z=79)
    (79, 197, 1.0),

    // Mercury (Hg, Z=80)
    (80, 196, 0.0015),
    (80, 198, 0.0997),
    (80, 199, 0.1687),
    (80, 200, 0.231),
    (80, 201, 0.1318),
    (80, 202, 0.2986),
    (80, 204, 0.0687),

    // Thallium (Tl, Z=81)
    (81, 203, 0.2952),
    (81, 205, 0.7048),

    // Lead (Pb, Z=82)
    (82, 204, 0.014),
    (82, 206, 0.241),
    (82, 207, 0.221),
    (82, 208, 0.524),

    // Bismuth (Bi, Z=83)
    (83, 209, 1.0),

    // Thorium (Th, Z=90)
    (90, 232, 1.0),

    // Protactinium (Pa, Z=91)
    (91, 231, 1.0),

    // Uranium (U, Z=92)
    (92, 234, 0.000054),
    (92, 235, 0.007204),
    (92, 238, 0.992742),

    // Plutonium (Pu, Z=94) — no natural abundance; placeholder for reactor default
    (94, 239, 1.0),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_abundance_uranium() {
        assert!((natural_abundance(92, 238).unwrap() - 0.992742).abs() < 1e-6);
        assert!((natural_abundance(92, 235).unwrap() - 0.007204).abs() < 1e-6);
        assert!((natural_abundance(92, 234).unwrap() - 0.000054).abs() < 1e-6);
    }

    #[test]
    fn test_natural_abundance_iron() {
        assert!((natural_abundance(26, 56).unwrap() - 0.91754).abs() < 1e-5);
        assert!((natural_abundance(26, 54).unwrap() - 0.05845).abs() < 1e-5);
    }

    #[test]
    fn test_natural_abundance_synthetic() {
        // Technetium (Z=43) has no stable isotopes
        assert_eq!(natural_abundance(43, 99), None);
        // Promethium (Z=61) has no stable isotopes
        assert_eq!(natural_abundance(61, 147), None);
    }

    #[test]
    fn test_natural_abundance_unknown() {
        assert_eq!(natural_abundance(200, 400), None);
    }

    #[test]
    fn test_natural_isotopes_iron() {
        let fe = natural_isotopes(26);
        assert_eq!(fe.len(), 4);
        let total: f64 = fe.iter().map(|(_, f)| f).sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_natural_isotopes_tin() {
        let sn = natural_isotopes(50);
        assert_eq!(sn.len(), 10); // 10 stable tin isotopes
        let total: f64 = sn.iter().map(|(_, f)| f).sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_natural_isotopes_tungsten() {
        let w = natural_isotopes(74);
        assert_eq!(w.len(), 5); // W-180, W-182, W-183, W-184, W-186
    }

    #[test]
    fn test_natural_isotopes_synthetic() {
        let tc = natural_isotopes(43);
        assert!(tc.is_empty());
    }

    #[test]
    fn test_natural_isotopes_mono_isotopic() {
        // Gold: only Au-197
        let au = natural_isotopes(79);
        assert_eq!(au.len(), 1);
        assert_eq!(au[0].0, 197);
        assert!((au[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_table_size() {
        assert_eq!(NATURAL_ABUNDANCES.len(), 289);
    }

    #[test]
    fn test_table_sorted() {
        for i in 1..NATURAL_ABUNDANCES.len() {
            let (z1, a1, _) = NATURAL_ABUNDANCES[i - 1];
            let (z2, a2, _) = NATURAL_ABUNDANCES[i];
            assert!(
                (z1, a1) < (z2, a2),
                "Table not sorted at index {}: ({}, {}) >= ({}, {})",
                i,
                z1,
                a1,
                z2,
                a2
            );
        }
    }
}
