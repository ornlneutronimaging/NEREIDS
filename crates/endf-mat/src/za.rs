//! ZA encoding/decoding utilities.
//!
//! The ENDF ZA identifier encodes an isotope as Z×1000 + A, where Z is the
//! atomic number and A is the mass number.

/// Encode an isotope as ENDF ZA = Z×1000 + A.
pub fn za(z: u32, a: u32) -> u32 {
    z * 1000 + a
}

/// Extract atomic number Z from an ENDF ZA value.
pub fn z_from_za(za: u32) -> u32 {
    za / 1000
}

/// Extract mass number A from an ENDF ZA value.
pub fn a_from_za(za: u32) -> u32 {
    za % 1000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_za_roundtrip() {
        for &(z, a) in &[(92, 238), (1, 1), (26, 56), (94, 239), (0, 1)] {
            let encoded = za(z, a);
            assert_eq!(z_from_za(encoded), z);
            assert_eq!(a_from_za(encoded), a);
        }
    }

    #[test]
    fn test_za_known_values() {
        assert_eq!(za(92, 238), 92238);
        assert_eq!(za(1, 1), 1001);
        assert_eq!(za(0, 1), 1);
    }
}
