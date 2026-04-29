//! ENDF material data: element symbols, MAT numbers, and natural abundances.
//!
//! A standalone, zero-dependency crate providing lookup tables for nuclear data:
//!
//! - **Element data**: Symbols and names for Z=0 (neutron) through Z=118 (Oganesson)
//! - **MAT numbers**: ENDF material identifiers for 535 ground-state isotopes
//! - **Natural abundances**: IUPAC 2016 isotopic compositions for 289 isotopes
//! - **ZA utilities**: ENDF ZA encoding/decoding (ZA = Z×1000 + A)
//!
//! All data is compiled into static arrays — no file I/O, no external dependencies.
//!
//! # Examples
//!
//! ```
//! // Element lookup
//! assert_eq!(endf_mat::element_symbol(92), Some("U"));
//! assert_eq!(endf_mat::symbol_to_z("Fe"), Some(26));
//!
//! // MAT number lookup
//! assert_eq!(endf_mat::mat_number(92, 235), Some(9228));
//! assert_eq!(endf_mat::isotope_from_mat(9228), Some((92, 235)));
//!
//! // Natural abundances
//! let u238 = endf_mat::natural_abundance(92, 238).unwrap();
//! assert!((u238 - 0.992742).abs() < 1e-6);
//!
//! // ZA encoding
//! assert_eq!(endf_mat::za(92, 238), 92238);
//! assert_eq!(endf_mat::z_from_za(92238), 92);
//! ```

mod abundances;
mod elements;
mod mat;
mod za;

// Element data
pub use elements::{element_name, element_symbol, symbol_to_z};

// MAT numbers
pub use mat::{
    has_endf_evaluation, has_endf_evaluation_cendl, has_endf_evaluation_tendl, isotope_from_mat,
    known_isotopes, known_isotopes_cendl, known_isotopes_tendl, mat_number, mat_number_cendl,
    mat_number_tendl,
};

// Natural abundances
pub use abundances::{natural_abundance, natural_isotopes};

// ZA utilities
pub use za::{a_from_za, z_from_za, za};
