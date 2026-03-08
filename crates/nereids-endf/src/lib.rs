//! # nereids-endf
//!
//! ENDF file retrieval, local caching, and resonance parameter parsing.
//!
//! ## Modules
//! - [`retrieval`] — Download ENDF files from IAEA, with local caching
//! - [`parser`] — Parse ENDF-6 File 2 (resonance parameters)
//! - [`resonance`] — Data structures for resonance parameters
//!
//! ## SAMMY Reference
//! - SAMMY manual Section 9 (ENDF-6 format)
//! - SAMMY source: `endf/` module, `SammyRMatrixParameters.h`
//!
//! ## PLEIADES Reference
//! - `pleiades/nuclear/manager.py` for URL patterns and caching strategy

pub mod parser;
pub mod resonance;
pub mod retrieval;
pub mod sammy;
