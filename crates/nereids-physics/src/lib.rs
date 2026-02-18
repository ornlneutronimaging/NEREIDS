//! # nereids-physics
//!
//! Cross-section calculation, Doppler/resolution broadening, and transmission
//! forward model for neutron resonance imaging.
//!
//! ## Modules (planned)
//! - `cross_section` — Reich-Moore R-matrix and SLBW formalisms
//! - `broadening` — Doppler (FGM, Leal-Hwang) and resolution broadening
//! - `transmission` — Beer-Lambert transmission forward model
//! - `coulomb` — Penetrability, shift, and phase shift functions
//!
//! ## SAMMY Reference
//! - Cross-sections: `rml/` (Reich-Moore), `mlb/` (SLBW/MLBW), manual Sec 2
//! - Doppler: `dop/` module, manual Sec 3.1
//! - Resolution: `convolution/` module, manual Sec 3.2
//! - Coulomb: `coulomb/` module
