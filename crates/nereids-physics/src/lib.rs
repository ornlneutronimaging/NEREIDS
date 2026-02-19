//! # nereids-physics
//!
//! Cross-section calculation, Doppler/resolution broadening, and transmission
//! forward model for neutron resonance imaging.
//!
//! ## Modules
//! - [`penetrability`] — Hard-sphere penetrability, shift, and phase shift functions
//! - [`channel`] — Wave number, ρ parameter, statistical weight calculations
//! - [`reich_moore`] — Reich-Moore R-matrix cross-section formalism
//! - [`slbw`] — Single-Level Breit-Wigner formalism (validation/comparison)
//! - [`doppler`] — Free Gas Model Doppler broadening
//! - [`resolution`] — Instrument resolution broadening (Gaussian convolution)
//!
//! ## Planned Modules
//! - `transmission` — Beer-Lambert transmission forward model
//!
//! ## SAMMY Reference
//! - Cross-sections: `rml/` (Reich-Moore), `mlb/` (SLBW/MLBW), manual Sec 2
//! - Penetrability: `rml/mrml07.f` (Pgh, Sinsix, Pf)
//! - Doppler: `dop/` module, manual Sec 3.1
//! - Resolution: `convolution/` module, manual Sec 3.2

pub mod channel;
pub mod doppler;
pub mod penetrability;
pub mod reich_moore;
pub mod resolution;
pub mod slbw;
