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
//! - [`transmission`] — Beer-Lambert transmission forward model
//! - [`urr`] — Unresolved Resonance Region (LRU=2) Hauser-Feshbach cross-sections
//!
//! ## SAMMY Reference
//! - Cross-sections: `rml/` (Reich-Moore), `mlb/` (SLBW/MLBW), manual Sec 2
//! - Penetrability: `rml/mrml07.f` (Pgh, Sinsix, Pf)
//! - Coulomb: `coulomb/mrml08.f90` (Coulfg, Steed's CF1+CF2)
//! - Doppler: `dop/` module, manual Sec 3.1
//! - Resolution: `convolution/` module, manual Sec 3.2
//! - Transmission: `cro/`, `xxx/` modules, manual Sec 2, Sec 5

pub mod auxiliary_grid;
pub mod channel;
pub mod coulomb;
pub mod doppler;
pub mod penetrability;
pub mod reich_moore;
pub mod resolution;
pub mod rmatrix_limited;
pub mod slbw;
pub mod surrogate;
pub mod transmission;
pub mod urr;
