//! # nereids-physics
//!
//! Cross-section calculation, Doppler/resolution broadening, and transmission
//! forward model for neutron resonance imaging.
//!
//! ## Modules
//! - [`auxiliary_grid`] — Auxiliary energy-grid construction for resolution broadening
//! - [`channel`] — Wave number, ρ parameter, statistical weight calculations
//! - [`coulomb`] — Coulomb wave functions (Steed's CF1+CF2) for charged-particle channels
//! - [`doppler`] — Free Gas Model Doppler broadening
//! - [`penetrability`] — Hard-sphere penetrability, shift, and phase shift functions
//! - [`reich_moore`] — Reich-Moore R-matrix cross-section formalism
//! - [`resolution`] — Instrument resolution broadening (Gaussian convolution)
//! - [`rmatrix_limited`] — R-Matrix Limited (LRF=7) multi-channel formalism
//! - [`slbw`] — Breit-Wigner formalisms, single- and multi-level (LRF=1/2)
//! - [`surrogate`] — Forward-model surrogates for multi-isotope accelerated fits
//! - [`transmission`] — Beer-Lambert transmission forward model
//!
//! ## SAMMY Reference
//! - Cross-sections: `rml/` (Reich-Moore), `mlb/` (SLBW/MLBW), manual Sec. II
//! - Penetrability: `rml/mrml07.f` (Pgh, Sinsix, Pf)
//! - Coulomb: `coulomb/mrml08.f90` (Coulfg, Steed's CF1+CF2)
//! - Doppler: `fgm/` module (Dopfgm), manual Sec. III.B.1
//! - Resolution: `convolution/` module, manual Sec. III.C
//! - Transmission: `cro/`, `xxx/` modules, manual Sec. II; transmission
//!   experiments Sec. III.E.1

pub mod auxiliary_grid;
pub mod channel;
pub mod coulomb;
pub mod doppler;
pub mod ikeda_carpenter;
pub mod penetrability;
pub mod reich_moore;
pub mod resolution;
pub mod rmatrix_limited;
pub mod slbw;
pub mod surrogate;
pub mod transmission;
