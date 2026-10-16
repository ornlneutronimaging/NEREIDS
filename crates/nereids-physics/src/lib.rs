//! # nereids-physics
//!
//! Cross-section calculation, Doppler/resolution broadening, and the
//! Beer-Lambert transmission forward model for neutron resonance imaging.
//!
//! ## Modules
//! - [`auxiliary_grid`] - auxiliary energy grid for resolution broadening
//! - [`channel`] - wave number, rho parameter, statistical weights
//! - [`continuous_doppler`] - the free-gas kernel integrated over the resonance equation
//! - [`doppler`] - free-gas-model Doppler broadening
//! - [`forward_scatter`] - neutrons scattered once in a slab that leave through its back face
//! - [`penetrability`] - hard-sphere penetrability, shift and phase shift
//! - [`reich_moore`] - Reich-Moore R-matrix cross-section formalism
//! - [`resolution`] - instrument resolution broadening
//! - [`slbw`] - single- and multi-level Breit-Wigner (LRF=1/2)
//! - [`thickness`] - transmission of a sample whose areal density varies
//! - [`transmission`] - Beer-Lambert transmission forward model
//!
//! ## SAMMY Reference
//! - Cross-sections: `rml/` (Reich-Moore), `mlb/` (SLBW/MLBW), manual Sec. II
//! - Penetrability: `rml/mrml07.f` (Pgh, Sinsix, Pf)
//! - Doppler: `fgm/` module (Dopfgm), manual Sec. III.B.1
//! - Resolution: `convolution/` module, manual Sec. III.C
//! - Transmission: `cro/`, `xxx/` modules, manual Sec. II

pub mod auxiliary_grid;
pub mod channel;
pub mod continuous_doppler;
pub mod doppler;
pub mod flight_time_grid;
pub mod forward_scatter;
pub mod ikeda_carpenter;
pub mod penetrability;
pub mod reich_moore;
pub mod resolution;
pub mod slbw;
pub mod thickness;
pub mod transmission;
