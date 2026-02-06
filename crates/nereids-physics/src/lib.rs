//! Physics kernels for NEREIDS.
//!
//! This crate implements the neutron resonance forward model pipeline:
//! 1. 0K cross sections via Reich-Moore R-matrix formalism
//! 2. Doppler broadening (free-gas model)
//! 3. Beer-Lambert transmission
//! 4. Self-shielding corrections
//! 5. Resolution convolution
//! 6. Normalization and background
//!
//! This crate is pure computation — no I/O, fitting, or ENDF dependencies.

pub mod broadening;
pub mod pipeline;
pub mod resolution;
pub mod rmatrix;
pub mod transmission;
