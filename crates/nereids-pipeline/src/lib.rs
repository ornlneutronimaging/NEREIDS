//! # nereids-pipeline
//!
//! End-to-end orchestration for neutron resonance imaging analysis.
//!
//! This crate ties together all NEREIDS components into a complete pipeline:
//! data loading → normalization → forward model → fitting → spatial mapping.
//!
//! ## Modules
//! - [`pipeline`] — Single-spectrum analysis pipeline (fit_spectrum_typed)
//! - [`spatial`] — Per-pixel parallel mapping with rayon (spatial_map_typed)
//! - [`detectability`] — Trace-detectability analysis (pre-experiment SNR check)

pub mod calibration;
pub mod detectability;
pub mod error;
/// Test utility: synthetic noise generation for integration tests.
pub(crate) mod noise;
pub mod pipeline;
pub mod spatial;

#[cfg(test)]
mod test_helpers;
