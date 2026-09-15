//! # nereids-pipeline
//!
//! End-to-end orchestration for neutron resonance imaging analysis.
//!
//! This crate ties together all NEREIDS components into a complete pipeline:
//! data loading → normalization → forward model → fitting → spatial mapping.
//!
//! ## Modules
//! - [`calibration`] — Energy calibration for TOF neutron instruments (t0 + flight-path scale)
//! - [`detectability`] — Trace-detectability analysis (pre-experiment SNR check)
//! - [`error`] — Pipeline error types
//! - [`pipeline`] — Single-spectrum analysis pipeline (fit_spectrum_typed)
//! - [`spatial`] — Per-pixel parallel mapping with rayon (spatial_map_typed)

pub mod calibration;
pub mod detectability;
pub mod error;
pub mod pipeline;
pub mod spatial;
/// Synthetic counts-domain measurements with known ground truth, for tests.
///
/// Gated on `#[cfg(any(test, feature = "test-support"))]` exactly as
/// `nereids_endf::resonance::test_support` is: visible to this crate's own
/// tests and to downstream crates that enable `test-support` in their
/// `[dev-dependencies]`. Never built into a release.
#[cfg(any(test, feature = "test-support"))]
pub mod synthetic;
