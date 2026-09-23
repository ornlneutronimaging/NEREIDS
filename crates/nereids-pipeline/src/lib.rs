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
pub mod joint_fit;
pub mod pipeline;
/// Expected detector counts from the counts measurement equation, for tests.
#[cfg(feature = "test-support")]
pub mod reference;
pub mod spatial;
/// Synthetic counts-domain measurements with known ground truth, for tests.
///
/// Gated on the `test-support` feature ALONE, not on `cfg(test)` as well.
/// `cfg(test)` does not enable the feature, so the unit-test target would
/// compile this module while its optional `rand_chacha` dependency stayed
/// off — it happens to build today only because the self dev-dependency
/// unifies the feature across the package, which is not something to rely
/// on. Never built into a release.
#[cfg(feature = "test-support")]
pub mod synthetic;
