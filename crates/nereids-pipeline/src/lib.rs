//! # nereids-pipeline
//!
//! End-to-end orchestration for neutron resonance imaging analysis.
//!
//! This crate ties together all NEREIDS components into a complete pipeline:
//! data loading → normalization → forward model → fitting → spatial mapping.
//!
//! ## Modules
//! - [`beam`] — The beam before the sample, per µs of flight time
//! - [`calibration`] — Energy calibration for TOF neutron instruments (t0 + flight-path scale)
//! - [`counts_fit`] — Densities, temperature, normalization and background fitted to the
//!   open-beam and sample counts (fit_counts)
//! - [`detectability`] — Trace-detectability analysis (pre-experiment SNR check)
//! - [`error`] — Pipeline error types
//! - [`open_beam`] — The beam fitted to the open-beam counts (fit_open_beam)
//! - [`pipeline`] — Single-spectrum analysis pipeline (fit_spectrum_typed)
//! - [`spatial`] — Per-pixel parallel mapping with rayon (spatial_map_typed)

pub mod beam;
pub mod calibration;
pub mod counts_fit;
pub mod detectability;
pub mod error;
pub mod joint_fit;
pub mod open_beam;
pub mod pipeline;
/// Expected detector counts from the counts measurement equation, for tests.
#[cfg(feature = "test-support")]
pub mod reference;
pub mod spatial;
