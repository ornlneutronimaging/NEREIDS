// Allow deprecated types within this crate — the old SolverChoice/SparseConfig/SparseResult
// API is implemented here and used by the old entry points that are themselves deprecated.
// External consumers will see the deprecation warnings when they use these types.
#![allow(deprecated)]

//! # nereids-pipeline
//!
//! End-to-end orchestration for neutron resonance imaging analysis.
//!
//! This crate ties together all NEREIDS components into a complete pipeline:
//! data loading → normalization → forward model → fitting → spatial mapping.
//!
//! ## Modules
//! - [`pipeline`] — Single-spectrum analysis pipeline (fit_spectrum)
//! - [`spatial`] — Per-pixel parallel mapping with rayon, ROI fitting
//! - [`sparse`] — TRINIDI-inspired two-stage reconstruction for low-count data
//! - [`detectability`] — Trace-detectability analysis (pre-experiment SNR check)

pub mod calibration;
pub mod detectability;
pub mod error;
/// Test utility: synthetic noise generation for integration tests.
pub(crate) mod noise;
pub mod pipeline;
pub mod regularization;
pub mod sparse;
pub mod spatial;

#[cfg(test)]
mod test_helpers;
