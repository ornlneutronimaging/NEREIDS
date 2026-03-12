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
//! - [`noise`] — Synthetic noise generation (Poisson, Gaussian) for testing

pub mod detectability;
pub mod error;
pub mod noise;
pub mod pipeline;
pub mod sparse;
pub mod spatial;

#[cfg(test)]
mod test_helpers;
