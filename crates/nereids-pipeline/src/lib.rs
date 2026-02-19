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

pub mod pipeline;
pub mod spatial;
