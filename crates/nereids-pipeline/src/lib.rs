//! # nereids-pipeline
//!
//! End-to-end orchestration for neutron resonance imaging analysis.
//!
//! This crate ties together all NEREIDS components into a complete pipeline:
//! data loading → normalization → forward model → fitting → spatial mapping.
//!
//! ## Modules (planned)
//! - `pipeline` — Single-spectrum analysis pipeline
//! - `spatial` — Per-pixel/voxel parallel mapping (rayon)
//! - `sparse` — TRINIDI-inspired two-stage reconstruction for low-count data
