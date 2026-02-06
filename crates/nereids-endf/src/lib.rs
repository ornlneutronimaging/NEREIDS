//! ENDF resonance parameter parsing, retrieval, and caching.
//!
//! This crate provides:
//! - ENDF-6 File 2 resonance parameter parsing
//! - HTTP retrieval from ENDF libraries (ENDF/B-VIII, JEFF, JENDL)
//! - Local filesystem cache at `~/.nereids/endf/`
//! - Conversion from ENDF records to `nereids-core` types

pub mod cache;
pub mod convert;
pub mod parser;
pub mod retriever;

/// Supported ENDF libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndfLibrary {
    /// ENDF/B-VIII.0
    EndfB8,
    /// JEFF-3.3
    Jeff33,
    /// JENDL-5
    Jendl5,
}
