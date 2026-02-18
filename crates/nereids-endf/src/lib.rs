//! # nereids-endf
//!
//! ENDF file retrieval, local caching, and resonance parameter parsing.
//!
//! This crate handles:
//! - Downloading ENDF files from IAEA (HTTP/FTP)
//! - Local file caching for offline use
//! - Parsing ENDF-6 File 2 (resonance parameters)
//! - Extracting resolved resonance region (RRR) data
//!
//! ## SAMMY Reference
//! - SAMMY manual Section 9 (ENDF-6 format)
//! - SAMMY source: `endf/` module, `SammyRMatrixParameters.h`
//!
//! ## PLEIADES Reference
//! - `pleiades/nuclear/manager.py` for URL patterns and caching strategy
