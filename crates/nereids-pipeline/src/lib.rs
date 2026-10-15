//! # nereids-pipeline
//!
//! Counts-domain fitting: densities, temperatures and backgrounds of one or
//! more regions, fitted to raw open-beam and sample counts.
//!
//! ## Modules
//! - [`beam`] - the beam before the sample, per µs of flight time
//! - [`counts_fit`] - one region or more fitted to raw counts (`fit_counts`)
//! - [`counts_map`] - patches of pixels fitted as one counts fit (`fit_map`)
//! - [`error`] - pipeline error types
//! - [`open_beam`] - the beam fitted to the open-beam counts (`fit_open_beam`)
//! - [`pulse_calibration`] - a pulse calibrated on a foil, as a prior

pub mod beam;
pub mod counts_fit;
pub mod counts_map;
pub mod error;
pub mod open_beam;
pub mod pulse_calibration;
/// Expected detector counts from the counts measurement equation, for tests.
#[cfg(feature = "test-support")]
pub mod reference;
