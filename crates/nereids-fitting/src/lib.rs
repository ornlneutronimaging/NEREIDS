//! # nereids-fitting
//!
//! Poisson/KL optimization for fitting counts.
//!
//! ## SAMMY Reference
//! - Fitting: `fit/` module, `fitAPI/`, manual Sec. IV

pub mod error;
pub mod lm;
pub mod parameters;
pub mod poisson;
pub mod statistics;
