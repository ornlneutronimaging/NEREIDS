//! # nereids-fitting
//!
//! Optimization engine for fitting resonance models to measured transmission data.
//!
//! ## Modules
//! - [`parameters`] — Fit parameter types, bounds, constraints
//! - [`lm`] — Levenberg-Marquardt least-squares optimizer
//! - [`poisson`] — Poisson-likelihood optimizer for low-count data
//! - [`transmission_model`] — Transmission forward model adapter for fitting
//!
//! ## SAMMY Reference
//! - Fitting: `fit/` module, `fitAPI/`, manual Sec 4
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` for Poisson-likelihood and APGM approach

pub mod error;
pub mod lm;
pub mod parameters;
pub mod poisson;
pub mod proximal;
pub mod transmission_model;
