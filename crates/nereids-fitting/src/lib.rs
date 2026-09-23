//! # nereids-fitting
//!
//! Optimization engine for fitting resonance models to measured transmission data.
//!
//! ## Modules
//! - [`active_mask`] — Active-bin masking for fit-energy-range restriction
//! - [`beam`] — The beam as a cubic spline of its logarithm in log flight time
//! - [`error`] — Error types for the fitting crate
//! - [`forward_model`] — Solver-agnostic forward model trait
//! - [`joint_poisson`] — Joint-Poisson profile binomial deviance (counts path)
//! - [`lm`] — Levenberg-Marquardt least-squares optimizer
//! - [`nelder_mead`] — Bounded Nelder-Mead polish optimizer
//! - [`parameters`] — Fit parameter types, bounds, constraints
//! - [`poisson`] — Poisson-likelihood optimizer for low-count data
//! - [`transmission_model`] — Transmission forward model adapter for fitting
//! - [`two_run`] — Expected open-beam and sample counts at calculation points
//!
//! ## SAMMY Reference
//! - Fitting: `fit/` module, `fitAPI/`, manual Sec. IV
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` for Poisson-likelihood and APGM approach

pub mod active_mask;
pub mod beam;
pub mod count_background;
pub mod error;
pub mod exact_count_model;
pub mod forward_model;
pub mod joint_poisson;
pub mod joint_resolution;
pub mod lm;
pub mod nelder_mead;
pub mod parameters;
pub mod poisson;
pub mod resolution_calib;
pub mod transmission_model;
pub mod two_run;
