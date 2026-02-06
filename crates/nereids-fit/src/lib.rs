//! Optimizer and pixel-dispatch fitting for NEREIDS.
//!
//! This crate provides:
//! - Bayes/GLS optimizer (generalized least squares with Bayesian priors)
//! - Parallel pixel dispatch via rayon for radiograph-scale fitting

pub mod bayes_gls;
pub mod pixel_dispatch;
