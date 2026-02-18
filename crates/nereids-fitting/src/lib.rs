//! # nereids-fitting
//!
//! Optimization engine for fitting resonance models to measured transmission data.
//!
//! ## Modules (planned)
//! - `lm` — Levenberg-Marquardt least-squares (Phase 6)
//! - `poisson` — Poisson-likelihood BFGS/L-BFGS optimizer (Phase 8)
//! - `parameters` — Fit parameter types, constraints, bounds
//! - `uncertainty` — Covariance matrix and uncertainty quantification
//!
//! ## SAMMY Reference
//! - Fitting: `fit/` module, `fitAPI/`, manual Sec 4
//!
//! ## TRINIDI Reference
//! - `trinidi/reconstruct.py` for Poisson-likelihood and APGM approach
