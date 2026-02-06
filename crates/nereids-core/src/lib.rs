//! Core data models and traits for NEREIDS.
//!
//! This crate defines the shared types, traits, and error hierarchy used
//! throughout the NEREIDS workspace. It has minimal dependencies (only `thiserror`)
//! to keep it lightweight and widely usable.

pub mod background;
pub mod constants;
pub mod energy;
pub mod error;
pub mod forward_model;
pub mod nuclear;
pub mod optimizer;
pub mod resolution;
pub mod transmission;

// Re-export key types at crate root for convenience.
pub use background::{BackgroundModel, ConstantBackground, PolynomialBackground};
pub use constants::*;
pub use energy::EnergyGrid;
pub use error::{Error, FitError, IoError, PhysicsError};
pub use forward_model::{ForwardModel, ForwardModelConfig};
pub use nuclear::{Channel, IsotopeParams, RMatrixParameters, Resonance, SpinGroup};
pub use optimizer::{FitConfig, FitResult, Optimizer};
pub use resolution::ResolutionFunction;
pub use transmission::{PixelData, TransmissionSpectrum};
