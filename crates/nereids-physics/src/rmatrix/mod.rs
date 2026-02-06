//! R-matrix cross section computation.

pub mod cross_section;
pub mod penetration;
pub mod reich_moore;

// Re-export key types and functions for convenience
pub use cross_section::compute_0k_cross_sections;
pub use penetration::{hard_sphere_phase, penetration_shift_factors};
pub use reich_moore::{reich_moore_cross_sections, CrossSections, RMatrixConfig};
