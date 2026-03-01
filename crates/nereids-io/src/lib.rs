//! # nereids-io
//!
//! Data I/O for VENUS beamline neutron imaging data.
//!
//! ## Modules
//! - [`error`] — Error types for I/O operations
//! - [`tiff_stack`] — Multi-frame TIFF stack loading → 3D arrays (tof, y, x)
//! - [`normalization`] — Raw + open beam → transmission (Method 2), dead pixel detection, ROI
//! - [`tof`] — TOF bin edges → energy conversion for imaging data
//!
//! ## PLEIADES Reference
//! - `pleiades/processing/normalization_ornl.py` for Method 2 normalization
//! - `pleiades/processing/helper_ornl.py` for data loading

pub mod error;
pub mod normalization;
pub mod spectrum;
pub mod tiff_stack;
pub mod tof;
