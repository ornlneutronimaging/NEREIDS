//! # nereids-io
//!
//! Data I/O for VENUS beamline neutron imaging data.
//!
//! ## Modules
//! - [`error`] — Error types for I/O operations
//! - [`export`] — Export spatial mapping results to TIFF, HDF5, and Markdown
//! - `nexus` — NeXus/HDF5 reading for rustpix-processed data (`hdf5` feature;
//!   not an intra-doc link so default-feature doc builds stay warning-free)
//! - [`normalization`] — Raw + open beam → transmission (Method 2), dead pixel detection, ROI
//! - `project` — Project file save/load for `.nrd.h5` archives (`hdf5` feature)
//! - [`rebin`] — Energy rebinning (coarsen the TOF/energy axis by an integer factor)
//! - [`spectrum`] — Spectrum file parser for TOF/energy bin edges or centers
//! - [`tiff_stack`] — Multi-frame TIFF stack loading → 3D arrays (tof, y, x)
//! - [`tof`] — TOF bin edges → energy conversion for imaging data
//!
//! ## PLEIADES Reference
//! - `pleiades/processing/normalization_ornl.py` for Method 2 normalization
//! - `pleiades/processing/helper_ornl.py` for data loading

pub mod error;
pub mod export;
#[cfg(feature = "hdf5")]
pub mod nexus;
pub mod normalization;
#[cfg(feature = "hdf5")]
pub mod project;
pub mod rebin;
pub mod spectrum;
pub mod tiff_stack;
pub mod tof;
