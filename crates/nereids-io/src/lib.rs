//! # nereids-io
//!
//! Data I/O for VENUS beamline neutron imaging data.
//!
//! ## Modules
//! - `daslogs` - DASlogs run-health summary (`hdf5` feature)
//! - [`error`] - error types for I/O operations
//! - [`export`] - export spatial mapping results to TIFF, HDF5 and Markdown
//! - `nexus` - NeXus/HDF5 reading for rustpix-processed data (`hdf5` feature)
//! - [`pixels`] - data-quality pixel masks and per-pixel spectrum extraction
//! - `project` - project file save/load for `.nrd.h5` archives (`hdf5` feature)
//! - [`rebin`] - rebinning the TOF/energy axis by an integer factor
//! - `runlog` - DASlogs transition logs to beam-state intervals (`hdf5` feature)
//! - [`spectrum`] - spectrum file parser for TOF/energy bin edges or centers
//! - [`tiff_stack`] - multi-frame TIFF stack loading to 3D arrays (tof, y, x)
//! - [`tof`] - TOF bin edges to energy conversion for imaging data

#[cfg(feature = "hdf5")]
pub mod daslogs;
pub mod error;
pub mod export;
#[cfg(feature = "hdf5")]
pub mod nexus;
pub mod pixels;
#[cfg(feature = "hdf5")]
pub mod project;
pub mod rebin;
#[cfg(feature = "hdf5")]
pub mod runlog;
pub mod spectrum;
pub mod tiff_stack;
pub mod tof;
