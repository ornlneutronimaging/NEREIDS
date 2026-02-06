//! Data I/O for NEREIDS.
//!
//! Supports reading and writing:
//! - NeXus/HDF5 files (rustpix format, histograms and event data)
//! - Legacy TIFF stacks (TPX1 detector)
//! - Spectra text files
//! - User-defined resolution function files
//! - SAMMY .par files (for migration/interoperability)
//!
//! HDF5 and TIFF support are feature-gated to keep the default build lightweight.

#[cfg(feature = "hdf5")]
pub mod nexus;

pub mod resolution;
pub mod sammy;
pub mod spectra;

#[cfg(feature = "tiff")]
pub mod tiff_io;
