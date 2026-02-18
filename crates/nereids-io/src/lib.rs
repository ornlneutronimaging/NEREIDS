//! # nereids-io
//!
//! Data I/O for VENUS beamline neutron imaging data.
//!
//! ## Modules (planned)
//! - `tiff` — Multi-frame TIFF stack loading → 3D arrays (tof, y, x)
//! - `normalization` — Raw + open beam → transmission (Method 2)
//! - `roi` — Region of interest selection and masking
//!
//! ## PLEIADES Reference
//! - `pleiades/processing/normalization_ornl.py` for Method 2 normalization
//! - `pleiades/processing/helper_ornl.py` for data loading
