//! Read rustpix NeXus/HDF5 files (histogram and event data).
//!
//! Supports the rustpix HDF5 schema:
//! - `/entry/histogram/counts` — up to 5D: `(time, rot_angle, y, x, tof)`
//! - `/entry/neutrons/event_id`, `event_time_offset` — event data
//! - `/entry/pixel_masks/` — dead and hot pixel masks
