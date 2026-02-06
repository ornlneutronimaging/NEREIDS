//! Parallel pixel dispatch for radiograph-scale fitting.
//!
//! Each pixel is an independent fitting problem. This module uses rayon
//! to distribute fits across CPU cores via work-stealing.

use nereids_core::{FitConfig, FitError, FitResult, Optimizer, PixelData};
use rayon::prelude::*;

/// Fit all pixels in a radiograph in parallel.
pub fn fit_radiograph(
    pixels: &[PixelData],
    optimizer: &dyn Optimizer,
    config: &FitConfig,
    params: &nereids_core::RMatrixParameters,
) -> Vec<Result<FitResult, FitError>> {
    pixels
        .par_iter()
        .map(|pixel| optimizer.fit(&pixel.observed, params, config))
        .collect()
}
