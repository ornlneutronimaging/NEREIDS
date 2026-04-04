//! Transmission normalization from raw neutron counts.
//!
//! Converts raw sample and open-beam (OB) neutron counts into a transmission
//! spectrum, following the ORNL Method 2 approach used in PLEIADES.
//!
//! ## Method 2 Normalization
//!
//! For each TOF bin and pixel:
//!
//!   T[tof, y, x] = (C_sample / C_ob) × (PC_ob / PC_sample)
//!
//! where:
//! - C_sample = raw sample counts (dark-current subtracted)
//! - C_ob = open-beam counts (dark-current subtracted)
//! - PC_sample = proton charge for sample run
//! - PC_ob = proton charge for open-beam run
//!
//! The proton charge ratio corrects for different beam exposures.
//!
//! ## Uncertainty
//!
//! Assuming Poisson counting statistics:
//!
//!   σ_T / T = √(1/C_sample + 1/C_ob)
//!
//! ## PLEIADES Reference
//! - `processing/normalization_ornl.py` — Method 2 implementation

use ndarray::{Array1, Array3, Axis};

use crate::error::IoError;

/// Parameters for transmission normalization.
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    /// Proton charge for the sample measurement.
    pub proton_charge_sample: f64,
    /// Proton charge for the open-beam measurement.
    pub proton_charge_ob: f64,
}

/// Result of normalization: transmission and its uncertainty.
#[derive(Debug)]
pub struct NormalizedData {
    /// Transmission values, shape (n_tof, height, width).
    pub transmission: Array3<f64>,
    /// Uncertainty on transmission, shape (n_tof, height, width).
    pub uncertainty: Array3<f64>,
}

/// Normalize raw data to transmission using Method 2.
///
/// T = (C_sample / C_ob) × (PC_ob / PC_sample)
///
/// # Arguments
/// * `sample` — Raw sample counts, shape (n_tof, height, width).
/// * `open_beam` — Open-beam counts, shape (n_tof, height, width).
/// * `params` — Normalization parameters (proton charges).
/// * `dark_current` — Optional dark-current image to subtract, shape (height, width).
///   If provided, it is subtracted from each TOF frame of both sample and OB.
///
/// # Returns
/// Normalized transmission and uncertainty arrays.
pub fn normalize(
    sample: &Array3<f64>,
    open_beam: &Array3<f64>,
    params: &NormalizationParams,
    dark_current: Option<&ndarray::Array2<f64>>,
) -> Result<NormalizedData, IoError> {
    if sample.shape() != open_beam.shape() {
        return Err(IoError::ShapeMismatch(format!(
            "Sample shape {:?} != open-beam shape {:?}",
            sample.shape(),
            open_beam.shape()
        )));
    }

    if !(params.proton_charge_sample > 0.0
        && params.proton_charge_sample.is_finite()
        && params.proton_charge_ob > 0.0
        && params.proton_charge_ob.is_finite())
    {
        return Err(IoError::InvalidParameter(
            "Proton charges must be finite and positive".into(),
        ));
    }

    if let Some(dc) = dark_current {
        let dc_shape = dc.shape();
        let s_shape = sample.shape();
        if dc_shape[0] != s_shape[1] || dc_shape[1] != s_shape[2] {
            return Err(IoError::ShapeMismatch(format!(
                "dark_current shape {:?} != spatial dimensions ({}, {})",
                dc_shape, s_shape[1], s_shape[2],
            )));
        }
    }

    let shape = sample.shape();
    let (n_tof, height, width) = (shape[0], shape[1], shape[2]);

    let pc_ratio = params.proton_charge_ob / params.proton_charge_sample;

    let mut transmission = Array3::<f64>::zeros((n_tof, height, width));
    let mut uncertainty = Array3::<f64>::zeros((n_tof, height, width));

    for t in 0..n_tof {
        for y in 0..height {
            for x in 0..width {
                let dc = dark_current.map_or(0.0, |dc| dc[[y, x]]);
                let c_s = (sample[[t, y, x]] - dc).max(0.0);
                let c_o = (open_beam[[t, y, x]] - dc).max(0.0);

                if c_o > 0.0 {
                    let t_val = (c_s / c_o) * pc_ratio;
                    transmission[[t, y, x]] = t_val;

                    // Poisson uncertainty via absolute error propagation.
                    //
                    // σ_T = pc_ratio / c_o * √(c_s_eff + c_s² / c_o)
                    //
                    // where c_s_eff is the Bayesian floor (Jeffreys prior,
                    // 0.5 counts) when c_s == 0.  This formula follows from
                    // propagating Var(c_s)=c_s_eff and Var(c_o)=c_o through
                    // T = (c_s / c_o) * pc_ratio.
                    //
                    // Unlike the relative-error form σ_T = T * √(1/c_s + 1/c_o),
                    // this absolute form produces σ > 0 even when c_s == 0 (T == 0),
                    // ensuring downstream weighted fits never see zero uncertainty.
                    //
                    // NOTE: c_o is always > 0 here (we are inside the if branch),
                    // so the old `c_o_eff` dead-code branch is removed.
                    let c_s_eff = if c_s > 0.0 { c_s } else { 0.5 };
                    let abs_var_t = (pc_ratio / c_o).powi(2) * (c_s_eff + c_s * c_s / c_o);
                    uncertainty[[t, y, x]] = abs_var_t.sqrt();
                } else {
                    // No open-beam counts: mark as invalid
                    transmission[[t, y, x]] = 0.0;
                    uncertainty[[t, y, x]] = f64::INFINITY;
                }
            }
        }
    }

    Ok(NormalizedData {
        transmission,
        uncertainty,
    })
}

/// Extract a single spectrum (all TOF bins) from a pixel in the 3D array.
///
/// # Arguments
/// * `data` — 3D array with shape (n_tof, height, width).
/// * `y` — Pixel row.
/// * `x` — Pixel column.
///
/// # Returns
/// 1D array of length n_tof.
pub fn extract_spectrum(data: &Array3<f64>, y: usize, x: usize) -> Array1<f64> {
    data.slice(ndarray::s![.., y, x]).to_owned()
}

/// Average spectra over a rectangular region of interest.
///
/// # Arguments
/// * `data` — 3D array with shape (n_tof, height, width).
/// * `y_range` — Row range (start..end).
/// * `x_range` — Column range (start..end).
///
/// # Errors
/// Returns `IoError::InvalidParameter` if the ROI is empty or exceeds the
/// spatial dimensions of `data`.
///
/// # Returns
/// Averaged 1D spectrum of length n_tof.
pub fn average_roi(
    data: &Array3<f64>,
    y_range: std::ops::Range<usize>,
    x_range: std::ops::Range<usize>,
) -> Result<Array1<f64>, IoError> {
    if y_range.is_empty() || x_range.is_empty() {
        return Err(IoError::InvalidParameter(
            "ROI ranges must be non-empty for average_roi".into(),
        ));
    }
    if y_range.end > data.shape()[1] || x_range.end > data.shape()[2] {
        return Err(IoError::InvalidParameter(format!(
            "ROI range ({}..{}, {}..{}) exceeds data spatial dims ({}, {})",
            y_range.start,
            y_range.end,
            x_range.start,
            x_range.end,
            data.shape()[1],
            data.shape()[2],
        )));
    }
    let roi = data.slice(ndarray::s![.., y_range, x_range]);
    // Mean over spatial dimensions (axes 1 and 2).
    // unwrap is safe here: the ROI is guaranteed non-empty by the check above.
    Ok(roi.mean_axis(Axis(2)).unwrap().mean_axis(Axis(1)).unwrap())
}

/// Detect dead pixels (zero counts across all TOF bins).
///
/// # Arguments
/// * `data` — 3D array with shape (n_tof, height, width).
///
/// # Returns
/// 2D boolean mask, shape (height, width). `true` = dead pixel.
pub fn detect_dead_pixels(data: &Array3<f64>) -> ndarray::Array2<bool> {
    let shape = data.shape();
    let (height, width) = (shape[1], shape[2]);
    let mut mask = ndarray::Array2::from_elem((height, width), false);

    for y in 0..height {
        for x in 0..width {
            let all_zero = (0..shape[0]).all(|t| data[[t, y, x]] == 0.0);
            mask[[y, x]] = all_zero;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_normalize_equal_charges() {
        // Equal proton charges, PC ratio = 1
        // C_s = 50, C_o = 100 → T = 0.5
        let sample = Array3::from_elem((1, 1, 1), 50.0);
        let ob = Array3::from_elem((1, 1, 1), 100.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None).unwrap();
        assert!((result.transmission[[0, 0, 0]] - 0.5).abs() < 1e-10);

        // Uncertainty: σ_T = T × √(1/50 + 1/100) = 0.5 × √(0.03) ≈ 0.0866
        let expected_unc = 0.5 * (1.0 / 50.0 + 1.0 / 100.0_f64).sqrt();
        assert!(
            (result.uncertainty[[0, 0, 0]] - expected_unc).abs() < 1e-10,
            "got {}, expected {}",
            result.uncertainty[[0, 0, 0]],
            expected_unc,
        );
    }

    #[test]
    fn test_normalize_proton_charge_correction() {
        // PC_sample = 2, PC_ob = 1 → ratio = 0.5
        // C_s = 100, C_o = 100 → T = 1.0 × 0.5 = 0.5
        let sample = Array3::from_elem((1, 1, 1), 100.0);
        let ob = Array3::from_elem((1, 1, 1), 100.0);
        let params = NormalizationParams {
            proton_charge_sample: 2.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None).unwrap();
        assert!((result.transmission[[0, 0, 0]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_with_dark_current() {
        // C_s_raw = 60, C_o_raw = 110, DC = 10
        // C_s = 50, C_o = 100 → T = 0.5
        let sample = Array3::from_elem((1, 1, 1), 60.0);
        let ob = Array3::from_elem((1, 1, 1), 110.0);
        let dc = Array2::from_elem((1, 1), 10.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, Some(&dc)).unwrap();
        assert!((result.transmission[[0, 0, 0]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_ob() {
        // Zero open-beam counts → T = 0, uncertainty = INF
        let sample = Array3::from_elem((1, 1, 1), 50.0);
        let ob = Array3::from_elem((1, 1, 1), 0.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None).unwrap();
        assert_eq!(result.transmission[[0, 0, 0]], 0.0);
        assert!(result.uncertainty[[0, 0, 0]].is_infinite());
    }

    #[test]
    fn test_normalize_shape_mismatch() {
        let sample = Array3::from_elem((2, 3, 4), 1.0);
        let ob = Array3::from_elem((2, 3, 5), 1.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_spectrum() {
        // 3 TOF bins, 2×2 image
        let mut data = Array3::<f64>::zeros((3, 2, 2));
        data[[0, 1, 0]] = 10.0;
        data[[1, 1, 0]] = 20.0;
        data[[2, 1, 0]] = 30.0;

        let spectrum = extract_spectrum(&data, 1, 0);
        assert_eq!(spectrum.len(), 3);
        assert_eq!(spectrum[0], 10.0);
        assert_eq!(spectrum[1], 20.0);
        assert_eq!(spectrum[2], 30.0);
    }

    #[test]
    fn test_average_roi() {
        // 2 TOF bins, 4×4 image. Set a 2×2 region to known values.
        let mut data = Array3::<f64>::zeros((2, 4, 4));
        // TOF bin 0: region [1..3, 1..3] = 100
        for y in 1..3 {
            for x in 1..3 {
                data[[0, y, x]] = 100.0;
                data[[1, y, x]] = 200.0;
            }
        }

        let avg = average_roi(&data, 1..3, 1..3).unwrap();
        assert_eq!(avg.len(), 2);
        assert!((avg[0] - 100.0).abs() < 1e-10);
        assert!((avg[1] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_sample_counts() {
        // Zero sample counts should produce finite (not NaN) uncertainty
        // thanks to the Bayesian floor of 0.5.
        let sample = Array3::from_elem((1, 1, 1), 0.0);
        let ob = Array3::from_elem((1, 1, 1), 100.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None).unwrap();
        assert_eq!(result.transmission[[0, 0, 0]], 0.0);
        assert!(
            result.uncertainty[[0, 0, 0]].is_finite(),
            "uncertainty should be finite for zero sample counts, got {}",
            result.uncertainty[[0, 0, 0]]
        );
        assert!(
            result.uncertainty[[0, 0, 0]] > 0.0,
            "uncertainty should be strictly positive for zero sample counts (Bayesian floor), got {}",
            result.uncertainty[[0, 0, 0]]
        );
    }

    #[test]
    fn test_normalize_zero_open_beam() {
        // Zero OB counts should produce infinite uncertainty (marking
        // the pixel as invalid), and the uncertainty must not be NaN.
        let sample = Array3::from_elem((1, 1, 1), 50.0);
        let ob = Array3::from_elem((1, 1, 1), 0.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None).unwrap();
        assert_eq!(result.transmission[[0, 0, 0]], 0.0);
        assert!(
            !result.uncertainty[[0, 0, 0]].is_nan(),
            "uncertainty must not be NaN for zero OB counts"
        );
        assert!(
            result.uncertainty[[0, 0, 0]].is_infinite(),
            "uncertainty should be infinite for zero OB counts"
        );
    }

    #[test]
    fn test_normalize_dark_current_shape_mismatch() {
        let sample = Array3::from_elem((2, 3, 4), 1.0);
        let ob = Array3::from_elem((2, 3, 4), 1.0);
        let dc = Array2::from_elem((2, 4), 0.0); // wrong shape
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, Some(&dc));
        assert!(
            result.is_err(),
            "should reject mismatched dark_current shape"
        );
    }

    /// Verify that σ > 0 for zero sample counts ensures finite LM weight.
    /// This is the Bayesian floor guarantee: weight = 1/σ² must not be ∞.
    #[test]
    fn test_normalize_zero_sample_produces_finite_lm_weight() {
        let sample = Array3::from_elem((5, 1, 1), 0.0);
        let ob = Array3::from_elem((5, 1, 1), 500.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None).unwrap();
        for t in 0..5 {
            let sigma = result.uncertainty[[t, 0, 0]];
            assert!(
                sigma.is_finite() && sigma > 0.0,
                "σ must be finite and positive at T=0, got {sigma}"
            );
            let weight = 1.0 / (sigma * sigma);
            assert!(
                weight.is_finite(),
                "LM weight 1/σ² must be finite at T=0, got {weight}"
            );
        }
    }

    /// Verify uncertainty at low OB counts is finite and well-behaved.
    #[test]
    fn test_normalize_low_ob_counts() {
        // OB = 2 counts: very low but nonzero
        let sample = Array3::from_elem((1, 1, 1), 1.0);
        let ob = Array3::from_elem((1, 1, 1), 2.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };

        let result = normalize(&sample, &ob, &params, None).unwrap();
        let sigma = result.uncertainty[[0, 0, 0]];
        assert!(sigma.is_finite() && sigma > 0.0, "σ = {sigma}");
        // σ should be large relative to T (very noisy at low counts)
        let t = result.transmission[[0, 0, 0]];
        assert!(
            sigma > 0.1 * t,
            "σ should be a significant fraction of T at low OB counts: σ={sigma}, T={t}"
        );
    }

    #[test]
    fn test_detect_dead_pixels() {
        let mut data = Array3::<f64>::zeros((3, 2, 2));
        // Pixel (0,0) is dead (all zeros)
        // Pixel (0,1) has a count in frame 1
        data[[1, 0, 1]] = 5.0;
        // Pixel (1,0) has counts
        data[[0, 1, 0]] = 10.0;
        // Pixel (1,1) is dead

        let mask = detect_dead_pixels(&data);
        assert!(mask[[0, 0]]); // dead
        assert!(!mask[[0, 1]]); // alive
        assert!(!mask[[1, 0]]); // alive
        assert!(mask[[1, 1]]); // dead
    }
}
