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
//! ## Pixel masks — pipeline integrity only
//!
//! The boolean masks produced by [`detect_dead_pixels`],
//! [`detect_dead_pixels_chunked`], [`detect_hot_pixels`], and
//! [`detect_bad_pixels`] exist for exactly one purpose: excluding pixels
//! whose data stream is broken in a way that would corrupt the downstream
//! pipeline.  Downstream, a mask is a hard exclude — masked pixels are never
//! fitted and appear as NaN in every result map (`nereids-pipeline`'s
//! `spatial_map_typed` skips them entirely; see
//! `crates/nereids-pipeline/src/spatial.rs`).
//!
//! The masks are **not** a data-quality or coverage filter:
//!
//! - **Low-count pixels are alive and MUST be kept.**  KL-domain fitting
//!   handles them correctly; a statistical low-count screen was measured to
//!   reject 13% of an ROI essentially at random (IPTS-37432).
//! - **Coverage / thickness inhomogeneity is a model concern** (free density
//!   per region), never a masking concern.
//! - Deadness/hotness is per-acquisition, so always union the sample and
//!   open-beam masks — [`detect_bad_pixels`] does this.
//!
//! See issue #643 for the methodology discussion.
//!
//! ## PLEIADES Reference
//! - `processing/normalization_ornl.py` — Method 2 implementation

use ndarray::{Array1, Array2, Array3, Axis, Zip};

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

    // Reject non-finite / negative raw counts up front.  These are detector
    // counts (and a dark-current estimate), so a NaN or a negative value
    // signals an upstream loader / TOF-normalisation bug.  Validating here —
    // rather than letting the per-bin `(x - dc).max(0.0)` clamp below absorb
    // it — is the whole point of this guard: `NaN.max(0.0) == 0.0` would have
    // silently turned a corrupt frame into a plausible "zero counts" bin,
    // exactly the masking the sibling `nereids_fitting::joint_poisson`
    // `validate_counts` exists to prevent.
    validate_counts(sample, "sample")?;
    validate_counts(open_beam, "open_beam")?;
    if let Some(dc) = dark_current {
        validate_counts(dc, "dark_current")?;
    }

    let shape = sample.shape();
    let (n_tof, height, width) = (shape[0], shape[1], shape[2]);

    let pc_ratio = params.proton_charge_ob / params.proton_charge_sample;

    let mut transmission = Array3::<f64>::zeros((n_tof, height, width));
    let mut uncertainty = Array3::<f64>::zeros((n_tof, height, width));

    for t in 0..n_tof {
        for y in 0..height {
            for x in 0..width {
                // Dark-current subtraction.  The DC noise contribution to
                // the uncertainty is omitted — Var(DC) is not included in
                // the error propagation below.  This is acceptable when DC
                // is small relative to signal counts (typical for VENUS MCP
                // detectors), but underestimates σ_T for very low-signal bins
                // where DC is comparable to the sample or OB counts.
                //
                // Inputs are validated finite & non-negative above, so the
                // subtraction is always finite here; the only way it can go
                // negative is the legitimate `dc > counts` low-count noise
                // case (the DC estimate overshoots the measured counts in a
                // single bin).  Floor that physical edge at 0 — this is NOT
                // masking bad input (a NaN / negative loader bug was already
                // rejected), it is the Method-2 convention for a dark-frame
                // estimate that exceeds the raw counts.
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

/// Reject a raw-counts array that contains a non-finite or negative value.
///
/// Detector counts are non-negative by construction (zero is legitimate), so a
/// NaN, ±∞, or negative entry signals an upstream loader / normalisation bug
/// that must be surfaced rather than silently clamped.  Reports the first
/// offending flat index and value.
///
/// The finite-&-non-negative invariant itself lives in
/// [`nereids_core::validation::first_non_finite_or_negative`] so that the
/// `nereids-fitting` joint-Poisson and this I/O loader enforce *identical*
/// semantics (`NaN < 0.0` is `false`, so the check pairs `is_finite()` with
/// the order comparison); this wrapper only maps the offending element onto
/// the `IoError` message wording.
fn validate_counts<D: ndarray::Dimension>(
    counts: &ndarray::ArrayBase<impl ndarray::Data<Elem = f64>, D>,
    field: &str,
) -> Result<(), IoError> {
    nereids_core::validation::first_non_finite_or_negative(counts.iter().copied()).map_err(
        |(i, v)| {
            IoError::InvalidParameter(format!(
                "{field} counts at flat index {i} must be finite and >= 0, got {v}"
            ))
        },
    )
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

/// Detect dead pixels (zero counts across all TOF bins of one stack).
///
/// Pipeline-integrity screen only — see the module-level "Pixel masks —
/// pipeline integrity only" section.  Prefer [`detect_bad_pixels`] as the
/// validating entry point; this function performs no input validation of its
/// own (backward compatibility: GUI, Python ABI, persisted masks).
///
/// Precondition: `data` has been validated finite and non-negative (e.g. by
/// [`normalize`]).  Under that invariant the exact `== 0.0` test is
/// intentional:
///
/// - Counts are validated non-negative, and `0.0 × efficiency == 0.0` holds
///   exactly in IEEE 754, so a `<= 0.0` test would be dead code.
/// - A NaN bin makes a pixel appear *alive* (`NaN == 0.0` is `false`).  This
///   is deliberate: corrupt input must be rejected upstream, never silently
///   masked (house anti-masking rule — cf. the validation rationale comment
///   in [`normalize`]).  [`detect_bad_pixels`] rejects such input up front.
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

/// Default MAD multiplier for [`detect_hot_pixels`].
///
/// The one-sided Gaussian tail at 6 robust σ is P(Z > 6) ≈ 9.9e-10, i.e.
/// ~2.6e-4 expected false flags on a full 512×512 frame (262 144 pixels) —
/// the screen essentially never rejects a statistically plausible pixel,
/// while a railed pixel sits tens of robust σ above any plausible median.
/// Real spatial structure (beam profile, sample absorption) only *inflates*
/// the MAD, making the cut more conservative, never less.
pub const HOT_PIXEL_K_MAD: f64 = 6.0;

/// Detect dead pixels across acquisition chunks (dead-in-any-chunk).
///
/// Catches *intermittent* deadness that [`detect_dead_pixels`] on the summed
/// stack cannot see: a pixel that was dead for one acquisition chunk but
/// alive in another has nonzero summed counts, yet its dead-chunk data
/// corrupts the combined spectrum.  A pixel is flagged iff it is all-zero
/// (exact `== 0.0` test, same rationale as [`detect_dead_pixels`]) in *any*
/// chunk.
///
/// False-positive control: a live pixel with expected total counts λ within
/// one chunk is all-zero in that chunk with probability P = e^(−λ) (Poisson);
/// over m chunks, P(misflag) ≤ m·e^(−λ).  Guidance: chunk the acquisition so
/// each live pixel has λ ≥ 20 expected counts per chunk — e^(−20) ≈ 2e-9,
/// i.e. ~5e-4·m expected false flags on a 512² detector.
///
/// There is deliberately **no** per-TOF-block zero-run variant.  Within one
/// TOF-summed stack, wall-clock-intermittent deadness is invisible — the
/// pixel just shows uniformly reduced counts across all TOF bins, with no
/// zeros to find.  And any within-stack zero-run cut is a statistical screen
/// on low-count pixels (a pixel at 0.01 counts/bin normally has ~100-bin
/// zero runs) — exactly the banned failure mode (see the module-level
/// "Pixel masks — pipeline integrity only" section).
///
/// Chunks may have differing TOF axis lengths (`n_tof`) — ragged event-mode
/// re-histogramming is fine; deadness is spatial, so only the spatial
/// dimensions must agree.
///
/// # Arguments
/// * `chunks` — One 3D counts array per acquisition chunk, each with shape
///   (n_tof, height, width).  `n_tof` may differ between chunks; (height,
///   width) must not.
///
/// # Returns
/// 2D boolean mask, shape (height, width). `true` = dead in at least one
/// chunk.
///
/// # Errors
/// Returns `IoError::InvalidParameter` if `chunks` is empty or any chunk
/// contains a non-finite or negative value, and `IoError::ShapeMismatch` if
/// the chunks' spatial dimensions differ.
pub fn detect_dead_pixels_chunked(chunks: &[Array3<f64>]) -> Result<Array2<bool>, IoError> {
    if chunks.is_empty() {
        return Err(IoError::InvalidParameter(
            "detect_dead_pixels_chunked requires at least one chunk".into(),
        ));
    }

    let first = chunks[0].shape();
    let (height, width) = (first[1], first[2]);
    for (i, chunk) in chunks.iter().enumerate() {
        let s = chunk.shape();
        // n_tof (s[0]) may differ between chunks; only spatial dims must agree.
        if s[1] != height || s[2] != width {
            return Err(IoError::ShapeMismatch(format!(
                "chunks[{i}] spatial dims ({}, {}) != chunks[0] spatial dims ({height}, {width})",
                s[1], s[2],
            )));
        }
        validate_counts(chunk, &format!("chunks[{i}]"))?;
    }

    let mut mask = Array2::from_elem((height, width), false);
    for chunk in chunks {
        let dead = detect_dead_pixels(chunk);
        Zip::from(&mut mask)
            .and(&dead)
            .for_each(|m, &d| *m = *m || d);
    }
    Ok(mask)
}

/// Detect hot (railed / runaway) pixels via a robust one-sided log-space
/// median + k·MAD screen on per-pixel total counts.
///
/// Pipeline-integrity screen only — see the module-level "Pixel masks —
/// pipeline integrity only" section.  The algorithm:
///
/// 1. `totals[y, x] = Σ_t data[t, y, x]`.
/// 2. The statistics sample is `ln(totals)` over `totals > 0` pixels *only* —
///    dead pixels are excluded *before* the median/MAD so `ln(0)` never
///    enters and a large dead population cannot drag the median down.  If no
///    live pixels exist, every pixel is unflagged (all-`false` mask).
/// 3. `med = median(ln totals)`, `mad = median(|ln totals − med|)`.
/// 4. `sigma = max(MAD_TO_SIGMA·mad, exp(−med/2))`.  The second term is the
///    delta-method Poisson floor of `ln N`: `Var[ln N] ≈ 1/N` evaluated at
///    `N = exp(med)` (medians commute with monotone maps, so `exp(med)` is
///    the median total), giving `σ_floor = 1/√exp(med) = exp(−med/2)`.  The
///    robust scale can never legitimately sit below counting noise; this
///    guards `mad == 0` on quantized low-count images *without* becoming a
///    low-count screen — it only ever raises the threshold.  Worked check:
///    an image where most totals are 1.0 and some are 2.0 has `mad = 0` and
///    floor `= 1`, so the threshold is `e^6 ≈ 403×` the median — the 2-count
///    pixels are NOT flagged, while a railed pixel still is.
/// 5. Flag iff `totals > 0 && ln(total) > med + k_mad·sigma` — **upper tail
///    only**.  A stuck-low pixel is indistinguishable from a low-count-alive
///    pixel and is deliberately kept (masking it would be the banned
///    low-count screen).  Railed/always-max pixels are subsumed by the upper
///    tail: no fixed saturation value exists after efficiency correction, so
///    a saturation-constant test would be wrong anyway.
///
/// # Arguments
/// * `data` — 3D counts array with shape (n_tof, height, width).
/// * `k_mad` — Robust-σ multiplier for the upper-tail cut; use
///   [`HOT_PIXEL_K_MAD`] unless you have a reason not to.
///
/// # Returns
/// 2D boolean mask, shape (height, width). `true` = hot pixel.
///
/// # Errors
/// Returns `IoError::InvalidParameter` if `data` contains a non-finite or
/// negative value, or if `k_mad` is not finite and positive.
pub fn detect_hot_pixels(data: &Array3<f64>, k_mad: f64) -> Result<Array2<bool>, IoError> {
    validate_counts(data, "data")?;
    // NaN bypasses `>`, so pair the order comparison with is_finite().
    if !(k_mad.is_finite() && k_mad > 0.0) {
        return Err(IoError::InvalidParameter(format!(
            "k_mad must be finite and > 0, got {k_mad}"
        )));
    }

    let totals = data.sum_axis(Axis(0));
    let mut mask = Array2::from_elem(totals.raw_dim(), false);

    // Statistics over live (totals > 0) pixels only: ln(0) never enters, and
    // a large dead population cannot drag the median down.
    let log_totals: Vec<f64> = totals
        .iter()
        .filter(|&&t| t > 0.0)
        .map(|&t| t.ln())
        .collect();
    let Some(med) = nereids_core::stats::median(&log_totals) else {
        // No live pixels at all: nothing to compare against — flag nothing.
        return Ok(mask);
    };
    // unwrap is safe here: median() returned Some, so log_totals is non-empty.
    let mad = nereids_core::stats::median_abs_deviation(&log_totals, med).unwrap();

    // Delta-method Poisson floor of ln N at the median total (see rustdoc
    // step 4): the robust scale can never sit below counting noise.
    let sigma = f64::max(nereids_core::stats::MAD_TO_SIGMA * mad, (-med / 2.0).exp());
    let threshold = med + k_mad * sigma;

    Zip::from(&mut mask)
        .and(&totals)
        .for_each(|m, &t| *m = t > 0.0 && t.ln() > threshold);
    Ok(mask)
}

/// Detect all pipeline-corrupting pixels: dead ∪ hot over sample and
/// (optionally) open beam.
///
/// This is the validating entry point that the GUI and Python bindings
/// should use.  Deadness/hotness is per-acquisition — a pixel dead only in
/// the open-beam run still corrupts every transmission ratio computed from
/// it — so the masks of both stacks are unioned:
///
/// `mask = dead(sample) ∪ hot(sample) [∪ dead(open_beam) ∪ hot(open_beam)]`
///
/// The stacks' TOF axis lengths may differ (deadness is spatial); only the
/// spatial dimensions must agree.
///
/// # Arguments
/// * `sample` — Sample counts, shape (n_tof, height, width).
/// * `open_beam` — Optional open-beam counts, shape (n_tof', height, width).
/// * `hot_k_mad` — `Some(k)` to include the [`detect_hot_pixels`] screen
///   with multiplier `k` (use [`HOT_PIXEL_K_MAD`]); `None` for dead-only
///   detection.
///
/// # Returns
/// 2D boolean mask, shape (height, width). `true` = exclude pixel.
///
/// # Errors
/// Returns `IoError::InvalidParameter` if either stack contains a non-finite
/// or negative value or `hot_k_mad` is `Some` of a non-finite/non-positive
/// value, and `IoError::ShapeMismatch` if the spatial dimensions differ.
pub fn detect_bad_pixels(
    sample: &Array3<f64>,
    open_beam: Option<&Array3<f64>>,
    hot_k_mad: Option<f64>,
) -> Result<Array2<bool>, IoError> {
    // Validate everything up front (house rule), before any detector runs.
    // The public detectors called below re-validate; that duplication is one
    // O(n) sweep and keeps each entry point independently safe.
    validate_counts(sample, "sample")?;
    if let Some(ob) = open_beam {
        validate_counts(ob, "open_beam")?;
        let (ss, os) = (sample.shape(), ob.shape());
        // n_tof may differ (deadness is spatial); spatial dims must agree.
        if ss[1] != os[1] || ss[2] != os[2] {
            return Err(IoError::ShapeMismatch(format!(
                "sample spatial dims ({}, {}) != open_beam spatial dims ({}, {})",
                ss[1], ss[2], os[1], os[2],
            )));
        }
    }
    if let Some(k) = hot_k_mad {
        // NaN bypasses `>`, so pair the order comparison with is_finite().
        if !(k.is_finite() && k > 0.0) {
            return Err(IoError::InvalidParameter(format!(
                "hot_k_mad must be finite and > 0, got {k}"
            )));
        }
    }

    let mut mask = detect_dead_pixels(sample);
    if let Some(k) = hot_k_mad {
        let hot = detect_hot_pixels(sample, k)?;
        Zip::from(&mut mask)
            .and(&hot)
            .for_each(|m, &h| *m = *m || h);
    }
    if let Some(ob) = open_beam {
        let ob_dead = detect_dead_pixels(ob);
        Zip::from(&mut mask)
            .and(&ob_dead)
            .for_each(|m, &d| *m = *m || d);
        if let Some(k) = hot_k_mad {
            let ob_hot = detect_hot_pixels(ob, k)?;
            Zip::from(&mut mask)
                .and(&ob_hot)
                .for_each(|m, &h| *m = *m || h);
        }
    }
    Ok(mask)
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
    fn test_normalize_rejects_nan_sample() {
        // A NaN in the sample frame used to be swallowed by
        // `(NaN - 0).max(0.0) == 0.0`, silently producing T = 0 as if the
        // bin had genuinely zero counts.  It must now be rejected up front.
        let mut sample = Array3::from_elem((1, 1, 1), 50.0);
        sample[[0, 0, 0]] = f64::NAN;
        let ob = Array3::from_elem((1, 1, 1), 100.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };
        let err = normalize(&sample, &ob, &params, None).unwrap_err();
        assert!(
            matches!(err, IoError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        assert!(err.to_string().contains("sample"));
    }

    #[test]
    fn test_normalize_rejects_negative_sample() {
        // Negative raw counts (loader bug) used to be clamped to 0.
        let mut sample = Array3::from_elem((1, 1, 1), 50.0);
        sample[[0, 0, 0]] = -5.0;
        let ob = Array3::from_elem((1, 1, 1), 100.0);
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };
        let err = normalize(&sample, &ob, &params, None).unwrap_err();
        assert!(err.to_string().contains("sample"));
    }

    #[test]
    fn test_normalize_rejects_nan_open_beam() {
        let sample = Array3::from_elem((1, 1, 1), 50.0);
        let mut ob = Array3::from_elem((1, 1, 1), 100.0);
        ob[[0, 0, 0]] = f64::INFINITY;
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };
        let err = normalize(&sample, &ob, &params, None).unwrap_err();
        assert!(err.to_string().contains("open_beam"));
    }

    #[test]
    fn test_normalize_rejects_negative_dark_current() {
        let sample = Array3::from_elem((1, 1, 1), 60.0);
        let ob = Array3::from_elem((1, 1, 1), 110.0);
        let mut dc = Array2::from_elem((1, 1), 10.0);
        dc[[0, 0]] = -1.0;
        let params = NormalizationParams {
            proton_charge_sample: 1.0,
            proton_charge_ob: 1.0,
        };
        let err = normalize(&sample, &ob, &params, Some(&dc)).unwrap_err();
        assert!(err.to_string().contains("dark_current"));
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

    #[test]
    fn test_detect_dead_pixels_chunked_catches_intermittent() {
        // ACCEPTANCE (#643): pixel (0, 1) is dead throughout chunk 0 but
        // alive in chunk 1 — intermittent deadness that corrupts the
        // combined spectrum.
        let mut chunk0 = Array3::from_elem((3, 2, 2), 5.0);
        for t in 0..3 {
            chunk0[[t, 0, 1]] = 0.0;
        }
        let chunk1 = Array3::from_elem((3, 2, 2), 5.0);

        let mask = detect_dead_pixels_chunked(&[chunk0.clone(), chunk1.clone()]).unwrap();
        assert!(mask[[0, 1]], "intermittently dead pixel must be flagged");
        assert!(!mask[[0, 0]]);
        assert!(!mask[[1, 0]]);
        assert!(!mask[[1, 1]]);

        // The gap being closed: on the element-wise summed stack the pixel
        // has nonzero counts everywhere, so detect_dead_pixels misses it.
        let summed = &chunk0 + &chunk1;
        let summed_mask = detect_dead_pixels(&summed);
        assert!(
            !summed_mask[[0, 1]],
            "summed-stack detection cannot see intermittent deadness — \
             that is exactly why the chunked variant exists"
        );
    }

    #[test]
    fn test_detect_dead_pixels_chunked_low_count_alive_not_flagged() {
        // ACCEPTANCE (#643): a pixel with a single count per chunk is alive
        // and must be kept — masks are pipeline-integrity only, never a
        // low-count screen.
        let mut chunk0 = Array3::from_elem((4, 2, 2), 5.0);
        let mut chunk1 = Array3::from_elem((4, 2, 2), 5.0);
        for t in 0..4 {
            chunk0[[t, 1, 1]] = 0.0;
            chunk1[[t, 1, 1]] = 0.0;
        }
        chunk0[[2, 1, 1]] = 1.0; // one lone count in chunk 0
        chunk1[[0, 1, 1]] = 1.0; // one lone count in chunk 1

        let mask = detect_dead_pixels_chunked(&[chunk0, chunk1]).unwrap();
        assert!(!mask[[1, 1]], "low-count-alive pixel must not be flagged");
    }

    #[test]
    fn test_detect_dead_pixels_chunked_empty_slice_err() {
        let err = detect_dead_pixels_chunked(&[]).unwrap_err();
        assert!(matches!(err, IoError::InvalidParameter(_)));
    }

    #[test]
    fn test_detect_dead_pixels_chunked_spatial_mismatch_err() {
        let chunk0 = Array3::from_elem((3, 2, 2), 1.0);
        let chunk1 = Array3::from_elem((3, 2, 3), 1.0);
        let err = detect_dead_pixels_chunked(&[chunk0, chunk1]).unwrap_err();
        assert!(matches!(err, IoError::ShapeMismatch(_)));
    }

    #[test]
    fn test_detect_dead_pixels_chunked_nan_err() {
        let chunk0 = Array3::from_elem((3, 2, 2), 1.0);
        let mut chunk1 = Array3::from_elem((3, 2, 2), 1.0);
        chunk1[[0, 0, 0]] = f64::NAN;
        let err = detect_dead_pixels_chunked(&[chunk0, chunk1]).unwrap_err();
        assert!(matches!(err, IoError::InvalidParameter(_)));
        assert!(err.to_string().contains("chunks[1]"));
    }

    #[test]
    fn test_detect_dead_pixels_chunked_ragged_n_tof_ok() {
        // Ragged event re-histogramming: n_tof may differ between chunks.
        let chunk0 = Array3::from_elem((3, 2, 2), 1.0);
        let chunk1 = Array3::from_elem((7, 2, 2), 1.0);
        let mask = detect_dead_pixels_chunked(&[chunk0, chunk1]).unwrap();
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_detect_hot_pixels_catches_railed() {
        // ACCEPTANCE (#643): a railed pixel (65535 counts/bin) amid a
        // realistic slightly varying background (~100/bin) is flagged;
        // its neighbors are not.
        let mut data = Array3::<f64>::zeros((4, 3, 3));
        for t in 0..4 {
            for y in 0..3 {
                for x in 0..3 {
                    // Background 95..103 counts/bin (totals 380..412).
                    data[[t, y, x]] = 95.0 + (y * 3 + x) as f64;
                }
            }
        }
        for t in 0..4 {
            data[[t, 1, 1]] = 65535.0;
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(mask[[1, 1]], "railed pixel must be flagged");
        for y in 0..3 {
            for x in 0..3 {
                if (y, x) != (1, 1) {
                    assert!(!mask[[y, x]], "neighbor ({y}, {x}) must not be flagged");
                }
            }
        }
    }

    #[test]
    fn test_detect_hot_pixels_low_count_alive_not_flagged() {
        // ACCEPTANCE (#643) — the 13%-rejection regression guard: a pixel
        // with 1 total count amid ~300-count pixels is low-count-ALIVE and
        // must be kept.  The screen is upper-tail only.
        let mut data = Array3::from_elem((3, 3, 3), 100.0);
        for t in 0..3 {
            data[[t, 0, 2]] = 0.0;
        }
        data[[1, 0, 2]] = 1.0; // total 1 count

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(!mask[[0, 2]], "low-count-alive pixel must not be flagged");
        assert!(mask.iter().all(|&m| !m), "nothing here is hot");
    }

    #[test]
    fn test_detect_hot_pixels_uniform_image_poisson_floor_path() {
        // Perfectly uniform background → MAD == 0 → the delta-method
        // Poisson floor exp(-med/2) is the active scale.  The railed pixel
        // must still be flagged, the uniform background must not.
        let mut data = Array3::from_elem((4, 3, 3), 100.0);
        for t in 0..4 {
            data[[t, 2, 0]] = 65535.0;
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(mask[[2, 0]]);
        assert_eq!(mask.iter().filter(|&&m| m).count(), 1);
    }

    #[test]
    fn test_detect_hot_pixels_quantized_low_counts_none_flagged() {
        // Worked check from the rustdoc: mostly 1-count totals with some
        // 2-count totals → mad = 0, floor = 1, threshold = e^6 ≈ 403× the
        // median — the 2-count pixels are NOT flagged.
        let mut data = Array3::from_elem((1, 3, 3), 1.0);
        data[[0, 0, 0]] = 2.0;
        data[[0, 2, 2]] = 2.0;

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(
            mask.iter().all(|&m| !m),
            "quantized low-count image must produce no hot flags"
        );
    }

    #[test]
    fn test_detect_hot_pixels_majority_dead_live_not_flagged() {
        // A large dead population must not drag the median down and get the
        // few live pixels flagged: the statistics sample is live-only.
        let mut data = Array3::<f64>::zeros((1, 4, 4));
        data[[0, 0, 0]] = 49.0;
        data[[0, 1, 1]] = 50.0;
        data[[0, 2, 2]] = 51.0;

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(
            mask.iter().all(|&m| !m),
            "live pixels amid a dead majority must not be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_all_dead_all_false() {
        let data = Array3::<f64>::zeros((3, 2, 2));
        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_detect_hot_pixels_nan_err() {
        let mut data = Array3::from_elem((2, 2, 2), 1.0);
        data[[1, 1, 0]] = f64::NAN;
        let err = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap_err();
        assert!(matches!(err, IoError::InvalidParameter(_)));
    }

    #[test]
    fn test_detect_hot_pixels_bad_k_err() {
        let data = Array3::from_elem((2, 2, 2), 1.0);
        for bad_k in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = detect_hot_pixels(&data, bad_k).unwrap_err();
            assert!(
                matches!(err, IoError::InvalidParameter(_)),
                "k_mad = {bad_k} must be rejected"
            );
        }
    }

    #[test]
    fn test_detect_bad_pixels_union() {
        // sample: (0,0) dead, (1,2) low-count-alive (1 total count),
        //         everything else 100 counts/bin.
        let mut sample = Array3::from_elem((3, 3, 3), 100.0);
        for t in 0..3 {
            sample[[t, 0, 0]] = 0.0;
            sample[[t, 1, 2]] = 0.0;
        }
        sample[[0, 1, 2]] = 1.0;
        // open beam: (0,1) dead, (2,2) railed, everything else 200/bin.
        let mut ob = Array3::from_elem((3, 3, 3), 200.0);
        for t in 0..3 {
            ob[[t, 0, 1]] = 0.0;
            ob[[t, 2, 2]] = 65535.0;
        }

        let mask = detect_bad_pixels(&sample, Some(&ob), Some(HOT_PIXEL_K_MAD)).unwrap();
        assert!(mask[[0, 0]], "dead-in-sample-only must be flagged");
        assert!(mask[[0, 1]], "dead-in-OB-only must be flagged");
        assert!(mask[[2, 2]], "hot-in-OB-only must be flagged");
        assert!(!mask[[1, 2]], "low-count-alive must be kept");
        assert!(!mask[[1, 1]], "normal pixel must be kept");
        assert_eq!(mask.iter().filter(|&&m| m).count(), 3);
    }

    #[test]
    fn test_detect_bad_pixels_spatial_mismatch_err() {
        let sample = Array3::from_elem((3, 2, 2), 1.0);
        let ob = Array3::from_elem((3, 2, 3), 1.0);
        let err = detect_bad_pixels(&sample, Some(&ob), None).unwrap_err();
        assert!(matches!(err, IoError::ShapeMismatch(_)));
    }

    #[test]
    fn test_detect_bad_pixels_ragged_n_tof_ok() {
        // Deadness is spatial: sample and OB may have different TOF axes.
        let sample = Array3::from_elem((3, 2, 2), 1.0);
        let ob = Array3::from_elem((7, 2, 2), 1.0);
        let mask = detect_bad_pixels(&sample, Some(&ob), Some(HOT_PIXEL_K_MAD)).unwrap();
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_detect_bad_pixels_hot_k_mad_none_is_dead_only() {
        let mut sample = Array3::from_elem((3, 3, 3), 100.0);
        for t in 0..3 {
            sample[[t, 1, 1]] = 65535.0; // railed
            sample[[t, 0, 0]] = 0.0; // dead
        }
        let dead_only = detect_bad_pixels(&sample, None, None).unwrap();
        assert!(dead_only[[0, 0]]);
        assert!(
            !dead_only[[1, 1]],
            "hot_k_mad = None must disable the hot screen"
        );

        let with_hot = detect_bad_pixels(&sample, None, Some(HOT_PIXEL_K_MAD)).unwrap();
        assert!(with_hot[[0, 0]]);
        assert!(with_hot[[1, 1]]);
    }

    #[test]
    fn test_detect_bad_pixels_bad_k_err() {
        let sample = Array3::from_elem((2, 2, 2), 1.0);
        for bad_k in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = detect_bad_pixels(&sample, None, Some(bad_k)).unwrap_err();
            assert!(
                matches!(err, IoError::InvalidParameter(_)),
                "hot_k_mad = {bad_k} must be rejected"
            );
        }
    }

    #[test]
    fn test_detect_bad_pixels_nan_err() {
        let mut sample = Array3::from_elem((2, 2, 2), 1.0);
        sample[[0, 0, 1]] = f64::NAN;
        let err = detect_bad_pixels(&sample, None, None).unwrap_err();
        assert!(err.to_string().contains("sample"));

        let sample = Array3::from_elem((2, 2, 2), 1.0);
        let mut ob = Array3::from_elem((2, 2, 2), 1.0);
        ob[[1, 0, 0]] = f64::NEG_INFINITY;
        let err = detect_bad_pixels(&sample, Some(&ob), None).unwrap_err();
        assert!(err.to_string().contains("open_beam"));
    }
}
