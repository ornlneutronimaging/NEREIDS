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
/// - An empty TOF axis (`shape[0] == 0`) makes the all-zero test vacuously
///   true for *every* pixel — the whole detector would be reported dead.
///   This non-validating function keeps that behaviour for backward
///   compatibility; the validating entry points ([`detect_bad_pixels`],
///   [`detect_dead_pixels_chunked`], [`detect_hot_pixels`]) reject empty
///   stacks up front.
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

/// Default MAD multiplier for the global (stage-1) cut of
/// [`detect_hot_pixels`].
///
/// The one-sided Gaussian tail at 6 robust σ is P(Z > 6) ≈ 9.9e-10, i.e.
/// ~2.6e-4 expected false flags on a full 512×512 frame (262 144 pixels) —
/// on a *unimodal* image the screen essentially never rejects a
/// statistically plausible pixel, while a railed pixel sits tens of robust
/// σ above any plausible median.
///
/// On a **bimodal** image the global cut alone is not trustworthy: when the
/// darker population holds the median (a sample covering >50 % of the FOV,
/// or an aperture-limited open beam), the MAD reflects only the dark
/// population's internal spread, and *every* bright-region pixel lands
/// above `med + k·MAD`.  [`detect_hot_pixels`] therefore never flags on the
/// global cut alone — the local-neighborhood confirmation
/// ([`HOT_LOCAL_FACTOR`]) must also pass.
pub const HOT_PIXEL_K_MAD: f64 = 6.0;

/// Local-neighborhood confirmation factor (stage 2) of
/// [`detect_hot_pixels`].
///
/// A pixel that passes the global cut is flagged only if its total also
/// exceeds `HOT_LOCAL_FACTOR ×` the median total of its available live
/// 8-neighbors.  The factor separates detector *point defects* from scene
/// structure:
///
/// - A railed/runaway pixel is spatially isolated and typically ≥100× its
///   neighbors, so it clears 10× with a wide margin.
/// - Adjacent-pixel scene gradients (beam profile, sample absorption) are
///   ≤2–3×; even directly across a sharp sample edge only one ring of
///   neighbors is mixed, and the neighbor *median* stays on the pixel's
///   own side of the edge.
/// - A fully-railed 1-px row/column still leaves each railed pixel with
///   ≥5 normal neighbors of 8, so the neighbor median stays normal and the
///   line IS caught in a single pass.  Railed CLUSTERS ≥2 px wide are
///   caught by the stage-2 fixpoint erosion — see the
///   "Fixpoint erosion of railed clusters" section of
///   [`detect_hot_pixels`].
///
/// **Width-1 limitation (accepted trade-off)**: a 1-px-wide bright *scene*
/// line at ≥`HOT_LOCAL_FACTOR`× local contrast is spatially
/// indistinguishable from a railed line and IS masked.  Contiguous bright
/// regions ≥2 px wide are safe (their boundary pixels keep a same-side
/// neighbor median, so the erosion never seeds — see [`detect_hot_pixels`],
/// "Why bright scene regions never erode").  Real scene features on VENUS
/// are PSF-blurred over ≥2 px, so ≥10× single-pixel scene contrast is
/// physically rare; masking it is the accepted price for catching railed
/// rows/columns.
pub const HOT_LOCAL_FACTOR: f64 = 10.0;

/// Reject a stack with an empty TOF axis (`shape[0] == 0`).
///
/// Every-bin predicates are vacuously true over zero bins: an empty stack
/// would mark the whole detector dead (and gives the hot screen an all-zero
/// totals image) with no error.  Called up front by the validating detector
/// entry points; [`detect_dead_pixels`] deliberately stays non-validating
/// (see its rustdoc).
fn validate_n_tof(data: &Array3<f64>, field: &str) -> Result<(), IoError> {
    if data.shape()[0] == 0 {
        return Err(IoError::InvalidParameter(format!(
            "{field} has an empty TOF axis (shape[0] == 0) — dead/hot detection \
             over zero bins is vacuous and would mask every pixel"
        )));
    }
    Ok(())
}

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
/// Returns `IoError::InvalidParameter` if `chunks` is empty, any chunk has
/// an empty TOF axis (`shape[0] == 0` — its all-zero test would vacuously
/// mark every pixel dead), or any chunk contains a non-finite or negative
/// value, and `IoError::ShapeMismatch` if the chunks' spatial dimensions
/// differ.
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
        validate_n_tof(chunk, &format!("chunks[{i}]"))?;
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

/// Detect hot (railed / runaway) pixels via a two-stage criterion: a
/// robust one-sided log-space median + k·MAD screen on per-pixel total
/// counts (stage 1, global), confirmed by a local-neighborhood isolation
/// test (stage 2).
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
/// 5. Stage 1 (global): a pixel is a *candidate* iff `totals > 0 &&
///    ln(total) > med + k_mad·sigma` — **upper tail only**.  A stuck-low
///    pixel is indistinguishable from a low-count-alive pixel and is
///    deliberately kept (masking it would be the banned low-count screen).
///    Railed/always-max pixels are subsumed by the upper tail: no fixed
///    saturation value exists after efficiency correction, so a
///    saturation-constant test would be wrong anyway.
/// 6. Stage 2 (local confirmation), iterated to a **fixpoint**: a candidate
///    is flagged iff its total also exceeds [`HOT_LOCAL_FACTOR`] × the
///    median of its 8-neighborhood reference sample, where each neighbor
///    contributes its total if live (`total > 0`) and not yet flagged,
///    contributes `0.0` if already flagged (a known defect cannot vouch
///    for its neighbors — see below), and is omitted if dead (a dead pixel
///    carries no scene information).  Edge pixels use whatever neighbors
///    exist.  A candidate whose reference sample is empty (every neighbor
///    dead — isolated live pixel in a dead field) keeps the global verdict;
///    a candidate whose neighbors are mostly flagged defects likewise stays
///    flagged (its reference median is 0).  After each full pass over the
///    fixed stage-1 candidate list, newly confirmed flags are applied and
///    the pass repeats until a pass adds no new flag.
///
/// # Fixpoint erosion of railed clusters
///
/// A single stage-2 pass misses the INTERIOR of a railed cluster ≥2 px
/// wide: an interior pixel's 8-neighbors are railed too, so its neighbor
/// median is railed and the ratio test refutes the flag.  Iterating to a
/// fixpoint erodes such clusters from the boundary inward — once the
/// cluster's outermost pixels are flagged they stop vouching (each
/// contributes `0.0` instead of its railed total), the reference median of
/// the next ring drops back to the background level, and the next pass
/// flags that ring.  Contributing `0.0` (rather than omitting the flagged
/// neighbor) is load-bearing: with omission, a 3-background + 3-railed
/// reference sample has an even-count [`nereids_core::stats::median`]
/// midpoint mid-gap (≈ railed/2), the ratio test reads ~2× and the erosion
/// stalls; with the zero contribution the median stays on the background
/// side and erosion completes.  Erosion fully consumes clusters up to 3 px
/// wide in their narrower dimension (point defects, 1-px lines, 2–3-px-wide
/// blobs/segments — the physical shapes of railed detector defects)
/// PROVIDED the cluster exposes at least one end cap or convex corner to
/// normal-scene neighbors — erosion must seed somewhere.  An EDGE-TO-EDGE
/// railed band ≥2 px wide (both ends off-detector — spanning the full
/// detector width or height) exposes neither: every interior band pixel
/// keeps ≥5 railed of 8 neighbors, and even the on-detector-border band
/// ends keep 3 railed of 5, so every neighbor median stays railed, no
/// pixel ever seeds, and the band is NOT caught
/// (`test_detect_hot_pixels_edge_to_edge_2px_band_not_flagged_by_design`).
/// This is deliberate, not an oversight: a slit-aperture open beam
/// produces a genuine full-width bright SCENE band that is
/// pixel-for-pixel indistinguishable from such a defect, so a full-span
/// row/column screen would mask it — re-introducing the exact bimodal
/// failure stage 2 exists to prevent.  Full-span detector pathologies of
/// width ≥2 belong in a declared/file mask.  (A full-span width-1 railed
/// line IS caught — each of its pixels keeps ≥4 normal neighbors,
/// `test_detect_hot_pixels_full_railed_column_caught`; and a ≥2-px band
/// with even one end cap inside the detector is fully consumed from that
/// cap, `test_detect_hot_pixels_2px_band_one_end_on_detector_fully_caught`.)
/// A hard-edged railed rectangle ≥4 px wide keeps its interior (only its
/// convex corners flag): it is pixel-for-pixel indistinguishable from a
/// hard-edged bright scene region, which must survive (below).
///
/// **Termination bound**: flags are only ever added, and every pass except
/// the last adds at least one, so at most `height·width` passes can do
/// work; the loop is additionally capped at `height·width` passes to make
/// the bound structural rather than reasoned.  In practice the pass count
/// is on the order of the defect-cluster radius (one pass for point
/// defects and 1-px lines).
///
/// # Why bright scene regions never erode
///
/// Erosion must *seed* at a bright-region boundary pixel.  A boundary
/// pixel of a contiguous bright scene region ≥2 px wide keeps ≥4 of its
/// 8 neighbors on its own (bright) side for any straight or diagonal
/// edge, so its reference median stays bright and its ratio is the scene
/// gradient (≤2–3× across real edges) — far below the ≥10×
/// [`HOT_LOCAL_FACTOR`].  Stage 1's global cut additionally gates which
/// pixels can seed at all.  With no seed, the fixpoint is reached with
/// zero flags in the region and it survives intact
/// (`test_detect_hot_pixels_large_psf_bright_region_not_eroded`).  Two
/// documented, test-pinned exceptions — both physically rare on VENUS,
/// where the detector PSF blurs real scene features over ≥2 px so ≥10×
/// single-pixel contrast steps do not occur in scene:
///
/// - a **width-1 bright line** at ≥10× local contrast is spatially
///   indistinguishable from a railed line and IS masked — the accepted
///   trade-off for catching railed rows/columns
///   (`test_detect_hot_pixels_1px_bright_line_flagged_by_design`);
/// - the single pixel at a **sharp convex (90°) corner** of a hard-edged
///   ≥10× region sees only 3 same-side neighbors, its reference median
///   falls on the dark side, and it is flagged (this predates the
///   fixpoint); erosion does NOT propagate past it — the adjacent edge
///   pixels keep ≥4 bright unflagged neighbors
///   (`test_detect_hot_pixels_hard_edged_bright_rectangle_corners_only`).
///
/// The two stages encode complementary definitions of "hot": stage 1 says
/// *statistically implausible for this image*, stage 2 says *spatially
/// isolated*, and a railed/hot pixel is a detector **point defect** — it
/// must be both.  Stage 2 exists because the global cut alone fails
/// catastrophically on **bimodal** images: with a dark majority holding the
/// median (a sample covering >50 % of the FOV, or an aperture-limited open
/// beam), the MAD reflects only the dark population's internal spread and
/// the ENTIRE bright minority exceeds `med + k·MAD`.  A contiguous bright
/// REGION is scene, not a defect — masking it would reject statistically
/// plausible pixels, the exact failure the module rules ban.  A true point
/// defect beats its neighbor median by ≥100× and is caught by both stages
/// even *inside* a bright region and even as part of a railed row/column
/// or a small railed cluster (see [`HOT_LOCAL_FACTOR`]).
///
/// # Raw counts required
///
/// `data` must be **raw detected counts** (unscaled).  The Poisson floor in
/// step 4 assumes `Var[N] = N`; any prior scaling silently breaks that
/// identity — down-scaling (proton-charge-normalized rates ≪ 1, per-pixel
/// gain division) inflates the floor and can suppress real flags, while
/// up-scaling (event weights > 1) deflates it below true counting noise.
/// Run the detectors on unscaled counts and normalize afterwards; the GUI
/// does exactly this (both of its normalization paths pass the raw
/// sample/open-beam stacks, before any normalization).
///
/// # Arguments
/// * `data` — 3D raw-counts array with shape (n_tof, height, width).
/// * `k_mad` — Robust-σ multiplier for the stage-1 upper-tail cut; use
///   [`HOT_PIXEL_K_MAD`] unless you have a reason not to.
///
/// # Returns
/// 2D boolean mask, shape (height, width). `true` = hot pixel.
///
/// # Errors
/// Returns `IoError::InvalidParameter` if `data` contains a non-finite or
/// negative value or has an empty TOF axis (`shape[0] == 0`), or if `k_mad`
/// is not finite and positive.
pub fn detect_hot_pixels(data: &Array3<f64>, k_mad: f64) -> Result<Array2<bool>, IoError> {
    validate_counts(data, "data")?;
    // An empty TOF axis would yield an all-zero totals image (empty live
    // set → all-false mask) rather than corrupt output, but the validating
    // entry points reject it uniformly (see validate_n_tof).
    validate_n_tof(data, "data")?;
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

    // Stage 1 (global robust cut), upper tail only: the candidate list is
    // fixed for the whole stage-2 fixpoint iteration below.
    let (height, width) = totals.dim();
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let total = totals[[y, x]];
            if total > 0.0 && total.ln() > threshold {
                candidates.push((y, x));
            }
        }
    }

    // Stage 2 (local confirmation): only a spatially ISOLATED excess is a
    // detector point defect — a contiguous bright region is scene.  This is
    // what keeps the global cut honest on bimodal images (see rustdoc).
    //
    // Iterated to a FIXPOINT to erode railed clusters from their boundary
    // inward (see rustdoc, "Fixpoint erosion of railed clusters"): each
    // pass evaluates the not-yet-flagged candidates against the PREVIOUS
    // pass's mask (batch update — order-independent within a pass), and
    // the loop ends when a pass adds no new flag.
    //
    // Termination bound: flags are only ever added and every pass except
    // the last adds at least one, so ≤ height·width passes can do work;
    // the explicit cap makes that bound structural.  Pass 1 sees an
    // all-false mask and is exactly the pre-fixpoint single pass.
    let max_passes = height * width;
    let mut neighbor_totals: Vec<f64> = Vec::with_capacity(8);
    for _pass in 0..max_passes {
        let mut newly_flagged: Vec<(usize, usize)> = Vec::new();
        for &(y, x) in &candidates {
            if mask[[y, x]] {
                continue;
            }
            let total = totals[[y, x]];
            // 8-neighborhood reference sample; edge pixels use what exists.
            neighbor_totals.clear();
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if (ny, nx) == (y, x) {
                        continue;
                    }
                    if mask[[ny, nx]] {
                        // Already-flagged neighbor: a known defect cannot
                        // vouch for the candidate.  It contributes the
                        // lowest possible scene value (zero total) so the
                        // reference median is not dragged up by the defect
                        // itself — load-bearing for cluster erosion (see
                        // rustdoc: an omitted neighbor leaves an even-count
                        // sample whose midpoint median stalls the erosion).
                        neighbor_totals.push(0.0);
                    } else {
                        // Live (total > 0) unflagged neighbors contribute
                        // their totals; dead neighbors carry no scene
                        // information and are omitted.
                        let t = totals[[ny, nx]];
                        if t > 0.0 {
                            neighbor_totals.push(t);
                        }
                    }
                }
            }
            let flagged = match nereids_core::stats::median(&neighbor_totals) {
                Some(local_med) => total > HOT_LOCAL_FACTOR * local_med,
                // Empty sample: every neighbor is dead (isolated live pixel
                // in a dead field) — nothing local can refute the global
                // verdict.  (A mostly-flagged neighborhood is NOT empty:
                // the zeros give a reference median of 0 and the candidate
                // stays flagged, the consistent generalization.)
                None => true,
            };
            if flagged {
                newly_flagged.push((y, x));
            }
        }
        if newly_flagged.is_empty() {
            break;
        }
        for &(y, x) in &newly_flagged {
            mask[[y, x]] = true;
        }
    }
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
/// Both stacks must be **raw detected counts** (unscaled) — see the "Raw
/// counts required" section of [`detect_hot_pixels`]: scaling distorts the
/// Poisson floor of the hot screen.  The GUI satisfies this: both of its
/// normalization paths call this function on the raw sample/open-beam
/// stacks, before any normalization.
///
/// # Arguments
/// * `sample` — Raw sample counts, shape (n_tof, height, width).
/// * `open_beam` — Optional raw open-beam counts, shape
///   (n_tof', height, width).
/// * `hot_k_mad` — `Some(k)` to include the [`detect_hot_pixels`] screen
///   with multiplier `k` (use [`HOT_PIXEL_K_MAD`]); `None` for dead-only
///   detection.
///
/// # Returns
/// 2D boolean mask, shape (height, width). `true` = exclude pixel.
///
/// # Errors
/// Returns `IoError::InvalidParameter` if either stack contains a
/// non-finite or negative value or has an empty TOF axis (`shape[0] == 0`
/// — the dead test over zero bins would vacuously mask every pixel) or
/// `hot_k_mad` is `Some` of a non-finite/non-positive value, and
/// `IoError::ShapeMismatch` if the spatial dimensions differ.
pub fn detect_bad_pixels(
    sample: &Array3<f64>,
    open_beam: Option<&Array3<f64>>,
    hot_k_mad: Option<f64>,
) -> Result<Array2<bool>, IoError> {
    // Validate everything up front (house rule), before any detector runs.
    // The public detectors called below re-validate; that duplication is one
    // O(n) sweep and keeps each entry point independently safe.
    validate_counts(sample, "sample")?;
    validate_n_tof(sample, "sample")?;
    if let Some(ob) = open_beam {
        validate_counts(ob, "open_beam")?;
        validate_n_tof(ob, "open_beam")?;
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
    fn test_detect_dead_pixels_chunked_empty_tof_chunk_err() {
        // A zero-frame chunk would vacuously mark the whole detector dead.
        let chunk0 = Array3::from_elem((3, 2, 2), 1.0);
        let chunk1 = Array3::<f64>::zeros((0, 2, 2));
        let err = detect_dead_pixels_chunked(&[chunk0, chunk1]).unwrap_err();
        assert!(matches!(err, IoError::InvalidParameter(_)));
        assert!(err.to_string().contains("chunks[1]"));
        assert!(err.to_string().contains("empty TOF axis"));
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
    fn test_detect_hot_pixels_bimodal_bright_region_not_flagged() {
        // THE PR-#646 P0 regression guard.  Dark-majority bimodal scene:
        // 60 % of the FOV at 50 counts (sample), a contiguous 40 % bright
        // region at 5000 counts (open beam past the sample edge).  The dark
        // population holds the median (med = ln 50) and the MAD is 0 (both
        // populations are internally uniform), so sigma falls to the Poisson
        // floor 1/√50 ≈ 0.141 and the stage-1 threshold is
        // ln 50 + 6·0.141 ≈ 4.76 — EVERY bright pixel (ln 5000 ≈ 8.52)
        // passes the global cut.  The local confirmation must veto them
        // all: each bright pixel's neighbor median is 5000, and
        // 5000 > 10 × 5000 is false.  Bright scene is not a defect.
        let mut data = Array3::from_elem((1, 10, 10), 50.0);
        for y in 0..10 {
            for x in 6..10 {
                data[[0, y, x]] = 5000.0;
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(
            mask.iter().all(|&m| !m),
            "no pixel of a bimodal scene may be flagged — the bright \
             minority region is scene, not a defect"
        );
    }

    #[test]
    fn test_detect_hot_pixels_railed_inside_bright_region_caught() {
        // Same bimodal scene, but with a genuinely railed pixel INSIDE the
        // bright region: 1e6 ≈ 200× its bright neighbors.  It must still be
        // caught (stage 2 compares against the LOCAL median of 5000, not
        // the global dark median).
        let mut data = Array3::from_elem((1, 10, 10), 50.0);
        for y in 0..10 {
            for x in 6..10 {
                data[[0, y, x]] = 5000.0;
            }
        }
        data[[0, 5, 8]] = 1.0e6;

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(
            mask[[5, 8]],
            "railed pixel inside bright region must be flagged"
        );
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            1,
            "only the railed pixel may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_railed_column_segment_caught() {
        // Three adjacent railed pixels in a column: each has ≥5 normal
        // neighbors of 8 (the middle one has 6), so the neighbor MEDIAN
        // stays at the background level and the whole segment is caught.
        let mut data = Array3::from_elem((4, 7, 7), 100.0);
        for t in 0..4 {
            for y in 2..5 {
                data[[t, y, 3]] = 65535.0;
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for y in 2..5 {
            assert!(mask[[y, 3]], "railed column pixel ({y}, 3) must be flagged");
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            3,
            "only the railed segment may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_2x2_railed_cluster_fully_caught() {
        // Fixpoint acceptance (#646 review R2, F1): a 2×2 railed cluster
        // has no interior — every pixel keeps 5 background neighbors of 8,
        // so the whole cluster is caught in the first pass.  Regression
        // guard for the smallest ≥2-px-wide cluster.
        let mut data = Array3::from_elem((4, 7, 7), 100.0);
        for t in 0..4 {
            for y in 2..4 {
                for x in 2..4 {
                    data[[t, y, x]] = 65535.0;
                }
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for y in 2..4 {
            for x in 2..4 {
                assert!(mask[[y, x]], "2x2 cluster pixel ({y}, {x}) must be flagged");
            }
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            4,
            "only the railed cluster may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_3x3_railed_blob_fully_caught() {
        // Fixpoint acceptance (#646 review R2, F1) — THE erosion proof.
        // A 3×3 railed blob: pass 1 flags only the 4 corners (5 background
        // neighbors each); the edge centers (3 bg + 5 railed) and the
        // interior (8 railed) are refuted by a single pass — the
        // pre-fixpoint code missed them.  With flagged neighbors
        // contributing zero totals, pass 2 flags the edge centers
        // (sample [0, 0, bg, bg, bg, R, R, R] → median bg) and pass 3 the
        // interior (all-flagged neighborhood → median 0).
        let mut data = Array3::from_elem((4, 9, 9), 100.0);
        for t in 0..4 {
            for y in 3..6 {
                for x in 3..6 {
                    data[[t, y, x]] = 65535.0;
                }
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for y in 3..6 {
            for x in 3..6 {
                assert!(
                    mask[[y, x]],
                    "3x3 blob pixel ({y}, {x}) must be flagged (interior included)"
                );
            }
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            9,
            "only the railed blob may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_2px_wide_railed_column_fully_caught() {
        // Fixpoint acceptance (#646 review R2, F1): the interior of a
        // 2-px-wide railed column (3 bg + 5 railed neighbors per interior
        // pixel) was invisible to the single pass — only the 4 end pixels
        // (5 bg neighbors each) flagged.  The fixpoint erodes the column
        // pairwise from both ends.
        let mut data = Array3::from_elem((4, 9, 9), 100.0);
        for t in 0..4 {
            for y in 2..7 {
                for x in 3..5 {
                    data[[t, y, x]] = 65535.0;
                }
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for y in 2..7 {
            for x in 3..5 {
                assert!(
                    mask[[y, x]],
                    "2-px-wide column pixel ({y}, {x}) must be flagged"
                );
            }
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            10,
            "only the railed column may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_edge_to_edge_2px_band_not_flagged_by_design() {
        // Documented-limitation pin (#646 review R3, F1): an EDGE-TO-EDGE
        // railed band ≥2 px wide (both ends off-detector) exposes no end
        // cap or convex corner, so the erosion has no seed — interior
        // band pixels keep 3 bg + 5 railed neighbors (median railed) and
        // even the on-detector-border band ends keep 2 bg + 3 railed
        // (median railed).  Nothing flags, deliberately: a slit-aperture
        // open beam produces a genuine full-width bright SCENE band that
        // is pixel-for-pixel indistinguishable from this defect, and a
        // full-span row/column screen would mask it (the bimodal
        // failure).  Such detector pathologies belong in a declared/file
        // mask (see rustdoc, "Fixpoint erosion of railed clusters").
        let mut data = Array3::from_elem((4, 9, 9), 100.0);
        for t in 0..4 {
            for y in 3..5 {
                for x in 0..9 {
                    data[[t, y, x]] = 65535.0;
                }
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(
            mask.iter().all(|&m| !m),
            "an edge-to-edge ≥2-px railed band must NOT be flagged \
             (documented limitation — geometrically ambiguous with a \
             slit-aperture bright scene band)"
        );
    }

    #[test]
    fn test_detect_hot_pixels_2px_band_one_end_on_detector_fully_caught() {
        // Companion to the edge-to-edge pin (#646 review R3, F1): the
        // same 2-row railed band, but with ONE end cap inside the
        // detector.  The two cap pixels keep 5 bg + 3 railed neighbors
        // (median bg) and seed in pass 1; the fixpoint then erodes the
        // band column-pair by column-pair all the way to the opposite
        // (detector-border) end.
        let mut data = Array3::from_elem((4, 9, 9), 100.0);
        for t in 0..4 {
            for y in 3..5 {
                for x in 0..7 {
                    data[[t, y, x]] = 65535.0;
                }
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for y in 3..5 {
            for x in 0..7 {
                assert!(mask[[y, x]], "band pixel ({y}, {x}) must be flagged");
            }
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            14,
            "only the railed band may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_full_railed_column_caught() {
        // Regression (#646 review R2, F1 test 3): a full-height 1-px railed
        // column — every pixel keeps ≥4 background neighbors, so the whole
        // line is caught in pass 1, exactly as before the fixpoint.
        let mut data = Array3::from_elem((4, 7, 7), 100.0);
        for t in 0..4 {
            for y in 0..7 {
                data[[t, y, 3]] = 65535.0;
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for y in 0..7 {
            assert!(mask[[y, 3]], "railed column pixel ({y}, 3) must be flagged");
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            7,
            "only the railed column may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_large_psf_bright_region_not_eroded() {
        // Fixpoint safety (#646 review R2, F1 test 5): a LARGE bright scene
        // region — 20×20 core at 100× background — with the ≥2-px PSF edge
        // blur that real VENUS scene features have (adjacent-pixel ratios
        // ≤5× through a 2-px transition ring: 100 → 400 → 2000 → 10000).
        // Every bright-layer pixel passes the stage-1 global cut (dark
        // majority: med = ln 100, mad = 0, Poisson-floor sigma = 0.1 →
        // threshold ≈ 5.21 < ln 400 ≈ 5.99), yet no pixel reaches 10× its
        // neighbor median, so the erosion never seeds and the fixpoint is
        // reached with ZERO flags — the region survives intact.
        let mut data = Array3::from_elem((1, 50, 50), 100.0);
        for y in 13..37 {
            for x in 13..37 {
                data[[0, y, x]] = 400.0; // outer transition ring
            }
        }
        for y in 14..36 {
            for x in 14..36 {
                data[[0, y, x]] = 2000.0; // inner transition ring
            }
        }
        for y in 15..35 {
            for x in 15..35 {
                data[[0, y, x]] = 10000.0; // 20×20 core at 100× background
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(
            mask.iter().all(|&m| !m),
            "a PSF-blurred bright scene region must not be eroded at all"
        );
    }

    #[test]
    fn test_detect_hot_pixels_hard_edged_bright_rectangle_corners_only() {
        // Pins the documented convex-corner caveat AND the fixpoint
        // no-propagation property (#646 review R2).  A hard-edged (0-px
        // transition) 20×20 region at 100× background: each sharp convex
        // corner pixel sees only 3 same-side neighbors (median falls on
        // the dark side) and flags — pre-existing single-pass behavior,
        // physically rare in scene (PSF blurs real edges over ≥2 px).
        // Crucially, the erosion must NOT propagate past the corners: the
        // corner-adjacent edge pixels keep ≥4 bright unflagged neighbors,
        // so the fixpoint stops at exactly the 4 corner pixels.
        let mut data = Array3::from_elem((1, 50, 50), 100.0);
        for y in 15..35 {
            for x in 15..35 {
                data[[0, y, x]] = 10000.0;
            }
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for &(y, x) in &[(15, 15), (15, 34), (34, 15), (34, 34)] {
            assert!(
                mask[[y, x]],
                "sharp convex corner ({y}, {x}) flags (documented caveat)"
            );
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            4,
            "erosion must not propagate past the convex corners"
        );
    }

    #[test]
    fn test_detect_hot_pixels_1px_bright_line_flagged_by_design() {
        // Width-1 limitation pin (#646 review R2, F3 — user-decided:
        // document + pin, no connected-component machinery).  A 1-px-wide
        // bright SCENE line at ≥10× local contrast is spatially
        // indistinguishable from a railed line and IS masked — the
        // accepted trade-off for catching railed rows/columns.  Real VENUS
        // scene features are PSF-blurred over ≥2 px, so this contrast is
        // physically rare in scene (see HOT_LOCAL_FACTOR rustdoc).
        let mut data = Array3::from_elem((1, 9, 9), 100.0);
        for y in 0..9 {
            data[[0, y, 4]] = 5000.0; // 50× the local background
        }

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        for y in 0..9 {
            assert!(
                mask[[y, 4]],
                "1-px bright line pixel ({y}, 4) is flagged BY DESIGN"
            );
        }
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            9,
            "only the width-1 line may be flagged"
        );
    }

    #[test]
    fn test_detect_hot_pixels_isolated_live_pixel_keeps_global_verdict() {
        // A stage-1 candidate whose 8-neighbors are ALL dead has no live
        // neighbor median to refute the global verdict — it stays flagged.
        // Scattered far pixels at 50 counts define the global statistics
        // (med = ln 50, threshold ≈ 4.76); the isolated 1e6 pixel passes.
        let mut data = Array3::<f64>::zeros((1, 5, 5));
        for &(y, x) in &[(0, 0), (0, 2), (0, 4), (4, 0), (4, 2), (4, 4)] {
            data[[0, y, x]] = 50.0;
        }
        data[[0, 2, 2]] = 1.0e6;

        let mask = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap();
        assert!(
            mask[[2, 2]],
            "isolated live candidate in a dead field keeps the global verdict"
        );
        assert_eq!(mask.iter().filter(|&&m| m).count(), 1);
    }

    #[test]
    fn test_detect_hot_pixels_empty_tof_err() {
        // n_tof == 0: the totals image would be all-zero (vacuous sums) —
        // validating entry points must reject rather than return all-false.
        let data = Array3::<f64>::zeros((0, 2, 2));
        let err = detect_hot_pixels(&data, HOT_PIXEL_K_MAD).unwrap_err();
        assert!(matches!(err, IoError::InvalidParameter(_)));
        assert!(err.to_string().contains("empty TOF axis"));
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
    fn test_detect_bad_pixels_empty_tof_err() {
        // An empty stack's all-zero test passes vacuously — without this
        // guard the whole detector would be masked dead with no error.
        let empty = Array3::<f64>::zeros((0, 2, 2));
        let err = detect_bad_pixels(&empty, None, None).unwrap_err();
        assert!(matches!(err, IoError::InvalidParameter(_)));
        assert!(err.to_string().contains("sample"));

        let sample = Array3::from_elem((3, 2, 2), 1.0);
        let err = detect_bad_pixels(&sample, Some(&empty), None).unwrap_err();
        assert!(matches!(err, IoError::InvalidParameter(_)));
        assert!(err.to_string().contains("open_beam"));
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
