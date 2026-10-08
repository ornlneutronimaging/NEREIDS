//! Maps of a material's areal densities and temperature over the detector,
//! fitted as one counts fit whose regions are patches of pixels.

use std::ops::Range;

use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, Side};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, s};
use nereids_fitting::error::FittingError;
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::{
    InformationInverse, NEWTON_DECREMENT_TOL, Prior, half_deviance, information_inverse,
};
use nereids_fitting::statistics::{Consistency, consistency};
use rayon::prelude::*;

use crate::counts_fit::{
    BeamStart, Material, Measurement, Region, RegionFit, RegionTerms, Role, SHARED, Starts, Value,
    checked, fit_counts_answer, labelled, shared_parameters,
};
use crate::error::PipelineError;
use crate::open_beam::{
    BOUND, Calibration, PULSE_NUMBERS, Pulse, fit_open_beam, validate_live, whole,
};
use crate::pipeline::TEMPERATURE_BOUNDS_K;

const MOST_STEPS: usize = 20;

const MOST_HALVINGS: i32 = 10;

/// The counts of every pixel of an open-beam run and a sample run recorded in
/// the same time bins, which pixels to fit, and the material behind the
/// sample.
#[derive(Debug, Clone)]
pub struct MapMeasurement<'a> {
    /// Time-bin edges in µs.
    pub time_edges_us: Vec<f64>,
    /// `c_q`, the sample run's proton charge over the open-beam run's.
    pub charge_ratio: f64,
    /// `a`, the normalization of the sample run, the same in every pixel.
    pub normalization: Value,
    /// Raw counts of the open-beam run, in (time bin, row, column).
    pub open_counts: ArrayView3<'a, f64>,
    /// Raw counts of the sample run, in (time bin, row, column).
    pub sample_counts: ArrayView3<'a, f64>,
    /// The fraction of the open-beam run's neutrons arriving in each bin that
    /// every pixel records, in (0, 1]; `None` records every one.
    pub open_live: Option<Vec<f64>>,
    /// The fraction of the sample run's neutrons arriving in each bin that
    /// every pixel records, in (0, 1]; `None` records every one.
    pub sample_live: Option<Vec<f64>>,
    /// Pixels left out of both runs, in (row, column).
    pub excluded: ArrayView2<'a, bool>,
    /// Pixels behind the sample, in (row, column).
    pub sample: ArrayView2<'a, bool>,
    /// Pixels the beam reaches through nothing and the sample scatters no
    /// neutrons into, in (row, column).
    pub empty: ArrayView2<'a, bool>,
    /// The side of a square patch, in pixels.
    pub binning: usize,
    /// The material behind the sample: its isotopes' densities and its
    /// temperature, known, fitted or within bounds, in every patch.
    pub material: Material,
    /// `b0`, `b1` and `b2` of the background of every patch behind the
    /// sample, as [`Region::background`], known, fitted or within bounds.
    pub background: [Value; 3],
    /// `b0`, `b1` and `b2` of the background of the empty pixels.
    pub empty_background: [Value; 3],
}

impl MapMeasurement<'_> {
    /// Each patch's kind, in (patch row, patch column); patch `(i, j)` holds
    /// the pixels from `(i·binning, j·binning)` up to the next patch or the
    /// image's edge.
    ///
    /// # Errors
    /// [`PipelineError::ShapeMismatch`] unless both runs' counts have one time
    /// bin per pair of edges and the same rows and columns as the three
    /// masks; [`PipelineError::InvalidParameter`] if `binning` is 0.
    pub fn patches(&self) -> Result<Array2<Patch>, PipelineError> {
        let bins = self.time_edges_us.len().saturating_sub(1);
        let (time_bins, height, width) = self.open_counts.dim();
        if self.sample_counts.dim() != self.open_counts.dim() || time_bins != bins {
            return Err(PipelineError::ShapeMismatch(format!(
                "open-beam counts of shape {:?} and sample counts of shape {:?} for {bins} \
                 time bins",
                self.open_counts.dim(),
                self.sample_counts.dim()
            )));
        }
        for (name, mask) in [
            ("excluded", self.excluded),
            ("sample", self.sample),
            ("empty", self.empty),
        ] {
            if mask.dim() != (height, width) {
                return Err(PipelineError::ShapeMismatch(format!(
                    "the {name} mask has shape {:?} for {height}×{width} pixels",
                    mask.dim()
                )));
            }
        }
        let b = self.binning;
        if b == 0 {
            return Err(PipelineError::InvalidParameter(
                "a patch is at least one pixel wide".into(),
            ));
        }
        Ok(Array2::from_shape_fn(
            (height.div_ceil(b), width.div_ceil(b)),
            |patch| {
                let pixels = self.pixels(patch);
                let behind = pixels.iter().filter(|&&pixel| self.sample[pixel]).count();
                match (behind, pixels.len() - behind) {
                    (0, _) => Patch::Outside,
                    (_, 0) => Patch::Sample,
                    _ => Patch::Mixed,
                }
            },
        ))
    }

    fn pixels(&self, (i, j): (usize, usize)) -> Vec<(usize, usize)> {
        let (height, width) = self.excluded.dim();
        let b = self.binning;
        (i * b..height.min((i + 1) * b))
            .flat_map(|y| (j * b..width.min((j + 1) * b)).map(move |x| (y, x)))
            .filter(|&pixel| !self.excluded[pixel])
            .collect()
    }
}

/// What the pixels of a patch that are not excluded lie behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Patch {
    /// The sample, every one: the patch is fitted.
    Sample,
    /// The sample and something else: the patch is not fitted.
    Mixed,
    /// Not the sample, or every pixel is excluded.
    Outside,
}

/// The fitted map, one value per patch.
#[derive(Debug, Clone)]
pub struct CountsMap<'a> {
    measurement: MapMeasurement<'a>,
    /// Each patch's kind, as [`MapMeasurement::patches`].
    pub patches: Array2<Patch>,
    /// Each [`Patch::Sample`]'s fit, the shared quantities held at the map's
    /// answer; `None` where the patch is not fitted or [`Self::failed`].
    pub fits: Array2<Option<RegionFit>>,
    /// Why the fit of a [`Patch::Sample`] left out of the shared quantities at
    /// the end failed: an error, no convergence, or a fitted temperature at
    /// 1 K or 5000 K, at the shared values it was last fitted at, marked when
    /// those are a halved step not taken.  `None` where the patch is not left
    /// out or not fitted.
    pub failed: Array2<Option<String>>,
    /// The empty pixels' fit; `None` when every one is excluded, or when their
    /// refit at the map's shared values fails, which ends the map
    /// unconverged.
    pub empty: Option<RegionFit>,
    /// Each isotope's areal density in atoms/barn, in the material's order:
    /// the known one, or the fitted one; NaN where the patch has no fit.
    pub densities: Vec<Array2<f64>>,
    /// Standard deviation of each density, from [`Self::covariance`]; NaN
    /// where the density is known, held on a bound or not determined by the
    /// counts, alone or through a shared quantity, where the patch has no fit,
    /// or where the map did not converge.
    pub density_sd: Vec<Array2<f64>>,
    /// The temperature in K: the known one, or the fitted one; NaN where the
    /// patch has no fit.
    pub temperature_k: Array2<f64>,
    /// Standard deviation of the temperature in K, as [`Self::density_sd`].
    pub temperature_sd_k: Array2<f64>,
    /// Each patch's [`RegionFit::overdispersion`] of the open-beam run, then
    /// of the sample run; NaN where it is `None` or the patch has no fit.
    pub overdispersion: [Array2<f64>; 2],
    /// Whether the map converged, the patch did not fail, and each of its
    /// fitted densities and its fitted temperature has a finite standard
    /// deviation or, for a density, is 0.
    pub trusted: Array2<bool>,
    /// The normalization `a`: the known one, or the fitted one.
    pub normalization: f64,
    /// The timing offset `t0` in µs: the known one, or the fitted one.
    pub t0_us: f64,
    /// The flight path in m: the known one, or the fitted one.
    pub flight_path_m: f64,
    /// The pulse's `[α₀, α₁]`, each the known or the fitted one.
    pub alpha: [f64; 2],
    /// The pulse's `[β₀, β₁]`, each the known or the fitted one.
    pub beta: [f64; 2],
    /// The pulse's `R`: the known one, or the fitted one.
    pub r: f64,
    /// The square of the pulse's triangle FWHM, `h²` in µs²: the known one, or
    /// the fitted one.
    pub fwhm_squared_us2: f64,
    /// The fitted shared quantities, among the normalization, `t0`, flight
    /// path, `α₀`, `α₁`, `β₀`, `β₁`, `R` and `h²`, in that order.
    pub shared: Vec<&'static str>,
    /// Covariance of [`Self::shared`]: the inverse of the information on them
    /// of every patch that did not fail and of the empty pixels, each with
    /// its own quantities and beam profiled out, plus `1/sd²` for each measured
    /// shared quantity and the pulse calibration's prior.  The row and column
    /// of one on a bound or not determined by the counts are NaN, and the rest
    /// are conditional on one on a bound held there.  `None` when the map did
    /// not converge.
    pub shared_covariance: Option<FlatMatrix>,
    propagated: Option<FlatMatrix>,
    /// For each measured one of [`Self::shared`], in that order, its value
    /// without the bounds less its measurement over `√(sd² − variance)`, the
    /// variance also without the bounds, as
    /// [`CountsFit::measured_pulls`](crate::counts_fit::CountsFit::measured_pulls).
    /// `None` when the map did not converge, or when the fit of a patch or of
    /// the empty pixels gives no Gaussian without its bounds.
    pub measured_pulls: Option<Vec<f64>>,
    /// Whether the counts accept the pulse's calibration, as
    /// [`CountsFit::pulse_consistency`](crate::counts_fit::CountsFit::pulse_consistency),
    /// from [`Self::shared`] and the quantities of every patch not
    /// [`Self::failed`], without their bounds.  `None` without a calibration,
    /// when [`Self::measured_pulls`] is, or when [`consistency`] gives none.
    pub pulse_consistency: Option<Consistency>,
    /// A patch's fitted quantities: each fitted density, in the material's
    /// order, the temperature if fitted, and each fitted background term.
    pub quantities: Vec<String>,
    /// Each patch's covariance of [`Self::quantities`], the shared
    /// quantities' uncertainty included; the row and column of one held on a
    /// bound, or not determined by the counts, alone or through a shared
    /// quantity, are NaN.  `None` where the patch has no fit or the map has no
    /// shared covariance.
    pub covariance: Array2<Option<FlatMatrix>>,
    /// Each patch's change of [`Self::quantities`] per unit change of each of
    /// [`Self::shared`] at fixed counts, `−F_oo⁻¹ F_os` of its information;
    /// the row of a quantity held on a bound or not determined by the counts
    /// is NaN.  `None` where the patch
    /// has no fit.
    pub sensitivity: Array2<Option<FlatMatrix>>,
    /// Each fit's half Poisson deviance over the overdispersion it weighted
    /// each run with, summed over the patches with a fit and the empty pixels.
    pub deviance: f64,
    /// Whether the Newton decrement of the shared quantities fell below
    /// [`NEWTON_DECREMENT_TOL`].
    pub converged: bool,
    /// Why the map did not converge: the empty pixels' failure, the step
    /// limit, or a step none of whose halvings lowered the objective.  `None`
    /// when it converged.
    pub unconverged: Option<String>,
    /// Newton steps tried on the shared quantities, those not taken included.
    pub steps: usize,
}

impl CountsMap<'_> {
    /// For the open-beam run, then the sample run, the signed deviance
    /// residual `sign(y − μ)·√(2(y·ln(y/μ) + μ − y))` of each patch's counts
    /// `y` in the measurement the map was fitted to and its fit's predicted
    /// counts `μ` in each bin, for the patch rows `rows`, in (time bin, patch
    /// row from `rows.start`, patch column): `−√(2μ)` where `y` is 0, NaN
    /// where the patch has no fit.  Its squares over twice the overdispersion
    /// each run was weighted with, summed over every patch row with the empty
    /// pixels' alike, are [`Self::deviance`].
    ///
    /// # Errors
    /// [`PipelineError::InvalidParameter`] if `rows` is not an increasing
    /// range within the map's patch rows.
    pub fn residuals(&self, rows: Range<usize>) -> Result<[Array3<f64>; 2], PipelineError> {
        let (height, width) = self.patches.dim();
        if !(rows.start <= rows.end && rows.end <= height) {
            return Err(PipelineError::InvalidParameter(format!(
                "patch rows {rows:?} are not within the map's {height}"
            )));
        }
        let map = &self.measurement;
        let bins = map.open_counts.dim().0;
        let mut residuals = [
            Array3::from_elem((bins, rows.len(), width), f64::NAN),
            Array3::from_elem((bins, rows.len(), width), f64::NAN),
        ];
        for i in rows.clone() {
            for j in 0..width {
                let Some(fit) = &self.fits[[i, j]] else {
                    continue;
                };
                let counts =
                    summed(map, &map.pixels((i, j))).expect("the fit summed this patch's counts");
                for (run, block) in residuals.iter_mut().enumerate() {
                    for (k, (&y, &mu)) in counts[run].iter().zip(&fit.predicted[run]).enumerate() {
                        block[[k, i - rows.start, j]] =
                            (2.0 * half_deviance(y, mu)).sqrt().copysign(y - mu);
                    }
                }
            }
        }
        Ok(residuals)
    }

    /// The covariance of patch `a`'s [`Self::quantities`] with a different
    /// patch `b`'s, which only the shared quantities carry, `G_a C_ss G_bᵀ` over
    /// those not held on a bound; for `a` = `b`, the part of
    /// [`Self::covariance`] they carry.  `None` where either patch is outside
    /// the grid or has no [`Self::sensitivity`], or the map has no
    /// [`Self::shared_covariance`].
    pub fn cross_covariance(&self, a: (usize, usize), b: (usize, usize)) -> Option<FlatMatrix> {
        let left = self.sensitivity.get(a)?.as_ref()?;
        let right = self.sensitivity.get(b)?.as_ref()?;
        let shared = self.propagated.as_ref()?;
        let k = shared.nrows;
        let undetermined = |patch: (usize, usize), i: usize| {
            self.covariance[patch]
                .as_ref()
                .is_none_or(|covariance| covariance.get(i, i).is_nan())
        };
        let mut cross = FlatMatrix::zeros(left.nrows, right.nrows);
        for i in 0..left.nrows {
            for j in 0..right.nrows {
                *cross.get_mut(i, j) = if undetermined(a, i) || undetermined(b, j) {
                    f64::NAN
                } else {
                    (0..k)
                        .flat_map(|s| (0..k).map(move |t| (s, t)))
                        .map(|(s, t)| left.get(i, s) * shared.get(s, t) * right.get(j, t))
                        .sum()
                };
            }
        }
        Some(cross)
    }
}

/// Fit `map.material` in each patch of `binning`×`binning` pixels whose
/// pixels not excluded all lie behind the sample, and the empty pixels not
/// excluded as one region with no material, as one counts fit whose regions
/// share the normalization, `t0`, the flight path and the pulse: one clock,
/// one flight path, one moderator, and a detector efficiency that changes
/// between the runs alike in every pixel.  A region's counts are the sums
/// over its pixels; each has its own beam and background.  The empty pixels
/// pin the normalization when their `b0` is known.
///
/// The fit is the joint one of [`fit_counts`](crate::counts_fit::fit_counts),
/// found region by region.  With the shared quantities held, each region is
/// fitted alone, in parallel, and gives its information and slope on them
/// with its own quantities and beam profiled out; their sum, with the
/// calibration's prior and each measured shared quantity's, sets a Newton
/// step on the shared quantities, clamped
/// to their bounds, with one on a bound its slope pushes out of held there.
/// From its second fit on, a region keeps the beam intervals of its first,
/// starts from its last answer, and counts in its sample run's
/// overdispersion the leverage the shared quantities have on its bins, as
/// the joint fit does.  The steps stop when their Newton decrement falls
/// below [`NEWTON_DECREMENT_TOL`] at fits that counted that leverage.  A
/// step that raises the regions' summed deviance plus those priors'
/// penalty, each run weighted with the overdispersion of the step
/// before, or at which a fitted patch fails, is halved, up to ten times;
/// twenty steps, a step that cannot fall, or a failure of the empty pixels at
/// the smallest halving of a step or at their refit at the current values end
/// the map unconverged.  The shared information and each region's
/// information on its own quantities are inverted as [`information_inverse`]
/// inverts them, the shared one scaled to its diagonal before the regions'
/// own quantities are profiled out, with the rounding of one region's bins:
/// the steps take every direction the information spans, and a quantity
/// along a direction it does not determine, or moving along one through the
/// shared quantities, has a NaN covariance.  Each of the map's regions holds
/// its grid's spread to its share of the map's counts times [`BOUND`], so
/// their summed spread is within the `BOUND` the joint fit holds it to.
///
/// A patch whose fit fails, does not converge, or ends at a fitted
/// temperature of 1 K or 5000 K, at the starting shared values, at the
/// smallest halving of a step none of whose halvings is taken, or at a refit
/// at the current values, adds nothing to the shared quantities; it is fitted again after every step taken and
/// adds to them again once it fits.  A patch left out at the end is
/// [`CountsMap::failed`].
///
/// A patch has one density per isotope and one temperature: where its
/// pixels differ, these are the uniform fit of the patch's summed counts,
/// which differs from the pixels' beam-weighted means.
///
/// The map holds, for each patch, its summed counts and its fits at the
/// current and the trial shared values, so its memory is up to several
/// times that of the counts it is given; each region's grid gets finer as
/// the number of regions grows.
///
/// # Errors
/// Everything [`MapMeasurement::patches`] refuses;
/// [`PipelineError::InvalidParameter`] if a pixel is both behind the sample
/// and empty, a density, the temperature or a background term of the
/// sample's patches is [`Value::Measured`], which would count its
/// measurement once per patch, a term of the empty pixels' background is
/// [`Value::Measured`], a count of a pixel summed into a region is not a
/// whole non-negative number, or no patch is fitted; every refusal of
/// [`fit_counts`](crate::counts_fit::fit_counts) for the material, the
/// backgrounds, the shared quantities and the pulse's calibration;
/// [`PipelineError::Fitting`] if the pulse's calibration has a covariance
/// that cannot be inverted or an eigendecomposition fails;
/// [`PipelineError::ShapeMismatch`] or [`PipelineError::InvalidParameter`] for
/// live fractions `fit_counts` refuses; [`PipelineError::FlightTimeGrid`] if
/// the grid refuses the time edges at the starting shared values.
/// [`PipelineError::InvalidParameter`] naming the first failing patch and its
/// error when no patch fits at the starting shared values, or naming the
/// empty pixels when their fit fails there; [`PipelineError::InvalidParameter`]
/// with the last patch's reason when every patch is left out later.
pub fn fit_map<'a>(
    map: &MapMeasurement<'a>,
    calibration: &Calibration,
) -> Result<CountsMap<'a>, PipelineError> {
    let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
    let patches = map.patches()?;
    if map.sample.iter().zip(&map.empty).any(|(&s, &e)| s && e) {
        return invalid("a pixel is both behind the sample and empty".into());
    }
    let material = &map.material;
    if material
        .isotopes
        .iter()
        .map(|(_, density)| density)
        .chain([&material.temperature_k])
        .chain(&map.background)
        .any(|value| matches!(value, Value::Measured { .. }))
    {
        return invalid(
            "a measured density, temperature or background of the sample's patches would count \
             its measurement once per patch; give it known, fitted or within bounds"
                .into(),
        );
    }
    if map
        .empty_background
        .iter()
        .any(|value| matches!(value, Value::Measured { .. }))
    {
        return invalid(
            "a measured background of the empty pixels is not supported by the map's region by \
             region fit; give it known, fitted or within bounds"
                .into(),
        );
    }
    let fitted_patches: Vec<(usize, usize)> = patches
        .indexed_iter()
        .filter(|&(_, &kind)| kind == Patch::Sample)
        .map(|(patch, _)| patch)
        .collect();
    if fitted_patches.is_empty() {
        let b = map.binning;
        return invalid(format!(
            "no patch of {b}×{b} pixels has all its pixels not excluded behind the sample"
        ));
    }
    let empty: Vec<(usize, usize)> = map
        .empty
        .indexed_iter()
        .filter(|&(pixel, &e)| e && !map.excluded[pixel])
        .map(|(pixel, _)| pixel)
        .collect();
    let region = |pixels: &[(usize, usize)], background, material| {
        let [open_counts, sample_counts] = summed(map, pixels)?;
        Ok::<_, PipelineError>(Region {
            open_counts,
            sample_counts,
            open_live: map.open_live.clone(),
            sample_live: map.sample_live.clone(),
            background,
            material,
        })
    };
    let mut regions = fitted_patches
        .iter()
        .map(|&patch| region(&map.pixels(patch), map.background, Some(material.clone())))
        .collect::<Result<Vec<Region>, PipelineError>>()?;
    if !empty.is_empty() {
        regions.push(region(&empty, map.empty_background, None)?);
    }
    let measurement = |normalization, regions| Measurement {
        time_edges_us: map.time_edges_us.clone(),
        charge_ratio: map.charge_ratio,
        normalization,
        regions,
    };

    let template = |background, material| Region {
        open_counts: Vec::new(),
        sample_counts: Vec::new(),
        open_live: None,
        sample_live: None,
        background,
        material,
    };
    checked(
        &measurement(
            map.normalization,
            std::iter::once(template(map.background, Some(material.clone())))
                .chain((!empty.is_empty()).then(|| template(map.empty_background, None)))
                .collect(),
        ),
        calibration,
    )?;
    let parameters = shared_parameters(&measurement(map.normalization, vec![]), calibration)?;
    let fitted: Vec<usize> = (0..SHARED.len())
        .filter(|&i| !parameters[i].1.fixed)
        .collect();
    let roles: Vec<Role> = fitted.iter().map(|&i| SHARED[i]).collect();
    let k = fitted.len();
    let mut prior = Mat::<f64>::zeros(k, k);
    let mut center = vec![0.0; k];
    for (a, &i) in fitted.iter().enumerate() {
        if let Value::Measured { value, sd } = parameters[i].0 {
            prior[(a, a)] = 1.0 / (sd * sd);
            center[a] = value;
        }
    }
    let mut calibrated = None;
    if let Some(pulse) = &calibration.pulse.prior {
        let at: Vec<usize> = pulse
            .numbers
            .iter()
            .map(|&n| fitted.iter().position(|&i| SHARED[i] == Role::Pulse(n)))
            .collect::<Option<_>>()
            .expect("a calibrated number is fitted");
        let numbers = Prior::correlated(&at, &pulse.mean, &pulse.covariance)?;
        let inverse = Mat::from_fn(at.len(), at.len(), |a, b| pulse.covariance.get(a, b))
            .llt(Side::Lower)
            .map_err(|e| FittingError::EvaluationFailed(format!("{e:?}")))?
            .inverse();
        for (a, &i) in at.iter().enumerate() {
            center[i] = pulse.mean[a];
            for (b, &j) in at.iter().enumerate() {
                prior[(i, j)] = inverse[(a, b)];
            }
        }
        calibrated = Some((numbers, at));
    }
    let penalty = |values: &[f64]| -> (f64, Vec<f64>) {
        let offset: Vec<f64> = (0..k).map(|a| values[fitted[a]] - center[a]).collect();
        let slope: Vec<f64> = (0..k)
            .map(|a| (0..k).map(|b| prior[(a, b)] * offset[b]).sum())
            .collect();
        (
            0.5 * offset.iter().zip(&slope).map(|(o, s)| o * s).sum::<f64>(),
            slope,
        )
    };

    let mut values: Vec<f64> = parameters.iter().map(|(_, p)| p.value).collect();
    let origin_us = values[1];
    let bins = map.open_counts.dim().0;
    validate_live("open-beam", map.open_live.as_deref(), bins)?;
    validate_live("sample", map.sample_live.as_deref(), bins)?;
    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    let fitted_temperature = !matches!(material.temperature_k, Value::Known(_));
    let labels: Vec<String> = fitted_patches
        .iter()
        .map(|patch| format!("patch {patch:?}"))
        .chain((!empty.is_empty()).then(|| "the empty pixels".to_string()))
        .collect();
    let patch_count = fitted_patches.len();
    let held_at = |values: &[f64]| {
        let known = |i: usize| Value::Known(values[i]);
        Calibration {
            t0_us: known(1),
            flight_path_m: known(2),
            pulse: Pulse {
                alpha: [known(3), known(4)],
                beta: [known(5), known(6)],
                r: known(7),
                fwhm_squared_us2: known(8),
                prior: None,
                ..calibration.pulse.clone()
            },
        }
    };
    let starting = held_at(&values);
    let counted_in =
        |region: &Region| -> f64 { region.open_counts.iter().chain(&region.sample_counts).sum() };
    let all_counts: f64 = regions.iter().map(counted_in).sum();
    let budgets: Vec<f64> = regions
        .iter()
        .map(|region| BOUND * counted_in(region) / all_counts)
        .collect();
    let fit_region = |r: usize,
                      values: &[f64],
                      previous: Option<&RegionFit>,
                      shared_leverage: Option<f64>|
     -> Swept {
        let held = held_at(values);
        let label = &labels[r];
        let one = measurement(
            Value::Known(values[0]),
            vec![restarted(&regions[r], previous)],
        );
        let beam = match previous {
            Some(fit) => BeamStart {
                beam: fit.beam.clone(),
                overdispersion: fit.overdispersion[0],
                at_limit: fit.beam_at_limit,
            },
            None => fit_open_beam(
                &map.time_edges_us,
                &regions[r].open_counts,
                &starting,
                regions[r].open_live.as_deref(),
            )
            .map_err(|e| named(label, e))?
            .into(),
        };
        let starts = Starts {
            beams: vec![beam],
            origin_us,
            shared_leverage: shared_leverage.map(|extra| vec![extra]),
            bound: budgets[r],
        };
        let (fit, answer) =
            fit_counts_answer(&one, &held, Some(&starts)).map_err(|e| named(label, e))?;
        let region_fit = fit.regions.into_iter().next().expect("one region");
        let edge = fitted_temperature
            && region_fit
                .temperature_k
                .is_some_and(|t| t == t_low || t == t_high);
        if edge {
            return Err(PipelineError::InvalidParameter(format!(
                "{label}: its temperature ended at {t_low} K or {t_high} K"
            )));
        }
        if !fit.converged {
            return Err(PipelineError::InvalidParameter(format!(
                "{label}: its fit did not converge"
            )));
        }
        let mut terms = answer
            .region_terms(0, &roles)
            .map_err(|e| named(label, e))?;
        if fit.unbounded.is_none() {
            terms.unbounded = None;
        }
        Ok(Inner {
            fit: region_fit,
            deviance: fit.deviance,
            terms,
        })
    };
    let sweep = |values: &[f64],
                 previous: &[Option<RegionFit>],
                 leverage: &[Option<f64>],
                 failures: &[Option<String>],
                 failed: bool|
     -> Vec<Option<Swept>> {
        (0..regions.len())
            .into_par_iter()
            .map(|r| {
                (failures[r].is_some() == failed)
                    .then(|| fit_region(r, values, previous[r].as_ref(), leverage[r]))
            })
            .collect()
    };
    let objective = |inner: &[Option<Inner>], weights: &[Option<Inner>]| -> f64 {
        inner
            .iter()
            .zip(weights)
            .zip(&regions)
            .filter_map(|((inner, weights), region)| {
                Some((inner.as_ref()?, weights.as_ref()?, region))
            })
            .map(|(inner, weights, region)| {
                [&region.open_counts, &region.sample_counts]
                    .into_iter()
                    .zip(&inner.fit.predicted)
                    .enumerate()
                    .map(|(run, (counts, predicted))| {
                        counts
                            .iter()
                            .zip(predicted)
                            .map(|(&y, &mu)| half_deviance(y, mu))
                            .sum::<f64>()
                            / weight(&weights.fit, run)
                    })
                    .sum::<f64>()
            })
            .sum()
    };

    let mut failures: Vec<Option<String>> = vec![None; regions.len()];
    let mut leverage: Vec<Option<f64>> = vec![None; regions.len()];
    let mut previous: Vec<Option<RegionFit>> = vec![None; regions.len()];
    let mut inner: Vec<Option<Inner>> = Vec::with_capacity(regions.len());
    let mut first_failure = None;
    for (r, swept) in sweep(&values, &previous, &leverage, &failures, false)
        .into_iter()
        .enumerate()
    {
        match swept.expect("every region is fitted at the start") {
            Ok(one) => inner.push(Some(one)),
            Err(error) if r >= patch_count => return Err(error),
            Err(error) => {
                failures[r] = Some(error.to_string());
                first_failure.get_or_insert(error);
                inner.push(None);
            }
        }
    }
    if inner[..patch_count].iter().all(Option::is_none) {
        return Err(first_failure.expect("a patch failed"));
    }
    let absorb = |swept: Vec<Option<Swept>>,
                  inner: &mut [Option<Inner>],
                  failures: &mut [Option<String>]| {
        let mut joined = false;
        for (r, one) in swept.into_iter().enumerate() {
            match one {
                Some(Ok(one)) => {
                    joined |= failures[r].is_some();
                    failures[r] = None;
                    inner[r] = Some(one);
                }
                Some(Err(error)) => {
                    failures[r] = Some(error.to_string());
                    inner[r] = None;
                }
                None => {}
            }
        }
        joined
    };
    let none_left = |failures: &[Option<String>]| {
        let reason = failures[..patch_count].iter().flatten().last();
        invalid(format!(
            "every patch of the map failed; the last: {}",
            reason.map_or("", String::as_str)
        ))
    };
    let counted = 2 * bins;
    let inverse_over = |total: &Mat<f64>, diagonal: &[f64], set: &[usize]| {
        let mut block = FlatMatrix::zeros(set.len(), set.len());
        for (a, &i) in set.iter().enumerate() {
            for (b, &j) in set.iter().enumerate() {
                *block.get_mut(a, b) = total[(i, j)];
            }
        }
        let diagonal: Vec<f64> = set.iter().map(|&i| diagonal[i]).collect();
        information_inverse(&block, &diagonal, counted)
    };
    let mut steps = 0;
    let mut converged = false;
    let mut unconverged = None;
    let mut weighted = false;
    let information = loop {
        for (kept, inner) in previous.iter_mut().zip(&inner) {
            if let Some(inner) = inner {
                *kept = Some(inner.fit.clone());
            }
        }
        let (_, prior_slope) = penalty(&values);
        let mut gradient = prior_slope;
        let mut total = prior.clone();
        let mut diagonal: Vec<f64> = (0..k).map(|a| prior[(a, a)]).collect();
        for terms in inner.iter().flatten().map(|inner| &inner.terms) {
            for a in 0..k {
                gradient[a] += terms.gradient[a];
                diagonal[a] += terms.diagonal[a];
                for b in 0..k {
                    total[(a, b)] += terms.information.get(a, b);
                }
            }
        }
        let free: Vec<usize> = (0..k)
            .filter(|&a| {
                let (value, parameter) = (values[fitted[a]], &parameters[fitted[a]].1);
                !(value == parameter.lower && gradient[a] > 0.0
                    || value == parameter.upper && gradient[a] < 0.0)
            })
            .collect();
        let off_bounds: Vec<usize> = (0..k)
            .filter(|&a| {
                let (value, parameter) = (values[fitted[a]], &parameters[fitted[a]].1);
                value != parameter.lower && value != parameter.upper
            })
            .collect();
        let spread = inverse_over(&total, &diagonal, &off_bounds)?.spanned;
        for (shared_leverage, inner) in leverage.iter_mut().zip(&inner) {
            if let Some(Inner { terms, .. }) = inner {
                *shared_leverage = Some(
                    (0..off_bounds.len())
                        .flat_map(|a| (0..off_bounds.len()).map(move |b| (a, b)))
                        .map(|(a, b)| {
                            terms.counted_information.get(off_bounds[a], off_bounds[b])
                                * spread.get(a, b)
                        })
                        .sum(),
                );
            }
        }
        let covariance = inverse_over(&total, &diagonal, &free)?.spanned;
        let step: Vec<f64> = (0..free.len())
            .map(|a| {
                -(0..free.len())
                    .map(|b| covariance.get(a, b) * gradient[free[b]])
                    .sum::<f64>()
            })
            .collect();
        let decrement: f64 = -0.5
            * (0..free.len())
                .map(|a| gradient[free[a]] * step[a])
                .sum::<f64>();
        if decrement < NEWTON_DECREMENT_TOL {
            if weighted {
                converged = true;
                let mut unbounded = Some((prior.clone(), penalty(&values).1));
                for terms in inner.iter().flatten().map(|inner| &inner.terms) {
                    match (&mut unbounded, &terms.unbounded) {
                        (Some((total, slope)), Some((information, region_slope))) => {
                            for a in 0..k {
                                slope[a] += region_slope[a];
                                for b in 0..k {
                                    total[(a, b)] += information.get(a, b);
                                }
                            }
                        }
                        _ => unbounded = None,
                    }
                }
                break Some((total, diagonal, unbounded));
            }
            weighted = true;
            let swept = sweep(&values, &previous, &leverage, &failures, false);
            absorb(swept, &mut inner, &mut failures);
            if let Some(reason) = failures[patch_count..].iter().flatten().next() {
                unconverged = Some(reason.clone());
                break None;
            }
            if inner[..patch_count].iter().all(Option::is_none) {
                return none_left(&failures);
            }
            continue;
        }
        if steps == MOST_STEPS {
            unconverged = Some(format!("{MOST_STEPS} Newton steps did not converge"));
            break None;
        }
        steps += 1;
        let before = objective(&inner, &inner) + penalty(&values).0;
        let mut lost: Vec<(usize, String)> = Vec::new();
        let accepted = (0..=MOST_HALVINGS).find_map(|halving| {
            let mut trial = values.clone();
            for (a, &f) in free.iter().enumerate() {
                let parameter = &parameters[fitted[f]].1;
                trial[fitted[f]] = (values[fitted[f]] + step[a] / f64::powi(2.0, halving))
                    .clamp(parameter.lower, parameter.upper);
            }
            let swept = sweep(&trial, &previous, &leverage, &failures, false);
            lost = swept
                .iter()
                .enumerate()
                .filter_map(|(r, one)| match one {
                    Some(Err(error)) => Some((r, error.to_string())),
                    _ => None,
                })
                .collect();
            if !lost.is_empty() {
                return None;
            }
            let candidate: Vec<Option<Inner>> = swept
                .into_iter()
                .map(|one| one.and_then(Result::ok))
                .collect();
            (objective(&candidate, &inner) + penalty(&trial).0 <= before)
                .then_some((trial, candidate))
        });
        match accepted {
            Some((trial, candidate)) => {
                values = trial;
                inner = candidate;
                let retried = sweep(&values, &previous, &leverage, &failures, true);
                weighted = !absorb(retried, &mut inner, &mut failures);
            }
            None if !lost.is_empty() && lost.iter().all(|&(r, _)| r < patch_count) => {
                for (r, reason) in lost {
                    failures[r] = Some(format!("{reason} (at a halved step not taken)"));
                    inner[r] = None;
                }
                weighted = false;
                if inner[..patch_count].iter().all(Option::is_none) {
                    return none_left(&failures);
                }
            }
            None => {
                let empty_first = lost
                    .iter()
                    .position(|&(r, _)| r >= patch_count)
                    .unwrap_or(0);
                unconverged = Some(lost.into_iter().nth(empty_first).map_or_else(
                    || "no halving of a Newton step lowered the objective".to_string(),
                    |(_, reason)| reason,
                ));
                break None;
            }
        }
    };

    let interior: Vec<usize> = (0..k)
        .filter(|&a| {
            let (value, parameter) = (values[fitted[a]], &parameters[fitted[a]].1);
            value != parameter.lower && value != parameter.upper
        })
        .collect();
    let embedded = |set: &[usize], inverse: &InformationInverse| {
        let mut covariance = FlatMatrix::zeros(k, k);
        covariance.data.fill(f64::NAN);
        for (a, &i) in set.iter().enumerate() {
            for (b, &j) in set.iter().enumerate() {
                if inverse.resolved[a] && inverse.resolved[b] {
                    *covariance.get_mut(i, j) = inverse.determined.get(a, b);
                }
            }
        }
        covariance
    };
    let interior_inverse = match &information {
        Some((total, diagonal, _)) => Some(inverse_over(total, diagonal, &interior)?),
        None => None,
    };
    let shared_covariance = interior_inverse
        .as_ref()
        .map(|inverse| embedded(&interior, inverse));
    let propagated = interior_inverse
        .as_ref()
        .map(|inverse| propagated(k, &interior, inverse));
    let null: Vec<Vec<f64>> = interior_inverse
        .iter()
        .flat_map(|inverse| &inverse.undetermined)
        .map(|direction| {
            let mut along = vec![0.0; k];
            for (a, &i) in interior.iter().enumerate() {
                along[i] = direction[a];
            }
            along
        })
        .collect();
    let every: Vec<usize> = (0..k).collect();
    let unbounded = match &information {
        Some((_, diagonal, Some((total, gradient)))) => {
            let inverse = inverse_over(total, diagonal, &every)?;
            let mean: Vec<f64> = (0..k)
                .map(|a| {
                    if inverse.resolved[a] {
                        values[fitted[a]]
                            - (0..k)
                                .map(|b| inverse.determined.get(a, b) * gradient[b])
                                .sum::<f64>()
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            Some((mean, embedded(&every, &inverse)))
        }
        _ => None,
    };
    let pulse_consistency = match (&calibrated, &unbounded) {
        (Some((numbers, at)), Some((mean, inverse))) => {
            let mut posterior = FlatMatrix::zeros(at.len(), at.len());
            for (a, &i) in at.iter().enumerate() {
                for (b, &j) in at.iter().enumerate() {
                    *posterior.get_mut(a, b) = inverse.get(i, j);
                }
            }
            let estimate: Vec<f64> = at.iter().map(|&i| mean[i]).collect();
            consistency(numbers, &estimate, &posterior)?
        }
        _ => None,
    };
    let measured_pulls = unbounded.as_ref().map(|(mean, inverse)| {
        (0..k)
            .filter_map(|a| match parameters[fitted[a]].0 {
                Value::Measured { value, sd } => {
                    Some((mean[a] - value) / (sd * sd - inverse.get(a, a)).sqrt())
                }
                _ => None,
            })
            .collect()
    });
    let template: Vec<(Role, String)> = material
        .isotopes
        .iter()
        .enumerate()
        .filter(|(_, (_, density))| !matches!(density, Value::Known(_)))
        .map(|(isotope, (data, _))| {
            (
                Role::Density { region: 0, isotope },
                format!("density of {}", data.isotope),
            )
        })
        .chain(
            (!matches!(material.temperature_k, Value::Known(_)))
                .then(|| (Role::Temperature { region: 0 }, "temperature".to_string())),
        )
        .chain(
            map.background
                .iter()
                .enumerate()
                .filter(|(_, value)| !matches!(value, Value::Known(_)))
                .map(|(term, _)| (Role::Background { region: 0, term }, format!("b{term}"))),
        )
        .collect();
    let q = template.len();

    let (rows, cols) = patches.dim();
    let blank = Array2::from_elem((rows, cols), f64::NAN);
    let isotopes = material.isotopes.len();
    let mut densities = vec![blank.clone(); isotopes];
    let mut density_sd = vec![blank.clone(); isotopes];
    let mut temperature_k = blank.clone();
    let mut temperature_sd_k = blank.clone();
    let mut overdispersion = [blank.clone(), blank];
    let mut trusted = Array2::from_elem((rows, cols), false);
    let mut failed = Array2::from_elem((rows, cols), None);
    let mut fits = Array2::from_elem((rows, cols), None);
    let mut covariance = Array2::from_elem((rows, cols), None);
    let mut sensitivity = Array2::from_elem((rows, cols), None);
    for (r, &patch) in fitted_patches.iter().enumerate() {
        let Some(Inner { fit, terms, .. }) = &inner[r] else {
            failed[patch] = failures[r].clone();
            continue;
        };
        let rows_of: Vec<Option<usize>> = template
            .iter()
            .map(|(role, _)| terms.quantities.iter().position(|q| q == role))
            .collect();
        let mut slopes = FlatMatrix::zeros(q, k);
        for (a, row) in rows_of.iter().enumerate() {
            for s in 0..k {
                *slopes.get_mut(a, s) = row.map_or(f64::NAN, |i| terms.sensitivity.get(i, s));
            }
        }
        let marginal = propagated.as_ref().map(|shared| {
            let mut marginal = FlatMatrix::zeros(q, q);
            for (a, row_a) in rows_of.iter().enumerate() {
                for (b, row_b) in rows_of.iter().enumerate() {
                    *marginal.get_mut(a, b) = match (row_a, row_b) {
                        (Some(i), Some(j)) => marginal_entry(
                            &terms.covariance,
                            &terms.sensitivity,
                            shared,
                            &null,
                            (*i, *j),
                        ),
                        _ => f64::NAN,
                    };
                }
            }
            marginal
        });
        let sd = |role: Role| {
            template
                .iter()
                .position(|(r, _)| *r == role)
                .zip(marginal.as_ref())
                .map_or(f64::NAN, |(a, marginal)| marginal.get(a, a).sqrt())
        };
        for (m, &density) in fit.densities.iter().enumerate() {
            densities[m][patch] = density;
            density_sd[m][patch] = sd(Role::Density {
                region: 0,
                isotope: m,
            });
        }
        temperature_k[patch] = fit.temperature_k.unwrap_or(f64::NAN);
        temperature_sd_k[patch] = sd(Role::Temperature { region: 0 });
        for (run, phi) in overdispersion.iter_mut().zip(fit.overdispersion) {
            run[patch] = phi.unwrap_or(f64::NAN);
        }
        trusted[patch] = converged
            && template.iter().all(|&(role, _)| match role {
                Role::Density { isotope, .. } => {
                    sd(role).is_finite() || fit.densities[isotope] == 0.0
                }
                Role::Temperature { .. } => sd(role).is_finite(),
                _ => true,
            });
        fits[patch] = Some(fit.clone());
        covariance[patch] = marginal;
        sensitivity[patch] = Some(slopes);
    }

    Ok(CountsMap {
        measurement: map.clone(),
        patches,
        fits,
        failed,
        empty: (!empty.is_empty())
            .then(|| inner.last().and_then(|i| i.as_ref().map(|i| i.fit.clone())))
            .flatten(),
        densities,
        density_sd,
        temperature_k,
        temperature_sd_k,
        overdispersion,
        trusted,
        normalization: values[0],
        t0_us: values[1],
        flight_path_m: values[2],
        alpha: [values[3], values[4]],
        beta: [values[5], values[6]],
        r: values[7],
        fwhm_squared_us2: values[8],
        shared: fitted
            .iter()
            .map(|&i| {
                ["normalization", "t0", "flight path"]
                    .into_iter()
                    .chain(PULSE_NUMBERS)
                    .nth(i)
                    .expect("a shared quantity")
            })
            .collect(),
        shared_covariance,
        propagated,
        measured_pulls,
        pulse_consistency,
        quantities: template.into_iter().map(|(_, name)| name).collect(),
        covariance,
        sensitivity,
        deviance: inner.iter().flatten().map(|i| i.deviance).sum(),
        converged,
        unconverged,
        steps,
    })
}

type Swept = Result<Inner, PipelineError>;

struct Inner {
    fit: RegionFit,
    deviance: f64,
    terms: RegionTerms,
}

fn propagated(k: usize, interior: &[usize], inverse: &InformationInverse) -> FlatMatrix {
    let mut propagated = FlatMatrix::zeros(k, k);
    for (a, &i) in interior.iter().enumerate() {
        for (b, &j) in interior.iter().enumerate() {
            *propagated.get_mut(i, j) = inverse.determined.get(a, b);
        }
    }
    propagated
}

fn marginal_entry(
    covariance: &FlatMatrix,
    sensitivity: &FlatMatrix,
    shared: &FlatMatrix,
    null: &[Vec<f64>],
    (i, j): (usize, usize),
) -> f64 {
    let coupled = |i: usize| moves_along(|s| sensitivity.get(i, s), null);
    if coupled(i) || coupled(j) {
        return f64::NAN;
    }
    let k = shared.nrows;
    covariance.get(i, j)
        + (0..k)
            .flat_map(|s| (0..k).map(move |t| (s, t)))
            .map(|(s, t)| sensitivity.get(i, s) * shared.get(s, t) * sensitivity.get(j, t))
            .sum::<f64>()
}

fn moves_along(sensitivity: impl Fn(usize) -> f64, null: &[Vec<f64>]) -> bool {
    null.iter().any(|direction| {
        let terms: Vec<f64> = direction
            .iter()
            .enumerate()
            .map(|(s, &v)| sensitivity(s) * v)
            .collect();
        let magnitude: f64 = terms.iter().map(|t| t.abs()).sum();
        terms.iter().sum::<f64>().abs() > f64::EPSILON.sqrt() * magnitude
    })
}

fn named(label: &str, error: PipelineError) -> PipelineError {
    match error {
        PipelineError::InvalidParameter(_) | PipelineError::ShapeMismatch(_) => {
            labelled(label, error)
        }
        other => PipelineError::InvalidParameter(format!("{label}: {other}")),
    }
}

fn weight(fit: &RegionFit, run: usize) -> f64 {
    fit.overdispersion[run]
        .or(fit.overdispersion[0])
        .unwrap_or(1.0)
}

fn restarted(region: &Region, previous: Option<&RegionFit>) -> Region {
    let mut region = region.clone();
    let Some(fit) = previous else {
        return region;
    };
    if let Some(material) = &mut region.material {
        for ((_, density), &at) in material.isotopes.iter_mut().zip(&fit.densities) {
            *density = restart(*density, at);
        }
        if let Some(t) = fit.temperature_k {
            material.temperature_k = restart(material.temperature_k, t);
        }
    }
    for (term, &at) in region.background.iter_mut().zip(&fit.background) {
        *term = restart(*term, at);
    }
    region
}

fn restart(value: Value, at: f64) -> Value {
    match value {
        Value::Fitted(_) => Value::Fitted(at),
        Value::Within { lower, upper, .. } => Value::Within {
            start: at,
            lower,
            upper,
        },
        Value::Known(_) | Value::Measured { .. } => value,
    }
}

fn summed(
    map: &MapMeasurement<'_>,
    pixels: &[(usize, usize)],
) -> Result<[Vec<f64>; 2], PipelineError> {
    let bins = map.open_counts.dim().0;
    let mut sums = [vec![0.0; bins], vec![0.0; bins]];
    for ((run, counts), sum) in [
        ("open-beam", map.open_counts),
        ("sample", map.sample_counts),
    ]
    .into_iter()
    .zip(&mut sums)
    {
        for &(y, x) in pixels {
            for (k, (&count, total)) in counts
                .slice(s![.., y, x])
                .iter()
                .zip(sum.iter_mut())
                .enumerate()
            {
                if !whole(count) {
                    return Err(PipelineError::InvalidParameter(format!(
                        "{run} counts must be whole non-negative numbers, got {count} in bin \
                         {k} of pixel ({y}, {x})"
                    )));
                }
                *total += count;
            }
        }
    }
    Ok(sums)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patch_quantity_has_the_joint_fits_variance_under_an_oblique_shared_degeneracy() {
        let own = [1.0, 2.0, 0.0, 1.0, 3.0];
        let first = [1.0, 0.0, 1.0, 2.0, 1.0];
        for (column, resolved) in [(own, true), (first, false)] {
            let jacobian: Vec<[f64; 3]> = (0..5)
                .map(|row| [column[row], first[row], 2.0 * first[row]])
                .collect();
            let information =
                |a: usize, b: usize| -> f64 { jacobian.iter().map(|row| row[a] * row[b]).sum() };
            let joint = FlatMatrix {
                data: (0..9).map(|e| information(e / 3, e % 3)).collect(),
                nrows: 3,
                ncols: 3,
            };
            let diagonal: Vec<f64> = (0..3).map(|a| information(a, a)).collect();
            let oracle = information_inverse(&joint, &diagonal, 5).unwrap();
            let sensitivity = FlatMatrix {
                data: (1..3)
                    .map(|s| -information(0, s) / information(0, 0))
                    .collect(),
                nrows: 1,
                ncols: 2,
            };
            let profiled = FlatMatrix {
                data: (0..4)
                    .map(|e| {
                        let (s, t) = (1 + e / 2, 1 + e % 2);
                        information(s, t)
                            - information(s, 0) * information(0, t) / information(0, 0)
                    })
                    .collect(),
                nrows: 2,
                ncols: 2,
            };
            let shared = information_inverse(&profiled, &diagonal[1..], 5).unwrap();
            let own_covariance = FlatMatrix {
                data: vec![1.0 / information(0, 0)],
                nrows: 1,
                ncols: 1,
            };
            let variance = marginal_entry(
                &own_covariance,
                &sensitivity,
                &propagated(2, &[0, 1], &shared),
                &shared.undetermined,
                (0, 0),
            );
            assert_eq!(oracle.resolved[0], resolved);
            if resolved {
                let expected = oracle.determined.get(0, 0);
                assert!(
                    (variance / expected - 1.0).abs() <= 1e-9,
                    "{variance} vs {expected}"
                );
            } else {
                assert!(variance.is_nan(), "{variance}");
            }
        }
    }

    #[test]
    fn a_quantity_moves_along_a_null_direction_unless_its_slopes_cancel_on_it() {
        let null = vec![vec![1.0, -2.0]];
        assert!(moves_along(|s| [1.0, 0.0][s], &null));
        assert!(!moves_along(|s| [2.0, 1.0][s], &null));
        assert!(!moves_along(|_| 0.0, &null));
    }
}
