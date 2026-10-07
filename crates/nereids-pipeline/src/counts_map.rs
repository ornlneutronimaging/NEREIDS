//! Maps of a material's areal densities and temperature over the detector,
//! fitted as one counts fit whose regions are patches of pixels.

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Mat, Side};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, s};
use nereids_fitting::error::FittingError;
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::{NEWTON_DECREMENT_TOL, Prior, half_deviance};
use rayon::prelude::*;

use crate::counts_fit::{
    BeamStart, Material, Measurement, Region, RegionFit, RegionTerms, Role, SHARED, Starts, Value,
    fit_counts_answer, labelled, shared_parameters,
};
use crate::error::PipelineError;
use crate::open_beam::{Calibration, PULSE_NUMBERS, Pulse, whole};
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
pub struct CountsMap {
    /// Each patch's kind, as [`MapMeasurement::patches`].
    pub patches: Array2<Patch>,
    /// Each [`Patch::Sample`]'s fit, the shared quantities held at the map's
    /// answer; `None` where the patch is not fitted or [`Self::failed`].
    pub fits: Array2<Option<RegionFit>>,
    /// Whether the fit of a [`Patch::Sample`] failed, did not converge, or
    /// ended at a temperature of 1 K or 5000 K: the patch adds nothing to the
    /// shared quantities.
    pub failed: Array2<bool>,
    /// The empty pixels' fit; `None` when every one is excluded, or its fit
    /// failed.
    pub empty: Option<RegionFit>,
    /// Each isotope's areal density in atoms/barn, in the material's order:
    /// the known one, or the fitted one; NaN where the patch has no fit.
    pub densities: Vec<Array2<f64>>,
    /// Standard deviation of each density, from [`Self::covariance`]; NaN
    /// where the density is known or held on a bound, or the patch has no
    /// fit.
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
    /// For the open-beam run, then the sample run, the signed deviance
    /// residual `sign(y − μ)·√(2(y·ln(y/μ) + μ − y))` of each patch's counts
    /// `y` and its fit's predicted counts `μ` in each bin, in (time bin, patch
    /// row, patch column): `−√(2μ)` where `y` is 0, NaN where the patch has no
    /// fit.  Its squares over twice the overdispersion each run was weighted
    /// with, summed with the empty pixels' alike, are [`Self::deviance`].
    pub residuals: [Array3<f64>; 2],
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
    /// its own quantities and beam profiled out, plus the calibration's
    /// prior.  The row and column of one held on a bound are NaN.  `None`
    /// when the map did not converge.
    pub shared_covariance: Option<FlatMatrix>,
    /// A patch's fitted quantities: each fitted density, in the material's
    /// order, the temperature if fitted, and each fitted background term.
    pub quantities: Vec<String>,
    /// Each patch's covariance of [`Self::quantities`], the shared
    /// quantities' uncertainty included; the row and column of one held on a
    /// bound are NaN.  `None` where the patch has no fit or the map has no
    /// shared covariance.
    pub covariance: Array2<Option<FlatMatrix>>,
    /// Each patch's change of [`Self::quantities`] per unit change of each of
    /// [`Self::shared`] at fixed counts, `−F_oo⁻¹ F_os` of its information;
    /// the row of a quantity held on a bound is NaN.  `None` where the patch
    /// has no fit.
    pub sensitivity: Array2<Option<FlatMatrix>>,
    /// Each fit's half Poisson deviance over the overdispersion it weighted
    /// each run with, summed over the patches with a fit and the empty pixels.
    pub deviance: f64,
    /// Whether the Newton decrement of the shared quantities fell below
    /// [`NEWTON_DECREMENT_TOL`].
    pub converged: bool,
    /// Newton steps taken on the shared quantities.
    pub steps: usize,
}

impl CountsMap {
    /// The covariance of patch `a`'s [`Self::quantities`] with patch `b`'s,
    /// which only the shared quantities carry, `G_a C_ss G_bᵀ` over those not
    /// held on a bound; `None` where either patch has no
    /// [`Self::sensitivity`] or the map no [`Self::shared_covariance`].
    pub fn cross_covariance(&self, a: (usize, usize), b: (usize, usize)) -> Option<FlatMatrix> {
        let (left, right) = (self.sensitivity[a].as_ref()?, self.sensitivity[b].as_ref()?);
        let shared = self.shared_covariance.as_ref()?;
        let free: Vec<usize> = (0..shared.nrows)
            .filter(|&i| shared.get(i, i).is_finite())
            .collect();
        let mut cross = FlatMatrix::zeros(left.nrows, right.nrows);
        for i in 0..left.nrows {
            for j in 0..right.nrows {
                *cross.get_mut(i, j) = free
                    .iter()
                    .flat_map(|&s| free.iter().map(move |&t| (s, t)))
                    .map(|(s, t)| left.get(i, s) * shared.get(s, t) * right.get(j, t))
                    .sum();
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
/// calibration's prior, sets a Newton step on the shared quantities, clamped
/// to their bounds, with one on a bound its slope pushes out of held there.
/// The steps stop when their Newton decrement falls below
/// [`NEWTON_DECREMENT_TOL`].  A step that raises the regions' summed
/// deviance, each run weighted with the overdispersion of the step before,
/// is halved, up to ten times; twenty steps, or a step that cannot fall,
/// end the map unconverged.  From its second fit on, a region keeps the beam
/// intervals of its first and starts from its last answer.  A region's own
/// grid meets its own rule, so the answer agrees with the joint fit to the
/// grids' [`BOUND`](crate::open_beam::BOUND).
///
/// A patch whose fit fails, does not converge, or ends at a temperature of
/// 1 K or 5000 K is [`CountsMap::failed`] and adds nothing to the shared
/// quantities.  A patch has one density per isotope and one temperature:
/// where its pixels differ, these are the uniform fit of the patch's summed
/// counts, which differs from the pixels' beam-weighted means.
///
/// # Errors
/// Everything [`MapMeasurement::patches`] refuses;
/// [`PipelineError::InvalidParameter`] if a pixel is both behind the sample
/// and empty, a density, the temperature or a background term of the
/// sample's patches is [`Value::Measured`], which would count its
/// measurement once per patch, a count of a pixel summed into a region is
/// not a whole non-negative number, no patch is fitted, or every region's
/// fit fails; a shared quantity's value, bounds or measurement, or the
/// pulse's calibration, that [`fit_counts`](crate::counts_fit::fit_counts)
/// refuses.
pub fn fit_map(
    map: &MapMeasurement<'_>,
    calibration: &Calibration,
) -> Result<CountsMap, PipelineError> {
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
    if let Some(pulse) = &calibration.pulse.prior {
        let at: Vec<usize> = pulse
            .numbers
            .iter()
            .map(|&n| fitted.iter().position(|&i| SHARED[i] == Role::Pulse(n)))
            .collect::<Option<_>>()
            .expect("a calibrated number is fitted");
        Prior::correlated(&at, &pulse.mean, &pulse.covariance)?;
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
    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    let labels: Vec<String> = fitted_patches
        .iter()
        .map(|patch| format!("patch {patch:?}"))
        .chain((!empty.is_empty()).then(|| "the empty pixels".to_string()))
        .collect();
    let sweep = |values: &[f64], previous: &[Option<RegionFit>]| -> Vec<Swept> {
        let known = |i: usize| Value::Known(values[i]);
        let held = Calibration {
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
        };
        regions
            .par_iter()
            .zip(previous)
            .zip(&labels)
            .map(|((region, previous), label)| {
                let one = measurement(known(0), vec![restarted(region, previous.as_ref())]);
                let starts = previous.as_ref().map(|fit| Starts {
                    beams: vec![BeamStart {
                        beam: fit.beam.clone(),
                        overdispersion: fit.overdispersion[0],
                        at_limit: fit.beam_at_limit,
                    }],
                    origin_us,
                });
                let (fit, answer) = fit_counts_answer(&one, &held, starts.as_ref())
                    .map_err(|e| labelled(label, e))?;
                let terms = answer
                    .region_terms(0, &roles)
                    .map_err(|e| labelled(label, e))?;
                let region_fit = fit.regions.into_iter().next().expect("one region");
                let edge = region_fit
                    .temperature_k
                    .is_some_and(|t| t == t_low || t == t_high);
                Ok((fit.converged && !edge).then_some(Inner {
                    fit: region_fit,
                    deviance: fit.deviance,
                    terms,
                }))
            })
            .collect()
    };
    let survivors = |swept: Vec<Swept>| -> Result<Vec<Option<Inner>>, PipelineError> {
        if swept.iter().all(|one| !matches!(one, Ok(Some(_)))) {
            return Err(swept.into_iter().find_map(Result::err).unwrap_or_else(|| {
                PipelineError::InvalidParameter("the fit of no region of the map converged".into())
            }));
        }
        Ok(swept.into_iter().map(|one| one.ok().flatten()).collect())
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

    let mut previous: Vec<Option<RegionFit>> = vec![None; regions.len()];
    let mut inner = survivors(sweep(&values, &previous))?;
    let mut steps = 0;
    let mut converged = false;
    let information = loop {
        for (kept, inner) in previous.iter_mut().zip(&inner) {
            if let Some(inner) = inner {
                *kept = Some(inner.fit.clone());
            }
        }
        let (_, prior_slope) = penalty(&values);
        let mut gradient = prior_slope;
        let mut total = prior.clone();
        for terms in inner.iter().flatten().map(|inner| &inner.terms) {
            for a in 0..k {
                gradient[a] += terms.gradient[a];
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
        let Ok(factor) =
            Mat::from_fn(free.len(), free.len(), |a, b| total[(free[a], free[b])]).llt(Side::Lower)
        else {
            break None;
        };
        let step = factor.solve(Mat::from_fn(free.len(), 1, |a, _| -gradient[free[a]]));
        let decrement: f64 = -0.5
            * (0..free.len())
                .map(|a| gradient[free[a]] * step[(a, 0)])
                .sum::<f64>();
        if decrement < NEWTON_DECREMENT_TOL {
            converged = true;
            break Some((free, factor.inverse()));
        }
        if steps == MOST_STEPS {
            break None;
        }
        steps += 1;
        let before = objective(&inner, &inner) + penalty(&values).0;
        let accepted = (0..=MOST_HALVINGS).find_map(|halving| {
            let mut trial = values.clone();
            for (a, &f) in free.iter().enumerate() {
                let parameter = &parameters[fitted[f]].1;
                trial[fitted[f]] = (values[fitted[f]] + step[(a, 0)] / f64::powi(2.0, halving))
                    .clamp(parameter.lower, parameter.upper);
            }
            let candidate = survivors(sweep(&trial, &previous)).ok()?;
            let kept = inner
                .iter()
                .zip(&candidate)
                .all(|(base, trial)| base.is_none() || trial.is_some());
            (kept && objective(&candidate, &inner) + penalty(&trial).0 <= before)
                .then_some((trial, candidate))
        });
        match accepted {
            Some((trial, candidate)) => {
                values = trial;
                inner = candidate;
            }
            None => break None,
        }
    };

    let shared_covariance = information.as_ref().map(|(free, inverse)| {
        let mut covariance = FlatMatrix::zeros(k, k);
        covariance.data.fill(f64::NAN);
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                *covariance.get_mut(i, j) = inverse[(a, b)];
            }
        }
        covariance
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
    let bins = map.open_counts.dim().0;
    let blank = Array2::from_elem((rows, cols), f64::NAN);
    let isotopes = material.isotopes.len();
    let mut densities = vec![blank.clone(); isotopes];
    let mut density_sd = vec![blank.clone(); isotopes];
    let mut temperature_k = blank.clone();
    let mut temperature_sd_k = blank.clone();
    let mut overdispersion = [blank.clone(), blank];
    let mut trusted = Array2::from_elem((rows, cols), false);
    let mut failed = Array2::from_elem((rows, cols), false);
    let mut fits = Array2::from_elem((rows, cols), None);
    let mut covariance = Array2::from_elem((rows, cols), None);
    let mut sensitivity = Array2::from_elem((rows, cols), None);
    let mut residuals = [
        Array3::from_elem((bins, rows, cols), f64::NAN),
        Array3::from_elem((bins, rows, cols), f64::NAN),
    ];
    for (r, &patch) in fitted_patches.iter().enumerate() {
        let Some(Inner { fit, terms, .. }) = &inner[r] else {
            failed[patch] = true;
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
        let marginal = shared_covariance.as_ref().map(|shared| {
            let mut marginal = FlatMatrix::zeros(q, q);
            for (a, row_a) in rows_of.iter().enumerate() {
                for (b, row_b) in rows_of.iter().enumerate() {
                    *marginal.get_mut(a, b) = match (row_a, row_b) {
                        (Some(i), Some(j)) => {
                            terms.covariance.get(*i, *j)
                                + (0..k)
                                    .flat_map(|s| (0..k).map(move |t| (s, t)))
                                    .filter(|&(s, t)| shared.get(s, t).is_finite())
                                    .map(|(s, t)| {
                                        terms.sensitivity.get(*i, s)
                                            * shared.get(s, t)
                                            * terms.sensitivity.get(*j, t)
                                    })
                                    .sum::<f64>()
                        }
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
        let counts = [&regions[r].open_counts, &regions[r].sample_counts];
        for run in 0..2 {
            overdispersion[run][patch] = fit.overdispersion[run].unwrap_or(f64::NAN);
            for (k, (&y, &mu)) in counts[run].iter().zip(&fit.predicted[run]).enumerate() {
                residuals[run][[k, patch.0, patch.1]] =
                    (2.0 * half_deviance(y, mu)).sqrt().copysign(y - mu);
            }
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
        residuals,
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
        quantities: template.into_iter().map(|(_, name)| name).collect(),
        covariance,
        sensitivity,
        deviance: inner.iter().flatten().map(|i| i.deviance).sum(),
        converged,
        steps,
    })
}

type Swept = Result<Option<Inner>, PipelineError>;

struct Inner {
    fit: RegionFit,
    deviance: f64,
    terms: RegionTerms,
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
