//! Maps of a material's areal densities and temperature over the detector,
//! fitted as one counts fit whose regions are patches of pixels.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, s};
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::half_deviance;
use nereids_fitting::statistics::common_mode;

use crate::counts_fit::{
    CountsFit, Material, Measurement, Region, Role, Value, fit_counts, quantities,
};
use crate::error::PipelineError;
use crate::open_beam::{Calibration, whole};

/// Most entries the Jacobian of [`fit_map`]'s fit may hold: a row for each
/// bin of both runs of every region, and a column for each of half the bins
/// of every region's beam, each quantity of every region, and the nine
/// shared ones.
pub const MAX_MAP_SIZE: usize = 12_500_000;

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
    /// Each isotope's areal density in atoms/barn, in the material's order:
    /// the known one, or the fitted one; NaN where the patch is not fitted.
    pub densities: Vec<Array2<f64>>,
    /// Standard deviation of each density, from [`Self::covariance`]; NaN
    /// where the density is known or has no finite variance, or the patch is
    /// not fitted.
    pub density_sd: Vec<Array2<f64>>,
    /// The temperature in K: the known one, or the fitted one; NaN where the
    /// patch is not fitted.
    pub temperature_k: Array2<f64>,
    /// Standard deviation of the temperature in K, as [`Self::density_sd`].
    pub temperature_sd_k: Array2<f64>,
    /// Each fitted patch's [`RegionFit::overdispersion`](crate::counts_fit::RegionFit::overdispersion)
    /// of the open-beam run, then of the sample run; NaN where it is `None`
    /// or the patch is not fitted.
    pub overdispersion: [Array2<f64>; 2],
    /// Whether the fit converged and each of the patch's fitted densities and
    /// fitted temperature has a finite standard deviation or, for a density,
    /// is 0.
    pub trusted: Array2<bool>,
    /// For the open-beam run, then the sample run, the signed deviance
    /// residual `sign(y − μ)·√(2(y·ln(y/μ) + μ − y))` of each patch's counts
    /// `y` and the fit's predicted counts `μ` in each bin, in (time bin, patch
    /// row, patch column): `−√(2μ)` where `y` is 0, NaN where the patch is not
    /// fitted.  Its squares over twice the overdispersion each run was
    /// weighted with ([`RegionFit::overdispersion`](crate::counts_fit::RegionFit::overdispersion)),
    /// summed with the empty pixels' alike, are [`CountsFit::deviance`].
    pub residuals: [Array3<f64>; 2],
    /// [`CountsFit::covariance`] in this order: each fitted patch, row by
    /// row, with its fitted densities, in the material's order, its
    /// temperature if fitted and its fitted `b0`, `b1` and `b2`; the empty
    /// pixels' fitted `b0`, `b1` and `b2`; then the fitted normalization,
    /// `t0`, flight path, `α₀`, `α₁`, `β₀`, `β₁`, `R` and `h²`.
    pub covariance: Option<FlatMatrix>,
    /// The part of [`Self::covariance`] the shared quantities carry, those of
    /// the normalization, `t0`, flight path and pulse that are fitted and not
    /// on a bound, by [`common_mode`]: between quantities of two patches, or of
    /// a patch and the empty pixels, it is their whole covariance; zero when
    /// no shared quantity is fitted off its bounds.  `None` when `covariance`
    /// is, or when that of the shared quantities is not finite and positive
    /// definite.
    pub common_mode: Option<FlatMatrix>,
    /// The fit, whose region `r` is the `r`-th [`Patch::Sample`] of
    /// [`MapMeasurement::patches`] row by row, and whose last region is the
    /// empty pixels when any is not excluded.
    pub fit: CountsFit,
}

/// Fit `map.material` in each patch of `binning`×`binning` pixels whose
/// pixels not excluded all lie behind the sample, by one [`fit_counts`] whose
/// regions are those patches, row by row, and then the empty pixels not
/// excluded as one region with no material.  A region's counts are the sums
/// over its pixels.
///
/// The patches share the normalization, `t0`, the flight path and the pulse,
/// as the regions of [`fit_counts`] do: one clock, one flight path, one
/// moderator, and a detector efficiency that changes between the runs alike
/// in every pixel.  Each patch has its own beam and background.  The empty
/// pixels pin the normalization when their `b0` is known.
///
/// A patch has one density per isotope and one temperature: where its
/// pixels' optical depths `τ` differ, their summed transmission `⟨e^−τ⟩`
/// exceeds `e^−⟨τ⟩` by about `Var(τ)/2` of it, and the fitted densities are
/// below the pixels' mean.  A temperature that a patch's counts barely
/// determine can run to 1 K or 5000 K, which blanks every entry of the
/// covariance; give it within bounds.
///
/// # Errors
/// Everything [`MapMeasurement::patches`] refuses;
/// [`PipelineError::InvalidParameter`] if a pixel is both behind the sample
/// and empty, a density, the temperature or a background
/// term of the sample's patches is [`Value::Measured`], which would count its
/// measurement once per patch, a count of a pixel summed into a region is
/// not a whole non-negative number, no patch is fitted, or the fit's
/// Jacobian may hold more than [`MAX_MAP_SIZE`] entries; everything
/// [`fit_counts`] refuses, with region `r` as in [`CountsMap::fit`];
/// [`PipelineError::Fitting`] if [`common_mode`] refuses the fit's
/// covariance.
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

    let bins = map.open_counts.dim().0;
    let region_count = fitted_patches.len() + usize::from(!empty.is_empty());
    let jacobian_rows = 2 * bins * region_count;
    let jacobian_cols = region_count * (bins / 2)
        + fitted_patches.len() * (material.isotopes.len() + 4)
        + 3 * usize::from(!empty.is_empty())
        + 9;
    if jacobian_rows.saturating_mul(jacobian_cols) > MAX_MAP_SIZE {
        return invalid(format!(
            "{} patches and {} empty pixels in {bins} time bins may need a {jacobian_rows}×\
             {jacobian_cols} Jacobian, more than the {MAX_MAP_SIZE} entries fit_map allows; \
             use fewer patches or fewer time bins",
            fitted_patches.len(),
            empty.len()
        ));
    }

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

    let measurement = Measurement {
        time_edges_us: map.time_edges_us.clone(),
        charge_ratio: map.charge_ratio,
        normalization: map.normalization,
        regions,
    };
    let fit = fit_counts(&measurement, calibration)?;

    let fitted: Vec<Role> = quantities(&measurement, calibration)
        .filter(|(_, value)| !matches!(value, Value::Known(_)))
        .map(|(role, _)| role)
        .collect();
    let shared = measurement.regions.len();
    let mut order: Vec<usize> = (0..fitted.len()).collect();
    order.sort_by_key(|&q| match fitted[q] {
        Role::Beam {
            region,
            coefficient,
        } => (region, 3, coefficient),
        Role::Density { region, isotope } => (region, 0, isotope),
        Role::Temperature { region } => (region, 1, 0),
        Role::Background { region, term } => (region, 2, term),
        Role::Normalization => (shared, 0, 0),
        Role::T0 => (shared, 1, 0),
        Role::FlightPath => (shared, 2, 0),
        Role::Pulse(number) => (shared, 3, number),
    });
    let covariance = fit.covariance.as_ref().map(|c| {
        let mut reordered = FlatMatrix::zeros(order.len(), order.len());
        for (a, &p) in order.iter().enumerate() {
            for (b, &q) in order.iter().enumerate() {
                *reordered.get_mut(a, b) = c.get(p, q);
            }
        }
        reordered
    });
    let carriers: Vec<usize> = (0..order.len())
        .filter(|&a| {
            let q = order[a];
            !fit.on_bound[q]
                && matches!(
                    fitted[q],
                    Role::Normalization | Role::T0 | Role::FlightPath | Role::Pulse(_)
                )
        })
        .collect();
    let common = match &covariance {
        Some(c) => common_mode(c, &carriers)?,
        None => None,
    };

    let sd = |q: usize| {
        fit.covariance
            .as_ref()
            .map_or(f64::NAN, |c| c.get(q, q).sqrt())
    };
    let sd_of = |role: Role| fitted.iter().position(|&r| r == role).map_or(f64::NAN, sd);
    let (rows, cols) = patches.dim();
    let blank = Array2::from_elem((rows, cols), f64::NAN);
    let isotopes = material.isotopes.len();
    let mut densities = vec![blank.clone(); isotopes];
    let mut density_sd = vec![blank.clone(); isotopes];
    let mut temperature_k = blank.clone();
    let mut temperature_sd_k = blank.clone();
    let mut overdispersion = [blank.clone(), blank];
    let mut trusted = Array2::from_elem((rows, cols), false);
    let mut residuals = [
        Array3::from_elem((bins, rows, cols), f64::NAN),
        Array3::from_elem((bins, rows, cols), f64::NAN),
    ];
    for (r, &patch) in fitted_patches.iter().enumerate() {
        let region = &fit.regions[r];
        for (m, &density) in region.densities.iter().enumerate() {
            densities[m][patch] = density;
            density_sd[m][patch] = sd_of(Role::Density {
                region: r,
                isotope: m,
            });
        }
        temperature_k[patch] = region.temperature_k.unwrap_or(f64::NAN);
        temperature_sd_k[patch] = sd_of(Role::Temperature { region: r });
        let counts = [
            &measurement.regions[r].open_counts,
            &measurement.regions[r].sample_counts,
        ];
        for run in 0..2 {
            overdispersion[run][patch] = region.overdispersion[run].unwrap_or(f64::NAN);
            for (k, (&y, &mu)) in counts[run].iter().zip(&region.predicted[run]).enumerate() {
                residuals[run][[k, patch.0, patch.1]] =
                    (2.0 * half_deviance(y, mu)).sqrt().copysign(y - mu);
            }
        }
        trusted[patch] = fit.converged
            && fitted.iter().enumerate().all(|(q, &role)| match role {
                Role::Density { region: p, isotope } if p == r => {
                    sd(q).is_finite() || region.densities[isotope] == 0.0
                }
                Role::Temperature { region: p } if p == r => sd(q).is_finite(),
                _ => true,
            });
    }

    Ok(CountsMap {
        patches,
        densities,
        density_sd,
        temperature_k,
        temperature_sd_k,
        overdispersion,
        trusted,
        residuals,
        covariance,
        common_mode: common,
        fit,
    })
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
