//! The areal densities and temperatures of the materials in one or more
//! regions of the detector, each region's background and gamma transmission,
//! the sample run's normalization, the time background, the timing offset,
//! the flight path and the pulse's numbers, fitted to the counts of an
//! open-beam run and a sample run recorded in the same time bins.

use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use faer::{Mat, Side};
use nereids_endf::resonance::{ResonanceCovariance, ResonanceData};
use nereids_endf::retrieval::EndfLibrary;
use nereids_fitting::error::FittingError;
use nereids_fitting::lm::{FitModel, FlatMatrix};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::{Prior, Unbounded, information_inverse};
use nereids_fitting::statistics::{Consistency, consistency};
use nereids_physics::continuous_doppler::{
    Channel, QuadratureBudget, SUPPORT_X, broaden_channel, broaden_with_budget,
};
use nereids_physics::detector_spread::region_shares;
use nereids_physics::doppler::DopplerParams;
use nereids_physics::flight_time_grid::{FlightTimeGrid, Rows};
use nereids_physics::forward_scatter::{ForwardYield, forward_yield, lowest_exit_ratio};
use nereids_physics::ikeda_carpenter::IkedaCarpenterParams;
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::thickness::{interacting_per_depth, uneven_transmission};
use nereids_physics::transmission::resonance_center_energies;
use rayon::prelude::*;

use crate::beam::BeamSpline;
use crate::error::PipelineError;
use crate::open_beam::{
    BOUND, COUNTS_TO_MEASURE_NOISE, Calibration, OpenBeamFit, PULSE_NUMBERS, Pulse, Recorded,
    combined, counted, fit_on_halved_grids, laws, open_beam_with_intervals, overdispersion,
    validate_counts, validate_live, weights_of,
};

/// A count in a bin predicted fewer counts than this has a chance below it
/// under the model.
pub const NEGLIGIBLE_PREDICTION: f64 = 1e-10;

const SETTLED_OVERDISPERSION: f64 = 0.01;
/// The most cells a spread's grid holds.
const MOST_SPREAD_CELLS: usize = 1 << 20;

pub(crate) const MOST_PASSES: usize = 20;

/// The `(low, high)` temperature bounds in K that a fit with a free
/// temperature uses.
pub(crate) const TEMPERATURE_BOUNDS_K: (f64, f64) = (1.0, 5000.0);

/// A quantity the fit holds at a known value, fits from a starting value
/// over the quantity's range, fits from `start` within `lower..=upper` inside
/// that range, or fits over the range from a `value` measured elsewhere with
/// standard deviation `sd`, a Gaussian prior on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Known(f64),
    Fitted(f64),
    Within { start: f64, lower: f64, upper: f64 },
    Measured { value: f64, sd: f64 },
}

impl Value {
    pub(crate) fn parameter(
        self,
        name: impl Into<Cow<'static, str>>,
        range: RangeInclusive<f64>,
        allowed: &str,
    ) -> Result<FitParameter, PipelineError> {
        let name = name.into();
        let (value, lower, upper) = match self {
            Self::Known(v) | Self::Fitted(v) | Self::Measured { value: v, .. } => {
                (v, *range.start(), *range.end())
            }
            Self::Within {
                start,
                lower,
                upper,
            } => (start, lower, upper),
        };
        if !(value.is_finite()
            && range.contains(&lower)
            && lower < upper
            && range.contains(&upper)
            && (lower..=upper).contains(&value)
            && !matches!(self, Self::Measured { sd, .. } if !(sd.is_finite() && sd > 0.0)))
        {
            return Err(PipelineError::InvalidParameter(match self {
                Self::Within { .. } => format!(
                    "{name} must be {allowed}, with bounds lower < upper and a finite start \
                     between them; got {self:?}"
                ),
                Self::Measured { .. } => format!(
                    "{name} must be finite and {allowed}, with a finite positive sd; got {self:?}"
                ),
                _ => format!("{name} must be finite and {allowed}; got {self:?}"),
            }));
        }
        Ok(FitParameter {
            name,
            value,
            lower,
            upper,
            fixed: matches!(self, Self::Known(_)),
        })
    }
}

/// An open-beam run and a sample run recorded in the same time bins, read in
/// one or more regions of the detector.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Time-bin edges in µs.
    pub time_edges_us: Vec<f64>,
    /// `c_q`, the sample run's proton charge over the open-beam run's.
    pub charge_ratio: f64,
    /// `a`, the normalization of the sample run, positive, the same in every
    /// region.
    pub normalization: Value,
    /// `[C, A, τ]` of the time background `C + A·exp(−(t − t₁)/τ)`, the counts
    /// per µs and per pixel at the open-beam run's proton charge that reach the
    /// detector without passing through the moderator pulse, at time `t` of the
    /// time edges' clock, `t₁` the first edge: `C` and `A` 0 or more, `τ` in µs
    /// positive.  [`NO_TIME_BACKGROUND`] gives none.
    pub time_background: [Value; 3],
    /// Counts that follow each run's duration, not its proton charge; `None`
    /// for none.
    pub room_background: Option<RoomBackground>,
    /// Counts the detector records in another region than the one they
    /// arrive in; `None` for none.
    pub spread: Option<Spread>,
    /// The regions, separate sets of pixels timed by one clock.
    pub regions: Vec<Region>,
}

/// A [`Measurement::time_background`] of no counts.
pub const NO_TIME_BACKGROUND: [Value; 3] =
    [Value::Known(0.0), Value::Known(0.0), Value::Known(1.0)];

/// Counts that arrive at the same rate at every time of every frame, in
/// both runs alike, whatever the proton charge and whatever is in front of
/// the pixels, such as the detector's dark counts: `ρ` per µs per pixel per
/// frame, `ρ · w_k · F` per pixel in a bin of width `w_k` of a run of `F`
/// frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoomBackground {
    /// `ρ`, 0 or more.
    pub rate: Value,
    /// The frames each run recorded, the open-beam run's then the sample
    /// run's, finite and positive.
    pub frames: [f64; 2],
}

/// A share `h` of the counts arriving in each cell of a detector grid that
/// the detector records elsewhere, spread by the radial kernel
/// `Σ_c w_c·exp(−r/s_c)/(2π s_c²)`
/// ([`transfer`](nereids_physics::detector_spread::transfer)): region `r`
/// records `(1 − h)·μ_r + h·Σ_q F_rq·μ_q` of the counts `μ` arriving in the
/// regions in each bin of both runs, `F_rq` the share of the spread counts
/// arriving evenly over region `q`'s cells that lands in region `r`'s cells
/// ([`region_shares`]).  Cells in no region neither send nor record counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Spread {
    /// `h`, within 0–1.
    pub share: Value,
    /// `(w_c, s_c)`: each term's weight, finite and positive, normalized by
    /// their sum, and length in cells, finite and positive.
    pub kernel: Vec<(f64, f64)>,
    /// The grid's rows and columns, at most 2^20 cells in all.
    pub grid: [usize; 2],
    /// Each region's cells as `[row, column]`, in the order of
    /// [`Measurement::regions`]; a cell is in one region at most.
    pub cells: Vec<Vec<[usize; 2]>>,
}

/// The counts of a set of pixels in both runs, its background and the
/// material the beam crosses there.
#[derive(Debug, Clone)]
pub struct Region {
    /// Raw counts of the open-beam run, one per bin.
    pub open_counts: Vec<f64>,
    /// Raw counts of the sample run, one per bin.
    pub sample_counts: Vec<f64>,
    /// The fraction of the open-beam run's counts arriving in each bin that
    /// the detector records, in (0, 1]; `None` records every one.
    pub open_live: Option<Vec<f64>>,
    /// The fraction of the sample run's counts arriving in each bin that the
    /// detector records, in (0, 1]; `None` records every one.
    pub sample_live: Option<Vec<f64>>,
    /// `[b0, b1, b2, d, f]` of the background
    /// `b(E) = b0 + b1/√E + b2·√E + d·exp(−f/√E)`: `b0` and `d`
    /// dimensionless, `b1` in √eV, `b2` in 1/√eV and `f` in √eV, `f` 0 or
    /// more and the others any real number.
    pub background: [Value; 5],
    /// The region's area in pixels, finite and positive.
    pub pixels: f64,
    /// The fraction of the time background that reaches the region's pixels in
    /// the sample run through what is in front of them, within 0–1.
    pub gamma_transmission: Value,
    /// `ε`, the share of the neutrons captured in the region's sample whose
    /// gammas its pixels record, 0 or more: the sample run gains
    /// `c_q·a·ε·φ·(1 − T)·Σ_m n_m σ_γm/D`, the captures at the neutrons' first
    /// collision, `σ_γm` each isotope's capture cross section broadened like
    /// its total one, at the neutrons' flight times.
    pub capture_gammas: Value,
    /// `Ω`, the share of the neutrons scattered once in the region's sample
    /// and leaving through its back face that its pixels record, 0 or more:
    /// the sample run gains `c_q·a·Ω·φ·Σ_m n_m σ_sm·G_m`, `σ_sm` each isotope's
    /// elastic cross section broadened like its total one and `G_m` its share
    /// leaving through the back face ([`forward_yield`]), at the mean areal
    /// density and the neutrons' flight times.
    pub in_scatter: Value,
    /// The material the beam crosses in the region; `None` for nothing, which
    /// transmits every neutron.
    pub material: Option<Material>,
}

/// A material's isotopes and temperature.
#[derive(Debug, Clone)]
pub struct Material {
    /// Each isotope with its areal density in atoms/barn, known, fitted or
    /// measured, at least 0.
    pub isotopes: Vec<(ResonanceData, Value)>,
    /// The temperature in K, within 1–5000 K.
    pub temperature_k: Value,
    /// `v`, the relative variance of the areal density over the region's
    /// pixels, weighted by the beam that reaches them, 0 or more: 0 for a
    /// uniform sample.  The region transmits `(1 + v·D)^(−1/v)`, the mean for a
    /// Gamma-distributed areal density, `D` the optical depth at the mean density
    /// ([`uneven_transmission`]).
    pub thickness_variance: Value,
    /// The library the isotopes' resonance data were read from.  The fit
    /// leaves out the sample run's bins that overlap the span between the mean
    /// arrival times, `t0 + L·TOF_FACTOR/√E + 3/α + R/β`, of the two ends of
    /// each window [`EndfLibrary::excluded_windows_ev`] gives for an isotope
    /// not known to be absent, at the fitted `t0`, flight path `L` and pulse.
    pub library: EndfLibrary,
    /// The covariance of the resonance parameters of the isotopes listed, each
    /// with its index in `isotopes`, which [`CountsFit::covariance`] carries;
    /// an isotope not listed has its resonance parameters taken as exact.
    pub covariances: Vec<(usize, ResonanceCovariance)>,
}

/// The fitted quantities of every region, and the normalization, time
/// background, timing offset, flight path and pulse numbers the regions share.
#[derive(Debug, Clone)]
pub struct CountsFit {
    /// Each region's fitted quantities and counts, in the order given.
    pub regions: Vec<RegionFit>,
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
    /// The pulse's `R₁`: the known one, or the fitted one.
    pub r1: f64,
    /// The time background's `[C, A, τ]`, each the known or the fitted one; a
    /// fitted `τ` is undetermined when `A` ends at 0.
    pub time_background: [f64; 3],
    /// The room background's `ρ`: the known one, the fitted one, or 0 without
    /// one.
    pub room_background: f64,
    /// The spread's `h`: the known one, the fitted one, or 0 without one.
    pub spread_share: f64,
    /// Covariance of the fitted quantities among, in this order, each region's
    /// densities, in the order given, temperature and thickness variance,
    /// region by region; the
    /// normalization; each region's `b0`, `b1`, `b2`, `d` and `f`, region by region;
    /// each region's gamma transmission, region by region; each region's
    /// capture gammas' `ε`, region by region; each region's in-scatter's `Ω`,
    /// region by region; the time background's `C`, `A` and `τ`; the room
    /// background's `ρ`; the spread's `h`; `t0`; the
    /// flight path; `α₀`, `α₁`,
    /// `β₀`, `β₁`, `R`, `h²` and `R₁`: the inverse of the information at the fit, each run's expected information over its
    /// overdispersion plus `1/sd²` for each measured quantity and `C⁻¹` over
    /// the pulse numbers a calibration's
    /// [`Pulse::prior`](crate::open_beam::Pulse::prior) covers.  The row and
    /// column of a quantity on one of its bounds, or that neither the counts
    /// nor a measurement determine, are NaN, and the other entries are
    /// conditional on every quantity that ended on a bound being held there;
    /// every entry is NaN when the fitted temperature of any region ends at
    /// 1 K or 5000 K.  `None` when the fit did not converge.
    ///
    /// Each covariance in a material's [`Material::covariances`] adds `S M Sᵀ`,
    /// first order in the resonance parameters it lists, so valid while a
    /// standard deviation of each moves the fit little: `M` that covariance,
    /// `S = −C Jᵀ W c`, `C` this covariance without the term, `J` and `c` the
    /// counts' slopes with respect to the fitted quantities and to those
    /// parameters, the latter by central differences at a tenth of each
    /// standard deviation, and `W` the counts' weights.  Every other known
    /// quantity is taken as exact.
    /// They are not reliable where the counts barely determine a fitted
    /// temperature or barely separate it from a density, as at few counts or
    /// for a thin sample at modest counts.
    pub covariance: Option<FlatMatrix>,
    /// The fitted quantities' Gaussian without their bounds, in the
    /// covariance's order (see [`Unbounded`]), with the resonance parameters
    /// taken as exact, NaN where `covariance` is at a temperature edge.  `None` when `covariance` is, or when the fitter
    /// reports none.
    pub unbounded: Option<Unbounded>,
    /// Whether each fitted quantity, in the covariance's order, ended on one
    /// of its bounds.
    pub on_bound: Vec<bool>,
    /// Each run's half Poisson deviance over the overdispersion it was
    /// weighted with, summed over both runs of every region at the fit, outside
    /// the sample runs' [`RegionFit::excluded_bins`].
    pub deviance: f64,
    /// Whether the fitter converged, the open-beam fits' time background
    /// settled at the fitted one, every sample run's overdispersion settled,
    /// the grid met its rule at the fitted temperatures, its flight times cover the fitted `t0`, flight path and pulse, and the bins left
    /// out at those are the ones the fit left out.
    pub converged: bool,
    /// For each measured quantity, in the covariance's order, its value
    /// without the fit's bounds ([`Self::unbounded`]) less its measurement
    /// over the standard deviation of that difference, `√(sd² − variance + N)`,
    /// the variance also without the bounds and `N` the term resonance
    /// covariances add to it: near 0 ± 1 when the counts agree
    /// with the measurement.  NaN for a quantity the counts and measurements
    /// do not determine, and for every quantity when a fitted temperature ends
    /// at 1 K or 5000 K; it loses precision as the counts' information on the
    /// quantity vanishes beside the measurement's.  `None` when `unbounded`
    /// is.
    pub measured_pulls: Option<Vec<f64>>,
    /// Whether the counts accept the pulse's calibration on the numbers it
    /// covers, by [`consistency`] with the fit's numbers and covariance
    /// without their bounds ([`Self::unbounded`]).  The counts' information is
    /// divided by each run's overdispersion, so the test weakens as the
    /// overdispersion grows.  The
    /// overdispersion is at least 1, so with Poisson counts its estimate
    /// inflates the covariances the test uses, and `p` rejects a right
    /// calibration less often than its value says, the more so the fewer bins
    /// each run counts.  A calibration fitted to the same open-beam run is not
    /// independent of the fit, which the test does not account for.  `None`
    /// without a calibration, when `unbounded` is `None` or withheld, or when
    /// [`consistency`] gives none.
    pub pulse_consistency: Option<Consistency>,
    /// Step, in µs, of the fit's grid.
    pub step_us: f64,
    /// Number of points of that grid.
    pub points: usize,
    /// How many times the grid of
    /// [`FlightTimeGrid::new`](nereids_physics::flight_time_grid::FlightTimeGrid::new)
    /// was halved to reach it; that grid is built at the starting `t0`,
    /// flight path and pulse, or at fitted ones when the fit rebuilt it.
    pub halvings: usize,
}

/// One region's fitted quantities and counts.
#[derive(Debug, Clone)]
pub struct RegionFit {
    /// Areal density of each isotope of the region's material in atoms/barn,
    /// in the order given: the known one, or the fitted one; empty without a
    /// material.
    pub densities: Vec<f64>,
    /// The material's temperature in K: the known one, or the fitted one;
    /// `None` without a material.
    pub temperature_k: Option<f64>,
    /// The material's thickness variance `v`: the known one, or the fitted
    /// one; `None` without a material.
    pub thickness_variance: Option<f64>,
    /// `[b0, b1, b2, d, f]` of [`Region::background`], each the known or the
    /// fitted one.
    pub background: [f64; 5],
    /// The gamma transmission: the known one, or the fitted one.
    pub gamma_transmission: f64,
    /// The capture gammas' `ε`: the known one, or the fitted one.
    pub capture_gammas: f64,
    /// The in-scatter's `Ω`: the known one, or the fitted one.
    pub in_scatter: f64,
    /// The region's beam per µs, as a function of the arrival time less the
    /// starting `t0`, fitted to both runs, with the intervals the region's
    /// open-beam fit chose.
    pub beam: BeamSpline,
    /// Whether the region's open-beam fit chose its richest beam; see
    /// [`OpenBeamFit::at_limit`](crate::open_beam::OpenBeamFit::at_limit).
    pub beam_at_limit: bool,
    /// The open-beam and sample counts the fit predicts in each bin at its
    /// answer, live fractions included, on the grid it accepted: the counts
    /// [`CountsFit::deviance`] compares with the measured ones.
    pub predicted: [Vec<f64>; 2],
    /// Variance of the counts of the open-beam run, then of the sample run,
    /// over their Poisson variance: the value each run's counts are divided by
    /// in the returned fit.  The open-beam run's is the region's open-beam
    /// fit's [`OpenBeamFit::overdispersion`](crate::open_beam::OpenBeamFit::overdispersion); the sample run's is measured the
    /// same way, on the bins outside `excluded_bins` the first fit predicts at
    /// least one count, by the
    /// previous fit, and within 1% by the returned one when the fit converged.
    /// `None` when the run's counts have not measured it; the open-beam run is
    /// then weighted with 1, and the sample run as the open-beam run.
    pub overdispersion: [Option<f64>; 2],
    /// Indices, ascending, of the sample run's bins its material's
    /// [`Material::library`] leaves out, and, under a spread whose share is not
    /// known 0, those of every region that spreads counts into it: they add
    /// nothing to the deviance, the information or the sample run's
    /// overdispersion.
    pub excluded_bins: Vec<usize>,
}

/// Fit the areal densities and temperatures of the regions' materials, each
/// region's background terms and gamma transmission, the normalization, the
/// time background, and the timing offset, flight path and pulse numbers of
/// `calibration`, that are not known, to the
/// raw counts of both runs in every region.  On a uniform grid of flight
/// times `u_i`, energies `E_i` and step `w`, the counts of region `r` in bin
/// `k` are
///
/// ```text
/// O_rk   = ℓ^O_rk · [w Σ_i φ_ri P_ki + p_r G_k + p_r ρ F^O w_k]
/// S_rk   = ℓ^S_rk · [c_q · a · (w Σ_i φ_ri [T_ri + b_r(E_i)] P_ki + γ_r p_r G_k)
///                    + p_r ρ F^S w_k]
/// T_ri   = (1 + v_r D_ri)^(−1/v_r), D_ri = Σ_m n_rm σ_m(E_i; T_r), or 1 without a material
/// b_r(E) = b0_r + b1_r/√E + b2_r·√E + d_r·exp(−f_r/√E)
/// G_k    = ∫ over bin k of (C + A·exp(−(t − t₁)/τ)) dt
/// ```
///
/// with `φ_r` the region's beam per µs, `P_ki` the chance of a neutron at
/// `u_i` arriving in bin `k`, `n_rm` and `σ_m` each isotope's density and
/// total cross section Doppler-broadened at the material's temperature `T_r`,
/// `c_q` the charge ratio, `a` the normalization and `ℓ^O_rk`, `ℓ^S_rk` each
/// run's live fraction.  The background is beam neutrons that reach the
/// detector another way, so it passes through the pulse and scales with the
/// normalization; SAMMY's `BackA`, `BackB`, `BackC`, `BackD` and `BackF`
/// (`cro/mnrm1.f90`) are `a·b0`, `a·b1`, `a·b2`, `a·d` and `f`, to within the
/// background's change over the pulse's delay.  The time
/// background `G` is counts that follow the proton charge but reach the
/// detector without passing through the moderator pulse, such as the prompt
/// gammas of each proton pulse and their tail, at the time `t` of the edges'
/// clock, `t₁` the first edge: it is not folded with the pulse, and the sample
/// run scales it with `c_q·a`, which correct the charge ratio and an
/// efficiency change alike in every pixel, and the fraction `γ_r` that reaches
/// the region's pixels, `p_r` the region's pixels.  It turns
/// the ratio `R` of the neutrons alone into `R + g·(γ_r − R)`, `g` its share of
/// the open-beam counts; SAMMY's backgrounds in time (`Bgfrpi`) are added to
/// the transmission instead.  The room background `ρ` ([`RoomBackground`])
/// follows each run's duration instead, `F^O` and `F^S` its frames and `w_k`
/// bin `k`'s width: it is neither folded with the pulse nor scaled by the
/// charge ratio, the normalization or the gamma transmission, so it separates
/// from the time background's `C` only where `c_q·a·γ_r` differs from
/// `F^S/F^O`.
///
/// The regions share the normalization, the run pair's scale; and `t0`, the
/// flight path and the pulse, which one clock, flight path and moderator set
/// for every pixel, and with them the grid and `P_ki`.  Each region has its
/// own beam, from its own open-beam counts, with the intervals its
/// [`fit_open_beam`](crate::open_beam::fit_open_beam) chooses, fitted with
/// the rest to both runs; and its own
/// background, which depends on what surrounds its pixels.  A region without
/// a material counts `c_q·a·(1 + b_r)` of its beam in the sample run, so it
/// pins the normalization when its `b0` is known, as zero in the open-region
/// normalization of imaging, and nothing about it when its `b0` is fitted.
/// The regions must be separate pixels: the covariance takes every bin as
/// independent.
///
/// The [`Calibration`]'s `t0` and flight path `L` are fitted unless known.
/// The grid is built at a `t0₀` and `L₀`, at first the starting ones.  Grid
/// point `i` keeps its energy and arrives at `t0 + (L/L₀)·u_i`, with `u_i`
/// its flight time at `t0₀` and `L₀`, standing for `(L/L₀)·w` of flight time;
/// the beams are functions of the arrival time less the starting `t0`.  This
/// is SAMMY's energy scale: its `Tzero` is `t0`, and its `Elzero` times its
/// flight path is `L` (`dat/mdat0.f90`, `Mtzero`).
///
/// The pulse's numbers are fitted unless known, each point's `P_ki` taken at
/// the pulse's laws at its energy `E_i`.  A trial pulse whose `α` or `β` is 0
/// at some energy is outside the model and rejected; a fit driven there ends
/// unconverged.  The open-beam fits that start the beams and weight the
/// open-beam runs use the starting pulse and, at first, the time background's
/// starting numbers; the fit is repeated with them at its fitted numbers, up
/// to twenty times, until the fitted numbers change the counts they took as
/// known by `Σ_r Σ_k (ΔO_rk)²/O_rk` at most [`BOUND`] over every region's
/// open-beam bins, and from the third time on each region keeps the beam
/// intervals its second open-beam fit chose.
///
/// The grid's first step is at most half the narrowest Doppler full width at
/// half maximum, in flight time, of any resonance of any region inside its
/// energy span, at that region's starting temperature; it is then halved
/// until the counts of every run of every region, together, meet
/// [`BOUND`].  The step is uniform and shared, so a
/// wide window whose span holds a narrow resonance at high energy, or a low
/// fitted temperature in any region, can exceed the grid's point cap; a
/// temperature a region's counts barely determine can run to 1 K and refuse
/// the fit that way.
///
/// The fit minimizes each run's half Poisson deviance over its
/// overdispersion, plus `½((x − value)/sd)²` for each [`Value::Measured`]
/// quantity `x`, and `½(θ − m)ᵀC⁻¹(θ − m)` over the pulse numbers `θ` a
/// calibration's [`Pulse::prior`](crate::open_beam::Pulse::prior) covers,
/// with `m` and `C` that prior's mean, which may lie past a number's bound,
/// and covariance.  Each region's open-beam run is weighted with its
/// open-beam fit's overdispersion, and its sample run first with the same.
/// The fit is repeated from its answer while any sample run's
/// overdispersion, measured on the bins outside those left out that the first
/// fit predicts at least one count, changes by more than 1%, the coarser grid of the accepted pair is
/// wider than the rule at the fitted temperatures, its flight times miss
/// some that the fitted `t0`, `L` and pulse need, or the bins left out at
/// those differ from the ones it left out, which for the first fit are those
/// at the starting ones.  The next first grid is the
/// finer of that pair's coarser grid and the rule's grid at the fitted
/// temperatures, or, when the flight times miss, the rule's grid of a grid
/// built at the fitted `t0`, `L` and pulse.  After twenty fits it is reported
/// unconverged.
///
/// The fitter finds a local minimum.  A thin sample hotter than about
/// 1,500 K fitted from room temperature can end in a false one, reported
/// converged with an overdispersion far above 1.  Where a black resonance
/// empties bins and the background is near zero, a fitted background with no
/// lower bound can drive a black bin's prediction to zero, and the fit ends
/// unconverged; with `b1` and `b2` known, `b0` bounded below by 0 ends on
/// that bound, converged.
///
/// # Errors
/// [`PipelineError::ShapeMismatch`] unless each run of each region has one
/// count, and one live fraction when given, per bin;
/// [`PipelineError::InvalidParameter`] if a count is not a whole non-negative
/// number, a run has no counts, a live fraction is not in (0, 1], the charge
/// ratio or a room background's frames are not finite and positive, a
/// region's pixels are not finite and
/// positive, a region without a material has capture gammas or in-scatter
/// other than known 0, a spread's kernel has a term whose weight or length is
/// not finite and positive, its grid holds no cell or more than 2^20, it does
/// not give each region one or more cells on the grid, or a cell is in two
/// regions, a known, starting or measured value is
/// not finite and in its quantity's range, a measured value's sd is not finite
/// and positive, bounds are not `lower < upper` in that range
/// with the start between them, the pulse's calibration covers a number that
/// is not fitted or is measured, no region holds a material, a material has
/// no isotopes or lists one twice, an isotope's resonance data are not
/// finite, an isotope not known to be absent has a resonance between the
/// energies of the last and first time edges, at the starting `t0` and flight
/// path or a converged fit's, outside the pulse's
/// [`line_span_ev`](crate::open_beam::Pulse::line_span_ev), or the energies
/// its broadened cross section reads on the grid at the starting `t0`, flight
/// path and pulse, or at the fitted ones of any pass that rebuilds it, at the
/// region's known temperature or at the upper bound of a fitted one, down to
/// zero for a window within the thermal spread of zero energy, are not inside
/// a single one of its evaluated (SLBW, MLBW or Reich–Moore) resolved ranges,
/// or, in a region with in-scatter, an isotope has a resonance below that
/// grid's lowest energy and above the lowest energy a neutron scattered forward
/// leaves with;
/// [`PipelineError::UnmodelledCounts`] if at the fit, converged or not, a bin
/// is predicted negative or non-finite counts, recorded or, under a spread,
/// arriving, or, outside the bins left
/// out, holds counts predicted below [`NEGLIGIBLE_PREDICTION`]: starting or known values the fitter cannot
/// leave;
/// [`PipelineError::InvalidParameter`] if a material's covariance names no
/// isotope of it, or one twice, lists no resonance, one twice or one the
/// isotope does not have, or is not a finite symmetric positive
/// semi-definite matrix over their energies and widths, or if it is given for
/// an isotope another region also holds or with a pulse calibration's prior;
/// everything [`fit_open_beam`](crate::open_beam::fit_open_beam) refuses for
/// any region;
/// [`PipelineError::FlightTimeGrid`] for the grid's refusals at the starting
/// `t0`, flight path and pulse or the fitted ones of any pass, including more
/// points than it allows; [`PipelineError::Fitting`] if the fitter fails, a
/// region's open-beam fit does not converge with the beam intervals it keeps,
/// the cross sections fail at the start or, for a listed resonance parameter
/// shifted, at the fit, or a decomposition of the pulse's
/// consistency test fails; a failure at a trial temperature is a rejected
/// step.
pub fn fit_counts(
    measurement: &Measurement,
    calibration: &Calibration,
) -> Result<CountsFit, PipelineError> {
    if measurement
        .regions
        .iter()
        .all(|region| region.material.is_none())
    {
        return Err(PipelineError::InvalidParameter(
            "no region of the measurement holds a material to fit".into(),
        ));
    }
    fit_settled(measurement, calibration, MOST_PASSES)
}

fn fit_settled(
    measurement: &Measurement,
    calibration: &Calibration,
    most_passes: usize,
) -> Result<CountsFit, PipelineError> {
    let bins = starts_and_widths_us(&measurement.time_edges_us);
    let frames = open_frames(measurement.room_background.as_ref());
    let mut open = starting_bypass(measurement)?;
    let mut intervals: Option<Vec<usize>> = None;
    let shares = spread_shares(measurement)?;
    for passes in 1.. {
        let (mut fit, _) = fit_counts_answer(
            measurement,
            calibration,
            BeamStarts::OpenFitsAt(open, intervals.clone()),
            shares.clone(),
        )?;
        let fitted = Bypass {
            time: fit.time_background,
            room: fit.room_background,
        };
        let used = open.open_counts_per_pixel(&bins, frames);
        let now = fitted.open_counts_per_pixel(&bins, frames);
        let mut spread = 0.0;
        for (r, region) in fit.regions.iter().zip(&measurement.regions) {
            for (k, &mu) in r.predicted[0]
                .iter()
                .enumerate()
                .filter(|(_, mu)| **mu > 0.0)
            {
                let live = region.open_live.as_ref().map_or(1.0, |live| live[k]);
                spread += (live * region.pixels * (now[k] - used[k])).powi(2) / mu;
            }
        }
        if !fit.converged || spread <= BOUND || passes == most_passes {
            if spread > BOUND {
                fit.converged = false;
                fit.covariance = None;
                fit.unbounded = None;
                fit.measured_pulls = None;
                fit.pulse_consistency = None;
            }
            return Ok(fit);
        }
        if passes >= 2 {
            intervals
                .get_or_insert_with(|| fit.regions.iter().map(|r| r.beam.intervals()).collect());
        }
        open = fitted;
    }
    unreachable!("the passes end at most_passes")
}

/// The time background's `[C, A, τ]` and the room background's `ρ`, which
/// together give the counts per pixel of a run that bypass the pulse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bypass {
    pub(crate) time: [f64; 3],
    pub(crate) room: f64,
}

impl Bypass {
    /// The counts per pixel these numbers put in each bin of the open-beam run
    /// of `frames` frames.
    pub(crate) fn open_counts_per_pixel(&self, bins: &[(f64, f64)], frames: f64) -> Vec<f64> {
        let (time, _) = time_counts_per_pixel(bins, self.time);
        time.iter()
            .zip(bins)
            .map(|(g, (_, width))| g + self.room * frames * width)
            .collect()
    }
}

/// The open-beam run's frames, 0 without a room background.
pub(crate) fn open_frames(room: Option<&RoomBackground>) -> f64 {
    room.map_or(0.0, |room| room.frames[0])
}

pub(crate) fn starting_bypass(measurement: &Measurement) -> Result<Bypass, PipelineError> {
    let start = |role| {
        let (value, name, range, allowed) = quantity(measurement, role).expect("a quantity");
        Ok::<f64, PipelineError>(value.parameter(name, range, &allowed)?.value)
    };
    Ok(Bypass {
        time: [
            start(Role::TimeBackground(0))?,
            start(Role::TimeBackground(1))?,
            start(Role::TimeBackground(2))?,
        ],
        room: start(Role::RoomBackground)?,
    })
}

pub(crate) struct Starts {
    pub(crate) beams: Vec<BeamStart>,
    pub(crate) origin_us: f64,
    pub(crate) shared_leverage: Option<Vec<f64>>,
    pub(crate) bound: f64,
    pub(crate) excluded: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub(crate) struct BeamStart {
    pub(crate) beam: BeamSpline,
    pub(crate) overdispersion: Option<f64>,
    pub(crate) at_limit: bool,
}

impl From<OpenBeamFit> for BeamStart {
    fn from(open: OpenBeamFit) -> Self {
        Self {
            beam: open.beam,
            overdispersion: open.overdispersion,
            at_limit: open.at_limit,
        }
    }
}

pub(crate) struct Checked<'a> {
    shape: Vec<Option<usize>>,
    sample_parameters: Vec<FitParameter>,
    measured: Vec<(usize, f64, f64)>,
    lines: Vec<Lines<'a>>,
    instrument: Vec<FitParameter>,
    base: FlightTimeGrid,
    resonances_in_span: Vec<Vec<Vec<f64>>>,
}

pub(crate) fn checked<'a>(
    measurement: &'a Measurement,
    calibration: &Calibration,
) -> Result<Checked<'a>, PipelineError> {
    let Measurement {
        time_edges_us,
        charge_ratio,
        regions,
        ..
    } = measurement;
    let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
    let materials = || {
        regions
            .iter()
            .enumerate()
            .filter_map(|(r, region)| Some((r, region.material.as_ref()?)))
    };
    if !(charge_ratio.is_finite() && *charge_ratio > 0.0) {
        return invalid(format!(
            "the charge ratio must be finite and positive, got {charge_ratio}"
        ));
    }
    if let Some(room) = &measurement.room_background
        && !room.frames.iter().all(|f| f.is_finite() && *f > 0.0)
    {
        return invalid(format!(
            "the room background's frames must be finite and positive, got {:?}",
            room.frames
        ));
    }
    if let Some(spread) = &measurement.spread {
        spread_checked(spread, regions.len())?;
    }
    if let Some((r, region)) = regions
        .iter()
        .enumerate()
        .find(|(_, region)| !(region.pixels.is_finite() && region.pixels > 0.0))
    {
        return invalid(format!(
            "the pixels of region {r} must be finite and positive, got {}",
            region.pixels
        ));
    }
    if let Some(r) = regions.iter().position(|region| {
        region.material.is_none()
            && [region.capture_gammas, region.in_scatter] != [Value::Known(0.0); 2]
    }) {
        return invalid(format!(
            "region {r} holds no material to capture or scatter neutrons, so its capture gammas \
             and in-scatter must be known 0"
        ));
    }
    for (r, material) in materials() {
        if material.isotopes.is_empty() {
            return invalid(format!("the material of region {r} has no isotopes"));
        }
        for (i, (isotope, _)) in material.isotopes.iter().enumerate() {
            if material.isotopes[..i]
                .iter()
                .any(|(other, _)| other.za == isotope.za)
            {
                return invalid(format!(
                    "{} is listed twice in region {r}; its densities cannot be told apart",
                    isotope.isotope
                ));
            }
            if !finite(isotope) {
                return invalid(format!(
                    "the resonance data of {} in region {r} are not finite",
                    isotope.isotope
                ));
            }
        }
        for (k, (i, covariance)) in material.covariances.iter().enumerate() {
            let n = 3 * covariance.resonances.len();
            let listed = &covariance.resonances;
            let placed = material.isotopes.get(*i).is_some_and(|(data, _)| {
                !listed.is_empty()
                    && listed.iter().enumerate().all(|(x, &[a, b, c])| {
                        !listed[..x].contains(&[a, b, c])
                            && data
                                .ranges
                                .get(a)
                                .and_then(|range| range.l_groups.get(b))
                                .is_some_and(|group| c < group.resonances.len())
                    })
            });
            let matrix = &covariance.covariance;
            let sd = |p: usize| matrix[p * n + p].abs().sqrt();
            let symmetric = matrix.len() == n * n
                && (0..n).all(|p| {
                    (0..n).all(|q| {
                        let scale = sd(p) * sd(q);
                        matrix[p * n + q].is_finite()
                            && (matrix[p * n + q] - matrix[q * n + p]).abs() <= 1e-12 * scale
                            && (p == q || scale > 0.0 || matrix[p * n + q] == 0.0)
                    })
                });
            let semidefinite = placed
                && symmetric
                && Mat::<f64>::from_fn(n, n, |p, q| {
                    let scale = sd(p) * sd(q);
                    if scale > 0.0 {
                        matrix[p * n + q] / scale
                    } else {
                        matrix[p * n + q]
                    }
                })
                .self_adjoint_eigenvalues(Side::Lower)
                .is_ok_and(|e| e[0] >= -f64::EPSILON * n as f64 * e[n - 1].max(1.0));
            if material.covariances[..k].iter().any(|(j, _)| j == i) || !semidefinite {
                return invalid(format!(
                    "covariance {k} of region {r} must name an isotope of its material once and \
                     distinct resonances it has, with a finite symmetric positive semi-definite \
                     matrix over their energies and widths"
                ));
            }
        }
    }
    let carried: Vec<(usize, u32)> = materials()
        .flat_map(|(r, m)| {
            m.covariances
                .iter()
                .map(move |(i, _)| (r, m.isotopes[*i].0.za))
        })
        .collect();
    if materials().any(|(s, m)| {
        m.isotopes
            .iter()
            .any(|(d, _)| carried.iter().any(|&(r, za)| r != s && za == d.za))
    }) {
        return invalid(
            "a resonance covariance is carried for an isotope another region also holds, whose \
             counts it would leave out"
                .into(),
        );
    }
    if calibration.pulse.prior.is_some() && materials().any(|(_, m)| !m.covariances.is_empty()) {
        return invalid(
            "a pulse calibration's prior records no sensitivity to the resonance parameters, so a \
             fit under it cannot carry their covariance"
                .into(),
        );
    }
    let shape = shape(measurement);
    let quantity_roles = roles(&vec![0; regions.len()], &shape);
    let sample_parameters = quantity_roles
        .iter()
        .filter_map(|&role| {
            let (value, name, range, allowed) = quantity(measurement, role)?;
            Some(value.parameter(name, range, &allowed))
        })
        .collect::<Result<Vec<FitParameter>, PipelineError>>()?;
    let measured: Vec<(usize, f64, f64)> = quantities(measurement, calibration)
        .enumerate()
        .filter_map(|(offset, (_, value))| match *value {
            Value::Measured { value, sd } => Some((offset, value, sd)),
            _ => None,
        })
        .collect();
    let lines: Vec<Lines<'_>> = materials()
        .map(|(region, material)| {
            let temperature = &sample_parameters[quantity_roles
                .iter()
                .position(|&role| role == Role::Temperature { region })
                .expect("a material has a temperature")];
            let reach_k = if temperature.fixed {
                temperature.value
            } else {
                temperature.upper
            };
            let dopplers = material
                .isotopes
                .iter()
                .map(|(isotope, _)| {
                    DopplerParams::new(reach_k, isotope.awr).map_err(|e| {
                        in_region(region, PipelineError::InvalidParameter(e.to_string()))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Lines {
                region,
                isotopes: &material.isotopes,
                dopplers,
                scatters: measurement.regions[region].in_scatter != Value::Known(0.0),
            })
        })
        .collect::<Result<_, PipelineError>>()?;
    let instrument = calibration.instrument()?;
    validate_pulse_prior(calibration)?;
    let start: Vec<f64> = instrument.iter().map(|parameter| parameter.value).collect();
    let base = FlightTimeGrid::new(
        time_edges_us,
        start[0],
        start[1],
        &calibration.pulse.at(&start[2..])?,
    )?;
    calibrated_lines(
        &lines,
        &calibration.pulse,
        time_edges_us,
        start[0],
        start[1],
    )?;
    let resonances_in_span = in_span(&lines, &base)?;
    Ok(Checked {
        shape,
        sample_parameters,
        measured,
        lines,
        instrument,
        base,
        resonances_in_span,
    })
}

pub(crate) fn excluded_bins(
    time_edges_us: &[f64],
    material: Option<&Material>,
    numbers: &[f64],
) -> Vec<usize> {
    let laws = laws(&numbers[2..]);
    let arrival_us = |energy: f64| {
        numbers[0]
            + TOF_FACTOR * numbers[1] / energy.sqrt()
            + 3.0 / laws.alpha.eval(energy)
            + laws.r.eval(energy) / laws.beta.eval(energy)
    };
    let spans: Vec<(f64, f64)> = material
        .into_iter()
        .flat_map(|m| {
            m.isotopes
                .iter()
                .filter(|(_, density)| *density != Value::Known(0.0))
                .flat_map(|(data, _)| m.library.excluded_windows_ev(&data.isotope))
        })
        .map(|&(low, high)| (arrival_us(high), arrival_us(low)))
        .collect();
    (0..time_edges_us.len() - 1)
        .filter(|&k| {
            spans
                .iter()
                .any(|&(first, last)| time_edges_us[k] < last && first < time_edges_us[k + 1])
        })
        .collect()
}

type Slopes = (Vec<Vec<f64>>, FlatMatrix);

fn nuclear_slopes(
    regions: &[Region],
    model: &Recorded<'_, RegionsModel>,
    dispersion: &[f64],
    parameters: &ParameterSet,
) -> Result<Option<Slopes>, PipelineError> {
    let (params, free) = (parameters.all_values(), parameters.free_indices());
    let mu = model.evaluate(&params)?;
    let jacobian = model
        .analytical_jacobian(&params, &free, &mu)
        .expect("the counts model has an analytical Jacobian");
    let mut picked: Vec<(&ResonanceCovariance, usize, Vec<f64>)> = Vec::new();
    for (r, material) in regions
        .iter()
        .enumerate()
        .filter_map(|(r, region)| Some((r, region.material.as_ref()?)))
    {
        for (i, covariance) in &material.covariances {
            let data = &material.isotopes[*i].0;
            let n = 3 * covariance.resonances.len();
            for p in (0..n).filter(|p| covariance.covariance[p * n + p] > 0.0) {
                let [a, b, c] = covariance.resonances[p / 3];
                let h = 0.1 * covariance.covariance[p * n + p].sqrt();
                let shifted = |by: f64| {
                    let mut copy = data.clone();
                    let resonance = &mut copy.ranges[a].l_groups[b].resonances[c];
                    *[&mut resonance.energy, &mut resonance.gn, &mut resonance.gg][p % 3] += by;
                    copy
                };
                let slope =
                    model
                        .model
                        .resonance_slope(&params, r, *i, [&shifted(h), &shifted(-h)], h)?;
                let cross = (0..free.len())
                    .map(|j| {
                        slope
                            .iter()
                            .enumerate()
                            .filter(|(k, _)| mu[*k] > 0.0)
                            .map(|(k, c)| {
                                jacobian.get(k, j) * c * model.live[k] / (mu[k] * dispersion[k])
                            })
                            .sum()
                    })
                    .collect();
                picked.push((covariance, p, cross));
            }
        }
    }
    let width = picked.len();
    if width == 0 {
        return Ok(None);
    }
    let mut prior = FlatMatrix::zeros(width, width);
    for (a, (first, p, _)) in picked.iter().enumerate() {
        for (b, (second, q, _)) in picked.iter().enumerate() {
            if std::ptr::eq(*first, *second) {
                *prior.get_mut(a, b) = first.covariance[p * 3 * first.resonances.len() + q];
            }
        }
    }
    Ok(Some((
        picked.into_iter().map(|(_, _, cross)| cross).collect(),
        prior,
    )))
}

fn sandwich(conditional: &FlatMatrix, slopes: &Slopes) -> FlatMatrix {
    let (crosses, prior) = slopes;
    let (free, width) = (conditional.nrows, crosses.len());
    let resolved: Vec<bool> = (0..free)
        .map(|i| conditional.get(i, i).is_finite())
        .collect();
    let sensitivity: Vec<Vec<f64>> = (0..free)
        .map(|i| {
            crosses
                .iter()
                .map(|cross| {
                    if !resolved[i] {
                        return f64::NAN;
                    }
                    -(0..free)
                        .filter(|&j| resolved[j])
                        .map(|j| conditional.get(i, j) * cross[j])
                        .sum::<f64>()
                })
                .collect()
        })
        .collect();
    let weighted: Vec<Vec<f64>> = sensitivity
        .iter()
        .map(|sx| {
            (0..width)
                .map(|b| (0..width).map(|a| sx[a] * prior.get(a, b)).sum())
                .collect()
        })
        .collect();
    let mut term = FlatMatrix::zeros(free, free);
    for (x, wx) in weighted.iter().enumerate() {
        for (y, sy) in sensitivity.iter().enumerate() {
            *term.get_mut(x, y) = wx.iter().zip(sy).map(|(w, s)| w * s).sum();
        }
    }
    term
}

fn calibrated_lines(
    lines: &[Lines<'_>],
    pulse: &Pulse,
    time_edges_us: &[f64],
    t0_us: f64,
    flight_path_m: f64,
) -> Result<(), PipelineError> {
    for lines in lines {
        if let Some(((low, high), line)) = pulse.line_span_ev.zip(pulse.uncalibrated_line(
            lines.isotopes,
            time_edges_us,
            t0_us,
            flight_path_m,
        )) {
            return Err(PipelineError::InvalidParameter(format!(
                "region {} has a resonance at {line} eV, outside the {low}–{high} eV of the \
                 resonances the pulse was calibrated on",
                lines.region
            )));
        }
    }
    Ok(())
}

fn in_span(
    lines: &[Lines<'_>],
    grid: &FlightTimeGrid,
) -> Result<Vec<Vec<Vec<f64>>>, PipelineError> {
    let energies = grid.energies_ev();
    let span_ev = (energies[energies.len() - 1], energies[0]);
    lines
        .iter()
        .map(|lines| {
            if lines.scatters {
                let lowest = span_ev.0
                    * lines
                        .isotopes
                        .iter()
                        .map(|(isotope, _)| lowest_exit_ratio(isotope.awr))
                        .fold(1.0, f64::min);
                if let Some((isotope, e)) = lines.isotopes.iter().find_map(|(isotope, _)| {
                    resonance_center_energies(&[isotope])
                        .into_iter()
                        .find(|e| (lowest..span_ev.0).contains(e))
                        .map(|e| (isotope, e))
                }) {
                    return Err(PipelineError::InvalidParameter(format!(
                        "the in-scatter of region {} leaves with energies down to {lowest:.6e} \
                         eV, below the flight-time grid's lowest {:.6e} eV, and {} has a \
                         resonance at {e:.6e} eV there",
                        lines.region, span_ev.0, isotope.isotope
                    )));
                }
            }
            lines
                .isotopes
                .iter()
                .zip(&lines.dopplers)
                .map(|((isotope, _), doppler)| {
                    let read = (
                        (span_ev.0.sqrt() - SUPPORT_X * doppler.u())
                            .max(0.0)
                            .powi(2),
                        (span_ev.1.sqrt() + SUPPORT_X * doppler.u()).powi(2),
                    );
                    if !isotope.ranges.iter().any(|range| {
                        range.is_evaluable()
                            && range.energy_low <= read.0
                            && read.1 <= range.energy_high
                    }) {
                        return Err(PipelineError::InvalidParameter(format!(
                            "the Doppler-broadened cross section of {} in region {} reads \
                             {:.6e}–{:.6e} eV, which no single one of its evaluated (SLBW, \
                             MLBW or Reich–Moore) resolved ranges holds",
                            isotope.isotope, lines.region, read.0, read.1
                        )));
                    }
                    Ok(resonance_center_energies(&[isotope])
                        .into_iter()
                        .filter(|e| (span_ev.0..=span_ev.1).contains(e))
                        .collect())
                })
                .collect()
        })
        .collect()
}

pub(crate) enum BeamStarts<'a> {
    Given(&'a Starts),
    OpenFitsAt(Bypass, Option<Vec<usize>>),
}

/// The spread's shares `F_rq` between the measurement's regions, or `None`
/// without a spread.
pub(crate) fn spread_shares(
    measurement: &Measurement,
) -> Result<Option<Arc<Vec<Vec<f64>>>>, PipelineError> {
    let Some(spread) = &measurement.spread else {
        return Ok(None);
    };
    spread_checked(spread, measurement.regions.len())?;
    Ok(Some(Arc::new(region_shares(
        &spread.kernel,
        spread.grid,
        &spread.cells,
    ))))
}

/// The fit of `measurement` from `beam_starts`, with the spread's shares
/// `mixing` from [`spread_shares`].
pub(crate) fn fit_counts_answer(
    measurement: &Measurement,
    calibration: &Calibration,
    beam_starts: BeamStarts<'_>,
    mixing: Option<Arc<Vec<Vec<f64>>>>,
) -> Result<(CountsFit, Answer), PipelineError> {
    let starts = match beam_starts {
        BeamStarts::Given(starts) => Some(starts),
        BeamStarts::OpenFitsAt(..) => None,
    };
    let Measurement {
        time_edges_us,
        charge_ratio,
        regions,
        ..
    } = measurement;
    let Checked {
        shape,
        sample_parameters,
        measured,
        lines,
        instrument,
        mut base,
        mut resonances_in_span,
    } = checked(measurement, calibration)?;
    let pulse = &calibration.pulse;
    let start: Vec<f64> = instrument.iter().map(|parameter| parameter.value).collect();
    let beam_origin_us = starts.map_or(start[0], |starts| starts.origin_us);
    let frames = measurement
        .room_background
        .map_or([0.0; 2], |room| room.frames);
    let spreading = measurement
        .spread
        .as_ref()
        .is_some_and(|spread| spread.share != Value::Known(0.0));
    let bins = time_edges_us.len() - 1;
    if let Some(starts) = starts {
        assert_eq!(starts.beams.len(), regions.len(), "one beam per region");
        if let Some(leverage) = &starts.shared_leverage {
            assert_eq!(leverage.len(), regions.len(), "one leverage per region");
        }
    }
    let excluded_at = |numbers: &[f64]| -> Vec<Vec<usize>> {
        let own: Vec<Vec<usize>> = regions
            .iter()
            .map(|region| excluded_bins(time_edges_us, region.material.as_ref(), numbers))
            .collect();
        match &mixing {
            Some(shares) if spreading => shares
                .iter()
                .enumerate()
                .map(|(r, from)| {
                    let mut out: Vec<usize> = own
                        .iter()
                        .enumerate()
                        .filter(|&(q, _)| q == r || from[q] > 0.0)
                        .flat_map(|(_, out)| out.iter().copied())
                        .collect();
                    out.sort_unstable();
                    out.dedup();
                    out
                })
                .collect(),
            _ => own,
        }
    };
    let dispersion_of = |excluded: &[Vec<usize>], weights: &[f64]| -> Vec<f64> {
        let mut dispersion: Vec<f64> = (0..2 * regions.len() * bins)
            .map(|k| weights[k / bins])
            .collect();
        for (r, out) in excluded.iter().enumerate() {
            for &k in out {
                dispersion[(2 * r + 1) * bins + k] = f64::INFINITY;
            }
        }
        dispersion
    };
    let mut excluded = starts.map_or_else(|| excluded_at(&start), |s| s.excluded.clone());
    let mut live = Vec::with_capacity(2 * regions.len() * bins);
    for (r, region) in regions.iter().enumerate() {
        let checked = || -> Result<[Vec<f64>; 2], PipelineError> {
            validate_counts("open-beam", &region.open_counts, bins)?;
            validate_counts("sample", &region.sample_counts, bins)?;
            Ok([
                validate_live("open-beam", region.open_live.as_deref(), bins)?,
                validate_live("sample", region.sample_live.as_deref(), bins)?,
            ])
        };
        live.extend(checked().map_err(|e| in_region(r, e))?.concat());
    }

    let opens: Vec<BeamStart> = match beam_starts {
        BeamStarts::Given(starts) => starts.beams.clone(),
        BeamStarts::OpenFitsAt(numbers, ref intervals) => regions
            .iter()
            .enumerate()
            .map(|(r, region)| {
                let per_pixel = numbers.open_counts_per_pixel(
                    &starts_and_widths_us(time_edges_us),
                    open_frames(measurement.room_background.as_ref()),
                );
                let known: Vec<f64> = per_pixel.iter().map(|g| region.pixels * g).collect();
                let open = open_beam_with_intervals(
                    time_edges_us,
                    &region.open_counts,
                    calibration,
                    Some(&live[2 * r * bins..(2 * r + 1) * bins]),
                    Some(&known),
                    intervals.as_ref().map(|chosen| chosen[r]),
                )
                .map_err(|e| in_region(r, e))?;
                Ok(BeamStart::from(open))
            })
            .collect::<Result<_, PipelineError>>()?,
    };
    let beams: Vec<BeamSpline> = opens.iter().map(|open| open.beam.clone()).collect();
    let sizes: Vec<usize> = beams.iter().map(|beam| beam.coefficients().len()).collect();
    let pixels: Vec<f64> = regions.iter().map(|region| region.pixels).collect();
    let switches: Vec<[bool; 2]> = regions
        .iter()
        .map(|region| {
            [region.capture_gammas, region.in_scatter].map(|value| value != Value::Known(0.0))
        })
        .collect();
    let layout = Arc::new(Layout::new(roles(&sizes, &shape), &pixels, &switches));
    let mut parameters = ParameterSet::new(
        beams
            .iter()
            .enumerate()
            .flat_map(|(r, beam)| {
                beam.coefficients().iter().enumerate().map(move |(i, &c)| {
                    FitParameter::unbounded(format!("beam {i} of region {r}"), c)
                })
            })
            .chain(sample_parameters)
            .chain(instrument)
            .collect(),
    );
    let resonances: Vec<Arc<[ResonanceData]>> = regions
        .iter()
        .map(|region| {
            region
                .material
                .iter()
                .flat_map(|material| material.isotopes.iter().map(|(data, _)| data.clone()))
                .collect()
        })
        .collect();
    let observed: Vec<f64> = regions
        .iter()
        .flat_map(|region| region.open_counts.iter().chain(&region.sample_counts))
        .copied()
        .collect();
    let calibrated_parameters: Vec<usize> = pulse
        .prior
        .iter()
        .flat_map(|prior| prior.numbers.iter().map(|n| layout.pulse + n))
        .collect();
    let pulse_prior = pulse
        .prior
        .as_ref()
        .map(|prior| Prior::correlated(&calibrated_parameters, &prior.mean, &prior.covariance))
        .transpose()?;
    let priors: Vec<Prior> = measured
        .iter()
        .map(|&(offset, mean, sd)| Prior::measured(layout.quantities + offset, mean, sd))
        .chain(pulse_prior.clone())
        .collect();
    let half_maximum = 2.0 * std::f64::consts::LN_2.sqrt();
    let narrowest_us = |grid: &FlightTimeGrid,
                        resonances_in_span: &[Vec<Vec<f64>>],
                        params: &[f64]|
     -> Result<f64, PipelineError> {
        let clock = TOF_FACTOR * grid.flight_path_m();
        let mut narrowest = f64::INFINITY;
        for (lines, energies) in lines.iter().zip(resonances_in_span) {
            let temperature_k = params[layout.regions[lines.region]
                .temperature
                .expect("a material has a temperature")];
            for ((isotope, _), energies) in lines.isotopes.iter().zip(energies) {
                let doppler = DopplerParams::new(temperature_k, isotope.awr)
                    .map_err(|e| PipelineError::InvalidParameter(e.to_string()))?;
                for &energy in energies {
                    let width_ev = half_maximum * doppler.doppler_width(energy);
                    narrowest = narrowest.min(clock / energy.sqrt() * width_ev / (2.0 * energy));
                }
            }
        }
        Ok(narrowest)
    };
    let first_grid = |grid: &FlightTimeGrid,
                      resonances_in_span: &[Vec<Vec<f64>>],
                      params: &[f64]|
     -> Result<(Arc<FlightTimeGrid>, usize), PipelineError> {
        let rule_us = 0.5 * narrowest_us(grid, resonances_in_span, params)?;
        let mut first = grid.clone();
        let mut halvings = 0;
        while first.step_us() > rule_us {
            first = first.halved()?;
            halvings += 1;
        }
        Ok((Arc::new(first), halvings))
    };
    let starting: Vec<f64> = parameters.params.iter().map(|p| p.value).collect();
    let mut first = first_grid(&base, &resonances_in_span, &starting)?;
    let mut weights: Vec<f64> = opens
        .iter()
        .flat_map(|open| [open.overdispersion.unwrap_or(1.0); 2])
        .collect();
    let mut sample_measured = vec![false; regions.len()];
    let mut noise_bins: Option<Vec<Vec<usize>>> = None;
    let mut passes = 0;
    let (fit, rule_halvings, overdispersions, settled) = loop {
        passes += 1;
        let dispersion = dispersion_of(&excluded, &weights);
        let fit = fit_on_halved_grids(
            &first.0,
            &mut parameters,
            &observed,
            &dispersion,
            &priors,
            |grid| {
                Ok(Recorded {
                    model: RegionsModel::new(
                        grid,
                        &beams,
                        &resonances,
                        (*charge_ratio, frames),
                        beam_origin_us,
                        &layout,
                    )
                    .spread(mixing.clone()),
                    live: &live,
                })
            },
            |spread, grid, params| {
                Ok(spread <= starts.map_or(BOUND, |starts| starts.bound)
                    || !grid.covers(
                        params[layout.t0],
                        params[layout.flight_path],
                        &laws(&params[layout.pulse..]),
                    )?)
            },
        )?;
        let fitted = |index: usize| fit.result.params[index];
        let noise_bins = noise_bins.get_or_insert_with(|| {
            (0..regions.len())
                .map(|r| counted(&fit, (2 * r + 1) * bins..(2 * r + 2) * bins))
                .collect()
        });
        let samples: Vec<Option<f64>> = noise_bins
            .iter()
            .enumerate()
            .map(|(r, counted)| {
                let weighted: Vec<usize> = counted
                    .iter()
                    .copied()
                    .filter(|&k| dispersion[k].is_finite())
                    .collect();
                overdispersion(
                    &observed,
                    &fit,
                    &weighted,
                    starts
                        .and_then(|starts| starts.shared_leverage.as_ref())
                        .map_or(0.0, |extra| extra[r]),
                )
            })
            .collect();
        let next: Vec<f64> = samples
            .iter()
            .enumerate()
            .map(|(r, sample)| sample.unwrap_or(weights[2 * r]))
            .collect();
        let each_settled: Vec<bool> = next
            .iter()
            .enumerate()
            .map(|(r, next)| (next / weights[2 * r + 1] - 1.0).abs() <= SETTLED_OVERDISPERSION)
            .collect();
        let at_fit = starts.map_or_else(
            || {
                excluded_at(
                    &[fitted(layout.t0), fitted(layout.flight_path)]
                        .into_iter()
                        .chain(fit.result.params[layout.pulse..].iter().copied())
                        .collect::<Vec<f64>>(),
                )
            },
            |s| s.excluded.clone(),
        );
        let settled = each_settled.iter().all(|&s| s) && at_fit == excluded;
        let resolved = 2.0 * fit.step_us
            <= 0.5 * narrowest_us(&base, &resonances_in_span, &fit.result.params)?;
        let covered = fit.converged
            && fit.coarse.covers(
                fitted(layout.t0),
                fitted(layout.flight_path),
                &laws(&fit.result.params[layout.pulse..]),
            )?;
        if !fit.converged || (settled && resolved && covered) || passes == MOST_PASSES {
            let weighted: Vec<[Option<f64>; 2]> = (0..regions.len())
                .map(|r| {
                    [
                        opens[r].overdispersion,
                        (sample_measured[r] || (each_settled[r] && samples[r].is_some()))
                            .then_some(weights[2 * r + 1]),
                    ]
                })
                .collect();
            break (fit, first.1, weighted, settled && resolved && covered);
        }
        if covered {
            let resumed = (Arc::clone(&fit.coarse), first.1 + fit.halvings - 1);
            first = [
                first_grid(&base, &resonances_in_span, &fit.result.params)?,
                resumed,
            ]
            .into_iter()
            .min_by(|a, b| a.0.step_us().total_cmp(&b.0.step_us()))
            .expect("two grids");
        } else {
            base = FlightTimeGrid::new(
                time_edges_us,
                fitted(layout.t0),
                fitted(layout.flight_path),
                &calibration.pulse.at(&fit.result.params[layout.pulse..])?,
            )?;
            resonances_in_span = in_span(&lines, &base)?;
            first = first_grid(&base, &resonances_in_span, &fit.result.params)?;
        }
        for (r, sample) in samples.iter().enumerate() {
            weights[2 * r + 1] = next[r];
            sample_measured[r] = sample.is_some();
        }
        excluded = at_fit;
    };
    let dispersion = dispersion_of(&excluded, &weights);
    let converged = fit.converged && settled;
    if converged {
        calibrated_lines(
            &lines,
            pulse,
            time_edges_us,
            fit.result.params[layout.t0],
            fit.result.params[layout.flight_path],
        )?;
    }

    if let Some((k, (&counts, &predicted))) =
        observed
            .iter()
            .zip(&fit.predicted)
            .enumerate()
            .find(|(k, (y, mu))| {
                !mu.is_finite()
                    || **mu < 0.0
                    || (dispersion[*k].is_finite() && **y > 0.0 && **mu < NEGLIGIBLE_PREDICTION)
            })
    {
        return Err(PipelineError::UnmodelledCounts {
            region: k / (2 * bins),
            run: ["open-beam", "sample"][(k / bins) % 2],
            bin: k % bins,
            counts,
            predicted,
        });
    }

    let carried = regions
        .iter()
        .flat_map(|region| &region.material)
        .any(|m| !m.covariances.is_empty());
    let slopes = if converged && carried && fit.result.covariance.is_some() {
        let model = Recorded {
            model: RegionsModel::new(
                &fit.fine,
                &beams,
                &resonances,
                (*charge_ratio, frames),
                beam_origin_us,
                &layout,
            )
            .spread(mixing.clone()),
            live: &live,
        };
        nuclear_slopes(regions, &model, &dispersion, &parameters)?
    } else {
        None
    };
    let carried_with = |c: Option<&FlatMatrix>| Some(sandwich(c?, slopes.as_ref()?));
    let nuclear = carried_with(fit.result.covariance.as_ref());
    let unbounded_nuclear = carried_with(fit.result.unbounded.as_ref().map(|u| &u.covariance));
    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    let free = parameters.free_indices();
    let on_edge = layout
        .regions
        .iter()
        .filter_map(|region| region.temperature)
        .any(|t| free.contains(&t) && [t_low, t_high].contains(&fit.result.params[t]));
    let sample_quantities: Vec<usize> = (0..free.len())
        .filter(|&p| free[p] >= layout.quantities)
        .collect();
    let block = |full: &FlatMatrix| {
        let size = sample_quantities.len();
        let mut block = FlatMatrix::zeros(size, size);
        for (a, &p) in sample_quantities.iter().enumerate() {
            for (b, &q) in sample_quantities.iter().enumerate() {
                *block.get_mut(a, b) = if on_edge { f64::NAN } else { full.get(p, q) };
            }
        }
        block
    };
    let covariance = fit
        .result
        .covariance
        .as_ref()
        .filter(|_| converged)
        .map(|conditional| {
            let mut full = conditional.clone();
            for (entry, added) in full
                .data
                .iter_mut()
                .zip(nuclear.iter().flat_map(|n| &n.data))
            {
                *entry += added;
            }
            block(&full)
        });
    let unbounded = fit
        .result
        .unbounded
        .as_ref()
        .filter(|_| converged)
        .map(|full| Unbounded {
            mean: sample_quantities
                .iter()
                .map(|&p| if on_edge { f64::NAN } else { full.mean[p] })
                .collect(),
            covariance: block(&full.covariance),
        });
    let params = &fit.result.params;
    let on_bound: Vec<bool> = sample_quantities
        .iter()
        .map(|&p| fit.result.on_bound[p])
        .collect();
    let position = |parameter: usize| {
        sample_quantities
            .iter()
            .position(|&p| free[p] == parameter)
            .expect("a measured or calibrated quantity is fitted")
    };
    let pulse_consistency = match (&pulse_prior, &unbounded) {
        (Some(prior), Some(unbounded)) => {
            let at: Vec<usize> = calibrated_parameters
                .iter()
                .map(|&parameter| position(parameter))
                .collect();
            let mut posterior = FlatMatrix::zeros(at.len(), at.len());
            for (a, &i) in at.iter().enumerate() {
                for (b, &j) in at.iter().enumerate() {
                    *posterior.get_mut(a, b) = unbounded.covariance.get(i, j);
                }
            }
            let estimate: Vec<f64> = at.iter().map(|&i| unbounded.mean[i]).collect();
            consistency(prior, &estimate, &posterior)?
        }
        _ => None,
    };
    let measured_pulls = unbounded.as_ref().map(|unbounded| {
        measured
            .iter()
            .map(|&(offset, mean, sd)| {
                let a = position(layout.quantities + offset);
                let p = sample_quantities[a];
                let nuclear = unbounded_nuclear.as_ref().map_or(0.0, |n| n.get(p, p));
                (unbounded.mean[a] - mean)
                    / (sd * sd - unbounded.covariance.get(a, a) + nuclear).sqrt()
            })
            .collect()
    });
    let region_fits = layout
        .regions
        .iter()
        .zip(&opens)
        .zip(overdispersions)
        .enumerate()
        .map(|(r, ((place, open), overdispersion))| RegionFit {
            densities: params[place.densities.clone()].to_vec(),
            temperature_k: place.temperature.map(|t| params[t]),
            thickness_variance: place.thickness.map(|v| params[v]),
            background: std::array::from_fn(|i| params[place.background + i]),
            gamma_transmission: params[place.gamma],
            capture_gammas: params[place.capture],
            in_scatter: params[place.scatter],
            beam: open.beam.with_coefficients(&params[place.beam.clone()]),
            beam_at_limit: open.at_limit,
            predicted: [
                fit.predicted[2 * r * bins..(2 * r + 1) * bins].to_vec(),
                fit.predicted[(2 * r + 1) * bins..(2 * r + 2) * bins].to_vec(),
            ],
            overdispersion,
            excluded_bins: excluded[r].clone(),
        })
        .collect();
    let mut fitted = vec![false; params.len()];
    let mut at_bound = vec![false; params.len()];
    for (&index, &bound) in free.iter().zip(&fit.result.on_bound) {
        fitted[index] = true;
        at_bound[index] = bound;
    }
    let answer = Answer {
        grid: Arc::clone(&fit.fine),
        dispersion,
        params: params.clone(),
        fitted,
        at_bound,
        beams,
        resonances,
        charge_ratio: *charge_ratio,
        frames,
        beam_origin_us,
        layout: Arc::clone(&layout),
        live,
        observed,
    };
    let fit = CountsFit {
        regions: region_fits,
        normalization: params[layout.normalization],
        t0_us: params[layout.t0],
        flight_path_m: params[layout.flight_path],
        alpha: [params[layout.pulse], params[layout.pulse + 1]],
        beta: [params[layout.pulse + 2], params[layout.pulse + 3]],
        r: params[layout.pulse + 4],
        fwhm_squared_us2: params[layout.pulse + 5],
        r1: params[layout.pulse + 6],
        time_background: [0, 1, 2].map(|i| params[layout.time_background + i]),
        room_background: params[layout.room_background],
        spread_share: params[layout.spread],
        covariance,
        unbounded,
        on_bound,
        deviance: fit.result.deviance,
        converged,
        measured_pulls,
        pulse_consistency,
        step_us: fit.step_us,
        points: fit.points,
        halvings: rule_halvings + fit.halvings,
    };
    Ok((fit, answer))
}

pub(crate) struct Answer {
    grid: Arc<FlightTimeGrid>,
    beams: Vec<BeamSpline>,
    resonances: Vec<Arc<[ResonanceData]>>,
    charge_ratio: f64,
    frames: [f64; 2],
    beam_origin_us: f64,
    layout: Arc<Layout>,
    live: Vec<f64>,
    observed: Vec<f64>,
    dispersion: Vec<f64>,
    params: Vec<f64>,
    fitted: Vec<bool>,
    at_bound: Vec<bool>,
}

pub(crate) struct RegionTerms {
    pub(crate) quantities: Vec<Role>,
    pub(crate) gradient: Vec<f64>,
    pub(crate) information: FlatMatrix,
    pub(crate) diagonal: Vec<f64>,
    pub(crate) unbounded: Option<(FlatMatrix, Vec<f64>)>,
    pub(crate) sensitivity: FlatMatrix,
    pub(crate) covariance: FlatMatrix,
    pub(crate) counted_information: FlatMatrix,
}

impl Answer {
    pub(crate) fn region_terms(
        &self,
        region: usize,
        shared: &[Role],
    ) -> Result<RegionTerms, PipelineError> {
        let roles = &self.layout.roles;
        let own: Vec<usize> = (0..roles.len())
            .filter(|&i| {
                self.fitted[i]
                    && matches!(
                        roles[i],
                        Role::Beam { region: r, .. }
                            | Role::Density { region: r, .. }
                            | Role::Temperature { region: r }
                            | Role::ThicknessVariance { region: r }
                            | Role::Background { region: r, .. }
                            | Role::GammaTransmission { region: r }
                            | Role::CaptureGammas { region: r }
                            | Role::InScatter { region: r }
                            if r == region
                    )
            })
            .collect();
        let shared: Vec<usize> = shared
            .iter()
            .map(|&role| {
                roles
                    .iter()
                    .position(|&r| r == role)
                    .expect("a shared role is in the layout")
            })
            .collect();
        let columns: Vec<usize> = own.iter().chain(&shared).copied().collect();
        let model = Recorded {
            model: RegionsModel::new(
                &self.grid,
                &self.beams,
                &self.resonances,
                (self.charge_ratio, self.frames),
                self.beam_origin_us,
                &self.layout,
            ),
            live: &self.live,
        };
        let predicted = model.evaluate(&self.params)?;
        let jacobian = model
            .analytical_jacobian(&self.params, &columns, &predicted)
            .expect("the counts model has an analytical Jacobian");
        let n = columns.len();
        let mut full = Mat::<f64>::zeros(n, n);
        let mut gradient = vec![0.0; n];
        let bins = self.observed.len() / (2 * self.layout.regions.len());
        let rows = 2 * region * bins..2 * (region + 1) * bins;
        let mut weights = vec![0.0; rows.len()];
        for ((((k, &mu), &dispersion), &observed), weight) in rows
            .clone()
            .zip(&predicted[rows.clone()])
            .zip(&self.dispersion[rows.clone()])
            .zip(&self.observed[rows.clone()])
            .zip(weights.iter_mut())
        {
            if mu > 0.0 {
                *weight = 1.0 / (mu * dispersion);
                for a in 0..n {
                    let slope = jacobian.get(k, a);
                    gradient[a] += *weight * slope * (mu - observed);
                    for b in 0..n {
                        full[(a, b)] += *weight * slope * jacobian.get(k, b);
                    }
                }
            } else {
                for (a, g) in gradient.iter_mut().enumerate() {
                    *g += jacobian.get(k, a) / dispersion;
                }
            }
        }
        let (o, s) = (own.len(), shared.len());
        let counted = rows.len();
        let profile = |set: Vec<usize>| -> Result<Profile, PipelineError> {
            let m = set.len();
            let mut block = FlatMatrix::zeros(m, m);
            for (a, &i) in set.iter().enumerate() {
                for (b, &j) in set.iter().enumerate() {
                    *block.get_mut(a, b) = full[(i, j)];
                }
            }
            let diagonal: Vec<f64> = set.iter().map(|&i| full[(i, i)]).collect();
            let inverse = information_inverse(&block, &diagonal, counted)?;
            let mut moved = FlatMatrix::zeros(m, s);
            for a in 0..m {
                for j in 0..s {
                    *moved.get_mut(a, j) = -(0..m)
                        .map(|b| inverse.spanned.get(a, b) * full[(set[b], o + j)])
                        .sum::<f64>();
                }
            }
            let profiled = |i: usize, j: usize| {
                full[(o + i, o + j)]
                    + (0..m)
                        .map(|a| full[(o + i, set[a])] * moved.get(a, j))
                        .sum::<f64>()
            };
            let mut information = FlatMatrix::zeros(s, s);
            for i in 0..s {
                for j in 0..s {
                    *information.get_mut(i, j) = 0.5 * (profiled(i, j) + profiled(j, i));
                }
            }
            let slope = (0..s)
                .map(|j| {
                    gradient[o + j]
                        + (0..m)
                            .map(|a| moved.get(a, j) * gradient[set[a]])
                            .sum::<f64>()
                })
                .collect();
            Ok(Profile {
                set,
                inverse: inverse.determined,
                resolved: inverse.resolved,
                moved,
                information,
                gradient: slope,
            })
        };
        let bounded = profile((0..o).filter(|&a| !self.at_bound[own[a]]).collect())?;
        let unbounded = profile((0..o).collect())?;
        let Profile {
            set,
            inverse,
            resolved,
            moved,
            information,
            gradient,
        } = bounded;
        let quantities: Vec<usize> = (0..set.len())
            .filter(|&a| !matches!(roles[own[set[a]]], Role::Beam { .. }))
            .collect();
        let mut sensitivity = FlatMatrix::zeros(quantities.len(), s);
        let mut covariance = FlatMatrix::zeros(quantities.len(), quantities.len());
        for (row, &a) in quantities.iter().enumerate() {
            for j in 0..s {
                *sensitivity.get_mut(row, j) = if resolved[a] {
                    moved.get(a, j)
                } else {
                    f64::NAN
                };
            }
            for (col, &b) in quantities.iter().enumerate() {
                *covariance.get_mut(row, col) = if resolved[a] && resolved[b] {
                    inverse.get(a, b)
                } else {
                    f64::NAN
                };
            }
        }
        let mut counted_information = FlatMatrix::zeros(s, s);
        let sample = rows.start + counted / 2..rows.end;
        for (k, &weight) in sample.clone().zip(&weights[counted / 2..]) {
            if predicted[k] < COUNTS_TO_MEASURE_NOISE {
                continue;
            }
            let slope: Vec<f64> = (0..s)
                .map(|j| {
                    jacobian.get(k, o + j)
                        + (0..set.len())
                            .map(|a| jacobian.get(k, set[a]) * moved.get(a, j))
                            .sum::<f64>()
                })
                .collect();
            for i in 0..s {
                for j in 0..s {
                    *counted_information.get_mut(i, j) += weight * slope[i] * slope[j];
                }
            }
        }
        Ok(RegionTerms {
            quantities: quantities.iter().map(|&a| roles[own[set[a]]]).collect(),
            gradient,
            information,
            diagonal: (0..s).map(|j| full[(o + j, o + j)]).collect(),
            unbounded: Some((unbounded.information, unbounded.gradient)),
            sensitivity,
            covariance,
            counted_information,
        })
    }
}

struct Profile {
    set: Vec<usize>,
    inverse: FlatMatrix,
    resolved: Vec<bool>,
    moved: FlatMatrix,
    information: FlatMatrix,
    gradient: Vec<f64>,
}

pub(crate) const BACKGROUND_TERMS: [&str; 5] = ["b0", "b1", "b2", "d", "f"];

pub(crate) const SHARED: [Role; 14] = [
    Role::Normalization,
    Role::TimeBackground(0),
    Role::TimeBackground(1),
    Role::TimeBackground(2),
    Role::RoomBackground,
    Role::T0,
    Role::FlightPath,
    Role::Pulse(0),
    Role::Pulse(1),
    Role::Pulse(2),
    Role::Pulse(3),
    Role::Pulse(4),
    Role::Pulse(5),
    Role::Pulse(6),
];

pub(crate) fn shared_parameters(
    measurement: &Measurement,
    calibration: &Calibration,
) -> Result<Vec<(Value, FitParameter)>, PipelineError> {
    validate_pulse_prior(calibration)?;
    let mut parameters = Vec::with_capacity(SHARED.len());
    for (value, name, range, allowed) in SHARED
        .iter()
        .filter_map(|&role| quantity(measurement, role))
    {
        parameters.push(value.parameter(name, range, &allowed)?);
    }
    parameters.extend(calibration.instrument()?);
    let values = quantities(measurement, calibration)
        .filter(|(role, _)| SHARED.contains(role))
        .map(|(_, value)| *value);
    Ok(values.zip(parameters).collect())
}

fn validate_pulse_prior(calibration: &Calibration) -> Result<(), PipelineError> {
    let pulse = &calibration.pulse;
    let numbers = pulse.numbers();
    match pulse
        .prior
        .iter()
        .flat_map(|prior| &prior.numbers)
        .find(|&&n| !matches!(numbers[n], Value::Fitted(_) | Value::Within { .. }))
    {
        Some(&number) => Err(PipelineError::InvalidParameter(format!(
            "the pulse's calibration covers {}, which must be fitted, without a measurement; \
             got {:?}",
            PULSE_NUMBERS[number], numbers[number]
        ))),
        None => Ok(()),
    }
}

pub(crate) fn quantities<'a>(
    measurement: &'a Measurement,
    calibration: &'a Calibration,
) -> impl Iterator<Item = (Role, &'a Value)> {
    let shape = shape(measurement);
    let numbers = calibration.pulse.numbers();
    roles(&vec![0; shape.len()], &shape)
        .into_iter()
        .filter_map(move |role| {
            let value = match role {
                Role::T0 => &calibration.t0_us,
                Role::FlightPath => &calibration.flight_path_m,
                Role::Pulse(number) => numbers[number],
                _ => quantity(measurement, role)?.0,
            };
            Some((role, value))
        })
}

fn shape(measurement: &Measurement) -> Vec<Option<usize>> {
    measurement
        .regions
        .iter()
        .map(|region| region.material.as_ref().map(|m| m.isotopes.len()))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Role {
    Beam { region: usize, coefficient: usize },
    Density { region: usize, isotope: usize },
    Temperature { region: usize },
    ThicknessVariance { region: usize },
    Normalization,
    Background { region: usize, term: usize },
    GammaTransmission { region: usize },
    CaptureGammas { region: usize },
    InScatter { region: usize },
    TimeBackground(usize),
    RoomBackground,
    SpreadShare,
    T0,
    FlightPath,
    Pulse(usize),
}

fn roles(beams: &[usize], isotopes: &[Option<usize>]) -> Vec<Role> {
    let beam = beams.iter().enumerate().flat_map(|(region, &n)| {
        (0..n).map(move |coefficient| Role::Beam {
            region,
            coefficient,
        })
    });
    let materials = isotopes
        .iter()
        .enumerate()
        .filter_map(|(region, n)| Some((region, (*n)?)))
        .flat_map(|(region, n)| {
            (0..n)
                .map(move |isotope| Role::Density { region, isotope })
                .chain([
                    Role::Temperature { region },
                    Role::ThicknessVariance { region },
                ])
        });
    let backgrounds = (0..isotopes.len())
        .flat_map(|region| (0..5).map(move |term| Role::Background { region, term }));
    let gammas = (0..isotopes.len()).map(|region| Role::GammaTransmission { region });
    let captures = (0..isotopes.len()).map(|region| Role::CaptureGammas { region });
    let scatters = (0..isotopes.len()).map(|region| Role::InScatter { region });
    beam.chain(materials)
        .chain([Role::Normalization])
        .chain(backgrounds)
        .chain(gammas)
        .chain(captures)
        .chain(scatters)
        .chain((0..3).map(Role::TimeBackground))
        .chain([
            Role::RoomBackground,
            Role::SpreadShare,
            Role::T0,
            Role::FlightPath,
        ])
        .chain((0..7).map(Role::Pulse))
        .collect()
}

fn material(measurement: &Measurement, region: usize) -> &Material {
    measurement.regions[region]
        .material
        .as_ref()
        .expect("a density or temperature names a region with a material")
}

fn quantity(
    measurement: &Measurement,
    role: Role,
) -> Option<(&Value, String, RangeInclusive<f64>, String)> {
    let (t_low, t_high) = TEMPERATURE_BOUNDS_K;
    Some(match role {
        Role::Density { region, isotope } => {
            let (data, density) = &material(measurement, region).isotopes[isotope];
            (
                density,
                format!("density of {} in region {region}", data.isotope),
                0.0..=f64::INFINITY,
                "0 or more".into(),
            )
        }
        Role::Temperature { region } => (
            &material(measurement, region).temperature_k,
            format!("temperature of region {region}"),
            t_low..=t_high,
            format!("within {t_low}–{t_high} K"),
        ),
        Role::ThicknessVariance { region } => (
            &material(measurement, region).thickness_variance,
            format!("thickness variance of region {region}"),
            0.0..=f64::INFINITY,
            "0 or more".into(),
        ),
        Role::Normalization => (
            &measurement.normalization,
            "normalization".into(),
            f64::MIN_POSITIVE..=f64::INFINITY,
            "positive".into(),
        ),
        Role::Background { region, term } => (
            &measurement.regions[region].background[term],
            format!("{} of region {region}", BACKGROUND_TERMS[term]),
            if term == 4 { 0.0 } else { f64::NEG_INFINITY }..=f64::INFINITY,
            if term == 4 {
                "0 or more"
            } else {
                "of any sign"
            }
            .into(),
        ),
        Role::CaptureGammas { region } => (
            &measurement.regions[region].capture_gammas,
            format!("capture gammas of region {region}"),
            0.0..=f64::INFINITY,
            "0 or more".into(),
        ),
        Role::InScatter { region } => (
            &measurement.regions[region].in_scatter,
            format!("in-scatter of region {region}"),
            0.0..=f64::INFINITY,
            "0 or more".into(),
        ),
        Role::GammaTransmission { region } => (
            &measurement.regions[region].gamma_transmission,
            format!("gamma transmission of region {region}"),
            0.0..=1.0,
            "within 0–1".into(),
        ),
        Role::TimeBackground(term) => {
            let (name, low, allowed) = [
                ("C", 0.0, "0 or more"),
                ("A", 0.0, "0 or more"),
                ("τ", f64::MIN_POSITIVE, "positive"),
            ][term];
            (
                &measurement.time_background[term],
                format!("time background {name}"),
                low..=f64::INFINITY,
                allowed.into(),
            )
        }
        Role::RoomBackground => (
            measurement
                .room_background
                .as_ref()
                .map_or(&Value::Known(0.0), |room| &room.rate),
            "room background ρ".into(),
            0.0..=f64::INFINITY,
            "0 or more".into(),
        ),
        Role::SpreadShare => (
            measurement
                .spread
                .as_ref()
                .map_or(&Value::Known(0.0), |spread| &spread.share),
            "detector spread h".into(),
            0.0..=1.0,
            "within 0–1".into(),
        ),
        Role::Beam { .. } | Role::T0 | Role::FlightPath | Role::Pulse(_) => return None,
    })
}

fn spread_checked(spread: &Spread, regions: usize) -> Result<(), PipelineError> {
    let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
    let [rows, columns] = spread.grid;
    if !rows
        .checked_mul(columns)
        .is_some_and(|cells| (1..=MOST_SPREAD_CELLS).contains(&cells))
    {
        return invalid(format!(
            "the spread's {rows}×{columns} grid must hold 1 to {MOST_SPREAD_CELLS} cells"
        ));
    }
    if spread.kernel.is_empty()
        || !spread
            .kernel
            .iter()
            .all(|&(w, s)| w.is_finite() && w > 0.0 && s.is_finite() && s > 0.0)
    {
        return invalid(format!(
            "the spread kernel needs terms of finite, positive weight and length, got {:?}",
            spread.kernel
        ));
    }
    if spread.cells.len() != regions {
        return invalid(format!(
            "the spread gives the cells of {} regions for {regions} regions",
            spread.cells.len()
        ));
    }
    let mut taken = vec![false; rows * columns];
    for (r, cells) in spread.cells.iter().enumerate() {
        if cells.is_empty() {
            return invalid(format!("the spread gives region {r} no cells"));
        }
        for &[row, column] in cells {
            if row >= rows || column >= columns {
                return invalid(format!(
                    "cell [{row}, {column}] of region {r} is off the {rows}×{columns} grid"
                ));
            }
            if std::mem::replace(&mut taken[row * columns + column], true) {
                return invalid(format!("cell [{row}, {column}] is in two regions"));
            }
        }
    }
    Ok(())
}

fn in_region(region: usize, error: PipelineError) -> PipelineError {
    labelled(&format!("region {region}"), error)
}

pub(crate) fn labelled(label: &str, error: PipelineError) -> PipelineError {
    match error {
        PipelineError::InvalidParameter(message) => {
            PipelineError::InvalidParameter(format!("{label}: {message}"))
        }
        PipelineError::ShapeMismatch(message) => {
            PipelineError::ShapeMismatch(format!("{label}: {message}"))
        }
        other => other,
    }
}

fn finite(isotope: &ResonanceData) -> bool {
    isotope.awr.is_finite()
        && isotope.ranges.iter().all(|range| {
            let radii = range
                .ap_table
                .iter()
                .flat_map(|table| table.points.iter().flat_map(|&(e, r)| [e, r]));
            let external = range.r_external.iter().flat_map(|r| {
                [
                    r.j, r.e_low, r.e_up, r.r_con, r.r_lin, r.s_con, r.s_lin, r.r_quad,
                ]
            });
            let groups = range.l_groups.iter().flat_map(|group| {
                [group.awr, group.apl, group.qx].into_iter().chain(
                    group
                        .resonances
                        .iter()
                        .flat_map(|r| [r.energy, r.j, r.gn, r.gg, r.gfa, r.gfb]),
                )
            });
            [
                range.energy_low,
                range.energy_high,
                range.target_spin,
                range.scattering_radius,
            ]
            .into_iter()
            .chain(radii)
            .chain(external)
            .chain(groups)
            .all(f64::is_finite)
        })
}

struct Lines<'a> {
    region: usize,
    isotopes: &'a [(ResonanceData, Value)],
    dopplers: Vec<DopplerParams>,
    scatters: bool,
}

struct Layout {
    roles: Vec<Role>,
    regions: Vec<Place>,
    quantities: usize,
    normalization: usize,
    time_background: usize,
    room_background: usize,
    spread: usize,
    t0: usize,
    flight_path: usize,
    pulse: usize,
}

struct Place {
    beam: Range<usize>,
    densities: Range<usize>,
    temperature: Option<usize>,
    thickness: Option<usize>,
    background: usize,
    gamma: usize,
    capture: usize,
    capture_on: bool,
    scatter: usize,
    scatter_on: bool,
    pixels: f64,
}

impl Layout {
    fn new(roles: Vec<Role>, pixels: &[f64], switches: &[[bool; 2]]) -> Self {
        let at = |wanted: Role| {
            roles
                .iter()
                .position(|&role| role == wanted)
                .expect("every region has a background and the instrument its numbers")
        };
        let span = |of: &dyn Fn(Role) -> bool| {
            let first = roles.iter().position(|&role| of(role));
            let last = roles.iter().rposition(|&role| of(role));
            first
                .zip(last)
                .map_or(0..0, |(first, last)| first..last + 1)
        };
        let places = (0..pixels.len())
            .map(|r| Place {
                beam: span(&|role| matches!(role, Role::Beam { region, .. } if region == r)),
                densities: span(
                    &|role| matches!(role, Role::Density { region, .. } if region == r),
                ),
                temperature: roles
                    .iter()
                    .position(|&role| role == Role::Temperature { region: r }),
                thickness: roles
                    .iter()
                    .position(|&role| role == Role::ThicknessVariance { region: r }),
                background: at(Role::Background { region: r, term: 0 }),
                gamma: at(Role::GammaTransmission { region: r }),
                capture: at(Role::CaptureGammas { region: r }),
                capture_on: switches[r][0],
                scatter: at(Role::InScatter { region: r }),
                scatter_on: switches[r][1],
                pixels: pixels[r],
            })
            .collect();
        Self {
            quantities: roles
                .iter()
                .position(|role| !matches!(role, Role::Beam { .. }))
                .expect("the instrument's numbers follow the beams"),
            normalization: at(Role::Normalization),
            time_background: at(Role::TimeBackground(0)),
            room_background: at(Role::RoomBackground),
            spread: at(Role::SpreadShare),
            t0: at(Role::T0),
            flight_path: at(Role::FlightPath),
            pulse: at(Role::Pulse(0)),
            regions: places,
            roles,
        }
    }
}

struct RegionsModel {
    grid: Arc<FlightTimeGrid>,
    bins: Vec<(f64, f64)>,
    splines: Vec<BeamSpline>,
    beam_origin_us: f64,
    absorbers: Vec<Absorber>,
    energies: Vec<f64>,
    flight_energies: Vec<f64>,
    shapes: [Vec<f64>; 3],
    charge_ratio: f64,
    frames: [f64; 2],
    layout: Arc<Layout>,
    scale: RefCell<Option<Scale>>,
    mixing: Option<Arc<Vec<Vec<f64>>>>,
}

struct Absorber {
    isotopes: Arc<[ResonanceData]>,
    cross_sections: [RefCell<Option<CrossSections>>; 3],
}

const CHANNELS: [Channel; 3] = [Channel::Total, Channel::Capture, Channel::Elastic];
const TOTAL: usize = 0;
const CAPTURE: usize = 1;
const ELASTIC: usize = 2;

struct Beams {
    open: Vec<f64>,
    normalized: Vec<f64>,
    sample: Vec<f64>,
    per_depth: Vec<f64>,
    per_variance: Vec<f64>,
    capture: Option<Captured>,
    scatter: Option<Scattered>,
}

/// The capture gammas' counts per grid point before the pulse, as slopes:
/// in `ε`, in the optical depth through the interacting fraction, per unit
/// capture cross section, and in the thickness variance.
struct Captured {
    per_efficiency: Vec<f64>,
    per_depth: Vec<f64>,
    share: Vec<f64>,
    per_variance: Vec<f64>,
}

/// The in-scatter's counts per grid point before the pulse per unit `Ω`, and
/// each isotope's share leaving through the back face with its slopes.
struct Scattered {
    per_acceptance: Vec<f64>,
    shares: Vec<ForwardYield>,
}

struct CrossSections {
    temperature_k: f64,
    values: Vec<Vec<f64>>,
    slopes: Vec<Vec<f64>>,
}

struct Scale {
    t0_us: f64,
    flight_path_m: f64,
    pulse: IkedaCarpenterParams,
    rows: Rows,
    basis: Vec<Vec<[(usize, f64); 5]>>,
    basis_slope: Vec<Vec<[(usize, f64); 5]>>,
}

impl Scale {
    fn new(
        (t0_us, flight_path_m): (f64, f64),
        pulse: IkedaCarpenterParams,
        rows: Rows,
        grid: &FlightTimeGrid,
        splines: &[BeamSpline],
        beam_origin_us: f64,
    ) -> Self {
        let (shift, stretch) = (t0_us - beam_origin_us, flight_path_m / grid.flight_path_m());
        let abscissae: Vec<f64> = grid
            .flight_times_us()
            .iter()
            .map(|u| shift + stretch * u)
            .collect();
        Self {
            t0_us,
            flight_path_m,
            pulse,
            rows,
            basis: splines
                .iter()
                .map(|spline| abscissae.iter().map(|&a| spline.basis(a)).collect())
                .collect(),
            basis_slope: splines
                .iter()
                .map(|spline| abscissae.iter().map(|&a| spline.basis_slope(a)).collect())
                .collect(),
        }
    }
}

fn predicted(rows: &Rows, values: &[f64]) -> Result<Vec<f64>, FittingError> {
    rows.predict(values)
        .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
}

pub(crate) fn starts_and_widths_us(time_edges_us: &[f64]) -> Vec<(f64, f64)> {
    time_edges_us
        .windows(2)
        .map(|edges| (edges[0] - time_edges_us[0], edges[1] - edges[0]))
        .collect()
}

pub(crate) fn time_counts_per_pixel(
    starts_and_widths_us: &[(f64, f64)],
    [c, a, tau]: [f64; 3],
) -> (Vec<f64>, [Vec<f64>; 3]) {
    let bins = starts_and_widths_us.len();
    let mut counts = Vec::with_capacity(bins);
    let mut slopes: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(bins));
    for &(start, width) in starts_and_widths_us {
        let decay = (-start / tau).exp();
        let after = (-(start + width) / tau).exp();
        let fraction = -(-width / tau).exp_m1();
        let integral = decay * (tau * fraction);
        counts.push(c * width + a * integral);
        slopes[0].push(width);
        slopes[1].push(integral);
        slopes[2].push(a * (decay * fraction + (decay * start * fraction - after * width) / tau));
    }
    (counts, slopes)
}

impl RegionsModel {
    fn new(
        grid: &Arc<FlightTimeGrid>,
        beams: &[BeamSpline],
        isotopes: &[Arc<[ResonanceData]>],
        (charge_ratio, frames): (f64, [f64; 2]),
        beam_origin_us: f64,
        layout: &Arc<Layout>,
    ) -> Self {
        let flight_energies = grid.energies_ev();
        let mut energies = flight_energies.clone();
        let shapes = [
            vec![1.0; energies.len()],
            energies.iter().map(|e| 1.0 / e.sqrt()).collect(),
            energies.iter().map(|e| e.sqrt()).collect(),
        ];
        energies.reverse();
        Self {
            grid: Arc::clone(grid),
            bins: starts_and_widths_us(grid.time_edges_us()),
            splines: beams.to_vec(),
            beam_origin_us,
            absorbers: isotopes
                .iter()
                .map(|isotopes| Absorber {
                    isotopes: Arc::clone(isotopes),
                    cross_sections: [RefCell::new(None), RefCell::new(None), RefCell::new(None)],
                })
                .collect(),
            energies,
            flight_energies,
            shapes,
            charge_ratio,
            frames,
            layout: Arc::clone(layout),
            scale: RefCell::new(Some(Scale::new(
                (grid.t0_us(), grid.flight_path_m()),
                grid.pulse().params().clone(),
                grid.rows().clone(),
                grid,
                beams,
                beam_origin_us,
            ))),
            mixing: None,
        }
    }

    /// The model with the spread's shares `F_rq` between the regions.
    fn spread(mut self, mixing: Option<Arc<Vec<Vec<f64>>>>) -> Self {
        self.mixing = mixing;
        self
    }

    fn at(
        &self,
        region: usize,
        temperature_k: f64,
        channel: usize,
    ) -> Result<std::cell::Ref<'_, CrossSections>, FittingError> {
        let absorber = &self.absorbers[region];
        let current = absorber.cross_sections[channel]
            .borrow()
            .as_ref()
            .is_some_and(|c| c.temperature_k.to_bits() == temperature_k.to_bits());
        if !current {
            let energies = &self.energies;
            let (values, slopes) = absorber
                .isotopes
                .par_iter()
                .map(|isotope| {
                    let broadened = broaden_with_budget(
                        energies,
                        isotope,
                        temperature_k,
                        CHANNELS[channel],
                        QuadratureBudget::default(),
                    )
                    .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?;
                    let (mut sigma, mut slope) = (broadened.values, broadened.derivatives);
                    sigma.reverse();
                    slope.reverse();
                    Ok((sigma, slope))
                })
                .collect::<Result<Vec<_>, FittingError>>()?
                .into_iter()
                .unzip();
            *absorber.cross_sections[channel].borrow_mut() = Some(CrossSections {
                temperature_k,
                values,
                slopes,
            });
        }
        Ok(std::cell::Ref::map(
            absorber.cross_sections[channel].borrow(),
            |c| c.as_ref().expect("computed above"),
        ))
    }

    fn scale(&self, params: &[f64]) -> Result<std::cell::Ref<'_, Scale>, FittingError> {
        let (t0_us, flight_path_m) = (params[self.layout.t0], params[self.layout.flight_path]);
        let pulse = laws(&params[self.layout.pulse..]);
        let key = |t0: f64, l: f64| (t0.to_bits(), l.to_bits());
        let current = self.scale.borrow().as_ref().is_some_and(|s| {
            key(s.t0_us, s.flight_path_m) == key(t0_us, flight_path_m) && s.pulse == pulse
        });
        if !current {
            let rows = self
                .grid
                .rows_at(t0_us, flight_path_m, &pulse)
                .map_err(|e| FittingError::EvaluationFailed(e.to_string()))?;
            *self.scale.borrow_mut() = Some(Scale::new(
                (t0_us, flight_path_m),
                pulse,
                rows,
                &self.grid,
                &self.splines,
                self.beam_origin_us,
            ));
        }
        Ok(std::cell::Ref::map(self.scale.borrow(), |s| {
            s.as_ref().expect("computed above")
        }))
    }

    fn beams(&self, params: &[f64], scale: &Scale, region: usize) -> Result<Beams, FittingError> {
        let place = &self.layout.regions[region];
        let densities = &params[place.densities.clone()];
        let sigma = place
            .temperature
            .map(|t| self.at(region, params[t], TOTAL))
            .transpose()?;
        let open: Vec<f64> = combined(&scale.basis[region], &params[place.beam.clone()])
            .into_iter()
            .map(f64::exp)
            .collect();
        let normalized: Vec<f64> = open
            .iter()
            .map(|phi| self.charge_ratio * params[self.layout.normalization] * phi)
            .collect();
        let terms = &params[place.background..place.background + 5];
        let exponential = self.exponential(terms[4]);
        let variance = place.thickness.map_or(0.0, |v| params[v]);
        let capture_sigma = match place.temperature {
            Some(t) if place.capture_on => Some(self.at(region, params[t], CAPTURE)?),
            _ => None,
        };
        let efficiency = params[place.capture];
        let n = normalized.len();
        let depths: Vec<f64> = (0..n)
            .map(|j| {
                sigma
                    .iter()
                    .flat_map(|sigma| densities.iter().zip(&sigma.values))
                    .map(|(n, sigma)| n * sigma[j])
                    .sum()
            })
            .collect();
        let elastic = match place.temperature {
            Some(t) if place.scatter_on => Some(self.at(region, params[t], ELASTIC)?),
            _ => None,
        };
        let acceptance = params[place.scatter];
        let mut scatter = elastic
            .as_ref()
            .map(|_| {
                Ok::<_, FittingError>(Scattered {
                    per_acceptance: Vec::with_capacity(n),
                    shares: self.absorbers[region]
                        .isotopes
                        .iter()
                        .map(|isotope| {
                            forward_yield(&self.flight_energies, &depths, isotope.awr)
                                .map_err(|e| FittingError::EvaluationFailed(e.to_string()))
                        })
                        .collect::<Result<_, _>>()?,
                })
            })
            .transpose()?;
        let (mut sample, mut per_depth, mut per_variance) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let mut capture = capture_sigma.as_ref().map(|_| Captured {
            per_efficiency: Vec::with_capacity(n),
            per_depth: Vec::with_capacity(n),
            share: Vec::with_capacity(n),
            per_variance: Vec::with_capacity(n),
        });
        for (j, beam) in normalized.iter().enumerate() {
            let depth = depths[j];
            let mut captured = 0.0;
            if let (Some(capture), Some(capture_sigma)) = (capture.as_mut(), &capture_sigma) {
                let absorbing: f64 = densities
                    .iter()
                    .zip(&capture_sigma.values)
                    .map(|(n, sigma)| n * sigma[j])
                    .sum();
                let (fraction, along_depth, along_variance) =
                    interacting_per_depth(depth, variance);
                captured = efficiency * absorbing * fraction;
                capture.per_efficiency.push(beam * absorbing * fraction);
                capture
                    .per_depth
                    .push(beam * efficiency * absorbing * along_depth);
                capture.share.push(beam * efficiency * fraction);
                capture
                    .per_variance
                    .push(beam * efficiency * absorbing * along_variance);
            }
            let background: f64 = terms
                .iter()
                .zip(&self.shapes)
                .map(|(b, g)| b * g[j])
                .sum::<f64>()
                + terms[3] * exponential[j];
            let mut scattered = 0.0;
            if let (Some(scatter), Some(elastic)) = (scatter.as_mut(), &elastic) {
                let per_acceptance: f64 = densities
                    .iter()
                    .zip(&elastic.values)
                    .zip(&scatter.shares)
                    .map(|((n, sigma), share)| n * sigma[j] * share.share[j])
                    .sum();
                scattered = acceptance * per_acceptance;
                scatter.per_acceptance.push(beam * per_acceptance);
            }
            let (transmission, along_depth, along_variance) = uneven_transmission(depth, variance);
            sample.push(beam * (transmission + background + captured + scattered));
            per_depth.push(beam * along_depth);
            per_variance.push(beam * along_variance);
        }
        Ok(Beams {
            open,
            normalized,
            sample,
            per_depth,
            per_variance,
            capture,
            scatter,
        })
    }

    /// The slope of the counts of every bin of both runs of every region, in
    /// one resonance parameter of region `region`'s isotope `isotope`, from the
    /// isotope's data with the parameter moved by `+h` and by `−h`.
    fn resonance_slope(
        &self,
        params: &[f64],
        region: usize,
        isotope: usize,
        [up, down]: [&ResonanceData; 2],
        h: f64,
    ) -> Result<Vec<f64>, PipelineError> {
        let place = &self.layout.regions[region];
        let temperature = params[place.temperature.expect("a material")];
        let density = params[place.densities.start + isotope];
        let scale = self.scale(params)?;
        let beams = self.beams(params, &scale, region)?;
        let slope = |channel| -> Result<Vec<f64>, PipelineError> {
            let sigma = |data| {
                broaden_channel(&self.energies, data, temperature, channel).map_err(|e| {
                    PipelineError::Fitting(FittingError::EvaluationFailed(e.to_string()))
                })
            };
            let (up, down) = (sigma(up)?, sigma(down)?);
            Ok(up
                .iter()
                .zip(&down)
                .rev()
                .map(|(u, d)| density * (u - d) / (2.0 * h))
                .collect())
        };
        let total = slope(Channel::Total)?;
        let mut column: Vec<f64> = total
            .iter()
            .zip(&beams.per_depth)
            .map(|(s, t)| -t * s)
            .collect();
        if let Some(capture) = &beams.capture {
            let captured = slope(Channel::Capture)?;
            for (j, c) in column.iter_mut().enumerate() {
                *c += capture.per_depth[j] * total[j] + capture.share[j] * captured[j];
            }
        }
        if let Some(scatter) = &beams.scatter {
            let scattered = slope(Channel::Elastic)?;
            let elastic = self.at(region, temperature, ELASTIC)?;
            let densities = &params[place.densities.clone()];
            for (j, c) in column.iter_mut().enumerate() {
                let mut through = scattered[j] * scatter.shares[isotope].share[j];
                for (m, share) in scatter.shares.iter().enumerate() {
                    let entry = share.per_entry[j] * total[j]
                        + share.per_exit[j]
                            .iter()
                            .map(|&(l, w)| w * total[l])
                            .sum::<f64>();
                    through += densities[m] * elastic.values[m][j] * entry;
                }
                *c += beams.normalized[j] * params[place.scatter] * through;
            }
        }
        let folded = predicted(&scale.rows, &column)?;
        let bins = folded.len();
        let mut arriving = vec![0.0; 2 * bins * self.layout.regions.len()];
        arriving[(2 * region + 1) * bins..(2 * region + 2) * bins].copy_from_slice(&folded);
        Ok(match &self.mixing {
            Some(shares) => mixed(shares, params[self.layout.spread], &arriving),
            None => arriving,
        })
    }

    fn time_background(&self, params: &[f64]) -> (Vec<f64>, [Vec<f64>; 3]) {
        let at = self.layout.time_background;
        time_counts_per_pixel(&self.bins, [params[at], params[at + 1], params[at + 2]])
    }

    fn exponential(&self, f: f64) -> Vec<f64> {
        self.shapes[1]
            .iter()
            .map(|root| (-f * root).exp())
            .collect()
    }

    fn room_per_pixel(&self) -> [Vec<f64>; 2] {
        self.frames
            .map(|frames| self.bins.iter().map(|(_, width)| frames * width).collect())
    }

    fn sample_time_scale(&self, params: &[f64], region: usize) -> f64 {
        let place = &self.layout.regions[region];
        self.charge_ratio * params[self.layout.normalization] * params[place.gamma] * place.pixels
    }
}

impl RegionsModel {
    fn arriving(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let scale = self.scale(params)?;
        let (background, _) = self.time_background(params);
        let [open_room, sample_room] = self.room_per_pixel();
        let rate = params[self.layout.room_background];
        let mut counts = Vec::new();
        for region in 0..self.absorbers.len() {
            let Beams { open, sample, .. } = self.beams(params, &scale, region)?;
            let pixels = self.layout.regions[region].pixels;
            let sample_scale = self.sample_time_scale(params, region);
            counts.extend(
                predicted(&scale.rows, &open)?
                    .iter()
                    .zip(&background)
                    .zip(&open_room)
                    .map(|((c, g), f)| c + pixels * (g + rate * f)),
            );
            counts.extend(
                predicted(&scale.rows, &sample)?
                    .iter()
                    .zip(&background)
                    .zip(&sample_room)
                    .map(|((c, g), f)| c + sample_scale * g + pixels * rate * f),
            );
        }
        Ok(counts)
    }
}

/// The counts each region records, `(1 − h)·μ_r + h·Σ_q F_rq·μ_q` in every
/// bin of both runs, from the counts `μ` arriving in the regions' blocks in
/// turn.
fn mixed(shares: &[Vec<f64>], h: f64, arriving: &[f64]) -> Vec<f64> {
    let block = arriving.len() / shares.len();
    (0..arriving.len())
        .map(|i| {
            let (r, k) = (i / block, i % block);
            (1.0 - h) * arriving[i]
                + h * shares[r]
                    .iter()
                    .enumerate()
                    .map(|(q, f)| f * arriving[q * block + k])
                    .sum::<f64>()
        })
        .collect()
}

impl FitModel for RegionsModel {
    /// The counts each region records; under a spread, a bin whose arriving
    /// counts are negative or not a number records them instead.
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let arriving = self.arriving(params)?;
        let Some(shares) = &self.mixing else {
            return Ok(arriving);
        };
        let mut recorded = mixed(shares, params[self.layout.spread], &arriving);
        for (recorded, &arrived) in recorded.iter_mut().zip(&arriving) {
            if arrived.is_nan() || arrived < 0.0 {
                *recorded = arrived;
            }
        }
        Ok(recorded)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let layout = &*self.layout;
        let (t0_us, flight_path_m) = (params[layout.t0], params[layout.flight_path]);
        let scale = self.scale(params).ok()?;
        let regions = self.absorbers.len();
        let beams: Vec<Beams> = (0..regions)
            .map(|region| self.beams(params, &scale, region))
            .collect::<Result<_, _>>()
            .ok()?;
        let bins = y_current.len() / (2 * regions);
        let (open_rows, sample_rows) = (|r: usize| 2 * r * bins, |r: usize| (2 * r + 1) * bins);
        let counts = |values: &[f64]| predicted(&scale.rows, values).ok();
        let times = |beam: &[f64], slope: &[f64]| -> Vec<f64> {
            beam.iter().zip(slope).map(|(b, s)| b * s).collect()
        };
        let mut arrival: Option<Rows> = None;
        let mut pulse_slopes: Option<[Rows; 4]> = None;
        let ones = vec![1.0; self.energies.len()];
        let root_energy = &self.shapes[2];
        let pulse_columns: [(usize, &[f64]); 7] = [
            (0, root_energy),
            (0, &ones),
            (1, root_energy),
            (1, &ones),
            (2, &ones),
            (3, &ones),
            (2, root_energy),
        ];
        let (background, background_slopes) = self.time_background(params);
        let scaled = |factor: f64, values: &[f64]| -> Vec<f64> {
            values.iter().map(|v| factor * v).collect()
        };
        let mut jacobian = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        for (col, &index) in free_param_indices.iter().enumerate() {
            let blocks: Vec<(usize, Vec<f64>)> = match layout.roles[index] {
                Role::Beam {
                    region,
                    coefficient,
                } => {
                    let slope = weights_of(&scale.basis[region], coefficient);
                    let Beams { open, sample, .. } = &beams[region];
                    vec![
                        (open_rows(region), counts(&times(open, &slope))?),
                        (sample_rows(region), counts(&times(sample, &slope))?),
                    ]
                }
                Role::Density { region, isotope } => {
                    let temperature = params[layout.regions[region].temperature?];
                    let sigma = self.at(region, temperature, TOTAL).ok()?;
                    let total = &sigma.values[isotope];
                    let mut column: Vec<f64> = beams[region]
                        .per_depth
                        .iter()
                        .zip(total)
                        .map(|(d, s)| -d * s)
                        .collect();
                    if let Some(capture) = &beams[region].capture {
                        let captures = self.at(region, temperature, CAPTURE).ok()?;
                        for (j, c) in column.iter_mut().enumerate() {
                            *c += capture.per_depth[j] * total[j]
                                + capture.share[j] * captures.values[isotope][j];
                        }
                    }
                    if let Some(scatter) = &beams[region].scatter {
                        let place = &layout.regions[region];
                        let elastic = self.at(region, temperature, ELASTIC).ok()?;
                        let densities = &params[place.densities.clone()];
                        for (j, c) in column.iter_mut().enumerate() {
                            let mut slope =
                                elastic.values[isotope][j] * scatter.shares[isotope].share[j];
                            for (m, share) in scatter.shares.iter().enumerate() {
                                let through = share.per_entry[j] * total[j]
                                    + share.per_exit[j]
                                        .iter()
                                        .map(|&(l, w)| w * total[l])
                                        .sum::<f64>();
                                slope += densities[m] * elastic.values[m][j] * through;
                            }
                            *c += beams[region].normalized[j] * params[place.scatter] * slope;
                        }
                    }
                    vec![(sample_rows(region), counts(&column)?)]
                }
                Role::Temperature { region } => {
                    let place = &layout.regions[region];
                    let sigma = self.at(region, params[index], TOTAL).ok()?;
                    let densities = &params[place.densities.clone()];
                    let along = |slopes: &[Vec<f64>], j: usize| -> f64 {
                        densities
                            .iter()
                            .zip(slopes)
                            .map(|(n, slope)| n * slope[j])
                            .sum()
                    };
                    let mut column: Vec<f64> = (0..self.energies.len())
                        .map(|j| -beams[region].per_depth[j] * along(&sigma.slopes, j))
                        .collect();
                    if let Some(capture) = &beams[region].capture {
                        let captures = self.at(region, params[index], CAPTURE).ok()?;
                        for (j, c) in column.iter_mut().enumerate() {
                            *c += capture.per_depth[j] * along(&sigma.slopes, j)
                                + capture.share[j] * along(&captures.slopes, j);
                        }
                    }
                    if let Some(scatter) = &beams[region].scatter {
                        let elastic = self.at(region, params[index], ELASTIC).ok()?;
                        let depth_slopes: Vec<f64> = (0..self.energies.len())
                            .map(|j| along(&sigma.slopes, j))
                            .collect();
                        for (j, c) in column.iter_mut().enumerate() {
                            let mut slope = 0.0;
                            for (m, share) in scatter.shares.iter().enumerate() {
                                let through = share.per_entry[j] * depth_slopes[j]
                                    + share.per_exit[j]
                                        .iter()
                                        .map(|&(l, w)| w * depth_slopes[l])
                                        .sum::<f64>();
                                slope += densities[m]
                                    * (elastic.slopes[m][j] * share.share[j]
                                        + elastic.values[m][j] * through);
                            }
                            *c += beams[region].normalized[j] * params[place.scatter] * slope;
                        }
                    }
                    vec![(sample_rows(region), counts(&column)?)]
                }
                Role::ThicknessVariance { region } => {
                    let mut column = beams[region].per_variance.clone();
                    if let Some(capture) = &beams[region].capture {
                        for (c, v) in column.iter_mut().zip(&capture.per_variance) {
                            *c += v;
                        }
                    }
                    vec![(sample_rows(region), counts(&column)?)]
                }
                Role::CaptureGammas { region } => vec![(
                    sample_rows(region),
                    match &beams[region].capture {
                        Some(capture) => counts(&capture.per_efficiency)?,
                        None => vec![0.0; bins],
                    },
                )],
                Role::InScatter { region } => vec![(
                    sample_rows(region),
                    match &beams[region].scatter {
                        Some(scatter) => counts(&scatter.per_acceptance)?,
                        None => vec![0.0; bins],
                    },
                )],
                Role::Normalization => {
                    let slope = vec![1.0 / params[index]; self.energies.len()];
                    (0..regions)
                        .map(|r| {
                            let place = &layout.regions[r];
                            let gamma = self.charge_ratio * params[place.gamma] * place.pixels;
                            let beam = counts(&times(&beams[r].sample, &slope))?;
                            let column = beam
                                .iter()
                                .zip(&background)
                                .map(|(b, g)| b + gamma * g)
                                .collect();
                            Some((sample_rows(r), column))
                        })
                        .collect::<Option<_>>()?
                }
                Role::Background { region, term } => {
                    let at = layout.regions[region].background;
                    let (d, f) = (params[at + 3], params[at + 4]);
                    let shape: Vec<f64> = match term {
                        3 => self.exponential(f),
                        4 => self
                            .exponential(f)
                            .iter()
                            .zip(&self.shapes[1])
                            .map(|(x, root)| -d * root * x)
                            .collect(),
                        _ => self.shapes[term].clone(),
                    };
                    vec![(
                        sample_rows(region),
                        counts(&times(&beams[region].normalized, &shape))?,
                    )]
                }
                Role::GammaTransmission { region } => vec![(
                    sample_rows(region),
                    scaled(
                        self.charge_ratio
                            * params[layout.normalization]
                            * layout.regions[region].pixels,
                        &background,
                    ),
                )],
                Role::TimeBackground(term) => (0..regions)
                    .flat_map(|r| {
                        let slope = &background_slopes[term];
                        [
                            (open_rows(r), scaled(layout.regions[r].pixels, slope)),
                            (
                                sample_rows(r),
                                scaled(self.sample_time_scale(params, r), slope),
                            ),
                        ]
                    })
                    .collect(),
                Role::SpreadShare => vec![],
                Role::RoomBackground => {
                    let [open_room, sample_room] = self.room_per_pixel();
                    (0..regions)
                        .flat_map(|r| {
                            let pixels = layout.regions[r].pixels;
                            [
                                (open_rows(r), scaled(pixels, &open_room)),
                                (sample_rows(r), scaled(pixels, &sample_room)),
                            ]
                        })
                        .collect()
                }
                Role::Pulse(number) => {
                    if pulse_slopes.is_none() {
                        pulse_slopes = Some(
                            self.grid
                                .pulse_slopes_at(t0_us, flight_path_m, &scale.pulse)
                                .ok()?,
                        );
                    }
                    let (law, weight) = pulse_columns[number];
                    let rows = &pulse_slopes.as_ref()?[law];
                    (0..regions)
                        .flat_map(|r| {
                            [
                                (open_rows(r), &beams[r].open),
                                (sample_rows(r), &beams[r].sample),
                            ]
                        })
                        .map(|(row, beam)| Some((row, predicted(rows, &times(beam, weight)).ok()?)))
                        .collect::<Option<_>>()?
                }
                Role::T0 | Role::FlightPath => {
                    if arrival.is_none() {
                        arrival = Some(
                            self.grid
                                .arrival_slopes_at(t0_us, flight_path_m, &scale.pulse)
                                .ok()?,
                        );
                    }
                    let arrival = arrival.as_ref()?;
                    let (reach, per_metre): (Vec<f64>, Option<f64>) = match layout.roles[index] {
                        Role::FlightPath => (
                            self.grid
                                .flight_times_us()
                                .iter()
                                .map(|u| u / self.grid.flight_path_m())
                                .collect(),
                            Some(flight_path_m),
                        ),
                        _ => (ones.clone(), None),
                    };
                    let mut blocks = Vec::with_capacity(2 * regions);
                    for r in 0..regions {
                        let log_beam_slope = combined(
                            &scale.basis_slope[r],
                            &params[layout.regions[r].beam.clone()],
                        );
                        for (row, beam) in [
                            (open_rows(r), &beams[r].open),
                            (sample_rows(r), &beams[r].sample),
                        ] {
                            let reached = times(beam, &reach);
                            let beam_part = counts(&times(&reached, &log_beam_slope))?;
                            let pulse_part = predicted(arrival, &reached).ok()?;
                            let mut column: Vec<f64> = beam_part
                                .iter()
                                .zip(&pulse_part)
                                .map(|(b, p)| b + p)
                                .collect();
                            if let Some(flight_path_m) = per_metre {
                                for (c, x) in column.iter_mut().zip(counts(beam)?) {
                                    *c += x / flight_path_m;
                                }
                            }
                            blocks.push((row, column));
                        }
                    }
                    blocks
                }
            };
            for (row, column) in blocks {
                for (offset, value) in column.into_iter().enumerate() {
                    *jacobian.get_mut(row + offset, col) = value;
                }
            }
        }
        let Some(shares) = &self.mixing else {
            return Some(jacobian);
        };
        let h = params[layout.spread];
        let arriving = self.arriving(params).ok()?;
        let along_share: Vec<f64> = mixed(shares, 1.0, &arriving)
            .iter()
            .zip(&arriving)
            .map(|(m, a)| m - a)
            .collect();
        let mut recorded = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        for (col, &index) in free_param_indices.iter().enumerate() {
            let column = if layout.roles[index] == Role::SpreadShare {
                along_share.clone()
            } else {
                let slope: Vec<f64> = (0..y_current.len()).map(|i| jacobian.get(i, col)).collect();
                mixed(shares, h, &slope)
            };
            for (i, value) in column.into_iter().enumerate() {
                *recorded.get_mut(i, col) = value;
            }
        }
        Some(recorded)
    }
}

#[cfg(test)]
mod tests {
    use nereids_endf::resonance::test_support::synthetic_isotope;

    use faer::Side;
    use faer::linalg::solvers::DenseSolveCore;

    use super::*;
    use crate::open_beam::Pulse;
    use crate::open_beam::tests::{ALPHA, BETA, EDGES_US, FLIGHT_PATH_M, R, T0_US, grid};

    #[test]
    fn the_jacobian_is_the_slope_of_every_region_s_counts() {
        for channel_fwhm_us in [None, Some(0.35)] {
            jacobian_against_central_differences(&grid(channel_fwhm_us));
        }
        let bins = starts_and_widths_us(&[350.0, 351.0, 400.0, 470.0]);
        let (_, slopes) = time_counts_per_pixel(&bins, [1.0, 2.0, f64::MIN_POSITIVE]);
        assert!(slopes.iter().flatten().all(|s| s.is_finite()), "{slopes:?}");
        let (counts, _) = time_counts_per_pixel(&bins, [1.0, 2.0, f64::MAX]);
        for (count, (_, width)) in counts.iter().zip(&bins) {
            assert!((count / (3.0 * width) - 1.0).abs() <= 1e-12, "{counts:?}");
        }
    }

    #[test]
    fn under_a_spread_a_negative_arriving_count_is_recorded_as_it_arrives() {
        let grid = grid(None);
        let (_, u_hi) = grid.range_us();
        let beams = [
            BeamSpline::constant(347.0, u_hi, 1.0e5),
            BeamSpline::constant(347.0, u_hi, 1.0e4),
        ];
        let layout = Arc::new(Layout::new(
            roles(&[beams[0].coefficients().len(); 2], &[None, None]),
            &[400.0, 1.0],
            &[[false; 2]; 2],
        ));
        let params: Vec<f64> = layout
            .roles
            .iter()
            .enumerate()
            .map(|(i, role)| match *role {
                Role::Beam { region, .. } => [12.0, 9.0][region] + 0.3 * (i as f64).sin(),
                Role::Background { region, term } => {
                    [[0.0; 5], [-1.5, 0.0, 0.0, 0.0, 4.0]][region][term]
                }
                Role::GammaTransmission { .. } => 1.0,
                Role::Normalization => 0.93,
                Role::TimeBackground(term) => [50.0, 300.0, 40.0][term],
                Role::SpreadShare => 0.5,
                Role::T0 => T0_US,
                Role::FlightPath => FLIGHT_PATH_M,
                Role::Pulse(n) => [0.37, 0.06, 0.01, 0.24, 0.16, 0.3, 0.0][n],
                _ => 0.0,
            })
            .collect();
        let model = RegionsModel::new(
            &grid,
            &beams,
            &[Arc::new([]), Arc::new([])],
            (1.2, [3.0e4, 2.5e4]),
            T0_US,
            &layout,
        )
        .spread(Some(Arc::new(vec![vec![0.9, 0.1], vec![0.1, 0.9]])));
        let arriving = model.arriving(&params).expect("arriving");
        let recorded = model.evaluate(&params).expect("recorded");
        let negative: Vec<usize> = (0..arriving.len()).filter(|&k| arriving[k] < 0.0).collect();
        assert!(!negative.is_empty());
        for k in negative {
            assert_eq!(recorded[k], arriving[k], "{k}");
        }
    }

    #[test]
    fn a_resonance_slope_is_the_slope_of_the_sample_counts() {
        let grid = grid(None);
        let (_, u_hi) = grid.range_us();
        let beams = [
            BeamSpline::constant(347.0, u_hi, 1.0e4),
            BeamSpline::constant(347.0, u_hi, 2.0e4),
        ];
        let layout = Arc::new(Layout::new(
            roles(&[beams[0].coefficients().len(); 2], &[Some(1), None]),
            &[4.0, 9.0],
            &[[true; 2], [false; 2]],
        ));
        let params: Vec<f64> = layout
            .roles
            .iter()
            .enumerate()
            .map(|(i, role)| match *role {
                Role::Beam { .. } => 9.0 + 0.3 * (i as f64).sin(),
                Role::Density { .. } => 3.0e-4,
                Role::Temperature { .. } => 300.0,
                Role::ThicknessVariance { .. } => 0.02,
                Role::Normalization => 0.93,
                Role::Background { term, .. } => [0.05, 0.5, -0.01, 0.03, 4.0][term],
                Role::GammaTransmission { region } => [0.8, 1.0][region],
                Role::CaptureGammas { region } => [0.05, 0.0][region],
                Role::InScatter { region } => [0.3, 0.0][region],
                Role::TimeBackground(term) => [50.0, 300.0, 40.0][term],
                Role::RoomBackground => 2.0e-3,
                Role::SpreadShare => 0.15,
                Role::T0 => T0_US + 0.05,
                Role::FlightPath => FLIGHT_PATH_M + 0.003,
                Role::Pulse(n) => [0.37, 0.06, 0.01, 0.24, 0.16, 0.3, 0.0][n],
            })
            .collect();
        let data = synthetic_isotope(72, 180, 20.0, 0.01, 0.06);
        let model = |data: &ResonanceData| {
            let isotopes: [Arc<[ResonanceData]>; 2] = [Arc::new([data.clone()]), Arc::new([])];
            RegionsModel::new(
                &grid,
                &beams,
                &isotopes,
                (1.2, [3.0e4, 2.5e4]),
                T0_US,
                &layout,
            )
            .spread(Some(Arc::new(vec![vec![0.9, 0.1], vec![0.2, 0.8]])))
        };
        let h = 1e-4 * data.ranges[0].l_groups[0].resonances[0].gg;
        let moved = |by: f64| {
            let mut copy = data.clone();
            copy.ranges[0].l_groups[0].resonances[0].gg += by;
            copy
        };
        let (up, down) = (moved(h), moved(-h));
        let slope = model(&data)
            .resonance_slope(&params, 0, 0, [&up, &down], h)
            .expect("slope");
        let counts = |data| model(data).evaluate(&params).expect("counts");
        let differences: Vec<f64> = counts(&up)
            .iter()
            .zip(&counts(&down))
            .map(|(u, d)| (u - d) / (2.0 * h))
            .collect();
        let column = differences.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        for (k, (&analytic, &difference)) in slope.iter().zip(&differences).enumerate() {
            assert!(
                (analytic - difference).abs() <= 1e-6 * column,
                "{k}: {analytic} vs {difference}"
            );
        }
    }

    fn jacobian_against_central_differences(grid: &Arc<FlightTimeGrid>) {
        let (_, u_hi) = grid.range_us();
        let coarse = BeamSpline::constant(347.0, u_hi, 1.0e4);
        let beams = [coarse.refined().refined(), coarse.clone(), coarse.refined()];
        let isotopes: [Arc<[ResonanceData]>; 3] = [
            Arc::new([
                synthetic_isotope(72, 180, 20.0, 0.01, 0.06),
                synthetic_isotope(74, 182, 20.3, 0.01, 0.06),
            ]),
            Arc::new([]),
            Arc::new([synthetic_isotope(74, 184, 20.6, 0.01, 0.06)]),
        ];
        let sizes: Vec<usize> = beams.iter().map(|b| b.coefficients().len()).collect();
        let layout = Arc::new(Layout::new(
            roles(&sizes, &[Some(2), None, Some(1)]),
            &[4.0, 9.0, 2.0],
            &[[true, true], [false, false], [true, true]],
        ));
        let params: Vec<f64> = layout
            .roles
            .iter()
            .enumerate()
            .map(|(i, role)| match *role {
                Role::Beam { .. } => 9.0 + 0.3 * (i as f64).sin(),
                Role::Density { region, isotope } => {
                    [[3.0e-4, 5.0e-4], [0.0; 2], [4.0e-4, 0.0]][region][isotope]
                }
                Role::Temperature { region } => [300.0, 0.0, 250.0][region],
                Role::ThicknessVariance { region } => [0.02, 0.0, 0.05][region],
                Role::Normalization => 0.93,
                Role::Background { region, term } => [
                    [0.05, 0.5, -0.01, 0.03, 4.0],
                    [0.02, 0.1, 0.003, -0.01, 2.0],
                    [0.04, -0.2, 0.005, 0.02, 6.0],
                ][region][term],
                Role::GammaTransmission { region } => [0.8, 1.0, 0.6][region],
                Role::CaptureGammas { region } => [0.05, 0.03, 0.02][region],
                Role::InScatter { region } => [0.3, 0.2, 0.4][region],
                Role::TimeBackground(term) => [50.0, 300.0, 40.0][term],
                Role::RoomBackground => 2.0e-3,
                Role::SpreadShare => 0.15,
                Role::T0 => T0_US + 0.05,
                Role::FlightPath => FLIGHT_PATH_M + 0.003,
                Role::Pulse(n) => [0.37, 0.06, 0.01, 0.24, 0.16, 0.3, 0.004][n],
            })
            .collect();
        let model = RegionsModel::new(
            grid,
            &beams,
            &isotopes,
            (1.2, [3.0e4, 2.5e4]),
            T0_US,
            &layout,
        )
        .spread(Some(Arc::new(vec![
            vec![0.8, 0.1, 0.05],
            vec![0.1, 0.7, 0.1],
            vec![0.05, 0.15, 0.8],
        ])));
        let live: Vec<f64> = (0..model.evaluate(&params).expect("counts").len())
            .map(|k| 0.9 + 0.1 * (0.3 * k as f64).sin())
            .collect();
        let model = Recorded { model, live: &live };
        let mut elsewhere = params.clone();
        for t in layout
            .regions
            .iter()
            .filter_map(|region| region.temperature)
        {
            elsewhere[t] -= 50.0;
        }
        elsewhere[layout.t0] = T0_US - 0.05;
        elsewhere[layout.flight_path] = FLIGHT_PATH_M - 0.003;
        elsewhere[layout.pulse + 4] = 0.2;
        model.evaluate(&elsewhere).expect("counts");
        let counts = model.evaluate(&params).expect("counts");
        let indices: Vec<usize> = (0..params.len()).collect();
        let jacobian = model
            .analytical_jacobian(&params, &indices, &counts)
            .expect("jacobian");
        let middle_us = grid.flight_times_us()[grid.flight_times_us().len() / 2];
        for index in indices {
            let h = if index == layout.flight_path {
                params[index] * 1e-4 * params[layout.t0] / middle_us
            } else {
                1e-4 * params[index].abs()
            };
            let shifted = |d: f64| {
                let mut p = params.clone();
                p[index] += d;
                model.evaluate(&p).expect("counts")
            };
            let (up, down) = (shifted(h), shifted(-h));
            let slopes: Vec<f64> = up
                .iter()
                .zip(&down)
                .map(|(u, d)| (u - d) / (2.0 * h))
                .collect();
            let column = slopes.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
            for (row, &slope) in slopes.iter().enumerate() {
                let analytic = jacobian.get(row, index);
                assert!(
                    (analytic - slope).abs() <= 1e-6 * column,
                    "{index} {row}: {analytic} vs {slope}"
                );
            }
        }
    }

    #[test]
    fn each_region_s_terms_rebuild_the_joint_covariance() {
        let grid = Arc::new(grid(None).halved().expect("grid").halved().expect("grid"));
        let (_, u_hi) = grid.range_us();
        let beam = BeamSpline::constant(347.0, u_hi, 1.0e4).refined();
        let beams = [beam.clone(), beam.clone(), beam];
        let materials = [
            Some(synthetic_isotope(72, 180, 20.0, 0.01, 0.06)),
            Some(synthetic_isotope(74, 182, 24.0, 0.01, 0.06)),
            None,
        ];
        let isotopes: Vec<Arc<[ResonanceData]>> = materials
            .iter()
            .map(|m| m.iter().cloned().collect())
            .collect();
        let shape: Vec<Option<usize>> = materials.iter().map(|m| m.as_ref().map(|_| 1)).collect();
        let sizes: Vec<usize> = beams.iter().map(|b| b.coefficients().len()).collect();
        let layout = Arc::new(Layout::new(
            roles(&sizes, &shape),
            &[1.0; 3],
            &[[false; 2]; 3],
        ));
        let truth: Vec<f64> = layout
            .roles
            .iter()
            .map(|role| match *role {
                Role::Beam {
                    region,
                    coefficient,
                } => beams[region].coefficients()[coefficient],
                Role::Density { region, .. } => [5.0e-4, 4.0e-4, 0.0][region],
                Role::Temperature { .. } => 300.0,
                Role::ThicknessVariance { .. } => 0.0,
                Role::Normalization => 0.93,
                Role::Background { region, term } => [
                    [0.05, 0.0, 0.0, 0.0, 0.0],
                    [0.05, 0.0, 0.0, 0.0, 0.0],
                    [0.0; 5],
                ][region][term],
                Role::GammaTransmission { .. } => 1.0,
                Role::CaptureGammas { .. } => 0.0,
                Role::InScatter { .. } => 0.0,
                Role::SpreadShare => 0.0,
                Role::TimeBackground(term) => [0.0, 0.0, 1.0][term],
                Role::RoomBackground => 0.0,
                Role::T0 => T0_US,
                Role::FlightPath => FLIGHT_PATH_M,
                Role::Pulse(n) => [0.35, 0.05, 0.0, 0.25, 0.15, 0.0, 0.0][n],
            })
            .collect();
        let model = RegionsModel::new(&grid, &beams, &isotopes, (1.2, [0.0; 2]), T0_US, &layout);
        let bins = model.evaluate(&truth).expect("counts").len() / 6;
        let live: Vec<f64> = (0..6 * bins)
            .map(|k| 0.9 + 0.1 * (0.3 * k as f64).sin())
            .collect();
        let counts: Vec<f64> = Recorded { model, live: &live }
            .evaluate(&truth)
            .expect("counts")
            .iter()
            .enumerate()
            .map(|(k, mu)| (mu + 3.0 * mu.sqrt() * (1.7 * k as f64).sin()).round())
            .collect();
        let measurement = Measurement {
            time_edges_us: EDGES_US.map(f64::from).collect(),
            charge_ratio: 1.2,
            normalization: Value::Fitted(1.0),
            time_background: NO_TIME_BACKGROUND,
            room_background: None,
            spread: None,
            regions: materials
                .into_iter()
                .enumerate()
                .map(|(r, material)| Region {
                    open_counts: counts[2 * r * bins..(2 * r + 1) * bins].to_vec(),
                    sample_counts: counts[(2 * r + 1) * bins..(2 * r + 2) * bins].to_vec(),
                    open_live: Some(live[2 * r * bins..(2 * r + 1) * bins].to_vec()),
                    sample_live: Some(live[(2 * r + 1) * bins..(2 * r + 2) * bins].to_vec()),
                    background: match material {
                        Some(_) => [
                            Value::Fitted(0.02),
                            Value::Known(0.0),
                            Value::Known(0.0),
                            Value::Known(0.0),
                            Value::Known(0.0),
                        ],
                        None => [Value::Known(0.0); 5],
                    },
                    pixels: 1.0,
                    gamma_transmission: Value::Known(1.0),
                    capture_gammas: Value::Known(0.0),
                    in_scatter: Value::Known(0.0),
                    material: material.map(|data| Material {
                        covariances: Vec::new(),
                        library: EndfLibrary::EndfB8_0,
                        isotopes: vec![(data, Value::Fitted(3.0e-4))],
                        temperature_k: Value::Fitted(350.0),
                        thickness_variance: Value::Known(0.0),
                    }),
                })
                .collect(),
        };
        let calibration = Calibration {
            t0_us: Value::Fitted(T0_US),
            flight_path_m: Value::Fitted(FLIGHT_PATH_M),
            pulse: Pulse {
                alpha: [Value::Known(0.35), Value::Known(0.05)],
                beta: [Value::Known(0.0), Value::Known(0.25)],
                r: Value::Known(0.15),
                fwhm_squared_us2: Value::Known(0.0),
                r1: Value::Known(0.0),
                energy_span_ev: (1.0, 200.0),
                n_tau: 256,
                line_span_ev: None,
                prior: None,
            },
        };
        let open = starting_bypass(&measurement).expect("numbers");
        let (fit, answer) = fit_counts_answer(
            &measurement,
            &calibration,
            BeamStarts::OpenFitsAt(open, None),
            None,
        )
        .expect("fit");
        assert!(fit.converged);
        let covariance = fit.covariance.as_ref().expect("covariance");
        let fitted: Vec<Role> = quantities(&measurement, &calibration)
            .filter(|(_, value)| !matches!(value, Value::Known(_)))
            .map(|(role, _)| role)
            .collect();
        let at = |role: Role| fitted.iter().position(|&r| r == role).expect("fitted");
        let close = |actual: f64, a: Role, b: Role| {
            let (i, j) = (at(a), at(b));
            let expected = covariance.get(i, j);
            let scale = (covariance.get(i, i) * covariance.get(j, j)).sqrt();
            assert!(
                (actual - expected).abs() <= 1e-8 * scale,
                "{a:?} {b:?}: {actual} vs {expected}"
            );
        };
        let shared = [Role::Normalization, Role::T0, Role::FlightPath];
        let terms: Vec<RegionTerms> = (0..3)
            .map(|r| answer.region_terms(r, &shared).expect("terms"))
            .collect();
        let k = shared.len();
        let joint = Mat::from_fn(k, k, |i, j| {
            terms.iter().map(|t| t.information.get(i, j)).sum::<f64>()
        })
        .llt(Side::Lower)
        .expect("positive definite")
        .inverse();
        for i in 0..k {
            for j in 0..k {
                close(joint[(i, j)], shared[i], shared[j]);
            }
        }
        let carried = |t: &RegionTerms, a: usize, j: usize| {
            (0..k)
                .map(|i| t.sensitivity.get(a, i) * joint[(i, j)])
                .sum::<f64>()
        };
        for (p, tp) in terms.iter().enumerate() {
            for (a, &ra) in tp.quantities.iter().enumerate() {
                for (j, &rs) in shared.iter().enumerate() {
                    close(carried(tp, a, j), ra, rs);
                }
                for (q, tq) in terms.iter().enumerate() {
                    for (b, &rb) in tq.quantities.iter().enumerate() {
                        let common: f64 = (0..k)
                            .map(|j| carried(tp, a, j) * tq.sensitivity.get(b, j))
                            .sum();
                        let own = if p == q { tp.covariance.get(a, b) } else { 0.0 };
                        close(own + common, ra, rb);
                    }
                }
            }
        }
        let leverage: f64 = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| terms[0].counted_information.get(i, j) * joint[(i, j)])
            .sum();
        let value = |role: Role| {
            answer.params[answer
                .layout
                .roles
                .iter()
                .position(|&r| r == role)
                .expect("a shared role")]
        };
        let alone = Measurement {
            normalization: Value::Known(value(Role::Normalization)),
            regions: vec![measurement.regions[0].clone()],
            ..measurement.clone()
        };
        let held = Calibration {
            t0_us: Value::Known(value(Role::T0)),
            flight_path_m: Value::Known(value(Role::FlightPath)),
            ..calibration.clone()
        };
        let joined = &fit.regions[0];
        let starts = Starts {
            excluded: vec![joined.excluded_bins.clone()],
            beams: vec![BeamStart {
                beam: joined.beam.clone(),
                overdispersion: joined.overdispersion[0],
                at_limit: joined.beam_at_limit,
            }],
            origin_us: answer.beam_origin_us,
            bound: BOUND,
            shared_leverage: Some(vec![leverage]),
        };
        let (region, _) =
            fit_counts_answer(&alone, &held, BeamStarts::Given(&starts), None).expect("fit");
        let [Some(phi), Some(expected)] =
            [&region.regions[0], joined].map(|fit| fit.overdispersion[1])
        else {
            panic!("a sample overdispersion");
        };
        assert!((phi / expected - 1.0).abs() <= 1e-4, "{phi} vs {expected}");
        let gradient: Vec<f64> = (0..k)
            .map(|i| terms.iter().map(|t| t.gradient[i]).sum())
            .collect();
        let decrement: f64 = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * gradient[i] * joint[(i, j)] * gradient[j])
            .sum();
        assert!(decrement <= 1e-6, "{decrement}");
        let mut nudged = answer;
        let roles = nudged.layout.roles.clone();
        for (i, &role) in roles.iter().enumerate() {
            if matches!(
                role,
                Role::Density { region: 0, .. } | Role::Temperature { region: 0 }
            ) {
                nudged.params[i] += covariance.get(at(role), at(role)).sqrt();
            }
        }
        let nudged_terms: Vec<RegionTerms> = (0..3)
            .map(|r| nudged.region_terms(r, &shared).expect("terms"))
            .collect();
        let gradient: Vec<f64> = (0..k)
            .map(|i| nudged_terms.iter().map(|t| t.gradient[i]).sum())
            .collect();
        let decrement: f64 = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * gradient[i] * joint[(i, j)] * gradient[j])
            .sum();
        assert!(decrement <= 1e-2, "{decrement}");

        let drifting = Measurement {
            time_background: [Value::Fitted(50.0), Value::Known(0.0), Value::Known(1.0)],
            room_background: None,
            spread: None,
            ..measurement
        };
        for (most_passes, settled) in [(1, false), (2, true)] {
            let fit = fit_settled(&drifting, &calibration, most_passes).expect("fit");
            assert_eq!(fit.converged, settled);
            assert_eq!(fit.covariance.is_some(), settled);
            assert_eq!(fit.unbounded.is_some(), settled);
            assert_eq!(fit.measured_pulls.is_some(), settled);
        }
    }

    #[test]
    fn a_map_under_a_pulse_calibration_is_the_joint_fit() {
        let grid = Arc::new(grid(None).halved().expect("grid").halved().expect("grid"));
        let (_, u_hi) = grid.range_us();
        let beam = BeamSpline::constant(347.0, u_hi, 1.0e4).refined();
        let beams = [beam.clone(), beam];
        let hafnium = synthetic_isotope(72, 180, 20.0, 0.01, 0.06);
        let isotopes: Vec<Arc<[ResonanceData]>> = vec![Arc::new([hafnium.clone()]), Arc::new([])];
        let sizes: Vec<usize> = beams.iter().map(|b| b.coefficients().len()).collect();
        let layout = Arc::new(Layout::new(
            roles(&sizes, &[Some(1), None]),
            &[1.0; 2],
            &[[false; 2]; 2],
        ));
        let truth: Vec<f64> = layout
            .roles
            .iter()
            .map(|role| match *role {
                Role::Beam {
                    region,
                    coefficient,
                } => beams[region].coefficients()[coefficient],
                Role::Density { .. } => 5.0e-4,
                Role::Temperature { .. } => 300.0,
                Role::ThicknessVariance { .. } => 0.0,
                Role::Normalization => 0.93,
                Role::Background { region, term } => {
                    [[0.05, 0.0, 0.0, 0.0, 0.0], [0.0; 5]][region][term]
                }
                Role::GammaTransmission { .. } => 1.0,
                Role::CaptureGammas { .. } => 0.0,
                Role::InScatter { .. } => 0.0,
                Role::SpreadShare => 0.0,
                Role::TimeBackground(term) => [0.0, 0.0, 1.0][term],
                Role::RoomBackground => 0.0,
                Role::T0 => T0_US,
                Role::FlightPath => FLIGHT_PATH_M,
                Role::Pulse(n) => [0.35, 0.05, 0.0, 0.25, 0.15, 0.0, 0.0][n],
            })
            .collect();
        let counts: Vec<f64> =
            RegionsModel::new(&grid, &beams, &isotopes, (1.2, [0.0; 2]), T0_US, &layout)
                .evaluate(&truth)
                .expect("counts")
                .iter()
                .map(|c| c.round())
                .collect();
        let bins = counts.len() / 4;
        let run =
            |r: usize, run: usize| counts[(2 * r + run) * bins..(2 * r + run + 1) * bins].to_vec();
        let cube = |which: usize| {
            ndarray::Array3::from_shape_fn((bins, 2, 1), |(k, y, _)| run(y, which)[k])
        };
        let (open, sample) = (cube(0), cube(1));
        let sample_pixel = ndarray::Array2::from_shape_fn((2, 1), |(y, _)| y == 0);
        let empty_pixel = ndarray::Array2::from_shape_fn((2, 1), |(y, _)| y == 1);
        let none = ndarray::Array2::from_elem((2, 1), false);
        let material = Material {
            covariances: Vec::new(),
            library: EndfLibrary::EndfB8_0,
            isotopes: vec![(hafnium, Value::Fitted(3.0e-4))],
            temperature_k: Value::Fitted(350.0),
            thickness_variance: Value::Known(0.0),
        };
        let background = [
            Value::Fitted(0.02),
            Value::Known(0.0),
            Value::Known(0.0),
            Value::Known(0.0),
            Value::Known(0.0),
        ];
        let map = crate::counts_map::MapMeasurement {
            time_edges_us: EDGES_US.map(f64::from).collect(),
            charge_ratio: 1.2,
            normalization: Value::Fitted(1.0),
            time_background: NO_TIME_BACKGROUND,
            room_background: None,
            open_counts: open.view(),
            sample_counts: sample.view(),
            open_live: None,
            sample_live: None,
            excluded: none.view(),
            sample: sample_pixel.view(),
            empty: empty_pixel.view(),
            binning: 1,
            material: material.clone(),
            background,
            gamma_transmission: Value::Known(1.0),
            capture_gammas: Value::Known(0.0),
            in_scatter: Value::Known(0.0),
            empty_background: [Value::Known(0.0); 5],
        };
        let calibration = Calibration {
            t0_us: Value::Measured {
                value: T0_US,
                sd: 0.01,
            },
            flight_path_m: Value::Fitted(FLIGHT_PATH_M),
            pulse: Pulse {
                alpha: [Value::Fitted(0.35), Value::Fitted(0.05)],
                beta: [Value::Known(0.0), Value::Known(0.25)],
                r: Value::Known(0.15),
                fwhm_squared_us2: Value::Known(0.0),
                r1: Value::Known(0.0),
                energy_span_ev: (1.0, 200.0),
                n_tau: 256,
                line_span_ev: None,
                prior: Some(crate::pulse_calibration::PulsePrior {
                    numbers: vec![0, 1],
                    mean: vec![0.36, 0.05],
                    covariance: FlatMatrix {
                        data: vec![1.0e-4, -2.0e-5, -2.0e-5, 1.0e-5],
                        nrows: 2,
                        ncols: 2,
                    },
                }),
            },
        };
        let result = crate::counts_map::fit_map(&map, &calibration).expect("map");
        assert!(result.converged);
        let region = |r: usize, material: Option<Material>, background| Region {
            open_counts: run(r, 0),
            sample_counts: run(r, 1),
            open_live: None,
            sample_live: None,
            background,
            pixels: 1.0,
            gamma_transmission: Value::Known(1.0),
            capture_gammas: Value::Known(0.0),
            in_scatter: Value::Known(0.0),
            material,
        };
        let measurement = Measurement {
            time_edges_us: map.time_edges_us.clone(),
            charge_ratio: 1.2,
            normalization: map.normalization,
            time_background: NO_TIME_BACKGROUND,
            room_background: None,
            spread: None,
            regions: vec![
                region(0, Some(material), background),
                region(1, None, [Value::Known(0.0); 5]),
            ],
        };
        let joint = fit_counts(&measurement, &calibration).expect("joint fit");
        assert!(joint.converged);
        let mut carrying = measurement.clone();
        carrying.regions[0]
            .material
            .as_mut()
            .expect("a material")
            .covariances = vec![(
            0,
            ResonanceCovariance {
                resonances: vec![[0, 0, 0]],
                covariance: [1e-6, 0.0, 0.0, 0.0, 1e-10, 0.0, 0.0, 0.0, 1e-8].into(),
            },
        )];
        assert!(matches!(
            fit_counts(&carrying, &calibration),
            Err(PipelineError::InvalidParameter(_))
        ));
        let covariance = joint.covariance.as_ref().expect("covariance");
        let fitted: Vec<Role> = quantities(&measurement, &calibration)
            .filter(|(_, value)| !matches!(value, Value::Known(_)))
            .map(|(role, _)| role)
            .collect();
        let at = |role: Role| fitted.iter().position(|&r| r == role).expect("fitted");
        let close = |actual: f64, a: Role, b: Role| {
            let (i, j) = (at(a), at(b));
            let scale = (covariance.get(i, i) * covariance.get(j, j)).sqrt();
            assert!(
                (actual - covariance.get(i, j)).abs() <= 1e-3 * scale,
                "{a:?} {b:?}: {actual} vs {}",
                covariance.get(i, j)
            );
        };
        let shared = [
            Role::Normalization,
            Role::T0,
            Role::FlightPath,
            Role::Pulse(0),
            Role::Pulse(1),
        ];
        let map_shared = result.shared_covariance.as_ref().expect("covariance");
        for (s, &a) in shared.iter().enumerate() {
            for (t, &b) in shared.iter().enumerate() {
                close(map_shared.get(s, t), a, b);
            }
        }
        let own = [
            Role::Density {
                region: 0,
                isotope: 0,
            },
            Role::Temperature { region: 0 },
            Role::Background { region: 0, term: 0 },
        ];
        let patch = result.covariance[[0, 0]].as_ref().expect("covariance");
        for (s, &a) in own.iter().enumerate() {
            for (t, &b) in own.iter().enumerate() {
                close(patch.get(s, t), a, b);
            }
        }
        let gaps = [
            (result.alpha[0] - joint.alpha[0])
                / covariance
                    .get(at(Role::Pulse(0)), at(Role::Pulse(0)))
                    .sqrt(),
            (result.t0_us - joint.t0_us) / covariance.get(at(Role::T0), at(Role::T0)).sqrt(),
            (result.densities[0][[0, 0]] - joint.regions[0].densities[0])
                / covariance.get(at(own[0]), at(own[0])).sqrt(),
        ];
        assert!(gaps.iter().all(|g| g.abs() <= 0.01), "{gaps:?}");
        let [Some(map_test), Some(joint_test)] =
            [result.pulse_consistency, joint.pulse_consistency]
        else {
            panic!("a pulse consistency");
        };
        assert_eq!(map_test.dof, joint_test.dof);
        assert!(
            (map_test.q - joint_test.q).abs() <= 1e-3 * joint_test.q,
            "{map_test:?} vs {joint_test:?}"
        );
    }

    #[test]
    fn the_background_lags_sammy_s_by_the_pulse_s_mean_delay() {
        let grid = grid(None);
        let (_, u_hi) = grid.range_us();
        let first_edge_us = f64::from(*EDGES_US.start());
        let beam = BeamSpline::constant(first_edge_us - T0_US, u_hi, 1.0e4);
        let isotopes: Arc<[ResonanceData]> = Arc::new([
            synthetic_isotope(72, 180, 20.0, 0.01, 0.06),
            synthetic_isotope(74, 182, 20.3, 0.01, 0.06),
        ]);
        let (charge_ratio, normalization) = (1.2, 0.93);
        let layout = Arc::new(Layout::new(
            roles(&[beam.coefficients().len()], &[Some(2)]),
            &[1.0],
            &[[false; 2]],
        ));
        let model = RegionsModel::new(
            &grid,
            std::slice::from_ref(&beam),
            &[isotopes],
            (charge_ratio, [0.0; 2]),
            T0_US,
            &layout,
        );
        let counts = |background: [f64; 5]| {
            let params: Vec<f64> = beam
                .coefficients()
                .iter()
                .copied()
                .chain([3.0e-4, 5.0e-4, 300.0, 0.0, normalization])
                .chain(background)
                .chain([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .chain([T0_US, FLIGHT_PATH_M])
                .chain([0.35, 0.05, 0.0, 0.25, 0.15, 0.0, 0.0])
                .collect();
            model.evaluate(&params).expect("counts")
        };
        let [b0, b1, b2] = [0.05, 0.5, -0.01];
        let (with, without) = (counts([b0, b1, b2, 0.0, 0.0]), counts([0.0; 5]));
        let bins = with.len() / 2;
        let clock = TOF_FACTOR * FLIGHT_PATH_M;
        for k in 0..bins {
            let u = first_edge_us + 0.5 + k as f64 - T0_US;
            let root_e = clock / u;
            let b = b0 + b1 / root_e + b2 * root_e;
            let slope = b1 / clock - b2 * clock / (u * u);
            let curvature = 2.0 * b2 * clock / u.powi(3);
            let energy = root_e * root_e;
            let (alpha, beta, r) = (ALPHA.eval(energy), BETA.eval(energy), R.eval(energy));
            let mean = 3.0 / alpha + r / beta;
            let square = mean * mean + 3.0 / (alpha * alpha) + r * (2.0 - r) / (beta * beta);
            let lagged = (with[bins + k] - without[bins + k]) / (charge_ratio * normalization);
            let residual = lagged / with[k] - b + slope * mean;
            let tolerance = 0.5 * slope.abs() + 0.5 * curvature.abs() * square;
            assert!(
                residual.abs() <= tolerance,
                "{k}: {residual} vs {tolerance}"
            );
        }
    }

    #[test]
    fn an_isotope_known_to_be_absent_leaves_no_window() {
        let edges: Vec<f64> = (380..=530).map(f64::from).collect();
        let numbers = [3.0, 25.0, 0.5, 1.0, 0.08, 0.0, 0.2, 0.0, 0.0];
        let bins = |density| {
            let material = Material {
                covariances: Vec::new(),
                isotopes: vec![(synthetic_isotope(73, 181, 13.9, 0.01, 0.06), density)],
                temperature_k: Value::Known(300.0),
                library: EndfLibrary::EndfB8_1,
                thickness_variance: Value::Known(0.0),
            };
            excluded_bins(&edges, Some(&material), &numbers)
        };
        assert!(bins(Value::Known(0.0)).is_empty());
        assert!(!bins(Value::Known(1e-6)).is_empty());
    }
}
