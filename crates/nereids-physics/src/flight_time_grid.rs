//! The flight times whose neutrons can be counted in a set of detector time
//! bins, and the counts a beam predicts there, on a uniform flight-time grid.
//!
//! A neutron of flight time `u` has energy `E = (TOF_FACTOR·L/u)²` and
//! arrives at `t0 + u + delay`, the delay drawn from the instrument pulse at
//! `E`.  The grid runs from `u_lo`, the shortest flight time whose latest
//! arrival reaches the first edge, to `u_hi`, the longest whose earliest
//! arrival reaches the last edge.

use std::fmt;

use crate::counts_response::{CountsResponseError, DetectorBinResponseMatrix};
use crate::resolution::{ResolutionFunction, ResolutionParseError, TOF_FACTOR};

/// Largest number of grid points a window may need.
pub const MAX_POINTS: usize = 100_000;

/// Why a window has no flight-time grid.
#[derive(Debug)]
pub enum FlightTimeGridError {
    /// Fewer than two edges, or edges not finite and strictly ascending.
    InvalidTimeEdges,
    /// The timing offset is not finite.
    InvalidTimingOffset(f64),
    /// The resolution has no pulse (Gaussian) or cannot be evaluated.
    Resolution(ResolutionParseError),
    /// The pulse is a single delay, with no rise to set the grid step.
    NoRise { energy_ev: f64 },
    /// Neutrons outside the pulse's calibrated energy range reach the window.
    OutsideCalibration { energy_ev: f64 },
    /// Neutrons of a flight time outside the range still reach the window.
    ReachedOutsideRange { flight_time_us: f64 },
    /// The grid would need more than [`MAX_POINTS`] points.
    TooManyPoints(usize),
    /// The bin probabilities could not be built.
    Response(CountsResponseError),
}

impl fmt::Display for FlightTimeGridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeEdges => {
                write!(
                    f,
                    "time edges must be at least two finite, strictly ascending values"
                )
            }
            Self::InvalidTimingOffset(t0) => {
                write!(f, "t0 = {t0} µs must be finite")
            }
            Self::Resolution(e) => write!(f, "resolution: {e}"),
            Self::NoRise { energy_ev } => {
                write!(
                    f,
                    "the pulse at {energy_ev} eV is a single delay with no rise"
                )
            }
            Self::OutsideCalibration { energy_ev } => write!(
                f,
                "neutrons of {energy_ev} eV, outside the pulse's calibrated energies, reach the window"
            ),
            Self::ReachedOutsideRange { flight_time_us } => write!(
                f,
                "neutrons of flight time {flight_time_us} µs, outside the range, reach the window"
            ),
            Self::TooManyPoints(n) => {
                write!(
                    f,
                    "the window needs {n} grid points, more than {MAX_POINTS}"
                )
            }
            Self::Response(e) => write!(f, "bin probabilities: {e}"),
        }
    }
}

impl std::error::Error for FlightTimeGridError {}

impl From<ResolutionParseError> for FlightTimeGridError {
    fn from(e: ResolutionParseError) -> Self {
        Self::Resolution(e)
    }
}

/// Uniform flight-time grid over a window, with the probability that each
/// grid point's neutrons are counted in each bin.
#[derive(Debug, Clone)]
pub struct FlightTimeGrid {
    time_edges_us: Vec<f64>,
    t0_us: f64,
    resolution: ResolutionFunction,
    range_us: (f64, f64),
    step_us: f64,
    flight_times_us: Vec<f64>,
    response: DetectorBinResponseMatrix,
}

impl FlightTimeGrid {
    /// The grid for bins `time_edges_us` (µs) with timing offset `t0_us`,
    /// the flight path taken from `resolution`.
    ///
    /// # Errors
    /// See [`FlightTimeGridError`].
    pub fn new(
        time_edges_us: &[f64],
        t0_us: f64,
        resolution: &ResolutionFunction,
    ) -> Result<Self, FlightTimeGridError> {
        if time_edges_us.len() < 2
            || !time_edges_us.iter().all(|t| t.is_finite())
            || time_edges_us.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(FlightTimeGridError::InvalidTimeEdges);
        }
        if !t0_us.is_finite() {
            return Err(FlightTimeGridError::InvalidTimingOffset(t0_us));
        }
        let Some(references) = resolution.reference_energies() else {
            return Err(FlightTimeGridError::Resolution(
                ResolutionParseError::InvalidFormat(
                    "a Gaussian resolution has no pulse to place neutrons in time bins".into(),
                ),
            ));
        };
        let clock = TOF_FACTOR * resolution.flight_path_m();
        let energy = |u: f64| (clock / u).powi(2);
        let delays = |u: f64| resolution.pulse_delays(energy(u));
        let (u_fast, u_slow) = (
            clock / references[references.len() - 1].sqrt(),
            clock / references[0].sqrt(),
        );
        let first_edge = time_edges_us[0] - t0_us;
        let last_edge = time_edges_us[time_edges_us.len() - 1] - t0_us;

        let latest = |u: f64| delays(u).map(|d| u + d.last_us - first_edge);
        let earliest = |u: f64| delays(u).map(|d| u + d.first_us - last_edge);
        let u_lo = crossing(&latest, u_fast, u_slow, energy)?;
        let u_hi = crossing(&earliest, u_fast, u_slow, energy)?;

        let mut rise = f64::INFINITY;
        let inside = |e: f64| e >= energy(u_hi) && e <= energy(u_lo);
        let probes = [energy(u_lo), energy(u_hi)];
        for e in probes
            .into_iter()
            .chain(references.iter().copied().filter(|&e| inside(e)))
        {
            let d = resolution.pulse_delays(e)?;
            if d.peak_us <= d.first_us {
                return Err(FlightTimeGridError::NoRise { energy_ev: e });
            }
            rise = rise.min(d.peak_us - d.first_us);
        }
        let narrowest_bin = time_edges_us
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, f64::min);
        let step = narrowest_bin.min(0.5 * rise);
        let intervals = ((u_hi - u_lo) / step).ceil();
        let spacing = step.max((u_slow - u_fast) / MAX_POINTS as f64);
        let faster = (0..)
            .map(|i| u_fast + spacing * f64::from(i))
            .take_while(|&u| u < u_lo);
        let slower = (0..)
            .map(|i| u_slow - spacing * f64::from(i))
            .take_while(|&u| u > u_hi);
        for u in faster.chain(slower) {
            let d = delays(u)?;
            if u + d.last_us > first_edge && u + d.first_us < last_edge {
                return Err(FlightTimeGridError::ReachedOutsideRange { flight_time_us: u });
            }
        }
        Self::build(
            time_edges_us,
            t0_us,
            resolution,
            (u_lo, u_hi),
            intervals as usize,
        )
    }

    fn build(
        time_edges_us: &[f64],
        t0_us: f64,
        resolution: &ResolutionFunction,
        range_us: (f64, f64),
        intervals: usize,
    ) -> Result<Self, FlightTimeGridError> {
        if intervals + 1 > MAX_POINTS {
            return Err(FlightTimeGridError::TooManyPoints(intervals + 1));
        }
        let (u_lo, u_hi) = range_us;
        let step_us = (u_hi - u_lo) / intervals as f64;
        let flight_times_us: Vec<f64> =
            (0..=intervals).map(|j| u_lo + step_us * j as f64).collect();
        let clock = TOF_FACTOR * resolution.flight_path_m();
        let energies: Vec<f64> = flight_times_us
            .iter()
            .map(|u| (clock / u).powi(2))
            .collect();
        let response = DetectorBinResponseMatrix::new(&energies, time_edges_us, t0_us, resolution)
            .map_err(FlightTimeGridError::Response)?;
        Ok(Self {
            time_edges_us: time_edges_us.to_vec(),
            t0_us,
            resolution: resolution.clone(),
            range_us,
            step_us,
            flight_times_us,
            response,
        })
    }

    /// The same window at half the step.
    ///
    /// # Errors
    /// As [`Self::new`], when the halved grid exceeds [`MAX_POINTS`].
    pub fn halved(&self) -> Result<Self, FlightTimeGridError> {
        Self::build(
            &self.time_edges_us,
            self.t0_us,
            &self.resolution,
            self.range_us,
            2 * (self.flight_times_us.len() - 1),
        )
    }

    /// `(u_lo, u_hi)`, the flight-time range in µs.
    #[must_use]
    pub fn range_us(&self) -> (f64, f64) {
        self.range_us
    }

    /// The grid step in µs.
    #[must_use]
    pub fn step_us(&self) -> f64 {
        self.step_us
    }

    /// The grid's flight times in µs, ascending.
    #[must_use]
    pub fn flight_times_us(&self) -> &[f64] {
        &self.flight_times_us
    }

    /// `step · Σ_j v_j P_k(u_j)` for each bin `k`: the counts a beam of
    /// `values[j]` neutrons per µs at each grid point predicts, or, for any
    /// other per-point values, the same linear map.  This is the trapezoid
    /// rule, since `P_k` falls to zero, or to [`NEGLIGIBLE_ARRIVAL_PROBABILITY`](crate::ikeda_carpenter::NEGLIGIBLE_ARRIVAL_PROBABILITY)
    /// for an unfolded Ikeda–Carpenter pulse, at both ends of the range.
    ///
    /// # Panics
    /// If `values` does not have one entry per grid point.
    #[must_use]
    pub fn predict(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(values.len(), self.flight_times_us.len());
        let mut counts = vec![0.0; self.response.n_detector_bins()];
        for (j, &v) in values.iter().enumerate() {
            for (k, p) in self.response.row_entries(j) {
                counts[k] += self.step_us * v * p;
            }
        }
        counts
    }
}

fn crossing(
    signed: &dyn Fn(f64) -> Result<f64, ResolutionParseError>,
    u_fast: f64,
    u_slow: f64,
    energy: impl Fn(f64) -> f64,
) -> Result<f64, FlightTimeGridError> {
    if signed(u_fast)? >= 0.0 {
        return Err(FlightTimeGridError::OutsideCalibration {
            energy_ev: energy(u_fast),
        });
    }
    if signed(u_slow)? < 0.0 {
        return Err(FlightTimeGridError::OutsideCalibration {
            energy_ev: energy(u_slow),
        });
    }
    let (mut low, mut high) = (u_fast, u_slow);
    while high - low > f64::EPSILON * high {
        let middle = 0.5 * (low + high);
        if signed(middle)? < 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(high)
}
