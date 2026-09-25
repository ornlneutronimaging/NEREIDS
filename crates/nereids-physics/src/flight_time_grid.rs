//! The flight times whose neutrons can be counted in a set of detector time
//! bins, and the counts a beam predicts there, on a uniform flight-time grid.
//!
//! A neutron of flight time `u` has energy `E = (TOF_FACTOR·L/u)²` and
//! arrives at `t0 + u + delay`, the delay drawn from the Ikeda–Carpenter
//! pulse at `E`.  The grid spans every flight time in the pulse's synthesis
//! grid whose neutrons can reach the bins; beyond the synthesis grid the
//! pulse is taken to be the pulse at its nearer end.

use std::fmt;
use std::sync::Arc;

use crate::counts_response::{CountsResponseError, DetectorBinResponseMatrix};
use crate::ikeda_carpenter::IkedaCarpenter;
use crate::resolution::{ResolutionFunction, ResolutionParseError, TOF_FACTOR};

/// Largest number of grid points a window may need.
pub const MAX_POINTS: usize = 100_000;

const SEARCH_WIDTH: f64 = 1e-9;

/// Why a window has no flight-time grid.
#[derive(Debug)]
pub enum FlightTimeGridError {
    /// Fewer than two edges, or edges not finite and strictly ascending.
    InvalidTimeEdges,
    /// The timing offset is not finite.
    InvalidTimingOffset(f64),
    /// The pulse cannot be evaluated.
    Pulse(ResolutionParseError),
    /// Neutrons with energies outside the pulse's synthesis grid,
    /// `low_ev` to `high_ev`, can reach the window, or none inside it can.
    OutsideCalibration { low_ev: f64, high_ev: f64 },
    /// A grid at `step_us` would need more than [`MAX_POINTS`] points.
    TooManyPoints { step_us: f64 },
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
            Self::InvalidTimingOffset(t0) => write!(f, "t0 = {t0} µs must be finite"),
            Self::Pulse(e) => write!(f, "pulse: {e}"),
            Self::OutsideCalibration { low_ev, high_ev } => write!(
                f,
                "the window is reached from outside the pulse's synthesis grid, \
                 {low_ev} to {high_ev} eV"
            ),
            Self::TooManyPoints { step_us } => write!(
                f,
                "a step of {step_us} µs needs more than {MAX_POINTS} grid points"
            ),
            Self::Response(e) => write!(f, "bin probabilities: {e}"),
        }
    }
}

impl std::error::Error for FlightTimeGridError {}

impl From<ResolutionParseError> for FlightTimeGridError {
    fn from(e: ResolutionParseError) -> Self {
        Self::Pulse(e)
    }
}

/// Uniform flight-time grid over a window, with the probability that each
/// grid point's neutrons are counted in each bin.
#[derive(Debug, Clone)]
pub struct FlightTimeGrid {
    time_edges_us: Vec<f64>,
    t0_us: f64,
    pulse: Arc<IkedaCarpenter>,
    range_us: (f64, f64),
    step_us: f64,
    flight_times_us: Vec<f64>,
    response: DetectorBinResponseMatrix,
}

impl FlightTimeGrid {
    /// The grid for bins `time_edges_us` (µs) with timing offset `t0_us`,
    /// the flight path taken from `pulse`.  The step is half the pulse's
    /// shorter rise at the two ends of the range.
    ///
    /// # Errors
    /// See [`FlightTimeGridError`].
    pub fn new(
        time_edges_us: &[f64],
        t0_us: f64,
        pulse: &Arc<IkedaCarpenter>,
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
        let references = pulse.ref_energies();
        let (e_min, e_max) = (references[0], references[references.len() - 1]);
        let outside = FlightTimeGridError::OutsideCalibration {
            low_ev: e_min,
            high_ev: e_max,
        };
        let clock = TOF_FACTOR * pulse.flight_path_m();
        let energy = |u: f64| (clock / u).powi(2);
        let flight_time = |e: f64| clock / e.sqrt();
        let first_edge = time_edges_us[0] - t0_us;
        let last_edge = time_edges_us[time_edges_us.len() - 1] - t0_us;

        let (_, fastest_last) = pulse.delay_bounds(e_max, e_max)?;
        let (slowest_first, _) = pulse.delay_bounds(e_min, e_min)?;
        if flight_time(e_max) + fastest_last > first_edge
            || flight_time(e_min) + slowest_first < last_edge
        {
            return Err(outside);
        }
        let reaches = |fast: f64, slow: f64| -> Result<bool, ResolutionParseError> {
            let (first, last) = pulse.delay_bounds(energy(slow), energy(fast))?;
            Ok(slow + last > first_edge && fast + first < last_edge)
        };
        let cells: Vec<(f64, f64)> = references
            .windows(2)
            .rev()
            .map(|w| (flight_time(w[1]), flight_time(w[0])))
            .collect();
        let u_lo = cells
            .iter()
            .find_map(|&(fast, slow)| outermost_reach(&reaches, fast, slow, true).transpose())
            .transpose()?;
        let u_hi = cells
            .iter()
            .rev()
            .find_map(|&(fast, slow)| outermost_reach(&reaches, fast, slow, false).transpose())
            .transpose()?;
        let (Some(u_lo), Some(u_hi)) = (u_lo, u_hi) else {
            return Err(outside);
        };

        let rise = pulse
            .rise_us(energy(u_lo))?
            .min(pulse.rise_us(energy(u_hi))?);
        let intervals = ((u_hi - u_lo) / (0.5 * rise)).ceil();
        Self::build(
            time_edges_us,
            t0_us,
            pulse,
            (u_lo, u_hi),
            intervals as usize,
        )
    }

    fn build(
        time_edges_us: &[f64],
        t0_us: f64,
        pulse: &Arc<IkedaCarpenter>,
        range_us: (f64, f64),
        intervals: usize,
    ) -> Result<Self, FlightTimeGridError> {
        let (u_lo, u_hi) = range_us;
        let step_us = (u_hi - u_lo) / intervals as f64;
        if intervals >= MAX_POINTS {
            return Err(FlightTimeGridError::TooManyPoints { step_us });
        }
        let flight_times_us: Vec<f64> =
            (0..=intervals).map(|j| u_lo + step_us * j as f64).collect();
        let clock = TOF_FACTOR * pulse.flight_path_m();
        let energies: Vec<f64> = flight_times_us
            .iter()
            .map(|u| (clock / u).powi(2))
            .collect();
        let response = DetectorBinResponseMatrix::new(
            &energies,
            time_edges_us,
            t0_us,
            &ResolutionFunction::IkedaCarpenter(Arc::clone(pulse)),
        )
        .map_err(FlightTimeGridError::Response)?;
        Ok(Self {
            time_edges_us: time_edges_us.to_vec(),
            t0_us,
            pulse: Arc::clone(pulse),
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
            &self.pulse,
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
    /// for an unfolded pulse, at both ends of the range.  Features of the
    /// values narrower than the step are not resolved.
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

fn outermost_reach(
    reaches: &dyn Fn(f64, f64) -> Result<bool, ResolutionParseError>,
    fast: f64,
    slow: f64,
    from_fast: bool,
) -> Result<Option<f64>, ResolutionParseError> {
    if !reaches(fast, slow)? {
        return Ok(None);
    }
    if slow - fast <= SEARCH_WIDTH * slow {
        return Ok(Some(if from_fast { fast } else { slow }));
    }
    let middle = 0.5 * (fast + slow);
    let (near, far) = if from_fast {
        ((fast, middle), (middle, slow))
    } else {
        ((middle, slow), (fast, middle))
    };
    match outermost_reach(reaches, near.0, near.1, from_fast)? {
        Some(u) => Ok(Some(u)),
        None => outermost_reach(reaches, far.0, far.1, from_fast),
    }
}
