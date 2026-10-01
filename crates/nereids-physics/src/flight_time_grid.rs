//! The flight times whose neutrons can be counted in a set of detector time
//! bins, and the counts a beam predicts there, on a uniform flight-time grid.
//!
//! A neutron of flight time `u` over the flight path `L` has energy
//! `E = (TOF_FACTOR·L/u)²` and arrives at `t0 + u + delay`, the delay drawn
//! from the Ikeda–Carpenter pulse at `E`.  The grid spans every flight time
//! whose neutrons reach the bins with more than [`NEGLIGIBLE_ARRIVAL_PROBABILITY`](crate::ikeda_carpenter::NEGLIGIBLE_ARRIVAL_PROBABILITY) chance, and refuses
//! a window that neutrons from outside the pulse's synthesis grid can reach.
//!
//! Its points keep their energies at any other `t0'` and `L'`: point `j`,
//! of flight time `u_j` at the grid's own `t0` and `L`, then arrives at
//! `t0' + (L'/L)·u_j` and stands for `(L'/L)` times the step.

use std::fmt;
use std::sync::Arc;

use rayon::prelude::*;

use crate::ikeda_carpenter::{EnergyLaw, IkedaCarpenter};
use crate::resolution::{ResolutionParseError, TOF_FACTOR};

/// Largest number of grid points a window may need.
pub const MAX_POINTS: usize = 100_000;

/// Why a window has no flight-time grid.
#[derive(Debug)]
pub enum FlightTimeGridError {
    /// Fewer than two edges, or edges not finite and strictly ascending.
    InvalidTimeEdges,
    /// The timing offset is not finite.
    InvalidTimingOffset(f64),
    /// The flight path is not finite and positive.
    InvalidFlightPath(f64),
    /// Not supported: the pulse's `parameter` law lengthens the pulse as
    /// energy rises (α falling, β falling with a storage term, or R rising).
    /// The range relies on
    /// arrival times growing with flight time, which such a pulse does not
    /// guarantee.
    LengthensWithEnergy { parameter: &'static str },
    /// The pulse cannot be evaluated.
    Pulse(ResolutionParseError),
    /// Neutrons with energies outside the pulse's synthesis grid,
    /// `low_ev` to `high_ev`, can reach the window.
    OutsideCalibration { low_ev: f64, high_ev: f64 },
    /// A grid at `step_us` would need more than [`MAX_POINTS`] points.
    TooManyPoints { step_us: f64 },
    /// `found` values were given for a grid of `expected` points.
    ValuesLength { expected: usize, found: usize },
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
            Self::InvalidFlightPath(l) => {
                write!(f, "the flight path, {l} m, must be finite and positive")
            }
            Self::LengthensWithEnergy { parameter } => write!(
                f,
                "unsupported pulse: its {parameter} law lengthens the pulse as energy rises; \
                 the flight-time grid needs α and β that do not fall and R that does not \
                 rise with energy"
            ),
            Self::Pulse(e) => write!(f, "pulse: {e}"),
            Self::OutsideCalibration { low_ev, high_ev } => write!(
                f,
                "neutrons from outside the pulse's synthesis grid, {low_ev} to \
                 {high_ev} eV, can reach the window"
            ),
            Self::TooManyPoints { step_us } => write!(
                f,
                "a step of {step_us} µs needs more than {MAX_POINTS} grid points"
            ),
            Self::ValuesLength { expected, found } => write!(
                f,
                "{found} values were given for a grid of {expected} points"
            ),
        }
    }
}

impl std::error::Error for FlightTimeGridError {}

impl From<ResolutionParseError> for FlightTimeGridError {
    fn from(e: ResolutionParseError) -> Self {
        Self::Pulse(e)
    }
}

/// One value per bin for each grid point, kept where non-zero, with the
/// step each point stands for.
#[derive(Debug, Clone)]
pub struct Rows {
    step_us: f64,
    bins: usize,
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}

impl Rows {
    fn new(
        step_us: f64,
        bins: usize,
        rows: Vec<Result<Vec<f64>, ResolutionParseError>>,
    ) -> Result<Self, FlightTimeGridError> {
        let mut row_offsets = vec![0];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for row in rows {
            for (k, v) in row?.into_iter().enumerate().filter(|&(_, v)| v != 0.0) {
                columns.push(k);
                values.push(v);
            }
            row_offsets.push(values.len());
        }
        Ok(Self {
            step_us,
            bins,
            row_offsets,
            columns,
            values,
        })
    }

    /// `step · Σ_j v_j R_k(j)` for each bin `k`, with `values[j]` the value
    /// at grid point `j`.
    ///
    /// # Errors
    /// [`FlightTimeGridError::ValuesLength`] unless `values` has one entry
    /// per grid point.
    pub fn predict(&self, values: &[f64]) -> Result<Vec<f64>, FlightTimeGridError> {
        let points = self.row_offsets.len() - 1;
        if values.len() != points {
            return Err(FlightTimeGridError::ValuesLength {
                expected: points,
                found: values.len(),
            });
        }
        let mut counts = vec![0.0; self.bins];
        for (&v, row) in values.iter().zip(self.row_offsets.windows(2)) {
            for (&k, &p) in self.columns[row[0]..row[1]]
                .iter()
                .zip(&self.values[row[0]..row[1]])
            {
                counts[k] += self.step_us * v * p;
            }
        }
        Ok(counts)
    }
}

/// Uniform flight-time grid over a window, with the probability that each
/// grid point's neutrons are counted in each bin.
#[derive(Debug, Clone)]
pub struct FlightTimeGrid {
    time_edges_us: Vec<f64>,
    t0_us: f64,
    flight_path_m: f64,
    pulse: Arc<IkedaCarpenter>,
    range_us: (f64, f64),
    step_us: f64,
    flight_times_us: Vec<f64>,
    rows: Rows,
}

impl FlightTimeGrid {
    /// The grid for bins `time_edges_us` (µs) with timing offset `t0_us` and
    /// flight path `flight_path_m`; the pulse's own flight path is not used.
    /// The step, half the pulse's shorter rise at the two ends of the range,
    /// is a starting step: the counts are converged once [`Self::halved`] no
    /// longer changes them.
    ///
    /// # Errors
    /// See [`FlightTimeGridError`].
    pub fn new(
        time_edges_us: &[f64],
        t0_us: f64,
        flight_path_m: f64,
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
        if !(flight_path_m.is_finite() && flight_path_m > 0.0) {
            return Err(FlightTimeGridError::InvalidFlightPath(flight_path_m));
        }
        let references = pulse.ref_energies();
        let (e_min, e_max) = (references[0], references[references.len() - 1]);
        let params = pulse.params();
        let change = |law: &EnergyLaw| law.eval(e_max) - law.eval(e_min);
        for (parameter, lengthens) in [
            ("α", change(&params.alpha) < 0.0),
            (
                "β",
                change(&params.beta) < 0.0 && params.r.eval(e_min) > 0.0,
            ),
            ("R", change(&params.r) > 0.0),
        ] {
            if lengthens {
                return Err(FlightTimeGridError::LengthensWithEnergy { parameter });
            }
        }
        let clock = TOF_FACTOR * flight_path_m;
        let energy = |u: f64| (clock / u).powi(2);
        let first_edge = time_edges_us[0] - t0_us;
        let last_edge = time_edges_us[time_edges_us.len() - 1] - t0_us;
        let latest = |u: f64| {
            pulse
                .delays_us(energy(u))
                .map(|(_, last)| u + last - first_edge)
        };
        let earliest = |u: f64| {
            pulse
                .delays_us(energy(u))
                .map(|(first, _)| u + first - last_edge)
        };
        let (fastest, slowest) = (clock / e_max.sqrt(), clock / e_min.sqrt());
        let (Some(u_lo), Some(u_hi)) = (
            crossing(&latest, fastest, slowest)?,
            crossing(&earliest, fastest, slowest)?,
        ) else {
            return Err(FlightTimeGridError::OutsideCalibration {
                low_ev: e_min,
                high_ev: e_max,
            });
        };

        let rise = pulse
            .rise_us(energy(u_lo))?
            .min(pulse.rise_us(energy(u_hi))?);
        let intervals = ((u_hi - u_lo) / (0.5 * rise)).ceil();
        Self::build(
            time_edges_us,
            (t0_us, flight_path_m),
            pulse,
            (u_lo, u_hi),
            intervals as usize,
        )
    }

    fn build(
        time_edges_us: &[f64],
        (t0_us, flight_path_m): (f64, f64),
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
        let mut grid = Self {
            time_edges_us: time_edges_us.to_vec(),
            t0_us,
            flight_path_m,
            pulse: Arc::clone(pulse),
            range_us,
            step_us,
            flight_times_us,
            rows: Rows::new(step_us, 0, Vec::new())?,
        };
        grid.rows = grid.rows_at(t0_us, flight_path_m)?;
        Ok(grid)
    }

    /// The same window at half the step.
    ///
    /// # Errors
    /// As [`Self::new`], when the halved grid exceeds [`MAX_POINTS`].
    pub fn halved(&self) -> Result<Self, FlightTimeGridError> {
        Self::build(
            &self.time_edges_us,
            (self.t0_us, self.flight_path_m),
            &self.pulse,
            self.range_us,
            2 * (self.flight_times_us.len() - 1),
        )
    }

    /// The probability that each grid point's neutrons are counted in each
    /// bin when the timing offset is `t0_us` and the flight path
    /// `flight_path_m`, with the step each point then stands for.
    ///
    /// # Errors
    /// [`FlightTimeGridError::Pulse`] if the pulse cannot be evaluated there.
    pub fn rows_at(&self, t0_us: f64, flight_path_m: f64) -> Result<Rows, FlightTimeGridError> {
        self.rows_with(t0_us, flight_path_m, |energy, arrival| {
            self.pulse
                .bin_probabilities_at(energy, arrival, &self.time_edges_us)
        })
    }

    /// The derivative of [`Self::rows_at`]'s probabilities with respect to
    /// each point's arrival time, per µs, with the same step.
    ///
    /// # Errors
    /// As [`Self::rows_at`].
    pub fn arrival_slopes_at(
        &self,
        t0_us: f64,
        flight_path_m: f64,
    ) -> Result<Rows, FlightTimeGridError> {
        self.rows_with(t0_us, flight_path_m, |energy, arrival| {
            self.pulse
                .bin_arrival_slopes_at(energy, arrival, &self.time_edges_us)
        })
    }

    fn rows_with(
        &self,
        t0_us: f64,
        flight_path_m: f64,
        row: impl Fn(f64, f64) -> Result<Vec<f64>, ResolutionParseError> + Sync,
    ) -> Result<Rows, FlightTimeGridError> {
        let scale = flight_path_m / self.flight_path_m;
        let clock = TOF_FACTOR * self.flight_path_m;
        let rows: Vec<Result<Vec<f64>, ResolutionParseError>> = self
            .energies_ev()
            .par_iter()
            .map(|&energy| row(energy, t0_us + scale * (clock / energy.sqrt())))
            .collect();
        Rows::new(self.step_us * scale, self.time_edges_us.len() - 1, rows)
    }

    /// The probabilities at the grid's own timing offset and flight path.
    #[must_use]
    pub fn rows(&self) -> &Rows {
        &self.rows
    }

    /// The timing offset in µs the grid was built at.
    #[must_use]
    pub fn t0_us(&self) -> f64 {
        self.t0_us
    }

    /// The flight path in m the grid was built at.
    #[must_use]
    pub fn flight_path_m(&self) -> f64 {
        self.flight_path_m
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

    /// The energy, in eV, of each grid point, descending.
    #[must_use]
    pub fn energies_ev(&self) -> Vec<f64> {
        let clock = TOF_FACTOR * self.flight_path_m;
        self.flight_times_us
            .iter()
            .map(|u| (clock / u).powi(2))
            .collect()
    }

    /// `step · Σ_j v_j P_k(u_j)` for each bin `k`: the counts a beam of
    /// `values[j]` neutrons per µs at each grid point predicts, or, for any
    /// other per-point values, the same linear map.  It differs from the
    /// trapezoid rule by half a step of the end points' terms, where `P_k`
    /// is zero, or at most [`NEGLIGIBLE_ARRIVAL_PROBABILITY`](crate::ikeda_carpenter::NEGLIGIBLE_ARRIVAL_PROBABILITY) at the fast end
    /// for an unfolded pulse.  Features of the values narrower than the step
    /// are not resolved.
    ///
    /// # Errors
    /// [`FlightTimeGridError::ValuesLength`] unless `values` has one entry
    /// per grid point.
    pub fn predict(&self, values: &[f64]) -> Result<Vec<f64>, FlightTimeGridError> {
        self.rows.predict(values)
    }
}

fn crossing(
    signed: &dyn Fn(f64) -> Result<f64, ResolutionParseError>,
    fast: f64,
    slow: f64,
) -> Result<Option<f64>, ResolutionParseError> {
    if signed(fast)? >= 0.0 || signed(slow)? < 0.0 {
        return Ok(None);
    }
    let (mut low, mut high) = (fast, slow);
    while high - low > f64::EPSILON * high {
        let middle = 0.5 * (low + high);
        if signed(middle)? < 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(Some(high))
}
