//! Energy grid representation with TOF conversion.

use crate::constants::TOF_TO_ENERGY_FACTOR;

/// An energy grid in eV, stored as a sorted `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct EnergyGrid {
    /// Energy values in eV, sorted in ascending order.
    pub values: Vec<f64>,
}

impl EnergyGrid {
    /// Create an energy grid from a vector of energy values in eV.
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Create an energy grid from time-of-flight values.
    ///
    /// Converts TOF (in microseconds) to energy (in eV) using:
    /// `E = TOF_TO_ENERGY_FACTOR / (flight_path_m^2 * tof_us^2)`
    ///
    /// The resulting energies are sorted in ascending order (TOF is inverted
    /// relative to energy).
    pub fn from_tof(tof_us: &[f64], flight_path_m: f64, tof_offset_us: f64) -> Self {
        let l_sq = flight_path_m * flight_path_m;
        let mut values: Vec<f64> = tof_us
            .iter()
            .map(|&t| {
                let t_corrected = t - tof_offset_us;
                TOF_TO_ENERGY_FACTOR / (l_sq * t_corrected * t_corrected)
            })
            .collect();
        // Energy is inversely proportional to TOF^2, so reverse to get ascending order.
        values.reverse();
        Self { values }
    }

    /// Number of energy points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
