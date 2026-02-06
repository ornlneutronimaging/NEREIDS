//! Energy grid representation with TOF conversion.

use crate::constants::TOF_TO_ENERGY_FACTOR;
use crate::error::PhysicsError;

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
    /// `E = TOF_TO_ENERGY_FACTOR * flight_path_m^2 / tof_us^2`
    ///
    /// The resulting energies are sorted in ascending order (TOF is inverted
    /// relative to energy).
    ///
    /// # Errors
    ///
    /// Returns `PhysicsError::InvalidParameter` if `flight_path_m` is not
    /// positive or if any corrected TOF value (`tof - tof_offset_us`) is not
    /// positive.
    pub fn from_tof(
        tof_us: &[f64],
        flight_path_m: f64,
        tof_offset_us: f64,
    ) -> Result<Self, PhysicsError> {
        if flight_path_m <= 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "flight_path_m must be positive, got {flight_path_m}"
            )));
        }
        let l_sq = flight_path_m * flight_path_m;
        let mut values: Vec<f64> = Vec::with_capacity(tof_us.len());
        for &t in tof_us {
            let t_corrected = t - tof_offset_us;
            if t_corrected <= 0.0 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "corrected TOF must be positive, got {t_corrected} (tof={t}, offset={tof_offset_us})"
                )));
            }
            values.push(TOF_TO_ENERGY_FACTOR * l_sq / (t_corrected * t_corrected));
        }
        // Sort ascending — TOF and energy are inversely related, but we don't
        // assume the input TOF array is in any particular order.
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Self { values })
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
