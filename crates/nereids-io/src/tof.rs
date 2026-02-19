//! Time-of-flight to energy conversion for imaging data.
//!
//! VENUS beamline data is recorded in time-of-flight bins. This module converts
//! TOF bin edges/centers to neutron energy using E = ½·m_n·(L/t)².
//!
//! The core conversion functions live in `nereids_core::constants`; this module
//! provides the imaging-specific wrappers for working with TOF bin arrays and
//! spectra aligned to energy grids.

use ndarray::Array1;
use nereids_core::constants;

use crate::error::IoError;

/// VENUS beamline parameters for TOF-to-energy conversion.
#[derive(Debug, Clone, Copy)]
pub struct BeamlineParams {
    /// Total flight path in meters (source to detector).
    pub flight_path_m: f64,
    /// Delay time in microseconds (electronic/moderator delay).
    pub delay_us: f64,
}

impl Default for BeamlineParams {
    fn default() -> Self {
        Self {
            flight_path_m: 25.0, // VENUS nominal flight path
            delay_us: 0.0,
        }
    }
}

/// Convert an array of TOF bin edges (μs) to energy bin edges (eV).
///
/// Energies are returned in descending order (longer TOF = lower energy,
/// but the array is reversed so energies are ascending for physics routines).
///
/// # Arguments
/// * `tof_edges` — TOF bin edges in microseconds, ascending.
/// * `params` — Beamline parameters (flight path, delay).
///
/// # Returns
/// Energy bin edges in eV, ascending order.
pub fn tof_edges_to_energy(
    tof_edges: &[f64],
    params: &BeamlineParams,
) -> Result<Array1<f64>, IoError> {
    if tof_edges.len() < 2 {
        return Err(IoError::InvalidParameter(
            "Need at least 2 TOF bin edges".into(),
        ));
    }

    if params.flight_path_m <= 0.0 {
        return Err(IoError::InvalidParameter(
            "Flight path must be positive".into(),
        ));
    }

    let mut energies: Vec<f64> = tof_edges
        .iter()
        .map(|&tof| {
            let corrected_tof = tof - params.delay_us;
            if corrected_tof <= 0.0 {
                f64::INFINITY
            } else {
                constants::tof_to_energy(corrected_tof, params.flight_path_m)
            }
        })
        .collect();

    // TOF ascending → energy descending. Reverse to get ascending energies.
    energies.reverse();
    Ok(Array1::from_vec(energies))
}

/// Convert TOF bin edges to energy bin centers.
///
/// Returns the geometric mean of adjacent energy bin edges, which is
/// more physically appropriate for log-spaced energy grids.
///
/// # Arguments
/// * `tof_edges` — TOF bin edges in μs, ascending.
/// * `params` — Beamline parameters.
///
/// # Returns
/// Energy bin centers in eV, ascending order. Length = len(tof_edges) - 1.
pub fn tof_edges_to_energy_centers(
    tof_edges: &[f64],
    params: &BeamlineParams,
) -> Result<Array1<f64>, IoError> {
    let edges = tof_edges_to_energy(tof_edges, params)?;
    let n = edges.len() - 1;
    let centers: Vec<f64> = (0..n)
        .map(|i| (edges[i] * edges[i + 1]).sqrt())
        .collect();
    Ok(Array1::from_vec(centers))
}

/// Generate linearly-spaced TOF bin edges.
///
/// # Arguments
/// * `tof_min` — Minimum TOF in μs.
/// * `tof_max` — Maximum TOF in μs.
/// * `n_bins` — Number of bins (returns n_bins + 1 edges).
pub fn linspace_tof_edges(tof_min: f64, tof_max: f64, n_bins: usize) -> Vec<f64> {
    let dt = (tof_max - tof_min) / n_bins as f64;
    (0..=n_bins).map(|i| tof_min + i as f64 * dt).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tof_to_energy_roundtrip() {
        let params = BeamlineParams {
            flight_path_m: 25.0,
            delay_us: 0.0,
        };

        // Create TOF edges covering 1-100 eV energy range
        let e_low = 1.0;
        let e_high = 100.0;
        let tof_high = constants::energy_to_tof(e_low, params.flight_path_m);
        let tof_low = constants::energy_to_tof(e_high, params.flight_path_m);

        let tof_edges = linspace_tof_edges(tof_low, tof_high, 100);
        let energy_edges = tof_edges_to_energy(&tof_edges, &params).unwrap();

        // After reversing: first energy ≈ e_low, last energy ≈ e_high (ascending)
        assert!(
            (energy_edges[0] - e_low).abs() / e_low < 0.01,
            "First energy {} != expected {}",
            energy_edges[0],
            e_low,
        );

        let last = energy_edges[energy_edges.len() - 1];
        assert!(
            (last - e_high).abs() / e_high < 0.01,
            "Last energy {} != expected {}",
            last,
            e_high,
        );

        // Energies should be monotonically decreasing (since TOF edges
        // were linspaced from low TOF to high TOF → energies go high to low,
        // then reversed → ascending)
        // Actually after reverse they should be ascending
        for i in 1..energy_edges.len() {
            assert!(
                energy_edges[i] >= energy_edges[i - 1],
                "Energies not ascending at index {}: {} < {}",
                i,
                energy_edges[i],
                energy_edges[i - 1],
            );
        }
    }

    #[test]
    fn test_tof_to_energy_with_delay() {
        let params_no_delay = BeamlineParams {
            flight_path_m: 25.0,
            delay_us: 0.0,
        };
        let params_with_delay = BeamlineParams {
            flight_path_m: 25.0,
            delay_us: 10.0,
        };

        let tof_edges = vec![100.0, 200.0, 300.0];

        let e_no_delay = tof_edges_to_energy(&tof_edges, &params_no_delay).unwrap();
        let e_with_delay = tof_edges_to_energy(&tof_edges, &params_with_delay).unwrap();

        // With delay subtracted, effective TOF is shorter → energies are higher
        for i in 0..e_no_delay.len() {
            assert!(
                e_with_delay[i] >= e_no_delay[i],
                "Delayed energy should be >= no-delay at index {}",
                i,
            );
        }
    }

    #[test]
    fn test_energy_centers_geometric_mean() {
        let params = BeamlineParams {
            flight_path_m: 25.0,
            delay_us: 0.0,
        };
        let tof_edges = vec![100.0, 200.0, 300.0, 400.0];
        let edges = tof_edges_to_energy(&tof_edges, &params).unwrap();
        let centers = tof_edges_to_energy_centers(&tof_edges, &params).unwrap();

        assert_eq!(centers.len(), edges.len() - 1);
        for i in 0..centers.len() {
            let expected = (edges[i] * edges[i + 1]).sqrt();
            assert!(
                (centers[i] - expected).abs() < 1e-10,
                "Center[{}] = {}, expected {}",
                i,
                centers[i],
                expected,
            );
        }
    }

    #[test]
    fn test_linspace_tof_edges() {
        let edges = linspace_tof_edges(100.0, 200.0, 5);
        assert_eq!(edges.len(), 6);
        assert!((edges[0] - 100.0).abs() < 1e-10);
        assert!((edges[5] - 200.0).abs() < 1e-10);
        assert!((edges[1] - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_insufficient_edges() {
        let params = BeamlineParams::default();
        let result = tof_edges_to_energy(&[100.0], &params);
        assert!(result.is_err());
    }
}
