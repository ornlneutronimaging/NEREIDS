//! Gaussian resolution function for testing and simple instruments.
//!
//! Teacher reference: `sammy/src/grp/mgrp1.f90` (DELTAL, DELTAG, DELTAB parameters)

use nereids_core::{EnergyGrid, PhysicsError, ResolutionFunction};

const GAUSSIAN_CUTOFF_SIGMAS: f64 = 5.0;
const MIN_NORMALIZATION: f64 = 1e-300;

/// Gaussian resolution function with constant width.
#[derive(Debug, Clone)]
pub struct GaussianResolution {
    /// Gaussian sigma in eV.
    pub sigma: f64,
}

fn validate_energy_grid(energies: &[f64]) -> Result<(), PhysicsError> {
    if energies.is_empty() {
        return Err(PhysicsError::EmptyEnergyGrid);
    }
    if energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "energy values must be finite".to_string(),
        ));
    }
    if energies.windows(2).any(|w| w[1] < w[0]) {
        return Err(PhysicsError::InvalidParameter(
            "energy values must be sorted ascending".to_string(),
        ));
    }
    Ok(())
}

fn interpolate_clamped(energies: &[f64], values: &[f64], x: f64) -> f64 {
    let n = energies.len();
    let idx = energies.partition_point(|&e| e < x);
    if idx == 0 {
        values[0]
    } else if idx >= n {
        values[n - 1]
    } else {
        let x0 = energies[idx - 1];
        let x1 = energies[idx];
        let y0 = values[idx - 1];
        let y1 = values[idx];
        if x1 == x0 {
            y1
        } else {
            let t = (x - x0) / (x1 - x0);
            y0 + t * (y1 - y0)
        }
    }
}

impl ResolutionFunction for GaussianResolution {
    fn convolve(&self, energy: &EnergyGrid, spectrum: &[f64]) -> Result<Vec<f64>, PhysicsError> {
        let energies = &energy.values;
        validate_energy_grid(energies)?;
        if spectrum.len() != energies.len() {
            return Err(PhysicsError::DimensionMismatch {
                expected: energies.len(),
                got: spectrum.len(),
            });
        }
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "gaussian sigma must be finite and non-negative, got {}",
                self.sigma
            )));
        }
        if self.sigma == 0.0 {
            return Ok(spectrum.to_vec());
        }

        let n = energies.len();
        let sigma = self.sigma;
        let mut out = Vec::with_capacity(n);

        for (i, &e_center) in energies.iter().enumerate() {
            let e_low = e_center - GAUSSIAN_CUTOFF_SIGMAS * sigma;
            let e_high = e_center + GAUSSIAN_CUTOFF_SIGMAS * sigma;

            let left = energies.partition_point(|&e| e < e_low).saturating_sub(1);
            let right = energies.partition_point(|&e| e <= e_high).min(n - 1);

            let mut weighted = 0.0;
            let mut norm = 0.0;

            if left < right {
                for j in (left + 1)..=right {
                    let seg_e0 = energies[j - 1];
                    let seg_e1 = energies[j];

                    let a = seg_e0.max(e_low);
                    let b = seg_e1.min(e_high);
                    if b <= a {
                        continue;
                    }

                    let s_a = interpolate_clamped(energies, spectrum, a);
                    let s_b = interpolate_clamped(energies, spectrum, b);
                    let z_a = (a - e_center) / sigma;
                    let z_b = (b - e_center) / sigma;
                    let g_a = (-0.5 * z_a * z_a).exp();
                    let g_b = (-0.5 * z_b * z_b).exp();
                    let h = b - a;

                    weighted += 0.5 * h * (g_a * s_a + g_b * s_b);
                    norm += 0.5 * h * (g_a + g_b);
                }
            }

            if norm > MIN_NORMALIZATION {
                out.push(weighted / norm);
            } else {
                out.push(spectrum[i]);
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_zero_sigma_identity() {
        let energy = EnergyGrid::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let spectrum = vec![10.0, 20.0, 30.0, 40.0];
        let r = GaussianResolution { sigma: 0.0 };
        let out = r.convolve(&energy, &spectrum).unwrap();
        assert_eq!(out, spectrum);
    }

    #[test]
    fn test_gaussian_constant_signal_preserved() {
        let energy = EnergyGrid::new((0..201).map(|i| 5.0 + i as f64 * 0.05).collect()).unwrap();
        let spectrum = vec![7.5; energy.len()];
        let r = GaussianResolution { sigma: 0.2 };
        let out = r.convolve(&energy, &spectrum).unwrap();
        for y in out {
            assert!((y - 7.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gaussian_spreads_peak() {
        let energy = EnergyGrid::new((0..401).map(|i| 9.0 + i as f64 * 0.005).collect()).unwrap();
        let mut spectrum = vec![0.0; energy.len()];
        let mid = spectrum.len() / 2;
        spectrum[mid] = 1000.0;

        let r = GaussianResolution { sigma: 0.03 };
        let out = r.convolve(&energy, &spectrum).unwrap();
        assert!(out[mid] < spectrum[mid]);
        assert!(out[mid - 1] > 0.0);
        assert!(out[mid + 1] > 0.0);
    }

    #[test]
    fn test_gaussian_invalid_sigma_errors() {
        let energy = EnergyGrid::new(vec![1.0, 2.0]).unwrap();
        let spectrum = vec![1.0, 2.0];
        let r = GaussianResolution { sigma: -1.0 };
        assert!(matches!(
            r.convolve(&energy, &spectrum),
            Err(PhysicsError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_gaussian_dimension_mismatch_errors() {
        let energy = EnergyGrid::new(vec![1.0, 2.0, 3.0]).unwrap();
        let spectrum = vec![1.0, 2.0];
        let r = GaussianResolution { sigma: 0.1 };
        assert!(matches!(
            r.convolve(&energy, &spectrum),
            Err(PhysicsError::DimensionMismatch { .. })
        ));
    }
}
