//! Energy-dependent Gaussian resolution in the SAMMY RSL style.
//!
//! This models the legacy DELTAL/DELTAG/DELTTT pathway where the broadening
//! width depends on energy through flight-path and timing terms.

use nereids_core::{EnergyGrid, PhysicsError, ResolutionFunction};

const SM2: f64 = 72.298_252_179_105_06;
const GAUSSIAN_CUTOFF_SIGMAS: f64 = 5.0;
const MIN_NORMALIZATION: f64 = 1e-300;

/// SAMMY-style RSL Gaussian-resolution parameters.
#[derive(Debug, Clone, Copy)]
pub struct RslGaussianResolution {
    /// Neutron flight path (meters).
    pub flight_path_m: f64,
    /// Spatial path-spread term (meters), SAMMY `DELTAL`.
    pub delta_l_m: f64,
    /// Exponential timing component (microseconds), SAMMY `DELTTT`.
    pub delta_t_exp_us: f64,
    /// Gaussian timing component (microseconds), SAMMY `DELTAG`.
    pub delta_t_gaus_us: f64,
}

impl RslGaussianResolution {
    /// Width parameter (eV) used in the RSL Gaussian kernel.
    pub fn width_parameter_ev(&self, energy_ev: f64) -> f64 {
        let dist_inv = 1.0 / self.flight_path_m;
        let ao2 = (1.201_122_408_786_449_8 / SM2 * dist_inv * self.delta_t_gaus_us).powi(2);
        let bo2 = (0.816_496_580_927_726 * self.delta_l_m * dist_inv).powi(2);
        let base = energy_ev * energy_ev * (ao2 * energy_ev + bo2);
        let exp_term = self.delta_t_exp_us * self.delta_t_exp_us;
        (base + exp_term).sqrt()
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !self.flight_path_m.is_finite() || self.flight_path_m <= 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "flight_path_m must be finite and positive, got {}",
                self.flight_path_m
            )));
        }
        if !self.delta_l_m.is_finite() || self.delta_l_m < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "delta_l_m must be finite and non-negative, got {}",
                self.delta_l_m
            )));
        }
        if !self.delta_t_exp_us.is_finite() || self.delta_t_exp_us < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "delta_t_exp_us must be finite and non-negative, got {}",
                self.delta_t_exp_us
            )));
        }
        if !self.delta_t_gaus_us.is_finite() || self.delta_t_gaus_us < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "delta_t_gaus_us must be finite and non-negative, got {}",
                self.delta_t_gaus_us
            )));
        }
        Ok(())
    }
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

impl ResolutionFunction for RslGaussianResolution {
    fn convolve(&self, energy: &EnergyGrid, spectrum: &[f64]) -> Result<Vec<f64>, PhysicsError> {
        let energies = &energy.values;
        validate_energy_grid(energies)?;
        self.validate()?;
        if spectrum.len() != energies.len() {
            return Err(PhysicsError::DimensionMismatch {
                expected: energies.len(),
                got: spectrum.len(),
            });
        }
        if self.delta_l_m == 0.0 && self.delta_t_exp_us == 0.0 && self.delta_t_gaus_us == 0.0 {
            return Ok(spectrum.to_vec());
        }

        let n = energies.len();
        let mut out = Vec::with_capacity(n);
        for (i, &e_center) in energies.iter().enumerate() {
            let width = self.width_parameter_ev(e_center);
            if !width.is_finite() || width <= 0.0 {
                out.push(spectrum[i]);
                continue;
            }
            let sigma = width / std::f64::consts::SQRT_2;
            let e_low = e_center - GAUSSIAN_CUTOFF_SIGMAS * width;
            let e_high = e_center + GAUSSIAN_CUTOFF_SIGMAS * width;

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
    fn test_rsl_gaussian_zero_width_identity() {
        let energy = EnergyGrid::new(vec![315.0, 320.0, 325.0, 330.0]).unwrap();
        let spectrum = vec![1.0, 2.0, 3.0, 4.0];
        let r = RslGaussianResolution {
            flight_path_m: 18.9,
            delta_l_m: 0.0,
            delta_t_exp_us: 0.0,
            delta_t_gaus_us: 0.0,
        };
        let out = r.convolve(&energy, &spectrum).unwrap();
        assert_eq!(out, spectrum);
    }

    #[test]
    fn test_rsl_gaussian_constant_signal_preserved() {
        let energy = EnergyGrid::new((0..301).map(|i| 315.0 + i as f64 * 0.05).collect()).unwrap();
        let spectrum = vec![7.5; energy.len()];
        let r = RslGaussianResolution {
            flight_path_m: 18.9,
            delta_l_m: 0.025,
            delta_t_exp_us: 0.0,
            delta_t_gaus_us: 0.05,
        };
        let out = r.convolve(&energy, &spectrum).unwrap();
        for y in out {
            assert!((y - 7.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rsl_gaussian_invalid_params_error() {
        let energy = EnergyGrid::new(vec![315.0, 320.0]).unwrap();
        let spectrum = vec![1.0, 2.0];
        let r = RslGaussianResolution {
            flight_path_m: 0.0,
            delta_l_m: 0.025,
            delta_t_exp_us: 0.0,
            delta_t_gaus_us: 0.05,
        };
        assert!(matches!(
            r.convolve(&energy, &spectrum),
            Err(PhysicsError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_rsl_gaussian_dimension_mismatch_errors() {
        let energy = EnergyGrid::new(vec![315.0, 320.0, 325.0]).unwrap();
        let spectrum = vec![1.0, 2.0];
        let r = RslGaussianResolution {
            flight_path_m: 18.9,
            delta_l_m: 0.025,
            delta_t_exp_us: 0.0,
            delta_t_gaus_us: 0.05,
        };
        assert!(matches!(
            r.convolve(&energy, &spectrum),
            Err(PhysicsError::DimensionMismatch { .. })
        ));
    }
}
