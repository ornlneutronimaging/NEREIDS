//! User-defined tabulated resolution function.
//!
//! For VENUS, the resolution function comes from a Monte Carlo simulation
//! capturing the SNS source signature and beamline optics. This is loaded
//! from a user-provided file (typically passed via SAMMY Card Set 16).

use nereids_core::{EnergyGrid, PhysicsError, ResolutionFunction};

const MIN_NORMALIZATION: f64 = 1e-300;

/// Tabulated resolution function loaded from user-provided data.
#[derive(Debug, Clone)]
pub struct TabulatedResolution {
    /// Kernel energy offsets (eV), sorted ascending.
    ///
    /// Each output value at `E` uses kernel samples over `[E + offset_i]`.
    pub offsets_ev: Vec<f64>,
    /// Kernel values at each energy-offset point.
    pub kernel: Vec<f64>,
}

fn validate_sorted_finite(values: &[f64], name: &str) -> Result<(), PhysicsError> {
    if values.is_empty() {
        return Err(PhysicsError::InvalidParameter(format!(
            "{name} must not be empty"
        )));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PhysicsError::InvalidParameter(format!(
            "{name} must contain only finite values"
        )));
    }
    if values.windows(2).any(|w| w[1] < w[0]) {
        return Err(PhysicsError::InvalidParameter(format!(
            "{name} must be sorted ascending"
        )));
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
        let e0 = energies[idx - 1];
        let e1 = energies[idx];
        let y0 = values[idx - 1];
        let y1 = values[idx];
        if e1 == e0 {
            y1
        } else {
            let t = (x - e0) / (e1 - e0);
            y0 + t * (y1 - y0)
        }
    }
}

impl ResolutionFunction for TabulatedResolution {
    fn convolve(&self, energy: &EnergyGrid, spectrum: &[f64]) -> Result<Vec<f64>, PhysicsError> {
        let energies = &energy.values;
        if energies.is_empty() {
            return Err(PhysicsError::EmptyEnergyGrid);
        }
        if spectrum.len() != energies.len() {
            return Err(PhysicsError::DimensionMismatch {
                expected: energies.len(),
                got: spectrum.len(),
            });
        }
        validate_sorted_finite(energies, "energy grid")?;

        if self.offsets_ev.len() != self.kernel.len() {
            return Err(PhysicsError::DimensionMismatch {
                expected: self.offsets_ev.len(),
                got: self.kernel.len(),
            });
        }
        if self.offsets_ev.len() < 2 {
            return Err(PhysicsError::InvalidParameter(
                "tabulated kernel requires at least two offset points".to_string(),
            ));
        }
        validate_sorted_finite(&self.offsets_ev, "kernel offsets")?;
        if self.kernel.iter().any(|k| !k.is_finite()) {
            return Err(PhysicsError::InvalidParameter(
                "kernel values must be finite".to_string(),
            ));
        }
        if self.kernel.iter().any(|&k| k < 0.0) {
            return Err(PhysicsError::InvalidParameter(
                "kernel values must be non-negative".to_string(),
            ));
        }

        let emin = energies[0];
        let emax = energies[energies.len() - 1];
        let mut convolved = Vec::with_capacity(energies.len());
        for (i, &e_center) in energies.iter().enumerate() {
            let mut weighted = 0.0;
            let mut norm = 0.0;

            for j in 1..self.offsets_ev.len() {
                let o0 = self.offsets_ev[j - 1];
                let o1 = self.offsets_ev[j];
                let dx = o1 - o0;
                if dx <= 0.0 {
                    continue;
                }

                let e0 = e_center + o0;
                let e1 = e_center + o1;
                let clip_start = e0.max(emin);
                let clip_end = e1.min(emax);
                if clip_end <= clip_start {
                    continue;
                }

                let oc0 = clip_start - e_center;
                let oc1 = clip_end - e_center;
                let w0 = self.kernel[j - 1];
                let w1 = self.kernel[j];
                let t0 = (oc0 - o0) / dx;
                let t1 = (oc1 - o0) / dx;
                let k0 = w0 + t0 * (w1 - w0);
                let k1 = w0 + t1 * (w1 - w0);
                let s0 = interpolate_clamped(energies, spectrum, clip_start);
                let s1 = interpolate_clamped(energies, spectrum, clip_end);
                let dx_eff = oc1 - oc0;

                weighted += 0.5 * dx_eff * (k0 * s0 + k1 * s1);
                norm += 0.5 * dx_eff * (k0 + k1);
            }

            if norm > MIN_NORMALIZATION {
                convolved.push(weighted / norm);
            } else {
                convolved.push(spectrum[i]);
            }
        }

        Ok(convolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tabulated_constant_signal_preserved() {
        let energy = EnergyGrid::new((0..201).map(|i| 1.0 + i as f64 * 0.05).collect()).unwrap();
        let spectrum = vec![3.5; energy.len()];
        let res = TabulatedResolution {
            offsets_ev: vec![-0.2, 0.0, 0.2],
            kernel: vec![0.0, 1.0, 0.0],
        };
        let out = res.convolve(&energy, &spectrum).unwrap();
        for y in out {
            assert!((y - 3.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_tabulated_spreads_peak() {
        let energy = EnergyGrid::new((0..2001).map(|i| 9.0 + i as f64 * 0.001).collect()).unwrap();
        let mut spectrum = vec![0.0; energy.len()];
        let mid = spectrum.len() / 2;
        spectrum[mid] = 100.0;

        let res = TabulatedResolution {
            offsets_ev: vec![-0.002, -0.001, 0.0, 0.001, 0.002],
            kernel: vec![0.0, 0.5, 1.0, 0.5, 0.0],
        };
        let out = res.convolve(&energy, &spectrum).unwrap();
        assert!(out[mid] < spectrum[mid]);
        assert!(out[mid - 1] > 0.0);
        assert!(out[mid + 1] > 0.0);
    }

    #[test]
    fn test_tabulated_dimension_mismatch_errors() {
        let energy = EnergyGrid::new(vec![1.0, 2.0]).unwrap();
        let spectrum = vec![1.0, 2.0];
        let res = TabulatedResolution {
            offsets_ev: vec![-0.1, 0.1],
            kernel: vec![1.0],
        };
        assert!(matches!(
            res.convolve(&energy, &spectrum),
            Err(PhysicsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_tabulated_unsorted_offsets_error() {
        let energy = EnergyGrid::new(vec![1.0, 2.0]).unwrap();
        let spectrum = vec![1.0, 2.0];
        let res = TabulatedResolution {
            offsets_ev: vec![0.1, -0.1],
            kernel: vec![1.0, 1.0],
        };
        assert!(matches!(
            res.convolve(&energy, &spectrum),
            Err(PhysicsError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_tabulated_negative_kernel_error() {
        let energy = EnergyGrid::new(vec![1.0, 2.0]).unwrap();
        let spectrum = vec![1.0, 2.0];
        let res = TabulatedResolution {
            offsets_ev: vec![-0.1, 0.1],
            kernel: vec![1.0, -1.0],
        };
        assert!(matches!(
            res.convolve(&energy, &spectrum),
            Err(PhysicsError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_tabulated_truncates_kernel_at_energy_edges() {
        let energy = EnergyGrid::new(vec![0.0, 1.0, 2.0]).unwrap();
        let spectrum = vec![0.0, 1.0, 2.0];
        let res = TabulatedResolution {
            offsets_ev: vec![-1.0, 0.0, 1.0],
            kernel: vec![1.0, 1.0, 1.0],
        };
        let out = res.convolve(&energy, &spectrum).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] - 1.5).abs() < 1e-12);
    }
}
