//! Free-gas model Doppler broadening.
//!
//! Broadens 0K cross sections to finite temperature by convolving with
//! a Gaussian kernel whose width depends on the atomic mass and temperature.
//!
//! # Physics
//!
//! At finite temperature, target atoms have a thermal velocity distribution.
//! The Doppler-shifted neutron-nucleus relative velocity creates an
//! energy-dependent Gaussian broadening:
//!
//! ```text
//! Doppler width:  Δ(E) = sqrt(4 · k_B · T · E / AWR)
//! Broadened XS:   σ(E; T) = (1/(√π·Δ)) ∫ σ₀(E') exp(-(E-E')²/Δ²) dE'
//! ```
//!
//! # Algorithm
//!
//! Direct trapezoidal convolution with kernel truncation at ±4Δ and
//! normalized kernel (compensates for truncation and grid-edge effects).
//!
//! # Important
//!
//! The input energy grid must be fine enough to resolve the 0K cross section
//! structure. For narrow resonances (meV-scale widths), create an auxiliary
//! fine grid via [`create_auxiliary_grid`] before calling [`broaden_cross_sections`].
//!
//! # References
//!
//! - SAMMY `mclm3.f90` (free-gas component)
//! - SAMMY Users' Guide (ORNL/TM-9179/R8), Section IV.E

use nereids_core::constants::BOLTZMANN_EV_PER_K;
use nereids_core::PhysicsError;

/// Number of Doppler widths for kernel truncation.
///
/// At ±4Δ, the Gaussian kernel is exp(-16) ≈ 1.1e-7.
const KERNEL_CUTOFF: f64 = 4.0;

/// Compute the Doppler width parameter Δ(E) for the free-gas model.
///
/// # Formula
///
/// ```text
/// Δ = sqrt(4 · k_B · T · E / AWR)
/// ```
///
/// where AWR = target mass / neutron mass (dimensionless).
pub fn doppler_width(energy_ev: f64, temp_k: f64, awr: f64) -> f64 {
    // Use |E| so negative incident energies (supported by Reich-Moore) produce
    // a real Doppler width instead of NaN.
    (4.0 * BOLTZMANN_EV_PER_K * temp_k * energy_ev.abs() / awr).sqrt()
}

/// Apply free-gas Doppler broadening to cross sections.
///
/// Convolves σ(E; 0K) with a Gaussian kernel using trapezoidal integration
/// and a normalized kernel to compensate for truncation and edge effects.
///
/// # Arguments
///
/// * `xs_0k` - 0K cross sections \[barns\], same length as `energies`
/// * `energies` - Energy grid \[eV\], sorted ascending
/// * `awr` - Atomic weight ratio (target mass / neutron mass, dimensionless)
/// * `temperature_k` - Sample temperature in Kelvin; 0 means no broadening
///
/// # Returns
///
/// Broadened cross sections \[barns\], same length as input.
pub fn broaden_cross_sections(
    xs_0k: &[f64],
    energies: &[f64],
    awr: f64,
    temperature_k: f64,
) -> Result<Vec<f64>, PhysicsError> {
    let n = energies.len();

    if n == 0 {
        return Err(PhysicsError::EmptyEnergyGrid);
    }

    if xs_0k.len() != n {
        return Err(PhysicsError::DimensionMismatch {
            expected: n,
            got: xs_0k.len(),
        });
    }

    if awr <= 0.0 || !awr.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "AWR must be positive and finite, got {awr}"
        )));
    }

    if !temperature_k.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "temperature must be finite, got {temperature_k}"
        )));
    }

    // No broadening at T <= 0
    if temperature_k <= 0.0 {
        return Ok(xs_0k.to_vec());
    }

    let mut broadened = vec![0.0; n];

    for i in 0..n {
        let e_i = energies[i];
        let delta = doppler_width(e_i, temperature_k, awr);

        if delta < 1e-20 {
            // Negligible broadening (e.g., near-zero energy)
            broadened[i] = xs_0k[i];
            continue;
        }

        let e_min = e_i - KERNEL_CUTOFF * delta;
        let e_max = e_i + KERNEL_CUTOFF * delta;
        let inv_delta_sq = 1.0 / (delta * delta);

        // Binary search for grid points within the kernel window
        let j_start = energies.partition_point(|&e| e < e_min);
        let j_end = energies.partition_point(|&e| e <= e_max);

        // Include one extra point on each side for trapezoidal coverage
        let j_lo = j_start.saturating_sub(1);
        let j_hi = j_end.min(n - 1);

        if j_lo >= j_hi {
            // Not enough grid points for trapezoidal integration
            broadened[i] = xs_0k[i];
            continue;
        }

        // Trapezoidal integration with normalized kernel
        let mut sum_weighted = 0.0;
        let mut sum_kernel = 0.0;

        for j in (j_lo + 1)..=j_hi {
            let h = energies[j] - energies[j - 1];

            let diff_prev = e_i - energies[j - 1];
            let diff_curr = e_i - energies[j];

            let g_prev = (-diff_prev * diff_prev * inv_delta_sq).exp();
            let g_curr = (-diff_curr * diff_curr * inv_delta_sq).exp();

            sum_weighted += 0.5 * h * (g_prev * xs_0k[j - 1] + g_curr * xs_0k[j]);
            sum_kernel += 0.5 * h * (g_prev + g_curr);
        }

        broadened[i] = if sum_kernel > 1e-100 {
            sum_weighted / sum_kernel
        } else {
            xs_0k[i]
        };
    }

    Ok(broadened)
}

/// Linearly interpolate values from a source grid to target energies.
///
/// Values outside the source range are clamped to the nearest edge value.
///
/// # Arguments
///
/// * `source_energies` - Source energy grid (sorted ascending)
/// * `source_values` - Source values, same length as `source_energies`
/// * `target_energies` - Target energies to interpolate to
pub fn interpolate_to_grid(
    source_energies: &[f64],
    source_values: &[f64],
    target_energies: &[f64],
) -> Result<Vec<f64>, PhysicsError> {
    if source_energies.is_empty() {
        return Err(PhysicsError::EmptyEnergyGrid);
    }
    if source_values.len() != source_energies.len() {
        return Err(PhysicsError::DimensionMismatch {
            expected: source_energies.len(),
            got: source_values.len(),
        });
    }

    let n_src = source_energies.len();
    Ok(target_energies
        .iter()
        .map(|&e| {
            let idx = source_energies.partition_point(|&x| x < e);
            if idx == 0 {
                source_values[0]
            } else if idx >= n_src {
                source_values[n_src - 1]
            } else {
                let e0 = source_energies[idx - 1];
                let e1 = source_energies[idx];
                let t = (e - e0) / (e1 - e0);
                source_values[idx - 1] + t * (source_values[idx] - source_values[idx - 1])
            }
        })
        .collect())
}

/// Create an auxiliary energy grid that resolves narrow resonances.
///
/// Adds fine-spaced points near each resonance center to ensure the 0K
/// cross section structure is captured for subsequent Doppler broadening.
///
/// # Arguments
///
/// * `data_energies` - Data/output energy grid
/// * `resonance_energies` - Resonance center energies \[eV\]
/// * `resonance_widths` - Total resonance widths \[eV\] (Γ = Γγ + Γn + ...)
/// * `temperature_k` - Sample temperature in Kelvin
/// * `awr` - Atomic weight ratio
///
/// # Returns
///
/// Sorted, deduplicated energy grid that includes both data points and
/// fine points near resonances.
pub fn create_auxiliary_grid(
    data_energies: &[f64],
    resonance_energies: &[f64],
    resonance_widths: &[f64],
    temperature_k: f64,
    awr: f64,
) -> Vec<f64> {
    let mut grid: Vec<f64> = data_energies.to_vec();

    for (&e_res, &gamma) in resonance_energies.iter().zip(resonance_widths.iter()) {
        // Doppler width at the resonance energy
        let delta = if temperature_k > 0.0 {
            doppler_width(e_res, temperature_k, awr)
        } else {
            0.0
        };

        // Span: max of 5*Γ (resolve resonance) or 5*Δ (cover Doppler kernel)
        let span = (5.0 * gamma).max(5.0 * delta).max(0.01);

        // Fine spacing: min of Γ/10 and Δ/20 (only consider Δ when broadening is active)
        let spacing = if delta > 0.0 {
            (gamma / 10.0).min(delta / 20.0).max(1e-5)
        } else {
            (gamma / 10.0).max(1e-5)
        };

        let e_lo = (e_res - span).max(0.0);
        let e_hi = e_res + span;

        let mut e = e_lo;
        while e <= e_hi {
            grid.push(e);
            e += spacing;
        }
    }

    // Sort and deduplicate (within tolerance)
    grid.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite energies"));
    grid.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doppler_width_values() {
        // At E=10 eV, T=300K, AWR=10:
        // Δ = sqrt(4 * 8.617e-5 * 300 * 10 / 10) = sqrt(0.10340) ≈ 0.3216
        let d = doppler_width(10.0, 300.0, 10.0);
        assert!((d - 0.3216).abs() < 0.001, "got {d}");
    }

    #[test]
    fn test_doppler_width_at_50k() {
        // At E=10 eV, T=50K, AWR=10:
        // Δ = sqrt(4 * 8.617e-5 * 50 * 10 / 10) = sqrt(0.01723) ≈ 0.1313
        let d = doppler_width(10.0, 50.0, 10.0);
        assert!((d - 0.1313).abs() < 0.001, "got {d}");
    }

    #[test]
    fn test_doppler_width_zero_temp() {
        assert_eq!(doppler_width(10.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn test_broaden_zero_temperature() {
        let xs = vec![1.0, 2.0, 3.0];
        let energies = vec![1.0, 2.0, 3.0];

        let result = broaden_cross_sections(&xs, &energies, 10.0, 0.0).unwrap();
        assert_eq!(result, xs);
    }

    #[test]
    fn test_broaden_constant_xs_unchanged() {
        // A constant cross section should remain constant after broadening.
        let n = 200;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + i as f64 * 0.05).collect();
        let xs = vec![100.0; n];

        let result = broaden_cross_sections(&xs, &energies, 10.0, 300.0).unwrap();
        // Interior points (away from edges) should be very close to 100.
        for (i, &r) in result.iter().enumerate().skip(20).take(n - 40) {
            assert!(
                (r - 100.0).abs() < 0.1,
                "constant XS changed at i={i}: got {r}"
            );
        }
    }

    #[test]
    fn test_broaden_empty_grid_errors() {
        let result = broaden_cross_sections(&[], &[], 10.0, 300.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_broaden_invalid_awr_errors() {
        let result = broaden_cross_sections(&[1.0], &[10.0], -1.0, 300.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_broaden_dimension_mismatch_errors() {
        let result = broaden_cross_sections(&[1.0, 2.0], &[10.0], 10.0, 300.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_broaden_peak_spreads() {
        // A delta-like peak on a grid should spread after broadening.
        let n = 401;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + i as f64 * 0.025).collect();
        let mid = n / 2;
        let mut xs = vec![0.0; n];
        xs[mid] = 1000.0;

        let result = broaden_cross_sections(&xs, &energies, 10.0, 300.0).unwrap();
        // Peak should be reduced
        assert!(result[mid] < 1000.0);
        // Neighbors should pick up value
        assert!(result[mid - 1] > 0.0);
        assert!(result[mid + 1] > 0.0);
    }

    #[test]
    fn test_broaden_preserves_area() {
        // A localized feature should approximately preserve its area.
        let n = 1001;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + i as f64 * 0.01).collect();
        let mid = n / 2;

        // Create a Gaussian-like feature
        let sigma = 0.1;
        let e_mid = energies[mid];
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| 1000.0 * (-(e - e_mid).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();

        // Compute area before broadening
        let area_before: f64 = (1..n)
            .map(|i| 0.5 * (energies[i] - energies[i - 1]) * (xs[i] + xs[i - 1]))
            .sum();

        let result = broaden_cross_sections(&xs, &energies, 10.0, 300.0).unwrap();

        // Compute area after broadening
        let area_after: f64 = (1..n)
            .map(|i| 0.5 * (energies[i] - energies[i - 1]) * (result[i] + result[i - 1]))
            .sum();

        let rel_error = (area_after - area_before).abs() / area_before;
        assert!(
            rel_error < 0.01,
            "area not preserved: before={area_before:.4}, after={area_after:.4}, rel_err={rel_error:.6}"
        );
    }

    #[test]
    fn test_interpolate_basic() {
        let src_e = vec![1.0, 2.0, 3.0, 4.0];
        let src_xs = vec![10.0, 20.0, 30.0, 40.0];
        let tgt_e = vec![1.5, 2.5, 3.5];

        let result = interpolate_to_grid(&src_e, &src_xs, &tgt_e).unwrap();
        assert!((result[0] - 15.0).abs() < 1e-10);
        assert!((result[1] - 25.0).abs() < 1e-10);
        assert!((result[2] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_extrapolation() {
        let src_e = vec![2.0, 4.0];
        let src_xs = vec![20.0, 40.0];

        // Below range: returns first value
        let result = interpolate_to_grid(&src_e, &src_xs, &[1.0]).unwrap();
        assert!((result[0] - 20.0).abs() < 1e-10);

        // Above range: returns last value
        let result = interpolate_to_grid(&src_e, &src_xs, &[5.0]).unwrap();
        assert!((result[0] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_create_auxiliary_grid_includes_data_points() {
        let data = vec![8.0, 9.0, 10.0, 11.0, 12.0];
        let res_e = vec![10.0];
        let res_w = vec![0.001]; // 1 meV width
        let grid = create_auxiliary_grid(&data, &res_e, &res_w, 50.0, 10.0);

        // Should include original data points
        for &e in &data {
            assert!(
                grid.iter().any(|&g| (g - e).abs() < 1e-9),
                "data point {e} missing from auxiliary grid"
            );
        }

        // Should have many more points than original
        assert!(grid.len() > data.len() * 10);
    }

    #[test]
    fn test_create_auxiliary_grid_resolves_resonance() {
        let data = vec![9.0, 11.0];
        let res_e = vec![10.0];
        let res_w = vec![0.001]; // 1 meV width
        let grid = create_auxiliary_grid(&data, &res_e, &res_w, 0.0, 10.0);

        // Near the resonance, grid spacing should be <= gamma/10
        let near_res: Vec<f64> = grid
            .iter()
            .copied()
            .filter(|&e| (e - 10.0).abs() < 0.005)
            .collect();
        assert!(
            near_res.len() >= 10,
            "not enough points near resonance: {}",
            near_res.len()
        );
    }
}
