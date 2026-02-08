//! Free-gas model Doppler broadening.
//!
//! Broadens 0K cross sections to finite temperature by convolving with
//! a Gaussian kernel whose width depends on the atomic mass and temperature.
//!
//! # Physics
//!
//! At finite temperature, target atoms have a thermal velocity distribution.
//! The free-gas model uses Gaussian broadening in `v = sign(E)·sqrt(|E|)`:
//!
//! ```text
//! Velocity width: D = sqrt(k_B · T · m_n / A_target)
//! Broadening:     σ(E; T) ∝ ∫ σ₀(E') exp(-(v(E)-v(E'))²/D²) dv'
//! ```
//! where `A_target` is the target mass in amu (SAMMY `DefTargetMass` convention).
//!
//! # Algorithm
//!
//! Direct trapezoidal convolution in sqrt(E) space (SAMMY free-gas style)
//! with kernel truncation at ±5·D and normalized kernel.
//!
//! # Important
//!
//! The input energy grid must be fine enough to resolve the 0K cross section
//! structure. For narrow resonances (meV-scale widths), create an auxiliary
//! fine grid via [`create_auxiliary_grid`] before calling [`broaden_cross_sections`].
//!
//! # References
//!
//! - SAMMY `mfgm1.f90` / `mfgm4.f90` (free-gas broadening path)
//! - SAMMY Users' Guide (ORNL/TM-9179/R8), Section IV.E

use nereids_core::constants::BOLTZMANN_EV_PER_K;
use nereids_core::PhysicsError;

/// Number of Doppler widths for kernel truncation.
///
/// At ±5Δ, the Gaussian kernel is exp(-25) ≈ 1.4e-11.
const KERNEL_CUTOFF: f64 = 5.0;
/// Small factor used by SAMMY when expanding auxiliary-grid bounds.
const AUX_BOUNDS_FUDGE: f64 = 1.001;

/// Safety guardrail for direct trapezoidal broadening cost.
const MAX_BROADENING_SEGMENTS: usize = 20_000_000;

/// Maximum total auxiliary grid size to avoid pathological broadening workloads.
const MAX_AUX_GRID_POINTS: usize = 50_000;
/// Maximum allowed adjacent spacing ratio before smoothing inserts points.
const AUX_SPACING_RATIO_LIMIT: f64 = 2.5;
/// Maximum smoothing passes for auxiliary-grid transition refinement.
const AUX_SMOOTHING_PASSES: usize = 4;

/// Minimum number of points across ±Γ used for under-resolved resonances.
///
/// SAMMY reports this condition as "fewer than 9 points across width."
const MIN_POINTS_ACROSS_WIDTH: usize = 9;

/// Default tail points in units of Γ around each refined resonance.
///
/// Mirrors SAMMY's default tail enrichment around narrow resonances.
const RESONANCE_TAIL_MULTIPLIERS: [f64; 6] = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0];

/// Small floor for effective resonance width to keep refinement stable.
const MIN_REFINEMENT_WIDTH_EV: f64 = 1.0e-12;

/// Neutron mass in amu (SAMMY Kvendf=1 convention).
const NEUTRON_MASS_AMU: f64 = 1.008_664_915_6;

/// Compute the Doppler width parameter Δ(E) for the free-gas model.
///
/// # Formula
///
/// ```text
/// Δ = sqrt(4 · k_B · T · |E| / A_target)
/// ```
///
/// where `A_target` is the target mass in amu (SAMMY `DefTargetMass` convention).
pub fn doppler_width(energy_ev: f64, temp_k: f64, awr: f64) -> f64 {
    // Use |E| so negative incident energies (supported by Reich-Moore) produce
    // a real Doppler width instead of NaN.
    (4.0 * BOLTZMANN_EV_PER_K * temp_k * energy_ev.abs() / awr).sqrt()
}

/// Apply free-gas Doppler broadening to cross sections.
///
/// Convolves σ(E; 0K) with a Gaussian kernel in `sqrt(E)` space using
/// trapezoidal integration and a normalized kernel.
///
/// # Arguments
///
/// * `xs_0k` - 0K cross sections \[barns\], same length as `energies`
/// * `energies` - Energy grid \[eV\], finite and sorted ascending
/// * `awr` - Target mass in amu (SAMMY `DefTargetMass`)
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
    if energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "energies must be finite".to_string(),
        ));
    }
    if energies.windows(2).any(|w| w[1] < w[0]) {
        return Err(PhysicsError::InvalidParameter(
            "energies must be sorted ascending".to_string(),
        ));
    }

    if awr <= 0.0 || !awr.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "AWR must be positive and finite, got {awr}"
        )));
    }

    if temperature_k < 0.0 || !temperature_k.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "temperature must be non-negative and finite, got {temperature_k}"
        )));
    }

    // No broadening at T = 0
    if temperature_k == 0.0 {
        return Ok(xs_0k.to_vec());
    }

    let ddo = (BOLTZMANN_EV_PER_K * temperature_k * NEUTRON_MASS_AMU / awr).sqrt();
    if ddo < 1e-20 {
        return Ok(xs_0k.to_vec());
    }
    let inv_ddo_sq = 1.0 / (ddo * ddo);
    let velocities: Vec<f64> = energies
        .iter()
        .map(|&e| if e < 0.0 { -(-e).sqrt() } else { e.sqrt() })
        .collect();

    let mut broadened = vec![0.0; n];
    let mut total_segments: usize = 0;

    for i in 0..n {
        let v_i = velocities[i];
        let v_min = v_i - KERNEL_CUTOFF * ddo;
        let v_max = v_i + KERNEL_CUTOFF * ddo;

        // Binary search for grid points within the kernel window in sqrt(E) space
        let j_start = velocities.partition_point(|&v| v < v_min);
        let j_end = velocities.partition_point(|&v| v <= v_max);

        // Include one extra point on each side for trapezoidal coverage
        let j_lo = j_start.saturating_sub(1);
        let j_hi = j_end.min(n - 1);

        if (j_hi - j_lo + 1) <= 2 {
            // Not enough points for stable broadening (SAMMY requires >2 points)
            broadened[i] = xs_0k[i];
            continue;
        }
        total_segments = total_segments.saturating_add(j_hi - j_lo);
        if total_segments > MAX_BROADENING_SEGMENTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "broadening workload too large for direct trapezoidal convolution (>{MAX_BROADENING_SEGMENTS} segments); reduce auxiliary-grid density"
            )));
        }

        // Trapezoidal integration with normalized kernel.
        //
        // SAMMY free-gas broadening evaluates the convolution in velocity
        // space and carries a Jacobian factor equivalent to E'/E in the
        // final expression (see mfgm2/mfgm4 path).
        let mut sum_weighted = 0.0;
        let mut sum_kernel = 0.0;
        let e_i_abs = energies[i].abs();

        if e_i_abs <= f64::EPSILON {
            broadened[i] = xs_0k[i];
            continue;
        }

        for j in (j_lo + 1)..=j_hi {
            let h = velocities[j] - velocities[j - 1];
            if h <= 0.0 {
                continue;
            }

            let diff_prev = v_i - velocities[j - 1];
            let diff_curr = v_i - velocities[j];

            let g_prev = (-diff_prev * diff_prev * inv_ddo_sq).exp();
            let g_curr = (-diff_curr * diff_curr * inv_ddo_sq).exp();

            let e_prev_abs = energies[j - 1].abs();
            let e_curr_abs = energies[j].abs();

            sum_weighted +=
                0.5 * h * (g_prev * e_prev_abs * xs_0k[j - 1] + g_curr * e_curr_abs * xs_0k[j]);
            sum_kernel += 0.5 * h * (g_prev + g_curr);
        }

        broadened[i] = if sum_kernel > 1e-100 {
            (sum_weighted / sum_kernel) / e_i_abs
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
/// * `target_energies` - Target energies to interpolate to (finite)
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
    if source_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "source energies must be finite".to_string(),
        ));
    }
    if source_energies.windows(2).any(|w| w[1] < w[0]) {
        return Err(PhysicsError::InvalidParameter(
            "source energies must be sorted ascending".to_string(),
        ));
    }
    if target_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "target energies must be finite".to_string(),
        ));
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
                let denom = e1 - e0;
                if denom == 0.0 {
                    // Adjacent duplicate energies; use right-endpoint value
                    source_values[idx]
                } else {
                    let t = (e - e0) / denom;
                    source_values[idx - 1] + t * (source_values[idx] - source_values[idx - 1])
                }
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
/// * `awr` - Target mass in amu (SAMMY `DefTargetMass`)
///
/// # Returns
///
/// Sorted, deduplicated energy grid that includes both data points and
/// fine points near resonances.
///
/// # Errors
///
/// Returns `PhysicsError::InvalidParameter` if any energy is non-finite.
pub fn create_auxiliary_grid(
    data_energies: &[f64],
    resonance_energies: &[f64],
    resonance_widths: &[f64],
    temperature_k: f64,
    awr: f64,
) -> Result<Vec<f64>, PhysicsError> {
    if resonance_energies.len() != resonance_widths.len() {
        return Err(PhysicsError::DimensionMismatch {
            expected: resonance_energies.len(),
            got: resonance_widths.len(),
        });
    }
    if data_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "data energies must be finite".to_string(),
        ));
    }
    if temperature_k < 0.0 || !temperature_k.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "temperature_k must be non-negative and finite, got {temperature_k}"
        )));
    }
    if awr <= 0.0 || !awr.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "awr must be positive and finite, got {awr}"
        )));
    }

    let mut grid: Vec<f64> = data_energies.to_vec();
    let data_min = data_energies.iter().copied().reduce(f64::min);
    let data_max = data_energies.iter().copied().reduce(f64::max);

    for (&e_res, &gamma) in resonance_energies.iter().zip(resonance_widths.iter()) {
        if !e_res.is_finite() || !gamma.is_finite() {
            return Err(PhysicsError::InvalidParameter(format!(
                "resonance energy and width must be finite, got E={e_res}, Γ={gamma}"
            )));
        }
        if gamma < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "resonance width must be non-negative, got Γ={gamma}"
            )));
        }
        if gamma == 0.0 {
            continue;
        }

        let width = gamma.max(MIN_REFINEMENT_WIDTH_EV);
        let e_lo = e_res - width;
        let e_hi = e_res + width;

        let points_across_width = grid.iter().filter(|&&e| e >= e_lo && e <= e_hi).count();
        if points_across_width < MIN_POINTS_ACROSS_WIDTH {
            let spacing = (2.0 * width) / ((MIN_POINTS_ACROSS_WIDTH - 1) as f64);
            for k in 0..MIN_POINTS_ACROSS_WIDTH {
                let e = e_lo + k as f64 * spacing;
                if data_min.is_none_or(|mn| e >= mn) && data_max.is_none_or(|mx| e <= mx) {
                    grid.push(e);
                }
            }
        }

        for mult in RESONANCE_TAIL_MULTIPLIERS {
            let e_left = e_res - mult * width;
            let e_right = e_res + mult * width;
            if data_min.is_none_or(|mn| e_left >= mn) && data_max.is_none_or(|mx| e_left <= mx) {
                grid.push(e_left);
            }
            if data_min.is_none_or(|mn| e_right >= mn) && data_max.is_none_or(|mx| e_right <= mx) {
                grid.push(e_right);
            }
        }

        if grid.len() > MAX_AUX_GRID_POINTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "auxiliary grid exceeds {} points; reduce resonance density/width inputs",
                MAX_AUX_GRID_POINTS
            )));
        }
    }

    // For T>0, extend the grid in sqrt(E) using SAMMY-like edge spacing.
    //
    // This mirrors the free-gas Escale/Vqcon behavior: derive a velocity
    // step from the first/last five data points and populate extra points
    // between [Emind, E_min) and (E_max, Emaxd].
    if temperature_k > 0.0 {
        let mut sorted_data = data_energies.to_vec();
        sorted_data.sort_unstable_by(f64::total_cmp);

        if let (Some(&min_e), Some(&max_e)) = (sorted_data.first(), sorted_data.last()) {
            let ddo = (BOLTZMANN_EV_PER_K * temperature_k * NEUTRON_MASS_AMU / awr).sqrt();
            if ddo > 0.0 && min_e > 0.0 && max_e > 0.0 {
                let v_min = min_e.sqrt();
                let v_max = max_e.sqrt();
                let bound_step = KERNEL_CUTOFF * AUX_BOUNDS_FUDGE * ddo;

                let emind = (v_min - bound_step).powi(2);
                let emaxd = (v_max + bound_step).powi(2);

                let n5 = sorted_data.len().min(5);
                if n5 >= 2 {
                    let e1 = sorted_data[0];
                    let e2 = sorted_data[1];
                    let e5_low = sorted_data[n5 - 1];
                    let d_low = (e5_low.sqrt() - e1.sqrt()) / ((n5 - 1) as f64);
                    if d_low > 0.0 {
                        let eefudg = (e2 - e1) * 1.0e-4;
                        if e1 >= emind + eefudg {
                            let x = e1.sqrt() - emind.sqrt();
                            let n = (x / d_low).floor() as usize + 1;
                            let start = e1.sqrt() - ((n + 1) as f64) * d_low;
                            for i in 1..=n {
                                let v = start + (i as f64) * d_low;
                                let e = v * v;
                                if e.is_finite() {
                                    grid.push(e);
                                }
                            }
                        }
                    }

                    let e_last = sorted_data[sorted_data.len() - 1];
                    let e5_high = sorted_data[sorted_data.len() - n5];
                    let d_high = (e_last.sqrt() - e5_high.sqrt()) / ((n5 - 1) as f64);
                    if d_high > 0.0 {
                        let x = emaxd.sqrt() - e_last.sqrt();
                        let n = (x / d_high).floor() as usize + 1;
                        for i in 1..=n {
                            let v = e_last.sqrt() + (i as f64) * d_high;
                            let e = v * v;
                            if e.is_finite() {
                                grid.push(e);
                            }
                        }
                    }
                }
            }
        }

        if grid.len() > MAX_AUX_GRID_POINTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "auxiliary grid exceeds {} points; reduce resonance density/width inputs",
                MAX_AUX_GRID_POINTS
            )));
        }
    }

    // Sort and deduplicate (within tolerance)
    if grid.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "non-finite energy in grid".to_string(),
        ));
    }
    grid.sort_unstable_by(f64::total_cmp);
    grid.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    // Smooth abrupt spacing jumps (SAMMY Adjust_Auxil style, simplified).
    // This keeps interpolation weights stable near refined resonance regions.
    for _ in 0..AUX_SMOOTHING_PASSES {
        if grid.len() < 3 {
            break;
        }
        let mut added = Vec::new();
        for i in 1..(grid.len() - 1) {
            let de_prev = grid[i] - grid[i - 1];
            let de_next = grid[i + 1] - grid[i];
            if de_prev <= 0.0 || de_next <= 0.0 {
                continue;
            }
            if (de_prev / de_next) > AUX_SPACING_RATIO_LIMIT {
                let mid = 0.5 * (grid[i - 1] + grid[i]);
                added.push(mid);
            } else if (de_next / de_prev) > AUX_SPACING_RATIO_LIMIT {
                let mid = 0.5 * (grid[i] + grid[i + 1]);
                added.push(mid);
            }
        }
        if added.is_empty() {
            break;
        }
        grid.extend(added);
        if grid.len() > MAX_AUX_GRID_POINTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "auxiliary grid exceeds {} points; reduce resonance density/width inputs",
                MAX_AUX_GRID_POINTS
            )));
        }
        grid.sort_unstable_by(f64::total_cmp);
        grid.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    }

    Ok(grid)
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
    fn test_broaden_unsorted_energies_error() {
        let result = broaden_cross_sections(&[1.0, 2.0], &[2.0, 1.0], 10.0, 300.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_broaden_non_finite_energies_error() {
        let result = broaden_cross_sections(&[1.0, 2.0], &[1.0, f64::NAN], 10.0, 300.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
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
    fn test_interpolate_empty_source_errors() {
        let result = interpolate_to_grid(&[], &[], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_mismatched_source_errors() {
        let result = interpolate_to_grid(&[1.0, 2.0], &[10.0], &[1.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_unsorted_source_energies_error() {
        let result = interpolate_to_grid(&[2.0, 1.0], &[20.0, 10.0], &[1.5]);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_interpolate_non_finite_target_energies_error() {
        let result = interpolate_to_grid(&[1.0, 2.0], &[10.0, 20.0], &[f64::NAN]);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_includes_data_points() {
        let data = vec![8.0, 9.0, 10.0, 11.0, 12.0];
        let res_e = vec![10.0];
        let res_w = vec![0.001]; // 1 meV width
        let grid = create_auxiliary_grid(&data, &res_e, &res_w, 50.0, 10.0).unwrap();

        // Should include original data points
        for &e in &data {
            assert!(
                grid.iter().any(|&g| (g - e).abs() < 1e-9),
                "data point {e} missing from auxiliary grid"
            );
        }

        // Refinement should add at least a few points.
        assert!(grid.len() > data.len());
    }

    #[test]
    fn test_create_auxiliary_grid_resolves_resonance() {
        let data = vec![9.0, 11.0];
        let res_e = vec![10.0];
        let res_w = vec![0.001]; // 1 meV width
        let grid = create_auxiliary_grid(&data, &res_e, &res_w, 0.0, 10.0).unwrap();

        // Near the resonance, we should have at least SAMMY's default
        // "9 points across width" coverage.
        let near_res: Vec<f64> = grid
            .iter()
            .copied()
            .filter(|&e| (e - 10.0).abs() <= 1.1e-3)
            .collect();
        assert!(
            near_res.len() >= MIN_POINTS_ACROSS_WIDTH,
            "not enough points near resonance: {}",
            near_res.len()
        );
    }

    #[test]
    fn test_create_auxiliary_grid_mismatched_resonance_arrays_error() {
        let result = create_auxiliary_grid(&[8.0, 12.0], &[10.0, 11.0], &[0.001], 50.0, 10.0);
        assert!(matches!(
            result,
            Err(PhysicsError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn test_create_auxiliary_grid_non_finite_data_energy_error() {
        let result = create_auxiliary_grid(&[8.0, f64::NAN, 12.0], &[10.0], &[0.001], 50.0, 10.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_invalid_temperature_error() {
        let result = create_auxiliary_grid(&[8.0, 12.0], &[10.0], &[0.001], -1.0, 10.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_invalid_awr_error() {
        let result = create_auxiliary_grid(&[8.0, 12.0], &[10.0], &[0.001], 50.0, 0.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_uses_true_min_from_unsorted_data() {
        let data = vec![12.0, 8.0, 10.0];
        let grid = create_auxiliary_grid(&data, &[9.0], &[0.001], 0.0, 10.0).unwrap();
        assert!(
            grid.iter().any(|&e| e < 9.0),
            "auxiliary grid should include points below resonance when min(data)<E_res"
        );
    }

    #[test]
    fn test_create_auxiliary_grid_bounded_for_narrow_resonance() {
        let data: Vec<f64> = (0..31).map(|i| 9.7 + i as f64 * 0.02).collect();
        let grid = create_auxiliary_grid(&data, &[10.0], &[1e-5], 300.0, 1.0).unwrap();
        assert!(
            grid.len() < 1_000,
            "grid unexpectedly large for narrow-resonance refinement: len={}",
            grid.len()
        );
    }

    #[test]
    fn test_create_auxiliary_grid_adds_tail_points() {
        let data: Vec<f64> = (0..81).map(|i| 9.6 + i as f64 * 0.01).collect();
        let grid = create_auxiliary_grid(&data, &[10.0], &[0.01], 50.0, 10.0).unwrap();
        let expected_left = 10.0 - 1.5 * 0.01;
        let expected_right = 10.0 + 6.0 * 0.01;
        assert!(
            grid.iter().any(|&e| (e - expected_left).abs() < 1e-10),
            "missing expected left-tail refinement point"
        );
        assert!(
            grid.iter().any(|&e| (e - expected_right).abs() < 1e-10),
            "missing expected right-tail refinement point"
        );
    }
}
