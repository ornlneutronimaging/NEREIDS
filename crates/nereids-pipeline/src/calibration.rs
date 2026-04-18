//! Energy calibration for TOF neutron instruments.
//!
//! Finds the flight path length (L) and TOF delay (t₀) that best align
//! a measured transmission spectrum with the ENDF resonance model.
//!
//! The energy-TOF relationship is:
//!
//!   E = C · (L / (t − t₀))²
//!
//! where C = mₙ / 2 ≈ 5.2276e-9 [eV·s²/m²].
//!
//! When L or t₀ differ from the values assumed during data reduction,
//! resonance positions shift in the energy domain, causing catastrophic
//! chi² degradation (e.g. 436 → 2.7 for a 0.3% L correction on VENUS).

use nereids_core::constants::{EV_TO_JOULES, NEUTRON_MASS_KG};
use nereids_endf::resonance::ResonanceData;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

use crate::error::PipelineError;

/// Neutron mass constant: C = m_n / (2 · eV) ≈ 5.2276e-9 eV·s²/m².
///
/// E [eV] = C · (L [m] / t [s])²
///
/// Uses the CODATA 2018 values from `nereids_core::constants` so that
/// this calibration path, `EnergyScaleTransmissionModel`, and
/// `core::tof_to_energy` all agree to machine precision.
const NEUTRON_MASS_CONSTANT: f64 = 0.5 * NEUTRON_MASS_KG / EV_TO_JOULES;

/// Result of energy calibration.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Fitted flight path length in metres.
    pub flight_path_m: f64,
    /// Fitted TOF delay in microseconds.
    pub t0_us: f64,
    /// Fitted total areal density in atoms/barn.
    pub total_density: f64,
    /// Reduced chi-squared at the best (L, t₀, n) values.
    pub reduced_chi_squared: f64,
    /// Corrected energy grid (ascending, eV).
    pub energies_corrected: Vec<f64>,
}

/// Calibrate the energy axis of a TOF neutron measurement.
///
/// Given a measured 1D transmission spectrum and known sample composition
/// (e.g. natural Hf), finds the (L, t₀) that minimize chi² by aligning
/// the ENDF resonance positions with the measured dips.
///
/// # Arguments
///
/// * `energies_nominal` — Energy grid computed with assumed L (ascending, eV)
/// * `transmission` — Measured transmission values (same length)
/// * `uncertainty` — Per-bin uncertainty (same length)
/// * `isotopes` — ENDF resonance data for each isotope
/// * `abundances` — Natural abundance fractions (same length as isotopes, sum ≤ 1)
/// * `assumed_flight_path_m` — The L used to compute `energies_nominal`
/// * `temperature_k` — Sample temperature for Doppler broadening
/// * `resolution` — Optional instrument resolution function.  When provided,
///   the forward model includes Doppler + resolution broadening, producing
///   more accurate (L, t₀) fits.  Without resolution, fitted parameters
///   absorb the missing broadening and may be biased.
///
/// # Returns
///
/// [`CalibrationResult`] with the fitted (L, t₀, n_total) and corrected energies.
#[allow(clippy::too_many_arguments)]
pub fn calibrate_energy(
    energies_nominal: &[f64],
    transmission: &[f64],
    uncertainty: &[f64],
    isotopes: &[ResonanceData],
    abundances: &[f64],
    assumed_flight_path_m: f64,
    temperature_k: f64,
    resolution: Option<&InstrumentParams>,
) -> Result<CalibrationResult, PipelineError> {
    let n = energies_nominal.len();
    if n == 0 {
        return Err(PipelineError::InvalidParameter(
            "energies_nominal must not be empty".into(),
        ));
    }
    if transmission.len() != n || uncertainty.len() != n {
        return Err(PipelineError::InvalidParameter(format!(
            "transmission ({}) and uncertainty ({}) must match energies ({})",
            transmission.len(),
            uncertainty.len(),
            n,
        )));
    }
    if isotopes.len() != abundances.len() {
        return Err(PipelineError::InvalidParameter(format!(
            "isotopes ({}) must match abundances ({})",
            isotopes.len(),
            abundances.len(),
        )));
    }

    // Recover TOF from nominal energies: t = L_assumed · √(C / E)
    let tof_s: Vec<f64> = energies_nominal
        .iter()
        .map(|&e| assumed_flight_path_m * (NEUTRON_MASS_CONSTANT / e).sqrt())
        .collect();

    // Pre-filter valid bins (finite T, positive sigma)
    let valid: Vec<bool> = transmission
        .iter()
        .zip(uncertainty.iter())
        .map(|(&t, &s)| t.is_finite() && s.is_finite() && s > 0.0)
        .collect();

    // ── Phase 1: Coarse grid search over (L, t₀, n_total) ──────────
    // L: ±1% around assumed (0.5% steps = 5 points each side)
    // t₀: -5 to +10 µs (1 µs steps)
    // n_total: scanned at each (L, t₀) via golden section on [1e-5, 1e-2]

    let l_center = assumed_flight_path_m;
    let mut best_chi2 = f64::INFINITY;
    let mut best_l = l_center;
    let mut best_t0_us = 0.0f64;
    let mut best_n = 1e-4;

    // Coarse L: 0.2% steps, ±1.5%
    let l_steps: Vec<f64> = (-15..=15)
        .map(|i| l_center * (1.0 + i as f64 * 0.001))
        .collect();
    // Coarse t₀: 1 µs steps, -5 to +10 µs
    let t0_steps: Vec<f64> = (-5..=10).map(|i| i as f64).collect();

    for &l in &l_steps {
        for &t0 in &t0_steps {
            let t0_s = t0 * 1e-6;
            // Correct energies
            let e_corr: Vec<f64> = tof_s
                .iter()
                .map(|&t| {
                    let t_corr = t - t0_s;
                    if t_corr <= 0.0 {
                        f64::NAN
                    } else {
                        NEUTRON_MASS_CONSTANT * (l / t_corr).powi(2)
                    }
                })
                .collect();

            // Skip if any NaN
            if e_corr.iter().any(|e| !e.is_finite() || *e <= 0.0) {
                continue;
            }

            // Quick n_total scan: 5 values on log scale
            for &n_total in &[5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4] {
                let chi2 = compute_chi2(
                    &e_corr,
                    transmission,
                    uncertainty,
                    isotopes,
                    abundances,
                    n_total,
                    temperature_k,
                    &valid,
                    resolution,
                );
                if chi2 < best_chi2 {
                    best_chi2 = chi2;
                    best_l = l;
                    best_t0_us = t0;
                    best_n = n_total;
                }
            }
        }
    }

    // ── Phase 2: Fine grid search around coarse best ────────────────
    // L: ±0.05%, 0.01% steps
    // t₀: ±2 µs, 0.25 µs steps
    // n: ±50%, 5% steps

    let l_fine: Vec<f64> = (-5..=5)
        .map(|i| best_l * (1.0 + i as f64 * 0.0001))
        .collect();
    let t0_fine: Vec<f64> = (-8..=8).map(|i| best_t0_us + i as f64 * 0.25).collect();
    let n_fine: Vec<f64> = (-10..=10)
        .map(|i| best_n * (1.0 + i as f64 * 0.05))
        .filter(|&n| n > 0.0)
        .collect();

    for &l in &l_fine {
        for &t0 in &t0_fine {
            let t0_s = t0 * 1e-6;
            let e_corr: Vec<f64> = tof_s
                .iter()
                .map(|&t| {
                    let t_corr = t - t0_s;
                    if t_corr <= 0.0 {
                        f64::NAN
                    } else {
                        NEUTRON_MASS_CONSTANT * (l / t_corr).powi(2)
                    }
                })
                .collect();
            if e_corr.iter().any(|e| !e.is_finite() || *e <= 0.0) {
                continue;
            }
            for &n_total in &n_fine {
                let chi2 = compute_chi2(
                    &e_corr,
                    transmission,
                    uncertainty,
                    isotopes,
                    abundances,
                    n_total,
                    temperature_k,
                    &valid,
                    resolution,
                );
                if chi2 < best_chi2 {
                    best_chi2 = chi2;
                    best_l = l;
                    best_t0_us = t0;
                    best_n = n_total;
                }
            }
        }
    }

    // ── Phase 3: Ultra-fine refinement ──────────────────────────────
    // L: ±0.005%, 0.001% steps
    // t₀: ±0.5 µs, 0.05 µs steps
    // n: ±10%, 1% steps

    let l_ultra: Vec<f64> = (-5..=5)
        .map(|i| best_l * (1.0 + i as f64 * 0.00001))
        .collect();
    let t0_ultra: Vec<f64> = (-10..=10).map(|i| best_t0_us + i as f64 * 0.05).collect();
    let n_ultra: Vec<f64> = (-10..=10)
        .map(|i| best_n * (1.0 + i as f64 * 0.01))
        .filter(|&n| n > 0.0)
        .collect();

    for &l in &l_ultra {
        for &t0 in &t0_ultra {
            let t0_s = t0 * 1e-6;
            let e_corr: Vec<f64> = tof_s
                .iter()
                .map(|&t| {
                    let t_corr = t - t0_s;
                    if t_corr <= 0.0 {
                        f64::NAN
                    } else {
                        NEUTRON_MASS_CONSTANT * (l / t_corr).powi(2)
                    }
                })
                .collect();
            if e_corr.iter().any(|e| !e.is_finite() || *e <= 0.0) {
                continue;
            }
            for &n_total in &n_ultra {
                let chi2 = compute_chi2(
                    &e_corr,
                    transmission,
                    uncertainty,
                    isotopes,
                    abundances,
                    n_total,
                    temperature_k,
                    &valid,
                    resolution,
                );
                if chi2 < best_chi2 {
                    best_chi2 = chi2;
                    best_l = l;
                    best_t0_us = t0;
                    best_n = n_total;
                }
            }
        }
    }

    // Compute corrected energy grid at the best parameters
    let t0_best_s = best_t0_us * 1e-6;
    let energies_corrected: Vec<f64> = tof_s
        .iter()
        .map(|&t| NEUTRON_MASS_CONSTANT * (best_l / (t - t0_best_s)).powi(2))
        .collect();

    // Final chi2r (reduced)
    let n_valid = valid.iter().filter(|&&v| v).count();
    let dof = if n_valid > 3 { n_valid - 3 } else { 1 }; // 3 fitted params
    let chi2r = best_chi2 / dof as f64;

    Ok(CalibrationResult {
        flight_path_m: best_l,
        t0_us: best_t0_us,
        total_density: best_n,
        reduced_chi_squared: chi2r,
        energies_corrected,
    })
}

/// Compute total chi² for a given (E_corrected, n_total) against measured data.
#[allow(clippy::too_many_arguments)]
fn compute_chi2(
    energies: &[f64],
    transmission: &[f64],
    uncertainty: &[f64],
    isotopes: &[ResonanceData],
    abundances: &[f64],
    n_total: f64,
    temperature_k: f64,
    valid: &[bool],
    resolution: Option<&InstrumentParams>,
) -> f64 {
    // Build (isotope, density) pairs
    let pairs: Vec<(ResonanceData, f64)> = isotopes
        .iter()
        .zip(abundances.iter())
        .map(|(iso, &abd)| (iso.clone(), abd * n_total))
        .collect();

    let sample = match SampleParams::new(temperature_k, pairs) {
        Ok(s) => s,
        Err(_) => return f64::INFINITY,
    };

    // P-5: Include resolution broadening when available.
    // Without it, fitted L and t₀ absorb the missing broadening bias.
    let model = match transmission::forward_model(energies, &sample, resolution) {
        Ok(m) => m,
        Err(_) => return f64::INFINITY,
    };

    // Chi²
    let mut chi2 = 0.0;
    for (i, (&t_data, &t_model)) in transmission.iter().zip(model.iter()).enumerate() {
        if !valid[i] {
            continue;
        }
        let residual = (t_data - t_model) / uncertainty[i];
        chi2 += residual * residual;
    }
    chi2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires network: downloads Hf-178 ENDF from IAEA"]
    fn test_calibrate_round_trip() {
        // Generate synthetic data with known L and t0, then recover them.
        let true_l = 25.08;
        let true_t0_us = 1.0;
        let true_n = 1.5e-4;
        let temperature_k = 293.6;

        // Use Hf-178 as a single-isotope reference (strong resonance at ~7.8 eV)
        let iso = {
            use nereids_core::types::Isotope;
            use nereids_endf::parser::parse_endf_file2;
            use nereids_endf::retrieval::{EndfLibrary, EndfRetriever, mat_number};

            let isotope = Isotope::new(72, 178).unwrap();
            let mat = mat_number(&isotope).expect("No MAT number for Hf-178");
            let retriever = EndfRetriever::new();
            let (_path, contents) = retriever
                .get_endf_file(&isotope, EndfLibrary::EndfB8_0, mat)
                .expect("Failed to retrieve Hf-178");
            parse_endf_file2(&contents).expect("Failed to parse Hf-178")
        };

        // Create nominal energy grid (as if L=25.0, t0=0)
        let assumed_l = 25.0;
        let e_nominal: Vec<f64> = (0..500)
            .map(|i| 5.0 + i as f64 * 0.4) // 5 to 205 eV
            .collect();

        // Recover TOF from nominal E at assumed L
        let tof_s: Vec<f64> = e_nominal
            .iter()
            .map(|&e| assumed_l * (NEUTRON_MASS_CONSTANT / e).sqrt())
            .collect();

        // Compute "true" energies using true L and t0
        let true_t0_s = true_t0_us * 1e-6;
        let e_true: Vec<f64> = tof_s
            .iter()
            .map(|&t| NEUTRON_MASS_CONSTANT * (true_l / (t - true_t0_s)).powi(2))
            .collect();

        // Generate synthetic transmission at true energies
        let sample = SampleParams::new(temperature_k, vec![(iso.clone(), true_n)])
            .expect("SampleParams creation failed");
        let t_model =
            transmission::forward_model(&e_true, &sample, None).expect("forward_model failed");

        // Add tiny noise (sigma = 0.01, no actual noise — just for chi2 weighting)
        let sigma = vec![0.01; e_nominal.len()];

        // Calibrate (no resolution — matches synthetic data generated without resolution)
        let result = calibrate_energy(
            &e_nominal,
            &t_model,
            &sigma,
            &[iso],
            &[1.0], // single isotope, abundance = 1.0
            assumed_l,
            temperature_k,
            None,
        )
        .expect("Calibration failed");

        // Check recovery
        assert!(
            (result.flight_path_m - true_l).abs() < 0.05,
            "L: got {}, expected {}",
            result.flight_path_m,
            true_l,
        );
        assert!(
            (result.t0_us - true_t0_us).abs() < 1.0,
            "t0: got {}, expected {}",
            result.t0_us,
            true_t0_us,
        );
        assert!(
            (result.total_density - true_n).abs() / true_n < 0.2,
            "n: got {}, expected {}",
            result.total_density,
            true_n,
        );
        assert!(
            result.reduced_chi_squared < 1.0,
            "chi2r should be < 1 for synthetic data, got {}",
            result.reduced_chi_squared,
        );
    }
}
