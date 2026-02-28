//! Trace-detectability analysis for neutron resonance imaging.
//!
//! Answers the pre-experiment question: *"Can I detect X ppm of isotope B
//! in a matrix of isotope A across a given energy window?"*
//!
//! ## Core concept
//!
//! For a given matrix + trace isotope pair, compute the **peak spectral SNR**
//! as a function of trace concentration:
//!
//!   SNR_peak(c) = max_E |ΔT(E, c)| / σ_noise
//!
//! where ΔT is the *signed* transmission difference:
//!
//!   ΔT(E, c) = T(E, n_matrix, 0) − T(E, n_matrix, n_trace = c·n_matrix)
//!
//! and σ_noise ≈ 1/√I₀ (off-resonance Poisson approximation).
//!
//! The stored `delta_t_spectrum` and all derived metrics (peak_delta_t_per_ppm,
//! peak_snr) use **|ΔT|**, discarding the sign.
//!
//! ## Reference
//!
//! Motivated by the observation that Fe-56 + Mn-55 have no resolved resonances
//! in 1–50 eV, while W-182 + Hf-178 give strong contrast in the same window.
//! VENUS can resolve up to ~1 keV, so many more pairs become accessible at
//! higher energies (e.g., Mn-55 resonance at ~337 eV).

use nereids_core::elements;
use nereids_endf::resonance::ResonanceData;
use nereids_physics::resolution::ResolutionFunction;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};
use rayon::prelude::*;

/// Result of a trace-detectability analysis for a single matrix+trace pair.
#[derive(Debug, Clone)]
pub struct TraceDetectabilityReport {
    /// Peak |ΔT| per ppm concentration at the most sensitive energy.
    ///
    /// Linearized sensitivity metric: `peak_delta_t / trace_ppm`.
    /// Accurate in the dilute-trace limit; at high concentrations the
    /// actual per-ppm sensitivity is smaller due to Beer-Lambert
    /// saturation (T = exp(−n·σ) is sub-linear in n).
    pub peak_delta_t_per_ppm: f64,
    /// Energy at which peak contrast occurs (eV).
    pub peak_energy_ev: f64,
    /// Estimated peak SNR at the given concentration and I₀.
    pub peak_snr: f64,
    /// Whether the combination is detectable (SNR > threshold).
    pub detectable: bool,
    /// Energy-resolved |ΔT| spectrum for the given concentration.
    pub delta_t_spectrum: Vec<f64>,
    /// Energies used (eV).
    pub energies: Vec<f64>,
}

/// Configuration for a trace-detectability analysis.
pub struct TraceDetectabilityConfig<'a> {
    /// Resonance data for the matrix isotope.
    pub matrix: &'a ResonanceData,
    /// Matrix areal density in atoms/barn.
    pub matrix_density: f64,
    /// Energy grid in eV (sorted ascending).
    pub energies: &'a [f64],
    /// Expected counts per bin (for Poisson noise estimate).
    pub i0: f64,
    /// Sample temperature in Kelvin.
    pub temperature_k: f64,
    /// Optional resolution broadening function.
    pub resolution: Option<&'a ResolutionFunction>,
    /// Detection threshold (3.0 = standard 3σ detection limit).
    pub snr_threshold: f64,
}

/// Build an `InstrumentParams` from the config's optional resolution function.
fn build_instrument(config: &TraceDetectabilityConfig) -> Option<InstrumentParams> {
    config.resolution.map(|r| InstrumentParams {
        resolution: r.clone(),
    })
}

/// Compute the matrix-only baseline transmission.
///
/// NOTE: `SampleParams` owns its `ResonanceData`, so we clone here.
/// For typical ENDF isotopes the data is small (a few KB) and the clone
/// cost is negligible compared to the cross-section evaluation.  If
/// `SampleParams` gains `Arc` support in the future, these clones can
/// be eliminated.
fn matrix_baseline(
    config: &TraceDetectabilityConfig,
    instrument: Option<&InstrumentParams>,
) -> Vec<f64> {
    let sample_matrix = SampleParams {
        temperature_k: config.temperature_k,
        isotopes: vec![(config.matrix.clone(), config.matrix_density)],
    };
    // forward_model only fails for unsorted energies, which is a
    // configuration bug caught before the analysis runs.
    transmission::forward_model(config.energies, &sample_matrix, instrument)
        .expect("matrix_baseline: energy grid must be sorted ascending")
}

/// Build a report from a precomputed matrix-only baseline and a trace isotope.
fn report_from_baseline(
    config: &TraceDetectabilityConfig,
    instrument: Option<&InstrumentParams>,
    t_matrix: &[f64],
    trace: &ResonanceData,
    trace_ppm: f64,
) -> TraceDetectabilityReport {
    // T_combined: transmission through matrix + trace
    let trace_density = trace_ppm * 1e-6 * config.matrix_density;
    let sample_combined = SampleParams {
        temperature_k: config.temperature_k,
        isotopes: vec![
            (config.matrix.clone(), config.matrix_density),
            (trace.clone(), trace_density),
        ],
    };
    let t_combined = transmission::forward_model(config.energies, &sample_combined, instrument)
        .expect("report_from_baseline: energy grid must be sorted ascending");

    // |ΔT| spectrum — absolute difference, sign discarded (see module docs).
    let delta_t_spectrum: Vec<f64> = t_matrix
        .iter()
        .zip(t_combined.iter())
        .map(|(&tm, &tc)| (tm - tc).abs())
        .collect();

    // Find peak |ΔT| and corresponding energy
    let (peak_idx, &peak_delta_t) = delta_t_spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    let peak_energy_ev = if config.energies.is_empty() {
        0.0
    } else {
        config.energies[peak_idx]
    };

    // SNR calculation: σ_noise ≈ 1/√I₀
    let sigma_noise = if config.i0 > 0.0 {
        1.0 / config.i0.sqrt()
    } else {
        f64::INFINITY
    };
    let peak_snr = peak_delta_t / sigma_noise;

    // Per-ppm normalisation
    let peak_delta_t_per_ppm = if trace_ppm > 0.0 {
        peak_delta_t / trace_ppm
    } else {
        0.0
    };

    TraceDetectabilityReport {
        peak_delta_t_per_ppm,
        peak_energy_ev,
        peak_snr,
        detectable: peak_snr > config.snr_threshold,
        delta_t_spectrum,
        energies: config.energies.to_vec(),
    }
}

/// Compute trace-detectability for a single matrix+trace isotope pair.
///
/// Two `forward_model` calls (with/without trace) + argmax over the
/// difference spectrum. Resolution broadening is included when provided,
/// so the SNR reflects realistic peak spreading at VENUS.
///
/// # Arguments
/// * `config`    — Shared analysis parameters (matrix, energies, I₀, etc.).
/// * `trace`     — Resonance data for the trace isotope.
/// * `trace_ppm` — Trace concentration in ppm by atom.
///
/// # Preconditions
/// * `config.energies` must be non-empty and sorted ascending.
/// * `config.i0` must be positive (used as `1/√I₀` for noise estimate).
/// * `config.matrix_density` must be positive.
/// * `config.snr_threshold` must be non-negative.
/// * `trace_ppm` must be non-negative.
///
/// The Python bindings validate all of these; Rust callers are responsible
/// for ensuring valid inputs.
///
/// # Returns
/// [`TraceDetectabilityReport`] with peak SNR, peak energy, detectability
/// flag, and the full |ΔT| spectrum.
pub fn trace_detectability(
    config: &TraceDetectabilityConfig,
    trace: &ResonanceData,
    trace_ppm: f64,
) -> TraceDetectabilityReport {
    let instrument = build_instrument(config);
    let t_matrix = matrix_baseline(config, instrument.as_ref());
    report_from_baseline(config, instrument.as_ref(), &t_matrix, trace, trace_ppm)
}

/// Survey multiple trace candidates against a single matrix.
///
/// The matrix-only baseline transmission and instrument resolution are
/// computed once and reused for all candidates. Each candidate is then
/// evaluated in parallel with rayon.
///
/// # Preconditions
/// Same as [`trace_detectability`], plus `trace_candidates` must be
/// non-empty.
///
/// # Returns
/// Vec of (isotope_name, report) sorted by `peak_snr` descending.
pub fn trace_detectability_survey(
    config: &TraceDetectabilityConfig,
    trace_candidates: &[ResonanceData],
    trace_ppm: f64,
) -> Vec<(String, TraceDetectabilityReport)> {
    // Build instrument and matrix baseline once for all candidates.
    let instrument = build_instrument(config);
    let t_matrix = matrix_baseline(config, instrument.as_ref());

    let mut results: Vec<(String, TraceDetectabilityReport)> = trace_candidates
        .par_iter()
        .map(|trace| {
            let name = elements::element_symbol(trace.isotope.z())
                .map(|sym| format!("{}-{}", sym, trace.isotope.a()))
                .unwrap_or_else(|| format!("Z{}-{}", trace.isotope.z(), trace.isotope.a()));

            let report =
                report_from_baseline(config, instrument.as_ref(), &t_matrix, trace, trace_ppm);

            (name, report)
        })
        .collect();

    // Sort by peak_snr descending
    results.sort_by(|(_, a), (_, b)| {
        b.peak_snr
            .partial_cmp(&a.peak_snr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceFormalism, ResonanceRange};
    use nereids_endf::retrieval::{EndfLibrary, EndfRetriever, mat_number};

    /// Helper: load real ENDF data for a given (Z, A).
    fn load_endf_data(z: u32, a: u32) -> ResonanceData {
        use nereids_core::types::Isotope;
        use nereids_endf::parser::parse_endf_file2;

        let isotope = Isotope::new(z, a).unwrap();
        let mat = mat_number(&isotope).unwrap_or_else(|| {
            panic!("No MAT number for Z={z} A={a}");
        });
        let retriever = EndfRetriever::new();
        let (_path, contents) = retriever
            .get_endf_file(&isotope, EndfLibrary::EndfB8_1, mat)
            .unwrap_or_else(|e| panic!("Failed to load ENDF for Z={z} A={a}: {e}"));
        parse_endf_file2(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse ENDF for Z={z} A={a}: {e}"))
    }

    /// Fe-56 + Mn-55 in 1–50 eV: no resolved resonances → NOT detectable at 2000 ppm.
    #[test]
    #[ignore = "requires network: downloads ENDF data from IAEA"]
    fn test_fe56_mn55_narrow_window_not_detectable() {
        let fe56 = load_endf_data(26, 56);
        let mn55 = load_endf_data(25, 55);

        let energies: Vec<f64> = (0..5000)
            .map(|i| 1.0 + (i as f64) * 49.0 / 4999.0)
            .collect();

        let config = TraceDetectabilityConfig {
            matrix: &fe56,
            matrix_density: 8.47e-3, // atoms/barn (1 mm Fe)
            energies: &energies,
            i0: 1000.0,
            temperature_k: 293.6,
            resolution: None,
            snr_threshold: 3.0,
        };

        let report = trace_detectability(&config, &mn55, 2000.0);

        assert!(
            !report.detectable,
            "Fe-56 + Mn-55 should NOT be detectable in 1–50 eV at 2000 ppm (SNR={:.4})",
            report.peak_snr,
        );
    }

    /// W-182 + Hf-178 in 1–5 eV: Hf-178 resonance at ~7.8 eV is outside window → NOT detectable.
    ///
    /// Demonstrates that a narrow window can miss resonances.
    /// Compare with `test_w182_hf178_wide_detectable` below.
    #[test]
    #[ignore = "requires network: downloads ENDF data from IAEA"]
    fn test_w182_hf178_narrow_not_detectable() {
        let w182 = load_endf_data(74, 182);
        let hf178 = load_endf_data(72, 178);

        let energies: Vec<f64> = (0..5000).map(|i| 1.0 + (i as f64) * 4.0 / 4999.0).collect();

        let config = TraceDetectabilityConfig {
            matrix: &w182,
            matrix_density: 6.38e-3,
            energies: &energies,
            i0: 1000.0,
            temperature_k: 293.6,
            resolution: None,
            snr_threshold: 3.0,
        };

        let report = trace_detectability(&config, &hf178, 500.0);

        assert!(
            !report.detectable,
            "W-182 + Hf-178 should NOT be detectable in 1–5 eV at 500 ppm (SNR={:.4})",
            report.peak_snr,
        );
    }

    /// W-182 + Hf-178 in 1–50 eV: Hf-178 resonance at ~7.8 eV is now in range → detectable.
    ///
    /// The wider window includes the Hf-178 resonance, enabling detection that was
    /// impossible in the 1–5 eV window. This is the core VENUS use case.
    #[test]
    #[ignore = "requires network: downloads ENDF data from IAEA"]
    fn test_w182_hf178_wide_detectable() {
        let w182 = load_endf_data(74, 182);
        let hf178 = load_endf_data(72, 178);

        let energies: Vec<f64> = (0..5000)
            .map(|i| 1.0 + (i as f64) * 49.0 / 4999.0)
            .collect();

        let config = TraceDetectabilityConfig {
            matrix: &w182,
            matrix_density: 6.38e-3, // atoms/barn (1 mm W)
            energies: &energies,
            i0: 1000.0,
            temperature_k: 293.6,
            resolution: None,
            snr_threshold: 3.0,
        };

        let report = trace_detectability(&config, &hf178, 500.0);

        assert!(
            report.detectable,
            "W-182 + Hf-178 should be detectable in 1–50 eV at 500 ppm (SNR={:.4})",
            report.peak_snr,
        );
        // Hf-178 has a strong resonance near 7.8 eV
        assert!(
            report.peak_energy_ev > 5.0 && report.peak_energy_ev < 15.0,
            "Peak energy should be near the Hf-178 resonance (~7.8 eV), got {:.1} eV",
            report.peak_energy_ev,
        );
    }

    #[test]
    #[ignore = "requires network: downloads ENDF data from IAEA"]
    fn test_survey_returns_sorted_by_snr() {
        let w182 = load_endf_data(74, 182);
        let hf178 = load_endf_data(72, 178);
        let fe56 = load_endf_data(26, 56);

        let energies: Vec<f64> = (0..5000)
            .map(|i| 1.0 + (i as f64) * 49.0 / 4999.0)
            .collect();

        let config = TraceDetectabilityConfig {
            matrix: &w182,
            matrix_density: 6.38e-3,
            energies: &energies,
            i0: 1000.0,
            temperature_k: 293.6,
            resolution: None,
            snr_threshold: 3.0,
        };

        // Hf-178 has strong resonances in 1-50 eV; Fe-56 does not
        let results = trace_detectability_survey(&config, &[fe56, hf178], 500.0);

        assert_eq!(results.len(), 2);
        // Results should be sorted by peak_snr descending
        assert!(
            results[0].1.peak_snr >= results[1].1.peak_snr,
            "Survey results should be sorted by SNR descending: {} >= {}",
            results[0].1.peak_snr,
            results[1].1.peak_snr,
        );
        // Hf-178 should be first (higher SNR)
        assert_eq!(
            results[0].0, "Hf-178",
            "Hf-178 should rank highest, got '{}'",
            results[0].0,
        );
    }

    // --- Offline synthetic tests (no network required) ---

    /// Build a minimal single-resonance isotope for offline testing.
    fn synthetic_isotope(z: u32, a: u32, res_energy: f64, gn: f64, gg: f64) -> ResonanceData {
        ResonanceData {
            isotope: nereids_core::types::Isotope::new(z, a).unwrap(),
            za: z * 1000 + a,
            awr: a as f64 - 0.009, // approximate
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin: 0.0,
                scattering_radius: 6.0,
                naps: 0,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: a as f64 - 0.009,
                    apl: 0.0,
                    qx: 0.0,
                    lrx: 0,
                    resonances: vec![Resonance {
                        energy: res_energy,
                        j: 0.5,
                        gn,
                        gg,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                urr: None,
                ap_table: None,
            }],
        }
    }

    /// Peak selection and SNR computation with synthetic data (offline).
    #[test]
    fn test_synthetic_peak_snr() {
        // Matrix: weak resonance at 20 eV
        let matrix = synthetic_isotope(74, 182, 20.0, 1e-3, 0.05);
        // Trace: strong resonance at 8 eV (should dominate the |ΔT| spectrum)
        let trace = synthetic_isotope(72, 178, 8.0, 0.5, 0.05);

        let energies: Vec<f64> = (0..500).map(|i| 1.0 + (i as f64) * 49.0 / 499.0).collect();

        let config = TraceDetectabilityConfig {
            matrix: &matrix,
            matrix_density: 6e-3,
            energies: &energies,
            i0: 1000.0,
            temperature_k: 293.6,
            resolution: None,
            snr_threshold: 3.0,
        };

        let report = trace_detectability(&config, &trace, 1000.0);

        // Basic sanity: spectrum has correct length
        assert_eq!(report.delta_t_spectrum.len(), energies.len());
        assert_eq!(report.energies.len(), energies.len());

        // Peak should be near the trace resonance at 8 eV (not the matrix at 20 eV)
        assert!(
            report.peak_energy_ev > 5.0 && report.peak_energy_ev < 12.0,
            "Peak energy should be near 8 eV, got {:.1} eV",
            report.peak_energy_ev,
        );

        // With a strong trace resonance at 1000 ppm, SNR should be well above 3
        assert!(
            report.peak_snr > 3.0,
            "Expected detectable (SNR > 3), got SNR={:.2}",
            report.peak_snr,
        );
        assert!(report.detectable);

        // peak_delta_t_per_ppm should be positive and consistent
        assert!(report.peak_delta_t_per_ppm > 0.0);
    }

    /// Survey sorting with synthetic isotopes (offline).
    #[test]
    fn test_synthetic_survey_sorting() {
        let matrix = synthetic_isotope(74, 182, 20.0, 1e-3, 0.05);
        // Strong trace (big resonance at 8 eV)
        let strong_trace = synthetic_isotope(72, 178, 8.0, 0.5, 0.05);
        // Weak trace (tiny resonance at 30 eV)
        let weak_trace = synthetic_isotope(26, 56, 30.0, 1e-5, 0.01);

        let energies: Vec<f64> = (0..500).map(|i| 1.0 + (i as f64) * 49.0 / 499.0).collect();

        let config = TraceDetectabilityConfig {
            matrix: &matrix,
            matrix_density: 6e-3,
            energies: &energies,
            i0: 1000.0,
            temperature_k: 293.6,
            resolution: None,
            snr_threshold: 3.0,
        };

        // Pass weak first — survey should reorder by SNR descending
        let results = trace_detectability_survey(&config, &[weak_trace, strong_trace], 500.0);

        assert_eq!(results.len(), 2);
        assert!(
            results[0].1.peak_snr >= results[1].1.peak_snr,
            "Survey should sort by SNR descending: {:.2} >= {:.2}",
            results[0].1.peak_snr,
            results[1].1.peak_snr,
        );
        // The strong isotope (Z=72) should rank first
        assert!(
            results[0].0.starts_with("Hf"),
            "Strong trace (Hf) should rank first, got '{}'",
            results[0].0,
        );
    }
}
