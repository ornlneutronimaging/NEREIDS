//! Fit a sample together with the calibrant that measured its resolution.
//!
//! The ordinary route calibrates the resolution on a known sample, pins the
//! result, and fits the unknown one. That reports the temperature as more
//! certain than it is, because resolution width and temperature broaden the
//! line the same way and pinning discards the part of the uncertainty that
//! belongs to their degeneracy.
//!
//! This route fits both spectra at once against one shared resolution, so the
//! calibrant's residuals are in the same objective as the sample's and the
//! reported temperature uncertainty already contains what the calibrant
//! failed to pin down. See [`nereids_fitting::joint_resolution`] for why a
//! summarized calibration cannot stand in for those residuals.

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::joint_resolution::JointResolutionModel;
use nereids_fitting::lm::{self, LmConfig};
use nereids_fitting::parameters::{FitParameter, ParameterSet};

use crate::error::PipelineError;
use crate::pipeline::TEMPERATURE_BOUNDS_K;

/// The known spectrum: what it is made of, at what temperature, and what was
/// measured.
///
/// Its densities and temperature are fixed during the fit. That is what makes
/// it a calibrant — the resolution is the only thing it is being asked about.
pub struct CalibrantSpectrum {
    /// Energy grid (eV), ascending.
    pub energies: Vec<f64>,
    /// Measured transmission on that grid.
    pub transmission: Vec<f64>,
    /// One-sigma uncertainty per point.
    pub uncertainty: Vec<f64>,
    /// Known composition: each isotope with its known areal density.
    pub isotopes: Vec<(ResonanceData, f64)>,
    /// Known temperature (K).
    pub temperature_k: f64,
}

/// What a joint fit reports.
///
/// The resolution is an output here rather than an input, so the fitted
/// widths come back alongside the sample's parameters.
#[derive(Debug, Clone)]
pub struct JointFitResult {
    /// Fitted areal densities (atoms/barn), one per sample isotope.
    pub densities: Vec<f64>,
    /// One-sigma uncertainty on each density, `None` when the covariance was
    /// not available.
    pub density_uncertainties: Option<Vec<f64>>,
    /// Fitted temperature (K), `None` when it was held fixed.
    pub temperature_k: Option<f64>,
    /// One-sigma uncertainty on the fitted temperature.
    ///
    /// Unlike the pinned route's, this includes the temperature's degeneracy
    /// with the resolution, because the calibrant that constrains the
    /// resolution is in the same objective.
    pub temperature_k_unc: Option<f64>,
    /// Fitted Gaussian timing width (µs), the W-parameter in `exp(-x²/W²)`.
    pub delta_t_us: f64,
    /// Fitted Gaussian flight-path width (m), same convention.
    pub delta_l_m: f64,
    /// Reduced chi-squared over both spectra together.
    pub reduced_chi_squared: f64,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Iterations taken.
    pub iterations: usize,
}

/// Parameter order in the shared vector: densities, then temperature when
/// fitted, then the two resolution widths.
const fn resolution_indices(n_density: usize, fit_temperature: bool) -> (usize, usize) {
    let after = n_density + if fit_temperature { 1 } else { 0 };
    (after, after + 1)
}

/// Fit `transmission` and the calibrant against one shared Gaussian
/// resolution.
///
/// `initial_densities` seeds the sample's densities and `temperature_k` its
/// temperature; `fit_temperature` decides whether that temperature is free.
/// `delta_t_init` / `delta_l_init` seed the shared resolution — the values a
/// standalone calibration returned are the natural starting point.
///
/// # Errors
/// [`PipelineError::ShapeMismatch`] when a spectrum's arrays disagree in
/// length, [`PipelineError::InvalidParameter`] when a grid is empty, an
/// uncertainty is not positive, a density on either arm is not finite and
/// positive, or a temperature falls outside the box every fit path shares,
/// and [`PipelineError::Fitting`] when the optimizer cannot run.
#[allow(clippy::too_many_arguments)]
pub fn fit_with_calibrant(
    transmission: &[f64],
    uncertainty: &[f64],
    energies: &[f64],
    isotopes: &[ResonanceData],
    initial_densities: &[f64],
    temperature_k: f64,
    fit_temperature: bool,
    calibrant: &CalibrantSpectrum,
    flight_path_m: f64,
    delta_t_init: f64,
    delta_l_init: f64,
) -> Result<JointFitResult, PipelineError> {
    check_spectrum("sample", energies, transmission, uncertainty)?;
    check_spectrum(
        "calibrant",
        &calibrant.energies,
        &calibrant.transmission,
        &calibrant.uncertainty,
    )?;
    if isotopes.is_empty() {
        return Err(PipelineError::InvalidParameter(
            "the sample needs at least one isotope".into(),
        ));
    }
    if initial_densities.len() != isotopes.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "{} initial densities for {} sample isotopes",
            initial_densities.len(),
            isotopes.len(),
        )));
    }
    if calibrant.isotopes.is_empty() {
        return Err(PipelineError::InvalidParameter(
            "the calibrant needs at least one isotope".into(),
        ));
    }
    check_densities("sample", initial_densities.iter().copied())?;
    // A calibrant arm whose densities are not positive is transparent: the
    // forward model skips a non-positive thickness, so the arm carries no
    // resonance and constrains no resolution, and the fit would return a
    // number that came only from the sample.
    check_densities("calibrant", calibrant.isotopes.iter().map(|(_, n)| *n))?;
    for (label, value) in [
        ("flight_path_m", flight_path_m),
        ("delta_t_init", delta_t_init),
        ("delta_l_init", delta_l_init),
        ("temperature_k", temperature_k),
        ("the calibrant temperature", calibrant.temperature_k),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PipelineError::InvalidParameter(format!(
                "{label} must be finite and non-negative, got {value}"
            )));
        }
    }
    if flight_path_m <= 0.0 {
        return Err(PipelineError::InvalidParameter(
            "flight_path_m must be positive".into(),
        ));
    }
    let (t_lo, t_hi) = TEMPERATURE_BOUNDS_K;
    for (label, value) in [
        ("temperature_k", temperature_k),
        ("the calibrant temperature", calibrant.temperature_k),
    ] {
        if !(t_lo..=t_hi).contains(&value) {
            return Err(PipelineError::InvalidParameter(format!(
                "{label} must lie in [{t_lo}, {t_hi}] K, got {value}"
            )));
        }
    }

    let (delta_t_index, delta_l_index) = resolution_indices(isotopes.len(), fit_temperature);
    let temperature_index = fit_temperature.then_some(isotopes.len());
    let (calibrant_data, calibrant_densities): (Vec<_>, Vec<_>) =
        calibrant.isotopes.iter().cloned().unzip();

    let model = JointResolutionModel::new(
        energies.to_vec(),
        isotopes.to_vec(),
        (0..isotopes.len()).collect(),
        temperature_k,
        temperature_index,
        calibrant.energies.clone(),
        calibrant_data,
        calibrant_densities,
        calibrant.temperature_k,
        flight_path_m,
        delta_t_index,
        delta_l_index,
    )
    .map_err(PipelineError::Fitting)?;

    let mut values: Vec<FitParameter> = initial_densities
        .iter()
        .enumerate()
        .map(|(i, &n)| FitParameter::non_negative(format!("density_{i}"), n))
        .collect();
    if fit_temperature {
        values.push(FitParameter {
            name: "temperature_k".into(),
            value: temperature_k,
            lower: TEMPERATURE_BOUNDS_K.0,
            upper: TEMPERATURE_BOUNDS_K.1,
            fixed: false,
        });
    }
    values.push(FitParameter::non_negative("delta_t_us", delta_t_init));
    values.push(FitParameter::non_negative("delta_l_m", delta_l_init));
    let mut params = ParameterSet::new(values);

    // The two spectra are concatenated in the order the model predicts them.
    let mut data = transmission.to_vec();
    data.extend_from_slice(&calibrant.transmission);
    let mut sigma = uncertainty.to_vec();
    sigma.extend_from_slice(&calibrant.uncertainty);

    let result = lm::levenberg_marquardt(
        &model,
        &data,
        &sigma,
        &mut params,
        &LmConfig {
            compute_covariance: true,
            ..Default::default()
        },
    )
    .map_err(PipelineError::Fitting)?;

    // `uncertainties` is indexed by FREE parameter; every parameter here is
    // free, so the two orders coincide.
    let sigma_of = |i: usize| {
        result
            .uncertainties
            .as_ref()
            .and_then(|u| u.get(i).copied())
    };
    Ok(JointFitResult {
        densities: result.params[..isotopes.len()].to_vec(),
        density_uncertainties: result
            .uncertainties
            .as_ref()
            .map(|u| u[..isotopes.len()].to_vec()),
        temperature_k: temperature_index.map(|i| result.params[i]),
        temperature_k_unc: temperature_index.and_then(sigma_of),
        delta_t_us: result.params[delta_t_index],
        delta_l_m: result.params[delta_l_index],
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
    })
}

fn check_densities(label: &str, densities: impl Iterator<Item = f64>) -> Result<(), PipelineError> {
    for (i, n) in densities.enumerate() {
        if !n.is_finite() || n <= 0.0 {
            return Err(PipelineError::InvalidParameter(format!(
                "{label} density {i} is {n}; every areal density must be finite \
                 and positive"
            )));
        }
    }
    Ok(())
}

fn check_spectrum(
    label: &str,
    energies: &[f64],
    transmission: &[f64],
    uncertainty: &[f64],
) -> Result<(), PipelineError> {
    if energies.is_empty() {
        return Err(PipelineError::InvalidParameter(format!(
            "the {label} energy grid is empty"
        )));
    }
    if transmission.len() != energies.len() || uncertainty.len() != energies.len() {
        return Err(PipelineError::ShapeMismatch(format!(
            "{label}: {} energies, {} transmission points, {} uncertainties",
            energies.len(),
            transmission.len(),
            uncertainty.len(),
        )));
    }
    if uncertainty.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err(PipelineError::InvalidParameter(format!(
            "the {label} uncertainties must all be finite and positive"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::test_support::synthetic_isotope;
    use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};
    use nereids_physics::transmission::{InstrumentParams, SampleParams, forward_model};

    const L: f64 = 25.0;
    const W: f64 = 0.30;
    const DL: f64 = 0.05;

    fn spectrum(
        iso: &nereids_endf::resonance::ResonanceData,
        density: f64,
        temperature_k: f64,
        energies: &[f64],
    ) -> Vec<f64> {
        let sample = SampleParams::new(temperature_k, vec![(iso.clone(), density)]).unwrap();
        let inst = InstrumentParams {
            resolution: ResolutionFunction::Gaussian(ResolutionParams::new(L, W, DL, 0.0).unwrap()),
        };
        forward_model(energies, &sample, Some(&inst)).unwrap()
    }

    /// The joint fit recovers the sample's density and temperature and the
    /// shared resolution, from a resolution seed that is wrong.
    ///
    /// Seeding the widths away from truth is the point: if the fit merely
    /// echoed its seed the recovered resolution would still be wrong and the
    /// sample parameters would absorb the difference.
    #[test]
    fn a_joint_fit_recovers_the_sample_and_the_shared_resolution() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..160).map(|i| 18.0 + i as f64 * 0.03).collect();
        let sample_t = spectrum(&iso, 2.0e-3, 320.0, &energies);
        let calibrant_t = spectrum(&iso, 2.0e-3, 300.0, &energies);
        let unc = vec![1.0e-3; energies.len()];

        let calibrant = CalibrantSpectrum {
            energies: energies.clone(),
            transmission: calibrant_t,
            uncertainty: unc.clone(),
            isotopes: vec![(iso.clone(), 2.0e-3)],
            temperature_k: 300.0,
        };

        let r = fit_with_calibrant(
            &sample_t,
            &unc,
            &energies,
            &[iso],
            &[1.6e-3],
            300.0,
            true,
            &calibrant,
            L,
            // Seeds off truth by 50 % and 60 %.
            0.45,
            0.02,
        )
        .expect("the joint fit runs");

        assert!(r.converged, "joint fit did not converge");
        assert!(
            (r.densities[0] - 2.0e-3).abs() < 0.05 * 2.0e-3,
            "density {} is not within 5 % of 2.0e-3",
            r.densities[0]
        );
        let temperature = r.temperature_k.expect("temperature was fitted");
        assert!(
            (temperature - 320.0).abs() < 10.0,
            "temperature {temperature} K is not within 10 K of 320 K"
        );
        assert!(
            (r.delta_t_us - W).abs() < 0.2 * W,
            "the shared width {} is not within 20 % of {W}, so the fit did not \
             move off its seed of 0.45",
            r.delta_t_us
        );
        assert!(
            r.temperature_k_unc
                .is_some_and(|s| s.is_finite() && s > 0.0),
            "a joint fit must report a temperature uncertainty"
        );
    }

    /// A calibrant arm that cannot constrain anything is rejected, not fitted.
    ///
    /// The forward model skips a non-positive thickness, so a calibrant with
    /// a zero or NaN density is transparent: the arm carries no resonance,
    /// the resolution is constrained by nothing, and the joint fit would
    /// still return a temperature uncertainty — one computed from the sample
    /// alone while claiming to carry the calibration.
    #[test]
    fn a_calibrant_that_constrains_nothing_is_rejected() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..40).map(|i| 18.0 + i as f64 * 0.05).collect();
        let t = spectrum(&iso, 2.0e-3, 300.0, &energies);
        let unc = vec![1.0e-3; energies.len()];

        for bad in [0.0, -1.0e-3, f64::NAN] {
            let calibrant = CalibrantSpectrum {
                energies: energies.clone(),
                transmission: t.clone(),
                uncertainty: unc.clone(),
                isotopes: vec![(iso.clone(), bad)],
                temperature_k: 300.0,
            };
            let Err(err) = fit_with_calibrant(
                &t,
                &unc,
                &energies,
                &[iso.clone()],
                &[2.0e-3],
                300.0,
                true,
                &calibrant,
                L,
                W,
                DL,
            ) else {
                panic!("a calibrant density of {bad} must be rejected");
            };
            assert!(
                matches!(err, PipelineError::InvalidParameter(ref m) if m.contains("calibrant")),
                "expected a calibrant density rejection for {bad}, got {err}"
            );
        }
    }

    /// A temperature outside the box every other fit path uses is rejected.
    #[test]
    fn a_temperature_outside_the_shared_box_is_rejected() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..40).map(|i| 18.0 + i as f64 * 0.05).collect();
        let t = spectrum(&iso, 2.0e-3, 300.0, &energies);
        let unc = vec![1.0e-3; energies.len()];
        let calibrant = CalibrantSpectrum {
            energies: energies.clone(),
            transmission: t.clone(),
            uncertainty: unc.clone(),
            isotopes: vec![(iso.clone(), 2.0e-3)],
            temperature_k: 300.0,
        };
        // Zero Kelvin has no Doppler broadening at all, so it is not a sample
        // this path describes.
        for seed in [0.0, 6000.0] {
            assert!(
                fit_with_calibrant(
                    &t,
                    &unc,
                    &energies,
                    &[iso.clone()],
                    &[2.0e-3],
                    seed,
                    true,
                    &calibrant,
                    L,
                    W,
                    DL,
                )
                .is_err(),
                "a sample temperature of {seed} K must be rejected"
            );
        }
    }

    /// Both spectra are checked before anything runs.
    #[test]
    fn mismatched_spectra_are_rejected_before_fitting() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..40).map(|i| 18.0 + i as f64 * 0.05).collect();
        let t = spectrum(&iso, 2.0e-3, 300.0, &energies);
        let unc = vec![1.0e-3; energies.len()];
        let calibrant = CalibrantSpectrum {
            energies: energies.clone(),
            transmission: t.clone(),
            // One short.
            uncertainty: unc[1..].to_vec(),
            isotopes: vec![(iso.clone(), 2.0e-3)],
            temperature_k: 300.0,
        };
        let err = fit_with_calibrant(
            &t,
            &unc,
            &energies,
            &[iso],
            &[2.0e-3],
            300.0,
            true,
            &calibrant,
            L,
            W,
            DL,
        )
        .expect_err("a calibrant whose arrays disagree must be rejected");
        assert!(
            matches!(err, PipelineError::ShapeMismatch(ref m) if m.contains("calibrant")),
            "expected a calibrant shape mismatch, got {err}"
        );
    }
}
