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
use nereids_fitting::joint_resolution::{Densities, JointResolutionModel, SpectrumSpec};
use nereids_fitting::lm::{self, LmConfig};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::resolution_calib::{GAUSSIAN_DELTA_L_BOUNDS_M, GAUSSIAN_DELTA_T_BOUNDS_US};

use crate::error::PipelineError;
use crate::pipeline::TEMPERATURE_BOUNDS_K;

/// The unknown spectrum and what the fit is free to move in it.
pub struct SampleSpectrum {
    /// Energy grid (eV), ascending.
    pub energies: Vec<f64>,
    /// Measured transmission on that grid.
    pub transmission: Vec<f64>,
    /// One-sigma uncertainty per point.
    pub uncertainty: Vec<f64>,
    /// Each isotope with the areal density to start it at.
    pub isotopes: Vec<(ResonanceData, f64)>,
    /// Temperature (K); the start value when it is fitted.
    pub temperature_k: f64,
    /// Whether the temperature is free.
    pub fit_temperature: bool,
}

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
/// fitted, then the two squared resolution widths.
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
/// positive, or a temperature or width seed falls outside the box every fit
/// path shares, and [`PipelineError::Fitting`] when the optimizer cannot run.
pub fn fit_with_calibrant(
    sample: &SampleSpectrum,
    calibrant: &CalibrantSpectrum,
    flight_path_m: f64,
    delta_t_init: f64,
    delta_l_init: f64,
) -> Result<JointFitResult, PipelineError> {
    for (label, energies, transmission, uncertainty, isotopes) in [
        (
            "sample",
            &sample.energies,
            &sample.transmission,
            &sample.uncertainty,
            &sample.isotopes,
        ),
        (
            "calibrant",
            &calibrant.energies,
            &calibrant.transmission,
            &calibrant.uncertainty,
            &calibrant.isotopes,
        ),
    ] {
        check_spectrum(label, energies, transmission, uncertainty)?;
        if isotopes.is_empty() {
            return Err(PipelineError::InvalidParameter(format!(
                "the {label} needs at least one isotope"
            )));
        }
    }
    // Every scalar the fit reads has a range it has to lie in. The widths'
    // is the box the calibration fits in: the broadening grid is extended by
    // five sigma of the width at each boundary, so a width far outside that
    // box asks for a grid the machine cannot hold before the first residual
    // is evaluated. A density of zero or less is transparent -- the forward
    // model skips a non-positive thickness -- so a calibrant made of them
    // carries no resonance and constrains no resolution.
    let positive = (f64::MIN_POSITIVE, f64::INFINITY);
    for (label, value, range) in [
        ("flight_path_m", flight_path_m, positive),
        ("temperature_k", sample.temperature_k, TEMPERATURE_BOUNDS_K),
        (
            "the calibrant temperature",
            calibrant.temperature_k,
            TEMPERATURE_BOUNDS_K,
        ),
        ("delta_t_init", delta_t_init, GAUSSIAN_DELTA_T_BOUNDS_US),
        ("delta_l_init", delta_l_init, GAUSSIAN_DELTA_L_BOUNDS_M),
    ]
    .into_iter()
    .chain(
        sample
            .isotopes
            .iter()
            .map(|(_, n)| ("a sample density", *n, positive)),
    )
    .chain(
        calibrant
            .isotopes
            .iter()
            .map(|(_, n)| ("a calibrant density", *n, positive)),
    ) {
        check_range(label, value, range)?;
    }

    let n_isotopes = sample.isotopes.len();
    let (delta_t_index, delta_l_index) = resolution_indices(n_isotopes, sample.fit_temperature);
    let temperature_index = sample.fit_temperature.then_some(n_isotopes);
    let (sample_data, initial_densities): (Vec<_>, Vec<_>) =
        sample.isotopes.iter().cloned().unzip();
    let (calibrant_data, calibrant_densities): (Vec<_>, Vec<_>) =
        calibrant.isotopes.iter().cloned().unzip();

    let model = JointResolutionModel::new(
        SpectrumSpec {
            energies: sample.energies.clone(),
            resonance_data: sample_data,
            densities: Densities::Fitted((0..n_isotopes).collect()),
            temperature_index,
            temperature_k: sample.temperature_k,
        },
        SpectrumSpec {
            energies: calibrant.energies.clone(),
            resonance_data: calibrant_data,
            densities: Densities::Known(calibrant_densities),
            temperature_index: None,
            temperature_k: calibrant.temperature_k,
        },
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
    if sample.fit_temperature {
        values.push(FitParameter {
            name: "temperature_k".into(),
            value: sample.temperature_k,
            lower: TEMPERATURE_BOUNDS_K.0,
            upper: TEMPERATURE_BOUNDS_K.1,
            fixed: false,
        });
    }
    // The optimizer works in the SQUARED widths. The kernel combines the two
    // terms in quadrature, so a width itself enters the width quadratically
    // and its derivative vanishes at zero, leaving a width seeded at zero
    // unfitted; the squares enter linearly and have a finite derivative
    // there.
    for (name, seed, (lo, hi)) in [
        ("delta_t_us_sq", delta_t_init, GAUSSIAN_DELTA_T_BOUNDS_US),
        ("delta_l_m_sq", delta_l_init, GAUSSIAN_DELTA_L_BOUNDS_M),
    ] {
        values.push(FitParameter {
            name: name.into(),
            value: seed * seed,
            lower: lo * lo,
            upper: hi * hi,
            fixed: false,
        });
    }
    let mut params = ParameterSet::new(values);

    // The two spectra are concatenated in the order the model predicts them.
    let mut data = sample.transmission.clone();
    data.extend_from_slice(&calibrant.transmission);
    let mut sigma = sample.uncertainty.clone();
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
        densities: result.params[..n_isotopes].to_vec(),
        density_uncertainties: result
            .uncertainties
            .as_ref()
            .map(|u| u[..n_isotopes].to_vec()),
        temperature_k: temperature_index.map(|i| result.params[i]),
        temperature_k_unc: temperature_index.and_then(sigma_of),
        delta_t_us: result.params[delta_t_index].max(0.0).sqrt(),
        delta_l_m: result.params[delta_l_index].max(0.0).sqrt(),
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
    })
}

/// Reject a scalar the fit cannot use.
fn check_range(label: &str, value: f64, (lo, hi): (f64, f64)) -> Result<(), PipelineError> {
    if value.is_finite() && (lo..=hi).contains(&value) {
        return Ok(());
    }
    Err(PipelineError::InvalidParameter(format!(
        "{label} must be finite and lie in [{lo}, {hi}], got {value}"
    )))
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
    // The forward model reads the grid as ascending positive energies and
    // interpolates the kernel across it; a repeated, reversed or non-positive
    // energy has no time of flight, and a non-finite transmission makes every
    // residual NaN, which an optimizer reads as a step it cannot improve on
    // rather than as bad input.
    if energies.iter().any(|e| !e.is_finite() || *e <= 0.0) {
        return Err(PipelineError::InvalidParameter(format!(
            "the {label} energies must all be finite and positive"
        )));
    }
    if let Some(i) = energies.windows(2).position(|w| w[1] <= w[0]) {
        return Err(PipelineError::InvalidParameter(format!(
            "the {label} energies must increase: [{i}] = {} is not below [{}] = {}",
            energies[i],
            i + 1,
            energies[i + 1],
        )));
    }
    if transmission.iter().any(|t| !t.is_finite()) {
        return Err(PipelineError::InvalidParameter(format!(
            "the {label} transmission values must all be finite"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::test_support::synthetic_isotope;
    use nereids_fitting::resolution_calib::{
        GAUSSIAN_DELTA_L_BOUNDS_M, GAUSSIAN_DELTA_T_BOUNDS_US,
    };
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

    /// A sample arm carrying `t` on `energies`, with its temperature free.
    fn sample_arm(
        iso: &nereids_endf::resonance::ResonanceData,
        energies: &[f64],
        t: &[f64],
        unc: &[f64],
        density: f64,
        temperature_k: f64,
    ) -> SampleSpectrum {
        SampleSpectrum {
            energies: energies.to_vec(),
            transmission: t.to_vec(),
            uncertainty: unc.to_vec(),
            isotopes: vec![(iso.clone(), density)],
            temperature_k,
            fit_temperature: true,
        }
    }

    /// A calibrant arm at a known density and temperature.
    fn calibrant_arm(
        iso: &nereids_endf::resonance::ResonanceData,
        energies: &[f64],
        t: &[f64],
        unc: &[f64],
        density: f64,
        temperature_k: f64,
    ) -> CalibrantSpectrum {
        CalibrantSpectrum {
            energies: energies.to_vec(),
            transmission: t.to_vec(),
            uncertainty: unc.to_vec(),
            isotopes: vec![(iso.clone(), density)],
            temperature_k,
        }
    }

    /// The joint fit recovers the sample's density and temperature and the
    /// shared resolution, from a resolution seed that is wrong.
    ///
    /// Seeding the widths away from truth is the point: if the fit merely
    /// echoed its seed the recovered resolution would still be wrong and the
    /// sample parameters would absorb the difference.
    ///
    /// The two arms get different isotopes on different grids. Sharing one
    /// would let the arms be crossed — calibrant parameters applied to the
    /// sample, or one grid used for both — without changing the result.
    #[test]
    fn a_joint_fit_recovers_the_sample_and_the_shared_resolution() {
        let sample_iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let calibrant_iso = synthetic_isotope(72, 177, 31.0, 0.04, 0.07);
        let sample_e: Vec<f64> = (0..160).map(|i| 18.0 + i as f64 * 0.03).collect();
        let calibrant_e: Vec<f64> = (0..140).map(|i| 29.0 + i as f64 * 0.032).collect();
        let sample_t = spectrum(&sample_iso, 2.0e-3, 320.0, &sample_e);
        let calibrant_t = spectrum(&calibrant_iso, 3.0e-3, 300.0, &calibrant_e);
        let unc = vec![1.0e-3; sample_e.len()];

        let calibrant = CalibrantSpectrum {
            energies: calibrant_e,
            transmission: calibrant_t,
            uncertainty: vec![1.0e-3; 140],
            isotopes: vec![(calibrant_iso, 3.0e-3)],
            temperature_k: 300.0,
        };

        let r = fit_with_calibrant(
            &sample_arm(&sample_iso, &sample_e, &sample_t, &unc, 1.6e-3, 300.0),
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

    /// A width the broadening grid cannot carry is rejected, and the
    /// optimizer cannot reach one either.
    ///
    /// The Gaussian grid is extended by five sigma of the width at each
    /// boundary, so its point count grows with the width; a forward model at
    /// 5000 µs on a 280-point eV-range grid already takes tens of seconds,
    /// and there is no upper limit at which it merely gets slow rather than
    /// unrunnable. The seed is checked against the box the calibration fits
    /// in, and the optimizer's own box is that same range squared.
    #[test]
    fn a_width_the_grid_cannot_carry_is_rejected_and_unreachable() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..40).map(|i| 18.0 + i as f64 * 0.05).collect();
        let t = spectrum(&iso, 2.0e-3, 300.0, &energies);
        let unc = vec![1.0e-3; energies.len()];
        let calibrant = calibrant_arm(&iso, &energies, &t, &unc, 2.0e-3, 300.0);
        let run = |dt: f64, dl: f64| {
            fit_with_calibrant(
                &sample_arm(&iso, &energies, &t, &unc, 2.0e-3, 300.0),
                &calibrant,
                L,
                dt,
                dl,
            )
        };

        let (t_lo, t_hi) = GAUSSIAN_DELTA_T_BOUNDS_US;
        let (_, l_hi) = GAUSSIAN_DELTA_L_BOUNDS_M;
        assert!(
            run(t_hi * 2.0, DL).is_err(),
            "a timing width above {t_hi} µs must be rejected before any \
             residual is evaluated"
        );
        assert!(
            run(t_lo / 2.0, DL).is_err(),
            "a timing width below {t_lo} µs must be rejected"
        );
        assert!(
            run(W, l_hi * 2.0).is_err(),
            "a flight-path width above {l_hi} m must be rejected"
        );
        // And the optimizer's own box is the same range squared, so no trial
        // step can reach a width the seed check would have refused.
        let r = run(W, DL).expect("a width inside the box is accepted");
        assert!(
            (t_lo..=t_hi).contains(&r.delta_t_us),
            "the fitted timing width {} left [{t_lo}, {t_hi}]",
            r.delta_t_us
        );
        assert!(
            r.delta_l_m <= l_hi,
            "the fitted flight-path width {} exceeded {l_hi}",
            r.delta_l_m
        );
    }

    /// A grid or a spectrum the forward model cannot read is rejected on both
    /// arms.
    ///
    /// Length agreement is not enough: the model reads the grid as ascending
    /// positive energies, and a non-finite transmission turns every residual
    /// into NaN, which the optimizer reads as a step that cannot be improved
    /// on rather than as bad input.
    #[test]
    fn unreadable_grids_and_spectra_are_rejected_on_either_arm() {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..40).map(|i| 18.0 + i as f64 * 0.05).collect();
        let good = spectrum(&iso, 2.0e-3, 300.0, &energies);
        let unc = vec![1.0e-3; energies.len()];

        let mut descending = energies.clone();
        descending.reverse();
        let mut repeated = energies.clone();
        repeated[7] = repeated[6];
        let mut negative = energies.clone();
        negative[0] = -1.0;
        let mut nan_t = good.clone();
        nan_t[3] = f64::NAN;

        let cases: [(&str, Vec<f64>, Vec<f64>); 4] = [
            ("descending energies", descending, good.clone()),
            ("a repeated energy", repeated, good.clone()),
            ("a negative energy", negative, good.clone()),
            ("a non-finite transmission", energies.clone(), nan_t),
        ];
        for (what, grid, values) in cases {
            // Once as the sample, once as the calibrant: both arms are read by
            // the same model and neither may skip the check.
            let sound = calibrant_arm(&iso, &energies, &good, &unc, 2.0e-3, 300.0);
            assert!(
                fit_with_calibrant(
                    &sample_arm(&iso, &grid, &values, &unc, 2.0e-3, 300.0),
                    &sound,
                    L,
                    W,
                    DL,
                )
                .is_err(),
                "{what} must be rejected on the sample arm"
            );
            let broken = CalibrantSpectrum {
                energies: grid,
                transmission: values,
                uncertainty: unc.clone(),
                isotopes: vec![(iso.clone(), 2.0e-3)],
                temperature_k: 300.0,
            };
            assert!(
                fit_with_calibrant(
                    &sample_arm(&iso, &energies, &good, &unc, 2.0e-3, 300.0),
                    &broken,
                    L,
                    W,
                    DL,
                )
                .is_err(),
                "{what} must be rejected on the calibrant arm"
            );
        }
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
                &sample_arm(&iso, &energies, &t, &unc, 2.0e-3, 300.0),
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
        let calibrant = calibrant_arm(&iso, &energies, &t, &unc, 2.0e-3, 300.0);
        // Zero Kelvin has no Doppler broadening at all, so it is not a sample
        // this path describes.
        for seed in [0.0, 6000.0] {
            assert!(
                fit_with_calibrant(
                    &sample_arm(&iso, &energies, &t, &unc, 2.0e-3, seed),
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
            &sample_arm(&iso, &energies, &t, &unc, 2.0e-3, 300.0),
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
