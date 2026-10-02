//! A pulse calibrated on a foil, carried to experiments as a correlated
//! prior on the pulse numbers the foil resolved.

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::Prior;
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::resonance_center_energies;

use crate::counts_fit::{CountsFit, Measurement, Value, fit_counts, quantities};
use crate::error::PipelineError;
use crate::open_beam::{Calibration, PULSE_NUMBERS, Pulse};

/// The pulse numbers a calibration resolved, as indices into
/// `(α₀, α₁, β₀, β₁, R, h²)`, with their fitted values and covariance, and
/// the energies in eV of the lowest and highest resonance in the calibration
/// foil's window of its isotopes with a positive density.
#[derive(Debug, Clone)]
pub struct PulsePrior {
    pub(crate) numbers: Vec<usize>,
    pub(crate) mean: Vec<f64>,
    pub(crate) covariance: FlatMatrix,
    pub(crate) line_span_ev: (f64, f64),
}

impl PulsePrior {
    pub(crate) fn uncalibrated_line(
        &self,
        isotopes: &[(ResonanceData, Value)],
        time_edges_us: &[f64],
        t0_us: f64,
        flight_path_m: f64,
    ) -> Option<f64> {
        let present = isotopes
            .iter()
            .filter(|(_, density)| *density != Value::Known(0.0))
            .map(|(isotope, _)| isotope);
        let (low, high) = self.line_span_ev;
        lines_in_window(present, time_edges_us, t0_us, flight_path_m)
            .into_iter()
            .find(|e| !(low..=high).contains(e))
    }
}

/// A pulse calibrated on a foil by [`fit_counts`].
///
/// A pulse number the calibration fitted is resolved, unless it ended on a
/// bound, where it is held at its fitted value; a number known in the
/// calibration stays known.  The resolved numbers keep the fit's covariance of
/// them, which is conditional on every quantity that ended on a bound being
/// held there.  When the calibration holds a number on its bound, or resolves
/// one near it, an experiment's error bars on `t0`, the flight path and the
/// pulse numbers are not standard errors; those on the densities and the
/// temperature are.
#[derive(Debug, Clone)]
pub struct PulseCalibration {
    fit: CountsFit,
    energy_span_ev: (f64, f64),
    n_tau: usize,
    prior: Option<PulsePrior>,
}

impl PulseCalibration {
    /// The pulse calibrated by fitting the counts of a foil, `measurement`,
    /// with `calibration`.
    ///
    /// # Errors
    /// Everything [`fit_counts`] refuses; [`PipelineError::InvalidParameter`]
    /// if the fit did not converge, a pulse number it fitted ended off its
    /// bounds without a finite positive variance, as when the counts do not
    /// determine it or a fitted temperature ended at 1 K or 5000 K, or some
    /// pulse number is resolved and no isotope of the foil fitted or known to
    /// a positive density has a resonance between the energies of its last and
    /// first time edges at the fitted `t0` and flight path; [`PipelineError::Fitting`] if the resolved numbers'
    /// covariance is refused by [`Prior::correlated`].
    pub fn new(
        measurement: &Measurement,
        calibration: &Calibration,
    ) -> Result<Self, PipelineError> {
        let fit = fit_counts(measurement, calibration)?;
        Self::from_fit(measurement, calibration, fit)
    }

    /// The fit of the foil's counts.
    pub fn fit(&self) -> &CountsFit {
        &self.fit
    }

    fn from_fit(
        measurement: &Measurement,
        calibration: &Calibration,
        fit: CountsFit,
    ) -> Result<Self, PipelineError> {
        let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
        let Some(covariance) = fit.covariance.as_ref().filter(|_| fit.converged) else {
            return invalid("a pulse calibration needs a converged fit".into());
        };
        let fitted: Vec<bool> = quantities(measurement, calibration)
            .map(|value| !matches!(value, Value::Known(_)))
            .collect();
        let first_number = fitted.len() - 6;
        let mut resolved = Vec::with_capacity(6);
        for (number, name) in PULSE_NUMBERS.into_iter().enumerate() {
            let quantity = first_number + number;
            let i = fitted[..quantity].iter().filter(|&&f| f).count();
            if !fitted[quantity] || fit.on_bound[i] {
                continue;
            }
            let variance = covariance.get(i, i);
            if !(variance.is_finite() && variance > 0.0) {
                return invalid(format!(
                    "the calibration fitted {name} without determining it: its variance is \
                     {variance}"
                ));
            }
            resolved.push((number, i));
        }
        let numbers = [
            fit.alpha[0],
            fit.alpha[1],
            fit.beta[0],
            fit.beta[1],
            fit.r,
            fit.fwhm_squared_us2,
        ];
        let prior = if resolved.is_empty() {
            None
        } else {
            let present = measurement
                .isotopes
                .iter()
                .zip(&fit.densities)
                .filter(|(_, density)| **density > 0.0)
                .map(|((isotope, _), _)| isotope);
            let lines = lines_in_window(
                present,
                &measurement.time_edges_us,
                fit.t0_us,
                fit.flight_path_m,
            );
            if lines.is_empty() {
                return invalid(
                    "the calibration foil has no resonance in its window to bound the energies \
                     its pulse holds over"
                        .into(),
                );
            }
            let line_span_ev = lines
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &e| {
                    (low.min(e), high.max(e))
                });
            let n = resolved.len();
            let mut block = FlatMatrix::zeros(n, n);
            for (a, &(_, i)) in resolved.iter().enumerate() {
                for (b, &(_, j)) in resolved.iter().enumerate() {
                    *block.get_mut(a, b) = 0.5 * (covariance.get(i, j) + covariance.get(j, i));
                }
            }
            let covered: Vec<usize> = resolved.iter().map(|&(number, _)| number).collect();
            let mean: Vec<f64> = covered.iter().map(|&number| numbers[number]).collect();
            Prior::correlated(&covered, &mean, &block)?;
            Some(PulsePrior {
                numbers: covered,
                mean,
                covariance: block,
                line_span_ev,
            })
        };
        Ok(Self {
            fit,
            energy_span_ev: calibration.pulse.energy_span_ev,
            n_tau: calibration.pulse.n_tau,
            prior,
        })
    }

    /// The calibration of an experiment: `t0` and the flight path fitted from
    /// the calibrated ones, the resolved pulse numbers fitted from theirs with
    /// their covariance as a prior, and the others known.
    pub fn calibration(&self) -> Calibration {
        let fit = &self.fit;
        let numbers = [
            fit.alpha[0],
            fit.alpha[1],
            fit.beta[0],
            fit.beta[1],
            fit.r,
            fit.fwhm_squared_us2,
        ];
        let value = |number: usize| {
            let resolved = self
                .prior
                .as_ref()
                .is_some_and(|prior| prior.numbers.contains(&number));
            if resolved {
                Value::Fitted(numbers[number])
            } else {
                Value::Known(numbers[number])
            }
        };
        Calibration {
            t0_us: Value::Fitted(fit.t0_us),
            flight_path_m: Value::Fitted(fit.flight_path_m),
            pulse: Pulse {
                alpha: [value(0), value(1)],
                beta: [value(2), value(3)],
                r: value(4),
                fwhm_squared_us2: value(5),
                energy_span_ev: self.energy_span_ev,
                n_tau: self.n_tau,
                prior: self.prior.clone(),
            },
        }
    }
}

fn lines_in_window<'a>(
    isotopes: impl Iterator<Item = &'a ResonanceData>,
    time_edges_us: &[f64],
    t0_us: f64,
    flight_path_m: f64,
) -> Vec<f64> {
    let energy = |t: f64| (TOF_FACTOR * flight_path_m / (t - t0_us).max(0.0)).powi(2);
    let window = energy(time_edges_us[time_edges_us.len() - 1])..=energy(time_edges_us[0]);
    isotopes
        .flat_map(|isotope| resonance_center_energies(&[isotope]))
        .filter(|e| window.contains(e))
        .collect()
}

#[cfg(test)]
mod tests {
    use nereids_endf::resonance::test_support::{synthetic_isotope, synthetic_isotope_multi};

    use super::*;
    use crate::beam::BeamSpline;

    fn measurement(absent: Option<f64>) -> Measurement {
        let lines = [(10.0, 0.05, 0.06), (25.0, 0.005, 0.06), (50.0, 0.01, 0.06)];
        let foil = (
            synthetic_isotope_multi(73, 181, &lines),
            Value::Fitted(2e-3),
        );
        let impurity = absent.map(|energy| {
            (
                synthetic_isotope(74, 184, energy, 0.01, 0.06),
                Value::Known(0.0),
            )
        });
        let edges: Vec<f64> = (0..=45).map(|i| 240.0 + 8.0 * f64::from(i)).collect();
        let bins = edges.len() - 1;
        Measurement {
            time_edges_us: edges,
            open_counts: vec![100.0; bins],
            sample_counts: vec![100.0; bins],
            open_live: None,
            sample_live: None,
            charge_ratio: 1.0,
            normalization: Value::Known(1.0),
            background: [Value::Known(0.0); 3],
            isotopes: std::iter::once(foil).chain(impurity).collect(),
            temperature_k: Value::Fitted(300.0),
        }
    }

    fn calibration(beta1: Value) -> Calibration {
        Calibration {
            t0_us: Value::Fitted(3.0),
            flight_path_m: Value::Fitted(25.0),
            pulse: Pulse {
                alpha: [Value::Known(0.5), Value::Fitted(1.0)],
                beta: [Value::Fitted(0.08), beta1],
                r: Value::Fitted(0.2),
                fwhm_squared_us2: Value::Fitted(0.1),
                energy_span_ev: (1.0, 200.0),
                n_tau: 256,
                prior: None,
            },
        }
    }

    fn fit(free: usize, bounded: usize) -> CountsFit {
        let mut covariance = FlatMatrix::zeros(free, free);
        for i in 0..free {
            *covariance.get_mut(i, i) = 0.01 * (i + 1) as f64;
        }
        let (r, h) = (free - 2, free - 1);
        for (i, j, c) in [(4, r, 0.01), (4, h, -0.005), (r, h, 0.02), (2, 4, 0.003)] {
            *covariance.get_mut(i, j) = c;
            *covariance.get_mut(j, i) = c;
        }
        let mut on_bound = vec![false; free];
        on_bound[bounded] = true;
        CountsFit {
            densities: vec![2e-3],
            temperature_k: 300.0,
            normalization: 1.0,
            background: [0.0; 3],
            t0_us: 3.0,
            flight_path_m: 25.0,
            alpha: [0.5, 1.1],
            beta: [0.0, 0.01],
            r: 0.21,
            fwhm_squared_us2: 0.12,
            covariance: Some(covariance),
            on_bound,
            beam: BeamSpline::constant(200.0, 600.0, 1.0),
            beam_at_limit: false,
            deviance: 0.0,
            converged: true,
            overdispersion: [None; 2],
            measured_pulls: None,
            pulse_consistency: None,
            step_us: 0.1,
            points: 10,
            halvings: 0,
        }
    }

    #[test]
    fn a_calibration_holds_numbers_on_a_bound_and_refuses_undetermined_ones() {
        let (m, c) = (measurement(Some(55.0)), calibration(Value::Known(0.01)));
        let with_impurity = CountsFit {
            densities: vec![2e-3, 0.0],
            ..fit(8, 5)
        };
        let calibrated = PulseCalibration::from_fit(&m, &c, with_impurity)
            .unwrap()
            .calibration();
        let pulse = &calibrated.pulse;
        assert_eq!(pulse.alpha, [Value::Known(0.5), Value::Fitted(1.1)]);
        assert_eq!(pulse.beta, [Value::Known(0.0), Value::Known(0.01)]);
        assert_eq!(
            [pulse.r, pulse.fwhm_squared_us2],
            [Value::Fitted(0.21), Value::Fitted(0.12)]
        );
        assert_eq!(
            [calibrated.t0_us, calibrated.flight_path_m],
            [Value::Fitted(3.0), Value::Fitted(25.0)]
        );
        let prior = pulse.prior.as_ref().expect("prior");
        assert_eq!(prior.numbers, [1, 4, 5]);
        assert_eq!(prior.mean, [1.1, 0.21, 0.12]);
        let covariance = fit(8, 5).covariance.expect("covariance");
        for (a, i) in [4, 6, 7].into_iter().enumerate() {
            for (b, j) in [4, 6, 7].into_iter().enumerate() {
                assert_eq!(prior.covariance.get(a, b), covariance.get(i, j));
            }
        }
        assert_eq!(prior.line_span_ev, (10.0, 50.0));

        let mut undetermined = fit(9, 5);
        let mut withheld = fit(8, 5);
        let covariance = undetermined.covariance.as_mut().expect("covariance");
        for i in 0..9 {
            *covariance.get_mut(6, i) = f64::NAN;
            *covariance.get_mut(i, 6) = f64::NAN;
        }
        withheld
            .covariance
            .as_mut()
            .expect("covariance")
            .data
            .fill(f64::NAN);
        for (beta1, refused) in [
            (Value::Fitted(0.01), undetermined),
            (Value::Known(0.01), withheld),
            (
                Value::Known(0.01),
                CountsFit {
                    converged: false,
                    ..fit(8, 5)
                },
            ),
            (
                Value::Known(0.01),
                CountsFit {
                    covariance: None,
                    ..fit(8, 5)
                },
            ),
        ] {
            assert!(PulseCalibration::from_fit(&m, &calibration(beta1), refused).is_err());
        }
    }

    #[test]
    fn a_calibration_without_lines_or_with_a_singular_block_is_refused() {
        let c = calibration(Value::Known(0.01));
        let mut beyond = measurement(None);
        beyond.isotopes[0].0 = synthetic_isotope(73, 181, 100.0, 0.05, 0.06);
        assert!(matches!(
            PulseCalibration::from_fit(&beyond, &c, fit(8, 5)),
            Err(PipelineError::InvalidParameter(_))
        ));
        let mut singular = fit(8, 5);
        let covariance = singular.covariance.as_mut().expect("covariance");
        let (a, b) = (covariance.get(4, 4), covariance.get(6, 6));
        *covariance.get_mut(4, 6) = (a * b).sqrt();
        *covariance.get_mut(6, 4) = (a * b).sqrt();
        assert!(matches!(
            PulseCalibration::from_fit(&measurement(None), &c, singular),
            Err(PipelineError::Fitting(_))
        ));
    }

    #[test]
    fn only_isotopes_that_may_be_present_must_lie_within_the_calibrated_lines() {
        let prior = PulsePrior {
            numbers: vec![1],
            mean: vec![1.0],
            covariance: FlatMatrix::zeros(1, 1),
            line_span_ev: (10.0, 50.0),
        };
        let mut m = measurement(Some(55.0));
        let line =
            |m: &Measurement| prior.uncalibrated_line(&m.isotopes, &m.time_edges_us, 3.0, 25.0);
        assert_eq!(line(&m), None);
        m.isotopes[1].1 = Value::Fitted(1e-4);
        assert_eq!(line(&m), Some(55.0));
    }
}
