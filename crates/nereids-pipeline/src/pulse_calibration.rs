//! A pulse calibrated on a foil, carried to experiments as a correlated
//! prior on the pulse numbers the foil resolved.

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::Prior;
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::resonance_center_energies;

use crate::counts_fit::{CountsFit, Measurement, Value, quantities};
use crate::error::PipelineError;
use crate::open_beam::{Calibration, Pulse};

/// The pulse numbers a calibration resolved, as indices into
/// `(α₀, α₁, β₀, β₁, R, h²)`, with their fitted values and covariance, and
/// the energies in eV of the calibration foil's lowest and highest resonance
/// in its window.
#[derive(Debug, Clone)]
pub struct PulsePrior {
    pub(crate) numbers: Vec<usize>,
    pub(crate) mean: Vec<f64>,
    pub(crate) covariance: FlatMatrix,
    pub(crate) line_span_ev: (f64, f64),
}

/// A pulse calibrated on a foil by [`fit_counts`](crate::counts_fit::fit_counts).
///
/// A pulse number is resolved when the calibration fitted it, it did not end
/// on a bound, and its variance is finite and positive; the others are held at
/// their fitted values.  The resolved numbers keep the fit's covariance of
/// them, which is conditional on every quantity that ended on a bound being
/// held there.  A number the counts do not determine has no variance and is
/// held.
#[derive(Debug, Clone)]
pub struct PulseCalibration {
    t0_us: f64,
    flight_path_m: f64,
    numbers: [f64; 6],
    energy_span_ev: (f64, f64),
    n_tau: usize,
    prior: Option<PulsePrior>,
}

impl PulseCalibration {
    /// The calibration given by `fit`, the result of
    /// [`fit_counts`](crate::counts_fit::fit_counts) on `measurement` with
    /// `calibration`.
    ///
    /// # Errors
    /// [`PipelineError::InvalidParameter`] if `fit` did not converge, has no
    /// covariance, or does not have the quantities `measurement` and
    /// `calibration` fit, or if some pulse number is resolved and the foil
    /// has no resonance between the energies of its last and first time
    /// edges at the fitted `t0` and flight path; [`PipelineError::Fitting`]
    /// if the resolved numbers' covariance is refused by
    /// [`Prior::correlated`].
    pub fn new(
        calibration: &Calibration,
        measurement: &Measurement,
        fit: &CountsFit,
    ) -> Result<Self, PipelineError> {
        let invalid = |message: &str| Err(PipelineError::InvalidParameter(message.into()));
        let Some(covariance) = fit.covariance.as_ref().filter(|_| fit.converged) else {
            return invalid("a pulse calibration needs a converged fit with a covariance");
        };
        let fitted: Vec<bool> = quantities(measurement, calibration)
            .map(|value| !matches!(value, Value::Known(_)))
            .collect();
        let free = fitted.iter().filter(|&&f| f).count();
        if covariance.nrows != free
            || fit.on_bound.len() != free
            || fit.densities.len() != measurement.isotopes.len()
        {
            return invalid(
                "the fit does not have the quantities of this measurement and calibration",
            );
        }
        let first_number = fitted.len() - 6;
        let resolved: Vec<(usize, usize)> = (0..6)
            .filter_map(|number| {
                let quantity = first_number + number;
                let i = fitted[..quantity].iter().filter(|&&f| f).count();
                let determined = |variance: f64| variance.is_finite() && variance > 0.0;
                (fitted[quantity] && !fit.on_bound[i] && determined(covariance.get(i, i)))
                    .then_some((number, i))
            })
            .collect();
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
            let lines = lines_in_window(
                &measurement.isotopes,
                &measurement.time_edges_us,
                fit.t0_us,
                fit.flight_path_m,
            );
            if lines.is_empty() {
                return invalid(
                    "the calibration foil has no resonance in its window to bound the energies \
                     its pulse holds over",
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
            t0_us: fit.t0_us,
            flight_path_m: fit.flight_path_m,
            numbers,
            energy_span_ev: calibration.pulse.energy_span_ev,
            n_tau: calibration.pulse.n_tau,
            prior,
        })
    }

    /// The calibration of an experiment: `t0` and the flight path fitted from
    /// the calibrated ones, the resolved pulse numbers fitted from theirs with
    /// their covariance as a prior, and the others known.
    pub fn calibration(&self) -> Calibration {
        let value = |number: usize| {
            let resolved = self
                .prior
                .as_ref()
                .is_some_and(|prior| prior.numbers.contains(&number));
            if resolved {
                Value::Fitted(self.numbers[number])
            } else {
                Value::Known(self.numbers[number])
            }
        };
        Calibration {
            t0_us: Value::Fitted(self.t0_us),
            flight_path_m: Value::Fitted(self.flight_path_m),
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

pub(crate) fn lines_in_window(
    isotopes: &[(ResonanceData, Value)],
    time_edges_us: &[f64],
    t0_us: f64,
    flight_path_m: f64,
) -> Vec<f64> {
    let energy = |t: f64| (TOF_FACTOR * flight_path_m / (t - t0_us).max(0.0)).powi(2);
    let window = energy(time_edges_us[time_edges_us.len() - 1])..=energy(time_edges_us[0]);
    isotopes
        .iter()
        .flat_map(|(isotope, _)| resonance_center_energies(&[isotope]))
        .filter(|e| window.contains(e))
        .collect()
}

#[cfg(test)]
mod tests {
    use nereids_endf::resonance::test_support::synthetic_isotope_multi;

    use super::*;
    use crate::beam::BeamSpline;

    fn measurement() -> Measurement {
        let lines = [(10.0, 0.05, 0.06), (25.0, 0.005, 0.06), (50.0, 0.01, 0.06)];
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
            isotopes: vec![(
                synthetic_isotope_multi(73, 181, &lines),
                Value::Fitted(2e-3),
            )],
            temperature_k: Value::Fitted(300.0),
        }
    }

    fn calibration() -> Calibration {
        Calibration {
            t0_us: Value::Fitted(3.0),
            flight_path_m: Value::Fitted(25.0),
            pulse: Pulse {
                alpha: [Value::Known(0.5), Value::Fitted(1.0)],
                beta: [Value::Fitted(0.08), Value::Fitted(0.01)],
                r: Value::Fitted(0.2),
                fwhm_squared_us2: Value::Fitted(0.1),
                energy_span_ev: (1.0, 200.0),
                n_tau: 256,
                prior: None,
            },
        }
    }

    fn fit() -> CountsFit {
        let n = 9;
        let mut covariance = FlatMatrix::zeros(n, n);
        for i in 0..n {
            *covariance.get_mut(i, i) = 0.01 * (i + 1) as f64;
        }
        for (i, j, c) in [(4, 7, 0.01), (4, 8, -0.005), (7, 8, 0.02), (2, 4, 0.003)] {
            *covariance.get_mut(i, j) = c;
            *covariance.get_mut(j, i) = c;
        }
        for i in 0..n {
            *covariance.get_mut(6, i) = f64::NAN;
            *covariance.get_mut(i, 6) = f64::NAN;
        }
        let mut on_bound = vec![false; n];
        on_bound[5] = true;
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
    fn a_calibration_holds_the_numbers_it_did_not_resolve() {
        let (m, c) = (measurement(), calibration());
        let calibrated = PulseCalibration::new(&c, &m, &fit()).unwrap().calibration();
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
        let covariance = fit().covariance.expect("covariance");
        for (a, i) in [4, 7, 8].into_iter().enumerate() {
            for (b, j) in [4, 7, 8].into_iter().enumerate() {
                assert_eq!(prior.covariance.get(a, b), covariance.get(i, j));
            }
        }
        assert_eq!(prior.line_span_ev, (10.0, 50.0));
        for unfinished in [
            CountsFit {
                converged: false,
                ..fit()
            },
            CountsFit {
                covariance: None,
                ..fit()
            },
        ] {
            assert!(PulseCalibration::new(&c, &m, &unfinished).is_err());
        }
    }
}
