//! A pulse calibrated on a foil, carried to experiments as a correlated
//! prior on the pulse numbers the foil resolved.

use nereids_core::types::Isotope;
use nereids_endf::resonance::ResonanceData;
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::Prior;
use nereids_fitting::statistics::{Consistency, agreement};
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::resonance_center_energies;
use serde::{Deserialize, Serialize};

use crate::counts_fit::{CountsFit, Measurement, Value, fit_counts, quantities};
use crate::error::PipelineError;
use crate::open_beam::{Calibration, PULSE_NUMBERS, PULSE_RANGES, Pulse};
use crate::pipeline::TEMPERATURE_BOUNDS_K;

/// The pulse numbers a calibration resolved, as indices into
/// `(α₀, α₁, β₀, β₁, R, h²)`, with their fitted values and covariance.
#[derive(Debug, Clone)]
pub struct PulsePrior {
    pub(crate) numbers: Vec<usize>,
    pub(crate) mean: Vec<f64>,
    pub(crate) covariance: FlatMatrix,
}

impl Pulse {
    pub(crate) fn uncalibrated_line(
        &self,
        isotopes: &[(ResonanceData, Value)],
        time_edges_us: &[f64],
        t0_us: f64,
        flight_path_m: f64,
    ) -> Option<f64> {
        let (low, high) = self.line_span_ev?;
        let present = isotopes
            .iter()
            .filter(|(_, density)| *density != Value::Known(0.0))
            .map(|(isotope, _)| isotope);
        lines_in_window(present, time_edges_us, t0_us, flight_path_m)
            .into_iter()
            .find(|e| !(low..=high).contains(e))
    }
}

const FORMAT_VERSION: u32 = 1;

const PULSE_MODEL: &str = "Ikeda–Carpenter pulse with α = α₀√E + α₁ and β = β₀√E + β₁ in 1/µs, \
     E in eV, a storage fraction R constant over the energy span, folded with the proton \
     pulse's triangle of FWHM h";

const NUMBERS: [(&str, &str); 6] = [
    ("alpha0", "1/(µs·√eV)"),
    ("alpha1", "1/µs"),
    ("beta0", "1/(µs·√eV)"),
    ("beta1", "1/µs"),
    ("r", "1"),
    ("fwhm_squared", "µs²"),
];

/// Where a calibration's counts came from: the identifiers of the physical
/// foil and of its open-beam and sample runs, none empty and the two runs
/// different.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub foil: String,
    pub open: String,
    pub sample: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Resolved,
    OnBound,
    Known,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stated {
    value: f64,
    #[serde(deserialize_with = "Option::deserialize")]
    sd: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FoilIsotope {
    #[serde(with = "nuclide")]
    isotope: Isotope,
    density: Stated,
}

mod nuclide {
    use nereids_core::types::Isotope;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Nuclide {
        z: u32,
        a: u32,
    }

    pub(super) fn serialize<S: Serializer>(isotope: &Isotope, s: S) -> Result<S::Ok, S::Error> {
        Nuclide {
            z: isotope.z(),
            a: isotope.a(),
        }
        .serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Isotope, D::Error> {
        let nuclide = Nuclide::deserialize(d)?;
        Isotope::new(nuclide.z, nuclide.a).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Foil {
    isotopes: Vec<FoilIsotope>,
    temperature_k: Stated,
}

/// A pulse calibrated on a foil by [`fit_counts`], as its calibration file
/// holds it.
///
/// A pulse number the calibration fitted is resolved, unless it ended on a
/// bound, where it is held at its fitted value; a number known in the
/// calibration stays known.  The resolved numbers keep the fit's covariance of
/// them, which is conditional on every quantity that ended on a bound being
/// held there.  When the calibration fitted a pulse number, resolved or held,
/// an experiment's pulse carries the foil's
/// [`line_span_ev`](Pulse::line_span_ev).  When the calibration holds a
/// number on its bound, or resolves one near it, an experiment's error bars
/// on `t0`, the flight path and the pulse numbers are not standard errors;
/// those on the densities and the temperature are.
#[derive(Debug, Clone)]
pub struct PulseCalibration {
    numbers: [f64; 6],
    status: [Status; 6],
    prior: Option<PulsePrior>,
    t0_us: f64,
    flight_path_m: f64,
    energy_span_ev: (f64, f64),
    n_tau: usize,
    line_span_ev: Option<(f64, f64)>,
    foil: Foil,
    provenance: Provenance,
    sample_overdispersion: Option<f64>,
    transfer: Option<TransferRecord>,
}

impl PulseCalibration {
    /// The pulse calibrated by fitting the counts of a foil, `measurement`,
    /// from `provenance`, with `calibration`, and that fit.  Each isotope of
    /// the foil not known to be absent (`Value::Known(0.0)`) has a measured
    /// density, and the foil's effective temperature is measured.
    ///
    /// # Errors
    /// [`PipelineError::InvalidParameter`] if a density or the temperature of
    /// the foil is not measured, as above, an identifier of `provenance` is
    /// empty or its runs are the same, or `calibration`'s pulse carries a
    /// prior or a measured number, since a calibration foil is calibrated
    /// alone; everything
    /// [`fit_counts`] refuses;
    /// [`PipelineError::InvalidParameter`] if the fit did not converge, a pulse
    /// number it fitted ended off its bounds without a finite positive
    /// variance, as when the counts do not determine it or a fitted
    /// temperature ended at 1 K or 5000 K, or the fit fitted a pulse number and
    /// no isotope of the foil fitted or known to a positive density has a
    /// resonance between the energies of its last and first time edges at the
    /// fitted `t0` and flight path; [`PipelineError::Fitting`] if the resolved
    /// numbers' covariance is refused by [`Prior::correlated`].
    pub fn new(
        measurement: &Measurement,
        calibration: &Calibration,
        provenance: Provenance,
    ) -> Result<(Self, CountsFit), PipelineError> {
        let foil = foil(measurement)?;
        check_provenance(&provenance)?;
        let pulse = &calibration.pulse;
        let measured = [pulse.r, pulse.fwhm_squared_us2]
            .iter()
            .chain(&pulse.alpha)
            .chain(&pulse.beta)
            .any(|value| matches!(value, Value::Measured { .. }));
        if pulse.prior.is_some() || measured {
            return Err(PipelineError::InvalidParameter(
                "a calibration foil is calibrated alone, with no pulse prior or measured pulse \
                 number"
                    .into(),
            ));
        }
        let fit = fit_counts(measurement, calibration)?;
        let pulse = Self::from_fit(measurement, calibration, foil, provenance, &fit)?;
        Ok((pulse, fit))
    }

    fn from_fit(
        measurement: &Measurement,
        calibration: &Calibration,
        foil: Foil,
        provenance: Provenance,
        fit: &CountsFit,
    ) -> Result<Self, PipelineError> {
        let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
        let Some(covariance) = fit.covariance.as_ref().filter(|_| fit.converged) else {
            return invalid("a pulse calibration needs a converged fit".into());
        };
        let fitted: Vec<bool> = quantities(measurement, calibration)
            .map(|value| !matches!(value, Value::Known(_)))
            .collect();
        let first_number = fitted.len() - 6;
        let mut status = [Status::Known; 6];
        let mut resolved = Vec::with_capacity(6);
        for (number, name) in PULSE_NUMBERS.into_iter().enumerate() {
            let quantity = first_number + number;
            let i = fitted[..quantity].iter().filter(|&&f| f).count();
            if !fitted[quantity] {
                continue;
            }
            if fit.on_bound[i] {
                status[number] = Status::OnBound;
                continue;
            }
            let variance = covariance.get(i, i);
            if !(variance.is_finite() && variance > 0.0) {
                return invalid(format!(
                    "the calibration fitted {name} without determining it: its variance is \
                     {variance}"
                ));
            }
            status[number] = Status::Resolved;
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
        let line_span_ev = if fitted[first_number..].contains(&true) {
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
            Some(
                lines
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &e| {
                        (low.min(e), high.max(e))
                    }),
            )
        } else {
            None
        };
        let prior = if resolved.is_empty() {
            None
        } else {
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
            })
        };
        Ok(Self {
            numbers,
            status,
            prior,
            t0_us: fit.t0_us,
            flight_path_m: fit.flight_path_m,
            energy_span_ev: calibration.pulse.energy_span_ev,
            n_tau: calibration.pulse.n_tau,
            line_span_ev,
            foil,
            provenance,
            sample_overdispersion: fit.overdispersion[1],
            transfer: None,
        })
    }

    /// The agreement of this calibration's pulse with `other`'s, a physically
    /// different foil calibrated alone: `d² = dᵀ(C_a + C_b)⁻¹d` over the pulse
    /// numbers both resolve, with `d` the difference of their values and
    /// `C_a`, `C_b` their covariances, against `χ²` with as many degrees of
    /// freedom.  A number one foil holds on a bound and the other resolves is
    /// held for both: the other's numbers are conditioned on it at the bound,
    /// `d²` leaves out its own difference, and the result keeps the other's
    /// value and sd for it.
    ///
    /// `d²` follows `χ²` when both foils see the same pulse and every held
    /// bound is the truth.  Each run's information is divided by its
    /// overdispersion, which is at least 1, so with Poisson counts the test is
    /// conservative, and it weakens as the overdispersion grows.  Two foils
    /// that share an open-beam run, or whose stated temperatures share an
    /// error, are not independent, which the test does not account for.
    ///
    /// # Errors
    /// [`PipelineError::InvalidParameter`] if the two calibrations name the
    /// same foil, as the same foil re-measured tests only repeatability,
    /// share a sample run, know different pulse numbers or know one at
    /// different values, hold one on different bounds, or resolve no number
    /// in common;
    /// [`PipelineError::Fitting`] if a decomposition fails.
    pub fn transfer(&self, other: &PulseCalibration) -> Result<Transfer, PipelineError> {
        let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
        if self.provenance.foil == other.provenance.foil
            || self.provenance.sample == other.provenance.sample
        {
            return invalid(format!(
                "a transfer is to a physically different foil with its own sample run; got {:?} \
                 and {:?}",
                self.provenance, other.provenance
            ));
        }
        for (n, name) in PULSE_NUMBERS.into_iter().enumerate() {
            let (a, b) = (self.status[n], other.status[n]);
            let known = a == Status::Known || b == Status::Known;
            let held = known || (a == Status::OnBound && b == Status::OnBound);
            if held && (a != b || self.numbers[n] != other.numbers[n]) {
                return invalid(format!(
                    "the two calibrations hold {name} differently, {a:?} at {} and {b:?} at {}, \
                     so they describe different pulse models",
                    self.numbers[n], other.numbers[n]
                ));
            }
        }
        let resolved =
            |n: usize| self.status[n] == Status::Resolved && other.status[n] == Status::Resolved;
        if !(0..6).any(resolved) {
            return invalid("the two calibrations resolve no pulse number in common".into());
        }
        let estimate = |this: &Self, that: &Self| -> Result<Prior, PipelineError> {
            let prior = this
                .prior
                .as_ref()
                .expect("a calibration that resolves a number has its prior");
            let held: Vec<(usize, f64)> = (0..6)
                .filter(|&n| {
                    this.status[n] == Status::Resolved && that.status[n] == Status::OnBound
                })
                .map(|n| (n, that.numbers[n]))
                .collect();
            Ok(
                Prior::correlated(&prior.numbers, &prior.mean, &prior.covariance)?
                    .conditioned(&held)?,
            )
        };
        let result = agreement(&estimate(self, other)?, &estimate(other, self)?)?;
        let bounds = (0..6)
            .filter_map(|n| {
                let (held_by, holder, resolver) = match (self.status[n], other.status[n]) {
                    (Status::OnBound, Status::Resolved) => (Holder::This, self, other),
                    (Status::Resolved, Status::OnBound) => (Holder::Other, other, self),
                    _ => return None,
                };
                let prior = resolver.prior.as_ref()?;
                let k = prior.numbers.iter().position(|&m| m == n)?;
                Some(BoundNumber {
                    name: NUMBERS[n].0.into(),
                    held_by,
                    bound: holder.numbers[n],
                    value: resolver.numbers[n],
                    sd: prior.covariance.get(k, k).sqrt(),
                })
            })
            .collect();
        Ok(Transfer {
            record: TransferRecord {
                foil: other.foil.clone(),
                provenance: FileProvenance::from(&other.provenance),
                sample_overdispersion: other.sample_overdispersion,
                d2: result.q,
                dof: result.dof,
                p: result.p,
                bounds,
            },
        })
    }

    /// This calibration with its [`Self::transfer`] to `other` recorded, in
    /// place of "transfer unchecked" or an earlier transfer, when its `p` is
    /// above 0.01.
    ///
    /// # Errors
    /// Those of [`Self::transfer`]; [`PipelineError::InvalidParameter`] if `p`
    /// is 0.01 or less: the pulse does not transfer, and the calibration,
    /// consumed, cannot be written.
    pub fn record_transfer(mut self, other: &PulseCalibration) -> Result<Self, PipelineError> {
        let record = self.transfer(other)?.record;
        if record.p <= TRANSFER_P {
            return Err(PipelineError::InvalidParameter(format!(
                "the pulse does not transfer: d² = {} on {} degrees of freedom gives p = {}, at \
                 most {TRANSFER_P}; the calibration is not written",
                record.d2, record.dof, record.p
            )));
        }
        self.transfer = Some(record);
        Ok(self)
    }

    /// The calibration of an experiment: `t0` and the flight path fitted from
    /// the calibrated ones, the resolved pulse numbers fitted from theirs with
    /// their covariance as a prior, and the others known.
    pub fn calibration(&self) -> Calibration {
        let value = |number: usize| match self.status[number] {
            Status::Resolved => Value::Fitted(self.numbers[number]),
            Status::OnBound | Status::Known => Value::Known(self.numbers[number]),
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
                line_span_ev: self.line_span_ev,
                prior: self.prior.clone(),
            },
        }
    }

    /// The calibration file: JSON of format version 1 that names the pulse
    /// model and, in the order `α₀, α₁, β₀, β₁, R, h²`, each pulse number with
    /// its unit, value and status (resolved, on a bound or known), then the
    /// resolved numbers' covariance and rank, `t0` and the flight path, the
    /// pulse's energy span and `n_tau`, the line span, the foil's isotopes and
    /// effective temperature with their stated uncertainties, the foil and run
    /// identifiers, the sample run's overdispersion, and the transfer to
    /// another foil: `"unchecked"`, or the other foil and its identifiers, its
    /// sample run's overdispersion, `d2`, `dof`, `p` and each number one foil
    /// held on a bound with the other's value and sd.
    pub fn to_json(&self) -> String {
        let covariance = self.prior.as_ref().map_or_else(Vec::new, |prior| {
            let k = prior.numbers.len();
            (0..k)
                .map(|a| (0..k).map(|b| prior.covariance.get(a, b)).collect())
                .collect()
        });
        let file = File {
            format_version: FORMAT_VERSION,
            pulse_model: PULSE_MODEL.into(),
            numbers: NUMBERS
                .iter()
                .zip(self.numbers)
                .zip(self.status)
                .map(|((&(name, unit), value), status)| Number {
                    name: name.into(),
                    unit: unit.into(),
                    value,
                    status,
                })
                .collect(),
            rank: covariance.len(),
            covariance,
            t0_us: self.t0_us,
            flight_path_m: self.flight_path_m,
            energy_span_ev: self.energy_span_ev,
            n_tau: self.n_tau,
            line_span_ev: self.line_span_ev,
            foil: self.foil.clone(),
            provenance: FileProvenance::from(&self.provenance),
            sample_overdispersion: self.sample_overdispersion,
            transfer: self.transfer.clone().map_or(
                FileTransfer::Unchecked(Unchecked::Unchecked),
                FileTransfer::Checked,
            ),
        };
        serde_json::to_string_pretty(&file).expect("these field types always serialize")
    }

    /// The calibration a file from [`Self::to_json`] holds.
    ///
    /// # Errors
    /// [`PipelineError::InvalidParameter`] if `text` is not such a file: not
    /// format version 1 of this pulse model, a field missing or unknown, the
    /// pulse numbers' names or units not in their order, the rank not the
    /// number of resolved numbers, the line span missing while a pulse number
    /// was fitted, present when none was, or not `low ≤ high` within the
    /// energy span, a provenance [`Self::new`] refuses, the overdispersion
    /// below 1, the foil without isotopes, with one listed twice, or not as
    /// [`Self::new`] takes it, a value outside its quantity's range or one
    /// [`DetectorPulse::new`](nereids_physics::ikeda_carpenter::DetectorPulse::new)
    /// refuses, or a recorded transfer [`Self::record_transfer`] could not
    /// have written: to this calibration's foil or sample run, with a
    /// provenance or foil [`Self::new`] refuses, an overdispersion below 1,
    /// degrees of freedom not the number of pulse numbers both foils resolve,
    /// `d2` negative, `p` not within a relative 1e-12 of the χ² survival of
    /// `d2` or 0.01 or less, or a bound named twice, not a pulse number, not
    /// as this calibration holds or resolves it, or with the other foil's
    /// number outside its range; [`PipelineError::Fitting`]
    /// if [`Prior::correlated`] refuses the covariance.
    pub fn from_json(text: &str) -> Result<Self, PipelineError> {
        let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
        let file: File = match serde_json::from_str(text) {
            Ok(file) => file,
            Err(e) => return invalid(format!("not a pulse calibration file: {e}")),
        };
        if file.format_version != FORMAT_VERSION || file.pulse_model != PULSE_MODEL {
            return invalid(format!(
                "a pulse calibration file of format version {FORMAT_VERSION} and the pulse model \
                 {PULSE_MODEL:?} is read; got version {} and the model {:?}",
                file.format_version, file.pulse_model
            ));
        }
        if file.numbers.len() != NUMBERS.len()
            || NUMBERS
                .iter()
                .zip(&file.numbers)
                .any(|(&(name, unit), number)| number.name != name || number.unit != unit)
        {
            return invalid(format!(
                "the pulse numbers are (name, unit) {NUMBERS:?}, in that order"
            ));
        }
        let numbers: [f64; 6] = std::array::from_fn(|n| file.numbers[n].value);
        let status: [Status; 6] = std::array::from_fn(|n| file.numbers[n].status);
        let resolved: Vec<usize> = (0..6).filter(|&n| status[n] == Status::Resolved).collect();
        let k = resolved.len();
        if file.rank != k
            || file.covariance.len() != k
            || file.covariance.iter().any(|row| row.len() != k)
        {
            return invalid(format!(
                "the covariance is {k}×{k}, of rank {k}, over the {k} resolved numbers; got rank \
                 {} and {} rows",
                file.rank,
                file.covariance.len()
            ));
        }
        let fitted = status.iter().any(|&s| s != Status::Known);
        let (first, last) = file.energy_span_ev;
        let span_holds = match file.line_span_ev {
            Some((low, high)) => fitted && first <= low && low <= high && high <= last,
            None => !fitted,
        };
        if !span_holds {
            return invalid(format!(
                "the line span is present, within the energy span {:?}, exactly when a pulse \
                 number was fitted; got {:?}",
                file.energy_span_ev, file.line_span_ev
            ));
        }
        let provenance = Provenance::from(file.provenance);
        check_provenance(&provenance)?;
        if file
            .sample_overdispersion
            .is_some_and(|phi| !(phi.is_finite() && phi >= 1.0))
        {
            return invalid(format!(
                "the overdispersion is 1 or more; got {:?}",
                file.sample_overdispersion
            ));
        }
        check_foil(&file.foil)?;
        let prior = if resolved.is_empty() {
            None
        } else {
            let mut covariance = FlatMatrix::zeros(k, k);
            for (a, row) in file.covariance.iter().enumerate() {
                for (b, &entry) in row.iter().enumerate() {
                    *covariance.get_mut(a, b) = entry;
                }
            }
            let mean: Vec<f64> = resolved.iter().map(|&number| numbers[number]).collect();
            Prior::correlated(&resolved, &mean, &covariance)?;
            Some(PulsePrior {
                numbers: resolved,
                mean,
                covariance,
            })
        };
        let transfer = match file.transfer {
            FileTransfer::Unchecked(_) => None,
            FileTransfer::Checked(record) => {
                check_record(&record, &provenance, &status, &numbers, prior.as_ref())?;
                Some(record)
            }
        };
        let pulse = Self {
            numbers,
            status,
            prior,
            t0_us: file.t0_us,
            flight_path_m: file.flight_path_m,
            energy_span_ev: file.energy_span_ev,
            n_tau: file.n_tau,
            line_span_ev: file.line_span_ev,
            foil: file.foil,
            provenance,
            sample_overdispersion: file.sample_overdispersion,
            transfer,
        };
        let experiment = pulse.calibration();
        experiment.instrument()?;
        experiment.pulse.at(&numbers)?;
        Ok(pulse)
    }
}

const TRANSFER_P: f64 = 0.01;

/// The transfer of a pulse calibration to another, physically different foil
/// calibrated alone, from [`PulseCalibration::transfer`].
#[derive(Debug, Clone, PartialEq)]
pub struct Transfer {
    record: TransferRecord,
}

impl Transfer {
    /// `d²`, its degrees of freedom and `p`.
    pub fn agreement(&self) -> Consistency {
        Consistency {
            q: self.record.d2,
            dof: self.record.dof,
            p: self.record.p,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum FileTransfer {
    Unchecked(Unchecked),
    Checked(TransferRecord),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Unchecked {
    Unchecked,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Holder {
    This,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundNumber {
    name: String,
    held_by: Holder,
    bound: f64,
    value: f64,
    sd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransferRecord {
    foil: Foil,
    provenance: FileProvenance,
    #[serde(deserialize_with = "Option::deserialize")]
    sample_overdispersion: Option<f64>,
    d2: f64,
    dof: usize,
    p: f64,
    bounds: Vec<BoundNumber>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    format_version: u32,
    pulse_model: String,
    numbers: Vec<Number>,
    covariance: Vec<Vec<f64>>,
    rank: usize,
    t0_us: f64,
    flight_path_m: f64,
    energy_span_ev: (f64, f64),
    n_tau: usize,
    #[serde(deserialize_with = "Option::deserialize")]
    line_span_ev: Option<(f64, f64)>,
    foil: Foil,
    provenance: FileProvenance,
    #[serde(deserialize_with = "Option::deserialize")]
    sample_overdispersion: Option<f64>,
    transfer: FileTransfer,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Number {
    name: String,
    unit: String,
    value: f64,
    status: Status,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileProvenance {
    foil: String,
    open: String,
    sample: String,
}

impl From<&Provenance> for FileProvenance {
    fn from(provenance: &Provenance) -> Self {
        Self {
            foil: provenance.foil.clone(),
            open: provenance.open.clone(),
            sample: provenance.sample.clone(),
        }
    }
}

impl From<FileProvenance> for Provenance {
    fn from(file: FileProvenance) -> Self {
        Self {
            foil: file.foil,
            open: file.open,
            sample: file.sample,
        }
    }
}

fn check_record(
    record: &TransferRecord,
    provenance: &Provenance,
    status: &[Status; 6],
    numbers: &[f64; 6],
    prior: Option<&PulsePrior>,
) -> Result<(), PipelineError> {
    let other = Provenance::from(record.provenance.clone());
    check_provenance(&other)?;
    check_foil(&record.foil)?;
    let rank = prior.map_or(0, |prior| prior.numbers.len());
    let held_there = record
        .bounds
        .iter()
        .filter(|bound| bound.held_by == Holder::Other)
        .count();
    let statistic = rank.checked_sub(held_there) == Some(record.dof)
        && Consistency::new(record.d2, record.dof)
            .is_ok_and(|test| (record.p / test.p - 1.0).abs() <= 1e-12)
        && record.p > TRANSFER_P;
    let sd = |n: usize| {
        prior.and_then(|prior| {
            let k = prior.numbers.iter().position(|&m| m == n)?;
            Some(prior.covariance.get(k, k).sqrt())
        })
    };
    let mut named = Vec::with_capacity(record.bounds.len());
    let bounds = record.bounds.iter().all(|bound| {
        let Some(n) = NUMBERS.iter().position(|&(name, _)| name == bound.name) else {
            return false;
        };
        let once = !named.contains(&n);
        named.push(n);
        let range = &PULSE_RANGES[n].0;
        let in_range = range.contains(&bound.bound) && range.contains(&bound.value);
        let as_here = match bound.held_by {
            Holder::This => {
                status[n] == Status::OnBound && bound.bound == numbers[n] && bound.sd > 0.0
            }
            Holder::Other => bound.value == numbers[n] && Some(bound.sd) == sd(n),
        };
        once && in_range && as_here
    });
    if other.foil == provenance.foil
        || other.sample == provenance.sample
        || record.sample_overdispersion.is_some_and(|phi| phi < 1.0)
        || !statistic
        || !bounds
    {
        return Err(PipelineError::InvalidParameter(format!(
            "a recorded transfer is to another foil with its own sample run, on as many degrees \
             of freedom as numbers both foils resolve, with d² of 0 or more, p its χ² survival \
             and above {TRANSFER_P}, and each bound once as this calibration holds or resolves \
             it; got {record:?}"
        )));
    }
    Ok(())
}

fn check_provenance(provenance: &Provenance) -> Result<(), PipelineError> {
    if provenance.foil.is_empty()
        || provenance.open.is_empty()
        || provenance.sample.is_empty()
        || provenance.open == provenance.sample
    {
        return Err(PipelineError::InvalidParameter(format!(
            "a calibration names its foil and its open-beam and sample runs, the two runs \
             different; got {provenance:?}"
        )));
    }
    Ok(())
}

fn foil(measurement: &Measurement) -> Result<Foil, PipelineError> {
    let unmeasured = |what: &str, value: &Value| {
        Err(PipelineError::InvalidParameter(format!(
            "a calibration foil's {what} is measured, with its stated uncertainty; got {value:?}"
        )))
    };
    let mut isotopes = Vec::with_capacity(measurement.isotopes.len());
    for (data, density) in &measurement.isotopes {
        let density = match *density {
            Value::Measured { value, sd } => Stated {
                value,
                sd: Some(sd),
            },
            Value::Known(value) if value == 0.0 => Stated { value, sd: None },
            other => return unmeasured(&format!("density of {}", data.isotope), &other),
        };
        isotopes.push(FoilIsotope {
            isotope: data.isotope,
            density,
        });
    }
    let Value::Measured { value, sd } = measurement.temperature_k else {
        return unmeasured("temperature", &measurement.temperature_k);
    };
    Ok(Foil {
        isotopes,
        temperature_k: Stated {
            value,
            sd: Some(sd),
        },
    })
}

fn check_foil(foil: &Foil) -> Result<(), PipelineError> {
    let value = |stated: Stated| match stated.sd {
        Some(sd) => Value::Measured {
            value: stated.value,
            sd,
        },
        None => Value::Known(stated.value),
    };
    let isotopes = &foil.isotopes;
    if isotopes.is_empty()
        || isotopes
            .iter()
            .any(|isotope| isotope.density.sd.is_none() && isotope.density.value != 0.0)
        || (0..isotopes.len()).any(|i| {
            isotopes[..i]
                .iter()
                .any(|other| other.isotope == isotopes[i].isotope)
        })
        || foil.temperature_k.sd.is_none()
    {
        return Err(PipelineError::InvalidParameter(format!(
            "a calibration foil has distinct isotopes, each measured or known to be absent, and \
             a measured temperature; got {foil:?}"
        )));
    }
    for isotope in &foil.isotopes {
        value(isotope.density).parameter("density", 0.0..=f64::INFINITY, "0 or more")?;
    }
    let (low, high) = TEMPERATURE_BOUNDS_K;
    value(foil.temperature_k).parameter(
        "temperature",
        low..=high,
        &format!("within {low}–{high} K"),
    )?;
    Ok(())
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

    type Edit = (&'static str, fn(&mut serde_json::Value));

    fn measurement(absent: Option<f64>) -> Measurement {
        let lines = [(10.0, 0.05, 0.06), (25.0, 0.005, 0.06), (50.0, 0.01, 0.06)];
        let foil = (
            synthetic_isotope_multi(73, 181, &lines),
            Value::Measured {
                value: 2e-3,
                sd: 2e-5,
            },
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
            temperature_k: Value::Measured {
                value: 300.0,
                sd: 10.0,
            },
        }
    }

    fn provenance() -> Provenance {
        Provenance {
            foil: "foil-a".into(),
            open: "open-1".into(),
            sample: "sample-1".into(),
        }
    }

    fn calibrated(
        m: &Measurement,
        c: &Calibration,
        fit: CountsFit,
    ) -> Result<PulseCalibration, PipelineError> {
        PulseCalibration::from_fit(m, c, foil(m)?, provenance(), &fit)
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
                line_span_ev: None,
                prior: None,
            },
        }
    }

    fn fit(free: usize, bounded: &[usize]) -> CountsFit {
        let mut covariance = FlatMatrix::zeros(free, free);
        for i in 0..free {
            *covariance.get_mut(i, i) = 0.01 * (i + 1) as f64;
        }
        let (r, h) = (free - 2, free - 1);
        let pairs = [(4, r, 0.01), (4, h, -0.005), (r, h, 0.02), (2, 4, 0.003)];
        for (i, j, c) in pairs.into_iter().filter(|&(i, j, _)| i.max(j) < free) {
            *covariance.get_mut(i, j) = c;
            *covariance.get_mut(j, i) = c;
        }
        let mut on_bound = vec![false; free];
        for &i in bounded {
            on_bound[i] = true;
        }
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
            ..fit(8, &[5])
        };
        let experiment = calibrated(&m, &c, with_impurity).unwrap().calibration();
        let pulse = &experiment.pulse;
        assert_eq!(pulse.alpha, [Value::Known(0.5), Value::Fitted(1.1)]);
        assert_eq!(pulse.beta, [Value::Known(0.0), Value::Known(0.01)]);
        assert_eq!(
            [pulse.r, pulse.fwhm_squared_us2],
            [Value::Fitted(0.21), Value::Fitted(0.12)]
        );
        assert_eq!(
            [experiment.t0_us, experiment.flight_path_m],
            [Value::Fitted(3.0), Value::Fitted(25.0)]
        );
        let prior = pulse.prior.as_ref().expect("prior");
        assert_eq!(prior.numbers, [1, 4, 5]);
        assert_eq!(prior.mean, [1.1, 0.21, 0.12]);
        let covariance = fit(8, &[5]).covariance.expect("covariance");
        for (a, i) in [4, 6, 7].into_iter().enumerate() {
            for (b, j) in [4, 6, 7].into_iter().enumerate() {
                assert_eq!(prior.covariance.get(a, b), covariance.get(i, j));
            }
        }
        assert_eq!(pulse.line_span_ev, Some((10.0, 50.0)));

        let mut undetermined = fit(9, &[5]);
        let mut withheld = fit(8, &[5]);
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
                    ..fit(8, &[5])
                },
            ),
            (
                Value::Known(0.01),
                CountsFit {
                    covariance: None,
                    ..fit(8, &[5])
                },
            ),
        ] {
            assert!(calibrated(&m, &calibration(beta1), refused).is_err());
        }
    }

    #[test]
    fn a_calibration_without_lines_or_with_a_singular_block_is_refused() {
        let c = calibration(Value::Known(0.01));
        let mut beyond = measurement(None);
        beyond.isotopes[0].0 = synthetic_isotope(73, 181, 100.0, 0.05, 0.06);
        assert!(matches!(
            calibrated(&beyond, &c, fit(8, &[5])),
            Err(PipelineError::InvalidParameter(_))
        ));
        let mut singular = fit(8, &[5]);
        let covariance = singular.covariance.as_mut().expect("covariance");
        let (a, b) = (covariance.get(4, 4), covariance.get(6, 6));
        *covariance.get_mut(4, 6) = (a * b).sqrt();
        *covariance.get_mut(6, 4) = (a * b).sqrt();
        assert!(matches!(
            calibrated(&measurement(None), &c, singular),
            Err(PipelineError::Fitting(_))
        ));
    }

    #[test]
    fn only_isotopes_that_may_be_present_must_lie_within_the_calibrated_lines() {
        let pulse = Pulse {
            line_span_ev: Some((10.0, 50.0)),
            ..calibration(Value::Known(0.01)).pulse
        };
        let mut m = measurement(Some(55.0));
        let line =
            |m: &Measurement| pulse.uncalibrated_line(&m.isotopes, &m.time_edges_us, 3.0, 25.0);
        assert_eq!(line(&m), None);
        m.isotopes[1].1 = Value::Fitted(1e-4);
        assert_eq!(line(&m), Some(55.0));
    }

    #[test]
    fn a_calibration_with_every_fitted_number_on_a_bound_still_bounds_the_lines() {
        let c = calibration(Value::Known(0.01));
        let on_bounds = CountsFit {
            beta: [0.08, 0.01],
            ..fit(8, &[4, 5, 6, 7])
        };
        let held = calibrated(&measurement(None), &c, on_bounds)
            .unwrap()
            .calibration();
        assert!(held.pulse.prior.is_none());
        assert_eq!(held.pulse.line_span_ev, Some((10.0, 50.0)));
        let mut wider = measurement(Some(55.0));
        wider.isotopes[1].1 = Value::Fitted(1e-4);
        match fit_counts(&wider, &held) {
            Err(PipelineError::InvalidParameter(message)) => {
                assert!(message.contains("55 eV, outside the"), "{message}")
            }
            other => panic!("{other:?}"),
        }

        let mut known = c;
        let pulse = &mut known.pulse;
        pulse.alpha[1] = Value::Known(1.0);
        pulse.beta[0] = Value::Known(0.08);
        pulse.r = Value::Known(0.2);
        pulse.fwhm_squared_us2 = Value::Known(0.1);
        let unfitted = calibrated(&measurement(None), &known, fit(4, &[]))
            .unwrap()
            .calibration();
        assert_eq!(unfitted.pulse.line_span_ev, None);
    }

    fn original() -> PulseCalibration {
        let fit = CountsFit {
            r: 0.012_537_345_881_063_615,
            overdispersion: [Some(1.25), Some(1.5)],
            ..fit(8, &[5])
        };
        calibrated(
            &measurement(Some(55.0)),
            &calibration(Value::Known(0.01)),
            fit,
        )
        .unwrap()
    }

    fn file() -> String {
        original().to_json()
    }

    #[test]
    fn a_calibration_reads_back_from_its_file_bit_for_bit() {
        let original = original();
        let text = original.to_json();
        let read = PulseCalibration::from_json(&text).unwrap();
        assert_eq!(format!("{read:?}"), format!("{original:?}"));
        assert_eq!(read.to_json(), text);
        assert_eq!(
            read.numbers[4].to_bits(),
            0.012_537_345_881_063_615_f64.to_bits()
        );
        for expected in [
            "\"format_version\": 1",
            "\"name\": \"alpha0\"",
            "\"unit\": \"1/(µs·√eV)\"",
            "\"status\": \"on_bound\"",
            "\"rank\": 3",
            "\"sample_overdispersion\": 1.5",
            "\"transfer\": \"unchecked\"",
        ] {
            assert!(text.contains(expected), "{expected} in {text}");
        }
    }

    #[test]
    fn files_that_are_not_such_a_calibration_are_refused() {
        let original: serde_json::Value = serde_json::from_str(&file()).unwrap();
        let edits: [Edit; 20] = [
            ("version", |v| v["format_version"] = 2.into()),
            ("model", |v| v["pulse_model"] = "Gaussian".into()),
            ("transfer", |v| v["transfer"] = "checked".into()),
            ("unknown field", |v| v["comment"] = "added".into()),
            ("name", |v| v["numbers"][1]["name"] = "alpha2".into()),
            ("unit", |v| v["numbers"][2]["unit"] = "1/µs".into()),
            ("rank", |v| v["rank"] = 2.into()),
            ("row", |v| {
                v["covariance"].as_array_mut().unwrap().pop();
            }),
            ("span missing", |v| {
                v["line_span_ev"] = serde_json::Value::Null
            }),
            ("span reversed", |v| {
                v["line_span_ev"] = serde_json::json!([50.0, 10.0])
            }),
            ("runs", |v| v["provenance"]["sample"] = "open-1".into()),
            ("overdispersion", |v| {
                v["sample_overdispersion"] = 0.5.into()
            }),
            ("negative rate", |v| {
                v["numbers"][0]["value"] = (-0.01).into()
            }),
            ("missing key", |v| {
                v.as_object_mut().unwrap().remove("sample_overdispersion");
            }),
            ("nuclide key", |v| {
                v["foil"]["isotopes"][0]["isotope"]["n"] = 1.into();
            }),
            ("same isotope", |v| {
                let isotope = v["foil"]["isotopes"][0].clone();
                v["foil"]["isotopes"].as_array_mut().unwrap().push(isotope);
            }),
            ("temperature sd", |v| {
                v["foil"]["temperature_k"]["sd"] = serde_json::Value::Null;
            }),
            ("span beyond the pulse", |v| {
                v["line_span_ev"] = serde_json::json!([10.0, 300.0]);
            }),
            ("singular covariance", |v| {
                let c = &v["covariance"];
                let r = (c[0][0].as_f64().unwrap() * c[1][1].as_f64().unwrap()).sqrt();
                v["covariance"][0][1] = r.into();
                v["covariance"][1][0] = r.into();
            }),
            ("n_tau", |v| v["n_tau"] = 4.into()),
        ];
        for (what, edit) in edits {
            let mut value = original.clone();
            edit(&mut value);
            assert!(
                PulseCalibration::from_json(&value.to_string()).is_err(),
                "{what}"
            );
        }
        assert!(PulseCalibration::from_json(&original.to_string()).is_ok());
    }

    #[test]
    fn a_foil_whose_density_or_temperature_is_not_measured_is_refused() {
        let c = calibration(Value::Known(0.01));
        let mut fitted = measurement(None);
        fitted.isotopes[0].1 = Value::Fitted(2e-3);
        let mut warm = measurement(None);
        warm.temperature_k = Value::Fitted(300.0);
        for m in [fitted, warm] {
            match PulseCalibration::new(&m, &c, provenance()) {
                Err(PipelineError::InvalidParameter(message)) => {
                    assert!(message.contains("is measured"), "{message}")
                }
                other => panic!("{other:?}"),
            }
        }
        let same = Provenance {
            open: "sample-1".into(),
            ..provenance()
        };
        let unnamed = Provenance {
            foil: String::new(),
            ..provenance()
        };
        let mut measured = c.clone();
        measured.pulse.r = Value::Measured {
            value: 0.2,
            sd: 0.01,
        };
        for (c, provenance, refusal) in [
            (&c, same, "names its foil"),
            (&c, unnamed, "names its foil"),
            (&original().calibration(), provenance(), "calibrated alone"),
            (&measured, provenance(), "calibrated alone"),
        ] {
            match PulseCalibration::new(&measurement(None), c, provenance) {
                Err(PipelineError::InvalidParameter(message)) => {
                    assert!(message.contains(refusal), "{message}")
                }
                other => panic!("{other:?}"),
            }
        }
    }

    const VERSION_1: &str = r#"{
  "format_version": 1,
  "pulse_model": "Ikeda–Carpenter pulse with α = α₀√E + α₁ and β = β₀√E + β₁ in 1/µs, E in eV, a storage fraction R constant over the energy span, folded with the proton pulse's triangle of FWHM h",
  "numbers": [
    {
      "name": "alpha0",
      "unit": "1/(µs·√eV)",
      "value": 0.5,
      "status": "known"
    },
    {
      "name": "alpha1",
      "unit": "1/µs",
      "value": 1.1,
      "status": "resolved"
    },
    {
      "name": "beta0",
      "unit": "1/(µs·√eV)",
      "value": 0.0,
      "status": "on_bound"
    },
    {
      "name": "beta1",
      "unit": "1/µs",
      "value": 0.01,
      "status": "known"
    },
    {
      "name": "r",
      "unit": "1",
      "value": 0.012537345881063615,
      "status": "resolved"
    },
    {
      "name": "fwhm_squared",
      "unit": "µs²",
      "value": 0.12,
      "status": "resolved"
    }
  ],
  "covariance": [
    [
      0.05,
      0.01,
      -0.005
    ],
    [
      0.01,
      0.07,
      0.02
    ],
    [
      -0.005,
      0.02,
      0.08
    ]
  ],
  "rank": 3,
  "t0_us": 3.0,
  "flight_path_m": 25.0,
  "energy_span_ev": [
    1.0,
    200.0
  ],
  "n_tau": 256,
  "line_span_ev": [
    10.0,
    50.0
  ],
  "foil": {
    "isotopes": [
      {
        "isotope": {
          "z": 73,
          "a": 181
        },
        "density": {
          "value": 0.002,
          "sd": 0.00002
        }
      },
      {
        "isotope": {
          "z": 74,
          "a": 184
        },
        "density": {
          "value": 0.0,
          "sd": null
        }
      }
    ],
    "temperature_k": {
      "value": 300.0,
      "sd": 10.0
    }
  },
  "provenance": {
    "foil": "foil-a",
    "open": "open-1",
    "sample": "sample-1"
  },
  "sample_overdispersion": 1.5,
  "transfer": "unchecked"
}"#;

    #[test]
    fn the_file_is_format_version_1_byte_for_byte() {
        let original = original();
        assert_eq!(original.to_json(), VERSION_1);
        let read = PulseCalibration::from_json(VERSION_1).unwrap();
        assert_eq!(format!("{read:?}"), format!("{original:?}"));
    }

    fn other_foil(density: f64) -> Measurement {
        let mut m = measurement(None);
        m.isotopes[0].1 = Value::Measured {
            value: density,
            sd: 0.01 * density,
        };
        m
    }

    fn other_provenance() -> Provenance {
        Provenance {
            foil: "foil-b".into(),
            open: "open-2".into(),
            sample: "sample-2".into(),
        }
    }

    fn other(alpha1: f64, alpha0: f64, bounded: &[usize]) -> PulseCalibration {
        let m = other_foil(3e-3);
        let mut fit = CountsFit {
            alpha: [alpha0, alpha1],
            beta: [0.004, 0.01],
            r: 0.205,
            fwhm_squared_us2: 0.11,
            ..fit(8, bounded)
        };
        let covariance = fit.covariance.as_mut().expect("covariance");
        for (i, j, c) in [(4, 5, 0.02), (5, 6, 0.015)] {
            *covariance.get_mut(i, j) = c;
            *covariance.get_mut(j, i) = c;
        }
        let c = calibration(Value::Known(0.01));
        PulseCalibration::from_fit(&m, &c, foil(&m).unwrap(), other_provenance(), &fit).unwrap()
    }

    fn inverse3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let cofactor = |i: usize, j: usize| {
            let (r, c) = ([(i + 1) % 3, (i + 2) % 3], [(j + 1) % 3, (j + 2) % 3]);
            m[r[0]][c[0]] * m[r[1]][c[1]] - m[r[0]][c[1]] * m[r[1]][c[0]]
        };
        let determinant: f64 = (0..3).map(|j| m[0][j] * cofactor(0, j)).sum();
        std::array::from_fn(|i| std::array::from_fn(|j| cofactor(j, i) / determinant))
    }

    #[test]
    fn a_transfer_compares_the_numbers_both_foils_resolve_given_the_bounds_either_holds() {
        let a = original();
        let b = other(1.15, 0.5, &[]);
        let transfer = a.transfer(&b).unwrap();
        let (beta0, b_beta0, var_beta0) = (0.0, 0.004, 0.06);
        let cross = [0.02, 0.015, 0.0];
        let mean_b = [1.15, 0.205, 0.11];
        let mean_a = [1.1, 0.012_537_345_881_063_615, 0.12];
        let c_a = [
            [0.05, 0.01, -0.005],
            [0.01, 0.07, 0.02],
            [-0.005, 0.02, 0.08],
        ];
        let c_b = c_a;
        let conditioned: [f64; 3] =
            std::array::from_fn(|i| mean_b[i] + cross[i] * (beta0 - b_beta0) / var_beta0);
        let sum: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| c_a[i][j] + c_b[i][j] - cross[i] * cross[j] / var_beta0)
        });
        let weight = inverse3(sum);
        let d: [f64; 3] = std::array::from_fn(|i| mean_a[i] - conditioned[i]);
        let q: f64 = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| d[i] * weight[i][j] * d[j])
            .sum();
        let result = transfer.agreement();
        assert_eq!(result.dof, 3);
        assert!((result.q / q - 1.0).abs() <= 1e-12, "{} vs {q}", result.q);
        assert_eq!(result, Consistency::new(result.q, 3).unwrap());
        let bound = |held_by| BoundNumber {
            name: "beta0".into(),
            held_by,
            bound: 0.0,
            value: 0.004,
            sd: var_beta0.sqrt(),
        };
        assert_eq!(transfer.record.bounds, [bound(Holder::This)]);
        let mirror = b.transfer(&a).unwrap();
        assert!((mirror.agreement().q / q - 1.0).abs() <= 1e-12);
        assert_eq!(mirror.record.bounds, [bound(Holder::Other)]);
    }

    #[test]
    fn a_transfer_needs_another_foil_with_the_same_pulse_model() {
        let a = original();
        let mut same_foil = other(1.15, 0.5, &[]);
        same_foil.provenance.foil = "foil-a".into();
        let mut shared = other(1.15, 0.5, &[]);
        shared.provenance.sample = "sample-1".into();
        let mut disjoint = other(1.15, 0.5, &[4, 5, 6, 7]);
        disjoint.numbers[2] = 0.0;
        for (b, refusal) in [
            (same_foil, "physically different foil"),
            (shared, "physically different foil"),
            (other(1.15, 0.6, &[]), "different pulse models"),
            (disjoint, "resolve no pulse number in common"),
        ] {
            match a.transfer(&b) {
                Err(PipelineError::InvalidParameter(message)) => {
                    assert!(message.contains(refusal), "{message}")
                }
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn only_a_transfer_that_passes_is_recorded_and_read_back() {
        let failing = other(3.0, 0.5, &[]);
        assert!(original().transfer(&failing).unwrap().agreement().p <= 0.01);
        match original().record_transfer(&failing) {
            Err(PipelineError::InvalidParameter(message)) => {
                assert!(message.contains("does not transfer"), "{message}")
            }
            other => panic!("{other:?}"),
        }

        let b = other(1.15, 0.5, &[]);
        let a = original().record_transfer(&b).unwrap();
        assert_eq!(a.transfer, Some(original().transfer(&b).unwrap().record));
        let b = b.record_transfer(&original()).unwrap();
        let held_here: [Edit; 4] = [
            ("held by the other", |v| {
                v["transfer"]["bounds"][0]["held_by"] = "other".into();
            }),
            ("not this bound", |v| {
                v["transfer"]["bounds"][0]["bound"] = 0.001.into();
            }),
            ("sd not positive", |v| {
                v["transfer"]["bounds"][0]["sd"] = 0.0.into();
            }),
            ("other value out of range", |v| {
                v["transfer"]["bounds"][0]["value"] = (-0.004).into();
            }),
        ];
        let held_there: [Edit; 4] = [
            ("held by this", |v| {
                let bound = &mut v["transfer"]["bounds"][0];
                bound["held_by"] = "this".into();
                bound["bound"] = bound["value"].clone();
                let d2 = v["transfer"]["d2"].as_f64().unwrap();
                v["transfer"]["dof"] = 4.into();
                v["transfer"]["p"] = Consistency::new(d2, 4).unwrap().p.into();
            }),
            ("not this value", |v| {
                v["transfer"]["bounds"][0]["value"] = 0.005.into();
            }),
            ("not this sd", |v| {
                v["transfer"]["bounds"][0]["sd"] = 0.2.into();
            }),
            ("other bound out of range", |v| {
                v["transfer"]["bounds"][0]["bound"] = (-1.0).into();
            }),
        ];
        for (c, held) in [(a, &held_here[..]), (b, &held_there[..])] {
            refuses_edited_records(&c, held);
        }
    }

    fn refuses_edited_records(c: &PulseCalibration, held: &[Edit]) {
        let text = c.to_json();
        let read = PulseCalibration::from_json(&text).unwrap();
        assert_eq!(format!("{read:?}"), format!("{c:?}"));
        assert_eq!(read.to_json(), text);

        let original: serde_json::Value = serde_json::from_str(&text).unwrap();
        let mut rounded = original.clone();
        let p = rounded["transfer"]["p"].as_f64().unwrap();
        rounded["transfer"]["p"] = (p * (1.0 + 1e-13)).into();
        assert!(PulseCalibration::from_json(&rounded.to_string()).is_ok());
        let edits: [Edit; 10] = [
            ("p not its d2's", |v| v["transfer"]["p"] = 0.5.into()),
            ("failed", |v| {
                v["transfer"]["d2"] = 50.0.into();
                v["transfer"]["p"] = Consistency::new(50.0, 3).unwrap().p.into();
            }),
            ("dof not the numbers both resolve", |v| {
                let d2 = v["transfer"]["d2"].as_f64().unwrap();
                v["transfer"]["dof"] = 4.into();
                v["transfer"]["p"] = Consistency::new(d2, 4).unwrap().p.into();
            }),
            ("same foil", |v| {
                v["transfer"]["provenance"]["foil"] = v["provenance"]["foil"].clone();
            }),
            ("same sample run", |v| {
                v["transfer"]["provenance"]["sample"] = v["provenance"]["sample"].clone();
            }),
            ("bound name", |v| {
                v["transfer"]["bounds"][0]["name"] = "gamma".into();
            }),
            ("bound twice", |v| {
                let bound = v["transfer"]["bounds"][0].clone();
                v["transfer"]["bounds"].as_array_mut().unwrap().push(bound);
            }),
            ("other overdispersion", |v| {
                v["transfer"]["sample_overdispersion"] = 0.5.into();
            }),
            ("other runs the same", |v| {
                v["transfer"]["provenance"]["open"] = v["transfer"]["provenance"]["sample"].clone();
            }),
            ("other foil", |v| {
                v["transfer"]["foil"]["isotopes"] = serde_json::json!([]);
            }),
        ];
        for (what, edit) in edits.iter().chain(held) {
            let mut value = original.clone();
            edit(&mut value);
            assert!(
                PulseCalibration::from_json(&value.to_string()).is_err(),
                "{what}"
            );
        }
    }
}
