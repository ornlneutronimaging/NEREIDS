//! A pulse calibrated on a foil, carried to experiments as a correlated
//! prior on the pulse numbers the foil fitted.

use nereids_core::types::Isotope;
use nereids_endf::resonance::ResonanceData;
use nereids_fitting::lm::FlatMatrix;
use nereids_fitting::poisson::Prior;
use nereids_fitting::statistics::{Consistency, agreement};
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::resonance_center_energies;
use serde::{Deserialize, Serialize};

use crate::counts_fit::{CountsFit, Material, Measurement, Region, Value, fit_counts, quantities};
use crate::error::PipelineError;
use crate::open_beam::{Calibration, PULSE_NUMBERS, Pulse};
use crate::pipeline::TEMPERATURE_BOUNDS_K;

/// The pulse numbers a calibration fitted, as indices into
/// `(α₀, α₁, β₀, β₁, R, h²)`, with the mean and covariance of its likelihood
/// in them without their bounds.
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

const FORMAT_VERSION: u32 = 2;

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
    Fitted,
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
/// A pulse number the calibration fitted keeps its fitted value and enters
/// the prior; a number known in the calibration stays known.  The prior is
/// the calibration's [`CountsFit::unbounded`] Gaussian in the fitted numbers,
/// marginal over the other quantities, so a number that ended on a bound
/// keeps its uncertainty and the experiment fit applies the bound.  The
/// bounds of the calibration foil's other quantities do not reach the prior
/// either: its mean and covariance are those without them.  When the
/// calibration fitted a pulse number, an experiment's pulse carries the
/// foil's [`line_span_ev`](Pulse::line_span_ev).  When an experiment ends a
/// calibrated pulse number on its bound, or near it, its error bars on `t0`,
/// the flight path and the pulse numbers are not standard errors; those on
/// the densities and the temperature are.
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
    /// from `provenance`, with `calibration`, and that fit.  The measurement
    /// is the foil alone, one region holding it.  Each isotope of the foil
    /// not known to be absent (`Value::Known(0.0)`) has a measured density,
    /// and the foil's effective temperature is measured.
    ///
    /// # Errors
    /// [`PipelineError::InvalidParameter`] if the measurement is not one
    /// region holding a material, a density or the temperature of the foil is
    /// not measured, as above, an identifier of `provenance` is
    /// empty or its runs are the same, or `calibration`'s pulse carries a
    /// prior or a pulse number measured or boxed, since a calibration foil is
    /// calibrated alone over the numbers' physical ranges; everything
    /// [`fit_counts`] refuses;
    /// [`PipelineError::InvalidParameter`] if the fit did not converge, or
    /// fitted a pulse number and gives no [`CountsFit::unbounded`] Gaussian, a
    /// pulse number it fitted has no finite positive variance without its
    /// bounds, as when the counts do not determine it, a number that ended on
    /// a bound carries no information, or a fitted temperature ended at 1 K or
    /// 5000 K,
    /// or the fit fitted a pulse number and no isotope of the foil fitted or
    /// known to a positive density has a resonance between the energies of its
    /// last and first time edges at the fitted `t0` and flight path;
    /// [`PipelineError::Fitting`] if the fitted numbers' covariance is refused
    /// by [`Prior::correlated`].
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
            .any(|value| matches!(value, Value::Measured { .. } | Value::Within { .. }));
        if pulse.prior.is_some() || measured {
            return Err(PipelineError::InvalidParameter(
                "a calibration foil is calibrated alone, its pulse numbers fitted over their \
                 physical ranges or known, with no pulse prior"
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
        if !fit.converged {
            return invalid("a pulse calibration needs a converged fit".into());
        }
        let fitted: Vec<bool> = quantities(measurement, calibration)
            .map(|value| !matches!(value, Value::Known(_)))
            .collect();
        let first_number = fitted.len() - 6;
        let mut status = [Status::Known; 6];
        let mut covered = Vec::with_capacity(6);
        for (number, name) in PULSE_NUMBERS.into_iter().enumerate() {
            let quantity = first_number + number;
            let i = fitted[..quantity].iter().filter(|&&f| f).count();
            if !fitted[quantity] {
                continue;
            }
            let Some(unbounded) = fit.unbounded.as_ref() else {
                return invalid(
                    "the calibration's fit gives no Gaussian without its bounds, as when a bin \
                     predicted zero pulls a quantity on its bound"
                        .into(),
                );
            };
            let variance = unbounded.covariance.get(i, i);
            if !(variance.is_finite() && variance > 0.0) {
                return invalid(format!(
                    "the calibration fitted {name} without determining it: its variance is \
                     {variance}"
                ));
            }
            status[number] = Status::Fitted;
            covered.push((number, i));
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
            let present = foil_material(measurement)?
                .isotopes
                .iter()
                .zip(&fit.regions[0].densities)
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
        let prior = if let Some(unbounded) = fit.unbounded.as_ref().filter(|_| !covered.is_empty())
        {
            let n = covered.len();
            let covariance = &unbounded.covariance;
            let mut block = FlatMatrix::zeros(n, n);
            for (a, &(_, i)) in covered.iter().enumerate() {
                for (b, &(_, j)) in covered.iter().enumerate() {
                    *block.get_mut(a, b) = 0.5 * (covariance.get(i, j) + covariance.get(j, i));
                }
            }
            let numbers: Vec<usize> = covered.iter().map(|&(number, _)| number).collect();
            let mean: Vec<f64> = covered.iter().map(|&(_, i)| unbounded.mean[i]).collect();
            Prior::correlated(&numbers, &mean, &block)?;
            Some(PulsePrior {
                numbers,
                mean,
                covariance: block,
            })
        } else {
            None
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
            sample_overdispersion: fit.regions[0].overdispersion[1],
            transfer: None,
        })
    }

    /// The agreement of this calibration's pulse with `other`'s, a physically
    /// different foil calibrated alone: `d² = dᵀ(C_a + C_b)⁻¹d` over the pulse
    /// numbers both fitted, with `d` the difference of their priors' means and
    /// `C_a`, `C_b` their covariances, against `χ²` with as many degrees of
    /// freedom.
    ///
    /// `d²` follows `χ²` when both foils see the same pulse, to the order the
    /// priors are Gaussian.  Each run's information is divided by its
    /// overdispersion, which is at least 1, so with Poisson counts the test is
    /// conservative, and it weakens as the overdispersion grows.  Two foils
    /// that share an open-beam run, or whose stated temperatures share an
    /// error, are not independent, which the test does not account for.
    ///
    /// # Errors
    /// [`PipelineError::InvalidParameter`] if the two calibrations name the
    /// same foil, as the same foil re-measured tests only repeatability,
    /// share a run other than the open-beam run, fit different pulse numbers
    /// or know one at different values, or fit none;
    /// [`PipelineError::Fitting`] if a decomposition fails.
    pub fn transfer(&self, other: &PulseCalibration) -> Result<Transfer, PipelineError> {
        let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
        if !separate(&self.provenance, &other.provenance) {
            return invalid(format!(
                "a transfer is to a physically different foil that shares no run but the \
                 open-beam run; got {:?} and {:?}",
                self.provenance, other.provenance
            ));
        }
        for (n, name) in PULSE_NUMBERS.into_iter().enumerate() {
            let (a, b) = (self.status[n], other.status[n]);
            if a != b || (a == Status::Known && self.numbers[n] != other.numbers[n]) {
                return invalid(format!(
                    "the two calibrations hold {name} differently, {a:?} at {} and {b:?} at {}, \
                     so they describe different pulse models",
                    self.numbers[n], other.numbers[n]
                ));
            }
        }
        let (Some(this), Some(that)) = (&self.prior, &other.prior) else {
            return invalid("the two calibrations fit no pulse number".into());
        };
        let prior = |p: &PulsePrior| Prior::correlated(&p.numbers, &p.mean, &p.covariance);
        let result = agreement(&prior(this)?, &prior(that)?)?;
        Ok(Transfer {
            record: TransferRecord {
                foil: other.foil.clone(),
                provenance: FileProvenance::from(&other.provenance),
                sample_overdispersion: other.sample_overdispersion,
                d2: result.q,
                dof: result.dof,
                p: result.p,
            },
        })
    }

    /// This calibration with its [`Self::transfer`] to `other` recorded, in
    /// place of "transfer unchecked" or an earlier transfer, when its `p` is
    /// above 0.01.
    ///
    /// # Errors
    /// Those of [`Self::transfer`]; [`PipelineError::InvalidParameter`] if `p`
    /// is 0.01 or less: the pulse does not transfer, and the calibration is
    /// consumed, since it is not to be written.
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
    /// the calibrated ones, the fitted pulse numbers fitted from theirs under
    /// the calibration's prior, and the others known.
    pub fn calibration(&self) -> Calibration {
        let value = |number: usize| match self.status[number] {
            Status::Fitted => Value::Fitted(self.numbers[number]),
            Status::Known => Value::Known(self.numbers[number]),
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

    /// The calibration file: JSON of format version 2 that names the pulse
    /// model and, in the order `α₀, α₁, β₀, β₁, R, h²`, each pulse number with
    /// its unit, value and status (fitted or known), then the prior over the
    /// fitted numbers: its mean, which may lie past a number's bound, its
    /// covariance and rank; then `t0` and the flight path, the pulse's energy
    /// span and `n_tau`, the line span, the foil's isotopes and effective
    /// temperature with their stated uncertainties, the foil and run
    /// identifiers, the sample run's overdispersion, and the transfer to
    /// another foil: `"unchecked"`, or the other foil and its identifiers, its
    /// sample run's overdispersion, `d2`, `dof` and `p`.
    pub fn to_json(&self) -> String {
        let covariance = self.prior.as_ref().map_or_else(Vec::new, |prior| {
            let k = prior.numbers.len();
            (0..k)
                .map(|a| (0..k).map(|b| prior.covariance.get(a, b)).collect())
                .collect()
        });
        let mean = self
            .prior
            .as_ref()
            .map_or_else(Vec::new, |prior| prior.mean.clone());
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
            mean,
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
    /// format version 2 of this pulse model, a field missing or unknown, the
    /// pulse numbers' names or units not in their order, the mean, the
    /// covariance or the rank not over the fitted numbers, the line span
    /// missing while a pulse number
    /// was fitted, present when none was, or not `low ≤ high` within the
    /// energy span, a provenance [`Self::new`] refuses, the overdispersion
    /// below 1, the foil without isotopes, with one listed twice, or not as
    /// [`Self::new`] takes it, a value outside its quantity's range or one
    /// [`DetectorPulse::new`](nereids_physics::ikeda_carpenter::DetectorPulse::new)
    /// refuses, or a recorded transfer [`Self::record_transfer`] could not
    /// have written: to this calibration's foil or sharing a run other than
    /// the open-beam run, with a
    /// provenance or foil [`Self::new`] refuses, an overdispersion below 1,
    /// degrees of freedom not the number of fitted numbers, `d2` negative, or
    /// `p` not within a relative 1e-12 of the χ² survival of `d2` or 0.01 or
    /// less; [`PipelineError::Fitting`] if [`Prior::correlated`] refuses the
    /// mean or the covariance.
    pub fn from_json(text: &str) -> Result<Self, PipelineError> {
        let invalid = |message: String| Err(PipelineError::InvalidParameter(message));
        let header: Header = match serde_json::from_str(text) {
            Ok(header) => header,
            Err(e) => return invalid(format!("not a pulse calibration file: {e}")),
        };
        if header.format_version != FORMAT_VERSION || header.pulse_model != PULSE_MODEL {
            return invalid(format!(
                "a pulse calibration file of format version {FORMAT_VERSION} and the pulse model \
                 {PULSE_MODEL:?} is read; got version {} and the model {:?}",
                header.format_version, header.pulse_model
            ));
        }
        let file: File = match serde_json::from_str(text) {
            Ok(file) => file,
            Err(e) => return invalid(format!("not a pulse calibration file: {e}")),
        };
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
        let covered: Vec<usize> = (0..6).filter(|&n| status[n] == Status::Fitted).collect();
        let k = covered.len();
        if file.rank != k
            || file.mean.len() != k
            || file.covariance.len() != k
            || file.covariance.iter().any(|row| row.len() != k)
        {
            return invalid(format!(
                "the mean has {k} entries and the covariance is {k}×{k}, of rank {k}, over the {k} \
                 fitted numbers; got rank {}, {} entries and {} rows",
                file.rank,
                file.mean.len(),
                file.covariance.len()
            ));
        }
        let fitted = k > 0;
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
        let prior = if fitted {
            let mut covariance = FlatMatrix::zeros(k, k);
            for (a, row) in file.covariance.iter().enumerate() {
                for (b, &entry) in row.iter().enumerate() {
                    *covariance.get_mut(a, b) = entry;
                }
            }
            Prior::correlated(&covered, &file.mean, &covariance)?;
            Some(PulsePrior {
                numbers: covered,
                mean: file.mean,
                covariance,
            })
        } else {
            None
        };
        let transfer = match file.transfer {
            FileTransfer::Unchecked(_) => None,
            FileTransfer::Checked(record) => {
                check_record(&record, &provenance, k)?;
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
}

#[derive(Deserialize)]
struct Header {
    format_version: u32,
    pulse_model: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    format_version: u32,
    pulse_model: String,
    numbers: Vec<Number>,
    mean: Vec<f64>,
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
    rank: usize,
) -> Result<(), PipelineError> {
    let other = Provenance::from(record.provenance.clone());
    check_provenance(&other)?;
    check_foil(&record.foil)?;
    let statistic = record.dof == rank
        && Consistency::new(record.d2, record.dof)
            .is_ok_and(|test| (record.p / test.p - 1.0).abs() <= 1e-12)
        && record.p > TRANSFER_P;
    if !separate(provenance, &other)
        || record
            .sample_overdispersion
            .is_some_and(|phi| !(phi.is_finite() && phi >= 1.0))
        || !statistic
    {
        return Err(PipelineError::InvalidParameter(format!(
            "a recorded transfer is to another foil sharing no run but the open-beam run, on as \
             many degrees of freedom as numbers fitted, with d² of 0 or more and p its χ² \
             survival and above {TRANSFER_P}; got {record:?}"
        )));
    }
    Ok(())
}

fn separate(a: &Provenance, b: &Provenance) -> bool {
    a.foil != b.foil && a.sample != b.sample && a.sample != b.open && a.open != b.sample
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

fn foil_material(measurement: &Measurement) -> Result<&Material, PipelineError> {
    let [
        Region {
            material: Some(material),
            ..
        },
    ] = measurement.regions.as_slice()
    else {
        return Err(PipelineError::InvalidParameter(
            "a pulse calibration fits the foil alone, as one region holding it".into(),
        ));
    };
    Ok(material)
}

fn foil(measurement: &Measurement) -> Result<Foil, PipelineError> {
    let material = foil_material(measurement)?;
    let unmeasured = |what: &str, value: &Value| {
        Err(PipelineError::InvalidParameter(format!(
            "a calibration foil's {what} is measured, with its stated uncertainty; got {value:?}"
        )))
    };
    let mut isotopes = Vec::with_capacity(material.isotopes.len());
    for (data, density) in &material.isotopes {
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
    let Value::Measured { value, sd } = material.temperature_k else {
        return unmeasured("temperature", &material.temperature_k);
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

    use nereids_fitting::poisson::Unbounded;

    use super::*;
    use crate::beam::BeamSpline;
    use crate::counts_fit::RegionFit;

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
            charge_ratio: 1.0,
            normalization: Value::Known(1.0),
            regions: vec![Region {
                open_counts: vec![100.0; bins],
                sample_counts: vec![100.0; bins],
                open_live: None,
                sample_live: None,
                background: [Value::Known(0.0); 3],
                material: Some(Material {
                    isotopes: std::iter::once(foil).chain(impurity).collect(),
                    temperature_k: Value::Measured {
                        value: 300.0,
                        sd: 10.0,
                    },
                }),
            }],
        }
    }

    fn material(m: &mut Measurement) -> &mut Material {
        m.regions[0].material.as_mut().expect("the foil")
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

    fn fit(free: usize) -> CountsFit {
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
        let unbounded = Unbounded {
            mean: (0..free).map(|i| -0.002 * (i + 1) as f64).collect(),
            covariance: FlatMatrix {
                data: covariance.data.iter().map(|c| 2.0 * c).collect(),
                ..covariance.clone()
            },
        };
        CountsFit {
            regions: vec![RegionFit {
                densities: vec![2e-3],
                temperature_k: Some(300.0),
                background: [0.0; 3],
                beam: BeamSpline::constant(200.0, 600.0, 1.0),
                beam_at_limit: false,
                predicted: [Vec::new(), Vec::new()],
                overdispersion: [None; 2],
            }],
            normalization: 1.0,
            t0_us: 3.0,
            flight_path_m: 25.0,
            alpha: [0.5, 1.1],
            beta: [0.0, 0.01],
            r: 0.21,
            fwhm_squared_us2: 0.12,
            covariance: Some(covariance),
            unbounded: Some(unbounded),
            on_bound: vec![false; free],
            deviance: 0.0,
            converged: true,
            measured_pulls: None,
            pulse_consistency: None,
            step_us: 0.1,
            points: 10,
            halvings: 0,
        }
    }

    #[test]
    fn a_calibration_keeps_every_fitted_number_with_its_unbounded_uncertainty() {
        let (m, c) = (measurement(Some(55.0)), calibration(Value::Known(0.01)));
        let mut with_impurity = fit(8);
        with_impurity.regions[0].densities = vec![2e-3, 0.0];
        let experiment = calibrated(&m, &c, with_impurity).unwrap().calibration();
        let pulse = &experiment.pulse;
        assert_eq!(pulse.alpha, [Value::Known(0.5), Value::Fitted(1.1)]);
        assert_eq!(pulse.beta, [Value::Fitted(0.0), Value::Known(0.01)]);
        assert_eq!(
            [pulse.r, pulse.fwhm_squared_us2],
            [Value::Fitted(0.21), Value::Fitted(0.12)]
        );
        assert_eq!(
            [experiment.t0_us, experiment.flight_path_m],
            [Value::Fitted(3.0), Value::Fitted(25.0)]
        );
        let prior = pulse.prior.as_ref().expect("prior");
        assert_eq!(prior.numbers, [1, 2, 4, 5]);
        let unbounded = fit(8).unbounded.expect("unbounded");
        let at = [4, 5, 6, 7];
        assert_eq!(prior.mean, at.map(|i| unbounded.mean[i]));
        for (a, i) in at.into_iter().enumerate() {
            for (b, j) in at.into_iter().enumerate() {
                assert_eq!(prior.covariance.get(a, b), unbounded.covariance.get(i, j));
            }
        }
        assert_eq!(pulse.line_span_ev, Some((10.0, 50.0)));

        let mut undetermined = fit(8);
        let mut withheld = fit(8);
        let covariance = &mut undetermined
            .unbounded
            .as_mut()
            .expect("unbounded")
            .covariance;
        for i in 0..8 {
            *covariance.get_mut(5, i) = f64::NAN;
            *covariance.get_mut(i, 5) = f64::NAN;
        }
        withheld
            .unbounded
            .as_mut()
            .expect("unbounded")
            .covariance
            .data
            .fill(f64::NAN);
        for (refused, refusal) in [
            (undetermined, "without determining it"),
            (withheld, "without determining it"),
            (
                CountsFit {
                    converged: false,
                    ..fit(8)
                },
                "converged fit",
            ),
            (
                CountsFit {
                    unbounded: None,
                    ..fit(8)
                },
                "no Gaussian without its bounds",
            ),
        ] {
            match calibrated(&m, &c, refused) {
                Err(PipelineError::InvalidParameter(message)) => {
                    assert!(message.contains(refusal), "{message}")
                }
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn a_calibration_without_lines_or_with_a_singular_block_is_refused() {
        let c = calibration(Value::Known(0.01));
        let mut beyond = measurement(None);
        material(&mut beyond).isotopes[0].0 = synthetic_isotope(73, 181, 100.0, 0.05, 0.06);
        assert!(matches!(
            calibrated(&beyond, &c, fit(8)),
            Err(PipelineError::InvalidParameter(_))
        ));
        let mut singular = fit(8);
        let covariance = &mut singular.unbounded.as_mut().expect("unbounded").covariance;
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
        let line = |m: &Measurement| {
            let isotopes = &foil_material(m).expect("the foil").isotopes;
            pulse.uncalibrated_line(isotopes, &m.time_edges_us, 3.0, 25.0)
        };
        assert_eq!(line(&m), None);
        material(&mut m).isotopes[1].1 = Value::Fitted(1e-4);
        assert_eq!(line(&m), Some(55.0));
    }

    #[test]
    fn a_calibration_carries_its_line_span_exactly_when_it_fits_a_number() {
        let c = calibration(Value::Known(0.01));
        let moderated = CountsFit {
            beta: [0.08, 0.01],
            ..fit(8)
        };
        let fitted = calibrated(&measurement(None), &c, moderated)
            .unwrap()
            .calibration();
        assert_eq!(fitted.pulse.line_span_ev, Some((10.0, 50.0)));
        let mut wider = measurement(Some(55.0));
        material(&mut wider).isotopes[1].1 = Value::Fitted(1e-4);
        match fit_counts(&wider, &fitted) {
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
        let without = CountsFit {
            unbounded: None,
            ..fit(4)
        };
        let unfitted = calibrated(&measurement(None), &known, without)
            .unwrap()
            .calibration();
        assert_eq!(unfitted.pulse.line_span_ev, None);
    }

    fn original() -> PulseCalibration {
        let mut fit = CountsFit {
            r: 0.012_537_345_881_063_615,
            ..fit(8)
        };
        fit.regions[0].overdispersion = [Some(1.25), Some(1.5)];
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
            "\"format_version\": 2",
            "\"name\": \"alpha0\"",
            "\"unit\": \"1/(µs·√eV)\"",
            "\"status\": \"fitted\"",
            "\"rank\": 4",
            "\"sample_overdispersion\": 1.5",
            "\"transfer\": \"unchecked\"",
        ] {
            assert!(text.contains(expected), "{expected} in {text}");
        }
    }

    #[test]
    fn files_that_are_not_such_a_calibration_are_refused() {
        let original: serde_json::Value = serde_json::from_str(&file()).unwrap();
        let edits: [Edit; 21] = [
            ("version", |v| v["format_version"] = 1.into()),
            ("mean", |v| {
                v["mean"].as_array_mut().unwrap().pop();
            }),
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
        let mut known = calibration(Value::Known(0.01));
        let pulse = &mut known.pulse;
        pulse.alpha[1] = Value::Known(1.1);
        pulse.beta[0] = Value::Known(0.0);
        pulse.r = Value::Known(0.21);
        pulse.fwhm_squared_us2 = Value::Known(0.12);
        let unfitted = calibrated(&measurement(None), &known, fit(4)).unwrap();
        let mut stray: serde_json::Value = serde_json::from_str(&unfitted.to_json()).unwrap();
        assert!(PulseCalibration::from_json(&stray.to_string()).is_ok());
        stray["mean"] = serde_json::json!([1.0]);
        assert!(PulseCalibration::from_json(&stray.to_string()).is_err());
        match PulseCalibration::from_json(r#"{"format_version": 1, "pulse_model": "", "runs": {}}"#)
        {
            Err(PipelineError::InvalidParameter(message)) => {
                assert!(message.contains("got version 1"), "{message}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_foil_whose_density_or_temperature_is_not_measured_is_refused() {
        let c = calibration(Value::Known(0.01));
        let mut fitted = measurement(None);
        material(&mut fitted).isotopes[0].1 = Value::Fitted(2e-3);
        let mut warm = measurement(None);
        material(&mut warm).temperature_k = Value::Fitted(300.0);
        for m in [fitted, warm] {
            match PulseCalibration::new(&m, &c, provenance()) {
                Err(PipelineError::InvalidParameter(message)) => {
                    assert!(message.contains("is measured"), "{message}")
                }
                other => panic!("{other:?}"),
            }
        }
        let mut two = measurement(None);
        two.regions.push(two.regions[0].clone());
        let mut empty = measurement(None);
        empty.regions[0].material = None;
        for m in [two, empty] {
            match PulseCalibration::new(&m, &c, provenance()) {
                Err(PipelineError::InvalidParameter(message)) => {
                    assert!(message.contains("the foil alone"), "{message}")
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
        let mut boxed = c.clone();
        boxed.pulse.r = Value::Within {
            start: 0.2,
            lower: 0.15,
            upper: 0.25,
        };
        for (c, provenance, refusal) in [
            (&c, same, "names its foil"),
            (&c, unnamed, "names its foil"),
            (&original().calibration(), provenance(), "calibrated alone"),
            (&measured, provenance(), "calibrated alone"),
            (&boxed, provenance(), "calibrated alone"),
        ] {
            match PulseCalibration::new(&measurement(None), c, provenance) {
                Err(PipelineError::InvalidParameter(message)) => {
                    assert!(message.contains(refusal), "{message}")
                }
                other => panic!("{other:?}"),
            }
        }
    }

    const VERSION_2: &str = r#"{
  "format_version": 2,
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
      "status": "fitted"
    },
    {
      "name": "beta0",
      "unit": "1/(µs·√eV)",
      "value": 0.0,
      "status": "fitted"
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
      "status": "fitted"
    },
    {
      "name": "fwhm_squared",
      "unit": "µs²",
      "value": 0.12,
      "status": "fitted"
    }
  ],
  "mean": [
    -0.01,
    -0.012,
    -0.014,
    -0.016
  ],
  "covariance": [
    [
      0.1,
      0.0,
      0.02,
      -0.01
    ],
    [
      0.0,
      0.12,
      0.0,
      0.0
    ],
    [
      0.02,
      0.0,
      0.14,
      0.04
    ],
    [
      -0.01,
      0.0,
      0.04,
      0.16
    ]
  ],
  "rank": 4,
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
    fn the_file_is_format_version_2_byte_for_byte() {
        let original = original();
        assert_eq!(original.to_json(), VERSION_2);
        let read = PulseCalibration::from_json(VERSION_2).unwrap();
        assert_eq!(format!("{read:?}"), format!("{original:?}"));
    }

    fn other_foil(density: f64) -> Measurement {
        let mut m = measurement(None);
        material(&mut m).isotopes[0].1 = Value::Measured {
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

    fn other(alpha1: f64, alpha0: f64) -> PulseCalibration {
        let m = other_foil(3e-3);
        let mut fit = CountsFit {
            alpha: [alpha0, alpha1],
            beta: [0.004, 0.01],
            r: 0.205,
            fwhm_squared_us2: 0.11,
            ..fit(8)
        };
        let unbounded = fit.unbounded.as_mut().expect("unbounded");
        unbounded.mean[4] += alpha1 - 1.1;
        for (i, j, c) in [(4, 5, 0.02), (5, 6, 0.015)] {
            *unbounded.covariance.get_mut(i, j) = c;
            *unbounded.covariance.get_mut(j, i) = c;
        }
        let c = calibration(Value::Known(0.01));
        PulseCalibration::from_fit(&m, &c, foil(&m).unwrap(), other_provenance(), &fit).unwrap()
    }

    fn solve(mut m: Vec<Vec<f64>>, mut v: Vec<f64>) -> Vec<f64> {
        let k = v.len();
        for i in 0..k {
            let pivot = m[i].clone();
            for r in i + 1..k {
                let factor = m[r][i] / pivot[i];
                for (entry, above) in m[r].iter_mut().zip(&pivot).skip(i) {
                    *entry -= factor * above;
                }
                v[r] -= factor * v[i];
            }
        }
        for i in (0..k).rev() {
            v[i] = (v[i] - (i + 1..k).map(|c| m[i][c] * v[c]).sum::<f64>()) / m[i][i];
        }
        v
    }

    #[test]
    fn a_transfer_compares_every_number_both_foils_fitted() {
        let (a, b) = (original(), other(1.15, 0.5));
        let result = a.transfer(&b).unwrap().agreement();
        let (pa, pb) = (a.prior.as_ref().unwrap(), b.prior.as_ref().unwrap());
        let k = pa.numbers.len();
        let d: Vec<f64> = (0..k).map(|i| pa.mean[i] - pb.mean[i]).collect();
        let sum = (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| pa.covariance.get(i, j) + pb.covariance.get(i, j))
                    .collect()
            })
            .collect();
        let q: f64 = d
            .iter()
            .zip(solve(sum, d.clone()))
            .map(|(x, y)| x * y)
            .sum();
        assert!(q > 0.0);
        assert_eq!(result.dof, 4);
        assert!((result.q / q - 1.0).abs() <= 1e-12, "{} vs {q}", result.q);
        assert_eq!(result, Consistency::new(result.q, 4).unwrap());
    }

    #[test]
    fn a_transfer_needs_another_foil_with_the_same_pulse_model() {
        let a = original();
        let mut same_foil = other(1.15, 0.5);
        same_foil.provenance.foil = "foil-a".into();
        let mut shared = other(1.15, 0.5);
        shared.provenance.sample = "sample-1".into();
        let mut open_as_sample = other(1.15, 0.5);
        open_as_sample.provenance.sample = "open-1".into();
        let mut sample_as_open = other(1.15, 0.5);
        sample_as_open.provenance.open = "sample-1".into();
        let m = other_foil(3e-3);
        let calibrated_other = |c: &Calibration, fit: CountsFit| {
            PulseCalibration::from_fit(&m, c, foil(&m).unwrap(), other_provenance(), &fit).unwrap()
        };
        let mut known_r = calibration(Value::Known(0.01));
        known_r.pulse.r = Value::Known(0.21);
        let mut known = known_r.clone();
        let pulse = &mut known.pulse;
        pulse.alpha[1] = Value::Known(1.1);
        pulse.beta[0] = Value::Known(0.0);
        pulse.fwhm_squared_us2 = Value::Known(0.12);
        let unfitted = calibrated(&measurement(None), &known, fit(4)).unwrap();
        for (a, b, refusal) in [
            (&a, same_foil, "physically different foil"),
            (&a, shared, "physically different foil"),
            (&a, open_as_sample, "physically different foil"),
            (&a, sample_as_open, "physically different foil"),
            (&a, other(1.15, 0.6), "different pulse models"),
            (
                &a,
                calibrated_other(&known_r, fit(7)),
                "different pulse models",
            ),
            (
                &unfitted,
                calibrated_other(&known, fit(4)),
                "fit no pulse number",
            ),
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
        let failing = other(3.0, 0.5);
        assert!(original().transfer(&failing).unwrap().agreement().p <= 0.01);
        match original().record_transfer(&failing) {
            Err(PipelineError::InvalidParameter(message)) => {
                assert!(message.contains("does not transfer"), "{message}")
            }
            other => panic!("{other:?}"),
        }

        let b = other(1.15, 0.5);
        let a = original().record_transfer(&b).unwrap();
        assert_eq!(a.transfer, Some(original().transfer(&b).unwrap().record));
        let text = a.to_json();
        let read = PulseCalibration::from_json(&text).unwrap();
        assert_eq!(format!("{read:?}"), format!("{a:?}"));
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
                v["transfer"]["p"] = Consistency::new(50.0, 4).unwrap().p.into();
            }),
            ("dof not the numbers fitted", |v| {
                let d2 = v["transfer"]["d2"].as_f64().unwrap();
                v["transfer"]["dof"] = 3.into();
                v["transfer"]["p"] = Consistency::new(d2, 3).unwrap().p.into();
            }),
            ("same foil", |v| {
                v["transfer"]["provenance"]["foil"] = v["provenance"]["foil"].clone();
            }),
            ("same sample run", |v| {
                v["transfer"]["provenance"]["sample"] = v["provenance"]["sample"].clone();
            }),
            ("this open run as the other's sample", |v| {
                v["transfer"]["provenance"]["sample"] = v["provenance"]["open"].clone();
            }),
            ("this sample run as the other's open", |v| {
                v["transfer"]["provenance"]["open"] = v["provenance"]["sample"].clone();
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
        for (what, edit) in edits {
            let mut value = original.clone();
            edit(&mut value);
            assert!(
                PulseCalibration::from_json(&value.to_string()).is_err(),
                "{what}"
            );
        }
    }
}
