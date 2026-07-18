//! Separate open-beam and sample count response in detector time.
//!
//! A detector does not observe a blurred transmission ratio. For discrete
//! true-energy quadrature points `E_j`, the two physical arms are
//!
//! ```text
//! O_i = sum_j F_j       R_i(E_j)
//! S_i = sum_j F_j T_j  R_i(E_j)
//! ```
//!
//! `F_j` is the incident fluence weight at true energy `E_j` (the incident
//! flux density multiplied by the caller's energy-integration weight), `T_j`
//! is the sample transmission before instrument response, and `R_i(E_j)` is
//! the probability that a neutron at `E_j` lands between the actual detector
//! time edges of bin `i`.
//!
//! Keeping `F_j` as an already-integrated weight is deliberate: this operator
//! never guesses energy-bin widths from centers. A caller using a continuous
//! source spectrum must choose and disclose its energy quadrature, then pass
//! `F_j = w_j Phi(E_j)`. The detector-time integration itself is performed by
//! the response model over the supplied measured bin edges.

use std::fmt;

use rayon::prelude::*;

use crate::resolution::{ResolutionFunction, ResolutionParseError};

type CompactResponseRow = (Vec<u32>, Vec<f64>);

/// Expected detector-bin counts for the open-beam and sample measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoArmCounts {
    /// Open-beam expectation `sum_j F_j R_i(E_j)`.
    pub open_beam: Vec<f64>,
    /// Sample expectation `sum_j F_j T_j R_i(E_j)`.
    pub sample: Vec<f64>,
}

/// Detector-bin probabilities for a fixed instrument response.
///
/// Rows correspond to `true_energies_ev`; columns correspond to consecutive
/// intervals in `detector_time_edges_us`. Building the matrix evaluates the
/// analytical IC/tabulated bin integrals once. Reusing it during optimization
/// changes only the true-energy sample transmission, not the detector physics.
///
/// Only entries whose evaluated probability is strictly greater than zero are
/// stored. This is lossless: there is no numerical cutoff, and every nonzero
/// value returned by the response model is retained.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorBinResponseMatrix {
    row_offsets: Vec<usize>,
    detector_bins: Vec<u32>,
    probabilities: Vec<f64>,
    n_true_energies: usize,
    n_detector_bins: usize,
}

impl DetectorBinResponseMatrix {
    /// Build a fixed detector-bin response matrix.
    pub fn new(
        true_energies_ev: &[f64],
        detector_time_edges_us: &[f64],
        timing_offset_us: f64,
        response: &ResolutionFunction,
    ) -> Result<Self, CountsResponseError> {
        if true_energies_ev.is_empty() {
            return Err(CountsResponseError::EmptyTrueEnergyGrid);
        }
        for (index, &energy) in true_energies_ev.iter().enumerate() {
            if !energy.is_finite() || energy <= 0.0 {
                return Err(CountsResponseError::InvalidTrueEnergy {
                    index,
                    value: energy,
                });
            }
        }

        let n_detector_bins = detector_time_edges_us.len().saturating_sub(1);
        if n_detector_bins > u32::MAX as usize {
            return Err(CountsResponseError::Resolution(
                ResolutionParseError::InvalidFormat(format!(
                    "detector response has {n_detector_bins} bins, exceeding the u32 storage limit"
                )),
            ));
        }
        // Each true-energy response is independent. Rayon preserves the input
        // order of this indexed parallel collect, so rows and all subsequent
        // accumulation orders remain deterministic.
        let row_results: Vec<Result<CompactResponseRow, ResolutionParseError>> = true_energies_ev
            .par_iter()
            .map(|&energy| {
                let row = response.detector_bin_probabilities(
                    energy,
                    detector_time_edges_us,
                    timing_offset_us,
                )?;
                debug_assert_eq!(row.len(), n_detector_bins);
                let mut bins = Vec::new();
                let mut values = Vec::new();
                for (detector_bin, probability) in row.into_iter().enumerate() {
                    if !probability.is_finite() || probability < 0.0 {
                        return Err(ResolutionParseError::InvalidFormat(format!(
                            "detector response probability at E = {energy} eV, bin {detector_bin} must be finite and >= 0, got {probability}"
                        )));
                    }
                    if probability > 0.0 {
                        bins.push(detector_bin as u32);
                        values.push(probability);
                    }
                }
                Ok::<_, ResolutionParseError>((bins, values))
            })
            .collect();
        // Resolve errors after the ordered collect so the first failing input
        // row is reported deterministically regardless of thread scheduling.
        let rows: Vec<CompactResponseRow> = row_results.into_iter().collect::<Result<_, _>>()?;
        let nonzero_count = rows.iter().try_fold(0_usize, |total, (_, values)| {
            total.checked_add(values.len()).ok_or_else(|| {
                CountsResponseError::Resolution(ResolutionParseError::InvalidFormat(
                    "detector response nonzero count overflows usize".into(),
                ))
            })
        })?;

        let mut row_offsets = Vec::with_capacity(true_energies_ev.len() + 1);
        let mut detector_bins = Vec::with_capacity(nonzero_count);
        let mut probabilities = Vec::with_capacity(nonzero_count);
        row_offsets.push(0);
        for (mut bins, mut values) in rows {
            detector_bins.append(&mut bins);
            probabilities.append(&mut values);
            row_offsets.push(probabilities.len());
        }

        Ok(Self {
            row_offsets,
            detector_bins,
            probabilities,
            n_true_energies: true_energies_ev.len(),
            n_detector_bins,
        })
    }

    /// Number of true-energy quadrature points.
    pub fn n_true_energies(&self) -> usize {
        self.n_true_energies
    }

    /// Number of measured detector-time bins.
    pub fn n_detector_bins(&self) -> usize {
        self.n_detector_bins
    }

    /// Number of stored nonzero probabilities.
    pub fn nnz(&self) -> usize {
        self.probabilities.len()
    }

    /// Heap bytes used by the compact probability storage.
    ///
    /// This excludes the small fixed-size `Self` value and allocator overhead.
    pub fn storage_bytes(&self) -> usize {
        self.row_offsets.capacity() * std::mem::size_of::<usize>()
            + self.detector_bins.capacity() * std::mem::size_of::<u32>()
            + self.probabilities.capacity() * std::mem::size_of::<f64>()
    }

    /// Stored `(detector_bin, probability)` pairs for one true-energy row.
    pub fn row_entries(&self, true_index: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.row_offsets[true_index];
        let end = self.row_offsets[true_index + 1];
        self.detector_bins[start..end]
            .iter()
            .map(|&bin| bin as usize)
            .zip(self.probabilities[start..end].iter().copied())
    }

    /// Probability that true-energy row `true_index` lands in detector bin
    /// `detector_bin`.
    pub fn probability(&self, true_index: usize, detector_bin: usize) -> f64 {
        assert!(
            detector_bin < self.n_detector_bins,
            "detector bin out of range"
        );
        let start = self.row_offsets[true_index];
        let end = self.row_offsets[true_index + 1];
        match self.detector_bins[start..end].binary_search(&(detector_bin as u32)) {
            Ok(offset) => self.probabilities[start + offset],
            Err(_) => 0.0,
        }
    }

    /// Apply the fixed response separately to open and sample arms.
    pub fn apply(
        &self,
        incident_fluence_weights: &[f64],
        transmission: &[f64],
    ) -> Result<TwoArmCounts, CountsResponseError> {
        if incident_fluence_weights.len() != self.n_true_energies
            || transmission.len() != self.n_true_energies
        {
            return Err(CountsResponseError::LengthMismatch {
                energies: self.n_true_energies,
                incident_fluence: incident_fluence_weights.len(),
                transmission: transmission.len(),
            });
        }
        for (index, &fluence) in incident_fluence_weights.iter().enumerate() {
            if !fluence.is_finite() || fluence < 0.0 {
                return Err(CountsResponseError::InvalidIncidentFluence {
                    index,
                    value: fluence,
                });
            }
        }
        for (index, &value) in transmission.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(CountsResponseError::InvalidTransmission { index, value });
            }
        }

        let mut open_beam = vec![0.0; self.n_detector_bins];
        let mut sample = vec![0.0; self.n_detector_bins];
        let mut open_compensation = vec![0.0; self.n_detector_bins];
        let mut sample_compensation = vec![0.0; self.n_detector_bins];
        for true_index in 0..self.n_true_energies {
            let fluence = incident_fluence_weights[true_index];
            let sample_weight = fluence * transmission[true_index];
            for (detector_bin, probability) in self.row_entries(true_index) {
                compensated_add(
                    &mut open_beam[detector_bin],
                    &mut open_compensation[detector_bin],
                    fluence * probability,
                );
                compensated_add(
                    &mut sample[detector_bin],
                    &mut sample_compensation[detector_bin],
                    sample_weight * probability,
                );
            }
        }
        for detector_bin in 0..self.n_detector_bins {
            open_beam[detector_bin] += open_compensation[detector_bin];
            sample[detector_bin] += sample_compensation[detector_bin];
        }
        Ok(TwoArmCounts { open_beam, sample })
    }
}

/// Signal, additive background, and total expected counts for one measured arm.
///
/// Background is expressed directly in the detector bins of that measurement.
/// It is added after source attenuation and instrument response.  This is not
/// the SAMMY additive transmission curve, and it is not convolved a second
/// time by the instrument response.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmCountPrediction {
    /// Neutron counts produced by source, sample, and instrument response.
    pub neutron_signal: Vec<f64>,
    /// Independently supplied additive expected counts in the measured bins.
    pub background: Vec<f64>,
    /// Complete expectation, `neutron_signal + background`.
    pub total: Vec<f64>,
}

/// Complete expected counts for the open-beam and sample measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoArmCountPrediction {
    /// Open-beam measurement components.
    pub open_beam: ArmCountPrediction,
    /// Sample measurement components.
    pub sample: ArmCountPrediction,
}

/// Invalid inputs or unsupported response models for [`two_arm_count_response`].
#[derive(Debug)]
pub enum CountsResponseError {
    /// At least one true-energy point is required.
    EmptyTrueEnergyGrid,
    /// True-energy, fluence, and transmission arrays must have equal lengths.
    LengthMismatch {
        energies: usize,
        incident_fluence: usize,
        transmission: usize,
    },
    /// A true energy was non-positive or non-finite.
    InvalidTrueEnergy { index: usize, value: f64 },
    /// An incident fluence weight was negative or non-finite.
    InvalidIncidentFluence { index: usize, value: f64 },
    /// A physical sample transmission was outside `[0, 1]` or non-finite.
    InvalidTransmission { index: usize, value: f64 },
    /// A signal or background expected count was negative or non-finite.
    InvalidExpectedCount {
        field: &'static str,
        index: usize,
        value: f64,
    },
    /// Signal and background arrays did not describe the same detector bins.
    DetectorBinCountMismatch {
        open_signal: usize,
        sample_signal: usize,
        open_background: usize,
        sample_background: usize,
    },
    /// The response could not evaluate detector-bin probabilities.
    Resolution(ResolutionParseError),
}

impl fmt::Display for CountsResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTrueEnergyGrid => write!(f, "true_energies_ev must not be empty"),
            Self::LengthMismatch {
                energies,
                incident_fluence,
                transmission,
            } => write!(
                f,
                "true_energies_ev ({energies}), incident_fluence_weights ({incident_fluence}), and transmission ({transmission}) must have equal lengths"
            ),
            Self::InvalidTrueEnergy { index, value } => write!(
                f,
                "true_energies_ev[{index}] must be positive and finite, got {value}"
            ),
            Self::InvalidIncidentFluence { index, value } => write!(
                f,
                "incident_fluence_weights[{index}] must be finite and >= 0, got {value}"
            ),
            Self::InvalidTransmission { index, value } => write!(
                f,
                "transmission[{index}] must be finite and in [0, 1], got {value}"
            ),
            Self::InvalidExpectedCount {
                field,
                index,
                value,
            } => write!(
                f,
                "{field}[{index}] must be finite and >= 0 expected counts, got {value}"
            ),
            Self::DetectorBinCountMismatch {
                open_signal,
                sample_signal,
                open_background,
                sample_background,
            } => write!(
                f,
                "open neutron signal ({open_signal} bins), sample neutron signal ({sample_signal}), open_background_counts ({open_background}), and sample_background_counts ({sample_background}) must have equal lengths"
            ),
            Self::Resolution(error) => write!(f, "detector-time response failed: {error}"),
        }
    }
}

/// Add detector-bin backgrounds to an already evaluated two-arm neutron signal.
///
/// Both background arrays are expected counts for the corresponding complete
/// acquisition, already mapped into the same detector-time bins as `signal`.
/// A measured dark/blocked-beam reference may therefore be supplied directly
/// after applying its independently justified run normalization.  The function
/// does not guess an exposure, fit a smooth curve, or reinterpret a SAMMY
/// transmission-level background as detector counts.
///
/// A sample-scattering calculation that starts from a true-energy spectrum is
/// not an input to this function: it must first be propagated through the
/// instrument response.  Only a scattering reference already measured in the
/// detector bins can enter here directly.
pub fn add_count_backgrounds(
    signal: TwoArmCounts,
    open_background_counts: &[f64],
    sample_background_counts: &[f64],
) -> Result<TwoArmCountPrediction, CountsResponseError> {
    let n_bins = signal.open_beam.len();
    if signal.sample.len() != n_bins
        || open_background_counts.len() != n_bins
        || sample_background_counts.len() != n_bins
    {
        return Err(CountsResponseError::DetectorBinCountMismatch {
            open_signal: n_bins,
            sample_signal: signal.sample.len(),
            open_background: open_background_counts.len(),
            sample_background: sample_background_counts.len(),
        });
    }

    validate_expected_counts("open_neutron_signal", &signal.open_beam)?;
    validate_expected_counts("sample_neutron_signal", &signal.sample)?;
    validate_expected_counts("open_background_counts", open_background_counts)?;
    validate_expected_counts("sample_background_counts", sample_background_counts)?;

    let open_total = sum_expected_counts(
        "open_total_expected_counts",
        &signal.open_beam,
        open_background_counts,
    )?;
    let sample_total = sum_expected_counts(
        "sample_total_expected_counts",
        &signal.sample,
        sample_background_counts,
    )?;

    Ok(TwoArmCountPrediction {
        open_beam: ArmCountPrediction {
            neutron_signal: signal.open_beam,
            background: open_background_counts.to_vec(),
            total: open_total,
        },
        sample: ArmCountPrediction {
            neutron_signal: signal.sample,
            background: sample_background_counts.to_vec(),
            total: sample_total,
        },
    })
}

fn validate_expected_counts(
    field: &'static str,
    values: &[f64],
) -> Result<(), CountsResponseError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(CountsResponseError::InvalidExpectedCount {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn sum_expected_counts(
    field: &'static str,
    neutron_signal: &[f64],
    background: &[f64],
) -> Result<Vec<f64>, CountsResponseError> {
    neutron_signal
        .iter()
        .zip(background)
        .enumerate()
        .map(|(index, (&neutron, &background))| {
            let total = neutron + background;
            if total.is_finite() {
                Ok(total)
            } else {
                Err(CountsResponseError::InvalidExpectedCount {
                    field,
                    index,
                    value: total,
                })
            }
        })
        .collect()
}

impl std::error::Error for CountsResponseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resolution(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ResolutionParseError> for CountsResponseError {
    fn from(value: ResolutionParseError) -> Self {
        Self::Resolution(value)
    }
}

/// Apply one instrument response to the open-beam and sample arms separately.
///
/// `incident_fluence_weights[j]` is an expected neutron count integrated over
/// the true-energy quadrature element represented by `true_energies_ev[j]`.
/// Probability outside `detector_time_edges_us` is intentionally not
/// renormalized into the acquisition window.
///
/// This function models only source fluence, sample attenuation, and the
/// instrument response. Detector/gamma/scattering backgrounds and exposure
/// normalization are separate physical terms and must be added by a higher
/// layer that states their measurement location.
pub fn two_arm_count_response(
    true_energies_ev: &[f64],
    incident_fluence_weights: &[f64],
    transmission: &[f64],
    detector_time_edges_us: &[f64],
    timing_offset_us: f64,
    response: &ResolutionFunction,
) -> Result<TwoArmCounts, CountsResponseError> {
    if incident_fluence_weights.len() != true_energies_ev.len()
        || transmission.len() != true_energies_ev.len()
    {
        return Err(CountsResponseError::LengthMismatch {
            energies: true_energies_ev.len(),
            incident_fluence: incident_fluence_weights.len(),
            transmission: transmission.len(),
        });
    }
    DetectorBinResponseMatrix::new(
        true_energies_ev,
        detector_time_edges_us,
        timing_offset_us,
        response,
    )?
    .apply(incident_fluence_weights, transmission)
}

#[inline]
fn compensated_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let next = *sum + value;
    if sum.abs() >= value.abs() {
        *compensation += (*sum - next) + value;
    } else {
        *compensation += (value - next) + *sum;
    }
    *sum = next;
}
