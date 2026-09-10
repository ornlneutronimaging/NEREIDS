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
//! `F_j` is the incident fluence weight at true energy `E_j` — the incident
//! flux density × detector efficiency × the caller's energy-integration
//! weight, the discrete form of the contract's `Φ·ε` (pipeline-map R5·7:
//! detector efficiency is folded into `F_j`, never a separate silent factor).
//! `T_j` is the sample transmission before instrument response, and
//! `R_i(E_j)` is the probability that a neutron at `E_j` lands between the
//! actual detector time edges of bin `i`.
//!
//! Keeping `F_j` as an already-integrated weight is deliberate: this operator
//! never guesses energy-bin widths from centers. A caller using a continuous
//! source spectrum must choose and disclose its energy quadrature, then pass
//! `F_j = w_j ε(E_j) Φ(E_j)`. The detector-time integration itself is
//! performed by the response model over the supplied measured bin edges.
//!
//! The `timing_offset_us` handed to the response is convention-dependent
//! (mode-relative for tabulated kernels, emission-onset-relative for the
//! analytical Ikeda–Carpenter pulse); see
//! [`ResolutionFunction::detector_bin_probabilities`] — a calibrated offset
//! is not transferable between response variants.

use std::fmt;

use rayon::prelude::*;

use crate::resolution::{ResolutionFunction, ResolutionParseError};

type CompactResponseRow = (Vec<u32>, Vec<f64>, f64);

/// Expected detector-bin counts for the open-beam and sample measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoArmCounts {
    /// Open-beam expectation `sum_j F_j R_i(E_j)`.
    pub open_beam: Vec<f64>,
    /// Sample expectation `sum_j F_j T_j R_i(E_j)`.
    pub sample: Vec<f64>,
    /// Expected open-beam counts falling outside the supplied acquisition
    /// window: `sum_j F_j (1 - sum_i R_i(E_j))`.
    ///
    /// The quantified acquisition-window loss required by the pipeline-map
    /// contract (R5·7): window loss is disclosed here, never renormalized
    /// back into the window.
    pub open_beam_window_loss: f64,
    /// Expected sample counts falling outside the supplied acquisition
    /// window: `sum_j F_j T_j (1 - sum_i R_i(E_j))`.
    pub sample_window_loss: f64,
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
/// value returned by the response model is retained. Probability outside the
/// acquisition window is likewise never renormalized into the stored rows —
/// each row's out-of-window fraction is kept alongside it so [`Self::apply`]
/// can report the quantified window loss exactly as
/// [`two_arm_count_response`] does (pipeline-map R5·7).
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorBinResponseMatrix {
    row_offsets: Vec<usize>,
    detector_bins: Vec<u32>,
    probabilities: Vec<f64>,
    /// Per-row out-of-window probability `(1 - sum_i R_i(E_j)).max(0)`.
    lost_fractions: Vec<f64>,
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
                let mut in_window = 0.0_f64;
                for (detector_bin, probability) in row.into_iter().enumerate() {
                    if !probability.is_finite() || probability < 0.0 {
                        return Err(ResolutionParseError::InvalidFormat(format!(
                            "detector response probability at E = {energy} eV, bin {detector_bin} must be finite and >= 0, got {probability}"
                        )));
                    }
                    in_window += probability;
                    if probability > 0.0 {
                        bins.push(detector_bin as u32);
                        values.push(probability);
                    }
                }
                Ok::<_, ResolutionParseError>((bins, values, (1.0 - in_window).max(0.0)))
            })
            .collect();
        // Resolve errors after the ordered collect so the first failing input
        // row is reported deterministically regardless of thread scheduling.
        let rows: Vec<CompactResponseRow> = row_results.into_iter().collect::<Result<_, _>>()?;
        let nonzero_count = rows.iter().try_fold(0_usize, |total, (_, values, _)| {
            total.checked_add(values.len()).ok_or_else(|| {
                CountsResponseError::Resolution(ResolutionParseError::InvalidFormat(
                    "detector response nonzero count overflows usize".into(),
                ))
            })
        })?;

        let mut row_offsets = Vec::with_capacity(true_energies_ev.len() + 1);
        let mut detector_bins = Vec::with_capacity(nonzero_count);
        let mut probabilities = Vec::with_capacity(nonzero_count);
        let mut lost_fractions = Vec::with_capacity(true_energies_ev.len());
        row_offsets.push(0);
        for (mut bins, mut values, lost) in rows {
            detector_bins.append(&mut bins);
            probabilities.append(&mut values);
            row_offsets.push(probabilities.len());
            lost_fractions.push(lost);
        }

        Ok(Self {
            row_offsets,
            detector_bins,
            probabilities,
            lost_fractions,
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
    /// This excludes the small fixed-size `Self` value, the per-row window-loss
    /// fractions, and allocator overhead.
    pub fn storage_bytes(&self) -> usize {
        self.row_offsets.capacity() * std::mem::size_of::<usize>()
            + self.detector_bins.capacity() * std::mem::size_of::<u32>()
            + self.probabilities.capacity() * std::mem::size_of::<f64>()
    }

    /// Stored `(detector_bin, probability)` pairs for one true-energy row.
    pub fn row_entries(&self, true_index: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        assert!(
            true_index < self.n_true_energies,
            "true-energy row out of range"
        );
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
            true_index < self.n_true_energies,
            "true-energy row out of range"
        );
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
    ///
    /// Numerically equivalent to [`two_arm_count_response`] on the same
    /// inputs, including the per-arm acquisition-window loss report.
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
        let mut open_window_loss = 0.0;
        let mut open_window_loss_compensation = 0.0;
        let mut sample_window_loss = 0.0;
        let mut sample_window_loss_compensation = 0.0;
        for true_index in 0..self.n_true_energies {
            let fluence = incident_fluence_weights[true_index];
            let sample_weight = fluence * transmission[true_index];
            let lost = self.lost_fractions[true_index];
            compensated_add(
                &mut open_window_loss,
                &mut open_window_loss_compensation,
                fluence * lost,
            );
            compensated_add(
                &mut sample_window_loss,
                &mut sample_window_loss_compensation,
                sample_weight * lost,
            );
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
        Ok(TwoArmCounts {
            open_beam,
            sample,
            open_beam_window_loss: open_window_loss + open_window_loss_compensation,
            sample_window_loss: sample_window_loss + sample_window_loss_compensation,
        })
    }
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
            Self::Resolution(error) => write!(f, "detector-time response failed: {error}"),
        }
    }
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
/// the true-energy quadrature element represented by `true_energies_ev[j]`,
/// with detector efficiency already folded in (`F_j = w_j ε(E_j) Φ(E_j)` —
/// see the module docs and pipeline-map R5·7). Probability outside
/// `detector_time_edges_us` is intentionally not renormalized into the
/// acquisition window; the expected counts lost outside it are reported per
/// arm on the returned [`TwoArmCounts`].
///
/// This function models only source fluence, detector efficiency (inside
/// `F_j`), sample attenuation, and the instrument response. Detector/gamma/
/// scattering backgrounds and exposure normalization are separate physical
/// terms and must be added by a higher layer that states their measurement
/// location.
pub fn two_arm_count_response(
    true_energies_ev: &[f64],
    incident_fluence_weights: &[f64],
    transmission: &[f64],
    detector_time_edges_us: &[f64],
    timing_offset_us: f64,
    response: &ResolutionFunction,
) -> Result<TwoArmCounts, CountsResponseError> {
    if true_energies_ev.is_empty() {
        return Err(CountsResponseError::EmptyTrueEnergyGrid);
    }
    if incident_fluence_weights.len() != true_energies_ev.len()
        || transmission.len() != true_energies_ev.len()
    {
        return Err(CountsResponseError::LengthMismatch {
            energies: true_energies_ev.len(),
            incident_fluence: incident_fluence_weights.len(),
            transmission: transmission.len(),
        });
    }
    for (index, &energy) in true_energies_ev.iter().enumerate() {
        if !energy.is_finite() || energy <= 0.0 {
            return Err(CountsResponseError::InvalidTrueEnergy {
                index,
                value: energy,
            });
        }
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

    // Calling the response once even when every fluence weight is zero is
    // important: malformed detector edges and unsupported Gaussian responses
    // must still fail clearly instead of appearing to succeed with all zeros.
    let n_bins = detector_time_edges_us.len().saturating_sub(1);
    let mut open_beam = vec![0.0; n_bins];
    let mut sample = vec![0.0; n_bins];
    let mut open_compensation = vec![0.0; n_bins];
    let mut sample_compensation = vec![0.0; n_bins];
    let mut open_window_loss = 0.0;
    let mut open_window_loss_compensation = 0.0;
    let mut sample_window_loss = 0.0;
    let mut sample_window_loss_compensation = 0.0;

    for ((&energy, &fluence), &sample_transmission) in true_energies_ev
        .iter()
        .zip(incident_fluence_weights)
        .zip(transmission)
    {
        let probabilities = response.detector_bin_probabilities(
            energy,
            detector_time_edges_us,
            timing_offset_us,
        )?;
        debug_assert_eq!(probabilities.len(), n_bins);

        // The bin probabilities come from a normalized CDF, so the in-window
        // total is <= 1; the remainder is the quantified acquisition-window
        // loss disclosed on the result (pipeline-map R5·7).
        let in_window: f64 = probabilities.iter().sum();
        let lost = (1.0 - in_window).max(0.0);
        compensated_add(
            &mut open_window_loss,
            &mut open_window_loss_compensation,
            fluence * lost,
        );
        compensated_add(
            &mut sample_window_loss,
            &mut sample_window_loss_compensation,
            fluence * sample_transmission * lost,
        );

        // Neumaier-compensated accumulation keeps a weak high-energy tail from
        // being lost when the same bin also contains a much larger prompt term.
        for (bin, probability) in probabilities.into_iter().enumerate() {
            compensated_add(
                &mut open_beam[bin],
                &mut open_compensation[bin],
                fluence * probability,
            );
            compensated_add(
                &mut sample[bin],
                &mut sample_compensation[bin],
                fluence * sample_transmission * probability,
            );
        }
    }
    for bin in 0..n_bins {
        open_beam[bin] += open_compensation[bin];
        sample[bin] += sample_compensation[bin];
    }

    Ok(TwoArmCounts {
        open_beam,
        sample,
        open_beam_window_loss: open_window_loss + open_window_loss_compensation,
        sample_window_loss: sample_window_loss + sample_window_loss_compensation,
    })
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
