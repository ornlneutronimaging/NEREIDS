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

use crate::resolution::{ResolutionFunction, ResolutionParseError};

/// Expected detector-bin counts for the open-beam and sample measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoArmCounts {
    /// Open-beam expectation `sum_j F_j R_i(E_j)`.
    pub open_beam: Vec<f64>,
    /// Sample expectation `sum_j F_j T_j R_i(E_j)`.
    pub sample: Vec<f64>,
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

    Ok(TwoArmCounts { open_beam, sample })
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
