//! Synthetic counts-domain measurements with known ground truth.
//!
//! Correctness of a fit is established by injecting values and checking they
//! come back. Measured VENUS data cannot do that — it has no ground truth —
//! so it answers a later question.
//!
//! What is generated is the observation model the counts-KL path is supposed
//! to invert, built from the same components the fit uses rather than from a
//! parallel implementation:
//!
//! ```text
//! E_true = corrected_energy_grid(E_nominal, t0, L_scale, L)
//! T_j    = exp(-sum_i (n d)_i sigma_i(E_true_j; T))
//! O_i    = sum_j F_j R_ij + B_o,i        S_i = sum_j F_j T_j R_ij + B_s,i
//! ```
//!
//! with `R_ij` the detector-bin response of the true resolution kernel, and
//! the recorded counts a Poisson draw around `O_i` and `S_i`.
//!
//! The two arms are broadened separately. A post-hoc broadened ratio is a
//! different quantity and would make the fixture agree with a model the
//! pipeline deliberately refuses.
//!
//! Backgrounds are per detector bin and differ between the arms: the sample
//! adds scatter and gammas, so `B_s = B_o` is not a physical case.

use nereids_endf::resonance::ResonanceData;
use nereids_fitting::resolution_calib::corrected_energy_grid;
use nereids_physics::counts_response::{add_count_backgrounds, two_arm_count_response};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_physics::transmission::{SampleParams, forward_model};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Poisson};

/// Everything injected into a synthetic measurement, shared across pixels.
///
/// `t0_us` and `l_scale` describe the instrument the data was recorded on.
/// At `(0.0, 1.0)` the true energies equal the nominal ones, which is the
/// case for a fixture that is not exercising calibration recovery.
pub struct Truth {
    /// Energy grid the fit will be given, ascending (eV).
    pub nominal_energies_ev: Vec<f64>,
    /// Nominal flight path (m).
    pub flight_path_m: f64,
    /// True TOF zero (µs).
    pub t0_us: f64,
    /// True flight-path scale.
    pub l_scale: f64,
    /// True sample temperature (K).
    pub temperature_k: f64,
    /// Isotopes present, in the order their densities are given.
    pub isotopes: Vec<ResonanceData>,
    /// True instrument resolution.
    pub resolution: ResolutionFunction,
    /// Trigger offset of the detector time axis (µs).
    pub timing_offset_us: f64,
    /// Extra detector bins beyond each end of the nominal grid's flight
    /// times.
    ///
    /// The acquisition window has to be wider than the region of interest,
    /// because the moderator's storage tail delivers neutrons well after the
    /// nominal arrival time. Counts outside the window are lost, and a lossy
    /// fixture does not announce itself — it looks like a normalization the
    /// fit must absorb, which is where a density bias would hide.
    pub window_pad_bins: usize,
    /// Expected open-beam neutron counts per detector bin, before background.
    pub open_beam_counts_per_bin: f64,
    /// Expected open-arm background counts per detector bin.
    pub open_background_per_bin: f64,
    /// Expected sample-arm background counts per detector bin. Larger than
    /// the open arm's: the sample scatters neutrons and emits gammas.
    pub sample_background_per_bin: f64,
}

/// One pixel's recorded counts, with the expectations they were drawn from.
pub struct Measurement {
    /// Poisson draw of the sample arm, per detector bin.
    pub sample_counts: Vec<f64>,
    /// Poisson draw of the open-beam arm, per detector bin.
    pub open_beam_counts: Vec<f64>,
    /// Expected sample counts, before the draw.
    pub expected_sample: Vec<f64>,
    /// Expected open-beam counts, before the draw.
    pub expected_open: Vec<f64>,
    /// True energies the cross-sections were evaluated at (eV).
    pub true_energies_ev: Vec<f64>,
    /// Detector-time bin edges the counts are binned into (µs), ascending.
    pub detector_time_edges_us: Vec<f64>,
    /// Incident fluence weight per true energy.
    pub incident_fluence_weights: Vec<f64>,
    /// Expected neutron counts that fell outside the acquisition window, per
    /// arm. Reported rather than renormalized away: a fixture that quietly
    /// lost counts here would look like an unexplained normalization error in
    /// whatever fit consumed it.
    pub window_loss: (f64, f64),
}

impl Truth {
    /// Detector-time bin edges spanning the nominal grid's flight times.
    ///
    /// Energies ascend, so flight times descend; the edges are reversed into
    /// ascending time, which is the order the response matrix and the
    /// recorded counts both use. Edges are bin boundaries, so there is one
    /// more of them than there are energies.
    /// # Panics
    /// Panics if `nominal_energies_ev` has fewer than two points: the bin
    /// widths at each end are taken from the first and last spacing, and a
    /// grid with no spacing has no bins to describe.
    pub fn detector_time_edges_us(&self) -> Vec<f64> {
        assert!(
            self.nominal_energies_ev.len() >= 2,
            "a detector time axis needs at least two nominal energies, got {}",
            self.nominal_energies_ev.len()
        );
        let kl = TOF_FACTOR * self.flight_path_m;
        let mut times: Vec<f64> = self
            .nominal_energies_ev
            .iter()
            .map(|&e| self.timing_offset_us + kl / e.sqrt())
            .collect();
        times.reverse();
        let first_width = times[1] - times[0];
        let last = times.len() - 1;
        let last_width = times[last] - times[last - 1];

        let mut edges = Vec::with_capacity(times.len() + 1 + 2 * self.window_pad_bins);
        for pad in (1..=self.window_pad_bins).rev() {
            edges.push(times[0] - (0.5 + pad as f64) * first_width);
        }
        edges.push(times[0] - 0.5 * first_width);
        for pair in times.windows(2) {
            edges.push(0.5 * (pair[0] + pair[1]));
        }
        edges.push(times[last] + 0.5 * last_width);
        for pad in 1..=self.window_pad_bins {
            edges.push(times[last] + (0.5 + pad as f64) * last_width);
        }
        edges
    }

    /// The energies a neutron recorded on this instrument actually had.
    pub fn true_energies_ev(&self) -> Vec<f64> {
        corrected_energy_grid(
            &self.nominal_energies_ev,
            self.t0_us,
            self.l_scale,
            self.flight_path_m,
        )
        .expect("valid energy-scale truth")
    }

    /// Generate one pixel's measurement at the given per-isotope densities.
    ///
    /// `seed` selects the noise realization; a fixed seed makes a recorded
    /// bias reproducible.
    pub fn measure(&self, densities: &[f64], seed: u64) -> Measurement {
        assert_eq!(
            densities.len(),
            self.isotopes.len(),
            "one density per isotope"
        );

        let true_energies_ev = self.true_energies_ev();
        let sample = SampleParams::new(
            self.temperature_k,
            self.isotopes
                .iter()
                .cloned()
                .zip(densities.iter().copied())
                .collect(),
        )
        .expect("valid sample truth");
        // No instrument here: resolution acts on counts through the two-arm
        // response below, not on transmission.
        let transmission =
            forward_model(&true_energies_ev, &sample, None).expect("valid forward model");

        let detector_time_edges_us = self.detector_time_edges_us();
        let n_bins = detector_time_edges_us.len() - 1;
        let incident_fluence_weights = vec![self.open_beam_counts_per_bin; true_energies_ev.len()];

        let signal = two_arm_count_response(
            &true_energies_ev,
            &incident_fluence_weights,
            &transmission,
            &detector_time_edges_us,
            self.timing_offset_us,
            &self.resolution,
        )
        .expect("valid two-arm response");

        let prediction = add_count_backgrounds(
            signal,
            &vec![self.open_background_per_bin; n_bins],
            &vec![self.sample_background_per_bin; n_bins],
        )
        .expect("valid backgrounds");

        let expected_open = prediction.open_beam.total.clone();
        let expected_sample = prediction.sample.total.clone();
        let window_loss = (
            prediction.open_beam.window_loss,
            prediction.sample.window_loss,
        );

        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let draw = |rng: &mut ChaCha12Rng, expected: &[f64]| -> Vec<f64> {
            expected
                .iter()
                .map(|&mu| {
                    // Poisson is undefined at zero rate and the draw is
                    // identically zero there.
                    if mu <= 0.0 {
                        0.0
                    } else {
                        Poisson::new(mu).expect("positive rate").sample(rng)
                    }
                })
                .collect()
        };
        let open_beam_counts = draw(&mut rng, &expected_open);
        let sample_counts = draw(&mut rng, &expected_sample);

        Measurement {
            sample_counts,
            open_beam_counts,
            expected_sample,
            expected_open,
            true_energies_ev,
            detector_time_edges_us,
            incident_fluence_weights,
            window_loss,
        }
    }

    /// A 4x4 detector: sixteen pixels sharing this truth, each with its own
    /// per-isotope densities and its own noise realization.
    pub fn measure_detector(&self, pixel_densities: &[Vec<f64>], seed: u64) -> Vec<Measurement> {
        pixel_densities
            .iter()
            .enumerate()
            .map(|(pixel, densities)| self.measure(densities, seed + pixel as u64))
            .collect()
    }
}
