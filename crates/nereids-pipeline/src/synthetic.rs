//! Synthetic counts-domain measurements with known ground truth:
//!
//! ```text
//! E_true = exact_true_energies(edges, timing_offset, L, t0, L_scale)
//! T_j    = exp(-sum_i (n d)_i sigma_i(E_true_j; T))
//! O_i    = sum_j F_j R_ij + B_o,i        S_i = sum_j F_j T_j R_ij + B_s,i
//! ```
//!
//! with `R_ij` the detector-bin response of the true resolution kernel,
//! per-bin backgrounds that differ between the arms, and the recorded
//! counts a Poisson draw around `O_i` and `S_i`.

use nereids_endf::resonance::ResonanceData;
use nereids_physics::counts_response::{add_count_backgrounds, two_arm_count_response};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_physics::transmission::{SampleParams, forward_model};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Poisson};

use crate::pipeline::exact_true_energies;

/// Everything injected into a synthetic measurement, shared across pixels.
///
/// `t0_us` and `l_scale` describe the instrument the data was recorded on.
/// At `(0.0, 1.0)` the true energies equal the nominal ones, which is the
/// case for a fixture that is not exercising calibration recovery.
pub struct Truth {
    /// Detector-time bin edges the counts are binned into (µs), ascending.
    pub detector_time_edges_us: Vec<f64>,
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

/// Detector-time bin edges around the flight times of an ascending energy
/// grid, with `window_pad_bins` extra bins of the end spacing beyond each
/// end; the padding must cover the kernel's tail past the last arrival, as
/// counts outside the window are lost.
///
/// # Panics
/// Panics if `nominal_energies_ev` has fewer than two points.
pub fn detector_time_edges_around(
    nominal_energies_ev: &[f64],
    flight_path_m: f64,
    timing_offset_us: f64,
    window_pad_bins: usize,
) -> Vec<f64> {
    assert!(
        nominal_energies_ev.len() >= 2,
        "a detector time axis needs at least two nominal energies, got {}",
        nominal_energies_ev.len()
    );
    let kl = TOF_FACTOR * flight_path_m;
    let mut times: Vec<f64> = nominal_energies_ev
        .iter()
        .map(|&e| timing_offset_us + kl / e.sqrt())
        .collect();
    times.reverse();
    let first_width = times[1] - times[0];
    let last = times.len() - 1;
    let last_width = times[last] - times[last - 1];

    let mut edges = Vec::with_capacity(times.len() + 1 + 2 * window_pad_bins);
    for pad in (1..=window_pad_bins).rev() {
        edges.push(times[0] - (0.5 + pad as f64) * first_width);
    }
    edges.push(times[0] - 0.5 * first_width);
    for pair in times.windows(2) {
        edges.push(0.5 * (pair[0] + pair[1]));
    }
    edges.push(times[last] + 0.5 * last_width);
    for pad in 1..=window_pad_bins {
        edges.push(times[last] + (0.5 + pad as f64) * last_width);
    }
    edges
}

impl Truth {
    /// The energy grid the fit is given: one per detector bin under the
    /// nominal clock, ascending.
    pub fn nominal_energies_ev(&self) -> Vec<f64> {
        exact_true_energies(
            &self.detector_time_edges_us,
            self.timing_offset_us,
            self.flight_path_m,
            0.0,
            1.0,
        )
        .expect("valid detector time axis")
    }

    /// The energies a neutron recorded on this instrument actually had.
    pub fn true_energies_ev(&self) -> Vec<f64> {
        exact_true_energies(
            &self.detector_time_edges_us,
            self.timing_offset_us,
            self.flight_path_m,
            self.t0_us,
            self.l_scale,
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

        let n_bins = self.detector_time_edges_us.len() - 1;
        let incident_fluence_weights = vec![self.open_beam_counts_per_bin; n_bins];

        let signal = two_arm_count_response(
            &true_energies_ev,
            &incident_fluence_weights,
            &transmission,
            &self.detector_time_edges_us,
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
            detector_time_edges_us: self.detector_time_edges_us.clone(),
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
