//! Synthetic counts-domain measurements with known ground truth:
//!
//! ```text
//! E_true = exact_true_energies(source edges, timing_offset, L, t0, L_scale, nodes_per_bin)
//! T_j    = exp(-sum_i (n d)_i sigma_i(E_true_j; T))
//! O_i    = sum_j F_j R_ij + B_o,i        S_i = sum_j F_j T_j R_ij + B_s,i
//! ```
//!
//! with `R_ij` the detector-bin response of the true resolution kernel read
//! against the true clock `timing_offset + t0` over `L · L_scale`, per-bin
//! backgrounds that differ between the arms, and the recorded counts a
//! Poisson draw around `O_i` and `S_i`.

use nereids_endf::resonance::ResonanceData;
use nereids_physics::counts_response::{add_count_backgrounds, two_arm_count_response};
use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use nereids_physics::transmission::{SampleParams, forward_model};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Poisson};

use crate::pipeline::{
    exact_node_fluence, exact_quadrature_edges, exact_true_energies, pad_detector_edges,
};

/// Everything injected into a synthetic measurement, shared across pixels.
///
/// `t0_us` and `l_scale` describe the instrument the data was recorded on.
/// At `(0.0, 1.0)` the true energies equal the nominal ones, which is the
/// case for a fixture that is not exercising calibration recovery.
pub struct Truth {
    /// Detector-time bin edges the counts are binned into (µs), ascending.
    pub detector_time_edges_us: Vec<f64>,
    /// Bins of the end bin's width the source is lit past each end of the
    /// route's quadrature.
    pub source_pad_bins: usize,
    /// Quadrature nodes per bin the counts are synthesized with.
    pub nodes_per_bin: usize,
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
    /// Incident neutrons per lit bin, before the response and before
    /// background.
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
    /// Incident neutrons per bin of the route's quadrature, before the
    /// response and before background.
    pub incident_fluence_weights: Vec<f64>,
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
    let last = times.len() - 1;
    let mut edges = Vec::with_capacity(times.len() + 1);
    edges.push(times[0] - 0.5 * (times[1] - times[0]));
    for pair in times.windows(2) {
        edges.push(0.5 * (pair[0] + pair[1]));
    }
    edges.push(times[last] + 0.5 * (times[last] - times[last - 1]));
    pad_detector_edges(&edges, window_pad_bins, window_pad_bins).expect("a padded time axis")
}

impl Truth {
    /// The route's quadrature bins for this window under the nominal clock.
    ///
    /// # Panics
    /// Panics if the detector time axis is not a valid quadrature support.
    pub fn quadrature_edges(&self) -> Vec<f64> {
        exact_quadrature_edges(
            &self.detector_time_edges_us,
            self.timing_offset_us,
            &self.resolution,
        )
        .expect("valid detector time axis")
    }

    /// The bins the source is lit on: the quadrature extended by
    /// `source_pad_bins` past each end.
    ///
    /// # Panics
    /// Panics if the detector time axis is not a valid quadrature support.
    pub fn source_edges(&self) -> Vec<f64> {
        pad_detector_edges(
            &self.quadrature_edges(),
            self.source_pad_bins,
            self.source_pad_bins,
        )
        .expect("a padded time axis")
    }

    fn source_fluence(&self) -> Vec<f64> {
        vec![self.open_beam_counts_per_bin; self.source_edges().len() - 1]
    }

    /// Incident neutrons per bin of the route's quadrature, before the
    /// response.
    ///
    /// # Panics
    /// Panics if the detector time axis is not a valid quadrature support.
    pub fn fluence_per_bin(&self) -> Vec<f64> {
        let fluence = self.source_fluence();
        fluence[self.source_pad_bins..fluence.len() - self.source_pad_bins].to_vec()
    }

    /// The energy grid a fit at `nodes_per_bin` is given: the quadrature
    /// under the nominal clock, ascending.
    ///
    /// # Panics
    /// Panics if the detector time axis is not a valid quadrature support.
    pub fn nominal_energies_ev(&self, nodes_per_bin: usize) -> Vec<f64> {
        exact_true_energies(
            &self.quadrature_edges(),
            self.timing_offset_us,
            self.flight_path_m,
            0.0,
            1.0,
            nodes_per_bin,
        )
        .expect("valid detector time axis")
    }

    /// The energies of the source's neutrons on this instrument, at the
    /// synthesis quadrature, ascending.
    ///
    /// # Panics
    /// Panics if the detector time axis is not a valid quadrature support.
    pub fn true_energies_ev(&self) -> Vec<f64> {
        exact_true_energies(
            &self.source_edges(),
            self.timing_offset_us,
            self.flight_path_m,
            self.t0_us,
            self.l_scale,
            self.nodes_per_bin,
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
        let node_fluence = exact_node_fluence(&self.source_fluence(), self.nodes_per_bin);
        let response = self
            .resolution
            .with_flight_path(self.flight_path_m * self.l_scale)
            .expect("valid flight path");
        let signal = two_arm_count_response(
            &true_energies_ev,
            &node_fluence,
            &transmission,
            &self.detector_time_edges_us,
            self.timing_offset_us + self.t0_us,
            &response,
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
            incident_fluence_weights: self.fluence_per_bin(),
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
