use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR};
use rayon::prelude::*;

/// The detector's time bins and the arrival time of a neutron of known
/// energy: `t0_us + TOF_FACTOR · flight_path_m / √E` plus an offset drawn
/// from `resolution`, whose zero follows that resolution's anchor (see
/// [`ResolutionFunction::detector_bin_probabilities`]).
pub struct Instrument {
    /// Time-bin edges in µs, strictly ascending.
    pub time_edges_us: Vec<f64>,
    pub flight_path_m: f64,
    pub t0_us: f64,
    pub resolution: ResolutionFunction,
}

/// Counts per time bin, and the chance that a neutron at the lowest and at
/// the highest integrated energy is recorded in any bin.  Both chances must
/// be negligible for the counts to include every neutron that reaches the
/// bins.
pub struct ExpectedCounts {
    pub counts: Vec<f64>,
    pub edge_probability: [f64; 2],
}

const CHUNK: usize = 1024;

impl Instrument {
    /// Expected counts per time bin from a beam `beam(E)` in neutrons per eV
    /// through a sample of transmission `transmission(energies)`, integrated
    /// over energies in `energy_range_ev` with flight-time step `step_us`.
    ///
    /// `transmission` receives ascending energies and returns one value per
    /// energy.
    ///
    /// # Panics
    /// If the energy range is not `0 < low < high`, the step is not finite
    /// and positive, `transmission` returns a different number of values
    /// than it was given energies, or the resolution rejects the flight
    /// path, the time edges, `t0_us` or an energy.
    pub fn expected_counts(
        &self,
        beam: &(dyn Fn(f64) -> f64 + Sync),
        transmission: &dyn Fn(&[f64]) -> Vec<f64>,
        energy_range_ev: (f64, f64),
        step_us: f64,
    ) -> ExpectedCounts {
        let (e_lo, e_hi) = energy_range_ev;
        assert!(
            e_lo.is_finite() && e_hi.is_finite() && 0.0 < e_lo && e_lo < e_hi,
            "the energy range must satisfy 0 < low < high, got {energy_range_ev:?}"
        );
        assert!(
            step_us.is_finite() && step_us > 0.0,
            "the flight-time step must be finite and positive, got {step_us}"
        );
        let resolution = self
            .resolution
            .with_flight_path(self.flight_path_m)
            .expect("the resolution accepts the flight path");
        let probabilities = |e: f64| {
            resolution
                .detector_bin_probabilities(e, &self.time_edges_us, self.t0_us)
                .expect("the resolution accepts the energy, the time edges and t0")
        };
        let reach = |e: f64| probabilities(e).iter().sum::<f64>();
        let edge_probability = [reach(e_lo), reach(e_hi)];

        let kl = TOF_FACTOR * self.flight_path_m;
        let (u_lo, u_hi) = (kl / e_hi.sqrt(), kl / e_lo.sqrt());
        let steps = ((u_hi - u_lo) / step_us).ceil() as usize;
        let h = (u_hi - u_lo) / steps as f64;
        let mut energies: Vec<f64> = (0..=steps)
            .map(|j| (kl / (u_lo + h * j as f64)).powi(2))
            .collect();
        energies.reverse();
        let t = transmission(&energies);
        assert_eq!(
            t.len(),
            energies.len(),
            "transmission must return one value per energy"
        );

        let n_bins = self.time_edges_us.len() - 1;
        let partials: Vec<Vec<f64>> = energies
            .par_chunks(CHUNK)
            .zip(t.par_chunks(CHUNK))
            .enumerate()
            .map(|(chunk, (es, ts))| {
                let mut acc = vec![0.0; n_bins];
                for (i, (&e, &t_e)) in es.iter().zip(ts).enumerate() {
                    let j = chunk * CHUNK + i;
                    let weight = if j == 0 || j == steps { 0.5 * h } else { h };
                    let u = kl / e.sqrt();
                    let density = weight * beam(e) * t_e * 2.0 * e / u;
                    for (a, p) in acc.iter_mut().zip(probabilities(e)) {
                        *a += density * p;
                    }
                }
                acc
            })
            .collect();
        let mut counts = vec![0.0; n_bins];
        for partial in partials {
            for (c, p) in counts.iter_mut().zip(partial) {
                *c += p;
            }
        }
        ExpectedCounts {
            counts,
            edge_probability,
        }
    }
}
