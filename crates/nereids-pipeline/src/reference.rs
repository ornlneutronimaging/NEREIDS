use nereids_physics::resolution::{ResolutionFunction, ResolutionParseError, TOF_FACTOR};
use rayon::prelude::*;

/// The detector's time bins and the arrival time of a neutron of known
/// energy: `t0_us + TOF_FACTOR · flight_path_m / √E` plus the delay drawn
/// from `resolution`.
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

impl Instrument {
    /// Expected counts per time bin from a beam `beam(E)` in neutrons per eV
    /// through a sample of transmission `transmission(energies)`, integrated
    /// over energies in `energy_range_ev` with flight-time step `step_us`.
    ///
    /// `transmission` receives ascending energies and returns one value per
    /// energy.
    ///
    /// # Errors
    /// [`ResolutionParseError`] if the flight path, the time edges or an
    /// energy are rejected by the resolution.
    pub fn expected_counts(
        &self,
        beam: &(dyn Fn(f64) -> f64 + Sync),
        transmission: &dyn Fn(&[f64]) -> Vec<f64>,
        energy_range_ev: (f64, f64),
        step_us: f64,
    ) -> Result<ExpectedCounts, ResolutionParseError> {
        let resolution = self.resolution.with_flight_path(self.flight_path_m)?;
        let kl = TOF_FACTOR * self.flight_path_m;
        let (u_lo, u_hi) = (kl / energy_range_ev.1.sqrt(), kl / energy_range_ev.0.sqrt());
        let steps = ((u_hi - u_lo) / step_us).ceil() as usize;
        let h = (u_hi - u_lo) / steps as f64;
        let mut energies: Vec<f64> = (0..=steps)
            .map(|j| (kl / (u_lo + h * j as f64)).powi(2))
            .collect();
        energies.reverse();
        let t = transmission(&energies);

        let n_bins = self.time_edges_us.len() - 1;
        let probabilities =
            |e: f64| resolution.detector_bin_probabilities(e, &self.time_edges_us, self.t0_us);
        let counts = energies
            .par_iter()
            .zip(&t)
            .enumerate()
            .try_fold(
                || vec![0.0; n_bins],
                |mut acc, (j, (&e, &t_e))| {
                    let end = j == 0 || j == steps;
                    let weight = if end { 0.5 * h } else { h };
                    let u = kl / e.sqrt();
                    let density = weight * beam(e) * t_e * 2.0 * e / u;
                    for (a, p) in acc.iter_mut().zip(probabilities(e)?) {
                        *a += density * p;
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || vec![0.0; n_bins],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    Ok(a)
                },
            )?;
        let reach = |e: f64| probabilities(e).map(|p| p.iter().sum::<f64>());
        Ok(ExpectedCounts {
            counts,
            edge_probability: [reach(energies[0])?, reach(energies[steps])?],
        })
    }
}
