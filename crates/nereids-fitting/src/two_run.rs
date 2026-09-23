//! Expected counts of an open-beam run and a sample run from the counts
//! measurement equation:
//!
//! ```text
//! O_k = Σ_j φ_j W_jk,    S_k = c · Σ_j φ_j T_j W_jk,
//! T_j = exp(−Σ_i n_i σ_i(E_j, temperature))
//! ```
//!
//! at calculation energies `E_j`, with [`BinWeights`] `W`, the beam `φ` from
//! a [`BeamSpline`], `c` the sample run's beam relative to the open-beam
//! run's, and Doppler-broadened cross sections `σ_i`.

use nereids_endf::resonance::ResonanceData;
use nereids_physics::bin_weights::BinWeights;
use nereids_physics::continuous_doppler::{self, Channel, QuadratureBudget, TierOneBroadening};
use nereids_physics::resolution::TOF_FACTOR;

use crate::beam::BeamSpline;
use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};

pub struct Points<'a> {
    energies: &'a [f64],
    weights: &'a BinWeights,
    basis: Vec<[(usize, f64); 4]>,
    n_beam: usize,
}

impl<'a> Points<'a> {
    /// `energies` ascending in eV with their `weights`, the beam described by
    /// the knots of `beam`, over the flight path `flight_path_m`.
    ///
    /// # Panics
    /// If `weights` does not have one row per energy.
    pub fn new(
        energies: &'a [f64],
        weights: &'a BinWeights,
        beam: &BeamSpline,
        flight_path_m: f64,
    ) -> Self {
        assert_eq!(energies.len(), weights.n_points(), "one row per energy");
        let kl = TOF_FACTOR * flight_path_m;
        Self {
            energies,
            weights,
            basis: energies
                .iter()
                .map(|&e| beam.basis(kl / e.sqrt()))
                .collect(),
            n_beam: beam.coefficients().len(),
        }
    }

    fn beam(&self, coefficients: &[f64]) -> Vec<f64> {
        self.basis
            .iter()
            .map(|b| {
                b.iter()
                    .map(|&(i, w)| w * coefficients[i])
                    .sum::<f64>()
                    .exp()
            })
            .collect()
    }

    fn beam_columns(
        &self,
        values: &[f64],
        free: &[usize],
        offset: usize,
        jacobian: &mut FlatMatrix,
    ) {
        for (col, &index) in free.iter().enumerate() {
            if index >= self.n_beam {
                continue;
            }
            let g: Vec<f64> = self
                .basis
                .iter()
                .zip(values)
                .map(|(b, &v)| b.iter().filter(|p| p.0 == index).map(|p| p.1).sum::<f64>() * v)
                .collect();
            for (k, c) in self.weights.apply(&g).into_iter().enumerate() {
                *jacobian.get_mut(offset + k, col) = c;
            }
        }
    }
}

/// The open-beam run alone; the parameters are the beam's coefficients.
pub struct OpenBeamModel<'a> {
    points: &'a Points<'a>,
}

impl<'a> OpenBeamModel<'a> {
    pub fn new(points: &'a Points<'a>) -> Self {
        Self { points }
    }
}

impl FitModel for OpenBeamModel<'_> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        Ok(self.points.weights.apply(&self.points.beam(params)))
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let mut jacobian = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        let phi = self.points.beam(params);
        self.points
            .beam_columns(&phi, free_param_indices, 0, &mut jacobian);
        Some(jacobian)
    }
}

/// Both runs; the parameters are the beam's coefficients, then one areal
/// density per isotope in atoms/barn, then the temperature in K.  The
/// prediction is the open-beam run's counts followed by the sample run's.
pub struct TwoRunModel<'a> {
    points: &'a Points<'a>,
    isotopes: &'a [ResonanceData],
    charge_ratio: f64,
}

impl<'a> TwoRunModel<'a> {
    /// `isotopes` in the sample, whose run received `charge_ratio` times the
    /// open-beam run's beam.
    pub fn new(points: &'a Points<'a>, isotopes: &'a [ResonanceData], charge_ratio: f64) -> Self {
        Self {
            points,
            isotopes,
            charge_ratio,
        }
    }

    pub fn temperature_index(&self) -> usize {
        self.points.n_beam + self.isotopes.len()
    }

    fn cross_sections(&self, temperature_k: f64) -> Result<Vec<TierOneBroadening>, FittingError> {
        self.isotopes
            .iter()
            .map(|isotope| {
                continuous_doppler::broaden_with_budget(
                    self.points.energies,
                    isotope,
                    temperature_k,
                    Channel::Total,
                    QuadratureBudget::default(),
                )
                .map_err(|e| FittingError::EvaluationFailed(format!("Doppler: {e}")))
            })
            .collect()
    }

    fn transmission(&self, params: &[f64], sigma: &[TierOneBroadening]) -> Vec<f64> {
        let n = self.points.n_beam;
        (0..self.points.energies.len())
            .map(|j| {
                let depth: f64 = sigma
                    .iter()
                    .enumerate()
                    .map(|(i, s)| params[n + i] * s.values[j])
                    .sum();
                (-depth).exp()
            })
            .collect()
    }
}

impl FitModel for TwoRunModel<'_> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let sigma = self.cross_sections(params[self.temperature_index()])?;
        let phi = self.points.beam(params);
        let transmitted: Vec<f64> = phi
            .iter()
            .zip(self.transmission(params, &sigma))
            .map(|(p, t)| self.charge_ratio * p * t)
            .collect();
        let mut counts = self.points.weights.apply(&phi);
        counts.extend(self.points.weights.apply(&transmitted));
        Ok(counts)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let temperature_index = self.temperature_index();
        let sigma = self.cross_sections(params[temperature_index]).ok()?;
        let n_bins = self.points.weights.n_bins();
        let mut jacobian = FlatMatrix::zeros(y_current.len(), free_param_indices.len());
        let phi = self.points.beam(params);
        let transmitted: Vec<f64> = phi
            .iter()
            .zip(self.transmission(params, &sigma))
            .map(|(p, t)| self.charge_ratio * p * t)
            .collect();
        self.points
            .beam_columns(&phi, free_param_indices, 0, &mut jacobian);
        self.points
            .beam_columns(&transmitted, free_param_indices, n_bins, &mut jacobian);
        let n_beam = self.points.n_beam;
        for (col, &index) in free_param_indices.iter().enumerate() {
            if index < n_beam {
                continue;
            }
            let rate: Vec<f64> = if index == temperature_index {
                (0..phi.len())
                    .map(|j| {
                        sigma
                            .iter()
                            .enumerate()
                            .map(|(i, s)| params[n_beam + i] * s.derivatives[j])
                            .sum()
                    })
                    .collect()
            } else {
                sigma[index - n_beam].values.clone()
            };
            let g: Vec<f64> = rate.iter().zip(&transmitted).map(|(r, s)| -r * s).collect();
            for (k, c) in self.points.weights.apply(&g).into_iter().enumerate() {
                *jacobian.get_mut(n_bins + k, col) = c;
            }
        }
        Some(jacobian)
    }
}
