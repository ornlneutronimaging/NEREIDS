//! The weight with which the beam times the transmission at each calculation
//! point enters each detector time bin.
//!
//! Between neighbouring points the product is a straight line in flight time
//! `u = TOF_FACTOR · L / √E`, and each straight-line piece is integrated
//! against every bin's probability, so the counts are
//!
//! ```text
//! C_k = Σ_j φ_j T_j W_jk,    W_jk = ∫ h_j(u) P_k(u) du
//! ```
//!
//! with `φ` the beam per µs of flight time, `h_j` the hat function that is 1
//! at point `j` and 0 at its neighbours, and `P_k(u)` the chance that the
//! resolution records a neutron of flight time `u` in bin `k`.
//!
//! SAMMY Ref: `udr/mudr4.f` Ud_Convolute and Udr_Add, which integrate
//! piecewise-linear pieces against the tabulated resolution.

use std::fmt;

use rayon::prelude::*;

use crate::resolution::{ResolutionFunction, ResolutionParseError, TOF_FACTOR};

const KRONROD_NODES: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
const KRONROD_WEIGHTS: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_83,
];
const GAUSS_WEIGHTS: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

const WEIGHT_TOLERANCE: f64 = 1e-10;

const MAX_HALVINGS: usize = 40;

#[derive(Debug)]
pub enum BinWeightsError {
    /// The calculation energies are not finite, positive and strictly
    /// ascending, or there are fewer than two.
    InvalidEnergies,
    /// The resolution rejected an energy, the time edges or `t0`.
    Resolution(ResolutionParseError),
    /// The weights of the piece between these energies did not reach the
    /// tolerance within the allowed halvings.
    NotConverged { low_ev: f64, high_ev: f64 },
}

impl fmt::Display for BinWeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEnergies => write!(
                f,
                "calculation energies must be at least two, finite, positive and strictly ascending"
            ),
            Self::Resolution(e) => write!(f, "resolution: {e}"),
            Self::NotConverged { low_ev, high_ev } => write!(
                f,
                "bin weights between {low_ev} and {high_ev} eV did not converge"
            ),
        }
    }
}

impl std::error::Error for BinWeightsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resolution(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ResolutionParseError> for BinWeightsError {
    fn from(e: ResolutionParseError) -> Self {
        Self::Resolution(e)
    }
}

#[derive(Debug, Clone)]
pub struct BinWeights {
    row_offsets: Vec<usize>,
    bins: Vec<u32>,
    weights: Vec<f64>,
    n_bins: usize,
    flight_path_m: f64,
}

type Piece = (Vec<(u32, f64)>, Vec<(u32, f64)>);

impl BinWeights {
    /// Weights for calculation `energies` in eV, the detector's `time_edges_us`
    /// and the clock offset `t0_us`, with neutrons recorded by `resolution`
    /// over its own flight path.
    ///
    /// # Errors
    /// [`BinWeightsError`] for invalid energies, a resolution that rejects
    /// its inputs (a Gaussian resolution always does), or a piece whose
    /// weights do not converge.
    pub fn new(
        energies: &[f64],
        time_edges_us: &[f64],
        t0_us: f64,
        resolution: &ResolutionFunction,
    ) -> Result<Self, BinWeightsError> {
        if energies.len() < 2
            || energies.iter().any(|e| !(e.is_finite() && *e > 0.0))
            || energies.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(BinWeightsError::InvalidEnergies);
        }
        let n_bins = time_edges_us.len().saturating_sub(1);
        let narrowest_bin_us = time_edges_us
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, f64::min);
        let flight_path_m = resolution.flight_path_m();
        let kl = TOF_FACTOR * flight_path_m;
        let probabilities =
            |u: f64| resolution.detector_bin_probabilities((kl / u).powi(2), time_edges_us, t0_us);
        let pieces: Vec<Piece> = energies
            .par_windows(2)
            .map(|pair| {
                let (u_short, u_long) = (kl / pair[1].sqrt(), kl / pair[0].sqrt());
                match integrate_piece(&probabilities, u_short, u_long, narrowest_bin_us, n_bins)? {
                    Some([to_lower, to_upper]) => Ok((sparse(&to_lower), sparse(&to_upper))),
                    None => Err(BinWeightsError::NotConverged {
                        low_ev: pair[0],
                        high_ev: pair[1],
                    }),
                }
            })
            .collect::<Result<_, BinWeightsError>>()?;

        let mut row_offsets = Vec::with_capacity(energies.len() + 1);
        let (mut bins, mut weights) = (Vec::new(), Vec::new());
        row_offsets.push(0);
        for j in 0..energies.len() {
            let mut row = vec![0.0; n_bins];
            let above = pieces.get(j).map(|p| &p.0);
            let below = j.checked_sub(1).map(|i| &pieces[i].1);
            for &(k, w) in above.into_iter().chain(below).flatten() {
                row[k as usize] += w;
            }
            for (k, w) in row.into_iter().enumerate() {
                if w > 0.0 {
                    bins.push(k as u32);
                    weights.push(w);
                }
            }
            row_offsets.push(weights.len());
        }
        Ok(Self {
            row_offsets,
            bins,
            weights,
            n_bins,
            flight_path_m,
        })
    }

    pub fn n_points(&self) -> usize {
        self.row_offsets.len() - 1
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    pub fn flight_path_m(&self) -> f64 {
        self.flight_path_m
    }

    /// Nonzero `(bin, W_jk)` pairs of point `j`.
    pub fn row(&self, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.row_offsets[j]..self.row_offsets[j + 1];
        self.bins[range.clone()]
            .iter()
            .map(|&k| k as usize)
            .zip(self.weights[range].iter().copied())
    }

    /// `Σ_j values_j W_jk` for every bin `k`.
    ///
    /// # Panics
    /// If `values` does not hold one value per calculation point.
    pub fn apply(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(values.len(), self.n_points(), "one value per point");
        let mut counts = vec![0.0; self.n_bins];
        for (j, &v) in values.iter().enumerate() {
            for (k, w) in self.row(j) {
                counts[k] += v * w;
            }
        }
        counts
    }
}

fn sparse(row: &[f64]) -> Vec<(u32, f64)> {
    row.iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(k, &w)| (k as u32, w))
        .collect()
}

fn integrate_piece(
    probabilities: &(dyn Fn(f64) -> Result<Vec<f64>, ResolutionParseError> + Sync),
    u_short: f64,
    u_long: f64,
    narrowest_bin_us: f64,
    n_bins: usize,
) -> Result<Option<[Vec<f64>; 2]>, ResolutionParseError> {
    let length = u_long - u_short;
    let mut total = [vec![0.0; n_bins], vec![0.0; n_bins]];
    let parts = (length / narrowest_bin_us).ceil().max(1.0) as usize;
    let part = length / parts as f64;
    let mut stack: Vec<(f64, f64, usize)> = (0..parts)
        .map(|i| {
            (
                u_short + part * i as f64,
                u_short + part * (i + 1) as f64,
                0,
            )
        })
        .collect();
    while let Some((a, b, halvings)) = stack.pop() {
        let (centre, half) = (0.5 * (a + b), 0.5 * (b - a));
        let mut kronrod = [vec![0.0; n_bins], vec![0.0; n_bins]];
        let mut gauss = [vec![0.0; n_bins], vec![0.0; n_bins]];
        for (i, &x) in KRONROD_NODES.iter().enumerate() {
            let nodes = if x == 0.0 { 1 } else { 2 };
            for u in [centre - half * x, centre + half * x]
                .into_iter()
                .take(nodes)
            {
                let row = probabilities(u)?;
                let share = (u - u_short) / length;
                for (k, &p) in row.iter().enumerate() {
                    for (side, part) in [share * p, (1.0 - share) * p].into_iter().enumerate() {
                        kronrod[side][k] += KRONROD_WEIGHTS[i] * part;
                        if i % 2 == 1 {
                            gauss[side][k] += GAUSS_WEIGHTS[i / 2] * part;
                        }
                    }
                }
            }
        }
        let error = half
            * kronrod
                .iter()
                .zip(&gauss)
                .flat_map(|(kr, g)| kr.iter().zip(g).map(|(kr, g)| (kr - g).abs()))
                .sum::<f64>();
        if error <= WEIGHT_TOLERANCE * (b - a) {
            for (sum, part) in total.iter_mut().zip(&kronrod) {
                for (s, p) in sum.iter_mut().zip(part) {
                    *s += half * p;
                }
            }
        } else if halvings < MAX_HALVINGS {
            stack.push((a, centre, halvings + 1));
            stack.push((centre, b, halvings + 1));
        } else {
            return Ok(None);
        }
    }
    Ok(Some(total))
}
