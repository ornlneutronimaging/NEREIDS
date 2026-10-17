//! Doppler broadening by nuclei bound in a harmonic crystal.
//!
//! Lamb's incoherent harmonic model, the one SAMMY's crystal-lattice model
//! (DOPUSH, after NJOY's LEAPR) expands in phonons: a neutron of energy `E`
//! absorbed by a nucleus of mass ratio `A` leaves an energy `ε` to the
//! lattice, drawn from
//!
//! ```text
//! P(ε) = (1/2π) ∫ e^{iεt} Φ(t) dt,   Φ(t) = exp(R·[G(t) − G(0)]),
//! G(t) = ∫ ρ(ω)/ω · [coth(ω/2kT)·cos ωt − i·sin ωt] dω,   R = E/(A + 1),
//! ```
//!
//! `ρ` the phonon density of states, so the cross section is
//!
//! ```text
//! σ_L(E) = e^{−2W}·σ(E + R) + ∫ P_c(ε)·σ(E + R − ε) dε,   2W = R·G(0),
//! ```
//!
//! the first term the zero-phonon line and `P_c = P − e^{−2W}·δ`.  `P` has
//! mean `R` and variance `2R·kT_eff` with `kT_eff = ½∫ρ·ω·coth(ω/2kT) dω`,
//! so a free gas at `T_eff` has the same two moments.
//!
//! `σ` is sampled once on a uniform grid of step `h`, the integral is the
//! trapezoid rule on that grid, and `P_c` comes on the grid's nodes from a
//! fast Fourier transform of `Φ − e^{−2W}`, sampled at `t = j·dt` up to
//! `π/h`.  A resonance of total width `Γ` makes the trapezoid rule's error
//! `≈ e^{−πΓ/h}`, so `h ≤ πΓ/30` for every resonance the kernel reaches;
//! and `h ≤ π·ω_max/100` resolves the phonon spectrum's structure, whose
//! weight `e^{−2W}·2W` the one-phonon term carries.

use std::f64::consts::PI;

use nereids_core::constants::BOLTZMANN_EV_PER_K;
use nereids_endf::resonance::ResonanceData;
use num_complex::Complex64;
use rayon::prelude::*;

use crate::continuous_doppler::{Channel, TierOneBroadening};
use crate::doppler::{DopplerError, validate_doppler_grid};
use crate::reich_moore::CrossSectionPlan;

/// Standard deviations of the kernel, and phonon energies beyond them, its
/// window spans on each side of its mean.
const KERNEL_SIGMAS: f64 = 8.0;
const KERNEL_PHONONS: f64 = 10.0;

/// Most points of the cross section's grid, or of one transform.
const MAX_POINTS: usize = 1 << 22;

/// Gauss–Legendre 8-point nodes and weights on `[−1, 1]`, non-negative half.
const GAUSS_LEGENDRE_8: [(f64, f64); 4] = [
    (0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
    (0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
];

/// Errors from lattice broadening.
#[derive(Debug)]
pub enum LatticeError {
    /// The phonon spectrum is not a valid density of states.
    InvalidSpectrum(String),
    /// The temperature is not finite and positive.
    InvalidTemperature(f64),
    /// The energy grid is invalid.
    Grid(DopplerError),
    /// The kernel reaches energies at or below zero.
    BelowZero(f64),
    /// The cross section's grid or a transform would exceed 2²² points.
    TooManyPoints(usize),
}

impl std::fmt::Display for LatticeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSpectrum(message) => write!(f, "invalid phonon spectrum: {message}"),
            Self::InvalidTemperature(t) => {
                write!(f, "the temperature must be finite and positive, got {t} K")
            }
            Self::Grid(error) => write!(f, "{error}"),
            Self::BelowZero(e) => {
                write!(
                    f,
                    "the lattice kernel reaches {e} eV, at or below zero energy"
                )
            }
            Self::TooManyPoints(n) => {
                write!(
                    f,
                    "lattice broadening needs {n} points, more than {MAX_POINTS}"
                )
            }
        }
    }
}

impl std::error::Error for LatticeError {}

impl From<DopplerError> for LatticeError {
    fn from(error: DopplerError) -> Self {
        Self::Grid(error)
    }
}

/// A crystal's phonon density of states `ρ(ω)`, `ω` in eV: the values
/// `ρ_i` at `ω = i·step`, `ρ_0 = 0`, quadratic in `ω` up to the first
/// point, linear between the others and zero beyond the last, scaled to
/// unit area.
#[derive(Debug, Clone, PartialEq)]
pub struct PhononSpectrum {
    step_ev: f64,
    density: Vec<f64>,
}

impl PhononSpectrum {
    /// The spectrum with values `density` every `step_ev` from 0.
    ///
    /// # Errors
    /// [`LatticeError::InvalidSpectrum`] unless `step_ev` is finite and
    /// positive and `density` holds at least two finite values, 0 or more,
    /// the first 0 and another positive.
    pub fn new(step_ev: f64, mut density: Vec<f64>) -> Result<Self, LatticeError> {
        if !(step_ev.is_finite() && step_ev > 0.0)
            || density.len() < 2
            || density[0] != 0.0
            || !density.iter().all(|r| r.is_finite() && *r >= 0.0)
            || !density.iter().any(|r| *r > 0.0)
        {
            return Err(LatticeError::InvalidSpectrum(format!(
                "needs a finite positive step and at least two finite values, 0 or more, the \
                 first 0 and another positive; got step {step_ev} and {} values",
                density.len()
            )));
        }
        let area = step_ev * density[1] / 3.0
            + step_ev
                * density[1..]
                    .windows(2)
                    .map(|w| 0.5 * (w[0] + w[1]))
                    .sum::<f64>();
        for r in &mut density {
            *r /= area;
        }
        Ok(Self { step_ev, density })
    }

    /// Debye's spectrum `3ω²/ω_D³` up to `ω_D = k·θ_D`, on `points` values.
    ///
    /// # Errors
    /// [`LatticeError::InvalidSpectrum`] unless `theta_k` is finite and
    /// positive and `points` is at least 2.
    pub fn debye(theta_k: f64, points: usize) -> Result<Self, LatticeError> {
        let cutoff = BOLTZMANN_EV_PER_K * theta_k;
        if !(cutoff.is_finite() && cutoff > 0.0) || points < 2 {
            return Err(LatticeError::InvalidSpectrum(format!(
                "Debye needs a finite positive temperature and at least 2 points, got {theta_k} K \
                 and {points}"
            )));
        }
        let step = cutoff / (points - 1) as f64;
        Self::new(
            step,
            (0..points)
                .map(|i| (i as f64 * step / cutoff).powi(2))
                .collect(),
        )
    }

    /// The highest phonon energy, eV.
    #[must_use]
    pub fn max_ev(&self) -> f64 {
        self.step_ev * (self.density.len() - 1) as f64
    }

    /// Quadrature nodes `(ω, w·ρ(ω))` over the spectrum, each panel spanning
    /// at most one radian of `ω·t` up to `t_max`.
    fn nodes(&self, t_max: f64) -> Vec<(f64, f64)> {
        let mut nodes = Vec::new();
        let h = self.step_ev;
        let pieces = ((h * t_max).ceil() as usize).max(1);
        for i in 0..self.density.len() - 1 {
            let (r0, r1) = (self.density[i], self.density[i + 1]);
            let at = |omega: f64| {
                if i == 0 {
                    r1 * (omega / h).powi(2)
                } else {
                    r0 + (r1 - r0) * (omega / h - i as f64)
                }
            };
            let width = h / pieces as f64;
            for p in 0..pieces {
                let centre = i as f64 * h + (p as f64 + 0.5) * width;
                for &(x, w) in &GAUSS_LEGENDRE_8 {
                    for omega in [centre - 0.5 * width * x, centre + 0.5 * width * x] {
                        nodes.push((omega, 0.5 * width * w * at(omega)));
                    }
                }
            }
        }
        nodes
    }
}

/// The spectrum's sums at one temperature: `G(t_j)` and `∂G/∂T` at
/// `t_j = j·dt`, `j = 0..=half`, and `kT_eff`.
struct Phonons {
    g: Vec<Complex64>,
    g_slope: Vec<f64>,
    kt_eff: f64,
}

impl Phonons {
    fn new(nodes: &[(f64, f64)], temperature_k: f64, dt: f64, half: usize) -> Self {
        let kt = BOLTZMANN_EV_PER_K * temperature_k;
        let terms: Vec<(f64, f64, f64, f64)> = nodes
            .iter()
            .map(|&(omega, weight)| {
                let x = omega / (2.0 * kt);
                let coth = 1.0 / x.tanh();
                let slope = x / (temperature_k * x.sinh().powi(2));
                (omega, weight / omega, coth, slope)
            })
            .collect();
        let kt_eff = 0.5
            * terms
                .iter()
                .map(|&(omega, a, coth, _)| a * omega * omega * coth)
                .sum::<f64>();
        let (g, g_slope) = (0..=half)
            .into_par_iter()
            .map(|j| {
                let t = j as f64 * dt;
                let (mut g, mut slope) = (Complex64::new(0.0, 0.0), 0.0);
                for &(omega, a, coth, dcoth) in &terms {
                    let (sin, cos) = (omega * t).sin_cos();
                    g += a * Complex64::new(coth * cos, -sin);
                    slope += a * dcoth * cos;
                }
                (g, slope)
            })
            .unzip();
        Self { g, g_slope, kt_eff }
    }
}

/// `values` replaced by its discrete Fourier transform,
/// `values[m] ← Σ_j values[j]·e^{−2πi·mj/N}`, `N` a power of 2.
fn dft(values: &mut [Complex64]) {
    let n = values.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = Complex64::from_polar(1.0, -2.0 * PI / len as f64);
        for start in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (values[start + k], values[start + k + len / 2] * w);
                values[start + k] = a + b;
                values[start + k + len / 2] = a - b;
                w *= step;
            }
        }
        len <<= 1;
    }
}

/// Value and temperature derivative of `channel` at every target energy,
/// broadened by nuclei bound in a harmonic crystal with phonon spectrum
/// `spectrum` at `temperature_k`.  `negative_values` counts the negative
/// values; nothing is roundoff-limited.
///
/// # Errors
/// [`LatticeError::InvalidTemperature`], [`LatticeError::Grid`] for an empty
/// or invalid grid, [`LatticeError::BelowZero`], [`LatticeError::TooManyPoints`].
pub fn broaden_in_lattice(
    energies: &[f64],
    data: &ResonanceData,
    temperature_k: f64,
    spectrum: &PhononSpectrum,
    channel: Channel,
) -> Result<TierOneBroadening, LatticeError> {
    if !(temperature_k.is_finite() && temperature_k > 0.0) {
        return Err(LatticeError::InvalidTemperature(temperature_k));
    }
    validate_doppler_grid(energies)?;
    if energies.is_empty() {
        return Err(DopplerError::EmptyGrid.into());
    }
    let recoil = |energy: f64| energy / (data.awr + 1.0);
    let (lowest, highest) = (energies[0], energies[energies.len() - 1]);
    let omega_max = spectrum.max_ev();
    let probe = Phonons::new(&spectrum.nodes(0.0), temperature_k, 0.0, 0);
    let reach =
        KERNEL_SIGMAS * (2.0 * recoil(highest) * probe.kt_eff).sqrt() + KERNEL_PHONONS * omega_max;
    let narrowest = data
        .ranges
        .iter()
        .filter(|range| range.is_evaluable())
        .flat_map(|range| &range.l_groups)
        .flat_map(|group| &group.resonances)
        .filter(|r| r.energy >= lowest - reach - 1.0 && r.energy <= highest + reach + 1.0)
        .map(|r| r.gn.abs() + r.gg.abs() + r.gfa.abs() + r.gfb.abs())
        .filter(|&width| width > 0.0)
        .fold(f64::INFINITY, f64::min);
    let h = (PI * narrowest / 30.0).min(PI * omega_max / 100.0);
    let points = (2.0 * reach / h).ceil() as usize + 1;
    let transform = points.next_power_of_two();
    let origin = lowest - reach - 2.0 * h;
    if origin <= 0.0 {
        return Err(LatticeError::BelowZero(origin));
    }
    let grid_points = ((highest + reach - origin) / h).ceil() as usize + 3;
    for needed in [transform, grid_points] {
        if needed > MAX_POINTS {
            return Err(LatticeError::TooManyPoints(needed));
        }
    }
    let dt = 2.0 * PI / (transform as f64 * h);
    let half = transform / 2;
    let phonons = Phonons::new(&spectrum.nodes(PI / h), temperature_k, dt, half);
    let plan = CrossSectionPlan::new(data);
    let sigma = |energy: f64| channel.pick(&plan.evaluate_one(energy));
    let grid: Vec<f64> = (0..grid_points)
        .into_par_iter()
        .map(|m| sigma(origin + m as f64 * h))
        .collect();
    let g0 = phonons.g[0].re;
    let g0_slope = phonons.g_slope[0];
    let results: Vec<(f64, f64)> = energies
        .par_iter()
        .map(|&energy| {
            let r = recoil(energy);
            let zero = (-r * g0).exp();
            let zero_slope = -zero * r * g0_slope;
            let shift = energy + r - origin;
            let mut buffer = vec![Complex64::new(0.0, 0.0); transform];
            for j in 0..half {
                let phi = (r * (phonons.g[j] - g0)).exp();
                let value = phi - zero;
                let slope = phi * r * (phonons.g_slope[j] - g0_slope) - zero_slope;
                let t = j as f64 * dt;
                let rotate = Complex64::from_polar(dt / (2.0 * PI), shift * t);
                buffer[j] = rotate * (value + Complex64::i() * slope);
                if j > 0 {
                    let back = rotate.conj() * (value.conj() + Complex64::i() * slope.conj());
                    buffer[transform - j] = back;
                }
            }
            dft(&mut buffer);
            let first = ((energy - reach - origin) / h).floor().max(0.0) as usize;
            let last = (((energy + reach - origin) / h).ceil() as usize).min(grid_points - 1);
            let (mut value, mut slope) = (0.0, 0.0);
            for (m, &s) in grid.iter().enumerate().take(last + 1).skip(first) {
                let kernel = buffer[m % transform];
                value += kernel.re * s;
                slope += kernel.im * s;
            }
            let at_recoil = sigma(energy + r);
            (
                h * value + zero * at_recoil,
                h * slope + zero_slope * at_recoil,
            )
        })
        .collect();
    let (values, derivatives): (Vec<f64>, Vec<f64>) = results.into_iter().unzip();
    if let Some((k, &value)) = values
        .iter()
        .chain(&derivatives)
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(DopplerError::NonFiniteIntegral {
            energy_ev: energies[k % energies.len()],
            value,
            derivative: k >= energies.len(),
        }
        .into());
    }
    let negative_values = values.iter().filter(|&&v| v < 0.0).count();
    Ok(TierOneBroadening {
        values,
        derivatives,
        negative_values,
        roundoff_limited: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transform_is_the_discrete_fourier_transform() {
        let n = 32;
        let values: Vec<Complex64> = (0..n)
            .map(|j| Complex64::new((0.7 * j as f64).sin(), (1.3 * j as f64).cos() - 0.2))
            .collect();
        let mut fast = values.clone();
        dft(&mut fast);
        for (m, fast) in fast.iter().enumerate() {
            let direct: Complex64 = values
                .iter()
                .enumerate()
                .map(|(j, v)| v * Complex64::from_polar(1.0, -2.0 * PI * (m * j) as f64 / n as f64))
                .sum();
            assert!((fast - direct).norm() <= 1e-12, "{m}: {fast} {direct}");
        }
    }
}
