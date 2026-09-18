//! Fit a sample and the calibrant that measured its resolution together.
//!
//! Pinning a calibrated resolution into a sample fit reports the temperature
//! as more certain than it is: resolution width and temperature broaden the
//! line the same way, so the uncertainty that belongs to their degeneracy is
//! dropped. Carrying the calibration forward as a Gaussian prior does not
//! recover it — the calibration's own uncertainty is neither Gaussian nor
//! separable. Its objective is flat where the kernel is narrower than the
//! line it broadens and a wall above, and for the Gaussian family the two
//! width coordinates trade off almost exactly, so a per-parameter sigma
//! describes a direction the calibration never moves in.
//!
//! What has none of those problems is the calibrant's residuals themselves.
//! This model evaluates the sample and the calibrant against ONE resolution
//! drawn from the shared parameter vector and returns both predictions, so
//! the optimizer sees a single objective
//!
//! ```text
//! chi^2(T, n, w) = chi^2_sample(T, n, w) + chi^2_calibrant(w)
//! ```
//!
//! whose temperature uncertainty already contains what the calibrant failed
//! to pin down.
//!
//! The objective is exact: no part of the calibration is summarized, so the
//! shape the summary would have lost is still there. The uncertainty READ OFF
//! it is not. `temperature_k_unc` comes from the optimizer's local curvature
//! at the solution and is a Gaussian approximation like any other, so on a
//! surface with a flat side and a wall it describes the solution's
//! neighbourhood rather than the whole interval. What the joint objective
//! fixes is that the neighbourhood is now the right one — it includes the
//! resolution's freedom instead of holding it fixed.
//!
//! The calibrant's own density and temperature are what make it a calibrant
//! and stay fixed; only the resolution is shared.
//!
//! The two shared slots hold the SQUARED widths, in µs² and m². The kernel
//! combines the timing and flight-path terms in quadrature,
//! `W² = timing(Δt)² + path(ΔL)²`, so a width itself enters `W` quadratically
//! and `∂W/∂ΔL` is exactly zero at `ΔL = 0`. `W²` is linear in the squares,
//! so `∂W/∂(ΔL²)` is finite there and a width seeded at zero is still fitted.

use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_physics::resolution::ResolutionParams;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

use crate::error::FittingError;
use crate::lm::FitModel;

/// Where a spectrum's areal densities come from.
pub enum Densities {
    /// `params[i]` for each index: the fit determines them.
    Fitted(Vec<usize>),
    /// One value per isotope, held at what the caller knows it to be.
    Known(Vec<f64>),
}

impl Densities {
    fn len(&self) -> usize {
        match self {
            Self::Fitted(indices) => indices.len(),
            Self::Known(values) => values.len(),
        }
    }

    fn at(&self, i: usize, params: &[f64]) -> f64 {
        match self {
            Self::Fitted(indices) => params[indices[i]],
            Self::Known(values) => values[i],
        }
    }
}

/// One spectrum in a joint fit: its grid, what is in the beam, and how its
/// free parameters are read out of the shared vector.
pub struct SpectrumSpec {
    /// Energy grid (eV), ascending.
    pub energies: Vec<f64>,
    /// One entry per isotope in the beam.
    pub resonance_data: Vec<ResonanceData>,
    /// Areal densities, fitted or known.
    pub densities: Densities,
    /// `params[temperature_index]` is the temperature, else `temperature_k`.
    pub temperature_index: Option<usize>,
    /// Temperature (K) when it is not fitted.
    pub temperature_k: f64,
}

impl SpectrumSpec {
    fn sample_params(&self, params: &[f64]) -> Result<SampleParams, FittingError> {
        let temperature_k = match self.temperature_index {
            Some(i) => params[i],
            None => self.temperature_k,
        };
        let isotopes = self
            .resonance_data
            .iter()
            .enumerate()
            .map(|(i, rd)| (rd.clone(), self.densities.at(i, params)))
            .collect();
        SampleParams::new(temperature_k, isotopes)
            .map_err(|e| FittingError::EvaluationFailed(format!("sample params: {e:?}")))
    }

    fn predict(
        &self,
        params: &[f64],
        instrument: &InstrumentParams,
    ) -> Result<Vec<f64>, FittingError> {
        let sample = self.sample_params(params)?;
        transmission::forward_model(&self.energies, &sample, Some(instrument))
            .map_err(|e| FittingError::EvaluationFailed(format!("forward: {e:?}")))
    }
}

/// A sample and its calibrant, sharing one Gaussian resolution.
///
/// [`FitModel::evaluate`] returns the sample's predictions followed by the
/// calibrant's, so the caller fits against the two spectra concatenated in
/// that order.
pub struct JointResolutionModel {
    sample: SpectrumSpec,
    calibrant: SpectrumSpec,
    flight_path_m: f64,
    /// `params[delta_t_sq_index]` / `params[delta_l_sq_index]` are the SQUARED
    /// Gaussian widths, in µs² and m², shared by both spectra.
    delta_t_sq_index: usize,
    delta_l_sq_index: usize,
}

impl JointResolutionModel {
    /// Build the joint model.
    ///
    /// `delta_t_sq_index` / `delta_l_sq_index` are the shared slots holding
    /// the squared widths. Everything else about each arm, including whether
    /// its densities and temperature are fitted, is in its
    /// [`SpectrumSpec`] — the calibrant's being known is what makes it a
    /// calibrant.
    ///
    /// # Errors
    /// [`FittingError::InvalidConfig`] when a spectrum's grid is empty, when
    /// its density count does not match its isotope count, or when two
    /// parameters share a slot.
    pub fn new(
        sample: SpectrumSpec,
        calibrant: SpectrumSpec,
        flight_path_m: f64,
        delta_t_sq_index: usize,
        delta_l_sq_index: usize,
    ) -> Result<Self, FittingError> {
        for (label, spec) in [("sample", &sample), ("calibrant", &calibrant)] {
            if spec.energies.is_empty() {
                return Err(FittingError::InvalidConfig(format!(
                    "the {label} needs a non-empty energy grid"
                )));
            }
            if spec.densities.len() != spec.resonance_data.len() {
                return Err(FittingError::InvalidConfig(format!(
                    "the {label} has {} densities for {} isotopes",
                    spec.densities.len(),
                    spec.resonance_data.len(),
                )));
            }
        }
        // Every slot the model reads must name one quantity. Two of them
        // sharing an index makes a single optimizer coordinate move two
        // different physical things at once.
        let mut slots = vec![delta_t_sq_index, delta_l_sq_index];
        for spec in [&sample, &calibrant] {
            if let Densities::Fitted(indices) = &spec.densities {
                slots.extend(indices);
            }
            slots.extend(spec.temperature_index);
        }
        let mut seen = slots.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != slots.len() {
            return Err(FittingError::InvalidConfig(format!(
                "two parameters share a slot: {slots:?}"
            )));
        }
        Ok(Self {
            sample,
            calibrant,
            flight_path_m,
            delta_t_sq_index,
            delta_l_sq_index,
        })
    }

    /// Number of data points the sample contributes, i.e. where the
    /// calibrant's predictions start in [`FitModel::evaluate`]'s output.
    #[must_use]
    pub fn sample_len(&self) -> usize {
        self.sample.energies.len()
    }

    /// Total length of the concatenated prediction.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sample.energies.len() + self.calibrant.energies.len()
    }

    /// Whether the joint prediction is empty. Never true: both grids are
    /// checked non-empty at construction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The resolution both spectra are evaluated with at this probe.
    fn instrument(&self, params: &[f64]) -> Result<Arc<InstrumentParams>, FittingError> {
        let resolution = ResolutionParams::new(
            self.flight_path_m,
            params[self.delta_t_sq_index].max(0.0).sqrt(),
            params[self.delta_l_sq_index].max(0.0).sqrt(),
            0.0,
        )
        .map_err(|e| FittingError::EvaluationFailed(format!("shared resolution: {e:?}")))?;
        Ok(Arc::new(InstrumentParams {
            resolution: nereids_physics::resolution::ResolutionFunction::Gaussian(resolution),
        }))
    }
}

impl FitModel for JointResolutionModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let instrument = self.instrument(params)?;
        let mut out = self.sample.predict(params, &instrument)?;
        out.extend(self.calibrant.predict(params, &instrument)?);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{LmConfig, levenberg_marquardt};
    use crate::parameters::{FitParameter, ParameterSet};
    use crate::resolution_calib::{CalibrationConfig, ResolutionFamily, calibrate_resolution};
    use nereids_endf::resonance::test_support::synthetic_isotope;
    use nereids_physics::resolution::ResolutionFunction;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;
    use rand_distr::{Distribution, Normal};

    const L: f64 = 25.0;
    const T_TRUE: f64 = 300.0;
    const DENSITY: f64 = 2.0e-3;
    const W_TRUE: f64 = 0.30;
    const DL_TRUE: f64 = 0.05;
    const NOISE: f64 = 0.002;

    /// The two arms for a test: the sample's density and temperature fitted
    /// at the given slots, the calibrant's known.
    fn arms(
        iso: &ResonanceData,
        energies: &[f64],
        temperature_index: Option<usize>,
    ) -> (SpectrumSpec, SpectrumSpec) {
        (
            SpectrumSpec {
                energies: energies.to_vec(),
                resonance_data: vec![iso.clone()],
                densities: Densities::Fitted(vec![0]),
                temperature_index,
                temperature_k: T_TRUE,
            },
            SpectrumSpec {
                energies: energies.to_vec(),
                resonance_data: vec![iso.clone()],
                densities: Densities::Known(vec![DENSITY]),
                temperature_index: None,
                temperature_k: T_TRUE,
            },
        )
    }

    fn fixture() -> (ResonanceData, Vec<f64>, Vec<f64>) {
        let iso = synthetic_isotope(72, 178, 20.0, 0.05, 0.06);
        let energies: Vec<f64> = (0..120).map(|i| 18.0 + i as f64 * 0.04).collect();
        let sample = SampleParams::new(T_TRUE, vec![(iso.clone(), DENSITY)]).unwrap();
        let inst = InstrumentParams {
            resolution: ResolutionFunction::Gaussian(
                ResolutionParams::new(L, W_TRUE, DL_TRUE, 0.0).unwrap(),
            ),
        };
        let truth = transmission::forward_model(&energies, &sample, Some(&inst)).unwrap();
        (iso, energies, truth)
    }

    /// The joint fit reports the temperature uncertainty the two-stage
    /// procedure actually has; pinning the calibrated kernel reports less.
    ///
    /// The oracle is repetition. Calibrate on a fresh noisy calibrant, fit a
    /// fresh noisy sample with that resolution pinned, and the fitted
    /// temperature scatters by the full two-stage uncertainty. The pinned fit
    /// cannot see the calibration noise and reports only its own; the joint
    /// fit has the calibrant residuals in its objective and should report
    /// both.
    #[test]
    fn the_joint_fit_reports_the_temperature_uncertainty_pinning_drops() {
        const REALIZATIONS: usize = 16;
        let (iso, energies, truth) = fixture();
        let unc = vec![NOISE; energies.len()];
        let mut rng = ChaCha12Rng::seed_from_u64(20260917);
        let normal = Normal::new(0.0, NOISE).unwrap();
        let mut noisy =
            || -> Vec<f64> { truth.iter().map(|t| t + normal.sample(&mut rng)).collect() };

        let cfg = CalibrationConfig {
            ic_n_energies: 8,
            ic_n_tau: 32,
            max_iter: 400,
            ..Default::default()
        };
        let calibrant_sample = SampleParams::new(T_TRUE, vec![(iso.clone(), DENSITY)]).unwrap();

        let pinned_fit = |data: &[f64], w: f64, dl: f64| -> Option<(f64, f64)> {
            let model = {
                let (sample, calibrant) = arms(&iso, &energies, Some(1));
                JointResolutionModel::new(sample, calibrant, L, 2, 3)
            }
            .unwrap();
            // Sample arm only: the calibrant half is masked out by fitting
            // against the sample data alone.
            let sample_only = SampleOnly { inner: model };
            let mut params = ParameterSet::new(vec![
                FitParameter::non_negative("density", DENSITY),
                FitParameter::non_negative("temperature_k", 285.0),
                FitParameter::fixed("delta_t_us_sq", w * w),
                FitParameter::fixed("delta_l_m_sq", dl * dl),
            ]);
            let r = levenberg_marquardt(
                &sample_only,
                data,
                &unc,
                &mut params,
                &LmConfig {
                    compute_covariance: true,
                    max_iter: 100,
                    ..Default::default()
                },
            )
            .ok()?;
            let sigma = r.uncertainties.as_ref()?.get(1).copied()?;
            Some((r.params[1], sigma))
        };

        let mut fitted = Vec::new();
        let mut pinned_sigmas = Vec::new();
        for _ in 0..REALIZATIONS {
            let cal_data = noisy();
            let Ok(cal) = calibrate_resolution(
                ResolutionFamily::Gaussian,
                &energies,
                &cal_data,
                &unc,
                &calibrant_sample,
                &cfg,
            ) else {
                continue;
            };
            let ResolutionFunction::Gaussian(p) = &cal.resolution else {
                unreachable!()
            };
            let sample_data = noisy();
            if let Some((t, s)) = pinned_fit(&sample_data, p.delta_t_us(), p.delta_l_m()) {
                fitted.push(t);
                pinned_sigmas.push(s);
            }
        }
        assert!(
            fitted.len() >= REALIZATIONS / 2,
            "only {} of {REALIZATIONS} two-stage realizations produced a fit",
            fitted.len()
        );
        let n = fitted.len() as f64;
        let mean = fitted.iter().sum::<f64>() / n;
        let observed = (fitted.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let pinned = pinned_sigmas.iter().sum::<f64>() / n;

        // Joint fit on one realization: resolution free, calibrant in the
        // objective.
        let cal_data = noisy();
        let sample_data = noisy();
        let model = {
            let (sample, calibrant) = arms(&iso, &energies, Some(1));
            JointResolutionModel::new(sample, calibrant, L, 2, 3)
        }
        .unwrap();
        let mut joint_data = sample_data.clone();
        joint_data.extend_from_slice(&cal_data);
        let joint_unc = vec![NOISE; joint_data.len()];
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", DENSITY),
            FitParameter::non_negative("temperature_k", 285.0),
            FitParameter::non_negative("delta_t_us_sq", W_TRUE * W_TRUE),
            FitParameter::non_negative("delta_l_m_sq", DL_TRUE * DL_TRUE),
        ]);
        let r = levenberg_marquardt(
            &model,
            &joint_data,
            &joint_unc,
            &mut params,
            &LmConfig {
                compute_covariance: true,
                max_iter: 200,
                ..Default::default()
            },
        )
        .expect("joint fit runs");
        let joint = r.uncertainties.as_ref().expect("joint covariance")[1];

        eprintln!(
            "observed {observed:.4}  pinned {pinned:.4}  joint {joint:.4}  T {:.3}",
            r.params[1]
        );
        assert!(
            pinned < 0.9 * observed,
            "pinned sigma_T {pinned:.4} does not understate the two-stage scatter \
             {observed:.4}; without that gap this test cannot show the joint fit \
             recovering anything"
        );
        // The claim, and the only assertion a joint fit that ignored its
        // calibrant would fail: dropping the calibrant half leaves the
        // sample-only fit, whose sigma_T is the pinned one exactly.
        assert!(
            joint > pinned,
            "the joint fit reports sigma_T {joint:.4}, no more than the pinned \
             {pinned:.4}; the calibrant residuals are not reaching the objective"
        );
        let ratio = joint / observed;
        assert!(
            (0.5..=2.0).contains(&ratio),
            "the joint fit reports sigma_T {joint:.4} against an observed \
             two-stage scatter of {observed:.4} (ratio {ratio:.2})"
        );
    }

    /// A flight-path width seeded at zero is still fitted.
    ///
    /// The kernel combines the two terms in quadrature, so `W` depends on a
    /// width through its square and `dW/d(dL)` is exactly zero at `dL = 0`. A
    /// finite-difference optimizer probing that coordinate sees only the
    /// second-order term and the width never moves. The measure is the
    /// Jacobian column the optimizer actually gets at that point, against the
    /// timing column as the scale of a column it can follow.
    #[test]
    fn a_zero_flight_path_width_still_has_a_usable_jacobian_column() {
        const FD_STEP: f64 = 1.0e-6;

        let (iso, energies, _) = fixture();
        let model = {
            let (sample, calibrant) = arms(&iso, &energies, None);
            JointResolutionModel::new(sample, calibrant, L, 1, 2)
        }
        .unwrap();

        // params = [density, delta_t^2, delta_l^2], the flight-path width at
        // zero and the timing width at its usual scale.
        let base = [DENSITY, W_TRUE * W_TRUE, 0.0];
        let column = |slot: usize| -> f64 {
            let mut probed = base;
            probed[slot] += FD_STEP * (1.0 + base[slot].abs());
            let (a, b) = (
                model.evaluate(&base).expect("base evaluates"),
                model.evaluate(&probed).expect("probe evaluates"),
            );
            a.iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max)
                / FD_STEP
        };

        let timing = column(1);
        let path = column(2);
        assert!(
            timing > 0.0,
            "the timing column is {timing}, so there is no scale to compare against"
        );
        assert!(
            path > 0.01 * timing,
            "at a zero flight-path width the column is {path:.4e} against a \
             timing column of {timing:.4e}; a column that small is below the \
             noise of any real measurement and the width would never move"
        );
    }

    /// No two parameters may name the same slot.
    ///
    /// A shared index does not fail loudly — it makes one optimizer
    /// coordinate move two physical quantities at once, and the fit returns a
    /// number.
    #[test]
    fn parameters_sharing_an_index_are_rejected() {
        let (iso, energies, _) = fixture();
        let build = |density: usize, temperature: Option<usize>, dt: usize, dl: usize| {
            let (mut sample, calibrant) = arms(&iso, &energies, temperature);
            sample.densities = Densities::Fitted(vec![density]);
            JointResolutionModel::new(sample, calibrant, L, dt, dl)
        };
        assert!(
            build(0, Some(1), 2, 3).is_ok(),
            "the distinct layout is legal"
        );
        for (label, density, temperature, dt, dl) in [
            ("temperature on the density slot", 0, Some(0), 2, 3),
            ("width on the density slot", 0, Some(1), 0, 3),
            ("width on the temperature slot", 0, Some(1), 1, 3),
            ("the two widths on one slot", 0, Some(1), 2, 2),
        ] {
            assert!(
                build(density, temperature, dt, dl).is_err(),
                "{label} must be rejected"
            );
        }
    }

    /// The sample arm alone, for the pinned comparison.
    struct SampleOnly {
        inner: JointResolutionModel,
    }

    impl FitModel for SampleOnly {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            let mut all = self.inner.evaluate(params)?;
            all.truncate(self.inner.sample_len());
            Ok(all)
        }
    }
}
