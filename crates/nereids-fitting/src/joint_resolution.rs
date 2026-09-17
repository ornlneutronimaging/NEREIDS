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
//! whose temperature uncertainty already contains everything the calibrant
//! failed to pin down. Nothing is summarized, so nothing is approximated.
//!
//! The calibrant's own density and temperature are what make it a calibrant
//! and stay fixed; only the resolution is shared.

use std::sync::Arc;

use nereids_endf::resonance::ResonanceData;
use nereids_physics::resolution::ResolutionParams;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

use crate::error::FittingError;
use crate::lm::FitModel;

/// One spectrum in a joint fit: its grid, what is in the beam, and how its
/// free parameters are read out of the shared vector.
struct Spectrum {
    energies: Vec<f64>,
    resonance_data: Vec<ResonanceData>,
    /// `params[density_indices[i]]` is isotope `i`'s areal density, or the
    /// density is fixed at `fixed_densities[i]` when this is empty.
    density_indices: Vec<usize>,
    fixed_densities: Vec<f64>,
    /// `params[temperature_index]` is the temperature, else `temperature_k`.
    temperature_index: Option<usize>,
    temperature_k: f64,
}

impl Spectrum {
    fn sample_params(&self, params: &[f64]) -> Result<SampleParams, FittingError> {
        let temperature_k = match self.temperature_index {
            Some(i) => params[i],
            None => self.temperature_k,
        };
        let isotopes = self
            .resonance_data
            .iter()
            .enumerate()
            .map(|(i, rd)| {
                let density = if self.density_indices.is_empty() {
                    self.fixed_densities[i]
                } else {
                    params[self.density_indices[i]]
                };
                (rd.clone(), density)
            })
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
    sample: Spectrum,
    calibrant: Spectrum,
    flight_path_m: f64,
    /// `params[delta_t_index]` / `params[delta_l_index]` are the Gaussian
    /// widths, shared by both spectra.
    delta_t_index: usize,
    delta_l_index: usize,
}

impl JointResolutionModel {
    /// Build the joint model.
    ///
    /// `sample_*` describe the unknown spectrum: its densities come from
    /// `sample_density_indices` and its temperature from `temperature_index`
    /// when fitted, otherwise from `sample_temperature_k`. `calibrant_*`
    /// describe the known one, whose densities and temperature are fixed —
    /// that is what makes it a calibrant.
    ///
    /// # Errors
    /// [`FittingError::InvalidConfig`] when a spectrum's grid is empty, when
    /// an index collection does not match its isotope count, or when the
    /// resolution indices collide with another parameter.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_energies: Vec<f64>,
        sample_resonance_data: Vec<ResonanceData>,
        sample_density_indices: Vec<usize>,
        sample_temperature_k: f64,
        temperature_index: Option<usize>,
        calibrant_energies: Vec<f64>,
        calibrant_resonance_data: Vec<ResonanceData>,
        calibrant_densities: Vec<f64>,
        calibrant_temperature_k: f64,
        flight_path_m: f64,
        delta_t_index: usize,
        delta_l_index: usize,
    ) -> Result<Self, FittingError> {
        if sample_energies.is_empty() || calibrant_energies.is_empty() {
            return Err(FittingError::InvalidConfig(
                "both the sample and the calibrant need a non-empty energy grid".into(),
            ));
        }
        if sample_density_indices.len() != sample_resonance_data.len() {
            return Err(FittingError::InvalidConfig(format!(
                "sample has {} density indices for {} isotopes",
                sample_density_indices.len(),
                sample_resonance_data.len(),
            )));
        }
        if calibrant_densities.len() != calibrant_resonance_data.len() {
            return Err(FittingError::InvalidConfig(format!(
                "calibrant has {} densities for {} isotopes",
                calibrant_densities.len(),
                calibrant_resonance_data.len(),
            )));
        }
        if delta_t_index == delta_l_index {
            return Err(FittingError::InvalidConfig(
                "the two resolution widths must be separate parameters".into(),
            ));
        }
        let taken = [delta_t_index, delta_l_index];
        if sample_density_indices
            .iter()
            .chain(temperature_index.iter())
            .any(|i| taken.contains(i))
        {
            return Err(FittingError::InvalidConfig(
                "a resolution index collides with a density or temperature parameter".into(),
            ));
        }
        Ok(Self {
            sample: Spectrum {
                energies: sample_energies,
                resonance_data: sample_resonance_data,
                density_indices: sample_density_indices,
                fixed_densities: Vec::new(),
                temperature_index,
                temperature_k: sample_temperature_k,
            },
            calibrant: Spectrum {
                energies: calibrant_energies,
                resonance_data: calibrant_resonance_data,
                density_indices: Vec::new(),
                fixed_densities: calibrant_densities,
                temperature_index: None,
                temperature_k: calibrant_temperature_k,
            },
            flight_path_m,
            delta_t_index,
            delta_l_index,
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
            params[self.delta_t_index],
            params[self.delta_l_index],
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
            let model = JointResolutionModel::new(
                energies.clone(),
                vec![iso.clone()],
                vec![0],
                T_TRUE,
                Some(1),
                energies.clone(),
                vec![iso.clone()],
                vec![DENSITY],
                T_TRUE,
                L,
                2,
                3,
            )
            .unwrap();
            // Sample arm only: the calibrant half is masked out by fitting
            // against the sample data alone.
            let sample_only = SampleOnly { inner: model };
            let mut params = ParameterSet::new(vec![
                FitParameter::non_negative("density", DENSITY),
                FitParameter::non_negative("temperature_k", 285.0),
                FitParameter::fixed("delta_t_us", w),
                FitParameter::fixed("delta_l_m", dl),
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
        let model = JointResolutionModel::new(
            energies.clone(),
            vec![iso.clone()],
            vec![0],
            T_TRUE,
            Some(1),
            energies.clone(),
            vec![iso.clone()],
            vec![DENSITY],
            T_TRUE,
            L,
            2,
            3,
        )
        .unwrap();
        let mut joint_data = sample_data.clone();
        joint_data.extend_from_slice(&cal_data);
        let joint_unc = vec![NOISE; joint_data.len()];
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("density", DENSITY),
            FitParameter::non_negative("temperature_k", 285.0),
            FitParameter::non_negative("delta_t_us", W_TRUE),
            FitParameter::non_negative("delta_l_m", DL_TRUE),
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
