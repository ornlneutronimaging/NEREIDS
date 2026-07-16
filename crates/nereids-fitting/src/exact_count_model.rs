//! Exact fixed-instrument model for a joint open/sample count likelihood.

use nereids_core::constants::PIVOT_FLOOR;
use nereids_physics::counts_response::DetectorBinResponseMatrix;

use crate::error::FittingError;
use crate::lm::{FitModel, FlatMatrix};

/// Maps a true-energy transmission model into measured detector-bin ratios.
///
/// For a fixed response matrix `R` and fixed incident fluence weights `F`, the
/// model supplied to the joint-Poisson objective is evaluated as
///
/// ```text
/// T_eff,i = sum_j F_j T_j R_ij / sum_j F_j R_ij.
/// ```
///
/// The open and sample arms are therefore broadened separately. This is not
/// the invalid post-hoc shortcut `R[T]`.
pub struct ExactTwoArmRatioModel {
    inner: Box<dyn FitModel>,
    weighted_response: Vec<f64>,
    open_expectation: Vec<f64>,
    n_true_energies: usize,
    n_detector_bins: usize,
}

impl ExactTwoArmRatioModel {
    /// Build the reusable fixed-response model.
    pub fn new(
        inner: Box<dyn FitModel>,
        response: &DetectorBinResponseMatrix,
        incident_fluence_weights: &[f64],
    ) -> Result<Self, FittingError> {
        if incident_fluence_weights.len() != response.n_true_energies() {
            return Err(FittingError::LengthMismatch {
                expected: response.n_true_energies(),
                actual: incident_fluence_weights.len(),
                field: "incident_fluence_weights",
            });
        }
        for (index, &fluence) in incident_fluence_weights.iter().enumerate() {
            if !fluence.is_finite() || fluence < 0.0 {
                return Err(FittingError::InvalidConfig(format!(
                    "incident_fluence_weights[{index}] must be finite and >= 0, got {fluence}"
                )));
            }
        }
        let fluence_scale = incident_fluence_weights
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        if fluence_scale <= 0.0 {
            return Err(FittingError::InvalidConfig(
                "incident_fluence_weights must contain at least one positive value".into(),
            ));
        }

        let n_true_energies = response.n_true_energies();
        let n_detector_bins = response.n_detector_bins();
        let mut weighted_response = vec![0.0; n_true_energies * n_detector_bins];
        let mut open_expectation = vec![0.0; n_detector_bins];
        let mut open_compensation = vec![0.0; n_detector_bins];
        for (true_index, &fluence) in incident_fluence_weights.iter().enumerate() {
            let fluence = fluence / fluence_scale;
            for detector_bin in 0..n_detector_bins {
                let value = fluence * response.probability(true_index, detector_bin);
                weighted_response[true_index * n_detector_bins + detector_bin] = value;
                compensated_add(
                    &mut open_expectation[detector_bin],
                    &mut open_compensation[detector_bin],
                    value,
                );
            }
        }
        for detector_bin in 0..n_detector_bins {
            open_expectation[detector_bin] += open_compensation[detector_bin];
        }

        Ok(Self {
            inner,
            weighted_response,
            open_expectation,
            n_true_energies,
            n_detector_bins,
        })
    }

    /// Expected open-arm source shape in detector-bin order.
    pub fn open_expectation(&self) -> &[f64] {
        &self.open_expectation
    }

    fn map_true_energy_values(&self, values: &[f64]) -> Result<Vec<f64>, FittingError> {
        if values.len() != self.n_true_energies {
            return Err(FittingError::LengthMismatch {
                expected: self.n_true_energies,
                actual: values.len(),
                field: "true_energy_model",
            });
        }
        let mut sample = vec![0.0; self.n_detector_bins];
        let mut compensation = vec![0.0; self.n_detector_bins];
        for (true_index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(FittingError::EvaluationFailed(format!(
                    "true-energy model output[{true_index}] is not finite: {value}"
                )));
            }
            for detector_bin in 0..self.n_detector_bins {
                compensated_add(
                    &mut sample[detector_bin],
                    &mut compensation[detector_bin],
                    value
                        * self.weighted_response[true_index * self.n_detector_bins + detector_bin],
                );
            }
        }

        Ok(sample
            .into_iter()
            .zip(compensation)
            .zip(&self.open_expectation)
            .map(|((sum, correction), &open)| {
                if open > PIVOT_FLOOR {
                    (sum + correction) / open
                } else {
                    1.0
                }
            })
            .collect())
    }
}

impl FitModel for ExactTwoArmRatioModel {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        let true_transmission = self.inner.evaluate(params)?;
        self.map_true_energy_values(&true_transmission)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        _y_current: &[f64],
    ) -> Option<FlatMatrix> {
        let true_transmission = self.inner.evaluate(params).ok()?;
        if true_transmission.len() != self.n_true_energies {
            return None;
        }
        let inner_jacobian =
            self.inner
                .analytical_jacobian(params, free_param_indices, &true_transmission)?;
        if inner_jacobian.nrows != self.n_true_energies
            || inner_jacobian.ncols != free_param_indices.len()
        {
            return None;
        }

        let mut output = FlatMatrix::zeros(self.n_detector_bins, free_param_indices.len());
        for true_index in 0..self.n_true_energies {
            for detector_bin in 0..self.n_detector_bins {
                let open = self.open_expectation[detector_bin];
                if open <= PIVOT_FLOOR {
                    continue;
                }
                let weight =
                    self.weighted_response[true_index * self.n_detector_bins + detector_bin] / open;
                for column in 0..free_param_indices.len() {
                    *output.get_mut(detector_bin, column) +=
                        weight * inner_jacobian.get(true_index, column);
                }
            }
        }
        Some(output)
    }
}

#[inline]
fn compensated_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let next = *sum + value;
    if sum.abs() >= value.abs() {
        *compensation += (*sum - next) + value;
    } else {
        *compensation += (value - next) + *sum;
    }
    *sum = next;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nereids_physics::resolution::{ResolutionFunction, TOF_FACTOR, TabulatedResolution};

    use super::*;

    struct LinearTrueEnergyModel;

    impl FitModel for LinearTrueEnergyModel {
        fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
            Ok(vec![params[0], 1.0 - 0.5 * params[0]])
        }

        fn analytical_jacobian(
            &self,
            _params: &[f64],
            free_param_indices: &[usize],
            _y_current: &[f64],
        ) -> Option<FlatMatrix> {
            if free_param_indices != [0] {
                return None;
            }
            Some(FlatMatrix {
                data: vec![1.0, -0.5],
                nrows: 2,
                ncols: 1,
            })
        }
    }

    fn triangle_response() -> ResolutionFunction {
        ResolutionFunction::Tabulated(Arc::new(
            TabulatedResolution::from_kernels(
                vec![25.0],
                vec![(vec![-1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0])],
                25.0,
            )
            .expect("valid triangle response"),
        ))
    }

    #[test]
    fn maps_values_and_analytical_jacobian_through_separate_arms() {
        let arrival_0 = TOF_FACTOR * 25.0 / 25.0_f64.sqrt();
        let arrival_1 = arrival_0 + 1.0;
        let energy_1 = (TOF_FACTOR * 25.0 / arrival_1).powi(2);
        let response = DetectorBinResponseMatrix::new(
            &[25.0, energy_1],
            &[arrival_0 - 1.0, arrival_0, arrival_0 + 1.0, arrival_0 + 2.0],
            0.0,
            &triangle_response(),
        )
        .expect("valid exact response");
        let model =
            ExactTwoArmRatioModel::new(Box::new(LinearTrueEnergyModel), &response, &[100.0, 200.0])
                .expect("valid exact ratio model");

        let values = model.evaluate(&[0.2]).expect("model evaluation");
        let expected = [0.2, 2.0 / 3.0, 0.9];
        for (got, want) in values.iter().zip(expected) {
            assert!((got - want).abs() < 2.0e-12, "{got} != {want}");
        }

        let jacobian = model
            .analytical_jacobian(&[0.2], &[0], &values)
            .expect("analytical Jacobian");
        let expected_jacobian = [1.0, 0.0, -0.5];
        for (row, want) in expected_jacobian.into_iter().enumerate() {
            assert!((jacobian.get(row, 0) - want).abs() < 2.0e-12);
        }
    }
}
