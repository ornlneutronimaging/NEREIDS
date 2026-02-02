//! Physics kernels for NEREIDS.

use nereids_core::{RMatrixParameters, TransmissionModel, TransmissionRequest, TransmissionResult};

#[derive(Debug, Default)]
pub struct StubTransmissionModel;

impl TransmissionModel for StubTransmissionModel {
    fn transmission(
        &self,
        _params: &RMatrixParameters,
        request: &TransmissionRequest,
    ) -> TransmissionResult {
        TransmissionResult {
            transmission: vec![1.0; request.energy.len()],
        }
    }
}
