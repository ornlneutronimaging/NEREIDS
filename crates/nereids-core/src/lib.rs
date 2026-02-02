//! Core data models and traits for NEREIDS.

pub type Energy = f64;

#[derive(Debug, Clone, Default)]
pub struct Resonance {
    pub energy: Energy,
    pub gamma_n: f64,
    pub gamma_g: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SpinGroup {
    pub spin: f64,
    pub resonances: Vec<Resonance>,
}

#[derive(Debug, Clone, Default)]
pub struct Isotope {
    pub name: String,
    pub spin_groups: Vec<SpinGroup>,
}

#[derive(Debug, Clone, Default)]
pub struct RMatrixParameters {
    pub isotopes: Vec<Isotope>,
}

#[derive(Debug, Clone)]
pub struct TransmissionRequest {
    pub energy: Vec<Energy>,
    pub thickness_cm: f64,
    pub number_density: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TransmissionResult {
    pub transmission: Vec<f64>,
}

pub trait TransmissionModel {
    fn transmission(
        &self,
        params: &RMatrixParameters,
        request: &TransmissionRequest,
    ) -> TransmissionResult;
}
