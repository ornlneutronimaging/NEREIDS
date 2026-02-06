//! Physical constants used in neutron resonance calculations.

/// Neutron mass in atomic mass units.
pub const NEUTRON_MASS_AMU: f64 = 1.008_664_916_06;

/// Neutron mass in kg.
pub const NEUTRON_MASS_KG: f64 = 1.674_927_498_04e-27;

/// Boltzmann constant in eV/K.
pub const BOLTZMANN_EV_PER_K: f64 = 8.617_333_262e-5;

/// Planck's constant in eV·s.
pub const PLANCK_EV_S: f64 = 4.135_667_696e-15;

/// Speed of light in m/s.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Conversion factor: 1 eV in Joules.
pub const EV_TO_JOULE: f64 = 1.602_176_634e-19;

/// Conversion factor for TOF (ms) to energy (eV) given flight path (m).
/// E = (`m_n` / 2) * (L / t)^2, but working in convenient units.
/// E [eV] = `5.227_037e6` / (L [m])^2 * (t [μs])^(-2)
/// (This constant absorbs `m_n/2` and unit conversions.)
pub const TOF_TO_ENERGY_FACTOR: f64 = 5.227_037e6;
