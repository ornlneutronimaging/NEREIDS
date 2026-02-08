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

/// Conversion factor for TOF to energy: E [eV] = FACTOR * (L [m])^2 / (t [μs])^2.
///
/// Derived from E = (`m_n` / 2) * (L / t)^2 with unit conversions:
/// FACTOR = `m_n` [kg] * 10^12 / (2 * e [C]) ≈ 5227.04.
pub const TOF_TO_ENERGY_FACTOR: f64 = 5_227.037_589_040_3;

/// SAMMY `SM2` constant in units of us*sqrt(eV)/m.
///
/// Used in time-of-flight conversions: `t_us = SM2 * L_m / sqrt(E_eV)`.
pub const SAMMY_SM2_US_SQRT_EV_PER_M: f64 = 72.298_252_179_105_06;
