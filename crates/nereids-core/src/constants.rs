//! Physical constants used throughout NEREIDS.
//!
//! Values from CODATA 2018 recommended values.
//! Reference: <https://physics.nist.gov/cuu/Constants/>

/// Neutron mass in kg.
pub const NEUTRON_MASS_KG: f64 = 1.674_927_498_04e-27;

/// Neutron mass in atomic mass units (u).
pub const NEUTRON_MASS_AMU: f64 = 1.008_664_915_95;

/// Neutron mass in MeV/c².
pub const NEUTRON_MASS_MEV: f64 = 939.565_420_52;

/// Boltzmann constant in eV/K.
pub const BOLTZMANN_EV_PER_K: f64 = 8.617_333_262e-5;

/// Planck constant (reduced, ħ) in eV·s.
pub const HBAR_EV_S: f64 = 6.582_119_514e-16;

/// Speed of light in m/s.
pub const SPEED_OF_LIGHT: f64 = 2.997_924_58e8;

/// 1 eV in joules.
pub const EV_TO_JOULES: f64 = 1.602_176_634e-19;

/// Avogadro's number in mol⁻¹.
pub const AVOGADRO: f64 = 6.022_140_76e23;

/// Convert neutron energy (eV) to wavelength (Å).
///
/// λ = h / √(2·m·E), result in angstroms.
pub fn energy_to_wavelength_angstrom(energy_ev: f64) -> f64 {
    // λ(Å) = 0.2860 / √(E in eV)  (standard neutron relation)
    0.286_014_3 / energy_ev.sqrt()
}

/// Convert neutron time-of-flight (μs) and flight path (m) to energy (eV).
///
/// E = ½·m_n·(L/t)²
pub fn tof_to_energy(tof_us: f64, flight_path_m: f64) -> f64 {
    let t_s = tof_us * 1.0e-6;
    let v = flight_path_m / t_s;
    0.5 * NEUTRON_MASS_KG * v * v / EV_TO_JOULES
}

/// Convert neutron energy (eV) to time-of-flight (μs) given flight path (m).
pub fn energy_to_tof(energy_ev: f64, flight_path_m: f64) -> f64 {
    let v = (2.0 * energy_ev * EV_TO_JOULES / NEUTRON_MASS_KG).sqrt();
    (flight_path_m / v) * 1.0e6
}

// ── Numerical tolerances ─────────────────────────────────────────────
// Named constants for magic-number epsilons scattered across physics code.

/// Epsilon for floating-point comparison of quantum numbers (J, L, spin).
pub const QUANTUM_NUMBER_EPS: f64 = 1e-10;

/// Floor for Poisson model values to avoid log(0) in NLL computation.
pub const POISSON_EPSILON: f64 = 1e-10;

/// Floor for denominators in physics evaluations (penetrability, shift, etc.)
/// to avoid division by zero.
pub const DIVISION_FLOOR: f64 = 1e-50;

/// Smallest meaningful cross-section contribution (barns).
/// Terms below this are negligible and can be skipped.
pub const CROSS_SECTION_FLOOR: f64 = 1e-60;

/// Floor for pivot detection and division safety in numerical linear algebra
/// (LM solver, Gaussian elimination). Values below this indicate a
/// (near-)singular system.
pub const PIVOT_FLOOR: f64 = 1e-30;

/// Floor for avoiding log(0) or division by zero in general computations.
pub const LOG_FLOOR: f64 = 1e-300;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tof_energy_roundtrip() {
        let energy = 6.67; // eV (first U-238 resonance)
        let flight_path = 25.0; // meters (VENUS)
        let tof = energy_to_tof(energy, flight_path);
        let energy_back = tof_to_energy(tof, flight_path);
        assert!((energy - energy_back).abs() < 1e-10);
    }

    #[test]
    fn test_wavelength_thermal() {
        // Thermal neutrons at 0.0253 eV should have λ ≈ 1.8 Å
        let lambda = energy_to_wavelength_angstrom(0.0253);
        assert!((lambda - 1.8).abs() < 0.1);
    }
}
