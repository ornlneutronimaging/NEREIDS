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
///
/// Derived from h = 6.626_070_15e-34 J·s (exact, 2019 SI) and
/// e = 1.602_176_634e-19 C (exact): ħ = h / (2π·e).
/// Rounded to 10 significant figures.
pub const HBAR_EV_S: f64 = 6.582_119_569e-16;

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
///
/// # Domain contract
/// Returns [`f64::NAN`] for out-of-domain input rather than a misleading
/// value. Both arguments must be finite and strictly positive:
/// * a non-positive `tof_us` is unphysical — the unguarded formula squares
///   the velocity, so a *negative* TOF would return a *positive* energy, and
///   `tof_us == 0` would return `+∞`;
/// * a non-positive `flight_path_m` is equally unphysical — `flight_path_m
///   == 0` makes the velocity (and energy) `0`, and a *negative* flight path
///   would (after squaring) return a *positive* energy, masking a bad
///   detector geometry the same way a negative TOF would;
/// * a non-finite `tof_us` or `flight_path_m` cannot map to a real energy.
///
/// Callers that need to surface bad input as an error (e.g. the PyO3
/// boundary) check the input before calling or test the result with
/// `is_finite()`; the in-crate callers (`nereids_io::tof`, the GUI) already
/// guard `tof > 0 && finite` up-front, so this NaN sentinel never fires on
/// valid data.
pub fn tof_to_energy(tof_us: f64, flight_path_m: f64) -> f64 {
    // `is_finite()` first excludes NaN, so the `<= 0.0` total-order
    // comparisons that follow are well-defined (clippy's
    // `neg_cmp_op_on_partial_ord`).
    if !tof_us.is_finite() || tof_us <= 0.0 || !flight_path_m.is_finite() || flight_path_m <= 0.0 {
        return f64::NAN;
    }
    let t_s = tof_us * 1.0e-6;
    let v = flight_path_m / t_s;
    0.5 * NEUTRON_MASS_KG * v * v / EV_TO_JOULES
}

/// Convert neutron energy (eV) to time-of-flight (μs) given flight path (m).
///
/// # Domain contract
/// Mirrors [`tof_to_energy`]: returns [`f64::NAN`] unless *both* arguments are
/// finite and strictly positive. A non-positive or non-finite `energy_ev` (the
/// unguarded formula takes `√energy`, so a negative energy would yield a `NaN`
/// velocity and `energy_ev == 0` would yield `+∞`), and a non-positive or
/// non-finite `flight_path_m` (which would yield a zero or *negative* TOF — a
/// physically impossible time), are all rejected. This keeps the two
/// directions consistent — both refuse out-of-domain input instead of
/// returning a plausible-looking number.
pub fn energy_to_tof(energy_ev: f64, flight_path_m: f64) -> f64 {
    if !energy_ev.is_finite()
        || energy_ev <= 0.0
        || !flight_path_m.is_finite()
        || flight_path_m <= 0.0
    {
        return f64::NAN;
    }
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

/// Generic tiny positive floor used as a near-zero tolerance across physics
/// calculations (e.g., cross-sections in barns, energies in eV, widths,
/// dimensionless parameters). Values below this are treated as negligible.
pub const NEAR_ZERO_FLOOR: f64 = 1e-60;

/// Floor for pivot detection and division safety in numerical linear algebra
/// (LM solver, Gaussian elimination). Values below this indicate a
/// (near-)singular system.
pub const PIVOT_FLOOR: f64 = 1e-30;

/// Floor for Levenberg-Marquardt diagonal elements to ensure damping stability.
/// Intentionally much larger than PIVOT_FLOOR — LM requires a meaningful
/// minimum curvature for numerical stability of the trust-region step.
pub const LM_DIAGONAL_FLOOR: f64 = 1e-10;

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

    #[test]
    fn test_tof_to_energy_rejects_non_positive_and_non_finite() {
        let l = 25.0;
        // Negative TOF used to return a *positive* energy (v² hides the
        // sign), masking an upstream sign/loader bug.
        assert!(
            tof_to_energy(-100.0, l).is_nan(),
            "negative TOF must be NaN"
        );
        // Zero TOF used to return +∞.
        assert!(tof_to_energy(0.0, l).is_nan(), "zero TOF must be NaN");
        assert!(tof_to_energy(f64::NAN, l).is_nan(), "NaN TOF must stay NaN");
        assert!(
            tof_to_energy(f64::INFINITY, l).is_nan(),
            "Inf TOF must be NaN"
        );
        assert!(
            tof_to_energy(100.0, f64::NAN).is_nan(),
            "NaN flight path must be NaN"
        );
        assert!(
            tof_to_energy(100.0, f64::INFINITY).is_nan(),
            "Inf flight path must be NaN"
        );
        // Zero flight path used to return a finite 0 energy (v = L/t = 0),
        // masking a bad detector geometry.
        assert!(
            tof_to_energy(100.0, 0.0).is_nan(),
            "zero flight path must be NaN"
        );
        // Negative flight path used to return a *positive* energy (v² hides
        // the sign), the same trap as a negative TOF.
        assert!(
            tof_to_energy(100.0, -25.0).is_nan(),
            "negative flight path must be NaN"
        );
        // Valid input still produces a finite, positive energy.
        assert!(tof_to_energy(100.0, l).is_finite());
        assert!(tof_to_energy(100.0, l) > 0.0);
    }

    #[test]
    fn test_energy_to_tof_rejects_non_positive_and_non_finite() {
        let l = 25.0;
        // Negative energy used to return NaN already (√ of negative), but
        // zero energy returned +∞ — both are now an explicit NaN contract.
        assert!(
            energy_to_tof(-1.0, l).is_nan(),
            "negative energy must be NaN"
        );
        assert!(energy_to_tof(0.0, l).is_nan(), "zero energy must be NaN");
        assert!(
            energy_to_tof(f64::NAN, l).is_nan(),
            "NaN energy must stay NaN"
        );
        assert!(
            energy_to_tof(f64::INFINITY, l).is_nan(),
            "Inf energy must be NaN"
        );
        assert!(
            energy_to_tof(10.0, f64::NAN).is_nan(),
            "NaN flight path must be NaN"
        );
        assert!(
            energy_to_tof(10.0, f64::INFINITY).is_nan(),
            "Inf flight path must be NaN"
        );
        // Zero / negative flight path used to return 0 / a *negative* TOF —
        // a physically impossible time.
        assert!(
            energy_to_tof(10.0, 0.0).is_nan(),
            "zero flight path must be NaN"
        );
        assert!(
            energy_to_tof(10.0, -25.0).is_nan(),
            "negative flight path must be NaN"
        );
        // Valid input still produces a finite, positive TOF.
        assert!(energy_to_tof(10.0, l).is_finite());
        assert!(energy_to_tof(10.0, l) > 0.0);
    }
}
