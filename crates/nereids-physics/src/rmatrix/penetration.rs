//! Shift and penetration factor calculations.
//!
//! Teacher references:
//! - `sammy/src/xxx/mxxxb.f90` (penetration factors)
//! - `sammy/src/xxx/mxxxa.f90` (shift factors)

use nereids_core::error::PhysicsError;

/// Compute penetration and shift factors for a given orbital angular momentum.
///
/// # Arguments
///
/// * `l` - Orbital angular momentum quantum number (0-4 supported: s, p, d, f, g)
/// * `rho` - Dimensionless wave number ρ = ka, where k is wave number and a is channel radius
///
/// # Returns
///
/// `(P_l, S_l)` where:
/// - `P_l` is the penetration factor
/// - `S_l` is the shift factor
///
/// # Errors
///
/// Returns `PhysicsError::InvalidParameter` if:
/// - `l > 4` (only s, p, d, f, g waves supported)
/// - `rho` is not finite
/// - Division by zero in denominators (`B_l` singularities)
///
/// # Edge cases
///
/// For very small ρ (< 1e-10), uses Taylor series expansion:
/// - `P_0` → ρ
/// - `P_1` → ρ³
/// - `P_2` → ρ⁵ / 9
/// - `P_3` → ρ⁷ / 225
/// - `P_4` → ρ⁹ / 11025
/// - `S_l` → -l for l > 0, `S_0` = 0
///
/// # References
///
/// SAMMY `mxxxb.f90` lines 6-62, `mxxxa.f90` lines 6-66
pub fn penetration_shift_factors(l: u32, rho: f64) -> Result<(f64, f64), PhysicsError> {
    if !rho.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "rho must be finite, got {rho}"
        )));
    }

    // Handle small rho with Taylor series (avoid division by zero)
    if rho.abs() < 1e-10 {
        return match l {
            0 => Ok((rho, 0.0)),
            1 => Ok((rho.powi(3), -1.0)),
            2 => Ok((rho.powi(5) / 9.0, -2.0)),
            3 => Ok((rho.powi(7) / 225.0, -3.0)),
            4 => Ok((rho.powi(9) / 11025.0, -4.0)),
            _ => Err(PhysicsError::InvalidParameter(format!(
                "l > 4 not supported, got {l}"
            ))),
        };
    }

    let rho_sq = rho * rho;

    match l {
        0 => {
            // l = 0 (s-wave)
            // P_0 = ρ
            // S_0 = 0
            Ok((rho, 0.0))
        }
        1 => {
            // l = 1 (p-wave)
            // B_1 = 1 + ρ²
            // P_1 = ρ³ / B_1 (NOT B_1²!)
            // S_1 = -1 / B_1
            let b1 = 1.0 + rho_sq;
            let p1 = rho_sq * rho / b1;
            let s1 = -1.0 / b1;
            Ok((p1, s1))
        }
        2 => {
            // l = 2 (d-wave)
            // B_2 = 9 + 3ρ² + ρ⁴ (using Horner: ρ²(ρ² + 3) + 9)
            // P_2 = ρ⁵ / B_2 (NOT B_2²!)
            // S_2 = -(18 + 3ρ²) / B_2
            let b2 = rho_sq * (rho_sq + 3.0) + 9.0;
            if b2.abs() < 1e-30 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "B_2 singularity at rho={rho}"
                )));
            }
            let p2 = rho_sq * rho_sq * rho / b2;
            let s2 = -(18.0 + 3.0 * rho_sq) / b2;
            Ok((p2, s2))
        }
        3 => {
            // l = 3 (f-wave)
            // B_3 = 225 + 45ρ² + 6ρ⁴ + ρ⁶ (using Horner: ρ²(ρ²(ρ² + 6) + 45) + 225)
            // P_3 = ρ⁷ / B_3 (NOT B_3²!)
            // S_3 = -(675 + 90ρ² + 6ρ⁴) / B_3
            let b3 = rho_sq * (rho_sq * (rho_sq + 6.0) + 45.0) + 225.0;
            if b3.abs() < 1e-30 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "B_3 singularity at rho={rho}"
                )));
            }
            let p3 = rho_sq * rho_sq * rho_sq * rho / b3;
            let s3 = -(675.0 + 90.0 * rho_sq + 6.0 * rho_sq * rho_sq) / b3;
            Ok((p3, s3))
        }
        4 => {
            // l = 4 (g-wave)
            // B_4 = 11025 + 1575ρ² + 135ρ⁴ + 10ρ⁶ + ρ⁸
            //     = ρ²(ρ²(ρ²(ρ² + 10) + 135) + 1575) + 11025 (Horner)
            // P_4 = ρ⁹ / B_4 (NOT B_4²!)
            // S_4 = -(44100 + 4725ρ² + 270ρ⁴ + 10ρ⁶) / B_4
            let rho_4 = rho_sq * rho_sq;
            let b4 = rho_sq * (rho_sq * (rho_sq * (rho_sq + 10.0) + 135.0) + 1575.0) + 11025.0;
            if b4.abs() < 1e-30 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "B_4 singularity at rho={rho}"
                )));
            }
            let p4 = rho_4 * rho_4 * rho / b4;
            let s4 = -(44100.0 + 4725.0 * rho_sq + 270.0 * rho_4 + 10.0 * rho_4 * rho_sq) / b4;
            Ok((p4, s4))
        }
        _ => Err(PhysicsError::InvalidParameter(format!(
            "l > 4 not supported, got {l}"
        ))),
    }
}

/// Compute hard-sphere phase shift for a given orbital angular momentum.
///
/// Returns `(cos(2φ_l), sin(2φ_l))` where `φ_l` is the hard-sphere phase shift.
///
/// # Arguments
///
/// * `l` - Orbital angular momentum quantum number (0-4 supported)
/// * `rho` - Dimensionless wave number ρ = ka
///
/// # Returns
///
/// `(cos(2φ_l), sin(2φ_l))` using double-angle formulas
///
/// # Errors
///
/// Returns `PhysicsError::InvalidParameter` if:
/// - `l > 4`
/// - `rho` is not finite
///
/// # References
///
/// SAMMY `mxxxa.f90` lines 6-66
pub fn hard_sphere_phase(l: u32, rho: f64) -> Result<(f64, f64), PhysicsError> {
    if !rho.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "rho must be finite, got {rho}"
        )));
    }

    let rho_sq = rho * rho;

    // SAMMY Facphi_NML defines φ_l = ρ - atan(B_l), then returns cos(2φ_l), sin(2φ_l).
    let b = match l {
        0 => 0.0,
        1 => rho,
        2 => rho * (3.0 / (3.0 - rho_sq)),
        3 => {
            let c = 15.0 - 6.0 * rho_sq;
            rho * ((15.0 - rho_sq) / c)
        }
        4 => {
            let rho_4 = rho_sq * rho_sq;
            let c = 105.0 - 45.0 * rho_sq + rho_4;
            rho * ((105.0 - 10.0 * rho_sq) / c)
        }
        _ => Err(PhysicsError::InvalidParameter(format!(
            "l > 4 not supported, got {l}"
        )))?,
    };

    let phi = rho - b.atan();
    Ok(((2.0 * phi).cos(), (2.0 * phi).sin()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penetration_shift_s_wave() {
        // l = 0: P_0 = ρ, S_0 = 0
        let (p, s) = penetration_shift_factors(0, 2.0).unwrap();
        assert!((p - 2.0).abs() < 1e-15);
        assert!(s.abs() < 1e-15);
    }

    #[test]
    fn test_penetration_shift_small_rho() {
        // For small ρ, should use Taylor series
        let (p, s) = penetration_shift_factors(1, 1e-12).unwrap();
        assert!((p - 1e-36).abs() < 1e-45); // P1 ~ ρ^3
        assert!((s + 1.0).abs() < 1e-15); // S_1 → -1
    }

    #[test]
    fn test_penetration_shift_p_wave() {
        // l = 1, ρ = 1: B_1 = 2, P_1 = ρ³/B_1 = 1/2, S_1 = -1/B_1 = -1/2
        let (p, s) = penetration_shift_factors(1, 1.0).unwrap();
        assert!((p - 0.5).abs() < 1e-15);
        assert!((s + 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_hard_sphere_phase_s_wave() {
        // l = 0: φ_0 = ρ, so cos(2φ) = cos(2ρ), sin(2φ) = sin(2ρ)
        let (cos_2phi, sin_2phi) = hard_sphere_phase(0, 1.0).unwrap();
        let expected_cos = (2.0_f64).cos();
        let expected_sin = (2.0_f64).sin();
        assert!((cos_2phi - expected_cos).abs() < 1e-15);
        assert!((sin_2phi - expected_sin).abs() < 1e-15);
    }

    #[test]
    fn test_hard_sphere_phase_p_wave_reference() {
        // l = 1: φ_1 = ρ - atan(ρ)
        let rho = 1.0_f64;
        let phi = rho - rho.atan();
        let (cos_2phi, sin_2phi) = hard_sphere_phase(1, rho).unwrap();
        assert!((cos_2phi - (2.0 * phi).cos()).abs() < 1e-15);
        assert!((sin_2phi - (2.0 * phi).sin()).abs() < 1e-15);
    }

    #[test]
    fn test_hard_sphere_phase_at_cos_zero_is_finite() {
        // ρ = π/2 is a valid point; phase factors remain finite.
        let (cos_2phi, sin_2phi) = hard_sphere_phase(0, std::f64::consts::FRAC_PI_2).unwrap();
        assert!((cos_2phi + 1.0).abs() < 1e-15);
        assert!(sin_2phi.abs() < 1e-15);
    }

    #[test]
    fn test_invalid_l() {
        // l > 4 should fail
        let result = penetration_shift_factors(5, 1.0);
        assert!(result.is_err());
    }
}
