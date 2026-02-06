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
/// - Division by zero in denominators (B_l singularities)
///
/// # Edge cases
///
/// For very small ρ (< 1e-10), uses Taylor series expansion:
/// - P_l → ρ^(2l+1) / (1·3·5·...·(2l+1))²
/// - S_l → -l for l > 0, S_0 = 0
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
            1 => Ok((rho.powi(3) / 3.0, -1.0)),
            2 => Ok((rho.powi(5) / 225.0, -2.0)),
            3 => Ok((rho.powi(7) / 11025.0, -3.0)),
            4 => Ok((rho.powi(9) / 893025.0, -4.0)),
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
/// Returns `(cos(2φ_l), sin(2φ_l))` where φ_l is the hard-sphere phase shift.
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
/// - `cos(rho) = 0` (critical singularity in tan(rho))
///
/// # Critical Singularity
///
/// ⚠️ When cos(ρ) = 0 (i.e., ρ = ±π/2, ±3π/2, ...), tan(ρ) is undefined.
/// This occurs at high energies and MUST be checked before computing phase shifts.
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

    // Check for critical singularity: cos(rho) = 0
    let cos_rho = rho.cos();
    if cos_rho.abs() < 1e-15 {
        return Err(PhysicsError::InvalidParameter(format!(
            "cos(rho) = 0 singularity (tan undefined) at rho={rho}"
        )));
    }

    let sin_rho = rho.sin();
    let rho_sq = rho * rho;

    match l {
        0 => {
            // l = 0: φ_0 = ρ
            // cos(2φ_0) = cos(2ρ) = 2cos²(ρ) - 1
            // sin(2φ_0) = sin(2ρ) = 2sin(ρ)cos(ρ)
            let cos_2phi = 2.0 * cos_rho * cos_rho - 1.0;
            let sin_2phi = 2.0 * sin_rho * cos_rho;
            Ok((cos_2phi, sin_2phi))
        }
        1 => {
            // l = 1: φ_1 = ρ - arctan(ρ)
            // B_1 = 1 + ρ²
            // cos(φ_1) = (cos(ρ) + ρ·sin(ρ)) / B_1
            // sin(φ_1) = (sin(ρ) - ρ·cos(ρ)) / B_1
            let b1 = 1.0 + rho_sq;
            let cos_phi = (cos_rho + rho * sin_rho) / b1;
            let sin_phi = (sin_rho - rho * cos_rho) / b1;
            // Double-angle formulas
            let cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi;
            let sin_2phi = 2.0 * sin_phi * cos_phi;
            Ok((cos_2phi, sin_2phi))
        }
        2 => {
            // l = 2: φ_2 = ρ - arctan(3ρ/(3-ρ²))
            // B_2 = 9 + 3ρ² + ρ⁴
            // cos(φ_2) = (cos(ρ)(9-ρ²) + 3ρ·sin(ρ)) / B_2
            // sin(φ_2) = (sin(ρ)(9-ρ²) - 3ρ·cos(ρ)) / B_2
            let b2 = rho_sq * (rho_sq + 3.0) + 9.0;
            if b2.abs() < 1e-30 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "B_2 singularity at rho={rho}"
                )));
            }
            let nine_minus_rho_sq = 9.0 - rho_sq;
            let cos_phi = (cos_rho * nine_minus_rho_sq + 3.0 * rho * sin_rho) / b2;
            let sin_phi = (sin_rho * nine_minus_rho_sq - 3.0 * rho * cos_rho) / b2;
            let cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi;
            let sin_2phi = 2.0 * sin_phi * cos_phi;
            Ok((cos_2phi, sin_2phi))
        }
        3 => {
            // l = 3: φ_3 = ρ - arctan(ρ(15-ρ²)/(15-6ρ²))
            // B_3 = 225 + 45ρ² + 6ρ⁴ + ρ⁶
            // Numerator for cos: cos(ρ)(225-105ρ²+ρ⁴) + ρ·sin(ρ)(60-ρ²)
            // Numerator for sin: sin(ρ)(225-105ρ²+ρ⁴) - ρ·cos(ρ)(60-ρ²)
            let rho_4 = rho_sq * rho_sq;
            let b3 = rho_sq * (rho_sq * (rho_sq + 6.0) + 45.0) + 225.0;
            if b3.abs() < 1e-30 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "B_3 singularity at rho={rho}"
                )));
            }
            let term1 = 225.0 - 105.0 * rho_sq + rho_4;
            let term2 = 60.0 - rho_sq;
            let cos_phi = (cos_rho * term1 + rho * sin_rho * term2) / b3;
            let sin_phi = (sin_rho * term1 - rho * cos_rho * term2) / b3;
            let cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi;
            let sin_2phi = 2.0 * sin_phi * cos_phi;
            Ok((cos_2phi, sin_2phi))
        }
        4 => {
            // l = 4: φ_4 = ρ - arctan(ρ(105-10ρ²)/(105-45ρ²+ρ⁴))
            // B_4 = 11025 + 1575ρ² + 135ρ⁴ + 10ρ⁶ + ρ⁸
            // Numerator for cos: cos(ρ)(11025-9450ρ²+630ρ⁴-ρ⁶) + ρ·sin(ρ)(4725-315ρ²+ρ⁴)
            // Numerator for sin: sin(ρ)(11025-9450ρ²+630ρ⁴-ρ⁶) - ρ·cos(ρ)(4725-315ρ²+ρ⁴)
            let rho_4 = rho_sq * rho_sq;
            let rho_6 = rho_4 * rho_sq;
            let b4 = rho_sq * (rho_sq * (rho_sq * (rho_sq + 10.0) + 135.0) + 1575.0) + 11025.0;
            if b4.abs() < 1e-30 {
                return Err(PhysicsError::InvalidParameter(format!(
                    "B_4 singularity at rho={rho}"
                )));
            }
            let term1 = 11025.0 - 9450.0 * rho_sq + 630.0 * rho_4 - rho_6;
            let term2 = 4725.0 - 315.0 * rho_sq + rho_4;
            let cos_phi = (cos_rho * term1 + rho * sin_rho * term2) / b4;
            let sin_phi = (sin_rho * term1 - rho * cos_rho * term2) / b4;
            let cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi;
            let sin_2phi = 2.0 * sin_phi * cos_phi;
            Ok((cos_2phi, sin_2phi))
        }
        _ => Err(PhysicsError::InvalidParameter(format!(
            "l > 4 not supported, got {l}"
        ))),
    }
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
        assert!(p.abs() < 1e-30); // ρ³/3 ≈ 0
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
    fn test_hard_sphere_phase_singularity() {
        // At ρ = π/2, cos(ρ) = 0, should return error
        let result = hard_sphere_phase(0, std::f64::consts::FRAC_PI_2);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_l() {
        // l > 4 should fail
        let result = penetration_shift_factors(5, 1.0);
        assert!(result.is_err());
    }
}
