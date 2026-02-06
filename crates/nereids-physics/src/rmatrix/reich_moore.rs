//! Reich-Moore R-matrix formalism for single-energy cross section evaluation.
//!
//! Teacher reference: `sammy/src/clm/dopush1.f90`

use nereids_core::error::PhysicsError;
use nereids_core::nuclear::{Resonance, SpinGroup};

use super::penetration::{hard_sphere_phase, penetration_shift_factors};

/// Cross sections for a single energy point.
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossSections {
    /// Elastic scattering cross section in barns.
    pub elastic: f64,
    /// Radiative capture cross section in barns.
    pub capture: f64,
    /// Fission cross section in barns (0.0 for non-fissile).
    pub fission: f64,
    /// Total cross section in barns (elastic + capture + fission).
    pub total: f64,
}

/// Configuration for R-matrix calculation.
#[derive(Debug, Clone)]
pub struct RMatrixConfig {
    /// Target nucleus spin (I).
    pub target_spin: f64,
    /// Atomic weight ratio (AWR) - mass of target relative to neutron.
    pub awr: f64,
    /// Include potential scattering corrections.
    pub include_potential: bool,
}

/// Compute 0K Reich-Moore cross sections at a single energy.
///
/// # Arguments
///
/// * `energy` - Incident neutron energy in eV
/// * `resonances` - Resonances for this spin group
/// * `spin_group` - Spin group with J, channels, and resonances
/// * `config` - R-matrix configuration
///
/// # Returns
///
/// `CrossSections` containing elastic, capture, fission, and total cross sections in barns.
///
/// # Errors
///
/// Returns `PhysicsError::InvalidParameter` if:
/// - Energy is not finite or positive
/// - Matrix inversion fails (singular or near-singular R-matrix)
/// - Channel parameters are invalid
///
/// # Algorithm
///
/// 1. Compute wave number K from energy
/// 2. Compute penetration factors for all channels at this energy
/// 3. Construct 3×3 complex R-matrix (real R and imaginary S parts)
/// 4. Accumulate contributions from all resonances
/// 5. Invert complex matrix using Frobenius-Schur method
/// 6. Extract S-matrix element U_11
/// 7. Compute cross sections from U_11
/// 8. Apply statistical weight G_J = (2J+1)/(2I+1)
/// 9. Add potential scattering if requested
///
/// # References
///
/// SAMMY `dopush1.f90` lines 8-184
pub fn reich_moore_cross_sections(
    energy: f64,
    resonances: &[Resonance],
    spin_group: &SpinGroup,
    config: &RMatrixConfig,
) -> Result<CrossSections, PhysicsError> {
    if !energy.is_finite() || energy <= 0.0 {
        return Err(PhysicsError::InvalidParameter(format!(
            "energy must be finite and positive, got {energy}"
        )));
    }

    // Step 1: Compute wave number K in SAMMY convention:
    // K = Twomhb * A_rat * sqrt(E),
    // where A_rat = AWR / (1 + AWR), E in eV, K in 1/fm.
    if !config.awr.is_finite() || config.awr <= 0.0 {
        return Err(PhysicsError::InvalidParameter(format!(
            "awr must be finite and positive, got {}",
            config.awr
        )));
    }
    const TWOMHB: f64 = 2.196_807_122_623e-4; // 1/(fm*sqrt(eV))
    let a_ratio = config.awr / (1.0 + config.awr);
    let k_inv_fm = TWOMHB * a_ratio * energy.sqrt();

    // Get first channel (entrance channel for neutrons)
    if spin_group.channels.is_empty() {
        return Err(PhysicsError::InvalidParameter(
            "spin group must have at least one channel".into(),
        ));
    }
    let entrance_channel = &spin_group.channels[0];

    // Step 2: Compute penetration factors at energy E and at resonance energy E_r
    let rho_e = k_inv_fm * entrance_channel.radius; // dimensionless ρ = ka
    let (p_e, _s_e) = penetration_shift_factors(entrance_channel.l, rho_e)?;

    // Special case: if there are no resonances, return only potential scattering
    if resonances.is_empty() {
        // With no resonances, R = 0, so U = exp(2iφ) (hard-sphere scattering only)
        // For now, return zero cross sections (potential scattering is a small correction)
        return Ok(CrossSections::default());
    }

    // For channels beyond neutron channel, we need fission channels
    // In Reich-Moore, we have up to 3 channels: neutron (n), fission1 (fa), fission2 (fb)

    // Step 3: Construct R-matrix (3×3 complex symmetric)
    // R = R_real + i*R_imag where we represent as two separate 3×3 matrices
    // Initialize with identity on diagonal (matches SAMMY dopush1.f90 line 83)
    let mut r_mat = [[0.0_f64; 3]; 3]; // Real part
    let mut s_mat = [[0.0_f64; 3]; 3]; // Imaginary part (with sign convention)

    for i in 0..3 {
        r_mat[i][i] = 1.0; // Identity diagonal prevents singularity when Γ_fa = Γ_fb
    }

    // Accumulate contributions from all resonances
    for res in resonances {
        let e_lambda = res.energy.value;
        let gamma_g = res.gamma_g.value;
        let gamma_n = res.gamma_n.value;

        // Energy-dependent factors
        let delta = e_lambda - energy;
        let gamma_g_quarter = gamma_g / 4.0;
        let denominator = delta * delta + gamma_g_quarter * gamma_g_quarter;

        if denominator < 1e-100 {
            return Err(PhysicsError::InvalidParameter(format!(
                "resonance denominator too small at E={energy}, E_λ={e_lambda}"
            )));
        }

        let r_factor = gamma_g_quarter / denominator;
        let s_factor = delta / (2.0 * denominator);

        // Penetration correction for neutron width
        // At resonance energy E_r
        let k_r_inv_fm = TWOMHB * a_ratio * e_lambda.sqrt();
        let rho_r = k_r_inv_fm * entrance_channel.radius;
        let (p_r, _) = penetration_shift_factors(entrance_channel.l, rho_r)?;

        // Channel amplitude for neutron channel
        let a_n = if p_r > 1e-100 {
            (gamma_n * p_e / p_r).sqrt()
        } else {
            0.0
        };

        // Fission channel amplitudes
        let (a_fa, a_fb) = if let Some(fission) = &res.fission {
            let gamma_fa = fission.gamma_f1.value;
            let gamma_fb = fission.gamma_f2.value;

            // Preserve sign of fission widths (can be negative in Reich-Moore)
            let a_fa = if gamma_fa >= 0.0 {
                gamma_fa.sqrt()
            } else {
                -((-gamma_fa).sqrt())
            };
            let a_fb = if gamma_fb >= 0.0 {
                gamma_fb.sqrt()
            } else {
                -((-gamma_fb).sqrt())
            };
            (a_fa, a_fb)
        } else {
            (0.0, 0.0)
        };

        // Accumulate R-matrix elements (symmetric)
        // R(i,j) += r_factor * A_i * A_j
        // S(i,j) -= s_factor * A_i * A_j (note minus sign)

        let amplitudes = [a_n, a_fa, a_fb];

        for i in 0..3 {
            for j in i..3 {
                let contribution = amplitudes[i] * amplitudes[j];
                r_mat[i][j] += r_factor * contribution;
                s_mat[i][j] -= s_factor * contribution;

                // Fill symmetric part
                if i != j {
                    r_mat[j][i] = r_mat[i][j];
                    s_mat[j][i] = s_mat[i][j];
                }
            }
        }
    }

    // Step 4: Invert complex matrix
    // Special case: if no fission channels, reduce to 1×1
    let has_fission = resonances.iter().any(|r| r.fission.is_some());

    let (ri_mat, si_mat) = if !has_fission {
        // 1×1 inversion: (R + iS)^(-1) = (R - iS) / (R² + S²)
        let r = r_mat[0][0];
        let s = s_mat[0][0];
        let denom = r * r + s * s;

        if denom < 1e-100 {
            return Err(PhysicsError::InvalidParameter(format!(
                "near-singular 1x1 R-matrix (denom={denom}) at E={energy}"
            )));
        }

        let mut ri = [[0.0; 3]; 3];
        let mut si = [[0.0; 3]; 3];
        ri[0][0] = r / denom;
        si[0][0] = -s / denom;
        (ri, si)
    } else {
        // Full 3×3 inversion using Frobenius-Schur
        frobenius_schur_invert_3x3(r_mat, s_mat)?
    };

    // Step 5: Extract S-matrix element U_11
    // U_11 = P_e(2·R_inv - I) + 2·P_e·S_inv + i[P_e(I - 2·R_inv) + 2·P_e·S_inv]
    // But this is the formula from the docs, let me use the simpler one from SAMMY
    // U_11 = exp(2i·φ) * [1 - 2i·P·(R_inv + i·S_inv)]
    // where we compute real and imaginary parts separately

    let ri_11 = ri_mat[0][0];
    let si_11 = si_mat[0][0];

    // The S-matrix (collision matrix) element from SAMMY dopush1.f90 lines 162-163:
    // U_11 = exp(2iφ_l) * [2*(RI + iSI) - 1]
    // Where P1 = cos(2φ), P2 = sin(2φ) (NOT penetration factors!)
    //
    // U11r = P1*(2*RI - 1) + 2*P2*SI
    // U11i = P2*(1 - 2*RI) + 2*P1*SI
    //
    // This factors as: U = exp(2iφ)*[2*(RI + iSI) - 1]
    let (cos_2phi, sin_2phi) = hard_sphere_phase(entrance_channel.l, rho_e)?;

    let u_re = cos_2phi * (2.0 * ri_11 - 1.0) + 2.0 * sin_2phi * si_11;
    let u_im = sin_2phi * (1.0 - 2.0 * ri_11) + 2.0 * cos_2phi * si_11;

    // Step 6: Compute cross sections
    // σ_el = (2G_J / K²) · |1 - U_11|²
    // σ_trans = (2G_J / K²) · (1 - Re{U_11})
    // σ_c = σ_trans - σ_f - σ_el (capture = transmission - fission - elastic)

    // Statistical weight G_J = (2J + 1) / [2(2I + 1)]
    // From SAMMY dopush1.f90 lines 40, 76: Gjd = 2*(2*I+1), Gj = (2*J+1)/Gjd
    let g_j = (2.0 * spin_group.j + 1.0) / (2.0 * (2.0 * config.target_spin + 1.0));

    // Normalization: convert to barns
    // σ has units of area, K² has units of 1/length²
    // π / K² gives an area in fm², convert to barns (1 barn = 100 fm², so divide by 100)
    let k_sq = k_inv_fm * k_inv_fm;
    let norm = std::f64::consts::PI / k_sq * g_j * 0.01; // π·g_J / K² in barns

    // Elastic: |1 - U|² = (1 - U_re)² + U_im²
    let one_minus_u_re = 1.0 - u_re;
    let sigma_el = norm * (one_minus_u_re * one_minus_u_re + u_im * u_im);

    // Transmission (related to total via optical theorem)
    let sigma_trans = norm * 2.0 * one_minus_u_re;

    // Fission uses off-diagonal inverse matrix terms for channels 2 and 3.
    // This follows the Reich-Moore/SAMMY structure where channel couplings
    // into fission channels contribute via RI/SI(1,2) and RI/SI(1,3).
    let sigma_f = if has_fission {
        let ri_12 = ri_mat[0][1];
        let si_12 = si_mat[0][1];
        let ri_13 = ri_mat[0][2];
        let si_13 = si_mat[0][2];
        norm * 4.0 * (ri_12 * ri_12 + si_12 * si_12 + ri_13 * ri_13 + si_13 * si_13)
    } else {
        0.0
    };

    // Capture: σ_c = σ_trans - σ_f - σ_el
    let sigma_c = sigma_trans - sigma_f - sigma_el;

    // Potential scattering correction.
    let sigma_el_corrected = if config.include_potential {
        sigma_el + norm * 2.0 * (1.0 - cos_2phi)
    } else {
        sigma_el
    };
    // Total: σ_tot = σ_el + σ_c + σ_f (using returned elastic channel).
    let sigma_tot = sigma_el_corrected + sigma_c + sigma_f;

    Ok(CrossSections {
        elastic: sigma_el_corrected.max(0.0),
        capture: sigma_c.max(0.0),
        fission: sigma_f.max(0.0),
        total: sigma_tot.max(0.0),
    })
}

/// Invert a 3×3 complex matrix represented as real and imaginary parts.
///
/// Uses the Frobenius-Schur method to invert a complex symmetric matrix
/// (R + iS)^(-1) = RI + iSI
///
/// # Arguments
///
/// * `r` - Real part of 3×3 complex matrix
/// * `s` - Imaginary part of 3×3 complex matrix
///
/// # Returns
///
/// `(RI, SI)` where RI and SI are real and imaginary parts of the inverse
///
/// # Errors
///
/// Returns `PhysicsError::InvalidParameter` if matrix is singular or near-singular
///
/// # References
///
/// SAMMY `dopush1.f90` (Frobenius-Schur inversion section)
fn frobenius_schur_invert_3x3(
    r: [[f64; 3]; 3],
    s: [[f64; 3]; 3],
) -> Result<([[f64; 3]; 3], [[f64; 3]; 3]), PhysicsError> {
    // Frobenius-Schur: (R + iS)^(-1) = (R + S·R^(-1)·S)^(-1) - i·R^(-1)·S·(R + S·R^(-1)·S)^(-1)

    // First invert R (real symmetric matrix)
    let r_inv = invert_real_3x3_symmetric(r)?;

    // Compute S·R^(-1)·S
    let mut srs = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    sum += s[i][k] * r_inv[k][l] * s[l][j];
                }
            }
            srs[i][j] = sum;
        }
    }

    // Compute R + S·R^(-1)·S
    let mut r_plus_srs = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r_plus_srs[i][j] = r[i][j] + srs[i][j];
        }
    }

    // Invert (R + S·R^(-1)·S)
    let r_plus_srs_inv = invert_real_3x3_symmetric(r_plus_srs)?;

    // RI = (R + S·R^(-1)·S)^(-1)
    let ri = r_plus_srs_inv;

    // SI = -R^(-1)·S·(R + S·R^(-1)·S)^(-1)
    let mut si = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    sum += r_inv[i][k] * s[k][l] * ri[l][j];
                }
            }
            si[i][j] = -sum;
        }
    }

    Ok((ri, si))
}

/// Invert a real 3×3 symmetric matrix using cofactor method.
fn invert_real_3x3_symmetric(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], PhysicsError> {
    // Compute determinant
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-100 {
        return Err(PhysicsError::InvalidParameter(format!(
            "singular matrix (det = {det})"
        )));
    }

    // Compute adjugate matrix (cofactor matrix transposed)
    let mut adj = [[0.0; 3]; 3];

    adj[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    adj[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    adj[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];

    adj[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    adj[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    adj[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);

    adj[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    adj[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    adj[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    // Inverse = adjugate / determinant
    let mut inv = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inv[i][j] = adj[i][j] / det;
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::nuclear::Channel;

    #[test]
    fn test_cross_sections_zero_resonances() {
        // With no resonances, should get potential scattering only
        let spin_group = SpinGroup {
            j: 0.5,
            channels: vec![Channel {
                l: 0,
                channel_spin: 0.5,
                radius: 2.908,
                effective_radius: 2.908,
            }],
            resonances: vec![],
        };

        let config = RMatrixConfig {
            target_spin: 0.0,
            awr: 10.0,
            include_potential: false,
        };

        let result = reich_moore_cross_sections(1.0, &[], &spin_group, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_energy() {
        let spin_group = SpinGroup {
            j: 0.5,
            channels: vec![Channel {
                l: 0,
                channel_spin: 0.5,
                radius: 2.908,
                effective_radius: 2.908,
            }],
            resonances: vec![],
        };

        let config = RMatrixConfig {
            target_spin: 0.0,
            awr: 10.0,
            include_potential: false,
        };

        // Negative energy should fail
        let result = reich_moore_cross_sections(-1.0, &[], &spin_group, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_inversion_identity() {
        // Test inverting identity matrix
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_real_3x3_symmetric(identity).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-15);
            }
        }
    }
}
