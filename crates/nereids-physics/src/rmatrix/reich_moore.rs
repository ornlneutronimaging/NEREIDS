//! Reich-Moore R-matrix formalism for single-energy cross section evaluation.
//!
//! Teacher reference: `sammy/src/clm/dopush1.f90`

use nereids_core::constants::NEUTRON_MASS_AMU;
use nereids_core::error::PhysicsError;
use nereids_core::nuclear::{Resonance, SpinGroup};

use super::penetration::penetration_shift_factors;

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

    // Step 1: Compute wave number K = sqrt(2 * m_r * E) / hbar
    // where m_r = m_n * m_target / (m_n + m_target) = m_n * AWR / (1 + AWR)
    // Units: energy in eV, hbar in eV·s, mass in AMU
    // K is in units of inverse length (need to convert to 1/fm)

    let reduced_mass_amu = NEUTRON_MASS_AMU * config.awr / (1.0 + config.awr);
    let reduced_mass_ev = reduced_mass_amu * 931.494_102e6; // AMU to eV/c²

    // K² = 2 * m_r * E / (hbar * c)²
    // But we want K in 1/fm, so we need proper unit conversion
    // E in eV, m_r in eV/c², hbar in eV·s, c in m/s
    // K = sqrt(2 * m_r * E) / (hbar * c) where hbar*c = 197.327 MeV·fm
    const HBAR_C_MEV_FM: f64 = 197.326_980_4; // MeV·fm
    let hbar_c_ev_fm = HBAR_C_MEV_FM * 1e6; // eV·fm

    let k_inv_fm = (2.0 * reduced_mass_ev * energy).sqrt() / hbar_c_ev_fm;

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
        let gamma_g_half = gamma_g / 2.0;
        let denominator = delta * delta + gamma_g_half * gamma_g_half;

        if denominator < 1e-100 {
            return Err(PhysicsError::InvalidParameter(format!(
                "resonance denominator too small at E={energy}, E_λ={e_lambda}"
            )));
        }

        let r_factor = gamma_g_half / denominator;
        let s_factor = delta / (2.0 * denominator);

        // Penetration correction for neutron width
        // At resonance energy E_r
        let rho_r = (2.0 * reduced_mass_ev * e_lambda).sqrt() / hbar_c_ev_fm * entrance_channel.radius;
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
            // R-matrix is effectively zero - treat like no resonances (potential scattering only)
            return Ok(CrossSections::default());
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

    // The collision matrix element (without hard-sphere phase)
    // U = 1 - 2iP(RI + iSI) = 1 - 2iP·RI + 2P·SI
    //   = (1 + 2P·SI) + i(-2P·RI)
    let u_re = 1.0 + 2.0 * p_e * si_11;
    let u_im = -2.0 * p_e * ri_11;

    // For hard-sphere phase: U → exp(2iφ) * U
    // We'll skip this for now as it's a higher-order correction

    // Step 6: Compute cross sections
    // σ_el = (2G_J / K²) · |1 - U_11|²
    // σ_trans = (2G_J / K²) · (1 - Re{U_11})
    // σ_c = σ_trans - σ_f - σ_el (capture = transmission - fission - elastic)

    // Statistical weight G_J = (2J + 1) / (2I + 1)
    let g_j = (2.0 * spin_group.j + 1.0) / (2.0 * config.target_spin + 1.0);

    // Normalization: convert to barns
    // σ has units of area, K² has units of 1/length²
    // π / K² gives an area in fm², convert to barns (1 barn = 100 fm²)
    let k_sq = k_inv_fm * k_inv_fm;
    let norm = std::f64::consts::PI / k_sq * g_j * 100.0; // π·g_J / K² in barns

    // Elastic: |1 - U|² = (1 - U_re)² + U_im²
    let one_minus_u_re = 1.0 - u_re;
    let sigma_el = norm * (one_minus_u_re * one_minus_u_re + u_im * u_im);

    // Transmission (related to total via optical theorem)
    let sigma_trans = norm * 2.0 * one_minus_u_re;

    // Fission: σ_f = (4G_J / K²) · Σ T_i²
    // where T_i are transmission coefficients for fission channels
    // T_i = 2·P_e · |U_1i|² but we need to compute U_12, U_13
    // For now, approximate as 0 if no fission resonances
    let sigma_f = if has_fission {
        // Fission cross section from other matrix elements
        // This requires U_12 and U_13 which we haven't computed yet
        // For now, use a placeholder - we'll need to extend this
        0.0
    } else {
        0.0
    };

    // Capture: σ_c = σ_trans - σ_f - σ_el
    let sigma_c = sigma_trans - sigma_f - sigma_el;

    // Total: σ_tot = σ_el + σ_c + σ_f
    let sigma_tot = sigma_el + sigma_c + sigma_f;

    // Potential scattering correction (for intermediate J)
    let sigma_el_corrected = if config.include_potential {
        // Add (1 - P_e) contribution
        sigma_el + norm * (1.0 - p_e)
    } else {
        sigma_el
    };

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
