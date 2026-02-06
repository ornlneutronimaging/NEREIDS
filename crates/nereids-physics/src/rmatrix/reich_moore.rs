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
/// 6. Extract S-matrix element `U_11`
/// 7. Compute cross sections from `U_11`
/// 8. Apply statistical weight `G_J` = (2J+1)/(2I+1)
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
    const TWOMHB: f64 = 2.196_807_132_5e-4; // 1/(fm*sqrt(eV)); SAMMY mmas7.f90 (Kvendf=2)
                                            // SAMMY constants use neutron mass in amu (Aneutr), so the reduced-mass
                                            // factor is A / (A + m_n) with m_n ~= 1.008664904.
    const ANEUTR_AMU: f64 = 1.008_664_904;
    let a_ratio = config.awr / (config.awr + ANEUTR_AMU);
    let k_inv_fm = TWOMHB * a_ratio * energy.sqrt();

    // Get first channel (entrance channel for neutrons)
    if spin_group.channels.is_empty() {
        return Err(PhysicsError::InvalidParameter(
            "spin group must have at least one channel".into(),
        ));
    }
    let entrance_channel = &spin_group.channels[0];
    // Channel radii in our model are already in SAMMY working units (fm).
    let ap_penetration = entrance_channel.radius;
    let ap_phase = entrance_channel.effective_radius;

    // Step 2: Compute penetration factors at energy E and at resonance energy E_r
    let rho_e = k_inv_fm * ap_penetration; // dimensionless ρ = ka (penetration radius)
    let rho_phase_e = k_inv_fm * ap_phase; // phase radius for hard-sphere phase
    let (p_e, _s_e) = penetration_shift_factors(entrance_channel.l, rho_e)?;
    let (cos_2phi, sin_2phi) = hard_sphere_phase(entrance_channel.l, rho_phase_e)?;
    let apply_potential_correction = config.include_potential
        && sammy_potential_correction_applies(
            spin_group.j,
            config.target_spin,
            entrance_channel.l,
        );

    // Special case: if there are no resonances, return potential-only scattering when requested.
    if resonances.is_empty() {
        if !config.include_potential {
            return Ok(CrossSections::default());
        }

        let g_j = (2.0 * spin_group.j + 1.0) / (2.0 * (2.0 * config.target_spin + 1.0));
        let k_sq = k_inv_fm * k_inv_fm;
        let norm = std::f64::consts::PI / k_sq * g_j * 0.01; // π·g_J / K² in barns
        let sigma_el_baseline = norm * 2.0 * (1.0 - cos_2phi);
        let sigma_el = if apply_potential_correction {
            sigma_el_baseline + norm * 2.0 * (1.0 - cos_2phi)
        } else {
            sigma_el_baseline
        };
        return Ok(CrossSections {
            elastic: sigma_el,
            capture: 0.0,
            fission: 0.0,
            total: sigma_el,
        });
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
        // SAMMY dopush1.f90:
        //   Den = Diff*Diff + 0.25*Gg*Gg
        //   Gg4 = 0.25*Gg / Den
        let denominator = delta * delta + 0.25 * gamma_g * gamma_g;

        if denominator < 1e-100 {
            return Err(PhysicsError::InvalidParameter(format!(
                "resonance denominator too small at E={energy}, E_λ={e_lambda}"
            )));
        }

        let r_factor = 0.25 * gamma_g / denominator;
        let s_factor = delta / (2.0 * denominator);

        // Penetration correction for neutron width
        // At resonance energy E_r
        let k_r_inv_fm = TWOMHB * a_ratio * e_lambda.abs().sqrt();
        let rho_r = k_r_inv_fm * ap_penetration;
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

    let (ri_mat, si_mat) = if has_fission {
        // Full 3×3 inversion using SAMMY Frobns/Thrinv path.
        sammy_frobns_invert_3x3(r_mat, s_mat)?
    } else {
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
    let sigma_el_corrected = if apply_potential_correction {
        sigma_el + norm * 2.0 * (1.0 - cos_2phi)
    } else {
        sigma_el
    };
    // Total: σ_tot = σ_el + σ_c + σ_f (using returned elastic channel).
    let sigma_tot = sigma_el_corrected + sigma_c + sigma_f;

    Ok(CrossSections {
        elastic: sigma_el_corrected,
        capture: sigma_c,
        fission: sigma_f,
        total: sigma_tot,
    })
}

/// Whether SAMMY adds the hard-sphere potential correction for this spin group.
///
/// Mirrors `dopush1.f90` logic:
/// - `Ajmin = abs(abs(I - l) - 1/2)`
/// - `Ajmax = I + l + 1/2`
/// - `Numj = (Ajmax - Ajmin) + 1.001`
/// - `Jjl = 0` only in the special non-s-wave branch, otherwise `1`
/// - correction applies when `JJ > Jjl && JJ < Numj`
fn sammy_potential_correction_applies(spin_j: f64, target_spin: f64, l: u32) -> bool {
    let fl = l as f64;
    let ajmin = ((target_spin - fl).abs() - 0.5).abs();
    let ajmax = target_spin + fl + 0.5;
    let numj = ((ajmax - ajmin) + 1.001).floor() as i32;
    if numj <= 0 {
        return false;
    }

    let jjl = if l != 0 && (fl > target_spin - 0.5 && fl <= target_spin) {
        0
    } else {
        1
    };

    // Map the actual J to SAMMY's JJ index (1-based).
    let tol = 1e-8;
    let mut jj_index = None;
    for jj in 1..=numj {
        let ajc = ajmin + (jj - 1) as f64;
        if (spin_j - ajc).abs() <= tol {
            jj_index = Some(jj);
            break;
        }
    }

    let Some(jj) = jj_index else {
        return false;
    };

    jj > jjl && jj < numj
}

/// Complex 3x3 inversion that mirrors SAMMY `Frobns` + `Thrinv` in `dopush1.f90`.
fn sammy_frobns_invert_3x3(
    r: [[f64; 3]; 3],
    s: [[f64; 3]; 3],
) -> Result<([[f64; 3]; 3], [[f64; 3]; 3]), PhysicsError> {
    let mut a = r;
    let mut c = a;
    let b = s;
    let mut d;

    let ind = sammy_thrinv_3x3(&mut a)?;
    if ind == 1 {
        Err(PhysicsError::InvalidParameter(
            "singular matrix in SAMMY Thrinv inversion".into(),
        ))
    } else {
        let q = matmul_3x3(a, b);
        d = matmul_3x3(b, q);
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] += d[i][j];
            }
        }
        let ind = sammy_thrinv_3x3(&mut c)?;
        if ind == 1 {
            return Err(PhysicsError::InvalidParameter(
                "singular matrix in SAMMY Thrinv second inversion".into(),
            ));
        }
        d = matmul_3x3(q, c);
        for row in &mut d {
            for x in row {
                *x = -*x;
            }
        }
        Ok((c, d))
    }
}

/// Matrix multiply C = A * B for 3x3 matrices.
fn matmul_3x3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// Real symmetric inversion used by SAMMY `Thrinv`.
///
/// Returns `0` on success and `1` on singularity to mirror SAMMY behavior.
fn sammy_thrinv_3x3(d: &mut [[f64; 3]; 3]) -> Result<i32, PhysicsError> {
    for j in 0..3 {
        for i in 0..=j {
            d[i][j] = -d[i][j];
            d[j][i] = d[i][j];
        }
        d[j][j] += 1.0;
    }

    for lr in 0..3 {
        let fooey = 1.0 - d[lr][lr];
        if fooey == 0.0 {
            return Ok(1);
        }

        d[lr][lr] = 1.0 / fooey;
        let mut s = [0.0; 3];
        for j in 0..3 {
            s[j] = d[lr][j];
            if j != lr {
                d[j][lr] *= d[lr][lr];
                d[lr][j] = d[j][lr];
            }
        }

        for j in 0..3 {
            if j == lr {
                continue;
            }
            for i in 0..=j {
                if i == lr {
                    continue;
                }
                d[i][j] += d[i][lr] * s[j];
                d[j][i] = d[i][j];
            }
        }
    }

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_core::nuclear::{Channel, Parameter, Resonance};

    #[test]
    fn test_cross_sections_zero_resonances() {
        // With no resonances and no potential term, all channels are zero.
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

        let result = reich_moore_cross_sections(1.0e6, &[], &spin_group, &config).unwrap();
        assert_eq!(result.elastic, 0.0);
        assert_eq!(result.capture, 0.0);
        assert_eq!(result.fission, 0.0);
        assert_eq!(result.total, 0.0);
    }

    #[test]
    fn test_cross_sections_zero_resonances_with_potential() {
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
            include_potential: true,
        };

        let result = reich_moore_cross_sections(1.0e6, &[], &spin_group, &config).unwrap();
        // Boundary groups still include the hard-sphere baseline term.
        assert!(result.elastic > 0.0);
        assert_eq!(result.capture, 0.0);
        assert_eq!(result.fission, 0.0);
        assert!((result.total - result.elastic).abs() < 1e-14);
    }

    #[test]
    fn test_cross_sections_zero_resonances_with_potential_interior_j() {
        let spin_group = SpinGroup {
            // For I=1 and l=1, J={0.5,1.5,2.5}; J=1.5 is interior and gets correction.
            j: 1.5,
            channels: vec![Channel {
                l: 1,
                channel_spin: 0.5,
                radius: 2.908,
                effective_radius: 2.908,
            }],
            resonances: vec![],
        };

        let config = RMatrixConfig {
            target_spin: 1.0,
            awr: 10.0,
            include_potential: true,
        };

        let result = reich_moore_cross_sections(1.0e6, &[], &spin_group, &config).unwrap();
        assert!(result.elastic > 0.0);
        assert_eq!(result.capture, 0.0);
        assert_eq!(result.fission, 0.0);
        assert!(result.total > 0.0);
    }

    #[test]
    fn test_cross_sections_use_effective_radius_for_phase() {
        let spin_group_a = SpinGroup {
            j: 1.5,
            channels: vec![Channel {
                l: 1,
                channel_spin: 0.5,
                radius: 2.0,
                effective_radius: 2.0,
            }],
            resonances: vec![],
        };
        let mut spin_group_b = spin_group_a.clone();
        spin_group_b.channels[0].effective_radius = 4.0;

        let config = RMatrixConfig {
            target_spin: 1.0,
            awr: 10.0,
            include_potential: true,
        };

        let a = reich_moore_cross_sections(1.0e6, &[], &spin_group_a, &config).unwrap();
        let b = reich_moore_cross_sections(1.0e6, &[], &spin_group_b, &config).unwrap();

        assert!((a.elastic - b.elastic).abs() > 1e-12);
    }

    #[test]
    fn test_sammy_potential_correction_gating() {
        // s-wave (l=0), I=0 => J=0.5 is boundary-only; no correction.
        assert!(!sammy_potential_correction_applies(0.5, 0.0, 0));
        // p-wave (l=1), I=1 => J=1.5 interior; correction applies.
        assert!(sammy_potential_correction_applies(1.5, 1.0, 1));
        // p-wave (l=1), I=1 => J=2.5 max boundary; no correction.
        assert!(!sammy_potential_correction_applies(2.5, 1.0, 1));
    }

    #[test]
    fn test_negative_resonance_energy_is_supported() {
        let spin_group = SpinGroup {
            j: 0.5,
            channels: vec![Channel {
                l: 0,
                channel_spin: 0.5,
                radius: 2.908,
                effective_radius: 2.908,
            }],
            resonances: vec![Resonance {
                energy: Parameter::fixed(-1.0),
                gamma_n: Parameter::fixed(1e-3),
                gamma_g: Parameter::fixed(1e-3),
                fission: None,
            }],
        };

        let config = RMatrixConfig {
            target_spin: 0.0,
            awr: 10.0,
            include_potential: false,
        };

        let result = reich_moore_cross_sections(1.0, &spin_group.resonances, &spin_group, &config);
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
    fn test_sammy_thrinv_singularity_flag() {
        // Zero matrix hits Fooey = 0 in SAMMY Thrinv and reports singularity.
        let mut zero = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let ind = sammy_thrinv_3x3(&mut zero).unwrap();
        assert_eq!(ind, 1);
    }

    #[test]
    fn test_sammy_thrinv_non_singular() {
        let mut m = [[1.2, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 1.4]];
        let ind = sammy_thrinv_3x3(&mut m).unwrap();
        assert_eq!(ind, 0);
        for row in m {
            for v in row {
                assert!(v.is_finite());
            }
        }
    }
}
