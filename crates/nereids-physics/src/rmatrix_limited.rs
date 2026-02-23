//! R-Matrix Limited (LRF=7) cross-section calculation.
//!
//! Computes energy-dependent cross-sections (total, elastic, capture, fission)
//! from ENDF LRF=7 resonance parameters using the full multi-channel R-matrix
//! formalism.
//!
//! ## Relationship to Reich-Moore (LRF=3)
//!
//! Reich-Moore is a special case of R-matrix theory that *eliminates* the
//! capture channel via the Wigner-Eisenbud approximation, collapsing the
//! level matrix to a scalar per spin group. LRF=7 retains all channels
//! (elastic, capture, fission) explicitly, requiring an NCH×NCH complex
//! matrix inversion per spin group per energy point.
//!
//! ## Key Formulas
//!
//! (ENDF-6 Formats Manual Appendix D; SAMMY manual §3.1; SAMMY `rml/mrml07.f` Pgh + `mrml11.f` Setxqx)
//!
//! ```text
//! R-matrix:
//!   R_cc'(E) = Σ_n γ_nc · γ_nc' / (E_n - E)   [complex for KRM=3, NCH×NCH, symmetric]
//!   γ_nc = reduced width amplitude for resonance n in channel c (eV^{1/2})
//!
//! Level denominator per channel:
//!   L_c(E) = S_c(E) - B_c + i·P_c(E)
//!   P_c = penetrability,  S_c = shift factor,  B_c = boundary condition
//!
//! Reduced level matrix (SAMMY "Ymat"):
//!   Ỹ_cc'(E) = δ_cc' / L_c(E) - R_cc'(E)   [complex, NCH×NCH]
//!   → Ỹinv = Ỹ⁻¹  (invert Ỹ)
//!
//! Intermediate matrix (SAMMY "XXXX"):
//!   Ξ_cc'(E) = (√P_c / L_c) · (Ỹinv · R)_cc' · √P_c'
//!
//! Collision matrix (SAMMY manual eq. III.D.4):
//!   W_cc' = δ_cc' + 2i·Ξ_cc'
//!   U_cc' = Ω_c · W_cc' · Ω_c'    where Ω_c = exp(iφ_c)
//!   Unitarity: |U| ≤ 1 always; hard sphere (R=0) → U = exp(2iφ)·I  ✓
//!
//! Cross sections per spin group (J,π), summed over entrance neutron channels c0:
//!   σ_total   = Σ_{c0} 2·(π/k²)·g_J·(1 - Re(U_{c0,c0}))
//!   σ_elastic = Σ_{c0} (π/k²)·g_J·|1 - U_{c0,c0}|²
//!   σ_fission = Σ_{c0} Σ_{c'∈fission} (π/k²)·g_J·|U_{c0,c'}|²
//!   σ_capture = σ_total - σ_elastic - σ_fission
//! ```
//!
//! ## Photon Channels
//!
//! For channels involving zero-rest-mass particles (photons, MA=0), the
//! penetrability is set to 1.0 and phase shifts to 0.0, following the
//! standard R-matrix convention (ENDF-6 Formats Manual §2.2.1.6 Note 4).
//!
//! ## SAMMY Reference
//! - `rml/mrml01.f` — LRF=7 reader (Scan_File_2, particle pair loop)
//! - `rml/mrml09.f` — Level matrix inversion (Yinvrs, Xspfa, Xspsl)
//! - `rml/mrml11.f` — Cross-section calculation (Sectio, Setxqx)
//! - SAMMY manual §3.1 (multi-channel R-matrix)

use num_complex::Complex64;

use nereids_endf::resonance::{ParticlePair, RmlData, SpinGroup};

use crate::{channel, coulomb, penetrability};

/// Compute cross-section contributions from an LRF=7 energy range.
///
/// Returns `(total, elastic, capture, fission)` in barns.
///
/// Iterates over all spin groups (J,π), sums their contributions.
pub fn cross_sections_for_rml_range(rml: &RmlData, energy_ev: f64) -> (f64, f64, f64, f64) {
    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for sg in &rml.spin_groups {
        let (t, e, cap, fis) = spin_group_cross_sections(
            sg,
            &rml.particle_pairs,
            energy_ev,
            rml.awr,
            rml.target_spin,
            rml.krm,
        );
        total += t;
        elastic += e;
        capture += cap;
        fission += fis;
    }

    (total, elastic, capture, fission)
}

/// Cross-section contribution from a single spin group (J,π).
///
/// Returns (total, elastic, capture, fission) in barns.
fn spin_group_cross_sections(
    sg: &SpinGroup,
    particle_pairs: &[ParticlePair],
    energy_ev: f64,
    awr: f64,
    target_spin: f64,
    krm: u32,
) -> (f64, f64, f64, f64) {
    let nch = sg.channels.len();
    if nch == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // KRM guard: the parser rejects KRM values other than 2 and 3 at load time,
    // so reaching here with an unsupported KRM indicates a programming error.
    // Panic rather than silently returning zero physics, which would look valid
    // to callers.  Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f KRM field.
    assert!(
        krm == 2 || krm == 3,
        "spin_group_cross_sections called with unsupported KRM={krm}; \
         should have been rejected at parse time"
    );

    // P2b: NRS=0 is valid — the R-matrix is zero but the hard-sphere phase shift
    // still produces nonzero potential-scattering cross sections (σ_el = 4π|Ω-1|²/k²
    // per spin group).  Do NOT return early here; the resonance loop simply executes
    // zero iterations, leaving R = 0, which is exactly the hard-sphere limit.

    let g_j = channel::statistical_weight(sg.j, target_spin);
    let pok2 = channel::pi_over_k_squared_barns(energy_ev, awr);

    // ── Per-channel quantities ────────────────────────────────────────────────
    let mut p_c = vec![0.0f64; nch]; // penetrability P_c
    let mut s_c = vec![0.0f64; nch]; // shift factor S_c
    let mut phi_c = vec![0.0f64; nch]; // hard-sphere phase φ_c
    let mut is_entrance = vec![false; nch];
    let mut is_fission = vec![false; nch];
    let mut is_capture = vec![false; nch]; // photon/gamma channels (MT=102)
    let mut is_inelastic = vec![false; nch]; // massive non-elastic/non-fission/non-capture (MT=51+)
    let mut is_closed = vec![false; nch]; // channel below threshold (e_c ≤ 0)

    // Entrance-channel CM energy: E_cm = E_lab × AWR/(1+AWR).
    // Each exit channel adds its Q-value to get its own available energy.
    // Reference: SAMMY rml/mrml03.f Fxradi — channel thresholds via Q.
    let e_cm = channel::lab_to_cm_energy(energy_ev, awr);

    for (c, ch) in sg.channels.iter().enumerate() {
        // P3: particle_pair_idx must be a valid index. The old `.min(len-1)` clamped
        // silently, misclassifying any channel with an OOB index as the last pair.
        // An OOB value indicates corrupted ENDF data; let Rust's bounds check panic.
        let pp = &particle_pairs[ch.particle_pair_idx];
        is_entrance[c] = pp.mt == 2;
        is_fission[c] = pp.mt == 18;
        is_capture[c] = pp.mt == 102;
        // Inelastic neutron channels (MT=51+): massive particle, not elastic/fission/capture.
        // Their flux appears in σ_total (optical theorem) but must not be assigned to capture.
        // Reference: ENDF MT number conventions, §3.4; SAMMY rml/mrml11.f Sectio.
        is_inelastic[c] = pp.ma >= 0.5 && !is_entrance[c] && !is_fission[c] && !is_capture[c];

        if pp.ma < 0.5 {
            // Photon channel (MA = 0): P=1, S=0, φ=0.
            // Convention per ENDF-6 Formats Manual §2.2.1.6 Note 4.
            // SAMMY: rml/mrml07.f sets penetrability = 1 for massless particles.
            p_c[c] = 1.0;
            s_c[c] = 0.0;
            phi_c[c] = 0.0;
        } else {
            // Massive particle channel: channel-specific kinematics (P1).
            // E_c = E_cm + Q (CM kinetic energy in this exit channel).
            // Reference: SAMMY rml/mrml03.f Fxradi — Zke = Twomhb*sqrt(Redmas*Factor)
            let e_c = e_cm + pp.q;
            if e_c <= 0.0 {
                // Closed channel (below threshold): P_c = 0, φ_c = 0.
                // S_c depends on SHF:
                //   SHF=0: convention is S_c = B_c; L_c = 0 when B_c = 0 (common).
                //   SHF=1: S_c is the analytic shift factor at imaginary argument
                //     ρ = iκ, which is real and finite.  L_c = (S_c − B_c) is generally
                //     non-zero and its dispersive contribution must be preserved.
                // Reference: SAMMY rml/mrml07.f Pgh — PH = 1/(S−B+iP).
                p_c[c] = 0.0;
                phi_c[c] = 0.0;
                is_closed[c] = true;
                s_c[c] = if pp.shf == 0 {
                    ch.boundary
                } else {
                    let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
                    let kappa = channel::wave_number_from_cm(e_c.abs(), redmas);
                    penetrability::shift_factor_closed(ch.l, kappa * ch.effective_radius)
                };
            } else {
                // Channel wave number from reduced mass μ = MA·MB/(MA+MB).
                // For elastic (MA=1, MB=AWR): k_c = wave_number(E_lab, AWR) [identical].
                let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
                let k_c = channel::wave_number_from_cm(e_c, redmas);
                // APE (effective radius) for P_c, S_c; APT (true radius) for φ_c.
                // Reference: SAMMY rml/mrml07.f (Pgh, Sinsix, Pf subroutines)
                let rho_eff = k_c * ch.effective_radius;
                let rho_true = k_c * ch.true_radius;
                // ── Coulomb vs hard-sphere routing ───────────────────────────
                // SAMMY rml/mrml07.f Pgh — `if (Zeta(I).NE.Zero)` branch.
                // Both particles charged → Coulomb wave functions F_L / G_L.
                // One neutral (za=0 or zb=0) → hard-sphere Blatt-Weisskopf.
                if pp.za.abs() > 0.5 && pp.zb.abs() > 0.5 {
                    // Coulomb channel (e.g. n+α→p+X, (n,p), fission fragments).
                    let eta = coulomb::sommerfeld_eta(pp.za, pp.zb, pp.ma, pp.mb, e_c);
                    p_c[c] = coulomb::coulomb_penetrability(ch.l, eta, rho_eff);
                    // SHF flag applies to Coulomb shift as well.
                    s_c[c] = if pp.shf == 1 {
                        coulomb::coulomb_shift(ch.l, eta, rho_eff)
                    } else {
                        ch.boundary
                    };
                    let (sin_phi, cos_phi) = coulomb::coulomb_phase(ch.l, eta, rho_true);
                    phi_c[c] = sin_phi.atan2(cos_phi);
                } else {
                    // Hard-sphere (Blatt-Weisskopf) channel.
                    p_c[c] = penetrability::penetrability(ch.l, rho_eff);
                    // SHF=0: shift factor not calculated; S_c = B_c so (S_c - B_c) = 0
                    // in the level matrix diagonal.
                    // SHF=1: calculate S_c analytically (Blatt-Weisskopf).
                    // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml07.f Pgh (Ishift check)
                    s_c[c] = if pp.shf == 1 {
                        penetrability::shift_factor(ch.l, rho_eff)
                    } else {
                        ch.boundary
                    };
                    phi_c[c] = penetrability::phase_shift(ch.l, rho_true);
                }
            }
        }
    }

    // ── R-matrix (complex for KRM=3, real for KRM=2) ─────────────────────────
    // KRM=2 (standard R-matrix):
    //   R_cc'(E) = Σ_n γ_nc · γ_nc' / (E_n - E)   [real, reduced amplitude widths]
    //
    // KRM=3 (Reich-Moore approximation):
    //   R_cc'(E) = Σ_n γ_nc · γ_nc' / (Ẽ_n - E)   [complex, Ẽ_n = E_n - i·Γ_γn/2]
    //   where γ_nc = √(Γ_nc / (2·P_c(E_n))) (partial width → reduced amplitude).
    //   The imaginary shift makes capture implicit — |U| < 1, with missing flux
    //   going to capture.
    //
    // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml07.f Setr subroutine
    let mut r_cplx = vec![vec![Complex64::ZERO; nch]; nch];
    for res in &sg.resonances {
        let (gamma_vals, e_tilde) = if krm == 3 {
            // KRM=3: convert formal partial widths to reduced amplitudes.
            // γ_nc = √(|Γ_nc| / (2·P_c(E_n))).  Sign preserved from Γ_nc.
            // For closed channels or P=0 (e.g. bound states at E_n<0): use
            // γ_nc = √(|Γ_nc|) directly (SAMMY convention for bound states).
            // Complex energy Ẽ_n = E_n - i·Γ_γ/2.
            // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f
            let e_tilde = Complex64::new(res.energy, -res.gamma_gamma / 2.0);
            // P_c must be evaluated at the resonance energy E_n, not at the
            // incident energy E.  γ_nc = √(Γ_nc / (2·P_c(E_n))) is a property
            // of the resonance and must be energy-independent.  Using P_c(E)
            // would make γ_nc depend on the evaluation point, distorting the
            // resonance shape away from the peak.
            // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f (reads Γ_nc then
            // converts via P at resonance energy in Pgh subroutine).
            let gamma_vals: Vec<f64> = (0..nch)
                .map(|c| {
                    let gamma_formal = res.widths[c];
                    let ch = &sg.channels[c];
                    let pp_c = &particle_pairs[ch.particle_pair_idx];
                    // P_c at the resonance energy E_n.  Returns None when the channel
                    // is closed at E_n (bound-state resonance), Some(P) otherwise.
                    //
                    // The fallback must be gated on whether e_cm_n ≤ 0, NOT on
                    // whether P is numerically small.  For genuinely open channels
                    // near threshold (high-l, small ρ), P is positive but tiny;
                    // a magnitude guard would replace √(|Γ|/(2P)) ≫ 1 with √|Γ| ≪ 1,
                    // underestimating the reduced amplitude by orders of magnitude.
                    // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml01.f Pgh.
                    let p_at_en: Option<f64> = if pp_c.ma < 0.5 {
                        Some(1.0) // photon: P = 1 by convention
                    } else {
                        // P1: res.energy is lab-frame; convert to CM before adding Q.
                        // Reference: SAMMY rml/mrml03.f Fxradi; ENDF-6 §2.2.1.6.
                        let e_cm_n = channel::lab_to_cm_energy(res.energy, awr) + pp_c.q;
                        if e_cm_n <= 0.0 {
                            None // channel closed at resonance energy (bound state)
                        } else {
                            let redmas = pp_c.ma * pp_c.mb / (pp_c.ma + pp_c.mb);
                            let k_cn = channel::wave_number_from_cm(e_cm_n, redmas);
                            let rho_eff_n = k_cn * ch.effective_radius;
                            // Must use the same penetrability type as the open-channel
                            // block: Coulomb P_c(E_n) for charged pairs, hard-sphere
                            // otherwise.  Mixing them produces inconsistent γ_nc
                            // normalisation: γ_nc = √(Γ_nc / (2·P_c(E_n))).
                            // SAMMY rml/mrml07.f Pgh — same Zeta check applies here.
                            let p = if pp_c.za.abs() > 0.5 && pp_c.zb.abs() > 0.5 {
                                let eta = coulomb::sommerfeld_eta(
                                    pp_c.za, pp_c.zb, pp_c.ma, pp_c.mb, e_cm_n,
                                );
                                coulomb::coulomb_penetrability(ch.l, eta, rho_eff_n)
                            } else {
                                penetrability::penetrability(ch.l, rho_eff_n)
                            };
                            Some(p)
                        }
                    };
                    match p_at_en {
                        None => {
                            // Closed channel at E_n: formal width used directly as reduced
                            // amplitude (SAMMY convention for bound-state resonances).
                            gamma_formal.abs().sqrt().copysign(gamma_formal)
                        }
                        Some(p) => {
                            // Open channel: γ = √(|Γ| / (2·P_c(E_n))) with sign of Γ.
                            let magnitude = (gamma_formal.abs() / (2.0 * p)).sqrt();
                            magnitude.copysign(gamma_formal)
                        }
                    }
                })
                .collect();
            (gamma_vals, e_tilde)
        } else {
            // KRM=2: widths are already reduced amplitudes; real denominator.
            // P2: Guard only against exact IEEE 754 zero; complex infrastructure
            // handles the Lorentzian width naturally via i·P_c in level matrix.
            let e_tilde = Complex64::new(res.energy, 0.0);
            (res.widths.clone(), e_tilde)
        };

        let denom = e_tilde - energy_ev;
        // Near-pole regularization: add a tiny imaginary offset ε to the denominator
        // so that evaluating exactly at E = E_n (where denom → 0 for real KRM=2 poles)
        // gives a finite, physically meaningful result via the Cauchy principal value.
        // For KRM=3, e_tilde already carries −iΓ_γ/2 so the denominator is never zero;
        // the correction is negligible (ε << Γ_γ/2).
        // Reference: Cauchy PV regularisation; SAMMY avoids the exact pole by perturbing
        // the resonance energy during input processing.
        let inv_denom = if denom.norm() < 1e-10 {
            (denom + Complex64::new(0.0, 1e-10)).inv()
        } else {
            denom.inv()
        };
        for (c, row) in r_cplx.iter_mut().enumerate() {
            for (cp, elem) in row.iter_mut().enumerate() {
                *elem += gamma_vals[c] * gamma_vals[cp] * inv_denom;
            }
        }
    }

    // ── L_c = (S_c - B_c) + i·P_c (per-channel level denominator) ───────────
    // Reference: SAMMY rml/mrml07.f Pgh subroutine, "PH = 1/(S-B+IP)"
    let l_c: Vec<Complex64> = (0..nch)
        .map(|c| Complex64::new(s_c[c] - sg.channels[c].boundary, p_c[c]))
        .collect();

    // ── Reduced level matrix Ỹ = L⁻¹ - R (SAMMY "Ymat") ─────────────────────
    // Ỹ_cc'(E) = (1/L_c)·δ_cc' - R_cc'
    // Reference: SAMMY rml/mrml07.f — "Ymat = (1/(S-B+IP) - Rmat)"
    //
    // NOTE: This is NOT (L - R). The SAMMY formulation inverts L⁻¹ - R, not
    // L - R. Using L - R gives |U| = 3 for the hard sphere (R=0, A=iP,
    // A⁻¹·P = -i/P·P = -i, W = 1+2i²·(−1)=3) — catastrophically wrong.
    // Using L⁻¹ - R gives |U| = 1 for R=0 (Ỹ = 1/L, Ỹinv = L, XQ = L·0 = 0,
    // XXXX = 0, W = 1, U = exp(2iφ)) — correct hard-sphere limit.
    let y_tilde: Vec<Vec<Complex64>> = (0..nch)
        .map(|c| {
            (0..nch)
                .map(|cp| {
                    // L_c = (S_c − B_c) + i·P_c.
                    // For SHF=0 closed channels: S_c = B_c and P_c = 0 ⇒ L_c = 0.
                    // Correct limit: 1/L_c → ∞ ⇒ Ỹ[c,c] >> R[c,c] ⇒ Ỹ⁻¹[c,c] ≈ 0
                    // ⇒ channel decouples from U.  Setting 1/L_c = 0 (old bug) removes
                    // the diagonal and lets R dominate — wrong coupling / Ỹ singular.
                    //
                    // For SHF=1 or non-matching B_c, L_c is generally finite even when
                    // P_c = 0; the dispersive (real) shift must be preserved.  Do NOT
                    // force the sentinel just because the channel is sub-threshold; check
                    // whether |L_c| is actually near zero.
                    //
                    // Reference: SAMMY rml/mrml07.f — PH = 1/(S−B+iP).
                    let inv_l = if l_c[c].norm_sqr() < 1e-60 {
                        // |L_c| < 1e-30: use finite-but-large sentinel so the diagonal
                        // dominates and the channel decouples without overflow in inversion.
                        Complex64::new(1e30, 0.0)
                    } else {
                        Complex64::new(1.0, 0.0) / l_c[c]
                    };
                    let diag = if c == cp { inv_l } else { Complex64::ZERO };
                    diag - r_cplx[c][cp]
                })
                .collect()
        })
        .collect();

    // ── Invert Ỹ to get Ỹinv (SAMMY "Yinv") ─────────────────────────────────
    // Reference: SAMMY rml/mrml09.f Yinvrs subroutine
    let y_inv = match invert_complex_matrix(&y_tilde, nch) {
        Some(inv) => inv,
        None => return (0.0, 0.0, 0.0, 0.0), // singular → skip spin group
    };

    // ── XQ = Ỹinv · R (matrix product, SAMMY "Xqr/Xqi") ─────────────────────
    // Reference: SAMMY rml/mrml11.f Setxqx — "Xqr(k,i) = (L**-1-R)**-1 * R"
    let xq: Vec<Vec<Complex64>> = (0..nch)
        .map(|c| {
            (0..nch)
                .map(|cp| (0..nch).map(|k| y_inv[c][k] * r_cplx[k][cp]).sum())
                .collect()
        })
        .collect();

    // ── XXXX = (√P_c / L_c) · XQ · √P_c' ────────────────────────────────────
    // Reference: SAMMY rml/mrml11.f Setxqx — "Xxxx = sqrt(P)/L * xq * sqrt(P)"
    let sqrt_p: Vec<f64> = p_c.iter().map(|&x| x.sqrt()).collect();
    let xxxx: Vec<Vec<Complex64>> = (0..nch)
        .map(|c| {
            // For a closed channel: sqrt_p[c] = 0 and L_c = 0 (0/0 indeterminate).
            // The full XXXX[c,cp] = (√P_c / L_c) · XQ[c,cp] · √P_c'.
            // Since √P_c = 0 for any closed channel c, the entire row is zero
            // regardless of the value of L_c. Setting sqrt_p_over_l = 0 is correct.
            // (The Ỹ sentinel handles Ỹ inversion correctly; this row zeroing is
            //  consistent: a closed channel contributes nothing to XXXX/U.)
            let sqrt_p_over_l = if is_closed[c] {
                Complex64::ZERO
            } else {
                sqrt_p[c] / l_c[c]
            };
            (0..nch)
                .map(|cp| sqrt_p_over_l * xq[c][cp] * sqrt_p[cp])
                .collect()
        })
        .collect();

    // ── Collision matrix U = Ω · W · Ω, W = I + 2i·Ξ ────────────────────────
    // Reference: SAMMY manual eq. III.D.4; SAMMY rml/mrml11.f Setxqx/Sectio
    // Hard-sphere check: R=0 → XQ=0 → XXXX=0 → W=I → U = exp(2iφ)·I, |U|=1 ✓
    let omega: Vec<Complex64> = phi_c
        .iter()
        .map(|&phi| Complex64::from_polar(1.0, phi))
        .collect();

    let u: Vec<Vec<Complex64>> = (0..nch)
        .map(|c| {
            (0..nch)
                .map(|cp| {
                    let delta = if c == cp {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::ZERO
                    };
                    let w_cc = delta + Complex64::new(0.0, 2.0) * xxxx[c][cp];
                    omega[c] * w_cc * omega[cp]
                })
                .collect()
        })
        .collect();

    // ── Cross-sections (sum over entrance channels) ───────────────────────────
    // Optical theorem gives σ_total = 2·(π/k²)·g_J·(1 - Re(U_cc)) per channel.
    // Reference: SAMMY rml/mrml11.f Sectio subroutine; SAMMY manual §3.1 Eq. 3.4
    let mut tot = 0.0;
    let mut elas = 0.0;
    let mut cap = 0.0;
    let mut fis = 0.0;
    let mut inel = 0.0; // inelastic neutron channels (MT=51+): tracked separately

    // Whether this spin group has explicit capture (photon) channels in the
    // level matrix.  KRM=2 with photon channels: yes.  KRM=3: no (capture is
    // implicit via complex poles; no MT=102 channel appears in NCH).
    let has_explicit_capture = is_capture.iter().any(|&x| x);

    for c0 in 0..nch {
        if !is_entrance[c0] {
            continue;
        }
        let u_diag = u[c0][c0];
        // σ_total (optical theorem, per entrance channel)
        tot += 2.0 * pok2 * g_j * (1.0 - u_diag.re);
        // σ_elastic: |1 - U_{c0,c0}|²
        elas += pok2 * g_j * (Complex64::new(1.0, 0.0) - u_diag).norm_sqr();

        for cp in 0..nch {
            if is_fission[cp] {
                // σ_fission: |U_{c0,c'}|² for fission channels c'
                fis += pok2 * g_j * u[c0][cp].norm_sqr();
            }
            if has_explicit_capture && is_capture[cp] {
                // σ_capture (explicit): |U_{c0,c'}|² for photon channels c'.
                // Avoids lumping inelastic neutron channels (MT=51+) into capture.
                // Reference: SAMMY rml/mrml11.f Sectio — explicit sum over γ channels.
                cap += pok2 * g_j * u[c0][cp].norm_sqr();
            }
            if is_inelastic[cp] {
                // σ_inelastic: |U_{c0,c'}|² for inelastic neutron channels (MT=51+).
                // Tracked separately so KRM=3 capture residual excludes this flux.
                // Reference: ENDF MT conventions §3.4; SAMMY rml/mrml11.f Sectio.
                inel += pok2 * g_j * u[c0][cp].norm_sqr();
            }
        }
    }

    if krm == 3 && !has_explicit_capture {
        // KRM=3 (Reich-Moore approximation): capture is implicit via complex poles
        // (Ẽ_n = E_n - i·Γγ/2).  Flux not going to elastic, fission, or inelastic
        // channels is capture.  Inelastic flux must be excluded; folding it into
        // capture would mislabel σ_capture when MT=51+ channels are present.
        // Clamp to ≥0 for floating-point safety near pole energies.
        // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml11.f Sectio.
        cap = (tot - elas - fis - inel).max(0.0);
    }
    // KRM=2: capture was accumulated explicitly above from MT=102 channels.
    // Do NOT add residual flux — it may include inelastic (MT=51+) contributions
    // and would mislabel them as capture, biasing channel-resolved fits.
    // Reference: SAMMY rml/mrml11.f Sectio (explicit γ-channel sum for KRM=2).

    (tot, elas, cap, fis)
}

// ── Complex Gauss-Jordan Elimination ─────────────────────────────────────────
//
// Inverts an n×n complex matrix using Gauss-Jordan elimination with partial
// pivoting. Returns None if the matrix is singular (pivot magnitude < 1e-300).
//
// For LRF=7 isotopes relevant to VENUS imaging, NCH ≤ 6, so O(n³) is fast.
// SAMMY uses a specialized complex symmetric factorization (Xspfa/Xspsl in
// rml/mrml10.f), but Gauss-Jordan is correct and sufficient for our purposes.

fn invert_complex_matrix(a: &[Vec<Complex64>], n: usize) -> Option<Vec<Vec<Complex64>>> {
    // Build augmented matrix [A | I] of size n × 2n
    let mut aug: Vec<Vec<Complex64>> = (0..n)
        .map(|r| {
            let mut row = a[r].clone();
            for c in 0..n {
                row.push(if c == r {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::ZERO
                });
            }
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivoting: find row with largest magnitude in this column
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            aug[r1][col]
                .norm()
                .partial_cmp(&aug[r2][col].norm())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.norm() < 1e-300 {
            return None; // singular
        }

        // Scale pivot row so leading entry becomes 1
        let inv_pivot = pivot.inv();
        for elem in aug[col].iter_mut() {
            *elem *= inv_pivot;
        }

        // Eliminate this column from all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor.norm() < 1e-300 {
                continue;
            }
            let col_scaled: Vec<Complex64> = aug[col].iter().map(|&x| factor * x).collect();
            for (r_elem, sub) in aug[row].iter_mut().zip(col_scaled) {
                *r_elem -= sub;
            }
        }
    }

    // Extract the right half (the inverse)
    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_inversion() {
        let n = 3;
        let a: Vec<Vec<Complex64>> = (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| {
                        if r == c {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::ZERO
                        }
                    })
                    .collect()
            })
            .collect();
        let inv = invert_complex_matrix(&a, n).unwrap();
        for (r, row) in inv.iter().enumerate() {
            for (c, val) in row.iter().enumerate() {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (val.re - expected).abs() < 1e-12,
                    "inv[{r}][{c}].re = {}, expected {expected}",
                    val.re
                );
                assert!(
                    val.im.abs() < 1e-12,
                    "inv[{r}][{c}].im = {} should be 0",
                    val.im
                );
            }
        }
    }

    #[test]
    fn test_2x2_complex_inversion() {
        // A = [[2+i, 1], [0, 3-2i]]  → A⁻¹ = [[1/(2+i), -1/((2+i)(3-2i))], [0, 1/(3-2i)]]
        let a00 = Complex64::new(2.0, 1.0);
        let a01 = Complex64::new(1.0, 0.0);
        let a11 = Complex64::new(3.0, -2.0);
        let a = vec![vec![a00, a01], vec![Complex64::ZERO, a11]];
        let inv = invert_complex_matrix(&a, 2).unwrap();

        // Verify A · A⁻¹ ≈ I
        let i00 = a00 * inv[0][0] + a01 * inv[1][0];
        let i01 = a00 * inv[0][1] + a01 * inv[1][1];
        let i11 = a11 * inv[1][1];
        assert!((i00.re - 1.0).abs() < 1e-12, "i00.re = {}", i00.re);
        assert!(i00.im.abs() < 1e-12, "i00.im = {}", i00.im);
        assert!(i01.norm() < 1e-12, "i01 = {}", i01);
        assert!((i11.re - 1.0).abs() < 1e-12, "i11.re = {}", i11.re);
    }

    #[test]
    fn test_singular_returns_none() {
        let a = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        assert!(invert_complex_matrix(&a, 2).is_none());
    }

    /// Hard-sphere unitarity check: with R = 0, U must equal exp(2iφ)·I
    /// and σ_total = 2·(π/k²)·g_J·(1 − cos 2φ) ≥ 0.
    ///
    /// This test is purely local (no network) and guards against the
    /// classic sign error where U = 3·exp(2iφ)·I (|U| = 3) that arises
    /// when using A = L − R instead of SAMMY's Ỹ = L⁻¹ − R.
    #[test]
    fn test_hard_sphere_unitarity() {
        use nereids_endf::resonance::{ParticlePair, RmlChannel, RmlData, SpinGroup};

        // Minimal synthetic LRF=7 / KRM=2 with a single elastic channel
        // and NO resonances.  Result must be pure hard-sphere scattering.
        let pp = ParticlePair {
            ma: 1.0,
            mb: 184.0,
            za: 0.0,
            zb: 74.0 * 184.0,
            ia: 0.5,
            ib: 0.0,
            q: 0.0,
            pnt: 1,
            shf: 0, // SHF=0 → S_c = B_c → L_c = iP_c
            mt: 2,
            pa: 1.0,
            pb: 1.0,
        };
        let channel = RmlChannel {
            particle_pair_idx: 0,
            l: 0,
            channel_spin: 0.5,
            boundary: 0.0,
            effective_radius: 8.3,
            true_radius: 8.3,
        };
        let sg = SpinGroup {
            j: 0.5,
            parity: 1.0,
            channels: vec![channel],
            resonances: vec![], // no resonances: pure hard sphere
            has_background_correction: false,
        };
        let rml = RmlData {
            target_spin: 0.0,
            awr: 183.0,
            scattering_radius: 8.3,
            krm: 2,
            particle_pairs: vec![pp],
            spin_groups: vec![sg],
        };

        // Evaluate at several energies.  σ_total must be non-negative,
        // and σ_capture/σ_fission must be zero (no absorption channels).
        //
        // Note: cap is computed as (tot - elas - fis); since tot and elas are
        // calculated via different floating-point paths they may differ by
        // ~1e-15 × pok2 (where pok2 can be ~1e5 b at low energy), giving a
        // residual |cap| ~ 1e-10 b.  Use a relative tolerance of 1e-9.
        for &e_ev in &[10.0, 50.0, 100.0, 500.0, 1000.0] {
            let (tot, elas, cap, fis) = cross_sections_for_rml_range(&rml, e_ev);
            assert!(
                tot >= 0.0,
                "hard sphere σ_total < 0 at {e_ev} eV: {tot:.6} b"
            );
            let tol = 1e-9 * tot.abs().max(1.0);
            assert!(
                cap.abs() < tol,
                "hard sphere σ_capture ≠ 0 at {e_ev} eV: {cap:.3e} b (tol={tol:.3e})"
            );
            assert!(
                fis.abs() < tol,
                "hard sphere σ_fission ≠ 0 at {e_ev} eV: {fis:.3e} b (tol={tol:.3e})"
            );
            // σ_elastic ≈ σ_total (capture=fission=0)
            assert!(
                (tot - elas).abs() < tol,
                "σ_total ≠ σ_elastic at {e_ev} eV: tot={tot:.6}, elas={elas:.6}"
            );
        }
    }

    /// Coulomb exit channels route through coulomb::coulomb_penetrability,
    /// not the hard-sphere Blatt-Weisskopf functions.
    ///
    /// Constructs a 2-channel spin group:
    ///   ch0: neutron (za=0)   + target — hard-sphere entrance channel
    ///   ch1: α (za=2004)      + O-16 (za=8016) — Coulomb exit, Q=+50 eV (always open)
    ///
    /// Verifies σ_total ≥ 0 (physics sanity) and no panic at both an open
    /// and a closed Coulomb channel (Q very negative).
    ///
    /// SAMMY ref: rml/mrml07.f Pgh — `if (Zeta(I).NE.Zero)` branch.
    #[test]
    fn test_coulomb_channel_open_and_closed_no_panic() {
        use nereids_endf::resonance::{ParticlePair, RmlChannel, RmlData, SpinGroup};

        // Entrance channel: neutron (za=0) + target
        let pp_entrance = ParticlePair {
            ma: 1.0,
            mb: 184.0,
            za: 0.0,
            zb: 74.0 * 1000.0 + 184.0, // ZA = Z*1000 + A
            ia: 0.5,
            ib: 0.0,
            q: 0.0,
            pnt: 1,
            shf: 0,
            mt: 2,
            pa: 1.0,
            pb: 1.0,
        };

        // Coulomb exit channel: α(za=2004) + O-16(za=8016), Q=+50 eV → always open.
        // sommerfeld_eta: Z_α=2, Z_O=8; η checks that we entered the Coulomb branch.
        let pp_coulomb_open = ParticlePair {
            ma: 4.0,
            mb: 16.0,
            za: 2004.0, // alpha: Z=2, A=4
            zb: 8016.0, // O-16: Z=8, A=16
            ia: 0.0,
            ib: 0.0,
            q: 50.0, // Q > 0 → e_c = e_cm + 50 > 0 for all positive energies
            pnt: 1,
            shf: 0,
            mt: 22, // (n,α)
            pa: 1.0,
            pb: 1.0,
        };

        // Coulomb exit channel with Q very negative → closed at all reasonable energies.
        let pp_coulomb_closed = ParticlePair {
            ma: 4.0,
            mb: 16.0,
            za: 2004.0,
            zb: 8016.0,
            ia: 0.0,
            ib: 0.0,
            q: -1e6, // far below threshold
            pnt: 1,
            shf: 0,
            mt: 22,
            pa: 1.0,
            pb: 1.0,
        };

        // Build and evaluate the open-channel case.
        for (desc, pp_exit, expect_positive_total) in [
            ("open Coulomb exit", &pp_coulomb_open, true),
            ("closed Coulomb exit", &pp_coulomb_closed, false),
        ] {
            let ch0 = RmlChannel {
                particle_pair_idx: 0,
                l: 0,
                channel_spin: 0.5,
                boundary: 0.0,
                effective_radius: 8.3,
                true_radius: 8.3,
            };
            let ch1 = RmlChannel {
                particle_pair_idx: 1,
                l: 0,
                channel_spin: 0.5,
                boundary: 0.0,
                effective_radius: 5.0,
                true_radius: 5.0,
            };
            let sg = SpinGroup {
                j: 0.5,
                parity: 1.0,
                channels: vec![ch0, ch1],
                resonances: vec![],
                has_background_correction: false,
            };
            let rml = RmlData {
                target_spin: 0.0,
                awr: 183.0,
                scattering_radius: 8.3,
                krm: 2,
                particle_pairs: vec![pp_entrance.clone(), pp_exit.clone()],
                spin_groups: vec![sg],
            };

            let (tot, _elas, _cap, _fis) = cross_sections_for_rml_range(&rml, 100.0);
            assert!(tot >= 0.0, "{desc}: σ_total = {tot:.6} b must be ≥ 0");
            if expect_positive_total {
                // Hard-sphere entrance channel alone gives positive σ_total
                // (the Coulomb channel merely adds a second channel but no resonances).
                assert!(
                    tot > 0.0,
                    "{desc}: σ_total = {tot} b should be > 0 (hard-sphere entrance channel)"
                );
            }
        }
    }

    /// Verify W-184 cross-sections show resonance structure.
    ///
    /// Downloads W-184 ENDF/B-VIII.0, parses LRF=7 parameters,
    /// then checks that σ_total at the first resonance (~101.9 eV) is
    /// significantly larger than the background off-resonance.
    ///
    /// Run with: cargo test -p nereids-physics -- --ignored test_w184_cross_section_peak
    #[test]
    #[ignore = "requires network: downloads W-184 ENDF from IAEA (~50 kB)"]
    fn test_w184_cross_section_peak() {
        use crate::reich_moore::cross_sections_at_energy;
        use nereids_core::types::Isotope;
        use nereids_endf::parser::parse_endf_file2;
        use nereids_endf::retrieval::{EndfLibrary, EndfRetriever};

        let retriever = EndfRetriever::new();
        let isotope = Isotope::new(74, 184);
        let (_, text) = retriever
            .get_endf_file(&isotope, EndfLibrary::EndfB8_0, 7437)
            .expect("Failed to download W-184 ENDF/B-VIII.0");

        let data = parse_endf_file2(&text).expect("Failed to parse W-184 ENDF");

        // W-184 ENDF/B-VIII.0 (KRM=3, 3 spin groups J=1/2+, J=1/2-, J=3/2-):
        //   First positive resonance in J=1/2+ spin group: ~101.9 eV
        //   Background between the -386 eV bound state and 101.9 eV resonance: ~0.07 b
        // Test that σ_total at the 101.9 eV resonance peak >> background at 50 eV.
        let xs_on_res = cross_sections_at_energy(&data, 101.9);
        let xs_off_res = cross_sections_at_energy(&data, 50.0);

        // Background must be non-negative (guards against the A=L-R sign error
        // that gave σ(50 eV) = −0.228 b).
        assert!(
            xs_off_res.total >= 0.0,
            "σ_total at 50 eV must be non-negative (hard-sphere background), got {:.4} b",
            xs_off_res.total
        );
        assert!(
            xs_on_res.total > 0.0,
            "σ_total at 101.9 eV should be positive, got {}",
            xs_on_res.total
        );
        assert!(
            xs_on_res.capture >= 0.0,
            "σ_capture at 101.9 eV should be non-negative, got {}",
            xs_on_res.capture
        );
        assert!(
            xs_on_res.total > xs_off_res.total * 5.0,
            "Resonance peak at 101.9 eV should be >5× the 50 eV background: \
             σ(101.9)={:.3} vs σ(50.0)={:.3} barns",
            xs_on_res.total,
            xs_off_res.total
        );

        println!(
            "W-184 σ_total: {:.3} b at 101.9 eV (resonance), {:.3} b at 50.0 eV (background)",
            xs_on_res.total, xs_off_res.total
        );
        println!(
            "W-184 σ_capture: {:.4} b, σ_elastic: {:.4} b at 101.9 eV",
            xs_on_res.capture, xs_on_res.elastic
        );
    }
}
