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
//! (ENDF-6 Formats Manual Appendix D; SAMMY manual §3.1)
//!
//! ```text
//! R-matrix:
//!   R_cc'(E) = Σ_n γ_nc · γ_nc' / (E_n - E)   [real, NCH×NCH, symmetric]
//!   γ_nc = reduced width amplitude for resonance n in channel c (eV^{1/2})
//!
//! Level matrix:
//!   A_cc'(E) = δ_cc'(S_c(E) - B_c + i·P_c(E)) - R_cc'(E)   [complex, NCH×NCH]
//!   P_c = penetrability,  S_c = shift factor,  B_c = boundary condition
//!
//! Collision matrix:
//!   U_cc'(E) = Ω_c · [δ_cc' + 2i·√P_c · (A⁻¹)_cc' · √P_c'] · Ω_c'
//!   Ω_c = exp(iφ_c)  [hard-sphere phase factor]
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

use crate::{channel, penetrability};

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
        let (t, e, cap, fis) =
            spin_group_cross_sections(sg, &rml.particle_pairs, energy_ev, rml.awr, rml.target_spin);
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
) -> (f64, f64, f64, f64) {
    let nch = sg.channels.len();
    if nch == 0 || sg.resonances.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let g_j = channel::statistical_weight(sg.j, target_spin);
    let pok2 = channel::pi_over_k_squared_barns(energy_ev, awr);

    // ── Per-channel quantities ────────────────────────────────────────────────
    let mut p_c = vec![0.0f64; nch]; // penetrability P_c
    let mut s_c = vec![0.0f64; nch]; // shift factor S_c
    let mut phi_c = vec![0.0f64; nch]; // hard-sphere phase φ_c
    let mut is_entrance = vec![false; nch];
    let mut is_fission = vec![false; nch];

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
                // Closed channel (below threshold): penetrability → 0.
                p_c[c] = 0.0;
                s_c[c] = 0.0;
                phi_c[c] = 0.0;
            } else {
                // Channel wave number from reduced mass μ = MA·MB/(MA+MB).
                // For elastic (MA=1, MB=AWR): k_c = wave_number(E_lab, AWR) [identical].
                let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
                let k_c = channel::wave_number_from_cm(e_c, redmas);
                // APE (effective radius) for P_c, S_c; APT (true radius) for φ_c.
                // Reference: SAMMY rml/mrml07.f (Pgh, Sinsix, Pf subroutines)
                let rho_eff = k_c * ch.effective_radius;
                let rho_true = k_c * ch.true_radius;
                p_c[c] = penetrability::penetrability(ch.l, rho_eff);
                s_c[c] = penetrability::shift_factor(ch.l, rho_eff);
                phi_c[c] = penetrability::phase_shift(ch.l, rho_true);
            }
        }
    }

    // ── R-matrix (real, NCH×NCH symmetric) ───────────────────────────────────
    // R_cc'(E) = Σ_n γ_nc · γ_nc' / (E_n - E)
    // Reference: SAMMY rml/mrml07.f Setr subroutine
    let mut r = vec![vec![0.0f64; nch]; nch];
    for res in &sg.resonances {
        let denom = res.energy - energy_ev;
        // P2: Retain the pole term even when denom → 0. The old `continue` dropped
        // the resonance entirely, causing a non-physical dip in σ at that energy.
        // The level matrix's imaginary diagonal i·P_c provides Lorentzian width
        // regularization so cross-sections remain finite at resonance energies.
        // Guard only against exact IEEE 754 zero (vanishingly rare in practice).
        // Reference: SAMMY rml/mrml07.f Setr — no special-casing for near-pole.
        let inv_denom = 1.0 / if denom == 0.0 { 1e-50_f64 } else { denom };
        for c in 0..nch {
            for cp in 0..nch {
                r[c][cp] += res.widths[c] * res.widths[cp] * inv_denom;
            }
        }
    }

    // ── Level matrix A (complex, NCH×NCH) ────────────────────────────────────
    // A_cc'(E) = δ_cc'(S_c - B_c + i·P_c) - R_cc'
    // Reference: SAMMY rml/mrml09.f Yinvrs subroutine (builds level matrix)
    let mut a: Vec<Vec<Complex64>> = (0..nch)
        .map(|c| {
            (0..nch)
                .map(|cp| {
                    let diag = if c == cp {
                        Complex64::new(s_c[c] - sg.channels[c].boundary, p_c[c])
                    } else {
                        Complex64::ZERO
                    };
                    diag - r[c][cp]
                })
                .collect()
        })
        .collect();

    // ── Invert level matrix ───────────────────────────────────────────────────
    let a_inv = match invert_complex_matrix(&mut a, nch) {
        Some(inv) => inv,
        None => return (0.0, 0.0, 0.0, 0.0), // singular level matrix → skip
    };

    // ── Collision matrix U (complex, NCH×NCH) ────────────────────────────────
    // U_cc' = Ω_c · [δ_cc' + 2i·√P_c · (A⁻¹)_cc' · √P_c'] · Ω_c'
    // where Ω_c = exp(iφ_c) is the hard-sphere S-matrix element.
    // Reference: SAMMY rml/mrml11.f Setxqx subroutine
    let omega: Vec<Complex64> = phi_c
        .iter()
        .map(|&phi| Complex64::from_polar(1.0, phi))
        .collect();
    let sqrt_p: Vec<f64> = p_c.iter().map(|&x| x.sqrt()).collect();

    let u: Vec<Vec<Complex64>> = (0..nch)
        .map(|c| {
            (0..nch)
                .map(|cp| {
                    let delta = if c == cp {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::ZERO
                    };
                    let coupling = Complex64::new(0.0, 2.0) * sqrt_p[c] * a_inv[c][cp] * sqrt_p[cp];
                    omega[c] * (delta + coupling) * omega[cp]
                })
                .collect()
        })
        .collect();

    // ── Cross-sections (sum over entrance channels) ───────────────────────────
    // Optical theorem gives σ_total = 2·(π/k²)·g_J·(1 - Re(U_cc)) per channel.
    // Reference: SAMMY rml/mrml11.f Sectio subroutine; SAMMY manual §3.1 Eq. 3.4
    let mut tot = 0.0;
    let mut elas = 0.0;
    let mut fis = 0.0;

    for c0 in 0..nch {
        if !is_entrance[c0] {
            continue;
        }
        let u_diag = u[c0][c0];
        // σ_total contribution (optical theorem, per entrance channel)
        tot += 2.0 * pok2 * g_j * (1.0 - u_diag.re);
        // σ_elastic: |1 - U_{c0,c0}|²
        elas += pok2 * g_j * (Complex64::new(1.0, 0.0) - u_diag).norm_sqr();
        // σ_fission: |U_{c0,c'}|² for fission channels c'
        for cp in 0..nch {
            if is_fission[cp] {
                fis += pok2 * g_j * u[c0][cp].norm_sqr();
            }
        }
    }

    // σ_capture = σ_total - σ_elastic - σ_fission (non-negative by unitarity)
    let cap = (tot - elas - fis).max(0.0);

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

fn invert_complex_matrix(a: &mut Vec<Vec<Complex64>>, n: usize) -> Option<Vec<Vec<Complex64>>> {
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
        for j in 0..2 * n {
            aug[col][j] *= inv_pivot;
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
            for j in 0..2 * n {
                let sub = factor * aug[col][j];
                aug[row][j] -= sub;
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
        let mut a: Vec<Vec<Complex64>> = (0..n)
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
        let inv = invert_complex_matrix(&mut a, n).unwrap();
        for r in 0..n {
            for c in 0..n {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (inv[r][c].re - expected).abs() < 1e-12,
                    "inv[{r}][{c}].re = {}, expected {expected}",
                    inv[r][c].re
                );
                assert!(
                    inv[r][c].im.abs() < 1e-12,
                    "inv[{r}][{c}].im = {} should be 0",
                    inv[r][c].im
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
        let mut a = vec![vec![a00, a01], vec![Complex64::ZERO, a11]];
        let inv = invert_complex_matrix(&mut a, 2).unwrap();

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
        let mut a = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        assert!(invert_complex_matrix(&mut a, 2).is_none());
    }

    /// Verify W-184 cross-sections show resonance structure.
    ///
    /// Downloads W-184 ENDF/B-VIII.0, parses LRF=7 parameters,
    /// then checks that σ_total at the first resonance (~7.6 eV) is
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

        // σ_total at first resonance (~7.6 eV) should be much larger than background
        let xs_on_res = cross_sections_at_energy(&data, 7.6);
        let xs_off_res = cross_sections_at_energy(&data, 5.0);

        assert!(
            xs_on_res.total > 0.0,
            "σ_total at 7.6 eV should be positive, got {}",
            xs_on_res.total
        );
        assert!(
            xs_on_res.total > xs_off_res.total * 5.0,
            "Resonance peak at 7.6 eV should be >5× the 5 eV background: \
             σ(7.6)={:.1} vs σ(5.0)={:.1} barns",
            xs_on_res.total,
            xs_off_res.total
        );

        println!(
            "W-184 σ_total: {:.1} b at 7.6 eV (resonance), {:.1} b at 5.0 eV (background)",
            xs_on_res.total, xs_off_res.total
        );
    }
}
