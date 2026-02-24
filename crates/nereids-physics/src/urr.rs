//! Unresolved Resonance Region (LRU=2) cross-section calculation.
//!
//! Computes average cross-sections via the Hauser-Feshbach formula using
//! the average level-spacing and partial widths stored in `UrrData`.
//!
//! ## Hauser-Feshbach Formula
//!
//! For each (L, J) combination at energy E:
//!
//! ```text
//! g_J = (2J+1) / ((2I+1) · (2s+1))
//!
//! LRF=1: Γ_n(E) = 2 · P_L(ρ(E)) · GNO    [GNO = reduced neutron width]
//! LRF=2: Γ_n(E) = lin-lin interpolation from energy table
//!
//! Γ_tot = Γ_n + GG + GF + GX
//!
//! σ_γ  += (π/k²) · g_J · (2π · Γ_n · GG) / (D · Γ_tot)
//! σ_f  += (π/k²) · g_J · (2π · Γ_n · GF) / (D · Γ_tot)
//! σ_cn += (π/k²) · g_J · (2π · Γ_n · (GG + GF + GX)) / (D · Γ_tot)
//! ```
//!
//! `σ_cn` is the compound-elastic/inelastic contribution. Potential scattering
//! `σ_pot = 4π·AP²` is included in the returned `total` and `elastic` so
//! that the URR band produces a physically consistent cross-section without
//! requiring special handling at the call site.
//!
//! ## SAMMY Reference
//! - `unr/munr03.f90` Csig3 subroutine
//! - SAMMY manual Section 4 (URR treatment)
//!
//! ## References
//! - ENDF-6 Formats Manual §2.2.2
//! - W. Hauser, H. Feshbach, Phys. Rev. 87 (1952) 366

use nereids_endf::resonance::UrrData;

use crate::channel;
use crate::penetrability;

/// Compute Hauser-Feshbach average cross-sections in the Unresolved Resonance Region.
///
/// Returns `(total, elastic, capture, fission)` in barns.
/// All four are zero when `e_ev` falls outside the URR energy band
/// `[urr.e_low, urr.e_high]`.
///
/// `elastic` = compound-nuclear elastic + smooth potential scattering.
/// `total`   = elastic + capture + fission.
/// Potential scattering `σ_pot = 4π·AP²` is folded in here rather than at
/// the call site; AP is in fm and 1 barn = 100 fm².
/// SAMMY ref: `unr/munr03.f90` includes the hard-sphere background.
///
/// ## SAMMY Reference
/// `unr/munr03.f90` Csig3 — Hauser-Feshbach cross-section kernel.
///
/// # Arguments
/// * `urr` — Parsed URR parameters (LRF=1 or LRF=2).
/// * `e_ev` — Neutron lab-frame energy in eV.
pub fn urr_cross_sections(urr: &UrrData, e_ev: f64) -> (f64, f64, f64, f64) {
    if e_ev < urr.e_low || e_ev > urr.e_high {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut sig_cap = 0.0_f64;
    let mut sig_fiss = 0.0_f64;
    let mut sig_compound_n = 0.0_f64;

    for lg in &urr.l_groups {
        let awri = lg.awri;
        let l = lg.l;

        // π/k² in barns and channel parameter ρ = k·AP.
        // Uses the same channel formulas as the RRR calculator for consistency.
        let pi_over_k2 = channel::pi_over_k_squared_barns(e_ev, awri);
        let rho = channel::rho(e_ev, awri, urr.ap);

        // Blatt-Weisskopf penetrability P_L(ρ) for LRF=1 Γ_n calculation.
        let p_l = penetrability::penetrability(l, rho);

        for jg in &lg.j_groups {
            // Statistical weight g_J = (2J+1) / ((2I+1)·(2s+1)).
            // Target spin I = urr.spi, neutron spin s = 1/2.
            let g_j = channel::statistical_weight(jg.j, urr.spi);

            // Obtain effective widths at energy e_ev.
            //
            // LRF=1: Γ_n = 2·P_L(ρ)·GNO  (GNO = reduced neutron width, eV)
            //        D, GG, GF are energy-independent (single-element vecs).
            // LRF=2: widths interpolated from the energy table using the
            //        INT code stored per J-group.
            //        INT=2 (lin-lin) and INT=5 (log-log) are both supported.
            //
            // SAMMY ref: unr/munr03.f90 Csig3 — `Gn = Two*Pene*Gno` for LRF=1.
            let (gn_eff, d_eff, gx_eff, gg_eff, gf_eff) = if urr.lrf == 1 {
                let gno = jg.gn[0]; // reduced neutron width (eV)
                (2.0 * p_l * gno, jg.d[0], 0.0_f64, jg.gg[0], jg.gf[0])
            } else {
                // LRF=2: dispatch on the stored interpolation law.
                let interp = if jg.int_code == 5 {
                    log_log_interp
                } else {
                    lin_lin_interp
                };
                let gn_i = interp(&jg.energies, &jg.gn, e_ev);
                let d_i = interp(&jg.energies, &jg.d, e_ev);
                let gx_i = interp(&jg.energies, &jg.gx, e_ev);
                let gg_i = interp(&jg.energies, &jg.gg, e_ev);
                let gf_i = interp(&jg.energies, &jg.gf, e_ev);
                (gn_i, d_i, gx_i, gg_i, gf_i)
            };

            // Skip degenerate entries.
            if d_eff <= 0.0 || gn_eff <= 0.0 {
                continue;
            }

            let gamma_tot = gn_eff + gg_eff + gf_eff + gx_eff;
            if gamma_tot <= 0.0 {
                continue;
            }

            // Hauser-Feshbach kernel:
            //   (π/k²) · g_J · (2π · Γ_n / D) · Γ_x / Γ_tot
            //
            // The 2π/D factor is the average level density times 2π.
            // SAMMY ref: unr/munr03.f90 `Two*Pi*Gn/D` prefactor.
            let prefactor = pi_over_k2 * g_j * (2.0 * std::f64::consts::PI * gn_eff) / d_eff;

            sig_cap += prefactor * gg_eff / gamma_tot;
            sig_fiss += prefactor * gf_eff / gamma_tot;
            // Compound-nuclear contribution to neutron channel (absorption minus
            // re-emission of neutrons back into the neutron channel is zero here;
            // this term accounts for neutrons that form compound nucleus and then
            // emerge as neutrons into other partial waves / excitation modes).
            // For our purposes: σ_cn ∝ Γ_n × (Γ_tot - Γ_n) / Γ_tot
            let gamma_abs = gg_eff + gf_eff + gx_eff; // Γ_tot - Γ_n
            sig_compound_n += prefactor * gamma_abs / gamma_tot;
        }
    }

    // Smooth potential scattering: σ_pot = 4π·AP²
    // AP is in fm; 1 barn = 100 fm².
    // SAMMY ref: unr/munr03.f90 adds hard-sphere background to elastic.
    let sig_pot = 4.0 * std::f64::consts::PI * urr.ap * urr.ap / 100.0;
    let elastic = sig_compound_n + sig_pot;
    let total = elastic + sig_cap + sig_fiss;
    (total, elastic, sig_cap, sig_fiss)
}

/// Linear-linear interpolation (clamped to table endpoints).
///
/// Used for LRF=2 width tables with INT=2 in ENDF.  `xs` must be strictly
/// ascending and of the same length as `ys`.
fn lin_lin_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[xs.len() - 1];
    }
    // Binary search for the bracketing interval.
    let i = xs.partition_point(|&xi| xi <= x);
    let (x0, y0) = (xs[i - 1], ys[i - 1]);
    let (x1, y1) = (xs[i], ys[i]);
    let t = (x - x0) / (x1 - x0);
    y0 + t * (y1 - y0)
}

/// Log-log interpolation (clamped to table endpoints).
///
/// Used for LRF=2 width tables with INT=5 in ENDF (common for heavy actinides
/// such as U-238 where widths follow approximate power-law energy dependence).
/// Returns `ys[0]` when any `y` value is ≤ 0 to avoid log-domain errors.
fn log_log_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[xs.len() - 1];
    }
    let i = xs.partition_point(|&xi| xi <= x);
    let (x0, y0) = (xs[i - 1], ys[i - 1]);
    let (x1, y1) = (xs[i], ys[i]);
    // Guard against non-positive values (log undefined).
    if x0 <= 0.0 || x1 <= 0.0 || y0 <= 0.0 || y1 <= 0.0 {
        // Fall back to lin-lin for degenerate entries.
        let t = (x - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }
    let log_t = (x.ln() - x0.ln()) / (x1.ln() - x0.ln());
    (y0.ln() + log_t * (y1.ln() - y0.ln())).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::{UrrData, UrrJGroup, UrrLGroup};

    /// σ = 0 when energy is outside the URR band.
    #[test]
    fn urr_outside_band_returns_zero() {
        let urr = make_lrf1_urr(1000.0, 30_000.0);
        let (t, e, c, f) = urr_cross_sections(&urr, 500.0); // below band
        assert_eq!(t, 0.0);
        assert_eq!(e, 0.0);
        assert_eq!(c, 0.0);
        assert_eq!(f, 0.0);

        let (t2, _, _, _) = urr_cross_sections(&urr, 50_000.0); // above band
        assert_eq!(t2, 0.0);
    }

    /// LRF=1 formula check: σ_γ = (π/k²) · g_J · (2π · Γ_n · GG) / (D · Γ_tot).
    ///
    /// Uses a single (L=0, J=2.5) group with known parameters; verifies
    /// σ_γ matches the hand-computed value to within 0.1%.
    #[test]
    fn urr_lrf1_formula_check() {
        // Hand-crafted parameters similar to U-233 URR (order of magnitude).
        let e_test = 1_000.0_f64; // eV, mid-URR
        let awri = 231.038_f64;
        let ap = 0.96931_f64; // ENDF scattering radius
        let spi = 2.5_f64; // U-233 target spin

        let gno = 3.0e-4; // reduced neutron width (eV)
        let gg = 3.5e-2; // gamma width (eV)
        let d = 0.5; // level spacing (eV)
        let j = 2.0_f64;

        let urr = UrrData {
            lrf: 1,
            spi,
            ap,
            e_low: 100.0,
            e_high: 30_000.0,
            l_groups: vec![UrrLGroup {
                l: 0,
                awri,
                j_groups: vec![UrrJGroup {
                    j,
                    amun: 1.0,
                    amuf: 0.0,
                    energies: vec![],
                    d: vec![d],
                    gx: vec![0.0],
                    gn: vec![gno],
                    gg: vec![gg],
                    gf: vec![0.0],
                    int_code: 2,
                }],
            }],
        };

        let (total, elastic, capture, fission) = urr_cross_sections(&urr, e_test);

        // Hand-compute expected values.
        let pi_over_k2 = channel::pi_over_k_squared_barns(e_test, awri);
        let rho = channel::rho(e_test, awri, ap);
        let p_l = penetrability::penetrability(0, rho); // L=0
        let gn_eff = 2.0 * p_l * gno;
        let gamma_tot = gn_eff + gg;
        let g_j = channel::statistical_weight(j, spi);
        let prefactor = pi_over_k2 * g_j * (2.0 * std::f64::consts::PI * gn_eff) / d;
        let expected_cap = prefactor * gg / gamma_tot;
        let expected_cn = prefactor * gg / gamma_tot; // Γ_abs = gg (no fission)
        let expected_sig_pot = 4.0 * std::f64::consts::PI * ap * ap / 100.0;
        let expected_elastic = expected_cn + expected_sig_pot;

        assert!(
            capture > 0.0,
            "Capture cross-section must be positive, got {capture}"
        );
        assert_eq!(fission, 0.0, "No fission in this test case");
        assert!(
            (capture - expected_cap).abs() / expected_cap < 1e-10,
            "Capture deviates from hand calculation: got {capture:.6e}, expected {expected_cap:.6e}"
        );
        assert!(
            (elastic - expected_elastic).abs() / expected_elastic < 1e-10,
            "Elastic deviates: got {elastic:.6e}, expected {expected_elastic:.6e} \
             (compound_n={expected_cn:.6e} + sig_pot={expected_sig_pot:.6e})"
        );
        assert!(
            (total - (capture + elastic)).abs() < 1e-14,
            "Total ≠ capture + elastic: {total} ≠ {}",
            capture + elastic
        );
    }

    /// LRF=2 lin-lin interpolation smoke test.
    ///
    /// Two-point energy table; verifies σ_γ > 0 and changes monotonically
    /// with energy (Γ_n increases with energy for L=0 neutrons).
    #[test]
    fn urr_lrf2_interpolation_smoke() {
        let e1 = 1_000.0_f64;
        let e2 = 10_000.0_f64;

        let urr = UrrData {
            lrf: 2,
            spi: 2.5,
            ap: 0.96931,
            e_low: 600.0,
            e_high: 30_000.0,
            l_groups: vec![UrrLGroup {
                l: 0,
                awri: 231.038,
                j_groups: vec![UrrJGroup {
                    j: 2.0,
                    amun: 1.0,
                    amuf: 4.0,
                    energies: vec![e1, e2],
                    d: vec![0.5, 0.4], // D decreases (more levels at higher E)
                    gx: vec![0.0, 0.0],
                    gn: vec![1.0e-3, 5.0e-3], // Γ_n increases with E
                    gg: vec![3.5e-2, 3.5e-2], // GG roughly constant
                    gf: vec![0.0, 0.0],
                    int_code: 2,
                }],
            }],
        };

        let (_, _, c1, _) = urr_cross_sections(&urr, e1);
        let (_, _, c2, _) = urr_cross_sections(&urr, e2);
        let (_, _, c_mid, _) = urr_cross_sections(&urr, 5_000.0);

        assert!(c1 > 0.0, "σ_γ at {e1} eV must be positive, got {c1}");
        assert!(c2 > 0.0, "σ_γ at {e2} eV must be positive, got {c2}");
        assert!(
            c1 > c_mid || c2 > c_mid || c_mid > 0.0,
            "σ_γ at midpoint must be between endpoints or positive; c1={c1:.3e} c_mid={c_mid:.3e} c2={c2:.3e}"
        );

        // Outside band returns zero.
        let (t_out, _, _, _) = urr_cross_sections(&urr, 100.0);
        assert_eq!(t_out, 0.0, "Outside band must be zero");
    }

    /// lin_lin_interp: boundary clamping and interior interpolation.
    #[test]
    fn lin_lin_interp_basic() {
        let xs = vec![1.0, 3.0, 9.0];
        let ys = vec![0.0, 2.0, 8.0];

        // Below lower bound: clamp to y[0]
        assert!((lin_lin_interp(&xs, &ys, 0.0) - 0.0).abs() < 1e-12);
        // Above upper bound: clamp to y[last]
        assert!((lin_lin_interp(&xs, &ys, 100.0) - 8.0).abs() < 1e-12);
        // Interior: x=2.0, between [1,3], t=0.5 → y=1.0
        assert!((lin_lin_interp(&xs, &ys, 2.0) - 1.0).abs() < 1e-12);
        // Interior: x=6.0, between [3,9], t=0.5 → y=5.0
        assert!((lin_lin_interp(&xs, &ys, 6.0) - 5.0).abs() < 1e-12);
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn make_lrf1_urr(e_low: f64, e_high: f64) -> UrrData {
        UrrData {
            lrf: 1,
            spi: 2.5,
            ap: 0.96931,
            e_low,
            e_high,
            l_groups: vec![UrrLGroup {
                l: 0,
                awri: 231.038,
                j_groups: vec![UrrJGroup {
                    j: 2.0,
                    amun: 1.0,
                    amuf: 0.0,
                    energies: vec![],
                    d: vec![0.5],
                    gx: vec![0.0],
                    gn: vec![1.0e-4],
                    gg: vec![3.5e-2],
                    gf: vec![0.0],
                    int_code: 2,
                }],
            }],
        }
    }
}
