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
//! LRF=2: Γ_n(E) from tabulated energy grid using INT interpolation
//!         INT=1 histogram, INT=2 lin-lin, INT=3 log-lin,
//!         INT=4 lin-log, INT=5 log-log (all 5 ENDF codes supported)
//!
//! Γ_tot = Γ_n + GG + GF + GX
//!
//! σ_γ  += (π/k²) · g_J · (2π · Γ_n · GG) / (D · Γ_tot)
//! σ_f  += (π/k²) · g_J · (2π · Γ_n · GF) / (D · Γ_tot)
//! σ_cn += (π/k²) · g_J · (2π · Γ_n · Γ_n) / (D · Γ_tot)    [neutron-out]
//! ```
//!
//! `σ_cn` is the compound-elastic contribution; the competitive (GX) channel
//! contributes to `total` but not `elastic`.  Potential scattering
//! `σ_pot = 4π·AP²/100` (AP in fm, result in barns) is included in the
//! returned `total` and `elastic` so that the URR band produces a physically
//! consistent cross-section without requiring special handling at the call site.
//!
//! ## Units
//! All energies in eV, all lengths (AP, channel radii) in fm (true physics
//! femtometers, 10⁻¹⁵ m), cross-sections in barns.
//!
//! ENDF stores radii in 10⁻¹² cm (= 10 fm); the parser converts to fm at
//! parse time by multiplying by 10 (see `ENDF_RADIUS_TO_FM` in `parser.rs`),
//! matching SAMMY's `FillSammyRmatrixFromRMat.cpp` line 422.
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
/// `total`   = elastic + capture + fission + competitive (GX).
/// Potential scattering `σ_pot = 4π·AP²/100` (AP in fm, result in barns)
/// is folded in here rather than at the call site.
/// SAMMY ref: `unr/munr03.f90` includes the hard-sphere background.
///
/// ## SAMMY Reference
/// `unr/munr03.f90` Csig3 — Hauser-Feshbach cross-section kernel.
///
/// # Arguments
/// * `urr` — Parsed URR parameters (LRF=1 or LRF=2).
/// * `e_ev` — Neutron lab-frame energy in eV.
/// * `ap_fm` — Scattering radius in fm at this energy.  The caller is
///   responsible for evaluating any AP(E) table (NRO≠0) or falling back
///   to the constant `urr.ap`.
pub fn urr_cross_sections(urr: &UrrData, e_ev: f64, ap_fm: f64) -> (f64, f64, f64, f64) {
    if e_ev < urr.e_low || e_ev > urr.e_high {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut sig_cap = 0.0_f64;
    let mut sig_fiss = 0.0_f64;
    let mut sig_compound_n = 0.0_f64;
    let mut sig_competitive = 0.0_f64;

    for lg in &urr.l_groups {
        let awri = lg.awri;
        let l = lg.l;

        // Guard: non-positive AWRI makes k² = 0 → π/k² = inf.
        // SAMMY skips URR entirely so never hits this; we must guard explicitly.
        if awri <= 0.0 {
            continue;
        }

        // π/k² in barns and channel parameter ρ = k·AP.
        // Uses the same channel formulas as the RRR calculator for consistency.
        let pi_over_k2 = channel::pi_over_k_squared_barns(e_ev, awri);
        let rho = channel::rho(e_ev, awri, ap_fm);

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
                // ENDF-6 §0.5: INT=1 histogram, 2 lin-lin, 3 log-lin,
                // 4 lin-log, 5 log-log.
                let interp: fn(&[f64], &[f64], f64) -> f64 = match jg.int_code {
                    1 => histogram_interp,
                    3 => log_lin_interp,
                    4 => lin_log_interp,
                    5 => log_log_interp,
                    _ => lin_lin_interp, // INT=2 and any unknown → lin-lin
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
            // Compound-elastic contribution: neutron absorbed into the compound
            // nucleus and re-emitted as a neutron.  The probability of re-emission
            // into the neutron channel is Γ_n / Γ_tot, giving:
            //   σ_cn = (π/k²) · g_J · (2π·Γ_n/D) · Γ_n / Γ_tot
            // SAMMY ref: unr/munr03.f90 — `Sigxxx` uses Gn for both transmission
            // coefficients in the neutron elastic formula.
            sig_compound_n += prefactor * gn_eff / gamma_tot;
            // Competitive (inelastic) channel: GX is included in Γ_tot but goes
            // into neither capture nor fission.  It contributes to the total
            // cross section without appearing in elastic, capture, or fission.
            // SAMMY ref: unr/munr03.f90 — GX field in Gamma_total.
            sig_competitive += prefactor * gx_eff / gamma_tot;
        }
    }

    // Smooth potential scattering: σ_pot = 4π·AP²
    // AP is in fm; 1 barn = 100 fm².
    // SAMMY ref: unr/munr03.f90 adds hard-sphere background to elastic.
    let sig_pot = 4.0 * std::f64::consts::PI * ap_fm * ap_fm / 100.0;
    let elastic = sig_compound_n + sig_pot;
    let total = elastic + sig_cap + sig_fiss + sig_competitive;
    (total, elastic, sig_cap, sig_fiss)
}

/// Shared binary-search kernel for the two interpolation modes.
///
/// Handles empty-table, lower-clamp, and upper-clamp, then locates the
/// bracketing interval and calls `interp_fn(x, x0, y0, x1, y1)`.
/// `xs` must be strictly ascending and the same length as `ys`.
#[inline]
fn table_interp(
    xs: &[f64],
    ys: &[f64],
    x: f64,
    interp_fn: impl Fn(f64, f64, f64, f64, f64) -> f64,
) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert!(
        !xs.is_empty(),
        "URR interpolation table must not be empty (parser bug?)"
    );
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
    interp_fn(x, xs[i - 1], ys[i - 1], xs[i], ys[i])
}

/// Linear-linear interpolation (INT=2, clamped to table endpoints).
fn lin_lin_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    table_interp(xs, ys, x, |x, x0, y0, _x1, y1| {
        let dx = _x1 - x0;
        if dx.abs() < f64::EPSILON {
            return y0;
        }
        y0 + (x - x0) / dx * (y1 - y0)
    })
}

/// Histogram interpolation (INT=1, clamped to table endpoints).
///
/// Returns the y-value at the left endpoint of each interval (step function).
/// ENDF-6 §0.5: "y = y_i for x_i ≤ x < x_{i+1}".
fn histogram_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    table_interp(xs, ys, x, |_x, _x0, y0, _x1, _y1| y0)
}

/// Log-x / Linear-y interpolation (INT=3, clamped to table endpoints).
///
/// y(x) = y₀ + [ln(x) - ln(x₀)] / [ln(x₁) - ln(x₀)] × (y₁ - y₀)
/// Falls back to lin-lin when any x ≤ 0 (log undefined).
fn log_lin_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    table_interp(xs, ys, x, |x, x0, y0, x1, y1| {
        if x <= 0.0 || x0 <= 0.0 || x1 <= 0.0 {
            let dx = x1 - x0;
            return if dx.abs() < f64::EPSILON {
                y0
            } else {
                y0 + (x - x0) / dx * (y1 - y0)
            };
        }
        let denom = x1.ln() - x0.ln();
        if denom.abs() < f64::EPSILON {
            return y0;
        }
        let t = (x.ln() - x0.ln()) / denom;
        y0 + t * (y1 - y0)
    })
}

/// Linear-x / Log-y interpolation (INT=4, clamped to table endpoints).
///
/// y(x) = y₀ × (y₁/y₀)^[(x - x₀) / (x₁ - x₀)]
/// Falls back to lin-lin when any y ≤ 0 (log undefined).
fn lin_log_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    table_interp(xs, ys, x, |x, x0, y0, x1, y1| {
        let dx = x1 - x0;
        if dx.abs() < f64::EPSILON {
            return y0;
        }
        if y0 <= 0.0 || y1 <= 0.0 {
            return y0 + (x - x0) / dx * (y1 - y0);
        }
        let t = (x - x0) / dx;
        (y0.ln() + t * (y1.ln() - y0.ln())).exp()
    })
}

/// Log-log interpolation (INT=5, clamped to table endpoints).
///
/// Used for LRF=2 width tables in ENDF (common for heavy actinides such as
/// U-238 where widths follow an approximate power-law energy dependence).
/// Falls back to lin-lin when any bracket value is ≤ 0 (log undefined) or
/// when the log-space denominator is too small for safe division.
fn log_log_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    table_interp(xs, ys, x, |x, x0, y0, x1, y1| {
        let dx = x1 - x0;
        // Guard against degenerate x-intervals (duplicate energy points).
        if dx.abs() < f64::EPSILON {
            return y0;
        }
        // Guard against non-positive values (log undefined).
        if x <= 0.0 || x0 <= 0.0 || x1 <= 0.0 || y0 <= 0.0 || y1 <= 0.0 {
            // Fall back to lin-lin for degenerate (non-positive) bracket entries.
            return y0 + (x - x0) / dx * (y1 - y0);
        }
        let denom_ln = x1.ln() - x0.ln();
        // Guard against nearly equal log values to avoid numerical blow-up.
        if denom_ln.abs() < f64::EPSILON {
            return y0;
        }
        let log_t = (x.ln() - x0.ln()) / denom_ln;
        (y0.ln() + log_t * (y1.ln() - y0.ln())).exp()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::{UrrData, UrrJGroup, UrrLGroup};

    /// σ = 0 when energy is outside the URR band.
    #[test]
    fn urr_outside_band_returns_zero() {
        let urr = make_lrf1_urr(1000.0, 30_000.0);
        let (t, e, c, f) = urr_cross_sections(&urr, 500.0, urr.ap); // below band
        assert_eq!(t, 0.0);
        assert_eq!(e, 0.0);
        assert_eq!(c, 0.0);
        assert_eq!(f, 0.0);

        let (t2, _, _, _) = urr_cross_sections(&urr, 50_000.0, urr.ap); // above band
        assert_eq!(t2, 0.0);
    }

    /// LRF=1 formula check: σ_γ = (π/k²) · g_J · (2π · Γ_n · GG) / (D · Γ_tot).
    ///
    /// Uses a single (L=0, J=2.0) group with known parameters; verifies
    /// σ_γ matches the hand-computed value within numerical precision (~1e-10 relative).
    #[test]
    fn urr_lrf1_formula_check() {
        // Hand-crafted parameters similar to U-233 URR (order of magnitude).
        let e_test = 1_000.0_f64; // eV, mid-URR
        let awri = 231.038_f64;
        let ap = 9.6931_f64; // scattering radius in fm (ENDF 0.96931 × 10)
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

        let (total, elastic, capture, fission) = urr_cross_sections(&urr, e_test, ap);

        // Hand-compute expected values.
        let pi_over_k2 = channel::pi_over_k_squared_barns(e_test, awri);
        let rho = channel::rho(e_test, awri, ap);
        let p_l = penetrability::penetrability(0, rho); // L=0
        let gn_eff = 2.0 * p_l * gno;
        let gamma_tot = gn_eff + gg;
        let g_j = channel::statistical_weight(j, spi);
        let prefactor = pi_over_k2 * g_j * (2.0 * std::f64::consts::PI * gn_eff) / d;
        let expected_cap = prefactor * gg / gamma_tot;
        let expected_cn = prefactor * gn_eff / gamma_tot; // neutron-out term
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
            ap: 9.6931,
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

        let (_, _, c1, _) = urr_cross_sections(&urr, e1, urr.ap);
        let (_, _, c2, _) = urr_cross_sections(&urr, e2, urr.ap);
        let (_, _, c_mid, _) = urr_cross_sections(&urr, 5_000.0, urr.ap);

        assert!(c1 > 0.0, "σ_γ at {e1} eV must be positive, got {c1}");
        assert!(c2 > 0.0, "σ_γ at {e2} eV must be positive, got {c2}");
        // For a lin-lin table with two energy points, any interior energy
        // should produce a σ_γ between the endpoint values.
        let (c_lo, c_hi) = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
        assert!(
            c_mid >= c_lo && c_mid <= c_hi,
            "σ_γ at midpoint must lie between endpoint values; \
             c1={c1:.3e} c_mid={c_mid:.3e} c2={c2:.3e}"
        );

        // Outside band returns zero.
        let (t_out, _, _, _) = urr_cross_sections(&urr, 100.0, urr.ap);
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

    /// log_log_interp: boundary clamping, power-law interior, and lin-lin fallback.
    #[test]
    fn log_log_interp_basic() {
        // y = x^2: log-log should recover exact values.
        let xs = vec![1.0, 10.0, 100.0];
        let ys = vec![1.0, 100.0, 10_000.0]; // y = x^2

        // Clamp below lower bound
        assert!((log_log_interp(&xs, &ys, 0.5) - 1.0).abs() < 1e-12);
        // Clamp above upper bound
        assert!((log_log_interp(&xs, &ys, 200.0) - 10_000.0).abs() < 1e-12);
        // Interior: x=sqrt(10)≈3.162, between [1,10]; exact power-law gives y=10
        let x_mid = (1.0_f64 * 10.0_f64).sqrt(); // geometric mean → y = x_mid^2 = 10
        assert!((log_log_interp(&xs, &ys, x_mid) - 10.0).abs() < 1e-10);
        // At table points: exact values
        assert!((log_log_interp(&xs, &ys, 10.0) - 100.0).abs() < 1e-10);
    }

    /// log_log_interp: falls back to lin-lin when a bracket value is ≤ 0.
    #[test]
    fn log_log_interp_nonpositive_fallback() {
        // y[0] = 0 forces the lin-lin fallback in the first interval.
        let xs = vec![1.0, 3.0, 9.0];
        let ys = vec![0.0, 2.0, 8.0]; // y[0] = 0 → log undefined

        // x=2 is in [1,3]; fallback: t=0.5, y = 0 + 0.5*(2-0) = 1.0
        assert!((log_log_interp(&xs, &ys, 2.0) - 1.0).abs() < 1e-12);
        // x=6 is in [3,9]; both y>0, so log-log applies (not a straight power law here,
        // but result must be between 2 and 8)
        let v = log_log_interp(&xs, &ys, 6.0);
        assert!(
            v > 2.0 && v < 8.0,
            "log-log interior must be between endpoints: {v}"
        );
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn make_lrf1_urr(e_low: f64, e_high: f64) -> UrrData {
        UrrData {
            lrf: 1,
            spi: 2.5,
            ap: 9.6931,
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
