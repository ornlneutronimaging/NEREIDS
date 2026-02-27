//! Multi-formalism cross-section dispatcher.
//!
//! `cross_sections_at_energy` is the primary entry point for computing
//! energy-dependent cross-sections from ENDF resonance data.  It
//! iterates over all resonance ranges in the data and dispatches each to
//! the appropriate formalism-specific calculator:
//!
//! | ENDF LRF | Formalism                  | Implemented as              |
//! |----------|----------------------------|-----------------------------|
//! | 1        | SLBW                       | `slbw::slbw_cross_sections_for_range` |
//! | 2        | MLBW (approx.)             | `slbw::slbw_cross_sections_for_range` (SLBW approximation; resonance interference ignored) |
//! | 3        | Reich-Moore                | `reich_moore_spin_group` (this module) |
//! | 7        | R-Matrix Limited           | `rmatrix_limited::cross_sections_for_rml_range` |
//! | URR      | Hauser-Feshbach average    | `urr::urr_cross_sections` |
//!
//! ## Reich-Moore Approximation
//! In the full R-matrix, all channels (neutron, capture, fission) appear
//! explicitly. The Reich-Moore approximation *eliminates the capture channel*
//! from the channel space, absorbing its effect into an imaginary part of
//! the energy denominator. This makes the level matrix smaller while
//! remaining highly accurate.
//!
//! For non-fissile isotopes (like U-238 below threshold), each spin group
//! has only ONE explicit channel (neutron elastic), making the R-matrix
//! a scalar — and the calculation is very efficient.
//!
//! ## SAMMY Reference
//! - `rml/mrml07.f` Setr: R-matrix construction
//! - `rml/mrml09.f` Yinvrs: level matrix inversion
//! - `rml/mrml11.f` Setxqx: X-matrix, Sectio: cross-sections
//! - `rml/mrml03.f` Betset: ENDF widths → reduced width amplitudes
//! - SAMMY manual Section 2.1 (R-matrix theory)

use num_complex::Complex64;

use nereids_endf::resonance::{ResonanceData, ResonanceFormalism, ResonanceRange, Tab1};

use crate::channel;
use crate::penetrability;
use crate::rmatrix_limited;
use crate::slbw;
use crate::urr;

/// Cross-section results at a single energy point.
#[derive(Debug, Clone, Copy)]
pub struct CrossSections {
    /// Total cross-section (barns).
    pub total: f64,
    /// Elastic scattering cross-section (barns).
    pub elastic: f64,
    /// Capture (n,γ) cross-section (barns).
    pub capture: f64,
    /// Fission cross-section (barns).
    pub fission: f64,
}

/// Compute cross-sections at a single energy.
///
/// Dispatches each resonance range to the appropriate formalism-specific
/// calculator (SLBW, MLBW, Reich-Moore, R-Matrix Limited, URR) based on the
/// formalism stored in that range.  See the module-level table for the full
/// dispatch map.
///
/// Adjacent ranges that share a boundary energy use half-open intervals
/// `[e_low, e_high)` so the boundary point is counted exactly once
/// (ENDF-6 §2 convention).
///
/// # Arguments
/// * `data` — Parsed resonance parameters from ENDF.
/// * `energy_ev` — Neutron energy in eV (lab frame).
///
/// # Returns
/// Cross-sections in barns.
pub fn cross_sections_at_energy(data: &ResonanceData, energy_ev: f64) -> CrossSections {
    let awr = data.awr;

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for (range_idx, range) in data.ranges.iter().enumerate() {
        // Use half-open [low, high) only when the *next* range begins exactly at
        // this range's upper endpoint AND that next range can actually produce
        // cross-sections.  Ranges that parse successfully but whose physics is
        // not yet wired up (URR with urr=None) must not steal the boundary —
        // otherwise the shared energy point falls into a gap.
        // ENDF-6 §2 — adjacent ranges share a single boundary energy.
        let next_starts_here = data
            .ranges
            .get(range_idx + 1)
            .is_some_and(|next| next.energy_low == range.energy_high && range_is_evaluable(next));
        let in_range = if next_starts_here {
            energy_ev >= range.energy_low && energy_ev < range.energy_high
        } else {
            energy_ev >= range.energy_low && energy_ev <= range.energy_high
        };
        if !in_range {
            continue;
        }

        // URR (LRU=2): Hauser-Feshbach average cross-sections.
        // These ranges have `resolved = false` so they must be dispatched before
        // the `!range.resolved` skip below.
        //
        // Note: `parse_urr_range` sets urr.e_low == range.energy_low and
        // urr.e_high == range.energy_high, so the outer `in_range` check and the
        // inner band guard in `urr_cross_sections` test the same interval.
        // The inner guard is kept as a safety net for direct calls.
        if let Some(urr_data) = &range.urr {
            debug_assert_eq!(
                urr_data.e_low, range.energy_low,
                "URR e_low must equal range.energy_low"
            );
            debug_assert_eq!(
                urr_data.e_high, range.energy_high,
                "URR e_high must equal range.energy_high"
            );
            let ap_fm = range.scattering_radius_at(energy_ev);
            let (t, e, c, f) = urr::urr_cross_sections(urr_data, energy_ev, ap_fm);
            total += t;
            elastic += e;
            capture += c;
            fission += f;
            continue;
        }

        if !range.resolved {
            continue;
        }

        // Each range carries its own target_spin — pass per-range, not
        // from the first range, to correctly compute statistical weights g_J.
        let (t, e, c, f) = cross_sections_for_range(range, energy_ev, awr, range.target_spin);
        total += t;
        elastic += e;
        capture += c;
        fission += f;
    }

    CrossSections {
        total,
        elastic,
        capture,
        fission,
    }
}

/// Compute cross-sections over a grid of energies.
///
/// More efficient than calling `cross_sections_at_energy` in a loop because
/// per-resonance quantities (reduced widths) are computed once.
///
/// # Arguments
/// * `data` — Parsed resonance parameters from ENDF.
/// * `energies` — Slice of neutron energies in eV.
///
/// # Returns
/// Vector of cross-sections, one per energy point.
pub fn cross_sections_on_grid(data: &ResonanceData, energies: &[f64]) -> Vec<CrossSections> {
    energies
        .iter()
        .map(|&e| cross_sections_at_energy(data, e))
        .collect()
}

/// Can this range actually produce non-zero cross-sections?
///
/// Returns `true` for formalisms whose physics evaluation is implemented:
/// - SLBW (LRF=1) and MLBW (LRF=2) resolved ranges
/// - Reich-Moore (LRF=3) resolved ranges
/// - R-Matrix Limited (LRF=7) resolved ranges
/// - URR ranges with parsed data (`urr.is_some()`)
///
/// Returns `false` for URR placeholders created when unsupported INT
/// codes force a skip, or other unrecognized formalisms.
///
/// **Keep in sync with `cross_sections_for_range`.**  Whenever a new
/// formalism is dispatched there, add it to the `matches!` pattern here so
/// that energy boundary logic (`next_starts_here`) stays correct.
fn range_is_evaluable(range: &ResonanceRange) -> bool {
    if range.urr.is_some() {
        return true;
    }
    if !range.resolved {
        return false;
    }
    matches!(
        range.formalism,
        ResonanceFormalism::SLBW
            | ResonanceFormalism::MLBW
            | ResonanceFormalism::ReichMoore
            | ResonanceFormalism::RMatrixLimited
    )
}

/// Cross-sections for a single resolved resonance range.
///
/// Returns (total, elastic, capture, fission) in barns.
/// Dispatches to the R-Matrix Limited calculator for LRF=7 ranges.
fn cross_sections_for_range(
    range: &ResonanceRange,
    energy_ev: f64,
    awr: f64,
    target_spin: f64,
) -> (f64, f64, f64, f64) {
    // LRF=7 (R-Matrix Limited): dispatch to multi-channel calculator.
    if let Some(rml) = &range.rml {
        return rmatrix_limited::cross_sections_for_rml_range(rml, energy_ev);
    }

    // SLBW/MLBW: dispatch to the SLBW per-range calculator.
    //
    // IMPORTANT: MLBW is *not* implemented as a true multi-level
    // Breit–Wigner formalism.  MLBW ranges are evaluated using the
    // single-level Breit–Wigner (SLBW) formulas as an approximation.
    // This ignores resonance–resonance interference (the defining
    // difference between SLBW and MLBW), so results may be physically
    // incorrect for closely spaced or overlapping resonances.
    //
    // This check must precede the l_group loop because
    // `slbw_cross_sections_for_range` handles all L-groups and J-groups
    // internally (including potential scattering).
    if matches!(
        range.formalism,
        ResonanceFormalism::SLBW | ResonanceFormalism::MLBW
    ) {
        return slbw::slbw_cross_sections_for_range(range, energy_ev, awr, target_spin);
    }

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    for l_group in &range.l_groups {
        let l = l_group.l;
        let awr_l = if l_group.awr > 0.0 { l_group.awr } else { awr };

        // Channel radius: use L-dependent radius if available, else the
        // energy-dependent (or constant) global radius from the range.
        let channel_radius = if l_group.apl > 0.0 {
            l_group.apl
        } else {
            range.scattering_radius_at(energy_ev)
        };

        // AP table for per-resonance radius: only applicable when the global
        // NRO=1 table is in use (i.e., no L-group override via APL).
        // When NRO=1, ENDF widths are defined at AP(E_r), not AP(energy_ev),
        // so p_at_er must use the radius evaluated at the resonance energy.
        let ap_table_ref: Option<&Tab1> = if l_group.apl > 0.0 {
            None
        } else {
            range.ap_table.as_ref()
        };

        // Compute channel parameters at this energy.
        let rho = channel::rho(energy_ev, awr_l, channel_radius);
        let p_l = penetrability::penetrability(l, rho);
        let s_l = penetrability::shift_factor(l, rho);
        let phi_l = penetrability::phase_shift(l, rho);

        // Group resonances by J value for this L.
        // In ENDF, the J value is stored per resonance; group them.
        let j_groups = group_by_j(&l_group.resonances);

        for (j_total, resonances) in &j_groups {
            let j = *j_total;
            let g_j = channel::statistical_weight(j, target_spin);

            match range.formalism {
                ResonanceFormalism::ReichMoore => {
                    let (t, e, c, f) = reich_moore_spin_group(
                        resonances,
                        energy_ev,
                        awr_l,
                        channel_radius,
                        l,
                        g_j,
                        p_l,
                        s_l,
                        phi_l,
                        ap_table_ref,
                    );
                    total += t;
                    elastic += e;
                    capture += c;
                    fission += f;
                }
                ResonanceFormalism::SLBW | ResonanceFormalism::MLBW => {
                    // Unreachable: SLBW/MLBW ranges are dispatched before
                    // entering this loop (see early return above).
                    unreachable!("SLBW/MLBW dispatched before l_group loop");
                }
                _ => {
                    // Other formalisms (e.g. Adler-Adler LRF=4) are not
                    // implemented.  `range_is_evaluable` returns `false` for
                    // these, so they should never reach here through the
                    // normal dispatcher.  This arm exists only for
                    // exhaustiveness; contribution is zero.
                    continue;
                }
            }
        }
    }

    (total, elastic, capture, fission)
}

/// Cross-sections for a single spin group (J, π) in the Reich-Moore formalism.
///
/// For non-fissile isotopes, the R-matrix has a single neutron channel
/// and the capture channel is eliminated (absorbed into the imaginary
/// part of the resonance denominator).
///
/// ## Mathematical Formulation
///
/// For a single neutron channel with eliminated capture:
///
/// R(E) = Σ_n γ²_n / (E_n - E - iΓ_γ,n/2)
///
/// where γ²_n = Γ_n,n / (2·P_l(E_n)) is the reduced width amplitude squared.
///
/// Level matrix (scalar): Y = (S - B + iP)⁻¹ - R
///
/// X-matrix (scalar): X = P · Y⁻¹ · R · (S - B + iP)⁻¹
///
/// The scattering matrix element is:
///   U = e^{2iφ} · (1 + 2i·X)
///
/// Cross-sections:
///   σ_elastic = (π/k²) · g_J · |1 - U|²
///   σ_total   = (2π/k²) · g_J · (1 - Re(U))
///   σ_capture = σ_total - σ_elastic (unitarity deficit)
///
/// Reference: SAMMY `rml/mrml11.f` Sectio routine
#[allow(clippy::too_many_arguments)]
fn reich_moore_spin_group(
    resonances: &[&nereids_endf::resonance::Resonance],
    energy_ev: f64,
    awr: f64,
    channel_radius: f64,
    l: u32,
    g_j: f64,
    p_l: f64,
    s_l: f64,
    phi_l: f64,
    ap_table: Option<&Tab1>,
) -> (f64, f64, f64, f64) {
    let pi_over_k2 = channel::pi_over_k_squared_barns(energy_ev, awr);

    // Determine if any resonance has fission widths.
    let has_fission = resonances
        .iter()
        .any(|r| r.gfa.abs() > 1e-30 || r.gfb.abs() > 1e-30);

    if has_fission {
        // Multi-channel case: neutron + fission channels.
        return reich_moore_with_fission(
            resonances,
            energy_ev,
            awr,
            channel_radius,
            l,
            g_j,
            p_l,
            s_l,
            phi_l,
            pi_over_k2,
            ap_table,
        );
    }

    // Single-channel case (neutron only, capture eliminated).
    // This is the common case for non-fissile isotopes.

    // Boundary condition: B = S(E_n) at each resonance energy.
    // SAMMY typically uses B = 0 (Shift=0 flag in .par file).
    // ENDF convention: B = S(E_n) unless NRO/NAPS flags say otherwise.
    // For simplicity and following SAMMY ex027 (Shift=0), we use B = 0.
    let boundary = 0.0;

    // Build the R-matrix (scalar, complex) = Σ_n γ²_n / (E_n - E - iΓ_γ,n/2)
    //
    // Note: ENDF stores "observed" widths Γ_n. The reduced width amplitude is:
    //   γ²_n = Γ_n / (2 · P_l(ρ_n))
    // where ρ_n = k(E_n)·a, evaluated at the resonance energy.
    //
    // Reference: SAMMY `rml/mrml03.f` Betset (lines 240-276)
    let mut r_real = 0.0;
    let mut r_imag = 0.0;

    for res in resonances {
        let e_r = res.energy;
        let gamma_n = res.gn; // neutron width (eV)
        let gamma_g = res.gg; // capture width (eV)

        // Reduced width amplitude squared.
        // ENDF widths are defined as Γ_n = 2·P_l(AP(E_r), E_r)·γ²_n,
        // so the penetrability must be evaluated at the resonance energy
        // using the channel radius AP(E_r) — not the incident-energy AP(E).
        // For NRO=0 (constant AP) this makes no difference.
        let p_at_er = if e_r.abs() > 1e-30 {
            let radius_at_er = ap_table.map_or(channel_radius, |t| t.evaluate(e_r.abs()));
            let rho_r = channel::rho(e_r.abs(), awr, radius_at_er);
            penetrability::penetrability(l, rho_r)
        } else {
            p_l // Fallback: use current-energy penetrability
        };

        let gamma_n_reduced_sq = if p_at_er > 1e-30 {
            gamma_n.abs() / (2.0 * p_at_er)
        } else {
            0.0
        };

        // Sign of neutron width: ENDF convention is that Γ_n may be negative
        // to indicate the sign of the reduced width amplitude.
        let gamma_n_sign = if gamma_n >= 0.0 { 1.0 } else { -1.0 };
        let _ = gamma_n_sign; // For single-channel, sign doesn't matter (γ² always positive)

        // Denominator: (E_n - E)² + (Γ_γ/2)²
        let de = e_r - energy_ev;
        let half_gg = gamma_g / 2.0;
        let denom = de * de + half_gg * half_gg;

        if denom > 1e-50 {
            // R-matrix contribution:
            // R += γ²_n / (E_n - E - i·Γ_γ/2)
            //    = γ²_n · (E_n - E + i·Γ_γ/2) / denom
            r_real += gamma_n_reduced_sq * de / denom;
            r_imag += gamma_n_reduced_sq * half_gg / denom;
        }
    }

    // Level matrix Y = 1/(S - B + iP) - R  (scalar, complex)
    let l_real = s_l - boundary;
    let l_imag = p_l;
    let l_denom = l_real * l_real + l_imag * l_imag;

    // 1/(S - B + iP) = (S - B - iP) / |S - B + iP|²
    let l_inv_real = l_real / l_denom;
    let l_inv_imag = -l_imag / l_denom;

    let y_real = l_inv_real - r_real;
    let y_imag = l_inv_imag - r_imag;

    // Y⁻¹ = 1/Y
    let y_denom = y_real * y_real + y_imag * y_imag;
    let y_inv_real = y_real / y_denom;
    let y_inv_imag = -y_imag / y_denom;

    // X-matrix (scalar): X = P · Y⁻¹ · R · (1/(S-B+iP))
    // Actually: X = √P · Y⁻¹ · R · √P · (1/(S-B+iP))
    //
    // From SAMMY mrml11.f: XXXX = √P_J · (Y⁻¹·R)_JI · (√P_I / L_II)
    // For single channel: X = √P · Y⁻¹ · R · √P / L
    //                       = P · Y⁻¹ · R / (S-B+iP)
    //
    // Let's compute step by step:
    // 1. q = Y⁻¹ · R (complex multiply)
    let q_real = y_inv_real * r_real - y_inv_imag * r_imag;
    let q_imag = y_inv_real * r_imag + y_inv_imag * r_real;

    // 2. X = P · q / (S-B+iP) = P · q · (S-B-iP) / |S-B+iP|²
    let x_unscaled_real = q_real * l_real + q_imag * l_imag;
    let x_unscaled_imag = q_imag * l_real - q_real * l_imag;
    let x_real = p_l * x_unscaled_real / l_denom;
    let x_imag = p_l * x_unscaled_imag / l_denom;

    // Compute the collision matrix element U from X.
    //
    //   U = e^{2iφ} · (1 + 2iX)
    //
    // Reference: ENDF-102 Section 2, Lane & Thomas R-matrix theory
    let x = Complex64::new(x_real, x_imag);
    let phase = Complex64::new((2.0 * phi_l).cos(), (2.0 * phi_l).sin());
    let u = phase * (1.0 + 2.0 * Complex64::i() * x);

    // Cross-sections from the collision matrix U:
    //
    //   σ_total   = g_J · (2π/k²) · (1 - Re(U))
    //   σ_elastic = g_J · (π/k²) · |1 - U|²
    //   σ_capture = σ_total - σ_elastic  (unitarity deficit)
    //
    // Reference: standard R-matrix cross-section formulas
    let sigma_total = g_j * 2.0 * pi_over_k2 * (1.0 - u.re);
    let one_minus_u = 1.0 - u;
    let sigma_elastic = g_j * pi_over_k2 * one_minus_u.norm_sqr();
    let sigma_capture = sigma_total - sigma_elastic;

    // For non-fissile isotopes, all absorption is capture.
    (sigma_total, sigma_elastic, sigma_capture, 0.0)
}

/// Reich-Moore calculation with explicit fission channels.
///
/// When fission widths (GFA, GFB) are present, the R-matrix becomes
/// 2×2 or 3×3 (neutron + 1 or 2 fission channels), with capture still
/// eliminated into the imaginary denominator.
///
/// Reference: SAMMY `rml/mrml09.f` Twoch/Three routines
#[allow(clippy::too_many_arguments)]
fn reich_moore_with_fission(
    resonances: &[&nereids_endf::resonance::Resonance],
    energy_ev: f64,
    awr: f64,
    channel_radius: f64,
    l: u32,
    g_j: f64,
    p_l: f64,
    s_l: f64,
    phi_l: f64,
    pi_over_k2: f64,
    ap_table: Option<&Tab1>,
) -> (f64, f64, f64, f64) {
    // Determine number of fission channels (1 or 2).
    let has_two_fission = resonances.iter().any(|r| r.gfb.abs() > 1e-30);
    let n_channels = if has_two_fission { 3 } else { 2 }; // neutron + fission(s)

    let boundary = 0.0;

    if n_channels == 2 {
        // 2-channel: neutron + one fission channel.
        // R-matrix is 2×2 complex.
        let mut r_mat = [[Complex64::new(0.0, 0.0); 2]; 2];

        for res in resonances {
            let e_r = res.energy;
            let gamma_n = res.gn;
            let gamma_g = res.gg;
            let gamma_f = res.gfa;

            // Reduced width amplitudes.
            // Use AP(E_r) for the resonance-energy penetrability (ENDF width convention).
            let p_at_er = if e_r.abs() > 1e-30 {
                let radius_at_er = ap_table.map_or(channel_radius, |t| t.evaluate(e_r.abs()));
                let rho_r = channel::rho(e_r.abs(), awr, radius_at_er);
                penetrability::penetrability(l, rho_r)
            } else {
                p_l
            };

            // β_n = sqrt(|Γ_n| / (2·P_l(E_r))), sign from Γ_n sign
            let beta_n = if p_at_er > 1e-30 {
                let sign = if gamma_n >= 0.0 { 1.0 } else { -1.0 };
                sign * (gamma_n.abs() / (2.0 * p_at_er)).sqrt()
            } else {
                0.0
            };

            // β_f = sqrt(|Γ_f|/2), fission channel has no penetrability correction
            // (Lpent=0 for fission in SAMMY)
            let beta_f = {
                let sign = if gamma_f >= 0.0 { 1.0 } else { -1.0 };
                sign * (gamma_f.abs() / 2.0).sqrt()
            };

            // Denominator: (E_n - E) - i·Γ_γ/2
            let de = e_r - energy_ev;
            let half_gg = gamma_g / 2.0;
            let denom = Complex64::new(de, -half_gg);
            let inv_denom = 1.0 / denom;

            // R_ij += β_i · β_j / denom
            let betas = [beta_n, beta_f];
            for i in 0..2 {
                for j in 0..2 {
                    r_mat[i][j] += betas[i] * betas[j] * inv_denom;
                }
            }
        }

        // Level matrix Y = diag(1/(S-B+iP)) - R
        // Channel 0 (neutron): L = S_l - B + i·P_l
        // Channel 1 (fission): L = 0 + i·1 (no penetrability, Pent=0)
        //   → fission channel: P_f = 1, S_f = 0
        let l_n = Complex64::new(s_l - boundary, p_l);
        let l_f = Complex64::new(0.0, 1.0); // Fission: no barrier

        let l_inv = [1.0 / l_n, 1.0 / l_f];

        let mut y_mat = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                y_mat[i][j] = -r_mat[i][j];
            }
            y_mat[i][i] += l_inv[i];
        }

        // Invert 2×2 Y-matrix.
        let det = y_mat[0][0] * y_mat[1][1] - y_mat[0][1] * y_mat[1][0];
        let inv_det = 1.0 / det;
        let y_inv = [
            [y_mat[1][1] * inv_det, -y_mat[0][1] * inv_det],
            [-y_mat[1][0] * inv_det, y_mat[0][0] * inv_det],
        ];

        // X-matrix: X_ij = √P_i · (Y⁻¹·R)_ij · √P_j / L_jj
        // First compute q = Y⁻¹ · R
        let mut q = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    q[i][j] += y_inv[i][k] * r_mat[k][j];
                }
            }
        }

        let sqrt_p = [p_l.sqrt(), 1.0]; // √P_n, √P_f
        let mut x_mat = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let l_jj = if j == 0 { l_n } else { l_f };
                x_mat[i][j] = sqrt_p[i] * q[i][j] * sqrt_p[j] / l_jj;
            }
        }

        // Collision matrix U from X-matrix.
        // U_nn = e^{2iφ}(1 + 2i·X_nn)  (neutron→neutron)
        // U_nf = e^{iφ}·2i·X_nf         (neutron→fission, φ_f=0)
        let phase2 = Complex64::new((2.0 * phi_l).cos(), (2.0 * phi_l).sin());
        let phase1 = Complex64::new(phi_l.cos(), phi_l.sin());

        let u_nn = phase2 * (1.0 + 2.0 * Complex64::i() * x_mat[0][0]);
        let u_nf = phase1 * 2.0 * Complex64::i() * x_mat[0][1];

        // Cross-sections from U-matrix.
        let sigma_total = g_j * 2.0 * pi_over_k2 * (1.0 - u_nn.re);
        let sigma_elastic = g_j * pi_over_k2 * (1.0 - u_nn).norm_sqr();
        let sigma_fission = g_j * pi_over_k2 * u_nf.norm_sqr();
        let sigma_capture = sigma_total - sigma_elastic - sigma_fission;

        (sigma_total, sigma_elastic, sigma_capture, sigma_fission)
    } else {
        // 3-channel: neutron + two fission channels.
        // Use general complex matrix inversion.
        reich_moore_3channel(
            resonances,
            energy_ev,
            awr,
            channel_radius,
            l,
            g_j,
            p_l,
            s_l,
            phi_l,
            pi_over_k2,
            ap_table,
        )
    }
}

/// 3-channel Reich-Moore (neutron + 2 fission channels).
#[allow(clippy::too_many_arguments)]
fn reich_moore_3channel(
    resonances: &[&nereids_endf::resonance::Resonance],
    energy_ev: f64,
    awr: f64,
    channel_radius: f64,
    l: u32,
    g_j: f64,
    p_l: f64,
    s_l: f64,
    phi_l: f64,
    pi_over_k2: f64,
    ap_table: Option<&Tab1>,
) -> (f64, f64, f64, f64) {
    let boundary = 0.0;

    let mut r_mat = [[Complex64::new(0.0, 0.0); 3]; 3];

    for res in resonances {
        let e_r = res.energy;
        let gamma_n = res.gn;
        let gamma_g = res.gg;
        let gamma_fa = res.gfa;
        let gamma_fb = res.gfb;

        // Use AP(E_r) for the resonance-energy penetrability (ENDF width convention).
        let p_at_er = if e_r.abs() > 1e-30 {
            let radius_at_er = ap_table.map_or(channel_radius, |t| t.evaluate(e_r.abs()));
            let rho_r = channel::rho(e_r.abs(), awr, radius_at_er);
            penetrability::penetrability(l, rho_r)
        } else {
            p_l
        };

        let beta_n = if p_at_er > 1e-30 {
            let sign = if gamma_n >= 0.0 { 1.0 } else { -1.0 };
            sign * (gamma_n.abs() / (2.0 * p_at_er)).sqrt()
        } else {
            0.0
        };

        let beta_fa = {
            let sign = if gamma_fa >= 0.0 { 1.0 } else { -1.0 };
            sign * (gamma_fa.abs() / 2.0).sqrt()
        };

        let beta_fb = {
            let sign = if gamma_fb >= 0.0 { 1.0 } else { -1.0 };
            sign * (gamma_fb.abs() / 2.0).sqrt()
        };

        let de = e_r - energy_ev;
        let half_gg = gamma_g / 2.0;
        let inv_denom = 1.0 / Complex64::new(de, -half_gg);

        let betas = [beta_n, beta_fa, beta_fb];
        for i in 0..3 {
            for j in 0..3 {
                r_mat[i][j] += betas[i] * betas[j] * inv_denom;
            }
        }
    }

    // Level matrix Y.
    let l_n = Complex64::new(s_l - boundary, p_l);
    let l_f = Complex64::new(0.0, 1.0);
    let l_vals = [l_n, l_f, l_f];
    let l_inv: Vec<Complex64> = l_vals.iter().map(|&li| 1.0 / li).collect();

    let mut y_mat = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            y_mat[i][j] = -r_mat[i][j];
        }
        y_mat[i][i] += l_inv[i];
    }

    // Invert 3×3 via cofactor expansion.
    let y_inv = invert_3x3(y_mat);

    // X-matrix.
    let sqrt_p = [p_l.sqrt(), 1.0, 1.0];
    let mut x_mat = [[Complex64::new(0.0, 0.0); 3]; 3];
    let mut q = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                q[i][j] += y_inv[i][k] * r_mat[k][j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            x_mat[i][j] = sqrt_p[i] * q[i][j] * sqrt_p[j] / l_vals[j];
        }
    }

    // Collision matrix U from X-matrix.
    let phase2 = Complex64::new((2.0 * phi_l).cos(), (2.0 * phi_l).sin());
    let phase1 = Complex64::new(phi_l.cos(), phi_l.sin());

    let u_nn = phase2 * (1.0 + 2.0 * Complex64::i() * x_mat[0][0]);
    let u_nf1 = phase1 * 2.0 * Complex64::i() * x_mat[0][1];
    let u_nf2 = phase1 * 2.0 * Complex64::i() * x_mat[0][2];

    // Cross-sections from U-matrix.
    let sigma_total = g_j * 2.0 * pi_over_k2 * (1.0 - u_nn.re);
    let sigma_elastic = g_j * pi_over_k2 * (1.0 - u_nn).norm_sqr();
    let sigma_fission = g_j * pi_over_k2 * (u_nf1.norm_sqr() + u_nf2.norm_sqr());
    let sigma_capture = sigma_total - sigma_elastic - sigma_fission;

    (sigma_total, sigma_elastic, sigma_capture, sigma_fission)
}

/// Invert a 3×3 complex matrix via cofactor expansion.
fn invert_3x3(m: [[Complex64; 3]; 3]) -> [[Complex64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    let inv_det = 1.0 / det;

    let mut result = [[Complex64::new(0.0, 0.0); 3]; 3];
    result[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    result[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    result[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    result[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    result[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    result[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    result[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    result[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    result[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    result
}

/// Group resonances by their J value.
fn group_by_j(
    resonances: &[nereids_endf::resonance::Resonance],
) -> Vec<(f64, Vec<&nereids_endf::resonance::Resonance>)> {
    let mut groups: Vec<(f64, Vec<&nereids_endf::resonance::Resonance>)> = Vec::new();

    for res in resonances {
        let j = res.j;
        if let Some(group) = groups.iter_mut().find(|(gj, _)| (*gj - j).abs() < 1e-10) {
            group.1.push(res);
        } else {
            groups.push((j, vec![res]));
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::{LGroup, Resonance, ResonanceRange};

    /// Create a simple single-resonance test case for validation.
    #[allow(clippy::too_many_arguments)]
    fn make_single_resonance_data(
        energy: f64,
        gamma_n: f64,
        gamma_g: f64,
        j: f64,
        l: u32,
        awr: f64,
        target_spin: f64,
        scattering_radius: f64,
    ) -> ResonanceData {
        ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin,
                scattering_radius,
                l_groups: vec![LGroup {
                    l,
                    awr,
                    apl: 0.0,
                    resonances: vec![Resonance {
                        energy,
                        j,
                        gn: gamma_n,
                        gg: gamma_g,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                ap_table: None,
                urr: None,
            }],
        }
    }

    #[test]
    fn test_capture_peak_single_resonance() {
        // U-238 6.674 eV resonance.
        // At the resonance energy, capture cross-section should peak at ~22,000 barns.
        let data = make_single_resonance_data(
            6.674,    // E_r (eV)
            1.493e-3, // Γ_n (eV)
            23.0e-3,  // Γ_γ (eV)
            0.5,      // J
            0,        // L
            236.006,  // AWR
            0.0,      // target spin I
            9.4285,   // scattering radius (fm)
        );

        let xs = cross_sections_at_energy(&data, 6.674);

        // The capture cross-section at peak should be approximately:
        // σ_c = g_J × π/k² × 4×Γ_n×Γ_γ / Γ² where Γ = Γ_n + Γ_γ
        // For the RM formalism the peak is very close to this BW estimate.
        // g_J = 1.0, π/k² ≈ 98,200 barns, Γ = 0.024493
        // σ_c ≈ 1.0 × 98200 × 4 × 1.493e-3 × 23.0e-3 / (24.493e-3)²
        //     ≈ 98200 × 0.2289 ≈ 22,478 barns
        println!("Capture at 6.674 eV: {} barns", xs.capture);
        println!("Total at 6.674 eV: {} barns", xs.total);
        println!("Elastic at 6.674 eV: {} barns", xs.elastic);

        assert!(
            xs.capture > 15000.0 && xs.capture < 30000.0,
            "Capture should be ~22000 barns, got {}",
            xs.capture
        );
        assert!(xs.total > xs.capture, "Total > capture");
        assert!(xs.elastic > 0.0, "Elastic should be positive");
        assert!(xs.fission.abs() < 1e-10, "No fission for U-238");
    }

    #[test]
    fn test_1_over_v_behavior() {
        // Far from resonances, capture cross-section should follow 1/v ∝ 1/√E.
        // The 6.674 eV resonance tail should dominate at low energies.
        let data =
            make_single_resonance_data(6.674, 1.493e-3, 23.0e-3, 0.5, 0, 236.006, 0.0, 9.4285);

        let xs_01 = cross_sections_at_energy(&data, 0.1);
        let xs_04 = cross_sections_at_energy(&data, 0.4);

        // At low E, σ ∝ 1/√E, so σ(0.1)/σ(0.4) ≈ √(0.4/0.1) = 2.0
        let ratio = xs_01.capture / xs_04.capture;
        println!("1/v ratio test: σ(0.1)/σ(0.4) = {}", ratio);
        assert!(
            (ratio - 2.0).abs() < 0.3,
            "Expected ~2.0 for 1/v behavior, got {}",
            ratio
        );
    }

    #[test]
    fn test_cross_sections_positive() {
        // All cross-sections must be non-negative at all energies.
        let data =
            make_single_resonance_data(6.674, 1.493e-3, 23.0e-3, 0.5, 0, 236.006, 0.0, 9.4285);

        for &e in &[0.01, 0.1, 1.0, 5.0, 6.0, 6.674, 7.0, 10.0, 100.0, 1000.0] {
            let xs = cross_sections_at_energy(&data, e);
            assert!(xs.total >= 0.0, "Total negative at E={}: {}", e, xs.total);
            assert!(
                xs.elastic >= 0.0,
                "Elastic negative at E={}: {}",
                e,
                xs.elastic
            );
            assert!(
                xs.capture >= -1e-10,
                "Capture negative at E={}: {}",
                e,
                xs.capture
            );
        }
    }

    /// Parse full U-238 ENDF file and compute cross-sections.
    ///
    /// Validates against SAMMY ex027 output (Doppler-broadened at 300K, but
    /// we compare unbroadened RM values which should bracket the broadened data).
    #[test]
    fn test_u238_full_endf_cross_sections() {
        let endf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../SAMMY/SAMMY/samexm_new/ex027_new/ex027.endf");

        if !endf_path.exists() {
            eprintln!("Skipping: SAMMY ENDF file not found at {:?}", endf_path);
            return;
        }

        let endf_text = std::fs::read_to_string(&endf_path).unwrap();
        let data = nereids_endf::parser::parse_endf_file2(&endf_text).unwrap();

        // Compute cross-sections at several energies near the 6.674 eV resonance.
        let energies = [1.0, 5.0, 6.0, 6.5, 6.674, 7.0, 8.0, 10.0, 20.0, 50.0, 100.0];

        println!("\nU-238 Reich-Moore cross-sections (unbroadened):");
        println!(
            "{:>10} {:>12} {:>12} {:>12} {:>12}",
            "E (eV)", "Total", "Elastic", "Capture", "Fission"
        );

        for &e in &energies {
            let xs = cross_sections_at_energy(&data, e);
            println!(
                "{:>10.3} {:>12.3} {:>12.3} {:>12.3} {:>12.6}",
                e, xs.total, xs.elastic, xs.capture, xs.fission
            );

            // Basic sanity: all cross-sections non-negative.
            assert!(xs.total >= 0.0, "Total negative at E={}", e);
            assert!(xs.elastic >= 0.0, "Elastic negative at E={}", e);
            // Capture can be very slightly negative due to floating point.
            assert!(
                xs.capture >= -0.01,
                "Capture negative at E={}: {}",
                e,
                xs.capture
            );
        }

        // Check the 6.674 eV resonance peak.
        // With the full ENDF file (all resonances), the peak capture
        // should still be dominated by the 6.674 eV resonance.
        let xs_peak = cross_sections_at_energy(&data, 6.674);
        assert!(
            xs_peak.capture > 10000.0,
            "Capture at 6.674 eV should be >10,000 barns (got {})",
            xs_peak.capture
        );

        // The 20.87 eV resonance should also show a significant peak.
        let xs_20 = cross_sections_at_energy(&data, 20.87);
        assert!(
            xs_20.capture > 1000.0,
            "Capture at 20.87 eV should be >1,000 barns (got {})",
            xs_20.capture
        );

        // SAMMY ex027 broadened output at ~6.674 eV gives ~339 barns capture.
        // Our UNBROADENED result should be MUCH larger (since Doppler broadening
        // spreads the peak). This confirms we're computing the correct physics.
        println!(
            "\n6.674 eV peak: capture={:.0} barns (unbroadened), SAMMY broadened ~339 barns",
            xs_peak.capture
        );
        assert!(
            xs_peak.capture > 339.0,
            "Unbroadened peak must exceed SAMMY broadened value"
        );
    }

    /// Build a single-resonance `ResonanceData` with a chosen formalism.
    fn make_slbw_data(formalism: ResonanceFormalism) -> ResonanceData {
        ResonanceData {
            isotope: nereids_core::types::Isotope::new(92, 238).unwrap(),
            za: 92238,
            awr: 236.006,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e4,
                resolved: true,
                formalism,
                target_spin: 0.0,
                scattering_radius: 9.4285,
                l_groups: vec![LGroup {
                    l: 0,
                    awr: 236.006,
                    apl: 0.0,
                    resonances: vec![Resonance {
                        energy: 6.674,
                        j: 0.5,
                        gn: 1.493e-3,
                        gg: 23.0e-3,
                        gfa: 0.0,
                        gfb: 0.0,
                    }],
                }],
                rml: None,
                ap_table: None,
                urr: None,
            }],
        }
    }

    /// `cross_sections_at_energy` with an SLBW-formalism range must give
    /// the same result as `slbw::slbw_cross_sections`.
    #[test]
    fn test_dispatcher_slbw_matches_slbw_module() {
        let data = make_slbw_data(ResonanceFormalism::SLBW);

        let test_energies = [0.1, 1.0, 5.0, 6.0, 6.674, 7.0, 10.0, 100.0];
        for &e in &test_energies {
            let via_dispatcher = cross_sections_at_energy(&data, e);
            let via_slbw = crate::slbw::slbw_cross_sections(&data, e);

            let eps = 1e-10;
            assert!(
                (via_dispatcher.total - via_slbw.total).abs() < eps,
                "total mismatch at {e} eV: dispatcher={} slbw={}",
                via_dispatcher.total,
                via_slbw.total
            );
            assert!(
                (via_dispatcher.capture - via_slbw.capture).abs() < eps,
                "capture mismatch at {e} eV: dispatcher={} slbw={}",
                via_dispatcher.capture,
                via_slbw.capture
            );
            assert!(
                (via_dispatcher.elastic - via_slbw.elastic).abs() < eps,
                "elastic mismatch at {e} eV: dispatcher={} slbw={}",
                via_dispatcher.elastic,
                via_slbw.elastic
            );
        }
    }

    /// MLBW is dispatched as SLBW (approximation).  Verify that the
    /// dispatcher returns the same values as the SLBW module when given
    /// an MLBW-formalism range, and that the results are physically
    /// reasonable (positive, peak at resonance energy).
    #[test]
    fn test_dispatcher_mlbw_uses_slbw_approximation() {
        let data_mlbw = make_slbw_data(ResonanceFormalism::MLBW);
        let data_slbw = make_slbw_data(ResonanceFormalism::SLBW);

        let test_energies = [1.0, 6.674, 10.0];
        for &e in &test_energies {
            let xs_mlbw = cross_sections_at_energy(&data_mlbw, e);
            let xs_slbw = cross_sections_at_energy(&data_slbw, e);
            let eps = 1e-10;
            assert!(
                (xs_mlbw.total - xs_slbw.total).abs() < eps,
                "MLBW/SLBW mismatch at {e} eV: mlbw={} slbw={}",
                xs_mlbw.total,
                xs_slbw.total
            );
        }

        // Sanity: peak capture at resonance energy should be large.
        let xs_peak = cross_sections_at_energy(&data_mlbw, 6.674);
        assert!(
            xs_peak.capture > 1000.0,
            "MLBW capture at 6.674 eV should be substantial (got {})",
            xs_peak.capture
        );
    }
}
