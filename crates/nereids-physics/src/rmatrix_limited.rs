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
//! Collision matrix (eq. III.D.4 in SAMMY-manual-R3 numbering, as cited by
//! `rml/mrml11.f`; R8: W is Eqs. (II B1.3)-(II B1.4), and U = Ω·W·Ω′ is
//! Eq. (II A.4) with Ω from Eq. (II A.5)):
//!   W_cc' = δ_cc' + 2i·Ξ_cc'
//!   U_cc' = Ω_c · W_cc' · Ω_c'    where Ω_c = exp(-iφ_c)
//!
//!   TRUTH SOURCE: SAMMY rml/mrml11.f lines 14-18 (W = I + 2i·XXXX)
//!   and lines 84-88 (elastic formula consistent with e^{-2iφ}).
//!   Unitarity: |U| ≤ 1 always; hard sphere (R=0) → U = exp(2iφ)·I  ✓
//!
//! Cross sections per spin group (J,π), summed over entrance neutron channels c0:
//!   σ_total   = Σ_{c0} 2·(π/k²)·g_J·(1 - Re(U_{c0,c0}))
//!   σ_elastic = Σ_{c0} (π/k²)·g_J·|1 - U_{c0,c0}|²
//!   σ_fission = Σ_{c0} Σ_{c'∈fission} (π/k²)·g_J·|U_{c0,c'}|²
//!   σ_capture = σ_total - σ_elastic - σ_fission
//! ```
//!
//! ## No-penetrability channels (PNT / Lpent = 0)
//!
//! The branch is keyed on the per-pair `PNT` (SAMMY `Lpent`) flag, NOT on
//! particle mass. For a `PNT=0` channel — the photon/eliminated channel always,
//! plus any pair flagged no-penetrability — the penetrability is set to 1.0, the
//! shift to `S_c = B_c` (so `S_c − B_c = 0`), and the phase to 0.0. This encodes
//! SAMMY's `Ymat(2,Ii) -= 1` (`rml/mrml07.f:118-122`): `L_c = (S_c−B_c)+iP_c = i`
//! so `1/L_c = −i`.
//!
//! ## SAMMY Reference
//! - `rml/mrml01.f` — LRF=7 reader (Scan_File_2, particle pair loop)
//! - `rml/mrml09.f` — Level matrix inversion (Yinvrs, Xspfa, Xspsl)
//! - `rml/mrml11.f` — Cross-section calculation (Sectio, Setxqx)
//! - SAMMY manual §3.1 (multi-channel R-matrix)

use num_complex::Complex64;

use nereids_core::constants::{LOG_FLOOR, NEAR_ZERO_FLOOR, PIVOT_FLOOR, QUANTUM_NUMBER_EPS};
use nereids_endf::resonance::{ParticlePair, RmlData, SpinGroup};

use crate::{channel, coulomb, penetrability};

/// Pre-allocated workspace for RML cross-section evaluation.
///
/// Eliminates per-energy-point allocation of NCH×NCH complex matrices
/// (`r_cplx`, `y_tilde`, `y_inv`, `xq`, `xxxx`, `u`) and per-channel
/// vectors (`p_c`, `s_c`, `phi_c`, flags, etc.).
///
/// Flat `Vec<Complex64>` buffers store matrices in row-major order.
/// For typical NCH=3-6, this avoids ~12 small heap allocations per
/// energy point per spin group.
struct RmlWorkspace {
    // ── NCH×NCH complex matrix buffers (flat, row-major) ──────────────
    r_cplx: Vec<Complex64>,
    y_tilde: Vec<Complex64>,
    y_inv: Vec<Complex64>,
    xq: Vec<Complex64>,
    xxxx: Vec<Complex64>,
    u: Vec<Complex64>,
    // ── Augmented matrix for Gauss-Jordan inversion (NCH × 2·NCH) ─────
    aug: Vec<Complex64>,
    // ── Temp row for elimination ───────────────────────────────────────
    aug_tmp: Vec<Complex64>,
    // ── Per-channel vectors ───────────────────────────────────────────
    p_c: Vec<f64>,
    s_c: Vec<f64>,
    phi_c: Vec<f64>,
    l_c: Vec<Complex64>,
    sqrt_p: Vec<f64>,
    omega: Vec<Complex64>,
    is_entrance: Vec<bool>,
    is_fission: Vec<bool>,
    is_capture: Vec<bool>,
    is_inelastic: Vec<bool>,
    is_closed: Vec<bool>,
    // ── Per-resonance scratch ─────────────────────────────────────────
    gamma_vals: Vec<f64>,
}

impl RmlWorkspace {
    fn new() -> Self {
        Self {
            r_cplx: Vec::new(),
            y_tilde: Vec::new(),
            y_inv: Vec::new(),
            xq: Vec::new(),
            xxxx: Vec::new(),
            u: Vec::new(),
            aug: Vec::new(),
            aug_tmp: Vec::new(),
            p_c: Vec::new(),
            s_c: Vec::new(),
            phi_c: Vec::new(),
            l_c: Vec::new(),
            sqrt_p: Vec::new(),
            omega: Vec::new(),
            is_entrance: Vec::new(),
            is_fission: Vec::new(),
            is_capture: Vec::new(),
            is_inelastic: Vec::new(),
            is_closed: Vec::new(),
            gamma_vals: Vec::new(),
        }
    }

    /// Resize all buffers for a spin group with `nch` channels and
    /// `max_widths` resonance width entries, then zero them out.
    fn resize_and_clear(&mut self, nch: usize, max_widths: usize) {
        let nn = nch * nch;
        let aug_len = nch * 2 * nch;

        // Complex matrix buffers — resize then fill with zero.
        resize_and_zero(&mut self.r_cplx, nn);
        resize_and_zero(&mut self.y_tilde, nn);
        resize_and_zero(&mut self.y_inv, nn);
        resize_and_zero(&mut self.xq, nn);
        resize_and_zero(&mut self.xxxx, nn);
        resize_and_zero(&mut self.u, nn);
        resize_and_zero(&mut self.aug, aug_len);
        resize_and_zero(&mut self.aug_tmp, 2 * nch);

        // Per-channel real/bool vectors.
        resize_and_zero_f64(&mut self.p_c, nch);
        resize_and_zero_f64(&mut self.s_c, nch);
        resize_and_zero_f64(&mut self.phi_c, nch);
        resize_and_zero(&mut self.l_c, nch);
        resize_and_zero_f64(&mut self.sqrt_p, nch);
        resize_and_zero(&mut self.omega, nch);
        resize_and_zero_bool(&mut self.is_entrance, nch);
        resize_and_zero_bool(&mut self.is_fission, nch);
        resize_and_zero_bool(&mut self.is_capture, nch);
        resize_and_zero_bool(&mut self.is_inelastic, nch);
        resize_and_zero_bool(&mut self.is_closed, nch);

        // Per-resonance scratch.
        resize_and_zero_f64(&mut self.gamma_vals, max_widths);
    }
}

fn resize_and_zero(buf: &mut Vec<Complex64>, len: usize) {
    buf.clear();
    buf.resize(len, Complex64::ZERO);
}

fn resize_and_zero_f64(buf: &mut Vec<f64>, len: usize) {
    buf.clear();
    buf.resize(len, 0.0);
}

fn resize_and_zero_bool(buf: &mut Vec<bool>, len: usize) {
    buf.clear();
    buf.resize(len, false);
}

/// Compute cross-section contributions from an LRF=7 energy range.
///
/// Returns `(total, elastic, capture, fission)` in barns.
///
/// Iterates over all spin groups (J,π), sums their contributions.
/// A single `RmlWorkspace` is allocated once and reused across spin groups.
///
/// # Panics
/// Panics if `energy_ev` is not finite or is non-positive.  This is a
/// defensive guard at the public boundary; the Python wrapper and the
/// SAMMY-style dispatcher already validate the energy grid via
/// `validate_energy_grid`, so this assertion only fires for direct
/// callers (other Rust crates, tests) that bypass the grid check.
pub fn cross_sections_for_rml_range(rml: &RmlData, energy_ev: f64) -> (f64, f64, f64, f64) {
    // Defensive input validation at the public boundary (issue #558).
    // See `urr::urr_cross_sections` for rationale.  Catches malformed
    // energies before any spin-group iteration, where empty spin-group
    // vecs would otherwise silently return (0, 0, 0, 0) for NaN/∞.
    assert!(
        energy_ev.is_finite() && energy_ev > 0.0,
        "expected positive finite energy_ev, got {energy_ev}"
    );

    let mut total = 0.0;
    let mut elastic = 0.0;
    let mut capture = 0.0;
    let mut fission = 0.0;

    let mut ws = RmlWorkspace::new();

    for sg in &rml.spin_groups {
        let (t, e, cap, fis) = spin_group_cross_sections(
            sg,
            &rml.particle_pairs,
            energy_ev,
            rml.awr,
            rml.target_spin,
            rml.krm,
            &mut ws,
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
///
/// The caller-owned `ws` workspace is resized and reused across spin groups
/// for a given energy evaluation to avoid per-call heap allocations.
fn spin_group_cross_sections(
    sg: &SpinGroup,
    particle_pairs: &[ParticlePair],
    energy_ev: f64,
    awr: f64,
    target_spin: f64,
    krm: u32,
    ws: &mut RmlWorkspace,
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

    // Resize workspace buffers for this spin group's channel count.
    let max_widths = sg
        .resonances
        .iter()
        .map(|r| r.widths.len())
        .max()
        .unwrap_or(nch)
        .max(nch);
    ws.resize_and_clear(nch, max_widths);

    let g_j = channel::statistical_weight(sg.j, target_spin);
    let pok2 = channel::pi_over_k_squared_barns(energy_ev, awr);

    // Entrance-channel CM energy: E_cm = E_lab × AWR/(1+AWR).
    // Each exit channel adds its Q-value to get its own available energy.
    // Reference: SAMMY rml/mrml03.f Fxradi — channel thresholds via Q.
    let e_cm = channel::lab_to_cm_energy(energy_ev, awr);

    for (c, ch) in sg.channels.iter().enumerate() {
        // P3: particle_pair_idx must be a valid index. The old `.min(len-1)` clamped
        // silently, misclassifying any channel with an OOB index as the last pair.
        // An OOB value indicates corrupted ENDF data; let Rust's bounds check panic.
        let pp = &particle_pairs[ch.particle_pair_idx];
        ws.is_entrance[c] = pp.mt == 2;
        ws.is_fission[c] = pp.mt == 18;
        ws.is_capture[c] = pp.mt == 102;
        // Inelastic neutron channels (MT=51+): massive particle, not elastic/fission/capture.
        // Their flux appears in σ_total (optical theorem) but must not be assigned to capture.
        // Reference: ENDF MT number conventions, §3.4; SAMMY rml/mrml11.f Sectio.
        ws.is_inelastic[c] =
            pp.ma >= 0.5 && !ws.is_entrance[c] && !ws.is_fission[c] && !ws.is_capture[c];

        // SAMMY rml/mrml07.f:118-122 (and mrml03.f:235) branch on the per-pair
        // penetrability flag Lpent (= pp.pnt), NOT on particle mass.  PNT≠1 is
        // the "no penetrability" branch: it covers photon/eliminated channels
        // (MA=0) and any pair the evaluation flags PNT=0.  SAMMY implements it as
        // `Ymat(2,Ii) -= 1`; NEREIDS encodes the same thing as P_c=1, S_c=B_c
        // (so S_c−B_c=0), φ_c=0 → L_c = (S_c−B_c)+iP_c = i → 1/L_c = −i, i.e. −1
        // on the imaginary level-matrix diagonal.  The parser guarantees a
        // massless pair carries PNT=0, so the massive-kinematics branch below
        // (which needs a nonzero reduced mass) is never reached by a photon.
        if pp.pnt != 1 {
            ws.p_c[c] = 1.0;
            ws.s_c[c] = ch.boundary;
            ws.phi_c[c] = 0.0;
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
                ws.p_c[c] = 0.0;
                ws.phi_c[c] = 0.0;
                ws.is_closed[c] = true;
                // SHF=0: S_c = B_c so (S_c − B_c) = 0 in the level matrix.
                // SHF=1: S_c is the analytic shift at imaginary argument ρ = iκ.
                //   For non-Coulomb channels we use the Blatt-Weisskopf formula.
                //   For Coulomb channels the imaginary-argument Coulomb shift is
                //   not yet implemented; fall back to S_c = B_c.  This matches
                //   SAMMY's convention: mrml07.f ELSE branch (Su ≤ Echan) sets
                //   Elinvr=1/Elinvi=0 (i.e. L_c = 1) for all closed channels,
                //   Coulomb and non-Coulomb alike, without calling Pghcou.
                let is_coulomb = pp.za.abs() > 0.5 && pp.zb.abs() > 0.5;
                ws.s_c[c] = if pp.shf == 1 && !is_coulomb {
                    let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
                    let kappa = channel::wave_number_from_cm(e_c.abs(), redmas);
                    // Shift factor uses APT (true radius), per SAMMY rml/mrml07.f
                    // Rho = Zkte·Ex with Zkte = Z·Rdtru (mrml03.f:174-177).
                    penetrability::shift_factor_closed(ch.l, kappa * ch.true_radius)
                } else {
                    ch.boundary
                };
            } else {
                // Channel wave number from reduced mass μ = MA·MB/(MA+MB).
                // For elastic (MA=1, MB=AWR): k_c = wave_number(E_lab, AWR) [identical].
                let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
                let k_c = channel::wave_number_from_cm(e_c, redmas);
                // SAMMY radius convention (rml/mrml07.f:118-166, mrml03.f:174-177):
                //   Rho  = Zkte·Ex  (Zkte = Z·Rdtru = APT, true radius)
                //   Rhof = Zkfe·Ex  (Zkfe = Z·Rdeff = APE, effective radius)
                // Penetrability P and shift S always use APT (Rho).  The EFFECTIVE
                // radius (APE, Rhof) drives the phase φ ONLY for the non-Coulomb
                // (hard-sphere) branch, via Sinsix (mrml07.f:131-137).  For a Coulomb
                // channel SAMMY passes Rho (APT) to Pghcou for P, S AND φ; Rhof/APE is
                // computed but never used (mrml07.f:144-161 — both Pghcou calls pass
                // Rho).  Radius roles confirmed by an independent SAMMY-source
                // derivation; cf. PLEIADES models.py:385-386.
                let rho_pen = k_c * ch.true_radius; // APT → P_c, S_c, and Coulomb φ_c
                let rho_phase = k_c * ch.effective_radius; // APE → non-Coulomb φ_c only
                // ── Coulomb vs hard-sphere routing ───────────────────────────
                // SAMMY rml/mrml07.f Pgh — `if (Zeta(I).NE.Zero)` branch.
                // Both particles charged → Coulomb wave functions F_L / G_L.
                // One neutral (za=0 or zb=0) → hard-sphere Blatt-Weisskopf.
                if pp.za.abs() > 0.5 && pp.zb.abs() > 0.5 {
                    // Coulomb channel (e.g. n+α→p+X, (n,p), fission fragments).
                    // SAMMY computes P_c, S_c AND φ_c from a single radius Rho (APT)
                    // via Pghcou; the effective radius (Rhof/APE) is NOT used for a
                    // Coulomb channel (mrml07.f:144-161 — both Pghcou calls pass Rho).
                    let eta = coulomb::sommerfeld_eta(pp.za, pp.zb, pp.ma, pp.mb, e_c);
                    match coulomb::coulomb_wave_functions(ch.l, eta, rho_pen) {
                        Some((f, g, fp, gp)) => {
                            // rho_pen (APT) succeeded: channel is genuinely open.
                            let fg_sq = f * f + g * g;
                            ws.p_c[c] = rho_pen / fg_sq;
                            // SHF=1: Coulomb shift ρ(F·F'+G·G')/(F²+G²).
                            // SHF=0: S_c = B_c so (S_c − B_c) = 0 in level matrix.
                            // Note: parser rejects Coulomb + SHF=1, so this arm is
                            // only reachable if that validation is later relaxed.
                            ws.s_c[c] = if pp.shf == 1 {
                                rho_pen * (f * fp + g * gp) / fg_sq
                            } else {
                                ch.boundary
                            };
                            // φ_c = atan2(F_L, G_L) from the SAME (F,G) solve at rho_pen
                            // (APT).  SAMMY's second Pghcou call reuses Rho, never Rhof,
                            // so the Coulomb phase is independent of APE.
                            //
                            // No dedicated regression test exists for this radius
                            // choice because it is unobservable in the angle-integrated
                            // cross-sections NEREIDS computes: an exit channel's phase
                            // enters only via Ω_c = e^{-iφ_c}, and σ_total uses the
                            // entrance U_{c0,c0} (entrance φ only) while σ_reaction uses
                            // |U_{c0,c'}|² (phase-independent magnitude).  Coulomb
                            // channels are always exit channels (entrance is mt=2,
                            // non-Coulomb).  The fix is for SAMMY-faithfulness and would
                            // matter only if angular distributions were added.
                            ws.phi_c[c] = f.atan2(g);
                        }
                        None => {
                            // rho_pen ≤ acch (≈ 1e-8, SAMMY Coulfg threshold):
                            // penetrability → 0 at threshold; treat as closed.
                            // Reference: SAMMY coulomb/mrml08.f90 Coulfg — acch.
                            ws.p_c[c] = 0.0;
                            ws.s_c[c] = ch.boundary;
                            ws.phi_c[c] = 0.0;
                            ws.is_closed[c] = true;
                        }
                    }
                } else {
                    // Hard-sphere (Blatt-Weisskopf) channel.
                    ws.p_c[c] = penetrability::penetrability(ch.l, rho_pen);
                    // SHF=0: shift factor not calculated; S_c = B_c so (S_c - B_c) = 0
                    // in the level matrix diagonal.
                    // SHF=1: calculate S_c analytically (Blatt-Weisskopf).
                    // Reference: ENDF-6 §2.2.1.6; SAMMY rml/mrml07.f Pgh (Ishift check)
                    ws.s_c[c] = if pp.shf == 1 {
                        penetrability::shift_factor(ch.l, rho_pen)
                    } else {
                        ch.boundary
                    };
                    ws.phi_c[c] = penetrability::phase_shift(ch.l, rho_phase);
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
    // ws.r_cplx is already zeroed by resize_and_clear.
    for res in &sg.resonances {
        let e_tilde = if krm == 3 {
            // KRM=3 (Reich-Moore): convert formal partial widths Γ_nc to reduced
            // amplitudes γ_nc and form the complex pole energy Ẽ_n = E_n − i·Γ_γ/2
            // (the −iΓ_γ/2 makes capture implicit).  Reference: ENDF-6 §2.2.1.6;
            // SAMMY rml/mrml01.f, rml/mrml03.f (Betset).
            let e_tilde = Complex64::new(res.energy, -res.gamma_gamma / 2.0);
            // γ_nc is a property of the resonance, evaluated at the resonance
            // energy E_n (not the incident energy E), so it is energy-independent.
            // SAMMY Betset (rml/mrml03.f:235-274) branches on the per-pair
            // penetrability flag Lpent (= pp.pnt), NOT on particle mass:
            //   PNT=0: γ_nc = √(½|Γ|)                                 (mrml03.f:236)
            //   PNT=1: γ_nc = √(½|Γ| / P),  P at ρ = κ·APT, κ from |Eres−Echan|
            //                                                         (mrml03.f:241-273)
            // The |Eres−Echan| absolute value means bound/subthreshold resonances
            // (e_cm_n ≤ 0) still receive a real, positive penetrability — they are
            // NOT treated as closed channels.  The sign of Γ is preserved either way.
            for c in 0..nch {
                let gamma_formal = res.widths[c];
                let ch = &sg.channels[c];
                let pp_c = &particle_pairs[ch.particle_pair_idx];
                let magnitude = if pp_c.pnt != 1 {
                    // No-penetrability branch (Lpent=0): photon/eliminated channels
                    // and any pair flagged PNT=0.  γ = √(½|Γ|).  mrml03.f:236.
                    (0.5 * gamma_formal.abs()).sqrt()
                } else {
                    // Penetrability branch (Lpent=1).  ρ = κ·APT evaluated at the
                    // absolute CM-energy separation |Eres−Echan| = |e_cm_n|, so a
                    // bound state (e_cm_n < 0) still gets a real P.  Radius is APT
                    // (true_radius) per Betset's Zkte = Z·Rdtru (mrml03.f:244).
                    let e_cm_n = channel::lab_to_cm_energy(res.energy, awr) + pp_c.q;
                    let ex = e_cm_n.abs();
                    let redmas = pp_c.ma * pp_c.mb / (pp_c.ma + pp_c.mb);
                    let k_cn = channel::wave_number_from_cm(ex, redmas);
                    let rho_pen_n = k_cn * ch.true_radius; // APT
                    // Same Coulomb-vs-hard-sphere routing as the open-channel block
                    // so the normalisation is self-consistent.  SAMMY rml/mrml07.f
                    // Pgh — same Zeta check applies here.
                    let p = if pp_c.za.abs() > 0.5 && pp_c.zb.abs() > 0.5 {
                        let eta = coulomb::sommerfeld_eta(pp_c.za, pp_c.zb, pp_c.ma, pp_c.mb, ex);
                        coulomb::coulomb_wave_functions(ch.l, eta, rho_pen_n)
                            .map_or(0.0, |(fl, gl, _, _)| rho_pen_n / (fl * fl + gl * gl))
                    } else {
                        penetrability::penetrability(ch.l, rho_pen_n)
                    };
                    // P = 0 only at the exact channel threshold (Ex = 0) or below
                    // the Coulomb acch cutoff — a measure-zero singularity.  Fall
                    // back to the no-penetrability form √(½|Γ|) to keep γ finite.
                    if p > 0.0 {
                        (0.5 * gamma_formal.abs() / p).sqrt()
                    } else {
                        (0.5 * gamma_formal.abs()).sqrt()
                    }
                };
                ws.gamma_vals[c] = magnitude.copysign(gamma_formal);
            }
            e_tilde
        } else {
            // KRM=2: widths are already reduced amplitudes; real denominator.
            // P2: Guard only against exact IEEE 754 zero; complex infrastructure
            // handles the Lorentzian width naturally via i·P_c in level matrix.
            ws.gamma_vals[..nch].copy_from_slice(&res.widths[..nch]);
            Complex64::new(res.energy, 0.0)
        };

        let denom = e_tilde - energy_ev;
        // Near-pole regularization: add a tiny imaginary offset ε to the denominator
        // so that evaluating exactly at E = E_n (where denom → 0 for real KRM=2 poles)
        // gives a finite, physically meaningful result via the Cauchy principal value.
        // For KRM=3, e_tilde already carries −iΓ_γ/2 so the denominator is never zero;
        // the correction is negligible (ε << Γ_γ/2).
        // Reference: Cauchy PV regularisation; SAMMY avoids the exact pole by perturbing
        // the resonance energy during input processing.
        let inv_denom = if denom.norm() < QUANTUM_NUMBER_EPS {
            (denom + Complex64::new(0.0, QUANTUM_NUMBER_EPS)).inv()
        } else {
            denom.inv()
        };
        for c in 0..nch {
            // SAMMY gates the R-matrix accumulation on BOTH channels being
            // open: rml/mrml07.f Setr lines 67-71 —
            //   IF (Su.GT.Echan(K) .AND. Su.GT.Echan(L) .AND.
            //     Beta(KL,Ires).NE.Zero) THEN
            //       Rmat(1,KL) = Rmat(1,KL) + Alphar(Ires)*Beta(KL,Ires)
            //       ...
            // `Su.GT.Echan(K)` is exactly the open-channel test (line 118), which
            // NEREIDS encodes as `!ws.is_closed[K]` (set in the channel-setup
            // loop above for e_c ≤ 0 and the Coulomb-threshold case).  A channel
            // below its threshold contributes nothing to R, including off-diagonal
            // terms via a shared resonance width.  Skipping the whole row when c
            // is closed avoids forming R[c,cp] for any cp.
            if ws.is_closed[c] {
                continue;
            }
            let gc = ws.gamma_vals[c];
            for cp in 0..nch {
                // Gate the off-/on-diagonal term on the partner channel cp also
                // being open (the Su.GT.Echan(L) half of the SAMMY condition).
                // (The Beta≠0 half is automatic: a zero width product adds 0.)
                if ws.is_closed[cp] {
                    continue;
                }
                ws.r_cplx[c * nch + cp] += gc * ws.gamma_vals[cp] * inv_denom;
            }
        }
    }

    // ── L_c = (S_c - B_c) + i·P_c (per-channel level denominator) ───────────
    // Reference: SAMMY rml/mrml07.f Pgh subroutine, "PH = 1/(S-B+IP)"
    for c in 0..nch {
        ws.l_c[c] = Complex64::new(ws.s_c[c] - sg.channels[c].boundary, ws.p_c[c]);
    }

    // ── Reduced level matrix Ỹ = L⁻¹ - R (SAMMY "Ymat") ─────────────────────
    // Ỹ_cc'(E) = (1/L_c)·δ_cc' - R_cc'
    // Reference: SAMMY rml/mrml07.f — "Ymat = (1/(S-B+IP) - Rmat)"
    //
    // NOTE: This is NOT (L - R). The SAMMY formulation inverts L⁻¹ - R, not
    // L - R. Using L - R gives |U| = 3 for the hard sphere (R=0, A=iP,
    // A⁻¹·P = -i/P·P = -i, W = 1+2i²·(−1)=3) — catastrophically wrong.
    // Using L⁻¹ - R gives |U| = 1 for R=0 (Ỹ = 1/L, Ỹinv = L, XQ = L·0 = 0,
    // XXXX = 0, W = 1, U = exp(2iφ)) — correct hard-sphere limit.
    for c in 0..nch {
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
        let inv_l = if ws.l_c[c].norm_sqr() < NEAR_ZERO_FLOOR {
            // |L_c|² < NEAR_ZERO_FLOOR: use finite-but-large sentinel so the diagonal
            // dominates and the channel decouples without overflow in inversion.
            Complex64::new(1e30, 0.0)
        } else {
            Complex64::new(1.0, 0.0) / ws.l_c[c]
        };
        for cp in 0..nch {
            let diag = if c == cp { inv_l } else { Complex64::ZERO };
            ws.y_tilde[c * nch + cp] = diag - ws.r_cplx[c * nch + cp];
        }
    }

    // ── Invert Ỹ to get Ỹinv (SAMMY "Yinv") ─────────────────────────────────
    // Reference: SAMMY rml/mrml09.f Yinvrs subroutine
    if !invert_complex_matrix_flat(
        &ws.y_tilde,
        nch,
        &mut ws.y_inv,
        &mut ws.aug,
        &mut ws.aug_tmp,
    ) {
        // Singular Ỹ matrix — regularize by adding a small real epsilon to
        // the diagonal and retry.  This can happen when channels are
        // near-degenerate (e.g. two channels at the same threshold).
        // The epsilon is real-only to preserve Hermitian symmetry of the
        // level matrix; an imaginary perturbation would break unitarity.
        //
        // Use a *relative* epsilon: ε = |diag| × QUANTUM_NUMBER_EPS, with a floor of
        // NEAR_ZERO_FLOOR for zero diagonals.  A fixed absolute epsilon
        // could be comparable to or larger than the diagonal value itself
        // for high-L channels with very small penetrabilities (where
        // 1/L_c ~ 1/P_c can be enormous, but R_cc' is also large, making
        // the net diagonal small).  The relative approach perturbs the
        // matrix by a fraction of its natural scale.
        //
        // Per-diagonal (not matrix-norm) regularization is intentional:
        // each channel's diagonal element lives on its own physical scale
        // (set by 1/L_c − R_cc), which can differ by orders of magnitude
        // across channels (e.g. an s-wave elastic channel vs. a high-L
        // fission channel).  A single matrix-norm epsilon would be
        // dominated by the largest channel and could either over-perturb
        // small channels or under-perturb large ones.  Per-diagonal
        // epsilon ensures each channel is nudged proportionally to its
        // own scale.

        // Copy y_tilde into y_inv as a temp buffer for the regularized matrix.
        ws.y_inv.copy_from_slice(&ws.y_tilde);
        for i in 0..nch {
            let diag_norm = ws.y_inv[i * nch + i].norm();
            // Relative regularization (QUANTUM_NUMBER_EPS × diagonal) with an
            // absolute floor of NEAR_ZERO_FLOOR for near-zero diagonals.
            let eps = (diag_norm * QUANTUM_NUMBER_EPS).max(NEAR_ZERO_FLOOR);
            ws.y_inv[i * nch + i] += Complex64::new(eps, 0.0);
        }
        // y_inv now holds the regularized y_tilde; copy it to y_tilde so the
        // inversion reads from the regularized version.
        ws.y_tilde.copy_from_slice(&ws.y_inv);
        if !invert_complex_matrix_flat(
            &ws.y_tilde,
            nch,
            &mut ws.y_inv,
            &mut ws.aug,
            &mut ws.aug_tmp,
        ) {
            return (0.0, 0.0, 0.0, 0.0); // truly degenerate
        }
    }

    // ── XQ = Ỹinv · R (matrix product, SAMMY "Xqr/Xqi") ─────────────────────
    // Reference: SAMMY rml/mrml11.f Setxqx — "Xqr(k,i) = (L**-1-R)**-1 * R"
    for c in 0..nch {
        for cp in 0..nch {
            let mut sum = Complex64::ZERO;
            for k in 0..nch {
                sum += ws.y_inv[c * nch + k] * ws.r_cplx[k * nch + cp];
            }
            ws.xq[c * nch + cp] = sum;
        }
    }

    // ── XXXX = (√P_c / L_c) · XQ · √P_c' ────────────────────────────────────
    // Reference: SAMMY rml/mrml11.f Setxqx — "Xxxx = sqrt(P)/L * xq * sqrt(P)"
    for c in 0..nch {
        ws.sqrt_p[c] = ws.p_c[c].sqrt();
    }
    for c in 0..nch {
        // For a closed channel: sqrt_p[c] = 0 and L_c = 0 (0/0 indeterminate).
        // The full XXXX[c,cp] = (√P_c / L_c) · XQ[c,cp] · √P_c'.
        // Since √P_c = 0 for any closed channel c, the entire row is zero
        // regardless of the value of L_c. Setting sqrt_p_over_l = 0 is correct.
        // (The Ỹ sentinel handles Ỹ inversion correctly; this row zeroing is
        //  consistent: a closed channel contributes nothing to XXXX/U.)
        // Guard: at exact channel threshold, both √P_c and L_c can be
        // zero simultaneously, producing 0/0 = NaN.  The is_closed flag
        // catches most cases, but a channel right at threshold might not
        // be flagged closed yet have |L_c| ≈ 0.  The extra norm check
        // prevents NaN propagation.
        let sqrt_p_over_l = if ws.is_closed[c] || ws.l_c[c].norm() < PIVOT_FLOOR {
            Complex64::ZERO
        } else {
            ws.sqrt_p[c] / ws.l_c[c]
        };
        for cp in 0..nch {
            ws.xxxx[c * nch + cp] = sqrt_p_over_l * ws.xq[c * nch + cp] * ws.sqrt_p[cp];
        }
    }

    // ── Collision matrix U = Ω · W · Ω, W = I + 2i·Ξ ────────────────────────
    //
    // Phase convention: Ω_c = exp(-iφ_c), NOT exp(+iφ_c).
    //
    // TRUTH SOURCE: SAMMY rml/mrml11.f lines 14-18:
    //   "W(c,c') = delta(c,c') + 2i XXXX(c,c')" (eq. III.D.4)
    // And mrml11.f lines 84-88, the elastic formula:
    //   "sin²(φ)·(1-2Xi) - sin(2φ)·Xr + Xr²+Xi²"
    // is consistent ONLY with U = e^{-2iφ}·(I+2iX), not e^{+2iφ}.
    //
    // For hard sphere (W=I): |1-e^{-2iφ}|² = |1-e^{+2iφ}|² = 4sin²φ,
    // so the sign error is invisible in unitarity tests. It ONLY manifests
    // when resonances are present (W ≠ I), producing wrong interference
    // patterns in elastic and incorrect total from optical theorem.
    //
    // History: same class of bug as the MLBW e^{+2iφ} error fixed in
    // slbw.rs (commit f0eadc1). The negative exponent is the ENDF/SAMMY
    // convention; the positive exponent is a common error.
    for c in 0..nch {
        ws.omega[c] = Complex64::from_polar(1.0, -ws.phi_c[c]);
    }

    for c in 0..nch {
        for cp in 0..nch {
            let delta = if c == cp {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::ZERO
            };
            let w_cc = delta + Complex64::new(0.0, 2.0) * ws.xxxx[c * nch + cp];
            ws.u[c * nch + cp] = ws.omega[c] * w_cc * ws.omega[cp];
        }
    }

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
    let has_explicit_capture = ws.is_capture[..nch].iter().any(|&x| x);

    for c0 in 0..nch {
        if !ws.is_entrance[c0] {
            continue;
        }
        let u_diag = ws.u[c0 * nch + c0];
        // σ_total (optical theorem, per entrance channel)
        tot += 2.0 * pok2 * g_j * (1.0 - u_diag.re);
        // σ_elastic: |1 - U_{c0,c0}|²
        elas += pok2 * g_j * (Complex64::new(1.0, 0.0) - u_diag).norm_sqr();

        for cp in 0..nch {
            if ws.is_fission[cp] {
                // σ_fission: |U_{c0,c'}|² for fission channels c'
                fis += pok2 * g_j * ws.u[c0 * nch + cp].norm_sqr();
            }
            if has_explicit_capture && ws.is_capture[cp] {
                // σ_capture (explicit): |U_{c0,c'}|² for photon channels c'.
                // Avoids lumping inelastic neutron channels (MT=51+) into capture.
                // Reference: SAMMY rml/mrml11.f Sectio — explicit sum over γ channels.
                cap += pok2 * g_j * ws.u[c0 * nch + cp].norm_sqr();
            }
            if ws.is_inelastic[cp] {
                // σ_inelastic: |U_{c0,c'}|² for inelastic neutron channels (MT=51+).
                // Tracked separately so KRM=3 capture residual excludes this flux.
                // Reference: ENDF MT conventions §3.4; SAMMY rml/mrml11.f Sectio.
                inel += pok2 * g_j * ws.u[c0 * nch + cp].norm_sqr();
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

// ── Complex Gauss-Jordan Elimination (flat-buffer version) ──────────────────
//
// Inverts an n×n complex matrix using Gauss-Jordan elimination with partial
// pivoting. Returns false if the matrix is singular (pivot magnitude < LOG_FLOOR).
//
// For LRF=7 isotopes relevant to VENUS imaging, NCH ≤ 6, so O(n³) is fast.
// SAMMY uses a specialized complex symmetric factorization (Xspfa/Xspsl in
// rml/mrml10.f), but Gauss-Jordan is correct and sufficient for our purposes.
//
// All buffers are caller-provided to avoid per-call allocation:
// - `a`: input matrix (flat row-major, n×n)
// - `out`: output inverse (flat row-major, n×n)
// - `aug`: augmented matrix workspace (flat row-major, n×2n)
// - `tmp`: temporary row buffer (length 2n)

fn invert_complex_matrix_flat(
    a: &[Complex64],
    n: usize,
    out: &mut [Complex64],
    aug: &mut [Complex64],
    tmp: &mut [Complex64],
) -> bool {
    let w = 2 * n; // augmented row width

    // Build augmented matrix [A | I] of size n × 2n
    for r in 0..n {
        for c in 0..n {
            aug[r * w + c] = a[r * n + c];
        }
        for c in 0..n {
            aug[r * w + n + c] = if c == r {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::ZERO
            };
        }
    }

    for col in 0..n {
        // Partial pivoting: find row with largest magnitude in this column
        let mut best = col;
        let mut best_norm = aug[col * w + col].norm();
        for r in (col + 1)..n {
            let norm = aug[r * w + col].norm();
            if norm > best_norm {
                best = r;
                best_norm = norm;
            }
        }
        // Swap rows col and best in aug
        if best != col {
            for j in 0..w {
                aug.swap(col * w + j, best * w + j);
            }
        }

        let pivot = aug[col * w + col];
        if pivot.norm() < LOG_FLOOR {
            return false; // singular
        }

        // Scale pivot row so leading entry becomes 1
        let inv_pivot = pivot.inv();
        for j in 0..w {
            aug[col * w + j] *= inv_pivot;
        }

        // Eliminate this column from all other rows.
        // Copy pivot row to tmp to avoid aliasing issues.
        tmp[..w].copy_from_slice(&aug[col * w..col * w + w]);
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * w + col];
            if factor.norm() < LOG_FLOOR {
                continue;
            }
            for j in 0..w {
                aug[row * w + j] -= factor * tmp[j];
            }
        }
    }

    // Extract the right half (the inverse)
    for r in 0..n {
        for c in 0..n {
            out[r * n + c] = aug[r * w + n + c];
        }
    }
    true
}

// ── Legacy wrapper for tests (allocating version) ───────────────────────────
#[cfg(test)]
fn invert_complex_matrix(a: &[Vec<Complex64>], n: usize) -> Option<Vec<Vec<Complex64>>> {
    let flat_a: Vec<Complex64> = a.iter().flat_map(|row| row.iter().copied()).collect();
    let mut flat_out = vec![Complex64::ZERO; n * n];
    let mut aug = vec![Complex64::ZERO; n * 2 * n];
    let mut tmp = vec![Complex64::ZERO; 2 * n];
    if invert_complex_matrix_flat(&flat_a, n, &mut flat_out, &mut aug, &mut tmp) {
        Some(
            (0..n)
                .map(|r| flat_out[r * n..(r + 1) * n].to_vec())
                .collect(),
        )
    } else {
        None
    }
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
            za: 0.0,  // neutron charge Z=0
            zb: 74.0, // W-184 charge Z=74 (ENDF LRF=7 stores charge directly)
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
    ///   ch0: neutron (za=0)  + target — hard-sphere entrance channel
    ///   ch1: α (za=2)        + O-16 (zb=8) — Coulomb exit, Q=+50 eV (always open)
    ///
    /// ENDF LRF=7 stores charge Z directly in ZA/ZB (neutron=0, alpha=2, O-16=8).
    ///
    /// Verifies σ_total ≥ 0 (physics sanity) and no panic at both an open
    /// and a closed Coulomb channel (Q very negative).
    ///
    /// SAMMY ref: rml/mrml07.f Pgh — `if (Zeta(I).NE.Zero)` branch.
    #[test]
    fn test_coulomb_channel_open_and_closed_no_panic() {
        use nereids_endf::resonance::{ParticlePair, RmlChannel, RmlData, SpinGroup};

        // Entrance channel: neutron (Z=0) + W-184 target (Z=74).
        // ENDF LRF=7 stores charge directly: neutron za=0, W-184 zb=74.
        let pp_entrance = ParticlePair {
            ma: 1.0,
            mb: 184.0,
            za: 0.0,  // neutron charge Z=0
            zb: 74.0, // W-184 charge Z=74
            ia: 0.5,
            ib: 0.0,
            q: 0.0,
            pnt: 1,
            shf: 0,
            mt: 2,
            pa: 1.0,
            pb: 1.0,
        };

        // Coulomb exit channel: α(Z=2) + O-16(Z=8), Q=+50 eV → always open.
        // sommerfeld_eta(2, 8, ...) gives η > 0, confirming Coulomb branch.
        let pp_coulomb_open = ParticlePair {
            ma: 4.0,
            mb: 16.0,
            za: 2.0, // alpha charge Z=2
            zb: 8.0, // O-16 charge Z=8
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
            za: 2.0, // alpha charge Z=2
            zb: 8.0, // O-16 charge Z=8
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

    /// Fix #6 (non-vacuous): a CLOSED channel must contribute nothing to the
    /// R-matrix, even when a resonance carries a shared off-diagonal width that
    /// couples it to an OPEN channel.
    ///
    /// SAMMY gates the R accumulation on both channels open (rml/mrml07.f Setr
    /// lines 67-71: `IF (Su.GT.Echan(K) .AND. Su.GT.Echan(L) ...)`).  Before
    /// this fix the un-gated loop formed R[open,closed] = γ_open·γ_closed/(E_n−E)
    /// ≠ 0, which leaks into the open-channel cross-section through the level-
    /// matrix inversion (XQ = Ỹ⁻¹·R picks up Ỹ⁻¹[0,1]·R[1,0]).
    ///
    /// This test (unlike the `resonances: vec![]` no-panic test, where R ≡ 0)
    /// carries a resonance with a large off-diagonal width into a closed
    /// channel, and asserts:
    ///   (a) the gated cross-section equals the open-only-submatrix result
    ///       (the same group with the closed channel's width zeroed); and
    ///   (b) it differs from the pre-fix un-gated value (reconstructed here by
    ///       building the full R including the closed-channel coupling and
    ///       running the rest of the pipeline), so the gate is load-bearing.
    #[test]
    fn test_rml_closed_channel_excluded_from_rmatrix() {
        use nereids_endf::resonance::{ParticlePair, RmlChannel, RmlData, RmlResonance, SpinGroup};

        // Entrance: neutron (Z=0) + target (Z=74), l=0 so the open channel has
        // full penetrability P_0 = ρ and a healthy cross-section.
        let pp_entrance = ParticlePair {
            ma: 1.0,
            mb: 183.0,
            za: 0.0,
            zb: 74.0,
            ia: 0.5,
            ib: 0.0,
            q: 0.0,
            pnt: 1,
            shf: 0, // l=0: shift is 0 regardless; L_0 = i·P_0 (finite, nonzero)
            mt: 2,
            pa: 1.0,
            pb: 1.0,
        };
        // Inelastic neutron exit channel (MT=51), massive, with a very negative
        // Q so e_c = e_cm + Q < 0 → CLOSED at the test energy.  l=1 with SHF=1 so
        // the closed channel keeps a finite analytic shift S_1(iκ) = −1/(1−(κa)²)
        // and L_c does NOT collapse to the 1/L_c → 1e30 decoupling sentinel.
        // That sentinel (used for L_c ≈ 0) would otherwise numerically swamp the
        // off-diagonal leak and make the test vacuous; a finite L_c keeps
        // Ỹ⁻¹[0,1]·R[1,0] observable in the open channel.
        let pp_closed = ParticlePair {
            ma: 1.0,
            mb: 183.0,
            za: 0.0,
            zb: 74.0,
            ia: 0.5,
            ib: 0.0,
            // E_cm ≈ 100·183/184 ≈ 99.5 eV at E_lab=100; Q=−200 → e_c ≈ −100.5 eV
            // (CLOSED), with |e_c| comparable so S_1(iκ) is moderate (not a pole).
            q: -200.0,
            pnt: 1,
            shf: 1,
            mt: 51,
            pa: 1.0,
            pb: 1.0,
        };

        let ch_open = RmlChannel {
            particle_pair_idx: 0,
            l: 0,
            channel_spin: 0.5,
            boundary: 0.0,
            effective_radius: 8.3,
            true_radius: 8.3,
        };
        let ch_closed = RmlChannel {
            particle_pair_idx: 1,
            l: 1,
            channel_spin: 0.5,
            boundary: 0.0,
            effective_radius: 8.3,
            true_radius: 8.3,
        };

        // KRM=2: widths ARE reduced amplitudes (copied verbatim), so the closed
        // channel keeps a nonzero γ and the off-diagonal coupling γ_0·γ_1 is
        // real and large pre-fix.
        let res_full = RmlResonance {
            energy: 120.0,
            gamma_gamma: 0.0,       // KRM=2: no implicit capture
            widths: vec![3.0, 5.0], // [open, closed]; large shared off-diagonal
        };
        // Open-only submatrix reference: zero the closed channel's width.
        let res_open_only = RmlResonance {
            energy: 120.0,
            gamma_gamma: 0.0,
            widths: vec![3.0, 0.0],
        };

        let make_rml = |res: RmlResonance| RmlData {
            target_spin: 0.0,
            awr: 183.0,
            scattering_radius: 8.3,
            krm: 2,
            particle_pairs: vec![pp_entrance.clone(), pp_closed.clone()],
            spin_groups: vec![SpinGroup {
                j: 0.5,
                parity: 1.0,
                channels: vec![ch_open.clone(), ch_closed.clone()],
                resonances: vec![res],
                has_background_correction: false,
            }],
        };

        // Evaluate off the pole (E ≠ E_n) so the result is well-conditioned.
        let energy = 100.0;
        let rml_full = make_rml(res_full.clone());
        let rml_open_only = make_rml(res_open_only);

        let (tot_g, elas_g, cap_g, fis_g) = cross_sections_for_rml_range(&rml_full, energy);
        let (tot_ref, elas_ref, cap_ref, fis_ref) =
            cross_sections_for_rml_range(&rml_open_only, energy);

        // (a) Gated full-width result == open-only-submatrix result.
        for (g, r, name) in [
            (tot_g, tot_ref, "total"),
            (elas_g, elas_ref, "elastic"),
            (cap_g, cap_ref, "capture"),
            (fis_g, fis_ref, "fission"),
        ] {
            assert!(
                (g - r).abs() <= 1e-9 * r.abs().max(1.0),
                "{name}: gated σ ({g}) must equal open-only-submatrix σ ({r}); \
                 a closed channel must not couple into R"
            );
        }

        // (b) The gate is load-bearing: reconstruct the PRE-FIX un-gated value
        // by computing the full collision matrix WITHOUT the both-open gate, and
        // confirm it differs from the gated result at the open entrance channel.
        let ungated_tot = ungated_total_for_two_channel(&rml_full, energy);
        // For this configuration the leak is large (gated ≈ 2.62 b vs un-gated
        // ≈ 5.54 b, a >100% difference), so the gate is decisively load-bearing.
        assert!(
            (ungated_tot - tot_g).abs() > 1e-3,
            "un-gated σ_total ({ungated_tot}) should differ from gated σ_total \
             ({tot_g}); if equal, the off-diagonal closed-channel coupling does \
             not affect the open channel and the test is vacuous"
        );
    }

    /// Reconstruct the pre-fix (un-gated) σ_total for a 2-channel KRM=2 spin
    /// group: builds the full complex R-matrix INCLUDING the closed-channel
    /// off-diagonal coupling, then runs the same Ỹ⁻¹/XXXX/U pipeline as the
    /// production code.  Used only to prove the Fix #6 gate is load-bearing.
    fn ungated_total_for_two_channel(rml: &RmlData, energy_ev: f64) -> f64 {
        use num_complex::Complex64;

        let awr = rml.awr;
        let sg = &rml.spin_groups[0];
        let nch = sg.channels.len();
        assert_eq!(nch, 2, "helper is specialised to 2 channels");
        let pps = &rml.particle_pairs;

        let e_cm = channel::lab_to_cm_energy(energy_ev, awr);
        let mut p_c = vec![0.0f64; nch];
        let mut s_c = vec![0.0f64; nch];
        let mut phi_c = vec![0.0f64; nch];
        let mut is_closed = vec![false; nch];
        let mut is_entrance = vec![false; nch];

        for (c, ch) in sg.channels.iter().enumerate() {
            let pp = &pps[ch.particle_pair_idx];
            is_entrance[c] = pp.mt == 2;
            let e_c = e_cm + pp.q;
            if e_c <= 0.0 {
                // Closed channel.  Honour SHF like production (lines 338-346):
                // SHF=1 → finite analytic shift at imaginary argument, so L_c is
                // finite and the off-diagonal R coupling is NOT swamped by the
                // 1/L_c → 1e30 sentinel.  This is what makes the leak observable.
                p_c[c] = 0.0;
                phi_c[c] = 0.0;
                is_closed[c] = true;
                let is_coulomb = pp.za.abs() > 0.5 && pp.zb.abs() > 0.5;
                s_c[c] = if pp.shf == 1 && !is_coulomb {
                    let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
                    let kappa = channel::wave_number_from_cm(e_c.abs(), redmas);
                    penetrability::shift_factor_closed(ch.l, kappa * ch.true_radius)
                } else {
                    ch.boundary
                };
            } else {
                let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
                let k_c = channel::wave_number_from_cm(e_c, redmas);
                let rho_pen = k_c * ch.true_radius;
                let rho_phase = k_c * ch.effective_radius;
                p_c[c] = penetrability::penetrability(ch.l, rho_pen);
                s_c[c] = if pp.shf == 1 {
                    penetrability::shift_factor(ch.l, rho_pen)
                } else {
                    ch.boundary
                };
                phi_c[c] = penetrability::phase_shift(ch.l, rho_phase);
            }
        }

        // Full R-matrix, NO both-open gate (the pre-fix behaviour).
        let mut r = vec![Complex64::ZERO; nch * nch];
        for res in &sg.resonances {
            let denom = Complex64::new(res.energy, 0.0) - Complex64::new(energy_ev, 0.0);
            let inv_denom = if denom.norm() < QUANTUM_NUMBER_EPS {
                (denom + Complex64::new(0.0, QUANTUM_NUMBER_EPS)).inv()
            } else {
                denom.inv()
            };
            for c in 0..nch {
                let gc = res.widths[c];
                for cp in 0..nch {
                    r[c * nch + cp] += gc * res.widths[cp] * inv_denom;
                }
            }
        }

        // L_c, Ỹ = diag(1/L_c) − R.
        let mut y = vec![Complex64::ZERO; nch * nch];
        for c in 0..nch {
            let l_c = Complex64::new(s_c[c] - sg.channels[c].boundary, p_c[c]);
            let inv_l = if l_c.norm_sqr() < NEAR_ZERO_FLOOR {
                Complex64::new(1e30, 0.0)
            } else {
                Complex64::new(1.0, 0.0) / l_c
            };
            for cp in 0..nch {
                let diag = if c == cp { inv_l } else { Complex64::ZERO };
                y[c * nch + cp] = diag - r[c * nch + cp];
            }
        }

        // Invert Ỹ (2×2 closed form).
        let det = y[0] * y[3] - y[1] * y[2];
        let yinv = [y[3] / det, -y[1] / det, -y[2] / det, y[0] / det];

        // XQ = Ỹ⁻¹·R.
        let mut xq = [Complex64::ZERO; 4];
        for c in 0..nch {
            for cp in 0..nch {
                let mut sum = Complex64::ZERO;
                for k in 0..nch {
                    sum += yinv[c * nch + k] * r[k * nch + cp];
                }
                xq[c * nch + cp] = sum;
            }
        }

        // XXXX, with the closed-channel √P-kill (matches production line 655).
        let sqrt_p: Vec<Complex64> = p_c.iter().map(|&p| Complex64::new(p, 0.0).sqrt()).collect();
        let mut xxxx = [Complex64::ZERO; 4];
        for c in 0..nch {
            let l_c = Complex64::new(s_c[c] - sg.channels[c].boundary, p_c[c]);
            let sqrt_p_over_l = if is_closed[c] || l_c.norm() < PIVOT_FLOOR {
                Complex64::ZERO
            } else {
                sqrt_p[c] / l_c
            };
            for cp in 0..nch {
                xxxx[c * nch + cp] = sqrt_p_over_l * xq[c * nch + cp] * sqrt_p[cp];
            }
        }

        // U and σ_total via the optical theorem on the open entrance channel.
        let omega: Vec<Complex64> = phi_c
            .iter()
            .map(|&phi| Complex64::from_polar(1.0, -phi))
            .collect();
        let pok2 = channel::pi_over_k_squared_barns(energy_ev, awr);
        // Use the SAME statistical weight as production so the only difference
        // between this helper and `cross_sections_for_rml_range` is the gate.
        let g_j = channel::statistical_weight(sg.j, rml.target_spin);
        let mut tot = 0.0;
        for c0 in 0..nch {
            if !is_entrance[c0] {
                continue;
            }
            let w_cc = Complex64::new(1.0, 0.0) + Complex64::new(0.0, 2.0) * xxxx[c0 * nch + c0];
            let u_diag = omega[c0] * w_cc * omega[c0];
            tot += 2.0 * pok2 * g_j * (1.0 - u_diag.re);
        }
        tot
    }

    /// Singular Y-matrix regularization: construct a KRM=2 spin group where
    /// the R-matrix nearly cancels L⁻¹ at the resonance energy, making
    /// the Y-matrix nearly singular.  The relative-epsilon regularization
    /// must produce finite, non-negative cross-sections (no NaN/Inf).
    #[test]
    fn test_rml_singular_y_matrix_regularization() {
        use nereids_endf::resonance::{ParticlePair, RmlChannel, RmlData, RmlResonance, SpinGroup};

        let pp = ParticlePair {
            ma: 1.0,
            mb: 184.0,
            za: 0.0,
            zb: 74.0,
            ia: 0.5,
            ib: 0.0,
            q: 0.0,
            pnt: 1,
            shf: 0,
            mt: 2,
            pa: 1.0,
            pb: 1.0,
        };
        let ch = RmlChannel {
            particle_pair_idx: 0,
            l: 0,
            channel_spin: 0.5,
            boundary: 0.0,
            effective_radius: 8.3,
            true_radius: 8.3,
        };
        // KRM=2: widths are reduced amplitudes γ_nc.
        // A very large reduced amplitude γ at E_r makes R ≈ γ²/(E_r - E)
        // huge near E_r, potentially cancelling L⁻¹ on the diagonal.
        let sg = SpinGroup {
            j: 0.5,
            parity: 1.0,
            channels: vec![ch],
            resonances: vec![RmlResonance {
                energy: 100.0,
                gamma_gamma: 0.0,  // KRM=2 has no implicit capture
                widths: vec![1e5], // very large reduced amplitude
            }],
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

        // Evaluate exactly at the resonance energy (worst case for singularity).
        let (tot, elas, cap, _fis) = cross_sections_for_rml_range(&rml, 100.0);
        assert!(
            tot.is_finite() && tot >= 0.0,
            "σ_total must be finite and ≥ 0 near singular Y-matrix, got {tot}"
        );
        assert!(
            elas.is_finite() && elas >= 0.0,
            "σ_elastic must be finite and ≥ 0, got {elas}"
        );
        assert!(cap.is_finite(), "σ_capture must be finite, got {cap}");
    }

    // ─── Defensive input validation at the public boundary ──────────────────
    //
    // `cross_sections_for_rml_range` is `pub`, so it can be called directly
    // from any crate without going through the Python wrapper's
    // `validate_energy_grid`.  An empty `spin_groups` vec would otherwise
    // silently return (0, 0, 0, 0) for malformed energies, hiding caller
    // bugs.  The entry assertion fires before any spin-group iteration.
    // See issue #558.

    fn make_empty_rml() -> nereids_endf::resonance::RmlData {
        nereids_endf::resonance::RmlData {
            target_spin: 0.0,
            awr: 183.0,
            scattering_radius: 8.3,
            krm: 2,
            particle_pairs: vec![],
            spin_groups: vec![],
        }
    }

    #[test]
    #[should_panic(expected = "expected positive finite energy_ev")]
    fn rml_for_range_panics_on_zero_energy() {
        let rml = make_empty_rml();
        let _ = cross_sections_for_rml_range(&rml, 0.0);
    }

    #[test]
    #[should_panic(expected = "expected positive finite energy_ev")]
    fn rml_for_range_panics_on_negative_energy() {
        let rml = make_empty_rml();
        let _ = cross_sections_for_rml_range(&rml, -1.0);
    }

    #[test]
    #[should_panic(expected = "expected positive finite energy_ev")]
    fn rml_for_range_panics_on_nan_energy() {
        let rml = make_empty_rml();
        let _ = cross_sections_for_rml_range(&rml, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "expected positive finite energy_ev")]
    fn rml_for_range_panics_on_infinite_energy() {
        let rml = make_empty_rml();
        let _ = cross_sections_for_rml_range(&rml, f64::INFINITY);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LRF=7 KRM=3 SAMMY-parity regression tests.
    //
    // These guard three SAMMY-divergence bugs that were INVISIBLE to every prior
    // RML test: all existing fixtures set effective_radius == true_radius (so the
    // APE/APT roles were indistinguishable), used all-positive resonance energies
    // (so the KRM=3 bound-state path never ran), and used PNT=1 (so the PNT flag
    // was never exercised).  The fixtures below deliberately use APE ≠ APT, a
    // negative-energy (bound) resonance, and a PNT toggle, with an explicit
    // non-vacuity guard in each test proving the fix is observable.
    //
    // Oracle: a hand re-derivation of the single-channel single-resonance elastic
    // U-matrix algebra (mirrors `spin_group_cross_sections`) using NEREIDS
    // primitives but the CORRECT SAMMY radius convention — APT (true_radius) for
    // P_c/S_c, APE (effective_radius) for the phase φ_c (SAMMY rml/mrml07.f:128-134,
    // mrml03.f:174-177,244).  Same house pattern as slbw_elastic_oracle.rs.

    /// Parameters for a single-channel elastic KRM=3 fixture.  Named fields (vs a
    /// long positional arg list) keep call sites unambiguous; `..Default::default()`
    /// lets each test override only the field it varies.
    #[derive(Clone, Copy)]
    struct ElasticKrm3 {
        l: u32,
        ape: f64,
        apt: f64,
        e_res: f64,
        gamma_n: f64,
        gamma_gamma: f64,
        pnt: i32,
        shf: i32,
    }

    impl Default for ElasticKrm3 {
        fn default() -> Self {
            Self {
                l: 0,
                ape: 8.3,
                apt: 5.0,
                e_res: 100.0,
                gamma_n: 2.0,
                gamma_gamma: 0.5,
                pnt: 1,
                shf: 0,
            }
        }
    }

    /// Build a single-channel (elastic, neutron + W-184), single-resonance KRM=3
    /// spin group with independent effective (APE) and true (APT) radii.
    fn rml_elastic_krm3(f: ElasticKrm3) -> RmlData {
        use nereids_endf::resonance::{RmlChannel, RmlResonance};
        let pp = ParticlePair {
            ma: 1.0,
            mb: 184.0,
            za: 0.0,  // neutron Z=0 → non-Coulomb (hard-sphere) entrance channel
            zb: 74.0, // W-184 Z=74
            ia: 0.5,
            ib: 0.0,
            q: 0.0,
            pnt: f.pnt,
            shf: f.shf,
            mt: 2,
            pa: 1.0,
            pb: 1.0,
        };
        let channel = RmlChannel {
            particle_pair_idx: 0,
            l: f.l,
            channel_spin: 0.5,
            boundary: 0.0,
            effective_radius: f.ape,
            true_radius: f.apt,
        };
        let sg = SpinGroup {
            j: 0.5,
            parity: 1.0,
            channels: vec![channel],
            resonances: vec![RmlResonance {
                energy: f.e_res,
                widths: vec![f.gamma_n],
                gamma_gamma: f.gamma_gamma,
            }],
            has_background_correction: false,
        };
        RmlData {
            target_spin: 0.0,
            awr: 183.0,
            scattering_radius: f.apt,
            krm: 3,
            particle_pairs: vec![pp],
            spin_groups: vec![sg],
        }
    }

    /// SAMMY Betset reduced-width amplitude for the single elastic channel,
    /// branching on PNT exactly as `mrml03.f:235-274`: PNT=1 evaluates the
    /// penetrability at |Eres−Echan| using APT; PNT≠1 uses √(½|Γ|).
    fn oracle_reduced_width(rml: &RmlData) -> f64 {
        let pp = &rml.particle_pairs[0];
        let ch = &rml.spin_groups[0].channels[0];
        let res = &rml.spin_groups[0].resonances[0];
        let g = res.widths[0];
        if pp.pnt != 1 {
            return (0.5 * g.abs()).sqrt().copysign(g);
        }
        let e_cm_n = channel::lab_to_cm_energy(res.energy, rml.awr) + pp.q;
        let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
        let k_cn = channel::wave_number_from_cm(e_cm_n.abs(), redmas);
        let p = penetrability::penetrability(ch.l, k_cn * ch.true_radius); // APT
        if p > 0.0 {
            (0.5 * g.abs() / p).sqrt().copysign(g)
        } else {
            (0.5 * g.abs()).sqrt().copysign(g)
        }
    }

    /// Single-channel single-resonance elastic U-matrix cross-section oracle for a
    /// given reduced-width amplitude `gamma`.  Mirrors `spin_group_cross_sections`:
    /// PNT=1 uses the SAMMY radius convention (APT→P/S, APE→φ); PNT≠1 uses the
    /// no-penetrability encoding (P=1, S=B_c, φ=0).  Returns
    /// (σ_total, σ_elastic, σ_capture) in barns.
    fn oracle_xs(rml: &RmlData, energy_ev: f64, gamma: f64) -> (f64, f64, f64) {
        let sg = &rml.spin_groups[0];
        let pp = &rml.particle_pairs[0];
        let ch = &sg.channels[0];
        let res = &sg.resonances[0];
        let g_j = channel::statistical_weight(sg.j, rml.target_spin);
        let pok2 = channel::pi_over_k_squared_barns(energy_ev, rml.awr);
        let (p, s, phi) = if pp.pnt != 1 {
            // No-penetrability branch (matches the evaluator's eval-path): P=1,
            // S=B_c, φ=0 → L_c = i, the Ymat(2,Ii) -= 1 encoding.
            (1.0, ch.boundary, 0.0)
        } else {
            let redmas = pp.ma * pp.mb / (pp.ma + pp.mb);
            let e_c = channel::lab_to_cm_energy(energy_ev, rml.awr) + pp.q;
            let k_c = channel::wave_number_from_cm(e_c, redmas);
            let p = penetrability::penetrability(ch.l, k_c * ch.true_radius); // APT
            let s = if pp.shf == 1 {
                penetrability::shift_factor(ch.l, k_c * ch.true_radius) // APT
            } else {
                ch.boundary
            };
            let phi = penetrability::phase_shift(ch.l, k_c * ch.effective_radius); // APE
            (p, s, phi)
        };
        let e_tilde = Complex64::new(res.energy, -res.gamma_gamma / 2.0);
        let r = Complex64::new(gamma * gamma, 0.0) / (e_tilde - energy_ev);
        let l_c = Complex64::new(s - ch.boundary, p);
        let y_tilde = l_c.inv() - r;
        let xq = y_tilde.inv() * r;
        let sqrt_p = p.sqrt();
        let xxxx = (sqrt_p / l_c) * xq * sqrt_p;
        let w = Complex64::new(1.0, 0.0) + Complex64::new(0.0, 2.0) * xxxx;
        let omega = Complex64::from_polar(1.0, -phi);
        let u = omega * w * omega;
        let tot = 2.0 * pok2 * g_j * (1.0 - u.re);
        let elas = pok2 * g_j * (Complex64::new(1.0, 0.0) - u).norm_sqr();
        (tot, elas, (tot - elas).max(0.0))
    }

    fn rml_rel_err(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Shared-primitive tolerance: the oracle and evaluator use the same P_l/φ_l
    /// primitives and identical algebra, so any disagreement is FP noise.
    const RML_ORACLE_REL_TOL: f64 = 1e-9;

    /// F2: APT (true_radius) drives P_c/S_c, APE (effective_radius) drives φ_c.
    /// The evaluator must match the correct-convention oracle; the explicit
    /// non-vacuity guard proves that swapping the two radii changes the result,
    /// so the match is meaningful and not a degenerate APE==APT pass.
    #[test]
    fn rml_lrf7_krm3_radius_roles_match_sammy_oracle() {
        // Two channel configs exercise both radius-bearing paths:
        //   l=0, shf=0 → P_c and φ_c (S_c = B_c is inert)
        //   l=1, shf=1 → P_c, S_c (analytic shift factor) and φ_c
        for &(l, shf, label) in &[(0u32, 0i32, "l=0,shf=0"), (1u32, 1i32, "l=1,shf=1")] {
            let rml = rml_elastic_krm3(ElasticKrm3 {
                l,
                shf,
                ..Default::default()
            });
            // APE/APT exchanged
            let swapped = rml_elastic_krm3(ElasticKrm3 {
                l,
                shf,
                ape: 5.0,
                apt: 8.3,
                ..Default::default()
            });
            for &e in &[50.0, 90.0, 110.0, 200.0] {
                let (tot, elas, cap, _fis) = cross_sections_for_rml_range(&rml, e);
                let gamma = oracle_reduced_width(&rml);
                let (otot, oelas, ocap) = oracle_xs(&rml, e, gamma);
                assert!(
                    rml_rel_err(tot, otot) < RML_ORACLE_REL_TOL,
                    "[{label}] σ_total @ {e} eV: eval={tot}, oracle={otot}"
                );
                assert!(
                    rml_rel_err(elas, oelas) < RML_ORACLE_REL_TOL,
                    "[{label}] σ_elastic @ {e} eV: eval={elas}, oracle={oelas}"
                );
                assert!(
                    rml_rel_err(cap, ocap) < RML_ORACLE_REL_TOL,
                    "[{label}] σ_capture @ {e} eV: eval={cap}, oracle={ocap}"
                );
                // Non-vacuity: the swapped-radius convention must give a materially
                // different σ_total, so matching the correct one above is meaningful.
                // For l=1,shf=1 the swap moves the shift-factor radius too.
                let g_sw = oracle_reduced_width(&swapped);
                let (stot, _, _) = oracle_xs(&swapped, e, g_sw);
                assert!(
                    rml_rel_err(otot, stot) > 0.01,
                    "[{label}] APE/APT swap unobservable at {e} eV (otot={otot}, swapped={stot}); test would be vacuous"
                );
            }
        }
    }

    /// F1: a KRM=3 bound-state resonance (E_res < 0) must receive a real
    /// penetrability at |Eres−Echan| and the √(½|Γ|/P) normalisation — NOT the
    /// old √|Γ| fallback.  The non-vacuity guard asserts the evaluator does NOT
    /// match the buggy √|Γ| reduced width.
    #[test]
    fn rml_lrf7_krm3_bound_state_uses_penetrability_not_sqrt_gamma() {
        let rml = rml_elastic_krm3(ElasticKrm3 {
            e_res: -50.0,
            ..Default::default()
        });
        let gamma = oracle_reduced_width(&rml);
        // Old (buggy) behaviour: γ = √|Γ| with no ½ factor and no penetrability.
        let gamma_buggy = {
            let g = rml.spin_groups[0].resonances[0].widths[0];
            g.abs().sqrt().copysign(g)
        };
        for &e in &[10.0, 50.0, 150.0] {
            let (tot, elas, cap, _fis) = cross_sections_for_rml_range(&rml, e);
            let (otot, oelas, ocap) = oracle_xs(&rml, e, gamma);
            assert!(
                rml_rel_err(tot, otot) < RML_ORACLE_REL_TOL,
                "bound σ_total @ {e} eV: eval={tot}, oracle={otot}"
            );
            assert!(
                rml_rel_err(elas, oelas) < RML_ORACLE_REL_TOL,
                "bound σ_elastic @ {e} eV: eval={elas}, oracle={oelas}"
            );
            assert!(
                rml_rel_err(cap, ocap) < RML_ORACLE_REL_TOL,
                "bound σ_capture @ {e} eV: eval={cap}, oracle={ocap}"
            );
            // Non-vacuity: the pre-fix √|Γ| reduced width gives a materially
            // different result, so the match above proves the new normalisation.
            let (btot, _, _) = oracle_xs(&rml, e, gamma_buggy);
            assert!(
                rml_rel_err(otot, btot) > 0.01,
                "√|Γ| vs √(½|Γ|/P) indistinguishable at {e} eV (otot={otot}, buggy={btot}); test would be vacuous"
            );
        }
    }

    /// F3: the per-pair PNT flag must change the physics.  Before the fix the
    /// evaluator branched on particle mass and ignored PNT entirely, so toggling
    /// PNT on a massive channel produced identical cross-sections.
    #[test]
    fn rml_lrf7_pnt_flag_is_respected() {
        let with_pen = rml_elastic_krm3(ElasticKrm3::default());
        let no_pen = rml_elastic_krm3(ElasticKrm3 {
            pnt: 0,
            ..Default::default()
        });
        for &e in &[50.0, 110.0, 200.0] {
            let (tot1, _, _, _) = cross_sections_for_rml_range(&with_pen, e);
            let (tot0, elas0, cap0, _) = cross_sections_for_rml_range(&no_pen, e);
            assert!(
                tot0.is_finite() && tot1.is_finite(),
                "PNT toggle produced non-finite σ at {e} eV: PNT=1 {tot1}, PNT=0 {tot0}"
            );
            // Pin the PNT=0 path against the oracle: this covers BOTH the
            // no-penetrability channel setup (P=1, S=B, φ=0) AND the √(½|Γ|)
            // width conversion, so the test fails if either regresses — not just
            // if the totals happen to differ.
            let g0 = oracle_reduced_width(&no_pen);
            let (otot0, oelas0, ocap0) = oracle_xs(&no_pen, e, g0);
            assert!(
                rml_rel_err(tot0, otot0) < RML_ORACLE_REL_TOL,
                "PNT=0 σ_total @ {e} eV: eval={tot0}, oracle={otot0}"
            );
            assert!(
                rml_rel_err(elas0, oelas0) < RML_ORACLE_REL_TOL,
                "PNT=0 σ_elastic @ {e} eV: eval={elas0}, oracle={oelas0}"
            );
            assert!(
                rml_rel_err(cap0, ocap0) < RML_ORACLE_REL_TOL,
                "PNT=0 σ_capture @ {e} eV: eval={cap0}, oracle={ocap0}"
            );
            // Non-vacuity: the PNT toggle must change σ (pre-fix, branching on
            // particle mass, gave identical results regardless of PNT).
            assert!(
                rml_rel_err(tot1, tot0) > 0.01,
                "PNT flag ignored at {e} eV: PNT=1 σ_total={tot1}, PNT=0 σ_total={tot0} (should differ)"
            );
        }
    }
}
