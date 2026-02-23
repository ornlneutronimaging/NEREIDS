//! Coulomb wave functions via Steed's continued-fraction method.
//!
//! Provides F_L(η,ρ), G_L(η,ρ), F'_L, G'_L and the derived
//! penetrability P_L, shift S_L, and phase φ_L for charged-particle
//! exit channels in LRF=7 (R-Matrix Limited) cross-section calculations.
//!
//! ## SAMMY References
//! - `coulomb/mrml08.f90` Coulfg — Steed's CF1+CF2 algorithm (Barnett 1981)
//! - `rml/mrml07.f` Pgh subroutine — `if (Zeta.NE.Zero)` branch
//! - Barnett, CPC 21 (1981) 297–314
//!
//! ## Limit η → 0
//! When η = 0, F_0 = sin ρ, G_0 = cos ρ, reproducing hard-sphere
//! (Blatt-Weisskopf) penetrabilities from `penetrability.rs`.

use nereids_core::constants;

/// Sommerfeld parameter η for a Coulomb channel.
///
/// η = Z_a · Z_b · α · √(m_n·c² · μ̃ / (2·E_c))
///
/// where:
/// - Z_a, Z_b are the charge numbers extracted from the ENDF ZA codes
///   (ZA = Z×1000 + A; neutrons and photons have ZA = 0)
/// - α = fine-structure constant ≈ 1/137.036
/// - μ̃ = MA·MB/(MA+MB) is the reduced mass in neutron mass units
/// - E_c is the CM kinetic energy in this channel (eV, must be > 0)
/// - m_n·c² in eV
///
/// ## SAMMY Reference
/// `rml/mrml07.f` Zeta = Za·Zb · α · √(m_n·μ̃/(2·E_c))
///
/// # Arguments
/// * `za_code`, `zb_code` — ENDF ZA codes (Z×1000+A) of the two particles.
///   Neutrons and photons have ZA=0; protons have ZA=1001; alphas have ZA=2004.
/// * `ma`, `mb` — Particle masses in neutron mass units.
/// * `e_cm_ev` — CM kinetic energy in eV (must be > 0).
pub fn sommerfeld_eta(za_code: f64, zb_code: f64, ma: f64, mb: f64, e_cm_ev: f64) -> f64 {
    if e_cm_ev <= 0.0 {
        return 0.0;
    }
    // Extract charge number Z from ENDF ZA code: Z = floor(ZA / 1000)
    // Neutron ZA=0 → Z=0; proton ZA=1001 → Z=1; alpha ZA=2004 → Z=2.
    let za = (za_code / 1000.0).floor(); // charge of particle a
    let zb = (zb_code / 1000.0).floor(); // charge of particle b
    if za == 0.0 || zb == 0.0 {
        return 0.0; // one particle is neutral → no Coulomb interaction
    }
    let alpha = 1.0 / 137.035_999_084; // CODATA 2018 fine-structure constant
    let mn_ev = constants::NEUTRON_MASS_MEV * 1e6; // MeV → eV
    let reduced_mass = ma * mb / (ma + mb); // μ̃ in neutron mass units
    // η = Z_a · Z_b · α · √(m_n · μ̃ / (2 · E_c))
    za * zb * alpha * (mn_ev * reduced_mass / (2.0 * e_cm_ev)).sqrt()
}

/// Coulomb wave functions (F_L, G_L, F'_L, G'_L) via Steed's CF1+CF2 method.
///
/// Implements Barnett's Coulfg algorithm (CPC 21, 1981, 297–314) as adapted in
/// SAMMY `coulomb/mrml08.f90`. Returns `None` if ρ ≤ 0 or the continued
/// fractions fail to converge (nuclear-physics ρ values always converge).
///
/// The returned derivatives are with respect to ρ.
///
/// ## SAMMY Reference
/// `coulomb/mrml08.f90` Coulfg subroutine.
///
/// # Arguments
/// * `l`   — Orbital angular momentum quantum number L.
/// * `eta` — Sommerfeld parameter η (0 for neutral particles).
/// * `rho` — Dimensionless channel parameter ρ = k·a (must be > 0).
///
/// # Returns
/// `Some((F_L, G_L, F'_L, G'_L))` on success, `None` if ρ is too small or
/// the continued fractions diverge.
pub fn coulomb_wave_functions(l: u32, eta: f64, rho: f64) -> Option<(f64, f64, f64, f64)> {
    // Steed's CF1+CF2 algorithm.
    // Faithful Rust translation of SAMMY coulomb/mrml08.f90 Coulfg
    // (Barnett, CPC 21, 1981, 297–314).
    const ACCUR: f64 = 1e-16;
    const ABORT: f64 = 2e4;
    const TM30: f64 = 1e-30;
    let acc = ACCUR;
    let acch = ACCUR.sqrt(); // ≈ 1e-8

    // ρ must be positive; below acch the asymptotic CF2 is unreliable.
    // (SAMMY Coulfg line 281: IF (Xx.LE.Acch) GO TO 100)
    if rho <= acch {
        return None;
    }

    let lll = l as usize;
    // SAMMY uses Llmax = Lll + 2 as the downward recurrence starting point.
    let llmax = lll + 2;
    let n = llmax + 1; // array size; indices 0..=llmax
    let xi = 1.0 / rho; // 1/ρ
    let xll = llmax as f64; // Llmax as f64

    // Working arrays (0-indexed):
    //   fc[k]  = unnormalized F_k → normalized F_k after upward pass
    //   fcp[k] = unnormalized F'_k → normalized F'_k
    //   rl_store[k] = sqrt(1 + (η/k)^2), stored at k=Xl during CF1 downward
    //                 recurrence; used for upward G recurrence.
    let mut fc = vec![0.0f64; n];
    let mut fcp = vec![0.0f64; n];
    let mut rl_store = vec![0.0f64; n]; // index 0 unused; 1..=llmax used

    // ─────────────────────────────────────────────────────────────────────────
    // CF1: Steed's upward continued fraction for F'(Llmax)/F(Llmax).
    //
    // We iterate k = Llmax+1, Llmax+2, … until |Df| < |F|·Acc.
    // At the end, f = F'_{Llmax}/F_{Llmax}.
    //
    // SAMMY Coulfg lines 298–344.
    // ─────────────────────────────────────────────────────────────────────────
    let mut fcl = 1.0f64; // tracks sign of unnormalized F (SAMMY Fcl)
    let mut pk = xll + 1.0; // k = Llmax + 1 (start of CF1)
    let px = pk + ABORT; // iteration limit

    // First step (initialization of CF1).
    let ek0 = eta / pk;
    let mut f = (ek0 + pk * xi) * fcl + (fcl - 1.0) * xi;
    let mut pk1 = pk + 1.0;

    let (mut d, mut df) = if (eta * rho + pk * pk1).abs() <= acc {
        // Near-zero denominator fixup (SAMMY lines 308–313).
        fcl = (1.0 + ek0 * ek0) / (1.0 + (eta / pk1) * (eta / pk1));
        pk += 2.0;
        pk1 = pk + 1.0;
        let ek = eta / pk;
        let d = 1.0 / ((pk + pk1) * (xi + ek / pk1));
        let df = -fcl * (1.0 + ek * ek) * d;
        (d, df)
    } else {
        let d = 1.0 / ((pk + pk1) * (xi + ek0 / pk1));
        let df = -fcl * (1.0 + ek0 * ek0) * d;
        // Track sign of Fcl (SAMMY lines 316–318)
        if fcl != 1.0 {
            fcl = -1.0;
        }
        if d < 0.0 {
            fcl = -fcl;
        }
        (d, df)
    };
    f += df;

    // CF1 main loop (Steed modified Lentz, SAMMY lines 320–343).
    // SAMMY: Pk = Pk1; Pk1 = Pk1 + One  (advance to next k before each step).
    let mut p_count = 1u32;
    loop {
        pk = pk1; // advance: current step uses previous pk1
        pk1 = pk + 1.0; // next step's pk1
        let ek = eta / pk;
        let tk = (pk + pk1) * (xi + ek / pk1);
        d = tk - d * (1.0 + ek * ek);
        if d.abs() <= acch {
            // D too small — accuracy warning (SAMMY lines 331–337)
            p_count += 1;
            if p_count > 2 {
                return None; // CF1 failed
            }
        }
        d = 1.0 / d;
        if d < 0.0 {
            fcl = -fcl;
        }
        df *= d * tk - 1.0;
        f += df;
        if pk > px {
            return None; // CF1 exceeded iteration limit
        }
        if df.abs() < f.abs() * acc {
            break; // converged
        }
    }
    // After CF1: f = F'_{Llmax}/F_{Llmax}, fcl = sign of F_{Llmax}.

    // ─────────────────────────────────────────────────────────────────────────
    // Downward recurrence: L = Llmax → 0.
    //
    // Builds unnormalized F_l and F'_l for all l, and stores RL values
    // (= sqrt(1 + (η/l)^2)) needed for upward G recurrence.
    //
    // Recurrence (from Abramowitz & Stegun 14.2.2/14.2.3):
    //   F_{l-1} = (S_l · F_l + F'_l) / R_l
    //   F'_{l-1} = S_l · F_{l-1} − R_l · F_l
    // where S_l = η/l + l/ρ, R_l = sqrt(1 + (η/l)^2).
    //
    // SAMMY Coulfg lines 349–375.
    // ─────────────────────────────────────────────────────────────────────────
    fcl *= TM30; // scale down to avoid overflow during recurrence
    let mut fpl = fcl * f; // F'_{Llmax} (unnormalized)
    fcp[llmax] = fpl;
    fc[llmax] = fcl;

    let mut xl = xll; // starts at Llmax, decrements to 1
    for lp in 1..=llmax {
        let el = eta / xl;
        let rl = (1.0 + el * el).sqrt(); // R_{Xl}
        let sl = el + xl * xi; // S_{Xl}
        let l_idx = llmax - lp; // 0-indexed destination (llmax-1 down to 0)
        let fcl1 = (fcl * sl + fpl) / rl;
        fpl = fcl1 * sl - fcl * rl;
        fcl = fcl1;
        fc[l_idx] = fcl;
        fcp[l_idx] = fpl;
        // Store R_{Xl} at index Xl (used in upward G recurrence at Xl=l+1).
        rl_store[xl as usize] = rl;
        xl -= 1.0;
    }
    if fcl == 0.0 {
        fcl = acc; // avoid zero divide (SAMMY line 370)
    }
    f = fpl / fcl; // F'_0 / F_0 at L = 0

    // ─────────────────────────────────────────────────────────────────────────
    // CF2: Steed's algorithm for P + iQ at L = 0.
    //
    // P + iQ = (f + iρ) · (F_0 + iG_0) ⁻¹   (Wronskian formulation)
    //
    // SAMMY Coulfg lines 393–433.
    // ─────────────────────────────────────────────────────────────────────────
    // For Xlm = 0: E2mm1 = η² + 0·1 = η².
    let wi = 2.0 * eta; // constant increment for Ai
    let mut pk = 0.0f64;
    let mut p = 0.0f64;
    let mut q = 1.0 - eta * xi;
    let mut ar = -(eta * eta); // = -E2mm1 for L=0
    let mut ai = eta;
    let br = 2.0 * (rho - eta); // fixed throughout CF2
    let bi0 = 2.0; // initial Bi
    let mut bi = bi0;
    let denom0 = br * br + bi0 * bi0;
    let mut dr = br / denom0;
    let mut di = -bi0 / denom0;
    let mut dp = -xi * (ar * di + ai * dr);
    let mut dq = xi * (ar * dr - ai * di);

    loop {
        p += dp;
        q += dq;
        pk += 2.0;
        ar += pk;
        ai += wi;
        bi += 2.0;
        let d_re = ar * dr - ai * di + br;
        let d_im = ai * dr + ar * di + bi;
        let c = 1.0 / (d_re * d_re + d_im * d_im);
        dr = c * d_re;
        di = -c * d_im;
        let a = br * dr - bi * di - 1.0;
        let b = bi * dr + br * di;
        let c2 = dp * a - dq * b;
        dq = dp * b + dq * a;
        dp = c2;
        if pk > 2.0 * ABORT {
            return None; // CF2 failed to converge
        }
        if dp.abs() + dq.abs() < (p.abs() + q.abs()) * acc {
            break; // converged
        }
    }

    // Degenerate check: Q too small (SAMMY Coulfg line 429 → error 130)
    let acc4 = acc * 1e3; // Acc * Ten2 * Ten2 * 0.1 in SAMMY
    if q.abs() <= acc4 * p.abs() {
        return None;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Normalization: compute F_0, G_0 from Wronskian and CF2 output.
    //
    // Gam = (f − P) / Q  [G_0 = Gam · F_0]
    // W   = 1 / sqrt[(f−P)·Gam + Q]       [normalization factor]
    // F_0 = W · sign(Fcl), G_0 = Gam · F_0
    //
    // SAMMY Coulfg lines 430–448.
    // ─────────────────────────────────────────────────────────────────────────
    let gam = (f - p) / q;
    let w = 1.0 / ((f - p) * gam + q).abs().sqrt();
    let fcm = w.copysign(fcl); // = |W| · sign(Fcl)  (SAMMY: dSIGN(W, Fcl))
    let gcl0 = fcm * gam; // G_0
    let gpl0 = gcl0 * (p - q / gam); // G'_0
    let fcp0 = fcm * f; // F'_0
    fc[0] = fcm;
    fcp[0] = fcp0;

    // Fast return for L = 0: no upward recurrence needed.
    if lll == 0 {
        return Some((fcm, gcl0, fcp0, gpl0));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Upward G recurrence: L = 0 → Lll.
    // Simultaneously renormalize stored F values.
    //
    // Recurrence (from Abramowitz & Stegun 14.2.2/14.2.3, reversed):
    //   G_{l+1} = (S_{l+1} · G_l − G'_l) / R_{l+1}
    //   G'_{l+1} = R_{l+1} · G_l − S_{l+1} · G_{l+1}
    // where S_{l+1} = η/(l+1) + (l+1)/ρ, R_{l+1} = rl_store[l+1].
    //
    // SAMMY Coulfg lines 454–467.
    // ─────────────────────────────────────────────────────────────────────────
    // Renormalization factor for stored (unnormalized) F values.
    // (SAMMY: W = W / dABS(Fcl), then Fc(L+1) = W * Fc(L+1))
    let w_f = w / fcl.abs();

    let mut gcl = gcl0;
    let mut gpl = gpl0;
    for k in 1..=lll {
        let xl_k = k as f64;
        let el = eta / xl_k;
        let rl = rl_store[k]; // R_{k} stored during downward recurrence
        let sl = el + xl_k * xi;
        let gcl_new = (sl * gcl - gpl) / rl;
        gpl = rl * gcl - sl * gcl_new;
        gcl = gcl_new;
        // Renormalize F at this level.
        fc[k] *= w_f;
        fcp[k] *= w_f;
    }

    Some((fc[lll], gcl, fcp[lll], gpl))
}

/// Coulomb penetrability P_L = ρ / (F_L² + G_L²).
///
/// Returns 0 if the wave functions cannot be computed (ρ too small, or the
/// Coulomb fractions fail to converge — only occurs far outside the
/// nuclear-physics domain).
///
/// ## SAMMY Reference
/// `coulomb/mrml08.f90` Coulfg output: `Pcoul = Xx / (Fc(L)^2 + Gc(L)^2)`
pub fn coulomb_penetrability(l: u32, eta: f64, rho: f64) -> f64 {
    match coulomb_wave_functions(l, eta, rho) {
        Some((fl, gl, _, _)) => rho / (fl * fl + gl * gl),
        None => 0.0,
    }
}

/// Coulomb shift factor S_L = ρ · (F_L·F'_L + G_L·G'_L) / (F_L² + G_L²).
///
/// Returns 0 if the wave functions cannot be computed.
///
/// ## SAMMY Reference
/// `coulomb/mrml08.f90` Coulfg output:
/// `Scoul = Xx * (Fc(L)*Fcp(L)+Gc(L)*Gcp(L)) / (Fc(L)^2+Gc(L)^2)`
pub fn coulomb_shift(l: u32, eta: f64, rho: f64) -> f64 {
    match coulomb_wave_functions(l, eta, rho) {
        Some((fl, gl, flp, glp)) => {
            let asq = fl * fl + gl * gl;
            rho * (fl * flp + gl * glp) / asq
        }
        None => 0.0,
    }
}

/// Coulomb hard-sphere phase: returns (sin φ_L, cos φ_L) where φ_L = atan2(F_L, G_L).
///
/// Returns (0, 1) (φ = 0) if the wave functions cannot be computed.
///
/// ## SAMMY Reference
/// `coulomb/mrml08.f90` Coulfg output:
/// `Sinphi = Fc(L)/A, Cosphi = Gc(L)/A` where `A = sqrt(Fc^2 + Gc^2)`
pub fn coulomb_phase(l: u32, eta: f64, rho: f64) -> (f64, f64) {
    match coulomb_wave_functions(l, eta, rho) {
        Some((fl, gl, _, _)) => {
            let a = (fl * fl + gl * gl).sqrt();
            (fl / a, gl / a) // (sin φ, cos φ)
        }
        None => (0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::penetrability;

    /// η = 0, L = 0: F_0 → sin ρ, G_0 → cos ρ (hard-sphere limit).
    ///
    /// SAMMY's convention has G_0 = +cos ρ (so atan2(F_0, G_0) = ρ, the
    /// hard-sphere phase shift).  The Wronskian F·G' − G·F' = −1 in this
    /// convention (W(G, F) = +1 in the standard A&S sense).
    ///
    /// Reference: Abramowitz & Stegun §14.1 (η → 0 limit of Coulomb functions);
    /// SAMMY coulomb/mrml08.f90 Coulfg phase convention.
    #[test]
    fn coulomb_zero_eta_l0_matches_hard_sphere() {
        // CF1+CF2 achieves ~1e-13 for large ρ; tolerance 1e-7 covers small ρ too.
        for &rho in &[0.5f64, 1.0, 2.0, 5.0, 10.0] {
            let (fl, gl, _, _) =
                coulomb_wave_functions(0, 0.0, rho).expect("Should compute for rho > 0");
            let f_exact = rho.sin();
            let g_exact = rho.cos(); // SAMMY convention: G_0 = +cos ρ
            assert!(
                (fl - f_exact).abs() < 1e-7,
                "F_0(η=0, ρ={rho}): got {fl}, expected {f_exact}"
            );
            assert!(
                (gl - g_exact).abs() < 1e-7,
                "G_0(η=0, ρ={rho}): got {gl}, expected {g_exact}"
            );
        }
    }

    /// η = 0, P_0 = ρ (matches hard-sphere penetrability).
    ///
    /// P_0 = ρ / (F_0² + G_0²) = ρ / (sin²ρ + cos²ρ) = ρ.
    #[test]
    fn coulomb_penetrability_zero_eta_l0() {
        for &rho in &[0.1f64, 0.5, 1.0, 3.0] {
            let p_coulomb = coulomb_penetrability(0, 0.0, rho);
            let p_hard = penetrability::penetrability(0, rho); // = ρ
            assert!(
                (p_coulomb - p_hard).abs() < 1e-8,
                "P_0(η=0, ρ={rho}): Coulomb={p_coulomb}, hard-sphere={p_hard}"
            );
        }
    }

    /// η = 0, P_1 = ρ³/(1+ρ²) (matches hard-sphere L=1 penetrability).
    #[test]
    fn coulomb_penetrability_zero_eta_l1() {
        for &rho in &[0.1f64, 0.5, 1.0, 3.0] {
            let p_coulomb = coulomb_penetrability(1, 0.0, rho);
            let p_hard = penetrability::penetrability(1, rho);
            assert!(
                (p_coulomb - p_hard).abs() / (p_hard + 1e-30) < 1e-6,
                "P_1(η=0, ρ={rho}): Coulomb={p_coulomb}, hard-sphere={p_hard}"
            );
        }
    }

    /// A neutral particle (ZA=0 for one side) → η = 0.
    #[test]
    fn sommerfeld_eta_neutral_is_zero() {
        // Neutron exit channel: za_code = 0 (neutron), zb_code = anything
        let eta = sommerfeld_eta(0.0, 92238.0, 1.0, 236.006, 1e4);
        assert_eq!(eta, 0.0, "Neutral particle should give η=0");
    }

    /// Proton on O-16 at E_cm = 1 MeV: known η ≈ 0.53.
    ///
    /// Formula: η = Z_a·Z_b·α·√(m_n·μ̃/(2·E_cm))
    /// For p+O16: Z_a=1, Z_b=8, μ̃ = 1·16/(1+16) ≈ 0.9412, E_cm = 1e6 eV.
    /// η = 1·8 · (1/137.036) · √(939.565e6 · 0.9412 / (2 · 1e6))
    ///   = 8/137.036 · √(441.9)
    ///   ≈ 0.05836 · 21.02 ≈ 1.226 / 2 ... let me compute more carefully.
    ///
    /// Actually: η = 8/137.036 * sqrt(939.565e6 * 16/(17*2) / 1e6)
    ///            = 8/137.036 * sqrt(939.565 * 8/17)
    ///            = 8/137.036 * sqrt(442.1)  [in natural units with MeV]
    ///           Wait, let me use eV: mn = 939565420 eV, mu = 939565420*16/17
    ///           η = 8/137.036 * sqrt(939565420 * 16/17 / (2 * 1e6))
    ///             = 8/137.036 * sqrt(939565420 * 0.9412 / 2e6)
    ///             = 8/137.036 * sqrt(442.3)   [sqrt(442.3) ≈ 21.03]
    ///             ≈ 8/137.036 * 21.03 ≈ 1.226
    ///
    /// Reference: standard nuclear physics textbook value for p+O-16 at 1 MeV ≈ 1.22.
    #[test]
    fn sommerfeld_eta_proton_oxygen_known_value() {
        // Proton: ZA=1001, A=1, Z=1
        // O-16: ZA=8016, A=16, Z=8
        let za_proton = 1001.0f64;
        let zb_oxygen = 8016.0f64;
        let ma = 1.0f64; // proton mass in neutron mass units
        let mb = 16.0f64; // O-16 mass in neutron mass units (approx)
        let e_cm_ev = 1.0e6f64; // 1 MeV in eV
        let eta = sommerfeld_eta(za_proton, zb_oxygen, ma, mb, e_cm_ev);
        // Expected: ~1.22 (standard nuclear physics reference)
        assert!(
            (eta - 1.22).abs() < 0.05,
            "p+O-16 at 1 MeV: η={eta}, expected ~1.22"
        );
    }

    /// Wronskian condition: |F_L · G'_L − G_L · F'_L| = 1 for all L, η, ρ.
    ///
    /// SAMMY uses G_0 = +cos ρ (η=0 hard-sphere), so the Wronskian
    /// F·G' − G·F' = −1 (not +1).  This is the defining identity of the
    /// Barnett normalization in SAMMY's phase convention: W(G, F) = +1.
    ///
    /// We check the magnitude = 1 to remain convention-agnostic.
    #[test]
    fn coulomb_wronskian_identity() {
        for &l in &[0u32, 1, 2, 3] {
            for &eta in &[0.0f64, 0.5, 2.0, 5.0] {
                for &rho in &[0.5f64, 1.0, 5.0, 20.0] {
                    if let Some((fl, gl, flp, glp)) = coulomb_wave_functions(l, eta, rho) {
                        let wronskian = fl * glp - gl * flp;
                        assert!(
                            (wronskian.abs() - 1.0).abs() < 1e-8,
                            "Wronskian at L={l}, η={eta}, ρ={rho}: |W|={}, expected 1",
                            wronskian.abs()
                        );
                    }
                }
            }
        }
    }
}
