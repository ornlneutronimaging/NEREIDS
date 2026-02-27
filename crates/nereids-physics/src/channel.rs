//! Neutron channel calculations: wave number, ρ parameter, statistical weights.
//!
//! Computes channel-level quantities needed for R-matrix cross-section
//! evaluation: the neutron wave number k(E), channel parameter ρ = k·a,
//! and spin statistical weight factors g_J.
//!
//! ## SAMMY Reference
//! - `rml/mrml03.f` Fxradi: wave number and radius computations
//! - SAMMY manual Section 2 (kinematics)
//!
//! ## Units
//! All quantities in natural units: energies in eV, lengths in fm.
//!
//! ## Key Relations
//! - k = √(2·μ·E_cm) / ℏc = [AWR/(1+AWR)] × √(2·m_n·E_lab) / ℏc
//! - ρ = k·a, where a = channel radius
//! - g_J = (2J+1) / ((2I+1)·(2s+1))

use nereids_core::constants;

/// Neutron wave number squared k² in fm⁻².
///
/// k² = 2·μ·E_cm / (ℏc)² = 2·m_n·[AWR/(1+AWR)]²·E_lab / (ℏc)²
///
/// where μ = m_n · AWR / (1 + AWR) is the reduced mass,
/// E_cm = E_lab · AWR / (1 + AWR) is the center-of-mass energy,
/// and the two factors of AWR/(1+AWR) come from μ and E_cm respectively.
///
/// ## SAMMY Reference
/// - `rml/mrml03.f` Fxradi: Zke = Twomhb × √(Redmas × Factor)
///   where Redmas = AWR/(1+AWR) and Factor = AWR/(1+AWR).
///
/// # Arguments
/// * `energy_ev` — Neutron energy in eV (lab frame, as stored in ENDF).
/// * `awr` — Ratio of target mass to neutron mass (from ENDF).
pub fn k_squared(energy_ev: f64, awr: f64) -> f64 {
    // Guard: bound-state resonances have negative E_r in ENDF, but k² is
    // evaluated at the lab energy which is always positive.  If a caller
    // passes E ≤ 0 (e.g. during extrapolation or misuse), return 0.0 to
    // prevent a negative k² from propagating NaN through sqrt downstream.
    if energy_ev <= 0.0 {
        return 0.0;
    }

    // μ/m_n = AWR / (1 + AWR)
    let mass_ratio = awr / (1.0 + awr);

    // k² = 2 · m_n · (AWR/(1+AWR))² · E_lab / (ℏc)²
    // One factor from reduced mass, one from lab→CM energy conversion.
    let mn_ev = constants::NEUTRON_MASS_MEV * 1e6; // MeV → eV
    let hbar_c = constants::HBAR_EV_S * constants::SPEED_OF_LIGHT * 1e15; // eV·fm

    2.0 * mn_ev * mass_ratio * mass_ratio * energy_ev / (hbar_c * hbar_c)
}

/// Neutron wave number k in fm⁻¹.
///
/// # Arguments
/// * `energy_ev` — Neutron energy in eV (must be positive).
/// * `awr` — Ratio of target mass to neutron mass.
pub fn wave_number(energy_ev: f64, awr: f64) -> f64 {
    k_squared(energy_ev, awr).sqrt()
}

/// Wave number k_c from reduced mass ratio and center-of-mass kinetic energy.
///
/// k_c = √(2·m_n·μ·E_cm) / ℏc
///
/// Used for non-elastic channels in LRF=7 (R-Matrix Limited) where each
/// channel may have a different particle pair (masses, Q-value).
///
/// For the elastic channel (MA=1 neutron, MB=AWR target):
///   μ = AWR/(1+AWR),  E_cm = E_lab·AWR/(1+AWR)
///   → k = `wave_number(E_lab, AWR)` (algebraically identical).
///
/// ## SAMMY Reference
/// - `rml/mrml03.f` Fxradi: `Zke = Twomhb × √(Redmas × Factor)`
///   where Redmas = MA·MB/(MA+MB) and Factor = E_cm for the channel.
///
/// # Arguments
/// * `e_cm` — Center-of-mass kinetic energy in this channel (eV). Must be ≥ 0.
/// * `reduced_mass_ratio` — μ = MA·MB/(MA+MB) in neutron mass units.
pub fn wave_number_from_cm(e_cm: f64, reduced_mass_ratio: f64) -> f64 {
    // Guard against non-positive CM energies, which can arise from numerical noise
    // near threshold energies. Zero wave number means no contribution from this channel.
    // Uses `<= 0.0` to match `k_squared` (line 41), which also catches zero.
    if e_cm <= 0.0 {
        return 0.0;
    }
    let mn_ev = constants::NEUTRON_MASS_MEV * 1e6; // MeV → eV
    let hbar_c = constants::HBAR_EV_S * constants::SPEED_OF_LIGHT * 1e15; // eV·fm
    (2.0 * mn_ev * reduced_mass_ratio * e_cm / (hbar_c * hbar_c)).sqrt()
}

/// Channel parameter ρ = k·a.
///
/// # Arguments
/// * `energy_ev` — Neutron energy in eV.
/// * `awr` — Mass ratio from ENDF.
/// * `channel_radius_fm` — Channel radius in fm.
pub fn rho(energy_ev: f64, awr: f64, channel_radius_fm: f64) -> f64 {
    wave_number(energy_ev, awr) * channel_radius_fm
}

/// Spin statistical weight factor g_J.
///
///   g_J = (2J + 1) / ((2I + 1) · (2s + 1))
///
/// where I = target spin, s = neutron spin (1/2), J = total angular momentum.
///
/// # Arguments
/// * `j_total` — Total angular momentum J.
/// * `target_spin` — Target nucleus spin I.
pub fn statistical_weight(j_total: f64, target_spin: f64) -> f64 {
    let neutron_spin = 0.5;
    (2.0 * j_total + 1.0) / ((2.0 * target_spin + 1.0) * (2.0 * neutron_spin + 1.0))
}

/// Convert between lab and center-of-mass energy.
///
/// E_cm = E_lab · AWR / (1 + AWR)
pub fn lab_to_cm_energy(energy_lab: f64, awr: f64) -> f64 {
    energy_lab * awr / (1.0 + awr)
}

/// π/k² factor that appears in all cross-section formulas.
///
/// Returns π/k² in barns (1 barn = 100 fm²).
///
/// # Arguments
/// * `energy_ev` — Neutron energy in eV.
/// * `awr` — Mass ratio from ENDF.
pub fn pi_over_k_squared_barns(energy_ev: f64, awr: f64) -> f64 {
    // Catch negative energies in debug builds: k_squared(-1.0) returns 0.0,
    // but pi_over_k_squared_barns would silently map -1.0 → 1e-20 via the
    // floor, producing a finite (wrong) result instead of the expected
    // infinity/zero. Callers should never pass negative energy here.
    //
    // Note on debug_assert vs assert: this fires only in debug builds.
    // In release builds, a non-positive energy is silently clamped to 1e-20
    // by the floor below, yielding a large but finite π/k² (~10¹⁸ barns).
    // This is acceptable because:
    //   1. All call sites guarantee positive energies (ENDF grids, fitting
    //      bounds, transmission forward model).
    //   2. The 1e-20 floor is defense-in-depth: it prevents division by zero
    //      and NaN propagation without paying for a branch in hot loops.
    //   3. A full `assert!` would add a branch to the hottest inner loop of
    //      cross-section evaluation for a condition that cannot occur in
    //      correct usage.
    debug_assert!(
        energy_ev > 0.0,
        "pi_over_k_squared_barns called with non-positive energy: {energy_ev}"
    );
    // Apply a tiny energy floor (1e-20 eV) so that k² is never zero.
    // This preserves the correct 1/E functional form at very low energies
    // instead of returning an arbitrary sentinel value that could mislead
    // a fitting optimizer.
    let e_safe = energy_ev.max(1e-20);
    let k2 = k_squared(e_safe, awr);
    // π/k² in fm², convert to barns (1 barn = 100 fm²)
    std::f64::consts::PI / k2 / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_number_u238() {
        // U-238: AWR ≈ 236.006
        // k = sqrt(2 m_n / (ℏc)²) × AWR/(1+AWR) × sqrt(E)
        // At E = 1 eV: k_free = 2.197e-4 fm⁻¹, mass_ratio = 236.006/237.006 = 0.99578
        // k = 2.197e-4 × 0.99578 ≈ 2.188e-4 fm⁻¹
        let awr = 236.006;
        let k = wave_number(1.0, awr);
        assert!(
            (k - 2.188e-4).abs() < 1e-6,
            "k = {} fm⁻¹, expected ~2.188e-4",
            k
        );
    }

    #[test]
    fn test_rho_u238() {
        // U-238 at 6.674 eV, channel radius 9.4285 fm
        // ρ = k·a ≈ 2.188e-4 × √6.674 × 9.4285 ≈ 5.33e-3
        let awr = 236.006;
        let r = rho(6.674, awr, 9.4285);
        assert!((r - 5.33e-3).abs() < 1e-4, "ρ = {}, expected ~5.33e-3", r);
    }

    #[test]
    fn test_statistical_weight() {
        // U-238: I=0, s=1/2, J=1/2 → g = 2/2 = 1.0
        assert!((statistical_weight(0.5, 0.0) - 1.0).abs() < 1e-15);
        // J=3/2, I=0 → g = 4/2 = 2.0
        assert!((statistical_weight(1.5, 0.0) - 2.0).abs() < 1e-15);
        // Ta-181: I=7/2, J=3 → g = 7/(8×2) = 7/16
        assert!((statistical_weight(3.0, 3.5) - 7.0 / 16.0).abs() < 1e-15);
    }

    #[test]
    fn test_pi_over_k2_u238() {
        // At 6.674 eV for U-238:
        // k² = 2·m_n·(AWR/(1+AWR))²·E / (ℏc)²
        // With corrected formula: π/k² ≈ 9.86e4 barns
        // (factor of (1+AWR)/AWR = 1.0042 larger than uncorrected)
        let val = pi_over_k_squared_barns(6.674, 236.006);
        assert!(
            (val - 9.86e4).abs() < 1e3,
            "π/k² = {} barns, expected ~9.86e4",
            val
        );
    }

    /// k_squared must return 0.0 for negative energies (guard path).
    /// Negative E_r can appear for bound-state resonances in ENDF; the guard
    /// prevents a negative k² from propagating NaN through sqrt downstream.
    #[test]
    fn test_k_squared_negative_energy() {
        assert_eq!(k_squared(-1.0, 236.006), 0.0);
        assert_eq!(k_squared(-100.0, 10.0), 0.0);
        assert_eq!(k_squared(0.0, 236.006), 0.0);
    }

    /// wave_number_from_cm must return 0.0 for negative CM energies.
    /// Negative E_cm arises near threshold energies from numerical noise.
    #[test]
    fn test_wave_number_from_cm_negative_energy() {
        assert_eq!(wave_number_from_cm(-1.0, 0.5), 0.0);
        assert_eq!(wave_number_from_cm(-1e-10, 0.9412), 0.0);
        // Zero CM energy → zero wave number
        assert_eq!(wave_number_from_cm(0.0, 0.5), 0.0);
        // Positive CM energy → positive wave number
        assert!(wave_number_from_cm(1.0, 0.5) > 0.0);
    }
}
