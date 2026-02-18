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
//! - k = √(2·μ·E) / ℏc, where μ = reduced mass
//! - ρ = k·a, where a = channel radius
//! - g_J = (2J+1) / ((2I+1)·(2s+1))

use nereids_core::constants;

/// Neutron wave number squared k² in fm⁻².
///
/// k² = 2·μ·E / (ℏc)²
///
/// where μ = m_n · AWR / (1 + AWR) is the reduced mass (in amu),
/// and the conversion uses m_n(amu) × 931.494 MeV/amu → MeV/c².
///
/// # Arguments
/// * `energy_ev` — Neutron energy in eV (center-of-mass).
/// * `awr` — Ratio of target mass to neutron mass (from ENDF).
pub fn k_squared(energy_ev: f64, awr: f64) -> f64 {
    // μ/m_n = AWR / (1 + AWR)
    let reduced_mass_ratio = awr / (1.0 + awr);

    // k² = 2 · m_n(eV/c²) · (μ/m_n) · E(eV) / (ℏc)²(eV·fm)²
    let mn_ev = constants::NEUTRON_MASS_MEV * 1e6; // MeV → eV
    let hbar_c = constants::HBAR_EV_S * constants::SPEED_OF_LIGHT * 1e15; // eV·fm

    2.0 * mn_ev * reduced_mass_ratio * energy_ev / (hbar_c * hbar_c)
}

/// Neutron wave number k in fm⁻¹.
///
/// # Arguments
/// * `energy_ev` — Neutron energy in eV (must be positive).
/// * `awr` — Ratio of target mass to neutron mass.
pub fn wave_number(energy_ev: f64, awr: f64) -> f64 {
    k_squared(energy_ev, awr).sqrt()
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
    let k2 = k_squared(energy_ev, awr);
    // π/k² in fm², convert to barns (1 barn = 100 fm²)
    std::f64::consts::PI / k2 / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_number_u238() {
        // U-238: AWR ≈ 236.006
        // At E = 1 eV, k should be approximately 2.197e-4 fm⁻¹ × √(AWR/(1+AWR))
        let awr = 236.006;
        let k = wave_number(1.0, awr);
        // Expected: k ≈ 2.197e-4 × 0.99789 ≈ 2.192e-4 fm⁻¹
        assert!(
            (k - 2.192e-4).abs() < 1e-6,
            "k = {} fm⁻¹, expected ~2.192e-4",
            k
        );
    }

    #[test]
    fn test_rho_u238() {
        // U-238 at 6.674 eV, channel radius 9.4285 fm
        let awr = 236.006;
        let r = rho(6.674, awr, 9.4285);
        // ρ = k·a ≈ 2.192e-4 × √6.674 × 9.4285 ≈ 5.34e-3
        assert!(
            (r - 5.34e-3).abs() < 1e-4,
            "ρ = {}, expected ~5.34e-3",
            r
        );
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
        // k² ≈ (2.192e-4)² × 6.674 ≈ 3.2e-7 fm⁻²
        // π/k² ≈ 9.82e6 fm² = 9.82e4 barns
        let val = pi_over_k_squared_barns(6.674, 236.006);
        assert!(
            (val - 9.82e4).abs() < 1e3,
            "π/k² = {} barns, expected ~9.82e4",
            val
        );
    }
}
