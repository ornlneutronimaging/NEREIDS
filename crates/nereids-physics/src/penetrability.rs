//! Hard-sphere penetrability, shift, and phase shift functions.
//!
//! These are the standard neutron penetrability factors for a hard-sphere
//! potential, needed for the R-matrix cross-section calculation.
//!
//! ## SAMMY Reference
//! - `rml/mrml07.f`: `Sinsix` (phase shifts), `Pgh` (penetrability + shift),
//!   `Pf` (penetrability only)
//! - SAMMY manual Section 2 (R-matrix theory)
//!
//! ## Definitions
//! For channel radius a and neutron wave number k:
//!   ρ = k·a
//!
//! The penetrability P_l(ρ), shift factor S_l(ρ), and hard-sphere phase
//! shift φ_l(ρ) depend on orbital angular momentum l.

/// Penetrability factor P_l(ρ) for orbital angular momentum l.
///
/// Returns the ratio of the outgoing wave amplitude at the nuclear surface
/// to the asymptotic amplitude. For s-wave (l=0), P = ρ.
///
/// Reference: SAMMY `rml/mrml07.f` Pf function (lines 474-522)
pub fn penetrability(l: u32, rho: f64) -> f64 {
    let rho2 = rho * rho;
    match l {
        0 => rho,
        1 => {
            let denom = 1.0 + rho2;
            rho2 * rho / denom
        }
        2 => {
            let rho4 = rho2 * rho2;
            let denom = 9.0 + 3.0 * rho2 + rho4;
            rho4 * rho / denom
        }
        3 => {
            let rho4 = rho2 * rho2;
            let rho6 = rho4 * rho2;
            let denom = 225.0 + 45.0 * rho2 + 6.0 * rho4 + rho6;
            rho6 * rho / denom
        }
        4 => {
            let rho4 = rho2 * rho2;
            let rho6 = rho4 * rho2;
            let rho8 = rho4 * rho4;
            let denom = 11025.0 + 1575.0 * rho2 + 135.0 * rho4 + 10.0 * rho6 + rho8;
            rho8 * rho / denom
        }
        _ => penetrability_general(l, rho),
    }
}

/// Shift factor S_l(ρ).
///
/// The shift factor modifies the effective resonance energy.
/// For s-wave, S_0 = 0. For higher l, S_l is negative and energy-dependent.
///
/// Reference: SAMMY `rml/mrml07.f` Pgh function (lines 347-469)
pub fn shift_factor(l: u32, rho: f64) -> f64 {
    let rho2 = rho * rho;
    match l {
        0 => 0.0,
        1 => {
            let denom = 1.0 + rho2;
            -1.0 / denom
        }
        2 => {
            let rho4 = rho2 * rho2;
            let denom = 9.0 + 3.0 * rho2 + rho4;
            -(18.0 + 3.0 * rho2) / denom
        }
        3 => {
            let rho4 = rho2 * rho2;
            let rho6 = rho4 * rho2;
            let denom = 225.0 + 45.0 * rho2 + 6.0 * rho4 + rho6;
            -(675.0 + 90.0 * rho2 + 6.0 * rho4) / denom
        }
        4 => {
            let rho4 = rho2 * rho2;
            let rho6 = rho4 * rho2;
            let rho8 = rho4 * rho4;
            let denom = 11025.0 + 1575.0 * rho2 + 135.0 * rho4 + 10.0 * rho6 + rho8;
            -(44100.0 + 4725.0 * rho2 + 270.0 * rho4 + 10.0 * rho6) / denom
        }
        _ => shift_factor_general(l, rho),
    }
}

/// Hard-sphere phase shift φ_l(ρ).
///
/// Reference: SAMMY `rml/mrml07.f` Sinsix function (lines 254-342)
pub fn phase_shift(l: u32, rho: f64) -> f64 {
    let rho2 = rho * rho;
    match l {
        0 => rho,
        1 => rho - rho.atan(),
        2 => rho - (3.0 * rho / (3.0 - rho2)).atan(),
        3 => {
            let num = rho * (15.0 - rho2);
            let den = 15.0 - 6.0 * rho2;
            rho - (num / den).atan()
        }
        4 => {
            let rho4 = rho2 * rho2;
            let num = rho * (105.0 - 10.0 * rho2);
            let den = 105.0 - 45.0 * rho2 + rho4;
            rho - (num / den).atan()
        }
        _ => phase_shift_general(l, rho),
    }
}

/// Derivative of penetrability dP_l/dρ.
///
/// Needed for converting between observed and reduced widths when
/// penetrability varies with energy.
///
/// Reference: SAMMY `rml/mrml07.f` Pf function
pub fn penetrability_derivative(l: u32, rho: f64) -> f64 {
    let rho2 = rho * rho;
    match l {
        0 => 1.0,
        1 => {
            let denom = 1.0 + rho2;
            rho2 * (3.0 + rho2) / (denom * denom)
        }
        2 => {
            let rho4 = rho2 * rho2;
            let denom = 9.0 + 3.0 * rho2 + rho4;
            rho4 * (45.0 + 9.0 * rho2 + rho4) / (denom * denom)
        }
        3 => {
            let rho4 = rho2 * rho2;
            let rho6 = rho4 * rho2;
            let denom = 225.0 + 45.0 * rho2 + 6.0 * rho4 + rho6;
            rho6 * (2205.0 + 315.0 * rho2 + 24.0 * rho4 + rho6) / (denom * denom)
        }
        _ => {
            // Numerical derivative as fallback for l >= 4
            let h = rho * 1e-6 + 1e-12;
            (penetrability(l, rho + h) - penetrability(l, rho - h)) / (2.0 * h)
        }
    }
}

/// General penetrability via recursion for l > 4.
///
/// Uses the recursion relation:
///   P_l = ρ / |F_l + i·G_l|²
/// where F_l, G_l are regular/irregular Coulomb functions (neutral case).
///
/// For the hard-sphere case (no Coulomb), we use the recursion:
///   P_l = ρ·P_{l-1} / [(2l-1) - ρ²·P_{l-1}/P_l ... ]
/// implemented via backwards recursion of f_l, g_l.
fn penetrability_general(l: u32, rho: f64) -> f64 {
    // Compute via backwards recursion of spherical Bessel functions.
    // P_l = ρ / (f_l² + g_l²) where f_l, g_l are defined by:
    //   f_l(ρ) = ρ·j_l(ρ), g_l(ρ) = -ρ·n_l(ρ)
    // Use the standard recursion relations.
    let (fl, gl) = bessel_fg(l, rho);
    rho / (fl * fl + gl * gl)
}

/// General shift factor via recursion for l > 4.
fn shift_factor_general(l: u32, rho: f64) -> f64 {
    let (fl, gl) = bessel_fg(l, rho);
    let denom = fl * fl + gl * gl;
    // S_l = ρ·(f_l·f_l' + g_l·g_l') / (f_l² + g_l²)
    // Using the Wronskian relation.
    // For the hard-sphere: S_l = (l+1) - ρ²/(S_{l-1} + (l) - iP_{l-1}) ... complicated
    // Simpler: use recursion S_l = ρ² / (l as f64 - S_{l-1}) - l as f64
    // But this is unstable. Use numerical derivative instead.
    let h = rho * 1e-6 + 1e-12;
    let (fl_p, gl_p) = bessel_fg(l, rho + h);
    let (fl_m, gl_m) = bessel_fg(l, rho - h);
    let p_p = (rho + h) / (fl_p * fl_p + gl_p * gl_p);
    let p_m = (rho - h) / (fl_m * fl_m + gl_m * gl_m);
    let _dp_drho = (p_p - p_m) / (2.0 * h);
    // S_l = ρ·dP_l/dρ / P_l - P_l + l(l+1)/ρ ... no, use relation:
    // Easier: S_l(ρ) = ρ·(f_l·df_l/dρ + g_l·dg_l/dρ) / (f_l² + g_l²)
    let dfl = (fl_p - fl_m) / (2.0 * h);
    let dgl = (gl_p - gl_m) / (2.0 * h);
    rho * (fl * dfl + gl * dgl) / denom
}

/// General phase shift via recursion for l > 4.
fn phase_shift_general(l: u32, rho: f64) -> f64 {
    let (fl, gl) = bessel_fg(l, rho);
    fl.atan2(gl)
}

/// Compute f_l(ρ) = ρ·j_l(ρ) and g_l(ρ) = -ρ·n_l(ρ) via upward recursion.
///
/// j_l and n_l are spherical Bessel functions of the first and second kind.
fn bessel_fg(l: u32, rho: f64) -> (f64, f64) {
    if rho.abs() < 1e-30 {
        return if l == 0 {
            (0.0, -1.0)
        } else {
            (0.0, f64::NEG_INFINITY)
        };
    }

    // Start with l=0:
    // f_0 = ρ·j_0(ρ) = sin(ρ)
    // g_0 = -ρ·n_0(ρ) = cos(ρ)
    let mut f_prev = rho.sin();
    let mut g_prev = rho.cos();

    if l == 0 {
        return (f_prev, g_prev);
    }

    // l=1:
    // f_1 = ρ·j_1(ρ) = sin(ρ)/ρ - cos(ρ)
    // g_1 = -ρ·n_1(ρ) = cos(ρ)/ρ + sin(ρ)
    let mut f_curr = f_prev / rho - g_prev;
    let mut g_curr = g_prev / rho + f_prev;

    if l == 1 {
        return (f_curr, g_curr);
    }

    // Upward recursion for l >= 2:
    // f_{l+1} = ((2l+1)/ρ)·f_l - f_{l-1}
    // g_{l+1} = ((2l+1)/ρ)·g_l - g_{l-1}
    for n in 1..(l as i64) {
        let factor = (2 * n + 1) as f64 / rho;
        let f_next = factor * f_curr - f_prev;
        let g_next = factor * g_curr - g_prev;
        f_prev = f_curr;
        g_prev = g_curr;
        f_curr = f_next;
        g_curr = g_next;
    }

    (f_curr, g_curr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penetrability_l0() {
        // P_0(ρ) = ρ
        assert!((penetrability(0, 0.5) - 0.5).abs() < 1e-15);
        assert!((penetrability(0, 1.0) - 1.0).abs() < 1e-15);
        assert!((penetrability(0, 0.001) - 0.001).abs() < 1e-15);
    }

    #[test]
    fn test_penetrability_l1() {
        // P_1(ρ) = ρ³/(1 + ρ²)
        let rho = 0.5;
        let expected = 0.5_f64.powi(3) / (1.0 + 0.25);
        assert!((penetrability(1, rho) - expected).abs() < 1e-15);

        let rho = 1.0;
        let expected = 1.0 / 2.0;
        assert!((penetrability(1, rho) - expected).abs() < 1e-15);
    }

    #[test]
    fn test_penetrability_l2() {
        // P_2(ρ) = ρ⁵/(9 + 3ρ² + ρ⁴)
        let rho = 1.0;
        let expected = 1.0 / (9.0 + 3.0 + 1.0);
        assert!((penetrability(2, rho) - expected).abs() < 1e-15);
    }

    #[test]
    fn test_shift_factor_l0() {
        // S_0 = 0 always
        assert!((shift_factor(0, 0.5)).abs() < 1e-15);
        assert!((shift_factor(0, 5.0)).abs() < 1e-15);
    }

    #[test]
    fn test_shift_factor_l1() {
        // S_1(ρ) = -1/(1 + ρ²)
        let rho = 1.0;
        assert!((shift_factor(1, rho) - (-0.5)).abs() < 1e-15);
        let rho = 0.0;
        assert!((shift_factor(1, rho) - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_phase_shift_l0() {
        // φ_0(ρ) = ρ
        assert!((phase_shift(0, 0.5) - 0.5).abs() < 1e-15);
        assert!((phase_shift(0, 1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_phase_shift_l1() {
        // φ_1(ρ) = ρ - arctan(ρ)
        let rho = 1.0;
        let expected = 1.0 - 1.0_f64.atan();
        assert!((phase_shift(1, rho) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_phase_shift_small_rho() {
        // For small ρ, φ_l ≈ ρ^(2l+1) / (1·3·5···(2l+1))²
        // φ_0(0.01) ≈ 0.01
        assert!((phase_shift(0, 0.01) - 0.01).abs() < 1e-10);
        // φ_1(0.01) ≈ 0.01 - arctan(0.01) ≈ 3.33e-7
        let expected = 0.01 - 0.01_f64.atan();
        assert!((phase_shift(1, 0.01) - expected).abs() < 1e-15);
    }

    #[test]
    fn test_penetrability_derivative_l0() {
        // dP_0/dρ = 1
        assert!((penetrability_derivative(0, 1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_penetrability_derivative_l1() {
        // dP_1/dρ = ρ²(3 + ρ²)/(1 + ρ²)²
        let rho = 1.0;
        let expected = 1.0 * (3.0 + 1.0) / (2.0 * 2.0);
        assert!((penetrability_derivative(1, rho) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_general_matches_explicit_l2() {
        // Verify that the general recursion gives the same result as explicit l=2
        let rho = 2.0;
        let p_explicit = penetrability(2, rho);
        let p_general = penetrability_general(2, rho);
        assert!(
            (p_explicit - p_general).abs() < 1e-10,
            "explicit={}, general={}",
            p_explicit,
            p_general
        );
    }

    #[test]
    fn test_general_matches_explicit_l3() {
        let rho = 1.5;
        let p_explicit = penetrability(3, rho);
        let p_general = penetrability_general(3, rho);
        assert!(
            (p_explicit - p_general).abs() < 1e-10,
            "explicit={}, general={}",
            p_explicit,
            p_general
        );
    }
}
