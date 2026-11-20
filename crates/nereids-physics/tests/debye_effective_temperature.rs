//! The effective temperature of a Debye crystal against the continuous integral.
//!
//! `T_eff = (3/θ_D³) ∫₀^θ_D (x³/2) coth(x/2T) dx`, with `x` the phonon energy in
//! kelvin, evaluated to 30 digits outside NEREIDS. Section `debye-teff` of
//! `tests/data/README.md` gives their provenance.

use nereids_physics::lattice_doppler::PhononSpectrum;

/// Spectrum points, both endpoints included.
const POINTS: usize = 2000;
/// Limit of `|T_eff(NEREIDS) − T_eff(integral)|` in K.
const LIMIT_K: f64 = 0.01;

/// `(θ_D, T, T_eff)` in K from the continuous integral.
const CASES: [(f64, f64, f64); 4] = [
    (225.0, 300.0, 308.381_580_746),
    (225.0, 301.79, 310.122_518_042),
    (240.0, 300.0, 309.527_712_826),
    (240.0, 301.79, 311.262_041_406),
];

#[test]
fn debye_effective_temperature_matches_the_continuous_integral() {
    let mut failures = Vec::new();
    for (theta, kelvin, expected) in CASES {
        let spectrum = PhononSpectrum::debye(theta, POINTS).expect("spectrum");
        let finer = PhononSpectrum::debye(theta, 2 * POINTS - 1).expect("finer spectrum");
        let got = spectrum.effective_temperature_k(kelvin);
        let sampling = (finer.effective_temperature_k(kelvin) - got).abs();
        let error = (got - expected).abs();
        eprintln!(
            "θ_D {theta} K, T {kelvin} K: T_eff {got:.6} K, integral {expected:.6} K, \
             error {error:.2e} K, sampling change {sampling:.2e} K"
        );
        // The sampling must converge to a tenth of the limit before the error is judged.
        if sampling.is_nan() || sampling > 0.1 * LIMIT_K {
            failures.push(format!(
                "θ_D {theta} K, T {kelvin} K: sampling change {sampling:.2e} K"
            ));
        }
        if error.is_nan() || error > LIMIT_K {
            failures.push(format!("θ_D {theta} K, T {kelvin} K: error {error:.2e} K"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
