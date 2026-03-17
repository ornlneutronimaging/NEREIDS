//! Synthetic noise generation utilities for testing and validation.
//!
//! Provides Poisson noise injection for generating realistic synthetic
//! neutron transmission data at specified photon count levels.

use ndarray::Array3;
use rand::prelude::*;
use rand_distr::Poisson;

/// Add Poisson counting noise to a 1D transmission spectrum.
///
/// Simulates `N ~ Poisson(I₀ × T(E))` counts per energy bin.
/// Returns `(noisy_transmission, uncertainty)` where:
/// - `noisy_transmission = N / I₀`
/// - `uncertainty = sqrt(max(N, 1)) / I₀`
///
/// # Arguments
/// * `transmission` — Clean 1D transmission spectrum, values in [0, 1].
/// * `n_photons` — Open beam intensity I₀ (expected counts per bin before attenuation).
/// * `seed` — Random seed for reproducibility.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "test utility for synthetic noise generation")
)]
pub(crate) fn add_poisson_noise(
    transmission: &[f64],
    n_photons: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(
        n_photons > 0.0,
        "n_photons must be positive, got {n_photons}"
    );
    assert!(!transmission.is_empty(), "transmission must not be empty");
    let mut rng = StdRng::seed_from_u64(seed);
    let n_e = transmission.len();
    let mut noisy = vec![0.0; n_e];
    let mut unc = vec![0.0; n_e];

    for e in 0..n_e {
        let expected = n_photons * transmission[e].max(0.0);
        let counts = if expected > 0.0 {
            match Poisson::new(expected) {
                Ok(dist) => rng.sample(dist),
                Err(_) => 0.0,
            }
        } else {
            0.0
        };
        noisy[e] = counts / n_photons;
        unc[e] = counts.max(1.0).sqrt() / n_photons;
    }

    (noisy, unc)
}

/// Generate a noisy 3D transmission cube from a 1D spectrum.
///
/// Tiles the 1D spectrum across all spatial pixels and adds
/// independent Poisson noise to each pixel.
///
/// Returns `(transmission_cube, uncertainty_cube)` with shape
/// `(n_energies, height, width)`.
///
/// # Arguments
/// * `transmission` — Clean 1D transmission spectrum.
/// * `shape` — Spatial dimensions `(height, width)`.
/// * `n_photons` — Open beam intensity I₀.
/// * `seed` — Random seed for reproducibility.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "test utility for synthetic noise generation")
)]
pub(crate) fn generate_noisy_cube(
    transmission: &[f64],
    shape: (usize, usize),
    n_photons: f64,
    seed: u64,
) -> (Array3<f64>, Array3<f64>) {
    assert!(
        n_photons > 0.0,
        "n_photons must be positive, got {n_photons}"
    );
    assert!(!transmission.is_empty(), "transmission must not be empty");
    assert!(
        shape.0 > 0 && shape.1 > 0,
        "shape dimensions must be positive, got ({}, {})",
        shape.0,
        shape.1
    );
    let (height, width) = shape;
    let n_e = transmission.len();
    let mut trans = Array3::<f64>::zeros((n_e, height, width));
    let mut unc = Array3::<f64>::zeros((n_e, height, width));

    let mut rng = StdRng::seed_from_u64(seed);

    for y in 0..height {
        for x in 0..width {
            for e in 0..n_e {
                let expected = n_photons * transmission[e].max(0.0);
                let counts: f64 = if expected > 0.0 {
                    match Poisson::new(expected) {
                        Ok(dist) => rng.sample(dist),
                        Err(_) => 0.0,
                    }
                } else {
                    0.0
                };
                trans[[e, y, x]] = counts / n_photons;
                unc[[e, y, x]] = counts.max(1.0).sqrt() / n_photons;
            }
        }
    }

    (trans, unc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_poisson_noise_shape() {
        let t = vec![0.8, 0.5, 0.2];
        let (noisy, unc) = add_poisson_noise(&t, 1000.0, 42);
        assert_eq!(noisy.len(), 3);
        assert_eq!(unc.len(), 3);
    }

    #[test]
    fn test_add_poisson_noise_reproducible() {
        let t = vec![0.8, 0.5, 0.2];
        let (a, _) = add_poisson_noise(&t, 1000.0, 42);
        let (b, _) = add_poisson_noise(&t, 1000.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_generate_noisy_cube_shape() {
        let t = vec![0.8, 0.6, 0.4, 0.2];
        let (cube, unc) = generate_noisy_cube(&t, (3, 5), 1000.0, 42);
        assert_eq!(cube.shape(), &[4, 3, 5]);
        assert_eq!(unc.shape(), &[4, 3, 5]);
    }

    #[test]
    fn test_generate_noisy_cube_reproducible() {
        let t = vec![0.8, 0.6, 0.4, 0.2];
        let (a, _) = generate_noisy_cube(&t, (3, 5), 1000.0, 42);
        let (b, _) = generate_noisy_cube(&t, (3, 5), 1000.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_noisy_cube_mean_converges() {
        // With high I₀, the mean should be close to the clean transmission.
        let t = vec![0.5; 10];
        let (cube, _) = generate_noisy_cube(&t, (100, 100), 10000.0, 42);
        let mean: f64 = cube.iter().sum::<f64>() / cube.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "Mean {mean} should be close to 0.5"
        );
    }
}
