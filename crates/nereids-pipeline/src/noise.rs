//! Synthetic noise generation for testing and tutorial workflows.
//!
//! Provides deterministic noise functions that model photon counting
//! statistics (Poisson) and additive measurement noise (Gaussian).
//! Each function takes an explicit `seed` for reproducibility.

use ndarray::Array3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

/// Add Poisson (shot) noise to a transmission spectrum.
///
/// Models photon counting statistics:
/// - counts_i ~ Poisson(T_i × n_photons)
/// - T_noisy_i = counts_i / n_photons
///
/// For bins where T ≤ 0 (opaque or unphysical), the count is 0.
///
/// Returns `(noisy_transmission, uncertainty)` where
/// uncertainty_i = √(max(counts_i, 1)) / n_photons.
/// The `max(counts_i, 1)` floor avoids zero uncertainty for zero-count bins.
pub fn add_poisson_noise(transmission: &[f64], n_photons: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut noisy = Vec::with_capacity(transmission.len());
    let mut uncertainty = Vec::with_capacity(transmission.len());

    for &t in transmission {
        let mean = t * n_photons;
        let counts = if mean <= 0.0 {
            0.0
        } else {
            Poisson::new(mean)
                .expect("Poisson mean must be finite and positive")
                .sample(&mut rng)
        };
        noisy.push(counts / n_photons);
        uncertainty.push((counts.max(1.0)).sqrt() / n_photons);
    }

    (noisy, uncertainty)
}

/// Add Gaussian noise to a transmission spectrum.
///
/// T_noisy_i = T_i + N(0, sigma)
///
/// Returns `(noisy_transmission, uncertainty)` where all uncertainty
/// values equal `sigma`.
pub fn add_gaussian_noise(transmission: &[f64], sigma: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, sigma).expect("sigma must be finite and positive");

    let noisy: Vec<f64> = transmission
        .iter()
        .map(|&t| t + normal.sample(&mut rng))
        .collect();
    let uncertainty = vec![sigma; transmission.len()];

    (noisy, uncertainty)
}

/// Generate a noisy 3D transmission cube from a 1D model spectrum.
///
/// Each pixel (y, x) gets an independent Poisson noise realization with
/// a deterministic per-pixel seed derived from the global seed.
///
/// Returns `(noisy_cube, uncertainty_cube)` with shape
/// `(n_energies, height, width)`.
pub fn generate_noisy_cube(
    transmission: &[f64],
    shape: (usize, usize),
    n_photons: f64,
    seed: u64,
) -> (Array3<f64>, Array3<f64>) {
    let (height, width) = shape;
    let n_energies = transmission.len();

    let mut data = Array3::<f64>::zeros((n_energies, height, width));
    let mut unc = Array3::<f64>::zeros((n_energies, height, width));

    for y in 0..height {
        for x in 0..width {
            // Deterministic per-pixel seed.
            let pixel_seed = seed.wrapping_add((y * width + x) as u64);
            let (noisy_spec, unc_spec) = add_poisson_noise(transmission, n_photons, pixel_seed);

            for (e, (&n, &u)) in noisy_spec.iter().zip(unc_spec.iter()).enumerate() {
                data[[e, y, x]] = n;
                unc[[e, y, x]] = u;
            }
        }
    }

    (data, unc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_noise_mean() {
        // Law of large numbers: mean of many realizations ≈ input.
        let transmission = vec![0.8, 0.5, 0.2];
        let n_photons = 10_000.0;
        let n_realizations = 1000;

        let mut sums = vec![0.0; transmission.len()];
        for s in 0..n_realizations {
            let (noisy, _) = add_poisson_noise(&transmission, n_photons, s as u64);
            for (i, &v) in noisy.iter().enumerate() {
                sums[i] += v;
            }
        }

        for (i, &t) in transmission.iter().enumerate() {
            let mean = sums[i] / n_realizations as f64;
            // With n_photons=10000 and 1000 realizations, standard error
            // of the mean is sqrt(T/n_photons)/sqrt(1000) ≈ 0.0003.
            // Use 5σ tolerance.
            let tol = 5.0 * (t / n_photons).sqrt() / (n_realizations as f64).sqrt();
            assert!(
                (mean - t).abs() < tol,
                "bin {i}: mean={mean:.6}, expected={t:.6}, tol={tol:.6}",
            );
        }
    }

    #[test]
    fn test_poisson_noise_variance() {
        // Shot noise: Var(T_noisy) ≈ T / n_photons.
        let t_val = 0.6;
        let transmission = vec![t_val; 1];
        let n_photons = 1000.0;
        let n_realizations = 5000;

        let mut values = Vec::with_capacity(n_realizations);
        for s in 0..n_realizations {
            let (noisy, _) = add_poisson_noise(&transmission, n_photons, s as u64);
            values.push(noisy[0]);
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let expected_variance = t_val / n_photons;

        // 20% relative tolerance for N=5000.
        assert!(
            (variance - expected_variance).abs() / expected_variance < 0.20,
            "variance={variance:.6e}, expected={expected_variance:.6e}",
        );
    }

    #[test]
    fn test_gaussian_noise_mean_and_std() {
        let transmission = vec![0.7; 1];
        let sigma = 0.01;
        let n_realizations = 5000;

        let mut values = Vec::with_capacity(n_realizations);
        for s in 0..n_realizations {
            let (noisy, _) = add_gaussian_noise(&transmission, sigma, s as u64);
            values.push(noisy[0]);
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev: f64 = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt();

        // Mean should be close to 0.7.
        let mean_tol = 5.0 * sigma / (n_realizations as f64).sqrt();
        assert!(
            (mean - 0.7).abs() < mean_tol,
            "mean={mean:.6}, expected=0.7, tol={mean_tol:.6}",
        );

        // Std should be close to sigma.
        assert!(
            (std_dev - sigma).abs() / sigma < 0.10,
            "std={std_dev:.6}, expected={sigma:.6}",
        );
    }

    #[test]
    fn test_seed_reproducibility() {
        let transmission = vec![0.9, 0.5, 0.1];
        let n_photons = 500.0;
        let seed = 42;

        let (a1, u1) = add_poisson_noise(&transmission, n_photons, seed);
        let (a2, u2) = add_poisson_noise(&transmission, n_photons, seed);
        assert_eq!(a1, a2, "same seed must produce same noisy values");
        assert_eq!(u1, u2, "same seed must produce same uncertainties");

        let (b1, _) = add_poisson_noise(&transmission, n_photons, seed + 1);
        assert_ne!(a1, b1, "different seeds should produce different values");
    }

    #[test]
    fn test_noisy_cube_shape() {
        let transmission = vec![0.8, 0.6, 0.4, 0.2];
        let (height, width) = (3, 5);
        let (cube, unc) = generate_noisy_cube(&transmission, (height, width), 1000.0, 0);

        assert_eq!(cube.shape(), &[4, 3, 5]);
        assert_eq!(unc.shape(), &[4, 3, 5]);
    }

    #[test]
    fn test_poisson_zero_transmission() {
        // T=0 → Poisson(0) = 0 deterministically.
        let transmission = vec![0.0, 0.0, 0.0];
        let (noisy, unc) = add_poisson_noise(&transmission, 1000.0, 99);

        for &v in &noisy {
            assert_eq!(v, 0.0, "zero transmission must stay zero");
        }
        // Uncertainty floor: sqrt(max(0, 1)) / n_photons = 1/1000.
        for &u in &unc {
            assert!((u - 1.0 / 1000.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_uncertainty_returned() {
        let transmission = vec![0.8, 0.5, 0.2];
        let (noisy, unc) = add_poisson_noise(&transmission, 500.0, 7);

        assert_eq!(noisy.len(), transmission.len());
        assert_eq!(unc.len(), transmission.len());
        for &u in &unc {
            assert!(u > 0.0, "uncertainty must be positive");
        }
    }

    #[test]
    fn test_gaussian_uncertainty_is_sigma() {
        let transmission = vec![0.9, 0.5, 0.1];
        let sigma = 0.02;
        let (_, unc) = add_gaussian_noise(&transmission, sigma, 123);

        for &u in &unc {
            assert!(
                (u - sigma).abs() < 1e-15,
                "Gaussian uncertainty must equal sigma",
            );
        }
    }
}
