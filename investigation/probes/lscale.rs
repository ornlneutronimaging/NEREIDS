use nereids_physics::resolution::TabulatedResolution;

fn sigma_e(e: &[f64], y: &[f64]) -> f64 {
    let a: Vec<f64> = y.iter().map(|&v| (1.0 - v).max(0.0)).collect();
    let sw: f64 = a.iter().sum();
    let m = e.iter().zip(&a).map(|(x, w)| x * w).sum::<f64>() / sw;
    (e.iter()
        .zip(&a)
        .map(|(x, w)| (x - m).powi(2) * w)
        .sum::<f64>()
        / sw)
        .sqrt()
}

fn main() {
    let n_k = 401usize;
    let offsets: Vec<f64> = (0..n_k)
        .map(|i| -2.0 + 4.0 * i as f64 / (n_k - 1) as f64)
        .collect();
    let weights: Vec<f64> = offsets
        .iter()
        .map(|&x| (-0.5 * (x / 0.5).powi(2)).exp())
        .collect();
    let energies: Vec<f64> = (0..=10_000).map(|i| 15.0 + 0.001 * i as f64).collect();
    let spectrum: Vec<f64> = energies
        .iter()
        .map(|&x| 1.0 - 0.8 * (-0.5 * ((x - 20.0) / 0.001).powi(2)).exp())
        .collect();
    for l_scale in [1.005, 1.02] {
        let current = TabulatedResolution::from_kernels(
            vec![20.0],
            vec![(offsets.clone(), weights.clone())],
            25.0,
        )
        .unwrap()
        .broaden(&energies, &spectrum)
        .unwrap();
        let physical = TabulatedResolution::from_kernels(
            vec![20.0],
            vec![(offsets.clone(), weights.clone())],
            25.0 * l_scale,
        )
        .unwrap()
        .broaden(&energies, &spectrum)
        .unwrap();
        let sigma_current = sigma_e(&energies, &current);
        let sigma_physical = sigma_e(&energies, &physical);
        let max_diff = current
            .iter()
            .zip(&physical)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!(
            "L_scale={l_scale:.3}: sigma_current={sigma_current:.9} \
             sigma_physical={sigma_physical:.9} ratio={:.7} \
             max_abs_output_diff={max_diff:.7}",
            sigma_current / sigma_physical
        );
    }
}
