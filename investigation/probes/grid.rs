use nereids_physics::resolution::TabulatedResolution;

fn dip(e: f64) -> f64 {
    1.0 - 0.8 * (-0.5 * ((e - 15.05) / 0.005).powi(2)).exp()
}

fn main() {
    let n_k = 81usize;
    let offsets: Vec<f64> = (0..n_k)
        .map(|i| -2.0 + 4.0 * i as f64 / (n_k - 1) as f64)
        .collect();
    let weights: Vec<f64> = offsets
        .iter()
        .map(|&x| (1.0 - (x / 2.0).abs()).max(0.0))
        .collect();
    let tab = TabulatedResolution::from_kernels(vec![15.0], vec![(offsets, weights)], 25.0)
        .unwrap();
    let fine: Vec<f64> = (0..=10_000).map(|i| 10.0 + 0.001 * i as f64).collect();
    let coarse: Vec<f64> = (0..=100).map(|i| 10.0 + 0.1 * i as f64).collect();
    let out_f = tab
        .broaden(&fine, &fine.iter().map(|&e| dip(e)).collect::<Vec<_>>())
        .unwrap();
    let out_c = tab
        .broaden(
            &coarse,
            &coarse.iter().map(|&e| dip(e)).collect::<Vec<_>>(),
        )
        .unwrap();
    let (mut max, mut at, mut fval, mut cval) = (0.0f64, 0.0, 0.0, 0.0);
    for (j, &c) in out_c.iter().enumerate() {
        let f = out_f[j * 100];
        let d = (f - c).abs();
        if d > max {
            (max, at, fval, cval) = (d, coarse[j], f, c);
        }
    }
    println!(
        "coarse_vs_dense_sample max_abs_diff={max:.6} at_E={at:.3} \
         fine_then_sample={fval:.6} coarse_direct={cval:.6}"
    );
    println!(
        "coarse_input_min={:.6} fine_input_min={:.6}",
        coarse
            .iter()
            .map(|&e| dip(e))
            .fold(f64::INFINITY, f64::min),
        fine.iter()
            .map(|&e| dip(e))
            .fold(f64::INFINITY, f64::min)
    );
}
