use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};

fn variance(o: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = w.iter().sum();
    let m = o.iter().zip(w).map(|(x, a)| x * a).sum::<f64>() / sw;
    o.iter()
        .zip(w)
        .map(|(x, a)| (x - m).powi(2) * a)
        .sum::<f64>()
        / sw
}

fn main() {
    let alpha: f64 = 26.0;
    let beta: f64 = 0.02;
    for r in [0.0, 1e-10, 1e-9, 1.0001e-9, 1e-8, 1e-7, 1e-6] {
        let params = IkedaCarpenterParams {
            alpha: EnergyLaw::Const(alpha),
            beta,
            r: EnergyLaw::Const(r),
            burst_sigma_us: None,
            channel_fwhm_us: Some(0.35),
        };
        let ic = IkedaCarpenter::new(
            params,
            25.0,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 2.0,
                n_energies: 2,
                n_tau: 500,
            },
        )
        .unwrap();
        let (o, w) = ic.kernel_at(1.0).unwrap();
        let v = variance(&o, &w);
        let expected = 3.0 / alpha.powi(2)
            + (2.0 * r - r * r) / beta.powi(2)
            + 0.35f64.powi(2) / 6.0;
        println!(
            "R={r:.4e} points={} dt={:.9} var={v:.9} expected={expected:.9} \
             rel={:+.2}% span={:.6}",
            o.len(),
            o[1] - o[0],
            100.0 * (v / expected - 1.0),
            o.last().unwrap() - o.first().unwrap()
        );
    }
}
