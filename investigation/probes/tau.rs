use nereids_physics::ikeda_carpenter::{
    IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
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
    for n_tau in [8usize, 16, 64, 600] {
        let ic = IkedaCarpenter::new(
            IkedaCarpenterParams::constant(1.0, 0.1, 0.0),
            25.0,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 2.0,
                n_energies: 2,
                n_tau,
            },
        )
        .unwrap();
        let (o, w) = ic.kernel_at(1.0).unwrap();
        let v = variance(&o, &w);
        println!(
            "n_tau={n_tau} points={} dt={:.6} variance={v:.9} \
             rel_err_vs_3={:+.3}%",
            o.len(),
            o[1] - o[0],
            100.0 * (v / 3.0 - 1.0)
        );
    }
}
