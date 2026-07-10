use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid, ic_pulse,
};

fn main() {
    for (a, b, r, t) in [
        (1e200, 1.0, 0.5, 2e-200),
        (1.0, 1.0, 1.0, 2.5),
        (0.05, 4.0, 0.5, 400.0),
    ] {
        println!(
            "ic_pulse(a={a:?},b={b:?},r={r:?},t={t:?})={:?}",
            ic_pulse(a, b, r, t)
        );
    }
    let neg_denom = EnergyLaw::InverseLambda {
        a0: -5e-10,
        a1: 0.0,
    };
    println!(
        "inverse_lambda_negative_tiny_denom_eval={:?}",
        neg_denom.eval(10.0)
    );
    let p = IkedaCarpenterParams {
        alpha: neg_denom,
        beta: 0.1,
        r: EnergyLaw::Const(0.0),
        burst_sigma_us: None,
        channel_fwhm_us: None,
    };
    println!(
        "negative_denom_model_constructor_ok={}",
        IkedaCarpenter::new(
            p,
            25.0,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 10.0,
                n_energies: 2,
                n_tau: 8,
            }
        )
        .is_ok()
    );
    let p2 = IkedaCarpenterParams {
        alpha: EnergyLaw::Const(1.0),
        beta: 0.1,
        r: EnergyLaw::ExpMilliEv { kappa: -1e-10 },
        burst_sigma_us: None,
        channel_fwhm_us: None,
    };
    let r_eval = p2.r.eval(10.0);
    println!(
        "negative_tiny_kappa_eval={r_eval:?} constructor_ok={}",
        IkedaCarpenter::new(
            p2,
            25.0,
            &SynthesisGrid {
                e_min_ev: 1.0,
                e_max_ev: 10.0,
                n_energies: 2,
                n_tau: 8,
            }
        )
        .is_ok()
    );

    let true_box_corner = IkedaCarpenterParams {
        alpha: EnergyLaw::SqrtE { a0: 5.0, a1: 2.0 },
        beta: 0.02,
        r: EnergyLaw::Const(1.0),
        burst_sigma_us: None,
        channel_fwhm_us: Some(0.35),
    };
    match IkedaCarpenter::new(
        true_box_corner,
        25.0,
        &SynthesisGrid {
            e_min_ev: 6.0,
            e_max_ev: 112.0,
            n_energies: 64,
            n_tau: 500,
        },
    ) {
        Ok(_) => println!("true_box_corner=Ok"),
        Err(error) => println!("true_box_corner=Err({error})"),
    }
}
