//! Debug test for tr007 cross-section comparison.

use nereids_endf::sammy::{
    parse_sammy_inp, parse_sammy_par, parse_sammy_plt, sammy_to_nereids_resolution,
    sammy_to_resonance_data,
};
use nereids_physics::reich_moore;
use nereids_physics::transmission;

use std::path::PathBuf;

fn samtry_data_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir.push("tests/data/samtry");
    dir
}

#[test]
#[ignore] // Diagnostic test with verbose output — run manually with `cargo test -- --ignored`.
fn debug_tr007_cross_sections() {
    let dir = samtry_data_dir().join("tr007_fe56_transmission_doppler_resolution");
    let inp = parse_sammy_inp(&std::fs::read_to_string(dir.join("t007a.inp")).unwrap()).unwrap();
    let par = parse_sammy_par(&std::fs::read_to_string(dir.join("t007a.par")).unwrap()).unwrap();
    let plt =
        parse_sammy_plt(&std::fs::read_to_string(dir.join("answers/raa.plt")).unwrap()).unwrap();
    let rd = sammy_to_resonance_data(&inp, &par).unwrap();

    eprintln!("\n=== ResonanceData ===");
    eprintln!("ZA={}, AWR={}", rd.za, rd.awr);
    for range in &rd.ranges {
        eprintln!(
            "Range: [{}, {}] eV, formalism={:?}, target_spin={}, radius={}",
            range.energy_low,
            range.energy_high,
            range.formalism,
            range.target_spin,
            range.scattering_radius
        );
        for lg in &range.l_groups {
            eprintln!("  L={}, awr={}, apl={}", lg.l, lg.awr, lg.apl);
            for r in &lg.resonances {
                eprintln!(
                    "    E={:.2} eV, J={}, Γ_n={:.6} eV, Γ_γ={:.6} eV",
                    r.energy, r.j, r.gn, r.gg
                );
            }
        }
    }

    // Compute broadened cross-sections (Doppler only)
    let energies: Vec<f64> = plt.iter().map(|r| r.energy_kev * 1000.0).collect();
    let broadened_doppler = transmission::broadened_cross_sections(
        &energies,
        std::slice::from_ref(&rd),
        inp.temperature_k,
        None,
        None,
    )
    .unwrap();

    // Compute broadened cross-sections (Doppler + Gaussian resolution).
    // Uses sammy_to_nereids_resolution() which converts SAMMY's (Deltal, Deltag)
    // to NEREIDS's (delta_t_us, delta_l_m), respecting BROADENING card overrides.
    use nereids_physics::resolution::{ResolutionFunction, ResolutionParams};
    use nereids_physics::transmission::InstrumentParams;
    let (flight_path, delta_t, delta_l) = sammy_to_nereids_resolution(&inp)
        .expect("tr007 should have non-zero resolution parameters");
    let res_params = ResolutionParams::new(flight_path, delta_t, delta_l).unwrap();
    let instrument = InstrumentParams {
        resolution: ResolutionFunction::Gaussian(res_params),
    };
    let broadened_full = transmission::broadened_cross_sections(
        &energies,
        std::slice::from_ref(&rd),
        inp.temperature_k,
        Some(&instrument),
        None,
    )
    .unwrap();

    let xs_doppler = &broadened_doppler[0];
    let xs_full = &broadened_full[0];

    eprintln!("\n=== Broadened comparison (every 10th + near peak) ===");
    eprintln!(
        "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "E(keV)", "SAMMY_th", "unbroad", "dop_only", "dop+res", "rel_err"
    );
    for (i, rec) in plt.iter().enumerate() {
        let e_ev = rec.energy_kev * 1000.0;
        let unbroad = reich_moore::cross_sections_at_energy(&rd, e_ev).total;
        let broad = xs_full[i];
        let rel = if rec.theory_initial.abs() > 1e-6 {
            (broad - rec.theory_initial) / rec.theory_initial
        } else {
            0.0
        };

        if i % 10 == 0 || (rec.energy_kev > 1.149 && rec.energy_kev < 1.153) || i == plt.len() - 1 {
            eprintln!(
                "{:12.4} {:12.4} {:12.4} {:12.4} {:12.4} {:12.6}",
                rec.energy_kev, rec.theory_initial, unbroad, xs_doppler[i], broad, rel
            );
        }
    }

    // Near resonance peak
    eprintln!("\n=== Near resonance peak (unbroadened) ===");
    for &e_kev in &[1.145, 1.150, 1.151, 1.1511, 1.152, 1.155, 1.160] {
        let e_ev = e_kev * 1000.0;
        let xs = reich_moore::cross_sections_at_energy(&rd, e_ev);
        eprintln!(
            "E={:.4} keV: total={:.4}, elastic={:.4}, capture={:.4}",
            e_kev, xs.total, xs.elastic, xs.capture
        );
    }
}
