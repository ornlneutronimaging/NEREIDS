//! File parsers for SAMMY test fixtures.

pub mod dat_parser;
pub mod lpt_parser;
pub mod par_parser;

#[allow(dead_code)]
pub fn compute_chi_squared(theory: &[f64], data: &[f64], uncertainties: &[f64]) -> f64 {
    assert_eq!(theory.len(), data.len());
    assert_eq!(theory.len(), uncertainties.len());
    theory
        .iter()
        .zip(data.iter())
        .zip(uncertainties.iter())
        .map(|((t, d), u)| {
            let r = (t - d) / u;
            r * r
        })
        .sum()
}

#[allow(dead_code)]
pub fn sorted_experimental_triplets(
    energies: &[f64],
    data: &[f64],
    uncertainties: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rows: Vec<(f64, f64, f64)> = energies
        .iter()
        .copied()
        .zip(data.iter().copied())
        .zip(uncertainties.iter().copied())
        .map(|((e, d), u)| (e, d, u))
        .collect();
    rows.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut e = Vec::with_capacity(rows.len());
    let mut d = Vec::with_capacity(rows.len());
    let mut u = Vec::with_capacity(rows.len());
    for (ee, dd, uu) in rows {
        e.push(ee);
        d.push(dd);
        u.push(uu);
    }
    (e, d, u)
}

#[allow(dead_code)]
pub fn filter_triplets_by_range(
    energies: &[f64],
    data: &[f64],
    uncertainties: &[f64],
    emin_ev: f64,
    emax_ev: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut e = Vec::new();
    let mut d = Vec::new();
    let mut u = Vec::new();
    for ((&ee, &dd), &uu) in energies.iter().zip(data.iter()).zip(uncertainties.iter()) {
        if ee >= emin_ev && ee <= emax_ev {
            e.push(ee);
            d.push(dd);
            u.push(uu);
        }
    }
    (e, d, u)
}

// Re-export key types and functions
pub use dat_parser::parse_dat_file;
#[allow(unused_imports)]
pub use lpt_parser::{parse_lpt_chi_squared, parse_lpt_theory_points};
#[allow(unused_imports)]
pub use par_parser::parse_par_file;
