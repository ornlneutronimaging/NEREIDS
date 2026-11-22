//! Writes the unbroadened total cross section of an ENDF File 2 on a uniform grid.
//!
//! `unbroadened_total <endf> <first eV> <step eV> <count> <out>` writes `count`
//! little-endian f64 values in barns at `first + k·step`, and prints the narrowest
//! level width in eV.

use nereids_endf::parser::parse_endf_file2;
use nereids_physics::reich_moore::CrossSectionPlan;
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "usage: unbroadened_total <endf> <first eV> <step eV> <count> <out>"
    );
    let text = std::fs::read_to_string(&args[1]).expect("ENDF file");
    let data = parse_endf_file2(&text).expect("File 2");
    let first: f64 = args[2].parse().expect("first eV");
    let step: f64 = args[3].parse().expect("step eV");
    let count: usize = args[4].parse().expect("count");
    let plan = CrossSectionPlan::new(&data);
    let values: Vec<f64> = (0..count)
        .into_par_iter()
        .map(|k| plan.evaluate_one(first + k as f64 * step).total)
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[5], bytes).expect("output");
    let narrowest = data
        .ranges
        .iter()
        .filter(|range| range.is_evaluable())
        .flat_map(|range| range.levels())
        .map(|(_, width)| width)
        .filter(|&width| width > 0.0)
        .fold(f64::INFINITY, f64::min);
    println!("narrowest width {narrowest:e} eV");
}
