//! Unbroadened ENDF/B-VIII.1 File 2 cross sections against SAMMY.
//!
//! The references in `tests/data/sammy-lrf7` come from SAMRML and the SAMMY
//! main program run on the extracts in `tests/data/endf/mf2`
//! (`scripts/sammy_lrf7_reference.py`).
//!
//! Each component may differ by 1e-6 of the total plus 1e-10 b. SAMMY's
//! √(2m)/ħ is 5.5e-8 above NEREIDS's CODATA 2018 value, which moves π/k² by
//! 1.1e-7; SAMMY.LST prints 10 decimals. SAMRML does not read background
//! R-function records, so its columns for Sr-88 hold NaN and are skipped.

use std::path::{Path, PathBuf};

use nereids_endf::parser::parse_endf_file2;
use nereids_physics::reich_moore::{CrossSections, cross_sections_on_grid};
use nereids_physics::transmission;

const RELATIVE: f64 = 1e-6;
const ABSOLUTE_BARN: f64 = 1e-10;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

struct Row {
    energy: f64,
    /// SAMRML elastic, nonelastic, then one reaction per pair beyond the second.
    samrml: Vec<f64>,
    /// SAMMY main program total, elastic and capture.
    sammy: [f64; 3],
}

fn reference(name: &str) -> Vec<Row> {
    let path = data_dir().join("sammy-lrf7").join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(|x| x.parse().unwrap_or_else(|e| panic!("{name}: {x}: {e}")))
                .collect();
            let n = v.len();
            Row {
                energy: v[0],
                samrml: v[1..n - 3].to_vec(),
                sammy: [v[n - 3], v[n - 2], v[n - 1]],
            }
        })
        .collect()
}

/// Largest |NEREIDS − SAMMY| over the allowance, and where.
#[derive(Default)]
struct Worst {
    ratio: f64,
    energy: f64,
    what: &'static str,
}

impl Worst {
    fn check(&mut self, what: &'static str, energy: f64, total: f64, ours: f64, theirs: f64) {
        let ratio = (ours - theirs).abs() / (RELATIVE * total + ABSOLUTE_BARN);
        if ratio > self.ratio {
            *self = Self {
                ratio,
                energy,
                what,
            };
        }
    }
}

fn compare(name: &str) {
    compare_with(name, true);
}

/// Compare with the SAMMY main program, and with SAMRML when `samrml` is set.
fn compare_with(name: &str, samrml: bool) {
    let path = data_dir().join("endf/mf2").join(format!("{name}.endf"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let data = parse_endf_file2(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
    let rows = reference(name);
    let energies: Vec<f64> = rows.iter().map(|r| r.energy).collect();
    let ours: Vec<CrossSections> = cross_sections_on_grid(&data, &energies);

    let (mut main, mut rml) = (Worst::default(), Worst::default());
    for (row, xs) in rows.iter().zip(&ours) {
        let e = row.energy;
        let t = xs.total;
        let [total, elastic, capture] = row.sammy;
        main.check("total", e, t, xs.total, total);
        main.check("elastic", e, t, xs.elastic, elastic);
        main.check("capture", e, t, xs.capture, capture);

        if !samrml || row.samrml.iter().any(|x| x.is_nan()) {
            continue;
        }
        let reactions: f64 = row.samrml[2..].iter().sum();
        rml.check("total", e, t, xs.total, row.samrml[0] + row.samrml[1]);
        rml.check("elastic", e, t, xs.elastic, row.samrml[0]);
        rml.check("capture", e, t, xs.capture, row.samrml[1] - reactions);
        rml.check("reaction", e, t, xs.reaction, reactions);
    }
    eprintln!(
        "{name}: SAMMY {:.3e} of allowance ({} at {} eV); SAMRML {:.3e} ({} at {} eV)",
        main.ratio, main.what, main.energy, rml.ratio, rml.what, rml.energy
    );
    assert!(
        main.ratio <= 1.0,
        "{name}: SAMMY {} differs by {:.3} times the allowance at {} eV",
        main.what,
        main.ratio,
        main.energy
    );
    assert!(
        rml.ratio <= 1.0,
        "{name}: SAMRML {} differs by {:.3} times the allowance at {} eV",
        rml.what,
        rml.ratio,
        rml.energy
    );
}

/// SAMMY's own LRF=7 case (samtry tr147): Al-27 with several neutron
/// channels, of different l and channel spin, in one spin group.
#[test]
fn tr147_al27() {
    let dir = data_dir().join("samtry/tr147_al27_lrf7");
    for case in ["a", "b"] {
        let read = |file: String| {
            let path = dir.join(file);
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"))
        };
        let data = parse_endf_file2(&read(format!("r{case}1.ndf")))
            .unwrap_or_else(|e| panic!("tr147 {case}: {e}"));
        let rows: Vec<[f64; 3]> = read(format!("r{case}2.lpt"))
            .lines()
            .filter_map(|line| {
                let f: Vec<&str> = line.split_whitespace().collect();
                f.first()?.parse::<usize>().ok()?;
                let v = |i: usize| f[i].replace('D', "E").parse::<f64>().unwrap();
                Some([v(1), v(2), v(3)])
            })
            .collect();
        assert_eq!(rows.len(), 13, "tr147 {case}");
        let energies: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let ours = cross_sections_on_grid(&data, &energies);
        let mut worst = Worst::default();
        for ([e, elastic, nonelastic], xs) in rows.iter().zip(&ours) {
            worst.check("total", *e, xs.total, xs.total, elastic + nonelastic);
            worst.check("elastic", *e, xs.total, xs.elastic, *elastic);
            worst.check("capture", *e, xs.total, xs.capture, *nonelastic);
        }
        eprintln!(
            "tr147 {case}: SAMRML {:.3e} of allowance ({} at {} eV)",
            worst.ratio, worst.what, worst.energy
        );
        assert!(
            worst.ratio <= 1.0,
            "tr147 {case}: SAMRML {} differs by {:.3} times the allowance at {} eV",
            worst.what,
            worst.ratio,
            worst.energy
        );
    }
}

/// W-182 at 1e-4 atoms/barn, free-gas Doppler broadened at 296 K with no
/// resolution function: −ln(T)/n differs by less than 1e-3, in the mean and
/// at each energy, the budget of the dense-grid free-gas SAMMY cases tr015
/// and tr016. The grid resolves the Doppler width, which SAMMY needs.
#[test]
fn w182_doppler_broadened() {
    let path = data_dir().join("sammy-lrf7/W-182-transmission.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let rows: Vec<[f64; 3]> = text
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            [v[0], v[1], v[2]]
        })
        .collect();
    let energies: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    let fixture = data_dir().join("endf/mf2/W-182.endf");
    let data = parse_endf_file2(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    let sigma = transmission::broadened_cross_sections(&energies, &[data], 296.0, None, None)
        .unwrap()
        .remove(0);
    let ours_t = transmission::beer_lambert(&sigma, 1e-4);

    let mut sum = 0.0;
    let mut worst = (0.0_f64, 0.0);
    let mut worst_t = 0.0_f64;
    for ((row, s), t) in rows.iter().zip(&sigma).zip(&ours_t) {
        let rel = (s - row[1]).abs() / row[1];
        sum += rel;
        if rel > worst.0 {
            worst = (rel, row[0]);
        }
        worst_t = worst_t.max((t - row[2]).abs());
    }
    let mean = sum / rows.len() as f64;
    eprintln!(
        "W-182 Doppler: mean relative {mean:.3e}, max {:.3e} at {} eV; max |dT| {worst_t:.3e}",
        worst.0, worst.1
    );
    assert!(
        mean < 1e-3,
        "W-182 Doppler: mean relative difference {mean:.3e}"
    );
    assert!(
        worst.0 < 1e-3,
        "W-182 Doppler: relative difference {:.3e} at {} eV",
        worst.0,
        worst.1
    );
}

#[test]
fn w182() {
    compare("W-182");
}

#[test]
fn w183() {
    compare("W-183");
}

#[test]
fn w184() {
    compare("W-184");
}

#[test]
fn w186() {
    compare("W-186");
}

#[test]
fn fe54() {
    compare("Fe-54");
}

/// LRF=3 through the Reich-Moore path, the reference for the LRF=7 form.
/// SAMRML gives the l = 0 and l = 2 groups (APL = 0) the formula radius for
/// the penetrability although NAPS = 1 calls for AP; above 20 keV its elastic
/// departs from the SAMMY main program by up to 1.9e-5 of the total.
#[test]
fn fe56() {
    compare_with("Fe-56", false);
}

#[test]
fn fe57() {
    compare("Fe-57");
}

#[test]
fn si28() {
    compare("Si-28");
}

#[test]
fn si29() {
    compare("Si-29");
}

#[test]
fn si30() {
    compare("Si-30");
}

#[test]
fn rh103() {
    compare("Rh-103");
}

#[test]
fn cu63() {
    compare("Cu-63");
}

#[test]
fn cu65() {
    compare("Cu-65");
}

#[test]
fn v51() {
    compare("V-51");
}

#[test]
fn sr88() {
    compare("Sr-88");
}
