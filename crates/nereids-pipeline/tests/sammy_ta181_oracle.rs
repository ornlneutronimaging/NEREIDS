//! The pulse-folded Ta-181 transmission and a `fit_counts` temperature fit,
//! against SAMMY.
//!
//! The SAMMY references in `tests/data/sammy-ta181` are for the File 2
//! extract `Ta-181-mf2.endf` at the VENUS bin-centre energies. Section
//! `sammy-ta181` of `tests/data/README.md` gives their provenance.
//! The criteria detect a 1 K error at 300 K.

use std::collections::HashMap;
use std::path::Path;

use nereids_core::constants::tof_to_energy;
use nereids_endf::parser::parse_endf_file2;
use nereids_endf::resonance::ResonanceData;
use nereids_endf::retrieval::EndfLibrary;
use nereids_physics::continuous_doppler::{Channel, QuadratureBudget, broaden_with_budget};
use nereids_physics::flight_time_grid::FlightTimeGrid;
use nereids_physics::ikeda_carpenter::{DEFAULT_N_TAU, DetectorPulse, IkedaCarpenterParams};
use nereids_physics::lattice_doppler::{PhononSpectrum, broaden_in_lattice};
use nereids_physics::thickness::uneven_transmission;
use nereids_pipeline::counts_fit::{
    CountsFit, Material, Measurement, NO_TIME_BACKGROUND, Region, Value, fit_counts,
};
use nereids_pipeline::open_beam::{Calibration, Pulse};
use nereids_pipeline::pulse_calibration::{Provenance, PulseCalibration};

/// The reference folder, relative to this crate.
const DATA: &str = "../../tests/data/sammy-ta181";
/// SAMMY's transmission: the energy, then the values at 299, 300, 301, 1099, 1100 and 1101 K.
const S2_FILE: &str = "s2_rpi_alpha2.txt";
/// SAMMY's convergence: temperature in K and `eps_S` on each line.
const EPS_S: &str = "eps_s.txt";
/// Areal density of Ta-181 in atoms/b.
const DENSITY: f64 = 7.04e-4;
/// Flight path in m.
const FLIGHT_PATH_M: f64 = 25.0;
/// Timing offset in µs.
const T0_US: f64 = 0.0;
/// Start of VENUS bin 0 in units of 0.01 µs.
const FIRST_EDGE_CENTI_US: usize = 112;
/// Width of a VENUS bin in units of 0.01 µs.
const BIN_CENTI_US: usize = 16;
/// The bin-centre energies in eV of the compared bins.
const WINDOW_EV: (f64, f64) = (7.8, 200.0);
// The span holds the energy of every neutron that reaches the bins.
/// Energies in eV over which the pulse laws hold.
const PULSE_SPAN_EV: (f64, f64) = (1.0, 1000.0);
/// Largest change of a bin's transmission on a halving that converges S2.
const GRID_CHANGE: f64 = 1e-8;
/// Bins per flight-time grid in S2.
const BLOCK_BINS: usize = 512;
/// Moderator rate `α` in 1/µs.
const ALPHA: f64 = 2.0;
/// Open-beam counts per µs of detector time in S2b.
const OPEN_PER_US: f64 = 1e9;
/// Start of the fitted temperature in S2b, relative to the true one.
const START_FACTOR: f64 = 1.1;
/// SAMMY's crystal-lattice transmission: the energy, then the values at 295, 296 and 297 K.
const S3_FILE: &str = "s3_clm_rpi_alpha2.txt";
/// SAMMY's lattice convergence: temperature in K, `eps_S` and `eps_L`.
const EPS_S3: &str = "eps_s3.txt";
/// The phonon spectrum both codes broaden S3 with.
const SPECTRUM: &str = "../../tests/data/sammy-clm/taper.clm";
/// The S3 foil's thermodynamic temperature in K.
const LATTICE_K: f64 = 296.0;
/// Limit of `|T_fit − T|` in K of the S3 lattice calibrations.
const LATTICE_FIT_LIMIT_K: f64 = 0.1;
/// Limit of the S3 lattice calibrations' `d²` from the true pulse.
const LATTICE_D2_LIMIT: f64 = 2.0;
/// Least `d²` from the true pulse of the S3 free-gas calibration, whose misfit it resolves.
const FREE_GAS_D2_LEAST: f64 = 25.0;
/// Check 1's SAMMY transmission: energy, then T − 1 to T + 1 K in ½ K steps at 300 and 1100 K.
const P1A_FILE: &str = "p1a_rpi_alpha2.txt";
/// Check 1's SAMMY convergence: temperature in K and `eps` on each line.
const P1A_EPS: &str = "p1a_rpi_alpha2_eps.txt";
/// The scored resonances: those from 0.5 to 200 eV.
const SCORED_EV: (f64, f64) = (0.5, 200.0);
/// The 4.280412 eV resonance, which every forward-model check must judge.
const LINE_4_28_EV: f64 = 4.280_412;
/// Energies in eV over which the pulse laws hold in the forward-model checks.
const P1_PULSE_SPAN_EV: (f64, f64) = (0.3, 1000.0);

/// Check 3's Debye temperature in K and its spectrum points, both endpoints included.
const P1C_DEBYE: (f64, usize) = (240.0, 2000);
/// Bin-centre energies in eV of check 3: to the midpoint of the 194.8900 and
/// 200.1447 eV resonances.
const P1C_BINS_EV: (f64, f64) = (0.5, 197.517_35);

/// One temperature of a forward-model check.
struct Scored {
    /// Temperature in K.
    kelvin: f64,
    /// Columns of the reference at T − 1, T − ½, T, T + ½ and T + 1 K.
    columns: [usize; 5],
    /// Limit of `|b|` in K, pass mark (i).
    bias_limit_k: f64,
}

const P1A_SCORED: [Scored; 2] = [
    Scored {
        kelvin: 300.0,
        columns: [1, 2, 3, 4, 5],
        bias_limit_k: 0.05,
    },
    Scored {
        kelvin: 1100.0,
        columns: [6, 7, 8, 9, 10],
        bias_limit_k: 0.2,
    },
];

/// Check 3's temperatures against the independent lattice reference.
const P1C_INDEPENDENT: [Scored; 2] = [
    Scored {
        kelvin: 301.79,
        columns: [1, 2, 3, 4, 5],
        bias_limit_k: 0.05,
    },
    Scored {
        kelvin: 1093.0,
        columns: [6, 7, 8, 9, 10],
        bias_limit_k: 0.2,
    },
];
/// Check 3's independent lattice reference (`lattice_reference.py`), recoil `E/A`.
const P1C_INDEPENDENT_FILE: &str = "p1c_lattice_reference.txt";
/// Its numerical error: temperature in K and `eps` on each line.
const P1C_INDEPENDENT_EPS: &str = "p1c_lattice_reference_eps.txt";

/// One resonance's window: bins whose centre energy lies in `(low, high]` eV,
/// or `[low, high]` for the lowest window.
struct Window {
    resonance_ev: f64,
    low_ev: f64,
    high_ev: f64,
}

/// The windows of the resonances in `SCORED_EV`, edges at the midpoints to
/// their neighbours in the whole File 2 list, clipped to `SCORED_EV`.
fn windows(data: &ResonanceData) -> Vec<Window> {
    let mut energies: Vec<f64> = data
        .ranges
        .iter()
        .flat_map(|range| &range.l_groups)
        .flat_map(|group| &group.resonances)
        .map(|resonance| resonance.energy)
        .collect();
    energies.sort_by(f64::total_cmp);
    energies.dedup();
    (0..energies.len())
        .filter(|&i| (SCORED_EV.0..=SCORED_EV.1).contains(&energies[i]))
        .map(|i| Window {
            resonance_ev: energies[i],
            low_ev: if i == 0 {
                SCORED_EV.0
            } else {
                (0.5 * (energies[i - 1] + energies[i])).max(SCORED_EV.0)
            },
            high_ev: energies.get(i + 1).map_or(SCORED_EV.1, |next| {
                (0.5 * (energies[i] + next)).min(SCORED_EV.1)
            }),
        })
        .collect()
}

/// The indices of the `centres_ev` in `window`.
fn members(window: &Window, centres_ev: &[f64], lowest: bool) -> Vec<usize> {
    (0..centres_ev.len())
        .filter(|&k| {
            let e = centres_ev[k];
            e <= window.high_ev && (e > window.low_ev || (lowest && e == window.low_ev))
        })
        .collect()
}

/// Which pass marks the reference can judge in one window, from the reference alone.
struct Judgeable {
    bias: bool,
    residual: bool,
    slope: bool,
    /// Whether the reference's numerical error was measured; a failure without it is undecided.
    measured: bool,
    /// Why a mark cannot be judged.
    reasons: Vec<String>,
}

/// The reference's change per K, `D = [Q(T + 1) − Q(T − 1)]/2 K`, at the bins `k`.
fn sensitivity(rows: &[Vec<f64>], columns: [usize; 5], k: &[usize]) -> Vec<f64> {
    k.iter()
        .map(|&i| 0.5 * (rows[i][columns[4]] - rows[i][columns[0]]))
        .collect()
}

/// Decides, before NEREIDS runs, which marks a window can judge: each reference
/// error must be at most a tenth of the mark it enters. `eps` holds the reference
/// error bounds at T − 1, T and T + 1 K, or `None` where they were not measured.
fn judgeable(rows: &[Vec<f64>], scored: &Scored, k: &[usize], eps: Option<[f64; 3]>) -> Judgeable {
    let measured = eps.is_some();
    let eps = eps.unwrap_or([0.0; 3]);
    let d = sensitivity(rows, scored.columns, k);
    // For a cubic in T the ±1 K difference's step error is 4/3 of its change from ±½ K.
    let half: Vec<f64> = k
        .iter()
        .map(|&i| rows[i][scored.columns[3]] - rows[i][scored.columns[1]])
        .collect();
    let step_error = 4.0 / 3.0
        * d.iter()
            .zip(&half)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let largest_d = d.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let rms_d = (d.iter().map(|x| x * x).sum::<f64>() / d.len().max(1) as f64).sqrt();
    let mut reasons = Vec::new();
    let bias = rms_d > 0.0 && eps[1] / rms_d <= 0.1 * scored.bias_limit_k;
    if !bias {
        reasons.push(format!(
            "(i) eps/RMS(D) = {:.2e} K exceeds {:.2e} K",
            eps[1] / rms_d,
            0.1 * scored.bias_limit_k
        ));
    }
    let residual = largest_d > 0.0 && eps[1] <= 0.1 * 0.1 * largest_d;
    if !residual {
        reasons.push(format!(
            "(ii) eps = {:.2e} exceeds {:.2e}",
            eps[1],
            0.01 * largest_d
        ));
    }
    let slope_error = 0.5 * (eps[0] + eps[2]) + step_error;
    let slope = largest_d > 0.0 && slope_error <= 0.1 * 0.01 * largest_d;
    if !slope {
        reasons.push(format!(
            "(iii) slope error {slope_error:.2e} exceeds {:.2e}",
            0.001 * largest_d
        ));
    }
    Judgeable {
        bias,
        residual,
        slope,
        measured,
        reasons,
    }
}

/// Bounds per bin the effect of SAMMY's RPI tail cut at 16/α and of its 1 ns FWHM burst.
///
/// The cut term is 145·e^(−16) times the largest change of the value, and of D, over delays
/// of 0–30 µs (T in [0, 1], D = 0 before the first bin); the burst term is σ² times the
/// second difference per bin width².
fn rpi_model_bound(rows: &[Vec<f64>], columns: [usize; 5]) -> ModelBound {
    let omitted = 145.0 * (-16.0_f64).exp();
    let reach = (30.0 / (BIN_CENTI_US as f64 / 100.0)).ceil() as usize;
    let sigma_us = 1e-3 / (8.0 * 2.0_f64.ln()).sqrt();
    let width_us = BIN_CENTI_US as f64 / 100.0;
    let at = columns[2];
    let d: Vec<f64> = rows
        .iter()
        .map(|row| 0.5 * (row[columns[4]] - row[columns[0]]))
        .collect();
    let spread = |values: &dyn Fn(usize) -> f64, j: usize, outside: (f64, f64)| {
        let first = j.saturating_sub(reach);
        let (mut low, mut high) = (values(j), values(j));
        for i in first..=j {
            low = low.min(values(i));
            high = high.max(values(i));
        }
        if j < reach {
            low = low.min(outside.0);
            high = high.max(outside.1);
        }
        high - low
    };
    let value = (0..rows.len())
        .map(|j| {
            let curvature = if j == 0 || j + 1 == rows.len() {
                0.0
            } else {
                (rows[j + 1][at] - 2.0 * rows[j][at] + rows[j - 1][at]).abs() / width_us.powi(2)
            };
            omitted * spread(&|i| rows[i][at], j, (0.0, 1.0)) + sigma_us * sigma_us * curvature
        })
        .collect();
    let slope = (0..rows.len())
        .map(|j| omitted * spread(&|i| d[i], j, (0.0, 0.0)))
        .collect();
    ModelBound { value, slope }
}

/// Bounds, per bin, of differences in what the reference computes that are not
/// numerical error: on its transmission and on its slope per K.
struct ModelBound {
    value: Vec<f64>,
    slope: Vec<f64>,
}

/// Scores one window's judgeable marks; returns its failures and its inconclusive ones.
///
/// A failure is inconclusive when `model` (indexed like `ours`) explains it or the reference's
/// error is unmeasured. NEREIDS's change on its last grid halving must stay within a tenth of
/// each mark.
#[allow(clippy::too_many_arguments)]
fn score_window(
    label: &str,
    scored: &Scored,
    judge: &Judgeable,
    k: &[usize],
    fold: &Folded,
    rows: &[Vec<f64>],
    eps: Option<f64>,
    model: Option<&ModelBound>,
) -> (Vec<String>, Vec<String>) {
    let eps = eps.unwrap_or(0.0);
    let (ours, slope) = (&fold.transmission, &fold.slope);
    let at = scored.columns[2];
    let d = sensitivity(rows, scored.columns, k);
    let r: Vec<f64> = k.iter().map(|&i| ours[i] - rows[i][at]).collect();
    let slope_error: Vec<f64> = k.iter().zip(&d).map(|(&i, d)| slope[i] - d).collect();
    let largest_d = d.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let bias_k = dot(&r, &d) / dot(&d, &d);
    let largest_r = r.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let largest_slope = slope_error.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let residual_bound = 0.1 * largest_d + eps;
    let slope_bound = 0.01 * largest_d;
    eprintln!(
        "{label}: {} bins, b = {bias_k:+.4} K (limit {} K, judged {}); max|r| = {largest_r:.3e} \
         (bound {residual_bound:.3e}, judged {}); max|D_N - D| = {largest_slope:.3e} /K \
         (bound {slope_bound:.3e}, judged {})",
        k.len(),
        scored.bias_limit_k,
        judge.bias,
        judge.residual,
        judge.slope
    );
    let mut failures = Vec::new();
    let mut inconclusive = Vec::new();
    let finite = k
        .iter()
        .all(|&i| ours[i].is_finite() && slope[i].is_finite());
    if !finite {
        failures.push(format!("{label}: a NEREIDS value is not finite"));
    }
    // NEREIDS's own error: at most a tenth of each mark.
    if let Some((value_change, slope_change)) = &fold.change {
        let moved: Vec<f64> = k.iter().map(|&i| value_change[i]).collect();
        let moved_bias = moved.iter().zip(&d).map(|(a, b)| a * b.abs()).sum::<f64>() / dot(&d, &d);
        let moved_value = moved.iter().fold(0.0_f64, |m, x| m.max(*x));
        let moved_slope = k.iter().fold(0.0_f64, |m, &i| m.max(slope_change[i]));
        if judge.bias && !within(moved_bias, 0.1 * scored.bias_limit_k) {
            failures.push(format!(
                "{label}: NEREIDS's grid moves b by {moved_bias:.3e} K"
            ));
        }
        if judge.residual && !within(moved_value, 0.01 * largest_d) {
            failures.push(format!(
                "{label}: NEREIDS's grid moves a value by {moved_value:.3e}"
            ));
        }
        if judge.slope && !within(moved_slope, 0.001 * largest_d) {
            failures.push(format!(
                "{label}: NEREIDS's grid moves a slope by {moved_slope:.3e}"
            ));
        }
    }
    // A failure the reference's model bound covers is not attributed to NEREIDS.
    let mut verdict = |failed: bool, explained: bool, note: String| {
        if failed {
            if explained {
                inconclusive.push(format!("{note} (within the reference's model bound)"));
            } else if !judge.measured {
                inconclusive.push(format!(
                    "{note} (the reference's numerical error here is not measured; measure it \
                     before attributing this to NEREIDS)"
                ));
            } else {
                failures.push(note);
            }
        }
    };
    let bound_of =
        |values: &dyn Fn(&ModelBound) -> &[f64], i: usize| model.map_or(0.0, |m| values(m)[i]);
    let bias_bound = k
        .iter()
        .zip(&d)
        .map(|(&i, d)| bound_of(&|m| &m.value, i) * d.abs())
        .sum::<f64>()
        / dot(&d, &d);
    verdict(
        judge.bias && !within(bias_k.abs(), scored.bias_limit_k),
        model.is_some() && within(bias_k.abs(), scored.bias_limit_k + bias_bound),
        format!("{label}: (i) |b| = {:.4} K", bias_k.abs()),
    );
    let residual_explained = k
        .iter()
        .zip(&r)
        .all(|(&i, r)| within(r.abs(), residual_bound + bound_of(&|m| &m.value, i)));
    verdict(
        judge.residual && !within(largest_r, residual_bound),
        model.is_some() && residual_explained,
        format!("{label}: (ii) max|r| = {largest_r:.3e}"),
    );
    let slope_explained = k
        .iter()
        .zip(&slope_error)
        .all(|(&i, e)| within(e.abs(), slope_bound + bound_of(&|m| &m.slope, i)));
    verdict(
        judge.slope && !within(largest_slope, slope_bound),
        model.is_some() && slope_explained,
        format!("{label}: (iii) max|D_N - D| = {largest_slope:.3e} /K"),
    );
    (failures, inconclusive)
}

/// A reference error bound per temperature from `name`: temperature in K and the bound.
fn eps_table(name: &str) -> Vec<(f64, f64)> {
    table(name, 2)
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect()
}

/// The bounds of `table` at T − 1, T and T + 1 K; `None` unless all three were measured.
fn eps_around(table: &[(f64, f64)], kelvin: f64) -> Option<[f64; 3]> {
    let found = [-1.0, 0.0, 1.0].map(|offset| {
        let found: Vec<f64> = table
            .iter()
            .filter(|(k, _)| (k - (kelvin + offset)).abs() < 1e-9)
            .map(|(_, e)| *e)
            .collect();
        assert!(
            found.len() <= 1,
            "at most one error bound at {} K",
            kelvin + offset
        );
        found.first().copied()
    });
    match found {
        [Some(a), Some(b), Some(c)] => Some([a, b, c]),
        _ => None,
    }
}

/// One compared temperature.
struct Case {
    /// Temperature in K.
    kelvin: f64,
    /// Columns of the SAMMY values at `kelvin − 1`, `kelvin` and `kelvin + 1` K.
    columns: [usize; 3],
    /// Limit of `|b|` in K, criterion (i).
    bias_limit_k: f64,
    /// Limit of `|T_fit − T|` in K, criterion (iv).
    fit_limit_k: f64,
}

const CASES: [Case; 2] = [
    Case {
        kelvin: 300.0,
        columns: [1, 2, 3],
        bias_limit_k: 0.05,
        fit_limit_k: 0.1,
    },
    Case {
        kelvin: 1100.0,
        columns: [4, 5, 6],
        bias_limit_k: 0.2,
        fit_limit_k: 0.3,
    },
];

fn read(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATA).join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The numbers of each line of `name` that is not blank and does not start
/// with `#`: `columns` finite values.
fn table(name: &str, columns: usize) -> Vec<Vec<f64>> {
    read(name)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let row: Vec<f64> = line
                .split_whitespace()
                .map(|x| x.parse().unwrap_or_else(|e| panic!("{name}: {x}: {e}")))
                .collect();
            assert!(
                row.len() == columns && row.iter().all(|x| x.is_finite()),
                "{name}: {line}"
            );
            row
        })
        .collect()
}

/// The VENUS bins with a centre energy in a window, in time order.
struct Bins {
    /// Time edges in µs, ascending.
    edges_us: Vec<f64>,
    /// Bin-centre energies in eV, descending.
    centres_ev: Vec<f64>,
}

fn bins() -> Bins {
    bins_in(WINDOW_EV)
}

/// The bins with a centre energy in `[window.0, window.1]` eV.
fn bins_in(window: (f64, f64)) -> Bins {
    let edge = |j: usize| (FIRST_EDGE_CENTI_US + BIN_CENTI_US * j) as f64 / 100.0;
    let centre = |j: usize| tof_to_energy(0.5 * (edge(j) + edge(j + 1)), FLIGHT_PATH_M);
    let kept: Vec<usize> = (0..)
        .take_while(|&j| centre(j) >= window.0)
        .filter(|&j| centre(j) <= window.1)
        .collect();
    Bins {
        edges_us: (kept[0]..=kept[kept.len() - 1] + 1).map(edge).collect(),
        centres_ev: kept.into_iter().map(centre).collect(),
    }
}

/// The rows of `S2_FILE` in the order of `bins`.
fn sammy(bins: &Bins) -> Vec<Vec<f64>> {
    reference(bins, S2_FILE, 7)
}

/// The rows of the SAMMY reference `name`, `columns` wide, in the order of `bins`.
fn reference(bins: &Bins, name: &str, columns: usize) -> Vec<Vec<f64>> {
    let mut rows = table(name, columns);
    assert_eq!(rows.len(), bins.centres_ev.len(), "{name}: one row per bin");
    rows.sort_by(|a, b| b[0].total_cmp(&a[0]));
    for (row, energy) in rows.iter().zip(&bins.centres_ev) {
        // 12 significant digits round by at most 5e-12.
        assert!(
            (row[0] / energy - 1.0).abs() <= 1e-11,
            "{name}: {} eV is not the bin centre {energy} eV",
            row[0]
        );
    }
    rows
}

/// SAMMY's convergence `eps_S` at `kelvin`, from `EPS_S`.
fn eps_s(kelvin: f64) -> f64 {
    let found: Vec<f64> = table(EPS_S, 2)
        .into_iter()
        .filter(|row| row[0] == kelvin)
        .map(|row| row[1])
        .collect();
    assert_eq!(found.len(), 1, "{EPS_S}: one row for {kelvin} K");
    found[0]
}

/// Ta-181 from `Ta-181-mf2.endf`.
///
/// SAMMY reads File 2 only, so the extract must give NEREIDS no File 3 cross
/// section to add.
fn ta181() -> ResonanceData {
    let data = parse_endf_file2(&read("Ta-181-mf2.endf")).expect("Ta-181-mf2.endf");
    assert_eq!(data.za, 73181);
    assert!(
        data.smooth.tables().next().is_none(),
        "Ta-181-mf2.endf holds File 3 cross sections"
    );
    data
}

/// The free-gas transmission of the sample and its slope per K at each of
/// `energies_ev`, which ascend, at `kelvin`.
fn free_gas(data: &ResonanceData, energies_ev: &[f64], kelvin: f64) -> Vec<(f64, f64)> {
    let broadened = broaden_with_budget(
        energies_ev,
        data,
        kelvin,
        Channel::Total,
        QuadratureBudget::default(),
    )
    .unwrap_or_else(|e| panic!("free gas at {kelvin} K: {e}"));
    broadened
        .values
        .iter()
        .zip(&broadened.derivatives)
        .map(|(&sigma, &sigma_per_k)| {
            // Without thickness variance T = exp(−nσ), and per_depth = T.
            let (transmission, per_depth, _) = uneven_transmission(DENSITY * sigma, 0.0);
            (transmission, -per_depth * DENSITY * sigma_per_k)
        })
        .collect()
}

/// The transmission and its slope per K in each bin of `grid` at `kelvin`.
///
/// Each is the counts of a beam of one neutron per µs of flight time through
/// the sample over its counts without the sample. `known` holds the free-gas
/// values of each energy already broadened, by its bits.
fn binned(
    grid: &FlightTimeGrid,
    data: &ResonanceData,
    kelvin: f64,
    known: &mut HashMap<u64, (f64, f64)>,
) -> (Vec<f64>, Vec<f64>) {
    let energies = grid.energies_ev();
    let mut missing: Vec<f64> = energies
        .iter()
        .copied()
        .filter(|energy| !known.contains_key(&energy.to_bits()))
        .collect();
    missing.reverse();
    if !missing.is_empty() {
        for (energy, values) in missing.iter().zip(free_gas(data, &missing, kelvin)) {
            known.insert(energy.to_bits(), values);
        }
    }
    let (transmission, slope): (Vec<f64>, Vec<f64>) = energies
        .iter()
        .map(|energy| known[&energy.to_bits()])
        .unzip();
    let predict = |values: &[f64]| grid.predict(values).expect("one value per grid point");
    let open = predict(&vec![1.0; energies.len()]);
    let over_open =
        |counts: Vec<f64>| -> Vec<f64> { counts.iter().zip(&open).map(|(c, o)| c / o).collect() };
    (
        over_open(predict(&transmission)),
        over_open(predict(&slope)),
    )
}

/// The transmission and its slope per K in each bin of `edges_us` at
/// `kelvin` with `pulse`.
///
/// Each block of `BLOCK_BINS` bins has its own grid, halved until no bin's
/// transmission changes by `GRID_CHANGE`. The last grid gives the values. A
/// grid leaves out neutrons with less than `NEGLIGIBLE_ARRIVAL_PROBABILITY`
/// chance to reach its bins.
fn resolved(
    data: &ResonanceData,
    edges_us: &[f64],
    pulse: &DetectorPulse,
    kelvin: f64,
    label: &str,
) -> (Vec<f64>, Vec<f64>) {
    let folded = folded(data, edges_us, pulse, kelvin, label, false);
    (folded.transmission, folded.slope)
}

/// A fold over the bins and each bin's change on the last halving of its grid.
struct Folded {
    transmission: Vec<f64>,
    slope: Vec<f64>,
    /// Each bin's change of transmission and of slope on the last halving; `None`
    /// where the calculation has no grid to halve.
    change: Option<(Vec<f64>, Vec<f64>)>,
}

/// [`resolved`]; with `slopes_too` each grid is also halved until no slope per K
/// changes by `GRID_CHANGE`.
fn folded(
    data: &ResonanceData,
    edges_us: &[f64],
    pulse: &DetectorPulse,
    kelvin: f64,
    label: &str,
    slopes_too: bool,
) -> Folded {
    let bins = edges_us.len() - 1;
    let (mut transmission, mut slope) = (Vec::with_capacity(bins), Vec::with_capacity(bins));
    let (mut transmission_change, mut slope_change) =
        (Vec::with_capacity(bins), Vec::with_capacity(bins));
    // A grid per block keeps the point cap far from the step S2 needs.
    for first in (0..bins).step_by(BLOCK_BINS) {
        let last = (first + BLOCK_BINS).min(bins);
        let edges = &edges_us[first..=last];
        let mut known = HashMap::new();
        let mut grid = FlightTimeGrid::new(edges, T0_US, FLIGHT_PATH_M, pulse)
            .unwrap_or_else(|e| panic!("{label}, bins {first}..{last}: {e}"));
        let mut values = binned(&grid, data, kelvin, &mut known);
        let mut halvings = 0;
        let mut last_moved: (Vec<f64>, Vec<f64>);
        loop {
            let finer = grid
                .halved()
                .unwrap_or_else(|e| panic!("{label}, bins {first}..{last}: {e}"));
            let refined = binned(&finer, data, kelvin, &mut known);
            let differences = |a: &[f64], b: &[f64]| -> Vec<f64> {
                a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect()
            };
            let (moved, moved_slope) = (
                differences(&values.0, &refined.0),
                differences(&values.1, &refined.1),
            );
            let largest = |v: &[f64]| v.iter().fold(0.0, |m: f64, x| m.max(*x));
            let change = largest(&moved);
            let slope_moved = largest(&moved_slope);
            (grid, values, halvings) = (finer, refined, halvings + 1);
            last_moved = (moved, moved_slope);
            if change < GRID_CHANGE && (!slopes_too || slope_moved < GRID_CHANGE) {
                eprintln!(
                    "{label}, bins {first}..{last}: {halvings} halvings, step {:.2e} µs, \
                     {} points, change {change:.1e}",
                    grid.step_us(),
                    grid.flight_times_us().len()
                );
                break;
            }
        }
        transmission.extend(values.0);
        slope.extend(values.1);
        transmission_change.extend(last_moved.0);
        slope_change.extend(last_moved.1);
    }
    Folded {
        transmission,
        slope,
        change: Some((transmission_change, slope_change)),
    }
}

/// Whether `value` is at most `bound`; false for NaN.
fn within(value: f64, bound: f64) -> bool {
    value <= bound
}

/// Whether `value` is less than `bound`; false for NaN.
fn below(value: f64, bound: f64) -> bool {
    value < bound
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The largest magnitude of `values` and the energy of its bin.
fn largest_at(values: &[f64], energies_ev: &[f64]) -> (f64, f64) {
    values
        .iter()
        .zip(energies_ev)
        .fold((0.0, f64::NAN), |(top, at), (value, &energy)| {
            if value.abs() > top {
                (value.abs(), energy)
            } else {
                (top, at)
            }
        })
}

/// Criteria (i) to (iii) for NEREIDS's `ours` and slope per K `slope` against
/// SAMMY's `rows` at `case`, in bin order; prints the numbers, returns failures.
///
/// With `r` NEREIDS less SAMMY and `D = [T(T+1) − T(T−1)]/2` of SAMMY:
/// (i) `|Σ r D / Σ D²| ≤ case.bias_limit_k`; (ii) `|r| ≤ 0.1·max|D| + eps_S`,
/// `eps_S < 0.02·max|D|`; (iii) `|slope − D| ≤ 0.01·max|D|`.
fn compare(
    label: &str,
    case: &Case,
    energies_ev: &[f64],
    ours: &[f64],
    slope: &[f64],
    rows: &[Vec<f64>],
    eps_s: f64,
) -> Vec<String> {
    assert!(ours.len() == rows.len() && slope.len() == rows.len());
    let [colder, at, warmer] = case.columns;
    let residual: Vec<f64> = ours.iter().zip(rows).map(|(t, row)| t - row[at]).collect();
    let sensitivity: Vec<f64> = rows
        .iter()
        .map(|row| 0.5 * (row[warmer] - row[colder]))
        .collect();
    let slope_error: Vec<f64> = slope.iter().zip(&sensitivity).map(|(s, d)| s - d).collect();
    let bias_k = dot(&residual, &sensitivity) / dot(&sensitivity, &sensitivity);
    let largest_d = sensitivity.iter().fold(0.0, |top: f64, d| top.max(d.abs()));
    let residual_bound = 0.1 * largest_d + eps_s;
    let eps_limit = 0.02 * largest_d;
    let slope_bound = 0.01 * largest_d;
    let (largest_r, r_at) = largest_at(&residual, energies_ev);
    let (largest_slope, slope_at) = largest_at(&slope_error, energies_ev);
    let r_over = residual
        .iter()
        .filter(|r| !within(r.abs(), residual_bound))
        .count();
    let slope_over = slope_error
        .iter()
        .filter(|e| !within(e.abs(), slope_bound))
        .count();
    eprintln!(
        "{label}: b = {bias_k:+.4} K (limit {} K); max|D| = {largest_d:.4e} /K; \
         eps_S = {eps_s:.2e} (limit {eps_limit:.2e}); max|r| = {largest_r:.3e} at \
         {r_at} eV (bound {residual_bound:.3e}, {r_over} bins over); \
         max|dT/dT - D| = {largest_slope:.3e} /K at {slope_at} eV (bound {slope_bound:.3e}, \
         {slope_over} bins over)",
        case.bias_limit_k
    );
    let mut failures = Vec::new();
    if !within(bias_k.abs(), case.bias_limit_k) {
        failures.push(format!(
            "{label}: (i) |b| = {:.4} K exceeds {} K",
            bias_k.abs(),
            case.bias_limit_k
        ));
    }
    if !below(eps_s, eps_limit) {
        failures.push(format!(
            "{label}: (ii) eps_S = {eps_s:.2e} is not below {eps_limit:.2e}"
        ));
    }
    if r_over > 0 {
        failures.push(format!(
            "{label}: (ii) {r_over} bins exceed {residual_bound:.3e}, the largest \
             {largest_r:.3e} at {r_at} eV"
        ));
    }
    if slope_over > 0 {
        failures.push(format!(
            "{label}: (iii) {slope_over} bins exceed {slope_bound:.3e} /K, the largest \
             {largest_slope:.3e} /K at {slope_at} eV"
        ));
    }
    failures
}

/// The references parse, and their energies are the VENUS bin centres.
#[test]
fn references_match_the_venus_bins() {
    ta181();
    let bins = bins();
    sammy(&bins);
    reference(&bins, S3_FILE, 4);
    eps_s3();
    taper();
    for case in &CASES {
        eps_s(case.kelvin);
    }
}

/// S2: the free-gas transmission and its slope per K folded over the bins with
/// only the prompt Ikeda-Carpenter term (α = 2 per µs, R = 0), with a unit beam.
#[test]
#[ignore = "slow; runs nightly"]
fn s2_ikeda_carpenter_transmission_matches_sammy_rpi() {
    let data = ta181();
    let bins = bins();
    let rows = sammy(&bins);
    // R = 0, so β does not change the pulse.
    let pulse = DetectorPulse::new(
        IkedaCarpenterParams::constant(ALPHA, 1.0, 0.0),
        PULSE_SPAN_EV,
        DEFAULT_N_TAU,
    )
    .expect("pulse");
    let mut failures = Vec::new();
    for case in &CASES {
        let label = format!("S2 alpha {ALPHA}/us at {} K", case.kelvin);
        let (ours, slope) = resolved(&data, &bins.edges_us, &pulse, case.kelvin, &label);
        failures.extend(compare(
            &label,
            case,
            &bins.centres_ev,
            &ours,
            &slope,
            &rows,
            eps_s(case.kelvin),
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Check 1: the free-gas transmission and its slope per K over the window of the
/// 4.280412 eV resonance, folded with only the prompt Ikeda-Carpenter term
/// (α = 2 per µs, R = 0), with a unit beam.
#[test]
#[ignore = "slow; runs nightly"]
fn check1_the_4_28_ev_line_matches_sammy_rpi() {
    let data = ta181();
    let line = windows(&data)
        .into_iter()
        .find(|w| w.resonance_ev == LINE_4_28_EV)
        .expect("the 4.280412 eV window");
    assert!(
        line.low_ev == SCORED_EV.0 && (line.high_ev - 7.313_616).abs() < 1e-9,
        "the 4.280412 eV window is {}-{} eV",
        line.low_ev,
        line.high_ev
    );
    let bins = bins_in((line.low_ev, line.high_ev));
    let rows = reference(&bins, P1A_FILE, 11);
    let k = members(&line, &bins.centres_ev, true);
    assert_eq!(k.len(), bins.centres_ev.len(), "every bin is in the window");
    let eps = eps_table(P1A_EPS);
    let mut failures = Vec::new();
    // Decided from the reference alone, before NEREIDS runs.
    let judged: Vec<Judgeable> = P1A_SCORED
        .iter()
        .map(|s| {
            let around = eps_around(&eps, s.kelvin);
            assert!(
                around.is_some(),
                "check 1's SAMMY error is measured around {} K",
                s.kelvin
            );
            judgeable(&rows, s, &k, around)
        })
        .collect();
    for (s, judge) in P1A_SCORED.iter().zip(&judged) {
        if !judge.reasons.is_empty() {
            failures.push(format!(
                "check 1 at {} K: not judged: {}",
                s.kelvin,
                judge.reasons.join("; ")
            ));
        }
    }
    // R = 0, so β does not change the pulse.
    let pulse = DetectorPulse::new(
        IkedaCarpenterParams::constant(ALPHA, 1.0, 0.0),
        P1_PULSE_SPAN_EV,
        DEFAULT_N_TAU,
    )
    .expect("pulse");
    let mut inconclusive = Vec::new();
    for (s, judge) in P1A_SCORED.iter().zip(&judged) {
        let label = format!("check 1, {LINE_4_28_EV} eV window at {} K", s.kelvin);
        let fold = folded(&data, &bins.edges_us, &pulse, s.kelvin, &label, true);
        let at = eps_around(&eps, s.kelvin).map(|e| e[1]);
        let bound = rpi_model_bound(&rows, s.columns);
        let (failed, open) = score_window(&label, s, judge, &k, &fold, &rows, at, Some(&bound));
        failures.extend(failed);
        inconclusive.extend(open);
    }
    eprintln!(
        "{} inconclusive:\n{}",
        inconclusive.len(),
        inconclusive.join("\n")
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(inconclusive.is_empty(), "{}", inconclusive.join("\n"));
}

/// Check 3 against an independent lattice calculation: the lattice transmission
/// without instrument and its slope per K at each bin centre, at 301.79 K and 1093 K,
/// with NEREIDS's recoil `E/A`.
#[test]
#[ignore = "slow; runs nightly"]
fn check3_lattice_matches_the_independent_reference() {
    let data = ta181();
    let all = windows(&data);
    let bins = bins_in(P1C_BINS_EV);
    let rows = reference(&bins, P1C_INDEPENDENT_FILE, 11);
    let eps = eps_table(P1C_INDEPENDENT_EPS);
    let spectrum = PhononSpectrum::debye(P1C_DEBYE.0, P1C_DEBYE.1).expect("spectrum");
    let ascending: Vec<f64> = bins.centres_ev.iter().rev().copied().collect();
    let groups: Vec<(&Window, Vec<usize>)> = all
        .iter()
        .map(|w| (w, members(w, &bins.centres_ev, w.low_ev == SCORED_EV.0)))
        .filter(|(_, k)| !k.is_empty())
        .collect();
    let mut failures = Vec::new();
    let mut unjudged = Vec::new();
    let mut inconclusive = Vec::new();
    for s in &P1C_INDEPENDENT {
        let around = eps_around(&eps, s.kelvin);
        assert!(
            around.is_some(),
            "the reference's error is measured around {} K",
            s.kelvin
        );
        let judged: Vec<Judgeable> = groups
            .iter()
            .map(|(_, k)| judgeable(&rows, s, k, around))
            .collect();
        for ((w, _), judge) in groups.iter().zip(&judged) {
            if judge.reasons.is_empty() {
                continue;
            }
            let note = format!(
                "check 3 at {} K, {} eV window: not judged: {}",
                s.kelvin,
                w.resonance_ev,
                judge.reasons.join("; ")
            );
            if w.resonance_ev == LINE_4_28_EV {
                failures.push(note);
            } else {
                unjudged.push(note);
            }
        }
        let broadened = broaden_in_lattice(&ascending, &data, s.kelvin, &spectrum, Channel::Total)
            .unwrap_or_else(|e| panic!("lattice at {} K: {e}", s.kelvin));
        let (transmission, slope): (Vec<f64>, Vec<f64>) = broadened
            .values
            .iter()
            .zip(&broadened.derivatives)
            .rev()
            .map(|(&sigma, &per_k)| {
                let transmission = (-DENSITY * sigma).exp();
                (transmission, -transmission * DENSITY * per_k)
            })
            .unzip();
        let fold = Folded {
            transmission,
            slope,
            change: None,
        };
        for ((w, k), judge) in groups.iter().zip(&judged) {
            let label = format!("check 3 at {} K, {} eV window", s.kelvin, w.resonance_ev);
            let at = around.map(|e| e[1]);
            let (failed, open) = score_window(&label, s, judge, k, &fold, &rows, at, None);
            failures.extend(failed);
            inconclusive.extend(open);
        }
    }
    eprintln!(
        "{} marks not judged:\n{}",
        unjudged.len(),
        unjudged.join("\n")
    );
    eprintln!(
        "{} inconclusive:\n{}",
        inconclusive.len(),
        inconclusive.join("\n")
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(inconclusive.is_empty(), "{}", inconclusive.join("\n"));
}

/// S2b: the temperature, convergence and finite σ_T of a `fit_counts` fit to counts made
/// from the SAMMY S2 transmission.
#[test]
#[ignore = "slow; runs nightly"]
fn s2b_fit_counts_recovers_the_temperature_of_sammy_rpi_counts() {
    let data = ta181();
    let bins = bins();
    let rows = sammy(&bins);
    // Counts must be whole numbers. Rounding moves a ratio by less than 4e-9.
    let open: Vec<f64> = bins
        .edges_us
        .windows(2)
        .map(|w| (OPEN_PER_US * (w[1] - w[0])).round())
        .collect();
    let calibration = Calibration {
        t0_us: Value::Known(T0_US),
        flight_path_m: Value::Known(FLIGHT_PATH_M),
        pulse: Pulse {
            alpha: [Value::Known(0.0), Value::Known(ALPHA)],
            beta: [Value::Known(0.0), Value::Known(1.0)],
            exponents: [Value::Known(0.5); 2],
            r: Value::Known(0.0),
            fwhm_squared_us2: Value::Known(0.0),
            r_high: None,
            detector_tail: None,
            energy_span_ev: PULSE_SPAN_EV,
            n_tau: DEFAULT_N_TAU,
            line_span_ev: None,
            prior: None,
        },
    };
    let mut failures = Vec::new();
    for case in &CASES {
        let [_, at, _] = case.columns;
        let sample: Vec<f64> = open
            .iter()
            .zip(&rows)
            .map(|(o, row)| (o * row[at]).round())
            .collect();
        let measurement = foil_alone(
            &bins.edges_us,
            [&open, &sample],
            (data.clone(), Value::Known(DENSITY)),
            Value::Fitted(START_FACTOR * case.kelvin),
            None,
        );
        let label = format!("S2b alpha {ALPHA}/us at {} K", case.kelvin);
        let fit = match fit_counts(&measurement, &calibration) {
            Ok(fit) => fit,
            Err(e) => {
                eprintln!("{label}: fit_counts refused: {e}");
                failures.push(format!("{label}: fit_counts refused: {e}"));
                continue;
            }
        };
        let region = &fit.regions[0];
        let fitted = region.temperature_k.expect("a material");
        // The temperature is the only fitted quantity, first in the covariance.
        let sigma_t = fit
            .covariance
            .as_ref()
            .map_or(f64::NAN, |c| c.get(0, 0).sqrt());
        let error = fitted - case.kelvin;
        eprintln!(
            "{label}: T_fit = {fitted:.4} K, T_fit - T = {error:+.4} K (limit {} K), \
             sigma_T = {sigma_t:.3e} K, converged {}, overdispersion {:?}, \
             step {:.2e} µs, {} points",
            case.fit_limit_k, fit.converged, region.overdispersion, fit.step_us, fit.points
        );
        if !within(error.abs(), case.fit_limit_k) {
            failures.push(format!(
                "{label}: (iv) |T_fit - T| = {:.4} K exceeds {} K",
                error.abs(),
                case.fit_limit_k
            ));
        }
        if !fit.converged {
            failures.push(format!("{label}: (iv) the fit did not converge"));
        }
        if !sigma_t.is_finite() {
            failures.push(format!("{label}: (iv) sigma_T = {sigma_t} is not finite"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The phonon spectrum of `SPECTRUM`: the step and values of its CONTINUOUS card set.
fn taper() -> PhononSpectrum {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(SPECTRUM);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text
        .lines()
        .skip_while(|line| !line.starts_with("CONTINUOUS"))
        .skip(1);
    let header: Vec<f64> = lines
        .next()
        .expect("the step line")
        .split_whitespace()
        .map(|x| x.parse().expect("a number"))
        .collect();
    let density: Vec<f64> = lines
        .take_while(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .map(|x| x.parse::<f64>().expect("a number"))
        })
        .collect();
    assert_eq!(density.len() as f64, header[1], "{SPECTRUM}: the values");
    PhononSpectrum::new(header[0], density).expect("the spectrum")
}

/// SAMMY's S3 convergence `[eps_S, eps_L]` at `LATTICE_K`, from `EPS_S3`.
fn eps_s3() -> [f64; 2] {
    let rows = table(EPS_S3, 3);
    assert!(
        rows.len() == 1 && rows[0][0] == LATTICE_K,
        "{EPS_S3}: one row, at {LATTICE_K} K"
    );
    [rows[0][1], rows[0][2]]
}

/// `d² = dᵀC⁻¹d` of the prior in the file of `calibration`, with `d` its mean
/// less `truth` over the numbers it fitted, of `(α₀, α₁, β₀, β₁, R, h²)`.
fn prior_distance(calibration: &PulseCalibration, truth: [f64; 6]) -> f64 {
    let file: serde_json::Value = serde_json::from_str(&calibration.to_json()).expect("json");
    let number = |v: &serde_json::Value| v.as_f64().expect("a number");
    let fitted = file["numbers"]
        .as_array()
        .expect("the numbers")
        .iter()
        .enumerate()
        .filter(|(_, n)| n["status"] == "fitted")
        .map(|(i, _)| truth[i]);
    let d: Vec<f64> = file["mean"]
        .as_array()
        .expect("the mean")
        .iter()
        .zip(fitted)
        .map(|(mean, truth)| number(mean) - truth)
        .collect();
    let covariance: Vec<Vec<f64>> = file["covariance"]
        .as_array()
        .expect("the covariance")
        .iter()
        .map(|row| row.as_array().expect("a row").iter().map(number).collect())
        .collect();
    // Cholesky C = LLᵀ, then d² = |y|² with Ly = d.
    let mut l: Vec<Vec<f64>> = Vec::with_capacity(d.len());
    for (i, row) in covariance.iter().enumerate() {
        let mut lower: Vec<f64> = Vec::with_capacity(i + 1);
        for (j, done) in l.iter().enumerate() {
            lower.push((row[j] - dot(&lower, &done[..j])) / done[j]);
        }
        lower.push((row[i] - dot(&lower, &lower)).sqrt());
        l.push(lower);
    }
    let mut y: Vec<f64> = Vec::with_capacity(d.len());
    for (row, difference) in l.iter().zip(&d) {
        let i = y.len();
        y.push((difference - dot(&row[..i], &y)) / row[i]);
    }
    dot(&y, &y)
}

/// The foil alone in one region, `isotope` at its density, with the open-beam
/// and sample `counts` in the bins of `edges_us`.
fn foil_alone(
    edges_us: &[f64],
    counts: [&[f64]; 2],
    isotope: (ResonanceData, Value),
    temperature_k: Value,
    lattice: Option<PhononSpectrum>,
) -> Measurement {
    Measurement {
        time_edges_us: edges_us.to_vec(),
        charge_ratio: 1.0,
        normalization: Value::Known(1.0),
        time_background: NO_TIME_BACKGROUND,
        room_background: None,
        spread: None,
        environment_scatter: None,
        filter: None,
        regions: vec![Region {
            open_counts: counts[0].to_vec(),
            sample_counts: counts[1].to_vec(),
            open_live: None,
            sample_live: None,
            background: [Value::Known(0.0); 5],
            pixels: 1.0,
            gamma_transmission: Value::Known(1.0),
            capture_gammas: Value::Known(0.0),
            in_scatter: Value::Known(0.0),
            timing_offset_us: Value::Known(0.0),
            material: Some(Material {
                isotopes: vec![isotope],
                temperature_k,
                thickness_variance: Value::Known(0.0),
                library: EndfLibrary::EndfB8_1,
                excluded_ranges: Vec::new(),
                covariances: Vec::new(),
                lattice,
            }),
        }],
    }
}

/// `t0`, the flight path, `α₀`, `α₁` and `R` fitted from starts away from the
/// truth; `β` and `h²` known.
fn fitted_pulse() -> Calibration {
    Calibration {
        t0_us: Value::Fitted(T0_US + 0.05),
        flight_path_m: Value::Fitted(FLIGHT_PATH_M + 0.01),
        pulse: Pulse {
            alpha: [Value::Fitted(0.02), Value::Fitted(START_FACTOR * ALPHA)],
            // β changes the pulse only through R, whose truth is 0.
            beta: [Value::Known(0.0), Value::Known(1.0)],
            exponents: [Value::Known(0.5); 2],
            r: Value::Fitted(0.05),
            fwhm_squared_us2: Value::Known(0.0),
            r_high: None,
            detector_tail: None,
            energy_span_ev: PULSE_SPAN_EV,
            n_tau: DEFAULT_N_TAU,
            line_span_ev: None,
            prior: None,
        },
    }
}

/// S3: pulse calibrations on counts from SAMMY's crystal-lattice transmission with the S2
/// pulse: the lattice at 310 ± 30 K and 296 ± 2 K, the free gas at T_eff ± 2 K.
#[test]
#[ignore = "slow; runs nightly"]
fn s3_a_lattice_foil_calibrates_on_sammy_crystal_lattice_counts() {
    let ta = ta181();
    // SAMMY's crystal-lattice model recoils by E/(A + 1).
    let data = ResonanceData {
        awr: ta.awr + 1.0,
        ..ta
    };
    let bins = bins();
    let rows = reference(&bins, S3_FILE, 4);
    let spectrum = taper();
    // Counts must be whole numbers. Rounding moves a ratio by less than 4e-9.
    let open: Vec<f64> = bins
        .edges_us
        .windows(2)
        .map(|w| (OPEN_PER_US * (w[1] - w[0])).round())
        .collect();
    let sample: Vec<f64> = open
        .iter()
        .zip(&rows)
        .map(|(o, row)| (o * row[2]).round())
        .collect();
    let largest_d = rows.iter().fold(0.0, |top: f64, row| {
        top.max((0.5 * (row[3] - row[1])).abs())
    });
    let [eps_s, eps_l] = eps_s3();
    let eps_limit = 0.02 * largest_d;
    eprintln!(
        "S3: max|D| = {largest_d:.4e} /K; eps_S = {eps_s:.2e}, eps_L = {eps_l:.2e} \
         (limit {eps_limit:.2e})"
    );
    let mut failures = Vec::new();
    for (eps, name) in [(eps_s, "eps_S"), (eps_l, "eps_L")] {
        if !below(eps, eps_limit) {
            failures.push(format!(
                "S3: {name} = {eps:.2e} is not below {eps_limit:.2e}"
            ));
        }
    }
    let effective_k = spectrum.effective_temperature_k(LATTICE_K);
    let truth = [0.0, ALPHA, 0.0, 1.0, 0.0, 0.0];
    let cases = [
        ("lattice, T 310 ± 30 K", Some(spectrum.clone()), 310.0, 30.0),
        ("lattice, T 296 ± 2 K", Some(spectrum), LATTICE_K, 2.0),
        ("free gas, T_eff ± 2 K", None, effective_k, 2.0),
    ];
    let density = Value::Measured {
        value: DENSITY,
        sd: 0.01 * DENSITY,
    };
    for (label, lattice, value, sd) in cases {
        let bound = lattice.is_some();
        let measurement = foil_alone(
            &bins.edges_us,
            [&open, &sample],
            (data.clone(), density),
            Value::Measured { value, sd },
            lattice,
        );
        let provenance = Provenance {
            foil: "sammy-s3".into(),
            open: "open".into(),
            sample: "sample".into(),
        };
        let start = std::time::Instant::now();
        let (calibration, fit): (PulseCalibration, CountsFit) =
            match PulseCalibration::new(&measurement, &fitted_pulse(), provenance) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("S3 {label}: refused: {e}");
                    failures.push(format!("S3 {label}: refused: {e}"));
                    continue;
                }
            };
        let fitted_k = fit.regions[0].temperature_k.expect("a material");
        let d2 = prior_distance(&calibration, truth);
        eprintln!(
            "S3 {label}: T_fit = {fitted_k:.4} K (T = {LATTICE_K} K, T_eff = {effective_k:.4} K), \
             t0 = {:+.3e} µs, L = {:.7} m, alpha = [{:.3e}, {:.6}], R = {:.3e}, \
             d2 = {d2:.4e}, converged {}, {} points, {:.0} s",
            fit.t0_us,
            fit.flight_path_m,
            fit.alpha[0],
            fit.alpha[1],
            fit.r,
            fit.converged,
            fit.points,
            start.elapsed().as_secs_f64()
        );
        let error_k = (fitted_k - LATTICE_K).abs();
        if bound && !within(error_k, LATTICE_FIT_LIMIT_K) {
            failures.push(format!(
                "S3 {label}: |T_fit - T| = {error_k:.4} K exceeds {LATTICE_FIT_LIMIT_K} K"
            ));
        }
        if bound && !within(d2, LATTICE_D2_LIMIT) {
            failures.push(format!(
                "S3 {label}: d2 = {d2:.4e} exceeds {LATTICE_D2_LIMIT}"
            ));
        }
        if !bound && !within(FREE_GAS_D2_LEAST, d2) {
            failures.push(format!(
                "S3 {label}: d2 = {d2:.4e} is below {FREE_GAS_D2_LEAST}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
