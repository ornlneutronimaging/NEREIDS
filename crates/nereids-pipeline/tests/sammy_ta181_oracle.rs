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
use nereids_physics::thickness::uneven_transmission;
use nereids_pipeline::counts_fit::{
    Material, Measurement, NO_TIME_BACKGROUND, Region, Value, fit_counts,
};
use nereids_pipeline::open_beam::{Calibration, Pulse};

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

/// The VENUS bins with a centre energy in `WINDOW_EV`, in time order.
struct Bins {
    /// Time edges in µs, ascending.
    edges_us: Vec<f64>,
    /// Bin-centre energies in eV, descending.
    centres_ev: Vec<f64>,
}

fn bins() -> Bins {
    let edge = |j: usize| (FIRST_EDGE_CENTI_US + BIN_CENTI_US * j) as f64 / 100.0;
    let centre = |j: usize| tof_to_energy(0.5 * (edge(j) + edge(j + 1)), FLIGHT_PATH_M);
    let kept: Vec<usize> = (0..)
        .take_while(|&j| centre(j) >= WINDOW_EV.0)
        .filter(|&j| centre(j) <= WINDOW_EV.1)
        .collect();
    Bins {
        edges_us: (kept[0]..=kept[kept.len() - 1] + 1).map(edge).collect(),
        centres_ev: kept.into_iter().map(centre).collect(),
    }
}

/// The rows of `S2_FILE` in the order of `bins`.
fn sammy(bins: &Bins) -> Vec<Vec<f64>> {
    let mut rows = table(S2_FILE, 7);
    assert_eq!(
        rows.len(),
        bins.centres_ev.len(),
        "{S2_FILE}: one row per bin"
    );
    rows.sort_by(|a, b| b[0].total_cmp(&a[0]));
    for (row, energy) in rows.iter().zip(&bins.centres_ev) {
        // 12 significant digits round by at most 5e-12.
        assert!(
            (row[0] / energy - 1.0).abs() <= 1e-11,
            "{S2_FILE}: {} eV is not the bin centre {energy} eV",
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
    let bins = edges_us.len() - 1;
    let (mut transmission, mut slope) = (Vec::with_capacity(bins), Vec::with_capacity(bins));
    // A grid per block keeps the point cap far from the step S2 needs.
    for first in (0..bins).step_by(BLOCK_BINS) {
        let last = (first + BLOCK_BINS).min(bins);
        let edges = &edges_us[first..=last];
        let mut known = HashMap::new();
        let mut grid = FlightTimeGrid::new(edges, T0_US, FLIGHT_PATH_M, pulse)
            .unwrap_or_else(|e| panic!("{label}, bins {first}..{last}: {e}"));
        let mut values = binned(&grid, data, kelvin, &mut known);
        let mut halvings = 0;
        loop {
            let finer = grid
                .halved()
                .unwrap_or_else(|e| panic!("{label}, bins {first}..{last}: {e}"));
            let refined = binned(&finer, data, kelvin, &mut known);
            let change = values
                .0
                .iter()
                .zip(&refined.0)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            (grid, values, halvings) = (finer, refined, halvings + 1);
            if change < GRID_CHANGE {
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
    }
    (transmission, slope)
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
    sammy(&bins());
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
        let measurement = Measurement {
            time_edges_us: bins.edges_us.clone(),
            charge_ratio: 1.0,
            normalization: Value::Known(1.0),
            time_background: NO_TIME_BACKGROUND,
            room_background: None,
            spread: None,
            regions: vec![Region {
                open_counts: open.clone(),
                sample_counts: sample,
                open_live: None,
                sample_live: None,
                background: [Value::Known(0.0); 5],
                pixels: 1.0,
                gamma_transmission: Value::Known(1.0),
                capture_gammas: Value::Known(0.0),
                in_scatter: Value::Known(0.0),
                timing_offset_us: Value::Known(0.0),
                material: Some(Material {
                    isotopes: vec![(data.clone(), Value::Known(DENSITY))],
                    temperature_k: Value::Fitted(START_FACTOR * case.kelvin),
                    thickness_variance: Value::Known(0.0),
                    library: EndfLibrary::EndfB8_1,
                    covariances: Vec::new(),
                    lattice: None,
                }),
            }],
        };
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
