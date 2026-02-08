//! Integration tests for SAMMY ex007 (ORR resolution function).

mod parsers;

use nereids_core::background::Background;
use nereids_core::energy::EnergyGrid;
use nereids_core::forward_model::{ForwardModel, ForwardModelConfig};
use nereids_core::nuclear::FissionWidths;
use nereids_core::nuclear::{
    Channel, IsotopeParams, Parameter, RMatrixParameters, Resonance, SpinGroup,
};
use nereids_physics::pipeline::DefaultForwardModel;
use nereids_physics::resolution::orr::{OrrChannelWidth, OrrDetector, OrrResolution, OrrTarget};
use parsers::{parse_dat_file, parse_lpt_chi_squared, parse_lpt_theory_points};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Ex007SpinGroupDef {
    j: f64,
    l: u32,
    channel_spin: f64,
}

#[derive(Debug, Clone)]
struct Ex007InputSpec {
    awr: f64,
    emin_ev: f64,
    emax_ev: f64,
    flight_path_m: f64,
    thickness_cm: f64,
    spin_groups: BTreeMap<usize, Ex007SpinGroupDef>,
}

#[derive(Debug, Clone)]
struct OrrParams {
    burst_width_ns: f64,
    target: OrrTarget,
    detector: OrrDetector,
    channel_widths: Vec<OrrChannelWidth>,
}

fn ex007_fixture_base_path() -> PathBuf {
    let base = PathBuf::from("tests/fixtures/sammy_reference/ex007");
    assert!(
        base.exists(),
        "Missing ex007 fixtures at {}",
        base.display()
    );
    base
}

fn compute_chi_squared(theory: &[f64], data: &[f64], uncertainties: &[f64]) -> f64 {
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

fn sorted_experimental_triplets(
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

fn filter_triplets_by_range(
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

fn parse_ex007_input_spec(path: &Path) -> Result<Ex007InputSpec, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    let mut awr = None;
    let mut emin_ev = None;
    let mut emax_ev = None;
    for line in &lines {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() >= 4
            && toks[0].chars().any(|c| c.is_alphabetic())
            && toks[1].parse::<f64>().is_ok()
            && toks[2].parse::<f64>().is_ok()
            && toks[3].parse::<f64>().is_ok()
        {
            awr = toks[1].parse::<f64>().ok();
            emin_ev = toks[2].parse::<f64>().ok();
            emax_ev = toks[3].parse::<f64>().ok();
            break;
        }
    }

    let mut flight_path_m = None;
    let mut thickness_cm = None;
    let mut saw_thermal_line = false;
    for line in &lines {
        let vals: Vec<f64> = line
            .split_whitespace()
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        if vals.len() == 5 && !saw_thermal_line {
            saw_thermal_line = true;
            flight_path_m = Some(vals[1]);
            continue;
        }
        if saw_thermal_line && vals.len() == 2 {
            thickness_cm = Some(vals[1]);
            break;
        }
    }

    let transmission_line = lines
        .iter()
        .position(|l| l.trim().eq_ignore_ascii_case("TRANSMISSION"))
        .ok_or("failed to locate TRANSMISSION block in ex007.inp")?;
    let mut spin_groups = BTreeMap::new();
    let mut i = transmission_line + 1;
    while i + 1 < lines.len() {
        let h = lines[i].trim();
        let c = lines[i + 1].trim();
        if h.is_empty() {
            break;
        }
        let h_tokens: Vec<&str> = h.split_whitespace().collect();
        let c_tokens: Vec<&str> = c.split_whitespace().collect();
        if h_tokens.len() < 4 || c_tokens.len() < 5 {
            i += 1;
            continue;
        }
        let Ok(group_idx) = h_tokens[0].parse::<usize>() else {
            i += 1;
            continue;
        };
        let Ok(j) = h_tokens[3].parse::<f64>() else {
            i += 1;
            continue;
        };
        let Ok(l) = c_tokens[3].parse::<u32>() else {
            i += 1;
            continue;
        };
        let Ok(channel_spin) = c_tokens[4].parse::<f64>() else {
            i += 1;
            continue;
        };
        spin_groups.insert(group_idx, Ex007SpinGroupDef { j, l, channel_spin });
        i += 2;
    }

    Ok(Ex007InputSpec {
        awr: awr.ok_or("failed to parse AWR from ex007.inp")?,
        emin_ev: emin_ev.ok_or("failed to parse Emin from ex007.inp")?,
        emax_ev: emax_ev.ok_or("failed to parse Emax from ex007.inp")?,
        flight_path_m: flight_path_m.ok_or("failed to parse flight path from ex007.inp")?,
        thickness_cm: thickness_cm.ok_or("failed to parse thickness from ex007.inp")?,
        spin_groups,
    })
}

fn parse_ex007_radius_map(
    path: &Path,
) -> Result<HashMap<usize, (f64, f64)>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut out = HashMap::new();
    let mut in_old_section = false;
    let mut current_kw_radius: Option<(f64, f64)> = None;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.eq_ignore_ascii_case("RADIUS PARAMETERS FOLLOW") {
            in_old_section = true;
            continue;
        }

        if trimmed.starts_with("Radii=") {
            let nums: Vec<f64> = trimmed
                .replace("Radii=", " ")
                .replace(',', " ")
                .replace("Flags=", " ")
                .split_whitespace()
                .filter_map(|t| t.parse::<f64>().ok())
                .collect();
            if nums.len() >= 2 {
                current_kw_radius = Some((nums[1], nums[0]));
            }
            continue;
        }

        if let Some((true_r, eff_r)) = current_kw_radius {
            if trimmed.starts_with("Group=") {
                let normalized = trimmed
                    .replace("Group=", " ")
                    .replace("Chan=", " ")
                    .replace(',', " ");
                let toks: Vec<&str> = normalized.split_whitespace().collect();
                if let Some(group_idx) = toks.first().and_then(|t| t.parse::<usize>().ok()) {
                    out.insert(group_idx, (true_r, eff_r));
                }
                continue;
            }
            current_kw_radius = None;
        }

        if in_old_section {
            let toks: Vec<&str> = trimmed.split_whitespace().collect();
            if toks.len() < 4 || !toks[2].contains('-') {
                continue;
            }
            let Ok(effective_radius) = toks[0].parse::<f64>() else {
                continue;
            };
            let Ok(true_radius) = toks[1].parse::<f64>() else {
                continue;
            };
            for tok in toks.iter().skip(3) {
                let Ok(group_idx) = tok.parse::<usize>() else {
                    continue;
                };
                if group_idx != 0 {
                    out.insert(group_idx, (true_radius, effective_radius));
                }
            }
        }
    }

    Ok(out)
}

fn parse_ex007_par_grouped(
    path: &Path,
) -> Result<Vec<(usize, Vec<Resonance>)>, Box<dyn std::error::Error>> {
    fn parse_fixed_11(line: &str, idx: usize) -> Option<f64> {
        let start = idx * 11;
        let end = start + 11;
        let f = line.get(start..end)?.trim();
        if f.is_empty() {
            Some(0.0)
        } else {
            f.parse::<f64>().ok()
        }
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut groups: BTreeMap<usize, Vec<Resonance>> = BTreeMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let Some(e_r) = parse_fixed_11(&line, 0) else {
            continue;
        };
        let Some(gamma_g_mev) = parse_fixed_11(&line, 1) else {
            continue;
        };
        let Some(gamma_n_mev) = parse_fixed_11(&line, 2) else {
            continue;
        };
        let Some(gamma_f1_mev) = parse_fixed_11(&line, 3) else {
            continue;
        };
        let Some(gamma_f2_mev) = parse_fixed_11(&line, 4) else {
            continue;
        };

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 6 {
            continue;
        }
        let group_idx = tokens
            .last()
            .and_then(|t| t.parse::<usize>().ok())
            .filter(|g| (1..=6).contains(g))
            .or_else(|| {
                tokens
                    .get(tokens.len().saturating_sub(2))
                    .and_then(|t| t.parse::<usize>().ok())
                    .filter(|g| (1..=6).contains(g))
            });
        let Some(group_idx) = group_idx else { continue };

        let resonance = Resonance {
            energy: Parameter::fixed(e_r),
            gamma_n: Parameter::fixed(gamma_n_mev / 1000.0),
            gamma_g: Parameter::fixed(gamma_g_mev / 1000.0),
            fission: if gamma_f1_mev.abs() > 1e-30 || gamma_f2_mev.abs() > 1e-30 {
                Some(FissionWidths {
                    gamma_f1: Parameter::fixed(gamma_f1_mev / 1000.0),
                    gamma_f2: Parameter::fixed(gamma_f2_mev / 1000.0),
                })
            } else {
                None
            },
        };
        groups.entry(group_idx).or_default().push(resonance);
    }

    Ok(groups.into_iter().collect())
}

fn parse_orr_params(path: &Path) -> Result<OrrParams, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut burst = None;
    let mut water = None;
    let mut water_m = None;
    let mut tanta_a_prime = None;
    let mut tanta_w_x = None;
    let mut tanta_x0_alpha = None;
    let mut lithi = None;
    let mut ne110_delta_mm = None;
    let mut ne110_const_lambda = None;
    let mut ne110_table: Vec<(f64, f64)> = Vec::new();
    let mut channels = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let Some(tag) = fields.next() else { continue };
        let nums: Vec<f64> = fields.filter_map(|s| s.parse::<f64>().ok()).collect();
        match tag {
            "BURST" => {
                if nums.len() >= 2 && burst.is_none() {
                    burst = Some(nums[1]);
                }
            }
            "WATER" => {
                if nums.len() >= 4 && nums[0] > 10.0 && water.is_none() {
                    water = Some((nums[1], nums[2], nums[3]));
                    water_m = Some((nums[0].round() as i64 % 10) as f64);
                }
            }
            "TANTA" => {
                if nums.len() >= 2 && (nums[0] - 1.0).abs() < 0.5 && tanta_a_prime.is_none() {
                    tanta_a_prime = Some(nums[1]);
                } else if nums.len() >= 5 && nums[0] >= 1000.0 && tanta_w_x.is_none() {
                    tanta_w_x = Some((nums[1], nums[2], nums[3], nums[4]));
                } else if nums.len() >= 3
                    && (nums[0] - 11.0).abs() < 0.5
                    && tanta_x0_alpha.is_none()
                {
                    tanta_x0_alpha = Some((nums[1], nums[2]));
                }
            }
            "LITHI" => {
                if nums.len() >= 4 && nums[0] > 10.0 && lithi.is_none() {
                    lithi = Some((nums[1], nums[2], nums[3]));
                }
            }
            "NE110" => {
                if nums.len() >= 4 {
                    // Header format: NE110 1 <N> <delta_mm> <unc> [<const_lambda_mm>]
                    if ne110_delta_mm.is_none() {
                        ne110_delta_mm = Some(nums[2]);
                        if nums.len() >= 5 {
                            ne110_const_lambda = Some(nums[4]);
                        }
                    }
                } else if nums.len() == 3 {
                    if ne110_delta_mm.is_none() {
                        // Compact header format: NE110 1 <delta_mm> <const_lambda_mm>
                        ne110_delta_mm = Some(nums[1]);
                        ne110_const_lambda = Some(nums[2]);
                    } else {
                        // Tabulated row format: NE110 <idx> <energy_eV> <lambda_sigma_mm>
                        ne110_table.push((nums[1], nums[2]));
                    }
                }
            }
            "CHANN" => {
                if nums.len() >= 3 {
                    channels.push(OrrChannelWidth {
                        max_energy_ev: nums[1],
                        width_ns: nums[2].abs(),
                    });
                }
            }
            _ => {}
        }
    }

    channels.sort_by(|a, b| a.max_energy_ev.total_cmp(&b.max_energy_ev));
    if channels.is_empty() {
        channels.push(OrrChannelWidth {
            max_energy_ev: f64::INFINITY,
            width_ns: 0.0,
        });
    }

    let target = if let Some((w0, w1, w2)) = water {
        OrrTarget::Water {
            lambda0_mm: w0,
            lambda1_mm: w1,
            lambda2_mm: w2,
            m: water_m.unwrap_or(4.0),
        }
    } else if let (Some(a_prime), Some((w_prime, x1, x2, x3)), Some((x0, alpha))) =
        (tanta_a_prime, tanta_w_x, tanta_x0_alpha)
    {
        OrrTarget::Tantalum {
            a_prime,
            w_prime,
            x1_prime: x1,
            x2_prime: x2,
            x3_prime: x3,
            x0_prime: x0,
            alpha,
        }
    } else {
        return Err("failed to parse ORR target parameters (WATER or TANTA)".into());
    };

    ne110_table.sort_by(|a, b| a.0.total_cmp(&b.0));
    let detector = if let Some((d, f, g)) = lithi {
        OrrDetector::LithiumGlass {
            d_ns: d,
            f_inv_ns: f,
            g,
        }
    } else if let Some(delta_mm) = ne110_delta_mm {
        OrrDetector::Ne110 {
            delta_mm,
            lambda_sigma_constant_mm: ne110_const_lambda.unwrap_or(0.0),
            lambda_sigma_mm: ne110_table,
        }
    } else {
        return Err("failed to parse ORR detector parameters (LITHI or NE110)".into());
    };

    Ok(OrrParams {
        burst_width_ns: burst.ok_or("failed to parse ORR BURST parameter")?,
        target,
        detector,
        channel_widths: channels,
    })
}

fn parse_norm_background(path: &Path) -> Option<(f64, f64)> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut want_next = false;
    for line_result in reader.lines() {
        let line = line_result.ok()?;
        if line.contains("NORMAlization and \"constant\" background follow") {
            want_next = true;
            continue;
        }
        if !want_next {
            continue;
        }
        let vals: Vec<f64> = line
            .split_whitespace()
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        if vals.len() >= 2 {
            return Some((vals[0], vals[1]));
        }
    }
    None
}

fn build_ex007_params(
    grouped_resonances: Vec<(usize, Vec<Resonance>)>,
    spec: &Ex007InputSpec,
    radius_map: &HashMap<usize, (f64, f64)>,
) -> RMatrixParameters {
    let spin_groups = grouped_resonances
        .into_iter()
        .filter(|(_, r)| !r.is_empty())
        .map(|(group_idx, resonances)| {
            let def = spec
                .spin_groups
                .get(&group_idx)
                .cloned()
                .unwrap_or(Ex007SpinGroupDef {
                    j: 0.5,
                    l: 0,
                    channel_spin: 0.5,
                });
            let (radius, effective_radius) =
                radius_map.get(&group_idx).copied().unwrap_or((6.4, 6.4));
            SpinGroup {
                j: def.j,
                channels: vec![Channel {
                    l: def.l,
                    channel_spin: def.channel_spin,
                    radius,
                    effective_radius,
                }],
                resonances,
            }
        })
        .collect();

    RMatrixParameters {
        isotopes: vec![IsotopeParams {
            name: "58NI".to_string(),
            awr: spec.awr,
            abundance: Parameter::fixed(1.0),
            thickness_cm: spec.thickness_cm,
            number_density: 1.0,
            spin_groups,
        }],
    }
}

fn build_orr_resolution(spec: &Ex007InputSpec, orr: &OrrParams) -> OrrResolution {
    OrrResolution {
        flight_path_m: spec.flight_path_m,
        burst_width_ns: orr.burst_width_ns,
        target: orr.target.clone(),
        detector: orr.detector.clone(),
        channel_widths: orr.channel_widths.clone(),
    }
}

fn ex007_variant_paths(base: &Path, suffix: &str) -> (PathBuf, PathBuf) {
    let par = base.join(format!("expected/ex007{suffix}.par"));
    let lpt = base.join(format!("expected/ex007{suffix}.lpt"));
    (par, lpt)
}

#[test]
fn test_ex007_fixture_smoke() {
    let base = ex007_fixture_base_path();
    let spec =
        parse_ex007_input_spec(&base.join("input/ex007.inp")).expect("failed ex007.inp parse");
    assert!(spec.emin_ev > 0.0 && spec.emax_ev > spec.emin_ev);
    assert!(spec.flight_path_m > 0.0);
    assert_eq!(spec.spin_groups.len(), 6);

    let exp = parse_dat_file(&base.join("input/ex007.dat")).expect("failed ex007.dat parse");
    assert_eq!(exp.energies.len(), 383);
    let (ef, _, _) = filter_triplets_by_range(
        &exp.energies,
        &exp.data,
        &exp.uncertainties,
        spec.emin_ev,
        spec.emax_ev,
    );
    assert_eq!(ef.len(), 56);

    let lpt = base.join("expected/ex007awl.lpt");
    let stats = parse_lpt_chi_squared(&lpt).expect("failed ex007awl.lpt chi² parse");
    assert!(stats.chi_squared.is_finite() && stats.chi_squared > 0.0);
    let theory_points = parse_lpt_theory_points(&lpt).expect("failed ex007awl.lpt theory parse");
    assert_eq!(theory_points.len(), 56);

    let r_input =
        parse_ex007_radius_map(&base.join("input/ex007wl.par")).expect("radius map input");
    let r_fit =
        parse_ex007_radius_map(&base.join("expected/ex007awl.par")).expect("radius map fitted");
    assert_eq!(r_input.len(), 6);
    assert_eq!(r_fit.len(), 6);

    let _orr_input = parse_orr_params(&base.join("input/ex007wlx.par")).expect("ORR input parse");
    let _orr_fit = parse_orr_params(&base.join("expected/ex007awl.par")).expect("ORR fitted parse");
    let _orr_fit_wn =
        parse_orr_params(&base.join("expected/ex007awn.par")).expect("ORR fitted WN parse");
    let _orr_fit_tl =
        parse_orr_params(&base.join("expected/ex007atl.par")).expect("ORR fitted TL parse");
    let _orr_fit_tn =
        parse_orr_params(&base.join("expected/ex007atn.par")).expect("ORR fitted TN parse");
}

#[test]
fn test_ex007_orr_resolution_improves_fit_with_input_parameters() {
    let base = ex007_fixture_base_path();
    let spec =
        parse_ex007_input_spec(&base.join("input/ex007.inp")).expect("failed ex007.inp parse");
    let exp = parse_dat_file(&base.join("input/ex007.dat")).expect("failed ex007.dat parse");
    let (e, d, u) = filter_triplets_by_range(
        &exp.energies,
        &exp.data,
        &exp.uncertainties,
        spec.emin_ev,
        spec.emax_ev,
    );
    let (energies, data, uncertainties) = sorted_experimental_triplets(&e, &d, &u);
    let energy_grid = EnergyGrid::new(energies).expect("failed to build energy grid");

    let grouped =
        parse_ex007_par_grouped(&base.join("input/ex007wl.par")).expect("failed input par parse");
    let radii =
        parse_ex007_radius_map(&base.join("input/ex007wl.par")).expect("failed input radius parse");
    let params = build_ex007_params(grouped, &spec, &radii);

    let orr = parse_orr_params(&base.join("input/ex007wlx.par")).expect("failed ORR input parse");
    let resolution = build_orr_resolution(&spec, &orr);

    let config = ForwardModelConfig {
        include_potential_scattering: true,
        normalization: 1.003_265_5,
        background: Background::Constant { value: 7.239e-2 },
        ..ForwardModelConfig::default()
    };

    let no_res = DefaultForwardModel { resolution: None }
        .transmission(&energy_grid, &params, &config)
        .expect("no-resolution transmission failed");
    let chi2_no_res = compute_chi_squared(&no_res, &data, &uncertainties);

    let with_res = DefaultForwardModel {
        resolution: Some(Box::new(resolution)),
    }
    .transmission(&energy_grid, &params, &config)
    .expect("ORR-resolution transmission failed");
    let chi2_with_res = compute_chi_squared(&with_res, &data, &uncertainties);

    assert!(
        chi2_with_res < chi2_no_res,
        "expected ORR resolution to improve fit: no_res={chi2_no_res:.3}, with_res={chi2_with_res:.3}"
    );
}

#[test]
fn test_ex007_fitted_parameters_replay_sammy_chi_squared() {
    let base = ex007_fixture_base_path();
    let spec =
        parse_ex007_input_spec(&base.join("input/ex007.inp")).expect("failed ex007.inp parse");

    let exp = parse_dat_file(&base.join("input/ex007.dat")).expect("failed ex007.dat parse");
    let (e, d, u) = filter_triplets_by_range(
        &exp.energies,
        &exp.data,
        &exp.uncertainties,
        spec.emin_ev,
        spec.emax_ev,
    );
    let (energies, data, uncertainties) = sorted_experimental_triplets(&e, &d, &u);
    let energy_grid = EnergyGrid::new(energies).expect("failed to build energy grid");

    for (suffix, reported_scale_limit) in [
        ("awl", 4.0_f64),
        ("awn", 4.0_f64),
        // Tantalum + lithium path should now stay within a reasonably tight
        // scale of SAMMY reported chi² for this fixture.
        ("atl", 2.0_f64),
        ("atn", 4.0_f64),
    ] {
        let (par_path, lpt_path) = ex007_variant_paths(&base, suffix);
        let grouped = parse_ex007_par_grouped(&par_path).expect("failed fitted par parse");
        let radii = parse_ex007_radius_map(&par_path).expect("failed fitted radius parse");
        let params = build_ex007_params(grouped, &spec, &radii);

        let orr = parse_orr_params(&par_path).expect("failed fitted ORR parse");
        let resolution = build_orr_resolution(&spec, &orr);
        let (norm, back_const) =
            parse_norm_background(&par_path).unwrap_or((1.003_265_5, 7.239e-2));

        let config = ForwardModelConfig {
            include_potential_scattering: true,
            normalization: norm,
            background: Background::Constant { value: back_const },
            ..ForwardModelConfig::default()
        };

        let theory = DefaultForwardModel {
            resolution: Some(Box::new(resolution)),
        }
        .transmission(&energy_grid, &params, &config)
        .expect("fitted-ORR transmission failed");
        let chi2 = compute_chi_squared(&theory, &data, &uncertainties);

        let sammy_reported = parse_lpt_chi_squared(&lpt_path)
            .expect("failed ex007 variant lpt chi² parse")
            .chi_squared;
        let mut sammy_theory = parse_lpt_theory_points(&lpt_path).expect("failed ex007 theory table parse");
        sammy_theory.sort_by(|a, b| a.energy_ev.total_cmp(&b.energy_ev));
        assert_eq!(sammy_theory.len(), data.len(), "unexpected theory-point count for {suffix}");
        let sammy_theory_vals: Vec<f64> = sammy_theory.into_iter().map(|p| p.theory).collect();
        let sammy_naive = compute_chi_squared(&sammy_theory_vals, &data, &uncertainties);

        eprintln!(
            "ex007 replay {suffix}: chi2_nereids={chi2:.3}, chi2_sammy_naive={sammy_naive:.3}, chi2_sammy_reported={sammy_reported:.3}"
        );
        assert!(
            chi2 < sammy_reported * reported_scale_limit,
            "expected fitted replay to stay on SAMMY reported chi2 scale for {suffix}"
        );
    }
}

#[test]
fn test_ex007_fitted_parameters_match_sammy_theory_table() {
    let base = ex007_fixture_base_path();
    let spec =
        parse_ex007_input_spec(&base.join("input/ex007.inp")).expect("failed ex007.inp parse");
    for (suffix, mae_limit) in [
        ("awl", 0.15_f64),
        ("awn", 0.25_f64),
        ("atl", 0.15_f64),
        ("atn", 0.25_f64),
    ] {
        let (par_path, lpt_path) = ex007_variant_paths(&base, suffix);
        let theory_pts =
            parse_lpt_theory_points(&lpt_path).expect("failed ex007 theory table parse");

        let energies: Vec<f64> = theory_pts.iter().map(|p| p.energy_ev).collect();
        let sammy_theory: Vec<f64> = theory_pts.iter().map(|p| p.theory).collect();
        let energy_grid = EnergyGrid::new(energies).expect("failed to build theory energy grid");

        let grouped = parse_ex007_par_grouped(&par_path).expect("failed fitted par parse");
        let radii = parse_ex007_radius_map(&par_path).expect("failed fitted radius parse");
        let params = build_ex007_params(grouped, &spec, &radii);
        let orr = parse_orr_params(&par_path).expect("failed fitted ORR parse");
        let resolution = build_orr_resolution(&spec, &orr);
        let (norm, back_const) =
            parse_norm_background(&par_path).unwrap_or((1.003_265_5, 7.239e-2));

        let config = ForwardModelConfig {
            include_potential_scattering: true,
            normalization: norm,
            background: Background::Constant { value: back_const },
            ..ForwardModelConfig::default()
        };

        let nereids_theory = DefaultForwardModel {
            resolution: Some(Box::new(resolution)),
        }
        .transmission(&energy_grid, &params, &config)
        .expect("theory replay transmission failed");

        let mae = nereids_theory
            .iter()
            .zip(sammy_theory.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / nereids_theory.len() as f64;
        eprintln!("ex007 {suffix} theory-table MAE={mae:.6}");
        assert!(
            mae < mae_limit,
            "expected close theory-table replay for {suffix}; got MAE={mae:.6}"
        );
    }
}
