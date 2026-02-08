//! Free-gas model Doppler broadening.
//!
//! Broadens 0K cross sections to finite temperature by convolving with
//! a Gaussian kernel whose width depends on the atomic mass and temperature.
//!
//! # Physics
//!
//! At finite temperature, target atoms have a thermal velocity distribution.
//! The free-gas model uses Gaussian broadening in `v = sign(E)·sqrt(|E|)`:
//!
//! ```text
//! Velocity width: D = sqrt(k_B · T · m_n / A_target)
//! Broadening:     σ(E; T) ∝ ∫ σ₀(E') exp(-(v(E)-v(E'))²/D²) dv'
//! ```
//! where `A_target` is the target mass in amu (SAMMY `DefTargetMass` convention).
//!
//! # Algorithm
//!
//! Direct trapezoidal convolution in sqrt(E) space (SAMMY free-gas style)
//! with kernel truncation at ±5·D and normalized kernel.
//!
//! # Important
//!
//! The input energy grid must be fine enough to resolve the 0K cross section
//! structure. For narrow resonances (meV-scale widths), create an auxiliary
//! fine grid via [`create_auxiliary_grid`] before calling [`broaden_cross_sections`].
//!
//! # References
//!
//! - SAMMY `mfgm1.f90` / `mfgm4.f90` (free-gas broadening path)
//! - SAMMY Users' Guide (ORNL/TM-9179/R8), Section IV.E

use nereids_core::constants::BOLTZMANN_EV_PER_K;
use nereids_core::PhysicsError;

use super::fgm_weights::fgm_weighted_sum;

/// Number of Doppler widths for kernel truncation.
///
/// At ±5Δ, the Gaussian kernel is exp(-25) ≈ 1.4e-11.
const KERNEL_CUTOFF: f64 = 5.0;
/// Small factor used by SAMMY when expanding auxiliary-grid bounds.
const AUX_BOUNDS_FUDGE: f64 = 1.001;

/// Safety guardrail for direct trapezoidal broadening cost.
const MAX_BROADENING_SEGMENTS: usize = 20_000_000;

/// Maximum total auxiliary grid size to avoid pathological broadening workloads.
const MAX_AUX_GRID_POINTS: usize = 50_000;
/// Maximum allowed adjacent spacing ratio before smoothing inserts points.
const AUX_SPACING_RATIO_LIMIT: f64 = 2.5;
/// Maximum smoothing passes for auxiliary-grid transition refinement.
const AUX_SMOOTHING_PASSES: usize = 6;

/// Minimum number of points across ±Γ used for under-resolved resonances.
///
/// SAMMY reports this condition as "fewer than 9 points across width."
const MIN_POINTS_ACROSS_WIDTH: usize = 9;

/// Number of tail-width multiples on each side for SAMMY `Fgpwid` defaults.
const EXTRA_TAIL_POINTS: usize = 5;

/// Small floor for effective resonance width to keep refinement stable.
const MIN_REFINEMENT_WIDTH_EV: f64 = 1.0e-12;

/// Neutron mass in amu (SAMMY Kvendf=1 convention).
const NEUTRON_MASS_AMU: f64 = 1.008_664_915_6;

fn aux_debug_enabled() -> bool {
    std::env::var_os("NEREIDS_AUX_DEBUG").is_some()
}

/// SAMMY `Pointr`: index of largest `a[i] < b` in an ascending grid.
fn pointr(a: &[f64], b: f64) -> isize {
    if a.is_empty() {
        return -1;
    }
    if b < a[0] {
        return -1;
    }
    if b > a[a.len() - 1] {
        return a.len() as isize;
    }
    let idx = a.partition_point(|&x| x < b);
    idx as isize - 1
}

/// SAMMY `Fgpwid` resonance-centered refinement points.
fn fgpwid_points(center: f64, width: f64) -> Vec<f64> {
    // SAMMY defaults from InputInfoData: iptdop=9 (odd), iptwid=5.
    let iptdop = MIN_POINTS_ACROSS_WIDTH;
    let iptwid = EXTRA_TAIL_POINTS;
    let eg = 2.0 * width / ((iptdop - 1) as f64);
    let s = center - width - eg;

    let mut sadd = Vec::with_capacity(iptdop + 2 + 2 * iptwid);

    // -(iptwid+1)Γ ... -2Γ
    for nk in (1..=iptwid).rev() {
        sadd.push(center - ((nk + 1) as f64) * width);
    }

    // -1.5Γ, then -Γ
    if let Some(&last) = sadd.last() {
        sadd.push(last + 0.5 * width);
    }
    sadd.push(s + eg);

    // (iptdop-1) points from -Γ to +Γ with step eg.
    for _ in 2..=iptdop {
        let next = sadd[sadd.len() - 1] + eg;
        sadd.push(next);
    }

    // +1.5Γ and +2Γ
    let next = sadd[sadd.len() - 1] + 0.5 * width;
    sadd.push(next);
    let next = sadd[sadd.len() - 1] + 0.5 * width;
    sadd.push(next);

    // +3Γ ... +(iptwid+1)Γ
    if iptwid > 1 {
        let mut nk = 1;
        loop {
            nk += 1;
            let next = sadd[sadd.len() - 1] + width;
            sadd.push(next);
            if nk >= iptwid {
                break;
            }
        }
    }

    sadd
}

/// SAMMY `Qmerge`: merge dense resonance points into a base monotonic grid.
fn qmerge(base: &[f64], dense: &[f64], tol: f64) -> Vec<f64> {
    if base.is_empty() || dense.is_empty() {
        return base.to_vec();
    }

    // Keep only dense points that lie within base bounds.
    let mut ibs = 0usize;
    while ibs < dense.len() && dense[ibs] <= base[0] {
        ibs += 1;
    }
    let mut ibe = dense.len();
    while ibe > ibs && dense[ibe - 1] > base[base.len() - 1] {
        ibe -= 1;
    }
    if ibs >= ibe {
        return base.to_vec();
    }

    let ja = pointr(base, dense[ibs]);
    if ja < 0 {
        return base.to_vec();
    }
    let mut ka = (ja as usize) + 1;
    if ka >= base.len() {
        return base.to_vec();
    }

    let mut kb = ibs;
    let mut out = Vec::with_capacity(base.len() + (ibe - ibs) + 4);
    out.extend_from_slice(&base[..=ja as usize]);

    // Equivalent to Fortran label 170 path.
    if (base[ka - 1] - dense[kb]).abs() > tol {
        let d1 = dense[kb] - out[out.len() - 1];
        let d2 = base[ka] - dense[kb];
        if d1 >= d2 || out.len() <= 2 {
            out.push(out[out.len() - 1] + 0.75 * d1);
        } else {
            let tmp = out[out.len() - 1];
            let d3 = tmp - out[out.len() - 2];
            let last_idx = out.len() - 1;
            out[last_idx] = tmp - 0.25 * d3;
            out.push(tmp);
        }
    } else {
        kb += 1;
        if kb >= ibe {
            // Dense points exhausted; append rest of base.
            out.extend_from_slice(&base[ka..]);
            return out;
        }
    }

    loop {
        // Fortran label 70.
        if base[ka] > dense[kb] {
            // Fortran label 150: candidate dense point.
            if (dense[kb] - base[ka]).abs() <= tol {
                kb += 1;
                if kb >= ibe {
                    break;
                }
                continue;
            }
            out.push(dense[kb]);
            kb += 1;
            if kb >= ibe {
                break;
            }
            continue;
        }

        // Copy base point.
        out.push(base[ka]);
        ka += 1;
        if kb < ibe && (dense[kb] - base[ka - 1]).abs() <= tol {
            kb += 1;
            if kb >= ibe {
                break;
            }
        }
        if ka >= base.len() {
            return out;
        }
    }

    // Fortran label 100: append remaining base with one transition point.
    if ka >= base.len() {
        return out;
    }
    if ka == base.len() - 1 {
        out.push(base[ka]);
        return out;
    }

    let d1 = out[out.len() - 1] - out[out.len() - 2];
    let d2 = base[ka] - out[out.len() - 1];
    if d1 <= d2 {
        out.push(out[out.len() - 1] + 0.25 * d2);
    } else {
        out.push(base[ka]);
        ka += 1;
        if ka >= base.len() {
            return out;
        }
        let d1n = base[ka] - out[out.len() - 1];
        out.push(out[out.len() - 1] + 0.25 * d1n);
    }
    out.extend_from_slice(&base[ka..]);
    out
}

/// SAMMY `Fspken/Add_Pnts`-style local refinement around one resonance.
///
/// This complements `Fgpwid/Qmerge` by adding geometric transitions between
/// resonance-local fine spacing and the neighboring coarse data spacing.
fn add_fspken_transition_points(grid: &mut Vec<f64>, center: f64, width: f64) {
    if grid.len() < 2 || !(center.is_finite() && width.is_finite() && width > 0.0) {
        return;
    }

    let jres = pointr(grid, center);
    if jres < 0 {
        return;
    }
    let jres = jres as usize;
    if jres + 1 >= grid.len() {
        return;
    }

    let j_down = jres;
    let j_up = jres + 1;

    // SAMMY Fspken default: Fractn = 2 / (iptdop + 5), with iptdop=9.
    let eg = 2.0 * width / ((MIN_POINTS_ACROSS_WIDTH + EXTRA_TAIL_POINTS) as f64);
    if eg <= 0.0 {
        return;
    }

    let qmid = center;
    let qd = qmid - grid[j_down];
    let qu = grid[j_up] - qmid;

    let mut added = Vec::new();

    // Down-side points.
    if qd > 0.4 * eg {
        added.push(qmid - 0.2 * eg);
    }
    if qd / width > 2.0 {
        let kd = (width / eg).floor() as usize + 1;
        let egd = width / (kd as f64);
        for kk in 1..=kd {
            added.push(qmid - egd * (kk as f64));
        }
        let q = qmid - width;
        if q > grid[j_down] {
            let diff = q - grid[j_down];
            if diff / egd >= 4.0 {
                added.extend(geometric_transition_points(grid[j_down], diff, egd, 1.0));
            }
            added.push(grid[j_down]);
        }
    } else {
        if qd / eg > 1.2 {
            let kd = (qd / eg).floor() as usize + 1;
            let egd = qd / (kd as f64);
            if kd > 1 {
                for kk in 1..kd {
                    added.push(qmid - egd * (kk as f64));
                }
            }
        }
        added.push(grid[j_down]);
    }

    // Up-side points.
    if qu > 0.4 * eg {
        added.push(qmid + 0.2 * eg);
    }
    if qu / width >= 2.0 {
        let ku = (width / eg).floor() as usize + 1;
        let egu = width / (ku as f64);
        for kk in 1..=ku {
            added.push(qmid + egu * (kk as f64));
        }
        let q = qmid + width;
        if grid[j_up] > q {
            let diff = grid[j_up] - q;
            if diff / egu >= 4.0 {
                added.extend(geometric_transition_points(grid[j_up], diff, egu, -1.0));
            }
            added.push(grid[j_up]);
        }
    } else {
        if qu / eg > 1.2 {
            let ku = (qu / eg).floor() as usize + 1;
            let egu = qu / (ku as f64);
            if ku > 1 {
                for kk in 1..ku {
                    added.push(qmid + egu * (kk as f64));
                }
            }
        }
        added.push(grid[j_up]);
    }

    let before = grid.len();
    grid.extend(added.into_iter().filter(|e| e.is_finite()));
    grid.sort_unstable_by(f64::total_cmp);
    grid.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    if aux_debug_enabled() {
        eprintln!(
            "[aux-grid] fspken transitions at E={:.6e}: added={} total={}",
            center,
            grid.len().saturating_sub(before),
            grid.len()
        );
    }
}

/// Mirror SAMMY `DopplerAndResolutionBroadener::calcIntegralSpan` (oneExtra=true).
fn calc_integral_span(
    energies: &[f64],
    mut integral_start: usize,
    integral_pts: usize,
    elow: f64,
    ehigh: f64,
) -> (usize, usize) {
    let nener = energies.len();
    if nener == 0 {
        return (0, 0);
    }

    let mut ihigh = integral_start.saturating_add(integral_pts);
    if ihigh >= nener {
        ihigh = nener - 1;
    }
    if energies[ihigh] > ehigh {
        ihigh = 0;
    }

    if integral_start >= nener {
        integral_start = nener - 1;
    }

    // Assume ascending energies.
    while energies[integral_start] > elow && integral_start > 0 {
        integral_start -= 1;
    }

    let mut ee = energies[integral_start];
    for (i, &e) in energies.iter().enumerate().skip(integral_start) {
        if e > elow {
            break;
        }
        integral_start = i;
        ee = e;
    }

    // oneExtra=true path
    if integral_start > 0 {
        integral_start -= 1;
    } else if ee == elow && integral_start > 0 {
        integral_start -= 1;
    }

    let mut pts = 1usize;
    for i in integral_start..(nener - 1) {
        pts += 1;
        if i < ihigh {
            continue;
        }
        if energies[i] >= ehigh {
            break;
        }
    }

    (integral_start, pts)
}

/// Build extra points between two grid locations using a SAMMY-style
/// geometric-halving transition.
fn geometric_transition_points(
    start: f64,
    diff: f64,
    target_spacing: f64,
    direction: f64,
) -> Vec<f64> {
    if !(diff > 0.0 && target_spacing > 0.0) {
        return Vec::new();
    }

    let mut delta = diff;
    let mut cumulative = Vec::new();

    // Mirror mdat5 addedPointsUDGrad: repeatedly halve until comparable
    // to the target spacing, then accumulate partial sums.
    while cumulative.len() < 30 {
        delta *= 0.5;
        if delta < target_spacing {
            break;
        }
        cumulative.push(delta);
    }

    for i in 1..cumulative.len() {
        let prev = cumulative[i - 1];
        cumulative[i] += prev;
    }

    if let Some(&last) = cumulative.last() {
        if last > (diff - target_spacing) {
            cumulative.pop();
        }
    }

    cumulative
        .into_iter()
        .map(|x| start + direction * x)
        .filter(|x| x.is_finite())
        .collect()
}

/// Single SAMMY `RefineGrid`-style refinement pass for abrupt spacing transitions.
fn refine_auxiliary_grid_transitions(grid: &[f64]) -> Vec<f64> {
    if grid.len() < 3 {
        return grid.to_vec();
    }

    let mut new_grid = Vec::with_capacity(grid.len() + 32);
    new_grid.push(grid[0]);

    for i in 1..(grid.len() - 1) {
        let de3e2 = grid[i + 1] - grid[i];
        let de2e1 = grid[i] - new_grid[new_grid.len() - 1];
        if de2e1 <= 0.0 || de3e2 <= 0.0 {
            new_grid.push(grid[i]);
            continue;
        }

        if (de2e1 / de3e2) > AUX_SPACING_RATIO_LIMIT {
            // Previous interval is much larger than next; add points in [prev, curr].
            let estart = new_grid[new_grid.len() - 1];
            let mut diff = de2e1;
            if new_grid.len() > 1 {
                diff = estart - new_grid[new_grid.len() - 2];
            }
            let de2e1_new = de2e1 * 0.5;

            // SAMMY special case: if this creates another abrupt jump, retroactively
            // smooth the previous interval too.
            if new_grid.len() > 1 && de2e1_new > 0.0 && (diff / de2e1_new) > AUX_SPACING_RATIO_LIMIT
            {
                let estart_prev = new_grid[new_grid.len() - 2];
                let rethink = geometric_transition_points(estart_prev, diff, de2e1_new, 1.0);
                if !rethink.is_empty() {
                    let old_last = new_grid[new_grid.len() - 1];
                    let last_idx = new_grid.len() - 1;
                    new_grid[last_idx] = rethink[0];
                    new_grid.extend(rethink.into_iter().skip(1));
                    new_grid.push(old_last);
                }
            }

            let added = geometric_transition_points(estart, de2e1, de3e2, 1.0);
            new_grid.extend(added);
            new_grid.push(grid[i]);
        } else if (de3e2 / de2e1) > AUX_SPACING_RATIO_LIMIT {
            // Next interval is much larger than previous; add points in [curr, next].
            new_grid.push(grid[i]);
            let added = geometric_transition_points(grid[i + 1], de3e2, de2e1, -1.0);
            new_grid.extend(added.into_iter().rev());
        } else {
            new_grid.push(grid[i]);
        }
    }

    new_grid.push(grid[grid.len() - 1]);
    new_grid
}

/// SAMMY `Adjust_Auxil/calculateWeights` diagnostic weights.
fn calculate_auxiliary_weights(grid: &[f64]) -> (Vec<f64>, f64, usize) {
    let n = grid.len();
    if n == 0 {
        return (Vec::new(), 0.0, 0);
    }
    let mut weights = vec![0.0; n];
    let mut sum_pos = 0.0;
    let mut num_pos = 0usize;
    let mut neg = 0usize;

    for k in 0..n {
        let mut x21_1 = 0.0;
        let mut x21_2 = 0.0;
        let mut x21_3 = 0.0;
        let mut x21_4 = 0.0;
        let mut x21_1_2 = 0.0;
        let mut x21_2_2 = 0.0;
        let mut x21_3_2 = 0.0;
        let mut x21_4_2 = 0.0;
        let mut x2_1 = 0.0;
        let mut x2_2 = 0.0;
        let mut r1 = 0.0;
        let mut r2 = 0.0;

        if k > 1 {
            x21_1 = grid[k - 1] - grid[k - 2];
            x21_1_2 = x21_1 * x21_1;
        }
        if k > 0 {
            x21_2 = grid[k] - grid[k - 1];
            x21_2_2 = x21_2 * x21_2;
            if x21_2 != 0.0 {
                r1 = 1.0 / x21_2;
            }
        }
        if k + 1 < n {
            x21_3 = grid[k + 1] - grid[k];
            x21_3_2 = x21_3 * x21_3;
            if x21_3 != 0.0 {
                r2 = 1.0 / x21_3;
            }
        }
        if k + 2 < n {
            x21_4 = grid[k + 2] - grid[k + 1];
            x21_4_2 = x21_4 * x21_4;
        }

        if k > 1 {
            x2_1 = r1 * (x21_3_2 - x21_1_2);
        }
        if k + 1 >= n - 1 {
            x2_1 = x21_1_2;
        }

        if k > 0 {
            x2_2 = r2 * (x21_4_2 - x21_2_2);
        }
        if k >= n - 2 {
            x2_2 = x21_2_2;
        }

        let mut w = x21_1 + 5.0 * x21_2 + 5.0 * x21_3 + x21_4 + x2_1 - x2_2;
        if k == n.saturating_sub(2) {
            w = x21_1 + 5.0 * x21_2 + 5.0 * x21_3 + x2_1;
        }
        if k == n.saturating_sub(1) {
            w = x21_1 + 5.0 * x21_2;
        }
        weights[k] = w;

        if (1..(n.saturating_sub(1))).contains(&k) {
            if w > 0.0 {
                sum_pos += w;
                num_pos += 1;
            } else {
                neg += 1;
            }
        }
    }

    let average = if num_pos > 0 {
        -sum_pos / (num_pos as f64)
    } else {
        0.0
    };
    (weights, average, neg)
}

/// Compute the Doppler width parameter Δ(E) for the free-gas model.
///
/// # Formula
///
/// ```text
/// Δ = sqrt(4 · k_B · T · |E| / A_target)
/// ```
///
/// where `A_target` is the target mass in amu (SAMMY `DefTargetMass` convention).
pub fn doppler_width(energy_ev: f64, temp_k: f64, awr: f64) -> f64 {
    // Use |E| so negative incident energies (supported by Reich-Moore) produce
    // a real Doppler width instead of NaN.
    (4.0 * BOLTZMANN_EV_PER_K * temp_k * energy_ev.abs() / awr).sqrt()
}

/// Apply free-gas Doppler broadening to cross sections on the same grid.
///
/// This is a convenience wrapper around [`broaden_cross_sections_to_grid`]
/// with identical source/target energy grids.
pub fn broaden_cross_sections(
    xs_0k: &[f64],
    energies: &[f64],
    awr: f64,
    temperature_k: f64,
) -> Result<Vec<f64>, PhysicsError> {
    broaden_cross_sections_to_grid(xs_0k, energies, energies, awr, temperature_k)
}

/// Apply free-gas Doppler broadening from a source grid onto a target grid.
///
/// Convolves `xs_0k_source(E'; 0K)` over `source_energies` and evaluates the
/// broadened result on `target_energies`. This matches SAMMY's workflow where
/// auxiliary-grid values are integrated directly onto the data grid.
pub fn broaden_cross_sections_to_grid(
    xs_0k_source: &[f64],
    source_energies: &[f64],
    target_energies: &[f64],
    awr: f64,
    temperature_k: f64,
) -> Result<Vec<f64>, PhysicsError> {
    let n_src = source_energies.len();
    let n_tgt = target_energies.len();

    if n_src == 0 || n_tgt == 0 {
        return Err(PhysicsError::EmptyEnergyGrid);
    }

    if xs_0k_source.len() != n_src {
        return Err(PhysicsError::DimensionMismatch {
            expected: n_src,
            got: xs_0k_source.len(),
        });
    }
    if source_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "source energies must be finite".to_string(),
        ));
    }
    if source_energies.windows(2).any(|w| w[1] < w[0]) {
        return Err(PhysicsError::InvalidParameter(
            "source energies must be sorted ascending".to_string(),
        ));
    }
    if target_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "target energies must be finite".to_string(),
        ));
    }
    if target_energies.windows(2).any(|w| w[1] < w[0]) {
        return Err(PhysicsError::InvalidParameter(
            "target energies must be sorted ascending".to_string(),
        ));
    }

    if awr <= 0.0 || !awr.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "AWR must be positive and finite, got {awr}"
        )));
    }

    if temperature_k < 0.0 || !temperature_k.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "temperature must be non-negative and finite, got {temperature_k}"
        )));
    }

    // No broadening at T = 0
    if temperature_k == 0.0 {
        if source_energies == target_energies {
            return Ok(xs_0k_source.to_vec());
        }
        return interpolate_to_grid(source_energies, xs_0k_source, target_energies);
    }

    let ddo = (BOLTZMANN_EV_PER_K * temperature_k * NEUTRON_MASS_AMU / awr).sqrt();
    if ddo < 1e-20 {
        if source_energies == target_energies {
            return Ok(xs_0k_source.to_vec());
        }
        return interpolate_to_grid(source_energies, xs_0k_source, target_energies);
    }
    let inv_ddo_sq = 1.0 / (ddo * ddo);
    let source_velocities: Vec<f64> = source_energies
        .iter()
        .map(|&e| if e < 0.0 { -(-e).sqrt() } else { e.sqrt() })
        .collect();

    let mut broadened = Vec::with_capacity(n_tgt);
    let mut total_segments: usize = 0;
    let mut integral_start: usize = 0;
    let mut integral_pts: usize = 0;

    for &e_tgt in target_energies {
        let v_i = if e_tgt < 0.0 {
            -(-e_tgt).sqrt()
        } else {
            e_tgt.sqrt()
        };
        let mut v_low = v_i - KERNEL_CUTOFF * ddo;
        let mut v_up = v_i + KERNEL_CUTOFF * ddo;
        if v_low < source_velocities[0] {
            v_low = source_velocities[0];
        }
        if v_up > source_velocities[n_src - 1] {
            v_up = source_velocities[n_src - 1];
        }

        let mut e_low = v_low * v_low;
        if v_low < 0.0 {
            e_low = -e_low;
        }
        let mut e_high = v_up * v_up;
        if v_up < 0.0 {
            e_high = -e_high;
        }

        let (j_lo, ipnts) =
            calc_integral_span(source_energies, integral_start, integral_pts, e_low, e_high);
        integral_start = j_lo;
        integral_pts = ipnts;
        let j_hi = j_lo + ipnts.saturating_sub(1);

        if ipnts <= 2 || j_hi >= n_src {
            // Not enough points for stable broadening (SAMMY requires >2 points)
            let val = interpolate_to_grid(source_energies, xs_0k_source, &[e_tgt])?[0];
            broadened.push(val);
            continue;
        }
        total_segments = total_segments.saturating_add(ipnts - 1);
        if total_segments > MAX_BROADENING_SEGMENTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "broadening workload too large for direct trapezoidal convolution (>{MAX_BROADENING_SEGMENTS} segments); reduce auxiliary-grid density"
            )));
        }

        let e_i_abs = e_tgt.abs();

        if e_i_abs <= f64::EPSILON {
            let val = interpolate_to_grid(source_energies, xs_0k_source, &[e_tgt])?[0];
            broadened.push(val);
            continue;
        }

        let use_fgm_weights = std::env::var_os("NEREIDS_DISABLE_FGM_WEIGHTS").is_none();
        if use_fgm_weights {
            if let Some(sum) =
                fgm_weighted_sum(&source_velocities, xs_0k_source, j_lo, ipnts, v_i, ddo)
            {
                // Match SAMMY Xdofgm post-processing: divide weighted sum by |E|.
                broadened.push(sum / e_i_abs);
                continue;
            }
        }

        // Fallback: normalized Gaussian trapezoid if weight generation fails.
        let mut sum_weighted = 0.0;
        let mut sum_kernel = 0.0;
        for j in (j_lo + 1)..=j_hi {
            let h = source_velocities[j] - source_velocities[j - 1];
            if h <= 0.0 {
                continue;
            }
            let diff_prev = v_i - source_velocities[j - 1];
            let diff_curr = v_i - source_velocities[j];
            let g_prev = (-diff_prev * diff_prev * inv_ddo_sq).exp();
            let g_curr = (-diff_curr * diff_curr * inv_ddo_sq).exp();
            let e_prev_abs = source_energies[j - 1].abs();
            let e_curr_abs = source_energies[j].abs();
            sum_weighted += 0.5
                * h
                * (g_prev * e_prev_abs * xs_0k_source[j - 1]
                    + g_curr * e_curr_abs * xs_0k_source[j]);
            sum_kernel += 0.5 * h * (g_prev + g_curr);
        }
        broadened.push(if sum_kernel > 1e-100 {
            (sum_weighted / sum_kernel) / e_i_abs
        } else {
            interpolate_to_grid(source_energies, xs_0k_source, &[e_tgt])?[0]
        });
    }

    Ok(broadened)
}

/// Linearly interpolate values from a source grid to target energies.
///
/// Values outside the source range are clamped to the nearest edge value.
///
/// # Arguments
///
/// * `source_energies` - Source energy grid (sorted ascending)
/// * `source_values` - Source values, same length as `source_energies`
/// * `target_energies` - Target energies to interpolate to (finite)
pub fn interpolate_to_grid(
    source_energies: &[f64],
    source_values: &[f64],
    target_energies: &[f64],
) -> Result<Vec<f64>, PhysicsError> {
    if source_energies.is_empty() {
        return Err(PhysicsError::EmptyEnergyGrid);
    }
    if source_values.len() != source_energies.len() {
        return Err(PhysicsError::DimensionMismatch {
            expected: source_energies.len(),
            got: source_values.len(),
        });
    }
    if source_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "source energies must be finite".to_string(),
        ));
    }
    if source_energies.windows(2).any(|w| w[1] < w[0]) {
        return Err(PhysicsError::InvalidParameter(
            "source energies must be sorted ascending".to_string(),
        ));
    }
    if target_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "target energies must be finite".to_string(),
        ));
    }

    let n_src = source_energies.len();
    Ok(target_energies
        .iter()
        .map(|&e| {
            let idx = source_energies.partition_point(|&x| x < e);
            if idx == 0 {
                source_values[0]
            } else if idx >= n_src {
                source_values[n_src - 1]
            } else {
                let e0 = source_energies[idx - 1];
                let e1 = source_energies[idx];
                let denom = e1 - e0;
                if denom == 0.0 {
                    // Adjacent duplicate energies; use right-endpoint value
                    source_values[idx]
                } else {
                    let t = (e - e0) / denom;
                    source_values[idx - 1] + t * (source_values[idx] - source_values[idx - 1])
                }
            }
        })
        .collect())
}

/// Create an auxiliary energy grid that resolves narrow resonances.
///
/// Adds fine-spaced points near each resonance center to ensure the 0K
/// cross section structure is captured for subsequent Doppler broadening.
///
/// # Arguments
///
/// * `data_energies` - Data/output energy grid
/// * `resonance_energies` - Resonance center energies \[eV\]
/// * `resonance_widths` - Total resonance widths \[eV\] (Γ = Γγ + Γn + ...)
/// * `temperature_k` - Sample temperature in Kelvin
/// * `awr` - Target mass in amu (SAMMY `DefTargetMass`)
///
/// # Returns
///
/// Sorted, deduplicated energy grid that includes both data points and
/// fine points near resonances.
///
/// # Errors
///
/// Returns `PhysicsError::InvalidParameter` if any energy is non-finite.
pub fn create_auxiliary_grid(
    data_energies: &[f64],
    resonance_energies: &[f64],
    resonance_widths: &[f64],
    temperature_k: f64,
    awr: f64,
) -> Result<Vec<f64>, PhysicsError> {
    if resonance_energies.len() != resonance_widths.len() {
        return Err(PhysicsError::DimensionMismatch {
            expected: resonance_energies.len(),
            got: resonance_widths.len(),
        });
    }
    if data_energies.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "data energies must be finite".to_string(),
        ));
    }
    if temperature_k < 0.0 || !temperature_k.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "temperature_k must be non-negative and finite, got {temperature_k}"
        )));
    }
    if awr <= 0.0 || !awr.is_finite() {
        return Err(PhysicsError::InvalidParameter(format!(
            "awr must be positive and finite, got {awr}"
        )));
    }

    let mut grid: Vec<f64> = data_energies.to_vec();
    // Use sorted unique view while building resonance-centered refinements.
    grid.sort_unstable_by(f64::total_cmp);
    grid.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    if aux_debug_enabled() {
        eprintln!(
            "[aux-grid] start: data={} unique_sorted={}",
            data_energies.len(),
            grid.len()
        );
    }

    for (&e_res, &gamma) in resonance_energies.iter().zip(resonance_widths.iter()) {
        if !e_res.is_finite() || !gamma.is_finite() {
            return Err(PhysicsError::InvalidParameter(format!(
                "resonance energy and width must be finite, got E={e_res}, Γ={gamma}"
            )));
        }
        if gamma < 0.0 {
            return Err(PhysicsError::InvalidParameter(format!(
                "resonance width must be non-negative, got Γ={gamma}"
            )));
        }
        if gamma == 0.0 {
            continue;
        }

        let width = gamma.max(MIN_REFINEMENT_WIDTH_EV);
        if grid.is_empty() {
            continue;
        }

        // SAMMY Weeres criterion: if there are not enough points in [E-Γ, E+Γ],
        // generate Fgpwid points and merge with transition points (Qmerge).
        let e_lo = e_res - width;
        let e_hi = e_res + width;
        let needs_refine = if grid[0] <= e_lo {
            let k = pointr(&grid, e_lo);
            let k = if k < 0 { 0usize } else { k as usize };
            let k_check = k + MIN_POINTS_ACROSS_WIDTH + 1;
            !(k_check < grid.len() && grid[k_check] <= e_hi)
        } else {
            true
        };

        if needs_refine {
            let use_fspken = data_energies.len() >= (MIN_POINTS_ACROSS_WIDTH + 2);
            let mut fspken_added = 0usize;
            if use_fspken {
                let before_fspken = grid.len();
                add_fspken_transition_points(&mut grid, e_res, width);
                fspken_added = grid.len().saturating_sub(before_fspken);
            }
            let mut qmerge_added = 0usize;
            if fspken_added == 0 {
                let dense = fgpwid_points(e_res, width);
                let eg = 2.0 * width / ((MIN_POINTS_ACROSS_WIDTH - 1) as f64);
                let merged = qmerge(&grid, &dense, eg * 0.2);
                qmerge_added = merged.len().saturating_sub(grid.len());
                grid = merged;
            }
            if aux_debug_enabled() {
                eprintln!(
                    "[aux-grid] resonance E={:.6e} Gamma={:.6e}: qmerge_added={} fspken_added={} total={}",
                    e_res,
                    width,
                    qmerge_added,
                    fspken_added,
                    grid.len()
                );
            }
        } else if aux_debug_enabled() {
            eprintln!(
                "[aux-grid] resonance E={:.6e} Gamma={:.6e}: refine=no total={}",
                e_res,
                width,
                grid.len()
            );
        }

        if grid.len() > MAX_AUX_GRID_POINTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "auxiliary grid exceeds {} points; reduce resonance density/width inputs",
                MAX_AUX_GRID_POINTS
            )));
        }
    }

    // For T>0, extend the grid in sqrt(E) using SAMMY-like edge spacing.
    //
    // This mirrors the free-gas Escale/Vqcon behavior: derive a velocity
    // step from the first/last five data points and populate extra points
    // between [Emind, E_min) and (E_max, Emaxd].
    if temperature_k > 0.0 {
        let before_temp = grid.len();
        let mut sorted_data = data_energies.to_vec();
        sorted_data.sort_unstable_by(f64::total_cmp);

        if let (Some(&min_e), Some(&max_e)) = (sorted_data.first(), sorted_data.last()) {
            let ddo = (BOLTZMANN_EV_PER_K * temperature_k * NEUTRON_MASS_AMU / awr).sqrt();
            if ddo > 0.0 && min_e > 0.0 && max_e > 0.0 {
                let v_min = min_e.sqrt();
                let v_max = max_e.sqrt();
                let bound_step = KERNEL_CUTOFF * AUX_BOUNDS_FUDGE * ddo;

                let emind = (v_min - bound_step).powi(2);
                let emaxd = (v_max + bound_step).powi(2);

                let n5 = sorted_data.len().min(5);
                if n5 >= 2 {
                    let e1 = sorted_data[0];
                    let e2 = sorted_data[1];
                    let e5_low = sorted_data[n5 - 1];
                    let d_low = (e5_low.sqrt() - e1.sqrt()) / ((n5 - 1) as f64);
                    if d_low > 0.0 {
                        let eefudg = (e2 - e1) * 1.0e-4;
                        if e1 >= emind + eefudg {
                            let x = e1.sqrt() - emind.sqrt();
                            let n = (x / d_low).floor() as usize + 1;
                            let start = e1.sqrt() - ((n + 1) as f64) * d_low;
                            for i in 1..=n {
                                let v = start + (i as f64) * d_low;
                                let e = v * v;
                                if e.is_finite() {
                                    grid.push(e);
                                }
                            }
                        }
                    }

                    let e_last = sorted_data[sorted_data.len() - 1];
                    let e5_high = sorted_data[sorted_data.len() - n5];
                    let d_high = (e_last.sqrt() - e5_high.sqrt()) / ((n5 - 1) as f64);
                    if d_high > 0.0 {
                        let x = emaxd.sqrt() - e_last.sqrt();
                        let n = (x / d_high).floor() as usize + 1;
                        for i in 1..=n {
                            let v = e_last.sqrt() + (i as f64) * d_high;
                            let e = v * v;
                            if e.is_finite() {
                                grid.push(e);
                            }
                        }
                    }
                }
            }
        }

        if grid.len() > MAX_AUX_GRID_POINTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "auxiliary grid exceeds {} points; reduce resonance density/width inputs",
                MAX_AUX_GRID_POINTS
            )));
        }
        if aux_debug_enabled() {
            eprintln!(
                "[aux-grid] edge extension: added={} total={}",
                grid.len().saturating_sub(before_temp),
                grid.len()
            );
        }
    }

    // Sort and deduplicate (within tolerance)
    if grid.iter().any(|e| !e.is_finite()) {
        return Err(PhysicsError::InvalidParameter(
            "non-finite energy in grid".to_string(),
        ));
    }
    grid.sort_unstable_by(f64::total_cmp);
    grid.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    if aux_debug_enabled() {
        eprintln!("[aux-grid] pre-smoothing sorted+dedup total={}", grid.len());
    }

    // Smooth abrupt spacing jumps using SAMMY `RefineGrid`-style transitions.
    for pass in 0..AUX_SMOOTHING_PASSES {
        let previous_len = grid.len();
        let mut refined = refine_auxiliary_grid_transitions(&grid);
        refined.sort_unstable_by(f64::total_cmp);
        refined.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        let added_count = refined.len().saturating_sub(previous_len);
        grid = refined;
        if added_count == 0 {
            if aux_debug_enabled() {
                eprintln!("[aux-grid] smoothing pass {}: no additions", pass + 1);
            }
            break;
        }
        if grid.len() > MAX_AUX_GRID_POINTS {
            return Err(PhysicsError::InvalidParameter(format!(
                "auxiliary grid exceeds {} points; reduce resonance density/width inputs",
                MAX_AUX_GRID_POINTS
            )));
        }
        if aux_debug_enabled() {
            eprintln!(
                "[aux-grid] smoothing pass {}: added={} total={}",
                pass + 1,
                added_count,
                grid.len()
            );
        }
    }
    if aux_debug_enabled() {
        let (weights, avg, neg) = calculate_auxiliary_weights(&grid);
        let flagged = weights
            .iter()
            .skip(1)
            .take(grid.len().saturating_sub(2))
            .filter(|&&w| w <= avg)
            .count();
        eprintln!("[aux-grid] final total={}", grid.len());
        eprintln!(
            "[aux-grid] weights: negative={} flagged(<=avg={:.6e})={}",
            neg, avg, flagged
        );
    }

    Ok(grid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        if n <= 1 {
            return vec![start];
        }
        let h = (end - start) / ((n - 1) as f64);
        (0..n).map(|i| start + (i as f64) * h).collect()
    }

    fn bruteforce_velocity_convolution(
        target_e: f64,
        awr: f64,
        temperature_k: f64,
        e_min: f64,
        e_max: f64,
        sigma0: impl Fn(f64) -> f64,
    ) -> f64 {
        let vi = target_e.sqrt();
        let ddo = (BOLTZMANN_EV_PER_K * temperature_k * NEUTRON_MASS_AMU / awr).sqrt();
        let vmin = e_min.sqrt().max(vi - KERNEL_CUTOFF * ddo);
        let vmax = e_max.sqrt().min(vi + KERNEL_CUTOFF * ddo);
        let n = 20_000usize;
        let h = (vmax - vmin) / (n as f64);

        let mut sum_w = 0.0;
        let mut sum_k = 0.0;
        for i in 0..=n {
            let v = vmin + (i as f64) * h;
            let e = v * v;
            let g = (-((vi - v) / ddo).powi(2)).exp();
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            sum_w += weight * g * e * sigma0(e);
            sum_k += weight * g;
        }
        let sum_w = sum_w * h;
        let sum_k = sum_k * h;
        (sum_w / sum_k) / target_e
    }

    fn naive_energy_gaussian_broadening(
        xs_0k: &[f64],
        energies: &[f64],
        target_energies: &[f64],
        awr: f64,
        temperature_k: f64,
    ) -> Vec<f64> {
        let mut out = Vec::with_capacity(target_energies.len());
        for &e_tgt in target_energies {
            let delta = doppler_width(e_tgt, temperature_k, awr).max(1e-16);
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 1..energies.len() {
                let e0 = energies[j - 1];
                let e1 = energies[j];
                let h = e1 - e0;
                let g0 = (-((e_tgt - e0) / delta).powi(2)).exp();
                let g1 = (-((e_tgt - e1) / delta).powi(2)).exp();
                num += 0.5 * h * (g0 * xs_0k[j - 1] + g1 * xs_0k[j]);
                den += 0.5 * h * (g0 + g1);
            }
            out.push(if den > 0.0 { num / den } else { xs_0k[0] });
        }
        out
    }

    #[test]
    fn test_doppler_width_values() {
        // At E=10 eV, T=300K, AWR=10:
        // Δ = sqrt(4 * 8.617e-5 * 300 * 10 / 10) = sqrt(0.10340) ≈ 0.3216
        let d = doppler_width(10.0, 300.0, 10.0);
        assert!((d - 0.3216).abs() < 0.001, "got {d}");
    }

    #[test]
    fn test_doppler_width_at_50k() {
        // At E=10 eV, T=50K, AWR=10:
        // Δ = sqrt(4 * 8.617e-5 * 50 * 10 / 10) = sqrt(0.01723) ≈ 0.1313
        let d = doppler_width(10.0, 50.0, 10.0);
        assert!((d - 0.1313).abs() < 0.001, "got {d}");
    }

    #[test]
    fn test_doppler_width_zero_temp() {
        assert_eq!(doppler_width(10.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn test_broaden_zero_temperature() {
        let xs = vec![1.0, 2.0, 3.0];
        let energies = vec![1.0, 2.0, 3.0];

        let result = broaden_cross_sections(&xs, &energies, 10.0, 0.0).unwrap();
        assert_eq!(result, xs);
    }

    #[test]
    fn test_broaden_constant_xs_unchanged() {
        // A constant cross section should remain constant after broadening.
        let n = 200;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + i as f64 * 0.05).collect();
        let xs = vec![100.0; n];

        let result = broaden_cross_sections(&xs, &energies, 10.0, 300.0).unwrap();
        // Interior points (away from edges) should be very close to 100.
        for (i, &r) in result.iter().enumerate().skip(20).take(n - 40) {
            assert!(
                (r - 100.0).abs() < 0.1,
                "constant XS changed at i={i}: got {r}"
            );
        }
    }

    #[test]
    fn test_broaden_empty_grid_errors() {
        let result = broaden_cross_sections(&[], &[], 10.0, 300.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_broaden_invalid_awr_errors() {
        let result = broaden_cross_sections(&[1.0], &[10.0], -1.0, 300.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_broaden_dimension_mismatch_errors() {
        let result = broaden_cross_sections(&[1.0, 2.0], &[10.0], 10.0, 300.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_broaden_unsorted_energies_error() {
        let result = broaden_cross_sections(&[1.0, 2.0], &[2.0, 1.0], 10.0, 300.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_broaden_non_finite_energies_error() {
        let result = broaden_cross_sections(&[1.0, 2.0], &[1.0, f64::NAN], 10.0, 300.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_broaden_peak_spreads() {
        // A delta-like peak on a grid should spread after broadening.
        let n = 401;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + i as f64 * 0.025).collect();
        let mid = n / 2;
        let mut xs = vec![0.0; n];
        xs[mid] = 1000.0;

        let result = broaden_cross_sections(&xs, &energies, 10.0, 300.0).unwrap();
        // Peak should be reduced
        assert!(result[mid] < 1000.0);
        // Neighbors should pick up value
        assert!(result[mid - 1] > 0.0);
        assert!(result[mid + 1] > 0.0);
    }

    #[test]
    fn test_broaden_preserves_area() {
        // A localized feature should approximately preserve its area.
        let n = 1001;
        let energies: Vec<f64> = (0..n).map(|i| 5.0 + i as f64 * 0.01).collect();
        let mid = n / 2;

        // Create a Gaussian-like feature
        let sigma = 0.1;
        let e_mid = energies[mid];
        let xs: Vec<f64> = energies
            .iter()
            .map(|&e| 1000.0 * (-(e - e_mid).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();

        // Compute area before broadening
        let area_before: f64 = (1..n)
            .map(|i| 0.5 * (energies[i] - energies[i - 1]) * (xs[i] + xs[i - 1]))
            .sum();

        let result = broaden_cross_sections(&xs, &energies, 10.0, 300.0).unwrap();

        // Compute area after broadening
        let area_after: f64 = (1..n)
            .map(|i| 0.5 * (energies[i] - energies[i - 1]) * (result[i] + result[i - 1]))
            .sum();

        let rel_error = (area_after - area_before).abs() / area_before;
        assert!(
            rel_error < 0.01,
            "area not preserved: before={area_before:.4}, after={area_after:.4}, rel_err={rel_error:.6}"
        );
    }

    #[test]
    fn test_broaden_matches_bruteforce_velocity_convolution() {
        let awr = 10.0;
        let temperature_k = 300.0;
        let energies = linspace(1.0, 25.0, 6001);
        let sigma0 = |e: f64| 5.0 + 0.8 * (0.7 * e.sqrt()).sin() + 0.3 * (0.2 * e).cos() + 0.02 * e;
        let xs_0k: Vec<f64> = energies.iter().map(|&e| sigma0(e)).collect();
        let targets = vec![2.0, 5.0, 10.0, 20.0];

        let model = broaden_cross_sections_to_grid(&xs_0k, &energies, &targets, awr, temperature_k)
            .unwrap();
        let reference: Vec<f64> = targets
            .iter()
            .map(|&e| bruteforce_velocity_convolution(e, awr, temperature_k, 1.0, 25.0, sigma0))
            .collect();

        for (m, r) in model.iter().zip(reference.iter()) {
            let rel = (m - r).abs() / r.abs().max(1e-12);
            assert!(rel < 6e-3, "model={m}, reference={r}, rel={rel}");
        }
    }

    #[test]
    fn test_broaden_preserves_one_over_e_law() {
        let awr = 10.0;
        let temperature_k = 300.0;
        let source = linspace(0.5, 30.0, 4001);
        let xs_0k: Vec<f64> = source.iter().map(|&e| 1.0 / e).collect();
        let target = linspace(1.0, 25.0, 80);
        let broad =
            broaden_cross_sections_to_grid(&xs_0k, &source, &target, awr, temperature_k).unwrap();

        for (&e, &b) in target.iter().zip(broad.iter()) {
            let expected = 1.0 / e;
            let rel = (b - expected).abs() / expected;
            assert!(rel < 7e-3, "E={e}, got={b}, expected={expected}, rel={rel}");
        }
    }

    #[test]
    fn test_naive_energy_gaussian_breaks_one_over_e_law() {
        let awr = 10.0;
        let temperature_k = 300.0;
        let source = linspace(0.5, 30.0, 4001);
        let xs_0k: Vec<f64> = source.iter().map(|&e| 1.0 / e).collect();
        let target = linspace(1.0, 25.0, 80);

        let fgm =
            broaden_cross_sections_to_grid(&xs_0k, &source, &target, awr, temperature_k).unwrap();
        let naive = naive_energy_gaussian_broadening(&xs_0k, &source, &target, awr, temperature_k);

        let max_rel_between_methods = fgm
            .iter()
            .zip(naive.iter())
            .map(|(a, b)| (a - b).abs() / a.abs().max(1e-12))
            .fold(0.0, f64::max);
        let max_rel_naive_vs_expected = target
            .iter()
            .zip(naive.iter())
            .map(|(&e, &n)| (n - 1.0 / e).abs() / (1.0 / e))
            .fold(0.0, f64::max);

        assert!(
            max_rel_between_methods > 3e-3,
            "naive and velocity-space methods unexpectedly close: max_rel_between_methods={max_rel_between_methods}"
        );
        assert!(
            max_rel_naive_vs_expected > 3e-3,
            "naive method unexpectedly preserved 1/E law: max_rel_naive_vs_expected={max_rel_naive_vs_expected}"
        );
    }

    #[test]
    fn test_interpolate_basic() {
        let src_e = vec![1.0, 2.0, 3.0, 4.0];
        let src_xs = vec![10.0, 20.0, 30.0, 40.0];
        let tgt_e = vec![1.5, 2.5, 3.5];

        let result = interpolate_to_grid(&src_e, &src_xs, &tgt_e).unwrap();
        assert!((result[0] - 15.0).abs() < 1e-10);
        assert!((result[1] - 25.0).abs() < 1e-10);
        assert!((result[2] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_extrapolation() {
        let src_e = vec![2.0, 4.0];
        let src_xs = vec![20.0, 40.0];

        // Below range: returns first value
        let result = interpolate_to_grid(&src_e, &src_xs, &[1.0]).unwrap();
        assert!((result[0] - 20.0).abs() < 1e-10);

        // Above range: returns last value
        let result = interpolate_to_grid(&src_e, &src_xs, &[5.0]).unwrap();
        assert!((result[0] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_empty_source_errors() {
        let result = interpolate_to_grid(&[], &[], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_mismatched_source_errors() {
        let result = interpolate_to_grid(&[1.0, 2.0], &[10.0], &[1.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_unsorted_source_energies_error() {
        let result = interpolate_to_grid(&[2.0, 1.0], &[20.0, 10.0], &[1.5]);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_interpolate_non_finite_target_energies_error() {
        let result = interpolate_to_grid(&[1.0, 2.0], &[10.0, 20.0], &[f64::NAN]);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_includes_data_points() {
        let data = vec![8.0, 9.0, 10.0, 11.0, 12.0];
        let res_e = vec![10.0];
        let res_w = vec![0.001]; // 1 meV width
        let grid = create_auxiliary_grid(&data, &res_e, &res_w, 50.0, 10.0).unwrap();

        // Should include original data points
        for &e in &data {
            assert!(
                grid.iter().any(|&g| (g - e).abs() < 1e-9),
                "data point {e} missing from auxiliary grid"
            );
        }

        // Refinement should add at least a few points.
        assert!(grid.len() > data.len());
    }

    #[test]
    fn test_create_auxiliary_grid_resolves_resonance() {
        let data = vec![9.0, 11.0];
        let res_e = vec![10.0];
        let res_w = vec![0.001]; // 1 meV width
        let grid = create_auxiliary_grid(&data, &res_e, &res_w, 0.0, 10.0).unwrap();

        // Near the resonance, we should have at least SAMMY's default
        // "9 points across width" coverage.
        let near_res: Vec<f64> = grid
            .iter()
            .copied()
            .filter(|&e| (e - 10.0).abs() <= 1.1e-3)
            .collect();
        assert!(
            near_res.len() >= MIN_POINTS_ACROSS_WIDTH,
            "not enough points near resonance: {}",
            near_res.len()
        );
    }

    #[test]
    fn test_create_auxiliary_grid_mismatched_resonance_arrays_error() {
        let result = create_auxiliary_grid(&[8.0, 12.0], &[10.0, 11.0], &[0.001], 50.0, 10.0);
        assert!(matches!(
            result,
            Err(PhysicsError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn test_create_auxiliary_grid_non_finite_data_energy_error() {
        let result = create_auxiliary_grid(&[8.0, f64::NAN, 12.0], &[10.0], &[0.001], 50.0, 10.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_invalid_temperature_error() {
        let result = create_auxiliary_grid(&[8.0, 12.0], &[10.0], &[0.001], -1.0, 10.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_invalid_awr_error() {
        let result = create_auxiliary_grid(&[8.0, 12.0], &[10.0], &[0.001], 50.0, 0.0);
        assert!(matches!(result, Err(PhysicsError::InvalidParameter(_))));
    }

    #[test]
    fn test_create_auxiliary_grid_uses_true_min_from_unsorted_data() {
        let data = vec![12.0, 8.0, 10.0];
        let grid = create_auxiliary_grid(&data, &[9.0], &[0.001], 0.0, 10.0).unwrap();
        assert!(
            grid.iter().any(|&e| e < 9.0),
            "auxiliary grid should include points below resonance when min(data)<E_res"
        );
    }

    #[test]
    fn test_create_auxiliary_grid_bounded_for_narrow_resonance() {
        let data: Vec<f64> = (0..31).map(|i| 9.7 + i as f64 * 0.02).collect();
        let grid = create_auxiliary_grid(&data, &[10.0], &[1e-5], 300.0, 1.0).unwrap();
        assert!(
            grid.len() < 1_000,
            "grid unexpectedly large for narrow-resonance refinement: len={}",
            grid.len()
        );
    }

    #[test]
    fn test_create_auxiliary_grid_adds_tail_points() {
        let data: Vec<f64> = (0..81).map(|i| 9.6 + i as f64 * 0.01).collect();
        let grid = create_auxiliary_grid(&data, &[10.0], &[0.01], 50.0, 10.0).unwrap();
        let expected_left = 10.0 - 1.5 * 0.01;
        let expected_right = 10.0 + 6.0 * 0.01;
        assert!(
            grid.iter().any(|&e| (e - expected_left).abs() < 1e-10),
            "missing expected left-tail refinement point"
        );
        assert!(
            grid.iter().any(|&e| (e - expected_right).abs() < 1e-10),
            "missing expected right-tail refinement point"
        );
    }
}
