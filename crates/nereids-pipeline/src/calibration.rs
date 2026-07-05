//! Energy calibration for TOF neutron instruments.
//!
//! Finds the flight path length (L) and TOF delay (t₀) that best align
//! a measured transmission spectrum with the ENDF resonance model.
//!
//! The energy-TOF relationship is:
//!
//!   E = C · (L / (t − t₀))²
//!
//! where C = mₙ / 2 ≈ 5.2276e-9 [eV·s²/m²].
//!
//! When L or t₀ differ from the values assumed during data reduction,
//! resonance positions shift in the energy domain, causing catastrophic
//! chi² degradation (e.g. 436 → 2.7 for a 0.3% L correction on VENUS).

#[cfg(test)]
use nereids_core::constants::{EV_TO_JOULES, NEUTRON_MASS_KG};
use nereids_core::types::IsotopeGroup;
use nereids_endf::resonance::ResonanceData;
use nereids_fitting::lm::LmConfig;
use nereids_fitting::resolution_calib::corrected_energy_grid;
use nereids_physics::resolution::TOF_FACTOR;
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

use crate::error::PipelineError;
use crate::pipeline::{
    InputData, SolverConfig, SpectrumFitResult, UnifiedFitConfig, fit_spectrum_typed,
};

/// Neutron mass constant: C = m_n / (2 · eV) ≈ 5.2276e-9 eV·s²/m².
///
/// E [eV] = C · (L [m] / t [s])²
///
/// Uses the CODATA 2018 values from `nereids_core::constants` so that
/// this calibration path, `EnergyScaleTransmissionModel`, and
/// `core::tof_to_energy` all agree to machine precision.  Production
/// code now routes the TOF↔energy transform through
/// `resolution_calib::corrected_energy_grid` (issue #634); this const
/// remains for the test fixtures that synthesize ground-truth grids.
#[cfg(test)]
const NEUTRON_MASS_CONSTANT: f64 = 0.5 * NEUTRON_MASS_KG / EV_TO_JOULES;

/// Lower / upper bounds (log10) on the `n_total` (areal density,
/// atoms/barn) search interval for `calibrate_energy`.  The search
/// runs in `log10(n)` so the band is sampled with relative — rather
/// than absolute — resolution.
///
/// The internal search band is `[~5e-6, ~2e-2]` atoms/barn (a third
/// of a decade beyond each documented edge on either side).  The
/// boundary-saturation guard (`CALIBRATION_LOG10_BOUNDARY_TOL`,
/// ≈ 5 % in linear density) trims a sliver off each end, leaving
/// the *documented* user-supported interval at exactly `[1e-5,
/// 1e-2]`: the doc-stated edges are inside the tolerance window,
/// not on it.
///
/// `[1e-5, 1e-2]` covers every realistic VENUS / paper-relevant
/// density: thin diluted samples down to ~1e-5 atoms/barn (trace
/// detectability ~ Hf in matrix), the Hf calibration foil at
/// ~1e-4, and 1 mm metal foils (U, W, Ni) up to ~1e-2 atoms/barn.
/// Sample densities at the exact documented edges (`1e-5` or
/// `1e-2`) are accepted because the search band extends ~0.3
/// decades beyond them — without the buffer, a true optimum at
/// the documented edge would trip the boundary guard with a
/// "lies outside the band" diagnostic that contradicted the
/// docstring.
const CALIBRATION_LOG10_N_LO: f64 = -5.301; // log10(5e-6)
const CALIBRATION_LOG10_N_HI: f64 = -1.699; // log10(2e-2)

/// Documented lower / upper edges of the user-supported density
/// interval in `log10(n)` space (`1e-5` and `1e-2` atoms/barn).
/// Used only by the error message so the diagnostic states the
/// edges the user expects to see, not the internal buffered band.
const CALIBRATION_LOG10_N_LO_DOC: f64 = -5.0;
const CALIBRATION_LOG10_N_HI_DOC: f64 = -2.0;

/// Tolerance (in `log10(n)` space) at which the golden-section
/// iteration terminates.  `5e-5` ≈ 0.01 % relative resolution on
/// `n_total`, well below the chi² landscape's per-decade curvature
/// floor for typical SAMMY-style resonance fits.
const CALIBRATION_LOG10_N_TOL: f64 = 5e-5;

/// Tolerance (in `log10(n)` space) for the boundary-saturation
/// guard.  An optimum within `0.02` of either bound — about 5 %
/// in linear density — almost always means the true minimum lies
/// outside the supported band and the user should be told rather
/// than silently handed a railed answer.  The internal search band
/// is widened so the *documented* edges (`1e-5`, `1e-2`) remain
/// strictly inside the tolerance window even after this margin is
/// applied.
const CALIBRATION_LOG10_BOUNDARY_TOL: f64 = 0.02;

/// Golden-section search for the `n_total` that minimises
/// `chi2_of_log_n(log10(n))` on `[CALIBRATION_LOG10_N_LO,
/// CALIBRATION_LOG10_N_HI]`.
///
/// Runs in `log10(n)` so the three-decade interval gets relative
/// resolution.  Returns `(best_n, best_chi2)` — `best_n` is the
/// linear-space optimum, not the log-space value.  Uses the
/// standard two-point golden-section update: maintain `(a, b)`,
/// probe at the two golden-ratio interior points `c, d`, and shrink
/// to whichever half-interval brackets the lower value.  The non-
/// finite case (every chi² along the search returns `+inf` — e.g.
/// `SampleParams::new` rejects the entire density range at this
/// (L, t₀)) returns `(best_n, +inf)` so the outer grid search can
/// move on without latching this candidate.
fn golden_section_n_total<F>(log_lo: f64, log_hi: f64, tol: f64, mut chi2_of_log_n: F) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    // Golden ratio reciprocal: (√5 − 1) / 2 ≈ 0.6180.
    let phi: f64 = (5.0_f64.sqrt() - 1.0) / 2.0;

    let mut a = log_lo;
    let mut b = log_hi;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = chi2_of_log_n(c);
    let mut fd = chi2_of_log_n(d);

    // Cap iterations defensively in case `tol` is hit by NaN
    // arithmetic; for the canonical (log_lo = -5, log_hi = -2,
    // tol = 5e-5) parameters, convergence is reached in ~25 steps.
    for _ in 0..200 {
        if (b - a) <= tol {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = chi2_of_log_n(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = chi2_of_log_n(d);
        }
    }

    // The bracket has shrunk to within `tol`; either endpoint of
    // the inner pair is within tolerance of the optimum.  Pick the
    // lower-chi² of the two final probes.
    if fc <= fd {
        (10f64.powf(c), fc)
    } else {
        (10f64.powf(d), fd)
    }
}

/// Plateau-robust dip-position anchor (issue #634 review P0).
///
/// The dip POSITIONS carry the complete (t₀, L_scale) information: with the
/// resonance energies known, matching each measured dip to its resonance and
/// solving the 2-parameter affine TOF map `u_dip = t₀ + L_scale · u_res` by
/// least squares recovers the energy scale directly — no grid search.  This
/// is the same idea as the fitters' peak-match seed, with one critical
/// difference: dip positions are the depth-weighted centres of contiguous
/// below-threshold RUNS, so saturated flat-bottomed dips (transmission ≈ 0
/// over several bins — the SoftwareX foil regime), which fail the strict
/// local-minimum test of `detect_transmission_dips`, are located robustly.
///
/// Used as one CANDIDATE for the stage-2 anchor (scored by the same exact
/// golden-section density as the lattice candidates); the coarse lattice
/// remains the fallback when fewer than two dips are detectable.  Restricted
/// to the plausible window (|t₀| ≤ 10 µs, L_scale ∈ [0.98, 1.02]) — wider
/// offsets are out of this function's documented band.
fn dip_match_anchor(
    energies_nominal: &[f64],
    transmission: &[f64],
    valid: &[bool],
    isotopes: &[ResonanceData],
    abundances: &[f64],
    assumed_flight_path_m: f64,
) -> Option<(f64, f64)> {
    let n = energies_nominal.len();
    // Baseline = 90th percentile of valid transmission; depth threshold at
    // 25 % of the maximum depth (matches the fitters' seed conventions).
    let mut vals: Vec<f64> = transmission
        .iter()
        .zip(valid.iter())
        .filter(|&(_, &v)| v)
        .map(|(&t, _)| t)
        .collect();
    if vals.len() < 5 {
        return None;
    }
    vals.sort_by(f64::total_cmp);
    let baseline = vals[(vals.len() * 9) / 10];
    let max_depth = baseline - vals[0];
    if !(max_depth.is_finite() && max_depth > 1e-6) {
        return None;
    }
    let threshold = baseline - 0.25 * max_depth;

    // Depth-weighted centre of each contiguous below-threshold run.
    let mut dips: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < n {
        if valid[i] && transmission[i] < threshold {
            let mut wsum = 0.0_f64;
            let mut ewsum = 0.0_f64;
            while i < n && valid[i] && transmission[i] < threshold {
                let w = (baseline - transmission[i]).max(0.0);
                wsum += w;
                ewsum += w * energies_nominal[i];
                i += 1;
            }
            if wsum > 0.0 {
                dips.push(ewsum / wsum);
            }
        } else {
            i += 1;
        }
    }
    if dips.len() < 2 {
        return None;
    }

    // Resonance energies of contributing isotopes inside the window.
    let e_lo = energies_nominal[0];
    let e_hi = energies_nominal[n - 1];
    let mut res_e: Vec<f64> = Vec::new();
    for (rd, &abd) in isotopes.iter().zip(abundances.iter()) {
        if abd <= 0.0 {
            continue;
        }
        for range in &rd.ranges {
            for lg in &range.l_groups {
                for r in &lg.resonances {
                    if r.energy > e_lo && r.energy < e_hi {
                        res_e.push(r.energy);
                    }
                }
            }
        }
    }
    res_e.sort_by(f64::total_cmp);
    res_e.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    if res_e.len() < 2 {
        return None;
    }

    // Match each resonance to its nearest dip within half the minimum
    // resonance spacing (floored at twice the grid resolution).
    let min_spacing = res_e
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let grid_res = (e_hi - e_lo) / (n as f64 - 1.0);
    let tol = (0.5 * min_spacing).max(2.0 * grid_res);
    let kl = TOF_FACTOR * assumed_flight_path_m;
    let mut pairs: Vec<(f64, f64)> = Vec::new(); // (u_dip, u_res) in µs
    for &er in &res_e {
        let mut best: Option<f64> = None;
        for &d in &dips {
            let dist = (d - er).abs();
            if dist < tol && best.is_none_or(|b: f64| dist < (b - er).abs()) {
                best = Some(d);
            }
        }
        if let Some(d) = best {
            pairs.push((kl / d.sqrt(), kl / er.sqrt()));
        }
    }
    if pairs.len() < 2 {
        return None;
    }

    // Least squares for u_dip = t0 + ls · u_res.
    let m = pairs.len() as f64;
    let su: f64 = pairs.iter().map(|p| p.0).sum();
    let sv: f64 = pairs.iter().map(|p| p.1).sum();
    let svv: f64 = pairs.iter().map(|p| p.1 * p.1).sum();
    let suv: f64 = pairs.iter().map(|p| p.0 * p.1).sum();
    let sxx = svv - sv * sv / m;
    if !(sxx.is_finite() && sxx > 0.0) {
        return None;
    }
    let ls = (suv - su * sv / m) / sxx;
    let t0 = (su - ls * sv) / m;
    if !(t0.is_finite() && ls.is_finite()) || t0.abs() > 10.0 || !(0.98..=1.02).contains(&ls) {
        return None;
    }
    Some((t0, ls))
}

/// Result of energy calibration.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Fitted flight path length in metres.
    pub flight_path_m: f64,
    /// Fitted TOF delay in microseconds.
    pub t0_us: f64,
    /// Fitted total areal density in atoms/barn.
    pub total_density: f64,
    /// Reduced chi-squared at the best (L, t₀, n) values.
    pub reduced_chi_squared: f64,
    /// Corrected energy grid (ascending, eV).
    pub energies_corrected: Vec<f64>,
}

/// Calibrate the energy axis of a TOF neutron measurement.
///
/// Given a measured 1D transmission spectrum and known sample composition
/// (e.g. natural Hf), finds the (L, t₀) that minimize chi² by aligning
/// the ENDF resonance positions with the measured dips.
///
/// # Search strategy
///
/// Issue #634: the former three-phase `(L, t₀)` grid scan — ~900 (L, t₀)
/// candidates × a ~25-evaluation golden-section density search each,
/// i.e. ~35 000 forward evaluations, >10 min on production windows — is
/// replaced by a staged global-then-local search built on the
/// **`fit_energy_scale` Levenberg–Marquardt path**:
///
/// 1. a coarse joint `(t₀, L_scale)` scan (7 × 7 candidates, each with an
///    exact golden-section density in `log10(n)` over the documented
///    `[1e-5, 1e-2]` atoms/barn band — the per-candidate density is what
///    keeps the anchor JOINT, like the old grid);
/// 2. a direct **dip-match anchor**: measured dip positions (plateau-robust
///    depth-weighted run centres, so saturated flat-bottom dips locate
///    correctly) matched to the known resonance energies and solved as an
///    affine TOF map by least squares — the discriminating anchor along
///    the `(t₀, L_scale)` degeneracy valley;
/// 3. a fine joint pit-scan (11 × 11 at 0.25 µs / 0.05 % steps, narrow-band
///    golden density) around EACH anchor — the compact descendant of the
///    old Phase-2/3 grids, needed because the chi² landscape's sub-bin
///    aliasing pits are ~±0.3 µs wide;
/// 4. a multi-start descent from both fine anchors (per anchor: its own
///    density plus log-spaced starts; per start: a direct joint LM AND an
///    exact-density ↔ alignment-only-LM alternation), with every candidate
///    scored by the original valid-bins chi² and the argmin returned.
///
/// Net cost is ~4× fewer forward evaluations than the old grid on
/// production windows, and the LM refinement removes the old grid's
/// resolution floor (0.001 % L, 0.05 µs t₀) — the optimum is continuous.
/// The internal LM fits disable the fitters' peak-match seed
/// (`with_energy_scale_seed(false)`): stages 1–3 already provide a
/// stronger anchor, and the seed's strict-local-minimum dip detector
/// mislocates saturated flat-bottom dips.
///
/// Each LM fit constrains `L_scale` to ±1 % (`ENERGY_SCALE_L_SCALE_*`);
/// when a fit rails on that box (a larger true offset), the search
/// re-anchors on the corrected grid and composes the affine TOF maps, so
/// the documented ±1.5 % flight-path band remains covered.
///
/// The golden-section seed runs on a slightly wider band (~5e-6 to
/// ~2e-2), and if the **fitted** density optimum saturates that band —
/// effectively at `1e-5` or `1e-2`, or anywhere beyond — the function
/// returns `Err(PipelineError::InvalidParameter)` rather than a silent
/// boundary-saturated answer, because a true minimum at or past the edge
/// almost always means the real optimum lies outside the supported
/// interval and the caller should supply a better initial estimate
/// or check the sample composition.
///
/// # Arguments
///
/// * `energies_nominal` — Energy grid computed with assumed L (ascending, eV)
/// * `transmission` — Measured transmission values (same length)
/// * `uncertainty` — Per-bin uncertainty (same length)
/// * `isotopes` — ENDF resonance data for each isotope
/// * `abundances` — Natural abundance fractions (same length as isotopes, sum ≤ 1)
/// * `assumed_flight_path_m` — The L used to compute `energies_nominal`
/// * `temperature_k` — Sample temperature for Doppler broadening
/// * `resolution` — Optional instrument resolution function.  When provided,
///   the forward model includes Doppler + resolution broadening, producing
///   more accurate (L, t₀) fits.  Without resolution, fitted parameters
///   absorb the missing broadening and may be biased.
///
/// # Returns
///
/// [`CalibrationResult`] with the fitted (L, t₀, n_total) and corrected energies.
#[allow(clippy::too_many_arguments)]
pub fn calibrate_energy(
    energies_nominal: &[f64],
    transmission: &[f64],
    uncertainty: &[f64],
    isotopes: &[ResonanceData],
    abundances: &[f64],
    assumed_flight_path_m: f64,
    temperature_k: f64,
    resolution: Option<&InstrumentParams>,
) -> Result<CalibrationResult, PipelineError> {
    let n = energies_nominal.len();
    if n == 0 {
        return Err(PipelineError::InvalidParameter(
            "energies_nominal must not be empty".into(),
        ));
    }
    if transmission.len() != n || uncertainty.len() != n {
        return Err(PipelineError::InvalidParameter(format!(
            "transmission ({}) and uncertainty ({}) must match energies ({})",
            transmission.len(),
            uncertainty.len(),
            n,
        )));
    }
    if isotopes.len() != abundances.len() {
        return Err(PipelineError::InvalidParameter(format!(
            "isotopes ({}) must match abundances ({})",
            isotopes.len(),
            abundances.len(),
        )));
    }

    // Validate abundance values up-front.  Without this guard, non-finite
    // or negative entries are silently multiplied into per-isotope
    // densities (`abd * n_total`), `SampleParams::new` rejects the
    // non-positive thickness, `compute_chi2` returns `INFINITY` for
    // every grid point, and the user sees "no finite chi²" or boundary
    // saturation rather than the actual cause (a bad abundance entry).
    // Equivalent guards already exist for `assumed_flight_path_m` and
    // `energies_nominal`; this closes the same gap for abundances.
    let mut total_abundance = 0.0;
    for (i, &abn) in abundances.iter().enumerate() {
        if !abn.is_finite() || abn < 0.0 {
            return Err(PipelineError::InvalidParameter(format!(
                "calibrate_energy: abundances[{i}] = {abn} is not finite and non-negative"
            )));
        }
        total_abundance += abn;
    }
    if total_abundance <= 0.0 {
        return Err(PipelineError::InvalidParameter(
            "calibrate_energy: sum of abundances must be strictly positive".into(),
        ));
    }

    // Validate scalar / array inputs up-front so the grid-search loop
    // cannot silently produce a "perfect calibration" result from
    // degenerate inputs.  Without these guards, all-NaN transmission
    // combined with the dof=1 fallback below would cause
    // chi²_reduced = 0.0 to be reported as a successful fit.
    if !assumed_flight_path_m.is_finite() || assumed_flight_path_m <= 0.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "assumed_flight_path_m must be finite and positive, got {assumed_flight_path_m}",
        )));
    }
    // Reject an invalid temperature at the entry (issue #634 review): the
    // sibling `UnifiedFitConfig::new` validates it, and without this guard a
    // NaN/negative temperature is only caught deep inside the repeated
    // search/LM stages (or silently converted to an all-INFINITY chi²).
    if !temperature_k.is_finite() || temperature_k < 0.0 {
        return Err(PipelineError::InvalidParameter(format!(
            "calibrate_energy: temperature_k must be finite and non-negative, \
             got {temperature_k}",
        )));
    }
    for (i, &e) in energies_nominal.iter().enumerate() {
        if !e.is_finite() || e <= 0.0 {
            return Err(PipelineError::InvalidParameter(format!(
                "energies_nominal[{i}] must be finite and positive, got {e}",
            )));
        }
        if i > 0 && e <= energies_nominal[i - 1] {
            return Err(PipelineError::InvalidParameter(format!(
                "energies_nominal must be strictly ascending; \
                 energies_nominal[{i}]={e} <= energies_nominal[{}]={}",
                i - 1,
                energies_nominal[i - 1],
            )));
        }
    }

    // Pre-filter valid bins (finite T, positive sigma)
    let valid: Vec<bool> = transmission
        .iter()
        .zip(uncertainty.iter())
        .map(|(&t, &s)| t.is_finite() && s.is_finite() && s > 0.0)
        .collect();

    // Require enough valid bins to constrain the three fitted
    // parameters (L, t₀, n_total).  Previously, when every bin was
    // invalid, `compute_chi2` returned 0.0 for every grid point, the
    // first candidate latched as "best", and the dof=1 fallback turned
    // that into a reported `chi²_reduced = 0.0` — i.e. a totally
    // degenerate input was indistinguishable from a perfect calibration.
    const N_FITTED_PARAMS: usize = 3;
    let n_valid = valid.iter().filter(|&&v| v).count();
    if n_valid < N_FITTED_PARAMS {
        return Err(PipelineError::InvalidParameter(format!(
            "calibrate_energy requires at least {N_FITTED_PARAMS} bins with finite \
             transmission and positive uncertainty, got {n_valid} valid out of {n}",
        )));
    }

    // ── Neutralise invalid bins for the LM ──────────────────────────────
    // The grid search masked invalid bins out of its chi² sum; the LM cost
    // has no per-bin validity mask (its active-mask is an energy-window
    // mask only), and a NaN transmission at an active bin poisons the
    // normal equations.  Replace invalid bins with a finite dummy and a
    // huge σ (weight ~1e-60 — numerically zero) so the energy grid stays
    // intact for resolution broadening while the bins carry no influence
    // on the fit.  The reported chi²_r below is still computed over VALID
    // bins only, preserving the original `dof = n_valid − 3` contract.
    let mut t_fit = transmission.to_vec();
    let mut sigma_fit = uncertainty.to_vec();
    for (i, &ok) in valid.iter().enumerate() {
        if !ok {
            t_fit[i] = 1.0;
            sigma_fit[i] = 1.0e30;
        }
    }

    // ── One fitted density parameter via an isotope group ───────────────
    // The grid fitted a single n_total with per-isotope densities
    // `abundance_i · n_total`.  Reproduce that exactly with one group whose
    // ratios are the normalised abundances: the fitted per-isotope density
    // is `D · abundance_i / S`, so `n_total = D / S` (S = Σ abundances).
    // Zero-abundance isotopes contribute nothing to the model (exactly as
    // in `compute_chi2`) and `IsotopeGroup` rejects non-positive ratios,
    // so they are dropped from the group.
    let mut members = Vec::new();
    let mut rd_list: Vec<ResonanceData> = Vec::new();
    for (rd, &abd) in isotopes.iter().zip(abundances.iter()) {
        if abd > 0.0 {
            members.push((rd.isotope, abd / total_abundance));
            rd_list.push(rd.clone());
        }
    }
    let group = IsotopeGroup::custom("calibration".into(), members)
        .map_err(|e| PipelineError::InvalidParameter(format!("calibrate_energy: {e}")))?;
    let group_pairs: [(&IsotopeGroup, &[ResonanceData]); 1] = [(&group, rd_list.as_slice())];

    // ── Local helpers ────────────────────────────────────────────────────
    // `run_lm`: one LM energy-scale fit on `grid`, cold-seeded at
    // `(t0, L_scale) = (0, 1)` — the fitters' internal peak-match seed is
    // deliberately DISABLED for these fits (see the
    // `with_energy_scale_seed(false)` call and its rationale below) — with
    // the grouped density either free or frozen (#633) at `d_init`.
    let run_lm = |grid: &[f64],
                  d_init: f64,
                  freeze_density: bool|
     -> Result<SpectrumFitResult, PipelineError> {
        let mut config = UnifiedFitConfig::new(
            grid.to_vec(),
            vec![rd_list[0].clone()],
            vec!["calibration".into()],
            temperature_k,
            resolution.map(|r| r.resolution.clone()),
            vec![d_init],
        )
        .map_err(|e| PipelineError::InvalidParameter(format!("calibrate_energy: {e}")))?
        .with_groups(&group_pairs, vec![d_init])
        .map_err(|e| PipelineError::InvalidParameter(format!("calibrate_energy: {e}")))?
        .with_solver(SolverConfig::LevenbergMarquardt(LmConfig {
            // Small first steps (issue #634): at saturated sharp dips the
            // true (t0, L_scale) pit is sub-0.3 µs narrow while the
            // along-valley Jacobian is near-singular — the default
            // lambda_init = 1e-3 lets the first LM step overshoot the
            // basin (observed: a start 0.12 µs from truth walking to a
            // 7 µs-wrong valley pit).  A large initial damping keeps the
            // early steps gradient-like and short; the LM anneals lambda
            // back down once inside the basin.
            lambda_init: 1.0e2,
            ..LmConfig::default()
        }))
        .with_energy_scale(0.0, 1.0, assumed_flight_path_m)
        // The anchor stages already seed (t0, L_scale); the fitters' internal
        // peak-match seed mislocates saturated flat-bottom dips and would
        // overwrite the anchor with an in-bounds wrong seed (issue #634).
        .with_energy_scale_seed(false);
        if freeze_density {
            config = config.with_fix_densities(true);
        }
        let input = InputData::Transmission {
            transmission: t_fit.clone(),
            uncertainty: sigma_fit.clone(),
        };
        fit_spectrum_typed(&input, &config).map_err(|e| {
            PipelineError::InvalidParameter(format!(
                "calibrate_energy: LM energy-scale fit failed: {e}"
            ))
        })
    };
    // `golden_at`: the ORIGINAL exact 1-D density optimisation (golden
    // section in log10 n against `compute_chi2`) at a given corrected grid.
    let golden_at = |e_corr: &[f64]| {
        golden_section_n_total(
            CALIBRATION_LOG10_N_LO,
            CALIBRATION_LOG10_N_HI,
            CALIBRATION_LOG10_N_TOL,
            |log_n| {
                compute_chi2(
                    e_corr,
                    transmission,
                    uncertainty,
                    isotopes,
                    abundances,
                    10f64.powf(log_n),
                    temperature_k,
                    &valid,
                    resolution,
                )
            },
        )
    };

    // ── Stage 1: density seed at the identity energy scale ──────────────
    // If no candidate yields a finite chi² (wildly out-of-scale data), fall
    // back to the band midpoint — the no-finite-chi² guard below reports
    // the failure.
    let (n_seed, seed_chi2) = golden_at(energies_nominal);
    let n_seed = if seed_chi2.is_finite() {
        n_seed
    } else {
        10f64.powf(0.5 * (CALIBRATION_LOG10_N_LO + CALIBRATION_LOG10_N_HI))
    };

    // ── Stage 2: coarse global JOINT alignment anchor ────────────────────
    // A compact descendant of the old Phase-1 scan: 7 L_scale (±1.5 % in
    // 0.5 % steps) × 7 t₀ (−5…+10 µs in 2.5 µs steps) candidates, each
    // scored with its OWN exact golden-section density — the per-candidate
    // density optimisation is what makes the anchor JOINT, exactly like the
    // old grid.  Scoring all candidates at one common density is NOT
    // sufficient (issue #634 review, empirically demonstrated): on
    // saturated (5e-3) or trace (2e-5) landscapes with production-scale
    // offsets (0.3–1.2 % L, the module header's own VENUS example
    // magnitude), the identity-seeded common density corrupts the candidate
    // ranking, every descent start then inherits a wrong-basin anchor, and
    // the calibration returns Ok with a density up to 139× off — at trace
    // density with chi²_r ≈ 1e-4, i.e. no user-visible failure signal.
    // ~49 golden sections ≈ 1 200 forward evaluations, still ~30× cheaper
    // than the old three-phase grid's ~35 000.
    let mut t0_tot = 0.0_f64;
    let mut ls_tot = 1.0_f64;
    let mut anchor_n = n_seed;
    // NaN-safe latch init (issue #634 review): a NaN identity chi² would
    // make every `chi2_c < anchor_chi2` comparison false and silently
    // discard the whole joint scan — the same NaN-latch pattern the winner
    // latch below is hardened against.  The old grid was immune
    // (`best_chi2 = INFINITY` with no privileged identity candidate).
    let mut anchor_chi2 = if seed_chi2.is_finite() {
        seed_chi2
    } else {
        f64::INFINITY
    };
    for i_l in -3..=3_i32 {
        let ls = 1.0 + f64::from(i_l) * 0.005;
        for i_t in 0..=6_i32 {
            let t0 = -5.0 + 2.5 * f64::from(i_t);
            if i_l == 0 && i_t == 2 {
                continue; // the identity candidate is (n_seed, seed_chi2) above
            }
            let Ok(e_c) = corrected_energy_grid(energies_nominal, t0, ls, assumed_flight_path_m)
            else {
                continue; // degenerate candidate (t0 past the shortest TOF)
            };
            let (n_c, chi2_c) = golden_at(&e_c);
            if chi2_c < anchor_chi2 {
                anchor_chi2 = chi2_c;
                t0_tot = t0;
                ls_tot = ls;
                anchor_n = n_c;
            }
        }
    }

    // ── Stage 2b: fine joint pit-scan around each anchor ────────────────
    // The coarse 2.5 µs / 0.5 % cell — and the dip-match solve at saturated
    // dips (its centroid is the bin-quantized saturation interval, biased up
    // to ~0.5 µs) — are both coarser than the (t0, L_scale) landscape's
    // sub-bin aliasing pits (~±0.3 µs wide, ~2 µs apart on the 0.2 eV test
    // grid), so a descent seeded from either can converge into a
    // NEIGHBOURING pit (observed: chi² 3.3 at 1.85 µs from the chi² ≈ 0 true
    // pit, with the LM reporting converged).  This fine scan — 11 × 11
    // candidates at 0.25 µs / 0.05 % steps spanning one coarse cell, each
    // with its own golden-section density on a ±0.5-decade band around the
    // coarse anchor density — is the compact descendant of the old
    // Phase-2/3 refinement grids and deterministically lands inside the
    // true pit, from which the descent below converges.  It runs around
    // BOTH anchors (lattice winner and dip-match).
    let fine_scan = |t0_a: f64, ls_a: f64, n_a: f64| -> (f64, f64, f64, f64) {
        let fine_lo = (n_a.log10() - 0.5).max(CALIBRATION_LOG10_N_LO);
        let fine_hi = (n_a.log10() + 0.5).min(CALIBRATION_LOG10_N_HI);
        let mut best = (t0_a, ls_a, n_a, f64::INFINITY);
        for i_l in -5..=5_i32 {
            let ls = ls_a + f64::from(i_l) * 0.0005;
            for i_t in -5..=5_i32 {
                let t0 = t0_a + f64::from(i_t) * 0.25;
                let Ok(e_c) =
                    corrected_energy_grid(energies_nominal, t0, ls, assumed_flight_path_m)
                else {
                    continue;
                };
                let (n_c, chi2_c) =
                    golden_section_n_total(fine_lo, fine_hi, CALIBRATION_LOG10_N_TOL, |log_n| {
                        compute_chi2(
                            &e_c,
                            transmission,
                            uncertainty,
                            isotopes,
                            abundances,
                            10f64.powf(log_n),
                            temperature_k,
                            &valid,
                            resolution,
                        )
                    });
                if chi2_c < best.3 {
                    best = (t0, ls, n_c, chi2_c);
                }
            }
        }
        best
    };

    // Anchor set for the descent: the fine-scanned lattice winner PLUS the
    // fine-scanned dip-match solve (see `dip_match_anchor`).  The dip-match
    // anchor is deliberately NOT ranked against the lattice winner by a
    // single chi² comparison: at sub-bin misalignment of sharp saturated
    // dips, chi²(n) is multimodal — the steep dip edges make a nearly-right
    // deep model score WORSE than a shallow one — so a chi² ranking can
    // discard the one anchor whose fine neighbourhood contains the true pit
    // (observed: dip-match landed 0.12 µs from truth yet scored chi² 1 809
    // vs the lattice winner's 124).  Running the full multi-start descent
    // from BOTH fine anchors and letting the final argmin-chi² decide is
    // robust to that ranking failure.
    let lattice_fine = fine_scan(t0_tot, ls_tot, anchor_n);
    let mut anchor_list: Vec<(f64, f64, f64)> =
        vec![(lattice_fine.0, lattice_fine.1, lattice_fine.2)];
    if let Some((t0_d, ls_d)) = dip_match_anchor(
        energies_nominal,
        transmission,
        &valid,
        isotopes,
        abundances,
        assumed_flight_path_m,
    ) {
        // Density context for the dip anchor's narrow golden band: the
        // lattice anchor's density is the best available estimate.
        let dip_fine = fine_scan(t0_d, ls_d, anchor_n);
        // Skip a duplicate anchor (both scans converged on the same pit).
        if (dip_fine.0 - lattice_fine.0).abs() > 0.05 || (dip_fine.1 - lattice_fine.1).abs() > 1e-4
        {
            anchor_list.push((dip_fine.0, dip_fine.1, dip_fine.2));
        }
    }

    // (t0_tot, ls_tot, n_total, chi2 over valid bins, LM converged flag)
    let mut winner: Option<(f64, f64, f64, f64, bool)> = None;
    // Last LM error across all starts: per-seed failures are skippable, but
    // when EVERY start fails (a config-class error — bad resolution kernel,
    // config rejection — fails all seeds identically) the no-winner
    // diagnostic must carry the actual root cause instead of misattributing
    // it to non-finite residuals (issue #634 review).
    let mut last_lm_err: Option<PipelineError> = None;
    for &(a_t0, a_ls, a_n) in &anchor_list {
        let anchor_grid = if a_t0 == 0.0 && a_ls == 1.0 {
            energies_nominal.to_vec()
        } else {
            match corrected_energy_grid(energies_nominal, a_t0, a_ls, assumed_flight_path_m) {
                Ok(g) => g,
                // Degenerate anchor — the other anchor still runs; if every
                // anchor is degenerate the no-winner guard below reports it.
                Err(_) => continue,
            }
        };
        // Density seeds: this anchor's own fine-scan density (the joint
        // optimum of its pit) plus log-spaced starts across the band (the
        // aliasing/multimodality guard).
        let mut seeds = vec![a_n];
        for exp in [-4.5_f64, -3.5, -2.5] {
            let sd = 10f64.powf(exp);
            // Skip seeds within 2× of an existing one — same basin.
            if seeds.iter().all(|&e| (sd / e).log10().abs() > 0.301) {
                seeds.push(sd);
            }
        }
        for &start_n in &seeds {
            // Two descent variants per (anchor, seed); both feed the same
            // final argmin:
            //
            //  (a) DIRECT joint LM from the anchor — robust at saturated
            //      dips, where the alternation below fails: a frozen
            //      slightly-wrong density has its alignment optimum in a
            //      COMPENSATING pit, so the alternation walks away from a
            //      good anchor before its joint stage ever runs (observed:
            //      an anchor 0.12 µs from truth descending to a chi² 3.3
            //      pit while the direct joint fit rolls into the chi² ≈ 0
            //      true pit).
            //
            //  (b) alternation (exact density ↔ alignment-only LM, then
            //      joint LM) — robust when the density seed is decades off,
            //      where a direct joint fit would trade density against a
            //      sub-bin shift (the aliasing degeneracy).
            let mut candidates: Vec<(f64, f64, SpectrumFitResult)> = Vec::new();

            // (a) direct joint fit from the anchor.
            match run_lm(&anchor_grid, start_n * total_abundance, false) {
                Ok(fit) => {
                    let t0_c = a_t0 + a_ls * fit.t0_us.unwrap_or(0.0);
                    let ls_c = a_ls * fit.l_scale.unwrap_or(1.0);
                    candidates.push((t0_c, ls_c, fit));
                }
                Err(e) => last_lm_err = Some(e),
            }

            // (b) alternation.
            let mut t0_tot = a_t0;
            let mut ls_tot = a_ls;
            let mut grid = anchor_grid.clone();
            let mut n_cur = start_n;
            let mut failed = false;
            for cycle in 0..3 {
                // First cycle keeps the start density (the whole point of the
                // multi-start); later cycles re-optimise it at the improved
                // alignment.
                if cycle > 0 {
                    let (n_new, chi2_n) = golden_at(&grid);
                    if chi2_n.is_finite() {
                        n_cur = n_new;
                    }
                }
                let align = match run_lm(&grid, n_cur * total_abundance, true) {
                    Ok(a) => a,
                    Err(e) => {
                        last_lm_err = Some(e);
                        failed = true;
                        break;
                    }
                };
                let t0_k = align.t0_us.unwrap_or(0.0);
                let ls_k = align.l_scale.unwrap_or(1.0);
                t0_tot += ls_tot * t0_k;
                ls_tot *= ls_k;
                // Alignment converged (no further shift found) → stop.
                if t0_k.abs() < 1e-6 && (ls_k - 1.0).abs() < 1e-9 {
                    break;
                }
                match corrected_energy_grid(energies_nominal, t0_tot, ls_tot, assumed_flight_path_m)
                {
                    Ok(g) => grid = g,
                    Err(_) => {
                        failed = true;
                        break;
                    }
                }
            }
            if !failed {
                // Joint refinement of the alternation result.
                let (n_stage, chi2_stage) = golden_at(&grid);
                if chi2_stage.is_finite() {
                    n_cur = n_stage;
                }
                match run_lm(&grid, n_cur * total_abundance, false) {
                    Ok(fit) => {
                        let t0_c = t0_tot + ls_tot * fit.t0_us.unwrap_or(0.0);
                        let ls_c = ls_tot * fit.l_scale.unwrap_or(1.0);
                        candidates.push((t0_c, ls_c, fit));
                    }
                    Err(e) => last_lm_err = Some(e),
                }
            }

            // Score every candidate with the ORIGINAL valid-bins objective at
            // its solution; keep the argmin.  The latch is gated on
            // finiteness: a bare `None => true` first-candidate arm would let
            // a NaN chi² latch (bypassing the `<` comparison), and every
            // later candidate would then compare `chi2 < NaN` (always false)
            // — a valid later calibration could never displace a NaN first
            // one and the function would return the no-finite-chi² error for
            // a calibratable spectrum.  Mirrors the old grid's
            // `best_chi2 = INFINITY` + `<` behaviour, where non-finite
            // candidates could never latch (issue #634 review).
            for (t0_c, ls_c, fit) in candidates {
                let n_c = fit.densities.first().copied().unwrap_or(f64::NAN) / total_abundance;
                let Ok(e_c) =
                    corrected_energy_grid(energies_nominal, t0_c, ls_c, assumed_flight_path_m)
                else {
                    continue;
                };
                let chi2_c = compute_chi2(
                    &e_c,
                    transmission,
                    uncertainty,
                    isotopes,
                    abundances,
                    n_c,
                    temperature_k,
                    &valid,
                    resolution,
                );
                let better = chi2_c.is_finite()
                    && match &winner {
                        None => true,
                        Some((_, _, _, best, _)) => chi2_c < *best,
                    };
                if better {
                    winner = Some((t0_c, ls_c, n_c, chi2_c, fit.converged));
                }
            }
        }
    }
    let Some((t0_tot, ls_tot, best_n, _, lm_converged)) = winner else {
        // Preserve the root cause: a config-class LM error fails every seed
        // identically, and its message is the actionable diagnostic — the
        // generic non-finite-residuals guess applies only when no LM error
        // occurred.
        let detail = match last_lm_err {
            Some(e) => format!("last LM error: {e}"),
            None => "likely cause is forward-model failure or non-finite \
                     residuals (e.g. wildly out-of-scale transmission)"
                .to_string(),
        };
        return Err(PipelineError::InvalidParameter(format!(
            "calibrate_energy: calibration produced no finite chi² from any \
             density start (best_chi2 = inf) — {detail}"
        )));
    };

    // Boundary-saturation guard: if the fitted n_total lies within
    // tolerance of either density-band edge — or anywhere beyond it (the
    // LM density is unbounded above, unlike the old grid) — the true
    // minimum almost certainly sits outside the supported range and the
    // calibration is unreliable.  Returning `Ok` with `best_n` ≈
    // boundary would silently rail the density and let the (L, t₀)
    // parameters absorb the missing density freedom by compensating
    // bias — exactly the silent-failure pattern the no-finite-chi²
    // guard below also defends against, but with a boundary-specific
    // diagnostic.
    //
    // The seed band is `[~5e-6, ~2e-2]`; the documented edges are
    // `[1e-5, 1e-2]`.  Fits at the documented edges lie outside the
    // tolerance window (a true optimum at `1e-5` sits ~0.3 decades
    // above the internal lower bound, comfortably past the ~0.02-log10
    // tolerance) so the guard fires only when the optimum has actually
    // saturated against — or escaped — the wider buffer.  One-sided
    // comparisons (≤ / ≥ rather than the former |·|) so a far
    // out-of-band LM optimum (e.g. n ≈ 1) fires the same diagnostic; a
    // NaN density falls through (NaN comparisons are false) to the
    // no-finite-chi² guard below.
    let log_best_n = best_n.log10();
    let n_lo = 10f64.powf(CALIBRATION_LOG10_N_LO_DOC);
    let n_hi = 10f64.powf(CALIBRATION_LOG10_N_HI_DOC);
    if log_best_n <= CALIBRATION_LOG10_N_LO + CALIBRATION_LOG10_BOUNDARY_TOL
        || log_best_n >= CALIBRATION_LOG10_N_HI - CALIBRATION_LOG10_BOUNDARY_TOL
    {
        return Err(PipelineError::InvalidParameter(format!(
            "calibrate_energy: n_total optimum {best_n:.3e} atoms/barn is at the \
             search boundary [{n_lo:.0e}, {n_hi:.0e}]; the true optimum likely lies \
             outside this band.  Provide a better initial density estimate, check \
             the sample composition / abundances, or extend the search range."
        )));
    }

    // Corrected energy grid at the composed (t0_tot, ls_tot): the same
    // canonical transform the LM evaluated (SAMMY −t0 convention,
    // `resolution_calib::corrected_energy_grid`), expressed relative to
    // the caller's ORIGINAL nominal grid and assumed flight path — the
    // pre-#634 output convention.
    let energies_corrected =
        corrected_energy_grid(energies_nominal, t0_tot, ls_tot, assumed_flight_path_m).map_err(
            |e| {
                PipelineError::InvalidParameter(format!(
                    "calibrate_energy: degenerate calibration: {e}"
                ))
            },
        )?;

    // Final chi² over VALID bins with the original grid objective
    // (`compute_chi2`), preserving the pre-#634 chi²_r semantics exactly:
    // invalid bins excluded, `dof = n_valid − 3` clamped to ≥ 1 for the
    // exact-fit edge case.  If the final chi² is non-finite
    // (forward-model failure, or residual overflow for
    // non-finite-but-passing transmission values such as 1e308), reject
    // explicitly so calibration failure is always an `Err`, never an
    // `Ok` with a sentinel chi² — the same contract as the old
    // post-grid-search guard.
    //
    // Note the gate is the chi², NOT the LM `converged` flag: the grid
    // search this replaced had no convergence concept — it reported the
    // best candidate found, quality-gated only by finite chi² and the
    // density boundary.  The LM matches that contract; its parameters at
    // lambda-breakout are the best point found (a stall is common for
    // trace densities, where the energy-scale Jacobian columns are nearly
    // zero and the step control gives up AFTER the density has already
    // converged).  The reported chi²_r carries the fit quality either way.
    let best_chi2 = compute_chi2(
        &energies_corrected,
        transmission,
        uncertainty,
        isotopes,
        abundances,
        best_n,
        temperature_k,
        &valid,
        resolution,
    );
    if !best_chi2.is_finite() {
        return Err(PipelineError::InvalidParameter(format!(
            "calibrate_energy: calibration produced no finite chi² \
             (LM converged = {converged}, best_chi2 = {best_chi2}) — likely cause is \
             forward-model failure or non-finite residuals (e.g. wildly \
             out-of-scale transmission)",
            converged = lm_converged,
        )));
    }

    // Final chi2r (reduced).  The up-front guard ensures
    // `n_valid >= N_FITTED_PARAMS`, so we always have a non-negative
    // dof.  We still clamp to `max(1)` so that the exact-fit edge case
    // (n_valid == N_FITTED_PARAMS, dof = 0) reports a finite value
    // instead of dividing by zero.
    let dof = n_valid.saturating_sub(N_FITTED_PARAMS).max(1);
    let chi2r = best_chi2 / dof as f64;

    Ok(CalibrationResult {
        flight_path_m: assumed_flight_path_m * ls_tot,
        t0_us: t0_tot,
        total_density: best_n,
        reduced_chi_squared: chi2r,
        energies_corrected,
    })
}

/// Compute total chi² for a given (E_corrected, n_total) against measured data.
#[allow(clippy::too_many_arguments)]
fn compute_chi2(
    energies: &[f64],
    transmission: &[f64],
    uncertainty: &[f64],
    isotopes: &[ResonanceData],
    abundances: &[f64],
    n_total: f64,
    temperature_k: f64,
    valid: &[bool],
    resolution: Option<&InstrumentParams>,
) -> f64 {
    // Build (isotope, density) pairs
    let pairs: Vec<(ResonanceData, f64)> = isotopes
        .iter()
        .zip(abundances.iter())
        .map(|(iso, &abd)| (iso.clone(), abd * n_total))
        .collect();

    let sample = match SampleParams::new(temperature_k, pairs) {
        Ok(s) => s,
        Err(_) => return f64::INFINITY,
    };

    // P-5: Include resolution broadening when available.
    // Without it, fitted L and t₀ absorb the missing broadening bias.
    let model = match transmission::forward_model(energies, &sample, resolution) {
        Ok(m) => m,
        Err(_) => return f64::INFINITY,
    };

    // Chi²
    let mut chi2 = 0.0;
    for (i, (&t_data, &t_model)) in transmission.iter().zip(model.iter()).enumerate() {
        if !valid[i] {
            continue;
        }
        let residual = (t_data - t_model) / uncertainty[i];
        chi2 += residual * residual;
    }
    chi2
}

#[cfg(test)]
mod tests {
    use super::*;
    use nereids_endf::resonance::test_support::synthetic_single_resonance;

    /// Round-trip exercise of the public `calibrate_energy` API on a
    /// synthetic spectrum.  Uses `synthetic_single_resonance` from
    /// `nereids_endf::resonance::test_support` so the test does not require network access and
    /// runs in every CI invocation.
    ///
    /// Note on tolerances: the grid-search calibrator's L resolution
    /// is fundamentally limited by chi² curvature (broader resonances
    /// or sparser bins → broader minimum).  With only synthetic
    /// single-resonance isotopes on a sparse 0.2 eV grid, the chi²
    /// landscape near L=true_l is shallow on the scale of the
    /// 0.001 % ultra-fine step — Doppler-broadened ≈ 33 meV resonance
    /// width vs 200 meV grid spacing means each resonance is sampled
    /// by ≤ 1 bin, so (L, t₀) can drift across a wide band before chi²
    /// degrades enough to lock the minimum down.  We therefore (a) use
    /// a small true offset (0.05 % in L, 0.5 µs in t₀) that the
    /// calibrator can resolve, and (b) test the *physics* — a fit
    /// converged, the corrected energies are close to truth on the
    /// data-relevant range, and density recovery is in the right
    /// decade — rather than chasing exact ENDF-style L recovery.
    /// The bit-exact precision question is owned by the SAMMY parity
    /// tests in `nereids-physics`, not by this API smoke test.
    #[test]
    fn test_calibrate_round_trip_synthetic() {
        // Generate synthetic data with known L and t0, then recover them.
        // Small offsets (0.05 % in L, 0.5 µs in t₀) so the chi² minimum
        // is well inside Phase-2 fine grid (±0.05 % in L, ±2 µs in t₀).
        //
        // Setup (resonances, energy grid, forward-model transmission,
        // and the `calibrate_energy` call itself) is shared with the
        // density-band tests below via `calibrate_round_trip_at_density`;
        // the helper returns `(result, e_true, assumed_l)` so this
        // smoke test can also assert on corrected-energy accuracy.
        let true_n = 1.5e-4;
        let (result, e_true, assumed_l) =
            calibrate_round_trip_at_density(true_n).expect("Calibration failed");

        // Check recovery.  Wider tolerances than the Hf-178 fixture
        // because the synthetic chi² minimum is broader (see the
        // doc comment on the test above).  These bands still
        // distinguish a successful fit from a degenerate one (the
        // zero-valid-bins failure mode would report L = assumed_l
        // and chi² = 0.0).
        //
        // With the n_total golden-section refactor, the previously-
        // narrow density grid no longer pins (L, t₀) at a single
        // coarse-grid point; the calibrator now expresses the
        // genuine (L, t₀, n) degeneracy this sparse-grid synthetic
        // admits — 33 meV Doppler-broadened resonances vs 200 meV
        // grid spacing samples each resonance with ≤ 1 bin, so any
        // (L, t₀) that places the resonance near the same bin gives
        // an indistinguishable fit.  The L and t₀ parameters are
        // therefore not independently identifiable from this
        // synthetic, but the *corrected energy grid* — the actual
        // downstream deliverable — is, and is the right thing to
        // assert on.
        //
        // L and t₀ are still required to be inside the search grid
        // (Phase 1 L ±1.5 %, t₀ ∈ [-5, +10] µs) and density inside
        // the search band — anything outside would signal a
        // calibrator regression, not a degeneracy.
        assert!(
            (result.flight_path_m - assumed_l).abs() / assumed_l <= 0.015,
            "L drift outside Phase 1 ±1.5 % grid: got {}",
            result.flight_path_m,
        );
        assert!(
            (-5.0..=10.0).contains(&result.t0_us),
            "t0 outside Phase 1 grid: got {}",
            result.t0_us,
        );
        assert!(
            (result.total_density - true_n).abs() / true_n < 0.5,
            "n: got {}, expected {}",
            result.total_density,
            true_n,
        );

        // The corrected energy grid is the deliverable a downstream
        // fit uses; even when (L, t₀, n) drift inside the chi²
        // basin allowed by this sparse-grid synthetic, the corrected
        // energies must still track `e_true` to within ~1 % at the
        // resonance positions and a few % across the grid.  A
        // median relative error check is more robust to per-bin
        // scaling than max-over-bins on a 150-bin grid; we still
        // assert max < 5 % to catch a wholly-wrong calibration.
        let rel_errs: Vec<f64> = result
            .energies_corrected
            .iter()
            .zip(e_true.iter())
            .map(|(&ec, &et)| (ec - et).abs() / et)
            .collect();
        let mut sorted = rel_errs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let max_err = sorted.last().copied().unwrap_or(0.0);
        assert!(
            median < 0.02,
            "corrected energies median rel err {median} should be < 2 %",
        );
        assert!(
            max_err < 0.05,
            "corrected energies max rel err {max_err} should be < 5 %",
        );
        // The synthetic round-trip uses noiseless data, so a grid point
        // that lands on (or arbitrarily close to) the true parameters
        // can legitimately yield chi² = 0.  Accept `>= 0.0` (the
        // physically valid range) rather than `> 0.0`, which was a
        // flake-prone strict-inequality.  The zero-valid-bins
        // regression that previously reported chi² = 0.0 is now
        // rejected up-front by the `n_valid >= N_FITTED_PARAMS` guard,
        // and the all-infinity grid-search case is rejected by the
        // post-search `best_chi2.is_finite()` guard — so finiteness
        // alone is the meaningful check here.
        assert!(
            result.reduced_chi_squared.is_finite() && result.reduced_chi_squared >= 0.0,
            "chi²_reduced must be finite and >= 0 (degenerate-input regressions \
             are caught up-front and post-search; this assertion guards against \
             chi² leaking as inf or NaN); got {}",
            result.reduced_chi_squared,
        );
    }

    // ── Degenerate-input guards ────────────────────────────────────────
    //
    // Before these guards, `compute_chi2` returned 0.0 when every bin
    // was skipped by the `valid` mask, the grid search latched the
    // first candidate as "best", and the dof=1 fallback at the end
    // turned that into a reported `chi²_reduced = 0.0` — a totally
    // degenerate input was indistinguishable from a perfect
    // calibration.

    /// `(energies_nominal, transmission, uncertainty, isotopes, abundances)`
    /// — the five array arguments to `calibrate_energy`.
    type CalibrationInputs = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<ResonanceData>, Vec<f64>);

    /// Build a minimal valid input set for `calibrate_energy`, then let
    /// the caller mutate one field to drive a specific error path.
    fn minimal_calibration_inputs() -> CalibrationInputs {
        let iso = synthetic_single_resonance(72, 178, 176.4, 7.8);
        let energies: Vec<f64> = (0..50).map(|i| 5.0 + i as f64 * 0.4).collect();
        let transmission = vec![0.95; energies.len()];
        let uncertainty = vec![0.01; energies.len()];
        (energies, transmission, uncertainty, vec![iso], vec![1.0])
    }

    #[test]
    fn test_calibrate_all_nan_transmission_rejected() {
        // All-NaN transmission would previously yield zero valid bins,
        // compute_chi2() returned 0.0 for every grid point, and the
        // dof=1 fallback reported chi²_reduced = 0.0 as success.
        let (energies, mut transmission, uncertainty, isotopes, abundances) =
            minimal_calibration_inputs();
        for t in transmission.iter_mut() {
            *t = f64::NAN;
        }
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("all-NaN transmission must be rejected");
        match err {
            PipelineError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("valid"),
                    "error message should mention valid-bin count, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_calibrate_all_zero_uncertainty_rejected() {
        // All-zero uncertainty is the other path to zero valid bins
        // (sigma > 0 is required by the valid mask).
        let (energies, transmission, mut uncertainty, isotopes, abundances) =
            minimal_calibration_inputs();
        for s in uncertainty.iter_mut() {
            *s = 0.0;
        }
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("all-zero uncertainty must be rejected");
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
    }

    #[test]
    fn test_calibrate_nonfinite_flight_path_rejected() {
        // All non-finite or non-positive flight paths must produce
        // InvalidParameter, naming the offending field so the caller
        // can diagnose the source.
        let (energies, transmission, uncertainty, isotopes, abundances) =
            minimal_calibration_inputs();
        for bad_l in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            let result = calibrate_energy(
                &energies,
                &transmission,
                &uncertainty,
                &isotopes,
                &abundances,
                bad_l,
                293.6,
                None,
            );
            match result {
                Ok(_) => panic!("expected Err for L={bad_l}, got Ok"),
                Err(PipelineError::InvalidParameter(msg)) => {
                    assert!(
                        msg.contains("assumed_flight_path_m"),
                        "error message should name the offending field for L={bad_l}, got: {msg}"
                    );
                }
                Err(other) => panic!("expected InvalidParameter for L={bad_l}, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_calibrate_nonascending_energies_rejected() {
        let (mut energies, transmission, uncertainty, isotopes, abundances) =
            minimal_calibration_inputs();
        // Introduce a non-ascending pair.
        energies[10] = energies[9];
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("non-ascending energies must be rejected");
        match err {
            PipelineError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("ascending"),
                    "error message should mention ascending, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_calibrate_all_infinite_chi2_rejected() {
        // Regression for the post-grid-search `best_chi2.is_finite()`
        // guard.  Before the guard, an input whose every `compute_chi2`
        // evaluation returned `f64::INFINITY` left `best_chi2`
        // initialised at `INFINITY`, no candidate ever beat it, and the
        // function returned `Ok(CalibrationResult { reduced_chi_squared:
        // inf, .. })` — the same silent-failure surface as the
        // zero-valid-bins case, just with infinity instead of zero.
        //
        // Driving the all-infinity path: feed finite but wildly
        // out-of-scale transmission (1e308).  Each finite uncertainty
        // (0.01) makes the residual ((1e308 − T_model) / 0.01) overflow
        // to `+inf`, residual² is `inf`, the per-bin sum is `inf`, and
        // every grid candidate returns `inf`.  Crucially, the
        // transmission values stay `finite()` so they pass the up-front
        // `t.is_finite()` mask and the new post-search guard is the
        // only line of defence.
        let (energies, _transmission, uncertainty, isotopes, abundances) =
            minimal_calibration_inputs();
        let transmission = vec![1e308; energies.len()];
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("all-infinity chi² across the grid must be rejected");
        match err {
            PipelineError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("finite chi²") || msg.contains("best_chi2"),
                    "error message should explain the all-infinity grid-search failure, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_calibrate_nonfinite_energy_rejected() {
        let (mut energies, transmission, uncertainty, isotopes, abundances) =
            minimal_calibration_inputs();
        energies[7] = f64::NAN;
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("NaN energy must be rejected");
        assert!(
            matches!(err, PipelineError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
    }

    #[test]
    fn test_calibrate_energy_rejects_negative_abundance() {
        // A negative abundance silently flips the sign of `abd * n_total`
        // and `SampleParams::new` rejects the non-positive thickness;
        // every grid point then returns chi² = INFINITY and the user
        // sees a "no finite chi²" boundary error.  The up-front guard
        // converts this to an actionable diagnostic that names the
        // offending index.
        let (energies, transmission, uncertainty, isotopes, mut abundances) =
            minimal_calibration_inputs();
        abundances[0] = -0.5;
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("negative abundance must be rejected");
        match err {
            PipelineError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("abundances[0]"),
                    "error message should name the offending index, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_calibrate_energy_rejects_nan_abundance() {
        // NaN bypasses naive `< 0.0` guards (NaN comparisons are always
        // false), so the up-front check must pair `is_finite()` with the
        // sign predicate.  Without this guard, `abd * n_total` is NaN
        // for every density, every chi² is NaN, and the user sees a
        // confusing boundary-saturation message.
        let (energies, transmission, uncertainty, isotopes, mut abundances) =
            minimal_calibration_inputs();
        abundances[0] = f64::NAN;
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("NaN abundance must be rejected");
        match err {
            PipelineError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("abundances[0]"),
                    "error message should name the offending index, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_calibrate_energy_rejects_all_zero_abundances() {
        // Each individual zero abundance is legal (the isotope is simply
        // not present in this sample), but the sum being zero means
        // every per-isotope density is zero and the transmission model
        // collapses to T == 1 — the calibrator has no signal to fit
        // (L, t₀, n_total) against.  Reject up-front rather than letting
        // the search bottom out at the band boundary.
        let (energies, transmission, uncertainty, isotopes, _) = minimal_calibration_inputs();
        let abundances = vec![0.0; isotopes.len()];
        let err = calibrate_energy(
            &energies,
            &transmission,
            &uncertainty,
            &isotopes,
            &abundances,
            25.0,
            293.6,
            None,
        )
        .expect_err("all-zero abundances must be rejected");
        match err {
            PipelineError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("sum of abundances"),
                    "error message should mention the zero-sum cause, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    // ── n_total search-band regression tests ──────────────────────────
    //
    // Before the golden-section refactor, `calibrate_energy` scanned
    // n_total at a hard-coded 5-point linear grid `{5e-5, 1e-4,
    // 1.5e-4, 2e-4, 3e-4}` and then refined multiplicatively, leaving
    // the final density anchored inside `[2.25e-5, 4.95e-4]`
    // atoms/barn — incompatible with every realistic VENUS /
    // SoftwareX paper density (1 mm metal foils at ~5e-3, trace
    // matrix densities up to ~1e-2).  The refactor replaces the
    // multi-stage density refinement with a true golden-section
    // search in log10(n) on `[1e-5, 1e-2]`, and adds a boundary-
    // saturation guard for the (now possible) case where the
    // optimum lies outside that band.
    //
    // The tests below verify recovery at three representative
    // densities that span the search range (1e-5 lower edge,
    // 1e-3 middle of band that the old code could not reach,
    // 5e-3 SoftwareX U-238 density that was a factor-10× outside
    // the old reachable max) plus the explicit boundary-failure
    // diagnostic at densities outside the band.

    /// Fully-parameterised synthetic round-trip helper.  Builds data with
    /// two well-separated Hf-style resonances at the given true density and
    /// injected `(true_l = assumed_l · true_l_factor, true_t0_us)` offsets,
    /// runs `calibrate_energy`, and on success returns
    /// `(result, e_true, assumed_l)` so tests can assert on parameter
    /// recovery and corrected-energy accuracy against `e_true`.
    fn calibrate_round_trip(
        true_l_factor: f64,
        true_t0_us: f64,
        true_n: f64,
    ) -> Result<(CalibrationResult, Vec<f64>, f64), PipelineError> {
        let assumed_l = 25.0;
        let true_l = assumed_l * true_l_factor;
        let temperature_k = 293.6;

        let iso_a = synthetic_single_resonance(72, 178, 176.4, 7.8);
        let iso_b = synthetic_single_resonance(72, 178, 176.4, 22.0);
        let isotopes = vec![iso_a, iso_b];
        let abundances = vec![0.5, 0.5];

        let e_nominal: Vec<f64> = (0..150).map(|i| 5.0 + i as f64 * 0.2).collect();
        let tof_s: Vec<f64> = e_nominal
            .iter()
            .map(|&e| assumed_l * (NEUTRON_MASS_CONSTANT / e).sqrt())
            .collect();
        let true_t0_s = true_t0_us * 1e-6;
        let e_true: Vec<f64> = tof_s
            .iter()
            .map(|&t| NEUTRON_MASS_CONSTANT * (true_l / (t - true_t0_s)).powi(2))
            .collect();

        let pairs: Vec<_> = isotopes
            .iter()
            .zip(abundances.iter())
            .map(|(iso, &abd)| (iso.clone(), abd * true_n))
            .collect();
        let sample = SampleParams::new(temperature_k, pairs).expect("SampleParams creation failed");
        let t_model =
            transmission::forward_model(&e_true, &sample, None).expect("forward_model failed");
        let sigma = vec![0.01; e_nominal.len()];

        let result = calibrate_energy(
            &e_nominal,
            &t_model,
            &sigma,
            &isotopes,
            &abundances,
            assumed_l,
            temperature_k,
            None,
        )?;
        Ok((result, e_true, assumed_l))
    }

    /// Synthetic round-trip helper parameterised on `true_n` at the small
    /// legacy offsets (`+0.05 % L`, `+0.5 µs`) — the easy regime.  The
    /// wide-offset production regime is covered separately by
    /// `assert_wide_offset_recovery` (issue #634 review P0).
    fn calibrate_round_trip_at_density(
        true_n: f64,
    ) -> Result<(CalibrationResult, Vec<f64>, f64), PipelineError> {
        calibrate_round_trip(25.0125 / 25.0, 0.5, true_n)
    }

    /// Issue #634 review P0 regression: production-scale offsets — the
    /// module header's own VENUS correction magnitude (0.3 % L) and
    /// beyond-one-±1 %-L_scale-box offsets (1.2 % L, 6 µs, which also pin
    /// the affine re-anchoring composition `t0 ← t0 + ls·t0_k, ls ← ls·ls_k`)
    /// — must recover `(L, t0, n)` and the corrected energy grid.  The
    /// single-common-density anchor variant returned Ok in the WRONG basin
    /// here (density up to 139× off; at trace density with chi²_r ≈ 1e-4,
    /// no visible failure signal); the per-candidate golden-section density
    /// in the stage-2 anchor is what defeats it.
    fn assert_wide_offset_recovery(
        true_l_factor: f64,
        true_t0_us: f64,
        true_n: f64,
        n_rel_tol: f64,
    ) {
        let (result, e_true, assumed_l) = calibrate_round_trip(true_l_factor, true_t0_us, true_n)
            .expect("wide-offset calibration must succeed");
        let true_l = assumed_l * true_l_factor;
        assert!(
            (result.flight_path_m - true_l).abs() / true_l < 2e-3,
            "L: got {}, expected {true_l}",
            result.flight_path_m,
        );
        assert!(
            (result.t0_us - true_t0_us).abs() < 0.5,
            "t0: got {}, expected {true_t0_us}",
            result.t0_us,
        );
        assert!(
            (result.total_density - true_n).abs() / true_n < n_rel_tol,
            "n: got {}, expected {true_n}",
            result.total_density,
        );
        // The deliverable: the corrected energy grid tracks the truth.
        let mut rel: Vec<f64> = result
            .energies_corrected
            .iter()
            .zip(e_true.iter())
            .map(|(&c, &t)| (c - t).abs() / t)
            .collect();
        rel.sort_by(f64::total_cmp);
        let med = rel[rel.len() / 2];
        assert!(
            med < 5e-3,
            "median corrected-energy rel err {med:.3e} exceeds 5e-3"
        );
    }

    #[test]
    fn test_calibrate_wide_offset_venus_scale_at_paper_density() {
        // 0.3 % L + 1 µs at the SoftwareX U-238 foil density.
        assert_wide_offset_recovery(1.003, 1.0, 5e-3, 0.15);
    }

    #[test]
    fn test_calibrate_wide_offset_venus_scale_at_trace_density() {
        // Same offsets at trace density — the silent-failure regime
        // (wrong answer previously carried chi²_r ≈ 1e-4).
        assert_wide_offset_recovery(1.003, 1.0, 2e-5, 0.3);
    }

    #[test]
    fn test_calibrate_wide_offset_beyond_one_box_at_paper_density() {
        // 1.2 % L exceeds the per-fit ±1 % L_scale box → exercises the
        // re-anchoring composition; 6 µs t0 is mid coarse-grid.
        assert_wide_offset_recovery(1.012, 6.0, 5e-3, 0.15);
    }

    #[test]
    fn test_calibrate_wide_offset_beyond_one_box_at_midband_density() {
        assert_wide_offset_recovery(1.012, 6.0, 1.5e-4, 0.15);
    }

    /// Near-lower-edge density: `true_n = 2e-5` sits just inside the
    /// `[1e-5, 1e-2]` documented user-supported interval — approximately
    /// 0.3 decades (a factor of 2) above the lower documented edge,
    /// comfortably outside the boundary guard's `5 %`-linear tolerance
    /// window.  Recovery must succeed.  Note the 30 % relative tolerance —
    /// at this low density the chi² landscape is shallow (single-resonance
    /// synthetic, weak signal), so the recovered density can drift further
    /// from the true value than at mid-band; the test still meaningfully
    /// distinguishes "we found roughly the right decade" from the old
    /// behaviour of being unable to reach the value at all.  The
    /// `test_calibrate_energy_boundary_saturation_error` test separately
    /// verifies the guard fires for genuinely-out-of-band densities.
    #[test]
    fn test_calibrate_energy_recovers_density_1e_5() {
        let true_n = 2e-5;
        let (result, _, _) = calibrate_round_trip_at_density(true_n)
            .expect("calibration at true_n=2e-5 must succeed");
        assert!(
            (result.total_density - true_n).abs() / true_n < 0.3,
            "n: got {}, expected {}",
            result.total_density,
            true_n,
        );
        assert!(result.reduced_chi_squared.is_finite());
    }

    /// Documented lower bound: `true_n = 1.0e-5` atoms/barn is exactly
    /// the lower edge promised by the `calibrate_energy` rustdoc.  Before
    /// the search-band widening the boundary-saturation guard's
    /// `~5 %`-linear tolerance trimmed a sliver off either side and
    /// rejected truly-at-the-edge optima with a "true optimum likely
    /// lies outside this band" diagnostic that contradicted the docs.
    /// With the internal band widened to `[~5e-6, ~2e-2]`, an optimum
    /// at the documented edge sits ~0.3 decades inside the buffer and
    /// is accepted — the user-facing contract here is that the call
    /// returns `Ok(_)` (no boundary-saturation error) and recovers a
    /// density close to truth in log-space, not that the recovered
    /// value is bit-exactly bounded by the documented interval: chi²
    /// minimisation can land slightly outside `[1e-5, 1e-2]` even when
    /// the true density sits at the edge, and that is correct
    /// behaviour for a smooth optimisation landscape.
    #[test]
    fn test_calibrate_energy_accepts_density_at_documented_lower_bound() {
        let true_n = 1.0e-5;
        let (result, _, _) = calibrate_round_trip_at_density(true_n)
            .expect("calibration at the documented lower edge 1e-5 must succeed");
        // Log-space tolerance because the chi² landscape is shallow at
        // this trace density (single-resonance synthetic, weak signal):
        // a recovered-vs-truth ratio of 2× corresponds to 0.3 in
        // log10(n) and is the empirically reasonable precision floor.
        let log_err = (result.total_density.log10() - true_n.log10()).abs();
        assert!(
            log_err < 0.3,
            "log10(n) error {log_err} too large; recovered {} vs truth {true_n}",
            result.total_density,
        );
        assert!(result.reduced_chi_squared.is_finite());
    }

    /// Documented upper bound: `true_n = 1.0e-2` atoms/barn is exactly
    /// the upper edge promised by `calibrate_energy`'s rustdoc — the
    /// `1 mm metal foil` use case that drives the SoftwareX paper's
    /// calibration narrative.  Sister test to
    /// `test_calibrate_energy_accepts_density_at_documented_lower_bound`;
    /// the search-band widening keeps the upper documented edge inside
    /// the boundary guard's tolerance buffer.  As with the lower-bound
    /// test, the assertion is on chi²-resolution recovery (log-space
    /// proximity to truth), not on hard-bounding the result inside
    /// `[1e-5, 1e-2]` — a smooth optimum at the edge can land just
    /// outside without indicating any defect.
    #[test]
    fn test_calibrate_energy_accepts_density_at_documented_upper_bound() {
        let true_n = 1.0e-2;
        let (result, _, _) = calibrate_round_trip_at_density(true_n)
            .expect("calibration at the documented upper edge 1e-2 must succeed");
        // Tighter log-space tolerance than the lower edge: at this
        // high density the resonance is saturated, the chi²
        // landscape is sharp and the (L, t₀, n) trade-off basin is
        // narrow.  log_err < 0.05 ≈ 12 % linear is comfortable.
        let log_err = (result.total_density.log10() - true_n.log10()).abs();
        assert!(
            log_err < 0.05,
            "log10(n) error {log_err} too large; recovered {} vs truth {true_n}",
            result.total_density,
        );
        assert!(result.reduced_chi_squared.is_finite());
    }

    /// Phase-1-grid-point density: `true_n = 1e-4` was the historical
    /// reachable-band centre.  Recovery is the easiest case for the
    /// chi² landscape and tightens the tolerance accordingly.
    #[test]
    fn test_calibrate_energy_recovers_density_1e_4() {
        let true_n = 1e-4;
        let (result, _, _) = calibrate_round_trip_at_density(true_n)
            .expect("calibration at true_n=1e-4 must succeed");
        assert!(
            (result.total_density - true_n).abs() / true_n < 0.1,
            "n: got {}, expected {}",
            result.total_density,
            true_n,
        );
        assert!(result.reduced_chi_squared.is_finite());
    }

    /// Mid-band density: `true_n = 1e-3` was **unreachable** under
    /// the previous 5-point scan + multiplicative refinement; the
    /// best the old code could return was ~4.95e-4.  After the
    /// log-space golden-section refactor this density must round-
    /// trip with full Phase-3 precision.
    #[test]
    fn test_calibrate_energy_recovers_density_1e_3() {
        let true_n = 1e-3;
        let (result, _, _) = calibrate_round_trip_at_density(true_n)
            .expect("calibration at true_n=1e-3 must succeed");
        assert!(
            (result.total_density - true_n).abs() / true_n < 0.1,
            "n: got {}, expected {} — old code saturated at ~4.95e-4",
            result.total_density,
            true_n,
        );
        assert!(result.reduced_chi_squared.is_finite());
    }

    /// SoftwareX U-238 reference density: 1 mm metal foil at ~5e-3
    /// atoms/barn.  This is the density the paper figure scripts
    /// (`gen_fig_physics.py`, `gen_fig_closed_loop.py`) use and
    /// that the paper's calibration narrative relies on; it sits a
    /// factor 10× above the old reachable maximum.
    #[test]
    fn test_calibrate_energy_recovers_density_5e_3() {
        let true_n = 5e-3;
        let (result, _, _) = calibrate_round_trip_at_density(true_n)
            .expect("calibration at true_n=5e-3 must succeed");
        assert!(
            (result.total_density - true_n).abs() / true_n < 0.1,
            "n: got {}, expected {} — old code saturated at ~4.95e-4 (10× too low)",
            result.total_density,
            true_n,
        );
        assert!(result.reduced_chi_squared.is_finite());
    }

    /// Out-of-band saturation: `true_n = 1.0` atoms/barn is two
    /// orders of magnitude above the upper search bound (`1e-2`).
    /// The golden-section minimum must land on the upper bound,
    /// and the boundary-saturation guard must turn that into an
    /// `Err(InvalidParameter)` rather than the silent railed answer
    /// the old 5-point scan would have returned (the old code
    /// would have railed to `4.95e-4`, six orders of magnitude
    /// below truth, with no diagnostic).
    #[test]
    fn test_calibrate_energy_boundary_saturation_error() {
        let true_n = 1.0;
        let err = calibrate_round_trip_at_density(true_n)
            .expect_err("density outside search band must trigger boundary guard");
        match err {
            PipelineError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("search boundary") || msg.contains("boundary"),
                    "error must explain boundary saturation, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }
}
