//! Bounded Nelder-Mead simplex minimizer.
//!
//! Derivative-free polish optimizer used after a gradient-based stage to
//! escape stall points.  Memo 35 §P2.1 and EG5 establish that, for
//! backgrounded counts-path fits, a single L-BFGS start frequently stalls
//! at the initial guess (1/20 self-flagged convergence on the EG2 S1 C_full
//! regime), while a Nelder-Mead polish from that stall point resolves the
//! failure cleanly (10/20 convergence, density bias from −5.94% to +0.013%,
//! D/DOF from 905 to 1.001).
//!
//! ## Algorithm
//!
//! Standard Nelder-Mead simplex with reflection / expansion / contraction /
//! shrink (Nelder & Mead 1965), using the classical coefficients
//! (α=1, γ=2, ρ=0.5, σ=0.5).
//!
//! Box bounds are enforced via **reflection at the wall**: when a proposed
//! vertex would leave the feasible box, each coordinate is reflected back
//! inside (`x_i ← 2·bound − x_i` once, then clamped).  This preserves the
//! simplex volume in bulk while keeping all vertices feasible.
//!
//! ## Convergence
//!
//! Terminates when both
//! - the maximum coordinate distance from any simplex vertex to the current
//!   best vertex (`simplex[0]`) is below `xatol`, AND
//! - the range of objective values across the simplex is below `fatol`.
//!
//! This matches scipy's `optimize.minimize(method='Nelder-Mead')` simplex-
//! spread check (`max(|sim[i] - sim[0]|)` over coordinates) behaviour.

use crate::error::FittingError;

/// Nelder-Mead configuration.
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Absolute tolerance on vertex displacement.
    pub xatol: f64,
    /// Absolute tolerance on objective range across the simplex.
    pub fatol: f64,
    /// Maximum number of simplex iterations (each iteration = at most a
    /// constant number of objective evaluations).
    pub max_iter: usize,
    /// Initial simplex edge length, used as a signed multiplier on each
    /// coordinate: `step_i = initial_step_frac * x0_i` (so 0.05 gives a
    /// 5 % perturbation in the direction of the coordinate's sign).
    /// When `|x0_i| < 1e-8` the fallback `initial_step_abs` is used
    /// instead.  Note: this is NOT `initial_step_frac * max(|x0|, 1)`
    /// — for `|x0| < 1` the perturbation is therefore smaller than
    /// `initial_step_frac` itself.
    pub initial_step_frac: f64,
    /// Small absolute initial step for parameters whose `|x_0| < 1e-8`.
    pub initial_step_abs: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        // Defaults match scipy.optimize.minimize(method='Nelder-Mead'):
        // xatol = 1e-4, fatol = 1e-4.  For the polish regime described in
        // EG5 we use tighter tolerances (1e-9 / 1e-10) on the caller side.
        Self {
            xatol: 1e-4,
            fatol: 1e-4,
            max_iter: 5000,
            initial_step_frac: 0.05,
            initial_step_abs: 0.00025,
        }
    }
}

/// Nelder-Mead result.
#[derive(Debug, Clone)]
pub struct NelderMeadResult {
    /// Best parameter vector found.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fun: f64,
    /// Number of simplex iterations performed.
    pub iterations: usize,
    /// Total objective evaluations (including initial simplex).
    pub n_evals: usize,
    /// `true` if both `xatol` and `fatol` were satisfied before hitting
    /// `max_iter`.  Per memo 35 §P2.3, acceptance should be judged from
    /// the deviance value, not this flag.
    pub self_converged: bool,
}

/// Minimize a scalar objective with optional per-coordinate box bounds.
///
/// - `f` must be non-panicking; it may return `Err` to signal an infeasible
///   point (the NM logic treats the vertex as +∞ and contracts away from it).
/// - `x0` is the initial point.  An initial simplex of `n+1` vertices is
///   built by perturbing each coordinate in turn.
/// - `bounds`, if present, must have the same length as `x0`.  Each pair is
///   `(lower, upper)`; use `f64::NEG_INFINITY` / `f64::INFINITY` to disable.
///
/// ## Panics
///
/// Does not panic on infeasible objective values.  Panics only if `x0` is
/// empty or `bounds.len() != x0.len()`.
pub fn nelder_mead_minimize<F>(
    mut f: F,
    x0: &[f64],
    bounds: Option<&[(f64, f64)]>,
    config: &NelderMeadConfig,
) -> Result<NelderMeadResult, FittingError>
where
    F: FnMut(&[f64]) -> Result<f64, FittingError>,
{
    let n = x0.len();
    assert!(n > 0, "nelder_mead_minimize: x0 must not be empty");
    if let Some(b) = bounds {
        assert_eq!(
            b.len(),
            n,
            "nelder_mead_minimize: bounds length {} != x0 length {}",
            b.len(),
            n
        );
        for (i, &(lo, hi)) in b.iter().enumerate() {
            assert!(
                lo <= hi,
                "nelder_mead_minimize: bound {i} has lo {lo} > hi {hi}"
            );
        }
    }
    // Classical Nelder-Mead coefficients.
    const ALPHA: f64 = 1.0; // reflection
    const GAMMA: f64 = 2.0; // expansion
    const RHO: f64 = 0.5; // contraction
    const SIGMA: f64 = 0.5; // shrink

    // Project a point onto the bounding box.
    let project = |x: &mut [f64]| {
        if let Some(b) = bounds {
            for (xi, &(lo, hi)) in x.iter_mut().zip(b.iter()) {
                if *xi < lo {
                    *xi = 2.0 * lo - *xi; // reflect
                    if *xi > hi {
                        *xi = hi;
                    }
                    if *xi < lo {
                        *xi = lo;
                    }
                } else if *xi > hi {
                    *xi = 2.0 * hi - *xi;
                    if *xi < lo {
                        *xi = lo;
                    }
                    if *xi > hi {
                        *xi = hi;
                    }
                }
            }
        }
    };

    // Objective evaluator that turns Err into +∞ (infeasible → avoid).
    let mut n_evals = 0usize;
    let mut eval = |x: &[f64], f: &mut F| -> f64 {
        n_evals += 1;
        match f(x) {
            Ok(v) if v.is_finite() => v,
            _ => f64::INFINITY,
        }
    };

    // Build initial simplex.  Vertex 0 is x0; vertex i>0 perturbs coord i-1.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut fvals: Vec<f64> = Vec::with_capacity(n + 1);
    let mut v0 = x0.to_vec();
    project(&mut v0);
    fvals.push(eval(&v0, &mut f));
    simplex.push(v0.clone());
    for i in 0..n {
        let mut v = v0.clone();
        let base = v[i];
        let step = if base.abs() > 1e-8 {
            config.initial_step_frac * base
        } else {
            config.initial_step_abs
        };
        v[i] = base + step;
        project(&mut v);
        // If projection collapsed the perturbation (e.g. vertex hit a wall
        // and the reflection / clamp put it back on the original coord),
        // try the opposite direction so the simplex remains non-degenerate.
        if (v[i] - base).abs() < 1e-14 {
            v[i] = base - step;
            project(&mut v);
            if (v[i] - base).abs() < 1e-14 {
                // Give up and use the tiny default step — the simplex is
                // near a corner but still has to start somewhere.
                v[i] = base
                    + config
                        .initial_step_abs
                        .copysign(if base >= 0.0 { 1.0 } else { -1.0 });
                project(&mut v);
            }
        }
        fvals.push(eval(&v, &mut f));
        simplex.push(v);
    }

    // Sort simplex by ascending f-value.
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    simplex = order.iter().map(|&i| simplex[i].clone()).collect();
    fvals = order.iter().map(|&i| fvals[i]).collect();

    let mut centroid = vec![0.0; n];
    let mut xr = vec![0.0; n];
    let mut xe = vec![0.0; n];
    let mut xc = vec![0.0; n];

    let mut iter = 0usize;
    let mut self_converged = false;
    while iter < config.max_iter {
        iter += 1;

        // Convergence check.
        let fmin = fvals[0];
        let fmax = fvals[n];
        let frange = fmax - fmin;
        // Max coordinate distance from any vertex to the best vertex
        // (`simplex[0]`).  Matches the scipy Nelder-Mead spread check.
        let mut xrange = 0.0f64;
        for v in simplex.iter() {
            for (j, &xj) in v.iter().enumerate() {
                let d = (xj - simplex[0][j]).abs();
                if d > xrange {
                    xrange = d;
                }
            }
        }
        if xrange <= config.xatol && frange <= config.fatol {
            self_converged = true;
            break;
        }

        // Centroid of all vertices except the worst.
        for (j, c) in centroid.iter_mut().enumerate() {
            let mut s = 0.0;
            for v in simplex.iter().take(n) {
                s += v[j];
            }
            *c = s / (n as f64);
        }

        // Reflection.
        for j in 0..n {
            xr[j] = centroid[j] + ALPHA * (centroid[j] - simplex[n][j]);
        }
        project(&mut xr);
        let fxr = eval(&xr, &mut f);

        if fvals[0] <= fxr && fxr < fvals[n - 1] {
            simplex[n] = xr.clone();
            fvals[n] = fxr;
        } else if fxr < fvals[0] {
            // Expansion.
            for j in 0..n {
                xe[j] = centroid[j] + GAMMA * (xr[j] - centroid[j]);
            }
            project(&mut xe);
            let fxe = eval(&xe, &mut f);
            if fxe < fxr {
                simplex[n] = xe.clone();
                fvals[n] = fxe;
            } else {
                simplex[n] = xr.clone();
                fvals[n] = fxr;
            }
        } else {
            // Contraction.  Outside contraction (fxr ≥ f[n-1]) chooses the
            // reflected side; inside contraction chooses the worst side.
            let (x_src, f_src) = if fxr < fvals[n] {
                (&xr, fxr)
            } else {
                (&simplex[n], fvals[n])
            };
            for j in 0..n {
                xc[j] = centroid[j] + RHO * (x_src[j] - centroid[j]);
            }
            project(&mut xc);
            let fxc = eval(&xc, &mut f);
            if fxc < f_src {
                simplex[n] = xc.clone();
                fvals[n] = fxc;
            } else {
                // Shrink toward the best vertex.  Snapshot the best vertex
                // first to avoid aliasing borrows when mutating
                // `simplex[i]`.
                let best = simplex[0].clone();
                for i in 1..=n {
                    for (j, xj) in simplex[i].iter_mut().enumerate() {
                        *xj = best[j] + SIGMA * (*xj - best[j]);
                    }
                    project(&mut simplex[i]);
                    fvals[i] = eval(&simplex[i], &mut f);
                }
            }
        }

        // Re-sort simplex (O(n log n) — n is small for our use).
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();
    }

    Ok(NelderMeadResult {
        x: simplex[0].clone(),
        fun: fvals[0],
        iterations: iter,
        n_evals,
        self_converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nm_quadratic_1d_converges() {
        // f(x) = (x − 3)².
        let f = |x: &[f64]| Ok((x[0] - 3.0).powi(2));
        let cfg = NelderMeadConfig {
            xatol: 1e-10,
            fatol: 1e-12,
            max_iter: 5000,
            initial_step_frac: 0.1,
            initial_step_abs: 0.01,
        };
        let r = nelder_mead_minimize(f, &[0.0], None, &cfg).unwrap();
        assert!((r.x[0] - 3.0).abs() < 1e-6, "x = {:?}", r.x);
        assert!(r.fun < 1e-12);
        assert!(r.self_converged);
    }

    #[test]
    fn test_nm_rosenbrock_2d() {
        // Classic: f(x,y) = (1-x)² + 100(y-x²)², minimum at (1,1) with f=0.
        let f = |x: &[f64]| Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2));
        let cfg = NelderMeadConfig {
            xatol: 1e-6,
            fatol: 1e-8,
            max_iter: 10_000,
            initial_step_frac: 0.1,
            initial_step_abs: 0.01,
        };
        let r = nelder_mead_minimize(f, &[-1.2, 1.0], None, &cfg).unwrap();
        assert!(
            (r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] - 1.0).abs() < 1e-3,
            "Rosenbrock minimizer off: x = {:?} fun = {}",
            r.x,
            r.fun
        );
        assert!(r.fun < 1e-6);
    }

    #[test]
    fn test_nm_respects_bounds_reflection() {
        // f(x) = (x − 5)²; but bound x to [0, 2] — true minimum inside the
        // box is at x = 2 (boundary).  Verify NM returns x ≈ 2 and never a
        // value outside the box during search.
        let lo = 0.0;
        let hi = 2.0;
        let f = {
            move |x: &[f64]| -> Result<f64, FittingError> {
                assert!(
                    x[0] >= lo - 1e-12 && x[0] <= hi + 1e-12,
                    "NM passed out-of-bounds x = {}",
                    x[0]
                );
                Ok((x[0] - 5.0).powi(2))
            }
        };
        let cfg = NelderMeadConfig::default();
        let bounds = [(lo, hi)];
        let r = nelder_mead_minimize(f, &[1.0], Some(&bounds), &cfg).unwrap();
        assert!(
            (r.x[0] - 2.0).abs() < 1e-2,
            "expected x ≈ 2, got {}",
            r.x[0]
        );
        assert!(r.x[0] >= lo - 1e-12 && r.x[0] <= hi + 1e-12);
    }

    #[test]
    fn test_nm_handles_infeasible_objective() {
        // f returns Err for x[0] < 0.1, otherwise (x-0.5)^2.  NM should
        // find x ≈ 0.5 and never return the infeasible region.
        let f = |x: &[f64]| -> Result<f64, FittingError> {
            if x[0] < 0.1 {
                Err(FittingError::EvaluationFailed("x too small".into()))
            } else {
                Ok((x[0] - 0.5).powi(2))
            }
        };
        let cfg = NelderMeadConfig {
            xatol: 1e-8,
            fatol: 1e-10,
            max_iter: 5000,
            initial_step_frac: 0.2,
            initial_step_abs: 0.05,
        };
        let r = nelder_mead_minimize(f, &[1.0], None, &cfg).unwrap();
        assert!(
            (r.x[0] - 0.5).abs() < 1e-3,
            "expected x ≈ 0.5, got {} (fun = {})",
            r.x[0],
            r.fun
        );
    }
}
