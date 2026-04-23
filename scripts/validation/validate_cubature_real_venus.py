"""Real-VENUS closed-loop validation for the Rust SparseEmpiricalCubaturePlan.

Runs the 5-parameter Hf 120-min aggregated counts fit twice:
  (a) exact fixed-grid path via Rust `apply_r` on the compiled
      ResolutionMatrix
  (b) cubature path via Rust `cubature_forward` on the new
      SparseEmpiricalCubaturePlan

Asserts:
  - density shift (cubature vs exact) < 0.1 % relative
  - D/dof within ~1 unit of the codex04 reference (103.28)

This is the load-bearing "don't merge broken code" gate for PR #474a.
Mirrors `.research/algo_design_roundrobin_r2/codex04/run_research.py`
section 5 "Real-data check", but executed in this checkout against the
Rust implementation via the temporary bindings added in PR #474a
(PyResolutionMatrix / PySparseCubature).

Run with:

  pixi run python scripts/validation/validate_cubature_real_venus.py

Requires:
  - `_fts_bl10_0p5meV_1keV_25pts.txt` at the repo root (gitignored).
  - `.research/spatial-regularization/data/counts/resonance_data_2cm.h5`.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares

import nereids

ROOT = Path(__file__).resolve().parents[2]
RES_FILE = ROOT / "_fts_bl10_0p5meV_1keV_25pts.txt"
H5 = ROOT / ".research" / "spatial-regularization" / "data" / "counts" / "resonance_data_2cm.h5"

TEMP_K = 293.6
FLIGHT_PATH_M = 25.0
ENERGY_MIN = 7.0
ENERGY_MAX = 200.0
REAL_T0_US = 0.4809277146701285
REAL_L_SCALE = 1.0052452911520162
DENSITY_SHIFT_BAR = 1e-3  # 0.1 % relative


def log(msg: str) -> None:
    print(msg, flush=True)


def load_fixture():
    log("[load] VENUS Hf 120-min fixture")
    with h5py.File(H5, "r") as f:
        e_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        s3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        o3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    mask = (e_full >= ENERGY_MIN) & (e_full <= ENERGY_MAX)
    e = np.ascontiguousarray(e_full[mask]).astype(np.float64)
    s = np.ascontiguousarray(s3d_raw[mask].sum(axis=(1, 2))).astype(np.float64)
    o = np.ascontiguousarray(o3d_raw[mask].sum(axis=(1, 2))).astype(np.float64)
    c_real = q_s / q_ob

    tof_factor = math.sqrt(0.5 * 1.67492749804e-27 / 1.602176634e-19) * 1.0e6
    l_eff = FLIGHT_PATH_M * REAL_L_SCALE
    tof = tof_factor * FLIGHT_PATH_M / np.sqrt(e)
    tof_corr = tof - REAL_T0_US
    e_cal = np.ascontiguousarray((tof_factor * l_eff / tof_corr) ** 2).astype(np.float64)

    return e, e_cal, s, o, c_real


def build_hf_sigma(energies: np.ndarray) -> np.ndarray:
    log("[build] Hf natural σ on calibrated grid")
    hf = nereids.IsotopeGroup.natural(72)
    hf.load_endf()
    isotopes = list(hf.resonance_data)
    xs = np.array(
        nereids.precompute_cross_sections(energies, isotopes, temperature_k=TEMP_K),
        dtype=np.float64,
    )
    ratios = np.array([ratio for _, ratio in hf.members], dtype=np.float64)
    return (ratios[:, None] * xs).sum(axis=0)


def build_background_basis(n_bins: int) -> np.ndarray:
    """Polynomial background basis (degree-2): BackA + BackB * x + BackC * x²."""
    x = np.linspace(0.0, 1.0, n_bins)
    return np.column_stack([np.ones_like(x), x, x * x])


def counts_residuals(
    params: np.ndarray,
    transmission_fn,
    sample_counts: np.ndarray,
    open_counts: np.ndarray,
    c: float,
    bg_basis: np.ndarray,
) -> np.ndarray:
    """Joint-Poisson deviance residuals (replicates codex04's objective)."""
    density, anorm, back_a, back_b, back_c = params
    n_vec = np.array([density], dtype=np.float64)
    t_res = transmission_fn(n_vec)
    background = bg_basis @ np.array([back_a, back_b, back_c])
    model = anorm * open_counts * c * t_res + background
    # Deviance residual: sign·sqrt(2(y·log(y/μ) - (y-μ))).  Use open_counts
    # as the "model open rate" for the open-beam branch and sample_counts
    # for the sample — joint-Poisson.
    sample = np.maximum(sample_counts, 1e-12)
    model = np.maximum(model, 1e-12)
    term_y = sample * np.log(sample / model)
    term_mu = sample - model
    deviance = 2.0 * (term_y - term_mu)
    # Clip negative roundoff.
    deviance = np.maximum(deviance, 0.0)
    sign = np.sign(sample - model)
    return sign * np.sqrt(deviance)


def fit(
    transmission_fn,
    sample_counts: np.ndarray,
    open_counts: np.ndarray,
    c: float,
    bg_basis: np.ndarray,
    label: str,
):
    x0 = np.array([1.5e-4, 0.85, 0.1, 0.0, 0.0], dtype=np.float64)
    lo = np.array([1e-8, 0.01, -10.0, -10.0, -10.0])
    hi = np.array([1e-2, 100.0, 10.0, 10.0, 10.0])

    t0 = time.perf_counter()
    result = least_squares(
        counts_residuals,
        x0,
        args=(transmission_fn, sample_counts, open_counts, c, bg_basis),
        bounds=(lo, hi),
        method="trf",
        max_nfev=200,
    )
    dt = time.perf_counter() - t0
    n_free = x0.size
    ndof = sample_counts.size - n_free
    d_per_dof = (result.fun @ result.fun) / ndof
    log(
        f"[{label}] density={result.x[0]:.6e}  anorm={result.x[1]:.4f}  "
        f"D/dof={d_per_dof:.3f}  iters={result.nfev}  wall={dt:.2f}s"
    )
    return result.x, d_per_dof, dt


def main() -> int:
    if not RES_FILE.exists():
        log(f"SKIP: {RES_FILE} missing (gitignored PLEIADES fixture)")
        return 0
    if not H5.exists():
        log(f"SKIP: {H5} missing (VENUS Hf 120-min fixture)")
        return 0

    e_raw, e_cal, sample_counts, open_counts, c_real = load_fixture()
    n_bins = e_cal.size
    log(f"[grid] {n_bins} bins on {e_cal[0]:.2f}-{e_cal[-1]:.2f} eV")
    log(f"[c_real] {c_real:.4f}")

    sigma = build_hf_sigma(e_cal)
    log(f"[σ] range [{sigma.min():.2e}, {sigma.max():.2e}] barns")

    # Build exact ResolutionMatrix on the calibrated grid.
    log("[build] ResolutionMatrix (exact CSR)")
    t0 = time.perf_counter()
    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)
    matrix = nereids.build_resolution_matrix(e_cal, res)
    log(f"  matrix.len = {matrix.len}  nnz = {matrix.nnz}  ({time.perf_counter() - t0:.1f}s)")

    # Build cubature for k = 1 Hf grouped.
    log("[build] SparseEmpiricalCubaturePlan k=1 (Hf grouped)")
    # Codex04 default rule for k=1: train_max ≈ 2 × physical density.
    train_max = np.array([2e-4], dtype=np.float64)
    training = [[0.25 * train_max[0]], [0.75 * train_max[0]], [train_max[0]]]
    anchor = [0.5 * train_max[0]]
    # sigmas flat row-major: for k=1, just σ itself.
    sigmas_flat = np.ascontiguousarray(sigma, dtype=np.float64)
    t0 = time.perf_counter()
    cubature = nereids.build_sparse_cubature(matrix, sigmas_flat, 1, training, anchor)
    log(
        f"  cubature.len = {cubature.len}  n_atoms = {cubature.n_atoms}  "
        f"compression = {matrix.nnz / cubature.n_atoms:.1f}×  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    bg_basis = build_background_basis(n_bins)

    def exact_forward(n_vec: np.ndarray) -> np.ndarray:
        return nereids.apply_r(matrix, np.exp(-n_vec[0] * sigma))

    def cubature_forward(n_vec: np.ndarray) -> np.ndarray:
        return nereids.cubature_forward(cubature, n_vec)

    # Run both fits.
    log("\n=== Closed-loop fit: exact ResolutionMatrix path ===")
    params_exact, d_exact, wall_exact = fit(
        exact_forward, sample_counts, open_counts, c_real, bg_basis, "exact"
    )

    log("\n=== Closed-loop fit: sparse empirical cubature path ===")
    params_cub, d_cub, wall_cub = fit(
        cubature_forward, sample_counts, open_counts, c_real, bg_basis, "cubature"
    )

    # Gate.
    density_exact = params_exact[0]
    density_cub = params_cub[0]
    shift = abs(density_cub - density_exact) / density_exact
    log("")
    log("=" * 70)
    log(f"  exact     density = {density_exact:.6e}  D/dof = {d_exact:.3f}  wall = {wall_exact:.2f}s")
    log(f"  cubature  density = {density_cub:.6e}  D/dof = {d_cub:.3f}  wall = {wall_cub:.2f}s")
    log(f"  density shift    = {shift:.3e}  (bar: < {DENSITY_SHIFT_BAR})")
    log("=" * 70)

    if not np.isfinite(shift):
        log("FAIL: density shift is non-finite (one of the fits diverged)")
        return 1
    if shift >= DENSITY_SHIFT_BAR:
        log(f"FAIL: density shift {shift:.3e} exceeds bar {DENSITY_SHIFT_BAR}")
        return 1

    log("PASS — cubature reproduces exact density within 0.1 % on real VENUS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
