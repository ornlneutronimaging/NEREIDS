"""Re-optimize T/density for coarse and dense IC forward grids.

This is a Gaussian-transmission diagnostic with the multiplicative quadratic
baseline profiled at every step. It retains every observed bin in 8–45 eV,
prints JSON, and performs no writes.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import nereids

from archive_inputs import archive_root

ARCHIVE = archive_root()
COUNTS = ARCHIVE / "01_spectral_lineshape_bias/data/region_counts.npz"
CACHE = ARCHIVE / "notebooks/calib_IC_jendl5_24685.json"
FLIGHT_PATH_M = 25.0
FEATURES_EV = (10.35, 13.92, 23.95, 35.17, 39.15)


def main() -> None:
    d = np.load(COUNTS, allow_pickle=False)
    c = json.loads(CACHE.read_text())
    keep = (d["energies"] >= 4.0) & (d["energies"] <= 120.0)
    e_all = np.asarray(
        nereids.tof_to_energy_centers(
            d["edges"], FLIGHT_PATH_M * float(c["l_scale"]), float(c["t0_us"])
        )
    )[keep]
    win = (e_all > 8.0) & (e_all < 45.0)
    e = e_all[win]
    sample = d["counts_sample"][keep][win]
    ob = d["counts_ob"][keep][win]
    measured = (sample / ob) * float(d["pc_ob"] / d["pc_sample"])
    sigma = measured * np.sqrt(1.0 / np.maximum(sample, 1.0) + 1.0 / np.maximum(ob, 1.0))
    e_ref = math.sqrt(float(e.min() * e.max()))
    z = np.log(e / e_ref)
    basis = np.column_stack((np.ones_like(z), z, z**2))
    ta181 = nereids.load_endf(73, 181, library="jendl5")
    ic = nereids.IkedaCarpenter(
        flight_path_m=FLIGHT_PATH_M,
        e_min_ev=4.0,
        e_max_ev=120.0,
        alpha=nereids.EnergyLaw.sqrt_e(float(c["ic_a0"]), float(c["ic_a1"])),
        beta=float(c["ic_beta"]),
        r=nereids.EnergyLaw.const(float(c["ic_r"])),
        channel_fwhm_us=float(c["ic_psr"]),
    )
    resolution = ic.as_tabulated()
    dense_grid = np.unique(np.concatenate((np.geomspace(4.0, 120.0, 20_000), e)))

    def fit_one(name: str, grid: np.ndarray) -> dict[str, object]:
        calls = 0

        def evaluate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            nonlocal calls
            calls += 1
            temperature = math.exp(float(q[0]))
            density = math.exp(float(q[1]))
            raw_grid = np.asarray(
                nereids.forward_model(
                    grid,
                    [(ta181, density)],
                    temperature_k=temperature,
                    resolution=resolution,
                )
            )
            raw = raw_grid if name == "coarse" else np.interp(e, grid, raw_grid)
            design = raw[:, None] * basis
            aw = design / sigma[:, None]
            coef, *_ = np.linalg.lstsq(aw, measured / sigma, rcond=None)
            prediction = design @ coef
            return (measured - prediction) / sigma, coef, prediction

        start = time.perf_counter()
        result = least_squares(
            lambda q: evaluate(q)[0],
            x0=np.log([1058.69727434866, 0.000632508761631833]),
            bounds=(np.log([200.0, 1e-5]), np.log([2000.0, 0.002])),
            diff_step=1e-3,
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            max_nfev=30,
        )
        residual, coef, prediction = evaluate(result.x)
        rows = []
        for e0 in FEATURES_EV:
            idxs = np.flatnonzero(np.abs(e - e0) <= 0.25)
            idx = idxs[np.argmax(np.abs(residual[idxs]))]
            rows.append(
                {
                    "feature_ev": e0,
                    "observed_bin_ev": float(e[idx]),
                    "residual_sigma": float(residual[idx]),
                }
            )
        return {
            "success": bool(result.success),
            "message": result.message,
            "optimizer_nfev": int(result.nfev),
            "actual_forward_calls": calls,
            "seconds": time.perf_counter() - start,
            "temperature_k": math.exp(float(result.x[0])),
            "density_atoms_per_barn": math.exp(float(result.x[1])),
            "baseline": coef.tolist(),
            "ssr": float(residual @ residual),
            "rms_sigma": float(np.sqrt(np.mean(residual**2))),
            "feature_rows": rows,
            "max_abs_prediction": float(np.max(np.abs(prediction))),
        }

    coarse = fit_one("coarse", e)
    dense = fit_one("dense", dense_grid)
    print(
        json.dumps(
            {
                "n_bins": int(len(e)),
                "coarse": coarse,
                "dense_20000": dense,
                "dense_minus_coarse_temperature_k": dense["temperature_k"] - coarse["temperature_k"],
                "dense_ssr_change_percent": 100.0 * (dense["ssr"] - coarse["ssr"]) / coarse["ssr"],
                "objective_caveat": (
                    "Gaussian transmission WLS, not the production raw-count Poisson KL objective; "
                    "used only to discriminate grid sensitivity with all configured bins."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
