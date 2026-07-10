"""A/B the current IC data-grid path against dense-grid convolution.

Uses the already reproduced IC/JENDL-5 sample-fit parameters. Prints JSON and
does not write data.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import nereids

from archive_inputs import archive_root

ARCHIVE = archive_root()
COUNTS = ARCHIVE / "01_spectral_lineshape_bias/data/region_counts.npz"
CACHE = ARCHIVE / "notebooks/calib_IC_jendl5_24685.json"
FLIGHT_PATH_M = 25.0
T_FIT_K = 1058.69727434866
DENSITY = 0.000632508761631833
BASELINE = np.array([0.9365930840171124, 0.0016145043811995664, 0.00689211751303141])
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

    def forward(grid: np.ndarray) -> np.ndarray:
        return np.asarray(
            nereids.forward_model(
                grid,
                [(ta181, DENSITY)],
                temperature_k=T_FIT_K,
                resolution=resolution,
            )
        )

    e_ref = math.sqrt(float(e.min() * e.max()))
    z = np.log(e / e_ref)
    fixed_baseline = BASELINE[0] + BASELINE[1] * z + BASELINE[2] * z**2

    def diagnostics(raw_model: np.ndarray) -> dict[str, object]:
        fixed = fixed_baseline * raw_model
        design = raw_model[:, None] * np.column_stack((np.ones_like(z), z, z**2))
        aw = design / sigma[:, None]
        yw = measured / sigma
        coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
        profiled = design @ coef
        rf = (measured - fixed) / sigma
        rp = (measured - profiled) / sigma
        return {
            "fixed_baseline_rms_sigma": float(np.sqrt(np.mean(rf**2))),
            "fixed_baseline_ssr": float(rf @ rf),
            "profiled_baseline": coef.tolist(),
            "profiled_rms_sigma": float(np.sqrt(np.mean(rp**2))),
            "profiled_ssr": float(rp @ rp),
            "fixed_residual": rf,
            "profiled_residual": rp,
            "fixed_prediction": fixed,
            "profiled_prediction": profiled,
        }

    start = time.perf_counter()
    coarse_raw = forward(e)
    timings: dict[str, float] = {"coarse_seconds": time.perf_counter() - start}
    models: dict[str, np.ndarray] = {"coarse": coarse_raw}
    for n in (10_000, 20_000, 40_000):
        grid = np.unique(np.concatenate((np.geomspace(4.0, 120.0, n), e)))
        start = time.perf_counter()
        dense = forward(grid)
        timings[f"dense_{n}_seconds"] = time.perf_counter() - start
        models[str(n)] = np.interp(e, grid, dense)

    diag = {name: diagnostics(model) for name, model in models.items()}
    finest = diag["40000"]
    feature_rows = []
    for e0 in FEATURES_EV:
        mask = np.abs(e - e0) <= 0.25
        idxs = np.flatnonzero(mask)
        local = idxs[np.argmax(np.abs(diag["coarse"]["fixed_residual"][idxs]))]
        feature_rows.append(
            {
                "feature_ev": e0,
                "observed_bin_ev": float(e[local]),
                "coarse_fixed_residual_sigma": float(diag["coarse"]["fixed_residual"][local]),
                "dense_fixed_residual_sigma": float(finest["fixed_residual"][local]),
                "dense_minus_coarse_prediction_sigma": float(
                    (finest["fixed_prediction"][local] - diag["coarse"]["fixed_prediction"][local])
                    / sigma[local]
                ),
            }
        )

    public_diag = {}
    for name, values in diag.items():
        public_diag[name] = {
            k: v
            for k, v in values.items()
            if k not in {
                "fixed_residual",
                "profiled_residual",
                "fixed_prediction",
                "profiled_prediction",
            }
        }
    result = {
        "n_observed_bins": int(len(e)),
        "timings": timings,
        "diagnostics": public_diag,
        "coarse_vs_40000_max_abs_raw_transmission": float(
            np.max(np.abs(models["coarse"] - models["40000"]))
        ),
        "coarse_vs_40000_rms_raw_transmission": float(
            np.sqrt(np.mean((models["coarse"] - models["40000"]) ** 2))
        ),
        "20000_vs_40000_max_abs_raw_transmission": float(
            np.max(np.abs(models["20000"] - models["40000"]))
        ),
        "20000_vs_40000_rms_raw_transmission": float(
            np.sqrt(np.mean((models["20000"] - models["40000"]) ** 2))
        ),
        "feature_rows": feature_rows,
        "caveat": "T, density, and fixed KL-fit baseline were not reoptimized for the dense model.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
