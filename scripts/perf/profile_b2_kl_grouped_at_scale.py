"""Profile driver: B.2 spatial KL+grouped+background at scale (32x32 / 64x64).

Validates whether per-pixel Vec clones and throwaway `InputData::Counts`
dispatch in `spatial.rs` scale enough on production-sized maps to be
worth optimizing (#459 discussion, ChatGPT candidate #4).

On 4x4 (the B.2 gate) per-pixel clone overhead is <1% of wall
(~430 KB alloc traffic per spatial call at 0.17 s wall).  At 32x32
(1024 pixels) alloc traffic scales to ~200 MB; at 64x64 (4096 pixels)
~800 MB.  This driver establishes the wall-time and profile
signatures at those scales so we can gauge whether the optimization
is worth implementing.

Usage:
    pixi run python scripts/perf/profile_b2_kl_grouped_at_scale.py [SIZE]
where SIZE ∈ {8, 16, 32, 64}.  Default 32.

Signals readiness via /tmp/b2_at_scale_ready (consumed by
run_sample_b2_at_scale.sh).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import nereids

ROOT = Path(__file__).resolve().parents[2]
H5 = ROOT / ".research/spatial-regularization/data/counts/resonance_data_2cm.h5"
RES_FILE = ROOT / "_fts_bl10_0p5meV_1keV_25pts.txt"

TEMP_K = 293.6
FLIGHT_PATH_M = 25.0
ENERGY_MIN, ENERGY_MAX = 7.0, 200.0
INIT_DENSITY = 1.6e-4
MAX_ITER = 200

T0_US = 0.4809277146701285
L_SCALE = 1.0052452911520162


def main() -> None:
    crop = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    center = 255
    half = crop // 2
    y0, y1 = center - half, center + half
    x0, x1 = center - half, center + half

    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    # SAMMY REGION-equivalent: keep the FULL spectral axis, apply
    # pre-calibrated TZERO to the whole grid, and let
    # `fit_energy_range` mask the cost function (#514).
    S3d = np.ascontiguousarray(S3d_raw[:, y0:y1, x0:x1]).astype(np.float64)
    O3d = np.ascontiguousarray(O3d_raw[:, y0:y1, x0:x1]).astype(np.float64)
    c = Q_s / Q_ob

    # Pre-calibrated TZERO (A.2 values), same as B.2 gate, applied to
    # the full grid.
    tof_factor = (0.5 * 1.67492749804e-27 / 1.602176634e-19) ** 0.5 * 1.0e6
    L_eff = FLIGHT_PATH_M * L_SCALE
    tof_full = tof_factor * FLIGHT_PATH_M / np.sqrt(E_full)
    tof_corr_full = tof_full - T0_US
    E_cal_full = np.ascontiguousarray((tof_factor * L_eff / tof_corr_full) ** 2)

    def _tzero_cal(e_nominal: float) -> float:
        tof = tof_factor * FLIGHT_PATH_M / np.sqrt(e_nominal)
        return float((tof_factor * L_eff / (tof - T0_US)) ** 2)

    fit_min_cal = _tzero_cal(ENERGY_MIN)
    fit_max_cal = _tzero_cal(ENERGY_MAX)

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)
    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)
    input_data = nereids.from_counts(S3d, O3d)

    # Warm
    _ = nereids.spatial_map_typed(
        data=input_data,
        energies=E_cal_full,
        solver="kl",
        c=c,
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[INIT_DENSITY],
        max_iter=5,
        background=True,
        resolution=res,
        dead_pixels=dead_pixels,
        fit_energy_range=(fit_min_cal, fit_max_cal),
    )

    Path("/tmp/b2_at_scale_ready").write_text("ready\n")

    t0 = time.time()
    r = nereids.spatial_map_typed(
        data=input_data,
        energies=E_cal_full,
        solver="kl",
        c=c,
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[INIT_DENSITY],
        max_iter=MAX_ITER,
        background=True,
        resolution=res,
        dead_pixels=dead_pixels,
        fit_energy_range=(fit_min_cal, fit_max_cal),
    )
    wall = time.time() - t0
    n_px = crop * crop
    conv = np.asarray(r.converged_map)
    dens = np.asarray(r.density_maps[0])
    print(
        f"B.2 KL {crop}x{crop}: wall={wall:.2f}s n_px={n_px} "
        f"converged={int(conv.sum())}/{n_px} "
        f"wall_per_px={wall / n_px * 1000:.2f}ms "
        f"median_density={np.nanmedian(dens):.4e}"
    )


if __name__ == "__main__":
    main()
