"""Profile driver: spatial B.2 KL+grouped on a 4x4 crop.

Signals readiness via /tmp/b2_ready (consumed by run_sample_b2.sh).
Baseline: ~0.5s/pixel on 64x64. A 4x4 crop runs in ~8s.
"""
from __future__ import annotations

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
CROP_Y0, CROP_Y1 = 252, 256
CROP_X0, CROP_X1 = 252, 256
INIT_DENSITY = 1.6e-4
MAX_ITER = 200


def main() -> None:
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    mask = (E_full >= ENERGY_MIN) & (E_full <= ENERGY_MAX)
    E = np.ascontiguousarray(E_full[mask])
    S3d = np.ascontiguousarray(S3d_raw[mask][:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]).astype(np.float64)
    O3d = np.ascontiguousarray(O3d_raw[mask][:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]).astype(np.float64)
    c = Q_s / Q_ob

    # Pre-calibrated TZERO (from baseline A.2) applied once to the data grid.
    # tof_factor_us_sqrtEv * L / sqrt(E)  → TOF, then correct with t0, L_scale
    # Use hard-coded calibration values from section_A_aggregated.json A2_KL_grouped.
    T0_US = 0.4809277146701285
    L_SCALE = 1.0052452911520162
    tof_factor = (0.5 * 1.67492749804e-27 / 1.602176634e-19) ** 0.5 * 1.0e6
    L_eff = FLIGHT_PATH_M * L_SCALE
    tof = tof_factor * FLIGHT_PATH_M / np.sqrt(E)
    tof_corr = tof - T0_US
    E_cal = (tof_factor * L_eff / tof_corr) ** 2
    E_cal = np.ascontiguousarray(E_cal)

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)

    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)

    input_data = nereids.from_counts(S3d, O3d)

    # Warm
    _ = nereids.spatial_map_typed(
        data=input_data,
        energies=E_cal,
        solver="kl",
        c=c,
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[INIT_DENSITY],
        max_iter=5,
        background=True,
        resolution=res,
        dead_pixels=dead_pixels,
    )

    Path("/tmp/b2_ready").write_text("ready\n")

    # Post-PR #468 the 4x4 B.2 fit runs in ~0.2 s, which is shorter
    # than the `/usr/bin/sample` startup handshake.  Repeat the fit
    # enough times to cover a full 5–6 s sampling window so the
    # profile isn't empty.  The per-run bit-exactness is preserved
    # (same inputs every call).
    N_REPEATS = 25
    t0 = time.time()
    for _ in range(N_REPEATS):
        r = nereids.spatial_map_typed(
            data=input_data,
            energies=E_cal,
            solver="kl",
            c=c,
            temperature_k=TEMP_K,
            groups=[hf_group],
            initial_densities=[INIT_DENSITY],
            max_iter=MAX_ITER,
            background=True,
            resolution=res,
            dead_pixels=dead_pixels,
        )
    wall = time.time() - t0
    n_px = (CROP_Y1 - CROP_Y0) * (CROP_X1 - CROP_X0)
    conv = np.asarray(r.converged_map)
    dens = np.asarray(r.density_maps[0])
    print(
        f"B.2 KL 4x4 (x{N_REPEATS}): wall={wall:.2f}s per-call={wall/N_REPEATS:.3f}s "
        f"n_px={n_px} converged={int(conv.sum())}/{n_px} "
        f"median_density={np.nanmedian(dens):.4e}"
    )


if __name__ == "__main__":
    main()
