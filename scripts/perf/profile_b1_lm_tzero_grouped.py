"""Profile driver: spatial B.1 LM+grouped + TZERO on a 2x2 crop.

Baseline for this workload (from issue #459 / `section_B_64x64.json`):
  per-pixel LM+grouped+TZERO on 64x64 = 10 230 s total → ~30 s / converged pixel,
  8.4 % convergence rate.

A 2x2 crop is 4 pixels; even with low max_iter, non-converged pixels burn
close to the max-iter budget.  We use `max_iter=50` to keep the profile
collection under ~2 min while still capturing the LM+TZERO hotspot mix.

Signals readiness via /tmp/b1_ready (consumed by run_sample_b1.sh).
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
T0_INIT_US = 0.5
L_SCALE_INIT = 1.005
INIT_DENSITY = 1.6e-4
MAX_ITER = 200
CROP_Y0, CROP_Y1 = 254, 256
CROP_X0, CROP_X1 = 254, 256


def main() -> None:
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    mask = (E_full >= ENERGY_MIN) & (E_full <= ENERGY_MAX)
    E = np.ascontiguousarray(E_full[mask])
    S3d = np.ascontiguousarray(
        S3d_raw[mask][:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    O3d = np.ascontiguousarray(
        O3d_raw[mask][:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    c = Q_s / Q_ob

    # Transmission + Gaussian-Poisson sigma per pixel.
    T3d = S3d / np.maximum(c * O3d, 1.0)
    sig3d = T3d * np.sqrt(
        1.0 / np.maximum(S3d, 1.0) + 1.0 / np.maximum(O3d, 1.0)
    )

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)

    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)
    input_data = nereids.from_transmission(T3d, sig3d)

    # Warm
    _ = nereids.spatial_map_typed(
        data=input_data,
        energies=E,
        solver="lm",
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[INIT_DENSITY],
        max_iter=2,
        background=True,
        fit_energy_scale=True,
        t0_init_us=T0_INIT_US,
        l_scale_init=L_SCALE_INIT,
        energy_scale_flight_path_m=FLIGHT_PATH_M,
        resolution=res,
        dead_pixels=dead_pixels,
    )

    Path("/tmp/b1_ready").write_text("ready\n")

    t0 = time.time()
    r = nereids.spatial_map_typed(
        data=input_data,
        energies=E,
        solver="lm",
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[INIT_DENSITY],
        max_iter=MAX_ITER,
        background=True,
        fit_energy_scale=True,
        t0_init_us=T0_INIT_US,
        l_scale_init=L_SCALE_INIT,
        energy_scale_flight_path_m=FLIGHT_PATH_M,
        resolution=res,
        dead_pixels=dead_pixels,
    )
    wall = time.time() - t0
    n_px = (CROP_Y1 - CROP_Y0) * (CROP_X1 - CROP_X0)
    conv = np.asarray(r.converged_map)
    print(
        f"B.1 LM+grouped+TZERO 2x2: wall={wall:.2f}s n_px={n_px} "
        f"converged={int(conv.sum())}/{n_px} max_iter={MAX_ITER}"
    )


if __name__ == "__main__":
    main()
