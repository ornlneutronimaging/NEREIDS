"""Profile driver: spatial KL+per-iso+TZERO on a 4x4 VENUS Hf crop.

This profile verifies whether the FD t0/L_scale cost (that motivates
#459 C1) is a TRUE hot path on the KL path too, given that
`joint_poisson` calls `analytical_jacobian` TWICE per iteration
(gradient + Fisher info) — 8 FD broaden_presorted per iter vs LM's
4/iter.

Wall per call is on the order of a few seconds on the 4×4 crop
(~3.2 s measured in the PR #469 baseline), so `N_REPEATS = 3`
yields an aggregate sampling window of roughly 9–10 s — enough
for `/usr/bin/sample`'s 15 s default window to collect a solid
profile without blowing up wall time for the driver.
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
CROP_Y0, CROP_Y1 = 254, 258
CROP_X0, CROP_X1 = 254, 258


def main() -> None:
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    # SAMMY REGION-equivalent: full nominal axis + `fit_energy_range`
    # mask in the per-pixel KL cost path (#514).  TZERO is FITTED here,
    # so the axis stays nominal.
    E = np.ascontiguousarray(E_full)
    S3d = np.ascontiguousarray(
        S3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    O3d = np.ascontiguousarray(
        O3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    c = Q_s / Q_ob

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    members = hf_group.members
    resonance_data_list = list(hf_group.resonance_data)
    initial_densities = [INIT_DENSITY * ratio for _, ratio in members]

    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)
    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)
    input_data = nereids.from_counts(S3d, O3d)

    # Warm
    _ = nereids.spatial_map_typed(
        data=input_data,
        energies=E,
        isotopes=resonance_data_list,
        solver="kl",
        c=c,
        temperature_k=TEMP_K,
        initial_densities=initial_densities,
        max_iter=2,
        background=True,
        fit_energy_scale=True,
        t0_init_us=T0_INIT_US,
        l_scale_init=L_SCALE_INIT,
        energy_scale_flight_path_m=FLIGHT_PATH_M,
        resolution=res,
        dead_pixels=dead_pixels,
        fit_energy_range=(ENERGY_MIN, ENERGY_MAX),
    )

    Path("/tmp/kl_periso_tzero_ready").write_text("ready\n")

    # Repeat for 5 s of active work to fill the sample window.
    N_REPEATS = 3
    t0 = time.time()
    for _ in range(N_REPEATS):
        r = nereids.spatial_map_typed(
            data=input_data,
            energies=E,
            isotopes=resonance_data_list,
            solver="kl",
            c=c,
            temperature_k=TEMP_K,
            initial_densities=initial_densities,
            max_iter=MAX_ITER,
            background=True,
            fit_energy_scale=True,
            t0_init_us=T0_INIT_US,
            l_scale_init=L_SCALE_INIT,
            energy_scale_flight_path_m=FLIGHT_PATH_M,
            resolution=res,
            dead_pixels=dead_pixels,
            fit_energy_range=(ENERGY_MIN, ENERGY_MAX),
        )
    wall = time.time() - t0
    n_px = (CROP_Y1 - CROP_Y0) * (CROP_X1 - CROP_X0)
    conv = np.asarray(r.converged_map)
    print(
        f"KL+per-iso+TZERO 4x4 (x{N_REPEATS}): wall={wall:.2f}s per-call={wall/N_REPEATS:.2f}s "
        f"n_px={n_px} converged={int(conv.sum())}/{n_px}"
    )


if __name__ == "__main__":
    main()
