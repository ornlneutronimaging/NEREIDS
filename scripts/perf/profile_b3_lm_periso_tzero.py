"""Profile driver: spatial B.3 LM+per-iso+TZERO on a 4x4 crop.

Baseline (from #459): per-pixel LM+per-iso+TZERO on 64x64 is
15 584 s / 64^2 ≈ 3.8 s / pixel at 20.6 % convergence.  Non-
converged pixels run the full max-iter budget.  4×4 crop at
max_iter=200 is typically 40–60 s wall, which is a solid sample
window.

Signals readiness via /tmp/b3_ready (consumed by run_sample_b3.sh).
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
    # mask in the per-pixel LM cost path (#514).  Spatial cropping in
    # (y, x) is unchanged.
    E = np.ascontiguousarray(E_full)
    S3d = np.ascontiguousarray(
        S3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    O3d = np.ascontiguousarray(
        O3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    c = Q_s / Q_ob

    T3d = S3d / np.maximum(c * O3d, 1.0)
    sig3d = T3d * np.sqrt(
        1.0 / np.maximum(S3d, 1.0) + 1.0 / np.maximum(O3d, 1.0)
    )

    # Per-iso: 6 natural-abundance Hf isotopes as separate fit params.
    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    members = hf_group.members  # list[((z, a), ratio)]
    resonance_data_list = list(hf_group.resonance_data)  # 6 ResonanceData
    initial_densities = [INIT_DENSITY * ratio for _, ratio in members]

    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)
    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)
    input_data = nereids.from_transmission(T3d, sig3d)

    # Warm
    _ = nereids.spatial_map_typed(
        data=input_data,
        energies=E,
        isotopes=resonance_data_list,
        solver="lm",
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

    Path("/tmp/b3_ready").write_text("ready\n")

    t0 = time.time()
    r = nereids.spatial_map_typed(
        data=input_data,
        energies=E,
        isotopes=resonance_data_list,
        solver="lm",
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
        f"B.3 LM+per-iso+TZERO 4x4: wall={wall:.2f}s n_px={n_px} "
        f"converged={int(conv.sum())}/{n_px} max_iter={MAX_ITER}"
    )


if __name__ == "__main__":
    main()
