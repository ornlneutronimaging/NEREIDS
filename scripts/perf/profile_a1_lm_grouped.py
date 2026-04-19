"""Profile driver: Section A.1 LM+grouped aggregated fit.

Runs the baseline LM+grouped fit on aggregated VENUS Hf 120min data.
Meant to be launched under samply:

    samply record --save-only -o /tmp/a1.profile.json -- \
        pixi run python scripts/perf/profile_a1_lm_grouped.py

Baseline wall time on main: ~5.4s / 33 iterations / converged=False.
The forward-model path is exercised 33 * (1 + 2 * (n_free=7)) ≈ 495 times.
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
MAX_ITER = 500
INIT_DENSITY = 1.6e-4


def main() -> None:
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    mask = (E_full >= ENERGY_MIN) & (E_full <= ENERGY_MAX)
    E = np.ascontiguousarray(E_full[mask])
    S3d = np.ascontiguousarray(S3d_raw[mask]).astype(np.float64)
    O3d = np.ascontiguousarray(O3d_raw[mask]).astype(np.float64)
    c = Q_s / Q_ob
    S_agg = S3d.sum(axis=(1, 2))
    O_agg = O3d.sum(axis=(1, 2))
    T_agg = S_agg / np.maximum(c * O_agg, 1.0)
    sigT_agg = T_agg * np.sqrt(1.0 / np.maximum(S_agg, 1.0) + 1.0 / np.maximum(O_agg, 1.0))

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()

    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)

    # Signal the profiler that load phase is done.
    Path("/tmp/a1_ready").write_text("ready\n")

    # Warm: first call triggers ENDF parse + resolution kernel cache init.
    _ = nereids.fit_spectrum_typed(
        transmission=np.ascontiguousarray(T_agg),
        uncertainty=np.ascontiguousarray(sigT_agg),
        energies=E,
        solver="lm",
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[INIT_DENSITY],
        max_iter=5,
        background=True,
        fit_energy_scale=True,
        t0_init_us=T0_INIT_US,
        l_scale_init=L_SCALE_INIT,
        energy_scale_flight_path_m=FLIGHT_PATH_M,
        resolution=res,
    )

    # Measured: the real workload.
    t0 = time.time()
    r = nereids.fit_spectrum_typed(
        transmission=np.ascontiguousarray(T_agg),
        uncertainty=np.ascontiguousarray(sigT_agg),
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
    )
    wall = time.time() - t0
    print(
        f"A.1 LM+grouped: wall={wall:.2f}s iter={r.iterations} "
        f"converged={r.converged} chi2r={r.reduced_chi_squared:.3f} "
        f"n={r.densities[0]:.4e} t0={r.t0_us:.4f} L={r.l_scale:.6f}"
    )


if __name__ == "__main__":
    main()
