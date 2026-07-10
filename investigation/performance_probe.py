"""Component timing probe for the current IC calibration forward path.

Prints JSON and performs no writes. Timings are medians after warm-up.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import nereids

from archive_inputs import archive_root

ARCHIVE = archive_root()
COUNTS = ARCHIVE / "01_spectral_lineshape_bias/data/region_counts.npz"
CACHE = ARCHIVE / "notebooks/calib_IC_jendl5_24685.json"
FLIGHT_PATH_M = 25.0
INITIAL_DENSITY = 7.0374e-4


def median_time(fn, repeats: int) -> tuple[float, object]:
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), result


def main() -> None:
    d = np.load(COUNTS, allow_pickle=False)
    cache = json.loads(CACHE.read_text())
    keep = (d["energies"] >= 4.0) & (d["energies"] <= 120.0)
    energy = np.asarray(
        nereids.tof_to_energy_centers(
            d["edges"], FLIGHT_PATH_M * float(cache["l_scale"]), float(cache["t0_us"])
        )
    )[keep]
    win = (energy > 8.0) & (energy < 45.0)
    energy = energy[win]
    ta181 = nereids.load_endf(73, 181, library="jendl5")

    def make_ic(n_energies: int, n_tau: int):
        return nereids.IkedaCarpenter(
            flight_path_m=FLIGHT_PATH_M,
            e_min_ev=0.5 * float(energy.min()),
            e_max_ev=2.0 * float(energy.max()),
            alpha=nereids.EnergyLaw.sqrt_e(float(cache["ic_a0"]), float(cache["ic_a1"])),
            beta=float(cache["ic_beta"]),
            r=nereids.EnergyLaw.const(float(cache["ic_r"])),
            n_energies=n_energies,
            n_tau=n_tau,
            channel_fwhm_us=float(cache["ic_psr"]),
        )

    def forward(resolution=None):
        return np.asarray(
            nereids.forward_model(
                energy,
                [(ta181, INITIAL_DENSITY)],
                temperature_k=float(d["T_calib_k"]),
                resolution=resolution,
            )
        )

    # Warm every path once before timing.
    fixed = make_ic(48, 400).as_tabulated()
    nores = forward(None)
    reused = forward(fixed)

    t_synth_48, _ = median_time(lambda: make_ic(48, 400), 7)
    t_synth_64, _ = median_time(lambda: make_ic(64, 500), 7)
    t_forward_nores, model_nores = median_time(lambda: forward(None), 7)
    t_forward_reused, model_reused = median_time(lambda: forward(fixed), 7)

    def composite():
        return forward(make_ic(48, 400).as_tabulated())

    t_composite, model_composite = median_time(composite, 7)
    result = {
        "n_bins": int(len(energy)),
        "synthesis_48x400_seconds": t_synth_48,
        "synthesis_64x500_seconds": t_synth_64,
        "forward_no_resolution_seconds": t_forward_nores,
        "forward_reused_resolution_seconds": t_forward_reused,
        "fresh_ic_plus_forward_seconds": t_composite,
        "composite_minus_synthesis_seconds": t_composite - t_synth_48,
        "fresh_vs_reused_max_abs": float(np.max(np.abs(model_composite - model_reused))),
        "warmup_nores_repeat_max_abs": float(np.max(np.abs(nores - model_nores))),
        "warmup_reused_repeat_max_abs": float(np.max(np.abs(reused - model_reused))),
        "archive_map_udr_seconds": 99,
        "archive_map_ic_seconds": 602,
        "archive_map_ic_over_udr": 602.0 / 99.0,
        "archive_ic_calibration_seconds_range": [665, 1490],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
