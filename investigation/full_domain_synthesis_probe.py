"""Test whether the cached IC law can be synthesized over every archive bin."""

from __future__ import annotations

import json

import numpy as np
import nereids

from archive_inputs import archive_root


ROOT = archive_root()
COUNTS = ROOT / "01_spectral_lineshape_bias/data/region_counts.npz"
CALIBRATION = ROOT / "notebooks/calib_IC_jendl5_24685.json"
FLIGHT_PATH_M = 25.0


def main() -> None:
    data = np.load(COUNTS, allow_pickle=False)
    calibration = json.loads(CALIBRATION.read_text())
    corrected = np.asarray(
        nereids.tof_to_energy_centers(
            data["edges"],
            FLIGHT_PATH_M * float(calibration["l_scale"]),
            float(calibration["t0_us"]),
        )
    )
    result = {
        "n_bins": int(len(corrected)),
        "raw_energy_span_ev": [
            float(data["energies"].min()),
            float(data["energies"].max()),
        ],
        "corrected_energy_span_ev": [float(corrected.min()), float(corrected.max())],
        "bins_raw_above_120_ev": int(np.count_nonzero(data["energies"] > 120.0)),
    }
    try:
        nereids.IkedaCarpenter(
            flight_path_m=FLIGHT_PATH_M,
            e_min_ev=float(corrected.min()),
            e_max_ev=float(corrected.max()),
            alpha=nereids.EnergyLaw.sqrt_e(
                float(calibration["ic_a0"]), float(calibration["ic_a1"])
            ),
            beta=float(calibration["ic_beta"]),
            r=nereids.EnergyLaw.const(float(calibration["ic_r"])),
            channel_fwhm_us=float(calibration["ic_psr"]),
        )
    except ValueError as error:
        result["full_domain_synthesis"] = "error"
        result["error"] = str(error)
    else:
        result["full_domain_synthesis"] = "ok"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
