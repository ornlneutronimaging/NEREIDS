"""Read-only replay of the archived cached IC+JENDL-5 whole-region fit.

Inputs are the archived region-counts NPZ and IC calibration JSON.  This does
not rebuild either cache, access raw TIFF/NeXus data, or write analysis output.
"""

import json
from pathlib import Path

import numpy as np
import nereids

from archive_inputs import archive_root

ROOT = archive_root()
COUNTS = ROOT / "01_spectral_lineshape_bias/data/region_counts.npz"
CALIB = ROOT / "notebooks/calib_IC_jendl5_24685.json"

FLIGHT_PATH_M = 25.0
INITIAL_DENSITY = 7.0374e-4
FIT_WINDOW_EV = (8.0, 45.0)
ENERGY_RANGE_EV = (4.0, 120.0)


def main() -> None:
    data = np.load(COUNTS, allow_pickle=False)
    calibration = json.loads(CALIB.read_text())

    energies = data["energies"]
    keep = (energies >= ENERGY_RANGE_EV[0]) & (energies <= ENERGY_RANGE_EV[1])

    energy_law = nereids.EnergyLaw
    ic = nereids.IkedaCarpenter(
        alpha=energy_law.sqrt_e(calibration["ic_a0"], calibration["ic_a1"]),
        beta=calibration["ic_beta"],
        r=energy_law.const(calibration["ic_r"]),
        e_min_ev=ENERGY_RANGE_EV[0],
        e_max_ev=ENERGY_RANGE_EV[1],
        flight_path_m=FLIGHT_PATH_M,
        channel_fwhm_us=calibration["ic_psr"],
    )
    resolution = ic.as_tabulated()
    calibrated_energies = np.asarray(
        nereids.tof_to_energy_centers(
            data["edges"],
            FLIGHT_PATH_M * calibration["l_scale"],
            calibration["t0_us"],
        )
    )[keep]

    ta181 = nereids.load_endf(73, 181, library="jendl5")
    fit = nereids.fit_counts_spectrum_typed(
        data["counts_sample"][keep],
        data["counts_ob"][keep],
        calibrated_energies,
        [(ta181, INITIAL_DENSITY)],
        temperature_k=1000.0,
        fit_temperature=True,
        c=float(data["pc_sample"] / data["pc_ob"]),
        baseline=True,
        resolution=resolution,
        fit_energy_range=FIT_WINDOW_EV,
        max_iter=500,
        scale_by_chi2=True,
    )

    print(f"n_resonances={ta181.n_resonances}")
    print(f"bins_total={len(energies)} bins_4_120={int(keep.sum())}")
    print(f"t0_us={calibration['t0_us']:.15g}")
    print(f"l_scale={calibration['l_scale']:.15g}")
    print(f"L_eff_m={FLIGHT_PATH_M * calibration['l_scale']:.15g}")
    print(f"converged={fit.converged} iterations={fit.iterations}")
    print(f"temperature_K={float(fit.temperature_k):.15g}")
    print(f"temperature_unc_K={float(fit.temperature_k_unc):.15g}")
    density = float(np.asarray(fit.densities)[0])
    print(f"density_atoms_per_barn={density:.15g}")
    print(f"density_nominal_fraction={density / INITIAL_DENSITY:.15g}")
    print(f"deviance_per_dof={float(fit.deviance_per_dof):.15g}")
    print(f"baseline={np.asarray(fit.baseline).tolist()}")


if __name__ == "__main__":
    main()
