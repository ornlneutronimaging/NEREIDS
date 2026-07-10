"""Compare inherited fit scopes using the same cached 4–120 eV IC table."""

from __future__ import annotations

import argparse
import json

import numpy as np
import nereids

from archive_inputs import archive_root

ARCHIVE = archive_root()
COUNTS = ARCHIVE / "01_spectral_lineshape_bias/data/region_counts.npz"
CACHE = ARCHIVE / "notebooks/calib_IC_jendl5_24685.json"
FLIGHT_PATH_M = 25.0
INITIAL_DENSITY = 7.0374e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-raw",
        action="store_true",
        help="activate all finite bins in region_counts.npz, including >120 eV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(COUNTS, allow_pickle=False)
    c = json.loads(CACHE.read_text())
    scoped = (d["energies"] >= 4.0) & (d["energies"] <= 120.0)
    if args.all_raw:
        keep = np.ones_like(d["energies"], dtype=bool)
        raw_selection: list[float] | str = "all"
    else:
        keep = scoped
        raw_selection = [4.0, 120.0]
    energy = np.asarray(
        nereids.tof_to_energy_centers(
            d["edges"], FLIGHT_PATH_M * float(c["l_scale"]), float(c["t0_us"])
        )
    )[keep]
    ic = nereids.IkedaCarpenter(
        flight_path_m=FLIGHT_PATH_M,
        e_min_ev=4.0,
        e_max_ev=120.0,
        alpha=nereids.EnergyLaw.sqrt_e(float(c["ic_a0"]), float(c["ic_a1"])),
        beta=float(c["ic_beta"]),
        r=nereids.EnergyLaw.const(float(c["ic_r"])),
        channel_fwhm_us=float(c["ic_psr"]),
    )
    ta181 = nereids.load_endf(73, 181, library="jendl5")
    fit = nereids.fit_counts_spectrum_typed(
        d["counts_sample"][keep],
        d["counts_ob"][keep],
        energy,
        [(ta181, INITIAL_DENSITY)],
        temperature_k=1000.0,
        fit_temperature=True,
        c=float(d["pc_sample"] / d["pc_ob"]),
        baseline=True,
        resolution=ic.as_tabulated(),
        max_iter=500,
        scale_by_chi2=True,
    )
    print(
        json.dumps(
            {
                "n_input_bins": int(len(energy)),
                "n_active_bins": int(len(energy)),
                "raw_selection_ev": raw_selection,
                "raw_energy_span_ev": [
                    float(d["energies"][keep].min()),
                    float(d["energies"][keep].max()),
                ],
                "corrected_energy_span_ev": [float(energy.min()), float(energy.max())],
                "resolution_synthesis_span_ev": [
                    4.0,
                    120.0,
                ],
                "resolution_out_of_range_bins": int(
                    np.count_nonzero((energy < 4.0) | (energy > 120.0))
                ),
                "resolution_out_of_range_behavior": "tabulated endpoint clamp",
                "converged": bool(fit.converged),
                "iterations": int(fit.iterations),
                "temperature_k": float(fit.temperature_k),
                "temperature_unc_k": float(fit.temperature_k_unc),
                "density_atoms_per_barn": float(np.asarray(fit.densities)[0]),
                "deviance_per_dof": float(fit.deviance_per_dof),
                "baseline": np.asarray(fit.baseline).tolist(),
                "baseline_e_ref_ev": float(fit.baseline_e_ref_ev),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
