"""Quantify L-scale and tau-grid effects at the archived fitted IC parameters."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import nereids

from archive_inputs import archive_root

ARCHIVE = archive_root()
COUNTS = ARCHIVE / "01_spectral_lineshape_bias/data/region_counts.npz"
CACHE = ARCHIVE / "notebooks/calib_IC_jendl5_24685.json"
L_NOM = 25.0
T_FIT = 1058.69727434866
DENSITY = 0.000632508761631833
BASELINE = np.array([0.9365930840171124, 0.0016145043811995664, 0.00689211751303141])
FEATURES_EV = (10.35, 13.92, 23.95, 35.17, 39.15)


def main() -> None:
    d = np.load(COUNTS, allow_pickle=False)
    c = json.loads(CACHE.read_text())
    keep = (d["energies"] >= 4.0) & (d["energies"] <= 120.0)
    energy_all = np.asarray(
        nereids.tof_to_energy_centers(
            d["edges"], L_NOM * float(c["l_scale"]), float(c["t0_us"])
        )
    )[keep]
    win = (energy_all > 8.0) & (energy_all < 45.0)
    energy = energy_all[win]
    sample = d["counts_sample"][keep][win]
    ob = d["counts_ob"][keep][win]
    measured = (sample / ob) * float(d["pc_ob"] / d["pc_sample"])
    sigma = measured * np.sqrt(1.0 / np.maximum(sample, 1.0) + 1.0 / np.maximum(ob, 1.0))
    z = np.log(energy / math.sqrt(float(energy.min() * energy.max())))
    baseline = BASELINE[0] + BASELINE[1] * z + BASELINE[2] * z**2
    ta181 = nereids.load_endf(73, 181, library="jendl5")

    def prediction(flight_path: float, n_tau: int) -> np.ndarray:
        ic = nereids.IkedaCarpenter(
            flight_path_m=flight_path,
            e_min_ev=4.0,
            e_max_ev=120.0,
            alpha=nereids.EnergyLaw.sqrt_e(float(c["ic_a0"]), float(c["ic_a1"])),
            beta=float(c["ic_beta"]),
            r=nereids.EnergyLaw.const(float(c["ic_r"])),
            n_energies=64,
            n_tau=n_tau,
            channel_fwhm_us=float(c["ic_psr"]),
        )
        raw = np.asarray(
            nereids.forward_model(
                energy,
                [(ta181, DENSITY)],
                temperature_k=T_FIT,
                resolution=ic.as_tabulated(),
            )
        )
        return baseline * raw

    reference = prediction(L_NOM, 600)
    residual_ref = (measured - reference) / sigma
    feature_residuals = {}
    for feature in FEATURES_EV:
        index = int(np.argmin(np.abs(energy - feature)))
        feature_residuals[f"{feature:.2f}"] = {
            "bin_energy_ev": float(energy[index]),
            "residual_sigma": float(residual_ref[index]),
        }
    worst = int(np.argmax(np.abs(residual_ref)))
    cases = {}
    for name, flight_path, n_tau in (
        ("calibration_grid_400", L_NOM, 400),
        ("calibrator_default_500", L_NOM, 500),
        ("science_default_600", L_NOM, 600),
        ("high_tau_1200", L_NOM, 1200),
        ("physical_L_eff_600", L_NOM * float(c["l_scale"]), 600),
    ):
        pred = prediction(flight_path, n_tau)
        delta = (pred - reference) / sigma
        residual = (measured - pred) / sigma
        cases[name] = {
            "flight_path_m": flight_path,
            "n_tau": n_tau,
            "max_abs_transmission_change": float(np.max(np.abs(pred - reference))),
            "rms_transmission_change": float(np.sqrt(np.mean((pred - reference) ** 2))),
            "max_abs_change_sigma": float(np.max(np.abs(delta))),
            "rms_change_sigma": float(np.sqrt(np.mean(delta**2))),
            "residual_ssr": float(residual @ residual),
            "residual_ssr_change_percent": float(
                100.0 * ((residual @ residual) - (residual_ref @ residual_ref))
                / (residual_ref @ residual_ref)
            ),
        }
    print(
        json.dumps(
            {
                "n_bins": int(len(energy)),
                "l_scale": float(c["l_scale"]),
                "reference_residual_ssr": float(residual_ref @ residual_ref),
                "feature_residuals": feature_residuals,
                "maximum_abs_residual": {
                    "bin_energy_ev": float(energy[worst]),
                    "residual_sigma": float(residual_ref[worst]),
                },
                "cases": cases,
                "caveat": "Fitted parameters and KL baseline fixed; coarse observed grid retained.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
