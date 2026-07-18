#!/usr/bin/env python3
"""Run the supported VENUS room-calibration then frozen-hot workflow.

The input NPZ follows the documented VENUS aggregate export keys.  This file
contains input loading and plotting only; response construction, source
inference, background fitting, room calibration, curve assembly, and the hot
fit are all library calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nereids


REFERENCE_T0_US = 1.0212486261382638
REFERENCE_PATH_M = 25.0 * 1.0008178597256119

# The starting point came from the earlier VENUS UDR calibration.  It is only
# an optimizer start; the room data determine the returned calibration.
ROOM_PHYSICAL_START = (0.742117988, 25.0745433, 0.946344901, 0.000826583693)
ROOM_BACKGROUND_START = (0.826764552, 0.157084784, -0.134339595, 0.00128658541)
RESPONSE_ENERGY_RANGE_EV = (4.0, 120.0)
FIT_ENERGY_RANGE_EV = (8.0, 45.0)

# These three factors were measured from the raw TPX1 pixel covariance for
# these exact open, room, and hot runs. They account for correlated detector
# hits and were fixed before the spectrum fit; they are not residual-tuning
# parameters. Use 1.0 for independent Poisson counts or supply factors measured
# independently for a different ROI or acquisition.
OPEN_VARIANCE_FACTOR = 3.819301921170809
ROOM_VARIANCE_FACTOR = 3.5796759175
HOT_VARIANCE_FACTOR = 3.7197893500316446


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-counts", type=Path, required=True)
    parser.add_argument("--jendl-ta181", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    factor_help = (
        "independently measured variance factor for this acquisition; "
        "use 1.0 for independent Poisson counts"
    )
    parser.add_argument(
        "--open-variance-factor", type=float, default=OPEN_VARIANCE_FACTOR, help=factor_help
    )
    parser.add_argument(
        "--room-variance-factor", type=float, default=ROOM_VARIANCE_FACTOR, help=factor_help
    )
    parser.add_argument(
        "--hot-variance-factor", type=float, default=HOT_VARIANCE_FACTOR, help=factor_help
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def report(stage: str, values: dict[str, float]) -> None:
        summary = " ".join(f"{name}={value:.7g}" for name, value in values.items())
        print(f"[{stage}] {summary}", flush=True)

    with np.load(args.region_counts, allow_pickle=False) as data:
        selected_edges, _, selected = nereids.select_energy_ordered_detector_bins(
            data["edges"],
            data["energies"],
            (data["counts_ob"], data["counts_calib"], data["counts_sample"]),
            RESPONSE_ENERGY_RANGE_EV,
        )
        open_counts, room_counts, hot_counts = selected
        room_exposure = float(data["pc_calib"] / data["pc_ob"])
        hot_exposure = float(data["pc_sample"] / data["pc_ob"])
        room_temperature_k = float(data["T_calib_k"])

    isotope = nereids.load_endf_file(str(args.jendl_ta181))
    calibration = nereids.calibrate_aggregated_1d(
        detector_time_edges_us=selected_edges,
        open_counts=open_counts,
        room_counts=room_counts,
        sample_over_open_exposure=room_exposure,
        isotopes=[isotope],
        room_temperature_k=room_temperature_k,
        reference_flight_path_m=REFERENCE_PATH_M,
        reference_timing_offset_us=REFERENCE_T0_US,
        initial_physical_parameters=ROOM_PHYSICAL_START,
        fit_energy_range_ev=FIT_ENERGY_RANGE_EV,
        ic_profile=nereids.VENUS_UDR_MATCHED_IC_PROFILE,
        physical_lower_bounds=(-5.0, 24.5, 0.2, 0.0),
        physical_upper_bounds=(5.0, 25.5, 5.0, 0.01),
        initial_background_parameters=ROOM_BACKGROUND_START,
        debye_temperature_k=217.0,
        physical_scale=(1.0, 0.05, 0.5, 0.00070374),
        open_variance_factor=args.open_variance_factor,
        room_variance_factor=args.room_variance_factor,
        progress=report,
    )
    hot = nereids.fit_frozen_aggregated_1d(
        calibration,
        hot_counts=hot_counts,
        sample_over_open_exposure=hot_exposure,
        initial_temperature_k=1000.0,
        initial_density_atoms_per_barn=0.00070374,
        hot_variance_factor=args.hot_variance_factor,
        progress=report,
    )

    def metrics(fit: nereids.Aggregated1DFitResult) -> dict[str, object]:
        active = fit.fit_mask
        return {
            "parameters": fit.parameters,
            "converged": fit.converged,
            "bound_hits": list(fit.bound_hits),
            "fit_energy_range_ev": list(fit.fit_energy_range_ev),
            "fit_bins": int(np.count_nonzero(active)),
            "supplied_bins": int(active.size),
            "fit_max_abs_residual": fit.max_abs_residual,
            "fit_rms_residual": fit.rms_residual,
            "fit_poisson_max_abs_residual": float(
                np.max(np.abs(fit.poisson_residual[active]))
            ),
            "fit_poisson_rms_residual": float(
                np.sqrt(np.mean(fit.poisson_residual[active] ** 2))
            ),
            "all_supplied_max_abs_residual": float(np.max(np.abs(fit.residual))),
            "all_supplied_rms_residual": float(
                np.sqrt(np.mean(fit.residual**2))
            ),
            "all_supplied_poisson_max_abs_residual": float(
                np.max(np.abs(fit.poisson_residual))
            ),
            "all_supplied_poisson_rms_residual": float(
                np.sqrt(np.mean(fit.poisson_residual**2))
            ),
            "fit_bins_above_five": fit.bins_above_five,
        }

    payload = {
        "provenance": {
            "response_energy_range_ev": list(RESPONSE_ENERGY_RANGE_EV),
            "fit_energy_range_ev": list(FIT_ENERGY_RANGE_EV),
            "fit_range_origin": (
                "pre-existing VENUS tantalum calibration region fixed before the "
                "IC comparison; all supplied response-support bins are also reported"
            ),
            "variance_factors": {
                "open": calibration.fit.open_variance_factor,
                "room": calibration.fit.sample_variance_factor,
                "hot": hot.sample_variance_factor,
                "origin": (
                    "measured from raw TPX1 pixel correlations for these exact "
                    "acquisitions before the spectrum fit"
                ),
            },
        },
        "room": metrics(calibration.fit),
        "hot": metrics(hot),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        constrained_layout=True,
    )
    fits = (("Room calibration", calibration.fit), ("Frozen hot fit", hot))
    for column, (label, fit) in enumerate(fits):
        active = fit.fit_mask
        inactive = ~active
        axes[0, column].scatter(
            fit.energy_ev[inactive],
            fit.measured_transmission[inactive],
            s=5,
            alpha=0.2,
            color="gray",
            label="supplied, diagnostic only",
        )
        axes[0, column].scatter(
            fit.energy_ev[active],
            fit.measured_transmission[active],
            s=7,
            alpha=0.65,
            color="black",
            label="measured in fit window",
        )
        axes[0, column].plot(
            fit.energy_ev,
            fit.model_prediction,
            color="tab:red",
            linewidth=1.4,
            label="NEREIDS",
        )
        axes[0, column].set_title(
            f"{label}, {fit.fit_energy_range_ev[0]:g}–{fit.fit_energy_range_ev[1]:g} eV fit: "
            f"max |residual|={fit.max_abs_residual:.3f}, RMS={fit.rms_residual:.3f}"
        )
        axes[0, column].set_ylabel("Transmission")
        axes[0, column].legend()
        axes[1, column].scatter(
            fit.energy_ev[inactive],
            fit.poisson_residual[inactive],
            s=5,
            alpha=0.12,
            color="gray",
            label="outside fit window",
        )
        axes[1, column].scatter(
            fit.energy_ev[active],
            fit.poisson_residual[active],
            s=7,
            alpha=0.35,
            color="tab:orange",
            label="independent-Poisson",
        )
        axes[1, column].scatter(
            fit.energy_ev[active],
            fit.residual[active],
            s=7,
            alpha=0.7,
            color="tab:blue",
            label="measured TPX1 covariance",
        )
        axes[1, column].axhline(0.0, color="black", linewidth=1.0)
        axes[1, column].axhline(5.0, color="gray", linestyle="--", linewidth=1.0)
        axes[1, column].axhline(-5.0, color="gray", linestyle="--", linewidth=1.0)
        axes[1, column].set_xlabel("Energy (eV)")
        axes[1, column].set_ylabel("Residual")
        axes[1, column].legend()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
