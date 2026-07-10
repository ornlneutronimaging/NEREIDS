"""Measure spatial open-beam variation in the real VENUS HDF5 fixture.

The production ``InputData3D::Counts`` route replaces every pixel's open-beam
spectrum with the spatial mean. This read-only diagnostic measures the
uniformity assumption over the archived 8–45 eV fit window. It reports all
pixels and, separately, the mechanically defined nonzero-total set; it does not
mask or write data.
"""

from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPEN_BEAM = ROOT / "tests/data/pleiades_data/venus_hf_open_beam.h5"
FLIGHT_PATH_M = 25.0
TOF_FACTOR_US_SQRT_EV_PER_M = 72.2977
ENERGY_WINDOW_EV = (8.0, 45.0)
SPATIAL_BLOCK = 32


def summarize(name: str, values: np.ndarray) -> None:
    q = np.quantile(values, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(
        f"{name}_quantiles_min_p01_p05_p50_p95_p99_max="
        + ",".join(f"{float(v):.17g}" for v in q)
    )
    print(f"{name}_mean={float(values.mean()):.17g}")
    print(f"{name}_std={float(values.std()):.17g}")
    print(f"{name}_cv={float(values.std() / values.mean()):.17g}")


def main() -> None:
    with h5py.File(OPEN_BEAM, "r") as handle:
        edges_ns = np.asarray(handle["/entry/histogram/time_of_flight"])
        centers_us = 0.5 * (edges_ns[:-1] + edges_ns[1:]) / 1000.0
        energies = (TOF_FACTOR_US_SQRT_EV_PER_M * FLIGHT_PATH_M / centers_us) ** 2
        selected = np.flatnonzero(
            (energies >= ENERGY_WINDOW_EV[0]) & (energies <= ENERGY_WINDOW_EV[1])
        )
        first, stop = int(selected[0]), int(selected[-1] + 1)
        counts = handle["/entry/histogram/counts"]
        height, width = int(counts.shape[1]), int(counts.shape[2])
        n_selected = stop - first
        total = np.zeros((height, width), dtype=np.float64)
        sum_all = np.zeros(n_selected, dtype=np.float64)
        sumsq_all = np.zeros(n_selected, dtype=np.float64)
        sum_live = np.zeros(n_selected, dtype=np.float64)
        sumsq_live = np.zeros(n_selected, dtype=np.float64)
        n_live = 0

        # The file is chunked (1,32,32,4367). Reading by spatial chunk and
        # taking the whole selected spectrum decompresses each HDF5 chunk once;
        # spectral-block slicing would repeatedly decompress every chunk.
        for y0 in range(0, height, SPATIAL_BLOCK):
            y1 = min(y0 + SPATIAL_BLOCK, height)
            for x0 in range(0, width, SPATIAL_BLOCK):
                x1 = min(x0 + SPATIAL_BLOCK, width)
                block = np.asarray(counts[0, y0:y1, x0:x1, first:stop], dtype=np.float64)
                region_total = block.sum(axis=2)
                total[y0:y1, x0:x1] = region_total
                flat = block.reshape(-1, n_selected)
                sum_all += flat.sum(axis=0)
                sumsq_all += np.square(flat).sum(axis=0)
                region_live = region_total.ravel() > 0.0
                live_flat = flat[region_live]
                n_live += int(region_live.sum())
                sum_live += live_flat.sum(axis=0)
                sumsq_live += np.square(live_flat).sum(axis=0)

        live = total > 0.0
        all_values = total.ravel()
        live_values = total[live]
        normalized = live_values / np.median(live_values)
        n_all = height * width
        mean_all = sum_all / n_all
        mean_live = sum_live / n_live
        std_all = np.sqrt(np.maximum(sumsq_all / n_all - np.square(mean_all), 0.0))
        std_live = np.sqrt(np.maximum(sumsq_live / n_live - np.square(mean_live), 0.0))
        bin_cv_all = np.divide(std_all, mean_all, out=np.zeros_like(std_all), where=mean_all > 0)
        bin_cv_live = np.divide(
            std_live, mean_live, out=np.zeros_like(std_live), where=mean_live > 0
        )

    print(f"hdf5_counts_shape={(1, height, width, len(edges_ns) - 1)}")
    print(f"selected_bin_range=[{first},{stop})")
    print(f"selected_bins={stop - first}")
    print(f"selected_energy_min_ev={float(energies[stop - 1]):.17g}")
    print(f"selected_energy_max_ev={float(energies[first]):.17g}")
    print(f"pixels_total={total.size}")
    print(f"pixels_zero_total={int((~live).sum())}")
    summarize("pixel_total_all", all_values)
    summarize("pixel_total_nonzero", live_values)
    summarize("pixel_total_nonzero_over_median", normalized)
    shot_cv = 1.0 / np.sqrt(live_values.mean())
    print(f"poisson_cv_at_mean_total={float(shot_cv):.17g}")

    half = width // 2
    left = total[:, :half][live[:, :half]].mean()
    right = total[:, half:][live[:, half:]].mean()
    print(f"left_half_nonzero_mean={float(left):.17g}")
    print(f"right_half_nonzero_mean={float(right):.17g}")
    print(f"right_over_left={float(right / left):.17g}")
    for y0, y1, x0, x1, label in (
        (0, height // 2, 0, width // 2, "top_left"),
        (0, height // 2, width // 2, width, "top_right"),
        (height // 2, height, 0, width // 2, "bottom_left"),
        (height // 2, height, width // 2, width, "bottom_right"),
    ):
        region = total[y0:y1, x0:x1]
        region_live = live[y0:y1, x0:x1]
        print(f"quadrant_{label}_nonzero_mean={float(region[region_live].mean()):.17g}")

    summarize("per_bin_spatial_cv_all", np.asarray(bin_cv_all))
    summarize("per_bin_spatial_cv_nonzero", np.asarray(bin_cv_live))


if __name__ == "__main__":
    main()
