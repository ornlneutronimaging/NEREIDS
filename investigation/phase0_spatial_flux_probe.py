"""Test spatial counts dispatch under a known spatial open-beam gradient.

The ordinary ``from_counts`` route intentionally replaces every pixel's open
beam by the spatial mean. ``from_counts_with_nuisance`` retains each pixel's
supplied flux. This noise-free matched-model control measures the consequence
without detector background or random noise.
"""

from pathlib import Path

import numpy as np
import nereids


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/venus/aggregated_hf_120min.npz"
ENDF = ROOT / "tests/data/endf/Hf-177.endf"
TRUE_DENSITY = 8.0e-5


def main() -> None:
    with np.load(FIXTURE) as data:
        # Thin the real VENUS grid deterministically to make this dispatch probe
        # quick while retaining the measured energy-axis structure.
        energies = np.ascontiguousarray(data["energies_ev"][::8])

    isotope = nereids.load_endf_file(str(ENDF))
    transmission = np.asarray(
        nereids.forward_model(
            energies,
            [(isotope, TRUE_DENSITY)],
            temperature_k=293.6,
        )
    )

    # Four known flux levels: their spatial mean is 1.25e4. Sample counts are
    # exact expectations, so any route difference is from flux handling alone.
    flux_levels = np.array([[5.0e3, 1.0e4], [1.5e4, 2.0e4]])
    flux = np.broadcast_to(flux_levels, (len(energies), 2, 2)).copy()
    sample = flux * transmission[:, None, None]
    background = np.zeros_like(sample)

    common = dict(
        energies=energies,
        isotopes=[isotope],
        solver="kl",
        temperature_k=293.6,
        initial_densities=[1.0e-5],
        max_iter=100,
        background=False,
        c=1.0,
    )

    averaged = nereids.spatial_map_typed(nereids.from_counts(sample, flux), **common)
    paired = nereids.spatial_map_typed(
        nereids.from_counts_with_nuisance(sample, flux, background), **common
    )

    averaged_density = np.asarray(averaged.density_maps)[0]
    paired_density = np.asarray(paired.density_maps)[0]
    averaged_converged = np.asarray(averaged.converged_map)
    paired_converged = np.asarray(paired.converged_map)

    print(f"true_density={TRUE_DENSITY:.17g}")
    print(f"flux_levels={flux_levels.tolist()}")
    print(f"averaged_flux={float(flux_levels.mean()):.17g}")
    print(f"from_counts_density_map={averaged_density.tolist()}")
    print(f"from_counts_converged_map={averaged_converged.tolist()}")
    print(f"paired_nuisance_density_map={paired_density.tolist()}")
    print(f"paired_nuisance_converged_map={paired_converged.tolist()}")
    print(
        "paired_max_relative_error="
        f"{float(np.nanmax(np.abs(paired_density - TRUE_DENSITY) / TRUE_DENSITY)):.17g}"
    )


if __name__ == "__main__":
    main()
