"""Public-Python smoke/rejection probes for Phase-0 matrix cells.

These matched, noise-free controls exercise routes not covered by the committed
real-data anchors. They establish dispatch and rejection behavior, not real-data
model adequacy.
"""

from pathlib import Path

import numpy as np
import nereids


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/venus/aggregated_hf_120min.npz"
ENDF = ROOT / "tests/data/endf/Hf-177.endf"
TRUE_DENSITY = 8.0e-5


def describe_error(label: str, call: object) -> None:
    try:
        result = call()
        print(f"{label}=accepted type={type(result).__name__}")
    except Exception as exc:
        print(f"{label}=rejected type={type(exc).__name__} message={exc}")


def main() -> None:
    with np.load(FIXTURE) as data:
        energies = np.ascontiguousarray(data["energies_ev"][::16])
    isotope = nereids.load_endf_file(str(ENDF))
    transmission = np.asarray(
        nereids.forward_model(energies, [(isotope, TRUE_DENSITY)], temperature_k=293.6)
    )
    uncertainty = np.full_like(transmission, 0.01)

    t_cube = transmission[:, None, None].copy()
    u_cube = uncertainty[:, None, None].copy()
    transmission_input = nereids.from_transmission(t_cube, u_cube)
    spatial_kl = nereids.spatial_map_typed(
        transmission_input,
        energies,
        isotopes=[isotope],
        solver="kl",
        initial_densities=[TRUE_DENSITY / 2.0],
        max_iter=100,
    )
    spatial_joint_alias = nereids.spatial_map_typed(
        transmission_input,
        energies,
        isotopes=[isotope],
        solver="joint_poisson",
        initial_densities=[TRUE_DENSITY / 2.0],
        max_iter=100,
    )
    print(f"spatial_transmission_kl_density={float(np.asarray(spatial_kl.density_maps)[0,0,0]):.17g}")
    print(
        "spatial_transmission_joint_poisson_alias_density="
        f"{float(np.asarray(spatial_joint_alias.density_maps)[0,0,0]):.17g}"
    )

    negative_cube = t_cube.copy()
    negative_cube[len(negative_cube) // 2, 0, 0] = -0.1
    negative_result = nereids.spatial_map_typed(
        nereids.from_transmission(negative_cube, u_cube),
        energies,
        isotopes=[isotope],
        solver="kl",
        initial_densities=[TRUE_DENSITY / 2.0],
        max_iter=100,
    )
    print(
        "spatial_transmission_kl_negative_observation="
        f"converged={bool(np.asarray(negative_result.converged_map)[0,0])} "
        f"density={float(np.asarray(negative_result.density_maps)[0,0,0]):.17g}"
    )

    open_beam = np.full_like(transmission, 1000.0)
    sample = open_beam * transmission
    zero_background = np.zeros_like(sample)
    nonzero_background = np.full_like(sample, 1.0)

    paired = nereids.fit_counts_spectrum_typed(
        sample,
        open_beam,
        energies,
        isotopes=[(isotope, TRUE_DENSITY / 2.0)],
        solver="kl",
        detector_background=zero_background,
        max_iter=100,
    )
    print(
        f"single_zero_background_nuisance=converged={bool(paired.converged)} "
        f"density={float(np.asarray(paired.densities)[0]):.17g}"
    )

    describe_error(
        "single_nonzero_detector_background",
        lambda: nereids.fit_counts_spectrum_typed(
            sample,
            open_beam,
            energies,
            isotopes=[(isotope, TRUE_DENSITY / 2.0)],
            solver="kl",
            detector_background=nonzero_background,
            max_iter=100,
        ),
    )
    describe_error(
        "single_fit_alpha_1",
        lambda: nereids.fit_counts_spectrum_typed(
            sample,
            open_beam,
            energies,
            isotopes=[(isotope, TRUE_DENSITY / 2.0)],
            solver="kl",
            fit_alpha_1=True,
            max_iter=100,
        ),
    )
    single_lm_alpha = nereids.fit_counts_spectrum_typed(
        sample,
        open_beam,
        energies,
        isotopes=[(isotope, TRUE_DENSITY / 2.0)],
        solver="lm",
        fit_alpha_1=True,
        max_iter=100,
    )
    print(
        "single_counts_lm_fit_alpha_1="
        f"accepted converged={bool(single_lm_alpha.converged)} "
        f"density={float(np.asarray(single_lm_alpha.densities)[0]):.17g} "
        f"alpha_1={getattr(single_lm_alpha, 'alpha_1', None)} "
        f"alpha_2={getattr(single_lm_alpha, 'alpha_2', None)}"
    )
    describe_error(
        "single_zero_background_nuisance_lm",
        lambda: nereids.fit_counts_spectrum_typed(
            sample,
            open_beam,
            energies,
            isotopes=[(isotope, TRUE_DENSITY / 2.0)],
            solver="lm",
            detector_background=zero_background,
            max_iter=100,
        ),
    )

    # Unlike the single-spectrum boundary, the spatial Bdet error is currently
    # caught inside the pixel loop and returned as an accepted failure map.
    try:
        spatial_nonzero_background = nereids.spatial_map_typed(
            nereids.from_counts_with_nuisance(
                sample[:, None, None],
                open_beam[:, None, None],
                nonzero_background[:, None, None],
            ),
            energies,
            isotopes=[isotope],
            solver="kl",
            initial_densities=[TRUE_DENSITY / 2.0],
            max_iter=100,
        )
        print(
            "spatial_nonzero_detector_background="
            f"accepted n_failed={int(spatial_nonzero_background.n_failed)} "
            f"converged={bool(np.asarray(spatial_nonzero_background.converged_map)[0,0])} "
            f"density={float(np.asarray(spatial_nonzero_background.density_maps)[0,0,0]):.17g}"
        )
    except Exception as exc:
        print(
            "spatial_nonzero_detector_background="
            f"rejected type={type(exc).__name__} message={exc}"
        )

    # Spatial raw-count LM: c=2 is accepted at the boundary but ignored by the
    # ratio conversion, just as in the single route.
    c = 2.0
    sample_c = c * sample
    spatial_counts_lm = nereids.spatial_map_typed(
        nereids.from_counts(sample_c[:, None, None], open_beam[:, None, None]),
        energies,
        isotopes=[isotope],
        solver="lm",
        c=c,
        initial_densities=[TRUE_DENSITY / 2.0],
        max_iter=100,
    )
    print(
        "spatial_counts_lm_c2="
        f"converged={bool(np.asarray(spatial_counts_lm.converged_map)[0,0])} "
        f"density={float(np.asarray(spatial_counts_lm.density_maps)[0,0,0]):.17g}"
    )

    describe_error(
        "spatial_counts_lm_fit_alpha_1",
        lambda: nereids.spatial_map_typed(
            nereids.from_counts(sample[:, None, None], open_beam[:, None, None]),
            energies,
            isotopes=[isotope],
            solver="lm",
            fit_alpha_1=True,
            initial_densities=[TRUE_DENSITY / 2.0],
            max_iter=100,
        ),
    )

    describe_error(
        "spatial_counts_nuisance_lm",
        lambda: nereids.spatial_map_typed(
            nereids.from_counts_with_nuisance(
                sample[:, None, None],
                open_beam[:, None, None],
                zero_background[:, None, None],
            ),
            energies,
            isotopes=[isotope],
            solver="lm",
            initial_densities=[TRUE_DENSITY / 2.0],
            max_iter=100,
        ),
    )


if __name__ == "__main__":
    main()
