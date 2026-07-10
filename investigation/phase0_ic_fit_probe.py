"""End-to-end ordinary-fit smoke for Python IC-as-tabulated handoff."""

from pathlib import Path

import numpy as np
import nereids


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/venus/aggregated_hf_120min.npz"
ENDF = ROOT / "tests/data/endf/Hf-177.endf"
TRUE_DENSITY = 8.0e-5


def main() -> None:
    with np.load(FIXTURE) as data:
        energies = np.ascontiguousarray(data["energies_ev"][::8])
    isotope = nereids.load_endf_file(str(ENDF))

    ic = nereids.IkedaCarpenter(
        flight_path_m=25.0,
        e_min_ev=float(energies[0]),
        e_max_ev=float(energies[-1]),
        alpha=nereids.EnergyLaw.sqrt_e(0.35, 0.05),
        beta=0.25,
        r=nereids.EnergyLaw.const(0.15),
        n_energies=32,
        n_tau=320,
        channel_fwhm_us=0.35,
    )
    tabulated = ic.as_tabulated()
    truth = np.asarray(
        nereids.forward_model(
            energies,
            [(isotope, TRUE_DENSITY)],
            temperature_k=293.6,
            resolution=tabulated,
        )
    )

    lm = nereids.fit_spectrum_typed(
        truth,
        np.full_like(truth, 0.003),
        energies,
        isotopes=[(isotope, TRUE_DENSITY / 2.0)],
        solver="lm",
        temperature_k=293.6,
        resolution=tabulated,
        max_iter=100,
    )
    open_beam = np.full_like(truth, 2000.0)
    counts = nereids.fit_counts_spectrum_typed(
        open_beam * truth,
        open_beam,
        energies,
        isotopes=[(isotope, TRUE_DENSITY / 2.0)],
        solver="kl",
        temperature_k=293.6,
        resolution=tabulated,
        max_iter=100,
    )

    print(f"true_density={TRUE_DENSITY:.17g}")
    print(
        f"transmission_lm_converged={bool(lm.converged)} "
        f"density={float(np.asarray(lm.densities)[0]):.17g}"
    )
    print(
        f"counts_joint_poisson_converged={bool(counts.converged)} "
        f"density={float(np.asarray(counts.densities)[0]):.17g}"
    )


if __name__ == "__main__":
    main()
