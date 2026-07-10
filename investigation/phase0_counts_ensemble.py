"""Matched-model count ensemble for Phase 0.

Compares the native joint-Poisson counts route with the counts-to-transmission
LM fallback at c=1, so the already-confirmed ignored-c defect is not the cause
of any difference. Every failed/non-converged fit is counted explicitly.
"""

from pathlib import Path

import numpy as np
import nereids


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/venus/aggregated_hf_120min.npz"
ENDF = ROOT / "tests/data/endf/Hf-177.endf"
TRUE_DENSITY = 8.0e-5
N_REPLICATES = 50
EXPOSURES = (25.0, 250.0, 2500.0)


def summarize(name: str, exposure: float, records: list[tuple[bool, float, float]]) -> None:
    converged = [r for r in records if r[0] and np.isfinite(r[1])]
    estimates = np.asarray([r[1] for r in converged])
    uncertainty_records = [r for r in converged if np.isfinite(r[2]) and r[2] >= 0.0]
    covered = sum(abs(r[1] - TRUE_DENSITY) <= r[2] for r in uncertainty_records)
    print(
        f"route={name} exposure={exposure:.0f} total={len(records)} "
        f"converged={len(converged)} finite_uncertainty={len(uncertainty_records)}"
    )
    if len(estimates):
        bias = float(estimates.mean() - TRUE_DENSITY)
        rmse = float(np.sqrt(np.mean((estimates - TRUE_DENSITY) ** 2)))
        print(
            f"  mean={float(estimates.mean()):.17g} bias={bias:.17g} "
            f"relative_bias={bias / TRUE_DENSITY:.17g} rmse={rmse:.17g}"
        )
    if uncertainty_records:
        print(
            f"  nominal_1sigma_coverage={covered / len(uncertainty_records):.17g} "
            f"covered={covered}/{len(uncertainty_records)}"
        )


def main() -> None:
    with np.load(FIXTURE) as data:
        # Deterministic thinning keeps the measured VENUS axis while making the
        # 300 fits fast. No generated observation is discarded or windowed.
        energies = np.ascontiguousarray(data["energies_ev"][::16])

    isotope = nereids.load_endf_file(str(ENDF))
    truth = np.asarray(
        nereids.forward_model(
            energies,
            [(isotope, TRUE_DENSITY)],
            temperature_k=293.6,
        )
    )
    rng = np.random.default_rng(20260709)

    for exposure in EXPOSURES:
        joint: list[tuple[bool, float, float]] = []
        fallback: list[tuple[bool, float, float]] = []
        for _ in range(N_REPLICATES):
            open_beam = rng.poisson(exposure, size=len(energies)).astype(float)
            sample = rng.poisson(exposure * truth).astype(float)
            common = dict(
                sample_counts=sample,
                open_beam_counts=open_beam,
                energies=energies,
                isotopes=[(isotope, TRUE_DENSITY / 2.0)],
                temperature_k=293.6,
                max_iter=100,
                background=False,
                c=1.0,
            )
            for solver, target in (("kl", joint), ("lm", fallback)):
                try:
                    result = nereids.fit_counts_spectrum_typed(
                        **common,
                        solver=solver,
                    )
                    estimate = float(np.asarray(result.densities)[0])
                    uncertainty = float(np.asarray(result.uncertainties)[0])
                    target.append((bool(result.converged), estimate, uncertainty))
                except Exception:
                    target.append((False, np.nan, np.nan))

        summarize("joint_poisson", exposure, joint)
        summarize("counts_lm_fallback", exposure, fallback)


if __name__ == "__main__":
    main()
