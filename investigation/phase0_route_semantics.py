"""Phase-0 semantic probes on the committed aggregated VENUS Hf fixture.

This script does not claim that the deliberately incomplete one-isotope,
no-background model describes the experiment. It tests route semantics that
should hold independently of fit quality:

* whether transmission Poisson-KL uses the supplied uncertainty in its
  optimization;
* whether the raw-count LM fallback honors the documented proton-charge ratio;
* whether explicitly pre-normalizing the open beam changes that fallback.
"""

from pathlib import Path

import numpy as np
import nereids


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/venus/aggregated_hf_120min.npz"
ENDF = ROOT / "tests/data/endf/Hf-177.endf"


def fit_transmission(
    transmission: np.ndarray,
    uncertainty: np.ndarray,
    energies: np.ndarray,
    isotope: object,
    solver: str,
) -> object:
    return nereids.fit_spectrum_typed(
        transmission,
        uncertainty,
        energies,
        isotopes=[(isotope, 1.0e-5)],
        solver=solver,
        temperature_k=293.6,
        max_iter=200,
        background=False,
        flight_path_m=25.0,
        delta_t_us=0.5,
        delta_l_m=0.005,
    )


def fit_counts(
    sample: np.ndarray,
    open_beam: np.ndarray,
    energies: np.ndarray,
    isotope: object,
    solver: str,
    c: float,
) -> object:
    return nereids.fit_counts_spectrum_typed(
        sample,
        open_beam,
        energies,
        isotopes=[(isotope, 1.0e-5)],
        solver=solver,
        temperature_k=293.6,
        max_iter=200,
        background=False,
        c=c,
        flight_path_m=25.0,
        delta_t_us=0.5,
        delta_l_m=0.005,
    )


def density(result: object) -> float:
    return float(np.asarray(result.densities)[0])


def main() -> None:
    with np.load(FIXTURE) as data:
        energies = np.ascontiguousarray(data["energies_ev"])
        sample = np.ascontiguousarray(data["sample_counts"])
        open_beam = np.ascontiguousarray(data["open_beam_counts"])
        c = float(data["pc_ratio"])

    isotope = nereids.load_endf_file(str(ENDF))

    transmission = sample / np.maximum(c * open_beam, 1.0)
    uncertainty = transmission * np.sqrt(
        1.0 / np.maximum(sample, 1.0) + 1.0 / np.maximum(open_beam, 1.0)
    )

    # Change relative weights by eight orders of magnitude across the spectrum,
    # not merely by a global scale. A weighted-likelihood route must notice.
    relative_weight_change = np.geomspace(1.0e-4, 1.0e4, len(uncertainty))
    uncertainty_reweighted = uncertainty * relative_weight_change

    lm_reference = fit_transmission(transmission, uncertainty, energies, isotope, "lm")
    lm_reweighted = fit_transmission(
        transmission, uncertainty_reweighted, energies, isotope, "lm"
    )
    kl_reference = fit_transmission(transmission, uncertainty, energies, isotope, "kl")
    kl_reweighted = fit_transmission(
        transmission, uncertainty_reweighted, energies, isotope, "kl"
    )
    transmission_with_negative = transmission.copy()
    transmission_with_negative[len(transmission_with_negative) // 2] = -0.1
    try:
        kl_negative = fit_transmission(
            transmission_with_negative, uncertainty, energies, isotope, "kl"
        )
        negative_observation_result = (
            f"accepted converged={bool(kl_negative.converged)} "
            f"density={density(kl_negative):.17g}"
        )
    except Exception as exc:
        negative_observation_result = f"rejected type={type(exc).__name__} message={exc}"

    jp_raw = fit_counts(sample, open_beam, energies, isotope, "kl", c)
    lm_raw = fit_counts(sample, open_beam, energies, isotope, "lm", c)
    # The fallback computes S/O internally and ignores c. Supplying c*O with
    # c=1 emulates the normalization that the fallback would need.
    lm_pre_normalized_ob = fit_counts(sample, c * open_beam, energies, isotope, "lm", 1.0)

    print(f"pc_ratio={c:.17g}")
    print(f"lm_transmission_reference_density={density(lm_reference):.17g}")
    print(f"lm_transmission_reweighted_density={density(lm_reweighted):.17g}")
    print(
        "lm_transmission_reweight_delta="
        f"{density(lm_reweighted) - density(lm_reference):.17g}"
    )
    print(f"kl_transmission_reference_density={density(kl_reference):.17g}")
    print(f"kl_transmission_reweighted_density={density(kl_reweighted):.17g}")
    print(
        "kl_transmission_reweight_delta="
        f"{density(kl_reweighted) - density(kl_reference):.17g}"
    )
    print(f"kl_negative_transmission_observation={negative_observation_result}")
    print(f"joint_poisson_raw_counts_density={density(jp_raw):.17g}")
    print(f"lm_raw_counts_with_c_density={density(lm_raw):.17g}")
    print(f"lm_counts_with_prescaled_ob_density={density(lm_pre_normalized_ob):.17g}")
    print(f"lm_raw_counts_converged={bool(lm_raw.converged)}")
    print(f"lm_prescaled_ob_converged={bool(lm_pre_normalized_ob.converged)}")


if __name__ == "__main__":
    main()
