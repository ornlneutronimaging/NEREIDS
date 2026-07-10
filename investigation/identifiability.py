"""Local IC identifiability probe on the archived RT region-count cache.

Prints a JSON record; it does not modify data or write output files.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import nereids

from archive_inputs import archive_root

ARCHIVE = archive_root()
COUNTS = ARCHIVE / "01_spectral_lineshape_bias/data/region_counts.npz"
CACHE = ARCHIVE / "notebooks/calib_IC_jendl5_24685.json"
ROBUST = ARCHIVE / "01_spectral_lineshape_bias/data/exp09_jendl5.json"

FLIGHT_PATH_M = 25.0
INITIAL_DENSITY = 7.0374e-4
ENERGY_RANGE_EV = (4.0, 120.0)
PSR_FWHM_US = 0.35


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pack(params: dict[str, float]) -> np.ndarray:
    r = params["r"]
    return np.array(
        [math.log(params["a0"]), math.log(params["a1"]),
         math.log(params["beta"]), math.log(r / (1.0 - r))],
        dtype=float,
    )


def unpack(q: np.ndarray) -> dict[str, float]:
    return {
        "a0": math.exp(float(q[0])),
        "a1": math.exp(float(q[1])),
        "beta": math.exp(float(q[2])),
        "r": sigmoid(float(q[3])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-min", type=float, default=8.0)
    parser.add_argument("--fit-max", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_window_ev = (args.fit_min, args.fit_max)
    if not (0.0 < args.fit_min < args.fit_max):
        raise ValueError("fit window endpoints must be positive and ascending")
    d = np.load(COUNTS, allow_pickle=False)
    cache = json.loads(CACHE.read_text())
    robust_json = json.loads(ROBUST.read_text())["ic"]["ic_params"]
    cache_p = {k: float(cache[f"ic_{k}"]) for k in ("a0", "a1", "beta", "r")}
    robust_p = {k: float(robust_json[k]) for k in ("a0", "a1", "beta", "r")}

    keep = (d["energies"] >= ENERGY_RANGE_EV[0]) & (d["energies"] <= ENERGY_RANGE_EV[1])
    e_corr = np.asarray(
        nereids.tof_to_energy_centers(
            d["edges"], FLIGHT_PATH_M * float(cache["l_scale"]), float(cache["t0_us"])
        )
    )[keep]
    win = (e_corr >= fit_window_ev[0]) & (e_corr <= fit_window_ev[1])
    energy = e_corr[win]
    cal = d["counts_calib"][keep][win]
    ob = d["counts_ob"][keep][win]
    transmission = (cal / ob) * float(d["pc_ob"] / d["pc_calib"])
    uncertainty = transmission * np.sqrt(1.0 / np.maximum(cal, 1.0) + 1.0 / np.maximum(ob, 1.0))
    uncertainty = np.maximum(uncertainty, 1e-9)
    x = np.linspace(-1.0, 1.0, len(energy))
    ta181 = nereids.load_endf(73, 181, library="jendl5")

    def model(q: np.ndarray) -> np.ndarray:
        p = unpack(q)
        ic = nereids.IkedaCarpenter(
            flight_path_m=FLIGHT_PATH_M,
            e_min_ev=0.5 * float(energy.min()),
            e_max_ev=2.0 * float(energy.max()),
            alpha=nereids.EnergyLaw.sqrt_e(p["a0"], p["a1"]),
            beta=p["beta"],
            r=nereids.EnergyLaw.const(p["r"]),
            n_energies=48,
            n_tau=400,
            channel_fwhm_us=PSR_FWHM_US,
        )
        return np.asarray(
            nereids.forward_model(
                energy,
                [(ta181, INITIAL_DENSITY)],
                temperature_k=float(d["T_calib_k"]),
                resolution=ic.as_tabulated(),
            )
        )

    def profiled(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        design = np.column_stack((m, np.ones_like(m), x))
        aw = design / uncertainty[:, None]
        yw = transmission / uncertainty
        coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
        residual = (transmission - design @ coef) / uncertainty
        return coef, residual, float(residual @ residual)

    q0 = pack(cache_p)
    m0 = model(q0)
    coef0, resid0, ssr0 = profiled(m0)
    step = 1e-4
    jac = np.empty((len(energy), 4), dtype=float)
    for j in range(4):
        dq = np.zeros(4)
        dq[j] = step
        jac[:, j] = (model(q0 + dq) - model(q0 - dq)) / (2.0 * step)

    # Concentrated least-squares Jacobian: scale by fitted amplitude, whiten,
    # and project away normalization + constant/linear additive background.
    nuisance = np.column_stack((m0, np.ones_like(m0), x)) / uncertainty[:, None]
    qn, _ = np.linalg.qr(nuisance, mode="reduced")
    jwhite = float(coef0[0]) * jac / uncertainty[:, None]
    jproj = jwhite - qn @ (qn.T @ jwhite)
    u, singular, vt = np.linalg.svd(jproj, full_matrices=False)
    del u
    col_norm = np.linalg.norm(jproj, axis=0)
    jnorm = jproj / col_norm[None, :]
    normalized_singular = np.linalg.svd(jnorm, compute_uv=False)
    derivative_corr = jnorm.T @ jnorm
    fisher = jproj.T @ jproj
    covariance = np.linalg.pinv(fisher, rcond=1e-12)
    sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    parameter_corr = covariance / np.outer(sd, sd)

    qr = pack(robust_p)
    mr = model(qr)
    coefr, residr, ssrr = profiled(mr)
    cache_fit = coef0[0] * m0 + coef0[1] + coef0[2] * x
    robust_fit = coefr[0] * mr + coefr[1] + coefr[2] * x
    delta_sigma = (robust_fit - cache_fit) / uncertainty
    eprobe = np.array([8.0, 10.0, 20.0, 45.0])
    alpha_cache = cache_p["a0"] * np.sqrt(eprobe) + cache_p["a1"]
    alpha_robust = robust_p["a0"] * np.sqrt(eprobe) + robust_p["a1"]

    names = ["log_a0", "log_a1", "log_beta", "logit_R"]
    result = {
        "n_bins": int(len(energy)),
        "energy_range_ev": [float(energy.min()), float(energy.max())],
        "coordinates": names,
        "cache_params": cache_p,
        "robust_grid_params": robust_p,
        "profiled_cache_coefficients": coef0.tolist(),
        "profiled_robust_coefficients": coefr.tolist(),
        "cache_ssr": ssr0,
        "robust_ssr": ssrr,
        "cache_reduced_chi2_4_outer": ssr0 / max(len(energy) - 3 - 4, 1),
        "robust_reduced_chi2_4_outer": ssrr / max(len(energy) - 3 - 4, 1),
        "singular_values": singular.tolist(),
        "jacobian_condition": float(singular[0] / singular[-1]),
        "column_normalized_singular_values": normalized_singular.tolist(),
        "column_normalized_condition": float(normalized_singular[0] / normalized_singular[-1]),
        "least_informed_direction": dict(zip(names, vt[-1].tolist(), strict=True)),
        "projected_derivative_correlation": derivative_corr.tolist(),
        "local_parameter_correlation_pinv": parameter_corr.tolist(),
        "column_norms": dict(zip(names, col_norm.tolist(), strict=True)),
        "cache_vs_robust_rms_sigma": float(np.sqrt(np.mean(delta_sigma**2))),
        "cache_vs_robust_max_abs_sigma": float(np.max(np.abs(delta_sigma))),
        "cache_vs_robust_max_abs_transmission": float(np.max(np.abs(robust_fit - cache_fit))),
        "alpha_probe_energy_ev": eprobe.tolist(),
        "alpha_cache_per_us": alpha_cache.tolist(),
        "alpha_robust_per_us": alpha_robust.tolist(),
        "alpha_relative_difference": ((alpha_robust - alpha_cache) / alpha_cache).tolist(),
        "cache_residual_rms_sigma": float(np.sqrt(np.mean(resid0**2))),
        "robust_residual_rms_sigma": float(np.sqrt(np.mean(residr**2))),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
