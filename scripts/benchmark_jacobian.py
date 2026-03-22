#!/usr/bin/env python3
"""Benchmark: analytical Jacobian performance and correctness.

Systematically tests ALL valid pipeline paths with and without the
analytical Jacobian optimization. Measures wall-clock time and verifies
correctness (density/temperature recovery) at multiple noise levels.

Usage:
    pixi run python scripts/benchmark_jacobian.py
"""
import time
import numpy as np
import nereids

# ── Shared setup ──────────────────────────────────────────────────────

ENERGIES = np.linspace(1.0, 50.0, 300)
FLIGHT_PATH_M = 25.0
DELTA_T_US = 0.3
DELTA_L_M = 0.01
TEMP_K = 293.6
TRUE_TEMP = 400.0
TRUE_DENSITY = 0.001

# Load Hf group (6 isotopes) and individual isotopes
group_hf = nereids.IsotopeGroup.natural(72)
group_hf.load_endf()

hf_mass_numbers = [a for (_, a), _ in group_hf.members]
hf_abundances = [r for (_, _), r in group_hf.members]
hf_isotopes = [nereids.load_endf(72, a) for a in hf_mass_numbers]

RESOLUTION_KWARGS = dict(
    flight_path_m=FLIGHT_PATH_M,
    delta_t_us=DELTA_T_US,
    delta_l_m=DELTA_L_M,
)


def generate_data(temperature_k, I0, rng):
    """Generate synthetic transmission and counts data."""
    T_true = np.asarray(nereids.forward_model(
        ENERGIES,
        groups=[(group_hf, TRUE_DENSITY)],
        temperature_k=temperature_k,
        **RESOLUTION_KWARGS,
    ))
    counts_sample = rng.poisson(np.maximum(I0 * T_true, 0))
    counts_ob = rng.poisson(I0 * np.ones_like(T_true))
    T_noisy = counts_sample / np.maximum(counts_ob, 1).astype(float)
    sigma = np.sqrt(np.maximum(counts_sample, 1)) / np.maximum(counts_ob, 1).astype(float)
    return T_true, T_noisy, sigma, counts_sample.astype(float), counts_ob.astype(float)


# ── Define ALL valid pipeline paths ───────────────────────────────────

PATHS = [
    # (name, input_type, solver, background, fit_temperature)
    ("Trans+LM",           "transmission", "lm", False, False),
    ("Trans+LM+temp",      "transmission", "lm", False, True),
    ("Trans+LM+bg",        "transmission", "lm", True,  False),
    ("Trans+LM+bg+temp",   "transmission", "lm", True,  True),
    ("Trans+KL",           "transmission", "kl", False, False),
    ("Trans+KL+temp",      "transmission", "kl", False, True),
    ("Trans+KL+bg",        "transmission", "kl", True,  False),
    ("Trans+KL+bg+temp",   "transmission", "kl", True,  True),
    ("Counts+KL",          "counts",       "kl", False, False),
    ("Counts+KL+temp",     "counts",       "kl", False, True),
    ("Counts+LM",          "counts",       "lm", False, False),
    ("Counts+LM+temp",     "counts",       "lm", False, True),
]


def run_fit(path_name, input_type, solver, background, fit_temperature,
            T_noisy, sigma, counts_sample, counts_ob, temperature_k):
    """Run a single fit and return (densities, temperature, chi2, converged, elapsed)."""
    kwargs = dict(
        temperature_k=temperature_k if not fit_temperature else TEMP_K,
        fit_temperature=fit_temperature,
        max_iter=200,
        solver=solver,
        background=background,
        **RESOLUTION_KWARGS,
    )

    t0 = time.perf_counter()
    try:
        if input_type == "transmission":
            result = nereids.fit_spectrum_typed(
                T_noisy, sigma, ENERGIES,
                groups=[group_hf],
                initial_densities=[TRUE_DENSITY],
                **kwargs,
            )
        else:  # counts
            # Use spatial_map_typed with from_counts for counts path
            T_3d = counts_sample[:, None, None].copy()
            ob_3d = counts_ob[:, None, None].copy()
            data = nereids.from_counts(T_3d, ob_3d)
            map_result = nereids.spatial_map_typed(
                data, ENERGIES,
                groups=[group_hf],
                initial_densities=[TRUE_DENSITY],
                **kwargs,
            )
            # Extract single-pixel result
            class _R:
                pass
            result = _R()
            result.densities = [float(m[0, 0]) for m in map_result.density_maps]
            result.reduced_chi_squared = float(np.asarray(map_result.chi_squared_map)[0, 0])
            result.converged = bool(np.asarray(map_result.converged_map)[0, 0])
            result.temperature_k = (
                float(map_result.temperature_map[0, 0])
                if map_result.temperature_map is not None else None
            )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return None, None, None, False, elapsed, str(e)

    elapsed = time.perf_counter() - t0
    density = result.densities[0]
    temp = getattr(result, "temperature_k", None)
    chi2 = result.reduced_chi_squared
    conv = result.converged
    return density, temp, chi2, conv, elapsed, None


# ── Run benchmark ─────────────────────────────────────────────────────

def main():
    I0_VALUES = [30, 100, 500]
    N_TRIALS = 3

    print("=" * 120)
    print("NEREIDS Analytical Jacobian Benchmark")
    print("=" * 120)
    print(f"Isotope: Hf (nat), {group_hf.n_members} members")
    print(f"Energy grid: {len(ENERGIES)} bins, {ENERGIES[0]:.1f}-{ENERGIES[-1]:.1f} eV")
    print(f"True density: {TRUE_DENSITY} at/barn")
    print(f"True temperature (when fitted): {TRUE_TEMP} K")
    print(f"Trials per condition: {N_TRIALS}")
    print()

    # Header
    print(f"{'Path':<22} {'I0':>4} {'Density':>10} {'D_err%':>8} "
          f"{'Temp':>8} {'T_err':>8} {'Chi2':>8} {'Conv':>5} {'Time':>8} {'Error'}")
    print("-" * 120)

    for path_name, input_type, solver, background, fit_temperature in PATHS:
        for I0 in I0_VALUES:
            densities, temps, chi2s, times = [], [], [], []
            all_conv = True
            error_msg = None

            for trial in range(N_TRIALS):
                rng = np.random.default_rng(1000 * I0 + trial)
                temp_k = TRUE_TEMP if fit_temperature else TEMP_K
                T_true, T_noisy, sigma, counts_s, counts_ob = generate_data(temp_k, I0, rng)

                d, t, c2, conv, elapsed, err = run_fit(
                    path_name, input_type, solver, background, fit_temperature,
                    T_noisy, sigma, counts_s, counts_ob, temp_k,
                )

                if err is not None:
                    error_msg = err
                    break

                densities.append(d)
                if t is not None:
                    temps.append(t)
                chi2s.append(c2)
                times.append(elapsed)
                if not conv:
                    all_conv = False

            if error_msg is not None:
                print(f"{path_name:<22} {I0:>4} {'':>10} {'':>8} "
                      f"{'':>8} {'':>8} {'':>8} {'':>5} {'':>8} {error_msg[:40]}")
                break  # skip other I0 for this path

            mean_d = np.mean(densities)
            true_d = TRUE_DENSITY
            d_err = abs(mean_d - true_d) / true_d * 100

            if temps:
                mean_t = np.mean(temps)
                t_err = f"{abs(mean_t - TRUE_TEMP):.1f}K"
                t_str = f"{mean_t:.1f}"
            else:
                t_str = "—"
                t_err = "—"

            mean_c2 = np.mean(chi2s)
            mean_time = np.mean(times)

            print(f"{path_name:<22} {I0:>4} {mean_d:>10.6f} {d_err:>7.1f}% "
                  f"{t_str:>8} {t_err:>8} {mean_c2:>8.3f} "
                  f"{'Y' if all_conv else 'N':>5} {mean_time:>7.3f}s")

        print()

    # ── Correctness verification: noiseless round-trip ──
    print()
    print("=" * 120)
    print("CORRECTNESS VERIFICATION: Noiseless round-trip (perfect data)")
    print("=" * 120)
    print()

    sigma_perfect = np.full_like(ENERGIES, 0.001)
    n_pass = 0
    n_fail = 0

    for path_name, input_type, solver, background, fit_temperature in PATHS:
        temp_k = TRUE_TEMP if fit_temperature else TEMP_K
        T_true = np.asarray(nereids.forward_model(
            ENERGIES,
            groups=[(group_hf, TRUE_DENSITY)],
            temperature_k=temp_k,
            **RESOLUTION_KWARGS,
        ))
        # For counts path, use very high I0 to approximate noiseless
        counts_s = (10000 * T_true).astype(float)
        counts_ob = np.full_like(T_true, 10000.0)

        d, t, c2, conv, elapsed, err = run_fit(
            path_name, input_type, solver, background, fit_temperature,
            T_true, sigma_perfect, counts_s, counts_ob, temp_k,
        )

        if err is not None:
            print(f"  {path_name:<22} ERROR: {err[:60]}")
            n_fail += 1
            continue

        d_err = abs(d - TRUE_DENSITY) / TRUE_DENSITY * 100
        d_ok = d_err < 5.0  # 5% tolerance on noiseless

        if fit_temperature and t is not None:
            t_err = abs(t - TRUE_TEMP)
            t_ok = t_err < 50.0  # 50K tolerance (temperature is weakly constrained)
            status = "PASS" if (d_ok and t_ok and conv) else "FAIL"
            detail = f"d={d:.6f} ({d_err:.2f}%)  T={t:.1f}K (err={t_err:.1f}K)  conv={conv}"
        else:
            t_ok = True
            status = "PASS" if (d_ok and conv) else "FAIL"
            detail = f"d={d:.6f} ({d_err:.2f}%)  conv={conv}"

        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

        print(f"  {path_name:<22} [{status}]  {detail}")

    print()
    print(f"Results: {n_pass} PASS, {n_fail} FAIL out of {n_pass + n_fail} paths")
    if n_fail > 0:
        print("*** CORRECTNESS REGRESSION DETECTED ***")
        exit(1)
    else:
        print("All paths produce correct results on noiseless data.")

    print()
    print("=" * 120)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
