#!/usr/bin/env python3
"""Independent expected counts of the Ta-181 count check, from SAMMY's transmission.

The open and sample counts in each VENUS bin are

    O_j = ∫ φ(u + t0) P_j(u) du,   S_j = ∫ φ(u + t0) T(E(u)) P_j(u) du,

with u the flight time in µs, E(u) = m_n (L/u)²/2, φ the beam per µs at the
detector-clock time, T SAMMY's free-gas transmission without instrument and P_j
the chance that a neutron of flight time u is recorded in bin j. A neutron's delay
is the sum of the Ikeda–Carpenter emission time (Ikeda and Carpenter 1985,
Eq. 10′), the proton pulse (a centred triangle of FWHM h) and the detector delay
(prompt with chance 1 − f, else exponential of mean τ_D). Nothing here calls
NEREIDS. Run ``counts_reference.py -h`` for the commands.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# CODATA 2018 neutron mass (kg) and electron volt (J).
NEUTRON_MASS_KG = 1.674_927_498_04e-27
EV_TO_JOULES = 1.602_176_634e-19
FLIGHT_PATH_M = 25.0
T0_US = 0.0
# Bin edge j is (112 + 16 j) / 100 µs.
EDGE0_CENTI_US = 112
WIDTH_CENTI_US = 16
# Scored bins: centre energy from 0.5 eV to the midpoint of the 194.8900 and
# 200.1447 eV resonances.
E_LOW_EV = 0.5
E_LAST_EV = 197.51735
# Energies in eV over which the pulse laws hold.
SPAN_EV = (0.3, 1000.0)

# The pulse numbers [α0, α1, β0, β1, R, h², R_high, f, τ_D, p_α, p_β], then t0 and L.
NAMES = ("alpha0", "alpha1", "beta0", "beta1", "r", "h2", "r_high", "f", "tau_d",
         "p_alpha", "p_beta", "t0", "flight_path")
NOMINAL = np.array([0.5, 1.0, 0.01, 0.04, 0.3, 0.1225, 0.1, 0.1, 2.0, 0.5, 0.5,
                    T0_US, FLIGHT_PATH_M])


def beam(clock_us: np.ndarray) -> np.ndarray:
    """Return the beam, neutrons per µs, at detector-clock time ``clock_us``."""
    return 1e4 * (clock_us / 100.0) ** -0.8


def energy_ev(u_us, flight_path_m: float):
    """Return the energy in eV of a neutron of flight time ``u_us`` over the path."""
    v = flight_path_m / (u_us * 1e-6)
    return 0.5 * NEUTRON_MASS_KG * v * v / EV_TO_JOULES


def flight_time_us(e_ev, flight_path_m: float):
    """Return the flight time in µs of a neutron of energy ``e_ev`` over the path."""
    v = np.sqrt(2.0 * e_ev * EV_TO_JOULES / NEUTRON_MASS_KG)
    return flight_path_m / v * 1e6


def laws(e_ev: float, p: np.ndarray) -> tuple[float, float, float]:
    """Return α and β in 1/µs and the storage fraction R at energy ``e_ev``."""
    alpha = p[0] * e_ev ** p[9] + p[1]
    beta = p[2] * e_ev ** p[10] + p[3]
    root_low, root_high = math.sqrt(SPAN_EV[0]), math.sqrt(SPAN_EV[1])
    r = p[4] + (p[6] - p[4]) * (math.sqrt(e_ev) - root_low) / (root_high - root_low)
    return alpha, beta, r


def delay_terms(alpha: float, beta: float, r: float, f: float, tau: float):
    """Return ``(c0, terms)``: the delay CDF before the proton pulse is
    ``c0 + Σ c·y^n·e^(−λy)`` for y > 0, ``terms`` a list of ``(n, λ, c)``."""
    # Eq. 10′ as a sum of c·t^n·e^(−λt).
    k = alpha * r * alpha**2 * beta / (alpha - beta) ** 3
    density = [(2, alpha, (1.0 - r) * alpha**3 / 2.0), (0, beta, k), (0, alpha, -k),
               (1, alpha, -k * (alpha - beta)), (2, alpha, -k * (alpha - beta) ** 2 / 2.0)]
    # ∫_0^y w^n e^(−λw) dw = n!/λ^(n+1) − e^(−λy) Σ_k n!/(k! λ^(n−k+1)) y^k.
    c0 = sum(c * math.factorial(n) / lam ** (n + 1) for n, lam, c in density)
    cdf = [(kk, lam, -c * math.factorial(n) / (math.factorial(kk) * lam ** (n - kk + 1)))
           for n, lam, c in density for kk in range(n + 1)]
    # The exponential delay: ∫_0^y g(y − z) e^(−z/τ)/τ dz for g = y^n e^(−λy), μ = 1/τ − λ.
    late = [(0, 1.0 / tau, -c0)]
    for n, lam, c in cdf:
        mu = 1.0 / tau - lam
        for kk in range(n + 1):
            late.append((n - kk, lam, c / tau * (-1) ** kk * math.factorial(n)
                         / (math.factorial(n - kk) * mu ** (kk + 1))))
        late.append((0, 1.0 / tau, -c / tau * (-1) ** n * math.factorial(n) / mu ** (n + 1)))
    terms = [(n, lam, (1.0 - f) * c) for n, lam, c in cdf]
    terms += [(n, lam, f * c) for n, lam, c in late]
    return c0, terms


def _exp_part(n: int, lam: float, y: float) -> float:
    """The part of ∫_0^y (y − w) w^n e^(−λw) dw that holds e^(−λy)."""
    first = sum(math.factorial(n) / (math.factorial(k) * lam ** (n - k + 1)) * y**k
                for k in range(n + 1))
    second = sum(math.factorial(n + 1) / (math.factorial(k) * lam ** (n - k + 2)) * y**k
                 for k in range(n + 2))
    return math.exp(-lam * y) * (second - y * first)


def _second_antiderivative(c0: float, terms, y: float) -> float:
    """J(y) = ∫_0^y (y − w) F(w) dw of the delay CDF F, 0 for y ≤ 0."""
    if y <= 0.0:
        return 0.0
    total = c0 * y * y / 2.0
    for n, lam, c in terms:
        linear = y * math.factorial(n) / lam ** (n + 1) - math.factorial(n + 1) / lam ** (n + 2)
        total += c * (linear + _exp_part(n, lam, y))
    return total


def arrival_cdf(s: float, c0: float, terms, h: float) -> float:
    """The chance that the whole delay is at most ``s`` µs.

    The triangle of FWHM h is two boxes of width h convolved, so the CDF is
    [J(s + h) − 2J(s) + J(s − h)]/h². For s ≥ h the linear parts of J cancel exactly.
    """
    if h == 0.0:
        if s <= 0.0:
            return 0.0
        return c0 + sum(c * y_pow(s, n) * math.exp(-lam * s) for n, lam, c in terms)
    if s <= -h:
        return 0.0
    if s < h:
        return (_second_antiderivative(c0, terms, s + h) - 2.0 * _second_antiderivative(
            c0, terms, s) + _second_antiderivative(c0, terms, s - h)) / (h * h)
    total = c0
    for n, lam, c in terms:
        if n == 0:
            # e^(−λy)/λ² second-differenced: e^(−λs)·4 sinh²(λh/2)/λ².
            total += c * math.exp(-lam * s) * (2.0 * math.sinh(lam * h / 2.0) / lam) ** 2 / h**2
        else:
            total += c * (_exp_part(n, lam, s + h) - 2.0 * _exp_part(n, lam, s)
                          + _exp_part(n, lam, s - h)) / h**2
    return total


def y_pow(y: float, n: int) -> float:
    """y to the n."""
    return y**n


def brute_cdf(s: float, e_ev: float, p: np.ndarray) -> float:
    """The same CDF by direct numerical convolution, for the self-check."""
    from scipy import integrate

    alpha, beta, r = laws(e_ev, p)
    h, f, tau = math.sqrt(p[5]), p[7], p[8]

    def psi(t):
        if t <= 0.0:
            return 0.0
        d = alpha - beta
        return alpha / 2.0 * ((1 - r) * (alpha * t) ** 2 * math.exp(-alpha * t)
                              + 2 * r * alpha**2 * beta / d**3
                              * (math.exp(-beta * t) - math.exp(-alpha * t)
                                 * (1 + d * t + d * d * t * t / 2)))

    opts = {"epsabs": 1e-14, "epsrel": 1e-13, "limit": 400}

    def moderator_cdf(y):
        return integrate.quad(psi, 0.0, y, **opts)[0] if y > 0 else 0.0

    def detected_cdf(y):
        if y <= 0:
            return 0.0
        late = integrate.quad(lambda z: moderator_cdf(y - z) * math.exp(-z / tau) / tau,
                              0.0, y, **opts)[0]
        return (1 - f) * moderator_cdf(y) + f * late

    if h == 0:
        return detected_cdf(s)
    points = [x for x in (-h, 0.0, h, s) if -h <= x <= h]
    return integrate.quad(lambda x: (h - abs(x)) / h**2 * detected_cdf(s - x), -h, h,
                          points=sorted(set(points)), **opts)[0]


def self_check() -> None:
    """Compare the closed-form CDF, in Python and in the compiled kernel the counts use, with
    direct convolution at a few energies and delays."""
    import counts_kernel as kernel

    worst = 0.0
    for e in (0.5, 4.28, 50.0, 197.0):
        alpha, beta, r = laws(e, NOMINAL)
        c0, terms = delay_terms(alpha, beta, r, NOMINAL[7], NOMINAL[8])
        k0, lam, coef = kernel.delay_terms(alpha, beta, r, NOMINAL[7], NOMINAL[8])
        h = math.sqrt(NOMINAL[5])
        for s in (-0.3, -0.1, 0.05, 0.2, 0.34, 0.36, 1.0, 3.0, 10.0, 40.0, 150.0):
            closed = arrival_cdf(s, c0, terms, h)
            compiled = kernel.arrival_cdf(s, k0, lam, coef, h)
            brute = brute_cdf(s, e, NOMINAL)
            if not all(math.isfinite(v) for v in (closed, compiled, brute)):
                raise SystemExit(f"E {e} eV, s {s} µs: a CDF value is not finite")
            worst = max(worst, abs(closed - brute), abs(compiled - brute))
            print(f"E {e:7.2f} eV  s {s:7.2f} µs  closed {closed:.15f}  compiled {compiled:.15f}"
                  f"  brute {brute:.15f}  diff {closed - brute:+.2e} {compiled - brute:+.2e}")
    print(f"largest difference {worst:.2e}")
    if worst > 1e-10:
        raise SystemExit("a closed-form CDF disagrees with direct convolution")


# Integration grid in flight time u at the nominal path: du/dw = 1/(1/(C·u²) + 1/STEP_CAP_US)
# with w uniform, so the step follows the Doppler width (∝ u²) where that is finer.
STEP_CAP_US = 0.01
STEP_COEFF_PER_US = 0.005 / 128.0**2
# The grid's flight times span 1000 eV to 0.4 eV at the nominal path.
GRID_EV = (1000.0, 0.4)
# Delays beyond 36 decay lengths hold less than e^-36 of the probability.
TAIL_LENGTHS = 36.0
NXTRA_P1B = 3
TEMPERATURES = ("299", "299.5", "300", "300.5", "301", "1099", "1099.5", "1100", "1100.5", "1101")


def grid_w(u_us):
    """w(u) of the integration grid."""
    return -1.0 / (STEP_COEFF_PER_US * u_us) + u_us / STEP_CAP_US


def grid_u(w):
    """u(w), the inverse of grid_w."""
    b, c = STEP_CAP_US, STEP_COEFF_PER_US
    return (b * w + np.sqrt((b * w) ** 2 + 4.0 * b / c)) / 2.0


def nodes(level: int) -> tuple[list[str], np.ndarray]:
    """Return the grid's energies as SAMMY reads them, in time order, and each node's
    Simpson weight in µs at the nominal path, on the grid halved ``level`` times."""
    u_lo, u_hi = (flight_time_us(e, FLIGHT_PATH_M) for e in GRID_EV)
    w_lo, w_hi = grid_w(u_lo), grid_w(u_hi)
    intervals = 2 * math.ceil((w_hi - w_lo) / 2.0) * 2**level
    w = np.linspace(w_lo, w_hi, intervals + 1)
    u = grid_u(w)
    du_dw = 1.0 / (1.0 / (STEP_COEFF_PER_US * u * u) + 1.0 / STEP_CAP_US)
    simpson = np.ones(intervals + 1)
    simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
    weights = simpson * du_dw * (w[1] - w[0]) / 3.0
    energies = [f"{e:.11e}" for e in energy_ev(u, FLIGHT_PATH_M)]
    return energies, weights


def bin_edges() -> tuple[np.ndarray, np.ndarray]:
    """Return the scored bins' edges in µs and centre energies in eV, both in time order."""
    k, edges = 0, []
    while True:
        lo, hi = (EDGE0_CENTI_US + WIDTH_CENTI_US * k) / 100, (EDGE0_CENTI_US
                                                              + WIDTH_CENTI_US * (k + 1)) / 100
        centre = energy_ev((lo + hi) / 2, FLIGHT_PATH_M)
        if centre < E_LOW_EV:
            break
        if centre <= E_LAST_EV:
            edges.append(k)
        k += 1
    first, last = edges[0], edges[-1] + 1
    t = np.array([(EDGE0_CENTI_US + WIDTH_CENTI_US * j) / 100 for j in range(first, last + 1)])
    centres = energy_ev((t[:-1] + t[1:]) / 2, FLIGHT_PATH_M)
    return t, centres


def counts(energies: np.ndarray, weights: np.ndarray, transmission: np.ndarray,
           edges: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return the expected counts in each bin for numbers ``p``."""
    from counts_kernel import fold

    u = flight_time_us(energies, p[12])
    w = weights * p[12] / FLIGHT_PATH_M
    return fold(energies, u, w, transmission, edges, p, SPAN_EV[0], SPAN_EV[1], TAIL_LENGTHS)


def sammy_transmission(exe: Path, work: Path, jobs: int, keep: bool) -> dict:
    """Run SAMMY's free gas without instrument at the grid's energies; return
    ``{kelvin: transmission}`` in time order."""
    import concurrent.futures

    import generate_phase1 as g

    awr = g.read_awr()
    energies = nodes(0)[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {k: pool.submit(g.run_case, exe, work, f"p1b_T{k}_g0", {
            "kelvin": k, "model": "free", "energies": energies,
            "inp": g.sammy_input(awr, k, NXTRA_P1B, "free")}, keep)
                   for k in TEMPERATURES}
        results = {k: future.result() for k, future in futures.items()}
    builds = {build for _, build in results.values()}
    if builds != {g.SAMMY_BUILD}:
        raise SystemExit(f"SAMMY builds: {builds}")
    return {key: np.array([float(x) for x in values]) for key, (values, _) in results.items()}


def jacobian(grids, edges) -> np.ndarray:
    """Return the central-difference derivative of the counts with respect to each number.

    The step starts at 1e-4 of the value (1e-4 µs for t0) and halves until two steps agree to
    1e-3 of the largest derivative; when the halved grid is given, the same step must agree
    there too.

    Parameters
    ----------
    grids
        ``(energies, weights, transmission)`` on the grid, then on the halved grid or ``None``.
    edges
        The scored bins' edges in µs.
    """
    columns = []
    for i, name in enumerate(NAMES):
        step = 1e-4 if name == "t0" else 1e-4 * abs(NOMINAL[i])

        def difference(h, level=0):
            energies, weights, transmission = grids[level]
            up, down = NOMINAL.copy(), NOMINAL.copy()
            up[i] += h
            down[i] -= h
            return (counts(energies, weights, transmission, edges, up)
                    - counts(energies, weights, transmission, edges, down)) / (2.0 * h)

        coarse = difference(step)
        for _ in range(12):
            step /= 2.0
            fine = difference(step)
            change = np.max(np.abs(fine - coarse))
            if change <= 1e-3 * np.max(np.abs(fine)):
                break
            coarse = fine
        else:
            raise SystemExit(f"{name}: the central difference did not converge")
        grid_change = np.nan
        if grids[1] is not None:
            grid_change = np.max(np.abs(difference(step, 1) - fine))
            if grid_change > 1e-3 * np.max(np.abs(fine)):
                raise SystemExit(f"{name}: the derivative moves by {grid_change:.2e}"
                                 " on the halved grid")
        print(f"{name}: step {step:.3e}, step change {change:.2e}, grid change {grid_change:.2e}"
              f" of {np.max(np.abs(fine)):.3e}", flush=True)
        columns.append(fine)
    return np.array(columns).T


def write(exe: Path, work: Path, jobs: int, keep: bool) -> None:
    """Run SAMMY, fold its transmission and write the count references."""
    import hashlib

    import generate_phase1 as g

    md5 = hashlib.md5(exe.read_bytes()).hexdigest()
    if md5 != g.SAMMY_MD5:
        raise SystemExit(f"{exe} has MD5 {md5}, not that of SAMMY {g.SAMMY_BUILD} {g.SAMMY_COMMIT}")
    transmission = sammy_transmission(exe, work, jobs, keep)
    edges, centres = bin_edges()
    grid = {level: nodes(level) for level in (0, 1)}
    energy = {level: np.array([float(e) for e in grid[level][0]]) for level in (0, 1)}

    def at(level, values):
        return counts(energy[level], grid[level][1], values, edges, NOMINAL)

    ones = {level: np.ones(len(energy[level])) for level in (0, 1)}
    open_counts = at(0, ones[0])
    eps = {"open": np.max(np.abs(open_counts - at(1, ones[1])))}
    if eps["open"] > 1e-7 * np.max(open_counts):
        raise SystemExit(f"open counts move by {eps['open']:.2e} on the halved grid")
    sample = {k: at(0, transmission[k]) for k in TEMPERATURES}
    stamp = [f"# SAMMY {g.SAMMY_BUILD} (commit {g.SAMMY_COMMIT}), executable MD5 {md5}; free gas",
             "# without instrument at the integration grid's energies, NXTRA "
             f"{NXTRA_P1B}. Written by counts_reference.py write."]
    # Every output is computed and checked before any file is replaced.
    blocks = [jacobian([(energy[level], grid[level][1], ones[level]) for level in (0, 1)], edges)]
    for k in ("300", "1100"):
        blocks.append(jacobian([(energy[0], grid[0][1], transmission[k]), None], edges))
    table = np.hstack(blocks)
    labels = [f"open_{n}" for n in NAMES] + [f"T{k}_{n}" for k in ("300", "1100") for n in NAMES]
    rows = [" ".join([f"{c:.11e}", f"{o:.15e}", *(f"{sample[k][j]:.15e}" for k in TEMPERATURES)])
            for j, (c, o) in enumerate(zip(centres, open_counts))]
    (HERE / "p1b_counts.txt").write_text("\n".join([
        *stamp,
        "# Expected counts per bin, beam 1e4 (tau/100 us)^-0.8 per us, n = 7.04e-4 atoms/b,",
        "# the pulse numbers " + ", ".join(f"{n} {v:g}" for n, v in zip(NAMES, NOMINAL)) + ".",
        "# Columns: e_ev open " + " ".join(f"T{k}" for k in TEMPERATURES), *rows]) + "\n")
    (HERE / "p1b_counts_eps.txt").write_text("\n".join([
        *stamp,
        "# The largest change of the open counts when the integration grid is halved.",
        "# Columns: run (open or temperature in K), eps.",
        *(f"{key} {value:.6e}" for key, value in eps.items())]) + "\n")
    (HERE / "p1b_jacobian.txt").write_text("\n".join([
        *stamp,
        "# Central differences of the expected counts at the numbers of p1b_counts.txt,",
        "# the beam held in clock time and the energies fixed.",
        "# Columns: e_ev " + " ".join(labels),
        *(" ".join([f"{c:.11e}", *(f"{x:.12e}" for x in row)])
          for c, row in zip(centres, table))]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="compare the closed-form CDF with direct convolution")
    run = sub.add_parser("write", help="run SAMMY and write the count references")
    run.add_argument("exe", type=Path, help="SAMMY executable")
    run.add_argument("work", type=Path, help="directory for the SAMMY runs")
    run.add_argument("--jobs", type=int, default=1, help="SAMMY runs at the same time")
    run.add_argument("--keep", action="store_true", help="read again finished SAMMY runs")
    args = parser.parse_args()
    if args.command == "check":
        self_check()
    else:
        write(args.exe.resolve(), args.work.resolve(), args.jobs, args.keep)


if __name__ == "__main__":
    main()
