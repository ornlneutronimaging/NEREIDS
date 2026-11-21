#!/usr/bin/env python3
"""Lattice-broadened Ta-181 transmission without instrument, independent of NEREIDS's broadening.

Lamb's incoherent harmonic model with a continuous Debye spectrum, theta_D = 240 K, recoil
R = E/A, A the ENDF AWR. The zero-phonon line and the one-phonon term are integrated in
energy; only the multi-phonon remainder goes through a Fourier transform. The unbroadened
total cross section comes from NEREIDS (``unbroadened_total`` example) on a uniform grid.
Run ``lattice_reference.py -h`` for the commands.

References: W. E. Lamb, Phys. Rev. 55, 190 (1939); F. H. Fröhner, JEFF Report 18 (2000),
Eqs. 241-243.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
from pathlib import Path

import numpy as np

from generate import HERE, THICKNESS_PER_B, read_awr
from generate_phase1 import E_LAST_EV, bin_energies

BOLTZMANN_EV_PER_K = 8.617_333_262e-5
DEBYE_K = 240.0
OMEGA_D = BOLTZMANN_EV_PER_K * DEBYE_K
TEMPERATURES = ("300.79", "301.29", "301.79", "302.29", "302.79",
                "1092", "1092.5", "1093", "1093.5", "1094")
# The unbroadened grid: SIGMA_FIRST + k·SIGMA_STEP eV, k < SIGMA_COUNT.
SIGMA_FIRST, SIGMA_STEP = 0.05, 1e-5
SIGMA_COUNT = int(round((205.0 - SIGMA_FIRST) / SIGMA_STEP)) + 1
# Above this 2W the zero- and one-phonon weights, e^(−2W)·(1 + 2W), are below 1e-18.
SMOOTH_2W = 50.0
# G(t) table: fine spacing where the multi-phonon kernel is wide, coarse where only the
# one-phonon tail remains.
T_SPLIT, T_FINE, T_COARSE, T_END = 2000.0, 0.02, 0.5, 70000.0


def gauss_legendre(a: float, b: float, panels: int, order: int = 16):
    """Nodes and weights of ``panels`` equal Gauss–Legendre panels on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


def density(omega):
    """Debye's spectrum 3ω²/ω_D³, 1/eV."""
    return 3.0 * omega**2 / OMEGA_D**3


class Lattice:
    """The spectrum's sums at one temperature: G(0), kT_eff and G(t) on a table."""

    def __init__(self, kelvin: float):
        self.kt = BOLTZMANN_EV_PER_K * kelvin
        om, w = gauss_legendre(0.0, OMEGA_D, 64)
        coth = 1.0 / np.tanh(om / (2.0 * self.kt))
        a = w * density(om) / om
        self.g0 = float(np.sum(a * coth))
        self.kt_eff = 0.5 * float(np.sum(a * om * om * coth))
        self.t = np.concatenate([np.arange(0.0, T_SPLIT, T_FINE),
                                 np.arange(T_SPLIT, T_END + T_COARSE, T_COARSE)])
        self.g = np.empty(len(self.t), complex)
        for lo, hi in ((0, int(T_SPLIT / T_FINE)), (int(T_SPLIT / T_FINE), len(self.t))):
            # One radian of ω·t per panel at the segment's last time.
            panels = max(int(math.ceil(OMEGA_D * self.t[hi - 1])), 1)
            om, w = gauss_legendre(0.0, OMEGA_D, panels)
            coth = 1.0 / np.tanh(om / (2.0 * self.kt))
            a = w * density(om) / om
            for s in range(lo, hi, 512):
                tt = self.t[s:min(s + 512, hi), None]
                self.g[s:s + len(tt)] = (a * (coth * np.cos(om * tt)
                                              - 1j * np.sin(om * tt))).sum(axis=1)

    def g_at(self, t: np.ndarray) -> np.ndarray:
        """G(t) by cubic interpolation of the table."""
        if t[-1] > T_END:
            raise SystemExit(f"G(t) is tabulated to {T_END}, needed to {t[-1]}")
        return cubic(self.t, self.g.real, t) + 1j * cubic(self.t, self.g.imag, t)

    def one_phonon(self, recoil: float, refine: int = 1):
        """Nodes ε and weights of e^(−2W)·R·g1(ε), g1 the one-phonon density, on [−ω_D, ω_D]."""
        # Panels of at most a quarter of the narrowest width resolve σ along ε.
        panels = refine * max(4, int(math.ceil(4.0 * OMEGA_D / NARROWEST[0])))
        om, w = gauss_legendre(0.0, OMEGA_D, panels)
        coth = 1.0 / np.tanh(om / (2.0 * self.kt))
        base = density(om) / om * w
        scale = math.exp(-recoil * self.g0) * recoil
        eps = np.concatenate([om, -om])
        weights = scale * np.concatenate([base * 0.5 * (coth + 1.0), base * 0.5 * (coth - 1.0)])
        return eps, weights


def cubic(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Catmull–Rom interpolation of y(x) at ``at``; x piecewise uniform."""
    k = np.clip(np.searchsorted(x, at) - 1, 0, len(x) - 3)
    x0, x1 = x[k], x[k + 1]
    s = (at - x0) / (x1 - x0)
    # Neighbours across a spacing change are rescaled to the local step; the first interval
    # takes a one-sided slope from a mirrored point.
    first = k == 0
    km = np.maximum(k - 1, 0)
    y0, y1, y2 = y[k], y[k + 1], y[k + 2]
    h0, h2 = x1 - x0, x[k + 2] - x1
    ym = np.where(first, 2 * y0 - y1, y[km])
    hm = np.where(first, h0, x0 - x[km])
    m0 = (y1 - ym) / (h0 + hm) * h0
    m1 = (y2 - y0) / (h2 + h0) * h0
    return ((2 * s**3 - 3 * s**2 + 1) * y0 + (s**3 - 2 * s**2 + s) * m0
            + (-2 * s**3 + 3 * s**2) * y1 + (s**3 - s**2) * m1)


def multi_phonon(lattice: Lattice, recoil: float, refine: int):
    """Energies ε and values of P(ε) − e^(−2W)·[δ(ε) + R·g1(ε)] on a periodic grid, step."""
    sigma_k = math.sqrt(2.0 * recoil * lattice.kt_eff)
    two_w = recoil * lattice.g0
    step = min(sigma_k / 40.0, NARROWEST[0] * math.pi / 60.0)
    if two_w < SMOOTH_2W:
        step = min(step, OMEGA_D / 200.0)
    span = 2.0 * (recoil + 14.0 * sigma_k + 12.0 * OMEGA_D)
    n = 1 << int(math.ceil(math.log2(span / step)))
    # Refined: half the step over twice the period.
    step /= refine
    n *= refine * refine
    dt = 2.0 * math.pi / (n * step)
    t = np.arange(n // 2) * dt
    g = lattice.g_at(t)
    zero = math.exp(-two_w)
    f = np.exp(recoil * (g - lattice.g0)) - zero * (1.0 + recoil * g)
    buffer = np.zeros(n, complex)
    buffer[:n // 2] = f
    buffer[n - n // 2 + 1:] = np.conj(f[1:])[::-1]
    p = (dt / (2.0 * math.pi)) * n * np.fft.ifft(buffer).real
    eps = np.fft.fftfreq(n, d=1.0 / (n * step))
    return eps, p, step


NARROWEST = [math.inf]
# The largest kernel weight a target left off the cross-section grid.
DROPPED = [0.0]
# Set in each worker process by `start_worker`.
STATE: dict = {}


def start_worker(sigma_path: Path, narrowest: float) -> None:
    """Load what `column` reads."""
    NARROWEST[0] = narrowest
    STATE.update(sigma0=load_sigma(sigma_path), awr=read_awr(),
                 energies=[float(e) for e in bin_energies(E_LAST_EV)])


def column(task) -> np.ndarray:
    """The transmission at every bin for ``(kelvin, refine)``."""
    kelvin, refine = task
    DROPPED[0] = 0.0
    lattice = Lattice(float(kelvin))
    values = np.exp(-THICKNESS_PER_B * np.array(
        [broadened(STATE["sigma0"], e, lattice, STATE["awr"], refine)
         for e in STATE["energies"]]))
    print(f"T {kelvin} K, refine {refine}: largest weight off the grid {DROPPED[0]:.1e}",
          flush=True)
    return values


def broadened(sigma0: np.ndarray, energy: float, lattice: Lattice, awr: float,
              refine: int = 1) -> float:
    """σ_L(E): the zero-phonon line, the one-phonon term and the multi-phonon remainder."""
    recoil = energy / awr
    # Published values interpolate σ every 2e-5 eV; the refined run every 1e-5 eV.
    stride = 2 if refine == 1 else 1
    table = sigma0[::stride]

    def sigma(e):
        x = (np.asarray(e) - SIGMA_FIRST) / (SIGMA_STEP * stride)
        if np.min(x) < 0.0 or np.max(x) > len(table) - 1:
            raise SystemExit(f"{energy} eV reads the cross section outside the grid")
        k = np.minimum(x.astype(np.int64), len(table) - 2)
        frac = x - k
        return (1.0 - frac) * table[k] + frac * table[k + 1]

    zero = math.exp(-recoil * lattice.g0)
    total = zero * float(sigma(np.array([energy + recoil]))[0])
    eps1, w1 = lattice.one_phonon(recoil, refine)
    total += float(np.sum(w1 * sigma(energy + recoil - eps1)))
    eps, p, step = multi_phonon(lattice, recoil, refine)
    at = energy + recoil - eps
    inside = (at >= SIGMA_FIRST) & (at <= SIGMA_FIRST + SIGMA_STEP * stride * (len(table) - 1))
    # Off-grid points carry FFT ripple, not kernel; σ is not tabulated there, so their skipped
    # weight, printed per column, is reported rather than bounded.
    dropped = step * float(np.sum(np.abs(p[~inside])))
    if dropped > 1e-7:
        raise SystemExit(f"{energy} eV: kernel weight {dropped:.1e} lies off the cross-section grid")
    DROPPED[0] = max(DROPPED[0], dropped)
    total += step * float(np.sum(p[inside] * sigma(at[inside])))
    return total


def kernel_moments(lattice: Lattice, recoil: float, refine: int = 1):
    """Mass, mean and the second to fourth cumulants of the whole kernel P(ε)."""
    zero = math.exp(-recoil * lattice.g0)
    eps1, w1 = lattice.one_phonon(recoil)
    eps, p, step = multi_phonon(lattice, recoil, refine)
    nodes = np.concatenate([[0.0], eps1, eps])
    weights = np.concatenate([[zero], w1, step * p])
    mass = weights.sum()
    mean = np.sum(weights * nodes) / mass
    c = [np.sum(weights * (nodes - mean) ** k) / mass for k in (2, 3, 4)]
    return mass, mean, c[0], c[1], c[2] - 3.0 * c[0] ** 2


def analytic_cumulants(lattice: Lattice, recoil: float):
    """κ_n = R·∫ρ·ω^(n−1)·[coth(ω/2kT) for even n, 1 for odd n] dω, n = 1..4."""
    om, w = gauss_legendre(0.0, OMEGA_D, 64)
    coth = 1.0 / np.tanh(om / (2.0 * lattice.kt))
    rho = w * density(om)
    return [recoil * float(np.sum(rho * om ** (n - 1) * (coth if n % 2 == 0 else 1.0)))
            for n in (1, 2, 3, 4)]


def load_sigma(path: Path) -> np.ndarray:
    sigma0 = np.fromfile(path, dtype="<f8")
    if len(sigma0) != SIGMA_COUNT:
        raise SystemExit(f"{path}: {len(sigma0)} values, expected {SIGMA_COUNT}")
    return sigma0


def check(narrowest: float) -> None:
    """Compare the kernel's moments with the spectrum's cumulants."""
    NARROWEST[0] = narrowest
    awr = read_awr()
    worst = 0.0
    for kelvin in (301.79, 1093.0):
        lattice = Lattice(kelvin)
        for energy in (0.5, 4.280412, 50.0, 197.5):
            r = energy / awr
            mass, mean, k2, k3, k4 = kernel_moments(lattice, r)
            a1, a2, a3, a4 = analytic_cumulants(lattice, r)
            errors = [abs(mass - 1.0), abs(mean / a1 - 1.0), abs(k2 / a2 - 1.0),
                      abs(k3 / a3 - 1.0), abs(k4 / a4 - 1.0)]
            if not all(math.isfinite(x) for x in errors):
                raise SystemExit(f"{kelvin} K, {energy} eV: a kernel moment is not finite")
            worst = max(worst, *errors[:3])
            print(f"{kelvin} K, {energy} eV: mass-1 {errors[0]:.1e}, mean {errors[1]:.1e},"
                  f" var {errors[2]:.1e}, k3 {errors[3]:.1e}, k4 {errors[4]:.1e} (relative)")
    if worst > 1e-8:
        raise SystemExit(f"a kernel's mass, mean or variance is off by {worst:.1e}")


def write(sigma_path: Path, narrowest: float, jobs: int) -> None:
    """Write the reference transmissions and their numerical error bound."""
    NARROWEST[0] = narrowest
    # Stops on a wrong-sized file before the workers start.
    load_sigma(sigma_path)
    awr = read_awr()
    energies = [float(e) for e in bin_energies(E_LAST_EV)]

    tasks = [(k, r) for k in TEMPERATURES for r in (1, 2)]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, initializer=start_worker,
            initargs=(sigma_path, narrowest)) as pool:
        results = dict(zip(tasks, pool.map(column, tasks)))
    stamp = ["# Lattice-broadened transmission without instrument, n = 7.04e-4 atoms/b, Debye",
             f"# spectrum theta_D = {DEBYE_K:g} K, recoil E/A, A = {awr}; unbroadened Ta-181 total",
             "# cross section from NEREIDS (ENDF/B-VIII.1 MAT 7328), interpolated every 2e-5 eV;",
             "# broadening independent of NEREIDS. Written by lattice_reference.py write."]
    rows = [" ".join([f"{e:.11e}", *(f"{results[(k, 1)][j]:.12e}" for k in TEMPERATURES)])
            for j, e in enumerate(energies)]
    (HERE / "p1c_lattice_reference.txt").write_text("\n".join(
        stamp + ["# Columns: e_ev " + " ".join(f"T{k}" for k in TEMPERATURES)] + rows) + "\n")
    eps = {k: float(np.max(np.abs(results[(k, 1)] - results[(k, 2)]))) for k in TEMPERATURES}
    (HERE / "p1c_lattice_reference_eps.txt").write_text("\n".join(
        ["# eps = max|T − T(refined)| over the bins; refined: Fourier step halved and period",
         "# doubled, one-phonon panels doubled, sigma interpolated every 1e-5 instead of 2e-5 eV.",
         "# Columns: temperature (K), eps."]
        + [f"{k} {v:.6e}" for k, v in eps.items()]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="compare kernel moments with the spectrum's cumulants")
    c.add_argument("narrowest", type=float, help="narrowest level width, eV")
    w = sub.add_parser("write", help="write the reference transmissions")
    w.add_argument("sigma", type=Path, help="unbroadened total cross section from the example")
    w.add_argument("narrowest", type=float, help="narrowest level width, eV")
    w.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "check":
        check(args.narrowest)
    else:
        write(args.sigma, args.narrowest, args.jobs)


if __name__ == "__main__":
    main()
