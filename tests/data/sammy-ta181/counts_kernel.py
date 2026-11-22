"""Expected counts per bin from the closed-form delay CDF, summed over the integration grid
in parallel."""

import math

import numpy as np
from numba import njit, prange

FACTORIAL = np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0])


@njit(cache=True)
def delay_terms(alpha, beta, r, f, tau):
    """Return ``c0``, the decay rates ``[α, β, 1/τ]`` and coefficients ``c[rate, n]`` with
    the delay CDF before the proton pulse ``c0 + Σ c·y^n·e^(−λy)``, y > 0."""
    lam = np.array([alpha, beta, 1.0 / tau])
    # Ikeda and Carpenter 1985, Eq. 10′, as c·t^n·e^(−λt): (rate index, n, c).
    k = alpha * r * alpha**2 * beta / (alpha - beta) ** 3
    d_rate = np.array([0, 1, 0, 0, 0])
    d_n = np.array([2, 0, 0, 1, 2])
    d_c = np.array([(1.0 - r) * alpha**3 / 2.0, k, -k, -k * (alpha - beta),
                    -k * (alpha - beta) ** 2 / 2.0])
    c0 = 0.0
    cdf = np.zeros((3, 3))
    for i in range(5):
        n, li, c = d_n[i], d_rate[i], d_c[i]
        c0 += c * FACTORIAL[n] / lam[li] ** (n + 1)
        for kk in range(n + 1):
            cdf[li, kk] -= c * FACTORIAL[n] / (FACTORIAL[kk] * lam[li] ** (n - kk + 1))
    coef = (1.0 - f) * cdf
    coef[2, 0] -= f * c0
    for li in range(2):
        mu = 1.0 / tau - lam[li]
        for n in range(3):
            c = cdf[li, n]
            if c == 0.0:
                continue
            for kk in range(n + 1):
                coef[li, n - kk] += (f * c / tau * (-1.0) ** kk * FACTORIAL[n]
                                     / (FACTORIAL[n - kk] * mu ** (kk + 1)))
            coef[2, 0] -= f * c / tau * (-1.0) ** n * FACTORIAL[n] / mu ** (n + 1)
    return c0, lam, coef


@njit(cache=True)
def q_part(n, lam, y):
    """The polynomial Q with e^(−λy)·Q(y) the exponential part of ∫_0^y (y − w) w^n e^(−λw) dw."""
    second = 0.0
    for k in range(n + 2):
        second += FACTORIAL[n + 1] / (FACTORIAL[k] * lam ** (n - k + 2)) * y**k
    first = 0.0
    for k in range(n + 1):
        first += FACTORIAL[n] / (FACTORIAL[k] * lam ** (n - k + 1)) * y**k
    return second - y * first


@njit(cache=True)
def second_antiderivative(c0, lam, coef, y):
    """J(y) = ∫_0^y (y − w) F(w) dw of the delay CDF F, 0 for y ≤ 0."""
    if y <= 0.0:
        return 0.0
    total = c0 * y * y / 2.0
    for li in range(3):
        for n in range(3):
            c = coef[li, n]
            if c != 0.0:
                linear = (y * FACTORIAL[n] / lam[li] ** (n + 1)
                          - FACTORIAL[n + 1] / lam[li] ** (n + 2))
                total += c * (linear + math.exp(-lam[li] * y) * q_part(n, lam[li], y))
    return total


@njit(cache=True)
def arrival_cdf(s, c0, lam, coef, h):
    """The chance that the whole delay, proton triangle of FWHM h included, is at most s."""
    if s <= -h:
        return 0.0
    if s < h:
        return (second_antiderivative(c0, lam, coef, s + h)
                - 2.0 * second_antiderivative(c0, lam, coef, s)
                + second_antiderivative(c0, lam, coef, s - h)) / (h * h)
    total = c0
    for li in range(3):
        decay = math.exp(-lam[li] * s)
        for n in range(3):
            c = coef[li, n]
            if c == 0.0:
                continue
            if n == 0:
                total += c * decay * (2.0 * math.sinh(lam[li] * h / 2.0) / lam[li]) ** 2 / (h * h)
            else:
                total += c * decay * (q_part(n, lam[li], s + h) * math.exp(-lam[li] * h)
                                      - 2.0 * q_part(n, lam[li], s)
                                      + q_part(n, lam[li], s - h) * math.exp(lam[li] * h)) / (h * h)
    return total


@njit(parallel=True, cache=True)
def fold(energies, u, w, transmission, edges, p, span_lo, span_hi, tail):
    """Return Σ_i w_i φ(u_i + t0) T_i [F(t_{j+1} − t0 − u_i) − F(t_j − t0 − u_i)] for each bin j."""
    nodes, bins = len(energies), len(edges) - 1
    chunks = 64
    partial = np.zeros((chunks, bins))
    h = math.sqrt(p[5])
    t0 = p[11]
    root_lo, root_hi = math.sqrt(span_lo), math.sqrt(span_hi)
    for chunk in prange(chunks):
        for i in range(chunk, nodes, chunks):
            e = energies[i]
            alpha = p[0] * e ** p[9] + p[1]
            beta = p[2] * e ** p[10] + p[3]
            r = p[4] + (p[6] - p[4]) * (math.sqrt(e) - root_lo) / (root_hi - root_lo)
            c0, lam, coef = delay_terms(alpha, beta, r, p[7], p[8])
            weight = w[i] * 1e4 * ((u[i] + t0) / 100.0) ** -0.8 * transmission[i]
            s_max = tail / min(alpha, min(beta, 1.0 / p[8]))
            first = np.searchsorted(edges, u[i] + t0 - h, side="right")
            last = np.searchsorted(edges, u[i] + t0 + s_max, side="right")
            if first >= len(edges) or last == 0:
                continue
            below = 0.0
            for k in range(first, min(last, len(edges))):
                here = arrival_cdf(edges[k] - t0 - u[i], c0, lam, coef, h)
                if k >= 1:
                    partial[chunk, k - 1] += weight * (here - below)
                below = here
            if last < len(edges):
                partial[chunk, last - 1] += weight * (c0 - below)
    return partial.sum(axis=0)
