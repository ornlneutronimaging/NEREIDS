"""Reference solutions for the Poisson fitter, computed with SciPy.

Writes cases.json next to this file.  Each case holds a model family and its
fixed data, the observed counts, box bounds, a start, and the reference
minimum of the half Poisson deviance found by SciPy's L-BFGS-B, with the
error bars at that minimum.

Run with: pixi run python crates/nereids-fitting/tests/poisson_oracle/generate.py
"""

import json
import pathlib

import numpy as np
from scipy.optimize import minimize

EPS = np.finfo(float).eps
DEGENERATE_EIGENVALUE = 1e-12
CERTIFIED_DECREMENT = 1e-8


def predict(case, theta):
    f = case["family"]
    if f == "decay":
        t = np.asarray(case["t"])
        a, b = theta
        mu = a * np.exp(-b * t)
        jac = np.column_stack([np.exp(-b * t), -a * t * np.exp(-b * t)])
    elif f == "split":
        t = np.asarray(case["t"])
        a, b, c = theta
        e = np.exp(-c * t)
        mu = (a + b) * e
        jac = np.column_stack([e, e, -(a + b) * t * e])
    elif f == "correlated":
        u, w = np.asarray(case["u"]), np.asarray(case["w"])
        amp, a, b = theta
        e = np.exp(-a * u - b * (u + w))
        mu = amp * e
        jac = np.column_stack([e, -amp * u * e, -amp * (u + w) * e])
    elif f == "resonance":
        energy, flux = np.asarray(case["energy"]), np.asarray(case["flux"])
        n, temp, bkg = theta
        width = case["width_300k"] * np.sqrt(temp / 300.0)
        delta = energy - case["center"]
        sigma = case["peak"] * np.exp(-(delta**2) / (2.0 * width**2))
        dsigma_dt = sigma * delta**2 / width**3 * case["width_300k"] / (2.0 * np.sqrt(300.0 * temp))
        trans = np.exp(-n * sigma)
        mu = flux * trans + bkg
        jac = np.column_stack([-flux * trans * sigma, -flux * trans * n * dsigma_dt, np.ones_like(energy)])
    elif f == "linear":
        x, offset = np.asarray(case["x"]), np.asarray(case["offset"])
        mu = offset + x @ np.asarray(theta)
        jac = x.copy()
    else:
        raise ValueError(f)
    return mu, jac


def half_deviance(y, mu):
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0.0):
        return np.inf
    ratio = np.where(y > 0, y / mu, 1.0)
    return float(np.sum(mu - y + y * np.log(ratio)))


def objective(case, y, theta):
    mu, jac = predict(case, theta)
    value = half_deviance(y, mu)
    if not np.isfinite(value):
        return np.inf, np.zeros_like(theta)
    return value, jac.T @ (1.0 - y / mu)


def check_jacobian(case, theta):
    mu, jac = predict(case, theta)
    for j in range(len(theta)):
        h = 1e-6 * max(1.0, abs(theta[j]))
        up, down = np.array(theta, float), np.array(theta, float)
        up[j] += h
        down[j] -= h
        numeric = (predict(case, up)[0] - predict(case, down)[0]) / (2 * h)
        scale = np.max(np.abs(jac[:, j])) + 1e-300
        assert np.max(np.abs(numeric - jac[:, j])) <= 1e-5 * scale, (case["name"], j)


def active_set(theta, grad, lower, upper):
    at_lower = np.isfinite(lower) & (theta == lower) & (grad > 0)
    at_upper = np.isfinite(upper) & (theta == upper) & (grad < 0)
    return at_lower | at_upper


def scaled_weighted_jacobian(case, theta, columns):
    mu, jac = predict(case, theta)
    weighted = jac[:, columns] / np.sqrt(mu)[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    keep = norms > 0
    return mu, weighted[:, keep] / norms[keep], norms, keep


def newton_decrement(case, y, theta, lower, upper):
    mu, jac = predict(case, theta)
    grad = jac.T @ (1.0 - y / mu)
    free = np.flatnonzero(~active_set(theta, grad, lower, upper))
    _, scaled, _, _ = scaled_weighted_jacobian(case, theta, free)
    residual = (mu - y) / np.sqrt(mu)
    if scaled.shape[1] == 0:
        return 0.0
    u, s, _ = np.linalg.svd(scaled, full_matrices=False)
    spanned = s > EPS * max(scaled.shape) * s.max()
    return 0.5 * float(np.sum((u[:, spanned].T @ residual) ** 2))


def polish(case, y, theta, lower, upper):
    """Poisson least-squares iterations (Fisher scoring) with step halving,
    solved by LAPACK; used only when no parameter is on a bound."""
    columns = np.arange(len(theta))
    for _ in range(100):
        mu, scaled, norms, keep = scaled_weighted_jacobian(case, theta, columns)
        value = half_deviance(y, mu)
        residual = (mu - y) / np.sqrt(mu)
        step = np.zeros_like(theta)
        solution = np.linalg.lstsq(scaled, residual, rcond=EPS * max(scaled.shape))[0]
        step[keep] = solution / norms[keep]
        t = 1.0
        while t > 1e-18:
            trial = theta - t * step
            if np.all((trial > lower) & (trial < upper)) and half_deviance(y, predict(case, trial)[0]) < value:
                theta = trial
                break
            t /= 2.0
        else:
            break
    return theta


def error_bars(case, theta, lower, upper):
    mu, jac = predict(case, theta)
    on_bound = (np.isfinite(lower) & (theta == lower)) | (np.isfinite(upper) & (theta == upper))
    interior = np.flatnonzero(~on_bound)
    weighted = jac[:, interior] / np.sqrt(mu)[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    keep = norms > 0
    cols = interior[keep]
    scaled = weighted[:, keep] / norms[keep]
    sigma = [None] * len(theta)
    covariance = [[None] * len(theta) for _ in theta]
    if len(cols):
        _, s, vt = np.linalg.svd(scaled, full_matrices=True)
        v = vt.T
        values = np.zeros(len(cols))
        values[: len(s)] = s
        determined = values**2 >= DEGENERATE_EIGENVALUE
        floor = max(scaled.shape) * EPS
        resolved = [np.sum(v[i, ~determined] ** 2) <= floor for i in range(len(cols))]
        for i, col in enumerate(cols):
            for j, row in enumerate(cols):
                if resolved[i] and resolved[j]:
                    part = np.sum(v[i, determined] * v[j, determined] / values[determined] ** 2)
                    covariance[col][row] = float(part / norms[keep][i] / norms[keep][j])
            if resolved[i]:
                sigma[col] = float(np.sqrt(np.sum(v[i, determined] ** 2 / values[determined] ** 2)) / norms[keep][i])
    return on_bound.tolist(), sigma, covariance


def reference(case, y, starts, truth, lower, upper):
    mu, jac = predict(case, np.asarray(truth, float))
    scale = 1.0 / np.sqrt(np.maximum(np.sum(jac**2 / mu[:, None], axis=0), 1e-300))
    bounds = [(None if not np.isfinite(lo) else lo / s, None if not np.isfinite(hi) else hi / s)
              for lo, hi, s in zip(lower, upper, scale)]
    best_theta, best_value = None, np.inf
    for start in starts:
        z = np.clip(np.asarray(start, float), lower, upper) / scale
        for _ in range(4):
            res = minimize(
                lambda zz: (lambda v, g: (v, g * scale))(*objective(case, y, zz * scale)),
                z,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 50000, "maxfun": 100000, "maxcor": 30},
            )
            z = res.x
        theta = np.clip(z * scale, lower, upper)
        value = half_deviance(y, predict(case, theta)[0])
        if value < best_value:
            best_theta, best_value = theta, value
    return best_theta, best_value


def case_record(case, y, truth, start, lower, upper, rng_starts):
    check_jacobian(case, np.asarray(truth, float))
    starts = [start, truth] + rng_starts
    theta, value = reference(case, y, starts, truth, lower, upper)
    if not np.any(theta == lower) and not np.any(theta == upper):
        theta = polish(case, y, theta, lower, upper)
        value = half_deviance(y, predict(case, theta)[0])
    decrement = newton_decrement(case, y, theta, lower, upper)
    certified = bool(np.isfinite(value) and decrement < CERTIFIED_DECREMENT)
    on_bound, sigma, covariance = error_bars(case, theta, lower, upper)
    record = dict(
        name=case["name"],
        observed=y.tolist(),
        truth=list(map(float, truth)),
        start=list(map(float, start)),
        lower=[float(v) if np.isfinite(v) else None for v in lower],
        upper=[float(v) if np.isfinite(v) else None for v in upper],
        reference=theta.tolist(),
        reference_deviance=value,
        reference_decrement=float(decrement),
        certified=certified,
        on_bound=on_bound,
        sigma=sigma,
        covariance=covariance,
    )
    return record


def main():
    rng = np.random.default_rng(20260924)
    records = []
    models = []

    def add(case, truth, lower, upper, starts, seeds):
        models.append({k: v for k, v in case.items() if k != "name"})
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        mu = predict(case, np.asarray(truth, float))[0]
        for seed in seeds:
            y = np.random.default_rng(seed).poisson(mu).astype(float)
            for k, start in enumerate(starts):
                jitter = [np.clip(np.asarray(truth) * rng.uniform(0.7, 1.3, len(truth)), lower, upper).tolist()]
                named = dict(case, name=f"{case['name']}/seed{seed}/start{k}")
                records.append(dict(case_record(named, y, truth, start, lower, upper, jitter), model=len(models) - 1))

    inf = np.inf
    t = np.linspace(0.0, 4.0, 25).tolist()
    for amp in (5.0, 100.0, 1.0e5):
        add(dict(name=f"decay/amp{amp:g}", family="decay", t=t), [amp, 1.5], [0, 0], [inf, inf],
            [[amp * 0.5, 0.5], [amp * 3.0, 4.0], [amp * 0.01, 0.01]], range(1, 7))
    add(dict(name="decay/upper-a", family="decay", t=t), [100.0, 1.5], [0, 0], [80.0, inf],
        [[50.0, 1.0], [80.0, 3.0]], range(1, 7))
    add(dict(name="decay/lower-b", family="decay", t=t), [100.0, 1.5], [0, 1.8], [inf, inf],
        [[50.0, 1.8], [150.0, 3.0]], range(1, 7))

    u = np.linspace(0.0, 2.0, 200)
    for eps in (0.1, 0.01, 0.001):
        for amp in (1.0e3, 1.0e5):
            add(dict(name=f"correlated/eps{eps:g}/amp{amp:g}", family="correlated", u=u.tolist(),
                     w=(eps * np.cos(7.0 * u)).tolist()),
                [amp, 0.5, 0.5], [-inf, -inf, -inf], [inf, inf, inf],
                [[amp * 0.8, 0.3, 0.7], [amp * 1.5, 1.0, 0.0]], range(1, 5))

    energy = np.linspace(4.0, 8.0, 150)
    for flux in (10.0, 1.0e3, 1.0e6):
        for truth in ([0.02, 300.0, 0.0], [0.02, 800.0, 0.05 * flux], [0.2, 300.0, 0.0]):
            add(dict(name=f"resonance/flux{flux:g}/n{truth[0]:g}/T{truth[1]:g}/B{truth[2]:g}", family="resonance",
                     energy=energy.tolist(), flux=(flux * (energy / 6.0) ** -0.5).tolist(),
                     center=6.0, peak=110.0, width_300k=0.12),
                truth, [0, 1, 0], [inf, 5000, inf],
                [[0.01, 200.0, 0.0], [0.04, 1000.0, 0.1 * flux], [0.005, 3000.0, 0.0]], range(1, 4))

    add(dict(name="split", family="split", t=t), [400.0, 600.0, 1.5], [0, 0, 0], [inf, inf, inf],
        [[300.0, 300.0, 1.0], [900.0, 50.0, 3.0]], range(1, 5))
    for delta in (1e-7, 1e-3):
        add(dict(name=f"linear/null-delta{delta:g}", family="linear",
                 x=[[1.0, 0.0, delta], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]], offset=[100.0, 100.0, 100.0]),
            [0.0, 0.0, 0.0], [-inf, -inf, -inf], [inf, inf, inf], [[5.0, -5.0, 3.0]], range(1, 4))
    for delta in (1e-7, 1e-5):
        signs = np.tile([1.0, -1.0], 10)
        add(dict(name=f"linear/near-twin-delta{delta:g}", family="linear",
                 x=np.column_stack([np.full(20, 1.0e3), 1.0e3 * (1.0 + signs * delta)]).tolist(),
                 offset=[1.0e6] * 20),
            [0.0, 0.0], [-inf, -inf], [inf, inf], [[0.0, 0.0], [300.0, -300.0]], range(1, 4))
    x = rng.uniform(0.0, 1.0, (40, 4))
    add(dict(name="linear/nonneg", family="linear", x=x.tolist(), offset=[1.0] * 40),
        [20.0, 0.0, 15.0, 0.0], [0, 0, 0, 0], [inf, inf, inf, inf],
        [[5.0, 5.0, 5.0, 5.0], [50.0, 0.0, 0.0, 50.0]], range(1, 7))

    out = pathlib.Path(__file__).with_name("cases.json")
    out.write_text(json.dumps({"models": models, "cases": records}))
    certified = sum(r["certified"] for r in records)
    print(f"{len(records)} cases, {certified} certified by the reference, written to {out}")
    for r in records:
        if not r["certified"]:
            print(f"  not certified: {r['name']} decrement {r['reference_decrement']:.3e}")


if __name__ == "__main__":
    main()
