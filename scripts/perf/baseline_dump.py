"""Capture bit-exact baselines for A.1 / B.1 / B.2 VENUS real-data fits.

Produces a JSON report with IEEE-754 `.hex()` encodings of the converged
fit outputs, so subsequent runs can assert byte-for-byte equality against
the saved baseline.  Use this as the correctness gate whenever a
resolution / forward-model optimisation is in flight.

Usage:
    pixi run python scripts/perf/baseline_dump.py --out /tmp/baseline.json
    pixi run python scripts/perf/baseline_dump.py --verify /tmp/baseline.json

The verify mode runs the same configs and compares every `.hex()` pair;
it prints a diff on mismatch and exits non-zero.

Baselines captured here are the outputs of:
- A.1 aggregated LM+grouped+background+TZERO on VENUS Hf 120 min
- B.2 spatial KL+grouped+background on a 4×4 crop (pre-calibrated TZERO)

B.1 is skipped by default because the spatial LM+TZERO path takes ~90 s
for a 2×2 crop at max_iter=200.  Pass --include-b1 to add it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import nereids

ROOT = Path(__file__).resolve().parents[2]
H5 = ROOT / ".research/spatial-regularization/data/counts/resonance_data_2cm.h5"
RES_FILE = ROOT / "_fts_bl10_0p5meV_1keV_25pts.txt"

TEMP_K = 293.6
FLIGHT_PATH_M = 25.0
ENERGY_MIN, ENERGY_MAX = 7.0, 200.0


def _float_hex(x: float) -> str:
    return float(x).hex()


def _array_hex(arr: np.ndarray) -> list[str]:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return [float(v).hex() for v in flat]


def run_a1() -> dict:
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    # SAMMY REGION-equivalent: pass the full energy grid + per-bin
    # data to NEREIDS and let the LM/JP cost paths mask residuals to
    # `[ENERGY_MIN, ENERGY_MAX]` via `fit_energy_range`.  The model is
    # evaluated on the full grid so resolution broadening at the
    # boundaries is correct (#514) — equivalent to SAMMY's REGION cards
    # which apply a kernel-margin internally.  Pre-PR-#514 this script
    # cropped the data array directly, which truncated the kernel at
    # the upper boundary near 200 eV.
    E = np.ascontiguousarray(E_full)
    S3d = np.ascontiguousarray(S3d_raw).astype(np.float64)
    O3d = np.ascontiguousarray(O3d_raw).astype(np.float64)
    c = Q_s / Q_ob
    S_agg = S3d.sum(axis=(1, 2))
    O_agg = O3d.sum(axis=(1, 2))
    T_agg = S_agg / np.maximum(c * O_agg, 1.0)
    sigT_agg = T_agg * np.sqrt(
        1.0 / np.maximum(S_agg, 1.0) + 1.0 / np.maximum(O_agg, 1.0)
    )

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)

    t0 = time.time()
    r = nereids.fit_spectrum_typed(
        transmission=np.ascontiguousarray(T_agg),
        uncertainty=np.ascontiguousarray(sigT_agg),
        energies=E,
        solver="lm",
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[1.6e-4],
        max_iter=500,
        background=True,
        fit_energy_scale=True,
        t0_init_us=0.5,
        l_scale_init=1.005,
        energy_scale_flight_path_m=FLIGHT_PATH_M,
        resolution=res,
        fit_energy_range=(ENERGY_MIN, ENERGY_MAX),
    )
    wall = time.time() - t0
    return {
        "name": "A1_LM_grouped_background_tzero",
        "wall_s": wall,
        "iterations": int(r.iterations),
        "converged": bool(r.converged),
        "chi2r_hex": _float_hex(r.reduced_chi_squared),
        "densities_hex": _array_hex(np.asarray(r.densities)),
        "t0_us_hex": _float_hex(r.t0_us),
        "l_scale_hex": _float_hex(r.l_scale),
    }


def run_b2() -> dict:
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    CROP_Y0, CROP_Y1 = 252, 256
    CROP_X0, CROP_X1 = 252, 256
    # SAMMY REGION-equivalent: keep the FULL spectral axis, apply the
    # pre-calibrated TZERO transformation to the whole grid, and let
    # `fit_energy_range` mask the cost function (#514).  Spatial cropping
    # in (y, x) is unchanged.
    S3d = np.ascontiguousarray(
        S3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    O3d = np.ascontiguousarray(
        O3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    c = Q_s / Q_ob

    T0_US = 0.4809277146701285
    L_SCALE = 1.0052452911520162
    tof_factor = (0.5 * 1.67492749804e-27 / 1.602176634e-19) ** 0.5 * 1.0e6
    L_eff = FLIGHT_PATH_M * L_SCALE
    # Apply TZERO calibration to the full nominal grid.  The
    # transformation is monotonic, so the calibrated bins originally
    # corresponding to nominal `[ENERGY_MIN, ENERGY_MAX]` map to a
    # well-defined `[fit_min_cal, fit_max_cal]` range on the calibrated
    # axis.
    #
    # Guard: pre-PR-#519 the `mask = E_full >= 7.0` step filtered any
    # bin with E ≤ 0 before sqrt; with the mask gone we assert the
    # input axis is strictly positive so `sqrt(E_full)` can't inject
    # NaN into `E_cal_full` and silently extend the active mask.
    assert (E_full > 0).all(), "energy axis must be strictly positive"
    tof_full = tof_factor * FLIGHT_PATH_M / np.sqrt(E_full)
    tof_corr_full = tof_full - T0_US
    E_cal_full = np.ascontiguousarray((tof_factor * L_eff / tof_corr_full) ** 2)

    def _tzero_cal(e_nominal: float) -> float:
        tof = tof_factor * FLIGHT_PATH_M / np.sqrt(e_nominal)
        return float((tof_factor * L_eff / (tof - T0_US)) ** 2)

    fit_min_cal = _tzero_cal(ENERGY_MIN)
    fit_max_cal = _tzero_cal(ENERGY_MAX)

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)
    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)

    input_data = nereids.from_counts(S3d, O3d)

    t0 = time.time()
    r = nereids.spatial_map_typed(
        data=input_data,
        energies=E_cal_full,
        solver="kl",
        c=c,
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[1.6e-4],
        max_iter=200,
        background=True,
        resolution=res,
        dead_pixels=dead_pixels,
        fit_energy_range=(fit_min_cal, fit_max_cal),
    )
    wall = time.time() - t0
    dens = np.asarray(r.density_maps[0])
    conv = np.asarray(r.converged_map)
    return {
        "name": "B2_spatial_KL_grouped_background_4x4",
        "wall_s": wall,
        "n_pixels": int(dens.size),
        "converged_count": int(conv.sum()),
        "density_hex": _array_hex(dens),
        "converged_hex": _array_hex(conv.astype(np.float64)),
    }


def run_b1() -> dict:
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    CROP_Y0, CROP_Y1 = 254, 256
    CROP_X0, CROP_X1 = 254, 256
    # SAMMY REGION-equivalent: full nominal axis + `fit_energy_range`
    # mask in the LM cost path (#514).  TZERO is FITTED (not pre-
    # calibrated) here, so the axis stays nominal.
    E = np.ascontiguousarray(E_full)
    S3d = np.ascontiguousarray(
        S3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    O3d = np.ascontiguousarray(
        O3d_raw[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    ).astype(np.float64)
    c = Q_s / Q_ob
    T3d = S3d / np.maximum(c * O3d, 1.0)
    sig3d = T3d * np.sqrt(
        1.0 / np.maximum(S3d, 1.0) + 1.0 / np.maximum(O3d, 1.0)
    )

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)
    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)
    input_data = nereids.from_transmission(T3d, sig3d)

    t0 = time.time()
    r = nereids.spatial_map_typed(
        data=input_data,
        energies=E,
        solver="lm",
        temperature_k=TEMP_K,
        groups=[hf_group],
        initial_densities=[1.6e-4],
        max_iter=200,
        background=True,
        fit_energy_scale=True,
        t0_init_us=0.5,
        l_scale_init=1.005,
        energy_scale_flight_path_m=FLIGHT_PATH_M,
        resolution=res,
        dead_pixels=dead_pixels,
        fit_energy_range=(ENERGY_MIN, ENERGY_MAX),
    )
    wall = time.time() - t0
    dens = np.asarray(r.density_maps[0])
    conv = np.asarray(r.converged_map)
    return {
        "name": "B1_spatial_LM_grouped_background_tzero_2x2",
        "wall_s": wall,
        "n_pixels": int(dens.size),
        "converged_count": int(conv.sum()),
        "density_hex": _array_hex(dens),
        "converged_hex": _array_hex(conv.astype(np.float64)),
    }


def _run_periso_tzero(solver: str, crop_size: int, label: str, max_iter: int) -> dict:
    """Shared runner for per-isotope spatial fits with TZERO (B.3 / KL+periso+TZERO).

    Uses a `crop_size × crop_size` pixel window centred on (255, 255).
    4x4 default keeps the fixture under ~20 s for LM and under ~5 s for
    KL on the current branch.
    """
    with h5py.File(H5, "r") as f:
        E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
        S3d_raw = f["Hf/120min/counts/sample"][:][::-1, :, :]
        O3d_raw = f["Hf/open_beam/counts"][:][::-1, :, :]
        Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
        Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])
    y0 = 255 - crop_size // 2
    x0 = 255 - crop_size // 2
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    # SAMMY REGION-equivalent: full nominal axis + `fit_energy_range`
    # mask in the per-pixel cost path (#514).
    E = np.ascontiguousarray(E_full)
    S3d = np.ascontiguousarray(S3d_raw[:, y0:y1, x0:x1]).astype(np.float64)
    O3d = np.ascontiguousarray(O3d_raw[:, y0:y1, x0:x1]).astype(np.float64)
    c = Q_s / Q_ob

    hf_group = nereids.IsotopeGroup.natural(72)
    hf_group.load_endf()
    resonance_data_list = list(hf_group.resonance_data)
    members = hf_group.members
    initial_densities = [1.6e-4 * ratio for _, ratio in members]

    res = nereids.load_resolution(str(RES_FILE), FLIGHT_PATH_M)
    dead_pixels = np.zeros((S3d.shape[1], S3d.shape[2]), dtype=bool)

    if solver == "lm":
        T3d = S3d / np.maximum(c * O3d, 1.0)
        sig3d = T3d * np.sqrt(
            1.0 / np.maximum(S3d, 1.0) + 1.0 / np.maximum(O3d, 1.0)
        )
        input_data = nereids.from_transmission(T3d, sig3d)
    else:
        input_data = nereids.from_counts(S3d, O3d)

    kwargs = dict(
        data=input_data,
        energies=E,
        isotopes=resonance_data_list,
        solver=solver,
        temperature_k=TEMP_K,
        initial_densities=initial_densities,
        max_iter=max_iter,
        background=True,
        fit_energy_scale=True,
        t0_init_us=0.5,
        l_scale_init=1.005,
        energy_scale_flight_path_m=FLIGHT_PATH_M,
        resolution=res,
        dead_pixels=dead_pixels,
        fit_energy_range=(ENERGY_MIN, ENERGY_MAX),
    )
    if solver == "kl":
        kwargs["c"] = c

    t0 = time.time()
    r = nereids.spatial_map_typed(**kwargs)
    wall = time.time() - t0
    # Capture all 6 per-isotope density maps so any per-iso regression
    # surfaces in the diff (not just the summed or primary isotope).
    density_hex: list[list[str]] = []
    for m in r.density_maps:
        density_hex.append(_array_hex(np.asarray(m)))
    conv = np.asarray(r.converged_map)
    return {
        "name": label,
        "wall_s": wall,
        "n_pixels": int(conv.size),
        "converged_count": int(conv.sum()),
        "density_per_iso_hex": density_hex,
        "converged_hex": _array_hex(conv.astype(np.float64)),
    }


def run_b3() -> dict:
    """B.3: spatial LM+per-iso+TZERO on 4x4 VENUS Hf crop."""
    return _run_periso_tzero(
        solver="lm",
        crop_size=4,
        label="B3_spatial_LM_periso_background_tzero_4x4",
        max_iter=200,
    )


def run_kl_periso_tzero() -> dict:
    """KL+per-iso+TZERO on 4x4 VENUS Hf crop — added in this PR to
    verify the solver-agnostic nature of the
    `EnergyScaleTransmissionModel` plan cache. Not in #459's
    benchmark set but enabled by
    `test_spatial_map_typed_allows_counts_kl_with_energy_scale`.
    """
    return _run_periso_tzero(
        solver="kl",
        crop_size=4,
        label="KL_spatial_periso_background_tzero_4x4",
        max_iter=200,
    )


def dump_baseline(out: Path, include_b1: bool) -> None:
    data = {
        "a1": run_a1(),
        "b2": run_b2(),
        "b3": run_b3(),
        "kl_periso_tzero": run_kl_periso_tzero(),
    }
    if include_b1:
        data["b1"] = run_b1()
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote baseline: {out}")
    for k, v in data.items():
        print(f"  {k}: wall={v.get('wall_s', 0):.2f}s")


def _diff(label: str, base: dict, cur: dict) -> list[str]:
    diffs: list[str] = []
    for key in base:
        if key == "wall_s":
            continue
        bv = base[key]
        cv = cur.get(key)
        if isinstance(bv, list):
            if not isinstance(cv, list):
                # A missing / non-list current value cannot be
                # zip-compared against the saved list — treat it as
                # an explicit mismatch so verify reports a clean
                # diff rather than raising `TypeError: NoneType ...`.
                diffs.append(
                    f"{label}.{key}: base is list of length {len(bv)} "
                    f"but cur is {type(cv).__name__} ({cv!r})"
                )
                continue
            if len(bv) != len(cv):
                diffs.append(f"{label}.{key}: length {len(bv)} vs {len(cv)}")
                continue
            if bv and isinstance(bv[0], list):
                # Nested list-of-lists (e.g. density_per_iso_hex): diff each
                # sub-list recursively so per-isotope mismatches surface
                # with their isotope + pixel indices.
                for i, (sub_b, sub_c) in enumerate(zip(bv, cv)):
                    diffs.extend(_diff(f"{label}.{key}[{i}]", {"v": sub_b}, {"v": sub_c}))
                continue
            mismatches = [i for i, (b, c) in enumerate(zip(bv, cv)) if b != c]
            if mismatches:
                diffs.append(
                    f"{label}.{key}: {len(mismatches)} element(s) differ "
                    f"(first at index {mismatches[0]}: base={bv[mismatches[0]]}, cur={cv[mismatches[0]]})"
                )
        else:
            if bv != cv:
                diffs.append(f"{label}.{key}: base={bv} cur={cv}")
    return diffs


def verify_baseline(path: Path, include_b1: bool) -> int:
    base = json.loads(path.read_text())
    # Every case captured in the baseline must be re-run on verify.
    # Otherwise a baseline saved with --include-b1 could silently print
    # "BASELINE OK" when verified without --include-b1, skipping the B.1
    # comparison entirely (Codex finding).  `--include-b1` on the CLI
    # stays as an explicit opt-in for the dump phase; on verify we
    # honour whatever the JSON actually holds.
    effective_include_b1 = include_b1 or ("b1" in base)
    cur = {
        "a1": run_a1(),
        "b2": run_b2(),
        "b3": run_b3(),
        "kl_periso_tzero": run_kl_periso_tzero(),
    }
    if effective_include_b1:
        cur["b1"] = run_b1()
    diffs: list[str] = []
    missing: list[str] = []
    for k in base:
        if k not in cur:
            missing.append(k)
            continue
        diffs.extend(_diff(k, base[k], cur[k]))
    if missing:
        print("BASELINE MISMATCH: saved cases missing from current run:")
        for k in missing:
            print(f"  {k}")
        return 1
    if diffs:
        print("BASELINE MISMATCH:")
        for d in diffs:
            print("  " + d)
        return 1
    print("BASELINE OK — bit-exact on all captured scalars + arrays.")
    for k, v in cur.items():
        print(f"  {k}: wall={v.get('wall_s', 0):.2f}s")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="Dump baseline to JSON path")
    mode.add_argument("--verify", type=Path, help="Verify against saved JSON")
    ap.add_argument(
        "--include-b1",
        action="store_true",
        help="Also run B.1 spatial LM+TZERO 2x2 (slow: ~60-120s).",
    )
    args = ap.parse_args()
    if args.out:
        dump_baseline(args.out, args.include_b1)
    else:
        sys.exit(verify_baseline(args.verify, args.include_b1))


if __name__ == "__main__":
    main()
