#!/usr/bin/env python3
"""Rebuild the SAMMY references of the Ta-181 forward-model check 1.

Free-gas transmission with the RPI prompt term over the bins of the 4.280412 eV
resonance. Run ``generate_phase1.py -h`` for the commands.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import shutil
import subprocess
from pathlib import Path

from generate import (
    ALPHA_PER_US,
    BURST_NS,
    CHANNEL_NS,
    EDGE0_CENTI_US,
    ENDF_FILE,
    FREE_GAS,
    HERE,
    RADIUS_FM,
    RESONANCES,
    SAMMY_NEUTRON_AMU,
    THICKNESS_PER_B,
    TRANSMISSION_COLUMN,
    WIDTH_CENTI_US,
    lpt_groups,
    read_awr,
    tof_to_energy,
)

# SAMMY.LPT names the build; the commit is that of the RNSD/SAMMY source it was built from.
SAMMY_BUILD = "8.2.b4"
SAMMY_COMMIT = "c974c02c"
# MD5 of the executable built from that commit; another executable is refused.
SAMMY_MD5 = "091b34fa0229a5793918956006e23f68"

# Scored bins: centre energy from 0.5 eV to the midpoint of the 194.8900 and
# 200.1447 eV resonances, the last below 200 eV.
E_LOW_EV = 0.5
E_LAST_EV = 197.51735
# Upper edge of the 4.280412 eV window: the midpoint to the 10.34682 eV resonance.
E_P1A_HIGH_EV = 7.313616

NXTRA = 29
NXTRA_CHECK = 99
P1A_K = ("299", "299.5", "300", "300.5", "301", "1099", "1099.5", "1100", "1100.5", "1101")
# Temperatures whose NXTRA change bounds the error.
P1A_CHECKED_K = ("299", "300", "301", "1099", "1100", "1101")

# GENERATE PLOT FILE writes SAMMY.LST, whose values carry more digits than SAMMY.LPT.
COMMANDS = ("INPUT IS ENDF/B FILE", "TWENTY", "DO NOT SOLVE BAYES EQUATIONS",
            "PRINT THEORETICAL VALUES", "GENERATE PLOT FILE AUTOMATICALLY")


def bin_energies(high_ev: float) -> list[str]:
    """Return the centre energies of the bins in [E_LOW_EV, high_ev] eV, in time order."""
    energies, j = [], 0
    while True:
        lo = (EDGE0_CENTI_US + WIDTH_CENTI_US * j) / 100
        hi = (EDGE0_CENTI_US + WIDTH_CENTI_US * (j + 1)) / 100
        energy = tof_to_energy((lo + hi) / 2)
        if energy < E_LOW_EV:
            return energies
        # A centre on a window edge belongs to the lower-energy window.
        if energy <= high_ev:
            energies.append(f"{energy:.11e}")
        j += 1


def sammy_input(awr: float, kelvin: str, nxtra: int, model: str) -> str:
    """Return the SAMMY INPut file of one case (SAMMY manual, Table VI A.1).

    ``model`` is ``rpi`` (free gas with the RPI prompt term) or ``free`` (free gas without
    instrument).
    """
    title = {"rpi": "free gas RPI", "free": "free gas"}[model]
    title = f"Ta-181 {title} {kelvin} K NXTRA {nxtra}"
    if model == "free":
        commands = (*COMMANDS, FREE_GAS)
        emin, emax = 0.3, 1000.0
    else:
        commands = (*COMMANDS, FREE_GAS, "DO NOT SHIFT RPI RESOLUTION FUNCTION TO CENTER")
        emin, emax = 0.3, 10.0
    card2 = (f"{'Ta181':<10}{awr * SAMMY_NEUTRON_AMU:10.6f}{emin:10.4f}{emax:10.4f}"
             f"{'':12}{nxtra:3d}")
    lines = [title, card2, *commands, "",
             # Card 5: TEMP (K), DIST (m), DELTAL, DELTAE, DELTAG.
             f"{float(kelvin):10.3f}{25.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}",
             f"{0.0:10.1f}{THICKNESS_PER_B:10.6f}",
             "TRANSMISSION", ""]
    if model == "rpi":
        lines += ["RPI Resolution function",
                  f"BURST00000{BURST_NS:10.1f}{0.0:10.1f}",
                  f"LAMBD00000{1000.0 / ALPHA_PER_US:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}"
                  f"{0.0:10.1f}",
                  f"{'':10}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}",
                  f"CHANN00000{1000.0:10.1f}{CHANNEL_NS:10.1f}{0.0:10.1f}", ""]
    return "\n".join(lines + [""])


def cases(awr: float) -> dict[str, dict]:
    """Return every SAMMY case of check 1 by name."""
    specs = {}
    for k in P1A_K:
        for n in ((NXTRA, NXTRA_CHECK) if k in P1A_CHECKED_K else (NXTRA,)):
            specs[f"p1a_T{k}_n{n}"] = {"kelvin": k, "high": E_P1A_HIGH_EV, "model": "rpi",
                                       "inp": sammy_input(awr, k, n, "rpi")}
    return specs


def check_lpt(case: Path, spec: dict, nbins: int) -> str:
    """Stop unless the SAMMY.LPT echo of the case is the reference input; return the build."""
    lpt = (case / "SAMMY.LPT").read_text()
    log = (case / "sammy.out").read_text()
    version = lpt_groups(lpt, r"SAMMY Version (\S+)", case)[0]
    if version != SAMMY_BUILD:
        raise SystemExit(f"{case}: SAMMY {version}, not {SAMMY_BUILD}")
    for text in ("REICH MOORE FORMALISM IS WANTED",
                 f"Number of experimental data points ={nbins:8d}"):
        if text not in lpt:
            raise SystemExit(f"{case}: SAMMY.LPT lacks {text!r}")
    resonances = lpt_groups(lpt, r"Total  number of resonances is\s+(\d+)", case)[0]
    radius = lpt_groups(lpt, r"SPIN GROUP NUMBERS FOR THESE RADII\s+(\S+)", case)[0]
    if int(resonances) != RESONANCES or float(radius) != RADIUS_FM:
        raise SystemExit(f"{case}: {resonances} resonances, radius {radius}")
    kelvin, thick = lpt_groups(lpt, r"TEMPERATURE\s+THICKNESS\s+(\S+)\s+(\S+)", case)
    if abs(float(kelvin) - float(spec["kelvin"])) > 1e-6 or float(thick) != THICKNESS_PER_B:
        raise SystemExit(f"{case}: SAMMY.LPT gives {kelvin} K and {thick} atoms/b")
    if spec["model"] == "free":
        if "SAMMY-RPI" in log or "*** SAMMY-FGM" not in log:
            raise SystemExit(f"{case}: the free gas did not run alone")
    else:
        if "SAMMY-RPI" not in log or "No resolution broadening occurred" in lpt:
            raise SystemExit(f"{case}: RPI broadening did not run on every point")
        lam = [float(lpt_groups(lpt, rf"Lambda{i} \([^)]*\) =\s*(\S+)", case)[0])
               for i in range(5)]
        chann = float(lpt_groups(lpt, r"0\.0\s+infinity\s+(\S+)", case)[0])
        burst = float(lpt_groups(
            lpt, r"Burst full-width-at-half-max \(nanosec\) =\s*(\S+)", case)[0])
        if lam != [1000.0 / ALPHA_PER_US, 0.0, 0.0, 0.0, 0.0] or chann != CHANNEL_NS \
                or burst != BURST_NS:
            raise SystemExit(f"{case}: Lambda {lam}, channel {chann}, burst {burst}")
        if "*** SAMMY-FGM" not in log:
            raise SystemExit(f"{case}: the free-gas model did not run")
    return version


def run_case(exe: Path, work: Path, name: str, spec: dict,
             keep: bool) -> tuple[list[str], str]:
    """Run SAMMY in work/name; return the SAMMY.LST transmission in time order and the build.

    The energies are ``spec["energies"]`` in time order, or the bin centres up to
    ``spec["high"]``. With ``keep``, a finished run whose case directory holds the same
    input bytes is read again.
    """
    energies = spec["energies"] if "energies" in spec else bin_energies(spec["high"])
    case = work / name
    ascending = list(reversed(energies))
    # The executable's MD5 is an input, so a kept run is one the same build made.
    inputs = {"case.inp": spec["inp"],
              "data.dat": "".join(f"{e.upper():>20}{1.0:20.6f}{0.1:20.6f}\n" for e in ascending),
              "sammy.md5": hashlib.md5(exe.read_bytes()).hexdigest() + "\n"}
    files = f"case.inp\n{ENDF_FILE}\ndata.dat\n"
    log = case / "sammy.out"
    kept = (keep and log.is_file() and "Normal finish to SAMMY" in log.read_text()
            and (case / ENDF_FILE).read_bytes() == (HERE / ENDF_FILE).read_bytes()
            and all((case / f).is_file() and (case / f).read_text() == text
                    for f, text in inputs.items()))
    if not kept:
        if case.exists():
            shutil.rmtree(case)
        case.mkdir(parents=True)
        shutil.copyfile(HERE / ENDF_FILE, case / ENDF_FILE)
        for f, text in inputs.items():
            (case / f).write_text(text)
        result = subprocess.run([str(exe)], input=files + "\n", text=True, cwd=case,
                                capture_output=True, check=False)
        log.write_text(result.stdout + result.stderr)
    if "Normal finish to SAMMY" not in (case / "sammy.out").read_text():
        raise SystemExit(f"SAMMY did not finish in {case}; see sammy.out")
    rows = [line.split() for line in (case / "SAMMY.LST").read_text().splitlines()
            if line.strip()]
    if len(rows) != len(energies):
        raise SystemExit(f"{case}: SAMMY.LST has {len(rows)} rows for {len(energies)} bins")
    for e, row in zip(ascending, rows):
        if abs(float(row[0]) - float(e)) > 1e-9:
            raise SystemExit(f"{case}: SAMMY.LST energy {row[0]} does not match {e}")
    build = check_lpt(case, spec, len(energies))
    print(f"{name}: ok{' (kept)' if kept else ''}", flush=True)
    return [row[TRANSMISSION_COLUMN] for row in reversed(rows)], build


def largest(a: list[str], b: list[str]) -> float:
    """Return max|a − b| over two runs on the same bins."""
    if len(a) != len(b):
        raise SystemExit("compared runs hold different bins")
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def write_set(prefix: str, temperatures: tuple[str, ...], run: dict, energies: list[str],
              header: list[str], eps_lines: list[str]) -> None:
    """Write ``prefix``.txt with one column per temperature and ``prefix``_eps.txt."""
    rows = [" ".join([e, *values]) for e, values in
            zip(energies, zip(*(run[k] for k in temperatures)))]
    head = header + ["# Columns: e_ev " + " ".join(f"T{k}" for k in temperatures)]
    (HERE / f"{prefix}.txt").write_text("\n".join(head + rows) + "\n")
    (HERE / f"{prefix}_eps.txt").write_text("\n".join(eps_lines) + "\n")


def run_all(exe: Path, work: Path, jobs: int, keep: bool) -> None:
    """Run every case of check 1 and write its reference files."""
    md5 = hashlib.md5(exe.read_bytes()).hexdigest()
    if md5 != SAMMY_MD5:
        raise SystemExit(f"{exe} has MD5 {md5}, not that of SAMMY {SAMMY_BUILD} {SAMMY_COMMIT}")
    specs = cases(read_awr())
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {name: pool.submit(run_case, exe, work, name, spec, keep)
                   for name, spec in specs.items()}
        results = {name: future.result() for name, future in futures.items()}
    builds = {build for _, build in results.values()}
    if builds != {SAMMY_BUILD}:
        raise SystemExit(f"SAMMY builds: {builds}")
    stamp = (f"# SAMMY {SAMMY_BUILD} (commit {SAMMY_COMMIT}), executable MD5 {md5};"
             " Ta-181-mf2.endf (ENDF/B-VIII.1")
    energies = bin_energies(E_P1A_HIGH_EV)
    run = {k: results[f"p1a_T{k}_n{NXTRA}"][0] for k in P1A_K}
    eps = {k: largest(run[k], results[f"p1a_T{k}_n{NXTRA_CHECK}"][0]) for k in P1A_CHECKED_K}
    write_set("p1a_rpi_alpha2", P1A_K, run, energies, [
        stamp, "# MAT 7328, Reich-Moore) at the VENUS bin-centre energies of the 4.280412 eV",
        "# window, in time order. Written by generate_phase1.py.",
        "# Transmission, n = 7.04e-4 atoms/b, free-gas Doppler and the RPI resolution",
        f"# function: chi-squared term only, Lambda0 = {1000.0 / ALPHA_PER_US:.1f} ns, a"
        f" {BURST_NS:.0f} ns Gaussian burst, square channel {CHANNEL_NS:.0f} ns, not shifted",
        f"# to centre, L = 25 m; NXTRA = {NXTRA}; SAMMY.LST column 8."],
        [stamp + " MAT 7328).",
         f"# eps = max|T(NXTRA {NXTRA}) - T(NXTRA {NXTRA_CHECK})| over the bins.",
         "# Columns: temperature (K), eps.",
         *(f"{k} {eps[k]:.6e}" for k in P1A_CHECKED_K)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exe", type=Path, help="SAMMY executable")
    parser.add_argument("work", type=Path, help="directory for the SAMMY runs")
    parser.add_argument("--jobs", type=int, default=1, help="SAMMY runs at the same time")
    parser.add_argument("--keep", action="store_true",
                        help="read again a finished run whose case directory holds the same"
                             " inputs")
    args = parser.parse_args()
    run_all(args.exe.resolve(), args.work.resolve(), args.jobs, args.keep)


if __name__ == "__main__":
    main()
