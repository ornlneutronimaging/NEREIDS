#!/usr/bin/env python3
"""Rebuild the SAMMY reference data of the Ta-181 VENUS thermometry oracle.

The files go next to this script. The SAMMY inputs come only from
``Ta-181-mf2.endf`` and the VENUS bins. Run ``generate.py -h`` for the commands.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENDF_FILE = "Ta-181-mf2.endf"
MAT = "7328"

# CODATA 2018 neutron mass (kg) and electron volt (J).
NEUTRON_MASS_KG = 1.674_927_498_04e-27
EV_TO_JOULES = 1.602_176_634e-19
# SAMMY neutron mass (amu); SAMMY gives the target mass as AWR times this value.
SAMMY_NEUTRON_AMU = 1.00866491578

FLIGHT_PATH_M = 25.0
# Bin edge j is (112 + 16 j) / 100 us.
EDGE0_CENTI_US = 112
WIDTH_CENTI_US = 16
E_LOW_EV = 7.8
E_HIGH_EV = 200.0

THICKNESS_PER_B = 7.04e-4
TEMPERATURES_K = (299, 300, 301, 1099, 1100, 1101)
CHECKED_K = (300, 1100)
ALPHA_PER_US = 2
CHANNEL_NS = 160.0
# A burst selects SAMMY's general RPI routine. Without one, SAMMY's chi-squared
# and channel routine loses kernel weight at the shortest delays.
BURST_NS = 1.0
NXTRA = 29
# eps_S is the largest change of a transmission from NXTRA to NXTRA_CHECK.
NXTRA_CHECK = 99
SAMMY_BUILD = "8.1.pre-b1 - 0aa6ee20"
SAMMY_MD5 = "ef403c81ef531ae3511eb007f6c0080e"
# The cached ENDF/B-VIII.1 Ta-181 tape.
CACHED_MD5 = "e9cae860785d7ea503430300e4428663"
RESONANCES = 565
RADIUS_FM = 7.86
# SAMMY moves EMIN and EMAX to the data ends and extends its own grid for the
# Doppler and RPI widths (SAMMY manual, Section III.A.2).
EMIN_EV = 7.0
EMAX_EV = 250.0

COMMANDS = (
    "INPUT IS ENDF/B FILE",
    "TWENTY",
    "USE FREE GAS MODEL OF DOPPLER BROADENING",
    "DO NOT SOLVE BAYES EQUATIONS",
    "PRINT THEORETICAL VALUES",
    "GENERATE PLOT FILE AUTOMATICALLY",
    "DO NOT SHIFT RPI RESOLUTION FUNCTION TO CENTER",
)
# SAMMY.LST column 8 (SAMMY manual, Table VII C.1) is the theoretical transmission.
TRANSMISSION_COLUMN = 7


def endf_number(field: str) -> float:
    """Return the value of an ENDF real field, with or without the E."""
    text = field.strip()
    for i in range(1, len(text)):
        if text[i] in "+-" and text[i - 1] not in "eE":
            return float(text[:i] + "e" + text[i:])
    return float(text)


def tag(record: bytes) -> tuple[str, str, str]:
    """Return the MAT, MF and MT columns of an ENDF record."""
    text = record.decode("ascii")
    return text[66:70], text[70:72], text[72:75]


def write_endf(cached: Path) -> None:
    """Copy MF1 MT451 and MF2 MT151 of MAT 7328, with their end records, unchanged."""
    data = cached.read_bytes()
    if hashlib.md5(data).hexdigest() != CACHED_MD5:
        raise SystemExit(f"{cached}: MD5 is not {CACHED_MD5}")
    records = data.splitlines(keepends=True)
    tags = [tag(r) for r in records]
    first = tags.index((MAT, " 1", "451"))
    last = tags.index((MAT, " 2", "  0")) + 1
    allowed = {(MAT, " 1", "451"), (MAT, " 1", "  0"), (MAT, " 0", "  0"),
               (MAT, " 2", "151"), (MAT, " 2", "  0")}
    if tags[last] != (MAT, " 0", "  0") or not set(tags[first:last + 1]) <= allowed:
        raise SystemExit(f"{cached}: unexpected records between MF1 MT451 and the MF2 FEND")
    if tags[-2:] != [("   0", " 0", "  0"), ("  -1", " 0", "  0")]:
        raise SystemExit(f"{cached}: the file does not end with MEND and TEND records")
    (HERE / ENDF_FILE).write_bytes(b"".join(records[first:last + 1] + records[-2:]))


def tof_to_energy(tof_us: float) -> float:
    """Return the neutron energy (eV) for a flight time (us) over the flight path."""
    # E = m_n v^2 / 2 with v = L / t, in the operation order of the test.
    t_s = tof_us * 1.0e-6
    v = FLIGHT_PATH_M / t_s
    return 0.5 * NEUTRON_MASS_KG * v * v / EV_TO_JOULES


def bin_energies() -> list[str]:
    """Return the centre energy of each bin in [7.8, 200] eV, in time order."""
    energies = []
    j = 0
    while True:
        lo = (EDGE0_CENTI_US + WIDTH_CENTI_US * j) / 100
        hi = (EDGE0_CENTI_US + WIDTH_CENTI_US * (j + 1)) / 100
        energy = tof_to_energy((lo + hi) / 2)
        if energy < E_LOW_EV:
            return energies
        if energy <= E_HIGH_EV:
            energies.append(f"{energy:.11e}")
        j += 1


def read_awr() -> float:
    """Return AWR from the MF2 MT151 HEAD record."""
    for record in (HERE / ENDF_FILE).read_bytes().splitlines():
        if tag(record) == (MAT, " 2", "151"):
            return endf_number(record.decode("ascii")[11:22])
    raise SystemExit(f"{ENDF_FILE}: no MF2 MT151 record")


def sammy_input(awr: float, kelvin: int, nxtra: int) -> str:
    """Return the SAMMY INPut file of one case (SAMMY manual, Table VI A.1)."""
    title = f"Ta-181 S2 Doppler and RPI alpha {ALPHA_PER_US}/us {kelvin} K NXTRA {nxtra}"
    # Card 2: ELMNT, AW, EMIN, EMAX in 10-column fields; NXTRA in columns 53-55.
    card2 = (f"{'Ta181':<10}{awr * SAMMY_NEUTRON_AMU:10.6f}{EMIN_EV:10.4f}{EMAX_EV:10.4f}"
             f"{'':12}{nxtra:3d}")
    return "\n".join([
        title, card2, *COMMANDS, "",
        # Card 5: TEMP (K), DIST (m), DELTAL, DELTAE, DELTAG.
        f"{kelvin:10.1f}{FLIGHT_PATH_M:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}",
        # Card 7: CRFN (0 keeps the ENDF radius), THICK (atoms/b).
        f"{0.0:10.1f}{THICKNESS_PER_B:10.6f}",
        "TRANSMISSION",
        # Card set 14 (SAMMY manual, Table VI B.2): burst FWHM (ns), chi-squared width
        # Lambda0 (ns) with its uncertainty line, one channel width (ns) up to ECRNCH (eV).
        "",
        "RPI Resolution function",
        f"BURST00000{BURST_NS:10.1f}{0.0:10.1f}",
        f"LAMBD00000{1000.0 / ALPHA_PER_US:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}",
        f"{'':10}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}",
        f"CHANN00000{1000.0:10.1f}{CHANNEL_NS:10.1f}{0.0:10.1f}",
        "", "",
    ])


def cases(awr: float) -> dict[str, dict]:
    """Return every SAMMY case by name, with its temperature, NXTRA and input."""
    runs = [(k, NXTRA) for k in TEMPERATURES_K] + [(k, NXTRA_CHECK) for k in CHECKED_K]
    return {f"s2_a{ALPHA_PER_US}_T{k}_n{n}": {"kelvin": k, "inp": sammy_input(awr, k, n)}
            for k, n in runs}


def lpt_groups(lpt: str, pattern: str, case: Path) -> tuple[str, ...]:
    """Return the groups of the first match of a pattern in SAMMY.LPT, or stop."""
    found = re.search(pattern, lpt)
    if not found:
        raise SystemExit(f"{case}: SAMMY.LPT has no match for {pattern!r}")
    return found.groups()


def check_lpt(case: Path, spec: dict, nbins: int) -> str:
    """Stop unless the SAMMY.LPT echo of the case is the reference input; return the build."""
    lpt = (case / "SAMMY.LPT").read_text()
    version = lpt_groups(lpt, r"SAMMY Version (\S+ - \w+)", case)[0]
    for text in ("ENDF file has Lrf=  3", "REICH MOORE FORMALISM IS WANTED",
                 f"Number of experimental data points ={nbins:8d}"):
        if text not in lpt:
            raise SystemExit(f"{case}: SAMMY.LPT lacks {text!r}")
    if version != SAMMY_BUILD:
        raise SystemExit(f"{case}: SAMMY {version}, not {SAMMY_BUILD}")
    resonances = lpt_groups(lpt, r"Total  number of resonances is\s+(\d+)", case)[0]
    radius = lpt_groups(lpt, r"SPIN GROUP NUMBERS FOR THESE RADII\s+(\S+)", case)[0]
    if int(resonances) != RESONANCES or float(radius) != RADIUS_FM:
        raise SystemExit(f"{case}: {resonances} resonances, radius {radius}")
    kelvin, thick = lpt_groups(lpt, r"TEMPERATURE\s+THICKNESS\s+(\S+)\s+(\S+)", case)
    if float(kelvin) != spec["kelvin"] or float(thick) != THICKNESS_PER_B:
        raise SystemExit(f"{case}: SAMMY.LPT gives {kelvin} K and {thick} atoms/b")
    if "SAMMY-RPI" not in (case / "sammy.out").read_text():
        raise SystemExit(f"{case}: RPI broadening did not run")
    if "No resolution broadening occurred" in lpt:
        raise SystemExit(f"{case}: SAMMY.LPT reports points without resolution broadening")
    lam = [float(lpt_groups(lpt, rf"Lambda{i} \([^)]*\) =\s*(\S+)", case)[0]) for i in range(5)]
    chann = float(lpt_groups(lpt, r"0\.0\s+infinity\s+(\S+)", case)[0])
    if lam != [1000.0 / ALPHA_PER_US, 0.0, 0.0, 0.0, 0.0] or chann != CHANNEL_NS:
        raise SystemExit(f"{case}: SAMMY.LPT gives Lambda {lam} and channel {chann}")
    burst = float(lpt_groups(lpt, r"Burst full-width-at-half-max \(nanosec\) =\s*(\S+)", case)[0])
    if burst != BURST_NS:
        raise SystemExit(f"{case}: SAMMY.LPT gives a burst of {burst} ns")
    # SAMMY ends the chi-squared term at 16 Lambda.
    chi_us = float(lpt_groups(
        lpt, r"Time width associated with chi-squared fnct is\s*(\S+)", case)[0])
    if abs(chi_us - 16.0 / ALPHA_PER_US) > 1e-3:
        raise SystemExit(f"{case}: chi-squared time width {chi_us} us")
    return version


def run_case(exe: Path, work: Path, name: str, spec: dict,
             energies: list[str]) -> tuple[list[str], str]:
    """Run SAMMY in work/name; return the SAMMY.LST transmission in bin order and the build."""
    case = work / name
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    shutil.copyfile(HERE / ENDF_FILE, case / ENDF_FILE)
    (case / "case.inp").write_text(spec["inp"])
    # TWENTY data format: energy, value and uncertainty in 20-column fields.
    ascending = list(reversed(energies))
    (case / "data.dat").write_text(
        "".join(f"{e.upper():>20}{1.0:20.6f}{0.1:20.6f}\n" for e in ascending))
    result = subprocess.run([str(exe)], input=f"case.inp\n{ENDF_FILE}\ndata.dat\n\n",
                            text=True, cwd=case, capture_output=True, check=False)
    (case / "sammy.out").write_text(result.stdout + result.stderr)
    if "Normal finish to SAMMY" not in result.stdout:
        raise SystemExit(f"SAMMY did not finish in {case}; see sammy.out")
    rows = [line.split() for line in (case / "SAMMY.LST").read_text().splitlines() if line.strip()]
    if len(rows) != len(energies):
        raise SystemExit(f"{case}: SAMMY.LST has {len(rows)} rows for {len(energies)} bins")
    for e, row in zip(ascending, rows):
        if abs(float(row[0]) - float(e)) > 1e-9:
            raise SystemExit(f"{case}: SAMMY.LST energy {row[0]} does not match {e}")
    build = check_lpt(case, spec, len(energies))
    print(f"{name}: ok", flush=True)
    return [row[TRANSMISSION_COLUMN] for row in reversed(rows)], build


def run_all(exe: Path, work: Path, jobs: int) -> None:
    """Run every case and write s2_rpi_alpha2.txt and eps_s.txt."""
    if hashlib.md5(exe.read_bytes()).hexdigest() != SAMMY_MD5:
        raise SystemExit(f"{exe}: MD5 is not {SAMMY_MD5}")
    energies = bin_energies()
    specs = cases(read_awr())
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {name: pool.submit(run_case, exe, work, name, spec, energies)
                   for name, spec in specs.items()}
        results = {name: future.result() for name, future in futures.items()}
    builds = {build for _, build in results.values()}
    if len(builds) != 1:
        raise SystemExit(f"SAMMY versions differ between runs: {builds}")
    build = builds.pop()
    run = {key: results[f"s2_a{ALPHA_PER_US}_T{key[0]}_n{key[1]}"][0]
           for key in [(k, NXTRA) for k in TEMPERATURES_K] + [(k, NXTRA_CHECK) for k in CHECKED_K]}
    header = [
        f"# SAMMY {build}; Ta-181-mf2.endf (ENDF/B-VIII.1 MAT 7328, Reich-Moore) at the",
        "# VENUS bin-centre energies, in time order. Written by generate.py sammy.",
        "# Transmission, n = 7.04e-4 atoms/b, free-gas Doppler and the RPI resolution",
        f"# function: chi-squared term only, Lambda0 = {1000.0 / ALPHA_PER_US:.1f} ns (alpha ="
        f" {ALPHA_PER_US} per us), a {BURST_NS:.0f} ns",
        f"# Gaussian burst, square channel {CHANNEL_NS:.0f} ns, not shifted to centre,"
        f" L = 25 m; NXTRA = {NXTRA}; SAMMY.LST column 8.",
        "# Columns: e_ev " + " ".join(f"T{k}" for k in TEMPERATURES_K),
    ]
    columns = [run[(k, NXTRA)] for k in TEMPERATURES_K]
    rows = [" ".join([e, *values]) for e, values in zip(energies, zip(*columns))]
    (HERE / f"s2_rpi_alpha{ALPHA_PER_US}.txt").write_text("\n".join(header + rows) + "\n")
    eps = [max(abs(float(a) - float(b)) for a, b in zip(run[(k, NXTRA)], run[(k, NXTRA_CHECK)]))
           for k in CHECKED_K]
    (HERE / "eps_s.txt").write_text("\n".join([
        f"# SAMMY {build}: eps_S = max|T(NXTRA = {NXTRA}) - T(NXTRA = {NXTRA_CHECK})| over the bins.",
        "# Columns: temperature (K), eps_S.",
        *(f"{k} {e:.6e}" for k, e in zip(CHECKED_K, eps)),
    ]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    endf = sub.add_parser("endf", help=f"write {ENDF_FILE} from the cached ENDF/B-VIII.1 file")
    endf.add_argument("cached", type=Path)
    sammy = sub.add_parser("sammy", help="run SAMMY and write s2_rpi_alpha2.txt and eps_s.txt")
    sammy.add_argument("exe", type=Path, help="SAMMY executable")
    sammy.add_argument("work", type=Path, help="directory for the SAMMY runs")
    sammy.add_argument("--jobs", type=int, default=1, help="SAMMY runs at the same time")
    args = parser.parse_args()
    if args.command == "endf":
        write_endf(args.cached)
    else:
        run_all(args.exe.resolve(), args.work.resolve(), args.jobs)


if __name__ == "__main__":
    main()
