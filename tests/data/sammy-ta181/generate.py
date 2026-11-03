#!/usr/bin/env python3
"""Rebuild the SAMMY reference data of the Ta-181 VENUS thermometry oracle.

The files go next to this script. The SAMMY inputs come only from
``Ta-181-mf2.endf`` and ``bins.txt``. Run ``generate.py -h`` for the commands.
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
BINS_FILE = "bins.txt"
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
ALPHAS_PER_US = (2, 8)
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

COMMON_COMMANDS = (
    "INPUT IS ENDF/B FILE",
    "TWENTY",
    "DO NOT SOLVE BAYES EQUATIONS",
    "PRINT THEORETICAL VALUES",
    "GENERATE PLOT FILE AUTOMATICALLY",
)
# SAMMY.LST columns (SAMMY manual, Table VII C.1): 4 is the theoretical cross
# section, 8 the theoretical transmission.
SIGMA_COLUMN = 3
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
    # E = m_n v^2 / 2 with v = L / t; this operation order gives the doubles of bins.txt.
    t_s = tof_us * 1.0e-6
    v = FLIGHT_PATH_M / t_s
    return 0.5 * NEUTRON_MASS_KG * v * v / EV_TO_JOULES


def edge_text(j: int) -> str:
    """Return bin edge j in us as an exact two-decimal string."""
    centi = EDGE0_CENTI_US + WIDTH_CENTI_US * j
    return f"{centi // 100}.{centi % 100:02d}"


def write_bins() -> None:
    """Write every bin whose centre energy is in [7.8, 200] eV, in time order."""
    rows = []
    j = 0
    while True:
        lo, hi = edge_text(j), edge_text(j + 1)
        energy = tof_to_energy((float(lo) + float(hi)) / 2)
        if energy < E_LOW_EV:
            break
        if energy <= E_HIGH_EV:
            rows.append(f"{lo} {hi} {energy:.11e}")
        j += 1
    header = [
        "# VENUS time bins of the Ta-181 SAMMY oracle; written by generate.py bins.",
        "# Edges t_j = 1.12 + 0.16 j us; flight path 25 m; t0 = 0.",
        "# e_center_ev = (m_n/2) (L/t_c)^2 at t_c = (t_lo + t_hi)/2, with the CODATA 2018",
        "# neutron mass and electron volt. Kept: 7.8 eV <= e_center_ev <= 200 eV.",
        "# Columns: t_lo_us t_hi_us e_center_ev",
    ]
    (HERE / BINS_FILE).write_text("\n".join(header + rows) + "\n")


def read_bins() -> list[str]:
    """Return the e_center_ev strings of bins.txt, in file order."""
    rows = [line.split() for line in (HERE / BINS_FILE).read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    return [row[2] for row in rows]


def read_awr() -> float:
    """Return AWR from the MF2 MT151 HEAD record."""
    for record in (HERE / ENDF_FILE).read_bytes().splitlines():
        if tag(record) == (MAT, " 2", "151"):
            return endf_number(record.decode("ascii")[11:22])
    raise SystemExit(f"{ENDF_FILE}: no MF2 MT151 record")


def sammy_input(title: str, awr: float, stage: str, kelvin: int = 0,
                alpha: int = 0, nxtra: int = 0, burst_ns: float | None = BURST_NS) -> str:
    """Return the SAMMY INPut file for one stage (SAMMY manual, Table VI A.1)."""
    # Card 2: ELMNT, AW, EMIN, EMAX in 10-column fields; NXTRA in columns 53-55.
    card2 = f"{'Ta181':<10}{awr * SAMMY_NEUTRON_AMU:10.6f}{EMIN_EV:10.4f}{EMAX_EV:10.4f}"
    if stage != "s0":
        card2 += f"{'':12}{nxtra:3d}"
    commands = list(COMMON_COMMANDS)
    if stage == "s0":
        commands.insert(2, "BROADENING IS NOT WANTED")
    else:
        commands.insert(2, "USE FREE GAS MODEL OF DOPPLER BROADENING")
    if stage == "s2":
        commands.append("DO NOT SHIFT RPI RESOLUTION FUNCTION TO CENTER")
    lines = [title, card2, *commands, ""]
    if stage != "s0":
        # Card 5: TEMP (K), DIST (m), DELTAL, DELTAE, DELTAG.
        lines.append(f"{kelvin:10.1f}{FLIGHT_PATH_M:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}")
    # Card 7: CRFN (0 keeps the ENDF radius), THICK (atoms/b).
    lines.append(f"{0.0:10.1f}{THICKNESS_PER_B:10.6f}")
    lines.append("TOTAL CROSS SECTION" if stage == "s0" else "TRANSMISSION")
    if stage == "s2":
        # Card set 14 (SAMMY manual, Table VI B.2): burst FWHM (ns), chi-squared width
        # Lambda0 (ns) with its uncertainty line, one channel width (ns) up to ECRNCH (eV).
        burst = [] if burst_ns is None else [f"BURST00000{burst_ns:10.1f}{0.0:10.1f}"]
        lines += [
            "",
            "RPI Resolution function",
            *burst,
            f"LAMBD00000{1000.0 / alpha:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}",
            f"{'':10}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}",
            f"CHANN00000{1000.0:10.1f}{CHANNEL_NS:10.1f}{0.0:10.1f}",
        ]
    return "\n".join(lines + ["", ""])


def cases(awr: float) -> dict[str, dict]:
    """Return every SAMMY case by name, with its stage, temperature, alpha and input."""
    out = {"s0": {"stage": "s0", "kelvin": 0, "alpha": 0, "nxtra": 0,
                  "inp": sammy_input("Ta-181 S0 unbroadened total cross section", awr, "s0")}}
    for nxtra in (NXTRA, NXTRA_CHECK):
        for kelvin in TEMPERATURES_K:
            out[f"s1_T{kelvin}_n{nxtra}"] = {
                "stage": "s1", "kelvin": kelvin, "alpha": 0, "nxtra": nxtra,
                "inp": sammy_input(f"Ta-181 S1 free-gas Doppler {kelvin} K NXTRA {nxtra}",
                                   awr, "s1", kelvin=kelvin, nxtra=nxtra)}
            for alpha in ALPHAS_PER_US:
                out[f"s2_a{alpha}_T{kelvin}_n{nxtra}"] = {
                    "stage": "s2", "kelvin": kelvin, "alpha": alpha, "nxtra": nxtra,
                    "inp": sammy_input(
                        f"Ta-181 S2 Doppler and RPI alpha {alpha}/us {kelvin} K NXTRA {nxtra}",
                        awr, "s2", kelvin=kelvin, alpha=alpha, nxtra=nxtra)}
    return out


def lpt_groups(lpt: str, pattern: str, case: Path) -> tuple[str, ...]:
    """Return the groups of the first match of a pattern in SAMMY.LPT, or stop."""
    found = re.search(pattern, lpt)
    if not found:
        raise SystemExit(f"{case}: SAMMY.LPT has no match for {pattern!r}")
    return found.groups()


def check_lpt(case: Path, spec: dict, nbins: int) -> dict[str, str]:
    """Check the SAMMY.LPT echo of the case and return the values for the record."""
    lpt = (case / "SAMMY.LPT").read_text()
    out = (case / "sammy.out").read_text()
    record = {"version": lpt_groups(lpt, r"SAMMY Version (\S+ - \w+)", case)[0]}
    for text in ("ENDF file has Lrf=  3", "REICH MOORE FORMALISM IS WANTED",
                 f"Number of experimental data points ={nbins:8d}"):
        if text not in lpt:
            raise SystemExit(f"{case}: SAMMY.LPT lacks {text!r}")
    if record["version"] != SAMMY_BUILD:
        raise SystemExit(f"{case}: SAMMY {record['version']}, not {SAMMY_BUILD}")
    record["resonances"] = lpt_groups(lpt, r"Total  number of resonances is\s+(\d+)", case)[0]
    record["radius_fm"] = lpt_groups(lpt, r"SPIN GROUP NUMBERS FOR THESE RADII\s+(\S+)", case)[0]
    if int(record["resonances"]) != RESONANCES or float(record["radius_fm"]) != RADIUS_FM:
        raise SystemExit(f"{case}: {record['resonances']} resonances, radius {record['radius_fm']}")
    kelvin, thick = lpt_groups(lpt, r"TEMPERATURE\s+THICKNESS\s+(\S+)\s+(\S+)", case)
    if float(kelvin) != spec["kelvin"] or float(thick) != THICKNESS_PER_B:
        raise SystemExit(f"{case}: SAMMY.LPT gives {kelvin} K and {thick} atoms/b")
    resolved = "SAMMY-RPI" in out
    if resolved != (spec["stage"] == "s2"):
        raise SystemExit(f"{case}: RPI broadening ran = {resolved} for stage {spec['stage']}")
    if spec["stage"] == "s0":
        return record
    record["doppler_ev"] = " ".join(lpt_groups(
        lpt, r"Doppler Width at Emin =\s*(\S+)\s+and at Emax =\s*(\S+)", case))
    record["aux_points"] = lpt_groups(lpt, r"Number of points in auxiliary grid =\s*(\d+)", case)[0]
    low = lpt_groups(lpt, r"Eminr\s+Emin\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", case)
    high = lpt_groups(lpt, r"Emaxd\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", case)
    record["grid_ev"] = f"Emind {low[0]} Eminr {low[2]} Emaxr {high[1]} Emaxd {high[3]}"
    if spec["stage"] == "s1":
        return record
    if "No resolution broadening occurred" in lpt:
        raise SystemExit(f"{case}: SAMMY.LPT reports points without resolution broadening")
    lam = [float(lpt_groups(lpt, rf"Lambda{i} \([^)]*\) =\s*(\S+)", case)[0]) for i in range(5)]
    chann = float(lpt_groups(lpt, r"0\.0\s+infinity\s+(\S+)", case)[0])
    chi_us = float(lpt_groups(
        lpt, r"Time width associated with chi-squared fnct is\s*(\S+)", case)[0])
    if lam != [1000.0 / spec["alpha"], 0.0, 0.0, 0.0, 0.0] or chann != CHANNEL_NS:
        raise SystemExit(f"{case}: SAMMY.LPT gives Lambda {lam} and channel {chann}")
    expected = spec.get("burst", BURST_NS)
    if expected is None:
        if "Burst full-width-at-half-max" in lpt:
            raise SystemExit(f"{case}: SAMMY.LPT gives a burst")
    else:
        burst = float(lpt_groups(
            lpt, r"Burst full-width-at-half-max \(nanosec\) =\s*(\S+)", case)[0])
        if burst != expected:
            raise SystemExit(f"{case}: SAMMY.LPT gives a burst of {burst} ns")
    # SAMMY ends the chi-squared term at 16 Lambda.
    if abs(chi_us - 16.0 / spec["alpha"]) > 1e-3:
        raise SystemExit(f"{case}: chi-squared time width {chi_us} us")
    record["chi_width_us"] = f"{chi_us}"
    return record


def run_case(exe: Path, work: Path, name: str, spec: dict,
             energies: list[str]) -> tuple[list[list[str]], dict[str, str]]:
    """Run SAMMY in work/name; return the SAMMY.LST rows in bins.txt order and the record."""
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
    record = check_lpt(case, spec, len(energies))
    print(f"{name}: {record}", flush=True)
    return list(reversed(rows)), record


def write_table(path: Path, header: list[str], energies: list[str],
                columns: list[list[str]]) -> None:
    """Write one row per bin: the bins.txt energy, then the SAMMY values."""
    rows = [" ".join([e, *values]) for e, values in zip(energies, zip(*columns))]
    path.write_text("\n".join(header + rows) + "\n")


def check_executable(exe: Path) -> None:
    """Stop unless exe is the SAMMY build of the references."""
    if hashlib.md5(exe.read_bytes()).hexdigest() != SAMMY_MD5:
        raise SystemExit(f"{exe}: MD5 is not {SAMMY_MD5}")


def run_specs(exe: Path, work: Path, jobs: int, specs: dict[str, dict],
              energies: list[str]) -> dict[str, tuple[list[list[str]], dict[str, str]]]:
    """Run the cases at most jobs at a time and return their rows and records."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {name: pool.submit(run_case, exe, work, name, spec, energies)
                   for name, spec in specs.items()}
        return {name: future.result() for name, future in futures.items()}


def diagnose(exe: Path, work: Path, jobs: int) -> None:
    """Print the S2 change at 300 K without a burst and with twice the burst."""
    check_executable(exe)
    energies = read_bins()
    awr = read_awr()
    bursts = (None, BURST_NS, 2.0 * BURST_NS)
    specs = {}
    for alpha in ALPHAS_PER_US:
        for burst in bursts:
            specs[f"diag_a{alpha}_b{burst or 0:g}"] = {
                "stage": "s2", "kelvin": 300, "alpha": alpha, "nxtra": NXTRA, "burst": burst,
                "inp": sammy_input(f"Ta-181 S2 alpha {alpha}/us 300 K burst {burst} ns",
                                   awr, "s2", kelvin=300, alpha=alpha, nxtra=NXTRA,
                                   burst_ns=burst)}
    results = run_specs(exe, work, jobs, specs, energies)
    for alpha in ALPHAS_PER_US:
        ref = [float(row[TRANSMISSION_COLUMN]) for row in results[f"diag_a{alpha}_b{BURST_NS:g}"][0]]
        stored = [line.split() for line in (HERE / f"s2_rpi_alpha{alpha}.txt").read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
        column = 1 + TEMPERATURES_K.index(300)
        drift = max(abs(float(row[column]) - r) for row, r in zip(stored, ref))
        print(f"alpha {alpha}/us, 300 K: {BURST_NS} ns run minus s2_rpi_alpha{alpha}.txt: "
              f"max |dT| {drift:.3e}")
        for burst in (None, 2.0 * BURST_NS):
            other = [float(row[TRANSMISSION_COLUMN])
                     for row in results[f"diag_a{alpha}_b{burst or 0:g}"][0]]
            k = max(range(len(ref)), key=lambda i: abs(other[i] - ref[i]))
            print(f"alpha {alpha}/us, 300 K: burst {burst} ns minus {BURST_NS} ns: "
                  f"max |dT| {abs(other[k] - ref[k]):.3e} at {energies[k]} eV")


def run_all(exe: Path, work: Path, jobs: int, only: str | None) -> None:
    """Run every case, write the reference files and print the NXTRA convergence."""
    check_executable(exe)
    energies = read_bins()
    specs = cases(read_awr())
    if only:
        specs = {only: specs[only]}
    results = run_specs(exe, work, jobs, specs, energies)
    if only:
        return
    versions = {record["version"] for _, record in results.values()}
    if len(versions) != 1:
        raise SystemExit(f"SAMMY versions differ between runs: {versions}")
    build = versions.pop()
    names = " ".join(f"T{k}" for k in TEMPERATURES_K)
    common = [
        f"# SAMMY {build}; Ta-181-mf2.endf (ENDF/B-VIII.1 MAT 7328, Reich-Moore) at the",
        "# e_center_ev of bins.txt, in bins.txt order. Written by generate.py sammy.",
    ]
    write_table(HERE / "s0_sigma.txt", common + [
        "# Unbroadened total cross section (barn), BROADENING IS NOT WANTED; SAMMY.LST column 4.",
        "# Columns: e_ev sigma_total_b",
    ], energies, [[row[SIGMA_COLUMN] for row in results["s0"][0]]])
    stages = {
        "s1_doppler.txt": ("s1", 0, [
            "# Transmission exp(-n sigma), n = 7.04e-4 atoms/b, free-gas Doppler only (no",
            f"# resolution), NXTRA = {NXTRA}; SAMMY.LST column 8. Temperatures in K.",
        ]),
    }
    for alpha in ALPHAS_PER_US:
        stages[f"s2_rpi_alpha{alpha}.txt"] = ("s2", alpha, [
            "# Transmission, n = 7.04e-4 atoms/b, free-gas Doppler and the RPI resolution",
            f"# function: chi-squared term only, Lambda0 = {1000.0 / alpha:.1f} ns (alpha = {alpha}"
            f" per us), a {BURST_NS:.0f} ns",
            f"# Gaussian burst, square channel {CHANNEL_NS:.0f} ns, not shifted to centre,"
            f" L = 25 m; NXTRA = {NXTRA}; SAMMY.LST column 8.",
        ])
    print(f"stage alpha kelvin eps_S max|D|(NXTRA {NXTRA}) eps_S/max|D|")
    eps_rows = []
    for file_name, (stage, alpha, text) in stages.items():
        prefix = f"{stage}_a{alpha}" if stage == "s2" else stage
        columns = [[row[TRANSMISSION_COLUMN] for row in results[f"{prefix}_T{k}_n{NXTRA}"][0]]
                   for k in TEMPERATURES_K]
        write_table(HERE / file_name, common + text + [f"# Columns: e_ev {names}"],
                    energies, columns)
        values = {k: [float(x) for x in col] for k, col in zip(TEMPERATURES_K, columns)}
        for kelvin in TEMPERATURES_K:
            check = [float(row[TRANSMISSION_COLUMN])
                     for row in results[f"{prefix}_T{kelvin}_n{NXTRA_CHECK}"][0]]
            eps = max(abs(a - b) for a, b in zip(values[kelvin], check))
            centre = min((300, 1100), key=lambda c: abs(c - kelvin))
            dmax = max(abs(h - l) / 2 for h, l in zip(values[centre + 1], values[centre - 1]))
            print(f"{stage} {alpha} {kelvin} {eps:.3e} {dmax:.4e} {eps / dmax:.4f}")
            eps_rows.append(f"{stage[1]} {kelvin} {alpha} {eps:.6e}")
    (HERE / "eps_s.txt").write_text("\n".join([
        f"# SAMMY {build}: eps_S = max|T(NXTRA = {NXTRA}) - T(NXTRA = {NXTRA_CHECK})| over the bins.",
        "# Columns: stage (1 = S1, 2 = S2), temperature (K), alpha (1/us, 0 for S1), eps_S.",
        *eps_rows,
    ]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    endf = sub.add_parser("endf", help=f"write {ENDF_FILE} from the cached ENDF/B-VIII.1 file")
    endf.add_argument("cached", type=Path)
    sub.add_parser("bins", help=f"write {BINS_FILE}")
    sammy = sub.add_parser("sammy", help="run SAMMY and write the s0, s1 and s2 files")
    sammy.add_argument("exe", type=Path, help="SAMMY executable")
    sammy.add_argument("work", type=Path, help="directory for the SAMMY runs")
    sammy.add_argument("--jobs", type=int, default=1, help="SAMMY runs at the same time")
    sammy.add_argument("--only", help="run only the case with this name")
    diag = sub.add_parser(
        "diagnose", help="print the S2 change without a burst and with twice the burst")
    diag.add_argument("exe", type=Path, help="SAMMY executable")
    diag.add_argument("work", type=Path, help="directory for the SAMMY runs")
    diag.add_argument("--jobs", type=int, default=1, help="SAMMY runs at the same time")
    args = parser.parse_args()
    if args.command == "endf":
        write_endf(args.cached)
    elif args.command == "bins":
        write_bins()
    elif args.command == "diagnose":
        diagnose(args.exe.resolve(), args.work.resolve(), args.jobs)
    else:
        run_all(args.exe.resolve(), args.work.resolve(), args.jobs, args.only)


if __name__ == "__main__":
    main()
