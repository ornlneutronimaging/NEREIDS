#!/usr/bin/env python3
"""Write SAMMY reference cross sections for the File 2 extracts in tests/data/endf/mf2.

For each isotope, run SAMRML once and the SAMMY main program for the total,
elastic and capture cross sections, unbroadened, on one energy grid, and write
``tests/data/sammy-lrf7/<isotope>.txt``. SAMRML does not read background
R-function (KBK) records; for a file with them its columns hold ``nan``.

Usage: ``python3 scripts/sammy_lrf7_reference.py SAMMY_BIN WORK_DIR ISOTOPE...``
where SAMMY_BIN holds the ``sammy`` and ``samrml`` executables. Run it from the
repository root. With ``--transmission KELVIN ATOMS_PER_BARN`` before the
arguments it writes ``<isotope>-transmission.txt`` instead: SAMMY main
transmission, free-gas Doppler broadened at KELVIN with no resolution function,
on a 0.5-200 eV grid spaced at a tenth of the Doppler width. SAMMY builds its
broadening grid from the data grid, so a coarser grid under-broadens it.

The grid has 1200 log-spaced points over 0.5-200 eV; the energies E, E +/- G/2
and E +/- G of every level in that band, G being its total width; and 40
log-spaced points over the range, inside its ends.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

NEUTRON_MASS_AMU = 1.00866491578  # SAMMY's value, as printed in SAMMY.LPT


def number(text: str) -> float:
    """ENDF real, with or without the E."""
    text = text.strip()
    if not text:
        return 0.0
    for i in range(1, len(text)):
        if text[i] in "+-" and text[i - 1] not in "eE":
            return float(text[:i] + "e" + text[i:])
    return float(text)


class Records:
    def __init__(self, lines: list[str]):
        self.lines = lines
        self.i = 0

    def cont(self) -> tuple:
        line = self.lines[self.i]
        self.i += 1
        f = [line[11 * k:11 * k + 11] for k in range(6)]
        return (number(f[0]), number(f[1]), *(int(number(x)) for x in f[2:]))

    def values(self, n: int) -> list[float]:
        out: list[float] = []
        while len(out) < n:
            line = self.lines[self.i]
            self.i += 1
            out += [number(line[11 * k:11 * k + 11]) for k in range(min(6, n - len(out)))]
        return out


def levels(mf2: list[str]) -> tuple[float, float, float, list[tuple[float, float]], bool]:
    """AWR, EL, EH, (E, total width) of each level of the first range, and
    whether the range has background R-function records."""
    r = Records(mf2)
    _, awr, *_ = r.cont()
    r.cont()
    el, eh, _, lrf, nro, _ = r.cont()
    if nro:
        raise SystemExit("NRO != 0 is not handled")
    found = []
    background = False
    if lrf == 7:
        _, _, _, _, njs, _ = r.cont()
        _, _, npp, _, _, _ = r.cont()
        r.values(12 * npp)
        for _ in range(njs):
            _, _, kbk, _, _, nch = r.cont()
            r.values(6 * nch)
            _, _, _, nrs, _, nx = r.cont()
            rows = r.values(6 * nx)
            stride = 6 * (nch // 6 + 1)
            for k in range(nrs):
                row = rows[k * stride:(k + 1) * stride]
                found.append((row[0], sum(abs(w) for w in row[1:nch + 1])))
            for _ in range(kbk):
                background = True
                _, _, _, lbk, _, _ = r.cont()
                if lbk == 2:
                    r.cont()
                    r.values(5)
    elif lrf == 3:
        _, _, _, _, nls, _ = r.cont()
        for _ in range(nls):
            _, _, _, _, _, nrs = r.cont()
            rows = r.values(6 * nrs)
            for k in range(nrs):
                er, _, gn, gg, gfa, gfb = rows[6 * k:6 * k + 6]
                found.append((er, abs(gn) + abs(gg) + abs(gfa) + abs(gfb)))
    else:
        raise SystemExit(f"LRF={lrf} is not handled")
    return awr, el, eh, found, background


def grid(el: float, eh: float, found: list[tuple[float, float]]) -> list[float]:
    points = {0.5 * (400.0 ** (i / 1199)) for i in range(1200)}
    for e, width in found:
        for x in (e - width, e - width / 2, e, e + width / 2, e + width):
            if 0.5 <= x <= 200.0:
                points.add(x)
    low = max(el, 1e-5)
    for i in range(1, 41):
        points.add(low * (eh / low) ** (i / 41))
    return sorted(float(f"{x:.12e}") for x in points)


def doppler_grid(kelvin: float, awr: float) -> list[float]:
    """0.5-200 eV in steps of a tenth of the Doppler width 2*sqrt(E*kT/A)."""
    kt = 8.617333262e-5 * kelvin
    points = [0.5]
    while points[-1] < 200.0:
        points.append(points[-1] + 0.2 * math.sqrt(points[-1] * kt / awr))
    points[-1] = 200.0
    return [float(f"{x:.12e}") for x in points]


def run(exe: Path, answers: str, cwd: Path) -> None:
    result = subprocess.run(
        [str(exe)], input=answers, text=True, cwd=cwd, capture_output=True, check=False
    )
    (cwd / f"{exe.name}.out").write_text(result.stdout + result.stderr)
    if "Normal" not in result.stdout:
        raise SystemExit(f"{exe} failed in {cwd}; see {exe.name}.out")


def samrml(bin_dir: Path, work: Path, energies: list[float]) -> list[list[str]]:
    work.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{e:20.12E}\n" for e in energies) + f"{0.0:20.12E}\n"
    (work / "energies.dat").write_text(text)
    run(bin_dir / "samrml", "energies.dat\nmf2.endf\n0\nn\n", work)
    rows = []
    for line in (work / "OUTPUT.LPT").read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            rows.append([x.replace("D", "E") for x in fields[2:]])
    if len(rows) != len(energies):
        raise SystemExit(f"SAMRML gave {len(rows)} rows for {len(energies)} energies")
    return rows


def sammy(
    bin_dir: Path,
    work: Path,
    name: str,
    awr: float,
    energies: list[float],
    reaction: str,
    broadening: tuple[float, float] | None = None,
) -> list[str]:
    work.mkdir(parents=True, exist_ok=True)
    (work / "data.dat").write_text("".join(f"{e:20.12E}{1.0:20.6f}\n" for e in energies))
    inp = "\n".join(
        [
            f"{name} LRF=7 reference",
            f"{name:<10}{awr * NEUTRON_MASS_AMU:10.6f}{energies[0] * 0.999:10.4E}{energies[-1] * 1.001:10.4E}",
            "INPUT IS ENDF/B FILE",
            "twenty",
            "use ten percent data uncertainty",
            "do not solve bayes equations",
            "print theoretical values",
            *([] if broadening else ["broadening is not wanted"]),
            "generate plot file automatically",
            "",
            *(
                [f"{broadening[0]:10.4f}      25.0       0.0       0.0       0.0", f"       0.0{broadening[1]:10.3E}"]
                if broadening
                else ["     300.0      25.0"]
            ),
            reaction,
            "",
        ]
    )
    (work / "case.inp").write_text(inp)
    run(bin_dir / "sammy", "case.inp\nmf2.endf\ndata.dat\n\n", work)
    rows = [line.split() for line in (work / "SAMMY.LST").read_text().splitlines() if line.strip()]
    if len(rows) != len(energies):
        raise SystemExit(f"SAMMY gave {len(rows)} rows for {len(energies)} energies in {work}")
    # SAMMY.LST gives energies in keV when they reach above 1 keV, to 10 decimals.
    unit = 1000.0 if energies[-1] > 1e3 else 1.0
    for e, row in zip(energies, rows):
        if abs(float(row[0]) * unit - e) > 0.6e-10 * unit + 1e-12 * e:
            raise SystemExit(f"SAMMY energy {row[0]} does not match {e} in {work}")
    return [row[3] for row in rows]


def main() -> None:
    args = sys.argv[1:]
    broadening = None
    if args[0] == "--transmission":
        broadening = (float(args[1]), float(args[2]))
        args = args[3:]
    bin_dir, work_root = Path(args[0]), Path(args[1])
    commit = subprocess.run(
        ["git", "-C", str(bin_dir), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    for name in args[2:]:
        source = Path("tests/data/endf/mf2") / f"{name}.endf"
        mf2 = [line for line in source.read_text().splitlines() if line[70:72] == " 2"]
        awr, el, eh, found, background = levels(mf2)
        energies = grid(el, eh, found)
        if broadening:
            energies = doppler_grid(broadening[0], awr)
            work = work_root / f"{name}-transmission"
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            (work / "mf2.endf").write_text("\n".join(mf2) + "\n")
            sigma = sammy(bin_dir, work, name.replace("-", ""), awr, energies, "transmission", broadening)
            # For transmission SAMMY.LST gives -ln(T)/n in column 4 and T in column 8.
            listed = [line.split() for line in (work / "SAMMY.LST").read_text().splitlines() if line.strip()]
            out = Path("tests/data/sammy-lrf7") / f"{name}-transmission.txt"
            header = [
                f"# SAMMY transmission for tests/data/endf/mf2/{name}.endf: free-gas Doppler at",
                f"# {broadening[0]} K, {broadening[1]} atoms/barn, no resolution function.",
                f"# SAMMY build: {commit or 'unknown'}; generated by scripts/sammy_lrf7_reference.py.",
                "# Columns: energy (eV); -ln(T)/n (barn); transmission T.",
            ]
            lines = [f"{e:.12e} {x} {row[7]}" for e, x, row in zip(energies, sigma, listed)]
            out.write_text("\n".join(header + lines) + "\n")
            print(f"{name}: {len(energies)} energies -> {out}")
            continue
        work = work_root / name
        if work.exists():
            shutil.rmtree(work)
        for sub in ("samrml", "total", "elastic", "capture"):
            (work / sub).mkdir(parents=True)
            (work / sub / "mf2.endf").write_text("\n".join(mf2) + "\n")
        rml = [["nan", "nan"]] * len(energies) if background else samrml(bin_dir, work / "samrml", energies)
        main_runs = {
            r: sammy(bin_dir, work / r, name.replace("-", ""), awr, energies, k)
            for r, k in (("total", "total cross section"), ("elastic", "elastic"), ("capture", "capture"))
        }
        extra = len(rml[0]) - 2
        header = [
            f"# SAMMY reference for tests/data/endf/mf2/{name}.endf, unbroadened.",
            f"# SAMMY build: {commit or 'unknown'}; generated by scripts/sammy_lrf7_reference.py.",
            "# Columns: energy (eV); SAMRML elastic and nonelastic"
            + "".join(f", SAMRML reaction to pair {p + 3}" for p in range(extra))
            + "; SAMMY total, elastic, capture (barn).",
        ]
        lines = [
            " ".join([f"{e:.12e}", *row, main_runs["total"][i], main_runs["elastic"][i], main_runs["capture"][i]])
            for i, (e, row) in enumerate(zip(energies, rml))
        ]
        out = Path("tests/data/sammy-lrf7") / f"{name}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(header + lines) + "\n")
        print(f"{name}: {len(energies)} energies -> {out}")


if __name__ == "__main__":
    main()
