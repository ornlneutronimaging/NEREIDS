# SAMMY references for the File 2 extracts

Each `<isotope>.txt` holds unbroadened cross sections of
`tests/data/endf/mf2/<isotope>.endf` (ENDF/B-VIII.1) from two SAMMY programs:
SAMRML, the LRF=7 point cross-section program, and the SAMMY main program with
`INPUT IS ENDF/B FILE` and `BROADENING IS NOT WANTED`.
`crates/nereids-physics/tests/lrf7_sammy.rs` compares NEREIDS with both.

## Source

- SAMMY `master` at commit `339d9f83` (code.ornl.gov `RNSD/SAMMY`, 2026-08-18),
  built on 2026-10-20 with its pixi environment (GCC and gfortran 15.3).
- `scripts/sammy_lrf7_reference.py` wrote the files. It gives both programs the
  MF=2 records only; SAMRML reads them with material number 0.
- The SAMMY main program takes the masses from the file: MB = AWR × 1.00866491578 amu.
  Its √(2m)/ħ is 2.19680781008e-4 fm⁻¹ eV^-½, 5.5e-8 above the CODATA 2018 value.

## Columns

Energy in eV; SAMRML elastic and nonelastic, then one reaction column per
particle pair beyond the second (Fe-57: inelastic, MT=51); SAMMY main total,
elastic and capture. Cross sections are in barn. SAMRML prints 13 significant
digits; SAMMY main prints 10 decimals.

SAMRML does not read background R-function (KBK) records, so its columns for
Sr-88 hold `nan`.

## Grid

1200 log-spaced points over 0.5–200 eV; E, E ± Γ/2 and E ± Γ of every level in
that band, Γ being the level's total width; and 40 log-spaced points over the
resolved range, inside its ends.

## Doppler-broadened transmission

`W-182-transmission.txt` holds the SAMMY main program's transmission at
1e-4 atoms/barn, free-gas Doppler broadened at 296 K with no resolution
function: energy (eV), −ln(T)/n (barn) and T. Its grid runs over 0.5–200 eV
in steps of a tenth of the Doppler width, 2√(E·kT/A)/10. SAMMY builds its
broadening grid from the data grid; on the coarser grid of the unbroadened
files it under-broadens the resonance wings by up to 3.5 %.

