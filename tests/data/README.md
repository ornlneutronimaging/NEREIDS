# NEREIDS Test Data

The ENDF, SAMMY and synthetic fixtures in this directory support the Rust and
Python tests. Large PLEIADES fixtures are optional local research data, not a
submodule or a dependency of the test suite.

## Optional PLEIADES data

`tests/data/pleiades_data/` is ignored. Existing local checkouts and project
saves can remain there. A normal NEREIDS clone does not fetch these files.

The external PLEIADES test-data repository is
<https://code.ornl.gov/sns-hfir-scse/infrastructure/test-data/pleiades_data>.
Its `LANL-ORNL_example.tif` contains synthetic transmission images, which
are not raw open-beam/sample counts and cannot be supplied to the counts
likelihood. Its VENUS Hf sample and open-beam HDF5 files contain raw counts
in rustpix NeXus format. Their time-bin edges and run charge ratio must be
supplied to a counts fit.

The VENUS fixtures are a 256×256 crop of a private measurement cube, with
4367 time bins and a nominal flight path of 25 m. The HDF5 files do not
store proton charge; the original run's approximate sample/open charge ratio
was 5.98. This approximate value is provenance, not a measured calibration
for another run. Regeneration requires the maintainers' private source data
and extraction tooling.

## venus/aggregated_hf_120min.npz

The retained aggregated VENUS Hf spectrum has `energies_ev`, `sample_counts`,
`open_beam_counts` and `pc_ratio`. It comes from the same private measurement
cube. An energy-center array is not a substitute for the original time-bin
edges required by the counts API; this archive is research evidence, not an
end-to-end counts-fitting validation fixture.

## sammy-ta181

`sammy-ta181/` holds SAMMY references for VENUS Ta-181 resonance thermometry.
`crates/nereids-pipeline/tests/sammy_ta181_oracle.rs` compares NEREIDS with
them in four stages. S0 compares the unbroadened total cross section at the
bin-centre energies. S1 compares the free-gas transmission at 7.04e-4 atoms/b
and its slope per K at those energies, at 300 K and 1100 K. S2 compares that
transmission and slope folded over the bins with only the prompt
Ikeda-Carpenter term (α = 2 and 8 per µs, R = 0), with a unit beam on the
test's own flight-time grids. S2b fits the temperature with `fit_counts` to
counts made from the S2 references at α = 2 per µs, and checks its error,
convergence and σ_T. No stage compares the counts that `fit_counts` predicts
with SAMMY. The test's criteria detect a 1 K error at 300 K.

| File | Content |
|---|---|
| `Ta-181-mf2.endf` | MF1 MT451 and MF2 MT151 of MAT 7328 with their SEND and FEND records, then MEND and TEND. |
| `bins.txt` | 3246 VENUS time bins: `t_lo_us t_hi_us e_center_ev`. |
| `s0_sigma.txt` | S0: unbroadened total cross section (barn) at each `e_center_ev`. |
| `s1_doppler.txt` | S1: free-gas Doppler transmission at 299, 300, 301, 1099, 1100 and 1101 K. |
| `s2_rpi_alpha2.txt`, `s2_rpi_alpha8.txt` | S2: Doppler and RPI resolution transmission at the same temperatures. |
| `eps_s.txt` | SAMMY convergence eps_S of S1 and S2 for each temperature and α. |
| `generate.py` | Standard-library Python 3.9 or later. It writes all the files above. |

The rows of the `s*` files follow `bins.txt` (time order, decreasing energy).
The first column is the `e_center_ev` string of `bins.txt`. Each other value is
the SAMMY.LST string, with 10 decimals.

### Evaluation and bins

The source is the cached ENDF/B-VIII.1 Ta-181 tape (MD5
`e9cae860785d7ea503430300e4428663`). `Ta-181-mf2.endf` holds lines 2 to 1696
and the last two lines of that tape, byte for byte, with its CRLF line ends.
No value is changed. The resolved range is Reich-Moore (LRF=3) up to 2554 eV,
with 565 l=0 resonances, AP = 7.86 fm and NAPS = 1.

The bin edges are t_j = 1.12 + 0.16 j µs, j = 792 to 4038, with L = 25 m and
t0 = 0. e_center_ev = (m_n/2)(L/t_c)² at the bin centre t_c, with the CODATA
2018 neutron mass and electron volt. `bins.txt` keeps the bins with
7.8 eV ≤ e_center_ev ≤ 200 eV, with 12 significant digits.

### SAMMY build

The SAMMY executable prints `SAMMY Version 8.1.pre-b1 - 0aa6ee20`, commit
`0aa6ee20` of code.ornl.gov `RNSD/SAMMY` (2025-09-03), built on 2025-09-04
(`sammy` MD5 `ef403c81ef531ae3511eb007f6c0080e`, `libsammyLib.so` MD5
`666adae132f0bf3b012559c2a5c7cdcd`). It ran with the MKL libraries of the SAMMY
8.1.pre-b1 bundle and one thread per run. `generate.py` stops on an executable
with another MD5 or a SAMMY.LPT with another version, and `generate.py endf`
stops on a tape with another MD5.

SAMMY's √(m/2) is 3.1e-8 below CODATA 2018. Its Boltzmann constant,
8.617342e-5 eV/K, is 1.01e-6 above CODATA 2018: SAMMY's kT at 300 K is the
CODATA kT at 300.0003 K.

### Input

The S2 input for α = 2 per µs at 300 K (SAMMY manual, Tables VI A.1 and VI B.2):

```text
Ta-181 S2 Doppler and RPI alpha 2/us 300 K NXTRA 29
Ta181     180.948030    7.0000  250.0000             29
INPUT IS ENDF/B FILE
TWENTY
USE FREE GAS MODEL OF DOPPLER BROADENING
DO NOT SOLVE BAYES EQUATIONS
PRINT THEORETICAL VALUES
GENERATE PLOT FILE AUTOMATICALLY
DO NOT SHIFT RPI RESOLUTION FUNCTION TO CENTER

     300.0      25.0       0.0       0.0       0.0
       0.0  0.000704
TRANSMISSION

RPI Resolution function
BURST00000       1.0       0.0
LAMBD00000     500.0       0.0       0.0       0.0       0.0
                 0.0       0.0       0.0       0.0       0.0
CHANN00000    1000.0     160.0       0.0

```

- α = 8 per µs uses Λ0 = 125.0 ns. S1 has no shift command and no card set 14.
  S0 has `BROADENING IS NOT WANTED`, no card 5, no NXTRA and
  `TOTAL CROSS SECTION`.
- The data file has the `TWENTY` format: `e_center_ev`, the value 1.0 and the
  uncertainty 0.1, in increasing energy.
- AW is AWR × 1.00866491578 amu, rounded to six decimals (2.4e-9). CRFN = 0
  keeps the ENDF radius.
- With τ = 0 the RPI moderator term is t² e^(−t/Λ)/(2Λ³) (SAMMY manual,
  Eq. III C3 a.2), the Gamma(3) delay of an Ikeda-Carpenter pulse with
  α = 1/Λ and R = 0. The channel term is a 160 ns square centred on the bin
  time. SAMMY ends the moderator term at 16 Λ and normalises the weights. The
  tail it leaves out is e^(−16)(1 + 16 + 128) = 1.6e-5 of the term.

### The burst

Without a burst, SAMMY computes the moderator and channel terms in its routine
`Chantd` (`src/rpi/mrpi6.f90`, line 758 at `0aa6ee20` and at `339d9f83`). For
X = (t_m − t′ + C/2)/Λ < 0.5 and t_m − t′ + C/2 ≤ C, that routine gives
(X/C)·F(X), where its other branch and the RPI definition give F(X)/C, with
F(X) = 1 − e^(−X)(1 + X + X²/2). The expansion of SAMMY's `Abcexp`
coefficients shows that (C(1 + X) − B·X/2)·X³ = F(X), so the extra factor is
X. The kernel then loses weight at the shortest delays.

A 1 ns Gaussian burst selects SAMMY's general RPI routine, which has no such
branch. `generate.py diagnose` runs S2 at 300 K with a 1 ns burst, without a
burst and with a 2 ns burst. It prints the largest change of the new 1 ns run
from the stored S2 tables, and the largest change of the other two runs from
the new 1 ns run. The new 1 ns run equals the stored tables. Without a burst
the change is 6.6e-4 (α = 2, at 99.32 eV)
and 4.9e-4 (α = 8, at 99.50 eV). With a 2 ns burst it is 2.0e-7 (α = 2) and
2.2e-6 (α = 8).

### Checks

`generate.py` stops if a check fails. In each SAMMY.LPT it finds the build
above, `ENDF file has Lrf=  3`, `REICH MOORE FORMALISM IS WANTED`,
565 resonances, the radius 7.86 fm, 3246 data points, the temperature and the
thickness 7.04e-4 atoms/b. RPI broadening runs only in S2. In S2 it finds
Λ0 = 500.0 or 125.0 ns, Λ1 to Λ4 = 0, a 1.000 ns burst, one channel width of
160.0 ns from 0 eV to infinity, a moderator time width of 16 Λ, and no
`No resolution broadening occurred` message.

### Convergence

eps_S = max over the bins of |T(NXTRA = 29) − T(NXTRA = 99)|. The references
use NXTRA = 29. D = [T(T+1) − T(T−1)]/2 per K. The test requires
eps_S < 0.02 max|D|.

| Stage | eps_S 300 K | eps_S 1100 K | max\|D\| 300 K | max\|D\| 1100 K |
|---|---|---|---|---|
| S1 | 2.35e-7 | 7.2e-8 | 8.438e-4 | 2.731e-4 |
| S2, α = 2 | 9.8e-8 | 9.4e-8 | 4.525e-4 | 1.610e-4 |
| S2, α = 8 | 2.4e-7 | 2.1e-7 | 6.509e-4 | 2.472e-4 |

eps_S does not include the 16 Λ cut of the moderator term, which can move a
transmission by up to about 1.3e-5 whatever NXTRA is. The test prints NEREIDS
minus SAMMY; in S2 it is at most 1.3e-5 (α = 2, 36.9 eV).

### Reproduce

Run these commands in `sammy-ta181/`:

```sh
python3 generate.py endf ~/Library/Caches/nereids/endf/ENDF-B-VIII.1/Ta-181.endf
python3 generate.py bins
LD_LIBRARY_PATH=<SAMMY MKL directory> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 generate.py sammy <sammy executable> <work directory> --jobs 8
LD_LIBRARY_PATH=<SAMMY MKL directory> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 generate.py diagnose <sammy executable> <work directory> --jobs 6
```

The `sammy` command runs 37 SAMMY cases, checks each SAMMY.LPT, and writes the
four `s*` files and `eps_s.txt`.
The `diagnose` command runs 6 cases in the work directory and does not change
the reference files.
