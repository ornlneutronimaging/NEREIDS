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

`crates/nereids-pipeline/tests/sammy_ta181_oracle.rs` compares NEREIDS with
these SAMMY references for VENUS Ta-181 thermometry. S2 compares the free-gas
transmission and its slope per K, folded over the bins with only the prompt
Ikeda-Carpenter term (α = 2 per µs, R = 0). S2b fits the temperature with
`fit_counts` to counts made from the S2 references. S3 calibrates the pulse with
`PulseCalibration::new` on counts made from SAMMY's crystal-lattice transmission
at 296 K, with the S2 pulse and the phonon spectrum `../sammy-clm/taper.clm`.

| File | Content |
|---|---|
| `Ta-181-mf2.endf` | MF1 MT451 and MF2 MT151 of ENDF/B-VIII.1 MAT 7328, with the CRLF bytes of the tape (MD5 `e9cae860785d7ea503430300e4428663`). |
| `s2_rpi_alpha2.txt` | SAMMY transmission at each VENUS bin centre, at 299, 300, 301, 1099, 1100 and 1101 K. |
| `eps_s.txt` | SAMMY's convergence at 300 K and 1100 K. |
| `s3_clm_rpi_alpha2.txt` | SAMMY crystal-lattice transmission at each VENUS bin centre, at 295, 296 and 297 K, to phonon order 1600. |
| `eps_s3.txt` | SAMMY's S3 convergence at 296 K: from phonon order 1600 to 3200 over all the bins (`eps_L`), and from NXTRA 29 to a run of NXTRA 99 on each of 7.8–9 and 190–200 eV alone, over its bins (`eps_S`). |
| `generate.py` | Writes the files above: `--set s2` the S2 files, `--set s3` the S3 files, both by default. |

The bins are t_j = 1.12 + 0.16 j µs (L = 25 m) with a centre energy from 7.8 to
200 eV. SAMMY is `8.1.pre-b1 - 0aa6ee20` (`sammy` MD5
`ef403c81ef531ae3511eb007f6c0080e`). Its RPI input has a 1 ns burst, which
selects the general RPI routine. Without a burst, `Chantd`
(`src/rpi/mrpi6.f90`, line 758) multiplies the small-delay kernel by an extra X.

```sh
python3 tests/data/sammy-ta181/generate.py endf <ENDF/B-VIII.1 Ta-181 tape>
python3 tests/data/sammy-ta181/generate.py sammy <sammy executable> <work directory> --jobs 8 --set s2
python3 tests/data/sammy-ta181/generate.py sammy <sammy executable> <work directory> --jobs 8 --set s3
```

### Phase 1 forward-model checks

Check 1 (`check1_the_4_28_ev_line_matches_sammy_rpi`) and check 3
(`check3_lattice_matches_the_independent_reference`) are in
`sammy_ta181_oracle.rs`; check 2 (`check2_counts_match_the_independent_fold`) is
in `crates/nereids-pipeline/src/counts_fit.rs`. SAMMY is `8.2.b4` from RNSD/SAMMY
commit `c974c02c`; each SAMMY file records the executable's MD5.

| File | Content |
|---|---|
| `p1a_rpi_alpha2.txt`, `p1a_rpi_alpha2_eps.txt` | Check 1: SAMMY free-gas transmission with the RPI prompt term over the 4.280412 eV window, at 299–301 and 1099–1101 K in ½ K steps, and its change from NXTRA 29 to 99. |
| `p1b_counts.txt`, `p1b_counts_eps.txt`, `p1b_jacobian.txt` | Check 2: expected open and sample counts per bin, the open counts' change on the halved integration grid, and the counts' derivatives with respect to the pulse numbers, t0 and L. |
| `p1c_lattice_reference.txt`, `p1c_lattice_reference_eps.txt` | Check 3: lattice transmission without instrument, Debye θ_D = 240 K, recoil E/A, at 300.79–302.79 and 1092–1094 K in ½ K steps, and its change on a refined run (Fourier step halved and period doubled, one-phonon panels doubled, σ every 1e-5 instead of 2e-5 eV). |
| `generate_phase1.py` | Writes the check 1 files with SAMMY. |
| `counts_reference.py`, `counts_kernel.py` | Write the check 2 files from SAMMY's transmission without instrument, with no NEREIDS code. |
| `lattice_reference.py` | Writes the check 3 files from NEREIDS's unbroadened cross section, with its own lattice broadening; `check` compares its kernel with the spectrum's cumulants. |

```sh
python3 tests/data/sammy-ta181/generate_phase1.py <sammy executable> <work directory> --jobs 16
python3 tests/data/sammy-ta181/counts_reference.py write <sammy executable> <work directory> --jobs 16
cargo run --release -p nereids-physics --example unbroadened_total -- tests/data/sammy-ta181/Ta-181-mf2.endf 0.05 1e-5 20495001 <sigma file>
python3 tests/data/sammy-ta181/lattice_reference.py check 5.05e-2
python3 tests/data/sammy-ta181/lattice_reference.py write <sigma file> 5.05e-2 --jobs 20
```

Run them on an analysis server: the SAMMY runs take hours, and the scripts need
numpy (check 2 also scipy and numba).

## sammy-clm

`crates/nereids-physics/tests/lattice_doppler.rs` compares NEREIDS with these
SAMMY `8.1.pre-b1 - 0aa6ee20` runs for U-238 from 6.4 to 6.9 eV. The inputs
are `clm.inp`, `u238.par` and `grid.dat`, and the crystal-lattice runs also
read the phonon spectrum `taper.clm`. The S3 runs of `sammy-ta181` read it too,
with its phonon order set by their `generate.py`.

| File | Content |
|---|---|
| `answers.txt` | Columns 1 and 4 of `SAMMY.LST` for `clm.inp`, and column 2 of `SAMUNB.DAT` at those energies for `clm.inp` with `PUT UNBROADENED CROSs sections into plot file` after its clm line. |
| `free_gas.txt` | Columns 1 and 4 of `SAMMY.LST` for `clm.inp` without its clm line. |

## debye-teff

`crates/nereids-physics/tests/debye_effective_temperature.rs` compares
`PhononSpectrum::effective_temperature_k` for Debye spectra with the continuous
integral. `integral.py` prints the four reference values, θ_D = 225 and 240 K at
300 and 301.79 K, computed with mpmath at 30 digits and printed to 12 significant
digits.
