# Archive inventory and cold audit

Archive examined: `/Users/chenzhang/Downloads/Archive.zip`

- SHA-256: `8b5afb3f1a4efcb5b822a955a633403afe48b62baa05f0b49c0d55567ab9e0d2`
- Integrity: `unzip -t /Users/chenzhang/Downloads/Archive.zip` ended with
  `No errors detected`.
- Contents: 71 members and 4,468,904 uncompressed bytes. No absolute or
  parent-traversal paths were found.
- Both notebooks pass `nbformat.validate`; all code, markdown, text outputs,
  and nine decoded embedded plots were inspected. The nine-page PDF was
  rendered and visually checked.
- The only experimental numeric cache sufficient for a spectrum-level replay
  is `region_counts.npz`: 5,304 bins spanning 4.525 eV–2.2788 MeV on its stored
  axis. The notebooks first restrict it to 4–120 eV, then fit 8–45 eV. Raw TIFF
  cubes, NeXus files, the UDR/FTS file, and spatial fit arrays are not included.

Exact validation commands (all exit 0 unless stated):

```text
unzip -t /Users/chenzhang/Downloads/Archive.zip
pixi run python investigation/archive_audit.py --extract-images /tmp/nereids-notebook-images-8b5afb3f1a4e
pdfinfo /tmp/nereids-archive-8b5afb3f1a4e/01_spectral_lineshape_bias/report/report.pdf
pdftoppm -png -r 120 /tmp/nereids-archive-8b5afb3f1a4e/01_spectral_lineshape_bias/report/report.pdf /tmp/nereids-prior-report-8b5afb3f1a4e
```

The durable audit reported SHA match `true`, integrity bad member `null`, 71
members, 4,468,904 uncompressed bytes, no unsafe paths, notebook code execution
counts 10/10 and 9/10, and nine extracted PNG outputs. `pdfinfo` reported 9
letter-size pages and `pdftoppm` rendered all
nine; the page images were visually inspected. The following non-evidentiary
contact-sheet attempt exited 1 because ImageMagick could not load a default
font; it was not used for inspection:

```text
montage /tmp/nereids-prior-report-8b5afb3f1a4e-1.png /tmp/nereids-prior-report-8b5afb3f1a4e-2.png /tmp/nereids-prior-report-8b5afb3f1a4e-3.png /tmp/nereids-prior-report-8b5afb3f1a4e-4.png /tmp/nereids-prior-report-8b5afb3f1a4e-5.png /tmp/nereids-prior-report-8b5afb3f1a4e-6.png /tmp/nereids-prior-report-8b5afb3f1a4e-7.png /tmp/nereids-prior-report-8b5afb3f1a4e-8.png /tmp/nereids-prior-report-8b5afb3f1a4e-9.png -thumbnail 300x400 -tile 3x3 -geometry +4+4 /tmp/nereids-prior-report-8b5afb3f1a4e-contact.png
```

## Complete member inventory

Sizes are the ZIP member's uncompressed and compressed byte counts. Roles state
how each item was used in the prior work or this audit.

| Path | Detected type | Bytes | Compressed bytes | Role |
|---|---:|---:|---:|---|
| `01_spectral_lineshape_bias/` | directory | 0 | 0 | Container directory; no payload. |
| `__MACOSX/._01_spectral_lineshape_bias` | AppleDouble encoded Macintosh file | 220 | 93 | macOS archive metadata; no research role. |
| `01_spectral_lineshape_bias/.DS_Store` | Apple Desktop Services Store | 6148 | 361 | macOS archive metadata; no research role. |
| `__MACOSX/01_spectral_lineshape_bias/._.DS_Store` | AppleDouble encoded Macintosh file | 120 | 53 | macOS archive metadata; no research role. |
| `01_spectral_lineshape_bias/HYPOTHESES.md` | Unicode text, UTF-8 text | 13257 | 5801 | Prior hypothesis/status ledger; later findings contain contradictions noted below. |
| `01_spectral_lineshape_bias/PROBLEM_STATEMENT.md` | Unicode text, UTF-8 text | 14559 | 6768 | Original research framing, observations, and proposed experiments. |
| `01_spectral_lineshape_bias/README.md` | Unicode text, UTF-8 text | 1509 | 868 | Archive research-package index and run instructions. |
| `01_spectral_lineshape_bias/figures/` | directory | 0 | 0 | Container directory; no payload. |
| `01_spectral_lineshape_bias/scripts/` | directory | 0 | 0 | Container directory; no payload. |
| `01_spectral_lineshape_bias/FINDINGS.md` | Unicode text, UTF-8 text | 17147 | 7800 | Prior synthesized findings and causal claims; cold-audited below. |
| `01_spectral_lineshape_bias/report/` | directory | 0 | 0 | Container directory; no payload. |
| `01_spectral_lineshape_bias/data/` | directory | 0 | 0 | Container directory; no payload. |
| `01_spectral_lineshape_bias/figures/diag_residuals.png` | PNG image data, 1690 x 910, 8-bit/color RGBA, non-interlaced | 95422 | 84579 | Cached residual-by-energy and parity diagnostic. |
| `01_spectral_lineshape_bias/figures/resolution_sensitivity.png` | PNG image data, 1690 x 585, 8-bit/color RGBA, non-interlaced | 60073 | 53569 | Cached model/library sensitivity comparison. |
| `01_spectral_lineshape_bias/figures/grid_consistency.png` | PNG image data, 1690 x 650, 8-bit/color RGBA, non-interlaced | 80151 | 72448 | Cached density/temperature consistency plot; its physical interpretation is disputed. |
| `01_spectral_lineshape_bias/figures/width_signature.png` | PNG image data, 1170 x 650, 8-bit/color RGBA, non-interlaced | 92689 | 89192 | Cached residual-width/signature diagnostic. |
| `01_spectral_lineshape_bias/figures/exp07_feature_14ev.png` | PNG image data, 1690 x 585, 8-bit/color RGBA, non-interlaced | 94019 | 89864 | Cached 14 eV feature comparison. |
| `01_spectral_lineshape_bias/figures/exp06_leverage.png` | PNG image data, 1430 x 520, 8-bit/color RGBA, non-interlaced | 28489 | 24134 | Cached heuristic line-leverage chart. |
| `01_spectral_lineshape_bias/scripts/exp07_feature_14ev.py` | Python script text executable, Unicode text, UTF-8 text | 5376 | 2189 | Analyzes Ta-181 peaks/equivalent width near 14 eV; does not run the promised contaminant scan. |
| `01_spectral_lineshape_bias/scripts/exp10_density_constrained.py` | Python script text executable, Unicode text, UTF-8 text | 3528 | 1427 | Fits hot spectra with density constrained to nominal. |
| `01_spectral_lineshape_bias/scripts/exp03_nuclear_data.py` | Python script text executable, Unicode text, UTF-8 text | 3960 | 1642 | Compares evaluated Ta-181 libraries with the archived UDR. |
| `01_spectral_lineshape_bias/scripts/exp02_resolution_correction.py` | Python script text executable, Unicode text, UTF-8 text | 6004 | 2346 | Calibrates Gaussian, corrected UDR, and IC families and evaluates sample fits. |
| `01_spectral_lineshape_bias/scripts/exp02b_fast.py` | Python script text executable, Unicode text, UTF-8 text | 5079 | 1803 | Reduced/fast resolution-family sensitivity replay. |
| `01_spectral_lineshape_bias/scripts/exp08_decomposition.py` | Python script text executable, Unicode text, UTF-8 text | 6396 | 2529 | Constructs the archived non-orthogonal temperature-sensitivity decomposition. |
| `01_spectral_lineshape_bias/scripts/anatomy.py` | Python script text executable, ASCII text | 2937 | 1229 | Residual-shape helper; two percentage labels are mathematically overstated. |
| `01_spectral_lineshape_bias/scripts/exp05_calib_and_consistency.py` | Python script text executable, Unicode text, UTF-8 text | 6115 | 1964 | Tests calibration transfer and RT/hot density consistency. |
| `01_spectral_lineshape_bias/scripts/exp09_ic_library_grid.py` | Python script text executable, Unicode text, UTF-8 text | 4818 | 1969 | Calibrates IC separately under six nuclear libraries. |
| `01_spectral_lineshape_bias/scripts/exp03b_library_coverage.py` | Python script text executable, Unicode text, UTF-8 text | 1960 | 984 | Checks library coverage/availability. |
| `01_spectral_lineshape_bias/scripts/exp01_calib_vs_sample_anatomy.py` | Python script text executable, Unicode text, UTF-8 text | 5022 | 2134 | Compares calibration-run and sample-run residual anatomy. |
| `01_spectral_lineshape_bias/scripts/exp04_baseline.py` | Python script text executable, Unicode text, UTF-8 text | 3987 | 1729 | Tests baseline sensitivity. |
| `01_spectral_lineshape_bias/scripts/__pycache__/` | directory | 0 | 0 | Container directory; no payload. |
| `01_spectral_lineshape_bias/scripts/exp06_leverage.py` | Python script text executable, Unicode text, UTF-8 text | 5478 | 2404 | Computes heuristic per-line residual/temperature influence scores. |
| `01_spectral_lineshape_bias/scripts/fig_width_signature.py` | Python script text executable, Unicode text, UTF-8 text | 2127 | 1051 | Generates the width-signature figure. |
| `01_spectral_lineshape_bias/scripts/exp11_grid_synthesis.py` | Python script text executable, Unicode text, UTF-8 text | 4913 | 2008 | Summarizes IC/library result grid. |
| `01_spectral_lineshape_bias/scripts/common.py` | Python script text executable, ASCII text | 5488 | 2305 | Shared archived data loading/reduction; searches only raw/radiography. |
| `01_spectral_lineshape_bias/scripts/fitlib.py` | Python script text executable, Unicode text, UTF-8 text | 5793 | 2022 | Shared fit/reconstruction helpers and 8–45 eV fit-window logic. |
| `01_spectral_lineshape_bias/scripts/diag_residuals.py` | Python script text executable, Unicode text, UTF-8 text | 6160 | 2642 | Builds residual diagnostics and headline residual locations. |
| `01_spectral_lineshape_bias/report/report.tex` | LaTeX 2e document text, ASCII text | 27878 | 10335 | Source for the nine-page prior report. |
| `01_spectral_lineshape_bias/report/report.pdf` | PDF document, version 1.5 | 585805 | 535186 | Rendered nine-page prior report; clean render, no bibliography. |
| `01_spectral_lineshape_bias/data/exp05_calib_consistency.json` | JSON data | 633 | 290 | Cached calibration/consistency metrics. |
| `01_spectral_lineshape_bias/data/exp01_calib_vs_sample.csv` | CSV text | 583 | 311 | Cached calibration-vs-sample line metrics. |
| `01_spectral_lineshape_bias/data/exp03_nuclear_data.json` | JSON data | 6372 | 862 | Cached multi-library UDR fit results. |
| `01_spectral_lineshape_bias/data/exp03_stdout.txt` | ASCII text | 1902 | 655 | Console provenance for nuclear-library comparison. |
| `01_spectral_lineshape_bias/data/grid_synthesis.csv` | CSV text | 743 | 359 | Cached cross-family/library summary grid. |
| `01_spectral_lineshape_bias/data/exp09_endf8.1.json` | JSON data | 1248 | 401 | Cached ENDF/B-VIII.1 IC calibration and hot-fit metrics. |
| `01_spectral_lineshape_bias/data/exp04_baseline.json` | JSON data | 691 | 291 | Cached baseline sensitivity metrics. |
| `01_spectral_lineshape_bias/data/exp03b_library_coverage.json` | JSON data | 1443 | 241 | Cached library-coverage check. |
| `01_spectral_lineshape_bias/data/exp09_tendl2023.json` | JSON data | 1245 | 397 | Cached TENDL-2023 IC calibration and hot-fit metrics. |
| `01_spectral_lineshape_bias/data/exp09_endf8.0.json` | JSON data | 1254 | 397 | Cached ENDF/B-VIII.0 IC calibration and hot-fit metrics. |
| `01_spectral_lineshape_bias/data/exp10_stdout.txt` | ASCII text | 592 | 335 | Console provenance for density-constrained fits. |
| `01_spectral_lineshape_bias/data/exp02_stdout.txt` | ASCII text | 3353 | 1243 | Console provenance for full resolution-family comparison. |
| `01_spectral_lineshape_bias/data/exp09_jeff3.3.json` | JSON data | 1246 | 399 | Cached JEFF-3.3 IC calibration and hot-fit metrics. |
| `01_spectral_lineshape_bias/data/exp06_leverage.json` | JSON data | 1696 | 441 | Cached heuristic line-leverage metrics. |
| `01_spectral_lineshape_bias/data/exp07_feature_14ev.json` | JSON data | 303 | 173 | Cached 14 eV feature metrics. |
| `01_spectral_lineshape_bias/data/decomposition.csv` | CSV text | 820 | 533 | Cached non-orthogonal temperature-sensitivity table; includes misleading TC-free label. |
| `01_spectral_lineshape_bias/data/exp09_cendl3.2.json` | JSON data | 1247 | 402 | Cached CENDL-3.2 IC calibration and hot-fit metrics. |
| `01_spectral_lineshape_bias/data/region_counts.npz` | Zip archive data, at least v4.5 to extract, compression method=store | 215778 | 150699 | Only rerunnable experimental cache: 5,304-bin summed sample/open/calibration counts, energy/TOF axes, charges, temperatures, pixel counts. |
| `01_spectral_lineshape_bias/data/exp02b_stdout.txt` | ASCII text | 2202 | 929 | Console provenance for fast resolution comparison. |
| `01_spectral_lineshape_bias/data/exp09_jendl5.json` | JSON data | 1249 | 399 | Cached JENDL-5 IC calibration and hot-fit metrics. |
| `01_spectral_lineshape_bias/data/exp02_resolution_correction.json` | JSON data | 1850 | 727 | Cached full resolution-family comparison. |
| `01_spectral_lineshape_bias/data/exp02b_fast.json` | JSON data | 1090 | 490 | Cached fast resolution comparison. |
| `01_spectral_lineshape_bias/data/exp10_density_constrained.json` | JSON data | 1729 | 365 | Cached density-constrained fit metrics. |
| `01_spectral_lineshape_bias/data/residual_anatomy.csv` | CSV text | 1032 | 566 | Cached per-line residual anatomy. |
| `01_spectral_lineshape_bias/data/nereids_resolution_internals.md` | Unicode text, UTF-8 text | 4931 | 2697 | Prior static code notes about resolution internals. |
| `01_spectral_lineshape_bias/scripts/__pycache__/fitlib.cpython-314.pyc` | data | 8606 | 4552 | Python bytecode cache; not treated as source or evidence. |
| `01_spectral_lineshape_bias/scripts/__pycache__/common.cpython-314.pyc` | data | 9533 | 4936 | Python bytecode cache; not treated as source or evidence. |
| `notebooks/` | directory | 0 | 0 | Container directory; no payload. |
| `__MACOSX/._notebooks` | AppleDouble encoded Macintosh file | 220 | 93 | macOS archive metadata; no research role. |
| `notebooks/venus_resonance_thermometry_24701_IC_JENDL5.ipynb` | Unicode text, UTF-8 text, with very long lines (55251) | 1448021 | 1078685 | IC/JENDL-5 science notebook; cached outputs inspected, final profile cell unexecuted. |
| `notebooks/venus_resonance_thermometry_24701.ipynb` | Unicode text, UTF-8 text, with very long lines (52845) | 1544375 | 1149747 | UDR/ENDF-B-VIII.1 reference notebook; cached outputs inspected. |
| `notebooks/calib_IC_jendl5_24685.json` | JSON data | 290 | 184 | Minimal cached RT-run position and IC parameters; omits objective, uncertainty, bounds, settings, version, and input hashes. |

## Notebook and input/output audit

| Item | UDR/ENDF-B-VIII.1 notebook | IC/JENDL-5 notebook |
|---|---:|---:|
| Structure | 10 markdown + 10 code; all code executed | 10 markdown + 10 code; 9 code cells executed |
| Whole-region temperature | 992.1 ± 9.2 K | 1058.7 ± 7.0 K |
| Areal density | 6.2920e-4 atoms/barn | 6.3251e-4 atoms/barn |
| Deviance/dof | 50.0 | 28.6 |
| Cached map runtime | 78 + 21 = 99 s | 473 + 129 = 602 s |
| Final strip profile | executed | unexecuted |

Exact external inputs referenced but absent locally and from the ZIP:

- `/SNS/VENUS/IPTS-37432/shared/autoreduce/images/tpx1/raw/radiography`
  plus its sibling `ob/` hierarchy and the per-run TIFF, `*_Spectra.txt`,
  and `summary.json` files;
- `/SNS/VENUS/IPTS-37432/nexus/VENUS_24701.nxs.h5` and
  `VENUS_24685.nxs.h5`;
- `/SNS/VENUS/shared/instrument/resonance/_fts_bl10_0p5meV_1keV_25pts.txt`.

The supplied `region_counts.npz` contains the summed spectra and metadata
needed for a whole-region cached-data replay, but not the raw cubes required to
recompute dead-pixel masks, maps, or profiles. The IC calibration JSON contains
only position, four IC parameters, fixed PSR, library/run labels, and a sanity
shift; it does not preserve convergence, loss, bounds, covariance, restart/grid
settings, code revision, or input hashes.

## Cold-audit verdict

**Partially verified.** Machine-readable cache files consistently show that
IC+JENDL-5 lowers the archived whole-region fit error while leaving large,
structured residuals. They do not independently establish the claimed physical
root cause.

### Verified

- The cached robust grid reports FTS/ENDF8.1 at 992.1 K and deviance/dof 50.0,
  IC/ENDF8.1 at 1049.4 K and 44.8, FTS/JENDL-5 at 1017.1 K and 37.4, and
  IC/JENDL-5 at 1060.5 K and 29.7.
- The notebook cache reports IC/JENDL-5 at 1058.7 ± 7.0 K and 28.6.
- Largest archived standardized residuals include +69.8σ near 39.15 eV,
  −47.9σ near 13.92 eV, and −25.0σ near 23.95 eV.
- Six IC calibrations took 665–1490 s. Cached IC mapping took 602 s versus 99 s
  for the UDR notebook (6.08×), although the IC notebook also keeps three
  5304×512×512 stacks resident.
- The IC parameters depend strongly on the chosen nuclear library:
  `a0` spans 0.723–1.062, `a1` 0.00203–0.18406, `beta`
  0.233–0.500, and `R` 0.151–0.267. The separate notebook cache has
  `a1=0.001000175`, effectively its 0.001 hard lower bound.
- Both cached map images retain a sharp detector-panel transition at column
  256. This is visual evidence only because the arrays were not supplied.

### Not verified

- The raw-data fits, raw-to-`region_counts.npz` reduction, maps, strip
  profiles, exact UDR reconstruction, and IC recalibration cannot be rerun from
  the ZIP alone.
- The VENUS UDR's provenance, centering convention, and in-window fidelity are
  unknown because the file is missing.
- No measured or transport-simulated VENUS response is supplied to distinguish
  moderator pulse physics from detector/path/binning/background effects.

### Wrong or overclaimed in the prior package

- “The FTS/UDR file is the root cause” is not mechanism-validated. Fit
  improvement establishes a useful hypothesis, not uniqueness.
- The reported “temperature-bias decomposition” is a sensitivity table, not an
  additive causal decomposition. IC's temperature shift varies by library
  from −17.3 K to +57.3 K.
- The calibration is sequential, not joint: UDR fits position, IC fits shape
  with position frozen, then a position sanity fit is discarded.
- “14 eV is not a contaminant” is unsupported: the script never models the
  promised W/Ta-180m/Fe/Cr/Ni/Cu candidates.
- The density-ratio “physics consistency” target is self-inconsistent because
  RT temperature is fixed and hot temperature is free; the later report admits
  this, while older hypothesis/data labels do not.
- IC room-temperature recovery is partly self-referential because that same RT
  spectrum calibrated IC.
- `anatomy.py` calls an RMS fraction “residual power” and a
  `1-RMS_after/RMS_before` metric “variance explained”; `exp06` calls
  squared transmission residual shares “Poisson deviance shares.” These are
  heuristic diagnostics with incorrect labels.
- A library-dependent sign flip does not exclude a contaminant coexisting with
  library-dependent Ta parameters.
- The claim that saturation/self-shielding is falsified does not test multiple
  scattering, in-scattering, finite geometry, or detector effects.
- JENDL-5 is best among the archived fits, but fit quality alone is not proof
  that its resonance parameters are physically correct.

No data were excluded, masked, smoothed, reweighted, or given a narrowed fit
window during this audit.
