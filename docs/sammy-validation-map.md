# SAMMY validation map for NEREIDS

Date: 2026-02-05
Scope: Inventory of SAMMY test cases and reference outputs relevant to validating NEREIDS physics, organized by module and porting priority.

Companion to: `sammy-teacher-map.md` (source code reference) and `adr/0002-sammy-reference-validation.md` (decision record).

All paths below are relative to the local SAMMY repo at `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY`.

---

## 1) SAMMY test infrastructure overview

SAMMY ships three collections of worked examples plus a C++ regression test framework.

| Location | Description | Count |
|----------|-------------|-------|
| `samexm/samexm/ex000–ex027` | Classic tutorial examples. Each has an `answers/` directory with reference `.par`, `.lst`, `.lpt`, and `.plt` files. | 28 |
| `samexm_new/ex000_new–ex029_new` | Modernized examples with Python driver scripts and `functions.py` helper. | 24 |
| `interactive-examples/` | Jupyter notebook workflows demonstrating end-to-end analysis. | 7 |
| `sammy/TestRunner/` | C++ test runner with plugin-based comparator framework. | — |
| `sammy/samtry/` | Automated regression test cases (CTest-integrated). | 194 |

### 1.1 Test runner comparator architecture

The C++ test runner (`sammy/TestRunner/test_example.cpp`) uses a comparator plugin hierarchy:

```
Comparator (base: file-size check, double parsing)
├── DefaultComparator       — binary file-size comparison
├── CompareSammyPARfiles    — resonance parameter files (5 x 11-char fields)
├── CompareSammyLSTfiles    — columnar cross-section data (configurable column width)
│   ├── CompareSammyLPTfiles — log/report files (chi-squared extraction)
│   ├── CompareLLLfiles      — LLL-format columns
│   ├── CompareMCData        — Monte Carlo data (15-char columns)
│   └── CompareGroupAvgCov   — covariance data (16-char columns)
└── (extensible for new formats)
```

Each test case contains:
- `ctest.inp` — execution steps (run SAMMY, copy files between stages)
- `*.cfg` — comparison rules: `<COMPARATOR> <reference_file> <results_file> [tolerance] [options]`
- `answers/` — reference outputs

Default numeric tolerance: **1.0e-4 relative error**. Configurable per comparator and per test.

### 1.2 Reference output file formats

| Format | Extension | Content | Field width |
|--------|-----------|---------|-------------|
| PAR | `.par` | Resonance energies and widths | 11 chars x 5 columns |
| LST | `.lst` | Tabular cross-section / transmission data | 20 chars (default) |
| LPT | `.lpt` | Fit report with chi-squared values | Free-form text, pattern-matched |
| PLT | `.plt` | Plot data (binary, converted to ASCII via `samplt`) | 16 chars after conversion |

### 1.3 Python test helpers (samexm_new)

`samexm_new/functions.py` provides:
- SAMMY binary location management
- Standard file movement and cleanup
- Output organization in `results/` subdirectories

---

## 2) Test cases relevant to resonance imaging

Organized by physics module in the same order as `sammy-teacher-map.md` Section 12 (suggested porting order).

### 2.1 R-matrix (Reich-Moore) cross sections

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `samexm/ex003` | Synthetic | All cross-section types: capture, elastic, fission, transmission, total | `answers/ex003a,c,e,f,x,t` |
| 1 | `samexm_new/ex003_new` | Synthetic | Same scope, modernized driver | `.lst` files |
| 2 | `samexm/ex001` | Synthetic (10xy) | Single resonance capture (simplest case) | `answers/ex001a-c` |
| 2 | `samexm/ex002` | Synthetic (fissile) | Multiple resonances, wide energy range | `answers/ex002a-c` |
| 3 | `samexm/ex004` | Synthetic | Higher angular momentum (el > 0), spin effects | `answers/ex004a-f` |
| 3 | `samexm_new/ex029_new` | O-16 / C-13 | Explicit Reich-Moore formalism with per-channel normalization | `.lst`, `.lpt` |

### 2.2 Transmission transform (Beer-Lambert)

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `samexm/ex003` | Synthetic | Transmission vs total cross-section conversion | `answers/ex003t` |
| 1 | `samexm/ex016` | Fe-56 | Three sample thicknesses, blacking-out effects | `answers/ex016a-e` |
| 1 | `samexm_new/ex016_new` | Fe-56 | Sequential transmission with covariance propagation | `.lst`, `.cov` |
| 2 | `samexm/ex017` | Fe-56 | Variable thickness and temperature per dataset | `answers/ex017a-e` |

### 2.3 Normalization and background

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `samexm/ex009` | Fe-56 | Normalization factor fitting | `answers/ex009a-b` |
| 1 | `samexm/ex010` | Fe-56 | Constant and energy-dependent background models | `answers/ex010a-d` |
| 1 | `samexm_new/ex009_new` | Fe-56 | Normalization (modernized) | `.lst` |
| 1 | `samexm_new/ex010_new` | Fe-56 | Background functions (modernized) | `.lst` |

### 2.4 Doppler broadening

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `samexm/ex005` | Synthetic | No broadening, 50 K, 300 K; temperature as fit parameter | `answers/ex005a-e` |
| 1 | `samexm_new/ex005_new` | Synthetic | Doppler broadening without FGM | `.lst` |

### 2.5 Resolution broadening

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `samexm/ex006` | Synthetic | Gaussian resolution (DELTAL, DELTAG, DELTAB) | `answers/ex006a-f` |
| 1 | `samexm/ex007` | Ni-58 | ORR resolution function (burst, water/Li detectors) | `answers/ex007a-b` |
| 1 | `samexm/ex008` | W-183 | RPI resolution function + multi-nuclide fitting | `answers/ex008a-b` |
| 1 | `samexm_new/ex006_new` | Pu-239 | Gaussian resolution on fission | `.lst`, `.lpt` |
| 1 | `samexm_new/ex007_new` | Ni-58 | ORR resolution (4 detector variants: tl, tn, wl, wn) | `.lst`, `.lpt`, `.cov` |
| 1 | `samexm_new/ex008_new` | W-183 | RPI resolution on transmission | `.lst`, `.lpt` |

### 2.6 Self-shielding and cylindrical corrections

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `samexm/ex019` | Ba-136 | 6 variants: no correction, self-shielding only, single/double scattering, infinite slab vs finite geometry | `answers/ex019a-f` |
| 1 | `samexm_new/ex019_new` | Ba (multi) | 5 progressive correction levels (a through e) | `.lst` per variant |

### 2.7 Multi-isotope composition

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `samexm/ex012` | nat. Si (28/29/30) | Multi-isotope transmission, abundance fitting, 22 spin groups | `answers/ex012a-b` |
| 1 | `samexm_new/ex012_new` | nat. Si | Same scope, modernized | `.lst` |
| 1 | `interactive-examples/multi-isotope` | nat. Zr (90/91/92/94/96) | Adjustable abundances, ENDF-driven, Jupyter workflow | Run-time (no static reference) |
| 2 | `samexm/ex021` | W (182/183/184/186) | RPI resolution + 4 tungsten isotopes | `answers/ex021b-h` |
| 2 | `samexm/ex022` | U-235-like | Complete multi-data evaluation (transmission + fission + capture) | `answers/ex022a-k` |

### 2.8 ENDF ingestion

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 1 | `interactive-examples/start-from-endf` | Pu-239 | ENDF-to-SAMMY initialization, cross-section from library | `pu9.png` |
| 2 | `samexm/ex026` | Ni-58 | Transmission + capture yield + ENDF File 2 output | `answers/ex026a-c` |
| 2 | `samexm/ex027` | Fe-56 | ENDF File 2 as input to SAMMY | `answers/ex027a-b` |
| 2 | `samexm_new/ex026_new` | Ni-58 | Three-step pipeline: fit, capture, ENDF generation | `.ndf`, `.endf` |

### 2.9 Uncertainty and covariance

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 2 | `samexm/ex013` | Synthetic | Bayesian priors, default vs explicit uncertainty | `answers/ex013a,cc` |
| 2 | `samexm/ex018` | Am-241 | Data covariance matrices (diagonal, off-diagonal, implicit) | `answers/ex018a-f` |
| 2 | `samexm_new/ex013_new` | Ni-58 | Prior uncertainty propagation in transmission | `.lst`, `.lpt` |
| 2 | `interactive-examples/calc-theor-uncertainty` | Ta-181 | Transmission with uncertainty bands from covariance | `trans-unc.png`, `trans-cov.png` |

### 2.10 Unresolved resonance region

| Priority | Example | Isotope | What it validates | Reference outputs |
|----------|---------|---------|-------------------|-------------------|
| 3 | `interactive-examples/generate-file33` | Ta-181 | URR fitting, ENDF Files 2/3/32/33 generation | Run-time |

---

## 3) Suggested validation porting order

Aligned with `sammy-teacher-map.md` Section 12 and the NEREIDS roadmap (Phase 1 first).

### Phase 1 — Forward model MVP

| Step | Physics module | Primary test case | Validation target |
|------|---------------|-------------------|-------------------|
| 1 | 0 K Reich-Moore cross sections | `ex003` | `.lst`: all cross-section channels match SAMMY within 1e-4 |
| 2 | Transmission transform | `ex003` (transmission output) + `ex016` | `.lst`: transmission vs thickness; blacking-out behavior |
| 3 | Normalization and background | `ex009` + `ex010` | `.lst`: fitted normalization factor; background subtraction |
| 4 | Doppler broadening | `ex005` | `.lst`: broadened cross sections at 50 K, 300 K |
| 5 | Resolution broadening | `ex006` (Gaussian) then `ex007` (ORR) | `.lst`: broadened line shapes |
| 6 | Self-indication corrections | `ex019` (all variants) | `.lst`: progressive correction levels match |
| 7 | Multi-isotope transmission | `ex012` (nat. Si) | `.lst` + `.par`: fitted abundances and residuals |

### Phase 2 — Inference and diagnostics

| Step | Physics module | Primary test case | Validation target |
|------|---------------|-------------------|-------------------|
| 8 | Multi-isotope abundance fitting | `interactive-examples/multi-isotope` (nat. Zr) | Fitted abundances within declared uncertainty |
| 9 | ENDF-initialized workflow | `interactive-examples/start-from-endf` | Cross sections match ENDF library values |
| 10 | Uncertainty propagation | `ex013` + `ex018` | Posterior covariance structure matches SAMMY |
| 11 | Unresolved resonance | `interactive-examples/generate-file33` | URR cross sections within tolerance |

---

## 4) Reference data extraction plan

For each ported test case, NEREIDS needs:

1. **Input data** — `.inp`, `.par`, `.dat` files defining the problem.
2. **Reference output** — `.lst` (cross-section/transmission tables), `.par` (fitted parameters), `.lpt` (chi-squared values).
3. **Comparison metadata** — field widths, skip lines, tolerance, cutoff (from SAMMY `.cfg` files).

Proposed directory structure in NEREIDS:

```
tests/
  fixtures/
    sammy_reference/
      ex003/
        input/          # .inp, .par, .dat copied from SAMMY
        expected/       # .lst, .par, .lpt copied from answers/
        config.toml     # comparison rules: tolerance, column widths, skip lines
      ex005/
        ...
```

Each `config.toml` captures the comparison semantics from the corresponding SAMMY `.cfg` file so that Rust and Python test code can replicate the same validation logic.

---

## 5) Comparator implementation in NEREIDS

NEREIDS will implement comparators in Rust (for `nereids-core` unit/integration tests) and expose them via Python bindings (for notebook-based validation).

Key design points extracted from SAMMY's C++ comparators:

- **PAR files**: 5 fields x 11 characters (energy, gamma width, neutron width, fission1, fission2). Compare each field with relative tolerance.
- **LST files**: configurable column width (default 20). Skip N header lines. Ignore values below a cutoff threshold. Relative tolerance on remaining values.
- **LPT files**: pattern-match lines containing both "RED" and "DIV" to extract chi-squared values. Compare extracted vectors element-wise.
- **Tolerance semantics**: `|a - b| / max(|a|, |b|)` with graceful zero handling. Default 1e-4.

---

## 6) Open questions

1. **samtry/ coverage**: The 194 regression tests in `sammy/samtry/` likely overlap with the tutorial examples but may contain additional edge cases. A systematic cross-reference would identify unique tests worth porting.
2. **Binary `.plt` files**: These require the `samplt` converter. Decide whether to support this format or only validate via `.lst` and `.par`.
3. **HDF5 outputs**: Modern SAMMY generates `.h5` covariance files. NEREIDS should validate against these once `nereids-io` supports HDF5.
4. **Run-time vs static references**: `interactive-examples/` and some `samexm_new/` cases have no pre-computed reference outputs. We need to generate and freeze reference outputs by running SAMMY once and committing the results.

---

End of document.
