# ENDF test fixtures

Public-domain ENDF/B-VIII.0 / VIII.1 nuclear data files, committed for
regression tests that depend on real-world resonance structure.

## Files

| File          | Formalism | Source                                  | Why committed |
|---------------|-----------|-----------------------------------------|---------------|
| `Hf-177.endf` | MLBW (LRF=2) | NNDC ENDF/B-VIII.1 (2023), NIST PML | Pins issue #465 — the batch Reich-Moore dispatcher silently treated MLBW ranges as SLBW, causing up to 55 % cross-section error on natural-Hf isotopes. The regression test reads this file directly; do not replace with a synthetic substitute without a companion synthetic MLBW test. |
| `Ta-181.endf` | MLBW (LRF=2) + URR (LRU=2) | NNDC ENDF/B-VIII.0 (73-Ta-181 LLNL EVAL-Jan11), MAT 7328 | Pins issue #638 — confirms the VIII.0 evaluation's resolved region is genuinely sparse (NER=2: MLBW RRR to 330 eV with **76** discrete resonances + an unresolved URR to 5 keV), so `total_resonance_count()==76` is faithful, **not** a dropped range. VIII.1 later extended the RRR to 2554 eV (565 resonances). `test_parse_ta181_endf8_0_resonance_count` in `nereids-endf/src/parser.rs` reads this file directly. |
| `Ta-181-file32.endf` | Reich-Moore (LRF=3) File 2 + File 32 (LCOMP=1) | NNDC ENDF/B-VIII.1 Ta-181, MAT 7328, cut to the resonances at 10.35, 20.31, 22.74 and 23.95 eV with the counts in its CONT and LIST records set to match; every other number is the tape's own field | `parse_resonance_covariance` tests in `nereids-endf/src/parser.rs`, and the several-resonance covariance of `covariance_against_information` in `nereids-pipeline/tests/counts_fit.rs`. |

## License

ENDF/B-VIII.0 and VIII.1 are distributed by the National Nuclear Data
Center (NNDC) at Brookhaven National Laboratory under a public-domain
policy (US Government work, no copyright restriction; see
[https://www.nndc.bnl.gov](https://www.nndc.bnl.gov)). Redistribution for
testing purposes is permitted.

## Sizing

Fixtures here are intentionally small (one isotope per formalism). If
you need more comprehensive ENDF coverage during development, use
`pixi run python -c "import nereids; nereids.load_endf(z, a)"` which
auto-fetches to `~/Library/Caches/nereids/endf/…`.

## `mf2/`: File 2 extracts

Each file holds the first record of MF=1/MT=451 and the full MF=2 of one
ENDF/B-VIII.1 evaluation (NNDC, fetched 2026-10-20), with no other records
and LF line ends.
File 3 is absent, so these files give File 2 cross sections only.

| Files | Formalism | Used by |
|---|---|---|
| `W-182`, `W-183`, `W-184`, `W-186`, `Fe-54`, `Fe-57`, `Si-28`, `Si-29`, `Si-30`, `Rh-103`, `Cu-63`, `Cu-65`, `V-51`, `Sr-88` | R-Matrix Limited (LRF=7, KRM=3) | LRF=7 parser tests in `nereids-endf/src/parser.rs`; LRF=7 cross-section tests |
| `Fe-56` | Reich-Moore (LRF=3) | LRF=3 against LRF=7 consistency test |

Source MD5 sums of the full files: W-182 `f2e0f35558086efbfa607938f8887ddd`,
Sr-88 `d770aaf49c6e9856c9d0539dc4da966c`.
