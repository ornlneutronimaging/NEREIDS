# ENDF test fixtures

Public-domain ENDF/B-VIII.0 / VIII.1 nuclear data files, committed for
regression tests that depend on real-world resonance structure.

## Files

| File          | Formalism | Source                                  | Why committed |
|---------------|-----------|-----------------------------------------|---------------|
| `Hf-177.endf` | MLBW (LRF=2) | NNDC ENDF/B-VIII.1 (2023), NIST PML | Pins issue #465 — the batch Reich-Moore dispatcher silently treated MLBW ranges as SLBW, causing up to 55 % cross-section error on natural-Hf isotopes. The regression test reads this file directly; do not replace with a synthetic substitute without a companion synthetic MLBW test. |
| `Ta-181.endf` | MLBW (LRF=2) + URR (LRU=2) | NNDC ENDF/B-VIII.0 (73-Ta-181 LLNL EVAL-Jan11), MAT 7328 | Pins issue #638 — confirms the VIII.0 evaluation's resolved region is genuinely sparse (NER=2: MLBW RRR to 330 eV with **76** discrete resonances + an unresolved URR to 5 keV), so `total_resonance_count()==76` is faithful, **not** a dropped range. VIII.1 later extended the RRR to 2554 eV (565 resonances). `test_parse_ta181_endf8_0_resonance_count` in `nereids-endf/src/parser.rs` reads this file directly. |

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
