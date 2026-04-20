# ENDF test fixtures

Public-domain ENDF-B-VIII.1 nuclear data files, committed for
regression tests that depend on real-world resonance structure.

## Files

| File          | Formalism | Source                                  | Why committed |
|---------------|-----------|-----------------------------------------|---------------|
| `Hf-177.endf` | MLBW (LRF=2) | NNDC ENDF-B-VIII.1 (2023), NIST PML | Pins issue #465 — the batch Reich-Moore dispatcher silently treated MLBW ranges as SLBW, causing up to 55 % cross-section error on natural-Hf isotopes. The regression test reads this file directly; do not replace with a synthetic substitute without a companion synthetic MLBW test. |

## License

ENDF-B-VIII.1 is distributed by the National Nuclear Data Center (NNDC)
at Brookhaven National Laboratory under a public-domain policy (US
Government work, no copyright restriction; see
[https://www.nndc.bnl.gov](https://www.nndc.bnl.gov)). Redistribution for
testing purposes is permitted.

## Sizing

Fixtures here are intentionally small (one isotope per formalism). If
you need more comprehensive ENDF coverage during development, use
`pixi run python -c "import nereids; nereids.load_endf(z, a)"` which
auto-fetches to `~/Library/Caches/nereids/endf/…`.
