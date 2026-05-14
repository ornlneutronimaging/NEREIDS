# Notebook Status

The tutorial notebooks live under `examples/notebooks/`. They are intended as
user-facing examples, not as the primary API contract. The API contract is the
Python type stubs, Python tests, Rust tests, mdBook guide, and Rustdoc.

## Verification Status

| Notebook group | Current status | Smoke-tested in CI | External requirements |
|----------------|----------------|--------------------|-----------------------|
| `foundations/` | Current examples for cross-sections, broadening, URR, and transmission physics. | No. Covered indirectly by Rust/Python physics tests. | ENDF downloads on first run for notebooks that call `load_endf(...)`. |
| `building_blocks/` | Current examples for ENDF loading, fitting, grouped isotopes, custom resolution, and TIFF I/O. | No. Core APIs are covered by `tests/test_nereids.py`. | ENDF downloads on first run; TIFF notebook may require local generated files. |
| `workflows/` | Current end-to-end synthetic workflows. | No. | ENDF downloads on first run. |
| `applications/` | Reference-data workflow. | No. | Requires external PLEIADES/Git LFS data, not bundled in normal PyPI installs. |

As of the docs workflow in this repository, notebooks are not executed by
`pixi run doc-guide`, `pixi run doc-build`, or GitHub Pages publishing.
Before using a notebook as release evidence, run it manually or add a
notebook execution job with controlled data and network policy.

## First-Run Network Behavior

Notebooks that call `nereids.load_endf(...)` may need network access the
first time they fetch an isotope/library combination. ENDF files are cached
locally after retrieval. To avoid network access, use local fixtures and
`nereids.load_endf_file(...)` where practical.

## Reference Data

The application notebook uses larger reference data from the PLEIADES test
data repository via `tests/data/pleiades_data/`. That data is a submodule and
uses Git LFS. It is not guaranteed to be available in a fresh source checkout
unless submodules and LFS objects have been initialized.

```bash
git submodule init
git submodule update
cd tests/data/pleiades_data
git lfs pull
```

## Release Expectation

For a release, document one of these states in the release notes:

- notebooks were smoke-run locally with the exact package version,
- notebooks were not run and remain tutorial examples only, or
- a subset was run, with any skipped notebooks and data/network reasons
  listed explicitly.
