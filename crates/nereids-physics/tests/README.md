# Integration tests for `nereids-physics`

## Fixture-gated tests

Some tests require external instrument-characterization data that
ORNL has not approved for public release. The fixture file is:

- `_fts_bl10_0p5meV_1keV_25pts.txt` — VENUS instrument tabulated
  resolution kernel (SAMMY USR format).  Place at the workspace root.

These tests use the early-return idiom rather than `#[ignore]`:

```rust
let Some(path) = common::venus_usr_resolution_path() else { return; };
```

When the fixture is present, the test exercises real-instrument
coverage.  When absent (CI, fresh checkouts), the test is a no-op
and `cargo test` reports it as passed — no "ignored" noise.

## Why not `#[ignore]`?

Historical reason: previous test scaffolding used
`#[ignore = "requires PLEIADES resolution file ..."]`.  The label
"PLEIADES" was a misnomer (PLEIADES is a Python wrapper around
SAMMY; this file is a SAMMY-format kernel for the VENUS
instrument).  The `#[ignore]` mechanism also conflated
fixture-gating with broken-on-purpose tests, and didn't enforce
that fixture-present runs actually executed.  See issue #497.
