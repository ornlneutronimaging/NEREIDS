# Synthetic ENDF Fixtures

Hand-crafted minimal MF=2 / MT=151 ENDF/B-VI fixtures, vendored from
inline string literals previously embedded in `concat!(...)` blocks
inside `crates/nereids-endf/src/parser.rs`. Each fixture covers a
single parser code path (LRF discriminator, edge case, validation
boundary) and is intentionally as small as possible while still being
a valid ENDF tape that `parse_endf_file2` will accept (or reject with
a specific error).

## Naming convention

Each filename matches the host `#[test] fn ...` it serves, after
stripping the `parses_` / `test_parse_` / `test_` prefix:

| Filename pattern        | Test fn pattern                 |
|-------------------------|---------------------------------|
| `lrf<N>_<descriptor>.endf` | `test_parse_lrf<N>_<descriptor>` |
| `<scope>_rejected.endf`    | `test_<scope>_rejected`          |

Each `.endf` file is loaded by exactly one test via
`include_str!("../../../tests/data/synthetic/<name>.endf")`.

## Format notes

- 80-char ASCII lines + `\n` terminator (no CRLF), one record per line.
- Tape-edit blocks (TPID/MEND/etc.) are omitted unless the test needs
  them; only the MF=2 MT=151 resonance-section bytes are present.
- These files are **synthetic** — do **not** use them as physics
  references. Real-world ENDF tapes live under `../endf/`.

## Maintenance

If a test in `parser.rs` is renamed, rename its `.endf` file to match
(this directory is grep-discoverable from the test fn name; keeping
them in sync keeps the convention working).

If a new minimal fixture is needed, add it here rather than inlining
a `concat!(...)` block — that pattern was removed in Wave 2 PR-3 of
the architecture refactor sprint (audit-r4-architecture-v2 finding F1).
