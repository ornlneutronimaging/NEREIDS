# ADR 0002: Validate NEREIDS physics against SAMMY reference outputs

Date: 2026-02-05
Status: Proposed

## Context

NEREIDS reimplements neutron resonance physics previously available only in the SAMMY Fortran/C++ codebase. Correctness of the physics is non-negotiable: an incorrect cross-section or transmission calculation would silently corrupt every downstream imaging result.

SAMMY ships hundreds of worked examples with reference outputs (`.lst`, `.par`, `.lpt` files) that have been validated over decades of use at ORNL and other facilities. These outputs cover every physics module NEREIDS needs: Reich-Moore cross sections, Doppler broadening, resolution functions, transmission transforms, self-shielding, multi-isotope fitting, and ENDF I/O.

We need a strategy that:
- Catches physics regressions early.
- Lets us validate module-by-module as we port.
- Runs in CI without requiring a SAMMY installation.

## Decision

1. **Extract SAMMY reference data into NEREIDS test fixtures.** For each ported physics module, copy the relevant SAMMY input and reference output files into `tests/fixtures/sammy_reference/<test_id>/`. Commit these as static fixtures so CI does not depend on a SAMMY checkout.

2. **Implement comparators in Rust** that replicate SAMMY's comparison semantics:
   - Fixed-width field parsing (11, 15, 16, 20 character columns).
   - Relative tolerance with configurable cutoff (default 1e-4).
   - Format-specific logic for PAR, LST, and LPT files.
   - Each fixture directory includes a `config.toml` specifying format, tolerance, column width, and skip-line rules.

3. **Expose comparators via Python bindings** so that notebook-based validation and `pytest` integration tests can use the same logic.

4. **Port test cases in porting order.** Follow the sequence in `sammy-validation-map.md` Section 3: start with 0 K cross sections (`ex003`), then transmission (`ex016`), normalization/background (`ex009`/`ex010`), broadening (`ex005`/`ex006`), self-shielding (`ex019`), and multi-isotope (`ex012`).

5. **Generate and freeze reference outputs for interactive examples.** Cases like `interactive-examples/multi-isotope` have no static reference files. Run SAMMY once, commit the outputs, and treat them as frozen references.

## Consequences

- **Pro**: Physics correctness is tested against a mature, independently validated codebase rather than hand-derived expectations.
- **Pro**: Module-by-module validation aligns with the incremental porting strategy in the roadmap.
- **Pro**: CI runs without SAMMY by using committed reference fixtures.
- **Con**: Reference files add to repository size (mitigated by selecting only high-priority test cases).
- **Con**: If SAMMY itself has a bug in a reference output, NEREIDS would replicate it. Mitigated by cross-checking against ENDF-processed values and independent tools where possible.
- **Con**: Fixed-width parsing is brittle. Mitigated by testing the comparators themselves against known inputs.
