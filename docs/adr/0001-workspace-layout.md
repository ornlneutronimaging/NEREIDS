# ADR 0001: Rust Workspace and Crate Layout

## Status

Accepted

## Context

NEREIDS is a Rust-based neutron resonance imaging library for the VENUS beamline
at SNS, ORNL. It replaces the previous approach of wrapping the legacy Fortran
SAMMY code via PLEIADES (Python/LANL). We need an architecture that:

- Separates concerns cleanly (physics, I/O, fitting, orchestration)
- Allows independent testing and validation of each physics module
- Supports three delivery targets: Rust library, PyO3 Python bindings, egui GUI
- Is maintainable by AI agents with clear module boundaries and documentation

A previous attempt (Codex) failed due to lack of structure, ad-hoc approximations,
temp file litter, and monolithic commits.

## Decision

We adopt a Cargo workspace with six core library crates, one Python binding crate,
and one GUI application crate:

```
nereids-core       Core types, physical constants, error types, traits
nereids-endf       ENDF file retrieval, caching, resonance parameter parsing
nereids-physics    Cross-section (Reich-Moore), broadening, transmission model
nereids-io         TIFF/NeXus data loading, VENUS normalization
nereids-fitting    Optimization (LM least-squares, Poisson/BFGS)
nereids-pipeline   End-to-end orchestration, spatial mapping, sparsity handling
nereids-python     PyO3 thin wrapper (bindings/python)
nereids-gui        egui desktop application (apps/gui)
```

Dependency flow is strictly acyclic: core ← endf ← physics, core ← io,
physics ← fitting, all ← pipeline ← bindings/apps.

## Physics Reference

All physics implementations reference SAMMY (ORNL/TM-9179) with:
- Exact formalism (no approximations)
- In-code comments citing SAMMY manual sections and source files
- Validation against SAMMY's own test cases at each phase

## Consequences

- Each crate can be built and tested independently
- Physics validation is traceable to SAMMY reference
- Python users get a thin, performant wrapper via PyO3
- GUI can be developed independently of the core library
- Worklog directory (gitignored) prevents temp file pollution
- Atomic commits at each phase enable early detection of issues
