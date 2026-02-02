# NEREIDS

NEREIDS (NEutron REsonance Imaging Diagnostic Suite) is an academic-focused platform for neutron resonance imaging. It provides a Rust physics core, Python bindings, and NeXus/HDF5-compliant data I/O, with a path to fast standalone GUI workflows for VENUS/MARS and broader beamline use.

Status: pre-alpha research repository. No public API guarantees yet.

## Initial scope
- Resonance imaging forward model for transmission and related corrections.
- User-defined resolution functions.
- Self-indication and cylindrical sample transmission corrections.
- Unresolved resonance handling and isotope discovery workflows.
- NeXus/HDF5 I/O and Python bindings.

## Repository layout
- `crates/` Rust libraries (core physics, I/O, models).
- `bindings/python/` Python bindings and packaging.
- `apps/` CLI and GUI applications.
- `docs/` Technical documentation and design notes.
- `examples/` Reference workflows and datasets.
- `scripts/` Developer tooling and utilities.

## References
See `docs/sammy-teacher-map.md` for the SAMMY reference map used to guide the teacher-student rewrite.
