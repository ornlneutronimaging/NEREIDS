# Python bindings

This crate provides the `nereids` Python extension module via pyo3.

## Build (developer)
- Use the workspace `pyproject.toml` from the repo root, or
- From this folder, run:
  - `maturin develop` (debug)
  - `maturin develop --release --strip` (release)

The module currently exposes a `version()` function and `__version__` as a stub.
