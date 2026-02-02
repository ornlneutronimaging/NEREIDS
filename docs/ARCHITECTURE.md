# Architecture

## High-level components
- Physics core (Rust): resonance models, broadening, resolution, and corrections.
- Data I/O (Rust): NeXus/HDF5 compliant read/write utilities.
- Inference layer: parameter estimation, uncertainty, and diagnostics.
- Python bindings: user-facing API and notebooks.
- Apps: CLI utilities and a standalone GUI.

## Design notes
- Keep physics kernels pure and testable.
- Prefer explicit data models over implicit global state.
- Keep the inference layer replaceable and backend-agnostic.
