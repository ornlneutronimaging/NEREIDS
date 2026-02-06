# ADR 0003: Crate architecture and workspace layout

Date: 2026-02-05
Status: Accepted

## Context

NEREIDS reimplements neutron resonance imaging physics in Rust, designed for the VENUS beamline at SNS/ORNL. The system must support:

- A multi-stage forward model: R-matrix cross sections, Doppler broadening, resolution convolution, Beer-Lambert transmission, normalization, and self-shielding corrections.
- Per-pixel parallel fitting across 264K+ pixels (514x514 detector), each with ~1500 energy bins and 1-6 isotopes.
- Multiple delivery surfaces: Rust library, Python bindings (PyO3), CLI, and GUI.
- ENDF resonance parameter retrieval and caching.
- NeXus/HDF5 and legacy TIFF I/O.

The initial scaffolding had three library crates (`nereids-core`, `nereids-physics`, `nereids-io`) plus Python bindings, all containing stubs. This ADR defines the full workspace layout.

## Decision

### Workspace layout (8 members)

```
crates/
  nereids-core/        Types, traits, error hierarchy. Minimal deps (thiserror only).
  nereids-physics/     Forward model: R-matrix, broadening, transmission, corrections.
  nereids-fit/         Optimizer trait + Bayes/GLS implementation. Pixel-dispatch parallelism.
  nereids-endf/        ENDF resonance parameter parsing, retrieval, caching.
  nereids-io/          NeXus/HDF5, TIFF, spectra, resolution file I/O.
apps/
  nereids-cli/         CLI binary.
  nereids-gui/         egui-based GUI application.
bindings/
  python/              PyO3 cdylib.
```

### Dependency graph

```
nereids-core  (thiserror only)
  ├── nereids-physics  (core, num-complex)
  │     └── nereids-fit  (core, physics, rayon, nalgebra)
  ├── nereids-endf  (core, serde, reqwest)
  └── nereids-io  (core, hdf5-metno[optional,static,zlib], tiff[optional], ndarray)

nereids-cli → all library crates + clap
nereids-gui → all library crates + eframe, egui, egui_plot, rfd
python/     → all library crates + pyo3, numpy
```

**Key constraint**: `nereids-physics` has NO dependency on I/O, fitting, or ENDF. It is pure computation.

### HDF5 crate choice

Following the rustpix pattern, use `hdf5-metno` (MET Norway's fork) aliased as `hdf5` at workspace level with `static` (no runtime system dependency) and `zlib` (compression) features. Feature-gated in `nereids-io`.

### Core abstractions

- `ForwardModel` trait: central abstraction replacing `TransmissionModel`. Computes predicted transmission spectrum from energy grid + R-matrix parameters + configuration. Supports Jacobian computation for gradient-based fitting.
- `ResolutionFunction` trait: pluggable instrument response (Gaussian for testing, tabulated for VENUS MC data).
- `BackgroundModel` trait: constant, polynomial, or energy-dependent background.
- `Optimizer` trait: pluggable fitting algorithm. Default implementation is Bayes/GLS from SAMMY.

### GUI distribution

The GUI application is distributed via PyPI (Python-installable binary), GitHub Releases (standalone binaries), and Homebrew (macOS tap). Development follows a prototype-first UX approach.

## Consequences

- **Pro**: Clean separation of concerns. Physics crate has zero I/O dependencies, enabling focused testing and benchmarking.
- **Pro**: Feature-gated I/O means the core library compiles without HDF5/TIFF when not needed.
- **Pro**: Workspace dependencies ensure consistent versions across all crates.
- **Pro**: `apps/` directory separates binary targets from library crates.
- **Con**: More crates means more `Cargo.toml` files to maintain. Mitigated by workspace dependency inheritance.
- **Con**: New crates (`nereids-fit`, `nereids-endf`) start as stubs and require implementation effort. This is intentional — the scaffolding establishes the architecture before physics implementation begins.
