# Architecture

NEREIDS is organized as a Rust workspace with seven library crates, a GUI
application, and Python bindings.

## Crate Dependency Graph

```text
                    endf-mat (standalone lookup tables)
                        |
                   nereids-core (types, constants)
                   /    |    \
          nereids-endf  |  nereids-io
            (ENDF)      |    (TIFF, NeXus)
               \        |
           nereids-physics
            (cross-sections)
                 \
              nereids-fitting
               (LM, Poisson)
                      |
              nereids-pipeline ── nereids-io
              (orchestration)
                /         \
       nereids-python    nereids-gui
        (PyO3 bindings)  (egui desktop)
```

## Crate Overview

| Crate | Purpose |
|-------|---------|
| [`endf-mat`](api/endf_mat/) | Zero-dependency lookup tables: element symbols, MAT numbers, natural abundances, ZA encoding |
| [`nereids-core`](api/nereids_core/) | Core types (`Isotope`, `Resonance`), physical constants, element data, error types |
| [`nereids-endf`](api/nereids_endf/) | ENDF file retrieval from IAEA, local caching, File 2 resonance parameter parsing |
| [`nereids-physics`](api/nereids_physics/) | Cross-section calculation (Reich-Moore, SLBW, RML, URR), Doppler/resolution broadening, Beer-Lambert transmission |
| [`nereids-io`](api/nereids_io/) | TIFF stack and NeXus/HDF5 loading, TOF-to-energy conversion, normalization, export |
| [`nereids-fitting`](api/nereids_fitting/) | Levenberg-Marquardt and Poisson KL divergence optimizers, transmission fit model |
| [`nereids-pipeline`](api/nereids_pipeline/) | Single-spectrum fitting, per-pixel spatial mapping (rayon), trace detectability |
| `nereids-python` | PyO3 Python bindings (not published to crates.io) |
| `nereids-gui` | egui desktop application (not published to crates.io) |

## Data Flow

The standard analysis pipeline processes data through these stages:

```text
Raw TOF data (TIFF/NeXus)
    │
    ▼
Normalization (sample / open_beam → transmission)     [nereids-io]
    │
    ▼
Energy conversion (TOF bin edges → energy centers)    [nereids-io]
    │
    ▼
ENDF data (fetch resonance parameters from IAEA)      [nereids-endf]
    │
    ▼
Forward model (cross-sections → broadening → T(E))    [nereids-physics]
    │
    ▼
Fitting (minimize |T_measured - T_model|)              [nereids-fitting]
    │
    ▼
Spatial mapping (fit each pixel in parallel)           [nereids-pipeline]
    │
    ▼
Density maps, chi² maps, convergence maps              [output]
```

## Key Design Decisions

### Exact SAMMY Physics

All physics modules implement the exact formalisms from the SAMMY Fortran code,
with no ad-hoc approximations. Every module references specific SAMMY source
files and equation numbers. See the [Physics Reference](./physics.md) for
details.

### Workspace Architecture

The workspace is structured so that each crate has a single responsibility
and minimal dependencies. `nereids-core` is the foundation with zero internal
dependencies. Higher-level crates compose lower-level ones.

See [ADR 0001](https://github.com/ornlneutronimaging/NEREIDS/blob/main/docs/adr/0001-workspace-layout.md)
for the full rationale.

### Parallel Spatial Mapping

Per-pixel fitting uses [rayon](https://docs.rs/rayon) for data parallelism.
The outer pixel loop runs on a dedicated thread pool to avoid deadlocking
with inner parallel operations (cross-section calculation, broadening).

### Python Bindings via PyO3

The Python bindings expose a high-level API (`load_endf`, `forward_model`,
`fit_spectrum`, `spatial_map`) that maps directly to the Rust pipeline.
NumPy arrays are zero-copy where possible via `numpy` crate integration.
