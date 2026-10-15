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
| [`nereids-physics`](api/nereids_physics/) | Cross-section calculation (Reich-Moore, SLBW, MLBW), Doppler/resolution broadening, Beer-Lambert transmission |
| [`nereids-io`](api/nereids_io/) | TIFF stack and NeXus/HDF5 loading, TOF-to-energy conversion, normalization, export |
| [`nereids-fitting`](api/nereids_fitting/) | Poisson/KL counts optimizer |
| [`nereids-pipeline`](api/nereids_pipeline/) | Counts-domain fitting: one region or a map of pixel patches |
| `nereids-python` | PyO3 Python bindings (not published to crates.io) |
| `nereids-gui` | egui desktop application (not published to crates.io) |

## Data Flow

The standard analysis pipeline processes data through these stages:

```text
Raw TOF counts (TIFF/NeXus), sample and open beam
    │
    ▼
ENDF data (resonance parameters)                       [nereids-endf]
    │
    ▼
Forward model (cross-sections → Doppler → pulse → counts) [nereids-physics]
    │
    ▼
Counts fit (Poisson/KL on the two count arms)          [nereids-fitting, nereids-pipeline]
    │
    ▼
Density and temperature, one region or a map of patches [output]
```

## Key Design Decisions

### Exact SAMMY Physics

All physics modules implement the exact formalisms from the SAMMY Fortran code,
with no ad-hoc approximations. Every module references specific SAMMY source
files and equation numbers. See the [API documentation](api/nereids_physics/) for
details.

### Workspace Architecture

The workspace is structured so that each crate has a single responsibility
and minimal dependencies. `nereids-core` is the foundation with zero internal
dependencies. Higher-level crates compose lower-level ones.

See [ADR 0001](https://github.com/ornlneutronimaging/NEREIDS/blob/main/docs/adr/0001-workspace-layout.md)
for the full rationale.

### Parallel counts mapping

`fit_map` fits patches of pixels as one counts fit and propagates the
cross-patch covariance.

### Python Bindings and MCP

The Python bindings expose the counts fit (`fit_counts`) and counts map
(`fit_map`) directly, plus ENDF loading, forward-model and TOF helpers
(`load_endf`, `forward_model`, `tof_to_energy_centers`). TIFF/NeXus I/O
helpers (`load_tiff_stack`, `load_tiff_folder`, `probe_nexus`,
`load_nexus_histogram`, `load_nexus_events`) expose the same spectral-axis
conventions as `nereids-io`. NumPy arrays are zero-copy where possible via the
`numpy` crate integration.

The MCP server is a thin Python package layer over these bindings. It exposes
low-level physics tools and manifest-driven workflow tools for local AI-agent
orchestration; it does not add a separate fitting engine.
