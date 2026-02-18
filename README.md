# NEREIDS

**N**eutron r**E**sonance **RE**solved **I**maging **D**ata analysis **S**ystem

A Rust-based library for neutron resonance imaging at the VENUS beamline, SNS, ORNL.

## Overview

NEREIDS provides end-to-end analysis for time-of-flight neutron resonance imaging:
input hyperspectral TOF data, output spatially resolved isotopic/elemental
composition maps.

## Workspace Crates

| Crate | Description |
|-------|-------------|
| `nereids-core` | Core types, physical constants, traits |
| `nereids-endf` | ENDF file retrieval, caching, resonance parameter parsing |
| `nereids-physics` | Cross-section calculation, broadening, transmission model |
| `nereids-io` | TIFF/NeXus data I/O, VENUS normalization |
| `nereids-fitting` | Optimization engine (LM, Poisson/BFGS) |
| `nereids-pipeline` | End-to-end orchestration and spatial mapping |
| `nereids-python` | PyO3 Python bindings for Jupyter |
| `nereids-gui` | egui desktop application |

## Building

```bash
cargo build --workspace
cargo test --workspace
```

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
