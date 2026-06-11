# NEREIDS

**N**eutron r**E**sonance **RE**solved **I**maging **D**ata analysis **S**ystem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18973054.svg)](https://doi.org/10.5281/zenodo.18973054)
[![CI](https://github.com/ornlneutronimaging/NEREIDS/actions/workflows/ci.yml/badge.svg)](https://github.com/ornlneutronimaging/NEREIDS/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nereids)](https://pypi.org/project/nereids/)
[![crates.io](https://img.shields.io/crates/v/nereids-core)](https://crates.io/crates/nereids-core)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-guide-blue)](https://ornlneutronimaging.github.io/NEREIDS/)
[![API](https://img.shields.io/badge/docs-rustdoc-orange)](https://ornlneutronimaging.github.io/NEREIDS/api/nereids_pipeline/)

NEREIDS is a Rust-based library for neutron resonance imaging at the
[VENUS beamline](https://neutrons.ornl.gov/venus), Spallation Neutron Source
(SNS), Oak Ridge National Laboratory. It provides end-to-end analysis for
time-of-flight neutron transmission imaging: input hyperspectral TOF data,
output spatially resolved isotopic composition maps.

## Features

- **R-matrix cross-sections** -- Reich-Moore, Breit-Wigner (single- and
  multi-level), R-Matrix Limited (LRF=7), Coulomb channels
- **Unresolved Resonance Region** -- energy-averaged Hauser-Feshbach
  cross-sections (width-fluctuation correction planned, not yet implemented)
- **Doppler broadening** -- Free Gas Model (crystal-lattice model planned, not yet implemented)
- **Resolution broadening** -- Gaussian (channel width + flight path) and
  tabulated instrument functions
- **ENDF nuclear data** -- automatic retrieval and caching (ENDF/B-VIII.0
  and VIII.1 from NNDC with IAEA fallback; JEFF-3.3, JENDL-5, TENDL-2023,
  and CENDL-3.2 from IAEA)
- **Spectrum fitting** -- Levenberg-Marquardt and Poisson/KL divergence
  optimizers with analytical Jacobians
- **Spatial mapping** -- parallel per-pixel fitting via rayon for 2D
  isotopic density maps
- **Detectability analysis** -- energy-window optimization for trace
  element sensitivity
- **Python bindings** -- full API via PyO3, pip-installable
- **Desktop GUI** -- egui application with Guided and Studio modes

## Installation

### Python (recommended)

```bash
pip install nereids
```

Requires Python >= 3.10. Prebuilt wheels are available for Linux (x86_64),
macOS (ARM), and Windows (x86_64).

### Rust

Add individual crates to your `Cargo.toml`:

```toml
[dependencies]
nereids-core = "0.1"
nereids-endf = "0.1"
nereids-physics = "0.1"
```

### GUI application

**macOS (Homebrew):**

```bash
brew tap ornlneutronimaging/nereids
brew install --cask nereids
```

**macOS/Linux (pip):**

```bash
pip install nereids-gui
nereids-gui
```

### MCP server for AI agents

NEREIDS can run as a local [Model Context Protocol](https://modelcontextprotocol.io/)
server so AI agents can inspect neutron-resonance data, fit spectra, and run
spatial density-map workflows through the Python bindings.

```bash
pip install "nereids[mcp]"
nereids-mcp
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "nereids": {
      "command": "nereids-mcp"
    }
  }
}
```

The server exposes low-level physics tools (`load_endf`,
`compute_cross_sections`, `compute_transmission`, `detect_isotopes`) and
workflow tools for agent-driven processing:

- `extract_resonance_manifest(dataset_path)` reads an sMCP-style manifest.
- `validate_resonance_dataset(dataset_path)` checks data, isotope, and
  resolution configuration.
- `process_resonance_dataset(dataset_path)` runs a `single_spectrum` or
  `density_map` workflow and writes compact `.npz` result artifacts.

Datasets can include `manifest_intermediate.md`, `smcp_manifest.md`,
`nereids_manifest.md`, `nereids_mcp.json`, or `analysis.json`. Markdown
manifests use frontmatter between `---` delimiters; JSON frontmatter is
supported without extra YAML dependencies.

See the [MCP server guide](docs/guide/src/mcp-server.md) for supported data
kinds, manifest fields, NeXus histogram handling, and output artifacts.

Minimal spectrum manifest:

```markdown
---
{
  "name": "venus-hf-spectrum",
  "tool": "nereids",
  "physics": "neutron-resonance",
  "analysis": {
    "mode": "single_spectrum",
    "data": {
      "kind": "counts_npz",
      "path": "aggregated_hf_120min.npz",
      "pc_ratio": 5.98
    },
    "isotopes": [
      {"isotope": "Hf-177", "endf_file": "Hf-177.endf", "initial_density": 1e-5}
    ],
    "fit": {"solver": "lm", "fit_domain": "transmission", "max_iter": 100},
    "resolution": {
      "kind": "gaussian",
      "flight_path_m": 25.0,
      "delta_t_us": 0.5,
      "delta_l_m": 0.005
    },
    "output": {"directory": "output"}
  }
}
---
```

### From source

```bash
git clone https://github.com/ornlneutronimaging/NEREIDS.git
cd NEREIDS
cargo build --workspace --release
cargo test --workspace --exclude nereids-python
```

Python bindings require [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin develop --release -m bindings/python/Cargo.toml
```

## Quick Start (Python)

```python
import numpy as np
import nereids

# Load ENDF resonance data
u238 = nereids.load_endf(92, 238)
fe56 = nereids.load_endf(26, 56)

# Energy grid (1-200 eV covers the strong U-238 resonances)
energies = np.linspace(1.0, 200.0, 5000)

# Forward model: transmission through a mixed sample
transmission = nereids.forward_model(
    energies,
    isotopes=[(u238, 0.005), (fe56, 0.01)],  # (data, areal density in at/barn)
    temperature_k=293.6,
)

# Spatial mapping with typed API
trans_3d = transmission[:, None, None] * np.ones((1, 4, 4))  # (n_e, ny, nx)
sigma_3d = np.full_like(trans_3d, 0.01)
data = nereids.from_transmission(trans_3d, sigma_3d)
result = nereids.spatial_map_typed(data, energies, [u238, fe56])

print(f"Converged: {result.n_converged}/{result.n_total} pixels")
```

See the [examples/notebooks/](examples/notebooks/) directory for 18 tutorial
notebooks covering foundations, building blocks, workflows, and applications.

## Architecture

NEREIDS is organized as a Rust workspace with layered crates:

```
nereids-core          Shared types, physical constants, isotope registry
    |
nereids-endf          ENDF file retrieval, parsing, resonance data
    |
nereids-physics       Cross-sections, broadening, transmission model
    |
nereids-fitting       LM and Poisson/KL optimizers
    |
nereids-io            TIFF/NeXus I/O, TOF normalization, rebinning
    |
nereids-pipeline      End-to-end orchestration, spatial mapping (rayon)
    |
    +-- nereids-python    PyO3 Python bindings
    +-- nereids-gui       egui desktop application
```

| Crate | Description |
|-------|-------------|
| `nereids-core` | Core types, physical constants, traits |
| `nereids-endf` | ENDF file retrieval, caching, resonance parameter parsing |
| `nereids-physics` | Cross-section calculation, broadening, transmission model |
| `nereids-io` | TIFF/NeXus data I/O, VENUS normalization |
| `nereids-fitting` | Optimization engine (LM, Poisson/KL) |
| `nereids-pipeline` | End-to-end orchestration and spatial mapping |
| `nereids-python` | PyO3 Python bindings for Jupyter |
| `nereids-gui` | egui desktop application |

## Documentation

- **[User Guide](https://ornlneutronimaging.github.io/NEREIDS/)** -- Installation, quickstart, architecture
- **[API Reference](https://ornlneutronimaging.github.io/NEREIDS/api/nereids_pipeline/)** -- Rustdoc for all crates
- **[Jupyter Notebooks](examples/notebooks/)** -- 18 tutorials organized by complexity

## Troubleshooting

### Log files

The NEREIDS desktop GUI (`nereids-gui`) writes daily-rotated log files
to a platform-specific data directory, retaining the last 7 days. Each
day's file is named `nereids-gui.YYYY-MM-DD.log` (UTC date):

| Platform | Log directory |
|----------|---------------|
| macOS    | `~/Library/Application Support/NEREIDS/logs/` |
| Linux    | `~/.local/share/NEREIDS/logs/` (honours `$XDG_DATA_HOME`) |
| Windows  | `%APPDATA%\NEREIDS\logs\` |

From the running app, use **Help → Open log folder** (reveals today's
log file in your platform's file manager) or **Help → Copy log path**
to copy the full path to the clipboard.

The default log level is `info`. Override with the `NEREIDS_LOG`
environment variable (or the standard `RUST_LOG`), e.g.:

```bash
NEREIDS_LOG=debug nereids-gui
NEREIDS_LOG="nereids_pipeline=trace,info" nereids-gui
```

`NEREIDS_LOG` takes precedence over `RUST_LOG`. Panics are captured to
the log file with a backtrace before the process exits — please attach
the relevant log file when reporting bugs.

## Citation

If you use NEREIDS in your research, please cite:

```bibtex
@software{nereids2025,
  author    = {{ORNL Neutron Imaging Team}},
  title     = {{NEREIDS}: Neutron Resonance Resolved Imaging Data Analysis System},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18973054},
  url       = {https://doi.org/10.5281/zenodo.18973054}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards,
and the PR process.

## License

MIT. See [LICENSE](LICENSE) for details.

Copyright (c) 2025-2026, UT-Battelle, LLC, Oak Ridge National Laboratory.
