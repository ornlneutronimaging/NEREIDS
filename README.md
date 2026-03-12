# NEREIDS

**N**eutron r**E**sonance **RE**solved **I**maging **D**ata analysis **S**ystem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18973054.svg)](https://doi.org/10.5281/zenodo.18973054)
[![CI](https://github.com/ornlneutronimaging/NEREIDS/actions/workflows/ci.yml/badge.svg)](https://github.com/ornlneutronimaging/NEREIDS/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nereids)](https://pypi.org/project/nereids/)
[![crates.io](https://img.shields.io/crates/v/nereids-core)](https://crates.io/crates/nereids-core)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-guide-blue)](https://ornlneutronimaging.github.io/NEREIDS/)
[![API](https://img.shields.io/badge/docs-rustdoc-orange)](https://ornlneutronimaging.github.io/NEREIDS/api/nereids_pipeline/)

NEREIDS is a Rust-based library for neutron resonance imaging at the
[VENUS beamline](https://neutrons.ornl.gov/venus), Spallation Neutron Source
(SNS), Oak Ridge National Laboratory. It provides end-to-end analysis for
time-of-flight neutron transmission imaging: input hyperspectral TOF data,
output spatially resolved isotopic composition maps.

## Features

- **R-matrix cross-sections** -- Reich-Moore, SLBW, R-Matrix Limited (LRF=7),
  Unresolved Resonance Region (URR/Hauser-Feshbach), Coulomb channels
- **Doppler broadening** -- Free Gas Model with crystal-lattice fallback
- **Resolution broadening** -- Gaussian (channel width + flight path) and
  tabulated instrument functions
- **ENDF/B data** -- automatic retrieval and caching from IAEA for all
  evaluated libraries (ENDF/B-VIII.0, JEFF-3.3, JENDL-5, etc.)
- **Spectrum fitting** -- Levenberg-Marquardt and Poisson/KL divergence
  optimizers with analytical Jacobians
- **Spatial mapping** -- parallel per-pixel fitting via rayon for 2D
  isotopic density maps
- **Detectability analysis** -- energy-window optimization for trace
  element sensitivity
- **Python bindings** -- full API via PyO3, pip-installable
- **Desktop GUI** -- egui application with guided workflow and studio mode

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

# Fit the spectrum to recover densities
result = nereids.fit_spectrum(
    measured_t=transmission,
    sigma=np.full_like(transmission, 0.01),
    energies=energies,
    isotopes=[u238, fe56],
    temperature_k=293.6,
)

print(f"U-238: {result.densities[0]:.4f} at/barn (true: 0.005)")
print(f"Fe-56: {result.densities[1]:.4f} at/barn (true: 0.01)")
print(f"Converged: {result.converged}, chi2r: {result.reduced_chi_squared:.3f}")
```

See the [examples/notebooks/](examples/notebooks/) directory for 17 tutorial
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
nereids-fitting       LM and Poisson/L-BFGS-B optimizers
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
| `nereids-fitting` | Optimization engine (LM, Poisson/L-BFGS-B) |
| `nereids-pipeline` | End-to-end orchestration and spatial mapping |
| `nereids-python` | PyO3 Python bindings for Jupyter |
| `nereids-gui` | egui desktop application |

## Documentation

- **[User Guide](https://ornlneutronimaging.github.io/NEREIDS/)** -- Installation, quickstart, architecture
- **[API Reference](https://ornlneutronimaging.github.io/NEREIDS/api/nereids_pipeline/)** -- Rustdoc for all crates
- **[Jupyter Notebooks](examples/notebooks/)** -- 17 tutorials organized by complexity

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

BSD-3-Clause. See [LICENSE](LICENSE) for details.

Copyright (c) 2025, UT-Battelle, LLC, Oak Ridge National Laboratory.
