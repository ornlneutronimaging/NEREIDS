# Introduction

**NEREIDS** (**N**eutron r**E**sonance **RE**solved **I**maging **D**ata Analysis
**S**ystem) is a Rust-based library for neutron resonance imaging at the
[VENUS beamline](https://neutrons.ornl.gov/venus), Spallation Neutron Source (SNS),
Oak Ridge National Laboratory (ORNL).

## What It Does

NEREIDS provides end-to-end analysis for time-of-flight (TOF) neutron resonance
imaging: input hyperspectral TOF data, output spatially resolved isotopic and
elemental composition maps.

The analysis pipeline:

1. **Load** raw TOF imaging data (TIFF stacks, NeXus/HDF5, or pre-normalized transmission)
2. **Normalize** sample and open-beam measurements to transmission
3. **Configure** isotopes of interest using ENDF nuclear data
4. **Fit** resonance models to measured transmission spectra
5. **Map** fitted parameters (areal density, temperature) across each pixel

## Interfaces

NEREIDS ships in several interfaces:

| Deliverable | Use case |
|-------------|----------|
| **Rust library** (`nereids-*` crates) | Embed in Rust applications, maximum performance |
| **Python bindings** (`pip install nereids`) | Jupyter notebooks, scripting, integration with NumPy/SciPy |
| **Desktop GUI** (`nereids-gui`) | Interactive analysis with visual feedback |
| **MCP server** (`nereids-mcp`) | Local AI-agent assisted analysis through manifest-driven workflows |

![NEREIDS landing page](images/landing.png)

## Relationship to SAMMY

NEREIDS implements the same physics as SAMMY
(a Fortran code for multilevel R-matrix analysis of neutron data, ORNL/TM-9179/R8), rewritten in
Rust with modern tooling. All physics modules reference specific SAMMY source files
and equation numbers in their documentation.

Key formalisms from SAMMY:

- Reich-Moore R-matrix (LRF=3)
- Breit-Wigner, single- and multi-level (LRF=1/2)
- Free Gas Model Doppler broadening
- Gaussian + exponential resolution broadening

R-Matrix Limited (LRF=7) and Unresolved Resonance Region (LRU=2) ranges are
parsed-and-skipped, not evaluated: they contribute zero cross-section, and a
file containing only such ranges is rejected at load time.

## Next Steps

- [Install NEREIDS](./installation.md) for your platform
- Try the [Rust quickstart](./quickstart-rust.md) or [Python quickstart](./quickstart-python.md)
- Explore the [GUI walkthrough](./gui-walkthrough.md) for interactive analysis
