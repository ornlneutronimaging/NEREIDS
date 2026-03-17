# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.5] - 2025-03-11

### Fixed
- Switched HTTP backend from `native-tls` to `rustls-tls` to eliminate
  OpenSSL runtime dependency on Linux clusters
- Fixed crates.io publish detection for already-existing crates
- Fixed PyPI sdist LICENSE file path
- Built Python wheels for all supported versions (3.10-3.13) on all platforms

## [0.1.0] - 2025-03-10

### Added

#### Cross-Section Physics
- Reich-Moore R-matrix cross-section calculation
- Single-Level Breit-Wigner (SLBW) formalism
- R-Matrix Limited (LRF=7) with KRM=2 and KRM=3 support
- Unresolved Resonance Region (URR) via Hauser-Feshbach
- Coulomb wave functions (Steed's continued-fraction method)
- Energy-dependent scattering radius (NRO=1 TAB1 interpolation)

#### Broadening Models
- Free Gas Model Doppler broadening (O(N*W) optimized)
- Resolution broadening: Gaussian (channel width + flight path) and
  tabulated instrument functions
- Joint temperature + density fitting with Fisher preconditioning

#### Data Processing
- ENDF/B resonance parameter loading from IAEA (all evaluated libraries)
- 535 built-in MAT numbers in `endf-mat` crate
- TIFF stack and folder I/O with TOF normalization
- NeXus/HDF5 histogram and event data loading
- Energy rebinning (sum for counts, average for transmission)

#### Fitting & Analysis
- Levenberg-Marquardt optimizer with analytical Beer-Lambert Jacobian
- Poisson/KL divergence optimizer (analytic path preferred; L-BFGS-B available) for low-count data
- Parallel per-pixel spatial mapping via rayon
- Region-of-interest (ROI) spectrum fitting
- Trace-element detectability analysis with energy-window optimization

#### Python Bindings
- Full PyO3 API: cross-sections, forward model, fitting, spatial mapping,
  I/O, detectability, element utilities
- Type stubs (PEP 561) for IDE support
- 25 Python tests

#### GUI Application
- egui desktop application with guided 5-step workflow
- Landing page with decision wizard (6 pipeline configurations)
- Studio mode: three-pane result explorer with density maps
- Forward Model, Detectability, and Periodic Table tools
- Project file save/load (.nrd.h5) with embedded and linked modes
- Session persistence via eframe::Storage
- macOS DMG and pip-installable wheel distribution

#### Notebooks
- 17 Jupyter tutorials across 4 tiers:
  - Foundations (6): cross-sections, SLBW, Doppler, resolution, URR, transmission
  - Building Blocks (6): ENDF loading, element utilities, fitting, multi-isotope,
    custom resolution, TIFF I/O
  - Workflows (4): enrichment analysis, trace analysis, forward model, spatial mapping
  - Applications (1): full 2D isotopic density mapping demo

#### Infrastructure
- CI/CD: cross-platform tests (Linux, macOS, Windows), coverage, rustdoc
- Publish pipeline: PyPI, crates.io, GitHub Releases, Homebrew cask
- Documentation site: mdBook user guide + rustdoc API reference on GitHub Pages
- SAMMY validation suite: 43 test cases validated against SAMMY reference code

[0.1.5]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/ornlneutronimaging/NEREIDS/releases/tag/v0.1.0
