# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- License switched from BSD-3-Clause to MIT across the entire workspace —
  the root `LICENSE` text, the workspace `Cargo.toml` license field inherited
  by every crate, and `endf-mat`'s former `MIT OR Apache-2.0` dual license
  (now also inherits the workspace MIT field). Python wheel metadata
  (`pyproject.toml` license + classifier for both `nereids` and
  `nereids-gui`), `CITATION.cff`, the README badge/License section, and the
  CONTRIBUTING contributor-license statement all updated to match. Copyright
  years are now 2025-2026, and the copyright holder reads
  "UT-Battelle, LLC (Oak Ridge National Laboratory)" everywhere. Per the
  repository's bump-at-release convention, in-tree metadata (manifests,
  CITATION.cff) describes the current source tree — which is MIT — while
  still carrying the last released version number until the next release
  commit bumps it; the 0.1.8 artifacts already on crates.io, PyPI, and
  Zenodo were published under BSD-3-Clause (endf-mat: MIT OR Apache-2.0)
  and are unaffected. MIT applies to every version published from here on.
  Each publishable crate directory (and `apps/gui` for the wheel) now
  carries a copy of the LICENSE text, so published artifacts ship the MIT
  permission notice they are licensed under.

### Changed (breaking — `nereids-io`)

- `NexusMetadata.tof_edges_ns: Option<Vec<f64>>` has been renamed to
  `NexusMetadata.tof_edges_us: Option<Vec<f64>>`.  The old field name was
  a misnomer: the values stored in it were already microseconds after
  `units`-attribute rescaling inside `probe_histogram_group` — they were
  never nanoseconds.  The field is now correctly named so callers do not
  need to second-guess the unit at the API boundary.  No data conversion
  is needed for downstream consumers; the numeric payload is unchanged.
  Per semantic versioning for `0.x` crates, this is an acceptable
  breaking change at this stage of the project; no external consumers
  of this field have been identified in the workspace or sibling repos.

## [0.1.8] - 2026-04-27

### Added

- TENDL-2023 ENDF library with full MAT-table coverage (#508)
- GUI: optional fitting of the energy scale — TZERO + flight-path L scale (#501)
- GUI: vertically stacked image + spectrum layout in the Analyze step (#506)
- Counts-domain joint-Poisson (KL) fitting with 2D spatial-map integration and
  Python API parity (#450)
- Sparse empirical-cubature forward-model surrogate for multi-isotope fits, plus
  a scalar k=1 Chebyshev surrogate (epic #472: #479, #481, #482)
- VENUS Hf 120-min NeXus test fixtures (via the PLEIADES submodule)

### Changed

- Resolution broadening rebuilt around a reusable `ResolutionPlan`, compiled to a
  CSR `ResolutionMatrix` (~4.2× faster apply, bit-exact) and wired through the
  fixed-grid fit models (#467, #468, #470, #478)
- Two-pointer walk in `broaden_presorted` (~4.2× on VENUS) (#464)
- Energy-scale (TZERO) Jacobian: density-column plan cache + partial-GAL
  default (#469, #484, #498)
- KL "polish" pass now off by default (~100× faster single-spectrum fits) (#487)

### Fixed

- MLBW batch-API correctness + dispatch consolidation, gated against VENUS (#466)
- GUI: invalidate the cached fit when solver controls change (#507)
- ENDF: use the OS trust roots for IAEA fetches (#505)
- Spatial map: NaN-on-failure handling, config guards, and binding
  validation (#461)
- Refuse to silently collapse multiple rotation angles in the NeXus histogram
  loader (#463)

### Build

- Bump `rustls-webpki` (#477) and `rand` (#449)

## [0.1.7] - 2026-04-04

### Added

- Uncertainty estimation and propagation, Phases 1-4 (#446)
- Python bindings for NeXus histogram and event loading (#447)
- Resolution-aware analytical Jacobians (#445)

### Fixed

- Apply resolution broadening *after* Beer-Lambert, including the temperature
  path of `TransmissionFitModel` (#442 → #443, #444)

## [0.1.6] - 2026-03-23

### Added

- Unified typed fitting pipeline: a solver-agnostic `ForwardModel` trait, typed
  `InputData` constructors (`from_counts` / `from_transmission`),
  `UnifiedFitConfig`, and the `fit_spectrum_typed` / `spatial_map_typed` entry
  points (Python + Rust)
- True multi-level Breit-Wigner (MLBW) with a coherent U-matrix, replacing the
  prior SLBW-dispatch approximation
- SAMMY-style normalization and background fitting for transmission, plus a
  KL-native background model for counts data
- Counts + KL temperature fitting with analytical Doppler temperature derivatives
- Unresolved-resonance region: LFW=1 (energy-dependent fission widths) and all
  five ENDF interpolation laws (INT=1..5)
- Energy calibration (`calibrate_energy`) with resolution broadening
- Constrained isotope-group fitting (core, GUI, and a tutorial notebook)
- Analytical Jacobian for the Poisson solver (incl. a temperature derivative)

### Changed

- Consolidated the fitting stack on a single optimizer and API: removed the
  external L-BFGS-B solver and the legacy pre-typed fitting/spatial API (the
  Poisson/KL path now uses an analytic Gauss-Newton step with an internal
  finite-difference L-BFGS fallback)
- ROI spectrum averaging now uses inverse-variance weighting instead of an
  arithmetic mean
- `spatial_map` / `sparse_reconstruct` `n_total` now count *attempted* pixels,
  not only successes
- Wrapped large shared buffers (`sample_data`, `open_beam_data`,
  `spectrum_values`) in `Arc` to avoid per-pixel clones (#374)
- Project files now persist normalized uncertainty, dead-pixel masks, and the
  anorm / background maps

### Fixed

- RML phase-convention correction (+ documented SAMMY truth sources); MLBW phase
  convention + total cross-section regression
- JENDL-5 Eu-151/153 parse failure: deduplicated URR energy grid (#402, #411)
- Detectability SNR computation and resolution-grid handling
- GUI ENDF-fetch defects and chip layout (#403, #410)
- Data-integrity hardening: reject synthetic energy axes, count load errors,
  eliminate silent pixel drops

### Project

- Added `CITATION.cff`, `CONTRIBUTING.md`, `CHANGELOG.md`, a Code of Conduct, and
  a Zenodo DOI badge

## [0.1.5] - 2026-03-11

### Fixed
- Switched HTTP backend from `native-tls` to `rustls-tls` to eliminate
  OpenSSL runtime dependency on Linux clusters
- Fixed crates.io publish detection for already-existing crates
- Fixed PyPI sdist LICENSE file path
- Built Python wheels for all supported versions (3.10-3.13) on all platforms

## [0.1.0] - 2026-03-11

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
- Joint temperature + density fitting with bounds-based preconditioning

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

[Unreleased]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/ornlneutronimaging/NEREIDS/releases/tag/v0.1.0
