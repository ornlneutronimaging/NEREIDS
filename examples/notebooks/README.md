# NEREIDS Jupyter Notebooks

17 tutorials organized into four tiers of increasing complexity.
Work through them in order, or jump to whichever tier matches your
experience level.

## Prerequisites

```bash
pip install nereids jupyter matplotlib numpy
```

Notebooks that load ENDF data (`load_endf()`) require an internet
connection on the first run; files are cached locally afterwards.

## Tier 1: Foundations

Core physics validation -- understand what NEREIDS computes and verify
it against analytical formulas and SAMMY reference values.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Cross-Sections](foundations/01_cross_sections.ipynb) | Compute resonance cross-sections from known parameters, validate against SAMMY |
| 2 | [SLBW Validation](foundations/02_slbw_validation.ipynb) | Compare Single-Level Breit-Wigner and Reich-Moore formalisms |
| 3 | [Doppler Broadening](foundations/03_doppler_broadening.ipynb) | Validate Free Gas Model Doppler broadening across temperatures |
| 4 | [Resolution Broadening](foundations/04_resolution_broadening.ipynb) | Validate instrument resolution broadening against SAMMY |
| 5 | [URR Cross-Sections](foundations/05_urr_cross_sections.ipynb) | Unresolved Resonance Region via Hauser-Feshbach |
| 6 | [Transmission Model](foundations/06_transmission_model.ipynb) | Full forward model: cross-sections through broadening to transmission |

## Tier 2: Building Blocks

Individual API features -- learn each piece before combining them.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [ENDF Loading](building_blocks/01_endf_loading.ipynb) | Fetch and cache evaluated nuclear data from the IAEA |
| 2 | [Element Utilities](building_blocks/02_element_utilities.ipynb) | Element/isotope lookup functions (symbol, name, Z, natural abundances) |
| 3 | [Spectrum Fitting](building_blocks/03_spectrum_fitting.ipynb) | Levenberg-Marquardt fitting of a single transmission spectrum |
| 4 | [Multi-Isotope Fitting](building_blocks/04_multi_isotope_fitting.ipynb) | Fit a mixed-sample spectrum to recover multiple isotope densities |
| 5 | [Tabulated Resolution](building_blocks/05_custom_resolution.ipynb) | Load and apply tabulated instrument resolution kernels |
| 6 | [TIFF I/O](building_blocks/06_tiff_io_normalization.ipynb) | Load TIFF stacks and normalize raw detector images |

## Tier 3: Workflows

End-to-end analysis recipes combining multiple building blocks.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Enrichment Analysis](workflows/01_enrichment_analysis.ipynb) | Determine uranium enrichment from spectral fingerprints |
| 2 | [Trace Analysis](workflows/02_trace_analysis.ipynb) | Detect trace elements in a bulk matrix via energy-window optimization |
| 3 | [Forward Model](workflows/03_forward_model_demo.ipynb) | Complete forward-modeling pipeline from ENDF data to synthetic spectra |
| 4 | [Spatial Mapping (Synthetic)](workflows/04_spatial_mapping_synthetic.ipynb) | Per-pixel isotopic composition mapping on a synthetic phantom |

## Tier 4: Applications

Full-scale demonstrations on reference datasets.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Spatial Mapping Demo](applications/01_spatial_mapping_demo.ipynb) | Complete NEREIDS pipeline on a reference dataset with real detector geometry |

## Suggested Learning Path

**New to NEREIDS:** Foundations 1-3, then Building Blocks 1 and 3, then
Workflows 3 (forward model). This covers the core physics and the most
common API pattern.

**Ready for fitting:** Building Blocks 3-4, then Workflows 1-2.

**Spatial imaging:** Workflows 4, then Applications 1.
