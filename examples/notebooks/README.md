# NEREIDS Jupyter Notebooks

Tutorials for the count-domain API, organized by tier. They are tutorial
examples, not the API contract: the contract is the Python type stubs, the
Python and Rust tests, the mdBook guide and Rustdoc. Notebooks are not
executed by CI.

## Prerequisites

```bash
pip install nereids jupyter matplotlib numpy
```

Notebooks that call `load_endf()` need network access on first run; files are
cached locally afterwards. Plotting needs `matplotlib`.

## Tier 1: Foundations

Physics validation: understand what NEREIDS computes and check it against
analytic formulas and SAMMY reference values.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Cross-Sections](foundations/01_cross_sections.ipynb) | Resonance cross-sections from known parameters |
| 2 | [SLBW Validation](foundations/02_slbw_validation.ipynb) | Single-level Breit-Wigner versus Reich-Moore |
| 3 | [Doppler Broadening](foundations/03_doppler_broadening.ipynb) | Free-gas-model broadening across temperatures |
| 4 | [Resolution Broadening](foundations/04_resolution_broadening.ipynb) | Instrument resolution broadening |
| 6 | [Transmission Model](foundations/06_transmission_model.ipynb) | Cross-sections through broadening to transmission |

## Tier 2: Building Blocks

Individual API features.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [ENDF Loading](building_blocks/01_endf_loading.ipynb) | Fetch and cache evaluated nuclear data from the IAEA |
| 2 | [Element Utilities](building_blocks/02_element_utilities.ipynb) | Element and isotope lookup helpers |

Counts-fitting, mapping and I/O tutorials will be re-added against
`fit_counts` and `fit_map`.
