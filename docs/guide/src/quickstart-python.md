# Quickstart: Python

This example uses the NEREIDS Python bindings to load ENDF data, compute a
forward model, and fit a transmission spectrum.

## Setup

```bash
pip install nereids numpy matplotlib
```

## Load ENDF Data and Compute Transmission

```python
import nereids
import numpy as np

# Load ENDF/B-VIII.1 resonance data for U-238
u238 = nereids.load_endf(92, 238)
print(f"U-238: {u238.n_resonances} resonances, AWR = {u238.awr:.1f}")

# Energy grid: 1 to 30 eV
energies = np.linspace(1.0, 30.0, 2000)

# Compute transmission for 0.001 atoms/barn at 300 K
transmission = nereids.forward_model(
    energies,
    [(u238, 0.001)],
    temperature_k=300.0,
    flight_path_m=25.0,
    delta_t_us=5.0,
    delta_l_m=0.005,
)
```

## Plot the Spectrum

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(energies, transmission, linewidth=0.8)
plt.xlabel("Energy (eV)")
plt.ylabel("Transmission")
plt.title("U-238 Forward Model (0.001 at/barn, 300 K)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Spatial Mapping

For imaging data (3D transmission arrays), use the typed API:

```python
# transmission_3d: shape (n_energies, height, width)
# uncertainty_3d:  shape (n_energies, height, width)

data = nereids.from_transmission(transmission_3d, uncertainty_3d)
result = nereids.spatial_map_typed(
    data,
    energies,
    [u238],
    temperature_k=300.0,
    flight_path_m=25.0,
    delta_t_us=5.0,
    delta_l_m=0.005,
)

# result.density_maps[0] is a 2D array of U-238 areal density at each pixel
# result.converged_map shows which pixels converged
print(f"Converged: {result.n_converged}/{result.n_total} pixels")
```

For raw count data (Poisson-optimal fitting):

```python
data = nereids.from_counts(sample_counts_3d, open_beam_counts_3d)
result = nereids.spatial_map_typed(data, energies, [u238])
```

## Single-Spectrum Fitting

Use `fit_spectrum_typed(...)` for pre-normalized transmission spectra:

```python
uncertainty = np.full_like(transmission, 0.01)
fit = nereids.fit_spectrum_typed(
    transmission,
    uncertainty,
    energies,
    [(u238, 0.0005)],
    temperature_k=300.0,
)
print(fit.densities, fit.reduced_chi_squared)
```

Use `fit_counts_spectrum_typed(...)` for raw sample/open-beam counts:

```python
fit = nereids.fit_counts_spectrum_typed(
    sample_counts_1d,
    open_beam_counts_1d,
    energies,
    [(u238, 0.0005)],
    c=sample_charge / open_beam_charge,
)
print(fit.densities, fit.deviance_per_dof)
```

See the [Python API reference](./python-api.md) for argument details, shape
contracts, and result objects.

## TIFF and NeXus Data

TIFF and NeXus loaders use the spectral axis first:

```python
stack = nereids.load_tiff_stack("transmission_stack.tif")

sample = nereids.load_nexus_histogram("sample.nxs")
open_beam = nereids.load_nexus_histogram("open_beam.nxs")
energies = nereids.tof_to_energy_centers(
    sample.tof_edges_us,
    sample.flight_path_m or 25.0,
)
```

NeXus counts are loaded in ascending TOF order. Reverse axis 0 before fitting
against the ascending energy centers returned by `tof_to_energy_centers(...)`.
The [Data I/O and NeXus/TOF](./data-io.md) chapter covers the full workflow.

## Detectability Analysis

Check whether a trace isotope is detectable in a given matrix:

```python
fe56 = nereids.load_endf(26, 56)  # matrix: Fe-56
ag107 = nereids.load_endf(47, 107)  # trace: Ag-107

report = nereids.trace_detectability(
    matrix=fe56,
    matrix_density=0.01,
    trace=ag107,
    trace_ppm=100.0,
    energies=energies,
    i0=1e6,
)

print(f"Detectable: {report.detectable}")
print(f"Peak SNR: {report.peak_snr:.1f} at {report.peak_energy_ev:.2f} eV")
```

## Next Steps

- See the [GUI walkthrough](./gui-walkthrough.md) for interactive analysis
- Use the [MCP server](./mcp-server.md) for local AI-agent assisted workflows
- Browse the [Python API reference](./python-api.md)
- Read [Data I/O and NeXus/TOF](./data-io.md) for TIFF, NeXus, and TOF data handling
- Explore the [Architecture](./architecture.md) chapter for the crate structure
- Browse the [API Reference](api/nereids_pipeline/) for the full Rust docs
