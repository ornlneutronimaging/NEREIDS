# Quickstart: Python

This example uses the NEREIDS Python bindings to load ENDF data, compute a
forward model, and fit raw open-beam and sample counts in the time domain.

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

## Counts Fitting

Use the original detector time-bin edges in microseconds. Do not reconstruct
them from energy centers. Build the measurement and fit the material:

```python
edges = np.asarray(time_edges_us, dtype=np.float64)

region = nereids.Region(
    open_counts=open_beam_counts_1d,
    sample_counts=sample_counts_1d,
    pixels=1.0,
    background=[nereids.Value.known(0.0)] * 3,
    gamma_transmission=nereids.Value.known(1.0),
    material=nereids.Material(
        isotopes=[(u238, nereids.Value.fitted(0.0005))],
        temperature_k=nereids.Value.known(300.0),
    ),
)
measurement = nereids.Measurement(
    time_edges_us=edges,
    charge_ratio=sample_charge / open_beam_charge,
    normalization=nereids.Value.fitted(1.0),
    regions=[region],
)
calibration = nereids.Calibration(
    t0_us=nereids.Value.known(3.0),
    flight_path_m=nereids.Value.known(25.0),
    pulse=nereids.Pulse(
        alpha0=nereids.Value.known(0.35),
        alpha1=nereids.Value.known(0.05),
        beta0=nereids.Value.known(0.25),
        beta1=nereids.Value.known(0.0),
        r=nereids.Value.known(0.15),
        fwhm_squared_us2=nereids.Value.known(0.0),
        energy_span_ev=(1.0, 200.0),
    ),
)
fit = nereids.fit_counts(measurement, calibration)
print(fit.regions[0].densities, fit.deviance, fit.converged)
```

Every fitted quantity is a Value: known holds it, fitted frees it from a start,
within bounds it, and measured gives it a Gaussian prior. The pulse is the
moderator model; a tabulated instrument resolution is not part of this engine.

## Counts Mapping

For imaging data, fit every patch of a counts cube in one fit:

```python
map_measurement = nereids.MapMeasurement(
    time_edges_us=edges,
    open_counts=open_beam_counts_3d,   # (n_bins, height, width)
    sample_counts=sample_counts_3d,    # (n_bins, height, width)
    excluded=np.zeros((height, width), dtype=bool),
    sample=np.ones((height, width), dtype=bool),
    empty=np.zeros((height, width), dtype=bool),
    material=material,
    charge_ratio=1.0,
    normalization=nereids.Value.fitted(1.0),
    background=[nereids.Value.known(0.0)] * 3,
    binning=1,
)
counts_map = nereids.fit_map(map_measurement, calibration)
print(counts_map.densities[0].shape, counts_map.converged)
```

See the [Python API reference](./python-api.md) for argument details, shape
contracts, and result objects.

## TIFF and NeXus Data

TIFF and NeXus loaders use the spectral axis first:

```python
stack = nereids.load_tiff_stack("counts_stack.tif")

sample = nereids.load_nexus_histogram("sample.nxs")
open_beam = nereids.load_nexus_histogram("open_beam.nxs")
edges = sample.tof_edges_us
```

The [Data I/O and NeXus/TOF](./data-io.md) chapter covers the full workflow.

## Next Steps

- See the [GUI walkthrough](./gui-walkthrough.md) for interactive analysis
- Use the [MCP server](./mcp-server.md) for local AI-agent assisted workflows
- Browse the [Python API reference](./python-api.md)
- Read [Data I/O and NeXus/TOF](./data-io.md) for TIFF, NeXus, and TOF data handling
- Explore the [Architecture](./architecture.md) chapter for the crate structure
- Browse the [API Reference](api/nereids_pipeline/) for the full Rust docs
