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

## Fit a Measured Spectrum

```python
# Simulate noisy measurement
sigma = np.full_like(transmission, 0.01)
noise = np.random.normal(0, 0.01, size=transmission.shape)
measured = np.clip(transmission + noise, 0, 1)

# Fit to recover the density
result = nereids.fit_spectrum(
    measured, sigma, energies, [u238],
    temperature_k=300.0,
    flight_path_m=25.0,
    delta_t_us=5.0,
    delta_l_m=0.005,
)

print(f"Fitted density: {result.densities[0]:.6f} atoms/barn")
print(f"Uncertainty:    {result.uncertainties[0]:.6f}")
print(f"Reduced chi^2:  {result.reduced_chi_squared:.3f}")
print(f"Converged:      {result.converged}")
```

## Spatial Mapping

For imaging data (3D transmission arrays), use `spatial_map`:

```python
# transmission_3d: shape (n_energies, height, width)
# uncertainty_3d:  shape (n_energies, height, width)

result = nereids.spatial_map(
    transmission_3d,
    uncertainty_3d,
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
- Explore the [Architecture](./architecture.md) chapter for the crate structure
- Browse the [API Reference](/api/nereids_pipeline/) for the full Rust docs
