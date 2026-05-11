# Python API Reference

The `nereids` Python package is a PyO3 layer over the Rust pipeline. This
page gives Python users a browsable reference for the typed API exposed by the
package and is maintained alongside the shipped `nereids/__init__.pyi` type
stubs.

Install the base package with:

```bash
pip install nereids
```

Optional extras are:

```bash
pip install "nereids[mcp]"  # MCP server console script
pip install "nereids[gui]"  # GUI wheel dependency, when available for your platform
```

## Data Objects

### `ResonanceData`

Returned by `load_endf(...)`, `load_endf_file(...)`, and
`create_resonance_data(...)`.

Important properties:

| Property | Type | Meaning |
|----------|------|---------|
| `z` | `int` | Atomic number. |
| `a` | `int` | Mass number. |
| `awr` | `float` | Atomic weight ratio. |
| `n_resonances` | `int` | Resonance count across parsed ranges. |
| `target_spin` | `float` | Target spin from the first range. |
| `scattering_radius` | `float` | Effective scattering radius in fm. |
| `l_values` | `list[int]` | Orbital angular momentum values present in the data. |

### `FitResult`

Returned by `fit_spectrum_typed(...)` and `fit_counts_spectrum_typed(...)`.

| Property | Type | Meaning |
|----------|------|---------|
| `densities` | `NDArray[float64]` | Fitted areal densities in atoms/barn. |
| `uncertainties` | `NDArray[float64]` | One-sigma density uncertainties; entries may be `NaN` when covariance is unavailable. |
| `reduced_chi_squared` | `float` | Pearson chi-squared per degree of freedom for LM/transmission paths. |
| `deviance_per_dof` | `float or None` | Primary goodness-of-fit for counts-KL fits. |
| `converged` | `bool` | Whether the optimizer converged. |
| `iterations` | `int` | Iteration count. |
| `temperature_k` | `float or None` | Fitted temperature when `fit_temperature=True`. |
| `t0_us`, `l_scale` | `float or None` | Fitted energy-scale parameters when `fit_energy_scale=True`. |

### `InputData`

Opaque typed 3D input for spatial mapping. Create it with:

```python
data = nereids.from_transmission(transmission, uncertainty)
data = nereids.from_counts(sample_counts, open_beam_counts)
```

The spectral axis is always axis 0, so arrays have shape
`(n_energy, height, width)`.

### `SpatialResult`

Returned by `spatial_map_typed(...)`.

| Property | Type | Meaning |
|----------|------|---------|
| `density_maps` | `list[NDArray[float64]]` | One `(height, width)` density map per isotope or isotope group. |
| `uncertainty_maps` | `list[NDArray[float64]]` | Per-pixel density uncertainty maps. |
| `chi_squared_map` | `NDArray[float64]` | Per-pixel reduced chi-squared for LM/transmission paths. |
| `deviance_per_dof_map` | `NDArray[float64] or None` | Primary GOF map for counts-KL spatial fits. |
| `converged_map` | `NDArray[bool_]` | Per-pixel convergence flags. |
| `n_converged`, `n_failed`, `n_total` | `int` | Pixel fit counts. |
| `temperature_map` | `NDArray[float64] or None` | Fitted temperature map when enabled. |
| `t0_us_map`, `l_scale_map` | `NDArray[float64] or None` | Energy-scale maps when enabled. |

### `NexusData`

Returned by `load_nexus_histogram(...)` and `load_nexus_events(...)`.

| Property | Type | Meaning |
|----------|------|---------|
| `counts` | `NDArray[float64]` | Counts cube with shape `(n_tof, height, width)`. |
| `tof_edges_us` | `NDArray[float64]` | TOF bin edges in microseconds, length `n_tof + 1`. |
| `flight_path_m` | `float or None` | Flight path from NeXus metadata when available. |
| `dead_pixels` | `NDArray[bool_] or None` | Dead-pixel mask, `True` means dead. |
| `n_rotation_angles` | `int` | Number of rotation angles in histogram input. |
| `event_total`, `event_kept` | `int or None` | Event loader statistics. |

## ENDF Loading

```python
u238 = nereids.load_endf(92, 238, library="endf8.1")
u238_local = nereids.load_endf_file("examples/data/u238_ex027.endf")
```

`load_endf(...)` fetches and caches evaluated nuclear data. Supported library
names include `endf8.0`, `endf8.1`, `jeff3.3`, `jendl5`, `tendl2023`, and
`cendl3.2`. First use can require network access; cached files are reused
afterwards. `load_endf_file(...)` parses a local ENDF file and does not
download data.

## Forward Modeling

```python
import numpy as np
import nereids

u238 = nereids.load_endf(92, 238)
energies = np.linspace(1.0, 30.0, 2000)

transmission = nereids.forward_model(
    energies,
    [(u238, 0.001)],
    temperature_k=300.0,
    flight_path_m=25.0,
    delta_t_us=5.0,
    delta_l_m=0.005,
)
```

`forward_model(...)` returns a 1D `float64` transmission spectrum on the
input energy grid. Pass either `isotopes=[(ResonanceData, density), ...]` or
`groups=[(IsotopeGroup, density), ...]`, but not both. Gaussian resolution is
enabled by the `flight_path_m`, `delta_t_us`, and `delta_l_m` parameters.
Tabulated resolution can be supplied with `resolution=load_resolution(...)`.

## Single-Spectrum Fitting

### Transmission Data

```python
result = nereids.fit_spectrum_typed(
    transmission,
    uncertainty,
    energies,
    [(u238, 0.0005)],
    temperature_k=300.0,
    solver="lm",
)
```

Shape contract:

- `transmission`, `uncertainty`, and `energies` are 1D arrays with the same
  length.
- `energies` is in eV and should be ascending.
- `isotopes` supplies `(ResonanceData, initial_density)` pairs.

Keyword arguments:

| Option | Meaning |
|--------|---------|
| `temperature_k=293.6` | Sample temperature in kelvin. |
| `fit_temperature=False` | Fit sample temperature in addition to densities. |
| `max_iter=200` | Maximum optimizer iterations. |
| `solver="lm"` | `"lm"`, `"kl"`, or `"auto"` for transmission data. |
| `background=False` | Enable SAMMY-style transmission background parameters. |
| `fit_back_d=False`, `fit_back_f=False` | Fit optional exponential background terms. |
| `back_d_init=0.01`, `back_f_init=1.0` | Initial exponential background values. |
| `fit_energy_scale=False` | Fit TOF energy-scale parameters `t0_us` and `l_scale`. |
| `t0_init_us=0.0`, `l_scale_init=1.0` | Initial energy-scale values. |
| `energy_scale_flight_path_m=25.0` | Nominal flight path for energy-scale fitting. |
| `resolution=...` | Tabulated resolution from `load_resolution(...)`. |
| `flight_path_m=...`, `delta_t_us=...`, `delta_l_m=...` | Gaussian resolution parameters. |
| `fit_energy_range=(emin, emax)` | Restrict the cost function to an energy window. |
| `groups=[...]` | Fit isotope groups instead of individual isotopes. |
| `initial_densities=[...]` | Initial density guesses when fitting groups. |
| `tzero_jacobian="..."` | Select the TZERO Jacobian implementation. |

### Raw Counts

```python
result = nereids.fit_counts_spectrum_typed(
    sample_counts,
    open_beam_counts,
    energies,
    [(u238, 0.0005)],
    solver="auto",
    c=1.0,
)
```

`solver="auto"`, `"kl"`, `"poisson"`, and `"joint_poisson"` all route counts
data to the counts-KL dispatch. Use `c=Q_s / Q_ob` when sample and open-beam
counts have different proton charge or dwell-time normalization. The primary
GOF for this path is `FitResult.deviance_per_dof`.

Counts fitting accepts the same temperature, background, group, resolution,
energy-scale, and `fit_energy_range` options as transmission fitting. Counts
specific options are:

| Option | Meaning |
|--------|---------|
| `detector_background=...` | Optional 1D detector background spectrum; required when `fit_alpha_2=True`. |
| `fit_alpha_1=False`, `fit_alpha_2=False` | Fit counts-domain nuisance/background terms. |
| `alpha_1_init=1.0`, `alpha_2_init=1.0` | Initial nuisance/background values. |
| `c=1.0` | Proton-charge ratio `Q_s / Q_ob`. |
| `enable_polish=True/False/None` | Override counts-KL polish behavior; `None` uses the dispatcher default. |

## Spatial Mapping

### Pre-Normalized Transmission Cubes

```python
data = nereids.from_transmission(transmission_3d, uncertainty_3d)
result = nereids.spatial_map_typed(
    data,
    energies,
    [u238],
    initial_densities=[0.0005],
    solver="auto",
)
```

`solver="auto"` uses LM for `from_transmission(...)` data. `density_maps[0]`
is the fitted U-238 map.

### Raw Count Cubes

```python
data = nereids.from_counts(sample_counts_3d, open_beam_counts_3d)
result = nereids.spatial_map_typed(
    data,
    energies,
    [u238],
    initial_densities=[0.0005],
    solver="auto",
    c=1.0,
)
```

`solver="auto"` uses counts-KL for `from_counts(...)` data and populates
`deviance_per_dof_map`.

Shape contract:

- `sample_counts_3d`, `open_beam_counts_3d`, `transmission_3d`, and
  `uncertainty_3d` use shape `(n_energy, height, width)`.
- `energies.shape == (n_energy,)`.
- `dead_pixels`, when supplied, uses shape `(height, width)` with `True`
  marking pixels to skip.

Keyword arguments:

| Option | Meaning |
|--------|---------|
| `temperature_k=293.6`, `fit_temperature=False` | Fixed or fitted sample temperature. |
| `initial_densities=[...]` | Initial density guesses. |
| `dead_pixels=...` | `(height, width)` skip mask. |
| `max_iter=200` | Maximum per-pixel optimizer iterations. |
| `solver="auto"` | Dispatch from input type unless explicitly set. |
| `background=False` | Enable SAMMY-style background for LM/transmission paths. |
| `fit_alpha_1=False`, `fit_alpha_2=False` | Fit counts-domain nuisance/background terms. |
| `alpha_1_init=1.0`, `alpha_2_init=1.0` | Initial nuisance/background values. |
| `c=1.0` | Proton-charge ratio for counts-KL spatial fitting. |
| `enable_polish=True/False/None` | Override counts-KL polish behavior; `None` auto-disables polish for multi-pixel maps. |
| `fit_energy_scale=False` | Fit per-pixel `t0_us` and `l_scale` maps. |
| `t0_init_us=0.0`, `l_scale_init=1.0` | Initial energy-scale values. |
| `energy_scale_flight_path_m=25.0` | Nominal flight path for energy-scale fitting. |
| `resolution=...` | Tabulated resolution from `load_resolution(...)`. |
| `flight_path_m=...`, `delta_t_us=...`, `delta_l_m=...` | Gaussian resolution parameters. |
| `groups=[...]` | Fit isotope groups instead of individual isotopes. |
| `tzero_jacobian="..."` | Select the TZERO Jacobian implementation. |
| `fit_energy_range=(emin, emax)` | Restrict the cost function to an energy window. |

## TIFF and NeXus I/O

```python
stack = nereids.load_tiff_stack("transmission_stack.tif")
folder_stack = nereids.load_tiff_folder("frames", pattern="frame_*.tif")

sample = nereids.load_nexus_histogram("sample.nxs")
open_beam = nereids.load_nexus_histogram("open_beam.nxs")
energies = nereids.tof_to_energy_centers(
    sample.tof_edges_us,
    sample.flight_path_m or 25.0,
)
```

See [Data I/O and NeXus/TOF](./data-io.md) for ordering and pairing rules.

## Element and Utility APIs

```python
nereids.element_symbol(92)        # "U"
nereids.element_name(92)          # "Uranium"
nereids.parse_isotope_str("U-238") # (92, 238)
nereids.natural_abundance(92, 238)
nereids.natural_isotopes(26)
nereids.tof_to_energy(tof_us, flight_path_m)
nereids.energy_to_tof(energy_ev, flight_path_m)
```

## API Generation Status

The published docs site currently renders this mdBook Python reference plus
the generated Rust API under `api/`. The Python page is maintained from the
PyO3 docstrings and the PEP 561 stub file at
`bindings/python/python/nereids/__init__.pyi`.

Before release, run:

```bash
pixi run doc-guide
pixi run doc-build
```

`doc-build` builds the mdBook guide, builds Rustdoc, and copies Rustdoc into
`target/book/api`. It does not execute notebooks or compile mdBook Rust
snippets.
