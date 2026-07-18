# Python API Reference

The `nereids` Python package is a PyO3 layer over the Rust pipeline. This
page is a curated narrative reference covering the typed APIs Python users
reach for most often, with argument tables, array-shape contracts, and
dispatch rules.

For the exhaustive auto-generated reference (every function, class, and
attribute exported by the package), see the **[generated Python API
reference](python/index.html)** built by [pdoc](https://pdoc.dev) from the
installed wheel and the shipped `nereids/__init__.pyi` type stubs.

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
| `prediction_energies_ev` | `NDArray[float64]` | Energy coordinate for the returned fitted curve, in measured-bin order. |
| `signal_prediction` | `NDArray[float64]` | Fitted signal after response and normalization, excluding the explicit additive background. |
| `background_prediction` | `NDArray[float64]` | Explicit additive SAMMY background contribution. |
| `model_prediction` | `NDArray[float64]` | Complete fitted curve; exactly `signal_prediction + background_prediction`. |

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
| `temperature_uncertainty_map` | `NDArray[float64] or None` | Per-pixel 1σ temperature uncertainty (K) when `fit_temperature=True`. On the raw-count joint-Poisson path this is a **covariance-only lower bound**: it captures only statistical curvature (inverse Fisher matrix) and omits baseline/model noise. Pass `scale_by_chi2=True` to multiply it by the square root of `deviance_per_dof`. The flag is a no-op on the LM transmission path, which already scales covariance by reduced chi-squared. |
| `anorm_map`, `background_maps` | `NDArray[float64] / list[...] or None` | SAMMY `Anorm` and the polynomial background `[BackA, BackB, BackC]` per pixel when `background=True`. |
| `back_d_map`, `back_f_map` | `NDArray[float64] or None` | SAMMY exponential background `BackD` / `BackF` per pixel when `background=True` and `fit_back_d=True` / `fit_back_f=True`. Counts-KL spatial runs always return `None` for both (the joint-Poisson dispatch never fits the exponential tail). |
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

### Exact Two-Arm Count Response

```python
open_signal, sample_signal = nereids.two_arm_count_response(
    true_energies_ev,
    incident_fluence_weights,
    transmission,
    detector_time_edges_us,
    resolution,
)
```

`two_arm_count_response(...)` applies the instrument response separately to
the incident spectrum and to the attenuated sample spectrum. It returns the
numerator and denominator needed to form the physical ratio
`response(flux * transmission) / response(flux)` without broadening a
transmission ratio directly. The fluence array must already include the
energy-integration weights. The result uses detector-time bins; probability
falling outside the supplied acquisition window is not silently moved back
into it. Both arrays are predictions for the same reference exposure. Scale
them to the measured proton charge, live time, or other documented exposure of
each complete acquisition before comparing them with observed counts.

For repeated work, build `DetectorResponseMatrix(...)` once. Its `project(...)`
and `apply(...)` methods reuse the compact native response, while
`transpose_project(...)` supplies the reverse operation needed by source
inference. `collapse_true_energy_groups(...)` pre-sums fixed quadrature points
inside each source cell. It changes neither the quadrature nor any response
probability; it avoids repeating the same sum during every source-optimizer
step.

## Aggregate 1D Room Calibration and Frozen Hot Fit

Use `calibrate_aggregated_1d(...)` for the room run and pass its result directly
to `fit_frozen_aggregated_1d(...)` for the hot run. The room function accepts no
hot data, so the instrument cannot be adjusted to improve the hot result.

```python
calibration = nereids.calibrate_aggregated_1d(
    detector_time_edges_us=detector_edges,
    open_counts=open_counts,
    room_counts=room_counts,
    sample_over_open_exposure=room_charge / open_charge,
    isotopes=[ta181],
    room_temperature_k=room_temperature,
    reference_flight_path_m=reference_path,
    reference_timing_offset_us=reference_t0,
    initial_physical_parameters=(t0_start, path_start, width_start, density_start),
    fit_energy_range_ev=(energy_min, energy_max),
    ic_profile=instrument_ic_profile,
    physical_lower_bounds=(t0_min, path_min, width_min, density_min),
    physical_upper_bounds=(t0_max, path_max, width_max, density_max),
    debye_temperature_k=217.0,
)

hot = nereids.fit_frozen_aggregated_1d(
    calibration,
    hot_counts=hot_counts,
    sample_over_open_exposure=hot_charge / open_charge,
    initial_temperature_k=1000.0,
    initial_density_atoms_per_barn=density_start,
)
```

The supported calculation is:

1. Integrate the analytical IC response over the actual detector-time edges
   with a fixed 15-point rule inside every true-energy cell.
2. Infer one nonnegative incident-source amount per cell from the open beam
   only.
3. Fit room timing offset, effective path, IC width scale, and material amount.
   At each physical trial, fit the four SAMMY apparent-transmission background
   coefficients inside that trial.
4. Freeze the room response, source, energy interval, nuclear evaluation,
   material model, and open-beam uncertainty.
5. Fit only hot temperature, material amount, and the same four background
   coefficients.

`Aggregated1DFitResult` returns measured transmission, uncertainty, signal,
background, complete model, residual, every fitted parameter, bound hits, and
summary residual values. It returns both `poisson_uncertainty` /
`poisson_residual` (independent count statistics) and `uncertainty` / `residual`
(including any independently measured detector variance factors supplied by
the caller). A notebook must not rebuild the background or fitted curve. The
measured arrays and evaluated nuclear values are never modified. Variance
factors must come from detector data independent of the spectrum fit; they must
not be adjusted to improve a residual.

The IC profile, fit-energy range, and physical bounds are required arguments.
There are no hidden VENUS geometry or fit-window defaults in the generic API.
The VENUS example uses a 4--120 eV true-energy response interval so neutrons
outside the fit window can still migrate into measured bins, while its 8--45
eV calibration interval is the pre-existing tantalum analysis region fixed
before the IC comparison. Its JSON and figure report all supplied 4--120 eV
bins as diagnostics as well as the bins that determine the fit.

The public VENUS example is
`examples/workflows/venus_aggregated_1d.py`. It uses scatter points for the
measured spectra and lines for the fitted curves. The fixed
`VENUS_UDR_MATCHED_IC_PROFILE` is an analytical approximation to the archived
VENUS UDR shape; it is not a new nuclear evaluation and is not claimed to be
more faithful than the high-fidelity UDR itself. The example's TPX1 variance
factors were measured from the raw pixels for those exact acquisitions and are
reported alongside the independent-Poisson residuals.

Current boundary: this workflow is for one aggregated spectrum. Exact-response
spatial fitting remains unsupported until one frozen detector response can be
reused safely across pixels. The aggregate calibration is also not an
interactive operation; source inference and repeated response construction are
still the main runtime costs.

### Independently Measured Count Backgrounds

```python
background_fit = nereids.fit_two_arm_background_templates(
    observed_open_counts,
    observed_sample_counts,
    open_signal,
    sample_signal,
    open_exposure_scale,
    sample_exposure_scale,
    ["blocked_beam"],
    open_background_templates,
    sample_background_templates,
    initial_amplitudes,
)
```

Each row of the two template matrices is a fixed detector-bin shape from an
independent measurement such as blocked-beam or detector-only data. NEREIDS
fits only a non-negative amplitude and returns all three pieces for each arm:
the fixed neutron signal, the fitted background, and their exact total. This
count background is added after the neutron response and is not the SAMMY
transmission background. The code does not generate a flexible residual curve
or establish template provenance; callers must retain the independent source
record. `open_exposure_scale` and `sample_exposure_scale` are required: they
convert the common reference neutron signal to the proton charge, live time, or
other documented exposure of each complete acquisition. A missing exposure
correction can otherwise be falsely absorbed by an additive background.

Background placement is part of the model, not a scripting choice:

| Background source | Required input to this API |
|-------------------|----------------------------|
| Detector dark counts, gamma counts, or a blocked-beam reference measured in detector-time bins | Add after the neutron response as a detector-bin template. |
| Sample scattering calculated as a function of true neutron energy | Pass through the instrument response first; only its resulting detector-bin template is fitted here. |
| Sample scattering measured directly in detector-time bins under independently documented conditions | Add after the neutron response as a detector-bin template. |
| SAMMY-style transmission baseline/background | Use the separate transmission model; it is not accepted as a count template by this API. |

The supported fit matrix is explicit:

| Data | Engine | Decision and background behavior |
|------|--------|----------------------------------|
| Pre-normalized transmission | LM least squares | Supported. `background=True` fits the SAMMY apparent-transmission curve. Resolution on this route is a ratio-level approximation because the separate count arms are unavailable. |
| Pre-normalized transmission | Poisson/KL | Rejected: a fractional ratio is not a Poisson count and its supplied uncertainty would be ignored. |
| Separate open/sample counts | Poisson/KL | Supported. With a detector response, the exact source weights and detector-time edges are required. `background=True` applies the SAMMY curve to measured apparent transmission after the two-arm detector calculation. |
| Separate open/sample counts | LM least squares | Rejected: silently dividing the arms loses the open-beam count uncertainty. |
| Count data with detector/gamma templates | Fixed-signal count-template fit | Supported by `fit_two_arm_background_templates(...)`. The template fit is deliberately separate from nuclear/calibration fitting; a nonzero `detector_background` on the main fitter is rejected rather than misinterpreted. |
| Spatial count cube with detector response | Any | Not yet supported: the spatial API does not carry the exact source/time inputs and does not yet cache one fixed detector matrix for all pixels. |

`TwoArmBackgroundFitResult` reports `names`, `amplitudes`, optional local
`amplitude_uncertainties`, `amplitudes_identifiable`, the
signal/background/total arrays for both arms, `poisson_deviance`,
`deviance_per_dof`, `converged`, and `iterations`. If
`amplitudes_identifiable` is false, two or more supplied shapes cannot be
separated: the summed background and total prediction remain usable, but the
individual named amounts are arbitrary and their uncertainties are returned as
`NaN`.

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
| `solver="lm"` | `"lm"` or `"auto"` for normalized transmission. Poisson/KL names are rejected because transmission ratios are not Poisson counts. |
| `background=False` | Enable SAMMY-style transmission background parameters. |
| `fit_back_d=False`, `fit_back_f=False` | Fit optional exponential background terms. |
| `back_d_init=0.01`, `back_f_init=1.0` | Initial exponential background values. |
| `fit_energy_scale=False` | Fit TOF energy-scale parameters `t0_us` and `l_scale`. |
| `t0_init_us=0.0`, `l_scale_init=1.0` | Initial energy-scale values. |
| `energy_scale_flight_path_m=25.0` | Nominal flight path for energy-scale fitting. |
| `resolution=...` | Tabulated resolution from `load_resolution(...)`. **Mutually exclusive with the Gaussian parameters below** — pass either `resolution=` (tabulated) or the `flight_path_m`/`delta_t_us`/`delta_l_m` trio (Gaussian), never both. |
| `flight_path_m=...`, `delta_t_us=...`, `delta_l_m=...` | Gaussian resolution parameters (mutually exclusive with `resolution=`). |
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

Counts fitting accepts temperature, isotope groups, energy-scale calibration,
and `fit_energy_range` when no detector response is active. Resolved counts use
the exact two-arm model by supplying `resolution`, `incident_fluence_weights`,
and `detector_time_edges_us` together. `energies` is then the true-energy
quadrature and may have a different length from the measured count arrays. The
fit evaluates `response(flux * transmission) / response(flux)`; it never uses
the post-hoc shortcut `response(transmission)`. Energy-scale fitting and an
energy fit window are rejected on this two-axis route until the detector clock
is fitted consistently.

The production counts route still rejects a nonzero `detector_background` and
fitted count nuisance factors. Use the independently measured template API
above for those backgrounds. Counts-specific options are:

| Option | Meaning |
|--------|---------|
| `detector_background=...` | Reserved compatibility input. The production fit rejects nonzero values; use `fit_two_arm_background_templates(...)` for independently supplied detector-bin shapes. |
| `fit_alpha_1=False`, `fit_alpha_2=False` | Research-only flags. The production fit rejects enabling them. |
| `alpha_1_init=1.0`, `alpha_2_init=1.0` | Research-only initial values; they do not enable nuisance fitting in the production route. |
| `c=1.0` | Proton-charge ratio `Q_s / Q_ob`. |
| `resolution=...` | Exact detector-time response (`TabulatedResolution` or `IkedaCarpenter`); requires the two arrays below. |
| `incident_fluence_weights=...` | Incident fluence integrated over the true-energy quadrature. |
| `detector_time_edges_us=...` | Measured detector-time edges; length is `len(sample_counts) + 1`. |
| `timing_offset_us=0.0` | Fixed detector-clock offset used by the response. |
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

For `from_transmission(...)` inputs, `solver="lm"` and `solver="auto"` both
route to LM. Poisson/KL solver names are rejected because the normalized ratio
no longer contains the separate open and sample count arms.
`density_maps[0]` is the fitted U-238 map.

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
| `fit_back_d=False`, `fit_back_f=False` | Fit the SAMMY exponential background tail (`BackD * exp(-BackF / √E)`).  Requires `background=True`.  Per-pixel `back_d_map` / `back_f_map` are populated on the returned `SpatialResult` (issue #538). |
| `back_d_init=0.01`, `back_f_init=1.0` | Initial values for the exponential tail. |
| `fit_alpha_1=False`, `fit_alpha_2=False` | Fit counts-domain nuisance/background terms. |
| `alpha_1_init=1.0`, `alpha_2_init=1.0` | Initial nuisance/background values. |
| `c=1.0` | Proton-charge ratio for counts-KL spatial fitting. |
| `enable_polish=True/False/None` | Override counts-KL polish behavior; `None` auto-disables polish for multi-pixel maps. |
| `fit_energy_scale=False` | Fit per-pixel `t0_us` and `l_scale` maps. |
| `t0_init_us=0.0`, `l_scale_init=1.0` | Initial energy-scale values. |
| `energy_scale_flight_path_m=25.0` | Nominal flight path for energy-scale fitting. |
| `resolution=...` | Tabulated resolution from `load_resolution(...)`. **Mutually exclusive with the Gaussian parameters below** — pass either `resolution=` (tabulated) or the `flight_path_m`/`delta_t_us`/`delta_l_m` trio (Gaussian), never both. |
| `flight_path_m=...`, `delta_t_us=...`, `delta_l_m=...` | Gaussian resolution parameters (mutually exclusive with `resolution=`). |
| `groups=[...]` | Fit isotope groups instead of individual isotopes. |
| `tzero_jacobian="..."` | Select the TZERO Jacobian implementation. |
| `fit_energy_range=(emin, emax)` | Restrict the cost function to an energy window. |

## Pixel Masks

Pixel masks exist **only to exclude pipeline-corrupting pixels** (dead or
hot/railed detector defects).  They are not a data-quality or coverage
filter: low-count pixels are alive and must be kept (the KL-domain fitters
handle them), and coverage/thickness inhomogeneity is a model concern, not a
masking concern.  Downstream, a masked pixel is hard-excluded — never
fitted, `NaN` in the result maps.  Masks are `(height, width)` boolean
arrays, `True` = exclude, and feed directly into
`spatial_map(dead_pixels=...)`.

```python
# Recommended entry point: dead ∪ hot over sample AND open beam.
mask = nereids.detect_bad_pixels(sample, open_beam=open_beam)

# Individual criteria:
dead = nereids.detect_dead_pixels(sample)             # exactly zero in every TOF bin
hot  = nereids.detect_hot_pixels(sample, k_mad=6.0)   # railed/hot point defects
gone = nereids.detect_dead_pixels_chunked([chunk_a, chunk_b])  # intermittent deadness

result = nereids.spatial_map(cube, energies, isotopes, dead_pixels=mask)
```

### `detect_bad_pixels(sample, open_beam=None, hot_k_mad=6.0)`

Union mask `dead(sample) ∪ hot(sample) [∪ dead(ob) ∪ hot(ob)]` — deadness
and hotness are per-acquisition, so a mask built from one stack alone misses
failures in the other.  This is the validating entry point (rejects
non-finite/negative counts and empty TOF axes with `ValueError`).
`hot_k_mad=None` disables the hot screen (dead-only mask).

### `detect_dead_pixels(data)`

Legacy single-stack detector: flags pixels that are exactly `0.0` in every
TOF bin.  Assumes counts already validated finite and non-negative; prefer
`detect_bad_pixels` for new code.

### `detect_hot_pixels(data, k_mad=6.0)`

Two-stage hot/railed screen on **raw counts**: a global robust cut on log
total counts (median + `k_mad` · max(1.4826 · MAD, Poisson floor)) nominates
candidates, and a local 8-neighbor confirmation (≥ 10× the neighbors'
median, iterated to a fixpoint) keeps contiguous bright *scene* regions —
open-beam areas, slit apertures — unmasked while catching isolated point,
line, and small-cluster defects.  Pass raw detected counts, not
proton-charge-normalized rates or transmission ratios: scaling breaks the
Poisson floor.

### `detect_dead_pixels_chunked(chunks)`

Intermittent deadness: a pixel dead for part of the acquisition is invisible
in a summed stack (uniformly reduced counts, no zeros).  Given per-chunk
stacks (e.g. per-run splits or event data re-histogrammed in time windows),
flags pixels all-zero in **any** chunk.  Chunk so that live pixels expect
λ ≥ 20 counts each (false-flag probability ≤ m·e^(−λ)).  Spatial dims must
match across chunks; the TOF axis may differ.

## TIFF and NeXus I/O

```python
stack = nereids.load_tiff_stack("transmission_stack.tif", pixel_policy="allow")
folder_stack, info = nereids.load_tiff_folder(
    "frames",
    pattern="frame_*.tif",
    sum_chunks=True,        # sum chunked VENUS runs element-wise (default)
    pixel_policy="reject",  # "reject" | "clip" | "allow" (default "reject")
    return_info=True,       # also return the load-provenance dict
)
info["n_chunks"]            # DAQ chunks detected (1 if not chunked)
info["chunks_summed"]       # True when they were summed element-wise
info["n_clipped_pixels"]    # pixels clamped under pixel_policy="clip"
# full key set: n_files, n_chunks, chunk_ids, chunks_summed,
# n_clipped_pixels, chunk_inconsistent, n_unrecognized_files,
# unrecognized_examples
edges_us = nereids.read_tof_sidecar(
    "run_764/run_764_Spectra.txt",
    n_frames=folder_stack.shape[0],
)

sample = nereids.load_nexus_histogram("sample.nxs")
open_beam = nereids.load_nexus_histogram("open_beam.nxs")
energies = nereids.tof_to_energy_centers(
    sample.tof_edges_us,
    sample.flight_path_m or 25.0,
)
health = nereids.run_health("sample.nxs")   # RunHealth: pause/beam-dip fractions
```

`load_tiff_folder` detects chunked VENUS folders
(`<prefix>_<chunk>_<frame>.tif`) and sums chunks element-wise by default,
emitting a `UserWarning` naming the summed chunks (and one with the
clipped-pixel count under `pixel_policy="clip"`); `return_info=True`
returns the load provenance as a second value. `read_tof_sidecar`
converts a VENUS `*_Spectra.txt` sidecar (frame **start** times in
seconds — the left bin edges, verified on measured autoreduce output)
into the N+1 ascending microsecond TOF bin edges that
`tof_to_energy_centers` expects. Negative or non-finite pixels are rejected
at load time unless `pixel_policy` says otherwise. `run_health` returns a
`RunHealth` summary of the `/entry/DASlogs` pause and beam-power logs using
last-value-held time-weighted integration (SNS PV-name defaults).

See [Data I/O and NeXus/TOF](./data-io.md) for ordering and pairing rules,
chunk semantics, the pixel-value policy, and run health.

## Beam-State Filtering (DASlogs and Event Banks)

Facility NeXus files record slow-control PVs under `/entry/DASlogs/<pv>` as
**transition logs**: each value takes effect at its timestamp and persists
until the next entry.  Averaging the value array directly is wrong whenever
entries are unevenly spaced — on a real VENUS run the entry-mean of the
`pause` log read 0.43 while the time-weighted pause fraction was 0.90.

### `read_run_log(path, pv)`

Returns a `RunLog` with `times` (seconds since run start), `values`,
`duration_s`, `offset_iso` (ISO-8601 epoch of the clock), and
`n_dropped_corrupt` — the number of corrupt device-reconnect records
(backward time jumps or subnormal garbage payloads, both seen in real SNS
files) dropped from the log.

### `intervals_where(times, values, duration_s, min_value=None, max_value=None)`

Derives `(t_start, t_end)` intervals where the PV satisfies the bounds,
under correct step-function semantics: the last value persists to
`duration_s` (padded one f32 ULP — SNS records duration in float32 while
pulse times are float64, and the final pulse of about half of real runs is
stamped just beyond it), time before the first entry never matches, `NaN`
never matches, and adjacent segments merge.

### `intervals_intersect(a, b)`

Composes conditions across PVs (e.g. not-paused AND beam power above
threshold).  Inputs are validated and normalised (sorted, merged).

### `load_nexus_bank_spectrum(path, bank, n_bins, tof_min_us, tof_max_us, keep_intervals=None)`

Loads one NXevent_data bank (e.g. `"monitor1"`) as a `BankSpectrum` — a 1-D
TOF spectrum (`tof_edges_us`, `counts`) with retention statistics
(`pulses_total`/`pulses_kept`, `events_total`/`events_kept`, drop counters,
`pulse_time_offset_iso`).  With `keep_intervals`, only pulses whose
`event_time_zero` falls inside the intervals (half-open, DASlogs clock)
are histogrammed.  `units` attributes are required on both event datasets
(never guess a scale); a bank with zero events loads gracefully to a zero
spectrum — on VENUS every imaging-detector bank is empty because tpx1 is
frame-mode, and only monitors carry events.

```python
pause = nereids.read_run_log("run.nxs.h5", "pause")
live = nereids.intervals_where(
    pause.times, pause.values, pause.duration_s, max_value=0.5
)
power = nereids.read_run_log("run.nxs.h5", "BL10:Det:rtdl:BeamPowerAvg")
stable = nereids.intervals_where(
    power.times, power.values, power.duration_s, min_value=1.5
)
keep = nereids.intervals_intersect(live, stable)

mon = nereids.load_nexus_bank_spectrum(
    "run.nxs.h5", "monitor1",
    n_bins=500, tof_min_us=0.0, tof_max_us=16667.0,
    keep_intervals=keep,
)
print(mon.pulses_kept, "/", mon.pulses_total, "pulses in stable beam")
```

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

## How This Page Is Generated

The published docs site renders three things side by side:

| Site path | Source | What it shows |
|-----------|--------|---------------|
| `/` (this page) | Hand-maintained `docs/guide/src/python-api.md` | Curated narrative tour of the typed APIs |
| `/python/` | [pdoc](https://pdoc.dev) over the installed `nereids` wheel and `nereids/__init__.pyi` stubs | Auto-generated exhaustive reference |
| `/api/` | `cargo doc` (rustdoc) | Rust crate API reference |

To rebuild the whole site locally:

```bash
pixi run doc-build   # depends on: doc-guide, doc-api, doc-python
pixi run doc         # serves target/book/ at http://localhost:8000
```

`doc-python` invokes `pdoc -o target/book/python --no-show-source nereids`
after `pixi run build` has produced an importable wheel. Whenever
`bindings/python/python/nereids/__init__.pyi` or PyO3 docstrings in
`bindings/python/src/lib.rs` change, both the auto-generated `python/`
reference and any affected sections of this curated page should be reviewed
in the same PR.

This page does not execute notebooks or compile-test Python snippets. The
Rust quickstart on this site IS compile-tested by `cargo check --workspace
--examples` (see `crates/nereids-fitting/examples/quickstart.rs`).
