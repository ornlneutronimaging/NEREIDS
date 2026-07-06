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
| `temperature_uncertainty_map` | `NDArray[float64] or None` | Per-pixel 1σ temperature uncertainty (K) when `fit_temperature=True`. **Covariance-only lower bound** on the Poisson-KL / joint-Poisson paths: it captures only statistical curvature (inverse Fisher matrix), omits baseline/model noise, and on real data can underestimate the observed per-superpixel scatter by ~3–4×. Pass `scale_by_chi2=True` for a goodness-of-fit-scaled estimate: σ is multiplied by `sqrt` of the reduced χ² each pixel's result reports (Gaussian `reduced_chi_squared` on the transmission paths, `deviance_per_dof` on the counts joint-Poisson path). This inflates σ for an under-fit pixel (χ²/dof > 1) and, less commonly, shrinks it for an over-fit one (χ²/dof < 1). The flag is a no-op on the already-χ²-scaled LM transmission path. |
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
| `solver="lm"` | `"lm"`, `"kl"`, `"auto"`, `"poisson"`, or `"joint_poisson"`. `"poisson"` and `"joint_poisson"` are aliases used by the counts dispatch and accepted here for symmetry. |
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

For `from_transmission(...)` inputs the default `solver="lm"` and `solver="auto"`
both route to LM (this is the dispatcher contract in `__init__.pyi`:
"`from_transmission + solver="lm"` (default for transmission) → LM"). The
explicit `solver="kl"` opt-in for `from_transmission` runs the legacy
Poisson-NLL-on-transmission path. `density_maps[0]` is the fitted U-238 map.

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
