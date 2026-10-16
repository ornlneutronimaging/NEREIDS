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
| `target_spin` | `float` | Target spin from the first evaluable range (parse-and-skip placeholders are skipped). |
| `scattering_radius` | `float` | Effective scattering radius in fm, from the first evaluable range. |
| `l_values` | `list[int]` | Orbital angular momentum values present in the data. |
| `has_unevaluated_ranges` | `bool` | True when the file carries parsed-but-not-evaluated spans (LRF=7 / URR / LRU=0); those spans contribute zero cross-section. |
| `skipped_ranges` | `list[str]` | One description per parsed-but-not-evaluated span. |

### Counts Fit Objects

A counts fit is built from raw open-beam and sample counts binned on shared
time-bin edges (microseconds), not an energy grid.

| Object | Meaning |
|--------|---------|
| Value | One fitted quantity: known, fitted from a start, within bounds, or measured with a Gaussian prior. |
| Material | Isotopes each with an areal density in atoms/barn, and a temperature in K. |
| Region | Both runs' counts for one set of pixels, its background [b0, b1, b2, d, f], its area in pixels, the gamma transmission, and the material. |
| RoomBackground | Counts per microsecond per pixel per frame that follow each run's duration, and each run's frames. |
| Spread | The share of each detector cell's counts recorded elsewhere, the spread kernel, and each region's cells. |
| Measurement | The time-bin edges, charge ratio, normalization, time and room backgrounds and regions of one fit. |
| Pulse | Ikeda-Carpenter pulse parameters and the energy span they hold over. |
| Calibration | Timing offset t0 (microseconds), flight path (m), and Pulse. |
| RegionFit | One region's fitted densities, temperature, background, predicted counts and overdispersion. |
| CountsFit | Every region's RegionFit plus the shared normalization, t0, flight path, pulse numbers, covariance and deviance. |
| Patch | What a patch's non-excluded pixels lie behind: sample, mixed, or outside. |
| MapMeasurement | A cube of both runs' counts, the sample/empty/excluded masks, the material and the patch side. |
| CountsMap | Per-patch fits, densities, density standard deviations, temperature and shared covariance. |

NO_TIME_BACKGROUND is a module tuple of three known Values, [0, 0, 1], that
Measurement and MapMeasurement accept as the time background.

#### Value

Value.known(value), Value.fitted(start), Value.within(start, lower, upper) and
Value.measured(value, sd) build the four kinds. The kind property returns
"known", "fitted", "within" or "measured"; value, start, lower, upper and sd
return the matching fields (None where a kind has no such field).

#### Material

Material(isotopes, temperature_k, library="endf8.1", thickness_variance=None)
takes a list of (ResonanceData, Value) pairs, each density in atoms/barn, a
temperature Value in K, and the relative variance v of the areal density over
the region's pixels, known 0 by default: the region transmits
(1 + v*D)**(-1/v), the mean for a Gamma-distributed areal density, D the
optical depth at the mean density.

#### Region

Region(open_counts, sample_counts, pixels, background, gamma_transmission,
material=None, open_live=None, sample_live=None, capture_gammas=None,
in_scatter=None) holds one region. background
is b(E) = b0 + b1/sqrt(E) + b2*sqrt(E) + d*exp(-f/sqrt(E)) as five Values
[b0, b1, b2, d, f], f 0 or more; three Values [b0, b1, b2] hold d and f at 0.
The optional live fractions are per bin in (0, 1]. capture_gammas is the share
epsilon of the sample's captures whose gammas the region records, known 0 by
default; the sample run gains c_q*a*epsilon*phi*(1 - T)*sum(n*sigma_gamma)/D.
in_scatter is the share Omega of the neutrons scattered once in the sample and
leaving through its back face that the region records, known 0 by default;
the sample run gains c_q*a*Omega*phi*sum(n*sigma_s*G), G each isotope's share
of single scatters leaving through the back face at the mean areal density.

#### Measurement

Measurement(time_edges_us, charge_ratio, normalization, regions,
time_background=None, room_background=None, spread=None) holds one open-beam/sample run
pair. time_edges_us are the shared time-bin edges in microseconds (length =
bins + 1), charge_ratio is the sample run's proton charge over the open-beam
run's, and normalization is the sample run's normalization a. time_background
is [C, A, tau] or NO_TIME_BACKGROUND. room_background is
RoomBackground(rate, frames) or None: rate counts per microsecond per pixel
per frame in both runs, not scaled by the charge ratio, the normalization or
the gamma transmission, and frames the (open-beam, sample) frames. spread is
Spread(share, kernel, grid, cells) or None: region r records
(1 - h)*mu_r + h*sum(F_rq*mu_q) of the counts mu arriving in the regions,
F_rq the share of region q's spread counts recorded in region r's cells, for
the kernel's (weight, length in cells) terms
sum(w*exp(-r/s)/(2*pi*s**2)) on a grid of (rows, columns) cells.

#### Pulse

Pulse(alpha0, alpha1, beta0, beta1, r, fwhm_squared_us2, energy_span_ev,
n_tau=600, line_span_ev=None): the Ikeda-Carpenter laws
alpha = alpha0*sqrt(E) + alpha1 and beta = beta0*sqrt(E) + beta1 in
1/microsecond, R constant over energy_span_ev (eV), and the proton triangle
with fwhm_squared_us2 equal to h^2 in microseconds squared. Each parameter is a
Value so it can be held or fitted.

#### Calibration

Calibration(t0_us, flight_path_m, pulse) gives the timing offset in
microseconds and the flight path in metres. Both are Values, known or fitted.

#### RegionFit and CountsFit

RegionFit.densities are the fitted areal densities in atoms/barn,
temperature_k the temperature in K (None without a material), predicted_open
and predicted_sample the counts per bin the fit predicts, overdispersion the
[open-beam, sample] values (None when unmeasured), and excluded_bins the
sample bins the material's library leaves out.

CountsFit.regions holds one RegionFit per region. normalization, t0_us,
flight_path_m, alpha, beta, r, fwhm_squared_us2, time_background and
room_background are the shared fitted values; covariance is the fitted-quantity covariance (None when
the fit did not converge), unbounded_mean and unbounded_covariance the same
Gaussian without the bounds, on_bound the bounds flags, deviance the half
Poisson deviance over the overdispersion, and converged the fit status.
measured_pulls and pulse_consistency are None when unavailable.

#### Patch, MapMeasurement and CountsMap

Patch.kind is "sample", "mixed" or "outside". MapMeasurement(time_edges_us,
open_counts, sample_counts, excluded, sample, empty, material, charge_ratio,
normalization, background, binning=1, empty_background=None, open_live=None,
sample_live=None, time_background=None, gamma_transmission=None,
room_background=None) holds the
(time bin, row, column) counts and the (row, column) masks; its patches()
method returns each patch's kind. time_background is [C, A, tau] per pixel,
shared by every patch, or NO_TIME_BACKGROUND by default; gamma_transmission
is that of every sample patch, known 1 by default, and the empty pixels take 1.

CountsMap exposes the per-patch fits (fits, failed, empty), density maps
densities and density_sd in atoms/barn, temperature_k and temperature_sd_k in
K, the two overdispersion maps, trusted, and the shared normalization,
time_background, room_background, t0_us, flight_path_m, alpha, beta, r, fwhm_squared_us2, shared, shared_covariance,
deviance and converged. n_fitted counts the fitted patches.

trusted records numerical convergence and uncertainty status only: fitted
densities and temperature must have finite standard deviations, except that
a zero density is accepted without one. It does not establish that the
physical model describes the data or that a temperature is accurate.

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

Both loaders evaluate resolved LRF=1/2/3 ranges only. A file with **no**
evaluable range (for example a pure LRF=7 evaluation such as W-182) raises
`ValueError` — loading it would yield zero cross-section everywhere. A mixed
evaluation loads, emits a `UserWarning` naming the skipped spans, and flags
them via `has_unevaluated_ranges` / `skipped_ranges`.

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

`delta_t_us` and `delta_l_m` are **W-parameters**, the width in
`exp(-x^2/W^2)`, not standard deviations.
The conversions are `sigma = W/sqrt(2)` and `FWHM = 2*sqrt(ln 2)*W = 1.6651*W`.
This is SAMMY's convention, shared with the Doppler width, so the two kernels
compose without a conversion.
A measured 1-sigma timing jitter must be converted before it is passed here;
supplying it unconverted gives a kernel 29 % too narrow, and a resolution that
is too narrow is absorbed into a fitted temperature that is too high.

Two helpers do the conversion, so no one has to scale by hand:

| Function | Converts | Formula |
|---|---|---|
| `width_from_sigma(sigma)` | a standard deviation | `W = sigma * sqrt(2)` |
| `width_from_fwhm(fwhm)` | a full width at half maximum | `W = FWHM / (2*sqrt(ln 2))` |

```python
resolution = nereids.resolution_broaden(
    energies,
    cross_sections,
    flight_path_m=25.0,
    delta_t_us=nereids.width_from_sigma(0.5),   # 0.5 us measured 1-sigma
    delta_l_m=nereids.width_from_sigma(2e-3),
)
```

`width_from_fwhm` is also the conversion SAMMY's `Deltag` needs, so a timing
width read from a SAMMY `.inp` can be passed straight through it.

## Counts Fitting

fit_counts(measurement, calibration) fits the areal densities and temperatures
of the regions' materials, each region's background terms and gamma
transmission, the normalization, the time background, and the timing offset,
flight path and pulse numbers that are not known, to the raw counts of both
runs.

```python
import numpy as np
import nereids

u238 = nereids.load_endf(92, 238)
edges = np.linspace(200.0, 400.0, 401)          # microseconds, bins + 1
open_counts = np.array([...])                   # one per bin
sample_counts = np.array([...])

region = nereids.Region(
    open_counts=open_counts,
    sample_counts=sample_counts,
    pixels=1.0,
    background=[nereids.Value.known(0.0)] * 3,
    gamma_transmission=nereids.Value.known(1.0),
    material=nereids.Material(
        isotopes=[(u238, nereids.Value.fitted(0.001))],
        temperature_k=nereids.Value.fitted(293.6),
    ),
)
measurement = nereids.Measurement(
    time_edges_us=edges,
    charge_ratio=1.0,
    normalization=nereids.Value.fitted(1.0),
    regions=[region],
)
pulse = nereids.Pulse(
    alpha0=nereids.Value.known(0.35),
    alpha1=nereids.Value.known(0.05),
    beta0=nereids.Value.known(0.25),
    beta1=nereids.Value.known(0.0),
    r=nereids.Value.known(0.15),
    fwhm_squared_us2=nereids.Value.known(0.0),
    energy_span_ev=(1.0, 200.0),
)
calibration = nereids.Calibration(
    t0_us=nereids.Value.known(3.0),
    flight_path_m=nereids.Value.known(25.0),
    pulse=pulse,
)
result = nereids.fit_counts(measurement, calibration)
print(result.regions[0].densities, result.deviance, result.converged)
```

The measurement is keyed on time-bin edges in microseconds; convert to or from
an energy axis with tof_to_energy and energy_to_tof when needed. Every Value is
either held (known) or fitted, so a fit can free or fix each density,
temperature, background term, normalization, t0, flight path and pulse number
independently. The material's library controls which sample bins are left out
of the likelihood. A tabulated instrument resolution is not part of this
engine: the pulse is the moderator model.

## Counts Mapping

fit_map(map, calibration) fits one material across patches of pixels as one
counts fit, with the empty pixels pinning the normalization. Build the
MapMeasurement from (time bin, row, column) open-beam and sample counts, the
(row, column) excluded/sample/empty masks, the material, and the patch side.

```python
map_measurement = nereids.MapMeasurement(
    time_edges_us=edges,
    open_counts=open_cube,          # (n_bins, height, width)
    sample_counts=sample_cube,      # (n_bins, height, width)
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

CountsMap.densities holds one (patch row, patch column) map per isotope in
atoms/barn; density_sd holds the matching standard deviations. Patches of kind
"mixed" and "outside" are not fitted.

## Pixel Masks

Pixel masks exist **only to exclude pipeline-corrupting pixels** (dead or
hot/railed detector defects).  They are not a data-quality or coverage
filter: low-count pixels are alive and must be kept (the KL-domain fitters
handle them), and coverage/thickness inhomogeneity is a model concern, not a
masking concern.  Downstream, a masked pixel is hard-excluded — never
fitted, `NaN` in the result maps.  Masks are `(height, width)` boolean
arrays, `True` = exclude, and feed directly into
`the excluded mask of a MapMeasurement`.

```python
# Recommended entry point: dead ∪ hot over sample AND open beam.
mask = nereids.detect_bad_pixels(sample, open_beam=open_beam)

# Individual criteria:
dead = nereids.detect_dead_pixels(sample)             # exactly zero in every TOF bin
hot  = nereids.detect_hot_pixels(sample, k_mad=6.0)   # railed/hot point defects
gone = nereids.detect_dead_pixels_chunked([chunk_a, chunk_b])  # intermittent deadness

excluded = mask          # pass as MapMeasurement(excluded=...)
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
--examples` (see `crates/nereids-pipeline/src/counts_fit.rs`).
