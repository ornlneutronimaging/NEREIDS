# Data I/O and NeXus/TOF

This page documents the Python-facing TIFF, NeXus and TOF behavior outside
the MCP-specific workflow page.

## Axis Convention

NEREIDS uses the spectral axis first:

```text
(n_energy_or_tof, height, width)
```

This applies to TIFF stacks, NeXus counts, and the raw count arrays passed
to the counts fit.

For single spectra, use 1D arrays with shape `(n_energy,)`.

## TIFF Stacks

Use `load_tiff_stack(...)` for a multi-frame TIFF:

```python
import nereids

transmission = nereids.load_tiff_stack("transmission_stack.tif", pixel_policy="allow")
# transmission.shape == (n_frames, height, width)
```

Use `load_tiff_folder(...)` for a directory of single-frame TIFF files:

```python
counts = nereids.load_tiff_folder("frames", pattern="frame_*.tif")
```

Folders that do not follow the chunked VENUS naming convention below are
sorted lexicographically by filename. Use zero-padded names such as
`frame_0001.tif`, `frame_0002.tif`, and so on. The optional `pattern`
matches filenames, not full paths, and supports `*` and `?`.

Note: load-time coverage/thickness masking is deliberately not part of the
loaders. Coverage and sample thickness are model concerns handled downstream,
not I/O concerns.

### Chunked VENUS folders

The VENUS DAQ sometimes splits one run into several chunks, each covering
the full TOF frame range, with files named `<prefix>_<chunk>_<frame>.tif`
(for example `run_764_00042.tif`). When every filename in the folder follows
this convention with a single common prefix, `load_tiff_folder(...)` detects
the layout automatically:

- frames are ordered by numeric frame index (identical to lexicographic
  order for zero-padded names, and correct where `_10` would sort before
  `_2` lexicographically);
- chunks covering identical frame ranges are summed element-wise into a
  single `(n_frames, height, width)` stack — the physical stack is the sum,
  not a concatenation. Pass `sum_chunks=False` for the legacy lexicographic
  concatenation. The flag only affects folders with two or more chunks:
  single-chunk (and non-chunk-patterned) folders load identically either
  way — chunk-patterned names in numeric frame order, others
  lexicographically;
- ragged chunks (differing frame counts or frame sets) or duplicate
  (chunk, frame) pairs raise `ValueError` on the default summing path —
  never a silent stack or a partial sum. With `sum_chunks=False` there is
  nothing to corrupt, so the same inconsistent folder loads as the
  lexicographic concatenation of every file (frame count = the sum of all
  files) and the irregularity is reported via `chunk_inconsistent` (and a
  `UserWarning`) instead of raising — this is exactly the case you reach for
  `sum_chunks=False` to inspect raw frames.

Folders with two or more distinct prefixes fall back to legacy
lexicographic loading (summing across prefixes would merge different runs);
use `pattern` to select one run, e.g. `pattern="run_764_*"`.

A *mixed* folder — at least one chunk-patterned name alongside files that
do not match (a stray overview TIFF, a misnamed frame) — also falls back
to lexicographic loading, and the Python loader emits a `UserWarning`
counting the non-conforming files (naming up to three); check for stray
TIFFs or use `pattern` to exclude them. The provenance dict reports the
same via `n_unrecognized_files` / `unrecognized_examples`.

Because summing changes the data semantics versus a per-file read, the
Python loader emits a `UserWarning` naming the chunk count and ids (and
the `sum_chunks=False` escape hatch) whenever chunks were summed, and a
`UserWarning` with the clipped-pixel count when `pixel_policy="clip"`
clamped anything. Pass `return_info=True` to get the full provenance as
a second return value:

```python
counts, info = nereids.load_tiff_folder("run_764", return_info=True)
# info == {"n_files": ..., "n_chunks": ..., "chunk_ids": [...],
#          "chunks_summed": ..., "n_clipped_pixels": ...,
#          "n_unrecognized_files": ..., "unrecognized_examples": [...],
#          "chunk_inconsistent": ...}
```

**One acquisition per folder.** The chunk heuristic assumes the folder
holds a single acquisition — the VENUS autoreduce layout, where each run
gets its own directory (verified on IPTS-37432 output; the `<chunk>`
field in real names is a run-ish id, e.g. `..._ob_0_116_00000.tif`). It
cannot distinguish same-prefix sibling *runs* co-located in one folder
from DAQ chunks — such siblings would be summed. When a folder may hold
multiple runs, select one with `pattern` or pass `sum_chunks=False`.

### Pixel-value policy

Raw detector counts are non-negative by construction, so a negative or
non-finite pixel signals file corruption or a signed-type readout bug. Both
TIFF loaders take a `pixel_policy` keyword:

- `"reject"` (default): raise `ValueError` naming the file, frame, flat
  index, and value. For corrupt readout pixels, mask them per acquisition
  with `detect_bad_pixels(...)` instead of relaxing the policy.
- `"clip"`: clamp negative values to `0.0`; NaN still raises (clipping a
  NaN would invent data).
- `"allow"`: accept all values verbatim — required for pre-normalized
  transmission stacks, where noise around zero legitimately produces small
  negative values.

### TOF sidecar (`*_Spectra.txt`)

Autoreduced VENUS folders ship a `<run>_Spectra.txt` sidecar whose first
CSV column is each frame's start time in seconds (one row per TOF frame).
`read_tof_sidecar(...)` converts it to the N+1 ascending microsecond bin
edges that `tof_to_energy_centers(...)` expects, synthesizing the closing
edge from the last frame width. Bin uniformity is not required — MCP
shutter segments change the frame width mid-run.

The start-time = left-bin-edge semantics is established from measured
VENUS autoreduce output (IPTS-37432, OB run 19385): the sidecar holds
exactly one row per TIFF frame, starts at 1.12 µs — not zero; the
autoreduce already drops the pre-trigger bins — in uniform 160 ns steps,
and every time value is an exact integer multiple of the 160 ns bin
width (1.12 µs = 7 × 0.16 µs). Bin *centers* would sit at
half-multiples, so `shutter_time` is definitively the frame start (left
bin edge). Note that PLEIADES's sidecar helper uses these values
directly as frame TOFs, which differs from the true bin centers by half
a bin width; NEREIDS uses edges plus geometric-mean centers. A constant
offset of this kind is absorbed by the fitted t₀ in the energy-scale
fit.

A hand-made sidecar whose first start time is exactly 0 s still parses
(0 is a valid TOF edge), but the t = 0 edge cannot be energy-converted —
E is undefined at t = 0. Crop the first frame from **both** the stack
and the edges (`stack[1:]`, `edges[1:]`) before conversion.

A complete VENUS run folder loads to a stack plus energy axis in three
calls:

```python
counts = nereids.load_tiff_folder("run_764")                # chunk-aware
edges_us = nereids.read_tof_sidecar(
    "run_764/run_764_Spectra.txt",
    n_frames=counts.shape[0],                               # validated
)
energies = nereids.tof_to_energy_centers(edges_us, flight_path_m=25.0)
```

## Counts

The fit takes raw sample and open-beam counts directly and forms no
transmission. Load both arms with the same shape and keep them on the TOF bin
edges; pass them to the counts fit (`fit_counts` for one spectrum, `fit_map`
for a map of pixel patches). See the [Python API](python-api.md).

## NeXus Histogram Loading

For agent-orchestrated NeXus workflows driven by a manifest, see
[MCP Server](./mcp-server.md). This section covers the raw-Python loader.

Use `probe_nexus(...)` to inspect a file without loading full data:

```python
meta = nereids.probe_nexus("sample.nxs")
print(meta.has_histogram, meta.has_events, meta.flight_path_m)
```

Use `load_nexus_histogram(...)` for pre-histogrammed data:

```python
sample = nereids.load_nexus_histogram("sample.nxs")
open_beam = nereids.load_nexus_histogram("open_beam.nxs")

assert sample.counts.shape[0] == sample.tof_edges_us.shape[0] - 1
```

The loader reads VENUS/rustpix-style histogram data from
`/entry/histogram/counts` and returns:

- `counts`: `float64` array with shape `(n_tof, height, width)`.
- `tof_edges_us`: ascending TOF bin edges in microseconds.
- `flight_path_m`: optional file metadata.
- `dead_pixels`: optional `(height, width)` mask.

Histogram files may contain multiple rotation angles. The default
`multi_angle_mode="error"` rejects those files because silently summing
projection angles loses information. Choose explicitly:

```python
summed = nereids.load_nexus_histogram("scan.nxs", multi_angle_mode="sum")
angle0 = nereids.load_nexus_histogram(
    "scan.nxs",
    multi_angle_mode="select",
    angle_index=0,
)
```

## NeXus Event Loading

Use `load_nexus_events(...)` when event data must be histogrammed at load
time:

```python
events = nereids.load_nexus_events(
    "events.nxs",
    n_bins=2000,
    tof_min_us=10.0,
    tof_max_us=50000.0,
    height=512,
    width=512,
)
```

The event loader reads `/entry/neutrons/event_time_offset`, `/x`, and `/y`,
bins events into a linear TOF grid, and returns the same `NexusData` shape
contract as the histogram loader.

## Run Health (DASlogs)

A run can be paused mid-acquisition or suffer accelerator beam dips; both
silently reduce the effective exposure of the summed stack.
`run_health(...)` summarizes the slow-control logs under `/entry/DASlogs`:

```python
health = nereids.run_health("sample.nxs")
print(health.pause_fraction)     # time-weighted fraction spent paused
print(health.beam_dip_fraction)  # fraction with power < 0.5 * median
print(health.median_power, health.duration_s)
```

DASlogs PVs log *transitions*, not regular samples — a run paused for 90%
of its duration may contain just two pause entries, so entry means are
wrong. All fractions use last-value-held time-weighted integration over
the run window (`/entry/duration` when present, else the latest log
timestamp, a lower bound).

The PV-name defaults are the SNS ones (`pause`, `proton_charge`); other
facilities pass their own names, and the dip threshold is adjustable:

```python
health = nereids.run_health(
    "sample.nxs",
    pause_pv="pause",
    power_pv="proton_charge",
    power_dip_fraction=0.5,
)
```

Absent PVs (or a missing `DASlogs` group) yield `None` fields — absence is
not an error. A PV that is present but malformed (length mismatch,
non-finite entries, decreasing timestamps) raises `ValueError`.

## TOF Edges to Energy Centers

NeXus loaders return counts in ascending TOF order. Neutron energy decreases
as TOF increases, so direct TOF-bin conversion would be descending in energy.
`tof_to_energy_centers(...)` returns ascending energy centers for display and
forward-model checks; the counts fit itself works on TOF bin edges:

```python
flight_path_m = sample.flight_path_m or 25.0
energies = nereids.tof_to_energy_centers(
    sample.tof_edges_us,
    flight_path_m,
    delay_us=0.0,
)
```

When pairing these energies with NeXus counts, keep arrays aligned. The MCP
workflow reverses counts to the same ascending-energy order before fitting.
For direct Python workflows, use this pattern:

```python
# load_nexus_histogram returns counts on ascending TOF bin edges; the counts
# fit consumes those edges and the counts directly.
edges_us = sample.tof_edges_us
sample_counts = sample.counts
open_beam_counts = open_beam.counts
```

If you construct an energy grid yourself, make sure the grid and every
spectral array use the same order and that `energies.shape[0]` matches the
first array dimension.

## Fitting counts

Raw sample and open-beam counts go straight to the counts fit:

```python
fit = nereids.fit_counts(measurement, calibration)      # one spectrum
counts_map = nereids.fit_map(map_measurement, calibration)  # pixel patches
```

Build the `Measurement`/`MapMeasurement` from the TOF edges, the two count
cubes and the material, and the `Calibration` from `t0`, the flight path and
the pulse. See the [Python API](python-api.md) for the constructors.
