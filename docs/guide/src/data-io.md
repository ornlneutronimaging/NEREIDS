# Data I/O and NeXus/TOF

This page documents the Python-facing TIFF, NeXus, normalization, and TOF
energy-grid behavior outside the MCP-specific workflow page.

## Axis Convention

NEREIDS uses the spectral axis first:

```text
(n_energy_or_tof, height, width)
```

This applies to TIFF stacks, NeXus counts, normalized transmission cubes,
uncertainty cubes, and the arrays passed to `from_counts(...)` and
`from_transmission(...)`.

For single spectra, use 1D arrays with shape `(n_energy,)`.

## TIFF Stacks

Use `load_tiff_stack(...)` for a multi-frame TIFF:

```python
import nereids

transmission = nereids.load_tiff_stack("transmission_stack.tif")
# transmission.shape == (n_frames, height, width)
```

Use `load_tiff_folder(...)` for a directory of single-frame TIFF files:

```python
transmission = nereids.load_tiff_folder("frames", pattern="frame_*.tif")
```

Folder frames are sorted lexicographically by filename. Use zero-padded names
such as `frame_0001.tif`, `frame_0002.tif`, and so on. The optional `pattern`
matches filenames, not full paths, and supports `*` and `?`.

## Normalization

Raw sample and open-beam arrays can be normalized to transmission:

```python
transmission, uncertainty = nereids.normalize(
    sample_counts,
    open_beam_counts,
    pc_sample=sample_proton_charge,
    pc_ob=open_beam_proton_charge,
)
```

The formula is:

```text
T = (C_sample / C_open_beam) * (PC_open_beam / PC_sample)
```

`sample_counts` and `open_beam_counts` must have identical shape. Optional
`dark_current` is a 2D `(height, width)` array.

For fitting raw counts directly, prefer `from_counts(...)` or
`fit_counts_spectrum_typed(...)` so the counts-KL dispatch can use the
counts-domain likelihood.

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

## TOF Edges to Energy Centers

NeXus loaders return counts in ascending TOF order. Neutron energy decreases
as TOF increases, so direct TOF-bin conversion would be descending in energy.
`tof_to_energy_centers(...)` returns ascending energy centers suitable for
NEREIDS fitting:

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
energies = nereids.tof_to_energy_centers(sample.tof_edges_us, flight_path_m)

# load_nexus_histogram returns ascending TOF. Reverse axis 0 to align with
# ascending energy centers.
sample_counts = sample.counts[::-1, :, :]
open_beam_counts = open_beam.counts[::-1, :, :]

data = nereids.from_counts(sample_counts, open_beam_counts)
result = nereids.spatial_map_typed(data, energies, [u238], c=charge_ratio)
```

If you construct an energy grid yourself, make sure the grid and every
spectral array use the same order and that `energies.shape[0]` matches the
first array dimension.

## Counts vs Transmission Fitting

Use counts APIs when you have raw sample and open-beam counts:

```python
fit = nereids.fit_counts_spectrum_typed(
    sample_counts_1d,
    open_beam_counts_1d,
    energies,
    [(u238, 0.0005)],
    c=charge_ratio,
)
```

Use transmission APIs when your data is already normalized:

```python
fit = nereids.fit_spectrum_typed(
    transmission_1d,
    uncertainty_1d,
    energies,
    [(u238, 0.0005)],
)
```

For spatial maps, the same distinction is encoded by the input constructor:

```python
counts_data = nereids.from_counts(sample_counts_3d, open_beam_counts_3d)
trans_data = nereids.from_transmission(transmission_3d, uncertainty_3d)
```
