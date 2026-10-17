# MCP Server

NEREIDS can run as a local Model Context Protocol (MCP) server so an AI
agent can inspect neutron-resonance inputs, validate a dataset manifest, and
launch spectrum or density-map fitting through the Python bindings. The MCP
server is intended for local agent orchestration, for example a user prompt
such as "help me process the data here" in a directory that contains a NEREIDS
manifest.

The MCP interface is experimental. It is useful for demos and agent-assisted
workflows, but the Python and Rust APIs remain the stable interfaces for
scripted analysis.

See also: [Data I/O and NeXus/TOF](./data-io.md) for the raw-Python NeXus,
TIFF, normalization, and TOF energy-grid flow that the MCP workflow wraps.

## Installation

Install the optional MCP dependency and run the stdio server:

```bash
pip install "nereids[mcp]"
nereids-mcp
```

From a source checkout, build the Python extension first:

```bash
pip install maturin
maturin develop --release -m bindings/python/Cargo.toml
pip install "fastmcp>=3.0"
python -m nereids.mcp
```

In the repository, use `pixi run -e mcp test-mcp` to build and test the isolated
MCP environment. `fastmcp` is kept out of the default environment because it
can conflict with conda metadata packages.

### Behavior when `fastmcp` is not installed

The two entry points differ on purpose:

| Entry point | Without `fastmcp` |
|-------------|-------------------|
| `nereids.mcp.main()` (called by the `nereids-mcp` console script) | Raises `ImportError("fastmcp is required for the MCP server. Install it with: pip install nereids[mcp]")`. |
| `nereids.mcp.mcp` (lazy attribute) | Raises `AttributeError` with the same install instruction embedded in the message. |

The attribute path raises `AttributeError` rather than `ImportError` so that
attribute-walking tools (pdoc, IDE introspection, `hasattr`) can treat `mcp`
as absent rather than crashing. Callers who explicitly catch `ImportError`
to detect a "MCP not installed" state should switch to either calling
`nereids.mcp.main()` (which still raises `ImportError`) or using
`hasattr(nereids.mcp, "mcp")` as the feature-detection probe.

## Client Configuration

An MCP client can launch the server with the `nereids-mcp` console script:

```json
{
  "mcpServers": {
    "nereids": {
      "command": "nereids-mcp"
    }
  }
}
```

## Tools

The server exposes two groups of tools.

Low-level physics tools operate on an in-memory isotope registry:

- `list_isotopes(z)` lists naturally occurring isotopes.
- `load_endf(isotope, library="endf8.1")` loads resonance data such as
  `"U-238"` or `"Fe-56"` into the registry.
- `get_resonance_parameters(isotope)` returns loaded resonance metadata.
- `compute_cross_sections(...)`, `compute_transmission(...)`, and
  `forward_model(...)` run direct physics calculations.

Workflow tools operate on a dataset directory or a manifest path:

- `extract_resonance_manifest(dataset_path)` reads the manifest and returns
  parsed metadata.
- `validate_resonance_dataset(dataset_path)` checks required paths, isotope
  entries, and resolution configuration.
- `process_resonance_dataset(dataset_path, output_dir=None, max_pixels=None,
  dry_run=False)` runs the analysis and writes compact result artifacts.

## Dataset Manifests

The workflow tools look for one of these files in the dataset directory:

- `manifest_intermediate.md`
- `smcp_manifest.md`
- `nereids_manifest.md`
- `nereids_mcp.json`
- `analysis.json`

Markdown manifests must contain JSON-compatible frontmatter between `---`
delimiters. Pure JSON manifests must contain the frontmatter object directly.
The workflow configuration may be placed under `analysis`, `workflow`, or
`processing`; otherwise the root object is treated as the workflow.

Minimal counts manifest:

```markdown
---
{
  "name": "u238-counts-demo",
  "tool": "nereids",
  "analysis": {
    "mode": "single_spectrum",
    "data": {
      "kind": "counts_npz",
      "path": "counts.npz",
      "time_edges_key": "time_edges_us"
    },
    "calibration": {
      "t0_us": 3.0,
      "flight_path_m": 25.0,
      "pulse": {
        "alpha0": 0.35,
        "alpha1": 0.05,
        "beta0": 0.25,
        "beta1": 0.0,
        "r": 0.15,
        "fwhm_squared_us2": 0.0,
        "energy_span_ev": [1.0, 200.0]
      }
    },
    "isotopes": [
      {
        "isotope": "U-238",
        "initial_density": 0.001,
        "library": "endf8.1"
      }
    ],
    "fit": {
      "fit_domain": "counts",
      "normalization": 1.0,
      "background": [0.0, 0.0, 0.0],
      "temperature_k": 293.6
    },
    "output": {
      "directory": "output"
    }
  }
}
---
```

The fit is always the counts domain: fit_domain must be "counts". An
instrument resolution block is rejected, because the counts fit uses the
Ikeda-Carpenter pulse in the calibration block, not a tabulated resolution.
Each fitted quantity is a number (known) or an object with a kind
("known", "fitted", "within" or "measured") and the matching fields, so a
manifest can free or fix the temperature, a background term, the
normalization, t0, the flight path and each pulse number. The pulse's
"r_high" is the storage fraction at the high end of its energy span, linear in
sqrt(E) from "r" at the low end; without it the storage fraction is "r"
throughout. Its "detector_tail", [f, tau_D], records a fraction f of the counts
late by an exponential delay of mean tau_D microseconds; without it no count is
late.
Isotope densities are fitted from their `initial_density` values. All
isotopes in a material must use the same ENDF library; its library also
determines which resonance windows the fit excludes.
Unsupported fit settings are rejected. Removed solver or parameter options
cannot silently change the requested fit.

## Supported Workflow Inputs

The fit is counts-only.

single_spectrum fits one region. It supports:

- counts_npz or counts: a .npz file with 1D or 3D sample_counts and
  open_beam_counts, plus the original time-bin edges in microseconds. The
  edges come from time_edges_us (or time_edges_key) and must increase in
  the counts' order. Energy centers cannot reconstruct detector bin edges.

For a single spectrum, `fit.time_background` supplies `[C, A, tau]`, each a
number or Value specification, and `fit.gamma_transmission` supplies the
sample's gamma transmission. The additive time term enters after the pulse.
It is distinct from the beam-borne `fit.background` scattering terms,
`[b0, b1, b2]`, `[b0, b1, b2, d, f]` or an object with `b0`, `b1`, `b2` and
optional `d`, `f`.
`fit.room_background` supplies `{rate, frames}`: counts per microsecond per
pixel per frame in both runs, and the `[open-beam, sample]` frames.
`fit.thickness_variance` supplies the relative variance of the sample's areal
density over the pixels a fit sums, `fit.capture_gammas` the share of the
sample's captures whose gammas the detector records, and `fit.in_scatter` the
share of the sample's forward single scatters the detector records.
`fit.debye_temperature_k` broadens the sample's cross sections by a harmonic
lattice with Debye's spectrum at that Debye temperature, and makes the fitted
temperature the thermodynamic one; without it they are broadened by the free
gas.

density_map and spatial_map fit every patch of a counts cube. They support
counts_npz/counts and nexus_histogram/nexus with 3D counts. Optional .npz
arrays excluded, sample and empty are (row, column) boolean masks; without
them every pixel is treated as behind the sample. binning sets the patch side
in pixels.

Map workflows take `fit.time_background` and `fit.room_background` per
pixel, shared by every patch, and `fit.gamma_transmission` for every sample
patch. The map's `identified`
array marks the patches whose fitted densities, unless 0, and fitted
temperature each have a finite standard deviation; it is not physical or
thermometry validation. Masks are cropped with the counts when an ROI or
pixel limit is selected.

Paired arrays must have matching shapes, and the time-bin edge count must be
one more than the first axis of the data arrays. For NeXus inputs the sample
and open-beam TOF edges must match.

Example NeXus density-map manifest:

```markdown
---
{
  "name": "venus-nexus-density-map",
  "tool": "nereids",
  "analysis": {
    "mode": "density_map",
    "data": {
      "kind": "nexus",
      "sample_path": "sample.nxs",
      "open_beam_path": "open_beam.nxs",
      "binning": 4
    },
    "calibration": {
      "t0_us": {"kind": "fitted", "value": 3.0},
      "flight_path_m": 25.0,
      "pulse": {
        "alpha0": 0.35,
        "alpha1": 0.05,
        "beta0": 0.25,
        "beta1": 0.0,
        "r": 0.15,
        "fwhm_squared_us2": 0.0,
        "energy_span_ev": [1.0, 200.0]
      }
    },
    "isotopes": [
      {"isotope": "U-238", "initial_density": 0.001}
    ],
    "fit": {
      "fit_domain": "counts",
      "normalization": {"kind": "fitted", "value": 1.0},
      "temperature_k": {"kind": "fitted", "value": 293.6}
    }
  }
}
---
```

An active instrument resolution is not part of this workflow.

## Result Files

`process_resonance_dataset(...)` writes outputs under the configured output
directory, or under `<dataset>/output` by default.

For `single_spectrum`, it writes:

- `nereids_spectrum_fit.npz`
- `nereids_mcp_result.json`

For `density_map` or `spatial_map`, it writes:

- `nereids_density_map.npz`
- `nereids_mcp_result.json`

The JSON summary is strict JSON: non-finite fit values are represented as
`null`, not `NaN` or `Infinity`.
