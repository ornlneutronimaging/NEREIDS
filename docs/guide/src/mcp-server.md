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

In the repository Pixi environment, `fastmcp` is intentionally not a default
dependency because it can conflict with conda metadata packages. Install it
manually in the environment when MCP support is needed.

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
- `compute_cross_sections(...)`, `compute_transmission(...)`,
  `forward_model(...)`, and `detect_isotopes(...)` run direct physics
  calculations.

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

Minimal single-spectrum manifest:

```markdown
---
{
  "name": "u238-spectrum-demo",
  "tool": "nereids",
  "physics": "neutron-resonance",
  "analysis": {
    "mode": "single_spectrum",
    "data": {
      "kind": "transmission_npz",
      "path": "spectrum.npz"
    },
    "isotopes": [
      {
        "isotope": "U-238",
        "initial_density": 0.001,
        "library": "endf8.1"
      }
    ],
    "fit": {
      "solver": "lm",
      "max_iter": 100
    },
    "resolution": {
      "kind": "gaussian",
      "flight_path_m": 25.0,
      "delta_t_us": 0.5,
      "delta_l_m": 0.005
    },
    "output": {
      "directory": "output"
    }
  }
}
---
```

For synthetic demo data, resolution can be disabled:

```json
"resolution": {"kind": "none"}
```

For real instrument data, use Gaussian resolution parameters or a tabulated
resolution file. Synthetic data often does not need an instrument resolution
file; real experiments normally do.

## Supported Workflow Inputs

`single_spectrum` fits one spectrum. It supports:

- `counts_npz` or `counts`: a `.npz` file with `sample_counts`,
  `open_beam_counts`, and `energies_ev` by default. Arrays may be 1D spectra
  or 3D cubes; 3D cubes are summed over pixels before fitting one spectrum.
- `transmission_npz`, `transmission`, or `spectrum`: a `.npz` or text file
  with `transmission`, optional `uncertainty`, and an energy grid.

`density_map` and `spatial_map` fit every pixel in a 3D cube. They support:

- `transmission_npz` or `transmission`: a `.npz` file with 3D
  `transmission` and optional 3D `uncertainty` arrays.
- `transmission_tiff` or `tiff`: a multi-frame TIFF transmission stack plus
  an `energy_grid` entry.
- `counts_npz` or `counts`: a `.npz` file with 3D `sample_counts` and
  `open_beam_counts` arrays.
- `nexus_histogram` or `nexus`: sample and open-beam NeXus histogram files,
  configured with `sample_path` and `open_beam_path`.

Paired arrays must have matching shapes. The number of energy points must
match the first axis of the data arrays. Energy grids must be strictly
monotonic; descending grids are reversed with the aligned arrays before
fitting.

For NeXus histogram inputs, NEREIDS loads counts in TOF-bin order and derives
ascending energy centers with `tof_to_energy_centers(...)`. The MCP workflow
converts the counts to the same ascending-energy order before fitting. If the
manifest does not specify `flight_path_m`, the loader metadata is used when
available, with a 25 m fallback. `delay_us` defaults to 0.

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
      "flight_path_m": 25.0,
      "delay_us": 0.0
    },
    "isotopes": [
      {"isotope": "U-238", "initial_density": 0.001}
    ],
    "fit": {
      "solver": "lm",
      "max_iter": 100
    },
    "resolution": {
      "kind": "gaussian",
      "flight_path_m": 25.0,
      "delta_t_us": 0.5,
      "delta_l_m": 0.005
    }
  }
}
---
```

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
