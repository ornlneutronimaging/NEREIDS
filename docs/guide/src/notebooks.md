# Notebook Status

The tutorial notebooks live under `examples/notebooks/`. They are examples,
not the API contract: the contract is the Python type stubs, the Python and
Rust tests, the mdBook guide and Rustdoc.

## Notebooks

| Notebook group | Contents |
|----------------|----------|
| `foundations/` | Cross-section, broadening and transmission physics. |
| `building_blocks/` | ENDF loading and element utilities. |

Counts-fitting, mapping and I/O tutorials will be re-added against
`fit_counts` and `fit_map`.

Notebooks are not executed by `pixi run doc-guide`, `pixi run doc-build` or
GitHub Pages publishing. Before using one as release evidence, run it manually
with a controlled data and network policy. Notebooks that call
`load_endf(...)` may need network access on first run; ENDF files are cached
after retrieval.
