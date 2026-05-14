# Quickstart: Rust

This example loads ENDF resonance data for U-238, computes a theoretical
transmission spectrum, and fits it to recover the areal density.

The snippets below are spliced from
[`crates/nereids-fitting/examples/quickstart.rs`](https://github.com/ornlneutronimaging/NEREIDS/blob/main/crates/nereids-fitting/examples/quickstart.rs)
so the rendered page cannot drift out of sync with the live crate APIs:
the example is compile-checked by `cargo check --workspace --examples` in
CI. Run the full example locally with
`cargo run --example quickstart -p nereids-fitting` (first run requires
network access to fetch ENDF/B-VIII.1).

## Setup

```toml
# Cargo.toml
[dependencies]
nereids-core = "0.1"
nereids-endf = "0.1"
nereids-physics = "0.1"
nereids-fitting = "0.1"
```

## Load ENDF Data

```rust,no_run
{{#include ../../../crates/nereids-fitting/examples/quickstart.rs:setup}}
```

## Compute a Forward Model

```rust,no_run
{{#include ../../../crates/nereids-fitting/examples/quickstart.rs:forward}}
```

## Fit a Measured Spectrum

```rust,no_run
{{#include ../../../crates/nereids-fitting/examples/quickstart.rs:fit}}
```

## Next Steps

- See the [API Reference](api/nereids_pipeline/) for the full API
- Explore the [Python quickstart](./quickstart-python.md) for a NumPy-based workflow
