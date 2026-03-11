# Installation

## Rust Library

Add the top-level orchestration crate (re-exports all lower-level crates):

```toml
[dependencies]
nereids-pipeline = "0.1"
```

Or add individual crates (`nereids-core`, `nereids-endf`, `nereids-physics`,
`nereids-fitting`, `nereids-io`) for finer-grained dependency control.

**Requirements**: Rust edition 2024 (rustc 1.85+).

### Optional: HDF5 support

The `nereids-io` crate has an optional `hdf5` feature for NeXus file support:

```toml
[dependencies]
nereids-io = { version = "0.1", features = ["hdf5"] }
```

This requires the HDF5 C library to be installed on your system.

## Python Bindings

```bash
pip install nereids          # available after first public release
```

**Requirements**: Python 3.10+ and NumPy.

## Desktop GUI

### macOS (Homebrew)

```bash
brew install --cask ornlneutronimaging/nereids/nereids   # available after first public release
```

### From Source

```bash
git clone https://github.com/ornlneutronimaging/NEREIDS.git
cd NEREIDS
cargo run --release -p nereids-gui
```

Building from source requires CMake (for HDF5) and a Rust toolchain.

## Development Setup

For contributors working on NEREIDS itself:

```bash
git clone https://github.com/ornlneutronimaging/NEREIDS.git
cd NEREIDS

# Build everything
cargo build --workspace

# Run tests
cargo test --workspace --exclude nereids-python

# Build Python bindings (requires pixi)
pixi run build
pixi run test-python
```

See [Contributing](./contributing.md) for the full development workflow.
