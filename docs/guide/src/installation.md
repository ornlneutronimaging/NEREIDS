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

### Linux system dependencies

NEREIDS uses GTK 3 for native file dialogs (no `xdg-desktop-portal`
daemon needed) and the standard egui/winit/wgpu stack for the rest of
the UI. Desktop Linux distros usually ship these, but minimal /
container / server installs do not.

**Debian / Ubuntu (apt):**

```bash
sudo apt-get install -y \
  libgtk-3-0t64 libxcursor1 libx11-xcb1 libxi6 libxrandr2 \
  libxinerama1 libxxf86vm1 libxkbcommon-x11-0 libwayland-client0 \
  libgl1 libegl1
```

On Ubuntu releases older than 24.04 the GTK 3 runtime is `libgtk-3-0`
instead of `libgtk-3-0t64`; the older name still resolves on 24.04 via
a transitional package.

Contributors building from source additionally need the GTK 3
development headers and `pkg-config`:

```bash
sudo apt-get install -y libgtk-3-dev pkg-config
```

**Fedora / RHEL (dnf):**

```bash
sudo dnf install -y \
  gtk3 libXcursor libXi libXrandr libXinerama libxkbcommon-x11 \
  libwayland-client libwayland-cursor mesa-libGL mesa-libEGL
```

Contributors building from source additionally need `gtk3-devel` and
`pkgconf-pkg-config`.

**Headless / Docker / VM fallback:**

If the GUI fails at startup with a GL initialisation error (common in
Docker without GPU passthrough, or over SSH-X without GLX), force
software rasterisation:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
nereids-gui
```

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
