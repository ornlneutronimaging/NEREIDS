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
pip install nereids
```

**Requirements**: Python 3.10+ and NumPy.

Optional extras published by the `nereids` package:

```bash
pip install "nereids[mcp]"  # installs the MCP server dependency
pip install "nereids[gui]"  # pulls in the GUI wheel package when available
```

## MCP Server

The MCP server is installed as an optional Python extra:

```bash
pip install "nereids[mcp]"
nereids-mcp
```

See the [MCP server](./mcp-server.md) chapter for client configuration and
manifest-driven workflows.

## Desktop GUI

### Python Wheel

```bash
pip install "nereids[gui]"
nereids-gui
```

The `[gui]` extra pulls in the separately-published `nereids-gui` wheel,
which is what provides the `nereids-gui` console script (it is not declared
in the base `nereids` package). If the install resolves but `nereids-gui`
is not found on `PATH`, the `nereids-gui` wheel has not been published for
your platform/Python version — verify with:

```bash
which nereids-gui    # should print a path; empty output means missing
pip show nereids-gui # should print metadata; "not installed" means the
                     # extra resolved a different way
```

You can also install the GUI distribution directly:

```bash
pip install nereids-gui
nereids-gui
```

### macOS (Homebrew)

```bash
brew tap ornlneutronimaging/nereids
brew install --cask nereids
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
  libgtk-3-0 libxcursor1 libx11-xcb1 libxi6 libxrandr2 \
  libxinerama1 libxxf86vm1 libxkbcommon-x11-0 libwayland-client0 \
  libgl1 libgl1-mesa-dri libegl1
```

`libgtk-3-0` is portable across Debian and all current Ubuntu LTS
releases. On Ubuntu 24.04 (Noble) it pulls in `libgtk-3-0t64` under
the hood via the t64 transitional package. `libgl1-mesa-dri` is
needed even with `LIBGL_ALWAYS_SOFTWARE=1` (below) because the
software rasteriser is shipped as a Mesa DRI driver.

Contributors building from source additionally need the GTK 3
development headers and `pkg-config`:

```bash
sudo apt-get install -y libgtk-3-dev pkg-config
```

**Fedora / RHEL (dnf):**

```bash
sudo dnf install -y \
  gtk3 libXcursor libXi libXrandr libXinerama libxkbcommon-x11 \
  libwayland-client libwayland-cursor \
  mesa-libGL mesa-libEGL mesa-dri-drivers
```

Contributors building from source additionally need `gtk3-devel` and
`pkgconf-pkg-config`.

**Headless / Docker / VM fallback:**

If the GUI fails at startup with a GL initialisation error (common in
Docker without GPU passthrough, or over SSH-X without GLX), force
software rasterisation by setting `LIBGL_ALWAYS_SOFTWARE=1` before
launching the GUI:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
cargo run --release -p nereids-gui   # from source
# or, if installed as a binary:
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
