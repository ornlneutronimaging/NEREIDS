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

The Linux wheel is built for `manylinux_2_28`, so it runs on any
x86_64 distribution with glibc ≥ 2.28: RHEL/AlmaLinux/Rocky 8+,
Ubuntu 20.04+, Debian 10+, Fedora 29+, and newer.

File dialogs use a three-tier chain with **no hard system
dependencies**:

1. **XDG desktop portal** (`org.freedesktop.portal.FileChooser` over
   D-Bus) — native dialogs on any desktop session (GNOME, KDE, …).
   Preinstalled on every mainstream desktop, including RHEL 8's GNOME.
2. **zenity** — automatic fallback when no portal is reachable.
   Recommended for `ssh -X` sessions and containers:
   `sudo dnf install zenity` / `sudo apt-get install zenity`.
3. **Built-in file browser** — rendered by the GUI itself, works in
   every environment (root, containers, no D-Bus). Selected
   automatically when neither portal nor zenity is available; the GUI
   shows a banner saying so.

The rest of the UI is the standard egui/winit/GL stack. Desktop Linux
distros ship these; minimal / container / server installs may not:

**Debian / Ubuntu (apt):**

```bash
sudo apt-get install -y \
  libxcursor1 libx11-xcb1 libxi6 libxrandr2 \
  libxinerama1 libxxf86vm1 libxkbcommon-x11-0 libwayland-client0 \
  libgl1 libgl1-mesa-dri libegl1
```

`libgl1-mesa-dri` is needed even with `LIBGL_ALWAYS_SOFTWARE=1`
(below) because the software rasteriser is shipped as a Mesa DRI
driver.

**Fedora / RHEL (dnf):**

```bash
sudo dnf install -y \
  libXcursor libXi libXrandr libXinerama libxkbcommon-x11 \
  libwayland-client libwayland-cursor \
  mesa-libGL mesa-libEGL mesa-dri-drivers
```

No GTK packages and no development headers are required — neither at
runtime nor for building from source (the dialog stack has no
build-time system libraries; only CMake for HDF5, as noted above).

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
