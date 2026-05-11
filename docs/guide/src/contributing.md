# Contributing

## Development Setup

```bash
git clone https://github.com/ornlneutronimaging/NEREIDS.git
cd NEREIDS
cargo build --workspace
```

For Python binding development, use [pixi](https://pixi.sh):

```bash
pixi run build        # maturin release build
pixi run test-python  # pytest
```

## Pre-Commit Checklist

Run these three commands before every commit:

```bash
cargo fmt --all
cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings
cargo test --workspace --exclude nereids-python
```

- `cargo fmt` applies formatting (not just `--check`)
- `cargo clippy` treats all warnings as errors
- `nereids-python` is excluded because it requires PyO3/maturin build setup

## Branch and PR Workflow

1. Create a feature branch from `main`
2. Make changes, commit with GPG signatures (`git commit -S`)
3. Push and open a PR against `main`
4. All PRs go through the review pipeline before merge

The repository uses a single remote (`origin` = `ornlneutronimaging/NEREIDS`).
All branches and PRs are pushed directly.

## Code Guidelines

### Physics Modules

- Implement exact SAMMY physics -- no ad-hoc approximations
- Reference SAMMY source files and equation numbers in doc comments
- Validate against SAMMY's own test cases as ground truth

### General

- Validate configuration up-front in public entry points
- Guard NaN with `.is_finite()` (NaN bypasses comparison guards)
- Guard empty collections explicitly (`.is_empty()`)
- Use named constants instead of magic numbers
- Prefer `return Err(...)` for input validation, not `debug_assert!`

### Testing

```bash
# Rust tests
cargo test --workspace --exclude nereids-python

# Python tests (requires pixi)
pixi run test-python

# Build docs locally
cd docs/guide && mdbook build && mdbook serve
```

## Documentation and Release Checklist

Before a release or a documentation-focused PR, verify:

- `pixi run doc-guide` builds the mdBook guide.
- `pixi run doc-build` builds mdBook, builds Rustdoc, and copies Rustdoc to
  `target/book/api`.
- The [Python API reference](./python-api.md) matches the shipped
  `bindings/python/python/nereids/__init__.pyi` stubs for public functions,
  arguments, array shapes, and result objects.
- The [Data I/O and NeXus/TOF](./data-io.md) page matches current TIFF,
  NeXus, normalization, and `tof_to_energy_centers(...)` behavior.
- PyPI metadata in `pyproject.toml` lists current optional extras, including
  `mcp` and `gui`.
- GUI installation docs cover both `pip install "nereids[gui]"` and the
  direct `nereids-gui` package, plus Homebrew/source options.
- GUI screenshots still match the current landing, guided workflow, studio,
  and tools screens.
- Notebook release notes state whether notebooks were smoke-run, which subset
  was run, and which require ENDF network access or external PLEIADES/Git LFS
  data.
- Rust quickstart snippets have been checked against current public APIs.

## Project Structure

```text
NEREIDS/
  crates/
    endf-mat/          # Element/MAT lookup tables
    nereids-core/      # Core types and constants
    nereids-endf/      # ENDF retrieval and parsing
    nereids-physics/   # Cross-section physics
    nereids-fitting/   # Optimization engines
    nereids-io/        # Data I/O (TIFF, NeXus)
    nereids-pipeline/  # Orchestration
  bindings/python/     # PyO3 Python bindings
  apps/gui/            # egui desktop application
  docs/
    guide/             # mdBook user guide (this site)
    adr/               # Architecture decision records
    references/        # Reference materials
```

## Useful Commands

| Task | Command |
|------|---------|
| Build all | `cargo build --workspace` |
| Run tests | `cargo test --workspace --exclude nereids-python` |
| Format | `cargo fmt --all` |
| Lint | `cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings` |
| Build Python | `pixi run build` |
| Test Python | `pixi run test-python` |
| Build docs | `cd docs/guide && mdbook build` |
| Build full docs site | `pixi run doc-build` |
| Serve docs | `cd docs/guide && mdbook serve` |
| Build rustdoc | `cargo doc --workspace --no-deps --exclude nereids-python` |
