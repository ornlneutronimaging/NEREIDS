# Contributing to NEREIDS

Thank you for your interest in contributing to NEREIDS! This document covers
the development setup, coding standards, and pull request process.

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/ornlneutronimaging/NEREIDS/issues)
with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Version information (`pip show nereids` or `cargo pkgid nereids-core`)

## Suggesting Features

Open a GitHub issue with the `enhancement` label. Include:
- The use case or problem it solves
- Any relevant physics references or SAMMY documentation

## Development Setup

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- [pixi](https://pixi.sh/) (Python environment manager)
- cmake (for HDF5 builds)

### Building

```bash
git clone https://github.com/ornlneutronimaging/NEREIDS.git
cd NEREIDS

# Rust library
cargo build --workspace

# Python bindings (uses pixi for environment management)
pixi run build
```

### Running Tests

```bash
# Rust tests (excludes Python bindings crate)
cargo test --workspace --exclude nereids-python

# Python tests (builds first, then runs pytest)
pixi run test-python
```

## Code Style

### Rust

All code must pass these checks before committing:

```bash
cargo fmt --all
cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings
cargo test --workspace --exclude nereids-python
```

- Run `cargo fmt` (not `--check`) so formatting is applied automatically
- Do not suppress clippy warnings with `#[allow(...)]` -- fix the underlying issue
- Use `Result` for error handling in public APIs, not `unwrap()`/`expect()`/`assert!()`

### Python

- Follow PEP 8
- Type stubs are maintained in `bindings/python/python/nereids/__init__.pyi`
- Tests live in `tests/` and use pytest

### Physics Code

- **No approximations**: implement exact SAMMY physics
- Document physics in-code with references to SAMMY source files and equation numbers
- Validate against SAMMY's own test cases (see `tests/data/samtry/`)

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with atomic, well-described commits
3. Ensure all three pre-commit checks pass (fmt, clippy, test)
4. Open a PR against `main`
5. PRs go through automated CI and code review before merge

### Branch Naming

- `feature/description` -- new functionality
- `fix/description` -- bug fixes
- `docs/description` -- documentation changes

### Commit Messages

- Use imperative mood ("Add feature" not "Added feature")
- First line: concise summary (< 72 chars)
- Body: explain *why*, not just *what*

## Project Structure

```
crates/
  nereids-core/       Shared types, constants, isotope registry
  nereids-endf/       ENDF file retrieval and resonance parsing
  nereids-physics/    Cross-section calculation and broadening
  nereids-fitting/    LM and Poisson optimizers
  nereids-io/         TIFF/NeXus I/O, normalization
  nereids-pipeline/   Spatial mapping orchestration
  nereids-python/     PyO3 Python bindings (excluded from workspace clippy/test)
apps/gui/             egui desktop application
examples/notebooks/   Jupyter tutorials (4 tiers)
tests/                Integration tests and SAMMY validation data
```

## License

By contributing, you agree that your contributions will be licensed under the
BSD-3-Clause license (see [LICENSE](LICENSE)).
