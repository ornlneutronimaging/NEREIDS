# Development

This repo uses Pixi to manage a reproducible development environment.

## Quickstart
- Install Pixi and create the environment:
  - `pixi install`
- Run tasks via Pixi:
  - `pixi run fmt`
  - `pixi run clippy`
  - `pixi run test`
  - `pixi run build` (Python bindings via maturin)

## Notes
- Some tasks will be no-ops until Rust crates and Python bindings are added.
- The default environment targets macOS (arm64) and Linux (x86_64).
