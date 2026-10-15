# NEREIDS

NEREIDS fits density, temperature, or both from raw sample and open-beam
counts on detector time-bin edges. State which result a change supports.

## Implementation

- Use `counts_fit::fit_counts` and `counts_map::fit_map` in
  `crates/nereids-pipeline`. They use the Poisson/KL solver in
  `crates/nereids-fitting/src/poisson.rs`.
- Keep one fitting path. Keep Python, MCP, and GUI adapters thin.
- Inspect current code and tests. Do not treat an issue or the pipeline map
  as proof that a feature exists.

## Correctness

- Do not introduce ad-hoc physics approximations.
- Preserve count statistics, units, time-bin edges, physical background
  placement, and calibration uncertainty. Reject unsupported inputs.
- Document physics with equations or source references. Validate changes
  against an independent oracle or an applicable SAMMY case.
- Confirm that a regression fails without the fix. Do not widen tolerances
  to conceal a failure. Convergence does not prove physical accuracy.
- Check public inputs for invalid shapes, empty arrays, and non-finite values.

## Work and review

- Give a short plan before edits. An approved task permits its routine fixes
  and checks. Ask again only for a new scientific choice or destructive work.
- Preserve user files, local data, and audit evidence. Use one feature branch
  for a sequential change. Use additional agents only when requested.
- Run `cargo fmt --all`,
  `cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings`,
  and `cargo test --workspace --exclude nereids-python` before a code commit.
  Rebuild Python and run its tests when bindings change. Run MCP tests with
  `pixi run -e mcp test-mcp` when that interface changes.
- Use signed commits. Keep commits focused. Keep the PR description to two
  sentences. Ask before publishing or merging unless the user has authorized
  that action. Do not infer merge approval from review approval.

## Text and files

- Use ASD-STE100 as the text style. Use short sentences, consistent technical
  terms, and direct instructions. State evidence limits.
- Comments explain behavior, units, or a necessary constraint. Keep process
  history in commits. Keep audit notes in `.research/`.
- `docs/` contains user guides, reference material, and durable architecture
  decisions. Do not add private review notes there.
- Keep shared instructions here. `CLAUDE.md` points here. Do not add a local
  agent framework or workflow ladder without a demonstrated need.

Repository: `ornlneutronimaging/NEREIDS`.
