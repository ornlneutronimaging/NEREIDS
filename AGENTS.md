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
  for a sequential change. Use additional agents only when requested, or for
  a review round.
- Run `cargo fmt --all`,
  `cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings`,
  and `cargo test --workspace --exclude nereids-python` before a code commit.
  Rebuild Python and run its tests when bindings change. Run MCP tests with
  `pixi run -e mcp test-mcp` when that interface changes.
- Use signed commits. Keep commits focused. Keep the PR description to two
  sentences. Ask before publishing or merging unless the user has authorized
  that action. Do not infer merge approval from review approval.

## Pull request review

Each PR gets one review round. The maintainer asks for it, and this request
authorizes its commits and its push. The maintainer decides if the PR needs
another round. In Claude Code, the `pr-review` skill runs the round.

1. Open the PR. When CodeRabbit skips or rate-limits its review, a maintainer
   asks for it with `@coderabbitai review`.
2. Review the PR diff with Claude and with Codex.
3. Collect the CodeRabbit, Claude and Codex findings in one fix table. Do not
   fix a comment when it arrives.
4. For each valid finding, compare the smallest fix with the fix of the root
   cause. Reject a fix that adds a special case, a guard for one input, a
   duplicate path, or a looser tolerance. The next review must check that
   code again. Give one fix to findings that have one cause.
5. Write the fix plan. Get a Codex risk review of the plan. Revise the plan
   for each high risk.
6. Apply, check and commit one fix at a time. Push once, after the last
   commit.
7. Report the fix table to the maintainer. Then stop. CodeRabbit does not
   review new commits automatically.

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
