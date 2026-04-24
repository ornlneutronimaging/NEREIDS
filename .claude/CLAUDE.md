# CLAUDE.md — NEREIDS Codebase Instructions

## Pre-Commit Checklist (mandatory before every commit)

Always run these three commands and fix all output before committing:

```
cargo fmt --all
cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings
cargo test --workspace --exclude nereids-python
```

Run `cargo fmt` (not `--check`) so formatting is applied — never rely on
targeted `Edit` patches. Do not suppress clippy warnings with `#[allow(...)]`.

## Physics Rules

- **No approximations**: implement exact SAMMY physics. Never introduce ad-hoc
  approximations.
- **Document physics in-code**: reference specific SAMMY source files and
  equation numbers in Rust comments.
- **Validate against SAMMY tests**: use SAMMY's own test cases as ground truth
  for each module.

## Commit Style

- **GPG-signed commits**: always use `git commit -S`. The GPG agent is working.
- **Atomic commits**: commit early and often. Huge monolithic commits make it
  hard to catch off-rail behaviour.
- **No temp file litter**: clean up all one-off scripts immediately.
- **Do not touch `.claude/worktrees/`**: this directory is managed by
  Claude Code for isolated worktree sessions. Never delete, modify, or
  flag its contents during reviews.

## Project Layout

- `crates/nereids-core`    — shared types (Isotope, etc.)
- `crates/nereids-endf`    — ENDF/B file parsing and resonance data structures
- `crates/nereids-physics` — cross-section physics (Reich-Moore, SLBW, RML)
- `crates/nereids-fitting` — Levenberg-Marquardt fitting engine
- `crates/nereids-io`      — TIFF I/O, TOF normalisation
- `crates/nereids-pipeline`— spatial mapping pipeline (rayon)
- `crates/nereids-python`  — PyO3 bindings (excluded from `--workspace` clippy/test runs)
- `apps/gui`               — egui desktop application

## Mandatory User Checkpoints (NEVER skip these)

The user MUST have the opportunity to review and intervene at these points.
**Always pause and present a summary, then STOP and wait for user response.**

0. **Before starting any work on a new issue/task**: Enter Plan mode.
   Conduct thorough research appropriate to the task — study reference
   implementations, trace affected code paths, form an independent
   analysis. Present the full findings and implementation plan.
   STOP here. Do NOT write any code until the user approves the plan.
1. **After creating PRs**: show PR URLs + summary of changes per PR.
   STOP here. Do NOT proceed to reviews until the user responds.
2. **After review pipeline completes**: present the review summary table.
   STOP here. Do NOT merge until the user explicitly says "merge" or
   "proceed".
3. **After post-merge gate**: report results before closing issues/epics.
   STOP here. Do NOT close issues until the user confirms.

**"STOP" means end your turn.** Do not add "Shall I proceed?" and then
continue in the same message. Do not launch background tasks that advance
to the next stage. The user's next message is the gate.

The user approving step N does NOT imply approval for step N+1.

## Multi-AI Review Pipeline (mandatory before merging PRs)

Every feature branch must pass a review before merge. There is exactly
ONE review mechanism: **`/review-pipeline`**. The standalone `/self-review`
and `/codex-review` skills have been merged into it.

**HARD RULE**: When asked to review, ALWAYS invoke `/review-pipeline` via
the Skill tool. NEVER substitute with ad-hoc review agents or custom
subagents. The skill contains the full workflow: Claude audit + Codex
external review + user gates + iteration logic + Copilot phase.

| Phase | What | When |
|-------|------|------|
| A | Claude self-audit + Codex CLI (inside `/review-pipeline`) | Before push |
| B | GitHub Copilot (manual trigger, fetched inside `/review-pipeline`) | After push |

**Post-merge**: run `/post-merge` for the integration gate on merged main.

## Validation Patterns

- **Validate config up-front** in public entry points, before rayon
  iterators. `Err(_) => None` in `filter_map` only for per-pixel edge cases.
- **NaN bypasses guards**: `NaN < 1.0` is `false`. Always pair with
  `.is_finite()`.
- **Empty collections pass equality**: `0 == 0` passes. Guard
  `is_empty()` separately.
- **`debug_assert!` is for impossible states only**, not input validation.
  Use hard errors (`return Err(...)`) for parser/config validation.
- **Magic number → named constant**: preserve the exact numeric value.
  Verify with a mapping table and grep-verify after bulk replacements.

## Execution Model (mandatory)

### Branching: choose the right approach for the work pattern

| Work pattern | Approach |
|--------------|----------|
| Multiple independent PRs (no file overlap) | Parallel worktrees + concurrent subagents |
| Single PR, sequential work | Feature branch at repo root |

**Decide at planning time** and state the choice explicitly in the plan.
Worktrees add friction (path confusion, cleanup errors) with zero benefit
when working single-threaded on one feature branch.

### When using parallel worktrees:
- Create one worktree per PR/branch, launch subagents in parallel (one
  per worktree), and let them work concurrently.
- **Launch subagents in a single message** so they run concurrently.

### Always:
- **Group related changes into PRs**, not individual commits.  Each PR
  should map to a coherent scope (e.g., one issue, one crate, one theme).
- **Close/comment on GitHub issues in parallel** with implementation work.

## Git Workflow

- **origin**: `ornlneutronimaging/NEREIDS` — primary repo, issues, releases,
  main branch.  All branches and PRs are pushed here directly.
- No other remotes configured.  The `KedoKudo/NEREIDS` fork exists on GitHub
  but is not added as a local remote (removed to avoid `gh` targeting the
  wrong repo).

## Reference Codebases (siblings of this repo)

- `../SAMMY`    — physics reference (resonance formalism, SAMMY source)
- `../PLEIADES` — ORNL data normalisation helpers, ENDF retrieval
- `../trinidi`  — sparsity-handling reference (Purdue/LANL, mostly abandoned)

## Documentation hygiene — no investigation / audit / debugging memos in `docs/`

`docs/` is **user-facing**. Only these subtrees belong there:

| Subtree | What belongs |
|---------|--------------|
| `docs/adr/`        | Architectural decision records — one per non-trivial, durable design choice |
| `docs/guide/`      | User-facing how-to guides |
| `docs/references/` | External-facing reference material (file formats, CLI, API) |

**Anything else does not belong in `docs/`.** In particular:

- **Audit memos** (impact analysis of a bug fix, which research scripts
  were affected, etc.) — belong in the PR body. When the PR merges,
  the memo is preserved in the commit + PR history where it is actually
  searchable. Do not create `docs/audit/` or similar.
- **Investigation / ablation / debugging notes** (step-by-step inquiry
  into a performance question, correctness hypothesis, etc.) — belong in
  `.research/` (gitignored; session-private) or in the PR body if the
  investigation directly motivates a concrete code change.
- **Design specs / requirements drafts** — belong inline with the code
  they specify (rustdoc on the struct / module they describe) or in
  `docs/adr/` if treated as a durable decision.  Not as a floating
  `docs/<feature>-requirements.md`.

If I feel compelled to commit a memo under `docs/`, I stop and check:
(a) is this user-facing reference that someone reading the project
cold will want to read?  If yes, it fits one of the subtrees above.
If no, it belongs in `.research/` or the PR body.  **When in doubt,
ask the user before creating the file** — do not invent a new
`docs/<whatever>/` directory to make the artifact commit-able.

This rule exists because past agent work (PR #466's `docs/audit/`,
this very kind of investigation) has repeatedly put transient memos
under `docs/` on the theory that "it's documentation, it belongs in
docs" — that's the wrong mental model.  User-facing documentation is
a narrower category than "anything that is prose + markdown."
