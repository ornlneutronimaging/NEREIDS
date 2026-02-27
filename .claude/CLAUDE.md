# CLAUDE.md — NEREIDS Codebase Instructions

## Pre-Commit Checklist (mandatory before every commit)

Always run these three commands and fix all output before committing:

```
cargo fmt --all
cargo clippy --workspace --exclude nereids-python -- -D warnings
cargo test --workspace --exclude nereids-python
```

`cargo fmt --all` must be run (not just `--check`) so that formatting is
actually applied. Never rely on targeted `Edit` patches to keep formatting
correct — rustfmt has its own opinions on argument layout, trailing commas,
and line width that must be respected.

`cargo clippy -- -D warnings` treats every warning as an error, matching CI.
Fix all warnings; do not suppress them with `#[allow(...)]` unless there is a
documented reason in a comment.

`cargo test` catches regressions from API changes that ripple across crates
(e.g., changing a return type in nereids-core breaks nereids-endf callers).
The workspace run is fast (~2 s) and prevents cross-crate breakage.

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

## Multi-AI Review Pipeline (mandatory before merging PRs)

Every feature branch must pass a multi-stage review before merge.
Run `/review-pipeline` to execute the full pipeline across active worktrees.

| Phase | Tool | Skill | When |
|-------|------|-------|------|
| A (local) | Claude subagent + Codex CLI | `/review-pipeline`, `/self-review`, `/codex-review` | Before push |
| B (remote) | GitHub Copilot (manual trigger) | — | After push to PR |

**Phase A** iterates until zero P1s (max 4 rounds, then escalate to human).
**Phase B** re-iterates if 3+ P1s or P1 ratio > 40%; otherwise fix inline
and merge. Dismiss Copilot comments that rehash addressed issues or flag
impossible edge cases.

**Post-merge**: always run `cargo test --workspace --exclude nereids-python`
on the merged main to catch cross-PR integration regressions.

## Validation Patterns

Lessons from review pipeline findings — apply these consistently:

- **Validate config up-front** in public entry points (`fit_spectrum`,
  `spatial_map`, `fit_roi`, `sparse_reconstruct`). Check lengths,
  emptiness, and finiteness *before* entering rayon parallel iterators.
  Silent `Err(_) => None` in `filter_map` is acceptable only for
  per-pixel numerical edge cases, never for config errors.
- **NaN bypasses guards**: `NaN < 1.0` is `false`, so range checks like
  `x < 1.0` don't catch NaN. Always pair with `.is_finite()`.
- **Empty collections pass equality**: `0 == 0` passes length-match
  checks. Guard `is_empty()` or `> 0` separately.
- **`debug_assert!` for impossible states**: use `obs.is_finite() &&
  obs >= 0.0` pattern — catches both NaN and negative in one assert.

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
