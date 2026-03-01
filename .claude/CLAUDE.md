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
**Always pause and present a summary, then wait for user confirmation.**

1. **After creating PRs**: show PR URLs + summary of changes per PR. Ask
   if the user wants to request Copilot review before proceeding.
2. **Before merging**: present the pre-merge checklist (review results,
   test status). Never `gh pr merge` without explicit user approval.
3. **After post-merge gate**: report results before closing issues/epics.

Do NOT chain these steps automatically. Each checkpoint is a full stop.
The user approving step N does NOT imply approval for step N+1.

## Multi-AI Review Pipeline (mandatory before merging PRs)

Every feature branch must pass a **full** multi-stage review before merge.
**Always use `/review-pipeline`** — never substitute a single self-review.
Self-review alone bypasses the independent Codex second opinion, which is
the highest-value check (cross-model confirmed findings are always real).

| Phase | Tool | Skill | When |
|-------|------|-------|------|
| A (local) | Claude self-review + Codex CLI | `/review-pipeline` | Before push |
| B (remote) | GitHub Copilot (manual trigger) | — | After push to PR |

**Phase A** iterates until zero P1s (max 4 rounds, then escalate to human).
**Phase B** re-iterates if 3+ P1s or P1 ratio > 40%; otherwise fix inline
and merge. Dismiss Copilot comments that rehash addressed issues or flag
impossible edge cases. Fetch Copilot comments reliably with:
`pixi run copilot-reviews {pr_numbers...}`

**Post-merge**: run `/post-merge` to execute the full integration gate on
merged main (cleanup, `cargo clean && pixi run build`, tests, issue closure,
memory update). `pixi run build` must run first — it catches cross-PR
signature mismatches that per-branch reviews miss.

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

When implementing fixes or features across multiple issues, crates, or files:

- **ALWAYS use parallel worktrees + concurrent subagents.** Create one
  worktree per PR/branch, launch Task subagents in parallel (one per
  worktree), and let them work concurrently.  Never fall back to
  sequential single-thread editing across multiple issues.
- **Group related changes into PRs**, not individual commits.  Each PR
  should map to a coherent scope (e.g., one issue, one crate, one theme).
- **Launch subagents in a single message** so they run concurrently.
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
