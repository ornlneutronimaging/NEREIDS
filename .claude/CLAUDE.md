# CLAUDE.md — NEREIDS Codebase Instructions

## Pre-Commit Checklist (mandatory before every commit)

Always run these two commands and fix all output before committing:

```
cargo fmt --all
cargo clippy --workspace --exclude nereids-python -- -D warnings
```

`cargo fmt --all` must be run (not just `--check`) so that formatting is
actually applied. Never rely on targeted `Edit` patches to keep formatting
correct — rustfmt has its own opinions on argument layout, trailing commas,
and line width that must be respected.

`cargo clippy -- -D warnings` treats every warning as an error, matching CI.
Fix all warnings; do not suppress them with `#[allow(...)]` unless there is a
documented reason in a comment.

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

Every feature branch must pass a 3-stage review before merge.  Run each
stage, fix all findings, then proceed to the next.

### Stage 1 — Self-review (Claude sub-agent)

After implementation and tests pass, spawn a separate review agent:

```
Task(subagent_type="general-purpose", prompt="""
Audit <files> for: logic bugs, panics, missing validation, physics
correctness, API consistency with existing patterns.  Report findings
as P1 (must fix) / P2 (should fix) with file:line references.
""")
```

Fix all P1s and re-run until clean.

### Stage 2 — Codex independent review

After committing, run the Codex CLI for an independent second opinion:

```bash
# Review uncommitted changes (before commit):
codex review --uncommitted "Focus on panics, validation gaps, edge cases, numerical stability"

# Review a branch against main (after commit):
codex review --base main
```

Codex outputs `[P1]`/`[P2]` findings with file:line references.  Fix all
P1s, re-commit, and re-run until clean.

### Stage 3 — GitHub Copilot PR review

After pushing the branch and creating/updating a PR, Copilot auto-reviews:

```bash
# Fetch review comments:
gh api repos/{owner}/{repo}/pulls/{pr}/comments \
  --jq '.[] | {path, line, body, created_at}'
```

Address all actionable comments, push fixes, and re-fetch until clean.
Dismiss comments that are: rehashing already-addressed issues, debating
design choices with no clear improvement, or flagging edge cases that
cannot occur in practice.

### Review stage summary

| Stage | Tool | When | Strength |
|-------|------|------|----------|
| 1 | Claude sub-agent | Before commit | Architecture, physics, design |
| 2 | `codex review` | After commit | Fresh eyes, panic/edge-case detection |
| 3 | Copilot (via `gh api`) | After push to PR | Diff-focused, code-centric |

## Git Workflow

- **Upstream**: `ornlneutronimaging/NEREIDS` — issues, releases, main branch
- **Fork**: `KedoKudo/NEREIDS` — development branches, PRs
- **Sync**: after merging a PR on the fork, push main to upstream:
  `git push upstream main`

## Reference Codebases (siblings of this repo)

- `../SAMMY`    — physics reference (resonance formalism, SAMMY source)
- `../PLEIADES` — ORNL data normalisation helpers, ENDF retrieval
- `../trinidi`  — sparsity-handling reference (Purdue/LANL, mostly abandoned)
