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

Every feature branch must pass a two-phase review before merge.

### Phase A — Local iterative review (Claude + Codex)

Run "review pipeline" to execute the local review loop. The pipeline runs
these stages per iteration:

1. **Self-audit** (Claude subagent) — architecture, physics, design
2. **External review** (Codex CLI) — fresh eyes, panic/edge-case detection
3. **Consolidation** — merge findings, present P1/P2 report for approval
4. **Fix** — parallel fix agents per worktree
5. **Verification** — re-audit to confirm fixes
6. **Commit & push**

Individual stages: "self-review", "codex-review"

**Iterate locally until zero P1s remain.** Then push to remote for Phase B.

Maximum **4 local iterations** per branch. If P1s persist after 4 rounds,
stop and escalate to human developer review — this signals systematic
issues that need manual assessment (scope too large, task decomposition
needed, or fundamental design problems).

### Phase B — Copilot review (manual trigger on GitHub)

After pushing, the developer manually triggers Copilot review on the PR.
Fetch comments:
```bash
gh api repos/{owner}/{repo}/pulls/{pr}/comments \
  --jq '.[] | {path, line, body, created_at}'
```

**Copilot re-iteration gate** (combined threshold):

Re-iterate locally (back to Phase A) if **either** condition is met:
- **3+ confirmed P1s** from Copilot review, OR
- **P1 ratio > 40%** of total Copilot findings

Otherwise (0-2 P1s and ratio ≤ 40%): fix the findings inline, push, and
merge the PR.

Dismiss Copilot comments that rehash addressed issues, debate design with
no clear improvement, or flag impossible edge cases. These do not count
toward the threshold.

### Why these thresholds

AI reviewers have higher false-positive rates than human reviewers, so a
ratio-based gate (not "any P1 blocks") avoids infinite loops from
hallucinated findings. The absolute count (3+) catches cases where few
total findings exist but most are critical.

## Git Workflow

- **origin**: `ornlneutronimaging/NEREIDS` — primary repo, issues, releases,
  main branch.  All branches and PRs are pushed here directly.
- **upstream** (secondary): `KedoKudo/NEREIDS` — personal fork, kept as
  backup remote.

## Reference Codebases (siblings of this repo)

- `../SAMMY`    — physics reference (resonance formalism, SAMMY source)
- `../PLEIADES` — ORNL data normalisation helpers, ENDF retrieval
- `../trinidi`  — sparsity-handling reference (Purdue/LANL, mostly abandoned)
