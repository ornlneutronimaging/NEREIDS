---
name: review-pipeline
description: Multi-stage iterative review pipeline across all active worktree branches
user-invokable: true
---

# Multi-Stage Review Pipeline

Run an iterative review pipeline on active worktree branches. Repeats
until zero P1s remain or the iteration limit is reached.

## Arguments

- No arguments: auto-discover all `.claude/worktrees/*/` branches diverged from main
- Branch name: scope to a single branch (e.g., `review-pipeline fix/my-branch`)
- `--skip-codex`: skip the Codex external review stage

## Iteration Policy (from CLAUDE.md)

- **Goal**: Zero P1s before pushing to remote
- **Max iterations**: 4 per branch
- **Escalation**: If P1s persist after 4 rounds, stop and report to the user.
  Do NOT attempt a 5th round. The user must decide whether to continue,
  restructure the task, or conduct manual review.

Track the current iteration number and report it in each consolidation
(e.g., "Round 2 of 4").

## Step 1: Discover Targets

Identify worktrees to review:

```bash
git worktree list
```

For each worktree under `.claude/worktrees/`, check if the branch has diverged
from main (`git log main..HEAD --oneline`). Skip worktrees with no new commits.

If `$ARGUMENTS` specifies a branch name, filter to just that one.

## Steps 2 + 3: Self-Audit & External Review (launch together)

Launch **all** self-audit and external-review tasks in a **single message**
(background mode) so they run concurrently. In practice this means N worktrees
produce 2N parallel tasks (N Claude subagents + N Codex commands).

### Self-Audit (Claude Subagents)

Launch one `Task(subagent_type="general-purpose")` per worktree:

Each agent receives this prompt template (fill in worktree path and branch):

> You are auditing the branch `{branch}` at `{worktree_path}`.
>
> 1. Run `git diff main...HEAD` to see all changes
> 2. Read each changed file in full
> 3. Audit for:
>    - Logic bugs, panics (`unwrap`, `expect`, array indexing)
>    - Missing input validation
>    - Numerical stability (division by zero, NaN propagation, overflow)
>    - Physics correctness (for nereids-physics/nereids-endf)
>    - API consistency with existing patterns
>    - Edge cases (empty inputs, zero counts, exactly-determined systems)
> 4. Report findings as:
>    - **P1** (must fix) — correctness bugs, panics, data corruption
>    - **P2** (should fix) — robustness, style, minor improvements
>    - Include `file:line` references for each finding
> 5. Run `cargo test --workspace --exclude nereids-python` to verify tests pass
>    (workspace-wide, because changes often ripple across crates)

### External Review (Codex CLI)

Unless `--skip-codex` is in `$ARGUMENTS`, also launch one `Bash` command per
worktree in the same message:

```bash
cd {worktree_path} && codex review --base main 2>&1
```

If Codex fails (network, license, or timeout), note the failure and continue.
Codex is supplementary, not blocking.

**Fallback**: If Codex is unavailable, use Gemini CLI instead:
```bash
cd {worktree_path} && git diff main...HEAD | gemini -p "Review this diff for a Rust neutron physics library. Focus on panics, validation gaps, edge cases, numerical stability. Report as P1/P2 with file:line references."
```

## Step 4: Consolidate Findings

After all reviews complete:

1. Collect self-audit findings (from Task results) and Codex findings (from Bash output)
2. Merge into a unified report grouped by worktree/branch:
   - **Cross-confirmed** issues (found by both Claude and Codex) — highest confidence
   - **Claude-only** issues
   - **Codex-only** issues
3. For each finding, classify as:
   - **Fix now** — P1s and high-confidence P2s
   - **Defer** — P2s that are out of scope, pre-existing, or belong to a different phase
   - **Dismiss** — false positives, style-only, or impossible edge cases
4. Report the iteration number: "Review Round N of 4"
5. **Present the consolidated report to the user** and ask which findings to fix

**Oscillating findings**: If a finding was "fixed" in a previous round but
reappears (or its fix was reverted/reclassified), flag it prominently as
**RECURRING**. This signals a design disagreement between reviewers — the
user must decide the correct approach rather than letting the pipeline
ping-pong. (Lesson: the ENDF LFW check oscillated for 3 rounds before the
user decided to remove it entirely.)

IMPORTANT: Do NOT proceed to fixing without user approval. The consolidation
step is an explicit gate.

## Step 5: Fix

After user approves the fix list, launch one `Task(subagent_type="general-purpose")`
per worktree **in parallel** with specific fix instructions for that branch.

Each fix agent must:
1. Apply the approved fixes
2. **Check downstream consumers** — if the fix changes a public API (return type,
   field visibility, function signature), grep for all call sites across the
   workspace and update them. Common ripple targets: `nereids-python` (PyO3
   bindings), `apps/gui`, and cross-crate callers in `nereids-endf`/`nereids-fitting`.
3. Run `cargo fmt --all`
4. Run `cargo clippy --workspace --exclude nereids-python -- -D warnings`
5. Run `cargo test --workspace --exclude nereids-python`
6. Show `git diff` of changes made

Do NOT commit — leave that for the verification step.

## Step 6: Verify

Launch one `Task(subagent_type="general-purpose")` per worktree **in parallel**
to verify each fix:

Each verification agent must:
1. Read the `git diff` of uncommitted changes
2. Verify each fix is correct (edge cases, mathematical consistency, no regressions)
3. Run tests for the specific crate
4. Report PASS/FAIL per fix with brief justification

## Step 7: Commit & Push

After all verifications pass:

1. For each worktree, commit with `git commit -S` (GPG-signed) and a descriptive message
2. Push each branch to origin
3. Report the commit hashes and branch status

If any verification FAILs, report the issue and ask the user how to proceed
before committing.

## Step 8: Iteration Decision

After committing and pushing, check:

- **Zero P1s found this round?** → Local review complete. Inform the user
  to trigger Copilot review on GitHub (Phase B per CLAUDE.md).
- **P1s found and fixed, iteration < 4?** → Loop back to Step 2 for the
  next round. The new round reviews the cumulative diff (main...HEAD),
  catching both leftover and newly-introduced issues.
- **Iteration == 4 and P1s still found?** → STOP. Report:
  "Iteration limit reached (4 rounds). P1s persist — escalating to human
  review. Consider whether the PR scope is too large or needs task
  decomposition."

## Step 9: Track Deferred P2 Findings

After the pipeline completes (zero P1s), create GitHub issues to track the
P2 findings that were deferred during consolidation:

1. Group deferred P2s by branch/crate
2. Create one issue per group with:
   - Title: `{crate} follow-up: {brief scope}` (e.g., "nereids-endf follow-up: parser robustness")
   - Body: bulleted list of P2 findings with `file:line` references
   - Reference the corresponding merged PR
3. Report the created issue numbers to the user

This ensures P2s aren't forgotten without blocking the merge.

## Pre-commit Checklist (per CLAUDE.md)

Before every commit in every worktree:
```bash
cargo fmt --all
cargo clippy --workspace --exclude nereids-python -- -D warnings
cargo test --workspace --exclude nereids-python
```

## Copilot Re-iteration Gate (Phase B — for reference)

After the developer triggers Copilot review and fetches comments, the
combined threshold from CLAUDE.md applies:

- **Re-iterate** if: 3+ confirmed P1s OR P1 ratio > 40% of findings
- **Merge** if: 0-2 P1s and ratio ≤ 40% (fix inline, push, merge)

This gate is evaluated by the developer + Claude together, not
automatically by this pipeline.
