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

## Step 1: Discover Targets & Merge Order

Identify worktrees to review:

```bash
git worktree list
```

For each worktree under `.claude/worktrees/`, check if the branch has diverged
from main (`git log main..HEAD --oneline`). Skip worktrees with no new commits.

If `$ARGUMENTS` specifies a branch name, filter to just that one.

### File Overlap Analysis (Merge Order)

For each discovered branch, collect changed files:

```bash
cd {worktree_path} && git diff --name-only main...HEAD
```

Build a file overlap matrix and suggest merge order:

1. **No overlap** branches can merge in any order (parallel-safe)
2. **Overlapping** branches should merge in order of increasing diff size
   (smallest first — the larger diff is more likely to need rebasing)
3. Report the suggested merge order in the Step 4 consolidation

This prevents the "last PR has merge conflicts" problem that occurred when
PRs #151 and #152 both modified `poisson.rs`.

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

**Known CLI pitfalls** (as of Feb 2026): `--approval` removed; `--base`
and `[PROMPT]` are mutually exclusive — use `--base main` alone, no custom
prompt. If this syntax fails, try `codex review` without flags.

If Codex fails (network, license, syntax change, or timeout), note the
failure and continue. Codex is supplementary, not blocking.

## Step 4: Consolidate Findings

After all reviews complete:

1. Collect self-audit findings (from Task results) and Codex findings (from Bash output)
2. Merge into a unified report grouped by worktree/branch:
   - **Cross-confirmed** issues (found by both Claude and Codex) — highest confidence
   - **Claude-only** issues
   - **Codex-only** issues
3. For each finding, classify as:
   - **Fix now** — P1s and high-confidence P2s
   - **Defer** — P2s that are genuinely out of scope (different crate/subsystem,
     pre-existing issue not introduced by this PR)
   - **Dismiss** — false positives, style-only, or impossible edge cases

### P2 Deferral Discipline

**IMPORTANT**: If the PR's purpose is P2 burndown or tech debt reduction,
the "Defer" category is restricted to findings in a *different crate or
subsystem* than the one being fixed. Same-crate P2s MUST be classified as
"Fix now" — otherwise P2 debt accumulates faster than it is paid down.

For feature PRs, deferring same-crate P2s is acceptable, but each deferred
P2 must result in a GitHub issue (Step 10).

4. Report the iteration number: "Review Round N of 4"
5. Include the **Suggested Merge Order** from Step 1's file overlap analysis
6. **Present the consolidated report to the user** and ask which findings to fix

**Oscillating findings**: If a finding reappears after being "fixed", flag
it as **RECURRING** — the user must decide the approach.

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

## Step 8.5: Phase B — Copilot Review (after push)

After Phase A completes (zero P1s) and branches are pushed:

1. The user triggers Copilot review on GitHub (manual — cannot be automated)
2. When Copilot reviews are in, fetch comments using the extraction script:

```bash
pixi run copilot-reviews {pr_numbers...} --dedup
```

The `--dedup` flag groups similar comments (word-level Jaccard + shared
backtick identifiers) to collapse Copilot's tendency to repeat the same
suggestion at multiple call sites.

For machine-readable output (useful for automated processing):
```bash
pixi run copilot-reviews {pr_numbers...} --json --dedup
```

3. Classify each Copilot comment as P1 or P2
4. **Decision criteria** (from CLAUDE.md):
   - 3+ P1s OR P1 ratio > 40% → re-iterate (back to Step 2)
   - Otherwise → fix P2s inline and merge
5. If fixing inline: launch fix agents per branch, commit, push
6. Dismiss Copilot comments that rehash already-addressed issues or flag
   impossible edge cases

## Step 9: Pre-Merge Checkpoint (user approval required)

**IMPORTANT: Do NOT merge without explicit user approval.**

Before merging, present the user with a concise summary table:

```markdown
### Pre-Merge Summary — Batch {name}

| PR | Branch | Issue | Key Changes | Review Status |
|----|--------|-------|-------------|---------------|
| #{n} | {branch} | #{issue} | {1-line summary} | Phase A ✓ Phase B ✓ |
| ... | ... | ... | ... | ... |

**Merge order**: {any order / sequential recommendation with rationale}

**Review rounds**: Phase A: {N} round(s), Phase B: {N} Copilot comment(s) ({M} after dedup)

**Findings resolved**: {X} P1s fixed, {Y} P2s fixed inline, {Z} P2s deferred → issues #{a}, #{b}

**Dismissed**: {N} false positives / duplicates / rehash

**Tests on branches**: {N} Rust tests, {M} Python tests — all pass

**Ready to merge?** (Merge will be in order: {order}. Post-merge gate runs afterward.)
```

Wait for the user to confirm before proceeding with any `gh pr merge` commands.
The user may want to:
- Inspect specific PRs on GitHub first
- Request additional changes
- Adjust merge order
- Defer specific PRs to a later batch

Once the user approves, merge in the recommended order using `gh pr merge --squash`.
After all PRs are merged, proceed to Step 10.

## Step 10: Post-Merge Integration Test

After all PRs in the batch are merged, run `/post-merge` which handles:
cleanup, `cargo clean && pixi run build`, workspace tests, Python tests,
issue verification, and memory updates. See the post-merge skill for details.

**IMPORTANT**: `pixi run build` must run first after `cargo clean`. It
compiles the full workspace including PyO3 bindings. When multiple PRs modify
the same function signature, per-branch reviews are blind to the conflict.
`pixi run build` catches these cross-PR integration regressions at compile
time.

## Step 11: Track Deferred P2 Findings

**Do NOT skip this step.** After the pipeline completes (zero P1s and PRs
merged), create GitHub issues for every P2 finding deferred during
consolidation:

1. Group deferred P2s by branch/crate
2. Create one issue per group with:
   - Title: `{crate} follow-up: {brief scope}` (e.g., "nereids-endf follow-up: parser robustness")
   - Body: bulleted list of P2 findings with `file:line` references
   - Reference the corresponding merged PR
3. Report the created issue numbers to the user

This ensures P2s aren't forgotten without blocking the merge.

## Subagent Prompt Requirements

When launching implementation or fix subagents, ALWAYS include:

- **Tooling**: "Use `pixi run build` / `pixi run test-python` — never
  raw `maturin develop` or `pip install`."
- **Commits**: "Use `scripts/worktree-commit.sh <worktree-name> '<message>' [files]`
  for all commits. This validates the working directory and GPG-signs.
  Do NOT use raw `git add && git commit` — it risks wrong-directory commits."
- **GitHub issues**: "Use `pixi run gh-issues` for issue/PR queries
  — never raw `gh` commands or `python scripts/...`."
- **Pattern matching**: "Match patterns already used in the file you're
  editing. Check existing code before introducing new API calls."
- **Pre-commit**: "Run the pre-commit checklist from CLAUDE.md."
