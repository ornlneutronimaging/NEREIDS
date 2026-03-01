---
name: review-pipeline
description: Multi-stage iterative review pipeline across all active worktree branches
user-invokable: true
---

# Multi-Stage Review Pipeline

Run an iterative review pipeline on active worktree branches. Repeats
until zero P1s remain or the iteration limit is reached.

**This is the ONLY review mechanism.** Do not substitute with ad-hoc
self-review agents, custom subagents, or any improvised review approach.
The standalone `/self-review` and `/codex-review` skills have been merged
into this single pipeline. When the user says "review", "run reviews",
or any variation, invoke THIS skill.

## Arguments

- No arguments: auto-discover all `.claude/worktrees/*/` branches diverged from main
- Branch name: scope to a single branch (e.g., `review-pipeline fix/my-branch`)
- `--skip-codex`: skip the Codex external review stage

## Iteration Policy

- **Goal**: Zero P1s before pushing to remote
- **Max iterations**: 4 per branch
- **Escalation**: If P1s persist after 4 rounds, stop and report to the user.
  Do NOT attempt a 5th round. The user must decide whether to continue,
  restructure the task, or conduct manual review.

Track the current iteration number and report it in each consolidation
(e.g., "Round 2 of 4").

---

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

---

## Steps 2 + 3: Self-Audit & External Review (launch together)

Launch **all** review tasks in a **single message** (background mode) so
they run concurrently. N worktrees produce up to 2N parallel tasks
(N Claude subagents + N Codex commands).

### Self-Audit (Claude Subagents)

Launch one `Agent(subagent_type="general-purpose", run_in_background=true)`
per worktree with this prompt:

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

Unless `--skip-codex` is in `$ARGUMENTS`, also launch one `Bash` command
per worktree in the **same message**:

```bash
cd {worktree_path} && codex review --base main 2>&1
```

**Codex CLI fallback chain** (syntax changes between versions):
1. `cd {path} && codex review --base main 2>&1` (primary)
2. `cd {path} && codex review 2>&1` (fallback if --base fails)

**Known pitfalls** (as of Feb 2026):
- `--approval` flag removed — do not use
- `--base` and positional `[PROMPT]` are mutually exclusive
- Piping via stdin does not work
- Notion MCP was removed from Codex config

If Codex fails (network, license, timeout), note the failure and continue.
Codex is supplementary, not blocking.

---

## Step 4: Consolidate Findings

After all reviews complete:

1. Collect self-audit findings (from Agent results) and Codex findings (from Bash output)
2. Merge into a unified report grouped by worktree/branch:
   - **Cross-confirmed** issues (found by both Claude and Codex) — highest confidence
   - **Claude-only** issues
   - **Codex-only** issues
3. For each finding, classify as:
   - **Fix now** — P1s and high-confidence P2s
   - **Defer** — P2s genuinely out of scope (different crate/subsystem,
     pre-existing issue not introduced by this PR)
   - **Dismiss** — false positives, style-only, or impossible edge cases

### P2 Deferral Discipline

**IMPORTANT**: If the PR's purpose is P2 burndown or tech debt reduction,
the "Defer" category is restricted to findings in a *different crate or
subsystem* than the one being fixed. Same-crate P2s MUST be classified as
"Fix now" — otherwise P2 debt accumulates faster than it is paid down.

4. Report the iteration number: "Review Round N of 4"
5. Include the **Suggested Merge Order** from Step 1
6. **Present the consolidated report to the user and STOP.**

**MANDATORY GATE**: Do NOT proceed to Step 5 without user approval.
The user must review the consolidation and tell you which findings to fix.
End your turn after presenting the report.

**Oscillating findings**: If a finding reappears after being "fixed", flag
it as **RECURRING** — the user must decide the approach.

---

## Step 5: Fix

After user approves the fix list, launch one
`Agent(subagent_type="general-purpose")` per worktree **in parallel**.

Each fix agent must:
1. Apply the approved fixes
2. **Check downstream consumers** — if the fix changes a public API,
   grep for all call sites across the workspace and update them
3. Run `cargo fmt --all`
4. Run `cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings`
5. Run `cargo test --workspace --exclude nereids-python`
6. Commit with `scripts/worktree-commit.sh` (GPG-signed)

## Step 6: Verify & Push

After all fix agents complete:
1. Verify each worktree has clean `git status`
2. Push each branch: `git push origin {branch}`
3. Report commit hashes and branch status

---

## Step 7: Iteration Decision

After pushing, check:

- **Zero P1s found this round?** → Phase A complete. Proceed to Phase B.
- **P1s found and fixed, iteration < 4?** → Loop back to Step 2.
- **Iteration == 4 and P1s still found?** → STOP. Report:
  "Iteration limit reached (4 rounds). P1s persist — escalating to human."

---

## Step 8: Phase B — Copilot Review (after push)

After Phase A completes (zero P1s) and branches are pushed:

1. Inform the user that Phase A is complete and branches are pushed.
   Ask them to trigger Copilot review on GitHub. **STOP and wait.**
2. When the user says Copilot reviews are in, fetch comments:

```bash
pixi run copilot-reviews {pr_numbers...} --dedup
```

3. Classify each Copilot comment as P1 or P2
4. **Decision criteria**:
   - 3+ P1s OR P1 ratio > 40% → re-iterate (back to Step 2)
   - Otherwise → fix P2s inline, commit, push
5. Dismiss Copilot comments that rehash already-addressed issues or flag
   impossible edge cases
6. Present Copilot resolution summary to the user.

---

## Step 9: Pre-Merge Checkpoint

**MANDATORY: End your turn here and wait for user approval.**

Present the user with a concise summary table:

```markdown
### Pre-Merge Summary — Batch {name}

| PR | Branch | Issue | Key Changes | Review Status |
|----|--------|-------|-------------|---------------|
| #{n} | {branch} | #{issue} | {1-line summary} | Phase A ✓ Phase B ✓ |

**Merge order**: {recommendation}
**Review rounds**: Phase A: {N} round(s), Phase B: {N} Copilot comment(s)
**Findings resolved**: {X} P1s fixed, {Y} P2s fixed, {Z} P2s deferred
**Tests on branches**: {N} Rust tests — all pass

Ready to merge? (User must explicitly approve.)
```

**Do NOT run `gh pr merge` until the user responds with explicit approval.**

---

## Step 10: Merge & Post-Merge

After user approves:

1. Merge PRs in recommended order using `gh pr merge --squash --delete-branch`
2. Clean up worktrees: `git worktree remove {path} --force` for each merged branch
3. Delete local branches: `git branch -D {branch}` for each
4. Run `/post-merge` which handles: pull main, `cargo clean && pixi run build`,
   workspace tests, Python tests, issue verification, memory updates

**IMPORTANT**: `pixi run build` must run first after `cargo clean`. It catches
cross-PR signature mismatches that per-branch reviews miss.

---

## Step 11: Track Deferred P2 Findings

**Do NOT skip this step.** Create GitHub issues for every P2 finding deferred
during consolidation:

1. Group deferred P2s by branch/crate
2. Create one issue per group with `file:line` references
3. Add to project tracker (project #8)
4. Report created issue numbers to the user

---

## Subagent Prompt Requirements

When launching implementation or fix subagents, ALWAYS include:

- **Tooling**: "Use `pixi run build` / `pixi run test-python` — never
  raw `maturin develop` or `pip install`."
- **Commits**: "Use `scripts/worktree-commit.sh <worktree-name> '<message>' [files]`
  for all commits."
- **GitHub issues**: "Use `pixi run gh-issues` for issue/PR queries."
- **Pattern matching**: "Match patterns already used in the file you're editing."
- **Pre-commit**: "Run the pre-commit checklist from CLAUDE.md."
