---
name: post-merge
description: Post-merge integration gate — cleanup, build, test, verify issues, update memory
user-invokable: true
---

# Post-Merge Pipeline

Run after all PRs in a batch are merged. Verifies the merged main is clean,
closes out tracking work, and updates project memory.

## Arguments

- No arguments: run the full pipeline
- `--skip-cleanup`: skip worktree/branch cleanup (already done manually)
- `--skip-memory`: skip the memory update step

## Step 1: Fetch & Update Local

```bash
git fetch origin
git checkout main
git pull origin main
```

If not already on main, stash or warn before switching.

## Step 2: Worktree & Branch Cleanup

1. List worktrees: `git worktree list`
2. For each worktree under `.Codex/worktrees/`:
   - Check if its branch is merged into main (`git branch --merged main`)
   - If merged: `git worktree remove {path}` then `git branch -d {branch}`
   - If not merged: warn the user — do NOT force-delete
3. Prune stale remote-tracking branches:
   ```bash
   git fetch --prune origin
   git branch -vv | grep ': gone]' | awk '{print $1}'
   ```
   Delete local branches whose remote is gone (only if merged).

Skip this step if `--skip-cleanup` is in `$ARGUMENTS`.

## Step 3: Clean Build

**IMPORTANT**: Always `cargo clean` before `pixi run build` after merging
multiple PRs. Stale incremental artifacts from worktrees cause spurious
"can't find crate" errors with maturin.

```bash
pixi run clean
pixi run build
```

If `pixi run build` fails, diagnose and fix on main immediately.

## Step 4: Rust Tests

```bash
cargo test --workspace --exclude nereids-python
```

All tests must pass. If failures occur:
1. Diagnose whether the failure is from our changes or pre-existing
2. Fix on main, commit with descriptive message
3. Re-run tests to confirm

## Step 5: Python Tests

```bash
pixi run test-python
```

Common failure mode: Python test `match=` regex patterns don't align with
changed Rust error messages. Fix the test patterns to match actual messages.

If failures occur, fix, commit, push, and re-run.

## Step 6: Push Fixes (if any)

If Steps 3-5 required fixes on main:

```bash
git push origin main
```

## Step 7: Verify Issue Closure

For each issue that was part of this batch:

```bash
gh issue view {number} --json state --jq '.state'
```

- If `CLOSED`: confirmed (auto-closed by PR merge)
- If `OPEN`: close with comment referencing the merged PR:
  ```bash
  gh issue close {number} -c "Resolved in PR #{pr_number}"
  ```

Also check if any parent epics should be closed (all sub-issues done).

## Step 7.5: Add Issues to Project Tracker

Add all closed issues AND any newly created deferred-P2 issues to the
NEREIDS Development project (number 8):

```bash
gh project item-add 8 --owner ornlneutronimaging \
  --url https://github.com/ornlneutronimaging/NEREIDS/issues/{number}
```

Repeat for every issue number involved in this batch (closed + deferred).
This keeps the project board in sync for velocity tracking.

## Step 8: Update Project Memory

Unless `--skip-memory` is in `$ARGUMENTS`:

1. Read `MEMORY.md`
2. Update the "Project Phase Status" section:
   - Add the new batch entry with commit hash, PR numbers, closed issues
   - Update the test count line
3. Add any new "Multi-AI Review Pipeline Lessons" discovered during this batch

## Step 9: Report

Present a summary table:

| Gate | Status |
|------|--------|
| `pixi run build` | PASS/FAIL |
| `cargo test` | PASS — N tests (M ignored) |
| `pixi run test-python` | PASS — N tests |
| Issues closed | #X, #Y, #Z |
| Fixes on main | N commits (if any) |

If all gates pass: "Post-merge integration complete. Main is clean."
