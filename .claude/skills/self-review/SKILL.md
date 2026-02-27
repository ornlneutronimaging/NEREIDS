---
name: self-review
description: Claude self-audit for logic bugs, panics, validation gaps, and physics correctness
user-invokable: true
---

# Self-Review (Claude Audit)

Run a Claude self-audit on code changes.

## Arguments

- No arguments: audit all active worktrees under `.claude/worktrees/`
- File path: audit a specific file (e.g., `/self-review crates/nereids-fitting/src/lm.rs`)
- Branch name: audit a specific branch (e.g., `/self-review fix/my-branch`)

## Execution

### Single file mode

If `$ARGUMENTS` is a file path, read the file and audit it directly in this
conversation (no subagent needed).

### Multi-worktree mode

If `$ARGUMENTS` is empty or a branch name, discover worktrees and launch one
`Task(subagent_type="general-purpose")` per worktree **in parallel** (background).

Each agent receives:

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

## Output

Present findings grouped by severity, then by file:

```
## Branch: fix/fitting-engine-correctness

### P1 (must fix)
- `lm.rs:523` — dof==0 causes Inf covariance scaling ...

### P2 (should fix)
- `poisson.rs:89` — comment references wrong equation number ...
```
