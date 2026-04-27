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

### External Review (Codex CLI) — TEMPORARILY DISABLED

**Status (April 2026):** Codex review is currently driven manually by the
user outside this pipeline. Skip the automated `codex exec` step until the
local codex-cli access issue is resolved (the installed binary is too old
for the current API and ChatGPT-account model gating prevents falling back
to an older model — see "Known pitfalls" below). When the user's local
codex-cli is upgraded and accepted by the API, re-enable by removing this
notice and uncommenting the launch step.

When re-enabled, also launch one `Bash` command per worktree in the **same
message** as the Claude self-audit, unless `--skip-codex` is in
`$ARGUMENTS`.

There is **no `codex review` subcommand** in current codex-cli (verified
against 0.46 and 0.125, April 2026). The slash command `/review` exists
but is interactive-only — it cannot be driven from `codex exec`. The
canonical headless invocation is `codex exec` with an explicit review
prompt; a native `codex exec review` is requested in
[openai/codex#6432](https://github.com/openai/codex/issues/6432) but
not yet shipped.

Use this pattern (one Bash call per worktree):

```bash
codex exec --sandbox read-only --skip-git-repo-check \
  -C {worktree_path} \
  --output-last-message /tmp/codex-review-{branch_slug}.md \
  "$(cat <<'PROMPT'
You are reviewing the changes on the current branch (HEAD) against `main`
in the NEREIDS repository.

1. Run `git diff main...HEAD` to see all changes.
2. Read each changed file in full.
3. Audit for:
   - Logic bugs, panics (unwrap, expect, array indexing)
   - Missing input validation
   - Numerical stability (division by zero, NaN propagation, overflow)
   - Physics correctness (for nereids-physics / nereids-endf)
   - API consistency with existing patterns
   - Edge cases (empty inputs, zero counts, exactly-determined systems)
4. Report findings as:
   - **P1** (must fix) — correctness bugs, panics, data corruption
   - **P2** (should fix) — robustness, style, minor improvements
   - Include `file:line` references for each finding.
5. If you find nothing significant, say so explicitly. Be terse.
PROMPT
)"
```

Then read the file at `/tmp/codex-review-{branch_slug}.md` for the final
review verdict; the JSONL on stdout is the streaming transcript and is
mostly noise for our purposes.

**Why these flags:**
- `--sandbox read-only` — review only reads code; no need for write access.
- `--skip-git-repo-check` — defensive; we always invoke from inside a repo
  but this avoids friction in nested-worktree edge cases.
- `-C {worktree_path}` — sets working dir explicitly so `git diff main...HEAD`
  resolves correctly per worktree.
- `--output-last-message <file>` — captures the agent's final message
  cleanly; far easier than parsing JSONL.

**Known pitfalls** (as of April 2026, codex-cli 0.46 → 0.125):

- The `review` subcommand was removed (or was never a thing in 0.x). Do
  not use `codex review --base main` — it errors with
  `unexpected argument '--base' found / Usage: codex <PROMPT>`.
- Slash commands (`/review`, `/test`, etc.) work only in interactive
  TUI sessions; they cannot be invoked from `codex exec`.
- The default model in codex-cli config (`~/.codex/config.toml`) must
  match what the local CLI binary supports. As of 0.46 the default
  `gpt-5.5` is rejected as *"requires a newer version of Codex"*; as
  of 0.122+ it works. If the binary is too old, upgrade before relying
  on Codex review — overriding with `-m gpt-5` does NOT help on
  ChatGPT-account auth (returns *"not supported when using Codex with
  a ChatGPT account"*). Per-version compatibility is unstable; keep
  the binary current.
- `codex exec` reads the prompt from stdin if you pass `-` or omit the
  positional, but heredoc-injected positional prompts (as above) are
  the most reliable form across versions.
- Avoid `--full-auto` for review — it grants `workspace-write` sandbox,
  which is broader than the read-only review needs.

If Codex fails (network, license, model rejection, binary out of date),
note the failure and continue. Codex is supplementary, not blocking. The
Claude self-audit is the load-bearing reviewer; Codex provides
cross-confirmation when available.

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
