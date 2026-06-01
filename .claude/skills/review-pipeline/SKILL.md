---
name: review-pipeline
description: Multi-stage iterative review pipeline across all active worktree branches
user-invokable: true
---

# Multi-Stage Review Pipeline

Run an iterative review pipeline on active worktree branches. Repeats
until zero P0s and P1s remain or the iteration limit is reached.

**This is the ONLY review mechanism.** Do not substitute with ad-hoc
self-review agents, custom subagents, or any improvised review approach.
The standalone `/self-review` and `/codex-review` skills have been merged
into this single pipeline. When the user says "review", "run reviews",
or any variation, invoke THIS skill.

## Architecture: deterministic core + human-gated orchestration

The non-interactive heart of each review round — **discover branches →
two-LLM-family review → consolidate** — is the `review-core` **Workflow**
(`.claude/workflows/review-core.js`), which runs the shared
`dual-family-review` engine. This skill is the **orchestrator**: it invokes
`review-core` once per round and owns everything interactive or
state-changing (the user gates, fixes, pushes, the Copilot phase, merges,
and issue-filing).

Why this split — and why the gates live here, not in the workflow:

- **Consistent format across models.** Inside `review-core`, every reviewer
  (Claude finder, Codex finder, and every cross-family verifier) emits the
  **same schema**; cross-confirmation and dedup are deterministic JS, not the
  main agent eyeballing prose. Claude and Codex findings come back in one
  shape every round.
- **No silently-skipped steps.** discover → find(2 families) → verify →
  consolidate is JS control flow run for every branch every round — not prose
  an LLM may under-execute. The deferred-P2 list comes back **as data**
  (Step 9 can no longer be "forgotten").
- **A Workflow cannot pause for user input.** It runs unattended to
  completion. So every mandatory STOP gate must fall on a workflow boundary —
  which is exactly why fix/push/merge and the three gates stay in this skill,
  with `review-core` invoked between them.

Invoking `review-core` from this skill is the sanctioned Workflow opt-in
(a skill whose instructions call the Workflow tool).

## Arguments

- No arguments: `review-core` auto-discovers all `.claude/worktrees/*/`
  branches diverged from main.
- Branch name: scope to a single branch — pass it through as
  `branches: ["<name>"]`.
- `--skip-codex`: pass `skipCodex: true` (single-LLM-family; all findings
  return as NEEDS-VERIFICATION).

## Iteration Policy

- **Goal**: Zero P0s and P1s before pushing to remote.
- **Max iterations**: 4 per branch.
- **Escalation**: If P0s/P1s persist after 4 rounds, stop and report to the user.
  Do NOT attempt a 5th round. The user must decide whether to continue,
  restructure the task, or conduct manual review.

Track the current round number and report it in each consolidation
(e.g., "Review Round 2 of 4"). Pass it to `review-core` as `round: N`.

---

## Step 1: Invoke `review-core` (one round)

Call the **Workflow** tool:

```
Workflow(name="review-core", args={
  round: <N>,                      // 1 on the first round
  base: "main",
  branches: <["branch"] | omit to auto-discover>,
  skipCodex: <true if --skip-codex>,
  priorFindings: <prev round's findings, or omit on round 1>,  // see Step 5
})
```

`review-core` runs in the background and returns a structured payload:

```
{ meta: { round, base, repoRoot, headSha, codexUsable, sammyRoot, isWorktree },
  mergeOrder: ["branch", ...],            // smallest-diff-first; parallel-safe if no overlap
  overlaps:   [{ a, b, sharedFiles[] }],
  perBranch:  [{ branch, crates, claudeAssessment, codexAssessment, tierCounts,
                 verifiedP0[], verifiedP1[], needsVerification[], refuted[],
                 p2s[] (each with .disposition: fix-now|defer), circular[], recurring[] }],
  deferredP2s: [...],                      // P2s outside the changed crates (Step 9)
  counts: { branches, verifiedP0, verifiedP1, needsVerification, refuted, p2, recurring } }
```

Each finding carries `status` (VERIFIED = ≥2 LLM families agreed;
NEEDS-VERIFICATION = single-family / split / codex unavailable; REFUTED =
cross-family refuted), `file`, `line`, `claim`, `reasoning`, `primarySource`,
`suggestedFix`, `confidence`, and `recurring` (matched a prior round).

**Codex note:** `review-core` runs Codex as the second family via the engine.
If `meta.codexUsable` is false (codex absent / `--skip-codex`), the round is
single-family and every **P0/P1** finding is NEEDS-VERIFICATION (P2s are
single-family-reported either way) — say so. If codex was
expected but consistently fails, check `codex --version` / upgrade
(`brew upgrade codex` or `npm i -g @openai/codex@latest`); do not work around
with model overrides. Codex is supplementary, not blocking.

---

## Step 2: Consolidation Gate

Render the returned payload as a consistent per-branch report:

1. **Per-branch table**: for each branch, Claude vs Codex tier counts and
   VERIFIED P0 / VERIFIED P1 / NEEDS-VERIFICATION / REFUTED / P2 / RECURRING.
2. **Findings detail**, grouped by branch and confidence:
   - **VERIFIED** (cross-confirmed) — highest confidence, fix now.
   - **NEEDS-VERIFICATION** (single-family) — call out that these did NOT meet
     the ≥2-LLM-family bar.
   - **REFUTED** — list with the refuter's reasoning so they are not
     re-litigated.
3. **Suggested disposition** (from each finding's `.disposition` / tier) — the
   workflow proposes fix-now / defer / dismiss; **the user decides**:
   - **Fix now** — VERIFIED P0/P1 and same-crate P2s.
   - **Defer** — P2s outside the changed crate(s) (already in `deferredP2s`).
   - **Dismiss** — false positives / impossible edge cases.
4. **Suggested Merge Order** from `meta`/`mergeOrder` (+ any `overlaps`).
5. Report the round: "Review Round N of 4".
6. **RECURRING**: any finding tagged `recurring` reappeared after a prior
   "fix" — flag it explicitly; the user must decide the approach.

### P2 Deferral Discipline

If the PR's purpose is P2 burndown / tech-debt reduction, "Defer" is
restricted to a *different crate or subsystem* than the one being fixed.
`review-core` already marks same-crate P2s `disposition: fix-now`; honor that —
do not defer same-crate P2s, or debt accrues faster than it is paid down.

**MANDATORY GATE — present the consolidation and STOP.** Do NOT proceed to
Step 3 without user approval. The user must tell you which findings to fix.
End your turn after presenting the report.

---

## Step 3: Fix

After the user approves the fix list, launch one fix subagent per worktree
**in parallel**, using `.claude/templates/fix-subagent-prompt.md` (it encodes
the DRY pre-step and no-scope-dodging rules). Each fix agent must:

1. Apply the approved fixes.
2. **Check downstream consumers** — if a fix changes a public API, grep for
   all call sites across the workspace and update them.
3. Run `cargo fmt --all`.
4. Run `cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings`.
5. Run `cargo test --workspace --exclude nereids-python`.
6. For physics-correctness fixes (SLBW/MLBW/RM/RML, Doppler, resolution,
   fitting math), also run `pixi run test-python` — `cargo test` cannot catch
   `pytest.approx` baseline regressions.
7. Commit with `scripts/worktree-commit.sh <worktree-name> "<msg>" [files]`
   (GPG-signed).

---

## Step 4: Verify & Push

After all fix agents complete:
1. Verify each worktree has clean `git status`.
2. If a fix touched a shared symbol, `cargo check` after any rebase before push.
3. Push each branch: `git push origin {branch}`.
4. Report commit hashes and branch status.

---

## Step 5: Iteration Decision

After pushing, decide:

- **Zero P0s and zero P1s this round?** → Phase A complete. Proceed to Step 6 (Phase B).
- **P0s/P1s found and fixed, round < 4?** → Loop back to Step 1 for round N+1.
  Pass `priorFindings` = this round's findings (each as
  `{branch, file, line, title}`) so `review-core` tags **RECURRING**.
- **Round == 4 and P0s/P1s still found?** → STOP. Report: "Iteration limit
  reached (4 rounds). P0s/P1s persist — escalating to human."

(P0 is the must-fix tier the `dual-family-review` engine assigns; the gate must
cover P0+P1, not P1 alone — a verified P0 with zero P1s must NOT pass.)

**Re-run between rounds, not just once.** File overlap can appear mid-pipeline
(a fix lands in a file another branch also touches); the fresh `mergeOrder`
each round reflects this.

---

## Step 6: Phase B — Copilot Review (after push)

After Phase A completes (zero P0s/P1s) and branches are pushed:

1. Inform the user Phase A is complete and branches are pushed. Ask them to
   trigger Copilot review on GitHub. **STOP and wait.**
2. When the user says Copilot reviews are in, fetch comments:

   ```bash
   pixi run copilot-reviews {pr_numbers...} --dedup
   ```

3. Classify each Copilot comment as P1 or P2.
4. **Decision criteria**:
   - 3+ P1s OR P1 ratio > 40% → re-iterate (back to Step 1).
   - Otherwise → fix P2s inline, commit, push.
5. Dismiss Copilot comments that rehash already-addressed issues or flag
   impossible edge cases.
6. Present the Copilot resolution summary to the user.

---

## Step 7: Pre-Merge Checkpoint

**MANDATORY: End your turn here and wait for user approval.**

Present a concise summary table:

```markdown
### Pre-Merge Summary — Batch {name}

| PR | Branch | Issue | Key Changes | Review Status |
|----|--------|-------|-------------|---------------|
| #{n} | {branch} | #{issue} | {1-line summary} | Phase A ✓ Phase B ✓ |

**Merge order**: {from review-core meta.mergeOrder}
**Review rounds**: Phase A: {N} round(s), Phase B: {N} Copilot comment(s)
**Findings resolved**: {W} P0s fixed, {X} P1s fixed, {Y} P2s fixed, {Z} P2s deferred
**Tests on branches**: {N} Rust tests — all pass
```

**Do NOT run `gh pr merge` until the user responds with explicit approval.**

---

## Step 8: Merge & Post-Merge

After the user approves:

1. Merge PRs in `mergeOrder` using `gh pr merge --squash --delete-branch`.
2. Clean up worktrees: `git worktree remove {path} --force` per merged branch.
3. Delete local branches: `git branch -D {branch}` per branch.
4. Run `/post-merge` (pulls main, `cargo clean && pixi run build`, workspace
   tests, Python tests, issue verification, memory updates).

**IMPORTANT**: `pixi run build` must run first after `cargo clean` — it catches
cross-PR signature mismatches that per-branch reviews miss.

---

## Step 9: Track Deferred P2 Findings

**Do NOT skip this step.** `review-core` returns `deferredP2s` — file them so
nothing is lost:

1. Group `deferredP2s` by branch/crate.
2. Create one issue per group with `file:line` references.
3. Add to the project tracker (project #8).
4. Report the created issue numbers to the user.

---

## Subagent Prompt Requirements

When launching fix subagents, ALWAYS include (and use
`.claude/templates/fix-subagent-prompt.md`):

- **Tooling**: "Use `pixi run build` / `pixi run test-python` — never raw
  `maturin develop` or `pip install`."
- **Commits**: "Use `scripts/worktree-commit.sh <worktree-name> '<message>' [files]`
  for all commits."
- **GitHub issues**: "Use `pixi run gh-issues` for issue/PR queries."
- **Pattern matching**: "Match patterns already used in the file you're editing."
- **DRY pre-step**: "ripgrep for existing logic before adding a new helper."
- **Pre-commit**: "Run the pre-commit checklist from CLAUDE.md."
