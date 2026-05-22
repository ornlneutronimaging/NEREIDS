# Fix-subagent prompt template

Use this template when prompting a subagent to apply a fix. It encodes the
DRY governance from `feedback_fixes_must_consolidate_not_duplicate.md` and
the no-scope-dodging rule from `feedback_no_scope_dodging.md`. Adapt the
[bracketed] sections per fix; do NOT delete the mandatory pre-steps.

---

You are applying a code fix to NEREIDS at commit `[HEAD-or-branch-head]`.

**Worktree (cd here first):** `[absolute path]`
**Branch:** `[branch name]`

## Mandatory pre-step 1 — search for existing logic

BEFORE writing any new code:

1. Search the crate(s) you'll touch for existing logic that matches the
   problem shape. Use ripgrep:
   ```
   rg -n '[pattern matching the problem shape]' crates/[crate]/src/
   ```
2. If similar logic already exists:
   - PREFER refactoring or extending the existing one
   - If the existing one is in the wrong place, MOVE it (and update callers)
   - DO NOT add a parallel implementation
3. If NO similar logic exists, prefer:
   - A newtype that carries the invariant (validated once, used freely)
   - A single shared function (called from N sites) over N inline checks
   - Deleting a buggy code path over guarding it

**State explicitly in your final report**: *"I searched for X using
`rg ...`. Found Y in `file:line` / found nothing similar. Therefore I
[refactored existing | added new because of justified absence]."*

## Mandatory pre-step 2 — LOC-delta projection

Before writing the fix, estimate:
- inserted: +N
- deleted:  -M
- net:      ±K

If NET > 0 for what's supposed to be a fix (not a feature), pause and ask
yourself:

> "Could this same correctness outcome be achieved by removing code instead
> of adding it?"

If yes, restructure the fix to do that. If no (because the problem genuinely
requires new structure — a real newtype, a consolidating function, etc.),
proceed and document why the addition is load-bearing.

## Your specific fix scope

[Describe the bug or refactor. Cite file:line. Cite primary source if
applicable: SAMMY src/, ENDF-6 Formats Manual, NJOY repo, PyO3 docs, etc.]

[For physics-correctness fixes, include the primary-source citation
verbatim. For refactors, link to the audit finding in
`.research/audit-r{N}-architecture/...`.]

## Process

1. cd to the worktree
2. Run the pre-step searches (Mandatory pre-step 1); document findings
3. Apply the fix
4. Run pre-commit gates:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings
   cargo test --workspace --exclude nereids-python
   # ONLY if touching physics-correctness code (SLBW/MLBW/RM/RML, Doppler,
   # resolution, fitting math): also run
   pixi run test-python
   ```
5. Commit via `./scripts/worktree-commit.sh <worktree-name> "<message>" <files>`
6. Stop. Report:
   - Commit hash
   - Actual LOC delta (run `git diff --stat <parent>..HEAD | tail -1`)
   - Mandatory pre-step 1 search findings (the exact `rg` commands and results)
   - Any adjacent issues you noticed but did NOT fix (so the parent agent
     can flag them for follow-up; do not silently expand scope)

## Strict rules

- **No GitHub-meta in code comments** (no `PR #N`, no `round-N review:`).
  Why-the-code-is-shaped-this-way goes in code comments; when/which-PR
  goes in commit messages and PR descriptions.
- **No new helpers without ≥1 deletion** (DRY governance). If you add a
  `validate_X`, you must delete the inline `X` checks at the call sites.
- **No scope-cutting.** Do the full fix as specified. Adjacent issues
  outside scope go in the "noticed but didn't fix" report, never into
  silent dropped scope.
- **No bypassing pre-commit gates** (no `--no-verify`, no skipped tests).
- **No skipping GPG signing.** Use `./scripts/worktree-commit.sh` which
  handles GPG.
- **No touching files outside your declared scope.** If the fix requires
  touching a sibling file, declare it explicitly in the scope section
  before proceeding.
