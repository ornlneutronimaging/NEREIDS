---
name: codex-review
description: Run Codex CLI external review on worktree branches for independent second opinion
user-invokable: true
---

# Codex External Review

Run OpenAI's Codex CLI (`codex review`) on code changes for an independent
second opinion from a different AI provider.

## Arguments

- No arguments: review all active worktrees under `.claude/worktrees/`
- Branch name: review a specific branch (e.g., `/codex-review fix/my-branch`)

## Execution

Discover worktrees, then launch one `Bash` command per worktree **in parallel**
(background mode).

### Codex CLI invocation (with fallback chain)

The Codex CLI changes syntax between versions. Try these in order:

```bash
# Primary — current stable syntax (no custom prompt with --base)
cd {worktree_path} && codex review --base main 2>&1

# If "unexpected argument" or similar → try without --base
cd {worktree_path} && codex review 2>&1
```

**Known CLI pitfalls** (as of Feb 2026):
- `--approval` flag was removed — do not use
- `--base` and positional `[PROMPT]` are mutually exclusive — use `--base main` alone
- Piping a prompt via stdin (`echo "..." | codex review --base main -`) does not work
- Notion MCP was removed from Codex config (caused `invalid_grant` noise).
  Codex review only needs local file access — no MCP servers required.

### Failure handling

Codex may fail due to:
- CLI syntax change — try fallback invocations above
- Network issues — retry once, then skip
- Xcode license (macOS) — inform user to run `sudo xcodebuild -license accept`
- Timeout — note and skip

If Codex is entirely unavailable, note the failure and continue. Codex is
supplementary, not blocking — the Claude self-audit is the primary review.

## Output

For each branch, extract and present actionable findings:

```
## Branch: fix/endf-parser-hardening

### Codex Findings
- **P1**: NAPS not propagated for URR ranges (parser.rs)
- **P2**: Multi-MAT detection comment could be clearer

### No actionable defects (if clean)
```

Ignore boilerplate Codex output (setup messages, model info). Focus on the
actual review content.
