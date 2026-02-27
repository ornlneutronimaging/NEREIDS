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
(background mode):

```bash
cd {worktree_path} && codex review --base main 2>&1
```

### Failure handling

Codex may fail due to:
- Network issues — retry once, then skip
- Xcode license (macOS) — inform user to run `sudo xcodebuild -license accept`
- Timeout — note and skip

If Codex is unavailable, offer to fall back to Gemini CLI:
```bash
cd {worktree_path} && git diff main...HEAD | gemini -p "Review this diff for a Rust neutron physics library. Focus on panics, validation gaps, edge cases, numerical stability. Report findings as P1 (must fix) / P2 (should fix) with file:line references."
```

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
