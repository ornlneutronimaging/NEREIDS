---
name: comment-audit
description: Mechanical gate on comments before a commit — strip, regenerate with a tool-less writer, judge, delete what the code already says
user-invokable: true
---

# Comment audit

A comment an agent can regenerate from the stripped code carries no
information and is deleted. A comment it cannot regenerate is kept only when
it is provenance; otherwise it is flagged as a place where code and comment
disagree.

Run before every commit on the changed Rust files. Nothing here needs
judgment except the flagged pairs. The script and its tests live in this
folder; run the tests with
`pixi run python -m pytest .claude/skills/comment-audit/`.

## Arguments

- none: audit the comments the working tree changed (`git diff HEAD`).
- a git range (`main...HEAD`): audit the comments that range added.
- `--whole-file`: audit every comment in each changed file.
- `--report-only`: do not apply deletions.

## Steps

1. Changed Rust files:

   ```
   git diff --name-only <range or HEAD> -- '*.rs'
   ```

2. Strip and bundle into the scratchpad, `$S`:

   ```
   python3 .claude/skills/comment-audit/comment_audit.py strip <files> --out $S/audit --root .
   python3 .claude/skills/comment-audit/comment_audit.py bundle <files> --out $S/audit --root . --range <range> --repo . > $S/audit/args.json
   ```

   Omit `--range` for `--whole-file`. Every file's stripped text and comment
   list go into the workflow's arguments; the writer sees only that text.

3. Run the workflow with the contents of `args.json` as `args`:

   ```
   Workflow(name="comment-audit", args=<args.json>)
   ```

   Named workflows are registered at session start; in the session that
   adds or edits the script, pass
   `scriptPath=".claude/workflows/comment-audit.js"` instead of `name`.

   It returns `{ perFile: [{ file, scoped, absent, same, keep, flagged }], totals }`.
   `absent` and `same` are ids to delete; `keep` are provenance; `flagged`
   carry the original, the regenerated comment and the judge's reason.

   Then check that no writer or judge touched a file, using the transcript
   directory the tool result names:

   ```
   python3 .claude/skills/comment-audit/comment_audit.py verify-isolation <transcript dir>
   ```

   A non-zero exit means an agent called a tool other than structured
   output; the run is invalid and nothing from it is applied.

4. Apply every `absent` and `same` id, then format and check:

   ```
   python3 .claude/skills/comment-audit/comment_audit.py apply <file> --comments $S/audit/<file>.comments.json --delete <ids>
   cargo fmt --all && cargo check --workspace --exclude nereids-python --all-targets
   ```

5. Write the result to `$S/audit/result.json` and print the table:

   ```
   python3 .claude/skills/comment-audit/comment_audit.py report $S/audit/result.json
   ```

   For each flagged pair propose one disposition: delete the comment, or
   change the code so it says what the comment says. Deletion is the
   default. STOP for the user when any pair is flagged; with none, report the
   totals and continue.

## Rules

- The writer and judge agent types have no file access (`tools: ListAgents`;
  an empty `tools:` list means every tool). Do not substitute a type with
  file access, and treat a failed isolation check as an invalid run: the
  measurement holds only when the writer cannot see the original.
- Never edit a comment to make it survive the gate. A comment that fails is
  deleted or the code is changed.
- Deletions are applied by the script, never by hand.
