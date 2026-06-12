<!--
NEREIDS PR template — keep this template visible in your PR body.
Replace each section's placeholder with your specific content.

The LOC-delta and existing-logic check exist because earlier fix sprints
accumulated duck-tape patches that a later architecture audit had to
unwind. Filling these in honestly is the load-bearing control, not the
template existing.
-->

## Summary

<!-- 1-3 sentences: what changes, and why. Cite the issue or audit finding that motivated it. -->

## LOC delta

<!-- Run: git diff --stat <merge-base>..HEAD | tail -1 -->

- inserted: +N
- deleted:  -M
- net:      ±K

## Existing-logic check (mandatory for refactor PRs; informational for feature PRs)

- [ ] I searched the crate(s) I touched for similar logic before adding new code
- [ ] If similar logic exists, this PR refactors/consolidates it rather than duplicating
- [ ] If this PR has positive net LOC, the new code is load-bearing structure (a newtype carrying an invariant, a centralized validator replacing scattered ones, a feature, etc.) — not a defensive guard or duck-tape patch around a symptom
- [ ] No new `validate_*` or `*_helper` function was added if a similar one already exists in the same crate

If any box above could not be checked, the PR body explains why.

## Test plan

<!--
- Which existing tests still pass (cargo test, pixi run test-python)
- Which tests were added/updated and what regime they exercise (especially: the regime that would have exposed the bug pre-fix)
- For physics-correctness changes: which SAMMY/ENDF/NJOY primary source was checked
-->

## Linked issues / audit references

<!--
- Closes #N (must be a real issue; do not invent numbers)
- Refs #M (related but not closed)
- Name the audit finding (e.g. "architecture-audit F13") if one motivated the change
-->

---

<!--
For AI-assisted PRs:
- Cite the session ID and name the audit findings the change is based on
- The agent must include the LOC-delta numbers; "I'll let you compute them" is not acceptable

Trailer for AI-coauthored commits is added by ./scripts/worktree-commit.sh.
-->
