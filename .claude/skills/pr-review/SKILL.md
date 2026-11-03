---
name: pr-review
description: Run one review round on a NEREIDS pull request. Collect the CodeRabbit review and a Claude and a Codex review in one fix table, assess each fix for root cause and recursion risk, get a Codex risk review of the fix plan, apply the plan, push, and report to the maintainer. Use when the user asks to review a PR, to run the review process, or to address review comments.
---

# PR review round

One round collects all review findings, plans the fixes, applies them, and
stops. The maintainer decides if another round is necessary. Do not fix a
review comment when it arrives, also when the desktop app relays it: add it
to the round.

## 1. Prepare

1. Find the PR: the argument, or `gh pr view --json number` on the current
   branch.
2. Use a clean worktree at the PR head (`gh pr view <N> --json headRefOid`).
   Fetch the PR base branch in it, so that the base commit is present.
3. Wait for CodeRabbit. Its review of the PR head is complete when the
   `CodeRabbit` check passes (`gh pr checks <N> --json name,state`).
   CodeRabbit reviews a new PR automatically. When its walkthrough comment
   says that the review was skipped or rate-limited, tell the maintainer.
   Post `@coderabbitai review` only with the maintainer's approval.

## 2. Run the review workflow

Call the Workflow tool with `name: "pr-review"` and these args:

| Argument | Value |
|---|---|
| `pr` | The PR number. |
| `repoRoot` | The absolute path of the worktree from step 1. |
| `scratch` | A new directory in the session scratchpad for each round. |
| `since` | Round 2 and later: the time of the previous round's push (ISO 8601). |
| `sinceSha` | Round 2 and later: the PR head that the previous round reviewed. |
| `sammyRoot` | The SAMMY source tree, when one is available. |

The workflow changes no repository file. It stops with an error when the
worktree is not a clean checkout of the PR head, when the base commit is
missing, or when CodeRabbit has not reviewed the PR head. It returns:

- `table`: one row for each defect, with the original findings, the check by
  another model family, and the fix assessment.
- `plan`: the fix items, one commit each, and the rows in `notFixed`.
- `planReviews`: the Codex risk reviews of the plan.
- `status`:
  - `approved`: the Codex risk review accepts the plan, and each decision
    to leave a row unfixed.
  - `plan-rejected`: the risk review still finds a high risk after one
    revision.
  - `incomplete`: a step failed or lost data; `gaps` gives each cause.
  - `needs-decision`: the plan is empty, and some rows need a decision.
  - `nothing-to-fix`: the risk review accepts that no row needs a change.
- `gaps` and `warnings`: report each one.

## 3. Apply the plan

Apply the plan only when `status` is `approved`. Otherwise go to step 4.

1. Apply the items in order, in the worktree from step 1. Add no work that the
   plan does not contain.
2. For each item, write or select the test that the item names, and confirm
   that it fails without the change. When the item names another check
   because no test applies, run that check.
3. Run the checks in AGENTS.md.
4. Make one signed commit for each item, with the item's commit subject.
5. Push the branch once, after all items pass.

Rows with the choice `decide` go to the maintainer. Rows with the choice
`defer` get a follow-up task. Do not reply to or resolve review threads
unless the maintainer asks.

## 4. Report and stop

Send the maintainer:

1. The verdict: the rows fixed, the rows not fixed, and the commits pushed,
   or the reason that the plan was not applied.
2. The fix table:

   | Row | Tier | Sources | Finding | Check | Fix depth | Decision | Commit |
   |---|---|---|---|---|---|---|---|

   Fix depth gives the recursion risk, and whether the fix is the root fix
   or the smallest fix.
3. Each `decide` row, as a question with at most two options.
4. The gaps and the warnings.

Then stop. The maintainer decides if a second round is necessary.
