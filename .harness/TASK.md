# Task contract

## Source

User message 1 (2026-07-10, verbatim):

> We will be using this thread to conduct maintainece work for NEIREDS.
>
> there are massive amount of bot PR on github, all CI are failing,
> investigae and suggestion resolution

User message 2 (after investigation presented options A/B/C, verbatim):

> B, and the corresponding PR should also auto close the bot PRs

Option B (as presented and chosen): "the full plan (clippy fix + coordinated
egui-0.35 PR + triage the majors)". The user-approved plan elaborating B is
`/Users/8cz/.claude/plans/harmonic-gathering-hejlsberg.md`; its deliverables
(quoted in the requirements below) are:

> PR 1 — fix(ci): Rust 1.97 lints + repin orphaned action SHAs
> PR 2 — build(deps): coordinated egui 0.34→0.35 ecosystem
>        (supersedes #681, #683, #684, #685, #686)
> PR 3 — build(deps): actions majors + zip 8 + sysinfo 0.39
>        (supersedes #677, #678, #679, #680, #682, #687)
> Process: create PR → STOP for user → /review-pipeline → STOP → merge on
> explicit approval → next PR rebases.
> End state: 11 bot PRs closed, main Dependabot Updates run green, next
> stable-Rust release can no longer redden unrelated PRs (pinned toolchain).

## Rules (enforced by the stop gate)

- One checkbox per requirement, quoting the Source's own words. Every
  imperative sentence in the Source must map to a requirement line.
- Checkboxes must use exactly `- [ ]` — other bullet styles are
  invisible to the gate.
- `[x]` requires evidence ON the line: the check command in backticks
  and what it showed. The gate blocks `[x]` lines with no backticked
  command; an unfilled `<placeholder>` in backticks does not count.
- If an item's recorded evidence is later invalidated (a retraction, a
  failed re-check), flip it back to `- [ ]` — a checked line with dead
  evidence is a false report.
- Dropping an item requires the user's explicit approval of that named
  item: `- [-] DROPPED: <item> (approved: "<user's words>")`. The quoted
  words must themselves address the dropped item — a generic "proceed"
  or "sounds good" is never approval. The gate blocks `[-]` lines with
  no `(approved:` quote.
- Blocked on the user? Mark it `- [?] <item> — BLOCKED ON USER:
  <specific question>`, finish everything not blocked, then end the
  turn asking about all `[?]` items together.
- Never delete or reword a requirement line. Discovered work is ADDED
  as new lines, not swapped in.
- Commit this file unless the user objects — tampering must show in
  git diff.

## Requirements

- [x] R1: "investigae and suggestion resolution" — three root causes identified with evidence and resolution plan approved — check: `for pr in 677 678 679 680 681 682 683 684 685 686 687; do gh pr checks $pr; done` + `gh run view` logs + local 1.97 repro (`CLIPPY EXIT: 101`, question_mark at poisson.rs:1457) + Dependabot log (`no such commit 50570a4f`); plan approved by user via ExitPlanMode.
- [x] R2 (PR1): "fix all 37 lints" (1× `clippy::question_mark` poisson.rs:1457; 36× `float_literal_f32_fallback` in apps/gui; plus linux-cfg candidates in file_dialog.rs ~347–770 incl. line 505) — check: scratchpad `cargo +1.97.0 clippy --workspace --exclude nereids-python --all-targets -- -D warnings` exits 0; ubuntu Clippy leg on the PR arbitrates cfg(linux) count. LOCAL DONE: `cargo clippy … -- -D warnings` → `CLIPPY -D warnings EXIT: 0` (darwin, 1.97.0); 37 fixes applied (`cargo clippy --fix` 36 + manual file_dialog.rs:505); CI ARBITER GREEN: `gh pr checks 707` → `Clippy (ubuntu-latest) pass 2m14s`, `Clippy (macos-latest) pass 3m36s`.
- [x] R3 (PR1): "Repin both poison SHAs — ci.yml:167 taiki-e/install-action → v2.83.0 release SHA c7eb1735…; align security.yml:24 to the same" — check: `grep -rc c7eb1735 .github/workflows/*.yml` = ci.yml:1 + security.yml:1, old SHAs 50570a4f/4684b840 grep = 0; `gh api repos/taiki-e/install-action/git/ref/tags/v2.83.0` → c7eb1735f09259a5035e8e5d44b1406b1cddc0fb commit.
- [x] R4 (PR1): "All 11 dtolnay/rust-toolchain steps … → master head SHA fa04a145…" — check: `grep -rc fa04a145 .github/workflows/*.yml` = 7+1+2+1 = 11; `grep -rn 29eef336` = 0; `gh api repos/dtolnay/rust-toolchain/branches/master` → fa04a1451ff1842e2626ccb99004d0195b455a88 "Add 1.96.1 patch release".
- [?] R5 (PR1): "add rust-toolchain.toml with channel = \"1.97.0\" only … align all `toolchain: stable` inputs → \"1.97.0\" … add the missing `toolchain:` input in security.yml" — check: file created (channel only + rationale/bump-ritual comment); `grep -rn 'toolchain: stable' .github/workflows/` = 0; security.yml:23-25 now has explicit input. DEVIATION (review round 1, VERIFIED P1): channel is 1.96.1, NOT the 1.97.0 this line quotes — rust-lang/rust#159035 is a P-critical x86-64 miscompile in 1.97.0 (1.96.1 unaffected; fix in no stable yet), so pinning 1.97.0 would lock release wheels onto a miscompiling compiler. Evidence at 1.96.1: `rust-toolchain.toml` channel = "1.96.1"; all 11 workflow inputs = "1.96.1" (`grep -rc '1.96.1' .github/workflows/*.yml` = 7+1+2+1); `rustc --version` in repo dir under rustup → rustc 1.96.1 (file honored). — BLOCKED ON USER: the line's quoted requirement says 1.97.0; delivered is 1.96.1 (round-1 VERIFIED P1). Approve the supersession so this line can be recorded as DROPPED-in-favor-of-R26 per contract rules.
- [?] R6 (PR1): "full pre-commit gate under the 1.97 scratchpad toolchain" + "`pixi run test-python`" — check: gates re-run under the FINAL 1.96.1 pin on the full branch tree (post-repin, post-poisson fixes): `cargo fmt --all` → exit 0; `cargo clippy … -- -D warnings` → `CLIPPY EXIT: 0`; `cargo test --workspace --exclude nereids-python` → `TEST EXIT: 0`, 1268 passed 0 failed (+4 new tests vs main); `pixi run test-python` → `PIXI EXIT: 0`, 212 passed 1 skipped. (Earlier 1.97.0-era run: 1264/0 + 212/1 — superseded by this one.) — BLOCKED ON USER: worded "under the 1.97 toolchain"; gates ran green under the delivered 1.96.1 pin. Approve with R5's supersession.
- [x] R7 (PR1): "create PR → STOP for user" — check: PR URL posted with summary; turn ends. Evidence: `gh pr create` → https://github.com/ornlneutronimaging/NEREIDS/pull/707; checkpoint-1 STOP taken in the same turn.
- [?] R8 (PR2): "Cargo.toml: egui 0.35, eframe 0.35 (same feature list), egui_extras 0.35, egui_plot 0.36, egui-file-dialog 0.14.1 (not 0.14.0)"; lockstep + winit comments updated; rfd stays `=0.17.2` — check: `grep -nE 'egui|eframe|rfd' Cargo.toml` shows the pins; `cargo tree` single egui version. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R9 (PR2): "Rename 8 deprecated show_inside → show" (app.rs:173,195; studio/mod.rs:49,619,1118; guided/sidebar.rs:31; widgets/toolbar.rs:21; widgets/statusbar.rs:17) + prose comment app.rs:98 — check: `grep -rn show_inside apps/gui/src/` returns 0 hits; clippy 1.97 exit 0. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R10 (PR2): "Re-verify the save-extension workaround comment in file_dialog.rs against egui-file-dialog 0.14 behavior" — check: comparison of 0.14.1 source vs comment recorded in PR body. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R11 (PR2): "dependabot.yml: egui group FIRST + exclude from the existing group" — check: yaml matches plan block; `patterns: ["egui*", "eframe"]`; cargo-minor-patch gains exclude-patterns. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R12 (PR2): "Gates: GTK tripwire + scripts/check_wheel_policy.sh + full pre-commit; manual GUI smoke test" — check: cargo-tree GTK assert clean; wheel-policy script passes; `cargo run -p nereids-gui` smoke observations recorded. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R13 (PR2): "create PR → STOP for user" — check: PR URL posted; turn ends. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R14 (PR3): "Replicate the bot SHA+comment diffs exactly: setup-pixi v0.10.0 (ci.yml + docs.yml), deploy-pages v5.0.0 (docs.yml), action-gh-release v3.0.1 (publish.yml), codecov-action v7.0.0 (ci.yml)" — check: `grep -rn` each new SHA; matches `gh pr diff 677/678/679/680`. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R15 (PR3): "Root Cargo.toml: zip = \"2\" → \"8\"" — check: grep + `cargo test -p nereids-endf` incl. `install_local_endf_accepts_zip_archive`. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R16 (PR3): "apps/gui/Cargo.toml: sysinfo = \"0.37\" → \"0.39\"" — check: grep + `cargo tree -d` shows single sysinfo copy. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R17 (PR3): "create PR → STOP for user" — check: PR URL posted; turn ends. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R18 (process): "/review-pipeline → STOP → merge on explicit approval" for each PR — check: review summary tables presented; merges only after the user's explicit word. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R19 (end state): "the corresponding PR should also auto close the bot PRs" — all 11 bot PRs closed — check: `gh pr list --author app/dependabot --state open` empty after PR3 merge (fallback `@dependabot close` comments recorded if used). — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R20 (end state): "main Dependabot Updates run green" — check: next `Dependabot Updates` github_actions run on main succeeds (`gh run list`), no `no such commit` error. — BLOCKED ON USER: PR #707 (PR1) is at the mandatory user checkpoint (repo CLAUDE.md: create PR -> STOP; the user's next message is the gate), and the approved plan sequences this work after PR1 merges. Approve /review-pipeline on #707, then merge, to unblock?
- [?] R21 (PR3): "cargo update for the bumped crates; full pre-commit gate" — check: Cargo.lock updated for zip/sysinfo/egui-family only; fmt/clippy/test pass. — BLOCKED ON USER: PR #707 (PR1) is mid-review-pipeline; PR2/PR3 are sequenced behind user-gated merges (create->STOP, review->STOP, merge on explicit approval). Unblocks as the user advances each gate.
- [?] R22 (process): "sequential PRs, each rebased on the previous merge" — check: PR2 branch created from post-PR1 main; PR3 from post-PR2 main. — BLOCKED ON USER: PR #707 (PR1) is mid-review-pipeline; PR2/PR3 are sequenced behind user-gated merges (create->STOP, review->STOP, merge on explicit approval). Unblocks as the user advances each gate.
- [x] R23 (process): "Commits GPG-signed, atomic" — check: git log --show-signature on the branch commits. Evidence: `git log --show-signature 19d0ea2..HEAD` → every branch commit `gpg: Good signature from "Chen Zhang (kedokudo)"` (count-free by design — re-run at each push; last verified at the round-2 fix push); atomic (one concern per commit: contract / lints / CI-pins / repin / poisson / gate-fix splits).
- [?] R24 (PR3): "CI green" on PR3 — check: gh pr checks <PR3> all pass. — BLOCKED ON USER: PR #707 (PR1) is mid-review-pipeline; PR2/PR3 are sequenced behind user-gated merges (create->STOP, review->STOP, merge on explicit approval). Unblocks as the user advances each gate.
- [?] R25 (release guard): "the next release must run gh workflow run publish.yml -f dry_run=true first" — check: recorded in PR3 body + verified before next tag. — BLOCKED ON USER: PR #707 (PR1) is mid-review-pipeline; PR2/PR3 are sequenced behind user-gated merges (create->STOP, review->STOP, merge on explicit approval). Unblocks as the user advances each gate.
- [x] R26 (PR1, supersedes R5/R6 wording — review round 1 VERIFIED P1): "Pin 1.96.1, not 1.97.0: rust-lang/rust#159035 is a P-critical x86-64 miscompile in 1.97.0 (1.96.1 unaffected; fix in no stable); release wheels for the ORNL x86-64 fleet build under this pin" — check: `rust-toolchain.toml` channel = "1.96.1"; 11 workflow inputs = "1.96.1"; full gates green under 1.96.1 (`CLIPPY EXIT: 0`, `TEST EXIT: 0` 1268/0, `PIXI EXIT: 0` 212/1); primary source independently fetched (issue OPEN, labels P-critical/I-miscompile/O-x86_64).

## Coverage note

Source sentence mapping: "We will be using this thread to conduct
maintainece work" — session-scoping statement, no directive. "there are
massive amount of bot PR … investigae and suggestion resolution" → R1.
"B" (chosen plan) → R2–R18, R21–R25. "the corresponding PR should also auto close
the bot PRs" → R19 (+ supersedes lists in R8/R14–R16). Plan end-state
line → R19, R20.
