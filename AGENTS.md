# AGENTS.md - NEREIDS Codebase Instructions

## North star (read before planning anything)

NEREIDS exists so an instrument scientist can reliably fit, from raw open-beam
and sample neutron counts on time-of-flight bin edges:

1. **density**
2. **temperature**
3. **density and temperature jointly**

The one fitting engine is the counts-domain Poisson/KL engine:
`crates/nereids-pipeline/src/counts_fit.rs` (`fit_counts`) and
`crates/nereids-pipeline/src/counts_map.rs` (`fit_map`) over
`crates/nereids-fitting/src/poisson.rs`. There is no second fitting path;
do not add one. The legacy typed-spectrum, transmission-LM, joint-Poisson,
spatial-regularization, resolution-calibration, energy-scale and detectability
paths were deleted and must not return.

Reliably means background, energy scale (t0, L) and the instrument pulse are
physically correct, not merely present - a wrong one of those is absorbed into
the fitted temperature. **Every plan states which of the three it serves and
how.** A plan that cannot name one is drift; stop and re-aim.

Full pipeline map: `docs/guide/src/pipeline-map.html`.

## Comment discipline (mandatory)

A comment says **what the code does and why it has this shape**, in the fewest
words that do it. Nothing else. In particular, delete on sight:

- History narration: "used to", "previously", "no longer", "before this
  change", "was changed/added/removed", "the old implementation".
- Provenance: PR numbers, issue numbers, review round labels, "Copilot
  review", "Codex", "Claude", commit hashes.
- Self-promotion and process talk: why the change is an improvement, what the
  author verified, what a reviewer asked, arguments aimed at a reviewer.
- Meta comments about comments, or about tests, branches or functions.
- Restatements of the code on the following line.

Keep: SAMMY source files and equation numbers, ENDF-6 references, units,
invariants, preconditions and `# Safety`/`# Errors`/`# Panics` sections,
and public-API rustdoc that states the contract. The project's Physics Rules
require physics documentation in-code; that is the only mandated prose.

History belongs in the commit message, which cannot go stale. Narration and
rambling have no place in this codebase.

## Pre-Commit Checklist (mandatory before every commit)

```
cargo fmt --all
cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings
cargo test --workspace --exclude nereids-python
```

Run `cargo fmt` (not `--check`) so formatting is applied. Do not suppress
clippy warnings with `#[allow(...)]`.

## Physics Rules

- **No approximations**: implement exact SAMMY physics. Never introduce ad-hoc
  approximations.
- **Document physics in-code**: reference specific SAMMY source files and
  equation numbers in the comments.
- **Validate against SAMMY tests**: use SAMMY's own test cases as ground truth
  for each module.

## Branches in physics code (mandatory)

**In code that computes a physical number, every branch must name the
physical distinction between its two sides.** If it cannot, the branch stands
in for something else - either logic a lower layer already handles, or physics
that was never implemented. Both are fixed by removing the branch, not by
keeping it. Validation stays at the public entry points; the evaluation core
gets no branches that are not physics.

A finding is answered by changing what the code **is**, not by adding a case
to what it **checks**. "Add an `if` for the edge case" fails this gate.

### Tests

Tests pin **physics**: analytic oracles, agreement between two independent
implementations, SAMMY baselines. A test written from the shape of the code
under test measures the code, not the physics. **Mutation-test every test
defending a physics change** - revert the fix and confirm the test fails.

## Commit Style

- **GPG-signed commits**: always use `git commit -S`.
- **Atomic commits**: commit early and often.
- **PR description: two sentences, or blank.** What changed and why. No
  headings, tables, bolded verdicts, emoji, validation inventory, risk
  section, or review narration. Running unattended, a blank body is preferred
  over generated prose. The same applies to issue comments and review replies.
- **No temp file litter**: clean up one-off scripts immediately.

## Mandatory user checkpoints

- **Before starting work on a new task**: research it, present the plan, and
  STOP. Do not write code until the user approves the plan.
- **After creating a PR**: show the URL and a short summary, then STOP.
- **Before merging**: STOP until the user explicitly says "merge".

"STOP" means end your turn. Approval of one step does not imply approval of
the next.

## Validation Patterns

- **Validate config up-front** in public entry points. `Err(_) => None` in
  `filter_map` only for per-pixel edge cases.
- **NaN bypasses guards**: `NaN < 1.0` is `false`. Always pair with
  `.is_finite()`.
- **Empty collections pass equality**: guard `is_empty()` separately.
- **`debug_assert!` is for impossible states only**, not input validation.
  Use hard errors (`return Err(...)`) for parser/config validation.
- **Magic number -> named constant**: preserve the exact numeric value and
  grep-verify after bulk replacements.

## Execution Model

| Work pattern | Approach |
|--------------|----------|
| Multiple independent PRs (no file overlap) | Parallel worktrees + concurrent subagents |
| Single PR, sequential work | Feature branch at repo root |

Decide at planning time and state the choice. When using parallel worktrees,
launch subagents in a single message so they run concurrently. Group related
changes into coherent PRs, not one commit per file.

## Git Workflow

- **origin**: `ornlneutronimaging/NEREIDS` - primary repo, issues, releases,
  main branch. All branches and PRs are pushed here directly.

## Reference Codebases (absolute paths; NOT siblings of this repo)

- `/Users/8cz/code.ornl.gov/zhangc/SAMMY` - physics reference (resonance
  formalism, SAMMY Fortran source under `sammy/src/`)
- `/Users/8cz/github.com/lanl/PLEIADES` - ORNL data normalisation helpers,
  ENDF retrieval
- `/Users/8cz/github.com/lanl/trinidi` - sparsity-handling reference

## Documentation hygiene - no investigation / audit / debugging memos in `docs/`

`docs/` is user-facing. Only these subtrees belong there:

| Subtree | What belongs |
|---------|--------------|
| `docs/adr/` | Architectural decision records |
| `docs/guide/` | User-facing how-to guides |
| `docs/references/` | External-facing reference material |

Audit memos, investigation notes and design drafts go in the PR body, in
`.research/` (gitignored), or inline with the code. Do not create
`docs/audit/` or any other floating memo directory.
