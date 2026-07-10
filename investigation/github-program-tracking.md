# IC calibration remediation program tracking

Verified 2026-07-10 against `ornlneutronimaging/NEREIDS` and the organization
project API. This document is the durable local index for the GitHub program;
the live source of status is [NEREIDS Development Project #8](https://github.com/orgs/ornlneutronimaging/projects/8).

## Current program state

- Program epic: [#688](https://github.com/ornlneutronimaging/NEREIDS/issues/688)
- Active phase: [G0 #689](https://github.com/ornlneutronimaging/NEREIDS/issues/689),
  `In Progress`, scientific gate `Pending`
- G1–G9: `Todo`, scientifically blocked by native issue dependencies
- Production boundary: no Rust, Python, GUI, or MCP behavior changes may begin
  until G0 reaches an independently reviewed `GO`
- Bootstrap base: `b01e077` (`docs: audit Phase 0 fitting paths`)
- Project node ID: `PVT_kwDOAPDgcs4BOcuc`

The local completion harness passed once during bootstrap:

```text
212 passed, 1 skipped, 3 warnings in 277.16s
PASS: NEREIDS completion gate (8m18s)
```

The full transcript was observed in this session. The harness remains the
replayable evidence; the generated runtime log is intentionally ignored.

## Native issue hierarchy and dependencies

| Phase | Issue | Native blocked-by relationships | Workstream | Gate |
|---|---|---|---|---|
| Program | [#688](https://github.com/ornlneutronimaging/NEREIDS/issues/688) | #689–#698 | Program governance | Pending |
| G0 | [#689](https://github.com/ornlneutronimaging/NEREIDS/issues/689) | #699–#705 | Evidence harness | Pending |
| G1 | [#690](https://github.com/ornlneutronimaging/NEREIDS/issues/690) | #689 | Safety and contracts | Pending |
| G2 | [#691](https://github.com/ornlneutronimaging/NEREIDS/issues/691) | #689, #690 | Objective semantics | Pending |
| G3 | [#692](https://github.com/ornlneutronimaging/NEREIDS/issues/692) | #625, #689–#691 | IC physics and numerics | Pending |
| G4 | [#693](https://github.com/ornlneutronimaging/NEREIDS/issues/693) | #691, #692 | Count response and background | Pending |
| G5 | [#694](https://github.com/ornlneutronimaging/NEREIDS/issues/694) | #692, #693 | Calibration and identifiability | Pending |
| G6 | [#695](https://github.com/ornlneutronimaging/NEREIDS/issues/695) | #690–#694 | Surface integration | Pending |
| G7 | [#696](https://github.com/ornlneutronimaging/NEREIDS/issues/696) | #529, #694, #695 | Real-data validation | Pending |
| G8 | [#697](https://github.com/ornlneutronimaging/NEREIDS/issues/697) | #459, #628, #692–#696 | Performance | Pending |
| G9 | [#698](https://github.com/ornlneutronimaging/NEREIDS/issues/698) | #691, #693, #695–#697 | Migration and removal | Pending |

All G0–G9 issues are native sub-issues and blockers of #688. G0 is blocked by
all seven anchor leaves; G3 is explicitly ordered after G1/G2. Native
dependencies are closure constraints, not merely prose references.

## G0 anchor backlog

These native sub-issues of #689 make the still-missing pre-change evidence
visible. An implementation phase stays blocked if its corresponding anchor is
not complete.

| Coverage | Issue |
|---|---|
| Background/nuisance routes M25–M49 | [#699](https://github.com/ornlneutronimaging/NEREIDS/issues/699) |
| Calibration/boundary routes M50–M61 | [#700](https://github.com/ornlneutronimaging/NEREIDS/issues/700) |
| Spatial routes M14–M24 | [#701](https://github.com/ornlneutronimaging/NEREIDS/issues/701) |
| Single-spectrum routes M01–M13 | [#702](https://github.com/ornlneutronimaging/NEREIDS/issues/702) |
| Real fixtures, checksums, and provenance | [#703](https://github.com/ornlneutronimaging/NEREIDS/issues/703) |
| Cleanup findings CL-01–CL-50 | [#704](https://github.com/ornlneutronimaging/NEREIDS/issues/704) |
| Cross-surface contracts X01–X17 | [#705](https://github.com/ornlneutronimaging/NEREIDS/issues/705) |

## Existing-work reconciliation

| Existing issue | Primary phase | Treatment |
|---|---|---|
| [#625](https://github.com/ornlneutronimaging/NEREIDS/issues/625) | G3 | Native child of #692 and blocker for the centering/operator decision. |
| [#628](https://github.com/ornlneutronimaging/NEREIDS/issues/628) | G8 | Native child of #697 and blocker for parity-first IC plan caching. |
| [#459](https://github.com/ornlneutronimaging/NEREIDS/issues/459) | G8 | Native child of #697; retained as the measured spatial performance backlog. |
| [#529](https://github.com/ornlneutronimaging/NEREIDS/issues/529) | G7 | Native child/blocker of #696; G4 supplies candidate operators but does not duplicate the real-data study. |
| #423, #427, #448, #458, #502 | Historical only | Closed work is not reopened without a fresh, narrow current-branch mechanism reproduction. |

## Project fields

All program items carry `Program = IC calibration remediation`.

| Field | Node ID | Options used by the program |
|---|---|---|
| Program | `PVTSSF_lADOAPDgcs4BOcuczhXnBIQ` | IC calibration remediation |
| Phase | `PVTSSF_lADOAPDgcs4BOcuczhXnBIU` | Program, G0–G9 |
| Workstream | `PVTSSF_lADOAPDgcs4BOcuczhXnBIY` | Program governance, Evidence harness, Safety and contracts, Objective semantics, IC physics and numerics, Count response and background, Calibration and identifiability, Surface integration, Real-data validation, Performance, Migration and removal |
| Scientific gate | `PVTSSF_lADOAPDgcs4BOcuczhXnBIc` | Pending, Blocked, NO-GO, GO |
| Disposition | `PVTSSF_lADOAPDgcs4BOcuczhXnBIg` | mixed, keep, keep+improve, deprecate, remove, complete-before-exposure |
| Risk | `PVTSSF_lADOAPDgcs4BOcuczhXnBIk` | Critical, High, Medium, Low |

`Status` remains the ordinary delivery state (`Todo`, `In Progress`, `Done`). It
must never substitute for `Scientific gate`: green delivery status or CI does
not establish scientific validity.

## Saved monitoring views

| View | URL | Filter |
|---|---|---|
| IC Program Table | [view 4](https://github.com/orgs/ornlneutronimaging/projects/8/views/4) | all IC program items |
| IC Gate Reviews | [view 5](https://github.com/orgs/ornlneutronimaging/projects/8/views/5) | Pending, Blocked, or NO-GO gates |
| IC Program Board | [view 6](https://github.com/orgs/ornlneutronimaging/projects/8/views/6) | all IC program items, board layout |
| IC Blocked Evidence | [view 7](https://github.com/orgs/ornlneutronimaging/projects/8/views/7) | Blocked or NO-GO gates |
| IC Deprecation and Removal | [view 8](https://github.com/orgs/ornlneutronimaging/projects/8/views/8) | deprecate or remove dispositions |
| IC Program Roadmap | [view 9](https://github.com/orgs/ornlneutronimaging/projects/8/views/9) | all IC program items, roadmap layout |

Dates are intentionally unset during bootstrap. Native phase dependencies are
authoritative until G0 produces evidence-backed estimates; adding speculative
dates would make the roadmap look more certain than the audit supports.

## Lossless primary-owner registry

G0 is transversal and owns no implementation identifier. It creates anchors for
all findings. Each finding below has exactly one G1–G9 implementation owner; the
machine-readable registry is `investigation/github-program-owner-map.csv`.

- G1: M01, M13, M24, M33, M47, M56, M57, M59, X07, X09, X10, X12,
  X13, X16, CL-01, CL-02, CL-13, CL-15, CL-16, CL-22, CL-25, CL-36,
  CL-39, CL-40, CL-43, CL-44, CL-46, CL-47, CL-48, CL-49, CL-50
- G2: M04, M05, M09, M10, M16, M17, M20, M21, M26, M29, M32, M34,
  M36, M39, M42, M44, M49, M58, M60, X03, X06, CL-07, CL-12,
  CL-14, CL-17, CL-42
- G3: M02, M03, M14, M15, CL-10, CL-18, CL-19, CL-26, CL-32,
  CL-33, CL-35, CL-37
- G4: M06, M07, M08, M11, M12, M18, M19, M22, M23, M25, M27, M28,
  M30, M31, M35, M37, M38, M40, M41, M43, M45, M46, M48, M61,
  X05, CL-05, CL-06, CL-09, CL-11, CL-20
- G5: M50, M51, M52, M53, M54, M55, X04, CL-08, CL-21, CL-24,
  CL-27, CL-28, CL-29, CL-30, CL-34, CL-38, CL-45
- G6: X01, X02, X08, X11, X14, X15, X17
- G7: CL-41
- G8: CL-31
- G9: CL-03, CL-04, CL-23

Validation must print exactly `61 M + 17 X + 50 CL = 128 identifiers; each has
exactly one primary owner`.

## Deprecation and rewiring policy

1. Reproduce first. No behavior change begins without its current success,
   failure, rejection, or compatibility anchor.
2. Fail closed after the anchor for silent ignored configuration, false-success
   results, typo fallback, raw-count/transmission reinterpretation, and accepted
   all-failed maps.
3. Hide or reject incomplete routes until their versioned mechanism passes its
   fixed Q gates; “complete before exposure” is not a deprecation shortcut.
4. Deprecate executable public F2/F4/S2/S4 routes with deterministic warnings,
   replacements, migration tests, and resolved-route provenance in v0.4.
   Production implementation deletion is ineligible before v0.5 and still
   requires a documented repository/downstream usage audit.
5. Typed `Auto`/default routing may select the robust route when the input domain
   uniquely determines it. Explicit solver/objective/version requests are never
   silently reinterpreted; they execute their still-supported exact behavior,
   warn, or fail with a migration error.
6. Archived serialized names retain deterministic tombstones beyond v0.5.
   Tombstone removal requires its own later versioned usage audit so old
   projects/manifests fail actionably rather than change statistics.

## Development gates

### Issue-ready

A leaf needs primary IDs, exact pre-change reproduction, copied fixed Q
thresholds, native dependencies, compatibility/migration impact, evidence path,
completion command, engineering owner, and independent reviewer. Data masking
or residual-driven reweighting is forbidden.

### PR merge

The PR cites the anchor, changes one mechanism/contract, passes route-specific
tests and `.harness/verify.sh`, preserves raw evidence, adds migration/rejection
tests for public changes, and receives fresh-context adversarial review.

### Phase science

Phase issues close only after the fixed GO/NO-GO checklist is independently
accepted. Failed thresholds produce an elimination record and corrective leaf;
they are not relaxed after results are seen. A downstream phase may prepare
non-production evidence, but its closure remains natively blocked.

### Post-merge

After merge, rerun the repository post-merge integration gate on synchronized
`main`; only then close the leaf. Component PRs never auto-close phase issues.

## Replay

```bash
pixi run python investigation/verify_github_program.py
./.harness/verify.sh
git diff --name-only b01e077..HEAD
```

The final command must contain no production Rust/Python/GUI/MCP source path for
the bootstrap commit.
