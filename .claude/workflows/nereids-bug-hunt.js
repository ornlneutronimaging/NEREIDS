export const meta = {
  name: 'nereids-bug-hunt',
  description:
    'Multi-agent bug hunt: Claude + Codex finders per domain, cross-LLM-family adversarial verification, consolidated P0/P1 report',
  whenToUse:
    'Broad defect sweep of the NEREIDS workspace ahead of a release / paper. RUN FROM THE MAIN REPO ROOT (not a .claude/worktrees/ session) so SAMMY primary sources resolve and subagents are not sandboxed away from them. Complements /review-pipeline (which is per-PR-diff + fix + merge gates); this workflow is whole-crate DETECTION only — it finds and locates defects, cross-verifies them across two LLM families, and reports. It does NOT fix anything.',
  phases: [
    { title: 'Preflight', detail: 'resolve repo root, codex + SAMMY availability, scope' },
    { title: 'Find', detail: 'Claude + Codex finder per domain (up to 2N agents)' },
    { title: 'Verify', detail: 'cross-LLM-family adversarial confirmation of single-family findings' },
    { title: 'Consolidate', detail: 'dedup, tier, write SUMMARY.md, return executive summary' },
  ],
}

// ---------------------------------------------------------------------------
// nereids-bug-hunt
//
// Whole-workspace defect sweep ahead of a release / paper. A THIN WRAPPER over
// the shared `dual-family-review` engine (.claude/workflows/dual-family-review.js):
// this file owns the 8-domain target list (one per crate / subsystem), the
// preflight (repo root / codex / SAMMY resolution), and the final consolidated
// .research/SUMMARY.md report. The engine owns the Find -> Verify mechanics:
//   - 2 independent LLM families per domain: Claude (agent()) + Codex (codex exec)
//   - P0/P1/P2 tiers (P3 excluded), file:line + primary-source evidence
//   - a cross-LLM-family confirmation pass on every SINGLE-family finding
//     (Claude findings verified by Codex; Codex findings verified by Claude)
//
// Detection-only (the engine never fixes; fixing is a separate, human-gated step
// via /review-pipeline + .claude/templates/fix-subagent-prompt.md) is enforced
// by the engine two ways: an always-on DETECTION_ONLY prompt contract, and
// optional tool-level restriction via the bug-hunt-{reader,runner} agent types
// (args.hardEnforce=true; the session must postdate those defs). See the engine
// file header for the full rationale (cross-family independence; non-idempotent
// audit->fix->review cycle).
//
// args (all optional):
//   args.domains      : string[]  subset of domain keys to audit (default: all 8)
//   args.mode         : 'sweep' (default, whole-crate) | 'diff' (only main...HEAD)
//   args.diffSpec     : string    diff range for mode='diff' (default 'main...HEAD')
//   args.contextNote  : string    recent-changes context injected into prompts
//                                  (e.g. "post audit-R3 fix batch at <sha>")
//   args.roundNote    : string    label for the report header (e.g. "Round 4")
//   args.skipCodex    : boolean   force Claude-only (single-family -> all findings
//                                  flagged NEEDS-VERIFICATION)
//   args.sammyRoot    : string    override SAMMY source root for physics checks
//   args.hardEnforce  : boolean   use the restricted bug-hunt-{reader,runner}
//                                  agent types (session must postdate those defs)
//   args.repoRoot     : string    explicit canonical root (overrides git-detected)
// ---------------------------------------------------------------------------

// Defensive: the Workflow tool's `args` should arrive as a real object, but a
// caller may pass a JSON-encoded string by mistake — parse it rather than
// silently ignore every override (which would default repoRoot/roundNote/etc.).
let A = {}
if (args && typeof args === 'object') A = args
else if (typeof args === 'string' && args.trim().startsWith('{')) {
  try {
    A = JSON.parse(args)
  } catch (_e) {
    A = {}
  }
}
const MODE = A.mode === 'diff' ? 'diff' : 'sweep'
const DIFF_SPEC = A.diffSpec || 'main...HEAD'
const CONTEXT_NOTE = A.contextNote || ''
const ROUND_NOTE = A.roundNote || 'bug-hunt'
const SKIP_CODEX = A.skipCodex === true

// Detection-only enforcement layers are implemented in the engine; this wrapper
// just resolves which agent type to use and passes hardEnforce through. Custom
// agent types load at SESSION START (not hot), so default to the always-available
// built-in `general-purpose`; pass args.hardEnforce=true from a session started
// AFTER the bug-hunt-{reader,runner} defs exist to switch the restriction on.
const HARD = A.hardEnforce === true
const READER_TYPE = HARD ? 'bug-hunt-reader' : 'general-purpose'
const RUNNER_TYPE = HARD ? 'bug-hunt-runner' : 'general-purpose'

// The 8 domains. `lookFor` is domain-specific; CORE_CHECKLIST is appended (in the engine).
// `sammy` true => physics/format primary-source comparison is in scope.
const ALL_DOMAINS = [
  {
    key: '01-core',
    name: 'nereids-core',
    paths: ['crates/nereids-core/'],
    sammy: false,
    blurb: 'shared types (Isotope, energy/TOF conversions, units).',
    p0: 'Off-by-factor in a unit / TOF<->energy conversion, a type invariant that can be violated to produce non-physical state, panic-on-valid-input.',
    lookFor: [
      '`tof_to_energy` / `energy_to_tof` sign + factor correctness (negative TOF must not yield a positive eV).',
      '`Isotope::new` rejecting non-physical / non-ENDF-encodable (Z, A) pairs.',
      'Newtype invariants actually enforced at construction, not just documented.',
    ],
  },
  {
    key: '02-endf',
    name: 'nereids-endf',
    paths: ['crates/nereids-endf/', 'crates/endf-mat/'],
    sammy: true,
    blurb: 'ENDF/B file parsing and resonance data structures (LRF=1/2/3/7, URR).',
    p0: 'Record-layout error (wrong field column / count), a resonance parsed into the wrong parameter slot, NIS/NER/NRS mismatch accepted silently, URR LSSF/per-J layout error.',
    lookFor: [
      'ENDF-6 fixed-field record layout: NRS/NLS/NER counts read from the correct field, continuation-record alignment, MF/MT routing.',
      'LRF=7 (KRM) layout, URR LFW/LRF per-J record layout, LSSF flag handling (double-counting).',
      'Negative-L / out-of-range quantum number guards (a NEREIDS recurring miss: one URR path guarded, a sibling path not).',
    ],
    primaryHint: 'ENDF-6 Formats Manual record layouts; SAMMY `src/endf/` for LRF=3/7 reference.',
  },
  {
    key: '03-physics',
    name: 'nereids-physics',
    paths: ['crates/nereids-physics/'],
    sammy: true,
    blurb:
      'cross-section physics (Reich-Moore, SLBW, MLBW, RML), U-matrix, phase shifts, penetrability, shift factors, Doppler broadening, multilevel coherent sums. HIGHEST-STAKES DOMAIN.',
    p0: 'Numerical/physics incorrectness vs SAMMY: off-by-N factor, sign error, missing 2pi/4pi/c, mis-applied formalism, velocity-factor double-count. Two latent multi-month sign/factor errors have already been caught here (SLBW `sin^2 phi` sign; SLBW/MLBW s-wave velocity factor) — both invisible at the peak energy where all SAMMY-parity tests sit.',
    lookFor: [
      'Each formalism U-matrix and sigma_T / sigma_E / sigma_R derivation, line-by-line vs SAMMY .f90.',
      'sign of `(1 - cos 2phi)*A` vs `sin 2phi*B`; factors of 2; (pi/k^2)*g_J prefactor; spin stat g_J = (2J+1)/(2*(2I+1)); barn vs cm^2; eV vs CMS frame.',
      'penetrability/shift ratios vs an extra velocity `sqrt(E/Er)` (the double-count tell); threshold E->0 limits; near-threshold subthreshold reduced widths sqrt(|G|) vs sqrt(0.5*|G|).',
      'Doppler kernel normalization; W (1/e half-width) vs sigma convention; FFT vs quadrature edge cases.',
      'TEST COVERAGE REGIME: prior bugs were invisible because every test sat at low rho / at the peak. Look for untested regimes (high rho, off-peak, alternate formalism) and oracle tests that mirror the implementation.',
    ],
    primaryHint:
      'SAMMY SLBW/MLBW: `src/mlb/mmlb3.f90` (Elastc_Mlb), `mmlb4.f90` (Cs/Si). Phase/penetrability: `src/xxx/mxxx9.f90` (Cs2sn2 cos/sin 2phi), `mxxx6.f90` (Cossin). RML: `src/rml/`. Reich-Moore: `src/endf/` + `src/mlb/`.',
  },
  {
    key: '04-fitting',
    name: 'nereids-fitting',
    paths: ['crates/nereids-fitting/'],
    sammy: false,
    blurb:
      'Levenberg-Marquardt engine, Poisson / joint-Poisson profile-binomial-deviance fitters, gradient/Fisher/covariance.',
    p0: 'Gradient/Fisher derivative evaluated at a different clamp than the objective, NaN injected into gradients via finite-difference probes, covariance panic on rank-deficient/exactly-determined systems, convergence declared on a NaN.',
    lookFor: [
      'Helper-contract consistency: do the gradient, Fisher, and deviance helpers all clamp at the SAME epsilon? (a known P0 shape near saturated transmission).',
      '`y_obs` / model-output finiteness + length checks in ALL branches (all-fixed branch, polish branch), not just the main path.',
      'sigma=0 / NaN / Inf weights silently clamped to a tiny number (masks bad data); polish-on-NaN-stage1 short-circuit.',
      'Exactly-determined (NRS == NX) and over-/under-determined system handling without panic.',
    ],
  },
  {
    key: '05-io',
    name: 'nereids-io',
    paths: ['crates/nereids-io/'],
    sammy: false,
    blurb: 'TIFF I/O, NeXus/HDF5 readers, TOF normalisation, dark-current / open-beam handling.',
    p0: 'Counts silently corrupted (NaN dark-current absorbed via `NAN.max(0.0)==0.0`), dead-pixel mask silently dropped, axis values not validated, wrong-shape data accepted.',
    lookFor: [
      '`normalize` validates finite counts; dark-current / open-beam NaN handling.',
      'NeXus TOF axis value-checked (monotonic, positive); zero-sized histogram axes rejected.',
      '`read_string_attr` handling both VarLenUnicode AND fixed-length-ASCII; BigTIFF (>4 GB) support or explicit rejection.',
      'dead-pixel masks applied, not dropped.',
    ],
  },
  {
    key: '06-pipeline',
    name: 'nereids-pipeline',
    paths: ['crates/nereids-pipeline/'],
    sammy: false,
    blurb: 'rayon spatial mapping over a 2D detector, energy calibration grid/golden-section search, per-pixel orchestration.',
    p0: 'Race/data-corruption under parallelism, whole-config error masked per-pixel (hides a config bug), calibration converging to wrong minimum, mid-run cancellation returning Ok with partial maps, panic-on-valid-input.',
    lookFor: [
      'config errors raised BEFORE the rayon iterator, not inside `par_iter().filter_map(Err=>None)` (only per-pixel edge cases may use that).',
      '2D<->flat index conversion (no off-by-one); spatial order preserved; masked pixels handled.',
      'calibration objective returns the GLOBAL minimum; `chi2.is_finite()` post-check; t0/L and temperature_k boundary guards.',
      '`with_precomputed_cross_sections` / project-restore rejecting wrong shapes (mismatched-shape SpatialResult -> GUI panic).',
      'cancellation semantics: partial result must NOT masquerade as a complete Ok(SpatialResult).',
    ],
  },
  {
    key: '07-python',
    name: 'bindings/python (PyO3)',
    paths: ['bindings/python/', 'crates/nereids-python/'],
    sammy: false,
    blurb: 'PyO3 bindings + type stubs for the public Python API.',
    p0: 'A PyO3 entry that panics (aborts the interpreter) on plausible input, an "absent when unset" accessor returning an f64 sentinel instead of Option, a type stub that lies about the runtime return type.',
    lookFor: [
      'Every `#[pyfunction]` / `#[getter]` that takes `energies` / arrays validates finiteness before entering Rust that can panic (panic across the FFI boundary aborts Python).',
      '"absent when unset" semantics return `Option<T>` (None -> Python None), never a magic f64; nested-flag combos gated on BOTH parent && child flags.',
      'Type-stub fidelity: `#[getter]` returning `[T; N]` becomes a Python LIST (stub must not claim `tuple`); InvalidParameter vs RuntimeError mapping; sibling fit_back_d / fit_back_f validate identically.',
      'Test the REAL bindings, not SimpleNamespace stubs.',
    ],
  },
  {
    key: '08-gui-ws',
    name: 'apps/gui + workspace tooling',
    paths: ['apps/gui/', 'Cargo.toml', 'scripts/', '.github/workflows/'],
    sammy: false,
    blurb: 'egui desktop app + workspace tooling, CI, packaging.',
    p0: 'GUI panic on a normal user action (e.g. clicking a restored mismatched-shape result), a CI/release step that can ship broken artifacts, an unpinned third-party GitHub Action (supply-chain).',
    lookFor: [
      'eframe lifecycle: macOS `process::exit` skipping Drop (WorkerGuard / flush), panic on click after project restore.',
      'CHANGELOG / version staleness vs released tags; CITATION.cff correctness.',
      'third-party GitHub Actions pinned to SHA (not a moving tag); workflow permission scoping.',
    ],
  },
]

// ---- structured-output schema (preflight only; FINDINGS/VERDICT live in the engine) ----

const PREFLIGHT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    repoRoot: { type: 'string' },
    isWorktree: { type: 'boolean' },
    codexAvailable: { type: 'boolean' },
    codexVersion: { type: 'string' },
    sammyRoot: { type: 'string', description: 'resolved SAMMY src root, or empty if not found' },
    headSha: { type: 'string' },
    presentPaths: { type: 'object', additionalProperties: { type: 'boolean' } },
    diffFileCount: { type: 'integer' },
  },
  required: ['repoRoot', 'isWorktree', 'codexAvailable', 'sammyRoot'],
}

// Tier ordering, used only to sort the consolidated report (the prompt builders
// and dedup / cross-family-verify helpers all live in the dual-family-review engine).
function tierRank(t) {
  return t === 'P0' ? 0 : t === 'P1' ? 1 : 2
}

// ---------------------------------------------------------------------------
// Phase: Preflight
// ---------------------------------------------------------------------------
phase('Preflight')
const domainKeys = Array.isArray(A.domains) && A.domains.length ? A.domains : ALL_DOMAINS.map((d) => d.key)
const DOMAINS = ALL_DOMAINS.filter((d) => domainKeys.includes(d.key))

const pf = await agent(
  `Resolve the environment for a NEREIDS bug-hunt. Run these and report via structured output:
- repoRoot: \`git rev-parse --show-toplevel\`
- isWorktree: true if repoRoot contains "/.claude/worktrees/" OR \`git rev-parse --git-common-dir\` differs from \`git rev-parse --git-dir\`
- headSha: \`git rev-parse --short HEAD\`
- codexAvailable / codexVersion: \`command -v codex && codex --version\` (false + empty if absent)
- sammyRoot: first of these that exists as a directory (test -d), else empty:
    /Users/8cz/code-int.ornl.gov/sammy/sammy/src
    "$(git rev-parse --show-toplevel)/../SAMMY/src"
    "$(git rev-parse --show-toplevel)/../SAMMY/sammy/src"
- presentPaths: for each of these, whether the path exists under repoRoot: ${DOMAINS.flatMap((d) => d.paths).join(', ')}
- diffFileCount: ${MODE === 'diff' ? `number of files in \`git diff --name-only ${DIFF_SPEC}\`` : '0 (sweep mode)'}
Be terse; just gather facts.`,
  { label: 'preflight', phase: 'Preflight', schema: PREFLIGHT_SCHEMA, agentType: READER_TYPE },
)

if (!pf) {
  log('Preflight was skipped — aborting bug-hunt.')
  return { aborted: true, reason: 'preflight skipped' }
}
if (A.repoRoot) pf.repoRoot = A.repoRoot // explicit canonical root (e.g. main) overrides git-detected worktree path
const codexUsable = pf.codexAvailable && !SKIP_CODEX
log(
  `repo ${pf.repoRoot} @ ${pf.headSha || '?'} | mode=${MODE} | domains=${DOMAINS.length} | ` +
    `codex=${codexUsable ? pf.codexVersion || 'yes' : 'DISABLED (single-family — findings will be NEEDS-VERIFICATION)'} | ` +
    `sammy=${pf.sammyRoot || 'NOT FOUND (physics findings lower-confidence)'}`,
)
if (pf.isWorktree && !pf.sammyRoot) {
  log(
    'WARNING: worktree session AND SAMMY not resolved — physics/endf primary-source verification ' +
      'will be lower-confidence. Re-run from the MAIN repo root with SAMMY reachable for a physics-grade sweep.',
  )
} else if (pf.isWorktree) {
  log(
    `Session is a worktree, but main source + SAMMY are reachable (repoRoot=${pf.repoRoot}) — proceeding at full fidelity.`,
  )
}

// ---------------------------------------------------------------------------
// Build the per-domain target list and hand off to the shared
// dual-family-review engine (Find -> Verify). The engine returns structured
// per-target findings; this wrapper consolidates them below. The scope
// directive is identical for every domain (it depends only on MODE/DIFF_SPEC),
// matching the original inline behaviour where `scope` was computed per call.
// ---------------------------------------------------------------------------
const scopeDirective =
  MODE === 'diff'
    ? `Audit ONLY the changes in \`git diff ${DIFF_SPEC}\` that touch this domain. Run the diff, read each changed file in full, but report findings only for code introduced or altered by the diff.`
    : `Audit the FULL crate(s) for this domain (a whole-codebase sweep, not a diff).`

const targets = DOMAINS.map((d) => ({
  key: d.key,
  name: d.name,
  paths: d.paths,
  sammy: d.sammy,
  blurb: d.blurb,
  p0: d.p0,
  lookFor: d.lookFor,
  primaryHint: d.primaryHint,
  scopeDirective,
}))

const engineOut = await workflow('dual-family-review', {
  pf: {
    repoRoot: pf.repoRoot,
    sammyRoot: pf.sammyRoot,
    codexAvailable: pf.codexAvailable,
    codexVersion: pf.codexVersion,
    headSha: pf.headSha,
    isWorktree: pf.isWorktree,
  },
  targets,
  config: {
    contextNote: CONTEXT_NOTE,
    roundNote: ROUND_NOTE,
    skipCodex: SKIP_CODEX,
    hardEnforce: HARD,
    sammyRootOverride: A.sammyRoot || '',
  },
})

const domainResults = (engineOut && engineOut.perTarget) || []

// ---------------------------------------------------------------------------
// Phase: Consolidate (barrier — needs every domain to dedup across domains)
// ---------------------------------------------------------------------------
phase('Consolidate')

const allVerified = domainResults.flatMap((r) => r.verified)
const verifiedP0 = allVerified.filter((f) => (f.verifierTier || f.tier) === 'P0').sort((a, b) => tierRank(a.tier) - tierRank(b.tier))
const verifiedP1 = allVerified.filter((f) => (f.verifierTier || f.tier) === 'P1')
const needsVerification = domainResults.flatMap((r) => r.needsVerification)
const refuted = domainResults.flatMap((r) => r.refuted)
const circular = domainResults.flatMap((r) => r.circular)
const p2Backlog = domainResults.flatMap((r) => r.p2s)

const tierTable = domainResults.map((r) => ({
  domain: r.domainKey,
  name: r.domainName,
  claude: r.tierCounts.claude,
  codex: r.tierCounts.codex,
  verifiedP0: r.verified.filter((f) => (f.verifierTier || f.tier) === 'P0').length,
  verifiedP1: r.verified.filter((f) => (f.verifierTier || f.tier) === 'P1').length,
  needsVerification: r.needsVerification.length,
  refuted: r.refuted.length,
}))

log(
  `Consolidating: ${verifiedP0.length} VERIFIED P0, ${verifiedP1.length} VERIFIED P1, ` +
    `${needsVerification.length} NEEDS-VERIFICATION, ${refuted.length} refuted, ${circular.length} circular-validation flags.`,
)

const payload = {
  meta: {
    repoRoot: pf.repoRoot,
    headSha: pf.headSha,
    mode: MODE,
    diffSpec: MODE === 'diff' ? DIFF_SPEC : null,
    roundNote: ROUND_NOTE,
    contextNote: CONTEXT_NOTE,
    codex: codexUsable ? pf.codexVersion || 'available' : 'DISABLED',
    sammyRoot: pf.sammyRoot || '(not found)',
    isWorktree: pf.isWorktree,
  },
  tierTable,
  verifiedP0,
  verifiedP1,
  needsVerification,
  refuted,
  circularValidationFlags: circular,
  p2Count: p2Backlog.length,
}

// One consolidator agent writes the durable markdown report and returns a
// short executive summary. (Bookkeeping above is deterministic JS; the agent
// only formats + stamps + narrates — it does not re-judge findings.)
const report = await agent(
  `Write a consolidated bug-hunt report for NEREIDS.

First run \`date -u +%Y%m%dT%H%M%SZ\` to get a stamp, then \`mkdir -p "${pf.repoRoot}/.research/bug-hunt-<stamp>"\`.
Write a thorough markdown report to \`${pf.repoRoot}/.research/bug-hunt-<stamp>/SUMMARY.md\` from this JSON payload (do not invent findings; render what is here). As your VERY FIRST write action, save this exact JSON payload verbatim to a file named raw-result.json inside that same bug-hunt-<stamp> directory — this guarantees a recoverable artifact even if report generation is later interrupted. THEN write the SUMMARY.md described below:

${JSON.stringify(payload, null, 2)}

The report MUST contain, in this order:
1. Header: repo, headSha, mode, roundNote, codex + sammy status. If meta.isWorktree is true AND meta.sammyRoot is "(not found)", add a prominent caveat that physics/endf findings may be lower-confidence because SAMMY was not reachable; otherwise omit that caveat (SAMMY was reachable and fidelity is full).
2. Per-domain tier table (Claude P0/P1/P2 vs Codex P0/P1/P2, plus verified/needs-verification/refuted counts) from tierTable.
3. "VERIFIED P0 (cross-LLM-family confirmed) — fix before release" — full detail (file:line, claim, reasoning, primary source, suggested fix) for each.
4. "VERIFIED P1".
5. "NEEDS-VERIFICATION (single-LLM-family only — NOT independently confirmed)" — explicitly state these were found by one family and either could not be cross-checked (codex disabled/unavailable) or got a split/uncertain vote, so they do NOT meet the >=2-distinct-LLM-family bar for VERIFIED.
6. "REFUTED by cross-family verification" — list with the refuter's reasoning (these are likely false positives; record them so they are not re-litigated).
7. "Circular-validation risks" — tests that may be validating buggy behavior.
8. A short "P2 backlog: N items" line (do not expand them).
9. A closing "Methodology + honesty note": this workflow DETECTS only; the audit->fix->review cycle is non-idempotent so fixes must go through /review-pipeline; VERIFIED means >=2 distinct LLM families agreed, NEEDS-VERIFICATION does not.

Then return via plain text (no schema): the absolute reportPath, the stamp, a 3-5 sentence executive summary, and a one-line verdict of the form "N verified P0, M verified P1, K need human verification".`,
  { label: 'consolidate', phase: 'Consolidate', agentType: RUNNER_TYPE },
)

return {
  reportPathNote: 'see consolidator output below for the written SUMMARY.md path',
  executiveSummary: report,
  counts: {
    verifiedP0: verifiedP0.length,
    verifiedP1: verifiedP1.length,
    needsVerification: needsVerification.length,
    refuted: refuted.length,
    circularValidationFlags: circular.length,
    p2Backlog: p2Backlog.length,
  },
  meta: payload.meta,
  tierTable,
  // full structured findings travel back to the caller for downstream use
  // (e.g. handing VERIFIED P0s to /review-pipeline's fix stage):
  verifiedP0,
  verifiedP1,
  needsVerification,
  refuted,
  circularValidationFlags: circular,
}
