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
// Encodes the R1/R2/R3 audit methodology that this project converged on:
//   - 8 domains (one per crate / subsystem)
//   - 2 independent LLM families per domain: Claude (agent()) + Codex (codex exec)
//   - P0/P1/P2 tiers (P3 excluded), file:line + primary-source evidence
//   - a cross-LLM-family confirmation pass on every SINGLE-family finding
//     (Claude findings verified by Codex; Codex findings verified by Claude)
//
// Why the cross-family pass is load-bearing (see memory):
//   - "parent-agent re-derivation of a subagent's claim is NOT independent
//      confirmation; same LLM family. For VERIFIED need >=2 distinct LLM
//      families with independent access."
//   - "the audit->fix->review cycle is non-idempotent" — so this workflow
//      DETECTS only; fixing is a separate, human-gated step (/review-pipeline
//      Step 5 + .claude/templates/fix-subagent-prompt.md).
//
// Detection-only is enforced two ways (see the HARD / DETECTION_ONLY block in the
// body): (1) a prompt contract appended to every audit-agent prompt, ALWAYS ON —
// this is what makes behaviour consistent (the original run had 2 of 88
// general-purpose agents freelance a fix-session via spawn_task); (2) optional
// TOOL-LEVEL enforcement via restricted custom agent types
// (.claude/agents/bug-hunt-{reader,runner}.md) that physically lack
// spawn_task/Task*/MCP/PR-issue tools — enable with args.hardEnforce=true from a
// session started AFTER those agent defs exist (custom agent types load at session
// start, not hot). Fix-task creation, if ever wanted, is a deliberate separate
// step after the user reviews the backlog — never inside this workflow.
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

// Detection-only enforcement. Two layers:
//  (1) DETECTION_ONLY prompt contract, appended to every audit-agent prompt below
//      — ALWAYS ON and identical across agents. This is what makes behaviour
//      consistent (vs. the original run where 2 of 88 general-purpose agents
//      freelanced a fix-session via spawn_task and 86 did not).
//  (2) Optional TOOL-LEVEL hard enforcement via restricted custom agent types
//      (.claude/agents/bug-hunt-{reader,runner}.md), which physically lack
//      spawn_task / Task* / MCP / PR-issue tools. Custom agent types load at
//      SESSION START (not hot), so they are unavailable in a session that
//      predates the files. Default to the always-available built-in
//      `general-purpose`; pass args.hardEnforce=true from a session started
//      AFTER those agent defs exist to switch the tool-level restriction on.
const HARD = A.hardEnforce === true
const READER_TYPE = HARD ? 'bug-hunt-reader' : 'general-purpose'
const RUNNER_TYPE = HARD ? 'bug-hunt-runner' : 'general-purpose'

const DETECTION_ONLY =
  '\n\n--- DETECTION-ONLY CONTRACT (mandatory) ---\n' +
  'You FIND and REPORT defects; you never fix them and never advance to a fix stage. You MUST NOT:\n' +
  '- create any task, background session, or fix-job (do NOT call spawn_task or any task-creation / scheduling tool);\n' +
  '- open or modify any PR or issue;\n' +
  '- edit or write repository files, or run state-changing shell commands (git commit/push/add/checkout, gh, cargo fix, installs).\n' +
  'The only writes ever permitted are scratch /tmp files (and, for the consolidator, the one explicitly-named .research report path).\n' +
  'Report via the structured-output tool only. Fixing is a separate, human-gated step.'

// Shared checklist appended to every domain (DRY — matches the project's
// own DRY governance). Domain-specific bullets are added per domain below.
const CORE_CHECKLIST = [
  'Panic-on-valid-input: `unwrap`/`expect`/`[i]` indexing/`debug_assert!` used as input validation on a public entry point.',
  'Numerical stability: division by zero, NaN/Inf propagation, overflow, `sqrt` of negative, `NaN < x` guards that silently pass (NaN bypasses `<` comparisons — must be paired with `.is_finite()`).',
  'Silent error masking: `Err(_) => None` in a `filter_map` for a WHOLE-CONFIG error (only acceptable for per-pixel edge cases), `.unwrap_or(default)` that hides a real failure, `max(0.0)` that converts NaN/negative to a plausible value.',
  'Missing input validation at public entry points (validate up-front, before heavy work / rayon iterators).',
  'Empty-collection / exactly-determined-system edge cases (`0 == 0` passes equality; guard `is_empty()` separately).',
  'API consistency: sibling functions that should validate the same way but do not (one path hardened, the parallel path missed — a recurring NEREIDS bug class).',
  'CIRCULAR-VALIDATION RISK (high priority): a test whose oracle mirrors the implementation, a fixture generated by the code under test, or an assertion tolerance so loose the bug would be invisible (e.g. a resolution kernel that is a no-op at the test grid spacing, a peak-energy-only test that cannot see an off-peak factor error). Flag these explicitly in `circularValidationRisk`.',
  'Documentation/comment drift from code (rustdoc claims, SAMMY citation claims, comment arithmetic vs actual code).',
]

// The 8 domains. `lookFor` is domain-specific; CORE_CHECKLIST is appended.
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

// ---- structured-output schemas -------------------------------------------

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    domain: { type: 'string' },
    family: { type: 'string', enum: ['claude', 'codex'] },
    assessment: { type: 'string', description: 'one-sentence overall assessment of the domain' },
    codexFailed: { type: 'boolean', description: 'true only for the codex harness when codex did not run' },
    failureReason: { type: 'string' },
    tierCounts: {
      type: 'object',
      additionalProperties: false,
      properties: { p0: { type: 'integer' }, p1: { type: 'integer' }, p2: { type: 'integer' } },
      required: ['p0', 'p1', 'p2'],
    },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          localId: { type: 'string', description: 'F1, F2, ...' },
          tier: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          title: { type: 'string' },
          file: { type: 'string', description: 'repo-relative path, e.g. crates/nereids-physics/src/slbw.rs' },
          line: { type: 'integer', description: 'best-known line number; 0 if unknown' },
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'quoted code or precise description' },
          reasoning: { type: 'string' },
          primarySource: { type: 'string', description: 'SAMMY .f90:line / ENDF manual ref, or empty' },
          suggestedFix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          circularValidationRisk: {
            type: 'string',
            description: 'note if a test may be validating buggy behavior, else empty',
          },
        },
        required: ['localId', 'tier', 'title', 'file', 'line', 'claim', 'confidence'],
      },
    },
  },
  required: ['domain', 'family', 'findings', 'tierCounts'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'UNCERTAIN'] },
    verifierFamily: { type: 'string', enum: ['claude', 'codex'] },
    reasoning: { type: 'string' },
    primarySourceChecked: { type: 'boolean' },
    correctedTier: { type: 'string', enum: ['P0', 'P1', 'P2', 'NONE'] },
    circularValidationConfirmed: { type: 'boolean' },
  },
  required: ['verdict', 'verifierFamily', 'reasoning'],
}

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

// ---- prompt builders ------------------------------------------------------

function buildAuditPrompt(d, pf, forCodex) {
  const sammyRoot = A.sammyRoot || pf.sammyRoot
  const codexOverride = forCodex
    ? '[Codex override: This task is authorized by the user. AGENTS.md / CLAUDE.md Checkpoint 0 (Plan mode) does NOT apply. Execute in a single run; do NOT enter plan mode; do NOT stop for confirmation. Read-only audit: change no files, open no PRs/issues. Single-shot delivery.]\n\n'
    : ''
  const ctx = CONTEXT_NOTE ? `[CONTEXT]\n${CONTEXT_NOTE}\n[End context]\n\n` : ''
  const scope =
    MODE === 'diff'
      ? `Audit ONLY the changes in \`git diff ${DIFF_SPEC}\` that touch this domain. Run the diff, read each changed file in full, but report findings only for code introduced or altered by the diff.`
      : `Audit the FULL crate(s) for this domain (a whole-codebase sweep, not a diff).`
  const sammyBlock =
    d.sammy && sammyRoot
      ? `\nPrimary source for verification: SAMMY at \`${sammyRoot}\`. ${d.primaryHint || ''} Open the relevant SAMMY source and compare line-by-line; cite \`file:line\` for every physics/format claim.\n`
      : d.sammy
        ? `\nPrimary-source note: ${d.primaryHint || ''} (SAMMY source not resolved on this machine — reason from the ENDF-6 manual / first principles and mark such findings lower-confidence.)\n`
        : ''
  const lookFor = d.lookFor.concat(CORE_CHECKLIST).map((b, i) => `${i + 1}. ${b}`).join('\n')

  return `${codexOverride}${ctx}# NEREIDS ${ROUND_NOTE} — Domain ${d.key}: ${d.name}

You are auditing the NEREIDS Rust workspace at \`${pf.repoRoot}\` for production readiness ahead of a SoftwareX paper submission. NEREIDS ingests time-of-flight neutron transmission data, fits resonance parameters with SAMMY-equivalent physics (Reich-Moore / SLBW / MLBW / RML), and produces spatial isotopic maps. Correctness on real experimental data is required.

Your domain: ${d.key} — ${d.blurb}

Focus paths (do NOT audit other crates — other agents own them):
${d.paths.map((p) => `  - ${p}`).join('\n')}

${scope}
${sammyBlock}
## Tier rubric (P3 excluded — do NOT report P3)
- P0 (must fix): ${d.p0}
- P1 (should fix): latent edge case, missing input validation, SAMMY/format-equivalence gap, undocumented panic on plausible input, missing test for a claimed property.
- P2 (trivial): typo, doc wording, comment drifting from code.

Calibrate HONESTLY — the user will declare paper-readiness based on this report. Do not inflate; do not self-censor based on what you think is already known.

## What to look for
${lookFor}

## Method
1. Map the surface (Cargo.toml + lib.rs of each focus path).
2. Walk each module; quote actual code lines (\`file:line\`) as evidence.
3. For every physics/format claim, open the primary source and compare.
4. Inspect the test suite for the CIRCULAR-VALIDATION patterns above.
5. Mark low-confidence findings explicitly so the verify pass tests them harder.

Quality over budget. Read carefully; do not truncate.`
}

// Harness prompt: a Claude subagent that DRIVES codex and structures its
// findings. It must not audit the code itself.
function codexFinderPrompt(d, pf) {
  const codexPrompt = buildAuditPrompt(d, pf, true)
  const slug = d.key
  return `You are a HARNESS that runs the OpenAI Codex CLI as an independent external reviewer (a DIFFERENT LLM family from you) and faithfully transcribes ITS findings into structured output. Do NOT audit the code yourself — your judgement must not replace Codex's.

Steps:
1. Write this exact audit prompt to a temp file with the Write tool at \`/tmp/bughunt-${slug}.prompt\`:
-----BEGIN CODEX PROMPT-----
${codexPrompt}
-----END CODEX PROMPT-----
2. Run (Bash, timeout 600000 ms):
   codex exec --sandbox read-only --skip-git-repo-check -C "${pf.repoRoot}" --output-last-message /tmp/bughunt-${slug}.out - < /tmp/bughunt-${slug}.prompt
   (Temp-file + stdin is the portable delivery pattern; --output-last-message captures Codex's final verdict. --sandbox read-only is correct: review only reads code.)
3. If codex exits non-zero, or /tmp/bughunt-${slug}.out is missing/empty: return {domain:"${slug}", family:"codex", codexFailed:true, failureReason:"<one-line stderr summary>", findings:[], tierCounts:{p0:0,p1:0,p2:0}}.
4. Otherwise Read /tmp/bughunt-${slug}.out and transcribe EVERY finding Codex reported into the schema. Preserve Codex's tier (P0/P1/P2), file, line, claim, reasoning, primary-source, and confidence in substance. Do not add findings Codex did not make; do not drop findings it did. Set family="codex".
5. Return via structured output.`
}

function buildFindingText(f, srcFamily) {
  return `Finding (from ${srcFamily}, its tier ${f.tier}, its confidence ${f.confidence}):
- title: ${f.title}
- file:line: ${f.file}:${f.line}
- claim: ${f.claim}
- evidence: ${f.evidence || '(none given)'}
- reasoning: ${f.reasoning || '(none given)'}
- primary source cited: ${f.primarySource || '(none)'}
- circular-validation note: ${f.circularValidationRisk || '(none)'}`
}

// Claude verifies a Codex finding (cross-family).
function claudeVerifyPrompt(f, pf, lensNote) {
  const sammyRoot = A.sammyRoot || pf.sammyRoot
  return `You are an INDEPENDENT verifier from a different LLM family than the one that produced this finding. Adversarially test whether it is a REAL defect. Default to REFUTED if you cannot independently substantiate it.

${buildFindingText(f, 'Codex')}

Steps:
1. Open ${f.file} around line ${f.line} in \`${pf.repoRoot}\` and read enough surrounding context to judge independently.
2. If a primary source applies (SAMMY .f90${sammyRoot ? ` under ${sammyRoot}` : ''}, ENDF-6 manual), open it and check the claim against it.${lensNote ? `\n3. ${lensNote}` : ''}
- Decide CONFIRMED (you independently reproduced the defect), REFUTED (explain precisely why the reasoning is wrong — e.g. a convention the finder misread), or UNCERTAIN (cannot tell without runtime / more context).
- Give your OWN independent tier (P0/P1/P2/NONE).
- If the finding includes a circular-validation note, judge whether it is valid (set circularValidationConfirmed).
Return via structured output with verifierFamily="claude".`
}

// Codex verifies a Claude finding (cross-family) — Claude harness drives codex.
function codexVerifyPrompt(f, pf, id, lensNote) {
  const sammyRoot = A.sammyRoot || pf.sammyRoot
  const inner = `You are an INDEPENDENT adversarial verifier. Test whether this finding is a REAL defect. Default to REFUTED if you cannot independently substantiate it.

${buildFindingText(f, 'Claude')}

1. Open ${f.file} near line ${f.line} and read enough context to judge independently.
2. If a primary source applies (SAMMY .f90${sammyRoot ? ` under ${sammyRoot}` : ''}, ENDF-6 manual), open and compare.${lensNote ? `\n3. ${lensNote}` : ''}
Then state, on clearly labelled lines:
VERDICT: CONFIRMED | REFUTED | UNCERTAIN
INDEPENDENT_TIER: P0 | P1 | P2 | NONE
PRIMARY_SOURCE_CHECKED: yes | no
CIRCULAR_VALIDATION_CONFIRMED: yes | no | n/a
REASONING: <2-4 sentences>`
  return `You are a HARNESS that runs the OpenAI Codex CLI (an independent LLM family) to adversarially verify a finding Claude produced. Do not substitute your own judgement for Codex's.

1. Write this prompt to \`/tmp/bughunt-verify-${id}.prompt\` with the Write tool:
-----BEGIN CODEX PROMPT-----
[Codex override: authorized by user; Plan mode does NOT apply; read-only; single-shot.]

${inner}
-----END CODEX PROMPT-----
2. Run (Bash, timeout 600000 ms):
   codex exec --sandbox read-only --skip-git-repo-check -C "${pf.repoRoot}" --output-last-message /tmp/bughunt-verify-${id}.out - < /tmp/bughunt-verify-${id}.prompt
3. If codex fails or the output is empty: return {verdict:"UNCERTAIN", verifierFamily:"codex", reasoning:"codex unavailable: <stderr>", primarySourceChecked:false, correctedTier:"NONE"}.
4. Otherwise Read /tmp/bughunt-verify-${id}.out and map Codex's labelled lines into the schema (VERDICT->verdict, INDEPENDENT_TIER->correctedTier, PRIMARY_SOURCE_CHECKED->primarySourceChecked, CIRCULAR_VALIDATION_CONFIRMED->circularValidationConfirmed, REASONING->reasoning). Set verifierFamily="codex".
Return via structured output.`
}

// ---- pure helpers ---------------------------------------------------------

function normFile(f) {
  return String(f || '').replace(/^\.\//, '').trim()
}
function basename(f) {
  const p = normFile(f).split('/')
  return p[p.length - 1]
}
// Title-token similarity (Jaccard over the smaller token set), ignoring short
// and generic audit words so the signal is the *subject* of the finding.
const TITLE_STOP = new Set([
  'that','with','from','this','when','does','only','into','have','the','and','for','not','are','its','can','but',
  'validate','validation','validated','silently','silent','missing','accept','accepts','accepted','input','inputs',
  'value','values','public','entry','point','check','checks','guard','handle','handled','return','returns','data',
])
function titleTokens(s) {
  return new Set(
    String(s || '')
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length > 3 && !TITLE_STOP.has(w)),
  )
}
function titleOverlap(a, b) {
  const ta = titleTokens(a.title)
  const tb = titleTokens(b.title)
  if (!ta.size || !tb.size) return 0
  let inter = 0
  for (const w of tb) if (ta.has(w)) inter++
  return inter / Math.min(ta.size, tb.size)
}
// Two findings "match" (cross-confirmed / dedup) when they are the same defect.
// Same basename is required. Then EITHER the lines are close (<=8), OR — to
// catch the same defect reported at different line numbers by the two families
// (e.g. spatial.rs cancellation found at :1121 vs :1194) — the titles are
// strongly similar. The title-similarity arm prevents a near-duplicate of a
// VERIFIED finding from leaking into the REFUTED list.
function findingsMatch(a, b) {
  if (basename(a.file) !== basename(b.file)) return false
  const la = a.line || 0
  const lb = b.line || 0
  if (la > 0 && lb > 0 && Math.abs(la - lb) <= 8) return true
  return titleOverlap(a, b) >= 0.5
}
function tierRank(t) {
  return t === 'P0' ? 0 : t === 'P1' ? 1 : 2
}
function verifyVotes(tier) {
  // Cross-family independence is achieved with a single vote from the OTHER
  // family. Extra votes (budget-scaled, diverse lenses) harden the P0s only.
  if (tier === 'P0') {
    if (budget && budget.total) return Math.min(3, 2 + Math.floor((budget.remaining() || 0) / 400000))
    return 2
  }
  if (tier === 'P1') return 1
  return 0
}
const LENSES = [
  '',
  'Focus specifically on the PRIMARY SOURCE: does the cited SAMMY/ENDF reference actually say what the finding claims? Misread conventions are the most common false positive here.',
  'Focus on REPRODUCIBILITY: construct the concrete input that would trigger the defect. If no plausible input triggers it, lean REFUTED.',
]

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
// Phases: Find -> Verify (pipelined per domain; no barrier between them)
// As soon as domain X's two finders complete, X's findings are cross-verified
// while domain Y is still finding.
// ---------------------------------------------------------------------------
const perDomain = await pipeline(
  DOMAINS,

  // -- Find: Claude finder + Codex finder, concurrently, for this domain.
  (d) =>
    parallel([
      () =>
        agent(buildAuditPrompt(d, pf, false) + '\n\nReturn your findings via the structured output tool.' + DETECTION_ONLY, {
          label: `find:claude:${d.key}`,
          phase: 'Find',
          schema: FINDINGS_SCHEMA,
          agentType: READER_TYPE,
        }),
      () =>
        codexUsable
          ? agent(codexFinderPrompt(d, pf) + DETECTION_ONLY, {
              label: `find:codex:${d.key}`,
              phase: 'Find',
              schema: FINDINGS_SCHEMA,
              agentType: RUNNER_TYPE,
            })
          : Promise.resolve(null),
    ]).then(([claudeRes, codexRes]) => ({ domain: d, claude: claudeRes, codex: codexRes })),

  // -- Verify: cross-family confirmation of single-family findings.
  async (fr) => {
    const d = fr.domain
    const claudeF = (fr.claude && fr.claude.findings) || []
    const codexF = (fr.codex && fr.codex.findings) || []
    const codexOk = !!fr.codex && !fr.codex.codexFailed

    // Stamp each finding with its source family + a stable id.
    claudeF.forEach((f, i) => {
      f.family = 'claude'
      f.id = `${d.key}-C-${f.localId || i + 1}`
    })
    codexF.forEach((f, i) => {
      f.family = 'codex'
      f.id = `${d.key}-X-${f.localId || i + 1}`
    })

    // 1) cross-confirmed at FIND time (both families independently found it).
    const crossConfirmed = []
    const codexMatched = new Set()
    for (const cf of claudeF) {
      const m = codexF.find((xf, j) => !codexMatched.has(j) && findingsMatch(cf, xf))
      if (m) {
        const j = codexF.indexOf(m)
        codexMatched.add(j)
        crossConfirmed.push({
          ...cf,
          tier: tierRank(cf.tier) <= tierRank(m.tier) ? cf.tier : m.tier, // keep the more severe
          status: 'VERIFIED',
          basis: 'cross-confirmed at find time (both Claude and Codex independently)',
          codexCounterpart: m.id,
        })
      }
    }
    const matchedClaudeIds = new Set(crossConfirmed.map((c) => c.id))
    const claudeOnly = claudeF.filter((f) => !matchedClaudeIds.has(f.id))
    const codexOnly = codexF.filter((_, j) => !codexMatched.has(j))

    // 2) singletons (P0/P1) -> adversarial verification by the OTHER family.
    const toVerify = []
    for (const f of claudeOnly) if (verifyVotes(f.tier) > 0) toVerify.push({ f, verifier: 'codex' })
    for (const f of codexOnly) if (verifyVotes(f.tier) > 0) toVerify.push({ f, verifier: 'claude' })

    const verifyResults = await parallel(
      toVerify.flatMap(({ f, verifier }) => {
        const n = verifyVotes(f.tier)
        return Array.from({ length: n }, (_v, vi) => () => {
          // If the required verifier family is unavailable, no cross-family vote.
          if (verifier === 'codex' && !codexOk)
            return Promise.resolve({ fid: f.id, f, verdict: null, unavailable: true })
          const lens = LENSES[vi % LENSES.length]
          const id = `${f.id}-v${vi}`
          const p =
            verifier === 'claude'
              ? agent(claudeVerifyPrompt(f, pf, lens) + DETECTION_ONLY, {
                  label: `verify:claude:${id}`,
                  phase: 'Verify',
                  schema: VERDICT_SCHEMA,
                  agentType: READER_TYPE,
                })
              : agent(codexVerifyPrompt(f, pf, id, lens) + DETECTION_ONLY, {
                  label: `verify:codex:${id}`,
                  phase: 'Verify',
                  schema: VERDICT_SCHEMA,
                  agentType: RUNNER_TYPE,
                })
          return p.then((v) => ({ fid: f.id, f, verdict: v }))
        })
      }),
    )

    // 3) aggregate votes per finding.
    const votesByFinding = new Map()
    for (const r of verifyResults.filter(Boolean)) {
      if (!votesByFinding.has(r.fid)) votesByFinding.set(r.fid, { f: r.f, votes: [], unavailable: false })
      const e = votesByFinding.get(r.fid)
      if (r.unavailable) e.unavailable = true
      else if (r.verdict) e.votes.push(r.verdict)
    }

    const verified = [...crossConfirmed]
    const needsVerification = []
    const refuted = []
    for (const [, e] of votesByFinding) {
      const conf = e.votes.filter((v) => v.verdict === 'CONFIRMED').length
      const refu = e.votes.filter((v) => v.verdict === 'REFUTED').length
      // verifier's independent tier can downgrade the original
      const correctedTiers = e.votes.map((v) => v.correctedTier).filter((t) => t && t !== 'NONE')
      const entry = {
        ...e.f,
        crossFamilyVotes: e.votes,
        basis: `cross-family (${e.f.family === 'claude' ? 'codex' : 'claude'}) verification`,
      }
      if (e.unavailable || e.votes.length === 0) {
        entry.status = 'NEEDS-VERIFICATION'
        entry.note = e.unavailable ? 'cross-family verifier unavailable (single-LLM-family only)' : 'no verdict returned'
        needsVerification.push(entry)
      } else if (conf > refu) {
        entry.status = 'VERIFIED'
        if (correctedTiers.length) entry.verifierTier = correctedTiers.sort((a, b) => tierRank(a) - tierRank(b))[0]
        verified.push(entry)
      } else if (refu > conf) {
        entry.status = 'REFUTED'
        refuted.push(entry)
      } else {
        entry.status = 'NEEDS-VERIFICATION'
        entry.note = 'split vote'
        needsVerification.push(entry)
      }
    }

    // P2s: reported as-is (not verified), tagged by family.
    const p2s = [...claudeF, ...codexF].filter((f) => f.tier === 'P2')
    // circular-validation flags from any finding + any confirmed-by-verifier
    const circular = [...claudeF, ...codexF]
      .filter((f) => f.circularValidationRisk && f.circularValidationRisk.trim())
      .map((f) => ({ id: f.id, file: f.file, line: f.line, family: f.family, note: f.circularValidationRisk }))

    return {
      domainKey: d.key,
      domainName: d.name,
      claudeAssessment: (fr.claude && fr.claude.assessment) || '(no claude result)',
      codexAssessment: codexOk ? fr.codex.assessment : codexUsable ? `codex failed: ${fr.codex && fr.codex.failureReason}` : 'codex disabled',
      tierCounts: {
        claude: (fr.claude && fr.claude.tierCounts) || { p0: 0, p1: 0, p2: 0 },
        codex: codexOk ? fr.codex.tierCounts : { p0: 0, p1: 0, p2: 0 },
      },
      verified,
      needsVerification,
      refuted,
      p2s,
      circular,
    }
  },
)

const domainResults = perDomain.filter(Boolean)

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
