export const meta = {
  name: 'dual-family-review',
  description:
    'Shared engine for two-LLM-family code review: per-target Claude + Codex finders, cross-LLM-family adversarial verification, structured per-target findings. Detection-only. A LEAF workflow (never calls workflow()).',
  whenToUse:
    'Not invoked directly by a human. This is the reusable core consumed via workflow() by nereids-bug-hunt (whole-crate sweep, target = crate/domain) and review-core (per-PR-diff, target = worktree branch). It runs the Find -> Verify pipeline over a generic list of audit targets and returns structured per-target findings; each caller does its own preflight (building `pf`), target construction, and final consolidation/reporting.',
  phases: [
    { title: 'Find', detail: 'Claude + Codex finder per target (up to 2N agents)' },
    { title: 'Verify', detail: 'cross-LLM-family adversarial confirmation of single-family findings' },
  ],
}

// ---------------------------------------------------------------------------
// dual-family-review — the shared two-family review engine.
//
// Extracted verbatim from nereids-bug-hunt.js (PR #591) so the two consumers
// (nereids-bug-hunt + review-core) share ONE implementation of the audit
// methodology this project converged on:
//   - 2 independent LLM families per target: Claude (agent()) + Codex (codex exec)
//   - P0/P1/P2 tiers (P3 excluded), file:line + primary-source evidence
//   - a cross-LLM-family confirmation pass on every SINGLE-family P0/P1 finding
//     (Claude findings verified by Codex and vice versa; P2s are reported as-is,
//     NOT cross-verified — verifyVotes() returns 0 for P2)
//
// Why the cross-family pass is load-bearing (see memory):
//   - "parent-agent re-derivation of a subagent's claim is NOT independent
//      confirmation; same LLM family. For VERIFIED need >=2 distinct LLM
//      families with independent access."
//   - "the audit->fix->review cycle is non-idempotent" — so this engine
//      DETECTS only; fixing is a separate, human-gated step.
//
// Detection-only is enforced two ways: (1) the DETECTION_ONLY prompt contract
// appended to every audit-agent prompt, ALWAYS ON — this is what makes
// behaviour consistent across models; (2) optional TOOL-LEVEL enforcement via
// restricted custom agent types (.claude/agents/bug-hunt-{reader,runner}.md)
// that physically lack spawn_task/Task*/MCP/PR-issue tools — enable with
// config.hardEnforce=true from a session started AFTER those agent defs exist.
//
// Two SEMANTIC generalizations vs the original inline bug-hunt code:
//   - the global MODE/DIFF_SPEC scope ternary is replaced by a per-target
//     `scopeDirective` string (each caller decides full-crate vs diff scope);
//   - the SAMMY-root override reads config.sammyRootOverride instead of A.sammyRoot.
// Plus behaviour-neutral re-plumbing: CONTEXT_NOTE / ROUND_NOTE / SKIP_CODEX /
// HARD now come from the passed-in `config` (not the caller's `A.*`),
// `codexUsable` is computed at module scope, and the engine RETURNS structured
// per-target results instead of continuing inline to a consolidation phase
// (preflight + consolidation now live in each wrapper).
// The prompt builders, schemas, dedup/verify helpers, and the Find->Verify
// pipeline are byte-for-byte the original logic (verified by a prompt-equivalence
// harness: 47/47 byte-identical agent prompts).
//
// args (passed in by the calling wrapper):
//   args.pf      : { repoRoot, sammyRoot, codexAvailable, codexVersion, headSha, isWorktree }
//                  — the calling wrapper's preflight result.
//   args.targets : [{ key, name, paths[], sammy, blurb, p0, lookFor[], primaryHint, scopeDirective }]
//                  — one entry per audit unit (a crate/domain, or a branch diff).
//   args.config  : { contextNote, roundNote, skipCodex, hardEnforce, sammyRootOverride }
//
// returns: { perTarget: [{ domainKey, domainName, claudeAssessment, codexAssessment,
//            tierCounts:{claude,codex}, verified[], needsVerification[], refuted[],
//            p2s[], circular[] }], codexUsable }
// ---------------------------------------------------------------------------

// Defensive arg unpacking — `args` should arrive as a real object, but a caller
// may pass a JSON-encoded string; parse it rather than silently default.
let A = {}
if (args && typeof args === 'object') A = args
else if (typeof args === 'string' && args.trim().startsWith('{')) {
  try {
    A = JSON.parse(args)
  } catch (_e) {
    A = {}
  }
}

const pf = A.pf && typeof A.pf === 'object' ? A.pf : {}
const TARGETS = Array.isArray(A.targets) ? A.targets : []
const CONFIG = A.config && typeof A.config === 'object' ? A.config : {}
const CONTEXT_NOTE = CONFIG.contextNote || ''
const ROUND_NOTE = CONFIG.roundNote || 'review'
const SKIP_CODEX = CONFIG.skipCodex === true
const SAMMY_ROOT_OVERRIDE = CONFIG.sammyRootOverride || ''

// Detection-only enforcement. Two layers: (1) the DETECTION_ONLY prompt contract
// below, appended to every audit-agent prompt — ALWAYS ON and identical across
// agents; (2) optional TOOL-LEVEL hard enforcement via restricted custom agent
// types. Custom agent types load at SESSION START (not hot), so default to the
// always-available built-in `general-purpose`; pass config.hardEnforce=true from
// a session started AFTER those agent defs exist to switch the restriction on.
const HARD = CONFIG.hardEnforce === true
const READER_TYPE = HARD ? 'bug-hunt-reader' : 'general-purpose'
const RUNNER_TYPE = HARD ? 'bug-hunt-runner' : 'general-purpose'

const codexUsable = !!pf.codexAvailable && !SKIP_CODEX

const DETECTION_ONLY =
  '\n\n--- DETECTION-ONLY CONTRACT (mandatory) ---\n' +
  'You FIND and REPORT defects; you never fix them and never advance to a fix stage. You MUST NOT:\n' +
  '- create any task, background session, or fix-job (do NOT call spawn_task or any task-creation / scheduling tool);\n' +
  '- open or modify any PR or issue;\n' +
  '- edit or write repository files, or run state-changing shell commands (git commit/push/add/checkout, gh, cargo fix, installs).\n' +
  'The only writes ever permitted are scratch /tmp files (and, for the consolidator, the one explicitly-named .research report path).\n' +
  'Report via the structured-output tool only. Fixing is a separate, human-gated step.'

// Shared checklist appended to every target (DRY — matches the project's own
// DRY governance). Target-specific bullets (target.lookFor) are prepended.
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

// ---- prompt builders ------------------------------------------------------

function buildAuditPrompt(d, pf, forCodex) {
  const sammyRoot = SAMMY_ROOT_OVERRIDE || pf.sammyRoot
  const codexOverride = forCodex
    ? '[Execution context: you are a non-interactive, read-only audit subprocess of this repository’s review pipeline; no human is at this terminal to answer prompts. The AGENTS.md / CLAUDE.md planning checkpoints govern interactive change-making sessions, not this detection-only run. Execute in a single run; do not stop for confirmation. Change no files, open no PRs/issues. Single-shot delivery.]\n\n'
    : ''
  const ctx = CONTEXT_NOTE ? `[CONTEXT]\n${CONTEXT_NOTE}\n[End context]\n\n` : ''
  const scope = d.scopeDirective
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
  const sammyRoot = SAMMY_ROOT_OVERRIDE || pf.sammyRoot
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
  const sammyRoot = SAMMY_ROOT_OVERRIDE || pf.sammyRoot
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
[Execution context: non-interactive read-only verification subprocess of the review pipeline; no human at this terminal; single-shot — do not stop for confirmation.]

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
// Find -> Verify (pipelined per target; no barrier between them).
// As soon as target X's two finders complete, X's findings are cross-verified
// while target Y is still finding.
//
// Progress grouping uses each agent's `phase:` opt (not top-level phase('Find')
// / phase('Verify') calls): because the pipeline interleaves one target's Verify
// with another target's Find, explicit phase() boundaries would misrepresent the
// interleaving. meta.phases still declares Find/Verify for the /workflows display.
// ---------------------------------------------------------------------------

// Fail loudly on a malformed target rather than degrade silently. A target
// missing `scopeDirective` would otherwise inject the literal string "undefined"
// into the audit prompt (missing `paths`/`lookFor` already throw in
// buildAuditPrompt -> the pipeline drops that target -> the caller's
// count-mismatch guard aborts; the missing-scopeDirective case is the silent one).
// Not reachable from the current callers (both always set scopeDirective) — this
// is defensive depth for future callers.
for (const t of TARGETS) {
  const key = (t && t.key) || '(unknown)'
  if (!t || typeof t.key !== 'string' || !t.key) throw new Error('dual-family-review: a target is missing a string `key`')
  if (typeof t.scopeDirective !== 'string' || !t.scopeDirective)
    throw new Error(`dual-family-review: target '${key}' is missing a non-empty 'scopeDirective'`)
  if (!Array.isArray(t.paths) || !t.paths.length) throw new Error(`dual-family-review: target '${key}' is missing non-empty 'paths'`)
  if (!Array.isArray(t.lookFor)) throw new Error(`dual-family-review: target '${key}' 'lookFor' must be an array`)
}

const perTargetRaw = await pipeline(
  TARGETS,

  // -- Find: Claude finder + Codex finder, concurrently, for this target.
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

// Return structured per-target findings to the calling wrapper. The wrapper
// owns final consolidation (cross-target dedup, report writing, disposition,
// RECURRING tagging) — this engine is detection + per-target structuring only.
//
// Surface dropped targets (a per-target pipeline that threw -> null) so the
// wrapper can reconcile perTarget.length against its target count and fail
// closed rather than silently under-report. The wrapper (not the engine) owns
// the empty-targets policy: a sweep with no matching targets is an error there,
// while a review round with no diverged branches is a valid no-op.
const perTarget = perTargetRaw.filter(Boolean)
const droppedTargets = perTargetRaw.length - perTarget.length
if (droppedTargets > 0) {
  log(
    `WARNING: ${droppedTargets}/${perTargetRaw.length} target(s) produced no result (pipeline error) and were dropped — the caller should fail closed.`,
  )
}
return { perTarget, droppedTargets, codexUsable }
