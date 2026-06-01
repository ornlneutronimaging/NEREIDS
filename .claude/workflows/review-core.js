export const meta = {
  name: 'review-core',
  description:
    'Deterministic core of /review-pipeline for ONE review round: discover diverged branches, run the two-LLM-family review engine over each branch diff, consolidate into per-branch findings + suggested disposition + merge order. Detection-only — returns structured data for the skill to gate on; never fixes, pushes, or merges.',
  whenToUse:
    'Invoked by the /review-pipeline skill once per review round (NOT run directly by a human). It owns the non-interactive fan-out — discover + dual-family review + consolidate — so reviewer output is one schema-forced format across models and the steps cannot be silently skipped. All user gates, fixes, pushes, merges, and issue-filing stay in the skill (a workflow cannot pause for input, so every gate must fall on a workflow boundary).',
  phases: [
    { title: 'Discover', detail: 'enumerate diverged branches, diffs, file-overlap, merge order' },
    { title: 'Find', detail: 'Claude + Codex finder per branch diff (dual-family-review engine)' },
    { title: 'Verify', detail: 'cross-LLM-family confirmation (dual-family-review engine)' },
    { title: 'Consolidate', detail: 'suggested disposition, RECURRING tags, merge order (deterministic JS)' },
  ],
}

// ---------------------------------------------------------------------------
// review-core — the deterministic heart of /review-pipeline.
//
// One invocation == one review round. The /review-pipeline skill calls this
// per round (passing the previous round's findings for RECURRING detection),
// renders the returned structured payload as a consistent per-branch table at
// the user gate, then — only after the user approves — drives the fix / push /
// Copilot / merge steps itself. Those are interactive or state-changing, and a
// Workflow runs unattended to completion, so they MUST stay in the skill.
//
// What this workflow guarantees (the two goals of the refactor):
//   - consistent format across models: every reviewer (Claude finder, Codex
//     finder, and every cross-family verifier) emits the SAME schema, via the
//     shared dual-family-review engine; cross-confirm + dedup are deterministic
//     JS, not the main agent eyeballing prose;
//   - no silently-skipped steps: discover -> find(2 families) -> verify ->
//     consolidate is JS control flow, run for every branch every round.
//
// Detection-only: this workflow + the engine FIND and REPORT; they never fix.
// Fixing is a separate, human-gated step (the skill's fix stage +
// .claude/templates/fix-subagent-prompt.md). The audit->fix->review cycle is
// non-idempotent, which is exactly why detection and fixing are separated.
//
// args (all optional):
//   args.branches     : string[]  explicit branch list to review (default:
//                                  auto-discover diverged worktree branches)
//   args.base         : string    base ref to diff against (default 'main')
//   args.round        : integer   review round number, for the header (default 1)
//   args.priorFindings: object[]  last round's findings ({branch,file,line,title})
//                                  used to tag RECURRING (default none)
//   args.skipCodex    : boolean   Claude-only (single-family -> P0/P1 NEEDS-VERIFICATION;
//                                  P2s are single-family-reported either way)
//   args.hardEnforce  : boolean   use restricted bug-hunt-{reader,runner} agents
//   args.repoRoot     : string    explicit canonical root (overrides git-detected)
//   args.sammyRoot    : string    override SAMMY source root for physics checks
// ---------------------------------------------------------------------------

// Defensive arg unpacking (a caller may pass a JSON-encoded string).
let A = {}
if (args && typeof args === 'object') A = args
else if (typeof args === 'string' && args.trim().startsWith('{')) {
  try {
    A = JSON.parse(args)
  } catch (_e) {
    A = {}
  }
}
const BRANCHES = Array.isArray(A.branches) && A.branches.length ? A.branches : null
const BASE = A.base || 'main'
const ROUND = Number.isInteger(A.round) ? A.round : 1
const PRIOR = Array.isArray(A.priorFindings) ? A.priorFindings : []
const SKIP_CODEX = A.skipCodex === true
const HARD = A.hardEnforce === true
const READER_TYPE = HARD ? 'bug-hunt-reader' : 'general-purpose'

// Primary-source hints for branches that touch physics / ENDF (so the engine's
// finder opens SAMMY / the ENDF manual). Compact on purpose — the engine's
// CORE_CHECKLIST covers the generic bug classes; this only adds the SAMMY hook.
const PHYSICS_HINT =
  'SAMMY SLBW/MLBW `src/mlb/`, phase/penetrability `src/xxx/`, RML `src/rml/`, Reich-Moore `src/endf/`. Compare physics line-by-line and cite `file:line`.'
const ENDF_HINT = 'ENDF-6 Formats Manual record layouts; SAMMY `src/endf/` for LRF=3/7 reference.'

// ---- pure helpers ---------------------------------------------------------

function normFile(f) {
  return String(f || '').replace(/^\.\//, '').trim()
}
function basename(f) {
  const p = normFile(f).split('/')
  return p[p.length - 1]
}
// Slug a branch name into something safe for finding-ids and the engine's
// `/tmp/bughunt-<slug>.prompt` codex temp path (branch names contain '/').
// COLLISION-RESISTANT: a short stable hash of the FULL name is appended so
// distinct branches (e.g. `fix/foo-bar` vs `fix/foo/bar`) do not collide to one
// key — a collision would attach wrong crate metadata to one branch's findings
// and race the two branches' codex temp files. The 32-bit djb2 hash is not
// provably injective (a hash CAN collide), but across the handful of branches in
// a review batch the probability is negligible. Deterministic (Date/random are
// unavailable in workflow scripts).
function slug(b) {
  const s = String(b || '')
  const base = s.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'branch'
  let h = 5381
  for (let i = 0; i < s.length; i++) h = (((h << 5) + h) ^ s.charCodeAt(i)) | 0
  return `${base}-${(h >>> 0).toString(36)}`
}
// Which crate (top-2 path segments) a changed file belongs to, for the blurb.
function crateOf(f) {
  const parts = normFile(f).split('/')
  if (parts[0] === 'crates' || parts[0] === 'bindings' || parts[0] === 'apps') return parts.slice(0, 2).join('/')
  return parts[0] || '(root)'
}
function classifyBranch(changedFiles) {
  const files = (changedFiles || []).map(normFile).filter(Boolean)
  const touchesPhysics = files.some((f) => f.startsWith('crates/nereids-physics/'))
  const touchesEndf = files.some((f) => f.startsWith('crates/nereids-endf/') || f.startsWith('crates/endf-mat/'))
  const sammy = touchesPhysics || touchesEndf
  const primaryHint = touchesPhysics ? PHYSICS_HINT : touchesEndf ? ENDF_HINT : ''
  const crates = [...new Set(files.map(crateOf))]
  // Focus the finder on the changed crate dirs (deduped, dir-level).
  const focusPaths = [...new Set(files.map((f) => normFile(f).split('/').slice(0, 2).join('/') + '/'))]
  return { sammy, primaryHint, crates, focusPaths: focusPaths.length ? focusPaths : ['(repo root)'] }
}
// A this-round finding "recurs" if a prior-round finding hit the same branch +
// same file basename and either a close line or a near-identical title.
function isRecurring(branch, f) {
  const fl = f.line || 0
  return PRIOR.some((p) => {
    if (p.branch && p.branch !== branch) return false
    if (basename(p.file) !== basename(f.file)) return false
    const pl = p.line || 0
    if (pl > 0 && fl > 0 && Math.abs(pl - fl) <= 8) return true
    return String(p.title || '').trim() && String(p.title).trim() === String(f.title || '').trim()
  })
}
// Branch whose changed files this finding lives in (for same-crate P2 discipline).
function fileInChangedCrates(file, crates) {
  const c = crateOf(file)
  return crates.includes(c)
}

const DISCOVER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    repoRoot: { type: 'string' },
    isWorktree: { type: 'boolean' },
    codexAvailable: { type: 'boolean' },
    codexVersion: { type: 'string' },
    sammyRoot: { type: 'string', description: 'resolved SAMMY src root, or empty if not found' },
    headSha: { type: 'string' },
    base: { type: 'string' },
    branches: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          branch: { type: 'string' },
          changedFiles: { type: 'array', items: { type: 'string' } },
          diffLineCount: { type: 'integer', description: 'total added+deleted lines vs base; 0 if unknown' },
          diverged: { type: 'boolean', description: 'true if the branch has commits not in base' },
        },
        required: ['branch', 'changedFiles', 'diverged'],
      },
    },
  },
  required: ['repoRoot', 'isWorktree', 'codexAvailable', 'base', 'branches'],
}

// ---------------------------------------------------------------------------
// Phase: Discover — enumerate target branches + their diffs (read-only git).
// ---------------------------------------------------------------------------
phase('Discover')

const branchInstruction = BRANCHES
  ? `Review exactly these branches (do NOT auto-discover): ${BRANCHES.join(', ')}.`
  : `Auto-discover targets: run \`git worktree list\` and, for each worktree under \`.claude/worktrees/\`, take its checked-out branch. Skip the base branch itself.`

const discover = await agent(
  `Resolve the environment + discover branches to review for a NEREIDS review round. Read-only git only; change nothing. Report via structured output.

Base ref: \`${BASE}\`.
${branchInstruction}

Gather:
- repoRoot: \`git rev-parse --show-toplevel\`${A.repoRoot ? ` (but report the canonical root \`${A.repoRoot}\`)` : ''}
- isWorktree: true if repoRoot contains "/.claude/worktrees/" OR \`git rev-parse --git-common-dir\` differs from \`git rev-parse --git-dir\`
- headSha: \`git rev-parse --short HEAD\`
- codexAvailable / codexVersion: \`command -v codex && codex --version\` (false + empty if absent)
- sammyRoot: first of these that exists as a directory (test -d), else empty:
    /Users/8cz/code-int.ornl.gov/sammy/sammy/src
    "$(git rev-parse --show-toplevel)/../SAMMY/src"
    "$(git rev-parse --show-toplevel)/../SAMMY/sammy/src"
- base: "${BASE}"
- branches: for each target branch, report:
    - branch: the branch name
    - diverged: true if \`git rev-list --count ${BASE}..<branch>\` > 0 (it has commits not in ${BASE})
    - changedFiles: the output of \`git diff --name-only ${BASE}...<branch>\` (the three-dot merge-base diff), as a list
    - diffLineCount: total added+deleted lines, from \`git diff --shortstat ${BASE}...<branch>\` (sum insertions+deletions; 0 if none)
  Use branch NAMES with git (refs are shared across worktrees); do NOT cd into sibling worktree paths and do NOT use \`git -C <other-worktree>\` (sandboxing blocks that). \`git diff ${BASE}...<branch>\` and \`git show <branch>:<path>\` resolve fine from here.

Be terse; just gather facts.`,
  { label: 'discover', phase: 'Discover', schema: DISCOVER_SCHEMA, agentType: READER_TYPE },
)

if (!discover) {
  log('Discover was skipped — aborting review-core.')
  return { aborted: true, reason: 'discover skipped' }
}
if (A.repoRoot) discover.repoRoot = A.repoRoot
const sammyRoot = A.sammyRoot || discover.sammyRoot || ''
const codexUsable = discover.codexAvailable && !SKIP_CODEX

// Fail closed on a requested branch that discovery did not return (mistyped /
// unresolvable ref, or a failed git command): a dropped EXPLICITLY-requested
// branch must abort, not masquerade as "nothing to review". (Auto-discover
// finding nothing is a valid no-op; an explicit request going missing is not.)
if (BRANCHES) {
  const found = new Set((discover.branches || []).map((b) => b.branch))
  const missing = BRANCHES.filter((b) => !found.has(b))
  if (missing.length) {
    throw new Error(
      `review-core: discovery did not return requested branch(es) [${missing.join(', ')}] vs base '${BASE}' — aborting rather than report a false all-clear. Check the branch name(s) and that '${BASE}' resolves.`,
    )
  }
}

// Only review branches that actually diverged from base (have new commits).
const reviewable = (discover.branches || []).filter((b) => b.diverged && (b.changedFiles || []).length)
log(
  `round ${ROUND} | base=${BASE} | branches=${reviewable.length}/${(discover.branches || []).length} reviewable | ` +
    `codex=${codexUsable ? discover.codexVersion || 'yes' : 'DISABLED (single-family — P0/P1 findings NEEDS-VERIFICATION)'} | ` +
    `sammy=${sammyRoot || 'NOT FOUND (physics findings lower-confidence)'}`,
)

if (!reviewable.length) {
  log('No diverged branches with changes — nothing to review.')
  return {
    meta: { round: ROUND, base: BASE, repoRoot: discover.repoRoot, headSha: discover.headSha, codexUsable, sammyRoot: sammyRoot || '(not found)', isWorktree: discover.isWorktree },
    perBranch: [],
    mergeOrder: [],
    overlaps: [],
    deferredP2s: [],
    counts: { branches: 0, verifiedP0: 0, verifiedP1: 0, needsVerification: 0, refuted: 0, p2: 0, recurring: 0 },
  }
}

// File-overlap matrix + suggested merge order (deterministic JS, not LLM-judged).
const fileSets = new Map(reviewable.map((b) => [b.branch, new Set((b.changedFiles || []).map(normFile))]))
const overlaps = []
for (let i = 0; i < reviewable.length; i++) {
  for (let j = i + 1; j < reviewable.length; j++) {
    const a = reviewable[i].branch
    const b = reviewable[j].branch
    const shared = [...fileSets.get(a)].filter((f) => fileSets.get(b).has(f))
    if (shared.length) overlaps.push({ a, b, sharedFiles: shared })
  }
}
// Merge order: smallest diff first (a larger diff is likelier to need rebasing
// after a smaller overlapping one lands). Non-overlapping branches are
// parallel-safe and fall out of the same size sort.
const mergeOrder = [...reviewable]
  .sort((x, y) => (x.diffLineCount || 0) - (y.diffLineCount || 0) || x.branch.localeCompare(y.branch))
  .map((b) => b.branch)

// ---------------------------------------------------------------------------
// Build one engine target per branch and hand off to dual-family-review.
// ---------------------------------------------------------------------------
const branchMeta = new Map()
const targets = reviewable.map((b) => {
  const cls = classifyBranch(b.changedFiles)
  branchMeta.set(slug(b.branch), { branch: b.branch, crates: cls.crates })
  return {
    key: slug(b.branch),
    name: `branch ${b.branch}`,
    paths: cls.focusPaths,
    sammy: cls.sammy,
    blurb: `the changes on branch \`${b.branch}\` vs \`${BASE}\` (${(b.changedFiles || []).length} files in: ${cls.crates.join(', ') || '(root)'})`,
    p0: 'A correctness regression introduced by this branch: a logic bug, a panic on valid input, a numerical/physics error vs SAMMY, data corruption, or a silently-masked whole-config error.',
    lookFor: [],
    primaryHint: cls.primaryHint,
    scopeDirective: `Audit ONLY the changes in \`git diff ${BASE}...${b.branch}\`. Run that diff, then read each changed file IN FULL via \`git show ${b.branch}:<path>\` (the branch may not be checked out here). Report findings only for code introduced or altered by this branch's diff, plus any pre-existing call site the diff newly breaks.`,
  }
})

const engineOut = await workflow('dual-family-review', {
  pf: {
    repoRoot: discover.repoRoot,
    sammyRoot,
    codexAvailable: discover.codexAvailable,
    codexVersion: discover.codexVersion,
    headSha: discover.headSha,
    isWorktree: discover.isWorktree,
  },
  targets,
  config: {
    contextNote: `Per-PR review round ${ROUND}. Each target is one branch's diff vs ${BASE}.`,
    roundNote: `review Round ${ROUND}`,
    skipCodex: SKIP_CODEX,
    hardEnforce: HARD,
    sammyRootOverride: A.sammyRoot || '',
  },
})

// Fail closed: a missing/malformed or count-mismatched engine result must abort,
// not silently become "zero reviewed branches". The engine surfaces droppedTargets
// (a per-target pipeline that threw) so we reconcile against the target count.
if (!engineOut || !Array.isArray(engineOut.perTarget)) {
  throw new Error('dual-family-review returned no usable result (engineOut.perTarget missing) — aborting rather than report zero reviewed branches.')
}
if (engineOut.perTarget.length !== targets.length) {
  throw new Error(
    `dual-family-review returned ${engineOut.perTarget.length} result(s) for ${targets.length} target(s)` +
      `${engineOut.droppedTargets ? ` (${engineOut.droppedTargets} dropped to a pipeline error)` : ''} — aborting rather than under-report.`,
  )
}
const perTarget = engineOut.perTarget

// ---------------------------------------------------------------------------
// Phase: Consolidate (deterministic JS) — per-branch findings, suggested
// disposition, RECURRING tags, deferred-P2 list. No file is written; the
// structured payload goes back to the skill, which renders + gates on it.
// ---------------------------------------------------------------------------
phase('Consolidate')

const deferredP2s = []
const perBranch = perTarget.map((r) => {
  const meta = branchMeta.get(r.domainKey) || { branch: r.domainKey, crates: [] }
  const branch = meta.branch
  const tag = (f) => ({ ...f, branch, recurring: isRecurring(branch, f) })

  const verified = (r.verified || []).map(tag)
  const verifiedP0 = verified.filter((f) => (f.verifierTier || f.tier) === 'P0')
  const verifiedP1 = verified.filter((f) => (f.verifierTier || f.tier) === 'P1')
  // Cross-family-CONFIRMED findings the verifier DOWNGRADED to effective P2
  // (original tier P0/P1) are real defects; route them into the P2 disposition
  // channel below so they are surfaced (with a disposition + their VERIFIED
  // status) rather than dropped between the P0/P1 buckets and r.p2s. Restrict to
  // GENUINELY-downgraded findings (verifierTier === 'P2' && tier !== 'P2'): a
  // born-P2 finding is already in r.p2s, so matching it here would double-count it.
  const verifiedP2 = verified.filter((f) => f.verifierTier === 'P2' && f.tier !== 'P2')
  const needsVerification = (r.needsVerification || []).map(tag)
  const refuted = (r.refuted || []).map(tag)
  const circular = r.circular || []

  // Suggested disposition (the skill presents these; the USER decides at the gate):
  //  - verified P0/P1  -> fix-now
  //  - P2 in a crate this branch already changes -> fix-now (same-crate P2 discipline)
  //  - P2 elsewhere    -> defer (collected into deferredP2s for issue-filing)
  //  - refuted         -> dismiss
  const p2s = [...(r.p2s || []), ...verifiedP2].map((f) => {
    const sameCrate = fileInChangedCrates(f.file, meta.crates)
    const disposition = sameCrate ? 'fix-now' : 'defer'
    const entry = { ...f, branch, recurring: isRecurring(branch, f), disposition }
    if (disposition === 'defer') deferredP2s.push(entry)
    return entry
  })

  // Dedupe the recurring set by finding id: a verifier-downgraded P2 lives in
  // both `verified` and the folded `p2s`, so a naive concat would count it twice.
  const recurring = [
    ...new Map([...verified, ...needsVerification, ...p2s].filter((f) => f.recurring).map((f) => [f.id, f])).values(),
  ]

  return {
    branch,
    crates: meta.crates,
    claudeAssessment: r.claudeAssessment,
    codexAssessment: r.codexAssessment,
    tierCounts: r.tierCounts,
    verifiedP0,
    verifiedP1,
    needsVerification,
    refuted,
    p2s,
    circular,
    recurring,
  }
})

const counts = {
  branches: perBranch.length,
  verifiedP0: perBranch.reduce((n, b) => n + b.verifiedP0.length, 0),
  verifiedP1: perBranch.reduce((n, b) => n + b.verifiedP1.length, 0),
  needsVerification: perBranch.reduce((n, b) => n + b.needsVerification.length, 0),
  refuted: perBranch.reduce((n, b) => n + b.refuted.length, 0),
  p2: perBranch.reduce((n, b) => n + b.p2s.length, 0),
  recurring: perBranch.reduce((n, b) => n + b.recurring.length, 0),
}

log(
  `round ${ROUND} consolidated: ${counts.verifiedP0} VERIFIED P0, ${counts.verifiedP1} VERIFIED P1, ` +
    `${counts.needsVerification} NEEDS-VERIFICATION, ${counts.refuted} refuted, ${counts.p2} P2, ${counts.recurring} RECURRING.`,
)

return {
  meta: {
    round: ROUND,
    base: BASE,
    repoRoot: discover.repoRoot,
    headSha: discover.headSha,
    codexUsable,
    sammyRoot: sammyRoot || '(not found)',
    isWorktree: discover.isWorktree,
  },
  mergeOrder,
  overlaps,
  perBranch,
  deferredP2s, // the skill files these as issues at the end (no longer a skippable plea)
  counts,
}
