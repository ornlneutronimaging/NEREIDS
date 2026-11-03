export const meta = {
  name: 'pr-review',
  description:
    'One review round for a NEREIDS pull request: the CodeRabbit review and a Claude and a Codex review in one fix table, cross-family checks, fix-depth assessment, a fix plan, and a Codex risk review of the plan. Changes no files.',
  whenToUse:
    'Run from the pr-review skill after CodeRabbit has reviewed the PR head. args: {pr, repoRoot, scratch, repo?, sammyRoot?}.',
  phases: [
    { title: 'Collect', detail: 'PR facts, the CodeRabbit review, a Claude and a Codex review' },
    { title: 'Verify', detail: 'another model family checks each single-source P0 and P1 finding' },
    { title: 'Assess', detail: 'smallest fix against root-cause fix, and recursion risk' },
    { title: 'Plan', detail: 'fix plan and Codex risk review' },
  ],
}

const A = args && typeof args === 'object' ? args : {}
const PR = Number(A.pr)
const ROOT = typeof A.repoRoot === 'string' ? A.repoRoot : ''
const REPO = A.repo || 'ornlneutronimaging/NEREIDS'
const SCRATCH = typeof A.scratch === 'string' ? A.scratch : ''
const SAMMY = A.sammyRoot || ''
if (!Number.isInteger(PR) || PR < 1) throw new Error('pr-review: args.pr must be a pull request number')
if (!ROOT) throw new Error('pr-review: args.repoRoot must be a clean checkout of the PR head')
if (!SCRATCH) throw new Error('pr-review: args.scratch must be a new directory outside args.repoRoot')
if (!/^[\w.-]+\/[\w.-]+$/.test(REPO)) throw new Error('pr-review: args.repo must be owner/name')

const q = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`

const READ_ONLY = `

Rules: change no file in the repository. Run no command that changes the repository, a remote, GitHub or EWM: no git commit, push, checkout, reset, stash or fetch, and no gh command that posts or edits. Write only under ${SCRATCH}. Return the result through the structured output tool.`

const CODEX_PREFIX =
  'You are a non-interactive, read-only review process. No person will answer questions. Do not stop for approval; do the task in one run. Change no files.\n\n'

const TIERS = ['P0', 'P1', 'P2']
const rank = (t) => (TIERS.includes(t) ? TIERS.indexOf(t) : TIERS.length)
const CR_TIER = { critical: 'P0', major: 'P1', error: 'P1', unknown: 'P1', minor: 'P2', trivial: 'P2', nitpick: 'P2', warning: 'P2' }
const ID_PREFIX = { coderabbit: 'CR', claude: 'CL', codex: 'CX' }
const FAMILIES = ['claude', 'codex']
const FIX_CHOICES = ['root-fix', 'smallest-fix']
const clip = (s, n) => {
  const t = String(s || '')
  return t.length > n ? `${t.slice(0, n)} [...]` : t
}
const chunks = (xs, n) => {
  const out = []
  for (let i = 0; i < xs.length; i += n) out.push(xs.slice(i, i + n))
  return out
}
const sameSha = (a, b) => !!a && !!b && (a.startsWith(b) || b.startsWith(a))

// Codex receives these schemas through --output-schema; its strict mode needs
// every property listed in `required` and no additional properties.
const str = { type: 'string' }
const int = { type: 'integer' }
const obj = (props) => ({ type: 'object', additionalProperties: false, required: Object.keys(props), properties: props })
const arr = (items) => ({ type: 'array', items })
const oneOf = (...values) => ({ type: 'string', enum: values })

const REVIEW = obj({
  assessment: str,
  findings: arr(
    obj({
      tier: oneOf(...TIERS),
      title: str,
      file: str,
      line: int,
      claim: str,
      evidence: str,
      reasoning: str,
      primarySource: str,
      suggestedFix: str,
      confidence: oneOf('high', 'medium', 'low'),
    }),
  ),
})
const VERDICTS = obj({
  verdicts: arr(obj({ row: str, verdict: oneOf('CONFIRMED', 'REFUTED', 'UNCERTAIN'), tier: oneOf(...TIERS, 'NONE'), reasoning: str })),
})
const RISK = obj({
  approve: { type: 'boolean' },
  summary: str,
  risks: arr(obj({ item: str, severity: oneOf('high', 'medium', 'low'), risk: str, mitigation: str })),
})
const FACTS = obj({
  headSha: str,
  prHeadSha: str,
  clean: { type: 'boolean' },
  state: str,
  baseRef: str,
  baseSha: str,
  baseAvailable: { type: 'boolean' },
  mergeBase: str,
  changedFiles: arr(str),
  shortstat: str,
  codex: str,
})
const CR_KINDS = ['inline', 'nitpick', 'outside-diff', 'duplicate', 'pre-merge']
const CODERABBIT = obj({
  status: oneOf('reviewed', 'in-progress', 'skipped', 'rate-limited', 'not-reviewed'),
  reviewedSha: str,
  note: str,
  counts: obj({ inline: int, nitpick: int, outsideDiff: int, duplicate: int, preMerge: int }),
  findings: arr(
    obj({
      kind: oneOf(...CR_KINDS),
      severity: oneOf('critical', 'major', 'minor', 'trivial', 'nitpick', 'warning', 'error', 'unknown'),
      title: str,
      file: str,
      line: int,
      claim: str,
      suggestedFix: str,
      url: str,
    }),
  ),
})
const GROUPS = obj({ groups: arr(obj({ ids: arr(str), reason: str })) })
const ASSESS = obj({
  rows: arr(
    obj({
      row: str,
      valid: oneOf('yes', 'no'),
      validReason: str,
      rootCause: str,
      smallestFix: str,
      rootFix: str,
      otherInstances: arr(str),
      recursionRisk: oneOf('low', 'medium', 'high'),
      recursionReason: str,
      choice: oneOf(...FIX_CHOICES, 'no-change', 'defer', 'decide'),
      choiceReason: str,
      test: str,
    }),
  ),
})
const PLAN = obj({
  items: arr(obj({ id: str, rows: arr(str), change: str, files: arr(str), test: str, commit: str, risk: str })),
  notFixed: arr(obj({ row: str, reason: str })),
})
const harnessSchema = (inner) => ({
  type: 'object',
  additionalProperties: false,
  required: ['ok', 'failure'],
  properties: { ok: { type: 'boolean' }, failure: str, result: inner },
})

function codexHarness(name, prompt, schema) {
  const [p, s, out, log, rc] = ['prompt', 'schema.json', 'json', 'log', 'rc'].map((x) => `${SCRATCH}/codex-${name}.${x}`)
  return `You run the Codex CLI, a reviewer of another model family, and return its answer. Do not do its task yourself, and do not change its answer.

1. Run: mkdir -p ${q(SCRATCH)} && rm -f ${q(rc)} ${q(out)}
2. With the Write tool, write the text between the PROMPT START line and the last PROMPT END line to ${p}:
----- PROMPT START -----
${prompt}
----- PROMPT END -----
3. With the Write tool, write this JSON schema to ${s}:
${JSON.stringify(schema)}
4. Start Codex with the Bash tool, with run_in_background set to true. The command stops Codex and its child processes after 45 minutes, and kills them 60 seconds later if they are still running:
   timeout -k 60 2700 codex exec --sandbox read-only --skip-git-repo-check --ephemeral -C ${q(ROOT)} --output-schema ${q(s)} -o ${q(out)} - < ${q(p)} > ${q(log)} 2>&1; echo $? > ${q(rc)}
5. Wait with this foreground Bash command, with a Bash timeout of 600000 ms. Repeat it until it prints "finished", at most 5 times:
   timeout 580 sh -c 'until [ -s "$1" ]; do sleep 5; done; echo finished' sh ${q(rc)}
6. When ${rc} is still missing after 5 waits, return ok=false with the failure "timeout".
7. When ${rc} does not contain 0, or \`jq -e . ${q(out)}\` fails, return ok=false with a short failure reason and the last 20 lines of ${log}.
8. Otherwise return ok=true, failure "", and result: the JSON object in ${out}, exactly as Codex wrote it.`
}

const factsPrompt = `Gather facts for a review of pull request #${PR} of ${REPO}. The review checkout is ${ROOT}. Run these commands in it:
- \`git rev-parse HEAD\`: headSha (full SHA).
- \`git status --porcelain\`: clean is true when the output is empty.
- \`gh pr view ${PR} --repo ${REPO} --json state,headRefOid,baseRefName,baseRefOid\`: state, prHeadSha (headRefOid), baseRef (baseRefName), baseSha (baseRefOid).
- \`git cat-file -e <baseSha>^{commit}\`: baseAvailable is true when it succeeds. When it fails, return the facts that you have, with mergeBase "", changedFiles [] and shortstat "".
- \`git merge-base HEAD <baseSha>\`: mergeBase (full SHA).
- \`git diff --name-only <mergeBase> HEAD\`: changedFiles. \`git diff --shortstat <mergeBase> HEAD\`: shortstat.
- \`codex --version\`: codex (an empty string when the command fails).`

const CR_DIR = `${SCRATCH}/coderabbit`
const crFile = (name) => q(`${CR_DIR}/${name}`)
const coderabbitPrompt = `Collect the CodeRabbit review of pull request #${PR} of ${REPO}. Use \`gh\` and \`jq\`. Read only: do not post, reply or resolve.

1. Save the raw data:
   mkdir -p ${q(CR_DIR)}
   gh api repos/${REPO}/issues/${PR}/comments --paginate --jq '.[] | select(.user.login == "coderabbitai[bot]")' > ${crFile('issue.jsonl')}
   gh api repos/${REPO}/pulls/${PR}/reviews --paginate --jq '.[] | select(.user.login == "coderabbitai[bot]")' > ${crFile('reviews.jsonl')}
   gh api repos/${REPO}/pulls/${PR}/comments --paginate --jq '.[] | select(.user.login == "coderabbitai[bot]" and .in_reply_to_id == null)' > ${crFile('inline.jsonl')}
   gh pr checks ${PR} --repo ${REPO} --json name,state --jq '.[] | select(.name == "CodeRabbit")' > ${crFile('check.json')}
2. status, reviewedSha and note. The walkthrough is the issue comment that contains "summarize by coderabbit.ai".
   - reviewed: the CodeRabbit check state is SUCCESS, and the walkthrough names the reviewed commits ("between <sha> and <sha>"). reviewedSha is the second SHA.
   - in-progress: the check is pending, or the walkthrough says that a review is in progress.
   - skipped or rate-limited: the walkthrough says so.
   - not-reviewed: none of the above.
   note: the walkthrough's words about the review status and coverage.
3. findings: transcribe every item. Drop none.
   - inline: each object in inline.jsonl. file = .path; line = .line, or .original_line when .line is null; url = .html_url. The first line of the body has labels; severity is critical, major, minor or trivial from them, or unknown. title = the bold title. claim = the explanation, without the "Prompt for AI Agents" block. suggestedFix = the suggested change, or "".
   - nitpick, outside-diff and duplicate: each item in the sections "Nitpick comments (N)", "Outside diff range comments (N)" and "Duplicate comments (N)" of each review body in reviews.jsonl. Section titles can start with an emoji. Each item names a file and a line range; line = the first line of the range. severity = nitpick for nitpick items; for the other two kinds, read it from the item's labels as for inline, or unknown. url = the .html_url of the review.
   - pre-merge: each pre-merge check in the walkthrough with the status Warning or Error. file = "", line = 0, severity = warning or error, title = the check name, claim = the explanation, suggestedFix = the resolution, url = the .html_url of the walkthrough.
   Do not include "Additional comments" items, the walkthrough summary or tips.
4. counts: inline = the output of \`jq -s length ${crFile('inline.jsonl')}\`; nitpick, outsideDiff and duplicate = the sums of the N in those section titles; preMerge = the number of checks with the status Warning or Error.`

function auditPrompt(f) {
  const sammy = SAMMY
    ? `The SAMMY source is at ${SAMMY}.`
    : 'No SAMMY source tree is available; check against the cited equations, and say so in the finding.'
  return `Review pull request #${PR} of ${REPO}. The checkout ${ROOT} is at the PR head ${f.headSha}. The change is \`git diff ${f.mergeBase} HEAD\`. Changed files: ${f.changedFiles.join(', ')}.

Read AGENTS.md and CONTRIBUTING.md first: their rules are review criteria. Read each changed file in full, and the code that the change calls or that calls it.

Report defects in the change, and defects in existing code that the change exposes. Do not report a style preference that no repository rule states. Report each defect once.

Tiers:
- P0: a wrong physical result or number, data loss, a panic or a wrong result on valid input, or a test that cannot fail when the code under test is wrong.
- P1: a missing input check at a public entry point, a claim that no test supports, documentation that differs from the code, a test oracle that depends on the code under test, a tolerance without a reason, or a repository-rule violation that affects users.
- P2: wording, a comment-rule violation, or a small clarity defect.

Check:
1. Physics, units and equations against the cited sources. ${sammy}
2. Numerical edge cases: NaN, infinity, empty input, division by zero, overflow.
3. Input checks and errors at public entry points.
4. Tests: each test must fail when the code under test is wrong, the oracle must be independent of it, and each tolerance needs a reason.
5. Documentation and comments against the code and against the repository's comment rules.
6. Consistency between sibling code paths that must behave the same.

For each finding give the repository-relative file, the line at the PR head (0 if none), the claim, the evidence (quoted code), the reasoning, the primary source (or ""), a suggested fix, and your confidence. In "assessment", give one sentence about the whole change.`
}

function rowText(r) {
  const members = r.members.map((m) => {
    const lines = [`  - ${m.source} ${m.id} (${m.tier}) ${m.file || '(PR)'}:${m.line}: ${clip(m.claim, 1500)}`]
    if (m.evidence) lines.push(`    evidence: ${clip(m.evidence, 800)}`)
    if (m.suggestedFix) lines.push(`    suggested fix: ${clip(m.suggestedFix, 800)}`)
    return lines.join('\n')
  })
  return `${r.id} [${r.tier}] ${r.file || '(PR)'}:${r.line}: ${r.title}\n${members.join('\n')}`
}

function groupPrompt(list) {
  return `CodeRabbit, Claude and Codex reviewed pull request #${PR} of ${REPO}. Group the findings that report the same defect: the same cause at the same place, or one cause that one fix removes. Do not group findings only because they are in the same file or near the same line. Put each id in exactly one group; a finding that matches no other is a group of one. Decide from the text; do not open files.

${list.map((x) => `${x.id} ${x.source} ${x.tier} ${x.file || '(PR)'}:${x.line} ${x.title}: ${clip(x.claim, 400)}`).join('\n')}`
}

function verifyPrompt(f, batch) {
  return `Check findings from a review of pull request #${PR} of ${REPO}. The checkout ${ROOT} is at the PR head ${f.headSha}; the change is \`git diff ${f.mergeBase} HEAD\`.

A reviewer of another model family reported each row. For each row, read the code and decide on your own whether the defect is real. Default to REFUTED when the code does not show the defect. Give your own tier: P0, P1, P2 or NONE. Put the row id in "row".

${batch.map(rowText).join('\n\n')}`
}

function assessPrompt(f, batch) {
  const rows = batch.map((r) => {
    const votes = r.votes.map((v) => `${v.family} ${v.verdict}: ${clip(v.reasoning, 500)}`).join('; ')
    return `${rowText(r)}\n  check: ${r.status}${votes ? `\n  votes: ${votes}` : ''}`
  })
  return `Assess how to fix the findings from the review of pull request #${PR} of ${REPO}. The checkout ${ROOT} is at the PR head ${f.headSha}; the change is \`git diff ${f.mergeBase} HEAD\`. Read AGENTS.md, then the code of each row.

For each row, put the row id in "row" and give:
1. valid: "yes" when the defect is real. A corroborated or confirmed row is "no" only with evidence from the code.
2. validReason: the evidence.
3. rootCause: why the defect exists.
4. smallestFix: the smallest change that makes the finding go away.
5. rootFix: the change that removes the cause. It can be the same as smallestFix.
6. otherInstances: other places in this PR, or in the code that it changes, with the same cause.
7. recursionRisk and recursionReason: the risk that the fix brings new defects that the next review must find. It is high when the fix adds a special case, a guard for one input, a new parameter, a second path that duplicates a first, a looser tolerance, or a comment that explains around the problem. It is low when the fix removes code, or moves a check to the one place that owns the invariant.
8. choice and choiceReason:
   - root-fix: the default for a valid finding.
   - smallest-fix: only when it removes the cause, its recursion risk is low, and no other instance exists.
   - no-change: the finding is valid, but a fix costs more than the defect, or a repository rule makes the current form correct.
   - defer: the fix is outside the purpose of this PR. Name the follow-up task.
   - decide: two repository rules conflict, or the fix needs a scientific choice. State both sides.
   When valid is "no", the choice is no-change.
9. test: the test that fails before the fix and passes after it, or why no test applies and which other check shows the fix.

check: corroborated means two reviewers found the row; confirmed or uncertain is the result of a check by another model family; not-checked means that only you check the row (P2 and PR-level rows).

${rows.join('\n\n')}`
}

function planPrompt(f, planRows) {
  const rows = planRows.map((r) => {
    const a = r.assessment
    return `${rowText(r)}
  root cause: ${a.rootCause}
  assessment (${a.choice}): ${a.choiceReason}
  root fix: ${a.rootFix}
  smallest fix: ${a.smallestFix}
  other instances: ${a.otherInstances.join('; ') || 'none'}
  recursion risk: ${a.recursionRisk}: ${a.recursionReason}
  test: ${a.test}`
  })
  return `Write the fix plan for pull request #${PR} of ${REPO}. The checkout ${ROOT} is at the PR head ${f.headSha}. Read AGENTS.md and the code that the rows name.

Rules:
- One item for each root cause. One item can fix several rows. Put each row below in exactly one item, or in notFixed with the reason.
- Each item is one atomic commit. Give the change, the files, the test that fails before the change (or the other check that shows the fix, when no test applies), the commit subject (imperative, at most 72 characters), and the risk of the change.
- Prefer a change that removes code. Do not add a special case, a guard for one input, a duplicate path or a looser tolerance.
- Add no work that no row asks for.
- Order the items so that each commit builds and passes the tests by itself. Use the ids C1, C2, and so on.

${rows.join('\n\n')}`
}

function decision(r, plan) {
  const items = plan.items.filter((it) => it.rows.includes(r.id)).map((it) => it.id)
  if (items.length) return `fixed by ${items.join(', ')}`
  const nf = plan.notFixed.find((x) => x.row === r.id)
  if (nf) return `left in notFixed: ${clip(nf.reason, 500)}`
  if (r.status === 'refuted') return `refuted: ${r.votes.map((v) => `${v.family}: ${clip(v.reasoning, 400)}`).join('; ')}`
  if (!r.assessment) return 'not assessed'
  return `${r.assessment.valid === 'yes' ? 'valid' : 'not valid'}, ${r.assessment.choice}: ${clip(r.assessment.choiceReason, 500)}`
}

function riskPrompt(f, plan, rows) {
  return `Review this fix plan for pull request #${PR} of ${REPO} before anyone applies it. The checkout ${ROOT} is at the PR head ${f.headSha}. Read the code that each item changes, and the code of each row that the plan does not fix.

For each plan item, find the risks: a regression; a fix that adds a new defect, or new code that the next review must check (a special case, a guard for one input, a duplicate path, a looser tolerance); a missing or circular test; a change outside the rows that the item fixes; a commit that does not build or pass by itself; or a conflict with AGENTS.md.
For each row that the plan does not fix, find the risk of that decision: a real defect that affects results or users, left without a sound reason.
Give a mitigation for each risk. In "item", give the plan item id or the row id. Set approve to false when any risk is high.

Rows and decisions:
${rows.map((r) => `${r.id} [${r.tier}] ${r.file || '(PR)'}:${r.line}: ${r.title}: ${decision(r, plan)}`).join('\n')}

Plan items:
${JSON.stringify(plan.items, null, 2)}`
}

function revisePrompt(f, plan, review, planRows) {
  return `${planPrompt(f, planRows)}

A Codex risk review of an earlier plan found the risks below. Revise the plan: remove each high risk, and apply each mitigation that is correct. When you keep an item that has a risk, or keep a row in notFixed, say why.

Earlier plan:
${JSON.stringify(plan, null, 2)}

Risks:
${JSON.stringify(review.risks, null, 2)}`
}

// gaps make the round incomplete; warnings do not.
const gaps = []
const warnings = []

phase('Collect')
const [facts, cr] = await parallel([
  () => agent(factsPrompt + READ_ONLY, { label: 'facts', phase: 'Collect', schema: FACTS }),
  () => agent(coderabbitPrompt + READ_ONLY, { label: 'coderabbit', phase: 'Collect', schema: CODERABBIT }),
])
if (!facts) throw new Error('pr-review: the facts step returned no result')
if (facts.state !== 'OPEN') throw new Error(`pr-review: PR #${PR} is ${facts.state}`)
if (!facts.clean || facts.headSha !== facts.prHeadSha) {
  throw new Error(
    `pr-review: ${ROOT} must be a clean checkout of the PR head ${facts.prHeadSha}; it is at ${facts.headSha}${facts.clean ? '' : ' with local changes'}`,
  )
}
if (!facts.baseAvailable) throw new Error(`pr-review: the PR base ${facts.baseSha} is not in ${ROOT}; fetch ${facts.baseRef} first`)
if (!cr) throw new Error('pr-review: the CodeRabbit step returned no result')
if (cr.status !== 'reviewed' || !sameSha(cr.reviewedSha, facts.headSha)) {
  throw new Error(
    `pr-review: CodeRabbit has not reviewed ${facts.headSha.slice(0, 8)}: ${cr.status}${cr.reviewedSha ? `, last reviewed ${cr.reviewedSha.slice(0, 8)}` : ''}. ${cr.note}`,
  )
}
const crCount = { inline: cr.counts.inline, nitpick: cr.counts.nitpick, 'outside-diff': cr.counts.outsideDiff, duplicate: cr.counts.duplicate, 'pre-merge': cr.counts.preMerge }
for (const kind of CR_KINDS) {
  const n = cr.findings.filter((x) => x.kind === kind).length
  if (n !== crCount[kind]) gaps.push(`CodeRabbit ${kind}: the review has ${crCount[kind]}, the collection has ${n}`)
}

const codexUsable = !!facts.codex
if (!codexUsable) gaps.push('Codex is not available')
log(`PR #${PR} at ${facts.headSha.slice(0, 8)}: ${facts.changedFiles.length} files; CodeRabbit ${cr.status}, ${cr.findings.length} items`)

const audit = auditPrompt(facts)
const [claudeRes, codexRes] = await parallel([
  () => agent(audit + READ_ONLY, { label: 'find:claude', phase: 'Collect', schema: REVIEW }),
  () =>
    codexUsable
      ? agent(codexHarness('find', CODEX_PREFIX + audit, REVIEW) + READ_ONLY, {
          label: 'find:codex',
          phase: 'Collect',
          schema: harnessSchema(REVIEW),
        })
      : Promise.resolve(null),
])
const codexReview = codexRes && codexRes.ok && codexRes.result ? codexRes.result : null
if (!claudeRes) gaps.push('The Claude review returned no result')
if (codexUsable && !codexReview) gaps.push(`The Codex review failed: ${(codexRes && codexRes.failure) || 'no result'}`)

const findings = []
const add = (source, f, extra) => {
  const n = findings.filter((x) => x.source === source).length + 1
  findings.push({
    kind: 'review',
    evidence: '',
    reasoning: '',
    primarySource: '',
    suggestedFix: '',
    confidence: '',
    url: '',
    ...f,
    ...extra,
    id: `${ID_PREFIX[source]}${n}`,
    source,
    file: f.file || '',
    line: Number.isInteger(f.line) ? f.line : 0,
  })
}
for (const f of cr.findings) add('coderabbit', f, { tier: CR_TIER[f.severity] || 'P1' })
for (const f of (claudeRes && claudeRes.findings) || []) add('claude', f, {})
for (const f of (codexReview && codexReview.findings) || []) add('codex', f, {})
const byId = new Map(findings.map((x) => [x.id, x]))
const count = (src) => findings.filter((x) => x.source === src).length
log(`findings: ${findings.length} (CodeRabbit ${count('coderabbit')}, Claude ${count('claude')}, Codex ${count('codex')})`)

const EMPTY_PLAN = { items: [], notFixed: [] }
const result = (status, rows, plan, planReviews) => ({
  pr: PR,
  headSha: facts.headSha,
  mergeBase: facts.mergeBase,
  status: gaps.length ? 'incomplete' : status,
  gaps,
  warnings,
  coderabbit: { status: cr.status, reviewedSha: cr.reviewedSha, note: cr.note, counts: cr.counts },
  assessments: { claude: claudeRes ? claudeRes.assessment : '', codex: codexReview ? codexReview.assessment : '' },
  table: rows.map((r) => ({
    row: r.id,
    tier: r.tier,
    sources: r.sources,
    file: r.file,
    line: r.line,
    title: r.title,
    check: r.status,
    votes: r.votes,
    missingVotes: r.missing || [],
    assessment: r.assessment || null,
    planItems: plan.items.filter((it) => it.rows.includes(r.id)).map((it) => it.id),
    findings: r.members,
  })),
  plan,
  planReviews,
})
if (!findings.length) return result('nothing-to-fix', [], EMPTY_PLAN, [])

phase('Verify')
const groups = []
if (findings.length === 1) groups.push([findings[0].id])
else {
  const g = await agent(groupPrompt(findings) + READ_ONLY, { label: 'group', phase: 'Verify', schema: GROUPS })
  if (!g) warnings.push('The grouping step returned no result: each finding is its own row')
  const seen = new Set()
  for (const grp of (g && g.groups) || []) {
    const ids = grp.ids.map((id) => id.trim()).filter((id) => byId.has(id) && !seen.has(id))
    ids.forEach((id) => seen.add(id))
    if (ids.length) groups.push(ids)
  }
  for (const x of findings) if (!seen.has(x.id)) groups.push([x.id])
}

const rows = groups
  .map((ids) => {
    const members = ids.map((id) => byId.get(id))
    const lead = members.slice().sort((a, b) => rank(a.tier) - rank(b.tier))[0]
    const sources = [...new Set(members.map((m) => m.source))]
    return { id: '', members, sources, tier: lead.tier, file: lead.file, line: lead.line, title: lead.title, votes: [] }
  })
  .sort((a, b) => rank(a.tier) - rank(b.tier) || a.file.localeCompare(b.file) || a.line - b.line)
rows.forEach((r, i) => {
  r.id = `R${i + 1}`
  if (r.sources.length > 1) r.status = 'corroborated'
  else if (rank(r.tier) > 1 || r.members.every((m) => m.kind === 'pre-merge')) r.status = 'not-checked'
  else {
    r.status = 'pending'
    r.need = FAMILIES.filter((fam) => !r.sources.includes(fam))
  }
})
const rowById = new Map(rows.map((r) => [r.id, r]))

const forFamily = (fam) => rows.filter((r) => r.need && r.need.includes(fam))
const verifyJobs = chunks(forFamily('claude'), 6).map((batch, i) => () =>
  agent(verifyPrompt(facts, batch) + READ_ONLY, { label: `verify:claude:${i + 1}`, phase: 'Verify', schema: VERDICTS }).then((v) => ({ family: 'claude', v })),
)
if (codexUsable) {
  chunks(forFamily('codex'), 6).forEach((batch, i) =>
    verifyJobs.push(() =>
      agent(codexHarness(`verify-${i + 1}`, CODEX_PREFIX + verifyPrompt(facts, batch), VERDICTS) + READ_ONLY, {
        label: `verify:codex:${i + 1}`,
        phase: 'Verify',
        schema: harnessSchema(VERDICTS),
      }).then((v) => ({ family: 'codex', v: v && v.ok && v.result ? v.result : null })),
    ),
  )
}
for (const out of await parallel(verifyJobs)) {
  if (!out || !out.v) continue
  for (const v of out.v.verdicts) {
    const r = rowById.get(v.row.trim())
    if (r && r.need && r.need.includes(out.family) && !r.votes.some((x) => x.family === out.family)) {
      r.votes.push({ family: out.family, verdict: v.verdict, tier: v.tier, reasoning: v.reasoning })
    }
  }
}
for (const r of rows.filter((x) => x.status === 'pending')) {
  const yes = r.votes.filter((v) => v.verdict === 'CONFIRMED').length
  const no = r.votes.filter((v) => v.verdict === 'REFUTED').length
  r.status = yes > no ? 'confirmed' : no > yes ? 'refuted' : 'uncertain'
  r.missing = r.need.filter((fam) => !r.votes.some((v) => v.family === fam))
  if (!r.votes.length) gaps.push(`${r.id}: no check by another model family`)
  else if (r.missing.length) warnings.push(`${r.id}: no ${r.missing.join(' or ')} check`)
}
log(`rows: ${rows.length}; ${rows.filter((r) => r.status === 'refuted').length} refuted`)

phase('Assess')
const toAssess = rows.filter((r) => r.status !== 'refuted')
const assessed = await parallel(
  chunks(toAssess, 8).map((batch, i) => () =>
    agent(assessPrompt(facts, batch) + READ_ONLY, { label: `assess:${i + 1}`, phase: 'Assess', schema: ASSESS }),
  ),
)
for (const out of assessed) {
  for (const a of (out && out.rows) || []) {
    const r = rowById.get(a.row.trim())
    if (r && r.status !== 'refuted' && !r.assessment) r.assessment = a
  }
}
const unassessed = toAssess.filter((r) => !r.assessment)
if (unassessed.length) gaps.push(`No assessment for ${unassessed.map((r) => r.id).join(', ')}`)

phase('Plan')
const isFix = (r) => r.assessment && r.assessment.valid === 'yes' && FIX_CHOICES.includes(r.assessment.choice)
let planRows = toAssess.filter(isFix)
const normalize = (p) => ({
  items: p.items.map((it) => ({ ...it, id: it.id.trim(), rows: it.rows.map((id) => id.trim()) })),
  notFixed: p.notFixed.map((x) => ({ ...x, row: x.row.trim() })),
})
const makePlan = (label, prompt) =>
  agent(prompt + READ_ONLY, { label, phase: 'Plan', schema: PLAN }).then((p) => (p ? normalize(p) : null))

const coverageGaps = (p) => {
  const placed = [...p.items.flatMap((it) => it.rows), ...p.notFixed.map((x) => x.row)]
  const want = new Set(planRows.map((r) => r.id))
  const missing = [...want].filter((id) => !placed.includes(id))
  const twice = [...new Set(placed.filter((id, i) => placed.indexOf(id) !== i))]
  const unknown = [...new Set(placed.filter((id) => !want.has(id)))]
  const out = []
  if (missing.length) out.push(`The plan does not place ${missing.join(', ')}`)
  if (twice.length) out.push(`The plan places ${twice.join(', ')} more than once`)
  if (unknown.length) out.push(`The plan names rows that it was not given: ${unknown.join(', ')}`)
  return out
}

let plan = EMPTY_PLAN
if (planRows.length) {
  plan = await makePlan('plan', planPrompt(facts, planRows))
  if (!plan) {
    gaps.push('The plan step returned no result')
    return result('incomplete', rows, EMPTY_PLAN, [])
  }
}
// The risk review also runs on an empty plan: it checks each decision to leave a row unfixed.
const planReviews = []
let accepted = false
let planGaps = coverageGaps(plan)
for (let round = 1; codexUsable && !planGaps.length && round <= 2; round++) {
  const rv = await agent(codexHarness(`risk-${round}`, CODEX_PREFIX + riskPrompt(facts, plan, rows), RISK) + READ_ONLY, {
    label: `risk:codex:${round}`,
    phase: 'Plan',
    schema: harnessSchema(RISK),
  })
  const review = rv && rv.ok && rv.result ? rv.result : null
  if (!review) {
    gaps.push(`The Codex risk review failed: ${(rv && rv.failure) || 'no result'}`)
    break
  }
  planReviews.push(review)
  if (review.approve && !review.risks.some((x) => x.severity === 'high')) {
    accepted = true
    break
  }
  if (round === 2) break
  const flagged = new Set(review.risks.map((x) => x.item.trim()))
  const reopened = toAssess.filter((r) => r.assessment && r.assessment.choice !== 'decide' && flagged.has(r.id) && !planRows.includes(r))
  planRows = [...planRows, ...reopened]
  if (!planRows.length) break
  const revised = await makePlan('plan:revise', revisePrompt(facts, plan, review, planRows))
  if (!revised) {
    gaps.push('The plan revision returned no result')
    break
  }
  plan = revised
  planGaps = coverageGaps(plan)
}
gaps.push(...planGaps)
const decide = toAssess.some((r) => r.assessment && r.assessment.choice === 'decide')
const status = !accepted ? 'plan-rejected' : plan.items.length ? 'approved' : decide ? 'needs-decision' : 'nothing-to-fix'
log(`plan: ${plan.items.length} commits, ${gaps.length ? 'incomplete' : status}`)
return result(status, rows, plan, planReviews)
