export const meta = {
  name: 'comment-audit',
  description:
    'For each stripped Rust file, a tool-less writer regenerates the comments a reader would need, a judge compares every original comment in scope with what was regenerated, and deterministic JS buckets each original as absent, same, keep (provenance) or flagged. Detection only: no file is written.',
  whenToUse:
    'Invoked by the /comment-audit skill before a commit, never by hand. args = { files: [{ path, windows: [{ start, end, lines }], comments }], writerType?, judgeType? } as scripts/comment_audit.py bundle prints it: the stripped source in windows around the comments in scope, and those comments with their anchors. The agent types default to the tool-less comment-writer and comment-judge.',
  phases: [
    { title: 'Regenerate', detail: 'one tool-less writer per file, from the stripped source alone' },
    { title: 'Judge', detail: 'one tool-less judge per batch of (original, regenerated) pairs' },
  ],
}

const WRITER_SCHEMA = {
  type: 'object',
  properties: {
    comments: {
      type: 'array',
      items: {
        type: 'object',
        properties: { line: { type: 'integer' }, text: { type: 'string' } },
        required: ['line', 'text'],
      },
    },
  },
  required: ['comments'],
}

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['same', 'different'] },
          provenance: { type: 'boolean' },
          reason: { type: 'string' },
        },
        required: ['id', 'verdict', 'provenance', 'reason'],
      },
    },
  },
  required: ['verdicts'],
}

const CONTEXT_LINES = 6
const JUDGE_BATCH = 50

const WRITER_RULES = [
  'You have no file access; everything you know about this code is below.',
  'Write a comment only where a reader of the code alone would be missing something: what a non-obvious function returns, why a branch exists, the physical meaning of a quantity. Where the code says it all, write nothing.',
  'At most two sentences per comment, stating what the code does or why it has this shape. Never what it used to do, never a measurement, never an argument.',
  'Report each comment by the line number of the code line it sits directly above (or on, for one that would trail the code), as numbered below, with the comment text without `//` markers.',
].join('\n')

const JUDGE_RULES = [
  'You have no file access; the code around each pair is below.',
  '`same`: the regenerated comment gives a reader the same information as the original, whatever the wording, so the original was recoverable from the code.',
  '`different`: the original states something the regenerated one does not, or contradicts it.',
  'For `different`, `provenance` is true only when the original cites an external source (a SAMMY file, routine or line, an ENDF format rule, a paper or equation number) or names the physical distinction between the two sides of a branch that the code itself cannot express.',
  '`reason` is one sentence naming what the original adds or gets wrong.',
].join('\n')

function writerPrompt(file) {
  const parts = file.windows.map((w) => numbered(w))
  return [
    WRITER_RULES,
    '',
    `File: ${file.path}`,
    `Below are ${file.windows.length} region(s) of the file with every comment removed; line numbers are the file's own.`,
    '',
    parts.join('\n\n[...]\n\n'),
  ].join('\n')
}

function numbered(w) {
  return w.lines.map((l, i) => `${String(w.start + i).padStart(5)}| ${l}`).join('\n')
}

function lineIndex(file) {
  const map = new Map()
  for (const w of file.windows) w.lines.forEach((l, i) => map.set(w.start + i, l))
  return map
}

function context(index, line) {
  const out = []
  for (let k = line - CONTEXT_LINES; k <= line + CONTEXT_LINES; k++) {
    if (!index.has(k)) continue
    out.push(`${k === line ? '>' : ' '}${String(k).padStart(5)}| ${index.get(k)}`)
  }
  return out.join('\n')
}

// A doc comment sits on its item: the anchor line, or the first line
// after the anchor's attributes. A line comment sits on its anchor, one
// line either way.
function acceptedLines(c, index) {
  if (c.kind === 'doc' || c.kind === 'inner_doc') {
    const lines = [c.anchor.line]
    let k = c.anchor.line
    while (index.has(k) && index.get(k).trim().startsWith('#[')) k++
    if (k !== c.anchor.line) lines.push(k)
    return lines
  }
  return [c.anchor.line - 1, c.anchor.line, c.anchor.line + 1]
}

function match(originals, regenerated, index) {
  const used = new Set()
  const pairs = []
  const absent = []
  for (const c of originals) {
    if (!c.anchor) {
      absent.push(c)
      continue
    }
    const accepted = acceptedLines(c, index)
    let best = null
    regenerated.forEach((r, k) => {
      if (used.has(k)) return
      const d = accepted.indexOf(r.line)
      if (d >= 0 && (best === null || d < best.d)) best = { k, d, r }
    })
    if (best === null) {
      absent.push(c)
    } else {
      used.add(best.k)
      pairs.push({ c, r: best.r })
    }
  }
  return { pairs, absent, unmatchedWriter: regenerated.length - used.size }
}

function judgePrompt(file, batch, index) {
  const blocks = batch.map(
    ({ c, r }) =>
      `### ${c.id}\nCode:\n${context(index, c.anchor.line)}\n\nOriginal comment:\n${c.text.join('\n')}\n\nRegenerated comment:\n${r.text}\n`,
  )
  return [JUDGE_RULES, '', `File: ${file.path}`, 'Return one verdict per id.', '', ...blocks].join('\n')
}

function bucket(file, matched, verdicts) {
  const byId = new Map(verdicts.map((v) => [v.id, v]))
  const same = []
  const keep = []
  const flagged = []
  for (const { c, r } of matched.pairs) {
    const v = byId.get(c.id)
    if (!v) {
      flagged.push({ id: c.id, line: c.first_line, original: c.text, regenerated: r.text, reason: 'no verdict returned' })
    } else if (v.verdict === 'same') {
      same.push(c.id)
    } else if (v.provenance) {
      keep.push(c.id)
    } else {
      flagged.push({ id: c.id, line: c.first_line, original: c.text, regenerated: r.text, reason: v.reason })
    }
  }
  return {
    file: file.path,
    scoped: file.comments.length,
    absent: matched.absent.map((c) => c.id),
    same,
    keep,
    flagged,
    unmatchedWriter: matched.unmatchedWriter,
  }
}

const files = (args && args.files) || []
if (!files.length) throw new Error('comment-audit: args.files is empty')
const writerType = (args && args.writerType) || 'comment-writer'
const judgeType = (args && args.judgeType) || 'comment-judge'

const perFile = await pipeline(
  files,
  (file) =>
    agent(writerPrompt(file), {
      label: `write:${file.path.split('/').pop()}`,
      phase: 'Regenerate',
      schema: WRITER_SCHEMA,
      agentType: writerType,
    }).then((w) => ({ file, regenerated: (w && w.comments) || [] })),
  async ({ file, regenerated }) => {
    const index = lineIndex(file)
    const matched = match(file.comments, regenerated, index)
    const batches = []
    for (let k = 0; k < matched.pairs.length; k += JUDGE_BATCH) batches.push(matched.pairs.slice(k, k + JUDGE_BATCH))
    const results = await parallel(
      batches.map((batch, k) => () =>
        agent(judgePrompt(file, batch, index), {
          label: `judge:${file.path.split('/').pop()}#${k + 1}`,
          phase: 'Judge',
          schema: JUDGE_SCHEMA,
          agentType: judgeType,
        }),
      ),
    )
    const verdicts = results.filter(Boolean).flatMap((r) => r.verdicts)
    const row = bucket(file, matched, verdicts)
    log(
      `${row.file}: ${row.scoped} in scope, ${row.absent.length} absent, ${row.same.length} same, ${row.keep.length} keep, ${row.flagged.length} flagged`,
    )
    return row
  },
)

const rows = perFile.filter(Boolean)
const totals = { scoped: 0, absent: 0, same: 0, keep: 0, flagged: 0 }
for (const r of rows) {
  totals.scoped += r.scoped
  totals.absent += r.absent.length
  totals.same += r.same.length
  totals.keep += r.keep.length
  totals.flagged += r.flagged.length
}
return { perFile: rows, totals }
