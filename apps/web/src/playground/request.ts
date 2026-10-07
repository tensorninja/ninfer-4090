// The System One request as the playground edits it.
//
// A request is `{state, questions[, model]}` (kev's SystemOneRequest, which ninfer serves under
// /typesafe/v1/systemone): the state is any JSON value, and each question is a `noul`, `choice`
// or `score` object with optional `instructions` and type-specific `criteria`. Each section has
// two editors — a form for the common shape and JSON for everything — and the text of either is
// the source of truth while it is showing. Conversion happens only when the editor changes, and
// JSON the form cannot show exactly is refused rather than approximated.
//
// Adapted from laya's playground (examples/server.py; Apache-2.0) to System One's schema.

import {
  compactJson,
  dedupe,
  formatJson,
  fromValue,
  kindOf,
  lastEntry,
  nullNode,
  objectNode,
  stringNode,
  toValue,
  tryParse,
  type JsonEntry,
  type JsonNode,
  type ObjectNode,
} from './json'

export type Protocol = 'typesafe' | 'openai'
export type QuestionType = 'noul' | 'choice' | 'score'
export const QUESTION_TYPES: readonly QuestionType[] = ['noul', 'choice', 'score']
/** kev's MAX_OPTIONS: a choice or score question takes 1 to 255 options. */
export const MAX_OPTIONS = 255

export const isQuestionType = (value: unknown): value is QuestionType =>
  typeof value === 'string' && (QUESTION_TYPES as readonly string[]).includes(value)

export type StateMode = 'fields' | 'json'
export type QuestionsMode = 'form' | 'json'
export type Section = 'state' | 'questions'

let sequence = 0
/** Identity for a form row, so focus and React keys follow it across edits. */
export const nextId = () => ++sequence

export interface Field {
  id: number
  name: string
  value: string
}

/** A choice option, or one of noul's true/false descriptions. */
export interface Option {
  id: number
  label: string
  desc: string
  /** An empty description is sent as null unless it was loaded as "". */
  wasNull: boolean
}

export interface Level {
  id: number
  text: string
}

export interface QuestionDraft {
  id: number
  key: string
  type: QuestionType
  instructions: string
  /** How the instructions arrived, so an untouched empty value is sent back unchanged. */
  insOrigin: 'absent' | 'null' | 'text'
  options: Option[]
  levels: Level[]
  /** Noul's descriptions of true and false; null when it has none. */
  tf: Option[] | null
  /** Keys System One ignores, kept as written and sent back. */
  extra: JsonEntry[]
}

export interface EditorState {
  requestText?: string
  protocol: Protocol
  stateMode: StateMode
  qMode: QuestionsMode
  fields: Field[]
  stateText: string
  questions: QuestionDraft[]
  qText: string
  /** The `model` to send; empty omits it, which the server answers as the SDK alias. */
  model: string
}

export const quote = (s: string) => `\u201c${s}\u201d`
export const orList = (items: readonly string[]) =>
  items.length > 1
    ? items.slice(0, -1).join(', ') + ' or ' + items[items.length - 1]
    : items.join('')
export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

// --- state: fields <-> JSON ------------------------------------------------------------------

/** Why the fields editor cannot show this state exactly, or '' when it can. */
export function fieldsBlocker(ast: JsonNode): string {
  if (ast.t === 'str') return 'State is plain text, so it is edited as JSON (a quoted string).'
  if (ast.t === 'arr') return 'State is a list, so it is edited as JSON.'
  if (ast.t !== 'obj') return `State is ${kindOf(ast)}, so it is edited as JSON.`
  const bad = ast.entries.find((en) => en.v.t !== 'str')
  return bad ? `Field ${quote(bad.k)} holds ${kindOf(bad.v)}, which only JSON can edit.` : ''
}

export function fieldsFromAst(ast: ObjectNode): Field[] {
  return ast.entries.map((en) => ({ id: nextId(), name: en.k, value: (en.v as { v: string }).v }))
}

/** Rows with neither a name nor a value are not part of the request. */
export const filledFields = (fields: readonly Field[]) =>
  fields.filter((f) => f.name !== '' || f.value !== '')

export function fieldsNode(fields: readonly Field[]): ObjectNode {
  return objectNode(filledFields(fields).map((f) => [f.name, stringNode(f.value)]))
}

// --- questions: form <-> JSON ----------------------------------------------------------------

const textOrNull = (node: JsonNode) => node.t === 'str' || (node.t === 'lit' && node.v === null)

/** Why the form cannot show these questions exactly, or '' when it can. */
export function formBlocker(ast: JsonNode): string {
  if (ast.t !== 'obj') return 'Questions must be a JSON object to use the form.'
  for (const en of ast.entries) {
    const name = quote(en.k)
    const v = en.v
    if (v.t !== 'obj') return `${name} is ${kindOf(v)}, not a question object.`
    const type = lastEntry(v, 'type')
    const ins = lastEntry(v, 'instructions')
    const crit = lastEntry(v, 'criteria')
    if (!type || type.v.t !== 'str' || !isQuestionType(type.v.v)) {
      return `${name} needs a type (${orList(QUESTION_TYPES)}) before the form can show it.`
    }
    if (ins && !textOrNull(ins.v)) return `The instructions of ${name} are not text.`
    if (!crit || (crit.v.t === 'lit' && crit.v.v === null)) continue
    const c = crit.v
    if (type.v.v === 'choice') {
      if (c.t === 'arr')
        return `The options of ${name} are a list; System One takes label → description.`
      if (!(c.t === 'obj' && c.entries.every((o) => textOrNull(o.v)))) {
        return `The options of ${name} are not all label: text pairs.`
      }
    }
    if (type.v.v === 'score' && !(c.t === 'arr' && c.items.every((x) => x.t === 'str'))) {
      return `The levels of ${name} are not all text.`
    }
    if (type.v.v === 'noul') {
      const keys = c.t === 'obj' ? c.entries.map((o) => o.k) : []
      const ok =
        c.t === 'obj' &&
        keys.length > 0 &&
        keys.every((k) => k === 'true' || k === 'false') &&
        new Set(keys).size === keys.length &&
        c.entries.every((o) => textOrNull(o.v))
      if (!ok)
        return `The criteria of ${name} are not true/false descriptions, which only JSON can edit.`
    }
  }
  return ''
}

const option = (label: string, value: JsonNode): Option => ({
  id: nextId(),
  label,
  desc: value.t === 'str' ? value.v : '',
  wasNull: value.t !== 'str',
})

/** A question the form can show: formBlocker accepted its entry. */
export function questionFromEntry(entry: JsonEntry): QuestionDraft {
  const v = entry.v as ObjectNode
  const type = (lastEntry(v, 'type')!.v as { v: QuestionType }).v
  const ins = lastEntry(v, 'instructions')
  const crit = lastEntry(v, 'criteria')?.v
  const q: QuestionDraft = {
    id: nextId(),
    key: entry.k,
    type,
    instructions: ins && ins.v.t === 'str' ? ins.v.v : '',
    insOrigin: !ins ? 'absent' : ins.v.t === 'str' ? 'text' : 'null',
    options: [],
    levels: [],
    tf: null,
    extra: [],
  }
  if (crit?.t === 'obj' && type === 'choice') q.options = crit.entries.map((o) => option(o.k, o.v))
  if (crit?.t === 'obj' && type === 'noul') q.tf = crit.entries.map((o) => option(o.k, o.v))
  if (crit?.t === 'arr' && type === 'score') {
    q.levels = crit.items.map((x) => ({ id: nextId(), text: (x as { v: string }).v }))
  }
  const seen = new Set(['type', 'instructions', 'criteria'])
  for (const en of v.entries) {
    if (seen.has(en.k)) continue
    seen.add(en.k)
    q.extra.push(lastEntry(v, en.k)!)
  }
  return q
}

export function newQuestion(key: string): QuestionDraft {
  return {
    id: nextId(),
    key,
    type: 'noul',
    instructions: '',
    insOrigin: 'text',
    options: [],
    levels: [],
    tf: null,
    extra: [],
  }
}

export const newOption = (label = ''): Option => ({ id: nextId(), label, desc: '', wasNull: true })
export const newLevel = (text = ''): Level => ({ id: nextId(), text })

/** A copy under `key` whose rows have identities of their own. */
export function copyQuestion(q: QuestionDraft, key: string): QuestionDraft {
  return {
    ...q,
    id: nextId(),
    key,
    options: q.options.map((o) => ({ ...o, id: nextId() })),
    levels: q.levels.map((l) => ({ ...l, id: nextId() })),
    tf: q.tf && q.tf.map((o) => ({ ...o, id: nextId() })),
  }
}

/** Rows the user left completely empty are not sent. */
export const filledOptions = (options: readonly Option[]) =>
  options.filter((o) => o.label !== '' || o.desc !== '')
export const filledLevels = (levels: readonly Level[]) => levels.filter((l) => l.text !== '')

const optionValue = (o: Option): JsonNode =>
  o.desc === '' && o.wasNull ? nullNode() : stringNode(o.desc)

export function questionNode(q: QuestionDraft): ObjectNode {
  const entries: Array<[string, JsonNode]> = [['type', stringNode(q.type)]]
  if (q.instructions !== '' || q.insOrigin === 'text') {
    entries.push(['instructions', stringNode(q.instructions)])
  } else if (q.insOrigin === 'null') {
    entries.push(['instructions', nullNode()])
  }
  if (q.type === 'choice') {
    const options = filledOptions(q.options)
    if (options.length)
      entries.push(['criteria', objectNode(options.map((o) => [o.label, optionValue(o)]))])
  } else if (q.type === 'score') {
    const levels = filledLevels(q.levels)
    if (levels.length)
      entries.push([
        'criteria',
        { t: 'arr', items: levels.map((l) => stringNode(l.text)), s: 0, e: 0 },
      ])
  } else if (q.tf) {
    entries.push(['criteria', objectNode(q.tf.map((o) => [o.label, optionValue(o)]))])
  }
  const node = objectNode(entries)
  for (const extra of q.extra) {
    if (!entries.some(([k]) => k === extra.k)) node.entries.push({ ...extra, ks: 0, ke: 0 })
  }
  return node
}

export function questionsNode(questions: readonly QuestionDraft[]): ObjectNode {
  return objectNode(questions.map((q) => [q.key, questionNode(q)]))
}

export function uniqueKey(questions: readonly QuestionDraft[], base: string): string {
  const keys = new Set(questions.map((q) => q.key))
  let key = base
  for (let n = 2; keys.has(key); n++) key = `${base}_${n}`
  return key
}

/** A type switch carries choice labels into score levels and back when the target is empty. */
export function withType(q: QuestionDraft, type: QuestionType): QuestionDraft {
  if (q.type === type) return q
  const next = { ...q, type }
  if (type === 'score' && !q.levels.some((l) => l.text)) {
    next.levels = q.options.filter((o) => o.label).map((o) => newLevel(o.label))
  }
  if (type === 'choice' && !q.options.some((o) => o.label)) {
    next.options = q.levels.filter((l) => l.text).map((l) => newOption(l.text))
  }
  return next
}

// --- the request body ------------------------------------------------------------------------

export interface RequestBody {
  protocol: Protocol
  state: JsonNode
  questions: JsonNode
  model: string
  /** The exact text sent. */
  text: string
  /**
   * Identity for "edited" and "stale": the request as the server reads it, where key order inside
   * a question and `"criteria": null` do not matter but question and option order do.
   */
  canon: string
  value: { state?: unknown; input?: unknown; questions: unknown; model?: string }
}

export function bodyNode(
  state: JsonNode,
  questions: JsonNode,
  model: string,
  protocol: Protocol = 'typesafe',
): ObjectNode {
  const entries: Array<[string, JsonNode]> = [
    [protocol === 'openai' ? 'input' : 'state', state],
    ['questions', questions],
  ]
  if (model || protocol === 'openai') entries.push(['model', stringNode(model)])
  return objectNode(entries)
}

function canonOf(state: JsonNode, questions: JsonNode, model: string): string {
  let qs = dedupe(questions)
  if (qs.t === 'obj') {
    qs = {
      ...qs,
      entries: qs.entries.map((en) => {
        if (en.v.t !== 'obj') return en
        const kept = en.v.entries
          .filter((x) => !(x.k === 'criteria' && x.v.t === 'lit' && x.v.v === null))
          .sort((a, b) => (a.k < b.k ? -1 : a.k > b.k ? 1 : 0))
        return { ...en, v: { ...en.v, entries: kept } }
      }),
    }
  }
  return [compactJson(dedupe(state)), compactJson(qs), model].join('\u0000')
}

export function buildBody(
  state: JsonNode,
  questions: JsonNode,
  model: string,
  protocol: Protocol = 'typesafe',
  requestText?: string,
): RequestBody {
  let root = bodyNode(state, questions, model, protocol)
  let text = compactJson(root)
  const envelope = requestText === undefined ? null : tryParse(requestText).ast
  if (envelope?.t === 'obj' && requestText !== undefined) {
    const changes: Array<{ s: number; e: number; text: string }> = []
    for (const entry of root.entries) {
      const previous = lastEntry(envelope, entry.k)
      if (previous) {
        if (compactJson(previous.v) !== compactJson(entry.v))
          changes.push({ s: previous.v.s, e: previous.v.e, text: compactJson(entry.v) })
      } else {
        changes.push({
          s: envelope.e - 1,
          e: envelope.e - 1,
          text: `${envelope.entries.length ? ',' : ''}${JSON.stringify(entry.k)}:${compactJson(entry.v)}`,
        })
      }
    }
    if (protocol === 'typesafe' && !model) {
      const previous = lastEntry(envelope, 'model')
      if (previous && compactJson(previous.v) !== '""')
        changes.push({ s: previous.v.s, e: previous.v.e, text: '""' })
    }
    text = requestText
    for (const change of changes.sort((a, b) => b.s - a.s))
      text = text.slice(0, change.s) + change.text + text.slice(change.e)
    root = tryParse(text).ast as ObjectNode
  }
  return {
    protocol,
    state,
    questions,
    model,
    text,
    canon:
      protocol +
      '\u0000' +
      (protocol === 'openai' ? compactJson(dedupe(root)) : canonOf(state, questions, model)),
    value: toValue(root) as RequestBody['value'],
  }
}

// --- snapshots: an editor's text, for drafts, presets, undo and share links ------------------

export interface Snap {
  requestText?: string
  protocol: Protocol
  stateMode: StateMode
  qMode: QuestionsMode
  stateText: string
  qText: string
  model: string
}

export function snapOf(editor: EditorState): Snap {
  return {
    requestText: editor.requestText,
    protocol: editor.protocol,
    stateMode: editor.stateMode,
    qMode: editor.qMode,
    stateText:
      editor.stateMode === 'json' ? editor.stateText : formatJson(fieldsNode(editor.fields)) + '\n',
    qText:
      editor.qMode === 'json' ? editor.qText : formatJson(questionsNode(editor.questions)) + '\n',
    model: editor.model,
  }
}

/**
 * The editors for a snapshot. Its modes are preferences: a section whose text the form cannot
 * show exactly opens as JSON.
 */
export function editorFromSnap(snap: Snap): EditorState {
  const state = tryParse(snap.stateText)
  const qs = tryParse(snap.qText)
  const fields =
    snap.protocol === 'typesafe' &&
    snap.stateMode !== 'json' &&
    state.ast &&
    !fieldsBlocker(state.ast)
      ? fieldsFromAst(state.ast as ObjectNode)
      : null
  const questions =
    snap.protocol === 'typesafe' && snap.qMode !== 'json' && qs.ast && !formBlocker(qs.ast)
      ? (qs.ast as ObjectNode).entries.map(questionFromEntry)
      : null
  return {
    requestText: snap.requestText,
    protocol: snap.protocol,
    stateMode: fields ? 'fields' : 'json',
    qMode: questions ? 'form' : 'json',
    fields: fields ?? [],
    stateText: snap.stateText,
    questions: questions ?? [],
    qText: snap.qText,
    model: snap.model,
  }
}

/** A snapshot of a request given as trees, in the preferred editor modes. */
export function snapFromNodes(
  state: JsonNode | undefined,
  questions: JsonNode | undefined,
  model: string,
  modes: { stateMode: StateMode; qMode: QuestionsMode },
  protocol: Protocol = 'typesafe',
): Snap {
  return {
    protocol,
    ...modes,
    stateText: formatJson(state ?? objectNode([])) + '\n',
    qText: formatJson(questions ?? objectNode([])) + '\n',
    model,
  }
}

export function snapFromValue(
  request: { state: unknown; questions: unknown; model?: unknown },
  modes: { stateMode: StateMode; qMode: QuestionsMode },
  protocol: Protocol = 'typesafe',
): Snap {
  return snapFromNodes(
    fromValue(request.state),
    fromValue(request.questions),
    typeof request.model === 'string' ? request.model : '',
    modes,
    protocol,
  )
}
