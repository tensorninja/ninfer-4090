// What is wrong with the request, and where.
//
// Problems come in two severities. An error is something System One's schema rejects (kev's
// SystemOneRequest), so it blocks Run: the server would only answer 422. A warning is something
// the server accepts but that is probably a mistake, such as a key written twice (Python keeps the
// last one) or a question without instructions.
//
// Both editors reduce a question to the same record, so one checker serves the form and the JSON.
// Adapted from laya's playground (examples/server.py: analyzeState, checkQuestions; Apache-2.0).

import {
  compactJson,
  lastEntry,
  lineCol,
  tryParse,
  type JsonEntry,
  type JsonNode,
  type JsonSyntaxError,
} from './json'
import {
  fieldsBlocker,
  fieldsNode,
  filledFields,
  filledLevels,
  filledOptions,
  formBlocker,
  isQuestionType,
  MAX_OPTIONS,
  orList,
  QUESTION_TYPES,
  questionsNode,
  quote,
  buildBody,
  type EditorState,
  type QuestionDraft,
  type RequestBody,
  type Section,
} from './request'
import { analyzeOpenAI } from './openai-validate'

export type Severity = 'error' | 'warning'

/** A span of a JSON editor's text, or a form control named by its `data-loc`. */
export type Loc = { text: Section; at: readonly [number, number] } | { el: string }

/** The `data-loc` names of form controls, so a problem or a server error can focus one. */
export const LOC = {
  model: 'model',
  addField: 'state:add',
  field: (id: number, part: 'name' | 'value') => `field:${id}:${part}`,
  addQuestion: 'questions:add',
  /** `add` is the card's add-option, add-level or describe button. */
  question: (id: number, part: 'key' | 'type' | 'ins' | 'add' | 'up' | 'down') => `q:${id}:${part}`,
  option: (question: number, option: number) => `q:${question}:opt:${option}`,
  level: (question: number, level: number) => `q:${question}:lvl:${level}`,
}

export interface Problem {
  sec: Section | 'model'
  severity: Severity
  msg: string
  /** A readable path such as "questions → urgency → criteria". */
  where: string
  loc: Loc | null
  /** Set when the problem is that the section's JSON does not parse. */
  parse?: JsonSyntaxError
}

export interface SectionResult {
  problems: Problem[]
  /** The section's value as it is sent; null when its JSON does not parse. */
  node: JsonNode | null
  /** In JSON mode, why the form cannot show this value; '' when it can. */
  blocker: string
}

export interface Analysis {
  state: SectionResult
  questions: SectionResult
  /** State problems first, then questions, then the model. */
  problems: Problem[]
  errors: number
  warnings: number
  /** Null while either section's JSON does not parse. */
  body: RequestBody | null
}

const ARROW = ' \u2192 '
const LAST = 'only the last one counts'
const TYPES = orList(QUESTION_TYPES)

const span = (text: Section, node: { s: number; e: number }): Loc => ({
  text,
  at: [node.s, node.e],
})
const keySpan = (text: Section, entry: JsonEntry): Loc => ({ text, at: [entry.ks, entry.ke] })

function parseProblem(sec: Section, text: string, error: JsonSyntaxError): Problem {
  return {
    sec,
    severity: 'error',
    msg: error.msg,
    where: `${sec}, line ${lineCol(text, error.at).line}`,
    loc: { text: sec, at: [error.at, error.at] },
    parse: error,
  }
}

const isEmptyValue = (node: JsonNode) =>
  (node.t === 'str' && !node.v.trim()) ||
  (node.t === 'obj' && !node.entries.length) ||
  (node.t === 'arr' && !node.items.length) ||
  (node.t === 'lit' && node.v === null)

/** Every key written twice in a subtree; `skip` names objects whose repeats are checked elsewhere. */
function repeatedKeys(
  node: JsonNode,
  path: string[],
  report: (entry: JsonEntry, path: string[]) => void,
  skip?: (path: string[]) => boolean,
) {
  if (node.t === 'arr') {
    node.items.forEach((item, j) => repeatedKeys(item, [...path, String(j)], report, skip))
    return
  }
  if (node.t !== 'obj') return
  const seen = new Set<string>()
  for (const entry of node.entries) {
    if (seen.has(entry.k) && !skip?.(path)) report(entry, path)
    seen.add(entry.k)
    repeatedKeys(entry.v, [...path, entry.k], report, skip)
  }
}

// --- state -----------------------------------------------------------------------------------

export function analyzeState(editor: EditorState): SectionResult {
  const problems: Problem[] = []
  const warn = (msg: string, loc: Loc | null, where = 'state') =>
    problems.push({ sec: 'state', severity: 'warning', msg, where, loc })

  if (editor.stateMode === 'fields') {
    const rows = filledFields(editor.fields)
    const first = editor.fields[0]
    if (!rows.length) {
      warn('State is empty, so the model has nothing to read', {
        el: first ? LOC.field(first.id, 'name') : LOC.addField,
      })
    }
    const seen = new Set<string>()
    rows.forEach((field, j) => {
      const loc = { el: LOC.field(field.id, 'name') }
      if (!field.name.trim()) warn(`Field ${j + 1} has no name`, loc)
      else if (seen.has(field.name)) warn(`Field ${quote(field.name)} appears twice; ${LAST}`, loc)
      seen.add(field.name)
    })
    return { problems, node: fieldsNode(editor.fields), blocker: '' }
  }

  const parsed = tryParse(editor.stateText)
  if (parsed.error) {
    return {
      problems: [parseProblem('state', editor.stateText, parsed.error)],
      node: null,
      blocker: '',
    }
  }
  const ast = parsed.ast
  if (isEmptyValue(ast))
    warn('State is empty, so the model has nothing to read', span('state', ast))
  if (ast.t === 'obj') {
    for (const entry of ast.entries) {
      if (!entry.k.trim()) warn('A field has no name', keySpan('state', entry))
    }
  }
  repeatedKeys(ast, [], (entry, path) =>
    warn(
      path.length
        ? `Key ${quote(entry.k)} appears twice in ${['state', ...path].join(ARROW)}; ${LAST}`
        : `Field ${quote(entry.k)} appears twice; ${LAST}`,
      keySpan('state', entry),
      ['state', ...path].join(ARROW),
    ),
  )
  return { problems, node: ast, blocker: fieldsBlocker(ast) }
}

// --- questions -------------------------------------------------------------------------------

interface Label {
  /** Null for a score level that is not text, which is valid and has no blank to check. */
  text: string | null
  loc: Loc
}

interface QuestionRecord {
  key: string
  keyLoc: Loc
  isObj: boolean
  qLoc: Loc
  /** The type as written (JSON text when it is not a string); null when absent. */
  type: string | null
  typeLoc: Loc
  hasInstructions: boolean
  insLoc: Loc
  crit: { kind: 'none' | 'obj' | 'arr' | 'other'; loc: Loc; labels: Label[] }
}

/** The control a card's criteria are reached by: its first row, or its add button. */
function criteriaEl(q: QuestionDraft): string {
  const option =
    q.type === 'choice'
      ? (filledOptions(q.options)[0] ?? q.options[0])
      : q.type === 'noul'
        ? q.tf?.[0]
        : undefined
  if (option) return LOC.option(q.id, option.id)
  const level = q.type === 'score' ? (filledLevels(q.levels)[0] ?? q.levels[0]) : undefined
  return level ? LOC.level(q.id, level.id) : LOC.question(q.id, 'add')
}

function recordsFromForm(questions: readonly QuestionDraft[]): QuestionRecord[] {
  return questions.map((q) => {
    const at = (part: 'key' | 'type' | 'ins' | 'add') => ({ el: LOC.question(q.id, part) })
    const labels: Label[] =
      q.type === 'choice'
        ? filledOptions(q.options).map((o) => ({
            text: o.label,
            loc: { el: LOC.option(q.id, o.id) },
          }))
        : q.type === 'score'
          ? filledLevels(q.levels).map((l) => ({
              text: l.text,
              loc: { el: LOC.level(q.id, l.id) },
            }))
          : []
    const kind =
      q.type === 'noul'
        ? q.tf
          ? 'obj'
          : 'none'
        : !labels.length
          ? 'none'
          : q.type === 'choice'
            ? 'obj'
            : 'arr'
    return {
      key: q.key,
      keyLoc: at('key'),
      isObj: true,
      qLoc: at('key'),
      type: q.type,
      typeLoc: at('type'),
      hasInstructions: q.instructions.trim() !== '',
      insLoc: at('ins'),
      crit: { kind, loc: { el: criteriaEl(q) }, labels },
    }
  })
}

function recordsFromAst(entries: readonly JsonEntry[]): QuestionRecord[] {
  return entries.map((entry) => {
    const v = entry.v
    const keyLoc = keySpan('questions', entry)
    const record: QuestionRecord = {
      key: entry.k,
      keyLoc,
      isObj: v.t === 'obj',
      qLoc: span('questions', v),
      type: null,
      typeLoc: keyLoc,
      hasInstructions: false,
      insLoc: keyLoc,
      crit: { kind: 'none', loc: keyLoc, labels: [] },
    }
    if (v.t !== 'obj') return record
    const type = lastEntry(v, 'type')
    const ins = lastEntry(v, 'instructions')
    const crit = lastEntry(v, 'criteria')?.v
    if (type) {
      record.type = type.v.t === 'str' ? type.v.v : compactJson(type.v)
      record.typeLoc = span('questions', type.v)
    }
    if (ins) {
      record.hasInstructions =
        !(ins.v.t === 'lit' && ins.v.v === null) && !(ins.v.t === 'str' && !ins.v.v.trim())
      record.insLoc = span('questions', ins.v)
    }
    if (crit && !(crit.t === 'lit' && crit.v === null)) {
      const loc = span('questions', crit)
      if (crit.t === 'obj') {
        record.crit = {
          kind: 'obj',
          loc,
          labels: crit.entries.map((o) => ({ text: o.k, loc: keySpan('questions', o) })),
        }
      } else if (crit.t === 'arr') {
        record.crit = {
          kind: 'arr',
          loc,
          labels: crit.items.map((x) => ({
            text: x.t === 'str' ? x.v : null,
            loc: span('questions', x),
          })),
        }
      } else {
        record.crit = { kind: 'other', loc, labels: [] }
      }
    }
    return record
  })
}

function checkQuestions(records: readonly QuestionRecord[], rootLoc: Loc | null): Problem[] {
  const problems: Problem[] = []
  const add = (severity: Severity, msg: string, loc: Loc | null, where = 'questions') =>
    problems.push({ sec: 'questions', severity, msg, where, loc })
  if (!records.length) add('error', 'Add at least one question', rootLoc)

  const keys = new Set<string>()
  for (const r of records) {
    const name = r.key.trim() ? quote(r.key) : 'A question'
    const where = ['questions', r.key || '\u2205'].join(ARROW)
    const typeWhere = where + ARROW + 'type'
    const critWhere = where + ARROW + 'criteria'
    if (!r.key.trim()) add('warning', 'A question has an empty key', r.keyLoc, where)
    else if (keys.has(r.key)) add('warning', `Key ${name} is used twice; ${LAST}`, r.keyLoc, where)
    keys.add(r.key)
    if (!r.isObj) {
      add('error', `${name} must be an object with a type (${TYPES})`, r.qLoc, where)
      continue
    }
    if (r.type === null) add('error', `${name} has no type; use ${TYPES}`, r.typeLoc, typeWhere)
    else if (!isQuestionType(r.type)) {
      add(
        'error',
        `${name} has an unknown type ${quote(r.type)}; use ${TYPES}`,
        r.typeLoc,
        typeWhere,
      )
    }
    if (!r.hasInstructions)
      add('warning', `${name} has no instructions`, r.insLoc, where + ARROW + 'instructions')

    const c = r.crit
    if (r.type === 'choice' || r.type === 'score') {
      const choice = r.type === 'choice'
      const noun = choice ? 'options' : 'levels'
      const want = choice ? 'obj' : 'arr'
      if (c.kind === 'none' || (c.kind === want && !c.labels.length)) {
        add('error', `${name} is a ${r.type} with no ${noun}`, c.loc, critWhere)
      } else if (c.kind !== want) {
        add(
          'error',
          choice
            ? c.kind === 'arr'
              ? `The options of ${name} are a list; System One takes an object of label \u2192 description`
              : `The options of ${name} must be an object of label \u2192 description`
            : `The levels of ${name} must be a list, lowest first`,
          c.loc,
          critWhere,
        )
      } else {
        const labels = new Set<string>()
        for (const label of c.labels) {
          if (label.text === null) continue
          if (!label.text.trim()) {
            add(
              'warning',
              `${name} has ${choice ? 'an option with no label' : 'an empty level'}`,
              label.loc,
              critWhere,
            )
          } else if (choice && labels.has(label.text)) {
            add(
              'warning',
              `${name} lists the option ${quote(label.text)} twice; ${LAST}`,
              label.loc,
              critWhere,
            )
          }
          labels.add(label.text)
        }
        // Python keeps one entry per repeated label, so the server counts distinct options.
        const count = choice ? labels.size : c.labels.length
        if (count > MAX_OPTIONS) {
          add(
            'error',
            `${name} has ${count} ${noun}; System One takes at most ${MAX_OPTIONS}`,
            c.loc,
            critWhere,
          )
        }
      }
    } else if (r.type === 'noul') {
      if (c.kind === 'arr' || c.kind === 'other') {
        add(
          'error',
          `The criteria of ${name} must be an object of true and false descriptions, or null`,
          c.loc,
          critWhere,
        )
      } else {
        for (const label of c.labels) {
          if (label.text !== 'true' && label.text !== 'false') {
            add(
              'warning',
              `${name} describes ${quote(label.text ?? '')}, which a noul question ignores; only true and false are read`,
              label.loc,
              critWhere,
            )
          }
        }
      }
    }
  }
  return problems
}

export function analyzeQuestions(editor: EditorState): SectionResult {
  if (editor.qMode === 'form') {
    const problems = checkQuestions(recordsFromForm(editor.questions), { el: LOC.addQuestion })
    return { problems, node: questionsNode(editor.questions), blocker: '' }
  }
  const parsed = tryParse(editor.qText)
  if (parsed.error) {
    return {
      problems: [parseProblem('questions', editor.qText, parsed.error)],
      node: null,
      blocker: '',
    }
  }
  const ast = parsed.ast
  if (ast.t !== 'obj') {
    return {
      problems: [
        {
          sec: 'questions',
          severity: 'error',
          msg: 'Questions must be an object that maps each key to a question',
          where: 'questions',
          loc: span('questions', ast),
        },
      ],
      node: ast,
      blocker: formBlocker(ast),
    }
  }
  const problems = checkQuestions(recordsFromAst(ast.entries), span('questions', ast))
  for (const entry of ast.entries) {
    // A choice's repeated labels are reported with the options above.
    const choice = (() => {
      const type = lastEntry(entry.v, 'type')?.v
      return type?.t === 'str' && type.v === 'choice'
    })()
    repeatedKeys(
      entry.v,
      [entry.k],
      (repeat, path) => {
        const where = ['questions', ...path].join(ARROW)
        const msg =
          path.length === 1
            ? `${quote(entry.k)} sets ${quote(repeat.k)} twice; ${LAST}`
            : path.length === 2 && path[1] === 'criteria'
              ? `${quote(entry.k)} describes ${quote(repeat.k)} twice; ${LAST}`
              : `Key ${quote(repeat.k)} appears twice in ${where}; ${LAST}`
        problems.push({
          sec: 'questions',
          severity: 'warning',
          msg,
          where,
          loc: keySpan('questions', repeat),
        })
      },
      (path) => choice && path.length === 2 && path[1] === 'criteria',
    )
  }
  return { problems, node: ast, blocker: formBlocker(ast) }
}

// --- the whole request -----------------------------------------------------------------------

/**
 * @param models The server's model names once its catalog has loaded, or null before; a name it
 * does not list is only a warning, since running it is how the 404 is seen.
 */
export function analyze(
  editor: EditorState,
  models: readonly string[] | null,
  vision?: boolean,
): Analysis {
  if (editor.protocol === 'openai') return analyzeOpenAI(editor, models, vision)
  const state = analyzeState(editor)
  const questions = analyzeQuestions(editor)
  const problems = [...state.problems, ...questions.problems]
  if (editor.model && models && !models.includes(editor.model)) {
    problems.push({
      sec: 'model',
      severity: 'warning',
      msg: `This server has no model ${quote(editor.model)}, so it would answer 404`,
      where: 'model',
      loc: { el: LOC.model },
    })
  }
  const errors = problems.filter((p) => p.severity === 'error').length
  return {
    state,
    questions,
    problems,
    errors,
    warnings: problems.length - errors,
    body:
      state.node && questions.node
        ? buildBody(state.node, questions.node, editor.model, 'typesafe', editor.requestText)
        : null,
  }
}

// --- locating a server error -----------------------------------------------------------------

/**
 * Where a 422 location such as ["questions", "urgency", "criteria"] is in the editors as they are
 * now, or null when it no longer exists.
 */
export function locate(editor: EditorState, path: readonly string[]): Loc | null {
  if (path[0] === 'model') return { el: LOC.model }
  if (path[0] === 'state' || path[0] === 'input') {
    if (editor.stateMode === 'fields') {
      const first = editor.fields[0]
      return { el: first ? LOC.field(first.id, 'name') : LOC.addField }
    }
    const parsed = tryParse(editor.stateText)
    return parsed.ast ? span('state', parsed.ast) : { text: 'state', at: [0, 0] }
  }
  if (path[0] !== 'questions') return null
  if (editor.qMode === 'form') {
    if (path.length < 2) return { el: LOC.addQuestion }
    const q = editor.questions.filter((x) => x.key === path[1]).pop()
    if (!q) return null
    if (path[2] === 'instructions') return { el: LOC.question(q.id, 'ins') }
    if (path[2] === 'type') return { el: LOC.question(q.id, 'type') }
    if (path[2] === 'criteria') return { el: criteriaEl(q) }
    return { el: LOC.question(q.id, 'key') }
  }
  const parsed = tryParse(editor.qText)
  if (!parsed.ast) return null
  let node: JsonNode = parsed.ast
  let at: readonly [number, number] = [node.s, node.s]
  for (const part of path.slice(1)) {
    const entry = lastEntry(node, part)
    if (entry) {
      at = [entry.ks, entry.ke]
      node = entry.v
    } else if (node.t === 'arr' && node.items[Number(part)]) {
      node = node.items[Number(part)]!
      at = [node.s, node.e]
    } else break
  }
  return { text: 'questions', at }
}
