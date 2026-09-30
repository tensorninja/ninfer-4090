// A failed run, explained.
//
// System One answers errors in FastAPI's shape, `{"detail": ...}`: a list of pydantic errors for
// a 422 the schema raised, a string for everything else. The list is folded into one line per
// location, as laya's playground does (examples/server.py: detailLines; Apache-2.0), with one
// addition for kev's schema: a question is a union of Noul, Choice and Score that pydantic tries
// in turn, so a bad question fails once per branch with the branch named in `loc`:
//
//   ["body", "questions", "q", "Noul", "type"]                               Input should be 'noul'
//   ["body", "questions", "q", "function-after[_check(), Choice]", "criteria"]  Field required
//
// Only the branch the question's own `type` selects is kept, without its tag. A question whose
// type is missing or unknown folds into one line saying which types exist.

import { lastEntry, type JsonNode } from './json'
import { isQuestionType, orList, QUESTION_TYPES, type QuestionType } from './request'

export interface DetailLine {
  /** The location without its leading "body", as strings. */
  loc: string[]
  msg: string
}

/** The fields of a completed run that explain its failure. */
export interface Failure {
  status?: number
  text: string
  data: unknown
  /** A 2xx whose body is not a System One response. */
  odd: boolean
  /** Set when there was no response at all. */
  net?: string
  retryAfter?: string
}

export interface ErrorView {
  title: string
  help: string
  lines: DetailLine[]
}

const VALID = 'Input should be a valid '
/** A union branch tag: the model's name, or a validator wrapper around it. */
const BRANCH = /^(?:function-[a-z]+\[.*,\s*)?(Noul|Choice|Score)\]?$/

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const describe = (value: unknown) =>
  value == null || value === '' ? '' : typeof value === 'string' ? value : JSON.stringify(value)

/** The sent question under `key`, as the server read it: its last occurrence. */
function sentQuestion(questions: JsonNode | null, key: string) {
  const entry = lastEntry(questions, key)
  if (!entry) return null
  if (entry.v.t !== 'obj') return { isObj: false, type: null }
  const type = lastEntry(entry.v, 'type')?.v
  return { isObj: true, type: type?.t === 'str' && isQuestionType(type.v) ? type.v : null }
}

export function detailLines(detail: unknown, questions: JsonNode | null): DetailLine[] {
  if (!Array.isArray(detail)) return detail == null ? [] : [{ loc: [], msg: describe(detail) }]
  const errors: DetailLine[] = []
  for (const item of detail) {
    const d = isRecord(item) ? item : {}
    let loc = Array.isArray(d.loc)
      ? d.loc.filter((part, j) => !(j === 0 && part === 'body')).map(String)
      : []
    let msg = typeof d.msg === 'string' ? d.msg.replace(/^Value error, /, '') : JSON.stringify(item)
    const tag = loc[0] === 'questions' && loc.length >= 3 ? BRANCH.exec(loc[2]!) : null
    if (tag) {
      const branch = tag[1]!.toLowerCase() as QuestionType
      const sent = sentQuestion(questions, loc[1]!)
      if (sent?.isObj && sent.type) {
        if (branch !== sent.type) continue
        loc = [loc[0]!, loc[1]!, ...loc.slice(3)]
      } else if (sent?.isObj) {
        loc = ['questions', loc[1]!, 'type']
        msg = `Input should be ${orList(QUESTION_TYPES.map((t) => `'${t}'`))}`
      } else {
        loc = [loc[0]!, loc[1]!, ...loc.slice(3)]
      }
    }
    errors.push({ loc, msg })
  }

  const groups = new Map<string, { loc: string[]; msgs: string[] }>()
  for (const error of errors) {
    const id = JSON.stringify(error.loc)
    const group = groups.get(id) ?? { loc: error.loc, msgs: [] }
    if (!group.msgs.includes(error.msg)) group.msgs.push(error.msg)
    groups.set(id, group)
  }
  return Array.from(groups.values(), ({ loc, msgs }) => {
    const kinds = msgs.filter((m) => m.startsWith(VALID)).map((m) => m.slice(VALID.length))
    return {
      loc,
      msg:
        msgs.length > 1 && kinds.length === msgs.length ? VALID + orList(kinds) : msgs.join('; '),
    }
  })
}

const retryHint = (after: string | undefined) => {
  const secs = after !== undefined ? Number(after) : NaN
  return Number.isFinite(secs) && secs > 0 ? ` The server suggests retrying after ${secs} s.` : ''
}

/** Title, help and one line per problem for a run that did not answer. */
export function describeFailure(
  run: Failure,
  questions: JsonNode | null,
  endpoint: string,
): ErrorView {
  if (run.net !== undefined) {
    return {
      title: 'The server did not answer',
      help: `Could not reach ${endpoint} (${run.net}). Check that ninfer-serve is still running, then run the request again.`,
      lines: [],
    }
  }
  const status = run.status ?? 0
  const detail =
    !run.odd && isRecord(run.data) && 'detail' in run.data ? run.data.detail : undefined
  let lines = run.odd ? [] : detailLines(detail ?? null, questions)
  const unknownModel = status === 404 && typeof detail === 'string' && detail.startsWith('model ')
  if (unknownModel) lines = [{ loc: ['model'], msg: detail }]
  if (!lines.length) lines = [{ loc: [], msg: run.text.slice(0, 600) || 'No details.' }]

  if (run.odd) {
    return {
      title: `Unexpected response (HTTP ${status})`,
      help: "The server answered, but not with System One's answers object. The JSON view shows the whole body.",
      lines,
    }
  }
  if (status === 422 && Array.isArray(detail)) {
    return {
      title: 'The server rejected the request (HTTP 422)',
      help: 'Fix the parts above and run again; locations are clickable.',
      lines,
    }
  }
  if (status === 422) {
    return {
      title: 'The decision does not fit (HTTP 422)',
      help: "The state and each question's branch must fit the lane's context. Shorten the state or the longest question, then run again.",
      lines,
    }
  }
  if (unknownModel) {
    return {
      title: 'Unknown model (HTTP 404)',
      help: 'Pick one of the models this server lists; they come from GET /typesafe/v1/models.',
      lines,
    }
  }
  if (status === 400) {
    return {
      title: 'The server could not read the body (HTTP 400)',
      help: 'It takes UTF-8 JSON without lone surrogate escapes, integers of at most 4,300 digits, and nesting at most 1,000 deep.',
      lines,
    }
  }
  if (status === 413) {
    return {
      title: 'The request is too large (HTTP 413)',
      help: "The body is over the server's --max-request-mib limit. Shorten the state or split the questions.",
      lines,
    }
  }
  if (status === 429) {
    return {
      title: 'The server is busy (HTTP 429)',
      help: `Its request queue is full. Wait a moment, then run again.${retryHint(run.retryAfter)}`,
      lines,
    }
  }
  if (status === 503) {
    return {
      title: 'The server is unavailable (HTTP 503)',
      help: `The request waited too long for a lane, or the engine is not ready. Wait a moment, then run again.${retryHint(run.retryAfter)}`,
      lines,
    }
  }
  if (status === 401) {
    return {
      title: 'Authentication required (HTTP 401)',
      help: 'This server was started with --api-key, which the playground cannot send. Use the curl or Python code with your key, or run a server without --api-key.',
      lines,
    }
  }
  return {
    title:
      status >= 500
        ? `The decision failed on the server (HTTP ${status})`
        : `Request failed (HTTP ${status})`,
    help: 'Nothing was answered. The server log has the details.',
    lines,
  }
}
