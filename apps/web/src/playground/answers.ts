// Decision answers as distributions to draw.
//
// TypeSafe rounds probabilities to four decimals; OpenAI preserves them. A noul answer is
// p(true); a choice answer names the most likely option and carries every option's probability; a
// score answer carries the expected level, the legend and the probability of each level. The
// headline of each is the probability of the answer shown, and an answer whose headline is below
// 0.6 is flagged uncertain with its runner-up, the rule laya's playground applies (there to a
// calibrated confidence System One does not return). kev's `confidence` of a choice or a score is
// shown beside it as a secondary figure; noul has none and none is made up.

import { compactJson, lastEntry, type JsonNode } from './json'
import type { Protocol } from './request'

export const UNSURE = 0.6

/** A percentage in which a near-certainty never rounds up to 100%, nor a long shot down to 0%. */
export function pct(p: number): string {
  if (p > 0 && p < 0.001) return '<0.1%'
  if (p > 0.999 && p < 1) return '>99.9%'
  return `${(p * 100).toFixed(1)}%`
}

export interface DistRow {
  label: string
  p: number
  /** The option's description as sent, or '' when it has none. */
  desc: string
  /** A score row's level index. */
  level?: string
}

export interface AnswerView {
  /** Stable identity within a response: the TypeSafe key or OpenAI array position. */
  key: string
  label: string
  type: string
  instructions: string
  /** Empty when the answer has a shape this page does not know; `raw` then holds it. */
  rows: DistRow[]
  win: number
  headline: number
  unsure: boolean
  runnerUp: DistRow | null
  /** kev's confidence, for choice and score. */
  confidence: number | null
  /** A score answer's expected level, and the top of its scale. */
  score: number | null
  maxLevel: number
  raw: unknown
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/** Text for a sent value: strings as they are, other JSON compact, null and absence as ''. */
export function describeNode(node: JsonNode | undefined): string {
  if (!node || (node.t === 'lit' && node.v === null)) return ''
  return node.t === 'str' ? node.v : compactJson(node)
}

const describeValue = (value: unknown) =>
  value == null || value === '' ? '' : typeof value === 'string' ? value : JSON.stringify(value)

const probability = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) ? value : 0

function distribution(
  type: string,
  answer: Record<string, unknown>,
  question: JsonNode | undefined,
) {
  const criteria = lastEntry(question, 'criteria')?.v
  const probs = isRecord(answer.probabilities) ? answer.probabilities : {}
  if (type === 'noul' && typeof answer.noul === 'number') {
    const p = answer.noul
    const desc = (k: string) => describeNode(lastEntry(criteria, k)?.v)
    return {
      rows: [
        { label: 'true', p, desc: desc('true') },
        { label: 'false', p: 1 - p, desc: desc('false') },
      ],
      win: p >= 0.5 ? 0 : 1,
    }
  }
  if (type === 'choice' && Object.keys(probs).length) {
    const rows = Object.keys(probs)
      .map((label) => ({
        label,
        p: probability(probs[label]),
        desc: describeNode(lastEntry(criteria, label)?.v),
      }))
      .sort((a, b) => b.p - a.p)
    return {
      rows,
      win: Math.max(
        0,
        rows.findIndex((r) => r.label === String(answer.choice)),
      ),
    }
  }
  if (type === 'score' && Object.keys(probs).length) {
    const legend = isRecord(answer.legend) ? answer.legend : {}
    const rows = Object.keys(probs)
      .sort((a, b) => Number(a) - Number(b))
      .map((level) => ({
        label: describeValue(legend[level]) || level,
        p: probability(probs[level]),
        desc: '',
        level,
      }))
    // kev's mode: the first most likely level.
    return { rows, win: rows.reduce((w, r, j) => (r.p > rows[w]!.p ? j : w), 0) }
  }
  return { rows: [] as DistRow[], win: 0 }
}

function openaiDistribution(
  type: string,
  answer: Record<string, unknown>,
  question: JsonNode | undefined,
) {
  if (type === 'predicate' && typeof answer.probability === 'number') {
    const p = answer.probability
    return {
      rows: [
        { label: 'no', p: 1 - p, desc: '' },
        { label: 'yes', p, desc: '' },
      ],
      win: p > 0.5 ? 1 : 0,
    }
  }
  if ((type === 'choice' || type === 'score') && Array.isArray(answer.probabilities)) {
    const probs = answer.probabilities
    if (!probs.length || !probs.every(isRecord)) return { rows: [], win: 0 }
    const options = lastEntry(question, type === 'choice' ? 'choices' : 'levels')?.v
    const typed = type === 'choice' && probs.some((option) => typeof option.value === 'boolean')
    const rows: DistRow[] = probs.map((option, index) => ({
      label:
        type === 'score'
          ? describeValue(option.label)
          : typed
            ? (JSON.stringify(option.value) ?? '')
            : describeValue(option.value),
      p: probability(option.probability),
      desc: describeNode(
        lastEntry(options?.t === 'arr' ? options.items[index] : undefined, 'description')?.v,
      ),
      ...(type === 'score' ? { level: String(option.value) } : {}),
    }))
    return { rows, win: rows.reduce((w, row, j) => (row.p > rows[w]!.p ? j : w), 0) }
  }
  return { rows: [] as DistRow[], win: 0 }
}

export const positionalLabel = (name: unknown, index: number): string =>
  `#${index + 1} ${typeof name === 'string' ? name || '\u2205' : '(unnamed)'}`

/**
 * One view per answer, in the server's order.
 *
 * @param questions The questions as sent, for their instructions and option descriptions.
 */
export function answerViews(
  answers: unknown,
  questions: JsonNode | null,
  protocol: Protocol = 'typesafe',
): AnswerView[] {
  const entries: Array<[string, unknown]> =
    protocol === 'openai'
      ? Array.isArray(answers)
        ? answers.map((answer, index) => [String(index), answer])
        : []
      : isRecord(answers)
        ? Object.entries(answers)
        : []
  return entries.map(([key, answer], index) => {
    const question =
      protocol === 'openai'
        ? questions?.t === 'arr'
          ? questions.items[index]
          : undefined
        : lastEntry(questions, key)?.v
    const sentType = lastEntry(question, 'type')?.v
    const type =
      isRecord(answer) && typeof answer.type === 'string'
        ? answer.type
        : sentType?.t === 'str'
          ? sentType.v
          : ''
    const { rows, win } = isRecord(answer)
      ? (protocol === 'openai' ? openaiDistribution : distribution)(type, answer, question)
      : { rows: [], win: 0 }
    const top = rows[win]
    const headline = top?.p ?? 0
    const unsure = rows.length > 0 && headline < UNSURE
    const runnerUp = unsure
      ? (rows.filter((_, j) => j !== win).sort((a, b) => b.p - a.p)[0] ?? null)
      : null
    const record = isRecord(answer) ? answer : {}
    return {
      key,
      label: protocol === 'openai' ? positionalLabel(record.name, index) : key,
      type,
      instructions: describeNode(lastEntry(question, 'instructions')?.v),
      rows,
      win,
      headline,
      unsure,
      runnerUp,
      confidence:
        (type === 'choice' || type === 'score') && typeof record.confidence === 'number'
          ? record.confidence
          : null,
      score: type === 'score' && typeof record.score === 'number' ? record.score : null,
      maxLevel: Math.max(0, rows.length - 1),
      raw: answer,
    }
  })
}

/** What each question can answer, shown before the first run. */
export function previewOf(question: JsonNode, protocol: Protocol = 'typesafe'): string {
  const type = lastEntry(question, 'type')?.v
  const criteria = lastEntry(question, 'criteria')?.v
  const kind = type?.t === 'str' ? type.v : ''
  const count = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`
  if (protocol === 'openai') {
    if (kind === 'predicate') return 'no / yes'
    const options = lastEntry(question, kind === 'choice' ? 'choices' : 'levels')?.v
    if (kind === 'choice' || kind === 'score') {
      const word = kind === 'choice' ? 'option' : 'level'
      return options?.t === 'arr' && options.items.length
        ? count(options.items.length, word)
        : `no ${word}s yet`
    }
    return '\u2014'
  }
  if (kind === 'noul') return 'true / false'
  if (kind === 'choice') {
    return criteria?.t === 'obj' && criteria.entries.length
      ? count(new Set(criteria.entries.map((e) => e.k)).size, 'option')
      : 'no options yet'
  }
  if (kind === 'score') {
    return criteria?.t === 'arr' && criteria.items.length
      ? count(criteria.items.length, 'level')
      : 'no levels yet'
  }
  return '\u2014'
}

/** Splits `backtick` spans, as laya's presets write field names, into code and text parts. */
export function codeSpans(text: string): Array<{ code: boolean; text: string }> {
  const parts = text.split('`')
  if (parts.length < 3 || parts.length % 2 === 0) return [{ code: false, text }]
  return parts
    .map((part, j) => ({ code: j % 2 === 1, text: part }))
    .filter((part) => part.text !== '' || part.code)
}
