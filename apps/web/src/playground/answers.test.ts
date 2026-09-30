import { expect, test } from 'bun:test'

import { answerViews, codeSpans, pct, previewOf } from './answers'
import { parseJson } from './json'

// The billing-email example and ninfer-serve's answer to it on 2026-09-30 (systemone-decision-v7,
// rk4v4 KV). refund_requested gained descriptions here, and its answer is the false-leaning 0.2835
// the email preset's is_phishing received, so both noul outcomes are covered.
const SENT = parseJson(`{
  "department": {"type": "choice", "instructions": "Which department should handle this request?",
    "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
      "sales": "pricing, new contracts", "other": "everything else"}},
  "urgency": {"type": "score", "instructions": "How urgent is this request?",
    "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
  "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
  "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?",
    "criteria": {"true": "asks for money back", "false": null}}
}`)
const ANSWERS = {
  department: {
    type: 'choice',
    choice: 'billing',
    confidence: 0.9801,
    probabilities: { billing: 0.9851, technical: 0.0008, sales: 0.0025, other: 0.0116 },
  },
  urgency: {
    type: 'score',
    score: 1.5016,
    legend: { '0': 'not urgent', '1': 'soon', '2': 'critical deadline or blocking issue' },
    probabilities: { '0': 0.0171, '1': 0.4642, '2': 0.5187 },
    confidence: 0.2524,
  },
  churn_risk: { type: 'noul', noul: 0.9764 },
  refund_requested: { type: 'noul', noul: 0.2835 },
}

const views = answerViews(ANSWERS, SENT)
const byKey = (key: string) => views.find((v) => v.key === key)!

test('a choice lists its options by probability, with the descriptions that were sent', () => {
  const v = byKey('department')
  expect(v.rows.map((r) => r.label)).toEqual(['billing', 'other', 'sales', 'technical'])
  expect(v.rows[0]!.desc).toBe('invoices, payments, refunds')
  expect(v.rows[v.win]!.label).toBe('billing')
  expect(v.headline).toBe(0.9851)
  expect(v.confidence).toBe(0.9801)
  expect(v.unsure).toBe(false)
})

test('a score headlines its most likely level and flags a close call', () => {
  const v = byKey('urgency')
  expect(v.rows.map((r) => [r.level, r.label])).toEqual([
    ['0', 'not urgent'],
    ['1', 'soon'],
    ['2', 'critical deadline or blocking issue'],
  ])
  expect(v.win).toBe(2)
  expect(v.headline).toBe(0.5187)
  expect(v.unsure).toBe(true)
  expect(v.runnerUp?.label).toBe('soon')
  expect(v.score).toBe(1.5016)
  expect(v.maxLevel).toBe(2)
  expect(v.confidence).toBe(0.2524)
})

test('a noul answer is true or false with no confidence of its own', () => {
  const yes = byKey('churn_risk')
  expect(yes.rows.map((r) => r.label)).toEqual(['true', 'false'])
  expect(yes.win).toBe(0)
  expect(yes.confidence).toBeNull()
  const no = byKey('refund_requested')
  expect(no.win).toBe(1)
  expect(no.headline).toBeCloseTo(0.7165, 10)
  expect(no.rows.map((r) => r.desc)).toEqual(['asks for money back', ''])
  expect(no.instructions).toBe('Does the user explicitly request a refund?')
})

test('an answer of an unknown shape is kept raw', () => {
  const [v] = answerViews({ q: { type: 'rank', order: ['a'] } }, null)
  expect(v!.rows).toEqual([])
  expect(v!.raw).toEqual({ type: 'rank', order: ['a'] })
})

test('percentages never round a near-certainty to 100% or a long shot to 0%', () => {
  expect([0, 0.0004, 0.001, 0.5187, 0.9994, 1].map(pct)).toEqual([
    '0.0%',
    '<0.1%',
    '0.1%',
    '51.9%',
    '>99.9%',
    '100.0%',
  ])
})

test('the preview says what each question can answer', () => {
  const q = (text: string) => previewOf(parseJson(text))
  expect(q('{"type": "noul"}')).toBe('true / false')
  expect(q('{"type": "choice", "criteria": {"a": 1, "b": 2, "a": 3}}')).toBe('2 options')
  expect(q('{"type": "score"}')).toBe('no levels yet')
  expect(q('{"type": "rank"}')).toBe('\u2014')
})

test('backtick spans become code', () => {
  expect(codeSpans('What does `message` say?')).toEqual([
    { code: false, text: 'What does ' },
    { code: true, text: 'message' },
    { code: false, text: ' say?' },
  ])
  expect(codeSpans('an `odd one')).toEqual([{ code: false, text: 'an `odd one' }])
})
