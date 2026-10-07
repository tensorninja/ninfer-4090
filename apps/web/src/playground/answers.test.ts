import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { AnswerList, PreviewList } from './AnswerList'
import { answerViews, codeSpans, pct, previewOf } from './answers'
import { parseJson } from './json'
import { buildBody } from './request'
import { ResponsePane } from './ResponsePane'
import { PlaygroundStore } from './store'

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

const OPENAI_QUESTIONS = parseJson(`[
  {"type":"predicate","name":"same","instructions":"Is a refund requested?"},
  {"type":"choice","name":"same","instructions":"Which value?","choices":[
    {"value":true,"description":"first boolean"},
    {"value":"true","description":"a string"},
    {"value":true,"description":"second boolean"}
  ]},
  {"type":"score","instructions":"How urgent?","levels":[
    {"label":"today","description":"first level"},
    {"label":"today","description":"second level"},
    {"label":"tomorrow"}
  ]}
]`)
const OPENAI_ANSWERS = [
  { type: 'predicate', name: 'same', probability: 0.23456789 },
  {
    type: 'choice',
    name: 'same',
    choice: true,
    probabilities: [
      { value: true, probability: 0.15 },
      { value: 'true', probability: 0.2 },
      { value: true, probability: 0.65 },
    ],
    confidence: 0.475,
  },
  {
    type: 'score',
    name: null,
    score: 0.5,
    probabilities: [
      { value: 0, label: 'today', probability: 0.5 },
      { value: 1, label: 'today', probability: 0.5 },
      { value: 2, label: 'tomorrow', probability: 0 },
    ],
    confidence: 0.25,
  },
]

test('OpenAI answers join by position, not name, and preserve native predicate probabilities', () => {
  const views = answerViews(OPENAI_ANSWERS, OPENAI_QUESTIONS, 'openai')
  expect(views.map((v) => v.key)).toEqual(['0', '1', '2'])
  expect(views.map((v) => v.label)).toEqual(['#1 same', '#2 same', '#3 (unnamed)'])
  expect(views.map((v) => v.instructions)).toEqual([
    'Is a refund requested?',
    'Which value?',
    'How urgent?',
  ])
  const v = views[0]!
  expect(v.rows.map((row) => row.label)).toEqual(['no', 'yes'])
  expect(v.rows[1]!.p).toBe(0.23456789)
  expect(v.headline).toBe(1 - 0.23456789)
  expect(v.win).toBe(0)
  expect(v.confidence).toBeNull()
})

test('OpenAI keeps ordered duplicate typed choices and identifies the winning occurrence', () => {
  const v = answerViews(OPENAI_ANSWERS, OPENAI_QUESTIONS, 'openai')[1]!
  expect(v.rows.map((row) => [row.label, row.desc, row.p])).toEqual([
    ['true', 'first boolean', 0.15],
    ['"true"', 'a string', 0.2],
    ['true', 'second boolean', 0.65],
  ])
  expect(v.win).toBe(2)
  expect(v.headline).toBe(0.65)
  expect(v.confidence).toBe(0.475)
  const [tied] = answerViews(
    [
      {
        ...OPENAI_ANSWERS[1],
        probabilities: [
          { value: false, probability: 0.5 },
          { value: 'false', probability: 0.5 },
        ],
      },
    ],
    null,
    'openai',
  )
  expect(tied!.win).toBe(0)
  expect(tied!.runnerUp?.label).toBe('"false"')
})

test('OpenAI score levels retain indices, labels and descriptions, with first-mode ties', () => {
  const v = answerViews(OPENAI_ANSWERS, OPENAI_QUESTIONS, 'openai')[2]!
  expect(v.rows.map((row) => [row.level, row.label, row.desc])).toEqual([
    ['0', 'today', 'first level'],
    ['1', 'today', 'second level'],
    ['2', 'tomorrow', ''],
  ])
  expect(v.win).toBe(0)
  expect(v.score).toBe(0.5)
  expect(v.maxLevel).toBe(2)
  expect(v.confidence).toBe(0.25)
  expect(v.unsure).toBe(true)
})

test('OpenAI preview and answer lists show every duplicate and distinguish typed choices', () => {
  const preview = renderToStaticMarkup(
    createElement(PreviewList, {
      questions: OPENAI_QUESTIONS,
      protocol: 'openai',
    }),
  )
  expect(preview).toContain('#1 same')
  expect(preview).toContain('#2 same')
  expect(preview).toContain('#3 (unnamed)')
  expect(preview).toContain('no / yes')
  expect(preview).toContain('3 options')
  expect(preview).toContain('3 levels')
  const html = renderToStaticMarkup(
    createElement(AnswerList, {
      views: answerViews(OPENAI_ANSWERS, OPENAI_QUESTIONS, 'openai'),
      collapsed: new Set(['1']),
      onToggle: () => {},
    }),
  )
  expect(html.match(/<details/g)).toHaveLength(3)
  expect(html.match(/<details[^>]+open=""/g)).toHaveLength(2)
  expect(html).toContain('&quot;true&quot;')
  expect(html).toContain('first boolean')
  expect(html).toContain('second boolean')
})

test('the prior OpenAI response retains its protocol, token/cache usage and answers after a switch', () => {
  const store = new PlaygroundStore()
  const initial = store.getSnapshot()
  const body = buildBody(parseJson('"state"'), OPENAI_QUESTIONS, 'systemone-decision-v7', 'openai')
  const data = {
    model: 'systemone-decision-v7',
    answers: OPENAI_ANSWERS,
    usage: {
      input_tokens: 123,
      input_tokens_details: { cached_tokens: 37, cache_write_tokens: 0 },
      output_tokens: 0,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 123,
    },
  }
  const html = renderToStaticMarkup(
    createElement(ResponsePane, {
      s: {
        ...initial,
        editor: { ...initial.editor, protocol: 'typesafe' },
        ui: { ...initial.ui, view: 'answers' },
        last: {
          id: 'openai-run',
          body,
          data,
          status: 200,
          ok: true,
          odd: false,
          text: JSON.stringify(data),
          ms: 19,
          doneAt: 0,
        },
      },
      store,
      engine: {
        connection: 'offline',
        source: 'live',
        decisions: [],
        activeDecisions: [],
        decisionErrors: [],
      },
      onGoLive: () => {},
      onJump: () => {},
      paneRef: null,
    }),
  )
  expect(html).toContain('The request changed since this run')
  expect(html).toContain('#2 same')
  expect(html).toContain('second boolean')
  expect(html).toContain('<dt>input tokens</dt><dd title="123">123</dd>')
  expect(html).toContain('<dt>output tokens</dt><dd title="0">0</dd>')
  expect(html).toContain('<dt>total tokens</dt><dd title="123">123</dd>')
  expect(html).toContain('<dt>cached tokens</dt><dd title="37">37</dd>')
  expect(html).toContain('<dt>cache write tokens</dt><dd title="0">0</dd>')
  expect(html).toContain('x-request-id,')
  expect(html).not.toContain('x-typesafe-request-id')
  expect(html).not.toContain('<dt>latency</dt>')
})
