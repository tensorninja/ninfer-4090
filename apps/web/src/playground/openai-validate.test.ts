import { expect, test } from 'bun:test'

import { compactJson } from './json'
import { analyzeOpenAI } from './openai-validate'
import type { EditorState } from './request'
import type { Analysis, Problem } from './validate'

const predicate = { type: 'predicate', instructions: 'Is this relevant?' }
const choice = {
  type: 'choice',
  instructions: 'Choose.',
  choices: [{ value: true }, { value: 'yes' }],
}
const score = {
  type: 'score',
  instructions: 'Rate.',
  levels: [{ label: 'low' }, { label: 'high' }],
}

const editor = (stateText = '"context"', qText = JSON.stringify([predicate])): EditorState => ({
  protocol: 'openai',
  stateMode: 'json',
  qMode: 'json',
  stateText,
  qText,
  model: 'decision-adapter',
  fields: [],
  questions: [],
})

const analyze = (input: unknown, questions: unknown = [predicate]) =>
  analyzeOpenAI(editor(JSON.stringify(input), JSON.stringify(questions)), null)

function errorAt(analysis: Analysis, where: string): Problem {
  const problem = analysis.problems.find(
    (item) => item.severity === 'error' && item.where === where,
  )
  expect(problem).toBeDefined()
  return problem!
}

function highlighted(e: EditorState, problem: Problem): string {
  const loc = problem.loc
  expect(loc && 'text' in loc).toBe(true)
  if (!loc || !('text' in loc)) return ''
  return (loc.text === 'state' ? e.stateText : e.qText).slice(...loc.at)
}

test('native questions preserve order, duplicate names, and typed duplicate choice values', () => {
  const input = [
    { role: 'user', type: 'message', content: [{ type: 'input_text', text: 'context' }] },
  ]
  const questions = [
    { ...score, name: 'same', levels: [{ label: 'same', description: '' }, { label: 'same' }] },
    { ...predicate, name: 'same' },
    {
      ...choice,
      choices: [{ value: false }, { value: 'false', description: 'text' }, { value: false }],
    },
  ]
  const result = analyze(input, questions)
  expect(result.problems).toEqual([])
  expect(result.state.blocker).toBe('')
  expect(result.questions.blocker).toBe('')
  expect(result.body?.protocol).toBe('openai')
  expect(JSON.parse(result.body!.text)).toEqual({ input, questions, model: 'decision-adapter' })
})

test.each(
  [
    '',
    [],
    [{ role: 'user', content: '' }],
    [{ role: 'user', content: [] }],
    [
      { role: 'user', content: 'one' },
      { role: 'user', content: [{ type: 'input_text', text: 'two' }] },
    ],
  ].map((input) => ({ input })),
)('accepts native empty and text-only input: $input', ({ input }) => {
  expect(analyze(input).errors).toBe(0)
})

test.each([
  [null, 'input', 'Expected an array.'],
  [{ state: 'context' }, 'input', 'Expected an array.'],
  [[null], 'input[0]', 'Expected an object.'],
  [[{ role: 'assistant', content: 'text' }], 'input[0].role', 'Only user messages are supported.'],
  [[{ role: 'system', content: 'text' }], 'input[0].role', 'Only user messages are supported.'],
  [[{ content: 'text' }], 'input[0].role', 'Missing required parameter.'],
  [[{ role: 'user' }], 'input[0].content', 'Missing required parameter.'],
  [[{ role: 'user', content: null }], 'input[0].content', 'Expected an array.'],
  [[{ role: 'user', content: 'text', type: null }], 'input[0].type', 'Expected a string.'],
  [[{ role: 'user', content: 'text', type: 'other' }], 'input[0].type', 'Expected message.'],
  [[{ role: 'user', content: 'text', name: 'user' }], 'input[0].name', 'Unknown parameter.'],
] as const)('rejects invalid native input at %s', (input, path, message) => {
  expect(errorAt(analyze(input), path).msg).toBe(message)
})

test.each([
  [null, 'input[0].content[0]', 'Expected an object.'],
  [{ text: 'text' }, 'input[0].content[0].type', 'Missing required parameter.'],
  [
    { type: 'input_image', image_url: 'data:image/png;base64,AA==' },
    'input[0].content[0].type',
    'Decisions supports text input only.',
  ],
  [{ type: 'output_text', text: 'text' }, 'input[0].content[0].type', 'Expected input_text.'],
  [{ type: 'input_text' }, 'input[0].content[0].text', 'Missing required parameter.'],
  [{ type: 'input_text', text: false }, 'input[0].content[0].text', 'Expected a string.'],
  [
    { type: 'input_text', text: 'text', annotations: [] },
    'input[0].content[0].annotations',
    'Unknown parameter.',
  ],
] as const)('rejects invalid content parts: %j', (part, path, message) => {
  expect(errorAt(analyze([{ role: 'user', content: [part] }]), path).msg).toBe(message)
})

test.each([
  [{ q: predicate }, 'questions', 'Expected an array.'],
  [[null], 'questions[0]', 'Expected an object.'],
  [[{ instructions: 'text' }], 'questions[0].type', 'Missing required parameter.'],
  [[{ type: null }], 'questions[0].type', 'Expected a string.'],
  [
    [{ type: 'noul', instructions: 'text' }],
    'questions[0].type',
    'Expected predicate, choice, or score.',
  ],
  [[{ type: 'predicate' }], 'questions[0].instructions', 'Missing required parameter.'],
  [[{ ...predicate, instructions: null }], 'questions[0].instructions', 'Expected a string.'],
  [[{ ...predicate, name: false }], 'questions[0].name', 'Expected a string.'],
  [[{ ...predicate, criteria: null }], 'questions[0].criteria', 'Unknown parameter.'],
  [[{ ...predicate, choices: [] }], 'questions[0].choices', 'Unknown parameter.'],
  [[{ ...choice, levels: [] }], 'questions[0].levels', 'Unknown parameter.'],
  [
    [{ type: 'choice', instructions: 'text' }],
    'questions[0].choices',
    'Missing required parameter.',
  ],
  [[{ ...choice, choices: {} }], 'questions[0].choices', 'Expected an array.'],
  [[{ ...score, choices: [] }], 'questions[0].choices', 'Unknown parameter.'],
  [[{ type: 'score', instructions: 'text' }], 'questions[0].levels', 'Missing required parameter.'],
] as const)('rejects invalid native questions: %j', (questions, path, message) => {
  expect(errorAt(analyze('context', questions), path).msg).toBe(message)
})

test.each([
  ['choice', 'choices', 'value', true],
  ['score', 'levels', 'label', 'high'],
] as const)(
  'validates %s option objects without TypeSafe coercion',
  (type, key, valueKey, validValue) => {
    for (const [option, suffix, message] of [
      [null, '', 'Expected an object.'],
      [{}, `.${valueKey}`, 'Missing required parameter.'],
      [{ [valueKey]: 1 }, `.${valueKey}`, 'Expected a string.'],
      [{ [valueKey]: null }, `.${valueKey}`, 'Expected a string.'],
      [{ [valueKey]: validValue, description: null }, '.description', 'Expected a string.'],
      [{ [valueKey]: validValue, extra: '' }, '.extra', 'Unknown parameter.'],
    ] as const) {
      const result = analyze('context', [
        { type, instructions: '', [key]: [option, { [valueKey]: validValue }] },
      ])
      expect(errorAt(result, `questions[0].${key}[0]${suffix}`).msg).toBe(message)
    }
    if (type === 'score') {
      expect(
        errorAt(
          analyze('context', [{ ...score, levels: [{ label: false }, { label: 'high' }] }]),
          'questions[0].levels[0].label',
        ).msg,
      ).toBe('Expected a string.')
    }
  },
)

test.each([
  ['questions', 1, 200],
  ['choices', 2, 255],
  ['levels', 2, 10],
] as const)(
  '%s enforces native array bounds, counting duplicates as separate items',
  (key, min, max) => {
    for (const count of [min - 1, min, max, max + 1]) {
      const item =
        key === 'questions' ? predicate : key === 'choices' ? { value: true } : { label: 'same' }
      const items = Array.from({ length: count }, () => item)
      const result = analyze(
        'context',
        key === 'questions' ? items : [{ ...(key === 'choices' ? choice : score), [key]: items }],
      )
      if (count < min || count > max) {
        expect(errorAt(result, key === 'questions' ? key : `questions[0].${key}`).msg).toBe(
          `Expected between ${min} and ${max} items.`,
        )
      } else {
        expect(result.problems).toEqual([])
      }
    }
  },
)

test('schema errors highlight exact values, unknown keys, and missing-field objects', () => {
  const e = editor(
    '[{"role":"assistant","content":"context"}]',
    '[{"type":"predicate","instructions":null,"extra":1},{"type":"choice","instructions":""}]',
  )
  const result = analyzeOpenAI(e, null)
  expect(highlighted(e, errorAt(result, 'input[0].role'))).toBe('"assistant"')
  expect(highlighted(e, errorAt(result, 'questions[0].instructions'))).toBe('null')
  expect(highlighted(e, errorAt(result, 'questions[0].extra'))).toBe('"extra"')
  expect(highlighted(e, errorAt(result, 'questions[1].choices'))).toBe(
    '{"type":"choice","instructions":""}',
  )
})

test('duplicate keys keep their spelling and last value, including overwritten large integers', () => {
  const e = editor(
    '[{"role":"assistant","role":"user","content":"context"}]',
    '[{"type":9007199254740993,"type":"choice","instructions":null,"instructions":"","choices":[{"value":9007199254740993,"value":true},{"value":false}]}]',
  )
  const result = analyzeOpenAI(e, null)
  expect(result.errors).toBe(0)
  expect(result.warnings).toBe(4)
  expect(result.body?.text).toContain(e.stateText)
  expect(result.body?.text).toContain(e.qText)
  expect(compactJson(result.questions.node!)).toBe(e.qText)
  const typeWarning = result.problems.find((problem) => problem.where === 'questions[0].type')!
  expect(typeWarning.severity).toBe('warning')
  expect(typeWarning.loc).toEqual({
    text: 'questions',
    at: [e.qText.lastIndexOf('"type"'), e.qText.lastIndexOf('"type"') + 6],
  })
})

test.each(['state', 'questions'] as const)(
  'invalid %s JSON reports syntax position and has no body',
  (section) => {
    const e = editor()
    const broken = '[\n  ] extra'
    if (section === 'state') e.stateText = broken
    else e.qText = broken
    const result = analyzeOpenAI(e, null)
    const problem = result.problems.find((item) => item.parse)!
    expect(result.body).toBeNull()
    expect(result[section].blocker).toBe('')
    expect(problem).toMatchObject({
      sec: section,
      severity: 'error',
      where: `${section === 'state' ? 'input' : 'questions'}, line 2`,
      loc: { text: section, at: [broken.indexOf('extra'), broken.indexOf('extra')] },
    })
  },
)

test('OpenAI always reads the native JSON instead of TypeSafe form drafts', () => {
  const e = { ...editor(), stateMode: 'fields' as const, qMode: 'form' as const }
  const result = analyzeOpenAI(e, null)
  expect(result.errors).toBe(0)
  expect(JSON.parse(result.body!.text).input).toBe('context')
  expect(JSON.parse(result.body!.text).questions).toEqual([predicate])
})

test('model is required, TypeSafe aliases are blocked, and unknown pool names only warn', () => {
  for (const model of ['', ' ', 'jev-latest']) {
    const result = analyzeOpenAI({ ...editor(), model }, null)
    expect(errorAt(result, 'model').loc).toEqual({ el: 'model' })
  }
  expect(analyzeOpenAI(editor(), null).problems).toEqual([])
  expect(analyzeOpenAI(editor(), ['decision-adapter']).problems).toEqual([])
  const unknown = analyzeOpenAI(editor(), ['other-adapter'])
  expect(unknown.errors).toBe(0)
  expect(unknown.warnings).toBe(1)
  expect(unknown.problems[0]).toMatchObject({ sec: 'model', severity: 'warning', where: 'model' })
})

test('string limits count Unicode characters, with a separate larger input limit', () => {
  const limit = 1048576
  const instructions = '\u{1d11e}'.repeat(limit)
  expect(analyze('context', [{ ...predicate, instructions }]).errors).toBe(0)
  expect(
    errorAt(
      analyze('context', [{ ...predicate, instructions: instructions + 'x' }]),
      'questions[0].instructions',
    ).msg,
  ).toBe(`String exceeds the maximum of ${limit} Unicode characters.`)
  expect(analyze('x'.repeat(limit + 1)).errors).toBe(0)
  expect(errorAt(analyze('x'.repeat(10485761)), 'input').msg).toBe(
    'String exceeds the maximum of 10485760 Unicode characters.',
  )
})
