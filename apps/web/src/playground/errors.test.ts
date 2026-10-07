import { expect, test } from 'bun:test'

import { parseJson } from './json'
import { describeFailure, detailLines, type Failure } from './errors'

// Bodies ninfer-serve answered on 2026-09-30 for the requests beside them (x-typesafe-request-id
// fixture*). They follow pydantic's union reporting: one error per branch, tagged in `loc`.
const LIVE = {
  choiceList: {
    sent: '{"q":{"type":"choice","criteria":["a","b"]}}',
    detail: [
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'q', 'Noul', 'type'],
        msg: "Input should be 'noul'",
      },
      {
        type: 'dict_type',
        loc: ['body', 'questions', 'q', 'Noul', 'criteria'],
        msg: 'Input should be a valid dictionary',
      },
      {
        type: 'dict_type',
        loc: ['body', 'questions', 'q', 'function-after[_check(), Choice]', 'criteria'],
        msg: 'Input should be a valid dictionary',
      },
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'q', 'Score', 'type'],
        msg: "Input should be 'score'",
      },
    ],
  },
  choiceEmpty: {
    sent: '{"q":{"type":"choice","criteria":{}}}',
    detail: [
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'q', 'Noul', 'type'],
        msg: "Input should be 'noul'",
      },
      {
        type: 'value_error',
        loc: ['body', 'questions', 'q', 'function-after[_check(), Choice]'],
        msg: 'Value error, criteria must have 1..255 options',
      },
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'q', 'Score', 'type'],
        msg: "Input should be 'score'",
      },
      {
        type: 'list_type',
        loc: ['body', 'questions', 'q', 'Score', 'criteria'],
        msg: 'Input should be a valid list',
      },
    ],
  },
  missingType: {
    sent: '{"q":{"instructions":"x"}}',
    detail: [
      { type: 'missing', loc: ['body', 'questions', 'q', 'Noul', 'type'], msg: 'Field required' },
      {
        type: 'missing',
        loc: ['body', 'questions', 'q', 'function-after[_check(), Choice]', 'type'],
        msg: 'Field required',
      },
      {
        type: 'missing',
        loc: ['body', 'questions', 'q', 'function-after[_check(), Choice]', 'criteria'],
        msg: 'Field required',
      },
      { type: 'missing', loc: ['body', 'questions', 'q', 'Score', 'type'], msg: 'Field required' },
      {
        type: 'missing',
        loc: ['body', 'questions', 'q', 'Score', 'criteria'],
        msg: 'Field required',
      },
    ],
  },
  notAnObject: {
    sent: '{"q":5}',
    detail: ['Noul', 'function-after[_check(), Choice]', 'Score'].map((tag) => ({
      type: 'model_attributes_type',
      loc: ['body', 'questions', 'q', tag],
      msg: 'Input should be a valid dictionary or object to extract fields from',
    })),
  },
  twoBad: {
    sent: '{"a":{"type":"choice","criteria":[]},"b":{"type":"score"}}',
    detail: [
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'a', 'Noul', 'type'],
        msg: "Input should be 'noul'",
      },
      {
        type: 'dict_type',
        loc: ['body', 'questions', 'a', 'Noul', 'criteria'],
        msg: 'Input should be a valid dictionary',
      },
      {
        type: 'dict_type',
        loc: ['body', 'questions', 'a', 'function-after[_check(), Choice]', 'criteria'],
        msg: 'Input should be a valid dictionary',
      },
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'a', 'Score', 'type'],
        msg: "Input should be 'score'",
      },
      {
        type: 'too_short',
        loc: ['body', 'questions', 'a', 'Score', 'criteria'],
        msg: 'List should have at least 1 item after validation, not 0',
      },
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'b', 'Noul', 'type'],
        msg: "Input should be 'noul'",
      },
      {
        type: 'literal_error',
        loc: ['body', 'questions', 'b', 'function-after[_check(), Choice]', 'type'],
        msg: "Input should be 'choice'",
      },
      {
        type: 'missing',
        loc: ['body', 'questions', 'b', 'function-after[_check(), Choice]', 'criteria'],
        msg: 'Field required',
      },
      {
        type: 'missing',
        loc: ['body', 'questions', 'b', 'Score', 'criteria'],
        msg: 'Field required',
      },
    ],
  },
}

const lines = (fixture: { sent: string; detail: unknown }) =>
  detailLines(fixture.detail, parseJson(fixture.sent)).map((l) => `${l.loc.join(' > ')}: ${l.msg}`)

test('only the branch of the type that was sent is kept, without its tag', () => {
  expect(lines(LIVE.choiceList)).toEqual([
    'questions > q > criteria: Input should be a valid dictionary',
  ])
  expect(lines(LIVE.choiceEmpty)).toEqual(['questions > q: criteria must have 1..255 options'])
  expect(lines(LIVE.twoBad)).toEqual([
    'questions > a > criteria: Input should be a valid dictionary',
    'questions > b > criteria: Field required',
  ])
})

test('a missing or unknown type folds into one line naming the types', () => {
  expect(lines(LIVE.missingType)).toEqual([
    "questions > q > type: Input should be 'noul', 'choice' or 'score'",
  ])
  expect(lines({ ...LIVE.missingType, sent: '{"q":{"type":"foo"}}' })).toEqual([
    "questions > q > type: Input should be 'noul', 'choice' or 'score'",
  ])
})

test('a question that is not an object says so once', () => {
  expect(lines(LIVE.notAnObject)).toEqual([
    'questions > q: Input should be a valid dictionary or object to extract fields from',
  ])
})

test('errors outside the questions pass through, without the leading body', () => {
  const detail = [
    {
      type: 'too_short',
      loc: ['body', 'questions'],
      msg: 'Dictionary should have at least 1 item after validation, not 0',
    },
    { type: 'string_type', loc: ['body', 'model'], msg: 'Input should be a valid string' },
  ]
  expect(lines({ sent: '{}', detail })).toEqual([
    'questions: Dictionary should have at least 1 item after validation, not 0',
    'model: Input should be a valid string',
  ])
})

test('each status gets its own explanation', () => {
  const run = (status: number, detail: unknown, extra = {}) => ({
    status,
    text: JSON.stringify({ detail }),
    data: { detail },
    odd: false,
    ...extra,
  })
  const questions = parseJson('{"q":{"type":"noul"}}')
  const unknown = describeFailure(
    run(
      404,
      "model 'nope' not found; System One models: systemone-decision-v7, jev-latest (systemone-decision-v7)",
    ),
    questions,
    '/x',
  )
  expect(unknown.title).toBe('Unknown model (HTTP 404)')
  expect(unknown.lines[0]!.loc).toEqual(['model'])
  expect(describeFailure(run(404, 'Not Found'), questions, '/x').title).toBe(
    'Request failed (HTTP 404)',
  )
  expect(describeFailure(run(422, 'branch too long: 80000 tokens'), questions, '/x').title).toBe(
    'The decision does not fit (HTTP 422)',
  )
  expect(
    describeFailure(run(429, 'queue full', { retryAfter: '1' }), questions, '/x').help,
  ).toContain('retrying after 1 s')
  expect(describeFailure(run(413, 'too big'), questions, '/x').title).toBe(
    'The request is too large (HTTP 413)',
  )
  expect(describeFailure(run(401, 'missing key'), questions, '/x').help).toContain('--api-key')
  expect(describeFailure(run(500, 'Internal Server Error'), questions, '/x').title).toBe(
    'The decision failed on the server (HTTP 500)',
  )
  const odd = describeFailure(
    { status: 200, text: '{"x":1}', data: { x: 1 }, odd: true },
    questions,
    '/x',
  )
  expect(odd.title).toBe('Unexpected response (HTTP 200)')
  expect(odd.lines).toEqual([{ loc: [], msg: '{"x":1}' }])
  const net = describeFailure(
    { text: '', data: null, odd: false, net: 'Failed to fetch' },
    questions,
    'http://h/x',
  )
  expect(net.help).toContain('Could not reach http://h/x (Failed to fetch)')
})

const openaiFailure = (
  status: number,
  code: string | null,
  param: string | null,
  message: string,
  extra: Partial<Failure> = {},
) => {
  const data = { error: { message, type: 'invalid_request_error', param, code } }
  return describeFailure(
    { status, text: JSON.stringify(data), data, odd: false, ...extra },
    parseJson('[{"type":"predicate","instructions":"i"}]'),
    '/v1/decisions',
    'openai',
  )
}

test('OpenAI errors retain the message, code, type and positional parameter path', () => {
  const error = openaiFailure(
    400,
    'invalid_type',
    'questions[0].choices[1].value',
    'Expected a string or boolean.',
  )
  expect(error.title).toBe('The server rejected the request (HTTP 400)')
  expect(error.lines).toEqual([
    {
      loc: ['questions', '0', 'choices', '1', 'value'],
      msg: 'Expected a string or boolean.',
      code: 'invalid_type',
      type: 'invalid_request_error',
    },
  ])
  expect(error.help).not.toContain('4,300')
  expect(openaiFailure(401, null, null, 'Invalid API key.').lines).toEqual([
    {
      loc: [],
      msg: 'Invalid API key.',
      type: 'invalid_request_error',
    },
  ])
})

test('OpenAI model, modality, overflow and retry errors explain their native contracts', () => {
  const model = openaiFailure(404, 'model_not_found', 'model', "decision model 'nope' not found")
  expect(model.title).toBe('Unknown model (HTTP 404)')
  expect(model.lines[0]!.loc).toEqual(['model'])
  expect(model.help).toContain('GET /v1/models')
  expect(model.help).toContain('jev-latest is TypeSafe-only')
  expect(
    openaiFailure(400, 'unsupported_modality', 'input[0].content[0]', 'Images unsupported.').title,
  ).toBe('Unsupported input modality (HTTP 400)')
  const overflow = openaiFailure(400, 'context_length_exceeded', 'input', 'Too many tokens.')
  expect(overflow.title).toBe('The decision does not fit (HTTP 400)')
  expect(overflow.help).toContain('rejects overflow instead of truncating')
  for (const [status, code] of [
    [429, 'server_overloaded'],
    [503, 'request_queue_timeout'],
  ] as const) {
    const busy = openaiFailure(status, code, null, 'Try later.', { retryAfter: '1' })
    expect(busy.lines[0]!.code).toBe(code)
    expect(busy.help).toContain('retrying after 1 s')
  }
  expect(openaiFailure(401, null, null, 'Invalid API key.').help).toContain('--api-key')
})

test('an unexpected OpenAI success explains the expected answers array', () => {
  const odd = openaiFailure(200, null, null, '', {
    odd: true,
    data: { answers: {} },
    text: '{"answers":{}}',
  })
  expect(odd.title).toBe('Unexpected response (HTTP 200)')
  expect(odd.help).toContain("OpenAI Decisions' answers array")
  expect(odd.lines).toEqual([{ loc: [], msg: '{"answers":{}}' }])
})
