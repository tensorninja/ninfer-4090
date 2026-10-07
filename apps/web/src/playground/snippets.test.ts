import { expect, test } from 'bun:test'

import { parseJson } from './json'
import { buildBody } from './request'
import { curlSnippet, endpointFor, pythonSnippet } from './snippets'

const QUESTIONS = parseJson(`[
  {"type":"predicate","name":"same","instructions":"Is it urgent?"},
  {"type":"choice","name":"same","instructions":"Which?","choices":[
    {"value":true,"description":"a boolean"},{"value":"true"},{"value":true}
  ]},
  {"type":"score","instructions":"How urgent?","levels":[
    {"label":"Today","description":"This afternoon"},{"label":"Later"}
  ]}
]`)

test('OpenAI curl sends the native input, ordered questions and actual model to decisions', () => {
  const body = buildBody(parseJson('"it\'s urgent"'), QUESTIONS, 'systemone-decision-v7', 'openai')
  const curl = curlSnippet('http://h:8080', body)
  expect(endpointFor('typesafe')).toBe('/typesafe/v1/systemone')
  expect(endpointFor('openai')).toBe('/v1/decisions')
  expect(curl.split('\n')[0]).toBe('curl -s http://h:8080/v1/decisions \\')
  const payload = curl.split("  -d '")[1]!.slice(0, -1).replace(/'\\''/g, "'")
  expect(JSON.parse(payload)).toEqual({
    input: "it's urgent",
    questions: [
      { type: 'predicate', name: 'same', instructions: 'Is it urgent?' },
      {
        type: 'choice',
        name: 'same',
        instructions: 'Which?',
        choices: [{ value: true, description: 'a boolean' }, { value: 'true' }, { value: true }],
      },
      {
        type: 'score',
        instructions: 'How urgent?',
        levels: [{ label: 'Today', description: 'This afternoon' }, { label: 'Later' }],
      },
    ],
    model: 'systemone-decision-v7',
  })
})

test('OpenAI Python uses the Decisions SDK with native typed options and message input', () => {
  const input = parseJson('[{"role":"user","content":[{"type":"input_text","text":"urgent"}]}]')
  const body = buildBody(input, QUESTIONS, 'systemone-decision-v7', 'openai')
  const py = pythonSnippet('http://h:8080', body)
  expect(py).toStartWith('from openai import OpenAI\n\n')
  expect(py).toContain('client = OpenAI(api_key="local", base_url="http://h:8080/v1")')
  expect(py).toContain('response = client.decisions.create(\n')
  expect(py).toContain('    model="systemone-decision-v7",\n')
  expect(py).toContain('    input=[\n')
  expect(py).toContain('"type": "input_text"')
  expect(py).toContain('    questions=[\n')
  expect(py.match(/"name": "same"/g)).toHaveLength(2)
  expect(py.match(/"value": True/g)).toHaveLength(2)
  expect(py).toContain('"value": "true"')
  expect(py).toContain('"label": "Today"')
  expect(py).toContain('"description": "This afternoon"')
  expect(py).toContain(
    'for index, answer in enumerate(response.answers):\n    print(index, answer)',
  )
  expect(py).not.toContain('TypeSafe')
  expect(py).not.toContain('state=')
  expect(py).not.toContain('#')
})

test('TypeSafe snippets retain their endpoint, SDK classes and optional model behavior', () => {
  const body = buildBody(
    parseJson('"s"'),
    parseJson('{"q":{"type":"noul","instructions":"i"}}'),
    '',
    'typesafe',
  )
  expect(curlSnippet('http://h', body)).toContain('http://h/typesafe/v1/systemone')
  const py = pythonSnippet('http://h', body)
  expect(py).toContain('from typesafe_sdk import Noul, TypeSafeClient')
  expect(py).toContain('base_url="http://h/typesafe"')
  expect(py).toContain('client.system_one(')
  expect(py).toContain('"q": Noul(instructions="i")')
  expect(py).not.toContain('model=')
})
