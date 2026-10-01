import { expect, test } from 'bun:test'

import { compactJson, formatJson, parseJson, type ObjectNode } from './json'
import { PRESETS } from './presets'
import {
  buildBody,
  editorFromSnap,
  fieldsBlocker,
  formBlocker,
  questionFromEntry,
  questionNode,
  snapFromValue,
  snapOf,
  withType,
  type EditorState,
  type Snap,
} from './request'
import { curlSnippet, pythonSnippet } from './snippets'
import { analyze, locate } from './validate'

const snap = (state: string, questions: string, over: Partial<Snap> = {}): Snap => ({
  stateMode: 'json',
  qMode: 'json',
  stateText: state,
  qText: questions,
  model: '',
  ...over,
})

const editor = (state: string, questions: string, over: Partial<Snap> = {}): EditorState =>
  editorFromSnap(snap(state, questions, over))

const problems = (e: EditorState) => analyze(e, null).problems.map((p) => [p.severity, p.msg])

test.each([...PRESETS])(
  '$name loads and restores without changing the request structure',
  (preset) => {
    const initial = editorFromSnap(snapFromValue(preset, { stateMode: 'fields', qMode: 'form' }))
    for (const e of [initial, editorFromSnap(snapOf(initial))]) {
      const analysis = analyze(e, null)
      expect(analysis.errors).toBe(0)
      expect(analysis.body?.text).toBe(
        JSON.stringify({ state: preset.state, questions: preset.questions }),
      )
    }
  },
)

test('the body is the text as typed, compact, with integers beyond 2^53 intact', () => {
  const body = buildBody(
    parseJson('{"id": 9007199254740993}'),
    parseJson('{"q": {"type": "noul"}}'),
    'm',
  )
  expect(body.text).toBe(
    '{"state":{"id":9007199254740993},"questions":{"q":{"type":"noul"}},"model":"m"}',
  )
  expect(buildBody(parseJson('"s"'), parseJson('{}'), '').text).toBe('{"state":"s","questions":{}}')
})

test('canon ignores key order inside a question and a null criteria, nothing else', () => {
  const canon = (questions: string) => buildBody(parseJson('"s"'), parseJson(questions), '').canon
  const a = canon('{"q": {"type": "noul", "instructions": "i"}, "r": {"type": "noul"}}')
  expect(
    canon('{"q": {"instructions": "i", "type": "noul", "criteria": null}, "r": {"type": "noul"}}'),
  ).toBe(a)
  expect(canon('{"r": {"type": "noul"}, "q": {"type": "noul", "instructions": "i"}}')).not.toBe(a)
  expect(canon('{"q": {"type": "choice", "criteria": {"a": null, "b": null}}}')).not.toBe(
    canon('{"q": {"type": "choice", "criteria": {"b": null, "a": null}}}'),
  )
})

test('the form refuses JSON it cannot show exactly', () => {
  const why = (text: string) => formBlocker(parseJson(text))
  expect(why('{"q": {"type": "noul", "instructions": "i"}}')).toBe('')
  expect(why('[]')).toBe('Questions must be a JSON object to use the form.')
  expect(why('{"q": {"instructions": "i"}}')).toBe(
    '\u201cq\u201d needs a type (noul, choice or score) before the form can show it.',
  )
  expect(why('{"q": {"type": "choice", "criteria": ["a"]}}')).toBe(
    'The options of \u201cq\u201d are a list; System One takes label \u2192 description.',
  )
  expect(why('{"q": {"type": "noul", "instructions": {"x": 1}}}')).toBe(
    'The instructions of \u201cq\u201d are not text.',
  )
  expect(why('{"q": {"type": "noul", "criteria": {"maybe": "m"}}}')).toContain('only JSON can edit')
  expect(fieldsBlocker(parseJson('{"a": "x", "b": 2}'))).toBe(
    'Field \u201cb\u201d holds a number, which only JSON can edit.',
  )
  expect(fieldsBlocker(parseJson('"text"'))).toContain('plain text')
})

test('a question survives the form: extra keys, null descriptions and absent instructions', () => {
  const text =
    '{"type": "choice", "criteria": {"a": null, "b": "", "c": "desc"}, "tag": [1, 2], "type": "choice"}'
  const q = questionFromEntry((parseJson(`{"q": ${text}}`) as ObjectNode).entries[0]!)
  expect(compactJson(questionNode(q))).toBe(
    '{"type":"choice","criteria":{"a":null,"b":"","c":"desc"},"tag":[1,2]}',
  )
  const noul = questionFromEntry(
    (parseJson('{"q": {"type": "noul", "instructions": null}}') as ObjectNode).entries[0]!,
  )
  expect(compactJson(questionNode(noul))).toBe('{"type":"noul","instructions":null}')
})

test('a type switch carries choice labels into score levels and back', () => {
  const q = questionFromEntry(
    (parseJson('{"q": {"type": "choice", "criteria": {"low": "l", "high": null}}}') as ObjectNode)
      .entries[0]!,
  )
  const score = withType(q, 'score')
  expect(score.levels.map((l) => l.text)).toEqual(['low', 'high'])
  expect(compactJson(questionNode(score))).toBe('{"type":"score","criteria":["low","high"]}')
  expect(withType(score, 'choice').options.map((o) => o.label)).toEqual(['low', 'high'])
})

test('snapshots keep invalid text and fall back to JSON where the form cannot show it', () => {
  const broken = editor('{"a": ', '{"q": {"type": "noul"}}', { stateMode: 'fields', qMode: 'form' })
  expect(broken.stateMode).toBe('json')
  expect(broken.qMode).toBe('form')
  expect(snapOf(broken).stateText).toBe('{"a": ')
  expect(snapOf(broken).qText).toBe(formatJson(parseJson('{"q": {"type": "noul"}}')) + '\n')
})

test("errors are what kev's schema rejects; warnings are what it accepts", () => {
  expect(problems(editor('"s"', '{}'))).toEqual([['error', 'Add at least one question']])
  expect(problems(editor('"s"', '{"q": {"instructions": "i"}}'))).toEqual([
    ['error', '\u201cq\u201d has no type; use noul, choice or score'],
  ])
  expect(problems(editor('"s"', '{"q": {"type": 5, "instructions": "i"}}'))).toEqual([
    ['error', '\u201cq\u201d has an unknown type \u201c5\u201d; use noul, choice or score'],
  ])
  expect(
    problems(editor('"s"', '{"q": {"type": "choice", "instructions": "i", "criteria": ["a"]}}')),
  ).toEqual([
    [
      'error',
      'The options of \u201cq\u201d are a list; System One takes an object of label \u2192 description',
    ],
  ])
  expect(
    problems(editor('"s"', '{"q": {"type": "score", "instructions": "i", "criteria": []}}')),
  ).toEqual([['error', '\u201cq\u201d is a score with no levels']])
  expect(
    problems(editor('"s"', '{"q": {"type": "noul", "instructions": "i", "criteria": "x"}}')),
  ).toEqual([
    [
      'error',
      'The criteria of \u201cq\u201d must be an object of true and false descriptions, or null',
    ],
  ])
  expect(problems(editor('"s"', '{"q": 1}'))).toEqual([
    ['error', '\u201cq\u201d must be an object with a type (noul, choice or score)'],
  ])
  expect(
    problems(
      editor(
        '" "',
        '{"q": {"type": "noul"}, "q": {"type": "noul", "instructions": "i", "type": "noul"}}',
      ),
    ),
  ).toEqual([
    ['warning', 'State is empty, so the model has nothing to read'],
    ['warning', '\u201cq\u201d has no instructions'],
    ['warning', 'Key \u201cq\u201d is used twice; only the last one counts'],
    ['warning', '\u201cq\u201d sets \u201ctype\u201d twice; only the last one counts'],
  ])
  expect(
    problems(
      editor(
        '{"a": 1, "a": 2}',
        '{"q": {"type": "choice", "instructions": "i", "criteria": {"x": 1, "x": 2}}}',
      ),
    ),
  ).toEqual([
    ['warning', 'Field \u201ca\u201d appears twice; only the last one counts'],
    ['warning', '\u201cq\u201d lists the option \u201cx\u201d twice; only the last one counts'],
  ])
  expect(
    problems(
      editor('"s"', '{"q": {"type": "noul", "instructions": "i", "criteria": {"maybe": 1}}}'),
    ),
  ).toEqual([
    [
      'warning',
      '\u201cq\u201d describes \u201cmaybe\u201d, which a noul question ignores; only true and false are read',
    ],
  ])
})

test('the option limit counts distinct labels, as the server does', () => {
  const options = (n: number, extra = '') =>
    `{"q": {"type": "choice", "instructions": "i", "criteria": {${Array.from({ length: n }, (_, j) => `"o${j}": null`).join(', ')}${extra}}}}`
  expect(analyze(editor('"s"', options(255)), null).errors).toBe(0)
  expect(analyze(editor('"s"', options(255, ', "o0": null')), null).errors).toBe(0)
  expect(problems(editor('"s"', options(256)))).toEqual([
    ['error', '\u201cq\u201d has 256 options; System One takes at most 255'],
  ])
})

test('a parse error blocks with its position, and there is no body', () => {
  const analysis = analyze(editor('{"a": }', '{}'), null)
  expect(analysis.body).toBeNull()
  expect(analysis.problems[0]).toMatchObject({
    severity: 'error',
    where: 'state, line 1',
    loc: { text: 'state', at: [6, 6] },
  })
})

test('a server location maps to the JSON the user is editing', () => {
  const text = '{\n  "q": {"type": "choice", "criteria": {"a": 1}}\n}'
  const e = editor('"s"', text)
  const at = (path: string[]) => {
    const loc = locate(e, path)
    return loc && 'at' in loc ? text.slice(loc.at[0], loc.at[1]) : loc
  }
  expect(at(['questions', 'q', 'criteria'])).toBe('"criteria"')
  expect(at(['questions', 'q', 'criteria', 'a'])).toBe('"a"')
  expect(at(['questions', 'missing'])).toBe('')
  expect(locate(e, ['model'])).toEqual({ el: 'model' })
})

test('curl and Python send what the playground sends', () => {
  const body = buildBody(
    parseJson('{"id": 9007199254740993, "note": "it\'s"}'),
    parseJson(
      '{"q": {"type": "noul", "instructions": "i", "criteria": {"true": "yes", "x": 1}, "tag": 1}}',
    ),
    'm',
  )
  const curl = curlSnippet('http://h:8080', body)
  expect(curl.split('\n')[0]).toBe('curl -s http://h:8080/typesafe/v1/systemone \\')
  expect(curl).toContain('"note": "it\'\\\'\'s"')
  const py = pythonSnippet('http://h:8080', body)
  expect(py).toContain('from typesafe_sdk import Noul, TypeSafeClient')
  expect(py).toContain('base_url="http://h:8080/typesafe"')
  expect(py).toContain('"id": 9007199254740993')
  expect(py).toContain('# criteria "x" left out: noul reads only true and false')
  expect(py).toContain('# "tag" left out: System One does not read it')
  expect(py).toContain('model="m",')
})
