import { expect, test } from 'bun:test'

import {
  compactJson,
  dedupe,
  formatJson,
  lastEntry,
  lineCol,
  lineOf,
  lineStart,
  parseJson,
  tokenize,
  toValue,
  tryParse,
} from './json'

test('numbers keep the spelling they were written with', () => {
  const text = '{"id": 12345678901234567890, "p": 1.50, "e": -0, "x": 1E+2}'
  const ast = parseJson(text)
  expect(compactJson(ast)).toBe('{"id":12345678901234567890,"p":1.50,"e":-0,"x":1E+2}')
  expect(formatJson(ast)).toBe(
    '{\n  "id": 12345678901234567890,\n  "p": 1.50,\n  "e": -0,\n  "x": 1E+2\n}',
  )
})

test('repeated keys stay in the tree and resolve as Python resolves them', () => {
  const ast = parseJson('{"a": 1, "b": 2, "a": 3}')
  expect(compactJson(ast)).toBe('{"a":1,"b":2,"a":3}')
  expect(compactJson(dedupe(ast))).toBe('{"a":3,"b":2}')
  expect(JSON.stringify(toValue(ast))).toBe('{"a":3,"b":2}')
  expect(lastEntry(ast, 'a')?.v).toMatchObject({ t: 'num', raw: '3' })
})

test('a __proto__ key is an ordinary property', () => {
  const value = toValue(parseJson('{"__proto__": {"polluted": true}}')) as Record<string, unknown>
  expect(Object.keys(value)).toEqual(['__proto__'])
  expect(({} as Record<string, unknown>).polluted).toBeUndefined()
})

test('key spans cover the quoted key', () => {
  const text = '{\n  "state": "x"\n}'
  const ast = parseJson(text)
  const entry = lastEntry(ast, 'state')!
  expect(text.slice(entry.ks, entry.ke)).toBe('"state"')
  expect(text.slice(entry.v.s, entry.v.e)).toBe('"x"')
})

test('escapes decode, and strings re-encode canonically', () => {
  const ast = parseJson('"caf\\u00e9 \\"q\\" \\/ \\n"')
  expect(ast).toMatchObject({ t: 'str', v: 'café "q" / \n' })
  expect(compactJson(ast)).toBe('"café \\"q\\" / \\n"')
})

test('syntax errors say what is wrong and where', () => {
  const cases: Array<[string, string, number]> = [
    ['', 'Empty; expected a JSON value', 0],
    ['{"a": 1,}', "Trailing comma before '}'", 8],
    ['[1, 2,]', "Trailing comma before ']'", 6],
    ["{'a': 1}", 'Expected a property name in double quotes', 1],
    ['{"a" 1}', "Expected ':' after the property name", 5],
    ['{"a": 1', "Unexpected end of JSON; missing '}'", 7],
    ['"abc', 'Unterminated string', 0],
    ['"a\nb"', 'Line break inside a string; close the quote or write \\n', 2],
    ['{"a": tru}', 'Unexpected "t"; expected a value', 6],
    ['1 2', 'Unexpected text after the end of the JSON value', 2],
    ['"\\x"', 'Invalid escape sequence', 1],
    ['01', 'Unexpected text after the end of the JSON value', 1],
  ]
  for (const [text, msg, at] of cases) {
    expect(tryParse(text).error).toEqual({ msg, at })
  }
})

test('line helpers agree with each other', () => {
  const text = 'ab\ncd\n\nef'
  expect(lineCol(text, 4)).toEqual({ line: 2, col: 2 })
  expect(lineOf(text, 4)).toBe(1)
  expect(lineStart(text, 3)).toBe(7)
  expect(lineStart(text, 9)).toBe(text.length)
})

test('tokenize tells keys from string values by the colon', () => {
  const classes = tokenize('  "k": "v", 12, true, oops').map((t) => [t.text, t.cls])
  expect(classes).toEqual([
    ['  ', ''],
    ['"k"', 'k'],
    [':', 'p'],
    [' ', ''],
    ['"v"', 's'],
    [',', 'p'],
    [' ', ''],
    ['12', 'n'],
    [',', 'p'],
    [' ', ''],
    ['true', 'l'],
    [',', 'p'],
    [' ', ''],
    ['oops', 'x'],
  ])
  expect(tokenize('"open').map((t) => t.cls)).toEqual(['s'])
})
