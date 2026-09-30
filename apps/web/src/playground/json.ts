// JSON with source positions.
//
// The playground edits JSON as text and needs more than JSON.parse gives it: the offset of a
// syntax error, the span of every key so a server error can be pointed at, duplicate keys kept
// visible instead of silently collapsed, and number spellings kept as typed. A request is
// serialized from this tree, never through JS numbers, so an integer beyond 2^53 reaches the
// server with the digits the user wrote.
//
// Ported from laya's playground (examples/server.py: parseJSON, fmt; Apache-2.0).

export interface ObjectNode {
  t: 'obj'
  entries: JsonEntry[]
  s: number
  e: number
}
export interface ArrayNode {
  t: 'arr'
  items: JsonNode[]
  s: number
  e: number
}
export interface StringNode {
  t: 'str'
  v: string
  s: number
  e: number
}
export interface NumberNode {
  t: 'num'
  v: number
  /** The spelling as written, which is what gets sent. */
  raw: string
  s: number
  e: number
}
export interface LiteralNode {
  t: 'lit'
  v: boolean | null
  s: number
  e: number
}
export type JsonNode = ObjectNode | ArrayNode | StringNode | NumberNode | LiteralNode

export interface JsonEntry {
  k: string
  /** Span of the key, quotes included. */
  ks: number
  ke: number
  v: JsonNode
}

export interface JsonSyntaxError {
  msg: string
  /** UTF-16 offset into the source. */
  at: number
}

class JsonParseError extends Error {
  readonly at: number
  constructor(msg: string, at: number) {
    super(msg)
    this.at = at
  }
}

const NUMBER = /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/y
const ESCAPES: Record<string, string> = {
  '"': '"',
  '\\': '\\',
  '/': '/',
  b: '\b',
  f: '\f',
  n: '\n',
  r: '\r',
  t: '\t',
}

/** Strict JSON (RFC 8259) with positions. Throws a JsonParseError carrying the offset. */
export function parseJson(src: string): JsonNode {
  let i = 0
  const n = src.length
  const fail = (msg: string, at: number = i): never => {
    throw new JsonParseError(msg, at)
  }
  const ws = () => {
    while (i < n && ' \n\t\r'.includes(src[i]!)) i++
  }

  function value(): JsonNode {
    ws()
    if (i >= n)
      fail(n ? 'Unexpected end of JSON; expected a value' : 'Empty; expected a JSON value')
    const c = src[i]!
    if (c === '{') return object()
    if (c === '[') return array()
    if (c === '"') return string()
    if (c === '-' || (c >= '0' && c <= '9')) {
      NUMBER.lastIndex = i
      const m = NUMBER.exec(src)
      if (!m) fail('Invalid number')
      const s = i
      i += m![0].length
      return { t: 'num', v: Number(m![0]), raw: m![0], s, e: i }
    }
    for (const [word, v] of [
      ['true', true],
      ['false', false],
      ['null', null],
    ] as const) {
      if (src.startsWith(word, i)) {
        const s = i
        i += word.length
        return { t: 'lit', v, s, e: i }
      }
    }
    if (c === "'") fail('Strings need double quotes, not single quotes')
    return fail(`Unexpected ${JSON.stringify(c)}; expected a value`)
  }

  function object(): ObjectNode {
    const s = i++
    const entries: JsonEntry[] = []
    ws()
    if (src[i] === '}') {
      i++
      return { t: 'obj', entries, s, e: i }
    }
    for (;;) {
      ws()
      if (src[i] !== '"') {
        fail(
          src[i] === '}'
            ? "Trailing comma before '}'"
            : 'Expected a property name in double quotes',
        )
      }
      const key = string()
      ws()
      if (src[i] !== ':') fail("Expected ':' after the property name")
      i++
      entries.push({ k: key.v, ks: key.s, ke: key.e, v: value() })
      ws()
      if (src[i] === ',') {
        i++
        continue
      }
      if (src[i] === '}') {
        i++
        return { t: 'obj', entries, s, e: i }
      }
      fail(i >= n ? "Unexpected end of JSON; missing '}'" : "Expected ',' or '}' after the value")
    }
  }

  function array(): ArrayNode {
    const s = i++
    const items: JsonNode[] = []
    ws()
    if (src[i] === ']') {
      i++
      return { t: 'arr', items, s, e: i }
    }
    for (;;) {
      ws()
      if (src[i] === ']') fail("Trailing comma before ']'")
      items.push(value())
      ws()
      if (src[i] === ',') {
        i++
        continue
      }
      if (src[i] === ']') {
        i++
        return { t: 'arr', items, s, e: i }
      }
      fail(i >= n ? "Unexpected end of JSON; missing ']'" : "Expected ',' or ']' after the item")
    }
  }

  function string(): StringNode {
    const s = i++
    let out = ''
    let run = i
    for (;;) {
      if (i >= n) fail('Unterminated string', s)
      const c = src[i]!
      if (c === '"') {
        out += src.slice(run, i)
        i++
        return { t: 'str', v: out, s, e: i }
      }
      if (c === '\n') fail('Line break inside a string; close the quote or write \\n')
      if (c < ' ') fail('Control character inside a string; escape it')
      if (c === '\\') {
        out += src.slice(run, i)
        const d = src[i + 1]
        const hex = src.slice(i + 2, i + 6)
        if (d !== undefined && Object.prototype.hasOwnProperty.call(ESCAPES, d)) {
          out += ESCAPES[d]
          i += 2
        } else if (d === 'u' && /^[0-9a-fA-F]{4}$/.test(hex)) {
          out += String.fromCharCode(parseInt(hex, 16))
          i += 6
        } else {
          fail('Invalid escape sequence')
        }
        run = i
        continue
      }
      i++
    }
  }

  const ast = value()
  ws()
  if (i < n) fail('Unexpected text after the end of the JSON value')
  return ast
}

export type Parsed =
  | { ast: JsonNode; value: unknown; error?: undefined }
  | { ast?: undefined; value?: undefined; error: JsonSyntaxError }

export function tryParse(text: string): Parsed {
  try {
    const ast = parseJson(text)
    return { ast, value: toValue(ast) }
  } catch (error) {
    if (error instanceof JsonParseError) return { error: { msg: error.message, at: error.at } }
    throw error
  }
}

/**
 * The value Python's json.loads builds, which is what the server sees: a repeated key keeps the
 * position of its first occurrence and the value of its last. A "__proto__" key stays an ordinary
 * own property.
 */
export function toValue(node: JsonNode): unknown {
  if (node.t === 'obj') {
    const out: Record<string, unknown> = {}
    for (const entry of node.entries) {
      Object.defineProperty(out, entry.k, {
        value: toValue(entry.v),
        enumerable: true,
        writable: true,
        configurable: true,
      })
    }
    return out
  }
  if (node.t === 'arr') return node.items.map(toValue)
  return node.v
}

/** A synthetic tree for a JS value, used for presets and the form editors. */
export function fromValue(value: unknown): JsonNode {
  if (value === null || value === undefined) return { t: 'lit', v: null, s: 0, e: 0 }
  if (typeof value === 'boolean') return { t: 'lit', v: value, s: 0, e: 0 }
  if (typeof value === 'number')
    return { t: 'num', v: value, raw: JSON.stringify(value), s: 0, e: 0 }
  if (typeof value === 'string') return { t: 'str', v: value, s: 0, e: 0 }
  if (Array.isArray(value)) return { t: 'arr', items: value.map(fromValue), s: 0, e: 0 }
  return {
    t: 'obj',
    entries: Object.keys(value as object).map((k) => ({
      k,
      ks: 0,
      ke: 0,
      v: fromValue((value as Record<string, unknown>)[k]),
    })),
    s: 0,
    e: 0,
  }
}

export const stringNode = (v: string): StringNode => ({ t: 'str', v, s: 0, e: 0 })
export const nullNode = (): LiteralNode => ({ t: 'lit', v: null, s: 0, e: 0 })
export const objectNode = (entries: Array<[string, JsonNode]>): ObjectNode => ({
  t: 'obj',
  entries: entries.map(([k, v]) => ({ k, ks: 0, ke: 0, v })),
  s: 0,
  e: 0,
})

/** The last entry of `key`: the one Python keeps. */
export function lastEntry(node: JsonNode | null | undefined, key: string): JsonEntry | null {
  if (!node || node.t !== 'obj') return null
  for (let j = node.entries.length - 1; j >= 0; j--) {
    if (node.entries[j]!.k === key) return node.entries[j]!
  }
  return null
}

/** Pretty-printed with two-space indentation; duplicate keys and number spellings survive. */
export function formatJson(node: JsonNode, indent = 0): string {
  const pad = '  '.repeat(indent + 1)
  const end = '\n' + '  '.repeat(indent)
  const list = (open: string, close: string, parts: string[]) =>
    parts.length ? open + '\n' + parts.join(',\n') + end + close : open + close
  if (node.t === 'obj') {
    return list(
      '{',
      '}',
      node.entries.map((en) => pad + JSON.stringify(en.k) + ': ' + formatJson(en.v, indent + 1)),
    )
  }
  if (node.t === 'arr')
    return list(
      '[',
      ']',
      node.items.map((it) => pad + formatJson(it, indent + 1)),
    )
  return node.t === 'num' ? node.raw : JSON.stringify(node.v)
}

/** Compact text, as sent: no whitespace, duplicates and number spellings kept. */
export function compactJson(node: JsonNode): string {
  if (node.t === 'obj') {
    return (
      '{' + node.entries.map((en) => JSON.stringify(en.k) + ':' + compactJson(en.v)).join(',') + '}'
    )
  }
  if (node.t === 'arr') return '[' + node.items.map(compactJson).join(',') + ']'
  return node.t === 'num' ? node.raw : JSON.stringify(node.v)
}

/** Repeated keys resolved as Python resolves them: first position, last value. */
export function dedupe(node: JsonNode): JsonNode {
  if (node.t === 'arr') return { ...node, items: node.items.map(dedupe) }
  if (node.t !== 'obj') return node
  const order: string[] = []
  const last = new Map<string, JsonEntry>()
  for (const entry of node.entries) {
    if (!last.has(entry.k)) order.push(entry.k)
    last.set(entry.k, entry)
  }
  return {
    ...node,
    entries: order.map((k) => {
      const entry = last.get(k)!
      return { ...entry, v: dedupe(entry.v) }
    }),
  }
}

export function lineCol(text: string, at: number): { line: number; col: number } {
  const before = text.slice(0, at).split('\n')
  return { line: before.length, col: before[before.length - 1]!.length + 1 }
}

export function lineOf(text: string, at: number): number {
  let line = 0
  for (let j = text.indexOf('\n'); j !== -1 && j < at; j = text.indexOf('\n', j + 1)) line++
  return line
}

/** The offset where line `line` (0-based) starts. */
export function lineStart(text: string, line: number): number {
  let at = 0
  for (let k = 0; k < line; k++) {
    const next = text.indexOf('\n', at)
    if (next === -1) return text.length
    at = next + 1
  }
  return at
}

const KINDS: Record<string, string> = {
  obj: 'an object',
  arr: 'a list',
  str: 'text',
  num: 'a number',
}

export function kindOf(node: JsonNode): string {
  if (node.t === 'lit') return node.v === null ? 'null' : 'true/false'
  return KINDS[node.t]!
}

// --- highlighting ----------------------------------------------------------------------------

/** Strings (possibly unterminated), brackets, punctuation, whitespace, and any other word. */
const TOKEN = /"(?:[^"\\]|\\.)*"?|[{}[\]]|[,:]|\s+|[^\s"{}[\],:]+/g

/**
 * Token classes: k key, s string, n number, l literal, b bracket, p punctuation, x a bare word
 * that is not JSON, '' whitespace.
 */
export type TokenClass = 'k' | 's' | 'n' | 'l' | 'b' | 'p' | 'x' | ''

export interface Token {
  text: string
  cls: TokenClass
  /** Offset within the line. */
  at: number
}

function tokenClass(token: string, rest: string): TokenClass {
  const c = token[0]!
  if (c === '"') return /^\s*:/.test(rest) ? 'k' : 's'
  if (c === '-' || (c >= '0' && c <= '9')) return 'n'
  if (token === 'true' || token === 'false' || token === 'null') return 'l'
  if ('{}[]'.includes(c)) return 'b'
  if (c === ',' || c === ':') return 'p'
  return /\s/.test(c) ? '' : 'x'
}

/** One line split into highlight tokens. Lines are tokenized alone, so a key is recognized by
 * the colon that follows it on the same line, as the editor writes it. */
export function tokenize(line: string): Token[] {
  const out: Token[] = []
  TOKEN.lastIndex = 0
  for (let m = TOKEN.exec(line); m; m = TOKEN.exec(line)) {
    out.push({ text: m[0], cls: tokenClass(m[0], line.slice(TOKEN.lastIndex)), at: m.index })
  }
  return out
}
